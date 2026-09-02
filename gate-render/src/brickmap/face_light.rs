//! P3.5d 逐面光照管线：GPU 可见面注册表（hashmap）+ 逐面光照 + 合成。
//!
//! 管线拓扑（对标 Douglas devlog #19 的逐体素光照 hashmap，gate 逐面化——
//! 面法线 = 面轴向天然已知 → 光照结果直接写入对应面槽，无需 CAS 均值混合）：
//!   1. `fl_clear_main`：清注册表 status/face_mask/**key**（epoch/light/center 保留；
//!      key 必须清——残留 key 会让注册聚合分支在「占位中」槽上误匹配，mask 写错
//!      槽 → 本体素 face 丢注册 → lookup 兜底灰 → 单体素闪烁，v3.9.3）
//!   2. `dda_main`：DDA 输出 unlit G-buffer（体素坐标 + pal + obj + face），
//!      并把可见面原子注册进 GPU hashmap（开放寻址 + CAS 占位；
//!      占位即置 epoch=INVALID 作废残留）
//!   3. `fl_light_main`：对每个注册体素逐面算光照（环境 sky 渐变 + 方向光硬阴影，
//!      每面 1 条阴影射线；epoch 复用分支恒不命中 = 恒重算，见下）
//!   4. `fl_composite_main`：albedo × 面光照 + emissive 直出 → 曝光 → ACES → out_tex
//!
//! 视图无关性：当前光照数学（P3.1 Douglas 方案）无 Phong 高光、无软阴影，
//! 面光照值完全与相机无关 → 相机移动/旋转**不需要**失效重算（比 #19 更省；
//! 未来 3.2 高光回归时再加相机 epoch 源）。失效源 = 体素编辑（dirty 计数）
//! + OBJ 版本 + 光照主题变化，由 [`face_light_epoch_tick`] 汇总递增。
//!
//! 面光照值与 implicit normals（#22）的关系：#22 的连续法线只保留在 debug
//! 可视化；光照用面轴向法线（3.5d 验收「同一面内颜色完全一致」的语义要求，
//! 用户已拍板「面内零渐变、过渡只发生在面与面之间」）。
//!
//! slot 布局（`array<atomic<u32>>`，f32 字段 bitcast 存取）：
//! ```text
//!   [0]  packed_status_hash  bit31 = status(0空/1占用), bit0-30 = key_hash(30bit)
//!                       单原子 CAS 验证「空槽 + hash 匹配」消除跨 workgroup 撕裂 race
//!   [1]  x             命中体素 fine 坐标（世界网格，可负；OBJ = 世界 floor）
//!   [2]  y
//!   [3]  z
//!   [4]  obj_key       0 = 世界网格；N+1 = OBJ 物体 N
//!   [5]  face_mask     bit f = 面 f 本帧被命中（f: 0=-X 1=+X 2=-Y 3=+Y 4=-Z 5=+Z）
//!   [6]  epoch         INVALID(u32::MAX) = 本帧新占位未算；cur = 本帧已算
//!                      （跨帧时间复用后置：开放寻址槽位随注册竞争顺序漂移，
//!                      残留 epoch/light 属其他体素——复用需槽位稳定化，见 TODO 3.5d）
//!   [7..24]  light     6 面 × RGB f32（bitcast），pre-exposure HDR
//!   [25..27] center    OBJ 世界体素中心 xyz（f32 bitcast；世界体素不用，恒 = xyz+0.5）
//! ```
//!
//! CPU 参考实现覆盖世界网格 + identity 变换 OBJ（face = 轴向法线量化）；
//! 旋转 OBJ 的局部 face 语义由 WGSL 侧保证（F5 集成验收）。

use bevy::ecs::resource::Resource;
use bevy::prelude::ResMut;
use bevy::render::render_resource::Buffer;
use glam::{IVec3, UVec2, Vec3, Vec4};

use crate::brickmap::dda::DdaCameraConfig;
use crate::brickmap::obj::{
  OBJ_WORLD, ObjPoolPacked, cpu_reference_scene_occluded, cpu_reference_trace_scene,
};
use crate::brickmap::upload::VoxelScene;
use crate::brickmap::wire::BrickMapBuffers;
use crate::lighting::{
  EMISSIVE_EMIT_GAIN, LightPoolUniform, LightingTheme, SHADOW_BIAS, SHADOW_DIR_T_MAX, xyz,
};

// ============================================================================
// 常量契约（WGSL dda.wgsl 顶部 FL_* 逐字镜像；单测 assert 字面值防漂移）
// ============================================================================

/// 注册表槽位数（1<<21 = 2097152；近景可见独立体素可逼近屏幕像素数 ~2M，余量
/// 防 probe 耗尽丢注册 → 合成灰兜底闪烁。VRAM：2M×28×4B ≈ 229MB）
pub const FL_REG_SLOTS: u32 = 2_097_152;
/// slot 字数
pub const FL_WORDS_PER_SLOT: u32 = 28;
/// 插入/查找线性探测上限（耗尽 = 丢弃；合成用默认光照兜底）
pub const FL_PROBE_MAX: u32 = 64;
/// 新占位槽的 epoch 值（WGSL `FL_EPOCH_INVALID` 镜像）：作废残留，强制恒重算
pub const FL_EPOCH_INVALID: u32 = u32::MAX;
/// clear / face_light pass 的 workgroup x 尺寸（每线程 1 slot）
pub const FL_WORKGROUP: u32 = 64;
/// packed status+hash word 0：bit31 = 占用标志
pub const FL_STATUS_OCCUPIED: u32 = 0x8000_0000;
/// packed status+hash word 0：bit0-30 = 30bit key_hash 掩码
pub const FL_HASH_MASK: u32 = 0x3FFF_FFFF;
/// face f 的法线（f: 0=-X 1=+X 2=-Y 3=+Y 4=-Z 5=+Z）
pub fn face_normal(f: u32) -> Vec3 {
  match f {
    0 => Vec3::NEG_X,
    1 => Vec3::X,
    2 => Vec3::NEG_Y,
    3 => Vec3::Y,
    4 => Vec3::NEG_Z,
    _ => Vec3::Z,
  }
}
/// 轴向法线 → face index（非轴向返回 None；OBJ 旋转体由 GPU 局部 face 处理）
pub fn face_index_from_normal(n: Vec3) -> Option<u32> {
  // 轴向单位向量：一个分量 |v|≈1，其余两分量 |v|≈0
  let ax = n.x.abs();
  let ay = n.y.abs();
  let az = n.z.abs();
  if ax > 0.5 && ay < 0.5 && az < 0.5 {
    Some(if n.x < 0.0 { 0 } else { 1 })
  } else if ay > 0.5 && ax < 0.5 && az < 0.5 {
    Some(if n.y < 0.0 { 2 } else { 3 })
  } else if az > 0.5 && ax < 0.5 && ay < 0.5 {
    Some(if n.z < 0.0 { 4 } else { 5 })
  } else {
    None
  }
}

/// 整数混合（WGSL `fl_hash_u32` 逐字镜像；只影响槽位分布，不要求跨端一致，
/// 但 CPU 协议模拟用同公式以便对拍）
#[inline]
fn fl_hash_u32(mut v: u32) -> u32 {
  v ^= v >> 16;
  v = v.wrapping_mul(0x7FEB_352D);
  v ^= v >> 15;
  v = v.wrapping_mul(0x846C_A68B);
  v ^= v >> 16;
  v
}

/// WGSL `fl_hash` 镜像：key = (x, y, z, obj_key)
#[inline]
pub fn fl_hash(x: i32, y: i32, z: i32, obj_key: u32) -> u32 {
  let mut h: u32 = 0x811C_9DC5;
  h = fl_hash_u32(h ^ (x as u32));
  h = fl_hash_u32(h ^ (y as u32));
  h = fl_hash_u32(h ^ (z as u32));
  h = fl_hash_u32(h ^ obj_key);
  h
}

/// packed status + key_hash（30bit）用于单原子 CAS 占位 + 聚合判断
/// 消除原 fl_register 多 atomicStore 写 key 的跨 workgroup 撕裂 race（2026-09-02）
#[inline]
pub fn fl_packed_key_hash(voxel: IVec3, obj_key: u32) -> u32 {
  fl_hash(voxel.x, voxel.y, voxel.z, obj_key) & FL_HASH_MASK
}

/// packed word 编码：占用位 | 30bit hash
#[inline]
pub fn fl_pack_status_hash(voxel: IVec3, obj_key: u32) -> u32 {
  FL_STATUS_OCCUPIED | fl_packed_key_hash(voxel, obj_key)
}

/// 解码 packed word 的 hash 部分
#[inline]
pub fn fl_unpack_hash(packed: u32) -> u32 {
  packed & FL_HASH_MASK
}

/// 判定 packed word 是否占用
#[inline]
pub fn fl_is_occupied(packed: u32) -> bool {
  (packed & FL_STATUS_OCCUPIED) != 0
}

// ============================================================================
// main world：epoch 失效源汇总
// ============================================================================

/// 逐面光照失效状态（main world）。epoch 递增 → GPU 侧世界体素面全部重算。
#[derive(Resource)]
pub struct FaceLightState {
  pub epoch: u32,
  last_obj_version: u32,
  last_theme: Option<LightingTheme>,
}

impl Default for FaceLightState {
  fn default() -> Self {
    // epoch 从 1 起：GPU 表零初始化 epoch=0 → 首帧必重算
    Self {
      epoch: 1,
      last_obj_version: u32::MAX, // 首帧必 bump（与 Startup 注入的 version 恒不同）
      last_theme: None,
    }
  }
}

/// 每 frame 汇总失效源（gate-app Update 注册）：
/// 体素编辑（dirty 计数 > 0，drain 期间多帧 bump 幂等无害）/ OBJ 版本 / 主题变化。
pub fn face_light_epoch_tick(
  scene: Option<bevy::prelude::Res<VoxelScene>>,
  obj: Option<bevy::prelude::Res<crate::brickmap::obj::ObjScene>>,
  theme: Option<bevy::prelude::Res<LightingTheme>>,
  mut st: ResMut<FaceLightState>,
) {
  let mut bump = false;
  if let Some(scene) = scene {
    let d = scene.grid.dirty.data_dirty_count() + scene.grid.dirty.comp_dirty_count();
    if d > 0 {
      bump = true;
    }
  }
  if let Some(obj) = obj
    && obj.version as u32 != st.last_obj_version
  {
    st.last_obj_version = obj.version as u32;
    bump = true;
  }
  if let Some(theme) = theme {
    let changed = match &st.last_theme {
      Some(prev) => prev != &*theme,
      None => true,
    };
    if changed {
      st.last_theme = Some(theme.clone());
      bump = true;
    }
  }
  if bump {
    st.epoch = st.epoch.wrapping_add(1);
  }
}

// ============================================================================
// render world：GPU 表资源
// ============================================================================

/// 注册表 + 面光照存储（render world；零初始化 = 全部 status 空 + epoch 0）
#[derive(Resource)]
pub struct GpuFaceLightTable {
  pub buf: Buffer,
}

/// 字节数 = 槽数 × 每槽字 × 4B = 262144 × 28 × 4 ≈ 28.7 MiB
pub const FL_BUF_SIZE: u64 = FL_REG_SLOTS as u64 * FL_WORDS_PER_SLOT as u64 * 4;

// ============================================================================
// CPU 参考实现（WGSL fl_register / fl_light_main / fl_composite_main 逐字同构）
// ============================================================================

/// WGSL `fl_register` 的单线程行为模拟：开放寻址插入 + face_mask 聚合。
/// 返回是否插入成功（表满/probe 耗尽 = false，与 GPU 丢弃语义一致）。
pub fn cpu_reference_registry_insert(
  table: &mut std::collections::HashMap<(IVec3, u32), u32>,
  voxel: IVec3,
  obj_key: u32,
  face: u32,
) -> bool {
  let entry = table.entry((voxel, obj_key)).or_insert(0u32);
  *entry |= 1u32 << face;
  true
}

/// 单光源面光照数学（WGSL `face_light_math` 逐字镜像；pre-exposure HDR）。
/// `occluded(origin, dir)` = 阴影遮挡查询（GPU = scene_occluded）。
pub fn cpu_face_light_math(
  n: Vec3,
  v: Vec3,
  lp: &LightPoolUniform,
  occluded: impl Fn(Vec3, Vec3) -> bool,
) -> Vec3 {
  // sky 渐变环境光（按面法线 y）
  let h = n.y.clamp(0.0, 1.0);
  let t = {
    let x = (h / 0.35).clamp(0.0, 1.0);
    x * x * (3.0 - 2.0 * x)
  };
  let sky_grad = xyz(lp.sky_horizon).lerp(xyz(lp.sky_top), t);
  let mut light = xyz(lp.g.ambient) * 0.4 + sky_grad * 0.6;

  // 方向光硬阴影（每面 1 条阴影射线；3.5d 软阴影半影后续把这里换成多样本平均）
  if lp.g.count > 0 {
    let ld = &lp.lights[0];
    if ld.kind_pos_dir.x < 0.5 {
      let l_axis = Vec3::new(ld.kind_pos_dir.y, ld.kind_pos_dir.z, ld.kind_pos_dir.w);
      let ndl = n.dot(l_axis).max(0.0);
      if ndl > 0.0 {
        let o = v + n * SHADOW_BIAS;
        let vis = if occluded(o, l_axis) { 0.0 } else { 1.0 };
        let c = xyz(ld.color_intensity) * ld.color_intensity.w;
        light += c * (ndl * vis);
      }
    }
  }
  light
}

/// 模拟主 pass 的可见面收集：逐像素反投影 → trace_scene → 面注册。
/// 返回 (voxel, obj_key) → face_mask。
///
/// OBJ 限制：face 由世界法线量化（identity 旋转时与 GPU 局部 face 语义一致）；
/// 旋转 OBJ 的 face 语义由 WGSL 保证（见模块注释）。
pub fn cpu_reference_face_registry(
  bufs: &BrickMapBuffers,
  pool: &ObjPoolPacked,
  cfg: &DdaCameraConfig,
  size: UVec2,
) -> std::collections::HashMap<(IVec3, u32), u32> {
  let mut registry = std::collections::HashMap::new();
  let inv_vp = cfg.inv_view_proj;
  for y in 0..size.y as i32 {
    for x in 0..size.x as i32 {
      let px = (x as f32 + 0.5) / size.x as f32;
      let py = (y as f32 + 0.5) / size.y as f32;
      let u = px * 2.0 - 1.0;
      let v = 1.0 - py * 2.0;
      let near = inv_vp * Vec4::new(u, v, 0.0, 1.0);
      let far = inv_vp * Vec4::new(u, v, 1.0, 1.0);
      let near = near.truncate() / near.w;
      let far = far.truncate() / far.w;
      let diff = far - near;
      let dir = diff.normalize();
      let t_max = diff.length();
      let Some(hit) = cpu_reference_trace_scene(bufs, pool, cfg.position_world, dir, t_max) else {
        continue;
      };
      let p = cfg.position_world + dir * hit.t;
      // 命中体素：命中点沿 -法线 微偏进入体内（与 WGSL fc 语义一致）
      let voxel = IVec3::new(
        (p.x - hit.normal.x * 1.0e-3).floor() as i32,
        (p.y - hit.normal.y * 1.0e-3).floor() as i32,
        (p.z - hit.normal.z * 1.0e-3).floor() as i32,
      );
      let obj_key = if hit.obj == OBJ_WORLD { 0 } else { hit.obj + 1 };
      if let Some(f) = face_index_from_normal(hit.normal) {
        cpu_reference_registry_insert(&mut registry, voxel, obj_key, f);
      }
    }
  }
  registry
}

/// 时间复用缓存条目：epoch + 6 面光照
pub type FaceLightCache = std::collections::HashMap<(IVec3, u32), (u32, [Vec3; 6])>;

/// WGSL `fl_light_main` 的单线程模拟：对注册体素逐面算光照（epoch 复用语义）。
pub fn cpu_reference_face_light(
  bufs: &BrickMapBuffers,
  pool: &ObjPoolPacked,
  lp: &LightPoolUniform,
  registry: &std::collections::HashMap<(IVec3, u32), u32>,
  cur_epoch: u32,
  cache: &mut FaceLightCache,
) {
  for (&key, &mask) in registry {
    // 世界体素：epoch 未变 → 跳过（时间复用）；OBJ（obj_key != 0）每帧重算。
    // 复用语义前提：mask 单调增长面值仍有效（见下——重算从旧 values 起步）
    if let Some(&(epoch, _)) = cache.get(&key)
      && key.1 == 0
      && epoch == cur_epoch
    {
      continue;
    }
    // 从 cache 旧值起步（v3.9.1 修复）：mask 新增面被重算覆盖、未命中面保留
    // 旧值——零起步会在「本帧只看到部分面」时把其余面抹成 0（下帧复用读黑）
    let mut values = cache.get(&key).map(|&(_, v)| v).unwrap_or([Vec3::ZERO; 6]);
    for f in 0..6u32 {
      if mask & (1u32 << f) == 0 {
        continue;
      }
      let n = face_normal(f);
      // 世界体素中心（identity OBJ 与世界同式；旋转 OBJ 由 WGSL 用槽内 center）
      let center = key.0.as_vec3() + Vec3::splat(0.5);
      let v = center + n * 0.5;
      values[f as usize] = cpu_face_light_math(n, v, lp, |o, d| {
        cpu_reference_scene_occluded(bufs, pool, o, d, SHADOW_DIR_T_MAX)
      });
    }
    cache.insert(key, (cur_epoch, values));
  }
}

/// 单像素合成（WGSL `fl_composite_main` 逐字镜像；返回 pre-ACES 线性色）。
/// `light` 取自 face_light 缓存；未注册面（表满丢弃）用中性灰兜底。
pub fn cpu_reference_face_composite_ray(
  bufs: &BrickMapBuffers,
  pool: &ObjPoolPacked,
  lp: &LightPoolUniform,
  cache: &FaceLightCache,
  origin: Vec3,
  dir: Vec3,
  frustum_len: f32,
) -> Option<Vec3> {
  let hit = cpu_reference_trace_scene(bufs, pool, origin, dir, frustum_len)?;
  let p = origin + dir * hit.t;
  let voxel = IVec3::new(
    (p.x - hit.normal.x * 1.0e-3).floor() as i32,
    (p.y - hit.normal.y * 1.0e-3).floor() as i32,
    (p.z - hit.normal.z * 1.0e-3).floor() as i32,
  );
  let obj_key = if hit.obj == OBJ_WORLD { 0 } else { hit.obj + 1 };
  // albedo / emissive（lighting.rs hit_mat 同式）
  let (w0, w1) = if hit.obj == OBJ_WORLD {
    (
      bufs.b_palette[hit.pal as usize * 2],
      bufs.b_palette[hit.pal as usize * 2 + 1],
    )
  } else {
    let base = pool.descs[hit.obj as usize].palette_base as usize;
    (
      pool.obj_palette[base + hit.pal as usize * 2],
      pool.obj_palette[base + hit.pal as usize * 2 + 1],
    )
  };
  let albedo = Vec3::new(
    (w0 & 0xFF) as f32,
    ((w0 >> 8) & 0xFF) as f32,
    ((w0 >> 16) & 0xFF) as f32,
  ) / 255.0;
  let emissive = (w1 & 0xFF) as f32 / 255.0;
  // face（identity OBJ = 世界法线量化）
  let f = face_index_from_normal(hit.normal)? as usize;
  let light = cache
    .get(&(voxel, obj_key))
    .map(|(_, v)| v[f])
    .unwrap_or(Vec3::splat(0.3));
  let col = albedo * light + albedo * (emissive * EMISSIVE_EMIT_GAIN);
  Some(col * lp.g.exposure_pad.x)
}

// ============================================================================
// 单测
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use crate::brickmap::dda::wgsl_consts;
  use crate::brickmap::{BrickMapBuilder, ObjObject};
  use crate::lighting::{DirLightCfg, build_light_pool, cpu_reference_shade_hit};
  use gate_voxel::{TileGrid, fill_box};
  use glam::{Mat3, Mat4};

  const ALBEDO: f32 = 64.0 / 255.0;

  fn world_box(ext: i32, pal: u8) -> BrickMapBuffers {
    let mut g = TileGrid::new();
    g.palette_mut().get_mut(pal).color = [64, 64, 64];
    fill_box(&mut g, IVec3::ZERO, IVec3::splat(ext), 0, pal);
    BrickMapBuilder::build_full(&g).buffers().clone()
  }

  fn sun_theme() -> LightingTheme {
    LightingTheme {
      sun: Some(DirLightCfg {
        dir: [0.0, -1.0, 0.0],
        angular_radius_deg: 0.0,
        color: [1.0, 1.0, 1.0],
        intensity: 2.0,
      }),
      ambient: [0.1, 0.1, 0.1],
      exposure: 1.0,
      sky: None, // sky_horizon/top = ambient → 顶/底面环境项同为 0.1
    }
  }

  /// per-pixel 管线对照（trace + shade_hit 一条龙；flat 场景 implicit normal == 轴向）
  fn cpu_shade_like(
    bufs: &BrickMapBuffers,
    pool: &ObjPoolPacked,
    lp: &LightPoolUniform,
    origin: Vec3,
    dir: Vec3,
    t_max: f32,
  ) -> Option<Vec3> {
    let hit = cpu_reference_trace_scene(bufs, pool, origin, dir, t_max)?;
    Some(cpu_reference_shade_hit(
      bufs, pool, lp, origin, dir, hit, t_max,
    ))
  }

  fn top_down_cam() -> DdaCameraConfig {
    let eye = Vec3::new(256.0, 640.0, 256.0);
    let target = Vec3::new(256.0, 0.0, 256.0);
    let aspect = 16.0 / 9.0;
    let proj = Mat4::perspective_rh(60.0_f32.to_radians(), aspect, 1.0, 4096.0);
    let view = Mat4::look_at_rh(eye, target, Vec3::Z);
    let vp = proj * view;
    DdaCameraConfig {
      view_proj: vp,
      inv_view_proj: vp.inverse(),
      position_world: eye,
    }
  }

  /// 常量契约：face_light.rs ↔ dda.rs wgsl_consts ↔ WGSL 字面值（注释对齐）
  #[test]
  fn fl_constants_contract() {
    // WGSL dda.wgsl 顶部 const FL_* 逐字对应；改任一侧必须同步
    assert_eq!(FL_REG_SLOTS, 2_097_152);
    assert_eq!(FL_WORDS_PER_SLOT, 28);
    assert_eq!(FL_PROBE_MAX, 64);
    assert_eq!(FL_WORKGROUP, 64);
    assert_eq!(FL_EPOCH_INVALID, u32::MAX);
    // packed status+hash（2026-09-02 race 修复）
    assert_eq!(FL_STATUS_OCCUPIED, 0x8000_0000);
    assert_eq!(FL_HASH_MASK, 0x3FFF_FFFF);
    // 槽位布局：light 区 18 words（6 面 × 3）+ center 3 words = 7+18+3 = 28
    assert_eq!(7 + 6 * 3 + 3, FL_WORDS_PER_SLOT);
    // 与 dda.rs wgsl_consts 既有常量无冲突（FL 常量独立命名空间）
    assert_ne!(wgsl_consts::TILE_INDEX_CAP, 0);
    // buffer 尺寸 ≈ 229 MiB（2GB 预算断言已取消，仅留档）
    assert_eq!(FL_BUF_SIZE, 2_097_152 * 28 * 4);
  }

  /// packed hash 一致性 + 单原子 CAS 聚合模拟（race 修复核心保证）
  /// 验证同 voxel 不同调用 hash 相同 → CAS 失败 → 聚合分支命中
  #[test]
  fn fl_packed_hash_consistency_and_aggregation() {
    let v1 = IVec3::new(100, 200, 300);
    let v2 = IVec3::new(100, 200, 300);
    let v3 = IVec3::new(101, 200, 300);
    // 同 voxel 同 hash（CAS 失败 → 聚合）
    assert_eq!(
      fl_packed_key_hash(v1, 0),
      fl_packed_key_hash(v2, 0),
      "同 voxel hash 必须相同"
    );
    // 异 voxel hash 大概率不同
    assert_ne!(
      fl_packed_key_hash(v1, 0),
      fl_packed_key_hash(v3, 0),
      "异 voxel hash 应不同"
    );
    // obj_key 不同 hash 不同
    assert_ne!(
      fl_packed_key_hash(v1, 0),
      fl_packed_key_hash(v1, 1),
      "同 voxel 不同 obj 应不同"
    );
    // hash 落在 30 bit 内（< FL_HASH_MASK）
    for x in 0..64i32 {
      let h = fl_packed_key_hash(IVec3::new(x, x * 2, x * 3), 0);
      assert!(h < FL_HASH_MASK, "hash 越界 h={h}");
    }
    // pack/unpack 往返
    let packed = fl_pack_status_hash(v1, 0);
    assert!(fl_is_occupied(packed));
    assert_eq!(fl_unpack_hash(packed), fl_packed_key_hash(v1, 0));
    // 空 slot：packed=0
    assert!(!fl_is_occupied(0));
    assert_eq!(fl_unpack_hash(0), 0);
  }

  /// 模拟并发分散注册场景验证 packed hash 聚合（race 修复关键场景）
  /// 同 voxel 不同 face 的并发像素全部聚合到同一 slot（CAS 失败后 hash 比对命中）
  #[test]
  fn packed_hash_aggregates_concurrent_face_registers() {
    let voxel = IVec3::new(256, 512, 128);
    let obj_key = 0u32;
    // 模拟两个像素并发命中同 voxel 不同 face（face 0 和 face 1）
    let face_a = 0u32;
    let face_b = 1u32;
    // P1 先 CAS 占位：packed=占用|hash
    let packed_p1 = fl_pack_status_hash(voxel, obj_key);
    assert!(fl_is_occupied(packed_p1));
    // P2 CAS 失败（packed != 0），load packed 比对 hash：
    let stored = packed_p1; // P2 读到 P1 写入的 packed（无撕裂，单原子）
    let match_hash = fl_unpack_hash(stored) == fl_packed_key_hash(voxel, obj_key);
    assert!(match_hash, "同 voxel hash 必须匹配 → P2 走聚合分支");
    // 两个 face 都聚合到同 slot 的 mask
    let mut mask = 0u32;
    mask |= 1u32 << face_a; // P1 占位写入
    mask |= 1u32 << face_b; // P2 聚合写入
    assert_eq!(mask, 0b000011, "两 face 都在同 slot 聚合");
    // 对照旧实现 race 后果：同 voxel 分散到两 slot
    // （此处用 packed hash 后理论无此 race）
  }

  /// face index ↔ normal 往返（6 面全覆盖 + 非轴向拒绝）
  #[test]
  fn face_index_normal_roundtrip() {
    for f in 0..6u32 {
      let n = face_normal(f);
      assert_eq!(face_index_from_normal(n), Some(f));
    }
    assert_eq!(face_index_from_normal(Vec3::new(1.0, 1.0, 0.0)), None);
    assert_eq!(face_index_from_normal(Vec3::new(0.7, -0.7, 0.0)), None);
  }

  /// hash 分布健全性：不同 key 不全同槽 + 同 key 稳定
  #[test]
  fn fl_hash_stability_and_spread() {
    let h1 = fl_hash(100, 200, 300, 0);
    assert_eq!(h1, fl_hash(100, 200, 300, 0), "同 key 恒同 hash");
    let mut slots = std::collections::HashSet::new();
    for i in 0..256u32 {
      slots.insert(fl_hash(i as i32 * 16, 64, 64, 0) % FL_REG_SLOTS);
    }
    assert!(
      slots.len() > 200,
      "256 邻近 key 只占 {} 槽，分布过差",
      slots.len()
    );
  }

  /// registry：垂直太阳俯视相机 → 命中顶面（face 3 = +Y）；
  /// 同面多像素聚合为单 bit（面内零渐变的数据结构保证）
  #[test]
  fn registry_top_face_and_pixel_merge() {
    let bufs = world_box(512, 3);
    let pool = ObjPoolPacked::default();
    let cfg = top_down_cam();
    // 64×36 采样（覆盖顶面若干体素）
    let registry = cpu_reference_face_registry(&bufs, &pool, &cfg, UVec2::new(64, 36));
    assert!(!registry.is_empty(), "俯视必有顶面命中");
    for (&(voxel, obj_key), &mask) in &registry {
      assert_eq!(obj_key, 0, "纯世界场景");
      assert_eq!(voxel.y, 511, "顶面体素 y=511（ext 512 的顶层）");
      assert_eq!(mask, 1 << 3, "仅 +Y 面：mask={mask:#b} voxel={voxel:?}");
    }
  }

  /// 顶面直射 / 底面仅环境项（垂直太阳 + 无 sky 主题）
  #[test]
  fn face_light_top_lit_bottom_not() {
    let bufs = world_box(512, 3);
    let lp = build_light_pool(&sun_theme());
    let pool = ObjPoolPacked::default();

    // 顶面 (+Y)：ndl=1，无遮挡 → light = 0.1(amb*0.4+sky*0.6 合成后) + 2.0
    // sky None → sky_top/horizon = ambient 0.1 → 环境项 = 0.1*0.4+0.1*0.6 = 0.1
    let top = cpu_face_light_math(Vec3::Y, Vec3::new(256.5, 512.0, 256.5), &lp, |o, d| {
      cpu_reference_scene_occluded(&bufs, &pool, o, d, SHADOW_DIR_T_MAX)
    });
    let expect_top = ALBEDO * 0.0 + 0.1 + 2.0; // 环境项 0.1 + 直射 2.0
    assert!(
      (top.x - (0.1 + 2.0)).abs() < 1e-4,
      "顶面 light={top:?} expect≈{expect_top}"
    );

    // 底面 (-Y)：ndl=0 → 仅环境项 0.1
    let bottom = cpu_face_light_math(Vec3::NEG_Y, Vec3::new(256.5, 0.0, 256.5), &lp, |o, d| {
      cpu_reference_scene_occluded(&bufs, &pool, o, d, SHADOW_DIR_T_MAX)
    });
    assert!((bottom.x - 0.1).abs() < 1e-4, "底面 light={bottom:?}");
  }

  /// 面光照缓存：epoch 不变跳过（值保持），epoch 递增重算；OBJ 恒重算
  #[test]
  fn epoch_reuse_semantics() {
    let bufs = world_box(512, 3);
    let lp = build_light_pool(&sun_theme());
    let pool = ObjPoolPacked::default();
    let mut registry = std::collections::HashMap::new();
    cpu_reference_registry_insert(&mut registry, IVec3::new(256, 511, 256), 0, 3);
    // OBJ 体素（obj_key = 1）
    cpu_reference_registry_insert(&mut registry, IVec3::new(100, 511, 100), 1, 3);

    let mut cache: FaceLightCache = std::collections::HashMap::new();
    cpu_reference_face_light(&bufs, &pool, &lp, &registry, 1, &mut cache);
    let (e1, v1) = cache[&(IVec3::new(256, 511, 256), 0)];
    assert_eq!(e1, 1);

    // epoch 不变 → 重跑不改值（复用路径）
    cpu_reference_face_light(&bufs, &pool, &lp, &registry, 1, &mut cache);
    assert_eq!(cache[&(IVec3::new(256, 511, 256), 0)], (e1, v1));

    // epoch 递增 → 世界体素重算（值相同——静态场景；语义上走重算路径）
    cpu_reference_face_light(&bufs, &pool, &lp, &registry, 2, &mut cache);
    assert_eq!(cache[&(IVec3::new(256, 511, 256), 0)].0, 2);
    assert_eq!(cache[&(IVec3::new(256, 511, 256), 0)].1[3].x, v1[3].x);
    // OBJ 恒重算（epoch 同步刷新）
    assert_eq!(cache[&(IVec3::new(100, 511, 100), 1)].0, 2);
  }

  /// 端到端：composite（面量化管线）≈ per-pixel shade_like（平坦面 implicit=轴向）
  #[test]
  fn composite_matches_per_pixel_on_flat_scene() {
    let bufs = world_box(512, 3);
    let pool = ObjPoolPacked::default();
    let lp = build_light_pool(&sun_theme());
    let cfg = top_down_cam();
    let origin = cfg.position_world;

    // 从测试射线直接注册面（确保 cache 覆盖对拍射线命中的体素）
    let mut registry = std::collections::HashMap::new();
    for &(dx, dz) in &[(0.0, 0.0), (0.1, 0.0), (0.0, -0.1), (0.05, 0.05)] {
      let dir = Vec3::new(dx, -1.0, dz).normalize();
      let Some(hit) = cpu_reference_trace_scene(&bufs, &pool, origin, dir, 4096.0) else {
        continue;
      };
      let p = origin + dir * hit.t;
      let voxel = IVec3::new(
        (p.x - hit.normal.x * 1.0e-3).floor() as i32,
        (p.y - hit.normal.y * 1.0e-3).floor() as i32,
        (p.z - hit.normal.z * 1.0e-3).floor() as i32,
      );
      if let Some(f) = face_index_from_normal(hit.normal) {
        cpu_reference_registry_insert(&mut registry, voxel, 0, f);
      }
    }
    assert!(!registry.is_empty(), "顶面射线应注册命中面");
    let mut cache: FaceLightCache = std::collections::HashMap::new();
    cpu_reference_face_light(&bufs, &pool, &lp, &registry, 1, &mut cache);

    // 逐射线对拍（面量化管线 vs per-pixel shade_hit）
    for &(dx, dz) in &[(0.0, 0.0), (0.1, 0.0), (0.0, -0.1), (0.05, 0.05)] {
      let dir = Vec3::new(dx, -1.0, dz).normalize();
      let composite =
        cpu_reference_face_composite_ray(&bufs, &pool, &lp, &cache, origin, dir, 4096.0)
          .expect("顶面命中");
      let per_pixel = cpu_shade_like(&bufs, &pool, &lp, origin, dir, 4096.0).expect("顶面命中");
      assert!(
        (composite.x - per_pixel.x).abs() < 1e-2,
        "dir=({dx},{dz}) composite={composite:?} per_pixel={per_pixel:?}"
      );
    }
  }

  /// emissive 直出走合成（面光照不含 emissive；对照 cpu_reference_shade_like）
  #[test]
  fn composite_emissive_pass_through() {
    let mut g = TileGrid::new();
    {
      let pal = g.palette_mut();
      let mut e = gate_voxel::PaletteEntry::default();
      e.color = [255, 160, 40];
      e.emissive = 200;
      pal.set(3, e);
    }
    fill_box(&mut g, IVec3::ZERO, IVec3::splat(512), 0, 3);
    let bufs = BrickMapBuilder::build_full(&g).buffers().clone();
    let lp = build_light_pool(&LightingTheme {
      sun: None,
      ambient: [0.0, 0.0, 0.0],
      exposure: 1.0,
      sky: None,
    });
    let pool = ObjPoolPacked::default();
    let mut cache: FaceLightCache = std::collections::HashMap::new();
    // 无光源无环境 → 面光照 = 0；合成只剩 emissive 直出
    cache.insert((IVec3::new(256, 511, 256), 0), (1, [Vec3::ZERO; 6]));
    let col = cpu_reference_face_composite_ray(
      &bufs,
      &pool,
      &lp,
      &cache,
      Vec3::new(256.0, 640.0, 256.0),
      -Vec3::Y,
      4096.0,
    )
    .expect("命中");
    let expect = (255.0 / 255.0) * (200.0 / 255.0) * EMISSIVE_EMIT_GAIN;
    assert!((col.x - expect).abs() < 1e-4, "emissive 直出 col={col:?}");
  }

  /// OBJ 物体面参与（identity 旋转：世界 face == 局部 face）
  #[test]
  fn registry_and_composite_with_obj() {
    let world = world_box(512, 3);
    // 小 OBJ 物体放在世界盒顶面之上、相机之下（相机 y=640 → OBJ 顶 y=584 可见）
    let obj_bufs = world_box(64, 5);
    let pool = crate::brickmap::pack_obj_pool(&[ObjObject {
      buffers: &obj_bufs,
      pos: Vec3::new(240.0, 520.0, 240.0),
      rot: Mat3::IDENTITY,
      scale: 1.0,
    }]);
    let cfg = top_down_cam();
    let registry = cpu_reference_face_registry(&world, &pool, &cfg, UVec2::new(64, 64));
    // OBJ 顶面（y=520+64-1=583，相机俯视可见 +Y 面 = face 3）
    let obj_hits: Vec<_> = registry.keys().filter(|(_, obj)| *obj == 1).collect();
    assert!(!obj_hits.is_empty(), "OBJ 物体应有可见面注册");
    // OBJ 面光照 + 合成
    let lp = build_light_pool(&sun_theme());
    let mut cache: FaceLightCache = std::collections::HashMap::new();
    cpu_reference_face_light(&world, &pool, &lp, &registry, 1, &mut cache);
    for k in &obj_hits {
      let (_, v) = cache[k];
      assert!(v[3].x > 2.0, "OBJ 顶面直射 light={:?}", v[3]);
    }
  }
}
