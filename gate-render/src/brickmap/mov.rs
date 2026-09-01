//! P2.10 MOV-1：动态体素对象（多网格渲染 + `trace_scene()` 抽象）
//!
//! 物体 = 独立小 brickmap（`TileGrid`/`BrickMapBuilder` 全链复用，v1 限 1 tile）+
//! object-pool storage buffer + descriptor 表（位置 / mat3 旋转 / 缩放 / 网格基址 /
//! AABB，128B/物体）。渲染 = 世界 DDA → 逐物体射线-OBB 剔除 → 局部两级 DDA →
//! 取最近命中，统一 `trace_scene()` 入口（P3 阴影射线 / P9 GI 天生感知物体网格）。
//!
//! pool 布局（CPU 打包，shader 零重映射）：
//! - `mov_struct`：逐对象 `[bitmap 1024w | dirs 32768w | node stream n w]` 顺序拼接；
//!   dirs/node 内部保留 builder 绝对字偏移（≥ NODE_STREAM_BASE），shader 端
//!   `node_base + (abs - NODE_STREAM_BASE)` 校正；bitmap/dir/node 基址进 descriptor。
//! - `mov_leaves`：brick slab 顺序拼接，descriptor 存 slab 基址（node 内 1-based
//!   slab 号直接加基址）。
//! - `mov_palette`：逐对象 256 条 ×2w 拼接，descriptor 存字基址。
//!
//! CPU 参考实现 [`cpu_reference_trace_scene`] 与 WGSL `trace_scene` 同构
//! （世界路径直接复用 `cpu_reference_dda_ray_two_level`，等价性由单测锁定）。

use bevy::{
  prelude::*,
  render::{
    Extract, Render, RenderApp, RenderStartup, RenderSystems,
    render_resource::{Buffer, BufferDescriptor, BufferUsages, ShaderType, UniformBuffer},
    renderer::{RenderDevice, RenderQueue},
  },
};
use std::sync::Arc;

use super::wire::{
  BITMAP_BASE, BRICK_SLAB_WORDS, BrickMapBuffers, CELL_DIR_WORDS, DIR_BASE, HDR_HAS_BRICK,
  HDR_HAS_L1, HDR_HAS_L2, HDR_HAS_L3, HDR_UNIFORM_MASK, NODE_STREAM_BASE, PALETTE_WORDS,
  SLOT_TAG_EMPTY, SLOT_TAG_LEAF, TILE_BITMAP_WORDS, slot_palette, slot_tag, unpack_slot_word,
};

/// descriptor 着色器镜像（storage，128B/物体）
///
/// 变换约定：`world = pos + R · (local · scale)`，R 列向量存 rot0/1/2（glam Mat3
/// 的 x_axis/y_axis/z_axis 即列）；逆变换 `local = ((w-pos)·col_i) / scale`——
/// `rd_local` 不归一化，t 标尺与世界一致（DDA 局部/全局 t 无需换算）。
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, ShaderType)]
pub struct MovDesc {
  /// xyz = 物体 tile 原点的世界坐标（fine 单位），w = scale
  pub pos_scale: Vec4,
  pub rot0: Vec4,
  pub rot1: Vec4,
  pub rot2: Vec4,
  /// 世界 AABB（局部 tile [0,512]³ 经变换的外包盒，CPU 预计算）
  pub aabb_min: Vec4,
  pub aabb_max: Vec4,
  /// mov_struct 内 bitmap 区字基址
  pub bitmap_base: u32,
  /// mov_struct 内 dirs 区字基址
  pub dir_base: u32,
  /// mov_struct 内 node stream 字基址（对应物体自身 NODE_STREAM_BASE）
  pub node_base: u32,
  /// mov_leaves 内 slab 基址（物体 node 内 slab 号 + 基址 = pool slab 号）
  pub leaves_base: u32,
  /// mov_palette 内字基址
  pub palette_base: u32,
  pub _pad0: u32,
  pub _pad1: u32,
  pub _pad2: u32,
}

/// BG2 uniform：物体数（WGSL `MovGlobals` 镜像，16B）
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, ShaderType)]
pub struct MovGlobals {
  pub count: u32,
  pub _pad0: u32,
  pub _pad1: u32,
  pub _pad2: u32,
}

/// 打包后的 CPU 侧 pool（与 GPU buffer 字节一一对应）
#[derive(Debug, Clone, Default, PartialEq)]
pub struct MovPoolPacked {
  pub mov_struct: Vec<u32>,
  pub mov_leaves: Vec<u32>,
  pub mov_palette: Vec<u32>,
  pub descs: Vec<MovDesc>,
}

/// 单个物体的打包输入（buffers = 该物体独立 TileGrid 的 build_full 产物）
#[derive(Debug, Clone, Copy)]
pub struct MovObject<'a> {
  pub buffers: &'a BrickMapBuffers,
  /// 物体 tile 原点（局部 [0,512]³ fine 的 [0,0,0] 角）的世界坐标
  pub pos: Vec3,
  /// 旋转（列向量即 glam Mat3 三轴）
  pub rot: Mat3,
  pub scale: f32,
}

/// 打包 N 个物体 → pool（v1 契约：每物体恰 1 tile，芯片级预制件）
///
/// # Panics
/// 物体 `tile_count != 1`（0 = 空网格，>1 = 越出 v1 单 tile 限制）时 panic。
pub fn pack_mov_pool(objs: &[MovObject]) -> MovPoolPacked {
  let mut out = MovPoolPacked::default();
  for obj in objs {
    let g = &obj.buffers.globals;
    assert_eq!(
      g.tile_count, 1,
      "MOV v1 契约：每物体恰 1 tile（got {}）",
      g.tile_count
    );
    let struct_len = obj.buffers.b_struct.len();
    let bitmap = BITMAP_BASE..(BITMAP_BASE + TILE_BITMAP_WORDS).min(struct_len);
    let dirs = DIR_BASE..(DIR_BASE + CELL_DIR_WORDS).min(struct_len);
    let node_end = (NODE_STREAM_BASE + g.node_words as usize).min(struct_len);
    let nodes = NODE_STREAM_BASE..node_end;
    let leaves_end = (g.brick_slabs as usize * BRICK_SLAB_WORDS).min(obj.buffers.b_leaves.len());
    let palette_end = PALETTE_WORDS.min(obj.buffers.b_palette.len());

    let mut dsc = MovDesc {
      pos_scale: obj.pos.extend(obj.scale),
      aabb_min: Vec4::ZERO,
      aabb_max: Vec4::ZERO,
      bitmap_base: out.mov_struct.len() as u32,
      dir_base: 0,
      node_base: 0,
      leaves_base: (out.mov_leaves.len() / BRICK_SLAB_WORDS) as u32,
      palette_base: out.mov_palette.len() as u32,
      rot0: Vec4::ZERO,
      rot1: Vec4::ZERO,
      rot2: Vec4::ZERO,
      _pad0: 0,
      _pad1: 0,
      _pad2: 0,
    };
    out
      .mov_struct
      .extend_from_slice(&obj.buffers.b_struct[bitmap]);
    dsc.dir_base = out.mov_struct.len() as u32;
    out
      .mov_struct
      .extend_from_slice(&obj.buffers.b_struct[dirs]);
    dsc.node_base = out.mov_struct.len() as u32;
    out
      .mov_struct
      .extend_from_slice(&obj.buffers.b_struct[nodes]);
    out
      .mov_leaves
      .extend_from_slice(&obj.buffers.b_leaves[..leaves_end]);
    out
      .mov_palette
      .extend_from_slice(&obj.buffers.b_palette[..palette_end]);

    // 旋转列（glam Mat3 三轴 = 列）+ 世界 AABB（局部 tile 盒 8 角变换外包）
    dsc.rot0 = obj.rot.x_axis.extend(0.0);
    dsc.rot1 = obj.rot.y_axis.extend(0.0);
    dsc.rot2 = obj.rot.z_axis.extend(0.0);
    let (mn, mx) = world_aabb(obj.pos, obj.rot, obj.scale);
    dsc.aabb_min = mn.extend(0.0);
    dsc.aabb_max = mx.extend(0.0);
    out.descs.push(dsc);
  }
  out
}

/// 局部 tile 盒 [0,512]³·scale 经旋转平移后的世界 AABB（剔除用）
pub fn world_aabb(pos: Vec3, rot: Mat3, scale: f32) -> (Vec3, Vec3) {
  let mut mn = Vec3::splat(f32::MAX);
  let mut mx = Vec3::splat(f32::MIN);
  for &x in &[0.0_f32, 512.0] {
    for &y in &[0.0, 512.0] {
      for &z in &[0.0, 512.0] {
        let local = Vec3::new(x, y, z) * scale;
        let w = pos + rot * local;
        mn = mn.min(w);
        mx = mx.max(w);
      }
    }
  }
  (mn, mx)
}

// ============================================================================
// 渲染侧资源：MovScene（main world）→ RenderMov（render world）→ GpuMovPool
// ============================================================================

/// main world 资源：app 构建后一次性插入（v1 静态；P4.7 平滑移动改每帧版本号）
#[derive(Resource, Clone)]
pub struct MovScene {
  pub packed: Arc<MovPoolPacked>,
  pub version: u64,
}

/// render world 提取产物
#[derive(Resource, Clone)]
pub(crate) struct RenderMov {
  packed: Arc<MovPoolPacked>,
  version: u64,
}

/// BG2 绑定的 GPU pool（render world）
#[derive(Resource)]
pub struct GpuMovPool {
  pub struct_buf: Buffer,
  pub leaves: Buffer,
  pub palette: Buffer,
  pub descs: Buffer,
  pub globals: UniformBuffer<MovGlobals>,
  pub version: u64,
}

pub struct MovPlugin;

impl Plugin for MovPlugin {
  fn build(&self, app: &mut App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };
    render_app
      .add_systems(bevy::render::ExtractSchedule, extract_mov_scene)
      .add_systems(RenderStartup, init_empty_mov_pool)
      .add_systems(
        Render,
        prepare_mov_pool.in_set(RenderSystems::PrepareBindGroups),
      );
  }
}

fn extract_mov_scene(mut commands: Commands, scene: Option<Extract<Res<MovScene>>>) {
  let Some(s) = scene else { return };
  commands.insert_resource(RenderMov {
    packed: s.packed.clone(),
    version: s.version,
  });
}

fn init_empty_mov_pool(device: Res<RenderDevice>, mut commands: Commands) {
  let make = |label: &str| -> Buffer {
    device.create_buffer(&BufferDescriptor {
      label: Some(label),
      size: 4, // 0 尺寸 buffer 非法；空 pool 由 count=0 表达
      usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
      mapped_at_creation: false,
    })
  };
  let globals = UniformBuffer::<MovGlobals>::default();
  commands.insert_resource(GpuMovPool {
    struct_buf: make("gate_mov_struct"),
    leaves: make("gate_mov_leaves"),
    palette: make("gate_mov_palette"),
    descs: make("gate_mov_descs"),
    globals,
    version: 0,
  });
}

fn u32_bytes(slice: &[u32]) -> &[u8] {
  unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 4) }
}

fn desc_bytes(slice: &[MovDesc]) -> &[u8] {
  const _: () = assert!(std::mem::size_of::<MovDesc>() == 128);
  unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const u8, slice.len() * 128) }
}

/// 版本变化时重建 pool buffers（v1 静态：仅 version 0→1 一次）
pub(crate) fn prepare_mov_pool(
  mut commands: Commands,
  scene: Option<Res<RenderMov>>,
  existing: Option<Res<GpuMovPool>>,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
) {
  let Some(s) = scene else { return };
  if existing.as_ref().is_some_and(|g| g.version == s.version) {
    return;
  }
  let p = &s.packed;
  let make = |label: &str, bytes: &[u8]| -> Buffer {
    let b = device.create_buffer(&BufferDescriptor {
      label: Some(label),
      size: (bytes.len() as u64).max(4),
      usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
      mapped_at_creation: false,
    });
    if !bytes.is_empty() {
      queue.write_buffer(&b, 0, bytes);
    }
    b
  };
  let struct_buf = make("gate_mov_struct", u32_bytes(&p.mov_struct));
  let leaves = make("gate_mov_leaves", u32_bytes(&p.mov_leaves));
  let palette = make("gate_mov_palette", u32_bytes(&p.mov_palette));
  let descs = make("gate_mov_descs", desc_bytes(&p.descs));
  let mut globals = UniformBuffer::from(MovGlobals {
    count: p.descs.len() as u32,
    ..Default::default()
  });
  globals.write_buffer(&device, &queue);
  commands.insert_resource(GpuMovPool {
    struct_buf,
    leaves,
    palette,
    descs,
    globals,
    version: s.version,
  });
}

// ============================================================================
// CPU 参考实现（与 WGSL trace_scene/trace_object 逐字同构）
// ============================================================================

/// 命中记录：obj = OBJ_WORLD 表示世界网格命中
/// normal = 命中面法线（世界空间单位向量，指向射线来向；起点在体内时 = -dir）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MovHit {
  pub t: f32,
  pub pal: u8,
  pub obj: u32,
  pub normal: Vec3,
}

pub const OBJ_WORLD: u32 = u32::MAX;

/// slab 法射线-AABB 求交（与 WGSL `slab_box` / `cpu_reference_dda_ray_aabb_skip` 同型）
/// 返回 (t_enter, t_exit)，t_exit < t_enter 表示不相交。
fn slab_box(ro: Vec3, rd: Vec3, mn: Vec3, mx: Vec3, t0: f32, t1: f32) -> (f32, f32) {
  let o = [ro.x, ro.y, ro.z];
  let d = [rd.x, rd.y, rd.z];
  let lo = [mn.x, mn.y, mn.z];
  let hi = [mx.x, mx.y, mx.z];
  let mut t_enter = t0;
  let mut t_exit = t1;
  for i in 0..3 {
    if d[i].abs() < 1e-30 {
      if o[i] < lo[i] || o[i] > hi[i] {
        return (1.0, 0.0); // 平行且在外 → miss 哨兵
      }
    } else {
      let ta = (lo[i] - o[i]) / d[i];
      let tb = (hi[i] - o[i]) / d[i];
      t_enter = t_enter.max(ta.min(tb));
      t_exit = t_exit.min(tb.max(ta));
    }
  }
  (t_enter, t_exit)
}

/// descriptor 的（列向量，scale）分解
fn desc_transform(dsc: &MovDesc) -> (Vec3, Mat3, f32) {
  (
    dsc.pos_scale.truncate(),
    Mat3::from_cols(
      dsc.rot0.truncate(),
      dsc.rot1.truncate(),
      dsc.rot2.truncate(),
    ),
    dsc.pos_scale.w,
  )
}

/// 物体 cell（0..31³）占用查询：bitmap 1 load
fn obj_cell_occupied(pool: &MovPoolPacked, dsc: &MovDesc, cc: [u32; 3]) -> bool {
  let ci = (cc[2] * 1024 + cc[1] * 32 + cc[0]) as usize;
  let w = pool.mov_struct[dsc.bitmap_base as usize + ci / 32];
  (w >> (ci % 32)) & 1 != 0
}

/// 物体局部最细格采样（局部 fine 0..511³，逐字镜像 view.rs get_voxel ④ 链，
/// 基址换成 descriptor：node_base + (abs - NODE_STREAM_BASE)、slab + leaves_base）
fn obj_sample_voxel(pool: &MovPoolPacked, dsc: &MovDesc, fine: IVec3) -> u8 {
  let m = fine.clamp(IVec3::ZERO, IVec3::splat(511));
  let it = m.as_uvec3();
  let ci = ((it.z >> 4) * 1024 + (it.y >> 4) * 32 + (it.x >> 4)) as usize;
  let bmp = pool.mov_struct[dsc.bitmap_base as usize + ci / 32];
  if (bmp >> (ci % 32)) & 1 == 0 {
    return 0;
  }
  let abs = pool.mov_struct[dsc.dir_base as usize + ci];
  if abs == 0 {
    return 0;
  }
  let mut p = (dsc.node_base + (abs - NODE_STREAM_BASE as u32)) as usize;
  let hdr = pool.mov_struct[p];
  if hdr & HDR_UNIFORM_MASK != 0 {
    return (hdr & HDR_UNIFORM_MASK) as u8;
  }
  p += 1;
  let sub = (m & 15).as_uvec3();
  let slot_at = |s: glam::UVec3, axis: u32| (s.x + s.y * axis + s.z * axis * axis) as usize;
  // L1（非 uniform 必有 l1）
  if hdr & HDR_HAS_L1 == 0 {
    return 0;
  }
  let slot = unpack_slot_word(
    pool.mov_struct[p + (slot_at(sub >> 3, 2) >> 1)],
    slot_at(sub >> 3, 2) & 1,
  );
  match slot_tag(slot) {
    SLOT_TAG_EMPTY => return 0,
    SLOT_TAG_LEAF => return slot_palette(slot),
    _ => {}
  }
  p += 4;
  if hdr & HDR_HAS_L2 == 0 {
    return 0;
  }
  let slot = unpack_slot_word(
    pool.mov_struct[p + (slot_at(sub >> 2, 4) >> 1)],
    slot_at(sub >> 2, 4) & 1,
  );
  match slot_tag(slot) {
    SLOT_TAG_EMPTY => return 0,
    SLOT_TAG_LEAF => return slot_palette(slot),
    _ => {}
  }
  p += 32;
  if hdr & HDR_HAS_L3 == 0 {
    return 0;
  }
  let slot = unpack_slot_word(
    pool.mov_struct[p + (slot_at(sub >> 1, 8) >> 1)],
    slot_at(sub >> 1, 8) & 1,
  );
  match slot_tag(slot) {
    SLOT_TAG_EMPTY => return 0,
    SLOT_TAG_LEAF => return slot_palette(slot),
    _ => {}
  }
  p += 256;
  if hdr & HDR_HAS_BRICK == 0 {
    return 0;
  }
  let slab_m1 = pool.mov_struct[p];
  if slab_m1 == 0 {
    return 0;
  }
  let slab = (slab_m1 - 1 + dsc.leaves_base) as usize;
  let idx = (sub.x + sub.y * 16 + sub.z * 256) as usize;
  let word = pool.mov_leaves[slab * BRICK_SLAB_WORDS + (idx >> 2)];
  let pal = (word >> ((idx & 3) * 8)) as u8;
  if pal != 0 { pal } else { 0 }
}

/// 物体局部细扫：单粗 cell（16³ fine）内有界 fine DDA（镜像 dda_fine_scan_cell，t 全局标尺）
/// 返回 (t_rel, pal, axis)；axis = 局部命中轴（0/1/2），3 = 起点即在体内。
#[allow(clippy::too_many_arguments)]
fn obj_fine_scan_cell(
  pool: &MovPoolPacked,
  dsc: &MovDesc,
  ro: Vec3,
  rd: Vec3,
  sign: [i32; 3],
  delta: [f32; 3],
  cc: [u32; 3],
  t_lo: f32,
  t_hi: f32,
) -> Option<(f32, u8, u8)> {
  if t_hi <= t_lo {
    return None;
  }
  let p = ro + rd * t_lo;
  let pc = [p.x, p.y, p.z];
  let base = [
    (cc[0] << 4) as i32,
    (cc[1] << 4) as i32,
    (cc[2] << 4) as i32,
  ];
  let mut fc = [0i32; 3];
  for i in 0..3 {
    fc[i] = (pc[i].floor() as i32).clamp(base[i], base[i] + 15);
  }
  let mut tmax_f = [f32::INFINITY; 3];
  for i in 0..3 {
    if [rd.x, rd.y, rd.z][i].abs() > 1e-30 {
      let bnd = (fc[i] + if sign[i] >= 0 { 1 } else { 0 }) as f32;
      let t = (bnd - pc[i]) / [rd.x, rd.y, rd.z][i];
      tmax_f[i] = if t < 0.0 { 0.0 } else { t };
    }
  }
  let span = t_hi - t_lo;
  let mut t_f = 0.0f32;
  let fine = IVec3::from_array(fc);
  let pal0 = obj_sample_voxel(pool, dsc, fine);
  if pal0 != 0 {
    return Some((t_lo, pal0, 3));
  }
  for _ in 0..48 {
    if t_f >= span {
      return None;
    }
    let axis = if tmax_f[0] <= tmax_f[1] && tmax_f[0] <= tmax_f[2] {
      t_f = tmax_f[0];
      tmax_f[0] += delta[0];
      fc[0] += sign[0];
      0u8
    } else if tmax_f[1] <= tmax_f[2] {
      t_f = tmax_f[1];
      tmax_f[1] += delta[1];
      fc[1] += sign[1];
      1u8
    } else {
      t_f = tmax_f[2];
      tmax_f[2] += delta[2];
      fc[2] += sign[2];
      2u8
    };
    let pal = obj_sample_voxel(pool, dsc, IVec3::from_array(fc));
    if pal != 0 {
      return Some((t_lo + t_f, pal, axis));
    }
  }
  None
}

/// 单物体两级 DDA（镜像 WGSL `trace_object`）：世界 AABB 预剔除 → 局部变换 →
/// 局部 tile 盒 slab → cell 粗步 + fine 细步。t 为全局标尺（rd 不归一化）。
///
/// 返回 (全局 t, palette, **局部**面法线) 或 None（t_cap 内无命中）。
/// n_local 为物体局部空间单位向量、指向射线来向（世界法线 = rot · n_local）。
pub fn cpu_reference_object_ray(
  pool: &MovPoolPacked,
  idx: usize,
  origin: Vec3,
  dir: Vec3,
  t_cap: f32,
) -> Option<(f32, u8, Vec3)> {
  let dsc = &pool.descs[idx];
  // ---- 世界 AABB 预剔除 ----
  let (pos, rot, scale) = desc_transform(dsc);
  let (w_mn, w_mx) = (dsc.aabb_min.truncate(), dsc.aabb_max.truncate());
  let (t_enter, t_exit) = slab_box(origin, dir, w_mn, w_mx, 0.0, t_cap);
  if t_exit < t_enter.max(0.0) || t_enter >= t_cap {
    return None;
  }
  let t_hi_cap = t_exit.min(t_cap);
  if t_hi_cap <= t_enter.max(0.0) {
    return None;
  }
  // ---- 局部变换（rd 不归一化 → t 标尺不变）----
  let wp = origin - pos;
  let ro = Vec3::new(wp.dot(rot.x_axis), wp.dot(rot.y_axis), wp.dot(rot.z_axis)) / scale;
  let rd = Vec3::new(
    dir.dot(rot.x_axis),
    dir.dot(rot.y_axis),
    dir.dot(rot.z_axis),
  ) / scale;
  // ---- 局部 tile 盒 [0,512]³ slab ----
  let (tl_enter, tl_exit) = slab_box(ro, rd, Vec3::ZERO, Vec3::splat(512.0), 0.0, t_hi_cap);
  if tl_exit < tl_enter.max(0.0) {
    return None;
  }
  let tl0 = tl_enter.max(0.0);
  let tl1 = tl_exit.min(t_hi_cap);
  if tl1 <= tl0 {
    return None;
  }
  // ---- 两级 A&W（cell 0..31³，粗步上限 96 = 3×32）----
  // 全程「相对 start 的 t」标尺（与世界版一致），返回时 +tl0 还原全局 t
  let d = [rd.x, rd.y, rd.z];
  let sign = [
    if rd.x >= 0.0 { 1 } else { -1 },
    if rd.y >= 0.0 { 1 } else { -1 },
    if rd.z >= 0.0 { 1 } else { -1 },
  ];
  let delta = [
    if d[0].abs() > 1e-30 {
      (1.0 / d[0]).abs()
    } else {
      f32::INFINITY
    },
    if d[1].abs() > 1e-30 {
      (1.0 / d[1]).abs()
    } else {
      f32::INFINITY
    },
    if d[2].abs() > 1e-30 {
      (1.0 / d[2]).abs()
    } else {
      f32::INFINITY
    },
  ];
  let delta_c = [delta[0] * 16.0, delta[1] * 16.0, delta[2] * 16.0];
  let start = ro + rd * tl0;
  let s = [start.x, start.y, start.z];
  let mut cc = [
    ((s[0].floor() as i32) >> 4).clamp(0, 31) as u32,
    ((s[1].floor() as i32) >> 4).clamp(0, 31) as u32,
    ((s[2].floor() as i32) >> 4).clamp(0, 31) as u32,
  ];
  let mut tmax_c = [f32::INFINITY; 3];
  for i in 0..3 {
    if d[i].abs() > 1e-30 {
      let bnd = ((cc[i] as i32 + if sign[i] >= 0 { 1 } else { 0 }) << 4) as f32;
      let t = (bnd - s[i]) / d[i];
      tmax_c[i] = if t < 0.0 { 0.0 } else { t };
    }
  }
  let t_rel_max = tl1 - tl0;
  let mut t_in = 0.0f32;
  for _ in 0..96 {
    if cc.iter().any(|&c| c > 31) {
      break; // 走出物体 tile
    }
    let t_out = tmax_c[0].min(tmax_c[1]).min(tmax_c[2]);
    if obj_cell_occupied(pool, dsc, cc)
      && let Some((t_rel_hit, pal, axis)) = obj_fine_scan_cell(
        pool,
        dsc,
        start,
        rd,
        sign,
        delta,
        cc,
        t_in,
        t_out.min(t_rel_max),
      )
    {
      // 局部命中法线：-sign[axis] 单位轴；axis=3（起点在体内）→ -rd
      let n_local = if axis < 3 {
        let a = axis as usize;
        let mut n = Vec3::ZERO;
        n[a] = -sign[a] as f32;
        n
      } else {
        -rd.normalize_or_zero()
      };
      return Some((tl0 + t_rel_hit, pal, n_local));
    }
    if t_out >= t_rel_max {
      return None;
    }
    if tmax_c[0] <= tmax_c[1] && tmax_c[0] <= tmax_c[2] {
      t_in = tmax_c[0];
      tmax_c[0] += delta_c[0];
      cc[0] = (cc[0] as i32 + sign[0]) as u32;
    } else if tmax_c[1] <= tmax_c[2] {
      t_in = tmax_c[1];
      tmax_c[1] += delta_c[1];
      cc[1] = (cc[1] as i32 + sign[1]) as u32;
    } else {
      t_in = tmax_c[2];
      tmax_c[2] += delta_c[2];
      cc[2] = (cc[2] as i32 + sign[2]) as u32;
    }
  }
  None
}

/// `trace_scene()` CPU 参考：世界两级 DDA + 逐物体（世界 AABB 预剔除、t_cap 剪枝）取最近。
pub fn cpu_reference_trace_scene(
  world: &BrickMapBuffers,
  pool: &MovPoolPacked,
  origin: Vec3,
  dir: Vec3,
  t_max: f32,
) -> Option<MovHit> {
  let mut best = cpu_reference_dda_ray_two_level(world, origin, dir, t_max, 16384).map(|h| {
    // 世界命中法线：-sign[axis] 单位轴（射线朝 +axis 穿入 → 面在 -axis 侧）；
    // axis=3（起点在体内）→ -dir
    let normal = if h.axis < 3 {
      let a = h.axis as usize;
      let mut n = Vec3::ZERO;
      n[a] = if dir[a] >= 0.0 { -1.0 } else { 1.0 };
      n
    } else {
      -dir
    };
    MovHit {
      t: h.t,
      pal: h.pal,
      obj: OBJ_WORLD,
      normal,
    }
  });
  for i in 0..pool.descs.len() {
    let cap = best.as_ref().map_or(t_max, |b| b.t.min(t_max));
    if let Some((t, pal, n_local)) = cpu_reference_object_ray(pool, i, origin, dir, cap)
      && best.as_ref().is_none_or(|b| t < b.t)
    {
      // 局部法线 → 世界（uniform scale 不改方向，renormalize 消旋转舍入）
      let dsc = &pool.descs[i];
      let rot = Mat3::from_cols(
        dsc.rot0.truncate(),
        dsc.rot1.truncate(),
        dsc.rot2.truncate(),
      );
      best = Some(MovHit {
        t,
        pal,
        obj: i as u32,
        normal: (rot * n_local).normalize_or_zero(),
      });
    }
  }
  best
}

/// `trace_scene()` 遮挡快路径（P3.1 阴影射线）：t_max 内**任一**命中即 true。
/// 世界/物体各自的两级 DDA 找到首个命中即返回，无「最近」比较；
/// 阴影射线占比大时（每像素 × 光源 × 采样），此路径省掉逐物体 t 排序。
pub fn cpu_reference_scene_occluded(
  world: &BrickMapBuffers,
  pool: &MovPoolPacked,
  origin: Vec3,
  dir: Vec3,
  t_max: f32,
) -> bool {
  if cpu_reference_dda_ray_two_level(world, origin, dir, t_max, 16384).is_some() {
    return true;
  }
  (0..pool.descs.len()).any(|i| cpu_reference_object_ray(pool, i, origin, dir, t_max).is_some())
}

use crate::brickmap::dda::cpu_reference_dda_ray_two_level;

// ============================================================================
// 单测：布局 / 遮挡 / 旋转 / 缩放 / 交叠 / 世界路径回归
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use crate::brickmap::BrickMapBuilder;
  use gate_voxel::{MAX_LEVEL, TileGrid, fill_box};

  /// 全 [0,512)³ L0 满铺盒的 1-tile 世界（pal 指定）
  fn world_box(pal: u8) -> BrickMapBuffers {
    let mut g = TileGrid::new();
    fill_box(&mut g, IVec3::ZERO, IVec3::splat(512), 0, pal);
    BrickMapBuilder::build_full(&g).buffers().clone()
  }

  /// 局部 [0,ext)³ L0 盒的 1-tile 物体（pal 指定）
  fn object_box(ext: i32, pal: u8) -> BrickMapBuffers {
    let mut g = TileGrid::new();
    fill_box(&mut g, IVec3::ZERO, IVec3::splat(ext), 0, pal);
    BrickMapBuilder::build_full(&g).buffers().clone()
  }

  fn pack_one(bufs: &BrickMapBuffers, pos: Vec3, rot: Mat3, scale: f32) -> MovPoolPacked {
    pack_mov_pool(&[MovObject {
      buffers: bufs,
      pos,
      rot,
      scale,
    }])
  }

  #[test]
  fn mov_desc_layout_128b() {
    assert_eq!(std::mem::size_of::<MovDesc>(), 128);
    assert_eq!(std::mem::size_of::<MovGlobals>(), 16);
  }

  #[test]
  fn pack_layout_bases_sequential_and_content() {
    let a = object_box(64, 3);
    let b = object_box(32, 5);
    let pool = pack_mov_pool(&[
      MovObject {
        buffers: &a,
        pos: Vec3::ZERO,
        rot: Mat3::IDENTITY,
        scale: 1.0,
      },
      MovObject {
        buffers: &b,
        pos: Vec3::new(600.0, 0.0, 0.0),
        rot: Mat3::IDENTITY,
        scale: 2.0,
      },
    ]);
    assert_eq!(pool.descs.len(), 2);
    // 对象 0 基址从 0 起，区间大小精确
    let d0 = &pool.descs[0];
    assert_eq!(d0.bitmap_base, 0);
    assert_eq!((d0.dir_base - d0.bitmap_base) as usize, TILE_BITMAP_WORDS);
    assert_eq!((d0.node_base - d0.dir_base) as usize, CELL_DIR_WORDS);
    // 对象 1 基址接续对象 0
    let d1 = &pool.descs[1];
    assert_eq!(
      d1.bitmap_base as usize,
      d0.node_base as usize + a.globals.node_words as usize
    );
    assert_eq!((d1.dir_base - d1.bitmap_base) as usize, TILE_BITMAP_WORDS);
    assert_eq!(d1.leaves_base as usize, 0); // 对象 0 无 brick（L0 盒无 L4 细分）
    // 内容逐字一致：dirs（含绝对 node 偏移）原样拷贝
    assert_eq!(
      &pool.mov_struct[d1.dir_base as usize..d1.dir_base as usize + CELL_DIR_WORDS],
      &b.b_struct[DIR_BASE..DIR_BASE + CELL_DIR_WORDS]
    );
    // palette 拼接
    assert_eq!(
      &pool.mov_palette[d1.palette_base as usize..d1.palette_base as usize + PALETTE_WORDS],
      &b.b_palette[..PALETTE_WORDS]
    );
    // AABB：v1 契约 = 保守 tile 盒 [0,512]³·scale → 世界 1024³ @ (600,0,0)
    assert!((d1.aabb_min.truncate() - Vec3::new(600.0, 0.0, 0.0)).length() < 1e-4);
    assert!((d1.aabb_max.truncate() - Vec3::new(1624.0, 1024.0, 1024.0)).length() < 1e-4);
  }

  /// 空 pool：世界路径回归锚（trace_scene == 两级 DDA）
  #[test]
  fn trace_scene_world_only_matches_two_level() {
    let world = world_box(1);
    let pool = MovPoolPacked::default();
    let mut state: u64 = 0x9E3779B9;
    let mut next = || {
      state ^= state << 13;
      state ^= state >> 7;
      state ^= state << 17;
      state
    };
    for _ in 0..100 {
      let origin = Vec3::new(
        (next() % 4096) as f32 - 2048.0,
        (next() % 4096) as f32 - 2048.0,
        (next() % 4096) as f32 - 2048.0,
      );
      let dir = Vec3::new(
        (next() % 2001) as f32 - 1000.0,
        (next() % 2001) as f32 - 1000.0,
        (next() % 2001) as f32 - 1000.0,
      )
      .normalize();
      let a = cpu_reference_trace_scene(&world, &pool, origin, dir, 4096.0);
      let b = cpu_reference_dda_ray_two_level(&world, origin, dir, 4096.0, 16384).map(|h| {
        MovHit {
          t: h.t,
          pal: h.pal,
          obj: OBJ_WORLD,
          normal: Vec3::ZERO, // 等价性断言不比对法线
        }
      });
      match (a, b) {
        (Some(ha), Some(hb)) => {
          assert_eq!(ha.pal, hb.pal);
          assert_eq!(ha.obj, OBJ_WORLD);
          assert!((ha.t - hb.t).abs() <= 1.0, "t {} vs {}", ha.t, hb.t);
        }
        (None, None) => {}
        (a, b) => panic!("mismatch {a:?} vs {b:?}"),
      }
    }
  }

  /// 遮挡（物体在前）与被遮挡（世界在前）——P2.10 验收①
  #[test]
  fn occlusion_front_and_back() {
    let world = world_box(1); // 世界盒 [0,512]³
    let obj = object_box(64, 2); // 物体 64³
    // 物体在 +x 侧 (600..664)
    let pool = pack_one(&obj, Vec3::new(600.0, 100.0, 100.0), Mat3::IDENTITY, 1.0);
    // 从 +x 看向 -x：先穿物体（pal 2），物体在前
    let hit = cpu_reference_trace_scene(
      &world,
      &pool,
      Vec3::new(2000.0, 132.0, 132.0),
      -Vec3::X,
      8192.0,
    )
    .expect("必命中");
    assert_eq!(hit.obj, 0, "物体应遮挡世界");
    assert_eq!(hit.pal, 2);
    assert!((hit.t - (2000.0 - 664.0)).abs() < 1.0, "t={}", hit.t);
    // 从 -x 看向 +x：世界面 x=0 在前
    let hit = cpu_reference_trace_scene(
      &world,
      &pool,
      Vec3::new(-1000.0, 132.0, 132.0),
      Vec3::X,
      8192.0,
    )
    .expect("必命中");
    assert_eq!(hit.obj, OBJ_WORLD, "世界应遮挡物体");
    assert_eq!(hit.pal, 1);
    assert!((hit.t - 1000.0).abs() < 1.0);
  }

  /// yaw 90° 旋转正确性——P2.10 验收②
  #[test]
  fn rotation_yaw_90() {
    let world = BrickMapBuilder::build_full(&TileGrid::new())
      .buffers()
      .clone(); // 空世界
    let obj = object_box(64, 2); // 局部 [0,64)³
    // glam from_rotation_y(π/2)：col0=(0,0,-1), col1=(0,1,0), col2=(1,0,0)
    // 局部 z → 世界 x，局部 x → 世界 -z；局部 [0,64)³ →
    // 世界盒 x∈[1000,1064]（pos.x+局部z），z∈[936,1000]（pos.z-局部x），y∈[0,64]
    let rot = Mat3::from_rotation_y(std::f32::consts::FRAC_PI_2);
    let pos = Vec3::new(1000.0, 0.0, 1000.0);
    let pool = pack_one(&obj, pos, rot, 1.0);
    let hit_inside = cpu_reference_trace_scene(
      &world,
      &pool,
      Vec3::new(1032.0, 32.0, 3000.0),
      -Vec3::Z,
      8192.0,
    );
    let hit_outside = cpu_reference_trace_scene(
      &world,
      &pool,
      Vec3::new(1100.0, 32.0, 3000.0),
      -Vec3::Z,
      8192.0,
    );
    assert!(hit_inside.is_some(), "旋转后应命中 (x=1032 ∈ [1000,1064])");
    assert!(hit_outside.is_none(), "x=1100 在旋转后盒外，不应命中");
    let h = hit_inside.unwrap();
    assert_eq!(h.obj, 0);
    // 命中面 = 物体世界 z 最大面 z=1000（局部 x=0 面）
    assert!((h.t - (3000.0 - 1000.0)).abs() < 1.5, "t={}", h.t);
  }

  /// scale 2 体积加倍——P2.10 验收②
  #[test]
  fn scale_2_doubles_extent() {
    let world = BrickMapBuilder::build_full(&TileGrid::new())
      .buffers()
      .clone();
    let obj = object_box(64, 2); // 局部 64³ → 世界 128³
    let pos = Vec3::new(1000.0, 0.0, 1000.0);
    let pool = pack_one(&obj, pos, Mat3::IDENTITY, 2.0);
    // +x 探测：世界 [1000..1128] 应命中，1129+ 不命中
    let hit = cpu_reference_trace_scene(
      &world,
      &pool,
      Vec3::new(2000.0, 64.0, 1064.0),
      -Vec3::X,
      8192.0,
    )
    .expect("scale 2 盒内应命中");
    assert!((hit.t - (2000.0 - 1128.0)).abs() < 1.0, "t={}", hit.t);
    let miss = cpu_reference_trace_scene(
      &world,
      &pool,
      Vec3::new(2000.0, 64.0, 1129.0),
      -Vec3::X,
      8192.0,
    );
    assert!(miss.is_none(), "z=1129 已越出 scale 2 盒");
  }

  /// 与世界网格交叠：最近面获胜、无 panic / 无漏检——P2.10 验收③
  #[test]
  fn overlap_nearest_surface_wins() {
    let world = world_box(1); // [0,512]³ 实心
    let obj = object_box(64, 2);
    // 物体 [480..544]³：越过世界 x=512 面外伸 32——
    // （实心世界内部的物体任何方向都被世界面包围，只有外伸部分可见）
    let pool = pack_one(&obj, Vec3::new(480.0, 400.0, 400.0), Mat3::IDENTITY, 1.0);
    // +x 侧看向 -x：物体外伸面 x=544（t=1456）比世界面 x=512（t=1488）近
    let hit = cpu_reference_trace_scene(
      &world,
      &pool,
      Vec3::new(2000.0, 432.0, 432.0),
      -Vec3::X,
      8192.0,
    )
    .expect("必命中");
    assert_eq!(hit.obj, 0, "外伸物体面更近");
    assert!((hit.t - (2000.0 - 544.0)).abs() < 1.0);
    // -x 侧看向 +x：世界面 x=0 在前
    let hit = cpu_reference_trace_scene(
      &world,
      &pool,
      Vec3::new(-500.0, 432.0, 432.0),
      Vec3::X,
      8192.0,
    )
    .expect("必命中");
    assert_eq!(hit.obj, OBJ_WORLD);
    assert!((hit.t - 500.0).abs() < 1.0);
    // 斜穿交叠区（对角线方向）不 panic 且有确定命中
    let diag = cpu_reference_trace_scene(
      &world,
      &pool,
      Vec3::new(-200.0, -200.0, -200.0),
      Vec3::ONE.normalize(),
      8192.0,
    );
    assert!(diag.is_some());
    assert_eq!(diag.unwrap().obj, OBJ_WORLD);
  }

  /// L4 细分物体（brick slab + leaves_base 偏移路径）
  #[test]
  fn l4_object_brick_slab_offset() {
    let world = BrickMapBuilder::build_full(&TileGrid::new())
      .buffers()
      .clone();
    // 两个 L4 物体：uniform 胞 + 单杂色细体素（L3 槽内非均匀，无法折叠 → brick slab）。
    // 注意：对齐 L1 边界（8 fine）的混色内容会合法折叠到 L1 表，不产生 brick。
    let a = {
      let mut g = TileGrid::new();
      fill_box(&mut g, IVec3::ZERO, IVec3::splat(16), 0, 5); // cell 0 uniform 5
      g.set_voxel(IVec3::ZERO, MAX_LEVEL, 3)
        .expect("杂色细体素写入应生效");
      BrickMapBuilder::build_full(&g).buffers().clone()
    };
    let b = {
      let mut g = TileGrid::new();
      fill_box(&mut g, IVec3::ZERO, IVec3::splat(16), 0, 6);
      g.set_voxel(IVec3::ZERO, MAX_LEVEL, 4)
        .expect("杂色细体素写入应生效");
      BrickMapBuilder::build_full(&g).buffers().clone()
    };
    assert!(a.globals.brick_slabs > 0 && b.globals.brick_slabs > 0);
    // 物体 y 向错开，两条探测射线各只穿一个物体
    let pool = pack_mov_pool(&[
      MovObject {
        buffers: &a,
        pos: Vec3::new(500.0, 0.0, 0.0),
        rot: Mat3::IDENTITY,
        scale: 1.0,
      },
      MovObject {
        buffers: &b,
        pos: Vec3::new(600.0, 100.0, 0.0),
        rot: Mat3::IDENTITY,
        scale: 1.0,
      },
    ]);
    // 对象 1 的 slab 基址 = 对象 0 的 brick_slabs
    assert_eq!(pool.descs[1].leaves_base as u32, a.globals.brick_slabs);
    // 各自命中各自 palette（slab 偏移错了会读到对方的 brick）
    let ha =
      cpu_reference_trace_scene(&world, &pool, Vec3::new(2000.0, 8.0, 8.0), -Vec3::X, 8192.0)
        .expect("对象 0 应命中");
    assert_eq!((ha.obj, ha.pal), (0, 5), "命中对象 0 的 brick 体素");
    let hb = cpu_reference_trace_scene(
      &world,
      &pool,
      Vec3::new(2000.0, 108.0, 8.0),
      -Vec3::X,
      8192.0,
    )
    .expect("对象 1 应命中");
    assert_eq!((hb.obj, hb.pal), (1, 6), "命中对象 1 的 brick 体素");
  }
}
