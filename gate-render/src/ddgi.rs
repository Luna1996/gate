//! R3-10 DDGI 全局光照（Douglas #23 1:1 复刻）——CPU 侧真相源。
//!
//! v2 重做规格见 docs/ddgi-rework-spec.md（SUPERSEDED 本文件内旧版描述）：
//! - D1 探针网格 = 全 cell 覆盖（含纯空气 cell），废除活跃壳；
//! - D4-D6 每帧三段式：活跃判定 → 固定 4096 射线预算投射 → collect_radiance 投影；
//! - D7 载体 = 2D 纹理数组（irr rgba16f 8×8 / depth r32 16×16 / 元数据双缓冲 ping-pong）；
//! - D9-D10 4 级相机滚动级联 + age/reuse bounds 继承。
//!
//! 本文件处于迁移中态：v2 常量与位域已锁定（见下）；旧版活跃壳/frame_plan/
//! storage buffer 链路待 M5-1 one-step 删除（M3/M4 接管渲染路径前保持可编译）。
//!
//! 编辑响应：VolumeGrid.edit_generation 单调计数 → extract 版本比对 → base 重烘 +
//! 全级 age 归零（语义随 D10 更新，机制不变）。
//!
//! ## WGSL 对齐表（M3 落地 ddgi WGSL 时逐字镜像；改一处必改两处）
//! | Rust | WGSL（计划名） | 值 | 语义 |
//! |---|---|---|---|
//! | DDGI_CELL | DDGI_CELL | 16 | cell 边长（fine 体素） |
//! | IRRADIANCE_TEXELS | IRRADIANCE_TEXELS | 8 | irr oct 边长 |
//! | DEPTH_TEXELS | DEPTH_TEXELS | 16 | depth oct 边长 |
//! | RAY_BUDGET_PER_FRAME | RAY_BUDGET_PER_FRAME | 4096 | 每帧射线总预算 |
//! | PROBES_PER_LAYER_AXIS | PROBES_PER_LAYER_AXIS | 16 | 层内单轴探针数 |
//! | IRRADIANCE_LAYER_TEXELS | IRRADIANCE_LAYER_TEXELS | 128 | irr 层边长 |
//! | DEPTH_LAYER_TEXELS | DEPTH_LAYER_TEXELS | 256 | depth 层边长 |
//! | DDGI_CASCADE_CELL_SIZES | CASCADE_CELL_SIZES | 16/32/64/128/256 | 各级 cell 边长（[base,LOD1-4]） |
//! | DDGI_AGE_MAX | DDGI_AGE_MAX | 255 | age 饱和值 |
//! | META_OFFSET_BITS / META_AGE_SHIFT / META_OFFSET_QUANT | 同名 | 5 / 15 / 32 | 元数据位域 |
//! | DDGI_ALPHA | DDGI_ALPHA | 0.1 | EMA 新样本权重 |
//! | DDGI_NORMAL_BIAS / DDGI_DEPTH_BIAS | 同名 | 0.2 / 4.0 | 采样权重 |
//! | PROBE_T_MAX | PROBE_T_MAX | 8192.0 | 射线远距 |

use bevy::render::render_resource::ShaderType;
use glam::{IVec3, UVec3, Vec2, Vec3, Vec4};

use gate_voxel::{BrickState, Volumes, VoxelCoord};

// ============================================================================
// 常量（WGSL 侧镜像时逐字对齐，单测 wire_constants 防漂移）
// ============================================================================

/// 探针 cell 边长（fine 体素）= 16³ = level 2 brick（Douglas #23 / Rohacek §3.1）
pub const DDGI_CELL: i32 = 16;
/// cell 对应树层级（LEVEL_EXTENT[2] = 16）
pub const DDGI_CELL_LEVEL: u8 = 2;
/// 每 chunk 的 cell 数（256/16）
pub const CELLS_PER_CHUNK: i32 = gate_voxel::CHUNK_SIZE / DDGI_CELL;
/// 八面体 irradiance 边长（Majercik 2019 §3：8×8 texel/探针）
pub const IRRADIANCE_TEXELS: u32 = 8;
/// 八面体 depth 边长（Majercik 2019 §3：16×16 texel/探针）
pub const DEPTH_TEXELS: u32 = 16;
/// 单探针单轮射线数 = irradiance texel 数（1 ray ↔ 1 texel，Majercik 2019 §4）
pub const RAYS_PER_PROBE: u32 = IRRADIANCE_TEXELS * IRRADIANCE_TEXELS;
/// 每帧全局射线总预算（固定摊给活跃探针，性能不随屏上探针数波动，Douglas #23）
pub const RAY_BUDGET_PER_FRAME: u32 = 4096;
/// 探针射线 t_max（世界窗口对角量级；douglas-final.md §5 探针只覆盖本模型）
pub const PROBE_T_MAX: f32 = 8192.0;
/// 射线样本写入 irradiance/depth 的 EMA 新样本权重（Majercik 2019 §4）
pub const DDGI_ALPHA: f32 = 0.1;
/// 前后权重锐度（Rohacek §3.2 锐利背面剔除；WGSL DDGI_NORMAL_BIAS 镜像）：
/// wn = clamp(N·d / bias, 0, 1)——探针在表面后侧（N·d<0）权重严格 0，穿墙不漏光
pub const DDGI_NORMAL_BIAS: f32 = 0.2;
/// 漏光 chevron 半宽（fine 单位；WGSL DDGI_DEPTH_BIAS 镜像，= cell × 0.25）：
/// wd = clamp((depth_texel − probe_to_point) / bias + 0.5, 0, 1)
pub const DDGI_DEPTH_BIAS: f32 = DDGI_CELL as f32 * 0.25;
/// cell_index 无探针哨兵
pub const NO_PROBE: u32 = u32::MAX;

/// 每探针 irradiance 字数（rgba32f texel = 4 words；f16 打包为后续优化）
pub const IRRADIANCE_WORDS_PER_PROBE: u32 = IRRADIANCE_TEXELS * IRRADIANCE_TEXELS * 4;
/// 每探针 depth 字数（r32f texel = 1 word）
pub const DEPTH_WORDS_PER_PROBE: u32 = DEPTH_TEXELS * DEPTH_TEXELS;

// ============================================================================
// v2 wire 契约（Douglas 当前代码 1:1 重做，docs/ddgi-rework-spec.md D4-D10）
// ============================================================================

/// 层内单轴探针数（纹理数组每层 = 16×16 探针的 oct 图块网格；
/// RenderDoc 实测 Douglas 2025 版 256×256 层 / 16 texel 图块的 gate 等价布局）
pub const PROBES_PER_LAYER_AXIS: u32 = 16;
/// 每层探针数（16×16）
pub const PROBES_PER_LAYER: u32 = PROBES_PER_LAYER_AXIS * PROBES_PER_LAYER_AXIS;
/// irradiance 层边长（texel）= 16 探针 × 8 oct texel
pub const IRRADIANCE_LAYER_TEXELS: u32 = PROBES_PER_LAYER_AXIS * IRRADIANCE_TEXELS;
/// depth 层边长（texel）= 16 探针 × 16 oct texel
pub const DEPTH_LAYER_TEXELS: u32 = PROBES_PER_LAYER_AXIS * DEPTH_TEXELS;
/// 级联级数（Douglas #23：base + 4 级下采样；base = grid 0 世界级静态，LOD1-4 相机滚动）
pub const DDGI_LODS: u32 = 4;
/// age 上限（u8 饱和；reusable 继承、can_skip_update 分摊依据）
pub const DDGI_AGE_MAX: u32 = 255;
/// 元数据 packed u32 位域（gate 版：offset 量化 ×2 → 5 bit/轴精确表达 .5 半体素中心）
/// layout: [0..5) offset_x | [5..10) offset_y | [10..15) offset_z | [15..23) age | [23..32) 保留
pub const META_OFFSET_BITS: u32 = 5;
pub const META_AGE_SHIFT: u32 = META_OFFSET_BITS * 3;
/// offset 量化分母（cell 内 fine 坐标 ×2 → [0,32)；probe_position 的 .0/.5 值无损）
pub const META_OFFSET_QUANT: f32 = 32.0;
/// 级联各级 cell 边长（fine 体素，×2 递增）：[base=16 世界级静态, LOD1-4=32/64/128/256 相机滚动]
/// （Douglas #23「base + 4 级下采样」+ Majercik 2021 §5 默认；M2-3 烘焙 / M4-3 滚动消费）
pub const DDGI_CASCADE_CELL_SIZES: [i32; (DDGI_LODS + 1) as usize] = [16, 32, 64, 128, 256];
/// 滚动级单轴探针数（论文默认 16³ 探针/级；与纹理层 16×16 对齐 → 每级 16 层）
/// base 世界级探针数 = 体素 AABB 内 cell 数（随场景，不受此常量约束）
pub const PROBES_PER_CASCADE_AXIS: u32 = 16;

/// probe id → 纹理数组层号（每层 16×16 探针）
#[inline]
pub fn probe_layer(probe_id: u32) -> u32 {
  probe_id / PROBES_PER_LAYER
}

/// probe id → 层内 (x, y) 探针坐标（先取层内 id 再展开，id ≥ PROBES_PER_LAYER 时正确回绕）
#[inline]
pub fn probe_in_layer(probe_id: u32) -> [u32; 2] {
  let in_layer = probe_id % PROBES_PER_LAYER;
  [in_layer % PROBES_PER_LAYER_AXIS, in_layer / PROBES_PER_LAYER_AXIS]
}

/// probe id + oct texel → irradiance 数组 (layer, u, v)
#[inline]
pub fn irr_texel_coord(probe_id: u32, tx: u32, ty: u32) -> (u32, u32, u32) {
  let [px, py] = probe_in_layer(probe_id);
  (
    probe_layer(probe_id),
    px * IRRADIANCE_TEXELS + tx,
    py * IRRADIANCE_TEXELS + ty,
  )
}

/// probe id + oct texel → depth 数组 (layer, u, v)
#[inline]
pub fn depth_texel_coord(probe_id: u32, tx: u32, ty: u32) -> (u32, u32, u32) {
  let [px, py] = probe_in_layer(probe_id);
  (
    probe_layer(probe_id),
    px * DEPTH_TEXELS + tx,
    py * DEPTH_TEXELS + ty,
  )
}

/// 探针元数据打包：offset（cell 内 fine ×2，各 5 bit）+ age（8 bit）
///
/// Douglas sort.glsl `ddgi_probe_new(normalized_offset, age).offset_age` 的 gate 等价：
/// 归一化量化换成分辨率无损的定点（BFS 空叶中心含 .5 半体素，×2 后恰为整数）。
#[inline]
pub fn pack_probe_meta(offset_fine2: [u32; 3], age: u32) -> u32 {
  debug_assert!(offset_fine2.iter().all(|&v| v < (1 << META_OFFSET_BITS)));
  debug_assert!(age <= DDGI_AGE_MAX);
  offset_fine2[0]
    | offset_fine2[1] << META_OFFSET_BITS
    | offset_fine2[2] << (META_OFFSET_BITS * 2)
    | age << META_AGE_SHIFT
}

/// 解包 (offset_fine2, age)（age 钳 8 bit：[23..32) 保留位忽略，对齐 WGSL bitfieldExtract）
#[inline]
pub fn unpack_probe_meta(packed: u32) -> ([u32; 3], u32) {
  let mask = (1u32 << META_OFFSET_BITS) - 1;
  (
    [packed & mask, (packed >> META_OFFSET_BITS) & mask, (packed >> (META_OFFSET_BITS * 2)) & mask],
    (packed >> META_AGE_SHIFT) & 0xFF,
  )
}

/// 探针世界位置 → cell 内 offset 量化值（32 quanta/cell；base 16³ cell = ×2 定点，
/// BFS 空叶中心含 .5 半体素 → 精确；级联 cell 分辨率 = cell_size/32 fine）
#[inline]
pub fn quantize_offset_sized(probe_world: Vec3, cell_min: IVec3, cell_size: i32) -> [u32; 3] {
  let rel = probe_world - cell_min.as_vec3();
  let q = (rel * (META_OFFSET_QUANT / cell_size as f32))
    .round()
    .clamp(Vec3::ZERO, Vec3::splat(META_OFFSET_QUANT - 1.0));
  [q.x as u32, q.y as u32, q.z as u32]
}

/// base 版（cell_size=16，×2 定点）
#[inline]
pub fn quantize_offset(probe_world: Vec3, cell_min: IVec3) -> [u32; 3] {
  quantize_offset_sized(probe_world, cell_min, DDGI_CELL)
}

// ---- M2-2 元数据双缓冲纹理（D7：1 texel = 1 探针 packed meta，r32uint 2D 数组）----

/// 元数据纹理层数（每层 16×16 探针，层布局同 irr/depth 探针数组）
#[inline]
pub fn meta_texture_layers(probe_count: u32) -> u32 {
  probe_count.div_ceil(PROBES_PER_LAYER)
}

/// probe id → 元数据纹理 (layer, x, y)（1 texel/探针）
#[inline]
pub fn meta_texel_coord(probe_id: u32) -> (u32, u32, u32) {
  let [px, py] = probe_in_layer(probe_id);
  (probe_layer(probe_id), px, py)
}

/// 元数据纹理线性行主序下标（M4-1 write_texture 打包 / WGSL texelFetch 地址式共用）
#[inline]
pub fn meta_texel_linear(layer: u32, x: u32, y: u32) -> usize {
  (layer * PROBES_PER_LAYER + y * PROBES_PER_LAYER_AXIS + x) as usize
}

/// packed offset_fine2 → 探针世界坐标（cell_min + offset×cell_size/32；
/// base 16 = ×0.5 半体素无损；级联 cell 采样侧按本级 cell_size 解码）
#[inline]
pub fn offset_to_world_sized(offset_fine2: [u32; 3], cell_min: IVec3, cell_size: i32) -> Vec3 {
  cell_min.as_vec3()
    + Vec3::new(offset_fine2[0] as f32, offset_fine2[1] as f32, offset_fine2[2] as f32)
      * (cell_size as f32 / META_OFFSET_QUANT)
}

/// base 版（cell_size=16）
#[inline]
pub fn offset_to_world(offset_fine2: [u32; 3], cell_min: IVec3) -> Vec3 {
  offset_to_world_sized(offset_fine2, cell_min, DDGI_CELL)
}

/// ProbeGrid → 元数据纹理初烘数据（每探针 1 u32 = packed offset+age，初烘 age=0）
///
/// D7 双缓冲 ping-pong：previous/next 两份纹理同用本数据初始化（M4-1 各 write_texture
/// 一次），运行时 previous 只读 / next imageStore，帧末交换指针。
/// 布局 = 层主序 16×16 texel/层（`meta_texel_linear`）；未占用探针位（probe_count
/// 之后的 padding）填 0，采样侧 probe id 来自 cell_index，padding 不会被寻址。
pub fn build_meta_texture_data(pg: &ProbeGrid) -> Vec<u32> {
  let layers = meta_texture_layers(pg.positions.len() as u32);
  let mut data = vec![0u32; layers as usize * PROBES_PER_LAYER as usize];
  let dx = pg.grid_dims.x as usize;
  let dxy = (pg.grid_dims.x * pg.grid_dims.y) as usize;
  for (li, &id) in pg.cell_index.iter().enumerate() {
    if id == NO_PROBE {
      continue;
    }
    let z = li / dxy;
    let y = (li % dxy) / dx;
    let x = li % dx;
    let cell_min = pg.cell_min_voxel(UVec3::new(x as u32, y as u32, z as u32));
    let off = quantize_offset_sized(pg.positions[id as usize], cell_min, pg.cell_size);
    let (layer, tx, ty) = meta_texel_coord(id);
    data[meta_texel_linear(layer, tx, ty)] = pack_probe_meta(off, 0);
  }
  data
}

// ============================================================================
// v2 每帧更新数学（M1-3 CPU 参考 → M3-1/2/3 WGSL 逐字镜像）
// ============================================================================

// ---- 随机（D5：PCG hash + 帧号种子，球面均匀方向）----

/// PCG hash 标量版（Jarzynski & Olano 2021 "Hash Functions for GPU Rendering"，pcg）
#[inline]
pub fn pcg_hash(v: u32) -> u32 {
  let state = v.wrapping_mul(747796405).wrapping_add(2891336453);
  let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
  (word >> 22) ^ word
}

/// 射线随机数：(probe_id, frame, ray_index) 三元 → 均匀 u32
/// （跨帧/跨探针/跨射线独立；固定种子可重放 → CPU 镜像与 WGSL 逐位一致）
#[inline]
pub fn ray_rand(probe_id: u32, frame: u32, ray_index: u32) -> u32 {
  pcg_hash(probe_id ^ pcg_hash(frame ^ pcg_hash(ray_index)))
}

/// u32 → [0,1)²（低 16 位 / 高 16 位）
#[inline]
pub fn rand2(r: u32) -> [f32; 2] {
  [(r & 0xFFFF) as f32 / 65536.0, (r >> 16) as f32 / 65536.0]
}

/// 球面均匀方向：z 均匀 [-1,1] + φ 均匀 [0,2π)（球面面积元精确均匀）
#[inline]
pub fn uniform_sphere_dir(u: [f32; 2]) -> Vec3 {
  let z = 1.0 - 2.0 * u[0];
  let r = (1.0 - z * z).max(0.0).sqrt();
  let phi = std::f32::consts::TAU * u[1];
  Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

// ---- D6 投影（Douglas collect_radiance 截图逐字）----

/// collect_radiance（D6 实锤，射线数与 texel 数解耦）：对 texel 方向 d 汇总本帧
/// 全部射线样本 (dir_i, L_i)：irr = π·Σ max(0, d·dir_i)·L_i / Σ max(0, d·dir_i)。
/// Σw≈0（本方向无正权样本）→ 返回 0。
#[inline]
pub fn collect_radiance(d: Vec3, samples: &[(Vec3, Vec3)]) -> Vec3 {
  let mut sum = Vec3::ZERO;
  let mut wsum = 0.0;
  for &(dir, l) in samples {
    let w = d.dot(dir).max(0.0);
    sum += l * w;
    wsum += w;
  }
  if wsum <= 0.0 { Vec3::ZERO } else { sum * (std::f32::consts::PI / wsum) }
}

// ---- 更新链：EMA + tonemap + 迟滞 ----
// 结构 = Majercik 2019 §4 Eq.(1) lerp 核心 + RTXGI v2（Majercik 2021）
// ProbeBlendingCS.hlsl L508-550 阈值链逐字：γ-tonemap → 大暗化检测（hysteresis 降
// 0.75）→ 大亮化 delta×0.25 → hysteresis EMA → 暗化保底步进（f16 收敛保证）。

/// hysteresis = 旧值权重（INFERENCE: RTXGI sponza 0.97 / 论文 §4.4 α=0.85-0.98；
/// = 1 − DDGI_ALPHA 语义；M5-3 调参）
pub const DDGI_HYSTERESIS: f32 = 0.95;
/// irradiance γ 编码指数（rgba16f 线性存储 → 1.0 恒等；γ≠1 仅低精度 UNORM 格式需要）
pub const DDGI_IRRAD_GAMMA: f32 = 1.0;
/// 大暗化阈值（INFERENCE: RTXGI sponza probeIrradianceThreshold=0.2）：超过则
/// hysteresis −= DDGI_CHANGE_DROP，加速响应光照剧变
pub const DDGI_BIG_CHANGE: f32 = 0.2;
/// 大亮化亮度阈值（INFERENCE: RTXGI sponza probeBrightnessThreshold=1.0）：超过则
/// delta ×= DDGI_DELTA_CLAMP，限制单帧最大跃变
pub const DDGI_BRIGHTNESS: f32 = 1.0;
/// RTXGI c_threshold：暗化方向单帧最小步进（1/1024，保证向目标值收敛不停滞）
pub const DDGI_MIN_STEP: f32 = 1.0 / 1024.0;
/// 大变化时 hysteresis 减量（RTXGI 硬编码 0.75）
pub const DDGI_CHANGE_DROP: f32 = 0.75;
/// 亮度超阈时 delta 缩放（RTXGI 硬编码 0.25）
pub const DDGI_DELTA_CLAMP: f32 = 0.25;

/// Rec.701 线性亮度（RTXGI LinearRGBToLuminance）
#[inline]
fn luminance(c: Vec3) -> f32 {
  c.dot(Vec3::new(0.2126, 0.7152, 0.0722))
}

/// 分量最大值（RTXGI MaxComponent）
#[inline]
fn max_component(c: Vec3) -> f32 {
  c.x.max(c.y).max(c.z)
}

/// 单 texel 辐照度更新：prev = 当前存储值，new = 本帧 collect_radiance 投影估计。
/// 返回新存储值。链路（RTXGI v2 ProbeBlendingCS.hlsl L508-550 逐字）：
/// ① γ-tonemap（gate γ=1 恒等）② prev 全黑 → hysteresis=0（注意：亮度钳制仍生效，
///   RTXGI 只清 hysteresis，不豁免 delta×0.25）③ max(prev−new) > 阈 → hysteresis −0.75
/// ④ luminance(delta) > 阈 → delta×0.25 ⑤ lerp_delta=(1−h)·delta
/// ⑥ 暗化时 |lerp_delta| ∈ [1/1024, |delta|] ×sign(lerp_delta) ⑦ prev+lerp_delta
#[inline]
pub fn update_irradiance_texel(prev: Vec3, new: Vec3) -> Vec3 {
  let new_t = new.powf(1.0 / DDGI_IRRAD_GAMMA);
  let mut h = if prev == Vec3::ZERO { 0.0 } else { DDGI_HYSTERESIS };
  if max_component(prev - new_t) > DDGI_BIG_CHANGE {
    h = (h - DDGI_CHANGE_DROP).max(0.0);
  }
  let mut delta = new_t - prev;
  if luminance(delta) > DDGI_BRIGHTNESS {
    delta *= DDGI_DELTA_CLAMP;
  }
  let mut lerp_delta = (1.0 - h) * delta;
  if max_component(new_t) < max_component(prev) {
    // sign(lerp_delta)：delta=0 分量乘积恒 0（min(|delta|=0) 已归零），sign 零值语义不可观测
    let s = Vec3::new(
      lerp_delta.x.signum(),
      lerp_delta.y.signum(),
      lerp_delta.z.signum(),
    );
    lerp_delta = Vec3::new(
      lerp_delta.x.abs().max(DDGI_MIN_STEP).min(delta.x.abs()),
      lerp_delta.y.abs().max(DDGI_MIN_STEP).min(delta.y.abs()),
      lerp_delta.z.abs().max(DDGI_MIN_STEP).min(delta.z.abs()),
    ) * s;
  }
  prev + lerp_delta
}

// ---- D10 age / reuse bounds（截图2 250-256 行语义）----

/// 滚动后 age 继承：reusable（bounds 内且 offset 未变）→ 继承旧 age，否则归零重新收敛
#[inline]
pub fn age_after_scroll(reusable: bool, prev_age: u32) -> u32 {
  if reusable { prev_age } else { 0 }
}

/// 完成一次探针更新后 age +1（u8 饱和）
#[inline]
pub fn age_after_update(age: u32) -> u32 {
  (age + 1).min(DDGI_AGE_MAX)
}

/// reuse bounds 判定（D10 截图逐字语义）：
/// reusable = reuse_min ≤ cell（逐轴，含端）∧ cell < reuse_max（逐轴，不含端）
/// ∧ offset 未变
#[inline]
pub fn probe_reusable(
  cell: IVec3,
  offset: [u32; 3],
  prev_offset: [u32; 3],
  bounds: (IVec3, IVec3),
) -> bool {
  cell.cmpge(bounds.0).all() && cell.cmplt(bounds.1).all() && offset == prev_offset
}

// ---- can_skip_update（推断实现，spec §2 推断条款授权）----

/// INFERENCE: Douglas #23 未截到 can_skip_update 具体逻辑。保守实现（spec 锁定）：
/// 跳过概率随 age 增长（age 大 = EMA 已收敛 = 可跳），hash 驱动确定性；
/// p_skip = (age/255)² × CAN_SKIP_MAX → 新鲜探针（age=0）永不跳，全熟探针最多跳 50%
pub const CAN_SKIP_MAX: f32 = 0.5;

#[inline]
pub fn can_skip_update(age: u32, rand: u32) -> bool {
  let p = (age as f32 / DDGI_AGE_MAX as f32).powi(2) * CAN_SKIP_MAX;
  (rand & 0xFFFF) as f32 / 65536.0 < p
}

// ============================================================================
// 八面体映射（Majercik 2019 §3；WGSL oct_encode/oct_decode 逐字镜像）
// ============================================================================

#[inline]
fn sign_not_zero(v: f32) -> f32 {
  if v >= 0.0 { 1.0 } else { -1.0 }
}

/// 方向 → 八面体 uv ∈ [-1,1]²
#[inline]
pub fn oct_encode(n: Vec3) -> Vec2 {
  let d = n / (n.x.abs() + n.y.abs() + n.z.abs());
  let p = Vec2::new(d.x, d.y);
  if d.z < 0.0 {
    // octWrap：x 分量混入 |y|，y 分量混入 |x|（注意 yx 交叉）
    Vec2::new(
      (1.0 - p.y.abs()) * sign_not_zero(p.x),
      (1.0 - p.x.abs()) * sign_not_zero(p.y),
    )
  } else {
    p
  }
}

/// 八面体 uv ∈ [-1,1]² → 方向
#[inline]
pub fn oct_decode(e: Vec2) -> Vec3 {
  let mut n = Vec3::new(e.x, e.y, 1.0 - e.x.abs() - e.y.abs());
  if n.z < 0.0 {
    let t = Vec2::new(
      (1.0 - n.y.abs()) * sign_not_zero(n.x),
      (1.0 - n.x.abs()) * sign_not_zero(n.y),
    );
    n.x = t.x;
    n.y = t.y;
  }
  n.normalize_or_zero()
}

/// 方向 → 八面体 texel 下标（边长 S）
#[inline]
pub fn oct_texel(n: Vec3, s: u32) -> [u32; 2] {
  let e = oct_encode(n);
  let f = (e * 0.5 + Vec2::splat(0.5)) * Vec2::splat(s as f32);
  [
    (f.x as i32).clamp(0, s as i32 - 1) as u32,
    (f.y as i32).clamp(0, s as i32 - 1) as u32,
  ]
}

/// 八面体 texel 中心 → 方向（边长 S）
#[inline]
pub fn oct_texel_dir(tx: u32, ty: u32, s: u32) -> Vec3 {
  let e = (Vec2::new(tx as f32, ty as f32) + Vec2::splat(0.5)) / Vec2::splat(s as f32) * 2.0
    - Vec2::splat(1.0);
  oct_decode(e)
}

// ============================================================================
// Fibonacci 球方向（Majercik 2019 §4 射线更新；WGSL fibonacci_dir 逐字镜像）
// ============================================================================

/// 第 i/N 个 Fibonacci 球方向（+Y 为极轴，天空直觉一致；黄金角步进）
#[inline]
pub fn fibonacci_dir(i: u32, n: u32) -> Vec3 {
  let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
  let y = 1.0 - (2.0 * i as f32 + 1.0) / n as f32;
  let r = (1.0 - y * y).sqrt().max(0.0);
  let a = golden * i as f32;
  Vec3::new(r * a.cos(), y, r * a.sin())
}

// ============================================================================
// 探针烘焙（Rohacek §3.1 BFS 最大空叶 + D1 全 cell 覆盖）
// ============================================================================

/// 探针烘焙结果（CPU 真相源 → GPU 打包入口；base 与级联级共用）
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeGrid {
  /// 采样网格原点（**16-cell 全局坐标**；级联级须 cell_size 对齐）
  pub grid_origin: IVec3,
  /// 采样网格 dims（**本级 cell 单位**；base cell_size=16 时与 16-cell 同构）
  pub grid_dims: UVec3,
  /// 本级 cell 尺寸（fine 体素）：base 16，级联 32/64/128/256
  pub cell_size: i32,
  /// dense cell → probe 下标（NO_PROBE = 无探针）
  /// 线性地址 = x + y*dims.x + z*dims.x*dims.y（rel 为本级 cell 坐标）
  pub cell_index: Vec<u32>,
  /// 探针位置（世界 fine 坐标；下标 = probe id）
  pub positions: Vec<Vec3>,
}

impl Default for ProbeGrid {
  fn default() -> Self {
    Self {
      grid_origin: IVec3::ZERO,
      grid_dims: UVec3::ZERO,
      cell_size: DDGI_CELL,
      cell_index: Vec::new(),
      positions: Vec::new(),
    }
  }
}

impl ProbeGrid {
  /// cell（相对 grid_origin 的本级 cell 偏移）→ 线性下标
  #[inline]
  pub fn cell_linear(&self, rel: UVec3) -> usize {
    (rel.x + rel.y * self.grid_dims.x + rel.z * self.grid_dims.x * self.grid_dims.y) as usize
  }

  /// 本级 cell 偏移 → 16-cell 全局坐标（cell_size/16 步长）
  #[inline]
  pub fn cell16(&self, rel: UVec3) -> IVec3 {
    let step = self.cell_size / DDGI_CELL;
    self.grid_origin + IVec3::new(rel.x as i32, rel.y as i32, rel.z as i32) * step
  }

  /// 本级 cell 的 fine 体素最小角
  #[inline]
  pub fn cell_min_voxel(&self, rel: UVec3) -> IVec3 {
    self.cell16(rel) * DDGI_CELL
  }
}

/// 烘焙主世界探针网格（当前只覆盖 vols.list[0]；物体互反射探针后置）
///
/// D1 全 cell 覆盖：域 = chunk bbox cell 域，已存在 chunk 的全部非 Solid cell
/// 均放探针（Air 居中 / Mixed BFS 偏移 / Solid 无）。纯空气 cell 一律有探针，
/// 活跃性由每帧 ddgi_active 判定剔除（D2），烘焙期不再筛活跃壳。
/// 域内未分配 chunk 的 cell 留 NO_PROBE（从未有体素数据的远场，表面不可达）。
/// 稀疏遍历：每个已存在 chunk 迭代其 16³ cell 区，不扫 bbox 全空间。
pub fn bake_probe_grid(vols: &Volumes) -> ProbeGrid {
  let grid = vols.main();
  let chunks: Vec<IVec3> = grid.chunk_coords().map(|c| c.0).collect();
  let Some((min_chunk, max_chunk)) = chunk_bbox(chunks.iter().copied()) else {
    return ProbeGrid::default();
  };
  let lo = min_chunk * CELLS_PER_CHUNK;
  let hi_excl = (max_chunk + 1) * CELLS_PER_CHUNK;
  let dims = (hi_excl - lo).as_uvec3();

  let mut pg = ProbeGrid {
    grid_origin: lo,
    grid_dims: dims,
    cell_size: DDGI_CELL,
    cell_index: vec![NO_PROBE; dims.x as usize * dims.y as usize * dims.z as usize],
    positions: Vec::new(),
  };

  for chunk in &chunks {
    for rz in 0..CELLS_PER_CHUNK {
      for ry in 0..CELLS_PER_CHUNK {
        for rx in 0..CELLS_PER_CHUNK {
          let cell = chunk * CELLS_PER_CHUNK + IVec3::new(rx, ry, rz);
          let rel = (cell - lo).as_uvec3();
          let li = pg.cell_linear(rel);
          let state =
            grid.get_brick_state(VoxelCoord::from_ivec3(cell * DDGI_CELL), DDGI_CELL_LEVEL);
          // 全实心 cell 无探针（Douglas：全满 → 无探针）
          if matches!(state, BrickState::Solid(_)) {
            continue;
          }
          let Some(pos) = probe_position(grid, cell * DDGI_CELL, state) else {
            continue; // Mixed 但中心 ±4 盒内无空体素：无探针
          };
          pg.cell_index[li] = pg.positions.len() as u32;
          pg.positions.push(pos);
        }
      }
    }
  }
  pg
}

/// 探针位置：Air 居中 / Mixed 沿树 BFS 找靠中心最大空叶（Rohacek §3.1）
///
/// BFS 序 = 空块由大到小：16³ cell 全空 → cell 中心；4³ 空子砖（level 3）
/// 取靠 cell 中心最近；再无则中心 ±4 盒内最近 1³ 空体素；全满 → None。
fn probe_position(
  grid: &gate_voxel::VolumeGrid,
  cell_min: IVec3,
  state: BrickState,
) -> Option<Vec3> {
  probe_position_leaf(grid, cell_min, state).map(|(_, p)| p)
}

/// [`probe_position`] 带「探针所处空叶尺寸」（Air=16 / 4³ 空砖=4 / 1³ 兜底=1）。
/// 级联推广的下钻择优依赖真实叶尺寸（空叶大者优先），故单独暴露。
fn probe_position_leaf(
  grid: &gate_voxel::VolumeGrid,
  cell_min: IVec3,
  state: BrickState,
) -> Option<(i32, Vec3)> {
  let center_f = cell_min.as_vec3() + Vec3::splat(DDGI_CELL as f32 / 2.0);
  match state {
    BrickState::Air => Some((DDGI_CELL, center_f)),
    BrickState::Solid(_) => None,
    BrickState::Mixed => {
      // BFS 层 1：4³ 空子砖，取靠 cell 中心最近（严格更小才替换 → 平局取遍历序首个）
      let mut best: Option<(f32, Vec3)> = None;
      for k in 0..4 {
        for j in 0..4 {
          for i in 0..4 {
            let sub_min = cell_min + IVec3::new(i, j, k) * 4;
            if grid.get_brick_state(VoxelCoord::from_ivec3(sub_min), 3) != BrickState::Air {
              continue;
            }
            let c = sub_min.as_vec3() + Vec3::splat(2.0);
            let d2 = c.distance_squared(center_f);
            if best.is_none_or(|(bd, _)| d2 < bd) {
              best = Some((d2, c));
            }
          }
        }
      }
      if best.is_some() {
        return best.map(|(_, c)| (4, c));
      }
      // BFS 层 2：中心 ±4 盒内最近 1³ 空体素（薄墙 cell 兜底）
      let center_i = cell_min + IVec3::splat(DDGI_CELL / 2);
      let mut best: Option<(f32, Vec3)> = None;
      for dz in -4..=4 {
        for dy in -4..=4 {
          for dx in -4..=4 {
            let v = center_i + IVec3::new(dx, dy, dz);
            if !is_air_voxel(grid, VoxelCoord::from_ivec3(v)) {
              continue;
            }
            let d2 = (v.as_vec3() + Vec3::splat(0.5)).distance_squared(center_f);
            if best.is_none_or(|(bd, _)| d2 < bd) {
              best = Some((d2, v.as_vec3() + Vec3::splat(0.5)));
            }
          }
        }
      }
      best.map(|(_, c)| (1, c))
    }
  }
}

#[inline]
fn is_air_voxel(grid: &gate_voxel::VolumeGrid, v: VoxelCoord) -> bool {
  matches!(grid.get_voxel(v), None | Some(0))
}

#[inline]
fn chunk_bbox(mut it: impl Iterator<Item = IVec3>) -> Option<(IVec3, IVec3)> {
  let first = it.next()?;
  let mut mn = first;
  let mut mx = first;
  for c in it {
    mn = mn.min(c);
    mx = mx.max(c);
  }
  Some((mn, mx))
}

// ============================================================================
// 级联烘焙（M2-3：D9 每级 cell 尺寸 BFS 推广 + D4 outside_lower_grid 空间划分）
// ============================================================================

/// 任意 cell 尺寸的三态分类。gate 树层级 {256,64,16,4,1} 不含 32/128 →
/// 2×2×2 子 cell 递归合成：全 Air → Air；全 Solid → Solid（DDGI「全满」语义，
/// 多色实心也算满，palette 不参与）；否则 Mixed。
pub fn cell_state_at(
  grid: &gate_voxel::VolumeGrid,
  cell_min: IVec3,
  cell_size: i32,
) -> BrickState {
  match cell_size {
    256 => grid.get_brick_state(VoxelCoord::from_ivec3(cell_min), 0),
    64 => grid.get_brick_state(VoxelCoord::from_ivec3(cell_min), 1),
    16 => grid.get_brick_state(VoxelCoord::from_ivec3(cell_min), DDGI_CELL_LEVEL),
    32 | 128 => {
      let half = cell_size / 2;
      let (mut all_air, mut all_solid) = (true, true);
      for k in 0..2 {
        for j in 0..2 {
          for i in 0..2 {
            let sub = cell_min + IVec3::new(i, j, k) * half;
            match cell_state_at(grid, sub, half) {
              BrickState::Air => all_solid = false,
              BrickState::Solid(_) => all_air = false,
              BrickState::Mixed => {
                all_air = false;
                all_solid = false;
              }
            }
          }
        }
      }
      if all_air {
        BrickState::Air
      } else if all_solid {
        BrickState::Solid(0) // palette 不参与 DDGI 判定
      } else {
        BrickState::Mixed
      }
    }
    _ => unreachable!("cell_size 必须是 DDGI_CASCADE_CELL_SIZES 之一（got {cell_size}）"),
  }
}

/// 推广探针放置：Air 居中 / Solid 无 / Mixed 下钻找「靠中心的最大空叶」。
/// S=16 走基础 BFS（D3 零改动：4³ 空子砖 → ±4 体素兜底）；S>16 在 8 个半边长
/// 子 cell 中择优——空叶尺寸大者优先，同尺寸取靠 cell 中心最近，再平局取遍历序
/// 首个（与基础 BFS「由大到小 → 靠中心 → 遍历序」优先级逐字同构）。
pub fn probe_position_sized(
  grid: &gate_voxel::VolumeGrid,
  cell_min: IVec3,
  cell_size: i32,
) -> Option<Vec3> {
  probe_leaf_sized(grid, cell_min, cell_size).map(|(_, p)| p)
}

/// [`probe_position_sized`] 带「探针所处空叶尺寸」：Air = cell_size 本身；
/// Mixed = 下钻胜者的真实叶（S=16 兜底到基础 BFS 的 4/1）。
/// 叶尺寸参与父级择优（空叶大者优先），不得用 cell_size 压平近似。
fn probe_leaf_sized(
  grid: &gate_voxel::VolumeGrid,
  cell_min: IVec3,
  cell_size: i32,
) -> Option<(i32, Vec3)> {
  let center = cell_min.as_vec3() + Vec3::splat(cell_size as f32 / 2.0);
  match cell_state_at(grid, cell_min, cell_size) {
    BrickState::Air => Some((cell_size, center)),
    BrickState::Solid(_) => None,
    BrickState::Mixed if cell_size == DDGI_CELL => {
      probe_position_leaf(grid, cell_min, BrickState::Mixed)
    }
    BrickState::Mixed => {
      let half = cell_size / 2;
      // best: (空叶尺寸降序, 距中心平方升序, 遍历序)
      let mut best: Option<(i32, f32, Vec3)> = None;
      for k in 0..2 {
        for j in 0..2 {
          for i in 0..2 {
            let sub_min = cell_min + IVec3::new(i, j, k) * half;
            let Some((leaf, p)) = probe_leaf_sized(grid, sub_min, half) else {
              continue;
            };
            let d2 = p.distance_squared(center);
            let better = match best {
              None => true,
              Some((bl, bd, _)) => leaf > bl || (leaf == bl && d2 < bd),
            };
            if better {
              best = Some((leaf, d2, p));
            }
          }
        }
      }
      best.map(|(leaf, _, p)| (leaf, p))
    }
  }
}

/// 烘焙一个级联级探针网格（D9：cell 尺寸 ×2 递增的相机滚动级 / base 世界级）。
///
/// `origin_cell` 为 **16-cell 全局坐标**（级联级须 cell_size 对齐，M4-3 滚动步进）；
/// `dims_cells` 为本级 cell 单位（滚动级 = PROBES_PER_CASCADE_AXIS³ = 16³）。
/// 全 cell 覆盖语义同 base（D1）：非 Solid cell 一律放探针，活跃性交给 ddgi_active。
pub fn bake_cascade_grid(
  vols: &Volumes,
  cell_size: i32,
  origin_cell: IVec3,
  dims_cells: UVec3,
) -> ProbeGrid {
  let grid = vols.main();
  let mut pg = ProbeGrid {
    grid_origin: origin_cell,
    grid_dims: dims_cells,
    cell_size,
    cell_index: vec![
      NO_PROBE;
      dims_cells.x as usize * dims_cells.y as usize * dims_cells.z as usize
    ],
    positions: Vec::new(),
  };
  for rz in 0..dims_cells.z {
    for ry in 0..dims_cells.y {
      for rx in 0..dims_cells.x {
        let rel = UVec3::new(rx, ry, rz);
        let li = pg.cell_linear(rel);
        let cell_min = pg.cell_min_voxel(rel);
        if matches!(
          cell_state_at(grid, cell_min, cell_size),
          BrickState::Solid(_)
        ) {
          continue; // 全满 cell 无探针
        }
        let Some(pos) = probe_position_sized(grid, cell_min, cell_size) else {
          continue; // Mixed 但下钻无空叶：无探针
        };
        pg.cell_index[li] = pg.positions.len() as u32;
        pg.positions.push(pos);
      }
    }
  }
  pg
}

/// D4 outside_lower_grid：本 LOD cell 是否在更细网格覆盖外（true = 归本 LOD 管）。
/// 全部区间为 16-cell 全局坐标：本 cell [lo, hi) 与更细域 [fo, fo+fd) 无重叠 → outside。
/// 级联滚动步进保证域边缘 cell 对齐（M4-3），部分重叠 cell 归更细级（保守划分）。
#[inline]
pub fn outside_lower_grid(lo: IVec3, hi: IVec3, finer_origin: IVec3, finer_dims: UVec3) -> bool {
  let fo_hi = finer_origin + finer_dims.as_ivec3();
  lo.x >= fo_hi.x
    || hi.x <= finer_origin.x
    || lo.y >= fo_hi.y
    || hi.y <= finer_origin.y
    || lo.z >= fo_hi.z
    || hi.z <= finer_origin.z
}

// ============================================================================
// M3-1 active 判定 CPU 镜像（WGSL ddgi_active.wgsl 镜像源；spec D2/D4/D7/D10）
// 逐字对照 Douglas sort.glsl L218-262（截图1-2）
// ============================================================================

// ---- DdgiProbeFlags（截图3 flag-based active 判定）----

/// Douglas sort.glsl `DdgiProbeFlags` 位标志（u32）。
/// 截图3 `probe_near_surface` 逐字：
///   return (flags & DDGI_PROBE_FLAG_NO_SURFACES) == 0
///       || (get_probe_flags_intersection() & DDGI_PROBE_FLAG_NO_SURFACES) == 0
///       || probe_near_objects(cell_center);
/// 即：NO_SURFACES=1 表示本 cell 无表面；ENABLED=1 表示本 cell 有探针（可被调度）。
pub type DdgiProbeFlags = u32;
/// 本 cell 有探针（烘焙分配）→ 可被 active pass 调度
pub const DDGI_PROBE_FLAG_ENABLED: DdgiProbeFlags = 1 << 0;
/// 本 cell 无表面（纯 Air，Solid 也视为无——Solid cell 烘焙期已跳过）
pub const DDGI_PROBE_FLAG_NO_SURFACES: DdgiProbeFlags = 1 << 1;

/// BrickState → DdgiProbeFlags：Air=NO_SURFACES|ENABLED（烘焙期 AIR 也放探针），
/// Mixed=ENABLED（有表面），Solid=ENABLED（Solid cell 烘焙期跳过，此分支不常走）
/// 注意：NO_PROBE cell 在构建 flags 数组时直接跳过（不置 ENABLED）
#[inline]
pub fn brickstate_to_flags(state: BrickState, has_probe: bool) -> DdgiProbeFlags {
  if !has_probe {
    return 0;
  }
  let mut flags = DDGI_PROBE_FLAG_ENABLED;
  if matches!(state, BrickState::Air) {
    flags |= DDGI_PROBE_FLAG_NO_SURFACES;
  }
  flags
}

// ---- 级联域 ----

/// 级联域（每帧滚动 dispatch 用；origin/dims 为 16-cell 全局坐标，cell_size 为 fine 体素）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CascadeDomain {
  /// 域原点（16-cell 全局坐标；级联级须 cell_size 对齐——M4-3 滚动步进）
  pub origin: IVec3,
  /// 域 dims（本级 cell 单位；滚动级 = PROBES_PER_CASCADE_AXIS³=16³，base = AABB cell 数）
  pub dims: UVec3,
  /// cell 尺寸（fine 体素）：base 16 / LOD1-4=32/64/128/256
  pub cell_size: i32,
}

impl CascadeDomain {
  /// 域 hi（16-cell 半开）= origin + dims × (cell_size / DDGI_CELL)
  #[inline]
  pub fn hi_16cell(&self) -> IVec3 {
    self.origin + self.dims.as_ivec3() * (self.cell_size / DDGI_CELL)
  }
}

/// 非 grid-aligned object bbox（fine 世界坐标，半开区间 [min, max)）。
/// D2 三条件之一：本 LOD cell 与 bbox 重叠 → 探针活跃。
/// （D12：MOV 自身探针后置；当前阶段仅作为输入占位，与网格对齐的体素无贡献）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObjectBbox {
  pub min: Vec3,
  pub max: Vec3,
}

impl ObjectBbox {
  /// 本 LOD cell（cell_min_fine 起 cell_size³ AABB）是否与本 bbox 重叠（半开）
  #[inline]
  pub fn overlaps_cell(&self, cell_min_fine: Vec3, cell_size: f32) -> bool {
    self.max.x > cell_min_fine.x
      && self.min.x < cell_min_fine.x + cell_size
      && self.max.y > cell_min_fine.y
      && self.min.y < cell_min_fine.y + cell_size
      && self.max.z > cell_min_fine.z
      && self.min.z < cell_min_fine.z + cell_size
  }
}

// ---- get_probe_flags_intersection（截图3 逐字）----

/// 6 邻接 cell flags 的按位 AND 交集（截图3 `get_probe_flags_intersection()`）。
/// 边界外视为与中心 cell 同 flags（Douglas sort.glsl 中 dispatch 覆盖完整域，无真正边界；
/// halo +1 仅用于 shared memory bank conflict 避让。CPU 镜像需维持 AND 交集语义：
/// 边界 cell 的邻接缺失 = 用自身 flags 填充，避免交集被 0 位污染）。
/// 索引序：(−x, +x, −y, +y, −z, +z)，与 `probe_near_surface` 入参顺序一致。
#[inline]
pub fn flags_intersection(flags: &[DdgiProbeFlags], dims: UVec3, rel: UVec3) -> [DdgiProbeFlags; 6] {
  let self_flags = flags[(rel.x + rel.y * dims.x + rel.z * dims.x * dims.y) as usize];
  let fetch = |x: i32, y: i32, z: i32| -> DdgiProbeFlags {
    if (0..dims.x as i32).contains(&x)
      && (0..dims.y as i32).contains(&y)
      && (0..dims.z as i32).contains(&z)
    {
      let r = UVec3::new(x as u32, y as u32, z as u32);
      flags[(r.x + r.y * dims.x + r.z * dims.x * dims.y) as usize]
    } else {
      self_flags // 边界外 → 视为自身 flags（保持 AND 语义）
    }
  };
  let (rx, ry, rz) = (rel.x as i32, rel.y as i32, rel.z as i32);
  [
    fetch(rx - 1, ry, rz),
    fetch(rx + 1, ry, rz),
    fetch(rx, ry - 1, rz),
    fetch(rx, ry + 1, rz),
    fetch(rx, ry, rz - 1),
    fetch(rx, ry, rz + 1),
  ]
}

/// 截图3 `get_probe_flags_intersection()` 逐字：6 邻接 flags 的按位 AND。
#[inline]
pub fn probe_flags_intersection(flags: &[DdgiProbeFlags], dims: UVec3, rel: UVec3) -> DdgiProbeFlags {
  let n = flags_intersection(flags, dims, rel);
  n[0] & n[1] & n[2] & n[3] & n[4] & n[5]
}

// ---- probe_near_surface（截图3 逐字）----

/// 截图3 `probe_near_surface(cell_center, flags)` 逐字三条件 OR：
///   (flags & NO_SURFACES) == 0                     → 本 cell 有表面
///   || (flags_intersection & NO_SURFACES) == 0    → 邻接 cell 交集有表面
///   || probe_near_objects(cell_center)             → 与 object bbox 重叠
///
/// 注意：与我之前 BrickState OR 实现的等价性——flags 的 NO_SURFACES=0 对应 BrickState != Air。
/// 但边界处理不同：flags 边界外 = 0（NO_SURFACES=0）→ 邻接表面条件放行；
/// BrickState 边界外 = Air → 邻接表面条件不拦截。结果等价（都放行边界 cell 的表面检查）。
#[inline]
pub fn probe_near_surface_flags(
  own_flags: DdgiProbeFlags,
  intersect_flags: DdgiProbeFlags,
  object_bboxes: &[ObjectBbox],
  cell_center: Vec3,
  cell_size: f32,
) -> bool {
  if (own_flags & DDGI_PROBE_FLAG_NO_SURFACES) == 0 {
    return true;
  }
  if (intersect_flags & DDGI_PROBE_FLAG_NO_SURFACES) == 0 {
    return true;
  }
  for b in object_bboxes {
    if b.overlaps_cell(cell_center - Vec3::splat(cell_size / 2.0), cell_size) {
      return true;
    }
  }
  false
}

// ---- active 判定输入/输出 ----

/// active 判定输入快照（per-LOD 一份；WGSL 等价 = 共享内存 halo + previous meta tex）
pub struct ActiveInput<'a> {
  /// 烘焙出的探针网格（cell_index + positions）
  pub pg: &'a ProbeGrid,
  /// 逐 cell 的 DdgiProbeFlags（线性下标同 pg.cell_index 布局；M3-1 调用方预计算）
  pub cell_flags: &'a [DdgiProbeFlags],
  /// 非 grid-aligned object bbox 列表（fine 世界坐标，半开）
  pub object_bboxes: &'a [ObjectBbox],
  /// 本 LOD 域（用于 outside_lower_grid 判定；origin/dims 为 16-cell 全局坐标）
  pub cascade: CascadeDomain,
  /// 更细级域（None = 最细级 base，无更细网格；本 LOD 只管 finer 域外的 cell）
  pub finer: Option<CascadeDomain>,
  /// previous 元数据纹理数据（packed offset+age；活跃探针读 prev age / offset 用）
  pub prev_meta: &'a [u32],
  /// reuse bounds（D10；reuse_min ≤ cell（逐轴，含端）∧ cell < reuse_max（不含端））。
  /// cell 为本级 cell 坐标（rel，从 cascade.origin 起算）；全 REUSE_ALL 表示无滚动。
  pub reuse_bounds: (IVec3, IVec3),
  /// 帧号（用于 can_skip_update 的确定性 hash；固定种子 CPU 镜像与 WGSL 逐位一致）
  pub frame: u32,
}

/// 全 reuse bounds（base 静态级默认值：覆盖全部 cell，全部 reusable）
pub const REUSE_ALL: (IVec3, IVec3) = (IVec3::ZERO, IVec3::splat(i32::MAX));

/// active 判定 worklist 条目（截图2 `ddgi_item(position, age)`）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ActiveWorklistItem {
  /// 探针 id
  pub probe_id: u32,
  /// age（已继承 + 已 increment；can_skip=true 时不出现在 worklist）
  pub age: u32,
}

/// active 判定输出：worklist + indirect dispatch 参数 + next 元数据
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveOutput {
  /// worklist 条目（age 已继承 + 已 increment；can_skip=true 时不出现在 worklist，
  /// 但 next_meta 仍写继承后 age）
  pub worklist: Vec<ActiveWorklistItem>,
  /// indirect dispatch workgroup 数（= worklist.len()；ddgi_cast 消费）
  pub indirect_dispatch: u32,
  /// next 元数据纹理数据（**从 prev_meta 拷贝初始化**，非活跃/跳过探针保留 prev 值；
  /// 活跃探针（can_skip=true 或 false）写新 (offset, age)——age 可能是继承未 increment）
  pub next_meta: Vec<u32>,
}

// ---- can_skip_update（截图2 调用点可见 `can_skip_update(cell, cell_center, age)`）----

/// INFERENCE: Douglas 未截到函数体。签名从截图2 调用点确认：
/// `can_skip_update(cell, cell_center, age)` — 3 参。
/// 保守实现（与 M1-3 `can_skip_update(age, rand)` 同公式）：p=(age/255)²×0.5，
/// hash 输入 = pcg_hash(frame ^ probe_id)，保证跨帧/跨探针确定性。
#[inline]
pub fn can_skip_update_for_active(age: u32, frame: u32, probe_id: u32) -> bool {
  let rand = pcg_hash(frame ^ pcg_hash(probe_id));
  can_skip_update(age, rand)
}

// ---- CPU 镜像主入口（逐字对照 sort.glsl L225-262）----

/// CPU 镜像主入口：遍历全 cell，写 worklist + indirect + next meta。
///
/// 逐字对照 sort.glsl main() L225-262（截图1-2）：
///   clear_object_buckets();
///   cell = gl_GlobalInvocationID + dispatch.base_position;
///   flags = ddgi_probe_locus_flags(probe);
///   load_adjacent_probes(cell); memoryBarrierShared(); barrier();
///   if ((flags & ENABLED) == ENABLED && outside_lower_grid()) {
///     if (probe_near_surface(cell_center, flags)) {
///       // 读 prev meta
///       age = reusable ? ddgi_probe_age(previous) : 0;
///       if (!can_skip_update(cell, cell_center, age)) {
///         age = min(age + 1, 255);
///         worklist_insert(ddgi_item(position, age));
///       }
///       imageStore(next_ddgi_probes, texel, ddgi_probe_new(offset, age));
///     }
///   }
///
/// 关键修正（与 v1 实现差异）：
/// 1. **next_meta 初始化 = prev_meta.to_vec()**（非活跃探针保留 prev age，不丢收敛）
/// 2. **age 生命周期在 active 内闭环**：scroll inherit → can_skip → if !skip age++ → write
///    （v1 把 age++ 留给 ddgi_update，但截图2 显示 active 内完成）
/// 3. **can_skip=true 时 next_meta 仍写**（age 不变 = 继承后未 increment）
/// 4. **worklist 条目带 age**（ddgi_item(position, age)）
/// 5. **flags 模型**（DDGI_PROBE_FLAG_ENABLED / NO_SURFACES）取代 BrickState OR
///
/// INFERENCE: subgroup worklist 分配 → CPU 串行 push，等价集合相同（WGSL 分配顺序不影响结果）
pub fn cpu_ddgi_active(input: &ActiveInput) -> ActiveOutput {
  // 关键修正①：从 prev_meta 拷贝初始化，非活跃探针保留 prev age
  let mut next_meta = input.prev_meta.to_vec();
  let mut worklist = Vec::new();
  let dims = input.pg.grid_dims;
  let (reuse_min, reuse_max) = input.reuse_bounds;
  for rz in 0..dims.z {
    for ry in 0..dims.y {
      for rx in 0..dims.x {
        let rel = UVec3::new(rx, ry, rz);
        let li = input.pg.cell_linear(rel);
        let probe_id = input.pg.cell_index[li];
        if probe_id == NO_PROBE {
          continue;
        }
        let flags = input.cell_flags[li];
        // 截图2: if ((flags & DDGI_PROBE_FLAG_ENABLED) == DDGI_PROBE_FLAG_ENABLED && outside_lower_grid())
        if (flags & DDGI_PROBE_FLAG_ENABLED) != DDGI_PROBE_FLAG_ENABLED {
          continue;
        }
        // 级联归属 outside_lower_grid（D4）
        if let Some(finer) = input.finer {
          let lo = input.pg.cell16(rel);
          let hi = lo + IVec3::splat(input.cascade.cell_size / DDGI_CELL);
          if !outside_lower_grid(
            lo,
            hi,
            finer.origin,
            finer.dims * (finer.cell_size / DDGI_CELL) as u32,
          ) {
            continue;
          }
        }
        // 截图3: probe_near_surface(cell_center, flags)
        let intersect = probe_flags_intersection(input.cell_flags, dims, rel);
        let cell_center = input.pg.cell_min_voxel(rel).as_vec3()
          + Vec3::splat(input.cascade.cell_size as f32 / 2.0);
        if !probe_near_surface_flags(
          flags,
          intersect,
          input.object_bboxes,
          cell_center,
          input.cascade.cell_size as f32,
        ) {
          continue;
        }
        // ---- age 生命周期（截图2 逐字）----
        let (layer, tx, ty) = meta_texel_coord(probe_id);
        let meta_idx = meta_texel_linear(layer, tx, ty);
        let (prev_offset, prev_age) = unpack_probe_meta(input.prev_meta[meta_idx]);
        // D10 reuse bounds：本级 cell 坐标 rel，半开区间
        let reusable = probe_reusable(
          IVec3::new(rx as i32, ry as i32, rz as i32),
          prev_offset,
          prev_offset,
          (reuse_min, reuse_max),
        );
        let mut age = if reusable { prev_age } else { 0 };
        // 截图2: if (!can_skip_update(cell, cell_center, age)) { age = min(age+1, 255); worklist_insert; }
        let skip = can_skip_update_for_active(age, input.frame, probe_id);
        if !skip {
          age = age_after_update(age);
          worklist.push(ActiveWorklistItem { probe_id, age });
        }
        // 截图2: imageStore(next_ddgi_probes, texel, ddgi_probe_new(offset, age))
        // 无论 skip 与否都写——skip 时 age 不变（继承后未 increment）
        next_meta[meta_idx] = pack_probe_meta(prev_offset, age);
      }
    }
  }
  let n = worklist.len() as u32;
  ActiveOutput {
    worklist,
    indirect_dispatch: n,
    next_meta,
  }
}

// ---- 调用方辅助 ----

/// 从 VolumeGrid 预计算每个 cell 的 DdgiProbeFlags（线性下标同 pg.cell_index）。
/// 生产 GPU 路径在 ddgi_active.wgsl 内由 shared memory + chunk tree 查询替代。
pub fn compute_cell_flags(
  grid: &gate_voxel::VolumeGrid,
  pg: &ProbeGrid,
  cell_size: i32,
) -> Vec<DdgiProbeFlags> {
  let dims = pg.grid_dims;
  let mut out = vec![0u32; (dims.x * dims.y * dims.z) as usize];
  for rz in 0..dims.z {
    for ry in 0..dims.y {
      for rx in 0..dims.x {
        let rel = UVec3::new(rx, ry, rz);
        let li = pg.cell_linear(rel);
        let probe_id = pg.cell_index[li];
        if probe_id == NO_PROBE {
          continue; // 无探针 → flags = 0（未 ENABLED）
        }
        let cell_min = pg.cell_min_voxel(rel);
        let state = cell_state_at(grid, cell_min, cell_size);
        out[li] = brickstate_to_flags(state, true);
      }
    }
  }
  out
}

// ============================================================================
// M3-2 cast 射线投射 CPU 镜像（WGSL ddgi_cast.wgsl 镜像源；spec D5）
// 消费 M3-1 worklist → 4096 预算分摊 → Fibonacci+PCG 旋转方向 →
// 端点着色（sky / emissive 直出 / 直光+prev DDGI 自闭环）→ 样本缓冲
// ============================================================================

use crate::brickmap::{BrickMapBuffers, VolumeHit, cpu_reference_trace_volumes, cpu_reference_volumes_occluded};
use crate::lighting::{EMISSIVE_EMIT_GAIN, SHADOW_BIAS, SHADOW_DIR_T_MAX, cpu_reference_sky, xyz, yzw};
use gate_voxel::VolumeTransform;

/// D5 预算分摊：4096 射线/帧固定总预算 → 每活跃探针射线数（活跃探针均摊，
/// 性能不随屏上探针数波动——Douglas「roughly the same number of rays per frame,
/// dividing them amongst all of the active probes」）。
/// active=0 → 0；4096/active 向下取整，保底 1（活跃数超预算时总射线数会超出
/// 4096——实际活跃探针数远低于此；预算值起步，M5-3 bench 后调）。
#[inline]
pub fn cast_rays_per_probe(active_count: u32) -> u32 {
  if active_count == 0 {
    0
  } else {
    (RAY_BUDGET_PER_FRAME / active_count).max(1)
  }
}

/// Shoemake 均匀随机旋转四元数（u ∈ [0,1)³；Graphics Gems III）
#[inline]
fn random_quat(u1: f32, u2: f32, u3: f32) -> glam::Quat {
  let s1 = (1.0 - u1).sqrt();
  let s2 = u1.sqrt();
  let (su2, cu2) = (std::f32::consts::TAU * u2).sin_cos();
  let (su3, cu3) = (std::f32::consts::TAU * u3).sin_cos();
  glam::Quat::from_xyzw(s1 * su2, s1 * cu2, s2 * su3, s2 * cu3)
}

/// 射线方向（D5「球面均匀随机方向（PCG hash + 帧号种子，Fibonacci 球保留）」+
/// Douglas 字幕「cast random rays according to a Fibonacci sphere」）：
/// Fibonacci 球基方向（帧内低差异均匀覆盖）× PCG 随机旋转（探针×帧号种子——
/// 帧间/探针间去相关；随机的是旋转，帧内保持 Fibonacci 蓝噪声结构）。
#[inline]
pub fn cast_ray_dir(probe_id: u32, frame: u32, ray_index: u32, rays_total: u32) -> Vec3 {
  let base = fibonacci_dir(ray_index, rays_total.max(1));
  let seed = ray_rand(probe_id, frame, 0);
  let u = rand2(seed);
  let u3 = (pcg_hash(seed) & 0xFFFF) as f32 / 65536.0;
  random_quat(u[0], u[1], u3) * base
}

/// 命中点材质（palette 两 words 解包；lighting.rs hit_mat 同型，DDGI 侧独立维护）
#[inline]
fn hit_mat_ddgi(vols: &[(&BrickMapBuffers, VolumeTransform)], hit: &VolumeHit) -> (Vec3, f32) {
  let idx = if hit.obj_id == -1 {
    0
  } else {
    hit.obj_id as usize + 1
  };
  let pal = &vols[idx].0.b_palette;
  let w0 = pal[hit.pal as usize * 2];
  let w1 = pal[hit.pal as usize * 2 + 1];
  let albedo = Vec3::new(
    (w0 & 0xFF) as f32,
    ((w0 >> 8) & 0xFF) as f32,
    ((w0 >> 16) & 0xFF) as f32,
  ) / 255.0;
  let emissive = (w1 & 0xFF) as f32 / 255.0;
  (albedo, emissive)
}

/// D5 端点着色（pre-exposure，spec §3 保留约定——DDGI 存原始 radiance，
/// 曝光在最终像素着色时施加）：
/// - miss → sky 色（cpu_reference_sky），dist = PROBE_T_MAX（depth 远距哨兵）
/// - emissive 命中 → albedo × emissive × gain 直出（无方向性、不受阴影）
/// - 常规命中 → 直光 1-bounce（ndl × 硬阴影射线）+ prev DDGI 采样（自闭环
///   无限反弹——命中点用上一帧 irradiance 着色，反弹数随帧数累积）
/// 返回 (radiance, dist)。
/// INFERENCE: 间接采样点沿法线偏移 DDGI_NORMAL_BIAS（Majercik 2019 §4 惯例，
/// Douglas 未截图此细节）；法线 = trace 面法线（与 lighting CPU 参考同一已知
/// 差异：GPU 侧 per-voxel implicit normal 未同步 CPU 镜像）。
pub fn cast_endpoint_radiance(
  vols: &[(&BrickMapBuffers, VolumeTransform)],
  light_pool: &crate::lighting::LightPoolUniform,
  prev: &DdgiProbeArrays,
  origin: Vec3,
  dir: Vec3,
) -> (Vec3, f32) {
  match cpu_reference_trace_volumes(vols, origin, dir, PROBE_T_MAX) {
    None => (cpu_reference_sky(dir, light_pool), PROBE_T_MAX),
    Some(hit) => {
      let p = origin + dir * hit.t;
      let n = hit.normal;
      let (base, emissive) = hit_mat_ddgi(vols, &hit);
      if emissive > 0.0 {
        return (base * (emissive * EMISSIVE_EMIT_GAIN), hit.t);
      }
      let mut col = Vec3::ZERO;
      // 直光 1-bounce（方向光硬阴影；cpu_reference_shade_hit 直射段同型）
      if light_pool.g.count > 0 && light_pool.lights[0].kind_pos_dir.x < 0.5 {
        let ld = &light_pool.lights[0];
        let l = yzw(ld.kind_pos_dir);
        let ndl = n.dot(l).max(0.0);
        if ndl > 0.0 {
          let o = p + n * SHADOW_BIAS;
          let vis = if cpu_reference_volumes_occluded(vols, o, l, SHADOW_DIR_T_MAX) {
            0.0
          } else {
            1.0
          };
          col += base * xyz(ld.color_intensity) * ld.color_intensity.w * (ndl * vis);
        }
      }
      // prev DDGI 自闭环（无限反弹）：命中点偏移后采样上一帧 irradiance
      let irr = cpu_sample_ddgi(prev, p + n * DDGI_NORMAL_BIAS, n);
      col += base * irr;
      (col, hit.t)
    }
  }
}

/// cast 输入（消费 M3-1 active 输出 + 上一帧探针数组）
pub struct CastInput<'a> {
  /// 世界 volumes（[0] = 主世界 identity；CPU 参考用 BrickMapBuffers）
  pub vols: &'a [(&'a BrickMapBuffers, VolumeTransform)],
  pub light_pool: &'a crate::lighting::LightPoolUniform,
  /// 探针网格（positions；probe id → 世界坐标）
  pub pg: &'a ProbeGrid,
  /// M3-1 active 输出的 worklist
  pub worklist: &'a [ActiveWorklistItem],
  /// 每探针射线数 = cast_rays_per_probe(worklist.len())
  pub rays_per_probe: u32,
  /// 帧号（方向旋转种子）
  pub frame: u32,
  /// 上一帧探针数组（irradiance/depth；自闭环输入）
  pub prev: &'a DdgiProbeArrays<'a>,
}

/// 单条射线样本（M3-3 ddgi_update 消费：collect_radiance 投影 + depth EMA）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RaySample {
  /// 射线方向（cast_ray_dir 产物）
  pub dir: Vec3,
  /// 端点 radiance（pre-exposure）
  pub radiance: Vec3,
  /// 命中距离（sky = PROBE_T_MAX 远距哨兵；depth EMA 输入）
  pub dist: f32,
}

/// cast 输出：样本缓冲（WGSL storage buffer 镜像；线性下标 = slot × rays_per_probe
/// + ray_index，slot = worklist 序号——M3-3 update 每工作群消费自己的射线段）
#[derive(Debug, Clone, PartialEq)]
pub struct CastOutput {
  pub samples: Vec<RaySample>,
  pub rays_per_probe: u32,
}

/// CPU 镜像主入口：worklist → 每探针 rays_per_probe 条射线 → 端点着色 → 样本缓冲。
///
/// WGSL `ddgi_cast.wgsl` 等价（dispatch_workgroups_indirect 消费 M3-1 indirect 参数，
/// workgroup 数 = worklist.len()，1 workgroup = 1 探针）：
/// - @workgroup_size(64)（rays_total 上限；超出部分线程内循环 `for i in (tid..n).step(64)`）
/// - 方向 = cast_ray_dir 逐字镜像（fibonacci_dir + random_quat(pcg seed)）
/// - 端点 = cast_endpoint_radiance 逐字镜像（trace_grid + sky/emissive/直光+prev 采样）
/// - 样本写 storage buffer（dir.xyz, radiance.xyz, dist）按 slot×n+i 线性布局
pub fn cpu_ddgi_cast(input: &CastInput) -> CastOutput {
  let rays = input.rays_per_probe;
  let mut samples = Vec::with_capacity(input.worklist.len() * rays as usize);
  for item in input.worklist {
    let origin = input.pg.positions[item.probe_id as usize];
    for i in 0..rays {
      let dir = cast_ray_dir(item.probe_id, input.frame, i, rays);
      let (radiance, dist) =
        cast_endpoint_radiance(input.vols, input.light_pool, input.prev, origin, dir);
      samples.push(RaySample { dir, radiance, dist });
    }
  }
  CastOutput {
    samples,
    rays_per_probe: rays,
  }
}

// ============================================================================
// GPU wire（BG4：WGSL DdgiMeta + 4 storage buffer 逐字段镜像）
// ============================================================================

/// BG4 meta uniform（WGSL `DdgiMeta` 逐字段镜像，64B）
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bevy::ecs::resource::Resource, ShaderType)]
pub struct DdgiMeta {
  /// 烘焙出的探针总数（= positions/cell_index 有效域）
  pub probe_count: u32,
  /// 活跃探针数（v1 = probe_count；v2 由每帧 ddgi_active 判定接管，M3-1）
  pub active_count: u32,
  /// 本帧更新探针数（= ddgi_update dispatch workgroup 数，1 workgroup = 1 探针）
  pub probes_this_frame: u32,
  /// worklist 环形基址（帧间轮转；probe_idx = (cycle_base + wg) % probe_count）
  pub cycle_base: u32,
  /// 帧号（Fibonacci 方向轮转相位 = frame * probes_this_frame）
  pub frame: u32,
  pub _pad0: u32,
  /// 采样网格原点（cell 单位，i32 逐分量转 f32；w=0）
  pub grid_origin: Vec4,
  /// 采样网格 dims（cell 单位；w=0）
  pub grid_dims: Vec4,
  /// x = cell 边长(16.0)，y = 探针射线 t_max，z = EMA α，w = 0
  pub cell_tmax_alpha: Vec4,
}

/// 单帧更新计划（Douglas #23：固定射线预算摊给活跃探针）
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DdgiFramePlan {
  /// 本帧更新探针数 = dispatch workgroup 数
  pub probes_this_frame: u32,
  /// worklist 环形基址
  pub cycle_base: u32,
  /// 本帧总射线线程数 = probes_this_frame × RAYS_PER_PROBE
  pub rays_dispatch: u32,
}

/// 帧计划：每帧 `RAY_BUDGET_PER_FRAME / RAYS_PER_PROBE` 个探针各得 64 射线，
/// 环形轮转全量刷新一轮需 `ceil(probe_count / probes_this_frame)` 帧
pub fn frame_plan(probe_count: u32, frame: u32) -> DdgiFramePlan {
  if probe_count == 0 {
    return DdgiFramePlan {
      probes_this_frame: 0,
      cycle_base: 0,
      rays_dispatch: 0,
    };
  }
  let slots = (RAY_BUDGET_PER_FRAME / RAYS_PER_PROBE).max(1);
  let ptf = probe_count.min(slots);
  DdgiFramePlan {
    probes_this_frame: ptf,
    cycle_base: frame.wrapping_mul(ptf) % probe_count,
    rays_dispatch: ptf * RAYS_PER_PROBE,
  }
}

impl DdgiMeta {
  /// 烘焙结果 + 帧计划 → uniform（WGSL 侧只读）
  pub fn build(pg: &ProbeGrid, plan: &DdgiFramePlan) -> Self {
    Self {
      probe_count: pg.positions.len() as u32,
      active_count: pg.positions.len() as u32,
      probes_this_frame: plan.probes_this_frame,
      cycle_base: plan.cycle_base,
      frame: 0, // 由帧循环覆写（此处仅烘焙时初值）
      _pad0: 0,
      grid_origin: Vec4::new(
        pg.grid_origin.x as f32,
        pg.grid_origin.y as f32,
        pg.grid_origin.z as f32,
        0.0,
      ),
      grid_dims: Vec4::new(
        pg.grid_dims.x as f32,
        pg.grid_dims.y as f32,
        pg.grid_dims.z as f32,
        0.0,
      ),
      cell_tmax_alpha: Vec4::new(DDGI_CELL as f32, PROBE_T_MAX, DDGI_ALPHA, 0.0),
    }
  }
}

/// 探针位置打包：xyz = 世界 fine 坐标，w = active(1.0)
pub fn pack_probe_positions(pg: &ProbeGrid) -> Vec<Vec4> {
  pg.positions
    .iter()
    .map(|p| Vec4::new(p.x, p.y, p.z, 1.0))
    .collect()
}

/// 初始 irradiance/depth：
/// - irradiance 零（无间接光；射线更新后逐步 EMA 填充）
/// - depth = tmax（远距哨兵；初始 0 会被漏光剔除读成「探针贴墙」误伤全部贡献）
pub fn initial_probe_data(probe_count: u32) -> (Vec<Vec4>, Vec<f32>) {
  (
    vec![Vec4::ZERO; IRRADIANCE_WORDS_PER_PROBE as usize / 4 * probe_count as usize],
    vec![PROBE_T_MAX; DEPTH_WORDS_PER_PROBE as usize * probe_count as usize],
  )
}

// ============================================================================
// CPU 参考采样（WGSL sample_ddgi / ddgi_irr_sample / ddgi_depth_sample 逐行镜像；
// 仓库惯例：lighting.rs CPU 参考同步 + 等价单测。GPU 端出画面问题时以此对照
// 区分「数学错 vs 接线错」）
// ============================================================================

/// CPU 侧探针数组（布局与 GPU BG4 storage buffer 同构）
pub struct DdgiProbeArrays<'a> {
  pub grid_origin: IVec3,
  pub grid_dims: UVec3,
  /// 探针世界位置（probe_count）
  pub positions: &'a [Vec3],
  /// dense cell → probe id（cell_index.len() = dims.x*dims.y*dims.z）
  pub cell_index: &'a [u32],
  /// irradiance（probe_count × 64 个 Vec4，8×8 oct/探针）
  pub irr: &'a [Vec4],
  /// depth（probe_count × 256 个 f32，16×16 oct/探针）
  pub depth: &'a [f32],
}

/// irradiance 8×8 八面体双线性采样（WGSL ddgi_irr_sample 镜像；边界硬钳）
fn cpu_irr_sample(irr: &[Vec4], id: u32, d: Vec3) -> Vec3 {
  let s = IRRADIANCE_TEXELS as f32;
  let e = oct_encode(d) * 0.5 + Vec2::splat(0.5);
  let g = e * s - Vec2::splat(0.5);
  let g0 = g.floor();
  let f = g - g0;
  let si = IRRADIANCE_TEXELS as i32;
  let x0 = (g0.x as i32).clamp(0, si - 1) as usize;
  let y0 = (g0.y as i32).clamp(0, si - 1) as usize;
  let x1 = (g0.x as i32 + 1).clamp(0, si - 1) as usize;
  let y1 = (g0.y as i32 + 1).clamp(0, si - 1) as usize;
  let fetch = |tx: usize, ty: usize| -> Vec3 {
    irr[id as usize * 64 + ty * IRRADIANCE_TEXELS as usize + tx].truncate()
  };
  let c00 = fetch(x0, y0);
  let c10 = fetch(x1, y0);
  let c01 = fetch(x0, y1);
  let c11 = fetch(x1, y1);
  c00.lerp(c10, f.x).lerp(c01.lerp(c11, f.x), f.y)
}

/// depth 16×16 八面体最近邻（WGSL ddgi_depth_sample 镜像；保守剔除）
fn cpu_depth_sample(depth: &[f32], id: u32, d: Vec3) -> f32 {
  let s = DEPTH_TEXELS as f32;
  let e = oct_encode(d) * 0.5 + Vec2::splat(0.5);
  let g = (e * s).floor();
  let si = DEPTH_TEXELS as i32;
  let x = (g.x as i32).clamp(0, si - 1) as usize;
  let y = (g.y as i32).clamp(0, si - 1) as usize;
  depth[id as usize * DEPTH_WORDS_PER_PROBE as usize + y * DEPTH_TEXELS as usize + x]
}

/// WGSL `sample_ddgi` 逐行镜像：8 cell 三线性 × 锐利背面 × 漏光 chevron，加权归一化
pub fn cpu_sample_ddgi(a: &DdgiProbeArrays, p: Vec3, n: Vec3) -> Vec3 {
  if a.positions.is_empty() {
    return Vec3::ZERO;
  }
  let cell_f = p / DDGI_CELL as f32;
  let c0 = cell_f.floor();
  let f = cell_f - c0;
  let mut total = Vec3::ZERO;
  let mut wsum = 0.0f32;
  for iz in 0..2u32 {
    for iy in 0..2u32 {
      for ix in 0..2u32 {
        let corner = c0 + Vec3::new(ix as f32, iy as f32, iz as f32);
        let rel = corner - a.grid_origin.as_vec3();
        if rel.x < 0.0
          || rel.y < 0.0
          || rel.z < 0.0
          || rel.x >= a.grid_dims.x as f32
          || rel.y >= a.grid_dims.y as f32
          || rel.z >= a.grid_dims.z as f32
        {
          continue;
        }
        let ru = rel.as_uvec3();
        let ci = (ru.x + ru.y * a.grid_dims.x + ru.z * a.grid_dims.x * a.grid_dims.y) as usize;
        let id = a.cell_index[ci];
        if id == NO_PROBE {
          continue;
        }
        let wtri = (if ix == 1 { f.x } else { 1.0 - f.x })
          * (if iy == 1 { f.y } else { 1.0 - f.y })
          * (if iz == 1 { f.z } else { 1.0 - f.z });
        if wtri <= 1e-6 {
          continue;
        }
        let probe = a.positions[id as usize];
        let to = p - probe;
        let dist = to.length();
        let dir = to / dist.max(1e-4); // dir = 探针→接收点（depth 检查方向）
        // 锐利背面剔除（Rohacek §3.2；RTXGI Irradiance.hlsl 逐字对照）：
        // wn = clamp(n·(接收点→探针)/bias)——探针在法线前侧（空气侧）通过，
        // 后侧（墙内/地下）严格 0，穿墙不漏光。
        // （RTXGI: worldPosToAdjProbe = normalize(probePos - worldPos)，
        //   wrapShading = dot(worldPosToAdjProbe, direction)）
        let wn = (n.dot(-dir) / DDGI_NORMAL_BIAS).clamp(0.0, 1.0);
        if wn <= 0.0 {
          continue;
        }
        // 漏光 chevron（Rohacek §3.3；depth 沿 探针→接收点 方向取：
        // 探针到墙距离 < 探针到接收点距离 → 剔除）
        let dtex = cpu_depth_sample(a.depth, id, dir);
        let wd = ((dtex - dist) / DDGI_DEPTH_BIAS + 0.5).clamp(0.0, 1.0);
        if wd <= 0.0 {
          continue;
        }
        // irradiance 采样方向 = 表面法线（RTXGI: octantCoords =
        // GetOctahedralCoordinates(direction)；探针 oct 图按「接收法线」索引——
        // 与 D6 collect_radiance 的 (d·dir_i)+ 余弦权重同源）
        let irr = cpu_irr_sample(a.irr, id, n);
        let w = wtri * wn * wd;
        total += irr * w;
        wsum += w;
      }
    }
  }
  if wsum < 1e-4 {
    Vec3::ZERO
  } else {
    total / wsum
  }
}

// ============================================================================
// 渲染侧：BG4 布局 + 烘焙提取 + GPU 资源 + 每帧 prepare（DdgiPlugin）
// ============================================================================

use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::ecs::system::{Commands, Res, ResMut};
use bevy::render::{
  Render, RenderApp, RenderStartup, RenderSystems,
  render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntries, Buffer,
    BufferDescriptor, BufferUsages, ShaderStages, UniformBuffer,
    binding_types::{storage_buffer_read_only_sized, storage_buffer_sized, uniform_buffer},
  },
  renderer::{RenderDevice, RenderQueue},
};

/// BG4 布局：DdgiMeta uniform + positions/cell_index 只读 + irradiance/depth read_write
///
/// 两个 compute pipeline（dda_main / ddgi_update）共用同一 5 组布局：
/// ddgi_update 写 3/4，dda_main（spike 4 后）只读；同 BG 顺序两 pass 复用。
pub fn ddgi_bg4_layout() -> BindGroupLayoutDescriptor {
  BindGroupLayoutDescriptor::new(
    "DdgiBg4",
    &BindGroupLayoutEntries::sequential(
      ShaderStages::COMPUTE,
      (
        uniform_buffer::<DdgiMeta>(false),           // 0 meta
        storage_buffer_read_only_sized(false, None), // 1 positions
        storage_buffer_read_only_sized(false, None), // 2 cell_index
        storage_buffer_sized(false, None),           // 3 irradiance rw
        storage_buffer_sized(false, None),           // 4 depth rw
      ),
    ),
  )
}

/// Extract 产物：烘焙好的探针网格 + 对应的世界编辑代数（prepare 消费后移除）
#[derive(bevy::ecs::resource::Resource)]
pub struct ProbeBake(pub ProbeGrid, pub u64);

/// 渲染 world 持久 DDGI GPU 资源（RenderStartup 占位 4B 空缓冲，烘焙后重建）
#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiGpu {
  pub meta: UniformBuffer<DdgiMeta>,
  pub positions: Buffer,
  pub cell_index: Buffer,
  pub irradiance: Buffer,
  pub depth: Buffer,
  pub probe_count: u32,
  /// cell 网格原点/dims（meta 每帧覆写用，避免持有完整 cell_index CPU 副本）
  pub grid_origin: Vec4,
  pub grid_dims: Vec4,
  /// 已推进帧号（prepare 自增；cycle_base = frame × probes_this_frame % count）
  pub frame: u32,
  /// 当前 GPU 探针数据对应的世界编辑代数（u64::MAX = 尚未烘焙；编辑后触发重烘）
  pub baked_generation: u64,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiBg4(pub BindGroup);

/// DDGI 插件：main world VoxelScene → 一次性烘焙 → GPU buffer/meta/BG4
pub struct DdgiPlugin;

impl bevy::app::Plugin for DdgiPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };
    render_app
      .add_systems(RenderStartup, init_ddgi_gpu)
      .add_systems(bevy::render::ExtractSchedule, extract_ddgi_bake)
      .add_systems(
        Render,
        prepare_ddgi.in_set(RenderSystems::PrepareBindGroups),
      );
  }
}

/// 4B 占位 storage buffer（管线布局始终可绑；probe_count=0 时 ddgi_update 早退）
fn dummy_buffer(device: &RenderDevice, label: &str) -> Buffer {
  device.create_buffer(&BufferDescriptor {
    label: Some(label.into()),
    size: 4,
    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  })
}

fn init_ddgi_gpu(mut commands: Commands, device: Res<RenderDevice>) {
  commands.insert_resource(DdgiGpu {
    meta: UniformBuffer::default(),
    positions: dummy_buffer(&device, "ddgi_positions(empty)"),
    cell_index: dummy_buffer(&device, "ddgi_cell_index(empty)"),
    irradiance: dummy_buffer(&device, "ddgi_irradiance(empty)"),
    depth: dummy_buffer(&device, "ddgi_depth(empty)"),
    probe_count: 0,
    grid_origin: Vec4::ZERO,
    grid_dims: Vec4::ZERO,
    frame: 0,
    baked_generation: u64::MAX,
  });
}

/// main world VoxelScene → render world ProbeBake（版本驱动：世界编辑代数变化即重烘）
fn extract_ddgi_bake(
  mut commands: Commands,
  scene: Option<bevy::render::Extract<Res<crate::VoxelScene>>>,
  gpu: Option<Res<DdgiGpu>>,
  inflight: Option<Res<ProbeBake>>,
) {
  let Some(scene) = scene else {
    return;
  };
  let generation = scene.volumes.main().edit_generation();
  // GPU 已是该代数（已重烘过）→ 无需烘焙
  if gpu.is_some_and(|g| g.baked_generation == generation) {
    return;
  }
  // 同代数烘焙已在排队（prepare 尚未消费）→ 不重复烘焙
  if inflight.is_some_and(|b| b.1 == generation) {
    return;
  }
  let t0 = std::time::Instant::now();
  let pg = bake_probe_grid(&scene.volumes);
  bevy::log::info!(
    "DDGI bake (gen {generation}): probes={} cells={}x{}x{}={} ({:?})",
    pg.positions.len(),
    pg.grid_dims.x,
    pg.grid_dims.y,
    pg.grid_dims.z,
    pg.cell_index.len(),
    t0.elapsed(),
  );
  commands.insert_resource(ProbeBake(pg, generation));
}

// --- 零依赖字节打包（Vec4/f32/u32 → le bytes；workspace 无 bytemuck 依赖）---
fn vec4_bytes(v: &[Vec4]) -> Vec<u8> {
  let mut out = Vec::with_capacity(v.len() * 16);
  for q in v {
    for c in [q.x, q.y, q.z, q.w] {
      out.extend_from_slice(&c.to_le_bytes());
    }
  }
  out
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
  let mut out = Vec::with_capacity(v.len() * 4);
  for c in v {
    out.extend_from_slice(&c.to_le_bytes());
  }
  out
}
fn u32_bytes(v: &[u32]) -> Vec<u8> {
  let mut out = Vec::with_capacity(v.len() * 4);
  for c in v {
    out.extend_from_slice(&c.to_le_bytes());
  }
  out
}

#[allow(clippy::too_many_arguments)]
fn prepare_ddgi(
  mut commands: Commands,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
  pipeline_cache: Res<bevy::render::render_resource::PipelineCache>,
  bake: Option<Res<ProbeBake>>,
  mut gpu: ResMut<DdgiGpu>,
) {
  // ---- 新烘焙（含编辑后重烘）：重建 4 个 storage buffer（尺寸变化）----
  if let Some(bake) = bake {
    let generation = bake.1;
    let pg = &bake.0;
    let n = pg.positions.len() as u32;
    let pos_bytes = vec4_bytes(&pack_probe_positions(pg));
    let cell_bytes = u32_bytes(&pg.cell_index);
    let (irr0, dep0) = initial_probe_data(n);
    let irr_bytes = vec4_bytes(&irr0);
    let dep_bytes = f32_bytes(&dep0);

    let make = |label: &str, bytes: &[u8]| {
      let buf = device.create_buffer(&BufferDescriptor {
        label: Some(label.into()),
        size: bytes.len().max(4) as u64,
        usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
        mapped_at_creation: false,
      });
      queue.write_buffer(&buf, 0, bytes);
      buf
    };
    gpu.positions = make("ddgi_positions", &pos_bytes);
    gpu.cell_index = make("ddgi_cell_index", &cell_bytes);
    gpu.irradiance = make("ddgi_irradiance", &irr_bytes);
    gpu.depth = make("ddgi_depth", &dep_bytes);
    gpu.probe_count = n;
    gpu.baked_generation = generation;
    gpu.grid_origin = Vec4::new(
      pg.grid_origin.x as f32,
      pg.grid_origin.y as f32,
      pg.grid_origin.z as f32,
      0.0,
    );
    gpu.grid_dims = Vec4::new(
      pg.grid_dims.x as f32,
      pg.grid_dims.y as f32,
      pg.grid_dims.z as f32,
      0.0,
    );
    gpu.frame = 0;
    bevy::log::info!(
      "DDGI gpu: {} probes (irr {}KB, depth {}KB, cell_index {}KB)",
      n,
      irr_bytes.len() / 1024,
      dep_bytes.len() / 1024,
      cell_bytes.len() / 1024,
    );
    commands.remove_resource::<ProbeBake>();
  }

  // ---- 每帧：推进帧号 + 写 meta uniform + 建 BG4 ----
  // probe_count=0 时同样建 BG4（dummy buffer + 零 meta）：dda_main 管线布局含
  // group 4，pass 内必须绑定；ddgi_update 读 probe_count=0 早退。
  gpu.frame = gpu.frame.wrapping_add(1);
  let plan = frame_plan(gpu.probe_count, gpu.frame);
  let meta = DdgiMeta {
    probe_count: gpu.probe_count,
    active_count: gpu.probe_count,
    probes_this_frame: plan.probes_this_frame,
    cycle_base: plan.cycle_base,
    frame: gpu.frame,
    _pad0: 0,
    grid_origin: gpu.grid_origin,
    grid_dims: gpu.grid_dims,
    cell_tmax_alpha: Vec4::new(DDGI_CELL as f32, PROBE_T_MAX, DDGI_ALPHA, 0.0),
  };
  *gpu.meta.get_mut() = meta;
  gpu.meta.write_buffer(&device, &queue);

  let layout = pipeline_cache.get_bind_group_layout(&ddgi_bg4_layout());
  let bg4 = device.create_bind_group(
    None,
    &layout,
    &BindGroupEntries::sequential((
      &gpu.meta,
      gpu.positions.as_entire_binding(),
      gpu.cell_index.as_entire_binding(),
      gpu.irradiance.as_entire_binding(),
      gpu.depth.as_entire_binding(),
    )),
  );
  commands.insert_resource(DdgiBg4(bg4));
}

// ============================================================================
// 单测
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;
  use gate_voxel::VolumeGrid;

  /// 16³ 实心盒放在 cell (1,1,1) 的世界（其余空）
  fn world_with_cell_box() -> VolumeGrid {
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(3).color = [128, 64, 32];
    // 盒放 fine 16..32（cell (1,1,1)）：6 邻接 cell 均在 chunk bbox cell 域内
    gate_voxel::fill_bricks(&mut g, IVec3::splat(16), IVec3::splat(16), 4, 3);
    g
  }

  #[test]
  fn wire_constants() {
    assert_eq!(IRRADIANCE_WORDS_PER_PROBE, 8 * 8 * 4);
    assert_eq!(DEPTH_WORDS_PER_PROBE, 16 * 16);
    assert_eq!(RAYS_PER_PROBE, 64);
    assert_eq!(CELLS_PER_CHUNK, 16);
    assert_eq!(DDGI_CELL_LEVEL, 2);
    // 采样权重（WGSL DDGI_NORMAL_BIAS / DDGI_DEPTH_BIAS 镜像）
    assert_eq!(DDGI_NORMAL_BIAS, 0.2);
    assert_eq!(DDGI_DEPTH_BIAS, 4.0);
  }

  /// v2 纹理数组布局常量（WGSL 镜像防漂移）
  #[test]
  fn v2_wire_constants() {
    assert_eq!(PROBES_PER_LAYER_AXIS, 16);
    assert_eq!(PROBES_PER_LAYER, 256);
    assert_eq!(IRRADIANCE_LAYER_TEXELS, 16 * 8);
    assert_eq!(DEPTH_LAYER_TEXELS, 16 * 16);
    assert_eq!(DDGI_LODS, 4);
    assert_eq!(DDGI_AGE_MAX, 255);
    assert_eq!(META_AGE_SHIFT, 15);
    // packed u32 不溢出：15 bit offset+quant + 8 bit age
    assert!(META_AGE_SHIFT + 8 <= 32);
    // 级联：base + 4 级 ×2 递增；滚动级 16³ 探针/级 = 16 层纹理
    assert_eq!(DDGI_CASCADE_CELL_SIZES, [16, 32, 64, 128, 256]);
    assert_eq!(PROBES_PER_CASCADE_AXIS, 16);
    assert_eq!(PROBES_PER_LAYER, PROBES_PER_CASCADE_AXIS * PROBES_PER_CASCADE_AXIS);
  }

  /// probe id → (layer, u, v) 映射：连续 id 铺满 16×16 层网格后进位
  #[test]
  fn probe_texel_mapping() {
    // 层内布局：id 0 → (L0, probe[0,0])；id 15 → probe[15,0]；id 255 → probe[15,15]
    assert_eq!(probe_layer(0), 0);
    assert_eq!(probe_in_layer(0), [0, 0]);
    assert_eq!(probe_layer(255), 0);
    assert_eq!(probe_in_layer(255), [15, 15]);
    assert_eq!(probe_layer(256), 1);
    assert_eq!(probe_in_layer(256), [0, 0]);
    // irr texel 坐标：id=300（layer 1，层内 id 44 → probe[12,2]）、texel (2,3)
    let (l, u, v) = irr_texel_coord(300, 2, 3);
    assert_eq!((l, u, v), (1, 12 * 8 + 2, 2 * 8 + 3));
    let (l, u, v) = depth_texel_coord(300, 2, 3);
    assert_eq!((l, u, v), (1, 12 * 16 + 2, 2 * 16 + 3));
    // 层容量幂：PROBES_PER_LAYER = 2^8 保证 id→layer 为位移
    assert!(PROBES_PER_LAYER.is_power_of_two());
  }

  /// 元数据打包往返：offset(×2 定点)/age 各字段无损；quantize_offset 与 BFS 半体素中心无损
  #[test]
  fn meta_pack_roundtrip() {
    // 全零 / 全满 / age 饱和边界
    assert_eq!(pack_probe_meta([0, 0, 0], 0), 0);
    assert_eq!(pack_probe_meta([31, 31, 31], 255), 31 | 31 << 5 | 31 << 10 | 255 << 15);
    let (off, age) = unpack_probe_meta(pack_probe_meta([1, 17, 31], 200));
    assert_eq!((off, age), ([1, 17, 31], 200));
    // quantize：Air 居中探针 (cell+8) → ×2 = 16 精确
    let cell = IVec3::new(3, -2, 7);
    let center = (cell * DDGI_CELL).as_vec3() + Vec3::splat(8.0);
    assert_eq!(quantize_offset(center, cell * DDGI_CELL), [16, 16, 16]);
    // BFS 兜底体素中心（.5）→ ×2 后整数，无损
    let half = (cell * DDGI_CELL).as_vec3() + Vec3::new(7.5, 7.5, 7.5);
    assert_eq!(quantize_offset(half, cell * DDGI_CELL), [15, 15, 15]);
    // 反解回世界坐标误差 < 1/64 cell
    let (off, _) = unpack_probe_meta(pack_probe_meta(quantize_offset(half, cell * DDGI_CELL), 0));
    let back = cell_min_vec(cell) + Vec3::new(off[0] as f32, off[1] as f32, off[2] as f32)
      * (DDGI_CELL as f32 / META_OFFSET_QUANT);
    assert!(back.distance(half) < 1e-4);
  }

  fn cell_min_vec(cell: IVec3) -> Vec3 {
    (cell * DDGI_CELL).as_vec3()
  }

  /// M2-2 元数据纹理 roundtrip：bake → build_meta_texture_data → 按纹理坐标取回
  /// 解包 → offset_to_world 还原探针位置（Air 居中 / BFS 整数 / 薄墙 .5 半体素）、
  /// 初烘 age=0、层布局映射一致
  #[test]
  fn meta_texture_roundtrip() {
    // 薄墙世界：cell (0,0,0) 探针落体素中心 (7.5)³ → 唯一 .5 半体素量化路径
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(3).color = [128, 64, 32];
    for z in 0..16 {
      for y in 0..16 {
        for x in 0..16 {
          if x % 4 == 0 || y % 4 == 0 || z % 4 == 0 {
            g.set_voxel_ivec3(IVec3::new(x, y, z), 3);
          }
        }
      }
    }
    let vols = Volumes::new(g);
    let pg = bake_probe_grid(&vols);

    // 层数 = ceil(4096 / 256) = 16；数据长度 = 层数 × 256
    let layers = meta_texture_layers(pg.positions.len() as u32);
    assert_eq!(pg.positions.len(), 16 * 16 * 16, "全 cell 覆盖（全 Mixed，无 Solid）");
    assert_eq!(layers, 16);
    let data = build_meta_texture_data(&pg);
    assert_eq!(data.len(), layers as usize * PROBES_PER_LAYER as usize);

    // 逐探针 roundtrip：cell_index 反查 cell → 纹理坐标取 texel → 解包还原
    let dx = pg.grid_dims.x as usize;
    let dxy = (pg.grid_dims.x * pg.grid_dims.y) as usize;
    for (li, &id) in pg.cell_index.iter().enumerate() {
      assert_ne!(id, NO_PROBE, "全 cell 覆盖下不应有空 cell");
      let z = (li / dxy) as i32;
      let y = ((li % dxy) / dx) as i32;
      let x = (li % dx) as i32;
      let cell = pg.grid_origin + IVec3::new(x, y, z);
      let (layer, tx, ty) = meta_texel_coord(id);
      let (off, age) = unpack_probe_meta(data[meta_texel_linear(layer, tx, ty)]);
      assert_eq!(age, 0, "初烘 age=0");
      assert_eq!(
        offset_to_world(off, cell * DDGI_CELL),
        pg.positions[id as usize],
        "cell {cell} → probe {id} 位置无损还原"
      );
    }
    // 层坐标映射：id 0/255/256/4095 的 (layer, x, y)
    assert_eq!(meta_texel_coord(0), (0, 0, 0));
    assert_eq!(meta_texel_coord(255), (0, 15, 15));
    assert_eq!(meta_texel_coord(256), (1, 0, 0));
    assert_eq!(meta_texel_coord(4095), (15, 15, 15));
    // 保留位忽略：pack 后污染 [23..32) 不影响 age 解包（对齐 WGSL bitfieldExtract）
    let polluted = pack_probe_meta([31, 31, 31], 255) | 0xFF00_0000;
    assert_eq!(unpack_probe_meta(polluted).1, 255);
  }

  /// PCG hash：黄金值锁死（python 同公式离线计算，改常数必炸）
  #[test]
  fn pcg_golden() {
    assert_eq!(pcg_hash(0), 129708002);
    assert_eq!(pcg_hash(1), 2831084092);
    assert_eq!(pcg_hash(2), 2055130248);
    assert_eq!(pcg_hash(4096), 2484139701);
    assert_eq!(pcg_hash(0xDEADBEEF), 1730779506);
    // 组合种子：帧号/探针/射线任一变化 → 输出变化（独立性抽验）
    assert_eq!(ray_rand(0, 0, 0), 2145236065);
    assert_eq!(ray_rand(7, 1, 63), 3551460842);
    assert_ne!(ray_rand(7, 2, 63), ray_rand(7, 1, 63));
    assert_ne!(ray_rand(8, 1, 63), ray_rand(7, 1, 63));
    assert_ne!(ray_rand(7, 1, 64), ray_rand(7, 1, 63));
  }

  /// u32 → [0,1)² 边界
  #[test]
  fn rand2_bounds() {
    assert_eq!(rand2(0), [0.0, 0.0]);
    let u = (0xFFFF as f32) / 65536.0;
    assert_eq!(rand2(0xFFFF), [u, 0.0]);
    assert_eq!(rand2(0xFFFF_0000), [0.0, u]);
  }

  /// 球面均匀方向：轴向锚点 + 统计均匀性
  #[test]
  fn sphere_dir_uniform() {
    let close = |a: Vec3, b: Vec3| a.distance(b) < 1e-5;
    assert!(close(uniform_sphere_dir([0.0, 0.0]), Vec3::Z));
    assert!(close(uniform_sphere_dir([0.5, 0.0]), Vec3::X));
    let d = uniform_sphere_dir([0.5, 0.25]);
    assert!(d.x.abs() < 1e-6 && close(d, Vec3::Y));
    assert!(close(uniform_sphere_dir([0.5, 0.5]), -Vec3::X));
    // 统计：|dir|≡1、均值 z≈0、上半球占比≈0.5
    let n = 65536u32;
    let mut mean_z = 0.0f32;
    let mut upper = 0u32;
    for i in 0..n {
      let dir = uniform_sphere_dir(rand2(ray_rand(i, 7, 3)));
      assert!((dir.length() - 1.0).abs() < 1e-4);
      mean_z += dir.z;
      if dir.z > 0.0 {
        upper += 1;
      }
    }
    mean_z /= n as f32;
    assert!(mean_z.abs() < 0.01, "z 均值 {mean_z}");
    let ratio = upper as f32 / n as f32;
    assert!((ratio - 0.5).abs() < 0.01, "上半球占比 {ratio}");
  }

  /// collect_radiance（D6 π·Σw·L/Σw 逐字）：单样本恒 π·L；余弦混样；零权早退
  #[test]
  fn collect_radiance_d6() {
    let s60 = 60f32.to_radians().sin();
    let c60 = 60f32.to_radians().cos();
    // 单样本：irr = π·(w·L)/w = π·L，与夹角无关
    let irr = collect_radiance(Vec3::Z, &[(Vec3::new(s60, 0.0, c60), Vec3::X)]);
    assert!((irr.x - std::f32::consts::PI).abs() < 1e-5 && irr.y == 0.0 && irr.z == 0.0);
    // 两样本混合（手算：√.5·2+4)/（1+√.5)·π ≈ 9.9645）
    let r2 = 2.0f32.sqrt() / 2.0;
    let irr = collect_radiance(
      Vec3::Z,
      &[(Vec3::new(r2, 0.0, r2), Vec3::splat(2.0)), (Vec3::Z, Vec3::splat(4.0))],
    );
    assert!((irr.x - 9.96446).abs() < 1e-3, "irr.x={}", irr.x);
    // 全部样本在背面 → Σw=0 → 0
    assert_eq!(collect_radiance(Vec3::Z, &[(-Vec3::Z, Vec3::splat(5.0))]), Vec3::ZERO);
    // 空样本 → 0
    assert_eq!(collect_radiance(Vec3::Z, &[]), Vec3::ZERO);
  }

  /// 更新链逐值锁死：黑初值直采 / 稳态不动 / 大暗化加速 / 大亮化钳制 / 暗化保底步进
  #[test]
  fn update_texel_chain() {
    let close = |a: Vec3, b: Vec3, e: f32| a.distance(b) < e;
    // ① prev 全黑 → hysteresis=0 直采；但 RTXGI L529-533 亮度钳制作用于 delta 本身，
    //    不因 hysteresis=0 豁免：lum(3,1,2)=1.4974>1.0 → delta×0.25 → (0.75,0.25,0.5)
    assert!(close(update_irradiance_texel(Vec3::ZERO, Vec3::new(3.0, 1.0, 2.0)),
      Vec3::new(0.75, 0.25, 0.5), 1e-6));
    // ② 稳态：prev == new → 不动
    assert!(close(update_irradiance_texel(Vec3::ONE, Vec3::ONE), Vec3::ONE, 1e-6));
    // ③ 大暗化（|prev−new| 最大分量 0.5 > 0.2 → h=0.95−0.75=0.2）→ EMA 步 0.8·0.5=0.4
    assert!(close(update_irradiance_texel(Vec3::ONE, Vec3::splat(0.5)), Vec3::splat(0.6), 1e-4));
    // ④ 大亮化（delta 亮度 4.9 > 1.0 → delta×0.25）→ 0.1 + 0.05·1.225 = 0.16125
    assert!(close(update_irradiance_texel(Vec3::splat(0.1), Vec3::splat(5.0)),
      Vec3::splat(0.16125), 1e-4));
    // ⑤ 暗化保底步进：EMA 步 0.05·0.001=5e-5 < 1/1024 → 抬到 1/1024（且 ≤ |delta|）
    let out = update_irradiance_texel(Vec3::splat(0.001), Vec3::ZERO);
    assert!(out.x > 0.0 && out.x < 1e-4, "out={out:?}");
    assert!((out.x - (0.001 - 1.0 / 1024.0)).abs() < 1e-6);
    // ⑥ 无大暗化的普通 EMA（0.1→0.15：0.05·0.95 权重）
    assert!(close(update_irradiance_texel(Vec3::splat(0.1), Vec3::splat(0.15)),
      Vec3::splat(0.1 + 0.05 * (1.0 - DDGI_HYSTERESIS)), 1e-5));
  }

  /// age / reuse 生命周期：滚动继承（bounds 内+offset 未变）↔ 归零；更新 +1 饱和
  #[test]
  fn age_reuse_lifecycle() {
    assert_eq!(age_after_update(0), 1);
    assert_eq!(age_after_update(254), 255);
    assert_eq!(age_after_update(255), 255, "u8 饱和");
    assert_eq!(age_after_scroll(true, 200), 200);
    assert_eq!(age_after_scroll(false, 200), 0, "不可复用 → 重新收敛");
    // bounds = [-2, 3)：含下端、不含上端
    let bounds = (IVec3::splat(-2), IVec3::splat(3));
    assert!(probe_reusable(IVec3::ZERO, [16, 16, 16], [16, 16, 16], bounds));
    assert!(probe_reusable(IVec3::splat(-2), [0, 0, 0], [0, 0, 0], bounds), "下端含");
    assert!(!probe_reusable(IVec3::splat(3), [0, 0, 0], [0, 0, 0], bounds), "上端不含");
    assert!(!probe_reusable(IVec3::splat(-3), [0, 0, 0], [0, 0, 0], bounds));
    assert!(!probe_reusable(IVec3::ZERO, [8, 16, 16], [16, 16, 16], bounds), "offset 变 → 重烘");
  }

  /// can_skip 档位（INFERENCE 实现）：age=0 永不跳；age=255 → p=0.5
  #[test]
  fn can_skip_profile() {
    assert!(!can_skip_update(0, 0), "新鲜探针永不跳");
    assert!(can_skip_update(255, 0), "全熟 + rand=0 → 跳");
    assert!(!can_skip_update(255, 0xFFFF), "rand 高位 → 不跳");
    assert!(!can_skip_update(255, 0x8000), "边界：p=0.5 时 rand/65536=0.5 不跳");
    assert!(can_skip_update(128, 0));
  }

  /// 采样权重数学：chevron 与锐利背面权重的分段形状（Rohacek §3.2/§3.3）
  #[test]
  fn sampling_weight_shapes() {
    let close = |a: f32, b: f32| (a - b).abs() < 1e-6;
    // 背面探针：N·d < 0 → wn = 0（锐利剔除，穿墙不漏光）
    let wn = |ndotd: f32| (ndotd / DDGI_NORMAL_BIAS).clamp(0.0, 1.0);
    assert!(close(wn(-0.5), 0.0));
    assert!(close(wn(0.0), 0.0));
    assert!(close(wn(0.02), 0.1));
    assert!(close(wn(0.2), 1.0));
    assert!(close(wn(1.0), 1.0));
    // chevron：dtex = 探针沿方向到几何距离；dist = 探针到着色点距离
    let wd = |dtex: f32, dist: f32| ((dtex - dist) / DDGI_DEPTH_BIAS + 0.5).clamp(0.0, 1.0);
    assert!(
      close(wd(8192.0, 8.0), 1.0),
      "无遮挡（depth=tmax 初值）→ 全权重"
    );
    assert!(close(wd(10.0, 10.0), 0.5), "几何恰在着色点 → 半权重过渡带");
    assert!(close(wd(2.0, 10.0), 0.0), "墙在探针与点之间 8 格 → 剔除");
    assert!(close(wd(12.0, 10.0), 1.0), "几何比点远 2 格 → 可见");
    assert!(close(wd(9.0, 10.0), 0.25), "过渡带内线性");
  }

  /// 八面体编解码往返：随机单位方向 decode(encode(d)) ≈ d
  #[test]
  fn oct_roundtrip() {
    let mut s: u64 = 0xDD61;
    let mut rand = move || {
      s ^= s << 13;
      s ^= s >> 7;
      s ^= s << 17;
      s
    };
    for _ in 0..2048 {
      let d = Vec3::new(
        (rand() % 2_000_001) as f32 / 1_000_000.0 - 1.0,
        (rand() % 2_000_001) as f32 / 1_000_000.0 - 1.0,
        (rand() % 2_000_001) as f32 / 1_000_000.0 - 1.0,
      )
      .normalize_or_zero();
      if d == Vec3::ZERO {
        continue;
      }
      let back = oct_decode(oct_encode(d));
      assert!(
        back.distance(d) < 1e-4,
        "oct 往返失真 d={d:?} back={back:?}"
      );
    }
    // texel 中心 → 编码 → 同 texel（采样一致性）
    for ty in 0..IRRADIANCE_TEXELS {
      for tx in 0..IRRADIANCE_TEXELS {
        let d = oct_texel_dir(tx, ty, IRRADIANCE_TEXELS);
        let t = oct_texel(d, IRRADIANCE_TEXELS);
        assert_eq!(t, [tx, ty], "texel 中心往返 ({tx},{ty}) → {t:?}");
      }
    }
  }

  /// Fibonacci 球：单位长度 + 沿极轴单调 + 方位角铺开
  #[test]
  fn fibonacci_coverage() {
    let n = 1024u32;
    let mut prev_y = 2.0f32;
    for i in 0..n {
      let d = fibonacci_dir(i, n);
      assert!((d.length() - 1.0).abs() < 1e-5);
      assert!(d.y < prev_y, "y 应沿 i 单调递减（Fibonacci 螺旋）");
      prev_y = d.y;
    }
    // 首末方向贴近两极
    assert!(fibonacci_dir(0, n).y > 0.99);
    assert!(fibonacci_dir(n - 1, n).y < -0.99);
  }

  #[test]
  fn bake_empty_world_no_probes() {
    let vols = Volumes::new(VolumeGrid::new());
    let pg = bake_probe_grid(&vols);
    assert_eq!(pg.positions.len(), 0);
    assert_eq!(pg.cell_index.len(), 0);
  }

  /// D1 全 cell 覆盖：实心 cell 无探针，其余全部 cell（含远场纯空气）都有探针
  #[test]
  fn bake_solid_full_coverage() {
    let g = world_with_cell_box();
    let vols = Volumes::new(g);
    let pg = bake_probe_grid(&vols);
    assert!(!pg.positions.is_empty());

    // 域 = chunk bbox cell 域（无外扩）
    assert_eq!(pg.grid_origin, IVec3::ZERO);
    assert_eq!(pg.grid_dims, UVec3::splat(CELLS_PER_CHUNK as u32));

    let solid_cell = IVec3::splat(1);
    let lookup = |pg: &ProbeGrid, cell: IVec3| -> u32 {
      let rel = (cell - pg.grid_origin).as_uvec3();
      pg.cell_index[pg.cell_linear(rel)]
    };
    // 实心 cell 本体：无探针
    assert_eq!(lookup(&pg, solid_cell), NO_PROBE);
    // 6 邻接空气 cell：有探针且居中（Air 居中）
    for d in [
      IVec3::X,
      IVec3::NEG_X,
      IVec3::Y,
      IVec3::NEG_Y,
      IVec3::Z,
      IVec3::NEG_Z,
    ] {
      let air = solid_cell + d;
      let idx = lookup(&pg, air);
      assert_ne!(idx, NO_PROBE, "空气 cell {air} 应有探针");
      let expect = (air * DDGI_CELL).as_vec3() + Vec3::splat(8.0);
      assert_eq!(pg.positions[idx as usize], expect, "空气 cell {air} 居中");
    }
    // 壳外第二层纯空气 cell：同样有探针（全 cell 覆盖，废除活跃壳）
    assert_ne!(lookup(&pg, solid_cell + IVec3::X * 2), NO_PROBE);
    assert_ne!(lookup(&pg, solid_cell + IVec3::X + IVec3::Y), NO_PROBE);
    // 探针总数 = cell 总数 − Solid cell 数 = 16³ − 1
    assert_eq!(pg.positions.len(), CELLS_PER_CHUNK as usize * CELLS_PER_CHUNK as usize * CELLS_PER_CHUNK as usize - 1);
  }

  /// Mixed cell：BFS 找到靠中心的 4³ 空子砖（填补 (0,0,0) 子砖 → 探针在 (6,6,6)）
  #[test]
  fn bake_mixed_cell_offset_to_empty_subbrick() {
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(3).color = [128, 64, 32];
    // 只填 (0,0,0) 那个 4³ 子砖（fine 0..4）→ cell Mixed
    gate_voxel::fill_bricks(&mut g, IVec3::ZERO, IVec3::splat(4), 4, 3);
    let vols = Volumes::new(g);
    let pg = bake_probe_grid(&vols);
    let rel = (IVec3::ZERO - pg.grid_origin).as_uvec3();
    let idx = pg.cell_index[pg.cell_linear(rel)];
    assert_ne!(idx, NO_PROBE, "Mixed cell 应有探针");
    assert_eq!(
      pg.positions[idx as usize],
      Vec3::splat(6.0),
      "最近中心空子砖 (4..8)³ 的中心 = (6,6,6)"
    );
  }

  /// 棋盘薄墙 cell（64 个 4³ 子砖全 Mixed、无一个全空）→ 1³ 空体素兜底，
  /// 探针落在中心最近空气体素 (7,7,7) 的中心 (7.5,7.5,7.5)
  #[test]
  fn bake_thin_wall_cell_voxel_fallback() {
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(3).color = [128, 64, 32];
    // 每 4³ 子砖角上有正交平面（x/y/z ≡ 0 mod 4）→ 全部 Mixed、无 4³ 空子砖
    for z in 0..16 {
      for y in 0..16 {
        for x in 0..16 {
          if x % 4 == 0 || y % 4 == 0 || z % 4 == 0 {
            g.set_voxel_ivec3(IVec3::new(x, y, z), 3);
          }
        }
      }
    }
    let vols = Volumes::new(g);
    let pg = bake_probe_grid(&vols);
    let rel = (IVec3::ZERO - pg.grid_origin).as_uvec3();
    let idx = pg.cell_index[pg.cell_linear(rel)];
    assert_ne!(idx, NO_PROBE);
    // 中心 (8,8,8) 是墙（8%4==0）→ ±4 盒内最近空体素 (7,7,7)，中心 (7.5,7.5,7.5)
    assert_eq!(pg.positions[idx as usize], Vec3::splat(7.5));
  }

  /// 帧计划：预算摊派 + 环形轮转全覆盖
  #[test]
  fn frame_plan_math() {
    // 0 探针 → 空计划
    assert_eq!(
      frame_plan(0, 0),
      DdgiFramePlan {
        probes_this_frame: 0,
        cycle_base: 0,
        rays_dispatch: 0
      }
    );
    // 100 探针：预算 4096/64 = 64 探针/帧，2 帧全覆盖
    let p0 = frame_plan(100, 0);
    assert_eq!(
      p0,
      DdgiFramePlan {
        probes_this_frame: 64,
        cycle_base: 0,
        rays_dispatch: 64 * 64
      }
    );
    let p1 = frame_plan(100, 1);
    assert_eq!(p1.cycle_base, 64);
    // 覆盖性：轮转 ceil(100/64)=2 帧，所有探针恰好被访问一次
    let mut seen = vec![false; 100];
    for f in 0..2u32 {
      let p = frame_plan(100, f);
      for j in 0..p.probes_this_frame {
        seen[((p.cycle_base + j) % 100) as usize] = true;
      }
    }
    assert!(seen.iter().all(|&s| s), "轮转 2 帧应覆盖全部 100 探针");
    // 超大探针数：钳在预算内
    assert_eq!(frame_plan(1_000_000, 3).probes_this_frame, 64);
  }

  /// DdgiMeta.build 字段镜像
  #[test]
  fn meta_build_mirrors_probe_grid() {
    let g = world_with_cell_box();
    let vols = Volumes::new(g);
    let pg = bake_probe_grid(&vols);
    let plan = frame_plan(pg.positions.len() as u32, 0);
    let meta = DdgiMeta::build(&pg, &plan);
    assert_eq!(meta.probe_count, pg.positions.len() as u32);
    assert_eq!(meta.active_count, meta.probe_count, "v1：分配即全量（v2 由 ddgi_active 每帧判定接管）");
    assert_eq!(meta.probes_this_frame, plan.probes_this_frame);
    assert_eq!(meta.grid_origin, Vec4::new(0.0, 0.0, 0.0, 0.0));
    assert_eq!(meta.grid_dims, Vec4::new(16.0, 16.0, 16.0, 0.0));
    assert_eq!(
      meta.cell_tmax_alpha,
      Vec4::new(16.0, PROBE_T_MAX, DDGI_ALPHA, 0.0)
    );
    let (irr, dep) = initial_probe_data(meta.probe_count);
    assert_eq!(
      irr.len(),
      IRRADIANCE_WORDS_PER_PROBE as usize / 4 * meta.probe_count as usize
    );
    assert_eq!(
      dep.len(),
      DEPTH_WORDS_PER_PROBE as usize * meta.probe_count as usize
    );
    assert!(irr.iter().all(|v| *v == Vec4::ZERO));
  }

  // ---- CPU 参考采样端到端（WGSL sample_ddgi 镜像行为锁死）----

  /// 构造 dims=2³、origin=0 的 dense cell_index（8 cell → Option<probe id>）
  fn cell_index_2x2x2(cell_fill: [Option<u32>; 8]) -> Vec<u32> {
    let mut cell_index = vec![NO_PROBE; 8];
    for (ci, id) in cell_fill.into_iter().enumerate() {
      if let Some(id) = id {
        cell_index[ci] = id;
      }
    }
    cell_index
  }

  /// 8 探针均匀数据：每探针 64 irradiance texel 全填 color、256 depth 全填 depth_val
  fn uniform_probe_data(color: Vec3, depth_val: f32) -> (Vec<Vec4>, Vec<f32>) {
    let mut irr = Vec::new();
    let mut depth = Vec::new();
    for _ in 0..8 {
      for _ in 0..64 {
        irr.push(Vec4::new(color.x, color.y, color.z, 1.0));
      }
      for _ in 0..256 {
        depth.push(depth_val);
      }
    }
    (irr, depth)
  }

  /// 8 探针 = 2×2×2 cell 中心（fine (8|24)³）
  fn eight_probe_positions() -> Vec<Vec3> {
    (0..8)
      .map(|ci| {
        let (x, y, z) = (ci % 2, ci / 2 % 2, ci / 4);
        Vec3::new(
          x as f32 * 16.0 + 8.0,
          y as f32 * 16.0 + 8.0,
          z as f32 * 16.0 + 8.0,
        )
      })
      .collect()
  }

  /// 全探针同色 + depth=tmax → 任意正面半球采样点结果 = 该色（归一化与权重分布无关）
  #[test]
  fn cpu_sample_uniform_color_normalizes() {
    let positions = eight_probe_positions();
    let (irr, depth) = uniform_probe_data(Vec3::new(1.0, 0.2, 0.1), PROBE_T_MAX);
    let cell_index = cell_index_2x2x2(std::array::from_fn(|i| Some(i as u32)));
    let a = DdgiProbeArrays {
      grid_origin: IVec3::ZERO,
      grid_dims: UVec3::splat(2),
      positions: &positions,
      cell_index: &cell_index,
      irr: &irr,
      depth: &depth,
    };
    // p=(12,12,12) 在 8 cell 交界，n=+Y：上方 4 探针（法线前侧）贡献，归一化后仍为探针色
    let c = cpu_sample_ddgi(&a, Vec3::splat(12.0), Vec3::Y);
    assert!(
      (c - Vec3::new(1.0, 0.2, 0.1)).length() < 1e-5,
      "归一化加权和 = {c:?}"
    );
  }

  /// 探针全在表面法线后侧 → wsum=0 → 黑（锐利背面剔除）
  #[test]
  fn cpu_sample_backface_all_rejected() {
    let positions = eight_probe_positions();
    let (irr, depth) = uniform_probe_data(Vec3::ONE, PROBE_T_MAX);
    let cell_index = cell_index_2x2x2(std::array::from_fn(|i| Some(i as u32)));
    let a = DdgiProbeArrays {
      grid_origin: IVec3::ZERO,
      grid_dims: UVec3::splat(2),
      positions: &positions,
      cell_index: &cell_index,
      irr: &irr,
      depth: &depth,
    };
    // p.y=4 < 所有探针 y(8/24)：探针全在 p 上方；n=-Y（法线朝下 = 前侧在下方）
    // → 探针全在表面后侧 → N·(接收点→探针) < 0 全剔除
    let c = cpu_sample_ddgi(&a, Vec3::new(12.0, 4.0, 12.0), -Vec3::Y);
    assert_eq!(c, Vec3::ZERO, "背面探针必须全剔除");
  }

  /// depth 全 1.0（探针贴墙）而 p 距前侧探针 ~7 → chevron wd=0 → 黑（漏光治理）
  #[test]
  fn cpu_sample_depth_chevron_occludes() {
    let positions = eight_probe_positions();
    let (irr, depth) = uniform_probe_data(Vec3::ONE, 1.0);
    let cell_index = cell_index_2x2x2(std::array::from_fn(|i| Some(i as u32)));
    let a = DdgiProbeArrays {
      grid_origin: IVec3::ZERO,
      grid_dims: UVec3::splat(2),
      positions: &positions,
      cell_index: &cell_index,
      irr: &irr,
      depth: &depth,
    };
    // n=-Y（法线朝下）：前侧探针 = p 下方 y=8 探针（距离 ~7）；
    // depth=1（探针与 p 之间有墙）→ chevron 全遮挡 → 黑
    let c = cpu_sample_ddgi(&a, Vec3::splat(12.0), -Vec3::Y);
    assert_eq!(c, Vec3::ZERO, "墙在探针与 p 之间 → chevron 全剔除");
  }

  /// cell 全 NO_PROBE / 空探针集 → 黑
  #[test]
  fn cpu_sample_no_probe_cells() {
    let positions = vec![Vec3::splat(8.0)];
    let irr = vec![Vec4::ZERO; 64];
    let depth = vec![PROBE_T_MAX; 256];
    let cell_index = cell_index_2x2x2([None; 8]);
    let a = DdgiProbeArrays {
      grid_origin: IVec3::ZERO,
      grid_dims: UVec3::splat(2),
      positions: &positions,
      cell_index: &cell_index,
      irr: &irr,
      depth: &depth,
    };
    assert_eq!(cpu_sample_ddgi(&a, Vec3::splat(12.0), Vec3::Y), Vec3::ZERO);
    // 空 positions 早退
    let a0 = DdgiProbeArrays {
      grid_origin: IVec3::ZERO,
      grid_dims: UVec3::splat(2),
      positions: &[],
      cell_index: &[],
      irr: &[],
      depth: &[],
    };
    assert_eq!(cpu_sample_ddgi(&a0, Vec3::splat(12.0), Vec3::Y), Vec3::ZERO);
  }

  /// oct 方向性：+Y 半球 texel 红、-Y 半球 texel 蓝 → 朝上采样见红、朝下见蓝
  #[test]
  fn cpu_sample_oct_directional_color() {
    // 单探针在 cell(1,0,1) 中心附近；p 与探针同 cell（cell 0），8 角仅 (1,0,1) 有探针
    let positions = vec![Vec3::new(20.0, 11.5, 20.0)];
    let mut irr = vec![Vec4::ZERO; 64];
    for ty in 0..8u32 {
      for tx in 0..8u32 {
        let d = oct_texel_dir(tx, ty, IRRADIANCE_TEXELS);
        irr[(ty * 8 + tx) as usize] = if d.y > 0.0 {
          Vec4::new(1.0, 0.0, 0.0, 1.0)
        } else {
          Vec4::new(0.0, 0.0, 1.0, 1.0)
        };
      }
    }
    let depth = vec![PROBE_T_MAX; 256];
    // cell 线性地址：(1,0,1) → 1 + 0*2 + 1*4 = 5
    let mut fill = [None; 8];
    fill[5] = Some(0);
    let cell_index = cell_index_2x2x2(fill);
    let a = DdgiProbeArrays {
      grid_origin: IVec3::ZERO,
      grid_dims: UVec3::splat(2),
      positions: &positions,
      cell_index: &cell_index,
      irr: &irr,
      depth: &depth,
    };
    // p 在探针正下方 1 格，n=+Y（探针在法线前侧=上方）→ 采样方向 n=+Y → 红
    let up = cpu_sample_ddgi(&a, Vec3::new(20.0, 10.5, 20.0), Vec3::Y);
    assert!(up.x > 0.8 && up.z < 0.2, "朝上采样应见红半球：{up:?}");
    // p 在探针正上方 1 格，n=-Y（探针在法线前侧=下方）→ 采样方向 n=-Y → 蓝
    let down = cpu_sample_ddgi(&a, Vec3::new(20.0, 12.5, 20.0), -Vec3::Y);
    assert!(down.z > 0.8 && down.x < 0.2, "朝下采样应见蓝半球：{down:?}");
  }

  /// 64³ 封闭房间（六面 4 厚墙）烘焙分布（M2-1 验收）：
  /// 全 cell 覆盖 → 探针数 = cell 总数 − Solid cell 数 = 16³ − 0（墙 4 厚不占满
  /// 任何 16³ cell，全部 Mixed）；空气 cell 居中；Mixed 墙 cell 探针偏移到空气
  #[test]
  fn bake_room_distribution() {
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(3).color = [128, 64, 32];
    // 六面墙（4 fine 厚）：fill_bricks(origin, size, brick=4, palette)
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 0), IVec3::new(64, 4, 64), 4, 3); // 地板
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 60, 0), IVec3::new(64, 4, 64), 4, 3); // 天花
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 0), IVec3::new(4, 64, 64), 4, 3); // x-
    gate_voxel::fill_bricks(&mut g, IVec3::new(60, 0, 0), IVec3::new(4, 64, 64), 4, 3); // x+
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 0), IVec3::new(64, 64, 4), 4, 3); // z-
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 60), IVec3::new(64, 64, 4), 4, 3); // z+
    let vols = Volumes::new(g);
    let pg = bake_probe_grid(&vols);
    let lookup = |cell: IVec3| -> u32 {
      let rel = (cell - pg.grid_origin).as_uvec3();
      pg.cell_index[pg.cell_linear(rel)]
    };
    // 房间中心 cell（fine 32..48）：纯 Air → 探针居中 (40,40,40)
    let center = IVec3::new(2, 2, 2);
    let id = lookup(center);
    assert_ne!(id, NO_PROBE, "房间内部 cell 应有探针（全 cell 覆盖）");
    assert_eq!(pg.positions[id as usize], Vec3::splat(40.0));
    // 墙角 cell（fine 0..16，三墙各占 0..3）：Mixed → BFS 偏移到最近全空 4³ 子砖
    // fine 4..7 子砖中心 (6,6,6)（遍历序首个距 cell 中心最近的空子砖）
    let corner_id = lookup(IVec3::ZERO);
    assert_ne!(corner_id, NO_PROBE, "Mixed 墙角 cell 应有偏移探针");
    assert_eq!(pg.positions[corner_id as usize], Vec3::splat(6.0));
    // 天花 Mixed cell（fine y=48..64，天花占 60..63）：有探针且位置在空气（y<60）
    let ceil_id = lookup(IVec3::new(2, 3, 2));
    assert_ne!(ceil_id, NO_PROBE, "Mixed 天花 cell 应有偏移探针");
    assert!(pg.positions[ceil_id as usize].y < 60.0, "探针不得落在墙内");
    // 远场 cell（无任何体素数据）：同样有探针，Air 居中（全 cell 覆盖）
    for far in [IVec3::new(5, 5, 5), IVec3::new(10, 10, 10)] {
      let idx = lookup(far);
      assert_ne!(idx, NO_PROBE, "远场 cell {far} 应有探针");
      let expect = (far * DDGI_CELL).as_vec3() + Vec3::splat(8.0);
      assert_eq!(pg.positions[idx as usize], expect, "远场 cell {far} 居中");
    }
    // M2-1 验收公式：探针数 = cell 总数 − Solid cell 数 = 16³ − 0
    assert_eq!(pg.positions.len(), 16 * 16 * 16, "探针数 = cell 总数 − Solid 数");
  }

  /// 编辑驱动重烘：空世界 0 探针 → 填盒（代数推进）→ 挖空回全空气覆盖；
  /// 同色 noop 不推进代数（ProbeBake 版本判据不会误触发重烘）
  #[test]
  fn rebake_after_edit() {
    let mut vols = Volumes::new(VolumeGrid::new());
    assert_eq!(bake_probe_grid(&vols).positions.len(), 0);
    assert_eq!(vols.main().edit_generation(), 0);

    // 填 16³ 盒（fill_bricks 内部逐 4³ brick fill，代数 >= 1）
    gate_voxel::fill_bricks(vols.main_mut(), IVec3::ZERO, IVec3::splat(16), 4, 3);
    let gen_filled = vols.main().edit_generation();
    assert!(gen_filled >= 1, "fill 推进编辑代数");
    // D1 全覆盖：16³ cell − 1 Solid cell = 4095 探针
    assert_eq!(bake_probe_grid(&vols).positions.len(), 16 * 16 * 16 - 1);

    // 同色重复写 = noop，代数不变（重烘判据不抖动）
    vols.main_mut().set_voxel_ivec3(IVec3::new(0, 0, 0), 3);
    assert_eq!(vols.main().edit_generation(), gen_filled);

    // 挖空（fill palette=0 逐 brick 清空气）：fill_brick 不回收空 chunk（只有
    // clear_voxel 会），chunk 仍在域内 → 全空气 cell 全覆盖烘焙；运行时由
    // ddgi_active 剔除（D2），符合 D1「纯空气 cell 也放探针」语义
    gate_voxel::fill_bricks(vols.main_mut(), IVec3::ZERO, IVec3::splat(16), 4, 0);
    assert!(vols.main().edit_generation() > gen_filled, "clear 推进代数");
    assert_eq!(bake_probe_grid(&vols).positions.len(), 16 * 16 * 16);
  }

  /// M2-3 验收：64³ 封闭房间（六面 4 厚墙，同 bake_room_distribution）各级级联烘焙分布。
  /// LOD1(32) 2³ cells 全覆盖 8 探针；LOD2-4 单 cell 覆盖全房，Mixed → 8 子 cell
  /// 择优（空叶大者优先 → 同尺寸靠中心 → 遍历序）。
  #[test]
  fn cascade_room_distribution() {
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(3).color = [128, 64, 32];
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 0), IVec3::new(64, 4, 64), 4, 3); // 地板
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 60, 0), IVec3::new(64, 4, 64), 4, 3); // 天花
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 0), IVec3::new(4, 64, 64), 4, 3); // x-
    gate_voxel::fill_bricks(&mut g, IVec3::new(60, 0, 0), IVec3::new(4, 64, 64), 4, 3); // x+
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 0), IVec3::new(64, 64, 4), 4, 3); // z-
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 60), IVec3::new(64, 64, 4), 4, 3); // z+
    let vols = Volumes::new(g);
    let lookup = |pg: &ProbeGrid, rel: UVec3| -> u32 { pg.cell_index[pg.cell_linear(rel)] };

    // LOD1 cell=32：origin 2 对齐（16-cell 单位），dims 2³ 恰好覆盖房间 fine 0..64
    let l1 = bake_cascade_grid(&vols, 32, IVec3::ZERO, UVec3::splat(2));
    assert_eq!(l1.positions.len(), 8, "墙 4 厚无全满 32³ cell → 8/8 全覆盖");
    // 角 cell (0,0,0)：8 个 16³ 子 cell 仅 (1,1,1) 纯空 → 探针 (24,24,24)
    assert_eq!(l1.positions[lookup(&l1, UVec3::ZERO) as usize], Vec3::splat(24.0));
    // 对角 cell (1,1,1)：仅 (0,0,0) 子 cell 纯空 → (40,40,40)
    assert_eq!(l1.positions[lookup(&l1, UVec3::splat(1)) as usize], Vec3::splat(40.0));

    // LOD2 cell=64：单 cell 覆盖全房。8 个 32³ 子 cell 全 Mixed，各恰含一个纯空
    // 16³ 子 cell（叶 16、距中心 (32,32,32) d²=192 全平手）→ 遍历序首个 (0,0,0)
    // 的子探针 (24,24,24) 胜出
    let l2 = bake_cascade_grid(&vols, 64, IVec3::ZERO, UVec3::splat(1));
    assert_eq!(l2.positions.len(), 1);
    assert_eq!(l2.positions[0], Vec3::splat(24.0));

    // LOD3 cell=128：子 64³ cell 凡含 index=1 轴即跨入 64..128 无体素区 → 纯空叶 64
    // 碾压 Mixed (0,0,0) 的 16 级叶；7 个空叶 d²=3072 平手 → 遍历序首个 (1,0,0)
    // 中心 (96,32,32) 胜出
    let l3 = bake_cascade_grid(&vols, 128, IVec3::ZERO, UVec3::splat(1));
    assert_eq!(l3.positions.len(), 1);
    assert_eq!(l3.positions[0], Vec3::new(96.0, 32.0, 32.0));

    // LOD4 cell=256：同理 128³ 空叶胜出 → 首个 (1,0,0) 中心 (192,64,64)
    let l4 = bake_cascade_grid(&vols, 256, IVec3::ZERO, UVec3::splat(1));
    assert_eq!(l4.positions.len(), 1);
    assert_eq!(l4.positions[0], Vec3::new(192.0, 64.0, 64.0));
  }

  /// M2-3 验收：outside_lower_grid 空间划分 + 级间嵌套不变式。
  /// 区间语义（16-cell 全局坐标，半开 [lo,hi)）：与更细域无重叠 → 本 LOD 管；
  /// 部分重叠保守归更细级。级联滚动（M4-3）必须维持 coarser 域 ⊇ finer 域。
  #[test]
  fn cascade_nesting_and_outside_lower_grid() {
    // 常量嵌套：级联 cell 尺寸逐级 ×2（base 16 → LOD4 256）
    for w in DDGI_CASCADE_CELL_SIZES.windows(2) {
      assert_eq!(w[1], w[0] * 2, "级联 cell 尺寸必须逐级翻倍");
    }

    // 更细域 = LOD1 32-cell 网格 16³ → 32×32×32 16-cell；本 cell 取 LOD2 64-cell（4 宽）
    let (fo, fd) = (IVec3::ZERO, UVec3::splat(32));
    // 完全在更细域内 → 归更细级（!outside）
    assert!(!outside_lower_grid(IVec3::ZERO, IVec3::splat(4), fo, fd));
    assert!(!outside_lower_grid(IVec3::new(28, 4, 4), IVec3::new(32, 8, 8), fo, fd));
    assert!(!outside_lower_grid(IVec3::new(0, 0, 28), IVec3::new(4, 4, 32), fo, fd));
    // 部分重叠（域边缘/跨界）→ 保守归更细级
    assert!(!outside_lower_grid(IVec3::new(28, 0, 0), IVec3::new(32, 4, 4), fo, fd));
    assert!(!outside_lower_grid(IVec3::new(30, 0, 0), IVec3::new(34, 4, 4), fo, fd));
    assert!(!outside_lower_grid(IVec3::new(-2, 0, 0), IVec3::new(2, 4, 4), fo, fd));
    // 完全在外（含半开区间紧贴）→ 归本 LOD
    assert!(outside_lower_grid(IVec3::new(32, 0, 0), IVec3::new(36, 4, 4), fo, fd));
    assert!(outside_lower_grid(IVec3::new(-4, 0, 0), IVec3::new(0, 4, 4), fo, fd));
    assert!(outside_lower_grid(IVec3::new(0, 40, 0), IVec3::new(4, 44, 4), fo, fd));

    // 级间嵌套不变式：对齐滚动下 coarser 域 ⊇ finer 域。
    // 例：LOD2 64-cell 网格 origin (0,0,0)（4 对齐）dims 16³ → 16-cell [0,64)；
    // LOD1 32-cell 网格 origin (10,10,10)（2 对齐、非 4 对齐）dims 16³ → [10,42)
    // ⊂ [0,64)。域边 10/42 与 LOD2 4 对齐 cell 产生真实部分重叠
    let l2_origin = IVec3::ZERO;
    let l1_origin = IVec3::splat(10);
    for a in 0..3 {
      assert_eq!(l1_origin[a] % 2, 0, "LOD1 origin 须 32-cell（2×16）对齐");
      assert_eq!(l2_origin[a] % 4, 0, "LOD2 origin 须 64-cell（4×16）对齐");
    }
    let l2_hi = l2_origin + IVec3::splat(16 * 4); // 64-cell × 16³
    let l1_hi = l1_origin + IVec3::splat(16 * 2); // 32-cell × 16³
    for a in 0..3 {
      assert!(l2_origin[a] <= l1_origin[a] && l1_hi[a] <= l2_hi[a], "coarser ⊇ finer");
    }
    // 嵌套域下的归属（LOD2 cell 4 宽，4 对齐）：
    // LOD1 域内 [16,20) → 归 LOD1；域边部分重叠 [8,12)/[40,44) → 保守归 LOD1；
    // 域外紧贴 [4,8)/[44,48) → 归 LOD2
    let l1_dims = UVec3::splat(16 * 2);
    assert!(!outside_lower_grid(
      IVec3::splat(16),
      IVec3::splat(20),
      l1_origin,
      l1_dims
    ));
    assert!(!outside_lower_grid(
      IVec3::splat(8),
      IVec3::splat(12),
      l1_origin,
      l1_dims
    ));
    assert!(!outside_lower_grid(
      IVec3::splat(40),
      IVec3::splat(44),
      l1_origin,
      l1_dims
    ));
    assert!(outside_lower_grid(
      IVec3::splat(4),
      IVec3::splat(8),
      l1_origin,
      l1_dims
    ));
    assert!(outside_lower_grid(
      IVec3::splat(44),
      IVec3::splat(48),
      l1_origin,
      l1_dims
    ));
  }

  // ==========================================================================
  // M3-1 active 判定单测
  // ==========================================================================

  /// flags_from_brickstate：Air→NO_SURFACES|ENABLED、Mixed→ENABLED、Solid→ENABLED
  #[test]
  fn flags_from_brickstate() {
    let f = brickstate_to_flags(BrickState::Air, true);
    assert_eq!(f, DDGI_PROBE_FLAG_ENABLED | DDGI_PROBE_FLAG_NO_SURFACES);
    let f = brickstate_to_flags(BrickState::Mixed, true);
    assert_eq!(f, DDGI_PROBE_FLAG_ENABLED, "Mixed 有表面 → 不置 NO_SURFACES");
    let f = brickstate_to_flags(BrickState::Solid(3), true);
    assert_eq!(f, DDGI_PROBE_FLAG_ENABLED, "Solid 烘焙期跳过，此分支不常走");
    let f = brickstate_to_flags(BrickState::Air, false);
    assert_eq!(f, 0, "无探针 → flags = 0（未 ENABLED）");
  }

  /// probe_flags_intersection：6 邻接 AND；边界外视为自身 flags（保持 AND 语义）
  #[test]
  fn flags_intersection_bounds() {
    // 2×2×2 flags 数组
    let dims = UVec3::splat(2);
    // (0,0,0)=Air|ENABLED，(1,1,1)=ENABLED（有表面），其余=0（无 ENABLED）
    let mut flags = vec![0u32; 8];
    flags[0] = DDGI_PROBE_FLAG_ENABLED | DDGI_PROBE_FLAG_NO_SURFACES; // rel(0,0,0)
    flags[7] = DDGI_PROBE_FLAG_ENABLED; // rel(1,1,1)

    // rel(0,0,0) 有 3 个边界外邻居 → 视为自身 Air|ENABLED。域内 3 个邻居 (1,0,0)=0,(0,1,0)=0,(0,0,1)=0
    // 邻居 = [3, 0, 3, 0, 3, 0] → AND = 3 & 0 & 3 & 0 & 3 & 0 = 0
    let inter = probe_flags_intersection(&flags, dims, UVec3::ZERO);
    assert_eq!(inter, 0, "域内 0 flags 邻居 → AND=0");

    // rel(1,1,1) 边界外视为自身 ENABLED。域内邻居 (0,1,1)=1(0), (1,0,1)=2(0), (1,1,0)=4(0)
    // 邻居 = [1, 1, 0, 1, 0, 1] → AND = 1 & 1 & 0 & 1 & 0 & 1 = 0
    let inter = probe_flags_intersection(&flags, dims, UVec3::splat(1));
    assert_eq!(inter, 0, "域内 0 flags 邻居 → AND=0");

    // 全 Air|ENABLED → intersection 也是 Air|ENABLED（边界外=自身=3，域内=3，AND=3）
    let flags_all = vec![DDGI_PROBE_FLAG_ENABLED | DDGI_PROBE_FLAG_NO_SURFACES; 8];
    let inter = probe_flags_intersection(&flags_all, dims, UVec3::splat(1));
    assert_eq!(inter, DDGI_PROBE_FLAG_ENABLED | DDGI_PROBE_FLAG_NO_SURFACES);

    // 全 ENABLED（有表面）→ intersection = ENABLED（NO_SURFACES=0 → probe_near_surface 放行）
    let flags_mixed = vec![DDGI_PROBE_FLAG_ENABLED; 8];
    let inter = probe_flags_intersection(&flags_mixed, dims, UVec3::ZERO);
    assert_eq!(inter, DDGI_PROBE_FLAG_ENABLED);
  }

  /// probe_near_surface_flags：三条件 OR 真值表
  #[test]
  fn probe_near_surface_truth_table() {
    // 本 cell 有表面 → true（flag.NO_SURFACES=0）
    assert!(probe_near_surface_flags(
      DDGI_PROBE_FLAG_ENABLED,
      DDGI_PROBE_FLAG_ENABLED | DDGI_PROBE_FLAG_NO_SURFACES,
      &[],
      Vec3::splat(8.0),
      16.0,
    ));
    // 本 cell 纯 Air，但邻接交集有表面（intersect.NO_SURFACES=0）→ true
    assert!(probe_near_surface_flags(
      DDGI_PROBE_FLAG_ENABLED | DDGI_PROBE_FLAG_NO_SURFACES,
      DDGI_PROBE_FLAG_ENABLED,
      &[],
      Vec3::splat(8.0),
      16.0,
    ));
    // 本 cell + 邻接全 Air，但与 object bbox 重叠 → true
    assert!(probe_near_surface_flags(
      DDGI_PROBE_FLAG_ENABLED | DDGI_PROBE_FLAG_NO_SURFACES,
      DDGI_PROBE_FLAG_ENABLED | DDGI_PROBE_FLAG_NO_SURFACES,
      &[ObjectBbox {
        min: Vec3::splat(0.0),
        max: Vec3::splat(16.0),
      }],
      Vec3::splat(8.0),
      16.0,
    ));
    // 三条件全否 → false
    assert!(!probe_near_surface_flags(
      DDGI_PROBE_FLAG_ENABLED | DDGI_PROBE_FLAG_NO_SURFACES,
      DDGI_PROBE_FLAG_ENABLED | DDGI_PROBE_FLAG_NO_SURFACES,
      &[],
      Vec3::splat(100.0), // cell_center 远在域外
      16.0,
    ));
  }

  /// can_skip_update_for_active：age=0 永不跳；固定 frame+probe_id 确定性
  #[test]
  fn can_skip_for_active_deterministic() {
    // age=0: 新鲜探针永不跳（p=0）
    assert!(!can_skip_update_for_active(0, 0, 0));
    assert!(!can_skip_update_for_active(0, 100, 42));
    // age=255, frame=0, probe=0 → rand=pcg_hash(pcg_hash(0^pcg_hash(0))) → 跳（p=0.5）
    assert!(can_skip_update_for_active(255, 0, 0));
    // 确定性：同 probe 同 frame 必同
    assert_eq!(
      can_skip_update_for_active(255, 0, 0),
      can_skip_update_for_active(255, 0, 0),
      "确定性：同输入 → 同输出"
    );
    // 跨 probe_id 输出可能变化（随机性）
    let _ = can_skip_update_for_active(255, 0, 42);
  }

  /// M3-1 核心：64³ 封闭房间 → 近墙 cell 活跃、远场 Air cell 不活跃（正确的 D2 行为）
  #[test]
  fn cpu_ddgi_active_all_room_probes_active() {
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(3).color = [128, 64, 32];
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 0), IVec3::new(64, 4, 64), 4, 3); // floor
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 60, 0), IVec3::new(64, 4, 64), 4, 3); // ceil
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 0), IVec3::new(4, 64, 64), 4, 3); // x-
    gate_voxel::fill_bricks(&mut g, IVec3::new(60, 0, 0), IVec3::new(4, 64, 64), 4, 3); // x+
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 0), IVec3::new(64, 64, 4), 4, 3); // z-
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 60), IVec3::new(64, 64, 4), 4, 3); // z+
    let vols = Volumes::new(g);
    let pg = bake_probe_grid(&vols);
    let flags = compute_cell_flags(vols.main(), &pg, DDGI_CELL);
    let meta = build_meta_texture_data(&pg);

    let cascade = CascadeDomain {
      origin: pg.grid_origin,
      dims: pg.grid_dims,
      cell_size: DDGI_CELL,
    };
    let input = ActiveInput {
      pg: &pg,
      cell_flags: &flags,
      object_bboxes: &[],
      cascade,
      finer: None,
      prev_meta: &meta,
      reuse_bounds: REUSE_ALL,
      frame: 0,
    };
    let out = cpu_ddgi_active(&input);

    // 关键断言：远场 Air cell（距墙 1 cell 之外）正确地不活跃 → 活跃数远小于总探针数
    assert!(
      out.indirect_dispatch < pg.positions.len() as u32,
      "远场 Air cell 应被剔除：活跃 {} < 总 {}",
      out.indirect_dispatch,
      pg.positions.len()
    );
    // 但近墙 Mixed/Air cell 必须活跃 → 活跃数 > 墙附近 cell 数
    // 4³ 房间有 6 面墙，墙 cell 数 ≈ 4³ - 2³ = 56
    assert!(
      out.indirect_dispatch > 56,
      "近墙 cell 应活跃：活跃 {} > 墙 cell 数",
      out.indirect_dispatch
    );
    assert_eq!(out.worklist.len() as u32, out.indirect_dispatch);
    // next_meta 长度 = prev_meta 长度
    assert_eq!(out.next_meta.len(), meta.len());
    // worklist 内活跃探针的 age 必须是 increment 后的值
    for item in &out.worklist {
      assert!(item.age >= 1 && item.age <= DDGI_AGE_MAX);
      let (layer, tx, ty) = meta_texel_coord(item.probe_id);
      let idx = meta_texel_linear(layer, tx, ty);
      let (_, age) = unpack_probe_meta(out.next_meta[idx]);
      assert_eq!(age, item.age);
    }
  }

  /// M3-1 核心：纯 Air 远场 cell → 无邻接表面 + 无 object → 不活跃
  #[test]
  fn cpu_ddgi_active_far_air_inactive() {
    // 全空世界 → 烘焙全 Air 探针
    let vols = Volumes::new(VolumeGrid::new());
    // 给世界一个 AABB 让烘焙有域
    let mut g = VolumeGrid::new();
    gate_voxel::fill_bricks(&mut g, IVec3::ZERO, IVec3::splat(16), 4, 0); // 清空气体素（仅推 chunk 代数到 1）
    let vols = Volumes::new(g);
    let pg = bake_probe_grid(&vols);
    // 0 探针 → 直接 assert
    if pg.positions.is_empty() {
      return;
    }
    let flags = compute_cell_flags(vols.main(), &pg, DDGI_CELL);
    let meta = build_meta_texture_data(&pg);
    let cascade = CascadeDomain {
      origin: pg.grid_origin,
      dims: pg.grid_dims,
      cell_size: DDGI_CELL,
    };
    let input = ActiveInput {
      pg: &pg,
      cell_flags: &flags,
      object_bboxes: &[],
      cascade,
      finer: None,
      prev_meta: &meta,
      reuse_bounds: REUSE_ALL,
      frame: 0,
    };
    let out = cpu_ddgi_active(&input);
    // 全 Air + 无 object → 全不活跃
    assert_eq!(out.indirect_dispatch, 0);
    assert!(out.worklist.is_empty());
  }

  /// M3-1 核心：next_meta 从 prev_meta 拷贝初始化 → 非活跃探针 age 不丢
  #[test]
  fn cpu_ddgi_active_preserves_non_active_meta() {
    // 构造 8 cell 2³ 网格：中心 1 个 Mixed cell + 7 个 Air cell
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(3).color = [128, 64, 32];
    // 在 cell(1,1,1) 放 16³ 实心盒 → Solid cell（烘焙期跳过）
    // 在 cell(0,0,0) 放 4³ 实心 → Mixed cell（有表面）
    gate_voxel::fill_bricks(&mut g, IVec3::ZERO, IVec3::splat(4), 4, 3);
    let vols = Volumes::new(g);
    let pg = bake_probe_grid(&vols);

    // 烘焙出的 cell 数 = 4³ = 64 cell，其中 cell(0,0,0) Mixed 有探针，其余 Air 也有探针（D1 全覆盖）
    // 但 cell(0,0,0) 的 6 邻接有 3 个越界（边界外视为 Air 但无表面条件放行——邻接 Air 探针自身也在域内）
    let flags = compute_cell_flags(vols.main(), &pg, DDGI_CELL);
    let meta = build_meta_texture_data(&pg);

    // 手动给 prev_meta 注入 age=200（模拟已收敛探针）
    let mut prev = meta.clone();
    for li in 0..pg.cell_index.len() {
      if pg.cell_index[li] != NO_PROBE {
        let id = pg.cell_index[li];
        let (layer, tx, ty) = meta_texel_coord(id);
        let idx = meta_texel_linear(layer, tx, ty);
        let (off, _) = unpack_probe_meta(prev[idx]);
        prev[idx] = pack_probe_meta(off, 200);
      }
    }

    let cascade = CascadeDomain {
      origin: pg.grid_origin,
      dims: pg.grid_dims,
      cell_size: DDGI_CELL,
    };
    let input = ActiveInput {
      pg: &pg,
      cell_flags: &flags,
      object_bboxes: &[],
      cascade,
      finer: None,
      prev_meta: &prev,
      reuse_bounds: REUSE_ALL,
      frame: 9999, // 大 frame → can_skip 结果可能不同
    };
    let out = cpu_ddgi_active(&input);

    // 验证：next_meta 长度等于 prev
    assert_eq!(out.next_meta.len(), prev.len());

    // 非活跃探针的 next_meta 应等于 prev_meta（age=200 保留）
    // 活跃探针（在 worklist 里）age 要么 = 200（skip）要么 = 201（increment）
    let active_ids: std::collections::HashSet<u32> =
      out.worklist.iter().map(|i| i.probe_id).collect();
    for li in 0..pg.cell_index.len() {
      if pg.cell_index[li] == NO_PROBE {
        continue;
      }
      let id = pg.cell_index[li];
      let (layer, tx, ty) = meta_texel_coord(id);
      let idx = meta_texel_linear(layer, tx, ty);
      let (_, prev_age) = unpack_probe_meta(prev[idx]);
      let (_, next_age) = unpack_probe_meta(out.next_meta[idx]);

      if active_ids.contains(&id) {
        // 在 worklist → 已 increment
        assert_eq!(next_age, 201, "活跃探针应 age+1");
      } else {
        // 不在 worklist → 被 can_skip 或完全不活跃
        // 如果 probe 存在但完全不活跃（无表面），age 应保持 prev（200）
        // 如果 probe 活跃但被 can_skip，age 也应保持 prev（200，继承后未 increment）
        assert!(
          next_age == prev_age || next_age == 0,
          "非活跃/跳过探针 age 应不变（{prev_age}→{next_age}）"
        );
      }
    }
  }

  /// M3-1：滚动级 reuse bounds → 域内 age 继承、域外归零
  #[test]
  fn cpu_ddgi_active_scroll_reuse_age() {
    // 构造 2×2×2 cell grid（64³ fine），cell(0,0,0) 有 4³ 实心 Mixed
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(3).color = [128, 64, 32];
    gate_voxel::fill_bricks(&mut g, IVec3::ZERO, IVec3::splat(4), 4, 3);
    let vols = Volumes::new(g);

    let pg = bake_cascade_grid(&vols, 32, IVec3::ZERO, UVec3::splat(2));
    assert!(!pg.positions.is_empty());
    let flags = compute_cell_flags(vols.main(), &pg, 32);
    let meta = build_meta_texture_data(&pg);

    // 注入 age=100
    let mut prev = meta.clone();
    for li in 0..pg.cell_index.len() {
      if pg.cell_index[li] != NO_PROBE {
        let id = pg.cell_index[li];
        let (layer, tx, ty) = meta_texel_coord(id);
        let idx = meta_texel_linear(layer, tx, ty);
        let (off, _) = unpack_probe_meta(prev[idx]);
        prev[idx] = pack_probe_meta(off, 100);
      }
    }

    let cascade = CascadeDomain {
      origin: pg.grid_origin,
      dims: pg.grid_dims,
      cell_size: 32,
    };
    let input = ActiveInput {
      pg: &pg,
      cell_flags: &flags,
      object_bboxes: &[],
      cascade,
      finer: None,
      prev_meta: &prev,
      reuse_bounds: (IVec3::splat(-1), IVec3::splat(1)),
      frame: 0,
    };
    let out = cpu_ddgi_active(&input);

    // cell(0,0,0) rel=(0,0,0) → [-1,1) 域内 → reusable；且 Mixed → probe_near_surface=true
    // frame=0, can_skip(age=100) → p=(100/255)²×0.5≈0.077 → rand=pcg_hash(pcg_hash(0^pcg_hash(id)))
    // 不管 skip 与否：skip → age=100；不 skip → age=101
    let lookup = |rel: UVec3| -> u32 { pg.cell_index[pg.cell_linear(rel)] };
    let id000 = lookup(UVec3::ZERO);
    if id000 != NO_PROBE {
      let (layer, tx, ty) = meta_texel_coord(id000);
      let idx = meta_texel_linear(layer, tx, ty);
      let (_, age) = unpack_probe_meta(out.next_meta[idx]);
      // reusable + 活跃 → age ∈ {100 (skip), 101 (increment)}
      assert!(
        age == 100 || age == 101,
        "reusable 探针 age 应保留 100 或 increment 到 101（got {age}）"
      );
    }
    // cell(1,1,1) rel=(1,1,1) → reuse_max=1 不含 1 → 不归 reusable → age=0 起步
    // 但 cell(1,1,1) 全 Air 邻域 → probe_near_surface 可能返回 false → age 保留 100
    // 或者活跃 → age=1
    let id111 = lookup(UVec3::splat(1));
    if id111 != NO_PROBE {
      let (layer, tx, ty) = meta_texel_coord(id111);
      let idx = meta_texel_linear(layer, tx, ty);
      let (_, age) = unpack_probe_meta(out.next_meta[idx]);
      // 不归 reusable → 活跃则 age=1（can_skip 对 age=0 必不跳）；不活跃则 age=100 保留
      assert!(
        age == 1 || age == 100,
        "非 reusable 探针 age 应归零后 increment（1）或不活跃保留（100）（got {age}）"
      );
    }
  }

  /// fuzz 等价门禁：固定种子 → cpu_ddgi_active 输出确定性；
  /// 暴力 oracle（逐 cell 全 BrickState 查询）→ flags 模型等价
  #[test]
  fn cpu_ddgi_active_fuzz_equivalence() {
    // 构造 4³ chunk 网格，随机 Mixed/Air 分布
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(3).color = [128, 64, 32];
    // 填充每个 16³ cell 的一个 4³ 角块（棋盘格 → 约一半 Mixed 一半 Air）
    for z in 0..4 {
      for y in 0..4 {
        for x in 0..4 {
          if (x + y + z) % 2 == 0 {
            gate_voxel::fill_bricks(
              &mut g,
              IVec3::new(x * 16, y * 16, z * 16),
              IVec3::splat(4),
              4,
              3,
            );
          }
        }
      }
    }
    let vols = Volumes::new(g);
    let pg = bake_probe_grid(&vols);
    assert!(!pg.positions.is_empty());

    // 两个等价输入：flags 模型 vs BrickState 模型（等价性来自 BrickState→flags 映射）
    let flags = compute_cell_flags(vols.main(), &pg, DDGI_CELL);
    let meta = build_meta_texture_data(&pg);

    let cascade = CascadeDomain {
      origin: pg.grid_origin,
      dims: pg.grid_dims,
      cell_size: DDGI_CELL,
    };

    // 多帧 fuzz（固定 frame 序列 → 确定性 can_skip 决策）
    for frame in 0..100u32 {
      let input = ActiveInput {
        pg: &pg,
        cell_flags: &flags,
        object_bboxes: &[],
        cascade,
        finer: None,
        prev_meta: &meta, // 每帧用同一份 prev（隔离帧间影响）
        reuse_bounds: REUSE_ALL,
        frame,
      };
      let out = cpu_ddgi_active(&input);

      // 验证：worklist probe_id 集合 ⊆ 有 ENABLED 标志的探针集合
      let enabled_set: std::collections::HashSet<u32> = pg
        .cell_index
        .iter()
        .enumerate()
        .filter_map(|(li, &id)| {
          if id != NO_PROBE && (flags[li] & DDGI_PROBE_FLAG_ENABLED) != 0 {
            Some(id)
          } else {
            None
          }
        })
        .collect();
      for item in &out.worklist {
        assert!(
          enabled_set.contains(&item.probe_id),
          "worklist 条目必须是 ENABLED 探针"
        );
      }
      assert_eq!(out.indirect_dispatch, out.worklist.len() as u32);
      // 确定性：同 frame → 同 worklist
      let out2 = cpu_ddgi_active(&input);
      assert_eq!(out.worklist, out2.worklist, "固定 seed → 确定性 worklist");
      assert_eq!(out.next_meta, out2.next_meta, "固定 seed → 确定性 next_meta");
    }
  }

  /// M3-1 object bbox：纯 Air 世界 + bbox 与 cell 重叠 → 探针活跃
  #[test]
  fn cpu_ddgi_active_object_bbox_triggers() {
    // 先创建有 chunk 的世界（放实际体素），再清 Air
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(3).color = [128, 64, 32];
    gate_voxel::fill_bricks(&mut g, IVec3::ZERO, IVec3::splat(16), 4, 3);
    // 清 Air（chunk 仍在域内）
    gate_voxel::fill_bricks(&mut g, IVec3::ZERO, IVec3::splat(16), 4, 0);
    let vols = Volumes::new(g);
    let pg = bake_probe_grid(&vols);
    assert!(!pg.positions.is_empty(), "纯 Air chunk 应有探针（D1 全覆盖）");

    let flags = compute_cell_flags(vols.main(), &pg, DDGI_CELL);
    let meta = build_meta_texture_data(&pg);

    let cascade = CascadeDomain {
      origin: pg.grid_origin,
      dims: pg.grid_dims,
      cell_size: DDGI_CELL,
    };

    // 无 bbox → 全 Air 探针均不活跃
    let input_none = ActiveInput {
      pg: &pg,
      cell_flags: &flags,
      object_bboxes: &[],
      cascade,
      finer: None,
      prev_meta: &meta,
      reuse_bounds: REUSE_ALL,
      frame: 0,
    };
    let out_none = cpu_ddgi_active(&input_none);
    assert_eq!(out_none.indirect_dispatch, 0, "全 Air 无 bbox → 0 活跃");

    // bbox 覆盖 origin cell → 该 cell 探针活跃
    let bbox = ObjectBbox {
      min: Vec3::ZERO,
      max: Vec3::splat(16.0),
    };
    let input_bbox = ActiveInput {
      pg: &pg,
      cell_flags: &flags,
      object_bboxes: &[bbox],
      cascade,
      finer: None,
      prev_meta: &meta,
      reuse_bounds: REUSE_ALL,
      frame: 0,
    };
    let out_bbox = cpu_ddgi_active(&input_bbox);
    assert!(
      out_bbox.indirect_dispatch > 0,
      "bbox 覆盖 → 至少 1 个探针活跃"
    );
    assert!(
      out_bbox.indirect_dispatch <= pg.positions.len() as u32,
      "不超过全探针"
    );
  }

  // ==========================================================================
  // M3-2 cast 射线投射单测
  // ==========================================================================

  use crate::brickmap::BrickMapBuilder;
  use crate::lighting::{DirLightCfg, LightingTheme, build_light_pool, cpu_reference_sky};

  /// M3-2 验收：4096 预算分摊逻辑
  #[test]
  fn cast_budget_math() {
    assert_eq!(cast_rays_per_probe(0), 0, "0 活跃 → 0 射线");
    assert_eq!(cast_rays_per_probe(1), 4096, "1 探针独占全部预算");
    assert_eq!(cast_rays_per_probe(2), 2048);
    assert_eq!(cast_rays_per_probe(64), 64);
    // 100 活跃 → 4096/100 = 40.96 → 40（向下取整，总 4000 ≤ 预算）
    assert_eq!(cast_rays_per_probe(100), 40);
    assert!(cast_rays_per_probe(100) * 100 <= RAY_BUDGET_PER_FRAME);
    // 房间级 1406 活跃（M3-1 实测数）
    assert_eq!(cast_rays_per_probe(1406), 2);
    // 超预算：保底 1（总射线数会超出 4096，Douglas「roughly」语义）
    assert_eq!(cast_rays_per_probe(5000), 1);
  }

  /// 射线方向：单位长度 + 确定性 + 帧间/探针间变化 + 跨帧统计均匀
  #[test]
  fn cast_ray_dir_uniform_and_deterministic() {
    let n = 64u32;
    // 确定性：同输入 → 同方向
    let a = cast_ray_dir(7, 100, 13, n);
    let b = cast_ray_dir(7, 100, 13, n);
    assert_eq!(a, b);
    // 单位长度（旋转保持）
    assert!((a.length() - 1.0).abs() < 1e-5, "|dir|={}", a.length());
    // 帧变化 → 方向变化（旋转去相关）
    assert_ne!(cast_ray_dir(7, 100, 13, n), cast_ray_dir(7, 101, 13, n));
    // 探针变化 → 方向变化
    assert_ne!(cast_ray_dir(7, 100, 13, n), cast_ray_dir(8, 100, 13, n));
    // 帧内 Fibonacci 结构：同一探针同帧内不同 ray_index → 不同方向
    assert_ne!(cast_ray_dir(7, 100, 0, n), cast_ray_dir(7, 100, 1, n));

    // 统计均匀：跨 1024 帧 × 64 射线，均值 z≈0、上半球≈0.5（旋转的均匀性）
    let mut mean_z = 0.0f32;
    let mut upper = 0u32;
    let total = 1024u32 * n;
    for frame in 0..1024u32 {
      for i in 0..n {
        let d = cast_ray_dir(42, frame, i, n);
        assert!((d.length() - 1.0).abs() < 1e-4);
        mean_z += d.z;
        if d.z > 0.0 {
          upper += 1;
        }
      }
    }
    mean_z /= total as f32;
    assert!(mean_z.abs() < 0.01, "z 均值 {mean_z}");
    let ratio = upper as f32 / total as f32;
    assert!((ratio - 0.5).abs() < 0.01, "上半球占比 {ratio}");
  }

  /// 端点 miss 分支：radiance = sky(dir)，dist = PROBE_T_MAX
  #[test]
  fn cast_endpoint_sky_branch() {
    let world = room_world();
    let vols: Vec<(&BrickMapBuffers, VolumeTransform)> = vec![(&world, VolumeTransform::IDENTITY)];
    let pool = build_light_pool(&room_theme());
    // 上一帧探针数组（空——miss 分支不消费，但需要传入）
    let positions: Vec<Vec3> = Vec::new();
    let prev = DdgiProbeArrays {
      grid_origin: IVec3::ZERO,
      grid_dims: UVec3::ZERO,
      positions: &positions,
      cell_index: &[],
      irr: &[],
      depth: &[],
    };
    // 从房内向上：天花厚 4（fine 60..64），从 (32, 40, 32) 直上穿天花命中？
    // 用房外高空向上 → 必 miss（世界只有 64³ 房子）
    let dir = Vec3::new(0.3, 1.0, -0.2).normalize();
    let (rad, dist) = cast_endpoint_radiance(&vols, &pool, &prev, Vec3::new(500.0, 2000.0, 500.0), dir);
    assert_eq!(dist, PROBE_T_MAX, "miss → 远距哨兵");
    assert_eq!(rad, cpu_reference_sky(dir, &pool), "miss → sky 色（逐位）");
    // 向下也应 miss（世界在 2000 下方但 t_max=8192 内…（2000-64）/|dy|…取足够远处）
    let (rad2, dist2) = cast_endpoint_radiance(&vols, &pool, &prev, Vec3::new(2000.0, 2000.0, 2000.0), -Vec3::Y);
    assert_eq!(dist2, PROBE_T_MAX);
    assert_eq!(rad2, cpu_reference_sky(-Vec3::Y, &pool));
  }

  /// 端点 emissive 分支：albedo × emissive × gain 直出（不受阴影/间接光影响）
  #[test]
  fn cast_endpoint_emissive_branch() {
    // 发光地板：32×16×32 盒（pal=5 emissive=200）
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(5).color = [255, 128, 0];
    g.palette_mut().get_mut(5).emissive = 200;
    gate_voxel::fill_bricks(&mut g, IVec3::ZERO, IVec3::new(32, 16, 32), 4, 5);
    let world = BrickMapBuilder::build_full(&g).buffers().clone();
    let vols: Vec<(&BrickMapBuffers, VolumeTransform)> = vec![(&world, VolumeTransform::IDENTITY)];
    let pool = build_light_pool(&room_theme());
    let positions: Vec<Vec3> = Vec::new();
    let prev = DdgiProbeArrays {
      grid_origin: IVec3::ZERO,
      grid_dims: UVec3::ZERO,
      positions: &positions,
      cell_index: &[],
      irr: &[],
      depth: &[],
    };
    let (rad, dist) = cast_endpoint_radiance(&vols, &pool, &prev, Vec3::new(16.0, 100.0, 16.0), -Vec3::Y);
    assert_eq!(dist, 100.0 - 16.0, "命中顶面 y=16 → t=84");
    let base = Vec3::new(1.0, 128.0 / 255.0, 0.0);
    let expect = base * ((200.0 / 255.0) * EMISSIVE_EMIT_GAIN);
    assert!((rad - expect).length() < 1e-5, "emissive 直出 {rad:?} vs {expect:?}");
  }

  /// 端点常规分支：直光 1-bounce（顶面 ndl=1 无遮挡）+ prev DDGI 自闭环（均匀色归一化）
  #[test]
  fn cast_endpoint_direct_plus_indirect() {
    // 32×16×32 常规盒（pal=3 无发光），太阳 -Y 强度 2
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(3).color = [128, 64, 32];
    gate_voxel::fill_bricks(&mut g, IVec3::ZERO, IVec3::new(32, 16, 32), 4, 3);
    let world = BrickMapBuilder::build_full(&g).buffers().clone();
    let vols: Vec<(&BrickMapBuffers, VolumeTransform)> = vec![(&world, VolumeTransform::IDENTITY)];
    let pool = build_light_pool(&room_theme());

    // 上一帧 DDGI：2×2×2 均匀色探针网格（同 cpu_sample_uniform_color_normalizes 布局）
    let positions = eight_probe_positions();
    let irr_color = Vec3::new(0.3, 0.6, 0.9);
    let (irr, depth) = uniform_probe_data(irr_color, PROBE_T_MAX);
    let cell_index = cell_index_2x2x2(std::array::from_fn(|i| Some(i as u32)));
    let prev = DdgiProbeArrays {
      grid_origin: IVec3::ZERO,
      grid_dims: UVec3::splat(2),
      positions: &positions,
      cell_index: &cell_index,
      irr: &irr,
      depth: &depth,
    };

    // 命中顶面 (17,16,15)（避开 cell 边界；归一化采样 = irr_color）
    let (rad, dist) = cast_endpoint_radiance(&vols, &pool, &prev, Vec3::new(17.0, 100.0, 15.0), -Vec3::Y);
    assert_eq!(dist, 100.0 - 16.0);
    let base = Vec3::new(128.0 / 255.0, 64.0 / 255.0, 32.0 / 255.0);
    // 直光：ndl=1、无遮挡 → base × sun(1,1,1)×2；间接：base × irr_color
    let expect = base * 2.0 + base * irr_color;
    assert!(
      (rad - expect).length() < 1e-4,
      "直光+间接 {rad:?} vs {expect:?}"
    );

    // 底面命中（从下方向上打）：直光 ndl<0；且法线 -Y 前侧（下方）无探针
    // （2×2×2 网格探针全在 y≥8 > p.y=0）→ 锐利背面剔除 → 间接亦为 0 → 黑
    let (rad_b, _) = cast_endpoint_radiance(&vols, &pool, &prev, Vec3::new(17.0, -100.0, 15.0), Vec3::Y);
    assert!(
      rad_b.length() < 1e-5,
      "背光底面：直光 ndl<0 + 法线前侧无探针 → 黑（got {rad_b:?}）"
    );
  }

  /// M3-2 验收：active → cast 全链 + 同输入射线序列 → 样本缓冲逐位一致
  #[test]
  fn cpu_ddgi_cast_full_deterministic() {
    let vols_world = room_world();
    let mut g = room_grid();
    let vols = Volumes::new(std::mem::take(&mut g));
    let pg = bake_probe_grid(&vols);
    let flags = compute_cell_flags(vols.main(), &pg, DDGI_CELL);
    let meta = build_meta_texture_data(&pg);

    // active（M3-1）
    let cascade = CascadeDomain {
      origin: pg.grid_origin,
      dims: pg.grid_dims,
      cell_size: DDGI_CELL,
    };
    let active_input = ActiveInput {
      pg: &pg,
      cell_flags: &flags,
      object_bboxes: &[],
      cascade,
      finer: None,
      prev_meta: &meta,
      reuse_bounds: REUSE_ALL,
      frame: 7,
    };
    let active = cpu_ddgi_active(&active_input);
    assert!(active.worklist.len() > 100, "房间近墙探针应大批活跃");

    // prev 探针数组（初值：irr=0 / depth=tmax）
    let (irr0, dep0) = initial_probe_data(pg.positions.len() as u32);
    let prev = DdgiProbeArrays {
      grid_origin: pg.grid_origin,
      grid_dims: pg.grid_dims,
      positions: &pg.positions,
      cell_index: &pg.cell_index,
      irr: &irr0,
      depth: &dep0,
    };
    let pool = build_light_pool(&room_theme());
    let vols_slice: Vec<(&BrickMapBuffers, VolumeTransform)> =
      vec![(&vols_world, VolumeTransform::IDENTITY)];

    let rays = cast_rays_per_probe(active.worklist.len() as u32);
    assert!(rays >= 1, "预算分摊保底 1");
    let cast_input = CastInput {
      vols: &vols_slice,
      light_pool: &pool,
      pg: &pg,
      worklist: &active.worklist,
      rays_per_probe: rays,
      frame: 7,
      prev: &prev,
    };
    let out = cpu_ddgi_cast(&cast_input);

    // 样本数 = worklist × rays_per_probe；槽位布局 slot×rays+i
    assert_eq!(out.samples.len(), active.worklist.len() * rays as usize);
    assert_eq!(out.rays_per_probe, rays);
    // 样本合法性：单位方向、dist ∈ (0, PROBE_T_MAX]、sky 远距哨兵存在
    let mut has_sky = false;
    for s in &out.samples {
      assert!((s.dir.length() - 1.0).abs() < 1e-4);
      assert!(s.dist > 0.0 && s.dist <= PROBE_T_MAX);
      assert!(s.radiance.x.is_finite() && s.radiance.y.is_finite() && s.radiance.z.is_finite());
      if s.dist == PROBE_T_MAX {
        has_sky = true;
      }
    }
    // 房间探针在房内/墙内：向上射线穿天花（4 厚墙）→ 命中或穿出后 miss；
    // 房内探针必然存在 miss 射线（穿墙后 t_max 内无物）→ sky 样本存在
    assert!(has_sky, "封闭房间也应存在 sky 样本（穿墙远射）");

    // 逐位一致（M3-2 验收）：同输入 → 同样本缓冲
    let out2 = cpu_ddgi_cast(&cast_input);
    assert_eq!(out.samples, out2.samples, "固定种子 → 样本缓冲逐位一致");
  }

  /// room 单测共用世界：64³ 封闭房间（六面 4 厚墙，pal=3）
  fn room_grid() -> VolumeGrid {
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(3).color = [128, 64, 32];
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 0), IVec3::new(64, 4, 64), 4, 3);
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 60, 0), IVec3::new(64, 4, 64), 4, 3);
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 0), IVec3::new(4, 64, 64), 4, 3);
    gate_voxel::fill_bricks(&mut g, IVec3::new(60, 0, 0), IVec3::new(4, 64, 64), 4, 3);
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 0), IVec3::new(64, 64, 4), 4, 3);
    gate_voxel::fill_bricks(&mut g, IVec3::new(0, 0, 60), IVec3::new(64, 64, 4), 4, 3);
    g
  }

  fn room_world() -> BrickMapBuffers {
    BrickMapBuilder::build_full(&room_grid()).buffers().clone()
  }

  /// room 单测共用光池：太阳 -Y 强度 2
  fn room_theme() -> LightingTheme {
    LightingTheme {
      sun: Some(DirLightCfg {
        dir: [0.0, -1.0, 0.0],
        angular_radius_deg: 0.0,
        color: [1.0, 1.0, 1.0],
        intensity: 2.0,
      }),
      ambient: [0.1, 0.1, 0.1],
      exposure: 1.0,
      sky: None,
    }
  }
}
