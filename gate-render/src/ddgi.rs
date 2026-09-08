//! R3-10 DDGI 全局光照（Douglas #23 1:1 复刻）——CPU 侧真相源。
//!
//! v2 重做规格见 docs/ddgi-rework-spec.md（SUPERSEDED 本文件内旧版描述）：
//! - D1 探针网格 = 全 cell 覆盖（含纯空气 cell），废除活跃壳；
//! - D4-D6 每帧三段式：活跃判定 → 固定 4096 射线预算投射 → collect_radiance 投影；
//! - D7 载体 = 2D 纹理数组（irr rgba16f 8×8 / depth r32 16×16 / 元数据双缓冲 ping-pong）；
//! - D9-D10 4 级相机滚动级联 + age/reuse bounds 继承。
//!
//! 本文件只含 GPU 编排（布局常量 / 探针烘焙 / 级联滚动 / BG4 资源与 pass 调度）；
//! 光照数学（active 判定 / 射线投射 / 投影更新 / 采样）全部在 WGSL 侧，
//! CPU 侧不做任何光照计算（CPU 镜像代码已按 2026-09-08 决策整体拆除）。
//!
//! 编辑响应：VolumeGrid.edit_generation 单调计数 → extract 版本比对 → base 重烘 +
//! 全级 age 归零（语义随 D10 更新，机制不变）。
//!
//! ## WGSL 对齐表（M3 落地 ddgi WGSL 时逐字镜像；改一处必改两处）
//! | Rust | WGSL（计划名） | 值 | 语义 |
//! |---|---|---|---|
//! | DDGI_CELL | DDGI_CELL | 16 | cell 边长（fine 体素）= level 2 brick |
//! | IRRADIANCE_TEXELS | IRRADIANCE_TEXELS | 8 | irr oct 边长 |
//! | DEPTH_TEXELS | DEPTH_TEXELS | 16 | depth oct 边长 |
//! | PROBES_PER_LAYER_AXIS | PROBES_PER_LAYER_AXIS | 16 | 层内单轴探针数 |
//! | IRRADIANCE_LAYER_TEXELS | IRRADIANCE_LAYER_TEXELS | 128 | irr 层边长 |
//! | DEPTH_LAYER_TEXELS | DEPTH_LAYER_TEXELS | 256 | depth 层边长 |
//! | DDGI_CASCADE_CELL_SIZES | CASCADE_CELL_SIZES | 16/32/64/128/256 | 各级 cell 边长（[base,LOD1-4]） |
//! | DDGI_AGE_MAX | DDGI_AGE_MAX | 255 | age 饱和值 |
//! | META_OFFSET_BITS / META_AGE_SHIFT / META_OFFSET_QUANT | 同名 | 5 / 15 / 32 | 元数据位域 |
//! | DDGI_NORMAL_BIAS / DDGI_DEPTH_BIAS | 同名 | 0.2 / 16.0 | 采样权重 |
//! | PROBE_T_MAX | PROBE_T_MAX | 8192.0 | 射线远距 |

use bevy::render::render_resource::ShaderType;
use glam::{IVec3, UVec3, Vec3, Vec4};

use gate_voxel::{BrickState, LEVEL_EXTENT, Volumes, VoxelCoord};

// ============================================================================
// 常量（WGSL 侧逐字对齐）
// ============================================================================

/// 探针 cell 边长（fine 体素）= 16³ = level 2 brick = **Douglas #23 LOD=0 绝对
/// 间距**（2026-09-08 活跃过滤 + 遮挡剔除 viz 目验定案：64³ 间距 = Douglas
/// LOD=1——比他稀 4×；最初"16³ 太密"实为 viz 画了全部探针（含天空远场）的
/// 语义偏差，活跃/遮挡过滤后 16³ 才是他的真实密度。放置规则 = cell 树层级
/// BFS 最大空叶 + LOD 下采样）
pub const DDGI_CELL: i32 = 16;
/// cell 对应树层级（LEVEL_EXTENT[2] = 16）
pub const DDGI_CELL_LEVEL: u8 = 2;
/// 每 chunk 的 cell 数（256/16）
pub const CELLS_PER_CHUNK: i32 = gate_voxel::CHUNK_SIZE / DDGI_CELL;
/// 八面体 irradiance 边长（Majercik 2019 §3：8×8 texel/探针）
pub const IRRADIANCE_TEXELS: u32 = 8;
/// 八面体 depth 边长（Majercik 2019 §3：16×16 texel/探针）
pub const DEPTH_TEXELS: u32 = 16;
/// 探针射线 t_max（世界窗口对角量级；douglas-final.md §5 探针只覆盖本模型）
pub const PROBE_T_MAX: f32 = 8192.0;
/// 前后权重锐度（Rohacek §3.2 锐利背面剔除；WGSL DDGI_NORMAL_BIAS 镜像）：
/// wn = clamp(N·d / bias, 0, 1)——探针在表面后侧（N·d<0）权重严格 0，穿墙不漏光
/// （结构性修复见 ddgi_sample_dom per-pixel jitter，而非调 bias）
pub const DDGI_NORMAL_BIAS: f32 = 0.2;
/// 每探针每次更新的射线数下限（WGSL DDGI_PROBE_RAYS_MIN 镜像）：RAY_BUDGET 摊派
/// 低于此值按下限执行。32 而非 16：base 轮换窗口恒钳 4096 探针，16 射线/探针的
/// Monte-Carlo 噪声让邻探针收敛值差一个量级（逐 cell 方块斑驳 + 轮换跳变）。
pub const DDGI_PROBE_RAYS_MIN: u32 = 32;
/// 每帧更新探针钳制上限（WGSL DDGI_PROBE_BUDGET 镜像；seal pass min(count, 此值)）
pub const DDGI_PROBE_BUDGET: u32 = 4096;
/// 漏光 chevron 半宽（fine 单位）＝ 所在域 cell × 0.25；本常量 = base 域（cell 16）
/// 参考值。WGSL ddgi_sample_dom 已改为按采样域逐域计算（级联 32/64/128/256 →
/// 8/16/32/64），固定值对粗级联相对过窄会误剔远角合法探针（网格状黑块）：
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
/// 元数据 packed u32 位域（Douglas #23 截图逐字：offset 量化 ×2 → 5 bit/轴，
/// [0,32) 半体素精度恰覆盖 16³ cell）
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
  [
    in_layer % PROBES_PER_LAYER_AXIS,
    in_layer / PROBES_PER_LAYER_AXIS,
  ]
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

/// 探针元数据打包：offset（cell 内 fine ×2，各 7 bit）+ age（8 bit）
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

/// 解包 (offset_fine2, age)（age 钳 8 bit：[29..32) 保留位忽略，对齐 WGSL bitfieldExtract）
#[inline]
pub fn unpack_probe_meta(packed: u32) -> ([u32; 3], u32) {
  let mask = (1u32 << META_OFFSET_BITS) - 1;
  (
    [
      packed & mask,
      (packed >> META_OFFSET_BITS) & mask,
      (packed >> (META_OFFSET_BITS * 2)) & mask,
    ],
    (packed >> META_AGE_SHIFT) & 0xFF,
  )
}

/// 探针世界位置 → cell 内 offset 量化值（128 quanta/cell；base 64³ cell = ×2 定点，
/// BFS 空叶中心含 .5 半体素 → 精确；级联 cell 分辨率 = cell_size/128 fine）
#[inline]
pub fn quantize_offset_sized(probe_world: Vec3, cell_min: IVec3, cell_size: i32) -> [u32; 3] {
  let rel = probe_world - cell_min.as_vec3();
  let q = (rel * (META_OFFSET_QUANT / cell_size as f32))
    .round()
    .clamp(Vec3::ZERO, Vec3::splat(META_OFFSET_QUANT - 1.0));
  [q.x as u32, q.y as u32, q.z as u32]
}

/// base 版（cell_size=64，×2 定点）
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

/// packed offset_fine2 → 探针世界坐标（cell_min + offset×cell_size/128；
/// base 64 = ×0.5 半体素无损；级联 cell 采样侧按本级 cell_size 解码）
#[inline]
pub fn offset_to_world_sized(offset_fine2: [u32; 3], cell_min: IVec3, cell_size: i32) -> Vec3 {
  cell_min.as_vec3()
    + Vec3::new(
      offset_fine2[0] as f32,
      offset_fine2[1] as f32,
      offset_fine2[2] as f32,
    ) * (cell_size as f32 / META_OFFSET_QUANT)
}

/// base 版（cell_size=64）
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
// 探针烘焙（Douglas #23：cell 树层级 BFS 最大空叶 + D1 全 cell 覆盖）
// ============================================================================

/// 探针烘焙结果（CPU 真相源 → GPU 打包入口；base 与级联级共用）
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeGrid {
  /// 采样网格原点（**base-cell 全局坐标**，DDGI_CELL 单位；级联级须 cell_size 对齐）
  pub grid_origin: IVec3,
  /// 采样网格 dims（**本级 cell 单位**；base cell_size=DDGI_CELL 时与 base-cell 同构）
  pub grid_dims: UVec3,
  /// 本级 cell 尺寸（fine 体素）：base 16，级联 32/64/128/256
  pub cell_size: i32,
  /// dense cell → probe 下标（NO_PROBE = 无探针）
  /// 线性地址 = x + y*dims.x + z*dims.x*dims.y（rel 为本级 cell 坐标）
  pub cell_index: Vec<u32>,
  /// 探针位置（世界 fine 坐标；下标 = probe id）
  pub positions: Vec<Vec3>,
  /// 活跃位（id 对齐 positions；Douglas #23 近表面语义 = 本 cell 或 6 面邻接
  /// cell 含体素，烘焙期 CPU 判定）：probe viz 只画活跃探针（positions.w 携带）；
  /// cast 预算剔除仍由每帧 GPU ddgi_active 判定（多一条非网格对齐物体 bbox
  /// 条件，烘焙期不可知，故烘焙位偏保守 = 只多不少）
  pub active: Vec<bool>,
}

impl Default for ProbeGrid {
  fn default() -> Self {
    Self {
      grid_origin: IVec3::ZERO,
      grid_dims: UVec3::ZERO,
      cell_size: DDGI_CELL,
      cell_index: Vec::new(),
      positions: Vec::new(),
      active: Vec::new(),
    }
  }
}

impl ProbeGrid {
  /// cell（相对 grid_origin 的本级 cell 偏移）→ 线性下标
  #[inline]
  pub fn cell_linear(&self, rel: UVec3) -> usize {
    (rel.x + rel.y * self.grid_dims.x + rel.z * self.grid_dims.x * self.grid_dims.y) as usize
  }

  /// 本级 cell 偏移 → base-cell 全局坐标（cell_size/DDGI_CELL 步长）
  #[inline]
  pub fn cell_base(&self, rel: UVec3) -> IVec3 {
    let step = self.cell_size / DDGI_CELL;
    self.grid_origin + IVec3::new(rel.x as i32, rel.y as i32, rel.z as i32) * step
  }

  /// 本级 cell 的 fine 体素最小角
  #[inline]
  pub fn cell_min_voxel(&self, rel: UVec3) -> IVec3 {
    self.cell_base(rel) * DDGI_CELL
  }
}

/// 烘焙主世界探针网格（当前只覆盖 vols.list[0]；物体互反射探针后置）
///
/// D1 全 cell 覆盖：域 = chunk bbox cell 域，已存在 chunk 的全部非 Solid cell
/// 均放探针（Air 居中 / Mixed BFS 偏移 / Solid 及多色全实无）。纯空气 cell 一律有探针；
/// 烘焙期并行记录活跃位（Douglas 近表面语义，probe viz 过滤用），cast 预算
/// 剔除仍由每帧 ddgi_active 判定（D2，多 object bbox 条件）。
/// 域内未分配 chunk 的 cell 留 NO_PROBE（从未有体素数据的远场，表面不可达）。
/// 稀疏遍历：每个已存在 chunk 迭代其 4³ cell 区，不扫 bbox 全空间。
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
    active: Vec::new(),
  };

  for chunk in &chunks {
    for rz in 0..CELLS_PER_CHUNK {
      for ry in 0..CELLS_PER_CHUNK {
        for rx in 0..CELLS_PER_CHUNK {
          let cell = chunk * CELLS_PER_CHUNK + IVec3::new(rx, ry, rz);
          let rel = (cell - lo).as_uvec3();
          let li = pg.cell_linear(rel);
          let cell_min = cell * DDGI_CELL;
          let state = grid.get_brick_state(VoxelCoord::from_ivec3(cell_min), DDGI_CELL_LEVEL);
          // 全实心 cell 无探针（Douglas：全满 → 无探针）
          if matches!(state, BrickState::Solid(_)) {
            continue;
          }
          // Air → cell 正中；Mixed → 树层级 BFS 最大空叶（#23 字幕：空 cell 居中 /
          // 半满推向空边 / 全满不放）。gate 树 Mixed 是双义的（空实混合 / 多色全实，
          // 见 BrickState 文档）：多色全实 cell 沿树无空叶 → None = DDGI「全满」
          // （palette 不参与）→ 无探针，与 Solid 同途。
          let Some(pos) = (match state {
            BrickState::Air => Some(cell_min.as_vec3() + Vec3::splat(DDGI_CELL as f32 / 2.0)),
            _ => probe_position_in_cell(grid, cell_min, DDGI_CELL_LEVEL).map(|(_, p)| p),
          }) else {
            continue;
          };
          pg.cell_index[li] = pg.positions.len() as u32;
          pg.positions.push(pos);
          pg.active
            .push(probe_is_active(grid, cell_min, DDGI_CELL));
        }
      }
    }
  }
  pg
}

/// 探针位置 + 所在空叶尺寸（Douglas #23 bake 逐字映射）：从 cell 所在树层级出发，
/// 沿**真实树层级**（4³ 细分，LEVEL_EXTENT）BFS 向下找「最大空叶」，探针放叶中心；
/// 同层空叶取靠 cell 中心最近（严格更小才替换 → 平局取遍历序首个 = 靠中心优先遍历）；
/// 全满 → None。返回 (空叶边长 fine, 叶中心)——叶尺寸参与级联父级择优（大者优先）。
///
/// gate 树语义：uniform（Air）节点即叶 → 任一层发现 Air 子块即停（空叶由大到小）；
/// 末层（1³ 体素）扫上层 Mixed 4³ 砖的全部体素取最近空体素。Mixed cell 沿树必有
/// 空叶（子块状态不全同 → 必含 Air 分支），恒 Some。
fn probe_position_in_cell(
  grid: &gate_voxel::VolumeGrid,
  cell_min: IVec3,
  level: u8,
) -> Option<(i32, Vec3)> {
  let center = cell_min.as_vec3() + Vec3::splat(LEVEL_EXTENT[level as usize] as f32 / 2.0);
  // parents = 当前层待下钻的 Mixed 节点（首层 = cell 本身；caller 已排除 Solid）
  let mut parents: Vec<IVec3> = vec![cell_min];
  for lvl in (level + 1)..=4 {
    let ext = LEVEL_EXTENT[lvl as usize];
    let half = ext as f32 / 2.0;
    let mut best: Option<(f32, Vec3)> = None;
    let mut mixed: Vec<IVec3> = Vec::new();
    for p in &parents {
      for k in 0..4 {
        for j in 0..4 {
          for i in 0..4 {
            let sub = *p + IVec3::new(i, j, k) * ext;
            if lvl == 4 {
              // 末层：Mixed 4³ 砖的体素（Mixed 必含空体素）
              if !is_air_voxel(grid, VoxelCoord::from_ivec3(sub)) {
                continue;
              }
              let c = sub.as_vec3() + Vec3::splat(0.5);
              let d2 = c.distance_squared(center);
              if best.is_none_or(|(bd, _)| d2 < bd) {
                best = Some((d2, c));
              }
            } else {
              match grid.get_brick_state(VoxelCoord::from_ivec3(sub), lvl) {
                BrickState::Air => {
                  let c = sub.as_vec3() + Vec3::splat(half);
                  let d2 = c.distance_squared(center);
                  if best.is_none_or(|(bd, _)| d2 < bd) {
                    best = Some((d2, c));
                  }
                }
                BrickState::Mixed => mixed.push(sub),
                BrickState::Solid(_) => {}
              }
            }
          }
        }
      }
    }
    if best.is_some() || lvl == 4 {
      return best.map(|(_, c)| (ext, c));
    }
    parents = mixed;
  }
  None
}

#[inline]
fn is_air_voxel(grid: &gate_voxel::VolumeGrid, v: VoxelCoord) -> bool {
  matches!(grid.get_voxel(v), None | Some(0))
}

/// 区域是否含体素（任一非 Air brick）：**从最粗到最细**取首个整除 cell_size 的
/// 对齐树层级扫描（256→64→16→4→1），查询数最少（16→1 次 level2 / 32→8 次
/// level2 / 64→1 次 level1 / 128→8 次 level1 / 256→1 次 level0）。cell 原点
/// 按 cell_size 对齐，而层级砖粒度整除 cell_size → 扫描坐标天然对齐。
/// （2026-09-08 白屏卡死根因：曾写成 (0..=4).rev() 细到粗——128³ cell 逐体素
/// 扫 200 万次/区域、1024 级联 30 万亿次，首帧烘焙永久阻塞主线程）
fn region_has_voxels(grid: &gate_voxel::VolumeGrid, min: IVec3, size: i32) -> bool {
  for lvl in 0..=4 {
    let ext = LEVEL_EXTENT[lvl as usize];
    if size % ext != 0 {
      continue;
    }
    let n = size / ext;
    for dz in 0..n {
      for dy in 0..n {
        for dx in 0..n {
          let p = min + IVec3::new(dx, dy, dz) * ext;
          if !matches!(
            grid.get_brick_state(VoxelCoord::from_ivec3(p), lvl),
            BrickState::Air
          ) {
            return true;
          }
        }
      }
    }
    return false;
  }
  false
}

/// 探针活跃位（Douglas #23 语义逐字）：本 cell 或 6 面邻接 cell 含体素 → 活跃。
/// 「附近无表面的探针其光照数据不会被用到」——probe viz 只画活跃探针，与
/// WGSL ddgi_active 的 probe_near_surface 前两条件同源（第三条 object bbox
/// 仅运行时可知，烘焙期保守忽略）。只查面邻接不查角邻接（字幕：
/// "left, right, down, up, back, front"）。
pub fn probe_is_active(grid: &gate_voxel::VolumeGrid, cell_min: IVec3, cell_size: i32) -> bool {
  if region_has_voxels(grid, cell_min, cell_size) {
    return true;
  }
  [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
  ]
  .iter()
  .any(|d| region_has_voxels(grid, cell_min + *d * cell_size, cell_size))
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
pub fn cell_state_at(grid: &gate_voxel::VolumeGrid, cell_min: IVec3, cell_size: i32) -> BrickState {
  match cell_size {
    // 树层级直查：256 = level 0 brick、64 = level 1、16 = level 2（DDGI_CELL）
    256 => grid.get_brick_state(VoxelCoord::from_ivec3(cell_min), 0),
    64 => grid.get_brick_state(VoxelCoord::from_ivec3(cell_min), 1),
    16 => grid.get_brick_state(VoxelCoord::from_ivec3(cell_min), DDGI_CELL_LEVEL),
    // 非树层级（32/128）：halves 32→16 / 128→64 必达树层级
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
      probe_position_in_cell(grid, cell_min, DDGI_CELL_LEVEL)
    }
    BrickState::Mixed => {
      let half = cell_size / 2;
      // 关键优化：Air 子 cell 的空叶尺寸 = half，必然大于任何 Mixed 子 cell 下钻叶
      // （≤ half/2）。先扫 8 子 cell 状态——存在 Air 时直接取靠中心最近的 Air 子 cell
      // 中心（零下钻、零 4³ BFS），仅 8 次 cell_state_at（各 1 次树走查）。
      // 全部子 cell 非 Air（全 Mixed/Solid）才下钻（建筑密集区的少数情况）。
      let mut best_air: Option<(f32, Vec3)> = None;
      let mut mixed_subs: Vec<IVec3> = Vec::new();
      for k in 0..2 {
        for j in 0..2 {
          for i in 0..2 {
            let sub_min = cell_min + IVec3::new(i, j, k) * half;
            match cell_state_at(grid, sub_min, half) {
              BrickState::Air => {
                let p = sub_min.as_vec3() + Vec3::splat(half as f32 / 2.0);
                let d2 = p.distance_squared(center);
                if best_air.is_none_or(|(bd, _)| d2 < bd) {
                  best_air = Some((d2, p));
                }
              }
              BrickState::Mixed => mixed_subs.push(sub_min),
              BrickState::Solid(_) => {}
            }
          }
        }
      }
      if let Some((_, p)) = best_air {
        return Some((half, p));
      }
      // 无 Air 子 cell：在 Mixed 子 cell 中下钻择优（空叶大者优先 → 靠中心 → 遍历序）
      let mut best: Option<(i32, f32, Vec3)> = None;
      for sub_min in mixed_subs {
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
      best.map(|(leaf, _, p)| (leaf, p))
    }
  }
}

/// 单 cell 探针烘焙（Solid / Mixed 下钻无空叶 → None；cell_min 由 origin/rel 推导）
fn bake_cascade_cell(
  grid: &gate_voxel::VolumeGrid,
  cell_size: i32,
  origin: IVec3,
  rel: UVec3,
) -> Option<Vec3> {
  let step = cell_size / DDGI_CELL;
  let cell_min = (origin + IVec3::new(rel.x as i32, rel.y as i32, rel.z as i32) * step) * DDGI_CELL;
  if matches!(
    cell_state_at(grid, cell_min, cell_size),
    BrickState::Solid(_)
  ) {
    return None;
  }
  probe_position_sized(grid, cell_min, cell_size)
}

/// 烘焙一个级联级探针网格（D9：cell 尺寸 ×2 递增的相机滚动级 / base 世界级）。
///
/// `origin_cell` 为 **base-cell 全局坐标**（级联级须 cell_size 对齐，M4-3 滚动步进）；
/// `dims_cells` 为本级 cell 单位（滚动级 = PROBES_PER_CASCADE_AXIS³ = 16³）。
/// 全 cell 覆盖语义同 base（D1）：非 Solid cell 一律放探针，活跃位同 base 烘焙
/// 期记录（probe viz 过滤用），cast 剔除交给每帧 ddgi_active。
///
/// **slot 身份 id**（M4-3 目验修订）：级联探针 id = 线性 cell 下标（base 仍是
/// 扫描序）→ meta/irr/depth texel 布局 = cell 网格本身，GPU shifted copy 的
/// 「矩形平移 = cell 平移」假设成立（internal id 布局下矩形搬运会搅乱数据）。
/// positions 按 slot 预分配（空位 Vec3::ZERO，永不寻址），len = cell 总数。
pub fn bake_cascade_grid(
  vols: &Volumes,
  cell_size: i32,
  origin_cell: IVec3,
  dims_cells: UVec3,
) -> ProbeGrid {
  let grid = vols.main();
  let total = (dims_cells.x * dims_cells.y * dims_cells.z) as usize;
  let mut pg = ProbeGrid {
    grid_origin: origin_cell,
    grid_dims: dims_cells,
    cell_size,
    cell_index: vec![NO_PROBE; total],
    positions: vec![Vec3::ZERO; total],
    active: vec![false; total],
  };
  for rz in 0..dims_cells.z {
    for ry in 0..dims_cells.y {
      for rx in 0..dims_cells.x {
        let rel = UVec3::new(rx, ry, rz);
        let li = pg.cell_linear(rel);
        let Some(pos) = bake_cascade_cell(grid, cell_size, origin_cell, rel) else {
          continue;
        };
        pg.cell_index[li] = li as u32;
        pg.positions[li] = pos;
        pg.active[li] = probe_is_active(grid, pg.cell_min_voxel(rel), cell_size);
      }
    }
  }
  pg
}

/// 滚动增量烘焙：重叠 cell 直接搬运旧 slot id/探针位置（零下钻），仅滚入的
/// 新列带现场 `bake_cascade_cell`——滚动帧 CPU 成本从全域 16³ 下钻降到一个/// 列带（~16×，消除滚动帧秒级卡顿）。旧重叠 cell 原本 NO_PROBE（Solid/无
/// 空叶）且世界未变 → 维持 NO_PROBE（确定性一致；世界编辑走全量重烘）。
pub fn bake_cascade_grid_shifted(
  vols: &Volumes,
  cell_size: i32,
  new_origin: IVec3,
  dims_cells: UVec3,
  old: &ProbeGrid,
  shift: IVec3,
) -> ProbeGrid {
  let grid = vols.main();
  let total = (dims_cells.x * dims_cells.y * dims_cells.z) as usize;
  let mut pg = ProbeGrid {
    grid_origin: new_origin,
    grid_dims: dims_cells,
    cell_size,
    cell_index: vec![NO_PROBE; total],
    positions: vec![Vec3::ZERO; total],
    active: vec![false; total],
  };
  let old_dims = old.grid_dims.as_ivec3();
  for rz in 0..dims_cells.z {
    for ry in 0..dims_cells.y {
      for rx in 0..dims_cells.x {
        let rel = UVec3::new(rx, ry, rz);
        let li = pg.cell_linear(rel);
        let old_rel = rel.as_ivec3() - shift;
        if old_rel.cmpge(IVec3::ZERO).all() && old_rel.cmplt(old_dims).all() {
          let old_li = old.cell_linear(old_rel.as_uvec3());
          if old.cell_index[old_li] != NO_PROBE {
            pg.cell_index[li] = li as u32;
            pg.positions[li] = old.positions[old_li];
            pg.active[li] = old.active[old_li];
          }
          continue;
        }
        let Some(pos) = bake_cascade_cell(grid, cell_size, new_origin, rel) else {
          continue;
        };
        pg.cell_index[li] = li as u32;
        pg.positions[li] = pos;
        pg.active[li] = probe_is_active(grid, pg.cell_min_voxel(rel), cell_size);
      }
    }
  }
  pg
}

/// D4 outside_lower_grid：本 LOD cell 是否在更细网格覆盖外（true = 归本 LOD 管）。
/// 全部区间为 base-cell（64³）全局坐标：本 cell [lo, hi) 与更细域 [fo, fo+fd) 无重叠 → outside。
/// 级联滚动步进保证域边缘 cell 对齐（M4-3），部分重叠 cell 归更细级（保守划分）。
/// **WGSL ddgi_active 调用点先把 finer 域向内收缩 2 本级 cell（光晕）再传入**：
/// 否则 finer 窗口边缘内 1 cell 的本级探针零数据（age=0 永不更新），而采样端
/// 过渡带恰在该带与本级混合 → mix(黑) = 跟随相机的纯黑环带 + cast 自闭环毒化扩散。
/// 本谓词保持纯几何语义（单测覆盖）；光晕收缩只发生在 WGSL 调用点。
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
// M4-3 级联滚动（CPU 侧滚动数学；WGSL 侧 reuse 继承已在 ddgi_active）
// ============================================================================

/// 相机 fine 坐标 → 级联滚动原点（**base-cell 全局坐标**，cell_size 对齐）。
/// 窗口以相机所在 cell 为中心对称展开（PROBES_PER_CASCADE_AXIS³）：
/// origin_cell = (⌊cam/cell_size⌋ − half) × cell_size / DDGI_CELL。
/// 相机移动不足一个 cell 时原点不动（cell 对齐步进 = 天然 reuse 分带）。
#[inline]
pub fn cascade_scroll_origin(cam_fine: Vec3, cell_size: i32) -> IVec3 {
  let cam_cell = (cam_fine / cell_size as f32).floor().as_ivec3();
  let half = (PROBES_PER_CASCADE_AXIS / 2) as i32;
  (cam_cell - half) * cell_size / DDGI_CELL
}

/// 滚动 reuse bounds（**新网格 rel 坐标**，半开 [lo, hi) 逐轴）：
/// shift = (new_origin − old_origin) / (cell_size/16)（级联 cell 单位）；
/// 新 rel 的世界 cell 与旧网格重叠 ⟺ rel − shift ∈ [0, dims) ⟺ rel ∈ [shift, dims+shift)，
/// 与 [0, dims) 交 = [max(shift,0), min(dims+shift, dims))。
/// ddgi_active 对 reusable 探针继承 prev age（D10），窗外归零重新收敛。
#[inline]
pub fn scroll_reuse_bounds(
  old_origin: IVec3,
  new_origin: IVec3,
  dims: UVec3,
  cell_size: i32,
) -> (IVec3, IVec3) {
  let shift = (new_origin - old_origin) / (cell_size / DDGI_CELL);
  let d = dims.as_ivec3();
  let lo = shift.max(IVec3::ZERO);
  let hi = (d + shift).min(d).max(lo);
  (lo, hi)
}

/// M4-3 级联滚动管理器（CPU 侧）：4 级滚动级联的 origin/网格/meta/reuse 快照。
/// 滚动时逐级：增量烘焙（`bake_cascade_grid_shifted`，重叠搬运/新列下钻）→
/// meta 重映射（重叠 cell 搬运旧 age/offset，新 cell age=0）→ 更新 reuse bounds。
/// GPU 接线（positions/cell_index/meta 上传 + irr/depth 分层 shifted copy）消费
/// 本结构快照。级联 id = slot（线性 cell 下标），texel 布局 = cell 网格。
pub struct CascadeManager {
  /// 级联 cell 尺寸（DDGI_CASCADE_CELL_SIZES[1..5] = 128/256/512/1024）
  pub cell_sizes: [i32; 4],
  /// 每级当前原点（base-cell 全局坐标）
  pub origins: [IVec3; 4],
  /// 每级当前探针网格（本级 cell 单位 16³）
  pub grids: [ProbeGrid; 4],
  /// 每级当前 meta（4096 word，packed offset+age；层布局 = build_meta_texture_data）
  pub metas: [Vec<u32>; 4],
  /// 每级当前 reuse bounds（新网格 rel 坐标，半开）
  pub reuse: [(IVec3, IVec3); 4],
  /// 本帧滚动位移（级联 cell 单位，extract 侧 scroll 写入、dispatch 侧 shifted
  /// copy 消费后清空）；None = 本级未动。prepare 读它触发 positions/ci/meta 上传。
  pub pending_shift: [Option<IVec3>; 4],
}

impl CascadeManager {
  /// 初建：以相机位置烘焙 4 级（age 全 0，reuse 全域）
  pub fn new(vols: &Volumes, cam_fine: Vec3) -> Self {
    let cell_sizes: [i32; 4] = DDGI_CASCADE_CELL_SIZES[1..5].try_into().unwrap();
    let dims = UVec3::splat(PROBES_PER_CASCADE_AXIS);
    let mut origins = [IVec3::ZERO; 4];
    let mut grids: [ProbeGrid; 4] = Default::default();
    let mut metas = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let reuse = [(IVec3::ZERO, dims.as_ivec3()); 4];
    for c in 0..4 {
      origins[c] = cascade_scroll_origin(cam_fine, cell_sizes[c]);
      grids[c] = bake_cascade_grid(vols, cell_sizes[c], origins[c], dims);
      metas[c] = build_meta_texture_data(&grids[c]);
    }
    Self {
      cell_sizes,
      origins,
      grids,
      metas,
      reuse,
      pending_shift: [None; 4],
    }
  }

  /// 滚动：相机驱动的逐级原点推进。返回是否有任何一级移动（GPU 上传/搬运触发用）。
  /// 每级：原点未动 → 仅更新 reuse=全域；移动 → 重烘 + meta 重映射（重叠 cell
  /// 搬运旧 age/offset，同世界 cell 烘焙确定性保证 offset 一致）+ 新 reuse bounds。
  /// **每帧至多滚一级（最细优先）**：滚动帧的 CPU 重烘 + GPU 上传/搬运集中在
  /// 一级，避免同帧多级齐滚造成卡顿；粗级顺延一帧（原点不动 → 下帧重算）。
  pub fn scroll(&mut self, vols: &Volumes, cam_fine: Vec3) -> bool {
    let dims = UVec3::splat(PROBES_PER_CASCADE_AXIS);
    let mut moved = false;
    for c in 0..4 {
      let new_origin = cascade_scroll_origin(cam_fine, self.cell_sizes[c]);
      if new_origin == self.origins[c] || moved {
        self.reuse[c] = (IVec3::ZERO, dims.as_ivec3());
        self.pending_shift[c] = None;
        continue;
      }
      let bounds = scroll_reuse_bounds(self.origins[c], new_origin, dims, self.cell_sizes[c]);
      let shift = (new_origin - self.origins[c]) / (self.cell_sizes[c] / DDGI_CELL);
      // 增量烘焙：重叠 cell 搬运（零下钻），仅滚入列带现场烘焙（slot id 布局下
      // GPU shifted copy 的矩形平移假设成立）
      let new_grid = bake_cascade_grid_shifted(
        vols,
        self.cell_sizes[c],
        new_origin,
        dims,
        &self.grids[c],
        shift,
      );
      // meta 重映射：新 bake 基线（age=0）→ 重叠 cell 从旧 meta 搬运 (offset, age)
      let mut new_meta = build_meta_texture_data(&new_grid);
      for rz in 0..dims.z {
        for ry in 0..dims.y {
          for rx in 0..dims.x {
            let rel = UVec3::new(rx, ry, rz);
            let new_id = new_grid.cell_index[new_grid.cell_linear(rel)];
            if new_id == NO_PROBE {
              continue;
            }
            let old_rel = rel.as_ivec3() - shift;
            if old_rel.cmpge(IVec3::ZERO).all() && old_rel.cmplt(dims.as_ivec3()).all() {
              let old_id = self.grids[c].cell_index[self.grids[c].cell_linear(old_rel.as_uvec3())];
              if old_id != NO_PROBE {
                // 同世界 cell → 烘焙确定性 → offset 相同，仅搬运 age
                let (layer, tx, ty) = meta_texel_coord(old_id);
                let (_, age) = unpack_probe_meta(self.metas[c][meta_texel_linear(layer, tx, ty)]);
                let (n_layer, n_tx, n_ty) = meta_texel_coord(new_id);
                let (off, _) = unpack_probe_meta(new_meta[meta_texel_linear(n_layer, n_tx, n_ty)]);
                new_meta[meta_texel_linear(n_layer, n_tx, n_ty)] = pack_probe_meta(off, age);
              }
            }
          }
        }
      }
      self.origins[c] = new_origin;
      self.grids[c] = new_grid;
      self.metas[c] = new_meta;
      self.reuse[c] = bounds;
      self.pending_shift[c] = Some(shift);
      moved = true;
      // #region debug-point D:scroll-mark（假设 D：滚动帧扰动与闪烁时刻相关性）
      bevy::log::info!(
        "[DEBUG][D] cascade c{} scroll shift=({},{},{}) cell_size={}（本帧滚动级）",
        c,
        shift.x,
        shift.y,
        shift.z,
        self.cell_sizes[c]
      );
      // #endregion
    }
    moved
  }
}

/// 单轴滚动位移 → 重叠区 (新格 lo, 宽)（本级 cell 单位；None = 无重叠整轴复位）。
/// shift = 新格 − 旧格：新 rel ∈ [max(0,s), min(dims, dims+s)) 与旧 rel − s 对齐；
/// `dims` = PROBES_PER_CASCADE_AXIS（16）。irr/depth shifted copy 与新列复位共用。
#[inline]
pub fn axis_shift_range(s: i32, dims: i32) -> Option<(i32, i32)> {
  if s.abs() >= dims {
    return None; // 一帧跨整窗：无重叠
  }
  let lo = s.max(0);
  let hi = (dims + s).min(dims);
  Some((lo, hi - lo))
}

/// dense cell_index → 全局探针 id 视图（NO_PROBE 保留，其余 + `id_base` 偏移）。
/// 级联 GPU 上传（BG binding 2 本级 ci / binding 14 all_ci 段）共用，
/// 使 ddgi_flags_cell/active/worklist/采样全链用同一全局 id 空间。
#[inline]
pub fn cascade_ci_global(cell_index: &[u32], id_base: u32) -> Vec<u32> {
  cell_index
    .iter()
    .map(|&v| if v == NO_PROBE { NO_PROBE } else { id_base + v })
    .collect()
}

// ============================================================================
// 级联域（每帧滚动 dispatch 的域描述；光照判定逻辑在 WGSL ddgi_active）
// ============================================================================

/// 级联域（每帧滚动 dispatch 用；origin/dims 为 base-cell 全局坐标，cell_size 为 fine 体素）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CascadeDomain {
  /// 域原点（base-cell 全局坐标；级联级须 cell_size 对齐——M4-3 滚动步进）
  pub origin: IVec3,
  /// 域 dims（本级 cell 单位；滚动级 = PROBES_PER_CASCADE_AXIS³=16³，base = AABB cell 数）
  pub dims: UVec3,
  /// cell 尺寸（fine 体素）：base 64 / LOD1-4=128/256/512/1024
  pub cell_size: i32,
}

impl CascadeDomain {
  /// 域 hi（base-cell 半开）= origin + dims × (cell_size / DDGI_CELL)
  #[inline]
  pub fn hi_base_cell(&self) -> IVec3 {
    self.origin + self.dims.as_ivec3() * (self.cell_size / DDGI_CELL)
  }
}

/// 全 reuse bounds（base 静态级默认值：覆盖全部 cell，全部 reusable）
pub const REUSE_ALL: (IVec3, IVec3) = (IVec3::ZERO, IVec3::splat(i32::MAX));

/// 探针位置打包：xyz = 世界 fine 坐标，w = 活跃位（1.0 活跃 / 0 不活跃；
/// Douglas #23 近表面语义，probe viz 只画 w≠0 的探针；采样/cast 不读 w）
pub fn pack_probe_positions(pg: &ProbeGrid) -> Vec<Vec4> {
  pg.positions
    .iter()
    .zip(&pg.active)
    .map(|(p, &a)| Vec4::new(p.x, p.y, p.z, if a { 1.0 } else { 0.0 }))
    .collect()
}

// ============================================================================
// 渲染侧：BG4 布局 + 烘焙提取 + GPU 资源 + 每帧 prepare（DdgiPlugin）
// ============================================================================

use bevy::asset::AssetServer;
use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::ecs::system::{Commands, Res, ResMut};
use bevy::prelude::RenderGraph;
use bevy::render::{
  Render, RenderApp, RenderStartup, RenderSystems,
  diagnostic::RecordDiagnostics,
  render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    Buffer, BufferBindingType, BufferDescriptor, BufferUsages, CachedComputePipelineId,
    ComputePassDescriptor, ComputePipelineDescriptor, Extent3d, MapMode, Origin3d, ShaderStages,
    StorageTextureAccess, TexelCopyBufferInfo, TexelCopyBufferLayout, TexelCopyTextureInfo,
    Texture, TextureAspect, TextureDescriptor, TextureDimension, TextureFormat, TextureSampleType,
    TextureUsages, TextureView, TextureViewDescriptor, TextureViewDimension, UniformBuffer,
  },
  renderer::{RenderDevice, RenderQueue},
};
use std::borrow::Cow;

/// BG4 v2 uniform（dda.wgsl `DdgiUniform` 逐字段镜像，112B）：
/// grid_origin.w = cell 边长、grid_dims.w = 探针数、params = (frame, rays 观测位,
/// 保留, object bbox 数)、finer_min.w <= 0 = 无更细级（rays_per_probe 由 GPU 侧
/// 从 dispatch count 自派生，无需 CPU 回读）。
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bevy::ecs::resource::Resource, ShaderType)]
pub struct DdgiUniform {
  pub grid_origin: Vec4,
  pub grid_dims: Vec4,
  pub params: Vec4,
  pub reuse_min: Vec4,
  pub reuse_max: Vec4,
  pub finer_min: Vec4,
  pub finer_size: Vec4,
}

impl DdgiUniform {
  /// ProbeGrid + 帧状态 + 调试参数 → uniform（M4-3 滚动后 reuse/finer 由 CPU 每帧覆写）
  ///
  /// mode/gain 来自渲染世界 DdgiDebugSettings（主世界 DebugView UI 驱动，extract
  /// 每帧拷贝），不再读 GATE_DDGI_DEBUG / GATE_DDGI_GAIN 环境变量。
  pub fn new(
    pg: &ProbeGrid,
    frame: u32,
    reuse_bounds: (IVec3, IVec3),
    finer: Option<CascadeDomain>,
    object_count: u32,
    mode: f32,
    gain: f32,
  ) -> Self {
    Self {
      grid_origin: Vec4::new(
        pg.grid_origin.x as f32,
        pg.grid_origin.y as f32,
        pg.grid_origin.z as f32,
        pg.cell_size as f32,
      ),
      grid_dims: Vec4::new(
        pg.grid_dims.x as f32,
        pg.grid_dims.y as f32,
        pg.grid_dims.z as f32,
        pg.positions.len() as f32,
      ),
      params: Vec4::new(frame as f32, mode, gain, object_count as f32),
      reuse_min: Vec4::new(
        reuse_bounds.0.x as f32,
        reuse_bounds.0.y as f32,
        reuse_bounds.0.z as f32,
        0.0,
      ),
      reuse_max: Vec4::new(
        reuse_bounds.1.x as f32,
        reuse_bounds.1.y as f32,
        reuse_bounds.1.z as f32,
        0.0,
      ),
      finer_min: finer.map_or(Vec4::ZERO, |f| {
        Vec4::new(
          f.origin.x as f32,
          f.origin.y as f32,
          f.origin.z as f32,
          f.cell_size as f32,
        )
      }),
      finer_size: finer.map_or(Vec4::ZERO, |f| {
        Vec4::new(f.dims.x as f32, f.dims.y as f32, f.dims.z as f32, 0.0)
      }),
    }
  }
}

/// BG4 v3 全域采样 uniform（dda.wgsl `DdgiDomains` 逐字段镜像，560B = base +
/// 4 级级联各 112B；binding 13，ddgi_sample 的 base 优先→级联 fallback 数据源）。
/// binding 0（本级 pass uniform）与之并存：base BG4 绑 base 域、级联 BG4_c 绑本级域。
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bevy::ecs::resource::Resource, ShaderType)]
pub struct DdgiDomains {
  pub base: DdgiUniform,
  pub cascades: [DdgiUniform; 4],
}

/// BG4 v3 布局（dda.wgsl group(4) 15 binding 逐字镜像；改 shader 必同步此处）：
/// 0=uniform(本级域) 1=positions(ro) 2=cell_index(ro,本级 dense) 3/4=irr/depth_prev
/// (纹理数组采样读) 5/6=irr/depth_next(storage write) 7/8=meta prev/next(r32uint)
/// 9=dispatch(rw atomic) 10=objects(ro) 11=samples(rw) 12=worklist(rw)
/// 13=uniform(全域 DdgiDomains 560B) 14=all_ci(ro,全域 dense cell_index)
pub fn ddgi_bg4_layout() -> BindGroupLayoutDescriptor {
  const C: ShaderStages = ShaderStages::COMPUTE;
  let tex = |binding: u32, sample_type: TextureSampleType| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Texture {
      sample_type,
      view_dimension: TextureViewDimension::D2Array,
      multisampled: false,
    },
    count: None,
  };
  let store = |binding: u32, format: TextureFormat| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::StorageTexture {
      access: StorageTextureAccess::WriteOnly,
      format,
      view_dimension: TextureViewDimension::D2Array,
    },
    count: None,
  };
  let buf = |binding: u32, read_only: bool| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "DdgiBg4v2",
    &[
      BindGroupLayoutEntry {
        binding: 0,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(DdgiUniform::min_size()),
        },
        count: None,
      },
      buf(1, true),
      buf(2, true),
      tex(3, TextureSampleType::Float { filterable: true }), // irr prev（rgba16f）
      tex(4, TextureSampleType::Float { filterable: false }), // depth prev（r32）
      store(5, TextureFormat::Rgba16Float),
      store(6, TextureFormat::R32Float),
      tex(7, TextureSampleType::Uint), // meta prev（r32uint）
      store(8, TextureFormat::R32Uint),
      buf(9, false),
      buf(10, true),
      buf(11, false),
      buf(12, false),
      BindGroupLayoutEntry {
        binding: 13,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(DdgiDomains::min_size()),
        },
        count: None,
      },
      buf(14, true),
    ],
  )
}

/// DDGI compute 管线（dda.wgsl 五 entry；布局 = BG0-3（dda 复用）+ BG4 v3）
#[derive(Debug, Clone, Copy)]
pub struct DdgiPipelines {
  pub clear: CachedComputePipelineId,
  pub active: CachedComputePipelineId,
  pub seal: CachedComputePipelineId,
  pub cast: CachedComputePipelineId,
  pub update: CachedComputePipelineId,
}

/// M4-3 级联 GPU 资源槽（每级联独立；纹理单张大数组按层偏移共享）
pub struct CascadeGpu {
  pub uniform: UniformBuffer<DdgiUniform>,
  pub cell_index: Buffer,
  pub worklist: Buffer,
  pub samples: Buffer,
  pub dispatch: Buffer,
  pub indirect: Buffer,
  pub bg4: Option<BindGroup>,
  /// 当前级联探针数（= 非 Solid cell 数，随滚动变化）
  pub probe_count: u32,
}

/// Extract 产物：烘焙好的探针网格 + 对应的世界编辑代数（prepare 消费后移除）
#[derive(bevy::ecs::resource::Resource)]
pub struct ProbeBake(pub ProbeGrid, pub u64);

/// 渲染 world 持久 DDGI GPU 资源（RenderStartup 建 1×1 占位纹理/4B 空缓冲，
/// 烘焙后重建；D7 双缓冲 ping-pong：prev 采样读 / next 更新写，帧末交换指针 M4-2）
#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiGpu {
  pub uniform: UniformBuffer<DdgiUniform>,
  pub positions: Buffer,
  pub cell_index: Buffer,
  pub objects: Buffer,
  /// [count, 1, 1, 0]；[0] 由 ddgi_active atomicAdd。**不能**兼 indirect——同
  /// dispatch scope 内 STORAGE 与 INDIRECT 互斥（wgpu 验证），indirect 走独立 buffer。
  pub dispatch: Buffer,
  /// indirect 专用（COPY_DST + INDIRECT）：active 后由 encoder 从 dispatch 整拷 16B
  pub indirect: Buffer,
  pub worklist: Buffer,
  pub samples: Buffer,
  pub irr_prev: Texture,
  pub irr_next: Texture,
  pub depth_prev: Texture,
  pub depth_next: Texture,
  pub meta_prev: Texture,
  pub meta_next: Texture,
  pub irr_prev_view: TextureView,
  pub irr_next_view: TextureView,
  pub depth_prev_view: TextureView,
  pub depth_next_view: TextureView,
  pub meta_prev_view: TextureView,
  pub meta_next_view: TextureView,
  /// 纹理数组层数（= meta_texture_layers(probe_count)；级联 64 层另计于纹理尺寸）
  pub layers: u32,
  /// base 层数（级联层偏移基址；纹理总层 = base_layers + 64）
  pub base_layers: u32,
  pub probe_count: u32,
  /// cell 网格（dispatch_ddgi 的 active dispatch 数 + uniform 模板来源）
  pub grid_origin: IVec3,
  pub grid_dims: UVec3,
  /// 已推进帧号（prepare 自增）
  pub frame: u32,
  /// 当前 GPU 探针数据对应的世界编辑代数（u64::MAX = 尚未烘焙；编辑后触发重烘）
  pub baked_generation: u64,
  /// 四 entry 管线（DdaPipelines 就绪后排队一次）
  pub pipelines: Option<DdgiPipelines>,
  /// 诊断回读（每 120 帧三阶段：copy → map_async → 读+unmap；wgpu 规则：
  /// submit 时 buffer 不得处于 Pending/Mapped，三阶段保证每次 submit 时 Unmapped）
  pub readback: Buffer,
  pub readback_rx: std::sync::Mutex<Option<std::sync::mpsc::Receiver<()>>>,
  /// 0=idle 1=copied 2=mapped(等回调)
  pub readback_state: u32,
  /// 诊断回读采样层距（copy 时 = (base_layers+64)/8 定格；解析侧据此还原层号，
  /// 防烘焙改变纹理层数后 copy/parse 错位）
  pub readback_step: u32,
  /// M4-3 级联 CPU 管理器（extract 侧滚动驱动；None = 尚未烘焙）
  pub manager: Option<CascadeManager>,
  /// 级联探针 id 基址 = base_layers × 256（id 空间偏移 → WGSL 坐标函数零改动）
  pub id_base: u32,
  /// 级联 GPU 槽（[0]=LOD1 … [3]=LOD4）
  pub cascades: Vec<CascadeGpu>,
  /// BG4 binding 13：全域采样 uniform（base + 4 级级联域参数，560B）
  pub casc_u: UniformBuffer<DdgiDomains>,
  /// BG4 binding 14：全域 dense cell_index（base_ci ++ 级联 ci，级联项已平移全局 id）
  pub all_ci: Buffer,
  /// base dense cell_index 字数（all_ci 中级联区起始偏移）
  pub base_ci_words: u32,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiBg4(pub BindGroup);

/// DDGI 运行时总开关（main world 由 UI toggle 写入；extract 拷到 render world）。
/// 关：dispatch_ddgi 整条 compute 链早退（省 GPU）+ trace shader 跳过探针采样
/// （gi=0）。默认开。
#[derive(bevy::ecs::resource::Resource, Clone, Copy)]
pub struct DdgiEnabled(pub bool);

impl Default for DdgiEnabled {
  fn default() -> Self {
    Self(true)
  }
}

/// DDGI 运行时调试参数（主世界 DebugView UI 写入 → extract 拷到渲染世界 →
/// prepare_ddgi 每帧写进 DdgiUniform.params.y/.z）。
///
/// 取代原 GATE_DDGI_DEBUG / GATE_DDGI_GAIN 环境变量：运行时可热切换，无需重启。
/// - mode：params.y，0=正常，1=GI 提亮，2=wsum 热度，3=选域 id，4=探针状态
/// - gain：params.z，诊断增益（调参/链路定位用）
/// - probe_viz：探针位置可视化开关（devlog #23 风格黄色方块）
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq)]
pub struct DdgiDebugSettings {
  pub mode: f32,
  pub gain: f32,
  pub probe_viz: bool,
  /// probe viz 层级选择（0=All 全部, 1..=4=LOD0~3 滚动级联 32/64/128/256,
  /// 5=Base 16³ 世界烘焙网格）。Douglas #23 = base 烘焙网格 + 4 LOD 同构，
  /// 他视频切换的 LOD 0~3 即这 4 个下采样级（base 不占 LOD 编号）
  pub probe_viz_lod: f32,
}

impl Default for DdgiDebugSettings {
  fn default() -> Self {
    Self {
      mode: 0.0,
      gain: 1.0,
      probe_viz: false,
      probe_viz_lod: 0.0,
    }
  }
}

/// DDGI 插件：main world VoxelScene → 一次性烘焙 → GPU 纹理数组/buffer/BG4 + 管线
pub struct DdgiPlugin;

impl bevy::app::Plugin for DdgiPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    app.init_resource::<DdgiEnabled>();
    app.init_resource::<DdgiDebugSettings>();
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };
    render_app
      .init_resource::<DdgiEnabled>()
      .init_resource::<DdgiDebugSettings>()
      .add_systems(RenderStartup, init_ddgi_gpu)
      .add_systems(bevy::render::ExtractSchedule, extract_ddgi_bake)
      .add_systems(
        Render,
        (queue_ddgi_pipelines, prepare_ddgi).in_set(RenderSystems::PrepareBindGroups),
      )
      // M4-2：四 pass 链在主 trace（dispatch_dda）之前编码
      .add_systems(
        RenderGraph,
        dispatch_ddgi
          .in_set(bevy::render::renderer::RenderGraphSystems::Render)
          .before(crate::brickmap::dda::dispatch_dda),
      );
  }
}

/// 4B 占位 storage buffer（管线布局始终可绑；probe_count=0 时 ddgi pass 早退）
fn dummy_buffer(device: &RenderDevice, label: &str) -> Buffer {
  dummy_sized_buffer(device, label, 4)
}

fn dummy_sized_buffer(device: &RenderDevice, label: &str, size: u64) -> Buffer {
  device.create_buffer(&BufferDescriptor {
    label: Some(label.into()),
    size: size.max(4),
    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  })
}

/// indirect dispatch 专用 buffer（INDIRECT + COPY_DST；与 storage 绑定互斥故独立）
fn dummy_indirect_buffer(device: &RenderDevice, label: &str) -> Buffer {
  device.create_buffer(&BufferDescriptor {
    label: Some(label.into()),
    size: 16,
    usage: BufferUsages::INDIRECT | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  })
}

/// 诊断 staging（16B dispatch + 8 irr 采样层 + 1 depth 采样层 + 8 meta 采样层
/// ——黑探针普查用：逐探针 age × irr 非零分类）；MAP_READ + COPY_DST
fn ddgi_readback_buffer(device: &RenderDevice) -> Buffer {
  device.create_buffer(&BufferDescriptor {
    label: Some("ddgi_readback".into()),
    size: 256u64
      + (8 * IRRADIANCE_LAYER_TEXELS * IRRADIANCE_LAYER_TEXELS * 8) as u64
      + (DEPTH_LAYER_TEXELS * DEPTH_LAYER_TEXELS * 4) as u64
      // meta 8 采样层：256B 行距 × 16 行/层（COPY_BYTES_PER_ROW_ALIGNMENT，
      // 行宽 64B 不达标 → Validation Error）
      + (8 * (META_COPY_BPR as u64 * PROBES_PER_LAYER_AXIS as u64))
      + 64,
    usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
    mapped_at_creation: false,
  })
}

fn ddgi_array_view(tex: &Texture) -> TextureView {
  tex.create_view(&TextureViewDescriptor {
    dimension: Some(TextureViewDimension::D2Array),
    ..Default::default()
  })
}

/// 纹理数组（D7 载体；usage = 采样读 + storage 写 + 初始数据上传）
fn ddgi_array_tex(
  device: &RenderDevice,
  label: &str,
  format: TextureFormat,
  size: (u32, u32),
  layers: u32,
) -> Texture {
  device.create_texture(&TextureDescriptor {
    label: Some(label.into()),
    size: Extent3d {
      width: size.0,
      height: size.1,
      depth_or_array_layers: layers,
    },
    mip_level_count: 1,
    sample_count: 1,
    dimension: TextureDimension::D2,
    format,
    usage: TextureUsages::TEXTURE_BINDING
      | TextureUsages::STORAGE_BINDING
      | TextureUsages::COPY_DST
      | TextureUsages::COPY_SRC, // ping-pong copy 的 source 侧
    view_formats: &[],
  })
}

fn init_ddgi_gpu(mut commands: Commands, device: Res<RenderDevice>) {
  let (irr, irr_v) = {
    let t = ddgi_array_tex(
      &device,
      "ddgi_irr(empty)",
      TextureFormat::Rgba16Float,
      (4, 4),
      1,
    );
    let v = ddgi_array_view(&t);
    (t, v)
  };
  let (dep, dep_v) = {
    let t = ddgi_array_tex(
      &device,
      "ddgi_depth(empty)",
      TextureFormat::R32Float,
      (4, 4),
      1,
    );
    let v = ddgi_array_view(&t);
    (t, v)
  };
  let (meta, meta_v) = {
    let t = ddgi_array_tex(
      &device,
      "ddgi_meta(empty)",
      TextureFormat::R32Uint,
      (4, 4),
      1,
    );
    let v = ddgi_array_view(&t);
    (t, v)
  };
  commands.insert_resource(DdgiGpu {
    uniform: UniformBuffer::default(),
    positions: dummy_buffer(&device, "ddgi_positions(empty)"),
    cell_index: dummy_buffer(&device, "ddgi_cell_index(empty)"),
    objects: dummy_buffer(&device, "ddgi_objects(empty)"),
    dispatch: dummy_buffer(&device, "ddgi_dispatch(empty)"),
    indirect: dummy_indirect_buffer(&device, "ddgi_indirect(empty)"),
    worklist: dummy_buffer(&device, "ddgi_worklist(empty)"),
    samples: dummy_buffer(&device, "ddgi_samples(empty)"),
    readback: ddgi_readback_buffer(&device),
    readback_rx: std::sync::Mutex::new(None),
    readback_state: 0,
    readback_step: 1,
    manager: None,
    id_base: 0,
    cascades: Vec::new(),
    casc_u: UniformBuffer::default(),
    all_ci: dummy_buffer(&device, "ddgi_all_ci(empty)"),
    base_ci_words: 0,
    irr_prev: irr.clone(),
    irr_next: irr,
    depth_prev: dep.clone(),
    depth_next: dep,
    meta_prev: meta.clone(),
    meta_next: meta,
    irr_prev_view: irr_v.clone(),
    irr_next_view: irr_v,
    depth_prev_view: dep_v.clone(),
    depth_next_view: dep_v,
    meta_prev_view: meta_v.clone(),
    meta_next_view: meta_v,
    layers: 1,
    base_layers: 1,
    probe_count: 0,
    grid_origin: IVec3::ZERO,
    grid_dims: UVec3::ONE,
    frame: 0,
    baked_generation: u64::MAX,
    pipelines: None,
  });
}

/// 四 entry 管线排队（一次；BG0-3 复用 dda 布局 + BG4 v2；layout 收 Descriptor）
fn queue_ddgi_pipelines(
  dda: Option<Res<crate::brickmap::dda::DdaPipelines>>,
  pipeline_cache: Res<bevy::render::render_resource::PipelineCache>,
  asset_server: Res<AssetServer>,
  mut gpu: ResMut<DdgiGpu>,
) {
  if gpu.pipelines.is_some() {
    return;
  }
  let Some(dda) = dda else {
    return;
  };
  let layout = vec![
    dda.bg0_layout.clone(),
    dda.bg1_layout.clone(),
    dda.bg2_layout.clone(),
    dda.bg3_layout.clone(),
    ddgi_bg4_layout(),
  ];
  let shader = asset_server.load(crate::brickmap::dda::DDA_SHADER_ASSET_PATH);
  let mk = |label: &'static str, entry: &'static str| {
    pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
      label: Some(Cow::from(label)),
      layout: layout.clone(),
      shader: shader.clone(),
      entry_point: Some(Cow::from(entry)),
      ..Default::default()
    })
  };
  gpu.pipelines = Some(DdgiPipelines {
    clear: mk("gate_ddgi_clear", "ddgi_clear"),
    active: mk("gate_ddgi_active", "ddgi_active"),
    seal: mk("gate_ddgi_seal", "ddgi_seal"),
    cast: mk("gate_ddgi_cast", "ddgi_cast"),
    update: mk("gate_ddgi_update", "ddgi_update"),
  });
}

/// 四 pass 编排（M4-2；在 dispatch_dda 主 trace 之前执行）：
/// copy（prev→next 重填，encoder 级无 pass 冲突）→ ①clear → ②active（间接计
/// 数 atomicAdd）→ ③cast（indirect）→ ④update（indirect）。
/// race 纪律：pass 间读写同 storage/texture 必须分 pass（wgpu 只在 pass 边界插
/// barrier）；clear→active 同 buffer 亦然。
/// ping-pong 语义：本帧 prev = 上一帧 update 产物（cast 的 ddgi_sample 读它 =
/// 「上一帧 DDGI 输出」自闭环）；next 经 copy 重填后只被本帧处理探针覆写，帧末
/// prepare 交换指针。
#[allow(clippy::too_many_arguments)]
fn dispatch_ddgi(
  mut ctx: bevy::render::renderer::RenderContext,
  bg0: Option<Res<crate::brickmap::dda::DdaBg0BindGroup>>,
  bg1: Option<Res<crate::brickmap::dda::DdaBg1BindGroup>>,
  bg2: Option<Res<crate::brickmap::dda::DdaBg2BindGroup>>,
  bg3: Option<Res<crate::brickmap::dda::DdaBg3BindGroup>>,
  bg4: Option<Res<DdgiBg4>>,
  enabled: Res<DdgiEnabled>,
  mut gpu: ResMut<DdgiGpu>,
  pipeline_cache: Res<bevy::render::render_resource::PipelineCache>,
) {
  // 运行时关：跳过整条 compute 链（clear/active/seal/cast/update + 级联），省 GPU。
  // trace shader 侧由 uniform 开关位同步返回 gi=0。仍排空 pending_shift 防位移堆积
  // （关闭期间 extract 已冻结 scroll，此处仅兜底清掉切换当帧可能残留的位移）。
  if !enabled.0 {
    if let Some(cm) = gpu.manager.as_mut() {
      let _ = std::mem::take(&mut cm.pending_shift);
    }
    return;
  }
  let (Some(bg0), Some(bg1), Some(bg2), Some(bg3), Some(bg4)) = (
    bg0.as_ref(),
    bg1.as_ref(),
    bg2.as_ref(),
    bg3.as_ref(),
    bg4.as_ref(),
  ) else {
    return;
  };
  if gpu.probe_count == 0 {
    return;
  }
  let Some(pipes) = gpu.pipelines else {
    return;
  };
  let (Some(p_clear), Some(p_active), Some(p_seal), Some(p_cast), Some(p_update)) = (
    pipeline_cache.get_compute_pipeline(pipes.clear),
    pipeline_cache.get_compute_pipeline(pipes.active),
    pipeline_cache.get_compute_pipeline(pipes.seal),
    pipeline_cache.get_compute_pipeline(pipes.cast),
    pipeline_cache.get_compute_pipeline(pipes.update),
  ) else {
    bevy::log::debug_once!("DDGI dispatch: pipelines not ready");
    return;
  };

  // M4-3：捕获并消费本帧级联滚动位移（extract 侧 scroll 写入；prepare 已完成
  // positions/ci/meta 上传与新列复位）。None = 本级未动（shift=0 → 整层 copy）。
  let pending_shift: [Option<IVec3>; 4] =
    gpu.manager.as_mut().map_or([None, None, None, None], |m| {
      std::mem::take(&mut m.pending_shift)
    });

  // ---- copy：base 区 prev→next 三对整体重填 + 级联区 shifted copy ----
  // 级联 irr/depth 不走整域 copy：滚动帧重叠矩形按位移搬运（保收敛，写 next），
  // 新列由 prepare 复位；无滚动帧 shift=0 退化为逐层整 copy（语义同 base 区）。
  // meta 体量小（16² × 4B/层），全层 copy（base + 级联 64 层）。
  {
    let encoder = ctx.command_encoder();
    for (c, shift_opt) in pending_shift.iter().enumerate() {
      let shift = shift_opt.unwrap_or(IVec3::ZERO);
      let layer0 = gpu.base_layers + c as u32 * 16;
      let dims = PROBES_PER_CASCADE_AXIS as i32;
      let pairs = [
        (
          &gpu.irr_prev,
          &gpu.irr_next,
          IRRADIANCE_TEXELS as i32,
          IRRADIANCE_LAYER_TEXELS,
        ),
        (
          &gpu.depth_prev,
          &gpu.depth_next,
          DEPTH_TEXELS as i32,
          DEPTH_LAYER_TEXELS,
        ),
      ];
      if shift == IVec3::ZERO {
        // 无滚动：整域单次 16 层 copy（语义同 base 区 prev→next）
        for (src, dst, _, size) in pairs {
          encoder.copy_texture_to_texture(
            TexelCopyTextureInfo {
              texture: src,
              mip_level: 0,
              origin: Origin3d {
                x: 0,
                y: 0,
                z: layer0,
              },
              aspect: TextureAspect::All,
            },
            TexelCopyTextureInfo {
              texture: dst,
              mip_level: 0,
              origin: Origin3d {
                x: 0,
                y: 0,
                z: layer0,
              },
              aspect: TextureAspect::All,
            },
            Extent3d {
              width: size,
              height: size,
              depth_or_array_layers: PROBES_PER_CASCADE_AXIS,
            },
          );
        }
        continue;
      }
      // xy 重叠矩形（texel；irr 每探针 cell 8、depth 16）；轴无重叠 → 本纹理跳过。
      // **双向拷贝**：prev→next 供本帧管线与 swap 后采样；next→prev 镜像回写
      // 供同帧 update 的 EMA 基准读到搬运后的新格数据（否则 EMA 混入旧格错位
      // 探针的纹理 → 滚动帧光斑污染）。
      for (tex_prev, tex_next, t, _) in pairs {
        let (Some((lx, wx)), Some((ly, wy))) = (
          axis_shift_range(shift.x, dims),
          axis_shift_range(shift.y, dims),
        ) else {
          continue;
        };
        for r in 0..dims {
          let r_old = r - shift.z;
          if !(0..dims).contains(&r_old) {
            continue;
          }
          let src = TexelCopyTextureInfo {
            texture: tex_prev,
            mip_level: 0,
            origin: Origin3d {
              x: ((lx - shift.x) * t) as u32,
              y: ((ly - shift.y) * t) as u32,
              z: layer0 + (r_old as u32),
            },
            aspect: TextureAspect::All,
          };
          let dst = TexelCopyTextureInfo {
            texture: tex_next,
            mip_level: 0,
            origin: Origin3d {
              x: (lx * t) as u32,
              y: (ly * t) as u32,
              z: layer0 + (r as u32),
            },
            aspect: TextureAspect::All,
          };
          let extent = Extent3d {
            width: (wx * t) as u32,
            height: (wy * t) as u32,
            depth_or_array_layers: 1,
          };
          encoder.copy_texture_to_texture(src, dst, extent);
          // 镜像（src/dst 互换）：两份纹理同持新格数据
          let TexelCopyTextureInfo {
            texture: st,
            origin: so,
            ..
          } = src;
          let TexelCopyTextureInfo {
            texture: dt,
            origin: do_,
            ..
          } = dst;
          encoder.copy_texture_to_texture(
            TexelCopyTextureInfo {
              texture: dt,
              mip_level: 0,
              origin: do_,
              aspect: TextureAspect::All,
            },
            TexelCopyTextureInfo {
              texture: st,
              mip_level: 0,
              origin: so,
              aspect: TextureAspect::All,
            },
            extent,
          );
        }
      }
    }
    for (src, dst, size, layers) in [
      (
        &gpu.meta_prev,
        &gpu.meta_next,
        PROBES_PER_LAYER_AXIS,
        gpu.base_layers + 64,
      ),
      (
        &gpu.irr_prev,
        &gpu.irr_next,
        IRRADIANCE_LAYER_TEXELS,
        gpu.layers,
      ),
      (
        &gpu.depth_prev,
        &gpu.depth_next,
        DEPTH_LAYER_TEXELS,
        gpu.layers,
      ),
    ] {
      encoder.copy_texture_to_texture(
        TexelCopyTextureInfo {
          texture: src,
          mip_level: 0,
          origin: Origin3d::ZERO,
          aspect: TextureAspect::All,
        },
        TexelCopyTextureInfo {
          texture: dst,
          mip_level: 0,
          origin: Origin3d::ZERO,
          aspect: TextureAspect::All,
        },
        Extent3d {
          width: size,
          height: size,
          depth_or_array_layers: layers,
        },
      );
    }
  }

  let set_bgs = |pass: &mut bevy::render::render_resource::ComputePass, bg4: &BindGroup| {
    pass.set_bind_group(0, &bg0.0, &[]);
    pass.set_bind_group(1, &bg1.0, &[]);
    pass.set_bind_group(2, &bg2.0, &[]);
    pass.set_bind_group(3, &bg3.0, &[]);
    pass.set_bind_group(4, bg4, &[]);
  };
  let recorder = ctx.diagnostic_recorder();
  let recorder = recorder.as_deref();

  // ①clear（1 线程清 indirect 计数）
  {
    let mut pass = ctx
      .command_encoder()
      .begin_compute_pass(&ComputePassDescriptor {
        label: Some("gate_ddgi_clear"),
        ..Default::default()
      });
    pass.set_pipeline(p_clear);
    set_bgs(&mut pass, &bg4.0);
    pass.dispatch_workgroups(1, 1, 1);
  }
  // ②active（@workgroup_size(4,4,4)，dispatch = ceil(dims/4)）
  {
    let span = recorder.time_span(ctx.command_encoder(), "gate_ddgi_active");
    {
      let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
          label: Some("gate_ddgi_active"),
          ..Default::default()
        });
      pass.set_pipeline(p_active);
      set_bgs(&mut pass, &bg4.0);
      pass.dispatch_workgroups(
        gpu.grid_dims.x.div_ceil(4),
        gpu.grid_dims.y.div_ceil(4),
        gpu.grid_dims.z.div_ceil(4),
      );
    }
    span.end(ctx.command_encoder());
  }
  // ②.5 seal（1 线程）：全量 count → 钳制值 dispatch[1]（≤ DDGI_PROBE_BUDGET，
  // 规避 max_compute_workgroups_per_dimension = 65535 静默跳过——gate 特有规模坑）
  {
    let mut pass = ctx
      .command_encoder()
      .begin_compute_pass(&ComputePassDescriptor {
        label: Some("gate_ddgi_seal"),
        ..Default::default()
      });
    pass.set_pipeline(p_seal);
    set_bgs(&mut pass, &bg4.0);
    pass.dispatch_workgroups(1, 1, 1);
  }
  // dispatch[1]（min）→ indirect[0] 桥接（4B；indirect[1]=y=1 [2]=z=1 由烘焙期一次性
  // 初始化常驻——copy 若带 dispatch[2..] 会把轮转 base/z=0 带进 y/z → 零 workgroup）
  {
    let encoder = ctx.command_encoder();
    encoder.copy_buffer_to_buffer(&gpu.dispatch, 4, &gpu.indirect, 0, 4);
  }
  // ③cast（indirect：x = dispatch[0] 活跃探针数）
  {
    let span = recorder.time_span(ctx.command_encoder(), "gate_ddgi_cast");
    {
      let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
          label: Some("gate_ddgi_cast"),
          ..Default::default()
        });
      pass.set_pipeline(p_cast);
      set_bgs(&mut pass, &bg4.0);
      pass.dispatch_workgroups_indirect(&gpu.indirect, 0);
    }
    span.end(ctx.command_encoder());
  }
  // ④update（indirect：同 count；1 wg = 1 探针）
  {
    let span = recorder.time_span(ctx.command_encoder(), "gate_ddgi_update");
    {
      let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
          label: Some("gate_ddgi_update"),
          ..Default::default()
        });
      pass.set_pipeline(p_update);
      set_bgs(&mut pass, &bg4.0);
      pass.dispatch_workgroups_indirect(&gpu.indirect, 0);
    }
    span.end(ctx.command_encoder());
  }

  // ---- M4-3 级联链：每级一组 ①clear → ②active（16³ cell → 4³ wg）→ ②.5seal →
  // 桥接 copy → ③cast → ④update（indirect）。BG4_c 的 binding 0/2/9/11/12 = 本级
  // uniform/ci/dispatch/samples/worklist；采样（ddgi_sample）经共享 binding 13/14
  // 读全域数据。race 纪律同 base：dispatch 读写跨 pass，写者/读者必须分 pass。
  for c in 0..4usize {
    let Some(cg) = gpu.cascades.get(c) else {
      break;
    };
    let Some(bg4c) = cg.bg4.as_ref() else {
      continue;
    };
    if cg.probe_count == 0 {
      continue;
    }
    {
      let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
          label: Some("gate_ddgi_clear_c"),
          ..Default::default()
        });
      pass.set_pipeline(p_clear);
      set_bgs(&mut pass, bg4c);
      pass.dispatch_workgroups(1, 1, 1);
    }
    {
      let span = recorder.time_span(ctx.command_encoder(), "gate_ddgi_active_c");
      {
        let mut pass = ctx
          .command_encoder()
          .begin_compute_pass(&ComputePassDescriptor {
            label: Some("gate_ddgi_active_c"),
            ..Default::default()
          });
        pass.set_pipeline(p_active);
        set_bgs(&mut pass, bg4c);
        // 滚动级恒为 16³ cell → ceil(16/4) = 4³ workgroup
        pass.dispatch_workgroups(
          PROBES_PER_CASCADE_AXIS / 4,
          PROBES_PER_CASCADE_AXIS / 4,
          PROBES_PER_CASCADE_AXIS / 4,
        );
      }
      span.end(ctx.command_encoder());
    }
    {
      let mut pass = ctx
        .command_encoder()
        .begin_compute_pass(&ComputePassDescriptor {
          label: Some("gate_ddgi_seal_c"),
          ..Default::default()
        });
      pass.set_pipeline(p_seal);
      set_bgs(&mut pass, bg4c);
      pass.dispatch_workgroups(1, 1, 1);
    }
    ctx
      .command_encoder()
      .copy_buffer_to_buffer(&cg.dispatch, 4, &cg.indirect, 0, 4);
    {
      let span = recorder.time_span(ctx.command_encoder(), "gate_ddgi_cast_c");
      {
        let mut pass = ctx
          .command_encoder()
          .begin_compute_pass(&ComputePassDescriptor {
            label: Some("gate_ddgi_cast_c"),
            ..Default::default()
          });
        pass.set_pipeline(p_cast);
        set_bgs(&mut pass, bg4c);
        pass.dispatch_workgroups_indirect(&cg.indirect, 0);
      }
      span.end(ctx.command_encoder());
    }
    {
      let span = recorder.time_span(ctx.command_encoder(), "gate_ddgi_update_c");
      {
        let mut pass = ctx
          .command_encoder()
          .begin_compute_pass(&ComputePassDescriptor {
            label: Some("gate_ddgi_update_c"),
            ..Default::default()
          });
        pass.set_pipeline(p_update);
        set_bgs(&mut pass, bg4c);
        pass.dispatch_workgroups_indirect(&cg.indirect, 0);
      }
      span.end(ctx.command_encoder());
    }
  }

  // ---- 诊断回读三阶段（wgpu 规则：submit 时 buffer 必须 Unmapped，故 copy 与
  // map_async 分帧；每阶段一帧，回调完成后读+unmap 回 idle）----
  match gpu.readback_state {
    1 => {
      // 上一帧 copy 已 submit → 请求映射（本帧 submit 时 buffer 无被录命令）
      let (tx, rx) = std::sync::mpsc::channel();
      gpu.readback.slice(..).map_async(MapMode::Read, move |_| {
        let _ = tx.send(());
      });
      *gpu.readback_rx.lock().unwrap() = Some(rx);
      gpu.readback_state = 2;
    }
    2 => {
      let done = gpu
        .readback_rx
        .lock()
        .unwrap()
        .as_ref()
        .and_then(|rx| rx.try_recv().ok())
        .is_some();
      if done {
        let data = gpu.readback.slice(..).get_mapped_range();
        let word = |i: usize| {
          u32::from_le_bytes([
            data[i * 4],
            data[i * 4 + 1],
            data[i * 4 + 2],
            data[i * 4 + 3],
          ])
        };
        let count = word(0);
        let indirect0 = word(4);
        let wl = [word(8), word(9), word(10), word(11)];
        // 8 个采样层（层距 = copy 时定格的 readback_step）：各层非零 texel 数
        let texels_per_layer = (IRRADIANCE_LAYER_TEXELS * IRRADIANCE_LAYER_TEXELS * 2) as usize;
        let mut summary = String::new();
        for k in 0..8usize {
          let base = 64 + k * texels_per_layer;
          let nz = (base..base + texels_per_layer)
            .filter(|&i| word(i) != 0)
            .count();
          let layer = k as u32 * gpu.readback_step;
          summary.push_str(&format!(" L{layer}:{nz}"));
        }
        // depth 采样层 0（tmax=8192 初值；EMA 拉低 = update 写入实证）
        let dep_base = 64 + 8 * texels_per_layer;
        let dep_texels = (DEPTH_LAYER_TEXELS * DEPTH_LAYER_TEXELS) as usize;
        let dep_lt = (dep_base..dep_base + dep_texels)
          .filter(|&i| f32::from_bits(word(i)) < 8000.0)
          .count();
        // 级联链：byte 96+16c = word 24+4c（dispatch[0]=全量/[1]=钳制）；
        // byte 160+16c = word 40+4c（indirect[0]=实际 cast workgroup 数）
        let mut casc = String::new();
        for c in 0..4usize {
          casc.push_str(&format!(
            " c{}:{}/{}/{}",
            c,
            word(24 + c * 4),
            word(25 + c * 4),
            word(40 + c * 4)
          ));
        }
        // 黑探针普查（同 8 采样层，2048 探针）：a0 = age=0（未激活，采样端已门控）；
        // blk = age≥1 且 irr 全零（被当有效探针采样 → 暗块/黑块直接来源）；ok = 正常。
        // blk 全局探针 id（= layer×256+p）取前 8 个供世界坐标定位（cell = id 反查）。
        // blk 的 age 直方图：新 age = 滚动重置/新激活列（假设 B/D）；高 age = 从未被
        // cast 更新的陈旧探针（假设 B/E：预算钳制或活跃集翻转）
        let meta_base =
          64 + 8 * texels_per_layer + (DEPTH_LAYER_TEXELS * DEPTH_LAYER_TEXELS) as usize;
        let (mut a0, mut blk, mut ok) = (0u32, 0u32, 0u32);
        let (mut b_a1, mut b_a8, mut b_a32, mut b_a64) = (0u32, 0u32, 0u32, 0u32);
        let mut blk_ids = String::new();
        for k in 0..8usize {
          let layer = k as u32 * gpu.readback_step;
          let ibase = 64 + k * texels_per_layer;
          // meta 区按拷贝行距 256B（64 word/行）读回：层 = 16 行 × 64 word，
          // 探针 p 的 word 下标 = 行 p/16 × 64 + 列 p%16（其余为 pad 零）
          let mb = meta_base + k * (PROBES_PER_LAYER_AXIS as usize * META_ROW_WORDS);
          for p in 0..PROBES_PER_LAYER as usize {
            let age = (word(
              mb + (p / PROBES_PER_LAYER_AXIS as usize) * META_ROW_WORDS
                + (p % PROBES_PER_LAYER_AXIS as usize),
            ) >> META_AGE_SHIFT)
              & 0xFF;
            if age == 0 {
              a0 += 1;
              continue;
            }
            let px = p % PROBES_PER_LAYER_AXIS as usize;
            let py = p / PROBES_PER_LAYER_AXIS as usize;
            let mut nz = false;
            'scan: for ty in 0..IRRADIANCE_TEXELS as usize {
              let row = ibase + (((py * 8 + ty) * IRRADIANCE_LAYER_TEXELS as usize) + px * 8) * 2;
              for tx in 0..IRRADIANCE_TEXELS as usize {
                if word(row + tx * 2) != 0 || word(row + tx * 2 + 1) != 0 {
                  nz = true;
                  break 'scan;
                }
              }
            }
            if nz {
              ok += 1;
            } else {
              blk += 1;
              // #region debug-point B:blk-age-hist（假设 B：黑探针 age 分布定位来源）
              match age {
                1..=7 => b_a1 += 1,
                8..=31 => b_a8 += 1,
                32..=63 => b_a32 += 1,
                _ => b_a64 += 1,
              }
              // #endregion
              if blk <= 8 {
                blk_ids.push_str(&format!(" {}@{}", layer as u32 * PROBES_PER_LAYER + p as u32, age));
              }
            }
          }
        }
        drop(data);
        gpu.readback.unmap();
        gpu.readback_state = 0;
        // #region debug-point B:server-relay（readback 汇总行 → 调试服务器 NDJSON；
        // std-only 裸 HTTP POST，服务器离线时连接即失败静默跳过；~4Hz 低频）
        fn dbg_relay(hyp: &str, msg: &str) {
          let url = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../.dbg/ddgi-black-spots-flicker.env"
          ))
          .ok()
          .and_then(|s| {
            s.lines()
              .find(|l| l.starts_with("DEBUG_SERVER_URL="))
              .map(|l| l["DEBUG_SERVER_URL=".len()..].trim().to_string())
          })
          .unwrap_or_else(|| "http://127.0.0.1:7777/event".into());
          let rest = url.strip_prefix("http://").unwrap_or(&url);
          let (authority, path) = match rest.split_once('/') {
            Some((a, p)) => (a, p),
            None => (rest, "event"),
          };
          let body = format!(
            "{{\"sessionId\":\"ddgi-black-spots-flicker\",\"runId\":\"pre-fix\",\"hypothesisId\":\"{hyp}\",\"location\":\"ddgi.rs:readback\",\"msg\":\"[DEBUG] {}\"}}",
            msg.replace('\\', "\\\\").replace('"', "\\\"")
          );
          if let Ok(mut s) = std::net::TcpStream::connect(authority) {
            use std::io::Write as _;
            let _ = s.write_all(
              format!(
                "POST /{path} HTTP/1.1\r\nHost: {authority}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
              )
              .as_bytes(),
            );
          }
        }
        // #endregion
        let line = format!(
          "DDGI readback: frame={} dispatch0={} indirect0={} wl=[{},{},{},{}]{} irr:{summary} depth650_lt8k:{} census: a0={a0} blk={blk} ok={ok} blk_ids:[{blk_ids}]",
          gpu.frame,
          count,
          indirect0,
          wl[0],
          wl[1],
          wl[2],
          wl[3],
          casc,
          dep_lt,
        );
        bevy::log::info!("{}", line);
        // #region debug-point B/E:relay-send（假设 B=黑探针普查、E=预算/活跃信号）
        dbg_relay("B", &format!("census a0={a0} blk={blk} ok={ok} blk_age[1-7/8-31/32-63/64+]={b_a1}/{b_a8}/{b_a32}/{b_a64} ids:[{blk_ids}] frame={}", gpu.frame));
        dbg_relay("E", &format!("dispatch0={count} indirect0={indirect0} wl=[{},{},{},{}]{} depth_lt8k={dep_lt} frame={}", wl[0], wl[1], wl[2], wl[3], casc, gpu.frame));
        // #endregion
      }
    }
    _ => {
      if gpu.frame % 120 == 0 {
        {
          let encoder = ctx.command_encoder();
          encoder.copy_buffer_to_buffer(&gpu.dispatch, 0, &gpu.readback, 0, 16);
          // 诊断：indirect[0]（桥接是否生效）+ worklist 头部（active 写入是否可见）
          encoder.copy_buffer_to_buffer(&gpu.indirect, 0, &gpu.readback, 16, 16);
          encoder.copy_buffer_to_buffer(&gpu.worklist, 0, &gpu.readback, 32, 64);
          // 每级联 dispatch/indirect（96..160 / 160..224）——级联链健康度对照
          for (i, cg) in gpu.cascades.iter().enumerate() {
            let o = 96 + (i as u64) * 16;
            encoder.copy_buffer_to_buffer(&cg.dispatch, 0, &gpu.readback, o, 16);
            encoder.copy_buffer_to_buffer(&cg.indirect, 0, &gpu.readback, o + 64, 16);
          }
          // 8 个均匀采样层（层距 = 总层数/8）+ depth 层 0——读 **next**（本帧 update
          // 刚写完，encoder 顺序在 update pass 之后；prev 要到下帧 swap 才有新值）。
          // 层距动态：base 16³ 重做后纹理总层 = base_layers+64 可能 < 旧硬编码 160，
          // 固定步距越界（2026-09-08 Validation Error：Z 160..161 > 85 层）。
          let step = ((gpu.base_layers + 64) / 8).max(1);
          gpu.readback_step = step;
          for k in 0..8u32 {
            encoder.copy_texture_to_buffer(
              TexelCopyTextureInfo {
                texture: &gpu.irr_next,
                mip_level: 0,
                origin: Origin3d { x: 0, y: 0, z: k * step },
                aspect: TextureAspect::All,
              },
              TexelCopyBufferInfo {
                buffer: &gpu.readback,
                layout: TexelCopyBufferLayout {
                  offset: 256
                    + (k as u64) * (IRRADIANCE_LAYER_TEXELS * IRRADIANCE_LAYER_TEXELS * 8) as u64,
                  bytes_per_row: Some(IRRADIANCE_LAYER_TEXELS * 8),
                  rows_per_image: Some(IRRADIANCE_LAYER_TEXELS),
                },
              },
              Extent3d {
                width: IRRADIANCE_LAYER_TEXELS,
                height: IRRADIANCE_LAYER_TEXELS,
                depth_or_array_layers: 1,
              },
            );
          }
          // depth 采样层 650（tmax=8192 初值；EMA 拉低 = update 写入实证）
          encoder.copy_texture_to_buffer(
            TexelCopyTextureInfo {
              texture: &gpu.depth_next,
              mip_level: 0,
              origin: Origin3d::ZERO,
              aspect: TextureAspect::All,
            },
            TexelCopyBufferInfo {
              buffer: &gpu.readback,
              layout: TexelCopyBufferLayout {
                offset: 256 + (8 * IRRADIANCE_LAYER_TEXELS * IRRADIANCE_LAYER_TEXELS * 8) as u64,
                bytes_per_row: Some(DEPTH_LAYER_TEXELS * 4),
                rows_per_image: Some(DEPTH_LAYER_TEXELS),
              },
            },
            Extent3d {
              width: DEPTH_LAYER_TEXELS,
              height: DEPTH_LAYER_TEXELS,
              depth_or_array_layers: 1,
            },
          );
          // 黑探针普查：同 8 个采样层的 meta（逐探针 age；布局 = 16×16 u32/层）
          for k in 0..8u32 {
            encoder.copy_texture_to_buffer(
              TexelCopyTextureInfo {
                texture: &gpu.meta_next,
                mip_level: 0,
                origin: Origin3d { x: 0, y: 0, z: k * step },
                aspect: TextureAspect::All,
              },
              TexelCopyBufferInfo {
                buffer: &gpu.readback,
                layout: TexelCopyBufferLayout {
                  offset: 256
                    + (8 * IRRADIANCE_LAYER_TEXELS * IRRADIANCE_LAYER_TEXELS * 8) as u64
                    + (DEPTH_LAYER_TEXELS * DEPTH_LAYER_TEXELS * 4) as u64
                    + (k as u64) * (META_COPY_BPR as u64 * PROBES_PER_LAYER_AXIS as u64),
                  // 行距 256B 对齐（行宽 64B 违规 → Validation Error）
                  bytes_per_row: Some(META_COPY_BPR),
                  rows_per_image: Some(PROBES_PER_LAYER_AXIS),
                },
              },
              Extent3d {
                width: PROBES_PER_LAYER_AXIS,
                height: PROBES_PER_LAYER_AXIS,
                depth_or_array_layers: 1,
              },
            );
          }
        }
        gpu.readback_state = 1;
      }
    }
  }
}

/// main world VoxelScene → render world ProbeBake（版本驱动：世界编辑代数变化即重烘）
fn extract_ddgi_bake(
  mut commands: Commands,
  scene: Option<bevy::render::Extract<Res<crate::VoxelScene>>>,
  view: Option<Res<crate::brickmap::dda::DdaViewUniform>>,
  gpu: Option<ResMut<DdgiGpu>>,
  inflight: Option<Res<ProbeBake>>,
  enabled: Option<bevy::render::Extract<Res<DdgiEnabled>>>,
  debug: Option<bevy::render::Extract<Res<DdgiDebugSettings>>>,
) {
  // 主世界开关 → 渲染世界（prepare/dispatch 读渲染世界副本）
  let on = enabled.map_or(true, |e| e.0);
  commands.insert_resource(DdgiEnabled(on));
  // 主世界 DDGI 调试参数 → 渲染世界（prepare_ddgi 写进 uniform.params.y/.z）
  let dbg = match debug {
    Some(d) => DdgiDebugSettings {
      mode: d.mode,
      gain: d.gain,
      probe_viz: d.probe_viz,
      probe_viz_lod: d.probe_viz_lod,
    },
    None => DdgiDebugSettings::default(),
  };
  commands.insert_resource(dbg);
  let Some(scene) = scene else {
    return;
  };
  let Some(mut gpu) = gpu else {
    return;
  };
  let cam = view
    .map(|v| v.cam_pos_fine.truncate())
    .unwrap_or(Vec3::splat(32.0));
  let generation = scene.volumes.main().edit_generation();
  // GPU 已是该代数（已重烘过）→ 只需驱动级联滚动
  if gpu.baked_generation == generation {
    // 关闭时冻结滚动：不产生 pending_shift（dispatch 早退也不消费），避免位移堆积
    if on {
      if let Some(cm) = gpu.manager.as_mut() {
        let ts = std::time::Instant::now();
        let moved = cm.scroll(&scene.volumes, cam);
        let us = ts.elapsed().as_micros();
        if moved && us > 500 {
          bevy::log::info!("DDGI scroll: {us}us (moved)");
        }
      }
    }
    return;
  }
  // 同代数烘焙已在排队（prepare 尚未消费）→ 不重复烘焙
  if inflight.is_some_and(|b| b.1 == generation) {
    return;
  }
  let t0 = std::time::Instant::now();
  let pg = bake_probe_grid(&scene.volumes);
  // M4-3：级联管理器随代数重建（base 烘焙 + 4 级级联初烘）
  gpu.manager = Some(CascadeManager::new(&scene.volumes, cam));
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

/// meta 拷贝行距：wgpu COPY_BYTES_PER_ROW_ALIGNMENT = 256B。meta 行宽
/// 16 texel × 4B = 64B 不达标——所有 buffer↔meta 纹理拷贝（write_texture /
/// copy_texture_to_buffer）必须按 256B 行距排布（每行 64 word：前 16 word 数据
/// + 48 pad word；violation = Validation Error 退出）。
pub const META_COPY_BPR: u32 = 256;
const META_ROW_WORDS: usize = META_COPY_BPR as usize / 4;

/// 紧凑 meta words（层主序 16×16 u32/层，`build_meta_texture_data` 布局）→
/// 256B 行距 padded 字节流。层内 (x,y) 不变，仅行尾补 pad。
fn meta_padded_bytes(flat_words: &[u32]) -> Vec<u8> {
  let per_layer = PROBES_PER_LAYER as usize;
  let layers = flat_words.len() / per_layer;
  debug_assert_eq!(flat_words.len() % per_layer, 0);
  let mut out = vec![0u8; layers * PROBES_PER_LAYER_AXIS as usize * META_ROW_WORDS * 4];
  for l in 0..layers {
    for y in 0..PROBES_PER_LAYER_AXIS as usize {
      for x in 0..PROBES_PER_LAYER_AXIS as usize {
        let src = l * per_layer + y * PROBES_PER_LAYER_AXIS as usize + x;
        let dst = l * (PROBES_PER_LAYER_AXIS as usize * META_ROW_WORDS) + y * META_ROW_WORDS + x;
        out[dst * 4..dst * 4 + 4].copy_from_slice(&flat_words[src].to_le_bytes());
      }
    }
  }
  out
}

/// 滚动复位矩形集（texel 单位 [x, y, w, h]，层内）：新窗 − 重叠窗（x 全高条 + y
/// 全高条，角部重复写同值无害）；任一轴无重叠 → 单矩形 = 整层。
/// `t` = 每探针 cell 的 texel 数（irr 8 / depth 16），`lt` = 层边长（128 / 256）。
fn scroll_reset_rects(shift: IVec3, t: i32, lt: i32) -> Vec<[i32; 4]> {
  let dims = PROBES_PER_CASCADE_AXIS as i32;
  let Some((lx, wx)) = axis_shift_range(shift.x, dims) else {
    return vec![[0, 0, lt, lt]];
  };
  let Some((ly, wy)) = axis_shift_range(shift.y, dims) else {
    return vec![[0, 0, lt, lt]];
  };
  let mut r = Vec::with_capacity(4);
  let (x0, x1, y0, y1) = (lx * t, (lx + wx) * t, ly * t, (ly + wy) * t);
  if x0 > 0 {
    r.push([0, 0, x0, lt]);
  }
  if x1 < lt {
    r.push([x1, 0, lt - x1, lt]);
  }
  if y0 > 0 {
    r.push([0, 0, lt, y0]);
  }
  if y1 < lt {
    r.push([0, y1, lt, lt - y1]);
  }
  r
}

/// 滚动帧级联 irr/depth 新列复位（queue 直写 prev/next 两份；与 dispatch 侧
/// shifted copy 的重叠区不相交，顺序无关——级联区不走整域 copy，故两份都要补）。
/// z 轴出层 → 整层复位。irr 填 0（无间接光）、depth 填 tmax（远距哨兵；初值 0
/// 会被漏光剔除读成「探针贴墙」误伤全部贡献）。
fn cascade_scroll_reset(
  queue: &RenderQueue,
  layer0: u32,
  shift: IVec3,
  irr: (&Texture, &Texture),
  depth: (&Texture, &Texture),
) {
  let z_dims = PROBES_PER_CASCADE_AXIS as i32;
  let irr_rects = scroll_reset_rects(
    shift,
    IRRADIANCE_TEXELS as i32,
    IRRADIANCE_LAYER_TEXELS as i32,
  );
  let dep_rects = scroll_reset_rects(shift, DEPTH_TEXELS as i32, DEPTH_LAYER_TEXELS as i32);
  let irr_full = vec![[
    0,
    0,
    IRRADIANCE_LAYER_TEXELS as i32,
    IRRADIANCE_LAYER_TEXELS as i32,
  ]];
  let dep_full = vec![[0, 0, DEPTH_LAYER_TEXELS as i32, DEPTH_LAYER_TEXELS as i32]];
  for r in 0..z_dims {
    let full = !(0..z_dims).contains(&(r - shift.z));
    for (tex_pair, rects, full_rects, words, pattern) in [
      (irr, &irr_rects, &irr_full, 2usize, [0u8; 4]),
      (
        depth,
        &dep_rects,
        &dep_full,
        1usize,
        PROBE_T_MAX.to_le_bytes(),
      ),
    ] {
      let list: &[[i32; 4]] = if full { full_rects } else { rects };
      for &[x, y, w, h] in list {
        // 行距 256B 对齐（COPY_BYTES_PER_ROW_ALIGNMENT）：w=8/16 列带的裸行宽
        // 64B 不达标 → 每行补 pad 到 256B 边界
        let row_bytes = (w as usize) * 4 * words;
        let bpr = ((row_bytes + 255) / 256) * 256;
        let row_texels = (w as usize) * words;
        let pad_words = bpr / 4 - row_texels;
        let mut data = Vec::with_capacity((row_texels + pad_words) * (h as usize) * 4);
        for _ in 0..h {
          for _ in 0..row_texels {
            data.extend_from_slice(&pattern);
          }
          for _ in 0..pad_words {
            data.extend_from_slice(&[0u8; 4]);
          }
        }
        for tex in [tex_pair.0, tex_pair.1] {
          queue.write_texture(
            TexelCopyTextureInfo {
              texture: tex,
              mip_level: 0,
              origin: Origin3d {
                x: x as u32,
                y: y as u32,
                z: layer0 + r as u32,
              },
              aspect: TextureAspect::All,
            },
            &data,
            TexelCopyBufferLayout {
              offset: 0,
              bytes_per_row: Some(bpr as u32),
              rows_per_image: Some(h as u32),
            },
            Extent3d {
              width: w as u32,
              height: h as u32,
              depth_or_array_layers: 1,
            },
          );
        }
      }
    }
  }
}

#[allow(clippy::too_many_arguments)]
fn prepare_ddgi(
  mut commands: Commands,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
  pipeline_cache: Res<bevy::render::render_resource::PipelineCache>,
  bake: Option<Res<ProbeBake>>,
  enabled: Res<DdgiEnabled>,
  dbg: Res<DdgiDebugSettings>,
  mut gpu: ResMut<DdgiGpu>,
) {
  let tp = std::time::Instant::now();
  // ---- D7 ping-pong 交换（无条件；烘焙帧两份同数据交换无害）----
  // 上一帧 next（已被 update 写入新值）变本帧 prev；旧 prev 由 dispatch_ddgi 的
  // copy pass 重填后作为本帧 next 被 active/update 覆写。
  {
    let g = &mut *gpu;
    let DdgiGpu {
      irr_prev,
      irr_next,
      irr_prev_view,
      irr_next_view,
      depth_prev,
      depth_next,
      depth_prev_view,
      depth_next_view,
      meta_prev,
      meta_next,
      meta_prev_view,
      meta_next_view,
      ..
    } = g;
    std::mem::swap(irr_prev, irr_next);
    std::mem::swap(irr_prev_view, irr_next_view);
    std::mem::swap(depth_prev, depth_next);
    std::mem::swap(depth_prev_view, depth_next_view);
    std::mem::swap(meta_prev, meta_next);
    std::mem::swap(meta_prev_view, meta_next_view);
  }

  // ---- 新烘焙（含编辑后重烘）：重建纹理数组 + buffer（尺寸变化）----
  if let Some(bake) = bake {
    let generation = bake.1;
    let pg = &bake.0;
    let n = pg.positions.len() as u32;
    let layers = meta_texture_layers(n);
    // 层数上限保护（验收：资源不支持时报错明确；fallback = spec §5 base cell 升 32）
    let max_layers = device.limits().max_texture_array_layers;
    if layers > max_layers {
      bevy::log::error!(
        "DDGI bake 需要 {layers} 层纹理数组，超设备上限 {max_layers}（探针 {n}）；\
         本代 DDGI 禁用。缓解 = spec §5 风险表：base cell 升 32（探针数 ÷8）"
      );
      commands.remove_resource::<ProbeBake>();
      return;
    }

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
    gpu.positions = make("ddgi_positions", &vec4_bytes(&pack_probe_positions(pg)));
    gpu.cell_index = make("ddgi_cell_index", &u32_bytes(&pg.cell_index));
    gpu.objects = dummy_sized_buffer(&device, "ddgi_objects", 16); // D12：MOV 后置；runtime-sized vec4 数组最小 1 元素
    // dispatch（atomic 计数 [count,1,1,0]）：STORAGE；indirect 走独立 buffer（同
    // dispatch scope 内 STORAGE 与 INDIRECT 互斥——wgpu 验证，active 后整拷桥接）
    let dispatch = device.create_buffer(&BufferDescriptor {
      label: Some("ddgi_dispatch".into()),
      size: 16,
      usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
      mapped_at_creation: false,
    });
    queue.write_buffer(&dispatch, 0, &0u32.to_le_bytes());
    gpu.dispatch = dispatch;
    // indirect：[x=min(seal 桥接覆写), y=1, z=1, pad=0]——y/z 一次性常驻
    gpu.indirect = dummy_indirect_buffer(&device, "ddgi_indirect");
    queue.write_buffer(&gpu.indirect, 4, &1u32.to_le_bytes());
    queue.write_buffer(&gpu.indirect, 8, &1u32.to_le_bytes());
    gpu.worklist = make("ddgi_worklist", &u32_bytes(&vec![0u32; n as usize]));
    // 样本缓冲容量：base = max(2×probe_count, 钳制满额×射线下限) vec4
    // （seal 钳 4096 × DDGI_PROBE_RAYS_MIN × [dir, radiance]）
    let sample_slots = (n * 2)
      .max(8192)
      .max(DDGI_PROBE_BUDGET * DDGI_PROBE_RAYS_MIN * 2);
    gpu.samples = dummy_sized_buffer(&device, "ddgi_samples", sample_slots as u64 * 16);

    // ---- D7 纹理数组（irr 128² rgba16f / depth 256² r32 / meta 16² r32uint）----
    // 层分配：base [0, layers) + 级联 4×16 层 [layers, layers+64)
    let total_layers = layers + 64;
    let (irr_a, irr_av) = {
      let t = ddgi_array_tex(
        &device,
        "ddgi_irr_a",
        TextureFormat::Rgba16Float,
        (IRRADIANCE_LAYER_TEXELS, IRRADIANCE_LAYER_TEXELS),
        total_layers,
      );
      let v = ddgi_array_view(&t);
      (t, v)
    };
    let (irr_b, irr_bv) = {
      let t = ddgi_array_tex(
        &device,
        "ddgi_irr_b",
        TextureFormat::Rgba16Float,
        (IRRADIANCE_LAYER_TEXELS, IRRADIANCE_LAYER_TEXELS),
        total_layers,
      );
      let v = ddgi_array_view(&t);
      (t, v)
    };
    let (dep_a, dep_av) = {
      let t = ddgi_array_tex(
        &device,
        "ddgi_depth_a",
        TextureFormat::R32Float,
        (DEPTH_LAYER_TEXELS, DEPTH_LAYER_TEXELS),
        total_layers,
      );
      let v = ddgi_array_view(&t);
      (t, v)
    };
    let (dep_b, dep_bv) = {
      let t = ddgi_array_tex(
        &device,
        "ddgi_depth_b",
        TextureFormat::R32Float,
        (DEPTH_LAYER_TEXELS, DEPTH_LAYER_TEXELS),
        total_layers,
      );
      let v = ddgi_array_view(&t);
      (t, v)
    };
    let (meta_a, meta_av) = {
      let t = ddgi_array_tex(
        &device,
        "ddgi_meta_a",
        TextureFormat::R32Uint,
        (PROBES_PER_LAYER_AXIS, PROBES_PER_LAYER_AXIS),
        total_layers,
      );
      let v = ddgi_array_view(&t);
      (t, v)
    };
    let (meta_b, meta_bv) = {
      let t = ddgi_array_tex(
        &device,
        "ddgi_meta_b",
        TextureFormat::R32Uint,
        (PROBES_PER_LAYER_AXIS, PROBES_PER_LAYER_AXIS),
        total_layers,
      );
      let v = ddgi_array_view(&t);
      (t, v)
    };
    let dep_data = f32_bytes(&vec![
      PROBE_T_MAX;
      (DEPTH_LAYER_TEXELS * DEPTH_LAYER_TEXELS * layers)
        as usize
    ]);
    // meta 行宽 64B 不达 256B 拷贝对齐 → padded 行距字节流
    let meta_data = meta_padded_bytes(&build_meta_texture_data(pg));
    for tex in [&dep_a, &dep_b, &meta_a, &meta_b] {
      let (bytes_per_row, rows, width, data): (u32, u32, u32, &[u8]) =
        if tex.format() == TextureFormat::R32Float {
          (
            DEPTH_LAYER_TEXELS * 4,
            DEPTH_LAYER_TEXELS,
            DEPTH_LAYER_TEXELS,
            &dep_data,
          )
        } else {
          (
            META_COPY_BPR,
            PROBES_PER_LAYER_AXIS,
            PROBES_PER_LAYER_AXIS,
            &meta_data,
          )
        };
      queue.write_texture(
        TexelCopyTextureInfo {
          texture: tex,
          mip_level: 0,
          origin: Origin3d::ZERO,
          aspect: TextureAspect::All,
        },
        data,
        TexelCopyBufferLayout {
          offset: 0,
          bytes_per_row: Some(bytes_per_row),
          rows_per_image: Some(rows),
        },
        Extent3d {
          width,
          height: rows,
          depth_or_array_layers: layers,
        },
      );
    }
    gpu.irr_prev = irr_a;
    gpu.irr_next = irr_b;
    gpu.depth_prev = dep_a;
    gpu.depth_next = dep_b;
    gpu.meta_prev = meta_a;
    gpu.meta_next = meta_b;
    gpu.irr_prev_view = irr_av;
    gpu.irr_next_view = irr_bv;
    gpu.depth_prev_view = dep_av;
    gpu.depth_next_view = dep_bv;
    gpu.meta_prev_view = meta_av;
    gpu.meta_next_view = meta_bv;
    gpu.layers = layers;
    gpu.base_layers = layers;
    gpu.probe_count = n;
    gpu.grid_origin = pg.grid_origin;
    gpu.grid_dims = pg.grid_dims;
    gpu.baked_generation = generation;
    gpu.frame = 0;
    gpu
      .uniform
      .get_mut()
      .clone_from(&DdgiUniform::new(pg, 0, REUSE_ALL, None, 0, dbg.mode, dbg.gain));

    // ---- M4-3 级联资源（4 级 × 4096 探针；纹理层 [base_layers + c×16, +16)）----
    // 级联探针 id = id_base + c×4096 + internal_id（id_base = base_layers×256，层
    // 取整规避 base_count 非整层碰撞）→ WGSL 坐标函数零改动。cell_index 上传时把
    // internal_id 平移为全局 id（ddgi_flags_cell/active/worklist/sampling 全链一致）；
    // positions 按 internal id 落位（pos_all[off + id]），meta 布局 = 内部 id 布局原样。
    gpu.id_base = layers * 256;
    let id_total = gpu.id_base + 4 * 4096;
    let mut pos_all = pack_probe_positions(pg);
    pos_all.resize(id_total as usize, Vec4::ZERO);
    // all_ci = base dense ci ++ 4 级平移后 ci（binding 14 采样用；binding 2 本级 ci 同数据）
    let mut all_ci_data = pg.cell_index.clone();
    gpu.base_ci_words = pg.cell_index.len() as u32;
    if let Some(cm) = &gpu.manager {
      for c in 0..4usize {
        let off = gpu.id_base + (c as u32) * 4096;
        let g = &cm.grids[c];
        for (id, pos) in g.positions.iter().enumerate() {
          pos_all[off as usize + id] =
            Vec4::new(pos.x, pos.y, pos.z, if g.active[id] { 1.0 } else { 0.0 });
        }
        all_ci_data.extend(cascade_ci_global(&g.cell_index, off));
      }
    } else {
      all_ci_data.resize(id_total as usize, NO_PROBE);
    }
    gpu.positions = make("ddgi_positions", &vec4_bytes(&pos_all));
    gpu.all_ci = make("ddgi_all_ci", &u32_bytes(&all_ci_data));
    // 级联 ci（全局 id 平移）与 probe_count 先快照到本地（manager 共享借用不得
    // 跨 gpu.cascades 的可变构建），再构建级联 GPU 槽
    let (casc_ci, casc_cnt): ([Vec<u32>; 4], [u32; 4]) = {
      let mut ci = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
      let mut cnt = [0u32; 4];
      if let Some(cm) = gpu.manager.as_ref() {
        for c in 0..4usize {
          let off = gpu.id_base + (c as u32) * 4096;
          ci[c] = cascade_ci_global(&cm.grids[c].cell_index, off);
          cnt[c] = cm.grids[c].positions.len() as u32;
        }
      } else {
        for c in 0..4usize {
          ci[c] = vec![NO_PROBE; 4096];
        }
      }
      (ci, cnt)
    };
    gpu.cascades = (0..4)
      .map(|c| CascadeGpu {
        uniform: UniformBuffer::default(),
        cell_index: make("ddgi_ci_c", &u32_bytes(&casc_ci[c])),
        worklist: make("ddgi_wl_c", &u32_bytes(&vec![0u32; 4096])),
        // 钳制满额 × 射线下限：4096 slot × DDGI_PROBE_RAYS_MIN × [dir, radiance] vec4
        samples: make(
          "ddgi_samp_c",
          &vec![0u8; 4096 * DDGI_PROBE_RAYS_MIN as usize * 2 * 16],
        ),
        dispatch: make("ddgi_disp_c", &[0u8; 16]),
        indirect: dummy_indirect_buffer(&device, "ddgi_ind_c"),
        bg4: None,
        probe_count: casc_cnt[c],
      })
      .collect();
    // 级联 indirect y/z 一次性常驻（同 base：copy 桥接只覆写 x，y/z 若为 0 → 零 workgroup）
    for cg in &gpu.cascades {
      queue.write_buffer(&cg.indirect, 4, &1u32.to_le_bytes());
      queue.write_buffer(&cg.indirect, 8, &1u32.to_le_bytes());
    }
    // 级联纹理区初始化：meta（manager 快照）+ depth tmax 写入 prev/next 两份
    // （irr 依赖 WebGPU 零初始化免上传）
    if let Some(cm) = &gpu.manager {
      let dep_data_c = f32_bytes(&vec![
        PROBE_T_MAX;
        (DEPTH_LAYER_TEXELS * DEPTH_LAYER_TEXELS * 16) as usize
      ]);
      for c in 0..4usize {
        let layer0 = layers + c as u32 * 16;
        let meta_c = meta_padded_bytes(&cm.metas[c]);
        for tex in [&gpu.meta_prev, &gpu.meta_next] {
          queue.write_texture(
            TexelCopyTextureInfo {
              texture: tex,
              mip_level: 0,
              origin: Origin3d {
                x: 0,
                y: 0,
                z: layer0,
              },
              aspect: TextureAspect::All,
            },
            &meta_c,
            TexelCopyBufferLayout {
              offset: 0,
              bytes_per_row: Some(META_COPY_BPR),
              rows_per_image: Some(PROBES_PER_LAYER_AXIS),
            },
            Extent3d {
              width: PROBES_PER_LAYER_AXIS,
              height: PROBES_PER_LAYER_AXIS,
              depth_or_array_layers: 16,
            },
          );
        }
        for tex in [&gpu.depth_prev, &gpu.depth_next] {
          queue.write_texture(
            TexelCopyTextureInfo {
              texture: tex,
              mip_level: 0,
              origin: Origin3d {
                x: 0,
                y: 0,
                z: layer0,
              },
              aspect: TextureAspect::All,
            },
            &dep_data_c,
            TexelCopyBufferLayout {
              offset: 0,
              bytes_per_row: Some(DEPTH_LAYER_TEXELS * 4),
              rows_per_image: Some(DEPTH_LAYER_TEXELS),
            },
            Extent3d {
              width: DEPTH_LAYER_TEXELS,
              height: DEPTH_LAYER_TEXELS,
              depth_or_array_layers: 16,
            },
          );
        }
      }
    }
    bevy::log::info!(
      "DDGI gpu v2: {n} probes {layers} layers + 4 cascades (64 layers) (irr {}KB, depth {}KB, meta {}KB)",
      IRRADIANCE_LAYER_TEXELS * IRRADIANCE_LAYER_TEXELS * layers * 8 / 1024,
      DEPTH_LAYER_TEXELS * DEPTH_LAYER_TEXELS * layers * 4 / 1024,
      meta_data.len() / 1024,
    );
    commands.remove_resource::<ProbeBake>();
  }

  // ---- 每帧：推进帧号 + 写 uniform + 建 BG4 v3（base 一份 + 每级联一份）----
  // probe_count=0 同样建 BG（占位资源可绑；ddgi pass 按 count=0 早退）。
  gpu.frame = gpu.frame.wrapping_add(1);
  gpu.uniform.get_mut().params.x = gpu.frame as f32;
  // DDGI 调试参数（主世界 DebugView UI 驱动，extract 每帧拷贝）：mode→params.y，
  // gain→params.z。base uniform 在烘焙帧一次性设置，此处每帧覆写保证运行时热切换生效。
  gpu.uniform.get_mut().params.y = dbg.mode;
  gpu.uniform.get_mut().params.z = dbg.gain;
  // 运行时开关位：reuse_max.w 是空闲通道（WGSL 只读 reuse_max.xyz）；trace shader
  // 据此跳过探针采样（gi=0）。1=开 0=关。
  gpu.uniform.get_mut().reuse_max.w = if enabled.0 { 1.0 } else { 0.0 };

  // M4-3 级联帧数据快照（ResMut 全域借用：manager 的共享借用不得跨 uniform/
  // cascades 的可变访问 → 一次性快照后立即释放；滚动帧才携带上传负载）
  struct CascFrame {
    dom: DdgiUniform,
    shift: Option<IVec3>,
    probe_count: u32,
    pos_packed: Option<Vec<Vec4>>,
    ci_words: Option<Vec<u32>>,
    meta_words: Option<Vec<u32>>,
  }
  let casc_frames: Option<[CascFrame; 4]> = gpu.manager.as_ref().map(|cm| {
    let id_base = gpu.id_base;
    let frame = gpu.frame;
    std::array::from_fn(|c| {
      // finer = 上一级（更细的）域，None = 本级最细
      // casc0 是最细级（跟随相机 32 voxels）→ finer=None
      // casc1/2/3 → finer = cm.grids[c-1]（上一级更细的）
      let finer = match c {
        0 => None,
        _ => Some(CascadeDomain {
          origin: cm.grids[c - 1].grid_origin,
          dims: cm.grids[c - 1].grid_dims,
          cell_size: cm.grids[c - 1].cell_size,
        }),
      };
      let shift = cm.pending_shift[c];
      CascFrame {
        dom: DdgiUniform::new(&cm.grids[c], frame, cm.reuse[c], finer, 0, dbg.mode, dbg.gain),
        shift,
        probe_count: cm.grids[c].positions.len() as u32,
        pos_packed: shift.map(|_| pack_probe_positions(&cm.grids[c])),
        ci_words: shift
          .map(|_| cascade_ci_global(&cm.grids[c].cell_index, id_base + (c as u32) * 4096)),
        meta_words: shift.map(|_| cm.metas[c].clone()),
      }
    })
  });
  // base uniform finer = LOD1 域（base 被 LOD1 覆盖的 cell 不再出探针）
  if let Some(frames) = &casc_frames {
    let u = gpu.uniform.get_mut();
    u.finer_min = frames[0].dom.grid_origin;
    u.finer_size = frames[0].dom.grid_dims;
  }
  let base_u = *gpu.uniform.get_mut();
  gpu.uniform.write_buffer(&device, &queue);

  // 级联 uniform（本级域 + finer = 上一级域）+ probe_count
  if let Some(frames) = &casc_frames {
    for (c, f) in frames.iter().enumerate() {
      let Some(cg) = gpu.cascades.get_mut(c) else {
        continue;
      };
      *cg.uniform.get_mut() = f.dom;
      cg.uniform.write_buffer(&device, &queue);
      cg.probe_count = f.probe_count;
    }
    // 滚动帧增量（pending_shift 由 extract 侧 scroll 写入）：本级 ci（binding 2）
    // 与 all_ci 段（binding 14）同数据（全局 id 平移）；positions 段按 internal id；
    // meta 快照重传 16 层（重叠 age 搬运 + 新列基线）+ irr/depth 新列复位
    for (c, f) in frames.iter().enumerate() {
      let Some(shift) = f.shift else {
        continue;
      };
      let Some(cg) = gpu.cascades.get(c) else {
        continue;
      };
      let ci_off = gpu.id_base + (c as u32) * 4096;
      if let Some(pos) = &f.pos_packed {
        queue.write_buffer(&gpu.positions, ci_off as u64 * 16, &vec4_bytes(pos));
      }
      if let Some(ci) = &f.ci_words {
        let bytes = u32_bytes(ci);
        queue.write_buffer(&cg.cell_index, 0, &bytes);
        queue.write_buffer(
          &gpu.all_ci,
          (gpu.base_ci_words + (c as u32) * 4096) as u64 * 4,
          &bytes,
        );
      }
      if let Some(meta) = &f.meta_words {
        let layer0 = gpu.base_layers + c as u32 * 16;
        queue.write_texture(
          TexelCopyTextureInfo {
            texture: &gpu.meta_prev,
            mip_level: 0,
            origin: Origin3d {
              x: 0,
              y: 0,
              z: layer0,
            },
            aspect: TextureAspect::All,
          },
          &meta_padded_bytes(meta),
          TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(META_COPY_BPR),
            rows_per_image: Some(PROBES_PER_LAYER_AXIS),
          },
          Extent3d {
            width: PROBES_PER_LAYER_AXIS,
            height: PROBES_PER_LAYER_AXIS,
            depth_or_array_layers: 16,
          },
        );
        cascade_scroll_reset(
          &queue,
          layer0,
          shift,
          (&gpu.irr_prev, &gpu.irr_next),
          (&gpu.depth_prev, &gpu.depth_next),
        );
      }
    }
  }

  // 全域采样 uniform（binding 13）：base + 4 级级联域快照（ddgi_sample 数据源）
  *gpu.casc_u.get_mut() = DdgiDomains {
    base: base_u,
    cascades: casc_frames
      .as_ref()
      .map_or([DdgiUniform::default(); 4], |f| {
        [f[0].dom, f[1].dom, f[2].dom, f[3].dom]
      }),
  };
  gpu.casc_u.write_buffer(&device, &queue);

  let bg4_layout = pipeline_cache.get_bind_group_layout(&ddgi_bg4_layout());
  let bg4 = device.create_bind_group(
    None,
    &bg4_layout,
    &BindGroupEntries::sequential((
      &gpu.uniform,
      gpu.positions.as_entire_binding(),
      gpu.cell_index.as_entire_binding(),
      &gpu.irr_prev_view,
      &gpu.depth_prev_view,
      &gpu.irr_next_view,
      &gpu.depth_next_view,
      &gpu.meta_prev_view,
      &gpu.meta_next_view,
      gpu.dispatch.as_entire_binding(),
      gpu.objects.as_entire_binding(),
      gpu.samples.as_entire_binding(),
      gpu.worklist.as_entire_binding(),
      &gpu.casc_u,
      gpu.all_ci.as_entire_binding(),
    )),
  );
  commands.insert_resource(DdgiBg4(bg4));
  // 每级联一份 BG4_c：0/2/9/11/12 = 本级 uniform/ci/dispatch/samples/worklist；
  // 1（positions）/3-8（纹理）/10（objects）/13（全域 uniform）/14（all_ci）共享。
  // 共享借用作用域内建 BG（owned BindGroup），结束后写回——规避 ResMut 借用冲突
  for c in 0..4usize {
    let bg = {
      let Some(cg) = gpu.cascades.get(c) else {
        continue;
      };
      device.create_bind_group(
        None,
        &bg4_layout,
        &BindGroupEntries::sequential((
          &cg.uniform,
          gpu.positions.as_entire_binding(),
          cg.cell_index.as_entire_binding(),
          &gpu.irr_prev_view,
          &gpu.depth_prev_view,
          &gpu.irr_next_view,
          &gpu.depth_next_view,
          &gpu.meta_prev_view,
          &gpu.meta_next_view,
          cg.dispatch.as_entire_binding(),
          gpu.objects.as_entire_binding(),
          cg.samples.as_entire_binding(),
          cg.worklist.as_entire_binding(),
          &gpu.casc_u,
          gpu.all_ci.as_entire_binding(),
        )),
      )
    };
    if let Some(cg) = gpu.cascades.get_mut(c) {
      cg.bg4 = Some(bg);
    }
  }
  let us = tp.elapsed().as_micros();
  if us > 10_000 {
    bevy::log::info!("DDGI prepare: {us}us");
  }
}


// ============================================================================
// 单测：活跃位语义（Douglas #23 近表面判定）+ positions.w 打包
// ============================================================================

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn probe_active_only_near_surface() {
    let mut grid = gate_voxel::VolumeGrid::new();
    // 64³ 实心砖 @ cell (1,0,0)（fine [64,128)³）
    let _ = grid.fill_brick(IVec3::new(64, 0, 0), 64, 1);
    // 本 cell 面邻接实心砖 → 活跃（cell(0,0,0) 自身为 Air）
    assert!(probe_is_active(&grid, IVec3::ZERO, 64));
    // 实心砖另一侧的 Air cell → 活跃
    assert!(probe_is_active(&grid, IVec3::new(128, 0, 0), 64));
    // 对角 cell（仅角/棱邻接，无面邻接）→ 不活跃（Douglas 只查 6 面邻接）
    assert!(!probe_is_active(&grid, IVec3::new(128, 64, 0), 64));
    // 远场（间隔 >1 cell）→ 不活跃
    assert!(!probe_is_active(&grid, IVec3::new(320, 320, 320), 64));
    // 16³ cell（DDGI_CELL）粒度：+x 面邻实心砖的 cell 活跃，斜向不活跃
    assert!(probe_is_active(&grid, IVec3::new(48, 0, 0), 16));
    assert!(!probe_is_active(&grid, IVec3::new(32, 32, 0), 16));
  }

  #[test]
  fn probe_active_region_query_covers_all_cell_sizes() {
    let mut grid = gate_voxel::VolumeGrid::new();
    // 整个 chunk(0,0,0) 填实（256³ = level 0 Solid）
    let _ = grid.fill_brick(IVec3::ZERO, 256, 2);
    // cell_size 256（level 0 单次查询）：紧邻 chunk 的 cell 活跃，隔一个不活跃
    assert!(probe_is_active(&grid, IVec3::new(256, 0, 0), 256));
    assert!(!probe_is_active(&grid, IVec3::new(512, 0, 0), 256));
    // cell_size 512（2³ level 0 查询）：覆盖到 chunk 0 的 cell 活跃，未覆盖不活跃
    assert!(probe_is_active(&grid, IVec3::ZERO, 512));
    assert!(!probe_is_active(&grid, IVec3::new(512, 512, 512), 512));
    // 单体素 Mixed 砖也算含体素
    let mut grid2 = gate_voxel::VolumeGrid::new();
    let _ = grid2.set_voxel_ivec3(IVec3::new(10, 10, 10), 3);
    assert!(probe_is_active(&grid2, IVec3::ZERO, 64));
  }

  #[test]
  fn bake_grid_carries_active_flags() {
    let mut vols = Volumes::new(gate_voxel::VolumeGrid::new());
    {
      // 64³ 实心砖 @ fine [64,128)×[0,64)² = rel cells (4..8, 0..4, 0..4)（16³ cell）
      let _ = vols.main_mut().fill_brick(IVec3::new(64, 0, 0), 64, 1);
    }
    let pg = bake_probe_grid(&vols);
    let main = vols.main();
    assert_eq!(pg.positions.len(), pg.active.len());
    for z in 0..pg.grid_dims.z as i32 {
      for y in 0..pg.grid_dims.y as i32 {
        for x in 0..pg.grid_dims.x as i32 {
          let rel = UVec3::new(x as u32, y as u32, z as u32);
          let id = pg.cell_index[pg.cell_linear(rel)];
          let cell_min = IVec3::new(x, y, z) * DDGI_CELL;
          if id == NO_PROBE {
            // 只有全 Solid cell 无探针（Air/Mixed 均有）
            assert!(matches!(
              cell_state_at(main, cell_min, DDGI_CELL),
              BrickState::Solid(_)
            ));
            continue;
          }
          // 活跃位与 probe_is_active 直算一致
          assert_eq!(
            pg.active[id as usize],
            probe_is_active(main, cell_min, DDGI_CELL)
          );
        }
      }
    }
    // cell (3,0,0) 面邻实心砖 → 活跃；cell (0,0,0) 远场 → 不活跃
    assert!(
      pg.active[pg.cell_index[pg.cell_linear(UVec3::new(3, 0, 0))] as usize]
    );
    assert!(
      !pg.active[pg.cell_index[pg.cell_linear(UVec3::new(0, 0, 0))] as usize]
    );
  }

  #[test]
  fn multicolor_solid_cell_bakes_no_probe() {
    // 树 Mixed 双义回归（2026-09-08 panic 修复）：多色全实 16³ cell 沿树无空叶
    // （BrickState::Mixed 含「多色混合」义），BFS None = DDGI「全满」→ 无探针
    let mut vols = Volumes::new(gate_voxel::VolumeGrid::new());
    let _ = vols.main_mut().fill_brick(IVec3::ZERO, 16, 1);
    // 打一个异色体素 → 树 Mixed（仍无任何空气）
    let _ = vols.main_mut().set_voxel_ivec3(IVec3::ZERO, 2);
    assert!(matches!(
      cell_state_at(vols.main(), IVec3::ZERO, DDGI_CELL),
      BrickState::Mixed
    ));
    let pg = bake_probe_grid(&vols);
    assert_eq!(pg.cell_index[pg.cell_linear(UVec3::ZERO)], NO_PROBE);
    // 对照：同砖留一个空体素（真空实混合）→ 必有探针
    let _ = vols.main_mut().set_voxel_ivec3(IVec3::new(8, 8, 8), 0);
    let pg = bake_probe_grid(&vols);
    assert_ne!(pg.cell_index[pg.cell_linear(UVec3::ZERO)], NO_PROBE);
  }

  #[test]
  fn pack_positions_encodes_active_in_w() {
    let mut pg = ProbeGrid::default();
    pg.positions = vec![Vec3::splat(1.0), Vec3::splat(2.0)];
    pg.active = vec![true, false];
    let packed = pack_probe_positions(&pg);
    assert_eq!(packed[0].w, 1.0);
    assert_eq!(packed[1].w, 0.0);
    assert_eq!(
      Vec3::new(packed[1].x, packed[1].y, packed[1].z),
      Vec3::splat(2.0)
    );
  }
}