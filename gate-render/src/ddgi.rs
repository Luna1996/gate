//! R3-10 DDGI 全局光照（Douglas #23 1:1 复刻）——CPU 侧真相源。
//!
//! 职责（对应 r3-lighting.md R3-10 spike 步骤 1）：
//! - **探针烘焙**（Rohacek §3.1）：cell 16³（= level 2 brick）三态判定 →
//!   Air 居中放探针 / Solid 无探针 / Mixed 沿树 BFS 找靠中心的最大空叶；
//! - **gate 探针分配 = 活跃壳**：探针仅存在于「cell 或 6 邻接 cell 含实体」的
//!   cell——远场纯空气探针在活跃门控（Douglas #23：cell+6 邻接无体素 → 不活跃）
//!   下永不更新、数据恒零，分配它们纯属显存浪费；表面着色点的三线性 8 邻域
//!   恰好全部落在活跃壳内（表面 cell + 表面上一层），采样完备性不受影响。
//!   物体（list[1..N]）互反射探针后置（P14 流式阶段一并裁决）。
//! - **GPU wire 契约**：DdgiMeta uniform + positions/cell_index/irradiance/depth
//!   四 storage buffer 的打包布局（WGSL BG4 逐字段镜像，改一处必改两处）；
//! - **WGSL 镜像数学**：octahedral 编解码（Majercik 2019 §3）+ Fibonacci 球方向
//!   （§4 射线更新），CPU/WGSL 逐字镜像 + 单测锁死；`cpu_sample_ddgi` 为着色采样
//!   （§5）的 CPU 逐行镜像，端到端单测锁住权重/剔除/归一化行为。
//!
//! 编辑响应：VolumeGrid.edit_generation 单调计数 → extract 版本比对 → 自动重烘
//! （buffer 重建 + irradiance/depth 清零重新 EMA 收敛）。
//!
//! 尚未落地（后续会话，均需目验/性能数字驱动）：探针状态机（Production §3
//! 活跃剪枝）、LOD 级联（N=10 显存告警再启）、oct 折缝 warp 精修、
//! 物体（list[1..N]）互反射探针（P14 流式阶段一并裁决）。

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
// 探针烘焙（Rohacek §3.1 BFS 最大空叶 + gate 活跃壳分配）
// ============================================================================

/// 探针烘焙结果（CPU 真相源 → GPU 打包入口）
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ProbeGrid {
  /// 采样网格原点（cell 单位）= chunk 窗口 cell 包围盒外扩 1 格（活跃壳）
  pub grid_origin: IVec3,
  /// 采样网格 dims（cell 单位）
  pub grid_dims: UVec3,
  /// dense cell → probe 下标（NO_PROBE = 无探针）
  /// 线性地址 = x + y*dims.x + z*dims.x*dims.y（cell 相对 grid_origin）
  pub cell_index: Vec<u32>,
  /// 探针位置（世界 fine 坐标；下标 = probe id）
  pub positions: Vec<Vec3>,
}

impl ProbeGrid {
  /// cell（相对 grid_origin 的偏移）→ 线性下标
  #[inline]
  pub fn cell_linear(&self, rel: UVec3) -> usize {
    (rel.x + rel.y * self.grid_dims.x + rel.z * self.grid_dims.x * self.grid_dims.y) as usize
  }
}

/// 烘焙主世界探针网格（当前只覆盖 vols.list[0]；物体互反射探针后置）
///
/// 稀疏遍历：每个已存在 chunk 迭代其 18³ cell 区（本 chunk 16³ + 边界 ±1 ring），
/// 不扫 chunk bbox 全空间（N=10 稀疏世界 320³ cell 稠密表 ≈130MB 浪费）。
/// ring cell 与邻 chunk 重复（border cell ≤2 次访问）：cell_index 已分配即跳过，
/// 且探针位置是状态的纯函数（Air 居中 / BFS 确定性），重复结果幂等。
pub fn bake_probe_grid(vols: &Volumes) -> ProbeGrid {
  let grid = vols.main();
  let chunks: Vec<IVec3> = grid.chunk_coords().map(|c| c.0).collect();
  let Some((min_chunk, max_chunk)) = chunk_bbox(chunks.iter().copied()) else {
    return ProbeGrid::default();
  };
  // cell 包围盒 = chunk 窗口 cell 域外扩 1 格（ring：贴面空气 cell 也要探针）
  let lo = min_chunk * CELLS_PER_CHUNK - 1;
  let hi_excl = (max_chunk + 1) * CELLS_PER_CHUNK + 1;
  let dims = (hi_excl - lo).as_uvec3();

  let mut pg = ProbeGrid {
    grid_origin: lo,
    grid_dims: dims,
    cell_index: vec![NO_PROBE; dims.x as usize * dims.y as usize * dims.z as usize],
    positions: Vec::new(),
  };

  for chunk in &chunks {
    // 18³ cell：ring = -1..=16（相对 chunk cell 原点）
    for rz in -1..=CELLS_PER_CHUNK {
      for ry in -1..=CELLS_PER_CHUNK {
        for rx in -1..=CELLS_PER_CHUNK {
          let cell = chunk * CELLS_PER_CHUNK + IVec3::new(rx, ry, rz);
          let rel = (cell - lo).as_uvec3();
          let li = pg.cell_linear(rel);
          if pg.cell_index[li] != NO_PROBE {
            continue; // 邻 chunk ring 已处理（幂等去重）
          }
          let state =
            grid.get_brick_state(VoxelCoord::from_ivec3(cell * DDGI_CELL), DDGI_CELL_LEVEL);
          // 全实心 cell 无探针（Douglas：全满 → 无探针）
          if matches!(state, BrickState::Solid(_)) {
            continue;
          }
          // 活跃壳：本 cell 或 6 邻接 cell 含实体（Air cell 必须邻实心才有探针；
          // Mixed cell 自含实体恒过）。远场纯空气 = 无探针 = 采样时权重剔除。
          if state == BrickState::Air && !neighbor_has_solid(grid, cell) {
            continue;
          }
          let Some(pos) = probe_position(grid, cell * DDGI_CELL, state) else {
            continue; // Mixed 但中心邻域全满（薄墙 cell）：无探针
          };
          pg.cell_index[li] = pg.positions.len() as u32;
          pg.positions.push(pos);
        }
      }
    }
  }
  pg
}

/// 6 邻接 cell 是否含实体（Douglas #23 活跃判据的最小闭合）
fn neighbor_has_solid(grid: &gate_voxel::VolumeGrid, cell: IVec3) -> bool {
  const DIRS: [IVec3; 6] = [
    IVec3::X,
    IVec3::NEG_X,
    IVec3::Y,
    IVec3::NEG_Y,
    IVec3::Z,
    IVec3::NEG_Z,
  ];
  DIRS.iter().any(|d| {
    grid.get_brick_state(
      VoxelCoord::from_ivec3((cell + *d) * DDGI_CELL),
      DDGI_CELL_LEVEL,
    ) != BrickState::Air
  })
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
  let center_f = cell_min.as_vec3() + Vec3::splat(DDGI_CELL as f32 / 2.0);
  match state {
    BrickState::Air => Some(center_f),
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
        return best.map(|(_, c)| c);
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
      best.map(|(_, c)| c)
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
// GPU wire（BG4：WGSL DdgiMeta + 4 storage buffer 逐字段镜像）
// ============================================================================

/// BG4 meta uniform（WGSL `DdgiMeta` 逐字段镜像，64B）
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bevy::ecs::resource::Resource, ShaderType)]
pub struct DdgiMeta {
  /// 烘焙出的探针总数（= positions/cell_index 有效域）
  pub probe_count: u32,
  /// 活跃探针数（v1 = probe_count：分配即活跃壳）
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
        let dir = to / dist.max(1e-4);
        // 锐利背面剔除（Rohacek §3.2）
        let wn = (n.dot(dir) / DDGI_NORMAL_BIAS).clamp(0.0, 1.0);
        if wn <= 0.0 {
          continue;
        }
        // 漏光 chevron（Rohacek §3.3）
        let dtex = cpu_depth_sample(a.depth, id, dir);
        let wd = ((dtex - dist) / DDGI_DEPTH_BIAS + 0.5).clamp(0.0, 1.0);
        if wd <= 0.0 {
          continue;
        }
        let irr = cpu_irr_sample(a.irr, id, dir);
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

  /// 16³ 实心盒放在 cell (0,0,0) 的世界（其余空）
  fn world_with_cell_box() -> VolumeGrid {
    let mut g = VolumeGrid::new();
    g.palette_mut().get_mut(3).color = [128, 64, 32];
    gate_voxel::fill_bricks(&mut g, IVec3::ZERO, IVec3::splat(16), 4, 3);
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

  /// 实心 cell 无探针；贴面空气 cell（活跃壳）有探针且 cell_index 往返一致
  #[test]
  fn bake_solid_and_shell() {
    let g = world_with_cell_box();
    let vols = Volumes::new(g);
    let pg = bake_probe_grid(&vols);
    assert!(!pg.positions.is_empty());

    let solid_cell = IVec3::ZERO;
    let lookup = |pg: &ProbeGrid, cell: IVec3| -> u32 {
      let rel = (cell - pg.grid_origin).as_uvec3();
      pg.cell_index[pg.cell_linear(rel)]
    };
    // 实心 cell 本体：无探针
    assert_eq!(lookup(&pg, solid_cell), NO_PROBE);
    // 贴面空气 cell（活跃壳）：有探针且在 cell 中心（Air 居中）
    for d in [
      IVec3::X,
      IVec3::NEG_X,
      IVec3::Y,
      IVec3::NEG_Y,
      IVec3::Z,
      IVec3::NEG_Z,
    ] {
      let air_shell = solid_cell + d;
      let idx = lookup(&pg, air_shell);
      assert_ne!(idx, NO_PROBE, "壳 cell {air_shell} 应有探针");
      let expect = (air_shell * DDGI_CELL).as_vec3() + Vec3::splat(8.0);
      assert_eq!(
        pg.positions[idx as usize], expect,
        "壳 cell {air_shell} 居中"
      );
    }
    // 壳外第二层纯空气 cell：无探针
    assert_eq!(lookup(&pg, solid_cell + IVec3::X * 2), NO_PROBE);
    assert_eq!(lookup(&pg, solid_cell + IVec3::X + IVec3::Y), NO_PROBE);
    // 探针总数 = 6 壳 cell（实心本体 + 壳外全空气都无探针）
    assert_eq!(pg.positions.len(), 6);
    // 网格覆盖 = chunk 域 ±1
    assert_eq!(pg.grid_origin, IVec3::splat(-1));
    assert_eq!(pg.grid_dims, UVec3::splat((CELLS_PER_CHUNK + 2) as u32));
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
    assert_eq!(meta.active_count, meta.probe_count, "v1 分配即活跃壳");
    assert_eq!(meta.probes_this_frame, plan.probes_this_frame);
    assert_eq!(meta.grid_origin, Vec4::new(-1.0, -1.0, -1.0, 0.0));
    assert_eq!(meta.grid_dims, Vec4::new(18.0, 18.0, 18.0, 0.0));
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
    // p=(12,12,12) 在 8 cell 交界，n=+Y：下方 4 探针贡献，归一化后仍为探针色
    let c = cpu_sample_ddgi(&a, Vec3::splat(12.0), Vec3::Y);
    assert!(
      (c - Vec3::new(1.0, 0.2, 0.1)).length() < 1e-5,
      "归一化加权和 = {c:?}"
    );
  }

  /// 探针全在表面法线后侧（p 在所有探针上方）→ wsum=0 → 黑
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
    // p.y=4 < 所有探针 y(8/24)：探针全在 p 上方，n=+Y → N·d<0 全剔除
    let c = cpu_sample_ddgi(&a, Vec3::new(12.0, 4.0, 12.0), Vec3::Y);
    assert_eq!(c, Vec3::ZERO, "背面探针必须全剔除");
  }

  /// depth 全 1.0（探针贴墙）而 p 距探针 ~7 → chevron wd=0 → 黑（漏光治理）
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
    // n=-Y 取上方探针（p 上方 y=24 的探针 dir.y>0 与 -Y 同向），depth=1 全遮挡
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
    // p 在探针正上方 1 格，n=+Y → dir≈+Y → 红
    let up = cpu_sample_ddgi(&a, Vec3::new(20.0, 12.5, 20.0), Vec3::Y);
    assert!(up.x > 0.8 && up.z < 0.2, "朝上采样应见红半球：{up:?}");
    // p 在探针正下方 1 格，n=-Y → dir≈-Y → 蓝
    let down = cpu_sample_ddgi(&a, Vec3::new(20.0, 10.5, 20.0), -Vec3::Y);
    assert!(down.z > 0.8 && down.x < 0.2, "朝下采样应见蓝半球：{down:?}");
  }

  /// 64³ 封闭房间（六面 4 厚墙）烘焙分布：内部空气 cell 有探针、墙 cell 无探针、
  /// 远场 cell 无探针、Mixed 墙顶 cell 探针偏移到空气
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
    // 房间中心 cell（fine 32..48）：Air 且邻接天花 Mixed cell → 有探针，居中 (40,40,40)
    let center = IVec3::new(2, 2, 2);
    let id = lookup(center);
    assert_ne!(id, NO_PROBE, "房间内部 cell 应有探针（活跃壳含墙邻接）");
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
    // 远场 cell（fine 80..96，距墙 ≥2 cell）：无探针
    assert_eq!(lookup(IVec3::new(5, 5, 5)), NO_PROBE);
    assert_eq!(lookup(IVec3::new(10, 10, 10)), NO_PROBE);
    // 探针数量合理性：dense 18³=5832 cell 中仅表面壳 + ring 贴面有探针（~100 量级）
    let n = pg.positions.len();
    assert!(n > 40 && n < 200, "探针数 = {n}");
  }

  /// 编辑驱动重烘：空世界 0 探针 → 填盒 6 探针（代数推进）→ 挖空回 0 探针；
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
    assert_eq!(bake_probe_grid(&vols).positions.len(), 6, "盒壳 6 探针");

    // 同色重复写 = noop，代数不变（重烘判据不抖动）
    vols.main_mut().set_voxel_ivec3(IVec3::new(0, 0, 0), 3);
    assert_eq!(vols.main().edit_generation(), gen_filled);

    // 挖空（fill palette=0 逐 brick 清空气）→ 回到无探针
    gate_voxel::fill_bricks(vols.main_mut(), IVec3::ZERO, IVec3::splat(16), 4, 0);
    assert!(vols.main().edit_generation() > gen_filled, "clear 推进代数");
    assert_eq!(bake_probe_grid(&vols).positions.len(), 0, "挖空后无探针");
  }
}
