//! DDGI（Dynamic Diffuse Global Illumination）——阶段一：世界空间探针烘焙 + 活跃探针筛选。
//!
//! 架构（严格对齐 Douglas Devlog #23 / Majercik 2019, 2021）：
//! - **嵌套级联 LOD，以相机为中心**：4 级 LOD 各自以相机为中心铺 16³ 网格（cell 边长
//!   16/32/64/128 voxel），覆盖范围逐级 ×2 且严格嵌套（LOD(l-1) 盒 ⊂ LOD(l) 盒）。
//!   相机移动使某级 origin 按该级 cell 对齐滚动时，只重烘「世界 cell 发生变化」的槽位
//!   （`ddgi_cell_id` 增量缓存），未变的槽位续龄。
//! - **烘焙（`ddgi_bake0..3`）**：世界数据变化（上传修订号自增）时才跑一次。逐 cell 沿 4³ 分裂树
//!   **BFS 找「最大的全空叶」并把探针放在其中心**（同级优先靠 cell 中心；全满 cell 无探针；
//!   全空 cell 居中）——即 Douglas 的探针放置启发式。结果写入 `ddgi_cell` storage buffer。
//!   按 LOD 拆成 4 个独立 compute pass（细→粗）：粗级要继承**本帧**细级的放置结果
//!   （Douglas 的 "down sample the generated data"），而 pass 边界才是内存屏障。
//! - **活跃判定（`ddgi_sort`）**：每帧逐 cell，读烘焙记录（不再重算树 BFS）；探针存在且
//!   「本 cell 或 6 邻接 cell 有体素」（或与非网格对齐物体 AABB 重叠）→ 活跃 → atomicAdd 进
//!   per-LOD worklist；同时刷新 age / slot_pos / meta。
//! - **seal**：按 LOD 把固定射线预算摊给活跃探针，写 cast/collect indirect args。
//!
//! 阶段二（cast）/ 阶段三（collect/着色）尚未接入；`irr/depth` 纹理沿用旧布局暂作占位。

use bevy::render::render_resource::{CachedComputePipelineId, ShaderType};
use glam::{IVec3, IVec4, UVec3, UVec4, Vec4};

/// 辐照度图每探针 4×4（= 16 纹素）。
///
/// 【为什么不用原版的 8×8】曾对齐 DDGI 原版试过 8×8 / 16×16（见 DEPTH_TEXELS 注释）：
/// 实测**画质没有明显改善**，代价却很实在（显存 ×4、collect 线程 ×4、帧轮换周期 ×4）。
/// 方向分辨率降低带来的模糊由 DDGI_ALPHA 的时间累积与纹素间插值承担。
/// 必须与 WGSL `DDGI_IRR_TEXELS` 一致。
pub const IRRADIANCE_TEXELS: u32 = 4;
/// 深度图每探针 8×8（= 64 纹素）。
///
/// 【为什么不用原版的 16×16】曾试过 16×16（每纹素 ~11°，现在 ~22°）：**实测漏光没有明显
/// 改善** —— 说明当时的漏光主因不在深度角分辨率，而在别处（射线方向未绑定纹素 / 借针跨墙 /
/// 级联硬切，见 dda.wgsl 对应注释）。代价则是深度图集 64MB → 256MB、collect 线程 80 → 320、
/// 帧轮换周期 ×4。故回退到 8×8。
/// 必须与 WGSL `DDGI_DEPTH_TEXELS` 一致。
pub const DEPTH_TEXELS: u32 = 8;
pub const PROBE_T_MAX: f32 = 8192.0;
/// 每帧射线总预算。WGSL `ddgi_seal` 把它**均分**给全部活跃探针（rpp = 预算 / 活跃数，
/// 钳在 [1, 256]），所以这是一条直接线性的画质/帧时旋钮：减半 → cast/collect 的帧时也
/// 大致减半，代价是每探针样本减半、图集噪声变大。
///
/// 【历史】曾试过 262144 / 1048576 来压制"探针晶格亮斑"与深度闸门抖动，**帧时涨了但问题
/// 没解决**（根因是探针网格对贴缝尺度欠采样 + 深度的角度均值偏差，不是射线数量），已退回
/// 131072。若要再动这条线，请先备好可验证的收益。
/// 必须与 WGSL `DDGI_RAY_BUDGET` 保持一致。
pub const DDGI_RAY_BUDGET: u32 = 131072;

pub const DDGI_LODS: u32 = 4;
/// 最细 LOD 的 cell 边长（voxel）。
pub const DDGI_BASE_CELL: i32 = 16;
/// 4 级 LOD cell 边长（voxel），等比 ×2。
///
/// 上限受 `ddgi_cell_state_sized` 支持（16/32/64/128/256）约束。取 [16,32,64,128]：
/// 探针数 / 射线预算 / 显存全不变（dims 不变），只是把同样的探针铺在**更小的体积**上 ——
/// 近场探针间距 64cm→32cm。这是"探针晶格"伪影（GI 场的空间变化比探针网格更细时，
/// 三线性插值把每个探针自己的值暴露成 0.64m 周期的亮斑）最直接的降压手段。
///
/// 【为什么不能把最粗级放大到 256 来换覆盖】2026-09-12 试过并回退：覆盖 82m → 164m 且
/// 槽位/预算/显存成本≈0（dims 没动），但**探针间距 2.56m → 5.12m** —— 对建筑尺度会严重
/// 跨几何（8 角探针跨到墙背面/屋面之上/地面之下）→ 该面被 `wn ≤ 0` 全剔 → GI 黑区
/// （Probe 档品红、GI/wsum 档纯黑），实测**比"覆盖不足"更伤画质**。
/// **cell 与 dims 是一对此消彼长的量**：要"覆盖更大 + 间距不变"，只能加大 dims
/// （或增加级数），代价落在 collect/sort/图集，而不是改 cell。
pub const DDGI_LOD_CELL_SIZES: [i32; DDGI_LODS as usize] = [16, 32, 64, 128];
/// 各级 LOD 的 cell 维度（4 级相同）。每级 cell ×2 且维度不变 → 覆盖范围逐级 ×2，
/// 形成严格嵌套的级联。水平 32 格、垂直 16 格（体素世界水平视野远大于垂直）。
/// cell [16,32,64,128] 时各级覆盖范围 = dims×cell：
/// 512×256×512 / 1024×512×1024 / 2048×1024×2048 / 4096×2048×4096 voxel
/// = 10.2×5.1×10.2 / 20.5×10.2×20.5 / 41×20.5×41 / 82×41×82 m（半宽到 ±41m）。
/// 每级 16384 槽 → 共 65536 槽（= 占位纹理容量，全部槽位可采样）。
pub const DDGI_LOD_DIMS: UVec3 = UVec3::new(32, 16, 32);
// 槽位映射是**世界锚定**的：shader 里 `slot = slot_base + (世界 cell 号 mod dims)`（见
// `ddgi_slot`）。因此「槽位 ↔ 世界 cell」的身份与相机无关 —— 相机滚动只会让「新进入窗口的
// 那条带」换掉世界 cell（旧数据本来就该丢），其余槽位保持自己的世界身份，图集不会因相机
// 移动而整体失效。旧版槽位是「相对相机窗口的格号」，滚一格就把整级所有槽位的世界 cell
// 全换掉 → 整级图集变成旧位置的读数 → 深度判定成片失败（Probe 大片红）+ 下一帧重写
// （大片绿），即「相机移动时的 GI 闪烁」。前提：`from_camera` 的原点必须是 cell 整数倍。

// ---- LOD0 的 chunk 锚定（其它 LOD 保持上面的世界 AABB 规则网格）----
//
// LOD0 不再铺「一整个世界 AABB 的规则网格」，而是**按 chunk 拥有**：世界体素卷按
// `DDGI_CHUNK_VOXELS`(256³) 切成 chunk，每个「需要探针」的 chunk 从探针池里领一段**固定
// 大小**的 LOD0 槽位（4096 = (256/16)³），chunk 内的 cell 编址为 chunk 局部：
//     slot = chunk_base[chunk] + local_cell_linear（局部 cell 索引，见 `DDGI_CHUNK_LOD0_AXIS`）
// 探针世界位置 = chunk 世界原点 + 局部 cell 中心 —— **相对世界固定**（与旧实现一致），
// 所以「相机移动不闪」这条不变式不受影响。
//
// 边界与不变量（务必与 WGSL 的 `ddgi_slot_own`/`ddgi_slot_world_cell` 对照）：
//   · **只服务 LOD0**。LOD1~3 仍由 `DdgiWorldGrid::from_world` 的规则网格提供（滚动频率低、
//     伪影不明显），槽位基址排在 LOD0 段之后。
//   · chunk 段的基址**一旦分配就不再改变**（只有 chunk 被释放才归还进空闲链表）。基址一变，
//     该 chunk 全部探针就会换槽位 → 图集整段错位 → 与「世界锚定」同样的闪烁。
//   · 池是**高水位**定容的：`lod0_slots = next_base`，空闲段的空洞同样占槽位（无流式加载时
//     没有释放，等价于紧凑分配）。
/// chunk 边长（voxel）：世界体素卷按它切块（= 引擎自己的 brick chunk 粒度）。
pub const DDGI_CHUNK_VOXELS: i32 = 256;
/// 每 chunk 每轴含多少个 LOD0 cell：256 / 16 = 16。
pub const DDGI_CHUNK_LOD0_AXIS: i32 = DDGI_CHUNK_VOXELS / DDGI_LOD_CELL_SIZES[0];
/// 每 chunk 的 LOD0 段大小 = (256/16)³ = 4096。
pub const DDGI_CHUNK_LOD0_SLOTS: u32 =
  (DDGI_CHUNK_LOD0_AXIS as u32) * (DDGI_CHUNK_LOD0_AXIS as u32) * (DDGI_CHUNK_LOD0_AXIS as u32);
/// 探针池里「该 chunk 未分配段」的哨兵基址。
pub const DDGI_CHUNK_NO_BASE: u32 = u32::MAX;

/// LOD0 的 chunk 网格几何（chunk 单位）。
///
/// 原点 = 世界 AABB 向下对齐到 256 后再**向外扩 1 chunk**，维度 = 覆盖 AABB 所需 chunk 数
/// **+2**。多这一圈是为了给「贴着 AABB 边界、需要在邻 chunk 放探针」的 chunk 留出编址空间；
/// 未分配段的 chunk 由 `DDGI_CHUNK_NO_BASE` 标记。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DdgiChunkGeom {
  pub origin: IVec3,
  pub dims: UVec3,
}

impl Default for DdgiChunkGeom {
  fn default() -> Self {
    Self {
      origin: IVec3::ZERO,
      dims: UVec3::ONE,
    }
  }
}

impl DdgiChunkGeom {
  pub fn from_world(aabb_min: IVec3, aabb_max: IVec3) -> Self {
    let c = DDGI_CHUNK_VOXELS;
    let aligned = IVec3::new(
      align_down(aabb_min.x, c),
      align_down(aabb_min.y, c),
      align_down(aabb_min.z, c),
    ) / c
      - IVec3::ONE;
    let span = (aabb_max - aligned * c).max(IVec3::ONE);
    let dims = UVec3::new(
      ((span.x + c - 1) / c).max(1) as u32 + 2,
      ((span.y + c - 1) / c).max(1) as u32 + 2,
      ((span.z + c - 1) / c).max(1) as u32 + 2,
    );
    Self {
      origin: aligned,
      dims,
    }
  }

  #[inline]
  pub fn len(&self) -> u32 {
    self.dims.x * self.dims.y * self.dims.z
  }

  /// chunk 坐标 → 网格内线性下标（含边界检查）
  #[inline]
  pub fn linear(&self, cc: IVec3) -> Option<u32> {
    let r = cc - self.origin;
    if r.x < 0
      || r.y < 0
      || r.z < 0
      || r.x as u32 >= self.dims.x
      || r.y as u32 >= self.dims.y
      || r.z as u32 >= self.dims.z
    {
      return None;
    }
    Some((r.x as u32) + (r.y as u32) * self.dims.x + (r.z as u32) * self.dims.x * self.dims.y)
  }

  #[inline]
  pub fn coord(&self, idx: u32) -> IVec3 {
    let d = self.dims;
    self.origin
      + IVec3::new(
        (idx % d.x) as i32,
        ((idx / d.x) % d.y) as i32,
        (idx / (d.x * d.y)) as i32,
      )
  }
}

/// LOD0 探针池：chunk → 段基址（段大小恒为 `DDGI_CHUNK_LOD0_SLOTS`），配一个空闲链表。
///
/// 【为什么要有池】现在没有 chunk 流式加载（见 `vox_scene`：整个 nuke.vox 一次性载入），
/// 所以这条路的收益是**内容驱动的内存节省** —— 只有「有几何」（或紧邻几何）的 chunk 才领段，
/// 空 chunk 不占 LOD0 槽位；池 + 空闲链表是为将来接流式（chunk 卸载时归还段）预留的接口。
///
/// 【为什么不重排基址】`sync` 只做「新 chunk 领段、消失的 chunk 归还」，**已有 chunk 的基址
/// 保持不变** —— 这是「移动/编辑不闪」的前提（基址变了 = 该 chunk 全部探针换槽位，图集里
/// 还是旧位置的值 → 误差被放大）。
#[derive(Clone, Debug, Default)]
pub struct DdgiChunkPool {
  pub geom: DdgiChunkGeom,
  /// 每 chunk 的段基址（`DDGI_CHUNK_NO_BASE` = 未分配）
  pub bases: Vec<u32>,
  /// 空闲段基址（chunk 释放时归还，LIFO 复用）
  pub free: Vec<u32>,
  /// 池高水位（下一个新段基址）→ LOD0 槽位数 = 它
  pub next_base: u32,
  /// 任何分配/归并都自增：`prepare_ddgi` 据此判定「池变了 → 需要重烘」。
  pub serial: u64,
}

impl DdgiChunkPool {
  #[inline]
  pub fn lod0_slots(&self) -> u32 {
    self.next_base
  }

  /// 与期望的 chunk 集合同步。返回「是否发生变化」。
  pub fn sync(&mut self, geom: DdgiChunkGeom, wanted: &[IVec3]) -> bool {
    let mut changed = false;
    if self.geom != geom || self.bases.len() != geom.len() as usize {
      self.geom = geom;
      self.bases = vec![DDGI_CHUNK_NO_BASE; geom.len() as usize];
      self.free.clear();
      self.next_base = 0;
      changed = true;
    }
    let mut want: Vec<u32> = wanted
      .iter()
      .filter_map(|cc| self.geom.linear(*cc))
      .collect();
    want.sort_unstable();
    want.dedup();
    for &l in want.iter() {
      if self.bases[l as usize] == DDGI_CHUNK_NO_BASE {
        let base = self.free.pop().unwrap_or_else(|| {
          let b = self.next_base;
          self.next_base += DDGI_CHUNK_LOD0_SLOTS;
          b
        });
        self.bases[l as usize] = base;
        changed = true;
      }
    }
    for l in 0..self.bases.len() {
      let b = self.bases[l];
      if b != DDGI_CHUNK_NO_BASE && want.binary_search(&(l as u32)).is_err() {
        self.bases[l] = DDGI_CHUNK_NO_BASE;
        self.free.push(b);
        changed = true;
      }
    }
    if changed {
      self.serial = self.serial.wrapping_add(1);
    }
    changed
  }
}

/// LOD0 段如何编址 —— uniform 里给 shader 的那两个 vec4（见 WGSL `DdgiUniform.chunk`）。
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct DdgiChunkUniform {
  /// xyz = chunk 网格原点（chunk 坐标），w = 每 chunk 每轴 cell 数（16）
  pub origin: IVec4,
  /// xyz = chunk 网格维度，w = LOD0 已分配槽数（池高水位）
  pub dims: UVec4,
}

/// 主世界算出的「LOD0 需要探针段的 chunk」集合（chunk 坐标，由主世界体素内容决定）。
///
/// 判定规则（内容驱动，见任务点 4）：chunk 自己有几何，**或**它的邻域（cell 粒度）内有几何
/// —— 后者保证「贴着几何表面的采样者，其 8 个插值角格能拿到探针」：采样者的角格最多跨到
/// 相邻 cell，而相邻 cell 若落在邻 chunk，就必须给那个 chunk 也分配段，否则那 4 个角会凭空
/// 缺失（在 chunk 边界上留下可见接缝）。不需要「借针」式间接表。
#[derive(bevy::ecs::resource::Resource, Clone, Debug, Default, PartialEq)]
pub struct DdgiLod0Chunks {
  /// 需要 LOD0 段的 chunk 坐标（世界体素坐标 / 256）。
  pub chunks: Vec<IVec3>,
}

// indirect args / 计数器合一 buffer（word 布局与 WGSL DDGI_INDIR_* 对应）：
//   [0..16) cast args ×4 LOD；[16..32) collect args ×4 LOD；[32..36) rpp；[36..40) 活跃计数器
pub const DDGI_INDIRECT_BYTES: u64 = 256;
pub const DDGI_COUNTER_CLEAR_OFFSET: u64 = 36 * 4;
pub const DDGI_COUNTER_CLEAR_BYTES: u64 = 16;
pub const DDGI_WORKLIST_ITEM_BYTES: u64 = 16; // vec4(probe_pos.xyz, packed age|lod)
/// meta word = age(8 bit) | ENABLED<<8 | ACTIVE<<9；0 = 无探针哨兵。
pub const DDGI_META_ENABLED: u32 = 1 << 8;
/// ACTIVE：本帧需要投线（本 cell 或 6 邻接有体素/物体，且不在更细 LOD 覆盖内）。
pub const DDGI_META_ACTIVE: u32 = 1 << 9;

#[inline]
fn align_down(v: i32, a: i32) -> i32 {
  v.div_euclid(a) * a
}

/// 世界空间探针网格（4 级嵌套级联，各自独立原点，以相机为中心）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DdgiWorldGrid {
  /// 各 LOD 的世界原点（voxel，按该级 cell 边长对齐）。
  pub lod_origins: [IVec3; DDGI_LODS as usize],
  /// 各 LOD 的 cell 维度（cell 单位）。
  pub lod_dims: [UVec3; DDGI_LODS as usize],
  /// 各 LOD 在全局 slot 数组中的起始下标。
  pub lod_slot_base: [u32; DDGI_LODS as usize],
  pub total_slots: u32,
}

impl DdgiWorldGrid {
  #[inline]
  pub fn lod_count(&self, lod: usize) -> u32 {
    let d = self.lod_dims[lod];
    d.x * d.y * d.z
  }

  #[inline]
  pub fn lod_cell_size(&self, lod: usize) -> i32 {
    DDGI_LOD_CELL_SIZES[lod]
  }

  /// 由**世界 AABB** 推导 4 级嵌套级联 —— 四级**全部锚定世界**，相机完全不参与。
  ///
  /// 【为什么这样】Douglas Devlog #23 的 DDGI 在相机移动时**绝不闪烁**，因为他的网格
  /// **不随相机变**；而"相机中心窗口"一滚就换主（槽位 `mod dims` 复用）→ 整圈探针换主
  /// → 移动时闪（本会话实测多轮）。四级都锚定世界后，相机移动**不改变任何一级的原点**
  /// → 结构上不存在"换主"。
  ///
  /// 【dims 按世界算】每级 dims = ceil(世界跨度 / cell)。cell 逐级 ×2、AABB 相同 ⇒
  /// dims 逐级减半 ⇒ LOD(l-1) 盒严格包含于 LOD(l) 盒内（嵌套不变式）。
  ///
  /// 【稀疏性从哪来】细级按世界算有 ~33 万个 cell（cell=16），但绝大多数是纯空气 →
  /// `near` 判定为假 → 不进 worklist → **不参与 cast/collect**。代价只在图集容量
  /// （409600 槽 ≈ 240MB）和 `sort`（每帧扫全槽位，+0.2ms）。
  ///
  /// 【原点对齐】按 cell 向下对齐，保证 shader 里「世界 cell 号 = (p - origin) / cell」
  /// 精确整除 —— 世界锚定的槽位映射（`slot = base + 世界 cell mod dims`）才成立。
  pub fn from_world(aabb_min: IVec3, aabb_max: IVec3) -> Self {
    let mut out = Self::default();
    let mut base = 0u32;
    for lod in 0..DDGI_LODS as usize {
      let cell = DDGI_LOD_CELL_SIZES[lod];
      let origin = IVec3::new(
        align_down(aabb_min.x, cell),
        align_down(aabb_min.y, cell),
        align_down(aabb_min.z, cell),
      );
      let span = (aabb_max - origin).max(IVec3::ONE);
      let c = cell as i32;
      let dims = UVec3::new(
        ((span.x + c - 1) / c).max(1) as u32,
        ((span.y + c - 1) / c).max(1) as u32,
        ((span.z + c - 1) / c).max(1) as u32,
      );
      out.lod_origins[lod] = origin;
      out.lod_dims[lod] = dims;
      out.lod_slot_base[lod] = base;
      base += dims.x * dims.y * dims.z;
    }
    out.total_slots = base;
    out
  }

  #[inline]
  pub fn is_empty(&self) -> bool {
    self.total_slots == 0
  }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct DdgiLod {
  /// xyz = 世界原点（voxel），w = cell 边长
  pub origin: IVec4,
  /// xyz = cell 维度，w = 全局 slot 起始下标
  pub dims: UVec4,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bevy::ecs::resource::Resource, ShaderType)]
pub struct DdgiUniform {
  pub lods: [DdgiLod; 4],
  /// x = frame, y = debug mode, z = gain, w = 保留（恒 0）
  pub params: Vec4,
  /// x = shade GI, y = total slots
  pub misc: Vec4,
  /// 脏区（世界 voxel AABB）：xyz = min，w = 1 表示有效（0 = 本帧无脏区）
  pub dirty_min: Vec4,
  /// xyz = max（不含）；与 dirty_min 一起决定哪些 cell 强制重烘（局部编辑增量）
  pub dirty_max: Vec4,
  /// LOD0 的 chunk 段编址（仅 lod==0 用；LOD1~3 走 `lods`）。见 `DdgiChunkUniform`。
  pub chunk: DdgiChunkUniform,
}

pub fn ddgi_bg4_layout() -> bevy::render::render_resource::BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
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
    "DdgiBg4",
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
      // 1/2：irradiance / depth 图集（采样侧；cast 回读 GI、着色 ddgi_sample 用）
      tex(1, TextureSampleType::Float { filterable: true }),
      tex(2, TextureSampleType::Float { filterable: false }),
      // 3：烘焙输出（bake 写 / sort 读）4：age/flags（读写）5：indirect/counter（读写）
      // 6：worklist（读写）7：slot_pos（读写）8：cell_id（读写，滚动增量）
      // 9：cast 射线样本（cast 写 / collect 读）
      // 10：cell→slot 间接表（读写）—— 允许一个 cell 指向**邻近 cell 的探针**，
      //     这样"探针必须离表面足够远"和"每个采样点都有 8 个可用角"可以同时成立。
      //     初始化成"指向自身"时与旧行为逐位等价（见 create 处的 identity 填充）。
      buf(3, false),
      buf(4, false),
      buf(5, false),
      buf(6, false),
      buf(7, false),
      buf(8, false),
      buf(9, false),
      buf(10, false),
    ],
  )
}

/// BG5（仅 collect 用）：图集的**写入侧**。与 BG4 分离是硬性要求——同一纹理不能在同一
/// bind group / 同一 pass 内既作采样纹理又作存储纹理；collect 只写图集，其余 pass 只读。
pub fn ddgi_bg5_layout() -> bevy::render::render_resource::BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
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
  BindGroupLayoutDescriptor::new(
    "DdgiBg5",
    &[
      store(0, TextureFormat::Rgba16Float),
      // depth 图集也从单通道升到 Rgba16Float：.x = mean、.y = std（距离标准差），
      // 供采样侧做 Chebyshev 软遮挡（参考 Majercik/RTXGI）。R32Float 只有均值，
      // 只能做刀锋判定 → 深度一抖就"入选/落选"翻转（亮区边界伸缩）。
      store(1, TextureFormat::Rgba16Float),
    ],
  )
}

/// BG6（仅 seal 用）：dispatch 参数 buffer 的**写入侧**。
/// 与 BG4 分离是硬性要求：该 buffer 在 cast/collect pass 里要作 indirect 参数源，
/// 若同时被绑成 storage 会被 wgpu 判为 usage 冲突。
pub fn ddgi_bg6_layout() -> bevy::render::render_resource::BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  BindGroupLayoutDescriptor::new(
    "DdgiBg6",
    &[BindGroupLayoutEntry {
      binding: 0,
      visibility: C,
      ty: BindingType::Buffer {
        ty: BufferBindingType::Storage { read_only: false },
        has_dynamic_offset: false,
        min_binding_size: None,
      },
      count: None,
    }],
  )
}

#[derive(Debug, Clone, Copy)]
pub struct DdgiPipelines {
  /// per-LOD 烘焙入口（lod 0..DDGI_LODS，细→粗）。**必须逐级拆成独立 pass**：
  /// 粗级的探针放置要读本帧细级的 bake 输出，而同一 pass 内没有顺序保证，
  /// 只有 pass 边界才是内存屏障（见 WGSL `ddgi_bake_one`）。
  pub bake: [CachedComputePipelineId; DDGI_LODS as usize],
  pub sort: CachedComputePipelineId,
  pub seal: CachedComputePipelineId,
  pub cast: CachedComputePipelineId,
  pub collect: CachedComputePipelineId,
}

/// 图集的一侧（纹理 + D2Array 视图）
pub struct DdgiAtlas {
  pub tex: bevy::render::render_resource::Texture,
  pub view: bevy::render::render_resource::TextureView,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiGpu {
  pub uniform: bevy::render::render_resource::UniformBuffer<DdgiUniform>,
  pub indirect: bevy::render::render_resource::Buffer,
  /// dispatch 参数（`INDIRECT`）。**必须与 `ddgi_indirect` 分开**：同一 pass 内一个 buffer 若
  /// 既作 storage 绑定又作 indirect 参数源，wgpu 会判定 usage 冲突（`STORAGE_READ_WRITE` 是独占用法）。
  /// 它只在 seal pass 里以 storage 绑定（BG6）；cast/collect 只把它当 indirect 源，不绑定。
  pub args: bevy::render::render_resource::Buffer,
  pub worklist: bevy::render::render_resource::Buffer,
  pub slot_pos: bevy::render::render_resource::Buffer,
  /// cell → slot 间接表（每 LOD 16384 项，u32）。初始化 = 指向自身；bake 可把"本格放不出
  /// 探针"的 cell 指向邻近 cell 的探针（见 WGSL `ddgi_slot`）。
  ///
  /// 【buffer 尾部还打包了 LOD0 chunk 段的两张表】BG4 的 storage binding 已经用满 8 个
  /// （WebGPU/WGSL 下限），所以不再新增 binding，而是复用这张 u32 数组的尾部：
  ///   [0, total_slots)                          —— cell → slot 间接表（identity 起步）
  ///   [total_slots, +num_chunks)                —— chunk_base：chunk 线性下标 → LOD0 段基址
  ///   [total_slots+num_chunks, +lod0_slots)     —— slot_chunk：LOD0 局部槽位 → 所属 chunk 线性下标
  /// 三个区间的偏移在 WGSL 里由 `misc.y`(=total_slots) 与 chunk 维度算出。
  pub cell_slot: bevy::render::render_resource::Buffer,
  /// LOD0 探针池（chunk → 段基址，见 `DdgiChunkPool`）。
  pub pool: DdgiChunkPool,
  /// 上一次同步进来的 LOD0 chunk 集合（变化才重建池/反查表）。
  pub last_chunks: Vec<IVec3>,
  /// `cell_slot` 的构建键 `(total_slots, num_chunks, lod0_slots, pool.serial)`；变了才重建。
  pub cell_slot_key: Option<(u32, u32, u32, u64)>,
  /// 烘焙输出：每 slot 一条 (flags | off_b)
  pub cell: bevy::render::render_resource::Buffer,
  /// 每 slot 已烘焙的世界 cell 键 + 有效标志（滚动增量烘焙）
  pub cell_id: bevy::render::render_resource::Buffer,
  /// 每帧 age / enabled
  pub meta: bevy::render::render_resource::Buffer,
  /// 图集 ping-pong：BG4 绑「当前」（上一帧 collect 写入的，供 cast 回读 + 着色采样），
  /// BG5 绑「目标」（本帧 collect 写入）。读写必须落在不同纹理 + 不同 bind group：
  /// wgpu 禁止同一 pass 内把同一纹理既当可写存储又当采样纹理。
  pub irr: [DdgiAtlas; 2],
  pub depth: [DdgiAtlas; 2],
  /// 采样侧索引（0/1）；dispatch_ddgi 在 collect 跑完后翻转
  pub parity: usize,
  /// cast 输出的射线样本（方向 + 命中距离）/(辐亮度 + 1)
  pub samples: bevy::render::render_resource::Buffer,
  pub frame: u32,
  pub grid: DdgiWorldGrid,
  pub total_slots: u32,
  /// 已消费的世界修订号（变化 → 需要重烘焙）
  pub last_revision: u64,
  /// 有一次烘焙请求已发出但**还没真正派发**（pipeline 未编译好 / 绑定组未就绪时
  /// `dispatch_ddgi` 会提前返回）。`prepare_ddgi` 一旦推进了 `grid`/`last_revision`，
  /// 这次请求就不会再被 `grid_changed || rev_changed` 重新检出 —— 必须靠这个标志把它
  /// 留到真正派发的那一帧，否则「启动时就打开 DDGI」会一次也不烘焙，整级图集恒空。
  pub bake_pending: bool,
  pub pipelines: Option<DdgiPipelines>,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiBg4(pub bevy::render::render_resource::BindGroup);

/// collect 专用 bind group（图集写入侧 + 样本/间接参数读取）
#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiBg5(pub bevy::render::render_resource::BindGroup);

/// seal 专用 bind group（dispatch 参数 buffer 写入侧）
#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiBg6(pub bevy::render::render_resource::BindGroup);

/// 本帧是否需要跑探针烘焙（由 prepare_ddgi 写入，dispatch_ddgi 读取）
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Default)]
pub struct DdgiBakeThisFrame(pub bool);

#[derive(bevy::ecs::resource::Resource, Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct DdgiStage(pub u8);

impl DdgiStage {
  pub const OFF: u8 = 0;
  pub const ACTIVE: u8 = 1;
  pub const CAST: u8 = 2;
  pub const FULL: u8 = 3;

  pub fn new(v: u8) -> Self {
    Self(v.min(Self::FULL))
  }
  /// `GATE_DDGI_STAGE=0..3` 覆盖启动阶段（与 GATE_NO_LOD / GATE_SKIP_CHUNKWALK 同风格）。
  ///
  /// **缺省 = FULL(3)**：DDGI 已稳定，默认开启（用户明确要求）。曾经缺省 Off、只能靠 UI
  /// 滑杆打开，`GATE_BENCH=1` 的无 UI 跑法因此拿不到 Full 的帧时数据 —— 现在反过来：
  /// 需要基准对比「DDGI=Off」时显式 `GATE_DDGI_STAGE=0`。
  fn from_env() -> Self {
    Self::new(
      std::env::var("GATE_DDGI_STAGE")
        .ok()
        .and_then(|s| s.trim().parse::<u8>().ok())
        .unwrap_or(Self::FULL),
    )
  }
  pub fn run_active(&self) -> bool {
    self.0 >= Self::ACTIVE
  }
  /// 是否跑阶段二/三（cast + collect）
  pub fn run_cast(&self) -> bool {
    self.0 >= Self::CAST
  }
  pub fn shade_gi(&self) -> bool {
    self.0 >= Self::FULL
  }
}

/// 已加载世界的 AABB（voxel 坐标，闭区间）。
///
/// DDGI 的 4 级网格**全部锚定到它**（见 `DdgiWorldGrid::from_world`）—— 相机移动时
/// 任何一级的原点都**不变**，从根上消除"槽位换主"（相机滚动 → 槽位 `mod dims` 复用 →
/// 整圈探针换主 → 移动时闪）。这正是 Douglas 的 DDGI 移动时不闪的原因：他的网格不随相机变。
///
/// 由主世界算出（`vox_scene` 的 AABB）并 extract 到 render world；未设置时取单位盒
/// （退化为 1×1×1 格，不会 panic）。
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq, Eq)]
pub struct DdgiWorldAabb {
  pub min: IVec3,
  pub max: IVec3,
}

impl Default for DdgiWorldAabb {
  fn default() -> Self {
    Self {
      min: IVec3::ZERO,
      max: IVec3::ONE,
    }
  }
}

#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq)]
pub struct DdgiDebugSettings {
  pub mode: f32,
  pub gain: f32,
  pub probe_viz: bool,
  pub probe_viz_lod: f32,
  /// 借针搜索半径（格），对应 WGSL `params.w`：0 = 关闭借针（Douglas 原架构：
  /// 无针 cell 的插值角直接缺席），1 = ±1 邻域，2 = ±2 邻域（现状默认）。
  /// 仅作诊断 A/B：验证跨墙借针对室内墙角漏光的贡献。sort 每帧重写间接表，
  /// 拖动滑杆下一帧即生效，无需 rebake。
  pub borrow_radius: f32,
  /// Chebyshev 里 **std 项的信任系数**，对应 WGSL `misc.z`（0..1，默认 1 = 正常使用 std）。
  ///
  /// 拖到 0 = 完全忽略 std，`soft` 退回固定下限 `DDGI_DEPTH_SOFT_MIN`（硬判定：更能压漏光，
  /// 但过渡带变窄、动态时更易闪）。保留作 A/B 诊断用。
  ///
  /// 注：该系数最初是为确认一个已修复的缺陷而加 —— 射线方向当时是「Fibonacci 球 + 每帧
  /// 随机四元数整体重旋」，每个深度纹素跨帧收到的是全球随机方向，`std` 度量的是「20° 锥内
  /// 几何起伏」而非「同方向噪声」，墙角虚高 → 软漏光。现在射线已**绑定到深度纹素**（见
  /// dda.wgsl 的 cast「射线 ↔ 深度纹素绑定」），std 语义已正确。
  pub depth_soft_k: f32,
  /// 级联覆盖**之外**的天光兜底强度，对应 WGSL `misc.w`（0..1，默认 0.25）。
  ///
  /// 覆盖内的环境光由 DDGI 算出，覆盖外只能靠常量兜底 —— 两者强度不匹配时，级联盒边界
  /// 就是一条"亮 ↔ 暗"的硬边（相机拉远必然出现"有 GI / 无 GI 同屏"）。
  /// **不能直接用 `DDGI_SKY_AMBIENT` 调大**：它还兼作 DDGI 关闭时的环境光，调大会让
  /// 未开 DDGI 的画面整体提亮。所以覆盖外单独一个系数，运行时滑杆调到与覆盖内衔接为止。
  pub far_ambient: f32,
}

impl Default for DdgiDebugSettings {
  fn default() -> Self {
    Self {
      mode: 0.0,
      gain: 1.0,
      probe_viz: false,
      probe_viz_lod: 0.0,
      borrow_radius: 2.0,
      depth_soft_k: 1.0,
      far_ambient: 0.25,
    }
  }
}

pub struct DdgiPlugin;

impl bevy::app::Plugin for DdgiPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    use bevy::ecs::schedule::IntoScheduleConfigs;
    use bevy::prelude::RenderGraph;
    let stage = DdgiStage::from_env();
    bevy::log::info!(
      target: "gate",
      "DDGI stage = {} ({}) —— 缺省 Full；GATE_DDGI_STAGE=0..3 可覆盖",
      stage.0,
      ["Off", "Active", "Cast", "Full"][stage.0.min(3) as usize],
    );
    app.insert_resource(stage);
    app.init_resource::<DdgiDebugSettings>();
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
      return;
    };
    render_app
      .init_resource::<DdgiStage>()
      .init_resource::<DdgiWorldAabb>()
      .init_resource::<DdgiLod0Chunks>()
      .init_resource::<DdgiDebugSettings>()
      .init_resource::<DdgiBakeThisFrame>()
      .add_systems(bevy::render::RenderStartup, init_ddgi_gpu)
      .add_systems(
        bevy::render::RenderStartup,
        queue_ddgi_pipelines.after(crate::brickmap::dda::init_dda_pipelines),
      )
      .add_systems(bevy::render::ExtractSchedule, extract_ddgi_settings)
      .add_systems(
        bevy::render::Render,
        prepare_ddgi
          .in_set(bevy::render::RenderSystems::PrepareBindGroups)
          .after(crate::brickmap::upload::prepare),
      )
      .add_systems(
        RenderGraph,
        dispatch_ddgi
          .in_set(bevy::render::renderer::RenderGraphSystems::Render)
          .before(crate::brickmap::dda::dispatch_dda),
      );
  }
}

fn dummy_sized_buffer(
  device: &bevy::render::renderer::RenderDevice,
  label: &str,
  size: u64,
) -> bevy::render::render_resource::Buffer {
  use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
  device.create_buffer(&BufferDescriptor {
    label: Some(label.into()),
    size: size.max(4),
    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  })
}

fn zero_storage_buffer(
  device: &bevy::render::renderer::RenderDevice,
  queue: &bevy::render::renderer::RenderQueue,
  label: &str,
  size: u64,
) -> bevy::render::render_resource::Buffer {
  let buf = dummy_sized_buffer(device, label, size);
  queue.write_buffer(&buf, 0, &vec![0u8; size.max(4) as usize]);
  buf
}

/// 容量不足时重建（并清零）。base 网格变化 = 世界窗口变化，低频。
fn ensure_storage_buffer(
  device: &bevy::render::renderer::RenderDevice,
  queue: &bevy::render::renderer::RenderQueue,
  buf: &mut bevy::render::render_resource::Buffer,
  label: &str,
  bytes: u64,
) {
  let bytes = bytes.max(4);
  if buf.size() >= bytes {
    return;
  }
  let new = dummy_sized_buffer(device, label, bytes);
  queue.write_buffer(&new, 0, &vec![0u8; bytes as usize]);
  *buf = new;
}

/// dispatch 参数 / 计数器 buffer：既要作为 storage（atomic）被 sort/seal 读写，
/// 又要（仅 args）作为 indirect dispatch 的参数源。两者必须是不同 buffer，见 `DdgiGpu::args`。
fn ddgi_indirect_buffer(
  device: &bevy::render::renderer::RenderDevice,
  queue: &bevy::render::renderer::RenderQueue,
  label: &str,
) -> bevy::render::render_resource::Buffer {
  use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
  let buf = device.create_buffer(&BufferDescriptor {
    label: Some(label.into()),
    size: DDGI_INDIRECT_BYTES,
    usage: BufferUsages::STORAGE
      | BufferUsages::INDIRECT
      | BufferUsages::COPY_DST
      | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  });
  queue.write_buffer(&buf, 0, &vec![0u8; DDGI_INDIRECT_BYTES as usize]);
  buf
}

fn ddgi_array_view(
  tex: &bevy::render::render_resource::Texture,
) -> bevy::render::render_resource::TextureView {
  use bevy::render::render_resource::{TextureViewDescriptor, TextureViewDimension};
  tex.create_view(&TextureViewDescriptor {
    dimension: Some(TextureViewDimension::D2Array),
    ..Default::default()
  })
}

/// irradiance / depth 图集容量：每层 `40×40 = 1600` 探针 × 256 层 = **409600**。
///
/// 【为什么是 40 / 现在的余量有多紧】LOD0 改为 chunk 锚定后，总槽位 = LOD0 池高水位
/// （nuke.vox：86 chunk × 4096 = 352256）+ LOD1~3 的世界规则网格（44640 + 5760 + 800）
/// = **403456**，距 409600 只剩 6144（改造前 393776）。**余量已经很薄**：世界再大一点、
/// 或 LOD0 需要更多 chunk 段，就会越界。真要扩，先看这两条测试。
///
/// ⚠️ 池 / `from_world` **都不做钳制**：超出容量会静默越界写图集。约束由测试守着 ——
/// `world_grid_layout_and_atlas_capacity`（旧规则网格）与 `chunk_lod0_atlas_capacity`
/// （新 chunk 池 + LOD1~3）断言实际场景的 total_slots ≤ 容量，
/// `wgsl_compile.rs::ddgi_worklist_pack_covers_atlas_capacity` 断言容量本身装得进 worklist
/// 的 19 位 cell 下标。
pub const DDGI_ATLAS_LAYERS: u32 = 256;
pub const DDGI_ATLAS_PROBES_PER_LAYER_AXIS: u32 = 40;
/// 阶段二射线样本缓冲：每样本 2×vec4 = (方向.xyz, 命中距离) + (辐亮度.xyz, 1)。
///
/// 样本下标 = **全局射线编号**（`ddgi_cast` 里 `si = tid * 2`）。容量必须覆盖 `total_ray`
/// 上界 —— 而它不是 `DDGI_RAY_BUDGET`：seal 的 `rpp = max(1, BUDGET / total_active)` 有
/// **下限 1**，所以活跃探针数超过预算时 `total_ray = total_active`（每针 1 条），上界是
/// **总槽位数**（每个活跃探针占一个不同槽位）。
///
/// 【旧前提已失效】原注释断言"seal 保证全局射线总数 ≤ DDGI_RAY_BUDGET" —— 那建立在
/// "总活跃探针数 ≤ 槽数 65536 < 预算 131072"之上。dims 改成按世界 AABB 算之后槽位涨到
/// 39 万，该前提不成立；越界写被 wgpu 丢弃（不报错）→ 那些探针的样本恒 0 → **成片无 GI**。
pub const DDGI_SAMPLE_SLOTS: u32 = DDGI_RAY_BUDGET;
pub const DDGI_SAMPLE_BYTES: u64 = (DDGI_SAMPLE_SLOTS as u64) * 32;

/// 覆盖 `total_slots` 个探针所需的样本缓冲字节数（见 `DDGI_SAMPLE_SLOTS` 的说明）。
#[inline]
fn ddgi_sample_bytes(total_slots: u32) -> u64 {
  DDGI_RAY_BUDGET.max(total_slots) as u64 * 32
}

fn ddgi_array_tex(
  device: &bevy::render::renderer::RenderDevice,
  label: &str,
  format: bevy::render::render_resource::TextureFormat,
  size: (u32, u32),
) -> bevy::render::render_resource::Texture {
  use bevy::render::render_resource::{
    Extent3d, TextureDescriptor, TextureDimension, TextureUsages,
  };
  device.create_texture(&TextureDescriptor {
    label: Some(label.into()),
    size: Extent3d {
      width: size.0,
      height: size.1,
      depth_or_array_layers: DDGI_ATLAS_LAYERS,
    },
    mip_level_count: 1,
    sample_count: 1,
    dimension: TextureDimension::D2,
    format,
    usage: TextureUsages::TEXTURE_BINDING
      | TextureUsages::STORAGE_BINDING
      | TextureUsages::COPY_DST
      | TextureUsages::COPY_SRC,
    view_formats: &[],
  })
}

/// 图集清零。Rgba16Float 的 0.0 是全 0 字节，直接写零块。
/// 必须清零：probe 一 enabled 就可能被采样，但它的图集纹素要等第一次 collect 才有效，
/// 否则会读到未初始化显存（NaN/垃圾污染 GI）。
fn zero_array_tex(
  queue: &bevy::render::renderer::RenderQueue,
  label: &str,
  tex: &bevy::render::render_resource::Texture,
  size: (u32, u32),
  bytes_per_texel: u32,
) {
  use bevy::render::render_resource::{
    Extent3d, Origin3d, TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect,
  };
  let bytes = (size.0 * size.1 * DDGI_ATLAS_LAYERS * bytes_per_texel) as usize;
  bevy::log::info!(target: "gate", "DDGI 图集清零 {label}: {:.1}MB", bytes as f64 / (1 << 20) as f64);
  let data = vec![0u8; bytes];
  queue.write_texture(
    TexelCopyTextureInfo {
      texture: tex,
      mip_level: 0,
      origin: Origin3d::ZERO,
      aspect: TextureAspect::All,
    },
    &data,
    TexelCopyBufferLayout {
      offset: 0,
      bytes_per_row: Some(size.0 * bytes_per_texel),
      rows_per_image: Some(size.1),
    },
    Extent3d {
      width: size.0,
      height: size.1,
      depth_or_array_layers: DDGI_ATLAS_LAYERS,
    },
  );
}

fn init_ddgi_gpu(
  mut commands: bevy::ecs::system::Commands,
  device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<bevy::render::renderer::RenderQueue>,
) {
  use bevy::render::render_resource::TextureFormat;
  let irr_axis = DDGI_ATLAS_PROBES_PER_LAYER_AXIS * IRRADIANCE_TEXELS;
  let dep_axis = DDGI_ATLAS_PROBES_PER_LAYER_AXIS * DEPTH_TEXELS;
  let mk_atlas = |label: &str, format: TextureFormat, size: (u32, u32), bpt: u32| {
    let tex = ddgi_array_tex(&device, label, format, size);
    // 两侧都要清零：首帧 BG4 采样 a、collect 写 b；翻转后 a 才被写，不清零会读到未初始化显存。
    zero_array_tex(&queue, label, &tex, size, bpt);
    let view = ddgi_array_view(&tex);
    DdgiAtlas { tex, view }
  };
  let irr = [
    mk_atlas("ddgi_irr_a", TextureFormat::Rgba16Float, (irr_axis, irr_axis), 8),
    mk_atlas("ddgi_irr_b", TextureFormat::Rgba16Float, (irr_axis, irr_axis), 8),
  ];
  let depth = [
    mk_atlas("ddgi_depth_a", TextureFormat::Rgba16Float, (dep_axis, dep_axis), 8),
    mk_atlas("ddgi_depth_b", TextureFormat::Rgba16Float, (dep_axis, dep_axis), 8),
  ];

  let slot_pos = zero_storage_buffer(&device, &queue, "ddgi_slot_pos", 4096 * 16);
  let worklist = zero_storage_buffer(&device, &queue, "ddgi_worklist", 4096 * 16);
  let cell = zero_storage_buffer(&device, &queue, "ddgi_cell", 4096 * 4);
  let cell_id = zero_storage_buffer(&device, &queue, "ddgi_cell_id", 4096 * 16);
  let meta = zero_storage_buffer(&device, &queue, "ddgi_meta", 4096 * 4);
  // cell→slot 间接表 + LOD0 chunk 段表（尾部打包，见 `DdgiGpu::cell_slot` 注释）。
  // 内容尺寸随世界 AABB / LOD0 chunk 集变化 → 这里只放占位，`prepare_ddgi` 按构建键重建。
  let cell_slot = zero_storage_buffer(&device, &queue, "ddgi_cell_slot", 4);
  let samples = zero_storage_buffer(&device, &queue, "ddgi_samples", DDGI_SAMPLE_BYTES);
  let indirect = ddgi_indirect_buffer(&device, &queue, "ddgi_indirect");
  let args = ddgi_indirect_buffer(&device, &queue, "ddgi_args");

  commands.insert_resource(DdgiGpu {
    uniform: bevy::render::render_resource::UniformBuffer::default(),
    indirect,
    args,
    worklist,
    slot_pos,
    cell_slot,
    pool: DdgiChunkPool::default(),
    last_chunks: Vec::new(),
    cell_slot_key: None,
    cell,
    cell_id,
    meta,
    irr,
    depth,
    parity: 0,
    samples,
    frame: 0,
    grid: DdgiWorldGrid::default(),
    total_slots: 0,
    last_revision: u64::MAX,
    bake_pending: false,
    pipelines: None,
  });
}

fn queue_ddgi_pipelines(
  dda: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaPipelines>>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  asset_server: bevy::ecs::system::Res<bevy::asset::AssetServer>,
  mut gpu: bevy::ecs::system::ResMut<DdgiGpu>,
) {
  use bevy::render::render_resource::{BindGroupLayoutDescriptor, ComputePipelineDescriptor, PipelineCache};
  use std::borrow::Cow;
  if gpu.pipelines.is_some() {
    return;
  }
  let Some(dda) = dda else {
    return;
  };
  let base = vec![
    dda.bg0_layout.clone(),
    dda.bg1_layout.clone(),
    dda.bg2_layout.clone(),
    dda.bg3_layout.clone(),
    ddgi_bg4_layout(),
  ];
  // collect 额外挂 BG5（图集写入侧）；seal 额外挂 BG6（dispatch 参数写入侧）。
  // 注意：layout 的**位置**就是 bind group 索引。base 已占 0..4，collect 的 BG5 正好落在 5；
  // seal 的参数侧在 @group(6)，而 @group(5) 已被 collect 占用（同一 WGSL 模块里同一个
  // group/binding 槽只能有一个变量），所以 seal 的 layout 在 5 号位留一个空的 layout。
  let mut collect_layout = base.clone();
  collect_layout.push(ddgi_bg5_layout());
  let mut seal_layout = base.clone();
  seal_layout.push(BindGroupLayoutDescriptor::new("DdgiBgEmpty5", &[]));
  seal_layout.push(ddgi_bg6_layout());
  let shader = asset_server.load(crate::brickmap::dda::DDA_SHADER_ASSET_PATH);
  let mk = |pipeline_cache: &PipelineCache,
            label: &'static str,
            entry: &'static str,
            layout: Vec<BindGroupLayoutDescriptor>| {
    pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
      label: Some(Cow::from(label)),
      layout,
      shader: shader.clone(),
      entry_point: Some(Cow::from(entry)),
      ..Default::default()
    })
  };
  gpu.pipelines = Some(DdgiPipelines {
    bake: [
      mk(&pipeline_cache, "gate_ddgi_bake0", "ddgi_bake0", base.clone()),
      mk(&pipeline_cache, "gate_ddgi_bake1", "ddgi_bake1", base.clone()),
      mk(&pipeline_cache, "gate_ddgi_bake2", "ddgi_bake2", base.clone()),
      mk(&pipeline_cache, "gate_ddgi_bake3", "ddgi_bake3", base.clone()),
    ],
    sort: mk(&pipeline_cache, "gate_ddgi_sort", "ddgi_sort", base.clone()),
    seal: mk(&pipeline_cache, "gate_ddgi_seal", "ddgi_seal", seal_layout),
    cast: mk(&pipeline_cache, "gate_ddgi_cast", "ddgi_cast", base.clone()),
    collect: mk(
      &pipeline_cache,
      "gate_ddgi_collect",
      "ddgi_collect",
      collect_layout,
    ),
  });
}

#[allow(clippy::too_many_arguments)]
fn dispatch_ddgi(
  mut ctx: bevy::render::renderer::RenderContext,
  bg0: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg0BindGroup>>,
  bg1: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg1BindGroup>>,
  bg2: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg2BindGroup>>,
  bg3: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaBg3BindGroup>>,
  bg4: Option<bevy::ecs::system::Res<DdgiBg4>>,
  bg5: Option<bevy::ecs::system::Res<DdgiBg5>>,
  bg6: Option<bevy::ecs::system::Res<DdgiBg6>>,
  bake: Option<bevy::ecs::system::Res<DdgiBakeThisFrame>>,
  stage: bevy::ecs::system::Res<DdgiStage>,
  dbg: Option<bevy::ecs::system::Res<DdgiDebugSettings>>,
  mut gpu: bevy::ecs::system::ResMut<DdgiGpu>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  mut profiler: bevy::ecs::system::ResMut<crate::profiler::GpuProfilerRes>,
) {
  if !stage.run_active() && !dbg.map_or(false, |d| d.probe_viz) {
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
  let Some(pipes) = gpu.pipelines else {
    return;
  };
  let Some(p_sort) = pipeline_cache.get_compute_pipeline(pipes.sort) else {
    return;
  };
  let Some(p_seal) = pipeline_cache.get_compute_pipeline(pipes.seal) else {
    return;
  };
  if gpu.total_slots == 0 {
    return;
  }

  let set_bgs = |pass: &mut bevy::render::render_resource::ComputePass,
                 bg4: &bevy::render::render_resource::BindGroup| {
    pass.set_bind_group(0, &bg0.0, &[]);
    pass.set_bind_group(1, &bg1.0, &[]);
    pass.set_bind_group(2, &bg2.0, &[]);
    pass.set_bind_group(3, &bg3.0, &[]);
    pass.set_bind_group(4, bg4, &[]);
  };
  // 1D dispatch：WG=64，各 pass 覆盖 total_slots（bake 仅在修订号变化时跑）
  let wg_slots = gpu.total_slots.div_ceil(64).min(65535);

  // 烘焙：世界数据变化时重算探针位置（BFS 最大空叶）。
  // **按 LOD 逐级派发 4 个独立 pass（细→粗）**：粗级的放置要继承本帧细级的 bake 输出，
  // 同一 pass 内没有顺序保证，只有 pass 边界才是内存屏障（见 WGSL `ddgi_bake_one`）。
  if bake.map_or(false, |b| b.0) {
    let n_lods = DDGI_LODS as usize;
    // 每级的 dispatch 大小按**该级实际槽数**算 —— LOD1~3 的 dims 随世界 AABB 变化（不再固定
    // 32×16×32），LOD0 的槽数则是探针池的高水位（chunk 段之和，不等于空间盒 dims 乘积）。
    // 写死/用 dims 乘积都会让烘焙只覆盖一部分 cell，症状是"大部分区域无 GI"（且无报错）。
    let lod0_slots = gpu.grid.lod_slot_base[1];
    let wg_per_lod: [u32; DDGI_LODS as usize] = std::array::from_fn(|lod| {
      let n = if lod == 0 {
        lod0_slots
      } else {
        let d = gpu.grid.lod_dims[lod];
        d.x * d.y * d.z
      };
      n.div_ceil(64).max(1).min(65535)
    });
    const LABELS: [&str; DDGI_LODS as usize] = [
      "gate_ddgi_bake0",
      "gate_ddgi_bake1",
      "gate_ddgi_bake2",
      "gate_ddgi_bake3",
    ];
    // 4 个入口必须**全部**就绪才开跑：否则会出现"细级烘了、粗级没烘"的半帧状态。
    let mut p_bake: [Option<&bevy::render::render_resource::ComputePipeline>; DDGI_LODS as usize] =
      [None; DDGI_LODS as usize];
    let mut ready = true;
    for lod in 0..n_lods {
      p_bake[lod] = pipeline_cache.get_compute_pipeline(pipes.bake[lod]);
      if p_bake[lod].is_none() {
        ready = false;
      }
    }
    if ready {
      for lod in 0..n_lods {
        let p = p_bake[lod].unwrap();
        crate::profiler::gpu_compute_pass(
          &mut profiler,
          ctx.command_encoder(),
          LABELS[lod],
          |pass| {
            pass.set_pipeline(p);
            set_bgs(pass, &bg4.0);
            pass.dispatch_workgroups(wg_per_lod[lod], 1, 1);
          },
        );
      }
      // 真正派发过了才撤销挂起标志（否则保留它，等下一帧管线就绪再烘）
      gpu.bake_pending = false;
    }
  }
  // sort：读烘焙结果做活跃判定 + worklist 压缩 + age/slot_pos/meta 刷新（probe_viz 依赖其新鲜度）
  crate::profiler::gpu_compute_pass(
    &mut profiler,
    ctx.command_encoder(),
    "gate_ddgi_sort",
    |pass| {
      pass.set_pipeline(p_sort);
      set_bgs(pass, &bg4.0);
      pass.dispatch_workgroups(wg_slots, 1, 1);
    },
  );
  // seal：为 cast/collect 准备 per-LOD indirect args；只在需要阶段二时跑。
  if stage.run_active() {
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_ddgi_seal",
      |pass| {
        pass.set_pipeline(p_seal);
        set_bgs(pass, &bg4.0);
        if let Some(bg6) = bg6.as_ref() {
          pass.set_bind_group(6, &bg6.0, &[]);
        }
        pass.dispatch_workgroups(1, 1, 1);
      },
    );
  }
  // 阶段二 cast / 阶段三 collect：seal 已备好 per-LOD indirect args
  // （byte 0 = cast×4 LOD，byte 64 = collect×4 LOD）与 rpp（byte 128）。
  if stage.run_cast() {
    if let (Some(p_cast), Some(p_coll), Some(bg5)) = (
      pipeline_cache.get_compute_pipeline(pipes.cast),
      pipeline_cache.get_compute_pipeline(pipes.collect),
      bg5.as_ref(),
    ) {
      // cast / collect 各一次间接 dispatch（覆盖全部 LOD；LOD 由 shader 用 seal 写的前缀和还原）。
      // 注意两个 args 必须放在不同 buffer 段：cast 在 ddgi_args 词 0（字节 0），
      // collect 在词 16（字节 64）。
      crate::profiler::gpu_compute_pass(
        &mut profiler,
        ctx.command_encoder(),
        "gate_ddgi_cast",
        |pass| {
          pass.set_pipeline(p_cast);
          set_bgs(pass, &bg4.0);
          pass.dispatch_workgroups_indirect(&gpu.args, 0);
        },
      );
      crate::profiler::gpu_compute_pass(
        &mut profiler,
        ctx.command_encoder(),
        "gate_ddgi_collect",
        |pass| {
          pass.set_pipeline(p_coll);
          set_bgs(pass, &bg4.0);
          pass.set_bind_group(5, &bg5.0, &[]);
          pass.dispatch_workgroups_indirect(&gpu.args, 64);
        },
      );
      // 本帧写过图集 → 下一帧采样侧翻到刚写完的那一半
      gpu.parity ^= 1;
    }
  }
}

fn extract_ddgi_settings(
  mut commands: bevy::ecs::system::Commands,
  stage: Option<bevy::render::Extract<bevy::ecs::system::Res<DdgiStage>>>,
  debug: Option<bevy::render::Extract<bevy::ecs::system::Res<DdgiDebugSettings>>>,
  world_aabb: Option<bevy::render::Extract<bevy::ecs::system::Res<DdgiWorldAabb>>>,
  lod0_chunks: Option<bevy::render::Extract<bevy::ecs::system::Res<DdgiLod0Chunks>>>,
) {
  let s = stage.map_or(DdgiStage::OFF, |s| s.0.min(DdgiStage::FULL));
  commands.insert_resource(DdgiStage(s));
  let dbg = debug.map_or_else(DdgiDebugSettings::default, |d| DdgiDebugSettings {
    mode: d.mode,
    gain: d.gain,
    probe_viz: d.probe_viz,
    probe_viz_lod: d.probe_viz_lod,
    borrow_radius: d.borrow_radius,
    depth_soft_k: d.depth_soft_k,
    far_ambient: d.far_ambient,
  });
  commands.insert_resource(dbg);
  commands.insert_resource(world_aabb.map_or_else(DdgiWorldAabb::default, |a| DdgiWorldAabb {
    min: a.min,
    max: a.max,
  }));
  commands.insert_resource(lod0_chunks.map_or_else(DdgiLod0Chunks::default, |c| c.clone()));
}

#[allow(clippy::too_many_arguments)]
fn prepare_ddgi(
  mut commands: bevy::ecs::system::Commands,
  device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<bevy::render::renderer::RenderQueue>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  revision: Option<bevy::ecs::system::Res<crate::brickmap::upload::BrickMapRevision>>,
  dirty: Option<bevy::ecs::system::Res<crate::brickmap::upload::BrickMapDirty>>,
  stage: bevy::ecs::system::Res<DdgiStage>,
  dbg: bevy::ecs::system::Res<DdgiDebugSettings>,
  world: bevy::ecs::system::Res<DdgiWorldAabb>,
  lod0_chunks: Option<bevy::ecs::system::Res<DdgiLod0Chunks>>,
  mut gpu: bevy::ecs::system::ResMut<DdgiGpu>,
) {
  // ---- 网格推导 ----
  // LOD1~3 仍**全部锚定世界 AABB**（相机移动不改变原点 → 不换主 → 不闪）。
  // LOD0 换成 **chunk 锚定**：由 `DdgiChunkPool` 从探针池给「需要探针的 chunk」分配固定
  // 4096 槽的段；只有分配到的段才占 LOD0 槽位（内容驱动）。两种网格的边界见 `DdgiChunkGeom`
  // 与 `DdgiChunkPool` 的注释；槽位段顺序恒为 LOD0（池）→ LOD1 → LOD2 → LOD3。
  let base_grid = DdgiWorldGrid::from_world(world.min, world.max);
  let chunk_geom = DdgiChunkGeom::from_world(world.min, world.max);
  let lod0_chunks = lod0_chunks.map_or_else(Vec::new, |c| c.chunks.clone());

  // 探针池同步：只在「chunk 集变化」或「chunk 网格几何变化」时真正做事。已有 chunk 的段基址
  // **保持不变**（见 `DdgiChunkPool::sync`）—— 基址一变即等价于该 chunk 换槽位。
  let chunks_changed = gpu.last_chunks != lod0_chunks;
  let geom_changed = gpu.pool.geom != chunk_geom;
  let pool_changed = if chunks_changed || geom_changed {
    let changed = gpu.pool.sync(chunk_geom, &lod0_chunks);
    gpu.last_chunks = lod0_chunks;
    changed
  } else {
    false
  };
  let lod0_slots = gpu.pool.lod0_slots();

  let mut grid = base_grid;
  grid.lod_slot_base[0] = 0;
  let mut base = lod0_slots;
  for lod in 1..DDGI_LODS as usize {
    grid.lod_slot_base[lod] = base;
    base += grid.lod_count(lod);
  }
  grid.total_slots = base;
  let total = grid.total_slots;
  let grid_changed = grid != gpu.grid || pool_changed;
  let num_chunks = chunk_geom.len();

  // ---- 推进网格/修订号状态（仅在本帧会跑 pass 时）----
  // 否则「关闭期间世界已更新/相机已移动」会被吞掉 → 之后打开时不会补烘。
  let rev = revision.map_or(0, |r| r.0);
  let will_run = stage.run_active() || dbg.probe_viz;
  let rev_changed = rev != gpu.last_revision;
  if will_run {
    gpu.grid = grid;
    gpu.total_slots = total;
    if grid_changed {
      // 网格变了就打一次实际数值：LOD0 的 dims 字段仍是**空间盒**（级联包含 / 混合带用它），
      // 槽位数不再等于 dims 乘积，而是池高水位 —— 出问题时第一件事就是核对它们。
      for lod in 0..DDGI_LODS as usize {
        let o = grid.lod_origins[lod];
        let d = grid.lod_dims[lod];
        let count = if lod == 0 {
          lod0_slots
        } else {
          grid.lod_count(lod)
        };
        bevy::log::info!(
          "DDGI LOD{lod}: cell={} origin=({},{},{}) dims=({},{},{}) slot_base={} count={}",
          DDGI_LOD_CELL_SIZES[lod],
          o.x,
          o.y,
          o.z,
          d.x,
          d.y,
          d.z,
          grid.lod_slot_base[lod],
          count,
        );
      }
      bevy::log::info!(
        "DDGI LOD0 chunk 池: chunks={}/{} lod0_slots={}（旧 AABB 规则网格 LOD0={}，差 {}）",
        gpu.pool.bases.iter().filter(|&&b| b != DDGI_CHUNK_NO_BASE).count(),
        num_chunks,
        lod0_slots,
        grid.lod_count(0),
        grid.lod_count(0) as i64 - lod0_slots as i64,
      );
      bevy::log::info!("DDGI 总槽位 = {total}（图集容量 409600）");
    }
    gpu.last_revision = rev;
  }
  // 相机滚动（grid 变化）/ 池变化 → 按「世界 cell 键」增量补烘；世界编辑 → 只失效脏区内的 cell
  // （bake 用 uniform 里的脏区 AABB 跳过 cell_id 键检查，不再整块清缓存）。
  // `gpu.bake_pending` 让请求**黏住**：本帧若因 pipeline 未编译好而没真正派发（见
  // dispatch_ddgi 的提前返回），下一帧仍会重试，而不是被「已推进的 grid/last_revision」
  // 悄悄吞掉（症状：启动时就打开 DDGI → 一次也不烘焙 → cast/collect 恒为 0.02ms）。
  let need_bake = (grid_changed || rev_changed || gpu.bake_pending) && total > 0 && will_run;
  gpu.bake_pending = need_bake;

  // ---- 脏区：本帧上传实际改动的世界 voxel AABB（max 不含）----
  // 全量上传 → ±1e9（等价于全部 cell 失效）；增量 → 改动 chunk 的合并包围盒。
  let (dirty_min, dirty_max, dirty_valid) = match dirty.as_ref() {
    Some(d) if d.full => (
      IVec3::splat(-1_000_000_000),
      IVec3::splat(1_000_000_000),
      1.0f32,
    ),
    Some(d) if d.min_voxel != d.max_voxel => (d.min_voxel, d.max_voxel, 1.0f32),
    _ => (IVec3::ZERO, IVec3::ZERO, 0.0f32),
  };

  // ---- 缓冲容量（槽数固定：4 LOD × 16³）----
  let s = total as u64;
  ensure_storage_buffer(&device, &queue, &mut gpu.cell, "ddgi_cell", s * 4);
  ensure_storage_buffer(&device, &queue, &mut gpu.meta, "ddgi_meta", s * 4);
  ensure_storage_buffer(&device, &queue, &mut gpu.slot_pos, "ddgi_slot_pos", s * 16);
  ensure_storage_buffer(&device, &queue, &mut gpu.worklist, "ddgi_worklist", s * 16);
  ensure_storage_buffer(&device, &queue, &mut gpu.cell_id, "ddgi_cell_id", s * 16);
  // `cell_slot`：尾部同时打包 LOD0 的两张 chunk 表（见 `DdgiGpu::cell_slot` 注释）。
  //   [0, total_slots)                        cell→slot 间接表，identity 起步
  //   [total_slots, +num_chunks)              chunk_base（未分配 = DDGI_CHUNK_NO_BASE 哨兵）
  //   [total_slots+num_chunks, +lod0_slots)   slot_chunk（LOD0 局部槽 → chunk 线性下标）
  // 构建键（total / chunk 数 / LOD0 槽数 / 池 serial）变了才重建 —— 内容尺寸变化、或池发生了
  // 「领段/归还」都要重铺；新增的间接表项必须填 identity，否则新槽位读到 0 会指向 slot 0
  // （症状：大面积无 GI 且无任何报错）。
  let key = (total, num_chunks, lod0_slots, gpu.pool.serial);
  if gpu.cell_slot_key != Some(key) {
    let words = (total + num_chunks + lod0_slots) as usize;
    let mut buf: Vec<u8> = Vec::with_capacity(words * 4);
    for i in 0..total {
      buf.extend_from_slice(&i.to_le_bytes());
    }
    for l in 0..num_chunks {
      let b = gpu.pool.bases[l as usize];
      buf.extend_from_slice(&b.to_le_bytes());
    }
    let mut slot_chunk = vec![0u32; lod0_slots as usize];
    for l in 0..num_chunks {
      let b = gpu.pool.bases[l as usize];
      if b != DDGI_CHUNK_NO_BASE {
        // 反向表：该段内 4096 个局部槽位都属于 chunk `l`
        for k in 0..DDGI_CHUNK_LOD0_SLOTS {
          slot_chunk[(b + k) as usize] = l;
        }
      }
    }
    for v in slot_chunk.iter() {
      buf.extend_from_slice(&v.to_le_bytes());
    }
    gpu.cell_slot = zero_storage_buffer(&device, &queue, "ddgi_cell_slot", (words as u64) * 4);
    queue.write_buffer(&gpu.cell_slot, 0, &buf);
    gpu.cell_slot_key = Some(key);
  }
  ensure_storage_buffer(
    &device,
    &queue,
    &mut gpu.samples,
    "ddgi_samples",
    ddgi_sample_bytes(total),
  );

  gpu.frame = gpu.frame.wrapping_add(1);

  // ---- uniform ----
  let mut u = DdgiUniform::default();
  for lod in 0..DDGI_LODS as usize {
    let cell = DDGI_LOD_CELL_SIZES[lod];
    let d = gpu.grid.lod_dims[lod];
    let o = gpu.grid.lod_origins[lod];
    u.lods[lod] = DdgiLod {
      origin: IVec4::new(o.x, o.y, o.z, cell),
      dims: UVec4::new(d.x, d.y, d.z, gpu.grid.lod_slot_base[lod]),
    };
  }
  // params: x=frame, y=debug mode, z=gain, w=借针半径（0=关闭，见 DdgiDebugSettings）
  u.params = Vec4::new(gpu.frame as f32, dbg.mode, dbg.gain, dbg.borrow_radius);
  u.misc = Vec4::new(
    if stage.shade_gi() { 1.0 } else { 0.0 },
    gpu.total_slots as f32,
    // z = Chebyshev std 信任系数（见 DdgiDebugSettings.depth_soft_k）
    dbg.depth_soft_k,
    // w = 级联覆盖外的天光兜底强度（见 DdgiDebugSettings.far_ambient）
    dbg.far_ambient,
  );
  u.dirty_min = Vec4::new(
    dirty_min.x as f32,
    dirty_min.y as f32,
    dirty_min.z as f32,
    dirty_valid,
  );
  u.dirty_max = Vec4::new(dirty_max.x as f32, dirty_max.y as f32, dirty_max.z as f32, 0.0);
  // LOD0 的 chunk 段编址（只有 lod==0 读它）：原点/维度（chunk 单位）、每 chunk cell 数、
  // 已分配槽数。WGSL 用 `misc.y`(=total_slots) + 这里的维度定位 cell_slot 尾部的两张表。
  u.chunk = DdgiChunkUniform {
    origin: IVec4::new(
      chunk_geom.origin.x,
      chunk_geom.origin.y,
      chunk_geom.origin.z,
      DDGI_CHUNK_LOD0_AXIS,
    ),
    dims: UVec4::new(
      chunk_geom.dims.x,
      chunk_geom.dims.y,
      chunk_geom.dims.z,
      lod0_slots,
    ),
  };
  *gpu.uniform.get_mut() = u;
  gpu.uniform.write_buffer(&device, &queue);

  // 每帧清零活跃计数器（indirect words [36..40)）；indirect args 由 seal 当帧覆写。
  queue.write_buffer(
    &gpu.indirect,
    DDGI_COUNTER_CLEAR_OFFSET,
    &[0u8; DDGI_COUNTER_CLEAR_BYTES as usize],
  );

  // ---- BG4（图集采样侧 + 各 pass 共用缓冲）----
  // p = 采样侧：上一帧 collect 写入的那一半（cast 回读 + 着色采样都读它）。
  // 写入侧 1-p 只给 collect（BG5）。
  use bevy::render::render_resource::BindGroupEntries;
  let p = gpu.parity & 1;
  let bg4_layout = pipeline_cache.get_bind_group_layout(&ddgi_bg4_layout());
  let bg4 = device.create_bind_group(
    None,
    &bg4_layout,
    &BindGroupEntries::sequential((
      &gpu.uniform,
      &gpu.irr[p].view,
      &gpu.depth[p].view,
      gpu.cell.as_entire_binding(),
      gpu.meta.as_entire_binding(),
      gpu.indirect.as_entire_binding(),
      gpu.worklist.as_entire_binding(),
      gpu.slot_pos.as_entire_binding(),
      gpu.cell_id.as_entire_binding(),
      gpu.samples.as_entire_binding(),
      gpu.cell_slot.as_entire_binding(),
    )),
  );
  commands.insert_resource(DdgiBg4(bg4));

  // ---- BG5（collect 图集写入侧）----
  let bg5_layout = pipeline_cache.get_bind_group_layout(&ddgi_bg5_layout());
  let bg5 = device.create_bind_group(
    None,
    &bg5_layout,
    &BindGroupEntries::sequential((
      &gpu.irr[1 - p].view,
      &gpu.depth[1 - p].view,
    )),
  );
  commands.insert_resource(DdgiBg5(bg5));

  // ---- BG6（seal 的 dispatch 参数写入侧）----
  let bg6_layout = pipeline_cache.get_bind_group_layout(&ddgi_bg6_layout());
  let bg6 = device.create_bind_group(
    None,
    &bg6_layout,
    &BindGroupEntries::single(gpu.args.as_entire_binding()),
  );
  commands.insert_resource(DdgiBg6(bg6));
  commands.insert_resource(DdgiBakeThisFrame(need_bake));
}

#[cfg(test)]
mod tests {
  use super::*;

  /// 判定 LOD(l-1) 盒是否严格包含于 LOD(l) 盒（嵌套级联的核心不变式）。
  fn lod_aabb(g: &DdgiWorldGrid, lod: usize) -> (IVec3, IVec3) {
    let o = g.lod_origins[lod];
    let ext = IVec3::new(
      g.lod_dims[lod].x as i32,
      g.lod_dims[lod].y as i32,
      g.lod_dims[lod].z as i32,
    ) * DDGI_LOD_CELL_SIZES[lod];
    (o, o + ext)
  }

  #[test]
  fn world_grid_covers_aabb_and_is_nested() {
    // AABB 起点故意不对齐到 cell，验证"向下对齐到 AABB 外侧"
    let lo = IVec3::new(37, -11, 5);
    let hi = lo + IVec3::new(1932, 615, 1136);
    let g = DdgiWorldGrid::from_world(lo, hi);
    for lod in 0..DDGI_LODS as usize {
      let cell = DDGI_LOD_CELL_SIZES[lod];
      let o = g.lod_origins[lod];
      // 原点落在 AABB 之外（向下对齐），且是 cell 的整数倍（shader 的整除前提）
      assert!(o.cmple(lo).all(), "lod {lod} 原点未向下对齐到 AABB 外侧");
      assert_eq!(o.x.rem_euclid(cell), 0);
      assert_eq!(o.y.rem_euclid(cell), 0);
      assert_eq!(o.z.rem_euclid(cell), 0);
      // 该级覆盖必须罩住整个 AABB
      let (l, h) = lod_aabb(&g, lod);
      assert!(l.cmple(lo).all() && hi.cmplt(h).all(), "lod {lod} 未罩住世界 AABB");
    }
    // 相邻级严格嵌套（cell ×2 → dims 减半）
    for lod in 1..DDGI_LODS as usize {
      let (plo, phi) = lod_aabb(&g, lod - 1);
      let (l, h) = lod_aabb(&g, lod);
      assert!(
        plo.cmpge(l).all() && phi.cmple(h).all(),
        "lod {lod} 未包含 lod {}",
        lod - 1
      );
    }
  }

  #[test]
  fn world_grid_layout_and_atlas_capacity() {
    // nuke.vox 的 AABB 量级（来自启动日志 aabb=[[-454,16,-56]]-[[1478,631,1080]]）
    let lo = IVec3::new(-454, 16, -56);
    let hi = IVec3::new(1478, 631, 1080);
    let g = DdgiWorldGrid::from_world(lo, hi);
    // 槽位块连续排布
    let mut acc = 0u32;
    for lod in 0..DDGI_LODS as usize {
      assert_eq!(g.lod_slot_base[lod], acc);
      acc += g.lod_count(lod);
    }
    assert_eq!(g.total_slots, acc);
    assert!(!g.is_empty());
    // 【关键】nuke.vox 量级下必须装得进图集 —— 否则运行期会越界写图集
    let capacity = DDGI_ATLAS_LAYERS
      * DDGI_ATLAS_PROBES_PER_LAYER_AXIS
      * DDGI_ATLAS_PROBES_PER_LAYER_AXIS;
    assert!(
      g.total_slots <= capacity,
      "槽位 {} 超出图集容量 {}（需调大 DDGI_ATLAS_PROBES_PER_LAYER_AXIS）",
      g.total_slots,
      capacity
    );
  }

  /// 世界锚定的槽位映射：**同一世界 cell 的槽位与相机无关**。
  ///
  /// 这是"移动时不闪"的根据 —— `from_world` 的签名里**根本没有相机参数**，网格是只依赖
  /// 世界 AABB 的纯函数。将来若有人把相机重新引入网格推导，这条会立刻失败。
  #[test]
  fn world_cell_slot_is_camera_independent() {
    let slot_of = |g: &DdgiWorldGrid, lod: usize, wc: IVec3| -> u32 {
      let dims = g.lod_dims[lod].as_ivec3();
      let r = wc.rem_euclid(dims);
      g.lod_slot_base[lod] + (r.x + r.y * dims.x + r.z * dims.x * dims.y) as u32
    };
    let lo = IVec3::new(-454, 16, -56);
    let hi = IVec3::new(1478, 631, 1080);
    // 同样的 AABB 必须给出完全相同的网格（纯函数 → 槽位映射恒定）
    let g1 = DdgiWorldGrid::from_world(lo, hi);
    let g2 = DdgiWorldGrid::from_world(lo, hi);
    assert_eq!(g1, g2, "from_world 不是纯函数");
    let wc = IVec3::new(512, 128, -128);
    for lod in 0..DDGI_LODS as usize {
      assert_eq!(slot_of(&g1, lod, wc), slot_of(&g2, lod, wc));
    }
  }

  /// chunk 网格必须能编址「内容 AABB 之内所有体素所属的 chunk」，且线性下标唯一。
  #[test]
  fn chunk_geom_covers_aabb_and_roundtrips() {
    let lo = IVec3::new(-454, 16, -56);
    let hi = IVec3::new(1478, 631, 1080);
    let g = DdgiChunkGeom::from_world(lo, hi);
    let cmin = lo.div_euclid(IVec3::splat(DDGI_CHUNK_VOXELS));
    let cmax = (hi - IVec3::ONE).div_euclid(IVec3::splat(DDGI_CHUNK_VOXELS));
    for z in cmin.z..=cmax.z {
      for y in cmin.y..=cmax.y {
        for x in cmin.x..=cmax.x {
          let cc = IVec3::new(x, y, z);
          let l = g.linear(cc).expect("内容 chunk 必须在 chunk 网格内");
          assert_eq!(g.coord(l), cc, "linear/coord 不是互逆");
        }
      }
    }
    // coord/linear 全域互逆
    for i in 0..g.len() {
      assert_eq!(g.linear(g.coord(i)), Some(i));
    }
  }

  /// 探针池：每 chunk 领一段固定 4096 槽、段两两不重叠；**已有基址不因再次同步而改变**。
  #[test]
  fn chunk_pool_segments_are_disjoint_and_stable() {
    let geom = DdgiChunkGeom::from_world(IVec3::new(-454, 16, -56), IVec3::new(1478, 631, 1080));
    let mut pool = DdgiChunkPool::default();
    // 86 个 chunk（nuke.vox 实测：内容 82 ∪ 边界邻域 4）
    let wanted: Vec<IVec3> = (0..86).map(|i| geom.coord(i * 5 + 3)).collect();
    assert!(pool.sync(geom, &wanted));
    assert_eq!(pool.lod0_slots(), 86 * DDGI_CHUNK_LOD0_SLOTS);
    // 段基址恰为 0,4096,8192,...（互不重叠）
    let mut bases: Vec<u32> = wanted
      .iter()
      .map(|c| pool.bases[geom.linear(*c).unwrap() as usize])
      .collect();
    bases.sort_unstable();
    for (i, b) in bases.iter().enumerate() {
      assert_eq!(*b, i as u32 * DDGI_CHUNK_LOD0_SLOTS);
    }
    // 幂等：同样集合再同步 → 无变化、基址逐位不变（这是「移动/编辑不闪」的前提）
    let before = pool.bases.clone();
    let serial = pool.serial;
    assert!(!pool.sync(geom, &wanted));
    assert_eq!(pool.bases, before);
    assert_eq!(pool.serial, serial);
  }

  /// 归还的段进空闲链表，新 chunk 复用（LIFO）；高水位不缩（池只增不减地定容）。
  #[test]
  fn chunk_pool_free_list_reuses_released_base() {
    let geom = DdgiChunkGeom::from_world(IVec3::ZERO, IVec3::splat(4096));
    let mut pool = DdgiChunkPool::default();
    let a = geom.coord(7);
    let b = geom.coord(9);
    assert!(pool.sync(geom, &[a, b]));
    let base_a = pool.bases[geom.linear(a).unwrap() as usize];
    let base_b = pool.bases[geom.linear(b).unwrap() as usize];
    assert_eq!(base_a, 0);
    assert_eq!(pool.lod0_slots(), 2 * DDGI_CHUNK_LOD0_SLOTS);
    // 释放 b：a 的基址不变，b 的段归还
    assert!(pool.sync(geom, &[a]));
    assert_eq!(pool.bases[geom.linear(a).unwrap() as usize], base_a);
    assert_eq!(pool.free, vec![base_b]);
    // 新 chunk c 复用 b 的段（LIFO），高水位保持
    let c = geom.coord(11);
    assert!(pool.sync(geom, &[a, c]));
    assert_eq!(pool.bases[geom.linear(c).unwrap() as usize], base_b);
    assert_eq!(pool.lod0_slots(), 2 * DDGI_CHUNK_LOD0_SLOTS);
  }

  /// 新布局（LOD0 chunk 池 + LOD1~3 世界网格）必须仍装得进图集，且 LOD0 局部下标装得进
  /// worklist 的 19 位。
  #[test]
  fn chunk_lod0_atlas_capacity() {
    let lod0_slots = 86 * DDGI_CHUNK_LOD0_SLOTS;
    let g = DdgiWorldGrid::from_world(IVec3::new(-454, 16, -56), IVec3::new(1478, 631, 1080));
    let total = lod0_slots + g.lod_count(1) + g.lod_count(2) + g.lod_count(3);
    let capacity =
      DDGI_ATLAS_LAYERS * DDGI_ATLAS_PROBES_PER_LAYER_AXIS * DDGI_ATLAS_PROBES_PER_LAYER_AXIS;
    assert!(
      total <= capacity,
      "LOD0 chunk 池 + LOD1~3 总槽位 {total} 超出图集容量 {capacity}",
    );
    assert!(lod0_slots <= 0x7FFFF, "LOD0 局部下标必须装进 worklist 的 19 位");
  }
}
