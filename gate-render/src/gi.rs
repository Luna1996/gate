//! GI：**屏幕空间逐面 ReSTIR**（唯一的 GI 路径）。
//! 算法与 reservoir 布局见 `assets/shaders/voxel_raytrace/gi/screen.wesl`；Rust 侧只负责开
//! reservoir 双缓冲（每 GI 像素 `GI_RES_WORDS` 个 u32，随 GI 分辨率重建）、上传 uniform、
//! 建 bind group、每帧 ping-pong 换绑。
//! 着色 pass（`gi_main`）在 `dispatch_dda` 里排在主 pass 之前，一次派发完成
//! 「新鲜候选 → 时域复用 → 空间复用 → 着色」；没有独立的 GI 更新 / 复用 pass。
//!
//! 历史（为什么没有世界空间缓存了）：曾经有一条「逐 (体素,面) 辐照度缓存」的旁路
//! （哈希网格 + 每帧轮转更新 + `1/n` 运行平均收敛，与屏幕空间路径二选一）。它与射线求值
//! 构成**跨帧反馈环**（缓存值进射线、射线值又进缓存），表现是「像拿刷子一层层刷、稳态要等
//! 二十秒、有无 GI 的异常区」—— 三条都是「逐帧确定」的反面。已整条删除；GI 的输入现在只依赖
//! 几何与光照，因此**不存在「收敛」这件事**：第 1 帧就是稳态，差别只有噪声（由降噪链压）。

use bevy::render::render_resource::{
  BindGroupLayoutDescriptor, CachedComputePipelineId, ShaderType,
};
use glam::{Mat4, UVec2, UVec4, Vec4};

use crate::wesl_consts::gi_consts;

/// GI uniform（WESL `bindings.wesl` 的 `GiUniform` 逐字段镜像，字节一致）。
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GiUniform {
  /// x = **二次顶点太阳反弹**（1 = 开、0 = 关；菜单「渲染/RESTIR GI/太阳反弹」）、
  /// y = 保留（恒 0）、z = GI 增益、w = 保留（恒 0）
  pub params: Vec4,
  /// x = GI 开关（0/1）、y = **每帧采样预算分摊 N**（见 `GiSettings::share`）、
  /// z = 误差驱动重分配档位、w = **第二条弹射的倍数**（`GiSettings::bounce2_mult`：0 = 关 /
  /// 4 = 稀疏 / 1 = 全；菜单「渲染/光照/二次弹射」）。
  pub misc: Vec4,
  /// x = 保留（恒 0）、y = GI 分辨率除数（1 = 全分辨率、2 = 半分辨率、4 = 四分之一；**整数值的 f32**，
  /// 只被 `gi_main` 用来把本 pass 的像素下标换成 beam 纹理下标）、
  /// z = **本帧允许复用历史**（时域 + 空间两条；1 = 允许）：遮挡关系没变、且这一帧不是光照阶跃。
  /// 「遮挡关系」= 间接光真的依赖的那些几何 —— 默认（「太阳反弹」关）只有**世界整体**变才算
  /// （见 [`GiGpu::wide_rev`]），流式挂载 / 卸载与逐体素编辑**不**算。见 `gi/screen.wesl` 文件头 ⑥、
  /// w = **降噪质量档位**（0 = 关 / 1 = 低 / 2 = 中 / 3 = 高；
  /// 菜单「渲染/RESTIR GI/降噪质量」）：`gi_ss_main` 按它取 1/4 档的新鲜候选数
  /// （`GI_SS_CAND_N` vs `..._HQ`）与记忆窗（`GI_SS_M_CAP_K` vs `..._HQ`）——**这两项与分辨率档无关**。
  /// 时域/atrous 的派发与核半径不在本 pass 里，由 Rust 的 `denoise_plan` 决定。
  pub flags: Vec4,
  /// x = 自增帧号（精确 u32）、**y = 二次顶点缓存的 epoch**（见 [`GiGpu::epoch`]；
  /// [`ShadeKey`] 的修订号 ⇒ `gi_sec_slots` 槽里键的掩码，跨帧持久的前提）、
  /// zw = 保留（恒 0）。所有整数帧逻辑（本帧的 RNG 种子混入、像素 hash）都用 x：
  /// 帧号曾经以 f32 存在 `params.x`，超过 2^24 后无法表示连续整数 ⇒ 种子会偶发重复。
  pub seq: UVec4,
  /// 上一帧的相机矩阵（时域复用：把本帧主命中点重投影到上帧 GI 网格）。
  /// `prepare_gi` 每帧把上帧实际用过的那一份写进 uniform，再把当前帧的存下来 ⇒ 与上帧逐位一致。
  /// 只被 `prev_view_proj` 消费（重投影三维点不需要逆矩阵）。
  pub prev_view_proj: Mat4,
}

/// 光照的**全部**输入（逐项比对，任一变化即"变过"）。
/// 只比位、不比语义（f32 一律比 `to_bits()`）—— 目的只是"变了就失效"，比"算不算真的变了"更保守才对。
/// 主光方向在**没有光量**时也要进 key：日月交接那一帧方向已经跳了、只是不携带能量，下一帧它就带着。
/// 唯一消费者是 [`light_jump`]（判「光照阶跃」⇒ `flags.z`）。
#[derive(Debug, Clone, Copy, PartialEq)]
struct LightKey {
  /// 太阳方向（世界系，指向光）与色 × 强度；无主光时为全 0。
  dir: [u32; 3],
  sun_c: [u32; 3],
  /// 天光色（sRGB 编码，与 `sky_rgb()` 的输入同一个）。
  sky: [u32; 3],
}

impl LightKey {
  fn of(theme: Option<&crate::lighting::LightingTheme>) -> Self {
    let bits3 = |v: [f32; 3]| v.map(f32::to_bits);
    let (dir, sun_c, sky) = match theme {
      Some(t) => {
        let (d, c) = match &t.sun {
          Some(s) => {
            let l = -glam::Vec3::from(s.dir).normalize_or_zero();
            (
              [l.x, l.y, l.z],
              [s.color[0] * s.intensity, s.color[1] * s.intensity, s.color[2] * s.intensity],
            )
          }
          None => ([0.0; 3], [0.0; 3]),
        };
        let sky = t.sky.as_ref().map_or(crate::consts::MINECRAFT_SKY, |s| s.color);
        (bits3(d), bits3(c), bits3(sky))
      }
      None => (bits3([0.0; 3]), bits3([0.0; 3]), bits3(crate::consts::MINECRAFT_SKY)),
    };
    Self { dir, sun_c, sky }
  }
}

/// ② **二次顶点缓存的 epoch 输入**（uniform `seq.y` 的来源）：`gi_face_shade` 的**全部**输入。
/// 变了就自增 ⇒ `gi_sec_slots` 槽里键的掩码跟着变、旧槽一律不匹配（整表自失效，无需清表）。
///
/// 它与 [`LightKey`] 是**两个集合**，差别有两处、都不是笔误（两处都是"值真的不依赖它"，不是放宽）：
///   · **主光只在「太阳反弹」打开时进**：关掉时 `sun_vis = 0` ⇒ `brdf_reflected` 里整支太阳项
///     （`l.sun_vis > 0` 才进）被折掉 ⇒ 出射辐亮度只剩 `(kd·albedo + f0)·amb + emissive`，
///     与主光的方向 / 颜色 / 强度**无关**（`gi/ray.wesl::gi_face_shade`）。
///     CONSTRAINT: 少了这一条，「时刻自动流逝」这类每帧都在动的主光会让整张表每帧自失效 ——
///     缓存等于不存在。实测（2026-09-25，720p/GI 1/2/分帧 8）：epoch 每帧 +1。
///   · **几何的"局部变化"不进本 key**：它由**区域修订表**承担（`gi_region_rev` / `gi_local_rev`）——
///     每个槽的掩码里折进"该面附近变没变"，于是只有附近真的变过的槽才失效。为什么不回到"任何上传
///     都作废整表"的老口径：流式世界里每帧都在挂载 / 卸载 ⇒ 整表每帧自失效 = 缓存等于
///     不存在（2026-09-25 实测：720p/GI 1/2 档下 `gate_gi` 15.56 ms 里的大头正是这个）。
/// 本 key 的 `geom` 因此**只看世界整体**（全量重建 / 调色板 / 键原点，= [`GiGpu::wide_rev`]）。
/// 逐面材质不在本 key 里：它由槽里的 `pal` 参与掩码覆盖（`gi_sec_key_masked`）。
#[derive(Debug, Clone, Copy, PartialEq)]
struct ShadeKey {
  /// **世界整体**几何：= [`GiGpu::wide_rev`]（全量重建 / 调色板 / 键原点；**不含**流式挂载与
  /// 逐体素编辑 —— 局部变化由区域修订表承担）。与 `flags.z` 用的是同一个 `occluder_rev`（见
  /// `prepare_gi`）：两处问的是同一个问题 ——「世界整体变了吗」。
  geom: u32,
  /// 主光（仅 `sun_bounce` 打开时有效，关掉时全 0 ⇒ 不进比较）。
  dir: [u32; 3],
  sun_c: [u32; 3],
  sky: [u32; 3],
  sun_bounce: bool,
}

impl ShadeKey {
  fn of(geom: u32, sun_bounce: bool, light: &LightKey) -> Self {
    let gated = |v: [u32; 3]| if sun_bounce { v } else { [0u32; 3] };
    Self { geom, dir: gated(light.dir), sun_c: gated(light.sun_c), sky: light.sky, sun_bounce }
  }
}

/// 主光方向项生效所需的最小光量（色 × 强度的模）。低于它就当作"这一侧没有光"：方向跳变不携带
/// 辐射量（`sky.rs` 的日月交接帧两侧强度都趋 0）。参考量级：天象表强度 0（地平线）~ 0.82（正午）。
const LIGHT_DIR_MIN_MAG: f32 = 0.02;
/// **光照阶跃**阈值（相对变化 / `1 − cosθ`）：主光或天光一帧内变化超过本值，或主光方向一帧内转过
/// 60° 以上 ⇒ 判定为阶跃（`flags.z = 0`，该帧整帧不复用历史，见 `prepare_gi`）。
/// 常规的逐帧变化都在其下：自动流逝 0.5 游戏小时/秒 ≈ 0.125°/帧；拖动「时刻」滑杆的常规步长同理。
const LIGHT_STEP_MAX: f32 = 0.5;

/// **面键编码原点的量化步长**（体素）。必须与 `gi/common.wesl::GI_KEY_ORG_Q` 逐字相等——
/// 单测 `key_org_q_matches_wesl` 从 `.wesl` 源码解析并断言，改一侧漏另一侧会被测出来。
/// 参见 [`GiGpu::key_org`] 的 CONSTRAINT（为什么要量化）。
pub const KEY_ORG_Q: i32 = 8192;

/// 一组窗口原点（chunk 单位）→ **量化后的体素编码原点**。截断整除，与 shader 侧 `i32 /` 一致。
pub fn key_origins_q(windows: &[glam::IVec3]) -> Vec<glam::IVec3> {
  let chunk = gate_voxel::CHUNK_SIZE as i32;
  windows
    .iter()
    .map(|w| {
      let o = *w * chunk;
      glam::IVec3::new(o.x / KEY_ORG_Q, o.y / KEY_ORG_Q, o.z / KEY_ORG_Q) * KEY_ORG_Q
    })
    .collect()
}

// ============================================================================
// 区域修订表（`gi_region_rev`）：二次顶点缓存的**局部失效**
// ============================================================================
// 语义与 shader 侧 `gi/common.wesl::gi_local_rev` 逐字同源（常量有单测 `region_consts_match_wesl`）。
// 每次上传把覆盖到的区域**及其 ±[`REGION_REACH`] 邻域** +1（膨胀在写侧做 ⇒ 读侧只付一次 load）。

/// 区域边长（chunk）。窗口 64³ chunk ⇒ 8³ = 512 个区域；取 8³ 还让
/// `ceil(太阳阴影行程 12 chunk / 本值) = 2` 的膨胀半径恰好盖住 NEE 的影响球。
pub const REGION_CHUNKS: i32 = 8;
/// 表每轴格数（16³ = 4096 个 u32 = 16 KB）；基址 = 主世界窗口的量化键原点。
pub const REGION_TABLE: i32 = 16;
/// 写侧膨胀半径（区域单位）= `ceil(NEE 行程 12 chunk / REGION_CHUNKS)`。
pub const REGION_REACH: i32 = 2;

/// 窗口原点（chunk）→ 区域表的**基址**（chunk）：与 `gi_key_org` 同一条量化
/// （截断整除，Q = [`KEY_ORG_Q`] / `CHUNK_SIZE` = 32 chunk）。
pub fn region_origin(window_origin: glam::IVec3) -> glam::IVec3 {
  let q = KEY_ORG_Q / gate_voxel::CHUNK_SIZE as i32;
  glam::IVec3::new(window_origin.x / q, window_origin.y / q, window_origin.z / q) * q
}

/// 世界 voxel → 区域格（与 shader `gi_local_rev` 同一条公式：相对键原点 → chunk → 区域 → 钳制；
/// 越界钳到边界 = 保守）。`org_chunks` = [`region_origin`]。
pub fn region_cell(v_world: glam::IVec3, org_chunks: glam::IVec3) -> glam::IVec3 {
  let chunk = gate_voxel::CHUNK_SIZE as i32;
  let rel = (v_world - org_chunks * chunk).max(glam::IVec3::ZERO);
  (rel / chunk / REGION_CHUNKS).min(glam::IVec3::splat(REGION_TABLE - 1))
}

/// 一个闭开世界盒 `[lo, hi)` → 标记区域表（含 ±[`REGION_REACH`] 膨胀）。
/// 盒大到超过 `BIG_BOX_CELLS` 格时退化为"全表 +1"（等价于整表失效，保守且便宜）。
fn region_mark(table: &mut [u32], lo: glam::IVec3, hi: glam::IVec3, org_chunks: glam::IVec3) {
  const BIG_BOX_CELLS: i32 = 256;
  let c0 = region_cell(lo, org_chunks);
  let c1 = if hi.cmpgt(lo).all() { region_cell(hi - glam::IVec3::ONE, org_chunks) } else { c0 };
  let span = c1 - c0 + glam::IVec3::ONE;
  if span.x * span.y * span.z > BIG_BOX_CELLS {
    for v in table.iter_mut() {
      *v = v.wrapping_add(1);
    }
    return;
  }
  for z in c0.z..=c1.z {
    for y in c0.y..=c1.y {
      for x in c0.x..=c1.x {
        for dz in -REGION_REACH..=REGION_REACH {
          for dy in -REGION_REACH..=REGION_REACH {
            for dx in -REGION_REACH..=REGION_REACH {
              let cx = (x + dx).clamp(0, REGION_TABLE - 1);
              let cy = (y + dy).clamp(0, REGION_TABLE - 1);
              let cz = (z + dz).clamp(0, REGION_TABLE - 1);
              let i = (cz * REGION_TABLE * REGION_TABLE + cy * REGION_TABLE + cx) as usize;
              table[i] = table[i].wrapping_add(1);
            }
          }
        }
      }
    }
  }
}

/// **光照阶跃**判据：当前 key 相对上一帧 key 的"最大相对变化"（0 = 没变）。三项取 max：
///   · 主光色 × 强度的相对变化（分母取两侧较大者 ⇒ "从 0 亮起"不会被除零放大）；
///   · 天光色的相对变化（同上）；
///   · 主光方向变化 `1 − cosθ`，只在两侧光量都超过 [`LIGHT_DIR_MIN_MAG`] 时才计。
fn light_jump(prev: &LightKey, now: &LightKey) -> f32 {
  let vec3 = |v: [u32; 3]| glam::Vec3::from(v.map(f32::from_bits));
  let rel = |a: [u32; 3], b: [u32; 3]| -> f32 {
    let (a, b) = (vec3(a), vec3(b));
    (a - b).length() / a.length().max(b.length()).max(1e-4)
  };
  let dir = if vec3(prev.dir).length().min(vec3(now.dir).length()) > LIGHT_DIR_MIN_MAG {
    1.0 - vec3(prev.dir).normalize_or_zero().dot(vec3(now.dir).normalize_or_zero())
  } else {
    0.0
  };
  rel(prev.sun_c, now.sun_c).max(rel(prev.sky, now.sky)).max(dir)
}

/// GI 档位（菜单「渲染/GI」）：`enabled` → uniform `misc.x`；`gi_div` → uniform `flags.y`。
#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq)]
pub struct GiSettings {
  /// GI 开关（关掉 = 整条 GI 链不派发，主 pass 只有太阳直射 + 天光兜底）。
  pub enabled: bool,
  /// GI **分辨率除数**：1 = 全分辨率、2 = 半分辨率、4 = 四分之一（GI 网格边长 = 渲染分辨率 ÷ 本值）。
  /// 不是开关：**每一档都跑 GI**，只是网格疏密与代价不同。
  /// 代价按 **GI 像素数**涨：射线、降噪（时域 + 5 轮 atrous）、reservoir/历史带宽全都 ∝ 1/除数²
  /// ⇒ 1/2 约是全分辨率的 1/4 时间、1/4 再降 4 倍。画质上 GI 是低频信号，1/4 几乎无差
  /// （回全分辨率由几何感知上采样兜住几何边界，见 `main.wesl` 的 `gi_upsample_joint`）。
  pub gi_div: u32,
  /// **降噪质量档**（菜单「渲染/RESTIR GI/降噪质量」）：把降噪链按成本分档，方便直接 A/B。
  /// 每档只比上一档多一件事，成本单调递增（数值的权威在 `gi/consts.wesl`，Rust 只决定派发与取值）；
  /// **与「分辨率」档正交**：下面这些项目对所有分辨率档一视同仁。
  ///
  /// | 档 | 时域累积 | atrous | 每像素候选数 | 记忆窗 | 比上一档多花什么 |
  /// |---|---|---|---|---|---|
  /// | 0 关 | ✗ | ✗ | 4 | 20 | —（基线：原始 GI + 几何感知上采样） |
  /// | 1 低 | ✓ | 5 轮 · 3×3（8 tap） | 4 | 20 | +1 时域 pass + 5 个 atrous pass |
  /// | 2 中 | ✓ | 5 轮 · 5×5（24 tap） | 4 | 20 | atrous 每轮 tap 8→24 |
  /// | 3 高 | ✓ | 5 轮 · 5×5 | 8 | 32 | **候选数翻倍 = GI 射线翻倍**（全链最贵的一项） |
  ///
  /// 档 0 时 `dda_main` 直接采样原始 GI（`gi_out`）——**完全不降噪**，也没有时域/atrous pass。
  /// 「候选数」与「记忆窗」只区分最高档（`screen.wesl` 读 `gi_u.flags.w`）；
  /// 「atrous 核半径」区分 0..=1 / 2..=3（写进降噪的小配置 buffer，见 `AuxTexCache::den_cfg`）；
  /// 「时域/atrous 是否派发、几轮」由 `denoise_plan` 决定（档 0 = 0 轮）。
  /// 「几何感知上采样」不属于降噪（那是把 GI 在几何边界上正确重建），所有档都保留。
  ///
  /// 与业界 NRD/RELAX 的对照见 README 的技术要点第 4 条：本链 = 时域累积（方差驱动权重 + 累积矩 +
  /// AABB 钳制）+ 5 轮迭代 atrous；**没有**独立的前滤 pass（等价物是时域内的共面邻域 mean ± K·σ
  /// 离群钳制）与 fast-history/history-fix。
  pub denoise: u32,
  /// **二次顶点的太阳反弹**（菜单「渲染/RESTIR GI/太阳反弹」）：**默认关**。
  /// GI 射线命中点（二次顶点）是否做一次太阳 NEE —— 朝太阳发一条阴影射线，把「被阳光照亮的
  /// 表面」这一路能量算进间接光。
  ///
  /// 关掉 ⇒ 二次顶点只剩天光项（`albedo · 天光 · AO / π`），**整条阴影射线都不发**。
  /// 实测 2K + 1/4 档：`gate_gi` 27.1 → 16.1ms（占该 pass 的 41%），全帧 31.6 → 20.4ms。
  /// 代价是阴影区/背光面失去「阳光经一次弹射照进来」的贡献 ⇒ 变暗变平；这是**能量层面的取舍**
  /// （不是采样层面的噪声取舍），降噪救不回来。静态场景实测画面差异很小。
  ///
  /// 为什么用"砍掉"而不是"缓存/查表"：那 11ms 里 90% 花在「>32 voxel 的长程遍历」上
  /// （短程只值 1.1ms），而太阳方向在昼夜循环下每帧变化 ⇒ 世界空间的太阳可见性缓存不成立
  /// （流式大世界更不成立）。
  pub sun_bounce: bool,
  /// **每帧采样预算分摊 N**（菜单「渲染/RESTIR GI/分帧」）：把 GI 的采样预算摊到 N 帧上 ——
  /// 每帧只发 `基准候选数 ÷ N` 条射线（至少 1 条），记忆窗同步 ×N。
  ///
  /// **帧时间是平的**（不开分帧 = 1 那条；见 `screen.wesl` 的 `share`）—— 这是它与"跳帧"方案的根本差别：
  /// 跳帧会把同样多的活集中到某一帧（帧时间尖峰），本方案把它均摊到每帧。
  /// 窗内样本总数不变（候选数 ÷N 与记忆窗 ×N 相乘抵消，`gi_res_cap` 的 `cap = m_cap_k × cand_n`）
  /// ⇒ **稳态噪声不变**；**填充不慢**：窗没填满时按「重分配」档的 `hi` 发新射线（见 `realloc`）
  /// ⇒ 填满一个窗约 1 s，稳态之后窗才覆盖 N× 更多的帧（响应更钝）。
  /// 方向不重复：种子含逐帧自增的帧号（`gi_ss_seed`）⇒ 每帧的候选方向都是新的独立样本。
  ///
  /// 整数条数约束：基准候选数 4（降噪质量 低/中）时 N 只能到 4（每帧至少 1 条，N=8 会退化成 4）；
  /// 想要真正 8 倍摊薄需要「降噪质量 = 高」（基准 8 条 ⇒ 1 条/帧）。
  pub share: u32,
  /// **预算重分配档**（菜单「渲染/RESTIR GI/重分配」；0 = 关，1/2/3 = 弱/中/强）。
  /// 按"这个像素的**记忆窗填满了没有**"把每帧的射线预算挪过去：代理量 = 窗内已累积样本数
  /// （`hist.m`；`screen.wesl` 的 `b_err`）—— 它**不是本帧候选数的函数**（单调增、被 `cap` 封顶），
  /// 所以档位表里的倍率真正生效，也不会有"少采样→更脏→要更多采样"的分配振荡。
  /// 窗没填满的像素多发（`hi` = 1.4 / 2.0 / 4.0 倍）⇒ **尽快把窗填满**（这就是"收敛快"的来源）；
  /// 填满之后少发（`lo` = 0.70 / 0.50 / 0.25 倍）⇒ 稳态。
  /// 记忆窗按 `cand_n0 / cand_n` 同步放大 ⇒ 两侧的窗内样本数相同、稳态噪声不变；而**一个样本就是
  /// 一条射线** ⇒ 填满一个窗的射线总账与速率无关，本档只把同一批射线**提前**花掉。
  /// CONSTRAINT: 不要换回二值的"本帧有没有接到历史"（`hist_ok`）：它在第 2 帧就翻真 ⇒ 整段填充期
  /// 都按 `lo` 发 ⇒ 256 个样本的窗要 1024 帧（≈17 s）才填满，表现为"GI 收敛很慢"。
  pub realloc: u32,
  /// **第二条弹射档位**（菜单「渲染/光照/二次弹射」；0 = 关 / 1 = 稀疏 / 2 = 全）。
  ///
  /// GI 原本只算**一次弹射**（二次顶点按「太阳 NEE + 天光 × AO」着色，见 `gi/ray.wesl`）⇒ 丢掉
  /// 二次以上的弹射（多面互反射 / 彩色渗色）。本档在**二次顶点上再发一条余弦射线**，把它的入射
  /// 辐亮度按 `kD·albedo·L2` 叠回二次顶点的出射辐亮度 —— 即"多一跳"，物理上就是缺少的那一项。
  ///
  /// 成本与做法（落点 `gi/screen.wesl` 的候选循环 + `gi/ray.wesl::gi_bounce2`）：
  ///   · 档 1「稀疏」= 每个候选**以 1/4 的概率**带上这一项、带上时乘 4 ⇒ 期望无偏、只有方差变大，
  ///     而**额外射线数 = 候选数的 1/4**（每候选 4 条时正好 +1 条，即 +25% 的 GI 射线）；
  ///   · 档 2「全」= 每个候选都带（额外射线数 = 候选数，即 GI 射线 ×2）—— 极限画质的对照点。
  ///   · 第三条顶点的着色（含它那条太阳 NEE）**走同一个二次顶点缓存** ⇒ 不随候选数增长。
  /// 方向逐候选随机（含帧号）⇒ 这一项不是"冻住的逐面常数"，降噪链的时域累积能把它平均掉。
  pub depth: u32,
}

/// 「降噪质量」档的派发计划（`(是否跑降噪, atrous 轮数, atrous 核半径)`）。
/// 核半径的权威值在 `gi/consts.wesl`；轮数取 `GI_DEN_ATROUS_ITER`（档 0 = 0 轮 = 完全不跑）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenoisePlan {
  /// 是否派发时域 + atrous（档 0 = false ⇒ `dda_main` 直接采样原始 GI）。
  pub on: bool,
  /// atrous 派发几轮（0 到 `GI_DEN_ATROUS_ITER`）。
  pub rounds: u32,
  /// atrous 单轮核半径（1 = 3×3，2 = 5×5）。
  pub radius: u32,
}

impl GiSettings {
  /// 菜单「渲染/GI/分辨率」的三个档位（网格除数）；**下标 = `switch_group` 的选中序号**。
  /// 菜单侧只认下标 ⇒ 档位增删必须改这里（`debug_menu.rs` 的观察者用同一份）。
  pub const DIV_CHOICES: [u32; 3] = [1, 2, 4];

  /// 菜单「渲染/GI/降噪质量」的档位数（0 = 关 ..= 3 = 高）；**下标 = 档位本身**。
  pub const DENOISE_TIERS: u32 = 4;

  /// 菜单「渲染/RESTIR GI/分帧」的四个档位（每帧采样预算分摊 N）；**下标 = 选中序号**。
  pub const SHARE_CHOICES: [u32; 4] = [1, 2, 4, 8];

  /// 生效的预算分摊份数（越界值钳回来）。
  pub fn share(&self) -> u32 {
    self.share.clamp(1, 8)
  }

  /// 菜单「渲染/RESTIR GI/重分配」的档位数（0 = 关 ..= 3 = 强）；**下标 = 档位本身**。
  pub const REALLOC_TIERS: u32 = 4;

  /// 菜单「渲染/光照/二次弹射」的档位数（0 = 关 / 1 = 稀疏 / 2 = 全）。
  pub const BOUNCE2_TIERS: u32 = 3;

  /// 生效的弹射档位（越界值钳回来）。
  pub fn depth_tier(&self) -> u32 {
    self.depth.min(Self::BOUNCE2_TIERS - 1)
  }

  /// 本档的**第二条弹射倍数**（uniform `gi_u.misc.w`）：0 = 关、4 = 稀疏、1 = 全。
  /// uniform 里传的是"倍数"：WESL 侧的抽签概率是它的倒数（`1.0 / max(倍, 1.0)`），
  /// 命中时把这一项乘上倍数补齐期望 —— 两者必须成对，改一处就要改另一处。
  pub fn bounce2_mult(&self) -> f32 {
    match self.depth_tier() {
      0 => 0.0,
      1 => 4.0,
      _ => 1.0,
    }
  }

  /// 生效的重分配档位（越界值钳回来）。
  pub fn realloc_tier(&self) -> u32 {
    self.realloc.min(Self::REALLOC_TIERS - 1)
  }

  /// 生效的分辨率除数（越界值钳回来）。
  pub fn div(&self) -> u32 {
    self.gi_div.clamp(1, 4)
  }

  /// 生效的降噪档位（越界值钳回来）。
  pub fn tier(&self) -> u32 {
    self.denoise.min(Self::DENOISE_TIERS - 1)
  }

  /// 本档的降噪派发计划（见 `DenoisePlan`；`gi/consts.wesl` 的 `GI_DEN_ATROUS_R*` / `_ITER` 是参数源）。
  pub fn denoise_plan(&self) -> DenoisePlan {
    let c = gi_consts();
    match self.tier() {
      // 关：一条降噪 pass 都不跑（`dda_main` 直接采样原始 GI）。
      0 => DenoisePlan { on: false, rounds: 0, radius: c.gi_den_atrous_r_fast },
      // 低：时域累积 + 5 轮 3×3（8 tap）atrous。
      1 => DenoisePlan { on: true, rounds: c.gi_den_atrous_iter, radius: c.gi_den_atrous_r_fast },
      // 中/高：atrous 换 5×5（24 tap）；高 额外动候选数与记忆窗（在 `screen.wesl` 里按档取）。
      _ => DenoisePlan { on: true, rounds: c.gi_den_atrous_iter, radius: c.gi_den_atrous_r },
    }
  }

  /// GI 网格尺寸 = 渲染分辨率 ÷ `div()`（逐轴向下取整，至少 1×1）。
  pub fn gi_size(&self, render_size: UVec2) -> UVec2 {
    let d = self.div();
    UVec2::new((render_size.x / d).max(1), (render_size.y / d).max(1))
  }
}

impl Default for GiSettings {
  fn default() -> Self {
    // 默认 1/4 档 + 「低」降噪档：与实测最划算的组合一致（时域 + 3×3 的 5 轮 atrous），
    // 想要更干净就往「中/高」拨，想量原始噪声与上限帧率就拨到「关」。
    // 太阳反弹默认关：实测画面差异细微（静态场景几乎看不出），代价却是 `gate_gi` 的 41%。
    // 二次弹射默认关（= 与引入前逐位一致）：它改的是**能量**（多一跳的间接光），要开就拨「稀疏」。
    Self { enabled: true, gi_div: 4, denoise: 1, sun_bounce: false, share: 2, realloc: 0, depth: 0 }
  }
}

/// BG4 布局：uniform(0) + reservoir 双缓冲（20 = 本帧写、21 = 上帧读）。
/// 被 `gi_main`（读写 reservoir）与降噪前的着色共用。
pub fn gi_bg4_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let buf = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: false },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "GiBg4",
    &[
      BindGroupLayoutEntry {
        binding: 0,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(GiUniform::min_size()),
        },
        count: None,
      },
      buf(20),
      buf(21),
    ],
  )
}

/// BG5 写入侧布局：GI 输出纹理（2）+ 降噪导引 buffer（6）。
/// 纹理尺寸 = GI 网格（渲染分辨率 ÷ `GiSettings.gi_div`），与全分辨率无关。
/// 采样侧（binding 4/6）在 `brickmap::dda` 里单独一份 layout，只给 `dda_main`。
/// 覆盖度 cov 不占绑定（它是「valid ? 1 : 0」，与 `gi_out.a` 同一个场）。
pub fn gi_bg5_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let store = |binding: u32, format: TextureFormat| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::StorageTexture {
      access: StorageTextureAccess::WriteOnly,
      format,
      view_dimension: TextureViewDimension::D2,
    },
    count: None,
  };
  let buf = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: false },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "GiBg5",
    &[
      // GI 网格分辨率下的 GI：rgb = gi·valid、a = valid
      store(2, TextureFormat::Rgba16Float),
      // 降噪导引（`gi_main` 写、两段降噪读；布局见 WESL `gi/consts.wesl`）
      BindGroupLayoutEntry {
        binding: 6,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      // 帧内逐面去重表（`voxel_raytrace/gi/common.wesl` 的 `FACE_*`；槽数按 GI 网格给）。
      // `read_only = false` 必须与 WESL 侧 `array<atomic<u32>>` 的声明一致：`gi_main` 认领（CAS）、
      // `dda_face_main` 写着色、`dda_main` 查表 —— 三条 pipeline 用的是同一份声明。
      buf(8),
      // 二次顶点按面缓存（`gi/common.wesl` 的 `GI_SEC_*`）：只有 `gi_main` 用（GI 射线的命中着色）。
      buf(9),
      // 区域修订表（`gi/common.wesl` 的「区域修订表」段 / `gi_local_rev`）：只有 `gi_main` 读
      // ⇒ `read_only = true`（与 WESL 侧 `var<storage, read>` 的声明一致；有独立的 Rust 写入者）。
      BindGroupLayoutEntry {
        binding: 10,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: true },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
    ],
  )
}

/// 降噪时域段的布局：**只有 group(0) 一份**（binding 号见 `bindings.wesl`）。
/// 单 group 是刻意的：两段降噪都不需要相机矩阵（上帧像素坐标由 `gi_main` 写进导引）、
/// 也不需要 `grid_descs`（平面检验容差同样写在导引里）⇒ pipeline layout 不必拉进 bg1..bg4。
pub fn gi_den_temporal_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let ro = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: true },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  let rw = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: false },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  let tex = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Texture {
      sample_type: TextureSampleType::Float { filterable: true },
      view_dimension: TextureViewDimension::D2,
      multisampled: false,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "GiDenTemporal",
    &[
      ro(10),  // 导引
      tex(11), // 本帧原始 GI（gi_out 的采样视图）
      ro(12),  // 历史（上帧）
      rw(13),  // 历史（本帧）
      BindGroupLayoutEntry {
        binding: 16, // 时域输出
        visibility: C,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::Rgba16Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      },
      rw(17), // φ（每像素亮度 range 尺度）
      ro(20), // 降噪小配置（`[0]` = atrous 核半径；降噪 pass 够不到 @group(4) 的 `gi_u`）
    ],
  )
}

/// 降噪空间段（迭代 atrous）的布局：group(0)，binding 14 = 采样输入、15 = 存储输出、10 = 导引、
/// 13 = 本帧历史（**只读 M**：短历史稳定化要按本像素的 M 缩放步长）、17 = φ。
/// Rust 每轮换绑 14/15（同一份 layout、同一个入口，见 `AuxTexCache::den_bg`）。
pub fn gi_den_atrous_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  BindGroupLayoutDescriptor::new(
    "GiDenAtrous",
    &[
      BindGroupLayoutEntry {
        binding: 10,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: true },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      // 13 = 本帧历史：shader 里声明成 `read_write`（时域 pass 要写它）⇒ 这里必须是同一档
      // （`read_only: false`），否则 wgpu 在 `create_compute_pipeline` 直接报「binding 不可用」。
      BindGroupLayoutEntry {
        binding: 13,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 14,
        visibility: C,
        ty: BindingType::Texture {
          sample_type: TextureSampleType::Float { filterable: true },
          view_dimension: TextureViewDimension::D2,
          multisampled: false,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 15,
        visibility: C,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::Rgba16Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 17,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 20,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: true },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
    ],
  )
}

/// ① 逐面合并的落地 pass（`gi_face_flatten`）的布局：**只有 group(0)**，三件东西 ——
/// 导引（10，只读）、逐面去重表（18，读写：`gi_main` 认领 + 累加、本 pass 读平均值）、
/// GI 写入侧（19，`gi_out` 的存储视图：本 pass 把面均值写回）。
/// 为什么另起一份而不复用主 pass 的 group(5)：本 pass 只读导引 + 写 GI，**绝不采样 GI 纹理**
/// （同一 pass 内同一张纹理既采样又写入是 wgpu 的硬错），也不碰 brickmap / 相机矩阵 ⇒
/// 一个瘦布局最省，且与 `gi_den_*` 的两个布局是同一个先例。
pub fn gi_flatten_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  BindGroupLayoutDescriptor::new(
    "GiFlatten",
    &[
      BindGroupLayoutEntry {
        binding: 10,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: true },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 18,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 19,
        visibility: C,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::Rgba16Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      },
    ],
  )
}

#[derive(bevy::ecs::resource::Resource)]
pub struct GiGpu {
  pub uniform: bevy::render::render_resource::UniformBuffer<GiUniform>,
  pub frame: u32,
  /// 上一帧 `gi_main` 实际用过的相机矩阵（uniform `prev_view_proj` 的来源）。
  /// 只在真正跑 GI 的帧更新 ⇒ 与上帧写入 reservoir 时用的矩阵逐位一致（时域重投影才准）。
  pub prev_view_proj: Mat4,
  /// reservoir 双缓冲的换绑状态：true ⇒ 本帧 `binding 20 = b`、`21 = a`（见 `prepare_gi`）。
  pub res_flip: bool,
  /// 每个 volume 上一帧的**量化编码原点**（`None` = 还没比过）。
  ///
  /// 键只有 16 bit/轴（±32768 体素 = ±655 m），而世界远超它 ⇒ shader 侧一律**减去窗口原点**再编码
  /// （`gi/common.wesl::gi_key_org`）。窗口随相机走 ⇒ 原点一变，上一帧的键就在另一个坐标系里
  /// ⇒ 必须把整条复用链作废。这里与 shader **逐字同源**：取 `GpuBrickMap::volume_windows`（chunk 单位），
  /// 乘 `CHUNK_SIZE` 得到体素原点，再按 [`KEY_ORG_Q`] 量化（截断整除，与 shader 的 `i32 /` 一致）。
  ///
  /// CONSTRAINT: 必须量化。流式窗口是相机居中的，相机每跨一个 chunk 它就平移一格 ⇒ 不量化的话
  /// 正常移动即 ~15 次/s 作废整链（静止之外的画面没有机会累积，表现为"一动就闪"）。量化后只在
  /// 相机跨过 `KEY_ORG_Q` 边界时才变。步长必须与 `gi/common.wesl::GI_KEY_ORG_Q` 相等（有单测闸门）。
  pub key_org: Option<Vec<glam::IVec3>>,
  /// **区域修订表**（`gi_region_rev`，`REGION_TABLE³` 个 u32）：Rust 维护、有变化时整表上传。
  /// 每次上传把覆盖区域 ±[`REGION_REACH`] +1（见 [`region_mark`]）⇒ 二次顶点缓存**局部失效**：
  /// 只有面附近真的变过的槽才失效（旧口径"任何上传作废整表"在流式世界里等于没有缓存）。
  region_rev: Vec<u32>,
  /// 区域表的 GPU 缓冲（16 KB，惰性创建；未创建时 BG5 绑占位 buffer ⇒ shader 侧
  /// [`gi_local_rev`] 由 `arrayLength` 守卫返回 0）。
  region_buf: Option<bevy::render::render_resource::Buffer>,
  /// 表当前的基址（chunk，见 [`region_origin`]；`None` = 还没比过）：基址一变（= 键原点变）
  /// 就整表 +1 重新起算（与 `wide_rev` 的作废同步）。
  region_org: Option<glam::IVec3>,
  /// 区域表上传的**暂存字节**（复用，避免每帧分配）。
  region_staging: Vec<u8>,
  /// 上一次真正跑 `gi_main` 的那一帧的 `occluder_rev`（= `wide_rev`；那正是 reservoir 双缓冲里
  /// 「上帧」的来源帧）⇒ 相等 ⇔ 上帧的累计量在本帧仍然成立。
  /// 它与「本帧是不是光照阶跃」合起来决定 uniform `flags.z`（**允许复用历史**）：
  /// 世界整体变了、或光照一帧内跳过了 [`LIGHT_STEP_MAX`]，都整帧不复用（见 `prepare_gi`）。
  pub world_rev_gi: u32,
  /// 世界几何修订号的**世界整体**口径：只在**全量重建 / 调色板变化**（以及键原点变化）时自增，
  /// **不含**流式挂载 / 卸载与逐体素编辑的脏盒。**它就是那个 `occluder_rev`**（`flags.z` 与
  /// `ShadeKey.geom` 都用它）—— 局部几何变化改由区域修订表承担，不再整帧丢历史（见 `prepare_gi`）。
  pub wide_rev: u32,
  /// 降噪 pipeline：`[0]` = 时域、`[1..6]` = atrous 第 1..5 轮（步长 1/2/4/8/16）。
  /// layout 只有 group(0) 一份（见 [`gi_den_temporal_layout`] / [`gi_den_atrous_layout`]）；
  /// 实际跑几轮由 `GI_DEN_ATROUS_ITER` 决定（1..=5，Rust 按它选 src→dst 链）。
  pub den_pipelines: [Option<CachedComputePipelineId>; 6],
  /// ① 逐面合并的落地 pass（`gi_face_flatten`）：layout 只有 group(0)（见 [`gi_flatten_layout`]），
  /// 派发在 `gi_main` 之后、降噪链之前。
  pub flatten_pipeline: Option<CachedComputePipelineId>,
  /// ② **二次顶点缓存的 epoch**（uniform `seq.y`）：`gi_face_shade` 的**全部**输入变化就自增。
  /// 它参与 `gi_sec_slots` 槽里键的掩码 ⇒ 一变整张表自失效。表**不清空**（省掉每帧的 `clear_buffer`），
  /// 所以 epoch 必须覆盖 `gi_face_shade` 的**全部**输入（见 [`ShadeKey`]）。
  pub epoch: u32,
  /// 上一帧用过的 epoch 输入（`None` = 还没比过 ⇒ 首帧自增一次，无妨）。
  epoch_key: Option<ShadeKey>,
  /// 上一帧的光照（`None` = 还没比过）：[`light_jump`] 的比较对象。
  light_key: Option<LightKey>,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct GiBg4(pub bevy::render::render_resource::BindGroup);

/// GI 写入侧 bind group（`gi_main` 用）
#[derive(bevy::ecs::resource::Resource)]
pub struct GiBg5(pub bevy::render::render_resource::BindGroup);

/// GI 写入侧的占位纹理（1×1）：GI 缓冲未就绪时 BG5 仍须为 binding 2 提供视图。
/// 占位纹理不会被真正写入。
#[derive(bevy::ecs::resource::Resource, Default)]
struct GiPlaceholder {
  tex: Option<bevy::render::render_resource::Texture>,
  view: Option<bevy::render::render_resource::TextureView>,
  /// BG4 binding 20/21 的占位（GI 缓冲未就绪时用；1 个 word，足够绑定，不会被读写）。
  res: Option<bevy::render::render_resource::Buffer>,
}

impl GiPlaceholder {
  fn view(
    &mut self,
    device: &bevy::render::renderer::RenderDevice,
  ) -> &bevy::render::render_resource::TextureView {
    use bevy::render::render_resource::*;
    if self.tex.is_none() {
      let t = device.create_texture(&TextureDescriptor {
        label: Some("gate_gi_placeholder"),
        size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
      });
      self.view = Some(t.create_view(&TextureViewDescriptor::default()));
      self.tex = Some(t);
    }
    self.view.as_ref().expect("刚创建")
  }

  /// BG4 binding 20/21 的占位 buffer（4 B）。
  fn res_buffer(
    &mut self,
    device: &bevy::render::renderer::RenderDevice,
  ) -> &bevy::render::render_resource::Buffer {
    use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
    self.res.get_or_insert_with(|| {
      device.create_buffer(&BufferDescriptor {
        label: Some("gate_gi_res_placeholder"),
        size: 4,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
      })
    })
  }
}

pub struct GiPlugin;

impl bevy::app::Plugin for GiPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    use bevy::ecs::schedule::IntoScheduleConfigs;
    app.init_resource::<GiSettings>();
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
      return;
    };
    render_app
      .init_resource::<GiSettings>()
      .init_resource::<GiPlaceholder>()
      .add_systems(bevy::render::RenderStartup, init_gi_gpu)
      .add_systems(
        bevy::render::RenderStartup,
        queue_gi_pipelines.after(crate::brickmap::dda::init_dda_pipelines),
      )
      .add_systems(bevy::render::ExtractSchedule, extract_gi_settings)
      .add_systems(
        bevy::render::Render,
        prepare_gi
          .in_set(bevy::render::RenderSystems::PrepareBindGroups)
          .after(crate::brickmap::upload::prepare)
          .after(crate::brickmap::dda::prepare_dda_bind_groups),
      );
  }
}

fn init_gi_gpu(mut commands: bevy::ecs::system::Commands) {
  commands.insert_resource(GiGpu {
    uniform: bevy::render::render_resource::UniformBuffer::default(),
    frame: 0,
    // 首帧没有「上一帧」⇒ 恒等矩阵；此时 reservoir 两块都是零（M = 0）⇒ 复用一律判无效。
    prev_view_proj: Mat4::IDENTITY,
    res_flip: false,
    key_org: None,
    region_rev: vec![0; (REGION_TABLE * REGION_TABLE * REGION_TABLE) as usize],
    region_buf: None,
    region_org: None,
    region_staging: Vec::new(),
    // 从 0 起 ⇒ 首帧若恰好没有检测到任何变化，`world_same` 会是真；那时 reservoir 两块
    // 都是零（M = 0 ⇒ 一律判无效），跳过与否都不会接受任何历史 ⇒ 安全。
    world_rev_gi: 0,
    wide_rev: 0,
    den_pipelines: [None; 6],
    flatten_pipeline: None,
    epoch: 0,
    epoch_key: None,
    light_key: None,
  });
}

fn queue_gi_pipelines(
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  dda_shader: bevy::ecs::system::Res<crate::shader::DdaShaderHandle>,
  mut gpu: bevy::ecs::system::ResMut<GiGpu>,
) {
  use bevy::render::render_resource::ComputePipelineDescriptor;
  use std::borrow::Cow;
  // 降噪（两步四 pass）：不依赖 DdaPipelines —— layout 只有 group(0) 一份。
  if gpu.den_pipelines[0].is_some() {
    return;
  }
  let den_temporal_layout = gi_den_temporal_layout();
  let den_atrous_layout = gi_den_atrous_layout();
  let label = [
    "gate_gi_denoise_temporal",
    "gate_gi_denoise_atrous1",
    "gate_gi_denoise_atrous2",
    "gate_gi_denoise_atrous4",
    "gate_gi_denoise_atrous8",
    "gate_gi_denoise_atrous16",
  ];
  let entry = [
    "gi_denoise_temporal",
    "gi_denoise_atrous1",
    "gi_denoise_atrous2",
    "gi_denoise_atrous4",
    "gi_denoise_atrous8",
    "gi_denoise_atrous16",
  ];
  for i in 0..6 {
    let layout =
      if i == 0 { vec![den_temporal_layout.clone()] } else { vec![den_atrous_layout.clone()] };
    gpu.den_pipelines[i] = Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
      label: Some(Cow::from(label[i])),
      layout,
      shader: dda_shader.0.clone(),
      entry_point: Some(Cow::from(entry[i])),
      ..Default::default()
    }));
  }
  // ① 逐面合并的落地 pass：layout 只有 group(0)（导引 / 逐面去重表 / GI 写入侧，见 `gi_flatten_layout`）。
  // 它不依赖 `DdaPipelines`（那份 layout 里有 brickmap 与相机矩阵），所以与降噪那几个排在同一处。
  gpu.flatten_pipeline = Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_gi_face_flatten")),
    layout: vec![gi_flatten_layout()],
    shader: dda_shader.0.clone(),
    entry_point: Some(Cow::from("gi_face_flatten")),
    ..Default::default()
  }));
}

fn extract_gi_settings(
  mut commands: bevy::ecs::system::Commands,
  settings: Option<bevy::render::Extract<bevy::ecs::system::Res<GiSettings>>>,
) {
  commands.insert_resource(settings.map_or_else(GiSettings::default, |s| GiSettings {
    enabled: s.enabled,
    gi_div: s.div(),
    denoise: s.tier(),
    sun_bounce: s.sun_bounce,
    share: s.share(),
    realloc: s.realloc_tier(),
    depth: s.depth_tier(),
  }));
}

#[allow(clippy::too_many_arguments)]
fn prepare_gi(
  mut commands: bevy::ecs::system::Commands,
  device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<bevy::render::renderer::RenderQueue>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  settings: bevy::ecs::system::Res<GiSettings>,
  view: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaViewUniform>>,
  lighting: Option<bevy::ecs::system::Res<crate::lighting::LightingTheme>>,
  aux: Option<bevy::ecs::system::Res<crate::brickmap::dda::AuxTexCache>>,
  dirty: Option<bevy::ecs::system::Res<crate::brickmap::upload::BrickMapDirty>>,
  brickmap: Option<bevy::ecs::system::Res<crate::brickmap::upload::GpuBrickMap>>,
  mut gi_ph: bevy::ecs::system::ResMut<GiPlaceholder>,
  mut gpu: bevy::ecs::system::ResMut<GiGpu>,
) {
  gpu.frame = gpu.frame.wrapping_add(1);

  // ---- 世界几何修订号（uniform `flags.z` = **本帧允不允许复用历史**）----
  // 键只能证明"上帧那个面还在"，证明不了"上帧累计进来的那些光路还成立"：reservoir 里存的是
  // `w_sum` / `M` 这条**累加量**（`gi/screen.wesl` 文件头 ⑥）。
  // 口径分两层（2026-10 起）：
  //   · **世界整体**（[`GiGpu::wide_rev`]）—— 全量重建 / 调色板 / 键原点：整链作废（`flags.z` = 0，
  //     二次顶点缓存的 `ShadeKey.geom` 也走它）；
  //   · **局部几何** —— 流式挂载 / 卸载与逐体素编辑：**不再**整帧丢历史，改由**区域修订表**
  //     （`gi_region_rev`，见 [`region_mark`]）在 shader 侧按面**局部**作废二次顶点缓存。
  //
  // CONSTRAINT: 别把局部几何塞回这里（即别用"任何上传"）—— 流式世界每帧都在挂载 / 卸载 chunk
  // （实测 720p：`UPLOAD[incremental]` 每帧 6~8 chunk）⇒ `flags.z` 恒 0、缓存每帧整表失效 ⇒
  // ① 每个像素都走"没有历史"的贵路径（候选数 ×2~4）；② 时域累积恒不成立（`M` 每帧从头来）
  // —— 这正是"光影噪声严重、静态也不收敛"的直接原因（2026-09-25 实测：720p/GI 1/2 档下
  // `gate_gi` 15.56 ms 里的大头就是缓存每帧自失效）。
  //
  // 运行时不存在别的几何变化源：物体变换只在建世界时设定，LOD / beam 只改遍历起点、不改最近命中。
  // 若将来加了「物体动画 / 运行时改变换」，必须让那条路径也同时推这两个修订号。
  if dirty.as_ref().is_some_and(|d| d.full || d.palette_changed) {
    gpu.wide_rev = gpu.wide_rev.wrapping_add(1);
  }
  let occluder_rev = gpu.wide_rev;
  let world_same = occluder_rev == gpu.world_rev_gi;

  // ---- 区域修订表：把本帧的**局部**几何变化记进去（二次顶点缓存的局部失效）----
  // 只认主世界窗口（`volume_windows[0]`）；基址 = 窗口原点的量化（与键原点同源，见 [`region_origin`]）。
  // 基址一变（= 键原点变；那一路 `wide_rev` 也会作废）就整表 +1 重新起算。
  let mut region_touched = false;
  if let Some(bm) = brickmap.as_ref()
    && let Some(w) = bm.volume_windows.first()
  {
    let org = region_origin(*w);
    if gpu.region_org != Some(org) {
      for v in gpu.region_rev.iter_mut() {
        *v = v.wrapping_add(1);
      }
      gpu.region_org = Some(org);
      region_touched = true;
    }
    if let Some(d) = dirty.as_ref() {
      // 盒可能属于物体 / 远场卷（已换算成世界坐标）—— 一律折进主世界表（保守：只会多失效）。
      for b in &d.boxes {
        region_mark(&mut gpu.region_rev, b.lo, b.hi, org);
        region_touched = true;
      }
    }
  }
  if region_touched {
    // 惰性建 16 KB storage buffer + 整表上传（有变化才写：流式每帧也就 16 KB，可忽略）。
    let buf = gpu
      .region_buf
      .get_or_insert_with(|| {
        device.create_buffer(&bevy::render::render_resource::BufferDescriptor {
          label: Some("gate_gi_region_rev"),
          size: (REGION_TABLE * REGION_TABLE * REGION_TABLE * 4) as u64,
          usage: bevy::render::render_resource::BufferUsages::STORAGE
            | bevy::render::render_resource::BufferUsages::COPY_DST,
          mapped_at_creation: false,
        })
      })
      .clone();
    // 暂存 vec 先 `take` 出来（借用分割：`gpu` 经 `ResMut` 解引用，字段级拆分通不过借用检查）。
    let mut staging = std::mem::take(&mut gpu.region_staging);
    staging.clear();
    for v in &gpu.region_rev {
      staging.extend_from_slice(&v.to_le_bytes());
    }
    queue.write_buffer(&buf, 0, &staging);
    gpu.region_staging = staging;
  }

  // ---- ② 二次顶点缓存的 epoch（同一次比对给出「光照阶跃」）----
  // 两个集合刻意不同、各服务一件事（见 [`LightKey`] / [`ShadeKey`] 的说明）：
  //   · `light_key`（全部光量）→ [`light_jump`] → `light_step` → `flags.z`；
  //   · `shade_key`（`gi_face_shade` 真正读到的那些）→ epoch → `gi_sec_slots` 的键掩码。
  //
  // `light_step` = 光照量在**一帧内**跳过了 [`LIGHT_STEP_MAX`]。它与"世界几何变过"同口径：
  // 该帧整帧不复用历史（`flags.z = 0`）。理由是两侧证据的时间基准已经劈开 ——
  // 二次顶点缓存被 epoch 作废 ⇒ 本帧的新鲜候选算的是**新光照**，而 reservoir 里存的 `w_sum` / `M`
  // 是**旧光照**下攒出来的。不劈开就只能靠记忆窗淡出：整屏一起滞后（每个像素的窗口相位不同，
  // 于是暗下来的先后也不同步），相机一动重投影滑到邻近 texel 还会借到旧值 ⇒ 大片鬼影。
  // 逐帧的小变化（自动流逝 / 拖动「时刻」滑杆）不触发：那种变化每个窗口帧只有一小步，历史跟着走
  // 正是想要的行为。
  let light = LightKey::of(lighting.as_deref());
  let mut light_step = false;
  if gpu.light_key != Some(light) {
    light_step = gpu.light_key.as_ref().is_some_and(|p| light_jump(p, &light) > LIGHT_STEP_MAX);
    gpu.light_key = Some(light);
  }
  let shade_key = ShadeKey::of(occluder_rev, settings.sun_bounce, &light);
  if gpu.epoch_key != Some(shade_key) {
    gpu.epoch_key = Some(shade_key);
    gpu.epoch = gpu.epoch.wrapping_add(1);
    bevy::log::debug!(target: "gate", "GI 二次顶点缓存 epoch → {} [sun_bounce={}]",
                      gpu.epoch, settings.sun_bounce);
  }
  if light_step {
    bevy::log::debug!(target: "gate", "GI 光照阶跃 ⇒ 本帧不复用历史");
  }

  // ---- uniform（字段与 WESL `GiUniform` 逐字段镜像）----
  let u = GiUniform {
    params: Vec4::new(
      if settings.sun_bounce { 1.0 } else { 0.0 },
      0.0,
      crate::consts::GI_GAIN,
      0.0,
    ),
    misc: Vec4::new(
      if settings.enabled { 1.0 } else { 0.0 },
      settings.share() as f32,
      settings.realloc_tier() as f32,
      // ④ 第二条弹射的倍数（0 = 关 / 4 = 稀疏 / 1 = 全；WESL 侧按它的倒数抽签）。
      settings.bounce2_mult(),
    ),
    flags: Vec4::new(
      0.0,
      settings.div() as f32,
      // z = 「本帧允许复用历史」（WESL `gi_u.flags.z`）：几何没变 **且** 这一帧不是光照阶跃。
      if world_same && !light_step { 1.0 } else { 0.0 },
      // w = 降噪质量档位（0..=3）：`gi_ss_main` 按它取候选数与记忆窗（只有最高档不同）。
      settings.tier() as f32,
    ),
    // 整数帧号走 u32 通道（`seq.x`）：`gpu.frame` 本就是 u32，不再经 `params.x` 的 f32 截断。
    // `seq.y` = 二次顶点缓存的 epoch（见上面的比对块）—— 它必须与本次写入的 `gi_sec_slots` 一致，
    // 所以同一次 uniform 写入里一起下发。
    seq: UVec4::new(gpu.frame, gpu.epoch, 0, 0),
    // 上一帧相机矩阵（时域重投影）：写「上帧真正用过的那一份」，再把本帧存下来。
    prev_view_proj: gpu.prev_view_proj,
  };
  *gpu.uniform.get_mut() = u;
  gpu.uniform.write_buffer(&device, &queue);
  // 只在真正会跑 `gi_main` 的帧更新「上一帧」⇒ 与上帧写 reservoir 时用的矩阵逐位一致
  // （GI 关掉一段时间再打开时，历史 reservoir 与 prev 矩阵都停留在最后一帧 GI，重投影仍自洽）。
  // 注意：**分辨率的任意取值都跑 GI** —— 这里只跟 `enabled` 走。
  // 「分帧」不在这里：它是"**每帧**少发几条射线、记忆窗同步拉长"（`screen.wesl` 的 `share`，
  // 见 `GiSettings::share`），GI 链本身仍然每帧都跑 ⇒ 帧时间是平的，没有跳帧。
  let gi_runs = settings.enabled;
  if gi_runs {
    // CONSTRAINT: `world_rev_gi` 只跟「GI 链跑没跑」走，**不能**被 `view` 的存在性门住 ——
    // 它的语义是"本帧的 reservoir 是在哪个 `occluder_rev` 下写出的"，与相机矩阵无关。
    // 曾经它与下面的矩阵更新共用一个 `&& view.is_some()`：只要有帧"GI 跑了但没有 view"，
    // 它就停住，而 `occluder_rev`（= `wide_rev`）在启动的全量重建 / 调色板上传时
    // 已经推走 ⇒ `world_same` **永久为假** ⇒ `flags.z` 恒 0 ⇒ WESL 侧 `hist_ok` 恒假
    // （`gi/screen.wesl`：`pk_ok && prev_ok && gi_u.flags.z > 0.5`）⇒ 时域累积永不成立
    // = **永不收敛 + 逐帧闪烁**，且与相机是否静止、与任何 GI 档位都无关。
    gpu.world_rev_gi = occluder_rev;
    // 上一帧相机矩阵只服务重投影 ⇒ 它才该跟 `view` 走（`view` 缺失时保持不动，重投影自洽）。
    if let Some(v) = view.as_ref() {
      gpu.prev_view_proj = v.view_proj;
    }
    // **面键的编码原点变了就把整条复用链作废**（shader 侧 `gi/common.wesl::gi_key_org`）：
    // 上一帧所有键都在另一个坐标系里，留着只会让"同面判定"误配。
    // 推 `wide_rev` 一次同时打到三处（`occluder_rev` 恒等于 `wide_rev`）：
    // ① reservoir 历史（`flags.z` = 0，见上面的 `world_same`）② 二次顶点缓存（`ShadeKey.geom` ⇒
    // epoch）③ 区域修订表（基址变 ⇒ 整表 +1，见上面的区域表维护）。
    // 逐 volume 比（主世界 / 远场级 / 物体各有自己的窗口），任一个变过就整链作废。
    // CONSTRAINT: 原点**已量化**（[`KEY_ORG_Q`]）—— 窗口逐 chunk 跟相机平移，不量化会每跨一格
    // 作废一次（正常移动 ~15 次/s）⇒ 画面没有机会累积。量化后只在相机跨过 Q 边界时才作废。
    if let Some(bm) = brickmap.as_ref() {
      let org = key_origins_q(&bm.volume_windows);
      if gpu.key_org.as_deref() != Some(org.as_slice()) {
        if gpu.key_org.is_some() {
          gpu.wide_rev = gpu.wide_rev.wrapping_add(1);
        }
        gpu.key_org = Some(org);
      }
    }
  }

  // ---- BG4：uniform + reservoir 双缓冲（绑定号 0/20/21，必须显式给 entry）----
  use bevy::render::render_resource::{BindGroupEntry, BindingResource};
  let bg4_layout = pipeline_cache.get_bind_group_layout(&gi_bg4_layout());
  // 两块由 AuxTexCache 随 GI 分辨率一起创建；未就绪（首帧 / prepare 提前返回）时用 4 B 占位
  // （该帧不会派发 `gi_main`）。
  let (res_cur, res_prev) = match aux.as_ref().and_then(|a| a.gi_res_buffers()) {
    Some((a, b)) if gpu.res_flip => (b, a),
    Some((a, b)) => (a, b),
    None => {
      let p = gi_ph.res_buffer(&device);
      (p, p)
    }
  };
  let bg4 = device.create_bind_group(
    None,
    &bg4_layout,
    &[
      BindGroupEntry { binding: 0, resource: gpu.uniform.binding().expect("uniform 已写入") },
      BindGroupEntry { binding: 20, resource: res_cur.as_entire_binding() },
      BindGroupEntry { binding: 21, resource: res_prev.as_entire_binding() },
    ],
  );
  commands.insert_resource(GiBg4(bg4));
  // 换绑：本帧的写入目标成为下一帧的读源。
  if gi_runs {
    gpu.res_flip = !gpu.res_flip;
  }

  // ---- BG5：GI 的写入侧（绑定号 2/6）----
  // GI 视图来自 `crate::brickmap::dda::AuxTexCache`；未就绪时用 1×1 占位纹理（bind group 必须给全条目）。
  // 视图/ buffer 都先 clone 成句柄（`TextureView`/`Buffer` 都是 Arc 包装）⇒ 之后还能再借一次 `gi_ph`。
  let bg5_layout = pipeline_cache.get_bind_group_layout(&gi_bg5_layout());
  let gi_view = match aux.as_ref().and_then(|a| a.gi_write_view()) {
    Some(v) => v.clone(),
    None => gi_ph.view(&device).clone(),
  };
  // binding 6 = 降噪导引（`gi_main` 写）；未就绪时用 4 B 占位（该帧不会派发 `gi_main`）。
  let guide = aux
    .as_ref()
    .and_then(|a| a.gi_guide_buffer())
    .cloned()
    .unwrap_or_else(|| gi_ph.res_buffer(&device).clone());
  // binding 8 = 帧内逐面去重表（`gi_main` 认领）；未就绪时用 4 B 占位 —— 此时该帧不会派发
  // `gi_main`（`prepare_dda_bind_groups` 提前返回），且 `face_slot_of` 会因 `arrayLength` 不足而跳过。
  let face_slots = aux
    .as_ref()
    .and_then(|a| a.face_slots_buffer())
    .cloned()
    .unwrap_or_else(|| gi_ph.res_buffer(&device).clone());
  // binding 9 = 二次顶点按面缓存（GI 射线命中着色用）；未就绪时同样退化成 4 B 占位
  // （`gi_sec_slot_of` 会因 `arrayLength` 不足而返回 ok = false ⇒ 走内联着色）。
  let gi_sec_slots = aux
    .as_ref()
    .and_then(|a| a.gi_sec_slots_buffer())
    .cloned()
    .unwrap_or_else(|| gi_ph.res_buffer(&device).clone());
  // binding 10 = 区域修订表（二次顶点缓存的局部失效）：本帧刚在上方维护/上传；未建时用 4 B 占位
  // （`gi_local_rev` 会因 `arrayLength` 不足而返回 0 ⇒ 退化为旧行为）。
  let region_rev = gpu.region_buf.clone().unwrap_or_else(|| gi_ph.res_buffer(&device).clone());
  let bg5 = device.create_bind_group(
    None,
    &bg5_layout,
    &[
      BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&gi_view) },
      BindGroupEntry { binding: 6, resource: guide.as_entire_binding() },
      BindGroupEntry { binding: 8, resource: face_slots.as_entire_binding() },
      BindGroupEntry { binding: 9, resource: gi_sec_slots.as_entire_binding() },
      BindGroupEntry { binding: 10, resource: region_rev.as_entire_binding() },
    ],
  );
  commands.insert_resource(GiBg5(bg5));
}
