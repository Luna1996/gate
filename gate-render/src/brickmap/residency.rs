//! M3 的**常驻决策核心**：纯逻辑，不碰 GPU、不碰树。
//!
//! 输入 = 相机所在 chunk + 当前常驻账目（字节 / 档位 / 钉住）+ 各 chunk 想要的档位；
//! 输出 = 该安装哪些（含**降级 / 升级**）、该换出哪些。真正的安装 / 释放由 [`super::builder`] 做，
//! 逐帧调度由 `upload` 做。
//!
//! ## 优先级由**距离**定，不由渲染结果定
//!
//! 「最近使用」的排序键是**相机到 chunk 的距离**（切比雪夫），不是渲染时打下的用途戳 —— 后者
//! 随 ray-guided 请求一起取消（`trace.wesl::REQ_ENABLE = 0`）。距离是同一个信号的免费近似：
//! 相机在哪儿，哪儿的块就该留；`want_level` 的档位阶梯本来就按距离给。
//!
//! ## 档位由**距离**定，不由预算定（这是"性能优先但不许牺牲视觉"的落点）
//!
//! [`want_level`] 按"proxy 块自身 ≤ 1 像素"给档位：`fp ≥ 块边长` 才允许把该块塌成一个颜色。
//! 所以**近处永远是全分辨率**，粗化只发生在远端；预算不够时只会从**最远的**开始整块丢掉
//! （那才是真正看得见的代价，且只由预算触发）。
//!
//! ## 四条规则（`docs/editable-gigavoxel.md` §1）
//! 1. **钉住**：被编辑过的 chunk 在 `pin_frames` 内不许换出 / 降级；
//! 2. **最小驻留**：刚安装的 chunk 在此期间不许换出；
//! 3. **档位迟滞**：`want_level` 带 3/4 迟滞 ⇒ 相机在阈值附近来回走不会反复重装（重装 = 0.9 ms 尖峰 + 画面跳变）；
//! 4. **每帧上限**：安装（唤醒 + 重装）每帧不超过 `max_install_per_frame`，把尖峰摊到多帧。
//!
//! 预算 `budget_bytes = 0` 表示**不限** ⇒ 只走档位阶梯、不整块丢弃（这是默认）。

use std::collections::{HashMap, HashSet};

use gate_voxel::{BRICK_FACTOR, CHUNK_SIZE, ChunkCoord};
use glam::IVec3;

/// 档位 = proxy 的 `keep_extent`（4 = 全分辨率不截断）。
pub type Level = i32;

/// **档位阶梯**：`(keep_extent, 允许该档的最小 fp)`。
/// `fp = t·px_ang`（`brickmap::dda::px_ang`，与 shader 同一入口）⇒ 判据与 `docs/editable-gigavoxel.md`
/// §3.3 表一致：**丢弃尺度 = 子块边长**（16³ 档丢 4 体素、64³ 档丢 16 体素…）。
/// 与叶级 `fp ≥ 1` 同一条口径（那里丢 1 体素 ⇒ fp ≥ 1）。
/// CONSTRAINT: 代表色取"块内首个非空体素色"，整块结构被替换 ⇒ 轮廓外扩最多一个块边长；
/// 在阈值处 16³ 档 = 16 体素 ≈ 4 像素，与叶级档同量级（已明确接受）。要更保守就把阈值 ×4
/// （即改判据为"块自身 ≤ 1 像素"：16³ ⇔ fp ≥ 16）。
pub const LADDER: &[(Level, f32)] = &[
  (CHUNK_SIZE, 64.0), // 整 chunk 单色
  (64, 16.0),
  (16, 4.0),
  (BRICK_FACTOR, 0.0), // 全分辨率
];

/// 迟滞系数：从细档退到粗档要 `fp ≥ 阈值`，从粗档回到细档要 `fp < 阈值 × 该值`（两者之间保持原档）。
const HYSTERESIS: f32 = 0.75;

/// **首次安装的引导档**（渐进加载：先给糊的，再变清晰）。
///
/// WHY：全分辨率首次安装要**整树序列化**（实测 ~99 µs/块）⇒ 跟不上一移动就留洞。先按这一档装上去
/// （`ChunkTree::proxy` 几 µs，比全树便宜约 15×），随后由 `plan` 的常规路径**升级**到目标档。
/// 配合 `world.wesl::trace_scene` 的远场兜底（保证"总有东西"），这条保证"多快变清晰"。
///
/// 取 `16`（16³ 档）：形状已经可辨（不是整块一个色），而代价仍在个位数 µs。
pub const BOOTSTRAP_LEVEL: Level = 16;

/// **首次安装（粗档引导）每帧上限**。
///
/// 它便宜（几 µs/块），所以**不占** `max_install_per_frame` 的名额（那个名额留给细化）；但不设上限
/// 时，冷启动那一下会把全部待装块塞进一帧 ⇒ 主线程尖峰。取 128 ≈ 0.8 ms 上界。
const BOOTSTRAP_PER_FRAME: usize = 128;

/// 想要哪一档：`cur_level` = 当前档（参与迟滞）。
/// 变粗按阈值（远端只看得到粗块）；**变细要退出迟滞带**（`fp < 阈值 × 0.75`）⇒ 相机在阈值附近
/// 来回走不会反复重装（重装 = 0.9 ms 尖峰 + 画面跳变）。
pub fn want_level(dist_voxels: f32, px_ang: f32, cur_level: Level) -> Level {
  let fp = dist_voxels * px_ang;
  let raw = raw_level(fp);
  if raw >= cur_level {
    return raw; // 相同或更粗：按阈值即可
  }
  if fp < threshold_of(cur_level) * HYSTERESIS { raw } else { cur_level }
}

/// 按阈值直接给档：`fp` 够大 ⇒ 允许更粗的档（`LADDER` 是粗 → 细序，取第一个够格的）。
///
/// `pub(crate)`：`plan_residency` 的账目同步要用它给"首次见到的 chunk"**记准档位**（见那里的说明）
/// —— 记错档会让每块白付一次降级重装。
pub(crate) fn raw_level(fp: f32) -> Level {
  for &(level, thr) in LADDER {
    if fp >= thr {
      return level;
    }
  }
  BRICK_FACTOR
}

fn threshold_of(level: Level) -> f32 {
  LADDER.iter().find(|&&(l, _)| l == level).map_or(0.0, |&(_, t)| t)
}

/// 单个常驻 chunk 的账目。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Entry {
  /// 该 chunk 在 GPU 上占的字节（含块内余量）。
  bytes: usize,
  /// 当前档位（`keep_extent`；`BRICK_FACTOR` = 全分辨率）。
  level: Level,
  /// 钉住到这一帧（含）：此前不许换出 / 降级。
  pinned_until: u64,
  /// 常驻起始帧（最小驻留判据）。
  since: u64,
}

/// 常驻策略参数。
#[derive(Debug, Clone, Copy)]
pub struct ResidencyPolicy {
  /// 常驻预算（字节）。**0 = 不限** ⇒ 只按档位阶梯走，不做整块丢弃。
  pub budget_bytes: usize,
  /// 最小驻留帧数：刚安装的 chunk 在此期间不许换出。
  pub min_resident_frames: u64,
  /// 编辑钉住帧数：被编辑过的 chunk 在此期间不许换出 / 降级。
  pub pin_frames: u64,
  /// 每帧安装上限（唤醒 + 重装合计）：安装是 0.9–2.4 ms/chunk 的一次性开销，必须摊到多帧。
  ///
  /// 取 8 而不是 2：**每帧上限只在"有尖峰要摊"时才有意义**。池空时（冷启动 / 大跨度移动）没有尖峰可摊，
  /// 摊的代价是整屏空洞要 `池容量 / 上限 / 帧率` 才补完 —— 实测上限 2 时 2048 块要 **17 s**（而全部安装
  /// 的 CPU 总账只有 `2048 × 1.65 ms ≈ 3.4 s`）。8/帧 把这一段压到 ~4 s（≈ 底线），代价是这 4 s 里
  /// 每帧多 6 × 1.65 ≈ 10 ms；池满之后需要安装的块是"新进视野的那几个"，自然不会每帧 8 个。
  pub max_install_per_frame: usize,
}

impl ResidencyPolicy {
  /// 默认：**预算不限**（只走档位阶梯，不整块丢弃）、每帧最多装 8 个。
  pub const DEFAULT: Self =
    Self { budget_bytes: 0, min_resident_frames: 30, pin_frames: 120, max_install_per_frame: 8 };

  /// 预算不限。
  pub fn budget_is_off(&self) -> bool {
    self.budget_bytes == 0
  }
}

/// 本帧的常驻动作（都由调用方落实到 builder）。
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ResidencyPlan {
  /// 要**安装**的 `(chunk, 档位)`：新唤醒 + 档位变化（降级 / 升级），按到相机距离升序。
  ///
  /// **两段**（见 [`BOOTSTRAP_LEVEL`]）：首次安装按引导档（粗、便宜）且**不占**每帧名额；档位变化（细化
  /// / 粗化）才受 `max_install_per_frame` 截断。已在目标档位的不出现。
  pub install: Vec<(ChunkCoord, Level)>,
  /// 要**换出**的 chunk（LRU 序：最久没用到的在前）。
  pub evict: Vec<ChunkCoord>,
}

/// 常驻账目 + 决策。**不持有任何 GPU 资源**。
#[derive(Debug, Default, Clone)]
pub struct Residency {
  entries: HashMap<ChunkCoord, Entry>,
  /// [`Self::resident_bytes`] 的**增量账**。`entries` 的全部写点只有
  /// [`Self::note_resident`] / [`Self::note_gone`] / [`Self::note_bytes`] 三处 ⇒ 在这三处按差值维护。
  ///
  /// WHY 不写成 `entries.values().map(|e| e.bytes).sum()`：它在 `plan` → `pick_evicts` 里**每帧**
  /// 被读到，而常驻可达 1–2 万块 ⇒ 每帧一次全表求和的白账（与 `recent_top` 的"只选不排"同一类取舍，
  /// 见 `docs/editable-gigavoxel.md` §10.4）。加这个字段后它是 O(1)。
  bytes_total: usize,
  frame: u64,
}

impl Residency {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn frame(&self) -> u64 {
    self.frame
  }

  /// 推进帧号（调用方每帧一次）。
  pub fn tick(&mut self, frame: u64) {
    self.frame = frame;
  }

  pub fn resident_count(&self) -> usize {
    self.entries.len()
  }

  pub fn resident_bytes(&self) -> usize {
    self.bytes_total
  }

  pub fn is_resident(&self, c: ChunkCoord) -> bool {
    self.entries.contains_key(&c)
  }

  /// 当前档位；None = 未常驻。
  pub fn resident_level(&self, c: ChunkCoord) -> Option<Level> {
    self.entries.get(&c).map(|e| e.level)
  }

  /// 报告"该 chunk 现在常驻（这个档位 / 这些字节）"（由 builder 的实际状态或刚做的安装同步过来）。
  pub fn note_resident(&mut self, c: ChunkCoord, bytes: usize, level: Level, frame: u64) {
    match self.entries.get_mut(&c) {
      Some(e) => {
        self.bytes_total = self.bytes_total.saturating_sub(e.bytes) + bytes;
        e.bytes = bytes;
        e.level = level;
      }
      None => {
        self.bytes_total += bytes;
        self.entries.insert(c, Entry { bytes, level, pinned_until: 0, since: frame });
      }
    }
  }

  /// 报告"该 chunk 已不在 GPU 上"。
  pub fn note_gone(&mut self, c: ChunkCoord) {
    if let Some(e) = self.entries.remove(&c) {
      self.bytes_total = self.bytes_total.saturating_sub(e.bytes);
    }
  }

  /// 只刷新字节（档位不动）：安装 / 编辑后块内余量会变，常驻预算按实际块算。
  pub fn note_bytes(&mut self, c: ChunkCoord, bytes: usize) {
    if let Some(e) = self.entries.get_mut(&c) {
      self.bytes_total = self.bytes_total.saturating_sub(e.bytes) + bytes;
      e.bytes = bytes;
    }
  }

  /// 该 chunk 被编辑 ⇒ 钉住到 `frame + pin_frames`（编辑优先于流式）。
  pub fn note_edit(&mut self, c: ChunkCoord, frame: u64, pin_frames: u64) {
    if let Some(e) = self.entries.get_mut(&c) {
      e.pinned_until = e.pinned_until.max(frame + pin_frames);
    }
  }

  /// 离相机最近的 `n` 个常驻 chunk（升序）。`n ≥ 候选数` 时返回全部。
  ///
  /// WHY 不能拿"CPU 里全量 chunk"当需求：那等于"常驻 = CPU 里恰好有的东西"，预算一开就变成
  /// **换出 → 下一帧又被想要 → 又装回来**的抖振（每帧 `max_install_per_frame` 全烧在装卸同一批上）。
  /// 调用方按池容量 `n` 截断 ⇒ 常驻收敛成一个稳定的滚动窗口（相机附近的进来、最远的被换出）。
  ///
  /// WHY 只选不排：常驻 1–2 万块时全排序是 O(n log n) 的每帧白账（`docs/editable-gigavoxel.md`
  /// §10.4）。截断只需"前 `n` 名"⇒ `select_nth_unstable` 就够；`n ≥ 候选数` 时连选都不用
  /// （消费端只拿它求档位，`plan` 自己会按距离重排）。
  pub fn nearest_top(&self, camera_chunk: IVec3, n: usize) -> Vec<ChunkCoord> {
    let mut v: Vec<ChunkCoord> = self.entries.keys().copied().collect();
    let key = |c: &ChunkCoord| (chunk_distance(camera_chunk, c.0), c.0.x, c.0.y, c.0.z);
    if n >= v.len() {
      // 不截断 ⇒ 顺序无人消费（调用方只遍历，`plan` 自己按距离重排）⇒ 省掉这次排序。
      return v;
    }
    v.select_nth_unstable_by(n, |a, b| key(a).cmp(&key(b)));
    v.truncate(n);
    v.sort_unstable_by_key(key);
    v
  }

  /// 决策。`wants` = 需求集：`(chunk, 想要的档位)`（调用方用 [`want_level`] 按距离算，可能含未常驻的）。
  /// `must_keep` = 本帧必须保留的（正在上传 / 正在编辑）。
  pub fn plan(
    &self,
    policy: &ResidencyPolicy,
    camera_chunk: IVec3,
    wants: impl Iterator<Item = (ChunkCoord, Level)>,
    must_keep: &HashSet<ChunkCoord>,
  ) -> ResidencyPlan {
    let mut install: Vec<(ChunkCoord, Level)> = wants
      .filter(|&(c, level)| {
        let cur = self.resident_level(c);
        if cur == Some(level) {
          return false; // 已在目标档位
        }
        // 钉住的 chunk 不许降级（升级不受限：更细总是正确的方向）。
        let downgrade = cur.is_some_and(|l| l < level);
        let pinned = self.entries.get(&c).is_some_and(|e| e.pinned_until >= self.frame);
        !(downgrade && pinned)
      })
      .collect();
    install.sort_by_key(|&(c, level)| {
      (dist_of(camera_chunk, c), std::cmp::Reverse(level), c.0.x, c.0.y, c.0.z)
    });
    // ---- 两段安装（渐进加载）----
    //   · **首次安装**（`cur == None`）⇒ 按 [`BOOTSTRAP_LEVEL`] 粗档引导。它便宜 ⇒ **不占**每帧名额
    //     （名额留给下面那类），但另有 [`BOOTSTRAP_PER_FRAME`] 防冷启动尖峰。
    //   · **档位变化**（`cur == Some`）= 细化 / 粗化 ⇒ 走 `max_install_per_frame`。
    //
    // CONSTRAINT: 顺序仍是"按相机距离升序"（`install` 已排好），两段只是**名额分配**不同，不重排 ——
    // 近了先装、先细，这条不能因为分段而破。
    let mut picked: Vec<(ChunkCoord, Level)> = Vec::with_capacity(install.len());
    let mut refine_budget = policy.max_install_per_frame;
    let mut boot_budget = BOOTSTRAP_PER_FRAME;
    for (c, level) in install {
      if self.resident_level(c).is_none() {
        if boot_budget == 0 {
          continue;
        }
        boot_budget -= 1;
        picked.push((c, level.max(BOOTSTRAP_LEVEL)));
      } else if refine_budget > 0 {
        refine_budget -= 1;
        picked.push((c, level));
      }
    }
    let install = picked;

    let evict = self.pick_evicts(policy, camera_chunk, must_keep);
    ResidencyPlan { install, evict }
  }

  /// 换出名单：只在**超预算**时给；**离相机最远的先走**，跳过（钉住 / 未满最小驻留 / `must_keep`）的。
  fn pick_evicts(
    &self,
    policy: &ResidencyPolicy,
    camera_chunk: IVec3,
    must_keep: &HashSet<ChunkCoord>,
  ) -> Vec<ChunkCoord> {
    if policy.budget_is_off() {
      return Vec::new();
    }
    let mut over = self.resident_bytes().saturating_sub(policy.budget_bytes);
    if over == 0 {
      return Vec::new();
    }
    let mut order: Vec<(&ChunkCoord, &Entry)> = self.entries.iter().collect();
    order.sort_by_key(|(c, _)| {
      (std::cmp::Reverse(chunk_distance(camera_chunk, c.0)), c.0.x, c.0.y, c.0.z)
    });
    let mut out = Vec::new();
    for (c, e) in order {
      if over == 0 {
        break;
      }
      if must_keep.contains(c) || e.pinned_until >= self.frame {
        continue;
      }
      if self.frame.saturating_sub(e.since) < policy.min_resident_frames {
        continue;
      }
      over = over.saturating_sub(e.bytes);
      out.push(*c);
    }
    out
  }
}

/// chunk 间的切比雪夫距离（y 同权：世界是 3D 的）。
pub fn chunk_distance(a: IVec3, b: IVec3) -> i32 {
  let d = (a - b).abs();
  d.x.max(d.y).max(d.z)
}

fn dist_of(a: IVec3, b: ChunkCoord) -> i32 {
  chunk_distance(a, b.0)
}

#[cfg(test)]
mod tests {
  use super::*;

  const P: ResidencyPolicy = ResidencyPolicy {
    budget_bytes: 300,
    min_resident_frames: 10,
    pin_frames: 100,
    max_install_per_frame: 2,
  };
  /// 720p 的单像素角大小（`2·tan(30°)/720`）——下面所有距离都按它算 fp。
  const PX_ANG: f32 = 1.604e-3;

  fn cc(x: i32, y: i32, z: i32) -> ChunkCoord {
    ChunkCoord(IVec3::new(x, y, z))
  }

  /// 档位阶梯：`fp ≥ 阈值` 才允许那一档；迟滞带内保持原档。
  #[test]
  fn ladder_thresholds_and_hysteresis() {
    let full = BRICK_FACTOR;
    assert_eq!(want_level(1000.0, PX_ANG, full), full, "近处（fp≈1.6 < 4）必须全分辨率");
    assert_eq!(want_level(4.0 / PX_ANG, PX_ANG, full), 16, "fp = 4 ⇒ 16³ 档");
    assert_eq!(want_level(16.0 / PX_ANG, PX_ANG, full), 64, "fp = 16 ⇒ 64³ 档");
    assert_eq!(want_level(64.0 / PX_ANG, PX_ANG, full), CHUNK_SIZE, "fp = 64 ⇒ 整 chunk 档");
    // 迟滞：已在 16³ 档时，fp 落到 3.2（> 4×0.75）仍留在 16³；落到 2.0 才回全分辨率
    assert_eq!(want_level(3.2 / PX_ANG, PX_ANG, 16), 16, "迟滞带内保持 16³");
    assert_eq!(want_level(2.0 / PX_ANG, PX_ANG, 16), full, "退出迟滞带 ⇒ 回全分辨率");
  }

  /// 预算不限 ⇒ 只按档位走（不换出）。
  #[test]
  fn budget_off_never_evicts() {
    let mut r = Residency::new();
    r.tick(100);
    for i in 0..8 {
      r.note_resident(cc(i, 0, 0), 100, BRICK_FACTOR, 0);
    }
    let off = ResidencyPolicy { budget_bytes: 0, ..P };
    let plan = r.plan(&off, IVec3::ZERO, std::iter::empty(), &HashSet::new());
    assert_eq!(plan, ResidencyPlan::default());
    assert_eq!(r.resident_bytes(), 800);
  }

  /// `resident_bytes` 是**增量账** ⇒ `entries` 的三个写点（`note_resident` / `note_bytes` / `note_gone`）
  /// 都必须维护它。漏掉任何一处，`pick_evicts` 的"超预算多少"就会算错（表现为换出永远不够、
  /// 或永远在换），而那是**静默**的 —— 所以在本测试里逐个写点钉住。
  #[test]
  fn resident_bytes_tracks_all_write_points() {
    let mut r = Residency::new();
    assert_eq!(r.resident_bytes(), 0);
    r.note_resident(cc(0, 0, 0), 100, BRICK_FACTOR, 0);
    r.note_resident(cc(1, 0, 0), 200, BRICK_FACTOR, 0);
    assert_eq!(r.resident_bytes(), 300, "两个新块");
    // 重复上报同一个 chunk：按**差值**替换，不是累加
    r.note_resident(cc(0, 0, 0), 150, BRICK_FACTOR, 0);
    assert_eq!(r.resident_bytes(), 350, "重复上报按差值替换");
    // 只刷字节（档位不动）
    r.note_bytes(cc(1, 0, 0), 50);
    assert_eq!(r.resident_bytes(), 200, "note_bytes 按差值替换");
    // 未常驻的 note_bytes 是 no-op
    r.note_bytes(cc(9, 9, 9), 999);
    assert_eq!(r.resident_bytes(), 200, "未常驻的 note_bytes 不改账");
    // 走了就减
    r.note_gone(cc(0, 0, 0));
    assert_eq!(r.resident_bytes(), 50, "note_gone 减掉该块的字节");
    r.note_gone(cc(9, 9, 9));
    assert_eq!(r.resident_bytes(), 50, "本来就不在的 note_gone 不减");
  }

  /// 档位不同就要重装（降级 / 升级都算），已在目标档位的不出现。
  #[test]
  fn level_change_means_reinstall() {
    let mut r = Residency::new();
    r.tick(100);
    r.note_resident(cc(0, 0, 0), 100, BRICK_FACTOR, 0);
    r.note_resident(cc(1, 0, 0), 100, 16, 0);
    let wants = [(cc(0, 0, 0), 16), (cc(1, 0, 0), 16), (cc(2, 0, 0), 16)];
    let plan = r.plan(&P, IVec3::ZERO, wants.into_iter(), &HashSet::new());
    assert_eq!(
      plan.install,
      vec![(cc(0, 0, 0), 16), (cc(2, 0, 0), 16)],
      "距离升序（近的先装），档位变了的要重装；已在 16³ 的 cc(1) 不出现"
    );
  }

  /// 钉住的 chunk 不许降级（升级不受限）；`must_keep` 不许换出。
  #[test]
  fn pinned_not_downgraded() {
    let mut r = Residency::new();
    r.tick(100);
    r.note_resident(cc(0, 0, 0), 100, BRICK_FACTOR, 0);
    r.note_edit(cc(0, 0, 0), 100, P.pin_frames);
    let plan = r.plan(&P, IVec3::ZERO, [(cc(0, 0, 0), 16)].into_iter(), &HashSet::new());
    assert!(plan.install.is_empty(), "钉住的 chunk 不许被降级");
    // 升级（16 → 全分辨率）应当允许
    let up = r.plan(&P, IVec3::ZERO, [(cc(0, 0, 0), BRICK_FACTOR)].into_iter(), &HashSet::new());
    assert!(up.install.is_empty(), "已在该档 ⇒ 无事发生");
  }

  /// 超预算 ⇒ 换出**离相机最远的**，且跳过钉住 / 刚驻留 / `must_keep`。
  #[test]
  fn over_budget_evicts_farthest_first() {
    let mut r = Residency::new();
    r.tick(100);
    for x in -20..=-16 {
      r.note_resident(cc(x, 0, 0), 100, BRICK_FACTOR, 0);
    }
    let plan = r.plan(&P, IVec3::ZERO, std::iter::empty(), &HashSet::new());
    assert_eq!(
      plan.evict,
      vec![cc(-20, 0, 0), cc(-19, 0, 0)],
      "500 字节超预算 200 ⇒ 淘汰离相机最远的两个"
    );
  }

  /// `nearest_top`：截断时给出**最近的 `n` 个**（距离升序）；不截断时只保证集合正确（顺序无消费者）。
  #[test]
  fn nearest_top_is_distance_ordered() {
    let mut r = Residency::new();
    r.tick(100);
    for x in [-5, 2, -1, 9, 3] {
      r.note_resident(cc(x, 0, 0), 100, BRICK_FACTOR, 0);
    }
    let mut all = r.nearest_top(IVec3::ZERO, usize::MAX);
    all.sort_by_key(|c| c.0.x);
    assert_eq!(
      all,
      vec![cc(-5, 0, 0), cc(-1, 0, 0), cc(2, 0, 0), cc(3, 0, 0), cc(9, 0, 0)],
      "全集"
    );
    assert_eq!(
      r.nearest_top(IVec3::ZERO, 2),
      vec![cc(-1, 0, 0), cc(2, 0, 0)],
      "截断到最近的 2 个（距离升序）"
    );
  }

  /// 钉住与刚驻留的都不许换出。
  #[test]
  fn pinned_and_fresh_survive_eviction() {
    let mut r = Residency::new();
    r.tick(100);
    r.note_resident(cc(-20, 0, 0), 200, BRICK_FACTOR, 0);
    r.note_resident(cc(-21, 0, 0), 200, BRICK_FACTOR, 5);
    r.note_resident(cc(-22, 0, 0), 200, BRICK_FACTOR, 95); // 只驻留 5 帧 < min_resident_frames
    r.note_edit(cc(-20, 0, 0), 100, P.pin_frames);
    let plan = r.plan(&P, IVec3::ZERO, std::iter::empty(), &HashSet::new());
    assert_eq!(plan.evict, vec![cc(-21, 0, 0)], "钉住与刚驻留的必须都跳过");
  }

  /// `must_keep` 优先于一切。
  #[test]
  fn must_keep_wins() {
    let mut r = Residency::new();
    r.tick(100);
    r.note_resident(cc(-20, 0, 0), 400, BRICK_FACTOR, 0);
    let keep: HashSet<ChunkCoord> = [cc(-20, 0, 0)].into_iter().collect();
    let plan = r.plan(&P, IVec3::ZERO, std::iter::empty(), &keep);
    assert!(plan.evict.is_empty());
  }

  /// **细化**（档位变化）仍受每帧上限约束，且近的优先。
  ///
  /// 首次安装**不**受这个上限（它走粗档引导、便宜 —— 见 [`first_install_bootstraps_without_spending_refine_budget`]），
  /// 所以这里用三个**已常驻的粗档块**构造"纯细化"的场景。
  #[test]
  fn refinements_are_capped_and_nearest_first() {
    let mut r = Residency::new();
    r.tick(100);
    for x in [2, 5, 9] {
      r.note_resident(cc(x, 0, 0), 100, CHUNK_SIZE, 0);
    }
    let wants = [(cc(9, 0, 0), 16), (cc(2, 0, 0), 16), (cc(5, 0, 0), 16)];
    let plan = r.plan(&P, IVec3::ZERO, wants.into_iter(), &HashSet::new());
    assert_eq!(plan.install, vec![(cc(2, 0, 0), 16), (cc(5, 0, 0), 16)], "距离升序 + 上限 2");
  }

  /// 距离 = 切比雪夫（含 y）。
  #[test]
  fn distance_is_chebyshev() {
    assert_eq!(chunk_distance(IVec3::new(0, 0, 0), IVec3::new(3, -5, 1)), 5);
  }

  /// **两段安装（渐进加载）**：首次安装按 [`BOOTSTRAP_LEVEL`] 粗档引导、且**不占**每帧名额
  /// （名额留给细化）；细化仍受 `max_install_per_frame` 截断；两段都不打乱"按相机距离升序"。
  #[test]
  fn first_install_bootstraps_without_spending_refine_budget() {
    let mut r = Residency::new();
    r.tick(100);
    // 两个已常驻的**粗档**块（想细化到全分辨率）
    r.note_resident(cc(0, 0, 0), 100, CHUNK_SIZE, 0);
    r.note_resident(cc(1, 0, 0), 100, CHUNK_SIZE, 0);
    let wants = [
      (cc(0, 0, 0), BRICK_FACTOR), // 细化
      (cc(1, 0, 0), BRICK_FACTOR), // 细化
      (cc(2, 0, 0), BRICK_FACTOR), // 首次安装
      (cc(3, 0, 0), BRICK_FACTOR), // 首次安装
      (cc(4, 0, 0), BRICK_FACTOR), // 首次安装
    ];
    let plan = r.plan(&P, IVec3::ZERO, wants.into_iter(), &HashSet::new());
    assert_eq!(plan.install.len(), 5, "2 个细化（= 每帧名额）+ 3 个引导，都在计划里");
    for x in 2..=4 {
      let e = plan.install.iter().find(|(c, _)| c.0.x == x).expect("首次安装应在计划里");
      assert_eq!(e.1, BOOTSTRAP_LEVEL, "首次安装必须走粗档引导（便宜 ⇒ 不占名额）");
    }
    for x in 0..=1 {
      let e = plan.install.iter().find(|(c, _)| c.0.x == x).expect("细化应在计划里");
      assert_eq!(e.1, BRICK_FACTOR, "细化到目标档");
    }
    let order: Vec<i32> = plan.install.iter().map(|(c, _)| c.0.x).collect();
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(order, sorted, "分段不能打乱「按相机距离升序」");
  }

  /// 引导档只会让首次安装**更粗**，绝不会把远端本来就该粗的块变细。
  #[test]
  fn bootstrap_never_refines_a_deliberately_coarse_chunk() {
    let mut r = Residency::new();
    r.tick(100);
    let plan = r.plan(&P, IVec3::ZERO, [(cc(0, 0, 0), CHUNK_SIZE)].into_iter(), &HashSet::new());
    assert_eq!(plan.install, vec![(cc(0, 0, 0), CHUNK_SIZE)], "远端整 chunk 档不受引导影响");
  }
}
