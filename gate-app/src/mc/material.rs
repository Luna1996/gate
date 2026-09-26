//! **逐 texel 颜色 → 调色板槽**：worker 线程按需认领，主线程负责把条目装进各 volume 的调色板。
//!
//! 为什么不能"每个方块族预留一段槽"：一条 MC 方块贴图是 16×16（最多 256 色），本仓贴图目录里有
//! 1021 张 ⇒ 逐 texel 上色最多出现 26 万个不同颜色，而调色板一共只有 65536 槽（`PALETTE_ENTRY_COUNT`）。
//! 所以按**内容去重**认领（同一张贴图里的同色只占一格，跨贴图的同色也共用），并留一条溢出回退：
//! 槽用满后按 8 级/通道量化找最近色 —— 颜色近似而不再新增槽（地图上通常是几个像素的偏差）。
//!
//! **调用契约**（M5 的生产管线）：`intern` 会被多个 worker 并发调用 ⇒ 句柄表用 `Mutex`；
//! 认领只增不减，且**日志只增不减**（[`Pool::log_from`]）：主线程按游标补装，换世界 / 重载时
//! 从头重放整份日志即可让新 volume 的调色板与 worker 的槽号一致。

use std::collections::HashMap;
use std::sync::Mutex;

use gate_voxel::{PaletteEntry, PaletteId};

/// 可用槽位上限（0 号保留给空气）
const SLOT_LIMIT: u16 = 65_000;

#[derive(Default)]
struct Inner {
  /// 条目 → 槽号（按 8 B 内容去重）
  map: HashMap<PaletteEntry, u16>,
  /// 槽号 → 条目，按认领顺序（主线程的重放来源）
  log: Vec<(PaletteId, PaletteEntry)>,
  next: u16,
  /// 溢出后的量化回退表（量化键 → 已挑好的槽），免得每来一个新颜色都线性扫一遍日志
  quant: HashMap<[u8; 3], u16>,
  /// 溢出次数（诊断：真地图上正常应为 0）
  overflow: usize,
}

/// 共享的槽位分配器（`Arc` 给 worker 与主线程共用）
pub struct Pool {
  inner: Mutex<Inner>,
  /// 每通道保留的高位数：`8` = 原样，`5` = 32 级。
  ///
  /// 这一刀是为了**内存与产出量**，不是为了省槽：MC 的方块贴图是噪点像素画（`stone` 的灰在
  /// 127..137 之间抖），逐 texel 原样上色会让几乎**每一个** `4³` 砖都"不同色"⇒ 每个砖都要一张
  /// 值表（128 B）⇒ 一个 section 的 CPU 树涨到 30 MB 级、产出一层 120 ms（实测出生点满实体层）。
  /// 5 位量化把 ±4 的抖动并成同色 ⇒ 大片区域整砖同色（退化成 uniform 节点），视觉上只是
  /// 32 级色阶（方块贴图本身就没有连续渐变）。见 `docs/mc_map.md` §6。
  bits: u8,
}

impl Default for Pool {
  fn default() -> Self {
    Self::new()
  }
}

impl Pool {
  pub fn new() -> Self {
    Self::quantized(5)
  }

  /// 指定每通道保留的高位数（`8` = 原样；越小越省内存、越平）
  pub fn quantized(bits: u8) -> Self {
    Self {
      inner: Mutex::new(Inner { next: 1, ..Default::default() }),
      bits: bits.clamp(3, 8),
    }
  }

  /// 认领一个槽位（内容相同 → 同一个槽）。溢出后走量化回退，仍返回**可用**的槽号。
  pub fn intern(&self, e: PaletteEntry) -> PaletteId {
    let e = PaletteEntry { color: quant(e.color, self.bits), ..e };
    let mut g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(i) = g.map.get(&e) {
      return PaletteId(*i);
    }
    if g.next < SLOT_LIMIT {
      let id = g.next;
      g.next += 1;
      g.map.insert(e, id);
      g.log.push((PaletteId(id), e));
      return PaletteId(id);
    }
    // 溢出：再降到 3 位/通道，同量化键只扫一次日志挑最近色
    let q = quantize(e.color);
    if let Some(i) = g.quant.get(&q) {
      return PaletteId(*i);
    }
    let best = g
      .log
      .iter()
      .min_by_key(|(_, o)| color_dist2(o.color, e.color))
      .map(|(i, _)| *i)
      .unwrap_or(PaletteId(1));
    g.quant.insert(q, best.0);
    g.overflow += 1;
    PaletteId(best.0)
  }

  /// 日志的 `[from..]` 段（空 = 没有新条目）。`from = 0` = 重放整份。
  pub fn log_from(&self, from: usize) -> Vec<(PaletteId, PaletteEntry)> {
    let g = self.inner.lock().unwrap_or_else(|e| e.into_inner());
    if from >= g.log.len() {
      return Vec::new();
    }
    g.log[from..].to_vec()
  }

  /// 已认领的槽数
  pub fn len(&self) -> usize {
    self.inner.lock().unwrap_or_else(|e| e.into_inner()).log.len()
  }

  /// 溢出到量化回退的次数（诊断：真地图上正常应为 0）
  pub fn overflow_count(&self) -> usize {
    self.inner.lock().unwrap_or_else(|e| e.into_inner()).overflow
  }
}

/// 量化键：每通道保留高 3 位
fn quantize(c: [u8; 3]) -> [u8; 3] {
  [c[0] >> 5, c[1] >> 5, c[2] >> 5]
}

/// 每通道只保留高 `bits` 位（低位清零；`bits >= 8` = 原样）
fn quant(c: [u8; 3], bits: u8) -> [u8; 3] {
  if bits >= 8 {
    return c;
  }
  let mask = 0xFFu8 << (8 - bits);
  [c[0] & mask, c[1] & mask, c[2] & mask]
}

fn color_dist2(a: [u8; 3], b: [u8; 3]) -> u32 {
  let d = |x: u8, y: u8| {
    let v = x as i32 - y as i32;
    (v * v) as u32
  };
  d(a[0], b[0]) + d(a[1], b[1]) + d(a[2], b[2])
}

#[cfg(test)]
mod tests {
  use super::*;

  fn e(color: [u8; 3]) -> PaletteEntry {
    PaletteEntry { color, ..Default::default() }
  }

  #[test]
  fn same_content_shares_one_slot() {
    let p = Pool::new();
    let a = p.intern(e([1, 2, 3]));
    let b = p.intern(e([1, 2, 3]));
    let c = p.intern(e([9, 9, 9]));
    assert_eq!(a, b, "内容相同必须同槽");
    assert_ne!(a, c);
    assert_eq!(p.len(), 2);
    assert_eq!(p.log_from(0).len(), 2);
    assert!(p.log_from(2).is_empty(), "游标到底后不该再吐条目");
  }

  /// 溢出回退：槽用满后不再新增槽，但仍返回可用的旧槽（颜色近似）
  #[test]
  fn overflow_falls_back_to_nearest_without_new_slots() {
    let p = Pool::new();
    // 直接把 next 顶到上限附近（走真实 API 会认领 6.5 万次，没必要）
    {
      let mut g = p.inner.lock().unwrap();
      let near = SLOT_LIMIT - 1;
      g.log.push((PaletteId(near), e([255, 0, 0])));
      g.map.insert(e([255, 0, 0]), near);
      g.next = SLOT_LIMIT;
    }
    let n = p.len();
    let got = p.intern(e([250, 0, 0]));
    assert_eq!(got, PaletteId(SLOT_LIMIT - 1), "应回退到最近的已认领槽");
    assert_eq!(p.len(), n, "溢出不新增槽");
    assert_eq!(p.overflow_count(), 1);
    // 同一个量化键第二次不再扫日志
    assert_eq!(p.intern(e([251, 0, 0])), got);
    assert_eq!(p.overflow_count(), 1);
  }
}
