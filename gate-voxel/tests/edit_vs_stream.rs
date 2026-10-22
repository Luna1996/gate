//! 「编辑」与「流式装载」的区分：**编辑会记编辑 AABB，流式装载不会**。
//!
//! GI 的时域历史作废判据就建立在这条区分上（`brickmap::upload::BrickMapDirty::edited`）：
//! 编辑会改变光照（自发光体素出现/消失 ⇒ 周围面的间接光变了）⇒ 历史必须作废；
//! 流式装载只是补内容 ⇒ 历史必须保留，否则流式世界里 GI 永远收敛不了。
//!
//! 这条测试锁住那个判据本身：哪天 `mount_chunk_tree` 也开始记编辑 AABB，
//! 或者编辑不再记 ⇒ GI 就会退回到"相机不动就永远不更新"的老毛病。
//! 同理，"物体消失"（`clear_all_chunks`）也必须记 —— 它同样改变了光照。

use gate_voxel::{ChunkCoord, PaletteId, VolumeGrid};
use glam::IVec3;

fn chunk_of(v: IVec3) -> ChunkCoord {
  ChunkCoord(v.div_euclid(IVec3::splat(gate_voxel::CHUNK_SIZE)))
}

#[test]
fn edit_records_aabb_but_stream_mount_does_not() {
  let mut grid = VolumeGrid::new();

  // ① 真编辑（笔刷落在 chunk 0 的一格上）⇒ 记下编辑 AABB。
  let voxel = IVec3::new(3, 7, 11);
  assert!(grid.set_voxel_ivec3(voxel, PaletteId::from(1)).is_some());
  let cc = chunk_of(voxel);
  let (lo, hi) = grid.take_edit_aabb(cc).expect("编辑必须记下编辑 AABB");
  assert!(lo.cmple(voxel).all() && hi.cmpgt(voxel).all(), "编辑 AABB 应覆盖被改的那一格：{lo:?}..{hi:?}");

  // ② AABB 是一次性的（取走即清）⇒ 同一 chunk 再取应是 None（不重复触发 GI 作废）。
  assert!(grid.take_edit_aabb(cc).is_none(), "编辑 AABB 必须是一次性的");

  // ③ 流式装载（`mount_chunk_tree`）⇒ **不**记编辑 AABB。
  let mut other = VolumeGrid::new();
  let cc2 = ChunkCoord(IVec3::new(5, 0, 0));
  other.mount_chunk_tree(cc2, gate_voxel::ChunkTree::uniform(PaletteId::from(2)), 0);
  assert!(
    other.take_edit_aabb(cc2).is_none(),
    "流式装载不得记编辑 AABB —— 否则每帧装新块都会作废 GI 历史（GI 永不收敛）"
  );

  // ④ 物体被清空（消失）⇒ **必须**记编辑 AABB：
  // 几何真的没了 ⇒ 它投出的光也没了 ⇒ GI 的时域历史必须作废。
  // 少了这条，"物体消失"就不会触发 GI 更新（光斑还留在原地）。
  let mut gone = VolumeGrid::new();
  let cc3 = ChunkCoord(IVec3::new(1, 2, 3));
  gone.mount_chunk_tree(cc3, gate_voxel::ChunkTree::uniform(PaletteId::from(3)), 0);
  gone.clear_all_chunks();
  assert!(
    gone.take_edit_aabb(cc3).is_some(),
    "清空 chunk（物体消失）必须记编辑 AABB —— 否则 GI 不知道光没了"
  );
}
