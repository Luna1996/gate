use gate_voxel::{ChunkCoord, PaletteId, VolumeGrid};
use glam::IVec3;

fn chunk_of(v: IVec3) -> ChunkCoord {
  ChunkCoord(v.div_euclid(IVec3::splat(gate_voxel::CHUNK_SIZE)))
}

#[test]
fn edit_records_aabb_but_stream_mount_does_not() {
  let mut grid = VolumeGrid::new();

  let voxel = IVec3::new(3, 7, 11);
  assert!(grid.set_voxel_ivec3(voxel, PaletteId::from(1)).is_some());
  let cc = chunk_of(voxel);
  let (lo, hi) = grid.take_edit_aabb(cc).expect("编辑必须记下编辑 AABB");
  assert!(
    lo.cmple(voxel).all() && hi.cmpgt(voxel).all(),
    "编辑 AABB 应覆盖被改的那一格：{lo:?}..{hi:?}"
  );

  assert!(grid.take_edit_aabb(cc).is_none(), "编辑 AABB 必须是一次性的");

  let mut other = VolumeGrid::new();
  let cc2 = ChunkCoord(IVec3::new(5, 0, 0));
  other.mount_chunk_tree(cc2, gate_voxel::ChunkTree::uniform(PaletteId::from(2)), 0);
  assert!(
    other.take_edit_aabb(cc2).is_none(),
    "流式装载不得记编辑 AABB —— 否则每帧装新块都会作废 GI 历史（GI 永不收敛）"
  );

  let mut gone = VolumeGrid::new();
  let cc3 = ChunkCoord(IVec3::new(1, 2, 3));
  gone.mount_chunk_tree(cc3, gate_voxel::ChunkTree::uniform(PaletteId::from(3)), 0);
  gone.clear_all_chunks();
  assert!(
    gone.take_edit_aabb(cc3).is_some(),
    "清空 chunk（物体消失）必须记编辑 AABB —— 否则 GI 不知道光没了"
  );
}
