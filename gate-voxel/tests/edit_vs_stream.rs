use gate_voxel::{ChunkCoord, PaletteId, VolumeGrid};
use glam::IVec3;

fn chunk_of(v: IVec3) -> ChunkCoord {
  ChunkCoord(v.div_euclid(IVec3::splat(gate_voxel::CHUNK_SIZE)))
}