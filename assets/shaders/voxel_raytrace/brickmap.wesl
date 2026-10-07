import package::bindings::{b_struct, b_struct_p1, g};
import package::common::{CHUNK_INDEX_CAP, CHUNK_SIZE, INDEX_ENTRY_EMPTY, MAX_LEVEL, NODE_FIXED_WORDS};

const SEG_WORDS: u32 = 536870912u;

fn ts(addr: u32) -> u32 {
  if (addr >= SEG_WORDS) { return b_struct_p1[addr - SEG_WORDS]; }
  return b_struct[addr];
}

struct Grid {
  w_mn: vec3<f32>,                w_mx: vec3<f32>,                l_min: vec3<f32>,               l_max: vec3<f32>,               max_chunk_steps: u32,           col0: vec3<f32>,                col1: vec3<f32>,
  col2: vec3<f32>,
  pos: vec3<f32>,                 scale: f32,
  tree_base: u32,                 palette_base: u32,              index_origin: vec3<i32>,        index_dims: vec3<u32>,          obj_id: i32,                    vol: u32,                       coverage_r: f32,              }

fn sample_brickmap(g: Grid, voxel: vec3<i32>) -> u32 {
    let m = ((voxel % vec3<i32>(i32(CHUNK_SIZE))) + vec3<i32>(i32(CHUNK_SIZE))) % vec3<i32>(i32(CHUNK_SIZE));
  let chunk_i = (voxel - m) / vec3<i32>(i32(CHUNK_SIZE));
  let origin = g.index_origin;
  let dims = g.index_dims;
  let rel = chunk_i - origin;
  if (any(rel < vec3<i32>(0))) { return 0u; }
  let rel_u = vec3<u32>(rel);
  if (any(rel_u >= dims)) { return 0u; }
    let index_addr = g.tree_base + rel_u.x + rel_u.y * CHUNK_INDEX_CAP + rel_u.z * (CHUNK_INDEX_CAP * CHUNK_INDEX_CAP);
  let entry = ts(index_addr);
  if (entry == 0u || entry == INDEX_ENTRY_EMPTY) { return 0u; }
    let chunk_base = g.tree_base + entry - 1u;

    let local = vec3<u32>(m);    var node_addr = chunk_base;
  for (var level = 0u; level < MAX_LEVEL; level = level + 1u) {
    let mask_lo = ts(node_addr);
    let mask_hi = ts(node_addr + 1u);
    let palette = ts(node_addr + 2u) & 0xFFFFu;  
    let shift = 8u - (level + 1u) * 2u;
    let cx = (local.x >> shift) & 3u;
    let cy = (local.y >> shift) & 3u;
    let cz = (local.z >> shift) & 3u;
    let child_idx = cz * 16u + cy * 4u + cx;

    var mask_word: u32 = mask_lo;
    var bit_in_word: u32 = child_idx;
    if (child_idx >= 32u) {
      mask_word = mask_hi;
      bit_in_word = child_idx - 32u;
    }
    let bit = 1u << bit_in_word;
    if ((mask_word & bit) == 0u) {
      return palette;      }

    if (level == 3u) {
            let w = ts(node_addr + NODE_FIXED_WORDS + (child_idx >> 1u));
      return (w >> ((child_idx & 1u) * 16u)) & 0xFFFFu;
    }

    var popcount_below: u32;
    if (child_idx < 32u) {
      popcount_below = countOneBits(mask_lo & ((1u << bit_in_word) - 1u));
    } else {
      popcount_below = countOneBits(mask_lo) + countOneBits(mask_hi & ((1u << bit_in_word) - 1u));
    }
    node_addr = chunk_base + ts(node_addr + NODE_FIXED_WORDS + popcount_below);
  }
  return ts(node_addr + 2u) & 0xFFFFu;  }
