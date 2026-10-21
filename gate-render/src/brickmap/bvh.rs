use glam::Vec3;

pub const BVH_HEADER_WORDS: usize = 4;
pub const BVH_NODE_WORDS: usize = 8;
const LEAF_MAX: usize = 4;

pub fn build(aabbs: &[(Vec3, Vec3)], far: &[bool]) -> Vec<u32> {
  let n = aabbs.len();
  let far_of = |i: usize| far.get(i).copied().unwrap_or(false);
  let order: Vec<u32> = (1..n).filter(|&i| !far_of(i)).map(|i| i as u32).collect();
  let far_idx: Vec<u32> = (1..n).filter(|&i| far_of(i)).map(|i| i as u32).collect();

  let mut nodes: Vec<[u32; BVH_NODE_WORDS]> = Vec::new();
  let mut sorted = order;
  if !sorted.is_empty() {
    build_node(aabbs, &mut sorted, 0, &mut nodes);
  }

  let node_count = nodes.len();
  let order_base = BVH_HEADER_WORDS + node_count * BVH_NODE_WORDS;
  let far_base = order_base + sorted.len();
  let mut out = Vec::with_capacity(far_base + far_idx.len());
  out.push(node_count as u32);
  out.push(order_base as u32);
  out.push(far_base as u32);
  out.push(far_idx.len() as u32);
  for node in &nodes {
    out.extend_from_slice(node);
  }
  out.extend_from_slice(&sorted);
  out.extend_from_slice(&far_idx);
  out
}

fn build_node(
  aabbs: &[(Vec3, Vec3)],
  order: &mut [u32],
  base: usize,
  nodes: &mut Vec<[u32; BVH_NODE_WORDS]>,
) -> u32 {
  let idx = nodes.len() as u32;
  nodes.push([0; BVH_NODE_WORDS]);
  let (mn, mx) = bounds(aabbs, order);
  if order.len() <= LEAF_MAX {
    nodes[idx as usize] = pack(mn, mx, base as u32, order.len() as u32);
    return idx;
  }
  let axis = widest(mx - mn);
  order.sort_unstable_by(|a, b| centroid(aabbs, *a)[axis].total_cmp(&centroid(aabbs, *b)[axis]));
  let mid = order.len() / 2;
  let (l, r) = order.split_at_mut(mid);
  let left = build_node(aabbs, l, base, nodes);
  debug_assert_eq!(left, idx + 1, "左子必须紧跟父节点");
  let right = build_node(aabbs, r, base + mid, nodes);
  nodes[idx as usize] = pack(mn, mx, right, 0);
  idx
}

fn pack(mn: Vec3, mx: Vec3, payload: u32, count: u32) -> [u32; BVH_NODE_WORDS] {
  [
    mn.x.to_bits(),
    mn.y.to_bits(),
    mn.z.to_bits(),
    payload,
    mx.x.to_bits(),
    mx.y.to_bits(),
    mx.z.to_bits(),
    count,
  ]
}

fn bounds(aabbs: &[(Vec3, Vec3)], order: &[u32]) -> (Vec3, Vec3) {
  let mut mn = Vec3::splat(f32::INFINITY);
  let mut mx = Vec3::splat(f32::NEG_INFINITY);
  for &i in order {
    let (a, b) = aabbs[i as usize];
    mn = mn.min(a);
    mx = mx.max(b);
  }
  (mn, mx)
}

fn centroid(aabbs: &[(Vec3, Vec3)], i: u32) -> Vec3 {
  let (a, b) = aabbs[i as usize];
  (a + b) * 0.5
}

fn widest(d: Vec3) -> usize {
  if d.x >= d.y && d.x >= d.z {
    0
  } else if d.y >= d.z {
    1
  } else {
    2
  }
}
