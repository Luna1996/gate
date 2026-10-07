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

#[cfg(test)]
mod tests {
  use super::*;

  fn node_at(words: &[u32], node: u32) -> (Vec3, Vec3, u32, u32) {
    let b = BVH_HEADER_WORDS + node as usize * BVH_NODE_WORDS;
    let f = |k: usize| f32::from_bits(words[b + k]);
    (Vec3::new(f(0), f(1), f(2)), Vec3::new(f(4), f(5), f(6)), words[b + 3], words[b + 7])
  }

  fn covers(words: &[u32], node: u32, aabbs: &[(Vec3, Vec3)]) -> bool {
    let (mn, mx, payload, count) = node_at(words, node);
    let order_base = words[1] as usize;
    if count == 0 {
      return covers(words, node + 1, aabbs) && covers(words, payload, aabbs);
    }
    for k in 0..count as usize {
      let i = words[order_base + payload as usize + k] as usize;
      let (a, b) = aabbs[i];
      if a.x < mn.x || a.y < mn.y || a.z < mn.z || b.x > mx.x || b.y > mx.y || b.z > mx.z {
        return false;
      }
    }
    true
  }

  fn covered_leaves(words: &[u32]) -> Vec<u32> {
    let node_count = words[0] as usize;
    let order_base = words[1] as usize;
    let mut out = Vec::new();
    for n in 0..node_count {
      let (_mn, _mx, payload, count) = node_at(words, n as u32);
      if count > 0 {
        for k in 0..count as usize {
          out.push(words[order_base + payload as usize + k]);
        }
      }
    }
    out.sort_unstable();
    out
  }

  #[test]
  fn bvh_covers_every_near_instance_once() {
    let mut aabbs = vec![(Vec3::ZERO, Vec3::ONE)];
    for i in 1..200 {
      let c = Vec3::new(i as f32 * 3.0 % 41.0, i as f32 * 1.7 % 23.0, i as f32 * 0.9 % 17.0);
      aabbs.push((c, c + Vec3::splat(4.0)));
    }
    let far = vec![false; aabbs.len()];
    let words = build(&aabbs, &far);
    assert_eq!(words[3], 0, "没有远场时远场个数应为 0");
    assert!(covers(&words, 0, &aabbs), "节点包围盒必须把叶实例全包住");
    let want: Vec<u32> = (1..aabbs.len() as u32).collect();
    assert_eq!(covered_leaves(&words), want, "每个实例恰好出现在一个叶子里");
  }

  #[test]
  fn far_instances_keep_original_order() {
    let mut aabbs = vec![(Vec3::ZERO, Vec3::ONE)];
    for i in 1..40 {
      aabbs.push((Vec3::splat(i as f32), Vec3::splat(i as f32 + 1.0)));
    }
    let mut far = vec![false; aabbs.len()];
    for i in [3usize, 7, 31] {
      far[i] = true;
    }
    let words = build(&aabbs, &far);
    assert_eq!(words[3], 3);
    let far_base = words[2] as usize;
    assert_eq!(&words[far_base..far_base + 3], &[3, 7, 31]);
    let near = covered_leaves(&words);
    assert!(!near.contains(&3) && !near.contains(&7) && !near.contains(&31));
    assert_eq!(near.len(), aabbs.len() - 1 - 3);
  }
}
