
//




//









//












//




//



import package::bindings::{gi_u, gi_wal};
import package::gi::common::{gi_key0, gi_key1, gi_key_ok, gi_key_org, gi_local_rev, gi_mix};

const GI_WAL_WORDS: u32 = 8u;

const GI_WAL_CAP_N: f32 = 256.0;

const GI_WAL_MATURE_N: f32 = 64.0;

const GI_WAL_RESET_N: f32 = 16.0;
















const GI_WAL_POOL_R: i32 = 3;


fn gi_wal_nmask() -> u32 {
  let n = arrayLength(&gi_wal) / GI_WAL_WORDS;
  if (n < 16u || (n & (n - 1u)) != 0u) { return 0u; }
  return n - 1u;
}


fn gi_wal_slot(v_world: vec3<i32>, face: u32, obj: i32, nmask: u32) -> u32 {
  let rkey = face | ((u32(obj) + 1u) << 3u);
  let h = gi_mix(u32(v_world.x)
          + gi_mix(u32(v_world.y)
            + gi_mix(u32(v_world.z) + gi_mix(rkey * 0x9E3779B1u))));
  return h & nmask;
}





fn gi_wal_mask_at(v_world: vec3<i32>, org: vec3<i32>, obj: i32, pal: u32) -> u32 {
  
  
  
  
  
  //
  
  
  
  
  
  
  //
  
  
  
  var rev = 0u;
  if (gi_u.params.x > 0.5) { rev = gi_local_rev(v_world, obj); }
  return gi_mix(u32(org.x)
          + gi_mix(u32(org.y)
            + gi_mix(u32(org.z) + gi_mix(rev + pal * 0x9E3779B1u))));
}



fn gi_wal_merge(v_world: vec3<i32>, obj: i32, pal: u32, k0: u32, k1: u32,
                sum_e: vec3<f32>, cnt: u32) {
  let nmask = gi_wal_nmask();
  if (nmask == 0u || cnt == 0u) { return; }
  let slot = gi_wal_slot(v_world, (k1 >> 16u) & 7u, obj, nmask);
  let b = slot * GI_WAL_WORDS;
  let mask = gi_wal_mask_at(v_world, gi_key_org(obj), obj, pal);
  let same = (atomicLoad(&gi_wal[b]) == k0) && (atomicLoad(&gi_wal[b + 1u]) == k1);
  var sum = vec3<f32>(0.0);
  var n = 0.0;
  var mature = false;
  if (same) {
    let n_old = bitcast<f32>(atomicLoad(&gi_wal[b + 6u]));
    if (n_old > 0.0) {
      sum = vec3<f32>(bitcast<f32>(atomicLoad(&gi_wal[b + 3u])),
                      bitcast<f32>(atomicLoad(&gi_wal[b + 4u])),
                      bitcast<f32>(atomicLoad(&gi_wal[b + 5u])));
      n = n_old;
      mature = (atomicLoad(&gi_wal[b + 7u]) & 1u) != 0u;
      if (atomicLoad(&gi_wal[b + 2u]) != mask) {
        
        let keep = min(n, GI_WAL_RESET_N);
        sum = (sum / n) * keep;
        n = keep;
      }
    }
  }
  
  let n_eff = min(n, max(GI_WAL_CAP_N - f32(cnt), 0.0));
  if (n > n_eff) { sum = sum * (n_eff / n); }
  sum = sum + sum_e;
  n = n_eff + f32(cnt);
  
  atomicStore(&gi_wal[b + 3u], bitcast<u32>(sum.x));
  atomicStore(&gi_wal[b + 4u], bitcast<u32>(sum.y));
  atomicStore(&gi_wal[b + 5u], bitcast<u32>(sum.z));
  atomicStore(&gi_wal[b + 6u], bitcast<u32>(n));
  atomicStore(&gi_wal[b + 7u], select(0u, 1u, mature || n >= GI_WAL_MATURE_N));
  atomicStore(&gi_wal[b + 2u], mask);
  atomicStore(&gi_wal[b + 1u], k1);
  atomicStore(&gi_wal[b], k0);
}


fn gi_wal_fetch(v_world: vec3<i32>, org: vec3<i32>, obj: i32, pal: u32, face: u32,
                nmask: u32, k0: u32, k1: u32) -> vec4<f32> {
  let b = gi_wal_slot(v_world, face, obj, nmask) * GI_WAL_WORDS;
  if (atomicLoad(&gi_wal[b]) != k0 || atomicLoad(&gi_wal[b + 1u]) != k1) { return vec4<f32>(0.0); }
  let n = bitcast<f32>(atomicLoad(&gi_wal[b + 6u]));
  if (!(n > 0.0)) { return vec4<f32>(0.0); }
  if (atomicLoad(&gi_wal[b + 2u]) != gi_wal_mask_at(v_world, org, obj, pal)) { return vec4<f32>(0.0); }
  return vec4<f32>(bitcast<f32>(atomicLoad(&gi_wal[b + 3u])),
                   bitcast<f32>(atomicLoad(&gi_wal[b + 4u])),
                   bitcast<f32>(atomicLoad(&gi_wal[b + 5u])), n);
}







fn gi_wal_pool(v_world: vec3<i32>, obj: i32, pal: u32, k1: u32) -> vec4<f32> {
  let nmask = gi_wal_nmask();
  if (nmask == 0u || !gi_key_ok(v_world, obj)) { return vec4<f32>(0.0); }
  let face = (k1 >> 16u) & 7u;
  
  let a1 = (face / 2u + 1u) % 3u;
  let a2 = (face / 2u + 2u) % 3u;
  let t1 = vec3<i32>(select(0, 1, a1 == 0u), select(0, 1, a1 == 1u), select(0, 1, a1 == 2u));
  let t2 = vec3<i32>(select(0, 1, a2 == 0u), select(0, 1, a2 == 1u), select(0, 1, a2 == 2u));
  let org = gi_key_org(obj);
  var sum = vec3<f32>(0.0);
  var cnt = 0.0;
  for (var i = -GI_WAL_POOL_R; i <= GI_WAL_POOL_R; i = i + 1) {
    for (var j = -GI_WAL_POOL_R; j <= GI_WAL_POOL_R; j = j + 1) {
      let v = v_world + t1 * i + t2 * j;
      if (!gi_key_ok(v, obj)) { continue; }
      let s = gi_wal_fetch(v, org, obj, pal, face, nmask, gi_key0(v, obj), gi_key1(v, face, obj));
      sum = sum + s.xyz;
      cnt = cnt + s.w;
    }
  }
  return vec4<f32>(sum, cnt);
}




fn gi_wal_sample(v_world: vec3<i32>, obj: i32, pal: u32, k1: u32) -> vec4<f32> {
  let p = gi_wal_pool(v_world, obj, pal, k1);
  if (!(p.w > 0.0)) { return vec4<f32>(0.0); }
  return vec4<f32>(p.xyz / p.w, min(1.0, p.w / GI_WAL_MATURE_N));
}
