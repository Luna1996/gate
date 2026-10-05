import package::bindings::{gi_den_cfg, gi_den_phi, gi_dn_dst, gi_dn_src, gi_dn_tmp, gi_guide, gi_guide_out, gi_hist_cur, gi_hist_prev, gi_raw_tex};
import package::gi::common::{gi_hist_search_offset, gi_pack_dir, gi_unpack_dir};
import package::gi::consts::{GI_DEN_CLAMP_K, GI_DEN_G_KEY0, GI_DEN_G_KEY1, GI_DEN_G_NRM, GI_DEN_G_PREV, GI_DEN_GUIDE_WORDS, GI_DEN_H_COL, GI_DEN_H_KEY0, GI_DEN_H_KEY1, GI_DEN_H_M, GI_DEN_H_M1, GI_DEN_H_M2, GI_DEN_H_NRM, GI_DEN_HIST_WORDS, GI_DEN_LUMA_FLOOR_REL, GI_DEN_LUMA_K, GI_DEN_M_MAX, GI_DEN_NBR_R, GI_DEN_N_DOT, GI_DEN_SIGMA_ALPHA, GI_DEN_STAB_FRAMES, GI_DEN_STAB_MAX, GI_DEN_W_HIST_MIN, GI_HIST_SEARCH_TAPS, GI_LUMA};




//























//











//

















fn gi_den_valid_b(b: u32) -> bool {
  return (gi_guide[b + GI_DEN_G_KEY0] | gi_guide[b + GI_DEN_G_KEY1]) != 0u;
}


fn gi_den_key0_b(b: u32) -> u32 {
  return gi_guide[b + GI_DEN_G_KEY0];
}


fn gi_den_key1_b(b: u32) -> u32 {
  return gi_guide[b + GI_DEN_G_KEY1];
}



fn gi_den_nrm_b(b: u32) -> vec3<f32> {
  return gi_unpack_dir(gi_guide[b + GI_DEN_G_NRM]);
}








fn gi_den_guide_store(
  coord: vec2<i32>,
  size: vec2<u32>,
  valid: bool,
  k0: u32,
  k1: u32,
  n: vec3<f32>,
  prev_ok: bool,
  prev_px: vec2<i32>,
) {
  let idx = u32(coord.y) * size.x + u32(coord.x);
  let b = idx * GI_DEN_GUIDE_WORDS;
  if (!valid) {
    for (var i = 0u; i < GI_DEN_GUIDE_WORDS; i = i + 1u) { gi_guide_out[b + i] = 0u; }
    return;
  }
  gi_guide_out[b + GI_DEN_G_KEY0] = k0;
  gi_guide_out[b + GI_DEN_G_KEY1] = k1;
  gi_guide_out[b + GI_DEN_G_NRM] = gi_pack_dir(n);
  
  var code = 0xFFFFFFFFu;
  if (prev_ok) { code = u32(prev_px.x) | (u32(prev_px.y) << 16u); }
  gi_guide_out[b + GI_DEN_G_PREV] = code;
}












fn gi_den_same_plane(hk0: u32, hk1: u32, k0: u32, k1: u32) -> bool {
  let face = (k1 >> 16u) & 7u;
  if (face != ((hk1 >> 16u) & 7u)) { return false; }
  if ((k1 >> 19u) != (hk1 >> 19u)) { return false; }
  let axis = face >> 1u;
  if (axis == 0u) { return (k0 & 0xFFFFu) == (hk0 & 0xFFFFu); }
  if (axis == 1u) { return (k0 >> 16u) == (hk0 >> 16u); }
  return (k1 & 0xFFFFu) == (hk1 & 0xFFFFu);
}










fn gi_same_surface_n(hk0: u32, hk1: u32, hn: vec3<f32>, k0: u32, k1: u32, n: vec3<f32>) -> bool {
  if (!gi_den_same_plane(hk0, hk1, k0, k1)) { return false; }
  return dot(hn, n) >= GI_DEN_N_DOT;
}



fn gi_same_surface(hk0: u32, hk1: u32, hn: u32, k0: u32, k1: u32, n: vec3<f32>) -> bool {
  return gi_same_surface_n(hk0, hk1, gi_unpack_dir(hn), k0, k1, n);
}


























fn gi_den_temporal(size: vec2<u32>, coord: vec2<i32>) {
  let x = i32(coord.x);
  let y = i32(coord.y);
  let idx = u32(y) * size.x + u32(x);
  let gb = idx * GI_DEN_GUIDE_WORDS;
  let hb = idx * GI_DEN_HIST_WORDS;
  if (!gi_den_valid_b(gb)) {
    for (var i = 0u; i < GI_DEN_HIST_WORDS; i = i + 1u) { gi_hist_cur[hb + i] = 0u; }
    gi_den_phi[idx] = 0.0;
    textureStore(gi_dn_tmp, coord, vec4<f32>(0.0));
    return;
  }
  let k0 = gi_den_key0_b(gb);
  let k1 = gi_den_key1_b(gb);
  
  let n_c = gi_den_nrm_b(gb);
  let c_raw = textureLoad(gi_raw_tex, coord, 0).rgb;

  
  
  
  var n3 = 0.0; 
  var m3 = 0.0;
  var q3 = 0.0;
  var c3 = vec3<f32>(0.0); 
  var cnt = 0.0; 
  var csum = vec3<f32>(0.0);
  var csum2 = vec3<f32>(0.0);
  let nr = i32(GI_DEN_NBR_R);
  for (var dy: i32 = -nr; dy <= nr; dy = dy + 1) {
    for (var dx: i32 = -nr; dx <= nr; dx = dx + 1) {
      let tx = x + dx;
      let ty = y + dy;
      if (tx < 0 || ty < 0 || tx >= i32(size.x) || ty >= i32(size.y)) { continue; }
      let tb = (u32(ty) * size.x + u32(tx)) * GI_DEN_GUIDE_WORDS;
      
      let tk0 = gi_guide[tb + GI_DEN_G_KEY0];
      let tk1 = gi_guide[tb + GI_DEN_G_KEY1];
      if ((tk0 | tk1) == 0u) { continue; }
      if (!gi_den_same_plane(tk0, tk1, k0, k1)) { continue; }
      
      if (dot(gi_den_nrm_b(tb), n_c) < GI_DEN_N_DOT) { continue; }
      let c = textureLoad(gi_raw_tex, vec2<i32>(tx, ty), 0).rgb;
      if (tk0 == k0 && tk1 == k1) {
        let l = dot(c, GI_LUMA);
        m3 = m3 + l;
        q3 = q3 + l * l;
        c3 = c3 + c;
        n3 = n3 + 1.0;
      }
      csum = csum + c;
      csum2 = csum2 + c * c;
      cnt = cnt + 1.0;
    }
  }
  let mu3 = m3 / max(n3, 1.0);
  let mo2 = q3 / max(n3, 1.0); 
  
  let sigma = select(0.0, sqrt(max(mo2 - mu3 * mu3, 0.0)), n3 > 1.5);
  let cmean = csum / max(cnt, 1.0);
  let cstd = sqrt(max(csum2 / max(cnt, 1.0) - cmean * cmean, vec3<f32>(0.0)));
  
  let l_ref = max(dot(cmean, GI_LUMA), mu3);
  
  var sigma_use = sigma;
  
  var c_in = select(c_raw, c3 / max(n3, 1.0), n3 > 1.5);
  
  
  
  
  
  
  if (n3 <= 1.5 && cnt > 1.5) { c_in = cmean; }
  
  
  
  
  
  if (cnt > 1.5) {
    c_in = clamp(c_in, cmean - GI_DEN_CLAMP_K * cstd, cmean + GI_DEN_CLAMP_K * cstd);
  }
  
  
  
  let l_in = dot(c_in, GI_LUMA);
  var m1_acc = l_in;
  var m2_acc = l_in * l_in;

  
  var color = c_in;
  var m = 1.0;
  let code = gi_guide[gb + GI_DEN_G_PREV];
  if (code != 0xFFFFFFFFu) {
    let prev_px = vec2<i32>(i32(code & 0xFFFFu), i32((code >> 16u) & 0xFFFFu));
    
    
    
    
    //
    
    
    
    
    var hb = 0u;
    var h_m = 0.0;
    var found = false;
    
    var shifted = false;
    for (var s = 0u; s <= GI_HIST_SEARCH_TAPS; s = s + 1u) {
      let px = prev_px.x + gi_hist_search_offset(s).x;
      let py = prev_px.y + gi_hist_search_offset(s).y;
      if (px < 0 || py < 0 || px >= i32(size.x) || py >= i32(size.y)) { continue; }
      let cand_b = (u32(py) * size.x + u32(px)) * GI_DEN_HIST_WORDS;
      let cand_m = bitcast<f32>(gi_hist_prev[cand_b + GI_DEN_H_M]);
      if (cand_m <= 0.0) { continue; }
      if (!gi_same_surface(gi_hist_prev[cand_b + GI_DEN_H_KEY0],
                           gi_hist_prev[cand_b + GI_DEN_H_KEY1],
                           gi_hist_prev[cand_b + GI_DEN_H_NRM], k0, k1, n_c)) { continue; }
      hb = cand_b;
      h_m = cand_m;
      found = true;
      shifted = s != 0u;
      break;
    }
    if (found) {
      m = min(h_m + 1.0, f32(GI_DEN_M_MAX));
      let hcol = vec3<f32>(
        bitcast<f32>(gi_hist_prev[hb + GI_DEN_H_COL]),
        bitcast<f32>(gi_hist_prev[hb + GI_DEN_H_COL + 1u]),
        bitcast<f32>(gi_hist_prev[hb + GI_DEN_H_COL + 2u]),
      );
      var clamped = hcol;
      if (cnt > 1.5) {
        clamped = clamp(hcol, cmean - GI_DEN_CLAMP_K * cstd, cmean + GI_DEN_CLAMP_K * cstd);
      } else {
        
        
        
        
        let pad = l_ref * GI_DEN_LUMA_FLOOR_REL;
        clamped = clamp(hcol, c_in * (1.0 / GI_DEN_CLAMP_K) - pad, c_in * GI_DEN_CLAMP_K + pad);
      }
      
      
      
      let h_m1 = bitcast<f32>(gi_hist_prev[hb + GI_DEN_H_M1]);
      let h_m2 = bitcast<f32>(gi_hist_prev[hb + GI_DEN_H_M2]);
      let sigma_hist = sqrt(max(h_m2 - h_m1 * h_m1, 0.0));
      sigma_use = max(sigma, sigma_hist);
      
      
      
      
      
      var w_var = 1.0;
      if (any(clamped != hcol)) {
        w_var = clamp(GI_DEN_SIGMA_ALPHA / max(sigma_use, 1e-5), GI_DEN_W_HIST_MIN, 1.0);
      }
      
      
      
      
      
      
      let dx = f32(prev_px.x - x);
      let dy = f32(prev_px.y - y);
      let mo = sqrt(dx * dx + dy * dy);
      let w_motion = select(1.0 / (1.0 + mo), 1.0, !shifted);
      
      let w = min(min(w_var, (m - 1.0) / m), w_motion);
      color = mix(c_in, clamped, w);
      
      
      m1_acc = mix(h_m1, l_in, 1.0 - w);
      m2_acc = mix(h_m2, l_in * l_in, 1.0 - w);
    }
  }

  
  
  
  
  
  
  
  
  let sigma_phi = select(dot(cstd, GI_LUMA), sigma_use, n3 > 1.5);
  gi_den_phi[idx] = GI_DEN_LUMA_K * max(max(sigma_phi, GI_DEN_LUMA_FLOOR_REL * l_ref), 1e-5);

  
  gi_hist_cur[hb + GI_DEN_H_COL] = bitcast<u32>(color.x);
  gi_hist_cur[hb + GI_DEN_H_COL + 1u] = bitcast<u32>(color.y);
  gi_hist_cur[hb + GI_DEN_H_COL + 2u] = bitcast<u32>(color.z);
  gi_hist_cur[hb + GI_DEN_H_M1] = bitcast<u32>(m1_acc);
  gi_hist_cur[hb + GI_DEN_H_M2] = bitcast<u32>(m2_acc);
  gi_hist_cur[hb + GI_DEN_H_M] = bitcast<u32>(m);
  gi_hist_cur[hb + GI_DEN_H_KEY0] = k0;
  gi_hist_cur[hb + GI_DEN_H_KEY1] = k1;
  
  gi_hist_cur[hb + GI_DEN_H_NRM] = gi_guide[gb + GI_DEN_G_NRM];
  textureStore(gi_dn_tmp, coord, vec4<f32>(color, 1.0));
}

















fn gi_den_atrous(size: vec2<u32>, coord: vec2<i32>, step: i32) {
  let x = i32(coord.x);
  let y = i32(coord.y);
  let idx = u32(y) * size.x + u32(x);
  let gb = idx * GI_DEN_GUIDE_WORDS;
  if (!gi_den_valid_b(gb)) {
    textureStore(gi_dn_dst, coord, vec4<f32>(0.0));
    return;
  }
  
  let r = i32(gi_den_cfg[0u]);
  let hb = idx * GI_DEN_HIST_WORDS;
  let m_hist = bitcast<f32>(gi_hist_cur[hb + GI_DEN_H_M]);
  let stab = clamp(GI_DEN_STAB_FRAMES / max(m_hist, 1.0), 1.0, GI_DEN_STAB_MAX);
  let st = max(1, i32(round(f32(step) * stab)));
  let k0 = gi_den_key0_b(gb);
  let k1 = gi_den_key1_b(gb);
  let n_c = gi_den_nrm_b(gb);
  let phi = max(gi_den_phi[idx], 1e-5);
  let c0 = textureLoad(gi_dn_src, coord, 0).rgb;
  let l0 = dot(c0, GI_LUMA);
  var acc = c0;
  var wsum = 1.0;
  for (var dy: i32 = -r; dy <= r; dy = dy + 1) {
    for (var dx: i32 = -r; dx <= r; dx = dx + 1) {
      if (dx == 0 && dy == 0) { continue; }
      let tx = x + dx * st;
      let ty = y + dy * st;
      if (tx < 0 || ty < 0 || tx >= i32(size.x) || ty >= i32(size.y)) { continue; }
      let tb = (u32(ty) * size.x + u32(tx)) * GI_DEN_GUIDE_WORDS;
      
      
      
      let tk0 = gi_guide[tb + GI_DEN_G_KEY0];
      let tk1 = gi_guide[tb + GI_DEN_G_KEY1];
      if ((tk0 | tk1) == 0u) { continue; }
      if (!gi_den_same_plane(tk0, tk1, k0, k1)) { continue; }
      let tn = gi_den_nrm_b(tb);
      if (dot(tn, n_c) < GI_DEN_N_DOT) { continue; }
      let s = textureLoad(gi_dn_src, vec2<i32>(tx, ty), 0).rgb;
      
      
      
      var w = 1.0;
      if (!(tk0 == k0 && tk1 == k1)) {
        w = clamp(dot(tn, n_c), 0.0, 1.0) * exp(-abs(dot(s, GI_LUMA) - l0) / phi);
      }
      acc = acc + s * w;
      wsum = wsum + w;
    }
  }
  textureStore(gi_dn_dst, coord, vec4<f32>(acc / max(wsum, 1e-6), 1.0));
}
