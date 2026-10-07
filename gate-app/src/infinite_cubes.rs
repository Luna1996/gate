use bevy::prelude::{Local, Res, ResMut, Resource};
use gate_voxel::{
  ChunkCoord, ChunkProducer, ChunkSource, ChunkTree, Detail, PaletteEntry, PaletteFlags, PaletteId,
  PbrOverrides, VolumeGrid, Volumes, fill_bricks,
};
use glam::{IVec3, Vec3};

pub const ROOM: i32 = 148;
pub const COLUMN: i32 = 24;
pub const CUBE: i32 = 44;
const _: () = assert!(ROOM % 4 == 0 && CUBE % 4 == 0 && (ROOM - CUBE) % 8 == 0);
pub const WHITE_SLOT: PaletteId = PaletteId(1);

pub const START_CHUNKS: i32 = 1;

pub(crate) fn initial_box(center: IVec3) -> (IVec3, IVec3) {
  let chunk = IVec3::splat(gate_voxel::CHUNK_SIZE);
  let c = center.div_euclid(chunk);
  ((c - IVec3::splat(START_CHUNKS)) * chunk, (c + IVec3::splat(START_CHUNKS + 1)) * chunk)
}

pub const FAR_SCALES: [i32; 3] = [4, 16, 64];

pub const DEFAULT_REQUEST_BYTES: usize = 2 * 1024 * 1024 * 1024;
pub const FAR_GRAIN: i32 = 16;

pub fn far_coverage_voxels(vol: usize) -> i32 {
  let ladder = far_radius_ladder(gate_render::pool_capacity_chunks_far(DEFAULT_REQUEST_BYTES));
  let (_, r_out) = ladder[(vol - 1).min(ladder.len() - 1)];
  (r_out + 1 - SEAM_MARGIN).max(1) * gate_voxel::CHUNK_SIZE * FAR_SCALES[vol - 1]
}

pub fn attach_far_levels(volumes: &mut Volumes, pbr_ids: &[String], center: IVec3) {
  let slots = material_slots(pbr_ids.len());
  attach_far_levels_with(volumes, center, |g| {
    for (id, entry) in slots.iter() {
      g.palette_mut().set(*id, *entry);
    }
  });
}

pub fn attach_far_levels_mc(volumes: &mut Volumes, center: IVec3) {
  attach_far_levels_with(volumes, center, |_| {});
}

fn attach_far_levels_with(
  volumes: &mut Volumes,
  center: IVec3,
  fill_palette: impl Fn(&mut gate_voxel::VolumeGrid),
) {
  const WINDOW_CHUNKS: i32 = 32;
  let ladder = far_radius_ladder(gate_render::pool_capacity_chunks_far(DEFAULT_REQUEST_BYTES));
  let mut dims = IVec3::splat(WINDOW_CHUNKS * 2);
  for (k, &scale) in FAR_SCALES.iter().enumerate() {
    let (_, r_out) = ladder[k];
    dims.x = 2 * r_out + 2;
    dims.z = 2 * r_out + 2;
    let dims = dims;
    let vol = volumes.add_far_level(scale as f32);
    let g = &mut volumes.list[vol];
    fill_palette(g);
    let c = (center / scale).div_euclid(IVec3::splat(gate_voxel::CHUNK_SIZE));
    g.set_stream_window(Some((c - dims / 2, dims)));
    g.set_coverage_r(far_coverage_voxels(vol) as f32);
    bevy::log::info!(
      "FAR L{vol} scale {scale}（1 级体素 = {scale} 世界体素）覆盖 ±{:.2}km；窗口 {}×{}×{} chunk × 256 级体素、\
       格 {FAR_GRAIN} 级体素 = {:.2}m；半径阶梯 r{r_in}..{r_out}",
      far_coverage_voxels(vol) as f32 * 0.02 / 1000.0,
      dims.x,
      dims.y,
      dims.z,
      FAR_GRAIN as f32 * scale as f32 * 0.02,
      r_in = ladder[k].0,
      r_out = r_out,
    );
  }
}

const PRODUCER_WORKERS: usize = 3;

const DISPATCH_PER_FRAME: usize = 128;

const DEMAND_TABLE_MAX: usize = 8192;

const DEMAND_REBUILD_FRAMES: u32 = 30;

const DEMAND_REBUILD_CHUNKS: i32 = 6;

const DEMAND_PATCH_MARGIN: i32 = 3;

const POLL_MAX: usize = 256;

const READY_MAX: usize = 384;

const SWEEP_FRAMES: u32 = 240;

const FAR_INFLIGHT_MAX: usize = PRODUCER_WORKERS * 4;

#[derive(Default)]
struct VolState {
  demand: std::collections::VecDeque<(IVec3, Detail)>,
  demand_center: Option<IVec3>,
  demand_age: u32,
  ready: std::collections::VecDeque<(ChunkCoord, Detail, ChunkTree, usize)>,
  ready_set: std::collections::HashSet<ChunkCoord>,
  detail: std::collections::HashMap<ChunkCoord, Detail>,
  last_seq: u64,
  last_used: std::collections::HashMap<ChunkCoord, u64>,
  empty: std::collections::HashSet<ChunkCoord>,
  demand_set: std::collections::HashSet<ChunkCoord>,
  trim_idle: bool,
  inflight: usize,
}

struct Pipeline {
  producer: ChunkProducer,
  source: std::sync::Arc<dyn ChunkSource>,
  vols: Vec<VolState>,
  use_seq: u64,
  use_stamp: u32,
}

#[derive(Clone, Copy)]
struct VolScope {
  center: IVec3,
  w_origin: IVec3,
  w_dims: IVec3,
  moved: bool,
  v_hy: i32,
}

impl VolScope {
  fn in_window(&self, c: IVec3) -> bool {
    let t = c - self.w_origin;
    t.cmpge(IVec3::ZERO).all() && t.cmplt(self.w_dims).all()
  }
}

#[derive(Resource)]
pub struct Streaming {
  pub paused: bool,
  pub load_radius: i32,
  pub unload_radius: i32,
  pub coarse_radius: i32,
  pub coarse_height: i32,
  pub mount_words: usize,
  pub mount_count: usize,
  pub request_bytes: usize,
  pub far_levels_max: usize,
  pub requests_load: bool,
  source: Option<std::sync::Arc<dyn ChunkSource>>,
  builtin: Option<(usize, std::sync::Arc<dyn ChunkSource>)>,
  palette_applied: usize,
  pipeline: Option<std::sync::Mutex<Pipeline>>,
}

impl Streaming {
  pub fn set_source(&mut self, source: Option<std::sync::Arc<dyn ChunkSource>>) {
    self.source = source;
  }

  pub fn has_custom_source(&self) -> bool {
    self.source.is_some()
  }

  pub fn source(&self) -> Option<std::sync::Arc<dyn ChunkSource>> {
    self.source.clone()
  }
}

impl Default for Streaming {
  fn default() -> Self {
    Self {
      paused: false,
      load_radius: 2,
      unload_radius: 3,
      coarse_radius: 3,
      coarse_height: 3,
      mount_words: 1024 * 1024,
      mount_count: 48,
      request_bytes: DEFAULT_REQUEST_BYTES,
      requests_load: true,
      far_levels_max: 3,
      source: None,
      builtin: None,
      palette_applied: 0,
      pipeline: None,
    }
  }
}

struct InfiniteCubes {
  n_pbr: usize,
}

impl ChunkSource for InfiniteCubes {
  fn produce(
    &self,
    vol: usize,
    coord: ChunkCoord,
    detail: Detail,
    scratch: &mut VolumeGrid,
  ) -> Option<ChunkTree> {
    let lo = coord.0 * gate_voxel::CHUNK_SIZE;
    let hi = lo + IVec3::splat(gate_voxel::CHUNK_SIZE);
    if vol == 0 {
      build_region(scratch, lo, hi, self.n_pbr, detail);
    } else {
      build_region_far(scratch, lo, hi, self.n_pbr, FAR_SCALES[vol - 1], FAR_GRAIN);
    }
    scratch.take_chunk(coord)
  }
}

#[allow(clippy::too_many_arguments)]
pub fn stream_chunks(
  mut stream: ResMut<Streaming>,
  mut scene: ResMut<gate_render::VoxelScene>,
  cam: Option<Res<gate_render::DdaCameraConfig>>,
  pbr: Option<Res<gate_render::PbrTextureSet>>,
  feed: Option<Res<gate_render::LodRequestFeed>>,
  use_feed: Option<Res<gate_render::ChunkUseFeed>>,
  mut frames: Local<u32>,
  mut diag: Local<(f64, u32)>,
  mut split: Local<Option<gate_render::profiler::SplitDiag>>,
) {
  let _t = gate_render::profiler::SysTimer::new("STREAM 流式装载", &mut diag);
  let sd = split.get_or_insert_with(|| {
    gate_render::profiler::SplitDiag::new(&[
      "⓪窗口",
      "取源peek",
      "①取回",
      "①a建表",
      "①b派发",
      "①c挂载",
      "②卸载",
      "尾",
    ])
  });
  sd.start();
  if stream.paused {
    return;
  }
  *frames = frames.wrapping_add(1);
  let Some(cam) = cam else { return };
  if scene.volumes.main().stream_window().is_none() {
    return;
  }
  let chunk = gate_voxel::CHUNK_SIZE;
  let forward = cam.forward;
  let cam_world = cam.position_world;

  scene.residency_budget_bytes = stream.request_bytes;

  let n_vol = scene.volumes.len();
  let content_y = stream.source().and_then(|s| s.content_y_range());
  let mut scopes: Vec<VolScope> = Vec::with_capacity(n_vol);
  for vol in 0..n_vol {
    let scale = scene.volumes.list[vol].transform().scale;
    let Some((o, d)) = scene.volumes.list[vol].stream_window() else {
      scopes.push(VolScope {
        center: IVec3::ZERO,
        w_origin: IVec3::ZERO,
        w_dims: IVec3::ZERO,
        moved: false,
        v_hy: FAR_PRELOAD_HY,
      });
      continue;
    };
    let cam_c = (cam_world / scale / chunk as f32).floor().as_ivec3();
    let (c, v_hy) = match content_y {
      Some((lo, hi)) => {
        let span = (chunk as f32 * scale).round() as i32;
        let lo_c = lo.div_euclid(span);
        let hi_c = (hi + span - 1).div_euclid(span);
        let rows = (hi_c - lo_c).max(1);
        (IVec3::new(cam_c.x, lo_c + (rows - 1) / 2, cam_c.z), rows / 2 + 1)
      }
      None => (cam_c, FAR_PRELOAD_HY),
    };
    let want = c - d / 2;
    let moved = want != o;
    if moved {
      scene.volumes.list[vol].set_stream_window(Some((want, d)));
    }
    scopes.push(VolScope { center: c, w_origin: want, w_dims: d, moved, v_hy });
  }
  sd.mark(0);
  let (load_r, unload_r, coarse_r, coarse_h, mount_words, mount_count) = (
    stream.load_radius,
    stream.unload_radius,
    stream.coarse_radius,
    stream.coarse_height,
    stream.mount_words,
    stream.mount_count,
  );
  let (request_bytes, requests_load) = (stream.request_bytes, stream.requests_load);
  let far_max = stream.far_levels_max;
  let cap = gate_render::pool_capacity_chunks(request_bytes);
  let cap_far = gate_render::pool_capacity_chunks_far(request_bytes);
  let ladder = far_radius_ladder(cap_far);
  let cap_main = DEMAND_TABLE_MAX.min(cap);

  let n_pbr = crate::scene::pbr_asset_count(pbr.as_deref());
  let source: std::sync::Arc<dyn ChunkSource> = if let Some(s) = stream.source.clone() {
    s
  } else {
    match stream.builtin.as_ref().filter(|(n, _)| *n == n_pbr).map(|(_, s)| s.clone()) {
      Some(s) => s,
      None => {
        let s: std::sync::Arc<dyn ChunkSource> = std::sync::Arc::new(InfiniteCubes { n_pbr });
        stream.builtin = Some((n_pbr, s.clone()));
        s
      }
    }
  };
  let stale = match &stream.pipeline {
    None => true,
    Some(p) => {
      !std::sync::Arc::ptr_eq(&p.lock().unwrap_or_else(|e| e.into_inner()).source, &source)
    }
  };
  let mut palette_applied = stream.palette_applied;
  if stale {
    stream.pipeline = Some(std::sync::Mutex::new(Pipeline {
      producer: ChunkProducer::new(source.clone(), PRODUCER_WORKERS),
      source: source.clone(),
      vols: (0..n_vol).map(|_| VolState::default()).collect(),
      use_seq: 0,
      use_stamp: 0,
    }));
    palette_applied = 0;
  }
  let requests: Vec<gate_render::LodRequest> =
    if requests_load { feed.as_ref().map(|f| f.peek()).unwrap_or_default() } else { Vec::new() };
  let uses: Vec<gate_render::LodUse> = use_feed.as_ref().map(|f| f.peek()).unwrap_or_default();
  sd.mark(1);

  let mut generated = 0usize;
  {
    let mut guard =
      stream.pipeline.as_ref().expect("上面刚装上").lock().unwrap_or_else(|e| e.into_inner());
    let Pipeline { producer, vols, use_seq, use_stamp, source: _ } = &mut *guard;
    while vols.len() < n_vol {
      vols.push(VolState::default());
    }

    let stamp = uses.iter().map(|u| u.stamp).max().unwrap_or(0);
    if stamp != 0 && stamp != *use_stamp {
      *use_stamp = stamp;
      *use_seq += 1;
      let seq = *use_seq;
      for u in &uses {
        if let Some(st) = vols.get_mut(u.vol as usize) {
          st.last_used.insert(ChunkCoord(u.chunk), seq);
        }
      }
    }

    for (v, cc, detail, tree, words) in producer.poll(POLL_MAX) {
      if v > far_max {
        if let Some(st) = vols.get_mut(v) {
          st.inflight = st.inflight.saturating_sub(1);
        }
        continue;
      }
      let Some(st) = vols.get_mut(v) else { continue };
      st.inflight = st.inflight.saturating_sub(1);
      match tree {
        Some(_) if !st.ready_set.insert(cc) => {}
        Some(tree) => st.ready.push_back((cc, detail, tree, words)),
        None => {
          st.empty.insert(cc);
          if let Some(g) = scene.volumes.list.get_mut(v) {
            g.mark_empty_chunk(cc);
          }
        }
      }
    }

    let updates = source.palette_log(palette_applied);
    if !updates.is_empty() {
      for (id, e) in &updates {
        for g in scene.volumes.list.iter_mut() {
          g.palette_mut().set(*id, *e);
        }
      }
      bevy::log::debug!(
        "MC 调色板 +{} 槽（累计 {}）",
        updates.len(),
        palette_applied + updates.len()
      );
      palette_applied += updates.len();
    }
    sd.mark(2);

    let mut words = 0usize;
    let mut words_mounted = 0usize;

    let first = (*frames as usize) % n_vol.max(1);
    for i in 0..n_vol {
      let vol = (first + i) % n_vol;
      let mut gen_req = 0usize;
      let scope = scopes[vol];
      let st = &mut vols[vol];
      let far_level = scene.volumes.list[vol].is_far_level();
      if !far_level && scene.volumes.list[vol].stream_window().is_none() {
        continue;
      }
      let cap = if far_level { cap_far } else { cap };
      let grid = &mut scene.volumes.list[vol];
      if far_level && vol > far_max {
        for cc in grid.chunk_coords().collect::<Vec<_>>() {
          grid.unmount_chunk(cc);
          st.detail.remove(&cc);
          st.last_used.remove(&cc);
        }
        st.demand.clear();
        st.ready.clear();
        st.ready_set.clear();
        continue;
      }

      st.demand_age += 1;
      let moved = match st.demand_center {
        Some(c) => (c - scope.center).abs().max_element(),
        None => i32::MAX,
      };
      let have = |c: IVec3, want: Detail| {
        let held = st.detail.get(&ChunkCoord(c)).copied().unwrap_or(Detail::FULL);
        (grid.chunk(ChunkCoord(c)).is_some() && held >= want) || st.empty.contains(&ChunkCoord(c))
      };
      let ch = if content_y.is_some() { scope.v_hy } else { coarse_h };
      let far_moved = far_level && moved > 0;
      if moved >= DEMAND_REBUILD_CHUNKS || st.demand_age >= DEMAND_REBUILD_FRAMES || far_moved {
        let batch: Vec<(IVec3, Detail)> = if far_level {
          let (r_in, r_out) = ladder[(vol - 1).min(ladder.len() - 1)];
          plan_generation_far(vol, scope, far_cap(vol, cap), r_in, r_out, &requests, have)
        } else {
          let (batch, from_req) = plan_generation(
            GenScope {
              center: scope.center,
              w_origin: scope.w_origin,
              w_dims: scope.w_dims,
              load_radius: load_r,
              coarse_radius: coarse_r,
              coarse_height: ch,
              forward,
            },
            cap_main,
            &requests,
            have,
          );
          gen_req = from_req;
          batch
        };
        if far_level {
          bevy::log::debug!(
            "DEMAND[v{vol} far] batch {}（上限 {}；常驻 {}）",
            batch.len(),
            far_cap(vol, cap),
            grid.chunk_count(),
          );
        } else {
          bevy::log::debug!(
            "DEMAND[v{vol} far{far_level}] batch {}（请求 {gen_req}；预载盘 {coarse_r}×{coarse_h}、上限 {cap_main}；常驻 {}）",
            batch.len(),
            grid.chunk_count(),
          );
        }
        st.demand = batch.into();
        st.demand_set = st.demand.iter().map(|(c, _)| ChunkCoord(*c)).collect();
        st.demand_center = Some(scope.center);
        st.demand_age = 0;
        st.trim_idle = false;
      } else if moved > 0 {
        let pr = load_r + DEMAND_PATCH_MARGIN;
        let mut patch: Vec<(IVec3, Detail)> = Vec::new();
        for dx in -pr..=pr {
          for dz in -pr..=pr {
            for dy in -ch..=ch {
              let c = scope.center + IVec3::new(dx, dy, dz);
              let t = c - scope.w_origin;
              if !t.cmpge(IVec3::ZERO).all() || !t.cmplt(scope.w_dims).all() {
                continue;
              }
              let want = preload_detail(c, scope.center);
              if have(c, want) || st.demand_set.contains(&ChunkCoord(c)) {
                continue;
              }
              patch.push((c, want));
            }
          }
        }
        if !patch.is_empty() {
          patch.sort_unstable_by_key(|(c, _)| plan_key(*c, scope.center, forward));
          for (c, want) in patch.into_iter().rev() {
            st.demand_set.insert(ChunkCoord(c));
            st.demand.push_front((c, want));
          }
          while st.demand.len() > DEMAND_TABLE_MAX {
            match st.demand.pop_back() {
              Some((c, _)) => {
                st.demand_set.remove(&ChunkCoord(c));
              }
              None => break,
            }
          }
          st.trim_idle = false;
        }
        st.demand_center = Some(scope.center);
      }
      sd.mark(3);

      let mut dispatched = 0usize;
      while !(far_level && st.inflight >= FAR_INFLIGHT_MAX) {
        if st.ready.len() + producer.inflight() >= READY_MAX {
          break;
        }
        let Some((c, detail)) = st.demand.front().copied() else { break };
        if producer.in_flight(vol, ChunkCoord(c)) {
          st.demand.pop_front();
          continue;
        }
        if st.ready_set.contains(&ChunkCoord(c)) {
          st.demand.pop_front();
          continue;
        }
        if !producer.request(vol, ChunkCoord(c), detail) {
          break;
        }
        st.inflight += 1;
        st.demand.pop_front();
        dispatched += 1;
        if dispatched >= DISPATCH_PER_FRAME {
          break;
        }
      }
      sd.mark(4);

      while !st.ready.is_empty() && words_mounted < mount_count {
        let w = st.ready.front().expect("刚看过 len").3;
        if words_mounted > 0 && words + w > mount_words {
          break;
        }
        let (cc, detail, tree, _words) = st.ready.pop_front().expect("刚看过 front");
        st.ready_set.remove(&cc);
        grid.mount_chunk_tree(cc, tree, 0);
        st.detail.insert(cc, detail);
        *use_seq += 1;
        st.last_used.insert(cc, *use_seq);
        words += w;
        words_mounted += 1;
        generated += 1;
      }
      sd.mark(5);

      let seq = grid.resident_seq();
      let over_cap = grid.chunk_count() > cap;
      let need_scan = scope.moved
        || frames.is_multiple_of(SWEEP_FRAMES)
        || (over_cap && !st.trim_idle && seq != st.last_seq);
      st.last_seq = seq;
      if !need_scan {
        sd.mark(6);
        continue;
      }
      let resident: Vec<ChunkCoord> = grid.chunk_coords().collect();
      let mut out: Vec<ChunkCoord> = Vec::new();
      for c in &resident {
        if !scope.in_window(c.0) {
          out.push(*c);
        }
      }
      let over = resident.len().saturating_sub(out.len()).saturating_sub(cap);
      st.trim_idle = false;
      if over > 0 {
        let mut cand: Vec<(u64, ChunkCoord)> = Vec::new();
        for c in &resident {
          if !scope.in_window(c.0) {
            continue;
          }
          if (c.0 - scope.center).abs().max_element() <= unload_r
            || st.demand_set.contains(c)
            || st.ready_set.contains(c)
            || producer.in_flight(vol, *c)
          {
            continue;
          }
          cand.push((st.last_used.get(c).copied().unwrap_or(0), *c));
        }
        st.trim_idle = cand.is_empty();
        cand.sort_unstable_by_key(|(t, c)| (*t, c.0.x, c.0.y, c.0.z));
        out.extend(cand.into_iter().take(over).map(|(_, c)| c));
      }
      let n_unload = out.len();
      for cc in out {
        grid.unmount_chunk(cc);
        st.detail.remove(&cc);
        st.last_used.remove(&cc);
      }
      sd.mark(6);

      if generated > 0 || n_unload > 0 || !st.ready.is_empty() {
        if far_level {
          let rmax = grid
            .chunk_coords()
            .map(|c| (c.0 - scope.center).abs().max_element())
            .max()
            .unwrap_or(0);
          let (y0, y1) = grid
            .chunk_coords()
            .map(|c| c.0.y)
            .fold((i32::MAX, i32::MIN), |(lo, hi), y| (lo.min(y), hi.max(y)));
          bevy::log::debug!(
            "STREAM[v{vol} far gen{generated} unload {n_unload} chunks {} cap {cap} rmax {rmax} \
             y {}..{} 中心 {} ready {}]",
            grid.chunk_count(),
            if y0 > y1 { 0 } else { y0 },
            if y0 > y1 { 0 } else { y1 },
            scope.center.y,
            st.ready.len(),
          );
        } else {
          let (y0, y1) = grid
            .chunk_coords()
            .map(|c| c.0.y)
            .fold((i32::MAX, i32::MIN), |(lo, hi), y| (lo.min(y), hi.max(y)));
          bevy::log::debug!(
            "STREAM[v0 gen{generated}(req {gen_req}) unload {n_unload} chunks {} cap {cap} y {}..{} 相机 {} ready {}]",
            grid.chunk_count(),
            if y0 > y1 { 0 } else { y0 },
            if y0 > y1 { 0 } else { y1 },
            scope.center.y,
            st.ready.len(),
          );
        }
      }
    }
  }
  stream.palette_applied = palette_applied;
  sd.mark(7);
  sd.frame_end("STREAM");
}

fn room_hash(room: IVec3) -> u64 {
  let mut h = 0x9E37_79B9_7F4A_7C15u64;
  let keys = [0x9E37_79B1_85EB_CA87u64, 0xC2B2_AE3D_27D4_EB4Fu64, 0x1656_67B1_9E37_79F9u64];
  for (k, v) in keys.into_iter().zip([room.x, room.y, room.z]) {
    h ^= (v as i64 as u64).wrapping_mul(k);
    h = h.rotate_left(31).wrapping_mul(0x9E37_79B9_7F4A_7C15);
  }
  h ^= h >> 30;
  h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
  h ^= h >> 27;
  h.wrapping_mul(0x94D0_49BB_1331_11EB) ^ (h >> 31)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
  Pbr,
  Light,
  Glass,
  Mirror,
  Metal,
  Plain,
}

impl Kind {
  fn of(h: u64) -> Self {
    match h % 6 {
      0 => Kind::Pbr,
      1 => Kind::Light,
      2 => Kind::Glass,
      3 => Kind::Mirror,
      4 => Kind::Metal,
      _ => Kind::Plain,
    }
  }
}

const HUE_COUNT: usize = 8;
const SLOTS_PER_KIND: u16 = (HUE_COUNT * 2) as u16;
const PBR_BASE: u16 = 2;
const HUE_TABLE: [[u8; 3]; HUE_COUNT] = [
  [0xE8, 0x4C, 0x3C],
  [0xF2, 0x9E, 0x2C],
  [0xF2, 0xE2, 0x3C],
  [0x5C, 0xD6, 0x4C],
  [0x3C, 0xC8, 0xC8],
  [0x4C, 0x7C, 0xF2],
  [0x9C, 0x5C, 0xE8],
  [0xE8, 0x5C, 0xB4],
];

fn pick(h: u64, shift: u32, a: u8, b: u8) -> u8 {
  if (h >> shift) & 1 == 0 { a } else { b }
}

fn kind_base(kind: Kind, n_pbr: usize) -> u16 {
  let block = match kind {
    Kind::Pbr => 0,
    Kind::Light => 1,
    Kind::Glass => 2,
    Kind::Mirror => 3,
    Kind::Metal => 4,
    Kind::Plain => 5,
  };
  PBR_BASE + n_pbr as u16 + block * SLOTS_PER_KIND
}

pub fn slot_of(kind: Kind, hue: usize, notch: u8, asset: u16, n_pbr: usize) -> PaletteId {
  if matches!(kind, Kind::Pbr) && n_pbr > 0 {
    return PaletteId(PBR_BASE + asset.min(n_pbr as u16 - 1));
  }
  let notch = if matches!(kind, Kind::Plain | Kind::Pbr) { 0 } else { notch & 1 };
  PaletteId(kind_base(kind, n_pbr) + slot_in_kind(hue, notch))
}

#[inline]
fn slot_in_kind(hue: usize, notch: u8) -> u16 {
  (hue.min(HUE_COUNT - 1) as u16) * 2 + notch as u16
}

fn entry_of(kind: Kind, color: [u8; 3], notch: u8, asset: u16) -> PaletteEntry {
  let hi = notch != 0;
  match kind {
    Kind::Pbr => PaletteEntry::pbr(asset, PbrOverrides::default(), PaletteFlags::default()),
    Kind::Light => {
      PaletteEntry { color, emissive: if hi { 255 } else { 127 }, ..Default::default() }
    }
    Kind::Glass => {
      PaletteEntry { color, transmission: if hi { 191 } else { 127 }, ..Default::default() }
    }
    Kind::Mirror => {
      PaletteEntry { color, roughness: if hi { 127 } else { 0 }, ..Default::default() }
    }
    Kind::Metal => PaletteEntry {
      color,
      metallic: 255,
      roughness: if hi { 127 } else { 0 },
      ..Default::default()
    },
    Kind::Plain => PaletteEntry { color, ..Default::default() },
  }
}

pub fn material_slots(n_pbr: usize) -> Vec<(PaletteId, PaletteEntry)> {
  let mut out = vec![(WHITE_SLOT, PaletteEntry { color: [255, 255, 255], ..Default::default() })];
  for asset in 0..n_pbr.min(u16::MAX as usize) as u16 {
    out.push((slot_of(Kind::Pbr, 0, 0, asset, n_pbr), entry_of(Kind::Pbr, [0; 3], 0, asset)));
  }
  for kind in [Kind::Light, Kind::Glass, Kind::Mirror, Kind::Metal, Kind::Plain] {
    let notches: &[u8] = if matches!(kind, Kind::Plain) { &[0] } else { &[0, 1] };
    for (hue, &color) in HUE_TABLE.iter().enumerate() {
      for &notch in notches {
        out.push((slot_of(kind, hue, notch, 0, n_pbr), entry_of(kind, color, notch, 0)));
      }
    }
  }
  out
}

pub fn material_of(room: IVec3, n_pbr: usize) -> (PaletteId, PaletteEntry) {
  let h = room_hash(room);
  let kind = match Kind::of(h) {
    Kind::Pbr if n_pbr == 0 => Kind::Plain,
    k => k,
  };
  let hue = ((h >> 16) % HUE_COUNT as u64) as usize;
  let notch = pick(h, 8, 0, 1);
  let asset = ((h >> 4) % n_pbr.max(1) as u64).min(u16::MAX as u64) as u16;
  (slot_of(kind, hue, notch, asset, n_pbr), entry_of(kind, HUE_TABLE[hue], notch, asset))
}

fn voxel_at(p: IVec3, n_pbr: usize) -> PaletteId {
  let half = COLUMN / 2;
  let near = |v: i32| {
    let m = v.rem_euclid(ROOM);
    m < half || m >= ROOM - half
  };
  if (near(p.x) && near(p.y)) || (near(p.y) && near(p.z)) || (near(p.x) && near(p.z)) {
    return WHITE_SLOT;
  }
  let room = p.div_euclid(IVec3::splat(ROOM));
  let (cmin, cmax) = cube_range(room);
  if p.cmpge(cmin).all() && p.cmplt(cmax).all() {
    return material_of(room, n_pbr).0;
  }
  PaletteId::AIR
}

fn cube_range(room: IVec3) -> (IVec3, IVec3) {
  let min = room * ROOM + IVec3::splat((ROOM - CUBE) / 2);
  (min, min + IVec3::splat(CUBE))
}

fn fill_clipped(
  grid: &mut VolumeGrid,
  min: IVec3,
  ext: IVec3,
  grain: i32,
  lo: IVec3,
  hi: IVec3,
  palette: PaletteId,
) -> u64 {
  let a = min.max(lo);
  let b = (min + ext).min(hi);
  if b.cmple(a).any() {
    return 0;
  }
  let voxels = (grain * grain * grain) as usize;
  (fill_bricks(grid, a, b - a, grain, palette) * voxels) as u64
}

pub fn build_region(
  grid: &mut VolumeGrid,
  lo: IVec3,
  hi: IVec3,
  n_pbr: usize,
  detail: Detail,
) -> u64 {
  let chunk = gate_voxel::CHUNK_SIZE;
  debug_assert!(
    lo % chunk == IVec3::ZERO && hi % chunk == IVec3::ZERO,
    "区域须对齐 {chunk} 的 chunk（半块会被 stream_chunks 当已生成而跳过）：lo={lo} hi={hi}"
  );
  if detail == Detail::FULL {
    build_region_full(grid, lo, hi, n_pbr)
  } else {
    build_region_quantized(grid, lo, hi, n_pbr, detail.grain())
  }
}

fn build_region_full(grid: &mut VolumeGrid, lo: IVec3, hi: IVec3, n_pbr: usize) -> u64 {
  let (span, half) = (hi - lo, COLUMN / 2);
  let mut covered = 0u64;
  let first = |v: i32| (v - 1).div_euclid(ROOM);
  let last = |v: i32| (v + 1).div_euclid(ROOM);
  let edges = |a0: i32, a1: i32| (first(a0)..=last(a1)).collect::<Vec<_>>();
  for &a in &edges(lo.x, hi.x) {
    for &b in &edges(lo.y, hi.y) {
      let min = IVec3::new(a * ROOM - half, b * ROOM - half, lo.z);
      covered += fill_clipped(grid, min, IVec3::new(COLUMN, COLUMN, span.z), 4, lo, hi, WHITE_SLOT);
    }
    for &b in &edges(lo.z, hi.z) {
      let min = IVec3::new(a * ROOM - half, lo.y, b * ROOM - half);
      covered += fill_clipped(grid, min, IVec3::new(COLUMN, span.y, COLUMN), 4, lo, hi, WHITE_SLOT);
    }
  }
  for &a in &edges(lo.y, hi.y) {
    for &b in &edges(lo.z, hi.z) {
      let min = IVec3::new(lo.x, a * ROOM - half, b * ROOM - half);
      covered += fill_clipped(grid, min, IVec3::new(span.x, COLUMN, COLUMN), 4, lo, hi, WHITE_SLOT);
    }
  }
  for rx in (lo.x - ROOM).div_euclid(ROOM)..=(hi.x + ROOM).div_euclid(ROOM) {
    for ry in (lo.y - ROOM).div_euclid(ROOM)..=(hi.y + ROOM).div_euclid(ROOM) {
      for rz in (lo.z - ROOM).div_euclid(ROOM)..=(hi.z + ROOM).div_euclid(ROOM) {
        let room = IVec3::new(rx, ry, rz);
        let (cmin, cmax) = cube_range(room);
        if cmax.cmple(lo).any() || cmin.cmpge(hi).any() {
          continue;
        }
        let id = material_of(room, n_pbr).0;
        covered += fill_clipped(grid, cmin, IVec3::splat(CUBE), 4, lo, hi, id);
      }
    }
  }
  covered
}

fn near_frac(a: i32, b: i32) -> f32 {
  let half = COLUMN / 2;
  let mut acc = 0i32;
  for k in (a - half).div_euclid(ROOM)..=(b + half).div_euclid(ROOM) {
    let lo = (k * ROOM - half).max(a);
    let hi = (k * ROOM + half).min(b);
    if hi > lo {
      acc += hi - lo;
    }
  }
  acc as f32 / (b - a) as f32
}

fn build_region_quantized(
  grid: &mut VolumeGrid,
  lo: IVec3,
  hi: IVec3,
  n_pbr: usize,
  grain: i32,
) -> u64 {
  let span = hi - lo;
  let n = (span / grain).to_array().map(|v| v as usize);
  let axis = |base: i32, count: usize| {
    (0..count)
      .map(|i| near_frac(base + i as i32 * grain, base + (i as i32 + 1) * grain))
      .collect::<Vec<_>>()
  };
  let (fx, fy, fz) = (axis(lo.x, n[0]), axis(lo.y, n[1]), axis(lo.z, n[2]));
  let vol = (grain * grain * grain) as f32;
  let off = (ROOM - CUBE) / 2;
  let mut covered = 0u64;
  let mut memo: Option<(IVec3, PaletteId)> = None;
  for (kz, &c) in fz.iter().enumerate() {
    for (ky, &b) in fy.iter().enumerate() {
      for (kx, &a) in fx.iter().enumerate() {
        let cell =
          IVec3::new(lo.x + kx as i32 * grain, lo.y + ky as i32 * grain, lo.z + kz as i32 * grain);
        let white = (a * b + b * c + a * c - 2.0 * a * b * c) * vol;
        let mut best: Option<(f32, PaletteId)> =
          if white > 0.5 { Some((white, WHITE_SLOT)) } else { None };
        let end = cell + IVec3::splat(grain);
        for rx in (end.x - off - CUBE).div_euclid(ROOM)..=(cell.x - off).div_euclid(ROOM) {
          for ry in (end.y - off - CUBE).div_euclid(ROOM)..=(cell.y - off).div_euclid(ROOM) {
            for rz in (end.z - off - CUBE).div_euclid(ROOM)..=(cell.z - off).div_euclid(ROOM) {
              let room = IVec3::new(rx, ry, rz);
              let (cmin, cmax) = cube_range(room);
              let a0 = cmin.max(cell);
              let b0 = cmax.min(end);
              if b0.cmple(a0).any() {
                continue;
              }
              let d = b0 - a0;
              let v = (d.x * d.y * d.z) as f32;
              let id = match memo {
                Some((r, id)) if r == room => id,
                _ => {
                  let id = material_of(room, n_pbr).0;
                  memo = Some((room, id));
                  id
                }
              };
              let better = match best {
                Some((bv, bid)) => v > bv || (v == bv && id < bid),
                None => true,
              };
              if better {
                best = Some((v, id));
              }
            }
          }
        }
        if let Some((_, id)) = best {
          covered += fill_clipped(grid, cell, IVec3::splat(grain), grain, lo, hi, id);
        }
      }
    }
  }
  covered
}

pub fn build_region_far(
  grid: &mut VolumeGrid,
  lo: IVec3,
  hi: IVec3,
  n_pbr: usize,
  scale: i32,
  grain: i32,
) -> u64 {
  let chunk = gate_voxel::CHUNK_SIZE;
  debug_assert!(
    lo % chunk == IVec3::ZERO && hi % chunk == IVec3::ZERO,
    "远场区域须对齐 {chunk} 的 chunk：lo={lo} hi={hi}"
  );
  debug_assert!(grain as i64 * scale as i64 > 0, "级体素与格宽必须为正");
  let half = grain / 2;
  let voxels = (grain * grain * grain) as u64;
  let mut covered = 0u64;
  let mut z = lo.z;
  while z < hi.z {
    let mut y = lo.y;
    while y < hi.y {
      let mut x = lo.x;
      while x < hi.x {
        let c = IVec3::new(x + half, y + half, z + half) * scale;
        let id = voxel_at(c, n_pbr);
        if !id.is_air() {
          covered +=
            fill_bricks(grid, IVec3::new(x, y, z), IVec3::splat(grain), grain, id) as u64 * voxels;
        }
        x += grain;
      }
      y += grain;
    }
    z += grain;
  }
  covered
}

#[derive(Clone, Copy)]
struct GenScope {
  center: IVec3,
  w_origin: IVec3,
  w_dims: IVec3,
  load_radius: i32,
  coarse_radius: i32,
  coarse_height: i32,
  forward: Vec3,
}

const VIEW_COS: f32 = 0.5;

pub(crate) const SEAM_MARGIN: i32 = 2;

const FAR_PRELOAD_INNER: i32 = crate::mc::NEAR_COVER_CHUNKS / 4 - 2;

const PRELOAD_DETAIL_CAP: Detail = Detail::FINE;

const PRELOAD_FULL_CHUNKS: i32 = 8;

const FAR_PRELOAD_OUTER: i32 = 30;

const FAR_PRELOAD_HY: i32 = 8;

fn far_cap(vol: usize, cap: usize) -> usize {
  let _ = vol;
  cap
}

const FAR_CONTENT_LAYERS: [i32; 3] = [3, 1, 1];

fn far_radius_ladder(cap: usize) -> [(i32, i32); 3] {
  let mut out = [(0, 0); 3];
  let mut r_in = FAR_PRELOAD_INNER;
  for k in 0..FAR_SCALES.len() {
    let budget = far_cap(k + 1, cap) as u64;
    let layers = FAR_CONTENT_LAYERS[k].max(1) as u64;
    let per_ring = |r: i32| 8 * layers * r.max(0) as u64;
    let (mut r_out, mut sum) = (r_in - 1, 0u64);
    while r_out < FAR_PRELOAD_OUTER && sum + per_ring(r_out + 1) <= budget {
      r_out += 1;
      sum += per_ring(r_out);
    }
    out[k] = (r_in, r_out);
    r_in = ((r_out - SEAM_MARGIN + 1) / 4 - 1).max(1);
  }
  out
}

fn plan_key(c: IVec3, center: IVec3, forward: Vec3) -> (i32, i32, i32, i32, i32) {
  let d = c - center;
  let off = if forward == Vec3::ZERO || d == IVec3::ZERO {
    0
  } else {
    (d.as_vec3().normalize_or_zero().dot(forward) <= VIEW_COS) as i32
  };
  (d.length_squared(), off, c.x, c.y, c.z)
}

fn preload_detail(c: IVec3, center: IVec3) -> Detail {
  let d = detail_at((c - center).abs().max_element());
  if (c - center).abs().max_element() <= PRELOAD_FULL_CHUNKS {
    d
  } else {
    Detail(d.0.min(PRELOAD_DETAIL_CAP.0))
  }
}

fn detail_at(dist_chunks: i32) -> Detail {
  const CHUNKS_PER_GRAIN: f32 = 3.65;
  let allowed = dist_chunks as f32 / CHUNKS_PER_GRAIN;
  if allowed >= 256.0 {
    Detail::CHUNK
  } else if allowed >= 64.0 {
    Detail::WIDE
  } else if allowed >= 16.0 {
    Detail::COARSE
  } else if allowed >= 4.0 {
    Detail::FINE
  } else {
    Detail::FULL
  }
}

fn detail_of_req(level: u8) -> Detail {
  match level {
    0 => Detail::FULL,
    1 => Detail::FINE,
    2 => Detail::WIDE,
    _ => Detail::CHUNK,
  }
}

fn plan_generation(
  scope: GenScope,
  cap: usize,
  requests: &[gate_render::LodRequest],
  have: impl Fn(IVec3, Detail) -> bool,
) -> (Vec<(IVec3, Detail)>, usize) {
  let GenScope { center, w_origin, w_dims, load_radius, coarse_radius, coarse_height, forward } =
    scope;
  let in_window = |c: IVec3| {
    let t = c - w_origin;
    t.cmpge(IVec3::ZERO).all() && t.cmplt(w_dims).all()
  };
  let mut picked: Vec<(IVec3, Detail)> = Vec::new();
  let mut requested: Vec<gate_render::LodRequest> =
    requests.iter().filter(|r| r.vol == 0 && in_window(r.chunk)).copied().collect();
  requested.sort_unstable_by_key(|r| (std::cmp::Reverse(r.votes), r.chunk.x, r.chunk.y, r.chunk.z));
  picked.extend(
    requested
      .into_iter()
      .filter(|r| !have(r.chunk, detail_of_req(r.level)))
      .map(|r| (r.chunk, detail_of_req(r.level))),
  );
  let from_req = picked.len();
  let budget = cap.saturating_sub(from_req);
  if budget > 0 {
    let r = coarse_radius.max(load_radius);
    let h = coarse_height;
    let (nx, ny, nz) = ((2 * r + 1) as usize, (2 * h + 1) as usize, (2 * r + 1) as usize);
    let mut seen = vec![0u64; (nx * ny * nz).div_ceil(64)];
    let bit_of = |c: IVec3| -> Option<(usize, u64)> {
      let d = c - center;
      if d.x.abs() > r || d.y.abs() > h || d.z.abs() > r {
        return None;
      }
      let i = ((d.x + r) as usize * ny + (d.y + h) as usize) * nz + (d.z + r) as usize;
      Some((i >> 6, 1u64 << (i & 63)))
    };
    for (c, _) in &picked {
      if let Some((w, m)) = bit_of(*c) {
        seen[w] |= m;
      }
    }
    let detail_for = |c: IVec3| preload_detail(c, center);
    let mut todo: Vec<IVec3> = Vec::new();
    let mut push = |c: IVec3, todo: &mut Vec<IVec3>| {
      if !in_window(c) || have(c, detail_for(c)) {
        return;
      }
      if let Some((w, m)) = bit_of(c) {
        if seen[w] & m != 0 {
          return;
        }
        seen[w] |= m;
      }
      todo.push(c);
    };
    for dx in -load_radius..=load_radius {
      for dz in -load_radius..=load_radius {
        for dy in -coarse_height..=coarse_height {
          push(center + IVec3::new(dx, dy, dz), &mut todo);
        }
      }
    }
    let r2 = coarse_radius * coarse_radius;
    for dx in -coarse_radius..=coarse_radius {
      for dz in -coarse_radius..=coarse_radius {
        if dx * dx + dz * dz > r2 {
          continue;
        }
        if dx.abs().max(dz.abs()) <= load_radius {
          continue;
        }
        for dy in -coarse_height..=coarse_height {
          push(center + IVec3::new(dx, dy, dz), &mut todo);
        }
      }
    }
    todo.sort_unstable_by_key(|c| plan_key(*c, center, forward));
    picked.extend(todo.into_iter().take(budget).map(|c| (c, detail_for(c))));
  }
  (picked, from_req)
}

fn plan_generation_far(
  vol: usize,
  scope: VolScope,
  cap: usize,
  r_in: i32,
  r_out: i32,
  requests: &[gate_render::LodRequest],
  have: impl Fn(IVec3, Detail) -> bool,
) -> Vec<(IVec3, Detail)> {
  let mut req: Vec<gate_render::LodRequest> = requests
    .iter()
    .filter(|r| r.vol as usize == vol && scope.in_window(r.chunk))
    .copied()
    .collect();
  req.sort_unstable_by_key(|r| (std::cmp::Reverse(r.votes), r.chunk.x, r.chunk.y, r.chunk.z));
  let mut out: Vec<(IVec3, Detail)> = req
    .into_iter()
    .filter(|r| !have(r.chunk, Detail::FULL))
    .map(|r| (r.chunk, Detail::FULL))
    .collect();
  let budget = cap.saturating_sub(out.len());
  if budget == 0 {
    return out;
  }
  let mut seen: std::collections::HashSet<IVec3> = out.iter().map(|(c, _)| *c).collect();
  let mut todo: Vec<IVec3> = Vec::new();
  let outer = r_out.min(scope.w_dims.min_element() / 2);
  let (lo2, hi2) = {
    let r = r_in.min(outer);
    (r * r, outer * outer)
  };
  'shell: for dx in -outer..=outer {
    for dz in -outer..=outer {
      let d2 = dx * dx + dz * dz;
      if d2 < lo2 || d2 > hi2 {
        continue;
      }
      for dy in -scope.v_hy..=scope.v_hy {
        let c = scope.center + IVec3::new(dx, dy, dz);
        if scope.in_window(c) && !have(c, Detail::FULL) && seen.insert(c) {
          todo.push(c);
        }
      }
      if todo.len() >= budget.saturating_mul(4) {
        break 'shell;
      }
    }
  }
  todo.sort_unstable_by_key(|c| {
    let d = *c - scope.center;
    (d.length_squared(), c.x, c.y, c.z)
  });
  out.extend(todo.into_iter().take(budget).map(|c| (c, Detail::FULL)));
  out
}
