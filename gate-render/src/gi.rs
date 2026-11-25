use bevy::render::render_resource::{BindGroup, BindGroupLayoutDescriptor, Buffer, ShaderType};
use glam::{Mat4, UVec2, UVec4, Vec3, Vec4};

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GiUniform {
  pub params: Vec4,

  pub misc: Vec4,

  pub flags: Vec4,

  pub seq: UVec4,

  pub prev_view_proj: Mat4,
}

#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq)]
pub struct GiSettings {
  pub enabled: bool,

  pub gi_div: u32,

  pub emit_gain: f32,

  pub cell_px: f32,

  pub wal: bool,

  pub nee: bool,
}

impl GiSettings {
  pub const DIV_CHOICES: [u32; 4] = [1, 2, 4, 8];

  pub const CELL_CHOICES: [f32; 4] = [1.0, 2.0, 4.0, 8.0];

  pub const GAIN_CHOICES: [f32; 5] = [0.0, 2.0, 4.0, 8.0, 16.0];

  pub const HIST_CAP: f32 = 32768.0;

  pub const HIST_CAP_RESET: f32 = 48.0;

  pub const EVICT_FRAMES: u32 = 900;

  pub const FORGET: f32 = 0.9989;

  pub const GI_SIZE_REF_PX: u64 = 320 * 180;

  pub const IRC_SLOTS_MIN: u32 = 1 << 19;

  pub const IRC_SLOTS_MAX: u32 = 1 << 20;

  pub fn div(&self) -> u32 {
    self.gi_div.clamp(1, 8)
  }

  pub fn cell_px(&self) -> f32 {
    self.cell_px.clamp(0.5, 16.0)
  }

  pub fn gain(&self) -> f32 {
    self.emit_gain.clamp(0.0, 64.0)
  }

  pub fn gi_size(&self, render_size: UVec2) -> UVec2 {
    let d = self.div();
    UVec2::new((render_size.x / d).max(1), (render_size.y / d).max(1))
  }

  pub fn irc_slots(&self, render_size: UVec2) -> u32 {
    let gi = self.gi_size(render_size);
    let px = gi.x as u64 * gi.y as u64;
    let k = (px / Self::GI_SIZE_REF_PX).max(1);
    (Self::IRC_SLOTS_MIN as u64 * k).min(Self::IRC_SLOTS_MAX as u64) as u32
  }
}

impl Default for GiSettings {
  fn default() -> Self {
    Self { enabled: true, gi_div: 4, emit_gain: 8.0, cell_px: 2.0, wal: false, nee: true }
  }
}

pub const LT_SLOTS: u64 = 1 << 17;

pub fn gi_bg0_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  BindGroupLayoutDescriptor::new(
    "GiBg0",
    &[BindGroupLayoutEntry {
      binding: 0,
      visibility: C,
      ty: BindingType::Buffer {
        ty: BufferBindingType::Uniform,
        has_dynamic_offset: false,
        min_binding_size: Some(GiUniform::min_size()),
      },
      count: None,
    }],
  )
}

pub fn gi_bg5_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let buf = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: false },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "GiBg5",
    &[buf(0), buf(1), buf(2), buf(3), buf(4), buf(5), buf(6), buf(7), buf(8), buf(9), buf(10)],
  )
}

pub const GI_MERGE_SLOTS: u32 = 8192;
pub const GI_MERGE_WORDS: u32 = 3;

#[derive(bevy::ecs::resource::Resource)]
pub struct GiGpu {
  pub uniform: bevy::render::render_resource::UniformBuffer<GiUniform>,

  pub frame: u32,

  pub hist_cap: f32,

  pub bg0: Option<BindGroup>,

  pub bg5: Option<BindGroup>,

  pub stats: Option<Buffer>,

  pub gi_merge: Option<Buffer>,

  pub gather_pipeline: Option<crate::brickmap::dda::ComputePipe>,

  pub resolve_pipeline: Option<crate::brickmap::dda::ComputePipe>,

  pub light_pipeline: Option<crate::brickmap::dda::ComputePipe>,

  pub restir_temporal_pipeline: Option<crate::brickmap::dda::ComputePipe>,

  pub restir_spatial_pipeline: Option<crate::brickmap::dda::ComputePipe>,

  pub merge_pipeline: Option<crate::brickmap::dda::ComputePipe>,

  irc: Option<Buffer>,

  irc_key: Option<Buffer>,

  lt_tile: Option<Buffer>,

  lt_tile_key: Option<Buffer>,

  lt_node: Option<Buffer>,

  lt_node_key: Option<Buffer>,

  res_prev: Option<Buffer>,

  res_tmp: Option<Buffer>,

  res_cur: Option<Buffer>,

  gi_size: UVec2,

  res_bytes: u64,

  slots: u32,

  slots_tag: u64,

  prev_track: Option<Mat4>,

  light_rev: u32,

  light_ref: LightRef,
}

impl GiGpu {
  pub fn irc_bind_group(&self) -> Option<&BindGroup> {
    self.bg5.as_ref()
  }

  pub fn stats_buffer(&self) -> Option<&Buffer> {
    self.stats.as_ref()
  }

  pub fn slots(&self) -> u32 {
    self.slots.max(1)
  }

  pub fn reservoir_copy(&self) -> Option<(&Buffer, &Buffer, u64)> {
    Some((self.res_cur.as_ref()?, self.res_prev.as_ref()?, self.res_bytes))
  }

  pub fn clear(&mut self) {
    self.irc = None;
    self.irc_key = None;
    self.lt_tile = None;
    self.lt_tile_key = None;
    self.lt_node = None;
    self.lt_node_key = None;
    self.res_prev = None;
    self.res_tmp = None;
    self.res_cur = None;
    self.gi_size = UVec2::ZERO;
    self.res_bytes = 0;
    self.slots = 0;
    self.slots_tag = 0;
  }
}

pub struct GiPlugin;

impl bevy::app::Plugin for GiPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    use bevy::ecs::schedule::IntoScheduleConfigs;
    app.init_resource::<GiSettings>();
    let Some(render_app) = app.get_sub_app_mut(bevy::render::RenderApp) else {
      return;
    };
    render_app
      .init_resource::<GiSettings>()
      .add_systems(
        bevy::render::RenderStartup,
        queue_gi_pipelines
          .after(crate::brickmap::dda::init_dda_pipelines)
          .after(crate::pipe_cache::init_gate_pipe_cache),
      )
      .add_systems(bevy::render::ExtractSchedule, extract_gi_settings)
      .add_systems(
        bevy::render::Render,
        prepare_gi
          .in_set(bevy::render::RenderSystems::PrepareBindGroups)
          .after(crate::brickmap::upload::prepare)
          .after(crate::brickmap::dda::prepare_dda_bind_groups),
      );
  }
}

fn queue_gi_pipelines(
  render_device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  dda_shader: bevy::ecs::system::Res<crate::shader::DdaShaderHandle>,
  gate: bevy::ecs::system::Res<crate::pipe_cache::GatePipeCache>,
  dda: bevy::ecs::system::Res<crate::brickmap::dda::DdaPipelines>,
  mut commands: bevy::ecs::system::Commands,
) {
  use bevy::render::render_resource::ComputePipelineDescriptor;
  use std::borrow::Cow;

  let layouts = vec![
    dda.bg0_gi_layout.clone(),
    dda.bg1_layout.clone(),
    dda.bg2_layout.clone(),
    dda.bg3_layout.clone(),
    gi_bg0_layout(),
    gi_bg5_layout(),
  ];
  let dda_shader = dda_shader.0.clone();
  let pipe = |label: &'static str, entry: &'static str| -> crate::brickmap::dda::ComputePipe {
    if let Some(raw) = crate::brickmap::dda::raw_compute_pipe(
      &render_device,
      &pipeline_cache,
      &gate,
      label,
      &layouts,
      entry,
    ) {
      return raw;
    }
    crate::brickmap::dda::ComputePipe::Bevy(pipeline_cache.queue_compute_pipeline(
      ComputePipelineDescriptor {
        label: Some(Cow::from(label)),
        layout: layouts.clone(),
        shader: dda_shader.clone(),
        entry_point: Some(Cow::from(entry)),
        ..Default::default()
      },
    ))
  };
  let gather = pipe("gate_gi_gather", "gi_gather");
  let resolve = pipe("gate_gi_resolve", "gi_resolve");
  let light = pipe("gate_gi_light", "gi_light_sweep");
  let restir_temporal = pipe("gate_gi_restir_temporal", "gi_restir_temporal");
  let restir_spatial = pipe("gate_gi_restir_spatial", "gi_restir_spatial");
  let merge_drain = pipe("gate_gi_merge_drain", "gi_merge_drain");

  commands.insert_resource(GiGpu {
    uniform: bevy::render::render_resource::UniformBuffer::default(),
    frame: 0,
    hist_cap: GiSettings::HIST_CAP,
    bg0: None,
    bg5: None,
    stats: None,
    gi_merge: None,
    gather_pipeline: Some(gather),
    resolve_pipeline: Some(resolve),
    light_pipeline: Some(light),
    restir_temporal_pipeline: Some(restir_temporal),
    restir_spatial_pipeline: Some(restir_spatial),
    merge_pipeline: Some(merge_drain),
    irc: None,
    irc_key: None,
    lt_tile: None,
    lt_tile_key: None,
    lt_node: None,
    lt_node_key: None,
    res_prev: None,
    res_tmp: None,
    res_cur: None,
    gi_size: UVec2::ZERO,
    res_bytes: 0,
    slots: 0,
    slots_tag: 0,
    prev_track: None,
    light_rev: 0,
    light_ref: LightRef::default(),
  });
}

fn extract_gi_settings(
  mut commands: bevy::ecs::system::Commands,
  settings: Option<bevy::render::Extract<bevy::ecs::system::Res<GiSettings>>>,
) {
  commands.insert_resource(settings.map_or_else(GiSettings::default, |s| GiSettings {
    enabled: s.enabled,
    gi_div: s.div(),
    emit_gain: s.gain(),
    cell_px: s.cell_px(),
    wal: s.wal,
    nee: s.nee,
  }));
}

#[derive(bevy::ecs::system::SystemParam)]
struct PrepareGiParams<'w, 's> {
  commands: bevy::ecs::system::Commands<'w, 's>,
  device: bevy::ecs::system::Res<'w, bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<'w, bevy::render::renderer::RenderQueue>,
  pipeline_cache: bevy::ecs::system::Res<'w, bevy::render::render_resource::PipelineCache>,
  settings: bevy::ecs::system::Res<'w, GiSettings>,
  view: Option<bevy::ecs::system::Res<'w, crate::brickmap::dda::DdaViewUniform>>,
  lighting: Option<bevy::ecs::system::Res<'w, crate::lighting::LightingTheme>>,
  dirty: Option<bevy::ecs::system::Res<'w, crate::brickmap::upload::BrickMapDirty>>,
  scale: Option<bevy::ecs::system::Res<'w, crate::brickmap::dda::RenderScale>>,
  gpu: bevy::ecs::system::ResMut<'w, GiGpu>,
}

fn prepare_gi(p: PrepareGiParams) {
  use bevy::render::render_resource::*;
  let PrepareGiParams {
    mut commands,
    device,
    queue,
    pipeline_cache,
    settings,
    view,
    lighting,
    dirty,
    scale,
    mut gpu,
  } = p;
  gpu.frame = gpu.frame.wrapping_add(1);

  let render_size = scale.as_deref().map_or(crate::consts::VIEW_SIZE, |s| s.size);
  let slots = settings.irc_slots(render_size);
  let slots_tag = slots as u64 | ((settings.div() as u64) << 40);
  let gi_size = settings.gi_size(render_size);
  if gpu.gi_size != gi_size || gpu.res_prev.is_none() {
    let c = crate::wesl_consts::gi_consts();
    let bytes = (gi_size.x as u64 * gi_size.y as u64 * c.res_words as u64 * 4).max(4);
    let make = |label: &str, bytes: u64| {
      device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
      })
    };
    gpu.res_prev = Some(make("gate_gi_res_prev", bytes));
    gpu.res_tmp = Some(make("gate_gi_res_tmp", bytes));
    gpu.res_cur = Some(make("gate_gi_res_cur", bytes));
    gpu.gi_size = gi_size;
    gpu.res_bytes = bytes;
    bevy::log::debug!(
      target: "gate",
      "GI reservoir → {}x{} × {} word（{:.1} MB ×3）",
      gi_size.x,
      gi_size.y,
      c.res_words,
      bytes as f64 / 1048576.0,
    );
  }
  if gpu.slots_tag != slots_tag {
    let c = crate::wesl_consts::gi_consts();
    let make = |label: &str, bytes: u64| {
      device.create_buffer(&BufferDescriptor {
        label: Some(label),
        size: bytes.max(4),
        usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
        mapped_at_creation: false,
      })
    };
    if gpu.slots_tag & 0xFF_FFFF != slots_tag & 0xFF_FFFF || gpu.irc.is_none() {
      gpu.irc = Some(make("gate_gi_irc", slots as u64 * c.irc_words as u64 * 4));
      gpu.irc_key = Some(make("gate_gi_irc_key", slots as u64 * 2 * 4));
    }
    if gpu.lt_tile.is_none() {
      gpu.lt_tile = Some(make("gate_gi_lt_tile", LT_SLOTS * c.lt_tile_words as u64 * 4));
      gpu.lt_tile_key = Some(make("gate_gi_lt_tile_key", LT_SLOTS * 2 * 4));
      gpu.lt_node = Some(make("gate_gi_lt_node", LT_SLOTS * c.lt_node_words as u64 * 4));
      gpu.lt_node_key = Some(make("gate_gi_lt_node_key", LT_SLOTS * 2 * 4));
    }
    gpu.stats = Some(make("gate_gi_stats", c.gi_stat_words as u64 * 4));
    let merge = make("gate_gi_merge_req", (1 + GI_MERGE_SLOTS as u64 * GI_MERGE_WORDS as u64) * 4);
    queue.write_buffer(&merge, 0, &0u32.to_ne_bytes());
    gpu.gi_merge = Some(merge);
    gpu.slots = slots;
    gpu.slots_tag = slots_tag;
    bevy::log::debug!(
      target: "gate",
      "GI 世界缓存 → {} 槽（{:.1} MB 键 {:.1} MB）| 光源瓦片 {} 槽 | GI 1/{}",
      slots,
      slots as f64 * c.irc_words as f64 * 4.0 / 1048576.0,
      slots as f64 * 2.0 * 4.0 / 1048576.0,
      LT_SLOTS,
      settings.div(),
    );
  }

  let track = view.as_deref().map(|v| v.view_proj);
  let prev_track = gpu.prev_track;
  let moved = match (prev_track, track) {
    (Some(a), Some(b)) => (a - b).to_cols_array().iter().any(|x| x.abs() > 1e-4),
    (None, Some(_)) => true,
    _ => false,
  };
  gpu.prev_track = track;

  let mut light_jump = gpu.light_ref.track(lighting.as_deref(), settings.emit_gain);
  if light_jump {
    gpu.light_rev = gpu.light_rev.wrapping_add(1);
  }
  let edited = dirty.as_ref().is_some_and(|d| d.edited || d.full || d.palette_changed);
  if edited {
    gpu.light_rev = gpu.light_rev.wrapping_add(1);
    light_jump = true;
  }
  let cap = if light_jump || edited { GiSettings::HIST_CAP_RESET } else { GiSettings::HIST_CAP };
  gpu.hist_cap = cap;

  *gpu.uniform.get_mut() = GiUniform {
    params: Vec4::new(
      settings.gain(),
      if settings.enabled { settings.cell_px() } else { 0.0 },
      1.0,
      cap,
    ),
    misc: Vec4::new(
      if settings.enabled { 1.0 } else { 0.0 },
      if settings.wal { 1.0 } else { 0.0 },
      GiSettings::FORGET,
      if settings.nee { 1.0 } else { 0.0 },
    ),
    flags: Vec4::new(
      if settings.nee { 1.0 } else { 0.0 },
      settings.div() as f32,
      if light_jump { 1.0 } else { 0.0 },
      if moved { 1.0 } else { 0.0 },
    ),
    seq: UVec4::new(gpu.frame, GiSettings::EVICT_FRAMES, gpu.light_rev, light_slots()),
    prev_view_proj: prev_track.unwrap_or_else(|| track.unwrap_or(Mat4::IDENTITY)),
  };
  gpu.uniform.write_buffer(&device, &queue);

  let bg0_layout = pipeline_cache.get_bind_group_layout(&gi_bg0_layout());
  let bg0 = device.create_bind_group(
    None,
    &bg0_layout,
    &[BindGroupEntry { binding: 0, resource: gpu.uniform.binding().expect("uniform 已写入") }],
  );
  gpu.bg0 = Some(bg0);

  let bg5_layout = pipeline_cache.get_bind_group_layout(&gi_bg5_layout());
  let (
    Some(irc),
    Some(irc_key),
    Some(tile),
    Some(tile_key),
    Some(node),
    Some(node_key),
    Some(st),
    Some(gm),
    Some(rprev),
    Some(rtmp),
    Some(rcur),
  ) = (
    gpu.irc.as_ref(),
    gpu.irc_key.as_ref(),
    gpu.lt_tile.as_ref(),
    gpu.lt_tile_key.as_ref(),
    gpu.lt_node.as_ref(),
    gpu.lt_node_key.as_ref(),
    gpu.stats.as_ref(),
    gpu.gi_merge.as_ref(),
    gpu.res_prev.as_ref(),
    gpu.res_tmp.as_ref(),
    gpu.res_cur.as_ref(),
  )
  else {
    return;
  };
  let bg5 = device.create_bind_group(
    None,
    &bg5_layout,
    &[
      BindGroupEntry { binding: 0, resource: irc.as_entire_binding() },
      BindGroupEntry { binding: 1, resource: irc_key.as_entire_binding() },
      BindGroupEntry { binding: 2, resource: tile.as_entire_binding() },
      BindGroupEntry { binding: 3, resource: tile_key.as_entire_binding() },
      BindGroupEntry { binding: 4, resource: node.as_entire_binding() },
      BindGroupEntry { binding: 5, resource: node_key.as_entire_binding() },
      BindGroupEntry { binding: 6, resource: st.as_entire_binding() },
      BindGroupEntry { binding: 7, resource: rprev.as_entire_binding() },
      BindGroupEntry { binding: 8, resource: rtmp.as_entire_binding() },
      BindGroupEntry { binding: 9, resource: rcur.as_entire_binding() },
      BindGroupEntry { binding: 10, resource: gm.as_entire_binding() },
    ],
  );
  gpu.bg5 = Some(bg5);
  let _ = &mut commands;
}

fn light_slots() -> u32 {
  LT_SLOTS as u32
}

const LIGHT_JUMP_DIR: f32 = 6.0e-4;
const LIGHT_JUMP_REL: f32 = 0.05;
const LIGHT_JUMP_ABS: f32 = 0.02;

fn light_scalar_jump(x: f32, r: f32) -> bool {
  (x - r).abs() > LIGHT_JUMP_REL * r.abs().max(LIGHT_JUMP_ABS)
}

fn light_vec_jump(x: [f32; 3], r: [f32; 3]) -> bool {
  x.iter().zip(r).any(|(a, b)| light_scalar_jump(*a, b))
}

#[derive(Default)]
struct LightRef {
  filled: bool,
  has_sun: bool,
  dir: Vec3,
  sun_color: [f32; 3],
  intensity: f32,
  ambient: [f32; 3],
  sky_color: Option<[f32; 3]>,
  emit_gain: f32,
}

impl LightRef {
  fn track(&mut self, theme: Option<&crate::lighting::LightingTheme>, emit_gain: f32) -> bool {
    let Some(t) = theme else {
      self.filled = false;
      return false;
    };
    let sun = t.sun.as_ref().map(|s| {
      let d = Vec3::from(s.dir);
      (d / d.length().max(1e-6), s.color, s.intensity)
    });
    let mut jump = false;
    if self.filled {
      let sun_jump = match (sun, self.has_sun) {
        (Some((d, c, i)), true) => {
          1.0 - d.dot(self.dir) > LIGHT_JUMP_DIR
            || light_vec_jump(c, self.sun_color)
            || light_scalar_jump(i, self.intensity)
        }
        (None, false) => false,
        _ => true,
      };
      let sky_jump = match (t.sky.as_ref().map(|s| s.color), self.sky_color) {
        (Some(c), Some(r)) => light_vec_jump(c, r),
        (None, None) => false,
        _ => true,
      };
      jump = sun_jump
        || sky_jump
        || light_vec_jump(t.ambient, self.ambient)
        || light_scalar_jump(emit_gain, self.emit_gain);
    }
    self.filled = true;
    self.has_sun = sun.is_some();
    if let Some((d, c, i)) = sun {
      self.dir = d;
      self.sun_color = c;
      self.intensity = i;
    }
    self.ambient = t.ambient;
    self.sky_color = t.sky.as_ref().map(|s| s.color);
    self.emit_gain = emit_gain;
    jump
  }
}
