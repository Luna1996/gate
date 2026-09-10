use bevy::render::render_resource::ShaderType;
use glam::{IVec3, IVec4, UVec3, Vec3, Vec4};

pub const IRRADIANCE_TEXELS: u32 = 8;
pub const DEPTH_TEXELS: u32 = 16;
pub const PROBE_T_MAX: f32 = 8192.0;
pub const DDGI_RAY_BUDGET: u32 = 65536;
pub const DDGI_PROBE_BUDGET: u32 = 4096;

pub const PROBES_PER_LAYER_AXIS: u32 = 16;
pub const PROBES_PER_LAYER: u32 = PROBES_PER_LAYER_AXIS * PROBES_PER_LAYER_AXIS;
pub const IRRADIANCE_LAYER_TEXELS: u32 = PROBES_PER_LAYER_AXIS * IRRADIANCE_TEXELS;
pub const DEPTH_LAYER_TEXELS: u32 = PROBES_PER_LAYER_AXIS * DEPTH_TEXELS;
pub const DDGI_LODS: u32 = 4;
pub const META_AGE_SHIFT: u32 = 24;
// indirect args / 计数器合一 buffer（word 布局与 WGSL DDGI_INDIR_* 对应）：
//   [0..16) cast args ×4 LOD；[16..32) collect args ×4 LOD；[32..36) rpp；[36..40) 活跃计数器
pub const DDGI_INDIRECT_BYTES: u64 = 256; // 256B 对齐
pub const DDGI_COUNTER_CLEAR_OFFSET: u64 = 36 * 4;
pub const DDGI_COUNTER_CLEAR_BYTES: u64 = 16;
pub const DDGI_WORKLIST_ITEM_BYTES: u64 = 16; // vec4(probe_pos.xyz, packed age|lod)

pub const DDGI_LOD_CELL_SIZES: [i32; DDGI_LODS as usize] = [16, 32, 64, 128];
pub const PROBES_PER_CASCADE_AXIS: u32 = 16;
pub const SLOTS_PER_LOD: u32 =
  PROBES_PER_CASCADE_AXIS * PROBES_PER_CASCADE_AXIS * PROBES_PER_CASCADE_AXIS;
pub const DDGI_TOTAL_SLOTS: u32 = SLOTS_PER_LOD * DDGI_LODS;
pub const DDGI_TOTAL_LAYERS: u32 = DDGI_TOTAL_SLOTS / PROBES_PER_LAYER;

#[inline]
pub fn ddgi_slot(lod: u32, cell: UVec3) -> u32 {
  let a = PROBES_PER_CASCADE_AXIS;
  lod * SLOTS_PER_LOD + cell.x + cell.y * a + cell.z * a * a
}

pub const DDGI_WINDOW_AXIS: i32 = PROBES_PER_CASCADE_AXIS as i32;
pub const DDGI_SCROLL_STEP: i32 = 8;
pub const DDGI_BAND_LO: i32 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct LodWindow {
  pub cell_size: i32,
  pub origin: IVec3,
  pub prev_origin: IVec3,
  pub reuse_min: IVec3,
  pub reuse_max: IVec3,
}

#[inline]
pub fn lod_window_origin(cam_fine: Vec3, cell_size: i32) -> IVec3 {
  let cam_cell = (cam_fine / cell_size as f32).floor().as_ivec3();
  let f = |c: i32| c.div_euclid(DDGI_SCROLL_STEP) * DDGI_SCROLL_STEP - DDGI_BAND_LO;
  IVec3::new(f(cam_cell.x), f(cam_cell.y), f(cam_cell.z))
}

#[inline]
pub fn compute_lod_window(
  cam_fine: Vec3,
  cell_size: i32,
  prev_origin: IVec3,
  have_prev: bool,
) -> LodWindow {
  let origin = lod_window_origin(cam_fine, cell_size);
  let n = DDGI_WINDOW_AXIS;
  let (reuse_min, reuse_max) = if have_prev {
    let lo = origin.max(prev_origin);
    let hi = (origin + n).min(prev_origin + n);
    (lo, hi)
  } else {
    (origin, origin)
  };
  LodWindow {
    cell_size,
    origin,
    prev_origin,
    reuse_min,
    reuse_max,
  }
}

use bevy::asset::AssetServer;
use bevy::ecs::schedule::IntoScheduleConfigs;
use bevy::ecs::system::{Commands, Res, ResMut};
use bevy::prelude::RenderGraph;
use bevy::render::{
  Render, RenderApp, RenderStartup, RenderSystems,
  render_resource::{
    BindGroup, BindGroupEntries, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingType,
    Buffer, BufferBindingType, BufferDescriptor, BufferUsages, CachedComputePipelineId,
    ComputePipelineDescriptor, Extent3d, Origin3d, ShaderStages, StorageTextureAccess,
    TexelCopyBufferLayout, TexelCopyTextureInfo, Texture, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
    TextureViewDescriptor, TextureViewDimension, UniformBuffer,
  },
  renderer::{RenderDevice, RenderQueue},
};
use std::borrow::Cow;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct DdgiLod {
  pub origin: IVec4,
  pub prev_origin: IVec4,
  pub reuse_min: IVec4,
  pub reuse_max: IVec4,
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, bevy::ecs::resource::Resource, ShaderType)]
pub struct DdgiUniform {
  pub lods: [DdgiLod; 4],
  pub params: Vec4,
  pub misc: Vec4,
}

pub fn ddgi_bg4_layout() -> BindGroupLayoutDescriptor {
  const C: ShaderStages = ShaderStages::COMPUTE;
  let tex = |binding: u32, sample_type: TextureSampleType| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Texture {
      sample_type,
      view_dimension: TextureViewDimension::D2Array,
      multisampled: false,
    },
    count: None,
  };
  let store = |binding: u32, format: TextureFormat| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::StorageTexture {
      access: StorageTextureAccess::WriteOnly,
      format,
      view_dimension: TextureViewDimension::D2Array,
    },
    count: None,
  };
  let buf = |binding: u32, read_only: bool| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "DdgiBg4",
    &[
      BindGroupLayoutEntry {
        binding: 0,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(DdgiUniform::min_size()),
        },
        count: None,
      },
      tex(1, TextureSampleType::Float { filterable: true }),
      tex(2, TextureSampleType::Float { filterable: false }),
      store(3, TextureFormat::Rgba16Float),
      store(4, TextureFormat::R32Float),
      tex(5, TextureSampleType::Uint),
      store(6, TextureFormat::R32Uint),
      // 7: indirect/计数器合一 rw；8: objects ro；9: worklist rw；10: slot_pos rw
      buf(7, false),
      buf(8, true),
      buf(9, false),
      buf(10, false),
    ],
  )
}

#[derive(Debug, Clone, Copy)]
pub struct DdgiPipelines {
  pub sort: CachedComputePipelineId,
  pub seal: CachedComputePipelineId,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiGpu {
  pub uniform: UniformBuffer<DdgiUniform>,
  pub objects: Buffer,
  pub indirect: Buffer,
  pub worklist: Buffer,
  pub slot_pos: Buffer,
  pub irr_prev: Texture,
  pub irr_next: Texture,
  pub depth_prev: Texture,
  pub depth_next: Texture,
  pub meta_prev: Texture,
  pub meta_next: Texture,
  pub irr_prev_view: TextureView,
  pub irr_next_view: TextureView,
  pub depth_prev_view: TextureView,
  pub depth_next_view: TextureView,
  pub meta_prev_view: TextureView,
  pub meta_next_view: TextureView,
  pub frame: u32,
  pub prev_origins: [IVec3; 4],
  pub have_prev: bool,
  pub pipelines: Option<DdgiPipelines>,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct DdgiBg4(pub BindGroup);

#[derive(bevy::ecs::resource::Resource, Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct DdgiStage(pub u8);

impl DdgiStage {
  pub const OFF: u8 = 0;
  pub const ACTIVE: u8 = 1;
  pub const CAST: u8 = 2;
  pub const FULL: u8 = 3;

  pub fn new(v: u8) -> Self {
    Self(v.min(Self::FULL))
  }
  pub fn run_active(&self) -> bool {
    self.0 >= Self::ACTIVE
  }
  pub fn run_cast(&self) -> bool {
    self.0 >= Self::CAST
  }
  pub fn shade_gi(&self) -> bool {
    self.0 >= Self::FULL
  }
}

#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq)]
pub struct DdgiDebugSettings {
  pub mode: f32,
  pub gain: f32,
  pub probe_viz: bool,
  pub probe_viz_lod: f32,
}

impl Default for DdgiDebugSettings {
  fn default() -> Self {
    Self {
      mode: 0.0,
      gain: 1.0,
      probe_viz: false,
      probe_viz_lod: 0.0,
    }
  }
}

pub struct DdgiPlugin;

impl bevy::app::Plugin for DdgiPlugin {
  fn build(&self, app: &mut bevy::app::App) {
    app.init_resource::<DdgiStage>();
    app.init_resource::<DdgiDebugSettings>();
    let Some(render_app) = app.get_sub_app_mut(RenderApp) else {
      return;
    };
    render_app
      .init_resource::<DdgiStage>()
      .init_resource::<DdgiDebugSettings>()
      .add_systems(RenderStartup, init_ddgi_gpu)
      .add_systems(
        RenderStartup,
        queue_ddgi_pipelines.after(crate::brickmap::dda::init_dda_pipelines),
      )
      .add_systems(bevy::render::ExtractSchedule, extract_ddgi_settings)
      .add_systems(
        Render,
        prepare_ddgi.in_set(RenderSystems::PrepareBindGroups),
      )
      .add_systems(
        RenderGraph,
        dispatch_ddgi
          .in_set(bevy::render::renderer::RenderGraphSystems::Render)
          .before(crate::brickmap::dda::dispatch_dda),
      );
  }
}

fn dummy_sized_buffer(device: &RenderDevice, label: &str, size: u64) -> Buffer {
  device.create_buffer(&BufferDescriptor {
    label: Some(label.into()),
    size: size.max(4),
    usage: BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  })
}

fn zero_storage_buffer(
  device: &RenderDevice,
  queue: &RenderQueue,
  label: &str,
  size: u64,
) -> Buffer {
  let buf = dummy_sized_buffer(device, label, size);
  queue.write_buffer(&buf, 0, &vec![0u8; size as usize]);
  buf
}

/// indirect args / 计数器合一 buffer：既要作为 storage（atomic）被 sort/seal 读写，
/// 又要在阶段二作为 indirect dispatch 参数源。
fn ddgi_indirect_buffer(device: &RenderDevice, queue: &RenderQueue) -> Buffer {
  let buf = device.create_buffer(&BufferDescriptor {
    label: Some("ddgi_indirect".into()),
    size: DDGI_INDIRECT_BYTES,
    usage: BufferUsages::STORAGE
      | BufferUsages::INDIRECT
      | BufferUsages::COPY_DST
      | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
  });
  queue.write_buffer(&buf, 0, &vec![0u8; DDGI_INDIRECT_BYTES as usize]);
  buf
}

fn ddgi_array_view(tex: &Texture) -> TextureView {
  tex.create_view(&TextureViewDescriptor {
    dimension: Some(TextureViewDimension::D2Array),
    ..Default::default()
  })
}

fn ddgi_array_tex(
  device: &RenderDevice,
  label: &str,
  format: TextureFormat,
  size: (u32, u32),
  layers: u32,
) -> Texture {
  device.create_texture(&TextureDescriptor {
    label: Some(label.into()),
    size: Extent3d {
      width: size.0,
      height: size.1,
      depth_or_array_layers: layers,
    },
    mip_level_count: 1,
    sample_count: 1,
    dimension: TextureDimension::D2,
    format,
    usage: TextureUsages::TEXTURE_BINDING
      | TextureUsages::STORAGE_BINDING
      | TextureUsages::COPY_DST
      | TextureUsages::COPY_SRC,
    view_formats: &[],
  })
}

fn init_ddgi_gpu(mut commands: Commands, device: Res<RenderDevice>, queue: Res<RenderQueue>) {
  let mk_pair = |label: &str, format: TextureFormat, size: (u32, u32)| {
    let t0 = ddgi_array_tex(
      &device,
      &format!("{label}_a"),
      format,
      size,
      DDGI_TOTAL_LAYERS,
    );
    let t1 = ddgi_array_tex(
      &device,
      &format!("{label}_b"),
      format,
      size,
      DDGI_TOTAL_LAYERS,
    );
    let v0 = ddgi_array_view(&t0);
    let v1 = ddgi_array_view(&t1);
    (t0, v0, t1, v1)
  };
  let (irr_a, irr_av, irr_b, irr_bv) = mk_pair(
    "ddgi_irr",
    TextureFormat::Rgba16Float,
    (IRRADIANCE_LAYER_TEXELS, IRRADIANCE_LAYER_TEXELS),
  );
  let (dep_a, dep_av, dep_b, dep_bv) = mk_pair(
    "ddgi_depth",
    TextureFormat::R32Float,
    (DEPTH_LAYER_TEXELS, DEPTH_LAYER_TEXELS),
  );
  let (meta_a, meta_av, meta_b, meta_bv) = mk_pair(
    "ddgi_meta",
    TextureFormat::R32Uint,
    (PROBES_PER_LAYER_AXIS, PROBES_PER_LAYER_AXIS),
  );

  let dep_data = f32_bytes(&vec![
    PROBE_T_MAX;
    (DEPTH_LAYER_TEXELS * DEPTH_LAYER_TEXELS * DDGI_TOTAL_LAYERS)
      as usize
  ]);
  for tex in [&dep_a, &dep_b] {
    queue.write_texture(
      TexelCopyTextureInfo {
        texture: tex,
        mip_level: 0,
        origin: Origin3d::ZERO,
        aspect: TextureAspect::All,
      },
      &dep_data,
      TexelCopyBufferLayout {
        offset: 0,
        bytes_per_row: Some(DEPTH_LAYER_TEXELS * 4),
        rows_per_image: Some(DEPTH_LAYER_TEXELS),
      },
      Extent3d {
        width: DEPTH_LAYER_TEXELS,
        height: DEPTH_LAYER_TEXELS,
        depth_or_array_layers: DDGI_TOTAL_LAYERS,
      },
    );
  }

  let slot_pos = zero_storage_buffer(
    &device,
    &queue,
    "ddgi_slot_pos",
    DDGI_TOTAL_SLOTS as u64 * 16,
  );
  // per-LOD 4096 项 ×4 LOD ×16B（vec4 item）
  let worklist = zero_storage_buffer(
    &device,
    &queue,
    "ddgi_worklist",
    DDGI_TOTAL_SLOTS as u64 * DDGI_WORKLIST_ITEM_BYTES,
  );
  let indirect = ddgi_indirect_buffer(&device, &queue);
  let objects = dummy_sized_buffer(&device, "ddgi_objects", 16);

  commands.insert_resource(DdgiGpu {
    uniform: UniformBuffer::default(),
    objects,
    indirect,
    worklist,
    slot_pos,
    irr_prev: irr_a,
    irr_next: irr_b,
    depth_prev: dep_a,
    depth_next: dep_b,
    meta_prev: meta_a,
    meta_next: meta_b,
    irr_prev_view: irr_av,
    irr_next_view: irr_bv,
    depth_prev_view: dep_av,
    depth_next_view: dep_bv,
    meta_prev_view: meta_av,
    meta_next_view: meta_bv,
    frame: 0,
    prev_origins: [IVec3::ZERO; 4],
    have_prev: false,
    pipelines: None,
  });
}

fn queue_ddgi_pipelines(
  dda: Option<Res<crate::brickmap::dda::DdaPipelines>>,
  pipeline_cache: Res<bevy::render::render_resource::PipelineCache>,
  asset_server: Res<AssetServer>,
  mut gpu: ResMut<DdgiGpu>,
) {
  if gpu.pipelines.is_some() {
    return;
  }
  let Some(dda) = dda else {
    return;
  };
  let layout = vec![
    dda.bg0_layout.clone(),
    dda.bg1_layout.clone(),
    dda.bg2_layout.clone(),
    dda.bg3_layout.clone(),
    ddgi_bg4_layout(),
  ];
  let shader = asset_server.load(crate::brickmap::dda::DDA_SHADER_ASSET_PATH);
  let mk = |label: &'static str, entry: &'static str| {
    pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
      label: Some(Cow::from(label)),
      layout: layout.clone(),
      shader: shader.clone(),
      entry_point: Some(Cow::from(entry)),
      ..Default::default()
    })
  };
  gpu.pipelines = Some(DdgiPipelines {
    sort: mk("gate_ddgi_sort", "ddgi_sort"),
    seal: mk("gate_ddgi_seal", "ddgi_seal"),
  });
}

#[allow(clippy::too_many_arguments)]
fn dispatch_ddgi(
  mut ctx: bevy::render::renderer::RenderContext,
  bg0: Option<Res<crate::brickmap::dda::DdaBg0BindGroup>>,
  bg1: Option<Res<crate::brickmap::dda::DdaBg1BindGroup>>,
  bg2: Option<Res<crate::brickmap::dda::DdaBg2BindGroup>>,
  bg3: Option<Res<crate::brickmap::dda::DdaBg3BindGroup>>,
  bg4: Option<Res<DdgiBg4>>,
  stage: Res<DdgiStage>,
  dbg: Option<Res<DdgiDebugSettings>>,
  gpu: Res<DdgiGpu>,
  pipeline_cache: Res<bevy::render::render_resource::PipelineCache>,
  mut profiler: ResMut<crate::profiler::GpuProfilerRes>,
) {
  if !stage.run_active() && !dbg.map_or(false, |d| d.probe_viz) {
    return;
  }
  let (Some(bg0), Some(bg1), Some(bg2), Some(bg3), Some(bg4)) = (
    bg0.as_ref(),
    bg1.as_ref(),
    bg2.as_ref(),
    bg3.as_ref(),
    bg4.as_ref(),
  ) else {
    return;
  };
  let Some(pipes) = gpu.pipelines else {
    return;
  };
  let (Some(p_sort), Some(p_seal)) = (
    pipeline_cache.get_compute_pipeline(pipes.sort),
    pipeline_cache.get_compute_pipeline(pipes.seal),
  ) else {
    return;
  };

  let set_bgs = |pass: &mut bevy::render::render_resource::ComputePass, bg4: &BindGroup| {
    pass.set_bind_group(0, &bg0.0, &[]);
    pass.set_bind_group(1, &bg1.0, &[]);
    pass.set_bind_group(2, &bg2.0, &[]);
    pass.set_bind_group(3, &bg3.0, &[]);
    pass.set_bind_group(4, bg4, &[]);
  };

  // sort：探针定位 + slot_pos/meta 刷新（probe_viz 依赖其新鲜度）；始终跑。
  crate::profiler::gpu_compute_pass(
    &mut profiler,
    ctx.command_encoder(),
    "gate_ddgi_sort",
    |pass| {
      pass.set_pipeline(p_sort);
      set_bgs(pass, &bg4.0);
      pass.dispatch_workgroups(4, 4, 16);
    },
  );
  // seal：为 cast/collect 准备 per-LOD indirect args；只在需要阶段二时跑。
  if stage.run_active() {
    crate::profiler::gpu_compute_pass(
      &mut profiler,
      ctx.command_encoder(),
      "gate_ddgi_seal",
      |pass| {
        pass.set_pipeline(p_seal);
        set_bgs(pass, &bg4.0);
        pass.dispatch_workgroups(1, 1, 1);
      },
    );
  }
  // 阶段二 cast / 阶段三 collect 待接入：seal 已在 indirect buffer 备好 per-LOD
  // dispatch 参数（offset 0 = cast×4 LOD，offset 64 = collect×4 LOD）与 rpp（offset 128）。
}

fn extract_ddgi_settings(
  mut commands: Commands,
  stage: Option<bevy::render::Extract<Res<DdgiStage>>>,
  debug: Option<bevy::render::Extract<Res<DdgiDebugSettings>>>,
) {
  let s = stage.map_or(DdgiStage::OFF, |s| s.0.min(DdgiStage::FULL));
  commands.insert_resource(DdgiStage(s));
  let dbg = debug.map_or_else(DdgiDebugSettings::default, |d| DdgiDebugSettings {
    mode: d.mode,
    gain: d.gain,
    probe_viz: d.probe_viz,
    probe_viz_lod: d.probe_viz_lod,
  });
  commands.insert_resource(dbg);
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
  let mut out = Vec::with_capacity(v.len() * 4);
  for c in v {
    out.extend_from_slice(&c.to_le_bytes());
  }
  out
}

#[allow(clippy::too_many_arguments)]
fn prepare_ddgi(
  mut commands: Commands,
  device: Res<RenderDevice>,
  queue: Res<RenderQueue>,
  pipeline_cache: Res<bevy::render::render_resource::PipelineCache>,
  view: Option<Res<crate::brickmap::dda::DdaViewUniform>>,
  stage: Res<DdgiStage>,
  dbg: Res<DdgiDebugSettings>,
  mut gpu: ResMut<DdgiGpu>,
) {
  {
    let g = &mut *gpu;
    let DdgiGpu {
      irr_prev,
      irr_next,
      irr_prev_view,
      irr_next_view,
      depth_prev,
      depth_next,
      depth_prev_view,
      depth_next_view,
      meta_prev,
      meta_next,
      meta_prev_view,
      meta_next_view,
      ..
    } = g;
    std::mem::swap(irr_prev, irr_next);
    std::mem::swap(irr_prev_view, irr_next_view);
    std::mem::swap(depth_prev, depth_next);
    std::mem::swap(depth_prev_view, depth_next_view);
    std::mem::swap(meta_prev, meta_next);
    std::mem::swap(meta_prev_view, meta_next_view);
  }

  gpu.frame = gpu.frame.wrapping_add(1);

  let cam_fine = view
    .map(|v| v.cam_pos_fine.truncate())
    .unwrap_or(Vec3::splat(32.0));

  let mut u = DdgiUniform::default();
  for lod in 0..DDGI_LODS as usize {
    let cell_size = DDGI_LOD_CELL_SIZES[lod];
    let w = compute_lod_window(cam_fine, cell_size, gpu.prev_origins[lod], gpu.have_prev);
    u.lods[lod] = DdgiLod {
      origin: IVec4::new(w.origin.x, w.origin.y, w.origin.z, w.cell_size),
      prev_origin: IVec4::new(w.prev_origin.x, w.prev_origin.y, w.prev_origin.z, 0),
      reuse_min: IVec4::new(w.reuse_min.x, w.reuse_min.y, w.reuse_min.z, 0),
      reuse_max: IVec4::new(w.reuse_max.x, w.reuse_max.y, w.reuse_max.z, 0),
    };
    gpu.prev_origins[lod] = w.origin;
  }
  gpu.have_prev = true;
  u.params = Vec4::new(gpu.frame as f32, dbg.mode, dbg.gain, 0.0);
  u.misc = Vec4::new(if stage.shade_gi() { 1.0 } else { 0.0 }, 0.0, 0.0, 0.0);
  *gpu.uniform.get_mut() = u;
  gpu.uniform.write_buffer(&device, &queue);

  // 每帧清零活跃计数器（indirect words [36..40)）；indirect args 由 seal 当帧覆写。
  queue.write_buffer(
    &gpu.indirect,
    DDGI_COUNTER_CLEAR_OFFSET,
    &[0u8; DDGI_COUNTER_CLEAR_BYTES as usize],
  );

  let bg4_layout = pipeline_cache.get_bind_group_layout(&ddgi_bg4_layout());
  let bg4 = device.create_bind_group(
    None,
    &bg4_layout,
    &BindGroupEntries::sequential((
      &gpu.uniform,
      &gpu.irr_prev_view,
      &gpu.depth_prev_view,
      &gpu.irr_next_view,
      &gpu.depth_next_view,
      &gpu.meta_prev_view,
      &gpu.meta_next_view,
      gpu.indirect.as_entire_binding(),
      gpu.objects.as_entire_binding(),
      gpu.worklist.as_entire_binding(),
      gpu.slot_pos.as_entire_binding(),
    )),
  );
  commands.insert_resource(DdgiBg4(bg4));
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn lod_window_camera_anchored_band_and_reuse() {
    for &cs in &DDGI_LOD_CELL_SIZES {
      for cam_cell in -40..40 {
        let cam_fine = Vec3::splat(cam_cell as f32 * cs as f32 + cs as f32 * 0.37);
        let origin = lod_window_origin(cam_fine, cs);
        let rel = cam_cell - origin.x;
        assert!(
          (DDGI_BAND_LO..DDGI_BAND_LO + DDGI_SCROLL_STEP).contains(&rel),
          "cs={cs} cam_cell={cam_cell} origin={} rel={rel} 不在 [4,12)",
          origin.x
        );
        assert_eq!((origin.x + DDGI_BAND_LO).rem_euclid(DDGI_SCROLL_STEP), 0);
        assert_eq!((origin.x * cs).rem_euclid(cs), 0);
      }
      let cam = Vec3::splat(100.0 * cs as f32);
      let o0 = lod_window_origin(cam, cs);
      let w = compute_lod_window(cam, cs, o0, true);
      assert_eq!(w.reuse_min, o0);
      assert_eq!(w.reuse_max, o0 + DDGI_WINDOW_AXIS);
      let cam8 = cam + Vec3::splat(8.0 * cs as f32);
      let o8 = lod_window_origin(cam8, cs);
      assert_eq!(o8.x, o0.x + 8);
      let w8 = compute_lod_window(cam8, cs, o0, true);
      assert_eq!(w8.reuse_min.x, o0.x + 8);
      assert_eq!(w8.reuse_max.x, o0.x + 16);
      let wf = compute_lod_window(cam, cs, IVec3::ZERO, false);
      assert_eq!(wf.reuse_min, wf.reuse_max);
    }
  }
}
