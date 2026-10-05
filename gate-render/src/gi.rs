

use bevy::render::render_resource::{
  BindGroupLayoutDescriptor, CachedComputePipelineId, ShaderType,
};
use glam::{Mat4, UVec2, UVec4, Vec4};

use crate::wesl_consts::gi_consts;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, ShaderType)]
pub struct GiUniform {
  
  
  
  pub params: Vec4,
  
  
  
  pub misc: Vec4,
  
  
  
  
  
  
  
  
  
  pub flags: Vec4,
  
  
  
  
  pub seq: UVec4,
  
  
  
  pub prev_view_proj: Mat4,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct LightKey {
  
  dir: [u32; 3],
  sun_c: [u32; 3],
  
  sky: [u32; 3],
}

impl LightKey {
  fn of(theme: Option<&crate::lighting::LightingTheme>) -> Self {
    let bits3 = |v: [f32; 3]| v.map(f32::to_bits);
    let (dir, sun_c, sky) = match theme {
      Some(t) => {
        let (d, c) = match &t.sun {
          Some(s) => {
            let l = -glam::Vec3::from(s.dir).normalize_or_zero();
            (
              [l.x, l.y, l.z],
              [s.color[0] * s.intensity, s.color[1] * s.intensity, s.color[2] * s.intensity],
            )
          }
          None => ([0.0; 3], [0.0; 3]),
        };
        let sky = t.sky.as_ref().map_or(crate::consts::MINECRAFT_SKY, |s| s.color);
        (bits3(d), bits3(c), bits3(sky))
      }
      None => (bits3([0.0; 3]), bits3([0.0; 3]), bits3(crate::consts::MINECRAFT_SKY)),
    };
    Self { dir, sun_c, sky }
  }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ShadeKey {
  
  
  
  geom: u32,
  
  dir: [u32; 3],
  sun_c: [u32; 3],
  sky: [u32; 3],
  sun_bounce: bool,
}

impl ShadeKey {
  fn of(geom: u32, sun_bounce: bool, light: &LightKey) -> Self {
    let gated = |v: [u32; 3]| if sun_bounce { v } else { [0u32; 3] };
    Self { geom, dir: gated(light.dir), sun_c: gated(light.sun_c), sky: light.sky, sun_bounce }
  }
}

const LIGHT_DIR_MIN_MAG: f32 = 0.02;

const LIGHT_STEP_MAX: f32 = 0.5;

pub const KEY_ORG_Q: i32 = 8192;

pub fn key_origins_q(windows: &[glam::IVec3]) -> Vec<glam::IVec3> {
  let chunk = gate_voxel::CHUNK_SIZE as i32;
  windows
    .iter()
    .map(|w| {
      let o = *w * chunk;
      glam::IVec3::new(o.x / KEY_ORG_Q, o.y / KEY_ORG_Q, o.z / KEY_ORG_Q) * KEY_ORG_Q
    })
    .collect()
}

pub const REGION_CHUNKS: i32 = 8;

pub const REGION_TABLE: i32 = 16;

pub const REGION_TABLE_BYTES: u64 =
  (REGION_TABLE as u64) * (REGION_TABLE as u64) * (REGION_TABLE as u64) * 4;

pub const REGION_REACH: i32 = 2;

pub const WAL_SLOTS: u64 = 1 << 20;

pub const WAL_WORDS: u64 = 8;

pub fn region_origin(window_origin: glam::IVec3) -> glam::IVec3 {
  let q = KEY_ORG_Q / gate_voxel::CHUNK_SIZE as i32;
  glam::IVec3::new(window_origin.x / q, window_origin.y / q, window_origin.z / q) * q
}

pub fn region_cell(v_world: glam::IVec3, org_chunks: glam::IVec3) -> glam::IVec3 {
  let chunk = gate_voxel::CHUNK_SIZE as i32;
  let rel = (v_world - org_chunks * chunk).max(glam::IVec3::ZERO);
  (rel / chunk / REGION_CHUNKS).min(glam::IVec3::splat(REGION_TABLE - 1))
}

fn region_mark(table: &mut [u32], lo: glam::IVec3, hi: glam::IVec3, org_chunks: glam::IVec3) {
  const BIG_BOX_CELLS: i32 = 256;
  let c0 = region_cell(lo, org_chunks);
  let c1 = if hi.cmpgt(lo).all() { region_cell(hi - glam::IVec3::ONE, org_chunks) } else { c0 };
  let span = c1 - c0 + glam::IVec3::ONE;
  if span.x * span.y * span.z > BIG_BOX_CELLS {
    for v in table.iter_mut() {
      *v = v.wrapping_add(1);
    }
    return;
  }
  for z in c0.z..=c1.z {
    for y in c0.y..=c1.y {
      for x in c0.x..=c1.x {
        for dz in -REGION_REACH..=REGION_REACH {
          for dy in -REGION_REACH..=REGION_REACH {
            for dx in -REGION_REACH..=REGION_REACH {
              let cx = (x + dx).clamp(0, REGION_TABLE - 1);
              let cy = (y + dy).clamp(0, REGION_TABLE - 1);
              let cz = (z + dz).clamp(0, REGION_TABLE - 1);
              let i = (cz * REGION_TABLE * REGION_TABLE + cy * REGION_TABLE + cx) as usize;
              table[i] = table[i].wrapping_add(1);
            }
          }
        }
      }
    }
  }
}

fn light_jump(prev: &LightKey, now: &LightKey) -> f32 {
  let vec3 = |v: [u32; 3]| glam::Vec3::from(v.map(f32::from_bits));
  let rel = |a: [u32; 3], b: [u32; 3]| -> f32 {
    let (a, b) = (vec3(a), vec3(b));
    (a - b).length() / a.length().max(b.length()).max(1e-4)
  };
  let dir = if vec3(prev.dir).length().min(vec3(now.dir).length()) > LIGHT_DIR_MIN_MAG {
    1.0 - vec3(prev.dir).normalize_or_zero().dot(vec3(now.dir).normalize_or_zero())
  } else {
    0.0
  };
  rel(prev.sun_c, now.sun_c).max(rel(prev.sky, now.sky)).max(dir)
}

#[derive(bevy::ecs::resource::Resource, Clone, Copy, Debug, PartialEq)]
pub struct GiSettings {
  
  pub enabled: bool,
  
  
  
  
  
  pub gi_div: u32,
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  pub denoise: u32,
  
  
  
  
  
  
  
  
  pub wal: bool,
  
  
  
  
  
  
  
  
  
  
  
  
  pub sun_bounce: bool,
  
  
  
  
  
  
  
  
  
  
  
  
  pub share: u32,
  
  
  
  
  
  
  
  
  
  
  pub realloc: u32,
  
  
  
  
  
  
  
  
  
  
  
  
  pub depth: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DenoisePlan {
  
  pub on: bool,
  
  pub rounds: u32,
  
  pub radius: u32,
}

impl GiSettings {
  
  
  pub const DIV_CHOICES: [u32; 3] = [1, 2, 4];

  
  pub const DENOISE_TIERS: u32 = 4;

  
  pub const SHARE_CHOICES: [u32; 4] = [1, 2, 4, 8];

  
  pub fn share(&self) -> u32 {
    self.share.clamp(1, 8)
  }

  
  pub const REALLOC_TIERS: u32 = 4;

  
  pub const BOUNCE2_TIERS: u32 = 3;

  
  pub fn depth_tier(&self) -> u32 {
    self.depth.min(Self::BOUNCE2_TIERS - 1)
  }

  
  
  
  pub fn bounce2_mult(&self) -> f32 {
    match self.depth_tier() {
      0 => 0.0,
      1 => 4.0,
      _ => 1.0,
    }
  }

  
  pub fn realloc_tier(&self) -> u32 {
    self.realloc.min(Self::REALLOC_TIERS - 1)
  }

  
  pub fn div(&self) -> u32 {
    self.gi_div.clamp(1, 4)
  }

  
  pub fn tier(&self) -> u32 {
    self.denoise.min(Self::DENOISE_TIERS - 1)
  }

  
  pub fn denoise_plan(&self) -> DenoisePlan {
    let c = gi_consts();
    match self.tier() {
      
      0 => DenoisePlan { on: false, rounds: 0, radius: c.gi_den_atrous_r_fast },
      
      1 => DenoisePlan { on: true, rounds: c.gi_den_atrous_iter, radius: c.gi_den_atrous_r_fast },
      
      _ => DenoisePlan { on: true, rounds: c.gi_den_atrous_iter, radius: c.gi_den_atrous_r },
    }
  }

  
  pub fn gi_size(&self, render_size: UVec2) -> UVec2 {
    let d = self.div();
    UVec2::new((render_size.x / d).max(1), (render_size.y / d).max(1))
  }
}

impl Default for GiSettings {
  fn default() -> Self {
    
    
    
    
    
    Self {
      enabled: true,
      gi_div: 4,
      denoise: 1,
      wal: false,
      sun_bounce: false,
      share: 2,
      realloc: 0,
      depth: 0,
    }
  }
}

pub fn gi_bg4_layout() -> BindGroupLayoutDescriptor {
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
    "GiBg4",
    &[
      BindGroupLayoutEntry {
        binding: 0,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Uniform,
          has_dynamic_offset: false,
          min_binding_size: Some(GiUniform::min_size()),
        },
        count: None,
      },
      buf(20),
      buf(21),
    ],
  )
}

pub fn gi_bg5_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let store = |binding: u32, format: TextureFormat| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::StorageTexture {
      access: StorageTextureAccess::WriteOnly,
      format,
      view_dimension: TextureViewDimension::D2,
    },
    count: None,
  };
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
    &[
      
      store(2, TextureFormat::Rgba16Float),
      
      BindGroupLayoutEntry {
        binding: 6,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      
      
      
      buf(8),
      
      buf(9),
      
      
      BindGroupLayoutEntry {
        binding: 10,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: true },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      
      
      buf(11),
    ],
  )
}

pub fn gi_den_temporal_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  let ro = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: true },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  let rw = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Buffer {
      ty: BufferBindingType::Storage { read_only: false },
      has_dynamic_offset: false,
      min_binding_size: None,
    },
    count: None,
  };
  let tex = |binding: u32| BindGroupLayoutEntry {
    binding,
    visibility: C,
    ty: BindingType::Texture {
      sample_type: TextureSampleType::Float { filterable: true },
      view_dimension: TextureViewDimension::D2,
      multisampled: false,
    },
    count: None,
  };
  BindGroupLayoutDescriptor::new(
    "GiDenTemporal",
    &[
      ro(10),  
      tex(11), 
      ro(12),  
      rw(13),  
      BindGroupLayoutEntry {
        binding: 16, 
        visibility: C,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::Rgba16Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      },
      rw(17), 
      ro(20), 
    ],
  )
}

pub fn gi_den_atrous_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  BindGroupLayoutDescriptor::new(
    "GiDenAtrous",
    &[
      BindGroupLayoutEntry {
        binding: 10,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: true },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      
      
      BindGroupLayoutEntry {
        binding: 13,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 14,
        visibility: C,
        ty: BindingType::Texture {
          sample_type: TextureSampleType::Float { filterable: true },
          view_dimension: TextureViewDimension::D2,
          multisampled: false,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 15,
        visibility: C,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::Rgba16Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 17,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 20,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: true },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
    ],
  )
}

pub fn gi_flatten_layout() -> BindGroupLayoutDescriptor {
  use bevy::render::render_resource::*;
  const C: ShaderStages = ShaderStages::COMPUTE;
  BindGroupLayoutDescriptor::new(
    "GiFlatten",
    &[
      BindGroupLayoutEntry {
        binding: 10,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: true },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 18,
        visibility: C,
        ty: BindingType::Buffer {
          ty: BufferBindingType::Storage { read_only: false },
          has_dynamic_offset: false,
          min_binding_size: None,
        },
        count: None,
      },
      BindGroupLayoutEntry {
        binding: 19,
        visibility: C,
        ty: BindingType::StorageTexture {
          access: StorageTextureAccess::WriteOnly,
          format: TextureFormat::Rgba16Float,
          view_dimension: TextureViewDimension::D2,
        },
        count: None,
      },
    ],
  )
}

#[derive(bevy::ecs::resource::Resource)]
pub struct GiGpu {
  pub uniform: bevy::render::render_resource::UniformBuffer<GiUniform>,
  pub frame: u32,
  
  
  pub prev_view_proj: Mat4,
  
  pub res_flip: bool,
  
  
  
  
  
  
  
  
  
  
  pub key_org: Option<Vec<glam::IVec3>>,
  
  
  
  region_rev: Vec<u32>,
  
  
  region_dirty: bool,
  
  
  region_org: Option<glam::IVec3>,
  
  region_staging: Vec<u8>,
  
  
  
  
  pub world_rev_gi: u32,
  
  
  
  pub wide_rev: u32,
  
  
  
  pub den_pipelines: [Option<CachedComputePipelineId>; 6],
  
  
  pub flatten_pipeline: Option<CachedComputePipelineId>,
  
  
  
  pub epoch: u32,
  
  epoch_key: Option<ShadeKey>,
  
  light_key: Option<LightKey>,
}

#[derive(bevy::ecs::resource::Resource)]
pub struct GiBg4(pub bevy::render::render_resource::BindGroup);

#[derive(bevy::ecs::resource::Resource)]
pub struct GiBg5(pub bevy::render::render_resource::BindGroup);

#[derive(bevy::ecs::resource::Resource, Default)]
struct GiPlaceholder {
  tex: Option<bevy::render::render_resource::Texture>,
  view: Option<bevy::render::render_resource::TextureView>,
  
  res: Option<bevy::render::render_resource::Buffer>,
}

impl GiPlaceholder {
  fn view(
    &mut self,
    device: &bevy::render::renderer::RenderDevice,
  ) -> &bevy::render::render_resource::TextureView {
    use bevy::render::render_resource::*;
    if self.tex.is_none() {
      let t = device.create_texture(&TextureDescriptor {
        label: Some("gate_gi_placeholder"),
        size: Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: 1,
        dimension: TextureDimension::D2,
        format: TextureFormat::Rgba16Float,
        usage: TextureUsages::STORAGE_BINDING | TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
      });
      self.view = Some(t.create_view(&TextureViewDescriptor::default()));
      self.tex = Some(t);
    }
    self.view.as_ref().expect("刚创建")
  }

  
  fn res_buffer(
    &mut self,
    device: &bevy::render::renderer::RenderDevice,
  ) -> &bevy::render::render_resource::Buffer {
    use bevy::render::render_resource::{BufferDescriptor, BufferUsages};
    self.res.get_or_insert_with(|| {
      device.create_buffer(&BufferDescriptor {
        label: Some("gate_gi_res_placeholder"),
        size: 4,
        usage: BufferUsages::STORAGE,
        mapped_at_creation: false,
      })
    })
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
      .init_resource::<GiPlaceholder>()
      .add_systems(bevy::render::RenderStartup, init_gi_gpu)
      .add_systems(
        bevy::render::RenderStartup,
        queue_gi_pipelines.after(crate::brickmap::dda::init_dda_pipelines),
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

fn init_gi_gpu(mut commands: bevy::ecs::system::Commands) {
  commands.insert_resource(GiGpu {
    uniform: bevy::render::render_resource::UniformBuffer::default(),
    frame: 0,
    
    prev_view_proj: Mat4::IDENTITY,
    res_flip: false,
    key_org: None,
    region_rev: vec![0; (REGION_TABLE * REGION_TABLE * REGION_TABLE) as usize],
    region_dirty: false,
    region_org: None,
    region_staging: Vec::new(),
    
    
    world_rev_gi: 0,
    wide_rev: 0,
    den_pipelines: [None; 6],
    flatten_pipeline: None,
    epoch: 0,
    epoch_key: None,
    light_key: None,
  });
}

fn queue_gi_pipelines(
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  dda_shader: bevy::ecs::system::Res<crate::shader::DdaShaderHandle>,
  mut gpu: bevy::ecs::system::ResMut<GiGpu>,
) {
  use bevy::render::render_resource::ComputePipelineDescriptor;
  use std::borrow::Cow;
  
  if gpu.den_pipelines[0].is_some() {
    return;
  }
  let den_temporal_layout = gi_den_temporal_layout();
  let den_atrous_layout = gi_den_atrous_layout();
  let label = [
    "gate_gi_denoise_temporal",
    "gate_gi_denoise_atrous1",
    "gate_gi_denoise_atrous2",
    "gate_gi_denoise_atrous4",
    "gate_gi_denoise_atrous8",
    "gate_gi_denoise_atrous16",
  ];
  let entry = [
    "gi_denoise_temporal",
    "gi_denoise_atrous1",
    "gi_denoise_atrous2",
    "gi_denoise_atrous4",
    "gi_denoise_atrous8",
    "gi_denoise_atrous16",
  ];
  for i in 0..6 {
    let layout =
      if i == 0 { vec![den_temporal_layout.clone()] } else { vec![den_atrous_layout.clone()] };
    gpu.den_pipelines[i] = Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
      label: Some(Cow::from(label[i])),
      layout,
      shader: dda_shader.0.clone(),
      entry_point: Some(Cow::from(entry[i])),
      ..Default::default()
    }));
  }
  
  
  gpu.flatten_pipeline = Some(pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
    label: Some(Cow::from("gate_gi_face_flatten")),
    layout: vec![gi_flatten_layout()],
    shader: dda_shader.0.clone(),
    entry_point: Some(Cow::from("gi_face_flatten")),
    ..Default::default()
  }));
}

fn extract_gi_settings(
  mut commands: bevy::ecs::system::Commands,
  settings: Option<bevy::render::Extract<bevy::ecs::system::Res<GiSettings>>>,
) {
  commands.insert_resource(settings.map_or_else(GiSettings::default, |s| GiSettings {
    enabled: s.enabled,
    gi_div: s.div(),
    denoise: s.tier(),
    wal: s.wal,
    sun_bounce: s.sun_bounce,
    share: s.share(),
    realloc: s.realloc_tier(),
    depth: s.depth_tier(),
  }));
}

#[allow(clippy::too_many_arguments)]
fn prepare_gi(
  mut commands: bevy::ecs::system::Commands,
  device: bevy::ecs::system::Res<bevy::render::renderer::RenderDevice>,
  queue: bevy::ecs::system::Res<bevy::render::renderer::RenderQueue>,
  pipeline_cache: bevy::ecs::system::Res<bevy::render::render_resource::PipelineCache>,
  settings: bevy::ecs::system::Res<GiSettings>,
  view: Option<bevy::ecs::system::Res<crate::brickmap::dda::DdaViewUniform>>,
  lighting: Option<bevy::ecs::system::Res<crate::lighting::LightingTheme>>,
  aux: Option<bevy::ecs::system::Res<crate::brickmap::dda::AuxTexCache>>,
  dirty: Option<bevy::ecs::system::Res<crate::brickmap::upload::BrickMapDirty>>,
  brickmap: Option<bevy::ecs::system::Res<crate::brickmap::upload::GpuBrickMap>>,
  mut gi_ph: bevy::ecs::system::ResMut<GiPlaceholder>,
  mut gpu: bevy::ecs::system::ResMut<GiGpu>,
) {
  gpu.frame = gpu.frame.wrapping_add(1);

  
  
  
  
  
  
  
  
    
  
  
  
  
    
  
  if dirty.as_ref().is_some_and(|d| d.full || d.palette_changed) {
    gpu.wide_rev = gpu.wide_rev.wrapping_add(1);
  }
  let occluder_rev = gpu.wide_rev;
  let world_same = occluder_rev == gpu.world_rev_gi;

  
  
  
  let mut region_touched = false;
  if let Some(bm) = brickmap.as_ref()
    && let Some(w) = bm.volume_windows.first()
  {
    let org = region_origin(*w);
    if gpu.region_org != Some(org) {
      for v in gpu.region_rev.iter_mut() {
        *v = v.wrapping_add(1);
      }
      gpu.region_org = Some(org);
      region_touched = true;
    }
    if let Some(d) = dirty.as_ref() {
      
      for b in &d.boxes {
        region_mark(&mut gpu.region_rev, b.lo, b.hi, org);
        region_touched = true;
      }
    }
  }
  if region_touched {
    gpu.region_dirty = true;
  }
  
  
  
  if gpu.region_dirty
    && let Some(buf) = aux.as_ref().and_then(|a| a.region_buffer()).cloned()
  {
    
    let mut staging = std::mem::take(&mut gpu.region_staging);
    staging.clear();
    for v in &gpu.region_rev {
      staging.extend_from_slice(&v.to_le_bytes());
    }
    queue.write_buffer(&buf, 0, &staging);
    gpu.region_staging = staging;
    gpu.region_dirty = false;
  }

  
  
  
  
    
  
  
  
  
  
  
  let light = LightKey::of(lighting.as_deref());
  let mut light_step = false;
  if gpu.light_key != Some(light) {
    light_step = gpu.light_key.as_ref().is_some_and(|p| light_jump(p, &light) > LIGHT_STEP_MAX);
    gpu.light_key = Some(light);
  }
  let shade_key = ShadeKey::of(occluder_rev, settings.sun_bounce, &light);
  if gpu.epoch_key != Some(shade_key) {
    gpu.epoch_key = Some(shade_key);
    gpu.epoch = gpu.epoch.wrapping_add(1);
    bevy::log::debug!(target: "gate", "GI 二次顶点缓存 epoch → {} [sun_bounce={}]",
                      gpu.epoch, settings.sun_bounce);
  }
  if light_step {
    bevy::log::debug!(target: "gate", "GI 光照阶跃 ⇒ 本帧不复用历史");
  }

  
  let u = GiUniform {
    params: Vec4::new(
      if settings.sun_bounce { 1.0 } else { 0.0 },
      
      
      if settings.wal { 1.0 } else { 0.0 },
      crate::consts::GI_GAIN,
      0.0,
    ),
    misc: Vec4::new(
      if settings.enabled { 1.0 } else { 0.0 },
      settings.share() as f32,
      settings.realloc_tier() as f32,
      
      settings.bounce2_mult(),
    ),
    flags: Vec4::new(
      0.0,
      settings.div() as f32,
      
      if world_same && !light_step { 1.0 } else { 0.0 },
      
      settings.tier() as f32,
    ),
    
    
    
    seq: UVec4::new(gpu.frame, gpu.epoch, 0, 0),
    
    prev_view_proj: gpu.prev_view_proj,
  };
  *gpu.uniform.get_mut() = u;
  gpu.uniform.write_buffer(&device, &queue);
  
  
  
  
  
  let gi_runs = settings.enabled;
  if gi_runs {
    
    
    
    
    
    
    
    gpu.world_rev_gi = occluder_rev;
    
    if let Some(v) = view.as_ref() {
      gpu.prev_view_proj = v.view_proj;
    }
    
    
    
    
    
    
    
    
    if let Some(bm) = brickmap.as_ref() {
      let org = key_origins_q(&bm.volume_windows);
      if gpu.key_org.as_deref() != Some(org.as_slice()) {
        if gpu.key_org.is_some() {
          gpu.wide_rev = gpu.wide_rev.wrapping_add(1);
        }
        gpu.key_org = Some(org);
      }
    }
  }

  
  use bevy::render::render_resource::{BindGroupEntry, BindingResource};
  let bg4_layout = pipeline_cache.get_bind_group_layout(&gi_bg4_layout());
  
  
  let (res_cur, res_prev) = match aux.as_ref().and_then(|a| a.gi_res_buffers()) {
    Some((a, b)) if gpu.res_flip => (b, a),
    Some((a, b)) => (a, b),
    None => {
      let p = gi_ph.res_buffer(&device);
      (p, p)
    }
  };
  let bg4 = device.create_bind_group(
    None,
    &bg4_layout,
    &[
      BindGroupEntry { binding: 0, resource: gpu.uniform.binding().expect("uniform 已写入") },
      BindGroupEntry { binding: 20, resource: res_cur.as_entire_binding() },
      BindGroupEntry { binding: 21, resource: res_prev.as_entire_binding() },
    ],
  );
  commands.insert_resource(GiBg4(bg4));
  
  if gi_runs {
    gpu.res_flip = !gpu.res_flip;
  }

  
  
  
  let bg5_layout = pipeline_cache.get_bind_group_layout(&gi_bg5_layout());
  let gi_view = match aux.as_ref().and_then(|a| a.gi_write_view()) {
    Some(v) => v.clone(),
    None => gi_ph.view(&device).clone(),
  };
  
  let guide = aux
    .as_ref()
    .and_then(|a| a.gi_guide_buffer())
    .cloned()
    .unwrap_or_else(|| gi_ph.res_buffer(&device).clone());
  
  
  let face_slots = aux
    .as_ref()
    .and_then(|a| a.face_slots_buffer())
    .cloned()
    .unwrap_or_else(|| gi_ph.res_buffer(&device).clone());
  
  
  let gi_sec_slots = aux
    .as_ref()
    .and_then(|a| a.gi_sec_slots_buffer())
    .cloned()
    .unwrap_or_else(|| gi_ph.res_buffer(&device).clone());
  
  
  
  let region_rev = aux
    .as_ref()
    .and_then(|a| a.region_buffer())
    .cloned()
    .unwrap_or_else(|| gi_ph.res_buffer(&device).clone());
  
  
  let wal = aux
    .as_ref()
    .and_then(|a| a.wal_buffer())
    .cloned()
    .unwrap_or_else(|| gi_ph.res_buffer(&device).clone());
  let bg5 = device.create_bind_group(
    None,
    &bg5_layout,
    &[
      BindGroupEntry { binding: 2, resource: BindingResource::TextureView(&gi_view) },
      BindGroupEntry { binding: 6, resource: guide.as_entire_binding() },
      BindGroupEntry { binding: 8, resource: face_slots.as_entire_binding() },
      BindGroupEntry { binding: 9, resource: gi_sec_slots.as_entire_binding() },
      BindGroupEntry { binding: 10, resource: region_rev.as_entire_binding() },
      BindGroupEntry { binding: 11, resource: wal.as_entire_binding() },
    ],
  );
  commands.insert_resource(GiBg5(bg5));
}
