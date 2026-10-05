@group(0) @binding(0) var out_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(2) var beam_depth: texture_storage_2d<r32float, read_write>;
@group(0) @binding(3) var<storage, read> eye_adapt_ro: array<u32>;

const BEAM_DIV: u32 = 4u;

const BEAM_BACKOFF: f32 = 4.0;

struct DdaViewUniform {
  view_proj: mat4x4<f32>,       
  inv_view_proj: mat4x4<f32>,   
  cam_pos_voxel: vec4<f32>,      
  debug_mode: vec4<f32>,        
  lod: vec4<f32>,               
}
@group(0) @binding(1) var<uniform> view_u: DdaViewUniform;

@group(1) @binding(0) var<storage, read> b_struct: array<u32>;
@group(1) @binding(1) var<storage, read> b_leaves: array<u64>;
@group(1) @binding(2) var<storage, read> b_palette: array<u32>;

struct Globals {
  index_origin_x: i32,
  index_origin_y: i32,
  index_origin_z: i32,
  index_origin_w: i32,
  index_dims_x: u32,
  index_dims_y: u32,
  index_dims_z: u32,
  index_dims_w: u32,
  tile_count: u32,
  node_words: u32,
  node_free_words: u32,
  brick_slabs: u32,
  brick_free: u32,
  rejected_tiles: u32,
  grid_count: u32,
  _pad1: u32,
  _pad2: u32,
  _pad3: u32,
  _pad4: u32,
}
@group(1) @binding(3) var<uniform> g: Globals;

@group(1) @binding(4) var light_samp: sampler;

struct MaterialAsset {
  albedo_slot: u32,        
  roughmetal_slot: u32,    
  emissive_slot: u32,      
  transmission_slot: u32,  
  height_slot: u32,        
  albedo_rough: u32,       
  
  
  
  
  
  emissive_metal: u32,
  transmission_ior: u32,   
};

@group(1) @binding(5) var<storage, read> material_assets: array<MaterialAsset>;

@group(1) @binding(6) var pbr_albedo_rough: texture_2d_array<f32>;  
@group(1) @binding(7) var pbr_metal: texture_2d_array<f32>;         

@group(1) @binding(8) var pbr_samp: sampler;

@group(1) @binding(9) var<storage, read_write> lod_diag: array<atomic<u32>>;

@group(1) @binding(10) var<storage, read_write> lod_req: array<atomic<u32>>;

@if(ray_query)
@group(1) @binding(11) var tlas: acceleration_structure;

@group(1) @binding(12) var<storage, read> occ: array<u32>;

struct GridDesc {
  pos_scale: vec4<f32>,   
  rot0: vec4<f32>,        
  rot1: vec4<f32>,        
  rot2: vec4<f32>,        
  aabb_min: vec4<f32>,    
  aabb_max: vec4<f32>,    
  tree_base: u32,         
  tree_depth: u32,        
  chunk_count: u32,       
  palette_base: u32,      
  index_origin_x: i32,    
  index_origin_y: i32,
  index_origin_z: i32,
  
  
  _pad0: u32,
  index_dims_x: u32,      
  index_dims_y: u32,
  index_dims_z: u32,
  
  
  coverage_r: f32,
}
@group(2) @binding(0) var<storage, read> grid_descs: array<GridDesc>;

struct LightDesc {
  kind_pos_dir: vec4<f32>,    
  color_intensity: vec4<f32>, 
  
  
  shape: vec4<f32>,
}
struct LightGlobals {
  count: u32,
  
  
  
  
  refl_tier: u32,
  
  
  refl_nest: u32,
  
  
  
  base_flags: u32,
  ambient: vec4<f32>,     
  exposure_pad: vec4<f32>, 
}
struct LightPool {
  g: LightGlobals,
  lights: array<LightDesc, 8u>,
  sky_color: vec4<f32>,   
}
@group(3) @binding(0) var<uniform> light_u: LightPool;

const BASE_FLAG_SHADOW: u32 = 1u;

const BASE_FLAG_IMPLICIT_NORMAL: u32 = 2u;

const BASE_FLAG_VOL_TINT: u32 = 4u;

struct GiUniform {
  
  
  
  
  
  
  params: vec4<f32>,
  
  
  
  
  misc: vec4<f32>,
  
  
  
  
  
  
  
  
  flags: vec4<f32>,
  
  
  
  
  
  
  seq: vec4<u32>,
  
  
  
  prev_view_proj: mat4x4<f32>,
};
@group(4) @binding(0) var<uniform> gi_u: GiUniform;

@group(4) @binding(20) var<storage, read_write> gi_res_cur: array<u32>;
@group(4) @binding(21) var<storage, read_write> gi_res_prev: array<u32>;

struct FogUniform {
  
  params: vec4<f32>,
  
  
  
  misc: vec4<f32>,
  
  
  
  
  misc2: vec4<f32>,
  
  misc3: vec4<f32>,
}
@group(4) @binding(1) var<uniform> fog_u: FogUniform;

@group(4) @binding(3) var fog_a_w: texture_storage_2d<rgba16float, write>;

@group(0) @binding(21) var<uniform> fog_u_blur: FogUniform;

@group(0) @binding(22) var fog_a_r_blur: texture_2d<f32>;
@group(0) @binding(24) var fog_b_r_blur: texture_2d<f32>;

@group(0) @binding(23) var fog_a_w_blur: texture_storage_2d<rgba16float, write>;
@group(0) @binding(25) var fog_b_w_blur: texture_storage_2d<rgba16float, write>;

@group(5) @binding(2) var gi_out: texture_storage_2d<rgba16float, write>;

@group(5) @binding(4) var gi_tex: texture_2d<f32>;

@group(5) @binding(6) var<storage, read_write> gi_guide_out: array<u32>;

@group(5) @binding(8) var<storage, read_write> face_slots: array<atomic<u32>>;

@group(5) @binding(9) var<storage, read_write> gi_sec_slots: array<atomic<u32>>;

@group(5) @binding(10) var<storage, read> gi_region_rev: array<u32>;

@group(5) @binding(11) var<storage, read_write> gi_wal: array<atomic<u32>>;

@group(6) @binding(0) var<uniform> fog_u_read: FogUniform;
@group(6) @binding(1) var fog_tex: texture_2d<f32>;

@group(0) @binding(10) var<storage, read> gi_guide: array<u32>;

@group(0) @binding(11) var gi_raw_tex: texture_2d<f32>;

@group(0) @binding(12) var<storage, read> gi_hist_prev: array<u32>;
@group(0) @binding(13) var<storage, read_write> gi_hist_cur: array<u32>;

@group(0) @binding(14) var gi_dn_src: texture_2d<f32>;
@group(0) @binding(15) var gi_dn_dst: texture_storage_2d<rgba16float, write>;

@group(0) @binding(16) var gi_dn_tmp: texture_storage_2d<rgba16float, write>;

@group(0) @binding(17) var<storage, read_write> gi_den_phi: array<f32>;

@group(0) @binding(20) var<storage, read> gi_den_cfg: array<u32>;

@group(0) @binding(18) var<storage, read_write> face_slots_flat: array<atomic<u32>>;
@group(0) @binding(19) var gi_out_flat: texture_storage_2d<rgba16float, write>;

@group(7) @binding(0) var out_tex_src: texture_2d<f32>;

@group(7) @binding(1) var<storage, read_write> eye_adapt: array<atomic<u32>>;
