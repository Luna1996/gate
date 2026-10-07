pub const VIEW_W: u32 = 1280;
pub const VIEW_H: u32 = 720;
pub const WORKGROUP_SIZE: u32 = 8;
pub const PITCH_LIMIT: f32 = 89.0_f32.to_radians();
pub const DIST_MIN: f32 = 32.0;
pub const BEAM_DIV: u32 = 4;
pub const DDA_FOV_Y: f32 = 60.0_f32.to_radians();
pub const CPU_TRACE_BUDGET: u32 = 65536;

pub const DDA_CHUNKWALK: bool = true;
pub const DDA_SKY_ONLY: bool = false;
pub const DDA_MAKEGRID_ONLY: bool = false;
pub const DDA_LOD: bool = true;
pub const RT_RAY_QUERY: bool = false;
pub const LOD_DIAG_WORDS: usize = 32;
pub const REQ_CAP: usize = 1024 * 1024;
pub const REQ_FEED_MAX: usize = 16 * 1024;
pub const USE_BASE: usize = 3;
pub const VOLUMES: usize = 4;
pub const LOD_REQ_WORDS: usize = 3 + USE_WORDS * VOLUMES + REQ_CAP;
pub const USE_WORDS: usize = 64 * 64 * 64;
pub const FAR_RESIDENT_TARGET: usize = FAR_POOL_CHUNKS / 4 * 3;

pub const FAR_POOL_CHUNKS: usize = 4096;
pub const FAR_RESERVE_WORDS_PER_CHUNK: usize = 2048;
pub const FAR_TREE_RESERVE_WORDS: usize = FAR_POOL_CHUNKS * FAR_RESERVE_WORDS_PER_CHUNK;

pub const INDEX_ENTRY_EMPTY: u32 = u32::MAX;

pub const REQ_BASE: usize = USE_BASE + USE_WORDS * VOLUMES;
pub const DDA_BEAM: bool = true;
pub const DDA_DIR_LUT: bool = true;
pub const EYE_ADAPT: bool = true;

pub const SINGLE_THRESHOLD_BYTES: u64 = (1 << 30) - 1;
pub const PER_CHUNK_BYTES: usize = 256 * 1024;
pub const UPLOAD_BYTES_PER_FRAME: usize = 4 * 1024 * 1024;
pub const BUFFER_SLICE_TARGET: u64 = 64 << 20;
pub const BUFFER_SLICE_MIN: u64 = 4 << 20;
pub const BUFFER_GROW_BIG: u64 = 8 << 20;
pub const BUFFER_GROW_RESERVE: u64 = 32 << 20;
pub const VRAM_WARN_BYTES: u64 = 2 << 30;
