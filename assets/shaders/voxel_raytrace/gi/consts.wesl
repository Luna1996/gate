const GI_T_MAX: f32 = 8192.0;
const GI_PI: f32 = 3.14159265;
const GI_LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

const GI_SKY_AMBIENT: f32 = 0.05;
const GI_AMBIENT_FLOOR: f32 = 0.001;
const GI_AMBIENT_OPEN: f32 = 0.3;
const GI_FALLBACK: f32 = 0.4;
const GI_REFRESH_N: f32 = 8.0;
const GI_REFRESH_MIN: f32 = 0.0625;
const GI_NEE_P: f32 = 1.0;
const GI_SAMPLE_CLAMP: f32 = 16.0;
const LT_NEE_MAX_D: f32 = 2048.0;
const SHADOW_SURFACE_EPS: f32 = 0.03125;

const IRC_WORDS: u32 = 16u;
const IRC_BUCKET: u32 = 8u;
const IRC_CHAIN: u32 = 4u;
const IRC_MAX_LEVEL: u32 = 11u;
const IRC_FALLBACK_LEVELS: u32 = 4u;
const IRC_EVICT_GUARD: u32 = 4u;

const IRC_ACC_SCALE: f32 = 16384.0;
const IRC_V_MAX: f32 = 32.0;
const IRC_W_MIN: f32 = 0.04;

const IRC_N_HALF: f32 = 6.0;
const IRC_PARENT_CONF: f32 = 0.9;
const IRC_SHRINK_C: f32 = 8.0;
const IRC_Z2: f32 = 16.0;
const IRC_VAR_ALPHA: f32 = 0.125;
const IRC_SUN_ALPHA: f32 = 0.125;

const GI_SPP_MAX: u32 = 8u;
const GI_SPP_BOOST_N0: f32 = 8.0;
const GI_SPP_BOOST_N1: f32 = 64.0;
const GI_SPP_EASE_N: f32 = 192.0;

const RES_WORDS: u32 = 24u;

const LT_TILE_WORDS: u32 = 12u;
const LT_NODE_WORDS: u32 = 12u;
const LT_BUCKET: u32 = 8u;
const LT_TOP: u32 = 12u;
const LT_EVICT_FRAMES: u32 = 900u;
const LT_POWER_SCALE: f32 = 64.0;
const LT_POWER_MAX: u32 = 1048576u;
const LT_MAX_DIST: f32 = 65536.0;
const LT_SHADOW_EPS: f32 = 0.02;

const GI_STAT_IRC_LIVE: u32 = 0u;
const GI_STAT_IRC_EVICT: u32 = 1u;
const GI_STAT_IRC_DROP: u32 = 2u;
const GI_STAT_LT_TILES: u32 = 3u;
const GI_STAT_LT_LINKED: u32 = 4u;
const GI_STAT_LT_DROP: u32 = 5u;
const GI_STAT_RAYS: u32 = 6u;
const GI_STAT_NEE: u32 = 7u;
const GI_STAT_RES_TEMPORAL: u32 = 8u;
const GI_STAT_RES_SPATIAL: u32 = 9u;
const GI_STAT_WORDS: u32 = 16u;
