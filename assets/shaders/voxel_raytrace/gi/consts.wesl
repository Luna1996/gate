const GI_T_MAX: f32 = 8192.0;
const GI_SKY_RADIANCE_SCALE: f32 = 1.0;
const GI_EMIT_GAIN: f32 = 1.0;
const GI_PI: f32 = 3.14159265;
const GI_LUMA: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);
const GI_SS_CAND_N: u32 = 4u;
const GI_SS_CAND_N_HQ: u32 = 8u;
const GI_SS_RAY_BATCH: u32 = 4u;
const GI_SS_ADAPT_LO: f32 = 0.5;
const GI_SS_ADAPT_HI: f32 = 2.0;
const GI_SS_ADAPT_EMA: f32 = 0.25;
const GI_SS_ADAPT_REF: f32 = 1.0;
const GI_SS_M_CAP_K: u32 = 20u;
const GI_SS_M_CAP_K_HQ: u32 = 32u;
const GI_SS_M_CAP_REST_K: u32 = 2u;
const GI_SS_CAND_CLAMP_K: f32 = 16.0;
const GI_SS_CAND_CLAMP_CEIL: f32 = 3.14159265;
const GI_SS_REUSE_TAPS: u32 = 8u;
const GI_SS_REUSE_M_K: f32 = 0.25;
const GI_SS_REUSE_M_K_STATIC: f32 = 4.0;
const GI_SS_REUSE_CLAMP_LO: f32 = 0.5;
const GI_SS_REUSE_CLAMP_HI: f32 = 2.0;
/// 过渡期内邻域 tap 的钳制范围（比稳态窄得多）。
///
/// 稳态的 [0.5×, 2×] 允许邻域把像素的值拖到自己的 0.5 倍 ⇒ 变化只能靠**扩散**传播，
/// 变暗时还会被卡在 50% 上下（"收敛不干净"）。
/// 过渡期把范围收窄 ⇒ 邻域仍然帮忙**去噪**（不能停：停了原始噪声会直接露出来 ⇒ 满屏"花"），
/// 但它不再有能力把像素按在旧值上 ⇒ 响应直达。
const GI_SS_REUSE_CLAMP_LO_RESET: f32 = 0.8;
const GI_SS_REUSE_CLAMP_HI_RESET: f32 = 1.25;
/// 世界内容变化时，**屏幕空间旧历史保留的权重**（0 = 全丢，1 = 全留）。
///
/// 1.0（稳态）不动；内容一变（事件当帧）就降到本值 ⇒ 旧值仍以小权重参与，
/// 新值平滑地把它带过去 ⇒ **不闪**（不会像"整屏历史作废"那样先抖一遍），
/// 同时比稳态快约 1/本值 倍追上新光照。
/// 代价：过渡期间新样本的相对权重变大 ⇒ 原始 GI 噪声略增（由降噪与 atrous 吸收）。
const GI_HIST_RESET_SCALE: f32 = 0.05;
/// 过渡期内**降噪历史**权重的缩放（比屏幕空间那档温和：降噪是最后一道平滑，
/// 压太狠会让过渡期的原始噪声直接露出来）。
const GI_DEN_RESET_SCALE: f32 = 0.25;
/// 过渡期长度（帧）。这段时间内：降噪历史持续降权 + **空间复用（邻域 tap）停用**。
///
/// 为什么空间复用要停用：邻域 tap 会把邻像素的值混进来（权重可到 4:1），
/// 于是"某个像素变暗/变亮"只能靠**扩散**传播过去 ⇒ 收敛被拖成慢慢爬，
/// 变暗时还会被邻域的 [0.5×, 2×] 钳制卡住（"收敛不干净"）。
/// 过渡期让像素用自己的样本直接决定估计值 ⇒ 响应直达；过渡完再把邻域复用接回来。
///
/// 取 64 ≈ 降噪时域外窗（`GI_DEN_M_MAX`）：过渡期内降噪历史已经追到新亮度，
/// 结束时把权重交回 1.0 不会把旧值拽回来。
const GI_HIST_RESET_FRAMES: u32 = 64u;
const GI_HIST_SEARCH_TAPS: u32 = 8u;
const GI_DEN_GUIDE_WORDS: u32 = 6u;
const GI_DEN_HIST_WORDS: u32 = 9u;
const GI_DEN_G_KEY0: u32 = 0u;
const GI_DEN_G_KEY1: u32 = 1u;
const GI_DEN_G_NRM: u32 = 2u;
const GI_DEN_G_PREV: u32 = 3u;
const GI_DEN_G_NOISE: u32 = 4u;
/// 导引里的「本次历史被降权」标志（1 = 降权）。
///
/// 世界内容变了（体素被改 / 物体出现或消失）时，屏幕空间历史与降噪历史**不是丢弃、而是降权**
/// （见 `GI_HIST_RESET_SCALE`）：丢掉 ⇒ 那一帧只剩几条新样本 ⇒ 整屏 GI 重新抖一遍（"闪一下"）；
/// 降权 ⇒ 旧值按小权重继续参与，新值平滑地把它带过去 ⇒ 无闪、且响应比稳态快 1/权重 倍。
const GI_DEN_G_RESET: u32 = 5u;
const GI_DEN_H_COL: u32 = 0u;
const GI_DEN_H_M1: u32 = 3u;
const GI_DEN_H_M2: u32 = 4u;
const GI_DEN_H_M: u32 = 5u;
const GI_DEN_H_KEY0: u32 = 6u;
const GI_DEN_H_KEY1: u32 = 7u;
const GI_DEN_H_NRM: u32 = 8u;
const GI_DEN_ATROUS_ITER: u32 = 5u;
const GI_DEN_ATROUS_R: u32 = 2u;
const GI_DEN_ATROUS_R_FAST: u32 = 1u;
const GI_DEN_STAB_FRAMES: f32 = 8.0;
const GI_DEN_STAB_MAX: f32 = 3.0;
const GI_DEN_NBR_R: u32 = 2u;
const GI_DEN_M_MAX: u32 = 64u;
const GI_DEN_SIGMA_ALPHA: f32 = 0.02;
const GI_DEN_W_HIST_MIN: f32 = 0.05;
const GI_DEN_CLAMP_K: f32 = 2.5;
const GI_DEN_N_DOT: f32 = 0.7;
const GI_DEN_LUMA_K: f32 = 1.0;
const GI_DEN_LUMA_FLOOR_REL: f32 = 0.05;

const GI_SKY_AMBIENT: f32 = 0.05;
const GI_AMBIENT_FLOOR: f32 = 0.01;
const SHADOW_SURFACE_EPS: f32 = 0.03125;