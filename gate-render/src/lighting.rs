//! 光源 wire 契约 + 数据驱动主题（方向光硬阴影 / sky 纯色环境光 / 发光体素直出）；CPU 侧只做数据打包。
//! Uniform 布局（WGSL `LightPool` 逐字段镜像）：LightGlobals(48B) + 8×LightDesc(384B) + sky_color(16B)。

use bevy::ecs::resource::Resource;
use bevy::render::render_resource::ShaderType;
use glam::{Vec3, Vec4};
use serde::Deserialize;

/// 光源上限（uniform 数组长度；当前只用 lights[0] = 方向光）
pub const MAX_LIGHTS: usize = 8;

/// 光源描述（shader 镜像，48B；uniform 数组 stride 16 的倍数 ✓）
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, ShaderType)]
pub struct LightDesc {
  /// x = kind（0 = 方向光）；yzw = L 轴（指向光，已归一）
  pub kind_pos_dir: Vec4,
  /// rgb = 线性色，w = 强度
  pub color_intensity: Vec4,
  /// **天体盘**的外观（只被 `volumetric.wesl::sky_primary` 读，见 `gate-render/src/sky.rs`）：
  /// x = 盘的亮度倍数（乘 WESL `SUN_DISK_GAIN`；太阳 `1`、月亮 `0.6`，见 `sky.rs` 的 `MOON_DISK_SCALE`）；
  /// yzw reserved（恒 0）—— **光晕**不在这里，它是大气散射，走光柱 uniform 的「光晕」。
  pub shape: Vec4,
}

/// 光池 header（48B）：count + 环境色 + 曝光
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, PartialEq, ShaderType)]
pub struct LightGlobals {
  pub count: u32,
  /// **镜面档位**（[`ReflectionSettings::tier`]，0 = 关 ..= 3 = 最高）：主 pass 的不透明镜面反射
  /// 按它决定"反射命中点着色到哪一档、要不要补阴影射线"。
  /// 原本是废弃填充 `_pad0`（恒 0）⇒ 复用空闲字节，48B 布局不变。写侧 = `prepare_dda_bind_groups`。
  pub refl_tier: u32,
  /// **镜面嵌套层级**（[`ReflectionSettings::nest`]，取值 0 / 1 / 2 / 4）：允许"镜子里的镜子"
  /// 再反射几级。同样复用废弃填充 `_pad1`。写侧 = `prepare_dda_bind_groups`。
  pub refl_nest: u32,
  pub _pad2: u32,
  /// rgb = 环境色（线性），w reserved
  pub ambient: Vec4,
  /// x = 曝光系数；yzw reserved
  pub exposure_pad: Vec4,
}

/// BG3 uniform 整体，WGSL `LightPool` 逐字段镜像。
#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Resource, ShaderType)]
pub struct LightPoolUniform {
  pub g: LightGlobals,
  pub lights: [LightDesc; MAX_LIGHTS],
  /// 天空纯色（miss 背景与 sky 环境光共用）
  pub sky_color: Vec4,
}

/// **镜面档位**（菜单「渲染/反射」）：`main.wesl` 里那条不透明镜面反射射线"追到什么程度"。
///
/// 全部档位都保持**逐体素面**量化（视向量 `face_view_dir`、抖动种子、法线都按面取 ⇒ 一个面一个
/// 反射值），所以镜面是"块状镜面"，与逐面平色的画风一致 —— 档位只改**反射内容的着色口径**，
/// 不改着色粒度。
///
/// - `0` = 关：不发反射射线（回到"只有 `F0·(amb+gi)` 环境近似"的旧口径）；
/// - `1` = 反射命中点带上太阳直射与自发光（**0 条额外射线**）：反射内容是"命中点自己的出射亮度"，
///   而不是只有天光环境项（后者比反射到的天空暗约 20 倍 ⇒ 镜像里除了天空亮斑什么都没有）；
/// - `2` = 再加"逃逸天空走 `sky_primary`"（**0 条额外射线**）：镜像里能看见太阳盘与光晕；
/// - `3` = 再给反射命中点补一次太阳 NEE 阴影射线（**每条反射射线 +1 次 DDA 遍历**）：镜像里的
///   明暗/阴影正确；前两档是"无遮挡近似"——背光的墙在镜子里也是亮的。
///
/// 与 `main.wesl` 的 `PBR_REFLECTION_ENABLED`（编译期总开关）的关系：那条是 A/B 自检用的**编译期**
/// 常量（关掉整段反射代码被折掉），本资源是**运行期**档位，只在总开关打开时起作用。
#[derive(Resource, Clone, Copy, Debug, PartialEq, bevy::render::extract_resource::ExtractResource)]
#[extract_app(bevy::render::RenderApp)]
pub struct ReflectionSettings {
  /// 档位（0..[`Self::TIERS`]），写入 uniform `LightGlobals::refl_tier`。
  pub tier: u32,
  /// **镜面嵌套层级**：镜子里的镜子再反射几级。取值是 [`Self::NEST_CHOICES`] 里的**枚举值**
  /// （0 / 1 / 2 / 4，不是连续档位号）；写入 uniform `LightGlobals::refl_nest`。
  ///
  /// - `0` = 不嵌套（**默认**）：反射命中点只做完整着色、不再发反射射线 —— 这就是"镜子里看不到
  ///   另一面镜子里的倒影"的那个行为；
  /// - `N` = 最多再嵌 `N` 级：反射射线打到一个**本身也是镜面**的面时，继续沿它反射 ⇒ 最多
  ///   `N + 1` 条反射射线串成一条链。
  ///
  /// **成本模型（为什么可以放心开）**：链条只在"上一级命中的面本身也是镜面"时才继续 ——
  /// 墙、地形、普通方块都会在**第一级就断链** ⇒ 普通场景下 0/1/2/4 的画面与成本**完全一样**
  /// （多出来的只是一次判据）。钱只花在"真的把镜子对着镜子"的地方：最坏每个触发像素
  /// `N + 1` 条反射射线（档 3 时每条再 +1 条阴影射线）。
  ///
  /// **画风不变**：每一级的反射方向都由**逐面**视向量给出（`face_view_dir` + 按面种子）⇒
  /// 嵌套的镜像仍然是块状镜面，与逐面平色同格。
  pub nest: u32,
}

impl ReflectionSettings {
  /// 档位数（菜单 `switch_group` 的选项数）。
  pub const TIERS: u32 = 4;
  /// 嵌套层级的四个档（关 / 1 层 / 2 层 / 4 层）；**下标 = `switch_group` 的选中序号**，值 = 实际层级。
  /// 菜单侧只认下标 ⇒ 改这里的值等于改档位语义（`debug_menu.rs` 的观察者用同一份）。
  /// 取 0/1/2/4 而不是 0..3：层级是**成本倍数**（`N + 1` 条射线），跳档才有意义。
  pub const NEST_CHOICES: [u32; 4] = [0, 1, 2, 4];

  /// 当前档位（钳到合法范围）。
  pub fn tier(&self) -> u32 {
    self.tier.min(Self::TIERS - 1)
  }

  /// 当前嵌套层级（把枚举值钳成 `NEST_CHOICES` 里最接近的合法值）。
  pub fn nest(&self) -> u32 {
    *Self::NEST_CHOICES
      .iter()
      .min_by_key(|c| c.abs_diff(self.nest))
      .expect("NEST_CHOICES 非空")
  }
}

impl Default for ReflectionSettings {
  /// 默认 **2 + 不嵌套**：0 条额外射线就拿到"带太阳直射 / 自发光 / 太阳盘"的镜像，
  /// 是"像镜子"与"不多发一条射线"之间的性价比档；要镜像里阴影正确再拨到 3（每条反射射线 +1 次 DDA）。
  /// 嵌套默认关（= 现有表现），要"镜子里的镜子"再自己拨（见 [`Self::nest`] 的成本模型）。
  fn default() -> Self {
    Self { tier: 2, nest: 0 }
  }
}

/// 方向光配置（主题资产 = 静态；时间驱动时由 `sky::apply_sky` 每帧覆写）
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct DirLightCfg {
  /// 光传播方向（指向场景）；打包时翻转为 L（指向光）
  pub dir: [f32; 3],
  /// 太阳盘角半径（rad）——硬阴影管线不使用此值
  #[serde(default = "default_angular_radius")]
  pub angular_radius_deg: f32,
  pub color: [f32; 3],
  pub intensity: f32,
  /// **天体盘**的亮度倍数（乘 WESL `SUN_DISK_GAIN`）：`1` = 太阳那个盘；
  /// 月亮用远小于 1 的值 ⇒ 亮而不白（见 `sky.rs` 的 `MOON_DISK_SCALE`）。
  /// 盘**外面**那圈光晕不在这里 —— 它是大气散射，走光柱 uniform 的「光晕」（随时间变）。
  #[serde(default = "one")]
  pub disk_scale: f32,
}
fn default_angular_radius() -> f32 {
  0.0
}
/// 新增可选项的缺省：`1` = 不改变既有观感（`assets/lighting/*.ron` 不写这两个字段也是原样）。
fn one() -> f32 {
  1.0
}

/// 天空颜色配置（纯色）
#[derive(Debug, Clone, PartialEq, Deserialize)]
pub struct SkyCfg {
  pub color: [f32; 3],
}

/// 光照主题（`assets/lighting/*.ron`）：方向光 + 环境 + 天空 + 曝光
/// ExtractResource：main world 资源自动提取进 render world（BG3 光池数据源）
#[derive(
  Debug, Clone, PartialEq, Resource, Deserialize, bevy::render::extract_resource::ExtractResource,
)]
#[extract_app(bevy::render::RenderApp)]
pub struct LightingTheme {
  pub sun: Option<DirLightCfg>,
  pub ambient: [f32; 3],
  pub exposure: f32,
  pub sky: Option<SkyCfg>,
}

impl Default for LightingTheme {
  fn default() -> Self {
    Self {
      sun: Some(DirLightCfg {
        dir: Vec3::new(0.5, -0.8, 0.3).normalize().to_array(),
        angular_radius_deg: 0.0,
        color: [1.0, 0.96, 0.88],
        intensity: 0.8,
        disk_scale: 1.0,
      }),
      ambient: [0.08, 0.09, 0.12],
      exposure: 1.0,
      sky: Some(SkyCfg { color: crate::consts::MINECRAFT_SKY }),
    }
  }
}

/// RON 解析
pub fn parse_lighting_ron(src: &str) -> Result<LightingTheme, ron::error::SpannedError> {
  ron::de::from_str(src)
}

/// 构建光池 uniform：方向光（lights[0]）+ 天空 + 环境 + 曝光
pub fn build_light_pool(theme: &LightingTheme) -> LightPoolUniform {
  let mut u = LightPoolUniform {
    g: LightGlobals {
      count: 0,
      // 这两格恒 0：真正的值由 `prepare_dda_bind_groups` 从 `ReflectionSettings` 覆写
      // （本函数只认主题资产，反射档位不是资产属性）。
      refl_tier: 0,
      refl_nest: 0,
      _pad2: 0,
      ambient: Vec4::new(theme.ambient[0], theme.ambient[1], theme.ambient[2], 0.0),
      exposure_pad: Vec4::new(theme.exposure, 0.0, 0.0, 0.0),
    },
    lights: [const {
      LightDesc { kind_pos_dir: Vec4::ZERO, color_intensity: Vec4::ZERO, shape: Vec4::ZERO }
    }; MAX_LIGHTS],
    sky_color: Vec4::new(
      crate::consts::MINECRAFT_SKY[0],
      crate::consts::MINECRAFT_SKY[1],
      crate::consts::MINECRAFT_SKY[2],
      0.0,
    ),
  };
  if let Some(sky) = &theme.sky {
    u.sky_color = Vec4::new(sky.color[0], sky.color[1], sky.color[2], 0.0);
  }
  if let Some(sun) = &theme.sun {
    let l = -Vec3::from(sun.dir).normalize_or_zero();
    u.lights[0] = LightDesc {
      kind_pos_dir: Vec4::new(0.0, l.x, l.y, l.z),
      color_intensity: Vec4::new(sun.color[0], sun.color[1], sun.color[2], sun.intensity),
      // 天体盘的外观（盘亮度倍数）：`sky_primary` 靠它区分太阳盘与月亮盘（负值当 0）。
      shape: Vec4::new(sun.disk_scale.max(0.0), 0.0, 0.0, 0.0),
    };
    u.g.count = 1;
  }
  u
}
