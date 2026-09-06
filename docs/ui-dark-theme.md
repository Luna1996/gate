# gate 暗色 UI 主题设计文档

> 规格来源：StyleKit Dark Mode UI（https://www.stylekit.top/zh/dark-mode-ui-prompts ）
> 落地目标：gate-ui（bevy_ui 原生 + 自研 widget 层）一套 token 统一**游戏内 UI** 与**调试 UI**。
> 本文档只定规格，不含代码改动。所有颜色为最终验收值；落地时改 `theme.ron` + `ThemeColors` 结构镜像。

---

## 1. 设计原则

暗色界面不是颜色反相。五条风格铁律（StyleKit）+ gate 的四条适配/总则：

1. **永远不用纯黑** `#000000`。OLED 拖影、层级线索失效。最暗从 `#09090B` 起步，给层级留出向下空间。
2. **亮度即海拔**。暗色里阴影不可见（bevy_ui 也不支持投影），用 5 层表面灰阶表达层级：越靠近用户越亮，每层亮约 3–4%，层间用 **1px 低对比边框**分隔。
3. **白字压光，三档文字**。纯白 `#FFFFFF` 在暗底上晃眼，禁止使用。主文本 `#FAFAFA` / 正文 `#D4D4D8` / 说明 `#A1A1AA`。
4. **一个强调色，花在刀刃上**。界面 90% 是灰阶。强调蓝只给四类元素：主操作按钮、当前选中态、关键数据（折线/读数）、focus ring。到处上色等于没有重点。
5. **对比度是硬指标**。正文 ≥ 4.5:1（WCAG AA），说明文字目标 AAA；`#52525B` 一档（2.6:1）禁止上线。
6. **gate 适配 A：UI 浮在 3D 体素场景上**。不透明表面用于菜单/模态；HUD 与调试 overlay 用半透明表面，alpha 不得低于 90%（`E6`），保证文字在最亮场景底色上仍过 AA（见 §4 复合验算）。
7. **gate 适配 B：一切颜色走 token**。禁止在 widget / overlay 代码里硬写 `Color::WHITE`、`Color::BLACK`、裸 rgba（现状 `main.rs` 调试面板黑底 0.65、白色折线均为待迁移项）。
8. **用户明确指令凌驾所有设计规则之上**。本文档是默认风格基线，不是约束：任何具体界面，用户的明确要求（配色、布局、透明度、字体、是否圆角等）直接覆盖对应规则，无需先改文档再执行；被覆盖的规则只在该指令范围内失效，其余界面继续遵守本基线。
9. **默认字体固定为 MapleMono NF CN**。路径 `gate-app/assets/fonts/MapleMono-NF-CN-Regular.ttf`（绝对路径 `c:\repo\repo.rust\gate\gate-app\assets\fonts\MapleMono-NF-CN-Regular.ttf`），经 `theme.ron` 的 `font_path` 配置并由主题系统 override Bevy 默认字体槽，全项目 UI 文本（游戏 UI + 调试 overlay）统一使用，不引入第二套字体；除非用户明确指定其他字体。

---

## 2. 颜色令牌

### 2.1 表面层级（5 层）

| Token | 色值 | 层级 | gate 用途 |
|---|---|---|---|
| `surface_base` | `#09090B` | L0 基底 | 模态 scrim 底色、全屏遮罩、空画布底 |
| `surface_card` | `#131316` | L1 卡片 | `panel()` 默认面、设置面板、HUD 卡片、调试 overlay 底板 |
| `surface_elevated` | `#1A1A20` | L2 抬升 | 次级按钮面、输入框、checkbox/slider 轨道槽、卡片 hover |
| `surface_overlay` | `#222228` | L3 浮层 | 下拉菜单、弹出层、RingList 浮层、表面 hover 态 |
| `surface_top` | `#2A2A31` | L4 顶层 | tooltip、toast、最高优先级浮层 |

半透明变体（仅用于浮在 3D 场景上的 HUD / 调试 overlay）：

| Token | 色值 | 用途 |
|---|---|---|
| `surface_card_hud` | `#131316E6` | 游戏 HUD、左上角调试面板（90% alpha） |
| `scrim` | `#09090B99` | 模态对话框背后遮罩（60% alpha） |

菜单、设置页、模态面板本体用**不透明** L1–L3；只有直接压在游戏画面上的常驻元素用 `_hud` / `scrim`。

### 2.2 边框（3 档，均为 1px）

| Token | 色值 | 用途 |
|---|---|---|
| `border_subtle` | `#232329` | L1 卡片与 L0 基底之间的分隔 |
| `border` | `#2A2A31` | 默认边框：面板、按钮、输入框 |
| `border_strong` | `#3F3F46` | hover / 选中 / 聚焦邻近时提亮 |

边框只做分隔与 hover 反馈，不做高对比描边。

### 2.3 文字（4 档）

| Token | 色值 | 用途 |
|---|---|---|
| `text_primary` | `#FAFAFA` | 主文本：KPI 读数、FPS 数值、面板标题、主按钮文字 |
| `text_body` | `#D4D4D8` | 正文：按钮标签、列表条目、FPS 行标签 |
| `text_muted` | `#A1A1AA` | 说明文字：轴标签、单位后缀、相机信息行、次级提示 |
| `text_faint` | `#71717A` | **仅限 ≥18px 的大号弱化标签**；小于 18px 禁止使用 |

### 2.4 强调色（蓝，唯一彩色主角）

| Token | 色值 | 用途 |
|---|---|---|
| `accent_fill` | `#2563EB` | 主按钮默认填充（blue-600，白字 5.2:1） |
| `accent_fill_hover` | `#3B82F6` | 主按钮 hover 填充（blue-500，提亮一档） |
| `accent_fill_pressed` | `#1D4ED8` | 主按钮按压填充（blue-700，压深一档） |
| `accent_text` | `#60A5FA` | 暗底上的强调文字/图标/**折线**/focus ring（blue-400，7.8:1） |

规则：
- 填充用 600/500/700 三态；暗底上的彩色文字、图标、数据线一律用 400（`accent_text`），不要用 fill 色做文字。
- 主按钮文字统一 `text_primary`。
- **强调色可替换性**：若「暗色实验室」主题日后要换青/琥珀，只改这 4 个 token 并复核 §4 对比度；5 层灰阶与文字档不动。

### 2.5 语义色（降饱和，暗底安全）

| Token | 色值 | 用途 |
|---|---|---|
| `success` | `#4ADE80` | 成功/通电状态文字与图标（green-400） |
| `success_fill` | `#22C55E` | 成功态填充 |
| `warning` | `#FBBF24` | 警告文字（amber-400）；填充时标签须用深色文字 |
| `warning_fill` | `#F59E0B` | 警告态填充 |
| `danger` | `#F87171` | 危险/错误文字（red-400） |
| `danger_fill` | `#DC2626` | 破坏性按钮填充（白字 ≥4.5:1） |

语义色同样遵守"花在刀刃上"：同一屏内语义色只标状态，不做装饰。

### 2.6 与现有 theme.ron 字段映射

| 现字段（Catppuccin Mocha） | 现值 | 新 token |
|---|---|---|
| `panel_bg` | `1E1E2ECC` | `surface_card` / `surface_card_hud` |
| `panel_border` | `45475ACC` | `border` |
| `text` | `CDD6F4` | `text_body`（标题用 `text_primary`） |
| `text_muted` | `9399B2` | `text_muted` |
| `accent` | `89B4FA` | `accent_fill`（按钮）/ `accent_text`（文字线条） |
| `accent_hover` | `74C7EC` | `accent_fill_hover` |
| `accent_pressed` | `7287FD` | `accent_fill_pressed` |
| `success` | `A6E3A1` | `success` |
| `warning` | `F9E2AF` | `warning` |
| `danger` | `F38BA8` | `danger` |
| —（硬编码） | 黑 `0.65` | `surface_card_hud` |
| —（硬编码） | `Color::WHITE` 折线 | `accent_text` |

---

## 3. 度量令牌

| Token | 值 | 说明 |
|---|---|---|
| `corner_radius` | 8.0 | 容器：面板、按钮、弹窗（rounded-lg，现状保留） |
| `corner_radius_sm` | 4.0 | 小控件：checkbox、slider thumb、标签 chip |
| `border_width` | 1.0 | 所有边框统一 1px |
| `spacing.xs/sm/md/lg` | 4 / 8 / 12 / 16 | 现状保留；调试 overlay 紧凑组用 xs，常规面板用 sm–lg |
| `font_size.sm/md/lg` | 12 / 14 / 18 | 轴标签/相机信息=sm；正文/按钮/FPS 行=md；面板标题/KPI=lg |
| 字体 | MapleMono-NF-CN | 项目唯一字体（现状保留）。等宽 → 数字读数/表格天然对齐；单字重，强调靠颜色与字号，不靠字重 |

面板内边距：左右 `lg`(16)、上下 `md`(12)（`panel()` 现状保留）。
调试 overlay 内边距：`xs`(4)，行间距 `xs`（FPS 行 + 折线 + 相机信息现状保留）。

---

## 4. 对比度验收表（WCAG 相对亮度）

| 前景 / 背景 | 对比度 | 等级 | 用途 |
|---|---|---|---|
| `#FAFAFA` / `#09090B` | 19.1:1 | AAA | 基底上主文本 |
| `#D4D4D8` / `#131316` | 12.6:1 | AAA | 卡片上正文 |
| `#A1A1AA` / `#09090B` | 7.8:1 | AAA | 基底上说明文字 |
| `#A1A1AA` / `#131316` | 7.2:1 | AAA | 卡片上说明文字 |
| `#60A5FA` / `#09090B` | 7.8:1 | AAA | 强调文字/图标/折线 |
| `#FAFAFA` / `#2563EB` | 5.2:1 | AA | 主按钮文字 |
| `#3B82F6` / `#09090B` | 5.4:1 | AA | 强调填充边界（WCAG 1.4.11） |
| `#71717A` / `#09090B` | 4.1:1 | AA-LG | 仅 ≥18px 大号弱化标签 |
| `#52525B` / 任意暗底 | 2.6:1 | **FAIL** | 禁止上线 |

**HUD 半透明最坏情况复合验算**：`surface_card_hud`（`#131316` @ 90%）盖在纯白场景上，复合后约 `#2A2A2A`：

- `#D4D4D8` 正文 → 9.7:1（AAA）
- `#A1A1AA` 说明 → 5.6:1（AA）

结论：90% alpha 是下限，不许更低；HUD 上文字最低用 `text_muted`，需要更高可读性处用 `text_body`。

---

## 5. 组件配方

所有配方只引用 §2/§3 token；bevy_ui 0.19 能力：`BackgroundColor`、`BorderColor`、`BorderRadius`、`Outline`（focus ring，支持 offset）、`Transform`（按压缩放）。

### 5.1 panel（`panel()`）

- 背景 `surface_card`（HUD 场景用 `surface_card_hud`）；边框 1px `border`；圆角 8。
- 不使用阴影。层级靠所在表面层与边框表达。
- 模态对话框：L3 `surface_overlay` 不透明 + 背后 `scrim`。
- tooltip / toast：L4 `surface_top`，圆角 8，文字 `text_body`。

### 5.2 button（三变体）

| 变体 | 默认 | hover | pressed | 文字 |
|---|---|---|---|---|
| primary（主操作） | 填充 `accent_fill` | `accent_fill_hover` | `accent_fill_pressed` | `text_primary` |
| secondary（次操作） | 填充 `surface_elevated` + 边框 `border` | 填充 `surface_overlay`，边框 `border_strong` | 填充 `surface_card` | `text_body` |
| ghost（工具栏/低调操作） | 透明 | 填充 `surface_elevated` | 填充 `surface_card` | `text_muted` → hover `text_body` |

统一规则：
- 圆角 8，1px 边框，padding 左右 md / 上下 sm（现状保留）。
- **按压反馈**：pressed 时 `Transform::scale(0.98)`，hover/默认回 1.0（触觉确认，不可省略）。
- **focus ring**：键盘聚焦时 `Outline` 2px `accent_text` + offset 2px（与底色分离才可见）。
- disabled：填充/文字降为 `text_faint` 档表面，不响应交互。
- 破坏性操作用 `danger_fill` 替换 primary 填充栈。
- 可选增强：primary 按钮顶部 1px 高光节点（白 10% alpha）模拟暗色光源；bevy_ui 无 inset shadow，用子节点实现，不做不阻塞。

### 5.3 checkbox

- 盒子：16×16，圆角 4，未选中填充 `surface_elevated` + 边框 `border`。
- hover：边框 `border_strong`。
- 选中：填充 `accent_fill`，对勾 `text_primary`。
- 标签文字 `text_body`，禁用态 `text_faint`（标签 ≥ sm 12px，faint 仅用于 disabled 场景时改用 `text_muted`）。

### 5.4 slider

- 轨道槽：高 4px，圆角 2，填充 `surface_elevated`。
- 已填充段：`accent_fill`。
- thumb：16×16 圆形（圆角 999），`surface_top` + 1px 边框 `border_strong`；拖拽时放大到 18×18。
- 值文字：`text_muted`，拖拽中变 `text_primary`。

### 5.5 list / RingList

- 条目默认透明、文字 `text_body`；hover 条目填充 `surface_elevated`；选中条目填充 `surface_elevated` + 左侧 2px `accent_text` 指示条（或文字改 `accent_text`，二选一，不叠加）。
- 浮层形态（下拉选择）：容器 L3 `surface_overlay` + 边框 `border` + 圆角 8。

### 5.6 label

- `label()` = `text_body` / md；`label_muted()` = `text_muted`。
- 标题：lg + `text_primary`；读数/KPI：lg（或更大）+ `text_primary`，单位后缀同字号 `text_muted`。
- 禁止 `Color::WHITE`。

### 5.7 plot（折线图，CPU 光栅化）

- 画布透明（现状保留），由底板面板透出 `surface_card(_hud)`。
- **折线**：`accent_text` `#60A5FA`（替换现状 `Color::WHITE`）——关键数据是强调色的唯一用法之一。
- 网格线：白 6% alpha（约 `#FFFFFF0F`），只画横线，不抢折线。
- 纵轴标签 / 极值 / 单位后缀：`text_muted`，sm 12px。
- 左上角读数（真实帧时，格式 `000.0` 不加单位）：`text_primary`，md。
- 多条数据线时（未来）：主序列 accent，次序列用 `text_muted` 或白 40% alpha，不引入新彩色。

### 5.8 world_anchor（3D 锚定标签）

- 背景 `surface_card_hud` + 边框 `border`，圆角 4（小标签用 sm 档）。
- 文字 `text_body` sm；被遮挡/背面剔除时整体隐藏，不做 X-ray 半透明（与项目剖面切割约定一致）。

---

## 6. 调试 overlay 专项（左上角 FPS 面板）

现状：黑底 0.65 无圆角边框、白色折线、两行纯 ASCII/文本。目标：

| 元素 | 现状 | 目标 |
|---|---|---|
| 底板 | `Color::srgba(0,0,0,0.65)` 硬编码 | `surface_card_hud`（`#131316E6`）+ 1px `border` + 圆角 8 |
| FPS 行 | 单色文本 | 标签 `FPS:` 用 `text_muted`；CUR/AVG/MIN/MAX 数值用 `text_primary`；其余 `text_body` |
| 帧时折线 | `Color::WHITE` | `accent_text`；网格线白 6% |
| 纵轴标签 | 随 widget | `text_muted` sm，`000.0` 定宽格式不变 |
| 相机信息行 | 单色 | `text_muted` sm（eye/yaw/pitch/dist 每 0.25s 刷新节奏不变） |

调试 UI 与游戏 UI 视觉同源：同样的表面、边框、字体、文字档；区别仅在紧凑间距（xs）与半透明底板。GATE_BENCH 后台诊断逻辑（logs/frame_time.log、gpu_frame.log）不受影响。

---

## 7. 游戏内 UI 专项

- **HUD**（常驻、压画面）：`surface_card_hud` 卡片 + `border`，圆角 8；信息文字 `text_body` 以上。
- **暂停/设置菜单**：不透明 L1 `surface_card` 面板 + `scrim` 遮罩；面板内分区用 L2 `surface_elevated` 嵌块，不靠画线。
- **模态确认**：L3 `surface_overlay` 面板 + `scrim`；主操作 accent 按钮居右，取消用 secondary/ghost。
- **弹窗层级**：tooltip/toast(L4) > 模态(L3) > 菜单(L1) > HUD(L1 hud)；bevy_ui 以后 spawn 为上，spawn 顺序按此层级。
- 体素编辑器的组件状态高亮（emissive 调制）属于 3D 场景内表达，不走 UI token；但编辑器面板内的状态文字用 `success`/`warning`/`danger`。

---

## 8. 状态总表

| 状态 | 表面 | 边框 | 文字 |
|---|---|---|---|
| default | 按组件所属层 | `border` | `text_body` |
| hover | 提亮一档（L1→L2→L3） | `border_strong` | `text_body`/`text_primary` |
| pressed | 压深一档 + scale 0.98 | `border` | 不变 |
| disabled | `surface_card` | `border_subtle` | `text_muted`（禁 faint 于小字） |
| focus（键盘） | 不变 | `Outline` 2px `accent_text` + 2px offset | 不变 |
| selected | `surface_elevated` | `accent_text` 指示条 | `text_body` 或 `accent_text` |

---

## 9. DO / DON'T

**要**
- 新表面先选 5 层之一；新文字先选 4 档之一；不自创灰。
- 面板/卡片用 1px 边框分隔层，hover 时边框提亮。
- 90% 界面保持灰阶；强调蓝只给主按钮、选中态、关键数据、focus ring。
- 可交互元素必须有 hover 提亮 + pressed 反馈（scale 0.98）。
- 焦点环必须带 offset，与底色分离。
- HUD 半透明面板 alpha ≥ 90%，文字 ≥ `text_muted`。
- 圆角：容器 8，小控件 4。

**不要**
- 不用纯黑 `#000000` 背景（scrim 也用 `#09090B`）。
- 不用纯白 `#FFFFFF` 文字/线条（含折线图）。
- 不在暗底上用饱和填充色做文字（文字用 400 档，填充用 500/600 档）。
- 不引入第二个强调色；语义色只标状态。
- 不用高对比边框、不用阴影表达层级。
- 不在 <18px 文字上使用 `text_faint`。
- 不在代码里硬编码颜色，一律走 `UiTheme` token。

---

## 10. 附录：目标 theme.ron 形态（落地参考，本文档不执行）

```ron
(
  colors: (
    // 表面 5 层
    surface_base:        "09090B",
    surface_card:        "131316",
    surface_elevated:    "1A1A20",
    surface_overlay:     "222228",
    surface_top:         "2A2A31",
    surface_card_hud:    "131316E6",
    scrim:               "09090B99",
    // 边框
    border_subtle:       "232329",
    border:              "2A2A31",
    border_strong:       "3F3F46",
    // 文字 4 档
    text_primary:        "FAFAFA",
    text_body:           "D4D4D8",
    text_muted:          "A1A1AA",
    text_faint:          "71717A",
    // 强调色
    accent_fill:         "2563EB",
    accent_fill_hover:   "3B82F6",
    accent_fill_pressed: "1D4ED8",
    accent_text:         "60A5FA",
    // 语义色
    success:             "4ADE80",
    success_fill:        "22C55E",
    warning:             "FBBF24",
    warning_fill:        "F59E0B",
    danger:              "F87171",
    danger_fill:         "DC2626",
  ),
  metrics: (
    corner_radius: 8.0,
    corner_radius_sm: 4.0,
    border_width: 1.0,
    spacing: (xs: 4.0, sm: 8.0, md: 12.0, lg: 16.0),
    font_size: (sm: 12.0, md: 14.0, lg: 18.0),
  ),
  ui_scale: 1.0,
  auto_fit_ui_scale: true,
  font_path: Some("fonts/MapleMono-NF-CN-Regular.ttf"),
)
```

### 落地步骤（执行时另行开工）

1. 扩 `ThemeColors`（10 → 25 字段）与 `ThemeMetrics`（+`corner_radius_sm`），内置默认主题同步换成本文档色值。
2. 更新 `theme.ron`；字段缺失回退机制保持现状。
3. widget 逐个对齐 §5 配方：panel / button（三变体 + scale + focus ring）/ checkbox / slider / list / label / plot（网格线 + token 化线色）。
4. `main.rs` 调试 overlay 迁移：黑底 → `surface_card_hud` + 边框圆角；白折线 → `accent_text`；FPS/相机行分档上色。
5. 全局搜硬编码 `Color::WHITE`/`Color::BLACK`/裸 rgba UI 用法，清干净。
6. 验证：release build 启动 gate-app，读 `logs/` 确认无主题加载 warn；按项目规矩不自行截图，渲染正确性由日志与人工目检确认。

---

## 11. 组件 API 约定（v2：Config / Handle / 事件）

widget 层统一为「三件套」自由函数 API，无 builder、无变体函数堆叠。约定见
`gate-ui/src/widgets/mod.rs` 顶层注释，本节为文档版。

### 11.1 三件套

1. **`XxxConfig`**：全部参数进 Config（含必填如 `text`），可选字段 `..default()` 补齐。
   - 布局字段（宽高/边距/flex）**不进 Config**——调用方拿到 Handle 后自行改 `Node`。
   - 预设档用 enum 表达（`LabelStyle`/`PanelSurface`/`ButtonVariant`/`PlotLayout`），
     颜色与字号永远成对来自主题 token，杜绝非法组合。
2. **`XxxHandle`**：spawn 返回类型，`Deref` 到根实体 `Entity`（零开销新类型）。
   多实体 widget 用具名字段：`ScrollViewHandle { entity, content }`、
   `TabViewHandle { entity, contents }`。不挂 sugar 方法，按需再加。
3. **事件**：仅交互型 widget 发事件；展示型不发。
   - 组件永远是真源，事件只是通知——主动读值仍可 Query（如 `SliderValue`/`Checked`）。
   - 命名：动作用名词（`UiClick`），变化用 -ed 过去式
     （`SliderValueChanged`/`CheckboxToggled`/`TabChanged`）。

豁免：`splitter` 无任何配置项，不设 Config；`row`/`column`/`spacer` 等布局容器
不纳入本层，直接用 bevy_ui 原生 `Node`。

### 11.2 组件清单

| widget | Config | Handle | 事件 |
|---|---|---|---|
| button | `ButtonConfig { text, variant }` | `ButtonHandle` | `UiClick` |
| checkbox | `CheckboxConfig { text, checked }` | `CheckboxHandle` | `CheckboxToggled { entity, checked }` |
| slider | `SliderConfig { min, max, value, step }` | `SliderHandle` | `SliderValueChanged { entity, value }` |
| tab_view | `TabConfig { tabs, active }` | `TabViewHandle { entity, contents }` | `TabChanged { entity, index }` |
| label | `LabelConfig { text, style }` | `LabelHandle` | — |
| panel | `PanelConfig { surface }` | `PanelHandle` | — |
| scroll_view | `ScrollConfig { height }` | `ScrollViewHandle { entity, content }` | — |
| plot | `PlotConfig { layout, image, capacity, … }` | `PlotHandle` | — |
| list | `ListConfig { capacity }` | `ListHandle` | — |
| table | `TableConfig { headers, rows }` | `TableHandle` | — |
| splitter | （无） | `Entity` | — |

### 11.3 用法示例

```rust
// spawn：Config 进、Handle 出
let b = button(&ctx, row, ButtonConfig { text: "primary".into(), ..default() });
let sv = scroll_view(&ctx, p, ScrollConfig { height: Val::Percent(100.0) });
// 多实体 widget 用具名字段
world.entity_mut(sv.content).with_children(|c| { ... });

// 交互：观察者订阅事件（组件仍是真源）
world.add_observer(|ev: On<SliderValueChanged>, mut q: Query<&mut Text>| {
  // ev.entity / ev.value
});

// 布局：拿 Handle 改 Node（Config 不含布局字段）
world.entity_mut(*b).insert(Node { flex_grow: 1.0, ..default() });
```

### 11.4 新增 widget 守则

- 先判交互型/展示型：交互型必须定义事件（真源组件 + -ed 通知），展示型零事件。
- 参数一律进 Config 并实现 `Default`；样式预设一律 enum，不新增变体函数。
- Handle 只包 Entity 与具名子实体，方法延迟到出现真实重复时再挂。
