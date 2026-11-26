# 临时交接（2026-11-27）

分支 `restir-gi`。本文件为临时交接，不需要长期保留。

## 已完成并提交

| commit | 内容 |
| --- | --- |
| `a1d8ea9` | HDR + bloom + 色调映射档位（Off / AgX / ACES / Neutral，默认 AgX） |
| `4857251` | 反射叠加空气透视 |
| `56d3b78` | 镜面 GI 对世界本体的采样、玻璃亮度、water 系列 flag 清理 |

`56d3b78` 要点：

- `gi_merge_one` 不再拦截 `obj < 0`，世界本体（green_field 为 `obj_id = -1`）的反射 GI 请求能够进入采样。
- `gi_merge_req` 容量按请求数算：`(arrayLength - 1) / GI_MERGE_WORDS`；计数器改由 Rust 侧 dispatch 后 `clear_buffer`。
- 补样 seed 混帧号，反射 GI 持续补到 `n >= 8`；队列满时回退读缓存而不是置黑。
- 反射 GI 的 face 与 `voxel_normal_fast` 选出的 face 统一；读值走 `irc_display`（与主视图同一显示路径）。
- `water_miss` / `water_surf` / `refract_hit` / 水下太阳行程比全部删除，玻璃只按真实命中点着色。
- 折射玻璃记录前表面，避免把不在原始射线上的折射命中点投影进 `dda_hit`。

`AGENTS.md` 当前仍有未提交的本地修改，与上述工作无关，不要顺手带进提交。

## 后续工作顺序

1. 逐对象 voxel 尺寸（voxel unit）
2. PBR 材质映射
3. 高度图 → 体素位移
4. 云（C+ 混合方案）
5. 方向性散射

### 1. 逐对象 voxel 尺寸

现状：`assets/shaders/voxel_raytrace/common.wesl` 有全局常量

```wgsl
const MATERIAL_VOXEL_PER_METER: f32 = 50.0;
```

它假设所有世界都是 50 vox/m，对 green_field（16 vox = 1 m）是错的，PBR 贴图密度、mip LOD、位移幅度、空气透视尺度都会跟着错。

目标：

- 每个世界 / 物体携带自己的 `voxel_m`（每米体素数），不再有全局常量。
- green_field 为 16；其他场景按各自换算。
- 作为 per-scene / per-object 元数据进 uniform 或 grid desc，随物体一起上传。
- 所有用到“体素 ↔ 米”的地方统一走这一个值：`pbr_mip_lod`、`triplanar` 的 `uv_scale`、位移幅度、雾/空气透视距离、任何以米为单位的半径。
- 检查 `fetch_material` 之外还有没有硬编码 50 或按米换算的常量。

### 2. PBR 材质映射

现状：`gate-render/src/pbr_texture.rs` 已有 16 套 PBR 资源（`textures/pbr/<id>_albedo.jpg`、`_roughmetal.jpg`、`_height.png`），采样已在 `common.wesl` 的 `fetch_material` 里按 palette 的 `FLAGS_IS_PBR` 走 triplanar。

目标：

- 方块名 → 16 套 PBR 资源的映射表，放在 `gate-app/src/mc/voxel.rs` 的调色板构建处（现在只做颜色平均）。
- 逐体素采样 albedo / roughness / metal，继续用 triplanar。
- **必须保持逐体素平色**：一个体素六个面同一个颜色。贴图采样只允许在体素内部取一次值（按体素中心或固定 uv），不能引入逐像素细节，否则画风被破坏。
- 颜色与粗糙度贴花留在逐体素层面；真正的 3D 浮雕只做 ≥ 2–3 体素尺度的特征。
- 法线贴图不转几何，只做高度图。

### 3. 高度图 → 体素位移

现状：`gate-app/src/height_field.rs` 与 `gate-app/src/edit.rs` 已有笔刷位移（`MaterialDisplaceCache`、`displace_fn`、`fill_brush_displaced`），但只在编辑笔刷里用。

目标：

- 用 `_height.png`，先减去最大值再刻，**只向内挖**，不向外长。
- 每材质有默认最大幅度，物体可覆盖。
- 薄墙（1–2 体素）按半厚度钳制，避免挖穿。
- 在加载时或近 LOD 烘焙进 `VolumeGrid`，不做逐帧运行时位移。
- 复现 `height_field.rs` 的缓存与采样逻辑，避免重复实现。

### 4. 云（C+ 混合）

- 外部视角：2–3 层视差 2.5D 云层，不用 billboard。
- 进入云层内部：局部真 3D 密度 raymarch，约 30–80 m 范围。
- 动态云影只影响直接太阳光，**绝不写进静态 IRC**（IRC 是世界空间辐照度缓存，烘焙云影会造成不散的错误暗区）。

### 5. 方向性散射

- 大气散射带方向性（前向/后向），与现有 `sky_ambient` / 空气透视共用 `LightGlobals`。
- 保持逐体素着色：同一体素各面共用散射颜色，只随体素中心到相机距离与方向变化。

## 构建 / 验证

```bash
cargo build --profile build-fast -p gate-app
cargo clippy --profile build-fast --workspace --all-targets --all-features -- -D warnings
cargo fmt --all -- --check
```

- 禁止 debug 构建（`gate-*/build.rs` 在 `PROFILE=debug` 时 panic）。
- DLSS 构建走 `powershell -NoProfile -ExecutionPolicy Bypass -File tools/build-dlss.ps1 -Profile build-fast`。
- 改 `.wesl` 会触发 `gate-render/build.rs` 的编译期校验；新增 compute 入口要同步 `ENTRY_POINTS`。
- 改 IRC 布局 / 常量必须同步 `gi/consts.wesl`、`src/wesl_consts.rs`、`src/gi.rs` 的 `gi_bg5_layout()`。
- 正确性靠实机运行验证，不新增测试。

## 必须守住的不变量

- 逐体素着色：一个体素六个面同一个平色，不引入逐像素 / 逐面差异。
- 空气透视按体素中心到相机的距离与方向混向 `sky_ambient`，不透明面和玻璃都要叠加。
- 隐式法相梯度退化时按体素自身邻域选轴，不按命中面取法线。
- 编辑只 `light_rev += 1` 冲掉 ReSTIR 时域水库；不得置 `light_jump`、不得压 `hist_cap`。只有主题/曝光、调色板重绘、整世界重建才允许 `HIST_CAP_RESET`。
- 不要做相机静止时的纯静态复用偷懒优化。

## 已知待调参数

- `REFL_GI_MERGE_N = 8`（`main.wesl`）：镜面 GI 仍有轻微残余噪声时先升到 16 / 32，不要改显示路径。
- bloom 强度固定 `0.08`，阈值 `1.0`，暂无滑杆；Off 档把 bloom 强度设 0。
