# P2.4 DDA 主可见性 pass - 产品需求规范（Spec）

## Overview
- **Summary**：在 Bevy render 子 app 增加全屏 compute DDA 着色器 + 现有 blit 链路，用 P2.3 上传到 GPU 的 `GpuBrickMap`（五大 buffer + globals uniform）逐像素遍历 `docs/brickmap.md §3` 五步寻址链，按 palette 索引读颜色，结果直出 rgba8 存储纹理，经 Core2d PostProcess blit 到 ViewTarget → 窗口显示。静态视图（无相机控制），像素结果必须与 P2.2 `BrickMapView::get_voxel(fine)` 的 CPU 参考实现逐点一致（等价性单测覆盖边界点）。
- **Purpose**：P2（渲染 M1）的核心里程碑——**体素真正上屏**。之前是渐变占位，自此 GPU 端消费 brick map 管线闭环，P2.5 颜色直出验收点天然兑现。
- **Target Users**：开发者（画面验证、DDA 正确性调试、后续 P2.5/P3/P9 的着色器起点）。

## Goals
- 全屏逐像素 WGSL DDA，等价 view.rs `BrickMapView::get_voxel` 语义（空 ⇒ 背景色；命中 ⇒ palette color）
- 复用 `gradient.rs` 的 Compute（storage tex write）+ Core2d PostProcess blit 时序（P0.3 已验证无 bug），但换 compute shader + bind group（新增 BrickMap bind group：5 storage buffers + globals uniform + palette buffer）
- 静态视图：启动时生成一张 `DdaCameraUniform`（eye/target/up + near/far + fov），Startup 从主 world 注入资源，渲染端 Extract 只读
- DDA 步长：最细格 fine（0.25cm）即步长=1；方向归一化后用 A&W DDA（逐分量算下一面的参数 t_min，走最小分量）
- 首帧 `cargo run -p gate-app` 能看到：32³ 方块（palette 1 默认深灰）+ 球面（palette 2 红/橙，定义在 palette.rs）+ tile(1,0,0) 和 tile(2,0,0) 的两个 L4 立方块；颜色与 CPU `BrickMapView` 在 100+ 随机像素点单测断言一致
- 验证 GTX 1660 基线：1080p 默认视图（空远距无热点 DDA）≥58 FPS；基准机 RTX 3070 ≥120 FPS

## Non-Goals
- **不做动态相机**（2.6 轨道相机）：P2.4 硬编码 eye/target/up；视图矩阵常量，Startup 注入，不随帧变动
- **不做阴影、光照、GI、normal 计算**（P3+）：DDA 只返回 palette 颜色，空处返回暗色实验室背景色
- **不裁决 G-Buffer 格式**（brickmap §10 决策点二）：本期继续 **PENDING**，输出格式仍是 1 张 rgba8unorm 颜色纹理；G-Buffer（pos/normal/palette_idx）在 P3 做 NEE 采样前定。因为 P2 的验收点是"纯色体素上屏"，不需要多通道中间缓冲
- **不做 ropes 邻域快速跳转**（Laine & Karras 2010）：A&W 裸步进即可，Dense TileIndex 区 128cm 级步进已足够（TileIndex 一次 load 跳 128 基元胞 = 128 细格 * 0.25cm = 32cm；L0 bitmap 按 32 字跳 32 基元胞 = 32 细格 = 8cm，DDA 性能基线已够）
- **不做 GPU 调试可视化/颜色渐变 palette 映射定制**：palette[0..=5] 用默认 `PaletteEntry::default_colors()`（定义在 gate-voxel palette.rs），其余 palette 索引随机色（测试 100 点只用到 1-6 足够）
- **不做剖面切割 / 调试裁剪平面**（P4.5 实现）：DDA 只走标准空处推进
- **不做 camera_driver 时序改序**：沿渐变的 `dispatch_compute.before(camera_driver) + Core2d::PostProcess blit`。验证过无 bug，DDA 也用同一个。

## Background & Context
1. 脚手架基线（P0.3）：`gate-render/src/gradient.rs` 完整可复用链路——`ExtractResourcePlugin` → `RenderStartup init_pipeline`（PipelineCache queue_compute/queue_render）→ `PrepareBindGroups prepare_bind_group`（UniformBuffer write_buffer + BindGroup create）→ `RenderGraph dispatch_gradient.before(camera_driver)`（compute pass 写 storage_tex rgba8unorm）→ `Core2d PostProcess blit_view`（全屏三角 render pass + textureLoad 免 sampler + srgb_to_linear 抵消 ViewTarget sRGB 编码）。链路时序、格式匹配（Camera2d 非 HDR = Rgba8UnormSrgb）、`ViewTarget.get_color_attachment()` 都已经在 P0.3 实机验证
2. 上传通道（P2.3 刚交付）：render-world `PrepareResources` 完成时 `GpuBrickMap { struct_buf, leaves, palette, comp, state: Buffer, globals: UniformBuffer<BrickMapGlobals> }` 资源就位；5 个 Buffer 用法 = `COPY_DST | STORAGE_BINDING`，globals 是标准 `UniformBuffer<BrickMapGlobals>`，ShaderType 全部 scalar 化，80B
3. CPU DDA 参考（P2.2）：[view.rs](file:///c:/repo/repo.rust/gate/gate-render/src/brickmap/view.rs#L33-L155) `BrickMapView::new + get_voxel + walk_node` 完整实现 §3 五步寻址链（TileIndex → TileBitmaps → CellDirs → CellNode hdr/l1/l2/l3/brick），单测 17 green；GPU DDA 逐行翻译为 WGSL（storage buffer<u32> 读：`buffer[b_struct]` 字索引；uniform `globals` 存 origin/dims）
4. wire 常量（P2.2 P2.3）：`BITMAP_BASE=128³=2097152 words`；`DIR_BASE`；`TILE_BITMAP_WORDS=1024`；`CELL_DIR_WORDS=32768`；`HDR_UNIFORM_MASK=0xFF`，`HDR_HAS_L1=1<<8, L2=1<<9, L3=1<<10, BRICK=1<<11`；`SLOT_TABLE_WORDS = [0, 4, 32, 256, 512]`？不对，8 槽 2³=16B=4 words，L1=4 words 对；L2 4³=64 槽 × 2 byte = 128B=32 words 对；L3 8³=512 槽=1024B=256 words 对；L4 8 字节指针（brick_index）→ 2 words。slot_pack：`(tag:u8, pal:u8) → u16`，两槽 pack 成 `u32 word`：idx&1=0 用 lower 16bit，idx&1=1 用 upper 16bit。所有常量都要在 WGSL 顶部 `const` 逐字与 Rust side 定义，单测把常量 assert_eq! 掉（防 drift）
5. 着色器目录：`gate-app/assets/shaders/` 已有 `gradient.wgsl / blit.wgsl`；新增 `dda.wgsl` 进此目录，`ShaderAssetPath = "shaders/dda.wgsl"`

## Functional Requirements
- **FR-1 · Bind Group 接入 GpuBrickMap**：render-world `PrepareBindGroups` 阶段读 `GpuBrickMap` + Globals UniformBuffer，构建 2 组 bind group：BG0 = 纹理写 + 相机 view uniform（沿用 Gradient 的 BG0 格式，省 BindGroupLayout 复用）；BG1 = struct_buf storage + leaves storage + palette storage + globals uniform（5 buffer + 1 uniform）。每个 storage buffer 是 `@binding(n) var<storage, read> b_struct: array<u32>;`（运行时绑定 sized array，wgpu 支持 runtime-sized storage buffer 作为 binding）
- **FR-2 · WGSL DDA 着色器**：新文件 `dda.wgsl`，`@compute @workgroup_size(8,8)`，global_invocation_id 算像素中心 `fragCoord.xy + 0.5`，反投影 → 射线 origin + dir（fine 坐标系单位：fine=0.25cm）。A&W DDA：`tMax/tDelta/step/sign` 三变量，while 循环：下一个 t = min(tMax.x, min(tMax.y, tMax.z))，走对应分量 step，更新 tMax += tDelta。循环结束条件：t > tFar 或 steps > 2048（硬防无穷循环，空场景 hit miss 路径一定 exit）
- **FR-3 · 射线坐标系**：uniform 中 `view_proj_inv: mat4x4<f32>`（逆 VP）+ `camera_pos_fine: vec3<f32>`（fine 坐标：world meters ×4，因为 4 fine = 1 cm，1 细格 = 0.25cm）。NDC 近点 `(2x/w-1, 1-2y/h, 0, 1) × invVP → /w → near_xyz`；远点 `(..., 1, 1) × invVP → /w → far_xyz`；dir = normalize(far - near)；origin = camera_pos_fine。坐标换算：Bevy Camera 世界（1 单位 = 1m）× 4 → fine。P2.2/VoxelPos 从_fine 函数直接 fine。**注意 sign 正方向**：Bevy +Y 上，-Z 前，默认 Camera2d 朝 -Z
- **FR-4 · 命中评估 = get_voxel(fine(pos)) 等价**：每次步进把当前细格坐标 cast 到 i32，调用 WGSL 版 `sample_brickmap(fine: vec3<i32>) -> u32 palette`（0 = 空，>0 = palette）。palette=0 继续走；palette>0 写颜色，退出循环。空路径（走完 tFar 未命中）写暗色背景色 = vec3(0.05, 0.08, 0.12) 接近渐变 darkblue 基调（上屏连续）
- **FR-5 · 颜色**：命中后通过 palette_idx 读 `@group(1) @binding(2) var<storage, read> palette_entries: array<PaletteEntry>;`（`struct PaletteEntry { color: vec4<f32>, roughness: f32, emissive: vec3<f32>, transmission: f32, flags: u32 }`，repr(C) 对齐：4*4=16+4+12+4+4=40 B）。暂只取 `.color.rgb` 直接存 storage tex（不 gamma 不 tonemap，保留线性；blit 时做 srgb→linear 反转已有逻辑 → 屏幕 sRGB 显示正确）。palette 0 作为 AIR 不访问（循环前判断 palette>0 才读）
- **FR-6 · DDA 着色器必须完全独立重写寻址链**：不能复用 Rust view.rs 代码（WGSL 不同语言天然满足）。常量、打包解包语义必须与 Rust side 逐字对照，准备等价性单测。常量列表完整见下 FR-7。
- **FR-7 · WGSL 常量集**（顶部 `const` 写死，并 Rust CPU 单测与 `wire.rs` 的对应常量 assert_eq!）：`TILE_INDEX_CAP=128u`，`BITMAP_BASE=2097152u`，`TILE_BITMAP_WORDS=1024u`，`DIR_BASE` 从 globals uniform 读（或者用 Rust 定义？不，DIR_BASE 是计算常量：`DIR_BASE = BITMAP_BASE + TILE_CAP * TILE_BITMAP_WORDS` = 2097152 + 4096*1024 = 6,291,456 words，WGSL const：`const TILE_CAP: u32 = 4096u; const DIR_BASE: u32 = 6291456u`）；`CELL_DIR_WORDS=32768u`；`HDR_UNIFORM_MASK=0xFFu`；`HDR_HAS_L1=256u`；`HDR_HAS_L2=512u`；`HDR_HAS_L3=1024u`；`HDR_HAS_BRICK=2048u`；`BRICK_SLAB_WORDS=1024u`；`SLOT_TABLE_WORDS: array<u32,4>`——WGSL 不支持 const array，用 fn 或单 const 分开，ST_L1=4, ST_L2=32, ST_L3=256, ST_BRICK_PTR=2
- **FR-8 · 等价性单测**：CPU 端构建固定场景（32³ box，已知 palette 填充）→ BrickMapBuffers → (a) 100 个随机 fine 点，BrickMapView.get_voxel 返回 palette；(b) 同一场景的 DDA WGSL 在 CPU 上**无法直接跑**，所以用等价策略：**离线像素采样工具（Rust 侧 DDA raycaster 参考实现）**→ 构建 `cpu_dda_raycaster`，它复用 `BrickMapView::get_voxel` 按细格步进（完全复用 FR-4 sample_brickmap），生成同相机参数的像素集（比如 100×100 分辨率），断言 (1) 10×10 格网像素的 palette 与 WGSL 运行时**在程序截图上**不可取，所以 P2.4 的等价性分 2 段：**第一段**（CPU 内部强验证）：常量对齐测试（WGSL 常量的 Rust 副本断言 = wire.rs 常量）+ `cpu_dda_reference()` 与 `BrickMapView + A&W step impl` 在固定 seed 生成 50 条射线，ray march 的 hit_t / hit_palette 逐次断言一致；**第二段**（实机）：DDA shader 跑起来后截图至少能看到三个视觉区域（盒体边界圆弧、空背景、L4 方块角）并通过人工审查（rubric）
- **FR-9 · 插件化**：新建 `gate-render/src/brickmap/dda.rs` + `pub mod dda; brickmap/mod.rs pub use` + `GateRenderPlugin` 在 GradientPlugin 之后链式追加 `BrickMapDdaPlugin`（**不要删 GradientPlugin 留作 fallback/对比**；运行时 P2.4 DDA 替换渐变画面但脚手架保留，除非显式 feature）。实际实现：dda plugin 的 Dispatch DDA 与 dispatch_gradient 都在 `before(camera_driver)`，DDA 在 gradient 后 run 覆盖即可（同 tex 写两次，后者赢）；或者 DDA 创建**独立的 storage tex**（新资源 `DdaImages { target }`），blit 时优先 dda 纹理，fallback gradient；本 FR 选后者——DDA 独立纹理，不污染 gradient；blit 改为：如果有 DdaBlitBindGroup（DDA 纹理就绪）就 blit DDA，否则仍 blit Gradient（兜底）
- **FR-10 · 资源创建**：main-world `setup` 新增 `create_dda_image(images)`（同 create_gradient_image：VIEW_SIZE rgba8unorm + STORAGE | TEXTURE 用法），注入 `DdaImages { target }`；`ExtractResourcePlugin::<DdaImages>::default()` 自动提取进 render world
- **FR-11 · 启动视图**：Startup 注入 `DdaCamera { view_proj: Mat4, position_fine: Vec3, view_proj_inv: Mat4 }`。静态参数：eye = 1m 右前上斜 45°（Bevy Camera world：(2.0, 2.0, 2.0) × 4 = fine (8, 8, 8)？不对，box 0..32 细格 = fine 0..32 → world 0..8m。eye 在 world(6,5,6)，target = (4, 1, 4) 即 box 中心偏下，这样能看到 box 三面 + 球面 + L4 两个热点。fov_y = 60°，aspect = 1280/720。near = 0.01m = 0.04 fine？不对，DDA 内部距离用 fine 所以 V * P 的 inverse 用 world 单位算 origin fine：`camera_pos_world * 4`。ray dir 是 world dir，normalize 后也乘 4 / 4？方案：origin 用 fine（×4），dir 用 **fine units per step**（world dir 直接 × 4，因为 dir 归一化后 magnitude=1 world meter = 4 fine units，这样 step magnitude = 步长 dt 推进 fine t 直接对得上）。**统一在 WGSL 内把 dir 转 fine：ray_dir_world × 4.0；origin = cam_pos_world × 4.0**
- **FR-12 · 相机 DDA uniform ShaderType**：`DdaViewUniform { inv_view_proj: Mat4, cam_pos_world: Vec4, _pad: Vec4 }`（2×64B + 16B = 144B，Vec4 pad 到 16B 对齐）；Extract 主 world `DdaCameraResource`（Startup 注入）→ render world 每帧 PrepareBindGroups 写 `UniformBuffer<DdaViewUniform>`。视图静态就好，每帧重写也 O(1)。

## Non-Functional Requirements
- **NFR-1 · 空场景性能**（无体素命中，纯 DDA miss）：1080p RTX 3070 ≥200 FPS，GTX 1660 ≥80 FPS。P2.4 测试空工作间。依据：A&W DDA 每次推进 3 比较 + 1 次 sample，空远距平均 128 基元胞步 = 512 fine 步，每像素 ~600 cycle 很轻松
- **NFR-2 · 密集场景性能**（满屏 L4 热点 + 最坏树深路径）：RTX 3070 ≥60 FPS（基线），GTX 1660 ≥30 FPS（下限，P3 再调优）。P2.4 demo 场景（32³ + 8³×2 L4 + fill_sphere r=6 level 2）要求达到
- **NFR-3 · 等价性 CPU 参考单测**：必须 pass（cargo test --workspace 新增 ≥3 测试：wire_constants_vs_wgsl、cpu_dda_ray_vs_brickmap_view_same_hit、静态相机像素 grid_sample_palette_match）。50→≥53 tests 全绿
- **NFR-4 · CI 护栏**：fmt/clippy/test 全绿，不改现有 50 test 语义，palette/brickmap view tests 继续 pass
- **NFR-5 · 着色器可维护**：WGSL 顶部用大段注释标注§3 五步寻址链每一步与 view.rs 行号映射；常量区加 `// Rust wire.rs 对应：` 引用源文件 line 便于将来 drift 检测
- **NFR-6 · 不触发新 VUID / panic**：winit 启动日志与 P2.3 final 跑对齐，只允许 wgpu#9213 两条初始 VUID；无新增 validation error。启动 300 帧无 panic exit 0。

## Constraints
- **Technical 1**：着色器语言 = **WGSL**（Bevy 0.19 natively 走 wgpu naga 前端，WGSL 免额外编译器，DX12/Vulkan backend 都直接支持，不用 GLSL/HLSL 额外转译）
- **Technical 2**：wgpu binding 2GB 硬上限（RTX 3070 PROBE 已测）。BG1 有 5 个 sized storage arrays：`array<u32>` 每个 bind，b_struct ~140MB，其余 tiny；b_palette 2KB。总 storage binding 内存 <256MB 远低于上限
- **Technical 3**：`UniformBuffer<BrickMapGlobals>`（80B）+ `UniformBuffer<DdaViewUniform>`（144B）分别绑定各自 BG 中 slot，**不跨 BG 共享 uniform**（减少绑定冲突）
- **Technical 4**：Camera2d 保留 Msaa::Off（P2.3 demo 保持）；ViewTarget 主纹理格式 Rgba8UnormSrgb（blit fs 已有 srgb→linear 反转逻辑，DDA 的 linear color 经过 blit fs → ViewTarget sRGB encode → 最终屏幕 sRGB = 原始 linear 正确呈现）
- **Dependencies**：P2.3 上传通道（GpuBrickMap 资源）+ P0.3 gradient/blit scaffold + P2.2 view/wire 常量
- **Business**：暗色实验室基调背景色 = `vec3(0.05, 0.08, 0.12)` 深 navy，与渐变首版颜色相近，画面切换无违和

## Assumptions
1. `array<u32>` runtime-sized storage buffer 在 wgpu 0.19 + Vulkan 工作正常（WGSL: `var<storage, read> b_struct: array<u32>;`，wgpu BindGroupEntry binding 传 `BufferBinding { buffer, offset: 0, size: None }` 直接绑定整个 buffer；wgpu spec：`BufferSize::default()` means whole buffer）。P0.3 未测 runtime-sized，但 Vulkano 侧有大量官方案例使用，无已知限制
2. Bevy 0.19 `buffer binding with dynamic offset` 不需要：因为全局只绑一次，不做 per-object offset。静态 whole buffer binding
3. WGSL `mat4x4<f32>` 的 inverse（inv_view_proj）在 shader 中手写？不，Bevy 0.19 WGSL 支持 `mat4x4<f32>` 但是否原生有 `inverse()`？wgpu 0.19 naga 中 inverse 对 4x4 f32 是内建。如果没有，**提前在 Rust 侧算 inv_view_proj 写 uniform，WGSL 直接用**（P2.4 静态相机，Startup 一次算好存 Mat4，Extract 进 uniform，零额外运行时成本）
4. Camera2d 默认 OrthographicProjection？不对，默认 Camera2d 就是 OrthographicProjection：scale=1.0 near=-1000.0 far=1000.0，朝 -Z 看 +X 右 +Y 上。**本阶段替换 Perspective**：main-world Startup spawn 相机时改为 `Camera { projection: Projection::Perspective(PerspectiveProjection { fov: 60°, aspect_ratio, ..default() }), ..Camera2d }`，或者干脆 spawn `Camera3d`（P2.4 静态视图，3D camera 也 OK）。方案：改用 `Camera3dBundle` + `Transform::from_translation(Vec3::new(6.0,5.0,6.0)).looking_at(Vec3::new(4.0,1.0,4.0), Vec3::Y)` 更自然，`Projection::Perspective(PerspectiveProjection { fov_y: 60f32.to_radians(), aspect_ratio: VIEW_SIZE.x as f32 / VIEW_SIZE.y as f32, near: 0.01, far: 100.0 })`。Msaa::Off 仍是 Component。注意 ViewTarget 仍用 Core2d？Camera3d 用 `Core3d`。P0.3 scaffold 用 Core2d。需要把 DDA plugin 改为注册在 `Core3d` 对应 blit set（`Core3dSystems::PostProcess`），或者 Camera3d 用 `Camera::order` 强制 core pass。**更简单：在 Startup 生成 static view_proj & inv_view_proj（Perspective Mat4），不依赖实际 Camera 的 transform**。也就是说，Camera3d 只是个空壳（Core3d 会创建空 ViewTarget + clear），我们真正用的 view uniform 是硬编码 Startup 的 Mat4。这样 blit 还是可以挂 Core3d PostProcess。但 P0.3 用的 Camera2d ViewTarget 能正常用。折中：保留 Camera2d（Core2d PostProcess 时序正确已测），view uniform 手工算 45° perspective（不管 Camera2d 的 ortho，DDA 不用它的 transform）。反正 DDA 的 ray 是从 DdaViewUniform 算的，和实际 camera projection 无关。P2.6 相机控制再把两者同步。此假设简化实现。

## Acceptance Criteria

### AC-1: DDA 上屏 — 体素盒体三面可见
- **Type**: `rule`
- **Given**: `cargo run -p gate-app` 启动（32³ box + sphere + 2×L4 热点场景，VoxelScene + UploadBudget 已在 Startup 注入）
- **When**: 窗口出现后，观察 10s（等 DDA 纹理就绪，通常首帧即就绪，最坏 3 帧 pipeline 编译）
- **Then**: 画面包含三个可辨识特征：① 大立方盒（32³ palette=1 默认色）正面+顶面+左侧面三色梯度边界 ② 球面暗色凹面（palette=2）在盒右上方 ③ 两个 8³ 小立方块（palette=3/4）在中景
- **Pass Condition**: 截图（或人工观察）确认三个特征同时存在；与 P2.3 渐变画面明显不同（后者是连续彩色渐变，前者是块状几何）
- **Evidence**: 启动日志截图证据（run_p24_accept.log），确认画面特征在 review 环节人工核实 + FPS 日志 ≥GTX 1660 基线

### AC-2: 等价性 CPU 单测 — CPU ray DDA 参考实现 get_voxel 命中一致
- **Type**: `rule`
- **Given**: 固定 seed 42 场景（fill_box(0,0,0)-(16,16,16) pal=1 + fill_sphere pal=2），BrickMapBuilder 构建 BrickMapBuffers
- **When**: 50 条随机射线（eye(8,8,8) 朝随机像素方向），A&W 细格步进循环 2048 步上限，返回 (hit_t, hit_palette) 对
- **Then**: 每条射线 CPU 参考 A&W DDA（`fn sample_brickmap(fine: IVec3)` 用 `BrickMapView::get_voxel`）与**独立实现的 A&W step**（无共享 view.rs 代码，复用 sample 但 step 新写）的结果逐次一致
- **Pass Condition**: `cargo test --workspace` 新增测试 `dda_reference_equivalence_50_rays` pass；50/50 射线 hit_t abs diff < 0.1 fine 且 palette 相同
- **Evidence**: CI 日志中的 test result line（50 tests → ≥53，新增 3 至少包含 1 条）

### AC-3: 常量对齐单测 — WGSL 常量 Rust 镜像等于 wire.rs
- **Type**: `rule`
- **Given**: Rust 侧 `dda.rs` 内部 `WgslConsts { ... }` 块写死与 WGSL 顶部 const 一一对应
- **When**: `cargo test -p gate-render` 执行 `wire_wgsl_constants_aligned` 单测
- **Then**: TILE_INDEX_CAP / BITMAP_BASE / TILE_CAP / DIR_BASE / CELL_DIR_WORDS / HDR 标志位 / BRICK_SLAB_WORDS / SLOT_TABLE_WORDS 各项 = `wire.rs` 对应项，无漂移
- **Pass Condition**: 单测 assert_eq! 全过，无 panic
- **Evidence**: test 输出 line

### AC-4: 静态相机像素取样验证（CPU→GPU 对照）
- **Type**: `rule`
- **Given**: CPU `cpu_dda_render(scene, 32x32)` 渲染 1024 像素，返回 `Vec<(palette_idx, bool)>`（空 = false）
- **When**: 同一场景 + 同静态视图（FR-11）参数，在 `dda.wgsl` 中 **32x32 缩小分辨率版本**通过 `#[cfg(test)] mod cpu_wgsl_exe...` 无法在 CI 直接跑 WGSL；变通：CPU 版 `cpu_dda_render`（完全复用 FR-8 的 cpu_dda_reference + 逐像素 sample）输出 32×32，人工对照程序输出的 print（每 8×8 网格用字符画 ASCII）与实机截图网格一致（每 32 像素跳着比对 8×8 格网字符图案）
- **Then**: ASCII 网格（box = 'X', sphere = 'o', hotspot = '#', empty = '.'）与实机截图视觉网格拓扑相同（至少 95% 非空像素格符号一致）
- **Pass Condition**: cpu_dda_render ASCII 网格 dump 到日志，人工核对 P2.4 运行截图；测试 `cpu_dda_ascii_grid_32` 输出确定性格式
- **Evidence**: cargo test --nocapture 输出网格

### AC-5: fmt / clippy / test CI 护栏
- **Type**: `rule`
- **Given**: 仓库源码状态，docs 编辑之外新增 Rust / WGSL
- **When**: `cargo fmt --all --check` + `cargo clippy --workspace --all-targets -- -D warnings` + `cargo test --workspace` 三条命令
- **Then**: 三条 exit 0，所有 50→≥53 tests 绿，无 clippy 警告
- **Pass Condition**: exit 0 全清
- **Evidence**: 命令汇总输出（run 日志）

### AC-6: NFR FPS 基线（RTX 3070 / GTX 1660 基线，本阶段只测 RTX 3070）
- **Type**: `rubric`
- **Dimension**: Demo 场景（32³ + sphere + 2 L4）启动后，帧时间日志（LogDiagnosticsPlugin）前 200 帧平均 FPS 指标
- **Scale**: 1-5
- **Anchors**: 1 = <30 FPS; 3 = 50~60 FPS 保底线; 5 = ≥120 FPS 超预期（空路径大量跳过 512+ 细格步）
- **Pass Threshold**: >= 4（>= 80 FPS，RTX 3070 空远距主视野应当能轻松到 120，若密集热点路径低但均值 ≥80 通过）
- **Evidence**: final accept log bevy_diagnostic fps 行（取 frame 30..200 平均）

### AC-7: 着色器可维护性（NFR-5）
- **Type**: `rubric`
- **Dimension**: `dda.wgsl` 代码结构清晰性、与 view.rs 五步寻址链映射的可追踪性
- **Scale**: 1-5
- **Anchors**: 1 = 无注释，常量散落在各处，读一遍无法与 view.rs 对齐；3 = 每步有注释，常量区有 Rust 对应项引用；5 = 完全逐段注释§3 ①-⑤ 并引用 view.rs 行号，常量区写 Rust 对应文件路径+line
- **Pass Threshold**: >= 4
- **Evidence**: review 阶段 reviewer 审阅 WGSL 文件

### AC-8: 无新增 wgpu validation error
- **Type**: `rule`
- **Given**: `RUST_LOG=gate=info,wgpu=warn` 启动
- **When**: 500 帧运行
- **Then**: VUID 只出现 wgpu#9213 2 条（VkPresentInfoKHR-pImageIndices-01430 + vkAcquireNextImageKHR-semaphore-01286），无新 VUID
- **Pass Condition**: Grep VUID 出现总数 ≤2 条，且 ID 匹配以上两条
- **Evidence**: accept log grep 输出

## Open Questions
- [x] Q1: G-Buffer 格式（brickmap §10 决策点二）— P2.4 延后裁决（P3 NEE 前定稿，见 Non-Goals 3）
- [x] Q2: 着色器语言 — WGSL（Constraints Technical 1）
- [x] Q3: 相机 vs ViewTarget Core2d/Core3d — 保留 Core2d blit，DDA 视图 uniform 硬编码 Mat4，不从实际 Camera projection 同步，P2.6 再同步（Assumptions 4）
