# P2.4 DDA 主可见性 pass - 实现计划（tasks.md）

## Task 1: DDA 视图 uniform 资源（相机静态 Mat4）+ main-world 接入
- **Status**: `pending`
- **Priority**: `high`
- **Depends On**: None
- **Description**:
  - 新建 `DdaCameraConfig { view_proj: Mat4, inv_view_proj: Mat4, position_world: Vec3 }` Resource（主 world Startup 注入），静态视图：eye(6.0,5.0,6.0), target(4.0,1.0,4.0), up=Y, fov_y=60°, aspect=1280/720, near=0.01m, far=100m；手算 Mat4::perspective_rh + Mat4::look_at_rh → view_proj，inverse 即 inv_view_proj
  - 新建 `DdaViewUniform`（render-world 绑定用，#[derive(ShaderType)]）：`inv_view_proj: Mat4, cam_pos_fine: Vec4, _pad: Vec4`（144B，16B 对齐）；cam_pos_fine = position_world * 4.0，Vec4 为 w=1
  - 新建 `DdaImages { target: Handle<Image> }`（ExtractResource derive），新增 `create_dda_image()` 工厂函数（复用 create_gradient_image 同配置：VIEW_SIZE rgba8unorm + STORAGE|TEXTURE + RENDER_WORLD usage）
  - main.rs `setup`：spawn 图片 + 插入 `DdaImages` + 插入 `DdaCameraConfig`（保留 Camera2d + Msaa::Off + Gradient 不动）
  - gate-render lib.rs pub use 导出 `DdaCameraConfig / DdaImages / create_dda_image`；brickmap/mod.rs pub mod dda（提前占位，Task 3 填充）
- **Acceptance Criteria Addressed**: FR-1, FR-10, FR-11, FR-12, AC-5
- **Test Requirements**:
  - `rule` TR-1.1: `cargo check -p gate-app` 通过（资源无类型错误）
  - `rule` TR-1.2: `cargo clippy -p gate-app -- -D warnings` clean
  - `rule` TR-1.3: Mat4 可逆：`assert_eq!(view_proj * inv_view_proj, Mat4::IDENTITY, eps=1e-5)` 单测通过

## Task 2: WGSL 常量镜像单测 + 槽位打包/解包原语单测（CPU 端，FR-7）
- **Status**: `pending`
- **Priority**: `high`
- **Depends On**: None
- **Description**:
  - `gate-render/src/brickmap/dda.rs` 中定义 `mod constants;` 内嵌 module：`WGL_TILE_INDEX_CAP / WGL_BITMAP_BASE / WGL_TILE_CAP / WGL_DIR_BASE / ... / WGL_ST_L1 = 4, ST_L2 = 32, ST_L3 = 256, ST_BRICK_PTR = 2` 等，全部以 `u32` Rust const 镜像声明
  - 单测 `wire_wgsl_constants_aligned`（AC-3）：assert_eq! 每一项 == wire.rs 对应常量（BITMAP_BASE / DIR_BASE 等；HDR_UNIFORM_MASK=0xFF 等）
  - 单测槽位打包：Rust 端写独立 `wgsl_style_slot_pack(tag: u8, pal: u8) -> u16` + `wgsl_slot_unpack(word: u32, half: u8) -> u16`，与 `wire.rs` 的 `pack/unpack` 对 200 组随机 (tag, pal) 断言一致。验证 WGSL 的位操作与 Rust 端等价
- **Acceptance Criteria Addressed**: AC-3, FR-7, NFR-3
- **Test Requirements**:
  - `rule` TR-2.1: `wire_wgsl_constants_aligned` pass（≥16 常量逐一比对）
  - `rule` TR-2.2: `slot_pack_roundtrip_wgsl_vs_rust_200_random` pass（200/200 tag,pal 对一致）

## Task 3: CPU DDA 参考实现（等价性测试，FR-8 + FR-4）
- **Status**: `pending`
- **Priority**: `high`
- **Depends On**: Task 2（常量镜像可用）
- **Description**:
  - `dda.rs` 中实现独立的 A&W 细格步进：`fn cpu_reference_dda_ray(buffers: &BrickMapBuffers, origin_fine: Vec3, dir_fine: Vec3, t_max: f32, max_steps: u32) -> Option<(f32, u8)>`，A&W step 完全新写（不得复用 view.rs 任何辅助结构，仅调用 `BrickMapView::get_voxel(fine_pos)` 读 palette）。tMax/tDelta/step 三变量，每次取 min 分量步进
  - 单测 `dda_reference_equivalence_50_rays`（AC-2）：固定 seed 场景（fill_box 16³ palette 1 + sphere r=4 level 2 pal 2），50 条随机射线（eye(8,8,8)，pixel 方向 NDC uv → 反投影 → ray dir），对 `(hit_t, pal)` 断言 50/50 相同；abs diff t < 0.1 fine
  - 单测 `cpu_dda_ascii_grid_32`（AC-4）：32x32 像素网格渲染，输出 ASCII 画（pal1='X', pal2='o', pal3/4='#', empty='.'），`--nocapture` 输出固定内容（有 assert_eq! 确定性字符网格，比如 box 中心应有一串 X 16 连）
- **Acceptance Criteria Addressed**: AC-2, AC-4, FR-8, NFR-3
- **Test Requirements**:
  - `rule` TR-3.1: `dda_reference_equivalence_50_rays` pass（50/50 射线一致）
  - `rule` TR-3.2: `cpu_dda_ascii_grid_32` pass（ASCII 网格内容 == 基线字符串）

## Task 4: `dda.wgsl` 着色器文件（§3 五步寻址链 DDA + palette 颜色）
- **Status**: `pending`
- **Priority**: `high`
- **Depends On**: Task 2（常量 WGSL 侧=Rust 镜像）
- **Description**:
  - 新建 `gate-app/assets/shaders/dda.wgsl`。顶部大段注释：映射 §3 五步寻址链 view.rs 行号；常量区每常量注释 `// Rust wire.rs L?` 引用
  - Bind groups：`@group(0) @binding(0) var out_tex: texture_storage_2d<rgba8unorm, write>`（写颜色）+ `@group(0) @binding(1) var<uniform> u_view: DdaViewUniform`（含 inv_view_proj, cam_pos_fine, pad）。`@group(1)` 五读：`@binding(0) var<storage, read> b_struct: array<u32>;`（index/bitmap/dir/node 一体）+ `@binding(1) var<storage, read> b_leaves: array<u32>;`（brick slabs）+ `@binding(2) var<storage, read> palette_entries: array<PaletteEntry>;` + `@binding(3) var<uniform> g_globals: BrickMapGlobals`（globals scalar 字段）。**comp/state buffer 本期不用**，BG1 不用绑 comp/state（FR-4 sample 只用 struct/leaves/palette/globals）
  - `struct PaletteEntry { color: vec4<f32>, roughness: f32, emissive: vec3<f32>, transmission: f32, flags: u32 };` 40B，必须与 Rust PaletteEntry repr(C) 字节一致（Rust 侧已有 test 断言 8B？哦原 palette 定义：PaletteEntry { color, roughness, emissive, transmission, flags } 单条 8B？！之前 palette 8B/条：`docs/brickmap.md §2` 写的是 "Palette: 256 × PaletteEntry { color, roughness, emissive, transmission, flags } (u8 索引 + flags 视觉变体；槽 0 = AIR 保留)"。这里有不一致！需要先看 palette.rs 的 repr(C) 大小和 PaletteEntry 字段类型）：先在 Task 4 前核对实际 PaletteEntry 定义（属于 Task 4 范围），若 palette_entries 中 color 是 vec4<u8> 或 glam::Vec4：调整 WGSL struct 到匹配。**关键要求：P2.4 首版先只用 storage buffer<u32> 读 palette buffer 前 256 字作为 packed u32，颜色用最低 3 字节 BGR，实现一个 palette_to_rgb(packed: u32) -> vec3<f32> 函数**，保证不被 PaletteEntry 对齐 drift block。这是最保险的临时方案，P3 再严格 repr(C) 对齐。Task 4 先写此 path
  - DDA compute：`@workgroup_size(8,8)`。像素 gid 反投影：NDC xy = (2*gid.x + 1) / size_x - 1, 1 - (2*gid.y + 1) / size_y。ray ndc_near = (ndc, 0, 1), ndc_far = (ndc, 1, 1)，`inv_view_proj * ndc_near` → /w → world near point；同理 far；dir = normalize(far - near) * 4.0（fine 单位）；origin = u_view.cam_pos_fine.xyz（已 fine）。A&W 循环：t = 0.0，tMax init = grid crossing，tDelta = 1/abs(dir)，sign = dir sign。每步走 min 分量，sample_brickmap(vec3<i32>(floor(origin + dir * t))) → palette；非 0 break，写 color = palette_to_rgb(packed_palette(pal))
  - 空路径：t > tFar（tFar = 200.0 fine，约 50m 视距）或 steps > 2048，写 background vec3(0.05, 0.08, 0.12)
  - `sample_brickmap(fine: vec3<i32>) -> u32`：完全独立实现五步寻址链，与 view.rs 对应（常量对齐，WGSL const 与 Rust mirror 一致）。注意 i32 除法舍入（tile = fine / 32，fine signed）、TILE_INDEX_CAP 线性寻址、位偏移 >> % 32、hdr uniform 快路径（bits 0-7 non-zero = palette and uniform, non-zero return hdr & 0xFF）
  - `slot_unpack(word: u32, half: u32) -> u32`：半 0 = lower 16bit (word & 0xFFFF)，半 1 = upper 16bit ((word >> 16) & 0xFFFF)；tag = slot >> 8，palette = slot & 0xFF
  - 注释逐段对应 view.rs：步骤 ① TileIndex (view.rs L59-63) → ② TileBitmaps (L65-70) → ③ CellDirs (L71-75) → ④ walk_node hdr+l1+l2+l3+brick (L80-147)
- **Acceptance Criteria Addressed**: FR-2, FR-4, FR-5, FR-6, FR-7, NFR-5, AC-7
- **Test Requirements**:
  - `rule` TR-4.1: WGSL 编译：pipeline_cache queue 成功，首帧不 panic（着色器语法无错，entry_point 找到）
  - `rubric` TR-4.2: 着色器可维护性；维度 = 注释密度+Rust 映射可追踪；scale 1-5；1=无注释无法对齐，3=每步有注释，5=逐段注释 view.rs 行号+常量引用；threshold >= 4；evidence = dda.wgsl source inspection

## Task 5: `BrickMapDdaPlugin`（Rust 插件，BG 构建 + dispatch + blit）
- **Status**: `pending`
- **Priority**: `high`
- **Depends On**: Task 1, Task 4
- **Description**:
  - 新建 `gate-render/src/brickmap/dda.rs` `BrickMapDdaPlugin` struct + impl Plugin。build(app)：
    - ExtractResourcePlugin::<DdaImages>::default()（主 world → render world）
    - 主 world `DdaCameraConfig` → render world：**不用 ExtractResource**（static 不变），在 render_app `ExtractSchedule` 放一个 system `extract_camera_config(cfg: Option<Extract<Res<DdaCameraConfig>>>, mut commands)` 拿到后转换成 `DdaViewUniform` 资源插入 render world（每帧覆盖一次，static 就 O(1)）
    - render_app 链：
      - RenderStartup: init_dda_pipeline（queue_compute DDA WGSL + queue_render blit；DDA blit 复用现有 blit.wgsl！**直接复用 gradient 的 blit pipeline，因为 texture blit 格式相同**。唯一不同是 blit source texture。所以 init_dda_pipeline 实际新建 DDA compute pipeline + DDA blit layout（和 blit 一样的 texture 2d single），但 queue blit 重新用同一 blit.wgsl entry point OK）
      - PrepareBindGroups prepare_dda_bind_groups：创建 BG0（storage_tex write + DdaViewUniform UniformBuffer write_buffer）+ BG1（GpuBrickMap.b_struct &binding(0), b_leaves &binding(1), palette &binding(2), globals UniformBuffer &binding(3)）。palette buffer = GpuBrickMap.palette Buffer（之前是 buffer<u32> 8 字节 per palette entry * 256 = 2048B，用上面 TR-4 的 packed 函数读即可）
      - RenderGraph：dispatch_dda.before(camera_driver)（dispatch_gradient 前或后都行，后写覆盖 out tex），workgroup count = VIEW_SIZE/8, VIEW_SIZE/8, 1
      - Core2d::PostProcess blit_dda_view：和 gradient 的 blit_view 结构相同，blit 采样 DdaImages target。核心时序 `target.get_color_attachment()` 与渐变一致
      - **blit 优先级**：DDA blit 在 gradient blit 后运行，覆盖 view target（因为 Core2d PostProcess set 中 system 顺序按 add_systems 顺序：先 GradientPlugin blit，再 Dda blit → DDA 赢）。实现：dda.rs 里 blit_dda_view 写一个独立函数，add_systems(Core2d, blit_dda_view.in_set(Core2dSystems::PostProcess)) 注册到同一 set，Bevy 默认为注册顺序 chain，后者在后者
- **Acceptance Criteria Addressed**: FR-1, FR-9, NFR-4, AC-5
- **Test Requirements**:
  - `rule` TR-5.1: `cargo check --workspace` 通过（无类型错误）
  - `rule` TR-5.2: `cargo build -p gate-app` 成功（无 link error）
  - `rule` TR-5.3: Pipeline layout 验证：首帧日志无 "invalid bind group layout" / "mismatch between shader and pipeline layout" validation error（wgpu=warn 日志）

## Task 6: 整合验证（final run - 画面特征 + FPS + 无 panic）
- **Status**: `pending`
- **Priority**: `high`
- **Depends On**: Task 1-5
- **Description**:
  - `cargo run -p gate-app` 后台 50s，RUST_LOG=gate=info,wgpu=warn；日志 run_p24_accept.log
  - 验证 AC-1：画面包含 box / sphere / two hotspots（本任务无法 screenshot，日志侧验证 PROBE + UPLOAD[incremental] 正常 + dispatch_dda / blit_dda 在运行；关键——**新增 FPS 基线记录**：GTX 1660 无法现场测，测 RTX 3070 baseline >= 80 FPS（阈值 AC-6 rubric >= 4）
  - 验证 AC-8：VUID 只有 wgpu#9213 2 条
  - 验证 NFR-6：无 panic，frames >= 500
- **Acceptance Criteria Addressed**: AC-1, AC-6, AC-8, NFR-1, NFR-2, NFR-6
- **Test Requirements**:
  - `rule` TR-6.1: 日志中 `frame_count >= 500`（LogDiagnosticsPlugin 最后一帧计数）
  - `rule` TR-6.2: 无 `panic|Encountered a panic` 字样，exit 0
  - `rule` TR-6.3: VUID 只出现 2 条且 ID 匹配（`VUID-VkPresentInfoKHR-pImageIndices-01430` + `VUID-vkAcquireNextImageKHR-semaphore-01286`）
  - `rubric` TR-6.4: RTX 3070 FPS 基线；dimension = 前 200 帧平均 fps；scale 1-5；1=<30, 3=50-60, 5=>=120；threshold >= 4（>=80）；evidence = bevy_diagnostic fps 行前 200 帧平均

## Issue Pool（空，留作 Review 失败回填）
