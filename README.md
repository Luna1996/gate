# GATE — GPU-Accelerated Tile Engine

> GPU 稀疏体素砖块图（NanoVDB-style linear brick map）+ 计算着色器 DDA 光追渲染电路模拟器。
> **当前里程碑：渲染 Demo 完成（P0–P2）。** RTX 3070 1080p 稳定 60+ FPS；300-tile 极限大陆场景 1s 内完整可见。

---

## 0. 快速开始

```powershell
# 工具链：需要 MSVC Rust 1.82+（见 rust-toolchain.toml）+ Vulkan GPU 驱动
cargo run -p gate-app                    # 启动渲染 Demo（10×10 大陆 + 天空城 + 雪峰 + 水晶矿簇热点交换）
cargo clippy --workspace --all-targets    # lint
```

- 启动后操作：左键拖拽旋转视角 / 中键拖拽平移 / 滚轮缩放；`FPS` 文本 + 实时折线在左上角；`WorldAnchor` 3D 标签贴附关键地标（天空堡金顶 / 雪峰 / 水晶矿区 / 入口大道）。
- 每 ~2 秒会看到 tile(1,0,0) L4 热点 32³ 方块 **金 ↔ 青** 闪烁一次，对应 stderr 增量上传日志 `UPLOAD[incremental]: bytes≈0.13MB elapsed≈200µs`（演示增量 dirty-range 部分写优化生效）。

---

## 1. 项目现状（渲染 Demo Milestone）

### 已实现
| 模块 | 状态 | 说明 |
|---|---|---|
| **SVO 体素数据结构** | ✅ 生产可用 | `HashMap<TileCoord, Tile>` + 每胞 4 层精化下钻（L0 32cm → L4 2cm）；调色板压缩；CPU/Rust `BrickMapBuffers` 与 GPU storage buffer 字节同构。Tile 跨坐标自动创建。|
| **GPU 砖块图线性化（upload）** | ✅ 生产可用 | 节点流编码为 `b_struct` + `b_leaves` + `b_palette` 三块 storage buffer；`dirty-range tracking` 增量部分写：3-tile MVP 热点交换 140MB→132KB，278ms→200µs。|
| **DDA 计算着色器光追** | ✅ 生产可用 | `workgroup 8×8`；AABB slab 三轴射线-砖块图求交跳过空段（empty-ray 0 步），1 voxel 胞 Amanatides&Woo 步进；相对标尺 v3（slab 全局 t → start_v 推进 → 循环变量全相对 start_v），`max_steps=16384` 覆盖 9000 voxel 对角穿越，AABB-skip 与 brute-force full DDA 头 100% 等价。|
| **GPU 调度集成（Bevy 0.19）** | ✅ 生产可用 | no render nodes；extract/prepare/queue/render/blit 全部系统级显式调度；half-res RenderScale；compute 输出 texture 2D → Core2d PostProcess blit 到 ViewTarget（sRGB 精确匹配）。MSAA 强制关闭（自定义 multisample.count=1 冲突）。|
| **相机输入** | ✅ 生产可用 | OrbitCamera：左键旋转 / 中键平移 / 滚轮缩放；取消最远距离上限（原 8000 voxel → 现无上限）；pitch ±89° clamp。|
| **极简 UI（fps 文本 + 折线）** | ✅ 生产可用 | 自研 gate-ui 组件库（bevy_ui 原生）：Panel / Label / Button / Slider / Checkbox / Plot（折线）/ WorldAnchor（3D 文字）；主题令牌化（暗色实验室风）。字体：MapleMono-NF-CN-Regular.ttf，中文无豆腐块。|
| **基准极限场景** | ✅ 生产可用 | 10×10×3=300 tile 大陆：正弦高度场 + 4 角雪峰 + 蜿蜒 S 河 + 170 确定性散点树；中央天空之城（10 层倒锥浮空岛底 + 城墙 + 4 金顶角楼 + 正殿 + 高塔 + L2 金顶球 + 4 L4 红飘带）；入口大道 + 11 字 "GATE ENGINE"；3 处 L4 青紫红水晶矿簇；WorldAnchor 4 标签。|

### 已知待做（不阻塞 Demo）
- L4 水晶矿簇 DDA 命中代价高：建议合胞上抬 L3/L2；
- WorldAnchor 标签不做体素遮挡判断（现在永远在顶层）；
- `gate-app/src/main.rs` 中 `build_demo_scene` 函数体 ~540 行，应拆 `demo_scene.rs`；
- 正式 sim tick / StateTable emissive（当前仅 GPU StateTex 占位心跳值）；
- wgpu 29.0.4 Vulkan 首帧 VUID 报错（上游 #9213 / #9361），已 LogPlugin filter 静默，等补丁合入。

---

## 2. 技术路线总览

```
[Bevy 主 world (Main)]
   ├─ Startup: VoxelScene (TileGrid) + 构建 300 tile Demo scene
   ├─ Update:  OrbitCameraInput（鼠标）→ OrbitCamera → DdaCameraConfig → WorldAnchorCameraSync
   │           edit_tile_every_120_frames（每 2s L4 热点调色板交换 1 tile dirty）
   │           fps UI feed + Plot 刷新
   └─ Last: poll_pending（UploadBudget × dirty_tiles → MainPending.data/comp + force_full flag）
      ↓
[ExtractSchedule (跨 world 镜像)]
   └─ extract_brickmap：MainPending → BuilderMirror（render world resource）；need_full = first || pending_full
      ↓  build_full | update_tile(n dirty tiles) → BrickMapBuffers + DirtyRanges → UploadSnapshot
[RenderSchedule (render world)]
   ├─ PrepareResources: prepare_upload（CPU 镜像 → GPU 3 块 storage buffer + globals uniform）
   │                   prepare_dda_bind_groups（BG0 out_tex+view_uniform / BG1 struct+leaves+pal+globals / blit BG）
   ├─ Render:          dispatch_dda（compute 8×8, half-res 640×360; AABB-skip DDA）
   └─ Core2d PostProcess: blit_dda_view（linear → Rgba8UnormSrgb cancel sRGB encode + bilinear upscale → ViewTarget）
```

### 关键技术要点（为什么 60 FPS）
1. **AABB slab 空射线 0 步**：远镜头下 99.8% 像素在 AABB slab 判定阶段（18 次浮点除法）直接 miss，return 背景色。不进入 DDA 循环（0 GPU cycle）。
2. **DDA 命中即 break**：1 voxel³ 最细颗粒命中后立即出循环，平均 ~120 步/射线。
3. **half-res RenderScale（640×360 compute → 1280×720 blit）**：射线数 4× 减少，bilinear upsample 视觉无明显锯齿（远距大地形自然低通）。
4. **brickmap 体素量与 GPU 负载无关**：屏幕是固定 921,600 像素射线，大陆从 3 tile 扩大 100× 到 300 tile 后 fps 仍 60+（RTx 3070）。
5. **增量 dirty-range 上传**：每 2 秒热点调色板交换仅 0.13MB struct 局部写 + 180µs CPU；Startup 批量 211 tile dirty 按 backlog>3×预算 一次性清空不逐帧 2.8s 阻塞 Prepare。
6. **扩容不丢数据**：`ensure_with_copy` 在 storage buffer 扩容时 2× reserve + 全量写入，后续 partial write 再覆 dirty range，避免了"扩容后 buffer 归零只写 6MB → tile index 表全丢 → 全画面空"的经典 bug。

---

## 3. 模块架构（4 个 crate）

```
gate-voxel/        纯体素核心：TileCoord / TileGrid / SVO cell+octree+palette / DirtyTracker+state_table+comp_layer
                   - coords.rs    TILE_SUB=512 voxel, SUB_PER_CELL=16, 分层常量 VoxelPos
                   - tile.rs      Tile 体：cell_bitmap + SVO chain+palette
                   - grid.rs      TileGrid(HashMap)：set_voxel / clear_voxel / get_voxel
                   - dirty.rs     DirtyTracker：按 TileCoord 入 data/comp dirty 双队列；drain_budget(n)
                   - palette.rs   PaletteFlags + 256 调色板
                   - scene.rs     几何帮助：fill_box / fill_sphere / draw_text / fill_ball / in_terrain_h
                   - stress.rs    压测构造函数

gate-render/       渲染管线（CPU）侧：构建 BrickMapBuffers + 上传调度 + DDA 参考实现
                   - brickmap/
                       wire.rs     BrickMapGlobals(BrickMapView struct 布局, repr(C) + ShaderType 对齐 WGSL Globals struct)
                                   BrickMapBuffers{b_struct, b_leaves, b_palette, globals} （Rust/WGSL 字节同构）
                       view.rs     BrickMapView::get_voxel(IVec3 voxel) 5 步寻址链（WGSL 抄自这里）
                                   index_pos(origin, dims, tile_coord) = x + y*CAP + z*CAP² (C order, x stride 1)
                       builder.rs  BrickMapBuilder::build_full(grid) | update_tile(grid, TileCoord) | take_dirty_ranges()
                                   DIR_REGION / BITMAP_REGION / INDEX_REGION 固定前缀
                       dda.rs      cpu_reference_dda_ray()（full brute 2M step 参考）
                                   cpu_reference_dda_ray_aabb_skip()（AABB slab v3 + max_steps 参考）
                                   OrbitCamera / DdaCameraConfig（from_orbit → inv_view_proj + frustum_length）
                                   prepare_dda_bind_groups / dispatch_dda / blit_dda_view（Bevy 系统）
                       upload.rs   poll_pending / extract_brickmap / prepare_upload / init_empty_gpu + prepare_dda_bind_groups
                                   ensure_with_copy（扩容 2× reserve + 全量写）
                                   UPLOAD[full|incremental] 日志（tiles/bytes/elapsed）
                       mod.rs      pub re-export
                   - gradient.rs   基础渐变渲染背景（与 DDA out_tex 同 blit）
                   - responsive.rs RenderScale（half-res）+ bevy window resize 响应

gate-ui/           自研 bevy_ui 组件库 + 世界标签
                   - widgets/     Panel / Label / Button / Slider / Checkbox / Plot（折线） + 列表滚动
                   - theme.rs     主题令牌（颜色/圆角/间距/字体/CJK 字体 override）
                   - world_anchor.rs  WorldAnchorCameraSync：世界坐标→屏幕 clip-space；锚定 3D 文字标签
                   - capture.rs   测试截图辅助
                   - lib.rs       Plugin 安装

gate-app/          Demo 应用入口（只有代码）
                   - src/main.rs  Startup / Update / 系统注册 / 极限场景 build_demo_scene（10×10 大陆）

assets/            运行期资源（只读；与可写 logs//data/ 分离，见 gate-render/src/paths.rs）
                   - shaders/    voxel_raytrace/  WESL 包（体素光追 GPU 程序，DDA 抄自 Rust cpu_reference_dda_ray_aabb_skip v3 同构）
                                 main.wesl=入口点 + import；bindings/common/brickmap/trace/world/
                                 lightfield + ddgi/*（consts/helpers/bake/sort/sample/collect）
                                 blit.wgsl（fullscreen triangle bilinear upsample + sRGB cancel）
                   - ui/         theme.ron 暗色实验室主题配置；debug_menu.toml 菜单初始值
                                 （运行期改动写 <安装根>/data/，不回写这里）
                   - locales/    zh-CN.yml 文案表（编译期 codegen 进二进制，运行期不读）
                   - fonts/      MapleMono-NF-CN-Regular.ttf（CJK）+ fa-solid-900.ttf（图标）
                   - lighting/   day_outdoor.ron（光照主题）
                   - vox/        nuke.vox（默认场景，gitignore）

logs/  data/       运行期可写目录（gitignore）：logs/latest.log 日志落盘、data/ui/debug_menu.toml 菜单存档
dist/              打包产物（bash package.sh 生成，gitignore）
```

---

## 4. 常量体系（所有世界坐标 voxel 单位 = 2cm）

| 符号 | 值 | 含义 |
|---|---|---|
| `TILE_SUB` | 512 voxel | 每 tile 512 voxel = 10.24m |
| `SUB_PER_CELL` | 16 voxel | 每粗胞 16 voxel = 32cm（L0 底）|
| `TILE_CELLS` | 32³ = 32,768 | 每 tile 粗胞数 |
| `TILE_INDEX_CAP` | 128 | 砖块图 index 表每轴容量 |
| `PALETTE_WORDS` | 512 | 512 字 × 2 slot/字 packed = 1024 palette 项；实际 0=AIR |
| `BRICK_SLAB_WORDS` | 1024 | 最细砖 4KB（L4 16×16×16 packed）|
| `INDEX_WORDS` | 128³ = 2,097,152（≈8MB）| 定长前缀：tile → slot_idx 映射 |
| `BITMAP_REGION_WORDS` | `TILE_CAP * TILE_BITMAP_WORDS`（1024×128）= 每 tile 4KB 粗胞占用位 |
| `DIR_REGION_WORDS` | `TILE_CAP * CELL_DIR_WORDS`（32768×128）= 每 tile 128KB 胞节点绝对字偏移 |

**DDA 坐标系约定**：DDA 内所有坐标（ray origin/ray dir/cell floor/tmax）、`DdaCameraConfig::position_world`、`BrickMapGlobals::index_origin_* × 512` 全部 **voxel 单位**。perspective_rh + look_at_rh + inv_view_proj unproject 直接操作 voxel 坐标，`frustum_length = length(far_world - near_world)`；无需 tile↔voxel 在 shader 内再次换算。

---

## 5. 关键 bug 备忘（回归必看）

详见 `project_memory.md` `Key Bugs Closed` 列表，已归档 8 条：
1. **[CRITICAL] `ensure()` 扩容不拷内容** → `ensure_with_copy`；
2. **[CRITICAL] `poll_pending` 2.8s 阻塞 Prepare** → backlog 一次性刷新；
3. **[HIGH] fill_box pal=0** → `clear_voxel` 三重循环替代；
4. **[HIGH] max_steps=2048 < AABB 厚度** → 升到 16384 + headless 5 zoom 档锁等价；
5. **[HIGH] Update 内 debug_aabb_report 2B CPU DDA / 0.5s → fps 个位** → 注释出调度；
6. **[MED] OrbitCamera clamp DIST_MAX 限远** → 删除上限、CAM_FAR 65536；
7. **[MED] AABB DDA 标尺混用 v1/v2** → 相对标尺 v3（headless 300 射线 + demo scene 5 档 zoom 锁正确）；
8. **[LOW] wgpu 首帧 VUID 告警** → 上游未修，LogPlugin filter `wgpu_hal::vulkan::instance=off,surface=off` 静默。

---

## 6. 质量门禁（本地自测 before commit）

```powershell
# 本地跑法（VS Code → 终端 → 运行任务，或直接敲命令）
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo build --workspace
```

- **`cargo fmt --check`**：代码风格。
- **`cargo clippy --workspace --all-targets -- -D warnings`**：lint 零警告。
- **`cargo build -p gate-app`**：dev 构建 0 error（unused 警告允许）。

渲染 Demo 手工验收（跑 `cargo run -p gate-app` 10 秒）：
- 画面必须出现：大陆全览 + 4 雪峰 + S 河 + 中央天空堡 + L2 金顶球 + 4 红旗 + 3 处 L4 水晶；
- fps 折线 ≥ 50（RTX 3070 基线 73），无 < 30 波谷；
- 滚 20 下向外不缺体素；
- stderr 每 ~2 秒一行 `UPLOAD[incremental]: bytes≤0.3MB tiles=1`，**无 1 秒以上 elapsed 增量上传长耗时**。

---

## 7. 打包发布

```sh
bash package.sh     # release 构建 → 组装便携目录（不压缩）
# 产物：dist/gate-<版本>-win64/{gate-app.exe, assets/}
```

- **产物形态**：`gate-app.exe` 与 `assets/` **同级**，双击即用；整个目录放到任何位置都行。要发 zip/7z 自己压。
- **路径怎么找**（`gate-render/src/paths.rs`，`install_root()` 一套规则覆盖两种形态）：
  1. `GATE_ROOT` 环境变量 → 直接当安装根（自定义安装位置 / 测试）；
  2. exe 同目录存在 `assets/`（便携发布形态）→ 安装根 = exe 所在目录；
  3. 否则 = 源码树根（`cargo run` / F5 时 exe 在 `target/<profile>/`，走这条）。
  于是 `assets/`（只读）、`logs/`、`data/`（可写）在开发与发布下都指向同一套相对位置。
- **必须随包发**：`assets/` 全部内容（字体、`blit.wgsl`、WESL 源码——启动时读盘编译、`ui/theme.ron`、`ui/debug_menu.toml`、`lighting/*.ron`、`vox/nuke.vox`）。
- **无需随包**：`assets/locales/*.yml`（编译期 codegen 进 exe）、`logs/`、`data/`（首次启动自建）。
- **不写安装目录**：日志 → `<安装根>/logs/latest.log`；菜单状态 → `<安装根>/data/ui/debug_menu.toml`（启动时优先读它，没有才用 `assets/ui/debug_menu.toml` 初版）。安装到只读目录（如 Program Files）时用 `GATE_ROOT` 把可写数据挪到别处。
