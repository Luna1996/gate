# GATE

> 稀疏体素砖块图（NanoVDB-style linear brick map）+ 计算着色器 DDA 光追的体素游戏引擎。
> **base 分支：干净直光 + 隐式法线 + 太阳硬阴影**（无 GI）。所有体素光照/GI 实验从这里拉新分支。

---

## 0. 快速开始

```powershell
cargo run -p gate-app                    # 启动渲染 Demo
cargo test --workspace                   # workspace 全测
cargo clippy --workspace --all-targets  # lint
```

操作：左键拖拽旋转 / 中键平移 / 滚轮缩放。`GATE_BENCH=1` 开启逐帧 GPU pass 日志。

---

## 1. 技术选型

### 语言 & 框架

| 层 | 选择 |
|---|---|
| **语言** | Rust + WGSL |
| **渲染框架** | Bevy 0.19（ECS 调度 + extract/prepare 跨 world 镜像 + render graph 手工编排） |
| **GPU API** | wgpu 29（Vulkan 后端） |
| **资源格式** | RON 配置（光照主题、UI 主题） |

### 体素数据结构

| 选择 | 规格 |
|---|---|
| **稀疏砖块图（Brick Map）** | 线性化 SVO：tile→cell→node 三级寻址，空区域零存储，纯数组 GPU 友好 |
| **Brick Tree** | 256³ chunk / 4³ 分裂 / 4 层（256→64→16→4→1），每层 mask 一次 load 进寄存器 |
| **Palette 压缩** | 256 色 × 2 u32 words（color + roughness + emissive 打包） |
| **统一 GridDesc** | 144B entry 描述主世界 + 物体 OBJ 的变换/树基址/调色板基址/AABB/chunk 窗口 |

### 光照管线

| 组件 | 实现 |
|---|---|
| **法线** | 逐体素隐式差分（6 邻域 occupancy，一体素一法线） |
| **方向光硬阴影** | 命中点向太阳投 1 条 DDA 射线 |
| **Sky 渐变环境光** | 按法线 y 混合 horizon/top，smoothstep 在 h=0.35 |
| **太阳盘光晕** | `pow(max(dot(dir,sun),0), 64) * 0.05` |
| **Emissive 直出** | palette emissive 通道 × `EMISSIVE_EMIT_GAIN=4.0` |

---

## 2. 渲染管线

### Pass 流程图

```
Bevy Update Schedule（main world）
├─ OrbitCameraInput → OrbitCamera → DdaCameraConfig
├─ LightingTheme（RON）→ ExtractResource
├─ DirtyTracker → UploadBudget → MainPending
└─ poll_pending → 跨 world BuilderMirror

[ExtractSchedule]
└─ extract_brickmap / extract_light_pool

[RenderSchedule]
│
├─ Prepare
│  ├─ prepare_upload      BrickMapBuffers（3 storage + globals uniform）
│  ├─ prepare_dda_bind_groups  BG0/BG1/BG2/BG3 + blit BG
│  └─ prepare_light_pool   LightPoolUniform（464B）
│
├─ RenderGraph
│  │
│  ├─ beam_main ──────────────────────┐
│  │  workgroup 8×8 × BEAM_DIV=4       │
│  │  trace_scene → 低分辨率最近 t     │
│  │                                    │
│  ├─ dda_main ◀────────────────────────┘
│  │  workgroup 8×8 × half-res
│  │  1. NDC 反投影 → ray origin/dir
│  │  2. trace_scene（主世界 + 逐物体 GridDesc）
│  │     chunk 间 AABB slab → chunk 内 4 层 mask DDA
│  │  3. 着色：
│  │     sky×0.6 + ambient×0.4 + sun×N·L×shadow_DDA + emissive×4.0
│  │     ACES → sRGB → textureStore(rgba8unorm)
│  │  miss: sky_color(d) 直接写
│  │
│  └─ blit.wgsl
│     全分辨率 bilinear upsample → ViewTarget
│     sRGB cancel encode（linear → srgb）
```

### BG 布局

| BG | binding | 类型 | 内容 |
|---|---|---|---|
| **BG0** | 0 | storage tex | out rgba8unorm（dda_main 写） |
| | 1 | uniform | DdaViewUniform（144B） |
| | 2 | storage tex | beam_depth r32float（低分辨率最近 t） |
| **BG1** | 0 | storage | b_struct：TileIndex + TileBitmaps + CellDirs + NodeStream |
| | 1 | storage | b_leaves：方向可达掩码 LUT（4KB） |
| | 2 | storage | b_palette：256 色 × 2 words |
| | 3 | uniform | BrickMapGlobals（80B） |
| **BG2** | 0 | storage | grid_descs：主世界 + 物体 GridDesc 数组（144B/entry） |
| **BG3** | 0 | uniform | LightPool（464B） |

### DDA 核心（trace_chunk）

```
chunk 内 4 层 DFS mask 遍历
  ├─ traverse: level 下钻
  │   split bit=1 → popcount 定位 child → 载入下一层
  │   uniform mask=0 → palette 直接返回
  ├─ 整砖 LUT 跳过（b_leaves[oct*64+entry]）
  │   reach & mask == 0 → 升一层跨步
  └─ brick 内 DDA（side distance 预乘 step_inc）
      v 沿最近轴整子块步进 → face 更新 → 循环
```

关键优化：DIR_LUT 方向位掩码查 1 次替代热循环内 6 次 select；firstTrailingBit 整砖跨步一次跳到可行最粗 level；depth_cap 保守投影保证 beam 预 pass 覆盖。

---

## 3. 模块架构

```
gate-voxel/              纯体素核心
  ├─ coords.rs           层级常量：TILE_SUB=512 voxel, SUB_PER_CELL=16, 4 层分裂
  ├─ volume.rs           VolumeGrid（HashMap<TileCoord, ChunkTree>）
  ├─ chunk_tree.rs       BrickState + 树遍历 + uniform/query
  ├─ dirty.rs            DirtyTracker：data/comp 双队列
  └─ palette.rs          PaletteEntry（color + roughness + emissive）

gate-render/             CPU 渲染侧 + WGSL 参考镜像
  ├─ brickmap/
  │  ├─ wire.rs          BrickMapGlobals（repr(C)+ShaderType 对齐 WGSL）
  │  ├─ view.rs          五步寻址链 + DdaCameraConfig
  │  ├─ builder.rs       build_full / update_tile / take_dirty_ranges
  │  ├─ upload.rs        poll_pending / extract / ensure_with_copy
  │  └─ dda.rs           管线编排 + dispatch_dda + OrbitCamera + 全套 CPU 参考 DDA
  ├─ lighting.rs         LightPoolUniform + 数据驱动主题 + CPU 参考着色
  └─ responsive.rs       RenderScale + resize

gate-ui/                 bevy_ui 组件库
  ├─ widgets/            Panel / Label / Button / Slider / Checkbox / Plot / …
  ├─ theme.rs            暗色实验室主题
  ├─ world_anchor.rs     3D 世界标签贴附
  └─ capture.rs          截图辅助

gate-app/                Demo 入口
  ├─ src/main.rs         Startup / Update 系统
  ├─ src/diagnostics.rs  GATE_BENCH=1 逐帧 GPU 日志
  ├─ src/vox_scene.rs    场景构造 helpers
  └─ assets/
      ├─ shaders/        dda.wgsl（~1000 行）+ blit.wgsl
      ├─ lighting/       day_outdoor.ron / dark_lab.ron
      └─ ui/theme.ron
```

---

## 4. 常量 & 单位

所有世界坐标以 **voxel**（0.25cm）为单位，直接喂 DDA，无需 shader 内换算。

| 常量 | 值 | 含义 |
|---|---|---|
| `TILE_SUB` | 512 | 每 tile = 128cm |
| `CHUNK_SIZE` | 256 | DDA chunk 步进 |
| `BRICK_FACTOR` | 4 | 每级分裂因子 |
| `MAX_LEVEL` | 4 | 256→64→16→4→1 |
| `SHADOW_DIR_T_MAX` | 8192.0 | 阴影射线 t_max |
| `BEAM_DIV` | 4 | 低分辨率预 pass 缩小 |
| `DDA_WORKGROUP_SIZE` | 8 | workgroup 8×8 |

---

## 5. 分支策略

```
base ─────────── 当前基线：直光 + normal + 硬阴影（无 GI）
                 所有 GI 实验从这里拉分支
wip/ddgi-v1 ─── DDGI 1:1 复刻存档
```

---

## 6. 测试 & CI

```powershell
scripts\ci.ps1     # fmt → clippy → test
```

CI 门禁：`cargo fmt --check` / `cargo clippy -D warnings` / `cargo test` / `cargo build -p gate-app`

---

## 7. 性能基线（RTX 3070，1080p，half-res compute）

| Pass | GPU 时间 |
|---|---|
| beam_main | ~0.02ms |
| dda_main | ~0.76ms |
| blit | ~0.1ms |
| **总计** | **~0.9ms** |

大陆 3 tile → 300 tile → FPS 不变（AABB slab 空射线跳过 99.8% 像素）。
