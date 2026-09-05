# Douglas Dwyer 体素引擎 Devlog 笔记索引

YouTube 频道 [@DouglasDwyer](https://www.youtube.com/@DouglasDwyer) 全部 38 个视频的字幕总结，用于对标 gate 的渲染/物理/玩法开发。

- 完整字幕清洗文本在 `transcripts/`（yt-dlp 下载 + `_strip_vtt.py` 去重去标）。
- `_batch_urls.txt` 为原始 URL 清单（按优先级分类）。
- ⭐ 数量 = 对 gate 的参考价值评级（⭐⭐⭐ 直接施工图 / ⭐⭐ 设计参照 / ⭐ 背景知识）。

## 一、基础渲染（9 篇）

| 文件 | 视频 | 评级 | 核心 |
|---|---|---|---|
| [01](01_devlog01_sparse_voxel_octrees.md) SVO 与 ray caster | https://youtu.be/zbmYzugnEC0 | ⭐⭐⭐ | 可变叶八叉树 + 分层 DDA，与 gate 架构同构 |
| [02](02_devlog02_textures_lighting_faster.md) 纹理/光照/提速 | https://youtu.be/PfaNQLs6E94 | ⭐ | 纹理图集、方向光阴影 |
| [03](03_devlog03_svo_modification_benchmark.md) SVO 修改与基准 | https://youtu.be/yjOLx4O634I | ⭐⭐⭐ | 体素编辑增量更新八叉树——gate 编辑路径直接参照 |
| [04](04_devlog04_parallax_ray_marching.md) parallax ray marching | https://youtu.be/h81I8hR56vQ | ⭐⭐ | 集显渲染百万体素的早期方案 |
| [07](07_devlog07_perfect_pipeline.md) 完整渲染管线 | https://youtu.be/IFUj53VwYvU | ⭐⭐ | 静态/动态两 pass、shadow map、驱动开销 |
| [08](08_devlog08_gpu_distance_fields.md) GPU 距离场 | https://youtu.be/REKcTBgkrsE | ⭐⭐⭐ | 距离场加速 DDA 空跳——gate max_steps 优化的进阶路线 |
| [16](16_devlog16_render_distance_lod.md) 渲染距离 ×3 | https://youtu.be/74M-IxtSVMg | ⭐⭐ | LOD 选择策略与远距裁剪 |
| [18](18_devlog18_dda_bitmask_beam.md) 提速一倍 | https://youtu.be/P2bGF6GPmfc | ⭐⭐⭐ | bitmask 跳步 + beam 优化，gate DDA hot loop 可直接借鉴 |
| [22](22_devlog22_rewrite_implicit_normals.md) 重写与隐式法线 | https://youtu.be/YTZBFz3Et40 | ⭐ | 光栅化重写、法线隐式存储 |

## 二、光影（4 篇）

| 文件 | 视频 | 评级 | 核心 |
|---|---|---|---|
| [15](15_devlog15_ambient_occlusion.md) 环境光遮蔽 | https://youtu.be/3WaLMBiezMU | ⭐⭐⭐ | 体素 AO 采样方案，gate M2 直光+阴影阶段的补充项 |
| [17](17_devlog17_ray_tracing_back.md) 光追回归 | https://youtu.be/aY4Zet_C9Zs | ⭐⭐ | compute shader 光追架构，与 gate 管线同型 |
| [19](19_devlog19_emissive_path_tracing.md) 自发光与路径追踪 | https://youtu.be/VPetAcm1heI | ⭐⭐⭐ | 自发光体素 + 廉价间接光，gate StateTable 发光可视化的直接路线 |
| [23](23_devlog23_ddgi.md) DDGI 全局光照 | https://youtu.be/L1vhle74AEU | ⭐⭐⭐ | 探针放置算法逐条映射 gate cell 层级；P9 GI 施工图 |

## 三、物理（7 篇）

| 文件 | 视频 | 评级 | 核心 |
|---|---|---|---|
| [11](11_devlog11_sat_collision_oit.md) SAT 碰撞 | https://youtu.be/PW1Xwc3zzNc | ⭐⭐ | 分离轴测试做体素刚体碰撞 |
| [13](13_devlog13_physics_optimization.md) 物理优化 | https://youtu.be/b_d-0EyOuVg | ⭐⭐ | island 划分、睡眠机制 |
| [20](20_devlog20_rigid_body_physics.md) 刚体物理 | https://youtu.be/byP6cA71Cgw | ⭐⭐ | 完整刚体管线设计 |
| [25](25_devlog25_prototyping_2d_procgen.md) 2D 原型与程序生成 | https://youtu.be/pY7Y2pSCnGo | ⭐ | 物理原型方法论 |
| [26](26_devlog26_physics_fixes_tgs.md) 物理修复全集 | https://youtu.be/R9bror0oqR0 | ⭐⭐⭐ | 抖动/穿透/睡眠全部坑与修法——gate 自研物理避坑清单 |
| [27](27_devlog27_micropool_lockfree.md) micropool 无锁线程池 | https://youtu.be/QFQkqFSg8Z4 | ⭐⭐ | 游戏要延迟优先调度；Rayon 三宗罪 + 根任务隔离带 |
| [28](28_devlog28_fracture.md) 体素碎裂 | https://youtu.be/lsTHpbEN0dE | ⭐⭐ | 破坏效果与体素分解 |

## 四、玩法（4 篇）

| 文件 | 视频 | 评级 | 核心 |
|---|---|---|---|
| [12](12_devlog12_tree_ccl.md) 砍树（CCL） | https://youtu.be/5e8ut4NgF-8 | ⭐⭐⭐ | 连通分量标记做结构破坏——与 gate flood-fill 元件识别同算法 |
| [10](10_devlog10_snow_model_imports.md) 雪与模型导入 | https://youtu.be/XNtdYQLiGbA | ⭐⭐ | 多分辨率模型导入与降采样 |
| [14](14_devlog14_terrain_interval_arithmetic.md) 区间算术地形 | https://youtu.be/m4-toCACxKU | ⭐⭐⭐ | GPU 地形管线 + 均质区间跳过，与 gate 空区域 bitmap 跳过同思想 |
| [29](29_devlog29_character_controller.md) 角色控制器 | https://youtu.be/HlCVvG7_HFY | ⭐⭐ | 自动跳阶、贴墙滑行、坡度处理 |

## 五、其他（9 篇）

| 文件 | 视频 | 评级 | 核心 |
|---|---|---|---|
| [05](05_devlog05_webrtc_multiplayer.md) WebRTC 多人 | https://youtu.be/bQXM0QSlsqs | ⭐ | 内部服务器架构；体素脏区同步 |
| [06](06_devlog06_rust_engine_design.md) Rust 引擎设计 | https://youtu.be/TRVrjSD6zgA | ⭐⭐ | ECS+事件混合架构、OctreeBuilder trait 零成本抽象 |
| [09](09_devlog09_geese_dependencies_lod.md) geese 2.0 与 LOD | https://youtu.be/YQ83XfZQHHA | ⭐ | 依赖 DAG 事件系统、客户端预测 |
| [21](21_devlog21_wasm_modding.md) wasm 模组系统 | https://youtu.be/fvxOI0nQsTA | ⭐ | 宿主 trait 导出、沙箱安全 |
| [24](24_devlog24_ui_egui_csharp.md) UI 选型教训 | https://youtu.be/LLKCnN5a_FY | ⭐ | ImGui 三宗罪 vs egui 闭包式 API——gate-ui 设计佐证 |
| [27](#三物理7-篇) Rayon → micropool | （见物理） | | |
| [30](30_extra_pong_geese_tutorial.md) PONG 教程（番外） | https://youtu.be/zqNTbttpmaY | ⭐ | geese 用法习语：Store vs System 二分 |
| [31](31_extra_compiler_ub_bug.md) 编译器优化 UB（番外） | https://youtu.be/hBjQ3HqCfxs | ⭐ | enum 构造 UB 消掉范围检查——先校验后构造 |
| [32](32_extra_trying_voxel_engines.md) 竞品调研（番外） | https://youtu.be/uQgoJYtuCYs | ⭐⭐ | 建造 UX：网格吸附/undo/光标投影——P4 编辑器需求清单 |

## 六、Shorts（5 则，合并 1 篇）

| 文件 | 内容 |
|---|---|
| [33](33_shorts_weekly_updates.md) | 线性物理 / 渲染优化包 / GPU 显存分配器 / SIMD 网格生成 / 无限世界 chunk 流式加载 |

## 缺口

- `FUx7nRS6mvw`（P3 分类）：无英文字幕可下载 + YouTube 反爬拦截无法取元数据，待手动确认内容后补写。
- `transcripts/` 目录较大，commit 时一并入库（纯文本，便于跨机查阅）。

## gate 落地速查（按 TODO 里程碑）

- **M2 光影**：#15 AO、#19 自发光/路径追踪（StateTable 发光）
- **M3+ 渲染性能**：#18 bitmask/beam、#08 距离场、#16 LOD
- **P9 GI**：#23 DDGI（直接施工图）
- **P4 编辑器**：#32 建造 UX 清单、#12 CCL（元件 flood-fill）、#03 编辑增量更新
- **P6+ 物理**：#26 修复清单、#11/#20 刚体、#28 碎裂
- **模拟层**：#30 Store/System 二分、#27 线程池纪律
