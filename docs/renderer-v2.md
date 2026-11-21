# Renderer v2

目标：保留 Bevy（ECS / UI / 窗口 / 输入 / 资源 / 物理），只替换 `gate-render` 的渲染管线；使用 wgpu Vulkan 后端 + `dlss_wgpu` 接入 DLSS，FSR 2/3 作为 fallback。保持逐体素平色风格，动态场景/动态直光下光影、玻璃、镜面、折射稳定。

## 硬约束

- 逐体素着色：一个体素六个面共用同一 albedo/emissive；镜面同样按体素，一个体素一个反射值。
- 直光和场景动态：不做静态光照预计算、跨帧静态缓存、预烘焙 shadow map。
- DLSS 是硬需求：wgpu Vulkan 后端 + `bevyengine/dlss_wgpu` + NGX；FSR 2/3 作为非 NVIDIA/兼容 fallback。
- UI 1:1 保留：继续使用现有 Bevy UI、`assets/ui/debug_menu.toml`、locale、主题；只删除失效的 wgpu/旧管线调试项。
- 已实现功能完整保留：场景、MC、体素编辑、物理、流式、相机、GI/反射/玻璃、曝光、体积光、bench、profiler。

## 已完成的 DLSS Spike

- `gate-render` 增加可选 feature `dlss`，依赖 `dlss_wgpu 6.0.0`。
- Bevy 启用 `raw_vulkan_init`，通过 `RawVulkanInitSettings` 注册 DLSS 所需的 Vulkan instance/device 扩展。
- 运行时日志确认：
  - `DLSS SR supported: yes`
  - `DLSS SDK initialized`
- DLSS SDK：`DLSS_SDK` 指向 NVIDIA DLSS SDK v310.9.1；运行目录需要 `nvngx_dlss.dll`。
- 无 DLSS SDK / 非 NVIDIA 时保持 feature 关闭，渲染器照常构建。

## 迁移原则

保留并复用：

- `gate-voxel`：体素数据、调色板、chunk tree、编辑 AABB、`Volumes`。
- `gate-physics`：物理、接触、静态场。
- `gate-app/src/mc`：MC 地图读取、LOD、材质、区块 source。
- `gate-app` 纯逻辑：`infinite_cubes` 生成器、`scene` 构建、`config` 数据模型、`consts`、对象/编辑/相机数学。
- `assets/shaders/voxel_raytrace/*.wesl`：DDA、brickmap、GI、反射、玻璃算法；WESL → WGSL（现有运行时编译）。
- Bevy UI、debug menu、locale、theme、配置格式、profiler/Tracy。
- 现有 pass 名、环境变量、bench 行为。

替换：

- `gate-render` 的主 pass 结构：从“主 trace 内完成着色”改为 G-buffer + deferred lighting + 独立 GI/反射/降噪。
- 现有 GI/反射/阴影的逐像素路径逐步替换为低分辨率 + 时域累积 + reservoir/radiance cache。
- 旧 wgpu 专用调试项、失效的渲染档位删除。

## 管线

内部渲染分辨率 → 时域累积/降噪 → DLSS/FSR → 输出。

1. `FramePrepare`
   - jitter（Halton）、当前/上一帧 view-proj、对象当前/上一帧变换、光源、帧序号。
2. `PrimaryTrace`（compute）
   - 主射线 DDA，透明体素 delta chain（反射/折射/IOR/Beer-Lambert）。
   - 输出 G-buffer：depth/view-z、voxel key、face、obj、material、几何法线、motion vector、coverage、roughness/metallic。
3. `DirectLighting`
   - 太阳 NEE + 天空光；阴影射线独立 dispatch，便于半分辨率/时域复用。
4. `DiffuseGI`（1/2 或 1/4 分辨率）
   - 低 spp 路径追踪 + ReSTIR GI reservoir + radiance cache；SVGF/à-trous 降噪。
5. `Specular`
   - GGX VNDF 采样，反射/折射队列 + 时域降噪；镜面按体素。
6. `Denoise`
   - 法线/深度/体素 key 引导的联合双边 + 时域历史；阴影、diffuse、specular 分开。
7. `Upscale`
   - `DlssSuperResolution`（wgpu Vulkan）或 FSR 2/3；输入 color/depth/motion/jitter。
   - 无 DLSS 时走 FSR/原生。
8. `Post`
   - bloom、曝光、tonemap、dither、UI。

## 实施阶段

- M0：分支 + DLSS Spike（已完成）。
- M1（进行中）：G-buffer + 世界空间直光/阴影/AO 缓存 + 软太阳。第一版已落地，见下节。
- M2（进行中）：世界空间 voxel GI cache，替换屏幕空间 `gi_upsample_joint`；删除跨面混合。第一版已落地，见下节。
- M3：emissive light BVH + NEE；把自发光体素聚成 light brick，接 MIS。
- M4：jitter + motion + history + DLSS/FSR 内部分辨率。
- M5：反射/折射/彩色玻璃 delta chain 与时域降噪收敛。
- M6：denoiser、性能收敛、caustics/体积光补充。

## M1 已落地（2026-11-20）

- G-buffer 继续使用 `dda_vis`（命中、voxel key、face）与 `dda_nrm`；`dda_shadow_tex.rgb` = 太阳可见度，`.a` = AO。
- 新增 `assets/shaders/voxel_raytrace/light_cache.wesl`：4-way 世界空间哈希缓存，键为 `(obj, voxel, face)`；每项累积太阳可见度、AO 与样本数，满 64 样本后停止更新。
- 缓存版本包含物体/内容 revision、灯光 revision、量化后的太阳方向与角半径；编辑、物体移动、光照变化会局部失效。
- 太阳软阴影：在 `LightDesc.shape.y` 给定的太阳角半径内做 cone sampling，每个缓存更新 2 条阴影射线；角半径由天空盘角径写入（当前场景约 2.45°）。接触处保持锐利，远离遮挡物时半影自然变宽。
- AO：每个缓存更新 1 条 12-voxel 余弦半球短射线，乘到 ambient/GI 上。
- `shade_split` 与 `shade_face` 都不再调用 `gi_upsample_joint`；GI 只读世界空间 voxel cache。
- 过渡调色：`GI_SHADE_GAIN = 1.0`、`AMB_SHADE_GAIN = 1.0`、`SUN_SHADE_GAIN = 1.35`。M2 世界空间 GI 落地后按实际能量继续校准。

### 当前实测与缺口

- 1280×720、VSync off、当前 MC 城市视角、RTX 3070：约 105 fps（9.5 ms）；改动前同视角约 17 ms。
- 画面仍偏暗、GI 颜色反弹不足：因为屏幕空间 GI 被移除后，新的世界空间 GI cache 尚未实现。M2 完成前不把这个版本当作最终画质。
- 发光体素 NEE、面积光软阴影、彩色玻璃完整透射、caustics 尚未接入；当前只有太阳软阴影与 AO。
- 太阳自动移动时，量化后的缓存 revision 每约 0.45° 才失效一次；快速时变太阳下需要更细的失效策略。

## M2 已落地（2026-11-21）

- 新增 `assets/shaders/voxel_raytrace/gi_cache.wesl`：键为 `(obj, voxel)` 的世界空间 GI 缓存，复用 `light_cache` 的 8-way 哈希表与 `LC_EMIT` 三通道累加位。
- GI 以 **4×4×4 voxel cell** 为缓存与采样单位，着色时对 8 个相邻 cell 做三线性插值；这是为了在保留世界空间颜色反弹的同时消除逐 voxel 随机噪声。
- cell key 走独立哈希（`gi_mix(gi_key1 ^ lod ^ const)`），不复用 face id 位；避免 cell 起点恰好是可见 face 6 表面时与直光缓存行冲突，互相覆盖 `LC_SUN` / `LC_AO_N`。
- 每个可见 cell 每帧最多更新一次；每帧预算 262144 个 cell，近处最多 64 样本；满样本后只查表。
- 所有 cell 共用同一组半球方向（seed 只与帧号有关），邻域采样误差在空间上相关，避免逐 cell 独立随机方向产生云雾状斑块。
- 新可见 cell 首次打 4 条射线，之后 2 条，再降到 1 条；读值按 confidence 与基础天空 ambient 混合，约 2–4 帧收敛。
- 每次更新 1 条余弦半球射线；命中表面读其材质、太阳可见度和 ambient，乘透射 tint 后写入；天空方向用 `sky * GI_SKY_AMBIENT` 而不是全天空 radiance，避免远端过曝。
- **GI 射线命中表面的太阳可见度读取直光缓存，并在命中面的 3×3 邻域做加权平均**（中心权重 2，四邻域 1）；缺失时用同一 key 写回一组 cone 采样（n<4 时一次写 4 组），供后续 GI 射线复用。避免单 voxel 二值阴影/稀疏条目把噪声带进 GI。写回使用独立预算（`light_cache_meta[2]`），不挤占主直光缓存。
- `shade_split` / `shade_face` / `refl_hit_gi_m` 都改读新 GI cache；`gi_upsample_joint`、`face_slots`、WAL 的屏幕空间路径在 `WORLD_GI=true` 时整段停用（旧代码保留，可切回）。
- 新增 `dda_gi_filter`：对每个可见 GI cell 做 3×3×3 cell 邻域滤波，再与上一帧滤波值做 0.85 时域混合；每 cell 每帧只有一个 writer。
- 滤波输入是原始累加值 `LC_EMIT`，输出写 `LC_SUN`；`LC_SUN` 只供着色读取，不能回喂滤波，否则首帧零值会写进历史。
- 每次 GI 样本写入 `LC_EMIT` 前按当前均值做亮度钳制（4× 均值 + 0.02），抑制太阳反弹的离群样本。
- 缓存淘汰使用 `max(raw stamp, LC_ACCESS)`；滤波/读取会刷新 access，避免正在使用的 GI 项因为 raw stamp 旧而被新 voxel 驱逐，造成高频闪烁和滤波历史重置。
- 不再用固定比例混入平滑 ambient 补亮；GI 差额由直光缓存里的太阳反弹补上。
- GI 射线命中自发光体素时不再把 emission 当作随机命中光；白灯/霓虹只在自身表面发光，等 M3 的 light BVH + NEE 再参与照明。否则少量射线命中亮灯会被滤波抹成更大的光斑。
- `refl_hit_gi_m` 只接受 confidence ≥ 0.25 的成熟项，避免反射里的未收敛噪声。
- 背光面不再走“直接退 ambient”的临时分支；由 DLSS RR 统一降噪。
- 1200 voxel 以外不再维护逐 voxel GI，直接使用平滑的天空 ambient LOD；这解决俯视/大范围视野下 per-voxel 缓存覆盖不足导致的成片噪声。
- 远处 LOD（`scale > 1.01`）不写逐 voxel GI，避免不同 LOD 的 voxel key 冲突。
- 直光缓存每帧 claim 预算 262144，哈希表 8-way；新条目首帧写 2 组太阳/AO 样本，GI 命中写回在 n<4 时一次写 4 组。避免预算末尾的 voxel 永远拿不到缓存、只能每帧重新随机采样而闪烁。
- 缓存表大小从 `px/2` 提到 `4*px`（上限 4M 项，约 256 MB），8-way set 降低冲突驱逐概率。

### M2 实测与缺口

- 1280×720、VSync off、静止城市视角、RTX 3070：GI 收敛后约 195–210 fps；移动/新 cell 预热期会下降。
- 画面亮度与颜色反弹明显好于 M1；街道/建筑开始接收环境反弹；4×4×4 cell + 三线性 + 3×3 可见度平均 + RR 后，近景墙面的 GI 光斑与云雾状斑块已消除。
- GI 的空间分辨率现在是 4×4×4 voxel cell + 三线性插值，不是逐 voxel GI；这是消除随机散点的代价，后续可用 NRD/更多样本再提高。
- 仍有极少数 shadow penumbra voxel 在缓存重建的前几十帧内闪烁；新条目首帧多写样本、直光预算覆盖全屏后已大幅缩短。
- 水面反射仍有成片噪声、玻璃/反射 GI 粗糙、GI 只有一次弹射、没有 bloom/大气；这些是后续 M3/M5/M6 的工作。

## M2.1 屏幕空间 GI（NAADF 方向，已切换）

- 主漫反射 GI 从世界空间 4³ cell 切到屏幕空间 reservoir 路径：`dda.rs::WORLD_GI=false`、`main.wesl::GI_WORLD_CACHE=false`。
- `dda_shade` / `shade_face` 在 `!GI_WORLD_CACHE` 时把 primary hit 写入 `dda_hit`；`gi_main`（`gi_ss_main`）按 GI 分辨率重投影 primary，做 cosine 半球采样、lit/unlit 候选、时域历史、8 邻域 spatial reuse，然后 `gi_denoise_temporal` + 多轮 à-trous 写 `gi_dn_src[3]`。
- 着色端用 `gi_upsample_joint` 从最终 GI 纹理按面/法线/平面做联合上采样，不再读世界 cell。
- 当前配置：GI 1/2 分辨率、`denoise=high`、`wal=false`、`sun_bounce=true`；`wal` 关闭时 GI 只依赖屏幕空间样本，不再写世界累积。
- 已接入 NAADF 核心：屏幕样本存 21 word/像素（primary 位置/面/法线、secondary 位置/法线/距离、path hash、primary/secondary token、secondary key）；时域/8 邻域样本经 reconnection（cos/Jacobian 0.3–2.5 钳制）+ 可见性射线后进入加权 reservoir；`dda_gi_spatial` 在 `gi_main` 之后对当前帧 8 邻域再做一次 reservoir 迭代（8 次），输出写 `gi_out`；选中样本乘以 WRS 修正、π 能量和 8.0 钳制，再交给 temporal/à-trous 降噪和 `gi_upsample_joint`。
- 屏幕 GI 能量修正：`gi_face_shade_cached` 内部是 `brdf/π`，之前输出又按 `/π` 处理，导致 GI 暗 π 倍；现在输出乘回 π。
- 软阴影：`day_outdoor.ron` 太阳盘角半径 3° → 8°，`LC_SUN_SAMPLES` 2 → 4。
- primary/secondary token 使用 pose salt + `gi_local_rev`，移动物体/编辑/区域流式变化会失效对应样本；历史失效像素当帧候选提到 4。
- 当前是 per-pixel reservoir + 8 邻域空间迭代，不是 8×8 bucket ring；bucket/更多迭代留作后续。
- 实测 720p：静态墙面无云雾状斑块、无黑墙；画面仍偏蓝/偏平，GI 能量和太阳反弹色需要在结构稳定后单独校准。
- 未完成：8×8 bucket、12 次 spatial reservoir 迭代、移动物体 object-space、反射 GI 改读屏幕 GI。

## DLSS RR（2026-11-21）

- 本机 RTX 3070 / 驱动 610.88：`DLSS SR supported: yes / RR supported: yes`
- `nvngx_dlssd.dll` + `nvngx_dlss.dll` 已放入运行目录（不提交）
- `DlssRayReconstruction::new(DLAA, Packed, Linear)` 初始化成功
- 已接入主路径：`dda_gbuffer` 输出 diffuse/specular albedo、normal+roughness、linear depth、motion vectors；RR 在 shade/glass 之后、reflection compose 之前运行，输出 `rr_out`；compose 从 `rr_out` 取 HDR 颜色后 tonemap。
- 当前是 DLAA（同分辨率）；RR 后 `dda_refl_compose` 仍负责反射叠加和 ACES/dither。
- 实测 1280×720 DLAA：约 82 fps；无 RR 时约 135–150 fps。RR 成本明显，后续切 DLSS Quality 内部降分辨率后再评估。
- 已知限制：motion vector 目前是相机重投影，动态物体可能拖影；specular guide 复用主 motion；RR 每帧 `reset=false`，相机 cut 需要后续加重置。
- 构建：`DLSS_SDK=C:\repo\DLSS cargo build --profile build-fast -p gate-app --features dlss`；运行目录需要 `nvngx_dlssd.dll` / `nvngx_dlss.dll`。无 feature 时自动回退到非 RR 路径。

## 性能预算（1080p 输出）

- 目标 120 fps：8.3 ms。
- DLSS Quality（720p 内部）时：主 trace ≤4 ms，阴影/透明 ≤1.5 ms，GI ≤1.5 ms，反射 ≤1 ms，denoise/upscale ≤1 ms。
- FSR/DLSS 是显式的画质/性能选项；不把它当作偷偷降分辨率。

## 验收

- 同视角、同光照、同体素数据下对比截图：逐体素平色不变，阴影/玻璃/反射无明显噪声或时域拖影。
- 动态直光旋转、物体移动、玻璃折射、镜面反射四组场景逐项验证。
- DLSS/FSR 开关、质量档位、fallback 均可用；GPU pass 计时和显存占用记录到 `profiler`。
