# R6 纹理与材质

> **新队列（2026-09-02 架构对齐后新增）**：对标 Douglas #22 triplanar PBR 纹理系统 + bake normal 管线。完整架构规格见 `docs/texture-pipeline.md`。架构决策见 README 决策表「体素 normal 存储策略」+「纹理路线」条目。
>
> **与其他队列的依赖关系**：
> - 数据层原语（T1 region fill/copy）→ 可并行（不依赖 R1-R5）
> - Upload/bake pipeline（T2-T3）→ 依赖现有脏 tile 上传通道（P3.3 脏队列已就绪）
> - Shader 集成（T3）→ 依赖 v5 single-pass shade_hit（**无 hashmap 依赖**——triplanar 纹理在 shade_hit 内乘 albedo，无中间 pass）
> - **关键路径位置**：M2 保底线要求 triplanar 纹理化材质（砖纹导线 / 金属 LED / 暗色实验室墙面）→ R6 在 M2 验收前必须完成

---

## 基础管线（voxel 数据层 + bake pipeline）

- [ ] **R6-1（N/A）** **Bake normal spike**：CPU Rust 代码验证邻域分析 normal 算法 → 输入 BrickMap（单 tile，16³ 体素）→ 输出每个 surface voxel 的精确 normal；**不依赖任何 GPU 代码**；验证方式：Rust 单测 + 手动打印 surface voxel normal 数组 + 对边界体素人工检查（+X 方向 AIR → normal.x ≈ -1.0）；**gate 天生对齐 #22：从来没存过 normal → 直接写 bake 算法，不需要重构**
- [ ] **R6-2（N/A）** **Baked attributes buffer 格式设计**：每 slot 12B（has_normal:u1 + normal:snorm10×4 + albedo:u8888 + roughness:u8 + metallic:u8）；与 BrickMap 1:1 映射（4096 slots × 12B = 48 KB/tile）；pow2 桶空闲链复用现有 BrickMap 驻留管理；**VRAM 预算**：驻留 512 tiles = 24 MB，驻留 4096 tiles = 192 MB，总预算 ≤ 220 MB（VRAM 2GB 剩余 ~1.7GB 给 DDGI + BrickMap）
- [ ] **R6-3（N/A）** **Dirty upload + bake 集成**：脏 tile BrickMap 上传 → CPU bake → baked_attributes buffer 上传；触发时机 = 现有脏队列（R1-4 comp_dirty）+ 每帧预算调度（每帧 N 个 tile，与 BrickMap 上传同 budget 窗口）；**fallback**：baked buffer 未就绪 tile → shader 用纯色块 Palette albedo 渲染（不崩，可渐进上线）
- [ ] **R6-4（N/A）** **Shader triplanar 采样集成**：v5 single-pass `shade_hit` 内部集成——命中后读 baked_attributes → triplanar_uv(world_pos, baked.normal) → sample_texture_atlas(mat.texture_id, uv) × mat.albedo = 最终 base_color（替换当前 Palette 纯色 `hit_mat.albedo`）；光照计算（硬阴影 + sky gradient + emissive）**全部不变**，triplanar 纹理乘在 albedo 层面、在 shade_hit 最顶端；**DDGI 间接光**（R3-10）后续在此基础上叠加 `max(直光, 间接光)` 再算
- [ ] **R6-5（N/A）** **Texture atlas + PBR 材质资产化**：材质 = 数据驱动资产（与 Palette 分离 → PaletteEntry.texture_id 字段）；首发材质集：砖纹（导线用）、金属（LED 外壳）、暗色墙面、PCB 绿基板、玻璃（transmission，R5-3 后置）；**不做 grass/leaves**（解谜游戏关卡是电路面板 + 暗色实验室）

## 编辑原语（CSG / fill / copy — 数据层，成本低）

> gate 天生不需要 normal 维护成本 → 这些原语对 gate 比 Douglas 简单得多（他写了 5000 行才搞定，gate 可以直接复用 R1-1 CCL BrickMap 扫描原语）

- [ ] **R6-6（N/A）** **Region fill 原语**：`TileGrid.fill_region(bounds: AABB, palette: u8, mask: Option<FillMask>)` → 内部 = 逐 tile BrickMap 扫描 + 直接改 palette；**自动 dirty 所有受影响 tile**；不触发任何 normal 维护（upload 时自然 bake）；依赖 R1-1 BrickMap 扫描原语（已存在）
- [ ] **R6-7（N/A）** **Region copy 原语**：`TileGrid.copy_region(src: AABB, dst: Vec3, palette_filter: Option<impl Fn(u8)->bool>)` → 同样 = 逐 tile 扫描 + 直接改 palette + dirty 目标 tile；**比 Douglas 容易**（他还要维护目标 tile 邻居的 normal → gate 天生不用）
- [ ] **R6-8（N/A）** **CSG 基础**：sphere/cylinder/torus 3D 栅格化 + union/subtract 操作（光栅化算法 → 直接改 palette bitmap + 自动 dirty）；**spike 目标**：`csg::draw_sphere(center, radius, palette)` 在 100 毫秒内栅格化 4cm sphere
- [ ] **R6-9（N/A）** **Fill/copy 上传集成验证**：fill 一个 region → 哪些 tile 脏 → 下帧 upload + bake 正确 → triplanar 纹理 + bake normal 立即正确显示；**验收**：1 秒内 fill 10³ 体素 region → 画面同帧更新（和 edit 单体素同延迟）

## 质量与验收

- [ ] **R6-10（N/A）** **性能预算断言**：CPU bake 每 tile < 0.1ms（16³ 体素扫描 + 4096 次邻域分析）；VRAM baked buffer ≤ 驻留预算；fill_region 10³ 体素 → 触发 ≤ 8 tile dirty（不超过 BrickMap 脏队列预算）
- [ ] **R6-11（N/A）** **视觉验收**：砖纹导线 / 金属 LED / 暗色实验室墙面 triplanar 纹理正确（边界体素 normal 正确 → 纹理方向正确、无硬切割）；**与 DDGI hybrid 叠加验收**（R3-10 完成后）：通电 LED 珠 triplanar 金属纹理 + emissive radiance 照亮周围墙 + DDGI 间接光缝隙阴影 = M2 核心画面
- [ ] **R6-12（N/A）** **Fallback 验证**：关 bake buffer（强制 fallback）→ 画面退化为纯色块 Palette albedo 渲染 → 光照正确（硬阴影 + sky gradient + emissive 不受 triplanar 影响）→ 不崩、不报错

---

**后续扩展（不在 R6 必须）**：GPU bake pass（从 CPU 迁到 compute shader）、LOD bake 加速（R1-12 粗叶 tile bake buffer 更小）、absorptive transparency（彩色玻璃，R5-3 后置）

**与 Douglas #22 的关系**：架构决策完全对齐（隐式 normal → upload bake → triplanar），但 gate 天生没有 normal 维护成本 → R6 实现复杂度比 #22 低一个量级。不需要 grass/leaves decorations（解谜游戏关卡不需要）。
