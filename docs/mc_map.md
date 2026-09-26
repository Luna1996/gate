# Minecraft 地图流式加载（Greenfield 1.17.1）

把 MC Java 世界的**方块**当成我们的体素世界来流式加载。本文记口径、事实、实测与切片进度。

## 1. 体素口径：1 方块 = 16³ 体素

体素单位取 MC 的**材质分辨率**（16×16）⇒ 一个方块 = 16³ 体素（我们一个体素 = 1/16 方块）。
由此得到一条**精确对齐**（不是近似）：

```text
我们的 chunk 边长 = 256 体素 = 256/16 = 16 方块 = MC 的**一个 chunk-section**
我们的 chunk 坐标 = (chunkX, sectionY, chunkZ)          // floor(方块/16) == floor(体素/256)
```

⇒ M5/M6/M8 那套流式环（`ChunkSource` + 窗口跟相机 + 射线请求驱动 + LRU + 多 volume）**原样可用**：
`ChunkSource::produce` 只要做"我们的一块 = MC 的一个 section"这一个翻译，**无缩放、无偏移**。
一个 section = 5.12 m（1 体素 = 2 cm），与 `infinite_cubes` 的 chunk 同尺寸。

高度：1.17 的世界高 0..255 ⇒ 16 个 section = 我们的 16 层 chunk。

## 2. 来源事实（已核实）

| 项 | 值 | 怎么核实的 |
|---|---|---|
| 存档 | `C:\game\Greenfield v0.5.4\Greenfield v0.5.4` | `region/*.mca` 356 个，1.39 GB |
| 版本 | **1.17.1**（`DataVersion = 2730`、`Version.Name = "1.17.1"`） | `level.dat`（gzip）解压后读 |
| 出生点 | 方块 **(115, 71, −2)**（⇒ 体素 `(1840, 1136, −32)`） | `level.dat` 的 `SpawnX/Y/Z` |
| region 格式 | 4 KB 索引（扇区偏移 3 B + 扇区数 1 B）+ 每块 `len(4 B BE) + 压缩类型`；本图实测**类型 2 = zlib** | 真地图冒烟读通 |
| section 格式 | `Level.Sections[]`：`Y`(byte) + `Palette`(list) + `BlockStates`(long[])，**1.16+ 紧凑位打包** | 同上 |
| 空段 | 整层空气的 section **只留一个 `Y`**（实测 tags = `["Y"]`） | `real_map_spawn_chunk_reads` |

出生点区块（chunk `7,-1`）各层读数（`real_map_spawn_chunk_reads` 的输出）：

```text
section y=0  调色板  3 项  实体 4096/4096      section y=4  调色板 78 项  实体 1084/4096
section y=1  调色板  2 项  实体 4096/4096      section y=5  调色板 67 项  实体  824/4096
section y=2  调色板  2 项  实体 4096/4096      section y=6  调色板 30 项  实体  414/4096
section y=3  调色板 16 项  实体 3849/4096
```

⇒ 城市的大部分高度是空气（`produce` 对这类层直接返回 `None`），实体的那一层里
`light_gray_stained_glass` / `smooth_stone_slab` / `stone_brick_stairs` / `white_terracotta` 都在
—— **台阶与楼梯出现 ⇒ 形状表 / 真实模型这一刀是必需的**（不能全当整块）。

## 3. 资产（模型 / blockstate / 贴图）

存档**不含**贴图与形状；形状在客户端 jar 的 `blockstates/*.json` + `models/block/*.json`，
贴图在 `textures/block/*.png`（16×16）。两份来源按 MC 的规则叠加：**资源包优先、客户端 jar 兜底**。

已验证的两份来源：

- 客户端 jar `…\.minecraft\versions\Greenfield\Greenfield.jar`（1.17.1）：1710 models / 900 blockstates / 818 textures
- 资源包 `Greenfield.Texture.Pack.1.17.zip`：919 block 贴图 + 224 models + 108 blockstates（**覆盖**同名）

一次性解包（把 jar 打底、包覆盖，解到仓库外）：

```powershell
$out = "C:\game\Greenfield v0.5.4\assets_mc"
# 先解 jar 的 assets/minecraft/{blockstates,models,textures}（去掉前缀），再解包的同名覆盖
# 结果：902 blockstates / 1823 models / 1021 textures，15.4 MB
```

⇒ 运行时按普通目录读 `<assets>/{blockstates,models/block,textures/block}`，**不读 zip**。

两个目录各有环境变量覆盖，缺省是本机的这两份：`GATE_MC_MAP`（存档根）、`GATE_MC_ASSETS`（资产根）。

## 4. 代码结构与依赖

| 文件 | 职责 | 状态 |
|---|---|---|
| `mc/world.rs` | 存档读取（region 索引 / 解压 / NBT → 区块结构）＋ 区块 LRU | **已落地** |
| `mc/assets.rs` | blockstate / model JSON + 贴图 PNG 的读取与缓存 | **已落地** |
| `mc/model.rs` | blockstate → 元素盒（`variants` / `multipart` / `parent` / 贴图变量 / 缺省 uv） | **已落地** |
| `mc/voxel.rs` | 元素盒 → 16³ 逐 texel 色 → **8 cm 写入计划**；材质与染色启发式 | **已落地** |
| `mc/material.rs` | 逐 texel 颜色 → 调色板槽（内容去重 + 颜色量化 + 溢出回退） | **已落地** |
| `mc/source.rs` | `ChunkSource`：一块 = 一个 section；`Detail` 的四档落地 | **已落地** |
| `mc/mod.rs` | 世界注册（`scene::build_world` / 菜单 / `Streaming` 接线）+ 端到端冒烟 | **已落地** |

**格式解析全部走现成 crate**（不自造轮子）：

| 环节 | 谁做 |
|---|---|
| NBT 反序列化 | `fastnbt`（serde） |
| region 4 KB 索引、扇区载荷、压缩（zlib / gzip / lz4 / 裸） | `fastanvil::Region::read_chunk` |
| `BlockStates` 位打包展开（1.15 跨长 / 1.16 留白两种） | `fastanvil::expand_blockstates` |
| `level.dat`（gzip+NBT）的出生点 | `flate2` + `fastnbt` |
| 贴图 PNG（灰度 / 调色板 / 16 位统一成 RGBA8） | `png`（`Transformations::normalize_to_color8`） |
| blockstate / model 的 JSON | `serde_json` |

**为什么不用 `fastanvil::pre18::JavaChunk`**：它把方块压成 `fastanvil::Block`，而 `Block` 只公开
`name()` / `encoded_description()`，后者**有意丢掉 `waterlogged` 与 `powered` 两个属性**（服务于
fastanvil 自己的彩色渲染）。而 `variants` 的条件匹配要按属性找形状（台阶朝向、楼梯形状、门、红石灯）
⇒ 属性必须原样保留，所以用 `fastnbt` 自定一层薄结构（fastanvil 文档的推荐用法）。

`mc/model.rs` 与 `mc/voxel.rs` 没有同类 crate 可替：MC 的资源包模型格式（`variants` 条件 /
`parent` 继承 / 贴图变量 / `elements` + 显式 uv / 块级旋转）在 Rust 生态里没有成熟实现，
`nucleation` 那类工程是自成一体的（同时含网格化与 GPU 渲染），与本仓的"体素化 + 自己的 DDA"不通用。

## 5. 形状与颜色的规则（首版覆盖）

- `variants`：键 = `性质=值` 用 `,` 连（AND）、值可用 `|` 给候选；按方块的 `Properties` 匹配；
- `multipart`：`when` 可为字符串、`{性质: 值}`、或 `{"OR": [...]}`（数组元素是对象）；
- `parent` 继承链：**父在前、子覆盖**（子有 `elements` 就整段替换，`textures` 逐键覆盖）；
- 贴图变量 `#all` 这类按 `textures` 表逐跳解析（深度上限 8，防环）；
- 面的 uv：缺省按元素盒六个面推（与 MC `FaceBakery` 同一套公式），显式 uv 原样取；
- 块级旋转（`x`/`y`）：在**体素层面**整体旋转（90° 的倍数 ⇒ 格到格、无插值），顺序 = **先 x 后 y**
  （用 `oak_log` 的 `axis=x` = `x:90,y:90` 与 `stairs` 的 `facing=east` = 无旋转反推出来的）；
- 逐格上色：取**最近的面**的 texel；同时贴两个面时按 `up > down > west > east > north > south`；
  贴不到任何面 = 元素内部 ⇒ 取该元素各面贴图均色（内部不可见，均色还能让整砖合并）；
- texel 的 α = 0 ⇒ **不写**（挖空：树叶的洞、玻璃中央）；`0 < α < 255` 当半透（玻璃走 transmission）；
- `tintindex` 存在时按方块名取**固定生物群系色**（草 `#91BD59` / 叶 `#77AB2F` / 水 `#3F76E4`，
  云杉与白桦用各自的原版色）—— **不读 `Biomes`**；
- 材质参数（粗糙度 / 自发光 / 透射 / 金属度）按**方块名**启发式给（`glowstone`/`torch`/`*_lamp[lit=true]`
  自发光、`*glass`/`*_pane` 透射、`*_ore`/含 `iron|gold|copper` 的金属、其余按粗糙石材）；
- 缺 blockstate 或形状整段缺失（水、熔岩这类流体）⇒ 兜底"整块 + 同名贴图"（`block/<名>`，退一步 `_still`）。

**不覆盖**（首版明确的取舍）：元素自带的 `rotation`（植物的 45° 交叉面退化成轴对齐薄片）、`uvlock`、
`display`、`ambientocclusion`、面内 `rotation` 的 90/270 符号（按 180 的公式推的，可能是镜像）。

## 6. 实测：为什么"逐 texel"不可用（重要）

**结论：MC 的"逐 texel 上色"在现有树结构下不可用，最终取 8 cm 档（每 4³ 体素一个代表色）。**

`real_map_produce`（出生点 chunk，`Detail::Full`，7 层含 3 层满实体）：

| 颜色量化 | 稳态产出 | 树大小 | 备注 |
|---|---|---|---|
| 8 位（原样） | **78 ms/层** | **20 832 KB/层** | 逐 texel（每个 `4³` 砖都是值表） |
| 5 位（当前默认） | **0.5 ms/层** | **74 KB/层** | 8 cm 档 |

原因是**结构性的**，颜色量化救不了：一个方块的表面层有 56 个 `4³` 砖，每个砖横跨 4×4 个 texel
⇒ 砖内必然多色 ⇒ 每个砖都要一张 4³ 值表（CPU 128 B / wire 24 B）。4096 个方块 = 2600 万个砖，
一层就是 20 MB 级、产出 80 ms 级；而池预算按 1 MiB/块折算 ⇒ 只能装几十块，"视距"退化成几步。

8 cm 档把"每砖一个代表色"当作写入单位：砖内同色 ⇒ 值表消失、能并进父层；一个方块在
**石/陶土/混凝土这类单色块**上收缩成**一条 `extent 16` 的写**（产出 0.5 ms/层、树 74 KB/层）。
观感上贴图的色块变化仍保留（4×4 texel 的多数色），只是细节粒度从 2 cm 变成 8 cm。

颜色量化（每通道 5 位，默认）另有两个好处：调色板槽从 4317 → **716**（贴图噪点被并成同色），
冷启动的建计划耗时 31 → 10 ms。溢出回退（槽满后按 8 级/通道找最近色）实测**从未触发**。

出生点 chunk `y=4` 的颜色抽样（每 4 格一次，`real_map_produce` 输出）—— 贴图确实被用上了：

```text
槽 135 × 9856  [216,216,216]   white_terracotta     槽 168 × 4864  [80,64,48]   birch_planks（棕）
槽  90 × 7326  [120,120,120]   stone（灰）          槽  92 × 4805  [104,104,96] stone_bricks
槽  88 × 5248  [112,112,112]   stone（深）          槽 110 × 4352  [200,200,192] 浅色石材
```

## 7. 接进流式世界（`Detail` 的落地）

| `Detail` | 粒度 | 每方块写几次 | 谁要它 |
|---|---|---|---|
| `Full` / `Fine` | `4³` 体素（8 cm） | 1（单色块）..64 | 射线请求的 level 0/1、半径环 15..58 chunk |
| `Coarse` | 整块（16） | 1 | 半径环 59..233 chunk、请求 level 1 |
| `Wide` / `Chunk` | `4³` / 整层个方块 | 1 / 每 64 个方块 | 更远的请求 |

- 生产源通过 `Streaming::source` 注入（`scene::build_world` 里设），流式环发现源变了就重建 worker 池；
- 槽位由 worker 认领 ⇒ 主线程按 `ChunkSource::palette_log` 的游标把条目装进**各 volume** 的调色板
  （每帧一次、在**挂载之前**；换世界 / 换源时游标清零重放）；
- 建世界**不预铺任何 chunk**（一层最多 4096 次模型体素化，同步铺会卡住启动）⇒ 内容全交给流式环；
- 相机的默认机位在出生点（`level.dat`），存档姿态若离出生点 > 500 m 也回退到出生点（免得上一个世界的
  机位把相机丢在空气里）；世界代码**只设生产源**（`Streaming::set_source`）。
- **流式旋钮（半径 / 帧额 / 预算）由 DebugMenu「世界」页独占**：`apply_initial_state` 会把菜单的最终
  控件值重放成 `MenuActionEvent`（在 `setup` 之后一帧执行）⇒ 世界代码写它们会被当场覆盖。
  这一条是实测发现的：原先的 `Streaming::tune_for_mc`（`coarse_radius=4 / coarse_height=2`）被菜单里
  上一轮的 8 / 6 顶掉，而"环 8×6 = 3757 块候选"正是常驻集涨到 1758 块（≈90 MB）的原因
  （`DEMAND[batch …；环 8×6；常驻 …]`）。所以 `tune_for_mc` 已删除，改调参只能走菜单。
- **远场级（M8）已接**：远场每格代表 `scale` 个方块，走 [`mc::summary`](../gate-app/src/mc/summary.rs)
  的**摘要金字塔**（每 section 一份代表色 + 每 `4³` 方块一份细格），按"格中心那一列"采样
  ⇒ L1/L2 每块读 16 / 256 个 chunk 列，L3 每块读 256 个（详见 §8.3）。`VolumeGrid` 的
  `attach_far` 标记由 `mc::build` 与 `build_infinite_cubes` 共同设置，`scene.rs` 按它挂三级远场。

实测（`GATE_LOG=info,gate_app=debug`，出生点附近）：

```text
MC 地图 ...，出生点体素 [1840, 1136, -32]；资产 ...；窗口 ±32 chunk（±164 m）
STEP 2 world mc_map instances=1 written=0 dropped=0 3.25ms
MC 首个 section 产出（chunk -14,2,35）：调色板 155 槽（溢出回退 0 次）、区块缓存 (3, 0)、缺失资产 1 个
MC section 累计产出 256 / 512 / 768 / 1024 / 1280        ← 约 900 section/s（3 个 worker）
RESID[resident 1502 91770KB install … evict 0]           ← 收敛后 0 换出、0 请求
```

## 8. 已知问题与待办

### 8.1 已解决：空 section 的请求洪水（"空"与"没加载"必须分开）

**症状**：城里大多数 section 是空气 ⇒ `produce` 返回 `None` ⇒ 那一块不进 `b_struct`。而索引条目用
一个 `0` 同时表示"还没加载"与"是空的"，shader 分不出来 ⇒ 射线在**永远装不上东西**的空 chunk 上
反复发请求。实测（出生点默认机位，相机自己那一层就是空的）：

```text
REQ[去重 4 chunk、最热 v0(5,6,-2)×1030603 v0(8,6,-2)×13243 …）；本窗口 6876834 条、超容丢失 5828258]
```

最热一块 **103 万票/窗**、请求环（1 M 字）被这一块整圈覆盖 ⇒ **真正的请求全被挤掉**（ray-guided
逐块加载事实上失效），且每秒数百万次原子写砸在同一个字上。`infinite_cubes` 不会遇到这个问题
（`every_chunk_has_content` 是它的前提）。

**修法**：让"已知空"成为索引里的一等状态（跨 4 层，没有新 binding、没有新 buffer）：

| 层 | 改动 |
|---|---|
| `gate_voxel::VolumeGrid` | `empty_chunks` + `mark_empty_chunk` + `empty_seq`（与 `stream_window` 同类的"流式提示"） |
| `gate_app::infinite_cubes` | 源产出 `None` 时除了记自己的 `st.empty`，同时 `grid.mark_empty_chunk(cc)` |
| `gate_render::brickmap::builder` | 索引条目写哨兵 `INDEX_ENTRY_EMPTY`（u32::MAX）：新建 / 新记 / **窗口平移后重打**（平移会把索引区整体清零）；`note_empty` 跳过已常驻的块；`install_blob` 写真地址时自然覆盖 |
| `gate_render::brickmap::upload` | `plan_residency` 的 ①' 按 `empty_seq` 把新增空块同步给 builder（**逐卷**：MC 的远场级同样会产出 `None`） |
| `trace.wesl` / `brickmap.wesl` | 请求闸门只看 `entry == 0`（`has_tree = entry != 0 && entry != INDEX_ENTRY_EMPTY`）；遍历把两种都当空气 |
| `common.wesl` + `consts.rs` + `wesl_consts.rs` | 哨兵常量两侧各一份，启动时**逐字核对**（不等 = 把哨兵当树块地址解引用） |

**结果**（同一机位、同一时长）：

```text
REQ[去重 4 chunk、取 4 条（最热 v0(14,5,-1)×4251 v0(14,5,0)×2507 …）；本窗口 9592 条、超容丢失 0]
```

每窗请求 687 万 → **9592 条**（−99.86%）、超容丢失 583 万 → **0**、最热一块 103 万票 → 4251 票。
而且请求**真的被消费**了：`DEMAND[batch 4（请求 4）] → STREAM[gen4]`（常驻 1758 → 1762）——
ray-guided 逐块加载从"被洪水打满"变成"正常工作"。单测见
`brickmap::builder::tests::empty_chunk_marker_is_written_and_reapplied`。

**同一机制覆盖到静态世界**（`.vox`）：那种世界**没有生产源**（`stream_chunks` 不跑）⇒ 渲染窗口里
"没有内容"就是**确定是空**，不必等"产出过 `None`"（`plan_residency` ①″，上限 64K 窗口槽）。
实测 `nuke.vox`：请求 **2 831 521 条/窗、超容丢失 1 782 945、最热一块 157 707 票 → 0 条**
（补标 318/400 槽）。流式世界**不能**这么推（那块可能正在产出）。

哨兵同步按 `VolumeGrid::empty_log`（追加日志）的**游标**只处理新增块：早先按 `HashSet` 全量重扫，
爬坡期逐块触发 ⇒ O(n²)（一轮填充 2000 块 ≈ 200 万次哈希探测的白扫）。

### 8.2 待办

- 逐 texel（2 cm）若要做，得先解决树的"每砖值表"结构（例如按方块状态预存 16³ 模板 + block 级间接），
  不是加内存能解决的（见 §6）。
- 缺失资产 19 个（原版 `barrier` / `light` 这类不可见方块，以及资源包里没有的旗帜、墙上告示牌）
  ⇒ 这些方块**不产出**（skip），`assets.missing_names()` 可查。
- 面内 `rotation` 的 90/270 符号、元素自带 `rotation`（45° 交叉面）、`uvlock` 未实现。
- 生物群系染色取固定色（不读 `Biomes`）；材质参数是按名字的启发式，不读方块表。
- **远场的填充速率**：L2/L3 一块要读 256 个 chunk 列（约 **0.5 s**，见 §8.3 的实测）⇒ 视锥内铺满
  要几分钟。要提速就得**离线预计算一份粗粒度世界**（见 §8.4），或把 L2/L3 的采样步长调到 2。

### 8.3 已解决：远场级（M8）的 MC 通路

**问题**：远场 chunk 覆盖 `16·scale` 个方块每轴，逐块采样时 L3 的一个 chunk 要碰 `64³` 个 MC chunk，
不可行。

**做法**（[`mc::summary`](../gate-app/src/mc/summary.rs)）：**摘要金字塔**，按级降采样，每格只读
"它需要的那一个 chunk 列"：

| 级 | `scale` | 一格（方块） | 采样 | 每块读 chunk 列 |
|---|---|---|---|---|
| L1 | 4 | 4 | 所在 section 的 `4³` 细格（每节预算一次、64 格） | 16 |
| L2 | 16 | 16 | 格 = 一个 section ⇒ 该 section 的多数色 | 256 |
| L3 | 64 | 64 | **只读格中心那一列**（1/16 的采样面） | 256 |

摘要按 section 缓存（`Sec`：64 个细格 + 整节多数色 + 实体数，~200 B/节）。格的判据与主世界粗档同源：
**实体占比 ≥ 1/8 才写**（否则街道 / 空地会糊成实心）；色 = 多数非空气色。

**实测**（`real_map_far_produce`，出生点，`--ignored --nocapture`）：

```text
L1 scale 4：chunk (1,1,-1) ⇒ 1450/4096 格非空、树 5984 字 = 23.4 KB、3846.2 ms（新增缓存区块 16）
L2 scale 16：chunk (0,0,-1) ⇒ 1444/4096 格非空、树 2870 字 = 11.2 KB、2473.5 ms（新增缓存区块 256）
L3 scale 64：chunk (0,0,0)  ⇒  450/4096 格非空、树 1915 字 =  7.5 KB、 480.8 ms（新增缓存区块 256）
```

**读入的 chunk 列数与上表逐字吻合**（16 / 256 / 256）。耗时里前两级的绝大部分是**首次建方块计划**
（`plan_for` 的 16³ 栅格化，按状态缓存）—— L3 计划已热，256 列 **481 ms** ⇒ 稳态约 **1.9 ms/chunk 列**，
即 L2/L3 各约 **0.5 s/块**。块本身很小（7.5–23 KB），所以 GPU 侧的三级远场合计只占几 MB。

**接线**：`mc::build` 置 `attach_far` ⇒ `scene.rs` 调 `attach_far_levels_mc`（**只挂 volume、不装
调色板**：MC 的槽号由 worker 认领，靠 `palette_log` 重放填进**所有** volume）。

实跑（`GATE_BENCH=orbit`，60 s）：

```text
FAR L1 scale 4（1 级体素 = 4 世界体素）覆盖 ±0.66km；… 格 16 级体素 = 1.28m
FAR L2 scale 16 … 覆盖 ±2.62km；…
FAR L3 scale 64 … 覆盖 ±10.49km；…
STREAM[v1 far gen… chunks 92] / STREAM[v2 far … 40]        ← L1/L2 在涨
RESID[resident 2048 97672KB install 0 evict 4 | 远场 131块/装 0 evict 0]   ← 远场已在 GPU 上
EMPTY[vol1] 哨兵 +2（已知空共 686）… EMPTY[vol3] 哨兵 +2（已知空共 783）
FPS[1s] 60 低 57 高 63                                     ← vsync on，未掉帧
```

**L3 几乎不涨**（`DEMAND[v3 far] batch 151（全为请求）` 但常驻恒 0）：这不是没接上，而是
**L2 挡住了射线** —— 城市半径约 2.8 km，L2 已覆盖 ±2.6 km ⇒ 能走到 L3 的只有穿过 L2 空隙的少数射线，
而它们指向的 L3 格基本在城市之外（空气）。真要 10 km 的**有料**远场，得先让 L2 铺满（§8.2 的速率问题）。

### 8.4 若要"10 km 有料远场"：离线粗粒度世界

上面的通路是**按需读存档**的，成本随"射线要看多少 chunk 列"线性涨 ⇒ 铺满 ±2.6 km 要几分钟、
±10.5 km 更不可行。真要走通，只有一条路：**把整张图预计算成一份粗粒度世界**（例如 4 方块/格，
每格一色 + 高度），落盘成一个几十 MB 的文件，开机直接映射。Greenfield 是 34×34 个 region
（约 365K chunk、1.4 GB）⇒ 首次构建需完整过一遍存档。
