# Triplanar 纹理 + Bake Normal 管线设计（对标 Douglas #22）

> **v4 决策：triplanar PBR 纹理系统已拍板**（见 `docs/todo/README.md` 决策表「纹理路线」条目）。本文档是架构规格 + 实现 checklist。

---

## 1. 架构总览：normal 的三个生命周期阶段

Douglas #22 的核心决策（gate 天生对齐）：

> 旧引擎：normal 是**体素固有属性**（存 CPU contree）→ edit 任何体素都要同步维护邻居 normal → 做不出 fill/copy API → 「huge headache」
> 新引擎：normal 是**可丢弃的派生数据**（upload GPU 时 bake）→ edit 不触发 CPU 维护 → 脏 tile 重 upload 自然重 bake

gate 的三个阶段明确：

```
阶段 1: CPU 数据层（gate-voxel）
  ├─ 体素数据 = palette u8 + occupancy bitmap + comp_layer（u16 ID）
  ├─ ❌ 无 normal 字段
  ├─ ✅ 天生不需要维护 normal — edit/delete/fill_region 直接改 palette bitmap
  └─ 脏跟踪：DirtyEdit → 脏 tile 列表

阶段 2: Upload Bake 阶段（gate-render，脏 tile 上传时触发）
  ├─ 输入：tile 的 BrickMap（palette array + occupancy 4KB bitmap）
  ├─ 扫描 surface voxel（邻居 6 方向任一为 AIR = surface）
  ├─ 每个 surface voxel → 计算精确 normal（邻域分析，见 §2）
  ├─ 生成 triplanar 纹理坐标（normal 派生，见 §3）
  ├─ 从 Palette 查材质（albedo/roughness/metallic/emissive/transmission）
  ├─ 输出 → GPU 渲染缓存（见 §4）
  └─ 触发时机：脏 tile 队列 + 预算调度（每帧可上传若干 tile）

阶段 3: GPU 渲染缓存（wgpu buffer + 可选 texture）
  ├─ 驻留 = 视锥内 + 距离预算内的 tile bake 结果
  ├─ 淘汰 = 超出范围 → 回收到 pow2 桶空闲链
  ├─ 重建 = tile dirty → 重新 upload → 重新 bake
  └─ 消费 = DDA trace 命中后 → 读缓存 → triplanar 采样 → 面光照 × 纹理 = 最终色
```

**Douglas #22 "不存 normal" vs "bake normal" 精确区分**：

| | 旧方案（Douglas 推翻的） | #22 新方案 | gate 方案 |
|---|---|---|---|
| normal 存在哪 | **CPU 体素数据结构里** | **GPU 渲染缓存里** | **GPU 渲染缓存里**（同 #22） |
| 什么时候生成 | edit 时**同步维护**所有邻居 | upload GPU 时**一次性 bake** | upload GPU 时**一次性 bake** |
| 生命周期 | 和体素数据**同寿**（永久） | 脏 → 重 upload → 重 bake | 脏 → 重 upload → 重 bake |
| edit 操作影响 | edit 1 个体素 → 所有邻居 normal 重算（同步） | edit 1 个体素 → tile 脏标记 → 下次 upload bake | 同 #22 |

---

## 2. Bake Normal 算法

### 2.1 输入

```
BrickMap tile:
  ├─ occupancy: u32[128]  (4KB, L2 层 occupancy bitmap)
  ├─ palette: u8[4096]    (L4 砖块 palette 索引)
  ├─ palette_lookup: Palette (256 entries, runtime 常驻)
  └─ tile_coord: TileCoord (世界坐标)
```

### 2.2 Surface Voxel 扫描

```rust
// 伪代码：扫出所有 surface voxel
fn collect_surface_voxels(tile: &BrickMap) -> Vec<(IVec3, u8)> {
    let mut result = Vec::new();
    for z in 0..16 { for y in 0..16 { for x in 0..16 {
        let slot = x + y*16 + z*256;
        let pal = tile.palette[slot];
        if pal == 0 { continue; } // AIR 跳过
        // 检查 6 邻居：任一方向为 AIR → 是 surface voxel
        let is_surface = [
            (x>0,  slot-1),           // -X
            (x<15, slot+1),           // +X
            (y>0,  slot-16),          // -Y
            (y<15, slot+16),          // +Y
            (z>0,  slot-256),         // -Z
            (z<15, slot+256),         // +Z
        ].iter().any(|(cond, neighbor_idx)| {
            *cond && tile.palette[*neighbor_idx] == 0
        });
        if is_surface {
            result.push((IVec3::new(x, y, z), pal));
        }
    }}}
    result
}
```

### 2.3 精确 Normal 计算

**不是简单的 "邻居 AIR 方向取反"** — 那只能得到轴向 normal。triplanar 纹理需要**精确考虑所有 6 邻居实心体素分布**后的精确表面 normal：

```wgsl
// WGSL: 表面体素精确 normal 计算
fn compute_baked_normal(occupied: vec3<bool>, world_normal: vec3<f32>, tile_pos: vec3<f32>)
    -> vec3<f32>
{
    // 6 方向邻居实心标志（+X/-X/+Y/-Y/+Z/-Z）
    let nx = occupied.x, px = occupied.y;  // 注意：vec3 重映射
    let ny = occupied.z, py = occupied.w;
    let nz = occupied.z, pz = occupied.w;
    // ... 简化：用 v5 现有 per-pixel 命中 normal（dda_main trace_grid 返回的隐式轴向 normal）+ 邻居存在情况微调
    // Douglas #22 的实际做法（从字幕推断）：
    //   邻居实心体素存在情况 → 加权和作为精确表面 normal
    //   边界体素（边缘邻域密度变化）→ 正常的轴向 normal
    //   对角体素 → 斜向 normal（triplanar 采样时能得到更好的纹理方向）
}
```

**gate 简化版**（先做，验证 triplanar 能工作后再优化）：

```rust
// Rust：简化 bake normal — 6 邻居分析 → 加权和
fn bake_normal_for_surface_voxel(tile: &BrickMap, local: IVec3) -> Vec3 {
    let mut normal = Vec3::ZERO;
    let slot = (local.x + local.y * 16 + local.z * 256) as usize;
    // +X 邻居是 AIR → 贡献 -X normal
    if tile.palette[slot + 1] == 0 { normal.x -= 1.0; }
    // -X 邻居是 AIR → 贡献 +X normal
    if tile.palette[slot - 1] == 0 { normal.x += 1.0; }
    // +Y / -Y / +Z / -Z 同理
    if tile.palette[slot + 16] == 0 { normal.y -= 1.0; }
    if tile.palette[slot - 16] == 0 { normal.y += 1.0; }
    if tile.palette[slot + 256] == 0 { normal.z -= 1.0; }
    if tile.palette[slot - 256] == 0 { normal.z += 1.0; }
    // 归一化
    if normal.length_squared() > 0.0 { normal = normal.normalize(); }
    else { normal = Vec3::Y; } // 兜底：应该不会到达
    normal
}
```

**关键**：这个算法在**体素边界**（surface voxel）上天然给出正确的轴向 normal；在 45° 斜角处（如果有细分 0.25cm 体素形成锯齿边界）能给出加权平均的斜向 normal — triplanar 纹理在斜向 normal 处看起来更好（不会出现硬切割）。

### 2.4 Triplanar 纹理坐标派生

triplanar 不需要显式存 UV — shader 里直接用 world position + normal 算：

```wgsl
fn triplanar_uv(world_pos: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    let abs_n = abs(normal);
    // 权重：normal 越大的方向，该方向平面纹理占比越高
    let w = abs_n / (abs_n.x + abs_n.y + abs_n.z);
    // 三轴平面 UV（world_pos 的两个轴就是 UV）
    let uv_x = world_pos.yz;
    let uv_y = world_pos.xz;
    let uv_z = world_pos.xy;
    return vec3<f32>(uv_x.x * w.x + uv_y.x * w.y + uv_z.x * w.z,
                     uv_x.y * w.x + uv_y.y * w.y + uv_z.y * w.z,
                     dot(uv_x, w));
}
```

**所以 bake 阶段不需要算 UV** — shader 每帧从 world_pos + normal 即时算。bake 阶段只需要算 normal + 材质参数。

---

## 3. GPU 存储布局

### 3.1 Bake Buffer（每 tile）

```
每 tile bake buffer 布局：
  [4] surface_voxel_count: u32      // 该 tile 有多少 surface voxel
  [N × 20] surface_voxel_data:      // 每个 surface voxel = 20 bytes
    ├─ world_pos: vec3<f32>         // 世界坐标（DDA 命中点直接读）
    ├─ normal: vec3<f32>            // bake 出来的精确 normal
    ├─ albedo: vec3<f32>            // Palette lookup 得到
    ├─ roughness: f32               // Palette lookup
    └─ metallic: f32                // Palette lookup
```

**或者**：更紧凑的方案 — 把 bake 结果存进一个统一的 "voxel attribute" buffer（和 BrickMap 分离），用 voxel slot index 直接索引：

```
gate 现状的 BrickMap（L4）：
  palette: u8[4096]     (1 byte / slot)
  occupancy: u32[128]   (4KB)

新增 baked_attributes buffer（per-tile，dirty 时重建）：
  per_slot:
    ├─ has_normal: u8  (1 = surface, 0 = air 或 内部体素)
    ├─ normal: ivec4   (snorm10 × 4，压缩到 40 bits)
    ├─ albedo: u32     (8888)
    ├─ roughness: u8
    └─ metallic: u8
  每 slot = 12 bytes → 4096 × 12 = 48 KB / tile
```

**12 bytes/slot 方案的优势**：与 BrickMap slot index 1:1 对应（O(1) 索引），不需要 surface_voxel_count + 动态列表；shader 里 DDA 命中一个 slot → 直接读同 index 的 baked_attributes。

### 3.2 全局布局（所有驻留 tile）

```
tile 驻留管理（复用现有 pow2 桶空闲链）：
  每个驻留 tile 有 slot id（来自空闲链）
  bake buffer 在全局 buffer 里用 slot id × TILE_BAKE_BYTES 定位
  DDA 命中 tile_coord → 查 TILE_INDEX → 找到 slot id → 读 baked_attributes[slot_id][voxel_index]
```

### 3.3 VRAM 预算

```
单 tile bake buffer = 48 KB
驻留 512 tiles（工作间场景） = 48 KB × 512 = 24 MB  ← 非常小
驻留 4096 tiles（流式大世界） = 48 KB × 4096 = 192 MB ← 可控

texture atlas（所有 triplanar 材质贴图合成）= 单独 budget（~64 MB 典型）
```

---

## 4. Upload + Bake 集成

### 4.1 触发时机

```
现有 dirty 队列（R1-4 comp_dirty）:
  edit → DirtyEdit → 脏 tile 列表 → 脏 BrickMap 重上传

新增 bake 阶段:
  脏 BrickMap 重上传 → Bake Pass（compute shader 或 CPU 扫）
    → 生成 baked_attributes buffer
    → mark tile 为 baked
```

### 4.2 CPU Bake vs GPU Bake

| | CPU Bake | GPU Bake |
|---|---|---|
| **实现难度** | 简单（Rust 代码，R1-1 已有 BrickMap 扫描原语） | 中等（compute shader + atomic） |
| **开销** | CPU 线程，和 upload 同步 | GPU compute pass，和 upload 并行 |
| **gate 当前状态** | ✅ R1-1 CCL 已有 BrickMap 扫原语 | ❌ 无 compute bake pass |
| **推荐** | **先 CPU，后 GPU** | spike 阶段 CPU 足够（dirty tile 预算小）；规模上来再 GPU bake |

### 4.3 与现有管线的集成点

```
v5 管线顺序（single-pass compute + blit render）：
  dda_main（trace → shade_hit → ACES → sRGB → out_tex）
  blit_dda_view（out_tex → ViewTarget）

新增 upload + bake 流程（在 render schedule 里，per-tile）：
  CPU side:
    dirty tile queue → budget dispatch（每帧可处理 N 个 tile）
    → upload BrickMap（palette + occupancy）
    → run CPU bake → upload baked_attributes buffer

  GPU side（DDA hit → baked_attribute lookup → triplanar 采样）：
    dda_main: trace → 命中 slot → baked_lookup(slot) → triplanar
    shade_hit（v5, per-pixel 直接算）:
      let mat = hit_mat(...)  // Palette lookup（现有）
      let baked = baked_attrs[slot]  // 新增：O(1) slot 索引
      let albedo = if baked.has_normal != 0 {
          triplanar_sample(world_pos, baked.normal, atlas) × mat.albedo
        } else {
          mat.albedo  // fallback：纯色块（内部体素/air slot）
        }
      let col = shade_hit_body(albedo, mat, ...)  // 硬阴影 + sky gradient + emissive，全部不变
      textureStore(out_tex, coord0, vec4<f32>(col, 1.0));
```

---

## 5. Shader 管线集成

### 5.1 dda_main 命中后 triplanar 集成

```wgsl
// v5 single-pass dda_main 内，命中分支 shade_hit 之前插入 triplanar：
fn baked_lookup(tiles: BakedTileMap, tile_coord: vec3<i32>, slot: u32)
    -> BakedAttr
{
    // tile_coord → tile_slot（BrickMap 驻留管理已有机制）
    // baked_attr = tiles.baked[tile_slot].slots[slot]  // 12B/slot O(1)
    // return baked_attr（normal + has_normal + 材质烘焙字段）
}
```

### 5.2 v5 最终 shade_hit 形态（替换旧 composite pass 伪代码）

```wgsl
fn shade_hit(origin_fine, dir_fine, best_t, best_pal, best_obj, best_n, shadow_max_t)
    -> vec3<f32>
{
    // ---- 新增：baked attribute lookup ----
    let slot = ...;  // DDA 命中时已算好的 BrickMap slot index
    let baked = baked_lookup(baked_map, tile_coord, slot);
    let mat = hit_mat(best_obj, best_pal);  // Palette lookup（保留，材质参数仍从 Palette 来）

    // ---- albedo：triplanar × palette（或 fallback 纯色块）----
    let albedo: vec3<f32>;
    if (baked.has_normal != 0u) {
        let uv = triplanar_uv(hit_world_pos, baked.normal);
        let tex_color = sample_texture_atlas(mat.texture_id, uv);
        albedo = tex_color * mat.albedo;
    } else {
        albedo = mat.albedo;  // 内部体素/air slot：纯色块兜底
    }

    // ---- 以下是 v5 现有 shade_hit 逻辑（硬阴影 + sky gradient + emissive），全部不变 ----
    // 1. 硬阴影：hit → 投 1 条太阳射线（per-pixel，无 hashmap）
    let shadow_t = trace_ray(...);
    let shadowing = if shadow_t < shadow_max_t { 0.0 } else { 1.0 };

    // 2. Sky gradient：按法线 y 混合天顶/地平线
    let sky_light = mix(horizon_color, zenith_color, best_n.y * 0.5 + 0.5);

    // 3. 直光 = albedo × N·L × shadowing × sun_color
    let ndl = max(dot(best_n, sun_dir), 0.0);
    let direct = albedo * ndl * shadowing * sun_color;

    // 4. 环境光 = albedo × sky_light
    let ambient = albedo * sky_light;

    // 5. emissive 直出（不受阴影）
    let emissive = albedo * mat.emissive * EMISSIVE_EMIT_GAIN;

    // 6. 合 → DDGI 间接光（R3-10 后续叠加 max(direct+ambient, ddgi_indirect)）
    let col = (direct + ambient) + emissive;
    return col * view_u.exposure_pad.x;
}
```

### 5.3 光照管线不变

**光照计算 per-pixel 直接算（无 hashmap）** — 直光硬阴影 + sky gradient + emissive 直出。triplanar 纹理乘在 **albedo 层面**（shade_hit 最顶端），光照计算部分**零修改**。

composite 最终公式（与决策表一致）：

```
final = triplanar_texture(× mat.albedo) × max(sky_grad(直光, per-pixel 1 条太阳射线), ddgi_probe_indirect(间接光))
      + emissive_radiance(发光直出)
```

---

## 6. 资源预算汇总

| 资源 | 大小 | 备注 |
|---|---|---|
| 单 tile bake buffer | 48 KB (12B × 4096 slots) | 与 BrickMap 1:1 对应 |
| 驻留 512 tiles（工作间） | 24 MB | 可忽略 |
| 驻留 4096 tiles（流式） | 192 MB | VRAM 2GB 预算内 |
| Triplanar texture atlas | ~64 MB（2048×2048，RGBA8，20 张材质） | 可调 |
| **新增总量** | **~280 MB 上限** | VRAM 2GB 预算剩余 ~1.7GB 给 DDGI + BrickMap |

---

## 7. 与 Douglas #22 的对比

| | Douglas #22 | gate |
|---|---|---|
| normal 存储 | ✅ 删 CPU contree 里的 normal → upload GPU bake | ✅ 天生没存过 normal → upload GPU bake |
| bake 阶段 | upload GPU 时一次性算 normal + 纹理坐标 + bake | upload GPU 时一次性算 normal + 材质参数（texture coord shader 即时算） |
| triplanar 实现 | ✅ 有 | ✅ 要上 |
| CSG fill/copy 原语 | ✅ 有（删 normal 维护成本后写的） | ❌ 还没有（但 gate 天生不需要 normal 维护，成本比 Douglas 低） |
| grass/leaves decorations | ✅ 有 | ❌ 跳过（解谜游戏不需要） |
| absorptive transparency | ✅ 有 | ❌ 后置 |
| bake 算法 | GPU bake（compute shader） | 先 CPU bake，后 GPU bake（spike 阶段） |

**gate 的优势**：Douglas 花几个月重构引擎才能摆脱 normal 显式存储 — gate 从第一天就没有这个问题。bake normal + triplanar 的复杂度对 gate 来说是**纯新增**，不是重构。

---

## 8. 实现 Checklist（与 R6 队列对齐）

| ID | 任务 | 依赖 |
|---|---|---|
| **T1** | bake normal 算法 spike（CPU Rust 代码，单 tile 输入输出 → 验证 normal 正确） | — |
| **T2** | baked_attributes buffer 格式 + BrickMap 上传通道修改（脏 tile → 上传 BrickMap + bake） | T1 |
| **T3** | shader triplanar sampling 实现 + composite pass 集成 | T2 |
| **T4** | fallback 纯色块兜底（baked buffer 未就绪 → 纯 palette 渲染不崩） | T3 |
| **T5** | region fill/copy 编辑原语（Rust TileGrid 层面） | 与 T1-T4 并行 |
| **T6** | region fill/copy 上传集成（fill 一个 region → 哪些 tile 脏 → 批量 upload + bake） | T5 + T2 |
| **T7** | 性能预算断言（CPU bake 每 tile < 0.1ms；VRAM bake buffer ≤ 驻留预算） | T2 |
| **T8** | 验收：砖纹导线 / 金属 LED / 暗室墙面 triplanar 纹理正确；编辑 fill_region 后新体素 normal + 纹理 Bake 正确 | T1-T7 |

---

## 9. 后续扩展

- **CSG 原语**（T5/T6）— sphere/cylinder/torus 3D 栅格化 + fill/copy/union/subtract（Douglas #22 写了 5000 行）
- **GPU bake pass**（从 CPU 迁到 GPU compute）— 性能提升 + 与 upload 流水线并行
- **absorptive transparency**（彩色玻璃）— 材质 transmission 参数 + DDA 穿透累积
- **grass/leaves decorations**（如果需要户外 demo）— material tag → aux buffer → 光栅化合成
- **LOD bake 加速**（R1-12 LOD）— 粗叶 tile 的 bake buffer 更小（4cm→16cm 每叶更少 surface voxel）
