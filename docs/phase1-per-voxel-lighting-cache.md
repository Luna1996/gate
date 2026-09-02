# Phase 1: 启用 P3.5d 逐体素光照 hashmap 管线（Douglas Devlog #19 复刻）

## 关键发现（拆解前提）

P3.5d 4 pass 管线**已完整实现**——WGSL + Rust + CPU 参考 + 单测全在：

| 层 | 位置 |
|----|------|
| WGSL 4 entry | [dda.wgsl L1129 fl_clear_main](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L1129) / [L1147 fl_light_main](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L1147) / [L1194 fl_composite_main](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L1194) |
| WGSL hashmap 函数 | [fl_register L260](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L260) / [fl_lookup L299](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L299) |
| Rust pipeline + buffer + BG4 | [dda.rs L1371-1663 init_dda_pipelines + prepare_dda_bind_groups](file:///c:/code/repo.rust/gate/gate-render/src/brickmap/dda.rs#L1371-1663) |
| Rust CPU 参考 + 单测 | [face_light.rs L199-665](file:///c:/code/repo.rust/gate/gate-render/src/brickmap/face_light.rs#L199-665) |

**当前关闭状态**（[dda.rs L1689-1735 dispatch_dda](file:///c:/code/repo.rust/gate/gate-render/src/brickmap/dda.rs#L1689-1735)）：

- ✅ Pass 2 dda_main 实际 dispatch
- ❌ Pass 1 fl_clear_main 注释跳过
- ❌ Pass 3 fl_light_main 注释跳过
- ❌ Pass 4 fl_composite_main 注释跳过
- ❌ [dda_main 命中分支 L1105-1109](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L1105-1109) 被改回 per-pixel shading（直接 `shade_hit → ACES → sRGB → out_tex`），没写 G-buffer、没调 fl_register

**关闭原因**（[dda.rs L1689-1693 注释](file:///c:/code/repo.rust/gate/gate-render/src/brickmap/dda.rs#L1689-1693)）：

> hashmap 架构有 WGSL relaxed memory ordering race —— 跨 workgroup CAS 写入不保证 key 可见性 → 同一 voxel 分散注册、fl_lookup 命中率不稳定 → 逐帧闪烁噪点。

## race 机制（[dda.wgsl L257-259 注释](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L257-259)）

[fl_register L260-293](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L260-293) 流程：

1. CAS `fl_table[base]` 0→1 占位（单原子，安全）
2. 占位成功后用 4 个 `atomicStore` 写 key（x/y/z/obj_key，**多 word 撕裂窗口**）
3. `atomicOr` 写 mask

**race 场景**：

- 像素 P1 命中 voxel V：CAS slot[i] 成功 → 开始写 key（多 word，中途）
- 像素 P2 命中 voxel V（不同 workgroup）：CAS slot[i] 失败（status=1）→ `atomicLoad` 读 key → 跨 workgroup relaxed ordering 读到旧值/撕裂值（0,0,0,0 或部分 word）→ 误判异 voxel → 探测 slot[i+1] CAS 成功 → 写 V 到 slot[i+1]
- 结果：同 voxel V 分散到 slot[i]（mask=face 0）+ slot[i+1]（mask=face 1）

**已有修复**（治标）：

- [fl_lookup L296-321](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L296-321) 「mask 不含 face 时继续 probe，找含 face 的槽」——容忍分散注册
- [fl_clear L1133-1139](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L1133-1139) 清 status + key + mask，但保留 epoch + light + center（时间复用意途，开放寻址漂移导致读脏）

**未解决**：probe 漂移每帧不同 + 偶发 probe 耗尽 → fl_lookup 找不到 face → 兜底灰 0.3 → 逐帧闪烁。

## 目标

让同一 voxel 所有像素共享一次光照计算结果，达到 Douglas #19「逐体素纯色」pixel art 观感，**消除逐帧闪烁**。

## 范围（Phase 1）

- 解决 fl_register race（消除分散注册）
- 启用 dispatch 4 pass 序
- dda_main 命中分支改造（写 G-buffer + 调 fl_register，移除直接 shade_hit）
- 验证逐帧闪烁消除 + 逐体素纯色效果

**不在 Phase 1**（后续 Phase）：时间复用 hashmap、à-trous 去噪、半分辨率 path tracing 间接光、emissive 通过 radiance 三分支返回

## race 解决方案候选

| 方案 | 思路 | 优点 | 缺点 |
|------|------|------|------|
| **A. key_hash 编码到 status word 单原子 CAS** (推荐) | slot[0] = `bit31=status, bit0-30=key_hash`；CAS(0→0x80000000\|hash) 同时验证 status 空 + hash 匹配；后来者 load 比对 hash 部分同则聚合 mask | 单原子操作无撕裂；改动小（fl_register/fl_lookup/fl_clear 各几行）；hash 30 bit 碰撞概率 1/2^30 ≈ 0 可忽略 | hash 碰撞误聚合（极低概率，可加 key 全比对二次验证） |
| B. workgroup 内 shared memory dedup | 8x8=64 像素先 workgroup shared dedup，代表像素调 fl_register | 跨 workgroup 冲突概率降低 | 跨 workgroup 仍 race，不彻底；改动较大 |
| C. 两阶段注册（Pass A 占位 + Pass B 聚合） | Pass A 现状写；Pass B 扫全表聚合同 key 多 slot 到一个 | 彻底解决分散 | Pass B 全表 2M slot 扫描 + 聚合算法复杂 |
| D. pinned slot（hash 决定唯一 slot 无开放寻址） | 直接 atomicOr mask 到 hash(voxel) 槽，无 CAS | 无 race | 碰撞不同 voxel 共享 slot 误聚合；需要 key 校验丢弃冲突 |
| E. fl_lookup 全表聚合容忍分散 | 接受分散注册，fl_lookup 聚合所有匹配 slot 的 mask + light | 改动小 | 不解决 probe 耗尽丢注册；性能损失 |

**推荐方案 A**：最小改动 + 彻底消除 race。slot[0] 用 packed status+hash，CAS 时单原子验证「空槽 + hash 匹配」两个条件。

## 任务列表

| ID | 任务 | 验收标准 | 优先级 | 依赖 |
|----|------|---------|--------|------|
| M1 | race 方案 A 实施：fl_register/fl_lookup/fl_clear | 单原子 CAS 验证 status+hash；同 voxel 像素聚合到同 slot；单测通过 | P0 | — |
| M2 | dda_main 命中分支改造 | 命中写 G-buffer + 调 fl_register；sky 写 GB_SKY_BIT；移除直接 shade_hit；debug 模式保留 | P0 | M1 |
| M3 | 启用 dispatch 4 pass 序 | 取消 [dda.rs L1698-1734](file:///c:/code/repo.rust/gate/gate-render/src/brickmap/dda.rs#L1698-1734) 三 pass 注释；按 fl_clear→dda_main→fl_light→fl_composite 序 | P0 | M1, M2 |
| M4 | 集成 + 验证 | F5 看逐体素纯色；**逐帧闪烁消除**；cargo test 全绿；性能基准 | P0 | M3 |

## 子任务详情

### M1: race 方案 A 实施

**slot 布局调整**（[face_light.rs L24-37](file:///c:/code/repo.rust/gate/gate-render/src/brickmap/face_light.rs#L24-37) + [dda.wgsl L60 附近](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L60) 同步）：

- word 0：`packed_status_hash` = `bit31=status(0空/1占用), bit0-30=key_hash(30 bit)`
- word 1-3：voxel xyz（保留，OBJ center 等仍用）
- word 4：obj_key
- word 5：face_mask
- word 6：epoch
- word 7-24：6 面 RGB light
- word 25-27：center

**WGSL 改造**：

1. 新增 `fl_packed_key_hash(voxel, obj_key) -> u32` 函数（30 bit hash，高 bit 0）
2. [fl_register L265](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L265) 改 CAS：
   - 旧：`atomicCompareExchangeWeak(&fl_table[base], 0u, 1u)`
   - 新：`atomicCompareExchangeWeak(&fl_table[base], 0u, 0x80000000 | hash)`
3. [L283-289](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L283-289) 已占用分支：load packed，比对 hash 部分而非全 key
4. [fl_lookup L304-309](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L304-309) 同步改 hash 比对
5. [fl_clear L1133](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L1133) 清 packed status word（写 0 即可，已是当前实现）

**Rust CPU 参考 + 单测同步**：

- [face_light.rs](file:///c:/code/repo.rust/gate/gate-render/src/brickmap/face_light.rs) 改 `cpu_reference_registry_insert` 用 hash 编码逻辑
- 加单测：模拟并发分散注册验证聚合正确

**验收标准**：

- 单原子 CAS 同时验证 status + hash，无撕裂窗口
- 同 voxel 不同像素聚合到同 slot（理论保证）
- `cargo test` 全绿

### M2: dda_main 命中分支改造

**工作内容**（替换 [dda.wgsl L1092-1116](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L1092-1116)）：

1. 命中分支：
   - 计算 `albedo`（保留 best_pal 用于 G-buffer 或后续读取）
   - pack `meta_w = (best_obj & 0xFFFF) | (best_face_id << 16) | GB_HIT_BIT`
   - pack voxel = `vec3(best_voxel_coord)`
   - `textureStore(gbuf_tex, coord0, vec4(voxel, meta_w))`
   - 调 `fl_register(best_voxel, best_obj_key, best_face_id, world_center)`
2. sky 分支：
   - `textureStore(gbuf_tex, coord0, vec4(0,0,0, GB_SKY_BIT))`
3. debug mode 0/1/2 分支早 return（保留 [L1094-1104](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L1094-1104)），不进 G-buffer 路径
4. 移除 `shade_hit` 调用（逻辑已分散到 fl_light + fl_composite）

**验收标准**：

- dda_main 不再直接写最终颜色到 out_tex（除 debug mode）
- G-buffer 写入格式与 fl_composite 读取匹配（核对 [GB_SKY_BIT](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L120) / face_id 位段）
- fl_register 调用参数正确
- debug mode 1/2 仍正常工作

### M3: 启用 dispatch 4 pass 序

**工作内容**（[dda.rs L1698-1734](file:///c:/code/repo.rust/gate/gate-render/src/brickmap/dda.rs#L1698-1734)）：

1. 取消 Pass 1 fl_clear_main 注释
2. 取消 Pass 3 fl_light_main 注释
3. 取消 Pass 4 fl_composite_main 注释
4. dispatch 序：
   - fl_clear: `dispatch(ceil(FL_REG_SLOTS/64), 1, 1)`
   - dda_main: 保持现有
   - fl_light: `dispatch(ceil(FL_REG_SLOTS/64), 1, 1)`
   - fl_composite: `dispatch(ceil(width/8), ceil(height/8), 1)`
5. 移除 [L1689-1693](file:///c:/code/repo.rust/gate/gate-render/src/brickmap/dda.rs#L1689-1693) 「绕过 4-pass」注释，改回正常说明

**验收标准**：

- 4 pass 按序 dispatch
- 无 wgpu synchronization error

### M4: 集成 + 验证

**验收标准**：

- 同一 voxel 同一面所有像素颜色相同（逐体素纯色）
- **逐帧闪烁消除**（race 已解决）
- 阴影亮度逐面均匀
- `cargo test` 全绿
- 性能不低于改造前 per-pixel shading 80%

## 风险预判

| 风险 | 影响 | 缓解 |
|------|------|------|
| hash 30 bit 碰撞 | 误聚合不同 voxel 共享 slot | 概率 1/2^30 可忽略；fl_lookup 加 key 全比对二次验证（防极小概率） |
| G-buffer 格式与 fl_composite 不匹配 | 渲染全黑 | M2 实施前先核对 GB_SKY_BIT / GB_HIT_BIT / face_id 位段 |
| dda_main 改造破坏 debug mode | mode 1/2 失效 | debug 分支早 return，不进 G-buffer 路径 |
| dispatch 顺序错误 | hashmap 读到空值 | 严格 fl_clear → dda_main → fl_light → fl_composite |
| fl_light 跨帧 epoch 复用读脏 | 残留 light 错误 | fl_register 占位时 epoch=INVALID 强制重算（已有逻辑） |
| Rust / WGSL 常量不同步 | wgpu panic（历史教训） | 改 slot 布局时同步 face_light.rs + dda.wgsl + 单测 assert |

## 内存预算

- fl_table: `2,097,152 × 28 × 4 ≈ 229 MiB`（[face_light.rs L193 FL_BUF_SIZE](file:///c:/code/repo.rust/gate/gate-render/src/brickmap/face_light.rs#L193)）
- 2GB GPU 上限内可控

## 后续 Phase

| Phase | 内容 |
|-------|------|
| Phase 2 | 时间复用 hashmap（epoch 跨帧复用 + 槽位稳定化） |
| Phase 3 | à-trous 边缘感知去噪 |
| Phase 4 | 半分辨率 path tracing 间接光 |
| Phase 5 | emissive 通过 radiance 三分支返回（替换当前直出） |
