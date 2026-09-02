# Phase 2: 时间复用 hashmap + pinned slot（Douglas Devlog #19 复刻）

## 拆解前提

Phase 1 race 修复方案 A 用「开放寻址 + packed status+hash 单原子 CAS」消除跨 workgroup 撕裂 race，但开放寻址天然漂移（同 key 每帧可能落不同 slot）→ epoch 跨帧复用失效（slot 内 stored_epoch 可能属其他 key）。

CPU 参考实现（[face_light.rs L336-341 cpu_reference_face_light](file:///c:/code/repo.rust/gate/gate-render/src/brickmap/face_light.rs#L336-341)）的 epoch 复用语义已存在，但 GPU 端被 Phase 1 的 `fl_register` 占位时 `FL_EPOCH_INVALID` 强制重算锁死（[dda.wgsl L285](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L285) + [L1187 fl_light epoch 复用分支](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L1187)）。

## Douglas Devlog #19 原版方案（[transcript](file:///c:/code/repo.rust/gate/docs/devlog-notes/transcripts/Emissive%20voxels%20and%20fancy%20lighting%20%5BVoxel%20Devlog%20%2319%5D%20%5BVPetAcm1heI%5D.en.txt)）

Douglas 关键描述（#19 原话）：

> each key slot in the hashmap is then initialized by copying the radiance from the previous frame then the keys from the previous frame are cleared because they're no longer necessary

**核心机制**：
- **pinned slot**：slot = hash(key) 唯一决定，无开放寻址 → 同 key 跨帧稳定落同 slot
- **上一帧 radiance 作初始化**：新帧 slot 内 light 用上一帧值作起点（时间复用）
- **清旧帧 key**：保留 light、清除 key（让新帧重新写 key）

**Douglas 未处理碰撞**：未提开放寻址、未提 key 校验、未提碰撞覆盖。假设 **FL_REG_SLOTS >> 可见体素数**（hash 分布均匀 + 表容量远大于注册量 → 碰撞概率可忽略）。

## gate 现状与 Douglas 假设的差距

| 项 | Douglas | gate 当前 | 处理 |
|----|---------|----------|------|
| slot 寻址 | pinned（hash 决定唯一） | 开放寻址（hash + probe） | Phase 2 改 pinned |
| 碰撞处理 | 假设可忽略 | race 修复方案 A 用开放寻址容忍 | Phase 2 改 Douglas 假设 |
| epoch 复用 | slot 内残留 radiance 作初始化 | 占位时置 INVALID 强制重算 | Phase 2 解锁复用分支 |
| key 校验 | 不做（假设无碰撞） | race 修复做 hash 比对 + 全 key 校验 | Phase 2 保留 key 校验防碰撞误读（gate 容量与可见体素持平，碰撞必现） |
| FL_REG_SLOTS | 未公开 | 2^21=2M | gate 近景可见体素峰值 ~2M 与容量持平 → 碰撞率高 |

**碰撞 mitigation**：与 Douglas 假设一致 = 接受饱和场景碰撞丢注册风险。若 F5 严重闪烁，扩容到 2^22=4M=458MB（VRAM 预算 2GB 内）。

## 目标

让 GPU 端 epoch 跨帧复用生效：静止场景光照不重算（性能提升 + 视觉稳定）；epoch bump 时（编辑/OBJ 移动/主题变）触发重算。

## 范围（Phase 2）

- fl_register 改 pinned slot（无开放寻址；3 分支：空槽 CAS / 同 key 聚合 / 异 key 覆盖）
- fl_lookup 改 pinned slot + key 校验
- fl_clear 调整：保留 epoch + light + key（仅清 status + mask）
- fl_light 解锁 epoch 复用分支
- face_light.rs CPU 参考 + 单测同步

**不在 Phase 2**（后续 Phase）：à-trous 去噪、半分辨率 path tracing 间接光、emissive radiance 三分支

## 任务列表

| ID | 任务 | 验收标准 | 优先级 | 依赖 |
|----|------|---------|--------|------|
| T1 | WGSL fl_register 改 pinned slot | 单原子 CAS 占位无开放寻址；同 key 同 slot 稳定；异 key 覆盖置 INVALID 强制重算 | P0 | — |
| T2 | WGSL fl_lookup 改 pinned slot + key 校验 | 查表命中正确；碰撞时 hash 不匹配返回兜底 0.3 | P0 | T1 |
| T3 | WGSL fl_clear 调整：保留 epoch + light + key | 与 Douglas「初始化用上一帧 radiance + 清旧 key」对齐；gate 保留 key 做 epoch 复用校验 | P0 | T1 |
| T4 | WGSL fl_light 解锁 epoch 复用分支 | 静止场景光照不重算；epoch bump 时重算 | P0 | T3 |
| T5 | face_light.rs CPU 参考 + 单测同步 | cargo test 全绿；CPU/GPU 行为镜像 | P0 | T1-T4 |
| T6 | 集成验证 | cargo test 全绿；wgsl_compile 校验；F5 静止场景无重算闪烁 | P0 | T5 |

## 子任务详情

### T1: WGSL fl_register 改 pinned slot

**工作内容**（替换 [dda.wgsl L275-308 fl_register](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L275-308)）：

```wgsl
fn fl_register(voxel: vec3<i32>, obj_key: u32, face: u32, world_center: vec3<f32>) {
  let h = fl_hash(voxel.x, voxel.y, voxel.z, obj_key) % FL_REG_SLOTS;
  let key_hash = fl_packed_key_hash(voxel, obj_key);
  let packed = FL_STATUS_OCCUPIED | key_hash;
  let base = h * FL_WORDS_PER_SLOT;

  // 3 分支：空槽 CAS 占位 / 同 key 聚合 mask / 异 key 覆盖 CAS + 置 INVALID
  let stored_packed = atomicLoad(&fl_table[base]);
  if (stored_packed == 0u) {
    // 空槽：CAS 占位（保留 stored_epoch 跨帧复用）
    let cas = atomicCompareExchangeWeak(&fl_table[base], 0u, packed);
    if (cas.exchanged) {
      // 占位成功：写 key + face mask（epoch 不动——跨帧 epoch 匹配可触发复用）
      atomicStore(&fl_table[base + 1u], u32(voxel.x));
      atomicStore(&fl_table[base + 2u], u32(voxel.y));
      atomicStore(&fl_table[base + 3u], u32(voxel.z));
      atomicStore(&fl_table[base + FL_OFF_OBJ], obj_key);
      atomicOr(&fl_table[base + FL_OFF_MASK], 1u << face);
      if (obj_key != 0u) {
        // OBJ：存世界体素中心（旋转/缩放后面中心 ≠ voxel+0.5）
        atomicStore(&fl_table[base + FL_OFF_CENTER], u32(world_center.x));
        atomicStore(&fl_table[base + FL_OFF_CENTER + 1u], u32(world_center.y));
        atomicStore(&fl_table[base + FL_OFF_CENTER + 2u], u32(world_center.z));
      }
      return;
    }
    // CAS 失败：其他线程刚占位，重读一次走聚合/覆盖判断
    let repacked = atomicLoad(&fl_table[base]);
    if ((repacked & FL_HASH_MASK) == key_hash) {
      atomicOr(&fl_table[base + FL_OFF_MASK], 1u << face);
      return;
    }
    // 异 key：走覆盖分支
  }

  if ((stored_packed & FL_HASH_MASK) == key_hash) {
    // 同 key hash 匹配：聚合 face mask（无撕裂 race）
    atomicOr(&fl_table[base + FL_OFF_MASK], 1u << face);
    return;
  }

  // 异 key 碰撞：覆盖（CAS packed_old → packed_new）+ 置 INVALID 强制重算
  let cas = atomicCompareExchangeWeak(&fl_table[base], stored_packed, packed);
  if (cas.exchanged) {
    // 覆盖成功：置 INVALID + 写新 key + face mask
    atomicStore(&fl_table[base + FL_OFF_EPOCH], FL_EPOCH_INVALID);
    atomicStore(&fl_table[base + 1u], u32(voxel.x));
    atomicStore(&fl_table[base + 2u], u32(voxel.y));
    atomicStore(&fl_table[base + 3u], u32(voxel.z));
    atomicStore(&fl_table[base + FL_OFF_OBJ], obj_key);
    atomicStore(&fl_table[base + FL_OFF_MASK], 1u << face);  // 重置 mask 为本帧 face
    if (obj_key != 0u) {
      atomicStore(&fl_table[base + FL_OFF_CENTER], u32(world_center.x));
      atomicStore(&fl_table[base + FL_OFF_CENTER + 1u], u32(world_center.y));
      atomicStore(&fl_table[base + FL_OFF_CENTER + 2u], u32(world_center.z));
    }
    return;
  }
  // 覆盖失败：丢弃（与 Douglas 假设一致——表满或并发冲突）
}
```

**关键变更**：
- 取消 `for probe` 循环，slot = `h % FL_REG_SLOTS` 唯一
- 占位成功后**不置 INVALID**（保留 stored_epoch 跨帧复用）
- 异 key 覆盖时**置 INVALID** 强制重算（避免读到旧 key 的 light）
- 异 key 覆盖时**重置 mask = 1u << face**（不是 atomicOr，否则旧 key 的 mask 残留）

### T2: WGSL fl_lookup 改 pinned slot + key 校验

**工作内容**（替换 [dda.wgsl L315-345 fl_lookup](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L315-345)）：

```wgsl
fn fl_lookup(voxel: vec3<i32>, obj_key: u32, face: u32) -> vec3<f32> {
  let h = fl_hash(voxel.x, voxel.y, voxel.z, obj_key) % FL_REG_SLOTS;
  let key_hash = fl_packed_key_hash(voxel, obj_key);
  let base = h * FL_WORDS_PER_SLOT;
  let stored_packed = atomicLoad(&fl_table[base]);
  if ((stored_packed & FL_STATUS_OCCUPIED) == 0u) { return vec3<f32>(0.3); }
  if ((stored_packed & FL_HASH_MASK) != key_hash) { return vec3<f32>(0.3); }
  // 碰撞保险：key 全比对
  let kx = i32(atomicLoad(&fl_table[base + 1u]));
  let ky = i32(atomicLoad(&fl_table[base + 2u]));
  let kz = i32(atomicLoad(&fl_table[base + 3u]));
  if (kx != voxel.x || ky != voxel.y || kz != voxel.z
      || atomicLoad(&fl_table[base + FL_OFF_OBJ]) != obj_key) {
    return vec3<f32>(0.3);
  }
  let mask = atomicLoad(&fl_table[base + FL_OFF_MASK]);
  if ((mask & (1u << face)) == 0u) { return vec3<f32>(0.3); }
  let w = FL_OFF_LIGHT + face * 3u;
  return vec3<f32>(
    f32(atomicLoad(&fl_table[base + w])),
    f32(atomicLoad(&fl_table[base + w + 1u])),
    f32(atomicLoad(&fl_table[base + w + 2u])));
}
```

### T3: WGSL fl_clear 调整

**工作内容**（[dda.wgsl L1152-1163 fl_clear_main](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L1152-1163)）：

清 status + mask；**保留 epoch + light + key + center**（让 epoch 跨帧复用生效——下一帧同 key 落同 slot 时 stored_epoch 仍属本 key 上一帧值）。

```wgsl
@compute @workgroup_size(64, 1, 1)
fn fl_clear_main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= FL_REG_SLOTS) { return; }
  let base = gid.x * FL_WORDS_PER_SLOT;
  atomicStore(&fl_table[base], 0u);                   // status = 0（空，供 fl_register CAS）
  atomicStore(&fl_table[base + FL_OFF_MASK], 0u);     // face_mask = 0（新帧重新收集）
  // epoch (word 6) + light (words 7-24) + key (words 1-4) + center (words 25-27) 保留
}
```

**变更**：取消清 key（words 1-4）；保留 epoch + light + center。

### T4: WGSL fl_light 解锁 epoch 复用分支

**工作内容**（[dda.wgsl L1186-1187 fl_light_main epoch 复用](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L1186-1187)）：

现状已正确：`if (obj_key == 0u && stored_epoch == cur_epoch) { return; }` —— 但 Phase 1 因 `fl_register` 强制 INVALID，此分支恒不命中。T1 解锁后此分支天然生效，无需改 WGSL。

**校验**：确认 [L1186 stored_epoch 读](file:///c:/code/repo.rust/gate/gate-app/assets/shaders/dda.wgsl#L1186) 仍是 `atomicLoad(&fl_table[base + FL_OFF_EPOCH])`，与 T1 占位时「不动 epoch」配合生效。

### T5: face_light.rs CPU 参考 + 单测同步

**工作内容**：

- `cpu_reference_registry_insert` 改 pinned slot（HashMap 语义天然 pinned，无需改）—— 现状已对齐，确认即可
- `cpu_reference_face_light` 确认 epoch 复用语义对齐 GPU（L336-341 已正确）
- 新增单测：
  1. `pinned_slot_collision_overwrites_and_invalidates`：异 key 碰撞覆盖时新 key 重算（INVALID）+ 旧 key 该帧丢注册（不命中）
  2. `epoch_reuse_skips_recompute_when_static`：epoch 不变静止场景 fl_light 跳过重算
  3. `pinned_slot_same_key_stable_across_frames`：同 key 跨帧落同 slot（HashMap 天然，CPU 参考语义对齐）
- 已有 `packed_hash_aggregates_concurrent_face_registers` / `epoch_reuse_semantics` 单测确认仍通过

### T6: 集成验证

- `cargo build --workspace` 通过
- `cargo test --workspace` 全绿
- `cargo test wgsl_shaders_parse_and_validate` 通过（naga parse + validate）
- F5 目验（用户做）：
  - 静止场景光照稳定（epoch 复用生效）
  - 相机移动无新闪烁（碰撞覆盖置 INVALID 触发重算）
  - 饱和场景碰撞闪烁可接受（与 Douglas 假设一致）
  - 编辑体素后该 voxel 重算（epoch bump 触发）

## 风险预判

| 风险 | 影响 | 缓解 |
|------|------|------|
| **饱和场景碰撞丢注册** | FL_REG_SLOTS=2M vs 可见体素 2M 峰值，碰撞率高 → 单帧单 voxel 丢注册 → 灰兜底 0.3 闪烁 | 与 Douglas 假设一致；若 F5 严重闪烁，扩容到 2^22=4M=458MB（VRAM 预算 2GB 内） |
| 异 key 覆盖漏置 INVALID | 不同 key 同 slot 复用旧 light → 错误颜色 | T1 强制异 key 覆盖时置 INVALID；T5 单测覆盖 |
| 异 key 覆盖 mask 残留 | 旧 key mask 残留 → fl_lookup mask 检查误命中 → 返回旧 key light | T1 覆盖时 mask 重置（atomicStore 而非 atomicOr） |
| epoch bump 不及时 | 编辑后未重算 → 残留旧 light | face_light_epoch_tick 已实现 dirty/OBJ/主题三源；F5 编辑后验证 |
| key 校验未丢过期初始化 | 跨帧 slot 内残留异 key light → fl_register 复用错误 | T3 保留 key 做 epoch 复用校验；T1 CAS 占位时若 stored key != new key 走覆盖分支置 INVALID |

## 内存预算

- 现状：FL_REG_SLOTS=2^21 × 28 × 4 ≈ 229 MiB（[face_light.rs L223 FL_BUF_SIZE](file:///c:/code/repo.rust/gate/gate-render/src/brickmap/face_light.rs#L223)）
- 备选扩容：2^22=4M × 28 × 4 ≈ 458 MiB（VRAM 预算 2GB 内可控）

## 后续 Phase

| Phase | 内容 |
|-------|------|
| Phase 3 | à-trous 边缘感知去噪（间接光噪声治理；当前无 path tracing 不需） |
| Phase 4 | 半分辨率 path tracing 间接光（Douglas radiance 三分支：撞天→天空色 / 撞物体→环境色×距离衰减=AO / 撞发光体素→返回其颜色） |
| Phase 5 | emissive 通过 radiance 三分支返回（替换当前直出；与 Phase 4 同步实现） |
