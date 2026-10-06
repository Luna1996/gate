---
version: 1
title: 多物体与刚体物理
---

# 多物体与刚体物理

## 1. 目标

在静态体素世界之上引入可独立变换的物体，并配套一套刚体物理引擎。碰撞几何直接复用渲染所用的体素树：不生成网格，不做凸分解，不复制体素数据。

**范围内**：物体生命周期、碰撞查询、刚体求解、并行与休眠、关节与马达、断裂与碎片回收、渲染耦合。

**范围外**：软体、流体、联机确定性同步。GPU 求解器列为可选 M7。

## 2. 既定决策

| # | 决策 | 依据 |
|---|------|------|
| D1 | 碰撞几何 = 渲染体素树 | 编辑即碰撞生效，零数据复制；Teardown 采用 voxel-vs-voxel |
| D2 | 接触原语为倒圆体素：面保持完整，角与棱倒圆 | 精确 SAT 在旋转下不具不变性，产生错误法线与抖动 |
| D3 | 体素先分类 corner / edge / face / interior，仅在 edge-edge 与 corner-(corner/edge/face) 之间生成接触 | 接触数量降到角与棱；地面上箱体仅四角出接触 |
| D4 | 求解器为通用约束系统 (C, J, λ)，接触是其一个实例 | 同一条 λ 表达式覆盖接触、关节、马达、鼠标拖拽 |
| D5 | Soft Step 子步进：1/60 切 4 子步，复用同一批接触，位置积分位于速度约束之后 | 子步进优于迭代；速度归零后位置不动，静止抖动消失 |
| D6 | 热启动 key = 两个相撞体素的坐标对 | 体素自带稳定接触 ID，无需额外分配 |
| D7 | 并行 = 接触图着色 + 宽 SIMD + 自有无锁任务池 | 无锁池：外部线程参与求解、根任务亲和、原子位图分优先级 |
| D8 | 静态世界只与驻留 chunk 碰撞 | 复用渲染驻留，不引入额外碰撞数据 |
| D9 | 物体姿态只写 `VolumeTransform`，不重建 brickmap | [builder.rs](../gate-render/src/brickmap/builder.rs) 逐帧 `sync` 已拷贝 transform，移动代价为一次 GridDesc 写入 |

**求解器选型不设 2D 预演**：以 M3 的数值门（堆叠高度、收敛时间、能量漂移）作为选型判据，在 3D 体素内直接判定。

## 3. 里程碑

| ID | 里程碑 | 验收标准 |
|----|--------|---------|
| M1 | 对象层 | 可 spawn / despawn 一个物体；落笔作用于选中物体；渲染逐帧跟随，无体素重传 |
| M2 | 碰撞查询件 | 查询 API 可脱离求解器单独调用；对给定位姿输出接触点、法线、穿透深度；点命中与 CPU raycast 结论一致 |
| M3 | 通用约束求解器 | 6 层箱体堆叠 2s 内收敛静止；自由落体落于主世界不穿透；接触、单关节共用同一求解路径 |
| M4 | 并行与休眠 | 1000 刚体 @60Hz 物理 ≤ 4ms；静止物体整帧零求解成本；帧时间无 >16ms 尖峰 |
| M5 | 渲染耦合 | 物体运动与旋转期间 GI 无拖影；醒体覆盖的 chunk 保持驻留；越界物体回收 |
| M6 | 关节与断裂 | 16 种连接可用；切断支撑后关节转移到新 body；断裂产生的新 body 进入模拟并回并主网格 |
| M7 | GPU 求解器（可选） | 窄相与求解迁至 compute shader，姿态直接写 GridDesc uniform，渲染路径零回读 |

## 4. 模块划分

新增 crate `gate-physics`，纯 Rust、不依赖 bevy，与 `gate-voxel` 同级风格，便于单测与后续迁移。

```
gate-physics/src/
  body.rs        RigidBody 状态、质量属性
  classify.rs    体素分类与接触加速结构
  query.rs       碰撞查询件（可脱离求解器调用）
  contact.rs     接触生成、倒圆体素原语、流形
  constraint.rs  约束抽象 (C, J, λ)
  joints.rs      16 种连接、马达、拖拽约束
  solver.rs      Soft Step 子步进、热启动
  broadphase.rs  SAP
  islands.rs     岛、图着色、休眠
  pool.rs        无锁任务池
  fracture.rs    断裂事件与碎片回并
```

## 5. 数据结构与接口

### 5.1 刚体

```rust
pub struct RigidBody {
  pub pos: Vec3,          // 质心
  pub rot: Quat,
  pub lin_vel: Vec3,
  pub ang_vel: Vec3,
  pub inv_mass: f32,
  pub inv_inertia: Mat3,  // 局部坐标系
  pub obj: i32,           // 对应 VolumeGrid::obj_id
  pub sleep_timer: f32,
}
```

质量属性在物体生成时按实心体素与逐体素密度统计一次；结构编辑后按脏 AABB 增量重算。

### 5.2 体素分类

```rust
pub enum VoxelClass { Interior, Face, Edge, Corner }
pub struct ContactAccel { pub edges: Vec<IVec3>, pub corners: Vec<IVec3> }
pub fn classify(tree: &ChunkTree) -> ContactAccel;
```

分类沿三轴判定：仅一轴两侧均有实心 → edge；三轴均有 → interior。加速结构随物体生成构建，编辑后按脏砖更新。

### 5.3 碰撞查询

```rust
pub trait CollisionQuery {
  fn solid_at(&self, world: Vec3) -> bool;
  fn voxel_at(&self, world: Vec3) -> Option<PaletteId>;
  fn sweep(&self, aabb: (Vec3, Vec3), dir: Vec3, t_max: f32) -> Option<SweepHit>;
}
```

实现者：静态世界（按 `ChunkCoord` 查树）、物体（world↔local 变换后查树）、角色控制器 hitbox。查询件不持有求解器状态。

### 5.4 约束

```rust
pub trait Constraint {
  fn count(&self) -> usize;
  fn c(&self, bodies: &[RigidBody]) -> f32;        // 误差，满足 ⟺ 0
  fn jacobian(&self, bodies: &[RigidBody]) -> Mat; // J = ∂C/∂q
}
```

λ 按统一表达式求解：`λ = (J M⁻¹ Jᵀ)⁻¹ (J M⁻¹ F + b)`，b 为偏置项。接触、关节、马达、拖拽均为 `Constraint` 实现。

### 5.5 接触键

```rust
pub struct ContactKey { pub obj_a: i32, pub voxel_a: IVec3, pub obj_b: i32, pub voxel_b: IVec3 }
```

跨帧复用冲量，堆叠稳定性由此保证。

## 6. 集成点

| 位置 | 改动 |
|------|------|
| [volume.rs](../gate-voxel/src/volume.rs) | `Volumes` 增加 `spawn_object` / `despawn_object`，包装既有 `add_object` |
| [edit.rs](../gate-app/src/edit.rs) | 移除 `hit.obj_id != -1` 早退，落笔可作用于物体 |
| [raytrace.rs](../gate-render/src/brickmap/raytrace.rs) | 复用为拾取与查询基准；不新增 CPU 射线实现 |
| [builder.rs](../gate-render/src/brickmap/builder.rs) | 无改动：`sync` 已逐帧拷贝 transform |
| [upload.rs](../gate-render/src/brickmap/upload.rs) | 物体位姿变化时，将旧 ∪ 新世界 AABB 喂入 `world_dirty_box` |
| [profiler.rs](../gate-render/src/profiler.rs) | 物理将醒体覆盖的 chunk 推入 `LodRequestFeed` |
| [chunk_tree.rs](../gate-voxel/src/chunk_tree.rs) | 使用既有 `node_desc` / `get_brick_state_extent` / `proxy` 做分层查询 |
| [dirty.rs](../gate-voxel/src/dirty.rs) | 编辑脏 AABB 作为唤醒来源 |
| [main.rs](../gate-app/src/main.rs) | 物理以固定步长系统接入，位于 `voxel_edit_input` 之后 |

## 7. 渲染耦合

物体姿态写入 `VolumeTransform` 后，渲染路径无需任何体素重传：

- `VolumesBuilder::sync` 每帧拷贝 transform；
- `GridDesc` 每帧重建，含位置、旋转三列、缩放；
- shader `trace_grid` 将射线变换到物体局部空间后遍历，任意刚体旋转精确成立。

唯一需要补充的是 GI 与降噪的历史失效：物体位姿变化时，把旧 ∪ 新世界 AABB 送入 `world_dirty_box`，否则该区域出现拖影。

## 8. 风险与对策

| 风险 | 对策 |
|------|------|
| 驻留缺失导致物体穿过未加载地形 | 物理推 `LodRequestFeed` 钉住醒体覆盖的 chunk；配越界与 kill plane 回收 |
| 物体运动使 GI 与降噪历史失效 | 旧 ∪ 新世界 AABB 喂 `world_dirty_box`；transform 变化时重算 GridDesc 的 AABB |
| 体素分类与加速结构开销 | 生成时一次性构建，编辑后按脏砖增量更新，不逐帧重建 |
| 关节附着失效 | 关节存可重绑定附着点；体素被切断后按新出现的 body 重建连接 |
| 求解器选型无 2D 预演 | 以 M3 数值门为判据：堆叠高度、收敛时间、能量漂移、休止残速 |
| 1000 刚体下的单线程瓶颈 | M4 前将 broad phase、窄相、求解分别计时，逐段定界 |
| 材料模拟与世界碰撞数据执行位置分离 | 短期按 chunk 回读变动区域并视为运动学更新；窄相迁 GPU 时 M7 转为必做 |

## 9. 元胞自动机兼容性

材料模拟以固定步长推进世界体素，与本文档的分层正交，但会使「世界体素只在用户编辑与破坏时变化」失效。以下四条为常驻约束，自 M1 起生效。

| # | 约束 | 理由 |
|---|------|------|
| C1 | 物理把世界当运动学体：子步内世界不动，世界变化以区块版本号暴露 | 子步进复用接触集要求步内静止；版本号用于失效接触缓存 |
| C2 | 体素分类与接触加速结构挂在砖块粒度 + 每砖脏标记，不随物体生成一次性构建 | 材料模拟逐帧触碰的区域必须可增量失效 |
| C3 | 热启动 key 由世界代次与体素坐标共同构成 | 区块回收与重挂后坐标别名 |
| C4 | 世界脏区统一为版本推进 + AABB，用户编辑与材料模拟共用同一条上行通道 | 两条通道重复触发 GI 失效与重传 |

**上行通道**：材料模拟的写入走 `mount_chunk_tree`（整块替换）与 `unmount_chunk`（块变空），不走逐体素的 `set_voxel`。

**体素状态编码**：当前 16 bit 全为调色板索引（`PALETTE_BITS = 16`，`LEAF_VOXELS_PER_WORD = 2`）。材料模拟所需的每体素状态（充盈度、寿命、温度）由并行状态叶承担，不改动既有位宽与叶打包；`comp_layer` 是同类侧层的先例。收缩材质位宽的方案会连带改动叶布局、着色器常量与 `.vox` 导入路径。

**交接边界**：岛从世界进入物理、碎片从物理回到世界，属一等公民边界，归 M6；岛检测采用区块边界六面标记与有界洪泛（上限约 8 个 chunk）。

**执行位置**：材料模拟置于 GPU 时，CPU 窄相按 chunk 回读变动区域；窄相迁移至 GPU 时 M7 转为必做。M2 的查询件按只读、可脱离求解器调用设计，用于容纳该迁移。


## 10. 验收基线

- 物体渲染跟随：位姿变化不触发 chunk 重建与体素上传。
- 碰撞查询可脱离求解器调用，供角色控制器、编辑器、拾取复用。
- 接触、关节、马达、拖拽共用同一条 λ 求解路径。
- 静止物体在无邻域变化时整帧不进入求解。
- 物理预算：1000 刚体 @60Hz ≤ 4ms。
