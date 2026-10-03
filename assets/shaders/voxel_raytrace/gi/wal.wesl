// GI **世界空间累积层（WAL）**：按可见体素面缓存的辐照度（M1：单层、认领者单写者）。
//
// 它替换的不是射线求值，而是"累积发生在哪"：现在（屏幕空间链）每个 GI texel 各攒各的
// `w_sum/M` 再重投影/降噪；WAL 把同一个**可见体素面**的全部样本并进"该面的一个持久条目"
// （同一个面的所有 texel 的候选本来就是同一积分的 i.i.d. 样本，见 `gi/screen.wesl` 文件头 ①）
// ⇒ 累积锚在世界上的面、**没有屏幕空间重投影** ⇒ 拖影/流光按构造消失。
//
// ---- 注入点（2026-10-03 第八轮起）----
// 值在 **`gi/screen.wesl::gi_ss_main` 写 `gi_out` 时**按成熟度混进该 texel 的估计
// （连同 `FACE_W_RAW*` 合并量），随后照常走整条降噪链（`gi_face_flatten` 的逐面平均 → 时域 →
// 5 轮 atrous）再进显示。
// ⚠️ **不要再改回"显示端替换"**（`main.wesl` 里被删掉的 `gi_wal_blend` 就是那个形态）：替换会把
// 降噪链的**空间滤波**白算，而 WAL 自己的池化只有 7×7 个面（≈38 屏幕像素、实测约一半 tap 是空的
// —— 没被认领过的面没有条目）⇒ 残余噪声是**世界空间固定**的斑纹（等它"收敛"不会消失，它已经
// 收敛了）；实测（19:29 那轮，Σn≈6700、成熟 100%）墙面仍有肉眼可见的 2~4% 斑纹，观感"脏"。
// 注入的值是**世界锚定**的 ⇒ 时域那一段不会把它拖出鬼影（重投影落回同一个面 ⇒ 历史里是同一个值）。
//
// M1 的边界（刻意做小、可 A/B）：
//   · 单层（没有粗层金字塔；冷条目按成熟度权重渐入 ⇒ 未熟处自动退回旧链的值）；
//   · **认领者单写者**：每面每帧只有它的认领 texel 入并（`screen.wesl` 的 `wal_claimer`）
//     ⇒ 无竞态、零额外射线（用的就是它本来要发的候选）；
//   · **同面邻面池化**（见 `GI_WAL_POOL_R`）：单个面每帧只有 1 条候选（分帧 8），逐面
//     独立用就是"一块一块的噪点"（2026-10-03 用户实测）；共面邻面物理上几乎是同一个积分
//     ⇒ 合并是纯降噪，且**不跨台阶/棱边**（只沿切向偏移 ⇒ 邻面与本面共面）。
//   · 多弹射不读本表；**写层与读层无环**：入并的只有 reservoir 的**新鲜候选**（射线），而
//     `gi_out` 那条链（含注入后的场）不回喂本表 —— `gi/ray.wesl` 从不读任何 GI 缓冲
//     （二次顶点的 GI 项恒 0，见那里的说明）。
//   · 失效靠掩码（原点 + pal；**区域修订号只在「太阳反弹」打开时折入**，见 `gi_wal_mask_at`
//     的 CONSTRAINT —— 那一条是 2026-10-03 修"编辑/飞行时全屏闪烁"的关键，别改回无条件折入）。
//
// 条目布局（`GI_WAL_WORDS = 8`，32 B/槽；槽数 2^20 = 32 MB，Rust `gi::WAL_SLOTS`）：
//   0/1 = 明文键 k0/k1（面身份）  2 = **失效掩码**（原点 + pal [+ 区域修订号，见上]）
//   3..5 = Σ π·L（与 reservoir 的 `w_sum` 同一口径）  6 = n（样本数，f32 位）
//   7 = 旗标（bit0 = mature：成熟过就**粘住**，失效时不清 ⇒ 注入权重不跳）
//
// 槽下标 = **世界坐标哈希**（`gi_unkey_voxel` 还原世界坐标后混合）——**不用窗口基址**：
// 相机移动 / 流式窗口滑动不改寻址、不重定基（键里的编码原点只进掩码）。
// 撞键（两个面哈希到同一槽）：互相覆盖 ⇒ 判据比对不过 ⇒ 当"没有条目"处理，只损失收益。
import package::bindings::{gi_u, gi_wal};
import package::gi::common::{gi_key0, gi_key1, gi_key_ok, gi_key_org, gi_local_rev, gi_mix};

const GI_WAL_WORDS: u32 = 8u;
/// 样本数上限（滑动窗）：到顶后同比例缩 ⇒ 值不变、窗继续跟着画面走（照 `gi_res_cap` 的手法）。
const GI_WAL_CAP_N: f32 = 256.0;
/// 成熟阈值：池子里的样本数（`Σn`）到本值就算成熟（注入权重 1）；不够按比例渐入。
const GI_WAL_MATURE_N: f32 = 64.0;
/// 失效（掩码变了）时把 n 压到本值：**值保留**、注入值不跳，接下来每帧以 `cnt/(n+cnt)` 快速追新真值。
const GI_WAL_RESET_N: f32 = 16.0;
/// **同面邻面池化的半径**（体素，切向）：`(2R+1)²` 个 tap（R=5 ⇒ 11×11 = 最多 121）。
/// 为什么不靠"各面自己多攒"：一个面每帧只有认领者的候选（分帧档下 ≈ 1 条），要压到旧链
/// （窗 256 × 逐面合并 × 5 轮 atrous ≈ σ/80）的噪声水平需要 ~6000 个样本 ⇒ 只能借邻面。
/// 为什么借得成立：只沿**切向**偏移 ⇒ 邻面与本面同 face、同法线轴坐标 = **共面**，
/// 同一张平面上的相邻面在物理上几乎是同一个积分；台阶 / 棱边 / 挡在前面的柱子都只能靠
/// **法向**偏移够到 ⇒ 池子天然不含它们（不会渗色）。
///
/// 半径越大越干净、代价越高（实测平均池 `Σn`）：R=2 ≈ 1.6k、R=3 ≈ 4k（MC 城市 / 1/4 + 分帧 8 /
/// 静止机位）。
/// **试过 R=5 并回退（2026-10-03 第九轮）**：针对用户实拍的"自发光体素周围网格状光纹"，
/// 按"小而亮发光体的半球采样方差（p≈1%~8% ⇒ 单面 σ 7%~23%，池到 Σn≈1 万仍有 3%~9%）"把半径
/// 提到 5（Σn ×2.5）—— 用户复拍**纹路依旧** ⇒ 那条纹路**不在面表池化这一层**，R=5 只换来
/// "落光变柔"（11×11 面 ≈ 22 体素）而没有变干净 ⇒ 回退。见 `docs/editable-gigavoxel.md` §10.26
/// 的"隔离清单"（分辨率档 / 降噪质量 / WAL 三个开关各切一次即可定位它到底在哪一层）。
/// **成本（若将来再调）**：池化在 **GI texel** 上跑（`gi/screen.wesl` 的注入点；§10.22 起不在
/// 逐像素的显示链上）⇒ 121 tap × ~6 万 texel ≈ +0.1~0.2 ms/帧（不是逐像素的 100 万次）。
const GI_WAL_POOL_R: i32 = 3;

/// 槽数掩码（`2^k - 1`）；表未就绪（占位 buffer / 非 2 的幂）⇒ 返回 `0u`（调用方据此跳过）。
fn gi_wal_nmask() -> u32 {
  let n = arrayLength(&gi_wal) / GI_WAL_WORDS;
  if (n < 16u || (n & (n - 1u)) != 0u) { return 0u; }
  return n - 1u;
}

/// 面 → 槽下标（世界坐标哈希）。`nmask` 由 [`gi_wal_nmask`] 给。
fn gi_wal_slot(v_world: vec3<i32>, face: u32, obj: i32, nmask: u32) -> u32 {
  let rkey = face | ((u32(obj) + 1u) << 3u);
  let h = gi_mix(u32(v_world.x)
          + gi_mix(u32(v_world.y)
            + gi_mix(u32(v_world.z) + gi_mix(rkey * 0x9E3779B1u))));
  return h & nmask;
}

/// 失效掩码（`org` 由调用方预先取出：池化要跑 49 次，`gi_key_org` 不必每次重读）。
/// **编码原点**（原点一变相对键就换基准 ⇒ 旧条目必须作废）+ **pal**（材质，逐面重刷要作废）
/// + **区域修订号**（流式挂载 / 编辑，`gi_local_rev`）——但**只在「太阳反弹」（NEE）打开时**才折入，
/// 见下面的 CONSTRAINT。任一项变 ⇒ 比对失败。
fn gi_wal_mask_at(v_world: vec3<i32>, org: vec3<i32>, obj: i32, pal: u32) -> u32 {
  // CONSTRAINT: **别把区域修订号改回"无条件折入"**。它与 `gi/ray.wesl::gi_face_shade_cached` 的
  // `rev` 是同一个口径（那里就是 `select(0, gi_local_rev(...), sun_bounce > 0.5)`）：只有太阳反弹
  // 打开时，"别的体素变过"才可能改变一个面的入射光（二次顶点的阴影射线）；关掉时入射光只由本面的
  // 材质 + 光 + 本面自己决定 ⇒ 区域修订号**不进掩码**（区域表本身也是为零个消费者设计的口径，
  // 见 `gi/common.wesl` 的「区域修订表」段）。
  //
  // WHY 必须门控（2026-10-03 实测，两个症状同一个根因）：区域表的 ±`REGION_REACH` 是为 **NEE 行程
  // （12 chunk）**膨胀的 = ±2 区域 = **±1024 体素（≈ ±20 m）**，而它的写入者包括**流式**（每帧挂载
  // 6~8 chunk ⇒ 边飞边写）。WAL 无条件折入它 ⇒ 编辑一次或飞行中，屏幕上绝大多数面的条目都被掩码
  // 打回 `GI_WAL_RESET_N`（16）的短窗：噪声立刻 ×4、再花 ~4 s 回填。实测读数（`DIAG[WAL …]`）：
  // 放 1 个体素后「平均池 Σn」**6700 → 933**、静止 ~4 s 才回到 6700；边飞边看则是反复掉到 ~800。
  // 观感 = "每次编辑全屏 GI 闪烁 + 相机一动就不稳定、收敛慢"。
  //
  // 代价（写明的取舍）：关掉太阳反弹时，编辑后的响应从"重置到 16 帧"变成**条目自己的滑窗**
  // （`GI_WAL_CAP_N` = 256 个样本 ⇒ τ ≈ 256 帧 ≈ 4 s）——即从"快而闪"换成"慢而稳"，与旧链的
  // 响应速度一致（旧链窗也是 256）。要更快就提高 `share` / 降低 `GI_WAL_CAP_N`，那是全局旋钮。
  var rev = 0u;
  if (gi_u.params.x > 0.5) { rev = gi_local_rev(v_world, obj); }
  return gi_mix(u32(org.x)
          + gi_mix(u32(org.y)
            + gi_mix(u32(org.z) + gi_mix(rev + pal * 0x9E3779B1u))));
}

/// 认领者把本帧的新鲜候选并入该面的条目。`sum_e` = Σ π·L（reservoir 的 `w_sum` 口径）、
/// `cnt` = 样本数。**单写者**（每面每帧只有认领者调用）⇒ 无需 CAS；并发只来自撞键。
fn gi_wal_merge(v_world: vec3<i32>, obj: i32, pal: u32, k0: u32, k1: u32,
                sum_e: vec3<f32>, cnt: u32) {
  let nmask = gi_wal_nmask();
  if (nmask == 0u || cnt == 0u) { return; }
  let slot = gi_wal_slot(v_world, (k1 >> 16u) & 7u, obj, nmask);
  let b = slot * GI_WAL_WORDS;
  let mask = gi_wal_mask_at(v_world, gi_key_org(obj), obj, pal);
  let same = (atomicLoad(&gi_wal[b]) == k0) && (atomicLoad(&gi_wal[b + 1u]) == k1);
  var sum = vec3<f32>(0.0);
  var n = 0.0;
  var mature = false;
  if (same) {
    let n_old = bitcast<f32>(atomicLoad(&gi_wal[b + 6u]));
    if (n_old > 0.0) {
      sum = vec3<f32>(bitcast<f32>(atomicLoad(&gi_wal[b + 3u])),
                      bitcast<f32>(atomicLoad(&gi_wal[b + 4u])),
                      bitcast<f32>(atomicLoad(&gi_wal[b + 5u])));
      n = n_old;
      mature = (atomicLoad(&gi_wal[b + 7u]) & 1u) != 0u;
      if (atomicLoad(&gi_wal[b + 2u]) != mask) {
        // 同面但失效：**保留值、n 压到复位值**（不跳变 + 快速追新；见文件头）。
        let keep = min(n, GI_WAL_RESET_N);
        sum = (sum / n) * keep;
        n = keep;
      }
    }
  }
  // EWMA 滑窗：先给新样本腾位置（同比缩 ⇒ 值不变）再加。
  let n_eff = min(n, max(GI_WAL_CAP_N - f32(cnt), 0.0));
  if (n > n_eff) { sum = sum * (n_eff / n); }
  sum = sum + sum_e;
  n = n_eff + f32(cnt);
  // 写入顺序：值 → n → 旗标 → 掩码 → 键（读者在后续 pass（dispatch 边界）；撞键的并发撕裂由比对兜底）。
  atomicStore(&gi_wal[b + 3u], bitcast<u32>(sum.x));
  atomicStore(&gi_wal[b + 4u], bitcast<u32>(sum.y));
  atomicStore(&gi_wal[b + 5u], bitcast<u32>(sum.z));
  atomicStore(&gi_wal[b + 6u], bitcast<u32>(n));
  atomicStore(&gi_wal[b + 7u], select(0u, 1u, mature || n >= GI_WAL_MATURE_N));
  atomicStore(&gi_wal[b + 2u], mask);
  atomicStore(&gi_wal[b + 1u], k1);
  atomicStore(&gi_wal[b], k0);
}

/// 单槽读取：键 + 掩码都逐位相符才有效。返回 `(Σ π·L, n)`（无效 ⇒ `(0,0,0,0)`）。
fn gi_wal_fetch(v_world: vec3<i32>, org: vec3<i32>, obj: i32, pal: u32, face: u32,
                nmask: u32, k0: u32, k1: u32) -> vec4<f32> {
  let b = gi_wal_slot(v_world, face, obj, nmask) * GI_WAL_WORDS;
  if (atomicLoad(&gi_wal[b]) != k0 || atomicLoad(&gi_wal[b + 1u]) != k1) { return vec4<f32>(0.0); }
  let n = bitcast<f32>(atomicLoad(&gi_wal[b + 6u]));
  if (!(n > 0.0)) { return vec4<f32>(0.0); }
  if (atomicLoad(&gi_wal[b + 2u]) != gi_wal_mask_at(v_world, org, obj, pal)) { return vec4<f32>(0.0); }
  return vec4<f32>(bitcast<f32>(atomicLoad(&gi_wal[b + 3u])),
                   bitcast<f32>(atomicLoad(&gi_wal[b + 4u])),
                   bitcast<f32>(atomicLoad(&gi_wal[b + 5u])), n);
}

/// **同面邻面池化**（`GI_WAL_POOL_R`，最多 25 个 tap）的原始输出：`(Σ π·L, Σn)`。
/// 均值 = `Σ(π·L) / Σn`（按样本数加权 ⇒ 统计上就是"把邻面的样本并进来"）；`Σn` 就是"这个池子里
/// 有多少样本"。调用方见 [`gi_wal_sample`]；诊断读数（平均 Σn）也走这条，用来判断"条目是不是
/// 被区域失效反复打回短窗"（正常应接近 `GI_WAL_CAP_N × 池子面数`）。
/// 邻居的**有效性由它自己的掩码判定**：几何/材质变过的邻面自动出局；越界键 / 缺席（空气、
/// 没被认领过）也自动出局 ⇒ 不需要额外的"同平面"判定（共面是由偏移方式保证的）。
fn gi_wal_pool(v_world: vec3<i32>, obj: i32, pal: u32, k1: u32) -> vec4<f32> {
  let nmask = gi_wal_nmask();
  if (nmask == 0u || !gi_key_ok(v_world, obj)) { return vec4<f32>(0.0); }
  let face = (k1 >> 16u) & 7u;
  // 切向基：法线轴 = face >> 1（`world.wesl::voxel_normal` 的约定），另两轴就是切向。
  let a1 = (face / 2u + 1u) % 3u;
  let a2 = (face / 2u + 2u) % 3u;
  let t1 = vec3<i32>(select(0, 1, a1 == 0u), select(0, 1, a1 == 1u), select(0, 1, a1 == 2u));
  let t2 = vec3<i32>(select(0, 1, a2 == 0u), select(0, 1, a2 == 1u), select(0, 1, a2 == 2u));
  let org = gi_key_org(obj);
  var sum = vec3<f32>(0.0);
  var cnt = 0.0;
  for (var i = -GI_WAL_POOL_R; i <= GI_WAL_POOL_R; i = i + 1) {
    for (var j = -GI_WAL_POOL_R; j <= GI_WAL_POOL_R; j = j + 1) {
      let v = v_world + t1 * i + t2 * j;
      if (!gi_key_ok(v, obj)) { continue; }
      let s = gi_wal_fetch(v, org, obj, pal, face, nmask, gi_key0(v, obj), gi_key1(v, face, obj));
      sum = sum + s.xyz;
      cnt = cnt + s.w;
    }
  }
  return vec4<f32>(sum, cnt);
}

/// 显示读取：见 [`gi_wal_pool`] 的说明。返回 `(π·L 均值, 权重)`：
/// 权重 = `min(1, Σn / 成熟阈值)` —— 池子一热就接管，冷时按样本数渐入（先糊后细、不跳变）。
/// 调用方把均值乘 `gain / π` 得到与 `gi_out` 同口径的 ⟨L⟩。
fn gi_wal_sample(v_world: vec3<i32>, obj: i32, pal: u32, k1: u32) -> vec4<f32> {
  let p = gi_wal_pool(v_world, obj, pal, k1);
  if (!(p.w > 0.0)) { return vec4<f32>(0.0); }
  return vec4<f32>(p.xyz / p.w, min(1.0, p.w / GI_WAL_MATURE_N));
}
