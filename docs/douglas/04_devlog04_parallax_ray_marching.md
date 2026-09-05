# Devlog #4 - Drawing MILLIONS of voxels on an integrated GPU with parallax ray marching

- 视频: https://youtu.be/h81I8hR56vQ （12:10，2022）
- 类别: 基础渲染 | ⭐ 本系列最重要的架构转折点

## 一句话
为兼容集显/浏览器，从「纯 compute 光线步进」转向「光栅化 bounding box + 盒内光线步进」的混合管线（他称之为 parallax ray marching），集显跑 100 万+ 表面体素 60fps。

## 内容要点
- **转型的依据**：ray marcher 是**内存带宽瓶颈**——1660 Ti 比 1070 快 30%，因为带宽多 1/3；低端卡上纯 compute 完全不可行。
- **产品定位变化**：目标变成「平台」——浏览器可玩（WASM）+ 集显可跑 → C# 在浏览器又慢又没人支持 → **转向 Rust**（WASM 近原生、生态成熟；OO 思维 → 数据导向思维的转变）。
- **为什么不 greedy meshing**：只对共面表面有效，想要斜坡/细节表面就不行；不想每个面一个 quad。
- **混合方案**：对每 8³ 体素区域画一个 **紧致 bounding box**（真实三角形），**在 fragment shader 里对盒内做光线步进**、按相机视角手写像素。三角形数降一个数量级以上。
  - 关键优化：只生成**紧致盒**（零空隙）→ 最小化昂贵的 fragment 调用次数。
  - 结果：36×256³ 体积、表面 100 万+可见体素，Intel UHD 集显带 shadow mapping 60fps+。
- **Early-Z 陷阱**（重要）：fragment shader 手写深度会禁用 early depth test → 所有片元必须全跑。解法：**两遍渲染**——第一遍正常画盒子（保留 early-Z）+ 把真实体素深度写到自定义 buffer；第二遍再覆写深度缓冲。
- **MSAA 与手写像素天然互斥** → 改用屏幕空间后处理 FXAA。

## 对 gate 的启示
- 直接印证了 gate 硬约束的一个风险面：**纯 compute DDA 的本质瓶颈是内存带宽**，1660（M2 基线机）带宽恰是短板；Douglas 的数据（带宽差 1/3 → 快 30%）可作为我们做 1660 降档决策时的量化参照。
- 我们的架构（compute DDA → storage texture → blit）已是"后处理式"，天然放弃 MSAA（项目 Msaa::Off 决策与他一致）；FXAA/后期 AA 是同款出路。
- 「紧致 bounding box 才省 fragment」与我们 demo 里「空 ray slab 直接 0 步」的思路一致，可继续往紧致化方向抠（如按 slab/tile 粒度 dispatch）。
- 若未来 1660 降档失败，他的「光栅化盒子 + 盒内步进」是已被验证的退路（早-Z 两遍法照抄即可）。
