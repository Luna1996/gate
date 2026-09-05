# 番外 - INSANE bug in my code from compiler optimization

- 视频: https://youtu.be/hBjQ3HqCfxs （4:44，2024）
- 类别: 其他 | ⭐ 编译器优化踩坑案例：enum 构造 UB 把检查优化没了

## 一句话
升级 nightly 后引擎崩溃：八分仪枚举在范围检查**之前** unsafe 构造，编译器据「enum 只有 0..7 合法值」这一 soundness 约定反推出 `counter < 8` 恒真 → 把检查直接优化掉；修复 = 先检查后构造。

## 内容要点
- 现象：`self.octant_counter < 8` 检查恒真，counter 实际一直涨过 8；打日志、看 wasm 汇编确认非幻觉。
- 根因：Rust 中构造超出定义域的 enum 即 UB（即使不使用该值）；编译器/LLVM 用「构造了合法 enum ⇒ 值必在 0..7」做推理，反向消掉了检查。
- 教训：**soundness 规则不是建议**——旧编译器上"碰巧能用"的 unsafe 代码，新优化器一出就炸；把构造挪到检查之后（if 内）即修复。

## 对 gate 的启示
- gate 的 palette u8 索引、组件 u16 ID、槽位 tag 位域若有 `transmute`/`from_repr_unchecked` 类操作，必须保证不变量在构造前成立——这正是「先校验后构造」纪律。
- 升级 nightly/rustc 后行为突变时，优先怀疑 UB 被 UB 驱动的优化暴露，而非编译器 bug；与项目记忆里「疑似 ICE 先看 panic 尾部」同属编译器诊断套路。
