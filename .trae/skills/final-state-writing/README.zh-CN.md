# Final-State Writing

[English](README.md) | 简体中文

> **「番茄炒蛋（无东坡肉）」**[^1]

这个中文公开案例呈现了一个具体的 AI 改稿问题：旧内容已经被纠正，但被否定的信息仍以否定句、解释、免责声明或其他形式留在最终文本中。

本项目将这类现象称为**纠错残留**（correction residue）。Final-State Writing 让最终稿描述当前接受的状态，而不是保留一路修改到该状态的历史。

```text
A + B
  ↓ 用户纠正：删除 B
A
```

而不是：

```text
A（不包含 B）
```

## 问题是什么

它处理的是多轮改稿后的特定失败模式：

```text
早期草稿
  ↓
用户反馈改变当前接受状态
  ↓
旧内容被删除
  ↓
旧内容的痕迹仍留在最终文稿中
```

典型表现包括：被否定方案仍出现在最终稿、已经过时的否定句、把纠错历史写进正文、因旧版错误而额外添加的防御性免责声明、只为解释删除原因而存在的句子，以及回答只在早期版本中才存在的反对意见。

这里的关键不是删除所有 `not`、`no` 或 `without`。如果一句关于缺失或限制的说明对最终产物本身有独立价值，它可以保留。例如部署说明中的“该 demo 无需外部基础设施”可能是有用的最终部署属性。判断标准是：如果从一开始就知道最终要求，这句话是否仍自然且有用？如果是，它不属于纠错残留。

## 前后对比

```text
草稿：
We use Method A and Method B.

反馈：
Use Method A only.

纠错残留：
We use Method A, not Method B.

最终状态：
We use Method A.
```

```text
草稿：
We compare Method A, Method B, and a transformer baseline.

后续反馈：
The transformer run was invalid.

纠错残留：
The transformer baseline is excluded because it used incorrect preprocessing.

最终状态：
We compare Method A and Method B.
```

## Final-State Writing 做什么

它把后续反馈视为当前目标的权威定义，并将受影响文本重写为仿佛最终要求从一开始就已知，同时保留事实、语气和有用细节。

它适用于收到反馈或纠正后的修订和定稿；不用于没有纠错历史的普通首次写作，也不替代通用润色、实现规划或风格检查。

## 验证

我们使用 GPT-5.6-terra 在一组专门用于触发该问题的多轮纠错任务上进行了测试：

- 未启用 Skill：20/21 输出出现纠错残留
- Skill 可用：4/21 输出出现纠错残留
- 7/7 类测试场景均得到改善

随后进行的定向发布回归测试：

- 任务正确性：12/12
- 按最终语义判定标准，纠错残留：0/12

这是一组针对该问题设计的高难度开发测试集，不代表所有 Codex 或 LLM 写作任务中的普遍错误率。

## 安装与使用

```bash
mkdir -p ~/.agents/skills
git clone https://github.com/intelland/final-state-writing \
  ~/.agents/skills/final-state-writing
```

显式调用：

```text
$final-state-writing
```

当任务与 Skill description 匹配时，Codex 也可能自动选择它。

## 参考来源

[^1]: 中文公开案例，[@songkeys 在 X 的帖子](https://x.com/songkeys/status/2090416137720999992)。本文中的「番茄炒蛋（无东坡肉）」仅作为该中文案例的引用；英文 README 中的对应表述是译写，不是对原作者英文原话的声称。