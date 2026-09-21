---
description:
globs:
alwaysApply: true
---

# 代码检索优先使用 codebase-memory-mcp

本项目已建立 codebase-memory 代码图谱索引，项目名固定为 `C-code-repo.rust-gate`。

查找符号、理解调用关系、评估改动影响时，优先使用 codebase-memory-mcp 的工具，而不是直接全量 Grep 或遍历目录：

- `search_graph`：按名称 / 正则 / 语义查找符号，返回 qn、文件、行号及 CALLS / USAGE / INHERITS 进出边
- `trace_path`：追踪两个符号之间的调用路径
- `get_file_outline`：查看单个文件的结构
- `get_code_snippet`：按 qn 读取符号源码或成员大纲
- `get_architecture`：查看模块结构、依赖、热点、分层
- `search_code`：带图谱排序的文本搜索（Grep 定位不到时的兜底）
- `detect_changes`：把 git diff 映射到受影响文件与调用方，提交前评估影响面

调用时必须显式传 `project = "C-code-repo.rust-gate"`。

索引由 [.trae/hooks.json](../hooks.json) 在会话开始与每轮问答结束时自动重建，无需手动重新索引。判断索引是否新鲜可看 `.codebase-memory/artifact.json` 的 `commit` / `indexed_at`。

`assets/shaders/**/*.wesl` 不在图谱覆盖范围内，查询着色器代码请直接读文件。

本规则只约束「如何检索与定位代码」；读取和修改文件仍使用 Read / Edit / Write。