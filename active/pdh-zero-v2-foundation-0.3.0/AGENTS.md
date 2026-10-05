# PDH-Zero v2基础包

先读README.md、docs/implementation-scope.md、rules/source-v0.8.md与TODO.md。规则源v0.8是当前语义权威；后续明确房规必须带来源更新，未知项不得默认关闭、跳局或私自设分。

## 工程边界

本仓库是新的纯Python参考基础，不是旧Mini-PDH重命名。不修改旧归档、不更换预装Torch、不启动云进程。每次变更同步CHANGELOG和测试。本轮已读取用户上传的coding-rule/SKILL.md；后续贡献者仍需自行加载其可访问版本。

## 可复用工具索引

| 功能 | 路径 | 调用 |
|---|---|---|
| 本批CPU验收 | tools/verify.py | `python3 tools/verify.py` |
| 规则就绪/115项追踪 | rules/cases.json、rules/open_decisions.json | 由验收工具汇总；原文位于rules/source-v0.8.md |

完整模型Checkpoint、旧实验打包等可复用工具仍在旧v1.1.1包，不为本批重复引入PyTorch或旧训练器。
