# 验收证据

- `acceptance.json`：本次发布的权威验收摘要。
- `unit-tests.log`：204 项测试，启用 `PYTHONWARNINGS=error`。
- `coverage.txt` / `coverage.json`：`src/pdh_v2` 行与分支覆盖率。
- `syntax.json`：全部正式 Python 文件按 Python 3.10 语法解析。
- `static-audit.json`：无第三方依赖的格式、安全、注释与复杂度审计。
- `random-legal-300.json`：300 手、102220 次合法决策的状态守恒压力证据。
- `verify-run-final/`：正式 `tools/verify.py` 报告；`report-no-overwrite.log` 证明报告目录拒绝覆盖。
- `review.md`：人工复审发现、修复与剩余边界。
- `fresh-directory.log`：从干净副本运行验证的输出。

这些证据支持 `engine_ready=true`，不支持 `training_ready=true` 或模型效果声明。
