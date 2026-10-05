# 0.5.0-rc1 验证证据

本目录是训练候选入口的本地 CPU 验证证据。它不声称目标 RTX 5090 CUDA
Smoke 已完成，也不批准正式长任务。

- `unit-tests-final.log`：212 项测试，启用 `PYTHONWARNINGS=error`。
- `training-module-coverage.json`：新增观察、策略、批量环境与训练模块覆盖率。
- `verify-training-final/`：非覆盖式训练候选门禁报告。
- `summary-final.json`：可机器读取的完成状态与明确限制。
- `strict-static-tools.json`：离线环境中 ruff/mypy 的可用状态。
