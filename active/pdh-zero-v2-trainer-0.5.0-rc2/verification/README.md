# 0.5.0-rc2 验证证据

`rc2/` 是本版本的发布证据：

- `unit-tests-final.log`：完整 CPU 自动测试；
- `unit-tests-long-final.log`：长任务专项测试；
- `verify-engine/`、`verify-training/`、`verify-long/`：三个结构化门禁；
- `job-control-integration.log`：真实后台启动、重复启动、停止、恢复、再次停止；
- `static-audit-final.json`：无第三方静态基础审计；
- `syntax-py310.json`：Python 3.10 语法解析；
- `release-summary.json`：发布摘要。

本地构建环境未安装 ruff 与 mypy，因此两项明确为 `NOT_INSTALLED`。目标 RTX 5090
Smoke 必须在用户实例上完成，发布包不伪造 GPU 长跑通过状态。
