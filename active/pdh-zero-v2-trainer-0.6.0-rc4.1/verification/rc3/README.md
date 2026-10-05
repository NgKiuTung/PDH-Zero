# 0.6.0-rc3 本地验收证据

本目录是本次代码候选的CPU验收，不是用户warmup或RTX 5090训练结果。
旧verification/rc2等目录仅是历史记录。

- checks.json / cpu-tests.log：298项回归，warnings视为错误，测试进程退出0。
- coverage.json：语句覆盖87.98%，含分支84.60%，独立分支72.38%。
- static.json：基础静态审计0硬错误；复杂度软提示保留。
- cli-report.json：实际CLI完整运行、候选保存、离线门禁、独立新墙纯评估、SIGINT清理。
  fixture宽度16，开发门禁退出3、独立验证退出2分别为真实INCONCLUSIVE/REGRESSION；
  集成测试验证退出口径和证据完整性，不将小型随机fixture的分数说成模型变强。
- benchmark-64.json：64桌/19139动作的一次串行、Pipe、共享数值缓冲参考微基准；
  三者终局哈希一致。共享比Pipe采样吞吐约提高13%，含启动/退出约提高12%。
  这不是含GPU推理与PPO的端到端基准，更不能外推用户实例的占用率。
- runtime.json：Python3.13.5、torch2.10.0+cpu、约4核CPU额度/4GiB内存，没有CUDA。
- lint-install.log：本地包源没有提供ruff，ruff和mypy均未运行；基础审计不是替代性通过。

已验证的候选包仍须在目标环境加载实际warmup。没有实际训练或独立牌墙优势结果，
不允许把formal_long_run_ready改为true。最终ZIP还需要独立解压运行统一验收，
该过程日志作为ZIP外的release-check交付，避免自引用哈希。
