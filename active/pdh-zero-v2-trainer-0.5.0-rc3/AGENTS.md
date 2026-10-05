# PDH-Zero v2 训练与长任务贡献者入口

先读 `README.md`、`docs/implementation-scope.md`、`docs/long-training-contract.md`、
`rules/source-v0.9.md` 和 `TODO.md`。`rules/source-v0.9.md` 是冻结房规的单一语义
来源；改变规则必须同时更新来源、配置、T案例、测试和SHA256，禁止在代码中静默补值。

## 工程边界

本仓库不是旧 Mini-PDH 重命名。不得修改旧归档、旧 Checkpoint、云实例 `env.sh` 或
已完成实验。短时训练与长任务入口均已交付，但长任务必须通过目标 Smoke 晋级门禁；
`formal_long_run_ready=false` 不能仅凭 CPU 测试改成 true。

## 可复用工具索引

| 功能 | 路径 | 适用场景 | 调用 |
|---|---|---|---|
| 训练候选 CPU 验收 | `tools/verify_training.py` | 发布前、上传实例后 | `python3 tools/verify_training.py` |
| 长任务专项验收 | `tools/verify_long.py` | 发布前、上传实例后 | `python3 tools/verify_long.py` |
| 限时训练 Smoke | `tools/train_smoke.py` | RTX 5090 晋级门禁 | `python3 tools/train_smoke.py --help` |
| 长任务控制 | `tools/long_job.py` | 晋级、启动、状态、停止、恢复 | `python3 tools/long_job.py --help` |
| 冻结规则 CPU 验收 | `tools/verify.py` | 发布前、上传实例后 | `python3 tools/verify.py` |
| 随机合法整局压力 | `tools/simulate_random.py` | 牌实体守恒与终局压力 | `PYTHONPATH=src:. python3 tools/simulate_random.py --hands 1000` |
| 无依赖静态审计 | `tools/static_audit.py` | 无ruff/mypy时的基础门禁 | `python3 tools/static_audit.py --output <新目录>/report.json` |
| 房规与案例追踪 | `rules/source-v0.9.md`、`rules/cases.json` | 规则实现与审查 | 只读或版本化更新 |

## 变更要求

- 编码前加载 Coding Rule；公共函数、关键状态和非显然房规必须有注释。
- 同步更新 `CHANGELOG.md`、`TODO.md` 和相关测试。
- 临时文件只放 `tmp/<用途-日期>/`，交付前清算。
- 失败测试、来源哈希不一致、牌实体不守恒、非零和或非有限指标均阻断交付。
- 性能实现必须与参考引擎做固定牌墙／动作流差分，不复制第二份房规目录。
- 不使用 `killall` 或全局 `pkill`；停止任务只操作登记的根 PID。
- `metrics/updates/` 与 `replays/` 是更新提交事实；便捷 JSONL 必须可由事实文件重建。
