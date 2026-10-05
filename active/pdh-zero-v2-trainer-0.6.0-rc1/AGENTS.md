# PDH-Zero v2 课程学习候选贡献者入口

先读 `README.md`、`docs/implementation-scope.md`、
`docs/curriculum-training-contract.md`、`rules/source-v0.9.md` 和 `TODO.md`。
`rules/source-v0.9.md` 是冻结房规的单一语义来源；改变规则必须同步更新来源、配置、
T案例、测试与 SHA256，禁止在模型代码中静默补值。

## 当前研究边界

0.5.0-rc3 的四家冷启动纯自博弈基线已经安全停止。它证明引擎、牌谱、检查点和 PPO
数值路径可运行，但约 99% 的训练手局流局，不能继续作为正式长跑方案。
0.6.0-rc1 只批准目标 RTX 5090 上的课程学习调优；`tools/long_job.py start` 被主动阻断，
课程门禁通过后再生成单独的长训练版本。

## 工具索引

| 功能 | 路径 | 调用 |
|---|---|---|
| CPU 完整验收 | `tools/verify_training.py` | `python3 tools/verify_training.py` |
| 课程学习调优 | `tools/train_curriculum.py` | `python3 tools/train_curriculum.py --help` |
| 调优晋级门禁 | `tools/curriculum_gate.py` | `python3 tools/curriculum_gate.py --run-dir <目录>` |
| 冻结规则验收 | `tools/verify.py` | `python3 tools/verify.py` |
| 静态审计 | `tools/static_audit.py` | `python3 tools/static_audit.py --output <新文件>` |
| 随机整局压力 | `tools/simulate_random.py` | `PYTHONPATH=src:. python3 tools/simulate_random.py --hands 1000` |
| 旧任务状态/停止 | `tools/long_job.py` | 仅用于已存在任务；本版禁止新建长任务 |

## 变更要求

- 编码前加载 Coding Rule；公共函数、关键状态与非显然房规必须有注释。
- 同步更新 `CHANGELOG.md`、`TODO.md`、相关契约和测试。
- 临时产物只放 `tmp/<用途-日期>/`，交付前清算或转正。
- 失败测试、来源哈希不一致、牌实体不守恒、非零和、信息泄漏或非有限指标阻断交付。
- 当前策略与启发式对手必须分开计数；启发式动作不得混入 on-policy PPO。
- 不通过增大模型或重复计算伪造 GPU 利用率；报告有效 Actor 决策与全桌决策。
- 不使用 `killall` 或全局 `pkill`；只停止登记的精确 PID。
