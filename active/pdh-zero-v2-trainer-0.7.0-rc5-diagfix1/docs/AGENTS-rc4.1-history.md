# PDH-Zero v2 0.6.0-rc4.1维护入口

先读README、docs/data-protocol-audit.md与docs/rc4-contract.md；冻结房规在rules/source-v0.9.md，禁止为通过门禁改规则或历史结果。CPP仅是数值核，Python合法动作/状态机/结算是参考。

## 可复用工具索引

| 功能 | 入口 | 场景/调用 |
|---|---|---|
| 准备/编译/回归/负载选择 | tools/prepare_rc4.py | --source-run 原rc3目录，不启动训练 |
| 目标ABI本地编译 | tools/build_native.py | 自动定位Python/PyTorch头，不下载依赖 |
| 全套CPU+静态+文件门禁 | tools/verify_refinement.py | native已构建则同时运行差分测试 |
| 真实牌谱差分 | tools/verify_native.py | --replays 牌谱 --hands 64 --output 新报告 |
| 顺序资源基准 | tools/benchmark_runtime.py | --selection 指针 --output 新目录 |
| 新受控训练 | tools/train_refinement.py | --runtime-plan 实测计划 --output-root 新父目录 |
| 账本/供牌链回算 | tools/audit_responsibility.py | --run-dir 旧run --output 新目录 |
| 开发结果门禁 | tools/refinement_gate.py | --run-dir 精确run |
| 完整手局与数据隔离离线审计 | tools/audit_data_protocol.py | --run-dir 旧run --output 新JSON，不占GPU |
| 独立牌墙确认 | tools/evaluate_candidates.py | 先读--help，冻结候选后只用新牌墙 |

责任监测不改奖励；供牌关联不称作因果贡献。风险头不进入Actor、不反传共享编码器。输入权重只从有哈希的selected_development导入，不盲用latest。候选先保存再评估，不自动长训/DUO，不承诺精确resume。

CPU配额用cgroup/affinity，不按宿主nproc。单次一个基准配置，进程PID/PGID登记，只清理自身。临时产物在tmp，构建在build，不进入源ZIP。禁止分享本地ABI的.so当作目标二进制。

源变更后重新生成FILES.sha256、编译和ZIP；最终干净解压验收。CPU夹具、真实权重短测与目标5090实测必须分开报告。旧工具为历史回归保留，不作为当前运行入口。

新运行默认新seed并记录真实值。不要恢复固定日期种子；完整单手与连续会话能力分开表述。开发集固定配对是有意重复；训练/验证跨用途已知重复必须阻断。保留.wall-exposures.sqlite3；历史缺失标partial，不能宣称全历史隔离。
