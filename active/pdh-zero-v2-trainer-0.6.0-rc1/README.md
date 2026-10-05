# PDH-Zero v2 课程学习候选 0.6.0-rc1

本版用于修复 0.5.0-rc3 完整纯大胡基线的冷启动失败：四个弱策略直接纯自博弈时，
训练手局约 99% 流局，终局净分信号不足。旧任务与检查点原样归档，不从旧模型继续。

## 本版改动

- Actor 观察增加合法信息范围内的八条并存牌路特征、资格状态和公开威胁。
- 候选动作增加执行后的牌路变化、不可逆资格损失、公开风险与选择价值。
- 状态与动作通过显式交互网络评分；模型默认隐藏宽度 384。
- 新增弃牌点炮风险辅助头；标签来自真实终局，Actor 仍只能看合法公开信息。
- 初期使用 `1个学习座位 + 3个规则启发式`，随后逐步增加学习座位数量。
- 启发式对手只提供环境行为；只有当前策略控制座位进入 on-policy PPO。
- 行为克隆预算扩大，并周期执行小规模教师复习，降低基础做牌能力遗忘。
- 日志分开报告学习者决策、全桌决策、启发式决策、座位分布、可胡机会、漏胡、
  点炮风险样本、学习者收益和课程阶段。
- 固定启发式三家对手评估用于比较调优前后收益与风险。

## 当前门禁

```text
engine_ready=true
full_rules_frozen=true
curriculum_entry_ready=true
formal_long_run_ready=false
```

`formal_long_run_ready=false` 是有意状态。只有目标 RTX 5090 上的课程学习调优通过
`tools/curriculum_gate.py` 后，才生成新的可恢复长训练包。本版执行
`tools/long_job.py start` 会明确拒绝启动，防止再次浪费长时间算力。

## CPU 验收

```bash
python3 tools/verify.py
python3 tools/verify_training.py
python3 tools/verify_long.py
```

`verify_long.py` 只验证旧恢复组件没有被破坏；报告中的 `long_job_entry_ready=false` 是预期。

## RTX 5090 课程学习调优

默认配置：

```text
启发式暖启动：2048手，3轮行为克隆
正式课程训练：3600秒（不包含暖启动和评估）
环境数：192
每次更新：至少384手
模型宽度：384
PPO：2轮，16384有效决策小批
教师复习：每10次更新，128手，1轮
固定评估：1024手，每500000学习者决策
```

执行：

```bash
RUN_DIR=/root/rivermind-data/pdh-zero/evidence/v2-curriculum/CURRICULUM_60M_$(date -u +%Y%m%dT%H%M%SZ)

python3 tools/train_curriculum.py \
  --output "$RUN_DIR" \
  --duration-seconds 3600 \
  --device cuda \
  --environments 192 \
  --hands-per-update 384 \
  --hidden-size 384 \
  --warmup-hands 2048 \
  --warmup-epochs 3 \
  --teacher-review-hands 128 \
  --teacher-review-interval-updates 10 \
  --ppo-epochs 2 \
  --minibatch-decisions 16384 \
  --evaluation-hands 1024 \
  --evaluation-interval-decisions 500000
```

总墙钟会高于 60 分钟，因为行为克隆、初始评估和最终评估不计入 3600 秒训练预算。

完成后运行：

```bash
python3 tools/curriculum_gate.py --run-dir "$RUN_DIR"
```

门禁同时检查：教师验证准确率、数值有限、混合对手末段流局率、末段合法胡牌总机会的
聚合捕获率、固定评估均分改善和点炮率是否明显恶化。门禁失败时保留全部牌谱并继续迭代，不直接启动长训练。

## 研究口径

主奖励仍然是冻结房规产生的逐玩家真实净分。行为克隆和混合对手课程只解决冷启动，
不改变最终优化目标。点炮风险是辅助监督，不读取对手暗牌，也不替代平均净分、CVaR、
胡牌率、点炮率和牌型分布的正式评估。
