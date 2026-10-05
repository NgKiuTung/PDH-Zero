# PDH-Zero v2 训练候选 0.5.0-rc1

本包在 `0.4.0-engine` 冻结房规参考引擎上新增合法观察、候选动作循环策略、
多桌批量推理、规则启发式暖启动、Recurrent PPO、紧凑全量牌谱、结构化指标和
更新边界检查点。它用于在 RTX 5090 上执行 **10 分钟 CUDA Smoke**。

当前状态：

```text
engine_ready=true
full_rules_frozen=true
training_entry_ready=true
formal_long_run_ready=false
```

因此，本包已经可以启动短时训练门禁，但不得直接作为论文正式长任务。通过目标
实例上的 10 分钟 CUDA Smoke 后，再根据真实吞吐、非零终局比例和 GPU/CPU 瓶颈
决定多进程采样与 30–60 分钟学习短试配置。

## 已交付训练闭环

- Actor 观察只含自身暗牌与真实公开信息；其他玩家暗牌、未来牌墙、未来马牌不进入输入。
- 每个牌桌、每个座位维护独立 GRU 隐状态；四家共享模型参数。
- 只对规则引擎返回的合法实体动作评分，不构造非法动作。
- 多张牌桌合并为一次模型前向；环境仍由单进程 Python 参考引擎推进。
- 每个 PPO 批次固定策略版本，达到目标手数后排空所有进行中的手牌再更新。
- 规则启发式教师只使用合法信息，用于无真人牌谱的短时行为克隆暖启动。
- PPO 主奖励只来自真实四家零和结算；没有碰杠固定惩罚、路线数量奖励或伪造鸡胡。
- 每手保存可确定性重放牌谱和四家原始账本。
- 检查点保存 Actor/Critic、优化器与 Python/Torch RNG；当前恢复范围仅为更新边界，
  尚未恢复进行中的环境会话，因此正式长跑门禁仍为 false。

## CPU 验收

```bash
python3 tools/verify.py
python3 tools/verify_training.py
```

第二条预期包含：

```text
training_candidate_check: PASS
training_entry_ready: true
formal_long_run_ready: false
required_next_gate: 10-minute CUDA smoke on target RTX 5090
```

## RTX 5090 十分钟门禁

确保 GPU 没有其他计算任务，然后执行：

```bash
RUN_DIR=/root/rivermind-data/pdh-zero/evidence/v2-training-smoke/$(date -u +%Y%m%dT%H%M%SZ)

python3 tools/train_smoke.py \
  --output "$RUN_DIR" \
  --duration-seconds 600 \
  --device cuda \
  --environments 96 \
  --hands-per-update 192 \
  --hidden-size 256 \
  --warmup-hands 96 \
  --warmup-epochs 2 \
  --ppo-epochs 3 \
  --minibatch-decisions 8192
```

任务会拒绝覆盖已有目录；默认检测已有 GPU 计算进程并拒绝共享。结果包含：

```text
manifest.json
metrics.jsonl
events.jsonl
replays/hands.jsonl.gz
checkpoints/checkpoint_latest.pt
```

## 本候选版的明确限制

- 环境推进仍是单进程 Python，GPU 利用率不是最终吞吐目标；本轮先验证完整规则下的
  训练正确性、批量前向和真实数据规模。
- GRU 在每手边界重置；庄位、连庄和亮马等跨手状态通过显式观察传递，但尚未做跨手
  记忆信用分配。
- 更新边界检查点不包含进行中的环境会话；中断后不能声称无损续跑同一批牌墙。
- 自然大胡终局稀疏，启动前使用规则启发式暖启动；正式研究必须单独报告该阶段，
  并与无暖启动或其他课程条件做受控对照。
- `formal_long_run_ready=false` 在目标 GPU Smoke 和后续 30–60 分钟学习门禁通过前保持。
