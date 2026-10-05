# PDH-Zero v2 训练与长任务候选 0.5.0-rc3

> rc3 是 rc2 的目标 CUDA 启动修复版。它兼容已经完成的 rc1 Smoke 检查点，
> 不需要重新跑 10 分钟 Smoke；请使用 rc3 重新创建长任务，不要恢复 rc2 的零更新失败目录。

本包在冻结房规参考引擎上提供两级训练入口：

1. `tools/train_smoke.py`：目标 RTX 5090 上的 10 分钟 CUDA Smoke；
2. `tools/long_job.py`：Smoke 通过后直接暖启动默认 48 小时／5000 万有效
   Actor 决策的可恢复长任务。

当前包内门禁状态：

```text
engine_ready=true
full_rules_frozen=true
training_entry_ready=true
long_job_entry_ready=true
formal_long_run_ready=false
```

`formal_long_run_ready=false` 表示发布包没有替代目标实例上的真实 Smoke。长任务启动器会
重新验证指定 Smoke 的清单、规则哈希、数值、零和账本、检查点和 gzip 牌谱；只有通过
后才创建后台任务。

## 已交付训练闭环

- Actor 只读取自己的暗牌与截至当前时刻真实公开的信息。
- 四家共享 Actor 参数，每桌每座位具有独立 GRU 隐状态。
- 只对规则引擎提供的合法实体动作评分，不构造非法动作。
- 多桌请求合并成 GPU 批量前向；环境推进仍使用单进程 Python 参考引擎。
- 每个 PPO 批次固定策略版本，达到目标手数后排空进行中的手牌再更新。
- 规则启发式教师用于无真人牌谱的短时行为克隆暖启动。
- PPO 主奖励只来自真实四家零和结算。
- 每手保存可确定性重放牌谱与四家原始账本。
- Smoke 保存模型、优化器和 RNG；长任务另外保存所有牌桌的下一手状态、庄位、连庄、
  牌墙随机流、预算游标和周期评估游标。

## CPU 发布门禁

```bash
python3 tools/verify.py
python3 tools/verify_training.py
python3 tools/verify_long.py
```

长任务专项报告应包含：

```text
long_job_candidate_check: PASS
long_job_entry_ready: true
smoke_promotion_required: true
default_max_hours: 48
default_target_decisions: 50000000
formal_long_run_ready: false
```

## RTX 5090 十分钟 Smoke

确保 GPU 没有其他计算任务后执行：

```bash
RUN_DIR=/root/rivermind-data/pdh-zero/evidence/v2-training-smoke/SMOKE_10M_$(date -u +%Y%m%dT%H%M%SZ)

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

Smoke 会拒绝覆盖已有目录，并默认拒绝与其他 GPU 计算任务共享设备。

## Smoke 通过后直接启动长任务

```bash
SMOKE_DIR=/root/rivermind-data/pdh-zero/evidence/v2-training-smoke/SMOKE_10M_<UTC时间>

python3 tools/long_job.py validate-smoke --smoke-dir "$SMOKE_DIR"
python3 tools/long_job.py start --smoke-dir "$SMOKE_DIR"
```

`start` 会再次执行同一晋级门禁，并在后台启动任务。默认预算：

```text
最长累计训练墙钟：48小时
或新增有效Actor决策：50,000,000
任一条件先满足即正常完成
评估间隔：每5,000,000决策
快照间隔：每5,000,000决策
磁盘安全线：剩余8GiB
```

启动返回 `job_dir` 和根 PID。之后使用：

```bash
python3 tools/long_job.py status
python3 tools/long_job.py stop
python3 tools/long_job.py resume
```

- `status` 读取精确任务指针，不按最新目录猜测。
- `stop` 只向登记的根 PID 发送 SIGTERM，在当前 PPO 更新完成后安全退出。
- `resume` 从最新完整更新边界恢复；原 Smoke 目录即使已归档也不影响恢复。
- 重复执行 `start` 时，如已有活动任务，会返回现有任务而不会创建一个注定失败的新目录。

## 长任务证据与恢复语义

- 每个已完成 PPO 更新写一个原子 gzip JSONL 牌谱分片，并按千次更新分目录。
- 每个更新同时写独立 JSON 指标；`metrics.jsonl` 是可重建的便捷视图。
- 崩溃发生在检查点提交前时，未提交的牌谱与指标会隔离到 `orphaned/`，恢复后从
  上一个完整更新边界重新计算；不会把孤儿样本重复计入正式数据。
- 周期评估使用固定启发式三家对手，并报告平均净分、CVaR-5%、胡牌、自摸、接炮胡、
  点炮和流局率。
- Smoke 到长任务属于模型与优化器暖启动，不延续同一批外生牌墙。

## 当前限制

- 环境推进仍是单进程 Python；本轮先取得正确、可恢复的长期基线，GPU 利用率不是唯一
  成功指标。
- GRU 在每手边界重置；跨手公开状态通过显式观察传递，尚未引入跨手记忆信用分配。
- 长任务检查点只在完整 PPO 更新边界提交；强制断电可能重算尚未提交的当前更新，但不
  会重复正式牌谱、付款或已提交样本。
- 本包没有在发布环境中预先运行目标 RTX 5090 Smoke，因此不能仅凭 CPU 门禁声明正式
  长任务已经取得研究结论。
