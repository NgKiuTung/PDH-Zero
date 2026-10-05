# P1-14 Gate B 7.5 小时长任务

## 目标

在同一 RTX 5090 实例上，以相同规则、模型规模、优化器、环境局数和固定评估牌墙，平衡训练：

- 普通 Critic × 3 个种子；
- Privileged Critic × 3 个种子。

训练按 seed 配对、方法交替的 Round-Robin 顺序推进，避免某种方法独占早期或晚期硬件时段。默认累计 **7.5 小时有效训练时间**，随后执行 5,000 个固定牌墙种子的四座位轮换评估，因此总墙钟时间通常超过 7.5 小时。

## 前置门禁

- Gate A A00-A06 已通过；
- GPU 上没有其他计算进程；
- 数据盘至少剩余 5GiB；
- 已执行 `source /root/rivermind-data/pdh-zero/env.sh`；
- 不修改预装 PyTorch/CUDA。

## 启动

```bash
cd "$PDH_PHASE1_ROOT"
source .venv/bin/activate
python scripts/14_long_job.py start \
  --training-hours 7.5 \
  --chunk-updates 20 \
  --methods plain,privileged \
  --seeds 20260929,3407,9102 \
  --evaluation-wall-seeds 5000 \
  --device cuda
```

启动器返回 `JOB_DIR` 和 `PID`。进程使用独立会话运行，关闭 Terminal 不会停止任务；关闭或释放实例会停止任务。

## 查看状态

```bash
python scripts/14_long_job.py status
```

查看实时事件：

```bash
tail -f "$(ls -dt "$PDH_RESULTS_ROOT/gate-b/long-runs"/* | head -n 1)/events.jsonl"
```

查看进程日志：

```bash
tail -f "$(ls -dt "$PDH_RESULTS_ROOT/gate-b/long-runs"/* | head -n 1)/stdout.log"
```

## 安全停止

```bash
python scripts/14_long_job.py stop
```

停止请求不会全局终止 Python 进程。当前训练 Chunk 完成并保存 Checkpoint 后，Campaign 进入 `stopped` 状态。

## 恢复

```bash
JOB_DIR="/root/rivermind-data/pdh-zero/evidence/gate-b/long-runs/<job-id>"
python scripts/14_long_job.py start \
  --resume-job-dir "$JOB_DIR" \
  --training-hours 7.5 \
  --chunk-updates 20 \
  --methods plain,privileged \
  --seeds 20260929,3407,9102 \
  --evaluation-wall-seeds 5000 \
  --device cuda
```

恢复参数必须与首次运行一致。累计的是有效训练秒数，不包含停机时间。

## 结果

```text
<job-dir>/
├─ manifest.json
├─ state.json
├─ events.jsonl
├─ stdout.log
├─ stderr.log
├─ telemetry/
├─ participants/<method_seed>/
│  ├─ config.json
│  ├─ training/metrics.jsonl
│  ├─ training/checkpoint_latest.pt
│  └─ snapshots/
├─ evaluations/
└─ checksums.sha256
```

正式验收必须检查：

- 6 个参与组均有 Checkpoint；
- 各组完成更新数和环境决策数接近；
- 无外部 GPU PID 介入；
- 训练全程无 NaN/Inf；
- 3 个种子均完成固定牌墙与四座位评估；
- 失败、停止和恢复记录均保留。
