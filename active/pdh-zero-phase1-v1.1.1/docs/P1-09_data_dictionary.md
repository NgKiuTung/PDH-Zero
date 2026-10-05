# P1-09 数据与日志字典

## 1. Actor 观察（159 维）

Actor 仅包含：

- 自己暗牌计数；
- 自己公开碰牌；
- 按相对座位排列的四家公开弃牌；
- 按相对座位排列的四家公开碰牌；
- 当前决策阶段；
- 相对庄家位置；
- 最近公开弃牌及其来源；
- 剩余牌墙数量；
- 当前决策数占上限的比例。

禁止包含：

- 其他玩家暗牌；
- 未知牌墙牌型或顺序；
- 对手未来动作；
- 终局标签。

## 2. Privileged Critic 观察（224 维）

在 Actor 159 维基础上，训练 Critic 可额外包含：

- 按相对座位排列的四家暗牌计数，共 52 维；
- 未知牌墙牌型计数，共 13 维。

Actor 与 Critic 使用独立编码器和 GRU。隐藏信息隔离实验必须证明：只修改对手暗牌和牌墙顺序时，Actor 输入、合法动作和 Actor 输出保持不变，Privileged Critic 输入发生变化。

## 3. `manifest.json`

| 字段 | 含义 |
|---|---|
| `schema_version` | 清单版本 |
| `experiment_id` | 实验编号 |
| `gate` | `A/B/C/LOCAL` |
| `status` | `running/success/failed/timeout` |
| `start_time_utc` | UTC 开始时间 |
| `end_time_utc` | UTC 结束时间 |
| `command` | 未经 Shell 拼接的命令数组 |
| `exit_code` | 子进程退出码 |
| `timed_out` | 是否由超时终止 |
| `git_commit` | 当前 Git 提交；非 Git 包时为 null |
| `python_version` | Python 版本 |
| `torch_version` | Torch 版本 |
| `backend` | `cpu/cuda/rocm` |
| `run_directory` | 本次不可覆盖结果目录 |

## 4. PPO `metrics.jsonl`

每行对应一次训练更新：

```text
timestamp
update
episodes
attempts
environment_decisions
total_environment_decisions
mean_score
win_rate
draw_rate
collection_seconds
update_seconds
policy_loss
value_loss
entropy
approximate_kl
gradient_norm
evaluation（达到评估周期时）
```

`mean_score` 是未裁剪的真实 Mini-PDH 净分；Critic 内部训练值使用固定 `reward_normalizer` 缩放。

## 5. 状态定义

- `success`：外层命令退出码为 0；仍需检查业务验收条件；
- `failed`：命令非零或内部断言失败；
- `timeout`：达到有界时长后由执行器终止；
- `UNVERIFIED`：尚未在目标云环境执行，不能由 CPU 结果替代。

外层 `success` 仅证明命令执行成功，不自动等于 Gate `PASS`。
