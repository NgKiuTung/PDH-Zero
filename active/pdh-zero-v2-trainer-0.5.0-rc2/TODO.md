# v2 实施进度

## F0 房规冻结 — completed
WHERE：`rules/`、`src/pdh_v2/resources/`
HOW：规则源推进到 v0.9，130 条 T 案例可追踪，`open_decisions.items=[]`。
EXPECTED_RESULT：正式规则没有静默未知分支，来源与配置哈希一致。

## F1 规则、结算与恢复基础 — completed
WHERE：`src/pdh_v2/{dealer,horses,responsibility,settlement,ledger,journal}.py`
HOW：实现庄位、三类马源、多响、流局、十二张落地、直接点杠杠爆和幂等账本。
EXPECTED_RESULT：普通及专项账单零和、可恢复、不重抽、不重付。

## F2 完整 136 张参考引擎 — completed
WHERE：`src/pdh_v2/game/`、`src/pdh_v2/scoring/`、`src/pdh_v2/replay/`
HOW：实现完整动作响应、胡牌解释、责任优先级、会话结算和确定性单手牌谱。
EXPECTED_RESULT：随机合法整局牌实体守恒，无鸡胡，固定输入可重放。

## F3 合法观察与全局牌谱 — candidate_complete
WHERE：`src/pdh_v2/observation/`、`src/pdh_v2/replay/`
HOW：公开／私有投影、跨局亮马留手、全局事件分片和信息泄漏差分测试。
EXPECTED_RESULT：每局可重放，Actor只见合法信息，隐藏牌置换不改变相同信息集观察。

## F4 批量采样与最小策略 — candidate_complete
WHERE：`src/pdh_v2/rollout/`、`src/pdh_v2/policy/`
HOW：多桌环境、批量推理、四座位独立记忆、同步固定版本采样。
EXPECTED_RESULT：单桌与批量语义一致，无串桌，形成真实吞吐报告。

## F5 Recurrent PPO 与训练恢复 — in_progress
WHERE：`src/pdh_v2/training/`、`tools/jobs/`
HOW：公开历史 Critic 基线、PPO、完整 Checkpoint、任务锁和总预算。
EXPECTED_RESULT：10 分钟端到端与 30–60 分钟学习短试通过后，才设置 `training_ready=true`。

## F6 新规则基线与方法实验 — not_started
WHERE：`experiments/`、`src/pdh_v2/evaluation/`
HOW：攻防指标、逐局配对、多路线表示、风险辅助和特权 Critic 逐项对照。
EXPECTED_RESULT：多种子、公平样本预算、完整不确定性和可审计牌谱。


## F5.1 目标 GPU Smoke — pending
WHERE：目标 RTX 5090 实例的独立 evidence 目录
HOW：运行 10 分钟 CUDA Smoke，检查非零终局、吞吐、数值、牌谱和检查点。
EXPECTED_RESULT：训练任务成功完成；无非法动作/非零和/NaN；得到真实容量报告。

## F5.2 正式恢复与多进程吞吐 — not_started
WHERE：`src/pdh_v2/rollout/`、`src/pdh_v2/training/`
HOW：保存进行中会话与座位记忆；在参考语义差分门禁下引入多进程环境。
EXPECTED_RESULT：可无损恢复完整批次；新规则有效采样吞吐达到实测目标。

## F5.3 长训练任务控制 — candidate_complete
WHERE：`src/pdh_v2/training/`、`src/pdh_v2/rollout/`、`tools/long_job.py`
HOW：Smoke 晋级门禁、更新边界完整恢复、任务查重、安全停止、牌谱分片、周期评估。
EXPECTED_RESULT：Smoke 通过后可直接启动默认48小时／5000万有效决策长任务；中断后从完整更新边界恢复，孤儿分片隔离且不计入正式样本。
