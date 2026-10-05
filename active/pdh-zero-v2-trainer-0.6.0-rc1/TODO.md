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

## F5 Recurrent PPO 与训练恢复 — candidate_complete
WHERE：`src/pdh_v2/training/`、`tools/jobs/`
HOW：公开历史 Critic 基线、PPO、完整 Checkpoint、任务锁和总预算。
EXPECTED_RESULT：10 分钟端到端与 30–60 分钟学习短试通过后，才设置 `training_ready=true`。

## F6 新规则基线与方法实验 — not_started
WHERE：`experiments/`、`src/pdh_v2/evaluation/`
HOW：攻防指标、逐局配对、多路线表示、风险辅助和特权 Critic 逐项对照。
EXPECTED_RESULT：多种子、公平样本预算、完整不确定性和可审计牌谱。


## F5.1 目标 GPU Smoke — completed
WHERE：目标 RTX 5090 实例的独立 evidence 目录
HOW：运行 10 分钟 CUDA Smoke，检查非零终局、吞吐、数值、牌谱和检查点。
EXPECTED_RESULT：10分钟CUDA Smoke已完成；旧长任务工程稳定但学习信号稀疏，证据已归档。

## F5.2 正式恢复与多进程吞吐 — not_started
WHERE：`src/pdh_v2/rollout/`、`src/pdh_v2/training/`
HOW：保存进行中会话与座位记忆；在参考语义差分门禁下引入多进程环境。
EXPECTED_RESULT：可无损恢复完整批次；新规则有效采样吞吐达到实测目标。

## F5.3 旧长训练任务控制 — superseded_for_new_runs
WHERE：`src/pdh_v2/training/`、`src/pdh_v2/rollout/`、`tools/long_job.py`
HOW：Smoke 晋级门禁、更新边界完整恢复、任务查重、安全停止、牌谱分片、周期评估。
EXPECTED_RESULT：恢复、查重和安全停止基础已验证；0.6.0-rc1禁止创建新长任务，等待课程门禁后重设预算。

## F5.4 稀疏奖励诊断与课程学习 — candidate_complete
WHERE：`src/pdh_v2/{strategy,policy,observation,training}/`、`tools/train_curriculum.py`
HOW：显式多路线特征、校准隔离的风险辅助、按手局分组验证的强行为克隆、1至4学习座位混合对手课程。
EXPECTED_RESULT：目标 GPU 调优末段混合对手流局率显著低于旧纯自博弈，并产生可审计攻防指标。

## F5.5 目标 GPU 课程调优 — pending
WHERE：目标 RTX 5090 的 `/root/rivermind-data/pdh-zero/evidence/v2-curriculum/`
HOW：运行60分钟课程任务及 `tools/curriculum_gate.py`，保存遥测、评估、牌谱和检查点。
EXPECTED_RESULT：课程门禁通过后才生成新长训练包；失败则依据具体门禁原因迭代。

## F5.6 新长训练基线 — blocked_by_F5.5
WHERE：后续版本 `tools/jobs/` 与独立长任务目录
HOW：按课程调优实测吞吐与效果设定预算，保留混合对手锚点和完整恢复。
EXPECTED_RESULT：不恢复0.5.0-rc3旧配置；多种子长任务在新门禁通过后启动。
