# v2 实施进度

## F0 规则追踪与边界 — completed
WHERE：`rules/`、`docs/implementation-scope.md`、`src/pdh_v2/resources/`
HOW：规则源推进到 v0.8；115条T案例可追踪；未决Q项显式登记。
EXPECTED_RESULT：S10/S11确认项不再被错误列为未决；未知房规没有默认值。

## F1 已确认规则基础 — completed
WHERE：`src/pdh_v2/`、`tests/`
HOW：136张基础、胡形候选、庄位/抓马、门清/海底、单赢家及多响纯算术、十二张落地计数、流局记录。
EXPECTED_RESULT：纯函数可独立验收；不把低层算术冒充完整规则引擎。

## F2 结算与恢复 — in_progress
WHERE：`src/pdh_v2/ledger.py`、`src/pdh_v2/journal.py`、后续多响/承包账本
HOW：单赢家和流局事件已可幂等恢复；多响统一马牌与第四碰责任已接入结算/恢复；杠爆和完整会话状态仍待后续。
EXPECTED_RESULT：所有结算类型均可中断恢复且不重抽、不重付。

## F3 完整136张引擎 — blocked_by_rule_decisions
WHERE：`src/pdh_v2/game/`、`src/pdh_v2/scoring/`
HOW：关闭Q项后实现完整动作/响应/胡牌/计分状态机。
EXPECTED_RESULT：随机合法整局守恒、无鸡胡、正确多响/承包/连庄。

## F4 牌谱、并行采样、模型与PPO — not_started
WHERE：`src/pdh_v2/replay/`、`observation/`、`rollout/`、`policy/`
HOW：按批准蓝图在完整引擎后推进。
EXPECTED_RESULT：可重放、无信息泄漏、多桌批推理、四家轨迹和可恢复训练。

## F2.1 多响抓马与十二张承包责任 — completed
WHERE：`src/pdh_v2/horses.py`、`ledger.py`、`journal.py`、`responsibility.py`、`settlement.py`、`tests/`
HOW：多响旧墙不足改用专门开马并重洗；第四碰锁定当局自摸承包者；吃胡走普通点炮结算；补齐恢复与规则追踪。
EXPECTED_RESULT：多响不借任何赢家下局手牌；十二张承包仅第四碰触发、仅同局自摸生效；测试、快照、哈希、文档一致。
