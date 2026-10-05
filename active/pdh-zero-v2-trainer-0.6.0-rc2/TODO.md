# 0.6.0-rc2 受控策略修订

## R1 证据与复现 — completed
WHERE: `tmp/refinement-20261003/`, `verification/rc2/`
HOW: 核对rc1源码、停止日志、风险梯度路径、指标分母和更新10教师复习；保存可复核脚本。
消费方: 研究负责人、代码审核者。
EXPECTED_RESULT: 区分实现事实与训练退化的未证实因果，不改房规或历史实验。

## R2 最小稳定修订 — completed
WHERE: `src/pdh_v2/training/`, `src/pdh_v2/policy/`
HOW: 复用warmup权重；冻结教师分布锚；降低步长、KL限幅回滚；风险诊断隔离；修复统计分母。
消费方: Actor、PPO、评估器。
EXPECTED_RESULT: 同策略重算一致、风险监督不改变Actor、异常更新不提交、三家启发式对照不自动升级。

## R3 采样与成对评估 — completed
WHERE: `src/pdh_v2/rollout/`, `src/pdh_v2/training/`
HOW: 缓存重复观察；spawn CPU worker；同墙四座位逐局评估与分层bootstrap；保存完整结算证据。
消费方: GPU推理进程、牌谱审计者。
EXPECTED_RESULT: 串并行固定动作严格等价、无跨桌数据串流、进程可回收、评估差值可重算。

## R4 可执行调优与退出 — completed
WHERE: `tools/train_refinement.py`, `src/pdh_v2/training/refinement*.py`
HOW: 预算包含初评/训练/复评；正确处理中断；只保留经评估的best；显式进度、进程锁和父子PID；禁止自动长训。
消费方: 云实例执行者。
EXPECTED_RESULT: 一条命令复用现有warmup，不重跑2048手；退出无残留、不覆盖旧目录，输出明确门禁。

## R5 回归及最终包 — completed
WHERE: `tests/`, `verification/rc2/`, `README.md`, `CHANGELOG.md`, `FILES.sha256`
HOW: 红例/回归/串并行/冷启动/中断/解压验收；仅报告实际运行结果；检查目标GPU未运行的边界。
消费方: 实验执行者和下一维护者。
EXPECTED_RESULT: 可上传ZIP、校验文件、日志、完整使用说明；CPU验收通过，模型优越性仍由实机同条件评估裁决。

## 交付边界

本地CPU修订与发布验收完成；目标RTX 5090实测、实际warmup文件导入、学习收益与长期训练仍未验收。
不把候选包验收写成模型已经优于旧版本；本CLI不提供自动长训或精确resume。
