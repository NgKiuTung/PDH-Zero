# 0.6.0-rc3 执行跟踪

## R1 可追溯候选与评估 — completed
WHERE: `src/pdh_v2/training/candidate_store.py`, `paired_*.py`, `loss_attribution.py`, `tests/`
HOW: 每个评估前永久保存权重；区分名义候选、通过门禁的候选；按牌墙组比较CVaR；逐手拆解损失并对账。
消费者: 评估程序、研究负责人。
EXPECTED_RESULT: INCONCLUSIVE不会丢权重；错误/中断不覆盖候选；所有支付来源可核对。

## R2 受控更新与早停 — completed
WHERE: `src/pdh_v2/training/stable_ppo.py`, `refinement_*.py`, `tools/train_refinement.py`
HOW: 完整批次测量anchor KL；有界自适应锚系数；后验KL越界回滚；两次无改善止步并独立导出最佳开发候选。
消费者: Actor、PPO与执行者。
EXPECTED_RESULT: 不改真实净分、不接回未校准风险头；选择最佳不伪装成学习提升或精确续训。

## R3 传输与独立验证 — completed
WHERE: `src/pdh_v2/rollout/`, `tools/evaluate_candidates.py`, `tools/benchmark_refinement.py`
HOW: 共享固定float32缓冲减少Pipe对象传输；保持同步语义与差分测试；独立新牌墙评估已冻结候选；组件计时。
消费者: 采样器、推理服务和研究验证者。
EXPECTED_RESULT: 串并行行为等价、原始牌谱完整、Worker无残留；报告实测加速而不保证GPU占用。

## R4 最终验收与交付 — completed（本地代码；目标GPU待验）
WHERE: `tests/`, `verification/rc3/`, `README.md`, `docs/refinement-contract.md`, `FILES.sha256`
HOW: CPU全回归、负向/中断/篡改用例、CLI小模型端到端、候选导出、干净ZIP重验；保留冻结房规逐字不变。
消费者: 云实例执行者与维护者。
EXPECTED_RESULT: 独立可上传ZIP及校验；旧实验无修改；未获真实权重和CUDA实测不写成完成。
