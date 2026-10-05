# PDH-Zero v2 0.6.0-rc3维护入口

先读README、docs/refinement-contract.md、docs/diagnosis-rc3.md和TODO。
房规唯一来源rules/source-v0.9.md；禁止改规则、旧实验或Checkpoint来使门禁通过。

训练唯一推荐入口tools/train_refinement.py；只导入教师warmup，不续训失败latest。
逐评估候选必须先永久保存，再评估；区分selected_development与qualified。
不自动DUO、不自动长训、不承诺精确resume。独立验证入口tools/evaluate_candidates.py，
禁止看过验证结果后再以同一批牌墙选择模型。

更改遵循Coding Rule，覆盖旧log概率重算、风险梯度隔离、KL回滚、共享缓冲所有权、
四座位分母与支付账本。私有牌谱真值不能流入Actor。
进程须登记PID，只清理本任务子进程。正式脚本放tools，临时分析写tmp。

统一CPU验收python3 tools/verify_refinement.py；实际GPU和学习提升须目标数据证实。
源码变更后重新生成FILES.sha256与ZIP，不将本地夹具称为用户权重或GPU实测。
