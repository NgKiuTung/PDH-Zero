# PDH-Zero v2 0.6.0-rc2维护入口

先读README、docs/refinement-contract.md、docs/diagnosis-rc1.md和TODO。
冻结房规唯一来源仍为rules/source-v0.9.md，禁止用调优修改规则或旧实验含义。

当前唯一推荐训练入口为tools/train_refinement.py。复用用户warmup，不自动续训latest；
不自动升级DUO、不自动长训、不声称有精确resume。其他入口保留历史回归，不用于本轮。

变更必须遵守Coding Rule，补充复现测试及证据。检查指标分母、策略版本、风险监督
梯度与公开信息边界；不要让隐私审计真值进入Actor。进程须有明确父子PID并回收。

全套CPU门禁：python3 tools/verify_refinement.py。
源码变动后先跑测试和基础静态审计，再重新生成FILES.sha256及ZIP校验。
实际GPU性能、学习提升、多种子统计由真实实验提供，不能用CPU测试数量替代。
