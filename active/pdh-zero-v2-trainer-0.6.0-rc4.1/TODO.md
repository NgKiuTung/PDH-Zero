# rc4范围与验收状态

- [completed] WHERE: native/, src/pdh_v2/native; HOW: 批量连续数据C++核及Python差分；EXPECTED_RESULT: 编译/导入/真实64手与专用动作边界测试通过。
- [completed] WHERE: replay/supply.py, training/responsibility_metrics.py, tools/audit_responsibility.py; HOW: 原账本逐笔回算、成功面子变更关联；EXPECTED_RESULT: 20480手金额守恒，100条尾部牌谱校验，不改变奖励。
- [completed] WHERE: runtime/, tools/prepare_rc4.py, tools/benchmark_runtime.py; HOW: 配额解析和单组多负载顺序测量，带语义/资源/哈希计划；EXPECTED_RESULT: 14配额不会当128，目标机选型入口与本地完整准备测试通过。
- [completed] WHERE: training/selection_input.py, refinement_*.py; HOW: 有哈希候选导入并新建优化器/锚；EXPECTED_RESULT: 实际d265权重短流程及停止回收通过，状态不冒充独立确认。
- [completed] WHERE: tests/, verification/rc4/; HOW: 全套CPU回归、静态审计、分支覆盖与真实数据验证；EXPECTED_RESULT: 338项通过，源规则逐字不改，目标GPU待实测明确记录。

交付门禁还包含最终ZIP重新解压、目标ABI本地编译与同一验收入口；最终证据见随ZIP交付的release-check.json。

未在本版承诺：完整C++游戏引擎、反事实贡献估计、精确断电resume、DUO晋级、正式长训资格。目标机程序/模型实测前不标记这些完成。
