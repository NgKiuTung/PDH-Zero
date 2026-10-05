# P1-07 第一阶段决策报告模板

## 1. 结论

```text
Verdict: PASS / PASS_WITH_RISKS / FAIL / UNVERIFIED
```

## 2. 需求—证据矩阵

| Requirement | 实现 | 证据 | 状态 |
|---|---|---|---|
| AMD ROCm 可训练 | A02 脚本 | 结果文件 | VERIFIED / UNVERIFIED |
| NVIDIA CUDA 可训练 | A02 脚本 | 结果文件 | ... |
| Checkpoint 可恢复 | A04 | 结果文件 | ... |
| Mini-PDH 不变量成立 | B01 | 测试日志 | ... |
| PPO 存在学习趋势 | B03/B04 | 评估报告 | ... |
| Actor 无隐藏信息泄漏 | B05 | isolation summary | ... |
| 资源足以进入二阶段 | C01-C03 | throughput report | ... |

## 3. Gate A

- 已验证设备：
- 后端版本：
- 稳定性：
- 持久性：
- 风险：

## 4. Gate B

- 规则测试：
- 回放：
- 普通 Critic：
- Privileged Critic：
- 三种子一致性：
- 信息隔离：

## 5. Gate C

- 纯环境 decisions/s：
- 最佳推理 Batch：
- 端到端 decisions/s：
- CPU 利用率：
- GPU 利用率：
- Full-PDH 单次训练估算：

## 6. 第二阶段决策

```text
[ ] 进入 Full-PDH
[ ] 先实现 Rust 高速引擎
[ ] 增加 CPU 核数
[ ] 追加 GPU 小时
[ ] 暂停并修复训练稳定性
```

## 7. 未验证事项

- 

## 8. 可复现附件

- 结果包：
- SHA256：
- 代码版本：
