# P1-03 门禁与验收标准

## Gate A

| 验收项 | PASS | FAIL | 证据 |
|---|---|---|---|
| RTX 5090 架构 | CUDA ≥12.8、能力12.0、`sm_120` Kernel、BF16 可用 | 任一条件缺失 | `A00 cuda_blackwell_preflight.json` |
| NVIDIA 后端 | CUDA 可用、设备名与显存可读 | CUDA 不可用 | `A01 artifacts/software.json` |
| FP32 | 前后向和 AdamW 完成 | 崩溃、NaN、未支持算子 | `torch_smoke_float32.json` |
| BF16 | 可完成或明确报告不支持 | 静默回退或错误报告成功 | `torch_smoke_bfloat16.json` |
| Checkpoint | 恢复后下一步参数和损失一致 | 仅能加载权重，训练状态不连续 | `checkpoint_test.json` |
| 持久性 | 重启后哨兵哈希一致 | 文件丢失或被修改 | A05 `sha256sum -c` 输出 |
| 稳定性 | 60 分钟完成，无 NaN/持续泄漏 | HIP/CUDA 崩溃、内存持续增长 | `soak_summary.json` 与 `soak_metrics.jsonl` |

Gate A 出现任一 P0 阻断项时，不进入 Gate B：

- GPU 设备不可用或 A00 Blackwell 预检失败；
- Checkpoint 无法恢复；
- 训练出现不可复现崩溃；
- 结果无法从实例导出；
- 预装 Torch 被覆盖且不能恢复。

## Gate B

| 验收项 | 标准 |
|---|---|
| 非法动作 | 0 |
| 牌数守恒错误 | 0 |
| 四家净分和 | 每局严格为 0 |
| 确定性回放 | 相同种子和动作序列的最终哈希一致 |
| Actor 信息隔离 | 隐藏牌置换后输入和输出误差为 0 或数值容差内 |
| PPO 数值稳定 | 无 NaN/Inf，梯度范数有界 |
| 学习趋势 | 至少 2/3 种子相对 Random Legal 改善 |
| Checkpoint 恢复 | 指标和训练步数连续 |
| 跨后端运行 | 同一模型定义可在 CPU、CUDA、ROCm 加载 |

“PPO 胜过随机”必须在固定牌墙、风位轮换的自然开局评估中成立，不能使用训练局内即时成绩代替。

## Gate C

| 端到端吞吐 | 决策 |
|---:|---|
| ≥ 30,000 decisions/s | 可进入 Full-PDH，保留当前资源方案 |
| 20,000-30,000 | 可进入，但缩减实验矩阵 |
| 10,000-20,000 | 优先优化批处理、环境或 Rust 引擎 |
| < 10,000 | 暂停正式大规模训练 |

Mini-PDH 的吞吐不能直接当作 Full-PDH 吞吐。进入第二阶段后还需对 Full-PDH 再做一次相同门禁。

## 最终判定矩阵

| Gate A | Gate B | Gate C | 第一阶段结论 |
|---|---|---|---|
| PASS | PASS | PASS | PASS |
| PASS | PASS | 风险区间 | PASS_WITH_RISKS |
| PASS | FAIL | 任意 | FAIL |
| FAIL | 未执行 | 未执行 | FAIL |
| 证据不完整 | 任意 | 任意 | UNVERIFIED |
