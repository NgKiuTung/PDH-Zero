
# P1-12 从魔搭 AMD 迁移到智川云 RTX 5090

旧 AMD 结果可以作为历史工程证据，但不能证明新 RTX 5090 环境通过。硬件、Torch、CUDA、存储生命周期和吞吐均发生变化，因此新实例必须重新执行 Gate A、Gate B 和 Gate C。

无需重做：Mini-PDH 规则定义、代码级单元测试设计、A04 Checkpoint 修复、研究假设和验收口径。

必须重做：A00-A06、Gate B 三种子、Gate C 共置吞吐，以及所有正式论文训练。

旧结果存放：`evidence/legacy-modelscope-amd/`。新结果只能进入 `evidence/gate-a/runs/`、`gate-b/runs/`、`gate-c/runs/`。
