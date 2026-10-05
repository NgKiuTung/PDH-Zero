
# PDH-Zero Phase 1 工程约束

## 目标

本仓库只负责第一阶段：云环境验收、Mini-PDH 最小四人自博弈闭环、端到端吞吐与第二阶段成本预测。不得在本阶段加入 Full-PDH、RGRC、LACC、在线搜索或商业产品功能。

## 开工前读取

1. `README.md`
2. `docs/P1-00_phase1_scope.md`
3. `docs/P1-01_experiment_protocol.md`
4. `docs/P1-02_rivermind_rtx5090_runbook.md`
5. `docs/P1-03_acceptance_criteria.md`
6. `docs/P1-08_mini_pdh_rules.md`
7. `docs/P1-11_stepwise_execution_checklist.md`

## 强制规则

- 任何“通过”“兼容”“稳定”声明必须对应当前代码、配置和环境产生的日志或测试证据。
- 禁止安装、卸载或升级镜像自带 PyTorch/CUDA/ROCm；不兼容时更换镜像。
- RTX 50 系列必须先通过 `00_cuda_blackwell_preflight.py`，确认 CUDA ≥ 12.8 与 `sm_120` Kernel。
- 代码、结果、缓存和 Checkpoint 必须落在数据盘；系统盘不得作为长期证据存储。
- 所有实验写入新目录，不得覆盖旧结果；失败运行同样属于证据。
- 所有长期命令必须有超时、步数或时长上限，并支持 Checkpoint。
- Actor 只能读取自身手牌与公开信息；Privileged Critic 可在训练时读取完整状态。
- Mini-PDH v0 只能证明工程与算法链路可行，不能替代完整纯大胡麻将结果。
- 日志不得包含 Token、账号、环境变量全量转储、GPU UUID 或硬件序列号。
- 临时产物只放 `tmp/<用途>-<YYYYMMDD>/`，结束前清算。

## 可复用工具索引

| 工具 | 路径 | 作用 | 调用方式 |
|---|---|---|---|
| 环境初始化 | `scripts/00_bootstrap.sh` | 复用预装 Torch，建立后端激活层和数据盘缓存路径 | `bash scripts/00_bootstrap.sh` |
| RTX 5090 预检 | `scripts/00_cuda_blackwell_preflight.py` | 验证 CUDA、sm_120、BF16 和基本矩阵计算 | `python scripts/00_cuda_blackwell_preflight.py` |
| 实验步骤执行器 | `scripts/run_step.py` | 新建证据目录、超时执行、记录日志与 Manifest | `python scripts/run_step.py --help` |
| 系统盘点 | `scripts/01_collect_system_info.sh` | 采集脱敏硬件、驱动、Torch 与存储信息 | `bash scripts/01_collect_system_info.sh <out>` |
| 存储哨兵 | `scripts/02_check_storage.sh` | 验证关机、开机后的数据盘持久性 | `bash scripts/02_check_storage.sh create|verify <dir>` |
| Torch 冒烟 | `scripts/03_torch_backend_smoke.py` | 验证前后向和 AdamW | `python scripts/03_torch_backend_smoke.py --help` |
| 精度对照 | `scripts/04_precision_test.py` | 对照 CPU 与加速器 FP32 误差 | `python scripts/04_precision_test.py --help` |
| Checkpoint | `scripts/05_checkpoint_test.py` | 验证模型、优化器与 RNG 连续恢复 | `python scripts/05_checkpoint_test.py --help` |
| 稳定性浸泡 | `scripts/06_stability_soak.py` | 有界持续训练和显存监控 | `python scripts/06_stability_soak.py --help` |
| 环境基准 | `scripts/07_engine_benchmark.py` | 测量 Mini-PDH steps/s 与 games/s | `python scripts/07_engine_benchmark.py --help` |
| 策略基准 | `scripts/08_policy_benchmark.py` | 扫描策略 Batch 吞吐 | `python scripts/08_policy_benchmark.py --help` |
| PPO 训练 | `scripts/09_train_smoke.py` | 运行可恢复 PPO | `python scripts/09_train_smoke.py --help` |
| 对战评估 | `scripts/10_evaluate.py` | 固定牌墙与四座位轮换评估 | `python scripts/10_evaluate.py --help` |
| 信息隔离 | `scripts/11_check_hidden_info_isolation.py` | 验证隐藏牌不影响 Actor | `python scripts/11_check_hidden_info_isolation.py --help` |
| 结果打包 | `scripts/12_pack_results.sh` | 生成 ZIP 与 SHA256 | `bash scripts/12_pack_results.sh <run-dir>` |
| 长时 Campaign | `scripts/13_long_gate_b_campaign.py` | 平衡训练普通/特权 Critic，多种子恢复与评估 | 由 `scripts/14_long_job.py` 启动 |
| 长任务控制 | `scripts/14_long_job.py` | 分离会话启动、状态查询、安全停止与恢复 | `python scripts/14_long_job.py --help` |
