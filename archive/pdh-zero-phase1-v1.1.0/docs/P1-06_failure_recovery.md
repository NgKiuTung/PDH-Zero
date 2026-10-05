# P1-06 故障与恢复

## 1. 立即停止的情况

- HIP/CUDA illegal memory access；
- 进程反复无响应；
- 损失出现 NaN/Inf；
- 显存或主机内存持续单调增长；
- Checkpoint 写入失败；
- 牌数守恒、非法动作或净分和断言失败；
- 结果目录被覆盖；
- 预装 PyTorch 版本被改变；
- 额度消耗明显超过预设上限。

## 2. 禁止的恢复方式

- 无限重试；
- 删除失败日志后重跑；
- 更换随机种子掩盖失败；
- 降级到 CPU 却仍报告 GPU 通过；
- 捕获异常后写入 `success`；
- 使用旧 Checkpoint 冒充当前运行结果；
- 批量终止系统中的全部 Python 进程；
- 覆盖原结果目录。

## 3. 标准恢复流程

1. 记录实验编号和失败时间；
2. 保留 `stdout.log`、`stderr.log`、`manifest.json` 和最后一个 Checkpoint；
3. 执行 `scripts/12_pack_results.sh`；该工具不依赖系统 `zip` 命令；
4. 手动停止 GPU 实例，防止继续消耗；
5. 在 CPU 环境构造最小复现；
6. 修复后创建新实验编号，不覆盖失败目录；
7. 先运行 Smoke，再恢复完整实验。

## 4. Checkpoint 恢复

训练 Checkpoint 包含：

- Actor/Critic 参数；
- Optimizer 状态；
- 下一更新编号和累计环境决策数；
- Python RNG；
- Torch CPU RNG；
- CUDA/ROCm RNG（可用时）；
- 冻结对手模型；
- 训练配置和规则哈希。

恢复示例：

```bash
python scripts/09_train_smoke.py \
  --config configs/ppo_phase1.yaml \
  --output results/manual_recovery \
  --resume results/<run>/artifacts/training/checkpoint_latest.pt \
  --updates 5
```

恢复后的 `update` 必须从原 Checkpoint 的 `next_update` 延续。重新从 0 开始不得称为恢复成功。

## 5. 平台中断

若实例被自动关闭：

- 先验证项目与结果目录是否仍存在；
- 对照 `checksums.sha256`；
- 只从最后一个校验通过的 Checkpoint 恢复；
- 若系统盘丢失，从本地、数据集挂载或 OSS/NAS 备份恢复；
- 丢失区间必须记录，不能补写不存在的日志。
