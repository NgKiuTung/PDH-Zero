
# P1-02 智川云 RTX 5090 执行手册

## 1. 创建实例

选择：

```text
1 × RTX 5090 32GB
14 vCPU / 60GB RAM
30GB 系统盘
50GB 数据盘起步
PyTorch 2.12.1 / CUDA 13.0 / Python 3.12 / Ubuntu 22.04
按量计费
```

不要选择 PyTorch 2.1 / CUDA 12.1。A00 将检查 `torch.version.cuda >= 12.8`、设备能力 `12.0` 和 `torch.cuda.get_arch_list()` 中的 `sm_120`。

## 2. 数据放置

```text
/root/rivermind-data/pdh-zero/   在线训练、Checkpoint、证据
/root/rivermind-fs/pdh-zero/     可选长期备份
/                                仅镜像和系统环境
```

数据盘随实例释放而删除；释放实例前必须把关键证据复制到文件存储或下载到本地。

## 3. 安装

把一体化 Bundle 上传到 `/root/rivermind-data/`，然后：

```bash
cd /root/rivermind-data
mkdir -p pdh-zero-installer-v1.1.0
python3 -m zipfile -e pdh-zero-rivermind-rtx5090-bootstrap-v1.1.0.zip pdh-zero-installer-v1.1.0
cd pdh-zero-installer-v1.1.0
sha256sum -c pdh-zero-rivermind-install-v1.1.0.sh.sha256
sha256sum -c pdh-zero-phase1-v1.1.0.zip.sha256
bash pdh-zero-rivermind-install-v1.1.0.sh
```

安装器拒绝覆盖已有活动版本，不调用 pip，也不修改系统 Torch。

## 4. 每次开机初始化

```bash
source /root/rivermind-data/pdh-zero/env.sh
cd "$PDH_PHASE1_ROOT"
bash scripts/00_bootstrap.sh
source .venv/bin/activate
```

## 5. Gate A 顺序

严格逐项执行：

```text
A00 CUDA/Blackwell 预检
A01 系统与存储盘点
A02 FP32/BF16 前后向
A03 CPU—GPU FP32 精度
A04 Checkpoint 连续恢复
A05 关机/开机数据盘持久性
A06 5 分钟后再做 60 分钟稳定性浸泡
```

A00 命令：

```bash
python scripts/run_step.py --gate A --experiment-id A00_cuda_blackwell_preflight   --results-root "$PDH_RESULTS_ROOT/gate-a/runs" --timeout-seconds 600 --   python scripts/00_cuda_blackwell_preflight.py
```

A00 未通过时停止计费实例并更换镜像，不安装其他 Torch 版本。

## 6. 磁盘策略

50GB 可完成第一阶段，但必须遵守：

- 不保存所有原始 Transition；
- 只保留最新、最佳和里程碑 Checkpoint；
- 每个实验完成后打包并下载；
- `cache/` 和 `tmp/` 可清理；
- 正式多种子训练前，建议扩至 100GB 或启用文件存储备份。

## 7. 计费策略

按量实例只在开机后运行实验。命令、配置和检查表先在本地准备；实验结束执行 `sync`，确认结果后立即关机。
