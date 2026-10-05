
# PDH-Zero 第一阶段便携实验包

版本：`1.1.0`  
发布日期：`2026-09-29`

本版本面向单卡 NVIDIA RTX 5090 云实例，同时保留 CPU、CUDA 与 ROCm 路径。第一阶段只验证环境、Mini-PDH 自博弈闭环和吞吐，不直接声称完整纯大胡麻将算法有效。

## 推荐智川云规格

```text
GPU：1 × RTX 5090 32GB
CPU：14 vCPU
内存：60GB
系统盘：30GB
数据盘：50GB 起步，最终实验推荐 100GB 或配置外部备份
镜像：PyTorch 2.12.1 + CUDA 13.0 + Python 3.12 + Ubuntu 22.04
```

禁止选择 `PyTorch 2.1.x + CUDA 12.1`。RTX 5090 的计算能力为 `sm_120`，要求 CUDA 12.8 或更新版本；本包的 A00 会在计费实验前检查 CUDA 运行时与 `sm_120` Kernel 是否同时存在。

一张 GPU 已足够。当前模型规模小，正式自博弈更可能受 CPU 环境 Worker 吞吐限制；增加第二张 GPU 不能直接解决该瓶颈。

## 存储边界

智川云默认路径：

```text
系统盘：/
数据盘：/root/rivermind-data/
文件存储：/root/rivermind-fs/（若已开通）
```

代码、Checkpoint、实验结果和缓存统一放数据盘；系统盘只承载镜像环境。文件存储适合备份发布包和关键证据，不作为在线训练目录。

## 首次安装

使用交付的 `pdh-zero-rivermind-rtx5090-bootstrap-v1.1.0.zip`。详细步骤见：

```text
docs/P1-02_rivermind_rtx5090_runbook.md
```

安装器不会安装、升级或卸载 PyTorch，也不会覆盖已有研究目录。

## 三个门禁

| 门禁 | 目标 | 通过后 |
|---|---|---|
| Gate A | RTX 5090/CUDA、数值、Checkpoint、存储与稳定性 | 运行 Mini-PDH |
| Gate B | 环境不变量、信息隔离、PPO 学习闭环 | 正式测吞吐 |
| Gate C | CPU/GPU 共置吞吐和成本外推 | 进入 Full-PDH 与 RGRC |

## 快速自检

```bash
source /root/rivermind-data/pdh-zero/env.sh
cd "$PDH_PHASE1_ROOT"
bash scripts/00_bootstrap.sh
source .venv/bin/activate
python -m compileall -q src scripts tests
python -m unittest discover -s tests -v
```

## 预装 PyTorch 保护

禁止执行：

```bash
pip install torch
pip install --upgrade torch
pip uninstall torch
```

`00_bootstrap.sh` 只建立本项目的激活层，复用云镜像内置 Torch。若 A00 判定镜像不支持 `sm_120`，应更换镜像，而不是在计费实例内反复修补环境。

## 结果目录

安装器创建：

```text
/root/rivermind-data/pdh-zero/
├─ active/
├─ archive/
├─ releases/
├─ evidence/gate-a/runs/
├─ evidence/gate-b/runs/
├─ evidence/gate-c/runs/
├─ state/persistence/
├─ cache/
└─ tmp/
```

所有实验通过 `scripts/run_step.py` 写入不可覆盖的新目录，并由 `scripts/12_pack_results.sh` 生成 ZIP 和 SHA256。
