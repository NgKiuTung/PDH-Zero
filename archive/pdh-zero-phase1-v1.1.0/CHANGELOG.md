
# Changelog

## 1.1.0 - 2026-09-29

- 新增 RTX 50 系列 Blackwell 预检：要求 CUDA 12.8+、Compute Capability 12.0、`sm_120` Kernel 和 BF16 可用。
- 面向智川云 RTX 5090 更新资源配置、安装流程、数据盘路径和文件存储备份边界。
- 所有 Gate 默认写入 `PDH_RESULTS_ROOT`，持久化哨兵写入 `PDH_PERSIST_ROOT`，不再硬编码 `/mnt/workspace`。
- Bootstrap 将 Torch、XDG、pip 和临时缓存定向到数据盘配置的缓存目录；仍然不安装或升级 PyTorch。
- 系统盘点增加 CUDA Compute Capability、Torch Architecture List、BF16、cuDNN 和三个存储层信息。
- 实验 Manifest 增加包版本、CUDA/HIP 版本、设备名和 Compute Capability。
- 新增智川云执行手册、平台迁移重测说明和存储保留策略。
- 保留 v1.0.2 的跨 CUDA/ROCm Checkpoint RNG 恢复修复和结果打包兼容性。

## 1.0.2 - 2026-09-29

- 修复 Checkpoint 通过目标设备 `map_location` 加载时，CPU RNG ByteTensor 被迁移到加速器的问题。
- Checkpoint 先在 CPU 暂存，再由模型和优化器状态加载器复制到目标设备。
- RNG 状态恢复前统一规范为连续 CPU `torch.uint8` Tensor。

## 1.0.1 - 2026-09-29

- 结果打包改为 Python 标准库，不依赖系统 `zip` 命令。
- 系统盘点默认脱敏 AMD ASIC 序列号。

## 1.0.0 - 2026-09-29

- 初始第一阶段环境、Mini-PDH、PPO、评估与证据打包工具。
