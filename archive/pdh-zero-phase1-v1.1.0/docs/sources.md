
# 外部资料与版本依据

核查日期：2026-09-29。

1. NVIDIA CUDA Toolkit、Driver 与 Architecture Matrix：Blackwell Compute Capability 12.0 首次由 CUDA 12.8 支持。  
   https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html
2. NVIDIA CUTLASS 支持表：GeForce RTX 50 系列为 Compute Capability 12.0，最低 CUDA 12.8。  
   https://docs.nvidia.com/cutlass/4.3.3/overview.html
3. PyTorch 官方论坛：RTX 50/Blackwell 要求 CUDA 12.8+ 构建；CUDA 12.1 构建不兼容。  
   https://discuss.pytorch.org/t/rtx-5090-not-detected-by-pytorch-2-2-0/222399
4. 智川云数据管理：系统盘 `/`、数据盘 `/root/rivermind-data/`、文件存储 `/root/rivermind-fs/` 的生命周期和用途。  
   https://docs.gpuhome.cc/docs/data/overview.html
5. 智川云容器实例：数据盘和文件存储路径，以及本地 SSD 无冗余备份提醒。  
   https://docs.gpuhome.cc/docs/instance/instance.html
6. 智川云镜像说明：更换镜像会重置系统盘，数据盘不受影响。  
   https://docs.gpuhome.cc/docs/environment/image.html
