# 本地交付验证

验证日期：2026-09-29  
验证环境：Linux、Python 3.13.5、PyTorch 2.10.0+cpu  
验证范围：v1.1.0 CPU 路径、路径治理、打包与 CUDA 兼容规则

## 已验证

| 项目 | 证据 | 状态 |
|---|---|---|
| Bootstrap 与数据盘缓存变量 | `verification/local_cpu/bootstrap.log` | VERIFIED |
| Python 编译 | `verification/local_cpu/compile.log` | VERIFIED |
| 22 项单元/集成测试 | `verification/local_cpu/tests.log` | VERIFIED |
| Gate A A01-A06 CPU Smoke | `verification/local_cpu/evidence/gate-a/` | VERIFIED（CPU 路径） |
| Gate B Mini-PDH Smoke | `verification/local_cpu/evidence/gate-b/` | VERIFIED（CPU 路径） |
| Gate C 小规模吞吐 Smoke | `verification/local_cpu/evidence/gate-c/` | VERIFIED（CPU 路径） |
| Checkpoint CPU 暂存和 RNG 规范化 | `tests/test_checkpoint.py` | VERIFIED |
| CUDA 12.8 / sm_120 兼容规则 | `tests/test_cuda_compat.py` | VERIFIED（纯函数） |
| 无系统 zip 的结果打包 | `tests/test_pack_results.py` | VERIFIED |

## 尚未验证

- 智川云 RTX 5090 的真实驱动、CUDA Runtime 和 `sm_120` Kernel；
- RTX 5090 FP32/BF16 前后向与真实性能；
- 新实例数据盘停机持久性；
- 60 分钟 GPU 稳定性；
- 14 vCPU 与 RTX 5090 共置时的 PPO 端到端吞吐；
- Gate B 正式 3 种子结果。

目标实例必须重新执行 Gate A-C；旧 AMD 结果不能替代。
