# PDH-Zero v2 Engine 贡献者入口

先读 `README.md`、`docs/implementation-scope.md`、`rules/source-v0.9.md` 和 `TODO.md`。`rules/source-v0.9.md` 是当前冻结房规的单一语义来源；改变任何规则必须更新来源、配置、T 案例、测试和 SHA256，不得在代码中静默补默认值。

## 工程边界

本仓库是新的纯 Python 参考引擎，不是旧 Mini-PDH 重命名。不得修改旧归档、旧 Checkpoint、云实例 `env.sh` 或已完成实验。模型训练尚未交付，`training_ready=false` 必须保持，直到观察、采样、PPO、恢复与短试门禁全部通过。

## 可复用工具索引

| 功能 | 路径 | 适用场景 | 调用 |
|---|---|---|---|
| 冻结规则 CPU 验收 | `tools/verify.py` | 发布前、上传实例后 | `python3 tools/verify.py` |
| 随机合法整局压力 | `tools/simulate_random.py` | 牌实体守恒、响应与终局压力 | `PYTHONPATH=src:. python3 tools/simulate_random.py --hands 1000` |
| 无依赖静态基础审计 | `tools/static_audit.py` | 无ruff/mypy环境的发布前检查 | `python3 tools/static_audit.py --output verification/static-audit.json` |
| 房规与案例追踪 | `rules/source-v0.9.md`、`rules/cases.json` | 规则实现、代码审查 | 只读或由版本化更新脚本处理 |

## 变更要求

- 编码前加载 Coding Rule；公共函数、关键状态和非显然房规必须有注释。
- 同步更新 `CHANGELOG.md`、`TODO.md` 和相关测试。
- 临时文件只放 `tmp/<用途-日期>/`，交付前清算。
- 失败测试、未预期异常、来源哈希不一致或牌实体不守恒均阻断交付。
- 性能实现必须与参考引擎做固定牌墙／动作流差分，不复制第二份房规目录。
