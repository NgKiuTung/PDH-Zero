# PDH-Zero 0.6.0-rc5

本版本实现完整单手局内的逐家预测与自身多路线联合决策。工程验收和模型收益是两件事：本地短流程不能证明胜率或净收益提升，默认回退/部署基线仍为 rc4.1 `d000000266137`。付费 GPU 未启动；目标 RTX5090 未测。

## 先读这三处

- `docs/rc5-design.md`：输入/标签边界、增量架构、三组对照、预算和已知限制。
- `docs/rc5-5090-runbook.md`：精确路径、CPU 验收、GPU 短流程、开发和一次性新墙确认命令。
- 随本 ZIP 交付的 `PDH-Zero-rc5-acceptance.md` 与 `PDH-Zero-rc5-evidence.zip`：最终解压后的结果及全部候选；`verification/rc5` 为打包前过程证据，不冒充最终解压结果。

## 唯一基线

`checkpoints/candidates/d000000266137/model.pt`

SHA256：`34e97857dc3ec6c243e7f69d07c6deea08900ebb487c10e2f63ae8a5827699dd`。

入口先校验内容，再读取可信旧检查点；拒绝 rc3、latest 或同名不同内容文件。新 rc5 候选是独立 schema，不能送进旧 rc4 evaluate/train 入口。

## 当前入口

```bash
python tools/rc5.py --help
python tools/build_native.py
python tools/verify_refinement.py
python tools/probe_rc5.py --help
```

`verify_refinement.py` 虽保留旧名称，但读取当前 FILES.sha256、当前全部测试与当前静态审计，适用于本包 CPU 门禁。C++ 必须在目标 Python ABI 本地编译，不携带本机构建的 .so。Python、PyTorch、NumPy、编译器为既有环境依赖；本次没有新增模型运行依赖。

`rc5.py` 子命令依次为 `prepare` → `audit` → `dataset` → `plan` → `study`；独立 `fit` 用于固定预算的辅助实验；`freeze` → `confirm` 仅供足量开发门禁通过后的新牌墙确认。`probe_rc5.py` 只做真实候选串/并行一致性与两个配置的实测，不训练。

## 文件与证据

源码 ZIP 不重复装入两个原始大归档，继续使用已经上传/留存在实例的原件。证据 ZIP 内保存辅助数据 v2、H1/H2 过程、最终干净目录研究、候选权重、完整牌谱、暴露登记及真实测量；没有将候选标为部署通过。源码、证据包各有独立 SHA256。

`FILES.sha256` 仅校验源码 ZIP 内的文件，不校验外部训练数据。训练归档、候选、数据集、候选提案、冻结协议各有各自内容校验。

## 不变量

冻结房规见 `rules/source-v0.9.md`。Actor 仅接收合法 DecisionView 和局内 GRU；暗牌/未来事件只进入训练标签或只读审计。奖励是完整自然单手真实净分除以32，gamma=1，无牌型/少碰/责任规避奖励。预测不是可观察意图；八类结构值不是精确向听数/成牌概率。GRU 与回报在自然单手重置，跨手长期回报尚未实现。

旧 held-out 已暴露，永久标历史证据/诊断；历史原始暴露库和部分祖先未随包提供，覆盖保持 partial。CPU 数据不等于目标5090结果，辅助校准不等于打法收益。

旧 README/AGENTS 原文在 docs/*rc4.1-history.md；旧工具和文档为回归/历史保留，不能按其中 rc3 路径启动 rc5。
