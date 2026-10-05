# PDH-Zero v2 Trainer 0.6.0-rc3

用途：从已经保存的教师 warmup 开始受控 SOLO 调优，永久保存每个评估候选，
按收益与尾部风险筛选，并用未参与调参的新牌墙验证一个预先选定的候选。
这是代码候选版，不包含训练好的新权重；不自动启动48小时长训。

## 1. 安装与验收

Linux + Python 3.10以上，已有可用 PyTorch 和 NumPy。RTX 5090沿用实例已经实测通过的
PyTorch/CUDA环境，不重装驱动、不修改旧 `env.sh` 或 `current_*` 指针。
ZIP在独立目录解压；禁止覆盖旧版本、实验或检查点。

```bash
python3 tools/verify_refinement.py
```

该命令校验文件、运行完整CPU回归和基础静态审计。应得到
`refinement_candidate_check=PASS`。`formal_long_run_ready=false`、
`target_cuda_verified=false` 是代码包验收边界，不代表目标GPU已经失败。
详细的历史与本次本地证据分别在 `verification/rc2` 和 `verification/rc3`；
本次代码验收不能代替用户GPU上的真实学习效果。

## 2. 本轮唯一训练入口

先确认GPU没有其他计算任务，再运行；旧实例目录保持不动。

```bash
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

python3 tools/train_refinement.py \
  --warmup-checkpoint /root/rivermind-data/pdh-zero/evidence/v2-curriculum/CURRICULUM_60M_20261003T015945Z/checkpoints/checkpoint_warmup.pt \
  --output-root /root/rivermind-data/pdh-zero/evidence/v2-refinement-rc3 \
  --workers 4 \
  --transport shared \
  --max-seconds 10800
```

起点仅接受项目的 `update=0` 教师warmup；不把旧latest冒充warmup。
仍使用原384宽度网络及合法多路线表示，风险头保持隔离且不参与选牌。
不重复生成2048手教师数据，不在更新间插入额外BC，不自动进入DUO。

默认值：50万学习者决策，每12.5万评估一次；每次4096手（1024牌墙×四座位）。
总墙钟预算3小时，包含导入、初评、采样、PPO和所有评估；连续两次没有符合风险要求的
名义开发收益改善会提前停止。预算不是完成保证，程序会预留最终评估时间；
边界检查与清理可能略晚于期限，没有完成的评估不晋级。

这是前台任务。保持原终端和实例运行，不重复启动。按一次Ctrl+C请求停止，
在可清理边界退出、回收登记Worker；本入口没有精确resume，不承诺恢复未提交批次。

输出第一行给出精确 `output`。只读查看该目录：

```bash
python3 -m json.tool /实际output目录/status.json
tail -n 5 /实际output目录/events.jsonl
tail -n 3 /实际output目录/metrics.jsonl
```

阶段包括 `load_warmup`、`evaluation`、`collecting`、`ppo_update`、`risk_diagnostics`。
CPU环境阶段GPU可能间歇空闲；以有效决策吞吐、组件耗时和进度判断，不以功率作通过标准。

## 3. 结束后先核对候选，不直接续训latest

```bash
python3 tools/refinement_gate.py --run-dir /实际output目录
```

输出可能是PASS、REGRESSION、INCONCLUSIVE或INCOMPLETE。任何一个都不会自动开启长训。
已完成实验结构：

```text
manifest.json                       实际配置、停止原因、所有计数
status.json                         阶段与心跳，结束后为最终状态
events.jsonl / metrics.jsonl         事件与每次更新报告
checkpoints/candidates/<标签>/      每次评估前保存的不可覆盖候选
  model.pt / model.pt.sha256
  candidate.json                    绑定更新号、权重与评估摘要
checkpoints/checkpoint_initial.pt   导入后起点
checkpoints/checkpoint_latest.pt    最近完整更新，未必最好
checkpoints/checkpoint_selected_development.pt
selected_development.json           满足预设名义收益/风险条件的开发候选
qualified.json                     仅出现PASS时存在，仍不是最终论文结论
evaluations/<标签>/results.jsonl    全部逐手结果和支付归因
evaluations/<标签>/hands.jsonl.gz   私有可重放牌谱
evaluations/<标签>/worst_hands.json 最差手局索引，不复制第二套牌谱
```

`selected_development` 可以来自INCONCLUSIVE；它表示下一步值得验证，不表示已证实提升。
`checkpoint_best.pt` 是历史兼容文件，不能仅按名称挑模型；以选择指针、绑定哈希和门禁为准。
没有候选满足名义风险筛选时，选择保留baseline，不以最后更新强行替代。
旧rc2未保存的265k／509k权重无法由本包补回；相同seed重跑也不保证逐位重现旧模型。

## 4. 独立新牌墙验证

先固定训练结束后的单个开发候选，再运行下面的纯评估命令。不同时比较很多候选再挑最好。

```bash
python3 tools/evaluate_candidates.py \
  --run-dir /已完成的rc3实验目录 \
  --output-root /root/rivermind-data/pdh-zero/evidence/v2-heldout-rc3 \
  --hands 4096 --workers 4 --max-seconds 5400
```

默认新seed为161000731。程序检查与该源实验所有开发集牌墙的真实key不重叠；
同一验证输出根目录还保留 `wall_reservations.json`，包括中断时已预留的牌墙，拒绝重复使用。
换根目录不能让已经看过的牌墙变回未见数据。默认选择仍为baseline时会拒绝无意义的重复比较。

两份冻结权重先复制并校验，然后在同一批新牌墙上各评4096手，共8192手。
这不是继续训练，不改变源实验。结果保存在新目录的 `report.json`。
它只检验一个训练种子的一个冻结候选；不能代替多种子复现或完整连续牌局评估。
旧rc2的真实initial/latest可显式用 `--candidate-checkpoint` 只读评估，但不能指定不存在的265k文件。

## 5. 门禁口径

主奖励仍是冻结房规的真实净分，不加碰杠惩罚、门清奖励或人工胡牌奖励。
配对比较以整个牌墙的四座位为重采样单元，同时输出均分、胜率、点炮率、流局率、CVaR-5%。

PASS需要：均分差至少+0.1且95%区间下界>0；点炮率差上界≤+0.01；
CVaR-5%差下界≥−`max_cvar_drop`。REGRESSION按对应明确恶化区间判定，其余INCONCLUSIVE。
`--max-cvar-drop 2.0` 是本轮显式风险容忍参数，不是房规，也不是已证明的最优值。
未经审阅不因为想要PASS而事后放宽它。重复开发集筛选不能写成独立显著性证明。

## 6. 性能与已知限制

共享内存只传合法观察与候选动作数值特征，Pipe保留少量控制消息和终局牌谱。
同步PPO和回合次序不变；不是全异步推理，也没有承诺用户机器上的加速或GPU占用。
可独立在CPU运行传输微基准：

```bash
python3 tools/benchmark_refinement.py --tables 128 --workers 0,4 \
  --transports pipe,shared --output benchmark-128.json
```

该基准不包含GPU模型，不能作为端到端CUDA吞吐。`worker_rpc_wall_seconds` 包含Worker计算与等待，
不是纯IPC时间。风险指标仍为同批诊断，不能当成独立校准表现。手局奖励和受控单手评估
不等于已验证跨手连庄长期策略。冻结房规仍唯一来源 `rules/source-v0.9.md`，本版逐字未改。
