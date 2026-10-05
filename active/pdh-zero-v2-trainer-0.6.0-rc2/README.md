# PDH-Zero v2 Trainer 0.6.0-rc2

用途：从已保存的 0.6.0-rc1 教师预训练权重开始，进行受控 SOLO 策略稳定性调优。
消费者：云实例执行者、实验分析者与代码维护者。
本包不改变冻结房规，不生成新真人牌谱，不自动启动长训练。

## 1. 与上一版的区别

- 不重新生成2048手教师数据；只读导入 `checkpoint_warmup.pt`，拒绝更新号非0的latest。
- 保持原模型宽度和路线表示，重新初始化优化器；不继承行为克隆的Adam动量。
- 修复风险辅助监督进入共享Actor编码器的梯度路径；风险输出在本次策略中权重为0，
  仅作诊断。因有效策略配置变化，本包首先重新建立自己的固定评估基线。
- 固定一个学习座位、三个启发式对手；本次不自动跨入DUO，也不宣称DUO是旧退化唯一原因。
- 学习率默认5e-5，冻结warmup策略作KL锚；更新前验证行为概率，更新后检查完整批次KL。
  超过0.01则回滚该次模型和优化器。连续3次拒绝后结束本次调优。
- 真实spawn CPU环境进程，缓存重复决策编码；默认4个worker和64桌，一个主GPU进程。
- 同牌墙四座位逐局开发评估、牌墙组配对bootstrap；报告PASS/REGRESSION/INCONCLUSIVE。
- SIGINT/SIGTERM记录stopped并回收本任务worker，不再把人工停止标成未知训练故障。

程序修复通过CPU测试不代表策略已提升。RTX 5090吞吐、真实warmup导入和学习效果尚需实测。

## 2. 安装与验收

两个上传文件放到：

`/root/rivermind-data/pdh-zero/tmp/incoming-v2-0.6.0-rc2/`

校验ZIP后旁路解压。不要覆盖旧包，不修改旧 `env.sh` 或 `current_*` 指针。

```bash
cd /root/rivermind-data/pdh-zero/tmp/incoming-v2-0.6.0-rc2
sha256sum -c pdh-zero-v2-trainer-0.6.0-rc2.zip.sha256
```

仅在校验通过且以下目标目录不存在时执行：

```bash
test ! -e /root/rivermind-data/pdh-zero/active/pdh-zero-v2-trainer-0.6.0-rc2 && \
python3 -m zipfile -e \
  /root/rivermind-data/pdh-zero/tmp/incoming-v2-0.6.0-rc2/pdh-zero-v2-trainer-0.6.0-rc2.zip \
  /root/rivermind-data/pdh-zero/active
```

```bash
cd /root/rivermind-data/pdh-zero/active/pdh-zero-v2-trainer-0.6.0-rc2
python3 tools/verify_refinement.py
```

这一个命令包含文件哈希、全部CPU回归和基础静态审计。预期
`refinement_candidate_check=PASS`、`refinement_entry_ready=true`，
`formal_long_run_ready=false`。它不会跑GPU长任务；ruff/mypy结果仍单独标明。
旧工具与旧测试保留用于回归，不使用 `train_curriculum.py` 或 `long_job.py start` 开本轮。

## 3. 启动受控调优

先用 `nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv`
确认GPU只有表头。然后从本包根目录执行：

```bash
python3 tools/train_refinement.py \
  --warmup-checkpoint /root/rivermind-data/pdh-zero/evidence/v2-curriculum/CURRICULUM_60M_20261003T015945Z/checkpoints/checkpoint_warmup.pt \
  --output-root /root/rivermind-data/pdh-zero/evidence/v2-refinement \
  --workers 4 \
  --max-seconds 5400
```

默认目标750000个学习者决策；总预算5400秒，从本次任务初始化计起，包括模型导入、
初评、采样、PPO、风险诊断与后续评估。到检查点或清理边界退出可能略有延迟；
时间不足时会记录未完成，不补造评估。不会把90分钟训练外再藏一轮无预算评估。

此入口是前台命令：保持该终端和云实例运行，不关闭浏览器来测试后台保活。
开头的 `run_requested.output` 是本次精确目录。不要用“最新目录”去猜是哪次任务。
目录包含 `status.json`、`events.jsonl`、`metrics.jsonl`、`manifest.json`。
阶段包括 `load_warmup`、`evaluation`、`collecting`、`ppo_update`、`risk_diagnostics`；
CPU阶段也每约10秒刷新心跳。更新指标保存的是本次已提交的完整更新。

在另一个终端查看已打印的精确目录中的 `status.json` 或 `events.jsonl` 即可。
按一次Ctrl+C可请求停止，程序关闭自己的子进程并标记stopped。
本CLI不支持中断后精确resume；保留最近完整更新检查点和已完成牌谱，未提交的当前更新不承诺恢复。

## 4. 结果审查

任务结束后：

```bash
python3 tools/refinement_gate.py --run-dir <run_requested.output中的精确目录>
```

此命令只读重算逐局配对比较，不改原始文件：

- PASS：本次重复开发集的工程晋级条件满足，仍需人工检查后再定长训。
- REGRESSION：有回归信号；停止扩大预算，保留结果。
- INCONCLUSIVE：证据不足，不把微小均值变化当成功或失败的因果证明。
- INCOMPLETE：人工停止、预算结束或没有完整有效更新评估，不作为策略晋级。

退出码分别为0、2、3、3；1表示输入/哈希等错误。
`checkpoints/checkpoint_initial.pt` 保留本次基线，`checkpoint_best.pt` 只被通过门禁且
均分更高的候选替换，`checkpoint_latest.pt` 不等于最佳模型。

## 5. 范围与诚实边界

- 房规来源固定为 `rules/source-v0.9.md`。不改点炮、自摸、抓马、门清和承包规则。
- 本次训练保留真实单手结算回报及连续环境状态，但尚未实现完整的跨手长期收益信用分配。
- 评估是明确的受控单手协议：每个牌墙恰好轮换四个候选座位，初始庄位0、连庄0，
  每手仍依法完成真实抓马。不能把它称为连续会话的最终评估。
- 风险诊断来自本次已选弃牌，不是未选动作的全信息反事实标签，亦非独立验证集。
- 配对区间使用反复查看的开发集，不构成最终论文的盲测、多种子显著性结论。
- 冻结策略锚、风险诊断模式、优化器重置和KL保护同时构成稳定化候选；性能若改善，
  还需消融才能归因于某一模块。

实现细节见 `docs/refinement-contract.md`；上一轮证据与更正见 `docs/diagnosis-rc1.md`；
实际测试和性能测量见 `verification/rc2/`。其他版本文档属于历史记录，不能覆盖本入口。
