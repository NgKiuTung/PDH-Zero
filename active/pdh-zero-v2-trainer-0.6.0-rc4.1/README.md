# PDH-Zero v2 Trainer 0.6.0-rc4.1

本候选版保留冻结房规和真实净分目标，加入C++批量特征/启发式、责任失分审计和容器配额内的实测选型。它是受控研究入口，不是已验证优于rc3的模型，也不自动启动正式长训。

## 先明确数据单位与泛化边界

本版是rc4的数据协议修订，不改变冻结房规、384 GRU结构、真实净分奖励或C++特征算法。

- **一手是从自然发牌开始，经过完整摸打、碰杠、胡牌或流局、抓马和结算的完整单手牌局**。4096手评估=1024组不同起始牌墙×四个候选座位，各自完整跑一遍，不是4096道何切题。四座位配对是有意复用牌墙，不算4096个独立牌墙。
- 模型每手内部按顺序使用GRU历史，PPO按完整座位轨迹重算，真实终局净分沿轨迹计算优势。训练会话保留庄位、连庄、累计分及留牌公开信息，但GRU及回报在手局边界重置；**不声称已训练出跨局记忆或整场长回报策略**。
- 新训练默认生成新的63位根种子，持久化在manifest；训练桌、开发牌墙按用途派生。每次新run不再静默重播旧版固定日期种子的起始牌墙。
- 同一次run的各候选仍共享固定开发牌墙以公平配对；开发集不参与PPO梯度，但反复选模型存在开发集过拟合风险。开发PASS不代表泛化验证PASS。
- 冻结候选后必须用独立新牌墙验证。已有`evaluate_candidates.py`默认新种子；先登记可达源run和祖先的实际牌墙，按牌种序列检查训练、开发与独立验证隔离。

详见`docs/data-protocol-audit.md`。部署前请用本版本重新prepare；旧rc4运行计划绑定旧源码，不能复用。

## 环境与资源

目标机已核实：Ubuntu22.04、Python3.12、PyTorch cu130、RTX5090、GCC11.4、Python.h及PyTorch附带的pybind11头可用。CPU `cpu.max=1400000 100000`对应14个CPU时间配额；`0-127`只是可调度逻辑CPU范围。内存上限64424509440字节，即60GiB。

不要据`nproc=128`创建128个Worker。准备工具每次只运行一个配置，自动检查quota、affinity和内存，按2/4/6/8/10/12 Worker测试；不足资源的配置不执行。每个Worker内部不启用OpenMP并行，PyTorch/BLAS限制为1线程。保留约2个CPU时间份额供主进程和系统，这是预算策略，不是物理绑核。

## 旁路安装

上传ZIP和同名.sha256到`/root/rivermind-data/pdh-zero/tmp/incoming-v2-0.6.0-rc4.1`，在该目录执行：

```bash
sha256sum -c pdh-zero-v2-trainer-0.6.0-rc4.1.zip.sha256
```

校验OK且目标目录不存在后解压，不覆盖旧版本：

```bash
test ! -e /root/rivermind-data/pdh-zero/active/pdh-zero-v2-trainer-0.6.0-rc4.1 &&
python3 -m zipfile -e pdh-zero-v2-trainer-0.6.0-rc4.1.zip /root/rivermind-data/pdh-zero/active
```

## 第一步：只准备，不训练

保持GPU无其他计算任务。进入新版本，执行唯一推荐准备入口：

```bash
cd /root/rivermind-data/pdh-zero/active/pdh-zero-v2-trainer-0.6.0-rc4.1
python3 tools/prepare_rc4.py   --source-run /root/rivermind-data/pdh-zero/evidence/v2-refinement-rc3/REFINE_20261003T070408232116Z
```

它会依次验证真实选择指针与权重、在本机编译、运行全套CPU测试及静态检查、对选中模型的16手真实评估牌谱差分、顺序测量Python参考与多种C++负载。每个采样配置重复两次、复用Worker并从同一组种子重置，检查行为/终局摘要一致，取中位吞吐。

不安装包、不升级编译器、不改CUDA、不改旧实验。主进程会输出`phase`；子进程PID、日志、计时保存在新的`PREPARE_...`目录。单个压测配置限时240秒；超时或内存安全边界失败会停止准备并保留证据，不尝试无限重跑。源码、ABI、设备或可用配额变化时计划需重新验证。

末尾必须为`status=PASS`，并给出`runtime_plan`路径与`recommendation`。若编译环境不满足，先查看日志；明确选择纯Python回退可重新执行同一准备命令并增加`--python-only`。不会在CPP失败时静默冒充CPP加速。

选型只测采样链（规则推进、观察编码、启发式、推理与牌谱），不含PPO。5%以内的近似最好配置优先选较少Worker；CPP不足Python参考5%优势则选Python。它是该机短测建议，不是全局最优或模型质量门禁。

## 第二步：按实测计划受控训练

不要手工复制Worker数字。使用准备步骤打印出的实际计划文件：

```bash
python3 tools/train_refinement.py   --runtime-plan /实际准备目录/benchmark/runtime-plan.json   --output-root /root/rivermind-data/pdh-zero/evidence/v2-refinement-rc4   --max-seconds 10800
```

计划绑定rc3的`selected_development.json`，导入其d265095权重，保持风险输出不参与策略，重新创建AdamW；本轮冻结锚为导入候选本身。默认500k学习者决策、每125k评估4096手、总预算3小时、连续两次没有合格名义改善提前结束。所有候选先保存再评估。预算不代表任务必须跑满。

新baseline相对于d265095，不能把它的改善量当成相对于最初warmup的新增收益。独立验证仍是单独一步；`long_training_allowed`保持false。

这是前台入口：保留实例和训练终端。需要停止只按一次Ctrl+C，等待`stopped`与Worker清理。支持保存完整已提交更新，但没有精确resume；不要把重开任务称为继续同一条轨迹。

## 监控和验收

启动输出精确run目录。查看`status.json`、`events.jsonl`、`metrics.jsonl`和`data-integrity.json`；不要依赖旧的`current_*`指针或猜测最新目录。完成后：

```bash
python3 tools/refinement_gate.py --run-dir /实际运行目录
```

**保留输出根目录的`.wall-exposures.sqlite3`，不要在每次实验后删除。** 它按真实有序牌种序列查重，忽略同牌种四个实体编号互换。检查包含已记录的预发下一局和独立马牌墙；报告中的“暴露墙数”不等于已打局数。已知跨用途或意外重复会在PPO更新前报错，不能删除登记库后把重放说成新数据。

只读导入可达祖先牌谱会占用启动时间，阶段为`audit_data_lineage`。同内容哈希的历史文件只导入一次。缺失旧目录会记录`historical_coverage=partial`，不会凭空保证所有过去数据未见。恢复旧归档时要保留manifest和replays。

`--seed`只用于明确的可复现研究；重复的显式种子可能被已有暴露登记阻断，这时应使用新run默认种子，不应静默绕开。随机采样不保证相似牌型永远不出现；完整同墙去重也不证明所有局面分布独立。

开发集PASS也不是独立确认。`tools/evaluate_candidates.py --help`提供冻结候选的新牌墙验证入口。

## 自责分与供牌链

新评估摘要的`responsibility`记录：直接责任支出、责任本金/中马增量、每百玩家—手局责任失分、责任事件频率、责任损失分位数以及最差1%/5%实际净分手局的付款来源。

普通被自摸分担单列，不称为纯运气。供牌链记录实际成功的碰/明杠、加杠对原碰的延续、资格与路线变化；被胡优先截断的碰牌声明不记供牌。该链是事实关联，不是反事实贡献估计，不参与奖励或门禁。

旧rc3结果无需重训即可审计：

```bash
python3 tools/audit_responsibility.py   --run-dir /root/rivermind-data/pdh-zero/evidence/v2-refinement-rc3/REFINE_20261003T070408232116Z   --output /root/rivermind-data/pdh-zero/evidence/rc3-responsibility-audit
```

输出必须是新目录。工具全量验证5份4096手评估的账本；默认每个评估抽取最差20手重放供牌链，不能把样本覆盖宣称为全部牌谱因果分析。

## 开发与证据

实现边界、二进制安全和统计限制见`docs/rc4-contract.md`。原rc4的历史实测证据在`verification/rc4/`；本次修订实测在`verification/data_protocol/`；用户目标14vCPU/RTX5090的性能与后续学习收益待准备/训练工具实测。发布包不含开发机.so，须在目标Python ABI下编译；不再传输旧210MB归档或模型权重副本。

早期README和训练工具为回归历史保留，不是当前推荐入口。禁止使用旧`long_job.py`绕过本版门禁。
