# rc4.1本地数据协议验收

`real-rc3-hand-audit.json`：全量读取26886手记录，48手抽样全动作与结算重放；不能称全量规则重放。

`unit-tests.log`：378项自动回归（原338项加40项），包括完整局、GRU历史、随机流、重复/交叉隔离及CPU短流程。

`runtime-validation.json`：实际d265095权重的CPU/C++/2Worker短流程；为边界验收只收集3手。该极小批次在默认学习率下触发local_kl拒绝，PPO正确回滚，未冒充学习效果改善。新墙4手冻结比较仅验证接口与隔离，不用于统计结论。

`coverage-summary.json`：全套加覆盖率运行再次通过；语句85.64%，含分支综合82.37%，单独分支71.07%。本地Python3.13/Torch2.10 CPU，目标Python3.12/Torch2.12 CUDA未执行。ruff/mypy未安装。

`unchanged-contract-files.json`：27个冻结规则、核心游戏/评分、网络、PPO、完整序列及native源文件逐字未变。最终ZIP的重新解压验收另行随包提供release-check.json。
