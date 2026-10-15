# PDH-Zero rc5 维护入口

先实际读 `instructions/coding-rule/SKILL.md`、README、docs/rc5-design.md，再读具体源码。不要相信旧工具不可用的断言；最小自检后在新目录工作。

唯一基线 rc4.1 d000000266137，SHA256 34e97857dc3ec6c243e7f69d07c6deea08900ebb487c10e2f63ae8a5827699dd。禁止 latest/rc3 自动替代。冻结房规、原Actor/特征、C++核不可为过门禁改动。旧入口保留供历史回归，不是本次训练入口。

## 可复用工具索引

|功能|入口|参数/约束|
|---|---|---|
|本机ABI编译|tools/build_native.py|不下载依赖，不带 .so 进源ZIP|
|全套CPU/文件/静态门禁|tools/verify_refinement.py|当前整个tests；先build；ruff/mypy单独记录|
|真实训练牌谱Python/C++差分|tools/verify_native.py|--replays --hands 64 --output 新JSON|
|输入/暴露登记|tools/rc5.py prepare|--train-run --heldout-run --registry --output|
|实际权重公开历史/隐藏扰动审计|tools/rc5.py audit|--baseline 或 --candidate；--replays --hands --pairs|
|因果特征、独立标签、整会话切分|tools/rc5.py dataset|仅冻结TRAIN源；--baseline --replays --registry --output|
|独立辅助拟合|tools/rc5.py fit|--dataset --seed --steps --batch --device --output|
|新seed完整预算配置|tools/rc5.py plan|--device --backend --workers --output；默认 smoke|
|三seed/三分支/三对手对照|tools/rc5.py study|--baseline --dataset --registry --config --output|
|真实rc5串并行实测|tools/probe_rc5.py|--baseline --candidate --registry --device --workers --output|
|全seed冻结/一次性确认|tools/rc5.py freeze / confirm|烟雾不能冻结；新墙只消费一次，失败也记暴露|

新代码在 src/pdh_v2/table，仍使用旧完整轨迹/结算/PPO。仅两个旧运行接口加默认关闭钩子：paired_evaluation 对手工厂、stable_ppo 回退前保存提案。

诊断必须不改Actor张量和评估轨迹。辅助梯度不能进原Actor；PPO不反传辅助头。标签不能倒填意图，当前能力与未来非强制行为分开。资格门禁按任务/座位和训练先验比较，不看旧held-out；失败头仅回退为先验。

受保护源码包括 src、native、tools、rules。试验后再改源码就属于新版本，不能继续旧冻结确认。模型/优化器实际回退，但所有提案留下；不会自动部署、长训或精确resume。

四座位按完整牌墙组统计；不把3～5个独立后续训练seed说成从零重复训练。训练每更新 fresh session/dealer0，是受控单手设计，不是跨手GRU。

变更维护 CHANGELOG，原详细文件保留。临时数据在 tmp，最终保留证据进入证据ZIP，不进入源包。新工具先查索引。打包后在干净目录编译、全部测试、实际权重短流程，不能用方案当结果。
