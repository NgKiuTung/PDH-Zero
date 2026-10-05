# PDH-Zero v2 冻结规则参考引擎

版本：`0.4.0-engine`。本包是完整房规冻结后的纯 Python 参考引擎，独立于旧 `pdh-zero-phase1-v1.1.1`。它用于验证 136 张行牌、胡牌评分、责任结算、抓马、庄位迁移和确定性牌谱；**尚未包含 Actor、Critic、PPO 或 GPU 训练入口**。

## 已实现范围

- 136 张实体牌、传统发牌、墙头普通摸牌、墙尾杠后补牌。
- 弃牌、碰、明杠、暗杠、加杠、过、胡和响应窗口。
- 一炮多响、只抢加杠、过胡锁定至自己下一次实际摸牌。
- 无鸡胡；七对家族、十三幺／十三面、四暗刻、十八罗汉、三元、四喜、花色与幺九系列的完整解释和去重叠加。
- 自动选择最终倍数最高的完整胡形，禁止跨解释拼接。
- 普通点炮、自摸、一炮多响、第四碰十二张落地、直接点杠杠爆的零和账本。
- 无封顶、流局零点数但保留事实。
- 连庄、流局保留、多人胡后的下一庄家选择。
- 旧墙、下一把手牌、专门开马三种牌源；超过 136 张时分轮重洗。
- 会话结束时通过仅结算的虚拟下一把发牌完成待结算。
- 初始牌序与动作流的确定性单手牌谱和最终状态哈希。

## 明确边界

本包的 `engine_ready=true` 只表示 CPU 参考规则引擎通过本包测试。以下模块尚未交付，因此 `training_ready=false`：

- Actor 合法观察与隐藏信息投影；
- 多路线特征与防守风险辅助；
- 多桌并行采样和 GPU 批量推理；
- Recurrent PPO、Checkpoint 和正式训练启动器；
- 连续会话的完整可恢复快照与大规模分片牌谱。

旧 Mini-PDH 模型、Checkpoint 和得分不能迁移为本规则下的正式结果。

## CPU 验收

```bash
python3 tools/verify.py
```

预期包含：

```text
engine_check: PASS
foundation_check: PASS
full_rules_frozen: true
training_ready: false
```

随机合法整局压力测试：

```bash
PYTHONPATH=src:. python3 tools/simulate_random.py \
  --hands 1000 \
  --seed 20260930 \
  --output verification/random-legal-1000.json
```

`--output` 必须指向不存在的新文件，避免覆盖已有证据。

## 目录

```text
rules/                 房规语义源、配置和 T 案例追踪
src/pdh_v2/game/       136 张单手状态机与连续会话
src/pdh_v2/scoring/    完整胡形解释和计分
src/pdh_v2/replay/     确定性单手牌谱
src/pdh_v2/            抓马、责任、账本、快照等基础模块
tests/                 CPU 单元与集成测试
tools/verify.py        正式 CPU 验收入口
tools/simulate_random.py 随机合法整局压力工具
verification/          本次发布的不可变验收证据
```

## 部署约束

- 不覆盖旧 v1.1.1，也不修改现有 `env.sh`。
- 建议旁路解压到 `active/pdh-zero-v2-engine-0.4.0/`。
- 当前不要把 `PDH_PHASE1_ROOT` 指向本包，也不要启动 GPU 长任务。
- 下一阶段在本引擎上实现合法观察、批量采样和 PPO；通过 10 分钟吞吐与 30–60 分钟学习门禁后，再批准正式长跑。
