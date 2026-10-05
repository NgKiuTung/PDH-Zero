# 本地验证结果

版本：0.3.0-foundation（候选工程包，不可训练）

- CPU测试：161项，失败0、错误0；另以 `PYTHONWARNINGS=error` 复跑通过。
- 规则源：v0.8，SHA256 与 confirmed/open_decisions/cases 一致。
- Python 3.13.5实际运行通过；26个Python文件通过Python 3.10语法解析检查。
- coverage 结果见 `coverage.json` / `coverage.txt`；源码语句覆盖率95.48%，行与分支综合覆盖率93.16%。
- 补充静态审计通过：100字符行长、尾随空白、Tab、简单未使用导入、`eval/exec`、pickle/marshal/shelve检查均无问题。
- 一炮多响旧墙不足时统一使用完整新牌序开马；共用一批马，各赢家按本局位置独立算m。
- 十二张落地责任仅由第四碰锁定，仅覆盖同局碰牌者自摸；点炮胡仍由实际点炮者支付普通一家费用。
- ruff与mypy：当前执行环境未安装；离线 `--no-index` 查找不到安装包，未伪造通过。
- 未操作云实例、未启动GPU、未实现完整模型/整局引擎/训练入口。

115项源案例追踪状态见 `rules/cases.json`。`KERNEL_COVERED`只表示对应参考纯函数/账本契约已有测试，不表示完整房规、整局状态机或正式训练已验收。
