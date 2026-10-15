# rc5 实机执行手册（5090 尚未实测）

本文命令使用交接记录的实例目录，不意味着本次已连入或启动该实例。本次交付只在 CPU 容器实测。付费 GPU 保持关闭。最后的实测数值、候选选择和文件哈希以源 ZIP 旁的 `PDH-Zero-rc5-acceptance.md/json` 与证据包为准。

## 1. 输入、路径与不覆盖原则

交接记录的目标环境是 Ubuntu 22.04 / Python 3.12 / NVIDIA RTX 5090 32GB、驱动 580.119.02；这些是历史信息，必须在实际开机后重新检查。不要安装本机 CPU 版 torch 覆盖实例自带的 CUDA torch，不要复制本机 `.so`。

源包为 `PDH-Zero-rc5-20261005.zip`，顶层目录 `pdh-zero-v2-trainer-0.6.0-rc5`。证据包为 `PDH-Zero-rc5-evidence.zip`，顶层 `PDH-Zero-rc5-evidence`。下载后把两个 ZIP 和 `PDH-Zero-rc5-SHA256SUMS.txt` 放在历史 releases 目录。先校验 ZIP 的实际 SHA256，再解压至新目录。不得覆盖 rc4.1 或旧证据。

```bash
set -euo pipefail
ROOT=/root/rivermind-data/pdh-zero
PY=/opt/conda/bin/python3
STAMP=$($PY -c 'from datetime import datetime,timezone; print(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))')
ARCHIVES="$ROOT/releases"
NEW="$ROOT/active/rc5-$STAMP"
OUT="$ROOT/evidence/rc5/$STAMP"
mkdir -p "$NEW" "$OUT"
cd "$ARCHIVES"
sha256sum --ignore-missing -c PDH-Zero-rc5-SHA256SUMS.txt
$PY -m zipfile -e PDH-Zero-rc5-20261005.zip "$NEW"
$PY -m zipfile -e PDH-Zero-rc5-evidence.zip "$NEW"
SRC="$NEW/pdh-zero-v2-trainer-0.6.0-rc5"
EVID="$NEW/PDH-Zero-rc5-evidence"
TRAIN="$ROOT/evidence/v2-refinement-rc4.1/REFINE_20261003T131918636050Z"
HELD="$ROOT/evidence/v2-heldout-rc4.1/HELDOUT_20261003T151645353474Z"
BASE="$TRAIN/checkpoints/candidates/d000000266137/model.pt"
REPLAY="$TRAIN/replays/hands.jsonl.gz"
REG="$OUT/exposure.sqlite3"
DATA="$EVID/auxiliary-v2.pt"
cd "$SRC"
sha256sum -c FILES.sha256
printf '%s  %s\n' 34e97857dc3ec6c243e7f69d07c6deea08900ebb487c10e2f63ae8a5827699dd "$BASE" | sha256sum -c -
```

如果实例内尚未保留 TRAIN/HELD 目录，可把本次上传的两份原始 tar.gz 解压到另一个新证据父目录后按其实际顶层路径设置 TRAIN/HELD。用压缩包内已有的 d000000266137，不能换成 latest、d000000265095 或其他权重。原档字节数/SHA256 见 input-receipt.json；证据包另有原始 d000000266137 权重副本可核对，但不取代旧牌谱输入。

## 2. 先在 CPU 运行环境和源码门禁

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$SRC/src"
env CUDA_VISIBLE_DEVICES='' "$PY" -c 'import sys,torch; print(sys.version); print(torch.__version__); print("CPU gate")'
env CUDA_VISIBLE_DEVICES='' "$PY" tools/build_native.py
env CUDA_VISIBLE_DEVICES='' "$PY" tools/verify_refinement.py
```

`verify_refinement.py` 沿用历史名称，实际读取本源包的 FILES.sha256、整个 tests 目录和静态门禁，不是只测 rc4.1。退出非零就停；它会打印新建报告目录。ruff/mypy 没有实际运行时仍单独标记，不把基础 AST 检查当成完整类型检查。

## 3. 合并已暴露墙的登记，不碰原注册表

先复制实例既有注册表（若存在），再导入 TRAIN 与旧 held-out 的已知八份牌谱。不要清空原注册表。SQLite 用 backup，不能随手复制正在写入的文件。

```bash
OLDREG="$ROOT/evidence/v2-refinement-rc4.1/.wall-exposures.sqlite3"
export OLDREG REG
"$PY" - <<'PYREG'
import os, sqlite3
from pathlib import Path
src, dst = Path(os.environ['OLDREG']), Path(os.environ['REG'])
if dst.exists():
    raise FileExistsError(dst)
if src.is_file():
    with sqlite3.connect(f'file:{src}?mode=ro', uri=True) as old, sqlite3.connect(dst) as new:
        old.backup(new)
PYREG
env CUDA_VISIBLE_DEVICES='' "$PY" tools/rc5.py prepare \
  --train-run "$TRAIN" --heldout-run "$HELD" --registry "$REG" \
  --output "$OUT/exposure-import.json"
```

没有原注册表时可以导入八份已知记录，但历史覆盖度仍为 partial，不能声称全谱系无交叉。旧 held-out 只在 prepare 中用于登记；禁止把它传给 dataset/fit。证据包中的 `exposure.sqlite3` 是本次 CPU 实验最终注册表；若要把本次所有 CPU 暴露也加入实例的新工作注册表，应使用该注册表作为 backup 的来源，并保留原实例注册表备份。两份注册表并存且需要合并时，**不要覆盖任一份**；先按源牌谱继续登记缺少的实验记录，或把它们作为两份独立的限制清单审计。本发布不提供无损通用双注册表合并器。

## 4. 原权重审计和真实牌谱差分

```bash
env CUDA_VISIBLE_DEVICES='' "$PY" tools/rc5.py audit \
  --baseline "$BASE" --replays "$REPLAY" --hands 16 --pairs 32 --backend cpp \
  --output "$OUT/baseline-audit.json"
env CUDA_VISIBLE_DEVICES='' "$PY" tools/verify_native.py \
  --replays "$REPLAY" --hands 64 --output "$OUT/native-real64.json"
```

隐藏干预结果必须为严格不变；公开历史敏感性不要求变大。正常 Actor 没有暗牌/未来墙接口，审计代码具有特权只为独立验证，不连接到实时策略。

## 5. 辅助训练数据：可直接用已验证缓存，也可从 TRAIN 重新制作

证据包 `auxiliary-v2.pt` SHA256 为 `8754e41f38f43824aa98a4d397aef0a04c012f07bbd91819bc7787b94dd3fee0`。它来自冻结 TRAIN 文件前 512 手完整自然轨迹，13,953 个采样时点。训练/验证按完整 session 分组，验证只对新辅助头未见，**不是对 rc4.1 或整项研究未见**。头的训练先验/资格检查都只使用这份 TRAIN 侧数据。

```bash
printf '%s  %s\n' 8754e41f38f43824aa98a4d397aef0a04c012f07bbd91819bc7787b94dd3fee0 "$DATA" | sha256sum -c -
# 需要重建缓存时执行；不要覆盖现有文件：
env CUDA_VISIBLE_DEVICES='' "$PY" tools/rc5.py dataset \
  --baseline "$BASE" --replays "$REPLAY" --registry "$REG" \
  --hands 512 --stride 12 --backend cpp --output "$OUT/auxiliary-v2.pt"
# 后续二选一，只用哈希验证成功的那份：
# DATA="$OUT/auxiliary-v2.pt"
```

## 6. 新工作目录真实权重短流程（仍不需要 GPU）

先生成新 seed，再在运行前一次性声明完整预算；运行后不能改 manifest/指标/受保护源码，也不要看到结果后挑 seed。

```bash
env CUDA_VISIBLE_DEVICES='' "$PY" tools/rc5.py plan --device cpu --backend cpp \
  --workers 0 --output "$OUT/cpu-plan.json"
export OUT
"$PY" - <<'PYPLAN'
import json, os
from pathlib import Path
p=Path(os.environ['OUT'])/'cpu-plan.json'
v=json.loads(p.read_text())
v.update(updates=2, hands_per_update=8, evaluation_hands=16,
         evaluation_environments=8, smoke_only=True)
p.write_text(json.dumps(v, indent=2)+'\n')
PYPLAN
env CUDA_VISIBLE_DEVICES='' "$PY" tools/rc5.py study \
  --baseline "$BASE" --dataset "$DATA" --registry "$REG" \
  --config "$OUT/cpu-plan.json" --output "$OUT/cpu-study"
```

每个 seed 都先评估未改动 rc4.1，再独立拟合辅助头，接着原策略/诊断/联合各走相同预算。读取 `cpu-study/report.json`、`results.tsv`、每分支 `trial.json`、完整 replay 和 `candidate-outcomes.json`。PASS 指工程流程完成，不指变强；smoke 必须 INCONCLUSIVE，不能冻结晋级。任何失败输出保留，不能只留下成功候选。

## 7. 确认本人开机后，才执行 CUDA 预检和 rc5 专用测量

这一节是精确实机步骤，**本次未执行**。在实例关闭时不要启动定时任务或无人值守作业。

```bash
nvidia-smi
"$PY" -c 'import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NO_CUDA")'
export EVID
CAND=$($PY - <<'PYCAND'
import json, os
from pathlib import Path
p=Path(os.environ['EVID'])
r=json.loads((p/'release-selection.json').read_text())
print(p/r['candidate_relative_path'])
PYCAND
)
"$PY" tools/probe_rc5.py --baseline "$BASE" --candidate "$CAND" \
  --registry "$REG" --device cuda --workers 4 --environments 64 --hands 64 \
  --transport shared --max-seconds 600 --output "$OUT/cuda-probe"
```

候选为本地最终计划中**预先列出的第一个 seed**的有效联合检查点，仅供集成验证，非按净分挑选、更非已晋级模型。probe 对比 workers=0 与指定 workers=4 的三个对手完整轨迹，先要求精确一致，再报告吞吐。推荐只针对这两档和该任务，不等于最优 workers，更不能把评估速度当训练速度。机器资源不足会拒绝；有别的 GPU 计算进程时会拒绝，而不是自动杀掉它。优先核对真实 cgroup 配额，不凭 5090 名称直接开 12 workers。

然后用新的 plan 和 probe 实测档位跑 CUDA 小样本。不要把上一个 CPU run 改名为 CUDA run：

```bash
"$PY" tools/rc5.py plan --device cuda --backend cpp --workers 0 \
  --output "$OUT/cuda-plan.json"
"$PY" - <<'PYCUDA'
import json, os
from pathlib import Path
p=Path(os.environ['OUT'])
v=json.loads((p/'cuda-plan.json').read_text())
probe=json.loads((p/'cuda-probe/report.json').read_text())
assert probe['status']=='PASS'
v.update(workers=probe['recommendation']['workers'], updates=2,
         hands_per_update=64, evaluation_hands=64, evaluation_environments=64,
         transport='shared', smoke_only=True)
(p/'cuda-plan.json').write_text(json.dumps(v, indent=2)+'\n')
PYCUDA
"$PY" tools/rc5.py study --baseline "$BASE" --dataset "$DATA" --registry "$REG" \
  --config "$OUT/cuda-plan.json" --output "$OUT/cuda-study"
```

只有真实执行并查过这些输出后才能更新目标设备验收状态。study 中固定的 `target_5090_tested=false` 是本发布声明；实际机器身份/显存峰值在 probe 中独立记录，需要新设备验收报告，不手改旧发布报告。

## 8. 足量开发与一次性新墙确认（接口已实现，正式大样本未跑）

工程 smoke 通过后，另起新 plan，声明 `smoke_only=false`、至少 3 个不同 continuation seed、每个对手 `evaluation_hands>=4096`。一个可执行的后续预算是 updates=20、hands_per_update=64、fit_steps=400、fit_batch=64、evaluation_hands=4096、environments=64、training cap=1800 秒、evaluation cap=3600 秒。它是待测预算，不是收敛保证；不临时延长失败分支预算，也不为通过门禁改阈值。

先修改新 JSON 中这些字段再运行 study，保留其所有分支/候选；结果没有全部达到原净分/尾部门禁时就停止晋级，继续用 d000000266137。不能把小样本或某个表现最好的 seed 送入确认。

```bash
# DEV 指向已完成、非 smoke 且所有预先规定条件通过的 study。
# 这两个变量仅在上述条件真实满足后设置，不是让当前 smoke 伪装成正式结果。
"$PY" tools/rc5.py freeze --study "$DEV" --output "$OUT/frozen-final" \
  --hands 4096 --max-seconds 21600
"$PY" tools/rc5.py confirm --freeze "$OUT/frozen-final" --baseline "$BASE" \
  --registry "$REG" --device cuda --workers 4 --environments 64 --backend cpp \
  --output "$OUT/independent-confirmation"
```

freeze 固定全部 seed 的候选和同预算原策略哈希/源码/手数。confirm 在冻结后生成新的秘密墙 seed，冻结 ID 先登记消费；失败也算暴露，不能复用冻结 ID 伪称再次未见。新墙只对已登记暴露证明不交叉，历史登记不完整仍明示 partial；对手家族是已知三个家族，不称为未见对手泛化。这里并未实现跨手长期回报或新 opponent-family benchmark。

## 9. 停止、失败与回退

每个 study/probe 输出 `process.json`，只针对该次记录的 pid/pgid。中断当前前台用 Ctrl+C；从另一终端给该次真实 pid 发送 SIGINT。不要 killall/pkill python，不杀别的训练。超时或自然手未完整收束会失败，不截断轨迹冒充完成。

PPO 超过 KL 门禁时先保存真实提案，再把模型和优化器恢复到更新前；候选保存不是“接受”。effective_checkpoint 指向实际有效权重。科学状态 discard/inconclusive 不覆盖原 d000000266137；不自动部署、不自动续长训。失败后保留目录/日志/注册表，开新 run 并用新 seed；本 rc5 没有精确断点续训接口。
