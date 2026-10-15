#!/usr/bin/env bash
set -uo pipefail

RC5="/root/rivermind-data/pdh-zero/active/pdh-zero-v2-trainer-0.7.0-rc5"
EXP3="/root/rivermind-data/pdh-zero/evidence/v2-rc5-autoresearch/AR_rc5-5090-20261005/EXP_003_20261005T071431Z"
MODEL="/root/rivermind-data/pdh-zero/evidence/v2-refinement-rc4.1/REFINE_20261003T131918636050Z/checkpoints/candidates/d000000266137/model.pt"
DATA="/root/rivermind-data/pdh-zero/evidence/v2-rc5-autoresearch/AR_rc5-5090-20261005/inputs/derived/auxiliary-v2-repair-v1/auxiliary-v2.pt"
REG="/root/rivermind-data/pdh-zero/evidence/v2-refinement-rc5/.wall-exposures.sqlite3"

export PYTHONPATH="$RC5/src"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$RC5" || {
    printf '%s\n' 97 > "$EXP3/launcher.exitcode"
    exit 97
}

python3 tools/rc5.py study     --baseline "$MODEL"     --dataset "$DATA"     --registry "$REG"     --config "$EXP3/config/cuda-smoke-plan.json"     --output "$EXP3/study"     > "$EXP3/study.stdout.log" 2>&1

CODE=$?

printf '%s\n' "$CODE"     > "$EXP3/launcher.exitcode"

exit "$CODE"
