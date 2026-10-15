#!/usr/bin/env bash
set -uo pipefail

RC5="/root/rivermind-data/pdh-zero/active/pdh-zero-v2-trainer-0.7.0-rc5"
EXP="/root/rivermind-data/pdh-zero/evidence/v2-rc5-autoresearch/AR_rc5-5090-20261005/EXP_002_20261005T065316Z"
MODEL="/root/rivermind-data/pdh-zero/evidence/v2-refinement-rc4.1/REFINE_20261003T131918636050Z/checkpoints/candidates/d000000266137/model.pt"
CAND="/root/rivermind-data/pdh-zero/evidence/v2-rc5-autoresearch/AR_rc5-5090-20261005/inputs/derived/probe-candidate-repair-v1/probe-candidate-seed-132846611674848.pt"
NEW_REG="/root/rivermind-data/pdh-zero/evidence/v2-refinement-rc5/.wall-exposures.sqlite3"

export PYTHONPATH="$RC5/src"

cd "$RC5" || {
    printf '%s\n' 97 > "$EXP/launcher.exitcode"
    exit 97
}

python3 tools/probe_rc5.py     --baseline "$MODEL"     --candidate "$CAND"     --registry "$NEW_REG"     --device cuda     --workers 4     --environments 64     --hands 64     --transport shared     --max-seconds 600     --output "$EXP/probe"     > "$EXP/probe.stdout.log" 2>&1

CODE=$?

printf '%s\n' "$CODE"     > "$EXP/launcher.exitcode"

exit "$CODE"
