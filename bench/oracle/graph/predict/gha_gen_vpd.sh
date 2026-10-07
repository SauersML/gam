#!/usr/bin/env bash
# vpd4l prediction data on a GitHub Actions runner (#2951): CPU, public inputs only (VPD's published target
# t-9d2b8f02 and subcomponents, Pile training rows), fetched from the rp-run-data release by sha256:
#   bench/runpod/gha-run predict-vpd-gha-K -- 'bash bench/oracle/graph/predict/gha_gen_vpd.sh K 4096'
# Task K reads Pile rows K*TEXTS.. of predict/vpd4l/pile_train_rows.npy (MATS's vpd4l array holds tasks 0-15).
set -Eeuo pipefail
here=$(cd "$(dirname "$0")" && pwd)
TASK=$1 TEXTS=$2
A=https://github.com/SauersML/gam/releases/download/rp-run-data
W=$RUNNER_TEMP/vpd
mkdir -p "$W/t"
fetch() { curl -sSfL --retry 3 -o "$2" "$A/$1"; echo "$1  $2" | sha256sum -c --quiet; }
fetch  "$W/t/model_config.yaml"
fetch 9664c12d3492ee58520f89703e67ea2790ea13de1f88bf8e3c4594943e0cc59d "$W/t/model_step_99999.safetensors"
fetch 155e0e9ee8f899ab100bdb84224751308ae86ed57f746dbea403b93276a96552 "$W/t/tokenizer.json"
fetch d4c0d99d84af59e9126913fafe5210822963e9a3065ee43e6833b358b0c2f825 "$W/uv.safetensors"
fetch 3caf2b0b16beb91ee261ce63bfc2b8df1f2e3976db80b62cc85d1f6ae58ee118 "$W/rows.npy"
python3 -m venv "$RUNNER_TEMP/predict-venv"
export PATH="$RUNNER_TEMP/predict-venv/bin:$PATH"
python3 -m pip install -q --upgrade pip
python3 -m pip install -q torch --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install -q safetensors numpy pyyaml tokenizers
python3 "$here/generate.py" --target vpd4l --vpd-target "$W/t" --uv "$W/uv.safetensors" --windows "$W/rows.npy" \
    --texts "$TEXTS" --offset $((TASK * TEXTS)) --seed "$TASK" --split train --batch 32 --out "$OUT/train_$TASK.jsonl"
