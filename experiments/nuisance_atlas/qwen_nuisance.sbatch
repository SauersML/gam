#!/bin/bash
#SBATCH --job-name=nuis_atlas
#SBATCH --account=hsiehph
#SBATCH --partition=msismall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=00:40:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/nuis_atlas_%j.log

set -euo pipefail
# MSI's real $HOME (/users/0/sauer354) is a tiny quota; force HOME + all caches
# onto /projects so the tokenizer loads from the cache the login run populated
# (SLURM does not inherit the msi wrapper's env). Offline: no network on compute.
export HOME=/projects/standard/hsiehph/sauer354
export XDG_CACHE_HOME=/projects/standard/hsiehph/sauer354/.cache
export HF_HOME=/projects/standard/hsiehph/sauer354/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
PY=/projects/standard/hsiehph/sauer354/saevenv/bin/python
HARV=/projects/standard/hsiehph/sauer354/harvest_out/qwen3_8b_wikitext
OUT=/projects/standard/hsiehph/sauer354/qwen_nuisance_atlas.json

POS=/projects/standard/hsiehph/sauer354/qwen_positions.npy
echo "[nuis] node $(hostname) start $(date)"
# Positions are precomputed on the login node (network) into $POS; the compute
# node needs no tokenizer/dataset/network — pure numpy over the banks.
"$PY" /projects/standard/hsiehph/sauer354/qwen_nuisance_msi.py \
  --harvest "$HARV" --layers 18,30,6 --positions "$POS" \
  --harmonics 16 --early-k 8 --out "$OUT"
echo "[nuis] done $(date)"
echo "[nuis] RESULT FILE: $OUT"
cat "$OUT"
