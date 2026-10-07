#!/usr/bin/env bash
# reader_score.py score on MATS (#2951): one L40 (or CPUs only, float32), Qwen3-8B (READER_MODEL), the venv of
# rl/mats_rl.sh (built when missing, same lock) or READER_PY.
#   MATS_GPUS=1 MATS_QOS=debug mats-run graph-reader-cal 8 48 1 -- bash /Users/user/gam/bench/oracle/graph/reader_mats.sh \
#       PROGRAMS.jsonl ITEMS.jsonl /Users/user/mpd-data/cluster/graph-reader-cal/OUT.json vpd4l
set -Eeuo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROGRAMS=$1 ITEMS=$2 OUT=$3 TARGET=${4:-vpd4l}
py=${READER_PY:-$HOME/rl-venv/bin/python}  # READER_PY: an existing venv with torch and transformers (a CPU job needs no vLLM)
[ -n "${READER_PY:-}" ] || (
    flock 9
    if ! "$py" -c "import vllm, peft" 2> /dev/null; then
        export UV_CACHE_DIR=$HOME/.cache/uv
        rm -rf "$HOME/rl-venv"
        ~/.local/bin/uv venv -q --python 3.12 "$HOME/rl-venv"
        ~/.local/bin/uv pip install -q --python "$py" "vllm==0.10.2" "transformers>=4.56,<5" "peft==0.21.2" "accelerate==1.15.0" numpy
    fi
) 9> "$HOME/.rl-venv.lock"
export TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1
mkdir -p "$(dirname "$OUT")"
nvidia-smi --query-gpu=name,memory.total --format=csv 2> /dev/null || echo "no GPU: CPU, float32, ${SLURM_CPUS_PER_TASK:-?} threads"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
"$py" "$here/reader_score.py" score --model "${READER_MODEL:-Qwen/Qwen3-8B}" --target "$TARGET" --programs "$PROGRAMS" --items "$ITEMS" --out "$OUT" ${READER_DTYPE:+--dtype $READER_DTYPE} ${READER_ARGS:-}
