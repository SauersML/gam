#!/usr/bin/env bash
# The oracle's program training on the MATS cluster (#2951), in ~/oracle-venv (vLLM 0.10.2 + peft):
#   MATS_GPUS=2 mats-run oracle-rl-NAME 16 96 12 -- bash /Users/user/gam/bench/oracle/graph/rl/mats_rl.sh \
#       --mode grpo --base Qwen/Qwen3-8B --model qwen3-0.6b --out /Users/user/mpd-data/cluster/oracle-rl-NAME [--init ADAPTER] ...
# With two GPUs vLLM samples on the first and the trainer runs on the second; with one, both share it
# (pass --gpu-memory 0.3 for a small policy). Smoke test (one GPU, Qwen3-0.6B as the oracle, mock scorer):
#   MATS_GPUS=1 MATS_QOS=debug mats-run oracle-rl-smoke 8 32 0.5 -- bash /Users/user/gam/bench/oracle/graph/rl/mats_rl.sh \
#       --mode grpo --base Qwen/Qwen3-0.6B --model vpd4l --behaviors /Users/user/mpd-data/scratch/g-rl/behaviors --scorer mock \
#       --out /Users/user/mpd-data/cluster/oracle-rl-smoke --steps 3 --behaviors-per-step 1 --samples 8 --max-tokens 512 --lora-rank 16 --gpu-memory 0.3
set -Eeuo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
py=$HOME/oracle-venv/bin/python
"$py" -c "import peft" 2> /dev/null || ~/.local/bin/uv pip install -q --python "$py" peft accelerate
export VLLM_WORKER_MULTIPROC_METHOD=spawn TOKENIZERS_PARALLELISM=false
exec "$py" "$here/train.py" "$@"
