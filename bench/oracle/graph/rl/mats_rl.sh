#!/usr/bin/env bash
# The oracle's program training on the MATS cluster (#2951), in ~/rl-venv: vLLM 0.10.2 (the node's driver
# 535 runs CUDA 12.x wheels only) with transformers below 5 (vLLM 0.10.2 under transformers 5.18 fails at
# engine start, bench/runpod/pyenv/oracle.txt), peft and accelerate; built by the first job that needs it.
#   MATS_GPUS=2 mats-run oracle-rl-NAME 16 96 12 -- bash /Users/user/gam/bench/oracle/graph/rl/mats_rl.sh \
#       --mode grpo --base Qwen/Qwen3-8B --model qwen3-0.6b --behaviors /Users/user/mpd-data/graph_oracle/behaviors --out /Users/user/mpd-data/cluster/oracle-rl-NAME [--init ADAPTER] ...
# With two GPUs vLLM samples on the first and the trainer runs on the second; with one, both share it
# (pass --gpu-memory 0.3 for a small policy). Smoke test (one GPU, Qwen3-0.6B as the oracle, mock scorer):
#   MATS_GPUS=1 MATS_QOS=debug mats-run oracle-rl-smoke 8 32 0.5 -- bash /Users/user/gam/bench/oracle/graph/rl/mats_rl.sh \
#       --mode grpo --base Qwen/Qwen3-0.6B --model qwen3-0.6b --behaviors /Users/user/mpd-data/graph_oracle/behaviors --scorer mock \
#       --out /Users/user/mpd-data/cluster/oracle-rl-smoke --steps 3 --behaviors-per-step 4 --samples 8 --max-tokens 512 --lora-rank 16 --gpu-memory 0.3
# Real score on vpd4l (the checker built at the job's commit, 4 servers):
#   MATS_GPUS=1 MATS_QOS=debug mats-run oracle-rl-vpd4l 16 64 2 -- bash /Users/user/gam/bench/oracle/graph/rl/mats_rl.sh \
#       --mode grpo --base Qwen/Qwen3-8B --model vpd4l --behaviors /Users/user/mpd-data/graph_oracle/behaviors --scorer checker \
#       --checker /Users/user/gam/target/release/examples/mpd_graph_2951 --export /Users/user/mpd-data/engine/vpd4l --score-workers 4 \
#       --out /Users/user/mpd-data/cluster/oracle-rl-vpd4l --steps 1000 --hours 1.8 --gpu-memory 0.42 --micro 1 --eval-every 20
set -Eeuo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
py=$HOME/rl-venv/bin/python
(
    flock 9
    if ! "$py" -c "import vllm, peft" 2> /dev/null; then
        export UV_CACHE_DIR=$HOME/.cache/uv
        rm -rf "$HOME/rl-venv"
        ~/.local/bin/uv venv -q --python 3.12 "$HOME/rl-venv"
        ~/.local/bin/uv pip install -q --python "$py" "vllm==0.10.2" "transformers>=4.56,<5" "peft==0.21.2" "accelerate==1.15.0" numpy
    fi
) 9> "$HOME/.rl-venv.lock"
# score.py starts the checker under the Mac's mem-lease unless MEM_LEASE_GIB is set; the cluster has no ledger.
export TOKENIZERS_PARALLELISM=false MEM_LEASE_GIB=${MEM_LEASE_GIB:-0}
exec "$py" "$here/train.py" "$@"
