#!/usr/bin/env bash
# The reader service's Python on the MATS cluster (#2951), run once as a CPU job:
#   MATS_QOS=debug MATS_REF=<commit> mats-run oracle-venv 8 16 1 -- bash /Users/user/gam/bench/oracle/mats_venv.sh
# ~/oracle-venv: vLLM with the torch it pins (kept apart from ~/mpd-venv, whose torch other jobs use),
# and the reader's weights (Qwen/Qwen3-8B) in the cluster's Hugging Face cache.
set -Eeuo pipefail
export UV_CACHE_DIR=$HOME/.cache/uv
new=$HOME/oracle-venv.new
rm -rf "$new"
~/.local/bin/uv venv -q --python 3.12 "$new"
~/.local/bin/uv pip install -q --python "$new/bin/python" vllm numpy
"$new/bin/python" -c "import vllm, torch, transformers; print('vllm', vllm.__version__, 'torch', torch.__version__, torch.version.cuda, 'transformers', transformers.__version__)"
[ -d "$HOME/oracle-venv" ] && { rm -rf "$HOME/oracle-venv.old"; mv "$HOME/oracle-venv" "$HOME/oracle-venv.old"; }
mv "$new" "$HOME/oracle-venv"
sed -i "s|$new|$HOME/oracle-venv|g" "$HOME"/oracle-venv/bin/* 2> /dev/null || true
"$HOME/oracle-venv/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('Qwen/Qwen3-8B'))"
