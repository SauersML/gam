#!/usr/bin/env bash
# Shared job Python (#2951), run once on the cluster as a Slurm job:
#   sbatch -J mpd-venv -p compute --qos=debug -c 8 --mem=16G -t 1:00:00 cluster_venv.sh
# Python 3.12, torch on CUDA 12.6 wheels (driver 535), the VPD/NLA/refusal stacks. The VPD paper code
# (spd-vpd-paper) imports under it once wandb_workspaces is stubbed (0.1.12 needs a wandb_gql that
# current wandb no longer ships).
set -Eeuo pipefail
export UV_CACHE_DIR=$HOME/.cache/uv
new=$HOME/mpd-venv.new
rm -rf "$new"
~/.local/bin/uv venv -q --python 3.12 "$new"
~/.local/bin/uv pip install -q --python "$new/bin/python" --index-url https://download.pytorch.org/whl/cu126 torch torchvision
~/.local/bin/uv pip install -q --python "$new/bin/python" numpy scipy pandas pyarrow pyyaml tqdm matplotlib safetensors \
    transformers tokenizers huggingface_hub datasets wandb einops jaxtyping fire wadler_lindig "pydantic<2.12" \
    python-dotenv sympy zstandard orjson httpx "aiolimiter>=1.2" "numba>=0.64.0" fastapi uvicorn "openrouter>=0.1.1" \
    kaleido==0.2.1 streamlit streamlit-antd-components ipykernel
"$new/bin/python" -c "import torch, numpy, scipy, transformers, pydantic, jaxtyping; print(torch.__version__, torch.version.cuda, pydantic.__version__)"
[ -d "$HOME/mpd-venv" ] && { rm -rf "$HOME/mpd-venv.old"; mv "$HOME/mpd-venv" "$HOME/mpd-venv.old"; }
mv "$new" "$HOME/mpd-venv"
sed -i "s|$new|$HOME/mpd-venv|g" "$HOME"/mpd-venv/bin/* 2>/dev/null || true
"$HOME/mpd-venv/bin/python" -c "import torch; print(\"ok\", torch.__file__)"
