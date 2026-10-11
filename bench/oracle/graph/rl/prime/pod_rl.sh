#!/bin/bash
# prime-rl RL of the graph oracle (#2951) on one RunPod GPU (run_pod.sh launches it): the registry built from VPD's
# subcomponents, the oracle's SFT adapter exported to an HF checkpoint (export_hf.py), prime-rl installed at the commit
# tested on MATS, serve_scores.py (two scoring processes) and prime-rl's vLLM and trainer all on the one GPU (the GPU
# listed twice, filesystem weight broadcast). Run from the source's bench/oracle/graph with the oracle environment's
# python first on PATH:
#   rl/prime/pod_rl.sh ADAPTER QUESTIONS OUT NAME [rl args]    (env: REVISE=1 for the two-turn episode)
# QUESTIONS: the directory holding questions.py's train.jsonl and heldout.jsonl. OUT gets the logs, metrics, trace
# stream and configs (copied every minute), gpu.csv, serve_scores.log and the last adapter; checkpoints stay on the pod.
set -u
ADAPTER=$1 Q=$2 O=$3 NAME=$4
shift 4
G=$PWD
E=/workspace/prime
PY=$(command -v python)
mkdir -p $E $O
log() { echo "$(date +%T) $*"; }

log "registry and export"
$PY part_tokens.py build --vpd ~/mpd-data/oracle/vpd/uv.safetensors --out $E/reg.safetensors > $O/export.log 2>&1
log "registry sha256 $(sha256sum $E/reg.safetensors | cut -c1-16) (the one the MATS SFT runs used: cf1cdce3808edb03)"
$PY rl/export_hf.py $ADAPTER $E/reg.safetensors $E/model >> $O/export.log 2>&1 &
EXPORT=$!

log "prime-rl"
export PATH=$HOME/.local/bin:$PATH UV_CACHE_DIR=$E/uv-cache
command -v uv > /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null
git clone -q https://github.com/PrimeIntellect-ai/prime-rl.git $E/prime-rl
git -C $E/prime-rl checkout -q 41a8ed01908100898ad411f4d06a4236780d7629
(cd $E/prime-rl && GIT_CONFIG_COUNT=1 GIT_CONFIG_KEY_0=url.https://github.com/.insteadOf GIT_CONFIG_VALUE_0=git@github.com: \
  git submodule update -q --init -- deps/verifiers deps/renderers deps/prime-envs deps/pydantic-config && uv sync -q --all-extras) > $O/install.log 2>&1 || { log "prime-rl install failed"; exit 1; }
# torch's CUDA 13.0 build needs a CUDA 13 driver; an older one runs it through NVIDIA's forward-compatibility libcuda.
driver=$(nvidia-smi | grep -o "CUDA Version: [0-9.]*" | grep -o "[0-9.]*$")
if [ "${driver%%.*}" -lt 13 ]; then
  curl -sSL -o $E/compat.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-compat-13-0_580.178.04-1ubuntu1_amd64.deb
  dpkg-deb -x $E/compat.deb $E/compat
  export LD_LIBRARY_PATH=$E/compat/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
fi
log "driver CUDA $driver; $($E/prime-rl/.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available())')"
wait $EXPORT || { log "export failed"; tail -5 $O/export.log; exit 1; }
log "exported: $(tail -n 1 $O/export.log)"

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader -l 5 > $O/gpu.csv &
SMI=$!
# serve_scores in its own process group, so its scoring processes end with it.
setsid $PY rl/serve_scores.py --port 8765 --workers 2 --texts ~/mpd-data/graph_oracle/texts/vpd4l > $O/serve_scores.log 2>&1 &
SCORER=$!
R=$E/runs/$NAME
sync_out() { rsync -a --exclude checkpoints --exclude broadcasts --exclude weights --exclude rollouts $R/ $O/run/ 2> /dev/null; }
(while sleep 60; do sync_out; done) &
SYNC=$!
trap 'kill -TERM -- -$SCORER 2> /dev/null; kill $SMI $SYNC 2> /dev/null' EXIT
until curl -s localhost:8765/health | grep -q '"alive": 2'; do kill -0 $SCORER 2> /dev/null || { log "serve_scores died"; exit 4; }; sleep 5; done
log "serve_scores up"

source $E/prime-rl/.venv/bin/activate
export PYTHONPATH=$G/rl/prime PRL_OUTPUT_DIR=$E/runs WANDB_MODE=disabled XDG_CACHE_HOME=$E/cache VLLM_CACHE_ROOT=$E/cache/vllm TORCHINDUCTOR_CACHE_DIR=$E/cache/inductor
cd $Q
start=$(date +%s)
# One GPU listed twice: the launcher gives vLLM and the trainer the same device; NCCL cannot broadcast between two
# processes on one GPU, so weights go through the filesystem.
CUDA_VISIBLE_DEVICES=0,0 rl @ $G/rl/prime/rl.toml ${REVISE:+@ $G/rl/prime/revise.toml} --model.name $E/model --run.name $NAME --no-monitors.wandb \
  --deployment.num-infer-gpus 1 --weight-broadcast.type filesystem --inference.vllm.gpu-memory-utilization 0.35 \
  --env-vars "{\"TRITON_CACHE_DIR\": \"$E/cache/triton\"}" "$@"
log "rl exit $? after $(( $(date +%s) - start )) s"
sync_out
last=$(ls -v $R/broadcasts 2> /dev/null | tail -n 1)
[ -n "$last" ] && cp -r $R/broadcasts/$last $O/adapter
