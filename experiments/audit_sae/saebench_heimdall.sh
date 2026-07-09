#!/bin/bash -l
# #1942 SAEBench pass on the datasci war-machine nodes (node1 = 8xB200 with the
# k25 data + SAE checkpoints + torch venv). heimdall is MANDATORY here — the
# cluster is shared; never raw nohup. This wrapper submits the same driver the
# MSI sbatch runs, through `heimdall submit` with VRAM sharing.
#
# Usage (override any path via the env):
#   PY=/models/wm/torchvenv/bin/python \
#   DRIVER=/models/wm/gam/experiments/audit_sae/saebench_eval.py \
#   SAEBENCH_SAE=/models/wm/saes/deepseek_v3_l40_topk.safetensors \
#   SAEBENCH_MODEL=deepseek-v3 SAEBENCH_ARCH=topk SAEBENCH_K=64 \
#   SAEBENCH_HOOK_LAYER=40 OUT_DIR=/models/wm/scratch/saebench_1942 \
#   ./saebench_heimdall.sh
#
# gamfit must already be importable in $PY's venv (node1/node2 have no GitHub
# access — deploy the wheel out of band). The public SAEBench suite still SKIPS
# cleanly unless the sae_bench datasets + a compatible host model are present;
# the manifold-native arms run from whatever ledgers exist.

set -uo pipefail

# heimdall runs inside the shared venv; unset the MSI RUSTFLAGS saboteur if it
# leaked into the environment (defensive — this job never compiles).
unset RUSTFLAGS

PY=${PY:-/models/wm/torchvenv/bin/python}
DRIVER=${DRIVER:-/models/wm/gam/experiments/audit_sae/saebench_eval.py}
OUT_DIR=${OUT_DIR:-/models/wm/scratch/saebench_1942}
NODE=${NODE:-node1}
GPUS=${GPUS:-1}
VRAM=${VRAM:-40}
ESTIMATED=${ESTIMATED:-120}
NAME=${NAME:-saebench-1942}

DOSE_LEDGER=${DOSE_LEDGER:-/models/wm/msae_l17/dose_calibration_real.json}
CHART_FIT=${CHART_FIT:-/models/wm/msae_l17/chart_fit_month_l17.json}
DECODER=${DECODER:-/models/wm/msae_l17/decoder.npy}
ACTIVATIONS=${ACTIVATIONS:-/models/wm/msae_l17/activations.npy}

SAEBENCH_SAE=${SAEBENCH_SAE:-}
SAEBENCH_MODEL=${SAEBENCH_MODEL:-}
SAEBENCH_EVALS=${SAEBENCH_EVALS:-absorption,scr,unlearning,sparse_probing}
SAEBENCH_ARCH=${SAEBENCH_ARCH:-topk}
SAEBENCH_K=${SAEBENCH_K:-64}
SAEBENCH_HOOK_LAYER=${SAEBENCH_HOOK_LAYER:-40}
SAEBENCH_DEVICE=${SAEBENCH_DEVICE:-cuda:0}
SAEBENCH_DTYPE=${SAEBENCH_DTYPE:-bfloat16}

# Build the argument string exactly as the MSI sbatch does, adding each arm only
# if its inputs exist so nothing is fabricated.
ARGS="--out $OUT_DIR/saebench_1942_report.json"
[ -f "$DOSE_LEDGER" ]  && ARGS="$ARGS --dose-ledger $DOSE_LEDGER"
[ -f "$CHART_FIT" ]    && ARGS="$ARGS --chart-fit $CHART_FIT"
{ [ -f "$DECODER" ] && [ -f "$ACTIVATIONS" ]; } && ARGS="$ARGS --decoder $DECODER --activations $ACTIVATIONS"
if [ -n "$SAEBENCH_SAE" ] && [ -n "$SAEBENCH_MODEL" ] && [ -f "$SAEBENCH_SAE" ]; then
  ARGS="$ARGS --saebench-sae $SAEBENCH_SAE --saebench-model $SAEBENCH_MODEL"
  ARGS="$ARGS --saebench-evals $SAEBENCH_EVALS --saebench-arch $SAEBENCH_ARCH --saebench-k $SAEBENCH_K"
  ARGS="$ARGS --saebench-hook-layer $SAEBENCH_HOOK_LAYER --saebench-device $SAEBENCH_DEVICE"
  ARGS="$ARGS --saebench-dtype $SAEBENCH_DTYPE --saebench-output-dir $OUT_DIR/saebench_out"
fi

# Deploy-skew preflight embedded in the submitted command: fail inside the
# heimdall slot BEFORE any model load if the venv's gamfit lacks a #1942 metric
# binding the driver calls (node1/node2 have no GitHub — the wheel is deployed
# out of band and can lag).
PREFLIGHT="$PY -c \"import os,sys,gamfit; m=[n for n in ('chart_interp_score','dose_response_calibration','audit_sae') if not hasattr(gamfit,n)]; sys.exit('gamfit '+getattr(gamfit,'__version__','?')+' at '+os.path.dirname(gamfit.__file__)+' missing '+repr(m)+'; redeploy the wheel with the #1942 SAEBench scorers') if m else print('preflight OK')\""
CMD="mkdir -p $OUT_DIR && unset RUSTFLAGS && $PREFLIGHT && $PY -u $DRIVER $ARGS"
echo "=== heimdall submit ($NODE, ${GPUS}gpu ${VRAM}GB): $CMD ==="
heimdall submit "$CMD" \
  --type custom --gpus "$GPUS" --vram "$VRAM" --node "$NODE" \
  --name "$NAME" --estimated "$ESTIMATED"
