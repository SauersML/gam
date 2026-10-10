#!/usr/bin/env bash
# The prompted baseline (prompted.py) on one RunPod GPU: an untrained large model (BASE, default Qwen3-32B in FP8 on a
# 48 GB card) revising the search's answers after the verifier's reports. A pod has no VPD importance network, so
# VPD's answer is scored elsewhere: PREV, an eval_samples.jsonl with the baselines already scored (prompted.py on the
# Mac or MATS), is copied into the output directory first and reused. The Python is origin/main's bench/oracle/graph
# at launch.
#   bench/oracle/graph/pods/prompted_pod.sh HOURS   (env: RUN, BASE, ROUNDS, SAMPLES, QUESTIONS, ARGS, PREV, GPU, VRAM, RAM, PRICE)
#   e.g. BASE=QuantTrio/Qwen3-30B-A3B-Thinking-2507-AWQ VRAM=30 GPU="NVIDIA GeForce RTX 5090" PRICE=0.70 \
#        PREV=~/mpd-data/graph_oracle/prompted_mac1/eval_samples.jsonl ARGS="--evidence-tokens 3000 --max-tokens 16000 --gpu-memory 0.8"
cd /Users/user/gam
H=${1:-6.0}
RUN=${RUN:-pp1}
BASE=${BASE:-Qwen/Qwen3-32B-FP8}
S=/Users/user/mpd-data/scratch/glead/v5/src_prompted_$RUN
git fetch -q origin main && rm -rf "$S" && mkdir -p "$S" && git archive origin/main bench/oracle/graph bench/vpd_2951/vpd_model.py | tar -x -C "$S" && echo "python source $(git rev-parse --short=10 origin/main)" > "$S/SOURCE"
G=$S/bench/oracle/graph
B=/Users/user/mpd-data/graph_oracle/texts
N=oracle-graph-$RUN
O=/Users/user/mpd-data/runpod/$N; W=/workspace/runs/$N
CMD="ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O; ${PREV:+cp $PREV $O/eval_samples.jsonl;} ls $S/bench/vpd_2951/vpd_model.py /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors $B/vpd4l $B/search_heldout > /dev/null; cd $G; python prompted.py --base $BASE --rounds ${ROUNDS:-3} --samples ${SAMPLES:-2} --questions ${QUESTIONS:-50} ${ARGS:-} --out $O > $O/log 2>&1; echo done"
RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_HF_MODELS="$BASE" RP_MAX_PRICE=${PRICE:-0.80} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=${RAM:-96} \
  bench/runpod/rp-run $N "${GPU:-NVIDIA RTX A6000,NVIDIA A40,NVIDIA L40S,NVIDIA L40,NVIDIA RTX 6000 Ada Generation}" $H -- bash -c "$CMD" > /Users/user/mpd-data/scratch/glead/v5/rp_$N.log 2>&1
echo "$N exit $?"
