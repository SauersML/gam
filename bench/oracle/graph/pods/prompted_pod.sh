#!/usr/bin/env bash
# The prompted baseline (prompted.py) on one RunPod GPU of 48 GB: an untrained large model (BASE, default Qwen3-32B in
# FP8) revising the search's answers after the verifier's reports. A pod has no VPD importance network, so VPD's answer
# is scored elsewhere. The Python is origin/main's bench/oracle/graph at launch.
#   bench/oracle/graph/pods/prompted_pod.sh HOURS   (env: RUN, BASE, ROUNDS, SAMPLES, QUESTIONS, GPU, PRICE)
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
CMD="ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O; ls $S/bench/vpd_2951/vpd_model.py /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors $B/vpd4l $B/search_heldout > /dev/null; cd $G; python prompted.py --base $BASE --rounds ${ROUNDS:-3} --samples ${SAMPLES:-2} --questions ${QUESTIONS:-50} --out $O > $O/log 2>&1; echo done"
RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_HF_MODELS="$BASE" RP_MAX_PRICE=${PRICE:-0.80} RP_MIN_VRAM_GB=44 RP_MIN_RAM_GB=96 \
  bench/runpod/rp-run $N "${GPU:-NVIDIA RTX A6000,NVIDIA A40,NVIDIA L40S,NVIDIA L40,NVIDIA RTX 6000 Ada Generation}" $H -- bash -c "$CMD" > /Users/user/mpd-data/scratch/glead/v5/rp_$N.log 2>&1
echo "$N exit $?"
