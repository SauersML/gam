#!/usr/bin/env bash
# prime-rl RL of the graph oracle (#2951) on one RunPod GPU (80 GB: vLLM, serve_scores and the trainer share it): the
# SFT adapter, the questions (questions.py's train.jsonl and heldout.jsonl), the texts and origin/main's
# bench/oracle/graph go up as tar files; VPD's subcomponents (uv.safetensors) and vpd4l are published inputs; Qwen3-4B
# comes from Hugging Face. rl/prime/pod_rl.sh runs there; its outputs come back to mpd-data/runpod/NAME.
#   bench/oracle/graph/rl/prime/run_pod.sh ADAPTER_DIR QUESTIONS_DIR HOURS [rl args]   (env: RUN, REVISE=1, PRICE, GPU)
cd /Users/user/gam
AD=$1 QD=$2 H=$3
shift 3
RUN=${RUN:-a}
N=oracle-graph-prime-$RUN
O=/Users/user/mpd-data/runpod/$N; W=/workspace/runs/$N
S=/Users/user/mpd-data/scratch/primerl/src_$RUN
mkdir -p "$(dirname "$S")"
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive origin/main bench/oracle/graph bench/vpd_2951/vpd_model.py | tar -x -C "$S" && echo "python source $(git rev-parse --short=10 origin/main)" > "$S/SOURCE"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"  # rp-run uploads every existing path the command names: only the archives exist here
IN=/Users/user/mpd-data/scratch/primerl/in_$RUN.tar
COPYFILE_DISABLE=1 tar -cf "$IN" -C /Users/user/mpd-data/graph_oracle texts/vpd4l -C "$(dirname "$AD")" "$(basename "$AD")" -C "$(dirname "$QD")" "$(basename "$QD")"
I='~/mpd-data/in'  # the inputs on the pod (unpacked from $IN; a Mac path here would be uploaded file by file)
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O $S ~/mpd-data/in ~/mpd-data/graph_oracle; tar -xf $IN -C ~/mpd-data/in; mv ~/mpd-data/in/texts ~/mpd-data/graph_oracle/; tar -xf $S.tar -C $S; \
ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors > /dev/null; \
cd $S/bench/oracle/graph; ${REVISE:+REVISE=1} bash rl/prime/pod_rl.sh $I/$(basename "$AD") $I/$(basename "$QD") $O $N $*"
RP_ALLOW_OVER=1 RP_UPLOAD_MAX_MB=${RP_UPLOAD_MAX_MB:-3500} RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_HF_MODELS="Qwen/Qwen3-4B" RP_MAX_PRICE=${PRICE:-2.0} \
  RP_MIN_VRAM_GB=80 RP_MIN_RAM_GB=96 RP_DISK_GB=120 bench/runpod/rp-run $N "${GPU:-auto}" $H -- bash -c "$CMD"
