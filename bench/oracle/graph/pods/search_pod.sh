#!/usr/bin/env bash
# Bootstrap search answers on one pod (10-09 night): native.py search for the first COUNT texts of SPLIT (train: SFT data;
# heldout, hard: the evaluation's search baselines), P processes sharing the GPU (vpd4l is small). The Python is origin/main's
# bench/oracle/graph at launch.
# SPLIT may list several splits ("heldout:50 train:124"): each split's first COUNT (or its own count) texts, one after another. The questions and
# the source go up as tar files (rp-run uploads a directory file by file).
#   bench/oracle/graph/pods/search_pod.sh HOURS   (env: RUN, SPLIT, COUNT, P, PRICE, VRAM, GPU)
cd /Users/user/gam
H=${1:-3.0}
RUN=${RUN:-a} SPLIT=${SPLIT:-train} COUNT=${COUNT:-400} P=${P:-6}
S=/Users/user/mpd-data/scratch/glead/v5/src_search_$RUN
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive origin/main bench/oracle/graph bench/vpd_2951/vpd_model.py | tar -x -C "$S" && echo "python source $(git rev-parse --short=10 origin/main)" > "$S/SOURCE"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"  # rp-run uploads every existing path the command names: only the archives exist here
G=$S/bench/oracle/graph
N=oracle-graph-search-$RUN
O=/Users/user/mpd-data/runpod/$N; W=/workspace/runs/$N
T=/Users/user/mpd-data/graph_oracle/texts-search-$RUN.tar
COPYFILE_DISABLE=1 tar -cf "$T" -C /Users/user/mpd-data/graph_oracle texts/vpd4l
STAGES=""
for entry in $SPLIT; do
  split=${entry%%:*}; n=$([ "$entry" = "$split" ] && echo $COUNT || echo ${entry#*:})  # "train:124" sets that split's count
  out=$([ "$split" = train ] && echo search || echo search_$split)
  JOBS=""
  for k in $(seq 0 $((P - 1))); do
    JOBS="$JOBS (python native.py search --split $split --n $n --offset $k --stride $P --out $O/$out > $O/${split}_$k.log 2>&1) &"
  done
  STAGES="$STAGES $JOBS wait;"
done
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O $S ~/mpd-data/graph_oracle; tar -xf $T -C ~/mpd-data/graph_oracle; tar -xf $S.tar -C $S; ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors > /dev/null; cd $G; $STAGES echo done"
RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.80} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=64 \
  bench/runpod/rp-run $N "${GPU:-auto}" $H -- bash -c "$CMD" > /Users/user/mpd-data/scratch/glead/v5/rp_$N.log 2>&1
echo "$N exit $?"
