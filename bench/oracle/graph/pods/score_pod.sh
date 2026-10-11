#!/usr/bin/env bash
# Score search answers with their baselines on one pod (prompted.py --rounds 0: the search's answer cut to the oracle's
# output budget, its whole answer, the empty answer, VPD's ranked answer), one run per SPLIT:DIR pair, each into
# $O/<name> where name is the pair's third field. DIR is a directory under ~/mpd-data (Mac path), sent up as a tar file
# with the questions and the source (rp-run uploads a directory file by file); VPD's checkpoint (its causal-importance
# network, 2.9 GB) goes up for VPD's answer. The Python is origin/main's bench/oracle/graph at launch.
#   bench/oracle/graph/pods/score_pod.sh HOURS "hard:/Users/user/mpd-data/X:name ..."   (env: RUN, P, PRICE, VRAM, GPU)
cd /Users/user/gam
H=${1:-1.5}
PAIRS=$2
RUN=${RUN:-a} P=${P:-6}
S=/Users/user/mpd-data/scratch/glead/v5/src_score_$RUN
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive origin/main bench/oracle/graph bench/vpd_2951/vpd_model.py | tar -x -C "$S" && echo "python source $(git rev-parse --short=10 origin/main)" > "$S/SOURCE"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"
G=$S/bench/oracle/graph
N=oracle-graph-score-$RUN
O=/Users/user/mpd-data/runpod/$N; W=/workspace/runs/$N
T=/Users/user/mpd-data/graph_oracle/texts-score-$RUN.tar
A=/Users/user/mpd-data/graph_oracle/answers-score-$RUN.tar
COPYFILE_DISABLE=1 tar -cf "$T" -C /Users/user/mpd-data/graph_oracle texts/vpd4l
DIRS=""
STAGES=""
for pair in $PAIRS; do
  IFS=: read -r split dir name <<< "$pair"
  rel=${dir#/Users/user/mpd-data/}
  DIRS="$DIRS $rel"
  STAGES="$STAGES mkdir -p $O/$name; python prompted.py --split $split --rounds 0 --questions 50 --workers $P --search-dir ~/mpd-data/$rel --out $O/$name > $O/$name/log 2>&1;"
done
COPYFILE_DISABLE=1 tar -cf "$A" -C /Users/user/mpd-data $DIRS
VPD_IN=/Users/user/mpd-data/vpd/s-55ea3f9b/model_400000.pth
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O $S ~/mpd-data/graph_oracle; tar -xf $T -C ~/mpd-data/graph_oracle; tar -xf $A -C ~/mpd-data; tar -xf $S.tar -C $S; ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors $VPD_IN > /dev/null; cd $G; $STAGES echo done"
RP_UPLOAD_MAX_MB=${RP_UPLOAD_MAX_MB:-3500} RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_MAX_PRICE=${PRICE:-0.90} RP_MIN_VRAM_GB=${VRAM:-44} RP_MIN_RAM_GB=64 \
  bench/runpod/rp-run $N "${GPU:-auto}" $H -- bash -c "$CMD" > /Users/user/mpd-data/scratch/glead/v5/rp_$N.log 2>&1
echo "$N exit $?"
