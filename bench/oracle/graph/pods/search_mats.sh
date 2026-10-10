#!/usr/bin/env bash
# Bootstrap search answers on MATS L40s (free): JOBS jobs of one GPU each, P search processes per job, all sharing
# SPLIT's first COUNT texts through claim files (native.py search; NODES_FROM=vpd: candidates in VPD's ranked order,
# VPD's checkpoint being on the cluster). Answers to /Users/user/mpd-data/cluster/oracle-graph-search-RUN/<search dir>.
# A job whose GPU stays under half free for WAIT hours (another process on it) ends without claiming any text, so the
# others still cover every text. RANKED=K: the first job first writes VPD's first K subcomponents at the target of every
# text (native.py ranked) to $O/vpd_ranked. Scratch and caches in /ephemeral/$USER.
#   bench/oracle/graph/pods/search_mats.sh HOURS   (env: RUN, SPLIT, COUNT, JOBS, P, NODES_FROM, RANKED, WAIT, MEM)
cd /Users/user/gam
H=${1:-6.0}
RUN=${RUN:-s1} SPLIT=${SPLIT:-train} COUNT=${COUNT:-400} JOBS=${JOBS:-2} P=${P:-2}
N=oracle-graph-search-$RUN
O=/Users/user/mpd-data/cluster/$N
OUT=$O/$([ "$SPLIT" = train ] && echo search || echo search_$SPLIT)
PY=/mnt/nw/home/s.sauers/rl-venv/bin/python
T='/ephemeral/$USER/'$N
WAIT=${WAIT:-1}
for j in $(seq 0 $((JOBS - 1))); do
  PROCS=""
  for k in $(seq 0 $((P - 1))); do
    PROCS="$PROCS ($PY native.py search --split $SPLIT --n $COUNT --nodes-from ${NODES_FROM:-ig} --out $OUT > $O/${SPLIT}_${j}_$k.log 2>&1) &"
  done
  [ "$j" = 0 ] && [ -n "$RANKED" ] && PROCS="$PY native.py ranked --split train heldout hard --n 100000 --top $RANKED --out $O/vpd_ranked > $O/ranked.log 2>&1; $PROCS"
  CMD="mkdir -p $T $OUT; export TMPDIR=$T TRITON_CACHE_DIR=$T/triton TORCHINDUCTOR_CACHE_DIR=$T/inductor OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; cd /Users/user/gam/bench/oracle/graph; $PY pods/gpu_free.py --hours $WAIT --share 0.5 > $O/gpu_free_$j.log 2>&1 || exit 1; ls /Users/user/mpd-data/graph_oracle/texts/vpd4l > /dev/null; $PROCS wait; echo done"
  MATS_GPUS=1 MATS_BUILD=0 bench/mats/mats-run $N-$j 8 ${MEM:-48} $(python3 -c "print(round($H + $WAIT, 2))") -- bash -c "$CMD"
done
