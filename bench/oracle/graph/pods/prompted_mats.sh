#!/usr/bin/env bash
# The prompted baseline (prompted.py) on one MATS L40: an untrained large model (BASE, default Qwen3-32B in FP8)
# revising the search's answers after the verifier's reports for ROUNDS rounds on the first QUESTIONS held-out texts.
# Submitted with mats-run from the pushed HEAD; queued until a GPU frees, then waits until that GPU's memory is free.
#   bench/oracle/graph/pods/prompted_mats.sh HOURS   (env: RUN, BASE, ROUNDS, SAMPLES, QUESTIONS)
cd /Users/user/gam
H=${1:-6.0}
RUN=${RUN:-p1}
N=oracle-graph-$RUN
O=/Users/user/mpd-data/cluster/$N
PY=/mnt/nw/home/s.sauers/rl-venv/bin/python
T='/ephemeral/$USER/'$N  # scratch and caches on the worker's local disk (the admins: /ephemeral/$USER, never /tmp); $USER expands on the cluster
WAIT=${WAIT:-3}  # hours the job may wait for its GPU to be free (gpu_free.py)
CMD="mkdir -p $T $O; export TMPDIR=$T TRITON_CACHE_DIR=$T/triton TORCHINDUCTOR_CACHE_DIR=$T/inductor OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; cd /Users/user/gam/bench/oracle/graph; $PY pods/gpu_free.py --hours $WAIT > $O/gpu_free.log 2>&1 || exit 1; ls /Users/user/mpd-data/graph_oracle/texts > /dev/null; $PY prompted.py --base ${BASE:-Qwen/Qwen3-32B-FP8} --rounds ${ROUNDS:-3} --samples ${SAMPLES:-2} --questions ${QUESTIONS:-50} --out $O > $O/log 2>&1; echo done"
MATS_GPUS=1 MATS_BUILD=0 bench/mats/mats-run $N 8 ${MEM:-56} $(python3 -c "print(round($H + $WAIT, 2))") -- bash -c "$CMD"
