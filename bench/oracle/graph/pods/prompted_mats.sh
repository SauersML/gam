#!/usr/bin/env bash
# The prompted baseline (prompted.py) on one MATS L40: an untrained large model (BASE, default Qwen3-32B in FP8)
# revising the search's answers after the verifier's reports for ROUNDS rounds on the first QUESTIONS held-out texts.
# Submitted with mats-run from the pushed HEAD; queued until a GPU frees.
#   bench/oracle/graph/pods/prompted_mats.sh HOURS   (env: RUN, BASE, ROUNDS, SAMPLES, QUESTIONS)
cd /Users/user/gam
H=${1:-6.0}
RUN=${RUN:-p1}
N=oracle-graph-$RUN
O=/Users/user/mpd-data/cluster/$N
PY=/mnt/nw/home/s.sauers/rl-venv/bin/python
CMD="mkdir -p $O; cd /Users/user/gam/bench/oracle/graph; ls /Users/user/mpd-data/graph_oracle/texts > /dev/null; $PY prompted.py --base ${BASE:-Qwen/Qwen3-32B-FP8} --rounds ${ROUNDS:-3} --samples ${SAMPLES:-2} --questions ${QUESTIONS:-50} --out $O > $O/log 2>&1; echo done"
MATS_GPUS=1 MATS_BUILD=0 bench/mats/mats-run $N 8 ${MEM:-56} $H -- bash -c "$CMD"
