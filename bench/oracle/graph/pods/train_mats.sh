#!/usr/bin/env bash
# Graph oracle training on one MATS L40 (10-09 night), the evidence run: LoRA SFT of Qwen3-4B on the search's programs
# (texts/search; no generated English) with the model's activations as input tokens; the same held-out evaluation of the SFT
# adapter (RL's starting point); rl2 for RL_HOURS (curves, edge credit,
# the verifier-feedback revision round, the English reader); evaluation on held-out questions against the search's answers
# and VPD's (necessity, VPD's shared adversary), then the swap test (another text's activations). VPD's full checkpoint is
# on the cluster, so VPD's answer is scored there. Submitted with mats-run from the pushed HEAD; queued until a GPU frees.
# SEARCH, HELD: the SFT answers' and the held-out search baseline's directories (default texts/search, texts/search_heldout;
# a cluster output directory /Users/user/mpd-data/cluster/... is used in place); HARD: a hard-split search baseline, adding
# the hard evaluation; VPD_LIST=K: the question lists VPD's first K subcomponents at the target (texts/vpd_ranked);
# RESPONSES=K: and where the prediction responds, at K positions (texts/vpd_responses); ROOT: the texts directory (default
# graph_oracle/texts; a cluster directory holding vpd4l, vpd_ranked and vpd_responses works in place); SFT_FROM: an adapter
# SFT starts from.
#   bench/oracle/graph/pods/train_mats.sh RL_HOURS   (env: RUN, SFT_STEPS, BPS, SAMPLES, EVIDENCE= for text only, INIT, SEARCH, HELD, HARD, VPD_LIST, RESPONSES, ROOT, SFT_FROM)
cd /Users/user/gam
RH=${1:-4.0}
RUN=${RUN:-m2}
N=oracle-graph-$RUN
O=/Users/user/mpd-data/cluster/$N
G=/Users/user/gam/bench/oracle/graph
B=${ROOT:-/Users/user/mpd-data/graph_oracle/texts}
D=${SEARCH:-$B/search}
PY=/mnt/nw/home/s.sauers/rl-venv/bin/python
T='/ephemeral/$USER/'$N  # scratch and caches on the worker's local disk (the admins: /ephemeral/$USER, never /tmp); $USER expands on the cluster
WAIT=${WAIT:-3}  # hours the job may wait for its GPU to be free (gpu_free.py)
EVIDENCE=${EVIDENCE-1}  # empty: the text-only oracle
if [ -n "$EVIDENCE" ]; then EV="--evidence --max-tokens 4096 --max-model-len 26624"; else EV="--max-tokens 4096 --max-model-len 14336"; fi
SHARE=$([ -n "$EVIDENCE" ] && echo "--gpu-memory 0.4 --score-workers 2" || echo "--gpu-memory 0.5 --score-workers 3")  # with evidence vLLM stays awake (train.py): a fixed share of the L40
BASIC="--base Qwen/Qwen3-4B --model vpd4l --behaviors $B --search $D --part-tokens $O/reg.safetensors --scorer native $EV ${VPD_LIST:+--vpd-list $VPD_LIST} ${RESPONSES:+--responses $RESPONSES} --share-gpu $SHARE --micro 1 --eval-behaviors 50 --oracle-runs $O/oracle"
COMMON="$BASIC --search-heldout ${HELD:-$B/search_heldout}"
SFT="$PY rl/train.py --mode sft $COMMON --samples 4 --sft-steps ${SFT_STEPS:-80} --batch 8 ${SFT_FROM:+--init $SFT_FROM} --out $O/sft --run-name ${N}_sft > $O/sft/log 2>&1"
A=${INIT:-$O/sft/adapter}  # INIT: an SFT adapter directory from an earlier run (RL starts from it, no SFT)
[ -n "$INIT" ] && SFT="ls $INIT > /dev/null"
EVAL_HARD=$([ -n "$HARD" ] && echo "mkdir -p $O/eval_hard; $PY rl/train.py --mode eval $BASIC --search-heldout $HARD --eval-split hard --revise --necessity --init $O/rl/adapter --samples 4 --out $O/eval_hard --run-name ${N}_hard > $O/eval_hard/log 2>&1;")
SWAP="$PY rl/train.py --mode eval $COMMON --swap-evidence --no-baselines --init $O/rl/adapter --samples 4 --out $O/eval_swap --run-name ${N}_swap > $O/eval_swap/log 2>&1;"
CMD="mkdir -p $T $O/sft $O/eval_sft $O/rl $O/eval_rl $O/eval_swap; export TMPDIR=$T TRITON_CACHE_DIR=$T/triton TORCHINDUCTOR_CACHE_DIR=$T/inductor OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; cd $G; $PY pods/gpu_free.py --hours $WAIT > $O/gpu_free.log 2>&1 || exit 1; $PY part_tokens.py build --vpd /Users/user/mpd-data/oracle/vpd/uv.safetensors --out $O/reg.safetensors && \
{ $SFT; }; \
$PY rl/train.py --mode eval $COMMON --revise --necessity --init $A --samples 4 --out $O/eval_sft --run-name ${N}_sft > $O/eval_sft/log 2>&1; \
$PY rl/train.py --mode rl2 --revise --reader $COMMON --init $A --samples ${SAMPLES:-6} --behaviors-per-step ${BPS:-2} --credit 8 --credit-answers 1 --refine 0 --steps 1000 --hours $RH --out $O/rl --run-name ${N}_rl > $O/rl/log 2>&1; \
$PY rl/train.py --mode eval $COMMON --revise --necessity --adversarial --transfer --init $O/rl/adapter --samples 4 --out $O/eval_rl --run-name ${N}_rl > $O/eval_rl/log 2>&1; \
$EVAL_HARD ${EVIDENCE:+$SWAP} echo done"
MATS_GPUS=1 MATS_BUILD=0 bench/mats/mats-run $N 8 ${MEM:-40} $(python3 -c "print(round($RH + 4.0 + $WAIT, 2))") -- bash -c "$CMD"
