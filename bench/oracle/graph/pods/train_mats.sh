#!/usr/bin/env bash
# Graph oracle training on one MATS L40 (10-09 night), the evidence run: LoRA SFT of Qwen3-4B on described search answers
# (DESCRIBED, describe.py's output) with the model's activations as input tokens; rl2 for RL_HOURS (curves, edge credit,
# the verifier-feedback revision round, the English reader); evaluation on held-out questions against the search's answers
# and VPD's (necessity, VPD's shared adversary), then the swap test (another text's activations). VPD's full checkpoint is
# on the cluster, so VPD's answer is scored there. Submitted with mats-run from the pushed HEAD; queued until a GPU frees.
#   bench/oracle/graph/pods/train_mats.sh RL_HOURS   (env: RUN, DESCRIBED, SFT_STEPS, BPS, SAMPLES)
cd /Users/user/gam
RH=${1:-4.0}
RUN=${RUN:-m2}
N=oracle-graph-$RUN
O=/Users/user/mpd-data/cluster/$N
G=/Users/user/gam/bench/oracle/graph
B=/Users/user/mpd-data/graph_oracle/texts
D=${DESCRIBED:-/Users/user/mpd-data/runpod/oracle-graph-r1/described}
PY=/mnt/nw/home/s.sauers/rl-venv/bin/python
COMMON="--base Qwen/Qwen3-4B --model vpd4l --behaviors $B --search $D --search-heldout $B/search_heldout --part-tokens $O/reg.safetensors --scorer native --evidence --max-tokens 4096 --max-model-len 26624 --share-gpu --gpu-memory 0.5 --micro 1 --eval-behaviors 50 --oracle-runs $O/oracle"
CMD="mkdir -p $O/sft $O/rl $O/eval_rl $O/eval_swap; cd $G; $PY part_tokens.py build --vpd /Users/user/mpd-data/oracle/vpd/uv.safetensors --out $O/reg.safetensors && \
$PY rl/train.py --mode sft $COMMON --samples 4 --sft-steps ${SFT_STEPS:-80} --batch 8 --out $O/sft --run-name graphsft_ev > $O/sft/log 2>&1; \
$PY rl/train.py --mode rl2 --revise --reader $COMMON --init $O/sft/adapter --samples ${SAMPLES:-6} --behaviors-per-step ${BPS:-2} --credit 8 --credit-answers 1 --refine 0 --steps 1000 --hours $RH --out $O/rl --run-name graphrl_ev > $O/rl/log 2>&1; \
$PY rl/train.py --mode eval $COMMON --revise --necessity --adversarial --init $O/rl/adapter --samples 4 --out $O/eval_rl --run-name graphrl_ev > $O/eval_rl/log 2>&1; \
$PY rl/train.py --mode eval $COMMON --swap-evidence --no-baselines --init $O/rl/adapter --samples 4 --out $O/eval_swap --run-name graphrl_ev_swap > $O/eval_swap/log 2>&1; echo done"
MATS_GPUS=1 MATS_BUILD=0 bench/mats/mats-run $N 8 64 $(python3 -c "print(round($RH + 4.0, 2))") -- bash -c "$CMD"
