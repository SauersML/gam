#!/usr/bin/env bash
# Submits one target's prediction pipeline to MATS from the Mac (#2951): training shards (a job array of
# disjoint text ranges), the held-out sets, then the oracle's SFT (Qwen3-8B + LoRA) with its held-out
# evaluation once both succeed. Outputs under ~/mpd-data/cluster/predict-TARGET/ (mats-pull).
#   mats_pipeline.sh q06|q8|vpd4l [COMMIT]
#   q06    Qwen3-0.6B as target: 16 tasks x 8,192 FineWeb texts (about 1.05 M questions)
#   q8     Qwen3-8B as its own target: 8 tasks x 1,024 texts, TF32 (about 65 k questions)
#   q06tc  Qwen3-0.6B with transcoder features PD.tc at layers 6, 14, 22 (hard links in predict/tc3): 8 tasks
#          x 4,096 texts after q06's and the Mac's (data only)
#   vpd4l  vpd4l with VPD subcomponents: 16 tasks x 4,096 Pile training rows (about 0.5 M questions)
# The q06 and q8 SFT runs use the same steps, batch and held-out design: the self-explanation comparison
# (SFT_TYPES=plain,edit,... restricts an SFT to the question types both runs have).
set -Eeuo pipefail
export PATH=$PATH:/Users/user/gam/bench/mats
REF=${2:-$(git -C /Users/user/gam rev-parse origin/main)}
G=/Users/user/gam/bench/oracle/graph/predict
FW=/Users/user/mpd-data/qwen3_fineweb
BEHQ=/Users/user/mpd-data/graph_oracle/behaviors/qwen3-0.6b
case $1 in
    q06) MEM=16 MODEL=Qwen/Qwen3-0.6B ARR=0-15%2 TEXTS=8192 HTEXTS=256 TRAINW=$FW/train/windows_T128.u32 HELDW=$FW/heldout/windows_T128.u32 BEH=$BEHQ
         EXTRA=() ;;
    q8)  MEM=24 MODEL=Qwen/Qwen3-8B ARR=0-7%2 TEXTS=1024 HTEXTS=128 TRAINW=$FW/train/windows_T128.u32 HELDW=$FW/heldout/windows_T128.u32 BEH=$BEHQ
         EXTRA=(--tf32 --batch 16) ;;
    q06tc) MEM=24 MODEL=Qwen/Qwen3-0.6B ARR=64-71%1 TEXTS=4096 HTEXTS=256 TRAINW=$FW/train/windows_T128.u32 HELDW=$FW/heldout/windows_T128.u32 BEH=$BEHQ
         EXTRA=(--transcoders /Users/user/mpd-data/graph_oracle/predict/tc3 --tc-layers 6,14,22) NOSFT=1 ;;  # tasks 64.. read texts 262144.. (after q06 and the Mac shards)
    vpd4l) MEM=16 MODEL=vpd4l ARR=0-15%2 TEXTS=4096 HTEXTS=256 TRAINW=/Users/user/mpd-data/graph_oracle/predict/vpd4l/pile_train_rows.npy
         HELDW=/Users/user/mpd-data/vpd/pile_val_4096x513.npy BEH=/Users/user/mpd-data/graph_oracle/behaviors/vpd4l
         EXTRA=(--uv /Users/user/mpd-data/oracle/vpd/uv.safetensors --vpd-target /Users/user/mpd-data/vpd/t-9d2b8f02) ;;
    *) sed -n '2,10p' "$0"; exit 2 ;;
esac
N=predict-$1
D=/Users/user/mpd-data/cluster/$N
sub() { MATS_REF=$REF MATS_MEM_EXACT=1 "$@" | tail -1; }
train=$(MATS_QOS=debug MATS_GPUS=1 MATS_ARRAY=$ARR sub mats-run "$N-train" 6 "$MEM" 2 -- bash "$G/mats_gen.sh" "$MODEL" "$D/train_{task}.jsonl" "$TRAINW" train train "$TEXTS" "{task}" "${EXTRA[@]}")
held=$(MATS_QOS=debug MATS_GPUS=1 sub mats-run "$N-heldout" 6 "$MEM" 2 -- bash "$G/mats_heldout.sh" "$MODEL" "$D" "$HELDW" "$HTEXTS" "$BEH" "${EXTRA[@]}")
if [ -n "${NOSFT:-}" ]; then echo "$N: train $train, heldout $held (commit ${REF:0:12})"; exit 0; fi
sft=$(MATS_GPUS=1 MATS_AFTEROK="$train:$held" sub mats-run "$N-sft" 6 32 8 -- bash "$G/mats_sft.sh" Qwen/Qwen3-8B "$D/sft" \
    "$D/train_*.jsonl,$D/behaviors_train.jsonl" 4000 5.2 "prompts=$D/heldout_prompts.jsonl" "pieces=$D/heldout_pieces.jsonl" "behaviors=$D/behaviors_heldout.jsonl" ${SFT_TYPES:+"--types=$SFT_TYPES"})
echo "$N: train $train, heldout $held, sft $sft (commit ${REF:0:12})"
