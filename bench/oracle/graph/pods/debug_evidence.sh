#!/usr/bin/env bash
# Which vLLM setting breaks the evidence oracle's sampling (prompts as embeddings + LoRA): the held-out evaluation of
# a fresh adapter on 2 questions, once per variant, each its own process (a CUDA error ends the process): as trained
# (compiled kernels, CUDA graphs), --enforce-eager, and text only. One pod, minutes.
#   bench/oracle/graph/pods/debug_evidence.sh   (env: RUN, GPU, PRICE)
cd /Users/user/gam
RUN=${RUN:-dbg1}
S=/Users/user/mpd-data/scratch/glead/v5/src_debug_$RUN
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive origin/main bench/oracle/graph bench/vpd_2951/vpd_model.py | tar -x -C "$S"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"
G=$S/bench/oracle/graph
N=oracle-graph-$RUN
O=/Users/user/mpd-data/runpod/$N; W=/workspace/runs/$N
T=/Users/user/mpd-data/graph_oracle/texts-$RUN.tar
COPYFILE_DISABLE=1 tar -cf "$T" -C /Users/user/mpd-data/graph_oracle texts/vpd4l texts/search texts/search_heldout
PT="~/mpd-data/graph_oracle/texts"
BASE="--base Qwen/Qwen3-4B --model vpd4l --behaviors $PT --search $PT/search --search-heldout $PT/search_heldout --part-tokens /root/reg.safetensors --scorer native --score-workers 4 --share-gpu --gpu-memory 0.5 --micro 1 --eval-behaviors 2 --samples 2 --no-baselines"
EV="--evidence --max-tokens 1024 --max-model-len 26624"
RUNS=""
for v in "compiled:$EV" "eager:$EV --enforce-eager" "text:--max-tokens 1024 --max-model-len 14336"; do
  name=${v%%:*}; args=${v#*:}
  RUNS="$RUNS mkdir -p $O/$name; python rl/train.py --mode eval $BASE $args --out $O/$name --run-name dbg_$name > $O/$name/log 2>&1; echo \"$name exit \$?\" >> $O/variants.txt;"
done
CMD="export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4; ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O $S ~/mpd-data/graph_oracle; tar -xf $T -C ~/mpd-data/graph_oracle; tar -xf $S.tar -C $S; ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors > /dev/null; cd $G; python part_tokens.py build --vpd /Users/user/mpd-data/oracle/vpd/uv.safetensors --out /root/reg.safetensors; $RUNS cat $O/variants.txt"
RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_HF_MODELS="Qwen/Qwen3-4B" RP_MAX_PRICE=${PRICE:-1.80} RP_MIN_VRAM_GB=78 RP_MIN_RAM_GB=96 \
  bench/runpod/rp-run $N "${GPU:-NVIDIA A100 80GB PCIe,NVIDIA A100-SXM4-80GB}" 1.0 -- bash -c "$CMD" > /Users/user/mpd-data/scratch/glead/v5/rp_$N.log 2>&1
echo "$N exit $?"
