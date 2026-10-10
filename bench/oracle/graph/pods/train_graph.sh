#!/usr/bin/env bash
# Graph oracle training on one pod (10-09 night, eps-free design): describe.py writes the English of the bootstrap search
# answers (texts/search, native.py search) with the base model; a light LoRA SFT on them (SFT_STEPS); rl2 for RL_HOURS
# (answers ranked by their curves, edge credit, refinement, the English credited by the frozen reader); evaluation on
# held-out questions against the search's answers (texts/search_heldout), plus the swap test (another text's activations).
# The native verifier runs vpd4l on the pod's GPU beside the policy. The Python is origin/main's bench/oracle/graph at launch.
#   bench/oracle/graph/pods/train_graph.sh RL_HOURS   (env: RUN, BASE, SFT_STEPS, EVIDENCE= for no activation tokens (default on), REVISE=1 for the verifier-feedback
#   revision round in RL and evaluation (synchronous loop), SAMPLES and BPS (RL group size, questions per step), CREDIT, CREDIT_ANSWERS, REFINE,
#   INIT (an SFT adapter directory to start RL from, skipping describe and SFT), DESCRIBED (described answers from an earlier run: no describe),
#   GPU, PRICE, VRAM). After the held-out evaluation, the same evaluation on the hard split (texts/search_hard as its search baseline).
# The questions, search answers, described answers and source go up as tar files (rp-run uploads a directory file by file).
cd /Users/user/gam
RH=${1:-1.0}
RUN=${RUN:-a}
EVIDENCE=${EVIDENCE-1}  # the activation tokens by default; EVIDENCE= for the text-only oracle
S=/Users/user/mpd-data/scratch/glead/v5/src_graph_$RUN
git fetch -q origin main && rm -rf "$S" "$S.tar" && mkdir -p "$S" && git archive origin/main bench/oracle/graph bench/vpd_2951/vpd_model.py | tar -x -C "$S" && echo "python source $(git rev-parse --short=10 origin/main)" > "$S/SOURCE"
COPYFILE_DISABLE=1 tar -cf "$S.tar" -C "$S" . && rm -rf "$S"  # rp-run uploads every existing path the command names: only the archives exist here
G=$S/bench/oracle/graph
B=/Users/user/mpd-data/graph_oracle/texts
N=oracle-graph-$RUN
O=/Users/user/mpd-data/runpod/$N; W=/workspace/runs/$N
T=/Users/user/mpd-data/graph_oracle/texts-$RUN.tar
COPYFILE_DISABLE=1 tar -cf "$T" -C /Users/user/mpd-data/graph_oracle texts/vpd4l texts/search texts/search_heldout $( [ -d $B/search_hard ] && echo texts/search_hard )
DT=/Users/user/mpd-data/graph_oracle/described-$RUN.tar  # not beside $O: rp-run counts paths under the run's output name as outputs
[ -n "$DESCRIBED" ] && COPYFILE_DISABLE=1 tar -cf "$DT" -C "$DESCRIBED" .
if [ -n "$EVIDENCE" ]; then LEN="--max-tokens 4096 --max-model-len 26624"; else LEN="--max-tokens 4096 --max-model-len 14336"; fi  # a revision prompt holds the first answer
PT="~/mpd-data/graph_oracle/texts"  # the texts on the pod (unpacked from $T; a Mac path here would be uploaded file by file)
COMMON="--base ${BASE:-Qwen/Qwen3-4B} --model vpd4l --behaviors $PT --search $O/described --search-heldout $PT/search_heldout --part-tokens /root/reg.safetensors --scorer native --score-workers 8 ${EVIDENCE:+--evidence} $LEN --share-gpu --gpu-memory 0.5 --micro 1 --eval-behaviors 50 --oracle-runs $O/oracle"
DESCRIBE="python describe.py $PT/search --split train --out $O/described --base ${BASE:-Qwen/Qwen3-4B} --max-tokens 4096 > $O/describe.log 2>&1"
SFT="mkdir -p $O/sft; python rl/train.py --mode sft $COMMON --samples 4 --sft-steps ${SFT_STEPS:-80} --batch 8 --out $O/sft --run-name graphsft > $O/sft/log 2>&1"
if [ -n "$REVISE" ]; then LOOP="--revise"; else LOOP="--async"; fi
RL="mkdir -p $O/rl; python rl/train.py --mode rl2 $LOOP --reader $COMMON --init $O/sft/adapter --samples ${SAMPLES:-8} --behaviors-per-step ${BPS:-4} --credit ${CREDIT:-16} --credit-answers ${CREDIT_ANSWERS:-2} --refine ${REFINE:-1} --steps 1000 --hours $RH --out $O/rl --run-name graphrl > $O/rl/log 2>&1"
EVAL="mkdir -p $O/eval_rl $O/eval_hard $O/eval_swap; python rl/train.py --mode eval $COMMON ${REVISE:+--revise} --necessity --adversarial --transfer --init $O/rl/adapter --samples 4 --out $O/eval_rl --run-name graphrl > $O/eval_rl/log 2>&1; \
python rl/train.py --mode eval ${COMMON/--search-heldout $PT\/search_heldout/--search-heldout $PT/search_hard} --eval-split hard ${REVISE:+--revise} --necessity --init $O/rl/adapter --samples 4 --out $O/eval_hard --run-name graphrl_hard > $O/eval_hard/log 2>&1"
SWAP="python rl/train.py --mode eval $COMMON --init $O/rl/adapter --samples 4 --swap-evidence --no-baselines --out $O/eval_swap --run-name graphrl_swap > $O/eval_swap/log 2>&1"
if [ -n "$INIT" ]; then  # an SFT adapter from an earlier run (its directory): RL starts from it, no describe or SFT
  RL=${RL/--init $O\/sft\/adapter/--init $INIT}
  START="ls $INIT > /dev/null"
elif [ -n "$DESCRIBED" ]; then  # described answers from an earlier run
  START="mkdir -p $O/described && tar -xf $DT -C $O/described && $SFT"
else
  START="$DESCRIBE && $SFT"
fi
CMD="ln -sfn $W/mpd-data ~/mpd-data; mkdir -p $O $S ~/mpd-data/graph_oracle; tar -xf $T -C ~/mpd-data/graph_oracle; tar -xf $S.tar -C $S; ls /Users/user/mpd-data/vpd/t-9d2b8f02/model_step_99999.safetensors /Users/user/mpd-data/vpd/t-9d2b8f02/model_config.yaml /Users/user/mpd-data/vpd/t-9d2b8f02/tokenizer.json /Users/user/mpd-data/oracle/vpd/uv.safetensors > /dev/null; cd $G; python part_tokens.py build --vpd /Users/user/mpd-data/oracle/vpd/uv.safetensors --out /root/reg.safetensors && { $START; }; $RL; $EVAL; ${EVIDENCE:+$SWAP;} echo done"
RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_HF_MODELS="${BASE:-Qwen/Qwen3-4B}" RP_MAX_PRICE=${PRICE:-1.20} RP_MIN_VRAM_GB=${VRAM:-48} RP_MIN_RAM_GB=96 \
  bench/runpod/rp-run $N "${GPU:-auto}" $(python3 -c "print(round($RH + 2.0, 2))") -- bash -c "$CMD" > /Users/user/mpd-data/scratch/glead/v5/rp_$N.log 2>&1
echo "$N exit $?"
