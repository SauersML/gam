#!/usr/bin/env bash
# Reward-hacking monitor testbed on one RunPod pod: agent runs (agent.py, vLLM), their activations (acts.py, HF),
# the text monitor (monitor.py, vLLM); each stage skipped when its output is complete, so a replaced pod resumes.
# The code and the ImpossibleBench data go up as one tar file; outputs come back to ~/mpd-data/runpod/NAME.
#   bench/monitor_2951/pod.sh NAME HOURS "AGENT ARGS"     (env: GPU, MAXGB, AGENT (default Qwen/Qwen3-8B), LAYERS)
cd /Users/user/gam
N=$1 H=$2 ARGS=$3 AGENT=${AGENT:-Qwen/Qwen3-8B}
IN=/Users/user/mpd-data/monitor/pod_in_$N.tar
S=/Users/user/mpd-data/scratch/monitor/stage_$N
rm -rf "$S" && mkdir -p "$S/data" && cp bench/monitor_2951/agent.py bench/monitor_2951/sandbox.py bench/monitor_2951/acts.py bench/monitor_2951/monitor.py "$S/" \
  && cp /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/original-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/oneoff-00000-of-00001.parquet \
        /Users/user/mpd-data/monitor/raw/impossible_livecodebench/data/conflicting-00000-of-00001.parquet "$S/data/" \
  && COPYFILE_DISABLE=1 tar -cf "$IN" -C "$S" . && rm -rf "$S"
O=/Users/user/mpd-data/runpod/$N
X=/Users/user/mpd-data/monitor/pod_src_$N
CMD="set -e; mkdir -p $O $X; tar -xf $IN -C $X; cd $X; export MONITOR_DATA=$X/data PYTHONUNBUFFERED=1;
[ -f $O/gen/done ] || { python agent.py --model $AGENT --out $O/gen $ARGS && touch $O/gen/done; };
[ -f $O/acts/feats.npz ] || python acts.py --model $AGENT --layers ${LAYERS:-8,16,24,32} --runs $O/gen/transcripts.jsonl --out $O/acts --max-gb ${MAXGB:-40};
[ -f $O/mon/scores.jsonl ] || python monitor.py --agent-model $AGENT --runs $O/gen/transcripts.jsonl --out $O/mon;
echo all stages done"
RP_ALLOW_OVER=1 RP_OWNER=lead RP_PARALLEL=1 RP_PYENV=oracle RP_DISK_GB=${DISK:-120} RP_MIN_VRAM_GB=44 \
  RP_HF_MODELS="$AGENT Qwen/Qwen3-30B-A3B-Instruct-2507-FP8" \
  bench/runpod/rp-run $N "${GPU:-NVIDIA A40,NVIDIA L40S,NVIDIA RTX 6000 Ada Generation}" $H -- bash -c "$CMD"
