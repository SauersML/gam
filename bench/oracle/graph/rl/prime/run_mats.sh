#!/bin/bash
# prime-rl RL of the graph oracle (#2951) on two MATS L40s: a vLLM replica on each (data parallel), serve_scores.py (two
# scoring processes) beside the first, the trainer beside the second. Submitted from a code snapshot's root on MATS:
#   sbatch bench/oracle/graph/rl/prime/run_mats.sh MODEL [rl args, e.g. --max-steps 100]    (env: RUN, REVISE=1)
# MODEL: export_hf.py's checkpoint. The questions are written afresh (questions.py), so training holds every text whose
# lists exist at the start. Uses the prime-rl install at /mnt/nw/home/s.sauers/prime-rl, with NVIDIA's CUDA 13
# forward-compatibility libcuda (the node's driver is CUDA 12.2). Logs, metrics and the last adapter go to
# mpd-data/cluster/oracle-graph-prime/RUN; checkpoints and caches stay on the worker's /ephemeral.
#SBATCH --job-name=oracle-graph-prime
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/mnt/nw/home/s.sauers/mpd-data/cluster/oracle-graph-prime/slurm-%j.out
set -u
MODEL=$1
shift
RUN=${RUN:-r$SLURM_JOB_ID}
G=$SLURM_SUBMIT_DIR/bench/oracle/graph
O=/mnt/nw/home/s.sauers/mpd-data/cluster/oracle-graph-prime/$RUN
E=/ephemeral/$USER/oracle-graph-prime
RLPY=/mnt/nw/home/s.sauers/rl-venv/bin/python
TEXTS=/mnt/nw/home/s.sauers/mpd-data/cluster/oracle-graph-lists/texts
mkdir -p $O/data $E/tmp $E/cache
export TMPDIR=$E/tmp OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

# Orphaned processes of other jobs sit on GPUs Slurm counts as free: every GPU of this job must have room.
for g in ${CUDA_VISIBLE_DEVICES//,/ }; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $g)
  echo "GPU $g: $free MiB free"
  [ "$free" -ge 40000 ] || { echo "GPU $g is taken; ending"; exit 3; }
done
A=${CUDA_VISIBLE_DEVICES%%,*}
B=${CUDA_VISIBLE_DEVICES##*,}
if curl -s localhost:8765/health > /dev/null; then echo "port 8765 is taken; ending"; exit 3; fi

cd $G
for s in train heldout; do $RLPY rl/questions.py $O/data/$s.jsonl --split $s --root $TEXTS --vpd-list 512 --responses 32; done

nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw --format=csv,noheader -i $A,$B -l 5 > $O/gpu.csv &
SMI=$!
# serve_scores in its own process group, so its scoring processes end with it.
CUDA_VISIBLE_DEVICES=$A setsid $RLPY rl/serve_scores.py --port 8765 --workers 2 --texts /mnt/nw/home/s.sauers/mpd-data/graph_oracle/texts/vpd4l > $O/serve_scores.log 2>&1 &
SCORER=$!
trap 'kill -TERM -- -$SCORER 2> /dev/null; kill $SMI 2> /dev/null' EXIT
until curl -s localhost:8765/health | grep -q '"alive": 2'; do kill -0 $SCORER 2> /dev/null || { echo "serve_scores died"; exit 4; }; sleep 5; done
start=$(date +%s)
curl -s -X POST localhost:8765/score -d '{"task": "text6000", "answers": [""], "seed": 0}' > /dev/null
echo "serve_scores up; first score (a text's baselines and the models' first use) took $(( $(date +%s) - start )) s"

source /mnt/nw/home/s.sauers/prime-rl/.venv/bin/activate
export LD_LIBRARY_PATH=/mnt/nw/home/s.sauers/cuda-compat/13-0/usr/local/cuda-13.0/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PYTHONPATH=$G/rl/prime  # graph_oracle: the taskset and its environment
export PRL_OUTPUT_DIR=$E/runs WANDB_MODE=disabled XDG_CACHE_HOME=$E/cache VLLM_CACHE_ROOT=$E/cache/vllm TORCHINDUCTOR_CACHE_DIR=$E/cache/inductor UV_CACHE_DIR=$E/cache/uv
cd $O/data
start=$(date +%s)
# The launcher takes inference GPUs first, then the trainer's, from this list: B twice puts a replica and the trainer on B.
CUDA_VISIBLE_DEVICES=$A,$B,$B rl @ $G/rl/prime/rl.toml ${REVISE:+@ $G/rl/prime/revise.toml} --model.name $MODEL --run.name $RUN --no-monitors.wandb \
  --inference.server.port 8417 --orchestrator.model.client.base-url http://localhost:8417/v1 \
  --env-vars "{\"TRITON_CACHE_DIR\": \"$E/cache/triton\"}" "$@"
echo "rl exit $? after $(( $(date +%s) - start )) s"
R=$PRL_OUTPUT_DIR/$RUN
cp -r $R/logs $R/configs $O/
cp -r $R/monitors $O/ 2> /dev/null
last=$(ls -v $R/broadcasts 2> /dev/null | tail -n 1)
[ -n "$last" ] && cp -r $R/broadcasts/$last $O/adapter
