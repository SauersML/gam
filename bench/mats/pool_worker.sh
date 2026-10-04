#!/usr/bin/env bash
# MATS pool worker (#2951): one long-lived normal-QOS allocation that runs queued tasks as job steps.
#
#   sbatch -J mpd-pool -p compute -c CPUS --mem=MEM [--gres=gpu:G] -t MIN pool_worker.sh MIN
#
# Tasks are files in ~/mpd-data/cluster/queue/todo/*.task (written by `MATS_POOL=1 mats-run`), each
# a few `# key=value` lines (name, cpus, mem, gpus, minutes, ready, script, log, task) naming a job
# script. The worker packs as many as fit its free CPUs, GPUs and memory. The compute node has no
# Slurm commands, so each task is its own process group pinned (taskset) to its share of the
# worker's cores, with SLURM_CPUS_PER_TASK (hence RAYON_NUM_THREADS) and CUDA_VISIBLE_DEVICES set to
# its share, under `timeout` for its minutes. The worker waits for a task's
# build when `ready` names a file that does not exist yet, and skips tasks longer than its remaining
# time (a fitting task waiting for its build keeps the worker for up to 30 minutes). A claimed task moves to running/, then done/ with its exit code. `mats-pool cancel TASK`
# drops a file running/TASK.cancel, and the worker stops that step and takes the next task: work
# changes by editing the queue, without giving up the allocation. With nothing running and nothing
# it can take for 10 minutes, the worker exits, so it never holds resources idle.
set -o pipefail
Q=$HOME/mpd-data/cluster/queue
mkdir -p "$Q/todo" "$Q/running" "$Q/done" "$Q/workers"
me=$SLURM_JOB_ID
CPUS=$SLURM_CPUS_PER_TASK
MEM=$(( ${SLURM_MEM_PER_NODE:-4096} / 1024 ))
GPUS=0
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && GPUS=$(awk -F, '{ print NF }' <<< "$CUDA_VISIBLE_DEVICES")
end=$(( $(date +%s) + ${1:-60} * 60 ))
left() { echo $(( (end - $(date +%s)) / 60 )); }
# The worker's cores and GPUs, handed out to tasks as disjoint shares.
cores=($(taskset -pc $$ | sed 's/.*: //' | tr ',' '\n' | awk -F- '{ if (NF == 2) for (i = $1; i <= $2; i++) print i; else print $1 }'))
gpus=($(tr ',' ' ' <<< "${CUDA_VISIBLE_DEVICES:-}"))
declare -A owner gowner
field() { sed -n "s/^# $1=//p" "$2" | head -1; }
log() { echo "== $(date '+%F %T') $*"; }
declare -A pid cpu mem gpu
used_c=0 used_m=0 used_g=0
idle=$(date +%s)
trap 'for t in "${!pid[@]}"; do kill -TERM -- "-${pid[$t]}" 2> /dev/null; mv "$Q/running/$t" "$Q/todo/$t" 2> /dev/null; done; rm -f "$Q/workers/$me"; exit 0' TERM INT
log "worker $me: $CPUS CPUs, ${MEM}G, $GPUS GPUs"
while :; do
    for t in "${!pid[@]}"; do
        if ! kill -0 "${pid[$t]}" 2> /dev/null; then
            wait "${pid[$t]}"
            rc=$?
            { cat "$Q/running/$t"; echo "# rc=$rc"; echo "# worker=$me"; echo "# ended=$(date '+%F %T')"; } > "$Q/done/$t" 2> /dev/null
            rm -f "$Q/running/$t" "$Q/running/$t.cancel"
            used_c=$(( used_c - ${cpu[$t]} )) used_m=$(( used_m - ${mem[$t]} )) used_g=$(( used_g - ${gpu[$t]} ))
            for k in "${!owner[@]}"; do [ "${owner[$k]}" = "$t" ] && unset "owner[$k]"; done
            for k in "${!gowner[@]}"; do [ "${gowner[$k]}" = "$t" ] && unset "gowner[$k]"; done
            unset "pid[$t]" "cpu[$t]" "mem[$t]" "gpu[$t]"
            log "done $t (exit $rc)"
        elif [ -e "$Q/running/$t.cancel" ]; then
            kill -TERM -- "-${pid[$t]}" 2> /dev/null
            log "cancelling $t"
        fi
    done
    mins=$(left)
    waiting=0
    for f in $(ls -1tr "$Q"/todo/*.task 2> /dev/null); do
        c=$(field cpus "$f") m=$(field mem "$f") g=$(field gpus "$f") n=$(field minutes "$f") r=$(field ready "$f")
        # A GPU worker takes only GPU tasks: a CPU task on it kept its GPU allocated and idle for hours
        # while GPU tasks waited (15564, 15585). CPU tasks go to CPU workers.
        (( GPUS == 0 || g > 0 )) || continue
        (( used_c + c <= CPUS && used_m + m <= MEM && used_g + g <= GPUS && n + 2 <= mins )) || continue
        [ -z "$r" ] || [ -e "$r" ] || { waiting=1; continue; }
        t=${f##*/}
        mv "$f" "$Q/running/$t" 2> /dev/null || continue
        mine=() mg=()
        for k in "${cores[@]}"; do (( ${#mine[@]} < c )) && [ -z "${owner[$k]}" ] && { mine+=("$k"); owner[$k]=$t; }; done
        for k in "${gpus[@]}"; do (( ${#mg[@]} < g )) && [ -z "${gowner[$k]}" ] && { mg+=("$k"); gowner[$k]=$t; }; done
        task=$(field task "$Q/running/$t")
        (
            export SLURM_CPUS_PER_TASK=$c MATS_POOL_TASK=$t CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${mg[*]}")
            [ -n "$task" ] && export SLURM_ARRAY_TASK_ID=$task
            exec setsid timeout -s TERM -k 60 "${n}m" taskset -c "$(IFS=,; echo "${mine[*]}")" \
                bash "$(field script "$Q/running/$t")" >> "$(field log "$Q/running/$t")" 2>&1
        ) &
        pid[$t]=$! cpu[$t]=$c mem[$t]=$m gpu[$t]=$g
        used_c=$(( used_c + c )) used_m=$(( used_m + m )) used_g=$(( used_g + g ))
        echo "# worker=$me" >> "$Q/running/$t"
        log "started $t (cores ${mine[*]}, ${m}G, GPUs ${mg[*]:-none}, $n min)"
    done
    echo "$CPUS $MEM $GPUS $used_c $used_m $used_g $mins" > "$Q/workers/$me"
    if (( ${#pid[@]} > 0 )); then
        idle=$(date +%s)
    elif (( waiting && $(date +%s) - idle < 1800 )); then
        :   # a task that fits waits for its build; give the build up to 30 minutes
    elif (( $(date +%s) - idle > 600 )); then
        log "nothing to run for 10 minutes; exiting"
        break
    fi
    sleep 20
done
rm -f "$Q/workers/$me"
