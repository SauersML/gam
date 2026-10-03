#!/usr/bin/env bash
# Keep our GPU use on the shared node within the lead's split (#2951), run on the login node:
#   nohup bash gpu_balance.sh >> ~/gpu_balance.log 2>&1 &
# Every minute:
#   - each reserved group (e2e 2, trainer's device trainer 1, robust's battery 1) is owed its GPUs;
#     what a group is not using or asking for right now is slack, and the gsweep arrays (1 GPU each
#     at base) take it by raising their array throttles; when the group submits, the throttles drop
#     back (a running gsweep task finishes; no new one starts in its place);
#   - our debug-QOS GPU jobs (which the 6-GPU QOS cap does not cover) are held while starting them
#     would put us above 6 GPUs, and released when there is room.
set -uo pipefail
SWEEPS=(15314 15316)
declare -A RESERVE=([e2e]=2 [trainer]=1 [battery]=1)
group() {
    case $1 in
        e2e*) echo e2e ;;
        dtrain* | gpu-box* | gpucheck*) echo trainer ;;
        advbat* | adv-* | word-* | pgd*) echo battery ;;
        gsweep*) echo sweep ;;
        *) echo other ;;
    esac
}
gpus_of() { grep -oE 'gpu:[0-9]+' <<< "$1" | cut -d: -f2 | head -1; }
held_file=$HOME/gpu_balance_held.txt
touch "$held_file"
while :; do
    declare -A demand=()
    running=0 others=0
    while read -r id name state qos gres reason; do
        g=$(gpus_of "$gres")
        [ -n "$g" ] || continue
        [ "$state" = R ] && running=$(( running + g ))
        [ "$state" = R ] && [ "$(group "$name")" != sweep ] && others=$(( others + g ))
        # A job someone else held is not asking for a GPU; one held here for the 6-GPU limit is.
        [ "$state" = PD ] && [ "$reason" = JobHeldUser ] && ! grep -qx "${id%%_*}" "$held_file" && continue
        k=$(group "$name")
        demand[$k]=$(( ${demand[$k]:-0} + g ))
    done < <(squeue -u "$USER" -h -o '%i %j %t %q %b %r')
    slack=0
    for k in "${!RESERVE[@]}"; do
        d=${demand[$k]:-0}
        (( d < RESERVE[$k] )) && slack=$(( slack + RESERVE[$k] - d ))
    done
    total=$(( ${#SWEEPS[@]} + slack ))
    # Never past 6 in all: GPUs our other jobs already hold (debug ones included) come off the top.
    (( total > 6 - others )) && total=$(( 6 - others ))
    (( total < ${#SWEEPS[@]} )) && total=${#SWEEPS[@]}
    i=0
    for s in "${SWEEPS[@]}"; do
        want=$(( total / ${#SWEEPS[@]} + (i < total % ${#SWEEPS[@]} ? 1 : 0) ))
        have=$(scontrol show job "$s" 2> /dev/null | grep -oP 'ArrayTaskThrottle=\K[0-9]+' | head -1)
        if [ -n "$have" ] && [ "$have" != "$want" ]; then
            scontrol update jobid="$s" ArrayTaskThrottle="$want" && echo "$(date +%T) $s throttle $have -> $want (slack $slack)"
        fi
        i=$(( i + 1 ))
    done
    # Debug-QOS GPU jobs: hold those that would take us past 6, release ours when there is room.
    room=$(( 6 - running ))
    while read -r id g; do
        if (( g <= room )); then
            scontrol release "$id" 2> /dev/null && sed -i "/^$id\$/d" "$held_file" && echo "$(date +%T) released debug GPU job $id"
            room=$(( room - g ))
        fi
    done < <(while read -r id; do squeue -h -j "$id" -t PD -o '%i %b' 2> /dev/null | sed -E 's/ .*gpu:([0-9]+).*/ \1/'; done < "$held_file")
    while read -r id g; do
        if (( g > room )); then
            scontrol hold "$id" && echo "$id" >> "$held_file" && echo "$(date +%T) held debug GPU job $id ($g GPUs, $running running)"
        else
            room=$(( room - g ))
        fi
    done < <(squeue -u "$USER" -h -t PD -q debug -o '%i %b %r' | awk '$3 != "JobHeldUser" && $2 ~ /gpu/ { sub(/.*gpu:/, "", $2); print $1, $2 }')
    sleep 60
done
