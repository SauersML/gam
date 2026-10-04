#!/usr/bin/env bash
# Keep our jobs on the shared MATS node within the team's budget (#2951). Run on the login node:
#   nohup bash budget.sh >> ~/budget.log 2>&1 &
# Every minute, our running jobs plus the pending jobs it lets through stay within CPU_CAP CPUs and
# GPU_CAP GPUs; it holds our pending jobs beyond that, and releases them smallest first (then in
# submission order) when room appears, so one agent's queue of large jobs can't block everyone
# else's. GPUs that sit free while no other user has a GPU job waiting are ours to use too, so idle
# GPUs get used and are never taken from someone waiting for one. Jobs held by anything else
# (an agent's own hold) are left alone.
set -uo pipefail
CPU_CAP=${CPU_CAP:-100}
GPU_CAP=${GPU_CAP:-3}
held=$HOME/budget_held.txt
touch "$held"
gpus_of() { grep -oE 'gpu(:[a-z0-9]+)?:[0-9]+' <<< "$1" | grep -oE '[0-9]+$' | head -1; }
while :; do
    cpus=0 gpus=0
    while read -r c b; do
        g=$(gpus_of "$b")
        cpus=$(( cpus + c )) gpus=$(( gpus + ${g:-0} ))
    done < <(squeue -u "$USER" -h -t R -o '%C %b')
    used=$(scontrol -d show node l40-worker | grep -oE 'GresUsed=gpu:[a-z0-9]+:[0-9]+' | grep -oE '[0-9]+$')
    free=$(( 8 - ${used:-8} ))
    waiting=$(squeue -h -t PD -o '%u %b' | awk -v me="$USER" '$1 != me && $2 ~ /gpu/' | wc -l)
    room_c=$(( CPU_CAP - cpus ))
    room_g=$(( GPU_CAP - gpus ))
    (( waiting == 0 && free > room_g )) && room_g=$free
    while read -r id name c b reason; do
        g=$(gpus_of "$b")
        g=${g:-0}
        ours=0
        grep -qx "$id" "$held" && ours=1
        [ "$reason" = JobHeldUser ] && (( ours == 0 )) && continue
        if (( c <= room_c && g <= room_g )); then
            room_c=$(( room_c - c )) room_g=$(( room_g - g ))
            if (( ours )); then
                scontrol release "$id" && sed -i "/^$id\$/d" "$held" && echo "$(date +%T) released $id $name ($c CPUs, $g GPUs)"
            fi
        elif [ "$reason" != JobHeldUser ]; then
            scontrol hold "$id" && echo "$id" >> "$held" && echo "$(date +%T) held $id $name ($c CPUs, $g GPUs; room $room_c CPUs, $room_g GPUs)"
        fi
    done < <(squeue -u "$USER" -h -t PD -S V -o '%i %j %C %b %r' | sort -s -n -k3,3)
    sleep 60
done
