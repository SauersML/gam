#!/usr/bin/env bash
# Keep our jobs on the shared MATS node fair to everyone (#2951). Run on the login node:
#   setsid nohup bash budget.sh >> ~/budget.log 2>&1 < /dev/null &
# Every minute it sizes our room from the node itself: its CPUs, GPUs and memory, less every other
# user's running job and every other user's job that is waiting for resources (counted as if
# already placed), less some slack for whoever submits next. Our pending jobs beyond that room are
# held and released smallest first (then in submission order) as room appears, so idle CPUs and
# GPUs get used and nobody else waits on us. Per-user QOS limits (normal: 124 CPUs, 6 GPUs) are
# Slurm's to enforce; debug has none, so a full idle node can be ours. Others' jobs pending for
# their own limits or dependencies don't shrink our room: we are not what holds them. Builds
# (mpd-build) are short and every queued run waits on them, so they are never held. Jobs held by
# anything else (an agent's own hold) are left alone.
set -uo pipefail
NODE=${NODE:-l40-worker}
SLACK_CPUS=${SLACK_CPUS:-24}
SLACK_GB=${SLACK_GB:-48}
held=$HOME/budget_held.txt
touch "$held"
# Slurm memory ("32G", "4000M", "1T") in whole GB.
gb_of() { awk -v m="$1" 'BEGIN { u = substr(m, length(m)); v = substr(m, 1, length(m) - 1) + 0
    print int(u == "T" ? v * 1024 : u == "M" ? v / 1024 : u == "K" ? v / 1048576 : u == "G" ? v : m / 1024) }'; }
gpus_of() { grep -oE 'gpu(:[a-z0-9]+)?:[0-9]+' <<< "$1" | grep -oE '[0-9]+$' | head -1; }
last=
while :; do
    node=$(scontrol show node "$NODE")
    total_c=$(grep -oE 'CPUTot=[0-9]+' <<< "$node" | cut -d= -f2)
    total_g=$(grep -oE 'Gres=gpu:[a-z0-9]+:[0-9]+' <<< "$node" | grep -oE '[0-9]+$')
    total_m=$(( $(grep -oE 'RealMemory=[0-9]+' <<< "$node" | cut -d= -f2) / 1024 ))
    other_c=0 other_g=0 other_m=0
    while read -r u st c b m r; do
        [ "$u" = "$USER" ] && continue
        [ "$st" = RUNNING ] || [ "$r" = Resources ] || [ "$r" = Priority ] || continue
        g=$(gpus_of "$b")
        other_c=$(( other_c + c )) other_g=$(( other_g + ${g:-0} )) other_m=$(( other_m + $(gb_of "$m") ))
    done < <(squeue -h -o '%u %T %C %b %m %r')
    cap_c=$(( total_c - other_c - SLACK_CPUS )) cap_g=$(( total_g - other_g )) cap_m=$(( total_m - other_m - SLACK_GB ))
    cpus=0 gpus=0 mem=0
    while read -r c b m; do
        g=$(gpus_of "$b")
        cpus=$(( cpus + c )) gpus=$(( gpus + ${g:-0} )) mem=$(( mem + $(gb_of "$m") ))
    done < <(squeue -u "$USER" -h -t R -o '%C %b %m')
    room_c=$(( cap_c - cpus )) room_g=$(( cap_g - gpus )) room_m=$(( cap_m - mem ))
    now="ours ≤ $cap_c CPUs, $cap_g GPUs, ${cap_m}G (others use or wait for $other_c CPUs, $other_g GPUs, ${other_m}G)"
    [ "$now" != "$last" ] && echo "$(date +%T) $now" && last=$now
    while read -r id name c b m reason; do
        g=$(gpus_of "$b")
        g=${g:-0} m=$(gb_of "$m")
        ours=0
        grep -Fqx "$id" "$held" && ours=1
        [ "$reason" = JobHeldUser ] && (( ours == 0 )) && continue
        # Waiting on another job (a build): it can't start, so it takes no room until it can.
        [ "$reason" = Dependency ] && continue
        if [ "$name" = mpd-build ]; then
            (( ours )) && scontrol release "$id" && sed -i "/^$id\$/d" "$held" && echo "$(date +%T) released build $id"
            continue
        fi
        if (( c <= room_c && g <= room_g && m <= room_m )); then
            room_c=$(( room_c - c )) room_g=$(( room_g - g )) room_m=$(( room_m - m ))
            if (( ours )); then
                scontrol release "$id" && sed -i "/^$id\$/d" "$held" && echo "$(date +%T) released $id $name ($c CPUs, $g GPUs)"
            fi
        elif [ "$reason" != JobHeldUser ]; then
            scontrol hold "$id" && echo "$id" >> "$held" && echo "$(date +%T) held $id $name ($c CPUs, $g GPUs, ${m}G; room $room_c CPUs, $room_g GPUs, ${room_m}G)"
        fi
    done < <(squeue -r -u "$USER" -h -t PD -S V -o '%i %j %C %b %m %r' | sort -s -n -k3,3)
    sleep 60
done
