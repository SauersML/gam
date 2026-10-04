#!/usr/bin/env bash
# Keep our jobs on the shared MATS node fair to everyone (#2951). Run on the login node:
#   setsid nohup bash budget.sh >> ~/budget.log 2>&1 < /dev/null &
# Every minute it sizes our room from the node itself: its CPUs and GPUs, less every other user's
# running job and every other user's job that is waiting for resources (counted as if already
# placed), less SLACK_CPUS for whoever submits next, and never past the per-user QOS limit. Our
# pending jobs beyond that room are held and released smallest first (then in submission order) as
# room appears, so idle CPUs and GPUs get used and nobody else waits on us. Others' jobs pending for
# their own limits or dependencies don't shrink our room: we are not what holds them. Builds
# (mpd-build) are short and every queued run waits on them, so they are never held. Jobs held by
# anything else (an agent's own hold) are left alone.
set -uo pipefail
NODE=${NODE:-l40-worker}
QOS_CPUS=${QOS_CPUS:-124}
QOS_GPUS=${QOS_GPUS:-6}
SLACK_CPUS=${SLACK_CPUS:-16}
held=$HOME/budget_held.txt
touch "$held"
gpus_of() { grep -oE 'gpu(:[a-z0-9]+)?:[0-9]+' <<< "$1" | grep -oE '[0-9]+$' | head -1; }
last=
while :; do
    node=$(scontrol show node "$NODE")
    total_c=$(grep -oE 'CPUTot=[0-9]+' <<< "$node" | cut -d= -f2)
    total_g=$(grep -oE 'Gres=gpu:[a-z0-9]+:[0-9]+' <<< "$node" | grep -oE '[0-9]+$')
    read -r other_c other_g < <(squeue -h -o '%u %T %C %b %r' | awk -v me="$USER" '
        $1 != me && ($2 == "RUNNING" || $5 == "Resources" || $5 == "Priority") {
            c += $3
            if (match($4, /gpu(:[a-z0-9]+)?:[0-9]+/)) { n = split(substr($4, RSTART, RLENGTH), a, ":"); g += a[n] }
        } END { print c + 0, g + 0 }')
    cap_c=$(( total_c - other_c - SLACK_CPUS )) cap_g=$(( total_g - other_g ))
    (( cap_c > QOS_CPUS )) && cap_c=$QOS_CPUS
    (( cap_g > QOS_GPUS )) && cap_g=$QOS_GPUS
    cpus=0 gpus=0
    while read -r c b; do
        g=$(gpus_of "$b")
        cpus=$(( cpus + c )) gpus=$(( gpus + ${g:-0} ))
    done < <(squeue -u "$USER" -h -t R -o '%C %b')
    room_c=$(( cap_c - cpus )) room_g=$(( cap_g - gpus ))
    now="ours ≤ $cap_c CPUs, $cap_g GPUs (others use or wait for $other_c CPUs, $other_g GPUs)"
    [ "$now" != "$last" ] && echo "$(date +%T) $now" && last=$now
    while read -r id name c b reason; do
        g=$(gpus_of "$b")
        g=${g:-0}
        ours=0
        grep -qx "$id" "$held" && ours=1
        [ "$reason" = JobHeldUser ] && (( ours == 0 )) && continue
        if [ "$name" = mpd-build ]; then
            (( ours )) && scontrol release "$id" && sed -i "/^$id\$/d" "$held" && echo "$(date +%T) released build $id"
            continue
        fi
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
