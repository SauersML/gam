#!/usr/bin/env bash
# What the MATS node's GPUs and our jobs' CPUs are actually doing (#2951). Run on the login node:
#   setsid nohup bash gpu_watch.sh >> ~/gpu_watch.log 2>&1 < /dev/null &
# Every minute it reads every L40's utilisation, memory and power (nvidia-smi through a job step
# overlapping one of our running jobs: the login node has no GPUs and the node's devices are
# visible from any job), names the job holding each GPU, and measures each of our running jobs'
# CPU efficiency (CPU time over elapsed × allocated CPUs, from sstat). ~/gpu_watch.txt holds the
# latest table; ~/gpu_watch.log one line per GPU per minute. A job of ours that holds a GPU at
# under 5% utilisation for 10 minutes, or uses under a quarter of its CPUs after its first 10
# minutes, is marked WASTE in both, so whoever owns it can shrink or move it.
set -uo pipefail
NODE=${NODE:-l40-worker}
declare -A idle
while :; do
    T=$(mktemp)
    stamp=$(date '+%F %T')
    host=$(squeue -u "$USER" -h -t R -o '%i' | head -1)
    declare -A owner=()
    while read -r id user name; do
        idx=$(scontrol show job -d "$id" | grep -oE 'IDX:[0-9,-]+' | head -1 | cut -d: -f2)
        for part in ${idx//,/ }; do
            if [[ $part == *-* ]]; then
                for ((i = ${part%-*}; i <= ${part#*-}; i++)); do owner[$i]="$id $user $name"; done
            else
                owner[$part]="$id $user $name"
            fi
        done
    done < <(squeue -h -t R -o '%i %u %j %b' | awk '$4 ~ /gpu/ {print $1, $2, $3}')
    {
        echo "$stamp  GPU  util  mem(MiB)  power  job"
        if [ -n "$host" ]; then
            while IFS=', ' read -r i u m p; do
                o=${owner[$i]:-free}
                flag=
                if [[ $o == *" $USER "* ]]; then
                    id=${o%% *}
                    if (( ${u%\%} < 5 )); then idle[$id]=$(( ${idle[$id]:-0} + 1 )); else idle[$id]=0; fi
                    (( ${idle[$id]:-0} >= 10 )) && flag="  WASTE: idle ${idle[$id]} min"
                fi
                printf '%s  %3s  %4s%%  %8s  %5sW  %s%s\n' "$stamp" "$i" "${u%\%}" "$m" "${p%.*}" "$o" "$flag"
                echo "$stamp gpu $i util ${u%\%} mem $m job ${o// /_}" >> "$T.log"
            done < <(timeout 30 srun --jobid="$host" --overlap -N1 -n1 nvidia-smi \
                --query-gpu=index,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits 2> /dev/null)
        else
            echo "$stamp  (no running job of ours to read the GPUs through)"
        fi
        echo
        echo "$stamp  our jobs: CPU efficiency (CPU time / elapsed × CPUs)"
        while read -r id name cpus elapsed; do
            secs=$(awk -F'[-:]' '{ if (NF == 4) print $1*86400 + $2*3600 + $3*60 + $4; else if (NF == 3) print $1*3600 + $2*60 + $3; else print $1*60 + $2 }' <<< "$elapsed")
            used=$(sstat -a -n -P -j "$id" -o JobID,AveCPU,NTasks 2> /dev/null | awk -F'|' '{
                t = $2; d = 0
                if (index(t, "-")) { split(t, a, "-"); d = a[1]; t = a[2] }
                m = split(t, b, ":"); s = 0
                for (i = 1; i <= m; i++) s = s * 60 + b[i]
                tot += (d * 86400 + s) * ($3 > 0 ? $3 : 1) } END { print int(tot) }')
            eff=$(( secs > 0 && cpus > 0 ? 100 * ${used:-0} / (secs * cpus) : 0 ))
            flag=
            (( secs > 600 && eff < 25 )) && flag="  WASTE: ${eff}% of $cpus CPUs"
            printf '%s  %-8s %-30s %4s CPUs  %4s%%%s\n' "$stamp" "$id" "$name" "$cpus" "$eff" "$flag"
        done < <(squeue -u "$USER" -h -t R -o '%i %j %C %M')
    } > "$T"
    mv -f "$T" "$HOME/gpu_watch.txt"
    [ -f "$T.log" ] && cat "$T.log" >> "$HOME/gpu_watch.log" && rm -f "$T.log"
    grep WASTE "$HOME/gpu_watch.txt" >> "$HOME/gpu_watch.log"
    sleep 60
done
