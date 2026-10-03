#!/usr/bin/env bash
# Good-citizen audit of our MATS jobs (#2951), run on the login node (squeue/sstat/sacct only):
#   ssh mats bash -s < cluster_audit.sh
# Prints one line per finding: IDLE (CPU use under 25% of the allocation), LOW (under 50%: ask fewer
# CPUs next time), STALE (log silent 45 min),
# MEM (asks far above the measured peak), GPU (holds a GPU with no process on it), THROTTLE (debug
# array without a %4-8 cap), SHRUNK (a pending job or pool task cut to 2 x peak + 4), GPUS (more than our 6 across QOSes), OTHERS (another user's job waiting on resources we hold).
set -uo pipefail
secs() { awk -F'[-:]' '{ n = NF; s = $n + 60 * $(n - 1); if (n >= 3) s += 3600 * $(n - 2); if (n == 4) s += 86400 * $1; print s }' <<< "$1"; }
gb() { awk '{ v = $1; u = substr(v, length(v)); n = v + 0; if (u == "K") n /= 1048576; else if (u == "M") n /= 1024; else if (u == "T") n *= 1024; printf "%.1f", n }' <<< "$1"; }
now=$(date +%s)
while read -r id name cpus mem el gres; do
    [ "$name" = mpd-build ] && continue
    e=$(secs "$el")
    (( e < 900 )) && continue
    jid=$(scontrol show job "$id" 2> /dev/null | grep -oP '^JobId=\K[0-9]+' | head -1)
    rss="" avecpu=""
    read -r rss avecpu < <(sstat -n -P -j "${jid:-$id}.batch" -o MaxRSS,AveCPU 2> /dev/null | head -1 | tr '|' ' ')
    if [ -n "${avecpu:-}" ]; then
        eff=$(awk -v c="$(secs "$avecpu")" -v e="$e" -v n="$cpus" 'BEGIN { printf "%d", 100 * c / (e * n) }')
        if (( eff < 25 )); then echo "IDLE $id $name: ${eff}% of $cpus CPUs busy over $el"
        elif (( eff < 50 && cpus > 4 )); then echo "LOW $id $name: ${eff}% of $cpus CPUs busy over $el"; fi
    fi
    if [ -n "${rss:-}" ]; then
        ask=$(gb "$mem") peak=$(gb "$rss")
        awk -v a="$ask" -v p="$peak" 'BEGIN { exit !(a > 2 * p + 4) }' && echo "MEM $id $name: asks ${ask}G, peak ${peak}G"
    fi
    log=$(scontrol show job "$id" 2> /dev/null | sed -n 's/^ *StdOut=//p')
    if [ -f "$log" ]; then
        quiet=$(( now - $(stat -c %Y "$log") ))
        (( quiet > 2700 )) && echo "STALE $id $name: log silent $(( quiet / 60 )) min ($log)"
    fi
    if [[ $gres == *gpu* ]] && (( e > 600 )); then
        procs=$(timeout 20 srun --jobid="${jid:-$id}" --overlap -n1 -c1 --mem=100M nvidia-smi --query-compute-apps=pid --format=csv,noheader 2> /dev/null | grep -c . || true)
        [ "${procs:-0}" = 0 ] && echo "GPU $id $name: holds a GPU with no process on it"
    fi
done < <(squeue -u "$USER" -h -t R -o '%i %j %C %m %M %b')
# Pending jobs and queued pool tasks asking more than 2 x their command's measured peak + 4 are cut
# to that in place (cluster_peaks.sh; a job submitted with MATS_MEM_EXACT=1 is left alone).
P=$HOME/mpd-data/cluster/_build/cluster_peaks.sh
if [ -f "$P" ]; then
    bash "$P" update > /dev/null 2>&1
    cmd_of() { sed -n 's/^echo "== \(.*\)"$/\1/p' "$1" | sed -n 2p; }
    while read -r id name m; do
        id=${id%%_*}
        script=$(scontrol show job "$id" 2> /dev/null | grep -oP 'Command=\K\S+' | head -1)
        [ -f "$script" ] && ! grep -q '^# mats-mem=exact' "$script" || continue
        ask=$(bash "$P" ask "$(cmd_of "$script")")
        have=$(gb "$m" | awk '{ print int($1 + 0.5) }')
        if [ -n "$ask" ] && (( have > ask )); then
            scontrol update jobid="$id" MinMemoryNode=$(( ask * 1024 )) && echo "SHRUNK $id $name: ${have}G -> ${ask}G (2 x measured peak + 4)"
        fi
    done < <(squeue -u "$USER" -h -t PD -o '%i %j %m' | awk '!seen[$1]++')
    for t in "$HOME"/mpd-data/cluster/queue/todo/*.task; do
        [ -f "$t" ] || continue
        script=$(sed -n 's/^# script=//p' "$t")
        [ -f "$script" ] && ! grep -q '^# mats-mem=exact' "$script" || continue
        ask=$(bash "$P" ask "$(cmd_of "$script")")
        have=$(sed -n 's/^# mem=//p' "$t")
        if [ -n "$ask" ] && (( have > ask )); then
            sed -i "s/^# mem=.*/# mem=$ask/" "$t" && echo "SHRUNK pool task ${t##*/}: ${have}G -> ${ask}G"
        fi
    done
fi
while read -r id name qos; do
    [ "$qos" = debug ] || continue
    t=$(scontrol show job "${id%%_*}" 2> /dev/null | grep -oP 'ArrayTaskThrottle=\K[0-9]+' | head -1)
    [ -z "$t" ] || [ "$t" = 0 ] || (( t > 8 )) && echo "THROTTLE ${id%%_*} $name: debug array throttle ${t:-none}"
done < <(squeue -u "$USER" -h -t PD -r -o '%i %j %q' | awk '$1 ~ /_/ && !seen[$2]++')
g=$(squeue -u "$USER" -h -t R -o '%b' | grep -oE 'gpu:[0-9]+' | cut -d: -f2 | paste -sd+ | bc 2> /dev/null)
(( ${g:-0} > 6 )) && echo "GPUS we hold ${g} GPUs, above the 6 the QOS gives us (debug has no GPU cap)"
squeue -h -t PD -o '%i %u %j %C %m %r' | awk -v me="$USER" '$2 != me && ($6 == "Resources" || $6 == "Priority") { print "OTHERS " $0 }'
true
