#!/usr/bin/env bash
# Good-citizen audit of our MATS jobs (#2951), run on the login node (squeue/sstat/sacct only):
#   ssh mats bash -s < cluster_audit.sh
# Prints one line per finding: IDLE (CPU use under 25% of the allocation), LOW (under 50%: ask fewer
# CPUs next time), STALE (log silent 45 min),
# MEM (asks far above the measured peak), GPU (holds a GPU with no process on it), THROTTLE (debug
# array without a %4-8 cap), OTHERS (another user's job waiting on resources we hold).
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
while read -r id name qos; do
    [ "$qos" = debug ] || continue
    t=$(scontrol show job "${id%%_*}" 2> /dev/null | grep -oP 'ArrayTaskThrottle=\K[0-9]+' | head -1)
    [ -z "$t" ] || [ "$t" = 0 ] || (( t > 8 )) && echo "THROTTLE ${id%%_*} $name: debug array throttle ${t:-none}"
done < <(squeue -u "$USER" -h -t PD -r -o '%i %j %q' | grep '_' | awk '!seen[$2]++')
squeue -h -t PD -o '%i %u %j %C %m %r' | awk -v me="$USER" '$2 != me && ($6 == "Resources" || $6 == "Priority") { print "OTHERS " $0 }'
true
