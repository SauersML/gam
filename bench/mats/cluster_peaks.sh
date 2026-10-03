#!/usr/bin/env bash
# Measured memory peaks of our cluster jobs, keyed by command signature (#2951). Login node only.
#
#   cluster_peaks.sh sig 'COMMAND'     the signature: program, script (for python/bash), and the bare
#                                      arguments without paths (so n=1e5 and n=1e6 differ, passages don't)
#   cluster_peaks.sh lookup 'COMMAND'  the largest measured peak in GB for its signature, or nothing
#   cluster_peaks.sh ask 'COMMAND'     the memory to ask in GB: 2 x peak + 4, or nothing when unmeasured
#   cluster_peaks.sh update            fold finished mats-run jobs (sacct MaxRSS of their batch steps)
#                                      and jobs running for 15+ minutes (sstat) into the tables
#
# Node RAM, not CPUs, is what fills the node, so mats-run asks for `ask` and the audit shrinks
# pending jobs that ask for more.
set -uo pipefail
CL=$HOME/mpd-data/cluster
DB=$CL/_peaks.tsv LIVE=$CL/_peaks_live.tsv SEEN=$CL/_peaks.seen
touch "$DB" "$LIVE" "$SEEN"
sig() {
    awk '{
        prog = ""; scr = ""; bare = ""
        for (i = 1; i <= NF; i++) {
            w = $i
            if (prog == "" && (w == "env" || w ~ /^[A-Za-z_][A-Za-z0-9_]*=/)) continue
            if (prog == "") { sub(/.*\//, "", w); prog = w; continue }
            if (scr == "" && (prog ~ /^python/ || prog == "bash") && w ~ /\//) { sub(/.*\//, "", w); scr = w; continue }
            if (w ~ /[\/$"\\]/ || length(w) > 24) continue
            bare = bare " " w
        }
        print prog (scr != "" ? " " scr : "") bare
    }' <<< "$1"
}
gb() { awk '{ v = $1; u = substr(v, length(v)); n = v + 0; if (u == "K") n /= 1048576; else if (u == "M") n /= 1024; else if (u == "T") n *= 1024; else if (u ~ /[0-9]/) n /= 1073741824; printf "%.1f\n", n }'; }
cmd_of() { sed -n 's/^echo "== \(.*\)"$/\1/p' "$1" | sed -n 2p; }
lookup() {
    local s
    s=$(sig "$1")
    S=$s awk -F'\t' '$1 == ENVIRON["S"] && $2 > m { m = $2 } END { if (m > 0) print m }' "$DB" "$LIVE"
}
record() {
    local s=$1 p=$2
    S=$s awk -F'\t' -v OFS='\t' -v p="$p" '$1 == ENVIRON["S"] { if (p > $2) $2 = p; $3++; f = 1 } { print } END { if (!f) print ENVIRON["S"], p, 1 }' "$DB" > "$DB.tmp" && mv "$DB.tmp" "$DB"
}
case ${1:-} in
    sig) sig "$2" ;;
    lookup) lookup "$2" ;;
    ask)
        p=$(lookup "$2")
        [ -n "$p" ] && awk -v p="$p" 'BEGIN { a = 2 * p + 4; print (a == int(a)) ? a : int(a) + 1 }' ;;
    update)
        for f in $(find "$CL" -mindepth 2 -maxdepth 2 -name JOBS -mtime -3); do
            d=${f%/JOBS}
            while read -r jid stamp _; do
                [[ $jid =~ ^[0-9]+$ ]] || continue
                grep -qx "$jid" "$SEEN" && continue
                states=$(sacct -n -X -P -j "$jid" -o State 2> /dev/null)
                [ -z "$states" ] || grep -qE 'PENDING|RUNNING|REQUEUED|SUSPENDED' <<< "$states" && continue
                echo "$jid" >> "$SEEN"
                [ -f "$d/job-$stamp.sh" ] || continue
                p=$(sacct -n -P -j "$jid" -o JobID,MaxRSS 2> /dev/null | awk -F'|' '$1 ~ /\.batch$/ && $2 != "" { print $2 }' | gb | sort -g | tail -1)
                awk -v p="${p:-0}" 'BEGIN { exit !(p > 0.05) }' && record "$(sig "$(cmd_of "$d/job-$stamp.sh")")" "$p"
            done < "$f"
        done
        : > "$LIVE.tmp"
        while read -r id el; do
            (( $(awk -F'[-:]' '{ n = NF; s = $n + 60 * $(n - 1); if (n >= 3) s += 3600 * $(n - 2); if (n == 4) s += 86400 * $1; print int(s / 60) }' <<< "$el") >= 15 )) || continue
            raw=$(scontrol show job "$id" 2> /dev/null | grep -oP '^JobId=\K[0-9]+' | head -1)
            script=$(scontrol show job "$id" 2> /dev/null | grep -oP 'Command=\K\S+' | head -1)
            [ -f "$script" ] || continue
            rss=$(sstat -n -P -j "$raw.batch" -o MaxRSS 2> /dev/null | head -1)
            [ -n "$rss" ] && printf '%s\t%s\t1\n' "$(sig "$(cmd_of "$script")")" "$(gb <<< "$rss")" >> "$LIVE.tmp"
        done < <(squeue -u "$USER" -h -t R -o '%i %M')
        mv "$LIVE.tmp" "$LIVE" ;;
    *) sed -n '2,13p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
