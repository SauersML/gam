#!/usr/bin/env bash
# One status snapshot of the cluster for the Mac's shared cache (#2951). Run by mats-poll on the
# login node, once per poll, over its single ssh connection; prints a gzipped tar on stdout:
#   STAMP            when the snapshot was taken (cluster clock)
#   squeue.txt       every user's jobs;  squeue_ours.txt  ours, one line per array task
#   sacct.txt        our jobs of the last 2 days (state, elapsed, CPUs, memory, exit code)
#   node.txt         the node's allocation;  gpu.txt (gpu_watch.sh: each GPU's utilisation and holder,
#                    each of our jobs' CPU efficiency, WASTE flags), budget.log, peaks.tsv, pool.txt
#   audit.txt        cluster_audit.sh's findings (when the first argument is 1)
#   files/...        under their path relative to the cluster home: every mats-run job directory
#                    touched in 2 days (JOBS, the last 400 lines of each log, JSON and text up to
#                    1 MB), the pool queue's task files, and each path or glob in watch.txt
#                    (relative to the home; a directory is listed, a file copied or tailed)
#   cluster_snapshot.sh AUDIT   (in the directory holding cluster_audit.sh and watch.txt)
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
cd "$HOME" || exit 1
date -u '+%F %T UTC' > "$T/STAMP"
squeue -o '%.14i %.10u %.28j %.2t %.6q %.4C %.7m %.12b %.10M %.10l %.6y %R' > "$T/squeue.txt" 2>&1
squeue -u "$USER" -r -o '%.16i %.28j %.2t %.6q %.4C %.7m %.12b %.10M %.10l %.6y %R' > "$T/squeue_ours.txt" 2>&1
sacct -u "$USER" -S now-2days -X -P -o JobID,JobName%40,State,Elapsed,AllocCPUS,ReqMem,ExitCode,Start,End > "$T/sacct.txt" 2>&1
scontrol show node l40-worker > "$T/node.txt" 2>&1
cp gpu_watch.txt "$T/gpu.txt" 2> /dev/null
tail -n 100 budget.log > "$T/budget.log" 2> /dev/null
cp mpd-data/cluster/_peaks.tsv "$T/peaks.tsv" 2> /dev/null
Q=mpd-data/cluster/queue
{
    echo "todo $(ls $Q/todo 2> /dev/null | grep -c '\.task$') running $(ls $Q/running 2> /dev/null | grep -c '\.task$') done $(ls $Q/done 2> /dev/null | wc -l)"
    for w in $Q/workers/*; do [ -f "$w" ] && echo "worker ${w##*/}: $(cat "$w") (cpus mem gpus, used cpus mem gpus, minutes left)"; done
} > "$T/pool.txt"
[ "${1:-0}" = 1 ] && [ -f "$here/cluster_audit.sh" ] && bash "$here/cluster_audit.sh" > "$T/audit.txt" 2>&1
copy() {
    local f=$1
    [ -f "$f" ] || return 0
    mkdir -p "$T/files/$(dirname "$f")"
    case $f in
        *.log | *.out | *.err) tail -n 400 "$f" > "$T/files/$f" ;;
        *) [ "$(stat -c %s "$f")" -le 1048576 ] && cp "$f" "$T/files/$f" ;;
    esac
}
while read -r f; do
    copy "$f"
done < <(find mpd-data/cluster -mindepth 2 -maxdepth 3 -mmin -2880 -type f \
    \( -name JOBS -o -name '*.log' -o -name '*.json' -o -name '*.txt' -o -name '*.task' \) \
    -not -path 'mpd-data/cluster/_status*' 2> /dev/null)
if [ -f "$here/watch.txt" ]; then
    while read -r pattern; do
        [ -n "$pattern" ] || continue
        pattern=${pattern#"$HOME"/}
        for p in $pattern; do
            if [ -d "$p" ]; then
                mkdir -p "$T/files/$p"
                ls -la "$p" > "$T/files/$p/.ls" 2> /dev/null
            else
                copy "$p"
            fi
        done
    done < "$here/watch.txt"
fi
tar -czf - -C "$T" .
