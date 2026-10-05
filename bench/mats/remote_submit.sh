#!/usr/bin/env bash
# Login-node half of mats-run (#2951): never builds or computes here, only fetches and submits.
#
#   remote_submit.sh NAME CPUS MEM_GB MINUTES WANT_COMMIT GPUS QOS CMD_B64 [ARRAY] [CHAIN] [NEEDBIN] [POOL] [SWAP] [AFTEROK]
#
# Requires origin/main to contain WANT_COMMIT and snapshots that exact commit's
# source into ~/mpd-src/C (the job's working directory). Binaries come from ~/mpd-bin/B, a directory
# of binaries built from commits whose Rust sources (crates/, Cargo.*, rust-toolchain.toml) equal C's,
# when it holds (or a queued build will install) every example the command names; only otherwise is
# a build submitted, and it compiles just those examples. Builds run in the debug QOS as one Slurm
# singleton; they neither wait behind day-long jobs nor hold CPUs while queued.
# MATS_CHAIN's segments are copies of the run job, each after the last (afterany). Prints the job id.
set -Eeuo pipefail
NAME=$1 CPUS=$2 MEM=$3 MINUTES=$4 WANT=$5 GPUS=$6 QOS=$7 CMD_B64=$8 ARRAY=${9:-} CHAIN=${10:-1} NEEDBIN=${11:-auto} POOL=${12:-0} SWAP=${13:-} AFTEROK=${14:-}
if [ -n "$AFTEROK" ]; then
    [[ $AFTEROK =~ ^[0-9]+(:[0-9]+)*$ && $POOL = 0 && -z $SWAP ]] || { echo "mats-run: invalid afterok job IDs or unsupported pool/swap dependency" >&2; exit 2; }
fi
HERE_WORKER=$HOME/mpd-data/cluster/_build/pool_worker.sh
REPO=$HOME/gam-cluster BIN=$HOME/mpd-bin SRC=$HOME/mpd-src CL=$HOME/mpd-data/cluster
OUT=$CL/$NAME
mkdir -p "$OUT" "$BIN" "$SRC" "$CL/_build"

# Installed by rename: a build job's bash reads its script while it runs, so rewriting the file in
# place would change the lines a running build executes next.
cat > "$CL/_build/build.sh.$$" <<'BUILD'
#!/usr/bin/env bash
# Build job: build.sh COMMIT DEST [EXAMPLE...] compiles the named gam-mpd examples (every example
# when none is named) at COMMIT and installs them in ~/mpd-bin/DEST/, a directory whose binaries
# all come from commits with COMMIT's Rust sources. Linking one example with thin LTO takes about a
# minute; linking all 59 took 15, and every queued run waited on it.
set -Eeuo pipefail
C=$1 C12=${1:0:12} D=${2:-${1:0:12}}
shift $(( $# < 2 ? $# : 2 ))
trap 'echo "${SLURM_JOB_ID:-?} $C" > "$HOME/mpd-bin/$C12.failed"' ERR
source "$HOME/.cargo/env"
exec 9> "$HOME/gam-cluster/.build.lock"
flock 9
cd "$HOME/gam-cluster"
git fetch -q origin main
# -f: a build can leave Cargo.lock rewritten in this clone; it must never block the next checkout
# (it did, and marked a commit that builds as failed).
git checkout -q -f --detach "$C"
export CARGO_BUILD_JOBS=${SLURM_CPUS_PER_TASK:-16}
# Every job runs on l40-worker, an AMD EPYC 7763 (Zen 3).
export RUSTFLAGS="-C target-cpu=znver3"
which=(--examples)
[ $# -gt 0 ] && which=($(printf -- '--example %s ' "$@"))
echo "== $(date '+%F %T') building ${*:-every example} at $C with $CARGO_BUILD_JOBS jobs"
time cargo build --release -p gam-mpd "${which[@]}" 2>&1 | grep -vE '^\s+(Compiling|Downloaded|Downloading)' | tail -n 40
dest=$HOME/mpd-bin/$D
mkdir -p "$dest"
[ -f "$dest/COMMIT" ] || echo "$C" > "$dest/COMMIT"
# Each binary lands by rename, so a job never runs a half-copied file; same-source binaries replace
# each other harmlessly.
n=0
for f in target/release/examples/*; do
    base=${f##*/}
    [[ -f $f && -x $f && $base != *-* && $base != *.* ]] || continue
    [ $# -eq 0 ] || printf '%s\n' "$@" | grep -qx "$base" || continue
    cp "$f" "$dest/.$base.$$" && mv "$dest/.$base.$$" "$dest/$base" && n=$(( n + 1 ))
done
[ $# -eq 0 ] && touch "$dest/READY"
touch "$dest/.built-${SLURM_JOB_ID:-0}"
echo "== $(date '+%F %T') installed $n binaries in $dest"
BUILD
chmod +x "$CL/_build/build.sh.$$"
mv "$CL/_build/build.sh.$$" "$CL/_build/build.sh"

exec 9> "$CL/_build/submit.lock"
flock 9
git -C "$REPO" fetch -q origin main
if git -C "$REPO" cat-file -e "$WANT^{commit}" 2> /dev/null && git -C "$REPO" merge-base --is-ancestor "$WANT" origin/main; then
    C=$(git -C "$REPO" rev-parse "$WANT")
else
    echo "mats-run: requested commit ${WANT:0:12} is not on origin/main; push it or set MATS_REF to a published commit. No job submitted." >&2
    exit 2
fi
C12=${C:0:12}
if [ ! -d "$SRC/$C12" ]; then
    mkdir -p "$SRC/$C12.tmp"
    git -C "$REPO" archive "$C" | tar -x -C "$SRC/$C12.tmp"
    mv "$SRC/$C12.tmp" "$SRC/$C12"
fi
find "$SRC" -mindepth 1 -maxdepth 1 -mtime +3 -exec rm -rf {} +

same_rust() { git -C "$REPO" diff --quiet "$1" "$2" -- crates Cargo.toml Cargo.lock rust-toolchain.toml .cargo 2> /dev/null; }
alive() { squeue -h -j "$1" -o %T 2> /dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; }
# A command that names no binary (only @@BIN@@/src, say a Python script) neither builds nor waits
# for a build; it gets a ready build of the same Rust sources on its PATH when there is one.
cmd=$(echo "$CMD_B64" | base64 -d)
probe=${cmd//@@BIN@@\/src/}
case $NEEDBIN in 1) need=1 ;; 0) need=0 ;; *) [[ $probe == *@@BIN@@* ]] && need=1 || need=0 ;; esac
B="" dep=()
[ $need = 1 ] && for f in $(ls -1t "$BIN"/*.failed 2> /dev/null); do
    read -r fj fc < "$f" || true
    if [ -n "${fc:-}" ] && same_rust "$fc" "$C"; then
        log=$(ls "$CL"/_build/build-"${fc:0:12}"-"$fj".log 2> /dev/null || true)
        echo "mats-run: $C12 does not build (same Rust sources as ${fc:0:12}, build job $fj); not submitting." >&2
        [ -n "$log" ] && grep -E -A6 '^error' "$log" | head -n 30 >&2
        echo "mats-run: pin a commit that builds with MATS_REF=<commit>" >&2
        exit 3
    fi
done
# The examples the command runs. A command that names none (it runs them through PATH) needs them all.
names=$(grep -oE '@@BIN@@/[A-Za-z0-9_]+' <<< "$probe" | sed 's|^@@BIN@@/||' | sort -u | tr '\n' ' ' || true)
has() { local d=$1 n; [ -z "$names" ] && { [ -f "$d/READY" ]; return; }; for n in $names; do [ -f "$d/$n" ] || return 1; done; }
same=""
for d in $(ls -1dt "$BIN"/*/ 2> /dev/null); do
    d=${d%/}
    [ -f "$d/COMMIT" ] && same_rust "$(cat "$d/COMMIT")" "$C" || continue
    [ -n "$same" ] || same=$(basename "$d")
    has "$d" && { B=$(basename "$d"); break; }
done
ready=""
[ -n "$B" ] && ready=$BIN/$B/COMMIT
if [ $need = 0 ]; then
    [ -n "$B" ] || B=${same:-none}
elif [ -z "$B" ]; then
    # A queued build of the same sources that installs every example this command names.
    for f in $(ls -1t "$BIN"/*.buildjob 2> /dev/null); do
        read -r bj bc bd bn < "$f" || true
        [ -n "${bc:-}" ] && alive "$bj" && same_rust "$bc" "$C" || continue
        bd=${bd:-${bc:0:12}} bn=${bn:-}
        if [ -z "$bn" ] || { [ -n "$names" ] && [ -z "$(comm -23 <(tr ' ' '\n' <<< "$names" | sed '/^$/d') <(tr ' ' '\n' <<< "$bn" | sed '/^$/d' | sort -u))" ]; }; then
            B=$bd
            break
        fi
    done
    if [ -z "$B" ]; then
        B=${same:-$C12}
        # 32 compile and link jobs; the measured peak of 8-job builds was 3.7 GB.
        bj=$(sbatch --parsable -J mpd-build --dependency=singleton -p compute --qos=debug -c 32 --mem=24G \
            -t 00:30:00 -o "$CL/_build/build-$C12-%j.log" "$CL/_build/build.sh" "$C" "$B" $names)
        echo "$bj $C $B $names" > "$BIN/$C12-$bj.buildjob"
        echo "mats-run: building ${names:-every example} at $C12 into $B in job $bj (log ~/mpd-data/cluster/_build/build-$C12-$bj.log)" >&2
    else
        echo "mats-run: waiting on build job $bj into $B (same Rust sources as $C12)" >&2
    fi
    ready=$BIN/$B/.built-$bj
    dep=(--dependency="afterok:$bj" --kill-on-invalid-dep=yes)
fi
if [ -n "$AFTEROK" ]; then
    if [ ${#dep[@]} -gt 0 ]; then
        dep[0]+=":$AFTEROK"
    else
        dep=(--dependency="afterok:$AFTEROK" --kill-on-invalid-dep=yes)
    fi
fi
find "$BIN" -maxdepth 1 \( -name '*.buildjob' -o -name '*.failed' \) -mtime +1 -delete
# Prune old builds here (the compute node cannot see the queue): beyond the 12 newest, a build goes
# only when it is 2 days old and no queued, running or pool job's script names it. Best effort: NFS
# keeps a binary a running process still executes.
inuse=$( { squeue -u "$USER" -h -o %o 2> /dev/null; sed -n 's/^# script=//p' "$CL"/queue/{todo,running}/*.task 2> /dev/null; } |
    sort -u | xargs -r grep -ohE "$BIN/[0-9a-f]{12}" 2> /dev/null | sort -u || true) || true
for d in $(ls -1dt "$BIN"/*/ | tail -n +13); do
    d=${d%/}
    [ -n "$(find "$d" -maxdepth 0 -mtime +2)" ] && ! grep -qx "$d" <<< "$inuse" && rm -rf "$d" 2> /dev/null
done
true

stamp=$(date +%Y%m%d-%H%M%S)-$$
job=$OUT/job-$stamp.sh
cmd=${cmd//@@BIN@@\/src/$SRC/$C12}
cmd=${cmd//@@BIN@@/$BIN/$B}
# Memory: at most 2 x this command's measured peak + 4 (cluster_peaks.sh), unless asked "=N" exactly.
ask=$(bash "$CL/_build/cluster_peaks.sh" ask "$cmd" 2> /dev/null || true)
exact=""
if [[ $MEM == =* ]]; then
    MEM=${MEM#=} exact="# mats-mem=exact (the audit leaves this job's memory alone)"
elif (( MEM == 0 )); then
    MEM=${ask:-$(( 2 * CPUS > 8 ? 2 * CPUS : 8 ))}
    if [ -n "$ask" ]; then why="2 x measured peak + 4"; else why="nothing measured yet: 2 GB per CPU, at least 8"; fi
    echo "mats-run: asking ${MEM}G ($why)" >&2
elif [ -n "$ask" ] && (( MEM > ask )); then
    echo "mats-run: asking ${ask}G, not ${MEM}G: this command's measured peak is $(bash "$CL/_build/cluster_peaks.sh" lookup "$cmd")G (MATS_MEM_EXACT=1 keeps ${MEM}G)" >&2
    MEM=$ask
fi
chained=$(( CHAIN > 1 ))
cat > "$job" <<JOB
#!/usr/bin/env bash
$exact
export RAYON_NUM_THREADS=\$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MPD_BIN=$BIN/$B GAM_SRC=$SRC/$C12 MPD_DATA=\$HOME/mpd-data MATS_OUT=$OUT
export PATH=$BIN/$B:\$HOME/mpd-venv/bin:\$HOME/.cargo/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:/usr/local/cuda-12.2/targets/x86_64-linux/lib:\${LD_LIBRARY_PATH:-}
# The node does not confine devices: a job Slurm gave no GPU would otherwise see (and take) all 8.
export CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES-}
cd $SRC/$C12
# A chain's later segments rerun this script; a task that already ended (exit 0, or a real error)
# skips at once, and one stopped by its time limit (SIGTERM, 143) resumes from its checkpoint.
mark=$OUT/.chain-$stamp-\${SLURM_ARRAY_TASK_ID:-0}
if [ $chained = 1 ] && [ -e "\$mark" ]; then echo "== chain segment \$SLURM_JOB_ID skipped: \$(cat "\$mark")"; exit 0; fi
echo "== \$(date '+%F %T') job \$SLURM_JOB_ID\${SLURM_ARRAY_TASK_ID:+ (task \$SLURM_ARRAY_TASK_ID of \$SLURM_ARRAY_JOB_ID)} on \$(hostname): source $C, binaries $B, cpus \$SLURM_CPUS_PER_TASK, mem ${MEM}G, gpus ${GPUS}"
echo "== $cmd"
start=\$(date +%s)
# mats-swap JOBID -- COMMAND writes the next command to \$next and stops this one (its process
# group): the allocation stays and runs the new command, with no requeue and no wait.
set -m
next=$OUT/.next-\$SLURM_JOB_ID\${SLURM_ARRAY_TASK_ID:+_\$SLURM_ARRAY_TASK_ID}\${MATS_POOL_TASK:+-\$MATS_POOL_TASK}
trap 'kill -TERM -- -\$cpid 2> /dev/null' TERM INT
$cmd &
cpid=\$!
echo \$cpid > "\$next.pid"
wait \$cpid
rc=\$?
while [ -s "\$next" ]; do
    mv "\$next" "\$next.run"
    echo "== \$(date '+%F %T') swapped in: \$(tail -n 1 "\$next.run")"
    bash "\$next.run" &
    cpid=\$!
    echo \$cpid > "\$next.pid"
    wait \$cpid
    rc=\$?
done
rm -f "\$next.pid" "\$next.run"
echo "== \$(date '+%F %T') exit \$rc after \$(( \$(date +%s) - start )) s"
case \$rc in 143|137|130) ;; *) [ $chained = 1 ] && echo "ended with exit \$rc in job \$SLURM_JOB_ID" > "\$mark" ;; esac
exit \$rc
JOB
chmod +x "$job"

if [ -n "$SWAP" ]; then
    # Run COMMAND inside the running job SWAP's allocation in place of what it runs now.
    info=$(scontrol show job "$SWAP" 2> /dev/null)
    raw=$(grep -oP '^JobId=\K[0-9]+' <<< "$info" | head -1 || true)
    [ -n "$raw" ] || { echo "mats-swap: no job $SWAP" >&2; exit 2; }
    task=$(grep -oP 'ArrayTaskId=\K[0-9]+' <<< "$info" | head -1 || true)
    jout=$(dirname "$(grep -oP 'StdOut=\K\S+' <<< "$info")")
    [ "$(grep -oP 'JobState=\K\S+' <<< "$info")" = RUNNING ] || { echo "mats-swap: job $SWAP is not running" >&2; exit 2; }
    if [ "$need" = 1 ] && [ ! -f "$ready" ]; then
        echo "mats-swap: binaries $B are still building (${dep[*]}); swap again once it finishes" >&2; exit 3
    fi
    next=$jout/.next-$raw${task:+_$task}
    sed -n '1,/^cd /p' "$job" > "$next.tmp"
    echo "echo \"== \$(date '+%F %T') swapped: source $C, binaries $B\"" >> "$next.tmp"
    echo "$cmd" >> "$next.tmp"
    mv "$next.tmp" "$next"
    pidf=$next.pid
    if [ ! -s "$pidf" ]; then
        rm -f "$next"
        echo "mats-swap: job $SWAP was not started by a mats-run that supports swapping (no $pidf)" >&2; exit 2
    fi
    srun --jobid="$raw" --overlap -n1 -c1 --mem=64M -t 1 kill -TERM -- "-$(cat "$pidf")" 2> /dev/null ||
        srun --jobid="$raw" --overlap -n1 -c1 --mem=64M -t 1 kill -TERM "$(cat "$pidf")"
    echo "mats-swap: job $SWAP now runs: $cmd" >&2
    echo "$SWAP"
    exit 0
fi

if [ "$POOL" = 1 ]; then
    # Queue the job's tasks for the pool workers instead of submitting it (pool_worker.sh).
    Q=$CL/queue
    mkdir -p "$Q/todo" "$Q/running" "$Q/done" "$Q/workers"
    cp "$HERE_WORKER" "$Q/pool_worker.sh"
    [ "$need" = 1 ] || ready=""
    idx=("")
    if [ -n "$ARRAY" ]; then
        idx=()
        for part in $(tr ',' ' ' <<< "${ARRAY%%%*}"); do
            if [[ $part == *-* ]]; then
                step=1; [[ $part == *:* ]] && step=${part##*:} part=${part%%:*}
                for (( i = ${part%%-*}; i <= ${part##*-}; i += step )); do idx+=("$i"); done
            else
                idx+=("$part")
            fi
        done
    fi
    for i in "${idx[@]}"; do
        t=$stamp-$NAME${i:+-$i}.task
        printf '# name=%s\n# cpus=%s\n# mem=%s\n# gpus=%s\n# minutes=%s\n# ready=%s\n# script=%s\n# log=%s\n# task=%s\n' \
            "$NAME" "$CPUS" "$MEM" "$GPUS" "$MINUTES" "$ready" "$job" "$OUT/pool-$stamp${i:+-$i}.log" "$i" > "$Q/todo/$t.tmp"
        mv "$Q/todo/$t.tmp" "$Q/todo/$t"
    done
    echo "pool $stamp src=$C12 bin=$B cpus=$CPUS mem=${MEM}G min=$MINUTES gpus=$GPUS tasks=${#idx[@]}" >> "$OUT/JOBS"
    echo "mats-run: queued ${#idx[@]} task(s) as $stamp-$NAME* in ~/mpd-data/cluster/queue/todo" >&2
    # Start a worker unless one is pending or a running one has room for a task of this size now.
    room=0
    for w in "$Q"/workers/*; do
        [ -f "$w" ] && squeue -h -j "${w##*/}" -t R 2> /dev/null | grep -q . || continue
        read -r wc wm wg uc um ug wl < "$w"
        (( wc - uc >= CPUS && wm - um >= MEM && wg - ug >= GPUS && wl >= MINUTES + 2 )) && room=1
    done
    if [ $room = 0 ] && [ -z "$(squeue -h -u "$USER" -n mpd-pool -t PD)" ]; then
        k=$(( ${#idx[@]} < 32 / CPUS ? ${#idx[@]} : 32 / CPUS ))
        (( k < 1 )) && k=1
        rounds=$(( (${#idx[@]} + k - 1) / k ))
        wmin=$(( MINUTES * rounds + 15 ))
        (( wmin > 1440 )) && wmin=1440
        wargs=(--parsable -J mpd-pool -p compute -c $(( CPUS * k )) --mem="$(( MEM * k ))G" -t "$wmin" -o "$Q/worker-%j.log" "${dep[@]}")
        (( GPUS > 0 )) && wargs+=(--gres="gpu:$(( GPUS * k ))")
        [ -n "$QOS" ] && wargs+=(--qos="$QOS")
        wj=$(sbatch "${wargs[@]}" "$Q/pool_worker.sh" "$wmin")
        echo "mats-run: started pool worker $wj ($(( CPUS * k )) CPUs, $(( MEM * k ))G, $(( GPUS * k )) GPUs, $wmin min)" >&2
    fi
    echo "$stamp"
    exit 0
fi

log=$OUT/slurm-%j.log
[ -n "$ARRAY" ] && log=$OUT/slurm-%A_%a.log
jids=()
for (( seg = 1; seg <= CHAIN; seg++ )); do
    args=(--parsable -J "$NAME" -p compute -c "$CPUS" --mem="${MEM}G" -t "$MINUTES" -o "$log" "${dep[@]}")
    [ -n "$ARRAY" ] && args+=(--array="$ARRAY")
    [ "$GPUS" != 0 ] && args+=(--gres="gpu:$GPUS")
    [ -n "$QOS" ] && args+=(--qos="$QOS")
    jid=$(sbatch "${args[@]}" "$job")
    jids+=("$jid")
    echo "$jid $stamp src=$C12 bin=$B cpus=$CPUS mem=${MEM}G min=$MINUTES gpus=$GPUS${QOS:+ qos=$QOS}${ARRAY:+ array=$ARRAY}$( (( CHAIN > 1 )) && echo " chain=$seg/$CHAIN")" >> "$OUT/JOBS"
    dep=(--dependency="afterany:$jid")
done
(( CHAIN > 1 )) && echo "mats-run: chain of $CHAIN segments: ${jids[*]}" >&2
echo "${jids[0]}"
