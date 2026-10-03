#!/usr/bin/env bash
# Login-node half of mats-run (#2951): never builds or computes here, only fetches and submits.
#
#   remote_submit.sh NAME CPUS MEM_GB MINUTES WANT_COMMIT GPUS QOS CMD_B64 [ARRAY] [CHAIN] [NEEDBIN]
#
# Picks the commit C (WANT_COMMIT when origin/main contains it, else origin/main) and snapshots its
# source into ~/mpd-src/C (the job's working directory). Binaries come from ~/mpd-bin/B for a built
# or queued commit B whose Rust sources (crates/, Cargo.*, rust-toolchain.toml) equal C's; only
# otherwise is C built. Builds run in the debug QOS (2 h) as one Slurm singleton, at 8 CPUs so they
# fit between running jobs; they neither wait behind day-long jobs nor hold CPUs while queued.
# MATS_CHAIN's segments are copies of the run job, each after the last (afterany). Prints the job id.
set -Eeuo pipefail
NAME=$1 CPUS=$2 MEM=$3 MINUTES=$4 WANT=$5 GPUS=$6 QOS=$7 CMD_B64=$8 ARRAY=${9:-} CHAIN=${10:-1} NEEDBIN=${11:-auto}
REPO=$HOME/gam-cluster BIN=$HOME/mpd-bin SRC=$HOME/mpd-src CL=$HOME/mpd-data/cluster
OUT=$CL/$NAME
mkdir -p "$OUT" "$BIN" "$SRC" "$CL/_build"

cat > "$CL/_build/build.sh" <<'BUILD'
#!/usr/bin/env bash
# Build job: compiles every gam-mpd example at commit $1 and installs them in ~/mpd-bin/<commit12>/.
set -Eeuo pipefail
C=$1 C12=${1:0:12}
trap 'echo "${SLURM_JOB_ID:-?} $C" > "$HOME/mpd-bin/$C12.failed"' ERR
source "$HOME/.cargo/env"
exec 9> "$HOME/gam-cluster/.build.lock"
flock 9
cd "$HOME/gam-cluster"
git fetch -q origin main
git checkout -q --detach "$C"
export CARGO_BUILD_JOBS=${SLURM_CPUS_PER_TASK:-16}
# Every job runs on l40-worker, an AMD EPYC 7763 (Zen 3).
export RUSTFLAGS="-C target-cpu=znver3"
echo "== $(date '+%F %T') building $C with $CARGO_BUILD_JOBS jobs"
time cargo build --release -p gam-mpd --examples 2>&1 | grep -vE '^\s+(Compiling|Downloaded|Downloading)' | tail -n 40
dest=$HOME/mpd-bin/$C12
rm -rf "$dest.tmp"
mkdir -p "$dest.tmp"
for f in target/release/examples/*; do
    base=${f##*/}
    [[ -f $f && -x $f && $base != *-* && $base != *.* ]] && cp "$f" "$dest.tmp/"
done
echo "$C" > "$dest.tmp/COMMIT"
rm -rf "$dest"
mv "$dest.tmp" "$dest"
touch "$dest/READY"
echo "== $(date '+%F %T') installed $(ls "$dest" | wc -l) entries in $dest"
# Keep the 12 newest builds. On NFS a binary a running job still executes cannot be removed yet
# (.nfs* placeholders), so pruning is best effort and never fails the build.
ls -1dt "$HOME"/mpd-bin/*/ | tail -n +13 | xargs -r rm -rf 2> /dev/null || true
BUILD

exec 9> "$CL/_build/submit.lock"
flock 9
git -C "$REPO" fetch -q origin main
if git -C "$REPO" cat-file -e "$WANT^{commit}" 2> /dev/null && git -C "$REPO" merge-base --is-ancestor "$WANT" origin/main; then
    C=$(git -C "$REPO" rev-parse "$WANT")
else
    C=$(git -C "$REPO" rev-parse origin/main)
    echo "mats-run: your HEAD ${WANT:0:12} is not on origin/main; running origin/main ${C:0:12}" >&2
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
for d in $(ls -1dt "$BIN"/*/ 2> /dev/null); do
    [ -f "$d/READY" ] && same_rust "$(cat "$d/COMMIT")" "$C" && { B=$(basename "$d"); break; }
done
if [ $need = 0 ]; then
    [ -n "$B" ] || B=none
elif [ -z "$B" ]; then
    for f in $(ls -1t "$BIN"/*.buildjob 2> /dev/null); do
        read -r bj bc < "$f" || true
        [ -n "${bc:-}" ] && alive "$bj" && same_rust "$bc" "$C" && { B=${bc:0:12}; break; }
    done
    if [ -z "$B" ]; then
        bj=$(sbatch --parsable -J mpd-build --dependency=singleton -p compute --qos=debug -c 8 --mem=12G \
            -t 00:30:00 -o "$CL/_build/build-$C12-%j.log" "$CL/_build/build.sh" "$C")
        echo "$bj $C" > "$BIN/$C12.buildjob"
        B=$C12
        echo "mats-run: building $C12 in job $bj (log ~/mpd-data/cluster/_build/build-$C12-$bj.log)" >&2
    else
        echo "mats-run: waiting on build job $bj of $B (same Rust sources as $C12)" >&2
    fi
    dep=(--dependency="afterok:$bj" --kill-on-invalid-dep=yes)
fi
find "$BIN" -maxdepth 1 \( -name '*.buildjob' -o -name '*.failed' \) -mtime +1 -delete

stamp=$(date +%Y%m%d-%H%M%S)-$$
job=$OUT/job-$stamp.sh
cmd=${cmd//@@BIN@@\/src/$SRC/$C12}
cmd=${cmd//@@BIN@@/$BIN/$B}
chained=$(( CHAIN > 1 ))
cat > "$job" <<JOB
#!/usr/bin/env bash
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
$cmd
rc=\$?
echo "== \$(date '+%F %T') exit \$rc after \$(( \$(date +%s) - start )) s"
case \$rc in 143|137|130) ;; *) [ $chained = 1 ] && echo "ended with exit \$rc in job \$SLURM_JOB_ID" > "\$mark" ;; esac
exit \$rc
JOB
chmod +x "$job"

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
