#!/usr/bin/env bash
# Login-node half of mats-run (#2951): never builds or computes here, only fetches and submits.
#
#   remote_submit.sh NAME CPUS MEM_GB MINUTES WANT_COMMIT GPUS QOS CMD_B64
#
# Picks the commit (WANT_COMMIT when origin/main contains it, else origin/main), submits a build job
# for it unless ~/mpd-bin/<commit> is already built (or a build of it is queued), writes the run
# script under ~/mpd-data/cluster/NAME/ and submits it after the build. Prints the run job id.
set -Eeuo pipefail
NAME=$1 CPUS=$2 MEM=$3 MINUTES=$4 WANT=$5 GPUS=$6 QOS=$7 CMD_B64=$8
REPO=$HOME/gam-cluster BIN=$HOME/mpd-bin CL=$HOME/mpd-data/cluster
OUT=$CL/$NAME
mkdir -p "$OUT" "$BIN" "$CL/_build"

cat > "$CL/_build/build.sh" <<'BUILD'
#!/usr/bin/env bash
# Build job: compiles every gam-mpd example at commit $1 and installs them in ~/mpd-bin/<commit12>/.
set -Eeuo pipefail
C=$1 C12=${1:0:12}
source "$HOME/.cargo/env"
exec 9> "$HOME/gam-cluster/.build.lock"
flock 9
cd "$HOME/gam-cluster"
git fetch -q origin main
git checkout -q --detach "$C"
export CARGO_BUILD_JOBS=${SLURM_CPUS_PER_TASK:-16}
echo "== $(date '+%F %T') building $C with $CARGO_BUILD_JOBS jobs"
time cargo build --release -p gam-mpd --examples 2>&1 | grep -vE '^\s+(Compiling|Downloaded|Downloading)' | tail -n 40
dest=$HOME/mpd-bin/$C12
rm -rf "$dest.tmp"
mkdir -p "$dest.tmp/src"
for f in target/release/examples/*; do
    base=${f##*/}
    [[ -f $f && -x $f && $base != *-* && $base != *.* ]] && cp "$f" "$dest.tmp/"
done
git archive "$C" | tar -x -C "$dest.tmp/src"
echo "$C" > "$dest.tmp/COMMIT"
rm -rf "$dest"
mv "$dest.tmp" "$dest"
touch "$dest/READY"
echo "== $(date '+%F %T') installed $(ls "$dest" | wc -l) entries in $dest"
# Keep the 12 newest builds; a running job keeps its deleted binary open.
ls -1dt "$HOME"/mpd-bin/*/ | tail -n +13 | xargs -r rm -rf
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
dep=()
if [ ! -f "$BIN/$C12/READY" ]; then
    bj=$(cat "$BIN/$C12.buildjob" 2> /dev/null || true)
    if [ -z "$bj" ] || ! squeue -h -j "$bj" -o %T 2> /dev/null | grep -qE 'PENDING|RUNNING|CONFIGURING|COMPLETING'; then
        bj=$(sbatch --parsable -J "mpd-build-$C12" -p compute -c 64 --mem=96G -t 01:00:00 \
            -o "$CL/_build/build-$C12-%j.log" "$CL/_build/build.sh" "$C")
        echo "$bj" > "$BIN/$C12.buildjob"
        echo "mats-run: building ${C12} in job $bj (log ~/mpd-data/cluster/_build/build-$C12-$bj.log)" >&2
    else
        echo "mats-run: waiting on build job $bj of ${C12}" >&2
    fi
    dep=(--dependency="afterok:$bj" --kill-on-invalid-dep=yes)
fi

stamp=$(date +%Y%m%d-%H%M%S)
job=$OUT/job-$stamp.sh
cmd=$(echo "$CMD_B64" | base64 -d)
cmd=${cmd//@@BIN@@/$BIN/$C12}
cat > "$job" <<JOB
#!/usr/bin/env bash
export RAYON_NUM_THREADS=\$SLURM_CPUS_PER_TASK OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export MPD_BIN=$BIN/$C12 GAM_SRC=$BIN/$C12/src MPD_DATA=\$HOME/mpd-data MATS_OUT=$OUT
export PATH=$BIN/$C12:\$HOME/mpd-venv/bin:\$HOME/.cargo/bin:\$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64:/usr/local/cuda-12.2/targets/x86_64-linux/lib:\${LD_LIBRARY_PATH:-}
cd $BIN/$C12/src
echo "== \$(date '+%F %T') job \$SLURM_JOB_ID on \$(hostname) commit $C cpus \$SLURM_CPUS_PER_TASK mem ${MEM}G gpus ${GPUS}"
echo "== $cmd"
start=\$(date +%s)
$cmd
rc=\$?
echo "== \$(date '+%F %T') exit \$rc after \$(( \$(date +%s) - start )) s"
exit \$rc
JOB
chmod +x "$job"

args=(--parsable -J "$NAME" -p compute -c "$CPUS" --mem="${MEM}G" -t "$MINUTES" -o "$OUT/slurm-%j.log" "${dep[@]}")
[ "$GPUS" != 0 ] && args+=(--gres="gpu:$GPUS")
[ -n "$QOS" ] && args+=(--qos="$QOS")
jid=$(sbatch "${args[@]}" "$job")
echo "$jid $stamp $C12 cpus=$CPUS mem=${MEM}G min=$MINUTES gpus=$GPUS" >> "$OUT/JOBS"
echo "$jid"
