#!/usr/bin/env bash
# Run MPD masked-pieces trainer jobs concurrently on one rented NVIDIA host (#2951).
#
#   [PORT=22] bench/gpu_run_2951.sh HOST RUN_DIR JOBS
#
# runs on your machine. HOST is an ssh destination (user@address, PORT its port) of a fresh Ubuntu
# box with an NVIDIA driver and the CUDA runtime (cuBLAS, NVRTC); RUN_DIR a local directory for the
# results; JOBS a file of jobs, one per line (`#` starts a comment):
#
#   NAME EXPORT_DIR [TRAINER ARGS...]
#
# where TRAINER ARGS are those of crates/gam-mpd/examples/mpd_pieces_masked_2951.rs after its
# EXPORT_DIR and OUT.json (OBSERVATIONS START TRAIN EVAL [CONTEXT] [GPU] [SETS|-] [corner|box]).
# Every local directory among them (also after `library:`) and every EXPORT_DIR goes up once, to
# HOST:~/mpd-run/inputs/<its name>, and the job's arguments name the copy. Job NAME writes
# RUN_DIR/NAME/out.json and everything beside it (progress, per-pass arrays, checkpoint).
#
# The host is untrusted and holds nothing of yours but the data you send: no agent or X11 is
# forwarded, it clones the public repository read-only over https, and every result comes back by
# rsync from here. The script
#   1. pushes the inputs, RUN_DIR (with any checkpoints) and itself to HOST:~/mpd-run/;
#   2. starts the remote side detached (it survives this ssh session): it installs the build tools
#      and the pinned Rust toolchain, clones the repository at REF (default main), builds, runs the
#      device tests and the 60-second benchmark (mpd_device_bench_2951, on the first job's export),
#      refuses the host when either fails or when a selection round runs below MIN_ROUNDS sequences
#      per second (default 0: any working device), and then runs every job at once, the host's
#      cores split evenly between them. With STOP=full-eval (the default) a training job is stopped
#      once its first full evaluation is written (a point over all EVAL eval sequences); STOP=none
#      lets every job run until it ends;
#   3. pulls ~/mpd-run/run/ into RUN_DIR every POLL seconds (default 120) until the remote side
#      ends, and once more at the end.
# After a spot interruption, rerun the same command on a new host: RUN_DIR holds the last pulled
# checkpoints, which go up in step 1, and every job resumes from its own.
#
# The remote side alone is `bench/gpu_run_2951.sh --remote` on the host, with ~/mpd-run/jobs.
set -Eeuo pipefail

REPO=https://github.com/SauersML/gam.git
REF=${REF:-main}
MIN_ROUNDS=${MIN_ROUNDS:-0}
POLL=${POLL:-120}
PORT=${PORT:-22}
STOP=${STOP:-full-eval}

remote_side() {
    local root=$HOME/mpd-run
    local run=$root/run
    mkdir -p "$run"
    state() { echo "$1" > "$run/STATE"; echo "== $(date '+%H:%M:%S') $1"; }
    trap 'state "failed (line $LINENO)"' ERR

    state setup
    if ! command -v nvidia-smi > /dev/null; then
        state "refused: no nvidia-smi (no NVIDIA driver)"
        exit 1
    fi
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$run/gpu.csv"
    nproc | tee "$run/cores"
    for dir in /usr/local/cuda/lib64 /usr/local/cuda/targets/x86_64-linux/lib; do
        [ -d "$dir" ] && export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:$dir
    done
    export DEBIAN_FRONTEND=noninteractive
    local sudo=sudo
    [ "$(id -u)" = 0 ] && sudo=
    if ! command -v cc > /dev/null || ! command -v git > /dev/null || ! command -v curl > /dev/null || ! command -v python3 > /dev/null; then
        $sudo apt-get update -q
        $sudo apt-get install -yq build-essential pkg-config git rsync curl python3
    fi
    if ! command -v cargo > /dev/null; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    fi
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
    if [ -d "$root/gam/.git" ]; then
        git -C "$root/gam" fetch -q origin "$REF"
        git -C "$root/gam" checkout -q --detach FETCH_HEAD
    else
        git clone -q "$REPO" "$root/gam"
        git -C "$root/gam" checkout -q --detach "origin/$REF" 2> /dev/null || git -C "$root/gam" checkout -q --detach "$REF"
    fi
    git -C "$root/gam" rev-parse HEAD | tee "$run/commit"
    cd "$root/gam"

    state building
    cargo build --release -q -p gam-mpd --example mpd_device_bench_2951 --example mpd_pieces_masked_2951 2>&1 | tail -n 20

    state testing
    cargo test --release -q -p gam-gpu --test tensor_kernels 2>&1 | tee "$run/tests.log"
    cargo test --release -q -p gam-mpd --lib -- device_ masked_device a_family_whose checkpoints_round_trip 2>&1 | tee -a "$run/tests.log"

    local names=() exports=() lines=()
    while read -r name export rest; do
        names+=("$name")
        exports+=("$export")
        lines+=("$rest")
    done < <(grep -v '^[[:space:]]*\(#\|$\)' "$root/jobs")
    [ ${#names[@]} -gt 0 ] || { state "refused: no jobs"; exit 1; }

    state benchmark
    ./target/release/examples/mpd_device_bench_2951 "${exports[0]}" 60 required "$run/bench.json" 2>&1 | tee "$run/bench.log"
    local rounds
    rounds=$(python3 -c "import json; print(json.load(open('$run/bench.json'))['best_selection_round_sequences_per_second'])")
    if python3 -c "import sys; sys.exit(0 if float('$rounds') >= float('$MIN_ROUNDS') else 1)"; then
        echo "== $rounds sequences per second per selection round (floor $MIN_ROUNDS)"
    else
        state "refused: $rounds sequences per second per selection round, below $MIN_ROUNDS"
        exit 1
    fi

    state training
    local threads=$(($(nproc) / ${#names[@]}))
    [ "$threads" -ge 1 ] || threads=1
    local pids=()
    for i in "${!names[@]}"; do
        local dir=$run/${names[$i]}
        mkdir -p "$dir"
        local args
        read -r -a args <<< "${lines[$i]}"
        RAYON_NUM_THREADS=$threads RUST_LOG=${RUST_LOG:-info} nohup ./target/release/examples/mpd_pieces_masked_2951 "${exports[$i]}" "$dir/out.json" "${args[@]}" >> "$dir/train.log" 2>&1 < /dev/null &
        pids+=($!)
        echo "== ${names[$i]} (pid ${pids[$i]}, $threads threads): ${lines[$i]}"
    done
    # A job's first full evaluation: a point over all its eval sequences (EVAL, its fourth argument).
    full_eval() {
        python3 - "$1" "$2" << 'EOF'
import json, sys
try:
    points = json.load(open(sys.argv[1])).get("points", [])
except (OSError, ValueError):
    sys.exit(1)
sys.exit(0 if any(p.get("eval_sequences") == int(sys.argv[2]) for p in points) else 1)
EOF
    }
    while true; do
        local running=0
        for i in "${!names[@]}"; do
            local pid=${pids[$i]} dir=$run/${names[$i]}
            [ "$pid" = 0 ] && continue
            local args
            read -r -a args <<< "${lines[$i]}"
            if ! kill -0 "$pid" 2> /dev/null; then
                wait "$pid" && echo "exited 0" > "$dir/STATE" || echo "exited $?" > "$dir/STATE"
                pids[i]=0
            elif [ "$STOP" = full-eval ] && [ "${args[2]}" != 0 ] && full_eval "$dir/out.json" "${args[3]}"; then
                kill "$pid"
                wait "$pid" || true
                echo "stopped after its first full evaluation" > "$dir/STATE"
                pids[i]=0
            else
                running=$((running + 1))
                echo "running" > "$dir/STATE"
            fi
        done
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader > "$run/gpu_now"
        [ "$running" -gt 0 ] || break
        sleep 30
    done
    state done
}

if [ "${1:-}" = "--remote" ]; then
    remote_side
    exit 0
fi

if [ $# -ne 3 ]; then
    awk 'NR > 1 && /^set -/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"
    exit 2
fi
HOST=$1
RUN=$2
JOBS=$3
mkdir -p "$RUN"
SSH=(ssh -p "$PORT" -o ForwardAgent=no -o ForwardX11=no -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
RSYNC_SSH="ssh -p $PORT -o ForwardAgent=no -o ForwardX11=no -o StrictHostKeyChecking=accept-new"
REMOTE_ROOT="$("${SSH[@]}" "$HOST" 'echo $HOME')/mpd-run"
"${SSH[@]}" "$HOST" "command -v rsync > /dev/null || { apt-get update -q && apt-get install -yq rsync; } > /dev/null"

# Every job's local directories go up once, and its line names the copies.
declare -A UP=()
remote_jobs=
while read -r line; do
    case "$line" in '' | '#'*) continue ;; esac
    out=
    for arg in $line; do
        path=${arg#library:}
        if [ -d "$path" ]; then
            copy="$REMOTE_ROOT/inputs/$(basename "$path")"
            UP[$path]=$copy
            [ "$arg" != "$path" ] && copy="library:$copy"
            out+=" $copy"
        else
            out+=" $arg"
        fi
    done
    remote_jobs+="${out# }"$'\n'
done < "$JOBS"

"${SSH[@]}" "$HOST" "mkdir -p $REMOTE_ROOT/inputs $REMOTE_ROOT/run"
for path in "${!UP[@]}"; do
    echo "== pushing $path"
    rsync -azL -e "$RSYNC_SSH" "$path/" "$HOST:${UP[$path]}/"
done
rsync -az -e "$RSYNC_SSH" "$RUN/" "$HOST:$REMOTE_ROOT/run/"
rsync -az -e "$RSYNC_SSH" "$0" "$HOST:$REMOTE_ROOT/gpu_run_2951.sh"
printf '%s' "$remote_jobs" | "${SSH[@]}" "$HOST" "cat > $REMOTE_ROOT/jobs"
"${SSH[@]}" "$HOST" "cd $REMOTE_ROOT && rm -f run/STATE && REF=$REF MIN_ROUNDS=$MIN_ROUNDS STOP=$STOP nohup setsid bash gpu_run_2951.sh --remote > run/remote.log 2>&1 < /dev/null &"
echo "== started on $HOST; pulling $REMOTE_ROOT/run into $RUN every ${POLL}s"

failures=0
while true; do
    sleep "$POLL"
    if rsync -az -e "$RSYNC_SSH" "$HOST:$REMOTE_ROOT/run/" "$RUN/"; then
        failures=0
    else
        failures=$((failures + 1))
        if [ "$failures" -ge 3 ]; then
            echo "== $HOST unreachable three times; $RUN holds the last checkpoints: rerun this command on a new host to resume"
            exit 1
        fi
        continue
    fi
    state=$(cat "$RUN/STATE" 2> /dev/null || echo starting)
    echo "== $(date '+%H:%M') $state; $(tail -n 1 "$RUN/remote.log" 2> /dev/null)"
    case "$state" in
        done) exit 0 ;;
        refused* | failed*) exit 1 ;;
    esac
done
