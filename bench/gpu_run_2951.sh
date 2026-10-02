#!/usr/bin/env bash
# One command to run the MPD masked-pieces trainer on a rented NVIDIA host (#2951).
#
#   bench/gpu_run_2951.sh HOST EXPORT_DIR RUN_DIR [TRAINER ARGS...]
#
# runs on your machine. HOST is an ssh destination (user@address) of a fresh Ubuntu box with an
# NVIDIA driver and the CUDA runtime (cuBLAS, NVRTC); EXPORT_DIR a language-model export
# (gam_mpd::import); RUN_DIR a local directory for the results; TRAINER ARGS the arguments of
# crates/gam-mpd/examples/mpd_pieces_masked_2951.rs after its EXPORT_DIR and OUT.json
# (OBSERVATIONS {wsvd|wsvd2|library:DIR} TRAIN EVAL [CONTEXT] [GPU] [SETS]); a local directory
# among them (also after `library:`) goes up with the export and is rewritten to its copy.
#
# The host is untrusted and holds nothing of yours but the data you send: it clones the public
# repository read-only over https, and every result comes back by rsync from here. The script
#   1. pushes the export, the trainer's input directories, RUN_DIR (with any checkpoint) and
#      itself to HOST:~/mpd-run/;
#   2. starts the remote side detached (it survives this ssh session): it installs the build
#      tools and the pinned Rust toolchain, clones the repository at REF (default main), builds,
#      runs the device tests and the 60-second benchmark (mpd_device_bench_2951), refuses the
#      host when either fails or when a selection round runs below MIN_ROUNDS sequences per
#      second (default 0: any working device), and then runs the trainer with its output and
#      checkpoint in ~/mpd-run/run/;
#   3. pulls ~/mpd-run/run/ into RUN_DIR every POLL seconds (default 300) until the remote side
#      ends, and once more at the end.
# After a spot interruption, rerun the same command on a new host: RUN_DIR holds the last pulled
# checkpoint, which goes up in step 1, and the trainer resumes from it.
#
# The remote side alone is `bench/gpu_run_2951.sh --remote [TRAINER ARGS...]` on the host.
set -Eeuo pipefail

REPO=https://github.com/SauersML/gam.git
REF=${REF:-main}
MIN_ROUNDS=${MIN_ROUNDS:-0}
POLL=${POLL:-300}

remote_side() {
    local root=$HOME/mpd-run
    local run=$root/run
    mkdir -p "$run"
    state() { echo "$1" > "$run/STATE"; echo "== $1"; }
    trap 'state "failed (line $LINENO)"' ERR

    state setup
    if ! command -v nvidia-smi > /dev/null; then
        state "refused: no nvidia-smi (no NVIDIA driver)"
        exit 1
    fi
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv | tee "$run/gpu.csv"
    if ! ldconfig -p | grep -q libnvrtc; then
        export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/usr/local/cuda/lib64
    fi
    export DEBIAN_FRONTEND=noninteractive
    local sudo=sudo
    [ "$(id -u)" = 0 ] && sudo=
    if ! command -v cc > /dev/null || ! command -v git > /dev/null || ! command -v rsync > /dev/null; then
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
    cargo build --release -q -p gam-mpd --example mpd_device_bench_2951 --example mpd_pieces_masked_2951

    state testing
    cargo test --release -q -p gam-gpu --test tensor_kernels 2>&1 | tee "$run/tests.log"
    cargo test --release -q -p gam-mpd --lib -- device_ masked_device a_family_whose checkpoints_round_trip 2>&1 | tee -a "$run/tests.log"

    state benchmark
    ./target/release/examples/mpd_device_bench_2951 "$root/export" 60 required "$run/bench.json" 2>&1 | tee "$run/bench.log"
    local rounds
    rounds=$(python3 -c "import json; print(json.load(open('$run/bench.json'))['best_selection_round_sequences_per_second'])")
    if python3 -c "import sys; sys.exit(0 if float('$rounds') >= float('$MIN_ROUNDS') else 1)"; then
        echo "== $rounds sequences per second per selection round (floor $MIN_ROUNDS)"
    else
        state "refused: $rounds sequences per second per selection round, below $MIN_ROUNDS"
        exit 1
    fi

    state training
    ./target/release/examples/mpd_pieces_masked_2951 "$root/export" "$run/out.json" "$@" 2>&1 | tee -a "$run/train.log"
    state done
}

if [ "${1:-}" = "--remote" ]; then
    shift
    remote_side "$@"
    exit 0
fi

if [ $# -lt 3 ]; then
    awk 'NR > 1 && /^set -/ { exit } NR > 1 { sub(/^# ?/, ""); print }' "$0"
    exit 2
fi
HOST=$1
EXPORT=$2
RUN=$3
shift 3
mkdir -p "$RUN"
REMOTE_ROOT="$(ssh "$HOST" 'echo $HOME')/mpd-run"

# The trainer's local input directories go up beside the export, and its arguments name the copies.
ARGS=()
INPUTS=()
for arg in "$@"; do
    path=${arg#library:}
    if [ -d "$path" ]; then
        INPUTS+=("$path")
        copy="$REMOTE_ROOT/inputs/$(basename "$path")"
        if [ "$arg" != "$path" ]; then ARGS+=("library:$copy"); else ARGS+=("$copy"); fi
    else
        ARGS+=("$arg")
    fi
done

ssh "$HOST" "mkdir -p $REMOTE_ROOT/inputs $REMOTE_ROOT/run"
rsync -azL "$EXPORT/" "$HOST:$REMOTE_ROOT/export/"
for path in ${INPUTS[@]+"${INPUTS[@]}"}; do
    rsync -azL "$path/" "$HOST:$REMOTE_ROOT/inputs/$(basename "$path")/"
done
rsync -az "$RUN/" "$HOST:$REMOTE_ROOT/run/"
rsync -az "$0" "$HOST:$REMOTE_ROOT/gpu_run_2951.sh"
quoted=
[ ${#ARGS[@]} -gt 0 ] && quoted=$(printf ' %q' "${ARGS[@]}")
ssh "$HOST" "cd $REMOTE_ROOT && rm -f run/STATE && REF=$REF MIN_ROUNDS=$MIN_ROUNDS nohup setsid bash gpu_run_2951.sh --remote$quoted > run/remote.log 2>&1 < /dev/null &"
echo "== started on $HOST; pulling $REMOTE_ROOT/run into $RUN every ${POLL}s"

failures=0
while true; do
    sleep "$POLL"
    if rsync -az "$HOST:$REMOTE_ROOT/run/" "$RUN/"; then
        failures=0
    else
        failures=$((failures + 1))
        if [ "$failures" -ge 3 ]; then
            echo "== $HOST unreachable three times; $RUN holds the last checkpoint: rerun this command on a new host to resume"
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
