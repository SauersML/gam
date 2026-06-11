#!/usr/bin/env bash
set -euo pipefail

repo=/mnt/work/gam
log=/mnt/work/exp/perf_log.txt

export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/mnt/work/target

cd "$repo"
git pull
mkdir -p /mnt/work/exp

{
    printf 'PERF_RUN git_sha=%s\n' "$(git rev-parse HEAD)"
    nice -n 10 cargo build --release --example sae_perf_harness 2>&1 | tail -1
    for shape in tiny color qwen; do
        "$CARGO_TARGET_DIR/release/examples/sae_perf_harness" "$shape"
    done
} | tee -a "$log"
