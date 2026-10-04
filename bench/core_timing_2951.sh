#!/usr/bin/env bash
# The core path's two speed targets (#2951), each stage's wall time against the 32-CPU baseline.
#
#   bench/core_timing_2951.sh submit TAG     queue both targets on one L40 (MATS_POOL, debug QOS)
#   bench/core_timing_2951.sh local TAG      run layer 0 here (the Apple GPU's f32 path, device=any)
#   bench/core_timing_2951.sh report TAG     each stage's seconds of both targets, and the speedup
#
# Target 1 (`l0`): mpd_e2e_2951 at layer 0, the measured baseline's settings (rounds=0 blocks=0
# n=1e5 sequences=8; 3821 s on 32 CPUs, job 15527). Target 2 (`all`): every layer fitted
# (n=1e5 sequences=8, the driver's default rounds and blocks) and scored. Each run writes
# OUT/e2e.json, whose "seconds" map holds every stage the driver times; a stage added to that map
# shows up here.
set -Eeuo pipefail
cmd=${1:?usage: core_timing_2951.sh submit|local|report TAG}
tag=${2:?a tag}
gam=$(cd "$(dirname "$0")/.." && pwd)
data=/Users/user/mpd-data
args=("$data/engine/vpd4l_e2e_train" "$data/engine/vpd4l_frontier32" "$data/pieces/vpd4l_library" "$data/pieces/vpd4l_sets")
l0=(rounds=0 blocks=0 n=1e5 sequences=8)
all=(n=1e5 sequences=8)
out() { echo "$data/cluster/core-timing-$1-$tag"; }
case $cmd in
    submit)
        cd "$gam"
        RUST_LOG=info MATS_POOL=1 MATS_GPUS=1 MATS_QOS=debug bench/mats/mats-run "core-timing-l0-$tag" 32 auto 2 -- \
            "$gam/target/release/examples/mpd_e2e_2951" blocks.0. "$(out l0)" "${args[@]}" "${l0[@]}"
        RUST_LOG=info MATS_POOL=1 MATS_GPUS=1 bench/mats/mats-run "core-timing-all-$tag" 32 auto 12 -- \
            "$gam/target/release/examples/mpd_e2e_2951" all "$(out all)" "${args[@]}" "${all[@]}" ;;
    local)
        cd "$gam"
        bin=$(./build.sh example mpd_e2e_2951 release | tail -1)
        dir="$data/core-timing/l0-$tag"
        mkdir -p "$dir"
        RUST_LOG=info mem-lease 24 "$bin" blocks.0. "$dir" "${args[@]}" "${l0[@]}" device=any 2> "$dir/log.txt"
        "$0" report "$tag" ;;
    report)
        for target in l0 all; do
            for dir in "$(out "$target")" "$data/core-timing/$target-$tag"; do
                [ -f "$dir/e2e.json" ] || continue
                python3 - "$dir/e2e.json" "$target" <<'EOF'
import json, sys
seconds = json.load(open(sys.argv[1])).get("seconds", {})
baseline = {"l0": 3821.0}.get(sys.argv[2])
print(f"{sys.argv[2]} ({sys.argv[1]})")
for stage, s in sorted(seconds.items(), key=lambda kv: -kv[1]):
    print(f"  {stage:20s} {s:10.1f} s")
if baseline and "total" in seconds:
    print(f"  {'speedup':20s} {baseline / seconds['total']:10.1f} x against {baseline:.0f} s (target 100 x)")
EOF
            done
        done ;;
    *) echo "usage: core_timing_2951.sh submit|local|report TAG" >&2; exit 2 ;;
esac
