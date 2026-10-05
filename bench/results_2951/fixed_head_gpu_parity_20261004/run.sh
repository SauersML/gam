#!/usr/bin/env bash
set -Eeuo pipefail
binary="$HOME/mpd-bin-targeted/ff80e73f7606/mpd_fixed_head_parity_2951"
export_dir="$HOME/mpd-data/engine/vpd4l_frontier32"
out="$HOME/mpd-data/cluster/codex-fixed-head-parity-ff80/measurement"
expected=2cc30ab261eb6d924e5b719b7aa2f19e6f31d80d7352cd6c2d9c7a1d97249e23
[[ $(sha256sum "$binary" | cut -d' ' -f1) = "$expected" ]]
[[ $(cat "$(dirname "$binary")/COMMIT") = ff80e73f760698e0f46bed9057a4595b5b014949 ]]
[[ ! -e "$out" ]]
mkdir -p "$out"
cp "$(dirname "$binary")/mpd_fixed_head_parity_2951.build-info" "$out/build-info.txt"
sha256sum "$binary" "$export_dir/export.json" "${BASH_SOURCE[0]}" > "$out/SHA256.txt"
nvidia-smi > "$out/gpu.txt"
start=$SECONDS
"$binary" cuda "$export_dir" > "$out/report.json"
printf 'elapsed_seconds=%s\n' "$((SECONDS-start))" > "$out/wall.txt"
