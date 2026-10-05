#!/usr/bin/env bash
# The fitting step's time and device memory on one GPU (#2951), run on a rented card by rp-run:
#
#   RP_EXAMPLES="mpd_library_step_bench_2951 mpd_interchange_bench_2951 mpd_qwen3_step_bench_2951 tensor_f32_speed" \
#   bench/runpod/rp-run bench-a5000 "RTX A5000" 0.5 -- bash /Users/user/gam/bench/runpod/gpu_bench.sh \
#       /Users/user/mpd-data/engine/vpd4l_clean4096 \
#       /Users/user/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
#       /Users/user/mpd-data/qwen3_fineweb/heldout/windows_T128.u32 /Users/user/mpd-data/runpod/bench-a5000
#
# Parts, each to OUT/PART.{out,err} and one line of OUT/parts.tsv (part, exit status, seconds, peak
# device MiB): the CUDA tensor operations in float64 and f32 (tensor_f32_speed); the library step on
# vpd4l at 8 and 32 sequences of 512 tokens (host and device posterior); the interchange step on
# vpd4l at 8 sequences; Qwen3-0.6B's forward and reverse passes at 8, 16 and 32 sequences of 128
# tokens in f32 and TF32 products; the interchange step on Qwen3-0.6B at 4 sequences; then the
# CUDA parity tests of gam-gpu (tensor_f32, tensor_posterior). The peak is the largest
# `nvidia-smi` reading of the device's used memory, taken every 100 ms while the part runs.
set -uo pipefail
[ $# -eq 4 ] || { sed -n '2,17p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
VPD=$1 QWEN=$2 WINDOWS=$3 OUT=$4
mkdir -p "$OUT"
nvidia-smi > "$OUT/nvidia-smi.txt"
printf 'part\tstatus\tseconds\tpeak_mib\n' > "$OUT/parts.tsv"

part() {
    local name=$1 smi rc start
    shift
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > "$OUT/$name.mem" &
    smi=$!
    start=$(date +%s.%N)
    "$@" > "$OUT/$name.out" 2> "$OUT/$name.err"
    rc=$?
    kill "$smi"
    wait "$smi" 2> /dev/null
    printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$(awk -v a="$start" -v b="$(date +%s.%N)" 'BEGIN { printf "%.1f", b - a }')" "$(sort -n "$OUT/$name.mem" | tail -n 1)" | tee -a "$OUT/parts.tsv"
}

part tensor_f32_speed "$MPD_BIN/tensor_f32_speed"
for s in 8 32; do
    part "vpd4l_step_s$s" "$MPD_BIN/mpd_library_step_bench_2951" "$VPD" "$s" 512 3 "$OUT/vpd4l_step_s$s"
done
part vpd4l_interchange_s8 "$MPD_BIN/mpd_interchange_bench_2951" "$VPD" 8 512 3 "$OUT/vpd4l_interchange_s8" device
for a in f32 tf32; do
    part "qwen3_passes_$a" "$MPD_BIN/mpd_qwen3_step_bench_2951" "$QWEN" "$WINDOWS" "$OUT/qwen3_passes_$a" "arithmetic=$a"
done
part qwen3_interchange_s4 "$MPD_BIN/mpd_interchange_bench_2951" "$QWEN" 4 128 3 "$OUT/qwen3_interchange_s4" device "$WINDOWS"

# The parity tests compile here, with the toolchain the source pins.
if ! command -v cargo > /dev/null; then
    channel=$(sed -nE 's/^channel = "([^"]+)"/\1/p' rust-toolchain.toml)
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal --default-toolchain "$channel" > /dev/null
    source "$HOME/.cargo/env"
fi
part gpu_build cargo test -p gam-gpu --test tensor_f32 --test tensor_posterior --no-run
part gpu_tests cargo test -p gam-gpu --test tensor_f32 --test tensor_posterior -- --test-threads=1
