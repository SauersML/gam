#!/usr/bin/env bash
# The fitting step's time and device memory on one GPU (#2951), run on a rented card by rp-run:
#
#   RP_EXAMPLES="mpd_library_step_bench_2951 tensor_f32_speed attention_speed" \
#   RP_TESTS="gam-gpu:tensor_f32 gam-gpu:tensor_posterior gam-gpu:tensor_decoder" \
#   bench/runpod/rp-run bench-4090 "RTX 4090" 0.5 -- bash /Users/user/gam/bench/runpod/gpu_bench.sh \
#       /Users/user/mpd-data/engine/vpd4l_clean4096 \
#       /Users/user/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
#       /Users/user/mpd-data/qwen3_fineweb/heldout_f284/windows_T512.u32 /Users/user/mpd-data/runpod/bench-4090
#
# Parts, each to OUT/PART.{out,err} and one line of OUT/parts.tsv (part, exit status, seconds, peak
# device MiB): the CUDA tensor operations in float64 and f32 (tensor_f32_speed) and fused causal
# attention (attention_speed); the library step (posterior on the device) on vpd4l at 8 and 32
# sequences of 512 tokens, with the program engine in f32 products and with the decoder engine in
# bfloat16 products; the step on Qwen3-0.6B at 8 sequences of 512 tokens with the decoder engine in
# bfloat16; then the CUDA parity tests of gam-gpu (tensor_f32, tensor_posterior, tensor_decoder),
# built off the GPU pod (RP_TESTS) and run from $MPD_BIN.
# The peak is the largest `nvidia-smi` reading of the device's used memory, every 100 ms while
# the part runs.
set -uo pipefail
[ $# -eq 4 ] || { sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
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
part attention_speed "$MPD_BIN/attention_speed"
for s in 8 32; do
    part "vpd4l_program_f32_s$s" "$MPD_BIN/mpd_library_step_bench_2951" "$VPD" "$s" 512 3 "$OUT/vpd4l_program_f32_s$s" engine=program products=f32
    part "vpd4l_decoder_bf16_s$s" "$MPD_BIN/mpd_library_step_bench_2951" "$VPD" "$s" 512 3 "$OUT/vpd4l_decoder_bf16_s$s" engine=decoder products=bf16
done
part qwen3_decoder_bf16_s8 "$MPD_BIN/mpd_library_step_bench_2951" "$QWEN" 8 512 3 "$OUT/qwen3_decoder_bf16_s8" "$WINDOWS" engine=decoder products=bf16

# The parity tests, built on the runner's CPU build pod (RP_TESTS) like the examples: a GPU pod
# does not compile.
for t in tensor_f32 tensor_posterior tensor_decoder; do
    part "test_$t" "$MPD_BIN/test-$t" --test-threads=1
done
