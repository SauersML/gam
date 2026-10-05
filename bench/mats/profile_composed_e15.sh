#!/usr/bin/env bash
# Per-stage wall times of the composed-causal interior e15 workload on one L40 (#2951).
#
#   bench/mats/profile_composed_e15.sh submit [--nsys]   submit at the pushed HEAD; prints NAME
#   bench/mats/profile_composed_e15.sh report NAME       pull NAME and print its stage table
#
# The workload is bench/results_2951/composed_causal_interior_20261004/e15: expression 15, shared
# and untied arms, 8 sequences x 64 positions, 6 cases, 256 Adam updates, 2-sequence heldout panel.
# The driver runs with --profile (each named stage synchronizes its device at both ends and
# PROFILE.json holds the inclusive totals). --nsys also records an Nsight Systems trace with the
# NVTX stages (NAME/e15.nsys-rep) and its summaries (NAME/nsys_*.txt).
set -Eeuo pipefail
here=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
D=/Users/user/mpd-data
BIN=/Users/user/gam/target/release/examples/mpd_composed_causal_search_2951
ARGS=("$D/engine/vpd4l_e2e_train" "$D/codex/composed-causal-interior-20261004/expression15.json")
HELDOUT=$D/codex/copy-rule-validation-20261004/export

case ${1:-} in
submit)
    nsys=0
    [ "${2:-}" = --nsys ] && nsys=1
    name=e15-profile-$(git -C "$here" rev-parse --short=10 HEAD)-$(date +%H%M%S)
    out=$D/cluster/$name
    run=("$BIN" "${ARGS[@]}" "$out/run" cuda "$HELDOUT" --profile)
    if [ $nsys = 1 ]; then
        # nsys writes the trace, then exports the NVTX, kernel and API summaries next to it.
        run=(bash -c "nsys profile -t cuda,nvtx,cublas --force-overwrite true -o $out/e15 $(printf '%q ' "${run[@]}") \
            && for r in nvtx_sum cuda_gpu_kern_sum cuda_api_sum cuda_gpu_mem_time_sum; do nsys stats -q --report \$r $out/e15.nsys-rep > $out/nsys_\$r.txt; done")
    fi
    MATS_GPUS=1 MATS_QOS=debug MATS_NO_UPLOAD=1 MATS_BUILD=1 "$here/mats-run" "$name" 8 48 2 -- "${run[@]}" >&2
    echo "$name" ;;
report)
    name=${2:?NAME}
    "$here/mats-pull" "$name" >&2
    python3 - "$D/cluster/$name/run" <<'PY'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
report = json.loads((run / "REPORT.json").read_text())
print(f"{'stage':<34}{'seconds':>10}")
for k, v in report["stage_seconds"].items():
    print(f"{k:<34}{v:>10.2f}")
for c in report["candidates"]:
    s = c.get("stage_seconds")
    if not s:
        continue
    fit = json.loads((run / c["id"] / "FIT.json").read_text())
    steps = fit["settings"]["iterations"]
    print(c["id"])
    for k in ("graft", "canonical_before_fit", "fit", "canonical_after_fit", "structural_cost"):
        print(f"  {k:<32}{s[k]:>10.2f}")
    print(f"  {'fit per Adam step':<32}{s['fit'] / steps:>10.4f}")
print(f"{'total':<34}{report['seconds']:>10.2f}")
profile = run / "PROFILE.json"
if profile.exists():
    print(f"\n{'inclusive stage (--profile)':<26}{'count':>8}{'mean s':>12}{'total s':>10}")
    for t in json.loads(profile.read_text())["inclusive_stage_seconds"]:
        print(f"{t['name']:<26}{t['count']:>8}{t['seconds'] / max(t['count'], 1):>12.5f}{t['seconds']:>10.2f}")
PY
    ls "$D/cluster/$name"/nsys_*.txt 2> /dev/null || true ;;
*) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac
