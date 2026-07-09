#!/bin/bash
# #1026 EV-vs-K FRONTIER matrix — fan out one independent job per (arm, K, seed).
#
# Publishes the frontier the close bar asks for: not just the single K=32768
# point (already MET + committed), but EV vs dictionary size K across the sweep,
# for BOTH the traditional-SAE bar (external_topk) and our hybrid contestant, at
# a MATCHED per-token active-scalar budget, plus two fixed reference lines:
#   - pca_bar        : the linear-optimum yardstick (run ONCE, K-independent).
#   - w32k_baseline  : the published W32K k=100 flat-SAE number, EV=0.523
#                      (a fixed EXTERNAL scalar, not a job — appended as a row).
#
# Grid: arm ∈ {external_topk, hybrid} × K ∈ {1024,4096,8192,16384,32768}
#       × seed ∈ {0,1}  = 20 GPU jobs, + 1 pca_bar job = 21 sbatch jobs.
# Each job appends ONE row to $OUT via run_frontier_arm.sbatch, so a failed
# point never blocks the rest and re-running a single point is a one-liner.
#
#   bash experiments/1026_close/frontier_matrix.sh          # submit the matrix
#   DRY_RUN=1 bash experiments/1026_close/frontier_matrix.sh # print, do not submit
#
# ---------------------------------------------------------------------------- #
# AUDIT — K-dependent assumptions in driver_1026_arms.py, and how this matrix
# neutralizes them so EV is comparable ACROSS K:
#
#   * external_topk : W_enc is (K, p) and top_k=32 actives are selected from K.
#     For every K in the grid K >= top_k, so topk() is always well-posed; cost
#     scales ~linearly in K (small-K jobs are dominated by fixed load/PCA time).
#     No break below 32k.
#
#   * hybrid : the flat tier uses the full K dictionary, but the CURVED tier's
#     size (curved_atoms) is an INDEPENDENT knob that the driver does NOT tie to
#     K. Left fixed (the 256/512 used at the 32k bar), the curved tier would add
#     a CONSTANT extra capacity that is a large RELATIVE boost at small K — the
#     hybrid would look unfairly strong at K=1024. Fix: hold the curved:flat
#     dictionary-capacity RATIO constant by scaling
#         curved_atoms = max(64, K/64)
#     (floored at 64 so the manifold tier stays expressive at the smallest K).
#     At K=32768 this is 512, so the 32k frontier point reproduces the committed
#     r4-a512 winner exactly. curved_k=2, d_atom=2 => the matched-budget split
#     k_lin = top_k - curved_k*(1+d_atom) = 32 - 6 = 26 is K-independent, so the
#     per-token active-scalar budget stays matched at every K.
#
#   * pca_bar : ranks are fixed [16,32,64,128,512] (all <= p=2048) and carry no K
#     dependence — it is a single K-independent linear reference, submitted once.
# ---------------------------------------------------------------------------- #
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SBATCH_JOB="$HERE/run_frontier_arm.sbatch"
R=${R:-/projects/standard/hsiehph/sauer354}
OUT=${OUT:-$R/1026_frontier/results_1026.jsonl}
KS=(1024 4096 8192 16384 32768)
SEEDS=(0 1)
DRY_RUN=${DRY_RUN:-0}

curved_atoms_for_k() { local k=$1; local c=$(( k/64 )); (( c < 64 )) && c=64; echo "$c"; }

submit() {  # submit <ARM> <K> <SEED> <CURVED_ATOMS>
  local arm=$1 k=$2 seed=$3 catoms=$4
  local cmd=(sbatch --export=ALL,ARM="$arm",K="$k",SEED="$seed",CURVED_ATOMS="$catoms",OUT="$OUT" "$SBATCH_JOB")
  echo "  $arm  K=$k seed=$seed curved_atoms=$catoms"
  if [ "$DRY_RUN" = "1" ]; then echo "    (dry-run) ${cmd[*]}"; else "${cmd[@]}"; fi
}

[ "$DRY_RUN" = "1" ] || mkdir -p "$(dirname "$OUT")"

# --- Fixed external reference row: W32K k=100 flat SAE, EV=0.523 -------------- #
# This is a PUBLISHED scalar (a width-32768 flat/TopK SAE with k=100 actives on
# the same activation family), not something we recompute. It enters the frontier
# as one results row with arm='w32k_baseline' so plot_frontier.py can draw it as
# a horizontal reference line without any special-casing.
BASELINE_ROW='{"issue": 1026, "arm": "w32k_baseline", "tag": "external-fixed", "N": 120000, "p": 2048, "K": 32768, "top_k": 100, "d_atom": 2, "curved_atoms": 0, "curved_k": 0, "steps": 0, "seed": 0, "ev": 0.523, "wall_s": 0.0, "note": "published W32K flat SAE k=100 baseline; fixed external number, not a job"}'
echo "== W32K baseline row (fixed external EV=0.523) =>  $OUT"
if [ "$DRY_RUN" = "1" ]; then
  echo "  (dry-run) would append: $BASELINE_ROW"
else
  printf '%s\n' "$BASELINE_ROW" >> "$OUT"
fi

# --- Submission order: yardstick first, then cheap bar, then slow hybrid ------ #
# Order is chosen so the fast reference lines land first (early partial figure)
# and the expensive hybrid points queue last. Failures are independent either way.
echo "== [1/3] pca_bar (K-independent linear yardstick, run once) =="
submit pca_bar 32768 0 0

echo "== [2/3] external_topk bar  (10 jobs: 5 K x 2 seeds, ~fast) =="
for k in "${KS[@]}"; do for s in "${SEEDS[@]}"; do
  submit external_topk "$k" "$s" "$(curved_atoms_for_k "$k")"
done; done

echo "== [3/3] hybrid contestant  (10 jobs: 5 K x 2 seeds, ~slow) =="
for k in "${KS[@]}"; do for s in "${SEEDS[@]}"; do
  submit hybrid "$k" "$s" "$(curved_atoms_for_k "$k")"
done; done

cat <<EOF

== EXPECTED WALL-TIME per job (extrapolated from the r4 committed rows) ==
  pca_bar        : ~6 s        (CPU, one job)
  external_topk  : ~15-55 s    (GPU; r4 bar bs2048 8000 steps = 49-51 s at K=32768;
                                cost ~linear in K, so K<=8192 is dominated by the
                                ~10-15 s fixed data-load/PCA floor)
  hybrid         : ~500-700 s  (GPU; flat tier ~50 s + curved torch-manifold tier
                                ~450-640 s. r4 a512 = 671 s at K=32768; the curved
                                tier cost is set by curved_steps/curved_atoms and
                                barely moves with K, so all K land in this band)

  Total wall if fully serial ~= 1 pca + 10 ext (~10 min) + 10 hybrid (~110 min)
  ~= 2 h; with the a100-4 queue running points in parallel it is far less.
  After all rows land in $OUT, render with:
    python experiments/1026_close/plot_frontier.py --results $OUT --outdir <dir>
EOF
