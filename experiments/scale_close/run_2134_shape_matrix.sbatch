#!/bin/bash -l
# #2134 close — the previously-REFUSED overcomplete (N, K, P) region (K > P) now
# RUNS through the front-door streaming lane with peak-RSS discipline. Drives the
# compiled `scale_k` example (crates/gam-sae/examples/scale_k.rs) — the public
# front-door + block-sparse streaming path whose peak-RSS assertions
# (measured peak < dense N x K, and < one N x P + payload) are ENFORCED INSIDE the
# example: it exits non-zero if peak RSS ever implies a dense (N, K) or a second
# full (N, P) allocation. This sbatch sweeps the shape matrix and requires EVERY
# cell to pass both the RSS invariants and an EV-sanity floor.
#
# Shape matrix (K > P in every cell): N in {2e5, 5e5}, K in {2048, 8192},
# P in {256, 1024}. The (N, P) input is synthetic (planted low-rank + noise), one
# .npy per (N, P) reused across K.
#
# Fresh-clone runner (like experiments/tiered_gpu_a100): clones origin/main into
# scratch and refuses to run a stale checkout, so it validates HEAD exactly.
#
# Submit (coordinator, from an MSI login node):
#   sbatch experiments/scale_close/run_2134_shape_matrix.sbatch
# Narrow the sweep with env, e.g. quick smoke:
#   NS="200000" KS="2048" PS="256" EPOCHS=1 sbatch run_2134_shape_matrix.sbatch
#
#SBATCH -A hsiehph
#SBATCH --job-name=sae_2134_shape
#SBATCH --partition=msismall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96g
#SBATCH --time=12:00:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/scale_close/sae_2134_shape_%j.out
set -uo pipefail

PROJ=${PROJ:-/projects/standard/hsiehph/sauer354}
OUT=$PROJ/scale_close
mkdir -p "$OUT"
RUN_ROOT="${OUT}/build_2134_${SLURM_JOB_ID:-manual}"
REPO="${RUN_ROOT}/gam"

# Shape matrix (override for a narrower sweep). EV-sanity floor: the block-sparse
# fit must explain at least this fraction of the planted structure.
NS=${NS:-"200000 500000"}
KS=${KS:-"2048 8192"}
PS=${PS:-"256 1024"}
DIRS=${DIRS:-32}            # planted directions in the synthetic input
NOISE=${NOISE:-0.1}
EPOCHS=${EPOCHS:-2}
MINIBATCH=${MINIBATCH:-4096}
BLOCK_SIZE=${BLOCK_SIZE:-4}
EV_FLOOR=${EV_FLOOR:-0.10}

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export RAYON_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export LD_LIBRARY_PATH="/common/software/install/manual/openblas/0.3.27_gcc11.3.0/lib:${LD_LIBRARY_PATH:-}"
# gam_env.sh exports a booby-trapped RUSTFLAGS (OpenBLAS saboteur); never source it
# and clear RUSTFLAGS so a login-profile export cannot leak into the build.
unset RUSTFLAGS
export CARGO_TARGET_DIR=/scratch.global/sauer354/gam-target-2134
LOCK="/scratch.global/sauer354/gam-target-2134.lock"

rm -rf "${RUN_ROOT}"
mkdir -p "${RUN_ROOT}"
git clone --single-branch --branch main https://github.com/SauersML/gam.git "${REPO}"
cd "${REPO}" || exit 2
HEAD="$(git rev-parse HEAD)"
REMOTE="$(git ls-remote origin refs/heads/main | awk '{print $1}')"
echo "=== #2134 shape-matrix  clone HEAD=${HEAD}  origin/main=${REMOTE}  node=$(hostname)  $(date) ==="
[ "${HEAD}" = "${REMOTE}" ] || { echo "=== FATAL: fresh clone is not at origin/main ==="; exit 99; }

CARGO="${PROJ}/.cargo/bin/cargo"
[ -x "${CARGO}" ] || CARGO=cargo
GEN="${REPO}/experiments/scale_close/gen_2134_planted_input.py"
test -f "${GEN}" || { echo "FATAL: generator missing: ${GEN}"; exit 2; }

# numpy-only venv for the synthetic input generator (no gamfit needed — the fit is
# the compiled Rust example).
GVENV="${RUN_ROOT}/genvenv"
python3 -m venv "${GVENV}"
"${GVENV}/bin/python" -m pip install --upgrade pip >/dev/null
"${GVENV}/bin/python" -m pip install --no-cache-dir numpy >/dev/null

mkdir -p "$(dirname "${LOCK}")"
exec 9>"${LOCK}"
echo "[2134] acquiring target lock ${LOCK}"; flock 9

echo "[2134] building example scale_k (release) at ${HEAD}"
"${CARGO}" build --release -p gam-sae --example scale_k 2>&1 \
  || { echo "=== FATAL: build failed ==="; exit 3; }

SUMMARY="${OUT}/summary_2134.jsonl"
: > "${SUMMARY}"
overall=0
for N in ${NS}; do
  for P in ${PS}; do
    INPUT="${RUN_ROOT}/planted_N${N}_P${P}.npy"
    echo "[2134] generating input N=${N} P=${P} -> ${INPUT}"
    "${GVENV}/bin/python" "${GEN}" --n "${N}" --p "${P}" --dirs "${DIRS}" \
      --noise "${NOISE}" --seed 0 --out "${INPUT}" \
      || { echo "FATAL: generator failed N=${N} P=${P}"; exit 4; }
    for K in ${KS}; do
      if [ "${K}" -le "${P}" ]; then
        echo "[2134] SKIP N=${N} K=${K} P=${P}: not overcomplete (K<=P)"; continue
      fi
      CELL_OUT="${OUT}/cell_N${N}_K${K}_P${P}"
      echo "=== [2134] CELL N=${N} K=${K} P=${P}  $(date) ==="
      "${CARGO}" run --release -p gam-sae --example scale_k -- \
        "${INPUT}" "${CELL_OUT}" \
        --rows "${N}" --atoms "${K}" --block-size "${BLOCK_SIZE}" \
        --epochs "${EPOCHS}" --minibatch "${MINIBATCH}" --gpu off --raw-ok
      rc=$?
      # scale_k exits non-zero if a peak-RSS invariant is violated; parse
      # numbers.json for the per-cell PASS/FAIL verdict the coordinator closes on.
      verdict=$("${GVENV}/bin/python" - "${CELL_OUT}/numbers.json" "${rc}" "${EV_FLOOR}" "${N}" "${K}" "${P}" <<'PY'
import json, sys
path, rc, ev_floor, N, K, P = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
rec = {"issue": 2134, "N": N, "K": K, "P": P, "example_rc": rc}
try:
    with open(path) as f:
        j = json.load(f)
    inv = j.get("invariants", {})
    ev = float(j.get("fit", {}).get("final_ev", float("nan")))
    sparse = bool(inv.get("sparse_front_door"))
    no_dense = bool(inv.get("no_dense_nxk_by_peak_rss"))
    no_2nxp = bool(inv.get("no_second_full_nxp_by_payload_bound"))
    ev_ok = ev == ev and ev >= ev_floor
    verdict = "PASS" if (rc == 0 and sparse and no_dense and no_2nxp and ev_ok) else "FAIL"
    rec.update({"final_ev": ev, "sparse_front_door": sparse,
                "no_dense_nxk": no_dense, "no_second_nxp": no_2nxp,
                "ev_ok": ev_ok, "ev_floor": ev_floor,
                "peak_rss_bytes": j.get("rss", {}).get("measured_peak_bytes"),
                "dense_nxk_bytes": j.get("front_door", {}).get("dense_assignment_bytes_f32"),
                "verdict": verdict})
except Exception as exc:  # noqa: BLE001
    rec.update({"verdict": "FAIL", "error": f"{type(exc).__name__}: {exc}"})
print(json.dumps(rec))
PY
)
      echo "[2134] CELL RESULT ${verdict}"
      echo "${verdict}" >> "${SUMMARY}"
      echo "${verdict}" | grep -q '"verdict": "PASS"' || overall=1
    done
  done
done

echo "=== #2134 SHAPE-MATRIX SUMMARY ($(date)) ==="
cat "${SUMMARY}"
cells=$(wc -l < "${SUMMARY}")
passes=$(grep -c '"verdict": "PASS"' "${SUMMARY}" || true)
if [ "${overall}" -eq 0 ] && [ "${cells}" -gt 0 ]; then
  echo "[#2134] VERDICT=PASS all ${passes}/${cells} overcomplete cells ran with EV sanity and peak-RSS discipline"
else
  echo "[#2134] VERDICT=FAIL ${passes}/${cells} cells passed"
fi
exit "${overall}"
