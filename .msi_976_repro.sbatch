#!/bin/bash
#SBATCH --job-name=m976repro
#SBATCH --partition=amd512
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=2:30:00
#SBATCH --output=/projects/standard/hsiehph/sauer354/m976_repro_%j.out
set -uo pipefail
G=/projects/standard/hsiehph/sauer354
source "$G/gam_env.sh" >/dev/null 2>&1
CLONE=$G/gam-976
echo "=== sync dedicated clone $CLONE to origin/main ==="
if [ ! -d "$CLONE/.git" ]; then git clone "$G/gam" "$CLONE"; fi
cd "$CLONE"
git fetch "$G/gam" main 2>&1 | tail -2
git reset --hard FETCH_HEAD
git log --oneline -1
echo "=== build _rust.abi3.so in-place (dedicated venv + clone-local target, no shared clobber) ==="
# Clone-local target dir so we do not race other agents building in the shared
# .../gam/target (gam_env.sh points CARGO_TARGET_DIR there).
export CARGO_TARGET_DIR=$CLONE/target976
VENV=$CLONE/.venv976
if [ ! -x "$VENV/bin/python" ]; then /usr/bin/python3.11 -m venv "$VENV"; fi
"$VENV/bin/python" -m pip install -q --upgrade maturin numpy 2>&1 | tail -2
PY=$VENV/bin/python
# maturin needs VIRTUAL_ENV to locate the interpreter to install into.
export VIRTUAL_ENV=$VENV
export PATH=$VENV/bin:$PATH
"$VENV/bin/maturin" develop --release 2>&1 | tail -8
echo "BUILD_RC=$?"

run_driver () {
  local TAG="$1"; local THREADS="$2"
  echo "############## DRIVER $TAG (RAYON_NUM_THREADS=$THREADS) ##############"
  RAYON_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS "$PY" - "$THREADS" <<'PYEOF' 2>&1 | grep -vE '^\['
import sys
import json, numpy as np, gamfit
import gamfit._rust as _r
print("rust so:", _r.__file__, flush=True)
G="/projects/standard/hsiehph/sauer354"; L=25
import os
# Freshly recompute Z each run AND report its hash: if ZHASH flips run-to-run,
# the numpy openblas SVD preprocessing is itself nondeterministic (the SAE input
# differs). Then run the SAE on a PINNED Z (computed once, reused) so any EV flip
# at fixed pinned Z is gam's own nondeterminism.
acts=np.load(f"{G}/olmo_data/base/activations.npy")[:,L,:].astype(np.float64)
Xc=acts-acts.mean(0); _,_,Vt=np.linalg.svd(Xc,full_matrices=False); Zf=Xc@Vt[:32].T; Zf=Zf/Zf.std(0).mean()
print("ZHASH_fresh=%d Zsum_fresh=%.12e"%(hash(Zf.tobytes()),float(Zf.sum())),flush=True)
ZP="/projects/standard/hsiehph/sauer354/Z_976_pinned.npy"
if not os.path.exists(ZP):
    np.save(ZP, Zf)
Z=np.load(ZP)
print("ZHASH_pinned=%d"%hash(Z.tobytes()),flush=True)
for K in [2,3]:
    try:
        m=gamfit.sae_manifold_fit(Z,K=K,d_atom=1,atom_topology="circle",n_iter=30,random_state=42,
            assignment="ibp_map",smoothness_weight=1.0,sparsity_weight=0.5)
        fitted=np.asarray(m.fitted); ev=1-((Z-fitted)**2).sum()/((Z-Z.mean(0))**2).sum()
        dec=np.concatenate([np.asarray(a.decoder_coefficients).ravel() for a in m.atoms])
        h=hash(dec.tobytes())
        print("RESULT K=%d EV=%.6f dechash=%d fittedsum=%.12e"%(K,ev,h,float(fitted.sum())),flush=True)
    except Exception as e:
        import traceback; traceback.print_exc()
        print("RESULT K=%d FAILED %s: %s"%(K,type(e).__name__,str(e)[:200]),flush=True)
PYEOF
}

run_driver "A_t32" 32
run_driver "B_t32" 32
run_driver "C_t8"  8
run_driver "D_t1"  1

echo "############## RUST REGRESSION TESTS (#976) ##############"
export CARGO_PROFILE_TEST_OPT_LEVEL=2
cargo test -p gam --lib -- \
  co_collapse_reseed_rotation_explores_distinct_subspaces \
  pca_seed_is_run_to_run_reproducible \
  decoder_norm_guard_reseeds_all_atoms_on_total_co_collapse_k3 \
  decoder_norm_guard_is_noop_for_k1 \
  --nocapture 2>&1 | tail -40
echo "RUST_TEST_RC=$?"
echo "REPRO_DONE"
