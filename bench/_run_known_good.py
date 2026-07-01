"""Run the known-good manifold-SAE e2e tests standalone (bypasses pandas conftest).

Confirms with the CURRENT build that (1) the e2e pipeline reaches r2>=0.95 at K=2
and (2) curved atoms beat linear shards. Threads pinned to 1 (RAM/CPU safe).
"""
import os
for _v in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))


def run(modname, fnname, label):
    mod = __import__(modname)
    fn = getattr(mod, fnname)
    t0 = time.time()
    try:
        fn()
        print(f"{label}: PASSED in {time.time()-t0:.1f}s", flush=True)
        return True
    except AssertionError as e:
        print(f"{label}: ASSERTION FAILED in {time.time()-t0:.1f}s: {str(e)[:400]}", flush=True)
    except Exception as e:
        print(f"{label}: ERROR in {time.time()-t0:.1f}s: {type(e).__name__}: {str(e)[:400]}", flush=True)
    return False


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "k2"):
        run("test_sae_manifold_synthetic_quality_ground_truth",
            "test_fit_learns_disjoint_periodic_atoms_without_inactive_leakage",
            "KNOWN-GOOD #1 (K=2 e2e r2>=0.95)")
    if which in ("all", "beat"):
        run("test_sae_manifold_curved_beats_linear",
            "test_curved_atom_beats_linear_shards_on_one_harmonic",
            "KNOWN-GOOD #2 (curved BEATS linear r2>=0.9)")
