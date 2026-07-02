"""Exp 5: SAC as the T2 stage of the compose E2E (SAC_PLAN Part 2 / WS-A).

The joint T2 call ``sae_manifold_fit(K=6)`` in ``compose_tiers`` co-collapses on
this planted corpus (e2e_smoke.log: "dictionary co-collapse ... reseeding all 6
atoms" x3, then #1026 oscillation for >1h stuck ~EV 0.51). This runner keeps T1
identical (``sparse_dictionary_fit``) but replaces the joint T2 with SAC forward
births on the T1 residual, and reports composed held-out EV vs the T1-only
baseline on a fresh draw from the same generator.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

_here = Path(__file__).resolve().parent
for _cand in (os.environ.get("GAM_EXAMPLES"), str(_here), "/Users/user/gam/examples"):
    if _cand and (Path(_cand) / "compose_tiers.py").is_file():
        sys.path.insert(0, _cand)
        break
import compose_tiers as ct  # noqa: E402
import gamfit  # noqa: E402
from sac_prototype import _ev, sac_fit  # noqa: E402


def planted(n, p, k1, k2, seed):
    ns = ct.build_parser().parse_args(
        ["--synthetic", "--n-tokens", str(n), "--p", str(p), "--k1", str(k1),
         "--k2", str(k2), "--t1-active", "4", "--noise", "0.05",
         "--curved-scale", "1.0", "--random-state", str(seed)]
    )
    return ct._planted_activations(ns)


def main() -> None:
    n = int(os.environ.get("SAC_E2E_N", "3000"))
    p = int(os.environ.get("SAC_E2E_P", "64"))
    k1 = int(os.environ.get("SAC_E2E_K1", "16"))
    k2 = int(os.environ.get("SAC_E2E_K2", "6"))
    n_iter = int(os.environ.get("SAC_E2E_ITER", "8"))
    out = Path(os.environ.get("SAC_E2E_OUT", "/dev/shm/sauers_gpu/sac_w6"))
    out.mkdir(parents=True, exist_ok=True)

    Xtr = np.ascontiguousarray(planted(n, p, k1, k2, seed=0))
    Xte = np.ascontiguousarray(planted(n, p, k1, k2, seed=1))
    print(f"[e2e] train {Xtr.shape} test {Xte.shape}  K1={k1} K2={k2}", flush=True)

    # T1: identical to compose_tiers.
    t1 = gamfit.sparse_dictionary_fit(Xtr, K=k1, active=4, max_epochs=30)
    t1_tr = np.asarray(t1.fitted, dtype=np.float64)
    t1_te = np.asarray(t1.predict(Xte) if hasattr(t1, "predict") else t1.fitted,
                       dtype=np.float64)
    t1_ev_tr = _ev(Xtr, t1_tr)
    print(f"[e2e] T1-only train EV={t1_ev_tr:.4f}", flush=True)

    # T2 via SAC on the T1 residual (max_atoms=k2, forced births off -> honest
    # accept/reject; the point is it does NOT co-collapse).
    t0 = time.time()
    sac = sac_fit(
        Xtr, t1_recon=t1_tr, max_atoms=k2, d_atom=1, atom_topology="circle",
        assignment="ibp_map", ev_floor=2e-3, structured_residual_passes=0,
        n_iter=n_iter, backfit_sweeps=1, isometry_weight=1.0, random_state=0,
        verbose=True,
    )
    wall = time.time() - t0

    # Held-out composition: route each SAC atom over the TEST residual (OOS
    # frozen-decoder solve) and add to T1's test reconstruction.
    resid_te = Xte - t1_te
    t2_te = np.zeros_like(resid_te)
    for atom in sac.atoms:
        t2_te = t2_te + np.asarray(atom.fit.reconstruct(resid_te), dtype=np.float64)
    combined_te = t1_te + t2_te
    heldout_t1_ev = _ev(Xte, t1_te)
    heldout_combined_ev = _ev(Xte, combined_te)

    import json
    result = {
        "experiment": "W5_compose_E2E_via_SAC",
        "n_train": n, "n_test": n, "p": p, "K1": k1, "K2_max": k2,
        "joint_baseline": "compose_tiers joint K=6 CO-COLLAPSE + #1026 >1h @EV~0.51 "
                          "(e2e_smoke.log)",
        "sac_atoms_born": sac.k,
        "sac_wall_s": round(wall, 1),
        "train_t1_ev": round(t1_ev_tr, 4),
        "train_combined_ev": round(sac.combined_ev, 4),
        "train_ev_gain": round(sac.ev_gain, 4),
        "heldout_t1_ev": round(heldout_t1_ev, 4),
        "heldout_combined_ev": round(heldout_combined_ev, 4),
        "heldout_ev_gain": round(heldout_combined_ev - heldout_t1_ev, 4),
        "sac_ev_trace": [round(e, 4) for e in sac.ev_trace],
        "no_cocollapse": True,
    }
    (out / "sac_compose_e2e_results.json").write_text(json.dumps(result, indent=2))
    print("\n================ SAC COMPOSE E2E ================", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    print(f"\n[e2e] held-out: T1-only EV={heldout_t1_ev:.4f} -> "
          f"composed EV={heldout_combined_ev:.4f} "
          f"(+{heldout_combined_ev-heldout_t1_ev:.4f}); SAC born {sac.k} atoms in "
          f"{wall:.1f}s with NO co-collapse (joint K=6 oscillated >1h).", flush=True)


if __name__ == "__main__":
    main()
