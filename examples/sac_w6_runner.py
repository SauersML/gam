"""SAC vs joint K=8 on the real W6 OLMo activations (SAC_PLAN Part 5 kill-test).

Runs on node2. Loads the EXACT (500, 128) matrix the joint ``sae_manifold_fit(
K=8, d_atom=1, circle, ibp_map, isometry_weight=1.0)`` fit timed out on (W6:
3x1500s TIMEOUT). Runs SAC as 8 forced sequential K=1 fits under the same
whitened / isometry-gauged settings and reports, per atom: marginal EV, whether
EV climbs monotonically, decoder/recon finiteness, gate mass, and wall time.

Kill-criterion (Part 5): if eight sequential K=1 fits do NOT produce eight
healthy atoms with climbing EV in well under 25 min (where the joint fit timed
out), the SAC diagnosis is WRONG. This runner prints a PASS/FAIL verdict.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sac_prototype import _ev, sac_fit  # noqa: E402

CACHE = os.environ.get("SAC_W6_CACHE", "/dev/shm/w6/cache_K8.npy")
OUT_DIR = Path(os.environ.get("SAC_W6_OUT", "/dev/shm/sauers_gpu/sac_w6"))
N_ITER = int(os.environ.get("SAC_W6_ITER", "8"))
K_TARGET = int(os.environ.get("SAC_W6_K", "8"))
# Kill-test matches the joint baseline's settings: isometry gauge on, no
# structured-residual (Sigma) whitening, no backfit -- it isolates the narrow
# Part-5 question "do 8 sequential K=1 fits work where the joint timed out".
SRP = int(os.environ.get("SAC_W6_SRP", "0"))
BACKFIT = int(os.environ.get("SAC_W6_BACKFIT", "0"))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X = np.ascontiguousarray(np.load(CACHE), dtype=np.float32)
    n, p = X.shape
    print(f"[sac_w6] loaded {CACHE}: X={X.shape} dtype={X.dtype}", flush=True)
    print(f"[sac_w6] running SAC: {K_TARGET} forced sequential K=1 fits "
          f"(d_atom=1, circle, ibp_map, isometry_weight=1.0, "
          f"structured_residual_passes={SRP}, n_iter={N_ITER}, backfit={BACKFIT})",
          flush=True)

    t0 = time.time()
    sac = sac_fit(
        X,
        max_atoms=K_TARGET,
        d_atom=1,
        atom_topology="circle",
        assignment="ibp_map",
        ev_floor=1e-3,
        structured_residual_passes=SRP,
        n_iter=N_ITER,
        backfit_sweeps=BACKFIT,
        isometry_weight=1.0,
        stop_on_rejections=False,   # force all 8 births to test the criterion
        random_state=0,
        verbose=True,
    )
    wall = time.time() - t0

    # Per-atom health + EV climb.
    per_atom = []
    running = np.zeros_like(X, dtype=np.float64)
    prev_ev = 0.0
    climbing = True
    for k, (atom, log) in enumerate(zip(sac.atoms, sac.birth_log)):
        running = running + atom.recon
        ev_now = _ev(X, running)
        d = ev_now - prev_ev
        healthy = (
            bool(np.all(np.isfinite(atom.recon)))
            and float(np.abs(atom.assignments).sum()) > 0.0
            and abs(log["delta_ev"]) < 1e6
        )
        if d < -1e-6:
            climbing = False
        per_atom.append({
            "atom": k,
            "delta_ev": float(log["delta_ev"]),
            "cumulative_ev": float(ev_now),
            "ev_step": float(d),
            "cleared_floor": bool(log["cleared_floor"]),
            "hybrid": log["hybrid"],
            "gate_mass": float(np.abs(atom.assignments).sum()),
            "recon_finite": bool(np.all(np.isfinite(atom.recon))),
            "healthy": healthy,
        })
        prev_ev = ev_now

    n_healthy = sum(a["healthy"] for a in per_atom)
    all_finite = all(a["recon_finite"] for a in per_atom)
    verdict_pass = (
        len(sac.atoms) == K_TARGET
        and n_healthy == K_TARGET
        and climbing
        and all_finite
        and wall < 25 * 60
    )
    result = {
        "experiment": "W6_OLMo_K8_via_SAC",
        "cache": CACHE,
        "n": n, "p": p, "K_target": K_TARGET, "n_iter": N_ITER,
        "joint_baseline": "sae_manifold_fit(K=8) TIMEOUT 3x1500s (W6 results_K8.json)",
        "sac_wall_s": round(wall, 1),
        "sac_atoms_born": len(sac.atoms),
        "sac_atoms_healthy": n_healthy,
        "sac_combined_ev": float(sac.combined_ev),
        "sac_ev_trace": [round(e, 4) for e in sac.ev_trace],
        "ev_climbing_monotone": bool(climbing),
        "all_recon_finite": bool(all_finite),
        "per_atom": per_atom,
        "VERDICT": "PASS" if verdict_pass else "FAIL",
    }
    (OUT_DIR / "sac_w6_results.json").write_text(json.dumps(result, indent=2))
    print("\n================ SAC W6 RESULT ================", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    print(f"\n[sac_w6] VERDICT: {result['VERDICT']}  "
          f"({n_healthy}/{K_TARGET} healthy atoms, combined EV={sac.combined_ev:.4f}, "
          f"wall={wall:.1f}s vs joint 3x1500s TIMEOUT)", flush=True)


if __name__ == "__main__":
    main()
