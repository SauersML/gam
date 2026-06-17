#!/usr/bin/env python
"""#1026 — STRUCTURAL-TRUTH LEDGER for the manifold-SAE on real OLMo activations.

Turns the manifold-SAE measurements into a PRINCIPLED, FALSIFIABLE determination
of what structure is GENUINELY present in LLM residual-stream activations and
which method choices are actually GOOD. Five falsifiable tests, each yielding a
number/verdict; aggregated into one ledger.

  1. EV-vs-Theta frontier (the discriminating measurement): per fitted atom,
     (fitted turning Theta = integral kappa ds, in-sample LOAO Delta-EV). The
     per-atom Delta-EV is an IN-SAMPLE leave-one-atom-out drop on the TRAINING
     matrix (#1226), not a held-out number — only the dictionary-level ev_out is
     measured on the held-out split. Classify GENUINE-CURVED (high Theta + earns
     EV) vs LINEAR (Theta~0 + earns EV) vs OVERFIT-CURVATURE (high Theta + ~0
     EV). Report the fraction of in-sample LOAO Delta-EV attributable to
     genuinely-curved atoms.
  2. EV-vs-K curve shape: K in {1,2,4,8,16,32}, curved vs linear dictionaries,
     held-out EV each. FLATTEN-early (structured minority thesis) vs KEEP-CLIMBING
     (pervasive structure, parity live).
  3. Evidence-based adjudication: per atom, the rank-aware Laplace evidence margin
     (NLE_linear - NLE_curved) from solver/evidence.rs select_hybrid_atom. Curved
     is TRUE only when it WINS on evidence, not merely fits better.
  4. Seed-stability: re-fit >=3 seeds; a structure is TRUE only if it reappears
     (similar Theta, similar in-sample LOAO Delta-EV) across seeds.
  5. Two-directional null test: (a) planted circle must recover; (b) matched
     ISOTROPIC NOISE (same dim/scale, no structure) must NOT hallucinate curved
     structure. Recover real AND reject noise = genuine truth-discrimination.

NO fitting math lives here. It calls gamfit.sae_manifold_fit + ManifoldSAE.reconstruct,
and reads the per-atom hybrid-split report (Theta, Delta-EV, evidence margin)
that the production engine emits.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np


# --------------------------------------------------------------------------
# #1204: ManifoldSAE now surfaces the `hybrid_split` report as a public field,
# so the per-atom verdicts (fitted_turning Theta, train_loao_delta_ev — the
# honest in-sample LOAO key, #1226 — curved_evidence_margin) are readable
# directly off the model — no monkey-patch of from_payload needed any more.
# Kept as a no-op for call-site compatibility.
# --------------------------------------------------------------------------
def _install_payload_capture():
    return None


def _ev(target: np.ndarray, fitted: np.ndarray) -> float:
    resid = target - fitted
    denom = float(np.sum((target - target.mean(axis=0, keepdims=True)) ** 2))
    if denom <= 0.0:
        return float("nan")
    return 1.0 - float(np.sum(resid ** 2)) / denom


def _pca_project(train, test, pcs):
    mean = train.mean(axis=0, keepdims=True)
    tc = train - mean
    _, _, vt = np.linalg.svd(tc, full_matrices=False)
    comp = vt[: min(pcs, vt.shape[0])].T
    z_tr = tc @ comp
    z_te = (test - mean) @ comp
    scale = float(np.sqrt(np.mean(z_tr ** 2))) or 1.0
    return z_tr / scale, z_te / scale


def _fit(z_tr, k, topology, seed, n_iter):
    from gamfit import sae_manifold_fit

    return sae_manifold_fit(
        z_tr, K=k, d_atom=1, atom_topology=topology,
        assignment="ibp_map", n_iter=n_iter, random_state=seed,
    )


def _hybrid_atoms(model):
    """Return the per-atom hybrid-split verdict list, or [] if absent.

    Reads the now-public ``ManifoldSAE.hybrid_split`` field (#1204).
    """
    hs = getattr(model, "hybrid_split", None)
    if not hs:
        return []
    return hs.get("atoms", [])


# --------------------------------------------------------------------------
# Test 1 + 3: EV-vs-Theta frontier + evidence adjudication (one fit).
# --------------------------------------------------------------------------
THETA_HI = 0.5  # rad; below this an atom is a linear-tail direction wearing a curve.
EV_FLOOR = 1e-3  # in-sample LOAO Delta-EV below which an atom earns ~nothing.


def classify_atom(theta, dev, margin):
    curved_geom = (theta is not None) and (theta >= THETA_HI)
    earns = (dev is not None) and (dev >= EV_FLOOR)
    wins_evidence = (margin is not None) and (margin > 0.0)
    if curved_geom and earns and wins_evidence:
        return "GENUINE_CURVED"
    if curved_geom and earns and not wins_evidence:
        return "CURVED_FITS_NOT_PAID"  # fits better but curvature doesn't pay
    if curved_geom and not earns:
        return "OVERFIT_CURVATURE"
    if (not curved_geom) and earns:
        return "LINEAR"
    return "INERT"


def test_ev_theta(z_tr, z_te, k, seed, n_iter):
    m = _fit(z_tr, k, "circle", seed, n_iter)
    ev_out = _ev(z_te, m.reconstruct(z_te))
    atoms = _hybrid_atoms(m)
    rows = []
    for a in atoms:
        theta = a.get("fitted_turning")
        # #1226 — this is the IN-SAMPLE leave-one-atom-out ΔEV (computed on the
        # training matrix during the fit), surfaced under the honest key
        # ``train_loao_delta_ev``. The legacy ``held_out_delta_ev`` key carried
        # the SAME in-sample number under a misleading name; read the honest key
        # and fall back to the deprecated alias only for older payloads.
        dev = a.get("train_loao_delta_ev")
        if dev is None:
            dev = a.get("held_out_delta_ev")
        margin = a.get("curved_evidence_margin")
        rows.append({
            "atom": a.get("atom"),
            "theta": theta,
            "delta_ev": dev,
            "evidence_margin": margin,
            "kept_curved": a.get("kept_curved"),
            "class": classify_atom(theta, dev, margin),
        })
    # Fraction of (positive) in-sample LOAO Delta-EV attributable to
    # genuinely-curved atoms (#1226: per-atom Delta-EV is in-sample, not held-out).
    pos = [r for r in rows if r["delta_ev"] is not None and r["delta_ev"] > 0]
    tot_dev = sum(r["delta_ev"] for r in pos) or float("nan")
    curved_dev = sum(r["delta_ev"] for r in pos if r["class"] == "GENUINE_CURVED")
    linear_dev = sum(r["delta_ev"] for r in pos if r["class"] == "LINEAR")
    return {
        "K": k, "seed": seed, "ev_out": ev_out,
        "atoms": rows,
        "ev_from_genuine_curved": (curved_dev / tot_dev) if tot_dev == tot_dev else None,
        "ev_from_linear": (linear_dev / tot_dev) if tot_dev == tot_dev else None,
        "n_genuine_curved": sum(1 for r in rows if r["class"] == "GENUINE_CURVED"),
        "n_linear": sum(1 for r in rows if r["class"] == "LINEAR"),
        "n_overfit": sum(1 for r in rows if r["class"] == "OVERFIT_CURVATURE"),
        "n_curved_wins_evidence": sum(
            1 for r in rows if r["evidence_margin"] is not None and r["evidence_margin"] > 0
        ),
        "n_atoms_adjudicated": len(rows),
    }


# --------------------------------------------------------------------------
# Test 2: EV-vs-K curve shape (curved vs Euclidean-quadratic-patch dictionaries).
#
# NOTE (#1221): the comparison arm uses ``atom_topology="euclidean"``, which is a
# degree-2 QUADRATIC monomial patch ``{1, t, t²}`` — NOT a true rank-1 linear
# atom ``γ(t)=t·b``. So this is "curved vs Euclidean quadratic patch", a STRONGER
# baseline than a true linear atom; the curved advantage it shows is a LOWER
# bound on the curved-vs-linear advantage. The fields are named
# ``euclidean_quadratic_*`` to stop mislabeling the quadratic patch as "linear".
# A first-class true-linear atom would require a degree-1 EuclideanPatch path in
# the FFI (see #1221).
# --------------------------------------------------------------------------
def test_ev_k(z_tr, z_te, ladder, seed, n_iter):
    table = []
    for k in ladder:
        ev_c = _ev(z_te, _fit(z_tr, k, "circle", seed, n_iter).reconstruct(z_te))
        ev_l = _ev(z_te, _fit(z_tr, k, "euclidean", seed, n_iter).reconstruct(z_te))
        table.append({"K": k, "curved_ev_out": ev_c,
                      "euclidean_quadratic_ev_out": ev_l,
                      "margin": ev_c - ev_l})
    # Flatten verdict: does curved EV gain flatten after the first few K?
    cs = [r["curved_ev_out"] for r in table]
    verdict = "INSUFFICIENT"
    if len(cs) >= 3 and all(c == c for c in cs):
        early = cs[min(2, len(cs) - 1)] - cs[0]   # gain over first ~3 rungs
        late = cs[-1] - cs[min(2, len(cs) - 1)]    # gain over the rest
        if early > 0 and late <= 0.25 * max(early, 1e-9):
            verdict = "FLATTEN_EARLY (structured minority)"
        elif late > 0.25 * max(early, 1e-9):
            verdict = "KEEP_CLIMBING (pervasive structure)"
        else:
            verdict = "FLAT_THROUGHOUT"
    return {"table": table, "shape_verdict": verdict}


# --------------------------------------------------------------------------
# Test 4: seed-stability.
# --------------------------------------------------------------------------
def test_seed_stability(z_tr, z_te, k, seeds, n_iter):
    runs = [test_ev_theta(z_tr, z_te, k, s, n_iter) for s in seeds]
    ev_outs = [r["ev_out"] for r in runs]
    thetas = []  # max-Theta per run (the dominant curved atom)
    for r in runs:
        ts = [a["theta"] for a in r["atoms"] if a["theta"] is not None]
        thetas.append(max(ts) if ts else None)
    fr = [r["ev_from_genuine_curved"] for r in runs]
    fr = [f for f in fr if f is not None]
    ev_arr = np.array([e for e in ev_outs if e == e])
    return {
        "seeds": list(seeds),
        "ev_out_per_seed": ev_outs,
        "ev_out_mean": float(ev_arr.mean()) if ev_arr.size else None,
        "ev_out_std": float(ev_arr.std()) if ev_arr.size else None,
        "max_theta_per_seed": thetas,
        "ev_from_genuine_curved_per_seed": fr,
        "n_genuine_curved_per_seed": [r["n_genuine_curved"] for r in runs],
        "stable": bool(ev_arr.size >= 2 and ev_arr.std() < 0.05 * max(abs(ev_arr.mean()), 1e-6)),
    }


# --------------------------------------------------------------------------
# Test 5: two-directional null test.
# --------------------------------------------------------------------------
def test_null_directional(seed, n_iter, n=400, d=8):
    rng = np.random.default_rng(seed)
    # (a) planted circle in 2 of d dims, small noise on the rest.
    t = rng.uniform(0, 2 * math.pi, n)
    circ = np.zeros((n, d))
    circ[:, 0] = np.cos(t)
    circ[:, 1] = np.sin(t)
    circ += 0.05 * rng.standard_normal((n, d))
    # global unit-RMS scale
    circ /= (np.sqrt(np.mean(circ ** 2)) or 1.0)
    perm = rng.permutation(n)
    ntr = int(0.8 * n)
    c_tr, c_te = circ[perm[:ntr]], circ[perm[ntr:]]
    mc = _fit(c_tr, 1, "circle", seed, n_iter)
    ev_circle = _ev(c_te, mc.reconstruct(c_te))
    atoms_c = _hybrid_atoms(mc)
    theta_c = max([a["theta"] for a in atoms_c if a["theta"] is not None], default=None)
    margin_c = max([a["curved_evidence_margin"] for a in atoms_c
                    if a["curved_evidence_margin"] is not None], default=None)
    kept_c = any(a.get("kept_curved") for a in atoms_c)

    # (b) matched isotropic noise: same n, d, unit-RMS scale, NO structure.
    noise = rng.standard_normal((n, d))
    noise /= (np.sqrt(np.mean(noise ** 2)) or 1.0)
    nn_tr, nn_te = noise[perm[:ntr]], noise[perm[ntr:]]
    mn = _fit(nn_tr, 1, "circle", seed, n_iter)
    ev_noise = _ev(nn_te, mn.reconstruct(nn_te))
    atoms_n = _hybrid_atoms(mn)
    theta_n = max([a["theta"] for a in atoms_n if a["theta"] is not None], default=None)
    margin_n = max([a["curved_evidence_margin"] for a in atoms_n
                    if a["curved_evidence_margin"] is not None], default=None)
    kept_n = any(a.get("kept_curved") for a in atoms_n)
    # On matched noise the engine must REJECT curved: either collapse to linear
    # (kept_curved False / evidence margin <= 0) or earn ~0 held-out EV.
    noise_rejected = (not kept_n) or (margin_n is not None and margin_n <= 0.0) \
        or (ev_noise < 0.1)
    circle_recovered = (ev_circle > 0.5) and (theta_c is not None and theta_c > THETA_HI)
    return {
        "circle": {"ev_out": ev_circle, "max_theta": theta_c,
                   "max_evidence_margin": margin_c, "kept_curved": kept_c,
                   "recovered": bool(circle_recovered)},
        "isotropic_noise": {"ev_out": ev_noise, "max_theta": theta_n,
                            "max_evidence_margin": margin_n, "kept_curved": kept_n,
                            "rejected": bool(noise_rejected)},
        "truth_discrimination": bool(circle_recovered and noise_rejected),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--npy", required=True, help="OLMo activation fixture (N,D) or 3D (needs --olmo-layer)")
    ap.add_argument("--olmo-layer", type=int, default=None)
    ap.add_argument("--pcs", type=int, default=32)
    ap.add_argument("--k-ladder", default="1,2,4,8,16,32")
    ap.add_argument("--theta-k", type=int, default=8, help="K for the EV-vs-Theta frontier + seed test")
    ap.add_argument("--seeds", default="42,7,1234")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--out", default=None, help="write the full ledger JSON here")
    args = ap.parse_args()

    _install_payload_capture()

    arr = np.load(args.npy)
    if arr.ndim == 3:
        if args.olmo_layer is None:
            raise SystemExit("3D npy needs --olmo-layer")
        arr = arr[:, args.olmo_layer, :]
    x = np.asarray(arr, dtype=np.float64)
    n = x.shape[0]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ladder = [int(s) for s in args.k_ladder.split(",") if s.strip()]
    seed0 = seeds[0]

    rng = np.random.default_rng(seed0)
    perm = rng.permutation(n)
    n_test = max(1, int(round(args.test_frac * n)))
    z_tr, z_te = _pca_project(x[perm[n_test:]], x[perm[:n_test]], args.pcs)

    print(f"=== STRUCTURAL-TRUTH LEDGER (#1026) ===")
    print(f"data {args.npy}: N={n} D={x.shape[1]} -> PCA-{args.pcs}, "
          f"train={n - n_test} test={n_test}, seeds={seeds}")

    ledger = {"data": args.npy, "N": int(n), "D": int(x.shape[1]),
              "pcs": args.pcs, "seeds": seeds, "theta_k": args.theta_k}

    print("\n[T2] EV-vs-K curve (curved vs Euclidean quadratic patch, #1221)")
    t2 = test_ev_k(z_tr, z_te, ladder, seed0, args.n_iter)
    for r in t2["table"]:
        print(f"  K={r['K']:>3}  curved={r['curved_ev_out']:.6f}  "
              f"euclid_quad={r['euclidean_quadratic_ev_out']:.6f}  margin={r['margin']:+.6f}")
    print(f"  shape verdict: {t2['shape_verdict']}")
    ledger["T2_ev_vs_k"] = t2

    print(f"\n[T1+T3] EV-vs-Theta frontier + evidence adjudication (K={args.theta_k}, seed={seed0})")
    t1 = test_ev_theta(z_tr, z_te, args.theta_k, seed0, args.n_iter)
    for a in t1["atoms"]:
        th = "None" if a["theta"] is None else f"{a['theta']:.4f}"
        de = "None" if a["delta_ev"] is None else f"{a['delta_ev']:+.5f}"
        mg = "None" if a["evidence_margin"] is None else f"{a['evidence_margin']:+.4f}"
        print(f"  {a['atom']:<12} theta={th:>8}  dEV={de:>9}  "
              f"evmargin={mg:>9}  kept_curved={a['kept_curved']}  [{a['class']}]")
    print(f"  atoms adjudicated={t1['n_atoms_adjudicated']}  "
          f"genuine_curved={t1['n_genuine_curved']}  linear={t1['n_linear']}  "
          f"overfit={t1['n_overfit']}  curved_wins_evidence={t1['n_curved_wins_evidence']}")
    print(f"  in-sample LOAO ΔEV from genuine-curved atoms: {t1['ev_from_genuine_curved']}")
    print(f"  in-sample LOAO ΔEV from linear atoms:         {t1['ev_from_linear']}")
    ledger["T1_ev_vs_theta"] = t1

    print(f"\n[T4] seed-stability (K={args.theta_k}, seeds={seeds})")
    t4 = test_seed_stability(z_tr, z_te, args.theta_k, seeds, args.n_iter)
    print(f"  ev_out per seed: {[round(e,5) for e in t4['ev_out_per_seed']]}")
    print(f"  ev_out mean={t4['ev_out_mean']} std={t4['ev_out_std']}  stable={t4['stable']}")
    print(f"  max-theta per seed: {t4['max_theta_per_seed']}")
    print(f"  n_genuine_curved per seed: {t4['n_genuine_curved_per_seed']}")
    ledger["T4_seed_stability"] = t4

    print(f"\n[T5] two-directional null test (seed={seed0})")
    t5 = test_null_directional(seed0, args.n_iter)
    c, nz = t5["circle"], t5["isotropic_noise"]
    print(f"  (a) planted circle: ev_out={c['ev_out']:.6f} max_theta={c['max_theta']} "
          f"evmargin={c['max_evidence_margin']} kept_curved={c['kept_curved']} "
          f"-> recovered={c['recovered']}")
    print(f"  (b) isotropic noise: ev_out={nz['ev_out']:.6f} max_theta={nz['max_theta']} "
          f"evmargin={nz['max_evidence_margin']} kept_curved={nz['kept_curved']} "
          f"-> rejected={nz['rejected']}")
    print(f"  TRUTH-DISCRIMINATION (recover real AND reject noise): {t5['truth_discrimination']}")
    ledger["T5_null_test"] = t5

    if args.out:
        with open(args.out, "w") as f:
            json.dump(ledger, f, indent=2, default=lambda o: None)
        print(f"\nledger JSON -> {args.out}")


if __name__ == "__main__":
    main()
