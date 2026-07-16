#!/usr/bin/env python3
"""#2228 Repro B — wide-p stagewise manifold fit MUST converge and RETURN.

The #2228 headline failure: `sae_manifold_fit_stagewise` raised
`RemlConvergenceError` ("inner solve did not converge at fixed rho; objective
stalled for 3 consecutive refine rounds ... and the undamped evidence
factorization failed at each stall point") on LLM-width residual-stream
activations (p=4096), and did not even RETURN within 5 min. Repro A (the p=8
synthetic circle) already passes on MSI (job 12896936) after the stall-acceptance
deflation fix. This driver is the remaining code-verification: the SAME failure
mode at LLM width, which the shared-root #2330 non-conservative-gradient fix
(de78ac795) + the #2230 objective-keyed incumbent + plateau termination are
expected to relieve.

What it asserts (two independent, objective bars — no reference tool):
  1. CONVERGENCE: the stagewise fit returns a model instead of raising the
     RemlConvergenceError / GamError inner-stall sentinel.
  2. LATENCY: it returns within a wall budget (default 300 s, the issue's own
     "does not return within 5 min" bar).

Data: pass `--data PATH.npy` to use a real (N, 4096) residual-act slice on MSI
(the faithful Repro B). Absent a path, it synthesises a wide-p analogue that
exercises the SAME rank-deficient wide-p geometry the issue blames: a few planted
1-D circles embedded in orthogonal 2-planes inside R^p, mean-centered, tiny N
(N << p) so the per-row H_tt is structurally rank-deficient exactly as at p=4096.
Both arms are reported; only the arm(s) actually run gate the verdict.

Uses gamfit's PUBLIC API only. Prints one machine-parseable RESULT line and a
PASS/FAIL VERDICT line so the job log alone verifies the issue.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def wide_p_circle_analogue(n, p, m_circles, radius, noise, seed):
    """N tokens on M planted unit circles in mutually-orthogonal 2-planes of R^p,
    mean-centered. N < p and 2*M << p, so the fit sees the wide-p, low-N,
    rank-deficient geometry #2228 blames — without needing the real 16 GB act
    tensor. This is the synthetic FALLBACK; --data is the faithful arm."""
    if p < 2 * m_circles:
        raise SystemExit(f"need p>=2*m_circles; got p={p}, m_circles={m_circles}")
    rng = np.random.default_rng(seed)
    # Orthonormal frame: first 2*M columns of a random orthogonal matrix.
    frame, _ = np.linalg.qr(rng.standard_normal((p, 2 * m_circles)))
    which = rng.integers(0, m_circles, size=n)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    coords = np.zeros((n, 2 * m_circles), dtype=np.float64)
    for i in range(n):
        c = which[i]
        coords[i, 2 * c] = radius * np.cos(theta[i])
        coords[i, 2 * c + 1] = radius * np.sin(theta[i])
    x = coords @ frame.T
    x += noise * rng.standard_normal((n, p))
    x -= x.mean(axis=0, keepdims=True)
    return np.ascontiguousarray(x, dtype=np.float32)


def load_real(path):
    x = np.load(path).astype(np.float64)
    if x.ndim != 2:
        raise SystemExit(f"[#2228] --data must be 2-D (N, p); got {x.shape}")
    x -= x.mean(axis=0, keepdims=True)
    return np.ascontiguousarray(x, dtype=np.float32)


def preflight():
    """Prove the wheel exposes sae_manifold_fit_stagewise before spending the job."""
    import gamfit

    ver = getattr(gamfit, "__version__", "?")
    where = os.path.dirname(getattr(gamfit, "__file__", "?"))
    if not hasattr(gamfit, "sae_manifold_fit_stagewise"):
        raise SystemExit(
            f"[preflight] gamfit {ver} at {where} has no sae_manifold_fit_stagewise")
    print(f"[#2228] preflight OK: gamfit {ver} at {where}", flush=True)


def run_one(name, x, args):
    """Fit; classify (converged | inner-stall-refusal | other-error) with wall time."""
    import gamfit

    print(f"[#2228] START {name}: X{tuple(x.shape)} d_atom={args.d_atom} "
          f"topology={args.atom_topology} max_births={args.max_births} "
          f"n_iter={args.n_iter} whitening={args.structured_whitening}", flush=True)
    t0 = time.time()
    outcome, detail = "converged", ""
    try:
        gamfit.sae_manifold_fit_stagewise(
            np.ascontiguousarray(x), d_atom=args.d_atom,
            atom_topology=args.atom_topology, assignment=args.assignment,
            structured_whitening=args.structured_whitening, fisher_factors=None,
            max_births=args.max_births, max_backfit_sweeps=args.max_backfit_sweeps,
            n_iter=args.n_iter, random_state=args.seed)
    except Exception as exc:  # noqa: BLE001 — classify, do not swallow
        msg = str(exc)
        detail = msg.splitlines()[0][:200] if msg else type(exc).__name__
        low = msg.lower()
        if "did not converge at fixed" in low or "inner solve" in low \
                or "objective stalled" in low:
            outcome = "inner_stall_refusal"
        else:
            outcome = "other_error"
    dt = time.time() - t0
    within = dt <= args.wall_budget
    row = {"arm": name, "shape": list(x.shape), "outcome": outcome,
           "detail": detail, "seconds": dt, "within_budget": within,
           "wall_budget": args.wall_budget}
    print(f"[#2228] {name}: outcome={outcome} seconds={dt:.1f} "
          f"within_budget={within} detail={detail!r}", flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="path to a real (N, 4096) mean-centerable .npy act slice")
    ap.add_argument("--p", type=int, default=4096)
    ap.add_argument("--n", type=int, default=70)
    ap.add_argument("--m-circles", type=int, default=3)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--d-atom", type=int, default=1)
    ap.add_argument("--atom-topology", default="circle")
    ap.add_argument("--assignment", default="ibp_map")
    ap.add_argument("--structured-whitening", action="store_true")
    ap.add_argument("--max-births", type=int, default=8)
    ap.add_argument("--max-backfit-sweeps", type=int, default=2)
    ap.add_argument("--n-iter", type=int, default=40)
    ap.add_argument("--wall-budget", type=float, default=300.0,
                    help="the issue's own 5-min return bar")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_2228_reproB.jsonl")
    args = ap.parse_args()

    preflight()

    arms = []
    if args.data is not None:
        arms.append(("real", load_real(args.data)))
    else:
        arms.append(("synthetic_wide_p",
                     wide_p_circle_analogue(args.n, args.p, args.m_circles,
                                            args.radius, args.noise, args.seed)))

    rows = [run_one(name, x, args) for name, x in arms]
    # Verdict: every arm run must converge AND return within budget.
    ok = all(r["outcome"] == "converged" and r["within_budget"] for r in rows)
    verdict = "PASS" if ok else "FAIL"
    result = {"issue": 2228, "repro": "B", "arms": rows,
              "structured_whitening": args.structured_whitening, "verdict": verdict}
    print("[#2228] RESULT " + json.dumps(result), flush=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(result) + "\n")
    print(f"[#2228] VERDICT={verdict} "
          f"outcomes={[(r['arm'], r['outcome'], round(r['seconds'], 1)) for r in rows]}",
          flush=True)
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
