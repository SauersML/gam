"""#2502 d>=2 arm, re-run at MEASURED parameter parity against a REAL sphere.

The published "d>=2 loses decisively" number is not supported by the run that
produced it, for two independent reasons:

  1. the sphere atom was built as a CYLINDER at the time, so the 1600 "sphere"
     atoms in the mixed portfolio were not pricing a sphere; and
  2. parity was computed from ASSUMED basis widths (a 7-column `(lat, lon)`
     chart). The pole-free ambient basis carries the full `l <= 2` harmonics --
     nine columns -- so the arm labelled "param parity" at K=1818 in fact
     carried ~18% MORE decoder parameters than the linear arm it lost to.

Both defects have the same root: the comparison asserted a parameter count
instead of measuring one. So this harness MEASURES the realized decoder
parameter count of each arm from a small probe fit, derives the mixed-arm K
that matches the linear arm's budget, and only then runs the comparison.
Nothing here assumes a basis width.

Held-out EV is computed on a disjoint split via `model.reconstruct`, so the
number is a generalization measure and not the training `reconstruction_r2`.
"""
from __future__ import annotations

import argparse
import sys
import time

import numpy as np

# The mixed portfolio, `atom % 5` -- the same shape as the published arm, but
# its sphere entries now resolve to the real ambient-harmonic sphere.
MIXED_CYCLE = ["linear", "euclidean", "periodic", "sphere", "sphere"]
MIXED_DIMS = {"linear": 1, "euclidean": 2, "periodic": 1, "sphere": 3}


def portfolio(k):
    bases = [MIXED_CYCLE[i % len(MIXED_CYCLE)] for i in range(k)]
    return bases, [MIXED_DIMS[b] for b in bases]


def planted(n, p, concepts, active, harmonics, noise, seed):
    """Curved planted data: each concept is a smooth curve, rows are sparse mixes."""
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(concepts, harmonics * 2, p))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True) + 1e-12
    out = np.zeros((n, p))
    for row in range(n):
        which = rng.choice(concepts, size=active, replace=False)
        for c in which:
            t = rng.uniform(0.0, 1.0)
            for h in range(harmonics):
                out[row] += np.cos(2 * np.pi * (h + 1) * t) * directions[c, 2 * h]
                out[row] += np.sin(2 * np.pi * (h + 1) * t) * directions[c, 2 * h + 1]
    out += noise * rng.normal(size=out.shape)
    return np.ascontiguousarray(out)


def decoder_params(model):
    """Realized decoder parameter count, summed over atoms. Measured, not assumed."""
    d = model.to_dict()
    atoms = d.get("atoms") or []
    total = 0
    key_used = None
    for a in atoms:
        for key in ("decoder", "decoder_matrix", "gamma", "loadings"):
            v = a.get(key) if isinstance(a, dict) else None
            if v is None:
                continue
            arr = np.asarray(v, dtype=float)
            if arr.size:
                total += arr.size
                key_used = key
                break
    return total, key_used, len(atoms)


def ev(model, x):
    """Held-out explained variance via the model's own reconstruction."""
    recon = np.asarray(model.reconstruct(x), dtype=float)
    resid = float(((x - recon) ** 2).sum())
    total = float(((x - x.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - resid / total if total > 0 else float("nan")


def fit(gamfit, x, k, arm, args):
    kw = dict(
        K=k, assignment=args.assignment, n_iter=args.n_iter,
        random_state=args.seed, top_k=args.top_k,
        sparsity_weight=args.sparsity_weight,
        smoothness_weight=args.smoothness_weight,
        isometry_weight=0.0, learning_rate=args.learning_rate,
    )
    if arm == "linear":
        return gamfit.sae_manifold_fit(x, d_atom=1, atom_topology="linear", **kw)
    bases, dims = portfolio(k)
    return gamfit.sae_manifold_fit(x, d_atom=dims, atom_basis=bases, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k-linear", type=int, default=400)
    ap.add_argument("--probe-k", type=int, default=40)
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-test", type=int, default=1500)
    ap.add_argument("--p", type=int, default=24)
    ap.add_argument("--true-concepts", type=int, default=12)
    ap.add_argument("--n-active", type=int, default=3)
    ap.add_argument("--harmonics", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--n-iter", type=int, default=30)
    ap.add_argument("--assignment", default="softmax")
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--sparsity-weight", type=float, default=0.01)
    ap.add_argument("--smoothness-weight", type=float, default=0.01)
    ap.add_argument("--learning-rate", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import gamfit
    gamfit.set_log_level("warn")

    print("# 2502 d>=2 re-run at MEASURED parity, real ambient sphere")
    print(f"# n_train={args.n_train} n_test={args.n_test} p={args.p} "
          f"n_iter={args.n_iter} assignment={args.assignment}")

    train = planted(args.n_train, args.p, args.true_concepts, args.n_active,
                    args.harmonics, args.noise, args.seed + 1)
    test = planted(args.n_test, args.p, args.true_concepts, args.n_active,
                   args.harmonics, args.noise, args.seed + 7717)

    # --- Stage 1: MEASURE per-atom cost of each arm on a small probe. ---
    print(f"\n# probe fits at K={args.probe_k} to measure realized decoder params")
    probes = {}
    for arm in ("linear", "mixed"):
        t0 = time.perf_counter()
        m = fit(gamfit, train, args.probe_k, arm, args)
        total, key, n_atoms = decoder_params(m)
        per_atom = total / n_atoms if n_atoms else float("nan")
        probes[arm] = per_atom
        print(f"#   {arm:<7} atoms={n_atoms:<5} params={total:<8} "
              f"per_atom={per_atom:8.3f}  (key={key}) {time.perf_counter()-t0:.1f}s")

    if not all(np.isfinite(v) and v > 0 for v in probes.values()):
        print("# FATAL: could not measure decoder params; refusing to guess a parity K")
        return 2

    k_lin = args.k_linear
    k_mixed = max(1, round(k_lin * probes["linear"] / probes["mixed"]))
    ratio = probes["mixed"] / probes["linear"]
    print(f"\n# measured mixed/linear cost = {ratio:.3f}x")
    print(f"# parity: linear K={k_lin}  vs  mixed K={k_mixed}")

    # --- Stage 2: the comparison, at the K the measurement chose. ---
    print(f"\n{'arm':<8} | {'K':>6} | {'params':>9} | {'wall_s':>8} | {'heldout_EV':>10}")
    print(f"{'-'*8}-+-{'-'*6}-+-{'-'*9}-+-{'-'*8}-+-{'-'*10}")
    out = {}
    for arm, k in (("linear", k_lin), ("mixed", k_mixed)):
        t0 = time.perf_counter()
        try:
            m = fit(gamfit, train, k, arm, args)
            total, _, _ = decoder_params(m)
            e = ev(m, test)
            out[arm] = (k, total, e)
            print(f"{arm:<8} | {k:>6} | {total:>9} | {time.perf_counter()-t0:>8.1f} | {e:>10.4f}")
        except Exception as exc:  # noqa: BLE001
            print(f"{arm:<8} | {k:>6} | {'-':>9} | {time.perf_counter()-t0:>8.1f} | "
                  f"FAILED {type(exc).__name__}: {exc}"[:160])

    if len(out) == 2:
        pl, pm = out["linear"][1], out["mixed"][1]
        skew = abs(pm - pl) / max(pl, 1)
        d = out["linear"][2] - out["mixed"][2]
        print(f"\n# realized parameter skew: {skew*100:.1f}% "
              f"(linear {pl} vs mixed {pm})")
        print(f"# linear EV - mixed EV = {d:+.4f}")
        if skew > 0.05:
            print("# WARNING: >5% skew -- this is NOT a parity comparison, do not "
                  "report it as one")
        else:
            print(f"# VERDICT: at measured parity, "
                  f"{'linear wins' if d > 0 else 'mixed wins'} by {abs(d):.4f}")
    print("\n=== D2_PARITY_DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
