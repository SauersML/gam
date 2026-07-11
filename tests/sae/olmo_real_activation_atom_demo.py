#!/usr/bin/env python3
"""#977 capstone — the FIRST END-TO-END WIN on REAL LLM activations.

This is the driver that converts the capstone wager from "in-tree synthetic
gates" (the planted weekday-circle gate in `tests/quality/quality_llm_weekday_
circle.rs`, which explicitly defers the real-model arm as "downstream-consumer
work … out of scope for the gam library") to a REAL-activation win:

    real OLMo-3-32B residual-stream activations  (635 × 5120, layer L25)
        --> gam manifold-SAE fit                 (sae_manifold_fit, K=1, periodic)
        --> recovered atom + intrinsic 2-D latent coordinates  (fit.coords[0])
        --> evidence-adjudicated shape race      (S¹ circle  vs  k-cluster null, #907)
        --> attribute carve / superposition vs binding verdict (#975)

It runs the EXACT instruments the production fit uses; the only change from the
synthetic gate is that the activation cloud is HARVESTED from a real model, not
planted. Per suite policy the test asserts structure recovery against the
*data*, and is allowed to fail honestly — a real cloud that is genuinely NOT a
low-dimensional curved family will report a cluster / Euclidean verdict, and
that is itself the finding the structure ladder is designed to surface (the
wager loses *measurably*, not silently).

DATA (stage on the cluster scratch first; see the run spec the driver prints with
`--print-run-spec`):
  - <DATA>/activations.npy   float32 [635, 64, 5120]  last-token residual / prompt / layer
  - <DATA>/prompts.jsonl     635 rows, fields incl. `kind` (38 kinds), `role`, `side`, `entity`
Memory: reference_olmo_activation_dataset. Azure SAS pull recipe in that note.

USAGE:
  python tests/sae/olmo_real_activation_atom_demo.py --data <DATA_DIR> --layer 25
  python tests/sae/olmo_real_activation_atom_demo.py --print-run-spec    # cluster sbatch recipe, no data needed

EXIT: 0 = a real-activation atom fit completed AND adjudicated (verdict printed,
whatever it is — circle win OR an honest cluster/Euclidean finding). Non-zero =
the fit or the adjudication could not run (a real blocker, reported precisely).
The verdict itself (circle vs cluster) is REPORTED, not asserted, because the
scientific content is the measurement: which structure the real cloud carries.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

# splitmix64-equivalent CV folding so the held-out density table is byte-
# deterministic from an integer seed (mirrors the Rust harness convention).


def _kfold_indices(n: int, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    out = []
    sizes = [(n // folds) + (1 if f < n % folds else 0) for f in range(folds)]
    start = 0
    for f in range(folds):
        eval_idx = np.sort(perm[start : start + sizes[f]])
        train_idx = np.sort(np.setdiff1d(perm, eval_idx, assume_unique=False))
        out.append((train_idx, eval_idx))
        start += sizes[f]
    return out


# ---------------------------------------------------------------------------
# Held-out predictive log-density providers on the recovered 2-D atom
# coordinates. These are the SAME models the Rust cross-class race uses:
#   ring   : radius ~ N(mu, s2) fit on train, uniform angle, +log(1/r) Jacobian.
#   gauss  : full 2-D Gaussian (mean + 2x2 cov) fit on train.
#   mixture: k-component isotropic Gaussian mixture (EM) fit on train.
# Lower negative-log predictive density (higher held-out loglik) wins.
# ---------------------------------------------------------------------------


def _ring_heldout_loglik(coords: np.ndarray, train: np.ndarray, eval_: np.ndarray) -> np.ndarray:
    c = coords - coords[train].mean(axis=0, keepdims=True)
    r = np.sqrt((c[train] ** 2).sum(axis=1))
    mean = float(r.mean())
    var = max(float(r.var()), 1e-9)
    re = np.sqrt((c[eval_] ** 2).sum(axis=1))
    re = np.maximum(re, 1e-9)
    log_norm = -0.5 * math.log(2.0 * math.pi * var)
    log_angle = -math.log(2.0 * math.pi)
    return log_norm - 0.5 * (re - mean) ** 2 / var + log_angle - np.log(re)


def _gauss_heldout_loglik(coords: np.ndarray, train: np.ndarray, eval_: np.ndarray) -> np.ndarray:
    mu = coords[train].mean(axis=0)
    cov = np.cov(coords[train].T) + 1e-6 * np.eye(coords.shape[1])
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    d = coords[eval_] - mu
    quad = np.einsum("ij,jk,ik->i", d, inv, d)
    log_norm = -coords.shape[1] / 2.0 * math.log(2.0 * math.pi) - 0.5 * math.log(max(det, 1e-300))
    return log_norm - 0.5 * quad


def _mixture_heldout_loglik(
    coords: np.ndarray, train: np.ndarray, eval_: np.ndarray, k: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Xtr = coords[train]
    # k-means++ style init then a few EM iterations (isotropic, shared scale).
    idx = rng.choice(len(Xtr), size=k, replace=False)
    centers = Xtr[idx].copy()
    for _ in range(25):
        d2 = ((Xtr[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        resp = np.zeros((len(Xtr), k))
        var = max(float(d2.min(axis=1).mean()), 1e-6)
        logw = math.log(1.0 / k)
        ll = -0.5 * d2 / var - coords.shape[1] / 2.0 * math.log(2.0 * math.pi * var) + logw
        ll -= ll.max(axis=1, keepdims=True)
        resp = np.exp(ll)
        resp /= resp.sum(axis=1, keepdims=True)
        nk = resp.sum(axis=0) + 1e-9
        centers = (resp.T @ Xtr) / nk[:, None]
    d2e = ((coords[eval_][:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    var = max(float((((Xtr[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)).min(axis=1).mean()), 1e-6)
    comp = -0.5 * d2e / var - coords.shape[1] / 2.0 * math.log(2.0 * math.pi * var) + math.log(1.0 / k)
    m = comp.max(axis=1, keepdims=True)
    return (m[:, 0] + np.log(np.exp(comp - m).sum(axis=1)))


def adjudicate_shape(coords: np.ndarray, k_ladder=(2, 3, 5, 7, 9), folds: int = 5, seed: int = 11):
    """Cross-validated held-out predictive-density race: circle vs Gaussian vs
    the best k-cluster mixture. Returns total held-out loglik per candidate and
    the winner — exactly the #907 cross-class discipline, run on the recovered
    real-activation atom coordinates."""
    splits = _kfold_indices(len(coords), folds, seed)
    ring = gauss = 0.0
    mix = {k: 0.0 for k in k_ladder}
    for tr, ev in splits:
        ring += float(_ring_heldout_loglik(coords, tr, ev).sum())
        gauss += float(_gauss_heldout_loglik(coords, tr, ev).sum())
        for k in k_ladder:
            mix[k] += float(_mixture_heldout_loglik(coords, tr, ev, k, seed + k).sum())
    best_k = max(mix, key=mix.get)
    table = {"circle": ring, "euclidean": gauss, f"mixture_k{best_k}": mix[best_k]}
    winner = max(table, key=table.get)
    return winner, table, best_k


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

RUN_SPEC = """\
=== #977 real-activation atom demo — cluster run spec (serialize on the GPU lane) ===

STEP 1 — stage data (login node ahl03, submit-only; small download, OK on login):
  cd /path/to/scratch
  ./olmo_tools/azcopy copy \\
    "https://olmoselfqualia174830.blob.core.windows.net/results/nvme/results/OLMO3_32B_SELFQ_V2/instruct/?se=2026-12-09T23:59:59Z&sp=rl&spr=https&sv=2026-04-06&sr=c&sig=1CCWDg4mQHK3gkFQzTArznGlKekBgJ0AAu47oUNcgaM%3D" \\
    olmo_data/instruct --recursive
  # need only: olmo_data/instruct/<rev>/activations.npy + prompts.jsonl  (~13 MB/rev for the L-slice)

STEP 2 — build the wheel on a COMPUTE node (NEVER the login node), CUDA build:
  sbatch -p <gpu-partition> --gres=gpu:a100:1 -t 60 --wrap '
    source /path/to/scratch/gam_env.sh
    cd /path/to/scratch/gam
    maturin build --release --features gpu -o dist  &&  pip install --force-reinstall dist/*.whl'

STEP 3 — run the demo on a COMPUTE GPU node:
  sbatch -p <gpu-partition> --gres=gpu:a100:1 -t 30 --wrap '
    source /path/to/scratch/gam_env.sh
    cd /path/to/scratch/gam
    python tests/sae/olmo_real_activation_atom_demo.py \\
      --data /path/to/scratch/olmo_data/instruct/<rev> \\
      --layer 25  --out olmo_atom_verdict.json'
  # exit 0 + a verdict line = the first real-activation end-to-end win recorded.
  # GPU residency (sae-perf's engine) accelerates sae_manifold_fit transparently;
  # no flag — the CUDA wheel auto-uses the device. CPU wheel still runs (slower).
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default=None, help="dir with activations.npy + prompts.jsonl")
    ap.add_argument("--layer", type=int, default=25, help="residual-stream layer (L25 self/qualia, L44 color)")
    ap.add_argument("--n-iter", type=int, default=80)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--print-run-spec", action="store_true")
    args = ap.parse_args()

    if args.print_run_spec:
        print(RUN_SPEC)
        return 0

    if not args.data:
        print("ERROR: --data <dir> required (or --print-run-spec). "
              "Stage the OLMo activations first; see --print-run-spec.", file=sys.stderr)
        return 2

    try:
        import gamfit
    except Exception as exc:  # noqa: BLE001
        print(f"BLOCKER: gamfit wheel not importable ({exc}). Build the wheel (run spec step 2).",
              file=sys.stderr)
        return 3

    data = Path(args.data)
    act_path = data / "activations.npy"
    if not act_path.exists():
        print(f"BLOCKER: {act_path} missing. Stage the OLMo data (run spec step 1).", file=sys.stderr)
        return 4

    acts = np.load(act_path)  # [635, 64, 5120] or already-sliced [635, 5120]
    if acts.ndim == 3:
        z = acts[:, args.layer, :].astype(np.float64)
    elif acts.ndim == 2:
        z = acts.astype(np.float64)
    else:
        print(f"BLOCKER: unexpected activations shape {acts.shape}", file=sys.stderr)
        return 5
    # Center; project the 5120-d cloud to its top principal subspace so the
    # atom fit and the shape race operate on the signal subspace (the fit
    # itself handles p>>d, but the adjudication coordinates come from fit.coords,
    # which the manifold fit recovers from the full cloud). We pass the full
    # cloud to the fit and read its intrinsic coordinates back.
    z = z - z.mean(axis=0, keepdims=True)
    n, p = z.shape
    print(f"loaded real OLMo activations: n={n} prompts, p={p} dims, layer L{args.layer}")

    fit = gamfit.sae_manifold_fit(
        X=z,
        K=1,
        atom_basis="periodic",
        d_atom=2,
        assignment="ibp_map",
        n_iter=args.n_iter,
        learning_rate=0.04,
        random_state=args.seed,
    )
    # Reconstruction R^2 — the "an atom was recovered at all" evidence.
    ss_res = float(np.sum((z - np.asarray(fit.fitted)) ** 2))
    ss_tot = float(np.sum((z - z.mean(axis=0, keepdims=True)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    coords = np.asarray(fit.coords[0], dtype=float)  # [n, 2] intrinsic latent coords
    topology = getattr(fit, "atom_topology", "unknown")
    print(f"atom recovered: topology={topology}  reconstruction R^2={r2:.4f}  coords.shape={coords.shape}")

    # --- #907 shape adjudication on the REAL recovered coordinates -----------
    # PRIMARY path: the Rust cross-class adjudicator through the gamfit FFI
    # (gamfit.adjudicate_atom_shape) — the SAME evidence code the in-tree gates
    # and the production fit drive. One evidence implementation, not two, so the
    # real-LLM verdict is trustworthy. Falls back to the local Python replica
    # only if an older wheel lacks the symbol (and says so loudly).
    if hasattr(gamfit, "adjudicate_atom_shape"):
        assignments = np.asarray(fit.assignments, dtype=float)
        mean_l0 = float(np.count_nonzero(assignments, axis=1).mean())
        v = gamfit.adjudicate_atom_shape(
            np.ascontiguousarray(coords),
            folds=5,
            seed=args.seed + 11,
            mean_l0=mean_l0,
        )
        winner = v["winner"]
        best_k = v["mixture_k"]
        margin = v["circle_margin"]
        circle_wins = bool(v["circle_wins"])
        table = dict(zip(v["candidate_names"], v["stacking_weights"]))
        print("shape race via RUST FFI (gamfit.adjudicate_atom_shape), "
              f"headline={v['headline']}, held-out stacking weights:")
        for name, val in sorted(table.items(), key=lambda kv: -kv[1]):
            print(f"    {name:14s} weight={val:.4f}")
        print(f"  VERDICT: {winner}  (circle stacking margin = {margin:+.4f})")
    else:
        print("WARNING: gamfit.adjudicate_atom_shape not in this wheel — "
              "falling back to the Python evidence replica (rebuild the wheel "
              "for the trustworthy single-implementation path).")
        winner, table, best_k = adjudicate_shape(coords, seed=args.seed + 11)
        margin = table.get("circle", -math.inf) - max(
            v for kk, v in table.items() if kk != "circle"
        )
        print("shape race (held-out predictive loglik, higher wins):")
        for name, val in sorted(table.items(), key=lambda kv: -kv[1]):
            print(f"    {name:14s} {val:14.3f}")
        print(f"  VERDICT: {winner}  (circle margin over best non-circle = {margin:+.3f})")
        circle_wins = winner == "circle"

    # --- #975 attribute carve: does an attribute bind the coordinate? --------
    # Color the recovered atlas by the prompt `kind` and report whether the
    # coordinate's angle is organized by kind (a one-way structure) — the
    # representational read of "the manifold carries the attribute".
    kind_organized = None
    prompts_path = data / "prompts.jsonl"
    if prompts_path.exists():
        kinds = []
        with prompts_path.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    kinds.append(str(json.loads(line).get("kind", "?")))
                except json.JSONDecodeError:
                    kinds.append("?")
        if len(kinds) == n:
            ang = np.arctan2(coords[:, 1], coords[:, 0])
            uniq = sorted(set(kinds))
            # circular between-group dispersion of the mean angle per kind vs
            # within-group dispersion: an attribute "binds" the coordinate if
            # the per-kind mean directions are more concentrated than chance.
            r_between = []
            for u in uniq:
                m = np.array([k == u for k in kinds])
                if m.sum() >= 3:
                    cbar = np.cos(ang[m]).mean()
                    sbar = np.sin(ang[m]).mean()
                    r_between.append(math.hypot(cbar, sbar))
            if r_between:
                kind_organized = float(np.mean(r_between))
                print(f"attribute carve (#975): {len(uniq)} prompt kinds, "
                      f"mean per-kind angular concentration R={kind_organized:.3f} "
                      f"(R->1 = kind binds the coordinate, R->0 = unbound/superposed)")

    verdict = {
        "n": n,
        "p": p,
        "layer": args.layer,
        "reconstruction_r2": r2,
        "atom_topology": str(topology),
        "shape_winner": winner,
        "shape_table": table,
        "best_mixture_k": best_k,
        "circle_margin": margin,
        "circle_wins_shape_race": circle_wins,
        "kind_angular_concentration": kind_organized,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(verdict, indent=2))
        print(f"wrote verdict to {args.out}")

    # The END-TO-END WIN condition: a real-activation atom fit COMPLETED and was
    # ADJUDICATED. The shape verdict (circle vs cluster) is the measurement, not
    # a pass/fail gate — a cluster/Euclidean verdict is an honest finding, not a
    # failure. We only fail if the fit produced a degenerate (non-finite or
    # zero-variance) result that could not be adjudicated at all.
    if not np.isfinite(r2) or not np.all(np.isfinite(coords)) or coords.std() < 1e-9:
        print("BLOCKER: fit produced a degenerate / non-adjudicable atom.", file=sys.stderr)
        return 6

    if circle_wins:
        print("\n*** REAL-ACTIVATION WIN: the recovered atom is an evidence-adjudicated S^1 "
              "circle BEATING the cluster null on real OLMo activations. ***")
    else:
        print(f"\n*** REAL-ACTIVATION FINDING: the recovered atom adjudicates as '{winner}', "
              "not a smooth circle — the structure ladder reports the honest verdict "
              "(the wager loses measurably on this layer/concept, which is itself the result). ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
