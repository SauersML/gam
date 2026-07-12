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
    ap.add_argument(
        "--pca-dims",
        type=int,
        default=None,
        help=(
            "fit on this many leading activation PCs; by default use the "
            "spectrum's participation-ratio effective rank"
        ),
    )
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
    # Center and ACTUALLY project the 5120-d cloud to its leading signal
    # subspace. The old example claimed this projection but passed the full
    # matrix, turning a 635-row demo into a >900 s wide-p solve (#2267). The
    # row-Gram eigendecomposition computes exact PCA scores without forming a
    # 5120x5120 covariance matrix.
    z = z - z.mean(axis=0, keepdims=True)
    n, input_p = z.shape
    gram = z @ z.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    nonnegative = np.maximum(eigenvalues, 0.0)
    spectral_mass = float(nonnegative.sum())
    spectral_square_mass = float(nonnegative @ nonnegative)
    effective_rank = spectral_mass * spectral_mass / max(
        spectral_square_mass, np.finfo(float).tiny
    )
    pca_dims = (
        int(args.pca_dims)
        if args.pca_dims is not None
        else max(2, int(math.ceil(effective_rank)))
    )
    if not 2 <= pca_dims <= min(n - 1, input_p):
        print(
            f"ERROR: --pca-dims must lie in [2, {min(n - 1, input_p)}]; got {pca_dims}",
            file=sys.stderr,
        )
        return 6
    order = np.argsort(eigenvalues)[::-1][:pca_dims]
    positive = np.maximum(eigenvalues[order], 0.0)
    explained = float(positive.sum() / max(spectral_mass, np.finfo(float).tiny))
    z = np.ascontiguousarray(eigenvectors[:, order] * np.sqrt(positive)[None, :])
    p = z.shape[1]
    print(
        f"loaded real OLMo activations: n={n} prompts, input_p={input_p}, "
        f"effective_rank={effective_rank:.2f}, fit_p={p}, "
        f"pca_ev={explained:.4f}, layer L{args.layer}"
    )

    fit = gamfit.sae_manifold_fit(
        X=z,
        K=1,
        atom_basis="periodic",
        d_atom=2,
        assignment="softmax",
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
    # The current Rust adjudicator is required. It owns all four density
    # classes, their certified fits, matched nulls, and cross-fit stacking.
    assignments = np.asarray(fit.assignments, dtype=float)
    mean_l0 = float(np.count_nonzero(assignments, axis=1).mean())
    v = gamfit.adjudicate_atom_shape(
        np.ascontiguousarray(coords),
        folds=5,
        seed=args.seed + 11,
        mean_l0=mean_l0,
    )
    winner_class = v["winner_class"]
    reporting_winner = v["reporting_winner"]
    mixture_reporting_k = v["mixture_reporting_k"]
    margin = v["circular_margin"]
    circle_wins = bool(v["circle_wins"])
    table = dict(zip(v["candidate_names"], v["stacking_weights"]))
    print(
        "shape race via RUST FFI (gamfit.adjudicate_atom_shape), "
        f"headline={v['headline']}, held-out stacking weights:"
    )
    for name, val in sorted(table.items(), key=lambda kv: -kv[1]):
        print(f"    {name:18s} weight={val:.4f}")
    print(
        f"  VERDICT: class={winner_class} reporting={reporting_winner} "
        f"(circular stacking margin = {margin:+.4f})"
    )

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
        "input_p": input_p,
        "fit_p": p,
        "pca_explained_variance": explained,
        "layer": args.layer,
        "reconstruction_r2": r2,
        "atom_topology": str(topology),
        "shape_winner_class": winner_class,
        "shape_reporting_winner": reporting_winner,
        "shape_table": table,
        "mixture_reporting_k": mixture_reporting_k,
        "circular_margin": margin,
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
        print("\n*** REAL-ACTIVATION WIN: the recovered atom is a predictively adjudicated S^1 "
              "circle BEATING the cluster null on real OLMo activations. ***")
    else:
        print(f"\n*** REAL-ACTIVATION FINDING: the recovered atom adjudicates as "
              f"class='{winner_class}' reporting='{reporting_winner}', "
              "not a smooth circle — the structure ladder reports the honest verdict "
              "(the wager loses measurably on this layer/concept, which is itself the result). ***")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
