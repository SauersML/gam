#!/usr/bin/env python3
"""Chart-configuration probe for gam#2234 E1 (harvest once, sweep cheaply).

The first real-model E1 run (Qwen3.5-4B-Base L16, 2026-07-31) minted a fit — the
historical `1e12` wall is gone — but the fitted chart was empty: `fit_ev`
0.0152 and weekday circular `R2` **0.0000**, so every steering delta was ~1e-9
and both arms measured zero. The cause is not the solver: a 70-row cloud built
from 10 prompt templates × 7 weekdays has its leading singular directions
spanned by TEMPLATE identity, and a PCA-64 chart of 70 rows is then essentially
noise. The calendar circle is a rank-2 object hiding under that.

This script separates the two costs so the chart config can be chosen on
evidence rather than guessed at 10 minutes per guess:

  `harvest`  one GPU pass; dumps the ambient activation cloud + labels + template
             indices to an `.npz` (no fit).
  `sweep`    CPU only; for each (per-template-centering, PCA dim) cell, fits
             `gamfit.sae.sae_manifold_fit` exactly as the E1 harness does and reports
             the fit EV together with how much of the *known* structure the
             fitted coordinate recovers (circular R² against the day-of-week
             phase for the cyclic ladder, linear R² against rank for the ordinal
             one). That recovery number — not EV — is what decides whether E1 can
             measure anything at all.

    python3 experiments/steering_e1/probe_chart_2234.py harvest \
        --structure weekday --model Qwen/Qwen3.5-4B-Base --layer-index 16 \
        --out cloud_cyclic.npz
    python3 experiments/steering_e1/probe_chart_2234.py sweep \
        --npz cloud_weekday.npz --structure weekday --pca-dims 2,3,4,6,8,12
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

TAU = 2.0 * math.pi


def _sibling(name: str):
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load sibling harness at {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def harvest(args) -> int:
    E1 = _sibling("run_e1")
    structure = E1.STRUCTURES[args.structure]
    model, tok = E1.load_model_and_tokenizer(args.model, args.cache_dir, args.dtype)
    layers = E1.resolve_layers(model)
    layer = layers[args.layer_index]

    templates, labels = structure.fit_templates(), structure.labels
    label_ids = E1.label_token_ids(tok, labels, args.candidate_prefix)

    rows, label_idx, template_idx = [], [], []
    for ti, template in enumerate(templates):
        for li, label in enumerate(labels):
            if args.capture_at == "label":
                ids, pos = E1.build_label_prompt(tok, template, label_ids[li])
                act, _ = E1.run_clean_at(model, layer, ids, pos)
            else:
                act, _ = E1.run_clean(model, tok, layer, template.format(label=label))
            rows.append(act.numpy().astype(np.float64))
            label_idx.append(li)
            template_idx.append(ti)
    X = np.ascontiguousarray(np.stack(rows))
    np.savez(
        args.out, X=X,
        label_index=np.asarray(label_idx), template_index=np.asarray(template_idx),
        model=np.asarray(args.model), layer_index=np.asarray(args.layer_index),
        structure=np.asarray(args.structure),
    )
    print(f"harvested {X.shape} -> {args.out}", flush=True)
    return 0


def structure_recovery(coord: np.ndarray, label_idx: np.ndarray, cyclic: bool,
                       n_labels: int) -> float:
    """How much of the KNOWN generator the fitted 1-D coordinate recovers.

    Delegates to `run_e1` so the probe and the harness can never disagree about
    what "recovery" means. The chart PERIOD is a property of the fitted object,
    not of this script, so both the period-one and the period-2*pi conventions
    are scored and the better is kept: reading a good chart in the wrong unit
    winds the phase 2*pi times too fast and returns R^2 == 0, which is
    indistinguishable from "the fit found nothing".
    """
    E1 = _sibling("run_e1")
    if cyclic:
        return max(E1.circular_recovery(coord, label_idx, n_labels, p)[0]
                   for p in (1.0, TAU))
    return E1.linear_recovery(coord, label_idx)[0]


def reference(args) -> int:
    """FIT-FREE geometry report — is the generator in the cloud at all?

    Pure numpy, seconds, no `gamfit`. A fitted chart that scores below these
    numbers is a deficit of the fitter; a cloud whose numbers are all near zero
    cannot support ANY steering measurement, fitted or not, and the honest move
    is to change the capture site rather than the solver.

    Reported per centering:
      * `label_variance_fraction` — between-label variance / total variance. The
        activation at the label position depends only on the CONTEXT HEAD (the
        model is causal), so head identity competes directly with label identity
        for the leading singular directions; this says who wins.
      * `plane_r2` — recovery from the top-2 plane's angle (cyclic) or PC1
        (ordinal), the chart E1 would actually fit in.
      * `best_pair_r2` — the best recovery over ALL pairs drawn from the top
        `--pc-scan` PCs. If this is high while `plane_r2` is low, the generator
        is present but not leading, and the chart needs a different subspace,
        not a different solver.
      * `centroid_r2` — the same read-off on the per-label CENTROIDS, i.e. with
        all within-label (context) variance averaged away. This is the upper
        bound any chart of this cloud can reach.
    """
    E1 = _sibling("run_e1")
    structure = E1.STRUCTURES[args.structure]
    data = np.load(args.npz, allow_pickle=False)
    X0, label_idx, template_idx = data["X"], data["label_index"], data["template_index"]
    if args.max_heads:
        keep = template_idx < args.max_heads
        X0, label_idx, template_idx = X0[keep], label_idx[keep], template_idx[keep]
    n_labels = int(label_idx.max()) + 1
    print(f"cloud {X0.shape} over {len(np.unique(template_idx))} heads", flush=True)

    out = []
    for center in (False, True):
        X = X0.copy()
        if center:
            for ti in np.unique(template_idx):
                rows = np.flatnonzero(template_idx == ti)
                X[rows] -= X[rows].mean(0, keepdims=True)
        Xc = X - X.mean(0, keepdims=True)
        total = float(np.sum(Xc ** 2))
        centroids = np.stack([Xc[label_idx == li].mean(0) for li in range(n_labels)])
        between = float(sum(int(np.sum(label_idx == li)) * float(np.sum(centroids[li] ** 2))
                            for li in range(n_labels)))
        _, svals, vt = np.linalg.svd(Xc, full_matrices=False)
        scores = Xc @ vt[: args.pc_scan].T

        def read(vec_a, vec_b, idx):
            if structure.cyclic:
                return structure_recovery(np.arctan2(vec_b, vec_a), idx, True, n_labels)
            return structure_recovery(vec_a, idx, False, n_labels)

        plane = read(scores[:, 0], scores[:, 1], label_idx)
        best_pair, best_ij = -1.0, (0, 1)
        for i in range(min(args.pc_scan, scores.shape[1])):
            for j in range(i + 1, min(args.pc_scan, scores.shape[1])):
                r2 = read(scores[:, i], scores[:, j], label_idx)
                if r2 > best_pair:
                    best_pair, best_ij = r2, (i, j)

        cc = centroids - centroids.mean(0, keepdims=True)
        _, _, cvt = np.linalg.svd(cc, full_matrices=False)
        cs = cc @ cvt[:2].T
        centroid_r2 = read(cs[:, 0], cs[:, 1], np.arange(n_labels))

        row = {
            "per_template_center": center,
            "label_variance_fraction": between / max(total, 1e-30),
            "plane_r2": float(plane),
            "best_pair_r2": float(best_pair),
            "best_pair": list(best_ij),
            "centroid_r2": float(centroid_r2),
            "top_singular_values": [float(v) for v in svals[:6]],
        }
        out.append(row)
        print(f"center={int(center)} label_var_frac={row['label_variance_fraction']:.4f} "
              f"plane_r2={plane:.4f} best_pair_r2={best_pair:.4f} pcs={best_ij} "
              f"centroid_r2={centroid_r2:.4f} svals={[round(v, 1) for v in svals[:6]]}",
              flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


def planted(args) -> int:
    """The control that decides WHOSE deficit a collapsed chart is.

    The real-activation scan found `sae_manifold_fit` minting a certified fit
    whose circle coordinate is degenerate on a cloud whose top-2 plane is a
    clean 7-point ring. Two explanations survive that on its own: the fitter
    cannot recover this geometry at all, or something specific to real
    activations defeats it. This subcommand builds surrogates of increasing
    idealisation from the SAME cloud and fits each with the SAME call, so the
    boundary between the two can be located instead of argued:

      `real`      the measured chart, unchanged (the reference point);
      `surrogate` the measured per-label centroids, with within-label scatter
                  replaced by isotropic Gaussian noise of the measured size —
                  identical first- and second-order label geometry, no real
                  activation idiosyncrasy left;
      `ideal`     an exact circle of the measured radius with the measured
                  noise, embedded in the same chart dimension — the textbook
                  case, which `tests_steering_e4` already certifies at R2 1.0.

    A collapse that survives all the way to `ideal` is a plain fitter defect
    with a seconds-long reproducer. A collapse that appears only on `real` is a
    property of activation geometry and belongs in the finding, not in a bug.
    """
    import gamfit

    E1 = _sibling("run_e1")
    structure = E1.STRUCTURES[args.structure]
    data = np.load(args.npz, allow_pickle=False)
    X0, label_idx, template_idx = data["X"], data["label_index"], data["template_index"]
    if args.max_heads:
        keep = template_idx < args.max_heads
        X0, label_idx, template_idx = X0[keep], label_idx[keep], template_idx[keep]
    n_labels = int(label_idx.max()) + 1

    X = X0.copy()
    for ti in np.unique(template_idx):
        rows = np.flatnonzero(template_idx == ti)
        X[rows] -= X[rows].mean(0, keepdims=True)
    Xc = X - X.mean(0, keepdims=True)
    _, svals, vt = np.linalg.svd(Xc, full_matrices=False)
    r = int(min(args.pca_dim, vt.shape[0]))
    real = np.ascontiguousarray(Xc @ vt[:r].T)

    cent = np.stack([real[label_idx == li].mean(0) for li in range(n_labels)])
    resid = real - cent[label_idx]
    noise = float(np.sqrt(np.mean(resid ** 2)))
    radius = float(np.mean(np.linalg.norm(cent[:, :2], axis=1)))
    rng = np.random.default_rng(args.seed)

    surrogate = cent[label_idx] + rng.normal(0.0, noise, size=real.shape)

    ideal = np.zeros_like(real)
    if structure.cyclic:
        theta = TAU * label_idx.astype(np.float64) / n_labels
        ideal[:, 0] = radius * np.cos(theta)
        ideal[:, 1] = radius * np.sin(theta)
    else:
        span = float(cent[:, 0].max() - cent[:, 0].min())
        ideal[:, 0] = span * (label_idx.astype(np.float64) / max(n_labels - 1, 1) - 0.5)
    ideal = ideal + rng.normal(0.0, noise, size=real.shape)

    print(f"chart {real.shape} noise_rms={noise:.4g} mean_centroid_radius={radius:.4g}",
          flush=True)
    out = []
    for name, Xr in (("real", real), ("surrogate", surrogate), ("ideal", ideal)):
        if structure.cyclic:
            v = Xr[:, :2] - Xr[:, :2].mean(0, keepdims=True)
            free = structure_recovery(
                np.arctan2(v[:, 1], v[:, 0]), label_idx, True, n_labels)
        else:
            free = structure_recovery(
                Xr[:, 0] - Xr[:, 0].mean(), label_idx, False, n_labels)
        try:
            fit = gamfit.sae.sae_manifold_fit(
                np.ascontiguousarray(Xr), K=1, d_atom=1, atom_topology=structure.topology,
                assignment="softmax", n_iter=args.n_iter, random_state=args.seed)
            ev = float(1.0 - np.sum((Xr - np.asarray(fit.fitted)) ** 2)
                       / max(np.sum((Xr - Xr.mean(0)) ** 2), 1e-30))
            c = np.asarray(fit.coords[0], dtype=float)
            coord = c[:, 0] if c.ndim == 2 else c
            rec = structure_recovery(coord, label_idx, structure.cyclic, n_labels)
            status = (f"ok coord_std={coord.std():.4g} "
                      f"distinct={int(np.unique(np.round(coord, 6)).size)}/{coord.size}")
        except Exception as error:
            ev, rec = float("nan"), float("nan")
            status = f"{type(error).__name__}: {error}"
        row = {"arm": name, "pca_dim": r, "fit_free_r2": float(free), "fit_ev": ev,
               "structure_recovery_r2": rec, "noise_rms": noise, "status": status[:220]}
        out.append(row)
        print(f"arm={name:9s} fit_free_r2={free:.4f} fit_ev={ev:.4f} "
              f"recovery_r2={rec:.4f} {row['status']}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    return 0


def sweep(args) -> int:
    import gamfit

    data = np.load(args.npz, allow_pickle=False)
    X0, label_idx, template_idx = data["X"], data["label_index"], data["template_index"]
    if args.max_heads:
        keep = template_idx < args.max_heads
        X0, label_idx, template_idx = X0[keep], label_idx[keep], template_idx[keep]
    E1 = _sibling("run_e1")
    structure = E1.STRUCTURES[args.structure]
    n_labels = int(label_idx.max()) + 1
    topology = structure.topology
    dims = [int(v) for v in args.pca_dims.split(",") if v.strip()]
    k_atoms = [int(v) for v in args.k_atoms.split(",") if v.strip()]
    centerings = [bool(int(v)) for v in args.centerings.split(",") if v.strip()]

    results = []
    for center in centerings:
        X = X0.copy()
        if center:
            for ti in np.unique(template_idx):
                rows = np.flatnonzero(template_idx == ti)
                X[rows] -= X[rows].mean(0, keepdims=True)
        mu = X.mean(0, keepdims=True)
        Xc = X - mu
        _, svals, vt = np.linalg.svd(Xc, full_matrices=False)

        # FIT-FREE REFERENCE. Before asking what the fitted chart recovers, ask
        # what is there to recover: the top-2 plane's angle (cyclic) or PC1
        # (ordinal), with no fit involved. A fitted chart that scores below this
        # is a deficit of the fitter, not of the data — and a reference that is
        # itself near zero means the cloud does not carry the generator and no
        # steering result of any kind is available from it.
        scores2 = Xc @ vt[:2].T
        if structure.cyclic:
            ref = structure_recovery(
                np.arctan2(scores2[:, 1], scores2[:, 0]), label_idx, True, n_labels)
        else:
            ref = structure_recovery(scores2[:, 0], label_idx, False, n_labels)
        results.append({
            "per_template_center": center, "pca_dim": 2, "k_atoms": 0,
            "pca_explained_variance": float((svals[:2] ** 2).sum()
                                            / max((svals ** 2).sum(), 1e-30)),
            "fit_ev": float("nan"), "structure_recovery_r2": float(ref),
            "status": "fit-free PCA reference",
        })
        print(f"center={int(center)} REFERENCE (fit-free top-2 plane) "
              f"recovery_r2={ref:.4f}", flush=True)

        for dim in dims:
            r = int(min(dim, vt.shape[0]))
            Xr = np.ascontiguousarray(Xc @ vt[:r].T)
            evr = float((svals[:r] ** 2).sum() / max((svals**2).sum(), 1e-30))
            for K in k_atoms:
                try:
                    fit = gamfit.sae.sae_manifold_fit(
                        Xr, K=K, d_atom=1, atom_topology=topology, assignment="softmax",
                        n_iter=args.n_iter, random_state=args.seed)
                    fit_ev = float(
                        1.0 - np.sum((Xr - np.asarray(fit.fitted)) ** 2)
                        / max(np.sum((Xr - Xr.mean(0)) ** 2), 1e-30))
                    recovery, coord_std = -1.0, 0.0
                    for k in range(K):
                        c = np.asarray(fit.coords[k], dtype=float)
                        coord = c[:, 0] if c.ndim == 2 else c
                        r2 = structure_recovery(coord, label_idx, structure.cyclic, n_labels)
                        if r2 > recovery:
                            recovery, coord_std = r2, float(coord.std())
                    # A CONSTANT fitted coordinate is the tell that the chart
                    # collapsed: recovery is then ~0 for a numerical reason, not
                    # because the coordinate disagrees with the generator.
                    status = f"ok coord_std={coord_std:.4g}"
                except Exception as error:  # a refusal is a datum, not a crash
                    fit_ev, recovery = float("nan"), float("nan")
                    status = f"{type(error).__name__}: {error}"
                row = {
                    "per_template_center": center, "pca_dim": r, "k_atoms": K,
                    "pca_explained_variance": evr, "fit_ev": fit_ev,
                    "structure_recovery_r2": recovery, "status": status[:200],
                }
                results.append(row)
                print(
                    f"center={int(center)} dim={r:3d} K={K} pca_evr={evr:.4f} "
                    f"fit_ev={fit_ev:.4f} recovery_r2={recovery:.4f} {row['status']}",
                    flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2) + "\n")
    ok = [r for r in results
          if np.isfinite(r["structure_recovery_r2"]) and r["k_atoms"] > 0]
    if ok:
        best = max(ok, key=lambda r: r["structure_recovery_r2"])
        print(f"BEST center={int(best['per_template_center'])} dim={best['pca_dim']} "
              f"K={best['k_atoms']} recovery_r2={best['structure_recovery_r2']:.4f} "
              f"fit_ev={best['fit_ev']:.4f}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    h = sub.add_parser("harvest")
    h.add_argument("--model", default="Qwen/Qwen3.5-4B-Base")
    h.add_argument("--cache-dir", default="")
    h.add_argument("--layer-index", type=int, default=16)
    h.add_argument("--structure", choices=("weekday", "ordinal"), default="weekday")
    h.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="fp32")
    h.add_argument("--capture-at", choices=("last", "label"), default="label",
                   help="'label' captures at the label token's own position (the representation); "
                        "'last' captures at the final token (a downstream trace, measured to carry "
                        "0.91%% of the cloud variance on Qwen3.5-4B-Base L16)")
    h.add_argument("--candidate-prefix", default=" ")
    h.add_argument("--out", required=True)
    h.set_defaults(func=harvest)

    f = sub.add_parser("reference")
    f.add_argument("--npz", required=True)
    f.add_argument("--structure", choices=("weekday", "ordinal"), default="weekday")
    f.add_argument("--pc-scan", type=int, default=12)
    f.add_argument("--max-heads", type=int, default=0,
                   help="restrict to the first N context heads — the same subset --max-heads "
                        "gives run_e1.py, so the fit-free bar is measured on the cloud the fit "
                        "actually sees")
    f.add_argument("--out", default="")
    f.set_defaults(func=reference)

    pl = sub.add_parser("planted")
    pl.add_argument("--npz", required=True)
    pl.add_argument("--structure", choices=("weekday", "ordinal"), default="weekday")
    pl.add_argument("--pca-dim", type=int, default=8)
    pl.add_argument("--max-heads", type=int, default=10)
    pl.add_argument("--n-iter", type=int, default=40)
    pl.add_argument("--seed", type=int, default=20260731)
    pl.add_argument("--out", default="")
    pl.set_defaults(func=planted)

    s = sub.add_parser("sweep")
    s.add_argument("--npz", required=True)
    s.add_argument("--structure", choices=("weekday", "ordinal"), default="weekday")
    s.add_argument("--pca-dims", default="2,3,4,6,8,12,16")
    s.add_argument("--k-atoms", default="1,2")
    s.add_argument("--max-heads", type=int, default=0,
                   help="restrict to the first N context heads, matching run_e1.py --max-heads")
    s.add_argument("--centerings", default="0,1",
                   help="which per-head-centering modes to sweep; measured 2026-07-31 on "
                        "Qwen3.5-4B-Base L16, the UNcentered cloud's best pair of the top 16 PCs "
                        "reaches only R2 0.56 while the centered top-2 plane reaches 0.93, so "
                        "'1' alone is usually the whole useful sweep")
    s.add_argument("--n-iter", type=int, default=60)
    s.add_argument("--seed", type=int, default=20260731)
    s.add_argument("--out", default="")
    s.set_defaults(func=sweep)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
