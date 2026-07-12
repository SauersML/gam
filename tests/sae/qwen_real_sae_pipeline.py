#!/usr/bin/env python3
"""#977 honest Stage-0(SGD SAE) -> Stage-1(gam adjudicates) pipeline on REAL Qwen acts.

This is the un-forced division of labor:

  Stage 0 (PyTorch, this file -> torch_sgd_sae.py):
      Train a REAL overcomplete L1/JumpReLU SGD sparse autoencoder on ~400k real
      Qwen3.5-35B-A3B layer-30 residual-post token activations. This is the wide
      monosemantic dictionary gam does NOT provide. Report EV / L0 / dead-frac.

  Stage 1a (feature geometry, this file):
      For the top-active features, build the feature co-activation graph (which
      features fire together across tokens) and cluster it into candidate groups
      (the #977 latent_seed co-activation idea). Each group is a bundle of
      features whose joint code lives in a low-D subspace.

  Stage 1b (gam adjudicates, gamfit.adjudicate_atom_shape):
      For each group, project the group's per-token SAE codes to a 2-D intrinsic
      coordinate (PCA of the group's code submatrix) and hand it to gam's
      cross-class adjudicator. gam auto-decides circular vs euclidean vs the
      free-mixture class on HELD-OUT predictive loglik. Discrete orders are
      selected inside each outer training fold. NO topology is forced.

  Full-pipeline controls (gamfit.run_shape_controlled_census):
      Repeat the complete Stage-0 -> Stage-1b callback from scratch on an
      independent per-dimension shuffle and covariance-matched Gaussian of the
      original activations, with the identical pipeline seed. The reported
      false-circle floor therefore includes SAE, grouping, projection, and
      adjudicator artifacts; it is not a post-PCA coordinate control.

  Honest report:
      Count groups that adjudicate as genuinely CURVED (circle beats the cluster
      null, positive circle margin) vs cluster/euclidean. A mostly-cluster result
      is a REAL finding — the wager partially failing is honest, and reported as
      such.

The activations are mmap'd; the SAE minibatches on GPU. Adjudication is CPU
(gam FFI). Everything is deterministic from --seed.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Data loading: mmap the f16 chunk .npy files, optionally subsample tokens.
# ---------------------------------------------------------------------------

def load_activations(act_dir: str, max_tokens: int | None, seed: int, log=print):
    files = sorted(glob.glob(os.path.join(act_dir, "*.npy")))
    if not files:
        raise FileNotFoundError(f"no .npy activation chunks under {act_dir}")
    log(f"found {len(files)} activation chunks under {act_dir}")
    mmaps = []
    total = 0
    d = None
    for f in files:
        m = np.load(f, mmap_mode="r")
        if m.ndim != 2:
            raise ValueError(f"expected [N, d] chunk, got {m.shape} in {f}")
        if d is None:
            d = m.shape[1]
        elif m.shape[1] != d:
            raise ValueError(f"dim mismatch {m.shape[1]} != {d} in {f}")
        mmaps.append(m)
        total += m.shape[0]
    log(f"total tokens available: {total}, dim={d}")

    rng = np.random.default_rng(seed)
    if max_tokens is not None and max_tokens < total:
        # Sample token indices across the concatenated stream, then gather.
        pick = np.sort(rng.choice(total, size=max_tokens, replace=False))
        out = np.empty((max_tokens, d), dtype=np.float32)
        offsets = np.cumsum([0] + [m.shape[0] for m in mmaps])
        wi = 0
        for ci, m in enumerate(mmaps):
            lo, hi = offsets[ci], offsets[ci + 1]
            sel = pick[(pick >= lo) & (pick < hi)] - lo
            if sel.size:
                out[wi : wi + sel.size] = np.asarray(m[sel], dtype=np.float32)
                wi += sel.size
        rng.shuffle(out, axis=0)
        log(f"subsampled {max_tokens} tokens (of {total})")
        return out, d
    # Load all (chunk-concatenate as float32).
    out = np.empty((total, d), dtype=np.float32)
    wi = 0
    for m in mmaps:
        out[wi : wi + m.shape[0]] = np.asarray(m, dtype=np.float32)
        wi += m.shape[0]
    rng.shuffle(out, axis=0)
    return out, d


# ---------------------------------------------------------------------------
# Stage 1a: feature co-activation clustering.
# ---------------------------------------------------------------------------

def feature_coactivation_groups(
    codes: np.ndarray,         # [N_eval, F] dense SAE codes on a held-out token sample
    act_count: np.ndarray,     # [F] full-data activation counts
    top_features: int,
    n_groups: int,
    min_group: int,
    seed: int,
    log=print,
):
    """Build candidate feature groups from co-activation correlation.

    1. Restrict to the `top_features` most-active (non-dead) features.
    2. Compute the feature-feature co-activation correlation on the binary
       firing pattern across tokens.
    3. Spectral-cluster the correlation graph into `n_groups`; keep groups with
       >= `min_group` features.
    Returns a list of feature-index arrays (into the full F dictionary).
    """
    order = np.argsort(-act_count)
    alive = order[act_count[order] > 0][:top_features]
    if alive.size < min_group * 2:
        log(f"WARNING: only {alive.size} live features; co-activation clustering thin")
    fire = (codes[:, alive] > 0).astype(np.float32)  # [N, M]
    # Center per-feature firing rate; correlation across tokens.
    fc = fire - fire.mean(axis=0, keepdims=True)
    denom = np.sqrt((fc * fc).sum(axis=0))
    denom[denom == 0] = 1.0
    corr = (fc.T @ fc) / (denom[:, None] * denom[None, :])  # [M, M]
    np.fill_diagonal(corr, 1.0)

    # Spectral embedding of the affinity (|corr| as similarity), then k-means.
    aff = np.abs(corr)
    deg = aff.sum(axis=1)
    deg[deg == 0] = 1.0
    dinv = 1.0 / np.sqrt(deg)
    lap = np.eye(aff.shape[0]) - (dinv[:, None] * aff * dinv[None, :])
    # Smallest non-trivial eigenvectors as the embedding.
    vals, vecs = np.linalg.eigh(lap)
    emb = vecs[:, 1 : 1 + min(n_groups, vecs.shape[1] - 1)]
    labels = _kmeans(emb, n_groups, seed, log)

    groups = []
    for g in range(n_groups):
        members = alive[labels == g]
        if members.size >= min_group:
            groups.append(np.sort(members))
    log(f"co-activation clustering: {len(groups)} candidate groups "
        f"(>= {min_group} features) from {alive.size} live top features")
    return groups


def _kmeans(X: np.ndarray, k: int, seed: int, log, iters: int = 50):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)
    centers = X[rng.choice(n, size=k, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new = d2.argmin(axis=1)
        if np.array_equal(new, labels):
            break
        labels = new
        for c in range(k):
            m = labels == c
            if m.any():
                centers[c] = X[m].mean(axis=0)
    return labels


# ---------------------------------------------------------------------------
# Stage 1b: per-group 2-D coordinates + gam adjudication.
# ---------------------------------------------------------------------------

def _active_group_codes(codes, members, max_rows, seed):
    sub = codes[:, members]
    active_rows = np.flatnonzero((sub > 0).sum(axis=1) >= max(2, members.size // 4))
    if active_rows.size < 16:
        return None
    if active_rows.size > max_rows:
        rng = np.random.default_rng(seed)
        active_rows = np.sort(rng.choice(active_rows, size=max_rows, replace=False))
    return np.asarray(sub[active_rows], dtype=np.float64)


def _circle_projection_score(coords):
    """Fast label-free circle score used only on discovery rows."""
    x = np.asarray(coords, dtype=np.float64)
    design = np.column_stack((2.0 * x[:, 0], 2.0 * x[:, 1], np.ones(x.shape[0])))
    target = np.square(x).sum(axis=1)
    try:
        parameters, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3 or not np.isfinite(parameters).all():
        return None
    center = parameters[:2]
    relative = x - center
    radii = np.linalg.norm(relative, axis=1)
    mean_radius = float(radii.mean())
    if not math.isfinite(mean_radius) or mean_radius <= np.finfo(np.float64).tiny:
        return None
    radial_cv = float(radii.std() / mean_radius)
    unit = (relative[:, 0] + 1j * relative[:, 1]) / np.maximum(
        radii, np.finfo(np.float64).tiny
    )
    # Harmonics 1 and 2 reject one/two-ended line clouds but leave every
    # regularly spaced cyclic concept with at least three states unpenalized.
    angular_defect = float(max(abs(unit.mean()), abs(np.square(unit).mean())))
    score = radial_cv + angular_defect
    if not math.isfinite(score):
        return None
    return score, radial_cv, angular_defect, center.tolist()


def group_coords_2d(
    discovery_codes,
    evaluation_codes,
    members,
    decoder,
    max_rows,
    max_search_pcs,
    seed,
):
    """Select a variance-normalized PC pair on independent discovery rows.

    The decoder contribution factors as ``C @ D`` with group width ``g``.
    A thin QR of ``D.T`` reduces PCA from the ambient activation width to at
    most ``g`` dimensions, then every PC pair among the retained rank is scored
    in O(n g²). Pair selection sees discovery rows only. The returned chart is
    the selected projection of disjoint evaluation rows, scaled by discovery
    variances, so a low-variance ring hidden behind a dominant linear factor is
    reachable without leaking shape-evaluation rows into subspace selection.
    """
    discovery = _active_group_codes(discovery_codes, members, max_rows, seed)
    evaluation = _active_group_codes(
        evaluation_codes,
        members,
        max_rows,
        seed ^ 0x5EED_EA11,
    )
    if discovery is None or evaluation is None:
        return None
    decoder_block = np.asarray(decoder[members], dtype=np.float64)
    discovery_mean = discovery.mean(axis=0, keepdims=True)
    centered_discovery = discovery - discovery_mean
    centered_evaluation = evaluation - discovery_mean
    try:
        _, decoder_r = np.linalg.qr(decoder_block.T, mode="reduced")
        reduced_discovery = centered_discovery @ decoder_r.T
        reduced_evaluation = centered_evaluation @ decoder_r.T
        left, singular_values, right_t = np.linalg.svd(
            reduced_discovery,
            full_matrices=False,
        )
    except np.linalg.LinAlgError:
        return None
    if singular_values.size < 2 or singular_values[0] <= 0.0:
        return None
    rank_tolerance = (
        max(reduced_discovery.shape)
        * np.finfo(singular_values.dtype).eps
        * singular_values[0]
    )
    numerical_rank = int(np.count_nonzero(singular_values > rank_tolerance))
    retained = min(numerical_rank, max_search_pcs)
    if retained < 2:
        return None
    discovery_scores = left[:, :retained] * singular_values[:retained]
    evaluation_scores = reduced_evaluation @ right_t[:retained].T
    scales = np.sqrt(np.mean(np.square(discovery_scores), axis=0))
    if not np.isfinite(scales).all() or np.any(scales <= 0.0):
        return None

    best = None
    for first in range(retained - 1):
        for second in range(first + 1, retained):
            pair = discovery_scores[:, [first, second]] / scales[[first, second]]
            scored = _circle_projection_score(pair)
            if scored is None:
                continue
            candidate = (scored[0], first, second, scored)
            if best is None or candidate[:3] < best[:3]:
                best = candidate
    if best is None:
        return None
    _, first, second, score_parts = best
    coords = evaluation_scores[:, [first, second]] / scales[[first, second]]
    if not np.isfinite(coords).all():
        return None
    metadata = {
        "selected_pc_axes": [first, second],
        "searched_pcs": retained,
        "discovery_circle_score": score_parts[0],
        "discovery_radial_cv": score_parts[1],
        "discovery_angular_defect": score_parts[2],
        "discovery_circle_center": score_parts[3],
        "n_discovery_rows": int(discovery.shape[0]),
        "n_evaluation_rows": int(evaluation.shape[0]),
    }
    return np.ascontiguousarray(coords, dtype=np.float64), metadata


def adjudicate_groups(
    gamfit,
    discovery_codes,
    evaluation_codes,
    groups,
    decoder,
    max_rows,
    max_search_pcs,
    seed,
    log=print,
):
    verdicts = []
    mean_l0 = float(np.count_nonzero(evaluation_codes, axis=1).mean())
    for gi, members in enumerate(groups):
        projected = group_coords_2d(
            discovery_codes,
            evaluation_codes,
            members,
            decoder,
            max_rows,
            max_search_pcs,
            seed + gi,
        )
        if projected is None:
            log(f"  group {gi:3d} ({members.size} feats): no certifiable 2-D chart, skipped")
            continue
        coords, subspace = projected
        try:
            v = gamfit.adjudicate_atom_shape(
                coords,
                folds=5,
                seed=seed + 11 + gi,
                mean_l0=mean_l0,
                matched_controls=False,
            )
        except Exception as exc:  # noqa: BLE001
            log(f"  group {gi:3d}: adjudication error: {exc}")
            continue
        rec = {
            "group": gi,
            "n_features": int(members.size),
            "n_rows": int(coords.shape[0]),
            "winner_class": v["winner_class"],
            "reporting_winner": v["reporting_winner"],
            "circle_wins": bool(v["circle_wins"]),
            "circular_margin": float(v["circular_margin"]),
            "mixture_reporting_k": int(v["mixture_reporting_k"]),
            "ring_clusters_reporting_k": int(v["ring_clusters_reporting_k"]),
            "mixture_fold_selected_k": [int(k) for k in v["mixture_fold_selected_k"]],
            "ring_clusters_fold_selected_k": [
                int(k) for k in v["ring_clusters_fold_selected_k"]
            ],
            "mixture_fold_k_histogram": {
                int(k): int(count) for k, count in v["mixture_fold_k_histogram"].items()
            },
            "ring_clusters_fold_k_histogram": {
                int(k): int(count)
                for k, count in v["ring_clusters_fold_k_histogram"].items()
            },
            "headline": v["headline"],
            "stacking_weights": dict(zip(v["candidate_names"], v["stacking_weights"])),
            "dictionary_mean_l0": float(v["dictionary_mean_l0"]),
            "subspace_selection": subspace,
        }
        verdicts.append(rec)
        log(f"  group {gi:3d} ({members.size:3d} feats, {coords.shape[0]:4d} rows): "
            f"{v['winner_class']:16s} margin={v['circular_margin']:+.4f} "
            f"reporting={v['reporting_winner']} pcs={subspace['selected_pc_axes']}")
    return verdicts


def run_complete_census_pipeline(
    activations,
    pipeline_seed,
    *,
    args,
    gamfit,
    sae_config_type,
    train_sae,
    log=print,
):
    """Fresh-fit every stage for one observed/control activation matrix."""
    run_started = time.time()
    d_in = activations.shape[1]
    cfg = sae_config_type(
        d_in=d_in,
        dict_size=args.dict_size,
        l1_coeff=args.l1_coeff,
        activation=args.activation,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=pipeline_seed,
        device=args.device,
    )
    if args.eval_tokens < 32 or args.eval_tokens >= activations.shape[0]:
        raise ValueError(
            "--eval-tokens must satisfy 32 <= eval_tokens < activation rows so "
            "SAE training, group discovery, and shape evaluation are disjoint"
        )
    training_activations = activations[: -args.eval_tokens]
    geometry_activations = activations[-args.eval_tokens :]
    discovery_rows = args.eval_tokens // 2
    discovery_activations = geometry_activations[:discovery_rows]
    evaluation_activations = geometry_activations[discovery_rows:]
    log(
        f"=== Stage 0: training overcomplete {args.activation} SAE "
        f"d_in={d_in} dict={args.dict_size} on {training_activations.shape[0]} "
        f"training-only tokens ==="
    )
    model, sae_stats, act_count = train_sae(training_activations, cfg)

    import torch

    device = next(model.parameters()).device
    norm_scale = sae_stats["norm_scale"]

    def encode(matrix):
        with torch.no_grad():
            encoded = []
            for start in range(0, matrix.shape[0], args.batch_size):
                batch = torch.from_numpy(matrix[start : start + args.batch_size]).float().to(device)
                encoded.append(model.encode(batch * norm_scale).cpu().numpy())
        return np.concatenate(encoded, axis=0)

    discovery_codes = encode(discovery_activations)
    evaluation_codes = encode(evaluation_activations)
    log(
        "=== Stage 1a: co-activation clustering on "
        f"{discovery_codes.shape[0]} discovery-only tokens ==="
    )
    groups = feature_coactivation_groups(
        discovery_codes,
        act_count,
        args.top_features,
        args.n_groups,
        args.min_group,
        pipeline_seed,
        log,
    )
    log(f"=== Stage 1b: gam cross-class adjudication of {len(groups)} groups ===")
    decoder = model.W_dec.detach().cpu().numpy()
    verdicts = adjudicate_groups(
        gamfit,
        discovery_codes,
        evaluation_codes,
        groups,
        decoder,
        args.max_group_rows,
        args.subspace_search_pcs,
        pipeline_seed,
        log,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    from collections import Counter

    winner_breakdown = Counter(verdict["winner_class"] for verdict in verdicts)
    circular_wins = sum(bool(verdict["circle_wins"]) for verdict in verdicts)
    dictionary_mean_l0 = float(np.count_nonzero(evaluation_codes, axis=1).mean())
    return {
        "sae": sae_stats,
        "n_candidate_groups": len(groups),
        "n_attempted": len(groups),
        "n_adjudicated": len(verdicts),
        "n_circular_wins": circular_wins,
        "circular_win_rate": circular_wins / len(verdicts) if verdicts else None,
        "dictionary_mean_l0": dictionary_mean_l0,
        "winner_breakdown": dict(winner_breakdown),
        "verdicts": verdicts,
        "wall_seconds": time.time() - run_started,
    }


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

RUN_SPEC = """\
=== #977 real Qwen SGD-SAE -> gam-adjudication pipeline — cluster run spec ===

Data (already staged):
  /path/to/scratch/qwen_acts/activations/layer_30_residual_post/*.npy

Build the wheel on a COMPUTE node (never login), then run on an a100:

  # STEP 1 (build, if the olmo_venv wheel lacks adjudicate_atom_shape):
  sbatch -p <gpu-partition> --gres=gpu:a100:1 -t 90 --wrap '
    source /path/to/scratch/gam_env.sh
    export HF_HOME=/path/to/scratch/hf
    export TMPDIR=/path/to/scratch/tmp
    cd /path/to/scratch/gam
    maturin build --release -o dist && pip install --force-reinstall dist/*.whl'

  # STEP 2 (three fresh SAE fits + adjudication: observed + two controls):
  sbatch -p <gpu-partition> --gres=gpu:a100:1 -t 360 --wrap '
    source /path/to/scratch/olmo_venv/bin/activate
    export HF_HOME=/path/to/scratch/hf
    export TMPDIR=/path/to/scratch/tmp
    cd /path/to/scratch/gam
    python tests/sae/qwen_real_sae_pipeline.py \\
      --act-dir /path/to/scratch/qwen_acts/activations/layer_30_residual_post \\
      --dict-size 16384 --epochs 8 --max-tokens 400000 \\
      --out /path/to/scratch/qwen_sae_verdict.json'

Tune down for a quick smoke: --dict-size 4096 --epochs 2 --max-tokens 50000.
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--act-dir", type=str, default=None,
                    help="dir with layer_30_residual_post/*.npy chunks")
    ap.add_argument("--dict-size", type=int, default=16384)
    ap.add_argument("--l1-coeff", type=float, default=5e-3)
    ap.add_argument("--activation", choices=["relu", "jumprelu"], default="relu")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--max-tokens", type=int, default=400000)
    ap.add_argument("--eval-tokens", type=int, default=20000,
                    help="held-out tokens split equally between group discovery and shape evaluation")
    ap.add_argument("--top-features", type=int, default=2048)
    ap.add_argument("--n-groups", type=int, default=64)
    ap.add_argument("--min-group", type=int, default=4)
    ap.add_argument("--max-group-rows", type=int, default=2000)
    ap.add_argument(
        "--subspace-search-pcs",
        type=int,
        default=12,
        help="maximum decoder-contribution PCs searched for a discovery-only circular pair",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--control-seed", type=int, default=17)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--print-run-spec", action="store_true")
    args = ap.parse_args()

    if args.print_run_spec:
        print(RUN_SPEC)
        return 0
    if not args.act_dir:
        print("ERROR: --act-dir required (or --print-run-spec).", file=sys.stderr)
        return 2
    if args.subspace_search_pcs < 2 or args.max_group_rows < 16:
        print(
            "ERROR: --subspace-search-pcs must be >= 2 and --max-group-rows must be >= 16.",
            file=sys.stderr,
        )
        return 2

    t0 = time.time()
    try:
        import torch  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"BLOCKER: torch not importable ({exc}).", file=sys.stderr)
        return 3
    try:
        import gamfit
    except Exception as exc:  # noqa: BLE001
        print(f"BLOCKER: gamfit wheel not importable ({exc}).", file=sys.stderr)
        return 3
    required_api = {"adjudicate_atom_shape", "run_shape_controlled_census"}
    missing_api = sorted(name for name in required_api if not hasattr(gamfit, name))
    if missing_api:
        print(
            f"BLOCKER: gamfit wheel lacks {', '.join(missing_api)} — rebuild the current wheel.",
            file=sys.stderr,
        )
        return 3

    from torch_sgd_sae import SAEConfig, train_sae

    activations, _ = load_activations(args.act_dir, args.max_tokens, args.seed)

    def complete_pipeline(matrix, seed):
        return run_complete_census_pipeline(
            matrix,
            seed,
            args=args,
            gamfit=gamfit,
            sae_config_type=SAEConfig,
            train_sae=train_sae,
        )

    controlled = gamfit.run_shape_controlled_census(
        activations,
        complete_pipeline,
        control_seed=args.control_seed,
        pipeline_seed=args.seed,
    )
    runs = {
        "observed": controlled.observed,
        "per_dimension_shuffle": controlled.per_dimension_shuffle,
        "covariance_matched_gaussian": controlled.covariance_matched_gaussian,
    }
    control_runs = [
        controlled.per_dimension_shuffle,
        controlled.covariance_matched_gaussian,
    ]
    control_wins = sum(run["n_circular_wins"] for run in control_runs)
    control_adjudicated = sum(run["n_adjudicated"] for run in control_runs)
    summary = {
        "runs": runs,
        "control_false_circle_rate": (
            control_wins / control_adjudicated if control_adjudicated else None
        ),
        "control_n_circular_wins": control_wins,
        "control_n_adjudicated": control_adjudicated,
        "seed_provenance": {
            "pipeline_seed": controlled.pipeline_seed,
            "per_dimension_shuffle_seed": controlled.per_dimension_shuffle_seed,
            "covariance_matched_gaussian_seed": controlled.covariance_matched_gaussian_seed,
        },
        "wall_seconds": time.time() - t0,
        "config": vars(args),
    }
    print("\n=== HONEST FULL-PIPELINE VERDICT ===")
    for name, run in runs.items():
        sae = run["sae"]
        print(
            f"{name}: EV={sae['explained_variance']:.4f} mean_L0={sae['mean_l0']:.1f} "
            f"dead_frac={sae['dead_feature_fraction']:.3f}; "
            f"circular={run['n_circular_wins']}/{run['n_adjudicated']}; "
            f"winners={run['winner_breakdown']}"
        )
    print(
        "pooled matched-control false-circle floor: "
        f"{control_wins}/{control_adjudicated}"
        + (
            f" ({100.0 * control_wins / control_adjudicated:.1f}%)"
            if control_adjudicated
            else " (no adjudicated control groups)"
        )
    )
    if controlled.observed["n_adjudicated"] == 0:
        print("FINDING: no observed group could be adjudicated (degenerate group geometry).")

    if args.out:
        Path(args.out).write_text(json.dumps(summary, indent=2, default=str))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
