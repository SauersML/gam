#!/usr/bin/env python3
"""Fit every featurizer on SynthSAEBench-style data and run the downstream battery.

This is the gamfit-consuming driver for ``bench/downstream_battery.py``. It fits
the manifold SAE (``gamfit.sae_manifold_fit``) and the flat baselines (L1 / TopK
/ BatchTopK, the PyTorch modules already in ``bench/synth_sae_compare.py``) on the
same SynthSAEBench-style train/test activations, marshals each fit's decoder
directions + codes + the planted factors into a :class:`FeaturizerCodes`, and runs
the external-validity battery on each. Results are written **append-per-config**
as JSONL so a crash loses nothing (the pattern established by
``experiments/real_manifold_sae/phase1_reml_real_revalidation.py``).

The battery metrics (single-concept detection F1, code-probing accuracy vs a
raw-activation ceiling, cosine direction recovery anchored to a random floor and
an activation ceiling, and gate-map smoothness) are defined in
``bench/downstream_battery.py``; this driver only supplies fitted featurizers.

Grid / smoothness: SynthSAEBench rows are i.i.d., so there is no *semantic* grid.
``--sequence-length L`` lays the test rows on a synthetic 1-D token-position axis
in groups of ``L`` purely to EXERCISE the sequence-smoothness hook (the same hook
an LLM bench uses with real token positions); the header records that this grid is
synthetic, not semantically meaningful. Off by default -- smoothness then reports
an honest ``unavailable`` marker rather than a fabricated number.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

import gamfit

from downstream_battery import FeaturizerCodes, run_battery
from synth_sae_bench_manifold import SynthConfig, SynthSAEBenchData
from synth_sae_compare import (
    BatchTopKSAE,
    L1SAE,
    TopKSAE,
    _basis_values,
)


def _torch_codes(
    cls: type[L1SAE],
    *,
    train_x: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    width: int,
    steps: int,
    batch_size: int,
    lr: float,
    l1: float,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Train a torch baseline; return (decoder_dirs, train_codes, test_codes)."""
    import torch

    torch.manual_seed(seed)
    x = torch.as_tensor(train_x, dtype=torch.float32)
    x_test = torch.as_tensor(test_x, dtype=torch.float32)
    if cls is BatchTopKSAE:
        model: L1SAE = BatchTopKSAE(train_x.shape[1], width, top_k, seed)
    elif cls is TopKSAE:
        model = TopKSAE(train_x.shape[1], width, top_k, seed)
    else:
        model = L1SAE(train_x.shape[1], width, seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for _step in range(steps):
        idx = rng.integers(0, train_x.shape[0], size=min(batch_size, train_x.shape[0]))
        batch = x[idx]
        recon, z = model(batch)
        loss = torch.mean((recon - batch) ** 2)
        if type(model) is L1SAE:
            loss = loss + float(l1) * torch.mean(torch.abs(z))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        train_codes = model.encode(x).detach().cpu().numpy()
        test_codes = model.encode(x_test).detach().cpu().numpy()
    decoder_dirs = model.decoder.weight.detach().cpu().numpy().T
    return decoder_dirs, train_codes, test_codes


def _manifold_codes(
    *,
    train_x: np.ndarray,
    test_x: np.ndarray,
    seed: int,
    atoms: int,
    top_k: int,
    max_iter: int,
    learning_rate: float,
    basis: str,
    atom_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, gamfit.ManifoldSAE]:
    """Fit the manifold SAE; return (decoder_dirs, train_codes, test_codes, fit).

    The extraction mirrors ``synth_sae_compare._score_manifold``: each atom's
    non-intercept basis rows with a live (nonzero-norm) decoder direction become a
    latent, whose activation is ``assignment_k * phi[:, row]``.
    """
    fit = gamfit.sae_manifold_fit(
        X=train_x,
        n_atoms=atoms,
        atom_topology=basis,
        d_atom=atom_dim,
        assignment="softmax",
        top_k=top_k,
        isometry_weight=0.0,
        ard_per_atom=False,
        sparsity_weight=0.01,
        smoothness_weight=0.01,
        n_iter=max_iter,
        learning_rate=learning_rate,
        random_state=seed,
    )
    payload = fit.converged_latents(test_x)
    dirs: list[np.ndarray] = []
    train_latents: list[np.ndarray] = []
    test_latents: list[np.ndarray] = []
    for k, block in enumerate(fit.decoder_blocks):
        n_harmonics = fit._n_harmonics[k] if k < len(fit._n_harmonics) else 1
        train_phi = _basis_values(fit.basis_specs[k], np.asarray(fit.coords[k], dtype=float), n_harmonics)
        test_phi = _basis_values(fit.basis_specs[k], np.asarray(payload["coords"][k], dtype=float), n_harmonics)
        train_assign = np.asarray(fit.assignments, dtype=float)[:, k]
        test_assign = np.asarray(payload["assignments"], dtype=float)[:, k]
        for row in range(1, min(block.shape[0], train_phi.shape[1])):
            direction = np.asarray(block[row], dtype=float)
            if np.linalg.norm(direction) <= 1e-10:
                continue
            dirs.append(direction)
            train_latents.append(train_assign * train_phi[:, row])
            test_latents.append(test_assign * test_phi[:, row])
    decoder_dirs = np.vstack(dirs) if dirs else np.zeros((0, train_x.shape[1]))
    train_codes = np.column_stack(train_latents) if train_latents else np.zeros((train_x.shape[0], 0))
    test_codes = np.column_stack(test_latents) if test_latents else np.zeros((test_x.shape[0], 0))
    return decoder_dirs, train_codes, test_codes, fit


def _grid(n_test: int, sequence_length: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Synthetic 1-D token-position grid: rows grouped into sequences of length L.

    Purely to exercise the sequence-smoothness hook (an LLM bench passes real
    token positions instead). Returns ``(positions, group)`` or ``(empty, None)``
    when disabled.
    """
    if sequence_length <= 1:
        return np.zeros((0, 0)), None
    positions = (np.arange(n_test) % sequence_length).reshape(-1, 1)
    group = (np.arange(n_test) // sequence_length).astype(int)
    return positions, group


def run_seed(args: argparse.Namespace, seed: int) -> list[dict[str, Any]]:
    synth = SynthSAEBenchData(
        SynthConfig(
            n_features=args.features,
            hidden_dim=args.hidden_dim,
            corr_rank=min(args.corr_rank, args.features),
            corr_scale=args.corr_scale,
            p_min=args.p_min,
            p_max=args.p_max,
            zipf_exponent=args.zipf_exponent,
            hierarchy_branching=args.hierarchy_branching,
            hierarchy_depth=args.hierarchy_depth,
            bias_norm=args.bias_norm,
            seed=seed,
        )
    )
    train_x, _train_coeff, train_fire = synth.sample(args.n_train, seed + 1)
    test_x, _test_coeff, test_fire = synth.sample(args.n_test, seed + 2)
    planted = synth.dictionary
    positions, group = _grid(test_x.shape[0], args.sequence_length)

    featurizers: list[FeaturizerCodes] = []

    for name, cls in (("l1_sae", L1SAE), ("topk_sae", TopKSAE), ("batchtopk_sae", BatchTopKSAE)):
        t0 = time.perf_counter()
        decoder_dirs, tr_codes, te_codes = _torch_codes(
            cls,
            train_x=train_x,
            test_x=test_x,
            seed=seed,
            width=args.width,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            l1=args.l1,
            top_k=args.top_k,
        )
        secs = time.perf_counter() - t0
        featurizers.append(
            (
                FeaturizerCodes(
                    name=name,
                    decoder_dirs=decoder_dirs,
                    train_codes=tr_codes,
                    test_codes=te_codes,
                    train_activations=train_x,
                    test_activations=test_x,
                    train_factors=train_fire,
                    test_factors=test_fire,
                    planted_dirs=planted,
                    positions=positions,
                    group=group,
                ),
                secs,
            )
        )

    if args.include_manifold:
        manifold_atoms = args.manifold_atoms
        if manifold_atoms is None:
            per_atom = 2 * max(1, args.manifold_atom_dim) if args.manifold_basis == "periodic" else 1
            manifold_atoms = max(1, int(np.ceil(args.width / per_atom)))
        t0 = time.perf_counter()
        decoder_dirs, tr_codes, te_codes, _fit = _manifold_codes(
            train_x=train_x,
            test_x=test_x,
            seed=seed,
            atoms=manifold_atoms,
            top_k=min(args.manifold_top_k, manifold_atoms),
            max_iter=args.manifold_max_iter,
            learning_rate=args.manifold_lr,
            basis=args.manifold_basis,
            atom_dim=args.manifold_atom_dim,
        )
        secs = time.perf_counter() - t0
        featurizers.append(
            (
                FeaturizerCodes(
                    name="gamfit_manifold_sae",
                    decoder_dirs=decoder_dirs,
                    train_codes=tr_codes,
                    test_codes=te_codes,
                    train_activations=train_x,
                    test_activations=test_x,
                    train_factors=train_fire,
                    test_factors=test_fire,
                    planted_dirs=planted,
                    positions=positions,
                    group=group,
                ),
                secs,
            )
        )

    reports: list[dict[str, Any]] = []
    for fc, secs in featurizers:
        report = run_battery(fc, seed=seed)
        report["seed"] = int(seed)
        report["fit_seconds"] = float(secs)
        report["n_features"] = int(args.features)
        report["hidden_dim"] = int(args.hidden_dim)
        report["n_train"] = int(train_x.shape[0])
        report["n_test"] = int(test_x.shape[0])
        report["grid"] = (
            "synthetic-sequence-position"
            if positions is not None and np.asarray(positions).size > 0
            else "none"
        )
        reports.append(report)
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=6)
    parser.add_argument("--n-train", type=int, default=256)
    parser.add_argument("--n-test", type=int, default=96)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--l1", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--include-manifold", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--manifold-atoms", type=int, default=None)
    parser.add_argument("--manifold-top-k", type=int, default=1)
    parser.add_argument("--manifold-max-iter", type=int, default=2)
    parser.add_argument("--manifold-lr", type=float, default=0.04)
    parser.add_argument("--manifold-basis", default="periodic")
    parser.add_argument("--manifold-atom-dim", type=int, default=1)
    parser.add_argument("--corr-rank", type=int, default=2)
    parser.add_argument("--corr-scale", type=float, default=0.1)
    parser.add_argument("--p-min", type=float, default=5e-4)
    parser.add_argument("--p-max", type=float, default=0.4)
    parser.add_argument("--zipf-exponent", type=float, default=0.5)
    parser.add_argument("--hierarchy-branching", type=int, default=4)
    parser.add_argument("--hierarchy-depth", type=int, default=1)
    parser.add_argument("--bias-norm", type=float, default=10.0)
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=0,
        help="If >1, lay test rows on a synthetic 1-D token-position grid in "
        "groups of this length to exercise the sequence-smoothness hook.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("downstream_battery_results.jsonl"),
        help="Append-per-config JSONL output path.",
    )
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        reports = run_seed(args, int(seed))
        with args.out.open("a") as fh:
            for report in reports:
                fh.write(json.dumps(report, sort_keys=True) + "\n")
        for report in reports:
            det = report["single_concept_detection_f1"]["mean_f1"]
            rec = report["cosine_probe_recovery"]["recovery"]
            print(
                f"[downstream] seed={seed} {report['featurizer']}: "
                f"detection_f1={det:.3f} cosine_recovery={rec:.3f}",
                flush=True,
            )
    print("[downstream] ALL CONFIGS DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
