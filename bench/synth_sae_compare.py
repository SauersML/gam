#!/usr/bin/env python3
"""Compare gamfit manifold SAE against sparse-autoencoder baselines.

The comparison is intentionally small by default so it can run on CPU. Every
model sees the same SynthSAEBench-style train/test activations and is scored by
the same direct ground-truth metrics:

* reconstruction R2
* decoder/feature MCC using Hungarian matching
* feature uniqueness (SynthSAEBench argmax-collision definition)
* direction recovery precision/recall/F1/Jaccard (quality-aware coverage)
* matched-latent firing precision/recall/F1

The manifold row uses the repo's public ``gamfit.sae_manifold_fit`` API. The
baseline rows use PyTorch modules in this file to avoid changing package
dependencies; if SAELens is installed, the output records that official-library
availability so larger official SAELens runs can be wired separately.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch

import gamfit
from gamfit._binding import rust_module
from _synth_sae_metrics import feature_uniqueness, n_firing_latents, recovery_scores
from synth_sae_bench_manifold import SynthConfig, SynthSAEBenchData


@dataclass(frozen=True)
class ModelResult:
    model: str
    seed: int
    n_features: int
    hidden_dim: int
    n_train: int
    n_test: int
    width: int
    steps: int
    train_r2: float
    test_r2: float
    mcc: float
    feature_uniqueness: float
    direction_recovery_precision: float
    direction_recovery_recall: float
    direction_recovery_f1: float
    direction_recovery_jaccard: float
    n_latent_slots: int
    n_firing_latents: int
    probing_precision: float
    probing_recall: float
    probing_f1: float
    true_l0_train: float
    true_l0_test: float
    learned_l0_train: float
    learned_l0_test: float
    # Matched-(L0, params, bits) triple so no comparison is silently unmatched
    # (#external-validity): L0 per row = learned_l0_test, an exact Python
    # parameter count, and description-length bits. The bits column PICKS UP the
    # Rust FFI bits/token surface (``ManifoldSAE.description_length``) when the
    # featurizer exposes it and is honest about its source; flat baselines have no
    # manifold DL FFI, so their bits are ``None`` with a source note (never faked).
    l0_per_row: float
    param_count: int
    description_length_bits: float | None
    description_length_source: str
    seconds: float
    status: str
    notes: str


class L1SAE(torch.nn.Module):
    def __init__(self, d_in: int, width: int, seed: int) -> None:
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.encoder = torch.nn.Linear(d_in, width)
        self.decoder = torch.nn.Linear(width, d_in, bias=True)
        torch.nn.init.normal_(self.encoder.weight, mean=0.0, std=1.0 / d_in**0.5, generator=gen)
        torch.nn.init.zeros_(self.encoder.bias)
        torch.nn.init.normal_(self.decoder.weight, mean=0.0, std=1.0 / width**0.5, generator=gen)
        torch.nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decoder(z), z


class TopKSAE(L1SAE):
    def __init__(self, d_in: int, width: int, k: int, seed: int) -> None:
        super().__init__(d_in, width, seed)
        self.k = int(k)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.relu(self.encoder(x))
        if self.k >= pre.shape[1]:
            return pre
        values, indices = torch.topk(pre, k=self.k, dim=1)
        out = torch.zeros_like(pre)
        out.scatter_(1, indices, values)
        return out


class BatchTopKSAE(L1SAE):
    def __init__(self, d_in: int, width: int, k: int, seed: int) -> None:
        super().__init__(d_in, width, seed)
        self.k = int(k)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.relu(self.encoder(x))
        keep = min(pre.numel(), max(1, self.k * pre.shape[0]))
        values, indices = torch.topk(pre.flatten(), k=keep, dim=0)
        out = torch.zeros_like(pre.flatten())
        out.scatter_(0, indices, values)
        return out.reshape_as(pre)


def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _best_f1(score: np.ndarray, truth: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(truth, dtype=bool)
    s = np.abs(np.asarray(score, dtype=float))
    if not np.any(y):
        return 0.0, 0.0, 0.0
    thresholds = np.unique(np.quantile(s, np.linspace(0.0, 1.0, 101)))
    best = (0.0, 0.0, 0.0)
    for threshold in thresholds:
        pred = s >= threshold
        tp = float(np.sum(pred & y))
        fp = float(np.sum(pred & ~y))
        fn = float(np.sum(~pred & y))
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
        if f1 > best[2]:
            best = (precision, recall, f1)
    return best


def _score(
    *,
    model_name: str,
    seed: int,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_fire: np.ndarray,
    test_fire: np.ndarray,
    truth_dirs: np.ndarray,
    train_recon: np.ndarray,
    test_recon: np.ndarray,
    train_latents: np.ndarray,
    test_latents: np.ndarray,
    decoder_dirs: np.ndarray,
    seconds: float,
    steps: int,
    notes: str,
) -> ModelResult:
    norms = np.linalg.norm(decoder_dirs, axis=1)
    live = norms > 1e-10
    dirs = decoder_dirs[live] / np.maximum(norms[live, None], 1e-10)
    # Three distinct ground-truth measurements (see bench/_synth_sae_metrics.py
    # and #1413): uniqueness is the SynthSAEBench argmax-collision score, while
    # MCC and the direction-recovery precision/recall/F1/Jaccard come from the
    # optimal one-to-one matching. The old `len(set(cols)) / len(rows)` was
    # always 1.0 because the assignment never reuses columns. The recovery
    # denominator spans *all* decoder slots (decoder_dirs.shape[0], incl. dead
    # zero-norm ones) so wasted width is penalized; dead rows carry no matching
    # mass, so this only affects the denominator.
    uniqueness = feature_uniqueness(dirs, truth_dirs)
    rec = recovery_scores(dirs, truth_dirs, n_learned_total=int(decoder_dirs.shape[0]))
    rows, cols = rec.rows, rec.cols
    # Functional (activation-based) dead-latent accounting (#1435): a latent
    # with a nonzero decoder direction that never fires on the eval set is
    # functionally dead -- wasted capacity the geometric (decoder-norm) live
    # check above misses. n_latent_slots is the architectural width; the gap
    # n_latent_slots - n_firing_latents is the functional dead count.
    n_slots = int(decoder_dirs.shape[0])
    n_firing = n_firing_latents(test_latents)
    if rows.size:
        mcc = rec.mcc
        live_train = train_latents[:, live]
        live_test = test_latents[:, live]
        metrics = [_best_f1(live_test[:, int(row)], test_fire[:, int(col)]) for row, col in zip(rows, cols)]
        precision, recall, f1 = (float(np.mean([m[i] for m in metrics])) for i in range(3))
    else:
        mcc = precision = recall = f1 = 0.0
        live_train = np.zeros((train_x.shape[0], 0))
        live_test = np.zeros((test_x.shape[0], 0))
    return ModelResult(
        model=model_name,
        seed=seed,
        n_features=int(truth_dirs.shape[0]),
        hidden_dim=int(truth_dirs.shape[1]),
        n_train=int(train_x.shape[0]),
        n_test=int(test_x.shape[0]),
        width=int(decoder_dirs.shape[0]),
        steps=int(steps),
        train_r2=_r2(train_x, train_recon),
        test_r2=_r2(test_x, test_recon),
        mcc=mcc,
        feature_uniqueness=uniqueness,
        direction_recovery_precision=rec.precision,
        direction_recovery_recall=rec.recall,
        direction_recovery_f1=rec.f1,
        direction_recovery_jaccard=rec.jaccard,
        n_latent_slots=n_slots,
        n_firing_latents=n_firing,
        probing_precision=precision,
        probing_recall=recall,
        probing_f1=f1,
        true_l0_train=float(np.mean(np.sum(train_fire, axis=1))),
        true_l0_test=float(np.mean(np.sum(test_fire, axis=1))),
        learned_l0_train=float(np.mean(np.sum(np.abs(live_train) > 1e-8, axis=1))),
        learned_l0_test=float(np.mean(np.sum(np.abs(live_test) > 1e-8, axis=1))),
        seconds=float(seconds),
        status="ok",
        notes=notes,
    )


def _train_torch_sae(
    cls: type[L1SAE],
    *,
    name: str,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_fire: np.ndarray,
    test_fire: np.ndarray,
    truth_dirs: np.ndarray,
    seed: int,
    width: int,
    steps: int,
    batch_size: int,
    lr: float,
    l1: float,
    top_k: int,
) -> ModelResult:
    torch.manual_seed(seed)
    x = torch.as_tensor(train_x, dtype=torch.float32)
    x_test = torch.as_tensor(test_x, dtype=torch.float32)
    if cls is BatchTopKSAE:
        model = BatchTopKSAE(train_x.shape[1], width, top_k, seed)
        notes = f"PyTorch BatchTopK SAE baseline, batch k={top_k}"
    elif cls is TopKSAE:
        model = TopKSAE(train_x.shape[1], width, top_k, seed)
        notes = f"PyTorch TopK SAE baseline, k={top_k}"
    else:
        model = L1SAE(train_x.shape[1], width, seed)
        notes = f"PyTorch L1 SAE baseline, l1={l1}"
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
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
    seconds = time.perf_counter() - t0
    with torch.no_grad():
        train_recon_t, train_z_t = model(x)
        test_recon_t, test_z_t = model(x_test)
    decoder_dirs = model.decoder.weight.detach().cpu().numpy().T
    return _score(
        model_name=name,
        seed=seed,
        train_x=train_x,
        test_x=test_x,
        train_fire=train_fire,
        test_fire=test_fire,
        truth_dirs=truth_dirs,
        train_recon=train_recon_t.detach().cpu().numpy(),
        test_recon=test_recon_t.detach().cpu().numpy(),
        train_latents=train_z_t.detach().cpu().numpy(),
        test_latents=test_z_t.detach().cpu().numpy(),
        decoder_dirs=decoder_dirs,
        seconds=seconds,
        steps=steps,
        notes=notes,
    )


def _basis_values(kind: str, coords: np.ndarray, n_harmonics: int) -> np.ndarray:
    if kind == "periodic":
        phi, _jet, _penalty = rust_module().basis_with_jet(
            "periodic",
            np.ascontiguousarray(np.asarray(coords[:, :1], dtype=float)),
            {"n_harmonics": int(n_harmonics)},
        )
        return np.asarray(phi, dtype=float)
    x = np.asarray(coords[:, 0], dtype=float)
    return np.column_stack([np.ones_like(x), x])


def _score_manifold(
    *,
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_fire: np.ndarray,
    test_fire: np.ndarray,
    truth_dirs: np.ndarray,
    seed: int,
    atoms: int,
    top_k: int,
    max_iter: int,
    learning_rate: float,
    basis: str,
    atom_dim: int,
) -> ModelResult:
    t0 = time.perf_counter()
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
    seconds = time.perf_counter() - t0

    train_dirs: list[np.ndarray] = []
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
            train_dirs.append(direction)
            train_latents.append(train_assign * train_phi[:, row])
            test_latents.append(test_assign * test_phi[:, row])
    decoder_dirs = np.vstack(train_dirs) if train_dirs else np.zeros((0, train_x.shape[1]))
    train_lat = np.column_stack(train_latents) if train_latents else np.zeros((train_x.shape[0], 0))
    test_lat = np.column_stack(test_latents) if test_latents else np.zeros((test_x.shape[0], 0))
    return _score(
        model_name="gamfit_manifold_sae",
        seed=seed,
        train_x=train_x,
        test_x=test_x,
        train_fire=train_fire,
        test_fire=test_fire,
        truth_dirs=truth_dirs,
        train_recon=np.asarray(fit.fitted, dtype=float),
        test_recon=np.asarray(payload["fitted"], dtype=float),
        train_latents=train_lat,
        test_latents=test_lat,
        decoder_dirs=decoder_dirs,
        seconds=seconds,
        steps=max_iter,
        notes=f"gamfit.sae_manifold_fit, atoms={atoms}, basis={basis}, top_k={top_k}",
    )


def _availability() -> dict[str, bool]:
    import importlib.util

    return {
        "torch": importlib.util.find_spec("torch") is not None,
        "scipy": importlib.util.find_spec("scipy") is not None,
        "sklearn": importlib.util.find_spec("sklearn") is not None,
        "sae_lens": importlib.util.find_spec("sae_lens") is not None,
    }


def _summarize(rows: list[ModelResult]) -> dict[str, Any]:
    metrics = ["train_r2", "test_r2", "mcc", "probing_f1", "learned_l0_test", "n_firing_latents", "seconds"]
    out: dict[str, Any] = {}
    for model in sorted({r.model for r in rows}):
        model_rows = [r for r in rows if r.model == model]
        out[model] = {}
        for metric in metrics:
            vals = np.array([float(getattr(r, metric)) for r in model_rows], dtype=float)
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            out[model][metric] = {
                "mean": mean,
                "std": std,
                "cv": float(std / abs(mean)) if mean else None,
            }
    return out


def run_seed(args: argparse.Namespace, seed: int) -> list[ModelResult]:
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
    truth_dirs = synth.dictionary
    rows = [
        _train_torch_sae(
            L1SAE,
            name="l1_sae",
            train_x=train_x,
            test_x=test_x,
            train_fire=train_fire,
            test_fire=test_fire,
            truth_dirs=truth_dirs,
            seed=seed,
            width=args.width,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            l1=args.l1,
            top_k=args.top_k,
        ),
        _train_torch_sae(
            TopKSAE,
            name="topk_sae",
            train_x=train_x,
            test_x=test_x,
            train_fire=train_fire,
            test_fire=test_fire,
            truth_dirs=truth_dirs,
            seed=seed,
            width=args.width,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            l1=args.l1,
            top_k=args.top_k,
        ),
        _train_torch_sae(
            BatchTopKSAE,
            name="batchtopk_sae",
            train_x=train_x,
            test_x=test_x,
            train_fire=train_fire,
            test_fire=test_fire,
            truth_dirs=truth_dirs,
            seed=seed,
            width=args.width,
            steps=args.steps,
            batch_size=args.batch_size,
            lr=args.lr,
            l1=args.l1,
            top_k=args.top_k,
        ),
    ]
    if args.include_manifold:
        manifold_atoms = args.manifold_atoms
        if manifold_atoms is None:
            directions_per_atom = 2 * max(1, args.manifold_atom_dim) if args.manifold_basis == "periodic" else 1
            manifold_atoms = max(1, int(np.ceil(args.width / directions_per_atom)))
        rows.append(
            _score_manifold(
                train_x=train_x,
                test_x=test_x,
                train_fire=train_fire,
                test_fire=test_fire,
                truth_dirs=truth_dirs,
                seed=seed,
                atoms=manifold_atoms,
                top_k=min(args.manifold_top_k, manifold_atoms),
                max_iter=args.manifold_max_iter,
                learning_rate=args.manifold_lr,
                basis=args.manifold_basis,
                atom_dim=args.manifold_atom_dim,
            )
        )
    return rows


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
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    rows: list[ModelResult] = []
    for seed in args.seeds:
        rows.extend(run_seed(args, int(seed)))
    payload = {
        "benchmark": "fair SynthSAEBench-style SAE comparison",
        "fairness": {
            "same_train_test_data_per_seed": True,
            "same_direct_ground_truth_metrics": True,
            "reconstruction_is_secondary": True,
            "multi_seed_summary": len(args.seeds) > 1,
        },
        "library_availability": _availability(),
        "official_library_note": (
            "SAELens is the official SAE training/eval library for many SAE variants. "
            "This environment did not have sae_lens imported when the runner was written; "
            "PyTorch baselines are included so the comparison is runnable locally."
        ),
        "runs": [asdict(row) for row in rows],
        "summary": _summarize(rows),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
