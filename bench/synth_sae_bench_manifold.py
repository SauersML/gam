#!/usr/bin/env python3
"""Benchmark gamfit's existing manifold SAE on SynthSAEBench-style data.

This is a compact CPU-friendly version of SynthSAEBench-16k. It keeps the
load-bearing benchmark properties from the paper: sparse ground-truth linear
features, Zipfian firing rates, low-rank firing correlation, hierarchy with
mutually exclusive siblings, superposition through an overcomplete dictionary,
and metrics that score feature recovery directly instead of only
reconstruction.

The SAE being benchmarked is the repo's public implementation:
``gamfit.sae_manifold_fit``. This file only supplies data generation and
ground-truth scoring.

Comparable-protocol convention (BSF toy)
----------------------------------------
To make our contribution-recovery number the SAME experiment as the BSF
("structure wins at every distortion") toy table, report per-manifold
contribution recovery ``m_i`` under an OPTIMAL atom<->factor assignment
(Hungarian matching of decoder directions to planted directions), NOT the
raw-activation reconstruction R2. The toy convention is:

* ``M`` manifolds  -- the number of planted ground-truth generative factors
  (``--features``; the BSF toy uses a small ``M``, e.g. 16).
* ``L0 = 4``       -- the fixed expected number of factors active per token; the
  BSF toy holds sparsity at 4. In this generator L0 is set by the firing
  probabilities (``--p-min`` / ``--p-max`` / ``--zipf-exponent``); the ACHIEVED
  L0 is reported as ``true_l0_test`` so any mismatch to the 4 target is visible
  and honest rather than assumed.
* ``N`` samples    -- matched to the toy (``--n-train`` / ``--n-test``).

Recommended matched invocation (documents the (M, L0, N) settings; achieved L0
is reported, not assumed)::

    python3 synth_sae_bench_manifold.py --features 16 --n-train 2048 \
        --n-test 1024 --atoms 16 --p-max 0.25 --zipf-exponent 0.0

Source: the BSF toy convention (M manifolds, L0=4, matched N). The
contribution-recovery metric itself is defined in
``bench/_synth_sae_metrics.py::contribution_recovery``.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

import gamfit
from gamfit._binding import rust_module
from _synth_sae_metrics import (
    contribution_recovery,
    feature_uniqueness,
    n_firing_latents,
    recovery_scores,
)


@dataclass(frozen=True)
class SynthConfig:
    n_features: int
    hidden_dim: int
    corr_rank: int
    corr_scale: float
    p_min: float
    p_max: float
    zipf_exponent: float
    hierarchy_branching: int
    hierarchy_depth: int
    bias_norm: float
    seed: int


@dataclass(frozen=True)
class BenchmarkMetrics:
    fit_api: str
    seed: int
    n_features: int
    hidden_dim: int
    n_train: int
    n_test: int
    atoms: int
    basis: str
    atom_dim: int
    assignment: str
    top_k: int | None
    true_l0_train: float
    true_l0_test: float
    learned_l0_train: float
    learned_l0_test: float
    train_r2: float
    test_r2: float
    mcc: float
    feature_uniqueness: float
    direction_recovery_precision: float
    direction_recovery_recall: float
    direction_recovery_f1: float
    direction_recovery_jaccard: float
    contribution_recovery_mean: float
    contribution_recovery_matched_factors: int
    n_latent_slots: int
    n_firing_latents: int
    probing_precision: float
    probing_recall: float
    probing_f1: float
    matching: str
    learned_directions: int
    fit_seconds: float
    score_seconds: float


class SynthSAEBenchData:
    def __init__(self, cfg: SynthConfig) -> None:
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        dictionary = rng.standard_normal((cfg.n_features, cfg.hidden_dim))
        dictionary /= np.maximum(np.linalg.norm(dictionary, axis=1, keepdims=True), 1e-12)
        self.dictionary = dictionary

        ranks = np.arange(1, cfg.n_features + 1, dtype=float)
        raw = ranks ** (-cfg.zipf_exponent)
        raw = (raw - raw.min()) / max(float(raw.max() - raw.min()), 1e-12)
        self.probs = cfg.p_min + raw * (cfg.p_max - cfg.p_min)
        self.thresholds = np.array(
            [NormalDist().inv_cdf(1.0 - float(p)) for p in self.probs],
            dtype=float,
        )
        self.mag_mean = np.linspace(5.0, 4.0, cfg.n_features)
        self.mag_std = np.abs(rng.normal(0.5, 0.5, size=cfg.n_features))

        self.factor, self.delta = self._make_low_rank_correlation(rng)
        self.parents, self.children_by_parent, self.nodes_by_depth = self._make_hierarchy()
        bias = rng.standard_normal(cfg.hidden_dim)
        bias /= max(float(np.linalg.norm(bias)), 1e-12)
        self.bias = cfg.bias_norm * bias

    def _make_low_rank_correlation(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        rank = max(0, min(int(self.cfg.corr_rank), int(self.cfg.n_features)))
        if rank == 0 or self.cfg.corr_scale == 0.0:
            return np.zeros((self.cfg.n_features, 0), dtype=float), np.ones(self.cfg.n_features)
        factor = self.cfg.corr_scale * rng.standard_normal((self.cfg.n_features, rank))
        row_power = np.sum(factor * factor, axis=1)
        max_power = float(row_power.max(initial=0.0))
        if max_power > 0.99:
            factor *= math.sqrt(0.99 / max_power)
            row_power = np.sum(factor * factor, axis=1)
        return factor, np.maximum(1.0 - row_power, 0.01)

    def _make_hierarchy(self) -> tuple[np.ndarray, dict[int, list[int]], list[list[int]]]:
        parents = np.full(self.cfg.n_features, -1, dtype=int)
        children_by_parent: dict[int, list[int]] = {}
        nodes_by_depth: list[list[int]] = [[] for _ in range(self.cfg.hierarchy_depth + 1)]
        next_node = 0
        while next_node < self.cfg.n_features:
            root = next_node
            nodes_by_depth[0].append(root)
            next_node += 1
            frontier = [root]
            for depth in range(1, self.cfg.hierarchy_depth + 1):
                new_frontier: list[int] = []
                for parent in frontier:
                    children: list[int] = []
                    for _ in range(self.cfg.hierarchy_branching):
                        if next_node >= self.cfg.n_features:
                            break
                        child = next_node
                        next_node += 1
                        parents[child] = parent
                        children.append(child)
                        new_frontier.append(child)
                        nodes_by_depth[depth].append(child)
                    if children:
                        children_by_parent[parent] = children
                    if next_node >= self.cfg.n_features:
                        break
                frontier = new_frontier
                if not frontier or next_node >= self.cfg.n_features:
                    break
        return parents, children_by_parent, nodes_by_depth

    def sample(self, n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        if self.factor.shape[1] == 0:
            g = rng.standard_normal((n, self.cfg.n_features))
        else:
            eps = rng.standard_normal((n, self.factor.shape[1]))
            eta = rng.standard_normal((n, self.cfg.n_features))
            g = eps @ self.factor.T + eta * np.sqrt(self.delta)[None, :]
        firing = g > self.thresholds[None, :]
        coeff = firing * np.maximum(
            self.mag_mean[None, :] + self.mag_std[None, :] * rng.standard_normal(firing.shape),
            0.0,
        )
        self._apply_hierarchy(coeff, rng)
        x = coeff @ self.dictionary + self.bias[None, :]
        return np.ascontiguousarray(x), np.ascontiguousarray(coeff), coeff > 0.0

    def _apply_hierarchy(self, coeff: np.ndarray, rng: np.random.Generator) -> None:
        for depth_nodes in self.nodes_by_depth[1:]:
            for child in depth_nodes:
                parent = int(self.parents[child])
                if parent >= 0:
                    coeff[:, child] *= coeff[:, parent] > 0.0
        for children in self.children_by_parent.values():
            if len(children) <= 1:
                continue
            active = coeff[:, children] > 0.0
            rows = np.flatnonzero(np.sum(active, axis=1) > 1)
            for row in rows:
                active_children = np.flatnonzero(active[row])
                keep = int(rng.choice(active_children))
                drop = [children[int(i)] for i in active_children if int(i) != keep]
                coeff[row, drop] = 0.0


def _r2(x: np.ndarray, fitted: np.ndarray) -> float:
    ss_res = float(np.sum((x - fitted) ** 2))
    ss_tot = float(np.sum((x - x.mean(axis=0, keepdims=True)) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-12)


def _basis_values(
    kind: str,
    coords: np.ndarray,
    n_harmonics: int,
    centers: np.ndarray | None = None,
) -> np.ndarray:
    """Evaluate an atom's *real* fitted basis matrix at ``coords`` (#1201, E4).

    Scoring must run through the same basis the decoder rows were fit against,
    not a stand-in. ``periodic``/``torus`` route through the Rust
    ``basis_with_jet`` evaluator; ``duchon`` evaluates the genuine polyharmonic
    radial+polynomial basis against the fit's stored centers (``_duchon_centers``);
    a genuinely ``linear``/``affine`` atom is the affine block ``[1, x]``.

    Any topology whose real evaluator is not reachable here (e.g. ``euclidean``,
    a degree-2 quadratic patch, or ``duchon`` with no stored centers) raises,
    rather than silently scoring a fake ``[1, x]`` and inflating the metrics.
    """
    if kind in ("periodic", "torus"):
        phi, _jet, _penalty = rust_module().basis_with_jet(
            kind,
            np.ascontiguousarray(np.asarray(coords[:, :1], dtype=float)),
            {"n_harmonics": int(n_harmonics)},
        )
        return np.asarray(phi, dtype=float)
    if kind == "duchon":
        if centers is None:
            raise ValueError(
                "duchon atom has no stored centers; cannot evaluate the real "
                "fitted basis for scoring (refusing to fake [1, x])."
            )
        return np.asarray(
            gamfit.duchon_basis(np.asarray(coords, dtype=float), np.asarray(centers, dtype=float)),
            dtype=float,
        )
    if kind in ("linear", "affine"):
        x = np.asarray(coords[:, 0], dtype=float)
        return np.column_stack([np.ones_like(x), x])
    raise ValueError(
        f"no faithful Python basis evaluator for atom topology {kind!r}; "
        "scoring would require the real fitted basis (refusing to fake [1, x])."
    )


def _learned_components(fit: gamfit.ManifoldSAE) -> tuple[np.ndarray, np.ndarray]:
    directions: list[np.ndarray] = []
    activations: list[np.ndarray] = []
    assignments = np.asarray(fit.assignments, dtype=float)
    for k, block in enumerate(fit.decoder_blocks):
        basis = fit.basis_specs[k]
        coords = np.asarray(fit.coords[k], dtype=float)
        n_harmonics = fit._n_harmonics[k] if k < len(fit._n_harmonics) else 1
        centers = fit._duchon_centers[k] if k < len(fit._duchon_centers) else None
        phi = _basis_values(basis, coords, n_harmonics, centers)
        rows = min(phi.shape[1], block.shape[0])
        for row in range(rows):
            direction = np.asarray(block[row], dtype=float)
            norm = float(np.linalg.norm(direction))
            if row == 0 or norm <= 1e-10:
                continue
            directions.append(direction / norm)
            activations.append(assignments[:, k] * phi[:, row])
    if not directions:
        return np.zeros((0, fit.training_data.shape[1])), np.zeros((fit.training_data.shape[0], 0))
    return np.vstack(directions), np.column_stack(activations)


def _manifold_total_slots(fit: gamfit.ManifoldSAE) -> int:
    """Architectural latent-slot count for a manifold SAE (#1435).

    A manifold SAE allocates ``atoms`` decoder blocks, each expanded over its
    basis harmonics (``phi.shape[1]``); the intercept row (``row == 0``) carries
    no direction and is excluded, matching :func:`_learned_components`. The total
    is ``sum_k (min(n_harmonics_k, block_rows_k) - 1)`` -- the capacity the fit
    *could* turn into directions, including geometrically-dead (zero-norm) slots
    that ``_learned_components`` drops. Passing it as the recovery denominator
    penalizes wasted atom capacity, consistent with how
    ``synth_sae_compare`` spans the full decoder width.
    """
    total = 0
    for k, block in enumerate(fit.decoder_blocks):
        basis = fit.basis_specs[k]
        coords = np.asarray(fit.coords[k], dtype=float)
        n_harmonics = fit._n_harmonics[k] if k < len(fit._n_harmonics) else 1
        phi = _basis_values(basis, coords, n_harmonics)
        # -1: the intercept row (row 0) is never a direction slot.
        total += max(min(phi.shape[1], block.shape[0]) - 1, 0)
    return total


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


def _component_scores_from_payload(fit: gamfit.ManifoldSAE, payload: dict[str, Any]) -> np.ndarray:
    assignments = np.asarray(payload["assignments"], dtype=float)
    all_scores: list[np.ndarray] = []
    for k, block in enumerate(fit.decoder_blocks):
        basis = fit.basis_specs[k]
        coords = np.asarray(payload["coords"][k], dtype=float)
        n_harmonics = fit._n_harmonics[k] if k < len(fit._n_harmonics) else 1
        phi = _basis_values(basis, coords, n_harmonics)
        for row in range(min(phi.shape[1], block.shape[0])):
            direction = np.asarray(block[row], dtype=float)
            if row == 0 or float(np.linalg.norm(direction)) <= 1e-10:
                continue
            all_scores.append(assignments[:, k] * phi[:, row])
    if not all_scores:
        n = np.asarray(payload["fitted"], dtype=float).shape[0]
        return np.zeros((n, 0))
    return np.column_stack(all_scores)


def run_one(args: argparse.Namespace, seed: int) -> BenchmarkMetrics:
    cfg = SynthConfig(
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
    synth = SynthSAEBenchData(cfg)
    train_x, train_coeff, _train_fire = synth.sample(args.n_train, seed + 1)
    test_x, test_coeff, test_fire = synth.sample(args.n_test, seed + 2)

    t0 = time.perf_counter()
    fit = gamfit.sae_manifold_fit(
        X=train_x,
        n_atoms=args.atoms,
        atom_basis=args.atom_basis,
        d_atom=args.atom_dim,
        assignment=args.assignment,
        top_k=args.top_k,
        isometry_weight=args.isometry_weight,
        ard_per_atom=args.ard_per_atom,
        sparsity_weight=args.sparsity_weight,
        smoothness_weight=args.smoothness_weight,
        n_iter=args.max_iter,
        learning_rate=args.learning_rate,
        random_state=seed,
    )
    fit_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    test_payload = fit.converged_latents(test_x)
    test_recon = np.asarray(test_payload["fitted"], dtype=float)
    learned_test = _component_scores_from_payload(fit, test_payload)
    learned_dirs, learned_train = _learned_components(fit)
    # Three distinct ground-truth measurements (see bench/_synth_sae_metrics.py
    # and #1413): uniqueness is the SynthSAEBench argmax-collision score, while
    # MCC and the direction-recovery precision/recall/F1/Jaccard come from the
    # optimal one-to-one matching. The old `len(set(cols)) / len(rows)` was
    # always 1.0 because the assignment never reuses columns. `_learned_components`
    # drops zero-norm atom directions; spanning the recovery denominator over the
    # architectural slot count (atoms x harmonics, excl. intercept) so geometrically
    # dead atom capacity is penalized, consistent with synth_sae_compare (#1435).
    n_slots = _manifold_total_slots(fit)
    uniqueness = feature_uniqueness(learned_dirs, synth.dictionary)
    rec = recovery_scores(learned_dirs, synth.dictionary, n_learned_total=n_slots)
    # Functional (activation-based) dead-latent count (#1435): a latent with a
    # nonzero decoder direction that never fires on the eval set is functionally
    # dead -- wasted capacity the geometric (decoder-norm) live check misses.
    n_firing = n_firing_latents(learned_test)
    # Comparable-protocol contribution recovery (BSF toy): per-planted-factor
    # signal recovery m_i under the OPTIMAL atom<->factor (Hungarian) assignment,
    # scored against the true per-token contribution (the planted coefficient
    # column) on the held-out split -- NOT the raw-activation reconstruction R2.
    # learned_test columns align with learned_dirs rows (both skip the intercept
    # row and zero-norm directions in the same order), so the matched atom's
    # activation is compared to its assigned factor's contribution. See module
    # header for the (M, L0, N) protocol and bench/_synth_sae_metrics.py.
    contrib = contribution_recovery(
        learned_dirs, learned_test, synth.dictionary, test_coeff
    )
    rows, cols, matching = rec.rows, rec.cols, rec.matching
    if rows.size:
        mcc = rec.mcc
        precision, recall, f1 = _probing_metrics_for_matches(
            fit,
            test_payload,
            test_fire,
            rows,
            cols,
        )
    else:
        mcc = 0.0
        precision = recall = f1 = 0.0
    score_seconds = time.perf_counter() - t1

    return BenchmarkMetrics(
        fit_api="gamfit.sae_manifold_fit",
        seed=seed,
        n_features=args.features,
        hidden_dim=args.hidden_dim,
        n_train=args.n_train,
        n_test=args.n_test,
        atoms=args.atoms,
        basis=args.atom_basis,
        atom_dim=args.atom_dim,
        assignment=args.assignment,
        top_k=args.top_k,
        true_l0_train=float(np.mean(np.sum(train_coeff > 0.0, axis=1))),
        true_l0_test=float(np.mean(np.sum(test_fire, axis=1))),
        learned_l0_train=float(np.mean(np.sum(np.abs(learned_train) > 1e-8, axis=1))),
        learned_l0_test=float(np.mean(np.sum(np.abs(learned_test) > 1e-8, axis=1))),
        train_r2=_r2(train_x, np.asarray(fit.fitted, dtype=float)),
        test_r2=_r2(test_x, np.asarray(test_recon, dtype=float)),
        mcc=mcc,
        feature_uniqueness=uniqueness,
        direction_recovery_precision=rec.precision,
        direction_recovery_recall=rec.recall,
        direction_recovery_f1=rec.f1,
        direction_recovery_jaccard=rec.jaccard,
        contribution_recovery_mean=contrib.mean,
        contribution_recovery_matched_factors=int(contrib.per_factor.size),
        n_latent_slots=n_slots,
        n_firing_latents=n_firing,
        probing_precision=precision,
        probing_recall=recall,
        probing_f1=f1,
        matching=matching,
        learned_directions=int(learned_dirs.shape[0]),
        fit_seconds=fit_seconds,
        score_seconds=score_seconds,
    )


def _summarize(metrics: list[BenchmarkMetrics]) -> dict[str, Any]:
    numeric_fields = [
        "true_l0_train",
        "true_l0_test",
        "learned_l0_train",
        "learned_l0_test",
        "train_r2",
        "test_r2",
        "mcc",
        "feature_uniqueness",
        "direction_recovery_precision",
        "direction_recovery_recall",
        "direction_recovery_f1",
        "direction_recovery_jaccard",
        "contribution_recovery_mean",
        "contribution_recovery_matched_factors",
        "n_latent_slots",
        "n_firing_latents",
        "probing_precision",
        "probing_recall",
        "probing_f1",
        "learned_directions",
        "fit_seconds",
        "score_seconds",
    ]
    out: dict[str, Any] = {}
    for field in numeric_fields:
        values = np.array([float(getattr(item, field)) for item in metrics], dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        out[field] = {
            "mean": mean,
            "std": std,
            "cv": float(std / abs(mean)) if mean != 0.0 else None,
        }
    return out


def _benchmark_notes() -> dict[str, Any]:
    return {
        "implemented": {
            "core": [
                "train_r2",
                "test_r2",
                "true_l0",
                "learned_l0",
            ],
            "synthsaebench": [
                "GT-MCC-style absolute-cosine feature recovery (Hungarian matched pairs)",
                "GT-F1-style matched-latent firing precision/recall/F1",
                "feature uniqueness (argmax-collision: #distinct best matches / n_learned)",
                "direction recovery precision/recall/F1/Jaccard (quality-aware matched cosine mass)",
                "BSF-toy per-factor contribution recovery m_i under optimal atom<->factor (Hungarian) matching (NOT raw-activation R2)",
            ],
            "saebench_compatible_synthetic": [
                "sparse-probing analogue on matched ground-truth features",
            ],
        },
        "not_applicable_without_llm_task_stack": {
            "SAEBench": [
                "SCR and TPP need task datasets, probes, and latent ablations on an LLM SAE.",
                "Autointerp needs natural-language activating examples.",
                "RAVEL needs entity-attribute prompts and causal interventions on an LLM.",
            ],
            "SAGE": [
                "Requires an LLM, task prompts, attribution/cross-section discovery, supervised feature dictionaries, and downstream logit-difference interventions.",
                "This synthetic activation harness can report direct ground-truth recovery, but cannot honestly run SAGE Test 1 sufficiency/necessity or Test 2 sparse controllability without the LLM task circuit.",
            ],
        },
    }


def _probing_metrics_for_matches(
    fit: gamfit.ManifoldSAE,
    payload: dict[str, Any],
    fire: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[float, float, float]:
    scores = _component_scores_from_payload(fit, payload)
    if scores.shape[1] == 0:
        return 0.0, 0.0, 0.0
    metrics = [_best_f1(scores[:, int(row)], fire[:, int(col)]) for row, col in zip(rows, cols)]
    return tuple(float(np.mean([m[i] for m in metrics])) for i in range(3))  # type: ignore[return-value]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--n-train", type=int, default=512)
    parser.add_argument("--n-test", type=int, default=256)
    parser.add_argument("--atoms", type=int, default=64)
    parser.add_argument("--atom-basis", default="duchon")
    parser.add_argument("--atom-dim", type=int, default=1)
    parser.add_argument("--assignment", default="softmax")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--sparsity-weight", type=float, default=0.01)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)
    parser.add_argument("--isometry-weight", type=float, default=0.0)
    parser.add_argument("--ard-per-atom", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--corr-rank", type=int, default=8)
    parser.add_argument("--corr-scale", type=float, default=0.1)
    parser.add_argument("--p-min", type=float, default=5e-4)
    parser.add_argument("--p-max", type=float, default=0.4)
    parser.add_argument("--zipf-exponent", type=float, default=0.5)
    parser.add_argument("--hierarchy-branching", type=int, default=4)
    parser.add_argument("--hierarchy-depth", type=int, default=3)
    parser.add_argument("--bias-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=0, help="Single seed used when --seeds is omitted.")
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    if args.atoms >= args.n_train:
        parser.error("--atoms must be smaller than --n-train")
    seeds = args.seeds if args.seeds is not None else [args.seed]
    metrics = [run_one(args, int(seed)) for seed in seeds]
    payload = {
        "benchmark": "SynthSAEBench-style direct ground-truth benchmark for gamfit manifold SAE",
        "fit_api": "gamfit.sae_manifold_fit",
        "notes": _benchmark_notes(),
        "runs": [asdict(item) for item in metrics],
        "summary": _summarize(metrics),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
