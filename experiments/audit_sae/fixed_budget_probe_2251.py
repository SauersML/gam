#!/usr/bin/env python3
"""Fixed-budget concept-probe acceptance and diagnostics for issue #2251.

This driver evaluates already-pooled representations on one frozen document
split.  SAE arms are feature-selected on the training rows only and are hard
limited to 64 probe latents.  The raw-residual arm is the dense skyline.  Each
SAE arm must also carry its token-level sparse route so the report diagnoses
feature frequency, local TopK competition, absorption, splitting, and the
per-concept tail instead of emitting one aggregate accuracy.

Input is one ``.npz`` with these required keys:

``labels``
    ``(N,)`` class labels.
``train_indices``, ``test_indices``
    The frozen, disjoint document split.
``reps__NAME``
    ``(N,F)`` mean-pooled representation for an arm. ``reps__raw`` is the dense
    residual skyline; every other arm is evaluated with at most 64 columns.
``active__NAME``, ``n_atoms__NAME``, ``block_size__NAME``
    Scalar per-token active budget, scalar dictionary capacity, and block width
    for every non-raw arm.  The candidate and scalar baseline must have the same
    ``n_atoms``.  ``active`` must equal 64; it is never inferred from shapes.
``route_indices__NAME``, ``route_values__NAME``
    Token-level fixed-width sparse route. Scalar values are ``(T,s)``; block
    values are ``(T,k,b)`` and indices name blocks.
``document_offsets``
    ``(N+1,)`` token offsets shared by the routes.  Offsets are retained in the
    contract so pooling provenance is checkable, although token diagnostics use
    the complete route directly.

The block candidate is trained with
``gamfit.fixed_budget_block_sparse_dictionary_fit``.  For the production
DeepSeek comparison, ``n_atoms=114688, active=64, block_size=4`` means 28,672
orthonormal blocks and exactly 16 selected blocks/token: identical scalar
capacity and identical 64-coordinate activity to the baseline, with correlated
within-concept directions competing once through a group-l2 gate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score


ACTIVE_BUDGET = 64
ACCURACY_FLOOR = 0.646
PRIMARY_ACCURACY = 0.667
MACRO_AUC_FLOOR = 0.926
TAIL_AUC_FLOOR = 0.83


@dataclass(frozen=True)
class Arm:
    name: str
    representation: np.ndarray
    active: int | None
    n_atoms: int | None
    block_size: int | None
    route_indices: np.ndarray | None
    route_values: np.ndarray | None


def _scalar(archive: Any, key: str) -> int:
    if key not in archive.files:
        raise ValueError(f"missing required scalar {key!r}")
    values = np.asarray(archive[key]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"{key} must contain one value; got shape {archive[key].shape}")
    return int(values[0])


def load_arms(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[Arm]]:
    archive = np.load(path, allow_pickle=False)
    for key in ("labels", "train_indices", "test_indices", "document_offsets"):
        if key not in archive.files:
            raise ValueError(f"{path} is missing required key {key!r}")
    labels = np.asarray(archive["labels"])
    if labels.ndim != 1 or labels.size < 2:
        raise ValueError(f"labels must be a nontrivial 1-D array; got {labels.shape}")
    train = np.asarray(archive["train_indices"], dtype=np.int64)
    test = np.asarray(archive["test_indices"], dtype=np.int64)
    offsets = np.asarray(archive["document_offsets"], dtype=np.int64)
    if train.ndim != 1 or test.ndim != 1 or train.size == 0 or test.size == 0:
        raise ValueError("train_indices and test_indices must be non-empty 1-D arrays")
    if np.intersect1d(train, test).size:
        raise ValueError("the frozen train/test document split overlaps")
    if min(train.min(), test.min()) < 0 or max(train.max(), test.max()) >= labels.size:
        raise ValueError("train/test indices exceed the labels array")
    if offsets.shape != (labels.size + 1,) or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError(
            f"document_offsets must be monotone with shape {(labels.size + 1,)} and start at zero"
        )

    names = sorted(key.removeprefix("reps__") for key in archive.files if key.startswith("reps__"))
    if "raw" not in names:
        raise ValueError("the dense skyline reps__raw is required")
    if len(names) < 2:
        raise ValueError("at least one SAE representation is required alongside reps__raw")
    arms: list[Arm] = []
    for name in names:
        rep = np.asarray(archive[f"reps__{name}"], dtype=np.float64)
        if rep.ndim != 2 or rep.shape[0] != labels.size or rep.shape[1] == 0:
            raise ValueError(f"reps__{name} must have shape (N,F); got {rep.shape}")
        if not np.isfinite(rep).all():
            raise ValueError(f"reps__{name} contains a non-finite value")
        if name == "raw":
            arms.append(Arm(name, rep, None, None, None, None, None))
            continue
        active = _scalar(archive, f"active__{name}")
        n_atoms = _scalar(archive, f"n_atoms__{name}")
        block_size = _scalar(archive, f"block_size__{name}")
        if active != ACTIVE_BUDGET:
            raise ValueError(
                f"arm {name!r} has active={active}; #2251 acceptance requires exactly {ACTIVE_BUDGET}"
            )
        if rep.shape[1] != n_atoms:
            raise ValueError(
                f"reps__{name} has F={rep.shape[1]} but n_atoms__{name}={n_atoms}"
            )
        ikey, vkey = f"route_indices__{name}", f"route_values__{name}"
        if ikey not in archive.files or vkey not in archive.files:
            raise ValueError(f"arm {name!r} requires both {ikey!r} and {vkey!r}")
        indices = np.asarray(archive[ikey], dtype=np.int64)
        values = np.asarray(archive[vkey], dtype=np.float64)
        if indices.ndim != 2 or values.shape[:2] != indices.shape:
            raise ValueError(
                f"route for {name!r} must align on (tokens,slots); got {indices.shape} and {values.shape}"
            )
        if indices.shape[0] != offsets[-1]:
            raise ValueError(
                f"route for {name!r} has {indices.shape[0]} tokens but document_offsets ends at {offsets[-1]}"
            )
        if block_size == 1 and values.ndim != 2:
            raise ValueError(f"scalar arm {name!r} requires 2-D route_values")
        if block_size > 1 and (values.ndim != 3 or values.shape[2] != block_size):
            raise ValueError(
                f"block arm {name!r} requires route_values (T,k,{block_size}); got {values.shape}"
            )
        structural_active = indices.shape[1] * block_size
        if structural_active != active:
            raise ValueError(
                f"arm {name!r} route exposes {structural_active} scalar slots/token, not active={active}"
            )
        n_units = n_atoms // block_size
        if n_atoms % block_size or indices.size and (indices.min() < 0 or indices.max() >= n_units):
            raise ValueError(f"arm {name!r} route indices do not fit its exact block partition")
        arms.append(Arm(name, rep, active, n_atoms, block_size, indices, values))
    return labels, train, test, offsets, arms


def anova_scores(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Training-only multiclass ANOVA F scores, with constant columns scored zero."""
    classes, inverse = np.unique(y, return_inverse=True)
    if classes.size < 2:
        raise ValueError("feature selection requires at least two training classes")
    n, width = x.shape
    overall = x.mean(axis=0)
    between = np.zeros(width, dtype=np.float64)
    within = np.zeros(width, dtype=np.float64)
    for class_index in range(classes.size):
        rows = x[inverse == class_index]
        mean = rows.mean(axis=0)
        between += rows.shape[0] * (mean - overall) ** 2
        within += ((rows - mean) ** 2).sum(axis=0)
    numerator = between / (classes.size - 1)
    denominator = within / max(n - classes.size, 1)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def select_features(x: np.ndarray, y: np.ndarray, budget: int) -> np.ndarray:
    if budget <= 0 or budget > x.shape[1]:
        raise ValueError(f"feature budget {budget} must be in [1, {x.shape[1]}]")
    scores = anova_scores(x, y)
    # Stable tie-breaking by ascending feature id makes the frozen protocol reproducible.
    return np.lexsort((np.arange(scores.size), -scores))[:budget]


def _fit_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0.0] = 1.0
    train_z = (x_train - mean) / scale
    test_z = (x_test - mean) / scale
    probe = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=10_000,
        random_state=seed,
    )
    probe.fit(train_z, y_train)
    return probe.predict(test_z), probe.predict_proba(test_z)


def probe_metrics(y: np.ndarray, prediction: np.ndarray, probability: np.ndarray) -> dict[str, Any]:
    classes = np.unique(y)
    if probability.shape[1] != classes.size:
        raise ValueError("probe probability columns do not match held-out classes")
    per_class: dict[str, Any] = {}
    aucs = []
    for column, label in enumerate(classes):
        binary = y == label
        auc = float(roc_auc_score(binary, probability[:, column]))
        aucs.append(auc)
        per_class[str(label)] = {
            "auc": auc,
            "accuracy": float(np.mean(prediction[binary] == label)),
            "n_test": int(binary.sum()),
        }
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "macro_f1": float(f1_score(y, prediction, average="macro")),
        "macro_auc": float(np.mean(aucs)),
        "min_class_auc": float(np.min(aucs)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y, prediction, labels=classes).tolist(),
        "class_order": [str(value) for value in classes],
    }


def class_signal_concentration(
    x: np.ndarray, y: np.ndarray, selected: np.ndarray
) -> dict[str, Any]:
    """Per-class standardized mean-shift concentration (feature splitting)."""
    report: dict[str, Any] = {}
    for label in np.unique(y):
        positive = x[y == label]
        negative = x[y != label]
        mean_shift = positive.mean(axis=0) - negative.mean(axis=0)
        pooled_var = positive.var(axis=0) + negative.var(axis=0)
        signal = np.divide(
            mean_shift * mean_shift,
            pooled_var,
            out=np.zeros_like(mean_shift),
            where=pooled_var > 0,
        )
        total = float(signal.sum())
        if total == 0.0:
            report[str(label)] = {
                "effective_discriminative_features": 0.0,
                "largest_feature_share": 0.0,
                "selected_signal_share": 0.0,
            }
            continue
        shares = signal / total
        report[str(label)] = {
            "effective_discriminative_features": float(1.0 / np.sum(shares * shares)),
            "largest_feature_share": float(shares.max()),
            "selected_signal_share": float(shares[selected].sum()),
        }
    return report


def _expanded_route_chunk(arm: Arm, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    assert arm.route_indices is not None and arm.route_values is not None
    assert arm.block_size is not None
    indices = arm.route_indices[start:end]
    values = arm.route_values[start:end]
    if arm.block_size == 1:
        return indices, values != 0.0
    offsets = np.arange(arm.block_size, dtype=np.int64)
    expanded = indices[:, :, None] * arm.block_size + offsets[None, None, :]
    return expanded.reshape(end - start, -1), values.reshape(end - start, -1) != 0.0


def route_diagnostics(arm: Arm, selected: np.ndarray) -> dict[str, Any]:
    assert arm.n_atoms is not None and arm.active is not None
    assert arm.route_indices is not None
    firing = np.zeros(arm.n_atoms, dtype=np.int64)
    selected_sorted = np.sort(selected)
    marginals = np.zeros(selected.size, dtype=np.int64)
    joint = np.zeros((selected.size, selected.size), dtype=np.int64)
    active_counts: list[np.ndarray] = []
    selected_active_counts: list[np.ndarray] = []
    chunk_rows = 65_536
    for start in range(0, arm.route_indices.shape[0], chunk_rows):
        end = min(start + chunk_rows, arm.route_indices.shape[0])
        feature, live = _expanded_route_chunk(arm, start, end)
        np.add.at(firing, feature[live], 1)
        active_counts.append(live.sum(axis=1))

        flat_feature = feature[live]
        flat_row = np.broadcast_to(np.arange(end - start)[:, None], feature.shape)[live]
        position = np.searchsorted(selected_sorted, flat_feature)
        keep = (position < selected_sorted.size) & (selected_sorted[position.clip(max=selected_sorted.size - 1)] == flat_feature)
        binary = np.zeros((end - start, selected.size), dtype=np.int16)
        binary[flat_row[keep], position[keep]] = 1
        selected_active_counts.append(binary.sum(axis=1))
        marginals += binary.sum(axis=0, dtype=np.int64)
        joint += binary.T.astype(np.int64) @ binary.astype(np.int64)

    per_row = np.concatenate(active_counts)
    selected_per_row = np.concatenate(selected_active_counts)
    live_firing = firing[firing > 0]
    probabilities = firing.astype(np.float64) / max(firing.sum(), 1)
    effective = float(1.0 / np.sum(probabilities * probabilities)) if firing.sum() else 0.0

    pairs = []
    for left in range(selected.size):
        for right in range(left + 1, selected.size):
            n_joint = int(joint[left, right])
            if n_joint == 0:
                continue
            p_left_given_right = n_joint / marginals[right]
            p_right_given_left = n_joint / marginals[left]
            union = marginals[left] + marginals[right] - n_joint
            pairs.append(
                {
                    "left": int(selected_sorted[left]),
                    "right": int(selected_sorted[right]),
                    "p_left_given_right": float(p_left_given_right),
                    "p_right_given_left": float(p_right_given_left),
                    "absorption_asymmetry": float(abs(p_left_given_right - p_right_given_left)),
                    "jaccard": float(n_joint / union),
                    "n_joint": n_joint,
                }
            )
    pairs.sort(key=lambda row: (-row["absorption_asymmetry"], row["left"], row["right"]))
    return {
        "tokens": int(per_row.size),
        "structural_scalar_budget": arm.active,
        "mean_nonzero_scalar_coordinates": float(per_row.mean()),
        "max_nonzero_scalar_coordinates": int(per_row.max(initial=0)),
        "live_features": int(live_firing.size),
        "dead_feature_fraction": float(np.mean(firing == 0)),
        "effective_feature_count_from_frequency": effective,
        "firing_count_quantiles_live": (
            np.quantile(live_firing, [0.0, 0.25, 0.5, 0.75, 1.0]).tolist()
            if live_firing.size
            else [0.0] * 5
        ),
        "mean_selected_probe_features_active_per_token": float(selected_per_row.mean()),
        "tokens_with_no_selected_probe_feature": float(np.mean(selected_per_row == 0)),
        "top_absorption_pairs": pairs[:16],
        "mean_selected_pair_jaccard": float(np.mean([row["jaccard"] for row in pairs]))
        if pairs
        else 0.0,
    }


def evaluate_once(
    arm: Arm,
    labels: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    train_rows: np.ndarray | None,
    seed: int,
) -> tuple[dict[str, Any], np.ndarray]:
    rows = train if train_rows is None else train_rows
    x_train = arm.representation[rows]
    y_train = labels[rows]
    if arm.name == "raw":
        selected = np.arange(x_train.shape[1], dtype=np.int64)
    else:
        selected = select_features(x_train, y_train, min(ACTIVE_BUDGET, x_train.shape[1]))
    prediction, probability = _fit_probe(
        x_train[:, selected], y_train, arm.representation[test][:, selected], seed
    )
    return probe_metrics(labels[test], prediction, probability), selected


def stratified_bootstrap(train: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pieces = []
    for label in np.unique(labels[train]):
        rows = train[labels[train] == label]
        pieces.append(rng.choice(rows, size=rows.size, replace=True))
    out = np.concatenate(pieces)
    rng.shuffle(out)
    return out


def summarize_bootstraps(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for metric in ("accuracy", "macro_f1", "macro_auc", "min_class_auc"):
        values = np.asarray([row[metric] for row in rows], dtype=np.float64)
        summary[metric] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "percentile_95_interval": np.quantile(values, [0.025, 0.975]).tolist(),
        }
    classes = rows[0]["per_class"]
    summary["per_class_auc"] = {}
    for label in classes:
        values = np.asarray([row["per_class"][label]["auc"] for row in rows])
        summary["per_class_auc"][label] = {
            "mean": float(values.mean()),
            "standard_deviation": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "percentile_95_interval": np.quantile(values, [0.025, 0.975]).tolist(),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--candidate", required=True, help="NAME of the fixed-budget block arm")
    parser.add_argument("--baseline", required=True, help="NAME of the scalar TopK-64 arm")
    parser.add_argument("--bootstrap-seeds", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_seeds < 2:
        raise ValueError("repeated-seed uncertainty requires --bootstrap-seeds >= 2")

    labels, train, test, offsets, arms = load_arms(args.input)
    by_name = {arm.name: arm for arm in arms}
    if args.candidate not in by_name or args.baseline not in by_name:
        raise ValueError(f"candidate/baseline must name loaded arms: {sorted(by_name)}")
    candidate, baseline = by_name[args.candidate], by_name[args.baseline]
    if candidate.name == "raw" or baseline.name == "raw":
        raise ValueError("candidate and baseline must be SAE arms, not the raw skyline")
    if candidate.n_atoms != baseline.n_atoms:
        raise ValueError(
            f"dictionary capacity mismatch: candidate K={candidate.n_atoms}, baseline K={baseline.n_atoms}"
        )

    report: dict[str, Any] = {
        "issue": 2251,
        "input": str(args.input),
        "frozen_split": {"train": int(train.size), "test": int(test.size)},
        "documents": int(labels.size),
        "tokens": int(offsets[-1]),
        "probe_latent_budget": ACTIVE_BUDGET,
        "per_token_scalar_budget": ACTIVE_BUDGET,
        "arms": {},
    }
    for arm in arms:
        primary, selected = evaluate_once(arm, labels, train, test, None, seed=0)
        bootstraps = []
        for seed in range(args.bootstrap_seeds):
            sampled = stratified_bootstrap(train, labels, seed)
            metrics, _ = evaluate_once(arm, labels, train, test, sampled, seed=seed)
            bootstraps.append(metrics)
        arm_report: dict[str, Any] = {
            "width": int(arm.representation.shape[1]),
            "selected_features": selected.tolist(),
            "selected_count": int(selected.size),
            "primary": primary,
            "repeated_seed_uncertainty": summarize_bootstraps(bootstraps),
            "per_class_signal_concentration": class_signal_concentration(
                arm.representation[train], labels[train], selected
            ),
        }
        if arm.name != "raw":
            arm_report.update(
                {
                    "n_atoms": arm.n_atoms,
                    "active": arm.active,
                    "block_size": arm.block_size,
                    "route_diagnostics": route_diagnostics(arm, selected),
                }
            )
        report["arms"][arm.name] = arm_report

    primary = report["arms"][candidate.name]["primary"]
    route = report["arms"][candidate.name]["route_diagnostics"]
    checks = {
        "same_dictionary_capacity": candidate.n_atoms == baseline.n_atoms,
        "at_most_64_probe_latents": report["arms"][candidate.name]["selected_count"] <= 64,
        "exactly_64_active_coordinates": candidate.active == 64
        and route["max_nonzero_scalar_coordinates"] <= 64,
        "accuracy_floor_0_646": primary["accuracy"] >= ACCURACY_FLOOR,
        "primary_accuracy_0_667": primary["accuracy"] >= PRIMARY_ACCURACY,
        "macro_auc_floor_0_926": primary["macro_auc"] >= MACRO_AUC_FLOOR,
        "tail_auc_floor_0_83": primary["min_class_auc"] >= TAIL_AUC_FLOOR,
    }
    report["acceptance"] = {
        "checks": checks,
        "required_pass": all(
            checks[key]
            for key in (
                "same_dictionary_capacity",
                "at_most_64_probe_latents",
                "exactly_64_active_coordinates",
                "accuracy_floor_0_646",
                "macro_auc_floor_0_926",
                "tail_auc_floor_0_83",
            )
        ),
        "primary_target_pass": all(checks.values()),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["acceptance"], sort_keys=True), flush=True)
    return 1 if args.require_pass and not report["acceptance"]["required_pass"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
