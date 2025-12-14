from __future__ import annotations

import argparse
import itertools
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from scipy.stats import t as student_t
from xgboost import XGBClassifier


SAMPLE_POPULATION_MAPPING_URL = (
    "https://raw.githubusercontent.com/SauersML/genomic_pca/refs/heads/main/data/sample_population_mapping.tsv"
)


@dataclass
class DatasetSummary:
    name: str
    num_samples: int
    num_features: int
    class_counts: Counter

    def describe(self) -> str:
        header = f"Dataset '{self.name}': {self.num_samples} samples, {self.num_features} features"
        parts = [header, "  Class distribution:"]
        for label, count in self.class_counts.most_common():
            parts.append(f"    - {label}: {count}")
        return "\n".join(parts)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare PCA projection TSV files using stratified cross-validated classifiers "
            "and dependence-aware paired statistical testing across projection sources."
        )
    )
    parser.add_argument(
        "--projection",
        action="append",
        metavar="LABEL=PATH",
        help=(
            "Projection TSV to include in the comparison, provided as LABEL=PATH. "
            "Supply at least two instances to compare multiple datasets."
        ),
        required=True,
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=20,
        help="Requested number of stratified CV folds (defaults to 20).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used when shuffling splits.",
    )
    args = parser.parse_args(argv)

    parsed: list[tuple[str, Path]] = []
    for raw in args.projection:
        if "=" not in raw:
            parser.error(
                f"Invalid --projection value '{raw}'. Expected format LABEL=PATH."
            )
        label, path_str = raw.split("=", 1)
        label = label.strip()
        if not label:
            parser.error("Projection label must be non-empty.")
        path = Path(path_str.strip())
        parsed.append((label, path))

    if len(parsed) < 2:
        parser.error("Provide at least two --projection arguments to perform a comparison.")

    seen_labels: set[str] = set()
    for label, _ in parsed:
        if label in seen_labels:
            parser.error(f"Duplicate projection label specified: '{label}'")
        seen_labels.add(label)

    args.projections = parsed
    return args


def _normalize_text(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _load_sample_population_mapping() -> pd.DataFrame:
    print(
        "Downloading sample population mapping TSV from "
        f"{SAMPLE_POPULATION_MAPPING_URL}…",
        flush=True,
    )
    df = pd.read_csv(SAMPLE_POPULATION_MAPPING_URL, sep="\t", dtype=str)
    print(
        "  Raw mapping shape: {} rows × {} columns".format(len(df), len(df.columns)),
        flush=True,
    )
    print(
        "  Mapping columns: {}".format(", ".join(df.columns.tolist()[:20])),
        flush=True,
    )
    string_like_columns = df.select_dtypes(include=["object", "string"])
    if not string_like_columns.empty:
        df[string_like_columns.columns] = string_like_columns.apply(
            lambda col: col.str.strip()
        )

    lower_to_actual = {column.lower(): column for column in df.columns}
    subpop_col = lower_to_actual.get("meta_subpopulation")
    superpop_col = lower_to_actual.get("meta_superpopulation")
    if subpop_col is None or superpop_col is None:
        raise ValueError(
            "Sample population mapping TSV is missing 'meta_subpopulation' or 'meta_superpopulation' columns."
        )

    sample_id_columns = [
        column for column in df.columns if column.lower().startswith("sample_id")
    ]
    if not sample_id_columns:
        raise ValueError(
            "Sample population mapping TSV does not contain any columns beginning with 'sample_id'."
        )

    records: list[dict[str, str | None]] = []
    for _, row in df.iterrows():
        subpop = _normalize_text(row[subpop_col])
        superpop = _normalize_text(row[superpop_col])
        for sample_col in sample_id_columns:
            sample_id = _normalize_text(row[sample_col])
            if sample_id is None:
                continue
            records.append(
                {
                    "SampleID": sample_id,
                    "Subpopulation": subpop,
                    "Superpopulation": superpop,
                }
            )

    if not records:
        raise ValueError(
            "Sample population mapping TSV did not yield any sample identifiers after processing."
        )

    expanded = pd.DataFrame.from_records(records)
    print(
        "  Expanded to {} sample → population assignments".format(len(expanded)),
        flush=True,
    )
    expanded = expanded.drop_duplicates(subset="SampleID", keep="first")
    print(
        "  After dropping duplicates: {} unique SampleID entries".format(
            len(expanded)
        ),
        flush=True,
    )
    expanded["SampleID"] = expanded["SampleID"].astype("string")
    expanded["Subpopulation"] = expanded["Subpopulation"].astype("string")
    expanded["Superpopulation"] = expanded["Superpopulation"].astype("string")
    print(
        "Finished loading population mapping: preview {}".format(
            expanded.head(3).to_dict(orient="records")
        ),
        flush=True,
    )
    return expanded


def _annotate_with_population_labels(
    df: pd.DataFrame, label: str, mapping: pd.DataFrame
) -> pd.DataFrame:
    merged = df.merge(mapping, how="left", left_on="SampleName", right_on="SampleID")
    merged = merged.drop(columns=["SampleID"])

    unmatched = merged["Subpopulation"].isna() & merged["Superpopulation"].isna()
    if unmatched.any():
        sample_names = merged.loc[unmatched, "SampleName"].dropna().astype(str).unique()
        preview = ", ".join(sample_names[:10])
        print(
            "Warning: {} samples in '{}' lacked population mapping entries.{}".format(
                len(sample_names),
                label,
                f" Examples: {preview}" if preview else "",
            ),
            flush=True,
        )

    missing_subpop = merged["Subpopulation"].isna() & ~unmatched
    if missing_subpop.any():
        sample_names = merged.loc[missing_subpop, "SampleName"].dropna().astype(str).unique()
        preview = ", ".join(sample_names[:10])
        print(
            "Warning: {} samples in '{}' were missing meta_subpopulation assignments.{}".format(
                len(sample_names),
                label,
                f" Examples: {preview}" if preview else "",
            ),
            flush=True,
        )

    missing_superpop = merged["Superpopulation"].isna() & ~unmatched
    if missing_superpop.any():
        sample_names = (
            merged.loc[missing_superpop, "SampleName"].dropna().astype(str).unique()
        )
        preview = ", ".join(sample_names[:10])
        print(
            "Warning: {} samples in '{}' were missing meta_superpopulation assignments.{}".format(
                len(sample_names),
                label,
                f" Examples: {preview}" if preview else "",
            ),
            flush=True,
        )

    return merged


def _read_projection(path: Path, label: str, mapping: pd.DataFrame) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Projection TSV for {label} not found: {path}")

    print(f"Loading projection data for {label} from {path}…", flush=True)
    df = pd.read_csv(path, sep="\t")
    print(
        "  Raw projection shape for {}: {} rows × {} columns".format(
            label, len(df), len(df.columns)
        ),
        flush=True,
    )
    print(
        "  Projection columns (first 15): {}".format(", ".join(df.columns[:15])),
        flush=True,
    )

    expected_cols = {"SampleName"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns in {label} projection TSV: {sorted(missing)}"
        )

    feature_columns = [col for col in df.columns if col.startswith("PC")]
    if not feature_columns:
        raise ValueError(f"No principal component columns (PC*) found in {label} TSV")

    df = df[["SampleName", *feature_columns]].copy()
    df = _annotate_with_population_labels(df, label, mapping)
    print(
        "  Annotated projection '{}' now has columns: {}".format(
            label, ", ".join(df.columns[:15])
        ),
        flush=True,
    )
    return df


def _normalize_subpop(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    disallowed = ("nan", "unk", "unknown", "other", "others", "na", "none")
    if any(token in lowered for token in disallowed):
        return None
    return text


def _filter_projection(df: pd.DataFrame, label: str) -> pd.DataFrame:
    df = df.copy()
    df["Subpopulation"] = df["Subpopulation"].map(_normalize_subpop)
    before = len(df)
    df = df.dropna(subset=["Subpopulation"])
    removed = before - len(df)
    if removed:
        print(
            f"Removed {removed} rows with undefined or disallowed subpopulation labels from {label}.",
            flush=True,
        )
    else:
        print(
            f"No rows removed for subpopulation filtering in {label}; {len(df)} rows remain.",
            flush=True,
        )

    duplicates = df.duplicated(subset="SampleName")
    if duplicates.any():
        dup_count = duplicates.sum()
        print(
            f"Warning: detected {dup_count} duplicated SampleName entries in {label}; keeping first occurrence.",
            flush=True,
        )
        df = df[~duplicates]
    else:
        print(
            f"No duplicate SampleName entries detected in {label}.",
            flush=True,
        )

    df = df.set_index("SampleName", drop=False)
    print(
        "  Filtered projection '{}' index size: {}".format(label, len(df)),
        flush=True,
    )
    return df


def _align_datasets(datasets: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    sample_sets = [set(df.index) for df in datasets.values()]
    common_samples = sorted(set.intersection(*sample_sets)) if sample_sets else []
    if not common_samples:
        raise ValueError("No overlapping samples found across the provided projections")

    print(
        "Alignment diagnostics:",
        flush=True,
    )
    for label, df in datasets.items():
        print(
            f"  Dataset '{label}' total samples: {len(df)} (unique index entries)",
            flush=True,
        )
    print(
        f"  Common overlapping samples across datasets: {len(common_samples)}",
        flush=True,
    )

    aligned: dict[str, pd.DataFrame] = {}
    for label, df in datasets.items():
        missing = len(df) - len(common_samples)
        if missing:
            print(
                f"Omitted {missing} samples unique to {label} data to align datasets.",
                flush=True,
            )
        else:
            print(
                f"  No samples omitted when aligning dataset '{label}'.",
                flush=True,
            )
        aligned[label] = df.loc[common_samples].copy()

    ref_label = next(iter(aligned))
    ref_subpops = aligned[ref_label]["Subpopulation"]
    for label, df in aligned.items():
        mismatched = df["Subpopulation"] != ref_subpops
        if mismatched.any():
            mismatching_samples = mismatched[mismatched].index.tolist()
            sample_preview = ", ".join(mismatching_samples[:10])
            raise ValueError(
                "Subpopulation labels disagree between projections '{}' and '{}' for samples: {}".format(
                    ref_label, label, sample_preview
                )
            )

    return aligned


def _determine_feature_columns(datasets: dict[str, pd.DataFrame]) -> list[str]:
    feature_sets = {
        label: {col for col in df.columns if col.startswith("PC")}
        for label, df in datasets.items()
    }
    common = set.intersection(*feature_sets.values()) if feature_sets else set()
    if not common:
        raise ValueError("No shared principal component columns across all datasets")

    def sort_key(name: str) -> tuple[int, str]:
        suffix = name[2:]
        if suffix.isdigit():
            return (0, int(suffix))
        return (1, name)

    selected = sorted(common, key=sort_key)

    for label, features in feature_sets.items():
        print(
            "  Dataset '{}' provides {} PC columns.".format(label, len(features)),
            flush=True,
        )

    for label, features in feature_sets.items():
        removed = sorted(features - set(selected), key=sort_key)
        if removed:
            print(
                f"Projection '{label}' lacks {len(removed)} PCs present in others;"
                f" they will be excluded: {', '.join(removed[:10])}",
                flush=True,
            )

    print(
        f"Using {len(selected)} shared principal components: {', '.join(selected[:20])}"
        + (" …" if len(selected) > 20 else ""),
        flush=True,
    )

    print(
        f"Final feature column selection ({len(selected)} PCs): {', '.join(selected[:25])}"
        + (" …" if len(selected) > 25 else ""),
        flush=True,
    )

    return selected


def _filter_small_classes(df: pd.DataFrame, min_size: int) -> pd.DataFrame:
    counts = df["Subpopulation"].value_counts()
    allowed = counts[counts >= min_size].index
    removed_labels = sorted(set(counts.index) - set(allowed))
    if removed_labels:
        print(
            "Removing subpopulations with insufficient support (< {} samples): {}".format(
                min_size, ", ".join(removed_labels)
            ),
            flush=True,
        )
    return df[df["Subpopulation"].isin(allowed)].copy()


def _filter_classes_across_datasets(
    datasets: dict[str, pd.DataFrame], min_size: int
) -> dict[str, pd.DataFrame]:
    """Keep only subpopulations that have at least `min_size` samples in *every* dataset."""
    counts_per = {label: df["Subpopulation"].value_counts() for label, df in datasets.items()}
    all_labels = set().union(*[c.index for c in counts_per.values()])
    allowed = sorted(
        lab for lab in all_labels
        if min(c.get(lab, 0) for c in counts_per.values()) >= min_size
    )
    removed = sorted(all_labels - set(allowed))
    if removed:
        print(
            "Removing subpopulations with insufficient support across datasets "
            f"(< {min_size} samples in at least one dataset): {', '.join(removed)}",
            flush=True,
        )
    else:
        print(
            "No subpopulations removed when enforcing minimum size across datasets.",
            flush=True,
        )
    for label, counts in counts_per.items():
        print(
            "  Dataset '{}' subpopulation counts (top 10): {}".format(
                label,
                counts.sort_values(ascending=False).head(10).to_dict(),
            ),
            flush=True,
        )
    if not allowed:
        raise ValueError(
            "After alignment, no subpopulation meets the required size across all datasets."
        )
    return {label: df[df["Subpopulation"].isin(allowed)].copy() for label, df in datasets.items()}


def _compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
) -> tuple[float, float, float]:
    """Compute accuracy, macro-F1, and balanced accuracy without scikit-learn warnings."""

    if y_true.size == 0:
        return 0.0, 0.0, 0.0

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    accuracy = float(np.mean(y_true == y_pred))

    truth = pd.Series(y_true, name="true")
    preds = pd.Series(y_pred, name="pred")
    all_classes = np.unique(np.concatenate((classes, np.unique(y_pred))))
    confusion = pd.crosstab(truth, preds, dropna=False)
    confusion = confusion.reindex(index=all_classes, columns=all_classes, fill_value=0)
    confusion_values = confusion.to_numpy(dtype=float)

    tp = np.diag(confusion_values)
    support = confusion_values.sum(axis=1)
    predicted = confusion_values.sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.divide(tp, support, out=np.zeros_like(tp), where=support != 0)
        precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted != 0)
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
        )

    macro_f1 = float(f1.mean()) if f1.size else 0.0
    balanced_accuracy = float(recall.mean()) if recall.size else 0.0

    return accuracy, macro_f1, balanced_accuracy


def _summarize_dataset(df: pd.DataFrame, name: str) -> DatasetSummary:
    feature_columns = [col for col in df.columns if col.startswith("PC")]
    counts = Counter(df["Subpopulation"].tolist())
    return DatasetSummary(
        name=name,
        num_samples=len(df),
        num_features=len(feature_columns),
        class_counts=counts,
    )


def _build_models() -> list[tuple[str, Callable[[], BaseEstimator]]]:
    return [
        (
            "LogisticRegression",
            lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000),
            ),
        ),
        ("KNeighborsClassifier",
         lambda: make_pipeline(
             StandardScaler(),
             KNeighborsClassifier(n_neighbors=10, n_jobs=-1),
         )),
        ("RandomForestClassifier",
         lambda: RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1)),
        (
            "XGBClassifier",
            lambda: XGBClassifier(
                random_state=42,
                eval_metric="mlogloss",
                use_label_encoder=False,
            ),
        ),
        (
            "SVC",
            lambda: make_pipeline(
                StandardScaler(),
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    decision_function_shape="ovr",
                    random_state=42,
                ),
            ),
        ),
    ]


def corrected_resampled_t(diff: np.ndarray, k: int, r: int = 1) -> tuple[float, float]:
    """Nadeau–Bengio corrected resampled t-test for dependent K-fold differences.

    diff: array of per-fold (and per-repeat) differences (len = r*k).
    k: number of folds; r: number of repeats.
    Returns (t_stat, two-sided p-value), using Student-t with df=r*k - 1.
    """
    diff = np.asarray(diff, dtype=float)
    dbar = float(diff.mean())
    s2 = float(diff.var(ddof=1)) if diff.size > 1 else 0.0
    var_corr = (1.0 / (r * k) + 1.0 / (k - 1.0)) * s2
    if var_corr == 0.0:
        if dbar == 0.0:
            return 0.0, 1.0
        return (np.inf if dbar > 0 else -np.inf), 0.0
    t = dbar / np.sqrt(var_corr)
    df = max(1, r * k - 1)
    p = 2.0 * (1.0 - student_t.cdf(abs(t), df=df))
    return t, p


def _evaluate_models(
    models: list[tuple[str, Callable[[], BaseEstimator]]],
    datasets: dict[str, pd.DataFrame],
    feature_columns: list[str],
    folds: int,
    random_state: int,
) -> list[dict[str, object]]:
    labels = list(datasets.keys())
    y = next(iter(datasets.values()))["Subpopulation"].to_numpy()
    classes = np.unique(y)
    label_to_index = {label: idx for idx, label in enumerate(classes)}
    y_encoded = np.array([label_to_index[label] for label in y], dtype=int)

    matrices = {
        label: df[feature_columns].to_numpy(dtype=float)
        for label, df in datasets.items()
    }

    for label, matrix in matrices.items():
        print(
            "  Matrix '{}' shape: {} samples × {} features".format(
                label, matrix.shape[0], matrix.shape[1]
            ),
            flush=True,
        )

    # With across-dataset filtering, smallest class size >= folds is guaranteed.
    min_class_count = next(iter(datasets.values()))["Subpopulation"].value_counts().min()
    if min_class_count < folds:
        raise ValueError(
            f"Cannot run {folds} folds: smallest class has only {min_class_count} samples after filtering."
        )

    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    splits = list(splitter.split(next(iter(matrices.values())), y_encoded))
    print(
        "  Generated {} stratified splits with fold sizes: {}".format(
            len(splits),
            [
                {
                    "train": int(train_idx.size),
                    "test": int(test_idx.size),
                }
                for (train_idx, test_idx) in splits
            ],
        ),
        flush=True,
    )

    results: list[dict[str, object]] = []
    for model_name, factory in models:
        print(f"\nEvaluating {model_name} with {folds}-fold stratified CV (seed={random_state})…", flush=True)
        # Track per-fold accuracy (used for tests); log macroF1 & balanced acc too.
        score_tracker: dict[str, list[float]] = {label: [] for label in labels}

        for fold_index, (train_idx, test_idx) in enumerate(splits, start=1):
            print(f"  Fold {fold_index:02d} results:", flush=True)
            for label in labels:
                model = factory()
                X = matrices[label]
                model.fit(X[train_idx], y_encoded[train_idx])
                preds = np.asarray(model.predict(X[test_idx]))

                if preds.dtype.kind in {"U", "S", "O"}:
                    predicted_labels = preds.astype(str)
                else:
                    predicted_labels = classes[preds.astype(int)]

                acc, mac_f1, bal_acc = _compute_classification_metrics(
                    y[test_idx], predicted_labels, classes
                )

                score_tracker[label].append(acc)
                print(
                    f"    {label:<20} acc = {acc:.4f}  macroF1 = {mac_f1:.4f}  balAcc = {bal_acc:.4f}",
                    flush=True,
                )

        dataset_summaries: dict[str, dict[str, object]] = {}
        for label, scores in score_tracker.items():
            arr = np.array(scores, dtype=float)
            dataset_summaries[label] = {
                "scores": arr,
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            }
            print(
                "  Summary for {}: mean = {:.4f}, std = {:.4f}".format(
                    label, dataset_summaries[label]["mean"], dataset_summaries[label]["std"]
                ),
                flush=True,
            )

        comparisons: list[dict[str, object]] = []
        for a_label, b_label in itertools.combinations(labels, 2):
            scores_a = dataset_summaries[a_label]["scores"]
            scores_b = dataset_summaries[b_label]["scores"]
            diff = scores_b - scores_a
            mean_diff = float(diff.mean())
            wins = int(np.sum(diff > 0))
            losses = int(np.sum(diff < 0))
            ties = int(diff.size - wins - losses)

            effect_size = mean_diff
            if diff.size > 1:
                diff_std = float(diff.std(ddof=1))
                if diff_std > 0.0:
                    effect_size = mean_diff / diff_std

            # Dependence-aware test
            t_corr, p_corr = corrected_resampled_t(diff, k=folds, r=1)

            comparisons.append(
                {
                    "a": a_label,
                    "b": b_label,
                    "mean_diff": mean_diff,
                    "effect_size": float(effect_size),
                    "wins": wins,
                    "losses": losses,
                    "ties": ties,
                    "t_stat_corrected": float(t_corr),
                    "p_value_corrected": float(p_corr),
                }
            )
            print(
                (
                    "  Corrected paired test ({} - {}): diff = {:.4f}"
                    " [wins={}, losses={}, ties={}], effect = {:.4f},"
                    " t = {:.4f}, p = {:.6f}"
                ).format(
                    b_label,
                    a_label,
                    mean_diff,
                    wins,
                    losses,
                    ties,
                    effect_size,
                    t_corr,
                    p_corr,
                ),
                flush=True,
            )

        results.append(
            {
                "model": model_name,
                "folds": folds,
                "datasets": dataset_summaries,
                "comparisons": comparisons,
            }
        )

    return results


def _print_summary(results: list[dict[str, object]]) -> None:
    print("\n===== Final Summary =====", flush=True)
    for entry in results:
        print(
            f"Model: {entry['model']} (folds={entry['folds']})",
            flush=True,
        )
        print("  Dataset accuracies:", flush=True)
        for label, stats_dict in entry["datasets"].items():
            print(
                f"    {label:<20} mean = {stats_dict['mean']:.4f}, std = {stats_dict['std']:.4f}",
                flush=True,
            )
        print("  Pairwise comparisons (B - A), corrected t-test:", flush=True)
        for comp in entry["comparisons"]:
            print(
                (
                    f"    {comp['b']} - {comp['a']:<16} diff = {comp['mean_diff']:.4f},"
                    f" effect = {comp['effect_size']:.4f},"
                    f" wins/losses/ties = {comp['wins']}/{comp['losses']}/{comp['ties']},"
                    f" t = {comp['t_stat_corrected']:.4f}, p = {comp['p_value_corrected']:.6f}"
                ),
                flush=True,
            )
        print("", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    print(
        "Received {} projection arguments: {}".format(
            len(args.projections),
            [label for label, _ in args.projections],
        ),
        flush=True,
    )
    print(
        f"Configured to run with folds={args.folds} and random_state={args.random_state}.",
        flush=True,
    )

    mapping = _load_sample_population_mapping()

    datasets: dict[str, pd.DataFrame] = {}
    for label, path in args.projections:
        datasets[label] = _filter_projection(
            _read_projection(path, label, mapping), label
        )
        print(
            "Dataset '{}' ready with {} rows post-filtering.".format(
                label, len(datasets[label])
            ),
            flush=True,
        )

    aligned = _align_datasets(datasets)

    # Guarantee we can run the requested number of folds by filtering *across datasets*.
    min_required = max(2, args.folds)
    filtered_datasets = _filter_classes_across_datasets(aligned, min_size=min_required)

    feature_columns = _determine_feature_columns(filtered_datasets)

    # Preserve identical sample order across datasets
    retained_samples = filtered_datasets[next(iter(filtered_datasets))].index.tolist()
    trimmed_datasets: dict[str, pd.DataFrame] = {
        label: df.loc[retained_samples, ["SampleName", "Subpopulation", *feature_columns]].copy()
        for label, df in filtered_datasets.items()
    }

    if not trimmed_datasets or next(iter(trimmed_datasets.values())).empty:
        raise ValueError("No samples remain after filtering invalid labels and small classes")

    for label, df in trimmed_datasets.items():
        print(_summarize_dataset(df, label).describe(), flush=True)

    models = _build_models()
    results = _evaluate_models(models, trimmed_datasets, feature_columns, args.folds, args.random_state)
    _print_summary(results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
