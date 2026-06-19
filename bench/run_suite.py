#!/usr/bin/env python3
import typing
from typing import TYPE_CHECKING
import argparse
from collections import deque
import importlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from shutil import disk_usage
from time import monotonic, perf_counter

# Statistical-regression gate. Lives in bench/gate.py so it can
# be invoked stand-alone post-hoc as well. Loaded lazily-safely: if the
# module is unavailable for any reason the gate just stays quiet.
try:
    _gate_path = Path(__file__).resolve().parent / "gate.py"
    _gate_spec = importlib.util.spec_from_file_location("_bench_gate", _gate_path)
    if _gate_spec is None or _gate_spec.loader is None:
        raise ImportError(f"failed to load statistical gate from {_gate_path}")
    _gate_module = importlib.util.module_from_spec(_gate_spec)
    _gate_spec.loader.exec_module(_gate_module)
    _gate_extract_fit_quality = _gate_module.extract_fit_quality
    _gate_cmd_check_results = _gate_module.cmd_check_results
except Exception:  # pragma: no cover - bench infra only
    _gate_extract_fit_quality = None
    _gate_cmd_check_results = None


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return int(default)
    try:
        return int(raw.strip())
    except Exception:
        return int(default)


# Optional strict serial mode for reproducibility/fairness studies.
# Default is parallel-friendly so the Rust contender can use Rayon threads.
_FORCE_SERIAL = _env_flag("BENCH_FORCE_SERIAL", default=False)
_RAYON_THREADS = _env_int("BENCH_RAYON_THREADS", 0)
_BLAS_THREADS = _env_int("BENCH_BLAS_THREADS", 0)
_CMD_TIMEOUT_SEC = _env_int("BENCH_CMD_TIMEOUT_SEC", 0)
_SERIAL_ENV_OVERRIDES = (
    {
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "RAYON_NUM_THREADS": "1",
        "CARGO_BUILD_JOBS": "1",
        "OMP_DYNAMIC": "FALSE",
        "MKL_DYNAMIC": "FALSE",
    }
    if _FORCE_SERIAL
    else {}
)
_EXPECTED_EXTERNAL_FAILURES = (RuntimeError, FileNotFoundError)
_EXPECTED_OPTIONAL_IMPORT_FAILURES = (ImportError,)
_EXPECTED_JSON_ARTIFACT_FAILURES = (OSError, json.JSONDecodeError, TypeError, ValueError)
if not _FORCE_SERIAL:
    # Explicit thread controls for performance runs.
    # 0 or unset means "leave tool defaults unchanged".
    if _RAYON_THREADS > 0:
        _SERIAL_ENV_OVERRIDES["RAYON_NUM_THREADS"] = str(_RAYON_THREADS)
    if _BLAS_THREADS > 0:
        _SERIAL_ENV_OVERRIDES["OMP_NUM_THREADS"] = str(_BLAS_THREADS)
        _SERIAL_ENV_OVERRIDES["OPENBLAS_NUM_THREADS"] = str(_BLAS_THREADS)
        _SERIAL_ENV_OVERRIDES["MKL_NUM_THREADS"] = str(_BLAS_THREADS)
        _SERIAL_ENV_OVERRIDES["VECLIB_MAXIMUM_THREADS"] = str(_BLAS_THREADS)
        _SERIAL_ENV_OVERRIDES["NUMEXPR_NUM_THREADS"] = str(_BLAS_THREADS)
        _SERIAL_ENV_OVERRIDES["BLIS_NUM_THREADS"] = str(_BLAS_THREADS)
for _k, _v in _SERIAL_ENV_OVERRIDES.items():
    os.environ[_k] = _v

# numpy and pandas are loaded via importlib (not `import ... as ...`) because
# threading env vars above must take effect before these C extensions initialise
# their BLAS/OpenMP pools at import time. Static analyzers still need to see the
# regular import so type annotations like ``pd.DataFrame`` resolve.
if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
else:
    np = importlib.import_module("numpy")
    pd = importlib.import_module("pandas")

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "bench"
DEFAULT_SCENARIOS = BENCH_DIR / "scenarios.json"
DATASET_DIR = BENCH_DIR / "datasets"


def _import_run_suite_exports(stem: str) -> typing.Any:
    module_path = BENCH_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"bench.{stem}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"failed to load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configure = getattr(module, "configure", None)
    if configure is not None:
        configure(globals())
    for name, value in vars(module).items():
        if name.startswith("__") or name in {"configure", "typing", "np", "pd"}:
            continue
        globals()[name] = value
    return module
CV_SPLITS = 5
CV_SEED = 42
_RUST_BIN_PATH: Path | None = None
_BENCH_RUST_LOADER: typing.Any = None
HEARTBEAT_INTERVAL_SEC = 15.0
# For short-lived commands, poll more frequently at startup so heartbeat
# diagnostics capture meaningful process stats before exit.
HEARTBEAT_INITIAL_WINDOW_SEC = 2.0
HEARTBEAT_INITIAL_INTERVAL_SEC = 0.25
_MAX_CAPTURE_CHARS = 200000
_OUTPUT_LOCK = threading.Lock()


def _load_bench_rust_loader() -> typing.Any:
    """Load `bench/_rust_loader.py` lazily and cache it."""

    global _BENCH_RUST_LOADER
    if _BENCH_RUST_LOADER is not None:
        return _BENCH_RUST_LOADER
    loader_path = BENCH_DIR / "_rust_loader.py"
    spec = importlib.util.spec_from_file_location("bench_rust_loader", loader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load bench rust loader from {loader_path}")
    loader_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader_mod)
    _BENCH_RUST_LOADER = loader_mod
    return loader_mod


def _gamfit_rust() -> typing.Any:
    """Load gamfit's Rust extension without importing the package facade.

    The .so/.pyd may live in the source tree (after `maturin develop`),
    in a pip-installed wheel under site-packages (CI's path), or in any
    other location on the sys.path / installed-distribution search list.
    The bench-shared loader enumerates every plausible location so it
    finds the compiled extension even when the source tree shadows the
    pip-installed wheel on the import path (which is exactly what
    `bench/fuzz_vs_mgcv.py`'s `sys.path.insert(0, str(ROOT))` triggers).
    The package facade in `gamfit/__init__.py` is deliberately bypassed
    by loading the `.so` directly via spec_from_file_location, so the
    heavy import-time work the fuzz scripts avoid stays out of the path.
    """

    return _load_bench_rust_loader().load_gamfit_rust_module(ROOT)


def _flat_float_list(values: typing.Any) -> list[float]:
    return np.asarray(values, dtype=float).reshape(-1).tolist()


def _sigma_float_list(values: typing.Any) -> list[float]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return [float(arr)]
    return arr.reshape(-1).tolist()


class _TerminalOutputSanitizer:
    def __init__(self) -> None:
        self._state = "normal"
        self._line_start = True
        self._pending_indent = ""

    def feed(self, text: str) -> str:
        out: list[str] = []
        for ch in text:
            state = self._state
            if state == "normal":
                if ch == "\x1b":
                    self._state = "esc"
                elif ch == "\r":
                    self._emit(out, "\n")
                elif ch in "\n\t" or (ord(ch) >= 0x20 and ch != "\x7f"):
                    self._emit(out, ch)
            elif state == "esc":
                if ch == "[":
                    self._state = "csi"
                elif ch == "]":
                    self._state = "osc"
                elif ch in "PX^_":
                    self._state = "string"
                else:
                    self._state = "normal"
            elif state == "csi":
                if "@" <= ch <= "~":
                    self._state = "normal"
            elif state == "osc":
                if ch == "\x07":
                    self._state = "normal"
                elif ch == "\x1b":
                    self._state = "osc_esc"
            elif state == "osc_esc":
                self._state = "normal" if ch == "\\" else "osc"
            elif state == "string":
                if ch == "\x1b":
                    self._state = "string_esc"
            elif state == "string_esc":
                self._state = "normal" if ch == "\\" else "string"
        return "".join(out)

    def flush(self) -> str:
        tail = self._pending_indent
        self._state = "normal"
        self._line_start = True
        self._pending_indent = ""
        return tail

    def _emit(self, out: list[str], ch: str) -> None:
        if self._line_start and ch in " \t":
            self._pending_indent += ch
            return
        if self._line_start:
            if ch == "[":
                self._pending_indent = ""
            else:
                out.append(self._pending_indent)
                self._pending_indent = ""
            self._line_start = False
        out.append(ch)
        if ch == "\n":
            self._line_start = True


def _write_stream(sink: typing.Any, text: str) -> None:
    if not text:
        return
    with _OUTPUT_LOCK:
        sink.write(text)
        sink.flush()


def _print_stderr(message: str) -> None:
    _write_stream(sys.stderr, f"{message}\n")


def _require_lifelines_survival_helpers() -> typing.Any:
    lifelines: typing.Any = importlib.import_module("lifelines")
    CoxPHFitter = lifelines.CoxPHFitter
    KaplanMeierFitter = lifelines.KaplanMeierFitter

    return CoxPHFitter, KaplanMeierFitter


def _require_lifelines_coxph() -> typing.Any:
    lifelines: typing.Any = importlib.import_module("lifelines")
    lifelines_exceptions: typing.Any = importlib.import_module("lifelines.exceptions")
    CoxPHFitter = lifelines.CoxPHFitter
    ConvergenceWarning = lifelines_exceptions.ConvergenceWarning

    return CoxPHFitter, ConvergenceWarning


def _require_lifelines_aft_fitters() -> typing.Any:
    lifelines: typing.Any = importlib.import_module("lifelines")
    lifelines_exceptions: typing.Any = importlib.import_module("lifelines.exceptions")
    LogNormalAFTFitter = lifelines.LogNormalAFTFitter
    WeibullAFTFitter = lifelines.WeibullAFTFitter
    ConvergenceWarning = lifelines_exceptions.ConvergenceWarning

    return LogNormalAFTFitter, WeibullAFTFitter, ConvergenceWarning


def _require_lifelines_kaplan_meier() -> typing.Any:
    lifelines: typing.Any = importlib.import_module("lifelines")
    KaplanMeierFitter = lifelines.KaplanMeierFitter

    return KaplanMeierFitter


def _require_lifelines_concordance_index() -> typing.Any:
    lifelines_utils: typing.Any = importlib.import_module("lifelines.utils")
    concordance_index = lifelines_utils.concordance_index

    return concordance_index


NON_BLOCKING_FAILURE_CONTENDERS = {
    # These external stacks are kept in the benchmark output for visibility,
    # but occasional fit/predict failures should not fail the whole CI shard.
    "r_gamlss",
    "rust_gamlss",
    "rust_gamlss_flexible",
    "rust_gamlss_marginal_slope",
    "rust_gamlss_marginal_slope_aniso",
    "rust_gamlss_survival_marginal_slope",
    # rust_gam_flexible drives the same custom-family / flexible-link code path
    # as rust_gamlss_flexible above; it is a diagnostic companion to rust_gam,
    # not the system under test. Keep its rows in the output for visibility,
    # but don't fail the shard when the flexible-link inner solve struggles.
    "rust_gam_flexible",
    "r_gamboostlss",
    "r_bamlss",
    "r_brms",
    # mgcv is an external comparison reference, not the system under test.
    # It deterministically cannot construct the requested basis on some
    # high-dimensional joint-PC scenarios (e.g. 16-D thin-plate where the
    # default null-space dimension M = C(d+m-1, m-1) far exceeds any
    # tractable k, producing the well-known "negative length vectors are
    # not allowed" allocation error in the tprs/Duchon constructor).
    # Surface those rows in the diagnostic output, but don't fail the
    # shard for an external-library limitation we don't control.
    "r_mgcv",
    "r_mgcv_gaulss",
    "r_mgcv_coxph",
    # External survival comparison references, none of which is the system
    # under test (that is `rust_gamlss_survival`, gated separately by the
    # required-contender check below). Like the mgcv lanes above, these
    # third-party stacks routinely hit numerical failures on degenerate
    # survival data we don't control — e.g. R `survival::coxph` aborting with
    # "NA/NaN/Inf in foreign function call (arg 5)" / "NaNs produced" inside
    # `coxpenal.fit` on tied/ill-conditioned times. Surface those rows in the
    # diagnostic output, but don't fail the whole shard for an external
    # library's fit failure.
    "r_survival_coxph",
    "r_glmnet_cox",
    "python_sksurv_rsf",
    "python_sksurv_coxnet",
    "python_sksurv_gb_coxph",
    "python_sksurv_componentwise_gb_coxph",
    "python_lifelines_coxph_enet",
    "python_lifelines_weibull_aft",
    "python_lifelines_lognormal_aft",
    "python_xgboost_aft",
}
_BENCH_CI_PROFILE = os.environ.get("BENCH_CI_PROFILE", "full").strip().lower() or "full"
_LEAN_PROFILE_EXCLUDED_CONTENDERS = {
    # These contenders materially increase CI runtime and peak memory on
    # GitHub-hosted runners. Keep them in the nightly/full profile.
    "r_bamlss",
    "r_brms",
    "python_sksurv_rsf",
    "python_sksurv_gb_coxph",
    "python_sksurv_componentwise_gb_coxph",
    "python_lifelines_weibull_aft",
    "python_lifelines_lognormal_aft",
    "python_xgboost_aft",
}
@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


@dataclass(frozen=True)
class SharedFoldArtifact:
    fold_id: int
    train_scaled_csv: Path
    test_scaled_csv: Path
    train_idx_path: Path
    test_idx_path: Path


_import_run_suite_exports("_run_suite_datasets")


def folds_for_dataset(ds: dict[str, typing.Any]) -> list[Fold]:
    n_splits = int(ds.get("_cv_splits", CV_SPLITS))
    if ds["family"] == "survival":
        _coerce_positive_survival_dataset_inplace(ds, dataset_name=ds.get("name", "survival_dataset"))
        y = np.array([float(r[ds["event_col"]]) for r in ds["rows"]], dtype=float)
        stratified = True
    else:
        y = np.array([float(r[ds["target"]]) for r in ds["rows"]], dtype=float)
        stratified = ds["family"] == "binomial"
    if ds["family"] == "binomial":
        y_bin = (y > 0.5).astype(int)
        n_pos = int(np.sum(y_bin == 1))
        n_neg = int(np.sum(y_bin == 0))
        min_class = min(n_pos, n_neg)
        min_required = 2 if n_splits == 1 else n_splits
        if min_class < min_required:
            raise RuntimeError(
                "invalid binomial CV configuration: "
                f"requested {n_splits} "
                f"{'holdout split' if n_splits == 1 else 'folds'} but class counts are "
                f"positives={n_pos}, negatives={n_neg}. "
                + (
                    "Each class must have at least two observations for a stratified holdout split. "
                    if n_splits == 1
                    else "Each class must have at least one observation per fold. "
                )
                + "Increase n or reduce the number of folds."
            )
    folds = make_folds(y, n_splits=n_splits, seed=CV_SEED, stratified=stratified)
    n = len(ds["rows"])
    seen = np.zeros(n, dtype=int)
    for f in folds:
        tr = np.asarray(f.train_idx, dtype=int)
        te = np.asarray(f.test_idx, dtype=int)
        if tr.size == 0 or te.size == 0:
            raise RuntimeError("invalid CV split: empty train or test fold")
        if np.intersect1d(tr, te).size != 0:
            raise RuntimeError("invalid CV split: train/test overlap detected")
        if np.any(tr < 0) or np.any(te < 0) or np.any(tr >= n) or np.any(te >= n):
            raise RuntimeError("invalid CV split: index out of bounds")
        seen[te] += 1
    if len(folds) == 1:
        if np.any(seen > 1):
            raise RuntimeError("invalid holdout split: a row appears more than once in the test set")
    elif not np.all(seen == 1):
        raise RuntimeError("invalid CV split: each row must appear exactly once in test sets")
    return folds


def build_shared_fold_artifacts(
    ds: dict[str, typing.Any],
    folds: list[Fold],
    root_dir: Path,
) -> list[SharedFoldArtifact]:
    root_dir.mkdir(parents=True, exist_ok=True)
    base_df = pd.DataFrame(ds["rows"])
    feature_cols = list(ds.get("features", []))
    artifacts: list[SharedFoldArtifact] = []
    for fold_id, fold in enumerate(folds):
        train_idx_path = root_dir / f"train_idx_{fold_id}.txt"
        test_idx_path = root_dir / f"test_idx_{fold_id}.txt"
        train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
        test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

        train_df = base_df.iloc[fold.train_idx].copy()
        test_df = base_df.iloc[fold.test_idx].copy()
        if feature_cols:
            train_df, test_df = zscore_train_test(train_df, test_df, feature_cols)
        train_csv = root_dir / f"train_scaled_{fold_id}.csv"
        test_csv = root_dir / f"test_scaled_{fold_id}.csv"
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        artifacts.append(
            SharedFoldArtifact(
                fold_id=fold_id,
                train_scaled_csv=train_csv,
                test_scaled_csv=test_csv,
                train_idx_path=train_idx_path,
                test_idx_path=test_idx_path,
            )
        )
    return artifacts


def aggregate_cv_rows(cv_rows: typing.Any, family: typing.Any) -> typing.Any:
    fit_sec = float(sum(float(r["fit_sec"]) for r in cv_rows))
    predict_sec = float(sum(float(r["predict_sec"]) for r in cv_rows))

    def wavg(key: typing.Any) -> typing.Any:
        if key == "mse":
            valid_rmse = [(r.get("rmse"), int(r["n_test"])) for r in cv_rows if r.get("rmse") is not None]
            if valid_rmse:
                denom = max(sum(w for _, w in valid_rmse), 1)
                return float(sum((float(v) ** 2) * w for v, w in valid_rmse) / denom)
        valid = [(r.get(key), int(r["n_test"])) for r in cv_rows if r.get(key) is not None]
        if not valid:
            return None
        denom = max(sum(w for _, w in valid), 1)
        return float(sum(float(v) * w for v, w in valid) / denom)

    if family == "binomial":
        return {
            "fit_sec": fit_sec,
            "predict_sec": predict_sec,
            "auc": wavg("auc"),
            "brier": wavg("brier"),
            "logloss": wavg("logloss"),
            "nagelkerke_r2": wavg("nagelkerke_r2"),
            "mse": None,
            "rmse": None,
            "mae": None,
            "r2": None,
        }
    if family == "survival":
        return {
            "fit_sec": fit_sec,
            "predict_sec": predict_sec,
            "auc": wavg("auc"),  # concordance index
            "brier": wavg("brier"),
            "logloss": wavg("logloss"),
            "c_index": wavg("auc"),
            "nagelkerke_r2": wavg("nagelkerke_r2"),
            "mse": None,
            "rmse": None,
            "mae": None,
            "r2": None,
        }
    return {
        "fit_sec": fit_sec,
        "predict_sec": predict_sec,
        "auc": None,
        "brier": None,
        "logloss": wavg("logloss"),
        "nagelkerke_r2": None,
        "mse": wavg("mse"),
        "rmse": wavg("rmse"),
        "mae": wavg("mae"),
        "r2": wavg("r2"),
    }


def _init_plot_payload(ds: dict[str, typing.Any]) -> dict[str, typing.Any]:
    family = str(ds["family"])
    features = [str(c) for c in ds.get("features", [])]
    primary_feature = features[0] if len(features) == 1 else None
    payload = {
        "family": family,
        "features": features,
        "primary_feature": primary_feature,
        "x": [],
    }
    if family == "survival":
        payload.update(
            {
                "time_col": str(ds["time_col"]),
                "event_col": str(ds["event_col"]),
                "time": [],
                "event": [],
                "risk": [],
                "pred_time": [],
            }
        )
    else:
        payload.update(
            {
                "target": str(ds["target"]),
                "actual": [],
                "predicted": [],
            }
        )
    return payload


def _append_supervised_plot_fold(payload: dict[str, typing.Any], test_df: pd.DataFrame, pred: np.ndarray, target_col: str) -> None:
    y_true = test_df[target_col].to_numpy(dtype=float).reshape(-1)
    y_pred = np.asarray(pred, dtype=float).reshape(-1)
    if y_true.shape[0] != y_pred.shape[0]:
        raise RuntimeError(
            f"plot payload supervised fold length mismatch: actual={y_true.shape[0]}, predicted={y_pred.shape[0]}"
        )
    payload["actual"].extend(float(v) for v in y_true)
    payload["predicted"].extend(float(v) for v in y_pred)
    primary_feature = payload.get("primary_feature")
    if primary_feature:
        x = test_df[str(primary_feature)].to_numpy(dtype=float).reshape(-1)
        payload["x"].extend(float(v) for v in x)


def _append_survival_plot_fold(
    payload: dict[str, typing.Any],
    test_df: pd.DataFrame,
    *,
    time_col: str,
    event_col: str,
    risk_score: np.ndarray,
    pred_time: np.ndarray | None = None,
) -> None:
    times = test_df[time_col].to_numpy(dtype=float).reshape(-1)
    events = test_df[event_col].to_numpy(dtype=float).reshape(-1)
    risk = np.asarray(risk_score, dtype=float).reshape(-1)
    if times.shape[0] != risk.shape[0] or events.shape[0] != risk.shape[0]:
        raise RuntimeError(
            "plot payload survival fold length mismatch: "
            f"time={times.shape[0]}, event={events.shape[0]}, risk={risk.shape[0]}"
        )
    payload["time"].extend(float(v) for v in times)
    payload["event"].extend(float(v) for v in events)
    payload["risk"].extend(float(v) for v in risk)
    primary_feature = payload.get("primary_feature")
    if primary_feature:
        x = test_df[str(primary_feature)].to_numpy(dtype=float).reshape(-1)
        payload["x"].extend(float(v) for v in x)
    if pred_time is not None:
        pred_time_arr = np.asarray(pred_time, dtype=float).reshape(-1)
        if pred_time_arr.shape[0] != risk.shape[0]:
            raise RuntimeError(
                f"plot payload predicted-time length mismatch: pred_time={pred_time_arr.shape[0]}, risk={risk.shape[0]}"
            )
        payload["pred_time"].extend(float(v) for v in pred_time_arr)


def _finalize_plot_payload(payload: dict[str, typing.Any]) -> dict[str, typing.Any] | None:
    family = payload.get("family")
    if family == "survival":
        n = len(payload.get("time", []))
        if n == 0:
            return None
        if not payload.get("pred_time"):
            payload["pred_time"] = None
        return payload
    n = len(payload.get("actual", []))
    if n == 0:
        return None
    return payload


def _finalize_cv_result(
    *,
    contender: str,
    scenario_name: str,
    family: str,
    cv_rows: list[dict[str, typing.Any]],
    plot_payload: dict[str, typing.Any] | None,
    model_spec: str,
    extra_metrics: dict[str, typing.Any] | None = None,
) -> dict[str, typing.Any]:
    metrics = aggregate_cv_rows(cv_rows, family)
    if extra_metrics:
        metrics.update(extra_metrics)
    n_folds = len(cv_rows)
    reserved_keys = {
        "contender",
        "family",
        "scenario_name",
        "status",
        "n_folds",
        "evaluation",
        "model_spec",
        "plot_payload",
    }
    conflicting_keys = sorted(reserved_keys.intersection(metrics))
    if conflicting_keys:
        raise RuntimeError(
            "aggregate_cv_rows/extra_metrics attempted to overwrite reserved result keys: "
            + ", ".join(conflicting_keys)
        )
    result = {
        "contender": contender,
        "family": family,
        "scenario_name": scenario_name,
        "status": "ok",
        "n_folds": int(n_folds),
        "model_spec": model_spec,
        "plot_payload": _finalize_plot_payload(plot_payload) if plot_payload is not None else None,
    }
    result.update(metrics)
    result["evaluation"] = _evaluation_label_for_n_folds(n_folds)
    # Aggregate per-fold fit_quality (final_neg_v, edf_per_term) for the
    # statistical-regression gate. Lanes without it (R / non-rust / paths
    # that don't load model.json) leave the field at None and the gate
    # skips them.
    fit_quality_aggr = _aggregate_fit_quality_rows(cv_rows)
    if fit_quality_aggr is not None:
        result["fit_quality"] = fit_quality_aggr
    return result


def _aggregate_fit_quality_rows(cv_rows: list[dict[str, typing.Any]]) -> dict[str, typing.Any] | None:
    """Mean-aggregate per-fold ``fit_quality`` into a lane-level summary.

    Each fold's ``fit_quality`` is either ``None`` or
    ``{"final_neg_v": float, "edf_per_term": {name: float}}``. The
    lane-level result reports unweighted means across folds plus a
    ``per_fold`` array so the gate can drill in if needed.
    """
    fqs: list[dict[str, typing.Any]] = []
    for r in cv_rows:
        fit_quality = r.get("fit_quality")
        if isinstance(fit_quality, dict):
            fqs.append(fit_quality)
    if not fqs:
        return None
    neg_vs = [float(fq["final_neg_v"]) for fq in fqs if isinstance(fq.get("final_neg_v"), (int, float))]
    edf_acc: dict[str, list[float]] = {}
    for fq in fqs:
        edf = fq.get("edf_per_term")
        if not isinstance(edf, dict):
            continue
        for k, v in edf.items():
            if not isinstance(v, (int, float)):
                continue
            edf_acc.setdefault(str(k), []).append(float(v))
    out: dict[str, typing.Any] = {"n_folds_with_quality": int(len(fqs))}
    if neg_vs:
        out["final_neg_v"] = float(sum(neg_vs) / len(neg_vs))
    if edf_acc:
        out["edf_per_term"] = {k: float(sum(vs) / len(vs)) for k, vs in edf_acc.items()}
    out["per_fold"] = fqs
    return out if ("final_neg_v" in out or "edf_per_term" in out) else None


def _normalize_result_metadata(results: list[dict[str, typing.Any]]) -> None:
    for result in results:
        if result.get("status") != "ok":
            continue
        if result.get("evaluation") in (None, ""):
            n_folds = result.get("n_folds")
            if n_folds is None:
                continue
            try:
                result["evaluation"] = _evaluation_label_for_n_folds(int(n_folds))
            except Exception:
                pass


def _validate_result_metadata(results: list[dict[str, typing.Any]]) -> None:
    for result in results:
        if result.get("status") != "ok":
            continue
        spec = str(result.get("model_spec", ""))
        evaluation = str(result.get("evaluation", "")).lower()
        spec_lower = spec.lower()
        if evaluation == "holdout":
            ok = "holdout" in spec_lower
        elif "cv" in evaluation:
            ok = "cv" in spec_lower
        else:
            ok = False
        if not ok:
            raise SystemExit(
                "model result metadata/spec mismatch for "
                f"{result.get('contender')} / {result.get('scenario_name')}: "
                f"evaluation={result.get('evaluation')} spec={spec}"
            )


_import_run_suite_exports("_run_suite_formulas")


def _make_far_ood_frame(
    train_df: pd.DataFrame,
    *,
    ds: dict[str, typing.Any],
    smooth_cols: list[str],
    linear_cols: list[str],
    n_points: int = 64,
) -> pd.DataFrame:
    cols = list(ds["features"])
    if not cols:
        return pd.DataFrame()
    n = max(int(n_points), 8)
    rows = []
    for i in range(n):
        row = {}
        bitmask = i
        for j, c in enumerate(cols):
            vals = train_df[c].to_numpy(dtype=float)
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            span = max(vmax - vmin, 1e-6)
            far = max(4.0 * span, 6.0)
            if c in smooth_cols:
                go_high = ((bitmask >> (j % 16)) & 1) == 1
                row[c] = (vmax + far) if go_high else (vmin - far)
            elif c in linear_cols:
                row[c] = float(np.mean(vals))
            else:
                row[c] = float(np.mean(vals))
        if ds["family"] == "binomial":
            row[ds["target"]] = 0.0
        elif ds["family"] == "gaussian":
            row[ds["target"]] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _compute_continuous_order_from_lambdas(
    lambda_tilde: list[float] | tuple[float, ...] | np.ndarray,
    normalization_scale: list[float] | tuple[float, ...] | np.ndarray | None = None,
    eps: float = 1e-12,
) -> dict[str, typing.Any]:
    if len(lambda_tilde) < 3:
        return {
            "status": "UndefinedZeroLambda",
            "lambda0": float("nan"),
            "lambda1": float("nan"),
            "lambda2": float("nan"),
            "r_ratio": None,
            "nu": None,
            "kappa2": None,
        }
    lt = [float(lambda_tilde[0]), float(lambda_tilde[1]), float(lambda_tilde[2])]
    scales = [1.0, 1.0, 1.0] if normalization_scale is None else [float(normalization_scale[i]) for i in range(3)]
    if any((not np.isfinite(c)) or c <= 0.0 for c in scales):
        return {
            "status": "UndefinedZeroLambda",
            "lambda0": float("nan"),
            "lambda1": float("nan"),
            "lambda2": float("nan"),
            "r_ratio": None,
            "nu": None,
            "kappa2": None,
        }
    lam = [lt[i] / scales[i] for i in range(3)]
    l0, l1, l2 = lam
    if any(not np.isfinite(v) for v in lam):
        return {"status": "UndefinedZeroLambda", "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": None, "nu": None, "kappa2": None}
    l_scale = max(abs(l0), abs(l1), abs(l2), 1.0)
    l_floor = eps * l_scale
    if l0 <= l_floor:
        if l1 > l_floor and l2 > l_floor:
            return {"status": "IntrinsicLimit", "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": None, "nu": 1.0, "kappa2": 0.0}
        return {"status": "UndefinedZeroLambda", "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": None, "nu": None, "kappa2": None}
    if l2 <= l_floor:
        if l1 > l_floor and np.isfinite(l1):
            return {"status": "FirstOrderLimit", "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": None, "nu": 1.0, "kappa2": l0 / l1}
        return {"status": "UndefinedZeroLambda", "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": None, "nu": None, "kappa2": None}
    r_ratio = (l1 * l1) / (l0 * l2)
    if not np.isfinite(r_ratio):
        return {"status": "UndefinedZeroLambda", "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": None, "nu": None, "kappa2": None}
    disc = l1 * l1 - 4.0 * l0 * l2
    disc_tol = eps * l_scale * l_scale
    status = "NonMaternRegime" if disc < -disc_tol else "Ok"
    if r_ratio <= 2.0 + eps:
        return {"status": status, "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": r_ratio, "nu": None, "kappa2": None}
    nu = r_ratio / (r_ratio - 2.0)
    kappa2 = l1 / ((r_ratio - 2.0) * l2)
    if (not np.isfinite(nu)) or (not np.isfinite(kappa2)):
        return {"status": "UndefinedZeroLambda", "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": r_ratio, "nu": None, "kappa2": None}
    return {"status": status, "lambda0": l0, "lambda1": l1, "lambda2": l2, "r_ratio": r_ratio, "nu": float(nu), "kappa2": float(kappa2)}


def _rank_corr(a: np.ndarray, b: np.ndarray) -> float | None:
    x = np.asarray(a, dtype=float).reshape(-1)
    y = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(mask)) < 3:
        return None
    x = x[mask]
    y = y[mask]
    xr = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    yr = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    sx = float(np.std(xr))
    sy = float(np.std(yr))
    if sx < 1e-12 or sy < 1e-12:
        return None
    corr = float(np.corrcoef(xr, yr)[0, 1])
    if not np.isfinite(corr):
        return None
    return corr


def _thread3_cliff_gradient_magnitude(
    collocation_points: np.ndarray, *, feature_cols: list[int], ds: dict[str, typing.Any]
) -> np.ndarray | None:
    coeff_map = ds.get("thread3_cliff_coefficients")
    if not isinstance(coeff_map, dict):
        return None
    if collocation_points.ndim != 2 or collocation_points.shape[0] == 0:
        return None
    if collocation_points.shape[1] != len(feature_cols):
        return None
    coeff_vec = np.array(
        [float(coeff_map.get(ds["features"][int(c)], 0.0)) for c in feature_cols],
        dtype=float,
    )
    jump = float(ds.get("thread3_cliff_jump", 0.0))
    sharpness = float(ds.get("thread3_cliff_sharpness", 1.0))
    values = _gamfit_rust().thread3_cliff_gradient_magnitude(
        np.asarray(collocation_points, dtype=float),
        coeff_vec.reshape(-1).tolist(),
        jump,
        sharpness,
    )
    return None if values is None else np.asarray(values, dtype=float)


def _extract_thread3_adaptive_fold_metrics(model_payload: dict[str, typing.Any] | None, ds: dict[str, typing.Any]) -> dict[str, typing.Any]:
    if not isinstance(model_payload, dict):
        return {}
    diag = model_payload.get("adaptive_regularization_diagnostics")
    if not isinstance(diag, dict):
        return {}
    out: dict[str, float | int | bool] = {}
    mm_iter = diag.get("mm_iterations")
    if isinstance(mm_iter, (int, float)):
        out["mm_iterations"] = int(mm_iter)
    converged = diag.get("converged")
    if isinstance(converged, bool):
        out["adaptive_converged"] = converged

    maps = diag.get("maps")
    if not isinstance(maps, list):
        return out

    def _json_ndarray(value: typing.Any) -> typing.Any:
        if isinstance(value, dict) and isinstance(value.get("data"), list):
            arr = np.asarray(value.get("data"), dtype=float)
            dim = value.get("dim")
            if isinstance(dim, list) and dim:
                try:
                    shape = tuple(int(d) for d in dim)
                    if int(np.prod(shape, dtype=np.int64)) == int(arr.size):
                        return arr.reshape(shape)
                except Exception:
                    pass
            return arr
        return np.asarray(value, dtype=float)

    corr_rows = []
    for m in maps:
        if not isinstance(m, dict):
            continue
        try:
            feature_cols = [int(x) for x in (m.get("feature_cols") or [])]
            points = _json_ndarray(m.get("collocation_points"))
            inv_g = _json_ndarray(m.get("inv_grad_weight")).reshape(-1)
            inv_c = _json_ndarray(m.get("inv_lap_weight")).reshape(-1)
        except Exception:
            continue
        if points.ndim != 2 or points.shape[0] == 0:
            continue
        if points.shape[0] != inv_g.shape[0] or points.shape[0] != inv_c.shape[0]:
            continue
        grad_mag = _thread3_cliff_gradient_magnitude(points, feature_cols=feature_cols, ds=ds)
        if grad_mag is None or grad_mag.shape[0] != points.shape[0]:
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            sens_g = np.where(inv_g > 0.0, 1.0 / inv_g, np.nan)
            sens_c = np.where(inv_c > 0.0, 1.0 / inv_c, np.nan)
        corr_rows.append(
            {
                "n": int(points.shape[0]),
                "corr_g": _rank_corr(sens_g, grad_mag),
                "corr_c": _rank_corr(sens_c, grad_mag),
            }
        )

    if corr_rows:
        denom_parts: list[int] = []
        valid_g: list[tuple[float, int]] = []
        valid_c: list[tuple[float, int]] = []
        for row in corr_rows:
            n_raw = row.get("n")
            if n_raw is None:
                continue
            n_int = int(n_raw)
            denom_parts.append(n_int)
            corr_g = row.get("corr_g")
            if corr_g is not None:
                valid_g.append((float(corr_g), n_int))
            corr_c = row.get("corr_c")
            if corr_c is not None:
                valid_c.append((float(corr_c), n_int))
        denom = max(sum(denom_parts), 1)
        if valid_g:
            out["thread3_weight_grad_corr"] = float(
                sum(v * w for v, w in valid_g) / max(sum(w for _, w in valid_g), 1)
            )
        if valid_c:
            out["thread3_weight_curvature_corr"] = float(
                sum(v * w for v, w in valid_c) / max(sum(w for _, w in valid_c), 1)
            )
        out["thread3_collocation_points"] = int(denom)
    return out


def _rust_native_survival_matrix_from_model(
    *,
    rust_bin: Path,
    model_path: Path,
    predict_df: pd.DataFrame,
    time_col: str,
    grid: np.ndarray,
    fold_dir: Path,
) -> np.ndarray:
    n = len(predict_df)
    grid = np.asarray(grid, dtype=float).reshape(-1)
    if n == 0 or grid.size == 0:
        raise RuntimeError("native Rust survival matrix requires non-empty rows and time grid")

    stacked = pd.concat(
        [
            predict_df.assign(**{time_col: float(t), "__grid_id": int(j)})
            for j, t in enumerate(grid)
        ],
        axis=0,
        ignore_index=True,
    )
    stacked_path = fold_dir / "native_survival_input.csv"
    stacked_pred_path = fold_dir / "native_survival_pred.csv"
    stacked.to_csv(stacked_path, index=False)

    code, out, err = run_cmd(
        [
            str(rust_bin),
            "predict",
            str(model_path),
            str(stacked_path),
            "--out",
            str(stacked_pred_path),
        ],
        cwd=ROOT,
    )
    if code != 0:
        raise RuntimeError((err.strip() or out.strip() or "rust native survival predict failed"))

    pred_df = pd.read_csv(stacked_pred_path)
    if "survival_prob" not in pred_df.columns:
        raise RuntimeError("rust survival prediction output missing 'survival_prob' column")
    surv = pred_df["survival_prob"].to_numpy(dtype=float)
    expected = n * grid.size
    if surv.shape[0] != expected:
        raise RuntimeError(
            f"rust native survival predict length mismatch: got {surv.shape[0]}, expected {expected}"
        )
    return np.asarray(surv.reshape(grid.size, n).T, dtype=float)


def _ensure_rust_binary() -> typing.Any:
    global _RUST_BIN_PATH
    if _RUST_BIN_PATH is not None:
        return _RUST_BIN_PATH
    prebuilt = os.environ.get("BENCH_GAM_BIN", "").strip()
    if prebuilt:
        candidate = Path(prebuilt).expanduser().resolve()
        if not candidate.exists():
            raise RuntimeError(f"BENCH_GAM_BIN points to missing file: {candidate}")
        if not os.access(candidate, os.X_OK):
            raise RuntimeError(f"BENCH_GAM_BIN is not executable: {candidate}")
        _RUST_BIN_PATH = candidate
        return _RUST_BIN_PATH
    code, out, err = run_cmd(["cargo", "build", "--release", "--bin", "gam"], cwd=ROOT)
    if code != 0:
        raise RuntimeError((err or out).strip() or "failed to build Rust release binary")
    _RUST_BIN_PATH = ROOT / "target" / "release" / "gam"
    if not _RUST_BIN_PATH.exists():
        raise RuntimeError(f"missing Rust binary at {_RUST_BIN_PATH}")
    return _RUST_BIN_PATH


def run_rust_scenario_cv(
    scenario: typing.Any,
    *,
    contender_name: str = "rust_gam",
    binomial_link: str | None = None,
    ds: dict[str, typing.Any] | None = None,
    folds: list[Fold] | None = None,
    shared_fold_artifacts: list[SharedFoldArtifact] | None = None,
    rust_cfg_override: dict[str, typing.Any] | None = None,
    eval_ood: bool = False,
    collect_continuous_order: bool = False,
    collect_adaptive_diagnostics: bool = False,
    rust_fit_extra_args: list[str] | None = None,
    formula_link: str | None = None,
) -> typing.Any:
    scenario_name = scenario["name"]
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] == "survival":
        raise RuntimeError(
            "run_rust_scenario_cv does not support survival scenarios; "
            "use run_rust_gamlss_survival_cv instead"
        )
    if folds is None:
        folds = folds_for_dataset(ds)

    try:
        rust_bin = _ensure_rust_binary()
    except _EXPECTED_EXTERNAL_FAILURES as e:
        return {
            "contender": contender_name,
            "scenario_name": scenario_name,
            "status": "failed",
            "error": str(e),
        }

    base_df = pd.DataFrame(ds["rows"])
    cv_rows = []
    plot_payload = _init_plot_payload(ds)
    eval_suffix = _evaluation_suffix(folds)
    ood_rows = []
    continuous_rows = []
    adaptive_rows = []
    rust_cfg = _effective_scenario_fit_mapping(scenario_name, rust_cfg_override) or {}
    smooth_cols = list(rust_cfg.get("smooth_cols") or ([rust_cfg["smooth_col"]] if "smooth_col" in rust_cfg else []))
    linear_cols = list(rust_cfg.get("linear_cols", []))
    # Use a workspace-local temp root to reduce /tmp lifecycle flakiness in CI.
    with _workspace_tempdir(prefix="gam_bench_rust_cv_") as td:
        td_path = Path(td)
        for fold_id, fold in enumerate(folds):
            train_df = base_df.iloc[fold.train_idx].copy()
            test_df = base_df.iloc[fold.test_idx].copy()
            model_path = td_path / f"model_{fold_id}.json"
            pred_path = td_path / f"pred_{fold_id}.csv"
            shared_artifact = (
                shared_fold_artifacts[fold_id]
                if shared_fold_artifacts is not None and fold_id < len(shared_fold_artifacts)
                else None
            )

            if shared_artifact is not None:
                train_path = shared_artifact.train_scaled_csv
                test_path = shared_artifact.test_scaled_csv
            else:
                train_path = td_path / f"train_{fold_id}.csv"
                test_path = td_path / f"test_{fold_id}.csv"
                train_df, test_df = zscore_train_test(train_df, test_df, ds["features"])
                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)
            _family, formula = _rust_formula_for_scenario(
                scenario_name,
                ds,
                cfg_override=rust_cfg_override,
            )
            formula = (
                _append_formula_link_term(formula, binomial_link)
                if ds["family"] == "binomial" and binomial_link
                else formula
            )
            formula = _append_formula_link_term(formula, formula_link)
            fit_cmd = [
                str(rust_bin),
                "fit",
                "--out",
                str(model_path),
            ]

            if rust_fit_extra_args:
                fit_cmd.extend([str(x) for x in rust_fit_extra_args])
            fit_cmd.extend([str(train_path), formula])

            t0 = perf_counter()
            code, out, err = run_cmd(fit_cmd, cwd=ROOT)
            fit_sec = perf_counter() - t0
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "error": (err.strip() or out.strip() or "rust fit failed"),
                }
            model_payload = None
            try:
                model_payload = json.loads(model_path.read_text())
                if isinstance(model_payload, dict) and "payload" in model_payload:
                    model_payload = model_payload.get("payload", {})
            except Exception:
                model_payload = None
            # Gate: extract REML score + per-term edf from the rust model.json
            # while it is already in memory. Goes into the cv row and is
            # aggregated lane-wide by _finalize_cv_result.
            fit_quality_row: dict[str, typing.Any] | None = None
            if model_payload is not None and _gate_extract_fit_quality is not None:
                try:
                    fit_quality_row = _gate_extract_fit_quality({"payload": model_payload})
                except Exception:
                    fit_quality_row = None
            if collect_continuous_order and model_payload is not None:
                lambdas = model_payload.get("fit_result", {}).get("lambdas", [])
                if isinstance(lambdas, list) and len(lambdas) >= 3:
                    co = _compute_continuous_order_from_lambdas(lambdas[:3], normalization_scale=[1.0, 1.0, 1.0], eps=1e-12)
                    co["n_test"] = int(len(fold.test_idx))
                    continuous_rows.append(co)
            if collect_adaptive_diagnostics:
                adaptive_row = _extract_thread3_adaptive_fold_metrics(model_payload, ds)
                if adaptive_row:
                    adaptive_row["n_test"] = int(len(fold.test_idx))
                    adaptive_rows.append(adaptive_row)

            pred_cmd = [
                str(rust_bin),
                "predict",
                str(model_path),
                str(test_path),
                "--out",
                str(pred_path),
            ]
            t1 = perf_counter()
            code, out, err = run_cmd(pred_cmd, cwd=ROOT)
            pred_sec = perf_counter() - t1
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "error": (err.strip() or out.strip() or "rust predict failed"),
                }
            pred_df = pd.read_csv(pred_path)
            if "mean" not in pred_df.columns:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "error": "rust prediction output missing 'mean' column",
                }
            pred = pred_df["mean"].to_numpy(dtype=float)

            if ds["family"] == "binomial":
                y_test = test_df[ds["target"]].to_numpy(dtype=float)
                y_train = train_df[ds["target"]].to_numpy(dtype=float)
                _append_supervised_plot_fold(plot_payload, test_df, pred, ds["target"])
                fitted_formula = formula
                cv_rows.append(
                    {
                        "fit_sec": float(fit_sec),
                        "predict_sec": float(pred_sec),
                        "auc": auc_score(y_test, pred),
                        "brier": brier_score(y_test, pred),
                        "logloss": log_loss_score(y_test, pred),
                        "nagelkerke_r2": nagelkerke_r2_score(y_test, pred, null_mean=float(np.mean(y_train))),
                        "n_test": int(len(fold.test_idx)),
                        "model_spec": f"{fitted_formula} via release binary {eval_suffix}",
                        "fit_quality": fit_quality_row,
                    }
                )
                if eval_ood and smooth_cols:
                    ood_df = _make_far_ood_frame(
                        train_df,
                        ds=ds,
                        smooth_cols=smooth_cols,
                        linear_cols=linear_cols,
                        n_points=max(32, int(len(fold.test_idx))),
                    )
                    if not ood_df.empty:
                        ood_path = td_path / f"ood_{fold_id}.csv"
                        ood_pred_path = td_path / f"ood_pred_{fold_id}.csv"
                        ood_df.to_csv(ood_path, index=False)
                        ood_pred_cmd = [
                            str(rust_bin),
                            "predict",
                            str(model_path),
                            str(ood_path),
                            "--out",
                            str(ood_pred_path),
                        ]
                        code, out, err = run_cmd(ood_pred_cmd, cwd=ROOT)
                        if code == 0 and ood_pred_path.is_file():
                            ood_pred_df = pd.read_csv(ood_pred_path)
                            if "mean" in ood_pred_df.columns:
                                p_ood = np.clip(ood_pred_df["mean"].to_numpy(dtype=float), 1e-9, 1 - 1e-9)
                                baseline = float(np.clip(np.mean(y_test), 1e-9, 1 - 1e-9))
                                ood_rows.append(
                                    {
                                        "n_test": int(len(fold.test_idx)),
                                        "ood_abs_dev_from_baseline": float(np.mean(np.abs(p_ood - baseline))),
                                        "ood_max_abs_dev_from_baseline": float(np.max(np.abs(p_ood - baseline))),
                                        "ood_mean_abs_logit": float(np.mean(np.abs(np.log(p_ood / (1.0 - p_ood))))),
                                    }
                                )
            elif ds["family"] == "gaussian":
                y_test = test_df[ds["target"]].to_numpy(dtype=float)
                _append_supervised_plot_fold(plot_payload, test_df, pred, ds["target"])
                try:
                    if model_payload is None:
                        model_payload = json.loads(model_path.read_text())
                        if isinstance(model_payload, dict) and "payload" in model_payload:
                            model_payload = model_payload.get("payload", {})
                    sigma_hat_raw = model_payload.get("fit_result", {}).get(
                        "standard_deviation", None
                    )
                    sigma_hat = float(sigma_hat_raw)
                except _EXPECTED_JSON_ARTIFACT_FAILURES as e:
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario_name,
                        "status": "failed",
                        "fold_id": int(fold_id),
                        "n_train": int(len(fold.train_idx)),
                        "n_test": int(len(fold.test_idx)),
                        "n_folds": int(len(folds)),
                        "error": (
                            "rust gaussian fit output missing/invalid "
                            f"fit_result.standard_deviation: {e}"
                        ),
                    }
                if (not np.isfinite(sigma_hat)) or sigma_hat <= 0.0:
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario_name,
                        "status": "failed",
                        "fold_id": int(fold_id),
                        "n_train": int(len(fold.train_idx)),
                        "n_test": int(len(fold.test_idx)),
                        "n_folds": int(len(folds)),
                        "error": (
                            "rust gaussian fit_result.standard_deviation must be finite and > 0; "
                            f"got {sigma_hat!r}"
                        ),
                    }
                fitted_formula = formula
                cv_rows.append(
                    {
                        "fit_sec": float(fit_sec),
                        "predict_sec": float(pred_sec),
                        **gaussian_prediction_scores(y_test, pred, sigma_hat),
                        "n_test": int(len(fold.test_idx)),
                        "model_spec": f"{fitted_formula} via release binary {eval_suffix}",
                        "fit_quality": fit_quality_row,
                    }
                )
            else:
                try:
                    risk_score, score_src = _survival_risk_from_rust_pred(pred_df)
                except RuntimeError as e:
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario_name,
                        "status": "failed",
                        "error": str(e),
                    }
                try:
                    survival_grid = _survival_score_grid(train_df, ds["time_col"])
                    survival_matrix = _rust_native_survival_matrix_from_model(
                        rust_bin=rust_bin,
                        model_path=model_path,
                        predict_df=test_df.assign(__entry=0.0),
                        time_col=ds["time_col"],
                        grid=survival_grid,
                        fold_dir=td_path,
                    )
                except RuntimeError as e:
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario_name,
                        "status": "failed",
                        "error": str(e),
                    }
                test_eval_df = test_df
                surv_metrics = score_survival_fold(
                    train_df,
                    test_eval_df,
                    time_col=ds["time_col"],
                    event_col=ds["event_col"],
                    risk_score=risk_score,
                    survival_grid=survival_grid,
                    survival_matrix=survival_matrix,
                )
                _append_survival_plot_fold(
                    plot_payload,
                    test_eval_df,
                    time_col=ds["time_col"],
                    event_col=ds["event_col"],
                    risk_score=risk_score,
                )
                cv_rows.append(
                    {
                        "fit_sec": float(fit_sec),
                        "predict_sec": float(pred_sec),
                        "auc": surv_metrics["auc"],
                        "brier": surv_metrics["brier"],
                        "logloss": surv_metrics["logloss"],
                        "nagelkerke_r2": surv_metrics["nagelkerke_r2"],
                        "n_test": int(len(fold.test_idx)),
                        "model_spec": (
                            "survival model via release binary "
                            f"(c-index on risk score from '{score_src}'; native survival curve scoring) {eval_suffix}"
                        ),
                        "fit_quality": fit_quality_row,
                    }
                )

    if not cv_rows:
        return {
            "contender": contender_name,
            "family": ds["family"],
            "scenario_name": scenario_name,
            "status": "failed",
            "error": "no cross-validation folds generated for scenario",
        }

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    if ood_rows:
        denom = max(sum(int(r["n_test"]) for r in ood_rows), 1)
        metrics["ood_abs_dev_from_baseline"] = float(
            sum(float(r["ood_abs_dev_from_baseline"]) * int(r["n_test"]) for r in ood_rows) / denom
        )
        metrics["ood_max_abs_dev_from_baseline"] = float(
            sum(float(r["ood_max_abs_dev_from_baseline"]) * int(r["n_test"]) for r in ood_rows) / denom
        )
        metrics["ood_mean_abs_logit"] = float(
            sum(float(r["ood_mean_abs_logit"]) * int(r["n_test"]) for r in ood_rows) / denom
        )
    if continuous_rows:
        status_counts: dict[str, int] = {}
        for r in continuous_rows:
            s = str(r.get("status", "UndefinedZeroLambda"))
            status_counts[s] = int(status_counts.get(s, 0) + 1)
        denom = max(sum(int(r["n_test"]) for r in continuous_rows), 1)
        valid_nu = [(float(r["nu"]), int(r["n_test"])) for r in continuous_rows if r.get("nu") is not None]
        valid_k2 = [(float(r["kappa2"]), int(r["n_test"])) for r in continuous_rows if r.get("kappa2") is not None]
        metrics["continuous_order_status_counts"] = status_counts
        metrics["continuous_order_status_mode"] = max(status_counts, key=lambda status: status_counts[status])
        metrics["continuous_order_nu"] = (
            float(sum(v * w for v, w in valid_nu) / max(sum(w for _, w in valid_nu), 1))
            if valid_nu
            else None
        )
        metrics["continuous_order_kappa2"] = (
            float(sum(v * w for v, w in valid_k2) / max(sum(w for _, w in valid_k2), 1))
            if valid_k2
            else None
        )
        true_nu = ds.get("continuous_order_true_nu")
        if true_nu is not None and metrics["continuous_order_nu"] is not None:
            metrics["continuous_order_nu_abs_error"] = float(
                abs(float(metrics["continuous_order_nu"]) - float(true_nu))
            )
        expected = ds.get("continuous_order_expected_statuses")
        if isinstance(expected, list) and expected:
            mode = str(metrics["continuous_order_status_mode"])
            metrics["continuous_order_boundary_ok"] = bool(mode in {str(x) for x in expected})
            metrics["continuous_order_expected_statuses"] = [str(x) for x in expected]
    if adaptive_rows:
        mm_vals = [(int(r["mm_iterations"]), int(r["n_test"])) for r in adaptive_rows if "mm_iterations" in r]
        if mm_vals:
            metrics["mm_iterations"] = float(
                sum(v * w for v, w in mm_vals) / max(sum(w for _, w in mm_vals), 1)
            )
        conv_vals = [
            (1.0 if bool(r.get("adaptive_converged")) else 0.0, int(r["n_test"]))
            for r in adaptive_rows
            if "adaptive_converged" in r
        ]
        if conv_vals:
            metrics["adaptive_converged_rate"] = float(
                sum(v * w for v, w in conv_vals) / max(sum(w for _, w in conv_vals), 1)
            )
        grad_corr = [
            (float(r["thread3_weight_grad_corr"]), int(r["n_test"]))
            for r in adaptive_rows
            if "thread3_weight_grad_corr" in r
        ]
        curv_corr = [
            (float(r["thread3_weight_curvature_corr"]), int(r["n_test"]))
            for r in adaptive_rows
            if "thread3_weight_curvature_corr" in r
        ]
        if grad_corr:
            metrics["thread3_weight_grad_corr"] = float(
                sum(v * w for v, w in grad_corr) / max(sum(w for _, w in grad_corr), 1)
            )
        if curv_corr:
            metrics["thread3_weight_curvature_corr"] = float(
                sum(v * w for v, w in curv_corr) / max(sum(w for _, w in curv_corr), 1)
            )
        colloc_rows = [
            (int(r["thread3_collocation_points"]), int(r["n_test"]))
            for r in adaptive_rows
            if "thread3_collocation_points" in r
        ]
        if colloc_rows:
            metrics["thread3_collocation_points"] = float(
                sum(v * w for v, w in colloc_rows) / max(sum(w for _, w in colloc_rows), 1)
            )
    return _finalize_cv_result(
        contender=contender_name,
        scenario_name=scenario_name,
        family=ds["family"],
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=str(cv_rows[0]["model_spec"]),
        extra_metrics=metrics,
    )


def _run_rust_gamlss_scenario_cv_variant(
    scenario: typing.Any,
    *,
    contender_name: str,
    binomial_cli_family: str,
    gaussian_formula_link: str | None = None,
    ds: dict[str, typing.Any] | None = None,
    folds: list[Fold] | None = None,
    shared_fold_artifacts: list[SharedFoldArtifact] | None = None,
) -> typing.Any:
    scenario_name = scenario["name"]
    # Run for any scenario with a valid formula mapping (not just geo scenarios).
    if _scenario_fit_mapping(scenario_name) is None:
        return None

    if ds is None:
        ds = dataset_for_scenario(scenario)
    family = ds["family"]
    if family not in ("binomial", "gaussian"):
        return None
    if folds is None:
        folds = folds_for_dataset(ds)

    try:
        rust_bin = _ensure_rust_binary()
    except _EXPECTED_EXTERNAL_FAILURES as e:
        return {
            "contender": contender_name,
            "scenario_name": scenario_name,
            "status": "failed",
            "error": str(e),
        }

    _, mean_formula = _rust_formula_for_scenario(scenario_name, ds)
    if family == "binomial":
        mean_formula = _append_formula_link_term(mean_formula, binomial_cli_family)
    elif gaussian_formula_link:
        mean_formula = _append_formula_link_term(mean_formula, gaussian_formula_link)
    base_df = pd.DataFrame(ds["rows"])
    cv_rows = []
    plot_payload = _init_plot_payload(ds)
    eval_suffix = _evaluation_suffix(folds)

    with _workspace_tempdir(prefix="gam_bench_rust_gamlss_cv_") as td:
        td_path = Path(td)
        for fold_id, fold in enumerate(folds):
            train_df = base_df.iloc[fold.train_idx].copy()
            test_df = base_df.iloc[fold.test_idx].copy()
            shared_artifact = (
                shared_fold_artifacts[fold_id]
                if shared_fold_artifacts is not None and fold_id < len(shared_fold_artifacts)
                else None
            )
            if shared_artifact is not None:
                train_path = shared_artifact.train_scaled_csv
                test_path = shared_artifact.test_scaled_csv
            else:
                # Z-score features (matches the main rust_gam contender).
                train_df, test_df = zscore_train_test(train_df, test_df, ds["features"])
                train_path = td_path / f"train_{fold_id}.csv"
                test_path = td_path / f"test_{fold_id}.csv"
                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)
            noise_formula = _formula_rhs_from_terms(
                _sigma_feature_terms(ds, scenario_name=scenario_name, backend="rust")
            )
            model_path = td_path / f"model_{fold_id}.json"
            pred_path = td_path / f"pred_{fold_id}.csv"

            fit_cmd = [
                str(rust_bin),
                "fit",
                "--predict-noise",
                noise_formula,
                "--out",
                str(model_path),
            ]
            fit_cmd.extend([str(train_path), mean_formula])
            t0 = perf_counter()
            code, out, err = run_cmd(fit_cmd, cwd=ROOT)
            fit_sec = perf_counter() - t0
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": (err.strip() or out.strip() or "rust gamlss fit failed"),
                }

            pred_cmd = [
                str(rust_bin),
                "predict",
                str(model_path),
                str(test_path),
                "--out",
                str(pred_path),
            ]
            t1 = perf_counter()
            code, out, err = run_cmd(pred_cmd, cwd=ROOT)
            pred_sec = perf_counter() - t1
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": (err.strip() or out.strip() or "rust gamlss predict failed"),
                }
            pred_df = pd.read_csv(pred_path)
            if "mean" not in pred_df.columns:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": "rust gamlss prediction output missing 'mean' column",
                }
            pred = pred_df["mean"].to_numpy(dtype=float)
            y_test = test_df[ds["target"]].to_numpy(dtype=float)
            _append_supervised_plot_fold(plot_payload, test_df, pred, ds["target"])

            if family == "binomial":
                # For binomial location-scale models, the Rust predictor's "mean"
                # column is already the final event probability implied by the
                # joint threshold/log-sigma model. Proper scoring rules are
                # therefore evaluated on this probability, not on a separate
                # standalone sigma output.
                y_train = train_df[ds["target"]].to_numpy(dtype=float)
                cv_rows.append(
                    {
                        "fit_sec": float(fit_sec),
                        "predict_sec": float(pred_sec),
                        "auc": auc_score(y_test, pred),
                        "brier": brier_score(y_test, pred),
                        "logloss": log_loss_score(y_test, pred),
                        "nagelkerke_r2": nagelkerke_r2_score(y_test, pred, null_mean=float(np.mean(y_train))),
                        "n_test": int(len(fold.test_idx)),
                        "model_spec": f"mu: {mean_formula}; sigma: {noise_formula} via release binary {eval_suffix}",
                    }
                )
            else:
                if "sigma" not in pred_df.columns:
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario_name,
                        "status": "failed",
                        "fold_id": int(fold_id),
                        "n_train": int(len(fold.train_idx)),
                        "n_test": int(len(fold.test_idx)),
                        "n_folds": int(len(folds)),
                        "error": "rust gamlss gaussian prediction output missing 'sigma' column",
                    }
                sigma_hat = pred_df["sigma"].to_numpy(dtype=float)
                if sigma_hat.shape[0] != pred.shape[0]:
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario_name,
                        "status": "failed",
                        "fold_id": int(fold_id),
                        "n_train": int(len(fold.train_idx)),
                        "n_test": int(len(fold.test_idx)),
                        "n_folds": int(len(folds)),
                        "error": (
                            "rust gamlss gaussian prediction output has invalid 'sigma' length "
                            f"(got {sigma_hat.shape[0]}, expected {pred.shape[0]})"
                        ),
                    }
                bad_sigma = ~np.isfinite(sigma_hat) | (sigma_hat <= 0.0)
                if np.any(bad_sigma):
                    return {
                        "contender": contender_name,
                        "scenario_name": scenario_name,
                        "status": "failed",
                        "fold_id": int(fold_id),
                        "n_train": int(len(fold.train_idx)),
                        "n_test": int(len(fold.test_idx)),
                        "n_folds": int(len(folds)),
                        "error": (
                            "rust gamlss gaussian prediction output has non-finite or non-positive "
                            f"'sigma' values ({int(np.sum(bad_sigma))} invalid rows)"
                        ),
                    }
                cv_rows.append(
                    {
                        "fit_sec": float(fit_sec),
                        "predict_sec": float(pred_sec),
                        **gaussian_prediction_scores(y_test, pred, sigma_hat),
                        "n_test": int(len(fold.test_idx)),
                        "model_spec": f"mu: {mean_formula}; sigma: {noise_formula} via release binary {eval_suffix}",
                    }
                )

    if not cv_rows:
        return {
            "contender": contender_name,
            "family": family,
            "scenario_name": scenario_name,
            "status": "failed",
            "error": "no cross-validation folds generated for scenario",
        }

    return _finalize_cv_result(
        contender=contender_name,
        scenario_name=scenario_name,
        family=family,
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=str(cv_rows[0]["model_spec"]),
    )


def run_rust_gamlss_scenario_cv(
    scenario: typing.Any,
    *,
    contender_name: str = "rust_gamlss",
    binomial_cli_family: str = "binomial-probit",
    gaussian_formula_link: str | None = None,
    ds: dict[str, typing.Any] | None = None,
    folds: list[Fold] | None = None,
    shared_fold_artifacts: list[SharedFoldArtifact] | None = None,
) -> typing.Any:
    return _run_rust_gamlss_scenario_cv_variant(
        scenario,
        contender_name=contender_name,
        binomial_cli_family=binomial_cli_family,
        gaussian_formula_link=gaussian_formula_link,
        ds=ds,
        folds=folds,
        shared_fold_artifacts=shared_fold_artifacts,
    )


def run_rust_gamlss_marginal_slope_cv(
    scenario: typing.Any,
    *,
    contender_name: str = "rust_gamlss_marginal_slope",
    ds: dict[str, typing.Any] | None = None,
    folds: list[Fold] | None = None,
    rust_fit_extra_args: list[str] | None = None,
) -> typing.Any:
    scenario_name = scenario["name"]
    if _scenario_fit_mapping(scenario_name) is None:
        return None
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "binomial":
        return None
    if folds is None:
        folds = folds_for_dataset(ds)

    try:
        rust_bin = _ensure_rust_binary()
    except _EXPECTED_EXTERNAL_FAILURES as e:
        return {
            "contender": contender_name,
            "scenario_name": scenario_name,
            "status": "failed",
            "error": str(e),
        }

    z_column, mean_formula, logslope_formula = _rust_marginal_slope_formulas_for_scenario(
        scenario_name,
        ds,
    )
    base_df = pd.DataFrame(ds["rows"])
    cv_rows = []
    plot_payload = _init_plot_payload(ds)
    eval_suffix = _evaluation_suffix(folds)

    with _workspace_tempdir(prefix="gam_bench_rust_gamlss_ms_cv_") as td:
        td_path = Path(td)
        for fold_id, fold in enumerate(folds):
            train_raw = base_df.iloc[fold.train_idx].copy()
            test_raw = base_df.iloc[fold.test_idx].copy()
            train_df = train_raw.copy()
            test_df = test_raw.copy()
            train_df, test_df = zscore_train_test(train_df, test_df, ds["features"])
            _apply_exact_train_fold_standardization(
                train_df,
                test_df,
                train_raw,
                test_raw,
                z_column,
            )
            train_path = td_path / f"train_{fold_id}.csv"
            test_path = td_path / f"test_{fold_id}.csv"
            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)
            model_path = td_path / f"model_{fold_id}.json"
            pred_path = td_path / f"pred_{fold_id}.csv"

            fit_cmd = [
                str(rust_bin),
                "fit",
                "--logslope-formula",
                logslope_formula,
                "--z-column",
                z_column,
                "--out",
                str(model_path),
            ]
            if rust_fit_extra_args:
                fit_cmd.extend([str(x) for x in rust_fit_extra_args])
            fit_cmd.extend([str(train_path), mean_formula])
            t0 = perf_counter()
            code, out, err = run_cmd(fit_cmd, cwd=ROOT)
            fit_sec = perf_counter() - t0
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": (err.strip() or out.strip() or "rust gamlss marginal-slope fit failed"),
                }

            pred_cmd = [
                str(rust_bin),
                "predict",
                str(model_path),
                str(test_path),
                "--out",
                str(pred_path),
            ]
            t1 = perf_counter()
            code, out, err = run_cmd(pred_cmd, cwd=ROOT)
            pred_sec = perf_counter() - t1
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": (err.strip() or out.strip() or "rust gamlss marginal-slope predict failed"),
                }
            pred_df = pd.read_csv(pred_path)
            if "mean" not in pred_df.columns:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "n_train": int(len(fold.train_idx)),
                    "n_test": int(len(fold.test_idx)),
                    "n_folds": int(len(folds)),
                    "error": "rust gamlss marginal-slope prediction output missing 'mean' column",
                }
            pred = pred_df["mean"].to_numpy(dtype=float)
            y_test = test_df[ds["target"]].to_numpy(dtype=float)
            y_train = train_df[ds["target"]].to_numpy(dtype=float)
            _append_supervised_plot_fold(plot_payload, test_df, pred, ds["target"])
            cv_rows.append(
                {
                    "fit_sec": float(fit_sec),
                    "predict_sec": float(pred_sec),
                    "auc": auc_score(y_test, pred),
                    "brier": brier_score(y_test, pred),
                    "logloss": log_loss_score(y_test, pred),
                    "nagelkerke_r2": nagelkerke_r2_score(y_test, pred, null_mean=float(np.mean(y_train))),
                    "n_test": int(len(fold.test_idx)),
                    "model_spec": (
                        f"marginal: {mean_formula}; logslope: {logslope_formula}; "
                        f"z: {z_column} via release binary {eval_suffix}"
                    ),
                }
            )

    if not cv_rows:
        return {
            "contender": contender_name,
            "family": ds["family"],
            "scenario_name": scenario_name,
            "status": "failed",
            "error": "no cross-validation folds generated for scenario",
        }

    return _finalize_cv_result(
        contender=contender_name,
        scenario_name=scenario_name,
        family=ds["family"],
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=str(cv_rows[0]["model_spec"]),
    )


def run_rust_gamlss_survival_cv(
    scenario: typing.Any,
    *,
    contender_name: str = "rust_gamlss_survival",
    survival_link: str | None = None,
    ds: dict[str, typing.Any] | None = None,
    folds: list[Fold] | None = None,
) -> typing.Any:
    """Run the Rust binary with --survival-likelihood location-scale for survival scenarios."""
    scenario_name = scenario["name"]
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    if folds is None:
        folds = folds_for_dataset(ds)

    try:
        rust_bin = _ensure_rust_binary()
    except _EXPECTED_EXTERNAL_FAILURES as e:
        return {
            "contender": contender_name,
            "scenario_name": scenario_name,
            "status": "failed",
            "error": str(e),
        }

    base_df = pd.DataFrame(ds["rows"])
    cv_rows = []
    plot_payload = _init_plot_payload(ds)

    with _workspace_tempdir(prefix="gam_bench_rust_gamlss_surv_cv_") as td:
        td_path = Path(td)
        for fold_id, fold in enumerate(folds):
            train_raw = base_df.iloc[fold.train_idx].copy()
            test_raw = base_df.iloc[fold.test_idx].copy()
            train_df = train_raw.copy()
            test_df = test_raw.copy()
            test_eval_df = test_raw.copy()
            train_path = td_path / f"train_{fold_id}.csv"
            test_path = td_path / f"test_{fold_id}.csv"
            model_path = td_path / f"model_{fold_id}.json"
            pred_path = td_path / f"pred_{fold_id}.csv"

            train_df, test_df = zscore_train_test(train_df, test_df, ds["features"])
            train_df["__entry"] = 0.0
            horizon = _survival_eval_horizon(train_df, ds["time_col"])
            test_pred_df = test_df.copy()
            test_pred_df["__entry"] = 0.0
            test_pred_df[ds["time_col"]] = horizon
            train_df.to_csv(train_path, index=False)
            test_pred_df.to_csv(test_path, index=False)

            rhs_formula = _rust_survival_formula_for_scenario(scenario_name)
            fit_formula = (
                f"Surv(__entry, {ds['time_col']}, {ds['event_col']}) ~ {rhs_formula}"
            )

            fit_cmd = [
                str(rust_bin),
                "fit",
                "--survival-likelihood",
                "location-scale",
                "--out",
                str(model_path),
            ]
            if survival_link:
                fit_cmd.extend(["--link", str(survival_link)])
            fit_cmd.extend(_rust_survival_fit_cli_args(scenario_name))
            fit_cmd.extend([str(train_path), fit_formula])

            t0 = perf_counter()
            code, out, err = run_cmd(fit_cmd, cwd=ROOT)
            fit_sec = perf_counter() - t0
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "error": (err.strip() or out.strip() or "rust gamlss survival fit failed"),
                }

            pred_cmd = [
                str(rust_bin), "predict",
                str(model_path), str(test_path),
                "--out", str(pred_path),
            ]
            t1 = perf_counter()
            code, out, err = run_cmd(pred_cmd, cwd=ROOT)
            pred_sec = perf_counter() - t1
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "error": (err.strip() or out.strip() or "rust gamlss survival predict failed"),
                }
            pred_df = pd.read_csv(pred_path)
            try:
                risk_score, score_src = _survival_risk_from_rust_pred(pred_df)
            except RuntimeError as e:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "error": str(e),
                }
            try:
                survival_grid = _survival_score_grid(train_df, ds["time_col"])
                survival_matrix = _rust_native_survival_matrix_from_model(
                    rust_bin=rust_bin,
                    model_path=model_path,
                    predict_df=test_df.assign(__entry=0.0),
                    time_col=ds["time_col"],
                    grid=survival_grid,
                    fold_dir=td_path,
                )
            except RuntimeError as e:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "error": str(e),
                }
            surv_metrics = score_survival_fold(
                train_df,
                test_eval_df,
                time_col=ds["time_col"],
                event_col=ds["event_col"],
                risk_score=risk_score,
                survival_grid=survival_grid,
                survival_matrix=survival_matrix,
            )
            _append_survival_plot_fold(
                plot_payload,
                test_eval_df,
                time_col=ds["time_col"],
                event_col=ds["event_col"],
                risk_score=risk_score,
            )
            cv_rows.append(
                {
                    "fit_sec": float(fit_sec),
                    "predict_sec": float(pred_sec),
                    "auc": surv_metrics["auc"],
                    "brier": surv_metrics["brier"],
                    "logloss": surv_metrics["logloss"],
                    "nagelkerke_r2": surv_metrics["nagelkerke_r2"],
                    "n_test": int(len(fold.test_idx)),
                    "model_spec": (
                        f"{fit_formula} [survival-likelihood=location-scale; "
                        f"risk_score={score_src}; native survival curve scoring] {_evaluation_suffix(folds)}"
                    ),
                }
            )

    if not cv_rows:
        return {
            "contender": contender_name,
            "family": ds["family"],
            "scenario_name": scenario_name,
            "status": "failed",
            "error": "no cross-validation folds generated for scenario",
        }

    return _finalize_cv_result(
        contender=contender_name,
        scenario_name=scenario_name,
        family=ds["family"],
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=str(cv_rows[0]["model_spec"]),
    )


def run_rust_gamlss_survival_marginal_slope_cv(
    scenario: typing.Any,
    *,
    contender_name: str = "rust_gamlss_survival_marginal_slope",
    ds: dict[str, typing.Any] | None = None,
    folds: list[Fold] | None = None,
) -> typing.Any:
    scenario_name = scenario["name"]
    if ds is None:
        ds = dataset_for_scenario(scenario)
    if ds["family"] != "survival":
        return None
    if folds is None:
        folds = folds_for_dataset(ds)

    try:
        rust_bin = _ensure_rust_binary()
    except _EXPECTED_EXTERNAL_FAILURES as e:
        return {
            "contender": contender_name,
            "scenario_name": scenario_name,
            "status": "failed",
            "error": str(e),
        }

    z_column, rhs_formula, rhs_logslope = _rust_survival_marginal_slope_formulas_for_scenario(
        scenario_name,
        ds,
    )
    base_df = pd.DataFrame(ds["rows"])
    cv_rows = []
    plot_payload = _init_plot_payload(ds)
    eval_suffix = _evaluation_suffix(folds)

    with _workspace_tempdir(prefix="gam_bench_rust_gamlss_surv_ms_cv_") as td:
        td_path = Path(td)
        for fold_id, fold in enumerate(folds):
            train_raw = base_df.iloc[fold.train_idx].copy()
            test_raw = base_df.iloc[fold.test_idx].copy()
            train_df = train_raw.copy()
            test_df = test_raw.copy()
            test_eval_df = test_raw.copy()
            train_path = td_path / f"train_{fold_id}.csv"
            test_path = td_path / f"test_{fold_id}.csv"
            model_path = td_path / f"model_{fold_id}.json"
            pred_path = td_path / f"pred_{fold_id}.csv"

            train_df, test_df = zscore_train_test(train_df, test_df, ds["features"])
            train_df["__entry"] = 0.0
            horizon = _survival_eval_horizon(train_df, ds["time_col"])
            test_pred_df = test_df.copy()
            test_pred_df["__entry"] = 0.0
            test_pred_df[ds["time_col"]] = horizon
            train_df.to_csv(train_path, index=False)
            test_pred_df.to_csv(test_path, index=False)

            fit_formula = f"Surv(__entry, {ds['time_col']}, {ds['event_col']}) ~ {rhs_formula}"
            logslope_formula = rhs_logslope
            fit_cmd = [
                str(rust_bin),
                "fit",
                "--survival-likelihood",
                "marginal-slope",
                "--logslope-formula",
                logslope_formula,
                "--z-column",
                z_column,
                "--out",
                str(model_path),
            ]
            fit_cmd.extend(_rust_survival_fit_cli_args(scenario_name))
            fit_cmd.extend([str(train_path), fit_formula])

            t0 = perf_counter()
            code, out, err = run_cmd(fit_cmd, cwd=ROOT)
            fit_sec = perf_counter() - t0
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "error": (
                        err.strip() or out.strip() or "rust gamlss survival marginal-slope fit failed"
                    ),
                }

            pred_cmd = [
                str(rust_bin),
                "predict",
                str(model_path),
                str(test_path),
                "--out",
                str(pred_path),
            ]
            t1 = perf_counter()
            code, out, err = run_cmd(pred_cmd, cwd=ROOT)
            pred_sec = perf_counter() - t1
            if code != 0:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "fold_id": int(fold_id),
                    "error": (
                        err.strip()
                        or out.strip()
                        or "rust gamlss survival marginal-slope predict failed"
                    ),
                }
            pred_df = pd.read_csv(pred_path)
            try:
                risk_score, score_src = _survival_risk_from_rust_pred(pred_df)
            except RuntimeError as e:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "error": str(e),
                }
            try:
                survival_grid = _survival_score_grid(train_df, ds["time_col"])
                survival_matrix = _rust_native_survival_matrix_from_model(
                    rust_bin=rust_bin,
                    model_path=model_path,
                    predict_df=test_df.assign(__entry=0.0),
                    time_col=ds["time_col"],
                    grid=survival_grid,
                    fold_dir=td_path,
                )
            except RuntimeError as e:
                return {
                    "contender": contender_name,
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "error": str(e),
                }
            surv_metrics = score_survival_fold(
                train_df,
                test_eval_df,
                time_col=ds["time_col"],
                event_col=ds["event_col"],
                risk_score=risk_score,
                survival_grid=survival_grid,
                survival_matrix=survival_matrix,
            )
            _append_survival_plot_fold(
                plot_payload,
                test_eval_df,
                time_col=ds["time_col"],
                event_col=ds["event_col"],
                risk_score=risk_score,
            )
            cv_rows.append(
                {
                    "fit_sec": float(fit_sec),
                    "predict_sec": float(pred_sec),
                    "auc": surv_metrics["auc"],
                    "brier": surv_metrics["brier"],
                    "logloss": surv_metrics["logloss"],
                    "nagelkerke_r2": surv_metrics["nagelkerke_r2"],
                    "n_test": int(len(fold.test_idx)),
                    "model_spec": (
                        f"{fit_formula} [survival-likelihood=marginal-slope; "
                        f"logslope={logslope_formula}; z={z_column}; risk_score={score_src}; "
                        f"native survival curve scoring] {eval_suffix}"
                    ),
                }
            )

    if not cv_rows:
        return {
            "contender": contender_name,
            "family": ds["family"],
            "scenario_name": scenario_name,
            "status": "failed",
            "error": "no cross-validation folds generated for scenario",
        }

    return _finalize_cv_result(
        contender=contender_name,
        scenario_name=scenario_name,
        family=ds["family"],
        cv_rows=cv_rows,
        plot_payload=plot_payload,
        model_spec=str(cv_rows[0]["model_spec"]),
    )


_import_run_suite_exports("_run_suite_external")


def _assert_basis_parity_for_scenario(s_cfg: typing.Any, *, ds: dict[str, typing.Any] | None = None) -> None:
    # Fairness guard: if Rust mapping requests a named spline family, mgcv must
    # emit the equivalent basis for the same scenario.
    if ds is None:
        ds = dataset_for_scenario(s_cfg)
    scenario_name = s_cfg["name"]
    rust_cfg = _scenario_fit_mapping(scenario_name)
    if rust_cfg is None:
        return
    basis = _canonical_smooth_basis(rust_cfg.get("smooth_basis", "ps"))
    if basis not in {"duchon", "matern"}:
        return

    mgcv_formula = _mgcv_formula_for_scenario(scenario_name, ds)
    expected = "bs='ds'" if basis == "duchon" else "bs='gp'"
    if expected not in mgcv_formula:
        raise SystemExit(
            "basis parity check failed for "
            f"{scenario_name}: rust basis='{basis}' but mgcv formula is '{mgcv_formula}'"
        )


def _extra_excluded_contenders_for_profile() -> set[str]:
    if _BENCH_CI_PROFILE == "lean":
        return set(_LEAN_PROFILE_EXCLUDED_CONTENDERS)
    return set()


def _is_contender_enabled(s_cfg: typing.Any, contender: str) -> bool:
    excluded = set(s_cfg.get("exclude_contenders", []))
    excluded.update(_extra_excluded_contenders_for_profile())
    return contender not in excluded


def _append_enabled_result_rows(results: list[dict[str, typing.Any]], rows: typing.Any, s_cfg: typing.Any) -> None:
    if rows is None:
        return
    if not isinstance(rows, list):
        rows = [rows]
    for row in rows:
        if row is None:
            continue
        contender = row.get("contender")
        if not isinstance(contender, str):
            raise RuntimeError(f"benchmark row missing valid contender name: {row!r}")
        if _is_contender_enabled(s_cfg, contender):
            results.append(row)


def _append_contender_result_if_enabled(
    results: list[dict[str, typing.Any]],
    s_cfg: typing.Any,
    contender: str,
    build_row: typing.Any,
) -> None:
    if not _is_contender_enabled(s_cfg, contender):
        return
    # Convert per-contender harness errors (e.g. formula-construction
    # RuntimeError from _emit_joint_pc_term when a configuration is
    # structurally infeasible) into a failed result row. Without this,
    # an unhandled RuntimeError would kill the whole shard before the
    # output JSON is written, hiding the very signal the bench is
    # meant to surface. The row still propagates to the blocking-
    # failure check below; it just no longer destroys the run record.
    try:
        row = build_row()
    except RuntimeError as e:
        row = {
            "contender": contender,
            "scenario_name": str(s_cfg.get("name", "")),
            "status": "failed",
            "error": f"harness error: {e}",
        }
    if row is not None:
        results.append(row)


def _required_contender_for_scenario(s_cfg: typing.Any, ds: dict[str, typing.Any]) -> str | None:
    contender = "rust_gamlss_survival" if ds["family"] == "survival" else "rust_gam"
    if not _is_contender_enabled(s_cfg, contender):
        return None
    return contender


def _is_non_blocking_failure(row: dict[str, typing.Any]) -> bool:
    return str(row.get("contender", "")) in NON_BLOCKING_FAILURE_CONTENDERS


def _format_blocking_failure(row: dict[str, typing.Any]) -> str:
    contender = str(row.get("contender", "?"))
    scenario_name = str(row.get("scenario_name", "?"))
    error = str(row.get("error", "unknown error")).strip()

    timeout_match = re.search(r"\[HEARTBEAT\]\s+command-timeout\b.*?\btimeout_sec=(\d+)", error)
    if timeout_match:
        timeout_sec = int(timeout_match.group(1))
        return f"{contender} / {scenario_name}: TIMEOUT after {timeout_sec}s"

    return f"{contender} / {scenario_name}: {error or 'unknown error'}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GAM benchmark suite with leakage-safe evaluation.")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--out", type=Path, default=BENCH_DIR / "results.json")
    parser.add_argument(
        "--scenario-name",
        action="append",
        dest="scenario_names",
        help="Run only the named scenario(s). Can be passed multiple times.",
    )
    parser.add_argument(
        "--gate",
        choices=["report", "strict", "off"],
        default=None,
        help=(
            "Statistical-regression gate mode. Default: strict. Set BENCH_GATE=report "
            "or pass --gate report to opt out of failing CI on regression. 'off' "
            "skips the gate entirely."
        ),
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write/refresh the per-lane baselines under bench/baselines/ from this run.",
    )
    args = parser.parse_args()
    print(
        "bench runtime config | "
        f"force_serial={_FORCE_SERIAL} "
        f"rayon_threads={_RAYON_THREADS if _RAYON_THREADS > 0 else 'auto'} "
        f"blas_threads={_BLAS_THREADS if _BLAS_THREADS > 0 else 'auto'} "
        f"cmd_timeout_sec={_CMD_TIMEOUT_SEC if _CMD_TIMEOUT_SEC > 0 else 'none'}",
        file=sys.stderr,
        flush=True,
    )

    cfg = json.loads(args.scenarios.read_text())
    scenarios = cfg.get("scenarios", [])
    if not scenarios:
        raise SystemExit("No scenarios found.")

    if args.scenario_names:
        wanted = set(args.scenario_names)
        scenarios = [s for s in scenarios if s.get("name") in wanted]
        missing = sorted(wanted - {s.get("name") for s in scenarios})
        if missing:
            raise SystemExit(f"Unknown scenario name(s): {', '.join(missing)}")
        if not scenarios:
            raise SystemExit("Scenario filter matched zero scenarios.")

    results: list[dict[str, typing.Any]] = []
    for s_cfg in scenarios:
        ds = dataset_for_scenario(s_cfg)
        folds = folds_for_dataset(ds)
        _assert_basis_parity_for_scenario(s_cfg, ds=ds)
        with _workspace_tempdir(prefix="gam_bench_shared_folds_") as shared_td:
            shared_fold_artifacts = build_shared_fold_artifacts(ds, folds, Path(shared_td))
            if ds["family"] != "survival":
                _append_contender_result_if_enabled(
                    results,
                    s_cfg,
                    "rust_gam",
                    lambda: run_rust_scenario_cv(
                        s_cfg,
                        contender_name="rust_gam",
                        ds=ds,
                        folds=folds,
                        shared_fold_artifacts=shared_fold_artifacts,
                    ),
                )
                if ds["family"] == "binomial":
                    _append_contender_result_if_enabled(
                        results,
                        s_cfg,
                        "rust_gam_flexible",
                        lambda: run_rust_scenario_cv(
                            s_cfg,
                            contender_name="rust_gam_flexible",
                            ds=ds,
                            folds=folds,
                            shared_fold_artifacts=shared_fold_artifacts,
                            formula_link=_flexible_link_name(
                                _default_rust_formula_link_for_family(ds["family"])
                            ),
                        ),
                    )
            if _is_matern_rust_scenario(s_cfg):
                _append_contender_result_if_enabled(
                    results,
                    s_cfg,
                    "rust_matern_decomposed",
                    lambda: run_rust_scenario_cv(
                        s_cfg,
                        contender_name="rust_matern_decomposed",
                        ds=ds,
                        folds=folds,
                        shared_fold_artifacts=shared_fold_artifacts,
                        rust_cfg_override={"double_penalty": True},
                        eval_ood=True,
                    ),
                )
                if ds["family"] == "binomial":
                    _append_contender_result_if_enabled(
                        results,
                        s_cfg,
                        "rust_matern_decomposed_flexible",
                        lambda: run_rust_scenario_cv(
                            s_cfg,
                            contender_name="rust_matern_decomposed_flexible",
                            ds=ds,
                            folds=folds,
                            shared_fold_artifacts=shared_fold_artifacts,
                            rust_cfg_override={"double_penalty": True},
                            eval_ood=True,
                            formula_link=_flexible_link_name(
                                _default_rust_formula_link_for_family(ds["family"])
                            ),
                        ),
                    )
                _append_contender_result_if_enabled(
                    results,
                    s_cfg,
                    "rust_matern_standard",
                    lambda: run_rust_scenario_cv(
                        s_cfg,
                        contender_name="rust_matern_standard",
                        ds=ds,
                        folds=folds,
                        shared_fold_artifacts=shared_fold_artifacts,
                        rust_cfg_override={"double_penalty": False},
                        eval_ood=True,
                    ),
                )
                if ds["family"] == "binomial":
                    _append_contender_result_if_enabled(
                        results,
                        s_cfg,
                        "rust_matern_standard_flexible",
                        lambda: run_rust_scenario_cv(
                            s_cfg,
                            contender_name="rust_matern_standard_flexible",
                            ds=ds,
                            folds=folds,
                            shared_fold_artifacts=shared_fold_artifacts,
                            rust_cfg_override={"double_penalty": False},
                            eval_ood=True,
                            formula_link=_flexible_link_name(
                                _default_rust_formula_link_for_family(ds["family"])
                            ),
                        ),
                    )
            if str(s_cfg.get("name", "")).startswith("continuous_order_"):
                _append_contender_result_if_enabled(
                    results,
                    s_cfg,
                    "rust_continuous_order_probe",
                    lambda: run_rust_scenario_cv(
                        s_cfg,
                        contender_name="rust_continuous_order_probe",
                        ds=ds,
                        folds=folds,
                        shared_fold_artifacts=shared_fold_artifacts,
                        rust_cfg_override={"double_penalty": True},
                        collect_continuous_order=True,
                    ),
                )
            if str(s_cfg.get("name", "")) == "thread3_admixture_cliff":
                _append_contender_result_if_enabled(
                    results,
                    s_cfg,
                    "rust_thread3_standard_reml",
                    lambda: run_rust_scenario_cv(
                        s_cfg,
                        contender_name="rust_thread3_standard_reml",
                        ds=ds,
                        folds=folds,
                        shared_fold_artifacts=shared_fold_artifacts,
                        rust_cfg_override={"double_penalty": True},
                    ),
                )
                if ds["family"] == "binomial":
                    _append_contender_result_if_enabled(
                        results,
                        s_cfg,
                        "rust_thread3_standard_reml_flexible",
                        lambda: run_rust_scenario_cv(
                            s_cfg,
                            contender_name="rust_thread3_standard_reml_flexible",
                            ds=ds,
                            folds=folds,
                            shared_fold_artifacts=shared_fold_artifacts,
                            rust_cfg_override={"double_penalty": True},
                            formula_link=_flexible_link_name(
                                _default_rust_formula_link_for_family(ds["family"])
                            ),
                        ),
                    )
                _append_contender_result_if_enabled(
                    results,
                    s_cfg,
                    "rust_thread3_adaptive_reml",
                    lambda: run_rust_scenario_cv(
                        s_cfg,
                        contender_name="rust_thread3_adaptive_reml",
                        ds=ds,
                        folds=folds,
                        shared_fold_artifacts=shared_fold_artifacts,
                        rust_cfg_override={"double_penalty": True},
                        collect_adaptive_diagnostics=True,
                        rust_fit_extra_args=[
                            "--adaptive-regularization",
                            "true",
                        ],
                    ),
                )
                if ds["family"] == "binomial":
                    _append_contender_result_if_enabled(
                        results,
                        s_cfg,
                        "rust_thread3_adaptive_reml_flexible",
                        lambda: run_rust_scenario_cv(
                            s_cfg,
                            contender_name="rust_thread3_adaptive_reml_flexible",
                            ds=ds,
                            folds=folds,
                            shared_fold_artifacts=shared_fold_artifacts,
                            rust_cfg_override={"double_penalty": True},
                            collect_adaptive_diagnostics=True,
                            rust_fit_extra_args=[
                                "--adaptive-regularization",
                                "true",
                            ],
                            formula_link=_flexible_link_name(
                                _default_rust_formula_link_for_family(ds["family"])
                            ),
                        ),
                    )
            # Route the rust_gamlss family of contenders through the same
            # helper used by the rust_gam contenders above. The helper turns
            # per-contender harness errors (notably the formula-construction
            # RuntimeError emitted by `_emit_joint_pc_term` when a Duchon
            # joint-PC smooth is structurally infeasible at the requested k)
            # into a failed result row instead of letting the exception
            # unwind through `main` and tear down the entire benchmark
            # shard before its output JSON is written.
            _append_contender_result_if_enabled(
                results,
                s_cfg,
                "rust_gamlss",
                lambda: run_rust_gamlss_scenario_cv(
                    s_cfg,
                    contender_name="rust_gamlss",
                    ds=ds,
                    folds=folds,
                    shared_fold_artifacts=shared_fold_artifacts,
                ),
            )
            if _is_contender_enabled(s_cfg, "rust_gamlss") and ds["family"] == "binomial":
                _append_contender_result_if_enabled(
                    results,
                    s_cfg,
                    "rust_gamlss_flexible",
                    lambda: run_rust_gamlss_scenario_cv(
                        s_cfg,
                        contender_name="rust_gamlss_flexible",
                        binomial_cli_family=_flexible_link_name("probit"),
                        ds=ds,
                        folds=folds,
                        shared_fold_artifacts=shared_fold_artifacts,
                    ),
                )
                _ms_cfg = _effective_scenario_fit_mapping(s_cfg["name"]) or {}
                _ms_extra = (
                    ["--scale-dimensions"] if _ms_cfg.get("scale_dimensions") else None
                )
                _ms_contender = (
                    "rust_gamlss_marginal_slope_aniso"
                    if _ms_cfg.get("scale_dimensions")
                    else "rust_gamlss_marginal_slope"
                )
                _append_contender_result_if_enabled(
                    results,
                    s_cfg,
                    _ms_contender,
                    lambda: run_rust_gamlss_marginal_slope_cv(
                        s_cfg,
                        contender_name=_ms_contender,
                        ds=ds,
                        folds=folds,
                        rust_fit_extra_args=_ms_extra,
                    ),
                )
            rust_gamlss_surv_row = (
                run_rust_gamlss_survival_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "rust_gamlss_survival")
                else None
            )
            if rust_gamlss_surv_row is not None:
                results.append(rust_gamlss_surv_row)
            rust_gamlss_surv_ms_row = (
                run_rust_gamlss_survival_marginal_slope_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "rust_gamlss_survival_marginal_slope")
                else None
            )
            if rust_gamlss_surv_ms_row is not None:
                results.append(rust_gamlss_surv_ms_row)
            r_gamlss_row = (
                run_external_r_gamlss_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "r_gamlss")
                else None
            )
            if r_gamlss_row is not None:
                results.append(r_gamlss_row)
            if _is_contender_enabled(s_cfg, "r_mgcv"):
                results.append(run_external_mgcv_cv(s_cfg, ds=ds, folds=folds))
            mgcv_gaulss_row = (
                run_external_mgcv_gaulss_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "r_mgcv_gaulss")
                else None
            )
            if mgcv_gaulss_row is not None:
                results.append(mgcv_gaulss_row)
            gamboostlss_row = (
                run_external_r_gamboostlss_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "r_gamboostlss")
                else None
            )
            if gamboostlss_row is not None:
                results.append(gamboostlss_row)
            bamlss_row = (
                run_external_r_bamlss_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "r_bamlss")
                else None
            )
            if bamlss_row is not None:
                results.append(bamlss_row)
            brms_row = (
                run_external_r_brms_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "r_brms")
                else None
            )
            if brms_row is not None:
                results.append(brms_row)
            mgcv_surv_row = (
                run_external_mgcv_survival_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "r_mgcv_coxph")
                else None
            )
            if mgcv_surv_row is not None:
                results.append(mgcv_surv_row)
            sksurv_rsf_row = (
                run_external_sksurv_rsf_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "python_sksurv_rsf")
                else None
            )
            if sksurv_rsf_row is not None:
                results.append(sksurv_rsf_row)
            sksurv_coxnet_row = (
                run_external_sksurv_coxnet_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "python_sksurv_coxnet")
                else None
            )
            if sksurv_coxnet_row is not None:
                results.append(sksurv_coxnet_row)
            lifelines_enet_row = (
                run_external_lifelines_coxph_enet_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "python_lifelines_coxph_enet")
                else None
            )
            if lifelines_enet_row is not None:
                results.append(lifelines_enet_row)
            glmnet_cox_row = (
                run_external_glmnet_cox_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "r_glmnet_cox")
                else None
            )
            if glmnet_cox_row is not None:
                results.append(glmnet_cox_row)
            sksurv_gb_rows = (
                run_external_sksurv_gb_cv(s_cfg, ds=ds, folds=folds)
                if (
                    _is_contender_enabled(s_cfg, "python_sksurv_gb_coxph")
                    or _is_contender_enabled(s_cfg, "python_sksurv_componentwise_gb_coxph")
                )
                else None
            )
            _append_enabled_result_rows(results, sksurv_gb_rows, s_cfg)
            lifelines_aft_rows = (
                run_external_lifelines_aft_cv(s_cfg, ds=ds, folds=folds)
                if (
                    _is_contender_enabled(s_cfg, "python_lifelines_weibull_aft")
                    or _is_contender_enabled(s_cfg, "python_lifelines_lognormal_aft")
                )
                else None
            )
            _append_enabled_result_rows(results, lifelines_aft_rows, s_cfg)
            xgb_aft_row = (
                run_external_xgboost_aft_cv(s_cfg, ds=ds, folds=folds)
                if _is_contender_enabled(s_cfg, "python_xgboost_aft")
                else None
            )
            if xgb_aft_row is not None:
                results.append(xgb_aft_row)

    deferred_exit_messages: list[str] = []
    for s_cfg in scenarios:
        s_name = s_cfg["name"]
        ds = dataset_for_scenario(s_cfg)
        required_contender = _required_contender_for_scenario(s_cfg, ds)
        if required_contender is None:
            continue
        has_required_ok = any(
            r.get("scenario_name") == s_name
            and r.get("contender") == required_contender
            and r.get("status") == "ok"
            for r in results
        )
        if not has_required_ok:
            required_rows = [
                r
                for r in results
                if r.get("scenario_name") == s_name
                and r.get("contender") == required_contender
            ]
            if required_rows:
                deferred_exit_messages.append(
                    "required contender failed for scenario "
                    f"'{s_name}': {_format_blocking_failure(required_rows[0])}"
                )
            else:
                deferred_exit_messages.append(
                    f"missing required successful {required_contender} result for scenario '{s_name}'"
                )

    # Hard guard: benchmark outputs must declare an evaluation mode consistent with model_spec.
    _normalize_result_metadata(results)
    _validate_result_metadata(results)

    # Always write the JSON output and figures BEFORE deciding whether
    # to raise SystemExit for blocking failures. Otherwise a single
    # blocking contender error nukes the entire shard's data record
    # — figures, comparison rows, everything — even for contenders
    # that ran cleanly, hiding the failure rather than surfacing it.
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation": {
            "default_n_splits": CV_SPLITS,
            "seed": CV_SEED,
            "leakage_safe": True,
            "scenario_specific_splits": True,
        },
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")

    # ---- Statistical-regression gate --------------------------------------
    # See bench/gate.py for tolerances and rationale. Default mode is
    # 'strict'; use BENCH_GATE=report or --gate report for non-blocking runs.
    gate_mode = args.gate or os.environ.get("BENCH_GATE", "").strip().lower() or "strict"
    if gate_mode not in ("report", "strict", "off"):
        gate_mode = "strict"
    if gate_mode != "off" and _gate_cmd_check_results is not None:
        gate_args = argparse.Namespace(
            results=str(args.out),
            gate=gate_mode,
            update_baseline=bool(args.update_baseline),
        )
        try:
            gate_rc = int(_gate_cmd_check_results(gate_args))
        except Exception as e:  # pragma: no cover - bench infra only
            print(f"[gate] error: {e}", file=sys.stderr)
            gate_rc = 0
        if gate_rc != 0 and gate_mode == "strict":
            raise SystemExit(f"benchmark statistical-regression gate failed (rc={gate_rc})")

    # Generate per-scenario comparison figures and bundle into a .zip.
    fig_dir = args.out.parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    generate_scenario_figures(results, fig_dir)
    generate_scenario_datapoint_figures(results, fig_dir)
    zip_path = args.out.parent / "figures.zip"
    zip_figure_dir(fig_dir, zip_path)
    print(f"Wrote {zip_path}")

    # Now signal CI: benchmark runs must fail for blocking contender failures
    # and unmet required-contender expectations collected above.
    # Non-blocking contenders still emit failed rows in the output for diagnostics.
    failed_blocking = [
        r for r in results if r.get("status") != "ok" and not _is_non_blocking_failure(r)
    ]
    msgs: list[str] = list(deferred_exit_messages)
    for r in failed_blocking:
        msgs.append(_format_blocking_failure(r))
    if msgs:
        raise SystemExit("benchmark run failed:\n" + "\n".join(msgs))


# ---------------------------------------------------------------------------
# Per-scenario comparison figures
# ---------------------------------------------------------------------------

_import_run_suite_exports("_run_suite_plots")


if __name__ == "__main__":
    main()
