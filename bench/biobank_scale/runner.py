#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import threading
import time
import tempfile
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


BENCH_DIR = Path(__file__).resolve().parent
ROOT = BENCH_DIR.parents[1]
DEFAULT_CONFIG = BENCH_DIR / "biobank_scale.yml"
HEARTBEAT_INTERVAL_SEC = 15.0
HEARTBEAT_INITIAL_WINDOW_SEC = 2.0
HEARTBEAT_INITIAL_INTERVAL_SEC = 0.25
MAX_CAPTURE_CHARS = 200000
MAX_SURVIVAL_GRID_POINTS = 256
SURVIVAL_ENTRY_COLUMN = "__entry"
SUPPORTED_BIOBANK_SURVIVAL_LIKELIHOODS = {"transformation", "location-scale"}
SUPPORTED_BIOBANK_SURVIVAL_DISTRIBUTIONS = {
    "gaussian",
    "probit",
    "gumbel",
    "cloglog",
    "logistic",
    "logit",
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    dataset: str
    backend: str
    family: str
    spatial_basis: str
    centers: int | None = None
    smooth_kind: str = "joint"
    include_sigma: bool = False
    survival_likelihood: str | None = None
    survival_distribution: str | None = None
    marginal_slope: bool = False
    scale_dimensions: bool = False
    z_column: str | None = None
    pc_count: int = 16
    mean_linkwiggle_knots: int | None = None
    logslope_linkwiggle_knots: int | None = None
    timewiggle_knots: int | None = None
    max_centers: int | None = None


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{path} must contain JSON-compatible YAML so this runner can parse it without external dependencies: {exc}"
        ) from exc


def validate_method_spec(spec: MethodSpec) -> None:
    if spec.pc_count <= 0:
        raise RuntimeError(f"method '{spec.name}' must set pc_count > 0")
    for key, value in (
        ("mean_linkwiggle_knots", spec.mean_linkwiggle_knots),
        ("logslope_linkwiggle_knots", spec.logslope_linkwiggle_knots),
        ("timewiggle_knots", spec.timewiggle_knots),
    ):
        if value is not None and value < 3:
            raise RuntimeError(f"method '{spec.name}' requires {key} >= 3")
    if spec.dataset == "disease":
        if spec.backend not in {"rust_gam", "r_mgcv"}:
            raise RuntimeError(
                f"unsupported disease backend '{spec.backend}' for '{spec.name}'"
            )
        if spec.survival_likelihood is not None or spec.survival_distribution is not None:
            raise RuntimeError(
                f"disease method '{spec.name}' cannot set survival_likelihood or survival_distribution"
            )
        return
    if spec.dataset != "survival":
        raise RuntimeError(f"unsupported dataset '{spec.dataset}' for '{spec.name}'")
    if spec.backend in {"rust_survival_transform", "rust_gamlss_survival"}:
        raise RuntimeError(
            f"legacy survival backend '{spec.backend}' is not supported for '{spec.name}'; "
            "use backend='rust_survival' with explicit survival_likelihood and survival_distribution"
        )
    if spec.backend == "rust_survival":
        if spec.survival_likelihood not in SUPPORTED_BIOBANK_SURVIVAL_LIKELIHOODS:
            supported = "|".join(sorted(SUPPORTED_BIOBANK_SURVIVAL_LIKELIHOODS))
            raise RuntimeError(
                f"survival method '{spec.name}' requires survival_likelihood in {supported}"
            )
        if spec.survival_distribution not in SUPPORTED_BIOBANK_SURVIVAL_DISTRIBUTIONS:
            supported = "|".join(sorted(SUPPORTED_BIOBANK_SURVIVAL_DISTRIBUTIONS))
            raise RuntimeError(
                f"survival method '{spec.name}' requires survival_distribution in {supported}"
            )
        if spec.include_sigma:
            raise RuntimeError(
                f"survival method '{spec.name}' cannot use include_sigma; choose survival_likelihood explicitly"
            )
        return
    if spec.backend == "r_mgcv_survival":
        if spec.survival_likelihood is not None or spec.survival_distribution is not None:
            raise RuntimeError(
                f"mgcv survival method '{spec.name}' cannot set survival_likelihood or survival_distribution"
            )
        return
    raise RuntimeError(
        f"unsupported survival backend '{spec.backend}' for '{spec.name}'"
    )


def survival_generation_params(cfg: dict[str, Any]) -> tuple[float, float]:
    shape = float(cfg.get("survival_weibull_shape", 1.65))
    scale = float(cfg.get("survival_weibull_scale", 11.5))
    if not math.isfinite(shape) or shape <= 0.0:
        raise RuntimeError("survival_weibull_shape must be finite and > 0")
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError("survival_weibull_scale must be finite and > 0")
    return shape, scale


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def logistic(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(np.asarray(x, dtype=float), -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


def standardize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    if (not np.isfinite(sd)) or sd < 1e-12:
        sd = 1.0
    return (arr - mu) / sd


def zscore_train_test(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    mu = float(np.mean(train))
    sd = float(np.std(train))
    if (not np.isfinite(sd)) or sd < 1e-12:
        sd = 1.0
    return (train - mu) / sd, (test - mu) / sd, mu, sd


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = (np.asarray(y_true, dtype=float) > 0.5).astype(int)
    p = np.asarray(y_score, dtype=float)
    pos = int(np.sum(y == 1))
    neg = int(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(p)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(p) + 1, dtype=float)
    pos_rank_sum = float(np.sum(ranks[y == 1]))
    return float((pos_rank_sum - pos * (pos + 1) / 2.0) / (pos * neg))


def compute_pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = (np.asarray(y_true, dtype=float) > 0.5).astype(int)
    p = np.asarray(y_score, dtype=float)
    pos = int(np.sum(y == 1))
    if pos == 0:
        return 0.0
    order = np.argsort(-p)
    y_ord = y[order]
    tp = np.cumsum(y_ord == 1)
    fp = np.cumsum(y_ord == 0)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    return float(np.trapezoid(precision, recall))


def classification_confusion_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y = (np.asarray(y_true, dtype=float) > 0.5).astype(int)
    p = np.asarray(y_prob, dtype=float)
    threshold = 0.5
    pred = (p >= threshold).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    recall = sensitivity
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(y), 1)
    balanced_accuracy = 0.5 * (sensitivity + specificity)
    return {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "precision": float(precision),
        "f1": float(f1),
    }


def compute_logloss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def compute_brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    return float(np.mean((y - p) ** 2))


def compute_nagelkerke(y_true: np.ndarray, y_prob: np.ndarray, null_mean: float) -> float | None:
    y = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), 1e-12, 1.0 - 1e-12)
    if y.size == 0 or null_mean <= 0.0 or null_mean >= 1.0:
        return None
    ll_null = float(np.sum(y * math.log(null_mean) + (1.0 - y) * math.log(1.0 - null_mean)))
    ll_model = float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))
    n = int(y.size)
    r2_cs = 1.0 - math.exp(min(700.0, (2.0 / n) * (ll_null - ll_model)))
    max_r2_cs = 1.0 - math.exp(min(700.0, (2.0 / n) * ll_null))
    if max_r2_cs <= 0.0 or (not np.isfinite(r2_cs)):
        return None
    return float(r2_cs / max_r2_cs)


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 20) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        lo = bins[i]
        hi = bins[i + 1]
        if i == n_bins - 1:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not np.any(mask):
            continue
        total += float(np.mean(mask)) * abs(float(np.mean(y[mask])) - float(np.mean(p[mask])))
    return float(total)


def _survival_score_grid(train_times: np.ndarray) -> np.ndarray:
    vals = np.asarray(train_times, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0.0)]
    if vals.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    grid = np.unique(vals)
    if grid.size > MAX_SURVIVAL_GRID_POINTS - 1:
        probs = np.linspace(0.0, 1.0, MAX_SURVIVAL_GRID_POINTS - 1, dtype=float)
        grid = np.unique(np.quantile(grid, probs, method="linear"))
    if grid[0] > 0.0:
        grid = np.concatenate([[0.0], grid])
    else:
        grid[0] = 0.0
    if grid.size == 1:
        grid = np.array([0.0, max(float(grid[0]), 1.0)], dtype=float)
    return grid


def _repeat_survival_curve(curve: np.ndarray, n_rows: int) -> np.ndarray:
    base = np.asarray(curve, dtype=float).reshape(1, -1)
    return np.repeat(base, n_rows, axis=0)


def _lifelines_concordance(event_times: np.ndarray, risk_score: np.ndarray, events: np.ndarray) -> float:
    try:
        from lifelines.utils import concordance_index
    except ModuleNotFoundError as exc:
        raise RuntimeError("lifelines is required for survival scoring") from exc
    return float(concordance_index(event_times, -np.asarray(risk_score, dtype=float), event_observed=events))


def _survival_null_curve(train_times: np.ndarray, train_events: np.ndarray, grid: np.ndarray) -> np.ndarray:
    try:
        from lifelines import KaplanMeierFitter
    except ModuleNotFoundError as exc:
        raise RuntimeError("lifelines is required for survival scoring") from exc
    kmf = KaplanMeierFitter()
    kmf.fit(train_times, event_observed=train_events)
    surv = kmf.predict(grid).to_numpy(dtype=float)
    surv[0] = 1.0
    surv = np.clip(surv, 1e-12, 1.0)
    return np.minimum.accumulate(surv)


def calibrated_survival_matrix(
    train_times: np.ndarray,
    train_events: np.ndarray,
    train_risk: np.ndarray,
    test_risk: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    try:
        import pandas as pd
        from lifelines import CoxPHFitter, KaplanMeierFitter
    except ModuleNotFoundError as exc:
        raise RuntimeError("pandas and lifelines are required for survival scoring") from exc
    tr_times = np.asarray(train_times, dtype=float)
    tr_events = (np.asarray(train_events, dtype=float) > 0.5).astype(float)
    tr_risk = np.asarray(train_risk, dtype=float)
    te_risk = np.asarray(test_risk, dtype=float)
    finite_mask = np.isfinite(tr_times) & np.isfinite(tr_events) & np.isfinite(tr_risk) & (tr_times > 0.0)
    if int(np.sum(finite_mask)) == 0:
        return _repeat_survival_curve(np.ones_like(grid, dtype=float), te_risk.shape[0])
    tr_times = tr_times[finite_mask]
    tr_events = tr_events[finite_mask]
    tr_risk = tr_risk[finite_mask]
    if tr_risk.size < 2 or float(np.nanstd(tr_risk)) < 1e-12:
        kmf = KaplanMeierFitter()
        kmf.fit(tr_times, event_observed=tr_events)
        surv = kmf.predict(grid).to_numpy(dtype=float)
        surv[0] = 1.0
        surv = np.clip(surv, 1e-12, 1.0)
        return _repeat_survival_curve(np.minimum.accumulate(surv), te_risk.shape[0])
    calib_train = pd.DataFrame({"__time": tr_times, "__event": tr_events, "__risk": tr_risk})
    calib_test = pd.DataFrame({"__risk": te_risk})
    cph = CoxPHFitter(penalizer=1e-8)
    cph.fit(calib_train, duration_col="__time", event_col="__event")
    surv_df = cph.predict_survival_function(calib_test, times=grid)
    surv = surv_df.to_numpy(dtype=float).T
    surv[:, 0] = 1.0
    surv = np.clip(surv, 1e-12, 1.0)
    return np.minimum.accumulate(surv, axis=1)


def survival_lifted_metrics(
    event_times: np.ndarray,
    events: np.ndarray,
    grid: np.ndarray,
    survival_matrix: np.ndarray,
    null_survival_matrix: np.ndarray,
) -> dict[str, float | None]:
    times = np.asarray(event_times, dtype=float).reshape(-1)
    obs = (np.asarray(events, dtype=float).reshape(-1) > 0.5)
    surv = np.asarray(survival_matrix, dtype=float)
    null_surv = np.asarray(null_survival_matrix, dtype=float)
    if surv.ndim != 2 or surv.shape[0] != times.shape[0]:
        return {"brier": None, "logloss": None, "lifted_brier": None, "lifted_logloss": None, "nagelkerke_r2": None}
    dt = np.diff(grid)
    if grid.shape[0] < 2 or not np.all(dt > 0.0):
        return {"brier": None, "logloss": None, "lifted_brier": None, "lifted_logloss": None, "nagelkerke_r2": None}
    surv = np.clip(surv, 1e-12, 1.0)
    null_surv = np.clip(null_surv, 1e-12, 1.0)
    surv[:, 0] = 1.0
    null_surv[:, 0] = 1.0
    surv = np.minimum.accumulate(surv, axis=1)
    null_surv = np.minimum.accumulate(null_surv, axis=1)
    cumhaz = -np.log(surv)
    null_cumhaz = -np.log(null_surv)
    haz = np.maximum(np.diff(cumhaz, axis=1) / dt.reshape(1, -1), 0.0)
    null_haz = np.maximum(np.diff(null_cumhaz, axis=1) / dt.reshape(1, -1), 0.0)
    haz_sq_prefix = np.concatenate(
        [np.zeros((surv.shape[0], 1), dtype=float), np.cumsum((haz ** 2) * dt.reshape(1, -1), axis=1)],
        axis=1,
    )
    null_sq_prefix = np.concatenate(
        [np.zeros((null_surv.shape[0], 1), dtype=float), np.cumsum((null_haz ** 2) * dt.reshape(1, -1), axis=1)],
        axis=1,
    )
    brier_losses = np.empty(times.shape[0], dtype=float)
    log_losses = np.empty(times.shape[0], dtype=float)
    null_log_losses = np.empty(times.shape[0], dtype=float)
    null_brier_losses = np.empty(times.shape[0], dtype=float)
    for i, z in enumerate(times):
        j = int(np.searchsorted(grid, z, side="left"))
        if j >= grid.shape[0]:
            j = grid.shape[0] - 1
        if abs(grid[j] - z) <= 1e-12:
            idx = max(j - 1, 0)
            hz = haz[i, idx]
            hcum = cumhaz[i, j]
            h2_int = haz_sq_prefix[i, j]
            hz_null = null_haz[i, idx]
            hcum_null = null_cumhaz[i, j]
            h2_null = null_sq_prefix[i, j]
        else:
            idx = max(j - 1, 0)
            elapsed = z - grid[idx]
            hz = haz[i, idx]
            hcum = cumhaz[i, idx] + hz * elapsed
            h2_int = haz_sq_prefix[i, idx] + (hz ** 2) * elapsed
            hz_null = null_haz[i, idx]
            hcum_null = null_cumhaz[i, idx] + hz_null * elapsed
            h2_null = null_sq_prefix[i, idx] + (hz_null ** 2) * elapsed
        log_losses[i] = float(hcum - (math.log(max(hz, 1e-12)) if obs[i] else 0.0))
        null_log_losses[i] = float(hcum_null - (math.log(max(hz_null, 1e-12)) if obs[i] else 0.0))
        brier_losses[i] = float(0.5 * h2_int - (hz if obs[i] else 0.0))
        null_brier_losses[i] = float(0.5 * h2_null - (hz_null if obs[i] else 0.0))
    brier = float(np.mean(brier_losses))
    logloss = float(np.mean(log_losses))
    null_brier = float(np.mean(null_brier_losses))
    null_logloss = float(np.mean(null_log_losses))
    ll_model = float(-np.sum(log_losses))
    ll_null = float(-np.sum(null_log_losses))
    n = int(times.shape[0])
    r2_cs = 1.0 - math.exp(min(700.0, (2.0 / n) * (ll_null - ll_model)))
    max_r2_cs = 1.0 - math.exp(min(700.0, (2.0 / n) * ll_null))
    nag = float(r2_cs / max_r2_cs) if max_r2_cs > 0.0 and np.isfinite(r2_cs) else None
    return {
        "brier": brier,
        "logloss": logloss,
        "lifted_brier": float((null_brier - brier) / max(abs(null_brier), 1e-12)),
        "lifted_logloss": float((null_logloss - logloss) / max(abs(null_logloss), 1e-12)),
        "nagelkerke_r2": nag,
    }


def classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, train_prev: float) -> dict[str, float | None]:
    metrics = {
        "auc": compute_auc(y_true, y_prob),
        "pr_auc": compute_pr_auc(y_true, y_prob),
        "brier": compute_brier(y_true, y_prob),
        "logloss": compute_logloss(y_true, y_prob),
        "nagelkerke_r2": compute_nagelkerke(y_true, y_prob, train_prev),
        "ece": ece_score(y_true, y_prob),
        "mean_pred": float(np.mean(y_prob)),
        "mean_obs": float(np.mean(y_true)),
    }
    metrics.update(classification_confusion_metrics(y_true, y_prob))
    return metrics


def survival_metrics(
    train_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    train_risk: np.ndarray,
    test_risk: np.ndarray,
) -> dict[str, float | None]:
    train_times = np.array([float(r["time"]) for r in train_rows], dtype=float)
    train_events = np.array([float(r["event"]) for r in train_rows], dtype=float)
    test_times = np.array([float(r["time"]) for r in test_rows], dtype=float)
    test_events = np.array([float(r["event"]) for r in test_rows], dtype=float)
    grid = _survival_score_grid(train_times)
    surv = calibrated_survival_matrix(train_times, train_events, train_risk, test_risk, grid)
    null_curve = _survival_null_curve(train_times, train_events, grid)
    proper = survival_lifted_metrics(test_times, test_events, grid, surv, _repeat_survival_curve(null_curve, len(test_rows)))
    horizon = float(np.median(train_times))
    horizon_idx = min(int(np.searchsorted(grid, horizon, side="left")), grid.shape[0] - 1)
    horizon_surv = surv[:, horizon_idx]
    y_horizon = ((test_events > 0.5) & (test_times <= horizon)).astype(float)
    return {
        "c_index": _lifelines_concordance(test_times, test_risk, test_events),
        "auc": _lifelines_concordance(test_times, test_risk, test_events),
        "brier": proper["brier"],
        "logloss": proper["logloss"],
        "lifted_brier": proper["lifted_brier"],
        "lifted_logloss": proper["lifted_logloss"],
        "nagelkerke_r2": proper["nagelkerke_r2"],
        "horizon_years": horizon,
        "horizon_auc": compute_auc(y_horizon, 1.0 - horizon_surv),
        "horizon_brier": compute_brier(y_horizon, 1.0 - horizon_surv),
        "event_rate": float(np.mean(test_events)),
    }


def ps_snapshot(pid: int) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            ["ps", "-p", str(pid), "-o", "pid=,%cpu=,%mem=,rss=,vsz=,etimes=,stat=,comm="],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        line = proc.stdout.strip()
        if not line:
            return {}
        parts = line.split(None, 7)
        if len(parts) < 8:
            return {}
        return {
            "pid": parts[0],
            "cpu_pct": parts[1],
            "mem_pct": parts[2],
            "rss_kib": int(parts[3]) if parts[3].isdigit() else None,
            "vsz_kib": int(parts[4]) if parts[4].isdigit() else None,
            "etimes": parts[5],
            "stat": parts[6],
            "comm": parts[7],
        }
    except Exception:
        return {}


def fmt_kib(kib: Any) -> str:
    if kib is None:
        return "n/a"
    return f"{float(kib) / (1024.0 * 1024.0):.2f} GiB"


def heartbeat_loop(proc: subprocess.Popen[bytes], cmd_preview: str, stop_event: threading.Event) -> None:
    start = time.monotonic()
    while True:
        elapsed = time.monotonic() - start
        snap = ps_snapshot(proc.pid)
        print(
            f"[HEARTBEAT] elapsed={elapsed:8.1f}s cmd='{cmd_preview}' pid={proc.pid} "
            f"cpu={snap.get('cpu_pct', 'n/a')}% mem={snap.get('mem_pct', 'n/a')}% "
            f"rss={fmt_kib(snap.get('rss_kib'))} vsz={fmt_kib(snap.get('vsz_kib'))}",
            file=sys.stderr,
            flush=True,
        )
        wait_sec = HEARTBEAT_INITIAL_INTERVAL_SEC if elapsed < HEARTBEAT_INITIAL_WINDOW_SEC else HEARTBEAT_INTERVAL_SEC
        if stop_event.wait(wait_sec):
            break
        if proc.poll() is not None:
            break


def run_cmd_stream(cmd: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd is not None else None,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=False,
        bufsize=0,
    )
    out_buf: list[str] = []
    err_buf: list[str] = []
    stop_event = threading.Event()
    preview = " ".join(cmd[:5]) + (" ..." if len(cmd) > 5 else "")

    def pump(pipe: Any, sink: Any, capture: list[str]) -> None:
        total = 0
        try:
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    break
                text = chunk.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
                sink.write(text)
                sink.flush()
                capture.append(text)
                total += len(text)
                if total > MAX_CAPTURE_CHARS:
                    del capture[0]
                    total = sum(len(x) for x in capture)
        finally:
            pipe.close()

    t_out = threading.Thread(target=pump, args=(proc.stdout, sys.stdout, out_buf), daemon=True)
    t_err = threading.Thread(target=pump, args=(proc.stderr, sys.stderr, err_buf), daemon=True)
    t_hb = threading.Thread(target=heartbeat_loop, args=(proc, preview, stop_event), daemon=True)
    t_out.start()
    t_err.start()
    t_hb.start()
    rc = proc.wait()
    stop_event.set()
    t_out.join()
    t_err.join()
    t_hb.join(timeout=1.0)
    return rc, "".join(out_buf), "".join(err_buf)


def tool_exists(name: str) -> bool:
    return shutil.which(name) is not None


def load_or_build_rust_binary() -> Path:
    prebuilt = ROOT / "target" / "release" / "gam"
    if prebuilt.exists():
        return prebuilt
    rc, out, err = run_cmd_stream(["cargo", "build", "--release", "--bin", "gam"], cwd=ROOT)
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or "failed to build Rust binary")
    if not prebuilt.exists():
        raise RuntimeError(f"missing Rust binary at {prebuilt}")
    return prebuilt


def subpop_templates() -> list[dict[str, Any]]:
    return [
        {"subpop": "CEU_Utah", "continent": "Europe", "superpop": "EUR", "lat": 40.76, "lon": -111.89},
        {"subpop": "GBR_England", "continent": "Europe", "superpop": "EUR", "lat": 52.36, "lon": -1.17},
        {"subpop": "TSI_Italy", "continent": "Europe", "superpop": "EUR", "lat": 43.77, "lon": 11.25},
        {"subpop": "YRI_Nigeria", "continent": "Africa", "superpop": "AFR", "lat": 6.52, "lon": 3.37},
        {"subpop": "LWK_Kenya", "continent": "Africa", "superpop": "AFR", "lat": -0.02, "lon": 37.91},
        {"subpop": "GWD_Gambia", "continent": "Africa", "superpop": "AFR", "lat": 13.45, "lon": -16.58},
        {"subpop": "CHB_Beijing", "continent": "Asia", "superpop": "EAS", "lat": 39.90, "lon": 116.40},
        {"subpop": "JPT_Tokyo", "continent": "Asia", "superpop": "EAS", "lat": 35.68, "lon": 139.65},
        {"subpop": "KHV_HCMC", "continent": "Asia", "superpop": "EAS", "lat": 10.82, "lon": 106.63},
        {"subpop": "GIH_Houston", "continent": "South Asia", "superpop": "SAS", "lat": 29.76, "lon": -95.37},
        {"subpop": "PJL_Lahore", "continent": "South Asia", "superpop": "SAS", "lat": 31.55, "lon": 74.34},
        {"subpop": "BEB_Dhaka", "continent": "South Asia", "superpop": "SAS", "lat": 23.81, "lon": 90.41},
        {"subpop": "MXL_LA", "continent": "Admixed America", "superpop": "AMR", "lat": 34.05, "lon": -118.24},
        {"subpop": "PEL_Lima", "continent": "Admixed America", "superpop": "AMR", "lat": -12.05, "lon": -77.04},
        {"subpop": "PUR_SanJuan", "continent": "Admixed America", "superpop": "AMR", "lat": 18.47, "lon": -66.11},
    ]


def build_pc_means(templates: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for idx, tpl in enumerate(templates):
        base = np.zeros(16, dtype=float)
        continent_block = idx // 3
        base[continent_block] = 2.5
        base[(continent_block + 5) % 16] = -1.2
        base[(2 * continent_block + 7) % 16] = 0.8
        if tpl["superpop"] == "AFR":
            base[8] = 1.7
        elif tpl["superpop"] == "EAS":
            base[9] = -1.6
        elif tpl["superpop"] == "SAS":
            base[10] = 1.1
        elif tpl["superpop"] == "AMR":
            base[11] = -1.1
        else:
            base[12] = 0.7
        out[str(tpl["subpop"])] = base
    return out


def sample_covariance(pc_means: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    jitter = rng.normal(scale=0.06, size=(16, 16))
    a = np.eye(16) * 0.55 + (jitter @ jitter.T) / 16.0
    return a


def disease_probability(lat: np.ndarray, lon: np.ndarray, pcs: np.ndarray, pgs: np.ndarray, age: np.ndarray, sex: np.ndarray) -> np.ndarray:
    lat_s = standardize(lat)
    lon_s = standardize(lon)
    linear = (
        -0.8
        + 0.9 * standardize(pgs)
        + 0.45 * pcs[:, 0]
        - 0.35 * pcs[:, 1]
        + 0.18 * standardize(age)
        + 0.22 * sex
        + 0.55 * np.sin(lat_s * 1.7)
        + 0.40 * np.cos(lon_s * 2.1)
        + 0.25 * lat_s * lon_s
    )
    return logistic(linear)


def survival_scale(
    lat: np.ndarray,
    lon: np.ndarray,
    pcs: np.ndarray,
    pgs: np.ndarray,
    age: np.ndarray,
    sex: np.ndarray,
) -> np.ndarray:
    lat_s = standardize(lat)
    lon_s = standardize(lon)
    lp = (
        0.35 * standardize(pgs)
        + 0.20 * pcs[:, 0]
        - 0.12 * pcs[:, 2]
        + 0.10 * standardize(age)
        + 0.08 * sex
        + 0.30 * np.sin(lat_s * 1.2)
        - 0.22 * np.cos(lon_s * 1.8)
    )
    return np.exp(lp)


def generate_raw_cohort(cfg: dict[str, Any], out_dir: Path, smoke: bool) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seed = int(cfg["seed"])
    rng = np.random.default_rng(seed)
    base_n = int(cfg["raw_subpop_n"])
    templates = subpop_templates()
    pc_means = build_pc_means(templates)
    rows: list[dict[str, Any]] = []
    subject_id = 0
    for tpl in templates:
        mean = pc_means[tpl["subpop"]]
        cov = sample_covariance(mean, rng)
        n_local = base_n if not smoke else max(48, base_n // 5)
        pcs = rng.multivariate_normal(mean=mean, cov=cov, size=n_local)
        for row_idx in range(n_local):
            subject_id += 1
            age_entry = rng.normal(56.0, 6.5)
            sex = int(rng.integers(0, 2))
            lat_true = tpl["lat"] + rng.normal(0.0, 0.85)
            lon_true = tpl["lon"] + rng.normal(0.0, 0.95)
            lat_obs = lat_true if rng.random() < float(cfg["observed_latlon_fraction"]) else math.nan
            lon_obs = lon_true if math.isfinite(lat_obs) else math.nan
            pgs = 0.55 * pcs[row_idx, 0] - 0.25 * pcs[row_idx, 2] + rng.normal(0.0, 1.0)
            rows.append(
                {
                    "subject_id": subject_id,
                    "subpopulation": tpl["subpop"],
                    "continent": tpl["continent"],
                    "superpopulation": tpl["superpop"],
                    "age_entry": float(age_entry),
                    "sex": sex,
                    "lat_true": float(lat_true),
                    "lon_true": float(lon_true),
                    "lat_obs": None if not math.isfinite(lat_obs) else float(lat_obs),
                    "lon_obs": None if not math.isfinite(lon_obs) else float(lon_obs),
                    "pgs_raw": float(pgs),
                    **{f"pc{pc_idx + 1}": float(pcs[row_idx, pc_idx]) for pc_idx in range(16)},
                }
            )
    meta = {
        "seed": seed,
        "raw_n": len(rows),
        "subpopulations": [tpl["subpop"] for tpl in templates],
    }
    dump_json(out_dir / "raw_generation_metadata.json", meta)
    return rows, meta


def impute_and_upsample(rows: list[dict[str, Any]], cfg: dict[str, Any], smoke: bool) -> list[dict[str, Any]]:
    target_n = int(cfg["smoke_target_n"] if smoke else cfg["target_n"])
    split_rng = np.random.default_rng(int(cfg["split_seed"]))
    by_subpop: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_subpop[str(row["subpopulation"])].append(row)
    out = [dict(r) for r in rows]
    templates = {str(r["subpopulation"]): r for r in rows[: len(by_subpop)]}
    batch = int(cfg.get("upsample_batch_size", 5000))
    next_id = max(int(r["subject_id"]) for r in out) + 1
    subpops = sorted(by_subpop.keys())
    weights = np.array([len(by_subpop[s]) for s in subpops], dtype=float)
    weights /= np.sum(weights)
    while len(out) < target_n:
        remaining = target_n - len(out)
        step = min(batch, remaining)
        sampled_subpops = split_rng.choice(subpops, size=step, replace=True, p=weights)
        for sp in sampled_subpops:
            source = by_subpop[sp][int(split_rng.integers(0, len(by_subpop[sp])))]
            row = dict(source)
            row["subject_id"] = next_id
            next_id += 1
            row["age_entry"] = float(np.clip(float(row["age_entry"]) + split_rng.normal(0.0, 0.9), 35.0, 82.0))
            row["pgs_raw"] = float(float(row["pgs_raw"]) + split_rng.normal(0.0, 0.12))
            lat_true = float(row["lat_true"]) + float(split_rng.normal(0.0, 0.05))
            lon_true = float(row["lon_true"]) + float(split_rng.normal(0.0, 0.05))
            row["lat_true"] = lat_true
            row["lon_true"] = lon_true
            if split_rng.random() < float(cfg["observed_latlon_fraction"]):
                row["lat_obs"] = lat_true + float(split_rng.normal(0.0, 0.02))
                row["lon_obs"] = lon_true + float(split_rng.normal(0.0, 0.02))
            else:
                row["lat_obs"] = None
                row["lon_obs"] = None
            out.append(row)
    ref_subpop = str(cfg["reference_subpopulation"])
    ref_rows = [r for r in out if str(r["subpopulation"]) == ref_subpop]
    ref_lat = float(np.mean([float(r["lat_true"]) for r in ref_rows]))
    ref_lon = float(np.mean([float(r["lon_true"]) for r in ref_rows]))
    for row in out:
        lat_obs = row.get("lat_obs")
        lon_obs = row.get("lon_obs")
        row["lat_final"] = float(lat_obs) if lat_obs is not None else ref_lat
        row["lon_final"] = float(lon_obs) if lon_obs is not None else ref_lon
    return out


def attach_outcomes(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    rng = np.random.default_rng(int(cfg["seed"]) + 17)
    lat = np.array([float(r["lat_final"]) for r in rows], dtype=float)
    lon = np.array([float(r["lon_final"]) for r in rows], dtype=float)
    pcs = np.column_stack([np.array([float(r[f"pc{i}"]) for r in rows], dtype=float) for i in range(1, 17)])
    pgs = np.array([float(r["pgs_raw"]) for r in rows], dtype=float)
    age = np.array([float(r["age_entry"]) for r in rows], dtype=float)
    sex = np.array([float(r["sex"]) for r in rows], dtype=float)
    disease_prob = disease_probability(lat, lon, pcs, pgs, age, sex)
    disease = rng.binomial(1, disease_prob).astype(int)
    shape, scale = survival_generation_params(cfg)
    surv_scale = survival_scale(lat, lon, pcs, pgs, age, sex)
    u = np.clip(rng.random(len(rows)), 1e-12, 1.0 - 1e-12)
    event_time = scale * surv_scale * (-np.log(1.0 - u)) ** (1.0 / shape)
    censor_time = rng.uniform(4.5, 14.5, size=len(rows))
    observed_time = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(int)
    for idx, row in enumerate(rows):
        row["phenotype_prob"] = float(disease_prob[idx])
        row["phenotype"] = int(disease[idx])
        row["time"] = float(observed_time[idx])
        row["event"] = int(event[idx])
    return rows


def write_cohort_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "subject_id",
        "subpopulation",
        "continent",
        "superpopulation",
        "age_entry",
        "sex",
        "lat_true",
        "lon_true",
        "lat_final",
        "lon_final",
        "pgs_raw",
        *[f"pc{i}" for i in range(1, 17)],
        "phenotype_prob",
        "phenotype",
        "time",
        "event",
        "lat_final_std",
        "lon_final_std",
        "age_entry_std",
        "pgs_std",
        *[f"pc{i}_std" for i in range(1, 17)],
    ]
    write_csv_rows(path, rows, fieldnames)


def split_rows(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(cfg["split_seed"]))
    idx = np.arange(len(rows))
    rng.shuffle(idx)
    split = int(round(float(cfg["train_fraction"]) * len(rows)))
    train_idx = set(int(i) for i in idx[:split])
    train = [rows[i] for i in range(len(rows)) if i in train_idx]
    test = [rows[i] for i in range(len(rows)) if i not in train_idx]
    return train, test


def add_standardized_columns(train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]]) -> None:
    numeric_cols = ["age_entry", "lat_final", "lon_final", "pgs_raw", *[f"pc{i}" for i in range(1, 17)]]
    for col in numeric_cols:
        tr = np.array([float(r[col]) for r in train_rows], dtype=float)
        te = np.array([float(r[col]) for r in test_rows], dtype=float)
        tr_std, te_std, _, _ = zscore_train_test(tr, te)
        out_col = col.replace("_raw", "") + "_std"
        for i, row in enumerate(train_rows):
            row[out_col] = float(tr_std[i])
        for i, row in enumerate(test_rows):
            row[out_col] = float(te_std[i])


def do_prepare(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    if args.target_n is not None:
        cfg["target_n"] = int(args.target_n)
    if args.smoke_target_n is not None:
        cfg["smoke_target_n"] = int(args.smoke_target_n)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, raw_meta = generate_raw_cohort(cfg, out_dir, args.smoke)
    rows = impute_and_upsample(rows, cfg, args.smoke)
    rows = attach_outcomes(rows, cfg)
    train_rows, test_rows = split_rows(rows, cfg)
    add_standardized_columns(train_rows, test_rows)
    write_cohort_csv(out_dir / "all_cohort.csv", rows)
    write_cohort_csv(out_dir / "disease_train.csv", train_rows)
    write_cohort_csv(out_dir / "disease_test.csv", test_rows)
    write_cohort_csv(out_dir / "survival_train.csv", train_rows)
    write_cohort_csv(out_dir / "survival_test.csv", test_rows)
    prep_meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "raw_generation": raw_meta,
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_test": len(test_rows),
        "smoke": bool(args.smoke),
    }
    dump_json(out_dir / "prep_metadata.json", prep_meta)
    print(f"Wrote prepared data to {out_dir}")
    return 0


def build_method_specs(cfg: dict[str, Any]) -> list[MethodSpec]:
    out = []
    for item in cfg.get("methods", []):
        spec = MethodSpec(
            name=str(item["name"]),
            dataset=str(item["dataset"]),
            backend=str(item["backend"]),
            family=str(item["family"]),
            spatial_basis=str(item["spatial_basis"]),
            centers=int(item["centers"]) if item.get("centers") is not None else None,
            smooth_kind=str(item.get("smooth_kind", "joint")),
            include_sigma=bool(item.get("include_sigma", False)),
            survival_likelihood=(
                str(item["survival_likelihood"])
                if item.get("survival_likelihood") is not None
                else None
            ),
            survival_distribution=(
                str(item["survival_distribution"])
                if item.get("survival_distribution") is not None
                else None
            ),
            marginal_slope=bool(item.get("marginal_slope", False)),
            scale_dimensions=bool(item.get("scale_dimensions", False)),
            z_column=(
                str(item["z_column"])
                if item.get("z_column") is not None
                else None
            ),
            pc_count=int(item.get("pc_count", 16)),
            mean_linkwiggle_knots=(
                int(item["mean_linkwiggle_knots"])
                if item.get("mean_linkwiggle_knots") is not None
                else None
            ),
            logslope_linkwiggle_knots=(
                int(item["logslope_linkwiggle_knots"])
                if item.get("logslope_linkwiggle_knots") is not None
                else None
            ),
            timewiggle_knots=(
                int(item["timewiggle_knots"])
                if item.get("timewiggle_knots") is not None
                else None
            ),
            max_centers=(
                int(item["max_centers"])
                if item.get("max_centers") is not None
                else None
            ),
        )
        validate_method_spec(spec)
        out.append(spec)
    return out


def count_csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return max(sum(1 for _ in fh) - 1, 0)


def effective_marginal_slope_centers(spec: MethodSpec, train_rows: int) -> int:
    centers = int(spec.centers or 50)
    if spec.max_centers is not None:
        centers = min(centers, int(spec.max_centers))
    if train_rows >= 250000:
        centers = min(centers, 28)
    if train_rows >= 350000:
        centers = min(centers, 24)
    if spec.mean_linkwiggle_knots is not None or spec.logslope_linkwiggle_knots is not None:
        centers = min(centers, 22)
    return max(10, centers)


def rust_formula_classification(spec: MethodSpec) -> tuple[str, str]:
    if spec.smooth_kind == "joint":
        if spec.spatial_basis == "thinplate":
            spatial = f"thinplate(lat_final_std, lon_final_std, knots={int(spec.centers or 60)})"
        elif spec.spatial_basis == "duchon":
            spatial = f"duchon(lat_final_std, lon_final_std, centers={int(spec.centers or 60)})"
        elif spec.spatial_basis == "matern":
            spatial = f"matern(lat_final_std, lon_final_std, centers={int(spec.centers or 60)})"
        else:
            raise RuntimeError(f"unsupported Rust spatial basis '{spec.spatial_basis}'")
        base_terms = [
            "pgs_std",
            "sex",
            "smooth(age_entry_std)",
            spatial,
            *[f"pc{i}_std" for i in range(1, 17)],
        ]
        mean_formula = "phenotype ~ " + " + ".join(base_terms)
        sigma_formula = " + ".join(
            ["smooth(age_entry_std)", spatial, *[f"pc{i}_std" for i in range(1, 5)]]
        )
        return mean_formula, sigma_formula
    mean_terms = [
        "pgs_std",
        "sex",
        "smooth(age_entry_std)",
        "smooth(lat_final_std)",
        "smooth(lon_final_std)",
        *[f"pc{i}_std" for i in range(1, 17)],
    ]
    sigma_terms = [
        "smooth(age_entry_std)",
        "smooth(lat_final_std)",
        "smooth(lon_final_std)",
        *[f"pc{i}_std" for i in range(1, 5)],
    ]
    return "phenotype ~ " + " + ".join(mean_terms), " + ".join(sigma_terms)


def rust_marginal_slope_formula_classification(spec: MethodSpec, centers: int) -> tuple[str, str]:
    """Build mean and logslope formulas for 16D marginal-slope Duchon classification."""
    pc_cols = ", ".join(f"pc{i}_std" for i in range(1, int(spec.pc_count) + 1))
    if spec.spatial_basis == "duchon":
        spatial = f"duchon({pc_cols}, centers={centers}, order=0, power=1)"
    elif spec.spatial_basis == "matern":
        spatial = f"matern({pc_cols}, centers={centers})"
    else:
        raise RuntimeError(
            f"unsupported marginal-slope spatial basis '{spec.spatial_basis}' (use duchon or matern)"
        )
    mean_terms = [
        "sex",
        "age_entry_std",
        spatial,
    ]
    logslope_terms = [
        "age_entry_std",
        spatial,
    ]
    if spec.mean_linkwiggle_knots is not None:
        mean_terms.append(f"linkwiggle(knots={int(spec.mean_linkwiggle_knots)})")
    if spec.logslope_linkwiggle_knots is not None:
        logslope_terms.append(f"linkwiggle(knots={int(spec.logslope_linkwiggle_knots)})")
    mean_formula = "phenotype ~ " + " + ".join(mean_terms)
    logslope_formula = " + ".join(logslope_terms)
    return mean_formula, logslope_formula


def run_rust_marginal_slope_classification(
    spec: MethodSpec,
    train_csv: Path,
    test_csv: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Run 16D marginal-slope Duchon classification with optional anisotropy."""
    rust_bin = load_or_build_rust_binary()
    train_rows = count_csv_rows(train_csv)
    centers = effective_marginal_slope_centers(spec, train_rows)
    mean_formula, logslope_formula = rust_marginal_slope_formula_classification(spec, centers)
    z_column = spec.z_column or "pgs_std"
    model_path = out_dir / f"{spec.name}.model.json"
    pred_path = out_dir / f"{spec.name}.pred.csv"
    fit_cmd = [
        str(rust_bin), "fit",
        "--logslope-formula", logslope_formula,
        "--z-column", z_column,
        "--out", str(model_path),
    ]
    if spec.scale_dimensions:
        fit_cmd.append("--scale-dimensions")
    fit_cmd.extend([str(train_csv), mean_formula])
    t0 = time.perf_counter()
    rc, out, err = run_cmd_stream(fit_cmd, cwd=ROOT)
    fit_sec = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or f"{spec.name} marginal-slope fit failed")
    pred_cmd = [str(rust_bin), "predict", str(model_path), str(test_csv), "--out", str(pred_path)]
    t1 = time.perf_counter()
    rc, out, err = run_cmd_stream(pred_cmd, cwd=ROOT)
    predict_sec = time.perf_counter() - t1
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or f"{spec.name} marginal-slope predict failed")
    pred_rows = read_csv_rows(pred_path)
    pred = np.array([float(r["mean"]) for r in pred_rows], dtype=float)
    y_train = csv_numeric_column(train_csv, "phenotype")
    y_test = csv_numeric_column(test_csv, "phenotype")
    metrics = classification_metrics(y_test, pred, float(np.mean(y_train)))
    return {
        "fit_sec": fit_sec,
        "predict_sec": predict_sec,
        "metrics": metrics,
        "prediction_path": str(pred_path),
        "model_spec": (
            f"Rust 16D Duchon marginal-slope"
            f"{' aniso' if spec.scale_dimensions else ''}"
            f" (z={z_column}, centers={centers}) holdout"
        ),
    }


def rust_survival_formula_rhs(spec: MethodSpec) -> str:
    distribution = spec.survival_distribution
    if distribution is None:
        raise RuntimeError(
            f"survival method '{spec.name}' is missing survival_distribution"
        )
    terms = [
        "pgs_std",
        "sex",
        "smooth(age_entry_std)",
        "smooth(lat_final_std)",
        "smooth(lon_final_std)",
        "pc1_std",
        "pc2_std",
        "pc3_std",
        "pc4_std",
        f"survmodel(spec=net, distribution={distribution})",
    ]
    if spec.mean_linkwiggle_knots is not None:
        terms.append(f"linkwiggle(knots={int(spec.mean_linkwiggle_knots)})")
    if spec.timewiggle_knots is not None:
        terms.append(f"timewiggle(knots={int(spec.timewiggle_knots)})")
    return " + ".join(terms)


def rust_survival_formula(spec: MethodSpec) -> str:
    return f"Surv({SURVIVAL_ENTRY_COLUMN}, time, event) ~ {rust_survival_formula_rhs(spec)}"


def survival_eval_horizon_from_rows(rows: list[dict[str, Any]]) -> float:
    times = np.array([float(r["time"]) for r in rows], dtype=float)
    horizon = float(np.median(times))
    if (not np.isfinite(horizon)) or horizon <= 0.0:
        horizon = 1.0
    return horizon


def prepare_survival_benchmark_rows(
    rows: list[dict[str, Any]],
    *,
    prediction_horizon: float | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        prepared = dict(row)
        prepared[SURVIVAL_ENTRY_COLUMN] = 0.0
        if prediction_horizon is not None:
            prepared["time"] = float(prediction_horizon)
        out.append(prepared)
    return out


def write_survival_benchmark_csv(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    prediction_horizon: float | None = None,
) -> None:
    prepared_rows = prepare_survival_benchmark_rows(
        rows,
        prediction_horizon=prediction_horizon,
    )
    if not prepared_rows:
        raise RuntimeError(f"cannot write empty survival benchmark frame to {path}")
    fieldnames = [SURVIVAL_ENTRY_COLUMN] + [
        key for key in prepared_rows[0].keys() if key != SURVIVAL_ENTRY_COLUMN
    ]
    write_csv_rows(path, prepared_rows, fieldnames)


def mgcv_formula_classification(spec: MethodSpec) -> str:
    terms = ["pgs_std", "sex", "s(age_entry_std, bs='ps', k=min(10, nrow(train_df)-1))"]
    if spec.smooth_kind == "joint":
        if spec.spatial_basis == "thinplate":
            terms.append(f"s(lat_final_std, lon_final_std, bs='tp', k=min({int(spec.centers or 60)}, nrow(train_df)-1))")
        elif spec.spatial_basis == "duchon":
            terms.append(f"s(lat_final_std, lon_final_std, bs='ds', k=min({int(spec.centers or 60)}, nrow(train_df)-1))")
        elif spec.spatial_basis == "matern":
            terms.append(f"s(lat_final_std, lon_final_std, bs='gp', m=c(-4,1.0), k=min({int(spec.centers or 60)}, nrow(train_df)-1))")
        else:
            raise RuntimeError(f"unsupported mgcv joint basis '{spec.spatial_basis}'")
    else:
        terms.extend(
            [
                "s(lat_final_std, bs='ps', k=min(12, nrow(train_df)-1))",
                "s(lon_final_std, bs='ps', k=min(12, nrow(train_df)-1))",
            ]
        )
    terms.extend(f"pc{i}_std" for i in range(1, 17))
    return "phenotype ~ " + " + ".join(terms)


def mgcv_survival_formula() -> str:
    return (
        "time ~ pgs_std + sex + "
        "s(age_entry_std, bs='ps', k=min(10, nrow(train_df)-1)) + "
        "s(lat_final_std, bs='ps', k=min(12, nrow(train_df)-1)) + "
        "s(lon_final_std, bs='ps', k=min(12, nrow(train_df)-1)) + "
        "pc1_std + pc2_std + pc3_std + pc4_std"
    )


def csv_numeric_column(path: Path, col: str) -> np.ndarray:
    rows = read_csv_rows(path)
    return np.array([float(r[col]) for r in rows], dtype=float)


def run_rust_classification(spec: MethodSpec, train_csv: Path, test_csv: Path, out_dir: Path) -> dict[str, Any]:
    rust_bin = load_or_build_rust_binary()
    mean_formula, sigma_formula = rust_formula_classification(spec)
    model_path = out_dir / f"{spec.name}.model.json"
    pred_path = out_dir / f"{spec.name}.pred.csv"
    fit_cmd = [str(rust_bin), "fit"]
    if spec.include_sigma:
        fit_cmd.extend(["--predict-noise", sigma_formula])
    fit_cmd.extend(["--out", str(model_path), str(train_csv), mean_formula])
    t0 = time.perf_counter()
    rc, out, err = run_cmd_stream(fit_cmd, cwd=ROOT)
    fit_sec = time.perf_counter() - t0
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or f"{spec.name} fit failed")
    pred_cmd = [str(rust_bin), "predict", str(model_path), str(test_csv), "--out", str(pred_path)]
    t1 = time.perf_counter()
    rc, out, err = run_cmd_stream(pred_cmd, cwd=ROOT)
    predict_sec = time.perf_counter() - t1
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or f"{spec.name} predict failed")
    pred_rows = read_csv_rows(pred_path)
    pred = np.array([float(r["mean"]) for r in pred_rows], dtype=float)
    y_train = csv_numeric_column(train_csv, "phenotype")
    y_test = csv_numeric_column(test_csv, "phenotype")
    metrics = classification_metrics(y_test, pred, float(np.mean(y_train)))
    return {
        "fit_sec": fit_sec,
        "predict_sec": predict_sec,
        "metrics": metrics,
        "prediction_path": str(pred_path),
        "model_spec": f"Rust {spec.spatial_basis} {'GAMLSS' if spec.include_sigma else 'GAM'} holdout",
    }


def run_rust_survival(spec: MethodSpec, train_csv: Path, test_csv: Path, out_dir: Path) -> dict[str, Any]:
    rust_bin = load_or_build_rust_binary()
    fit_formula = rust_survival_formula(spec)
    model_path = out_dir / f"{spec.name}.model.json"
    pred_path = out_dir / f"{spec.name}.pred.csv"
    likelihood_mode = spec.survival_likelihood
    if likelihood_mode is None:
        raise RuntimeError(
            f"survival method '{spec.name}' is missing survival_likelihood"
        )
    train_rows_raw = read_csv_rows(train_csv)
    test_rows_raw = read_csv_rows(test_csv)
    horizon = survival_eval_horizon_from_rows(train_rows_raw)
    with tempfile.TemporaryDirectory(prefix="gam_biobank_survival_", dir=out_dir) as td:
        td_path = Path(td)
        train_fit_path = td_path / "train_fit.csv"
        train_pred_input_path = td_path / "train_predict.csv"
        test_pred_input_path = td_path / "test_predict.csv"
        train_pred_path = td_path / "trainpred.csv"
        write_survival_benchmark_csv(train_fit_path, train_rows_raw)
        write_survival_benchmark_csv(
            train_pred_input_path,
            train_rows_raw,
            prediction_horizon=horizon,
        )
        write_survival_benchmark_csv(
            test_pred_input_path,
            test_rows_raw,
            prediction_horizon=horizon,
        )
        fit_cmd = [
            str(rust_bin), "fit",
            "--survival-likelihood", likelihood_mode,
            "--time-basis", "ispline",
            "--time-degree", "3",
            "--time-num-internal-knots", "8",
            "--time-smooth-lambda", "0.01",
            "--ridge-lambda", "1e-6",
            "--out", str(model_path),
            str(train_fit_path),
            fit_formula,
        ]
        if spec.timewiggle_knots is not None:
            fit_cmd.extend(["--baseline-target", "gompertz-makeham"])
        t0 = time.perf_counter()
        rc, out, err = run_cmd_stream(fit_cmd, cwd=ROOT)
        fit_sec = time.perf_counter() - t0
        if rc != 0:
            raise RuntimeError(err.strip() or out.strip() or f"{spec.name} fit failed")
        pred_cmd = [str(rust_bin), "predict", str(model_path), str(test_pred_input_path), "--out", str(pred_path)]
        t1 = time.perf_counter()
        rc, out, err = run_cmd_stream(pred_cmd, cwd=ROOT)
        predict_sec = time.perf_counter() - t1
        if rc != 0:
            raise RuntimeError(err.strip() or out.strip() or f"{spec.name} predict failed")
        pred_rows = read_csv_rows(pred_path)
        risk_cols = ["failure_prob", "risk_score", "eta"]
        risk_key = next((k for k in risk_cols if k in pred_rows[0]), None)
        if risk_key is None:
            raise RuntimeError(f"{spec.name} prediction output missing risk column")
        test_risk = np.array([float(r[risk_key]) for r in pred_rows], dtype=float)
        pred_train_cmd = [str(rust_bin), "predict", str(model_path), str(train_pred_input_path), "--out", str(train_pred_path)]
        rc, out, err = run_cmd_stream(pred_train_cmd, cwd=ROOT)
        if rc != 0:
            raise RuntimeError(err.strip() or out.strip() or f"{spec.name} train predict failed")
        train_pred_rows = read_csv_rows(train_pred_path)
        train_risk = np.array([float(r[risk_key]) for r in train_pred_rows], dtype=float)
    train_rows = [{k: (float(v) if k in {"time", "event"} else v) for k, v in r.items()} for r in train_rows_raw]
    test_rows = [{k: (float(v) if k in {"time", "event"} else v) for k, v in r.items()} for r in test_rows_raw]
    metrics = survival_metrics(train_rows, test_rows, train_risk, test_risk)
    return {
        "fit_sec": fit_sec,
        "predict_sec": predict_sec,
        "metrics": metrics,
        "prediction_path": str(pred_path),
        "model_spec": f"{fit_formula} [survival-likelihood={likelihood_mode}; predict_horizon={horizon:.6g}]",
    }


def run_r_mgcv_classification(spec: MethodSpec, train_csv: Path, test_csv: Path, out_dir: Path) -> dict[str, Any]:
    if not tool_exists("Rscript"):
        raise RuntimeError("Rscript is required for mgcv methods")
    formula = mgcv_formula_classification(spec)
    script = f"""
args <- commandArgs(trailingOnly = TRUE)
train_path <- args[1]
test_path <- args[2]
out_path <- args[3]
suppressPackageStartupMessages({{
  library(mgcv)
  library(jsonlite)
}})
train_df <- read.csv(train_path)
test_df <- read.csv(test_path)
t0 <- proc.time()[["elapsed"]]
fit <- gam(as.formula("{formula}"), family=binomial(link="logit"), data=train_df, method="REML", select=TRUE)
fit_sec <- proc.time()[["elapsed"]] - t0
t1 <- proc.time()[["elapsed"]]
pred <- as.numeric(predict(fit, newdata=test_df, type="response"))
pred_sec <- proc.time()[["elapsed"]] - t1
write.csv(data.frame(mean=pred), out_path, row.names=FALSE)
cat(jsonlite::toJSON(list(fit_sec=fit_sec, predict_sec=pred_sec), auto_unbox=TRUE))
"""
    script_path = out_dir / f"{spec.name}.R"
    script_path.write_text(script, encoding="utf-8")
    pred_path = out_dir / f"{spec.name}.pred.csv"
    rc, out, err = run_cmd_stream(["Rscript", str(script_path), str(train_csv), str(test_csv), str(pred_path)], cwd=ROOT)
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or f"{spec.name} mgcv fit failed")
    meta = json.loads(out.strip().splitlines()[-1])
    pred_rows = read_csv_rows(pred_path)
    pred = np.array([float(r["mean"]) for r in pred_rows], dtype=float)
    y_train = csv_numeric_column(train_csv, "phenotype")
    y_test = csv_numeric_column(test_csv, "phenotype")
    metrics = classification_metrics(y_test, pred, float(np.mean(y_train)))
    return {
        "fit_sec": float(meta["fit_sec"]),
        "predict_sec": float(meta["predict_sec"]),
        "metrics": metrics,
        "prediction_path": str(pred_path),
        "model_spec": f"mgcv {spec.spatial_basis} holdout",
    }


def run_r_mgcv_survival(spec: MethodSpec, train_csv: Path, test_csv: Path, out_dir: Path) -> dict[str, Any]:
    if not tool_exists("Rscript"):
        raise RuntimeError("Rscript is required for mgcv survival methods")
    formula = mgcv_survival_formula()
    script = f"""
args <- commandArgs(trailingOnly = TRUE)
train_path <- args[1]
test_path <- args[2]
out_pred <- args[3]
suppressPackageStartupMessages({{
  library(mgcv)
  library(jsonlite)
}})
train_df <- read.csv(train_path)
test_df <- read.csv(test_path)
t0 <- proc.time()[["elapsed"]]
fit <- gam(as.formula("{formula}"), family=cox.ph(), weights=as.numeric(train_df$event), data=train_df, method="REML", select=TRUE)
fit_sec <- proc.time()[["elapsed"]] - t0
t1 <- proc.time()[["elapsed"]]
train_risk <- as.numeric(predict(fit, newdata=train_df, type="link"))
test_risk <- as.numeric(predict(fit, newdata=test_df, type="link"))
pred_sec <- proc.time()[["elapsed"]] - t1
write.csv(data.frame(train_risk=train_risk), paste0(out_pred, ".train"), row.names=FALSE)
write.csv(data.frame(risk=test_risk), out_pred, row.names=FALSE)
cat(toJSON(list(fit_sec=fit_sec, predict_sec=pred_sec), auto_unbox=TRUE))
"""
    script_path = out_dir / f"{spec.name}.R"
    script_path.write_text(script, encoding="utf-8")
    pred_path = out_dir / f"{spec.name}.pred.csv"
    rc, out, err = run_cmd_stream(["Rscript", str(script_path), str(train_csv), str(test_csv), str(pred_path)], cwd=ROOT)
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or f"{spec.name} mgcv survival fit failed")
    meta = json.loads(out.strip().splitlines()[-1])
    test_risk = np.array([float(r["risk"]) for r in read_csv_rows(pred_path)], dtype=float)
    train_risk = np.array([float(r["train_risk"]) for r in read_csv_rows(Path(str(pred_path) + ".train"))], dtype=float)
    train_rows = [{k: (float(v) if k in {"time", "event"} else v) for k, v in r.items()} for r in read_csv_rows(train_csv)]
    test_rows = [{k: (float(v) if k in {"time", "event"} else v) for k, v in r.items()} for r in read_csv_rows(test_csv)]
    metrics = survival_metrics(train_rows, test_rows, train_risk, test_risk)
    return {
        "fit_sec": float(meta["fit_sec"]),
        "predict_sec": float(meta["predict_sec"]),
        "metrics": metrics,
        "prediction_path": str(pred_path),
        "model_spec": "mgcv cox.ph holdout",
    }


def run_method(spec: MethodSpec, prep_dir: Path, out_dir: Path) -> dict[str, Any]:
    disease_train = prep_dir / "disease_train.csv"
    disease_test = prep_dir / "disease_test.csv"
    survival_train = prep_dir / "survival_train.csv"
    survival_test = prep_dir / "survival_test.csv"
    if spec.dataset == "disease":
        if spec.backend == "rust_gam" and spec.marginal_slope:
            result = run_rust_marginal_slope_classification(spec, disease_train, disease_test, out_dir)
        elif spec.backend == "rust_gam":
            result = run_rust_classification(spec, disease_train, disease_test, out_dir)
        elif spec.backend == "r_mgcv":
            result = run_r_mgcv_classification(spec, disease_train, disease_test, out_dir)
        else:
            raise RuntimeError(f"unsupported disease backend '{spec.backend}'")
    elif spec.dataset == "survival":
        if spec.backend == "rust_survival":
            result = run_rust_survival(spec, survival_train, survival_test, out_dir)
        elif spec.backend == "r_mgcv_survival":
            result = run_r_mgcv_survival(spec, survival_train, survival_test, out_dir)
        else:
            raise RuntimeError(f"unsupported survival backend '{spec.backend}'")
    else:
        raise RuntimeError(f"unsupported dataset '{spec.dataset}'")
    return {
        "method": spec.name,
        "dataset": spec.dataset,
        "family": spec.family,
        **result,
    }


def do_run_method(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    specs = {spec.name: spec for spec in build_method_specs(cfg)}
    if args.method not in specs:
        raise RuntimeError(f"unknown method '{args.method}'")
    spec = specs[args.method]
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        result = run_method(spec, args.prep_dir.resolve(), out_dir)
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "ok",
            **result,
        }
    except Exception as exc:
        payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "method": spec.name,
            "dataset": spec.dataset,
            "family": spec.family,
            "error": str(exc),
        }
    dump_json(args.out_json.resolve(), payload)
    print(f"Wrote {args.out_json}")
    if payload["status"] != "ok":
        return 1
    return 0


def read_json_files(paths: list[Path]) -> list[dict[str, Any]]:
    out = []
    for path in paths:
        out.append(json.loads(path.read_text(encoding="utf-8")))
    return out


def make_metric_table(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table = []
    for row in results:
        metrics = row.get("metrics", {})
        base = {
            "method": row.get("method"),
            "dataset": row.get("dataset"),
            "family": row.get("family"),
            "status": row.get("status"),
            "fit_sec": row.get("fit_sec"),
            "predict_sec": row.get("predict_sec"),
            "model_spec": row.get("model_spec"),
        }
        merged = dict(base)
        merged.update(metrics if isinstance(metrics, dict) else {})
        table.append(merged)
    return table


def plot_aggregate(results: list[dict[str, Any]], prep_dir: Path, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def resolve_prediction_path(raw_path: str) -> Path | None:
        direct = Path(raw_path)
        if direct.exists():
            return direct
        artifacts_root = prep_dir.parent / "artifacts"
        if not artifacts_root.exists():
            return None
        matches = list(artifacts_root.rglob(direct.name))
        return matches[0] if matches else None

    paths: list[Path] = []
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        return paths
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ok:
        grouped[str(row["dataset"])].append(row)
    for dataset, rows in grouped.items():
        if dataset == "disease":
            metrics = [("auc", "AUC"), ("pr_auc", "PR-AUC"), ("brier", "Brier"), ("logloss", "LogLoss"), ("ece", "ECE")]
        else:
            metrics = [("c_index", "C-index"), ("lifted_brier", "Lifted Brier"), ("lifted_logloss", "Lifted LogLoss"), ("brier", "Brier"), ("logloss", "LogLoss")]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
        ax_metrics = axes[0, 0]
        names = [r["method"] for r in rows]
        x = np.arange(len(names))
        width = 0.14
        for i, (key, label) in enumerate(metrics):
            vals = [float(r["metrics"].get(key, np.nan)) for r in rows]
            ax_metrics.bar(x + (i - len(metrics) / 2) * width, vals, width=width, label=label)
        ax_metrics.set_xticks(x)
        ax_metrics.set_xticklabels(names, rotation=25, ha="right")
        ax_metrics.set_title(f"{dataset.title()} metrics")
        ax_metrics.legend(fontsize=8)

        ax_time = axes[0, 1]
        fit = [float(r["fit_sec"]) for r in rows]
        pred = [float(r["predict_sec"]) for r in rows]
        ax_time.bar(x, fit, label="fit_sec")
        ax_time.bar(x, pred, bottom=fit, label="predict_sec")
        ax_time.set_xticks(x)
        ax_time.set_xticklabels(names, rotation=25, ha="right")
        ax_time.set_title("Train/Test runtime")
        ax_time.legend(fontsize=8)

        all_rows = read_csv_rows(prep_dir / "all_cohort.csv")
        lat = np.array([float(r["lat_final"]) for r in all_rows], dtype=float)
        lon = np.array([float(r["lon_final"]) for r in all_rows], dtype=float)
        if dataset == "disease":
            value = np.array([float(r["phenotype_prob"]) for r in all_rows], dtype=float)
            title = "Simulated disease risk map"
        else:
            value = np.array([float(r["event"]) for r in all_rows], dtype=float)
            title = "Observed event map"
        sc = axes[1, 0].scatter(lon[::max(1, len(lon) // 25000)], lat[::max(1, len(lat) // 25000)], c=value[::max(1, len(value) // 25000)], s=5, alpha=0.6, cmap="viridis")
        axes[1, 0].set_xlabel("Longitude")
        axes[1, 0].set_ylabel("Latitude")
        axes[1, 0].set_title(title)
        fig.colorbar(sc, ax=axes[1, 0], fraction=0.046, pad=0.04)

        if dataset == "disease":
            test_rows = read_csv_rows(prep_dir / "disease_test.csv")
            y = np.array([float(r["phenotype"]) for r in test_rows], dtype=float)
            best = max(rows, key=lambda r: float(r["metrics"].get("auc", -1.0)))
            pred_path = resolve_prediction_path(str(best["prediction_path"]))
            if pred_path is not None:
                pred_rows = read_csv_rows(pred_path)
                p = np.array([float(r["mean"]) for r in pred_rows], dtype=float)
                order = np.argsort(p)
                axes[1, 1].plot(np.linspace(0, 1, len(order)), p[order], label="predicted")
                axes[1, 1].plot(np.linspace(0, 1, len(order)), y[order], label="observed", alpha=0.8)
                axes[1, 1].set_title(f"Best model probability profile: {best['method']}")
            else:
                axes[1, 1].text(0.5, 0.5, "prediction artifact missing", ha="center", va="center")
                axes[1, 1].set_title("Best-model profile unavailable")
        else:
            test_rows = read_csv_rows(prep_dir / "survival_test.csv")
            times = np.array([float(r["time"]) for r in test_rows], dtype=float)
            events = np.array([float(r["event"]) for r in test_rows], dtype=float)
            best = max(rows, key=lambda r: float(r["metrics"].get("c_index", -1.0)))
            pred_path = resolve_prediction_path(str(best["prediction_path"]))
            if pred_path is not None:
                risk_rows = read_csv_rows(pred_path)
                risk_key = "risk" if "risk" in risk_rows[0] else ("failure_prob" if "failure_prob" in risk_rows[0] else "eta")
                risk = np.array([float(r[risk_key]) for r in risk_rows], dtype=float)
                q = np.quantile(risk, [0.0, 0.33, 0.66, 1.0])
                groups = np.digitize(risk, q[1:-1], right=True)
                try:
                    from lifelines import KaplanMeierFitter
                except ModuleNotFoundError as exc:
                    raise RuntimeError("lifelines is required for survival plotting") from exc
                kmf = KaplanMeierFitter()
                for grp, color in zip([0, 1, 2], ["#2ca02c", "#ff7f0e", "#d62728"]):
                    mask = groups == grp
                    if not np.any(mask):
                        continue
                    label = f"Q{grp + 1}"
                    kmf.fit(times[mask], event_observed=events[mask], label=label)
                    sf = kmf.survival_function_
                    axes[1, 1].step(sf.index, sf.iloc[:, 0], where="post", color=color, label=label)
                axes[1, 1].set_title(f"Risk-stratified KM: {best['method']}")
                axes[1, 1].set_xlabel("Years")
                axes[1, 1].set_ylabel("Survival")
            else:
                axes[1, 1].text(0.5, 0.5, "prediction artifact missing", ha="center", va="center")
                axes[1, 1].set_title("Risk KM unavailable")
        axes[1, 1].legend(fontsize=8)

        out_path = out_dir / f"{dataset}_summary.png"
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path)
    return paths


def do_aggregate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    result_files = sorted(args.results_dir.resolve().glob("*.json"))
    if not result_files:
        raise RuntimeError(f"no result json files found in {args.results_dir}")
    results = read_json_files(result_files)
    dump_json(out_dir / "config_snapshot.json", cfg)
    if (args.prep_dir.resolve() / "prep_metadata.json").exists():
        shutil.copy2(args.prep_dir.resolve() / "prep_metadata.json", out_dir / "prep_metadata.json")
    combined = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "results": results,
    }
    dump_json(out_dir / "combined_results.json", combined)
    metric_table = make_metric_table(results)
    fieldnames = sorted({k for row in metric_table for k in row.keys()})
    write_csv_rows(out_dir / "all_metrics.csv", metric_table, fieldnames)
    write_csv_rows(out_dir / "per_run_results.csv", results, sorted({k for row in results for k in row.keys()}))
    for dataset in sorted({str(r.get("dataset")) for r in results}):
        sub = [r for r in metric_table if str(r.get("dataset")) == dataset]
        write_csv_rows(out_dir / f"{dataset}_metrics.csv", sub, fieldnames)
    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = plot_aggregate(results, args.prep_dir.resolve(), plot_dir)
    zip_path = out_dir / "biobank_scale_bundle.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        already_added: set[str] = set()
        core_paths = [
            out_dir / "combined_results.json",
            out_dir / "all_metrics.csv",
            out_dir / "per_run_results.csv",
            out_dir / "config_snapshot.json",
        ]
        if (out_dir / "prep_metadata.json").exists():
            core_paths.append(out_dir / "prep_metadata.json")
        for path in core_paths:
            zf.write(path, arcname=path.name)
            already_added.add(path.name)
        for path in sorted(out_dir.glob("*_metrics.csv")):
            if path.name in already_added:
                continue
            zf.write(path, arcname=path.name)
        for path in result_files:
            zf.write(path, arcname=f"results/{path.name}")
        for path in plot_paths:
            zf.write(path, arcname=f"plots/{path.name}")
    print(f"Wrote {zip_path}")
    return 0


def do_matrix(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    payload = {"include": [spec.__dict__ for spec in build_method_specs(cfg)]}
    print(json.dumps(payload))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Biobank-scale synthetic benchmark runner")
    sub = parser.add_subparsers(dest="command", required=True)

    prep = sub.add_parser("prepare")
    prep.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    prep.add_argument("--out-dir", type=Path, required=True)
    prep.add_argument("--smoke", action="store_true")
    prep.add_argument("--target-n", type=int, default=None)
    prep.add_argument("--smoke-target-n", type=int, default=None)
    prep.set_defaults(func=do_prepare)

    matrix = sub.add_parser("matrix")
    matrix.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    matrix.set_defaults(func=do_matrix)

    run = sub.add_parser("run-method")
    run.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    run.add_argument("--prep-dir", type=Path, required=True)
    run.add_argument("--method", required=True)
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--out-json", type=Path, required=True)
    run.set_defaults(func=do_run_method)

    agg = sub.add_parser("aggregate")
    agg.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    agg.add_argument("--prep-dir", type=Path, required=True)
    agg.add_argument("--results-dir", type=Path, required=True)
    agg.add_argument("--out-dir", type=Path, required=True)
    agg.set_defaults(func=do_aggregate)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
