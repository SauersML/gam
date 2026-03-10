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
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "biobank_scale.yml"
HEARTBEAT_INTERVAL_SEC = 15.0
HEARTBEAT_INITIAL_WINDOW_SEC = 2.0
HEARTBEAT_INITIAL_INTERVAL_SEC = 0.25
MAX_CAPTURE_CHARS = 200000
MAX_SURVIVAL_GRID_POINTS = 256


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


def load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{path} must contain JSON-compatible YAML so this runner can parse it without external dependencies: {exc}"
        ) from exc


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
        commands = [
            ["ps", "-p", str(pid), "-o", "pid=,%cpu=,%mem=,rss=,vsz=,etimes=,stat=,comm="],
            ["ps", "-p", str(pid), "-o", "pid=", "-o", "%cpu=", "-o", "%mem=", "-o", "rss=", "-o", "vsz=", "-o", "etime=", "-o", "stat=", "-o", "comm="],
        ]
        for cmd in commands:
            proc = subprocess.run(
                cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            line = proc.stdout.strip()
            if not line:
                continue
            parts = line.split(None, 7)
            if len(parts) < 8:
                continue
            rss_raw = parts[3]
            vsz_raw = parts[4]
            rss_kib = int(rss_raw) if rss_raw.isdigit() else None
            vsz_kib = int(vsz_raw) if vsz_raw.isdigit() else None
            return {
                "pid": parts[0],
                "cpu_pct": parts[1],
                "mem_pct": parts[2],
                "rss_kib": rss_kib,
                "vsz_kib": vsz_kib,
                "etimes": parts[5],
                "stat": parts[6],
                "comm": parts[7],
            }
        return {}
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
        {"subpop": "FIN_Finland", "continent": "Europe", "superpop": "EUR", "lat": 60.17, "lon": 24.94},
        {"subpop": "YRI_Nigeria", "continent": "Africa", "superpop": "AFR", "lat": 7.38, "lon": 3.95},
        {"subpop": "LWK_Kenya", "continent": "Africa", "superpop": "AFR", "lat": 0.52, "lon": 35.27},
        {"subpop": "MSL_SierraLeone", "continent": "Africa", "superpop": "AFR", "lat": 8.48, "lon": -13.23},
        {"subpop": "CHB_Beijing", "continent": "Asia", "superpop": "EAS", "lat": 39.90, "lon": 116.41},
        {"subpop": "JPT_Tokyo", "continent": "Asia", "superpop": "EAS", "lat": 35.68, "lon": 139.69},
        {"subpop": "KHV_Vietnam", "continent": "Asia", "superpop": "EAS", "lat": 10.82, "lon": 106.63},
        {"subpop": "GIH_Gujarat", "continent": "Asia", "superpop": "SAS", "lat": 23.02, "lon": 72.57},
        {"subpop": "BEB_Bangladesh", "continent": "Asia", "superpop": "SAS", "lat": 23.81, "lon": 90.41},
        {"subpop": "MXL_Mexico", "continent": "NorthAmerica", "superpop": "AMR", "lat": 19.43, "lon": -99.13},
        {"subpop": "PUR_PuertoRico", "continent": "NorthAmerica", "superpop": "AMR", "lat": 18.47, "lon": -66.11},
        {"subpop": "PEL_Peru", "continent": "SouthAmerica", "superpop": "AMR", "lat": -12.05, "lon": -77.05},
        {"subpop": "CLM_Colombia", "continent": "SouthAmerica", "superpop": "AMR", "lat": 4.71, "lon": -74.07},
        {"subpop": "PNG_Highlands", "continent": "Oceania", "superpop": "OCE", "lat": -6.08, "lon": 145.39},
        {"subpop": "MEL_Bougainville", "continent": "Oceania", "superpop": "OCE", "lat": -6.23, "lon": 155.57},
    ]


def build_seed_panel(cfg: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    templates = subpop_templates()
    n_per_subpop = int(cfg["raw_subpop_n"])
    observed_fraction = float(cfg["observed_latlon_fraction"])
    n_pcs = 16
    pc_cols = [f"pc{i}" for i in range(1, n_pcs + 1)]
    continent_order = ["Europe", "Africa", "Asia", "NorthAmerica", "SouthAmerica", "Oceania"]
    continent_center = {
        cont: rng.normal(loc=0.0, scale=np.linspace(2.8, 0.6, n_pcs), size=n_pcs)
        for cont in continent_order
    }
    superpop_shift = {
        code: rng.normal(loc=0.0, scale=np.linspace(0.9, 0.2, n_pcs), size=n_pcs)
        for code in sorted({t["superpop"] for t in templates})
    }
    rows: list[dict[str, Any]] = []
    sample_id = 0
    for template in templates:
        sub_shift = rng.normal(loc=0.0, scale=np.linspace(0.45, 0.08, n_pcs), size=n_pcs)
        lat0 = float(template["lat"])
        lon0 = float(template["lon"])
        lat_rad = math.radians(lat0)
        lon_rad = math.radians(lon0)
        geo_basis = np.array(
            [
                math.sin(lat_rad),
                math.cos(lat_rad),
                math.sin(lon_rad),
                math.cos(lon_rad),
                math.sin(lat_rad) * math.cos(lon_rad),
                math.cos(lat_rad) * math.sin(lon_rad),
                lat0 / 90.0,
                lon0 / 180.0,
                (lat0 / 90.0) ** 2,
                (lon0 / 180.0) ** 2,
                math.sin(2.0 * lat_rad),
                math.cos(2.0 * lon_rad),
                math.sin(lat_rad + lon_rad),
                math.cos(lat_rad - lon_rad),
                math.sin(0.5 * lat_rad * lon_rad),
                math.cos(0.5 * lat_rad * lon_rad),
            ],
            dtype=float,
        )
        for _ in range(n_per_subpop):
            lat = float(np.clip(lat0 + rng.normal(0.0, 1.4), -55.0, 72.0))
            lon = float(((lon0 + rng.normal(0.0, 1.9) + 180.0) % 360.0) - 180.0)
            indiv_noise = rng.normal(loc=0.0, scale=np.linspace(0.35, 0.08, n_pcs), size=n_pcs)
            pcs = (
                1.35 * continent_center[template["continent"]]
                + 0.95 * superpop_shift[template["superpop"]]
                + 0.75 * sub_shift
                + 0.55 * geo_basis
                + indiv_noise
            )
            has_observed = bool(rng.random() < observed_fraction)
            row = {
                "sample_id": f"seed_{sample_id:07d}",
                "source": "seed",
                "generation": 0,
                "parent_a": "",
                "parent_b": "",
                "subpop": template["subpop"],
                "superpop": template["superpop"],
                "continent": template["continent"],
                "lat_true": lat,
                "lon_true": lon,
                "lat_obs": lat if has_observed else "",
                "lon_obs": lon if has_observed else "",
                "lat_missing": 0 if has_observed else 1,
                "lon_missing": 0 if has_observed else 1,
            }
            for j, col in enumerate(pc_cols):
                row[col] = float(pcs[j])
            rows.append(row)
            sample_id += 1
    return {"rows": rows, "pc_cols": pc_cols}


def fit_latlon_imputer(pc_known: np.ndarray, latlon_known: np.ndarray) -> dict[str, Any]:
    if pc_known.shape[0] < 8:
        raise RuntimeError("not enough rows with observed lat/lon to fit imputer")
    try:
        import xgboost as xgb
    except ModuleNotFoundError:
        xgb = None
    if xgb is not None:
        models = []
        for dim in range(2):
            model = xgb.XGBRegressor(
                n_estimators=260,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="reg:squarederror",
                reg_lambda=1.0,
                random_state=20260310 + dim,
                n_jobs=1,
            )
            model.fit(pc_known, latlon_known[:, dim])
            models.append(model)
        return {"kind": "xgboost", "models": models}
    return {"kind": "knn_inverse_distance", "pc_known": pc_known.copy(), "latlon_known": latlon_known.copy()}


def predict_latlon_imputer(model: dict[str, Any], pc_all: np.ndarray) -> np.ndarray:
    if model["kind"] == "xgboost":
        lat = np.asarray(model["models"][0].predict(pc_all), dtype=float)
        lon = np.asarray(model["models"][1].predict(pc_all), dtype=float)
        return np.column_stack([lat, lon])
    known = np.asarray(model["pc_known"], dtype=float)
    latlon = np.asarray(model["latlon_known"], dtype=float)
    out = np.empty((pc_all.shape[0], 2), dtype=float)
    for i in range(pc_all.shape[0]):
        d = np.sqrt(np.maximum(np.sum((known - pc_all[i]) ** 2, axis=1), 0.0))
        nn = np.argsort(d)[:8]
        w = 1.0 / np.maximum(d[nn], 1e-6)
        w = w / np.sum(w)
        out[i] = np.sum(latlon[nn] * w.reshape(-1, 1), axis=0)
    return out


def impute_latlon(seed_panel: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    rows = seed_panel["rows"]
    pc_cols = seed_panel["pc_cols"]
    pcs = np.array([[float(r[c]) for c in pc_cols] for r in rows], dtype=float)
    known_mask = np.array([r["lat_missing"] == 0 for r in rows], dtype=bool)
    latlon_known = np.array(
        [[float(rows[i]["lat_obs"]), float(rows[i]["lon_obs"])] for i in np.where(known_mask)[0]],
        dtype=float,
    )
    model = fit_latlon_imputer(pcs[known_mask], latlon_known)
    pred = predict_latlon_imputer(model, pcs)
    for i, row in enumerate(rows):
        lat_pred = float(np.clip(pred[i, 0], -55.0, 72.0))
        lon_pred = float(((pred[i, 1] + 180.0) % 360.0) - 180.0)
        row["lat_imputed"] = lat_pred
        row["lon_imputed"] = lon_pred
        if row["lat_missing"] == 0:
            row["lat_final"] = float(row["lat_obs"])
            row["lon_final"] = float(row["lon_obs"])
        else:
            row["lat_final"] = lat_pred
            row["lon_final"] = lon_pred
        row["lat_final"] = float(np.clip(row["lat_final"] + rng.normal(0.0, 0.18), -55.0, 72.0))
        row["lon_final"] = float(((row["lon_final"] + rng.normal(0.0, 0.22) + 180.0) % 360.0) - 180.0)
        row["latlon_source"] = "observed" if row["lat_missing"] == 0 else "imputed"
    return {"rows": rows, "pc_cols": pc_cols, "imputer_kind": model["kind"]}


def eigen_weights(rows: list[dict[str, Any]], pc_cols: list[str]) -> np.ndarray:
    x = np.array([[float(r[c]) for c in pc_cols] for r in rows], dtype=float)
    sd = np.std(x, axis=0)
    sd[(~np.isfinite(sd)) | (sd < 1e-6)] = 1.0
    return 1.0 / sd


def weighted_distance(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    diff = (a - b) * weight
    return float(np.sqrt(np.maximum(np.sum(diff * diff), 0.0)))


def nearest_subpop_map(rows: list[dict[str, Any]], pc_cols: list[str], weight: np.ndarray) -> tuple[dict[str, str], dict[str, np.ndarray]]:
    by_subpop: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        by_subpop[str(row["subpop"])].append(np.array([float(row[c]) for c in pc_cols], dtype=float))
    centroids = {k: np.mean(v, axis=0) for k, v in by_subpop.items()}
    nearest: dict[str, str] = {}
    keys = sorted(centroids)
    for key in keys:
        best_name = ""
        best_dist = float("inf")
        for other in keys:
            if other == key:
                continue
            dist = weighted_distance(centroids[key], centroids[other], weight)
            if dist < best_dist:
                best_dist = dist
                best_name = other
        nearest[key] = best_name
    return nearest, centroids


def upsample_panel(seed_rows: list[dict[str, Any]], pc_cols: list[str], cfg: dict[str, Any], rng: np.random.Generator) -> list[dict[str, Any]]:
    target_n = int(cfg["target_n"])
    if len(seed_rows) >= target_n:
        return seed_rows[:target_n]
    rows = [dict(r) for r in seed_rows]
    weight = eigen_weights(rows, pc_cols)
    nearest_map, _ = nearest_subpop_map(rows, pc_cols, weight)
    subpop_index: dict[str, list[int]] = defaultdict(list)
    subpop_probs: dict[str, float] = {}
    for idx, row in enumerate(rows):
        subpop_index[str(row["subpop"])].append(idx)
    total_seed = float(len(rows))
    for subpop, idxs in subpop_index.items():
        subpop_probs[subpop] = len(idxs) / total_seed
    ordered_subpops = sorted(subpop_index)
    probs = np.array([subpop_probs[s] for s in ordered_subpops], dtype=float)
    probs = probs / np.sum(probs)
    next_id = len(rows)
    batch_size = int(cfg.get("upsample_batch_size", 5000))
    while len(rows) < target_n:
        take = min(batch_size, target_n - len(rows))
        target_subpops = rng.choice(ordered_subpops, size=take, replace=True, p=probs)
        new_rows: list[dict[str, Any]] = []
        for target_subpop in target_subpops:
            first_pool = subpop_index[target_subpop]
            parent_a_idx = int(rng.choice(first_pool))
            cross = bool(rng.random() < float(cfg["cross_subpop_pairing_rate"]))
            partner_subpop = nearest_map[target_subpop] if cross else target_subpop
            partner_pool = subpop_index[partner_subpop]
            parent_b_idx = int(rng.choice(partner_pool))
            a = rows[parent_a_idx]
            b = rows[parent_b_idx]
            pc_a = np.array([float(a[c]) for c in pc_cols], dtype=float)
            pc_b = np.array([float(b[c]) for c in pc_cols], dtype=float)
            alpha = float(rng.beta(2.2, 2.2))
            midpoint = pc_a + alpha * (pc_b - pc_a)
            pair_dist = weighted_distance(pc_a, pc_b, weight)
            jitter_sd = 0.025 + 0.06 * np.tanh(pair_dist / 4.0)
            child_pc = midpoint + rng.normal(0.0, jitter_sd / np.maximum(weight, 1e-6), size=len(pc_cols))
            lat = 0.5 * (float(a["lat_final"]) + float(b["lat_final"])) + rng.normal(0.0, 0.22)
            lon = 0.5 * (float(a["lon_final"]) + float(b["lon_final"])) + rng.normal(0.0, 0.28)
            lat = float(np.clip(lat, -55.0, 72.0))
            lon = float(((lon + 180.0) % 360.0) - 180.0)
            lat_true = 0.5 * (float(a["lat_true"]) + float(b["lat_true"])) + rng.normal(0.0, 0.14)
            lon_true = 0.5 * (float(a["lon_true"]) + float(b["lon_true"])) + rng.normal(0.0, 0.18)
            lat_true = float(np.clip(lat_true, -55.0, 72.0))
            lon_true = float(((lon_true + 180.0) % 360.0) - 180.0)
            child = {
                "sample_id": f"sim_{next_id:07d}",
                "source": "simulated",
                "generation": 1 + max(int(a["generation"]), int(b["generation"])),
                "parent_a": str(a["sample_id"]),
                "parent_b": str(b["sample_id"]),
                "subpop": target_subpop,
                "superpop": str(a["superpop"]),
                "continent": str(a["continent"]),
                "lat_true": lat_true,
                "lon_true": lon_true,
                "lat_obs": "",
                "lon_obs": "",
                "lat_missing": 1,
                "lon_missing": 1,
                "lat_imputed": lat,
                "lon_imputed": lon,
                "lat_final": lat,
                "lon_final": lon,
                "latlon_source": "simulated",
            }
            for j, col in enumerate(pc_cols):
                child[col] = float(child_pc[j])
            new_rows.append(child)
            next_id += 1
        start_idx = len(rows)
        rows.extend(new_rows)
        for offset, row in enumerate(new_rows):
            subpop_index[str(row["subpop"])].append(start_idx + offset)
    return rows


def representative_geo_targets() -> list[dict[str, Any]]:
    return [
        {"continent": "Europe", "lat": 52.0, "lon": 10.0, "target": 0.14},
        {"continent": "Europe", "lat": 41.0, "lon": 13.0, "target": 0.11},
        {"continent": "Africa", "lat": 7.0, "lon": 5.0, "target": 0.05},
        {"continent": "Africa", "lat": 0.0, "lon": 36.0, "target": 0.04},
        {"continent": "Asia", "lat": 40.0, "lon": 116.0, "target": 0.10},
        {"continent": "Asia", "lat": 23.0, "lon": 73.0, "target": 0.08},
        {"continent": "NorthAmerica", "lat": 41.0, "lon": -112.0, "target": 0.13},
        {"continent": "NorthAmerica", "lat": 19.0, "lon": -99.0, "target": 0.09},
        {"continent": "SouthAmerica", "lat": -12.0, "lon": -77.0, "target": 0.08},
        {"continent": "SouthAmerica", "lat": 5.0, "lon": -74.0, "target": 0.07},
        {"continent": "Oceania", "lat": -6.0, "lon": 145.0, "target": 0.11},
        {"continent": "Oceania", "lat": -6.0, "lon": 155.0, "target": 0.12},
    ]


def prevalence_surface_components(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat_abs = np.abs(lat)
    temp_band = np.exp(-((lat_abs - 42.0) / 18.0) ** 2)
    equator_penalty = np.exp(-((lat_abs - 4.0) / 9.5) ** 2)
    polar_penalty = np.exp(-((lat_abs - 67.0) / 7.5) ** 2)
    lon_rad = np.radians(lon)
    lat_rad = np.radians(lat)
    waves = (
        0.55 * np.sin(1.35 * lon_rad + 0.35 * lat_rad)
        + 0.42 * np.cos(2.1 * lat_rad - 0.55 * lon_rad)
        + 0.33 * np.sin(0.75 * lat_rad * np.maximum(np.cos(lon_rad), -0.95))
        + 0.18 * np.cos(np.radians(0.9 * lat * lon / 30.0))
    )
    hotspot = (
        0.22 * np.exp(-(((lat - 50.0) / 8.0) ** 2 + ((lon - 10.0) / 12.0) ** 2))
        + 0.18 * np.exp(-(((lat - 40.0) / 9.0) ** 2 + ((lon + 110.0) / 14.0) ** 2))
        + 0.20 * np.exp(-(((lat + 8.0) / 7.0) ** 2 + ((lon - 148.0) / 10.0) ** 2))
    )
    return 0.95 * temp_band - 0.78 * equator_penalty - 0.20 * polar_penalty + waves + hotspot


def prevalence_surface_score(
    lat: np.ndarray,
    lon: np.ndarray,
    continents: list[str],
    continent_offsets: dict[str, float],
    amplitude: float,
) -> np.ndarray:
    base = prevalence_surface_components(lat, lon)
    offsets = np.array([float(continent_offsets[c]) for c in continents], dtype=float)
    return offsets + float(amplitude) * base


def calibrate_prevalence_surface() -> tuple[float, dict[str, float], list[dict[str, Any]]]:
    reps = representative_geo_targets()
    best = None
    targets = np.array([float(rep["target"]) for rep in reps], dtype=float)
    logits = np.log(np.clip(targets, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - targets, 1e-6, 1.0))
    continents = [str(rep["continent"]) for rep in reps]
    lat = np.array([float(rep["lat"]) for rep in reps], dtype=float)
    lon = np.array([float(rep["lon"]) for rep in reps], dtype=float)
    base = prevalence_surface_components(lat, lon)
    for amplitude in np.linspace(0.25, 1.75, 151):
        offsets: dict[str, float] = {}
        for continent in sorted(set(continents)):
            idx = [i for i, c in enumerate(continents) if c == continent]
            offsets[continent] = float(np.mean(logits[idx] - amplitude * base[idx]))
        pred_score = prevalence_surface_score(lat, lon, continents, offsets, amplitude)
        preds = logistic(pred_score)
        err = float(np.mean((preds - targets) ** 2))
        if best is None or err < best[0]:
            best = (err, amplitude, offsets)
    assert best is not None
    amplitude = float(best[1])
    offsets = dict(best[2])
    spot = []
    for rep in reps:
        pred = float(
            logistic(
                prevalence_surface_score(
                    np.array([rep["lat"]], dtype=float),
                    np.array([rep["lon"]], dtype=float),
                    [str(rep["continent"])],
                    offsets,
                    amplitude,
                )
            )[0]
        )
        spot.append({**rep, "predicted": pred})
    return amplitude, offsets, spot


def simulate_traits(rows: list[dict[str, Any]], pc_cols: list[str], cfg: dict[str, Any], rng: np.random.Generator) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lat = np.array([float(r["lat_final"]) for r in rows], dtype=float)
    lon = np.array([float(r["lon_final"]) for r in rows], dtype=float)
    continents = [str(r["continent"]) for r in rows]
    amplitude, continent_offsets, spot = calibrate_prevalence_surface()
    geo_prev = logistic(prevalence_surface_score(lat, lon, continents, continent_offsets, amplitude))
    age = np.clip(
        rng.normal(
            loc=np.array([50.0 if r["continent"] == "Europe" else 46.0 for r in rows], dtype=float),
            scale=9.0,
            size=len(rows),
        ),
        37.0,
        80.0,
    )
    sex = (rng.random(len(rows)) < 0.49).astype(float)
    genetic_liability = rng.normal(0.0, 1.0, size=len(rows))
    disease_logit = (
        np.log(np.clip(geo_prev, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - geo_prev, 1e-6, 1.0))
        + 0.48 * genetic_liability
        + 0.18 * standardize(age)
        + 0.08 * sex
    )
    disease_prob = logistic(disease_logit)
    phenotype = (rng.random(len(rows)) < disease_prob).astype(float)
    weight = eigen_weights(rows, pc_cols)
    _, centroids = nearest_subpop_map(rows, pc_cols, weight)
    ref_subpop = str(cfg["reference_subpopulation"])
    if ref_subpop not in centroids:
        raise RuntimeError(f"reference subpopulation '{ref_subpop}' not found in synthetic panel")
    ref_centroid = centroids[ref_subpop]
    portability_distance = np.empty(len(rows), dtype=float)
    for i, row in enumerate(rows):
        pc = np.array([float(row[c]) for c in pc_cols], dtype=float)
        portability_distance[i] = weighted_distance(pc, ref_centroid, weight)
    portability_noise_sd = 0.22 + 1.00 * (1.0 - np.exp(-(portability_distance ** 2) / 10.0))
    pgs_raw = 0.85 * genetic_liability + rng.normal(0.0, portability_noise_sd, size=len(rows))
    pgs = standardize(pgs_raw)
    entry_year = rng.uniform(2006.0, 2013.5, size=len(rows))
    age_entry = age
    shape = 1.65
    lp_surv = (
        0.38 * standardize(age_entry)
        + 0.52 * genetic_liability
        + 0.24 * phenotype
        + 0.12 * sex
        + 0.35 * standardize(np.log(np.clip(geo_prev, 1e-6, 1.0)) - np.log(np.clip(1.0 - geo_prev, 1e-6, 1.0)))
    )
    base_scale = 11.5
    u = np.clip(rng.random(len(rows)), 1e-9, 1.0 - 1e-9)
    event_time = base_scale * ((-np.log(u)) / np.exp(lp_surv)) ** (1.0 / shape)
    dropout_rate = np.exp(-2.35 + 0.22 * standardize(age_entry) + 0.08 * phenotype)
    dropout_time = rng.exponential(scale=1.0 / np.clip(dropout_rate, 1e-4, None), size=len(rows))
    admin_time = np.clip(2026.0 - entry_year, 1.0, None)
    obs_time = np.minimum(event_time, np.minimum(dropout_time, admin_time))
    event = (event_time <= dropout_time) & (event_time <= admin_time)
    for i, row in enumerate(rows):
        row["age_entry"] = float(age_entry[i])
        row["sex"] = float(sex[i])
        row["geo_prevalence"] = float(geo_prev[i])
        row["genetic_liability"] = float(genetic_liability[i])
        row["phenotype_prob"] = float(disease_prob[i])
        row["phenotype"] = float(phenotype[i])
        row["portability_distance"] = float(portability_distance[i])
        row["portability_noise_sd"] = float(portability_noise_sd[i])
        row["pgs_raw"] = float(pgs_raw[i])
        row["pgs"] = float(pgs[i])
        row["entry_year"] = float(entry_year[i])
        row["time"] = float(max(obs_time[i], 1e-6))
        row["event"] = float(event[i])
        row["dropout_time"] = float(dropout_time[i])
        row["admin_time"] = float(admin_time[i])
        row["event_time_true"] = float(event_time[i])
    return rows, {
        "reference_subpopulation": ref_subpop,
        "reference_centroid_pc": {pc_cols[i]: float(ref_centroid[i]) for i in range(len(pc_cols))},
        "prevalence_surface_amplitude": amplitude,
        "prevalence_surface_continent_offsets": continent_offsets,
        "spot_checks": spot,
        "pgs_portability_distance_quantiles": {
            "q05": float(np.quantile(portability_distance, 0.05)),
            "q50": float(np.quantile(portability_distance, 0.50)),
            "q95": float(np.quantile(portability_distance, 0.95)),
        },
        "pgs_noise_sd_quantiles": {
            "q05": float(np.quantile(portability_noise_sd, 0.05)),
            "q50": float(np.quantile(portability_noise_sd, 0.50)),
            "q95": float(np.quantile(portability_noise_sd, 0.95)),
        },
        "pgs_correlation_with_latitude": float(np.corrcoef(pgs, lat)[0, 1]),
        "pgs_correlation_with_longitude": float(np.corrcoef(pgs, lon)[0, 1]),
    }


def split_rows(rows: list[dict[str, Any]], cfg: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = np.random.default_rng(int(cfg["split_seed"]))
    y = np.array([float(r["phenotype"]) for r in rows], dtype=int)
    test_fraction = 1.0 - float(cfg["train_fraction"])
    test_indices: list[int] = []
    for cls in [0, 1]:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = int(round(len(idx) * test_fraction))
        n_test = max(1, min(len(idx) - 1, n_test))
        test_indices.extend(idx[:n_test].tolist())
    test_set = set(test_indices)
    train_rows = [rows[i] for i in range(len(rows)) if i not in test_set]
    test_rows = [rows[i] for i in range(len(rows)) if i in test_set]
    return train_rows, test_rows


def prepare_dataset_views(rows: list[dict[str, Any]], train_rows: list[dict[str, Any]], test_rows: list[dict[str, Any]], prep_dir: Path) -> dict[str, Any]:
    all_fieldnames = [
        "sample_id", "source", "generation", "parent_a", "parent_b", "subpop", "superpop", "continent",
        "lat_true", "lon_true", "lat_obs", "lon_obs", "lat_missing", "lon_missing", "lat_imputed", "lon_imputed",
        "lat_final", "lon_final", "latlon_source", "age_entry", "sex", "geo_prevalence", "genetic_liability",
        "phenotype_prob", "phenotype", "portability_distance", "portability_noise_sd", "pgs_raw", "pgs",
        "entry_year", "time", "event", "dropout_time", "admin_time", "event_time_true",
    ] + [f"pc{i}" for i in range(1, 17)]
    write_csv_rows(prep_dir / "all_cohort.csv", rows, all_fieldnames)

    feature_cols = ["lat_final", "lon_final", "age_entry", "sex", "pgs"] + [f"pc{i}" for i in range(1, 17)]
    scaled_stats: dict[str, dict[str, float]] = {}
    train_copy = [dict(r) for r in train_rows]
    test_copy = [dict(r) for r in test_rows]
    for col in feature_cols:
        train_arr = np.array([float(r[col]) for r in train_copy], dtype=float)
        test_arr = np.array([float(r[col]) for r in test_copy], dtype=float)
        tr_scaled, te_scaled, mu, sd = zscore_train_test(train_arr, test_arr)
        scaled_stats[col] = {"mean": mu, "sd": sd}
        for i, row in enumerate(train_copy):
            row[f"{col}_std"] = float(tr_scaled[i])
        for i, row in enumerate(test_copy):
            row[f"{col}_std"] = float(te_scaled[i])
    disease_fields = [
        "subpop", "superpop", "continent", "source", "lat_final", "lon_final", "age_entry", "sex",
        "pgs", "phenotype", "geo_prevalence", "portability_distance",
    ] + [f"pc{i}" for i in range(1, 17)] + [f"{c}_std" for c in feature_cols]
    survival_fields = [
        "subpop", "superpop", "continent", "source", "lat_final", "lon_final", "age_entry", "sex",
        "pgs", "time0", "time", "event", "entry_year", "geo_prevalence", "portability_distance",
    ] + [f"pc{i}" for i in range(1, 17)] + [f"{c}_std" for c in feature_cols]
    for row in train_copy:
        row["time0"] = 0.0
    for row in test_copy:
        row["time0"] = 0.0
    write_csv_rows(prep_dir / "disease_train.csv", train_copy, disease_fields)
    write_csv_rows(prep_dir / "disease_test.csv", test_copy, disease_fields)
    write_csv_rows(prep_dir / "survival_train.csv", train_copy, survival_fields)
    write_csv_rows(prep_dir / "survival_test.csv", test_copy, survival_fields)
    return {
        "feature_cols": feature_cols,
        "scaled_stats": scaled_stats,
        "train_n": len(train_copy),
        "test_n": len(test_copy),
        "all_n": len(rows),
    }


def build_method_specs(cfg: dict[str, Any]) -> list[MethodSpec]:
    specs = []
    for raw in cfg["methods"]:
        specs.append(
            MethodSpec(
                name=str(raw["name"]),
                dataset=str(raw["dataset"]),
                backend=str(raw["backend"]),
                family=str(raw["family"]),
                spatial_basis=str(raw["spatial_basis"]),
                centers=int(raw["centers"]) if raw.get("centers") is not None else None,
                smooth_kind=str(raw.get("smooth_kind", "joint")),
                include_sigma=bool(raw.get("include_sigma", False)),
            )
        )
    return specs


def make_prepared_payload(cfg: dict[str, Any], target_n: int) -> dict[str, Any]:
    run_cfg = dict(cfg)
    run_cfg["target_n"] = int(target_n)
    run_cfg.setdefault("cross_subpop_pairing_rate", 0.28)
    run_cfg.setdefault("upsample_batch_size", 5000)
    rng = np.random.default_rng(int(run_cfg["seed"]))
    seed_panel = build_seed_panel(run_cfg, rng)
    seed_panel = impute_latlon(seed_panel, rng)
    seed_n = len(seed_panel["rows"])
    weight = eigen_weights(seed_panel["rows"], seed_panel["pc_cols"])
    nearest_map, centroids = nearest_subpop_map(seed_panel["rows"], seed_panel["pc_cols"], weight)
    rows = upsample_panel(seed_panel["rows"], seed_panel["pc_cols"], run_cfg, rng)
    rows, trait_meta = simulate_traits(rows, seed_panel["pc_cols"], run_cfg, rng)
    train_rows, test_rows = split_rows(rows, run_cfg)
    return {
        "rows": rows,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "pc_cols": seed_panel["pc_cols"],
        "trait_meta": trait_meta,
        "imputer_kind": seed_panel["imputer_kind"],
        "seed_n": seed_n,
        "simulated_n": int(max(len(rows) - seed_n, 0)),
        "upsample_factor": float(len(rows) / max(seed_n, 1)),
        "nearest_subpopulation": nearest_map,
        "subpopulation_centroids": {
            subpop: {seed_panel["pc_cols"][i]: float(vec[i]) for i in range(len(seed_panel["pc_cols"]))}
            for subpop, vec in centroids.items()
        },
    }


def do_prepare(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    target_n = int(cfg["smoke_target_n"] if args.smoke else cfg["target_n"])
    prep_dir = args.out_dir.resolve()
    prep_dir.mkdir(parents=True, exist_ok=True)
    payload = make_prepared_payload(cfg, target_n)
    prep_meta = prepare_dataset_views(payload["rows"], payload["train_rows"], payload["test_rows"], prep_dir)
    cohort = payload["rows"]
    continent_summary: dict[str, dict[str, float]] = {}
    for continent in sorted({str(r["continent"]) for r in cohort}):
        subset = [r for r in cohort if str(r["continent"]) == continent]
        continent_summary[continent] = {
            "n": len(subset),
            "mean_geo_prevalence": float(np.mean([float(r["geo_prevalence"]) for r in subset])),
            "mean_phenotype": float(np.mean([float(r["phenotype"]) for r in subset])),
            "mean_age_entry": float(np.mean([float(r["age_entry"]) for r in subset])),
        }
    meta = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(args.config),
        "target_n": target_n,
        "mode": "smoke" if args.smoke else "full",
        "imputer_kind": payload["imputer_kind"],
        "trait_meta": payload["trait_meta"],
        "seed_n": payload["seed_n"],
        "simulated_n": payload["simulated_n"],
        "upsample_factor": payload["upsample_factor"],
        "nearest_subpopulation": payload["nearest_subpopulation"],
        "subpopulation_centroids": payload["subpopulation_centroids"],
        "prep": prep_meta,
        "continent_summary": continent_summary,
    }
    dump_json(prep_dir / "prep_metadata.json", meta)
    print(f"Wrote prepared cohort to {prep_dir}")
    return 0


def rust_joint_term(basis: str, cols: list[str], centers: int) -> str:
    joined = ", ".join(cols)
    if basis == "duchon":
        return f"duchon({joined}, centers={centers}, order=0, power=1)"
    raise RuntimeError(f"unsupported Rust biobank basis '{basis}'; use duchon")


def rust_formula_classification(spec: MethodSpec) -> tuple[str, str]:
    if spec.smooth_kind != "joint":
        raise RuntimeError(f"{spec.name} must use a joint Rust smooth")
    linear_terms = ["linear(age_entry_std)", "linear(sex)"]
    smooth_cols = ["lat_final_std", "lon_final_std", "pgs_std"]
    smooth_cols.extend(f"pc{i}_std" for i in range(1, 17))
    linear_terms.append(rust_joint_term(spec.spatial_basis, smooth_cols, int(spec.centers or 60)))
    mean_formula = "phenotype ~ " + " + ".join(linear_terms) + " + link(type=logit)"
    sigma_terms = ["linear(pgs_std)", "linear(age_entry_std)", "linear(lat_final_std)", "linear(lon_final_std)"]
    sigma_formula = "phenotype ~ " + " + ".join(sigma_terms)
    return mean_formula, sigma_formula


def rust_formula_survival(spec: MethodSpec) -> str:
    terms = [
        "linear(pgs_std)",
        "linear(sex)",
        "s(age_entry_std, type=ps, knots=8)",
        "s(lat_final_std, type=ps, knots=10)",
        "s(lon_final_std, type=ps, knots=10)",
    ]
    terms.extend(f"linear(pc{i}_std)" for i in range(1, 5))
    return " + ".join(terms)


def mgcv_formula_classification(spec: MethodSpec) -> str:
    terms = ["pgs_std", "age_entry_std", "sex"]
    if spec.smooth_kind == "joint":
        if spec.spatial_basis == "thinplate":
            terms.append(f"s(lat_final_std, lon_final_std, bs='tp', k=min({int(spec.centers or 60)}, nrow(train_df)-1))")
        elif spec.spatial_basis == "duchon":
            terms.append(f"s(lat_final_std, lon_final_std, bs='ds', m=c(1,0), k=min({int(spec.centers or 60)}, nrow(train_df)-1))")
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
    fit_cmd.extend(["--no-summary", "--out", str(model_path), str(train_csv), mean_formula])
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
        "model_spec": f"Rust joint {spec.spatial_basis} {'GAMLSS' if spec.include_sigma else 'GAM'} holdout",
    }


def run_rust_survival(spec: MethodSpec, train_csv: Path, test_csv: Path, out_dir: Path) -> dict[str, Any]:
    rust_bin = load_or_build_rust_binary()
    formula_rhs = rust_formula_survival(spec)
    model_path = out_dir / f"{spec.name}.model.json"
    pred_path = out_dir / f"{spec.name}.pred.csv"
    likelihood_mode = "probit-location-scale"
    fit_cmd = [
        str(rust_bin), "fit", "--no-summary",
        "--survival-likelihood", likelihood_mode,
        "--time-basis", "ispline",
        "--time-degree", "3",
        "--time-num-internal-knots", "8",
        "--time-smooth-lambda", "0.01",
        "--ridge-lambda", "1e-6",
        "--out", str(model_path),
        str(train_csv),
        f"Surv(time0, time, event) ~ {formula_rhs}",
    ]
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
    risk_cols = ["failure_prob", "risk_score", "eta"]
    risk_key = next((k for k in risk_cols if k in pred_rows[0]), None)
    if risk_key is None:
        raise RuntimeError(f"{spec.name} prediction output missing risk column")
    test_risk = np.array([float(r[risk_key]) for r in pred_rows], dtype=float)
    train_pred_path = out_dir / f"{spec.name}.trainpred.csv"
    pred_train_cmd = [str(rust_bin), "predict", str(model_path), str(train_csv), "--out", str(train_pred_path)]
    rc, out, err = run_cmd_stream(pred_train_cmd, cwd=ROOT)
    if rc != 0:
        raise RuntimeError(err.strip() or out.strip() or f"{spec.name} train predict failed")
    train_pred_rows = read_csv_rows(train_pred_path)
    train_risk = np.array([float(r[risk_key]) for r in train_pred_rows], dtype=float)
    train_rows = [{k: (float(v) if k in {"time", "event"} else v) for k, v in r.items()} for r in read_csv_rows(train_csv)]
    test_rows = [{k: (float(v) if k in {"time", "event"} else v) for k, v in r.items()} for r in read_csv_rows(test_csv)]
    metrics = survival_metrics(train_rows, test_rows, train_risk, test_risk)
    return {
        "fit_sec": fit_sec,
        "predict_sec": predict_sec,
        "metrics": metrics,
        "prediction_path": str(pred_path),
        "model_spec": f"Rust survival {likelihood_mode} holdout",
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
        if spec.backend == "rust_gam":
            result = run_rust_classification(spec, disease_train, disease_test, out_dir)
        elif spec.backend == "r_mgcv":
            result = run_r_mgcv_classification(spec, disease_train, disease_test, out_dir)
        else:
            raise RuntimeError(f"unsupported disease backend '{spec.backend}'")
    elif spec.dataset == "survival":
        if spec.backend == "rust_gamlss_survival":
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
