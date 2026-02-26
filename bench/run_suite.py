#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

# Hard-force single-thread execution across Python/R/Rust/native math libs.
_SERIAL_ENV_OVERRIDES = {
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
for _k, _v in _SERIAL_ENV_OVERRIDES.items():
    os.environ[_k] = _v

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from pygam import LinearGAM, LogisticGAM, l, s

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "bench"
DEFAULT_SCENARIOS = BENCH_DIR / "scenarios.json"
DATASET_DIR = BENCH_DIR / "datasets"
CV_SPLITS = 5
CV_SEED = 42
_RUST_BIN_PATH: Path | None = None


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


def run_cmd(cmd, cwd=None):
    env = os.environ.copy()
    env.update(_SERIAL_ENV_OVERRIDES)
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        bufsize=1,
    )

    out_buf = []
    err_buf = []

    def _pump(stream, sink, buf):
        try:
            for line in iter(stream.readline, ""):
                sink.write(line)
                sink.flush()
                buf.append(line)
        finally:
            stream.close()

    t_out = threading.Thread(target=_pump, args=(proc.stdout, sys.stdout, out_buf), daemon=True)
    t_err = threading.Thread(target=_pump, args=(proc.stderr, sys.stderr, err_buf), daemon=True)
    t_out.start()
    t_err.start()
    rc = proc.wait()
    t_out.join()
    t_err.join()
    return rc, "".join(out_buf), "".join(err_buf)


def make_folds(y: np.ndarray, n_splits: int = 5, seed: int = 42, stratified: bool = False):
    n = int(len(y))
    if n < n_splits:
        raise ValueError(f"Need at least {n_splits} rows for {n_splits}-fold CV; got {n}")
    rng = np.random.default_rng(seed)

    if stratified:
        classes = np.unique(y)
        if len(classes) >= 2:
            fold_bins = [[] for _ in range(n_splits)]
            for c in classes:
                idx = np.where(y == c)[0]
                rng.shuffle(idx)
                for i, row_idx in enumerate(idx.tolist()):
                    fold_bins[i % n_splits].append(row_idx)
            all_idx = np.arange(n, dtype=int)
            folds = []
            for fold_rows in fold_bins:
                test_idx = np.array(sorted(fold_rows), dtype=int)
                train_mask = np.ones(n, dtype=bool)
                train_mask[test_idx] = False
                folds.append(Fold(train_idx=all_idx[train_mask], test_idx=test_idx))
            return folds

    perm = np.arange(n, dtype=int)
    rng.shuffle(perm)
    test_parts = np.array_split(perm, n_splits)
    all_idx = np.arange(n, dtype=int)
    folds = []
    for test_idx in test_parts:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        folds.append(Fold(train_idx=all_idx[train_mask], test_idx=np.sort(test_idx)))
    return folds


def auc_score(y: np.ndarray, p: np.ndarray) -> float:
    order = np.argsort(p)
    y_sorted = y[order]
    n_pos = float(np.sum(y_sorted > 0.5))
    n_neg = float(np.sum(y_sorted <= 0.5))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = np.arange(1, len(y_sorted) + 1, dtype=float)
    rank_sum_pos = float(np.sum(ranks[y_sorted > 0.5]))
    return (rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def rmse_score(y: np.ndarray, mu: np.ndarray) -> float:
    return math.sqrt(float(np.mean((y - mu) ** 2)))


def r2_score(y: np.ndarray, mu: np.ndarray) -> float:
    sst = float(np.sum((y - float(np.mean(y))) ** 2))
    if sst <= 0.0:
        return 0.0
    sse = float(np.sum((y - mu) ** 2))
    return 1.0 - sse / sst


def _augment_bmi_spline_linear_hinges(train_df: pd.DataFrame, test_df: pd.DataFrame, n_knots: int = 6):
    if "bmi" not in train_df.columns or n_knots < 2:
        return train_df, test_df, []
    x_tr = train_df["bmi"].to_numpy(dtype=float)
    # Leakage-safe: knots from train-fold only.
    qs = np.linspace(0.1, 0.9, n_knots - 1)
    knots = np.unique(np.quantile(x_tr, qs))

    tr = train_df.copy()
    te = test_df.copy()
    cols = []
    base_col = "bmi_spline_0"
    tr[base_col] = tr["bmi"]
    te[base_col] = te["bmi"]
    cols.append(base_col)
    for j, k in enumerate(knots, start=1):
        cn = f"bmi_spline_{j}"
        tr[cn] = np.maximum(0.0, tr["bmi"] - float(k))
        te[cn] = np.maximum(0.0, te["bmi"] - float(k))
        cols.append(cn)
    tr = tr.drop(columns=["bmi"])
    te = te.drop(columns=["bmi"])
    return tr, te, cols


def _load_lidar_dataset():
    d = pd.read_csv(DATASET_DIR / "lidar.csv")
    d = d[["range", "logratio"]].dropna()
    rows = [{"range": float(r), "y": float(y)} for r, y in zip(d["range"], d["logratio"])]
    return {
        "family": "gaussian",
        "rows": rows,
        "features": ["range"],
        "target": "y",
    }


def _load_bone_dataset():
    d = pd.read_csv(DATASET_DIR / "bone.csv")
    d = d[["trt", "t", "d"]].dropna()
    rows = [
        {
            "trt_auto": 1.0 if str(trt).strip().strip('"').lower() == "auto" else 0.0,
            "t": float(t),
            "y": float(y),
        }
        for trt, t, y in zip(d["trt"], d["t"], d["d"])
    ]
    return {
        "family": "binomial",
        "rows": rows,
        "features": ["trt_auto", "t"],
        "target": "y",
    }


def _load_prostate_dataset():
    d = pd.read_csv(DATASET_DIR / "prostate.csv")
    d = d[["pc1", "pc2", "y"]].dropna()
    rows = [{"pc1": float(a), "pc2": float(b), "y": float(y)} for a, b, y in zip(d["pc1"], d["pc2"], d["y"])]
    return {
        "family": "binomial",
        "rows": rows,
        "features": ["pc1", "pc2"],
        "target": "y",
    }


def _load_wine_dataset():
    d = pd.read_csv(DATASET_DIR / "wine.csv")
    d = d[["year", "h_rain", "w_rain", "h_temp", "s_temp", "price"]]
    d = d.replace({"NA": np.nan}).dropna(subset=["price"])
    rows = [
        {
            "year": float(y),
            "h_rain": float(hr),
            "w_rain": float(wr),
            "h_temp": float(ht),
            "s_temp": float(st),
            "y": float(p),
        }
        for y, hr, wr, ht, st, p in zip(
            d["year"], d["h_rain"], d["w_rain"], d["h_temp"], d["s_temp"], d["price"]
        )
    ]
    return {
        "family": "gaussian",
        "rows": rows,
        "features": ["year", "h_rain", "w_rain", "h_temp", "s_temp"],
        "target": "y",
    }


def _load_wine_temp_vs_year_dataset():
    d = pd.read_csv(DATASET_DIR / "wine.csv")
    d = d[["year", "s_temp"]].dropna()
    rows = [{"year": float(y), "y": float(t)} for y, t in zip(d["year"], d["s_temp"])]
    return {
        "family": "gaussian",
        "rows": rows,
        "features": ["year"],
        "target": "y",
    }


def _load_wine_price_vs_temp_dataset():
    d = pd.read_csv(DATASET_DIR / "wine.csv")
    d = d[["s_temp", "price"]].replace({"NA": np.nan}).dropna()
    rows = [{"temp": float(t), "y": float(p)} for t, p in zip(d["s_temp"], d["price"])]
    return {
        "family": "gaussian",
        "rows": rows,
        "features": ["temp"],
        "target": "y",
    }


def _load_horse_dataset():
    d = pd.read_csv(DATASET_DIR / "horse.csv")
    d = d[["outcome", "pulse", "rectal_temp", "packed_cell_volume"]].copy()
    d["outcome"] = d["outcome"].astype(str).str.strip().str.lower()
    d = d[d["outcome"].isin({"lived", "died", "euthanized"})]
    for c in ["pulse", "rectal_temp", "packed_cell_volume"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["pulse", "rectal_temp", "packed_cell_volume"])
    rows = [
        {
            "rectal_temp": float(rt),
            "pulse": float(pu),
            "packed_cell_volume": float(pcv),
            "y": 0.0 if out == "lived" else 1.0,
        }
        for out, pu, rt, pcv in zip(d["outcome"], d["pulse"], d["rectal_temp"], d["packed_cell_volume"])
    ]
    if not rows:
        raise RuntimeError("horse dataset has no complete rows after filtering")
    return {
        "family": "binomial",
        "rows": rows,
        "features": ["rectal_temp", "pulse", "packed_cell_volume"],
        "target": "y",
    }


def _parse_f64_opt(raw):
    if raw is None:
        return None
    s = str(raw).strip()
    if not s or s.lower() == "na":
        return None
    try:
        v = float(s)
        if not np.isfinite(v):
            return None
        return v
    except ValueError:
        return None


def _mode_or_default(values, default):
    counts = {}
    for v in values:
        if v is None:
            continue
        t = str(v).strip()
        if not t or t.lower() == "na":
            continue
        counts[t] = counts.get(t, 0) + 1
    if not counts:
        return default
    return max(counts.items(), key=lambda kv: kv[1])[0]


def _encode_bool_yn(raw: str) -> float:
    return 1.0 if (raw or "").strip().upper() == "Y" else 0.0


def _encode_edema(raw: str) -> float:
    t = (raw or "").strip().upper()
    if t == "Y":
        return 2.0
    if t == "S":
        return 1.0
    return 0.0


def _load_cirrhosis_survival_dataset():
    d = pd.read_csv(DATASET_DIR / "cirrhosis.csv")
    numeric_cols = [
        "Age",
        "Bilirubin",
        "Cholesterol",
        "Albumin",
        "Copper",
        "Alk_Phos",
        "SGOT",
        "Tryglicerides",
        "Platelets",
        "Prothrombin",
        "Stage",
    ]
    rows = []
    for r in d.to_dict(orient="records"):
        drug = str(r.get("Drug", "")).strip()
        if not drug or drug.lower() == "na":
            continue
        status = str(r.get("Status", "")).strip().upper()
        if status not in {"D", "C", "CL"}:
            continue

        time = _parse_f64_opt(r.get("N_Days", ""))
        if time is None:
            continue
        time = max(time, 1.0)

        sex = str(r.get("Sex", "")).strip().upper()
        asc = str(r.get("Ascites", "")).strip().upper()
        hep = str(r.get("Hepatomegaly", "")).strip().upper()
        spi = str(r.get("Spiders", "")).strip().upper()
        ede = str(r.get("Edema", "")).strip().upper()
        if sex not in {"M", "F"}:
            continue
        if asc not in {"Y", "N"} or hep not in {"Y", "N"} or spi not in {"Y", "N"}:
            continue
        if ede not in {"N", "S", "Y"}:
            continue

        parsed_num = {}
        ok = True
        for col in numeric_cols:
            v = _parse_f64_opt(r.get(col, ""))
            if v is None:
                ok = False
                break
            parsed_num[col.lower()] = float(v)
        if not ok:
            continue

        out = {
            "time": float(time),
            "event": 1.0 if status == "D" else 0.0,
            "drug": 1.0 if "penicillamine" in drug.lower() else 0.0,
            "sex_male": 1.0 if sex == "M" else 0.0,
            "ascites": _encode_bool_yn(asc),
            "hepatomegaly": _encode_bool_yn(hep),
            "spiders": _encode_bool_yn(spi),
            "edema": _encode_edema(ede),
            **parsed_num,
        }
        rows.append(out)

    if not rows:
        raise RuntimeError("cirrhosis.csv has no complete-case rows for CV benchmarking")

    return {
        "family": "survival",
        "rows": rows,
        "features": [
            "drug",
            "sex_male",
            "ascites",
            "hepatomegaly",
            "spiders",
            "edema",
            "age",
            "bilirubin",
            "cholesterol",
            "albumin",
            "copper",
            "alk_phos",
            "sgot",
            "tryglicerides",
            "platelets",
            "prothrombin",
            "stage",
        ],
        "time_col": "time",
        "event_col": "event",
    }


def _normalize_us48_timestamp_to_hour(ts: str) -> float:
    # Example input: "2/21/2026 12 a.m. EST"
    cleaned = (
        ts.replace(" a.m.", " AM")
        .replace(" p.m.", " PM")
        .replace(" EST", "")
        .strip()
    )
    dt = pd.to_datetime(cleaned, format="%m/%d/%Y %I %p", errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"failed to parse timestamp '{ts}'")
    return float(dt.hour)


def _load_us48_demand_dataset(filename: str):
    d = pd.read_csv(DATASET_DIR / filename)
    needed = [
        "Timestamp (Hour Ending)",
        "Demand (MWh)",
        "Demand Forecast (MWh)",
        "Net Generation (MWh)",
        "Total Interchange (MWh)",
    ]
    d = d[needed].copy()
    for c in needed[1:]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=needed[1:])
    rows = [
        {
            "hour": _normalize_us48_timestamp_to_hour(str(ts)),
            "demand_forecast": float(f),
            "net_generation": float(g),
            "total_interchange": float(i),
            "y": float(y),
        }
        for ts, y, f, g, i in zip(
            d["Timestamp (Hour Ending)"],
            d["Demand (MWh)"],
            d["Demand Forecast (MWh)"],
            d["Net Generation (MWh)"],
            d["Total Interchange (MWh)"],
        )
    ]
    if not rows:
        raise RuntimeError(f"dataset '{filename}' has no complete rows")
    return {
        "family": "gaussian",
        "rows": rows,
        "features": ["hour", "demand_forecast", "net_generation", "total_interchange"],
        "target": "y",
    }


def _load_haberman_dataset():
    d = pd.read_csv(DATASET_DIR / "haberman.csv")
    d = d.iloc[:, :4].copy()
    d.columns = ["age", "op_year", "axil_nodes", "status"]
    for c in d.columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    rows = [
        {
            "age": float(a),
            "op_year": float(y),
            "axil_nodes": float(n),
            "y": 1.0 if int(s) == 2 else 0.0,
        }
        for a, y, n, s in zip(d["age"], d["op_year"], d["axil_nodes"], d["status"])
    ]
    if not rows:
        raise RuntimeError("haberman.csv has no rows")
    return {
        "family": "binomial",
        "rows": rows,
        "features": ["age", "op_year", "axil_nodes"],
        "target": "y",
    }


def _load_icu_survival_death_dataset():
    d = pd.read_csv(DATASET_DIR / "icu_survival_death.csv")
    d = d[["time", "age", "bmi", "hr_max", "sysbp_min", "event"]].dropna()
    rows = [
        {
            "time": float(t),
            "age": float(a),
            "bmi": float(b),
            "hr_max": float(h),
            "sysbp_min": float(s),
            "event": float(e),
        }
        for t, a, b, h, s, e in zip(
            d["time"], d["age"], d["bmi"], d["hr_max"], d["sysbp_min"], d["event"]
        )
    ]
    if not rows:
        raise RuntimeError("icu_survival_death dataset has no rows")
    return {
        "family": "survival",
        "rows": rows,
        "features": ["age", "bmi", "hr_max", "sysbp_min"],
        "time_col": "time",
        "event_col": "event",
    }


def _load_icu_survival_los_dataset():
    d = pd.read_csv(DATASET_DIR / "icu_survival_los.csv")
    d = d[["age", "bmi", "hr_max", "sysbp_min", "temp_apache", "time", "event"]].dropna()
    rows = [
        {
            "age": float(a),
            "bmi": float(b),
            "hr_max": float(h),
            "sysbp_min": float(s),
            "temp_apache": float(t),
            "time": float(tm),
            "event": float(e),
        }
        for a, b, h, s, t, tm, e in zip(
            d["age"], d["bmi"], d["hr_max"], d["sysbp_min"], d["temp_apache"], d["time"], d["event"]
        )
    ]
    if not rows:
        raise RuntimeError("icu_survival_los dataset has no rows")
    return {
        "family": "survival",
        "rows": rows,
        "features": ["age", "bmi", "hr_max", "sysbp_min", "temp_apache"],
        "time_col": "time",
        "event_col": "event",
    }


def _load_heart_failure_survival_dataset():
    d = pd.read_csv(DATASET_DIR / "heart_failure_clinical_records_dataset.csv")
    cols = [
        "time",
        "DEATH_EVENT",
        "age",
        "anaemia",
        "creatinine_phosphokinase",
        "diabetes",
        "ejection_fraction",
        "high_blood_pressure",
        "platelets",
        "serum_creatinine",
        "serum_sodium",
        "sex",
        "smoking",
    ]
    d = d[cols].copy()
    for c in cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna()
    rows = []
    for r in d.itertuples(index=False):
        cpk = float(r.creatinine_phosphokinase)
        plt = float(r.platelets)
        scr = float(r.serum_creatinine)
        rows.append(
            {
                "time": float(r.time),
                "event": float(r.DEATH_EVENT),
                "age": float(r.age),
                "anaemia": float(r.anaemia),
                "log_creatinine_phosphokinase": float(np.log1p(max(cpk, 0.0))),
                "diabetes": float(r.diabetes),
                "ejection_fraction": float(r.ejection_fraction),
                "high_blood_pressure": float(r.high_blood_pressure),
                "log_platelets": float(np.log1p(max(plt, 0.0))),
                "log_serum_creatinine": float(np.log1p(max(scr, 0.0))),
                "serum_sodium": float(r.serum_sodium),
                "sex": float(r.sex),
                "smoking": float(r.smoking),
            }
        )
    if not rows:
        raise RuntimeError("heart_failure_clinical_records_dataset.csv has no rows")
    return {
        "family": "survival",
        "rows": rows,
        "features": [
            "age",
            "anaemia",
            "log_creatinine_phosphokinase",
            "diabetes",
            "ejection_fraction",
            "high_blood_pressure",
            "log_platelets",
            "log_serum_creatinine",
            "serum_sodium",
            "sex",
            "smoking",
        ],
        "time_col": "time",
        "event_col": "event",
    }


def _synthetic_binomial_dataset(n, p, seed):
    p = max(int(p), 3)
    rng = np.random.default_rng(int(seed))
    x = np.zeros((int(n), p), dtype=float)
    x[:, 0] = 1.0
    if p > 1:
        x[:, 1:] = rng.normal(size=(int(n), p - 1))

    eta = -0.25 + 1.1 * x[:, 1] - 0.9 * x[:, 2] + 0.2 * np.sin(x[:, 2])
    pr = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(int(n)) < pr).astype(float)

    rows = []
    for i in range(int(n)):
        rows.append({"x1": float(x[i, 1]), "x2": float(x[i, 2]), "y": float(y[i])})

    return {
        "family": "binomial",
        "rows": rows,
        "features": ["x1", "x2"],
        "target": "y",
    }


def _synthetic_geo_disease_dataset(n=4000, seed=20260226):
    n = int(max(500, n))
    rng = np.random.default_rng(int(seed))

    # Hidden geospatial drivers (never exposed as model features).
    lat = rng.uniform(-1.0, 1.0, size=n)
    lon = rng.uniform(-1.0, 1.0, size=n)

    # Disease prevalence is highest near the equator (lat ~= 0).
    equator_closeness = 1.0 - np.abs(lat)
    geo_signal = (
        -1.00
        + 2.20 * equator_closeness
        + 0.55 * np.sin(np.pi * lon)
        + 0.35 * np.cos(2.25 * np.pi * lon)
        + 0.30 * np.sin(2.0 * np.pi * equator_closeness * lon)
    )

    # Heteroscedastic latent noise: stronger farther south (lat < 0).
    southness = np.clip(-lat, 0.0, 1.0)
    eta_noise_sd = 0.20 + 0.85 * (southness**1.35)
    eta = geo_signal + rng.normal(0.0, eta_noise_sd, size=n)
    pr = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < pr).astype(float)

    # PCs are noisy transforms of hidden geo variables.
    rows = []
    for i in range(n):
        row = {}
        for j in range(16):
            a = 0.95 - 0.045 * j
            b = 0.25 + 0.035 * j
            c = ((-1.0) ** j) * (0.10 + 0.01 * j)
            noise_sd = 0.15 + 0.015 * j
            pc = a * lat[i] + b * lon[i] + c * lat[i] * lon[i] + rng.normal(0.0, noise_sd)
            row[f"pc{j + 1}"] = float(pc)
        row["y"] = float(y[i])
        rows.append(row)

    return {
        "family": "binomial",
        "rows": rows,
        "features": [f"pc{i}" for i in range(1, 17)],
        "target": "y",
    }


def _synthetic_geo_disease_eas_dataset(n=6000, seed=20260301, n_pcs=16):
    n = int(max(1000, n))
    n_pcs = int(max(3, n_pcs))
    rng = np.random.default_rng(int(seed))

    # Hidden superpopulation assignment.
    eas = rng.random(n) < 0.23
    n_eas = int(np.sum(eas))

    # Hidden geographic coordinates.
    lat = rng.uniform(-55.0, 70.0, size=n)
    lon = rng.uniform(-175.0, 175.0, size=n)

    # East Asia geographic region for the EAS superpopulation.
    lat[eas] = rng.uniform(15.0, 52.0, size=n_eas)
    lon[eas] = rng.uniform(95.0, 145.0, size=n_eas)

    eta = np.full(n, math.log(0.02 / 0.98), dtype=float)
    eta[eas] = math.log(0.10 / 0.90)

    # Massive longitude/latitude effects are EAS-only by construction.
    lat_e = (lat[eas] - 33.5) / 11.0
    lon_e = (lon[eas] - 120.0) / 10.0
    eas_geo_signal = (
        3.25 * np.sin(1.35 * lat_e)
        - 2.85 * np.cos(1.55 * lon_e)
        + 2.50 * np.sin(1.10 * lat_e * lon_e)
        + 1.90 * np.cos(1.60 * lat_e + 0.45 * lon_e)
    )
    eas_geo_signal = eas_geo_signal - float(np.mean(eas_geo_signal))
    eta[eas] = eta[eas] + eas_geo_signal + rng.normal(0.0, 0.20, size=n_eas)
    eta[~eas] = eta[~eas] + rng.normal(0.0, 0.08, size=int(np.sum(~eas)))

    pr = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < pr).astype(float)

    lat_s = lat / 90.0
    lon_s = lon / 180.0

    rows = []
    for i in range(n):
        row = {}
        for j in range(n_pcs):
            a = 0.98 - 0.05 * j
            b = 0.26 + 0.03 * j
            c = ((-1.0) ** j) * (0.12 + 0.01 * j)
            d = 0.22 if j >= 8 else 0.06
            noise_sd = 0.13 + 0.018 * j
            pc = (
                a * lat_s[i]
                + b * lon_s[i]
                + c * lat_s[i] * lon_s[i]
                + d * float(eas[i])
                + rng.normal(0.0, noise_sd)
            )
            row[f"pc{j + 1}"] = float(pc)
        row["y"] = float(y[i])
        rows.append(row)

    return {
        "family": "binomial",
        "rows": rows,
        "features": [f"pc{i}" for i in range(1, n_pcs + 1)],
        "target": "y",
    }


def _geo_disease_eas_scenario_cfg(name):
    m = re.match(r"^geo_disease_(eas|eas3)_(tp|duchon|matern|psperpc)_k([0-9]+)$", str(name))
    if m is None:
        return None
    family_code = m.group(1)
    basis_code = m.group(2)
    knots = max(4, int(m.group(3)))
    n_pcs = 3 if family_code == "eas3" else 16
    smooth_pcs = 3 if n_pcs == 3 else 3
    linear_start = smooth_pcs + 1
    if basis_code == "tp":
        return {
            "smooth_basis": "thinplate",
            "smooth_cols": [f"pc{i}" for i in range(1, smooth_pcs + 1)],
            "linear_cols": [f"pc{i}" for i in range(linear_start, n_pcs + 1)],
            "knots": knots,
            "basis_code": basis_code,
            "n_pcs": n_pcs,
        }
    if basis_code == "duchon":
        return {
            "smooth_basis": "duchon",
            "smooth_cols": [f"pc{i}" for i in range(1, smooth_pcs + 1)],
            "linear_cols": [f"pc{i}" for i in range(linear_start, n_pcs + 1)],
            "knots": knots,
            "basis_code": basis_code,
            "n_pcs": n_pcs,
        }
    if basis_code == "matern":
        return {
            "smooth_basis": "matern",
            "smooth_cols": [f"pc{i}" for i in range(1, smooth_pcs + 1)],
            "linear_cols": [f"pc{i}" for i in range(linear_start, n_pcs + 1)],
            "knots": knots,
            "basis_code": basis_code,
            "n_pcs": n_pcs,
        }
    return {
        "smooth_basis": "bspline_per_pc",
        "smooth_cols": [f"pc{i}" for i in range(1, n_pcs + 1)],
        "linear_cols": [],
        "knots": knots,
        "basis_code": basis_code,
        "n_pcs": n_pcs,
    }


def _papuan_oce_scenario_cfg(name):
    m = re.match(r"^papuan_oce(4)?_(tp|duchon|matern|psperpc)(_basic)?_k([0-9]+)$", str(name))
    if m is None:
        return None
    is_four_pc = m.group(1) is not None
    basis_code = m.group(2)
    is_basic = m.group(3) is not None
    knots = max(4, int(m.group(4)))
    n_pcs = 4 if is_four_pc else 16
    if basis_code == "psperpc":
        return {
            "smooth_basis": "bspline_per_pc",
            "smooth_cols": [f"pc{i}" for i in range(1, n_pcs + 1)],
            "linear_cols": [],
            "knots": knots,
            "basis_code": basis_code,
            "n_pcs": n_pcs,
            "is_basic": is_basic,
        }
    smooth_basis = {"tp": "thinplate", "duchon": "duchon", "matern": "matern"}[basis_code]
    return {
        "smooth_basis": smooth_basis,
        "smooth_cols": [f"pc{i}" for i in range(1, min(3, n_pcs) + 1)],
        "linear_cols": [f"pc{i}" for i in range(min(3, n_pcs) + 1, n_pcs + 1)],
        "knots": knots,
        "basis_code": basis_code,
        "n_pcs": n_pcs,
        "is_basic": is_basic,
    }


def _compute_pcs_from_genotypes(g, n_pcs=16):
    # Proper PCA from standardized genotype-like matrix using SVD.
    n = g.shape[0]
    x = g.astype(float)
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, ddof=1, keepdims=True)
    sd[~np.isfinite(sd) | (sd < 1e-8)] = 1.0
    x = (x - mu) / sd
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    k = min(int(n_pcs), u.shape[1])
    pcs = u[:, :k] * s[:k]
    pcs = pcs / max(math.sqrt(max(n - 1, 1)), 1.0)
    pcs_mu = pcs.mean(axis=0, keepdims=True)
    pcs_sd = pcs.std(axis=0, ddof=1, keepdims=True)
    pcs_sd[~np.isfinite(pcs_sd) | (pcs_sd < 1e-8)] = 1.0
    return (pcs - pcs_mu) / pcs_sd


def _synthetic_papuan_oce_dataset(n=6000, seed=20260315, n_pcs=16):
    n = int(max(1200, n))
    rng = np.random.default_rng(int(seed))

    # Hidden group assignment only; labels are never emitted as model features.
    groups = [
        ("Papuan", 0.12),
        ("Papuan Sepik", 0.08),
        ("Papuan,Papuan Highlands", 0.06),
        ("Papuan,Papuan Sepik", 0.06),
        ("Bougainville", 0.08),
        ("Han", 0.17),
        ("Japanese", 0.10),
        ("CEU", 0.14),
        ("Yoruba", 0.11),
        ("Pima", 0.08),
    ]
    names = [g[0] for g in groups]
    probs = np.array([g[1] for g in groups], dtype=float)
    probs = probs / probs.sum()
    pop_idx = rng.choice(len(names), size=n, p=probs)
    pop = np.array([names[i] for i in pop_idx], dtype=object)

    # Latent ancestry coordinates and genotype-like matrix.
    centers = {
        "Papuan": np.array([2.7, -0.8, 1.7, 0.7]),
        "Papuan Sepik": np.array([3.0, -0.6, 1.5, 0.8]),
        "Papuan,Papuan Highlands": np.array([2.9, -0.3, 1.9, 0.4]),
        "Papuan,Papuan Sepik": np.array([3.1, -0.5, 1.6, 0.8]),
        "Bougainville": np.array([2.5, -1.0, 1.8, 0.5]),
        "Han": np.array([0.4, 2.0, -0.1, 0.2]),
        "Japanese": np.array([0.7, 1.8, -0.2, 0.1]),
        "CEU": np.array([-2.1, 0.4, 0.1, -0.2]),
        "Yoruba": np.array([-1.9, -2.2, 0.5, 0.0]),
        "Pima": np.array([-0.3, -0.9, -1.6, 0.4]),
    }

    z = np.zeros((n, 4), dtype=float)
    for i in range(n):
        z[i] = centers[str(pop[i])] + rng.normal(0.0, 0.55, size=4)

    n_snps = 700
    loadings = rng.normal(0.0, 0.28, size=(4, n_snps))
    logits = -0.2 + z @ loadings + rng.normal(0.0, 0.35, size=(n, n_snps))
    p = 1.0 / (1.0 + np.exp(-np.clip(logits, -10.0, 10.0)))
    g = rng.binomial(2, p).astype(float)

    pcs = _compute_pcs_from_genotypes(g, n_pcs=max(3, int(n_pcs)))
    if pcs.shape[1] < n_pcs:
        pcs = np.concatenate([pcs, np.zeros((n, n_pcs - pcs.shape[1]), dtype=float)], axis=1)

    # Prevalence: 0.4 in Papuan* and Bougainville; 0.02 in all others.
    high_risk = np.array([("Papuan" in str(lbl)) or (str(lbl) == "Bougainville") for lbl in pop], dtype=bool)
    prevalence = np.where(high_risk, 0.40, 0.02)
    y = (rng.random(n) < prevalence).astype(float)

    rows = []
    for i in range(n):
        row = {f"pc{j + 1}": float(pcs[i, j]) for j in range(int(n_pcs))}
        row["y"] = float(y[i])
        rows.append(row)

    return {
        "family": "binomial",
        "rows": rows,
        "features": [f"pc{i}" for i in range(1, int(n_pcs) + 1)],
        "target": "y",
    }


def dataset_for_scenario(s):
    name = s["name"]
    if name in {"small_dense", "medium", "pathological_ill_conditioned"}:
        return _synthetic_binomial_dataset(s["n"], s["p"], s.get("seed", 42))
    if name.startswith("geo_disease_eas3_"):
        return _synthetic_geo_disease_eas_dataset(s.get("n", 6000), s.get("seed", 20260301), n_pcs=3)
    if name.startswith("geo_disease_eas_"):
        return _synthetic_geo_disease_eas_dataset(s.get("n", 6000), s.get("seed", 20260301))
    if name.startswith("papuan_oce4_"):
        return _synthetic_papuan_oce_dataset(s.get("n", 6000), s.get("seed", 20260315), n_pcs=4)
    if name.startswith("papuan_oce_"):
        return _synthetic_papuan_oce_dataset(s.get("n", 6000), s.get("seed", 20260315), n_pcs=16)
    if name.startswith("geo_disease_"):
        return _synthetic_geo_disease_dataset(s.get("n", 4000), s.get("seed", 20260226))
    if name == "lidar_semipar":
        return _load_lidar_dataset()
    if name == "bone_gamair":
        return _load_bone_dataset()
    if name == "prostate_gamair":
        return _load_prostate_dataset()
    if name == "wine_gamair":
        return _load_wine_dataset()
    if name == "wine_temp_vs_year":
        return _load_wine_temp_vs_year_dataset()
    if name == "wine_price_vs_temp":
        return _load_wine_price_vs_temp_dataset()
    if name == "horse_colic":
        return _load_horse_dataset()
    if name == "us48_demand_5day":
        return _load_us48_demand_dataset("five_day.csv")
    if name == "us48_demand_31day":
        return _load_us48_demand_dataset("31_day.csv")
    if name == "haberman_survival":
        return _load_haberman_dataset()
    if name == "icu_survival_death":
        return _load_icu_survival_death_dataset()
    if name == "icu_survival_los":
        return _load_icu_survival_los_dataset()
    if name == "heart_failure_survival":
        return _load_heart_failure_survival_dataset()
    if name == "cirrhosis_survival":
        return _load_cirrhosis_survival_dataset()
    raise RuntimeError(f"No scenario-specific dataset loader configured for '{name}'")


def folds_for_dataset(ds):
    if ds["family"] == "survival":
        y = np.array([float(r[ds["event_col"]]) for r in ds["rows"]], dtype=float)
        stratified = True
    else:
        y = np.array([float(r[ds["target"]]) for r in ds["rows"]], dtype=float)
        stratified = ds["family"] == "binomial"
    folds = make_folds(y, n_splits=CV_SPLITS, seed=CV_SEED, stratified=stratified)
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
    if not np.all(seen == 1):
        raise RuntimeError("invalid CV split: each row must appear exactly once in test sets")
    return folds


def aggregate_cv_rows(cv_rows, family):
    total_n = max(sum(int(r["n_test"]) for r in cv_rows), 1)
    fit_sec = float(sum(float(r["fit_sec"]) for r in cv_rows))
    predict_sec = float(sum(float(r["predict_sec"]) for r in cv_rows))

    def wavg(key):
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
            "rmse": None,
            "r2": None,
        }
    if family == "survival":
        return {
            "fit_sec": fit_sec,
            "predict_sec": predict_sec,
            "auc": wavg("auc"),  # concordance index
            "brier": None,
            "c_index": wavg("auc"),
            "rmse": None,
            "r2": None,
        }
    return {
        "fit_sec": fit_sec,
        "predict_sec": predict_sec,
        "auc": None,
        "brier": None,
        "rmse": wavg("rmse"),
        "r2": wavg("r2"),
    }


def write_dataset_csv(ds, path):
    if ds["family"] == "survival":
        cols = list(ds["features"]) + [ds["time_col"], ds["event_col"]]
    else:
        cols = list(ds["features"]) + [ds["target"]]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in ds["rows"]:
            writer.writerow({k: row[k] for k in cols})


def _rust_fit_mapping(scenario_name):
    geo_eas_cfg = _geo_disease_eas_scenario_cfg(scenario_name)
    papuan_cfg = _papuan_oce_scenario_cfg(scenario_name)
    if geo_eas_cfg is not None:
        return {
            "family": "binomial-logit",
            "smooth_cols": geo_eas_cfg["smooth_cols"],
            "smooth_basis": geo_eas_cfg["smooth_basis"],
            "linear_cols": geo_eas_cfg["linear_cols"],
            "knots": int(geo_eas_cfg["knots"]),
        }
    if papuan_cfg is not None:
        return {
            "family": "binomial-logit",
            "smooth_cols": papuan_cfg["smooth_cols"],
            "smooth_basis": papuan_cfg["smooth_basis"],
            "linear_cols": papuan_cfg["linear_cols"],
            "knots": int(papuan_cfg["knots"]),
        }
    return {
        "small_dense": dict(
            family="binomial-logit",
            smooth_col="x2",
            linear_cols=["x1"],
            smooth_basis="ps",
            knots=7,
            double_penalty=False,
        ),
        "medium": dict(family="binomial-logit", smooth_col="x2", linear_cols=["x1"], smooth_basis="ps", knots=7),
        "pathological_ill_conditioned": dict(
            family="binomial-logit", smooth_col="x2", linear_cols=["x1"], smooth_basis="ps", knots=7
        ),
        "lidar_semipar": dict(family="gaussian", smooth_col="range", linear_cols=[], smooth_basis="ps", knots=24),
        "bone_gamair": dict(family="binomial-logit", smooth_col="t", linear_cols=["trt_auto"], smooth_basis="ps", knots=8),
        "prostate_gamair": dict(
            family="binomial-logit",
            smooth_col="pc2",
            linear_cols=["pc1"],
            smooth_basis="ps",
            knots=8,
            double_penalty=False,
        ),
        "horse_colic": dict(
            family="binomial-logit",
            smooth_col="pulse",
            linear_cols=["rectal_temp", "packed_cell_volume"],
            smooth_basis="ps",
            knots=8,
            double_penalty=False,
        ),
        "wine_gamair": dict(
            family="gaussian",
            smooth_col="s_temp",
            linear_cols=["year", "h_rain", "w_rain", "h_temp"],
            smooth_basis="ps",
            knots=7,
        ),
        "wine_temp_vs_year": dict(
            family="gaussian", smooth_col="year", linear_cols=[], smooth_basis="ps", knots=7
        ),
        "wine_price_vs_temp": dict(
            family="gaussian", smooth_col="temp", linear_cols=[], smooth_basis="ps", knots=7
        ),
        "us48_demand_5day": dict(
            family="gaussian",
            smooth_col="hour",
            linear_cols=["demand_forecast", "net_generation", "total_interchange"],
            smooth_basis="ps",
            knots=8,
        ),
        "us48_demand_31day": dict(
            family="gaussian",
            smooth_col="hour",
            linear_cols=["demand_forecast", "net_generation", "total_interchange"],
            smooth_basis="ps",
            knots=12,
        ),
        "haberman_survival": dict(
            family="binomial-logit", smooth_col="axil_nodes", linear_cols=["age", "op_year"], smooth_basis="ps", knots=8
        ),
        "icu_survival_death": dict(
            family="binomial-logit",
            smooth_col="time",
            linear_cols=["age", "bmi", "hr_max", "sysbp_min"],
            smooth_basis="ps",
            knots=7,
        ),
        "icu_survival_los": dict(
            family="binomial-logit",
            smooth_col="age",
            linear_cols=["bmi", "hr_max", "sysbp_min", "temp_apache", "time"],
            smooth_basis="ps",
            knots=7,
        ),
        "geo_disease_tp": dict(
            family="binomial-logit",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="thinplate",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
        ),
        "geo_disease_tp_basic": dict(
            family="binomial-logit",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="thinplate",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
        ),
        "geo_disease_duchon": dict(
            family="binomial-logit",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="duchon",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
        ),
        "geo_disease_duchon_basic": dict(
            family="binomial-logit",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="duchon",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
        ),
        "geo_disease_matern": dict(
            family="binomial-logit",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="matern",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
        ),
        "geo_disease_shrinkage": dict(
            family="binomial-logit",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="matern",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
        ),
        "geo_disease_matern_basic": dict(
            family="binomial-logit",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="matern",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
        ),
        "geo_disease_ps_per_pc": dict(
            family="binomial-logit",
            smooth_cols=[f"pc{i}" for i in range(1, 17)],
            smooth_basis="ps",
            linear_cols=[],
            knots=10,
        ),
        "geo_disease_ps_per_pc_basic": dict(
            family="binomial-logit",
            smooth_cols=[f"pc{i}" for i in range(1, 17)],
            smooth_basis="ps",
            linear_cols=[],
            knots=10,
        ),
    }.get(scenario_name)


def _rust_formula_for_scenario(scenario_name, ds):
    cfg = _rust_fit_mapping(scenario_name)
    if cfg is None:
        raise RuntimeError(f"No Rust formula mapping configured for scenario '{scenario_name}'")
    target = ds["target"]
    terms = [f"linear({c})" for c in cfg.get("linear_cols", [])]
    basis = str(cfg.get("smooth_basis", "ps")).lower()
    knots = int(cfg.get("knots", 8))
    smooth_cols = cfg.get("smooth_cols")
    if smooth_cols:
        for col in smooth_cols:
            if basis in {"ps", "bspline", "p-spline"}:
                if "double_penalty" in cfg:
                    dp = "true" if bool(cfg["double_penalty"]) else "false"
                    terms.append(f"s({col}, type=ps, knots={knots}, double_penalty={dp})")
                else:
                    terms.append(f"s({col}, type=ps, knots={knots})")
            elif basis in {"thinplate", "tps"}:
                terms.append(f"s({col}, type=tps, centers={knots})")
            elif basis == "duchon":
                terms.append(f"s({col}, type=duchon, centers={knots})")
            elif basis == "matern":
                terms.append(f"s({col}, type=matern, centers={knots})")
            else:
                if "double_penalty" in cfg:
                    dp = "true" if bool(cfg["double_penalty"]) else "false"
                    terms.append(f"s({col}, type=ps, knots={knots}, double_penalty={dp})")
                else:
                    terms.append(f"s({col}, type=ps, knots={knots})")
    else:
        col = cfg["smooth_col"]
        if "double_penalty" in cfg:
            dp = "true" if bool(cfg["double_penalty"]) else "false"
            terms.append(f"s({col}, type=ps, knots={knots}, double_penalty={dp})")
        else:
            terms.append(f"s({col}, type=ps, knots={knots})")

    if not terms:
        raise RuntimeError(f"empty Rust term list for scenario '{scenario_name}'")
    formula = f"{target} ~ " + " + ".join(terms)
    return cfg["family"], formula


def _rust_survival_formula_for_scenario(scenario_name):
    if scenario_name == "icu_survival_death":
        return "linear(age) + s(bmi, type=ps, knots=6) + linear(hr_max) + linear(sysbp_min)"
    if scenario_name == "heart_failure_survival":
        return (
            "linear(age) + linear(anaemia) + linear(log_creatinine_phosphokinase) + linear(diabetes) + "
            "linear(ejection_fraction) + linear(high_blood_pressure) + linear(log_platelets) + "
            "linear(log_serum_creatinine) + linear(serum_sodium) + linear(sex) + linear(smoking)"
        )
    if scenario_name == "cirrhosis_survival":
        return (
            "linear(drug) + linear(sex_male) + linear(ascites) + linear(hepatomegaly) + linear(spiders) + "
            "linear(edema) + linear(age) + linear(bilirubin) + linear(cholesterol) + linear(albumin) + "
            "linear(copper) + linear(alk_phos) + linear(sgot) + linear(tryglicerides) + linear(platelets) + "
            "linear(prothrombin) + linear(stage)"
        )
    if scenario_name == "haberman_survival":
        return "linear(age) + linear(op_year) + s(axil_nodes, type=ps, knots=8)"
    if scenario_name == "icu_survival_los":
        return "linear(age) + linear(bmi) + linear(hr_max) + linear(sysbp_min) + linear(temp_apache)"
    raise RuntimeError(f"No Rust survival formula configured for scenario '{scenario_name}'")

def _rust_survival_fit_options_for_scenario(scenario_name):
    # Keep defaults close to CLI behavior and only enable a flexible time basis
    # where it provides clear discrimination gains.
    if scenario_name in {"heart_failure_survival", "cirrhosis_survival"}:
        return {
            "time_basis": "bspline",
            "time_degree": 3,
            "time_num_internal_knots": 8,
            "time_smooth_lambda": 1e-2,
            "ridge_lambda": 1e-6,
        }
    return {
        "time_basis": "linear",
        "ridge_lambda": 1e-4,
    }


def _ensure_rust_binary():
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


def rust_cli_cv_args(scenario_name, ds):
    if ds["family"] == "survival":
        return [
            "--family",
            "survival",
            "--time-col",
            ds["time_col"],
            "--event-col",
            ds["event_col"],
        ]

    geo_eas_cfg = _geo_disease_eas_scenario_cfg(scenario_name)
    papuan_cfg = _papuan_oce_scenario_cfg(scenario_name)
    if geo_eas_cfg is not None:
        cfg = {
            "family": "binomial",
            "smooth_cols": geo_eas_cfg["smooth_cols"],
            "smooth_basis": geo_eas_cfg["smooth_basis"],
            "linear_cols": geo_eas_cfg["linear_cols"],
            "knots": geo_eas_cfg["knots"],
            "double_penalty": True,
        }
    elif papuan_cfg is not None:
        cfg = {
            "family": "binomial",
            "smooth_cols": papuan_cfg["smooth_cols"],
            "smooth_basis": papuan_cfg["smooth_basis"],
            "linear_cols": papuan_cfg["linear_cols"],
            "knots": papuan_cfg["knots"],
            "double_penalty": not papuan_cfg["is_basic"],
        }
    else:
        cfg = {
        "small_dense": dict(family="binomial", smooth_col="x2", linear_cols=["x1"], knots=7, double_penalty=True),
        "medium": dict(family="binomial", smooth_col="x2", linear_cols=["x1"], knots=7, double_penalty=True),
        "pathological_ill_conditioned": dict(
            family="binomial", smooth_col="x2", linear_cols=["x1"], knots=7, double_penalty=True
        ),
        "lidar_semipar": dict(family="gaussian", smooth_col="range", linear_cols=[], knots=24, double_penalty=True),
        "bone_gamair": dict(family="binomial", smooth_col="t", linear_cols=["trt_auto"], knots=8, double_penalty=True),
        "prostate_gamair": dict(family="binomial", smooth_col="pc2", linear_cols=["pc1"], knots=8, double_penalty=True),
        "horse_colic": dict(
            family="binomial",
            smooth_col="pulse",
            linear_cols=["rectal_temp", "packed_cell_volume"],
            knots=8,
            double_penalty=True,
        ),
        "wine_gamair": dict(
            family="gaussian",
            smooth_col="s_temp",
            linear_cols=["year", "h_rain", "w_rain", "h_temp"],
            knots=7,
            double_penalty=False,
        ),
        "wine_temp_vs_year": dict(
            family="gaussian", smooth_col="year", linear_cols=[], knots=7, double_penalty=False
        ),
        "wine_price_vs_temp": dict(
            family="gaussian", smooth_col="temp", linear_cols=[], knots=7, double_penalty=False
        ),
        "us48_demand_5day": dict(
            family="gaussian",
            smooth_col="hour",
            linear_cols=["demand_forecast", "net_generation", "total_interchange"],
            knots=8,
            double_penalty=False,
        ),
        "us48_demand_31day": dict(
            family="gaussian",
            smooth_col="hour",
            linear_cols=["demand_forecast", "net_generation", "total_interchange"],
            knots=12,
            double_penalty=False,
        ),
        "haberman_survival": dict(
            family="binomial", smooth_col="axil_nodes", linear_cols=["age", "op_year"], knots=8, double_penalty=True
        ),
        "icu_survival_death": dict(
            family="binomial",
            smooth_col="time",
            linear_cols=["age", "bmi", "hr_max", "sysbp_min"],
            knots=7,
            double_penalty=False,
        ),
        "icu_survival_los": dict(
            family="binomial",
            smooth_col="age",
            linear_cols=["bmi", "hr_max", "sysbp_min", "temp_apache", "time"],
            knots=7,
            double_penalty=False,
        ),
        "geo_disease_tp": dict(
            family="binomial",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="thinplate",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
            double_penalty=True,
        ),
        "geo_disease_tp_basic": dict(
            family="binomial",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="thinplate",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
            double_penalty=False,
        ),
        "geo_disease_duchon": dict(
            family="binomial",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="duchon",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
            double_penalty=True,
        ),
        "geo_disease_duchon_basic": dict(
            family="binomial",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="duchon",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
            double_penalty=False,
        ),
        "geo_disease_matern": dict(
            family="binomial",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="matern",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
            double_penalty=True,
        ),
        # Backward-compatible alias.
        "geo_disease_shrinkage": dict(
            family="binomial",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="matern",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
            double_penalty=True,
        ),
        "geo_disease_matern_basic": dict(
            family="binomial",
            smooth_cols=["pc1", "pc2", "pc3", "pc4"],
            smooth_basis="matern",
            linear_cols=[f"pc{i}" for i in range(5, 17)],
            knots=12,
            double_penalty=False,
        ),
        "geo_disease_ps_per_pc": dict(
            family="binomial",
            smooth_cols=[f"pc{i}" for i in range(1, 17)],
            smooth_basis="bspline_per_pc",
            linear_cols=[],
            knots=10,
            double_penalty=True,
        ),
        "geo_disease_ps_per_pc_basic": dict(
            family="binomial",
            smooth_cols=[f"pc{i}" for i in range(1, 17)],
            smooth_basis="bspline_per_pc",
            linear_cols=[],
            knots=10,
            double_penalty=False,
        ),
        }.get(scenario_name)
    if cfg is None:
        raise RuntimeError(f"No Rust CLI CV mapping configured for scenario '{scenario_name}'")

    args = [
        "--family",
        cfg["family"],
        "--target-col",
        ds["target"],
        "--linear-cols",
        ",".join(cfg["linear_cols"]),
        "--num-internal-knots",
        str(int(cfg["knots"])),
        "--double-penalty",
        "true" if cfg["double_penalty"] else "false",
    ]
    if cfg.get("smooth_basis"):
        args.extend(["--smooth-basis", cfg["smooth_basis"]])
    smooth_cols = cfg.get("smooth_cols")
    if smooth_cols:
        args.extend(["--smooth-cols", ",".join(smooth_cols)])
    else:
        args.extend(["--smooth-col", cfg["smooth_col"]])
    return args


def run_rust_scenario_cv(scenario):
    scenario_name = scenario["name"]
    ds = dataset_for_scenario(scenario)
    folds = folds_for_dataset(ds)

    try:
        rust_bin = _ensure_rust_binary()
    except Exception as e:
        return {
            "contender": "rust_gam",
            "scenario_name": scenario_name,
            "status": "failed",
            "error": str(e),
        }

    base_df = pd.DataFrame(ds["rows"])
    cv_rows = []
    # Use a workspace-local temp root to reduce /tmp lifecycle flakiness in CI.
    with tempfile.TemporaryDirectory(prefix="gam_bench_rust_cv_", dir=str(BENCH_DIR)) as td:
        td_path = Path(td)
        for fold_id, fold in enumerate(folds):
            train_df = base_df.iloc[fold.train_idx].copy()
            test_df = base_df.iloc[fold.test_idx].copy()
            test_eval_df = test_df.copy()
            train_path = td_path / f"train_{fold_id}.csv"
            test_path = td_path / f"test_{fold_id}.csv"
            model_path = td_path / f"model_{fold_id}.json"
            pred_path = td_path / f"pred_{fold_id}.csv"

            if ds["family"] == "survival":
                for col in ds["features"]:
                    mu = float(train_df[col].mean())
                    sdv = float(train_df[col].std())
                    if (not np.isfinite(sdv)) or sdv < 1e-8:
                        sdv = 1.0
                    train_df[col] = (train_df[col] - mu) / sdv
                    test_df[col] = (test_df[col] - mu) / sdv
                train_df["__entry"] = 0.0
                test_df["__entry"] = 0.0
                # Use a fixed time horizon per fold to avoid leakage from scoring at each
                # subject's observed event/censoring time.
                horizon = float(np.median(train_df[ds["time_col"]].to_numpy(dtype=float)))
                if not np.isfinite(horizon) or horizon <= 0.0:
                    horizon = 1.0
                test_df[ds["time_col"]] = horizon
                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)
                formula = _rust_survival_formula_for_scenario(scenario_name)
                fit_cfg = _rust_survival_fit_options_for_scenario(scenario_name)
                fit_cmd = [
                    str(rust_bin),
                    "survival",
                    str(train_path),
                    "--entry",
                    "__entry",
                    "--exit",
                    ds["time_col"],
                    "--event",
                    ds["event_col"],
                    "--formula",
                    formula,
                    "--ridge-lambda",
                    str(fit_cfg["ridge_lambda"]),
                    "--time-basis",
                    fit_cfg["time_basis"],
                    "--out",
                    str(model_path),
                ]
                if fit_cfg["time_basis"] == "bspline":
                    fit_cmd.extend(
                        [
                            "--time-degree",
                            str(int(fit_cfg["time_degree"])),
                            "--time-num-internal-knots",
                            str(int(fit_cfg["time_num_internal_knots"])),
                            "--time-smooth-lambda",
                            str(float(fit_cfg["time_smooth_lambda"])),
                        ]
                    )
            else:
                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)
                family, formula = _rust_formula_for_scenario(scenario_name, ds)
                fit_cmd = [
                    str(rust_bin),
                    "fit",
                    "--family",
                    family,
                    "--formula",
                    formula,
                    "--out",
                    str(model_path),
                    str(train_path),
                ]

            def _looks_like_missing_csv(msg: str) -> bool:
                m = (msg or "").lower()
                return ("failed to open csv" in m) and ("no such file or directory" in m)

            def _ensure_fold_csvs():
                train_df.to_csv(train_path, index=False)
                test_df.to_csv(test_path, index=False)

            t0 = perf_counter()
            code, out, err = run_cmd(fit_cmd, cwd=ROOT)
            # Defensive retry for sporadic CI file-not-found while opening fold CSVs.
            if code != 0 and _looks_like_missing_csv(err or out):
                _ensure_fold_csvs()
                code, out, err = run_cmd(fit_cmd, cwd=ROOT)
            fit_sec = perf_counter() - t0
            if code != 0:
                return {
                    "contender": "rust_gam",
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "error": (err.strip() or out.strip() or "rust fit failed"),
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
            if code != 0 and _looks_like_missing_csv(err or out):
                _ensure_fold_csvs()
                code, out, err = run_cmd(pred_cmd, cwd=ROOT)
            pred_sec = perf_counter() - t1
            if code != 0:
                return {
                    "contender": "rust_gam",
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "error": (err.strip() or out.strip() or "rust predict failed"),
                }
            pred_df = pd.read_csv(pred_path)
            if "mean" not in pred_df.columns:
                return {
                    "contender": "rust_gam",
                    "scenario_name": scenario_name,
                    "status": "failed",
                    "error": "rust prediction output missing 'mean' column",
                }
            pred = pred_df["mean"].to_numpy(dtype=float)

            if ds["family"] == "binomial":
                y_test = test_df[ds["target"]].to_numpy(dtype=float)
                cv_rows.append(
                    {
                        "fit_sec": float(fit_sec),
                        "predict_sec": float(pred_sec),
                        "auc": auc_score(y_test, pred),
                        "brier": brier_score(y_test, pred),
                        "n_test": int(len(fold.test_idx)),
                        "model_spec": "gam fit/predict via release binary [5-fold CV]",
                    }
                )
            elif ds["family"] == "gaussian":
                y_test = test_df[ds["target"]].to_numpy(dtype=float)
                cv_rows.append(
                    {
                        "fit_sec": float(fit_sec),
                        "predict_sec": float(pred_sec),
                        "rmse": rmse_score(y_test, pred),
                        "r2": r2_score(y_test, pred),
                        "n_test": int(len(fold.test_idx)),
                        "model_spec": "gam fit/predict via release binary [5-fold CV]",
                    }
                )
            else:
                event_times = test_eval_df[ds["time_col"]].to_numpy(dtype=float)
                events = test_eval_df[ds["event_col"]].to_numpy(dtype=float)
                risk_score = (
                    -pred_df["eta"].to_numpy(dtype=float)
                    if "eta" in pred_df.columns
                    else pred
                )
                cidx = float(concordance_index(event_times, risk_score, event_observed=events))
                cv_rows.append(
                    {
                        "fit_sec": float(fit_sec),
                        "predict_sec": float(pred_sec),
                        "auc": cidx,
                        "n_test": int(len(fold.test_idx)),
                        "model_spec": "gam survival/predict via release binary (fixed-horizon c-index on survival score) [5-fold CV]",
                    }
                )

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": "rust_gam",
        "scenario_name": scenario_name,
        "status": "ok",
        **metrics,
        "model_spec": cv_rows[0]["model_spec"],
    }


def run_external_mgcv_cv(scenario):
    ds = dataset_for_scenario(scenario)
    folds = folds_for_dataset(ds)

    with tempfile.TemporaryDirectory(prefix="gam_bench_mgcv_cv_") as td:
        td_path = Path(td)
        data_path = td_path / "data.json"
        out_path = td_path / "out.json"
        script_path = td_path / "run_mgcv_cv.R"

        payload = {"dataset": ds, "scenario_name": scenario["name"]}
        data_path.write_text(json.dumps(payload))

        script = r'''
args <- commandArgs(trailingOnly = TRUE)
data_path <- args[1]
out_path <- args[2]
train_idx_path <- args[3]
test_idx_path <- args[4]

suppressPackageStartupMessages({
  library(mgcv)
  library(jsonlite)
})

payload <- fromJSON(data_path, simplifyVector = TRUE)
df <- as.data.frame(payload$dataset$rows)
family_name <- as.character(payload$dataset$family)
scenario_name <- as.character(payload$scenario_name)

train_idx <- scan(train_idx_path, what=integer(), quiet=TRUE) + 1L
test_idx <- scan(test_idx_path, what=integer(), quiet=TRUE) + 1L

train_df <- df[train_idx, , drop=FALSE]
test_df <- df[test_idx, , drop=FALSE]
y_test <- NULL
if (family_name != "survival") {
  y_all <- as.numeric(df[[payload$dataset$target]])
  y_test <- y_all[test_idx]
}

fam <- if (family_name == "binomial") binomial(link="logit") else gaussian(link="identity")

if (family_name == "survival") {
  suppressPackageStartupMessages(library(survival))
  feature_cols <- as.character(payload$dataset$features)
  for (cn in feature_cols) {
    mu <- mean(train_df[[cn]])
    sdv <- stats::sd(train_df[[cn]])
    if (!is.finite(sdv) || sdv < 1e-8) sdv <- 1.0
    train_df[[cn]] <- (train_df[[cn]] - mu) / sdv
    test_df[[cn]] <- (test_df[[cn]] - mu) / sdv
  }
  if (scenario_name == "icu_survival_death") {
    ftxt <- "Surv(time, event) ~ age + pspline(bmi, df=6) + hr_max + sysbp_min"
  } else if (scenario_name == "heart_failure_survival") {
    ftxt <- "Surv(time, event) ~ age + anaemia + log_creatinine_phosphokinase + diabetes + ejection_fraction + high_blood_pressure + log_platelets + log_serum_creatinine + serum_sodium + sex + smoking"
  } else if (scenario_name == "cirrhosis_survival") {
    ftxt <- "Surv(time, event) ~ drug + sex_male + ascites + hepatomegaly + spiders + edema + age + bilirubin + cholesterol + albumin + copper + alk_phos + sgot + tryglicerides + platelets + prothrombin + stage"
  } else {
    ftxt <- "Surv(time, event) ~ age + bmi + hr_max + sysbp_min + temp_apache"
  }
  t0 <- proc.time()[["elapsed"]]
  fit <- coxph(as.formula(ftxt), data=train_df, ties="efron")
  fit_sec <- proc.time()[["elapsed"]] - t0

  pred_t0 <- proc.time()[["elapsed"]]
  lp <- as.numeric(predict(fit, newdata=test_df, type="lp"))
  pred_sec <- proc.time()[["elapsed"]] - pred_t0

  cidx <- as.numeric(concordance(Surv(time, event) ~ lp, data=test_df, reverse=TRUE)$concordance)
  out <- list(
    status="ok",
    fit_sec=fit_sec,
    predict_sec=pred_sec,
    auc=cidx,
    brier=NULL,
    rmse=NULL,
    r2=NULL,
    model_spec=ftxt
  )
  write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
  quit(save="no")
}

  if (family_name == "gaussian") {
  if (scenario_name == "lidar_semipar") {
    ftxt <- "y ~ s(range, bs='ps', k=min(25, nrow(train_df)-1))"
  } else if (scenario_name == "us48_demand_5day" || scenario_name == "us48_demand_31day") {
    ftxt <- "y ~ s(hour, bs='ps', k=min(10, nrow(train_df)-1)) + demand_forecast + net_generation + total_interchange"
  } else if (scenario_name == "wine_gamair") {
    ftxt <- "y ~ year + h_rain + w_rain + h_temp + s(s_temp, bs='ps', k=min(10, nrow(train_df)-1))"
  } else if (scenario_name == "wine_temp_vs_year") {
    ftxt <- "y ~ s(year, bs='ps', k=min(10, nrow(train_df)-1))"
  } else if (scenario_name == "wine_price_vs_temp") {
    ftxt <- "y ~ s(temp, bs='ps', k=min(10, nrow(train_df)-1))"
  } else if (scenario_name == "icu_survival_los") {
    ftxt <- "y ~ bmi + hr_max + sysbp_min + temp_apache + s(age, bs='ps', k=min(10, nrow(train_df)-1))"
  } else {
    ftxt <- "y ~ s(range, bs='ps', k=min(25, nrow(train_df)-1))"
  }
} else if (scenario_name == "bone_gamair") {
  ftxt <- "y ~ trt_auto + s(t, bs='ps', k=min(8, nrow(train_df)-1))"
} else if (scenario_name == "prostate_gamair") {
  ftxt <- "y ~ pc1 + s(pc2, bs='ps', k=min(8, nrow(train_df)-1))"
} else if (scenario_name == "icu_survival_death") {
  ftxt <- "y ~ age + bmi + hr_max + sysbp_min + s(los_days, bs='ps', k=min(10, nrow(train_df)-1))"
} else if (scenario_name == "horse_colic") {
  ftxt <- "y ~ rectal_temp + s(pulse, bs='ps', k=min(8, nrow(train_df)-1)) + packed_cell_volume"
} else if (scenario_name == "haberman_survival") {
  ftxt <- "y ~ age + op_year + s(axil_nodes, bs='ps', k=min(8, nrow(train_df)-1))"
} else if (scenario_name == "geo_disease_tp" || scenario_name == "geo_disease_tp_basic") {
  ftxt <- paste(
    "y ~",
    "s(pc1, bs='tp', k=min(12, nrow(train_df)-1)) +",
    "s(pc2, bs='tp', k=min(12, nrow(train_df)-1)) +",
    "s(pc3, bs='tp', k=min(12, nrow(train_df)-1)) +",
    "s(pc4, bs='tp', k=min(12, nrow(train_df)-1)) +",
    paste(sprintf("pc%d", 5:16), collapse = " + ")
  )
} else if (scenario_name == "geo_disease_duchon" || scenario_name == "geo_disease_duchon_basic") {
  ftxt <- paste(
    "y ~",
    "s(pc1, bs='ds', k=min(12, nrow(train_df)-1)) +",
    "s(pc2, bs='ds', k=min(12, nrow(train_df)-1)) +",
    "s(pc3, bs='ds', k=min(12, nrow(train_df)-1)) +",
    "s(pc4, bs='ds', k=min(12, nrow(train_df)-1)) +",
    paste(sprintf("pc%d", 5:16), collapse = " + ")
  )
} else if (scenario_name == "geo_disease_shrinkage" || scenario_name == "geo_disease_matern" || scenario_name == "geo_disease_matern_basic") {
  ftxt <- paste(
    "y ~",
    "s(pc1, bs='ts', k=min(12, nrow(train_df)-1)) +",
    "s(pc2, bs='ts', k=min(12, nrow(train_df)-1)) +",
    "s(pc3, bs='ts', k=min(12, nrow(train_df)-1)) +",
    "s(pc4, bs='ts', k=min(12, nrow(train_df)-1)) +",
    paste(sprintf("pc%d", 5:16), collapse = " + ")
  )
} else if (scenario_name == "geo_disease_ps_per_pc" || scenario_name == "geo_disease_ps_per_pc_basic") {
  ftxt <- paste("y ~", paste(sprintf("s(pc%d, bs='ps', k=min(10, nrow(train_df)-1))", 1:16), collapse = " + "))
} else if (grepl("^geo_disease_eas3?_", scenario_name)) {
  match <- regmatches(scenario_name, regexec("^geo_disease_(eas|eas3)_(tp|duchon|matern|psperpc)_k([0-9]+)$", scenario_name))[[1]]
  if (length(match) != 4) stop(sprintf("Invalid geo_disease_eas/eas3 scenario name: %s", scenario_name))
  family_code <- match[2]
  basis_code <- match[3]
  k_raw <- as.integer(match[4])
  n_pcs <- if (family_code == "eas3") 3 else 16
  if (!is.finite(k_raw) || k_raw < 4) k_raw <- 4
  if (basis_code == "psperpc") {
    ftxt <- paste("y ~", paste(sprintf("s(pc%d, bs='ps', k=min(%d, nrow(train_df)-1))", 1:n_pcs, k_raw), collapse = " + "))
  } else {
    bs_code <- if (basis_code == "tp") "tp" else if (basis_code == "duchon") "ds" else "ts"
    smooth_terms <- paste(sprintf("s(pc%d, bs='%s', k=min(%d, nrow(train_df)-1))", 1:3, bs_code, k_raw), collapse = " + ")
    linear_terms <- if (n_pcs > 3) paste(sprintf("pc%d", 4:n_pcs), collapse = " + ") else ""
    rhs <- if (nchar(linear_terms) > 0) paste(smooth_terms, linear_terms, sep = " + ") else smooth_terms
    ftxt <- paste(
      "y ~", rhs
    )
  }
} else if (grepl("^papuan_oce4?_", scenario_name)) {
  match <- regmatches(scenario_name, regexec("^papuan_oce(4)?_(tp|duchon|matern|psperpc)(_basic)?_k([0-9]+)$", scenario_name))[[1]]
  if (length(match) != 5) stop(sprintf("Invalid papuan_oce scenario name: %s", scenario_name))
  family_code <- match[2]
  basis_code <- match[3]
  k_raw <- as.integer(match[5])
  n_pcs <- if (family_code == "4") 4 else 16
  if (!is.finite(k_raw) || k_raw < 4) k_raw <- 4
  if (basis_code == "psperpc") {
    ftxt <- paste("y ~", paste(sprintf("s(pc%d, bs='ps', k=min(%d, nrow(train_df)-1))", 1:n_pcs, k_raw), collapse = " + "))
  } else {
    bs_code <- if (basis_code == "tp") "tp" else if (basis_code == "duchon") "ds" else "ts"
    smooth_hi <- min(3, n_pcs)
    smooth_terms <- paste(sprintf("s(pc%d, bs='%s', k=min(%d, nrow(train_df)-1))", 1:smooth_hi, bs_code, k_raw), collapse = " + ")
    linear_terms <- if (n_pcs > smooth_hi) paste(sprintf("pc%d", (smooth_hi + 1):n_pcs), collapse = " + ") else ""
    rhs <- if (nchar(linear_terms) > 0) paste(smooth_terms, linear_terms, sep = " + ") else smooth_terms
    ftxt <- paste("y ~", rhs)
  }
} else {
  ftxt <- "y ~ x1 + s(x2, bs='ps', k=min(8, nrow(train_df)-1))"
}

t0 <- proc.time()[["elapsed"]]
use_select <- !grepl("_basic$", scenario_name)
fit <- gam(as.formula(ftxt), family=fam, data=train_df, method="REML", select=use_select)
fit_sec <- proc.time()[["elapsed"]] - t0

pred_t0 <- proc.time()[["elapsed"]]
p <- as.numeric(predict(fit, newdata=test_df, type="response"))
pred_sec <- proc.time()[["elapsed"]] - pred_t0

if (family_name == "binomial") {
  ord <- order(p)
  yy <- y_test[ord]
  n_pos <- sum(yy > 0.5)
  n_neg <- sum(yy <= 0.5)
  if (n_pos == 0 || n_neg == 0) {
    auc <- 0.5
  } else {
    ranks <- seq_along(yy)
    rank_sum_pos <- sum(ranks[yy > 0.5])
    auc <- (rank_sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
  }
  brier <- mean((y_test - p)^2)
  out <- list(
    status="ok",
    fit_sec=fit_sec,
    predict_sec=pred_sec,
    auc=auc,
    brier=brier,
    rmse=NULL,
    r2=NULL,
    model_spec=ftxt
  )
} else {
  rmse <- sqrt(mean((y_test - p)^2))
  sst <- sum((y_test - mean(y_test))^2)
  if (sst <= 0) {
    r2 <- 0.0
  } else {
    r2 <- 1.0 - sum((y_test - p)^2) / sst
  }
  out <- list(
    status="ok",
    fit_sec=fit_sec,
    predict_sec=pred_sec,
    auc=NULL,
    brier=NULL,
    rmse=rmse,
    r2=r2,
    model_spec=ftxt
  )
}

write(toJSON(out, auto_unbox=TRUE, null="null"), file=out_path)
'''
        script_path.write_text(script)

        cv_rows = []
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    "Rscript",
                    str(script_path),
                    str(data_path),
                    str(out_path),
                    str(train_idx_path),
                    str(test_idx_path),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": "r_mgcv",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "error": err.strip() or out.strip(),
                }
            fold_row = json.loads(out_path.read_text())
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": "r_mgcv",
        "scenario_name": scenario["name"],
        "status": "ok",
        **metrics,
        "model_spec": f"{cv_rows[0]['model_spec']} [5-fold CV]",
    }


def run_external_pygam_cv(scenario):
    ds = dataset_for_scenario(scenario)
    folds = folds_for_dataset(ds)
    df = pd.DataFrame(ds["rows"])
    x = df[ds["features"]].to_numpy(dtype=float) if ds["features"] else np.empty((len(df), 0))
    y = df[ds["target"]].to_numpy(dtype=float) if ds.get("target") else None

    cv_rows = []
    for fold in folds:
        x_train = x[fold.train_idx]
        x_test = x[fold.test_idx]
        y_train = y[fold.train_idx] if y is not None else None
        y_test = y[fold.test_idx] if y is not None else None

        if ds["family"] == "survival":
            time_col = ds["time_col"]
            event_col = ds["event_col"]
            train_df = df.iloc[fold.train_idx].copy()
            test_df = df.iloc[fold.test_idx].copy()
            feature_cols = ds["features"]
            for col in feature_cols:
                mu = float(train_df[col].mean())
                sdv = float(train_df[col].std())
                if (not np.isfinite(sdv)) or sdv < 1e-8:
                    sdv = 1.0
                train_df[col] = (train_df[col] - mu) / sdv
                test_df[col] = (test_df[col] - mu) / sdv
            fit_feature_cols = feature_cols
            if scenario["name"] == "icu_survival_death" and "bmi" in feature_cols:
                train_df, test_df, bmi_spline_cols = _augment_bmi_spline_linear_hinges(
                    train_df, test_df, n_knots=6
                )
                fit_feature_cols = [c for c in feature_cols if c != "bmi"] + bmi_spline_cols
            fit_start = datetime.now(timezone.utc)
            cph = CoxPHFitter(penalizer=1e-4)
            cph.fit(train_df[[*fit_feature_cols, time_col, event_col]], duration_col=time_col, event_col=event_col)
            fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()

            pred_start = datetime.now(timezone.utc)
            _risk = cph.predict_partial_hazard(test_df[fit_feature_cols]).to_numpy(dtype=float).reshape(-1)
            pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
            cidx = float(
                cph.score(
                    test_df[[*fit_feature_cols, time_col, event_col]],
                    scoring_method="concordance_index",
                )
            )
            cv_rows.append(
                {
                    "fit_sec": fit_sec,
                    "predict_sec": pred_sec,
                    "auc": cidx,
                    "n_test": int(len(fold.test_idx)),
                    "model_spec": (
                        "CoxPHFitter(train-fold z-score; bmi 6-knot hinge spline; penalizer=1e-4)"
                        if scenario["name"] == "icu_survival_death"
                        else "CoxPHFitter(linear terms; train-fold z-score; penalizer=1e-4)"
                    ),
                }
            )
            continue

        if ds["family"] == "binomial":
            if x.shape[1] == 1:
                model = LogisticGAM(s(0, n_splines=8))
                model_spec = "LogisticGAM(s(0, n_splines=8))"
            elif scenario["name"] in {
                "geo_disease_tp",
                "geo_disease_tp_basic",
                "geo_disease_duchon",
                "geo_disease_duchon_basic",
                "geo_disease_shrinkage",
                "geo_disease_matern",
                "geo_disease_matern_basic",
            }:
                terms = s(0, n_splines=12) + s(1, n_splines=12) + s(2, n_splines=12)
                for j in range(3, x.shape[1]):
                    terms = terms + l(j)
                model = LogisticGAM(terms)
                model_spec = "LogisticGAM(s(0)+s(1)+s(2)+linear(3:15), n_splines=12)"
            elif scenario["name"] in {"geo_disease_ps_per_pc", "geo_disease_ps_per_pc_basic"}:
                terms = s(0, n_splines=10)
                for j in range(1, x.shape[1]):
                    terms = terms + s(j, n_splines=10)
                model = LogisticGAM(terms)
                model_spec = "LogisticGAM(sum_j s(j, n_splines=10), j=0..15)"
            elif _geo_disease_eas_scenario_cfg(scenario["name"]) is not None:
                geo_cfg = _geo_disease_eas_scenario_cfg(scenario["name"])
                k = int(geo_cfg["knots"])
                n_pcs = int(geo_cfg["n_pcs"])
                if geo_cfg["basis_code"] == "psperpc":
                    terms = s(0, n_splines=k)
                    for j in range(1, x.shape[1]):
                        terms = terms + s(j, n_splines=k)
                    model = LogisticGAM(terms)
                    model_spec = f"LogisticGAM(sum_j s(j, n_splines={k}), j=0..{n_pcs - 1})"
                else:
                    terms = s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k)
                    for j in range(3, x.shape[1]):
                        terms = terms + l(j)
                    model = LogisticGAM(terms)
                    linear_hi = n_pcs - 1
                    linear_part = f"+linear(3:{linear_hi})" if linear_hi >= 3 else ""
                    model_spec = (
                        f"LogisticGAM(s(0)+s(1)+s(2){linear_part}, "
                        f"n_splines={k}, basis={geo_cfg['basis_code']})"
                    )
            elif _papuan_oce_scenario_cfg(scenario["name"]) is not None:
                papuan_cfg = _papuan_oce_scenario_cfg(scenario["name"])
                k = int(papuan_cfg["knots"])
                n_pcs = int(papuan_cfg["n_pcs"])
                if papuan_cfg["basis_code"] == "psperpc":
                    terms = s(0, n_splines=k)
                    for j in range(1, x.shape[1]):
                        terms = terms + s(j, n_splines=k)
                    model = LogisticGAM(terms)
                    model_spec = f"LogisticGAM(sum_j s(j, n_splines={k}), j=0..{n_pcs - 1}) [papuan_oce]"
                else:
                    terms = s(0, n_splines=k) + s(1, n_splines=k) + s(2, n_splines=k)
                    for j in range(3, x.shape[1]):
                        terms = terms + l(j)
                    model = LogisticGAM(terms)
                    linear_hi = n_pcs - 1
                    linear_part = f"+linear(3:{linear_hi})" if linear_hi >= 3 else ""
                    model_spec = (
                        f"LogisticGAM(s(0)+s(1)+s(2){linear_part}, "
                        f"n_splines={k}, basis={papuan_cfg['basis_code']}) [papuan_oce]"
                    )
            elif scenario["name"] == "icu_survival_death":
                model = LogisticGAM(l(0) + l(1) + l(2) + l(3) + s(4, n_splines=10))
                model_spec = "LogisticGAM(l(0) + l(1) + l(2) + l(3) + s(4, n_splines=10))"
            elif scenario["name"] == "horse_colic":
                model = LogisticGAM(l(0) + s(1, n_splines=8) + l(2))
                model_spec = "LogisticGAM(l(0) + s(1, n_splines=8) + l(2))"
            elif scenario["name"] == "haberman_survival":
                model = LogisticGAM(l(0) + l(1) + s(2, n_splines=8))
                model_spec = "LogisticGAM(l(0) + l(1) + s(2, n_splines=8))"
            else:
                model = LogisticGAM(l(0) + s(1, n_splines=8))
                model_spec = "LogisticGAM(l(0) + s(1, n_splines=8))"
            fit_start = datetime.now(timezone.utc)
            lam_grid = np.logspace(-4, 4, 17)
            model.gridsearch(x_train, y_train, lam=lam_grid, objective="AICc", progress=False)
            fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()

            pred_start = datetime.now(timezone.utc)
            pred = model.predict_mu(x_test).astype(float)
            pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
            cv_rows.append(
                {
                    "fit_sec": fit_sec,
                    "predict_sec": pred_sec,
                    "auc": auc_score(y_test, pred),
                    "brier": brier_score(y_test, pred),
                    "n_test": int(len(fold.test_idx)),
                    "model_spec": model_spec,
                }
            )
            continue

        if scenario["name"] == "wine_gamair":
            model = LinearGAM(l(0) + l(1) + l(2) + l(3) + s(4, n_splines=10))
            model_spec = "LinearGAM(l(0) + l(1) + l(2) + l(3) + s(4, n_splines=10))"
        elif scenario["name"] in {"us48_demand_5day", "us48_demand_31day"}:
            model = LinearGAM(s(0, n_splines=10) + l(1) + l(2) + l(3))
            model_spec = "LinearGAM(s(0, n_splines=10) + l(1) + l(2) + l(3))"
        elif scenario["name"] == "icu_survival_los":
            model = LinearGAM(s(0, n_splines=10) + l(1) + l(2) + l(3) + l(4))
            model_spec = "LinearGAM(s(0, n_splines=10) + l(1) + l(2) + l(3) + l(4))"
        elif scenario["name"] in {"wine_temp_vs_year", "wine_price_vs_temp"}:
            model = LinearGAM(s(0, n_splines=10))
            model_spec = "LinearGAM(s(0, n_splines=10))"
        else:
            model = LinearGAM(s(0, n_splines=25))
            model_spec = "LinearGAM(s(0, n_splines=25))"

        fit_start = datetime.now(timezone.utc)
        lam_grid = np.logspace(-4, 4, 17)
        model.gridsearch(x_train, y_train, lam=lam_grid, objective="AICc", progress=False)
        fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()

        pred_start = datetime.now(timezone.utc)
        pred = model.predict(x_test).astype(float)
        pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()

        cv_rows.append(
            {
                "fit_sec": fit_sec,
                "predict_sec": pred_sec,
                "n_test": int(len(fold.test_idx)),
                "model_spec": model_spec,
            }
        )

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    return {
        "contender": "python_lifelines" if ds["family"] == "survival" else "python_pygam",
        "scenario_name": scenario["name"],
        "status": "ok",
        **metrics,
        "model_spec": (
            f"{cv_rows[0]['model_spec']} [5-fold CV]"
            if ds["family"] == "survival"
            else f"{cv_rows[0]['model_spec']} [lam by UBRE/GCV; 5-fold CV]"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Run GAM benchmark suite with leakage-safe 5-fold CV.")
    parser.add_argument("--scenarios", type=Path, default=DEFAULT_SCENARIOS)
    parser.add_argument("--out", type=Path, default=BENCH_DIR / "results.json")
    parser.add_argument(
        "--scenario-name",
        action="append",
        dest="scenario_names",
        help="Run only the named scenario(s). Can be passed multiple times.",
    )
    args = parser.parse_args()

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

    results = []
    for s_cfg in scenarios:
        results.append(run_rust_scenario_cv(s_cfg))
        results.append(run_external_mgcv_cv(s_cfg))
        results.append(run_external_pygam_cv(s_cfg))

    # Hard guard: benchmark runs must fail fast in CI if any contender fails.
    failed = [r for r in results if r.get("status") != "ok"]
    if failed:
        msgs = []
        for r in failed:
            msgs.append(
                f"{r.get('contender','?')} / {r.get('scenario_name','?')}: {r.get('error','unknown error')}"
            )
        raise SystemExit("benchmark run failed:\n" + "\n".join(msgs))

    # Hard guard: benchmark outputs must remain CV-based and leakage-safe.
    for r in results:
        if r.get("status") != "ok":
            continue
        spec = str(r.get("model_spec", ""))
        if "CV" not in spec and "cv" not in spec:
            raise SystemExit(
                f"non-CV model result detected for {r.get('contender')} / {r.get('scenario_name')}: {spec}"
            )

    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cv": {
            "n_splits": CV_SPLITS,
            "seed": CV_SEED,
            "leakage_safe": True,
        },
        "results": results,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
