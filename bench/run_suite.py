#!/usr/bin/env python3
import argparse
import csv
import json
import math
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from pygam import LinearGAM, LogisticGAM, l, s

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIR = ROOT / "benchmarks"
DEFAULT_SCENARIOS = BENCH_DIR / "scenarios.json"
DATASET_DIR = BENCH_DIR / "datasets"
CV_SPLITS = 5
CV_SEED = 42


@dataclass(frozen=True)
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray


def run_cmd(cmd, cwd=None):
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


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


def dataset_for_scenario(s):
    name = s["name"]
    if name in {"small_dense", "medium", "pathological_ill_conditioned"}:
        return _synthetic_binomial_dataset(s["n"], s["p"], s.get("seed", 42))
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
            "c_index": None,
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
        "c_index": None,
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
    }.get(scenario_name)
    if cfg is None:
        raise RuntimeError(f"No Rust CLI CV mapping configured for scenario '{scenario_name}'")

    return [
        "--family",
        cfg["family"],
        "--target-col",
        ds["target"],
        "--smooth-col",
        cfg["smooth_col"],
        "--linear-cols",
        ",".join(cfg["linear_cols"]),
        "--num-internal-knots",
        str(int(cfg["knots"])),
        "--double-penalty",
        "true" if cfg["double_penalty"] else "false",
    ]


def run_rust_scenario_cv(scenario):
    ds = dataset_for_scenario(scenario)
    folds = folds_for_dataset(ds)

    build_code, build_out, build_err = run_cmd(
        ["cargo", "build", "--release", "--bin", "cli"], cwd=ROOT
    )
    if build_code != 0:
        return {
            "contender": "rust_gam",
            "scenario_name": scenario["name"],
            "status": "failed",
            "error": build_err.strip() or build_out.strip(),
        }

    bin_path = ROOT / "target" / "release" / "cli"
    with tempfile.TemporaryDirectory(prefix="gam_bench_rust_cv_") as td:
        td_path = Path(td)
        data_csv = td_path / "data.csv"
        write_dataset_csv(ds, data_csv)

        cv_rows = []
        for fold_id, fold in enumerate(folds):
            train_idx_path = td_path / f"train_idx_{fold_id}.txt"
            test_idx_path = td_path / f"test_idx_{fold_id}.txt"
            train_idx_path.write_text("\n".join(str(int(i)) for i in fold.train_idx) + "\n")
            test_idx_path.write_text("\n".join(str(int(i)) for i in fold.test_idx) + "\n")

            code, out, err = run_cmd(
                [
                    str(bin_path),
                    "--data",
                    str(data_csv),
                    "--train-idx",
                    str(train_idx_path),
                    "--test-idx",
                    str(test_idx_path),
                    *rust_cli_cv_args(scenario["name"], ds),
                ],
                cwd=ROOT,
            )
            if code != 0:
                return {
                    "contender": "rust_gam",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "error": err.strip() or out.strip(),
                }
            line = out.strip().splitlines()[-1]
            fold_row = json.loads(line)
            if fold_row.get("status") != "ok":
                return {
                    "contender": "rust_gam",
                    "scenario_name": scenario["name"],
                    "status": "failed",
                    "error": json.dumps(fold_row),
                }
            fold_row["n_test"] = int(len(fold.test_idx))
            cv_rows.append(fold_row)

    metrics = aggregate_cv_rows(cv_rows, ds["family"])
    out = {
        "contender": "rust_gam",
        "scenario_name": scenario["name"],
        "status": "ok",
        **metrics,
        "model_spec": "5-fold CV (train-only spline fit; holdout scoring)",
    }
    if ds["family"] == "survival":
        out["model_spec"] = "5-fold CV (Rust native Royston-Parmar survival)"
    return out


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
    ftxt <- "Surv(time, event) ~ age + bmi + hr_max + sysbp_min"
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
} else {
  ftxt <- "y ~ x1 + s(x2, bs='ps', k=min(8, nrow(train_df)-1))"
}

t0 <- proc.time()[["elapsed"]]
fit <- gam(as.formula(ftxt), family=fam, data=train_df, method="REML")
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
            fit_start = datetime.now(timezone.utc)
            cph = CoxPHFitter(penalizer=1e-4)
            cph.fit(train_df[[*ds["features"], time_col, event_col]], duration_col=time_col, event_col=event_col)
            fit_sec = (datetime.now(timezone.utc) - fit_start).total_seconds()

            pred_start = datetime.now(timezone.utc)
            _risk = cph.predict_partial_hazard(test_df[ds["features"]]).to_numpy(dtype=float).reshape(-1)
            pred_sec = (datetime.now(timezone.utc) - pred_start).total_seconds()
            cidx = float(
                cph.score(
                    test_df[[*ds["features"], time_col, event_col]],
                    scoring_method="concordance_index",
                )
            )
            cv_rows.append(
                {
                    "fit_sec": fit_sec,
                    "predict_sec": pred_sec,
                    "auc": cidx,
                    "n_test": int(len(fold.test_idx)),
                    "model_spec": "CoxPHFitter(linear terms; train-fold z-score; penalizer=1e-4)",
                }
            )
            continue

        if ds["family"] == "binomial":
            if x.shape[1] == 1:
                model = LogisticGAM(s(0, n_splines=8))
                model_spec = "LogisticGAM(s(0, n_splines=8))"
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
                "rmse": rmse_score(y_test, pred),
                "r2": r2_score(y_test, pred),
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
            else f"{cv_rows[0]['model_spec']} [lam by AICc; 5-fold CV]"
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
