"""Strict synthetic oracle+noise PGS pipeline for MSI.

Scope:
  * Demographies: serial1d and grid2d only, custom msprime models.
  * PGS: oracle+noise only.
  * Methods: z-norm, linear-PC, gamfit.
  * Outcomes: binary and survival.
  * Phenotype modes: deme-varying baseline and constant baseline.

All discrimination metrics are pooled over the held-out test split. Calibration
metrics are computed overall and by distance/deme strata. No within-stratum AUC
or C-index is computed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import msprime
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

import gamfit


NPER = 36
SEQLEN = 8_000_000
RECOMB = 1e-8
MU = 1.25e-8
PREV = 0.18
HORIZON = 12.0

DEMOGRAPHIES = ("serial1d", "grid2d")
PHENO_MODES = ("varying_baseline", "constant_baseline")
OUTCOMES = ("binary", "survival")
METHODS = ("z-norm", "linear-PC", "gamfit")
RHO_REGIMES = {
    "mild_decay": (0.82, 0.07),
    "moderate_decay": (0.82, 0.13),
    "steep_decay": (0.82, 0.21),
}


class TimeoutError(RuntimeError):
    pass


@contextmanager
def wall_timeout(seconds: int):
    def handler(_signum, _frame):
        raise TimeoutError(f"timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


@contextmanager
def quiet_native_output():
    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)
    with open(os.devnull, "w") as sink:
        os.dup2(sink.fileno(), 1)
        os.dup2(sink.fileno(), 2)
        try:
            yield
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


@dataclass(frozen=True)
class DemographySpec:
    demography: msprime.Demography
    samples: dict[str, int]
    deme: np.ndarray
    distance: np.ndarray
    train_deme: int
    n_pc: int


def dem_serial1d(demes: int = 10, n: int = 700, nanc: int = 3000, migration: float = 8e-4) -> DemographySpec:
    dem = msprime.Demography()
    for k in range(demes):
        dem.add_population(name=f"d{k}", initial_size=n)
    dem.add_population(name="ANC", initial_size=nanc)
    for k in range(demes - 1):
        dem.set_migration_rate(f"d{k}", f"d{k + 1}", migration)
        dem.set_migration_rate(f"d{k + 1}", f"d{k}", migration)
    for k in range(demes - 1, 0, -1):
        t = 90 + (demes - 1 - k) * 120
        dem.add_migration_rate_change(time=t, rate=0, source=f"d{k - 1}", dest=f"d{k}")
        dem.add_migration_rate_change(time=t, rate=0, source=f"d{k}", dest=f"d{k - 1}")
        dem.add_mass_migration(time=t + 1, source=f"d{k}", dest=f"d{k - 1}", proportion=1.0)
    dem.add_population_split(time=90 + demes * 120 + 500, derived=["d0"], ancestral="ANC")
    dem.sort_events()
    deme = np.repeat(np.arange(demes), NPER)
    train_deme = 0
    distance = np.abs(deme - train_deme).astype(float)
    return DemographySpec(dem, {f"d{k}": NPER for k in range(demes)}, deme, distance, train_deme, 2)


def dem_grid2d(side: int = 6, n: int = 700, nanc: int = 3000, migration: float = 8e-4) -> DemographySpec:
    dem = msprime.Demography()

    def name(r: int, c: int) -> str:
        return f"d_{r}_{c}"

    for r in range(side):
        for c in range(side):
            dem.add_population(name=name(r, c), initial_size=n)
    dem.add_population(name="ANC", initial_size=nanc)
    for r in range(side):
        for c in range(side):
            for dr, dc in ((1, 0), (0, 1)):
                rr, cc = r + dr, c + dc
                if rr < side and cc < side:
                    dem.set_migration_rate(name(r, c), name(rr, cc), migration)
                    dem.set_migration_rate(name(rr, cc), name(r, c), migration)
    dem.add_population_split(
        time=6 * n,
        derived=[name(r, c) for r in range(side) for c in range(side)],
        ancestral="ANC",
    )
    dem.sort_events()
    coords = [(r, c) for r in range(side) for c in range(side)]
    deme = np.repeat(np.arange(side * side), NPER)
    distance_by_deme = np.array([r + c for r, c in coords], dtype=float)
    distance = np.repeat(distance_by_deme, NPER)
    return DemographySpec(
        dem,
        {name(r, c): NPER for r in range(side) for c in range(side)},
        deme,
        distance,
        0,
        3,
    )


def simulate_core(dem_name: str, rho_regime: str, seed: int) -> tuple[pd.DataFrame, dict]:
    if dem_name == "serial1d":
        spec = dem_serial1d()
    elif dem_name == "grid2d":
        spec = dem_grid2d()
    else:
        raise ValueError(f"unsupported demography: {dem_name}")

    rng = np.random.default_rng(10_000 + seed * 37)
    ts = msprime.sim_ancestry(
        samples=spec.samples,
        demography=spec.demography,
        sequence_length=SEQLEN,
        recombination_rate=RECOMB,
        ploidy=2,
        random_seed=101 + seed * 11,
    )
    ts = msprime.sim_mutations(ts, rate=MU, random_seed=102 + seed * 11)
    genotypes = ts.genotype_matrix()
    dosage = (genotypes[:, 0::2] + genotypes[:, 1::2]).T.astype(np.float32)
    af = dosage.mean(axis=0) / 2.0
    maf = np.minimum(af, 1 - af)
    common = dosage[:, maf >= 0.03]
    if common.shape[1] < 20:
        raise RuntimeError(f"too few common variants: {common.shape[1]}")

    pcs_raw = PCA(
        min(6, common.shape[1]),
        svd_solver="randomized",
        random_state=seed,
    ).fit_transform(StandardScaler().fit_transform(common))
    pcs = StandardScaler().fit_transform(pcs_raw)[:, : spec.n_pc]

    n_causal = min(900, common.shape[1])
    causal = rng.choice(common.shape[1], n_causal, replace=False)
    effects = rng.normal(0, 1, n_causal) * (rng.random(n_causal) < 0.35)
    genetic = StandardScaler(with_std=False).fit_transform(common[:, causal]) @ effects
    genetic = (genetic - genetic.mean()) / (genetic.std() + 1e-12)

    rho0, decay = RHO_REGIMES[rho_regime]
    rho = np.clip(rho0 * np.exp(-decay * spec.distance), 0, 0.999)
    pgs_raw = rho * genetic + np.sqrt(1 - rho**2) * rng.normal(0, 1, len(genetic))

    source = np.where(spec.deme == spec.train_deme)[0]
    rng.shuffle(source)
    source_fit = source[: max(2, int(0.6 * len(source)))]
    source_test = source[max(2, int(0.6 * len(source))) :]
    non_source = np.where(spec.deme != spec.train_deme)[0]
    rng.shuffle(non_source)
    non_source_fit = non_source[: len(non_source) // 2]
    non_source_test = non_source[len(non_source) // 2 :]
    split = np.full(len(genetic), "unused", dtype=object)
    split[np.r_[source_fit, non_source_fit]] = "fit"
    split[np.r_[source_test, non_source_test]] = "test"
    panel = np.where(spec.deme == spec.train_deme, "training_ancestry", "other_ancestry")

    pgs_mean = pgs_raw[source_fit].mean()
    pgs_sd = pgs_raw[source_fit].std() + 1e-12
    df = pd.DataFrame(
        {
            "iid": np.arange(len(genetic)),
            "demography": dem_name,
            "rho_regime": rho_regime,
            "seed": seed,
            "deme": spec.deme.astype(int),
            "dist_from_train": spec.distance,
            "panel": panel,
            "split": split,
            "true_liability": genetic,
            "oracle_rho": rho,
            "PGS_raw": pgs_raw,
            "PGS_z": (pgs_raw - pgs_mean) / pgs_sd,
        }
    )
    for i in range(spec.n_pc):
        df[f"PC{i + 1}"] = pcs[:, i]

    sidecar = {
        "demography": dem_name,
        "rho_regime": rho_regime,
        "rho0": rho0,
        "rho_decay": decay,
        "seed": seed,
        "n": int(len(df)),
        "n_common_variants": int(common.shape[1]),
        "n_pc": int(spec.n_pc),
        "train_deme": int(spec.train_deme),
    }
    return df, sidecar


def add_phenotypes(df: pd.DataFrame, pheno_mode: str, seed: int) -> pd.DataFrame:
    out = df.copy()
    rng = np.random.default_rng(20_000 + seed * 41 + (0 if pheno_mode == "varying_baseline" else 1000))
    dist = out["dist_from_train"].to_numpy(float)
    scaled = (dist - dist.min()) / (dist.max() - dist.min() + 1e-12)
    if pheno_mode == "varying_baseline":
        baseline = 0.55 * np.sin(2 * np.pi * scaled) + 0.25 * (scaled - 0.5)
        baseline -= baseline.mean()
    elif pheno_mode == "constant_baseline":
        baseline = np.zeros(len(out))
    else:
        raise ValueError(f"unsupported phenotype mode: {pheno_mode}")

    eta = out["true_liability"].to_numpy(float) + baseline
    threshold = brentq(lambda t: (eta > t).mean() - PREV, eta.min() - 5, eta.max() + 5)
    y_binary = (eta > threshold).astype(int)

    hazard = 0.08 * np.exp(eta - eta.mean())
    event_time = rng.exponential(1.0 / np.clip(hazard, 1e-6, None))
    admin = np.quantile(event_time, 0.68)
    censor_time = np.minimum(rng.exponential(admin * 1.4, len(out)), admin)
    surv_time = np.minimum(event_time, censor_time)
    surv_event = (event_time <= censor_time).astype(float)
    entry_time = surv_time * rng.uniform(0.0, 0.35, len(out))
    entry_time = np.minimum(entry_time, surv_time - 1e-4)
    entry_time = np.clip(entry_time, 0.0, None)

    out["pheno_mode"] = pheno_mode
    out["baseline_shift"] = baseline
    out["y_binary"] = y_binary
    out["entry_time"] = entry_time
    out["surv_time"] = surv_time
    out["surv_event"] = surv_event
    return out


def pc_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("PC")]


def z_norm_score(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    pcs = pc_columns(train)
    mean_model = LinearRegression().fit(train[pcs], train["PGS_z"])
    train_res = train["PGS_z"].to_numpy(float) - mean_model.predict(train[pcs])
    all_res_test = test["PGS_z"].to_numpy(float) - mean_model.predict(test[pcs])
    var_model = LinearRegression().fit(train[pcs], np.log(train_res**2 + 1e-6))
    train_sd = np.sqrt(np.exp(var_model.predict(train[pcs])))
    test_sd = np.sqrt(np.exp(var_model.predict(test[pcs])))
    return train_res / np.clip(train_sd, 1e-6, None), all_res_test / np.clip(test_sd, 1e-6, None)


def fit_binary_methods(df: pd.DataFrame, centers: int, timeout: int) -> tuple[list[dict], list[dict], list[dict]]:
    train = df[df["split"] == "fit"].copy()
    test = df[df["split"] == "test"].copy()
    y_train = train["y_binary"].to_numpy(int)
    y_test = test["y_binary"].to_numpy(int)
    pcs = pc_columns(df)
    pred_rows: list[dict] = []
    issue_rows: list[dict] = []
    predictions: dict[str, np.ndarray] = {}

    zn_train, zn_test = z_norm_score(train, test)
    predictions["z-norm"] = LogisticRegression(max_iter=4000).fit(zn_train[:, None], y_train).predict_proba(zn_test[:, None])[:, 1]

    linear_cols = ["PGS_z", *pcs]
    predictions["linear-PC"] = (
        LogisticRegression(max_iter=4000)
        .fit(train[linear_cols], y_train)
        .predict_proba(test[linear_cols])[:, 1]
    )

    pcn = ", ".join(pcs)

    def gf_frame(frame: pd.DataFrame, with_y: bool) -> pd.DataFrame:
        vals = {"prs_z": frame["PGS_z"].to_numpy(float)}
        for pc in pcs:
            vals[pc] = frame[pc].to_numpy(float)
        if with_y:
            vals["event"] = frame["y_binary"].to_numpy(float)
        return pd.DataFrame(vals)

    try:
        with wall_timeout(timeout), quiet_native_output():
            model = gamfit.fit(
                gf_frame(train, True),
                f"event ~ matern({pcn}, centers={centers})",
                family="bernoulli-marginal-slope",
                link="probit",
                z_column="prs_z",
                logslope_formula=f"matern({pcn}, centers={centers})",
            )
            pred = model.predict(gf_frame(test, False))
        predictions["gamfit"] = pred["mean"].to_numpy(float) if hasattr(pred, "columns") else np.asarray(pred).ravel()
    except Exception as exc:
        issue_rows.append(issue_record(df, "binary", "gamfit", exc))

    for method, prob in predictions.items():
        prob = np.clip(prob, 1e-6, 1 - 1e-6)
        pred_rows.extend(row_predictions(df, test, "binary", method, prob))

    metrics = binary_metrics(df, test, y_test, predictions)
    return pred_rows, metrics, issue_rows


def fit_survival_methods(df: pd.DataFrame, centers: int, timeout: int) -> tuple[list[dict], list[dict], list[dict]]:
    train = df[df["split"] == "fit"].copy()
    test = df[df["split"] == "test"].copy()
    pcs = pc_columns(df)
    risk: dict[str, np.ndarray] = {}
    pred_rows: list[dict] = []
    issue_rows: list[dict] = []

    def fit_cox(x_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        model = sm.PHReg(
            endog=train["surv_time"].to_numpy(float),
            exog=x_train,
            status=train["surv_event"].to_numpy(float),
            entry=train["entry_time"].to_numpy(float),
        )
        res = model.fit(disp=0)
        return x_test @ res.params

    try:
        zn_train, zn_test = z_norm_score(train, test)
        risk["z-norm"] = fit_cox(zn_train[:, None], zn_test[:, None])
    except Exception as exc:
        issue_rows.append(issue_record(df, "survival", "z-norm", exc))

    try:
        linear_cols = ["PGS_z", *pcs]
        risk["linear-PC"] = fit_cox(train[linear_cols].to_numpy(float), test[linear_cols].to_numpy(float))
    except Exception as exc:
        issue_rows.append(issue_record(df, "survival", "linear-PC", exc))

    pcn = ", ".join(pcs)

    def gf_frame(frame: pd.DataFrame, with_y: bool) -> pd.DataFrame:
        vals = {"prs_z": frame["PGS_z"].to_numpy(float)}
        for pc in pcs:
            vals[pc] = frame[pc].to_numpy(float)
        if with_y:
            vals["entry_time"] = frame["entry_time"].to_numpy(float)
            vals["surv_time"] = frame["surv_time"].to_numpy(float)
            vals["event"] = frame["surv_event"].to_numpy(float)
        return pd.DataFrame(vals)

    try:
        with wall_timeout(timeout), quiet_native_output():
            model = gamfit.fit(
                gf_frame(train, True),
                f"Surv(entry_time, surv_time, event) ~ matern({pcn}, centers={centers})",
                survival_likelihood="marginal-slope",
                z_column="prs_z",
                logslope_formula=f"matern({pcn}, centers={centers})",
            )
            pred = model.predict(gf_frame(test, False))
        if hasattr(pred, "columns"):
            key = next((c for c in ("linear_predictor", "eta", "lp", "log_hazard", "mean") if c in pred.columns), pred.columns[0])
            risk["gamfit"] = pred[key].to_numpy(float)
        else:
            risk["gamfit"] = np.asarray(pred).ravel()
    except Exception as exc:
        issue_rows.append(issue_record(df, "survival", "gamfit", exc))

    for method, score in risk.items():
        pred_rows.extend(row_predictions(df, test, "survival", method, score))
    metrics = survival_metrics(df, test, risk)
    return pred_rows, metrics, issue_rows


def issue_record(df: pd.DataFrame, outcome: str, method: str, exc: BaseException) -> dict:
    first = df.iloc[0]
    return {
        "demography": first["demography"],
        "rho_regime": first["rho_regime"],
        "pheno_mode": first["pheno_mode"],
        "seed": int(first["seed"]),
        "outcome": outcome,
        "method": method,
        "error_type": type(exc).__name__,
        "error": str(exc)[:500],
        "traceback": traceback.format_exc()[-2000:],
    }


def row_predictions(df: pd.DataFrame, test: pd.DataFrame, outcome: str, method: str, values: np.ndarray) -> list[dict]:
    first = df.iloc[0]
    rows = []
    for (_, row), value in zip(test.iterrows(), values):
        rows.append(
            {
                "demography": first["demography"],
                "rho_regime": first["rho_regime"],
                "pheno_mode": first["pheno_mode"],
                "seed": int(first["seed"]),
                "outcome": outcome,
                "method": method,
                "iid": int(row["iid"]),
                "deme": int(row["deme"]),
                "dist_from_train": float(row["dist_from_train"]),
                "panel": row["panel"],
                "prediction": float(value),
                "y_binary": int(row["y_binary"]),
                "entry_time": float(row["entry_time"]),
                "surv_time": float(row["surv_time"]),
                "surv_event": float(row["surv_event"]),
            }
        )
    return rows


def binary_metrics(df: pd.DataFrame, test: pd.DataFrame, y: np.ndarray, predictions: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    for method, pred in predictions.items():
        pred = np.clip(np.asarray(pred, float), 1e-6, 1 - 1e-6)
        rows.append(
            metric_record(
                df,
                "binary",
                method,
                "global",
                {
                    "auc": roc_auc_score(y, pred) if len(np.unique(y)) == 2 else np.nan,
                    "log_loss": log_loss(y, pred, labels=[0, 1]),
                    "brier": brier_score_loss(y, pred),
                    "cindex": np.nan,
                },
                int(len(y)),
                int(y.sum()),
            )
        )
    return rows


def survival_metrics(df: pd.DataFrame, test: pd.DataFrame, risk: dict[str, np.ndarray]) -> list[dict]:
    rows = []
    time_v = test["surv_time"].to_numpy(float)
    event_v = test["surv_event"].to_numpy(float)
    for method, score in risk.items():
        rows.append(
            metric_record(
                df,
                "survival",
                method,
                "global",
                {
                    "auc": np.nan,
                    "log_loss": np.nan,
                    "brier": np.nan,
                    "cindex": harrell_c(time_v, event_v, np.asarray(score, float)),
                },
                int(len(test)),
                int(event_v.sum()),
            )
        )
    return rows


def metric_record(df: pd.DataFrame, outcome: str, method: str, stratum: str, values: dict, n: int, events: int) -> dict:
    first = df.iloc[0]
    return {
        "demography": first["demography"],
        "rho_regime": first["rho_regime"],
        "pheno_mode": first["pheno_mode"],
        "seed": int(first["seed"]),
        "outcome": outcome,
        "method": method,
        "stratum": stratum,
        "n": n,
        "events": events,
        "auc": clean_float(values.get("auc")),
        "cindex": clean_float(values.get("cindex")),
        "brier": clean_float(values.get("brier")),
        "log_loss": clean_float(values.get("log_loss")),
    }


def calibration_rows(preds: pd.DataFrame) -> list[dict]:
    rows = []
    key_cols = ["demography", "rho_regime", "pheno_mode", "seed", "outcome", "method"]
    for key, group in preds.groupby(key_cols, sort=False):
        base = dict(zip(key_cols, key))
        if base["outcome"] == "binary":
            rows.extend(binary_calibration_for_group(base, group, "all", np.ones(len(group), dtype=bool)))
            rows.extend(stratified_binary_calibration(base, group))
        else:
            rows.extend(survival_calibration_for_group(base, group, "all", np.ones(len(group), dtype=bool)))
            rows.extend(stratified_survival_calibration(base, group))
    return rows


def binary_calibration_for_group(base: dict, group: pd.DataFrame, stratum: str, mask: np.ndarray) -> list[dict]:
    sub = group.loc[mask]
    if len(sub) < 20 or sub["y_binary"].nunique() < 2:
        return []
    prob = np.clip(sub["prediction"].to_numpy(float), 1e-6, 1 - 1e-6)
    y = sub["y_binary"].to_numpy(int)
    lp = np.log(prob / (1 - prob))
    try:
        fit = LogisticRegression(C=1e6, max_iter=4000).fit(lp[:, None], y)
        intercept = float(fit.intercept_[0])
        slope = float(fit.coef_[0, 0])
    except Exception:
        intercept = np.nan
        slope = np.nan
    ece, ici = ece_ici(y, prob)
    row = {**base, "stratum": stratum, "n": int(len(sub)), "events": int(y.sum())}
    row.update({"calibration_intercept": clean_float(intercept), "calibration_slope": clean_float(slope), "ece": clean_float(ece), "ici": clean_float(ici)})
    return [row]


def stratified_binary_calibration(base: dict, group: pd.DataFrame) -> list[dict]:
    rows = []
    for label, mask in distance_masks(group):
        rows.extend(binary_calibration_for_group(base, group, label, mask))
    for deme, sub_idx in group.groupby("deme").groups.items():
        mask = group.index.isin(sub_idx)
        rows.extend(binary_calibration_for_group(base, group, f"deme:{int(deme)}", mask))
    return rows


def survival_calibration_for_group(base: dict, group: pd.DataFrame, stratum: str, mask: np.ndarray) -> list[dict]:
    sub = group.loc[mask]
    if len(sub) < 25 or sub["surv_event"].sum() < 5 or sub["prediction"].std() < 1e-8:
        return []
    try:
        model = sm.PHReg(
            endog=sub["surv_time"].to_numpy(float),
            exog=(sub["prediction"].to_numpy(float) - sub["prediction"].mean()).reshape(-1, 1),
            status=sub["surv_event"].to_numpy(float),
            entry=sub["entry_time"].to_numpy(float),
        )
        fit = model.fit(disp=0)
        slope = float(fit.params[0])
    except Exception:
        slope = np.nan
    row = {**base, "stratum": stratum, "n": int(len(sub)), "events": int(sub["surv_event"].sum())}
    row.update({"calibration_intercept": "", "calibration_slope": clean_float(slope), "ece": "", "ici": ""})
    return [row]


def stratified_survival_calibration(base: dict, group: pd.DataFrame) -> list[dict]:
    rows = []
    for label, mask in distance_masks(group):
        rows.extend(survival_calibration_for_group(base, group, label, mask))
    for deme, sub_idx in group.groupby("deme").groups.items():
        mask = group.index.isin(sub_idx)
        rows.extend(survival_calibration_for_group(base, group, f"deme:{int(deme)}", mask))
    return rows


def distance_masks(group: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    dist = group["dist_from_train"].to_numpy(float)
    if len(np.unique(dist)) < 3:
        return []
    q1, q2 = np.quantile(dist, [1 / 3, 2 / 3])
    return [
        ("distance:near", dist <= q1),
        ("distance:mid", (dist > q1) & (dist <= q2)),
        ("distance:far", dist > q2),
    ]


def ece_ici(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> tuple[float, float]:
    order = np.argsort(prob)
    chunks = np.array_split(order, min(bins, len(order)))
    weighted = 0.0
    abs_errors = np.zeros(len(prob), dtype=float)
    for chunk in chunks:
        if len(chunk) == 0:
            continue
        obs = y[chunk].mean()
        exp = prob[chunk].mean()
        err = abs(obs - exp)
        weighted += len(chunk) / len(prob) * err
        abs_errors[chunk] = err
    return float(weighted), float(abs_errors.mean())


def harrell_c(time_v: np.ndarray, event_v: np.ndarray, risk_v: np.ndarray) -> float:
    order = np.argsort(time_v)
    time_v = time_v[order]
    event_v = event_v[order].astype(bool)
    risk_v = risk_v[order]
    comparable = 0
    concordant = 0.0
    for i in range(len(time_v)):
        if not event_v[i]:
            continue
        later = time_v > time_v[i]
        if not later.any():
            continue
        comparable += int(later.sum())
        later_risk = risk_v[later]
        concordant += float(np.sum(later_risk < risk_v[i]) + 0.5 * np.sum(later_risk == risk_v[i]))
    return concordant / comparable if comparable else np.nan


def clean_float(value) -> float | str:
    if value is None:
        return ""
    try:
        val = float(value)
    except Exception:
        return ""
    return round(val, 6) if math.isfinite(val) else ""


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir).resolve()
    data_dir = outdir / "data"
    results_dir = outdir / "results"
    plots_dir = outdir / "plots"
    for path in (data_dir, results_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    all_predictions: list[dict] = []
    all_metrics: list[dict] = []
    all_issues: list[dict] = []
    split_rows: list[dict] = []
    sidecars: list[dict] = []

    start = time.time()
    for dem in args.demographies:
        for regime in args.rho_regimes:
            for seed in args.seeds:
                core, sidecar = simulate_core(dem, regime, seed)
                sidecars.append(sidecar)
                for pheno_mode in args.pheno_modes:
                    dataset = add_phenotypes(core, pheno_mode, seed)
                    data_path = data_dir / f"{dem}_{regime}_{pheno_mode}_seed{seed}.csv"
                    dataset.to_csv(data_path, index=False)
                    split_rows.extend(split_count_rows(dataset, data_path))

                    pred, metrics, issues = fit_binary_methods(dataset, args.centers, args.timeout)
                    all_predictions.extend(pred)
                    all_metrics.extend(metrics)
                    all_issues.extend(issues)

                    pred, metrics, issues = fit_survival_methods(dataset, args.centers, args.timeout)
                    all_predictions.extend(pred)
                    all_metrics.extend(metrics)
                    all_issues.extend(issues)
                    print(
                        f"done dem={dem} regime={regime} pheno={pheno_mode} seed={seed} "
                        f"elapsed={time.time() - start:.1f}s",
                        flush=True,
                    )

    write_csv(results_dir / "predictions_test.csv", all_predictions)
    write_csv(results_dir / "metrics_global.csv", all_metrics)
    write_csv(results_dir / "split_counts.csv", split_rows)
    write_csv(results_dir / "gamfit_issues.csv", all_issues)
    with (results_dir / "dataset_sidecars.json").open("w") as handle:
        json.dump(sidecars, handle, indent=2)

    pred_df = pd.DataFrame(all_predictions)
    cal = calibration_rows(pred_df) if len(pred_df) else []
    write_csv(results_dir / "calibration.csv", cal)

    comparisons = comparison_rows(pd.DataFrame(all_metrics), pd.DataFrame(cal))
    write_csv(results_dir / "comparisons.csv", comparisons)
    make_plots(results_dir, plots_dir)


def split_count_rows(df: pd.DataFrame, data_path: Path) -> list[dict]:
    first = df.iloc[0]
    rows = []
    for (split, panel), sub in df.groupby(["split", "panel"]):
        rows.append(
            {
                "demography": first["demography"],
                "rho_regime": first["rho_regime"],
                "pheno_mode": first["pheno_mode"],
                "seed": int(first["seed"]),
                "split": split,
                "panel": panel,
                "n": int(len(sub)),
                "data_path": str(data_path),
            }
        )
    return rows


def comparison_rows(metrics: pd.DataFrame, calibration: pd.DataFrame) -> list[dict]:
    rows = []
    if len(metrics):
        key = ["demography", "rho_regime", "pheno_mode", "seed", "outcome"]
        specs = [
            ("auc", "higher"),
            ("cindex", "higher"),
            ("brier", "lower"),
            ("log_loss", "lower"),
        ]
        for group_key, group in metrics.groupby(key):
            for metric, direction in specs:
                values = group[["method", metric]].dropna()
                values = values[values[metric] != ""]
                if "gamfit" not in set(values["method"]) or len(values) < 2:
                    continue
                values[metric] = pd.to_numeric(values[metric], errors="coerce")
                values = values.dropna()
                base = values[values["method"].isin(["z-norm", "linear-PC"])]
                gam = values[values["method"] == "gamfit"]
                if not len(base) or not len(gam):
                    continue
                best = base[metric].max() if direction == "higher" else base[metric].min()
                gv = float(gam[metric].iloc[0])
                delta = gv - best
                if direction == "lower":
                    verdict = "win" if delta < -1e-4 else "loss" if delta > 1e-4 else "tie"
                    advantage = -delta
                else:
                    verdict = "win" if delta > 1e-4 else "loss" if delta < -1e-4 else "tie"
                    advantage = delta
                rows.append(comparison_record(group_key, "global", metric, direction, gv, float(best), advantage, verdict))

    if len(calibration):
        key = ["demography", "rho_regime", "pheno_mode", "seed", "outcome"]
        cal = calibration[calibration["stratum"] == "all"].copy()
        cal["slope_error"] = (pd.to_numeric(cal["calibration_slope"], errors="coerce") - 1).abs()
        for group_key, group in cal.groupby(key):
            values = group[["method", "slope_error"]].dropna()
            if "gamfit" not in set(values["method"]):
                continue
            base = values[values["method"].isin(["z-norm", "linear-PC"])]
            gam = values[values["method"] == "gamfit"]
            if not len(base) or not len(gam):
                continue
            best = float(base["slope_error"].min())
            gv = float(gam["slope_error"].iloc[0])
            delta = gv - best
            verdict = "win" if delta < -1e-4 else "loss" if delta > 1e-4 else "tie"
            rows.append(comparison_record(group_key, "all", "calibration_slope_error", "lower", gv, best, -delta, verdict))
    return rows


def comparison_record(group_key, stratum: str, metric: str, direction: str, gamfit_value: float, best_baseline: float, advantage: float, verdict: str) -> dict:
    demography, rho_regime, pheno_mode, seed, outcome = group_key
    return {
        "demography": demography,
        "rho_regime": rho_regime,
        "pheno_mode": pheno_mode,
        "seed": int(seed),
        "outcome": outcome,
        "stratum": stratum,
        "metric": metric,
        "direction": direction,
        "gamfit_value": round(gamfit_value, 6),
        "best_baseline_value": round(best_baseline, 6),
        "gamfit_advantage": round(advantage, 6),
        "verdict": verdict,
    }


def make_plots(results_dir: Path, plots_dir: Path) -> None:
    metrics_path = results_dir / "metrics_global.csv"
    comparisons_path = results_dir / "comparisons.csv"
    calibration_path = results_dir / "calibration.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        for outcome, metric, ylabel in [
            ("binary", "auc", "AUC"),
            ("binary", "log_loss", "Log loss"),
            ("binary", "brier", "Brier"),
            ("survival", "cindex", "Harrell C-index"),
        ]:
            sub = metrics[(metrics["outcome"] == outcome)].copy()
            if metric not in sub or not len(sub):
                continue
            sub[metric] = pd.to_numeric(sub[metric], errors="coerce")
            sub = sub.dropna(subset=[metric])
            if not len(sub):
                continue
            plot_metric(sub, metric, ylabel, plots_dir / f"{outcome}_{metric}.png")

    if calibration_path.exists():
        cal = pd.read_csv(calibration_path)
        cal = cal[cal["stratum"] == "all"].copy()
        cal["slope_error"] = (pd.to_numeric(cal["calibration_slope"], errors="coerce") - 1).abs()
        cal = cal.dropna(subset=["slope_error"])
        if len(cal):
            plot_metric(cal, "slope_error", "|calibration slope - 1|", plots_dir / "calibration_slope_error_overall.png")

    if comparisons_path.exists():
        comp = pd.read_csv(comparisons_path)
        if len(comp):
            fig, ax = plt.subplots(figsize=(9, 4.8))
            counts = comp.groupby(["outcome", "metric", "verdict"]).size().unstack(fill_value=0)
            counts.plot(kind="bar", stacked=True, ax=ax, color={"win": "#2ca25f", "tie": "#bdbdbd", "loss": "#de2d26"})
            ax.set_ylabel("cell count")
            ax.set_title("gamfit vs best allowed baseline across strict cells")
            fig.tight_layout()
            fig.savefig(plots_dir / "gamfit_win_tie_loss_counts.png", dpi=140)
            plt.close(fig)


def plot_metric(df: pd.DataFrame, metric: str, ylabel: str, path: Path) -> None:
    summary = df.groupby(["rho_regime", "pheno_mode", "method"])[metric].mean().reset_index()
    labels = [f"{r}\n{p.replace('_', ' ')}" for r, p in summary[["rho_regime", "pheno_mode"]].drop_duplicates().itertuples(index=False)]
    combos = list(summary[["rho_regime", "pheno_mode"]].drop_duplicates().itertuples(index=False, name=None))
    x = np.arange(len(combos))
    width = 0.24
    fig, ax = plt.subplots(figsize=(max(8, 1.2 * len(combos)), 4.8))
    colors = {"z-norm": "#7570b3", "linear-PC": "#1b9e77", "gamfit": "#d95f02"}
    for offset, method in zip([-width, 0, width], METHODS):
        vals = []
        for combo in combos:
            row = summary[(summary["rho_regime"] == combo[0]) & (summary["pheno_mode"] == combo[1]) & (summary["method"] == method)]
            vals.append(float(row[metric].iloc[0]) if len(row) else np.nan)
        ax.bar(x + offset, vals, width=width, label=method, color=colors[method])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.set_title(ylabel + " on pooled held-out test data")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--demographies", choices=DEMOGRAPHIES, nargs="+", default=list(DEMOGRAPHIES))
    parser.add_argument("--rho-regimes", choices=tuple(RHO_REGIMES), nargs="+", default=list(RHO_REGIMES))
    parser.add_argument("--pheno-modes", choices=PHENO_MODES, nargs="+", default=list(PHENO_MODES))
    parser.add_argument("--centers", type=int, default=18)
    parser.add_argument("--timeout", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
