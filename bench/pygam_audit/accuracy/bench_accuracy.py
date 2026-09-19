#!/usr/bin/env python3
"""Head-to-head out-of-sample accuracy benchmark: gamfit vs pyGAM.

Permanent-benchmark harness. For each case (real pyGAM dataset, repo CSV, or
synthetic truth) it runs K-fold CV (default 5, shuffled, fixed seed) and fits

  * ``pygam_default``  -- pyGAM with its default lam=0.6, n_splines=20
  * ``pygam_grid``     -- pyGAM ``gridsearch()`` (its intended use: shared
                          lam over logspace(-3,3,11), GCV/UBRE objective)
  * ``gamfit``         -- ``gamfit.fit(data, formula)`` with the default-
                          formula equivalent (REML/LAML, posterior mean)

and records held-out metrics per fold:

  gaussian : rmse, dev (= mean squared error)
  binomial : logloss, brier, auc
  poisson  : dev (mean Poisson deviance), rmse
  gamma    : dev (mean Gamma deviance), rmse
  synthetic: truth_mse = mean((mu_hat - mu_true)^2) on the held-out rows

Results are appended to ``results/<case>.json`` (one file per case, so the run
is resumable). ``report_accuracy.py`` aggregates them into win/loss/tie tables.

No timing is recorded (by design).

Usage:
  python bench_accuracy.py [--only REGEX] [--skip REGEX] [--folds 5] [--force]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import re
import sys
import traceback
import warnings
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
PYGAM_DATA = HERE / "pygam_data"
REPO_DATASETS = Path("/home/user/gam/bench/datasets")

import gamfit  # noqa: E402
import pygam  # noqa: E402
from pygam import GAM, LinearGAM, LogisticGAM, PoissonGAM, GammaGAM, s, f, te, l  # noqa: E402
import pygam.datasets.load_datasets as L  # noqa: E402

L.PATH = str(PYGAM_DATA)  # the 0.12.0 wheel ships loaders without the CSVs


# ----------------------------------------------------------------------------
# Case definition
# ----------------------------------------------------------------------------
@dataclass
class Case:
    name: str
    family: str  # gaussian | binomial | poisson | gamma
    cols: dict  # name -> 1-D float array (predictors)
    y: np.ndarray
    formula: str  # gamfit formula (response is "y")
    terms: Callable  # () -> pyGAM TermList, positional over list(cols)
    mu_true: np.ndarray | None = None
    kind: str = "real"
    group: str = ""
    pygam_kwargs: dict = field(default_factory=dict)

    def __post_init__(self):
        self.y = np.asarray(self.y, dtype=float)
        self.cols = {k: np.asarray(v, dtype=float) for k, v in self.cols.items()}
        if self.mu_true is not None:
            self.mu_true = np.asarray(self.mu_true, dtype=float)


PYGAM_CLASS = {
    "gaussian": LinearGAM,
    "binomial": LogisticGAM,
    "poisson": PoissonGAM,
    "gamma": GammaGAM,
}


def s_all(p, cats=()):
    t = None
    for j in range(p):
        term = f(j) if j in cats else s(j)
        t = term if t is None else t + term
    return t


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def metrics(family, y, mu, mu_true=None):
    y = np.asarray(y, float)
    mu = np.asarray(mu, float)
    out = {}
    if not np.all(np.isfinite(mu)):
        out["nonfinite"] = int(np.sum(~np.isfinite(mu)))
        mu = np.where(np.isfinite(mu), mu, np.nanmean(y))
    out["rmse"] = float(np.sqrt(np.mean((y - mu) ** 2)))
    if family == "gaussian":
        out["dev"] = float(np.mean((y - mu) ** 2))
    elif family == "binomial":
        p = np.clip(mu, 1e-15, 1 - 1e-15)
        out["logloss"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))
        out["brier"] = float(np.mean((y - mu) ** 2))
        out["auc"] = float(roc_auc_score(y, mu)) if len(np.unique(y)) == 2 else float("nan")
        out["dev"] = 2 * out["logloss"]
    elif family == "poisson":
        m = np.maximum(mu, 1e-300)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(y > 0, y * np.log(y / m), 0.0)
        out["dev"] = float(np.mean(2 * (t - (y - m))))
    elif family == "gamma":
        m = np.maximum(mu, 1e-300)
        out["dev"] = float(np.mean(2 * (-np.log(y / m) + (y - m) / m)))
    if mu_true is not None:
        out["truth_mse"] = float(np.mean((mu - mu_true) ** 2))
    return out


# ----------------------------------------------------------------------------
# Fitters
# ----------------------------------------------------------------------------
def fit_pygam(case, Xtr, ytr, Xte, grid):
    cls = PYGAM_CLASS[case.family]
    g = cls(case.terms(), **case.pygam_kwargs)
    if grid:
        g.gridsearch(Xtr, ytr, progress=False)
    else:
        g.fit(Xtr, ytr)
    info = {
        "edof": float(g.statistics_["edof"]),
        "lam": [float(np.ravel(v)[0]) for v in g.lam] if hasattr(g, "lam") else None,
    }
    return g.predict_mu(Xte) if case.family != "gaussian" else g.predict(Xte), info


def fit_gamfit(case, tr, te_idx):
    names = list(case.cols)
    dtr = {k: case.cols[k][tr] for k in names}
    dtr["y"] = case.y[tr]
    dte = {k: case.cols[k][te_idx] for k in names}
    m = gamfit.fit(dtr, case.formula, family=case.family)
    mu = np.asarray(m.predict(dte), float).ravel()
    sm = m.summary()
    info = {}
    for attr in ("edf_total", "iterations", "reml_score", "convergence", "deviance"):
        if hasattr(sm, attr):
            try:
                v = getattr(sm, attr)
                info[attr] = v if isinstance(v, (int, float, bool, str)) else str(v)
            except Exception:
                pass
    try:
        info["lambdas"] = {str(k): float(v) for k, v in m.smoothing_parameters().items()}
    except Exception:
        pass
    return mu, info


METHODS = ("pygam_default", "pygam_grid", "gamfit")


def run_case(case: Case, folds=5, seed=0):
    n = len(case.y)
    X = np.column_stack([case.cols[k] for k in case.cols]).astype(float)
    if case.family == "binomial":
        splitter = StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, case.y)
    else:
        splitter = KFold(folds, shuffle=True, random_state=seed).split(X)
    rows = []
    for k, (tr, te_idx) in enumerate(splitter):
        mt = case.mu_true[te_idx] if case.mu_true is not None else None
        for meth in METHODS:
            rec = {"case": case.name, "fold": k, "method": meth}
            try:
                if meth == "gamfit":
                    mu, info = fit_gamfit(case, tr, te_idx)
                else:
                    mu, info = fit_pygam(case, X[tr], case.y[tr], X[te_idx], meth == "pygam_grid")
                rec.update(metrics(case.family, case.y[te_idx], mu, mt))
                rec["info"] = info
                rec["ok"] = True
            except Exception as e:  # a failure is recorded, never hidden
                rec["ok"] = False
                rec["error"] = f"{type(e).__name__}: {str(e)[:400]}"
            rows.append(rec)
    return rows


# ----------------------------------------------------------------------------
# Case battery
# ----------------------------------------------------------------------------
def pygam_cases():
    cs = []
    X, y = L.wage()
    cs.append(Case("wage", "gaussian", {"year": X[:, 0], "age": X[:, 1], "edu": X[:, 2]}, y,
                   "y ~ s(year) + s(age) + factor(edu)", lambda: s(0) + s(1) + f(2)))
    X, y = L.mcycle()
    cs.append(Case("mcycle", "gaussian", {"times": X[:, 0]}, y, "y ~ s(times)", lambda: s(0)))
    X, y = L.coal()
    cs.append(Case("coal", "poisson", {"year": X[:, 0]}, y, "y ~ s(year)", lambda: s(0)))
    X, y = L.faithful()
    cs.append(Case("faithful", "poisson", {"erupt": X[:, 0]}, y, "y ~ s(erupt)", lambda: s(0)))
    X, y = L.trees()
    cs.append(Case("trees", "gamma", {"girth": X[:, 0], "height": X[:, 1]}, y,
                   "y ~ s(girth) + s(height)", lambda: s(0) + s(1)))
    X, y = L.default()
    cs.append(Case("default", "binomial", {"student": X[:, 0], "balance": X[:, 1], "income": X[:, 2]}, y,
                   "y ~ factor(student) + s(balance) + s(income)", lambda: f(0) + s(1) + s(2)))
    X, y = L.cake()
    cs.append(Case("cake", "gaussian", {"recipe": X[:, 0], "replicate": X[:, 1], "temp": X[:, 2]}, y,
                   "y ~ factor(recipe) + factor(replicate) + s(temp)", lambda: f(0) + f(1) + s(2)))
    X, y = L.hepatitis()
    cs.append(Case("hepatitis", "gaussian", {"age": X[:, 0]}, y, "y ~ s(age)", lambda: s(0)))
    np.random.seed(0)
    X, y = L.toy_classification(n=5000)
    cs.append(Case("toy_classification", "binomial",
                   {"c0": X[:, 0], "c1": X[:, 1], "i0": X[:, 2], "i1": X[:, 3], "i2": X[:, 4], "cat": X[:, 5]},
                   y.astype(float), "y ~ s(c0) + s(c1) + s(i0) + s(i1) + s(i2) + factor(cat)",
                   lambda: s(0) + s(1) + s(2) + s(3) + s(4) + f(5)))
    X, y = L.head_circumference()
    cs.append(Case("head_circumference", "gaussian", {"age": X[:, 0]}, y, "y ~ s(age)", lambda: s(0)))
    X, y = L.chicago()
    cs.append(Case("chicago", "poisson", {"time": X[:, 0], "tmpd": X[:, 1], "pm10": X[:, 2], "o3": X[:, 3]}, y,
                   "y ~ s(time) + s(tmpd) + s(pm10) + s(o3)", lambda: s(0) + s(1) + s(2) + s(3)))
    # pyGAM's own docs model for chicago: s(time, 200 splines) + te(o3, tmpd) + s(pm10)
    cs.append(Case("chicago_docs", "poisson", {"time": X[:, 0], "tmpd": X[:, 1], "pm10": X[:, 2], "o3": X[:, 3]}, y,
                   "y ~ s(time, k=200) + te(o3, tmpd) + s(pm10)",
                   lambda: s(0, n_splines=200) + te(3, 1) + s(2)))
    rng = np.random.default_rng(0)
    xi = rng.uniform(-1, 1, (5000, 2))
    mu = np.sin(xi[:, 0] * 2 * np.pi * 1.5) * xi[:, 1]
    yi = mu + rng.normal(0, 0.1, 5000)
    cs.append(Case("toy_interaction", "gaussian", {"x0": xi[:, 0], "x1": xi[:, 1]}, yi,
                   "y ~ te(x0, x1)", lambda: te(0, 1), mu_true=mu, kind="synth"))
    for c in cs:
        c.group = c.group or "pygam.datasets"
    return cs


def repo_csv_cases():
    import pandas as pd
    cs = []
    p = REPO_DATASETS

    def fl(a):
        return np.asarray(a, dtype=float)

    d = pd.read_csv(p / "lidar.csv")
    cs.append(Case("lidar", "gaussian", {"range": fl(d["range"])}, fl(d["logratio"]), "y ~ s(range)", lambda: s(0)))
    d = pd.read_csv(p / "gagurine.csv")
    cs.append(Case("gagurine", "gaussian", {"age": fl(d["Age"])}, fl(np.log(d["GAG"])), "y ~ s(age)", lambda: s(0)))
    d = pd.read_csv(p / "prostate.csv")
    cs.append(Case("prostate_pc", "binomial", {"pc1": fl(d["pc1"]), "pc2": fl(d["pc2"])}, fl(d["y"]),
                   "y ~ s(pc1) + s(pc2)", lambda: s(0) + s(1)))
    d = pd.read_csv(p / "haberman.csv", header=None, names=["age", "year", "nodes", "status"])
    cs.append(Case("haberman", "binomial", {"age": fl(d["age"]), "year": fl(d["year"]), "nodes": fl(d["nodes"])},
                   fl(d["status"] == 2), "y ~ s(age) + s(year) + s(nodes)", lambda: s(0) + s(1) + s(2)))
    d = pd.read_csv(p / "heart_failure_clinical_records_dataset.csv")
    feats = ["age", "ejection_fraction", "serum_creatinine", "serum_sodium", "platelets", "creatinine_phosphokinase"]
    cs.append(Case("heart_failure", "binomial", {c: fl(d[c]) for c in feats}, fl(d["DEATH_EVENT"]),
                   "y ~ " + " + ".join(f"s({c})" for c in feats), lambda: s_all(6)))
    d = pd.read_csv(p / "quakes.csv")
    cs.append(Case("quakes", "gaussian", {"depth": fl(d["depth"]), "stations": fl(d["stations"])}, fl(d["mag"]),
                   "y ~ s(depth) + s(stations)", lambda: s(0) + s(1)))
    cs.append(Case("quakes_space", "gaussian", {"lat": fl(d["lat"]), "lon": fl(d["long"])}, fl(d["depth"]),
                   "y ~ te(lat, lon)", lambda: te(0, 1)))
    d = pd.read_csv(p / "badhealth.csv")
    cs.append(Case("badhealth", "poisson", {"age": fl(d["age"]), "badh": fl(d["badh"])}, fl(d["numvisit"]),
                   "y ~ s(age) + factor(badh)", lambda: s(0) + f(1)))
    d = pd.read_csv(p / "penguins.csv").dropna(subset=["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"])
    cs.append(Case("penguins_mass", "gaussian", {"bl": fl(d["bill_length_mm"]), "bd": fl(d["bill_depth_mm"]),
                                                 "fl": fl(d["flipper_length_mm"])}, fl(d["body_mass_g"]) / 1000,
                   "y ~ s(bl) + s(bd) + s(fl)", lambda: s(0) + s(1) + s(2)))
    d = pd.read_csv(p / "nottem_monthly_temp.csv")
    t = fl(d["year"]) + (fl(d["month"]) - 0.5) / 12
    cs.append(Case("nottem", "gaussian", {"t": t, "month": fl(d["month"])}, fl(d["temp"]),
                   "y ~ s(t) + s(month)", lambda: s(0) + s(1)))
    d = pd.read_csv(p / "global_major_city_temp.csv")
    cs.append(Case("city_temp", "gaussian", {"lat": fl(d["lat"]), "lon": fl(d["lon"])}, fl(d["temp"]),
                   "y ~ s(lat) + s(lon)", lambda: s(0) + s(1)))
    d = pd.read_csv(p / "bike_sharing_torus.csv")
    cs.append(Case("bike", "gaussian", {"season": fl(d["season"]), "hour": fl(d["hour"])}, fl(d["log_count"]),
                   "y ~ te(season, hour)", lambda: te(0, 1)))
    d = pd.read_csv(p / "sleepstudy.csv")
    cs.append(Case("sleepstudy", "gaussian", {"days": fl(d["Days"]), "subj": fl(d["Subject"])}, fl(d["Reaction"]),
                   "y ~ s(days) + factor(subj)", lambda: s(0) + f(1)))
    for c in cs:
        c.group = "bench/datasets"
    return cs


# ---------------------------- synthetic truths ------------------------------
def f_sin(k):
    return lambda x: np.sin(2 * np.pi * k * x)


def doppler(x):
    return np.sqrt(x * (1 - x)) * np.sin(2.1 * np.pi / (x + 0.05)) * 3


def gamsim(X):
    f0 = 2 * np.sin(np.pi * X[:, 0])
    f1 = np.exp(2 * X[:, 1])
    f2 = 0.2 * X[:, 2] ** 11 * (10 * (1 - X[:, 2])) ** 6 + 10 * (10 * X[:, 2]) ** 3 * (1 - X[:, 2]) ** 10
    return f0 + f1 + f2  # x3 is pure noise


def synth_cases(big=True):
    cs = []

    def add(name, family, cols, y, mu, formula, terms, group):
        cs.append(Case(name, family, cols, y, formula, terms, mu_true=mu, kind="synth", group=group))

    # 1-D wiggly gaussian
    for fname, fn in [("sin1", f_sin(1)), ("sin3", f_sin(3)), ("sin6", f_sin(6)), ("doppler", doppler)]:
        for n in ([100, 500, 2000] + ([10000] if fname in ("sin3", "sin6") else [])):
            # zlib.crc32 is stable across processes (hash() of a str is salted per process)
            rng = np.random.default_rng(zlib.crc32(f"{fname}:{n}".encode()))
            x = rng.uniform(0, 1, n)
            mu = fn(x)
            y = mu + rng.normal(0, 0.3 * np.std(mu) + 0.1, n)
            add(f"g1d_{fname}_n{n}", "gaussian", {"x": x}, y, mu, "y ~ s(x)", lambda: s(0), "synth: 1-D wiggly")
    # null truth
    for n in (200, 2000):
        rng = np.random.default_rng(n + 7)
        X = rng.uniform(0, 1, (n, 3))
        mu = np.zeros(n)
        y = rng.normal(0, 1, n)
        add(f"null3_n{n}", "gaussian", {"a": X[:, 0], "b": X[:, 1], "c": X[:, 2]}, y, mu,
            "y ~ s(a) + s(b) + s(c)", lambda: s(0) + s(1) + s(2), "synth: null")
    # additive 4-d (gamSim eg 1)
    for n in (200, 1000, 5000):
        rng = np.random.default_rng(n)
        X = rng.uniform(0, 1, (n, 4))
        mu = gamsim(X)
        y = mu + rng.normal(0, 2, n)
        add(f"add4_n{n}", "gaussian", {f"x{j}": X[:, j] for j in range(4)}, y, mu,
            "y ~ s(x0) + s(x1) + s(x2) + s(x3)", lambda: s_all(4), "synth: additive")
    # interaction
    for n in (1000, 4000):
        rng = np.random.default_rng(n + 1)
        X = rng.uniform(0, 1, (n, 2))
        mu = 3 * np.exp(-((X[:, 0] - .3) ** 2 + (X[:, 1] - .6) ** 2) / .05) + 2 * np.exp(-((X[:, 0] - .75) ** 2 + (X[:, 1] - .25) ** 2) / .02)
        y = mu + rng.normal(0, .3, n)
        add(f"bump2d_n{n}", "gaussian", {"x0": X[:, 0], "x1": X[:, 1]}, y, mu, "y ~ te(x0, x1)", lambda: te(0, 1),
            "synth: interaction")
    # heteroscedastic
    for n in (500, 3000):
        rng = np.random.default_rng(n + 2)
        x = rng.uniform(0, 1, n)
        mu = np.sin(2 * np.pi * 2 * x) + x
        y = mu + rng.normal(0, 0.05 + 1.2 * x ** 2, n)
        add(f"hetero_n{n}", "gaussian", {"x": x}, y, mu, "y ~ s(x)", lambda: s(0), "synth: heteroscedastic")
    # outliers (5% gross)
    for n in (300, 2000):
        rng = np.random.default_rng(n + 3)
        x = rng.uniform(0, 1, n)
        mu = np.sin(2 * np.pi * 2 * x)
        y = mu + rng.normal(0, 0.3, n)
        idx = rng.choice(n, n // 20, replace=False)
        y[idx] += rng.standard_t(1.5, len(idx)) * 5
        add(f"outlier_n{n}", "gaussian", {"x": x}, y, mu, "y ~ s(x)", lambda: s(0), "synth: outliers")
    # binomial
    for n in (300, 1000, 5000) + ((20000,) if big else ()):
        rng = np.random.default_rng(n + 4)
        X = rng.uniform(0, 1, (n, 4))
        eta = (gamsim(X) - 7.5) / 1.5
        mu = 1 / (1 + np.exp(-eta))
        y = (rng.uniform(size=n) < mu).astype(float)
        add(f"binom_add4_n{n}", "binomial", {f"x{j}": X[:, j] for j in range(4)}, y, mu,
            "y ~ s(x0) + s(x1) + s(x2) + s(x3)", lambda: s_all(4), "synth: binomial")
    for n in (500, 3000):
        rng = np.random.default_rng(n + 5)
        x = rng.uniform(0, 1, n)
        eta = 2 * np.sin(2 * np.pi * 2 * x)
        mu = 1 / (1 + np.exp(-eta))
        y = (rng.uniform(size=n) < mu).astype(float)
        add(f"binom_sin2_n{n}", "binomial", {"x": x}, y, mu, "y ~ s(x)", lambda: s(0), "synth: binomial")
    # near separation: steep logistic
    for n in (200, 1000):
        rng = np.random.default_rng(n + 6)
        x = rng.uniform(0, 1, n)
        z = rng.uniform(0, 1, n)
        eta = 25 * (x - 0.5) + np.sin(2 * np.pi * z)
        mu = 1 / (1 + np.exp(-eta))
        y = (rng.uniform(size=n) < mu).astype(float)
        add(f"nearsep_n{n}", "binomial", {"x": x, "z": z}, y, mu, "y ~ s(x) + s(z)", lambda: s(0) + s(1),
            "synth: near-separation")
    # poisson
    for n in (300, 2000, 10000):
        rng = np.random.default_rng(n + 8)
        X = rng.uniform(0, 1, (n, 2))
        eta = 1 + np.sin(2 * np.pi * 1.5 * X[:, 0]) + 0.8 * np.cos(2 * np.pi * X[:, 1])
        mu = np.exp(eta)
        y = rng.poisson(mu).astype(float)
        add(f"pois_add2_n{n}", "poisson", {"x0": X[:, 0], "x1": X[:, 1]}, y, mu, "y ~ s(x0) + s(x1)",
            lambda: s(0) + s(1), "synth: poisson")
    # low-count poisson
    for n in (500,):
        rng = np.random.default_rng(n + 9)
        x = rng.uniform(0, 1, n)
        mu = np.exp(-1.5 + 1.5 * np.sin(2 * np.pi * 2 * x))
        y = rng.poisson(mu).astype(float)
        add(f"pois_lowcount_n{n}", "poisson", {"x": x}, y, mu, "y ~ s(x)", lambda: s(0), "synth: poisson")
    # gamma
    for n in (300, 2000):
        rng = np.random.default_rng(n + 10)
        X = rng.uniform(0, 1, (n, 2))
        mu = np.exp(0.5 + np.sin(2 * np.pi * X[:, 0]) + X[:, 1] ** 2)
        shape = 2.0
        y = rng.gamma(shape, mu / shape)
        add(f"gamma_add2_n{n}", "gamma", {"x0": X[:, 0], "x1": X[:, 1]}, y, mu, "y ~ s(x0) + s(x1)",
            lambda: s(0) + s(1), "synth: gamma")
    if big:
        for fname, fn in [("sin6", f_sin(6)), ("doppler", doppler)]:
            n = 50000
            rng = np.random.default_rng(n + 11 + len(fname))
            x = rng.uniform(0, 1, n)
            mu = fn(x)
            y = mu + rng.normal(0, 0.3 * np.std(mu) + 0.1, n)
            add(f"g1d_{fname}_n{n}", "gaussian", {"x": x}, y, mu, "y ~ s(x)", lambda: s(0), "synth: large n")
        n = 50000
        rng = np.random.default_rng(99)
        X = rng.uniform(0, 1, (n, 4))
        mu = gamsim(X)
        y = mu + rng.normal(0, 2, n)
        add(f"add4_n{n}", "gaussian", {f"x{j}": X[:, j] for j in range(4)}, y, mu,
            "y ~ s(x0) + s(x1) + s(x2) + s(x3)", lambda: s_all(4), "synth: large n")
    return cs


def all_cases(big=True):
    return pygam_cases() + repo_csv_cases() + synth_cases(big=big)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip", default=None)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-big", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)
    cases = all_cases(big=not a.no_big)
    if a.only:
        cases = [c for c in cases if re.search(a.only, c.name)]
    if a.skip:
        cases = [c for c in cases if not re.search(a.skip, c.name)]
    if a.list:
        for c in cases:
            print(c.name, c.family, len(c.y), c.formula)
        return
    for c in cases:
        out = RESULTS / f"{c.name}.json"
        if out.exists() and not a.force:
            continue
        print(f"[run] {c.name} n={len(c.y)} {c.family} :: {c.formula}", flush=True)
        try:
            rows = run_case(c, folds=a.folds)
        except Exception:
            traceback.print_exc()
            continue
        meta = {"name": c.name, "family": c.family, "n": int(len(c.y)), "formula": c.formula,
                "kind": c.kind, "group": c.group, "rows": rows}
        out.write_text(json.dumps(meta, default=str))
        for meth in METHODS:
            r = [x for x in rows if x["method"] == meth]
            ok = [x for x in r if x["ok"]]
            key = "truth_mse" if c.mu_true is not None else "dev"
            vals = [x[key] for x in ok]
            print(f"   {meth:14s} ok={len(ok)}/{len(r)} {key}={np.mean(vals) if vals else float('nan'):.5g}"
                  + ("" if len(ok) == len(r) else f"  ERR={[x.get('error') for x in r if not x['ok']][:1]}"),
                  flush=True)


if __name__ == "__main__":
    main()
