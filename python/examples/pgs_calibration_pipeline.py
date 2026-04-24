"""Two-stage polygenic score (PGS) calibration and outcome modelling demo.

Methods
-------
Stage 1 — conditional Gaussianization. For a polygenic score ``PGS`` and
ancestry/PC coordinates ``(pc1, pc2, pc3, pc4)`` we fit a transformation-
normal model ``h(PGS | PCs) ~ N(0, 1)`` using a Duchon spline with triple
penalty operators on the PC manifold. The anchored deviation invariant is
that, after the fitted conditional-Gaussianization map is applied, the
predicted z-scores are (a) marginally standard normal and (b) uncorrelated
with every PC coordinate. The fitted residual, ``PGS_cal``, is an
ancestry-corrected score that can be reused across downstream analyses
without re-fitting the calibration map.

Stage 2a — binary outcome via Bernoulli marginal-slope. With ``PGS_cal``
as the exposure ``z``, we fit ``disease ~ z + duchon(PCs) + linkwiggle``
under a probit link. The logslope formula ``duchon(PCs) + linkwiggle``
folds the PC manifold and an I-spline-based score-warp into the exposure
slope so that effect-size heterogeneity across ancestry is expressed as
an anchored monotone deviation from the identity slope.

Stage 2b — survival marginal-slope. The same ``PGS_cal`` exposure drives
a left-truncated survival fit ``Surv(age_entry, age_exit, event) ~ z +
duchon(PCs) + linkwiggle + timewiggle`` with a Gompertz-Makeham baseline
hazard. The timewiggle lets us depart from proportional hazards while
keeping a GM parametric anchor. Hazard and survival predictions are
queried at an arbitrary age grid via the SurvivalPrediction helper.

Evaluation is AUC (binary), C-index (survival), and the z-moment
diagnostics that define Stage 1's success. This mirrors the benchmark
table that the Nature-Genetics draft reports for the Duchon + linkwiggle
+ timewiggle configuration versus the parametric-only baselines.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

import gam
from gam.pgs import PgsCalibration


N_PCS = 4
PC_COLUMNS = [f"pc{i + 1}" for i in range(N_PCS)]


def load_biobank_sample(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    """Tiny biobank-like DataFrame matching the conftest synthetic_biobank schema."""
    rng = np.random.default_rng(seed)
    pc1 = rng.normal(0.0, 1.0, n)
    pc2 = 0.3 * pc1 + math.sqrt(1.0 - 0.09) * rng.normal(0.0, 1.0, n)
    pc3 = rng.normal(0.0, 1.0, n)
    pc4 = rng.normal(0.0, 1.0, n)

    raw = 0.4 * pc1 - 0.2 * pc2 + 0.15 * pc3 + rng.normal(0.0, 0.9, n)
    pgs = (raw - raw.mean()) / raw.std(ddof=0)

    def _phi(x: np.ndarray) -> np.ndarray:
        return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))

    probs = _phi(0.5 * pgs + 0.1 * pc1)
    disease = (rng.uniform(0.0, 1.0, n) < probs).astype(np.float64)

    age_entry = rng.uniform(40.0, 70.0, n)
    lam = np.exp(-1.2 - 0.3 * pgs)
    tte = rng.exponential(scale=1.0 / np.clip(lam, 1e-6, None), size=n)
    raw_exit = age_entry + tte
    censor = 85.0
    age_exit = np.minimum(raw_exit, censor)
    event = (raw_exit < censor).astype(np.float64)
    eps = 0.01
    too_short = age_exit <= age_entry + eps
    age_exit = np.where(too_short, age_entry + eps, age_exit)
    event = np.where(too_short, 0.0, event)

    return pd.DataFrame(
        {
            "pc1": pc1, "pc2": pc2, "pc3": pc3, "pc4": pc4,
            "PGS": pgs, "disease": disease,
            "age_entry": age_entry, "age_exit": age_exit, "event": event,
        }
    )


def _pc_duchon(centers: int) -> str:
    args = ", ".join(PC_COLUMNS)
    return f"duchon({args}, centers={centers}, order=1, power=1, double_penalty=true)"


def stage1_calibrate(df: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """Fit h(PGS | PCs) ~ N(0, 1) and return (calibration_model, df_with_PGS_cal)."""
    calibration = PgsCalibration.fit(
        df,
        pgs_column="PGS",
        pc_columns=PC_COLUMNS,
        centers=N_PCS + 20,
    )
    augmented = df.copy()
    augmented["PGS_cal"] = calibration.transform(df)
    return calibration, augmented


def stage2_binary(df: pd.DataFrame) -> Any:
    """Bernoulli marginal-slope with PC-varying log-slope + linkwiggle score-warp."""
    pc = _pc_duchon(centers=N_PCS + 20)
    main_formula = f"disease ~ z + {pc} + linkwiggle(degree=3, internal_knots=10)"
    logslope_formula = f"{pc} + linkwiggle(degree=3, internal_knots=10)"
    return gam.fit(
        df,
        main_formula,
        family="bernoulli-marginal-slope",
        link="probit",
        z_column="PGS_cal",
        logslope_formula=logslope_formula,
    )


def stage2_survival(df: pd.DataFrame) -> Any:
    """Survival marginal-slope: Gompertz-Makeham baseline + timewiggle + score-warp."""
    pc = _pc_duchon(centers=N_PCS + 20)
    main_formula = (
        "Surv(age_entry, age_exit, event) ~ z "
        f"+ {pc} + linkwiggle(degree=3, internal_knots=10) "
        "+ timewiggle(degree=3, internal_knots=8)"
    )
    logslope_formula = f"{pc} + linkwiggle(degree=3, internal_knots=10)"
    return gam.fit(
        df,
        main_formula,
        family="survival",
        survival_likelihood="marginal-slope",
        baseline_target="gompertz-makeham",
        z_column="PGS_cal",
        logslope_formula=logslope_formula,
    )


def evaluate(model: Any, df: pd.DataFrame, kind: str) -> dict[str, float]:
    """Return AUC for binary, C-index for survival, and z-moments for calibration."""
    if kind == "binary":
        probs = np.asarray(model.predict(df, return_type="dict")["mean"], float)
        return {"auc": _auc(df["disease"].to_numpy(), probs)}
    if kind == "survival":
        pred = model.predict(df)
        mid = 0.5 * (float(df["age_entry"].min()) + float(df["age_exit"].max()))
        risk = -pred.survival_at(np.array([mid]))[:, 0]
        return {
            "c_index": _c_index(
                df["age_exit"].to_numpy(), df["event"].to_numpy(), risk
            )
        }
    if kind == "calibration":
        z = np.asarray(model.transform(df), float)
        return {"z_mean": float(z.mean()), "z_std": float(z.std(ddof=0))}
    raise ValueError(f"unknown evaluation kind: {kind}")


def _auc(y_true: np.ndarray, score: np.ndarray) -> float:
    order = np.argsort(score)
    y = y_true[order]
    pos = float(y.sum())
    neg = float(y.shape[0] - pos)
    if pos == 0.0 or neg == 0.0:
        return float("nan")
    ranks = np.arange(1, y.shape[0] + 1, dtype=float)
    return float((ranks[y > 0.5].sum() - pos * (pos + 1.0) / 2.0) / (pos * neg))


def _c_index(times: np.ndarray, events: np.ndarray, risk: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0
    for i in range(times.shape[0]):
        if events[i] < 0.5:
            continue
        for j in range(times.shape[0]):
            if j == i or times[j] <= times[i]:
                continue
            comparable += 1
            if risk[i] > risk[j]:
                concordant += 1.0
            elif math.isclose(float(risk[i]), float(risk[j])):
                concordant += 0.5
    return concordant / comparable if comparable else float("nan")


def main() -> None:
    df = load_biobank_sample(n=2000, seed=0)
    split = len(df) // 2
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    print(f"[stage 1] fitting h(PGS | PCs) ~ N(0, 1) on n={len(train)}")

    calibration, train_cal = stage1_calibrate(train)
    test_cal = test.copy()
    test_cal["PGS_cal"] = calibration.transform(test)
    print(f"  train z: mean={float(train_cal['PGS_cal'].mean()):+.3f}, "
          f"sd={float(train_cal['PGS_cal'].std(ddof=0)):.3f}")

    print("[stage 2a] fitting Bernoulli marginal-slope + linkwiggle + score-warp")
    disease_model = stage2_binary(train_cal)
    binary = evaluate(disease_model, test_cal, kind="binary")

    print("[stage 2b] fitting survival marginal-slope + GM + timewiggle")
    surv_model = stage2_survival(train_cal)
    survival = evaluate(surv_model, test_cal, kind="survival")

    print("\n=== pipeline summary ===")
    print(f"  Stage 1  z-mean = {float(test_cal['PGS_cal'].mean()):+.3f}")
    print(f"  Stage 1  z-std  = {float(test_cal['PGS_cal'].std(ddof=0)):.3f}")
    print(f"  Stage 2a AUC    = {binary['auc']:.3f}")
    print(f"  Stage 2b C-idx  = {survival['c_index']:.3f}")


if __name__ == "__main__":
    main()
