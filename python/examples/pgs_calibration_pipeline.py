"""Two-stage polygenic score (PGS) calibration and outcome modelling demo.

Method overview
---------------
Stage 1 is phenotype-blind: we fit a Duchon spline of the raw PGS on the
ancestry/PC manifold so that, after conditioning, the score is anchored to
a common Gaussian reference ("conditional Gaussianization"). The fitted
residual ``PGS_cal`` is an ancestry-corrected score that can be reused
across downstream analyses without re-fitting the calibration map.

Stage 2 fits the phenotype on top of ``PGS_cal`` as the exposure ``z``.
The log-slope model on ``z`` is parameterised by a smooth function on the
same PC manifold plus a link-wiggle, so effect-size heterogeneity is
expressed as an "anchored monotone deviation" from the identity slope.
For survival outcomes, a Gompertz-Makeham baseline with a timewiggle
captures non-proportional hazards while keeping a parametric anchor.

GAP annotations
---------------
Lines tagged with ``# GAP:`` call out config keys / families the demo
WOULD use if the Python binding plumbing existed. Today the binding at
``crates/gam-pyffi/src/lib.rs:24-31`` only accepts ``family``, ``offset``,
``weights``, ``ridge_lambda`` (``#[serde(deny_unknown_fields)]``) and only
dispatches ``FitRequest::Standard`` (``lib.rs:211-219``), so every non-
standard family below is currently unreachable from Python.

Run with: ``python -m python.examples.pgs_calibration_pipeline``
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

import gam


N_PCS = 4
PC_COLUMNS = [f"pc{i + 1}" for i in range(N_PCS)]


def load_biobank_sample(n: int = 2000, seed: int = 0) -> dict[str, list[float]]:
    """Synthesize a tiny biobank-like table: PCs, PGS, disease, survival."""
    rng = np.random.default_rng(seed)
    pcs = rng.normal(size=(n, N_PCS))
    ancestry_shift = 0.4 * pcs[:, 0] - 0.2 * pcs[:, 1] ** 2
    raw_pgs = rng.normal(loc=ancestry_shift, scale=1.0)
    true_logit = -2.0 + 0.8 * (raw_pgs - ancestry_shift) + 0.3 * pcs[:, 2]
    disease = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-true_logit))).astype(float)
    entry = np.zeros(n)
    hazard = 0.01 * np.exp(0.7 * (raw_pgs - ancestry_shift))
    exit_time = rng.exponential(scale=1.0 / np.clip(hazard, 1e-4, None))
    censor = rng.uniform(5.0, 20.0, size=n)
    event = (exit_time < censor).astype(float)
    exit_time = np.minimum(exit_time, censor)
    table: dict[str, list[float]] = {name: pcs[:, i].tolist() for i, name in enumerate(PC_COLUMNS)}
    table["PGS"] = raw_pgs.tolist()
    table["disease"] = disease.tolist()
    table["entry"] = entry.tolist()
    table["exit"] = exit_time.tolist()
    table["event"] = event.tolist()
    return table


def _pc_duchon_term(centers: int) -> str:
    args = ", ".join(PC_COLUMNS)
    return (
        f"duchon({args}, centers={centers}, order=1, power=1, double_penalty=true)"
    )


def stage1_calibrate_pgs(df: dict[str, list[float]]) -> tuple[Any, dict[str, list[float]]]:
    """Fit PGS ~ duchon(PCs) to produce ancestry-corrected PGS_cal."""
    formula = f"PGS ~ {_pc_duchon_term(centers=N_PCS + 20)}"
    calib_model = gam.fit(
        df,
        formula=formula,
        family="gaussian",
        # GAP: transformation_normal payload key not plumbed. CLI has
        # --transformation-normal at src/main.rs:290, but PyFitConfig
        # (crates/gam-pyffi/src/lib.rs:24-31) rejects unknown fields and
        # fit_table_impl at lib.rs:211-219 only dispatches FitRequest::Standard.
        # Need to: (a) add `transformation_normal: Option<bool>` to PyFitConfig,
        # (b) route FitRequest::TransformationNormal through fit_table_impl.
        # GAP: scale_dimensions payload key not plumbed. CLI flag at
        # src/main.rs:358-359; FitConfig field at src/solver/workflow.rs:949.
        # Add `scale_dimensions: Option<bool>` to PyFitConfig and forward to
        # FitConfig in parse_fit_config (lib.rs:428).
    )
    predicted = calib_model.predict(df, return_type="dict")
    pgs_cal = [obs - mean for obs, mean in zip(df["PGS"], predicted["mean"])]
    df_augmented = dict(df)
    df_augmented["PGS_cal"] = pgs_cal
    return calib_model, df_augmented


def stage2_binary(df: dict[str, list[float]]) -> Any:
    """Bernoulli marginal-slope on PGS_cal with PC-varying log-slope."""
    pc_duchon = _pc_duchon_term(centers=N_PCS + 10)
    formula = f"disease ~ z + {pc_duchon} + linkwiggle(degree=3, internal_knots=10)"
    logslope_formula = f"{pc_duchon} + linkwiggle(degree=3, internal_knots=10)"
    # GAP: family="bernoulli-marginal-slope" is defined in Rust
    # (src/main.rs:561) but crates/gam-pyffi/src/lib.rs:211-219 explicitly
    # rejects any FitRequest variant other than Standard. Need to:
    #   (a) add the family to parse_fit_config's family routing,
    #   (b) extend fit_table_impl to persist FittedModelPayload for
    #       FitRequest::BernoulliMarginalSlope,
    #   (c) extend build_standard_predict_input for marginal-slope prediction.
    # GAP: link="probit", z_column=..., logslope_formula=... payload keys
    # not plumbed. CLI wires them at src/main.rs:252/257/296 and FitConfig
    # holds them at src/solver/workflow.rs:941-944. Add `link`,
    # `z_column`, `logslope_formula` fields to PyFitConfig at lib.rs:24.
    disease_model = gam.fit(
        df,
        formula=formula,
        family="binomial",
        config={
            "family_hint": "bernoulli-marginal-slope",  # GAP: key ignored today.
            "z_column": "PGS_cal",                      # GAP: deny_unknown_fields rejects.
            "link": "probit",                           # GAP: deny_unknown_fields rejects.
            "logslope_formula": logslope_formula,       # GAP: deny_unknown_fields rejects.
        },
    )
    return disease_model


def stage2_survival(df: dict[str, list[float]]) -> Any:
    """Survival marginal-slope with Gompertz-Makeham baseline + timewiggle."""
    pc_duchon = _pc_duchon_term(centers=N_PCS + 10)
    formula = (
        "Surv(entry, exit, event) ~ z + "
        f"{pc_duchon} + linkwiggle(degree=3, internal_knots=10) + "
        "timewiggle(degree=3, internal_knots=8)"
    )
    logslope_formula = f"{pc_duchon} + linkwiggle(degree=3, internal_knots=10)"
    # GAP: Surv(...) LHS and family="survival" route to FitRequest::Survival*
    # which fit_table_impl (lib.rs:211-219) rejects. Need (a) survival
    # dispatch in fit_table_impl, (b) a Surv-aware response_column_name
    # (currently returns None at lib.rs:578), (c) prediction path for
    # survival models in build_standard_predict_input (lib.rs:688).
    # GAP: survival_likelihood="marginal-slope" (src/main.rs:296, FitConfig
    # at workflow.rs:928) not exposed; baseline_target="gompertz-makeham"
    # (src/main.rs:301-302) not exposed. Both need new PyFitConfig fields.
    surv_model = gam.fit(
        df,
        formula=formula,
        family="survival",                                   # GAP: non-standard family.
        config={
            "survival_likelihood": "marginal-slope",         # GAP: deny_unknown_fields.
            "baseline_target": "gompertz-makeham",           # GAP: deny_unknown_fields.
            "z_column": "PGS_cal",                           # GAP: deny_unknown_fields.
            "logslope_formula": logslope_formula,            # GAP: deny_unknown_fields.
        },
    )
    return surv_model


def binary_auc(y_true: list[float], score: list[float]) -> float:
    pairs = sorted(zip(score, y_true), key=lambda pair: pair[0])
    pos = sum(1 for _, y in pairs if y > 0.5)
    neg = len(pairs) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = 0.0
    for rank, (_, y) in enumerate(pairs, start=1):
        if y > 0.5:
            rank_sum += rank
    return (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)


def survival_concordance(times: list[float], events: list[float], risk: list[float]) -> float:
    concordant = 0
    comparable = 0
    for i, (t_i, e_i, r_i) in enumerate(zip(times, events, risk)):
        if e_i < 0.5:
            continue
        for j, (t_j, _, r_j) in enumerate(zip(times, events, risk)):
            if i == j or t_j <= t_i:
                continue
            comparable += 1
            if r_i > r_j:
                concordant += 1
            elif math.isclose(r_i, r_j):
                concordant += 0.5  # type: ignore[assignment]
    return concordant / comparable if comparable else float("nan")


def main() -> None:
    data = load_biobank_sample()
    n = len(data["PGS"])
    split = n // 2
    df_train = {key: values[:split] for key, values in data.items()}
    df_test = {key: values[split:] for key, values in data.items()}

    print(f"[stage 1] fitting PGS ~ duchon(PCs) on n={split}")
    calib_model, df_train_cal = stage1_calibrate_pgs(df_train)
    test_pred = calib_model.predict(df_test, return_type="dict")
    df_test_cal = dict(df_test)
    df_test_cal["PGS_cal"] = [
        obs - mean for obs, mean in zip(df_test["PGS"], test_pred["mean"])
    ]
    print(f"  train PGS_cal var = {float(np.var(df_train_cal['PGS_cal'])):.3f}")

    print("[stage 2a] fitting disease ~ z + duchon(PCs) + linkwiggle")
    disease_model = stage2_binary(df_train_cal)
    disease_pred = disease_model.predict(df_test_cal, return_type="dict")
    auc = binary_auc(df_test_cal["disease"], disease_pred["mean"])
    print(f"  test AUC = {auc:.3f}")

    print("[stage 2b] fitting Surv(entry, exit, event) ~ z + duchon(PCs) + wiggles")
    surv_model = stage2_survival(df_train_cal)
    surv_pred = surv_model.predict(df_test_cal, return_type="dict")
    c_index = survival_concordance(
        df_test_cal["exit"], df_test_cal["event"], surv_pred["eta"]
    )
    print(f"  test C-index = {c_index:.3f}")


if __name__ == "__main__":
    main()
