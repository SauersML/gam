"""Estimated-scale prediction intervals use Student-t on n - edf (L2, L3).

The pyGAM audit (bench/pygam_audit, inference.md L2) found that Gaussian and
other estimated-scale prediction intervals used the standard-normal quantile
(``interval_policy.rs`` ``standard_normal_quantile``). The standardized mean
``(eta - x^T beta) / se`` is Gaussian only when the dispersion is known. With
phi estimated from the residuals, marginalizing phi under the reference prior
``1/phi`` gives Student-t on the residual degrees of freedom ``n - edf``. That
is the same reference the Wald smooth-term summary already uses. At n=60 the
normal quantile is too narrow. The audit's ``gauss_small`` cell (mc.py:
``y = sin(2 pi x1) + 0.3 cos(2 pi x3) + N(0, 0.5^2)``, three smooths,
fixed 60-row test design) measured 95% coverage of 0.934 (corrected), 0.921
(conditional) and 0.937 (observation). On the tree this lane was cut from,
the same 500 replicates measured 0.955 corrected and 0.940 observation with
the multiplier still exactly 1.95996. With the t_{n - edf} reference they
measure 0.959 corrected and 0.945 observation. Families with known scale (Poisson,
binomial, fixed dispersion) keep z, because their pivot really is Gaussian.

L3: ``PredictOptions.multi_point_joint`` was always false in every caller.
Joint bands go through ``effect_report`` instead, so the option has been
deleted per SPEC.md: "Unnecessary choices and options should be deleted, not
included."
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import gamfit

TWO_PI = 2.0 * np.pi
N_TRAIN = 60
N_TEST = 60
SIGMA = 0.5
REPLICATES = 500
# Monte Carlo SE of a 95% coverage mean over 500 replicates of 60 rows is
# about 0.003 (the audit's measured spread). The band is the lane's contract.
COVERAGE_BAND = (0.94, 0.96)


def _truth(x: np.ndarray) -> np.ndarray:
    return np.sin(TWO_PI * x[:, 0]) + 0.3 * np.cos(TWO_PI * x[:, 2])


def _replicate(rep: int, test_design: np.ndarray) -> tuple[float, float]:
    rng = np.random.default_rng(1000 + rep)
    x = rng.uniform(0.0, 1.0, (N_TRAIN, 3))
    y = _truth(x) + rng.normal(0.0, SIGMA, N_TRAIN)
    mu_test = _truth(test_design)
    y_new = mu_test + rng.normal(0.0, SIGMA, N_TEST)
    data = {"x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2], "y": y}
    test = {"x1": test_design[:, 0], "x2": test_design[:, 1], "x3": test_design[:, 2]}
    model = gamfit.fit(data, "y ~ s(x1) + s(x2) + s(x3)", family="gaussian")
    pred = model.predict(test, interval=0.95, observation_interval=True)
    lo = np.asarray(pred["posterior_mean_lower"])
    hi = np.asarray(pred["posterior_mean_upper"])
    obs_lo = np.asarray(pred["observation_lower"])
    obs_hi = np.asarray(pred["observation_upper"])
    return (
        float(np.mean((mu_test >= lo) & (mu_test <= hi))),
        float(np.mean((y_new >= obs_lo) & (y_new <= obs_hi))),
    )


@pytest.mark.slow
def test_gauss_small_estimated_scale_interval_coverage_is_nominal() -> None:
    test_design = np.random.default_rng(12345).uniform(0.02, 0.98, (N_TEST, 3))
    coverage = np.array([_replicate(rep, test_design) for rep in range(REPLICATES)])
    corrected, observation = coverage.mean(axis=0)
    lo, hi = COVERAGE_BAND
    assert lo <= corrected <= hi, f"corrected mean-interval coverage {corrected:.4f}"
    assert lo <= observation <= hi, f"observation-interval coverage {observation:.4f}"


def test_estimated_scale_multiplier_is_student_t_not_z() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 1.0, (N_TRAIN, 3))
    data = {"x1": x[:, 0], "x2": x[:, 1], "x3": x[:, 2]}
    y = _truth(x) + rng.normal(0.0, SIGMA, N_TRAIN)
    model = gamfit.fit({**data, "y": y}, "y ~ s(x1) + s(x2) + s(x3)", family="gaussian")
    pred = model.predict(data, interval=0.95, covariance_mode="conditional")
    half_width = np.asarray(pred["posterior_mean_upper"]) - np.asarray(pred["posterior_mean"])
    multiplier = half_width / np.asarray(pred["posterior_mean_standard_error"])
    np.testing.assert_allclose(multiplier, multiplier[0], rtol=1e-12)
    edf = model.summary().edf_total
    assert 3.0 < edf < 20.0, edf
    # t_{n - edf} 97.5% quantile for n - edf in (40, 57): strictly between
    # t_57 = 2.002465 and t_40 = 2.021075, and never the normal 1.959964.
    # The exact value against a high-precision reference is pinned in Rust
    # (gam-math student_t_quantile, gam-predict interval-reference tests).
    assert 2.002465 < multiplier[0] < 2.021075, multiplier[0]


def test_multi_point_joint_option_is_deleted_from_crates() -> None:
    crates = Path(__file__).resolve().parents[1] / "crates"
    assert crates.is_dir(), crates
    offenders = [
        str(path.relative_to(crates))
        for path in crates.rglob("*")
        if path.is_file()
        and path.suffix in {".rs", ".toml", ".md", ".py", ".pyi"}
        and "multi_point_joint" in path.read_text(encoding="utf-8", errors="replace")
    ]
    assert not offenders, offenders
