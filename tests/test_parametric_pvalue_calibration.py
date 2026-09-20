"""Parametric-term p-values in `summary()` are calibrated, and one value per surface.

A linear term carries its own REML ridge. Its Wald statistic is scaled by the
estimate's covariance with that ridge's own prior removed, which makes the
statistic the one of the fit with the term unpenalized whatever the ridge's
lambda is. Before this, the statistic used the posterior covariance, which
charges the ridge prior's variance to the estimate: at a true null the p-values
piled up near 1 (size ~0.012 at 0.05, KS p ~1e-250 over 500 Gaussian reps).

The seeded Monte Carlo checks here are a small version of
`bench/pvalue_calibration/pv-parametric/calibrate.py`. At a true null the
rejection rate at 0.10, 0.05 and 0.01 lies within two Monte Carlo standard
errors of the level on BOTH sides: a conservative test fails exactly as an
anti-conservative one does. The null p-values also must not reject U(0, 1)
under a two-sided Kolmogorov-Smirnov test.

`REPS` is the smallest count at which the band at 0.01 excludes a zero rate,
`2 sqrt(0.01 * 0.99 / REPS) < 0.01`, so every level is tested from below.

The Gaussian scale is profiled: `RSS / (n - edf)` with the smoothing
parameters fit to the same residuals, which biases it low, so a t on
`n - edf` rejected too often (x1 at 12000 null reps: +2.8 MCSE at 0.10). Each
estimated-scale reference is now on `n - tau`, where `tau` charges the EDF for
the uncertainty of the smoothing parameters the statistic depends on. An
unpenalized coefficient depends on all of them, so its `tau` is the fit's
smoothing-corrected EDF; a ridged one is conditioned on its own ridge, so its
`tau` lies between the plain and the corrected EDF.
"""

import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import gamfit

_REPO_BIN = Path(__file__).resolve().parent.parent / "target" / "release" / "gam"
GAM = str(_REPO_BIN) if _REPO_BIN.exists() else shutil.which("gam")

LEVELS = ("a", "b", "c", "d")
REPS = 400


def _null_frame(family: str, n: int, rep: int) -> pd.DataFrame:
    """A real smooth in `x2`; neither `x1` nor the factor `g` enters."""
    rng = np.random.default_rng([11, n, rep])
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    codes = rng.integers(0, len(LEVELS), n)
    eta = np.sin(2.0 * np.pi * x2)
    if family == "gaussian":
        y = eta + rng.normal(0.0, 1.0, n)
    else:
        y = rng.poisson(np.exp(0.5 + 0.8 * eta)).astype(float)
    g = pd.Categorical(np.asarray(LEVELS)[codes], categories=LEVELS)
    return pd.DataFrame({"x1": x1, "x2": x2, "g": g, "y": y})


def _null_pvalues(family: str, n: int):
    linear, factor = [], []
    for rep in range(REPS):
        summary = gamfit.fit(
            _null_frame(family, n, rep), "y ~ x1 + g + s(x2)", family=family
        ).summary()
        linear.append({r["name"]: r for r in summary.parametric_terms}["x1"]["p_value"])
        factor.append({r["name"]: r for r in summary.parametric_term_tests}["g"]["p_value"])
    return np.asarray(linear, dtype=float), np.asarray(factor, dtype=float)


def _assert_calibrated(p: np.ndarray, label: str) -> None:
    assert len(p) == REPS and np.all(np.isfinite(p)), label
    for alpha in (0.10, 0.05, 0.01):
        rate = float(np.mean(p <= alpha))
        mcse = math.sqrt(alpha * (1.0 - alpha) / len(p))
        assert abs(rate - alpha) <= 2.0 * mcse, (label, alpha, rate, mcse)
    ks = stats.kstest(p, "uniform").pvalue
    assert ks > 0.01, (label, ks)


def test_gaussian_null_linear_and_factor_pvalues_are_calibrated():
    linear, factor = _null_pvalues("gaussian", 100)
    _assert_calibrated(linear, "x1")
    _assert_calibrated(factor, "g")


def test_poisson_null_linear_and_factor_pvalues_are_calibrated():
    linear, factor = _null_pvalues("poisson", 200)
    _assert_calibrated(linear, "x1")
    _assert_calibrated(factor, "g")


def test_ridged_row_reports_its_null_sd_and_the_factor_is_one_joint_test():
    summary = gamfit.fit(_null_frame("gaussian", 100, 0), "y ~ x1 + g + s(x2)").summary()
    rows = {r["name"]: r for r in summary.parametric_terms}
    assert rows["x1"]["penalized"] is True
    assert rows["Intercept"]["penalized"] is False
    # The ridged factor's level deviations are a variance-component block with
    # the smooth terms; only its joint test is parametric.
    assert not any(name.startswith("g") for name in rows)
    # The statistic is the estimate over the SE the row reports.
    x1 = rows["x1"]
    assert x1["statistic"] == x1["estimate"] / x1["std_error"]
    # One joint Wald test for the factor on its L - 1 contrasts; a one-column term's test
    # is its row squared, with the same p-value bit for bit.
    tests = {r["name"]: r for r in summary.parametric_term_tests}
    assert tests["g"]["df"] == len(LEVELS) - 1
    assert tests["x1"]["df"] == 1
    assert tests["x1"]["p_value"] == x1["p_value"]
    assert "Ridge-penalized (x1)" in str(summary)


def test_cli_and_python_read_the_same_parametric_pvalues(tmp_path):
    assert GAM is not None, "the gam CLI must be built (target/release/gam or PATH)"
    model = gamfit.fit(_null_frame("gaussian", 100, 1), "y ~ x1 + g + s(x2)")
    path = tmp_path / "model.gam"
    model.save(str(path))
    # `gam summary` renders the payload the saved model yields; the loaded model
    # yields the same p-values as the in-memory one, bit for bit.
    loaded = gamfit.load(str(path)).summary()
    fresh = model.summary()
    for table in ("parametric_terms", "parametric_term_tests"):
        before = [float(r["p_value"]).hex() for r in getattr(fresh, table)]
        after = [float(r["p_value"]).hex() for r in getattr(loaded, table)]
        assert before == after, table
    cli = subprocess.run([GAM, "summary", str(path)], capture_output=True, text=True)
    assert cli.returncode == 0, cli.stderr
    assert cli.stdout == str(fresh)
    assert "Parametric terms:" in cli.stdout
    # The CLI's payload carries the same p-values, bit for bit.
    cli_json = subprocess.run(
        [GAM, "summary", "--json", str(path)], capture_output=True, text=True
    )
    assert cli_json.returncode == 0, cli_json.stderr
    payload = json.loads(cli_json.stdout)
    for table in ("parametric_terms", "parametric_term_tests"):
        before = [float(r["p_value"]).hex() for r in getattr(fresh, table)]
        after = [float(r["p_value"]).hex() for r in payload[table]]
        assert before == after, table


def test_gaussian_residual_df_charges_the_smoothing_parameters_the_statistic_depends_on():
    frame = _null_frame("gaussian", 100, 2)
    n = len(frame)
    # An unpenalized coefficient's statistic depends on every smoothing
    # parameter, so its residual df is n minus the smoothing-corrected EDF.
    unpenalized = gamfit.fit(
        frame, "y ~ linear(x1, double_penalty=false) + g + s(x2)"
    ).summary()
    rows = {r["name"]: r for r in unpenalized.parametric_terms}
    assert rows["x1"]["penalized"] is False
    corrected = n - unpenalized.edf_corrected
    for name in ("Intercept", "x1"):
        assert math.isclose(rows[name]["residual_df"], corrected, rel_tol=1e-9), (
            name, rows[name]["residual_df"], corrected
        )
    # A ridged coefficient's own ridge is conditioned on: its residual df sits
    # between n minus the corrected EDF and n minus the plain EDF.
    ridged = gamfit.fit(frame, "y ~ x1 + g + s(x2)").summary()
    rows = {r["name"]: r for r in ridged.parametric_terms}
    tests = {r["name"]: r for r in ridged.parametric_term_tests}
    assert rows["x1"]["penalized"] is True
    for df in (rows["x1"]["residual_df"], tests["g"]["residual_df"]):
        assert n - ridged.edf_corrected <= df <= n - ridged.edf_total, (
            df, ridged.edf_corrected, ridged.edf_total
        )
    assert tests["x1"]["residual_df"] == rows["x1"]["residual_df"]
    assert "t residual df, charged for the smoothing parameters" in str(ridged)


def test_a_known_scale_reference_publishes_no_residual_df():
    summary = gamfit.fit(
        _null_frame("poisson", 200, 0), "y ~ x1 + g + s(x2)", family="poisson"
    ).summary()
    for table in (summary.parametric_terms, summary.parametric_term_tests):
        assert all("residual_df" not in r or r["residual_df"] is None for r in table)
