"""Parametric-term p-values in `summary()` are valid, and one value per surface.

A linear term carries its own REML ridge. Its Wald statistic is scaled by the
estimate's covariance with that ridge's own prior removed, which makes the
statistic the one of the fit with the term unpenalized whatever the ridge's
lambda is. Before this, the statistic used the posterior covariance, which
charges the ridge prior's variance to the estimate: at a true null the p-values
piled up near 1 (size ~0.012 at 0.05, KS p ~1e-250 over 500 Gaussian reps).

The seeded Monte Carlo checks here are a small version of
`bench/pvalue_calibration/pv-parametric/calibrate.py`: the rejection rate at a
true null lies within two Monte Carlo standard errors of the level, and the
null p-values do not reject U(0, 1) under a Kolmogorov-Smirnov test.
"""

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
REPS = 200


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
    for alpha in (0.10, 0.05):
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
    contrasts = [name for name in rows if name.startswith("g[")]
    assert len(contrasts) == len(LEVELS) - 1
    assert all(rows[name]["penalized"] is False for name in contrasts)
    # The statistic is the estimate over the SE the row reports.
    x1 = rows["x1"]
    assert x1["statistic"] == x1["estimate"] / x1["std_error"]
    # One joint Wald test for the factor on L - 1 df; a one-column term's test
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
