"""The fitted-model summary is one Rust-rendered report.

`print(model.summary())` and `gam summary MODEL` print the same string, rendered
by one function from one typed payload, and the deviance explained it reports is
the proper proportion of null deviance `1 − D/D₀` (not an inverted McFadden
ratio): it lies in [0, 1] for these fits and equals the closed-form value
computed here from the data alone.
"""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gamfit

# CI builds target/release/gam without putting it on PATH.
_REPO_BIN = Path(__file__).resolve().parent.parent / "target" / "release" / "gam"
GAM = str(_REPO_BIN) if _REPO_BIN.exists() else shutil.which("gam")


def _gaussian_frame(n: int = 200, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-1.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    y = 0.5 + 0.8 * x1 + np.sin(2.0 * np.pi * x2) + rng.normal(0.0, 0.3, n)
    return pd.DataFrame({"x1": x1, "x2": x2, "y": y})


def test_cli_and_python_print_the_same_summary(tmp_path):
    assert GAM is not None, "the gam CLI must be built (target/release/gam or PATH)"
    model = gamfit.fit(_gaussian_frame(), "y ~ x1 + s(x2)")
    path = tmp_path / "model.gam"
    model.save(str(path))

    cli = subprocess.run([GAM, "summary", str(path)], capture_output=True, text=True)
    assert cli.returncode == 0, cli.stderr

    python_text = str(model.summary())
    assert cli.stdout == python_text
    assert str(gamfit.load(str(path)).summary()) == python_text
    # The report carries every section the summary promises.
    for needle in (
        "Family: Gaussian Identity",
        "Link function: identity",
        "Formula: y ~ x1 + s(x2)",
        "n: 200",
        "Parametric coefficients:",
        "t value",
        "Approximate significance of smooth terms:",
        "Deviance explained: ",
        "Adjusted R-squared: ",
        "Scale estimate: ",
        "REML score: ",
        "Log-likelihood: ",
        "Conditional AIC: ",
        "Corrected AIC: ",
        "Convergence: certified",
    ):
        assert needle in python_text, (needle, python_text)


def test_gaussian_deviance_explained_is_one_minus_deviance_over_null_deviance():
    df = _gaussian_frame()
    y = df["y"].to_numpy()
    summary = gamfit.fit(df, "y ~ x1 + s(x2)").summary()

    null_deviance = float(np.sum((y - y.mean()) ** 2))
    assert summary.null_deviance == pytest.approx(null_deviance, rel=1e-10)
    explained = summary.deviance_explained
    assert 0.0 <= explained <= 1.0
    assert explained == pytest.approx(1.0 - summary.deviance / null_deviance, rel=1e-12)
    assert summary.deviance_explained_unavailable is None


def test_linear_gaussian_deviance_explained_is_the_fits_r_squared():
    # The linear coefficient carries a REML-selected ridge, so the fit is not
    # OLS; the reference is the R-squared of the fit's own fitted values.
    df = _gaussian_frame()
    y = df["y"].to_numpy()
    n = len(y)
    model = gamfit.fit(df, "y ~ x1")
    summary = model.summary()

    rss = float(np.sum((y - np.asarray(model.predict(df))) ** 2))
    tss = float(np.sum((y - y.mean()) ** 2))
    assert summary.deviance == pytest.approx(rss, rel=1e-12)
    r_squared = 1.0 - rss / tss
    assert 0.0 <= summary.deviance_explained <= 1.0
    assert summary.deviance_explained == pytest.approx(r_squared, rel=1e-12)
    # Residual df is n minus the effective degrees of freedom.
    assert summary.adjusted_r_squared == pytest.approx(
        1.0 - (rss / (n - summary.edf_total)) / (tss / (n - 1)), rel=1e-12
    )
    assert summary.parametric_statistic == "t"
    assert summary.parametric_term_statistic == "F"
    assert [row["name"] for row in summary.parametric_terms] == ["Intercept", "x1"]


def test_poisson_null_deviance_is_the_intercept_only_deviance():
    rng = np.random.default_rng(11)
    n = 300
    x = rng.uniform(0.0, 1.0, n)
    y = rng.poisson(np.exp(0.3 + np.sin(2.0 * np.pi * x)))
    df = pd.DataFrame({"x": x, "y": y.astype(float)})
    summary = gamfit.fit(df, "y ~ s(x)", family="poisson").summary()

    mean = y.mean()
    positive = y > 0
    null_deviance = 2.0 * float(np.sum(y[positive] * np.log(y[positive] / mean)))
    assert summary.null_deviance == pytest.approx(null_deviance, rel=1e-9)
    assert 0.0 <= summary.deviance_explained <= 1.0
    assert summary.deviance_explained == pytest.approx(
        1.0 - summary.deviance / null_deviance, rel=1e-12
    )
    # Poisson has no estimated scale: z statistics, and the smooth row carries
    # its variance-component score statistic, referred to its own law.
    assert summary.adjusted_r_squared is None
    assert summary.parametric_statistic == "z"
    assert summary.parametric_term_statistic == "Chi.sq"
    (smooth,) = summary.smooth_terms
    assert "statistic" not in smooth
    assert smooth["chi_sq"] > 0.0
    assert 0.0 <= smooth["p_value"] < 1e-6
    assert "Score" in str(summary)
