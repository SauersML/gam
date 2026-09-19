"""Smooth-term p-values of multi-predictor fits read the predictor they name.

The per-smooth table of ``Model.summary()`` describes the mean predictor. In a
multi-block fit that predictor is one block of the flat coefficient vector —
after the time block of a location-scale survival fit, before the scale block
of a Gaussian location-scale fit — so every coefficient and penalty index into
the fit has to be shifted by the widths of the blocks before it.

Before the fix the table indexed the flat vector from zero:

* a location-scale survival fit tested the TIME block's coefficients under the
  covariate smooths' names, so a null covariate read ``p ~ 1e-17`` and the real
  covariate got no p-value;
* a Gaussian location-scale fit (no block named "mean") compared the design's
  penalty count to the total of every block's smoothing parameters and
  reported the table unavailable with "Refit the model";
* a transformation (Royston-Parmar) survival fit, whose covariates follow the
  time basis inside one block, got that same stale-save message, and a Weibull
  fit, whose time basis is one ``log t`` column, read every smooth one column
  early.

The multinomial model gains the all-classes term test, whose null — the
covariate moves no class probability — does not mention the reference class.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _survival_frame(seed: int, n: int = 300) -> Any:
    rng = np.random.default_rng(seed)
    age = rng.uniform(40, 75, n)
    noise = rng.uniform(0, 1, n)
    eta = 0.05 * (age - 57.5)
    u = rng.uniform(1e-12, 1, n)
    t = 10.0 * (-np.log(u) * np.exp(-eta)) ** (1 / 1.5)
    c = np.minimum(rng.exponential(30.0, n), 25.0)
    return pd.DataFrame(
        {
            "entry": np.zeros(n),
            "exit": np.minimum(t, c),
            "event": (t <= c).astype(int),
            "age": age,
            "noise": noise,
        }
    )


def _rows(summary: Any) -> dict:
    assert summary.smooth_terms_unavailable is None, summary.smooth_terms_unavailable
    return {row["name"]: row for row in summary.smooth_terms}


def test_location_scale_survival_tests_the_threshold_block():
    df = _survival_frame(seed=1)
    model = gamfit.fit(
        df,
        "Surv(entry, exit, event) ~ s(age) + s(noise)",
        survival_likelihood="location-scale",
    )
    rows = _rows(model.summary())
    # The age effect is real and strong: it must be tested, and rejected.
    assert rows["s(age)"].get("p_value") is not None, rows["s(age)"]
    assert rows["s(age)"]["p_value"] < 1e-6, rows["s(age)"]
    # noise has no effect; reading the time block called it p ~ 1e-17.
    assert rows["s(noise)"]["p_value"] > 1e-3, rows["s(noise)"]


def test_gaussian_location_scale_mean_table_is_available():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    y = np.sin(2 * np.pi * z) + np.exp(-1.0 + 1.2 * x) * rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "z": z, "y": y})
    model = gamfit.fit(df, "y ~ s(x) + s(z)", noise_formula="s(x)")
    rows = _rows(model.summary())
    assert rows["s(z)"]["p_value"] < 1e-10, rows["s(z)"]
    # x drives only the noise scale: the mean-predictor null holds.
    assert rows["s(x)"]["p_value"] > 1e-3, rows["s(x)"]


@pytest.mark.parametrize("likelihood", ["transformation", "weibull"])
def test_shared_time_block_survival_tests_the_covariate_columns(likelihood):
    # The covariates follow the time basis inside one predictor block.
    df = _survival_frame(seed=1)
    model = gamfit.fit(
        df,
        "Surv(entry, exit, event) ~ s(age) + s(noise)",
        survival_likelihood=likelihood,
    )
    rows = _rows(model.summary())
    assert rows["s(age)"]["p_value"] < 1e-6, rows["s(age)"]
    assert rows["s(noise)"]["p_value"] > 1e-3, rows["s(noise)"]


def test_location_scale_scale_only_covariate_holds_its_size():
    """A small seeded size check of the mean-predictor s(x) test when x moves
    only the noise scale (the full study is bench/pvalue_calibration/
    pv-multi-predictor)."""
    rng = np.random.default_rng(20260919)
    n, reps = 300, 60
    pvalues = []
    for _ in range(reps):
        x = rng.uniform(0, 1, n)
        z = rng.uniform(0, 1, n)
        y = np.sin(2 * np.pi * z) + np.exp(-1.0 + 1.2 * x) * rng.standard_normal(n)
        df = pd.DataFrame({"x": x, "z": z, "y": y})
        model = gamfit.fit(df, "y ~ s(x) + s(z)", noise_formula="s(x)")
        pvalues.append(_rows(model.summary())["s(x)"]["p_value"])
    pvalues = np.asarray(pvalues)
    # Binomial(60, 0.05) exceeds 9 rejections with probability < 0.002.
    assert (pvalues <= 0.05).sum() <= 9, np.sort(pvalues)[:12]


def test_multinomial_joint_term_test_rows():
    rng = np.random.default_rng(7)
    n = 400
    x = rng.uniform(0, 1, n)
    z = rng.uniform(0, 1, n)
    eta = np.stack([np.zeros(n), 1.5 * np.sin(2 * np.pi * z), 1.5 * (z - 0.5)], axis=1)
    prob = np.exp(eta)
    prob /= prob.sum(axis=1, keepdims=True)
    y = (rng.uniform(size=n)[:, None] > np.cumsum(prob, axis=1)).sum(axis=1)
    df = pd.DataFrame({"x": x, "z": z, "y": np.array(["a", "b", "c"])[y]})
    model = gamfit.fit(df, "y ~ s(x) + s(z)", family="multinomial")
    joint = {row["term"]: row for row in model.joint_smooth_significance()}
    per_class = model.smooth_significance()
    assert set(joint) == {row["term"] for row in per_class}
    for term, row in joint.items():
        assert set(row) >= {"term", "edf", "ref_df", "statistic", "p_value", "unavailable"}
        assert row["unavailable"] is None, row
        # One test over every class block: its EDF is the per-class EDFs' sum.
        class_edf = sum(r["edf"] for r in per_class if r["term"] == term)
        assert row["edf"] == pytest.approx(class_edf, rel=1e-9)
    z_term = next(t for t in joint if "z" in t)
    x_term = next(t for t in joint if "x" in t)
    assert joint[z_term]["p_value"] < 1e-8, joint[z_term]
    assert joint[x_term]["p_value"] > 1e-3, joint[x_term]
