"""Public Python acceptance for honest uncertainty covariance provenance (#2296).

The default interval request is a *required* smoothing-corrected request.  A
successful prediction must therefore identify the resolved source as
``"smoothing-corrected"`` and agree with the explicit smoothing request, never
quietly reuse the conditional covariance.  Conversely, a valid fit that cannot
form the smoothing correction must refuse the default request with a typed
public error; conditional uncertainty remains available only when requested
explicitly.

Both tests fit and predict through the installed ``gamfit`` surface.  No FFI
payload is mocked or edited.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit


def test_default_uncertainty_uses_and_reports_smoothing_corrected_covariance() -> None:
    """Default uncertainty resolves to corrected covariance, not conditional."""
    x = np.linspace(-1.0, 1.0, 96)
    y = (
        0.4
        + 0.8 * x
        + 1.1 * np.sin(2.6 * x)
        + 0.16 * np.cos(7.0 * x)
        + 0.05 * np.sin(19.0 * x)
    )
    model = gamfit.fit(
        {"x": x, "y": y},
        "y ~ s(x, k=10)",
        family="gaussian",
    )
    grid = {"x": np.linspace(-0.9, 0.9, 13)}

    default = model.predict(grid, interval=0.9, return_type="dict")
    smoothing = model.predict(
        grid,
        interval=0.9,
        covariance_mode="smoothing",
        return_type="dict",
    )
    conditional = model.predict(
        grid,
        interval=0.9,
        covariance_mode="conditional",
        return_type="dict",
    )

    assert default.covariance_source == "smoothing-corrected"
    assert default["covariance_source"] == "smoothing-corrected"
    assert smoothing.covariance_source == "smoothing-corrected"
    assert conditional.covariance_source == "conditional"

    default_se = np.asarray(default.std_error, dtype=float)
    smoothing_se = np.asarray(smoothing.std_error, dtype=float)
    conditional_se = np.asarray(conditional.std_error, dtype=float)
    np.testing.assert_array_equal(default_se, smoothing_se)
    assert np.all(default_se >= conditional_se - 1e-12)
    assert np.any(default_se > conditional_se + 1e-10), (
        "the default interval is numerically indistinguishable from conditional "
        "covariance despite reporting smoothing-corrected provenance"
    )


def test_default_uncertainty_refuses_when_corrected_covariance_is_unavailable() -> None:
    """A required corrected request cannot downgrade on a conditional-only fit."""
    x = np.linspace(0.0, 1.0, 48)
    # The exact constant-Gaussian solution is a complete, valid fit with a
    # zero conditional covariance.  There is no smoothing-parameter correction
    # to report, so it is the canonical reachable refusal case rather than a
    # corrupted or hand-edited saved model.
    model = gamfit.fit(
        {"x": x, "y": np.full_like(x, 2.75)},
        "y ~ s(x, k=8)",
        family="gaussian",
    )
    grid = {"x": np.array([0.2, 0.5, 0.8])}

    with pytest.raises(gamfit.GamError) as raised:
        model.predict(grid, interval=0.9, return_type="dict")

    assert type(raised.value) is gamfit.GamError
    assert str(raised.value) == (
        "prediction with uncertainty failed: Invalid input: fit result does not "
        "contain smoothing-corrected covariance"
    )

    with pytest.raises(gamfit.GamError) as partial_dependence_raised:
        model.partial_dependence("s(x)", {"x": x}, grid=grid["x"])
    assert type(partial_dependence_raised.value) is gamfit.GamError
    assert str(partial_dependence_raised.value) == (
        "partial_dependence requires smoothing-corrected covariance; refit before "
        "requesting partial-dependence standard errors"
    )

    conditional = model.predict(
        grid,
        interval=0.9,
        covariance_mode="conditional",
        return_type="dict",
    )
    assert conditional.covariance_source == "conditional"
    np.testing.assert_array_equal(
        np.asarray(conditional.std_error, dtype=float),
        np.zeros(3),
    )

def _weibull_survival_frame(seed: int = 2296, n: int = 400):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    scale = np.exp(0.5 + 1.0 * x)
    event_time = scale * rng.weibull(1.3, n)
    censor_time = rng.exponential(6.0, n)
    observed = np.minimum(event_time, censor_time)
    event = (event_time <= censor_time).astype(float)
    return {"time": observed, "event": event, "x": x}


def test_single_cause_survival_interval_refuses_corrected_and_labels_conditional() -> None:
    """#2296 close gate (survival): the single-cause survival interval honors
    ``covariance_mode`` exactly. Location-scale fits persist no
    smoothing-corrected covariance, so the DEFAULT (required-corrected)
    interval request and the explicit ``"smoothing"`` request both refuse —
    they must never quietly return the narrower conditional-``Vb`` band — and
    the explicit conditional request succeeds with result-owned provenance in
    the returned payload."""
    df = _weibull_survival_frame()
    model = gamfit.fit(
        df,
        "Surv(time, event) ~ x",
        survival_likelihood="location-scale",
    )
    new_data = {"time": np.array([2.0, 2.0]), "event": np.array([1.0, 1.0]),
                "x": np.array([0.2, 0.8])}

    with pytest.raises(gamfit.GamError) as default_refusal:
        model.predict(new_data, interval=0.9, return_type="dict")
    assert "smoothing-corrected" in str(default_refusal.value)

    with pytest.raises(gamfit.GamError) as smoothing_refusal:
        model.predict(
            new_data,
            interval=0.9,
            covariance_mode="smoothing",
            return_type="dict",
        )
    assert "smoothing-corrected" in str(smoothing_refusal.value)

    conditional = model.predict(
        new_data,
        interval=0.9,
        covariance_mode="conditional",
        return_type="dict",
    )
    assert conditional["covariance_source"] == "conditional"
    survival_se = np.asarray(conditional["survival_se"], dtype=float)
    assert survival_se.shape[0] == 2
    assert np.all(np.isfinite(survival_se))

    # A point-only survival prediction consults no coefficient covariance and
    # must not claim one.
    plain = model.predict(new_data, return_type="dict")
    assert plain["covariance_source"] is None


def test_spline_scan_interval_refuses_corrected_and_labels_conditional() -> None:
    """#2296 close gate (specialized Python route): the exact O(n) spline-scan
    posterior variance is conditional on the profiled smoothing parameter. A
    corrected interval request (including the default) must refuse instead of
    silently returning the conditional band under an invented scan-specific
    label; the conditional request must be labeled ``"conditional"``."""
    rng = np.random.default_rng(22960)
    n = 4000  # large 1-D Gaussian smooth routes onto the exact O(n) scan path
    x = np.sort(rng.uniform(-1.0, 1.0, n))
    y = np.sin(3.0 * x) + rng.normal(0.0, 0.25, n)
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="gaussian")
    summary = model.summary()
    if "spline-scan" not in str(getattr(summary, "model_class", "")):
        pytest.skip("fit did not route onto the exact spline-scan path")
    grid = {"x": np.linspace(-0.9, 0.9, 7)}

    with pytest.raises(gamfit.GamError) as refusal:
        model.predict(grid, interval=0.9, return_type="dict")
    assert "conditional" in str(refusal.value)

    conditional = model.predict(
        grid,
        interval=0.9,
        covariance_mode="conditional",
        return_type="dict",
    )
    assert conditional["covariance_source"] == "conditional"
    assert "spline-scan-posterior" not in str(conditional.get("covariance_source"))


def test_curved_link_interval_reports_point_and_band_sources_separately() -> None:
    """#2296 close gate (curved-link dual source): a binomial posterior-mean
    point integrates the CONDITIONAL posterior by definition, even when the
    band is smoothing-corrected — two facts one tag cannot represent. The
    payload must carry both, each owned by the evaluator result."""
    rng = np.random.default_rng(22961)
    n = 220
    x = rng.uniform(-1.0, 1.0, n)
    eta = 0.4 + 1.6 * np.sin(2.2 * x)
    y = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-eta))).astype(float)
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x, k=8)", family="binomial")
    grid = {"x": np.linspace(-0.8, 0.8, 9)}

    smoothing = model.predict(grid, interval=0.9, return_type="dict")
    assert smoothing["covariance_source"] == "smoothing-corrected"
    assert smoothing["point_covariance_source"] == "conditional"

    conditional = model.predict(
        grid,
        interval=0.9,
        covariance_mode="conditional",
        return_type="dict",
    )
    assert conditional["covariance_source"] == "conditional"
    assert conditional["point_covariance_source"] == "conditional"


def test_summary_reports_definition_consistent_se_source() -> None:
    """#2296 close gate (summary provenance): the summary's SE column, the
    exported covariance, and their labels must come from ONE covariance
    definition, recorded on the payload."""
    x = np.linspace(-1.0, 1.0, 96)
    y = 0.3 + 0.9 * x + 1.2 * np.sin(2.4 * x) + 0.1 * np.cos(9.0 * x)
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x, k=10)", family="gaussian")
    summary = model.summary()
    se_source = summary["coefficient_se_source"]
    assert se_source in ("conditional", "smoothing-corrected")
    kind = summary.covariance_kind
    if kind is not None:
        assert kind == se_source, (
            "summary paired a covariance matrix from one definition with SEs "
            f"from another: covariance_kind={kind!r}, se source={se_source!r}"
        )
