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
