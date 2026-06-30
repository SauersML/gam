"""#332 contract: a near-constant Gaussian response is rejected pre-fit with a
clear, actionable error — not an opaque ``reml_score must be finite, got inf``.

A Gaussian fit's marginal REML log-likelihood carries a ``-n/2 * log sigma^2``
term that diverges to ``+inf`` as the maximum-likelihood scale ``sigma -> 0``.
For an effectively-constant response (the issue's reproducer:
``y = rng.normal(scale=1e-13, size=200)``) every outer seed is therefore
rejected and the fit aborts with ``reml_score must be finite, got inf``, which
gives the user no hint that the real problem is a signal-free response column.

The fix adds a family-owned degeneracy guard
(``ResponseFamily::validate_response_degeneracy`` Gaussian arm,
``crates/gam-spec/src/lib.rs``): it computes the two-pass mean-centred sample
standard deviation and refuses ``sd <= GAUSSIAN_MIN_SAMPLE_SD = 1e-10`` *before*
the marginal diverges, with a message that names the observed sd, the threshold,
and concrete remedies (check units, centre/rescale, or drop the column).

This test pins both directions: the near-constant reproducer raises a clean
``gamfit.GamError`` (no panic, no inf-crash message), and a well-conditioned
response with the same shape fits successfully (no over-rejection).
"""
from __future__ import annotations

import numpy as np
import pytest

import gamfit


def test_near_constant_gaussian_response_rejected_cleanly() -> None:
    # The issue reproducer: a response that is constant up to f64 noise.
    rng = np.random.default_rng(0)
    n = 200
    data = {
        "x": np.linspace(0.0, 1.0, n),
        "y": rng.normal(scale=1e-13, size=n),
    }

    with pytest.raises(gamfit.GamError) as excinfo:
        gamfit.fit(data, "y ~ s(x)", family="gaussian")

    msg = str(excinfo.value)
    # The actionable degeneracy message, NOT the opaque internal crash.
    assert "effectively constant" in msg, msg
    assert "reml_score must be finite" not in msg, (
        "near-constant Gaussian response should be rejected by the pre-fit "
        f"degeneracy guard, not crash deep in the REML loop: {msg!r}"
    )


def test_well_conditioned_gaussian_response_still_fits() -> None:
    # Same shape, but a genuine O(1) signal: the guard must not over-reject.
    rng = np.random.default_rng(0)
    n = 200
    x = np.linspace(0.0, 1.0, n)
    data = {"x": x, "y": np.sin(2.0 * np.pi * x) + rng.normal(scale=0.1, size=n)}

    model = gamfit.fit(data, "y ~ s(x)", family="gaussian")
    out = model.predict(data)
    mean = out if isinstance(out, np.ndarray) else out["mean"]
    mean = np.asarray(mean, dtype=float).ravel()
    assert mean.shape[0] == n
    assert np.all(np.isfinite(mean))


def test_small_but_real_gaussian_signal_not_over_rejected() -> None:
    # sd ~ 1e-6 is small but far above the 1e-10 floor: a legitimately
    # finely-resolved measurement must fit, not be rejected as degenerate.
    rng = np.random.default_rng(1)
    n = 200
    x = np.linspace(0.0, 1.0, n)
    data = {"x": x, "y": 1.0 + 1e-6 * (x + rng.normal(scale=0.1, size=n))}

    # Must not raise: this is a valid (if tiny-variance) signal.
    model = gamfit.fit(data, "y ~ s(x)", family="gaussian")
    out = model.predict(data)
    mean = out if isinstance(out, np.ndarray) else out["mean"]
    mean = np.asarray(mean, dtype=float).ravel()
    assert np.all(np.isfinite(mean))
