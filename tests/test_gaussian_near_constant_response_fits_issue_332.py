"""#332 contract: a tiny-scale Gaussian response is fit, not crashed on and not
refused.

The issue's reproducer (``y = rng.normal(scale=1e-13, size=200)``) once aborted
deep in the REML loop with an opaque ``reml_score must be finite, got inf``.
The first fix refused it pre-fit with an absolute floor
(``sample sd <= GAUSSIAN_MIN_SAMPLE_SD = 1e-10``), which broke scale invariance:
``c * y`` was refused for small ``c`` while ``y`` fit. A Gaussian REML fit is
equivariant under ``y -> c * y`` (the scale estimate absorbs ``c**2`` and the
criterion shifts by a constant), so the floor was removed and the reproducer
must now fit to exactly ``1e-13`` times the unit-scale fit.

The pins: the reproducer fits and matches the rescaled unit fit (no floor, no
inf crash), and well-conditioned and small-but-real responses still fit.
"""
from __future__ import annotations

import numpy as np
import gamfit


def test_near_constant_gaussian_response_fits_at_its_own_scale() -> None:
    # The issue reproducer: a response that is pure noise at scale 1e-13.
    rng = np.random.default_rng(0)
    n = 200
    x = np.linspace(0.0, 1.0, n)
    noise = rng.normal(size=n)

    unit = gamfit.fit({"x": x, "y": noise}, "y ~ s(x)", family="gaussian")
    tiny = gamfit.fit({"x": x, "y": noise * 1e-13}, "y ~ s(x)", family="gaussian")

    def mean(model):
        out = model.predict({"x": x})
        m = out if isinstance(out, np.ndarray) else out["posterior_mean"]
        return np.asarray(m, dtype=float).ravel()

    np.testing.assert_allclose(mean(tiny) / 1e-13, mean(unit), rtol=1e-6, atol=1e-9)


def test_well_conditioned_gaussian_response_still_fits() -> None:
    # Same shape, but a genuine O(1) signal: the guard must not over-reject.
    rng = np.random.default_rng(0)
    n = 200
    x = np.linspace(0.0, 1.0, n)
    data = {"x": x, "y": np.sin(2.0 * np.pi * x) + rng.normal(scale=0.1, size=n)}

    model = gamfit.fit(data, "y ~ s(x)", family="gaussian")
    out = model.predict(data)
    mean = out if isinstance(out, np.ndarray) else out["posterior_mean"]
    mean = np.asarray(mean, dtype=float).ravel()
    assert mean.shape[0] == n
    assert np.all(np.isfinite(mean))


def test_small_but_real_gaussian_signal_not_over_rejected() -> None:
    # sd ~ 1e-7 on an offset of 1: a legitimately finely-resolved measurement
    # must fit, not be rejected as degenerate.
    rng = np.random.default_rng(1)
    n = 200
    x = np.linspace(0.0, 1.0, n)
    data = {"x": x, "y": 1.0 + 1e-6 * (x + rng.normal(scale=0.1, size=n))}

    # Must not raise: this is a valid (if tiny-variance) signal.
    model = gamfit.fit(data, "y ~ s(x)", family="gaussian")
    out = model.predict(data)
    mean = out if isinstance(out, np.ndarray) else out["posterior_mean"]
    mean = np.asarray(mean, dtype=float).ravel()
    assert np.all(np.isfinite(mean))
