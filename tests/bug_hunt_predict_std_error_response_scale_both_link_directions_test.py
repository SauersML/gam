"""Regression (companion to #1536, different angle): ``predict(interval=...)``
must report ``std_error`` on the RESPONSE scale for *both* directions of
inverse-link curvature.

The original failing test (``bug_hunt_predict_std_error_reported_on_link_scale_
not_response_scale_test.py``) pins the Poisson log-link arm, where the Jacobian
``dμ/dη = μ > 1`` makes a link-scale ``std_error`` too *small* (the band implies
a SE ``μ×`` larger). This test attacks the same root cause from the opposite
curvature sign and a second family, so a fix that only special-cased the log
link — or only the posterior-mean path for one family — would still fail here:

* **Bernoulli logit** — ``dμ/dη = p(1−p) < ¼``, so the *link*-scale SE is
  several-fold *larger* than the response-scale SE.  A link-scale ``std_error``
  would over-state the uncertainty (ratio to the band ≈ ``1/(p(1−p)) ≈ 4`` near
  ``p≈½``), the mirror image of the Poisson failure.
* **Gamma log** — a second curved (log) link on a continuous positive response,
  confirming the fix is not Poisson-count-specific.

In every case the contract is identical and link-agnostic: the response-scale
``std_error`` must agree with the response-scale Wald band half-width
``(mean_upper − mean_lower) / (2 z)`` that sits in the same row.
"""

import os

os.environ.setdefault("GAM_LOG", "off")
os.environ.setdefault("RUST_LOG", "off")

import numpy as np
import pytest

import gamfit

# Two-sided z for a 0.90 central interval (the level requested below).
_Z_90 = 1.6448536269514722


def _band_se(out):
    mean = np.asarray(out["mean"], dtype=float)
    se = np.asarray(out["std_error"], dtype=float)
    lower = np.asarray(out["mean_lower"], dtype=float)
    upper = np.asarray(out["mean_upper"], dtype=float)
    return mean, se, (upper - lower) / (2.0 * _Z_90)


def test_logit_std_error_is_response_scale_not_link_scale():
    """Logit: response SE < link SE, so a link-scale std_error is too *wide*."""
    rng = np.random.default_rng(1)
    n = 1500
    x = rng.uniform(-2.0, 2.0, n)
    p = 1.0 / (1.0 + np.exp(-2.0 * np.sin(1.5 * x)))
    y = (rng.uniform(0.0, 1.0, n) < p).astype(float)
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="binomial")

    # Evaluate near p≈0.5 where dμ/dη = p(1−p) ≈ 0.25 is far from 1, so the
    # link-vs-response discrepancy is largest and the band is unclamped.
    grid = np.array([-0.5, 0.0, 0.5])
    mean, se, band_se = _band_se(model.predict({"x": grid}, interval=0.9, return_type="dict"))

    # Pick the point closest to p = 0.5 (maximal curvature mismatch).
    j = int(np.argmin(np.abs(mean - 0.5)))
    assert 0.3 < mean[j] < 0.7, f"expected an interior probability, got {mean[j]}"
    ratio = se[j] / band_se[j]
    # Response-scale → ratio ≈ 1.  The buggy link-scale SE would give
    # ratio ≈ 1/(p(1−p)) ≈ 4 here (several-fold too wide).
    assert abs(ratio - 1.0) <= 0.25, (
        "logit predict() std_error is not on the response scale: at "
        f"p={mean[j]:.4f} reported std_error={se[j]:.5f} but the response-scale "
        f"SE implied by the band is {band_se[j]:.5f} (ratio={ratio:.3f}; a "
        "link-scale SE would be ~1/(p(1-p)) too wide)."
    )


def test_gamma_log_std_error_is_response_scale():
    """A second curved (log) link on a continuous positive response."""
    rng = np.random.default_rng(2)
    n = 1500
    x = rng.uniform(-2.0, 2.0, n)
    mu = np.exp(0.7 + 0.5 * np.sin(1.5 * x))
    # Gamma draws with shape k=4 (moderate dispersion), mean mu.
    y = rng.gamma(shape=4.0, scale=mu / 4.0)
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="gamma")

    grid = np.array([-1.0, 0.0, 1.0])
    mean, se, band_se = _band_se(model.predict({"x": grid}, interval=0.9, return_type="dict"))

    j = int(np.argmax(mean))
    assert mean[j] > 1.5, f"expected an interior mean > 1.5, got {mean[j]}"
    ratio = se[j] / band_se[j]
    assert abs(ratio - 1.0) <= 0.25, (
        "gamma-log predict() std_error is not on the response scale: at "
        f"mean={mean[j]:.4f} reported std_error={se[j]:.5f} but the band implies "
        f"{band_se[j]:.5f} (ratio={ratio:.3f})."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
