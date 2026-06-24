"""Bug hunt: ``predict(interval=...)`` reports ``std_error`` on the LINK scale
for non-identity-link families, not the response scale it is documented and
laid out to be.

The Python ``Model.predict`` docstring states that the ``std_error`` column is
the "response-scale standard error including both fixed-effect and smoothing
uncertainty", and it sits in the output table directly beside the
response-scale ``mean`` and the response-scale credible band
``mean_lower`` / ``mean_upper``.

For an identity-link Gaussian model the response scale and the linear-predictor
(link) scale coincide, so ``std_error`` is correct there.  For a non-identity
link (Poisson/Gamma ``log``, Bernoulli ``logit``, ...) they differ by the
inverse-link Jacobian ``dμ/dη``.  The credible band is built correctly on the
response scale (it uses the delta-method / posterior-integrated response-scale
SE ``mean_standard_error``), so

    (mean_upper - mean_lower) / (2 z)  ≈  SE_response

to within the mild asymmetry of the transformed band.  A response-scale
``std_error`` must therefore agree with that band half-width.

Observed (gamfit 0.1.224): the ``std_error`` column is populated from the
LINK-scale SE ``eta_standard_error`` instead of the response-scale
``mean_standard_error``
(``crates/gam-pyffi/src/geometry_ffi.rs:5910-5913`` and the siblings at
``:5960-5963`` / ``:6145-6146`` all insert ``prediction.eta_standard_error``).
For a Poisson ``log``-link fit this makes the reported ``std_error`` a factor of
``dμ/dη = μ`` too small relative to the band it accompanies — e.g. at a point
with ``mean ≈ 3.6`` the reported ``std_error ≈ 0.05`` while the response-scale
SE implied by the band is ``≈ 0.19`` (≈ 3.6× larger).  A user reading the
predicted count ``3.6 ± 0.05`` is given an uncertainty 3.6× too tight; for a
``logit`` model the same defect makes the reported SE several-fold too *wide*.

This test asserts the documented response-scale contract: ``std_error`` must be
consistent with the response-scale Wald band.  The identity-link Gaussian arm
is the passing control (response == link), pinning that the band itself is
correct and the discrepancy is link-specific.  When the FFI populates
``std_error`` from the response-scale SE, this test passes without edits.
"""

import os

os.environ.setdefault("GAM_LOG", "off")
os.environ.setdefault("RUST_LOG", "off")

import numpy as np
import pytest

import gamfit

# Two-sided z for a 0.90 central interval (the level requested below).
_Z_90 = 1.6448536269514722


def _predict_table(model, grid, level=0.9):
    out = model.predict({"x": grid}, interval=level, return_type="dict")
    mean = np.asarray(out["mean"], dtype=float)
    se = np.asarray(out["std_error"], dtype=float)
    lower = np.asarray(out["mean_lower"], dtype=float)
    upper = np.asarray(out["mean_upper"], dtype=float)
    return mean, se, lower, upper


def test_gaussian_std_error_matches_response_scale_band_control():
    """Identity link: response == link, so std_error already matches the band.

    This is the control: it must pass today and after any fix, demonstrating
    the band/SE machinery is sound on the identity link.
    """
    rng = np.random.default_rng(0)
    n = 800
    x = rng.uniform(-2.0, 2.0, n)
    y = np.sin(1.5 * x) + rng.normal(0.0, 0.4, n)
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="gaussian")

    grid = np.array([-1.0, 0.0, 1.0])
    _, se, lower, upper = _predict_table(model, grid)
    band_se = (upper - lower) / (2.0 * _Z_90)

    np.testing.assert_allclose(se, band_se, rtol=0.05, atol=1e-6)


def test_poisson_std_error_is_response_scale_consistent_with_band():
    """Log link: the reported ``std_error`` must be on the response scale.

    The response-scale SE implied by the credible band is
    ``(mean_upper - mean_lower) / (2 z)``.  A response-scale ``std_error`` must
    agree with it.  Today ``std_error`` is the link-scale SE, smaller by the
    factor ``dμ/dη = μ`` — at the largest-mean grid point (``μ`` clearly above
    1, band unclamped) the discrepancy is several-fold, far outside tolerance.
    """
    rng = np.random.default_rng(0)
    n = 800
    x = rng.uniform(-2.0, 2.0, n)
    y = rng.poisson(np.exp(0.5 + 0.8 * np.sin(1.5 * x)))
    model = gamfit.fit({"x": x, "y": y}, "y ~ s(x)", family="poisson")

    # Evaluate where the mean is comfortably above 1, so the link-vs-response
    # Jacobian dμ/dη = μ is well away from 1 and the response band is not
    # clamped at the Poisson support floor of 0.
    grid = np.array([0.0, 0.5, 1.0, 1.5])
    mean, se, lower, upper = _predict_table(model, grid)
    band_se = (upper - lower) / (2.0 * _Z_90)

    # Focus on the point with the largest mean (largest, most unambiguous
    # link-vs-response discrepancy).
    j = int(np.argmax(mean))
    assert mean[j] > 1.5, f"expected an interior mean > 1.5, got {mean[j]}"

    ratio = se[j] / band_se[j]
    # Response-scale std_error would give ratio ≈ 1; the link-scale SE actually
    # reported gives ratio ≈ 1/μ.  A 25% band is far wider than the transformed
    # band's asymmetry yet far tighter than the ~1/3.6 factor seen here.
    assert abs(ratio - 1.0) <= 0.25, (
        "predict() std_error is not on the response scale it is documented to be: "
        f"at mean={mean[j]:.4f} the reported std_error={se[j]:.5f} but the "
        f"response-scale SE implied by the band is {band_se[j]:.5f} "
        f"(ratio={ratio:.3f}; link-scale SE is a factor dμ/dη≈μ too small)."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
