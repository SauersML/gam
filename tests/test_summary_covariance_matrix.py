"""``Model.summary().covariance`` is the coefficient covariance as one matrix.

The summary hands the coefficient covariance to Python as a ``(p, p)`` float64
array built in Rust, not as a row-major list of boxed floats plus a side length
for the caller to reshape.
"""

from __future__ import annotations

import numpy as np

import gamfit


def test_summary_covariance_is_the_coefficient_covariance_matrix() -> None:
    rng = np.random.default_rng(11)
    n = 300
    x0 = rng.uniform(size=n)
    x1 = rng.uniform(size=n)
    y = np.sin(2.0 * np.pi * x0) + np.cos(2.0 * np.pi * x1) + rng.normal(scale=0.3, size=n)
    model = gamfit.fit({"x0": x0, "x1": x1, "y": y}, "y ~ s(x0) + s(x1)")
    summary = model.summary()

    covariance = summary.covariance
    p = len(summary.coefficients)
    assert isinstance(covariance, np.ndarray)
    assert covariance.dtype == np.float64
    assert covariance.shape == (p, p)
    np.testing.assert_array_equal(covariance, covariance.T)

    std_error = np.array([row["std_error"] for row in summary.coefficients])
    np.testing.assert_allclose(np.sqrt(np.diag(covariance)), std_error, rtol=1e-12)

    assert "covariance_flat" not in summary
    assert "covariance_n" not in summary
    assert "covariance" not in summary._repr_html_()
