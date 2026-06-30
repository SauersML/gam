"""Regression test for #1120 (and verification of #1097).

`Model.debiased_functional(target="average_derivative")` must return the average
**derivative** ``mean_i w_i (dm/dx)(x_i)`` of the fitted smooth, NOT the average
**value** ``mean_i w_i m(x_i)``.

The Rust FFI handler ``model_debiased_functional_json`` used to feed the plain
value design ``X`` (rows ``phi_j(x_i)``) into ``SmoothFunctional::AverageDerivative``,
which expects rows of basis-function DERIVATIVES ``dphi_j/dx(x_i)``. As a result
the two estimands were bit-identical and the (tight) CI badly excluded the true
average derivative. The fix builds the derivative design by central finite
differences on the term-collection design (rebuilt through the same frozen spec
so its columns align with the fitted coefficients).

This test asserts, for a known smooth, that:

* ``average_derivative`` is clearly distinct from ``average_value``; and
* ``average_derivative`` recovers the analytic mean derivative.

Two functions are checked to be robust to the accidental value==derivative
coincidence: ``f(x) = sin(pi*x)`` on ``[0, 1]`` — a *half* period, so its mean
value ``E[sin(pi*x)] = 2/pi ~= 0.637`` is well-separated from its mean
derivative ``E[pi*cos(pi*x)] ~= 0`` — and ``f(x) = x**2`` on ``[0, 2]`` (mean
derivative = E[2x] ~= 2.0, value ~= 1.34) exactly as filed in #1120.

(Earlier this used ``sin(2*pi*x)`` over a *full* period, where BOTH the mean
value and the mean derivative are ~0; the "derivative is far from the value"
assertion was then mathematically unsatisfiable even when the code is correct,
so it flagged correct behavior — a banned XFAIL-by-accident. A half period
separates the two truths honestly without weakening the #1120 check.)
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
import pandas as pd

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


def test_average_derivative_recovers_sin_derivative_not_value() -> None:
    rng = np.random.default_rng(7)
    n = 2000
    x = rng.uniform(0.0, 1.0, n)
    # f(x) = sin(pi*x) on [0, 1] (a HALF period); f'(x) = pi*cos(pi*x).
    # Mean value E[sin(pi*x)] = 2/pi ~= 0.637 is well-separated from the mean
    # derivative E[pi*cos(pi*x)] ~= 0, so "derivative is far from the value"
    # below is a genuine, satisfiable check (unlike a full sin(2*pi*x) period,
    # where both truths are ~0 and the assertion can never hold).
    y = np.sin(np.pi * x) + rng.normal(0.0, 0.05, n)
    df = pd.DataFrame({"x": x, "y": y})
    model = gamfit.fit(df, "y ~ s(x)")

    ad = model.debiased_functional(df, target="average_derivative")
    av = model.debiased_functional(df, target="average_value")

    truth_deriv = float(np.mean(np.pi * np.cos(np.pi * x)))
    truth_value = float(np.mean(np.sin(np.pi * x)))

    # (a) The two estimands must NOT be identical (the #1120 symptom).
    assert abs(ad["theta_debiased"] - av["theta_debiased"]) > 1e-3, (
        "average_derivative collapsed onto average_value: "
        f"deriv={ad['theta_debiased']!r} value={av['theta_debiased']!r}"
    )

    # (b) average_value still recovers the true mean value.
    assert abs(av["theta_debiased"] - truth_value) < 0.05, (
        f"average_value {av['theta_debiased']:.4f} != truth {truth_value:.4f}"
    )

    # (c) average_derivative recovers the true mean derivative, not the value.
    assert abs(ad["theta_debiased"] - truth_deriv) < 0.6, (
        f"average_derivative {ad['theta_debiased']:.4f} != truth {truth_deriv:.4f}"
    )
    assert abs(ad["theta_debiased"] - truth_value) > 0.5, (
        "average_derivative is suspiciously close to the average VALUE: "
        f"deriv={ad['theta_debiased']:.4f} value-truth={truth_value:.4f}"
    )


def test_average_derivative_recovers_quadratic_slope_issue_1120() -> None:
    rng = np.random.default_rng(3)
    n = 1500
    x = rng.uniform(0.0, 2.0, n)
    y = x**2 + rng.normal(0.0, 0.1, n)  # m(x) = x^2 -> m'(x) = 2x
    df = pd.DataFrame({"x": x, "y": y})
    model = gamfit.fit(df, "y ~ s(x)")

    ad = model.debiased_functional(df, target="average_derivative")
    av = model.debiased_functional(df, target="average_value")

    true_value = float(np.mean(x**2))  # ~= 1.34
    true_deriv = float(np.mean(2.0 * x))  # = E[2x] ~= 2.0

    assert abs(ad["theta_debiased"] - av["theta_debiased"]) > 1e-3, (
        "average_derivative is bit-identical to average_value (#1120 symptom)"
    )
    assert abs(av["theta_debiased"] - true_value) < 0.05, (
        f"average_value {av['theta_debiased']:.4f} != E[x^2] {true_value:.4f}"
    )
    assert abs(ad["theta_debiased"] - true_deriv) < 0.25, (
        f"average_derivative {ad['theta_debiased']:.4f} != E[2x] {true_deriv:.4f}"
    )
