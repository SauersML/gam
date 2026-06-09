"""Regression: the numeric ``by=`` variable of a varying-coefficient smooth
``s(x, by=z)`` must NOT be axis-clipped to its training range at predict time.

A by-variable smooth fits the term ``z * f(x)`` — ``z`` is a *linear multiplier*
of the centred smooth basis, not a basis evaluation coordinate. The prediction
is therefore exactly **affine in ``z``** for any fixed ``x``:

    pred(x, z) = intercept + z * f(x)

so it must hold for every ``z`` — including ``z == 0`` (which zeroes the smooth
contribution, leaving the baseline) and ``z`` outside the sampled range.

Today the by-variable column is run through
``FittedModel::axis_clip_to_training_ranges`` (``src/inference/model.rs``) and
**clamped to ``[min(z), max(z)]`` from training**. The #806 fix exempted smooth
*basis* axes (and #792 the random-effect axes) from that clip, and
``collect_smooth_extrapolation_axes`` recurses into a ``ByVariable``'s *inner*
smooth so the ``x`` axis is exempt — but the by-multiplier column itself is left
in no exemption set, so it is clipped like an ordinary continuous predictor.

Consequences (training ``z`` in ``[1, 2]``, fixed ``x``):
* ``pred(x, z=0) == pred(x, z=1)`` — the natural ``z==0`` baseline is silently
  replaced by the ``z==min`` prediction.
* ``pred(x, z=3) == pred(x, z=2)`` — predictions plateau, slope 0, above the
  training max; the varying-coefficient effect stops growing.
* the prediction is no longer affine in ``z``, so any extrapolation in the
  modulating covariate is wrong (error grows without bound in ``z``).

The fix is to exempt the by-variable column from the predict-time clip (it is a
linear multiplier, exactly like a parametric ``linear`` axis, already exempt via
``training_linear_axes``). These assertions only require the affine-in-``z``
contract; they pass under any correct implementation.
"""
from __future__ import annotations

import numpy as np

import gamfit


def _model_and_x():
    rng = np.random.default_rng(7)
    n = 3000
    x = rng.uniform(-3.0, 3.0, n)
    z = rng.uniform(1.0, 2.0, n)  # by-variable training range [1, 2]
    y = 0.5 + z * np.sin(x) + rng.normal(0.0, 0.3, n)
    data = {"x": x, "z": z, "y": y}
    model = gamfit.fit(data, "y ~ s(x, by=z)", family="gaussian")
    return model


def _pred_at(model, x_fixed, zvals):
    zvals = np.asarray(zvals, dtype=float)
    grid = {"x": np.full(zvals.shape, float(x_fixed)), "z": zvals}
    return np.asarray(model.predict(grid))


def test_by_variable_prediction_is_affine_in_z():
    model = _model_and_x()
    x_fixed = 1.2

    # Per-unit slope of the effect, measured strictly INSIDE the training range
    # [1, 2] where no clip can act. This is the true coefficient f(x_fixed).
    p_lo, p_hi = _pred_at(model, x_fixed, [1.25, 1.75])
    slope = (p_hi - p_lo) / 0.5
    intercept = p_lo - slope * 1.25  # affine model fitted from interior points

    # The varying-coefficient term is affine in z, so predictions OUTSIDE the
    # training range must follow the same line. The clip makes them plateau.
    for z_query in (-0.5, 0.0, 3.0, 4.0):
        predicted = float(_pred_at(model, x_fixed, [z_query])[0])
        expected = intercept + slope * z_query
        assert abs(predicted - expected) < 0.05 * (1.0 + abs(expected)), (
            f"s(x, by=z) prediction is not affine in z: at x={x_fixed}, "
            f"z={z_query} got {predicted:.4f}, affine extension gives "
            f"{expected:.4f} (interior slope f(x)={slope:.4f}). The by-variable "
            f"is being clamped to the training range [1, 2]."
        )


def test_by_variable_zero_is_not_clamped_to_training_min():
    model = _model_and_x()
    x_fixed = 1.2
    p0, p_min = _pred_at(model, x_fixed, [0.0, 1.0])
    # z=0 zeroes the smooth term and must NOT collapse onto the z=min prediction.
    assert abs(p0 - p_min) > 1.0e-6, (
        f"s(x, by=z): pred(z=0)={p0:.6f} equals pred(z=min=1)={p_min:.6f}; the "
        "by-variable was clamped to the training minimum (z=0 should leave only "
        "the baseline)."
    )
