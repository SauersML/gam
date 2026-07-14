"""Public regression for the isotropic Duchon frame (gam#2319)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def _rotated_problem() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(0)
    n = 400
    radius = 3.0
    angle = rng.uniform(0.0, 2.0 * np.pi, n)
    radial = radius * np.sqrt(rng.uniform(0.0, 1.0, n))
    x = np.column_stack((radial * np.cos(angle), radial * np.sin(angle)))
    r = np.linalg.norm(x, axis=1)
    y = (
        2.0 * np.exp(-((r - radius / 3.0) ** 2) / (0.6 * (radius / 3.0) ** 2))
        + (1.2 / radius) * r
        + rng.normal(0.0, 0.03, n)
    )

    grid = np.linspace(-2.0, 2.0, 9)
    gx, gz = np.meshgrid(grid, grid)
    query = np.column_stack((gx.ravel(), gz.ravel()))

    theta = 0.698
    c, s = np.cos(theta), np.sin(theta)
    rotation = np.array(((c, -s), (s, c)))
    center = x.mean(axis=0)
    x_rot = (x - center) @ rotation.T + center
    query_rot = (query - center) @ rotation.T + center

    train = pd.DataFrame({"x": x[:, 0], "z": x[:, 1], "y": y})
    train_rot = pd.DataFrame({"x": x_rot[:, 0], "z": x_rot[:, 1], "y": y})
    pred = pd.DataFrame({"x": query[:, 0], "z": query[:, 1]})
    pred_rot = pd.DataFrame({"x": query_rot[:, 0], "z": query_rot[:, 1]})
    return train, train_rot, pred, pred_rot


def test_duchon_fit_is_rotation_equivariant_in_one_scalar_frame_2319() -> None:
    train, train_rot, pred, pred_rot = _rotated_problem()
    fit = gamfit.fit(train, "y ~ duchon(x, z)", family="gaussian")
    fit_rot = gamfit.fit(train_rot, "y ~ duchon(x, z)", family="gaussian")

    mean = np.asarray(fit.predict(pred), dtype=float)
    mean_rot = np.asarray(fit_rot.predict(pred_rot), dtype=float)
    signal_range = float(np.ptp(mean))
    assert signal_range > 1.0
    prediction_defect = float(np.max(np.abs(mean - mean_rot)) / signal_range)

    # A scalar isotropic input frame, an equivariant center set, and an exact
    # REML solve describe the same statistical problem after a rigid rotation.
    # The comparison is against the optimizer's numerical resolution, not the
    # loose 0.18 downstream classification band that exposed the regression.
    assert prediction_defect <= 5.0e-6, prediction_defect

    summary = fit.summary()
    summary_rot = fit_rot.summary()
    np.testing.assert_allclose(
        summary_rot.edf_total,
        summary.edf_total,
        rtol=5.0e-6,
        atol=5.0e-8,
    )
    np.testing.assert_allclose(
        summary_rot.reml_score,
        summary.reml_score,
        rtol=5.0e-8,
        atol=5.0e-8,
    )

    smooth = fit.smoothing_parameters()
    smooth_rot = fit_rot.smoothing_parameters()
    assert smooth.keys() == smooth_rot.keys()
    np.testing.assert_allclose(
        [smooth_rot[key] for key in smooth],
        [smooth[key] for key in smooth],
        rtol=5.0e-6,
        atol=5.0e-8,
    )
