"""Regression for #2162: terminal curv() κ polish must retreat on NaN covariance.

The data are deliberately low-signal Gaussian noise. Peer 2-D smooths can fit it,
so the constant-curvature smooth must not abort just because the terminal local
κ polish reaches an inference-only NaN covariance at an infeasible trial point.
"""

import numpy as np
import pandas as pd

import gamfit


def _low_signal_frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    return pd.DataFrame(
        {
            "x": rng.uniform(0.01, 0.99, 400),
            "z": rng.uniform(0.01, 0.99, 400),
            "y": rng.standard_normal(400),
        }
    )


def test_peer_2d_smooths_fit_the_same_low_signal_data():
    df = _low_signal_frame()

    for formula in ("y ~ te(x, z)", "y ~ mjs(x, z)", "y ~ s(x) + s(z)"):
        model = gamfit.fit(df, formula, family="gaussian")
        pred = np.asarray(model.predict(df))
        assert pred.shape[0] == len(df)
        assert np.all(np.isfinite(pred))


def test_curv_smooth_fits_low_signal_data_without_aborting():
    df = _low_signal_frame()

    model = gamfit.fit(df, "y ~ curv(x, z)", family="gaussian")
    pred = np.asarray(model.predict(df))

    assert pred.shape[0] == len(df)
    assert np.all(np.isfinite(pred))
