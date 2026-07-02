"""Regression test for issue #2077: the analytic weighted-Gaussian observation
(prediction) interval must be heteroscedastic in the per-row prior weights.

For a WEIGHTED Gaussian fit the conditional response variance is
``Var(y_i | mu_i) = sigma^2 / w_i``, so a weight-9 row's prediction interval
should be ``1/sqrt(9) = 1/3`` as wide as a weight-1 row. ``Model.sample_replicates``
was made weight-aware in #2025 (``sigma_i = sigma_hat / sqrt(w_i)``), but the
ANALYTIC ``predict(observation_interval=True)`` path broadcast a single pooled
``sigma_hat^2`` to every row — homoscedastic, weight-blind — so the two paths
reported contradictory predictive spreads for the same model and rows.

This test fits ``y ~ s(x)`` with heteroscedastic noise whose scale is driven by
the prior weights, then asks for the analytic observation interval at a weight-9
grid row and a weight-1 grid row. The ratio of the two interval widths must be
~1/3, NOT ~1.0 (the buggy weight-blind broadcast).
"""

import numpy as np

import gamfit


# 90% central interval => z = Phi^{-1}(0.95).
Z_90 = 1.6448536269514722


def test_weighted_gaussian_observation_interval_scales_with_prior_weights():
    rng = np.random.default_rng(2077)
    n = 6000
    x = rng.uniform(-2.0, 2.0, size=n)
    w = np.where(x < 0.0, 9.0, 1.0)
    # Var(y_i) = sigma^2 / w_i with sigma = 1  =>  sd_i = 1/sqrt(w_i).
    y = 2.0 + 0.7 * x + rng.normal(0.0, 1.0 / np.sqrt(w), size=n)

    data = {"x": x, "y": y, "w": w}
    model = gamfit.fit(data, "y ~ s(x)", weights="w")

    grid = {"x": [-1.0, 1.0], "w": [9.0, 1.0]}
    pred = model.predict(
        grid,
        interval=0.90,
        observation_interval=True,
        return_type="dict",
    )

    obs_lower = np.asarray(pred["observation_lower"], dtype=float)
    obs_upper = np.asarray(pred["observation_upper"], dtype=float)
    assert obs_lower.shape == (2,)
    assert obs_upper.shape == (2,)

    # Analytic observation-noise SD implied by the reported symmetric band.
    sd = (obs_upper - obs_lower) / (2.0 * Z_90)
    sd_w9, sd_w1 = sd[0], sd[1]
    assert sd_w9 > 0.0 and sd_w1 > 0.0

    ratio = sd_w9 / sd_w1
    # Var(y|mu) = sigma^2 / w  =>  width ratio = sqrt(w1/w9) = sqrt(1/9) = 1/3.
    # The weight-blind broadcast bug reports ratio ~ 1.0.
    assert 0.28 <= ratio <= 0.40, (
        f"weight-9 observation interval must be ~1/3 as wide as weight-1 "
        f"(Var(y|mu)=sigma^2/w_i, #2077); got width ratio {ratio:.4f} "
        f"(sd_w9={sd_w9:.4f}, sd_w1={sd_w1:.4f}). A ratio near 1.0 means the "
        f"analytic observation band ignored the per-row prior weights."
    )
