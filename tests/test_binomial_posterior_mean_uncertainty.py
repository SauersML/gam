"""Regression tests for the binomial / posterior-mean predict dispatch arm.

Issues #811 and #812 were two symptoms of the same root cause: the
``(interval=Some, uses_posterior_mean=true)`` dispatch arm in
``crates/gam-pyffi/src/lib.rs`` — the arm taken by *every* binomial / Bernoulli
model (it is the only standard family whose default point is the posterior mean
``E[g⁻¹(η)]``) — silently dropped two documented options:

* ``observation_interval=True`` produced no ``observation_lower`` /
  ``observation_upper`` columns (#811), even though the engine implements the
  Bernoulli observation band ``p·(1−p)``; and
* ``covariance_mode`` was inert (#812): ``"conditional"`` and ``"smoothing"``
  returned bitwise-identical SEs, so the smoothing-parameter
  correction ``J·Var(ρ̂)·Jᵀ`` that every other family includes by default was
  never applied and binomial intervals systematically under-covered.

The companion bug-hunt files
(``test_bug_hunt_binomial_observation_interval_dropped.py`` and
``test_bug_hunt_binomial_covariance_mode_ignored.py``) pin the two reported
symptoms. This file attacks the same fix from independent angles so a regression
of the root cause is caught even if those exact assertions drift:

* the *point* estimate is invariant to ``covariance_mode`` /
  ``observation_interval`` (issue #398 generalised — the fix must add columns,
  never move ``mean`` / ``linear_predictor``);
* ``covariance_mode="smoothing"`` genuinely forms the corrected covariance
  instead of silently substituting the conditional covariance;
* the response-scale *credible band* — not merely ``std_error`` — widens under
  smoothing (the bounds consume the covariance-mode SE);
* the rare-event Bernoulli observation band is genuinely informative
  (``p(1−p)`` does not saturate ``[0, 1]`` for small ``p``); and
* the observation columns appear for non-default links (probit, cloglog), since
  the dispatch gap was per-family, not per-link.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def _fit_binomial(seed: int = 0, n: int = 2000, link: str | None = None) -> "gamfit.Model":
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-(-0.5 + 2.0 * x)))
    y = rng.binomial(1, p, n)
    frame = pd.DataFrame({"y": y, "x": x})
    if link is None:
        return gamfit.fit(frame, "y ~ s(x)", family="binomial")
    return gamfit.fit(frame, "y ~ s(x)", family="binomial", link=link)


def _grid(npts: int = 8) -> pd.DataFrame:
    return pd.DataFrame({"x": np.linspace(0.1, 0.9, npts)})


def test_point_invariant_to_covariance_mode_and_observation() -> None:
    # Issue #398 generalised to the new knobs: the reported point estimate is a
    # property of the model and inputs only. Adding an interval, choosing a
    # covariance mode, or requesting an observation band must add/alter columns
    # but never move `mean` or `linear_predictor`. (The posterior-mean point
    # always integrates the conditional posterior, regardless of mode.)
    model = _fit_binomial()
    grid = _grid()
    # Plain ``predict`` (no interval) returns the response-mean array; the
    # linear predictor is available via the dict return type.
    ref_mean = np.asarray(model.predict(grid), dtype=float)
    ref_lp = np.asarray(
        model.predict(grid, return_type="dict")["linear_predictor"], dtype=float
    )

    for cm in ("conditional", "smoothing"):
        for obs in (False, True):
            out = model.predict(
                grid, interval=0.95, covariance_mode=cm, observation_interval=obs
            )
            np.testing.assert_allclose(
                np.asarray(out["mean"], dtype=float),
                ref_mean,
                atol=1e-9,
                rtol=0.0,
                err_msg=f"mean moved for covariance_mode={cm}, observation={obs}",
            )
            np.testing.assert_allclose(
                np.asarray(out["linear_predictor"], dtype=float),
                ref_lp,
                atol=1e-9,
                rtol=0.0,
                err_msg=f"linear_predictor moved for covariance_mode={cm}, observation={obs}",
            )


def test_smoothing_covariance_mode_is_nontrivial() -> None:
    # `covariance_mode="smoothing"` must form the smoothing-corrected covariance
    # for a REML-selected smooth, never silently substitute the conditional
    # covariance. The conditional SE must be strictly smaller somewhere,
    # proving the correction is non-trivial.
    model = _fit_binomial(seed=2)
    grid = _grid(12)

    def se(mode: str) -> np.ndarray:
        return np.asarray(
            model.predict(grid, interval=0.95, covariance_mode=mode)["std_error"],
            dtype=float,
        )

    cond, smooth = se("conditional"), se("smoothing")
    assert np.all(smooth >= cond - 1e-12)
    assert np.any(smooth > cond + 1e-9), (
        "smoothing correction added no variance anywhere — covariance_mode is inert"
    )


def test_mean_credible_band_widens_under_smoothing() -> None:
    # The fix must thread covariance_mode into the *bounds*, not only std_error:
    # the response-scale credible band is built from the covariance-mode SE, so
    # it must be no narrower under smoothing and strictly wider somewhere.
    model = _fit_binomial(seed=4)
    grid = _grid(12)
    cond = model.predict(grid, interval=0.95, covariance_mode="conditional")
    smooth = model.predict(grid, interval=0.95, covariance_mode="smoothing")
    cond_w = np.asarray(cond["mean_upper"], dtype=float) - np.asarray(
        cond["mean_lower"], dtype=float
    )
    smooth_w = np.asarray(smooth["mean_upper"], dtype=float) - np.asarray(
        smooth["mean_lower"], dtype=float
    )
    assert np.all(smooth_w >= cond_w - 1e-9)
    assert np.any(smooth_w > cond_w + 1e-9), (
        "credible band did not widen under smoothing — bounds ignore covariance_mode"
    )


def test_rare_event_observation_band_is_informative() -> None:
    # The substantive point of #811: for an imbalanced / rare-event binomial the
    # Bernoulli observation band `p(1−p)` does NOT saturate the unit interval, so
    # dropping it loses real information. Fit a small-p model and check the upper
    # observation bound stays well below 1 while still bracketing the point.
    rng = np.random.default_rng(3)
    n = 4000
    x = rng.uniform(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-(-3.2 + 0.8 * x)))  # p ≈ 0.035–0.075
    y = rng.binomial(1, p, n)
    model = gamfit.fit(pd.DataFrame({"y": y, "x": x}), "y ~ s(x)", family="binomial")

    out = model.predict(_grid(), interval=0.95, observation_interval=True)
    mean = np.asarray(out["mean"], dtype=float)
    olo = np.asarray(out["observation_lower"], dtype=float)
    ohi = np.asarray(out["observation_upper"], dtype=float)

    tol = 1e-9
    assert np.all(olo >= -tol) and np.all(ohi <= 1.0 + tol)
    assert np.all(olo <= mean + tol) and np.all(ohi >= mean - tol)
    # For p well below 0.5 the band must not be the trivial saturated [0, 1].
    assert np.all(ohi < 0.9), f"rare-event observation band saturated to ~1: {ohi}"
    # Observation band is never narrower than the mean credible band (it adds the
    # non-negative p(1-p) observation-noise term on top of estimation variance).
    obs_w = ohi - olo
    mean_w = np.asarray(out["mean_upper"], dtype=float) - np.asarray(
        out["mean_lower"], dtype=float
    )
    assert np.all(obs_w >= mean_w - 1e-7)


def test_observation_interval_emitted_for_nondefault_links() -> None:
    # The dispatch gap was per-family (every binomial link routes through the
    # posterior-mean arm), not per-link: probit and cloglog must emit the columns
    # too, and the band must be a valid [0, 1] response band.
    for link in ("probit", "cloglog"):
        model = _fit_binomial(seed=7, link=link)
        out = model.predict(_grid(6), interval=0.95, observation_interval=True)
        cols = list(out.columns)
        assert "observation_lower" in cols and "observation_upper" in cols, (
            f"binomial link={link} dropped observation columns; got {cols}"
        )
        olo = np.asarray(out["observation_lower"], dtype=float)
        ohi = np.asarray(out["observation_upper"], dtype=float)
        mean = np.asarray(out["mean"], dtype=float)
        tol = 1e-9
        assert np.all(olo >= -tol) and np.all(ohi <= 1.0 + tol)
        assert np.all(olo <= mean + tol) and np.all(ohi >= mean - tol)


def test_observation_columns_absent_unless_requested() -> None:
    # The schema must stay clean when observation intervals are not requested:
    # the two columns appear only on an explicit observation_interval=True.
    model = _fit_binomial(seed=9)
    plain = model.predict(_grid(), interval=0.95)
    assert "observation_lower" not in plain.columns
    assert "observation_upper" not in plain.columns
    with_obs = model.predict(_grid(), interval=0.95, observation_interval=True)
    assert "observation_lower" in with_obs.columns
    assert "observation_upper" in with_obs.columns
