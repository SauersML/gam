"""Bug hunt: ``predict(observation_interval=True)`` is silently dropped for
binomial / Bernoulli models.

``Model.predict`` documents an ``observation_interval`` switch that, when
``True`` together with ``interval``, adds response-scale *prediction*-interval
columns ``observation_lower`` / ``observation_upper`` — the band
``Var(y_new|x) = Var(mu_hat) + Var(Y|mu)`` — "for families that support it"
(``gamfit/_model.py:96-102``). The engine *does* support binomial here: the
per-family observation-noise block of ``predict_gamwith_uncertainty`` has an
explicit ``ResponseFamily::Binomial`` arm that forms the conditional Bernoulli
variance ``p·(1-p)`` and emits the interval
(``src/inference/predict/mod.rs:5827-5835``). Gaussian, Poisson and Gamma all
return the two extra columns when asked.

Binomial returns neither column and raises no error. The request is dropped on
the floor.

Root cause (``crates/gam-pyffi/src/lib.rs``, ``predict_columns``): the engine is
dispatched on ``(options.interval, model.prediction_uses_posterior_mean())``.
``prediction_uses_posterior_mean`` is ``true`` for *exactly* the binomial family
(every link) plus the link-/baseline-wiggle models
(``src/inference/model.rs:2681-2692``). So a binomial fit with an interval takes
the ``(Some(level), true)`` match arm (``lib.rs:25389-25415``), which calls
``predict_posterior_mean`` — a path that has no observation-interval capability —
and **never references ``options.observation_interval``**. Only the sibling
``(Some(level), false)`` arm (``lib.rs:25416-25464``, taken by Poisson / Gamma /
Gaussian, whose ``prediction_uses_posterior_mean`` is ``false``) threads
``includeobservation_interval`` into ``predict_full_uncertainty`` and copies the
``observation_lower`` / ``observation_upper`` columns out. The capability the
engine implements for binomial is therefore unreachable through the API.

Why it matters: for an imbalanced / rare-event binomial (small ``p``) the
binomial observation interval is genuinely informative — the band
``mu ± z·sqrt(Var(mu_hat) + p(1-p))`` clamped to ``[0, 1]`` does not saturate
the whole unit interval — so silently dropping it loses a real, documented
feature, not just a degenerate Bernoulli edge case.

This test asserts the contract: a binomial model asked for an observation
interval must return the columns, and they must be a valid response-scale
prediction band (inside ``[0, 1]``, containing the point prediction, and no
narrower than the credible interval for the mean, because they add the
non-negative ``p(1-p)`` observation-variance term). It fails today (the columns
are absent) and will pass once the binomial / posterior-mean dispatch threads
``observation_interval`` through, the same way the Poisson / Gamma arm already
does. The Poisson control anchors that the feature is wired for some families,
so binomial's absence is a per-family gap rather than a missing global feature.

Related: #800 (Poisson observation interval crossed below support), #801 (Beta
observation interval ignored estimated phi), #802 (NegBin observation interval
froze theta) — that family addressed the *values* of observation intervals;
this one is the binomial interval being *absent entirely*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gamfit


def _fit_binomial() -> "gamfit.Model":
    rng = np.random.default_rng(6)
    n = 2000
    x = rng.uniform(0.0, 1.0, n)
    p = 1.0 / (1.0 + np.exp(-(-0.5 + 2.0 * x)))
    y = rng.binomial(1, p, n)
    return gamfit.fit(pd.DataFrame({"y": y, "x": x}), "y ~ s(x)", family="binomial")


def _grid() -> pd.DataFrame:
    return pd.DataFrame({"x": np.linspace(0.1, 0.9, 8)})


def test_binomial_predict_emits_observation_interval_columns() -> None:
    model = _fit_binomial()
    out = model.predict(_grid(), interval=0.95, observation_interval=True)
    cols = list(out.columns)

    # The core defect: the requested observation-interval columns are simply
    # missing (the `(Some, true)` posterior-mean dispatch arm never looks at
    # `observation_interval`). Both must be present.
    assert "observation_lower" in cols, (
        "binomial predict(observation_interval=True) dropped 'observation_lower' "
        f"silently; got columns {cols}"
    )
    assert "observation_upper" in cols, (
        "binomial predict(observation_interval=True) dropped 'observation_upper' "
        f"silently; got columns {cols}"
    )


def test_binomial_observation_interval_is_a_valid_response_band() -> None:
    model = _fit_binomial()
    out = model.predict(_grid(), interval=0.95, observation_interval=True)

    mean = np.asarray(out["mean"], dtype=float)
    mlo = np.asarray(out["mean_lower"], dtype=float)
    mhi = np.asarray(out["mean_upper"], dtype=float)
    olo = np.asarray(out["observation_lower"], dtype=float)
    ohi = np.asarray(out["observation_upper"], dtype=float)

    tol = 1e-9
    # Probability/proportion support: the observation band must stay in [0, 1].
    assert np.all(olo >= -tol) and np.all(ohi <= 1.0 + tol), (
        f"observation band escaped [0,1]: lower={olo}, upper={ohi}"
    )
    # A prediction interval must bracket its own point prediction.
    assert np.all(olo <= mean + tol) and np.all(ohi >= mean - tol), (
        "observation band does not contain the point prediction"
    )
    # Var(y_new) = Var(mu_hat) + p(1-p) >= Var(mu_hat), so the observation band
    # is never narrower than the credible interval for the mean.
    obs_width = ohi - olo
    mean_width = mhi - mlo
    assert np.all(obs_width >= mean_width - 1e-7), (
        "observation interval is narrower than the mean credible interval: "
        f"obs_width={obs_width}, mean_width={mean_width}"
    )


def test_poisson_observation_interval_present_control() -> None:
    # Control: the same request IS honoured for a family whose dispatch does not
    # route through the posterior-mean arm (Poisson is `uses_posterior_mean=False`).
    # This anchors that observation intervals are a real, wired feature, so the
    # binomial omission above is a per-family dispatch gap, not a global absence.
    rng = np.random.default_rng(6)
    n = 2000
    x = rng.uniform(0.0, 1.0, n)
    y = rng.poisson(np.exp(0.3 + x))
    model = gamfit.fit(pd.DataFrame({"y": y, "x": x}), "y ~ s(x)", family="poisson")
    out = model.predict(_grid(), interval=0.95, observation_interval=True)
    cols = list(out.columns)
    assert "observation_lower" in cols and "observation_upper" in cols, (
        f"Poisson control unexpectedly lacks observation columns; got {cols}"
    )
