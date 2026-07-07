"""Regression for #2141: a binomial flexible/learnable link must reconstruct the
SAME warped link at predict time that the fit scored its deviance with.

For every ordinary GLM family the reported training deviance is computed at the
model's fitted values, so recomputing the deviance from ``predict`` on the SAME
training rows reproduces it (the fixed-link control does, ratio ~1.00). With
``flexible_link=True`` the model reported Deviance ~412 while ``predict`` on the
same rows gave ~641 (ratio ~1.55): a model cannot both *fit* at deviance 412 and
*predict* the same rows at deviance 641.

Root cause (#2141): the frozen-basis de-aliased binomial-mean link fit pins the
monotone I-spline warp basis ``B`` at the frozen index ``η̂ = X·β`` and scores
``q = X·β_saved + B(η̂)·γ`` (with ``β_saved = β − A·γ`` the de-aliased mean). The
predict path evaluated the warp basis at the *de-aliased* base predictor
``X·β_saved`` instead of at ``η̂``; those differ by the identifiable projection
``X·A·γ`` (and, when the monotone re-fit fires, by the re-fit's mean movement),
so predict reconstructed a *different* link. The fix persists the frozen-index
shift ``s = β_frozen_source − β_saved`` (payload ``link_wiggle_index_shift``) so
predict evaluates the warp at ``η̂ = X·(β_saved + s)`` and reproduces the fitted
``q`` exactly.

Before the fix this test fails (~412 vs ~641); after it, fit and predict agree.
"""

import os

os.environ.setdefault("GAM_LOG", "off")

import numpy as np
import pytest

import gamfit


def _binom_dev(y, mu):
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, 1.0 - 1e-12)
    y = np.asarray(y, dtype=float)
    return float(-2.0 * np.sum(y * np.log(mu) + (1.0 - y) * np.log(1.0 - mu)))


def _reported_deviance(model):
    summary = model.summary()
    if isinstance(summary, dict):
        return float(summary["deviance"])
    return float(summary.deviance)


def test_flexible_link_predict_deviance_matches_fit_2141():
    # Deterministic repro from issue #2141 (logit data fit with a learnable link).
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(-2.0, 2.0, n)
    p = 1.0 / (1.0 + np.exp(-(0.5 + 1.5 * x)))
    y = (rng.uniform(size=n) < p).astype(float)
    data = {"y": y.tolist(), "x": x.tolist()}

    # Fixed-link control: predict(train) reproduces the reported deviance, as it
    # must for any ordinary GLM. Anchors the comparison so the flexible-link arm
    # is judged against the same standard.
    fixed = gamfit.fit(data, "y ~ s(x)", family="binomial")
    fixed_reported = _reported_deviance(fixed)
    fixed_predicted = _binom_dev(y, np.asarray(fixed.predict(data), dtype=float))
    assert abs(fixed_predicted - fixed_reported) < 0.05 + 1e-3 * abs(fixed_reported), (
        f"fixed-link control diverged: reported {fixed_reported}, "
        f"predicted {fixed_predicted}"
    )

    flex = gamfit.fit(data, "y ~ s(x)", family="binomial", flexible_link=True)
    flex_reported = _reported_deviance(flex)
    flex_mu = np.asarray(flex.predict(data), dtype=float)
    assert flex_mu.shape == (n,)
    assert np.all(np.isfinite(flex_mu))
    flex_predicted = _binom_dev(y, flex_mu)

    # The core contract (#2141): predict reconstructs the link the fit scored, so
    # the recomputed deviance matches the reported one closely. Before the fix
    # this gap was ~228 (412 vs 641); the frozen-index reconstruction closes it.
    assert abs(flex_predicted - flex_reported) < 1.0 + 1e-3 * abs(flex_reported), (
        f"predict reconstructed a different link than the fit: reported deviance "
        f"{flex_reported} vs predict-on-training deviance {flex_predicted} "
        f"(ratio {flex_predicted / flex_reported:.3f}); the warp basis is being "
        f"evaluated at a different index than the fit froze it at (#2141)."
    )
