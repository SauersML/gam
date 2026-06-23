"""Bug hunt: ``PosteriorSamples.predict_draws()`` (the posterior-predictive path)
drops the model offset — every drawn ``η`` is ``Xβ`` with the offset omitted, so
for a Poisson rate model the predictive mean comes out ``exp(-offset)`` times too
small.

A GLM fitted with ``offset=...`` targets the linear predictor ``η = Xβ + offset``.
The point-prediction path adds the offset back (its predictions match the data),
and the *coefficient* posterior is sampled against the offset target too (the
#882 fix: ``src/inference/sample.rs:516-543`` resolves and re-applies the offset
column). But the posterior-*predictive* evaluation at new points does not: the
FFI ``posterior_predict_table_impl`` builds ``eta = samples · Xᵀ``
(``crates/gam-pyffi/src/manifold_and_posterior_ffi.rs:454``) and never calls
``resolve_offset_column`` (contrast the sample path at the same file, line 313).
So the posterior predictive ``η`` is missing the offset entirely.

The effect is large and deterministic: with a Poisson log-link rate model whose
offset averages ~2 on the link scale, the posterior-predictive mean is
``exp(-2) ≈ 1/7`` of the correct value and matches the *offset-less* prediction
to machine precision, while the point prediction is correct.

This test fits a Poisson model with a sizeable offset, then asserts the
per-row posterior-predictive mean (a) tracks the offset-using point prediction
(correlation ≈ 1, since the offset dominates the row-to-row variation) and
(b) is the same order of magnitude as it. Both currently fail — the predictive
draws instead reproduce the offset-less prediction. When ``posterior_predict``
re-applies the offset, the assertions hold without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def test_posterior_predictive_uses_model_offset() -> None:
    rng = np.random.default_rng(5)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    # A genuine rate-model offset on the link scale (mean ~2), so dropping it
    # shifts the mean by exp(-2) ~ 7x and dominates the row-to-row spread.
    off = rng.uniform(1.0, 3.0, n)
    mu = np.exp(0.5 + 1.0 * x + off)
    y = rng.poisson(mu)
    frame = pd.DataFrame({"x": x, "y": y, "off": off})

    model = gamfit.fit(frame, "y ~ s(x)", family="poisson", offset="off")

    point = np.asarray(model.predict(frame)).ravel()  # uses the offset (correct)
    # Sanity: the point prediction recovers the offset-driven scale.
    assert 0.5 < point.mean() / mu.mean() < 2.0, (
        f"point prediction should track the true mean ({mu.mean():.1f}); "
        f"got {point.mean():.1f}"
    )

    samples = model.sample(frame, samples=300, chains=2, seed=1)
    predictive = samples.predict_draws(frame)
    pp_mean = np.asarray(predictive.mean).mean(axis=0)  # per-row posterior mean mu
    if pp_mean.shape[0] != n:
        pp_mean = np.asarray(predictive.mean).mean(axis=1)

    # (a) Structural check: the predictive must follow the offset-using point
    #     prediction, not the offset-less one. The offset spans exp(2)~7x, so if
    #     it is present the correlation is ~1; if dropped it collapses.
    corr = float(np.corrcoef(pp_mean, point)[0, 1])
    assert corr > 0.9, (
        "posterior-predictive mean must track the offset-using point prediction; "
        f"got correlation {corr:.3f} (offset appears to be dropped from predict_draws)"
    )

    # (b) Magnitude check: same order of magnitude as the point prediction
    #     (a small Jensen inflation from exp of the coefficient posterior is fine).
    ratio = pp_mean.mean() / point.mean()
    assert 0.5 < ratio < 2.0, (
        "posterior-predictive mean must be the same scale as the point prediction; "
        f"got ratio {ratio:.3f} (pp mean {pp_mean.mean():.2f} vs point {point.mean():.2f}) "
        "— consistent with the offset missing from the predictive eta"
    )
