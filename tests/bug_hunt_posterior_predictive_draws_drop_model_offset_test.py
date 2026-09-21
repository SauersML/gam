"""Bug hunt: ``PosteriorSamples.predict_draws()`` (the posterior-predictive path)
drops the model offset — every drawn ``η`` is ``Xβ`` with the offset omitted, so
for a Poisson rate model the predictive mean comes out ``exp(-offset)`` times too
small.

A GLM fitted with ``offset=...`` targets the linear predictor ``η = Xβ + offset``.
The point-prediction path adds the offset back (its predictions match the data),
and the *coefficient* posterior is sampled against the offset target too (the
#882 fix: ``src/inference/sample.rs:516-543`` resolves and re-applies the offset
column). But the posterior-*predictive* evaluation at new points did not: the
FFI ``posterior_predict_table_impl`` builds ``eta = samples · Xᵀ``
(``crates/gam-pyffi/src/manifold_and_posterior_ffi.rs:454``) and never called
``resolve_offset_column`` (contrast the sample path at the same file, line 313).
So the posterior predictive ``η`` was missing the offset entirely.

The effect is large and deterministic: with a Poisson log-link rate model whose
offset averages ~2 on the link scale, the posterior-predictive mean is
``exp(-2) ≈ 1/7`` of the correct value and matches the *offset-less* prediction
to machine precision, while the point prediction is correct.

The gate is the offset's defining identity rather than a magnitude band (#4532).
An offset enters only as an additive term of ``η``, so shifting it by a constant
``c`` on the frame handed to a predictor must shift every predicted ``η`` by
exactly ``c`` and multiply every predicted mean by exactly ``exp(c)``. That is
exact arithmetic, not statistics: the bars below are floating-point accumulation
bands of order 1e-14, so they fail for a dropped offset, an offset applied at the
wrong scale, an offset applied on the response scale instead of the link scale,
and an offset read from the training frame instead of ``new_data`` — including
the mistakes small enough to land inside a factor of two of the truth, which the
ratio bands this test used to carry accepted.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

#: Link-scale constant added to the offset column of the prediction frame. Any
#: nonzero value works; 1.0 makes the expected response-scale factor ``e``.
SHIFT = 1.0


def _link_accumulation_band(n_terms: int, magnitude: float) -> float:
    """Absolute band on the difference between two linear predictors that
    differ only in the value of their offset term.

    ``η`` is a dot product of ``n_terms`` summands (the design columns plus the
    offset). Changing one summand perturbs the rounding of every partial sum
    after it, so the two runs' results differ by at most the standard
    ``n_terms · eps · max|partial sum|`` accumulation band, and ``magnitude``
    is an upper bound on that partial-sum magnitude. Everything else — the
    design matrix, the frozen basis, the coefficient draws — is bit-identical
    between the two calls, so nothing else contributes.
    """
    return float(n_terms) * float(np.finfo(float).eps) * float(magnitude)


def _response_relative_band(link_band: float) -> float:
    """Relative band on ``exp(η + c)`` against ``exp(c) · exp(η)``.

    On top of ``link_band`` (which passes through ``exp`` as a relative error)
    this allows one unit in the last place for each of the two ``exp``
    evaluations and for ``exp(c)``, plus half a unit for the product: four
    rounding units in all.
    """
    return link_band + 4.0 * float(np.finfo(float).eps)


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
    shifted_frame = frame.assign(off=frame["off"] + SHIFT)

    model = gamfit.fit(frame, "y ~ s(x)", family="poisson", offset="off")

    point = np.asarray(model.predict(frame), dtype=float).ravel()
    n_terms = len(model.summary().coefficients) + 1

    # (a) The point path carries the offset, and carries it additively on the
    #     link scale. `predict` on a frame whose offset is `off + SHIFT` must
    #     return exactly `exp(SHIFT)` times the original prediction. A dropped
    #     offset returns the original unchanged (factor 1), an offset applied
    #     at scale s returns `exp(s * SHIFT)`, and an offset read from the
    #     training frame returns the original unchanged.
    point_shifted = np.asarray(model.predict(shifted_frame), dtype=float).ravel()
    point_band = _response_relative_band(
        _link_accumulation_band(
            n_terms, float(np.abs(np.log(point)).max() + np.abs(off).max() + SHIFT)
        )
    )
    np.testing.assert_allclose(
        point_shifted,
        np.exp(SHIFT) * point,
        rtol=point_band,
        atol=0.0,
        err_msg=(
            "predict() must shift the linear predictor by exactly the change in "
            "the offset column of the frame it is given"
        ),
    )

    samples = model.sample(frame, samples=300, seed=1)
    predictive = samples.predict_draws(frame)
    eta = np.asarray(predictive.eta, dtype=float)
    mean = np.asarray(predictive.mean, dtype=float)
    assert eta.shape == (samples.n_draws, n), (
        f"posterior-predictive matrices are (draws, rows); got {eta.shape} for "
        f"{samples.n_draws} draws and {n} rows"
    )
    pp_mean = mean.mean(axis=0)  # per-row posterior mean mu

    # (b) Structural check: the predictive must follow the offset-using point
    #     prediction, not the offset-less one. The offset spans exp(2)~7x, so if
    #     it is present the correlation is ~1; if dropped it collapses.
    corr = float(np.corrcoef(pp_mean, point)[0, 1])
    assert corr > 0.9, (
        "posterior-predictive mean must track the offset-using point prediction; "
        f"got correlation {corr:.3f} (offset appears to be dropped from predict_draws)"
    )

    # (c) The same additivity identity on the predictive path, evaluated on the
    #     SAME draw matrix (`samples` is reused, so every coefficient draw is
    #     bit-identical between the two calls). On the link scale the identity
    #     is a pure addition; on the response scale it is a multiplication by
    #     `exp(SHIFT)`. Both are asserted: the link check localizes a failure to
    #     the offset itself, the response check pins the scale the defect moved.
    shifted = samples.predict_draws(shifted_frame)
    shifted_eta = np.asarray(shifted.eta, dtype=float)
    shifted_mean = np.asarray(shifted.mean, dtype=float)

    eta_band = _link_accumulation_band(
        n_terms, float(np.abs(eta).max() + np.abs(off).max() + SHIFT)
    )
    np.testing.assert_allclose(
        shifted_eta,
        eta + SHIFT,
        rtol=0.0,
        atol=eta_band,
        err_msg=(
            "predict_draws() must add the offset of the frame it is given to "
            "every drawn linear predictor"
        ),
    )
    np.testing.assert_allclose(
        shifted_mean,
        np.exp(SHIFT) * mean,
        rtol=_response_relative_band(eta_band),
        atol=0.0,
        err_msg=(
            "predict_draws() response-scale means must scale by exp of the "
            "change in the offset column"
        ),
    )
