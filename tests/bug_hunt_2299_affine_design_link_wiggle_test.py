"""Regression contract for #2299: fitted predictors have one typed affine API.

The old public design-matrix endpoint returned a bare matrix for ordinary GAMs
and rejected link-wiggle fits.  That shape encoded neither a model offset nor
the coefficient coordinate system.  A link-wiggle predictor is affine at its
fitted state as ``base + B(warp_index) @ beta_w``; critically, #2141 requires
``B`` to be evaluated at the saved frozen index, not at the de-aliased base.
"""

import numpy as np

import gamfit


def _linear_predictor(model, data) -> np.ndarray:
    prediction = model.predict(data, return_type="dict")
    return np.asarray(prediction["linear_predictor"], dtype=float)


def _assert_affine_identity(model, data, expected_frame: str) -> gamfit.AffineDesign:
    affine = model.design_matrix(data)
    assert isinstance(affine, gamfit.AffineDesign)
    assert affine.coefficient_frame == expected_frame
    assert affine.coefficient_start == 0
    assert affine.coefficient_stop == affine.coefficients.shape[0]
    assert affine.coefficient_slice == slice(0, affine.coefficients.shape[0])
    assert affine.offset.ndim == 1
    assert affine.matrix.ndim == 2
    assert affine.coefficients.ndim == 1
    assert affine.matrix.shape == (affine.offset.shape[0], affine.coefficients.shape[0])

    reconstructed = affine.offset + affine.matrix @ affine.coefficients
    expected = _linear_predictor(model, data)
    np.testing.assert_allclose(reconstructed, expected, rtol=2e-12, atol=2e-12)
    return affine


def test_ordinary_affine_design_exposes_model_offset_and_full_frame() -> None:
    rng = np.random.default_rng(2299)
    n = 160
    x = rng.uniform(-1.5, 1.5, n)
    offset = 0.35 * np.sin(1.7 * x) - 0.1
    y = 0.6 + 1.2 * x + offset + rng.normal(0.0, 0.08, n)
    data = {"y": y, "x": x, "offset": offset}

    model = gamfit.fit(data, "y ~ s(x)", family="gaussian", offset="offset")
    affine = _assert_affine_identity(model, data, "full")

    # The ordinary affine row offset is the supplied model offset itself; it is
    # not silently dropped into a bare X matrix.
    np.testing.assert_allclose(affine.offset, offset, rtol=0.0, atol=0.0)


def test_link_wiggle_affine_design_uses_saved_base_and_exact_2141_index() -> None:
    # Deterministic flexible-link repro inherited from #2141.  On this geometry
    # the de-alias shift is material: evaluating B at the base predictor instead
    # of the saved frozen index produced a dramatically different fitted link.
    rng = np.random.default_rng(0)
    n = 500
    x = rng.uniform(-2.0, 2.0, n)
    offset = 0.2 * np.sin(1.3 * x)
    eta = 0.5 + 1.5 * x + offset
    probability = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < probability).astype(float)
    data = {"y": y, "x": x, "offset": offset}

    model = gamfit.fit(
        data,
        "y ~ s(x)",
        family="binomial",
        offset="offset",
        flexible_link=True,
    )
    affine = _assert_affine_identity(model, data, "link_wiggle")

    # The link-wiggle row offset is the fitted base predictor, not the raw model
    # offset.  A coincidental equality would mean the Mean block was dropped.
    assert float(np.max(np.abs(affine.offset - offset))) > 0.1


def test_design_matrix_array_returns_the_same_typed_affine_contract() -> None:
    rng = np.random.default_rng(23)
    x = rng.normal(size=(96, 2))
    y = 0.4 + 0.8 * x[:, 0] - 0.3 * x[:, 1] + rng.normal(0.0, 0.05, 96)
    model = gamfit.fit_array(x, y, formula="y ~ x0 + x1", family="gaussian")

    affine = model.design_matrix_array(x)
    assert isinstance(affine, gamfit.AffineDesign)
    assert affine.coefficient_frame == "full"
    reconstructed = affine.offset + affine.matrix @ affine.coefficients
    expected = _linear_predictor(
        model,
        {"x0": x[:, 0], "x1": x[:, 1], "y": y},
    )
    np.testing.assert_allclose(reconstructed, expected, rtol=2e-12, atol=2e-12)
