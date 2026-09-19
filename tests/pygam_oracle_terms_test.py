"""pyGAM oracle: term construction (pygam/tests/test_terms.py, test_GAM_params.py).

Seeded synthetic stand-ins for pyGAM's datasets; pyGAM is never imported.
Tests that depend on a hand-supplied lambda, gridsearch or pyGAM internals are
deliberately not ported (bench/pygam_audit/pygam_tests.md section 3).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

import gamfit


def _width(model: Any, prefix: str) -> int:
    blocks = [b for b in model.term_blocks if str(b.name).startswith(prefix)]
    assert len(blocks) == 1, model.term_blocks
    return int(blocks[0].end - blocks[0].start)


def _r2(y: np.ndarray, fitted: np.ndarray) -> float:
    return float(1.0 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2))


@pytest.fixture(scope="module")
def smooth_1d() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(50)
    n = 150
    x = rng.uniform(2.0, 58.0, n)
    y = 5.0 + 40.0 * np.sin(x / 7.0) * np.exp(-x / 30.0) + rng.normal(0.0, 5.0, n)
    return {"x": x, "y": y}


@pytest.fixture(scope="module")
def surface() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(51)
    n = 600
    a = rng.uniform(0.0, 1.0, n)
    b = 0.5 * a + 0.5 * rng.uniform(0.0, 1.0, n)
    mu = np.exp(1.0 + np.sin(3.0 * a) * np.cos(4.0 * b))
    return {"a": a, "b": b, "y": rng.poisson(mu).astype(float)}


@pytest.fixture(scope="module")
def interaction() -> dict[str, np.ndarray]:
    """pyGAM's toy_interaction: y = x0 * sin(x1) + noise (n reduced to 3000)."""
    rng = np.random.default_rng(52)
    n = 3000
    x0 = rng.uniform(-5.0, 5.0, n)
    x1 = rng.uniform(-5.0, 5.0, n)
    return {"x0": x0, "x1": x1, "y": x0 * np.sin(x1) + rng.normal(0.0, 0.1, n)}


# test_terms::test_n_coefs / test_GAM_params::test_n_splines
def test_spline_block_width_is_k_minus_centering(smooth_1d: dict[str, np.ndarray]) -> None:
    m = gamfit.fit(smooth_1d, "y ~ s(x, k=10)")
    assert _width(m, "s(") == 9


def test_tensor_block_width_is_product_minus_centering(surface: dict[str, np.ndarray]) -> None:
    m = gamfit.fit(surface, "y ~ te(a, b, k=[5, 4])", family="poisson")
    assert _width(m, "te(") == 5 * 4 - 1


# test_GAM_params::test_intercept
def test_intercept_only_model_is_the_mean(smooth_1d: dict[str, np.ndarray]) -> None:
    m = gamfit.fit(smooth_1d, "y ~ 1")
    np.testing.assert_allclose(m.predict(smooth_1d), np.mean(smooth_1d["y"]), rtol=1e-12)


# test_GAM_params::test_linear_term
def test_unpenalized_linear_term_is_ols(smooth_1d: dict[str, np.ndarray]) -> None:
    m = gamfit.fit(smooth_1d, "y ~ linear(x)")
    coef = np.polyfit(smooth_1d["x"], smooth_1d["y"], 1)
    np.testing.assert_allclose(
        m.predict(smooth_1d), np.polyval(coef, smooth_1d["x"]), rtol=1e-9
    )


# test_terms::test_tensor_invariance_to_scaling (skipped as failing in pyGAM)
def test_tensor_fit_is_invariant_to_covariate_rescaling(surface: dict[str, np.ndarray]) -> None:
    m1 = gamfit.fit(surface, "y ~ te(a, b)", family="poisson")
    rescaled = dict(surface)
    rescaled["b"] = 100.0 * surface["b"] + 7.0
    m2 = gamfit.fit(rescaled, "y ~ te(a, b)", family="poisson")
    np.testing.assert_allclose(m1.predict(surface), m2.predict(rescaled), rtol=1e-4)
    assert float(m2.summary().edf_total) == pytest.approx(float(m1.summary().edf_total), rel=1e-3)


# test_terms::test_by_variable: a numeric by= smooth is a tensor with a linear margin.
def test_numeric_by_matches_tensor_with_linear_margin(interaction: dict[str, np.ndarray]) -> None:
    d = interaction
    by = gamfit.fit(d, "y ~ s(x1, by=x0)")
    te = gamfit.fit(d, "y ~ te(x0, x1, degree=[1, 3])")
    r2_by, r2_te = _r2(d["y"], by.predict(d)), _r2(d["y"], te.predict(d))
    assert r2_by > 0.99 and r2_te > 0.99
    assert abs(r2_by - r2_te) < 1e-3


# test_terms::test_by_variable_doesnt_exist
def test_missing_by_column_raises(smooth_1d: dict[str, np.ndarray]) -> None:
    with pytest.raises(Exception, match="nope"):
        gamfit.fit(smooth_1d, "y ~ s(x, by=nope)")


# test_terms::test_correct_smoothing_in_tensors
def test_reml_tensor_recovers_the_interaction(interaction: dict[str, np.ndarray]) -> None:
    d = interaction
    m = gamfit.fit(d, "y ~ te(x0, x1)")
    truth = d["x0"] * np.sin(d["x1"])
    assert _r2(truth, np.asarray(m.predict(d))) > 0.99


# test_terms::test_cyclic / test_GAM_params::test_cyclic_basis
def test_cyclic_smooth_is_periodic() -> None:
    rng = np.random.default_rng(53)
    n = 500
    x = rng.uniform(0.0, 24.0, n)
    y = np.sin(2.0 * np.pi * x / 24.0) + rng.normal(0.0, 0.2, n)
    m = gamfit.fit({"x": x, "y": y}, "y ~ cyclic(x, period=24)")
    base = np.asarray(m.predict({"x": np.array([0.0, 3.0, 11.5])}))
    shifted = np.asarray(m.predict({"x": np.array([24.0, 27.0, 35.5])}))
    np.testing.assert_allclose(shifted, base, rtol=0.0, atol=1e-10)


# test_GAM_params::test_cyclic_basis_on_non_cyclic
def test_cyclic_fits_worse_than_free_smooth_on_aperiodic_data() -> None:
    rng = np.random.default_rng(54)
    n = 120
    x = rng.uniform(0.0, 80.0, n)
    y = 100.0 * (1.0 - np.exp(-x / 25.0)) + rng.normal(0.0, 4.0, n)
    d = {"x": x, "y": y}
    cyc = gamfit.fit(d, "y ~ cyclic(x)").summary().deviance
    free = gamfit.fit(d, "y ~ s(x)").summary().deviance
    assert free is not None and cyc is not None and free < cyc


# test_terms::test_build_from_info / test_GAM_methods save-load
def test_save_load_round_trip_is_bit_exact(tmp_path: Path, smooth_1d: dict[str, np.ndarray]) -> None:
    m = gamfit.fit(smooth_1d, "y ~ s(x)")
    path = tmp_path / "m.gam"
    m.save(str(path))
    np.testing.assert_array_equal(gamfit.load(str(path)).predict(smooth_1d), m.predict(smooth_1d))
    np.testing.assert_array_equal(gamfit.loads(m.dumps()).predict(smooth_1d), m.predict(smooth_1d))


# test_terms::test_tensor_terms: gamfit documents broadcasting of a one-value k.
def test_tensor_single_k_broadcasts(surface: dict[str, np.ndarray]) -> None:
    a = gamfit.fit(surface, "y ~ te(a, b, k=[5])", family="poisson")
    b = gamfit.fit(surface, "y ~ te(a, b, k=5)", family="poisson")
    np.testing.assert_array_equal(a.predict(surface), b.predict(surface))
    assert _width(a, "te(") == 5 * 5 - 1


# test_utils::test_check_X_categorical_prediction_exceeds_training
def test_unseen_fixed_factor_level_at_predict_raises() -> None:
    rng = np.random.default_rng(55)
    n = 300
    age = rng.uniform(18.0, 80.0, n)
    edu = np.array(["a", "b", "c"])[rng.integers(0, 3, n)]
    y = np.sin(age / 10.0) + (edu == "b") * 1.0 + rng.normal(0.0, 0.3, n)
    m = gamfit.fit({"age": age, "edu": edu, "y": y}, "y ~ s(age) + factor(edu)")
    with pytest.raises(Exception, match="zzz"):
        m.predict({"age": np.array([40.0]), "edu": np.array(["zzz"])})
