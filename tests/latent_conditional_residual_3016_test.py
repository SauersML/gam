"""gam#3016: ``Model.latent_conditional_residual`` returns the declared conditional law's
standardized residual ``zeta = (z - m(a)) / sqrt(v(a))`` for new rows.

The fixture is a location-scale score on a raw scale, ``x, zeta ~ N(0, 1)`` independent and
``z = 3 + 2 * (0.6 * x + 0.8 * zeta)``, so the raw score moves with the context and the
residual does not. The held-out checks are calibrated: under the fitted law the standardized
correlation and mean of the residual are asymptotically N(0, 1), and a bound of 6 has a
two-sided false-alarm probability below 2e-9.
"""

from __future__ import annotations

import numpy as np
import pytest

import gamfit

M_SHIFT = 0.6
Z_BOUND = 6.0


def _sample(n: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n)
    zeta = rng.standard_normal(n)
    z = 3.0 + 2.0 * (M_SHIFT * x + np.sqrt(1.0 - M_SHIFT**2) * zeta)
    eta = -0.2 + 0.5 * x + 0.6 * zeta
    y = (rng.standard_normal(n) < eta).astype(float)
    return {"x": x, "z": z, "y": y}


def _fit(latent_measure: str):
    return gamfit.fit(
        _sample(1500, 3016),
        "y ~ x",
        family="bernoulli-marginal-slope",
        z_column="z",
        slope_formula="1",
        config={"latent_measure": latent_measure},
    )


@pytest.fixture(scope="module")
def conditional_model():
    return _fit("conditional-location-scale")


def _standardized_correlation(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1] * np.sqrt(a.size))


def test_the_residual_is_the_conditional_law_residual_on_held_out_rows(conditional_model) -> None:
    held_out = _sample(2000, 6103)
    frame = {"x": held_out["x"], "z": held_out["z"]}
    residual = conditional_model.latent_conditional_residual(frame)
    assert isinstance(residual, np.ndarray)
    assert residual.shape == (2000,)
    assert np.all(np.isfinite(residual))

    raw = _standardized_correlation(held_out["z"], held_out["x"])
    assert abs(raw) > Z_BOUND, f"the fixture's raw score must move with x (r*sqrt(n) = {raw:.2f})"
    conditional = _standardized_correlation(residual, held_out["x"])
    assert abs(conditional) < Z_BOUND, (
        f"the residual still moves with x: r*sqrt(n) = {conditional:.2f}"
    )
    centring = float(residual.mean() / residual.std(ddof=1) * np.sqrt(residual.size))
    assert abs(centring) < Z_BOUND, f"the residual is off centre: mean z = {centring:.2f}"


def test_a_reloaded_model_returns_the_same_residual(conditional_model) -> None:
    frame = {key: value for key, value in _sample(300, 77).items() if key != "y"}
    reloaded = gamfit.loads(conditional_model.dumps())
    assert np.array_equal(
        reloaded.latent_conditional_residual(frame),
        conditional_model.latent_conditional_residual(frame),
    )


def test_id_column_returns_a_residual_table(conditional_model) -> None:
    sample = _sample(40, 5)
    frame = {"id": [f"row{i}" for i in range(40)], "x": sample["x"], "z": sample["z"]}
    table = conditional_model.latent_conditional_residual(frame, id_column="id")
    plain = conditional_model.latent_conditional_residual({"x": sample["x"], "z": sample["z"]})
    assert list(table["id"]) == frame["id"]
    assert np.array_equal(np.asarray(table["residual"], dtype=float), plain)


def test_a_frame_without_the_score_column_is_a_schema_mismatch(conditional_model) -> None:
    with pytest.raises(gamfit.errors.SchemaMismatchError, match="'z'"):
        conditional_model.latent_conditional_residual({"x": np.zeros(5)})


def test_a_fit_without_a_conditional_law_returns_none() -> None:
    frame = {key: value for key, value in _sample(20, 9).items() if key != "y"}
    assert _fit("global-empirical").latent_conditional_residual(frame) is None
