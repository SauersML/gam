"""Offset, prior-weight and expectile-asymmetry argument hygiene.

pyGAM audit findings (bench/pygam_audit):

* **F9** — an all-zero or 0/1 offset column was stored as a ``binary`` schema
  column at fit time, so any other offset at predict time was refused
  (``column 'off' is binary in schema``). An offset or a prior weight is
  real-valued by role, whatever values the training rows held.
* **F10** — ``expectile_tau`` was silently ignored unless the family was
  expectile, and ``expectile_tau=1.5`` was accepted there. The asymmetry is a
  parameter of the expectile family: any other family, or ``tau`` outside
  ``(0, 1)``, is a typed configuration error.
* **F13** — zero-weight rows still moved the fit through the data-dependent
  basis and constraint quantities (knots, ranges, centering), so a weight of
  zero differed from deleting the row (λ about 1 %, predictions up to 7e-4).
  Weight zero is now exactly row deletion.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any, cast

import numpy as np

pytest = cast(Any, import_module("pytest"))
gamfit = cast(Any, import_module("gamfit"))

GRID = {"x": np.linspace(0.05, 0.95, 41)}


def _poisson_data(seed: int = 0, n: int = 300) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    y = rng.poisson(np.exp(0.5 + np.sin(6.0 * x))).astype(float)
    return {"x": x, "y": y}


# ---------------------------------------------------------------------------
# F9: offset / weight role columns are continuous in the saved schema.
# ---------------------------------------------------------------------------


def test_zero_offset_at_fit_accepts_any_offset_at_predict() -> None:
    data = _poisson_data()
    data["off"] = np.zeros(data["x"].size)
    model = gamfit.fit(data, "y ~ s(x)", family="poisson", offset="off")
    at_zero = model.predict(dict(GRID, off=np.zeros(GRID["x"].size)))
    at_log2 = model.predict(dict(GRID, off=np.full(GRID["x"].size, np.log(2.0))))
    np.testing.assert_allclose(at_log2 / at_zero, 2.0, rtol=1e-12, atol=0.0)


def test_zero_one_offset_at_fit_accepts_a_fractional_offset_at_predict() -> None:
    data = _poisson_data(seed=1)
    rng = np.random.default_rng(11)
    data["off"] = (rng.uniform(size=data["x"].size) > 0.5).astype(float)
    model = gamfit.fit(data, "y ~ s(x)", family="poisson", offset="off")
    at_zero = model.predict(dict(GRID, off=np.zeros(GRID["x"].size)))
    at_half = model.predict(dict(GRID, off=np.full(GRID["x"].size, 0.5)))
    np.testing.assert_allclose(at_half / at_zero, np.exp(0.5), rtol=1e-12, atol=0.0)


# ---------------------------------------------------------------------------
# F10: expectile_tau is a parameter of the expectile family, in (0, 1).
# ---------------------------------------------------------------------------


def _gaussian_data(seed: int = 3, n: int = 200) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    return {"x": x, "y": np.sin(6.0 * x) + rng.normal(0.0, 0.3, n)}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expectile_tau": 0.9},
        {"family": "auto", "expectile_tau": 0.9},
        {"family": "gaussian", "expectile_tau": 0.9},
        {"family": "poisson", "expectile_tau": 0.5},
    ],
)
def test_expectile_tau_with_another_family_is_refused(kwargs: dict[str, Any]) -> None:
    data = _gaussian_data()
    if kwargs.get("family") == "poisson":
        data["y"] = np.abs(np.round(data["y"] * 3.0))
    with pytest.raises(gamfit.errors.InvalidConfigurationError, match="expectile_tau"):
        gamfit.fit(data, "y ~ s(x)", **kwargs)


@pytest.mark.parametrize("tau", [0.0, 1.0, 1.5, -0.2])
def test_expectile_tau_outside_the_open_unit_interval_is_refused(tau: float) -> None:
    with pytest.raises(gamfit.errors.InvalidConfigurationError, match=r"strictly in \(0, 1\)"):
        gamfit.fit(_gaussian_data(), "y ~ s(x)", family="expectile", expectile_tau=tau)


@pytest.mark.parametrize("tau", [float("nan"), float("inf")])
def test_non_finite_expectile_tau_is_refused(tau: float) -> None:
    # A non-finite float has no JSON spelling, so the wire request itself is
    # refused before field validation; it is still a configuration error.
    with pytest.raises(gamfit.errors.InvalidConfigurationError):
        gamfit.fit(_gaussian_data(), "y ~ s(x)", family="expectile", expectile_tau=tau)


def test_expectile_tau_with_the_expectile_family_still_fits() -> None:
    data = _gaussian_data()
    explicit = gamfit.fit(data, "y ~ s(x)", family="expectile", expectile_tau=0.9)
    inline = gamfit.fit(data, "y ~ s(x)", family="expectile(0.9)")
    np.testing.assert_allclose(explicit.predict(GRID), inline.predict(GRID), rtol=1e-12)


# ---------------------------------------------------------------------------
# F13: a zero prior weight is exactly row deletion.
# ---------------------------------------------------------------------------


def _zero_weight_problem(family: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(8)
    n = 300
    x = rng.uniform(0.0, 1.0, n)
    w = (rng.uniform(size=n) > 0.3).astype(float)
    # The zero-weight rows include the extreme covariate values, so the
    # training range, knots and centering of the weighted fit would all see
    # rows that deletion removes.
    w[np.argmin(x)] = 0.0
    w[np.argmax(x)] = 0.0
    if family == "poisson":
        y = rng.poisson(np.exp(0.5 + np.sin(6.0 * x))).astype(float)
        y[w == 0] = 40.0
    else:
        y = np.sin(6.0 * x) + rng.normal(0.0, 0.3, n)
        y[w == 0] = 1e3
    return {"x": x, "y": y, "w": w}, w > 0


def _assert_same_fit(weighted: Any, deleted: Any, grid: dict[str, Any]) -> None:
    np.testing.assert_allclose(
        weighted.predict(grid), deleted.predict(grid), rtol=1e-10, atol=1e-12
    )
    sw, sd = weighted.summary(), deleted.summary()
    np.testing.assert_allclose(sw.lambdas, sd.lambdas, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(sw.edf_total, sd.edf_total, rtol=1e-10, atol=0.0)
    assert sw.n_obs == sd.n_obs


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
@pytest.mark.parametrize(
    "formula", ["y ~ s(x)", "y ~ s(x, knot_placement=uniform)", "y ~ s(x, bs='cr')"]
)
def test_zero_weight_rows_are_exactly_deleted_rows(family: str, formula: str) -> None:
    data, keep = _zero_weight_problem(family)
    weighted = gamfit.fit(data, formula, family=family, weights="w")
    deleted = gamfit.fit({"x": data["x"][keep], "y": data["y"][keep]}, formula, family=family)
    _assert_same_fit(weighted, deleted, GRID)


def test_zero_weight_rows_drop_factor_levels_they_alone_carry() -> None:
    data, keep = _zero_weight_problem("gaussian")
    rng = np.random.default_rng(21)
    g = np.where(rng.uniform(size=keep.size) > 0.5, "a", "b").astype(object)
    # Level "c" appears only on zero-weight rows: deleting them removes it.
    g[np.flatnonzero(~keep)[:5]] = "c"
    data["g"] = g
    formula = "y ~ s(x) + g"
    weighted = gamfit.fit(data, formula, weights="w")
    deleted = gamfit.fit(
        {"x": data["x"][keep], "y": data["y"][keep], "g": g[keep]}, formula
    )
    grid = dict(GRID, g=np.where(np.arange(GRID["x"].size) % 2 == 0, "a", "b").astype(object))
    _assert_same_fit(weighted, deleted, grid)


def test_all_zero_weights_are_refused() -> None:
    data, _ = _zero_weight_problem("gaussian")
    data["w"] = np.zeros_like(data["w"])
    with pytest.raises(gamfit.errors.InvalidConfigurationError, match="zero on every row"):
        gamfit.fit(data, "y ~ s(x)", weights="w")
