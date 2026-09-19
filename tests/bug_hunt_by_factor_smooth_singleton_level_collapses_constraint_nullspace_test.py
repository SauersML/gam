"""A factor-by smooth whose level has a single row must fit, not abort.

``y ~ g + s(x, by=g)`` gives every level of ``g`` its own centred smooth beside
the factor main effect. When a level has one row, its gated smooth design is
one row of basis values; after centring against the gated level indicator
(which is exactly the main effect's column for that level), no function is
left for the smooth to carry. The collection's gauge used to report that as
``ConstraintNullspaceCollapsed`` and fail the whole fit (the pyGAM-audit term
fuzzer found it in every ``by=`` case with a singleton level).

The principled outcome is that the level's smooth keeps no coefficients: its
one row is fitted by the level's main-effect coefficient, and the other levels'
smooths are unaffected. This pins that: the fit succeeds, the singleton level
predicts a constant at its one observation (up to the noise scale: the level's
coefficient is estimated, not interpolated), the other levels still track their
curves, and intervals and the summary are finite.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

_ROWS_PER_LEVEL = (40, 40, 1)
_NOISE_SD = 0.1


def _sample(seed: int):
    rng = np.random.default_rng(seed)
    codes = np.concatenate(
        [np.full(count, level) for level, count in enumerate(_ROWS_PER_LEVEL)]
    )
    x = rng.uniform(0.0, 1.0, codes.size)
    truth = np.sin(2.0 * np.pi * x) * (1.0 + codes)
    y = truth + rng.normal(0.0, _NOISE_SD, codes.size)
    labels = np.array([f"L{c}" for c in codes])
    return pd.DataFrame({"x": x, "g": labels, "y": y})


@pytest.mark.parametrize("seed", [0, 1])
def test_singleton_by_level_fits_and_predicts(seed: int) -> None:
    data = _sample(seed)
    model = gamfit.fit(data, "y ~ g + s(x, by=g)")

    grid = np.linspace(0.05, 0.95, 7)
    single = data[data["g"] == "L2"]
    query = pd.DataFrame({"x": grid, "g": np.full(grid.size, "L2")})
    pred = np.asarray(model.predict(query), dtype=float).reshape(-1)
    # The singleton level has no smooth left: its prediction is the level's
    # main effect alone, flat in x and at its one observation.
    np.testing.assert_allclose(pred, pred[0], rtol=0.0, atol=1e-10)
    assert abs(pred[0] - float(single["y"].iloc[0])) < _NOISE_SD, (
        f"seed {seed}: singleton level predicts {pred[0]:.4f}, "
        f"its observation is {float(single['y'].iloc[0]):.4f}"
    )

    for level, amplitude in (("L0", 1.0), ("L1", 2.0)):
        query = pd.DataFrame({"x": grid, "g": np.full(grid.size, level)})
        pred = np.asarray(model.predict(query), dtype=float).reshape(-1)
        truth = amplitude * np.sin(2.0 * np.pi * grid)
        assert float(np.max(np.abs(pred - truth))) < 0.25 * amplitude, (
            f"seed {seed}: level {level} lost its curve beside a singleton level"
        )

    res = model.predict(
        data[["x", "g"]],
        interval=0.95,
        observation_interval=True,
        return_type="dict",
    )
    for key in ("posterior_mean_lower", "posterior_mean_upper"):
        assert np.all(np.isfinite(np.asarray(res[key], dtype=float))), key
    summary = model.summary()
    assert summary.edf_total is not None and np.isfinite(float(summary.edf_total))
