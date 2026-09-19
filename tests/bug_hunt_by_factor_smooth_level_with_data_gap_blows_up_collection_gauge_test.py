"""A factor-by smooth whose level has a data gap must stay a well-scaled fit.

``y ~ g + s(x, by=g)`` gives every level of ``g`` its own centred smooth. When a
level's rows cover only the ends of the ``x`` range, its gated design is nearly
rank deficient: the basis functions supported inside the gap see no data. The
collection's gauge used to whiten that design's Gram (``U Λ^{-1/2}``), dividing
each retained direction by the square root of its tiny eigenvalue. The span was
right but the chart was not: design entries on the gap reached hundreds, the
penalty in that chart spread over ten decades, and a Poisson fit's outer REML
solve could not certify (the pyGAM-audit term fuzzer found it as a
``DominatedCertifiedPlateau`` on ``fz0072``).

The gauge is now an orthonormal frame of the same span, so it cannot amplify a
design row. This pins that: both families fit; every design entry on a grid
through the gap is bounded by the raw basis scale; predictions there are finite
and stay on the scale of the data.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit

_FULL_ROWS = 40
_GAP_LEVEL = "C"
_GAP_LOW = 0.2
_GAP_HIGH = 0.8
_GAP_ROWS = (5, 4)
_NOISE_SD = 0.3
_UNIT_BASIS_SCALE = 1.0
_ORDER_OF_MAGNITUDE = 10.0


def _sample(seed: int, family: str):
    rng = np.random.default_rng(seed)
    frames = [
        pd.DataFrame({"x": rng.uniform(0.0, 1.0, _FULL_ROWS), "g": level})
        for level in ("A", "B")
    ]
    x_gap = np.concatenate(
        [
            rng.uniform(0.0, _GAP_LOW, _GAP_ROWS[0]),
            rng.uniform(_GAP_HIGH, 1.0, _GAP_ROWS[1]),
        ]
    )
    frames.append(pd.DataFrame({"x": x_gap, "g": _GAP_LEVEL}))
    data = pd.concat(frames, ignore_index=True)
    eta = 0.5 + 0.5 * np.sin(2.0 * np.pi * data["x"].to_numpy())
    if family == "poisson":
        data["y"] = rng.poisson(np.exp(eta)).astype(float)
    else:
        data["y"] = eta + rng.normal(0.0, _NOISE_SD, eta.size)
    return data


@pytest.mark.parametrize("family", ["gaussian", "poisson"])
@pytest.mark.parametrize("seed", [0, 1])
def test_gap_level_design_is_well_scaled_and_predicts_finitely(
    seed: int, family: str
) -> None:
    data = _sample(seed, family)
    model = gamfit.fit(data, "y ~ g + s(x, by=g)", family=family)

    grid = pd.DataFrame(
        {"x": np.linspace(0.0, 1.0, 21), "g": np.full(21, _GAP_LEVEL)}
    )
    design = np.asarray(model.design_matrix(grid).matrix, dtype=float)
    # The raw B-spline rows are a partition of unity (unit l2 bound) and the
    # indicator columns are 0/1, so an orthonormal chart keeps every entry on
    # the unit scale; the whitened chart put entries in the hundreds here.
    # The check is an order of magnitude, not a tolerance on the unit bound.
    largest = float(np.max(np.abs(design)))
    assert largest < _ORDER_OF_MAGNITUDE * _UNIT_BASIS_SCALE, (
        f"seed {seed} {family}: design entries on the gap reach {largest:.3g}"
    )

    pred = model.predict(grid, return_type="dict")
    mean = np.asarray(pred["posterior_mean"], dtype=float)
    assert np.all(np.isfinite(mean)), f"seed {seed} {family}: {mean}"
    y = data["y"].to_numpy()
    span = float(np.max(y) - np.min(y))
    assert float(np.max(np.abs(mean - np.mean(y)))) < 2.0 * span, (
        f"seed {seed} {family}: gap predictions leave the data scale: {mean}"
    )
