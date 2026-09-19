"""``y ~ s(x) + te(x, z)`` must predict after it fits.

The tensor's main effects overlap ``s(x)``, so the collection's gauge
reparameterizes the tensor and freezes that chart into the model: the tensor's
identifiability basis composed with the fit's spectral whitener. Prediction
rebuilds the tensor's null-function block ridges in the frozen chart. The
whitener's column scales span as many decades as the penalty spectrum, so a
relative rank cutoff on the primary penalty in those coordinates found spurious
null directions, the ANOVA blocks could never span them, and prediction raised
"tensor null blocks span 4 of the chart's N null directions" (the pyGAM-audit
term fuzzer found it in most concurvity cases). The blocks are now built on an
orthonormal frame of the chart's range, where the rank cutoff sees the true
null space.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _truth(x, z):
    return np.sin(2.0 * np.pi * x) + x * z


@pytest.mark.parametrize("seed", [0, 3])
def test_tensor_beside_margin_smooth_predicts(seed: int) -> None:
    n = 200
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    y = _truth(x, z) + rng.normal(0.0, 0.3, n)
    model = gamfit.fit(pd.DataFrame({"x": x, "z": z, "y": y}), "y ~ s(x) + te(x, z)")

    gx, gz = np.meshgrid(np.linspace(0.1, 0.9, 5), np.linspace(0.1, 0.9, 5))
    query = pd.DataFrame({"x": gx.ravel(), "z": gz.ravel()})
    pred = np.asarray(model.predict(query), dtype=float).reshape(-1)
    assert np.all(np.isfinite(pred))
    rmse = float(np.sqrt(np.mean((pred - _truth(query["x"], query["z"])) ** 2)))
    assert rmse < 0.3, f"seed {seed}: prediction RMSE {rmse:.3f} against the truth"

    # Rows predict the same alone as in a batch: the frozen chart, not the
    # query, decides the prediction design.
    alone = np.asarray(model.predict(query.iloc[:3]), dtype=float).reshape(-1)
    np.testing.assert_allclose(alone, pred[:3], rtol=0.0, atol=1e-10)
