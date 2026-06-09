"""Bug hunt: a global smooth + factor smooth on the same covariate cannot be
predicted — the fit residualizes the overlapping term but predict rebuilds the
full, un-residualized design.

`docs/difference-smooths.md` recommends two idioms that put a *global* smooth
and a *factor* smooth on the same covariate:

    y ~ s(x) + s(g, x, bs=sz)     # "Sum-to-zero factor smooths"
    y ~ s(x) + s(x, by=g)         # "Reference-plus-by smooths"  (s(x)+fs(x,g) too)

When two smooths share column space over a nested feature set, the fit applies
"automatic hierarchical ownership": the lower-order smooth `s(x)` keeps the
shared realized subspace and the broader factor smooth is **residualized**
against it before fitting (see the `fit-end` warning and
`src/terms/smooth.rs:8475`). That residualization removes columns from the
fitted design, so the saved coefficient vector is shorter than the raw design.

The prediction path (`src/inference/predict/input.rs`, "build prediction
design" → the `beta.len() != design.ncols()` guard at line 277) rebuilds the
design from the term specs **without** replaying that hierarchical
residualization. The reconstructed design therefore has more columns than the
fitted `beta`, and every prediction — even on the model's own training data —
aborts with:

    model/design mismatch: model beta has N coefficients but new-data design
    has M columns   (M > N)

So a model fit with the documented `s(x) + s(g, x, bs=sz)` (or `s(x) +
fs(x, g)`) form is silently un-predictable. The standalone factor smooth
(`y ~ s(g, x, bs=sz)`, no overlapping `s(x)`) predicts fine, which is what
isolates the residualization-replay gap as the cause.

When predict replays the same hierarchical-ownership transform the fit applied
(so the prediction design matches the fitted coefficient count), these fits
predict normally and this test passes without edits.
"""

from __future__ import annotations

import importlib
from typing import Any

pytest: Any = importlib.import_module("pytest")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("gamfit._rust")

import gamfit


def _dataset(seed: int = 0, n: int = 600):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, n)
    lab = np.array(["A", "B", "C"])[rng.integers(0, 3, n)]
    bump = np.where(lab == "A", 0.5 * np.sin(2 * np.pi * x),
                    np.where(lab == "B", -0.5 * np.sin(2 * np.pi * x), 0.0))
    y = 2.0 * x + bump + rng.normal(0.0, 0.2, n)
    return pd.DataFrame({"x": x, "g": lab, "y": y})


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ s(x) + s(g, x, bs=sz)",  # documented "sum-to-zero factor smooths"
        "y ~ s(x) + fs(x, g)",        # global smooth + factor-smooth random effect
    ],
)
def test_overlapping_global_and_factor_smooth_can_be_predicted(formula: str) -> None:
    df = _dataset()

    # The fit succeeds (it residualizes the broader factor smooth against s(x)).
    model = gamfit.fit(df, formula)

    # A fitted model must be predictable on its OWN training data. Today this
    # raises "model/design mismatch: model beta has N coefficients but new-data
    # design has M columns" because the prediction design is rebuilt without the
    # hierarchical-ownership residualization the fit applied.
    grid = pd.DataFrame(
        {"x": np.linspace(0.02, 0.98, 30), "g": (["A", "B", "C"] * 10)}
    )
    preds = model.predict(grid, return_type="dict")
    mean = np.asarray(preds["mean"], dtype=float)

    assert mean.shape == (30,), f"expected 30 predictions, got {mean.shape}"
    assert np.all(np.isfinite(mean)), "predictions must be finite"

    # And the fit must be useful, not a flat constant: in-sample predictions
    # should track the response (guards against a degenerate "fix" that drops
    # the factor smooth entirely).
    in_sample = np.asarray(model.predict(df, return_type="dict")["mean"], dtype=float)
    corr = float(np.corrcoef(in_sample, df["y"].to_numpy())[0, 1])
    assert corr > 0.5, f"in-sample predictions barely track y (corr={corr:.3f})"
