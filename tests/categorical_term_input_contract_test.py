"""Input contracts for categorical columns (pyGAM audit F2, F3).

Every categorical spelling (a bare ``+ g``, ``factor(g)``, ``C(g)``,
``group(g)``, ``re(g)``) builds one coefficient per level under a ridge whose
strength REML estimates, so the model can recover the null of no level effect.
Two input contracts keep categorical columns out of the wrong terms:

* a categorical column in a term that treats its inputs as numeric axes
  (``s()``, ``linear()``, ``te()``, ...) is refused with a pointer to
  ``factor()``/``group()`` instead of fitting the level codes as positions on
  a line (F2);
* the categorical wrappers take no options, so a typo such as
  ``factor(g, foo=1)`` is an error instead of being silently ignored (F3).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

import gamfit

_LEVELS = np.array(["alpha", "beta", "gamma", "delta"])
_GROUP_MEANS = np.array([1.0, 2.5, -0.7, 0.3])


def _gaussian_frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gi = rng.integers(0, _LEVELS.size, 400)
    y = _GROUP_MEANS[gi] + rng.normal(0.0, 1.0, gi.size)
    return pd.DataFrame({"g": _LEVELS[gi], "x": np.linspace(0.0, 1.0, gi.size), "y": y})


def _g_block(model: Any) -> Any:
    blocks = [block for block in model.term_blocks if block.name == "g"]
    assert len(blocks) == 1, f"expected exactly one `g` block, got {model.term_blocks}"
    return blocks[0]


@pytest.mark.parametrize("formula", ["y ~ g", "y ~ factor(g)", "y ~ C(g)", "y ~ group(g)", "y ~ re(g)"])
def test_every_categorical_spelling_is_a_reml_penalized_level_block(formula: str) -> None:
    model = gamfit.fit(_gaussian_frame(seed=1), formula)
    block = _g_block(model)
    assert block.end - block.start == _LEVELS.size, (
        f"{formula}: the level block keeps one column per level, got {block}"
    )
    assert len(model.smoothing_parameters()) == 1, (
        f"{formula}: the level block carries one REML-estimated penalty, got "
        f"{model.smoothing_parameters()}"
    )


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ s(g)",
        "y ~ linear(g)",
        "y ~ te(x, g)",
        'y ~ s(g, bs="cc")',
        "y ~ thinplate(x, g)",
        "y ~ matern(g)",
    ],
)
def test_categorical_column_in_a_numeric_axis_term_is_refused(formula: str) -> None:
    data = _gaussian_frame(seed=2)
    for column in (data["g"], data["g"].astype("category")):
        frame = data.assign(g=column)
        with pytest.raises(gamfit.errors.FormulaError) as excinfo:
            gamfit.fit(frame, formula)
        message = str(excinfo.value)
        assert "'g' is categorical" in message, message
        assert "factor(g)" in message and "group(g)" in message, message


@pytest.mark.parametrize(
    "formula",
    [
        "y ~ factor(g, foo=1)",
        "y ~ factor(g, double_penalty=false)",
        "y ~ group(g, bogus=3)",
        "y ~ re(g, k=4)",
    ],
)
def test_categorical_wrappers_reject_unknown_options(formula: str) -> None:
    with pytest.raises(gamfit.errors.FormulaError, match="does not accept option"):
        gamfit.fit(_gaussian_frame(seed=3), formula)
