"""Each formula, family and link behavior has one spelling.

Any other spelling is refused with an error that names the accepted one, so a
user who types it learns the supported form rather than getting a silent alias.
Every case pairs a refused spelling with its canonical spelling and checks that
the canonical one fits.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

import gamfit


def _frame() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(25)
    n = 240
    x = rng.uniform(0.0, 1.0, n)
    z = rng.uniform(0.0, 1.0, n)
    g = np.array(["a", "b", "c", "d"])[rng.integers(0, 4, n)]
    mu = np.exp(0.4 + 0.6 * np.sin(2.0 * np.pi * x) + 0.3 * z)
    return {
        "x": x,
        "z": z,
        "g": g,
        "y": np.sin(2.0 * np.pi * x) + z + 0.2 * rng.standard_normal(n),
        "count": rng.negative_binomial(4.0, 4.0 / (4.0 + mu)).astype(float),
        "wald": rng.wald(mu, 1.0 / 0.3),
    }


FORMULA_CASES = [
    ("y ~ s(x, z, bs='tp')", "unknown smooth type `tp`; use `tps`", "y ~ s(x, z, bs='tps')"),
    ("y ~ s(x, bs='cs')", "unknown smooth type `cs`; use `cr`", "y ~ s(x, bs='cr')"),
    ("y ~ s(x, type=ps)", "unknown option `type` in s(); use `bs`", "y ~ s(x, bs=ps)"),
    ("y ~ s(x, basis_dim=8)", "unknown option `basis_dim` in s(); use `k`", "y ~ s(x, k=8)"),
    ("y ~ s(x, m=2)", "unknown option `m` in s(); use `penalty_order`", "y ~ s(x, penalty_order=2)"),
    ("y ~ x + re(g)", "unknown term function `re`; use `group()`", "y ~ x + group(g)"),
    ("y ~ tensor(x, z)", "unknown term function `tensor`; use `te()`", "y ~ te(x, z)"),
    ("y ~ cc(x)", "unknown term function `cc`; use `cyclic()`", "y ~ cyclic(x)"),
    ("y ~ constrain(x, min=0)", "unknown term function `constrain`; use `linear()`", "y ~ linear(x, min=0)"),
    ("y ~ linear(x, lower=0)", "unknown option `lower` in linear(); use `min`", "y ~ linear(x, min=0)"),
]


@pytest.mark.parametrize(("removed", "message", "canonical"), FORMULA_CASES)
def test_removed_formula_spelling_names_the_canonical_one(removed: str, message: str, canonical: str) -> None:
    data = _frame()
    with pytest.raises(Exception, match=re.escape(message)):
        gamfit.fit(data, removed)
    model = gamfit.fit(data, canonical)
    fitted = np.asarray(model.predict(data), dtype=float).reshape(-1)
    assert np.all(np.isfinite(fitted)), canonical


@pytest.mark.parametrize(("removed", "canonical"), [("nb", "negative-binomial"), ("negbin", "negative-binomial")])
def test_removed_family_spelling_names_the_canonical_one(removed: str, canonical: str) -> None:
    data = _frame()
    with pytest.raises(Exception, match=re.escape(f"unknown family `{removed}`; use `{canonical}`")):
        gamfit.fit(data, "count ~ s(x)", family=removed)
    model = gamfit.fit(data, "count ~ s(x)", family=canonical)
    fitted = np.asarray(model.predict(data), dtype=float).reshape(-1)
    assert np.all(fitted > 0.0)


@pytest.mark.parametrize("removed", ["inv_squared", "inv-squared", "1/mu^2"])
def test_removed_link_spelling_names_the_canonical_one(removed: str) -> None:
    data = _frame()
    with pytest.raises(Exception, match=re.escape(f"unknown link `{removed}`; use `inverse-squared`")):
        gamfit.fit(data, "wald ~ s(x)", family="inverse-gaussian", link=removed)
    model = gamfit.fit(data, "wald ~ s(x)", family="inverse-gaussian", link="inverse-squared")
    fitted = np.asarray(model.predict(data), dtype=float).reshape(-1)
    assert np.all(fitted > 0.0)
