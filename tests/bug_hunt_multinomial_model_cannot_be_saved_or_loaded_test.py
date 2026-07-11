"""Regression: a fitted multinomial-logit GAM must survive the public
save/load (and dumps/loads) round-trip.

Issue #2078: `gamfit.fit(..., family="multinomial")` returns a
`MultinomialModel` that fits and predicts in memory, but the public
persistence API could not round-trip it:

  * `MultinomialModel` defined no `save`/`dumps`, so `gamfit.save(m, path)`
    raised `TypeError` and `m.dumps()` raised `AttributeError`.
  * `gamfit.loads` only ever rebuilt a plain `Model`; the multinomial payload
    has a different on-disk schema, so no branch could reconstruct a
    `MultinomialModel`.

This test fits a 3-class multinomial GAM, persists it through the public API,
reloads it, and asserts the reload is a `MultinomialModel` whose classes and
softmax predictions match the original bit-for-bit (to atol 1e-12).
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import gamfit
from gamfit._model import MultinomialModel


def _make_frame(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, size=n)
    eta = np.column_stack([np.zeros(n), 1.5 * x, -1.0 + 2.0 * x])
    eta -= eta.max(axis=1, keepdims=True)
    probs = np.exp(eta)
    probs /= probs.sum(axis=1, keepdims=True)
    labels = np.array(["a", "b", "c"])
    y = np.array([labels[rng.choice(3, p=p)] for p in probs])
    return pd.DataFrame({"x": x, "y": y})


def test_multinomial_model_save_load_round_trip():
    frame = _make_frame()
    m = gamfit.fit(frame, "y ~ s(x)", family="multinomial")
    assert type(m) is MultinomialModel

    expected = np.asarray(m.predict(frame))

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "multinomial.gam"
        gamfit.save(m, path)          # raised TypeError before the fix
        m2 = gamfit.load(path)

    assert type(m2) is MultinomialModel
    assert m2.classes_ == m.classes_
    assert m2.training_table_kind == "pandas"

    got = np.asarray(m2.predict(frame))
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, atol=1e-12)


def test_multinomial_model_dumps_loads_round_trip():
    frame = _make_frame()
    m = gamfit.fit(frame, "y ~ s(x)", family="multinomial")

    payload = m.dumps()               # raised AttributeError before the fix
    m2 = gamfit.loads(payload)

    assert type(m2) is MultinomialModel
    assert m2.classes_ == m.classes_
    assert m2.training_table_kind == "pandas"
    np.testing.assert_allclose(
        np.asarray(m2.predict(frame)),
        np.asarray(m.predict(frame)),
        atol=1e-12,
    )
