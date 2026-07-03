"""Regression for #2095: ``difference_smooth`` must not crash on binary models.

``Model.difference_smooth`` is a design-matrix contrast on the link scale; the
response value is never used by the contrast. Before the fix the FFI filled the
response column of its template rows with the mid-range value
``0.5 * (0 + 1) = 0.5``, which the binary schema validator rejected with
``column 'y' is binary in schema but row 1 has value 0.5; expected 0 or 1`` for
EVERY Bernoulli / binomial-logit model. The fix fills binary columns with a
schema-valid ``0``.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

pytest: Any = importlib.import_module("pytest")

pytest.importorskip("gamfit._rust")

import gamfit


def _binary_frame(n: int = 80) -> dict[str, list[Any]]:
    rng = np.random.default_rng(2095)
    x = np.linspace(0.0, 1.0, n)
    g = np.where(np.arange(n) % 2 == 0, "A", "B")
    # Group-specific smooth effect on the logit scale, then sample Bernoulli.
    eta = np.sin(2 * np.pi * x) + (g == "B") * (0.5 + 1.5 * x)
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.random(n) < p).astype(float)
    return {"y": list(y), "x": list(x), "g": list(g)}


def test_difference_smooth_on_binary_response_model() -> None:
    frame = _binary_frame()
    model = gamfit.fit(frame, "y ~ g + s(x, by=g)", family="bernoulli")

    # Predict must work on the fitted binary model.
    preds = model.predict(frame)
    assert preds is not None

    # The contrast is on the link scale and never consumes the response, so it
    # must succeed for a binary model rather than raising the 0.5 schema error.
    out = model.difference_smooth(
        view="x", group="g", pairs=[("A", "B")], n=10
    )
    assert len(out) == 10
    assert np.all(np.isfinite(out["diff"]))
    assert np.all(out["se"] >= 0)
    assert np.all(out["upper"] >= out["lower"])
