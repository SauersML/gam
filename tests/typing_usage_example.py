"""A user's-eye view of the gamfit API, type-checked by ``mypy --strict``.

``tests/test_static_typing.py`` runs mypy over this file: it is never executed,
so every call only has to be a correct use of the public API. Each binding is
annotated with the type a user would expect, so a public signature that
degrades to ``Any``, loses an export, or changes its return type fails here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import gamfit


def fit_save_load_predict(data: dict[str, list[float]], path: Path) -> Any:
    model: gamfit.Model = gamfit.fit(data, "y ~ s(x)")
    summary: gamfit.Summary = model.summary()
    formula: str = summary.formula
    assert formula
    model.save(path)
    loaded = gamfit.load(path)
    assert isinstance(loaded, gamfit.Model)
    reloaded: gamfit.Model = loaded
    return reloaded.predict(data, interval=0.95, return_type="dict")


def conformal_band(model: gamfit.Model, data: dict[str, list[float]]) -> Any:
    return model.predict(data, interval="conformal", conformal_level=0.9)


def round_trip_bytes(model: gamfit.Model) -> gamfit.Model:
    payload: bytes = model.dumps()
    restored = gamfit.loads(payload)
    if not isinstance(restored, gamfit.Model):
        raise TypeError(type(restored).__name__)
    return restored


def multinomial(data: dict[str, list[Any]]) -> NDArray[np.float64]:
    model: gamfit.MultinomialModel = gamfit.fit(data, "y ~ s(x)", family="multinomial")
    probabilities: NDArray[np.float64] = model.predict(data)
    return probabilities


def atom_shape(coords: NDArray[np.float64]) -> dict[Any, Any]:
    return gamfit.adjudicate_atom_shape(coords, folds=5, seed=0)


def compiled_version() -> str:
    return gamfit.__version__
