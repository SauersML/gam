"""Fitted models pickle, copy and cross process boundaries (pyGAM audit api F1 / PKG-02).

``pickle.dumps(model)`` raised ``TypeError: cannot pickle '_FittedModel'``: the
``Model`` shell holds a compiled Rust prediction handle with no pickle support.
That broke ``copy.deepcopy``, ``joblib.dump``, ``joblib.Parallel`` and every
sklearn workflow that ships a fitted estimator to a worker process.

Pickling now goes through the one saved-model byte archive (``Model.dumps``),
rebuilt by ``gamfit.loads``. These tests pin that a round trip is exact: the
archive is byte-identical and every public accessor returns bit-identical
values.
"""

from __future__ import annotations

import copy
import dataclasses
import inspect
import math
import pickle
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

import gamfit
from gamfit._model import Model, MultinomialModel
from gamfit.sklearn import GAMClassifier, GAMRegressor

_N = 300


def _frame(seed: int) -> Any:
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, 1.0, _N)
    z = rng.uniform(0.0, 1.0, _N)
    return pd.DataFrame({"x": x, "z": z, "eta": np.sin(2.0 * np.pi * x) + 0.5 * z})


def _gaussian() -> tuple[Any, Any]:
    df = _frame(1)
    rng = np.random.default_rng(10)
    df["y"] = df["eta"] + 0.3 * rng.standard_normal(_N)
    return gamfit.fit(df, "y ~ s(x) + z", family="gaussian"), df


def _binomial() -> tuple[Any, Any]:
    df = _frame(2)
    rng = np.random.default_rng(20)
    df["y"] = (rng.random(_N) < 1.0 / (1.0 + np.exp(-df["eta"]))).astype(float)
    return gamfit.fit(df, "y ~ s(x) + z", family="binomial"), df


def _gaussian_location_scale() -> tuple[Any, Any]:
    df = _frame(3)
    rng = np.random.default_rng(30)
    df["y"] = df["eta"] + np.exp(-1.5 + df["x"]) * rng.standard_normal(_N)
    model = gamfit.fit(df, "y ~ s(x) + z", family="gaussian", noise_formula="s(x)")
    return model, df


_FITS = {
    "gaussian": _gaussian,
    "binomial": _binomial,
    "gaussian_location_scale": _gaussian_location_scale,
}


def _assert_identical(a: Any, b: Any, where: str) -> None:
    """Recursive bit-for-bit equality over the value shapes accessors return."""
    assert type(a) is type(b), f"{where}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, pd.DataFrame):
        pd.testing.assert_frame_equal(a, b, check_exact=True, obj=where)
    elif isinstance(a, pd.Series):
        pd.testing.assert_series_equal(a, b, check_exact=True, obj=where)
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape, where
        if a.dtype.kind == "f":
            assert np.array_equal(a.view(np.uint64 if a.itemsize == 8 else np.uint32),
                                  b.view(np.uint64 if b.itemsize == 8 else np.uint32)), where
        else:
            assert np.array_equal(a, b), where
    elif isinstance(a, float):
        assert a == b or (math.isnan(a) and math.isnan(b)), f"{where}: {a!r} vs {b!r}"
    elif isinstance(a, dict):
        assert list(a) == list(b), where
        for key in a:
            _assert_identical(a[key], b[key], f"{where}[{key!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), where
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_identical(x, y, f"{where}[{i}]")
    elif dataclasses.is_dataclass(a):
        for field in dataclasses.fields(a):
            _assert_identical(
                getattr(a, field.name), getattr(b, field.name), f"{where}.{field.name}"
            )
    elif hasattr(type(a), "__slots__"):
        for slot in type(a).__slots__:
            _assert_identical(getattr(a, slot), getattr(b, slot), f"{where}.{slot}")
    else:
        assert a == b, f"{where}: {a!r} vs {b!r}"


def _call(fn: Any) -> tuple[str, Any]:
    try:
        return "ok", fn()
    except Exception as exc:  # an accessor that refuses must refuse identically
        return "raised", (type(exc), str(exc))


def _public_properties(cls: type) -> list[str]:
    return sorted(
        name
        for name, member in inspect.getmembers(cls)
        if isinstance(member, property) and not name.startswith("_")
    )


# Accessors every family here supports; the others may refuse a model class,
# and then the restored model must refuse it identically.
_MUST_SUCCEED = {"summary", "smoothing_parameters", "predict", "predict(interval)", "sample"}


def _assert_model_round_trip(original: Any, restored: Any, data: Any) -> None:
    assert type(restored) is type(original)
    assert restored is not original
    assert restored.dumps() == original.dumps()
    for name in _public_properties(type(original)):
        _assert_identical(
            _call(lambda: getattr(restored, name)),
            _call(lambda: getattr(original, name)),
            name,
        )
    calls = {
        "summary": lambda m: m.summary(),
        "str(summary)": lambda m: str(m.summary()),
        "smoothing_parameters": lambda m: m.smoothing_parameters(),
        "predict": lambda m: m.predict(data),
        "predict(interval)": lambda m: m.predict(data, interval=0.95),
        "predict(observation_interval)": lambda m: m.predict(
            data, interval=0.9, observation_interval=True
        ),
        "design_matrix": lambda m: m.design_matrix(data),
        "check": lambda m: m.check(data),
        "sample": lambda m: m.sample(data.head(20), samples=16, seed=7),
    }
    for label, fn in calls.items():
        got = _call(lambda: fn(restored))
        expected = _call(lambda: fn(original))
        if label in _MUST_SUCCEED:
            assert expected[0] == "ok", f"{label}: {expected[1]}"
        _assert_identical(got, expected, label)


@pytest.mark.parametrize("family", sorted(_FITS))
def test_pickle_round_trip_is_bit_identical(family: str) -> None:
    model, df = _FITS[family]()
    assert type(model) is Model
    for protocol in range(pickle.DEFAULT_PROTOCOL, pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(model, protocol=protocol))
        _assert_model_round_trip(model, restored, df)


@pytest.mark.parametrize("copier", [copy.copy, copy.deepcopy], ids=["copy", "deepcopy"])
def test_copy_and_deepcopy_are_bit_identical(copier: Any) -> None:
    model, df = _gaussian_location_scale()
    _assert_model_round_trip(model, copier(model), df)


def test_multinomial_model_pickles_and_deepcopies() -> None:
    rng = np.random.default_rng(4)
    x = rng.uniform(0.0, 1.0, _N)
    eta = np.column_stack([np.zeros(_N), 1.5 * x, -1.0 + 2.0 * x])
    probs = np.exp(eta - eta.max(axis=1, keepdims=True))
    probs /= probs.sum(axis=1, keepdims=True)
    labels = np.array(["a", "b", "c"])
    df = pd.DataFrame({"x": x, "y": [labels[rng.choice(3, p=p)] for p in probs]})
    model = gamfit.fit(df, "y ~ s(x)", family="multinomial")
    assert type(model) is MultinomialModel
    for restored in (pickle.loads(pickle.dumps(model)), copy.deepcopy(model)):
        assert type(restored) is MultinomialModel
        assert restored.dumps() == model.dumps()
        for name in _public_properties(MultinomialModel):
            _assert_identical(getattr(restored, name), getattr(model, name), name)
        _assert_identical(restored.predict(df), model.predict(df), "predict")
        _assert_identical(
            restored.predict(df, interval="confidence"),
            model.predict(df, interval="confidence"),
            "predict(interval)",
        )


def test_joblib_parallel_predict_over_pickled_model() -> None:
    model, df = _binomial()
    chunks = [df.iloc[idx] for idx in np.array_split(np.arange(len(df)), 4)]
    expected = [model.predict(chunk, interval=0.95) for chunk in chunks]
    # loky pickles the model into each worker process, which predicts there.
    got = joblib.Parallel(n_jobs=2)(
        joblib.delayed(Model.predict)(model, chunk, interval=0.95) for chunk in chunks
    )
    for i, (g, want) in enumerate(zip(got, expected)):
        _assert_identical(g, want, f"parallel predict chunk {i}")


def test_sklearn_estimators_pickle_and_clone() -> None:
    df = _frame(5)
    rng = np.random.default_rng(50)
    X = df[["x", "z"]].to_numpy()
    y_reg = df["eta"].to_numpy() + 0.3 * rng.standard_normal(_N)
    y_clf = (rng.random(_N) < 1.0 / (1.0 + np.exp(-df["eta"].to_numpy()))).astype(int)
    fitted = [
        (GAMRegressor(formula="s(x0) + x1").fit(X, y_reg), "predict"),
        (GAMClassifier(formula="s(x0) + x1").fit(X, y_clf), "predict_proba"),
    ]
    for estimator, method in fitted:
        expected = getattr(estimator, method)(X)
        for restored in (pickle.loads(pickle.dumps(estimator)), copy.deepcopy(estimator)):
            assert type(restored) is type(estimator)
            assert restored.model_.dumps() == estimator.model_.dumps()
            assert restored.get_params() == estimator.get_params()
            _assert_identical(getattr(restored, method)(X), expected, method)
        assert clone(estimator).get_params() == estimator.get_params()
