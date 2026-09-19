"""Shared fixtures for the pyGAM-oracle translation suite (runs against the installed gamfit wheel).

Datasets are pyGAM's own CSVs (copied to ./data) loaded via pygam.datasets so the
inputs are byte-identical to what pyGAM's tests use.
"""
import inspect
import os
import warnings

import numpy as np
import pytest

warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))

import pygam.datasets.load_datasets as _L  # noqa: E402

_L.PATH = os.path.join(HERE, "data")
from pygam import datasets as _ds  # noqa: E402

import gamfit  # noqa: E402


def pdep(model, term, data, grid=None, n_points=100):
    """partial_dependence across wheel (term, data, grid) and HEAD (term, grid) signatures."""
    params = inspect.signature(model.partial_dependence).parameters
    if "data" in params:
        return model.partial_dependence(term, data, grid=grid, n_points=n_points)
    return model.partial_dependence(term, grid=grid, n_points=n_points)


def eta_of(model, data):
    return np.asarray(model.predict(data, interval=0.95)["linear_predictor_plugin"], float)


def intercept(model, data):
    return float(model.design_matrix(data).coefficients[0])


@pytest.fixture(scope="session")
def mcycle():
    X, y = _ds.mcycle(return_X_y=True)
    return {"x": X[:, 0].astype(float), "y": np.asarray(y, float)}


@pytest.fixture(scope="session")
def hepatitis():
    X, y = _ds.hepatitis(return_X_y=True)
    return {"x": X[:, 0].astype(float), "y": np.asarray(y, float)}


@pytest.fixture(scope="session")
def coal():
    X, y = _ds.coal(return_X_y=True)
    return {"x": X[:, 0].astype(float), "y": np.asarray(y, float)}


@pytest.fixture(scope="session")
def trees():
    X, y = _ds.trees(return_X_y=True)
    return {"girth": X[:, 0].astype(float), "height": X[:, 1].astype(float), "y": np.asarray(y, float)}


@pytest.fixture(scope="session")
def wage():
    X, y = _ds.wage(return_X_y=True)
    return {
        "year": X[:, 0].astype(float),
        "age": X[:, 1].astype(float),
        "edu": np.asarray([f"e{int(v)}" for v in X[:, 2]]),
        "y": np.asarray(y, float),
    }


@pytest.fixture(scope="session")
def default():
    X, y = _ds.default(return_X_y=True)
    return {
        "student": np.asarray([f"s{int(v)}" for v in X[:, 0]]),
        "balance": X[:, 1].astype(float),
        "income": X[:, 2].astype(float),
        "y": np.asarray(y, float),
    }


@pytest.fixture(scope="session")
def chicago():
    X, y = _ds.chicago(return_X_y=True)
    return {
        "time": X[:, 0].astype(float),
        "tmpd": X[:, 1].astype(float),
        "pm10": X[:, 2].astype(float),
        "o3": X[:, 3].astype(float),
        "y": np.asarray(y, float),
    }


@pytest.fixture(scope="session")
def toy_interaction():
    # pyGAM's toy_interaction: y = x0 * sin(x1) + noise, n=50000; we use n=5000 for CPU budget.
    rng = np.random.default_rng(0)
    n = 5000
    x0 = rng.uniform(-1, 1, n) * 5
    x1 = rng.uniform(-1, 1, n) * 5
    y = x0 * np.sin(x1) + rng.normal(0, 0.1, n)
    return {"x0": x0, "x1": x1, "y": y}


@pytest.fixture(scope="session")
def mcycle_model(mcycle):
    return gamfit.fit(mcycle, "y ~ s(x)")
