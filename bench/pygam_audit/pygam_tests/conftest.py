"""Shared fixtures for the pyGAM-oracle translation suite (runs against the installed gamfit wheel).

Datasets are pyGAM's own CSVs, vendored in ./data from pyGAM v0.12.0 (the version the
bench venv pins; see data/README.md) and loaded via pygam.datasets, so the inputs are
byte-identical to what pyGAM's tests use. The pygam wheel ships the loaders without the
CSVs, which is why they are vendored.

The helpers the tests call live in ``pg_helpers``: under pyproject's
``--import-mode=importlib`` this file is never importable as ``conftest``.
"""
import os
import warnings

import numpy as np
import pytest

warnings.simplefilter("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))

import pygam.datasets.load_datasets as _L  # noqa: E402

from bench.pygam_audit.pygam_tests.pg_helpers import dataset_dir  # noqa: E402

_L.PATH = dataset_dir()
from pygam import datasets as _ds  # noqa: E402

import gamfit  # noqa: E402


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
