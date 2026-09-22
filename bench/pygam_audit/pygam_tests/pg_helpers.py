"""Helpers the pyGAM-oracle translation tests share (#1512).

They lived in ``conftest.py`` and were imported with ``from conftest import ...``, which
works only when the test directory is on ``sys.path``. Under pyproject's
``--import-mode=importlib`` it never is, so the tests import these as
``bench.pygam_audit.pygam_tests.pg_helpers``, the package path the Python Contracts
bench step provides, and the probe scripts beside them import ``pg_helpers`` directly.
"""
import inspect

import numpy as np


def pdep(model, term, data, grid=None, n_points=100):
    """partial_dependence across wheel (term, data, grid) and HEAD (term, grid) signatures,
    as the wheel's dict: HEAD returns a gamfit.PartialEffect, read here by the same keys."""
    params = inspect.signature(model.partial_dependence).parameters
    if "data" in params:
        return model.partial_dependence(term, data, grid=grid, n_points=n_points)
    effect = model.partial_dependence(term, grid=grid, n_points=n_points)
    if isinstance(effect, dict):
        return effect
    return {
        "grid": effect.x if len(effect.axes) == 1 else effect.grid,
        "axes": list(effect.axes),
        "predicted": effect.fit,
        "standard_error": effect.se,
        "covariance_source": effect.covariance_source,
    }


def eta_of(model, data):
    return np.asarray(model.predict(data, interval=0.95)["linear_predictor_plugin"], float)


def intercept(model, data):
    return float(model.design_matrix(data).coefficients[0])
