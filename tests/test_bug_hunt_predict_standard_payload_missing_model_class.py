"""Bug hunt: ``Model.predict`` raises ``KeyError: 'model_class'`` on every
standard (non-survival) model.

Commit ``38ee68993`` ("Remove compatibility shims and unify GPU policy")
deleted the ``fallback_model_class`` / ``fallback_family`` shim from
``shape_predict_response`` (``gamfit/_predict_shape.py``). Before that commit
the dispatcher read the discriminators *defensively*::

    model_class = str(parsed.get("model_class") or fallback_model_class)
    family = str(parsed.get("family") or parsed.get("family_kind") or fallback_family)

After it the access is unconditional::

    model_class = str(parsed["model_class"])   # _predict_shape.py:119
    family = str(parsed["family"])             # _predict_shape.py:120

But the Rust FFI never started emitting those keys for the standard path.
``predict_dataset_impl`` serializes ``PredictionPayload`` (``crates/gam-pyffi/
src/lib.rs:519-522``), whose *only* serialized field is ``columns``::

    #[derive(Serialize)]
    struct PredictionPayload {
        columns: BTreeMap<String, Vec<f64>>,
    }

So the JSON returned by ``predict_table`` for a Gaussian / binomial / Poisson /
... model is ``{"columns": {...}}`` — no ``model_class``, no ``family``, no
``class`` discriminator (only the *survival* payload,
``SurvivalPredictionPayload``, carries ``model_class``). The unconditional
``parsed["model_class"]`` therefore raises ``KeyError: 'model_class'`` for every
non-survival model.

This is a live regression on ``main``: ``38ee68993`` is an ancestor of the
``0.1.180`` release commit, so the shipped wheel cannot ``predict`` at all from
the Python wrapper. The Rust CLI (``gam predict``) is unaffected — it never goes
through ``shape_predict_response`` — which is why the engine itself is fine and
only the Python binding boundary is broken.

The fix is to make the standard ``PredictionPayload`` carry ``model_class`` and
``family`` (the survival payload already does), or to restore the defensive
``.get(...)`` access. Either way these tests, which only ask that a plain
``predict`` round-trips, must start passing without edits.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def _mean(out: object) -> np.ndarray:
    """Response-scale point predictions from any ``Model.predict`` return shape."""
    if isinstance(out, np.ndarray):
        return np.asarray(out, dtype=float).ravel()
    return np.asarray(out["mean"], dtype=float)


def test_gaussian_smooth_predict_round_trips() -> None:
    """A plain Gaussian ``s(x)`` fit must be able to predict on new rows.

    Currently raises ``KeyError: 'model_class'`` inside
    ``shape_predict_response`` because the standard ``PredictionPayload`` omits
    the discriminator the post-shim dispatcher now reads unconditionally.
    """
    rng = np.random.RandomState(0)
    n = 300
    x = np.linspace(0.0, 10.0, n)
    y = np.sin(x) + 0.2 * rng.randn(n)
    train = pd.DataFrame({"x": x, "y": y})

    model = gamfit.fit(train, "y ~ s(x)")

    grid = pd.DataFrame({"x": np.linspace(0.5, 9.5, 11)})
    out = model.predict(grid)  # must not raise KeyError: 'model_class'

    mean = _mean(out)
    assert mean.shape[0] == grid.shape[0]
    assert np.all(np.isfinite(mean))


def test_gaussian_smooth_predict_with_interval_round_trips() -> None:
    """The interval (table) path must also resolve a model_class/family.

    The same ``parsed["model_class"]`` access is on the interval branch, so an
    interval predict is broken for the identical reason.
    """
    rng = np.random.RandomState(1)
    n = 300
    x = np.linspace(0.0, 10.0, n)
    y = np.sin(x) + 0.2 * rng.randn(n)
    train = pd.DataFrame({"x": x, "y": y})

    model = gamfit.fit(train, "y ~ s(x)")

    grid = pd.DataFrame({"x": np.linspace(0.5, 9.5, 11)})
    out = model.predict(grid, interval=0.95)  # must not raise KeyError

    mean = _mean(out)
    assert mean.shape[0] == grid.shape[0]
    assert np.all(np.isfinite(mean))
    # An interval predict returns a table carrying the uncertainty band.
    lower = np.asarray(out["mean_lower"], dtype=float)
    upper = np.asarray(out["mean_upper"], dtype=float)
    assert np.all(lower <= mean + 1e-9)
    assert np.all(mean <= upper + 1e-9)


def test_binomial_logit_predict_round_trips() -> None:
    """A non-identity-link model (binomial logit) is broken identically.

    This guards that the missing discriminator is not specific to the Gaussian
    posterior-mean==plug-in path: it is the standard payload schema for *every*
    non-survival family.
    """
    rng = np.random.RandomState(2)
    n = 400
    x = rng.uniform(-3.0, 3.0, n)
    eta = 0.8 * x
    p = 1.0 / (1.0 + np.exp(-eta))
    y = (rng.uniform(size=n) < p).astype(float)
    train = pd.DataFrame({"x": x, "y": y})

    model = gamfit.fit(train, "y ~ s(x)", family="binomial")

    grid = pd.DataFrame({"x": np.linspace(-2.5, 2.5, 9)})
    out = model.predict(grid)  # must not raise KeyError: 'model_class'

    mean = _mean(out)
    assert mean.shape[0] == grid.shape[0]
    assert np.all(np.isfinite(mean))
    # Response-scale binomial predictions are probabilities.
    assert np.all(mean >= -1e-9) and np.all(mean <= 1.0 + 1e-9)
