"""Shared pytest fixtures for the Python test suite.

The ``synthetic_biobank`` fixture provides a compact biobank-like DataFrame
with PCs, a raw PGS, a binary disease outcome, and left-truncated survival
columns. It is the single source of truth for the three pipeline smoke
tests in ``tests/test_python_api.py`` and for the demo in
``examples/pgs_calibration_pipeline.py`` (which imports it through
``load_biobank_sample``).

The column schema is intentionally the same schema the CLI-based
integration tests at ``tests/integration_pit_pipeline.py`` exercise, so
the Python-binding smoke tests and the CLI contract tests run the same
kind of data through two different entry points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _build_synthetic_biobank(seed: int = 0, n: int = 200) -> pd.DataFrame:
    """Construct the biobank-like frame used by pipeline smoke tests.

    Columns:
        pc1..pc4       ancestry-like PCs (pc1 and pc2 weakly correlated)
        PGS            linear combo of PCs + noise, roughly unit variance
        disease        Bernoulli from Phi(0.5 * PGS + 0.1 * pc1)
        age_entry      uniform on [40, 70)
        age_exit       age_entry + Exp(lam) with lam = exp(-1.2 - 0.3 * PGS)
                       censored at 85 so PGS predicts shorter survival
        event          1 if age_exit < 85 (i.e. uncensored), else 0
    """
    rng = np.random.default_rng(seed)

    pc1 = rng.normal(0.0, 1.0, n)
    # pc2 gets a bit of structure from pc1 so the PC manifold is not
    # trivially isotropic — the Duchon smoother has something to do.
    pc2 = 0.3 * pc1 + np.sqrt(1.0 - 0.3 ** 2) * rng.normal(0.0, 1.0, n)
    pc3 = rng.normal(0.0, 1.0, n)
    pc4 = rng.normal(0.0, 1.0, n)

    pgs = 0.4 * pc1 - 0.2 * pc2 + 0.15 * pc3 + rng.normal(0.0, 0.9, n)

    # Bernoulli via probit so marginal-slope + probit-link has real signal.
    from math import erf, sqrt

    def _phi(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    probs = np.array([_phi(0.5 * z + 0.1 * p) for z, p in zip(pgs, pc1)])
    disease = (rng.uniform(0.0, 1.0, n) < probs).astype(np.float64)

    # Left-truncated survival with PGS-driven hazard.
    age_entry = rng.uniform(40.0, 70.0, n)
    lam = np.exp(-1.2 - 0.3 * pgs)
    time_to_event = rng.exponential(scale=1.0 / np.clip(lam, 1e-6, None), size=n)
    age_exit_raw = age_entry + time_to_event
    censor_age = 85.0
    age_exit = np.minimum(age_exit_raw, censor_age)
    event = (age_exit_raw < censor_age).astype(np.float64)
    # Guarantee age_exit > age_entry even after the censor clip.
    eps = 0.01
    too_short = age_exit <= age_entry + eps
    age_exit = np.where(too_short, age_entry + eps, age_exit)
    event = np.where(too_short, 0.0, event)

    return pd.DataFrame(
        {
            "pc1": pc1,
            "pc2": pc2,
            "pc3": pc3,
            "pc4": pc4,
            "PGS": pgs,
            "disease": disease,
            "age_entry": age_entry,
            "age_exit": age_exit,
            "event": event,
        }
    )


@pytest.fixture
def synthetic_biobank() -> pd.DataFrame:
    """Default fixture: seed=0, n=200."""
    return _build_synthetic_biobank(seed=0, n=200)


@pytest.fixture
def synthetic_biobank_factory():
    """Factory form for tests that want a custom (seed, n)."""
    def _factory(seed: int = 0, n: int = 200) -> pd.DataFrame:
        return _build_synthetic_biobank(seed=seed, n=n)
    return _factory
