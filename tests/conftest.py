"""Shared pytest fixtures for the Python test suite."""

from __future__ import annotations

from collections.abc import Callable
from importlib import import_module
from math import erf, sqrt
from typing import Protocol, TypeVar, cast

import numpy as np
import pandas as pd

_Fixture = TypeVar("_Fixture", bound=Callable[..., object])


class _Pytest(Protocol):
    def fixture(self, fixture_function: _Fixture) -> _Fixture: ...


pytest = cast(_Pytest, import_module("pytest"))

SyntheticBiobankFactory = Callable[[int, int], pd.DataFrame]


def _build_synthetic_biobank(seed: int = 0, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    pc1 = rng.normal(0.0, 1.0, n)
    pc2 = 0.3 * pc1 + np.sqrt(1.0 - 0.3 ** 2) * rng.normal(0.0, 1.0, n)
    pc3 = rng.normal(0.0, 1.0, n)
    pc4 = rng.normal(0.0, 1.0, n)

    pgs = 0.4 * pc1 - 0.2 * pc2 + 0.15 * pc3 + rng.normal(0.0, 0.9, n)
    probs = np.array(
        [0.5 * (1.0 + erf((0.5 * z + 0.1 * p) / sqrt(2.0))) for z, p in zip(pgs, pc1)]
    )
    disease = (rng.uniform(0.0, 1.0, n) < probs).astype(np.float64)

    age_entry = rng.uniform(40.0, 70.0, n)
    lam = np.exp(-1.2 - 0.3 * pgs)
    time_to_event = rng.exponential(scale=1.0 / np.clip(lam, 1e-6, None), size=n)
    age_exit_raw = age_entry + time_to_event
    censor_age = 85.0
    age_exit = np.minimum(age_exit_raw, censor_age)
    event = (age_exit_raw < censor_age).astype(np.float64)
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
def synthetic_biobank_factory() -> SyntheticBiobankFactory:
    def _factory(seed: int = 0, n: int = 200) -> pd.DataFrame:
        return _build_synthetic_biobank(seed=seed, n=n)

    return _factory
