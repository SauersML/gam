"""An event-history formula whose lowering changes what a term means warns at
fit time and keeps the advisory on the model, as :func:`gamfit.fit` does."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gamfit
from gamfit.errors import GamInferenceWarning


def _simulate_cohort(n: int, follow_up: float, seed: int):
    """Single-mark cohort with a constant unit rate and a standard-normal
    score ``g`` that carries nothing."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(n)
    ids, times = [], []
    for i in range(n):
        t = 0.0
        while True:
            t -= np.log(rng.uniform())
            if t >= follow_up:
                break
            ids.append(f"s{i}")
            times.append(t)
    subjects = pd.DataFrame({"id": [f"s{i}" for i in range(n)], "entry": 0.0, "exit": follow_up})
    events = pd.DataFrame({"id": ids, "time": times, "mark": "event"})
    covariates = pd.DataFrame({"id": [f"s{i}" for i in range(n)], "start": 0.0, "g": g})
    return subjects, events, covariates


def test_a_feature_owned_by_a_smooth_and_a_line_warns_and_is_recorded() -> None:
    subjects, events, covariates = _simulate_cohort(240, 4.0, seed=7)
    # `time` in both a smooth and a linear term makes the fit residualize the
    # smooth against the line: the term builder's advisory must reach the user.
    with pytest.warns(GamInferenceWarning, match="appear both"):
        model = gamfit.fit_event_history(subjects, events, covariates, "time + s(time)")
    assert any("appear both" in note and "[time]" in note for note in model.notes)
