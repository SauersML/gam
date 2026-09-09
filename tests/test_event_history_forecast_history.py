"""A forecast can be made for a history that was never in the training
cohort, a history cut at an assessment time forecasts what was known then,
the predictive PIT carries the censored tail, and one formula per mark gives
each mark its own terms. All through the public Python API
(``docs/event-history.md``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import gamfit


def _simulate(n: int, follow_up: float, rates: dict[str, float], seed: int):
    """Constant-hazard competing risks: a once-only ``disease`` and a
    terminal ``death`` with the given rates, a standard-normal score ``g``
    that carries nothing, and administrative censoring at ``follow_up``."""
    rng = np.random.default_rng(seed)
    g = rng.standard_normal(n)
    ids, times, marks, exits = [], [], [], []
    for i in range(n):
        t, at_risk, exit_ = 0.0, {"disease": True}, follow_up
        while True:
            total = rates["death"] + (rates["disease"] if at_risk["disease"] else 0.0)
            t -= np.log(rng.uniform()) / total
            if t >= follow_up:
                break
            if rng.uniform() * total < rates["death"]:
                ids.append(f"s{i}"), times.append(t), marks.append("death")
                exit_ = t
                break
            ids.append(f"s{i}"), times.append(t), marks.append("disease")
            at_risk["disease"] = False
        exits.append(exit_)
    subjects = pd.DataFrame({"id": [f"s{i}" for i in range(n)], "entry": 0.0, "exit": exits})
    events = pd.DataFrame({"id": ids, "time": times, "mark": marks})
    covariates = pd.DataFrame({"id": [f"s{i}" for i in range(n)], "start": 0.0, "g": g})
    return subjects, events, covariates


MARKS = {"disease": "once", "death": "terminal"}


def test_forecast_history_matches_the_training_subject_and_is_prefix_invariant() -> None:
    subjects, events, covariates = _simulate(250, 4.0, {"disease": 0.25, "death": 0.15}, seed=3)
    model = gamfit.fit_event_history(subjects, events, covariates, "g", marks=MARKS)
    # A subject censored alive at the end of follow-up, with a disease event.
    alive = set(subjects.loc[subjects["exit"] == 4.0, "id"])
    diseased = set(events.loc[events["mark"] == "disease", "id"])
    sid = sorted(alive & diseased)[0]
    own = events[events["id"] == sid]
    history = [(float(t), str(m)) for t, m in zip(own["time"], own["mark"])]
    g = float(covariates.loc[covariates["id"] == sid, "g"].iloc[0])
    horizons = [5.0, 6.0]
    training = model.forecast(sid, horizons=horizons)
    standalone = model.forecast_history(0.0, 4.0, history, {"g": g}, horizons)
    np.testing.assert_allclose(standalone["survival"], training["survival"], rtol=1e-10)
    np.testing.assert_allclose(standalone["expected_counts"], training["expected_counts"], rtol=1e-10, atol=1e-14)
    # Once diseased, the disease's first-occurrence probability is zero.
    assert np.all(standalone["expected_counts"][:, 0] == 0.0)
    # Cut before the disease: the forecast is what was known at the cutoff,
    # whatever was appended later.
    disease_time = history[0][0]
    cutoff = 0.5 * disease_time
    early = model.forecast_history(0.0, 4.0, history, {"g": g}, [cutoff + 1.0], cutoff=cutoff)
    without = model.forecast_history(0.0, cutoff, [], {"g": g}, [cutoff + 1.0])
    np.testing.assert_allclose(early["survival"], without["survival"], rtol=1e-12)
    np.testing.assert_allclose(early["expected_counts"], without["expected_counts"], rtol=1e-12)
    assert early["expected_counts"][0, 0] > 0.0, "before its disease the subject is at risk of it"
    # A cutoff past the exit would fabricate follow-up.
    try:
        model.forecast_history(0.0, 4.0, history, {"g": g}, [6.0], cutoff=5.0)
    except Exception as error:  # noqa: BLE001 - the binding's error type is not part of the contract
        assert "after the exit" in str(error)
    else:
        raise AssertionError("a cutoff after the exit must be refused")


def test_pit_carries_the_censored_tail_and_the_distance_is_censor_aware() -> None:
    # Heavy censoring: a death rate of 0.1 over one unit of follow-up leaves
    # about ninety percent of the subjects censored. The event PITs alone are
    # then uniform on [0, 1 − e^{−0.1}], and comparing them to the uniform
    # law on [0, 1] would flag a correctly specified model.
    subjects, events, covariates = _simulate(400, 1.0, {"disease": 0.0, "death": 0.1}, seed=8)
    model = gamfit.fit_event_history(subjects, events, covariates, "1", marks=MARKS)
    pits = model.pit("s0")
    assert pits["pit"].shape == pits["observed"].shape == pits["time"].shape
    assert len(pits["marks"]) == len(pits["pit"])
    if subjects.loc[0, "exit"] == 1.0:
        assert not pits["observed"][-1] and pits["marks"][-1] == []
    else:
        assert pits["observed"][-1] and pits["marks"][-1] == ["death"]
    summary = model.pit_distance()
    assert summary["spells"] == 400
    assert 0 < summary["events"] < 100
    assert summary["distance"] < 0.12, summary
    # The event-only Kolmogorov–Smirnov distance sits at its censoring floor.
    event_pits = np.concatenate(
        [model.pit(f"s{i}")["pit"][model.pit(f"s{i}")["observed"]] for i in range(400)]
    )
    n = len(event_pits)
    grid = np.sort(event_pits)
    event_only = max(np.max(np.arange(1, n + 1) / n - grid), np.max(grid - np.arange(n) / n))
    assert event_only > 0.8, event_only


def test_one_formula_per_mark_gives_each_mark_its_own_terms() -> None:
    subjects, events, covariates = _simulate(200, 3.0, {"disease": 0.3, "death": 0.1}, seed=5)
    model = gamfit.fit_event_history(subjects, events, covariates, ["g", "1"], marks=MARKS)
    assert len(model.coefficients("disease")) == 2
    assert len(model.coefficients("death")) == 1
    try:
        gamfit.fit_event_history(subjects, events, covariates, ["g"] * 3, marks=MARKS)
    except Exception as error:  # noqa: BLE001
        assert "one per mark" in str(error)
    else:
        raise AssertionError("three formulas for two marks must be refused")
