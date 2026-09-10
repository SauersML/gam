"""Frontend/native-boundary regressions; these do not substitute for native fits."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def frontend(monkeypatch):
    calls = []

    def fit(*args):
        calls.append(args)
        return SimpleNamespace(subject_ids=lambda: args[5])

    package = ModuleType("event_history_input_contract")
    package.__path__ = []
    binding = ModuleType("event_history_input_contract._binding")
    binding.rust_module = lambda: SimpleNamespace(fit_event_history=fit)
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, binding.__name__, binding)
    spec = importlib.util.spec_from_file_location(
        "event_history_input_contract._event_history",
        Path(__file__).parents[1] / "gamfit" / "_event_history.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, calls


def fit_reference(frontend, profiles, strata=None):
    module, calls = frontend
    ids = list(range(12))
    module.fit_event_history(
        {"id": ids, "entry": [0.0] * 12, "exit": [1.0] * 12},
        {"id": [], "time": [], "mark": []},
        {"id": ids, "start": [0.0] * 12, "x": ids},
        "1", marks={"disease": "once"},
        reference_profiles=profiles, reference_stratum=strata,
    )
    return calls[-1][-2:]


def test_numpy_zero_profile_keeps_reference_normalisation(frontend):
    rows, strata = fit_reference(frontend, np.array([0]))
    assert rows == [0]
    assert strata == [0] * 12


def test_twelve_permuted_numeric_strata_keep_positional_meaning(frontend):
    strata = [11, 2, 10, 1, 7, 3, 0, 9, 8, 6, 4, 5]
    rows, sent = fit_reference(frontend, np.arange(12), np.array(strata))
    assert rows == list(range(12))
    assert sent == strata


def test_unused_reference_profiles_do_not_relabel_subjects(frontend):
    rows, strata = fit_reference(frontend, [1, 0], [1] * 12)
    assert rows == [1, 0]
    assert strata == [1] * 12


@pytest.mark.parametrize("profiles", [[0.9], [True], [np.bool_(False)], ["0"], [-1], [12], []])
def test_invalid_profiles_are_rejected_before_native_fit(frontend, profiles):
    with pytest.raises(ValueError):
        fit_reference(frontend, profiles)
    assert frontend[1] == []


@pytest.mark.parametrize("strata", [[0.0] * 12, [True] * 12, ["0"] * 12, [-1] * 12, [2] * 12, [0]])
def test_invalid_strata_are_rejected_before_native_fit(frontend, strata):
    with pytest.raises(ValueError):
        fit_reference(frontend, [0, 1], strata)
    assert frontend[1] == []


def test_none_explicitly_selects_prior_centring(frontend):
    assert fit_reference(frontend, None) == ([], [])


@pytest.mark.parametrize("stratum", [0.5, True, "0", -1])
def test_forecast_strata_are_not_truncated(frontend, stratum):
    module, _ = frontend
    native_calls = []
    native = SimpleNamespace(
        subject_ids=lambda: [], covariate_names=lambda: ["x"],
        covariate_levels=lambda: [[]],
        population_forecast=lambda *args: native_calls.append(args),
    )
    with pytest.raises(ValueError):
        module.EventHistoryModel(native).population_forecast({"x": 0.0}, 0.0, [1.0], stratum=stratum)
    assert native_calls == []
