"""Static/pure-numpy contract gates for the #2234 real-model steering harness."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments" / "steering_e1" / "run_e1.py"
SPEC = importlib.util.spec_from_file_location("steering_e1_run", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
E1 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = E1
SPEC.loader.exec_module(E1)


def test_next_day_target_accounts_for_prompt_continuation() -> None:
    # Tuesday source shifted one day becomes Wednesday; these prompts ask for the
    # following day, so the correct next-token target is Thursday.
    assert E1.WEEKDAYS[E1.continuation_target_index(1, 1)] == "Thursday"
    assert E1.WEEKDAYS[E1.continuation_target_index(6, 1)] == "Tuesday"


def test_target_probability_is_full_softmax_not_weekday_renormalized() -> None:
    logits = np.asarray([0.0] * 7 + [10.0], dtype=np.float64)
    probabilities = E1.weekday_token_probabilities(logits, list(range(7)))
    assert probabilities.sum() < 0.001
    assert E1.token_probability(logits, 0) == pytest.approx(probabilities[0])


def test_target_excluded_collateral_does_not_charge_intended_mass_move() -> None:
    base = np.asarray([0.0, 0.0, 0.0], dtype=np.float64)
    target_only = np.asarray([5.0, 0.0, 0.0], dtype=np.float64)
    assert E1.target_excluded_kl_model_to_base(target_only, base, 0) == pytest.approx(0.0)

    changed_non_target = np.asarray([5.0, 2.0, 0.0], dtype=np.float64)
    q = np.exp(np.asarray([2.0, 0.0]) - np.logaddexp(2.0, 0.0))
    expected_model_to_base = float(np.sum(q * (np.log(q) - np.log(0.5))))
    assert E1.target_excluded_kl_model_to_base(
        changed_non_target, base, 0
    ) == pytest.approx(expected_model_to_base)


def test_dose_contract_includes_zero_fractional_and_integer_endpoints() -> None:
    shifts = E1.parse_target_shifts("1,2,6")
    fractions = E1.parse_dose_fractions("0,0.25,0.5,1")
    doses = {shift * fraction for shift in shifts for fraction in fractions}
    assert 0.0 in doses
    assert 0.25 in doses
    assert 1.0 in doses
    assert 6.0 in doses


def test_obsolete_conditional_and_wrong_direction_metrics_are_absent() -> None:
    source = SCRIPT.read_text()
    for forbidden in (
        "restricted_probs",
        "full_vocab_kl",
        "target_prob_plus",
        "base_target_prob_plus",
        "kl_base_to_patched",
        "cyclic_advance_accuracy",
        "--max-k",
    ):
        assert forbidden not in source
