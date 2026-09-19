"""Numpy-only contract gates for the #2263 gates 3/4 real-model producer."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "qwen_target_dose_matrix.py"
SPEC = importlib.util.spec_from_file_location("qwen_target_dose_matrix", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
PRODUCER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = PRODUCER
SPEC.loader.exec_module(PRODUCER)


def test_chart_lift_recovers_the_in_chart_part_of_an_ambient_move() -> None:
    rng = np.random.default_rng(2263)
    x = rng.standard_normal((40, 12))
    mean, lift = PRODUCER.chart_projection(x, 5)
    np.testing.assert_allclose(lift @ lift.T, np.eye(5), atol=1e-12)
    move = rng.standard_normal(12)
    in_chart = (move @ lift.T)
    np.testing.assert_allclose(PRODUCER.lift_move(in_chart, lift), (move @ lift.T) @ lift, atol=1e-12)
    np.testing.assert_allclose(
        PRODUCER.to_chart(x[:3] + move, mean, lift) - PRODUCER.to_chart(x[:3], mean, lift),
        np.broadcast_to(in_chart, (3, 5)),
        atol=1e-12,
    )


def test_projected_factors_are_the_chart_pullback_of_the_harvested_operator() -> None:
    rng = np.random.default_rng(7)
    factors = rng.standard_normal((4, 12, 3))
    _, lift = PRODUCER.chart_projection(rng.standard_normal((30, 12)), 5)
    projected = PRODUCER.project_factors(factors, lift)
    assert projected.shape == (4, 5, 3)
    for n in range(4):
        ambient = factors[n] @ factors[n].T
        np.testing.assert_allclose(projected[n] @ projected[n].T, lift @ ambient @ lift.T, atol=1e-10)


def test_realized_shift_accounts_for_the_prompt_asking_for_the_next_label() -> None:
    # January (0) moved by +1 answers March (2); that realizes a +1 shift.
    assert PRODUCER.realized_shift(2, 0, 12) == 1
    # November (10) moved by +3 answers March (2) of the next year.
    assert PRODUCER.realized_shift(2, 10, 12) == 3
    # An unmoved base answers the next label: zero shift.
    assert PRODUCER.realized_shift(4, 3, 12) == 0


def test_the_relative_landing_error_is_symmetric_about_the_target() -> None:
    assert PRODUCER.relative_landing_error(1.25, 1.0) == 0.25
    assert PRODUCER.relative_landing_error(0.75, 1.0) == 0.25
    assert PRODUCER.relative_landing_error(0.5, 0.5) == 0.0


def test_a_landing_is_certified_only_at_the_closer_endpoint_of_its_final_bracket() -> None:
    # Expansion reads 0.3 and 0.6 under the target 1.0 and crosses at 1.4; the
    # resolution then reads 0.9, 1.2, 0.95 and 1.1. The final bracket is the last
    # reading on each side, [0.95, 1.1], and 0.95 is the closer endpoint.
    readings = [0.3, 0.6, 1.4, 0.9, 1.2, 0.95, 1.1]
    landing = PRODUCER.landing_certificate(1.0, readings, 0.95)
    assert landing["final_bracket_nats"] == [0.95, 1.1]
    assert landing["certified"]
    assert landing["landing_error_nats"] <= landing["landing_bound_nats"]
    # The farther endpoint, or an earlier reading outside the final bracket, is
    # not what the solve returns.
    assert not PRODUCER.landing_certificate(1.0, readings, 1.1)["certified"]
    assert not PRODUCER.landing_certificate(1.0, readings, 0.9)["certified"]
    # A tie goes to the upper endpoint, as in the solve.
    tie = [0.5, 1.5]
    assert PRODUCER.landing_certificate(1.0, tie, 1.5)["certified"]
    assert not PRODUCER.landing_certificate(1.0, tie, 0.5)["certified"]
    # With no reading under the target the lower endpoint is the unprobed zero
    # move: [0, 1.5], and 1.5 is closer to 1.0 than 0 is.
    overshoot = PRODUCER.landing_certificate(1.0, [3.0, 1.5], 1.5)
    assert overshoot["final_bracket_nats"] == [0.0, 1.5]
    assert overshoot["certified"]
    # An exact landing must be one of the readings, and its bound is 0.
    exact = PRODUCER.landing_certificate(1.0, [0.5, 1.0], 1.0)
    assert exact["certified"] and exact["landing_bound_nats"] == 0.0
    assert not PRODUCER.landing_certificate(1.0, [0.5, 1.2], 1.0)["certified"]
    # Every reading under the target never brackets it, so nothing is certified.
    assert not PRODUCER.landing_certificate(1.0, [0.2, 0.4], 0.4)["certified"]


def test_nearest_fitted_row_wraps_on_a_circle() -> None:
    coords = np.asarray([0.02, 0.5, 0.97])
    assert PRODUCER.nearest_fitted_row(coords, 0.99, 1.0) == 2
    assert PRODUCER.nearest_fitted_row(coords, 0.01, 1.0) == 0
    assert PRODUCER.nearest_fitted_row(coords, 0.99, float("inf")) == 2
