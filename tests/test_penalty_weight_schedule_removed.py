"""Regression test (#3689): analytic penalties take no ``weight_schedule``.

The schedule was accepted end to end but never advanced: the per-penalty
``apply_schedule`` override macro was empty and nothing called it. A scheduled
penalty therefore ran at ``w_start`` forever, silently replacing the
descriptor's own ``weight``. The strength of an analytic penalty is its
REML-selected log-weight (``weight="auto"``), so the unapplied schedule surface
was removed and passing one must fail loudly instead of being dropped.
"""

from __future__ import annotations

import pytest

import gamfit.penalties as penalties
from gamfit.penalties import ARDPenalty, IsometryPenalty, SparsityPenalty, TopKActivationPenalty


def test_scalar_weight_schedule_is_not_exported() -> None:
    assert not hasattr(penalties, "ScalarWeightSchedule")
    assert "ScalarWeightSchedule" not in penalties.__all__


@pytest.mark.parametrize(
    "build",
    [
        lambda: ARDPenalty(weight_schedule={"w_start": 10.0, "w_end": 1.0}),
        lambda: IsometryPenalty(weight_schedule={"w_start": 10.0, "w_end": 1.0}),
        lambda: SparsityPenalty(weight_schedule={"w_start": 10.0, "w_end": 1.0}),
        lambda: TopKActivationPenalty(2, weight_schedule={"w_start": 10.0, "w_end": 1.0}),
    ],
)
def test_weight_schedule_keyword_is_refused(build) -> None:
    with pytest.raises(TypeError, match="weight_schedule"):
        build()


def test_penalties_have_no_set_weight_schedule() -> None:
    assert not hasattr(ARDPenalty(), "set_weight_schedule")
