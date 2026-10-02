"""Topology selection is one Rust owner; Python only marshals (#2899 P10)."""
from __future__ import annotations

from pathlib import Path

import pytest

import gamfit._select_topology as st


def test_python_selector_runs_no_candidate_loop_or_score_policy() -> None:
    source = Path(st.__file__).read_text(encoding="utf-8")
    for deleted_helper in (
        "def _score_for_kind(",
        "def _scale_score(",
        "def _score_disagreement_warnings(",
        "def _candidate_failure(",
        "def _fitted_candidate_outcome(",
        "def _formula_for_candidate(",
        "def _select_candidate_lifecycle(",
    ):
        assert deleted_helper not in source
    assert "from ._api import fit" not in source
    assert ".select_topology_table(" in source


def test_bic_is_not_a_topology_score_kind() -> None:
    """SPEC R11/R25: topology candidates are ranked by REML/LAML evidence only; the
    Rust selector refuses any other kind before it fits anything."""
    assert st._SCORE_KINDS == ("reml", "laml", "tk")
    with pytest.raises(ValueError, match="score must be one of: 'reml', 'laml', 'tk'"):
        st.select_topology({"x": [0.0, 1.0], "y": [0.0, 1.0]}, "y", score="bic")
    assert '"deviance"' not in Path(st.__file__).read_text(encoding="utf-8")


def test_per_effective_dim_is_not_a_topology_score_scale() -> None:
    """#4556: dividing each candidate's evidence by its own effective dimension lets a
    shared constant reverse the race, so the Rust selector refuses the scale."""
    with pytest.raises(ValueError, match="score_scale must be one of"):
        st.select_topology(
            {"x": [0.0, 1.0], "y": [0.0, 1.0]}, "y", score_scale="per_effective_dim"
        )
