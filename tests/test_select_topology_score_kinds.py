"""Topology score policy must live behind the typed Rust lifecycle entry."""
from __future__ import annotations

import json
from pathlib import Path

import gamfit._select_topology as st


def test_python_selector_has_no_local_score_or_failure_policy() -> None:
    source = Path(st.__file__).read_text(encoding="utf-8")
    for deleted_helper in (
        "def _score_for_kind(",
        "def _scale_score(",
        "def _score_disagreement_warnings(",
        "def _candidate_failure(",
    ):
        assert deleted_helper not in source
    assert ".select_topology_candidate_lifecycle(" in source
    assert ".rank_topology_candidates(" not in source


def test_lifecycle_marshalling_preserves_distinct_reml_and_tk_kinds(monkeypatch) -> None:
    requests: list[dict[str, object]] = []

    class _Rust:
        def select_topology_candidate_lifecycle(self, request_json: str) -> str:
            request = json.loads(request_json)
            requests.append(request)
            return json.dumps(
                {
                    "ranked": [],
                    "winner_index": None,
                    "failed": [],
                    "warnings": [],
                }
            )

    monkeypatch.setattr(st, "_topology_rust", lambda: _Rust())
    st._select_candidate_lifecycle("reml", "raw", [])
    st._select_candidate_lifecycle("tk", "per_observation", [])

    assert requests[0]["score_kind"] == "reml"
    assert requests[0]["score_scale"] == "raw"
    assert requests[1]["score_kind"] == "tk"
    assert requests[1]["score_scale"] == "per_observation"


def test_bic_is_not_a_topology_score_kind() -> None:
    """SPEC R11/R25: topology candidates are ranked by REML/LAML evidence only.

    The deviance-plus-``log n`` surrogate was deleted with its plumbing, so the
    public entry refuses it before any fit and the marshalled fit metadata no
    longer carries the deviance it needed.
    """
    import pytest

    assert st._SCORE_KINDS == ("reml", "laml", "tk")
    with pytest.raises(ValueError, match="score must be one of: 'reml', 'laml', 'tk'"):
        st.select_topology({"x": [0.0, 1.0], "y": [0.0, 1.0]}, "y", score="bic")
    assert '"deviance"' not in Path(st.__file__).read_text(encoding="utf-8")


def test_rust_lifecycle_refuses_bic_score_kind_and_deviance_metadata() -> None:
    import pytest

    from gamfit._binding import rust_module

    rust = rust_module()
    fitted = {
        "status": "fitted",
        "name": "circle",
        "raw_reml": 1.0,
        "laml": None,
        "null_dim": 0.0,
        "null_space_logdet": None,
        "effective_dim": 2.0,
        "basis_size": 4,
        "n_obs": 20,
    }
    accepted = json.loads(
        rust.select_topology_candidate_lifecycle(
            json.dumps({"score_kind": "reml", "score_scale": "raw", "candidates": [fitted]})
        )
    )
    assert accepted["ranked"][0]["name"] == "circle"
    with pytest.raises(ValueError, match="bic"):
        rust.select_topology_candidate_lifecycle(
            json.dumps({"score_kind": "bic", "score_scale": "raw", "candidates": [fitted]})
        )
    with pytest.raises(ValueError, match="deviance"):
        rust.select_topology_candidate_lifecycle(
            json.dumps(
                {
                    "score_kind": "reml",
                    "score_scale": "raw",
                    "candidates": [dict(fitted, deviance=3.0)],
                }
            )
        )
