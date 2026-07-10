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
