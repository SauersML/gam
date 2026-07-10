import json

import gamfit._select_topology as st


def test_select_topology_uses_ranked_order_from_comparison_layer(monkeypatch):
    class _Fit:
        def __init__(self, reml, edf):
            self._summary = {"reml_score": reml, "effective_dim": edf, "coefficients": [0.0, 0.0]}

        def summary(self):
            return self._summary

    monkeypatch.setattr(st, "_formula_from_response", lambda data, response: ("y ~ s(x, type=AUTO)", 1, 5))
    monkeypatch.setattr(st, "_normalize_candidates", lambda candidates, feature_dim: [
        st._Candidate("a", object.__new__(st.PeriodicSplineCurve)),
        st._Candidate("b", object.__new__(st.PeriodicSplineCurve)),
    ])
    monkeypatch.setattr(st, "_formula_for_candidate", lambda formula, candidate, *, strict_dimension: formula)
    monkeypatch.setattr(st, "fit", lambda data, formula, **kwargs: _Fit(-1.0 if "a" in formula else -2.0, 1.0))
    monkeypatch.setattr(st, "_extract_reml_score_raw", lambda m: float(m.summary()["reml_score"]))

    class _Rust:
        def select_topology_candidate_lifecycle(self, request_json):
            request = json.loads(request_json)
            assert [row["name"] for row in request["candidates"]] == ["a", "b"]
            return json.dumps(
                {
                    "winner_index": 0,
                    "ranked": [
                        {"name": "b", "score": -2.0, "raw_reml": -2.0, "effective_dim": 1.0, "basis_size": 2, "n_obs": 5},
                        {"name": "a", "score": -1.0, "raw_reml": -1.0, "effective_dim": 1.0, "basis_size": 2, "n_obs": 5},
                    ],
                    "failed": [],
                    "warnings": [],
                }
            )

    monkeypatch.setattr(st, "_topology_rust", lambda: _Rust())

    result = st.select_topology({"y": [1, 2, 3], "x": [0, 1, 2]}, "y")

    assert result.rankings == [("b", -2.0), ("a", -1.0)]
