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
    monkeypatch.setattr(st, "_formula_for_candidate", lambda formula, auto, candidate, strict_dimension: formula)
    monkeypatch.setattr(st, "fit", lambda data, formula, **kwargs: _Fit(-1.0 if "a" in formula else -2.0, 1.0))
    monkeypatch.setattr(st, "_extract_reml_score_raw", lambda m: float(m.summary()["reml_score"]))
    monkeypatch.setattr(st, "_basis_size", lambda m: 2)
    monkeypatch.setattr(st, "_effective_dim", lambda m: 1.0)
    monkeypatch.setattr(st, "_fitted_or_candidate_null_dim", lambda fit, candidate, basis_size: 0.0)
    monkeypatch.setattr(st, "_score_disagreement_warnings", lambda *args, **kwargs: [])
    monkeypatch.setattr(st, "compare_models", lambda payload, names: {"winner": "b", "ranking": [("b", 0.0), ("a", 0.0)]})

    result = st.select_topology({"y": [1, 2, 3], "x": [0, 1, 2]}, "y")

    assert result.rankings == [("b", result.scores["b"]), ("a", result.scores["a"])], "select_topology should return rankings in the exact score order produced by the ranking/comparison layer."
