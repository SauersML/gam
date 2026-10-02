import json

import gamfit._select_topology as st


def test_select_topology_reads_the_rust_ranking_and_its_fits(monkeypatch):
    """The candidate loop and the ranking are the Rust owner's (#2899 P10): Python
    marshals the candidates and the request, and reads the ranking back verbatim."""
    requests: list[dict[str, object]] = []

    class _Rust:
        def select_topology_table(
            self,
            headers,
            rows,
            candidates_json,
            defaults,
            score_kind,
            score_scale,
            config_json=None,
            response=None,
            formula=None,
            latent=None,
        ):
            requests.append(
                {
                    "candidates": [row["name"] for row in json.loads(candidates_json)],
                    "defaults": defaults,
                    "score_kind": score_kind,
                    "score_scale": score_scale,
                    "response": response,
                    "formula": formula,
                    "latent": latent,
                }
            )
            ranking = {
                "winner_index": 0,
                "ranked": [
                    {"name": "b", "score": -2.0, "raw_reml": -2.0, "effective_dim": 1.0, "basis_size": 2, "n_obs": 3},
                    {"name": "a", "score": -1.0, "raw_reml": -1.0, "effective_dim": 1.0, "basis_size": 2, "n_obs": 3},
                ],
                "failed": [],
                "warnings": [],
            }
            return json.dumps(ranking), [("a", b"model-a"), ("b", b"model-b")], []

    monkeypatch.setattr(st, "_topology_rust", lambda: _Rust())
    candidates = [
        ("a", st._default_topology_candidate("circle", 1).topology),
        ("b", st._default_topology_candidate("circle", 1).topology),
    ]
    result = st.select_topology(
        {"y": [1.0, 2.0, 3.0], "x": [0.0, 1.0, 2.0]},
        "y",
        candidates,
        score="tk",
        return_fits=True,
    )

    assert requests == [
        {
            "candidates": ["a", "b"],
            "defaults": False,
            "score_kind": "tk",
            "score_scale": "per_observation",
            "response": "y",
            "formula": None,
            "latent": None,
        }
    ]
    assert result.rankings == [("b", -2.0), ("a", -1.0)]
    assert result.winner_name == "b"
    assert result.winner_fit._model_bytes == b"model-b"
    assert set(result.fits or {}) == {"a", "b"}
