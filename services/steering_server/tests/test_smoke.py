from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


API_KEY = "dev-steering-key"


def test_register_fit_and_steer_smoke() -> None:
    with TestClient(app) as client:
        headers = {"x-api-key": API_KEY}
        create_response = client.post(
            "/v1/steerers",
            headers=headers,
            json={
                "model": "mock-transformer",
                "layer": 12,
                "concept_targets": [
                    {
                        "name": "helpfulness",
                        "target": 0.8,
                        "anchors": ["clear", "direct", "grounded"],
                        "locality_weight": 1.4,
                    }
                ],
            },
        )
        assert create_response.status_code in {200, 201}
        steerer = create_response.json()
        assert steerer["status"] == "ready"

        steer_response = client.post(
            f"/v1/steerers/{steerer['id']}/steer",
            headers=headers,
            json={
                "concept_targets": [
                    {
                        "name": "helpfulness",
                        "target": 0.6,
                        "anchors": ["clear"],
                        "locality_weight": 1.0,
                    }
                ],
                "alpha": 1.2,
                "prompts": ["Explain gauge fitting in one sentence."],
                "max_tokens": 64,
            },
        )
        assert steer_response.status_code == 200
        payload = steer_response.json()
        assert payload["steerer_id"] == steerer["id"]
        assert len(payload["completions"]) == 1
        assert payload["completions"][0]["finish_reason"] in {"stop", "length"}
