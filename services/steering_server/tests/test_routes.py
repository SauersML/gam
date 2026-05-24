from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_healthz() -> None:
    with TestClient(app) as client:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_auth_required_for_steerers() -> None:
    with TestClient(app) as client:
        response = client.get("/v1/steerers")
        assert response.status_code == 401
