# Steering Server

FastAPI service scaffold for registering gauge-fitted LLM concept steerers and applying them to prompt batches.

## Local Run

Install the service dependencies from this directory:

```bash
python -m pip install -r requirements.txt
uvicorn app.main:app --reload
```

The local bootstrap credential is:

```text
x-api-key: dev-steering-key
```

The service creates a SQLite database at `services/steering_server/steering_server.db` by default.

## Implemented Surface

- `POST /v1/steerers`
- `GET /v1/steerers`
- `POST /v1/steerers/{steerer_id}/steer`
- `GET /v1/steerers/{steerer_id}/diagnostics`
- `WebSocket /v1/stream`
- `GET /healthz`
- `GET /metrics`

Gauge fitting and steering use deterministic local implementations so requests produce stable, typed responses. The implementation is deliberately marked with `stub: to-be-implemented` fields where a real hidden-state gauge fitter and LLM runtime must be integrated.

## Docker Compose

```bash
docker compose up --build
```

Compose starts the API, Postgres, Redis, Prometheus, an OTLP collector, and Grafana. The API container copies `config/docker.toml` to `config/local.toml`, switching persistence to Postgres and scheduler execution to background mode.

## Smoke Flow

```bash
curl -X POST http://localhost:8000/v1/steerers \
  -H 'content-type: application/json' \
  -H 'x-api-key: dev-steering-key' \
  -d '{"model":"mock-transformer","layer":12,"concept_targets":[{"name":"helpfulness","target":0.8,"anchors":["clear","grounded"],"locality_weight":1.2}]}'
```

Use the returned `id` with `POST /v1/steerers/{id}/steer`.
