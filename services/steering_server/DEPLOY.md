# Deploy

## Compose

`docker-compose.yml` is intended for local integration testing and small single-node demos.

```bash
docker compose up --build
```

Service URLs:

- API: `http://localhost:8000`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000`

## Kubernetes

The Helm chart is in `helm/steering-server`.

```bash
helm lint helm/steering-server
helm upgrade --install steering-server helm/steering-server
```

The chart deploys multiple API replicas and an HPA. It expects Postgres and an OTLP collector to be reachable at the values configured in `values.yaml`.

## Production Gaps

- Replace the mock LLM backend with a real model runtime.
- Move background fitting to a distributed queue before running more than one API replica.
- Provision managed Postgres, Redis, Prometheus, OTLP collector, and Grafana with durable storage and access controls.
- Replace the bootstrap credential with an identity-provider backed OAuth2 flow.
