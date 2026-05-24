# ADR-004: Observability

## Status

Accepted for scaffold.

## Decision

The API exposes Prometheus metrics from `prometheus_client` and instruments FastAPI with OpenTelemetry exported over OTLP. Gauge fitting and steering calls create explicit spans.

## Consequences

Metrics and traces are available in local Compose without changing application code. GPU memory is reported as zero unless a CUDA-enabled PyTorch runtime is present.
