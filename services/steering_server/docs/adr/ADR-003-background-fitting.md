# ADR-003: Background Fitting

## Status

Accepted for scaffold.

## Decision

Gauge fitting is routed through an APScheduler-backed queue object. Local configuration runs fits inline for deterministic smoke tests. Docker and Helm configuration set queue execution to background mode.

## Consequences

The queue boundary exists, but APScheduler is process-local. Production multi-replica deployments need a distributed queue with idempotent workers and fit leasing before expensive model fitting is enabled.
