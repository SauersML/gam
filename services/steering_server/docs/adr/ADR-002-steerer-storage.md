# ADR-002: Steerer Storage

## Status

Accepted for scaffold.

## Decision

Steerer registrations are persisted in SQLAlchemy tables with owner, model, layer, normalized concept target JSON, deterministic concept-target hash, fit status, gauge payload, diagnostics payload, and error state.

## Consequences

SQLite works for local development and Postgres works in Docker and Kubernetes. The gauge is stored as JSON because the current implementation is deterministic metadata. Production gauge tensors should move to object storage or a vector/tensor store with only references kept in SQL.
