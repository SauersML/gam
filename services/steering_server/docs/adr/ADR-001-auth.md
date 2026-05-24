# ADR-001: Authentication

## Status

Accepted for scaffold.

## Decision

The service accepts either `x-api-key` or OAuth2 Bearer credentials through FastAPI Security dependencies. API keys are stored as SHA-256 hashes in SQLAlchemy-managed persistence. A bootstrap user is created from file-based configuration at startup.

## Consequences

This provides one auth path for local, test, and compose deployments. It is not a complete OAuth2 identity-provider integration; production should replace token issuance with an external IdP and keep the same Bearer-token dependency boundary.
