# Contributing

## Analytic Penalties

New analytic penalty primitives must use the registry pattern in `src/terms/penalties`.

The canonical workflow is documented in `docs/architecture/adding_a_penalty.md`: add one penalty module, implement `PenaltyManifest`, and add one `register!(Variant, Type)` line in `src/terms/penalties/mod.rs`.

Do not add manual enum forwarding arms, Arrow-Schur accept-list entries, or registry metadata in separate files. Shared metadata belongs in `PenaltyManifest`.

