# Adding An Analytic Penalty

Analytic penalties are registered from `src/terms/penalties/mod.rs`. The registry is the only list that should grow when a new primitive is added.

## Steps

1. Add `src/terms/penalties/<snake_name>.rs`.
2. Implement the concrete penalty type and `AnalyticPenalty` behavior in that module, or move an existing implementation there.
3. Implement `PenaltyManifest` for the concrete type:

```rust
impl PenaltyManifest for MyPenalty {
    const KIND_TAG: &'static str = "my_penalty";
    const PYTHON_WRAPPER: &'static str = "MyPenalty";
    const ROW_BLOCK_DIAGONAL: bool = true;
}
```

4. Register it once in `src/terms/penalties/mod.rs`:

```rust
register!(MyPenalty, MyPenalty);
```

The registry macro generates the `AnalyticPenaltyKind` variant and the common dispatch methods (`tier`, `rho_count`, `value`, `grad_target`, `grad_rho`, `hessian_diag`, `hvp`, schedule application, kind tag, Python wrapper name, and row-block eligibility).

## Worked Example

For `MechanismSparsityPenalty`, the manifest lives in `src/terms/penalties/mechanism_sparsity.rs`:

```rust
impl PenaltyManifest for MechanismSparsityPenalty {
    const KIND_TAG: &'static str = "mechanism_sparsity";
    const PYTHON_WRAPPER: &'static str = "MechanismSparsityPenalty";
    const ROW_BLOCK_DIAGONAL: bool = false;
}
```

The single registry line is:

```rust
register!(MechanismSparsity, MechanismSparsityPenalty);
```

Changing its tier stays inside `MechanismSparsityPenalty::tier`. Changing its row-block policy stays inside the manifest. The enum, common dispatch methods, and Arrow-Schur accept-list do not need hand edits.

## Rules

Do not add one-off `AnalyticPenaltyKind::<Variant>` arms for generic dispatch. If a consumer needs metadata that every primitive has, add it to `PenaltyManifest` and consume it through `AnalyticPenaltyKind`.

