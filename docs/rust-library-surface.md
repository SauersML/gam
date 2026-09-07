# Rust library surface and removal decisions

The Rust library is a product, alongside the CLI and Python extension. A public
item reachable through an exported Rust module, re-export, public type, or trait
implementation is a source-graph root in every supported configuration. Its
absence from a CLI or extension symbol table does not make it unused. Generic
instantiation, inlining, and LTO do not change that rule.

The maintained high-level API is `gam::fit_from_formula` for formula/data input,
`gam::materialize` followed by `gam::fit_model` for typed requests, and
`gam::predict` for prediction. These route to the same production model owners
used by the CLI and Python wrapper. Domain crates also expose their documented
numerical and model-building contracts directly; consumers do not have to link
either executable to use them.

Rust's `dead_code` analysis operates on source definitions and visibility,
including generic bodies and re-exports. The workspace's deny-warnings production
builds retain exported library items and reject unreferenced private production
items. Test targets must also compile: test helpers stay under `#[cfg(test)]`
and are evaluated within their test source graph. A private production helper
used only by tests is a candidate to move into test scope, not a reason to delete
the tests. Compiler acceptance establishes source validity, not whether a public
contract is valuable; removing a public contract still requires a semantic
decision and review of its callers and documentation.

There is no compatibility obligation to recreate deleted convenience names.
Keep one current API per behavior, and use explicit model inputs where defaults
would silently choose geometry. An exported name with no internal caller can
still be useful; an abandoned configuration/report carrier with no callable
producer is not a substitute for an implemented public contract.

## Decisions for the concrete #2829 cases

| Historical item | Current contract and decision |
| --- | --- |
| `fit_gam` | Maintain `gam::fit_model` / `gam::fit_from_formula` as the unified model front doors and `gam_solve::estimate::fit_gam_with_penalty_specs` as the explicit external-design API. The generic owned/borrowed design contract is exercised by `public_library_surface_2829.rs`. Recreating an alias that only supplies defaults is unnecessary. |
| `canonicalize_for_identifiability` | Maintain `gam_identifiability::canonical::canonicalize_for_identifiability_with_operating_scalars`. `None` explicitly requests the zero-state audit; supplied scalars audit the actual family operating point. `canonical_recovery.rs` exercises this public route. |
| `under_identified_subspace` | Maintain `gam_solve::estimate::reml::jeffreys_subspace::under_identified_subspace_in_metric`. The caller supplies the model's metric; an identity-metric convenience must not hide a coordinate convention. The public integration target tests both identity coordinates and congruent nonorthogonal coordinates. |
| `arrow_log_det_from_cache` | The public `ArrowSchurCache::arrow_log_det` reads the recorded undamped criterion value. `compute_undamped_arrow_log_det` and `undamped_arrow_log_det_with_schur` own its construction. Do not duplicate this with an evidence-module accessor. |
| `matrix_free_arrow_evidence_log_det` | Maintain `matrix_free_arrow_evidence_log_det_surrogate`, with an explicit rational-lane state for a differentiable frozen criterion and `None` for value-only SLQ. Internal documentation now names this actual public entry point. |
| `coordinate_block_log_det`, `criterion_as_atoms` | The current exact-A criterion assembles `log_det` and `log_det_tt` from the same observed information and uses `rank_adjusted_quasi_laplace_complexity`. The removed majorizer convenience is not the ranked coordinate term. Independent adjoint/criterion acceptance is tracked by #2333 and #2828. |
| `Dual2::{seed_directional,from_channels,seed_inner,seed_outer}` | Maintain the public `Dual2` value/first/second fields, `constant`, `variable`, and `Dual22::channels`. The analytic polynomial channel-order test in `gam-math` verifies the actual representation; redundant setters and getter/setter roundtrips are not separate product behavior. |
| `enable_outer_gradient_fd_capture` | Maintain `enable_outer_gradient_fd_capture_over_theta`; it arms the complete current diagnostic and owns its typed sink. Consumers now document that actual function. |
| `triangle_cocycle_defect`, `triangle_sign_product` | Consumers compose the public `ChartTransition` rotations/signs. The restored `local_chart_recovery_tests.rs` independently checks the composition on plane, sphere-band, and Swiss-roll fixtures; no duplicate arithmetic wrapper is required. |
| `LocalAtlas::co_collapse_candidates` | **Unresolved.** The three historical #2280 co-collapse contracts have no verified replacement. Public transition composition does not establish duplicate-chart detection. #2280 must resolve this behavior explicitly. |
| `cofit_block_and_curved`, `cofit_linear_via_arrow`, `cofit_composed_via_arrow` | **Unresolved.** Their modules currently retain configuration/report types but no callable producer. `tiered::fit_tiered` is the active fitting route; its fixed-point and stationarity contracts must be compared to the five historical #2023 pins before the old bridge is retired. |

This is a decision record for the issue's concrete examples, not a claim that all
1,206 historical public declarations have been audited. Remaining identities must
be resolved by crate, module/type, and signature; a same-named definition in a
different owner is not evidence of recovery. #2829 remains open until the
remaining public behavior decisions and their acceptance evidence are complete.

The external acceptance targets executed on MSI: two public-library tests, four
structural-coordinate tests, and four measured-span tests passed. The generic
fit test instantiates both owned and borrowed designs and verifies the exact
normal equations including the solver's declared stabilization ridge, with
bit-identical coefficients across ownership forms. The measured-span shear
control demonstrates that supplying the wrong metric changes the selected
dimension. These are compiled external consumers, not symbol-table probes.
Execution receipts and remaining historical acceptance gaps are recorded in
[`test-census-2818-recovery.md`](test-census-2818-recovery.md).
