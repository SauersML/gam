# #1521 Carve — Back-Edge Inventory & Resolution Plan

Worktree `/Users/user/gam-wt-carve`, HEAD `1548e78da`. READ-ONLY; no edits made.

Intended DAG (low→high): `gam-spec`/`gam-problem` < `gam-math`/`gam-runtime`/`gam-linalg`/`gam-geometry`
< `gam-model-api`/`gam-gpu`/`gam-identifiability` < `gam-terms` < `gam-solve` (generic, family-agnostic)
< `gam-models` (concrete families + GAM fit_orchestration) < `gam-predict` < `gam-inference` < `gam-sae`
< `gam-pyffi`/`gam-cli`.

Confirmed Cargo deps: `gam-solve` deps `gam-terms` but **NOT** `gam-models`. `gam-models` deps
`gam-solve`+`gam-terms`+`gam-model-api`+`gam-problem`. `gam-terms` deps neither `gam-solve` nor
`gam-models`. `gam-terms` has **no** internal `solver`/`families`/`estimate`/`pirls`/`custom_family`
modules and **no** direct `gam_solve::`/`gam_models::` refs — so every `crate::solver::*` /
`crate::estimate::*` / `crate::families::*` in `gam-terms/src` is a **leftover-monolith path that does
not resolve**: `gam-terms` does not currently compile. These are the back-edges.

`gam-solve` crate root is `lib.rs`→`include!("mod.rs")`; it has **no** `solver` submodule — monolith
`crate::solver::rho_optimizer` etc. map to top-level `crate::rho_optimizer` here. So driver path rewrites
are: `crate::solver::rho_optimizer`→`gam_solve::rho_optimizer`, `crate::solver::estimate::reml::X` /
`crate::estimate::X`→`gam_solve::estimate::X`, `crate::solver::latent_cache`→`gam_solve::latent_cache`,
`crate::solver::glm_sufficient_lane`→`gam_solve::glm_sufficient_lane`, `crate::pirls`→`gam_solve::pirls`,
`crate::custom_family`/`crate::families::custom_family`→`gam_solve::custom_family`,
`crate::families::{block_layout,family_runtime,...}`→`crate::` (gam-models self).

---

## ZONE A — `gam-terms` → higher crates

### A-summary (compile-breaking sites)
| file | refs | targets | strategy |
|---|---|---|---|
| `smooth/spatial_optimization.rs` | 70 | solver/estimate/pirls/custom_family/families | **UP** (whole file) |
| `smooth/design_construction.rs` | 40 | solver/estimate/pirls/custom_family/families | **UP** (whole file) |
| `smooth/prelude.rs` | 3 use-blocks | custom_family, estimate, families::family_runtime | **UP** (split: gam-solve/models imports lift) |
| `smooth/error.rs:12` | 1 | `families::block_layout::block_count::BlockCountMismatch` | **CD** → gam-problem |
| `decoders/behavioral_head.rs:56` | 1 | `solver::row_measure::RowSubsampleMask` | **CD** → gam-problem (see note) |
| `structure/anova_atom.rs:95` | 1 | `solver::grid_spline_2d::{GridSpline2dDesign, axis_basis_at}` | **CD/down** → gam-terms |

Doc-comment-only (non-breaking, rewrite the `[\`...\`]` link target):
- `latent.rs:223` `/// [\`crate::solver::arrow_schur::ArrowSchurSystem::apply_riemannian_latent_geometry\`]`
- `structure/coefficient_group_resolver.rs:7` `//! [\`crate::families::custom_family\`]`

Internal-and-fine (NOT back-edges — `gam-terms` owns `pub mod inference`): every `crate::inference::*`
(lawley/higher_order/structure_evidence/smooth_test/formula_dsl), incl. `term_builder.rs:23/4250`,
`term_specs.rs:863`, `decoders/behavioral_head.rs:54-55`, `structure/anova_atom.rs:92`. No `crate::model_types`
hits. `EstimationError` already lives in `gam-problem` (descended) — `FittedTermCollection`'s `EstimationError`
field is fine.

### A-detail / classification

**1. The two drivers — `smooth/design_construction.rs` (8331 lines) + `smooth/spatial_optimization.rs`
(9185 lines).** Both are `include!`d into `mod smooth` (see `smooth.rs:26-27`). Their `crate::` prefix
histogram (combined): `crate::solver::rho_optimizer` ×23, `crate::estimate::reml` ×21,
`crate::solver::estimate` ×19, `crate::solver::latent_cache` ×14, `crate::custom_family::*` ×3+,
`crate::estimate::{fit_gam_with_penalty_specs,dispersion_from_likelihood}` ×3, `crate::pirls::PirlsStatus`
×4, `crate::families::custom_family::OuterDerivativePolicy` ×1, `crate::families::block_layout::…
::validate_block_count` ×1. **These are the REML-outer / PIRLS / custom-family fit orchestration loop** —
genuinely higher-tier (gam-models GAM-fit layer), misplaced in gam-terms. **→ UP**: move both files (and
their `FittedTermCollection*` structs, below) into `gam-models` (the GAM fit_orchestration layer). After the
move every `crate::`-path above is reachable (gam-models deps solve+terms).

**2. `smooth/prelude.rs` (shared import surface for all of `mod smooth`).** Imports a gam-terms-internal
half (`crate::basis::*`, `crate::construction::*`, `penalty_priors::*`, `crate::mixture_link::*`,
`crate::util::quantile`, plus gam_linalg/gam_problem/gam_spec — all fine) and a **gam-solve/models half** that
is the back-edge: `use crate::custom_family::{CustomFamily, BlockwiseFitOptions, fit_custom_family, …}` (line
32), `use crate::estimate::{EstimationError, UnifiedFitResult, PenaltySpec, FitOptions, …, reml::DirectionalHyperParam}`
(line 40), `use crate::families::family_runtime::{FamilyStrategy, strategy_for_spec}` (line 48). **→ UP**:
the custom_family/estimate/family_runtime import block lifts with the drivers (only the drivers consume these);
gam-terms keeps the basis/construction/penalty_priors block. *Residual to verify post-lift:* staying
`term_specs.rs` uses bare `PenaltySpec` ×3 (`term_specs.rs:4621/4635/4639`, `Vec<PenaltySpec>`,
`PenaltySpec::Dense`, `PenaltySpec::from_blockwise`). gam-terms **already has its own**
`crate::penalty_spec::PenaltySpec` (`gam-terms/src/penalty_spec.rs:37`, has `Dense`) — repoint these 3 to it
(CD, type already descended) OR confirm the enclosing fn is itself driver-orchestration that lifts. Decide
during M4.

**3. `smooth/error.rs:12` — `BlockCountMismatch` + `design_construction.rs:6405` `validate_block_count`.**
Home: `gam-models/src/block_layout/block_count.rs`, which has **zero `use`/`crate::` deps** (fully
self-contained). **→ CD**: descend the whole `block_count` module (`BlockCountMismatch` + generic
`validate_block_count<E>`) to `gam-problem`. Unblocks staying `error.rs`; the lifted driver also reaches it.
(gam-models'/gamlss'/survival's other `BlockCountMismatch` types are unrelated namesakes.)

**4. `decoders/behavioral_head.rs:56` — `RowSubsampleMask`.** Home: `gam-solve/src/row_measure.rs`
(197 lines). Single import + used only in `pub fn with_row_measure` (line 183). **→ CD** (descend
`RowSubsampleMask` to gam-problem). **NOTE/risk:** `row_measure.rs` also defines `from_options(&BlockwiseFitOptions,…)`
and wraps `OuterScoreSubsample` (gam-solve types), so a *full* module descent drags those down too. Cleanest
minimal cut: descend only the `RowSubsampleMask` data type (Arc<indices>+weights) + `full_data`/`subsample`/
`indices_and_weights` to gam-problem, and leave the `BlockwiseFitOptions`-coupled `from_options` constructor
up in gam-solve. Alternative if that's messy: **UP** the `with_row_measure` constructor into gam-models
(behavioral-head fit assembly). Recommend CD-minimal.

**5. `structure/anova_atom.rs:95` — `GridSpline2dDesign`, `axis_basis_at`.** Home:
`gam-solve/src/grid_spline_2d.rs`, **zero `crate::families/estimate/pirls/fit_orchestration` deps** — it is 2-D
spline basis/design machinery sitting in the wrong crate. **→ CD/down**: descend `grid_spline_2d` from
gam-solve into `gam-terms` (basis/construction tier). Clean (self-contained). *Caveat:* verify it imports no
other gam-solve-only top-level symbol before moving (greps show only ndarray/linalg-level use).

---

## ZONE B — `gam-solve` → `gam-models` concrete families

`gam-solve` has **no** internal `families`/`family_runtime`/`block_layout` dir; every `crate::families::*` /
`crate::family_runtime::*` / `crate::block_layout::*` resolves into `gam-models`. Outside `fit_orchestration/`,
only **two** files carry such refs: `protocol.rs` and `gpu/pirls_gpu.rs` (verified by exclusion grep).

### B-summary
| file(s) | refs | strategy |
|---|---|---|
| `fit_orchestration/**` (8 files w/ refs, 19 .rs total, 12 655 lines) | ~57 | **UP** (lift whole subtree to gam-models) |
| `protocol.rs` (124 lines) | 2 | **UP** (file is family-specific) |
| `gpu/pirls_gpu.rs` device-PCG path | 1 prod field + 2 prod calls + 3 test | **UP** (recommended) / **TI** (alt) |

Per-file `crate::{families,family_runtime,block_layout}` counts: `fit_orchestration/fit.rs` 19,
`materialize/survival.rs` 11, `fit_orchestration.rs` 11, `request.rs` 6, `materialize/survival_time.rs` 6,
`gpu/pirls_gpu.rs` 4, `protocol.rs` 2, `materialize/family.rs` 2, `materialize/{validation,standard}.rs` 1 each.

### B-detail

**1. `fit_orchestration/**` → UP (one move).** Target symbols are concrete families: `crate::families::survival`
(`construction` ×17, `location_scale`, `royston_parmar`, `lognormal_kernel`, `cause_count_from_event_codes`,
`latent`, …), `crate::families::multinomial::fit_penalized_multinomial`, `crate::families::transformation_normal`,
`crate::families::marginal_slope_orthogonal::score_influence_jacobian`, `crate::families::gamlss::gaussian`. This is
the concrete GAM fit-assembly layer — squarely gam-models. After lift, `crate::families::X` → `crate::X`
(self-ref in gam-models). See ZONE C for the clean-slice proof.

**2. `protocol.rs` → UP.** Defines `LatentScoreSemantics`, `MarginalSlopeCalibrationProtocol`,
`SurvivalMarginalSlopeProtocol` — all marginal-slope/survival-specific — and imports
`crate::families::bms::{DEFAULT_EMPIRICAL_LATENT_GRID_SIZE, DeviationBlockConfig, LatentMeasureSpec,
LatentZCheckMode, LatentZNormalizationMode, LatentZPolicy}` (lines 1-4, home `gam-models/src/bms/mod.rs`) and
`crate::families::survival::construction::SurvivalBaselineTarget` (line 5, home
`gam-models/src/survival/{construction,predict}.rs`). **The whole file is family-specific** (no generic
solver code), and **no gam-solve-core consumer** (`mod protocol` at `mod.rs:30` is the only intra-solve ref;
`grep` finds no other `protocol::` user). **→ UP** to gam-models (e.g. `gam-models/src/marginal_slope/protocol.rs`),
not CD: these aren't neutral contracts, they're concrete family policy. No external consumers found, so the
move is consumer-free.

**3. `gpu/pirls_gpu.rs` device-resident row-Hessian PCG.** Production back-edge:
`DeviceResidentPcgInput.storage: &crate::families::bms::gpu::row::DeviceResidentRowHess` (line **3466**). The
**production** `mod pcg_device` (line 3566, `#[cfg(target_os="linux")]`, NOT a test) calls the BMS launches
directly inside the CG loop: `launch_bms_flex_row_diagonal(input.storage)` (line **3689**, Jacobi
preconditioner) and `launch_bms_flex_row_hvp_into_device(input.storage, …)` (line **3836**, the matvec).
Imports at 3570/3571 are in `pcg_device` (production); refs at 4524 are `#[cfg(test)]`. The public entry
`run_pcg_against_row_hessian_device` (line 4055) has **no caller** outside the file's own tests (no gam-models
/ gam-pyffi consumer found) — it's a wired-but-currently-test-only BMS-flex artifact.
  - **Recommended → UP**: move `DeviceResidentPcgInput/Output`, `pcg_device`, `run_pcg_against_row_hessian_device`,
    and the parity tests into `gam-models/src/bms/gpu/` (next to `row::`). Rationale: the entire path exists
    only to serve BMS-flex (operator + preconditioner are BMS launches), has no other consumer, and the
    generic device PCG need is already met by `arrow_schur`'s generic PCG in gam-solve. Lowest surface; no
    cudarc trait plumbing across the `gam-gpu`/`gam-model-api` boundary.
  - **Alternative → TI** (only if a second device-row-Hessian family ever appears): introduce trait
    `DeviceResidentRowHvp` (in `gam-gpu` or `gam-model-api`) with `fn p_total(&self)->usize`,
    `fn diagonal_host(&self, …)->Result<Vec<f64>,String>`, `fn hvp_into_device(&self, d_p, d_q, stream)->Result<(),String>`;
    `bms::gpu::row::DeviceResidentRowHess` implements it in gam-models; `DeviceResidentPcgInput` becomes
    `storage: &dyn DeviceResidentRowHvp` and `pcg_device` stays generic in gam-solve. Heavier (cudarc
    `CudaStream`/device-buffer types cross the boundary). Not worth it for a single consumer today.

---

## ZONE C — `fit_orchestration` lift: clean top-slice proof

**Claim: gam-solve core touches `fit_orchestration` via exactly one re-export and nothing consumes it.**
Verified:
- `mod.rs:9` `pub mod fit_orchestration;` (declaration) and `mod.rs:101`
  `pub use fit_orchestration::descriptors::build_analytic_penalty_registry_from_descriptors;` are the **only**
  gam-solve-core references to the subtree.
- `grep 'fit_orchestration::' outside fit_orchestration/` → **only** `mod.rs:101`. No core module calls any
  `fit_orchestration::` item.
- The re-exported `build_analytic_penalty_registry_from_descriptors` has **no gam-solve-core consumer** — its
  only callers are *inside* fit_orchestration (`materialize/latent.rs:701`, `descriptors.rs` tests). So the
  re-export line is dead weight for core.

**Therefore the ONLY gam-solve-core→fit_orchestration reference that breaks on move is `mod.rs:101`** (plus the
`mod.rs:9` declaration) — both simply **deleted**, no core re-wiring. The subtree lifts wholesale to gam-models.

**Downstream repoint (external consumers of `gam_solve::fit_orchestration::…`):** gam-models (`multinomial.rs`,
`sigma_link.rs`, `gamlss/dispersion_family.rs`, `survival/base.rs` — become `crate::fit_orchestration`),
gam-pyffi (`latent_basis_and_sae_ffi.rs`, `geometry_ffi.rs`, `ffi_prelude.rs`, `manifold_and_posterior_ffi.rs`),
gam-cli (`main.rs`, `main/family_resolve.rs`, `main/run_fit.rs`). All already dep gam-models → mechanical
`gam_solve::fit_orchestration`→`gam_models::fit_orchestration` rename.

---

## ZONE D — driver / `term_specs` spec-accessor cut

`prelude.rs`, `term_specs.rs` (7864 lines), `design_construction.rs`, `spatial_optimization.rs` are all
`include!`d into one `mod smooth` (`smooth.rs:2,18,26,27`) → they currently share **private** visibility. After
the drivers lift to gam-models, cross-references must go through `pub`.

### D.1 — Orchestration that LIFTS UP (out of `term_specs.rs`, with the drivers)
The **only** hard back-edge inside `term_specs.rs` is the two structs that hold `UnifiedFitResult` (home
`gam-solve/src/model_types/result_types.rs`):
- `FittedTermCollection { fit: UnifiedFitResult, design, adaptive_diagnostics }` (`term_specs.rs:2398`)
- `FittedTermCollectionWithSpec { fit: UnifiedFitResult, design, resolvedspec, adaptive_diagnostics, kappa_timing }` (`term_specs.rs:2432`)
- carry along `SpatialLengthScaleOptimizationTiming` (`term_specs.rs:2404`; pure f64s, no back-edge — moves with
  `FittedTermCollectionWithSpec` which owns it via `kappa_timing`).

These are used **only by the two drivers** (every `FittedTermCollection*` reference is in
`spatial_optimization.rs` / `design_construction.rs`; none elsewhere). → lift to gam-models with the drivers.
Everything else in `term_specs.rs` is pure spec/design machinery and **stays**.

### D.2 — Pure spec-accessors that STAY DOWN in `gam-terms` (must be made `pub`)
The lifted drivers call these `term_specs`-defined helpers; make them `pub` so gam-models can reach them
(gam-models deps gam-terms). Real (non-std/ndarray) ones, by driver call-count:
`spatial_term_psi_to_length_scale_and_aniso` (6), `constant_curvature_term_spec` (5),
`spatial_term_supports_hyper_optimization` (3), `set_spatial_aniso_log_scales` (3), `realized_design_column` (3),
`plan_joint_spatial_centers_for_term_blocks` (3), `measure_jet_term_spec` (3), `measure_jet_enrolls_psi` (3),
`get_spatial_feature_dim` (3), `get_spatial_aniso_log_scales` (3), `constant_curvature_kappa_bounds` (3),
`build_smooth_design_withworkspace_unvalidated` (3), `weighted_blockwise_penalty_sum` (2),
`validate_term_collection_finite_inputs` (2), `realize_coefficient_groups` (2),
`normalize_penalty_in_constrained_space` (2), `measure_jet_psi_dim` (2), `log_spatial_aniso_scales` (2),
`kronecker_penalty_system` (2), `build_single_local_smooth_term` (2), `build_random_effect_block` (2),
`assemble_term_collection_design_matrix` (2), and ×1: `validate_frozen`,
`transform_penalized_hessian_to_original`, `transform_blockwise_penalties_to_internal`,
`standardized_spatial_term_data`, `spatial_term_has_locked_kappa`, `spatial_term_center_strategy`,
`set_single_term_measure_jet_psi_dials`, `set_single_term_constant_curvature_kappa`,
`set_measure_jet_psi_dials`, `set_constant_curvature_kappa`,
`matern_operator_penalty_triplet_from_metadata`, `is_marginally_centered_tensor`, `internal_bounds_for`,
`build_smooth_design`. (Plus `TermCollectionSpec`/`TermCollectionDesign` impl methods invoked by the drivers —
those `impl` blocks get `pub` methods.)

### D.3 — The reverse cut (verify, don't assume)
Some names *defined in the driver files* appear in `term_specs.rs` — if any is genuinely driver-owned and
called by **staying** code, it must NOT lift (stays down) — but none creates a cycle (it just stays). Candidates
to disambiguate (most are method/field accessors `.spec/.design/.label` = false positives): `get_spatial_length_scale`,
`frozen_global_orthogonality`, `spatial_term_uses_per_axis_psi`, `apply_global_smooth_identifiability`,
`freeze_term_collection_from_design`, `try_build_spatial_term_log_kappa_derivativeinfo`,
`try_build_spatial_term_log_kappa_aniso_derivativeinfos`, `freeze_smooth_basis_from_metadata`,
`canonical_penalties_at_psi`, `build_term_collection_design_inner`, `build_parametric_constraint_block_for_term`.
Action during M4: for each, confirm its `fn` definition is in `term_specs.rs`/`structure_analysis.rs` (→ stays
pub) vs in a driver file (→ if called by staying code, keep it down). **No bidirectional type dependency exists**
(drivers consume term_specs spec types; term_specs does not need any driver-only *type*) → **no true cycle**.

---

## Resolution DAG (ordering so each step leaves lower tiers compilable)

gam-terms must compile before gam-solve (gam-solve deps gam-terms); gam-solve before gam-models.

**Phase 1 — make `gam-terms` compile (remove all ZONE-A back-edges).** Independent moves, parallelizable:
- **M1 (CD)** `block_count` → gam-problem. Unblocks `error.rs`. *Independent.*
- **M2 (CD/down)** `grid_spline_2d` gam-solve → gam-terms. Unblocks `anova_atom.rs`. *Independent.*
- **M3 (CD)** `RowSubsampleMask` → gam-problem (minimal data-type slice). Unblocks `behavioral_head.rs`. *Independent.*
- **M4 (UP, the big extraction)** physically remove `design_construction.rs` + `spatial_optimization.rs` +
  `FittedTermCollection*`/`SpatialLengthScaleOptimizationTiming` from gam-terms; split `prelude.rs`
  (gam-terms keeps basis/construction block, the custom_family/estimate/family_runtime block leaves).
  **Coupled with M5** (make D.2 accessors `pub` first/simultaneously) and **M6** (below). After M1-M4,
  gam-terms has zero back-edges → compiles. Fix the residual `term_specs.rs` `PenaltySpec` ×3 → gam-terms
  `penalty_spec::PenaltySpec`. Rewrite doc links `latent.rs:223`, `coefficient_group_resolver.rs:7`.
- **M5 (down/visibility)** `pub` the D.2 spec-accessors. *Must precede/accompany M4.*

**Phase 2 — make `gam-solve` compile (remove ZONE-B back-edges).** All independent of each other and of M4:
- **M6 (visibility)** gam-solve `latent_cache` `pub(crate)`→`pub` (`mod.rs:17`) — drivers (now in gam-models)
  call `crate::solver::latent_cache` ×14. *Required by M4's landing in gam-models.*
- **M7 (UP)** lift `fit_orchestration/**` → gam-models; delete `mod.rs:9`+`mod.rs:101`; repoint
  gam-pyffi/gam-cli/gam-models consumers (ZONE C). *Independent.*
- **M8 (UP)** lift `protocol.rs` → gam-models. *Independent.*
- **M9 (UP)** lift `gpu/pirls_gpu.rs` device-PCG block → gam-models/bms/gpu. *Independent.*

**Phase 3 — `gam-models` receives M4/M7/M8/M9.** Rewrite the moved files' `crate::` paths per the mapping at
top (`crate::solver::X`→`gam_solve::X`, `crate::estimate/pirls/custom_family`→`gam_solve::…`,
`crate::families::Y`→`crate::Y`). gam-models deps solve+terms+model-api+problem → all resolve.

**Parallelism:** M1, M2, M3, M7, M8, M9 are mutually independent (disjoint files/types) — six concurrent
agents. M4+M5+M6 are one coupled unit (the driver lift). M6 lands with Phase 2 but is *consumed* by M4 in
Phase 3 — sequence M5→M4-extract→(M6 ready)→M4-reland.

**True cycles:** **none.** Every ZONE-A edge is CD or UP-of-misplaced-orchestration; every ZONE-B edge is
UP-of-family-specific-code (with a TI fallback for B-3). The only candidate cycle (staying `term_specs` calling
a driver-owned fn) resolves by *leaving that fn down* (D.3) — a one-directional dependency, never a cycle. The
`fit_orchestration` slice is provably clean (ZONE C). Proven none.

---

## Sanity numbers (agent sizing)

**Back-edge counts per zone:**
- ZONE A (gam-terms→higher), compile-breaking: **~113 refs across 6 files** (70 spatial_opt + 40 design_construction
  + 3 prelude + 1 error + 1 behavioral_head + 1 anova_atom) + 2 doc-only.
- ZONE B (gam-solve→gam-models): **~62 refs** — ~57 in `fit_orchestration/**`, 2 protocol, 3 prod pirls_gpu
  (+3 test).
- ZONE C: **1** core→slice reference to delete (`mod.rs:101`, +`mod.rs:9` decl).

**Per-move file/line sizing:**
| move | files | lines moved/touched |
|---|---|---|
| M1 block_count CD | 1 (`block_count.rs`) | ~small, self-contained module |
| M2 grid_spline_2d down | 1 (`grid_spline_2d.rs`) + `anova_atom.rs` import | self-contained module + 1 import |
| M3 RowSubsampleMask CD | `row_measure.rs` (197) partial + `behavioral_head.rs` import | ~60-100 lines (type+3 ctors) |
| M4 driver lift | `design_construction.rs` (8331) + `spatial_optimization.rs` (9185) + `term_specs.rs` 2 structs (~40) + `prelude.rs` split (~30) | **~17 600 lines** — biggest; one focused agent |
| M5 spec-accessor pub | `term_specs.rs` in place | ~40 fn/`impl` signatures `pub` |
| M6 latent_cache pub | `mod.rs:17` | 1 line |
| M7 fit_orchestration lift | 19 .rs files (12 655 lines) + 11 consumer files repoint | **~12 700 lines** + ~11 import-renames |
| M8 protocol lift | `protocol.rs` (124) + `mod.rs:30` | ~125 lines |
| M9 pirls_gpu device-PCG lift | `gpu/pirls_gpu.rs` ~1300-line block | ~1300 lines (struct+pcg_device+entry+tests) |

Two ~12-17 kLOC moves (M4, M7) dominate — give each a dedicated agent. M1/M2/M3/M6/M8 are small and
parallel; M9 is medium.
