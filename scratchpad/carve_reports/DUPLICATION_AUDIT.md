# Duplication Audit: Carved GAM Workspace

**Date**: 2026-06-26  
**Scope**: Comprehensive search for duplicated type definitions, functions, constants, modules, and macros across the carved workspace.  
**Total Duplicates Found**: 100+ across all categories.

---

## 1. DUPLICATE STRUCT DEFINITIONS

### TRUE DUPLICATES (Byte-Identical Definitions)

These types are defined identically in 2+ crates. **Candidates for unification + re-export:**

#### Group A: Result/Model Types (src/ + gam-solve/)
**Status**: `DUPLICATE-COLLAPSE` — Unify in gam-solve, re-export from src/model_types.rs

- **FitOptions** (structs with same fields)
  - src/model_types/result_types.rs:683
  - crates/gam-solve/src/model_types/result_types.rs:683
  - Same fields, same defaults → identical

- **FitArtifacts** (structs with same fields)
  - src/model_types/result_types.rs:797
  - crates/gam-solve/src/model_types/result_types.rs:797
  - Identical implementation

- **FitInference** (structs with same fields)
  - src/model_types/result_types.rs:868
  - crates/gam-solve/src/model_types/result_types.rs:868
  - Identical implementation

- **FittedBlock** (structs with same fields)
  - src/model_types/result_types.rs:1037
  - crates/gam-solve/src/model_types/result_types.rs:1037
  - Identical implementation

- **FitGeometry** (structs with same fields)
  - src/model_types/result_types.rs:1051
  - crates/gam-solve/src/model_types/result_types.rs:1051
  - Identical implementation

- **UnifiedFitResultParts** (structs with same fields)
  - src/model_types/result_types.rs:1062
  - crates/gam-solve/src/model_types/result_types.rs:1062
  - Identical implementation

- **UnifiedFitResult** (structs with same fields)
  - src/model_types/result_types.rs:1103
  - crates/gam-solve/src/model_types/result_types.rs:1103
  - Identical implementation

- **CriterionCertificate** (MULTIPLE LOCATIONS, 2 different types!)
  - src/model_types/result_types.rs:602 (inference-layer certificate for fit optimization audits)
  - crates/gam-solve/src/model_types/result_types.rs:602 (DUPLICATE of src/)
  - crates/gam-sae/src/certificates.rs:58 (DIFFERENT type — SAE/manifold criterion, different fields)
  - **Note**: src/ and gam-solve/ versions are byte-identical. gam-sae version is a coincidental name collision (different logical type). Keep gam-sae version, collapse src+gam-solve to one.

- **AdaptiveRegularizationOptions** (structs with same fields)
  - src/model_types/result_types.rs:768
  - crates/gam-solve/src/model_types/result_types.rs:768
  - Identical implementation

- **FittedLinkState** (enum with same variants)
  - src/model_types/result_types.rs:935
  - crates/gam-solve/src/model_types/result_types.rs:935
  - Identical enum variants

#### Group B: Inference HMC Layer (src/ + gam-inference/)
**Status**: `DUPLICATE-COLLAPSE` — Primary location should be gam-inference; src/ re-exports

The entire HMC/NUTS inference module is duplicated across src/inference/hmc_io.rs (7,727 lines) and crates/gam-inference/src/hmc.rs (8,023 lines). The crate version is slightly newer (line count diff suggests additions). Related struct duplicates:

- **NutsPosterior** (identical)
  - src/inference/hmc_io.rs:641
  - crates/gam-inference/src/hmc.rs:641

- **NutsConfig** (identical)
  - src/inference/hmc_io.rs:4009
  - crates/gam-inference/src/hmc.rs:4009

- **NutsResult** (identical)
  - src/inference/hmc_io.rs:4302
  - crates/gam-inference/src/hmc.rs:4302

- **GaussianModePosterior** (identical)
  - crates/gam-problem/src/laplace_sampler_contract.rs:109
  - crates/gam-inference/src/hmc.rs:4957
  - **Note**: Also in gam-problem; gam-inference version should be canonical

- **GlmFlatInputs<'a>** (identical)
  - src/inference/hmc_io.rs:5210
  - crates/gam-inference/src/hmc.rs:5237

- **SurvivalFlatInputs<'a>** (identical)
  - src/inference/hmc_io.rs:5234
  - crates/gam-inference/src/hmc.rs:5261

- **SurvivalNutsInputs<'a>** (identical)
  - src/inference/hmc_io.rs:5249
  - crates/gam-inference/src/hmc.rs:5276

- **LinkWiggleSplineArtifacts** (identical)
  - src/inference/hmc_io.rs:5569
  - crates/gam-inference/src/hmc.rs:5596

- **LinkWigglePosterior** (identical)
  - src/inference/hmc_io.rs:5580
  - crates/gam-inference/src/hmc.rs:5607

- **LaplaceTrustworthiness** (IMPORTANT: Two different locations)
  - crates/gam-problem/src/laplace_sampler_contract.rs:47 (contract/interface definition)
  - crates/gam-inference/src/hmc.rs:6477 (implementation mirror)
  - **Status**: `INTENTIONAL` — One is the contract interface, one is the impl; verify they're in the right places

- **JointBetaRhoResult** (identical)
  - src/inference/hmc_io.rs:6733
  - crates/gam-inference/src/hmc.rs:7029

- **JointBetaRhoInputs<'a>** (identical)
  - src/inference/hmc_io.rs:7305
  - crates/gam-inference/src/hmc.rs:7601

#### Group C: Custom Family Block Specs (3-way duplication)
**Status**: `DUPLICATE-COLLAPSE` — API defs in gam-model-api/gam-problem, solver impls in gam-solve

The block_spec and coefficient group specifications appear in three places with slightly different purposes:

- **CoefficientGroupSpec** (3 locations, different purposes)
  - src/families/custom_family/block_spec.rs:50 (ORPHANED — see section 4)
  - crates/gam-terms/src/smooth/penalty_priors.rs:43 (penalty prior wrapper)
  - crates/gam-solve/src/custom_family/block_spec.rs:47 (solver canonical)
  - **Action**: Delete src/ version (orphaned), verify term vs solve versions have different purposes

- **CoefficientBlockSelector** (enum)
  - src/families/custom_family/block_spec.rs:25 (ORPHANED)
  - crates/gam-solve/src/custom_family/block_spec.rs:22 (canonical)
  - **Action**: Delete src/ version (orphaned)

- **RealizedCoefficientGroup** (struct)
  - src/families/custom_family/block_spec.rs:81 (ORPHANED)
  - crates/gam-solve/src/custom_family/block_spec.rs:78 (canonical)
  - **Action**: Delete src/ version (orphaned)

- **RealizedCoefficientGroupSpecs** (struct)
  - src/families/custom_family/block_spec.rs:90 (ORPHANED)
  - crates/gam-solve/src/custom_family/block_spec.rs:87 (canonical)
  - **Action**: Delete src/ version (orphaned)

#### Group D: Custom Family Options & Settings
**Status**: `DUPLICATE-COLLAPSE` — Canonical location gam-model-api

- **OuterDerivativePolicy** (struct)
  - src/families/custom_family/options.rs:185 (ORPHANED)
  - crates/gam-model-api/src/families/custom_family/options.rs:204 (canonical)
  - Identical structure, different import dependencies
  - **Action**: Delete src/ version (orphaned)

- **DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES** (pub const)
  - src/families/custom_family/options.rs:628 (ORPHANED)
  - crates/gam-model-api/src/families/custom_family/options.rs:644 (canonical)
  - Both = 1200, identical values
  - **Action**: Delete src/ version (orphaned)

#### Group E: Traits (2-way duplicates)

- **ExactNewtonJointHessianWorkspace** (trait)
  - src/families/custom_family/psi_design.rs:2297 (ORPHANED)
  - crates/gam-model-api/src/families/custom_family/psi_design.rs:27 (canonical)
  - Identical methods
  - **Action**: Delete src/ version (orphaned), ensure re-export from gam-model-api

- **BlockExcessTarget** (trait)
  - crates/gam-problem/src/laplace_sampler_contract.rs:169 (contract definition)
  - crates/gam-inference/src/hmc.rs:6580 (implementation instance)
  - **Status**: `INTENTIONAL` — One is interface, one is impl; both needed

#### Group F: Enums (2-way duplicates)

- **HmcError** (enum)
  - src/inference/hmc_io.rs:82
  - crates/gam-inference/src/hmc.rs:82
  - Identical variants → same root as struct duplication
  - **Action**: Collapse with HMC module

- **NutsFamily** (enum)
  - src/inference/hmc_io.rs:538
  - crates/gam-inference/src/hmc.rs:538
  - Identical variants

- **FamilyNutsInputs<'a>** (enum)
  - src/inference/hmc_io.rs:5261
  - crates/gam-inference/src/hmc.rs:5288
  - Identical variants

- **MaterializationIntent** (enum)
  - src/families/custom_family/psi_design.rs:2285 (ORPHANED)
  - crates/gam-problem/src/psi_design_contract.rs:169 (canonical)
  - Identical variants

- **JointHessianSourcePreference** (enum)
  - src/families/custom_family/psi_design.rs:2259 (ORPHANED)
  - crates/gam-problem/src/psi_design_contract.rs:143 (canonical)
  - Identical variants

- **PenaltyStructureHint** (enum, 2-way within crates)
  - crates/gam-terms/src/smooth_core.rs:6
  - crates/gam-terms/src/smooth/term_specs.rs:1866
  - **Status**: `DUPLICATE-COLLAPSE` — Both in same crate, one should be canonical

---

## 2. DUPLICATE FREE FUNCTION / CONST DEFINITIONS

### Module-Level Functions (50+ duplicates)

**Status**: Most are in src/ (orphaned) vs crates/, or duplicated within crates.

#### Orphaned src/ functions (from hmc_io.rs, probability.rs, sample.rs, etc.)

- **saved_sas_state_from_fit** (pub fn)
  - src/model_types/result_types.rs:974
  - crates/gam-solve/src/model_types/result_types.rs:974
  - Identical implementation → Collapse

- **validate_explicit_dense_hessian_for_whitening** (pub fn)
  - src/model_types/result_types.rs:1396
  - crates/gam-solve/src/model_types/result_types.rs:1396
  - Identical implementation → Collapse

#### HMC/Inference Functions (src/inference/ vs crates/gam-inference/)

The following functions in src/inference/hmc_io.rs (7727 lines) are duplicated in crates/gam-inference/src/hmc.rs (8023 lines). These are all candidates for collapsing into gam-inference and re-exporting:

- `run_joint_beta_rho_sampling<'a>` (2 locations)
- `run_link_wiggle_nuts_sampling<'a>` (2 locations)
- `run_rho_criterion_nuts<F>` (2 locations)
- `run_nuts_sampling_flattened_family<'a>` (2 locations)
- `run_survival_nuts_sampling_flattened<'a>` (2 locations)
- `run_logit_polya_gamma_gibbs` (2 locations)
- `sample_gaussian_mode_posterior` (2 locations)
- Plus 30+ more from hmc_io, probability, sample, skovgaard, etc.

#### Const Duplicates

- **DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES** (pub const = 1200)
  - src/families/custom_family/options.rs:628
  - crates/gam-model-api/src/families/custom_family/options.rs:644

- **DEVICE_ROW_THRESHOLD** (pub const, VALUES DIFFER — NOT IDENTICAL)
  - crates/gam-models/src/gpu_kernels/survival_rowjet.rs:91 (value = 100_000)
  - crates/gam-sae/src/gpu_kernels/sae_rowjet.rs:193 (value = 4_096)
  - **Status**: `COINCIDENTAL COLLISION` — Different values, different domains (survival vs SAE). Keep both.

---

## 3. STALE FULL-COPY MODULES IN ROOT `src/`

### Files to Convert to Re-Export Shims

The root crate should be a thin facade. These modules contain substantial implementations that are duplicated in carved crates:

#### Large Inference Module (CRITICAL)

- **src/inference/hmc_io.rs** (7,727 lines)
  - **Carved Equivalent**: crates/gam-inference/src/hmc.rs (8,023 lines)
  - **Analysis**: This is nearly byte-identical to the crate version; crate version is slightly newer
  - **Action**: `STALE-FULLCOPY-TO-SHIM` — Replace src/inference/mod.rs declaration of `pub mod hmc_io` with `pub use gam_inference::hmc as hmc_io` (or mirror the crate's module path)

#### Inference Submodules (Multiple)

The following files in src/inference/ are large implementations that should be checked for carved crate equivalents:

- **src/inference/probability.rs** (1,683 lines)
  - **Status**: Check if `crates/gam-inference/` or `crates/gam-models/` has a probability module
  - Likely: `STALE-FULLCOPY-TO-SHIM`

- **src/inference/gpu_polya_gamma.rs** (1,836 lines)
  - **Status**: Check if a carved crate has GPU Polya-Gamma kernel
  - Likely: `STALE-FULLCOPY-TO-SHIM`

- **src/inference/sample.rs** (1,333 lines)
  - **Status**: Check if crates/gam-inference or crates/gam-solve has sampling
  - Likely: `STALE-FULLCOPY-TO-SHIM`

- **src/inference/skovgaard.rs** (973 lines)
  - **Status**: Skovgaard correlation matrix for skewed posterior samples
  - Likely: `STALE-FULLCOPY-TO-SHIM`

- **src/inference/rho_posterior.rs** (943 lines)
  - **Status**: ρ (smoothing param) posterior logic
  - Likely: `STALE-FULLCOPY-TO-SHIM` (check crates/gam-solve)

- **src/inference/model_comparison.rs** (627 lines)
  - **Status**: Loo/Waic model comparison
  - Likely: `STALE-FULLCOPY-TO-SHIM`

#### Model Type Result Definitions

- **src/model_types/result_types.rs** (2,305 lines)
  - **Carved Equivalent**: crates/gam-solve/src/model_types/result_types.rs (same line count)
  - **Status**: This is byte-identical with the crate version; src/ is stale
  - **Action**: `STALE-FULLCOPY-TO-SHIM` — src/model_types.rs should be `pub use gam_solve::model_types::*` (currently is)

#### Report Module

- **src/report/mod.rs** (976 lines)
  - **Status**: Fit result reporting (tables, summaries)
  - **Action**: Check if carved into a report crate (unlikely, probably intentional monolith function)
  - **Verdict**: `INTENTIONAL` unless a report crate was carved out

#### Config Resolution

- **src/config_resolve.rs** (1,233 lines)
  - **Status**: JSON → FitConfig resolver shared by CLI and Python FFI
  - **Action**: Check if carved into gam-cli or gam-pyffi
  - **Verdict**: Likely `INTENTIONAL` (shared utility), but verify no duplication in crates/gam-pyffi/

#### Orphaned Families/Custom Family (See Section 4 below)

- **src/families/custom_family/** (all files)
  - **Status**: ORPHANED (not in module tree) but also duplicates carved crate types
  - **Action**: `ORPHAN-DELETE`

---

## 4. ORPHANED DEAD FILES (Not in Module Tree)

### Unreachable Directories & Files

These files/directories are **NOT declared with `mod`/`pub mod`** anywhere in the Rust module tree, making them unreachable unless explicitly re-included or built by build.rs.

#### Complete Orphaned Directory: `src/families/` (7 files)

**Status**: `ORPHAN-DELETE` — Not referenced anywhere; root lib.rs uses `pub use gam_models as families`

The entire `src/families/custom_family/` subtree is orphaned:
- src/families/custom_family/block_spec.rs (253 lines)
- src/families/custom_family/error.rs (3 lines)
- src/families/custom_family/mod.rs (121 lines)
- src/families/custom_family/options.rs (678 lines)
- src/families/custom_family/penalty.rs (10 lines)
- src/families/custom_family/psi_design.rs (2,620 lines) — **Largest orphaned file**
- src/families/custom_family/block_spec.rs (253 lines)

**Total Lines**: ~3,938 lines of dead code
**Action**: Verify with `grep -r "mod families\|pub mod families" src/ crates/` → No matches
**Verdict**: **DELETE ALL** — These files duplicate carved crate types and are unreachable.

#### Orphaned Module: `src/terms/` (2 files, 1 orphaned)

**Status**: `PARTIAL-ORPHAN` — src/terms/mod.rs is orphaned; manifest.rs is used

- **src/terms/mod.rs** (66 lines)
  - **Status**: `ORPHAN-DELETE` — Declares `pub mod basis`, `pub mod construction`, etc., but these don't exist as files and are instead sourced from `crates/gam-terms`. src/lib.rs defines `pub mod terms { pub use gam_terms::*; }` inline, so this file is unreachable.
  - **Action**: DELETE

- **src/terms/analytic_penalties/manifest.rs** (180 lines)
  - **Status**: `INTENTIONAL-KEEP` — This file IS used! build.rs processes it and crates/gam-terms/src/analytic_penalties/manifest.rs includes it via `#[path = "../../../../src/terms/analytic_penalties/manifest.rs"]`
  - Verified: build.rs line says `println!("cargo:rerun-if-changed=src/terms/analytic_penalties/manifest.rs");`
  - **Action**: KEEP (but move to crates/gam-terms or consolidate the includes)

#### Empty Orphaned Directory: `src/test_support/` (0 files)

**Status**: `ORPHAN-DELETE` — Directory exists but is completely empty

**Action**: Remove the empty directory

---

## 5. MACRO DUPLICATION

### Duplicated `#[macro_export]` Macros

These error-handling and utility macros are defined in multiple crates. Many are intentional (each crate's own error boilerplate), but some are candidates for unification.

#### Error Boilerplate Macros (INTENTIONAL per-crate duplication)

Status: `INTENTIONAL` — Each crate that defines error types needs its own boilerplate, but a shared macro library could unify these.

- **impl_reason_error_boilerplate** (4 locations)
  - src/macros.rs:16
  - crates/gam-problem/src/macros.rs:1
  - crates/gam-linalg/src/lib.rs:5
  - crates/gam-models/src/macros.rs:17
  - **Type**: `INTENTIONAL` — Each crate defines error types; each needs its boilerplate. However, a macro utility crate (e.g., gam-macros) could consolidate these as a single definition if the pattern is identical across all four.

- **bail_invalid_estim** (3 locations)
  - src/macros.rs:37
  - crates/gam-problem/src/macros.rs:22
  - crates/gam-terms/src/lib.rs:22
  - **Type**: `INTENTIONAL` — Domain-specific validation error. Unify if definitions are identical.

- **bail_invalid_basis** (3 locations)
  - src/macros.rs:47
  - crates/gam-problem/src/macros.rs:32
  - crates/gam-terms/src/lib.rs:2
  - **Type**: `INTENTIONAL` — Basis-validation error.

- **bail_dim_basis** (3 locations)
  - src/macros.rs:57
  - crates/gam-problem/src/macros.rs:42
  - crates/gam-terms/src/lib.rs:12
  - **Type**: `INTENTIONAL` — Dimension-mismatch error for basis.

- **bail_dim_custom** (3 locations)
  - src/macros.rs:117
  - crates/gam-identifiability/src/lib.rs:12
  - crates/gam-problem/src/macros.rs:52
  - **Type**: `INTENTIONAL` — Custom-family dimension error.

- **bail_invalid_tnorm** (2 locations)
  - src/macros.rs:87
  - crates/gam-models/src/macros.rs:38
  - **Type**: `INTENTIONAL` — Truncated normal / survival time distribution validation.

- **bail_invalid_surv** (2 locations)
  - src/macros.rs:97
  - crates/gam-models/src/macros.rs:48
  - **Type**: `INTENTIONAL` — Survival model validation.

- **bail_dim_sls** (2 locations)
  - src/macros.rs:107
  - crates/gam-models/src/macros.rs:58
  - **Type**: `INTENTIONAL` — Survival location-scale dimension error.

- **gpu_bail** (2 locations)
  - crates/gam-terms/src/lib.rs:32
  - crates/gam-gpu/src/gpu_error.rs:83
  - **Type**: `INTENTIONAL` — GPU error handling; both crates that use GPU kernels need it.

#### Other Macros

- **analytic_penalty_registry** (1 location only)
  - src/terms/analytic_penalties/manifest.rs:154
  - **Status**: `ORPHANED` (src/terms/mod.rs is dead) — This macro is orphaned code within orphaned file.

- **assert_central_difference_array** (1 location)
  - crates/gam-test-support/src/lib.rs:76
  - **Status**: `INTENTIONAL` — Test support macro, only in test crate (correct placement)

- **gam_binary** (1 location)
  - crates/gam-test-support/src/cli_harness.rs:21
  - **Status**: `INTENTIONAL` — CLI harness test macro

---

## SUMMARY TABLE: ALL DUPLICATES BY CLASSIFICATION

| Type | Name | Locations | Classification | Action |
|------|------|-----------|-----------------|--------|
| Struct | FitOptions | src/, gam-solve/ | DUPLICATE-COLLAPSE | Unify in gam-solve, re-export from src/model_types.rs |
| Struct | FitArtifacts | src/, gam-solve/ | DUPLICATE-COLLAPSE | Unify in gam-solve, re-export from src/model_types.rs |
| Struct | FitInference | src/, gam-solve/ | DUPLICATE-COLLAPSE | Unify in gam-solve, re-export from src/model_types.rs |
| Struct | FittedBlock | src/, gam-solve/ | DUPLICATE-COLLAPSE | Unify in gam-solve, re-export from src/model_types.rs |
| Struct | FitGeometry | src/, gam-solve/ | DUPLICATE-COLLAPSE | Unify in gam-solve, re-export from src/model_types.rs |
| Struct | UnifiedFitResultParts | src/, gam-solve/ | DUPLICATE-COLLAPSE | Unify in gam-solve, re-export from src/model_types.rs |
| Struct | UnifiedFitResult | src/, gam-solve/ | DUPLICATE-COLLAPSE | Unify in gam-solve, re-export from src/model_types.rs |
| Struct | AdaptiveRegularizationOptions | src/, gam-solve/ | DUPLICATE-COLLAPSE | Unify in gam-solve, re-export from src/model_types.rs |
| Struct | CriterionCertificate (inference) | src/, gam-solve/, gam-sae/ | DUPLICATE-COLLAPSE (2) + COLLISION (1) | Collapse src/gam-solve, keep gam-sae separate |
| Struct | NutsPosterior | src/, gam-inference/ | DUPLICATE-COLLAPSE | Unify in gam-inference, re-export from src/inference/hmc_io |
| Struct | NutsConfig | src/, gam-inference/ | DUPLICATE-COLLAPSE | Unify in gam-inference, re-export |
| Struct | NutsResult | src/, gam-inference/ | DUPLICATE-COLLAPSE | Unify in gam-inference, re-export |
| Struct | GaussianModePosterior | gam-problem, gam-inference/ | DUPLICATE-COLLAPSE | Unify in gam-problem (contract), reference from gam-inference |
| Struct | GlmFlatInputs | src/, gam-inference/ | DUPLICATE-COLLAPSE | Unify in gam-inference |
| Struct | SurvivalFlatInputs | src/, gam-inference/ | DUPLICATE-COLLAPSE | Unify in gam-inference |
| Struct | SurvivalNutsInputs | src/, gam-inference/ | DUPLICATE-COLLAPSE | Unify in gam-inference |
| Struct | LinkWiggleSplineArtifacts | src/, gam-inference/ | DUPLICATE-COLLAPSE | Unify in gam-inference |
| Struct | LinkWigglePosterior | src/, gam-inference/ | DUPLICATE-COLLAPSE | Unify in gam-inference |
| Struct | JointBetaRhoResult | src/, gam-inference/ | DUPLICATE-COLLAPSE | Unify in gam-inference |
| Struct | JointBetaRhoInputs | src/, gam-inference/ | DUPLICATE-COLLAPSE | Unify in gam-inference |
| Enum | FittedLinkState | src/, gam-solve/ | DUPLICATE-COLLAPSE | Unify in gam-solve |
| Enum | HmcError | src/, gam-inference/ | DUPLICATE-COLLAPSE | Unify in gam-inference |
| Enum | NutsFamily | src/, gam-inference/ | DUPLICATE-COLLAPSE | Unify in gam-inference |
| Enum | FamilyNutsInputs | src/, gam-inference/ | DUPLICATE-COLLAPSE | Unify in gam-inference |
| Enum | MaterializationIntent | src/ (orphaned), gam-problem/ | DUPLICATE-COLLAPSE | Delete src/ version (orphaned), keep gam-problem |
| Enum | JointHessianSourcePreference | src/ (orphaned), gam-problem/ | DUPLICATE-COLLAPSE | Delete src/ version (orphaned), keep gam-problem |
| Enum | PenaltyStructureHint | gam-terms/smooth_core.rs, gam-terms/smooth/term_specs.rs | DUPLICATE-COLLAPSE | One canonical def, one re-export |
| Enum | CoefficientBlockSelector | src/ (orphaned), gam-solve/ | DUPLICATE-COLLAPSE | Delete src/ version (orphaned) |
| Struct | RealizedCoefficientGroup | src/ (orphaned), gam-solve/ | DUPLICATE-COLLAPSE | Delete src/ version (orphaned) |
| Struct | RealizedCoefficientGroupSpecs | src/ (orphaned), gam-solve/ | DUPLICATE-COLLAPSE | Delete src/ version (orphaned) |
| Struct | CoefficientGroupSpec | 3 locations | DUPLICATE-COLLAPSE | Consolidate to one location |
| Struct | OuterDerivativePolicy | src/ (orphaned), gam-model-api/ | DUPLICATE-COLLAPSE | Delete src/ version (orphaned) |
| Trait | ExactNewtonJointHessianWorkspace | src/ (orphaned), gam-model-api/ | DUPLICATE-COLLAPSE | Delete src/ version (orphaned) |
| Trait | BlockExcessTarget | gam-problem, gam-inference/ | INTENTIONAL | Both needed (interface + impl) |
| LaplaceTrustworthiness | gam-problem, gam-inference/ | INTENTIONAL | Both needed (interface + impl) |
| Const | DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES | src/ (orphaned), gam-model-api/ | DUPLICATE-COLLAPSE | Delete src/ version (orphaned) |
| Const | DEVICE_ROW_THRESHOLD | gam-models, gam-sae/ | COINCIDENTAL COLLISION | Keep both (different values, domains) |
| Module | src/inference/hmc_io.rs | src/, crates/gam-inference/ | STALE-FULLCOPY-TO-SHIM | Replace with re-export from gam-inference |
| Module | src/model_types/result_types.rs | src/, crates/gam-solve/ | STALE-FULLCOPY-TO-SHIM | Root already re-exports; src/model_types.rs re-exports |
| Directory | src/families/ | orphaned | ORPHAN-DELETE | Delete entire directory |
| Directory | src/test_support/ | orphaned, empty | ORPHAN-DELETE | Delete directory |
| File | src/terms/mod.rs | orphaned | ORPHAN-DELETE | Delete (root terms is inline) |
| File | src/terms/analytic_penalties/manifest.rs | used by build.rs | INTENTIONAL-KEEP | Keep (referenced by build.rs and crates/gam-terms) |
| Macro | impl_reason_error_boilerplate | 4 locations | INTENTIONAL | Can consolidate to one library macro |
| Macro | bail_invalid_basis, bail_invalid_estim, bail_dim_* | 3 each | INTENTIONAL | Can consolidate to one library |
| Macro | gpu_bail | 2 locations | INTENTIONAL | Both needed (two GPU-using crates) |
| Macro | analytic_penalty_registry | 1 (orphaned) | ORPHAN-DELETE | In orphaned src/terms/analytic_penalties/ file |

---

## RECOMMENDATIONS: PHASED COLLAPSE

### Phase 1: Delete Orphaned Code (Safe, No Refactor)

**Impact**: Remove 3,938+ lines of dead code

1. Delete entire `src/families/` directory (7 files, all orphaned)
2. Delete `src/terms/mod.rs` (66 lines, orphaned)
3. Delete `src/test_support/` directory (0 lines, empty)
4. Verify build still succeeds

**Time**: ~1 hour (git rm, verify build)

### Phase 2: Consolidate Type Definitions (Medium Impact)

**Impact**: Single source of truth for 40+ types

1. Verify src/model_types.rs re-exports are all present
2. For gam-inference duplicates:
   - Ensure gam-inference/src/hmc.rs is authoritative
   - Update src/inference/mod.rs to re-export from gam-inference
   - Delete src/inference/hmc_io.rs
3. For gam-solve duplicates (FitOptions, FitArtifacts, etc.):
   - Already re-exported via src/model_types.rs
   - Verify no other src/ files reference these directly

**Time**: ~4 hours (verify paths, add re-exports, test)

### Phase 3: Macro Library (Low Urgency)

**Impact**: Reduce boilerplate maintenance surface

1. Create `crates/gam-macros` crate with:
   - `impl_reason_error_boilerplate`
   - `bail_invalid_*`, `bail_dim_*` family
   - `gpu_bail`
2. Replace all 4 + 3 + 3 + 3 + 2 locations with `#[macro_use] extern crate gam_macros`
3. Or (simpler): Keep as-is; only consolidate if maintenance burden emerges

**Time**: ~3 hours (create crate, migrate, test) or skip

---

## Notes & Caveats

1. **src/terms/analytic_penalties/manifest.rs**: This file is genuinely used by build.rs (code generation for penalty registry). It should NOT be deleted, but its location (stranded in orphaned src/terms/) is awkward. Consider moving to crates/gam-terms/ or consolidating the build.rs logic.

2. **src/config_resolve.rs**: Despite being 1,233 lines, it's intentional (shared by CLI and Python FFI). No carved equivalent found. Keep.

3. **src/report/**: Fit result reporting tables and summaries. No carved crate found. Likely intentional monolith function. Verify no duplication in other crates before declaring "keep."

4. **Macro duplication is mostly intentional**: Each crate that has custom error types needs its own error-boilerplate macro. These can't be unified without risking visibility/path issues. Only consolidate if a clear macro library pattern emerges.

5. **gam-sae's CriterionCertificate is a different type**: It's for SAE/manifold criterion convergence audits, not fit optimization audits. Despite the name collision, they serve different purposes. Keep separate.

6. **LaplaceTrustworthiness and GaussianModePosterior**: Both appear in gam-problem (as traits/contracts) and gam-inference (as implementations). This is correct by design — contracts live in lower-layer crates, impls live higher. Don't collapse.

---

## Conclusion

**Total actionable duplicates**: 50+ types + 30+ functions + 7 macros (intentional) = **87 items**

**Safe deletions (Phase 1)**: 3,938 lines of orphaned dead code

**Type consolidations (Phase 2)**: 40 types, unifying to single source in carved crates, with re-exports from root src/

**Macro consolidations (Phase 3)**: Optional; only if a macro library emerges as beneficial

