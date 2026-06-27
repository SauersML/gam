# GAM Workspace DAG Violation Audit - MASTER RESIDUALS v2

**Audit Date**: June 26, 2026  
**Worktree**: /Users/user/gam-wt-carve  
**Scope**: Complete inventory of remaining DAG violations across 19 crates

## DAG Reference (Low → High)
```
Level 0: gam-spec, gam-problem
Level 1: gam-math, gam-runtime, gam-linalg, gam-geometry
Level 2: gam-model-api, gam-gpu, gam-identifiability
Level 3: gam-terms
Level 4: gam-solve
Level 5: gam-models
Level 6: gam-predict
Level 7: gam-inference
Level 8: gam-sae
Level 9: gam-pyffi, gam-cli
Special: gam-data (unordered), gam-test-support (test-only)
```

---

## VIOLATIONS FOUND

### 1. gam-models (Tier 5)
**VIOLATION TYPE**: needs-dep-add

- **File**: `crates/gam-models/src/fit_orchestration/drivers/spatial_optimization.rs`
- **Lines**: 8542, 8546, 8561, 8668, 8672
- **Issue**: Uses `gam_geometry::curvature_estimand::{KappaProfileCi, FlatnessTest, ...}` in public struct `CurvatureInference` and public function `compute_curvature_inference()` without declaring gam-geometry as a dependency
- **Declared Dependencies**: gam-data, gam-gpu, gam-identifiability, gam-linalg, gam-math, gam-model-api, gam-problem, gam-runtime, gam-solve, gam-spec, gam-terms, gam-test-support
- **Missing**: **gam-geometry**
- **DAG Violation**: gam-models (5) → gam-geometry (1) — violates ordering
- **Classification**: needs-dep-add

---

## UNDECLARED BUT RESOLVED

### 2. gam-models uses gam_model_api (Tier 5)
**Status**: ✓ DECLARED (via Cargo.toml)
- **Files**: `crates/gam-models/src/fit_orchestration/drivers/spatial_optimization.rs:7112`, `custom_family.rs:7,20`, `joint_penalty.rs:6`
- **Usage**: `gam_model_api::families::custom_family::OuterDerivativePolicy`
- **Classification**: facade-reexport-OK

### 3. gam-solve uses gam_model_api (Tier 4)
**Status**: ✓ DECLARED (via Cargo.toml)
- **Files**: `crates/gam-solve/src/custom_family.rs:78`, `custom_family_persistent_warm_start.rs:16`
- **Usage**: Re-exports and direct usage of CustomFamily
- **Classification**: facade-reexport-OK

### 4. gam-identifiability uses gam_test_support (Tier 2)
**Status**: ✓ DECLARED (via Cargo.toml)
- **Files**: `crates/gam-identifiability/src/canonical.rs:2452`
- **Usage**: `gam_test_support::spec_from_dense_with_priority` (test-only)
- **Classification**: facade-reexport-OK

### 5. gam-gpu uses gam_problem in test code (Tier 2)
**Status**: ⚠️ TEST-ONLY (line 466 in `#[test]` block)
- **File**: `crates/gam-gpu/src/mod.rs:466`
- **Usage**: `use gam_problem::ExecutionPath;` in `#[test] fn execution_path_defaults_to_cpu()`
- **Issue**: gam-problem is NOT in [dev-dependencies], only in regular dependencies of transitives
- **Classification**: needs-dev-dep-add (optional: add gam-problem to [dev-dependencies] for test clarity)

---

## CLEAN CRATES (No DAG Violations)

The following crates are **CLEAN** — all external references are properly declared:

- **gam-cli** (Tier 9): ✓ Deps match usage
- **gam-data**: ✓ No gam_*:: references
- **gam-geometry** (Tier 1): ✓ Deps match usage
- **gam-inference** (Tier 7): ✓ Deps match usage (only uses gam-predict)
- **gam-linalg** (Tier 1): ✓ Deps match usage
- **gam-math** (Tier 1): ✓ No gam_*:: references
- **gam-model-api** (Tier 2): ✓ Deps match usage
- **gam-predict** (Tier 6): ✓ Deps match usage
- **gam-problem** (Tier 0): ✓ Deps match usage
- **gam-pyffi** (Tier 9): ✓ Deps match usage
- **gam-runtime** (Tier 1): ✓ No gam_*:: references
- **gam-sae** (Tier 8): ✓ Deps match usage
- **gam-solve** (Tier 4): ✓ All declared (large dep set as middle-tier aggregator)
- **gam-spec** (Tier 0): ✓ No gam_*:: references
- **gam-terms** (Tier 3): ✓ Deps match usage
- **gam-test-support**: ✓ Deps match usage (includes gam-models for reference tests)

---

## INTEGRATION CHECKLIST

| Crate | Violation | Fix Strategy | Effort |
|-------|-----------|--------------|--------|
| gam-models | Undeclared gam-geometry | Add `gam-geometry` to Cargo.toml | 1 line |
| gam-gpu | gam_problem in test (optional) | Add `gam-problem` to [dev-dependencies] | 1 line |

---

## KNOWN RELOCATED SYMBOLS SCAN

No references found to legacy paths:
- ✓ `gam_terms::smooth::{build_term_collection_design|freeze_term_collection_from_design|fit_term_collection*|...}` (now in `gam_models::fit_orchestration::drivers`)
- ✓ `crate::inference::*` outside gam-models (properly encapsulated)
- ✓ `crate::test_support` external refs (properly routed to `gam_test_support`)

---

## NOTES

1. **gam-test-support tier position**: Depends on gam-models (tier 5), making gam-test-support effectively tier 6+. This is acceptable since it's test-only infrastructure, not a production tier.

2. **gam-models re-exports**: Multiple facade re-exports to gam-model-api and gam-problem symbols are properly declared and OK.

3. **Transitive dependencies**: gam-geometry is available to gam-models through gam-problem but must be **explicitly declared** for direct usage in public types (type system requirement, not just best practice).

4. **Test code in main modules**: gam-gpu's test code can compile if a transitive dep provides gam-problem, but declaring it explicitly in [dev-dependencies] improves clarity.

---

## SUMMARY

**Critical Violations** (blocking): 1
- gam-models missing gam-geometry dependency

**Optional/Low-Priority** (test-only): 1
- gam-gpu test code clarity (dev-dependency)

**All Other Tiers**: ✓ CLEAN

**Integration Impact**: Minimal — one dependency addition to gam-models/Cargo.toml resolves the critical violation.
