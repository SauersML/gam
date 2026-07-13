// GAM fit-orchestration drivers, relocated from `gam-terms/src/smooth/`
// (`design_construction.rs` + `spatial_optimization.rs`) up into `gam-models`
// per #1521. They were `include!`d into `gam_terms::smooth` (one flat module
// alongside `prelude.rs` + `term_specs.rs`); to preserve that single-module
// flat namespace (and the heavy cross-references between the two files) byte
// for byte, they are `include!`d here as well. The shared import surface that
// `prelude.rs`/`term_specs.rs` used to provide is reconstructed below with the
// relocated paths (families now resolve as `crate::*`, the solver as
// `gam_solve::*`, basis/term machinery as `gam_terms::*`).
use gam_terms::basis::{
    BasisError, BasisMetadata, BasisPsiDerivativeResult, BasisPsiSecondDerivativeResult,
    BasisWorkspace, CenterStrategy, MaternIdentifiability, PenaltySource,
    build_constant_curvature_basis_kappa_derivatives,
    build_matern_basis_log_kappa_aniso_derivatives, build_matern_basis_log_kappa_derivatives,
    build_matern_collocation_operator_matrices, build_measure_jet_basis_psi_derivatives,
    build_thin_plate_basis_log_kappa_derivatives, estimate_penalty_nullity,
    initial_aniso_contrasts,
};

use gam_custom_family::{
    BlockEffectiveJacobian, BlockGeometryDirectionalDerivative, BlockWorkingSet,
    BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative, CustomFamilyOwnedMode,
    CustomFamilyWarmStart, ExactNewtonOuterObjective, FamilyEvaluation, FamilyLinearizationState,
    ParameterBlockSpec, ParameterBlockState, PenaltyMatrix,
    evaluate_custom_family_joint_hyper_efs_owned, evaluate_custom_family_joint_hyper_owned,
    fit_custom_family, fit_custom_family_fixed_log_lambdas_from_owned_mode,
};

use gam_solve::estimate::{
    EstimationError, ExternalOptimOptions, FitInference, FitOptions, FittedLinkState, PenaltySpec,
    UnifiedFitResult, UnifiedFitResultParts, fit_gamwith_heuristic_lambdas,
};

use gam_solve::estimate::reml::DirectionalHyperParam;

// #1521: `freeze_term_collection_from_design` relocated DOWN into gam_terms::smooth
// (was an `include!`d `pub fn` in spatial_optimization.rs). Re-export here so the
// `crate::fit_orchestration::drivers::freeze_term_collection_from_design` path used
// by families + pyffi resolves unchanged.
pub use gam_terms::smooth::freeze_term_collection_from_design;

use gam_solve::mixture_link::{
    inverse_link_jet_for_inverse_link, logit_inverse_link_jet5, state_from_beta_logisticspec,
    state_from_sasspec, state_fromspec,
};

use gam_math::quantile::quantile_from_sorted;

use gam_linalg::faer_ndarray::{fast_ab, fast_atb, fast_atv};

use gam_linalg::matrix::{DesignBlock, DesignMatrix, RandomEffectOperator, SymmetricMatrix};

use gam_problem::{ExactNewtonJointPsiTerms, LinearInequalityConstraints};

use gam_spec::{
    InverseLink, LatentCLogLogState, LikelihoodSpec, MixtureLinkState, ResponseFamily,
    SasLinkState, StandardLink,
};

use gam_terms::smooth::input_standardization::{
    apply_input_standardization, compensate_length_scale_for_standardization,
    compensate_optional_length_scale_for_standardization,
};

use gam_terms::smooth::penalty_priors::{
    realize_keyed_penalty_block_gamma_priors, realize_penalty_block_gamma_priors,
};

use gam_terms::smooth::shape_constraints::{
    linear_constraints_from_lower_bounds_global, merge_linear_constraints_global,
    shape_lower_bounds_local,
};

// Every `pub` item that `gam_terms::smooth` exposes (the `term_specs.rs`
// spec/design machinery, `SmoothError`, the `penalty_priors`/`structure_analysis`
// re-exports, …). This reconstructs the sibling-module visibility the drivers
// had while textually pasted inside `gam_terms::smooth`.
use gam_terms::smooth::*;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::atomic::AtomicUsize;
use std::sync::{Arc, Mutex};

// Fit-result carriers relocated out of `gam_terms::smooth::term_specs` with the
// drivers (they hold a `gam_solve` `UnifiedFitResult` and are consumed only by
// the drivers / the surrounding fit-orchestration layer).
#[derive(Clone)]
pub struct FittedTermCollection {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub adaptive_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SpatialLengthScaleOptimizationTiming {
    pub log_kappa_dim: usize,
    pub cost_calls: usize,
    pub cost_total_s: f64,
    pub eval_calls: usize,
    pub eval_total_s: f64,
    pub efs_calls: usize,
    pub efs_total_s: f64,
    pub slow_path_resets: u64,
    pub design_revision_delta: u64,
    /// #1868 deterministic n-independence instrument: the number of length-`n`
    /// row-element touches the Gaussian zero-iteration inner synthesis performed
    /// on the #1033 n-free κ-trial *skip* path during this κ-optimisation phase
    /// (excludes the one-time priming eval). The #1033 architectural invariant
    /// requires each in-window trial to touch only k×k objects, so this MUST NOT
    /// scale with `n`. A value that grows with `n` is exactly the #1868
    /// O(n)-per-callback regression. This replaces the old noisy wall-clock
    /// per-callback ratio with an exact, millisecond-fast integer gate.
    pub nfree_skip_row_touches: u64,
    pub nfree_miss_shape: u64,
    pub nfree_miss_value: u64,
    pub nfree_miss_gradient: u64,
    pub nfree_miss_penalty: u64,
    pub nfree_miss_revision: u64,
    pub nfree_miss_second_order: u64,
    pub nfree_miss_other: u64,
    pub optim_total_s: f64,
}

impl SpatialLengthScaleOptimizationTiming {
    pub fn trial_total_s(self) -> f64 {
        self.cost_total_s + self.eval_total_s + self.efs_total_s
    }
}

#[derive(Clone)]
pub struct FittedTermCollectionWithSpec {
    pub fit: UnifiedFitResult,
    pub design: TermCollectionDesign,
    pub resolvedspec: TermCollectionSpec,
    pub adaptive_diagnostics: Option<AdaptiveRegularizationDiagnostics>,
    pub kappa_timing: Option<SpatialLengthScaleOptimizationTiming>,
}

include!("design_construction.rs");
include!("spatial_optimization.rs");

#[cfg(test)]
mod test_support {
    use super::*;

    /// Test-only default-policy constructor. Production callers must supply the
    /// fit's intrinsic resource policy through `new_with_policy`; keeping this
    /// adapter inside the test-support module prevents a permissive constructor
    /// from entering the library surface.
    pub(super) trait SingleBlockExactJointDesignCacheTestExt<'d>: Sized {
        fn new(
            data: ArrayView2<'d, f64>,
            spec: TermCollectionSpec,
            design: TermCollectionDesign,
            spatial_terms: Vec<usize>,
            rho_dim: usize,
            dims_per_term: Vec<usize>,
        ) -> Result<Self, String>;
    }

    impl<'d> SingleBlockExactJointDesignCacheTestExt<'d> for SingleBlockExactJointDesignCache<'d> {
        fn new(
            data: ArrayView2<'d, f64>,
            spec: TermCollectionSpec,
            design: TermCollectionDesign,
            spatial_terms: Vec<usize>,
            rho_dim: usize,
            dims_per_term: Vec<usize>,
        ) -> Result<Self, String> {
            let policy = gam_runtime::resource::ResourcePolicy::default_library();
            Self::new_with_policy(
                data,
                spec,
                design,
                spatial_terms,
                rho_dim,
                dims_per_term,
                &policy,
            )
        }
    }
}

// #901 re-home: the end-to-end iso-κ joint REML outer-gradient FD oracles on
// real Duchon/Matérn smooths. Authored in the pre-#1521 monolith, orphaned out
// of the build by #1601 (its private driver deps live HERE post-carve, not in
// `gam_terms::smooth` where the `include!` was commented out). The file is a
// self-contained `#[cfg(test)] mod`, so it adds nothing to the non-test build.
include!("iso_kappa_reml_gradient_fd_tests.rs");
// #901 re-home: the Matérn κ-optimizer convergence/monotone gates the issue
// listed as stalling on the wrong projected-logdet gradient. Same #1601
// orphaning story — driver deps live HERE post-carve. Self-contained
// `#[cfg(test)] mod`, so it adds nothing to the non-test build.
include!("spatial_length_scale_monotone_tests.rs");
// #1264/#1033 re-home: the production ψ-Gram fast-path skip guard
// (`reduced_basis_equal` soundness, β̂ vs streamed to 1e-6) and the #1033
// forced-rotation frontier measurement. Same #1601 orphaning story as the two
// siblings above — its private driver deps live HERE post-carve, and the
// monolith `include!` in `gam_terms::smooth::tests` was commented out and never
// relocated, so both guards compiled into NO binary. Self-contained
// `#[cfg(test)] mod`, so it adds nothing to the non-test build.
include!("psi_gram_tensor_fast_path_tests.rs");
// #901 re-home: the custom-family ADAPTIVE-ψ projected-logdet REML
// hypergradient + outer-Hessian FD oracle on a real `SpatialAdaptiveExactFamily`
// — the half of #901 the engine fix (joint_jeffreys_information_depends_on_psi)
// directly targets, plus the #426 unified-dispatch parity pin. Same #1601
// orphaning story as the two oracles above; driver deps live HERE post-carve.
// Self-contained `#[cfg(test)] mod`, so it adds nothing to the non-test build.
include!("spatial_adaptive_hyper_fd_tests.rs");
// #1274 re-home: the Matérn n-free penalty re-key topology/byte-identity gates.
// Authored in the pre-#1521 monolith under `tests/src_modules/smooths/`, they
// were orphaned by #1601 (the `gam_terms::smooth::tests` `include!` was
// commented out and the body needs the gam-models-private
// `FrozenTermCollectionIncrementalRealizer`), so the #1274 guard compiled
// nowhere. Re-homed HERE where the private realizer lives; self-contained
// `#[cfg(test)] mod`, so it adds nothing to the non-test build.
include!("matern_nfree_rekey_topology_tests.rs");
// #1601 relocation debt: the 88 design-assembly / constraint / IFT-cache
// regression guards. Same orphaning story as the siblings above — their
// `build_term_collection_design` / freeze / incremental-realizer / tensor+streamed
// eval deps live HERE post-#1521 carve, but #1601 commented the include! out of
// `gam_terms::smooth::tests` "for relocation" that never happened (the parked
// `tests/src_modules/` tree was `mod`'d into no binary). Self-contained
// `#[cfg(test)] mod`.
include!("design_assembly_constraint_tests.rs");
// #1601 relocation debt: the LAST of the three orphaned smooth test files — 48
// adaptive / bounded / pure-Duchon / Charbonnier regression guards. Same story:
// commented out of `gam_terms::smooth::tests` by #1601 "for relocation" and
// parked in the `tests/src_modules/` tree that compiled into no binary. Re-homed
// here where its `build_term_collection_design` / freeze / SAS-link-state /
// joint-hyper FD deps resolve post-#1521 carve. Self-contained `#[cfg(test)] mod`.
include!("adaptive_bounded_duchon_tests.rs");
