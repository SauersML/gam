//! Canonical `InnerSolution` assembler.
//!
//! No production code outside this module may construct
//! `InnerSolutionBuilder::new(...)` or call `reml_laml_evaluate(...)`.
//! Tests are exempt.
//!
//! All families and runtime paths provide ingredients and call
//! [`InnerAssembly::evaluate`] or [`InnerAssembly::build`].

use super::FirthDenseOperator;
use super::unified::{
    BarrierConfig, DispersionHandling, EvalMode, FixedDriftDerivFn, HessianDerivativeProvider,
    HessianOperator, HyperCoord, HyperCoordPair, InnerSolution, InnerSolutionBuilder,
    PenaltyCoordinate, PenaltyLogdetDerivs, RemlLamlResult, penalty_matrix_root,
    reml_laml_evaluate,
};
use ndarray::{Array1, Array2};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
//  InnerAssembly — the single entry point for InnerSolution construction
// ═══════════════════════════════════════════════════════════════════════════

/// All ingredients needed to assemble an `InnerSolution`.
///
/// Callers fill in the required fields and override optional ones as needed.
/// The assembler builds the `InnerSolution` via `InnerSolutionBuilder` and
/// calls `reml_laml_evaluate` — the only production code path that does so.
pub struct InnerAssembly<'dp> {
    // === Required core ===
    pub log_likelihood: f64,
    pub penalty_quadratic: f64,
    pub beta: Array1<f64>,
    pub n_observations: usize,
    pub hessian_op: std::sync::Arc<dyn HessianOperator>,
    pub penalty_coords: Vec<PenaltyCoordinate>,
    pub penalty_logdet: PenaltyLogdetDerivs,
    pub dispersion: DispersionHandling,
    pub rho_curvature_scale: f64,
    pub hessian_logdet_correction: f64,

    // === Optional decorations (sensible defaults when None/zero) ===
    pub deriv_provider: Option<Box<dyn HessianDerivativeProvider + 'dp>>,
    pub tk_correction: f64,
    pub tk_gradient: Option<Array1<f64>>,
    pub firth: Option<Arc<FirthDenseOperator>>,
    pub nullspace_dim: Option<f64>,
    pub barrier_config: Option<BarrierConfig>,

    // === Extended hyperparameter coordinates ===
    pub ext_coords: Vec<HyperCoord>,
    pub ext_coord_pair_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,
    pub rho_ext_pair_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPair + Send + Sync>>,
    pub fixed_drift_deriv: Option<FixedDriftDerivFn>,
}

impl<'dp> InnerAssembly<'dp> {
    /// Build the `InnerSolution` from these ingredients.
    pub fn build(self) -> InnerSolution<'dp> {
        let mut builder = InnerSolutionBuilder::new(
            self.log_likelihood,
            self.penalty_quadratic,
            self.beta,
            self.n_observations,
            self.hessian_op,
            self.penalty_coords,
            self.penalty_logdet,
            self.dispersion,
        );
        builder = builder.rho_curvature_scale(self.rho_curvature_scale);
        builder = builder.hessian_logdet_correction(self.hessian_logdet_correction);

        if let Some(dp) = self.deriv_provider {
            builder = builder.deriv_provider(dp);
        }
        builder = builder.tk(self.tk_correction, self.tk_gradient);
        builder = builder.firth(self.firth);
        if let Some(nd) = self.nullspace_dim {
            builder = builder.nullspace_dim_override(nd);
        }
        builder = builder.barrier_config(self.barrier_config);

        if !self.ext_coords.is_empty() {
            builder = builder.ext_coords(self.ext_coords);
        }
        if let Some(f) = self.ext_coord_pair_fn {
            builder = builder.ext_coord_pair_fn(f);
        }
        if let Some(f) = self.rho_ext_pair_fn {
            builder = builder.rho_ext_pair_fn(f);
        }
        if let Some(f) = self.fixed_drift_deriv {
            builder = builder.fixed_drift_deriv(f);
        }

        builder.build()
    }

    /// Build and evaluate in one step.
    pub fn evaluate(
        self,
        rho: &[f64],
        mode: EvalMode,
        prior: Option<(f64, Array1<f64>, Option<Array2<f64>>)>,
    ) -> Result<RemlLamlResult, String> {
        let solution = self.build();
        reml_laml_evaluate(&solution, rho, mode, prior)
    }
}

/// Evaluate a pre-built `InnerSolution` through the unified evaluator.
///
/// Use this when the caller needs the `InnerSolution` to outlive the evaluation
/// (e.g., for EFS step computation after evaluation). Prefer
/// [`InnerAssembly::evaluate`] when the solution is not needed afterwards.
pub fn evaluate_solution(
    solution: &InnerSolution<'_>,
    rho: &[f64],
    mode: EvalMode,
    prior: Option<(f64, Array1<f64>, Option<Array2<f64>>)>,
) -> Result<RemlLamlResult, String> {
    reml_laml_evaluate(solution, rho, mode, prior)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Penalty coordinate helpers for family modules
// ═══════════════════════════════════════════════════════════════════════════

/// Descriptor for a single penalty block within the parameter vector.
pub struct PenaltyBlockDesc<'a> {
    pub matrix: &'a Array2<f64>,
    pub range_start: usize,
    pub range_end: usize,
}

/// Build `PenaltyCoordinate`s from block descriptors.
///
/// Replaces the manual `penalty_matrix_root` + `from_block_root` loops
/// in `survival.rs` and `custom_family.rs`.
pub fn penalty_coords_from_blocks(
    blocks: &[PenaltyBlockDesc],
    total_dim: usize,
) -> Result<Vec<PenaltyCoordinate>, String> {
    blocks
        .iter()
        .map(|b| {
            let root = penalty_matrix_root(b.matrix)?;
            Ok(PenaltyCoordinate::from_block_root(
                root,
                b.range_start,
                b.range_end,
                total_dim,
            ))
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Soft prior helper
// ═══════════════════════════════════════════════════════════════════════════

/// Compute the soft rho prior tuple for a given evaluation mode.
///
/// Extracts the repeated prior-assembly pattern that was copy-pasted 4×
/// in `runtime.rs`. The caller provides cost/gradient/hessian via closures
/// (typically `self.compute_soft_priorcost/grad/hess`).
pub fn soft_prior_for_mode<F, G, H>(
    rho: &Array1<f64>,
    mode: EvalMode,
    cost_fn: F,
    grad_fn: G,
    hess_fn: H,
) -> Option<(f64, Array1<f64>, Option<Array2<f64>>)>
where
    F: Fn(&Array1<f64>) -> f64,
    G: Fn(&Array1<f64>) -> Array1<f64>,
    H: Fn(&Array1<f64>) -> Option<Array2<f64>>,
{
    if mode == EvalMode::ValueOnly {
        let pc = cost_fn(rho);
        if pc.abs() > 0.0 {
            Some((pc, Array1::zeros(rho.len()), None))
        } else {
            None
        }
    } else {
        let pc = cost_fn(rho);
        let pg = grad_fn(rho);
        let ph = if mode == EvalMode::ValueGradientHessian {
            hess_fn(rho)
        } else {
            None
        };
        if pc.abs() > 0.0 || pg.iter().any(|&v| v != 0.0) {
            Some((pc, pg, ph))
        } else {
            None
        }
    }
}
