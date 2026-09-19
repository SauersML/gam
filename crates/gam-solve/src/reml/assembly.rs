//! Canonical `InnerSolution` assembler.
//!
//! No production code outside this module may construct
//! `InnerSolutionBuilder::new(...)` or call `reml_laml_evaluate(...)`.
//! Tests are exempt.
//!
//! All families and runtime paths provide ingredients and call
//! [`InnerAssembly::evaluate`] or [`InnerAssembly::build`].

use super::reml_outer_engine::{
    BarrierConfig, ContractedPsiSecondOrderFn, DispersionHandling, EvalMode, FixedDriftDerivFn,
    HessianDerivativeProvider, HessianFactorization, HyperCoord, HyperCoordPairResult,
    InnerSolution, InnerSolutionBuilder, PenaltyCoordinate, PenaltyLogdetDerivs,
    PenaltySubspaceTrace, RemlLamlResult, reml_laml_evaluate,
};
use crate::model_types::ProjectedKktResidual;
use gam_linalg::faer_ndarray::{fast_xt_diag_x, fast_xt_diag_y};
use ndarray::{Array1, Array2};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
//  Streaming weighted dense-design products
// ═══════════════════════════════════════════════════════════════════════════

pub(crate) const DENSE_ROW_SCALE_PAR_CELLS: usize = 64 * 1024;

#[derive(Clone, Copy)]
pub(crate) enum DenseRowScaleMode {
    Direct,
    InversePositiveOrZero,
}

/// Write `diag(scale) · x` into `out`, preserving `out`'s allocation when its
/// shape already matches `x`.
///
/// This replaces the former clone-and-row-scale pattern used by REML assembly
/// tests and Firth kernels. It is intentionally simple and deterministic for a
/// fixed row order.
pub(crate) fn row_scale_dense_into(x: &Array2<f64>, scale: &Array1<f64>, out: &mut Array2<f64>) {
    assert_eq!(x.nrows(), scale.len(), "scale length must match row count");
    if out.raw_dim() != x.raw_dim() {
        *out = Array2::<f64>::zeros(x.raw_dim());
    }
    out.assign(x);
    row_scale_dense_in_place(out, scale, DenseRowScaleMode::Direct);
}

/// Scale each row of `out` by `1 / scale[row]`, writing zero rows where
/// `scale[row] <= 0`.
pub(crate) fn row_scale_dense_in_place_by_inverse_positive_or_zero(
    out: &mut Array2<f64>,
    scale: &Array1<f64>,
) {
    row_scale_dense_in_place(out, scale, DenseRowScaleMode::InversePositiveOrZero);
}

pub(crate) fn row_scale_dense_in_place(
    out: &mut Array2<f64>,
    scale: &Array1<f64>,
    mode: DenseRowScaleMode,
) {
    assert_eq!(
        out.nrows(),
        scale.len(),
        "scale length must match row count"
    );
    let ncols = out.ncols();
    if ncols == 0 {
        return;
    }

    let cells = out.nrows().saturating_mul(ncols);
    if cells >= DENSE_ROW_SCALE_PAR_CELLS
        && rayon::current_num_threads() > 1
        && out.is_standard_layout()
        && let Some(slice) = out.as_slice_memory_order_mut()
    {
        slice
            .par_chunks_mut(ncols)
            .zip(
                scale
                    .as_slice()
                    .expect("Array1 must be contiguous")
                    .par_iter(),
            )
            .for_each(|(row_values, &w)| scale_dense_row_values(row_values, w, mode));
        return;
    }

    ndarray::Zip::from(out.rows_mut())
        .and(scale.view())
        .for_each(|mut row, &w| {
            if let Some(row_values) = row.as_slice_mut() {
                scale_dense_row_values(row_values, w, mode);
            } else {
                match mode {
                    DenseRowScaleMode::Direct => row *= w,
                    DenseRowScaleMode::InversePositiveOrZero => {
                        if w > 0.0 {
                            row *= w.recip();
                        } else {
                            row.fill(0.0);
                        }
                    }
                }
            }
        });
}

#[inline]
pub(crate) fn scale_dense_row_values(row_values: &mut [f64], scale: f64, mode: DenseRowScaleMode) {
    match mode {
        DenseRowScaleMode::Direct => {
            for value in row_values {
                *value *= scale;
            }
        }
        DenseRowScaleMode::InversePositiveOrZero => {
            if scale > 0.0 {
                let inv = scale.recip();
                for value in row_values {
                    *value *= inv;
                }
            } else {
                for value in row_values {
                    *value = 0.0;
                }
            }
        }
    }
}

/// Compute `leftᵀ diag(weights) right` through the shared streamed kernel
/// ([`fast_xt_diag_y`]), which never materializes a weighted n×q copy.
///
/// There is one path at every size and every pool width. The kernel picks its
/// row split from the shape alone, so the product's bits do not follow
/// `RAYON_NUM_THREADS`; a separate "parallel" branch here, taken only when the
/// pool had more than one worker, summed the rows in a different order than the
/// one-worker GEMM and made every large fit's bits a function of the pool width.
pub(crate) fn weighted_cross_dense(
    left: &Array2<f64>,
    right: &Array2<f64>,
    weights: &Array1<f64>,
) -> Array2<f64> {
    assert_eq!(left.nrows(), right.nrows());
    assert_eq!(left.nrows(), weights.len());
    fast_xt_diag_y(left, weights, right)
}

/// Compute `xᵀ diag(diag) x` through the shared streamed symmetric kernel
/// ([`fast_xt_diag_x`]), which accumulates the lower triangle and mirrors it, so
/// the Gram is exactly symmetric: a full GEMM of `xᵀ·(diag·x)` rounds `[a, b]`
/// and `[b, a]` along different accumulation orders. Signed `diag` is kept
/// exactly (`Xᵀ·(D·X)`, never a square root). As with [`weighted_cross_dense`],
/// one path serves every size and pool width, so the bits never depend on
/// `RAYON_NUM_THREADS`.
pub(crate) fn xt_diag_x_dense(x: &Array2<f64>, diag: &Array1<f64>) -> Array2<f64> {
    assert_eq!(diag.len(), x.nrows(), "diag length must match row count");
    fast_xt_diag_x(x, diag)
}

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
    pub hessian_op: std::sync::Arc<dyn HessianFactorization>,
    /// Distinct operator for the IFT mode response `v_k = ∂β̂/∂θ_k`; see
    /// [`InnerSolution::mode_response_op`]. `None` — the default every caller
    /// wants unless the criterion's `log|·|` is deliberately taken on a
    /// curvature surrogate — keeps the mode response on `hessian_op`.
    pub mode_response_op: Option<std::sync::Arc<dyn HessianFactorization>>,
    pub penalty_coords: Vec<PenaltyCoordinate>,
    pub penalty_logdet: PenaltyLogdetDerivs,
    pub dispersion: DispersionHandling,
    pub rho_curvature_scale: f64,
    pub rho_prior: gam_problem::RhoPrior,
    pub hessian_logdet_correction: f64,
    pub penalty_subspace_trace: Option<Arc<PenaltySubspaceTrace>>,

    // === Optional decorations (sensible defaults when None/zero) ===
    pub deriv_provider: Option<Box<dyn HessianDerivativeProvider + 'dp>>,
    /// Jeffreys/Firth scalar contribution to the LAML cost. Tier-A GLM callers
    /// construct it from the dense operator (`ExactJeffreysTerm::new`); the
    /// Tier-B coupled joint path installs the value-only carrier
    /// (`ExactJeffreysTerm::value_only`) so the cost subtracts the same gated
    /// `Φ(β̂)` its inner Newton optimized (gam#979).
    pub firth: Option<crate::estimate::reml::reml_outer_engine::ExactJeffreysTerm>,
    pub nullspace_dim: Option<f64>,
    pub barrier_config: Option<BarrierConfig>,
    pub kkt_residual: Option<ProjectedKktResidual>,
    /// Active linear-inequality constraint rows at the converged inner
    /// iterate. When `Some`, the unified evaluator builds the
    /// constraint-aware kernel `K_T = K_S − K_S Aᵀ (A K_S Aᵀ)⁻¹ A K_S`
    /// for per-coordinate mode responses `v_k = ∂β/∂ρ_k`.
    pub active_constraints: Option<Arc<crate::model_types::ActiveLinearConstraintBlock>>,
    /// The constraint system and KKT gradient the constrained Laplace normalizer reads
    /// (gam#2765); `None` prices no truncation.
    pub cone_normalizer: Option<Arc<crate::estimate::reml::reml_outer_engine::ConeNormalizerInput>>,

    // === Extended hyperparameter coordinates ===
    pub ext_coords: Vec<HyperCoord>,
    pub ext_coord_pair_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPairResult + Send + Sync>>,
    pub rho_ext_pair_fn: Option<Box<dyn Fn(usize, usize) -> HyperCoordPairResult + Send + Sync>>,
    pub fixed_drift_deriv: Option<FixedDriftDerivFn>,
    /// Direction-contracted ψψ second-order hook (#740). When set, the
    /// outer-Hessian operator builder skips the `K²` per-pair ψψ assembly and
    /// applies this once per matvec.
    pub contracted_psi_second_order: Option<ContractedPsiSecondOrderFn>,
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
        builder = builder.mode_response_op(self.mode_response_op);
        builder = builder.rho_curvature_scale(self.rho_curvature_scale);
        builder = builder.rho_prior(self.rho_prior);
        builder = builder.hessian_logdet_correction(self.hessian_logdet_correction);
        builder = builder.penalty_subspace_trace(self.penalty_subspace_trace);

        if let Some(dp) = self.deriv_provider {
            builder = builder.deriv_provider(dp);
        }
        builder = builder.firth_term(self.firth);
        if let Some(nd) = self.nullspace_dim {
            builder = builder.nullspace_dim_override(nd);
        }
        builder = builder.barrier_config(self.barrier_config);
        builder = builder.kkt_residual(self.kkt_residual);
        builder = builder.active_constraints(self.active_constraints);
        builder = builder.cone_normalizer(self.cone_normalizer);

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
        builder = builder.contracted_psi_second_order(self.contracted_psi_second_order);

        builder.build()
    }

    /// Build and evaluate in one step.
    pub fn evaluate(
        self,
        rho: &[f64],
        mode: EvalMode,
        prior: Option<(f64, Array1<f64>, Option<Array2<f64>>)>,
    ) -> Result<RemlLamlResult, super::reml_outer_engine::RemlLamlError> {
        let solution = self.build();
        // The rho outer audit is a thread-local and a no-op unless armed. This
        // route gets the same fresh window and criterion record as the standard
        // assemble-and-evaluate path. Without them the coupled custom-family route
        // published its per-coordinate gradient parts but never the value
        // components those parts differentiate (#2695).
        crate::estimate::outer_eval_capture::begin_rho_outer_audit_eval();
        let result = reml_laml_evaluate(&solution, rho, mode, prior)?;
        crate::estimate::outer_eval_capture::record_rho_outer_criterion(
            result.cost,
            [
                result.criterion_components.fixed_beta,
                result.criterion_components.logdet_h,
                result.criterion_components.logdet_s,
                result.criterion_components.kkt,
            ],
        );
        crate::estimate::outer_eval_capture::record_certificate_criterion(
            crate::estimate::outer_eval_capture::CertificateCriterion {
                cost: result.cost,
                fixed_beta: result.criterion_components.fixed_beta,
                logdet_h: result.criterion_components.logdet_h,
                logdet_s: result.criterion_components.logdet_s,
                kkt: result.criterion_components.kkt,
                inner_residual_energy: result.ift_residual_energy,
            },
        );
        Ok(result)
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
) -> Result<RemlLamlResult, super::reml_outer_engine::RemlLamlError> {
    reml_laml_evaluate(solution, rho, mode, prior)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    pub(crate) fn assert_matrix_close(
        got: &Array2<f64>,
        expected: &Array2<f64>,
        epsilon: f64,
        max_relative: f64,
    ) {
        assert_eq!(got.dim(), expected.dim());
        for ((i, j), &value) in got.indexed_iter() {
            assert_relative_eq!(
                value,
                expected[[i, j]],
                epsilon = epsilon,
                max_relative = max_relative
            );
        }
    }

    pub(crate) fn deterministic_matrix(n: usize, p: usize, phase: f64) -> Array2<f64> {
        Array2::from_shape_fn((n, p), |(i, j)| {
            let a = ((i as f64 + 1.0) * (j as f64 + 3.0) + phase).sin();
            let b = ((i as f64 + 5.0) / (j as f64 + 2.0) + phase).cos();
            0.25 * a + 0.75 * b
        })
    }

    pub(crate) fn deterministic_weights(n: usize) -> Array1<f64> {
        Array1::from_shape_fn(n, |i| {
            if i % 17 == 0 {
                0.0
            } else {
                0.2 + ((i as f64 + 1.0) * 0.013).sin().abs()
            }
        })
    }

    pub(crate) fn weighted_cross_reference(
        left: &Array2<f64>,
        right: &Array2<f64>,
        weights: &Array1<f64>,
    ) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((left.ncols(), right.ncols()));
        for i in 0..weights.len() {
            for a in 0..left.ncols() {
                let scaled = weights[i] * left[[i, a]];
                for b in 0..right.ncols() {
                    out[[a, b]] += scaled * right[[i, b]];
                }
            }
        }
        out
    }

    #[test]
    pub(crate) fn row_scale_dense_into_reuses_buffer_and_matches_reference() {
        let x = deterministic_matrix(37, 11, 0.3);
        let weights = deterministic_weights(x.nrows());
        let mut out = Array2::<f64>::zeros(x.raw_dim());
        let ptr = out.as_ptr();
        row_scale_dense_into(&x, &weights, &mut out);
        assert_eq!(out.as_ptr(), ptr);
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert_relative_eq!(out[[i, j]], x[[i, j]] * weights[i], epsilon = 0.0);
            }
        }
    }

    #[test]
    pub(crate) fn weighted_cross_dense_matches_rowwise_reference_at_large_scale_block_size() {
        let left = deterministic_matrix(2048, 96, 0.1);
        let right = deterministic_matrix(2048, 64, 0.7);
        let weights = deterministic_weights(left.nrows());
        let got = weighted_cross_dense(&left, &right, &weights);
        let expected = weighted_cross_reference(&left, &right, &weights);
        assert_matrix_close(&got, &expected, 5e-10, 5e-12);
    }

    fn assert_bitwise_symmetric(got: &Array2<f64>) {
        for i in 0..got.nrows() {
            for j in 0..i {
                assert_eq!(
                    got[[i, j]].to_bits(),
                    got[[j, i]].to_bits(),
                    "Gram entry [{i}, {j}] = {:e} differs from its transpose [{j}, {i}] = {:e}",
                    got[[i, j]],
                    got[[j, i]]
                );
            }
        }
    }

    /// Every PIRLS Hessian is formed here, small or large, so the Gram must be
    /// exactly symmetric on both sides of the row-split threshold.
    #[test]
    pub(crate) fn xt_diag_x_dense_is_exactly_symmetric_small_and_large() {
        for (n, p) in [(768, 96), (40_000, 24)] {
            let x = deterministic_matrix(n, p, 1.1);
            let weights = deterministic_weights(x.nrows());
            let got = xt_diag_x_dense(&x, &weights);
            let expected = weighted_cross_reference(&x, &x, &weights);
            assert_matrix_close(&got, &expected, 1e-9, 5e-12);
            assert_bitwise_symmetric(&got);
        }
    }

    /// Observed-information weights are signed; the Gram keeps their sign.
    #[test]
    pub(crate) fn xt_diag_x_dense_keeps_signed_weights_at_scale() {
        let x = deterministic_matrix(40_000, 12, 0.4);
        let weights = Array1::from_shape_fn(x.nrows(), |i| ((i as f64) * 0.37).sin());
        let got = xt_diag_x_dense(&x, &weights);
        let expected = weighted_cross_reference(&x, &x, &weights);
        assert_matrix_close(&got, &expected, 1e-9, 5e-12);
    }

    /// The dense weighted products every large fit forms carry the same bits at
    /// every pool width. A branch that went parallel only when the pool had
    /// more than one worker summed the rows in a different order from the
    /// one-worker GEMM, so a fit's bits followed `RAYON_NUM_THREADS`.
    #[test]
    pub(crate) fn dense_weighted_products_are_bitwise_identical_across_pool_widths() {
        let x = deterministic_matrix(60_000, 20, 0.9);
        let right = deterministic_matrix(60_000, 7, 0.2);
        let weights = deterministic_weights(x.nrows());
        let run = |threads: usize| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("test pool")
                .install(|| {
                    (
                        xt_diag_x_dense(&x, &weights),
                        weighted_cross_dense(&x, &right, &weights),
                    )
                })
        };
        let (gram_1, cross_1) = run(1);
        for threads in [2, 3, 8] {
            let (gram_t, cross_t) = run(threads);
            for (a, b) in gram_1.iter().zip(gram_t.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "Gram differs at {threads} threads");
            }
            for (a, b) in cross_1.iter().zip(cross_t.iter()) {
                assert_eq!(a.to_bits(), b.to_bits(), "cross differs at {threads} threads");
            }
        }
    }
}
