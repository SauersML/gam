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
    PenaltyCoordinate, PenaltyLogdetDerivs, PenaltySubspaceTrace, RemlLamlResult,
    penalty_matrix_root, reml_laml_evaluate,
};
use crate::faer_ndarray::fast_xt_diag_y;
use ndarray::{Array1, Array2};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
//  Streaming weighted dense-design products
// ═══════════════════════════════════════════════════════════════════════════

/// Dense weighted-product work below this approximate flop count stays on the
/// caller thread and uses the existing faer GEMM path. Above the threshold we
/// stream rows through rayon-local accumulation buffers to avoid materializing
/// weighted n×p design copies at biobank scale.
const DENSE_WEIGHTED_PRODUCT_PAR_FLOPS: usize = 8_000_000;

#[inline]
fn dense_weighted_chunk_rows(cols: usize) -> usize {
    const TARGET_BYTES: usize = 2 * 1024 * 1024;
    const MIN_ROWS: usize = 256;
    const MAX_ROWS: usize = 4096;
    let bytes_per_row = cols.max(1) * std::mem::size_of::<f64>();
    (TARGET_BYTES / bytes_per_row).clamp(MIN_ROWS, MAX_ROWS)
}

/// Write `diag(scale) · x` into `out`, preserving `out`'s allocation when its
/// shape already matches `x`.
///
/// This replaces the former clone-and-row-scale pattern used by REML assembly
/// tests and Firth kernels. It is intentionally simple and deterministic for a
/// fixed row order.
pub(crate) fn row_scale_dense_into(x: &Array2<f64>, scale: &Array1<f64>, out: &mut Array2<f64>) {
    debug_assert_eq!(x.nrows(), scale.len(), "scale length must match row count");
    if out.raw_dim() != x.raw_dim() {
        *out = Array2::<f64>::zeros(x.raw_dim());
    }
    ndarray::Zip::from(out.rows_mut())
        .and(x.rows())
        .and(scale.view())
        .for_each(|mut dst, src, &w| {
            dst.assign(&src);
            dst *= w;
        });
}

fn accumulate_weighted_cross_rows(
    out: &mut Array2<f64>,
    left: &Array2<f64>,
    right: &Array2<f64>,
    weights: &Array1<f64>,
    row_start: usize,
    row_end: usize,
) {
    let p = left.ncols();
    let q = right.ncols();
    for i in row_start..row_end {
        let wi = weights[i];
        if wi == 0.0 {
            continue;
        }
        for a in 0..p {
            let scaled = wi * left[[i, a]];
            if scaled == 0.0 {
                continue;
            }
            for b in 0..q {
                out[[a, b]] += scaled * right[[i, b]];
            }
        }
    }
}

fn accumulate_xt_diag_x_upper_rows(
    out: &mut Array2<f64>,
    x: &Array2<f64>,
    diag: &Array1<f64>,
    row_start: usize,
    row_end: usize,
) {
    let p = x.ncols();
    for i in row_start..row_end {
        let wi = diag[i];
        if wi == 0.0 {
            continue;
        }
        for a in 0..p {
            let scaled = wi * x[[i, a]];
            if scaled == 0.0 {
                continue;
            }
            for b in a..p {
                out[[a, b]] += scaled * x[[i, b]];
            }
        }
    }
}

/// Compute `leftᵀ diag(weights) right` using streamed row-block
/// accumulation for large products. The parallel path allocates one dense
/// p×q accumulator per rayon worker/task instead of allocating an n×q weighted
/// design matrix.
pub(crate) fn weighted_cross_dense(
    left: &Array2<f64>,
    right: &Array2<f64>,
    weights: &Array1<f64>,
) -> Array2<f64> {
    debug_assert_eq!(left.nrows(), right.nrows());
    debug_assert_eq!(left.nrows(), weights.len());
    let n = weights.len();
    let p = left.ncols();
    let q = right.ncols();
    if n == 0 || p == 0 || q == 0 {
        return Array2::<f64>::zeros((p, q));
    }

    let work = n.saturating_mul(p).saturating_mul(q);
    if rayon::current_num_threads() <= 1 || work < DENSE_WEIGHTED_PRODUCT_PAR_FLOPS {
        return fast_xt_diag_y(left, weights, right);
    }

    let chunk_rows = dense_weighted_chunk_rows(p + q).min(n);
    let chunks = n.div_ceil(chunk_rows);
    (0..chunks)
        .into_par_iter()
        .fold(
            || Array2::<f64>::zeros((p, q)),
            |mut local, chunk| {
                let start = chunk * chunk_rows;
                let end = (start + chunk_rows).min(n);
                accumulate_weighted_cross_rows(&mut local, left, right, weights, start, end);
                local
            },
        )
        .reduce(
            || Array2::<f64>::zeros((p, q)),
            |mut a, b| {
                a += &b;
                a
            },
        )
}

/// Compute `xᵀ diag(diag) x`. For small products this reuses `weighted` as an
/// n×p row-scaled scratch and dispatches to faer GEMM. For large products it
/// streams rows into rayon-local p×p buffers and mirrors the accumulated upper
/// triangle, avoiding weighted design materialization.
pub(crate) fn xt_diag_x_dense_into(
    x: &Array2<f64>,
    diag: &Array1<f64>,
    weighted: &mut Array2<f64>,
) -> Array2<f64> {
    let (n, p) = x.dim();
    debug_assert_eq!(diag.len(), n, "diag length must match row count");
    if n == 0 || p == 0 {
        return Array2::<f64>::zeros((p, p));
    }

    let work = n.saturating_mul(p).saturating_mul(p);
    if rayon::current_num_threads() <= 1 || work < DENSE_WEIGHTED_PRODUCT_PAR_FLOPS {
        row_scale_dense_into(x, diag, weighted);
        return crate::faer_ndarray::fast_atb(x, weighted);
    }

    let chunk_rows = dense_weighted_chunk_rows(p).min(n);
    let chunks = n.div_ceil(chunk_rows);
    let mut out = (0..chunks)
        .into_par_iter()
        .fold(
            || Array2::<f64>::zeros((p, p)),
            |mut local, chunk| {
                let start = chunk * chunk_rows;
                let end = (start + chunk_rows).min(n);
                accumulate_xt_diag_x_upper_rows(&mut local, x, diag, start, end);
                local
            },
        )
        .reduce(
            || Array2::<f64>::zeros((p, p)),
            |mut a, b| {
                a += &b;
                a
            },
        );
    for a in 0..p {
        for b in 0..a {
            out[[a, b]] = out[[b, a]];
        }
    }
    out
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
    pub hessian_op: std::sync::Arc<dyn HessianOperator>,
    pub penalty_coords: Vec<PenaltyCoordinate>,
    pub penalty_logdet: PenaltyLogdetDerivs,
    pub dispersion: DispersionHandling,
    pub rho_curvature_scale: f64,
    pub rho_prior: crate::types::RhoPrior,
    pub hessian_logdet_correction: f64,
    pub penalty_subspace_trace: Option<Arc<PenaltySubspaceTrace>>,

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

    // === Implicit-function-theorem correction input ===
    /// Optional inner KKT residual r = ∇_β L_pen(β̂) at the converged β̂.
    /// `Some` activates the IFT correction in `reml_laml_evaluate` that
    /// absorbs inner KKT slack into the outer cost and gradient (see
    /// `InnerSolution::kkt_residual`). `None` keeps the envelope-only
    /// behaviour for callers that genuinely guarantee exact KKT.
    pub kkt_residual: Option<Array1<f64>>,
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
        builder = builder.rho_prior(self.rho_prior);
        builder = builder.hessian_logdet_correction(self.hessian_logdet_correction);
        builder = builder.penalty_subspace_trace(self.penalty_subspace_trace);

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
        builder = builder.kkt_residual(self.kkt_residual);

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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::Array2;

    fn assert_matrix_close(
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

    fn deterministic_matrix(n: usize, p: usize, phase: f64) -> Array2<f64> {
        Array2::from_shape_fn((n, p), |(i, j)| {
            let a = ((i as f64 + 1.0) * (j as f64 + 3.0) + phase).sin();
            let b = ((i as f64 + 5.0) / (j as f64 + 2.0) + phase).cos();
            0.25 * a + 0.75 * b
        })
    }

    fn deterministic_weights(n: usize) -> Array1<f64> {
        Array1::from_shape_fn(n, |i| {
            if i % 17 == 0 {
                0.0
            } else {
                0.2 + ((i as f64 + 1.0) * 0.013).sin().abs()
            }
        })
    }

    fn weighted_cross_reference(
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
    fn row_scale_dense_into_reuses_buffer_and_matches_reference() {
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
    fn weighted_cross_dense_matches_rowwise_reference_at_biobank_block_size() {
        let left = deterministic_matrix(2048, 96, 0.1);
        let right = deterministic_matrix(2048, 64, 0.7);
        let weights = deterministic_weights(left.nrows());
        let got = weighted_cross_dense(&left, &right, &weights);
        let expected = weighted_cross_reference(&left, &right, &weights);
        assert_matrix_close(&got, &expected, 5e-10, 5e-12);
    }

    #[test]
    fn xt_diag_x_dense_into_matches_symmetric_reference_at_biobank_block_size() {
        let x = deterministic_matrix(1024, 96, 1.1);
        let weights = deterministic_weights(x.nrows());
        let mut scratch = Array2::<f64>::zeros((0, 0));
        let got = xt_diag_x_dense_into(&x, &weights, &mut scratch);
        let expected = weighted_cross_reference(&x, &x, &weights);
        assert_matrix_close(&got, &expected, 3e-10, 5e-12);
        for i in 0..got.nrows() {
            for j in 0..got.ncols() {
                assert_relative_eq!(got[[i, j]], got[[j, i]], epsilon = 0.0);
            }
        }
    }
}
