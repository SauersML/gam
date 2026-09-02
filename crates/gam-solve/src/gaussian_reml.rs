use crate::estimate::EstimationError;
use crate::rho_optimizer::{FallbackPolicy, OuterProblem};
use faer::Side;
use gam_linalg::faer_ndarray::{
    FaerCholesky, FaerEigh, default_rrqr_rank_alpha, fast_ab, fast_atb, fast_xt_diag_x,
    fast_xt_diag_y, rrqr_with_permutation,
};
use gam_problem::{
    DeclaredHessianForm, Derivative, HessianValue, OuterEval, StationarityStandard,
};
use gam_terms::construction::CanonicalPenalty;
use gam_terms::smooth::BlockwisePenalty;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, s};
use opt::{RidgeSchedule, escalate_ridge};
use rayon::prelude::*;
use std::sync::Once;

/// One-time warning latch for backward-pass graceful degradation on a
/// near-singular penalized Hessian `K = XᵀWX + λS`. When `λ_k` saturates
/// (e.g. 1e10+), `K` becomes effectively rank-deficient and the analytic VJP
/// cannot be evaluated. Rather than raising, the backward returns zero
/// gradients of the correct shape: this is the statistically correct
/// "shrink-out" gradient — when `λ` has saturated, the atom is unused, so
/// every input's contribution to the loss is zero in the limit.
static ILL_CONDITIONED_BACKWARD_WARNED: Once = Once::new();

fn warn_ill_conditioned_backward_once(p: usize, d: usize, condition_number: f64) {
    ILL_CONDITIONED_BACKWARD_WARNED.call_once(|| {
        log::warn!(
            "gaussian_reml_fit_backward: K = XᵀWX + λS is near-singular \
             (p={p}, d={d}, cond≈{condition_number:.2e}); returning zero gradients \
             for this fit (λ has saturated, atom is effectively unused). \
             Further occurrences are silent."
        );
    });
}

fn zero_backward_result(n: usize, p: usize, d: usize) -> GaussianRemlBackwardResult {
    GaussianRemlBackwardResult {
        grad_x: Array2::<f64>::zeros((n, p)),
        grad_y: Array2::<f64>::zeros((n, d)),
        grad_penalty: Array2::<f64>::zeros((p, p)),
        grad_weights: Array1::<f64>::zeros(n),
    }
}

/// Smoothing-parameter search box in log strength. Public because a caller that
/// differentiates the λ̂ ROOT (the implicit-function channel) must apply the same
/// interior test this file's own backward VJP applies — an interior premise
/// checked against a privately duplicated bound is the desync this crate exists
/// to prevent.
pub const RHO_LOWER: f64 = -30.0;
pub const RHO_UPPER: f64 = 30.0;
const EIGEN_REL_TOL: f64 = 1.0e-10;
const MIN_DEVIANCE: f64 = 1.0e-300;
/// Relative first-order convergence certificate for the block-orthogonal
/// alternation: the largest per-block |dV/drho|, normalized by the score's
/// natural magnitude `d * max(1, rank)`, must fall below this and the analytic
/// profiled Hessian must be PSD before a fit is minted. See
/// `gaussian_reml_blocks_orthogonal_shared_scale`.
const BLOCK_ORTHOGONAL_SCORE_TOL: f64 = 1.0e-7;
/// Exhaustion-escalation bound on outer alternation passes. It never selects
/// the estimator: reaching it without the score/curvature certificate is a typed
/// `BlockOrthogonalRemlDidNotConverge` error carrying the rho checkpoint.
const BLOCK_ORTHOGONAL_MAX_OUTER_PASSES: usize = 200;
/// Work allocated to each one-dimensional block polish within an outer pass.
/// This is not a convergence criterion: the joint analytic score below is the
/// only condition that can mint a fit.
const BLOCK_ORTHOGONAL_BLOCK_UPDATES_PER_PASS: usize = 32;

/// Canonical coefficient-domain contract for the raw multi-block Gaussian
/// REML entry point.
///
/// Every block is classified once by the terms-layer spectral policy.  The
/// resulting roots define both the nullities supplied to the forward REML
/// objective and the augmented operator whose full column rank identifies the
/// coefficient map:
///
/// ```text
/// A(lambda) = [sqrt(W) X; sqrt(lambda_1) R_1; ...; sqrt(lambda_F) R_F].
/// ```
///
/// Thus forward and backward cannot disagree about penalty rank, and a shared
/// design/penalty null direction is rejected before an optimizer can mint
/// gauge-dependent coefficients.  There is deliberately no ridge, spectral
/// floor, pseudoinverse coefficient solve, or compatibility fallback.
#[derive(Clone)]
pub struct GaussianRemlBlocksDomain {
    p_total: usize,
    canonical_penalties: Vec<CanonicalPenalty>,
    nullspace_dims: Vec<usize>,
}

impl GaussianRemlBlocksDomain {
    pub fn from_blockwise_penalties(
        p_total: usize,
        penalties: &[BlockwisePenalty],
    ) -> Result<Self, EstimationError> {
        if p_total == 0 || penalties.is_empty() {
            return Err(EstimationError::InvalidInput(
                "block Gaussian REML domain requires at least one coefficient and one penalty block"
                    .to_string(),
            ));
        }

        let mut canonical_penalties = Vec::with_capacity(penalties.len());
        let mut nullspace_dims = Vec::with_capacity(penalties.len());
        let mut expected_start = 0_usize;
        for (block, penalty) in penalties.iter().enumerate() {
            if penalty.col_range.start != expected_start
                || penalty.col_range.end <= penalty.col_range.start
            {
                return Err(EstimationError::InvalidInput(format!(
                    "block Gaussian REML penalties must form a non-empty contiguous partition: \
                     block {block} has range {:?}, expected start {expected_start}",
                    penalty.col_range
                )));
            }
            expected_start = penalty.col_range.end;

            let spec = gam_terms::PenaltySpec::from_blockwise_ref(penalty);
            let canonical = gam_terms::construction::canonicalize_penalty_spec(
                &spec,
                p_total,
                block,
                "block Gaussian REML domain",
            )?
            .ok_or_else(|| {
                EstimationError::InvalidInput(format!(
                    "block Gaussian REML penalty {block} has no positive-curvature direction"
                ))
            })?;
            let block_dim = canonical.block_dim();
            let rank = canonical.rank();
            if rank + canonical.nullity != block_dim {
                return Err(EstimationError::InvalidInput(format!(
                    "block Gaussian REML penalty {block} is not positive semidefinite under the \
                     canonical spectral classification: rank={rank}, nullity={}, dimension={block_dim}",
                    canonical.nullity
                )));
            }
            if canonical.positive_eigenvalues.len() != rank {
                return Err(EstimationError::InvalidInput(format!(
                    "block Gaussian REML penalty {block} canonical root/eigenspectrum mismatch: \
                     root rank={rank}, positive eigenvalues={}",
                    canonical.positive_eigenvalues.len()
                )));
            }
            nullspace_dims.push(canonical.nullity);
            canonical_penalties.push(canonical);
        }
        if expected_start != p_total {
            return Err(EstimationError::InvalidInput(format!(
                "block Gaussian REML penalty partition ends at {expected_start}, \
                 but the joint design has {p_total} columns"
            )));
        }

        Ok(Self {
            p_total,
            canonical_penalties,
            nullspace_dims,
        })
    }

    #[inline]
    pub fn nullspace_dims(&self) -> &[usize] {
        &self.nullspace_dims
    }

    fn local_penalties(&self) -> Vec<Array2<f64>> {
        self.canonical_penalties
            .iter()
            .map(CanonicalPenalty::local_penalty)
            .collect()
    }

    fn normal_matrix(
        &self,
        xtwx: &Array2<f64>,
        lambdas: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        if xtwx.dim() != (self.p_total, self.p_total) {
            return Err(EstimationError::InvalidInput(format!(
                "block Gaussian REML Gram shape mismatch: expected {}x{}, got {}x{}",
                self.p_total,
                self.p_total,
                xtwx.nrows(),
                xtwx.ncols()
            )));
        }
        if lambdas.len() != self.canonical_penalties.len() {
            return Err(EstimationError::InvalidInput(format!(
                "block Gaussian REML lambda count mismatch: expected {}, got {}",
                self.canonical_penalties.len(),
                lambdas.len()
            )));
        }
        let mut normal = xtwx.clone();
        for (block, penalty) in self.canonical_penalties.iter().enumerate() {
            let lambda = lambdas[block];
            if !lambda.is_finite() || lambda <= 0.0 {
                return Err(EstimationError::InvalidInput(format!(
                    "block Gaussian REML lambda[{block}] must be finite and positive; got {lambda}"
                )));
            }
            penalty.accumulate_weighted(&mut normal, lambda);
        }
        gam_linalg::matrix::symmetrize_in_place(&mut normal);
        Ok(normal)
    }

    /// Moore-Penrose inverse on the canonical positive-curvature range.
    ///
    /// This is not a coefficient-solve fallback: it is the derivative of the
    /// REML penalty pseudo-determinant.  Building it from the already
    /// classified root/eigenvalue pairs prevents a second rank policy from
    /// silently changing the differentiated objective.
    fn penalty_pseudoinverses(&self) -> Result<Vec<Array2<f64>>, EstimationError> {
        let mut out = Vec::with_capacity(self.canonical_penalties.len());
        for (block, penalty) in self.canonical_penalties.iter().enumerate() {
            let k = penalty.block_dim();
            let mut pinv = Array2::<f64>::zeros((k, k));
            for (row, &eigenvalue) in penalty.positive_eigenvalues.iter().enumerate() {
                if !eigenvalue.is_finite() || eigenvalue <= 0.0 {
                    return Err(EstimationError::InvalidInput(format!(
                        "block Gaussian REML penalty {block} has invalid canonical positive \
                         eigenvalue {row}: {eigenvalue}"
                    )));
                }
                let scale = 1.0 / (eigenvalue * eigenvalue);
                for i in 0..k {
                    for j in 0..k {
                        pinv[[i, j]] += scale * penalty.root[[row, i]] * penalty.root[[row, j]];
                    }
                }
            }
            if pinv.iter().any(|value| !value.is_finite()) {
                return Err(EstimationError::InvalidInput(format!(
                    "block Gaussian REML penalty {block} canonical pseudoinverse is not representable"
                )));
            }
            out.push(pinv);
        }
        Ok(out)
    }

    /// Certify that the supplied design and positive penalty scales determine
    /// a unique coefficient vector, returning the exact normal matrix used by
    /// the strict solve.
    pub fn certify_joint_coefficient_map(
        &self,
        design: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
        lambdas: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, EstimationError> {
        if design.ncols() != self.p_total || weights.len() != design.nrows() {
            return Err(EstimationError::InvalidInput(format!(
                "block Gaussian REML domain shape mismatch: design={}x{}, weights={}, coefficients={}",
                design.nrows(),
                design.ncols(),
                weights.len(),
                self.p_total
            )));
        }
        if lambdas.len() != self.canonical_penalties.len() {
            return Err(EstimationError::InvalidInput(format!(
                "block Gaussian REML lambda count mismatch: expected {}, got {}",
                self.canonical_penalties.len(),
                lambdas.len()
            )));
        }
        if let Some(((row, col), value)) =
            design.indexed_iter().find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "block Gaussian REML design[{row},{col}] must be finite; got {value}"
            )));
        }
        if let Some((row, value)) = weights
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite() || **value < 0.0)
        {
            return Err(EstimationError::InvalidInput(format!(
                "block Gaussian REML weights[{row}] must be finite and non-negative; got {value}"
            )));
        }
        if let Some((block, value)) = lambdas
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite() || **value <= 0.0)
        {
            return Err(EstimationError::InvalidInput(format!(
                "block Gaussian REML lambda[{block}] must be finite and positive; got {value}"
            )));
        }

        let augmented_rows = design.nrows()
            + self
                .canonical_penalties
                .iter()
                .map(CanonicalPenalty::rank)
                .sum::<usize>();
        let mut augmented = Array2::<f64>::zeros((augmented_rows, self.p_total));
        for row in 0..design.nrows() {
            let scale = weights[row].sqrt();
            for col in 0..self.p_total {
                augmented[[row, col]] = scale * design[[row, col]];
            }
        }
        let mut augmented_row = design.nrows();
        for (block, penalty) in self.canonical_penalties.iter().enumerate() {
            let scale = lambdas[block].sqrt();
            for root_row in 0..penalty.rank() {
                for local_col in 0..penalty.block_dim() {
                    augmented[[
                        augmented_row + root_row,
                        penalty.col_range.start + local_col,
                    ]] = scale * penalty.root[[root_row, local_col]];
                }
            }
            augmented_row += penalty.rank();
        }

        let rank = rrqr_with_permutation(&augmented, default_rrqr_rank_alpha())
            .map_err(|error| {
                EstimationError::InvalidInput(format!(
                    "block Gaussian REML augmented-rank certificate failed: {error}"
                ))
            })?
            .rank;
        if rank != self.p_total {
            return Err(EstimationError::InvalidInput(format!(
                "block Gaussian REML joint coefficient map is not identified: \
                 augmented operator [sqrt(W)X; sqrt(lambda_k)R_k] has numerical \
                 rank {rank} < {}; constrain shared design/penalty-null directions \
                 before fitting",
                self.p_total
            )));
        }

        let xtwx = fast_xt_diag_x(&design, &weights);
        let normal = self.normal_matrix(&xtwx, lambdas)?;
        gam_linalg::utils::certified_spd_factorize(
            &normal,
            "block Gaussian REML penalized normal matrix",
        )
        .map_err(|error| {
            EstimationError::InvalidInput(format!(
                "block Gaussian REML requires an exact SPD penalized normal matrix: {error}"
            ))
        })?;
        Ok(normal)
    }
}

#[derive(Clone, Debug)]
pub struct GaussianRemlBlocksResult {
    pub coefficients: Array2<f64>,
    pub fitted: Array2<f64>,
    pub lambdas: Array1<f64>,
    pub log_lambdas: Array1<f64>,
    pub reml_score: f64,
    pub edf: Array1<f64>,
}

struct GaussianRemlBlocksProfile {
    domain: GaussianRemlBlocksDomain,
    design: Array2<f64>,
    weights: Array1<f64>,
    y: Array1<f64>,
    xtwx: Array2<f64>,
    xtwy: Array1<f64>,
    nu: f64,
}

struct GaussianRemlBlocksProfileEval {
    cost: f64,
    gradient: Array1<f64>,
    hessian: Array2<f64>,
    lambdas: Array1<f64>,
    coefficients: Array1<f64>,
    fitted: Array1<f64>,
    edf: Array1<f64>,
}

impl GaussianRemlBlocksProfile {
    fn evaluate(
        &self,
        rhos: ArrayView1<'_, f64>,
    ) -> Result<GaussianRemlBlocksProfileEval, EstimationError> {
        let f_blocks = self.domain.canonical_penalties.len();
        if rhos.len() != f_blocks {
            return Err(EstimationError::InvalidInput(format!(
                "block Gaussian REML rho count mismatch: expected {f_blocks}, got {}",
                rhos.len()
            )));
        }
        let lambdas = Array1::from_vec(
            gam_problem::checked_exp_log_strengths(rhos.iter().copied())
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
        );
        let normal = self.domain.normal_matrix(&self.xtwx, lambdas.view())?;
        let inverse = gam_linalg::utils::certified_spd_inverse(
            &normal,
            "block Gaussian REML penalized normal matrix",
        )
        .map(gam_linalg::utils::CertifiedSpdInverse::into_inverse)
        .map_err(|error| {
            EstimationError::InvalidInput(format!(
                "block Gaussian REML requires an exact SPD penalized normal matrix: {error}"
            ))
        })?;
        // The inverse above certifies this exact, unperturbed normal matrix.
        // A second strict Cholesky supplies its determinant without exposing a
        // separate repaired spectrum or rank policy.
        let lower = normal
            .cholesky(Side::Lower)
            .map_err(|error| {
                EstimationError::InvalidInput(format!(
                    "block Gaussian REML penalized normal log-determinant failed: {error}"
                ))
            })?
            .lower_triangular();
        let logdet_normal = 2.0 * lower.diag().iter().map(|value| value.ln()).sum::<f64>();
        if !logdet_normal.is_finite() {
            return Err(EstimationError::InvalidInput(
                "block Gaussian REML penalized normal log-determinant is not finite".to_string(),
            ));
        }

        let coefficients = inverse.dot(&self.xtwy);
        let fitted = self.design.dot(&coefficients);
        let residual = &self.y - &fitted;

        // q = y'Wy - b'K^-1b = r'Wr + beta'P beta.  The right-hand form is a
        // sum of non-negative terms and remains representable for nearly
        // interpolating designs; replacing a non-positive q by a floor would
        // change both the objective and the VJP, so the profile is refused.
        let mut q = residual
            .iter()
            .zip(self.weights.iter())
            .map(|(&value, &weight)| weight * value * value)
            .sum::<f64>();
        let mut logdet_penalty = 0.0_f64;
        let mut p_betas = Vec::with_capacity(f_blocks);
        let mut rp_matrices = Vec::with_capacity(f_blocks);
        let mut b_values = Array1::<f64>::zeros(f_blocks);
        let mut t_values = Array1::<f64>::zeros(f_blocks);
        let mut edf = Array1::<f64>::zeros(f_blocks);
        for (block, penalty) in self.domain.canonical_penalties.iter().enumerate() {
            let start = penalty.col_range.start;
            let end = penalty.col_range.end;
            let beta_block = coefficients.slice(s![start..end]);
            let local_p_beta = penalty.local.dot(&beta_block);
            let lambda = lambdas[block];
            let mut p_beta = Array1::<f64>::zeros(self.domain.p_total);
            for local in 0..penalty.block_dim() {
                p_beta[start + local] = lambda * local_p_beta[local];
            }
            let b_value = coefficients.dot(&p_beta);
            q += b_value;
            b_values[block] = b_value;

            let weighted_penalty = penalty.local.mapv(|value| lambda * value);
            let rp_block = inverse
                .slice(s![.., start..end])
                .dot(&weighted_penalty);
            let mut rp = Array2::<f64>::zeros((self.domain.p_total, self.domain.p_total));
            rp.slice_mut(s![.., start..end]).assign(&rp_block);
            let trace = (0..penalty.block_dim())
                .map(|local| rp_block[[start + local, local]])
                .sum::<f64>();
            t_values[block] = trace;
            edf[block] = penalty.block_dim() as f64 - trace;
            logdet_penalty += penalty
                .positive_eigenvalues
                .iter()
                .map(|eigenvalue| eigenvalue.ln())
                .sum::<f64>()
                + penalty.rank() as f64 * rhos[block];
            p_betas.push(p_beta);
            rp_matrices.push(rp);
        }
        if !q.is_finite() || q <= 0.0 {
            return Err(EstimationError::InvalidInput(format!(
                "block Gaussian REML profiled residual quadratic form must be finite and positive; got {q}"
            )));
        }
        if !logdet_penalty.is_finite() {
            return Err(EstimationError::InvalidInput(
                "block Gaussian REML penalty pseudo-log-determinant is not finite".to_string(),
            ));
        }

        let tau = self.nu / q;
        let tau_q = -self.nu / (q * q);
        let cost = 0.5
            * (self.nu
                * (1.0 + (2.0 * std::f64::consts::PI * q / self.nu).ln())
                + logdet_normal
                - logdet_penalty);
        let mut gradient = Array1::<f64>::zeros(f_blocks);
        for block in 0..f_blocks {
            gradient[block] = 0.5
                * (t_values[block]
                    - self.domain.canonical_penalties[block].rank() as f64
                    + tau * b_values[block]);
        }

        let mut hessian = Array2::<f64>::zeros((f_blocks, f_blocks));
        for k in 0..f_blocks {
            for j in 0..f_blocks {
                let trace_pair = gam_linalg::utils::trace_of_product(
                    rp_matrices[k].view(),
                    rp_matrices[j].view(),
                );
                let beta_pk_r_pj_beta = p_betas[k].dot(&inverse.dot(&p_betas[j]));
                hessian[[k, j]] = 0.5
                    * ((if k == j { t_values[k] } else { 0.0 }) - trace_pair
                        + tau_q * b_values[k] * b_values[j]
                        + tau
                            * ((if k == j { b_values[k] } else { 0.0 })
                                - 2.0 * beta_pk_r_pj_beta));
            }
        }
        gam_linalg::matrix::symmetrize_in_place(&mut hessian);
        if !cost.is_finite()
            || coefficients.iter().any(|value| !value.is_finite())
            || fitted.iter().any(|value| !value.is_finite())
            || edf.iter().any(|value| !value.is_finite())
            || gradient.iter().any(|value| !value.is_finite())
            || hessian.iter().any(|value| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(
                "block Gaussian REML profile evaluation produced a non-finite value".to_string(),
            ));
        }

        Ok(GaussianRemlBlocksProfileEval {
            cost,
            gradient,
            hessian,
            lambdas,
            coefficients,
            fitted,
            edf,
        })
    }
}

fn gaussian_reml_blocks_profile_cost(
    state: &mut GaussianRemlBlocksProfile,
    rhos: &Array1<f64>,
) -> Result<f64, EstimationError> {
    Ok(state.evaluate(rhos.view())?.cost)
}

fn gaussian_reml_blocks_profile_outer_eval(
    state: &mut GaussianRemlBlocksProfile,
    rhos: &Array1<f64>,
) -> Result<OuterEval, EstimationError> {
    let evaluated = state.evaluate(rhos.view())?;
    Ok(OuterEval {
        cost: evaluated.cost,
        gradient: evaluated.gradient,
        hessian: HessianValue::Dense(evaluated.hessian),
        inner_beta_hint: Some(evaluated.coefficients),
    })
}

/// Exact profiled Gaussian REML for a joint additive design with one
/// smoothing parameter per coefficient block.
///
/// The scalar value, score, Hessian, and analytic backward all use the same
/// criterion
///
/// `V = 1/2 { nu [1 + log(2 pi q / nu)] + log|K| - log|P|_+ }`,
///
/// where `K = X'WX + P`, `P = blockdiag(lambda_k S_k)`, and
/// `q = y'Wy - (X'Wy)' K^-1 (X'Wy)`.  The one-block case deliberately reduces
/// through the established grid-free scalar solver, making that algebraic
/// reduction exact rather than merely numerically close.
pub fn gaussian_reml_fit_blocks_exact(
    designs: &[Array2<f64>],
    penalties: &[Array2<f64>],
    y: ArrayView1<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_rhos: Option<&[f64]>,
) -> Result<GaussianRemlBlocksResult, EstimationError> {
    let f_blocks = designs.len();
    if f_blocks == 0 || penalties.len() != f_blocks {
        return Err(EstimationError::InvalidInput(format!(
            "exact block Gaussian REML requires equal non-zero design and penalty block counts; \
             got designs={}, penalties={}",
            f_blocks,
            penalties.len()
        )));
    }
    if let Some(rhos) = init_rhos {
        if rhos.len() != f_blocks {
            return Err(EstimationError::InvalidInput(format!(
                "exact block Gaussian REML init_rhos length mismatch: expected {f_blocks}, got {}",
                rhos.len()
            )));
        }
        if let Some((block, value)) = rhos
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "exact block Gaussian REML init_rhos[{block}] must be finite; got {value}"
            )));
        }
    }

    let n = y.len();
    if n == 0 {
        return Err(EstimationError::InvalidInput(
            "exact block Gaussian REML requires at least one observation".to_string(),
        ));
    }
    if let Some((row, value)) = y.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(EstimationError::InvalidInput(format!(
            "exact block Gaussian REML y[{row}] must be finite; got {value}"
        )));
    }

    let mut offsets = Vec::with_capacity(f_blocks + 1);
    offsets.push(0_usize);
    let mut p_total = 0_usize;
    for (block, (design, penalty)) in designs.iter().zip(penalties.iter()).enumerate() {
        if design.nrows() != n {
            return Err(EstimationError::InvalidInput(format!(
                "exact block Gaussian REML designs[{block}] has {} rows, expected {n}",
                design.nrows()
            )));
        }
        if design.ncols() == 0 || penalty.dim() != (design.ncols(), design.ncols()) {
            return Err(EstimationError::InvalidInput(format!(
                "exact block Gaussian REML block {block} requires a non-empty square penalty \
                 matching its {} design columns; got {}x{}",
                design.ncols(),
                penalty.nrows(),
                penalty.ncols()
            )));
        }
        if let Some(((row, col), value)) =
            design.indexed_iter().find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "exact block Gaussian REML designs[{block}][{row},{col}] must be finite; got {value}"
            )));
        }
        if let Some(((row, col), value)) =
            penalty.indexed_iter().find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "exact block Gaussian REML penalties[{block}][{row},{col}] must be finite; got {value}"
            )));
        }
        p_total += design.ncols();
        offsets.push(p_total);
    }

    let weight = gaussian_reml_weights(n, weights)?;
    let mut design = Array2::<f64>::zeros((n, p_total));
    let mut blockwise_penalties = Vec::with_capacity(f_blocks);
    let mut canonical_keys = Vec::with_capacity(f_blocks);
    for block in 0..f_blocks {
        design
            .slice_mut(s![.., offsets[block]..offsets[block + 1]])
            .assign(&designs[block]);
        blockwise_penalties.push(BlockwisePenalty::new(
            offsets[block]..offsets[block + 1],
            penalties[block].clone(),
        ));
        canonical_keys.push(fnv1a_mix(
            matrix_fingerprint(designs[block].view()),
            matrix_fingerprint(penalties[block].view()),
        ));
    }
    let domain =
        GaussianRemlBlocksDomain::from_blockwise_penalties(p_total, &blockwise_penalties)?;
    let unit_lambdas = Array1::<f64>::ones(f_blocks);
    domain.certify_joint_coefficient_map(design.view(), weight.view(), unit_lambdas.view())?;

    let n_effective = effective_observation_count(weight.view());
    let nullity = domain.nullspace_dims.iter().sum::<usize>();
    if n_effective <= nullity {
        return Err(EstimationError::InvalidInput(format!(
            "exact block Gaussian REML requires more positive-weight rows than total penalty \
             nullity; got n_effective={n_effective}, nullity={nullity}"
        )));
    }

    if f_blocks == 1 {
        // The scalar solver whitens by X'WX.  Certify that exact matrix before
        // delegating so this block entry point never reaches the scalar
        // compatibility jitter path.
        let xtwx = fast_xt_diag_x(&design.view(), &weight.view());
        gam_linalg::utils::certified_spd_factorize(
            &xtwx,
            "one-block Gaussian REML unpenalized normal matrix",
        )
        .map_err(|error| {
            EstimationError::InvalidInput(format!(
                "one-block Gaussian REML requires an exact SPD unpenalized normal matrix: {error}"
            ))
        })?;
        let scalar = gaussian_reml_closed_form(
            design.view(),
            y,
            penalties[0].view(),
            Some(weight.view()),
            init_rhos.map(|rhos| rhos[0]),
        )?;
        let lambdas = Array1::from_elem(1, scalar.lambda);
        domain.certify_joint_coefficient_map(design.view(), weight.view(), lambdas.view())?;
        return Ok(GaussianRemlBlocksResult {
            coefficients: scalar.coefficients.insert_axis(Axis(1)),
            fitted: scalar.fitted.insert_axis(Axis(1)),
            lambdas,
            log_lambdas: Array1::from_elem(1, scalar.rho),
            reml_score: scalar.reml_score,
            edf: Array1::from_elem(1, scalar.edf),
        });
    }

    let xtwx = fast_xt_diag_x(&design.view(), &weight.view());
    let y_owned = y.to_owned();
    let y_matrix = y_owned.view().insert_axis(Axis(1));
    let xtwy = fast_xt_diag_y(&design.view(), &weight.view(), &y_matrix)
        .column(0)
        .to_owned();
    let profile = GaussianRemlBlocksProfile {
        domain,
        design,
        weights: weight,
        y: y_owned,
        xtwx,
        xtwy,
        nu: (n_effective - nullity) as f64,
    };

    let mut seed_config = gam_problem::SeedConfig::default();
    seed_config.bounds = (RHO_LOWER, RHO_UPPER);
    seed_config.risk_profile = gam_problem::SeedRiskProfile::Gaussian;
    let mut problem = OuterProblem::new(f_blocks)
        .with_gradient(Derivative::Analytic)
        .with_hessian(DeclaredHessianForm::Dense)
        .with_prefer_gradient_only(false)
        .with_disable_fixed_point(true)
        .with_tolerance(1.0e-10)
        .with_required_projected_gradient_norm(Some(1.0e-8))
        .with_max_iter(200)
        .with_bounds(
            Array1::from_elem(f_blocks, RHO_LOWER),
            Array1::from_elem(f_blocks, RHO_UPPER),
        )
        .with_rho_bound(RHO_UPPER)
        .with_seed_config(seed_config)
        .with_rho_canonical_keys(Some(canonical_keys))
        .with_fallback_policy(FallbackPolicy::Disabled)
        .with_problem_size(n, p_total);
    if let Some(rhos) = init_rhos {
        problem = problem
            .with_initial_rho(Array1::from_iter(
                rhos
                    .iter()
                    .map(|rho| rho.clamp(RHO_LOWER, RHO_UPPER)),
            ))
            .with_screen_initial_rho(true);
    }
    let mut objective = problem.build_objective(
        profile,
        gaussian_reml_blocks_profile_cost,
        gaussian_reml_blocks_profile_outer_eval,
        None::<fn(&mut GaussianRemlBlocksProfile)>,
        None::<
            fn(
                &mut GaussianRemlBlocksProfile,
                &Array1<f64>,
            ) -> Result<gam_problem::EfsEval, EstimationError>,
        >,
    );
    let optimum = problem.run(&mut objective, "exact block Gaussian REML")?;
    let final_eval = objective.state.evaluate(optimum.rho.view())?;
    objective.state.domain.certify_joint_coefficient_map(
        objective.state.design.view(),
        objective.state.weights.view(),
        final_eval.lambdas.view(),
    )?;

    Ok(GaussianRemlBlocksResult {
        coefficients: final_eval.coefficients.insert_axis(Axis(1)),
        fitted: final_eval.fitted.insert_axis(Axis(1)),
        lambdas: final_eval.lambdas,
        log_lambdas: optimum.rho,
        reml_score: final_eval.cost,
        edf: final_eval.edf,
    })
}

#[derive(Clone, Copy)]
struct BlockOrthogonalControls {
    score_tol: f64,
    max_outer_passes: usize,
    block_updates_per_pass: usize,
}

impl Default for BlockOrthogonalControls {
    fn default() -> Self {
        Self {
            score_tol: BLOCK_ORTHOGONAL_SCORE_TOL,
            max_outer_passes: BLOCK_ORTHOGONAL_MAX_OUTER_PASSES,
            block_updates_per_pass: BLOCK_ORTHOGONAL_BLOCK_UPDATES_PER_PASS,
        }
    }
}

/// Canonicalize a penalty matrix to its symmetric average.
///
/// Closed-form Gaussian REML treats `S` as symmetric throughout — the
/// eigendecomposition, the pseudo-determinant `log|S|₊`, the rank detector,
/// and every per-helper VJP all assume `S = Sᵀ`. To make that contract
/// explicit (rather than implicit in `eigh(Side::Lower)` reading the lower
/// triangle and silently ignoring the upper), every entry point that takes a
/// penalty matrix replaces it with `0.5 (S + Sᵀ)` before any downstream use.
/// For symmetric input this is a numerical no-op; for asymmetric input it
/// defines the function as operating on the symmetric average.
fn canonicalize_penalty(penalty: ArrayView2<'_, f64>) -> Array2<f64> {
    let p = penalty.nrows();
    let mut out = penalty.to_owned();
    for i in 0..p {
        for j in (i + 1)..p {
            let avg = 0.5 * (out[[i, j]] + out[[j, i]]);
            out[[i, j]] = avg;
            out[[j, i]] = avg;
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct GaussianRemlEigenCache {
    pub penalty_eigenvalues: Array1<f64>,
    pub eigenvectors: Array2<f64>,
    pub coefficient_basis: Array2<f64>,
    pub xtwx_fingerprint: u64,
    pub penalty_fingerprint: u64,
    pub logdet_xtwx: f64,
    pub logdet_penalty_positive: f64,
    pub penalty_rank: usize,
    pub nullity: usize,
}

#[derive(Clone, Debug, Default)]
pub struct GaussianRemlWarmStart {
    pub lambda: Option<f64>,
    pub eigen_cache: Option<GaussianRemlEigenCache>,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlResult {
    pub lambda: f64,
    pub rho: f64,
    pub coefficients: Array1<f64>,
    pub fitted: Array1<f64>,
    pub reml_score: f64,
    pub reml_grad_lambda: f64,
    pub reml_hess_lambda: f64,
    pub reml_grad_rho: f64,
    pub reml_hess_rho: f64,
    pub edf: f64,
    pub sigma2: f64,
    pub cache: GaussianRemlEigenCache,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlMultiResult {
    pub lambda: f64,
    pub rho: f64,
    pub coefficients: Array2<f64>,
    pub fitted: Array2<f64>,
    pub reml_score: f64,
    /// Forward-error bound on `reml_score`, accumulated by the evaluator that
    /// produced it from the magnitudes of the log-determinants it differenced
    /// and the cancellation that formed each profiled deviance (#2729).
    ///
    /// `None` means NO bound was accumulated — the only producer of `Some` is
    /// the closed-form evaluator itself, so a result rebuilt from a serialized
    /// wire format (which does not carry it) says so rather than inventing one.
    /// A consumer that compares two REML scores must treat `None` as "this
    /// comparison has no established resolution", never as zero.
    pub reml_score_roundoff: Option<f64>,
    pub reml_grad_lambda: f64,
    pub reml_hess_lambda: f64,
    pub reml_grad_rho: f64,
    pub reml_hess_rho: f64,
    pub edf: f64,
    pub sigma2: Array1<f64>,
    pub cache: GaussianRemlEigenCache,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlFreeBScore {
    pub reml_score: f64,
    pub grad_coefficients: Array2<f64>,
    pub grad_penalty: Array2<f64>,
    pub grad_log_lambda: f64,
    pub fitted: Array2<f64>,
    pub sigma2: Array1<f64>,
    pub edf: f64,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlBackwardResult {
    pub grad_x: Array2<f64>,
    pub grad_y: Array2<f64>,
    pub grad_penalty: Array2<f64>,
    pub grad_weights: Array1<f64>,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlMultiBackwardProblem<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView2<'a, f64>,
    pub weights: Option<ArrayView1<'a, f64>>,
    pub fit: &'a GaussianRemlMultiResult,
    pub grad_lambda: f64,
    pub grad_coefficients: Option<ArrayView2<'a, f64>>,
    pub grad_fitted: Option<ArrayView2<'a, f64>>,
    pub grad_reml_score: f64,
    pub grad_edf: f64,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlNoAllocWorkspace {
    pub xtwy: Array2<f64>,
    pub ywy: Array1<f64>,
    pub projected_rhs: Array2<f64>,
    pub projected_rhs_squared: Array2<f64>,
    pub scaled_projected_rhs: Array2<f64>,
}

impl GaussianRemlNoAllocWorkspace {
    pub fn new(n_coefficients: usize, n_outputs: usize) -> Self {
        Self {
            xtwy: Array2::zeros((n_coefficients, n_outputs)),
            ywy: Array1::zeros(n_outputs),
            projected_rhs: Array2::zeros((n_coefficients, n_outputs)),
            projected_rhs_squared: Array2::zeros((n_coefficients, n_outputs)),
            scaled_projected_rhs: Array2::zeros((n_coefficients, n_outputs)),
        }
    }

}

#[derive(Clone, Copy, Debug)]
pub struct GaussianRemlNoAllocFit {
    pub lambda: f64,
    pub rho: f64,
    pub reml_score: f64,
    pub reml_grad_lambda: f64,
    pub reml_hess_lambda: f64,
    pub reml_grad_rho: f64,
    pub reml_hess_rho: f64,
    pub edf: f64,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlMultiBatchProblem<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView2<'a, f64>,
    pub weights: Option<ArrayView1<'a, f64>>,
    pub init_rho: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct GaussianRemlBlockOrthogonalResult {
    pub coefficients: Vec<Array2<f64>>,
    pub fitted: Array2<f64>,
    pub lambdas: Array1<f64>,
    pub log_lambdas: Array1<f64>,
    pub reml_score: f64,
    pub edf: Array1<f64>,
}

#[derive(Clone)]
struct GaussianRemlPrepared {
    cache: GaussianRemlEigenCache,
    ywy: Array1<f64>,
    projected_rhs_squared: Array2<f64>,
    projected_rhs: Array2<f64>,
    /// Number of rows with a strictly positive prior weight — the effective
    /// sample size that enters the REML residual degrees of freedom `ν`. Rows
    /// with weight `0` are excluded (see [`effective_observation_count`]).
    n_effective: usize,
    n_outputs: usize,
}

#[derive(Clone, Copy)]
struct ObjectiveEval {
    cost: f64,
    grad: f64,
    hess: f64,
    edf: f64,
    /// Forward-error bound on `cost`: the accumulated floating-point roundoff of
    /// the very additions and cancellations that produced it (#2729). Carried
    /// alongside the cost for the same reason `pairwise_mean_with_roundoff`
    /// carries one — a score compared against another score is only a decision
    /// above this magnitude; below it the comparison has no digits left.
    cost_roundoff: f64,
}

/// Unit roundoff `u = ½·eps`, the per-operation relative error bound every
/// forward-error accumulation in this file is denominated in.
const UNIT_ROUNDOFF: f64 = 0.5 * f64::EPSILON;

/// Standard `gamma_m = m·u / (1 − m·u)` forward-error growth factor for a
/// deterministic chain of `m` rounded operations. Returns infinity once `m·u`
/// reaches 1, where no finite bound exists.
fn roundoff_growth(operation_count: usize) -> f64 {
    let accumulated = operation_count as f64 * UNIT_ROUNDOFF;
    if accumulated < 1.0 {
        accumulated / (1.0 - accumulated)
    } else {
        f64::INFINITY
    }
}

/// A single Gaussian closed-form REML objective term, carrying its analytic
/// VALUE together with its analytic ρ-GRADIENT and ρ-HESSIAN.
///
/// Single source of truth: each term's value and its (already hand-derived,
/// closed-form) ρ-derivatives are returned from ONE function body, so a future
/// edit to the value formula cannot silently leave the derivatives stale.
/// Mirrors the `PenaltyLogdetDerivs`-returning-tuple pattern used by the
/// unified outer evaluator — the structural cure for the objective↔gradient
/// desync class (#752/#748/#808). The three contributions are accumulated
/// through [`ObjectiveEval`] at one site, so they cannot drift apart.
#[derive(Clone, Copy)]
struct TermDerivs {
    value: f64,
    grad: f64,
    hess: f64,
    /// Forward-error bound on `value`, accumulated from the SAME intermediates
    /// the value is built from (#2729). Single-sourced with the value for the
    /// same reason the derivatives are: a bound derived anywhere else is a
    /// guess about an expression nobody evaluated.
    roundoff: f64,
}

/// Boundary-stable kernels for one nonnegative affine mode
/// `t = exp(rho) * delta`.
///
/// Forming `t` first is numerically wrong at the finite rho boundaries: a
/// large, finite `log(t) = rho + log(delta)` can overflow even though all four
/// ratios below have finite limits.  Keeping the mode in log-space makes the
/// objective and both derivatives regular at both smoothing boundaries.
#[derive(Clone, Copy)]
struct ModalKernels {
    log_one_plus_t: f64,
    /// `t / (1 + t)`.
    u: f64,
    /// `1 / (1 + t)`.
    v: f64,
    /// `t / (1 + t)^2 = u * v`.
    w: f64,
    /// `t(1 - t) / (1 + t)^3 = u * v * (v - u)`.
    k: f64,
}

fn modal_kernels(rho: f64, delta: f64) -> ModalKernels {
    if delta == 0.0 {
        return ModalKernels {
            log_one_plus_t: 0.0,
            u: 0.0,
            v: 1.0,
            w: 0.0,
            k: 0.0,
        };
    }
    let log_t = rho + delta.ln();
    let (log_one_plus_t, u, v) = if log_t >= 0.0 {
        let reciprocal_t = (-log_t).exp();
        let v = reciprocal_t / (1.0 + reciprocal_t);
        (log_t + reciprocal_t.ln_1p(), 1.0 - v, v)
    } else {
        let t = log_t.exp();
        let u = t / (1.0 + t);
        (t.ln_1p(), u, 1.0 - u)
    };
    let w = u * v;
    ModalKernels {
        log_one_plus_t,
        u,
        v,
        w,
        k: w * (v - u),
    }
}

impl std::ops::AddAssign<TermDerivs> for ObjectiveEval {
    /// Fold a term's `(value, grad, hess)` triple into the running totals in
    /// lock-step, so value and derivative can never be added at separate sites.
    fn add_assign(&mut self, rhs: TermDerivs) {
        self.cost += rhs.value;
        self.grad += rhs.grad;
        self.hess += rhs.hess;
        // The term's own bound plus the rounding of THIS addition, priced on the
        // running total it just produced.
        self.cost_roundoff += rhs.roundoff + UNIT_ROUNDOFF * self.cost.abs();
    }
}

/// `½d·(log|H| − log|S|_+)` value with its analytic ρ-gradient/Hessian.
///
/// The penalty-eigenvalue sum produces all three quantities from the SAME
/// `t = λδ` intermediates in one pass, so the value (`log|1+t|`) and its
/// derivatives (`t/(1+t)`, `t/(1+t)²`) are single-sourced.
fn gaussian_reml_logdet_term(
    cache: &GaussianRemlEigenCache,
    rho: f64,
    n_outputs: f64,
) -> (TermDerivs, f64) {
    let mut logdet_h = cache.logdet_xtwx;
    let mut trace_h = 0.0;
    let mut trace_h_deriv = 0.0;
    let mut edf = 0.0;
    // #2729: magnitude of every term summed into the log-determinant difference,
    // accumulated in the SAME loop that sums the terms themselves. It is what the
    // forward-error bound below is denominated in — `log|H|` and `log|S|₊` are
    // individually large and differenced, so the difference's absolute error is
    // set by the SUMMANDS' magnitudes, not by the (possibly tiny) difference.
    let mut logdet_magnitude = cache.logdet_xtwx.abs();
    // ONE predicate: `δ > 0.0` below is applied to the CLASSIFIED value, so the
    // directions summed here are exactly the `penalty_rank` directions the
    // offset below counts (see [`PenaltyRangeSpectrum`], #2740).
    for delta in PenaltyRangeSpectrum::of(cache).iter() {
        let mode = modal_kernels(rho, delta);
        logdet_h += mode.log_one_plus_t;
        logdet_magnitude += mode.log_one_plus_t.abs();
        if delta > 0.0 {
            trace_h += mode.u;
            trace_h_deriv += mode.w;
        }
        edf += mode.v;
    }
    let logdet_s = cache.logdet_penalty_positive + (cache.penalty_rank as f64) * rho;
    logdet_magnitude += cache.logdet_penalty_positive.abs() + logdet_s.abs();
    let value = 0.5 * n_outputs * (logdet_h - logdet_s);
    // One `log1p` and one accumulation per penalty eigendirection, the two
    // additions forming `log|S|₊`, the difference, and the two outer multiplies.
    let operation_count = cache
        .penalty_eigenvalues
        .len()
        .saturating_mul(2)
        .saturating_add(5);
    let term = TermDerivs {
        value,
        grad: 0.5 * n_outputs * (trace_h - cache.penalty_rank as f64),
        hess: 0.5 * n_outputs * trace_h_deriv,
        roundoff: 0.5 * n_outputs * roundoff_growth(operation_count) * logdet_magnitude,
    };
    (term, edf)
}

/// Residual-deviance decomposition `dp_j(ρ) = r0_j + Σ_i c²_ij·u_i(ρ)`.
///
/// The profiled residual is defined as `dp_j(ρ) = ywy_j − Σ_i c²_ij·v_i(ρ)`.
/// Evaluated that way it is a DIFFERENCE of two quantities that become equal as
/// ρ → −∞ on a design that interpolates its response (`p = n`, no residual
/// degrees of freedom), so it loses every significant digit exactly where the
/// smoothing search needs it. At the saturated `24×24` tensor fixture of gam#2585
/// the small-λ end has `u ≈ 9e−17`, small enough that `v = 1 − u` rounds to
/// exactly `1.0` and the difference returns literally zero.
///
/// `ModalKernels` guarantees `u + v == 1` exactly — one is always built as
/// `1 − other` — so the identity
///
/// ```text
///   dp_j(ρ) = (ywy_j − Σ_i c²_ij) + Σ_i c²_ij·u_i(ρ) = r0_j + Σ_i c²_ij·u_i(ρ)
/// ```
///
/// is exact algebra, and the right-hand form is a SUM OF NON-NEGATIVES: `r0_j`
/// is the ρ-independent unpenalized residual deviance (`≥ 0`, exactly `0` for a
/// saturated design) and every `u_i ≥ 0`. The cancellation is confined to `r0_j`,
/// which no longer depends on ρ and therefore cannot differ between two cells of
/// the ρ search.
///
/// Shared by the evaluator, the domain check, the profiled dispersion and the
/// interval enclosure: a bound that encloses a different expression than the
/// evaluator computes is the objective↔enclosure desync this file exists to
/// prevent.
#[inline]
fn dispersion_residual_parts(
    cache: &GaussianRemlEigenCache,
    ywy: ArrayView1<'_, f64>,
    projected_rhs_squared: ArrayView2<'_, f64>,
    output: usize,
    rho: f64,
) -> DispersionResidualParts {
    let mut total_c2 = 0.0;
    let mut penalized_residual = 0.0;
    let mut dp_grad = 0.0;
    let mut dp_hess = 0.0;
    let spectrum = PenaltyRangeSpectrum::of(cache);
    for eig in 0..spectrum.len() {
        let c2 = projected_rhs_squared[[eig, output]];
        let mode = modal_kernels(rho, spectrum.get(eig));
        total_c2 += c2;
        penalized_residual += c2 * mode.u;
        dp_grad += c2 * mode.w;
        dp_hess += c2 * mode.k;
    }
    // `r0 ≥ 0` mathematically (it is the residual deviance of the unpenalized
    // weighted least-squares fit), so clamping at zero only removes roundoff
    // that has no sign information left in it.
    let unpenalized_residual = (ywy[output] - total_c2).max(0.0);
    DispersionResidualParts {
        unpenalized_residual,
        penalized_residual,
        dp_grad,
        dp_hess,
        total_c2,
    }
}

/// The `dp = r0 + Σ c²·u` decomposition of one output's profiled residual
/// deviance, plus the `Σ c²` whose cancellation against `ywy` produced `r0`.
///
/// `total_c2` is not an extra output for convenience: `r0` is a DIFFERENCE of
/// two same-signed accumulations, so the only honest scale for its absolute
/// error is `|ywy| + Σ c²` — the magnitudes that cancelled — and that scale is
/// unrecoverable once the difference has been taken (#2729).
#[derive(Clone, Copy)]
struct DispersionResidualParts {
    unpenalized_residual: f64,
    penalized_residual: f64,
    dp_grad: f64,
    dp_hess: f64,
    total_c2: f64,
}

/// Per-output dispersion-prior term `½ν·(1 + log(2π·dp/ν))` with its analytic
/// ρ-gradient/Hessian.
///
/// `dp`, `dp_grad`, `dp_hess` are computed from the SAME eigenvalue sum, then
/// the value `log(dp)` and its derivatives `dp_grad/dp`,
/// `dp_hess/dp − (dp_grad/dp)²` are returned together so they cannot desync.
fn gaussian_reml_dispersion_term(
    cache: &GaussianRemlEigenCache,
    ywy: ArrayView1<'_, f64>,
    projected_rhs_squared: ArrayView2<'_, f64>,
    output: usize,
    nu: f64,
    rho: f64,
) -> TermDerivs {
    let parts = dispersion_residual_parts(cache, ywy, projected_rhs_squared, output, rho);
    let dp = parts.unpenalized_residual + parts.penalized_residual;
    let value = 0.5 * nu * (1.0 + (2.0 * std::f64::consts::PI * dp / nu).ln());
    // #2729. `dp` is formed by cancelling `Σ c²` against `ywy` and then adding a
    // sum of non-negatives, so its ABSOLUTE error is set by the magnitudes that
    // cancelled, not by `dp` itself. Two multiplies and one accumulation per
    // eigendirection, the final subtraction, the clamp and one addition.
    let operation_count = cache
        .penalty_eigenvalues
        .len()
        .saturating_mul(3)
        .saturating_add(3);
    let dp_magnitude = ywy[output].abs() + parts.total_c2.abs() + parts.penalized_residual.abs();
    let dp_roundoff = roundoff_growth(operation_count) * dp_magnitude;
    // `d/d(dp) of ½ν·log(dp) = ½ν/dp`: the logarithm converts `dp`'s RELATIVE
    // error into the value's absolute error, which is why a deviance sitting at
    // its own cancellation floor leaves this term with no significant digits.
    // Plus the rounding of the log, the division and the two multiplies.
    TermDerivs {
        value,
        grad: 0.5 * nu * parts.dp_grad / dp,
        hess: 0.5 * nu * (parts.dp_hess / dp - (parts.dp_grad * parts.dp_grad) / (dp * dp)),
        roundoff: 0.5 * nu * (dp_roundoff / dp) + roundoff_growth(4) * value.abs(),
    }
}

pub fn gaussian_reml_closed_form(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_rho: Option<f64>,
) -> Result<GaussianRemlResult, EstimationError> {
    gaussian_reml_closed_form_with_nullspace_dim(x, y, penalty, None, weights, init_rho)
}

pub fn gaussian_reml_closed_form_with_nullspace_dim(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    weights: Option<ArrayView1<'_, f64>>,
    init_rho: Option<f64>,
) -> Result<GaussianRemlResult, EstimationError> {
    let y2 = y.insert_axis(Axis(1));
    let result = gaussian_reml_multi_closed_form_with_nullspace_dim(
        x,
        y2,
        penalty,
        nullspace_dim,
        weights,
        init_rho,
    )?;
    scalar_result_from_multi(result)
}

fn scalar_result_from_multi(
    result: GaussianRemlMultiResult,
) -> Result<GaussianRemlResult, EstimationError> {
    Ok(GaussianRemlResult {
        lambda: result.lambda,
        rho: result.rho,
        coefficients: result.coefficients.column(0).to_owned(),
        fitted: result.fitted.column(0).to_owned(),
        reml_score: result.reml_score,
        reml_grad_lambda: result.reml_grad_lambda,
        reml_hess_lambda: result.reml_hess_lambda,
        reml_grad_rho: result.reml_grad_rho,
        reml_hess_rho: result.reml_hess_rho,
        edf: result.edf,
        sigma2: result.sigma2[0],
        cache: result.cache,
    })
}

/// Point evaluation of the closed-form Gaussian REML objective at a FIXED
/// log-smoothing parameter, with no optimization. Exposes the same REML score,
/// effective df, σ², and posterior-mean coefficients the optimizer sees at that
/// `rho`, so callers can trace the REML score surface as a function of `rho`
/// (e.g. to audit λ-selection against a reference tool).
#[derive(Clone, Debug)]
pub struct GaussianRemlPointEval {
    pub rho: f64,
    pub lambda: f64,
    pub reml_score: f64,
    pub edf: f64,
    pub sigma2: f64,
    pub coefficients: Array1<f64>,
}

/// Successful finite-window certificate for the profiled Gaussian REML
/// ρ-objective.
///
/// `roots` contains one representative from every stationary bracket isolated
/// on `rho_window`; `root_brackets` records those location certificates and
/// `root_gradients` makes their numerical residuals directly auditable. The
/// selected ρ is the lowest evaluated representative or boundary. This route's
/// convergence claim is a *location* certificate, not a gradient one: it accepts
/// on bracket width in ρ (`root_location_resolution`), which bounds the returned
/// point's gradient by `h · width` through the mean value theorem and is a
/// strictly stronger statement than any residual threshold. A search cell whose
/// stationary structure remains ambiguous at
/// `root_location_resolution` is not represented by a flag in a successful
/// value: the search returns [`EstimationError::RemlDidNotConverge`] instead.
#[derive(Clone, Debug)]
pub struct GaussianRemlStationarySet {
    pub roots: Vec<f64>,
    pub root_brackets: Vec<[f64; 2]>,
    pub root_gradients: Vec<f64>,
    pub selected_rho: f64,
    pub endpoint_costs: [f64; 2],
    pub rho_window: [f64; 2],
    pub root_location_resolution: f64,
}

/// Enumerate the closed-form Gaussian REML stationary set at the given design,
/// exposing the [`GaussianRemlStationarySet`] certificate. Thin wrapper over the
/// shared enumeration used by the production optimizer — added beside
/// [`gaussian_reml_point_eval_at_rho`] rather than changing any existing public
/// signature.
pub fn gaussian_reml_stationary_set(
    x: ArrayView2<'_, f64>,
    y: ArrayView1<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    weights: Option<ArrayView1<'_, f64>>,
    init_rho: Option<f64>,
) -> Result<GaussianRemlStationarySet, EstimationError> {
    if init_rho.is_some_and(|rho| !rho.is_finite()) {
        crate::bail_invalid_estim!("Gaussian REML stationary search requires a finite rho hint");
    }
    let y2 = y.insert_axis(Axis(1));
    let prepared = prepare_gaussian_reml(x, y2.view(), penalty, nullspace_dim, weights, None)?;
    let endpoint_costs = [
        prepared.evaluate(RHO_LOWER).cost,
        prepared.evaluate(RHO_UPPER).cost,
    ];
    validate_reml_profile_residuals(
        &prepared.cache,
        prepared.ywy.view(),
        prepared.projected_rhs_squared.view(),
        RHO_LOWER,
    )?;
    if prepared.cache.penalty_rank == 0 {
        return Ok(GaussianRemlStationarySet {
            roots: Vec::new(),
            root_brackets: Vec::new(),
            root_gradients: Vec::new(),
            selected_rho: init_rho.unwrap_or(0.0).clamp(RHO_LOWER, RHO_UPPER),
            endpoint_costs,
            rho_window: [RHO_LOWER, RHO_UPPER],
            root_location_resolution: RHO_BRACKET_RESOLUTION,
        });
    }
    let eval = |rho: f64| prepared.evaluate(rho);
    let enclose = |a: f64, b: f64| {
        reml_deriv_enclosure(
            &prepared.cache,
            prepared.ywy.view(),
            prepared.projected_rhs_squared.view(),
            prepared.n_effective,
            prepared.n_outputs,
            a,
            b,
        )
    };
    let mut roots = Vec::new();
    let mut root_brackets = Vec::new();
    let mut root_gradients = Vec::new();
    let selection = {
        let mut observer = |root: StationaryRoot, e: &ObjectiveEval| {
            roots.push(root.rho);
            root_brackets.push(root.bracket);
            root_gradients.push(e.grad);
        };
        enumerate_and_select_rho(&eval, &enclose, init_rho, Some(&mut observer))?
    };
    Ok(GaussianRemlStationarySet {
        roots,
        root_brackets,
        root_gradients,
        selected_rho: selection.rho,
        endpoint_costs,
        rho_window: [RHO_LOWER, RHO_UPPER],
        root_location_resolution: RHO_BRACKET_RESOLUTION,
    })
}

pub fn gaussian_reml_multi_closed_form(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_rho: Option<f64>,
) -> Result<GaussianRemlMultiResult, EstimationError> {
    gaussian_reml_multi_closed_form_with_nullspace_dim(x, y, penalty, None, weights, init_rho)
}

/// Closed-form multi-response Gaussian REML with one SHARED dispersion across
/// all response columns.
///
/// This is the appropriate likelihood when the columns are coordinates of one
/// vector-valued observation rather than unrelated responses with independently
/// estimable noise scales.  The coefficient matrix and smoothing parameter are
/// still shared exactly as in [`gaussian_reml_multi_closed_form`], but the
/// profiled deviance is pooled before taking its logarithm:
///
/// `dp = sum_j dp_j`, `nu = d * (n_eff - nullity)`.
///
/// Pooling is essential for coordinate-chart races.  A chart made from a linear
/// projection of the response reconstructs those projection axes tautologically;
/// independently profiling each output dispersion lets one exact axis drive its
/// variance to zero and dominate evidence even when another ambient direction is
/// badly missed.  A shared ambient dispersion scores the reconstruction of the
/// vector as one object and cannot be gamed by that coordinate leakage.
pub fn gaussian_reml_multi_shared_dispersion_closed_form(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_rho: Option<f64>,
) -> Result<GaussianRemlMultiResult, EstimationError> {
    if y.ncols() == 0 {
        crate::bail_invalid_estim!(
            "shared-dispersion Gaussian REML requires at least one response column"
        );
    }
    let prepared = prepare_gaussian_reml(x, y, penalty, None, weights, None)?;
    let init_rho = init_rho
        .map(f64::exp)
        .map(validate_initial_lambda)
        .transpose()?
        .map(f64::ln);
    let d = prepared.n_outputs;
    let mut pooled_ywy = Array1::<f64>::zeros(1);
    pooled_ywy[0] = prepared.ywy.iter().copied().sum();
    let mut pooled_projected_rhs_squared =
        Array2::<f64>::zeros((prepared.cache.penalty_eigenvalues.len(), 1));
    for eig in 0..prepared.cache.penalty_eigenvalues.len() {
        pooled_projected_rhs_squared[[eig, 0]] = prepared
            .projected_rhs_squared
            .row(eig)
            .iter()
            .copied()
            .sum();
    }
    let per_output_nu = prepared.n_effective as f64 - prepared.cache.nullity as f64;
    let shared_nu = (d as f64) * per_output_nu;
    validate_reml_profile_residuals(
        &prepared.cache,
        pooled_ywy.view(),
        pooled_projected_rhs_squared.view(),
        RHO_LOWER,
    )?;
    let eval = |rho: f64| {
        evaluate_reml_profile(
            &prepared.cache,
            pooled_ywy.view(),
            pooled_projected_rhs_squared.view(),
            d,
            shared_nu,
            rho,
        )
    };
    let rho = if prepared.cache.penalty_rank == 0 {
        init_rho.unwrap_or(0.0).clamp(RHO_LOWER, RHO_UPPER)
    } else {
        let enclose = |a: f64, b: f64| {
            reml_deriv_enclosure_profile(
                &prepared.cache,
                pooled_ywy.view(),
                pooled_projected_rhs_squared.view(),
                d,
                shared_nu,
                a,
                b,
            )
        };
        enumerate_and_select_rho(eval, enclose, init_rho, None)?.rho
    };
    let objective = eval(rho);
    let lambda = gam_problem::checked_exp_log_strength(rho)
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let coefficients = prepared.coefficients(lambda);
    let fitted = dense_ab(x, coefficients.view());
    let mut fitted_quadratic = 0.0_f64;
    // Same classified spectrum the objective's `dp` uses — `σ̂²·ν` and `dp(ρ̂)`
    // are the same quantity computed two ways and must not read the spectrum
    // through two different range/null tests (#2740).
    let spectrum = PenaltyRangeSpectrum::of(&prepared.cache);
    for eig in 0..spectrum.len() {
        let denom = 1.0 + lambda * spectrum.get(eig);
        fitted_quadratic += pooled_projected_rhs_squared[[eig, 0]] / denom;
    }
    let shared_sigma2 = (pooled_ywy[0] - fitted_quadratic) / shared_nu;
    let (reml_grad_lambda, reml_hess_lambda) =
        rho_derivatives_to_lambda(lambda, objective.grad, objective.hess);
    Ok(GaussianRemlMultiResult {
        lambda,
        rho,
        coefficients,
        fitted,
        reml_score: objective.cost,
        reml_score_roundoff: Some(objective.cost_roundoff),
        reml_grad_lambda,
        reml_hess_lambda,
        reml_grad_rho: objective.grad,
        reml_hess_rho: objective.hess,
        edf: objective.edf,
        sigma2: Array1::from_elem(d, shared_sigma2),
        cache: prepared.cache,
    })
}

pub fn gaussian_reml_multi_closed_form_with_nullspace_dim(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    weights: Option<ArrayView1<'_, f64>>,
    init_rho: Option<f64>,
) -> Result<GaussianRemlMultiResult, EstimationError> {
    let init_lambda = init_rho.map(f64::exp);
    gaussian_reml_multi_closed_form_from_parts(
        x,
        y,
        penalty,
        nullspace_dim,
        weights,
        init_lambda,
        None,
    )
}

pub fn gaussian_reml_multi_closed_form_with_cache(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    eigen_cache: Option<&GaussianRemlEigenCache>,
) -> Result<GaussianRemlMultiResult, EstimationError> {
    gaussian_reml_multi_closed_form_from_parts(
        x,
        y,
        penalty,
        None,
        weights,
        init_lambda,
        eigen_cache,
    )
}

struct BlockOrthogonalEval {
    beta: Array2<f64>,
    logdet: f64,
    trace: f64,
    trace_pair: f64,
    fitted_energy: Array1<f64>,
    penalty_energy: Array1<f64>,
    curvature_energy: Array1<f64>,
    edf: f64,
}

fn block_penalty_rank_logdet(
    penalty: ArrayView2<'_, f64>,
) -> Result<(usize, f64), EstimationError> {
    let eigs = penalty
        .to_owned()
        .eigh(Side::Lower)
        .map_err(|_| EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        })?
        .0;
    let max_abs = eigs.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let tol = (EIGEN_REL_TOL * max_abs).max(1.0e-14);
    let mut rank = 0_usize;
    let mut logdet = 0.0;
    for eig in eigs.iter().copied() {
        if eig > tol {
            rank += 1;
            logdet += eig.ln();
        }
    }
    Ok((rank, logdet))
}

fn block_orthogonal_eval(
    gram: &Array2<f64>,
    rhs: &Array2<f64>,
    penalty: &Array2<f64>,
    rho: f64,
) -> Result<BlockOrthogonalEval, EstimationError> {
    let lambda = gam_problem::checked_exp_log_strength(rho)
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    validate_initial_lambda(lambda)?;
    let scaled_penalty = penalty * lambda;
    let hessian = canonicalize_penalty((gram + &scaled_penalty).view());
    let chol = gaussian_reml_cholesky_lower(hessian)?;
    let beta = solve_spd_from_lower_factor(&chol, rhs)?;
    let solved_penalty = solve_spd_from_lower_factor(&chol, &scaled_penalty)?;
    let logdet = 2.0 * chol.diag().iter().map(|value| value.ln()).sum::<f64>();
    let trace = (0..solved_penalty.nrows())
        .map(|i| solved_penalty[[i, i]])
        .sum::<f64>();
    let trace_pair =
        gam_linalg::utils::trace_of_product(solved_penalty.view(), solved_penalty.view());
    let fitted_energy = (rhs * &beta).sum_axis(Axis(0));
    let p_beta = scaled_penalty.dot(&beta);
    let penalty_energy = (&beta * &p_beta).sum_axis(Axis(0));
    let solved_p_beta = solve_spd_from_lower_factor(&chol, &p_beta)?;
    let curvature_energy = (&p_beta * &solved_p_beta).sum_axis(Axis(0));
    Ok(BlockOrthogonalEval {
        beta,
        logdet,
        trace,
        trace_pair,
        fitted_energy,
        penalty_energy,
        curvature_energy,
        edf: penalty.nrows() as f64 - trace,
    })
}

/// Block-orthogonal shared-scale REML objective VALUE together with its
/// analytic ρ-gradient and ρ-Hessian.
///
/// Single source of truth: the value `½d·logdet − ½·fit − ½d·rank·ρ` and its
/// ρ-derivatives are returned from ONE function body, so a future edit to the
/// objective cannot leave the Newton gradient/Hessian (previously written at a
/// physically separate site inside `solve_block_orthogonal_rho`) stale. This
/// closes a genuine `(value_here, gradient_there)` loose pair. Mirrors the
/// `PenaltyLogdetDerivs` single-source pattern; behavior is identical (the same
/// closed-form formulas, reorganized).
struct BlockOrthogonalScaleDerivs {
    value: f64,
    /// Forward roundoff bound on `value`, i.e. the smallest value difference
    /// this channel can still decide.
    ///
    /// `value` is a three-term sum whose terms individually reach `½·τ·⟨y,fit⟩`
    /// — a quantity of order `n·τ` — while its ρ-variation near the optimum is
    /// of order the score squared. A descent test on such a sum is meaningful
    /// only while the step's predicted decrease exceeds this bound; below it,
    /// `candidate_value < current_value` is decided by rounding rather than by
    /// descent. `solve_block_orthogonal_rho` uses this to hand the endgame to
    /// the certificate's own metric instead of walking on value noise.
    value_roundoff: f64,
    grad: f64,
    hess: f64,
}

fn block_orthogonal_scale_objective(
    eval: &BlockOrthogonalEval,
    rho: f64,
    scale_precision: ArrayView1<'_, f64>,
    rank: usize,
) -> BlockOrthogonalScaleDerivs {
    let d = scale_precision.len() as f64;
    let fit_term = scale_precision
        .iter()
        .zip(eval.fitted_energy.iter())
        .map(|(scale, energy)| scale * energy)
        .sum::<f64>();
    // VALUE: ½d·log|H| − ½ Σ_o w_o ⟨y_o, fit_o⟩ − ½d·rank·ρ.
    let logdet_term = 0.5 * d * eval.logdet;
    let rank_term = 0.5 * d * (rank as f64) * rho;
    let value = logdet_term - 0.5 * fit_term - rank_term;
    // Standard forward bound for the three-term sum: no summation order can
    // resolve a difference below the unit roundoff times the sum of the term
    // magnitudes.
    let value_roundoff =
        f64::EPSILON * (logdet_term.abs() + 0.5 * fit_term.abs() + rank_term.abs());
    // ρ-GRADIENT: d/dρ of the same scalar. The logdet term contributes
    // ½d·(tr(H⁻¹λS) − rank); the (data-independent-at-fixed-β envelope) fit term
    // contributes +½ Σ_o w_o βᵀ(λS)β. Both share `eval`'s cached energies.
    let grad = 0.5 * d * (eval.trace - rank as f64)
        + 0.5
            * scale_precision
                .iter()
                .zip(eval.penalty_energy.iter())
                .map(|(scale, energy)| scale * energy)
                .sum::<f64>();
    // ρ-HESSIAN: d²/dρ². Logdet term: ½d·(tr(H⁻¹λS) − tr((H⁻¹λS)²)); penalty
    // term: ½ Σ_o w_o (βᵀλSβ − 2 βᵀλS H⁻¹ λS β).
    let hess = 0.5 * d * (eval.trace - eval.trace_pair)
        + 0.5
            * scale_precision
                .iter()
                .zip(eval.penalty_energy.iter().zip(eval.curvature_energy.iter()))
                .map(|(scale, (energy, curvature))| scale * (energy - 2.0 * curvature))
                .sum::<f64>();
    BlockOrthogonalScaleDerivs {
        value,
        value_roundoff,
        grad,
        hess,
    }
}

/// One warm-started 1-D Newton polish of a single block's rho at fixed scale
/// precisions. `max_iter` is a per-pass WORK bound, not a convergence
/// selector: the caller (`gaussian_reml_blocks_orthogonal_shared_scale`)
/// re-enters this solve every outer pass and certifies the joint fit by the
/// analytic score residual, erroring typed if the certificate is never met —
/// so an iterate returned at this cap never silently becomes the estimator.
fn solve_block_orthogonal_rho(
    gram: &Array2<f64>,
    rhs: &Array2<f64>,
    penalty: &Array2<f64>,
    rho0: f64,
    scale_precision: ArrayView1<'_, f64>,
    rank: usize,
    max_iter: usize,
) -> Result<(f64, BlockOrthogonalEval), EstimationError> {
    let mut rho = rho0;
    let mut current = block_orthogonal_eval(gram, rhs, penalty, rho)?;
    for _ in 0..max_iter {
        // Value, ρ-gradient, and ρ-Hessian all come from the SINGLE
        // single-source objective evaluation — they cannot desync.
        let derivs = block_orthogonal_scale_objective(&current, rho, scale_precision, rank);
        let grad = derivs.grad;
        let hess = derivs.hess;
        if !(grad.is_finite() && hess.is_finite()) {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }
        if grad == 0.0 {
            break;
        }
        // Positive curvature gives the Newton direction. Else use the exact
        // negative-gradient direction, which is descending regardless of the
        // local curvature. A representability-terminated backtracking search
        // globalizes either direction; it has no arbitrary finite trial list,
        // step clamp, or line-search iteration budget.
        let direction = if hess > 0.0 { -grad / hess } else { -grad };
        if !direction.is_finite() || grad * direction >= 0.0 {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }
        let current_value = derivs.value;
        // Exact decrease the local quadratic model predicts for the FULL step:
        // `−g·p − ½·h·p²`. For the Newton direction that is `g²/(2h)`; for the
        // negative-gradient direction under nonpositive curvature it is at
        // least `g²`. The value channel can only adjudicate a step whose
        // predicted decrease exceeds the value's own forward roundoff — below
        // that, `candidate_value < current_value` reports rounding, and
        // accepting on it walks the iterate around on noise while |g| stands
        // still. This is not a tolerance: it is the point where the comparison
        // stops carrying information, computed from the value's own terms.
        let model_decrease = -grad * direction - 0.5 * hess * direction * direction;
        let value_decides = model_decrease.is_finite() && model_decrease > derivs.value_roundoff;
        let accepted = if value_decides {
            let mut step_scale = 1.0_f64;
            loop {
                let candidate_rho = rho + step_scale * direction;
                if candidate_rho == rho {
                    break None;
                }
                if let Ok(candidate_eval) = block_orthogonal_eval(gram, rhs, penalty, candidate_rho)
                {
                    let candidate_value = block_orthogonal_scale_objective(
                        &candidate_eval,
                        candidate_rho,
                        scale_precision,
                        rank,
                    )
                    .value;
                    if candidate_value.is_finite() && candidate_value < current_value {
                        break Some((candidate_rho, candidate_eval));
                    }
                }
                // Bisection is intrinsic to backtracking, not a tuned step-size
                // schedule. Floating-point representability above is the stopping
                // rule, so every feasible improving step remains reachable.
                step_scale *= 0.5;
            }
        } else {
            None
        };
        // Endgame: once the value channel cannot resolve the predicted decrease
        // (and whenever it simply refused every representable step), judge by
        // the certificate's own metric instead — accept a step that strictly
        // shrinks |g|. In a positive-curvature 1-D basin a gradient-magnitude
        // decrease is descent, and it stays measurable down to ulp(g) rather
        // than ulp(V). This is the only channel that reaches the score
        // tolerance the fit is certified against: on an `n`-row fit the value's
        // roundoff already exceeds `g²/(2h)` at `|g| ≈ sqrt(2h·ulp(V))`, which
        // is orders of magnitude ABOVE that tolerance.
        let accepted = accepted.or_else(|| {
            if hess <= 0.0 {
                return None;
            }
            let mut step_scale = 1.0_f64;
            loop {
                let candidate_rho = rho + step_scale * direction;
                if candidate_rho == rho {
                    break None;
                }
                if let Ok(candidate_eval) = block_orthogonal_eval(gram, rhs, penalty, candidate_rho)
                {
                    let candidate = block_orthogonal_scale_objective(
                        &candidate_eval,
                        candidate_rho,
                        scale_precision,
                        rank,
                    );
                    if candidate.grad.is_finite() && candidate.grad.abs() < grad.abs() {
                        break Some((candidate_rho, candidate_eval));
                    }
                }
                step_scale *= 0.5;
            }
        });
        let Some((next_rho, next_eval)) = accepted else {
            break;
        };
        rho = next_rho;
        current = next_eval;
    }
    Ok((rho, current))
}

fn block_orthogonal_conditional_scale(
    evals: &[BlockOrthogonalEval],
    ywy: ArrayView1<'_, f64>,
    nu: f64,
) -> Result<Array1<f64>, EstimationError> {
    let mut explained = Array1::<f64>::zeros(ywy.len());
    for eval in evals {
        explained += &eval.fitted_energy;
    }
    let q = &ywy - &explained;
    if q.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }
    let scale = q.mapv(|value| nu / value);
    if scale
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }
    Ok(scale)
}

/// Verify the defining contract of the decomposed block objective.  For every
/// pair of design columns this checks `x_a' W x_b = 0` against the standard
/// `gamma_m` forward-error bound for the two multiplications and two
/// accumulations performed per row.  The tolerance therefore scales with the
/// actual product magnitudes and row count; it is not a data-scale knob.
fn validate_weighted_block_orthogonality(
    designs: &[Array2<f64>],
    weight: ArrayView1<'_, f64>,
) -> Result<(), EstimationError> {
    let unit_roundoff = 0.5 * f64::EPSILON;
    let operation_count = weight.len().saturating_mul(4);
    let accumulated = operation_count as f64 * unit_roundoff;
    if accumulated >= 1.0 {
        crate::bail_invalid_estim!(
            "block-orthogonality verification has no finite floating-point error bound for {} rows",
            weight.len()
        );
    }
    let gamma = accumulated / (1.0 - accumulated);
    for left_block in 0..designs.len() {
        for right_block in (left_block + 1)..designs.len() {
            let left = &designs[left_block];
            let right = &designs[right_block];
            for left_col in 0..left.ncols() {
                for right_col in 0..right.ncols() {
                    let mut cross_product = 0.0_f64;
                    let mut magnitude_sum = 0.0_f64;
                    for row in 0..weight.len() {
                        let term = weight[row] * left[[row, left_col]] * right[[row, right_col]];
                        cross_product += term;
                        magnitude_sum += term.abs();
                    }
                    let roundoff = gamma * magnitude_sum;
                    if !cross_product.is_finite()
                        || !roundoff.is_finite()
                        || cross_product.abs() > roundoff
                    {
                        crate::bail_invalid_estim!(
                            "block-orthogonal Gaussian REML requires X[{left_block}]' W X[{right_block}] = 0, but columns ({left_col}, {right_col}) have weighted cross-product {cross_product:.6e} beyond the arithmetic bound {roundoff:.3e}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct BlockOrthogonalProfileCurvature {
    min_eigenvalue: f64,
    roundoff: f64,
}

/// Analytic rho Hessian after profiling out the exact conditional scale.
///
/// With `tau_o = nu / q_o` and `e_bo = beta_bo' lambda_b S_b beta_bo`,
/// eliminating the exact conditional scale block contributes the dense Schur
/// correction
///
/// `H_profile[b,c] = 1[b=c] H_fixed_scale[b,b]
///                    - (1/(2 nu)) sum_o tau_o^2 e_bo e_co`.
///
fn block_orthogonal_profile_hessian(
    evals: &[BlockOrthogonalEval],
    rhos: ArrayView1<'_, f64>,
    scale_precision: ArrayView1<'_, f64>,
    ranks: &[usize],
    nu: f64,
) -> Result<Array2<f64>, EstimationError> {
    let blocks = evals.len();
    let mut hessian = Array2::<f64>::zeros((blocks, blocks));
    for block in 0..blocks {
        hessian[[block, block]] = block_orthogonal_scale_objective(
            &evals[block],
            rhos[block],
            scale_precision.view(),
            ranks[block],
        )
        .hess;
    }
    for left in 0..blocks {
        for right in 0..=left {
            let correction = evals[left]
                .penalty_energy
                .iter()
                .zip(evals[right].penalty_energy.iter())
                .zip(scale_precision.iter())
                .map(|((&left_energy, &right_energy), &scale)| {
                    0.5 * scale * scale * left_energy * right_energy / nu
                })
                .sum::<f64>();
            hessian[[left, right]] -= correction;
            if left != right {
                hessian[[right, left]] -= correction;
            }
        }
    }
    if hessian.iter().any(|value| !value.is_finite()) {
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }
    Ok(hessian)
}

/// Eigendecomposition of the analytic profiled Hessian.
///
/// One decomposition per outer pass serves both consumers: the curvature
/// certificate (a first-order score can vanish at a REML maximum or saddle, so
/// nonnegative curvature up to eigensolver roundoff is required before a fit is
/// minted) and the profiled Newton direction that drives the score to that
/// certificate.
struct BlockOrthogonalProfileSpectrum {
    curvature: BlockOrthogonalProfileCurvature,
    eigenvalues: Array1<f64>,
    eigenvectors: Array2<f64>,
}

fn block_orthogonal_profile_spectrum(
    hessian: &Array2<f64>,
) -> Result<BlockOrthogonalProfileSpectrum, EstimationError> {
    let blocks = hessian.nrows();
    let (eigenvalues, eigenvectors) =
        hessian
            .clone()
            .eigh(Side::Lower)
            .map_err(|_| EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            })?;
    let min_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let spectral_scale = eigenvalues
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let roundoff = f64::EPSILON * blocks.max(1) as f64 * spectral_scale.max(f64::MIN_POSITIVE);
    Ok(BlockOrthogonalProfileSpectrum {
        curvature: BlockOrthogonalProfileCurvature {
            min_eigenvalue,
            roundoff,
        },
        eigenvalues,
        eigenvectors,
    })
}

impl BlockOrthogonalProfileSpectrum {
    /// Exact Newton direction `−H⁻¹g` of the scale-profiled objective, or
    /// `None` when the profiled Hessian is not positive definite (there the
    /// alternation, which is descent under any curvature, owns the pass).
    fn newton_direction(&self, gradient: ArrayView1<'_, f64>) -> Option<Array1<f64>> {
        if self.curvature.min_eigenvalue.is_nan() || self.curvature.min_eigenvalue <= 0.0 {
            return None;
        }
        let projected = self.eigenvectors.t().dot(&gradient);
        let scaled = Array1::from_iter(
            projected
                .iter()
                .zip(self.eigenvalues.iter())
                .map(|(component, eigenvalue)| -component / eigenvalue),
        );
        let direction = self.eigenvectors.dot(&scaled);
        direction
            .iter()
            .all(|value| value.is_finite())
            .then_some(direction)
    }
}

/// The scale-profiled REML objective VALUE at `rhos`, with the forward roundoff
/// bound of its own term sum.
///
/// This is the function whose gradient the score certificate measures (the
/// exact conditional scale `τ_o = ν/q_o` makes the scale block of the joint
/// score vanish, so the envelope theorem identifies the profiled ρ-derivative
/// with the cached partial ρ-gradient) and whose Hessian
/// `block_orthogonal_profile_hessian` returns. Line searches on the profiled
/// objective compare against `roundoff` for the same reason
/// `BlockOrthogonalScaleDerivs::value_roundoff` exists.
struct BlockOrthogonalProfileValue {
    value: f64,
    roundoff: f64,
}

fn block_orthogonal_profile_value(
    evals: &[BlockOrthogonalEval],
    rhos: ArrayView1<'_, f64>,
    ranks: &[usize],
    ywy: ArrayView1<'_, f64>,
    nu: f64,
    d: usize,
) -> Option<BlockOrthogonalProfileValue> {
    let mut explained = Array1::<f64>::zeros(ywy.len());
    for eval in evals {
        explained += &eval.fitted_energy;
    }
    let mut q = ywy.to_owned();
    q -= &explained;
    if q.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return None;
    }
    let determinant_term = 0.5
        * d as f64
        * evals
            .iter()
            .enumerate()
            .map(|(block, eval)| eval.logdet - ranks[block] as f64 * rhos[block])
            .sum::<f64>();
    let deviance_term = 0.5 * nu * q.iter().map(|value| value.ln()).sum::<f64>();
    let value = determinant_term + deviance_term;
    if !value.is_finite() {
        return None;
    }
    Some(BlockOrthogonalProfileValue {
        value,
        roundoff: f64::EPSILON * (determinant_term.abs() + deviance_term.abs()),
    })
}

/// Everything the certificate and the profiled Newton step read at one
/// `(rhos, evals, scale_precision)` state. Assembled once per evaluation so the
/// certificate's score and the direction that chases it can never come from
/// different points.
struct BlockOrthogonalStateMeasurement {
    score_residual: f64,
    gradient: Array1<f64>,
    spectrum: BlockOrthogonalProfileSpectrum,
}

fn measure_block_orthogonal_state(
    evals: &[BlockOrthogonalEval],
    rhos: ArrayView1<'_, f64>,
    scale_precision: ArrayView1<'_, f64>,
    ranks: &[usize],
    nu: f64,
    d: usize,
) -> Result<BlockOrthogonalStateMeasurement, EstimationError> {
    let mut gradient = Array1::<f64>::zeros(evals.len());
    let mut score_residual = 0.0_f64;
    for (block, eval) in evals.iter().enumerate() {
        let derivs =
            block_orthogonal_scale_objective(eval, rhos[block], scale_precision, ranks[block]);
        let residual = derivs.grad.abs() / ((d as f64) * (ranks[block].max(1) as f64));
        if !residual.is_finite() {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }
        gradient[block] = derivs.grad;
        score_residual = score_residual.max(residual);
    }
    let hessian = block_orthogonal_profile_hessian(evals, rhos, scale_precision, ranks, nu)?;
    Ok(BlockOrthogonalStateMeasurement {
        score_residual,
        gradient,
        spectrum: block_orthogonal_profile_spectrum(&hessian)?,
    })
}

pub fn gaussian_reml_blocks_orthogonal_shared_scale(
    designs: &[Array2<f64>],
    penalties: &[Array2<f64>],
    y: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_rhos: Option<&[f64]>,
) -> Result<GaussianRemlBlockOrthogonalResult, EstimationError> {
    gaussian_reml_blocks_orthogonal_shared_scale_with_controls(
        designs,
        penalties,
        y,
        weights,
        init_rhos,
        BlockOrthogonalControls::default(),
    )
}

fn gaussian_reml_blocks_orthogonal_shared_scale_with_controls(
    designs: &[Array2<f64>],
    penalties: &[Array2<f64>],
    y: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_rhos: Option<&[f64]>,
    controls: BlockOrthogonalControls,
) -> Result<GaussianRemlBlockOrthogonalResult, EstimationError> {
    if designs.is_empty() {
        crate::bail_invalid_estim!("block-orthogonal Gaussian REML requires at least one block");
    }
    if designs.len() != penalties.len() {
        crate::bail_invalid_estim!(
            "block-orthogonal Gaussian REML block mismatch: {} designs, {} penalties",
            designs.len(),
            penalties.len()
        );
    }
    let n = y.nrows();
    let d = y.ncols();
    if d == 0 {
        crate::bail_invalid_estim!("block-orthogonal Gaussian REML requires at least one output");
    }
    if y.iter().any(|value| !value.is_finite()) {
        crate::bail_invalid_estim!("block-orthogonal Gaussian REML response must be finite");
    }
    let weight = gaussian_reml_weights(n, weights)?;
    if let Some(rhos) = init_rhos {
        if rhos.len() != designs.len() {
            crate::bail_invalid_estim!(
                "block-orthogonal Gaussian REML init_rhos length mismatch: expected {}, got {}",
                designs.len(),
                rhos.len()
            );
        }
        if rhos.iter().any(|value| !value.is_finite()) {
            crate::bail_invalid_estim!("block-orthogonal Gaussian REML init_rhos must be finite");
        }
    }

    let mut ywy = Array1::<f64>::zeros(d);
    for row in 0..n {
        for output in 0..d {
            ywy[output] += weight[row] * y[[row, output]] * y[[row, output]];
        }
    }
    let mut grams = Vec::with_capacity(designs.len());
    let mut rhs_blocks = Vec::with_capacity(designs.len());
    let mut penalties_owned = Vec::with_capacity(penalties.len());
    let mut ranks = Vec::with_capacity(penalties.len());
    let mut penalty_logdets = Vec::with_capacity(penalties.len());
    let mut nullity_total = 0_usize;
    for (block, (design, penalty)) in designs.iter().zip(penalties.iter()).enumerate() {
        let penalty_owned = canonicalize_penalty(penalty.view());
        validate_gaussian_reml_design(design.view(), penalty_owned.view(), Some(weight.view()))?;
        if design.nrows() != n {
            crate::bail_invalid_estim!(
                "block-orthogonal Gaussian REML designs[{block}] has {} rows, expected {n}",
                design.nrows()
            );
        }
        let gram = dense_xt_diag_x(design.view(), weight.view());
        let rhs = dense_xt_diag_y(design.view(), weight.view(), y);
        let (rank, logdet) = block_penalty_rank_logdet(penalty_owned.view())?;
        nullity_total += penalty_owned.nrows().saturating_sub(rank);
        grams.push(canonicalize_penalty(gram.view()));
        rhs_blocks.push(rhs);
        penalties_owned.push(penalty_owned);
        ranks.push(rank);
        penalty_logdets.push(logdet);
    }
    validate_weighted_block_orthogonality(designs, weight.view())?;
    let n_effective = effective_observation_count(weight.view());
    if n_effective <= nullity_total {
        crate::bail_invalid_estim!(
            "block-orthogonal Gaussian REML requires more positive-weight rows than the total penalty nullity; got n_effective={n_effective}, nullity={nullity_total}"
        );
    }
    let nu = (n_effective - nullity_total) as f64;
    let mut rhos = match init_rhos {
        Some(values) => Array1::from_vec(values.to_vec()),
        None => Array1::zeros(designs.len()),
    };
    // A rho checkpoint is sufficient to resume exactly because scale is a
    // closed-form conditional block. Reconstruct that block from the supplied
    // rhos before any new rho update instead of discarding it and restarting
    // from the response-only scale.
    let mut evals = (0..designs.len())
        .map(|block| {
            block_orthogonal_eval(
                &grams[block],
                &rhs_blocks[block],
                &penalties_owned[block],
                rhos[block],
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    let mut scale_precision = block_orthogonal_conditional_scale(&evals, ywy.view(), nu)?;
    // Convergence is certified by the analytic score of the joint REML
    // objective, never by the iteration cap (SPEC rule 20). Each outer pass
    // (a) solves every block's 1-D rho Newton at the current scale precisions
    // and (b) applies the EXACT conditional-optimum scale update
    // `scale_o = nu / q_o`, so at the post-update point the scale block of the
    // joint score vanishes identically and — by the envelope theorem — the
    // profiled objective's total rho-derivative equals the partial
    // rho-gradient there. That gradient is available exactly from the cached
    // block evaluations because `block_orthogonal_eval` depends only on rho,
    // not on the scale precisions. First-order certification is therefore
    // `max_b |dV/drho_b| / (d * max(1, rank_b)) <= BLOCK_ORTHOGONAL_SCORE_TOL`
    // (the normalizer is the score's natural magnitude: every gradient term is
    // a sum of `d * rank`-order quantities, making the test relative). The
    // analytic Schur-profiled rho Hessian must additionally be PSD within its
    // dimension-scaled eigensolver roundoff; score-zero maxima and saddles are
    // not converged estimators.
    //
    // The alternation alone is block Gauss-Seidel on `(rho, scale)`: it is
    // globally descending but only LINEARLY convergent, at the spectral radius
    // of the Schur coupling the profiled Hessian already carries. That rate is
    // data-dependent and can be arbitrarily close to one, so a pass budget can
    // never bound how close it gets to the score certificate. Each pass
    // therefore ends with an exact Newton step on the SCALE-PROFILED objective,
    // whose gradient is the certificate's own score and whose Hessian is the
    // matrix assembled for the curvature certificate — no extra derivative
    // work. The alternation keeps the pass wherever that Hessian is not
    // positive definite (it descends under any curvature); the Newton step owns
    // the endgame, where it converges quadratically and lands the score orders
    // of magnitude below the tolerance instead of within a factor of two of it.
    //
    // Exhausting the pass budget without the certificate is a typed error
    // carrying the rho checkpoint, resumable through `init_rhos`.
    let mut converged = false;
    let mut cycle_detected = false;
    let mut outer_passes = 0usize;
    let mut last_score_residual = f64::INFINITY;
    let mut last_min_profile_curvature = f64::NEG_INFINITY;
    let mut last_profile_curvature_roundoff = 0.0_f64;
    let mut last_scale_step = f64::INFINITY;
    let mut recent_states: [Option<(Array1<f64>, Array1<f64>)>; 2] = [None, None];
    while outer_passes < controls.max_outer_passes {
        outer_passes += 1;
        let scale_at_pass_start = scale_precision.clone();
        evals.clear();
        for block in 0..designs.len() {
            let (rho, eval) = solve_block_orthogonal_rho(
                &grams[block],
                &rhs_blocks[block],
                &penalties_owned[block],
                rhos[block],
                scale_precision.view(),
                ranks[block],
                controls.block_updates_per_pass,
            )?;
            rhos[block] = rho;
            evals.push(eval);
        }
        scale_precision = block_orthogonal_conditional_scale(&evals, ywy.view(), nu)?;
        let mut measured = measure_block_orthogonal_state(
            &evals,
            rhos.view(),
            scale_precision.view(),
            &ranks,
            nu,
            d,
        )?;
        // Profiled Newton step. Skipped once the alternation already certified,
        // so a converged pass costs exactly what it did before.
        let alternation_certified = measured.score_residual <= controls.score_tol
            && measured.spectrum.curvature.min_eigenvalue >= -measured.spectrum.curvature.roundoff;
        let newton_step = if alternation_certified {
            None
        } else {
            measured
                .spectrum
                .newton_direction(measured.gradient.view())
                .zip(block_orthogonal_profile_value(
                    &evals,
                    rhos.view(),
                    &ranks,
                    ywy.view(),
                    nu,
                    d,
                ))
        };
        if let Some((direction, current_profile)) = newton_step {
            // Decrease the quadratic model predicts for the full step,
            // `−g'p − ½p'Hp = ½g'H⁻¹g`. The profiled value can only adjudicate
            // a step larger than its own forward roundoff; below that the
            // certificate's own score residual is the honest metric, exactly as
            // in the one-dimensional block polish.
            let model_decrease = -0.5 * measured.gradient.dot(&direction);
            let value_decides =
                model_decrease.is_finite() && model_decrease > current_profile.roundoff;
            let mut step_scale = 1.0_f64;
            let accepted = loop {
                let candidate_rhos = &rhos + &direction.mapv(|value| step_scale * value);
                if candidate_rhos == rhos {
                    break None;
                }
                let candidate = (0..designs.len())
                    .map(|block| {
                        block_orthogonal_eval(
                            &grams[block],
                            &rhs_blocks[block],
                            &penalties_owned[block],
                            candidate_rhos[block],
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .ok()
                    .and_then(|candidate_evals| {
                        let candidate_scale =
                            block_orthogonal_conditional_scale(&candidate_evals, ywy.view(), nu)
                                .ok()?;
                        let candidate_measured = measure_block_orthogonal_state(
                            &candidate_evals,
                            candidate_rhos.view(),
                            candidate_scale.view(),
                            &ranks,
                            nu,
                            d,
                        )
                        .ok()?;
                        let improves = if value_decides {
                            block_orthogonal_profile_value(
                                &candidate_evals,
                                candidate_rhos.view(),
                                &ranks,
                                ywy.view(),
                                nu,
                                d,
                            )
                            .is_some_and(|profile| profile.value < current_profile.value)
                        } else {
                            candidate_measured.score_residual < measured.score_residual
                        };
                        improves.then_some((candidate_evals, candidate_scale, candidate_measured))
                    });
                if let Some((candidate_evals, candidate_scale, candidate_measured)) = candidate {
                    break Some((
                        candidate_rhos,
                        candidate_evals,
                        candidate_scale,
                        candidate_measured,
                    ));
                }
                // Backtracking bisection, stopped by floating-point
                // representability rather than a trial budget.
                step_scale *= 0.5;
            };
            if let Some((next_rhos, next_evals, next_scale, next_measured)) = accepted {
                rhos = next_rhos;
                evals = next_evals;
                scale_precision = next_scale;
                measured = next_measured;
            }
        }
        last_scale_step = scale_precision
            .iter()
            .zip(scale_at_pass_start.iter())
            .map(|(next, old)| (next.ln() - old.ln()).abs())
            .fold(0.0_f64, f64::max);
        last_score_residual = measured.score_residual;
        last_min_profile_curvature = measured.spectrum.curvature.min_eigenvalue;
        last_profile_curvature_roundoff = measured.spectrum.curvature.roundoff;
        if last_score_residual <= controls.score_tol
            && last_min_profile_curvature >= -last_profile_curvature_roundoff
        {
            converged = true;
            break;
        }
        // Cycle guard: one outer pass is a pure function of the state
        // `(rhos, scale_precision)`. Revisiting a state from one or two passes
        // ago (bitwise) means the alternation is in a floating-point limit
        // cycle that can never certify, so stop escalating immediately instead
        // of burning the remaining budget on the same orbit.
        let state = (rhos.clone(), scale_precision.clone());
        if recent_states
            .iter()
            .flatten()
            .any(|prev| prev.0 == state.0 && prev.1 == state.1)
        {
            cycle_detected = true;
            break;
        }
        recent_states[1] = recent_states[0].take();
        recent_states[0] = Some(state);
    }
    if !converged {
        return Err(EstimationError::BlockOrthogonalRemlDidNotConverge {
            iterations: outer_passes,
            max_score_residual: last_score_residual,
            score_tol: controls.score_tol,
            min_profile_curvature: last_min_profile_curvature,
            profile_curvature_roundoff: last_profile_curvature_roundoff,
            last_scale_step,
            cycle_detected,
            rho_checkpoint: rhos.to_vec(),
        });
    }

    let coefficients = evals
        .iter()
        .map(|eval| eval.beta.clone())
        .collect::<Vec<_>>();
    let mut fitted = Array2::<f64>::zeros((n, d));
    for (design, coef) in designs.iter().zip(coefficients.iter()) {
        fitted += &fast_ab(&design.view(), &coef.view());
    }
    let mut explained = Array1::<f64>::zeros(d);
    for eval in evals.iter() {
        explained += &eval.fitted_energy;
    }
    let q = &ywy - &explained;
    if q.iter().any(|value| !value.is_finite() || *value <= 0.0) {
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }
    let lambdas = Array1::from_vec(gam_problem::checked_exp_log_strengths(
        rhos.iter().copied(),
    )?);
    let edf = Array1::from_iter(evals.iter().map(|eval| eval.edf));
    let logdet_term = evals
        .iter()
        .enumerate()
        .map(|(block, eval)| {
            eval.logdet - penalty_logdets[block] - (ranks[block] as f64) * rhos[block]
        })
        .sum::<f64>();
    let scale_term = q
        .iter()
        .map(|value| nu * (1.0 + (2.0 * std::f64::consts::PI * value / nu).ln()))
        .sum::<f64>();
    Ok(GaussianRemlBlockOrthogonalResult {
        coefficients,
        fitted,
        lambdas,
        log_lambdas: rhos,
        reml_score: 0.5 * (d as f64) * logdet_term + 0.5 * scale_term,
        edf,
    })
}

/// Exact envelope derivative of shared-dispersion Gaussian REML with respect
/// to its symmetric penalty matrix at a converged inner fit.
///
/// The coefficient matrix and log smoothing strength are stationary in
/// [`gaussian_reml_multi_shared_dispersion_closed_form`], so their implicit
/// derivatives vanish from the outer derivative. What remains is the explicit
/// penalty derivative of the restricted determinant and the single pooled
/// deviance. This is the authority for continuously optimized reference-metric
/// parameters: a metric supplies `dS/dtheta`, and the outer derivative is the
/// Frobenius contraction `<dV/dS, dS/dtheta>`.
pub fn gaussian_reml_multi_shared_dispersion_penalty_gradient_from_fit(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    fit: &GaussianRemlMultiResult,
) -> Result<Array2<f64>, EstimationError> {
    validate_gaussian_reml_forward_fit(x, y, penalty, weights, fit)?;
    let n = x.nrows();
    let p = x.ncols();
    let d = y.ncols();
    if d == 0 {
        crate::bail_invalid_estim!(
            "shared-dispersion REML penalty gradient requires at least one response column"
        );
    }
    let weight = gaussian_reml_weights(n, weights)?;
    let n_effective = effective_observation_count(weight.view());
    let per_output_nu = n_effective.checked_sub(fit.cache.nullity).ok_or_else(|| {
        EstimationError::InvalidInput(
            "shared-dispersion REML penalty gradient has non-positive residual degrees of freedom"
                .to_string(),
        )
    })?;
    if per_output_nu == 0 {
        crate::bail_invalid_estim!(
            "shared-dispersion REML penalty gradient requires positive residual degrees of freedom"
        );
    }
    let shared_nu = (d as f64) * (per_output_nu as f64);
    // Use the deviance represented by the forward fit itself.  Reconstructing
    // the mathematically equivalent quantity as RSS + lambda * beta' S beta
    // follows a different floating-point path from the modal subtraction used
    // by `gaussian_reml_multi_shared_dispersion_closed_form`.  On a nearly
    // interpolating chart the two paths lose different low bits, making this
    // gradient disagree with value probes even though both formulas are exact
    // over the reals.  A nested metric optimizer then follows the mismatched
    // derivative until every Armijo step is rejected.  The shared forward fit
    // stores its single profiled dispersion in every output slot, so recover
    // the authoritative pooled deviance from that state instead.
    let shared_sigma2 = fit.sigma2[0];
    if fit
        .sigma2
        .iter()
        .any(|sigma2| sigma2.to_bits() != shared_sigma2.to_bits())
    {
        crate::bail_invalid_estim!(
            "shared-dispersion REML penalty gradient requires one shared forward dispersion"
        );
    }
    let pooled_deviance = shared_sigma2 * shared_nu;
    // DENOMINATE THE BAR IN WHAT PRODUCED THE QUANTITY.  The forward fit forms
    // this pooled deviance by cancellation: `pooled_ywy - sum_k c_k^2/(1 +
    // lambda*delta_k)` (see `gaussian_reml_multi_shared_dispersion_closed_form`),
    // a difference of two accumulations that are individually bounded in
    // magnitude by the pooled weighted response energy.  On the nearly
    // interpolating chart the comment above describes, that difference is the
    // roundoff residue of its own summation, and a bare `> 0.0` accepts it: a
    // positive value at the arithmetic floor is indistinguishable from a real
    // deviance to that predicate, and it then enters `deviance_scale` as a
    // DENOMINATOR, so the accepted debris is amplified by `1/floor` into every
    // entry the nested metric optimizer follows.
    //
    // The floor below is the standard `gamma_m` forward-error bound for the way
    // the quantity is actually formed, exactly as
    // `validate_weighted_block_orthogonality` bounds its own cancellation: two
    // multiplications and one accumulation per weighted response entry
    // (`n*d` of them), one reciprocal-scale multiply and one accumulation per
    // penalty eigendirection (`p` of them), and the final subtraction.  It is
    // derived from the machine epsilon, the problem dimensions and the measured
    // response energy; there is no tolerance to tune.  Below it, the pooled
    // deviance carries no significant digit, `1/pooled_deviance` has no
    // meaning, and there is no finite limit to substitute -- `deviance_scale`
    // diverges as the chart approaches interpolation -- so the honest branch is
    // a named refusal rather than a fabricated derivative.
    let mut pooled_response_energy = 0.0_f64;
    for output in 0..d {
        for row in 0..n {
            let value = y[[row, output]];
            pooled_response_energy += weight[row] * value * value;
        }
    }
    let unit_roundoff = 0.5 * f64::EPSILON;
    let operation_count = n
        .saturating_mul(d)
        .saturating_mul(3)
        .saturating_add(p.saturating_mul(2))
        .saturating_add(1);
    let accumulated = operation_count as f64 * unit_roundoff;
    if accumulated >= 1.0 {
        crate::bail_invalid_estim!(
            "shared-dispersion REML penalty gradient has no finite floating-point error bound for {n} rows, {d} responses and {p} coefficients"
        );
    }
    let deviance_roundoff = (accumulated / (1.0 - accumulated)) * pooled_response_energy;
    if !(pooled_deviance.is_finite()
        && deviance_roundoff.is_finite()
        && pooled_deviance > deviance_roundoff)
    {
        crate::bail_invalid_estim!(
            "shared-dispersion REML penalty gradient requires a forward deviance resolved above the roundoff of its own formation; the chart is interpolating to arithmetic precision: pooled deviance {pooled_deviance:.6e} does not exceed the forward bound {deviance_roundoff:.6e} on the cancellation that produced it from pooled response energy {pooled_response_energy:.6e}"
        );
    }

    let inverse_hessian = gaussian_reml_inverse_hessian_from_cache(&fit.cache, fit.lambda)?;
    let penalty_pseudoinverse = gaussian_reml_penalty_pseudoinverse_from_cache(&fit.cache)?;
    let mut gradient = Array2::<f64>::zeros((p, p));
    for row in 0..p {
        for col in 0..p {
            gradient[[row, col]] = 0.5
                * (d as f64)
                * (fit.lambda * inverse_hessian[[col, row]] - penalty_pseudoinverse[[col, row]]);
        }
    }
    let deviance_scale = 0.5 * shared_nu * fit.lambda / pooled_deviance;
    for output in 0..d {
        add_rank_one_penalty_vjp(
            deviance_scale,
            fit.coefficients.column(output),
            &mut gradient,
        );
    }
    for row in 0..p {
        for col in (row + 1)..p {
            let mean = 0.5 * (gradient[[row, col]] + gradient[[col, row]]);
            gradient[[row, col]] = mean;
            gradient[[col, row]] = mean;
        }
    }
    if gradient.iter().any(|value| !value.is_finite()) {
        crate::bail_invalid_estim!(
            "shared-dispersion REML penalty gradient produced a non-finite value"
        );
    }
    Ok(gradient)
}

fn gaussian_reml_multi_closed_form_from_parts(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    eigen_cache: Option<&GaussianRemlEigenCache>,
) -> Result<GaussianRemlMultiResult, EstimationError> {
    let prepared = prepare_gaussian_reml(x, y, penalty, nullspace_dim, weights, eigen_cache)?;
    let init_rho = init_lambda
        .map(validate_initial_lambda)
        .transpose()?
        .map(f64::ln);
    let rho = optimize_rho(&prepared, init_rho)?;
    let eval = prepared.evaluate(rho);
    let lambda = gam_problem::checked_exp_log_strength(rho)
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let coefficients = prepared.coefficients(lambda);
    let fitted = dense_ab(x, coefficients.view());
    let sigma2 = prepared.sigma2(rho);
    let (reml_grad_lambda, reml_hess_lambda) =
        rho_derivatives_to_lambda(lambda, eval.grad, eval.hess);
    Ok(GaussianRemlMultiResult {
        lambda,
        rho,
        coefficients,
        fitted,
        reml_score: eval.cost,
        reml_score_roundoff: Some(eval.cost_roundoff),
        reml_grad_lambda,
        reml_hess_lambda,
        reml_grad_rho: eval.grad,
        reml_hess_rho: eval.hess,
        edf: eval.edf,
        sigma2,
        cache: prepared.cache,
    })
}

pub fn gaussian_reml_free_b_score(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    coefficients: ArrayView2<'_, f64>,
    log_lambda: f64,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<GaussianRemlFreeBScore, EstimationError> {
    let lambda = gam_problem::checked_exp_log_strength(log_lambda)
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let penalty_owned = canonicalize_penalty(penalty);
    let penalty = penalty_owned.view();
    let n = x.nrows();
    let p = x.ncols();
    let d = y.ncols();
    validate_gaussian_reml_design(x, penalty, weights)?;
    if y.nrows() != n {
        crate::bail_invalid_estim!(
            "Gaussian REML row mismatch: X has {n} rows but Y has {}",
            y.nrows()
        );
    }
    if coefficients.dim() != (p, d) {
        crate::bail_invalid_estim!(
            "Gaussian REML coefficient shape mismatch: expected {p}x{d}, got {}x{}",
            coefficients.nrows(),
            coefficients.ncols()
        );
    }
    if y.iter().chain(coefficients.iter()).any(|v| !v.is_finite()) {
        crate::bail_invalid_estim!("Gaussian REML inputs must be finite");
    }

    let weight = gaussian_reml_weights(n, weights)?;
    let n_effective = effective_observation_count(weight.view());
    let cache =
        build_gaussian_reml_eigen_cache_with_nullspace_dim(x, penalty, None, Some(weight.view()))?;
    if n_effective <= cache.nullity {
        crate::bail_invalid_estim!(
            "Gaussian REML requires more positive-weight rows than the nullspace dimension; got n_effective={n_effective}, nullity={}",
            cache.nullity
        );
    }
    let nu = n_effective as f64 - cache.nullity as f64;
    let fitted = dense_ab(x, coefficients);
    let residual = y.to_owned() - &fitted;
    let xtw_residual = dense_xt_diag_y(x, weight.view(), residual.view());
    let s_beta = dense_ab(penalty, coefficients);

    let mut logdet_h = cache.logdet_xtwx;
    let mut trace_h = 0.0;
    let mut edf = 0.0;
    // ONE predicate, as in `gaussian_reml_logdet_term` (#2740): the directions
    // summed into `trace_h` are the `penalty_rank` directions subtracted from it.
    for delta in PenaltyRangeSpectrum::of(&cache).iter() {
        let t = lambda * delta;
        logdet_h += (1.0 + t).ln();
        if delta > 0.0 {
            trace_h += t / (1.0 + t);
        }
        edf += 1.0 / (1.0 + t);
    }
    let logdet_s = cache.logdet_penalty_positive + (cache.penalty_rank as f64) * log_lambda;
    let mut reml_score = 0.5 * (d as f64) * (logdet_h - logdet_s);
    let mut grad_log_lambda = 0.5 * (d as f64) * (trace_h - cache.penalty_rank as f64);
    let mut grad_coefficients = Array2::<f64>::zeros((p, d));
    let inverse_hessian = {
        let xtwx = dense_xt_diag_x(x, weight.view());
        let mut hessian = xtwx;
        hessian += &(penalty.to_owned() * lambda);
        hessian
            .cholesky(Side::Lower)
            .map_err(EstimationError::LinearSystemSolveFailed)?
            .solve_mat(&Array2::<f64>::eye(p))
    };
    let penalty_pinv = gaussian_reml_penalty_pseudoinverse_from_cache(&cache)?;
    let mut grad_penalty = Array2::<f64>::zeros((p, p));
    for row in 0..p {
        for col in 0..p {
            grad_penalty[[row, col]] += 0.5
                * (d as f64)
                * (lambda * inverse_hessian[[col, row]] - penalty_pinv[[col, row]]);
        }
    }
    let mut sigma2 = Array1::<f64>::zeros(d);

    for output in 0..d {
        let mut weighted_rss = 0.0;
        for row in 0..n {
            let r = residual[[row, output]];
            weighted_rss += weight[row] * r * r;
        }
        let beta_col = coefficients.column(output);
        let s_beta_col = s_beta.column(output);
        let penalty_quadratic = beta_col.dot(&s_beta_col);
        let dp = (weighted_rss + lambda * penalty_quadratic).max(MIN_DEVIANCE);
        sigma2[output] = dp / nu;
        reml_score += 0.5 * nu * (1.0 + (2.0 * std::f64::consts::PI * dp / nu).ln());
        grad_log_lambda += 0.5 * nu * lambda * penalty_quadratic / dp;
        let scale = nu / dp;
        for coeff in 0..p {
            grad_coefficients[[coeff, output]] =
                scale * (-xtw_residual[[coeff, output]] + lambda * s_beta[[coeff, output]]);
        }
        add_rank_one_penalty_vjp(0.5 * scale * lambda, beta_col, &mut grad_penalty);
    }
    for i in 0..p {
        for j in (i + 1)..p {
            let avg = 0.5 * (grad_penalty[[i, j]] + grad_penalty[[j, i]]);
            grad_penalty[[i, j]] = avg;
            grad_penalty[[j, i]] = avg;
        }
    }

    Ok(GaussianRemlFreeBScore {
        reml_score,
        grad_coefficients,
        grad_penalty,
        grad_log_lambda,
        fitted,
        sigma2,
        edf,
    })
}

pub fn gaussian_reml_multi_closed_form_backward(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    init_lambda: Option<f64>,
    upstream_lambda: f64,
    upstream_coefficients: Option<ArrayView2<'_, f64>>,
    upstream_fitted: Option<ArrayView2<'_, f64>>,
    upstream_reml_score: f64,
    upstream_edf: f64,
) -> Result<GaussianRemlBackwardResult, EstimationError> {
    let fit =
        gaussian_reml_multi_closed_form_with_cache(x, y, penalty, weights, init_lambda, None)?;
    gaussian_reml_multi_closed_form_backward_from_fit(
        x,
        y,
        penalty,
        weights,
        &fit,
        upstream_lambda,
        upstream_coefficients,
        upstream_fitted,
        upstream_reml_score,
        upstream_edf,
    )
}

pub fn gaussian_reml_multi_closed_form_backward_from_fit(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    fit: &GaussianRemlMultiResult,
    upstream_lambda: f64,
    upstream_coefficients: Option<ArrayView2<'_, f64>>,
    upstream_fitted: Option<ArrayView2<'_, f64>>,
    upstream_reml_score: f64,
    upstream_edf: f64,
) -> Result<GaussianRemlBackwardResult, EstimationError> {
    validate_gaussian_reml_backward_upstreams(
        x,
        y,
        penalty,
        upstream_lambda,
        upstream_coefficients,
        upstream_fitted,
        upstream_reml_score,
        upstream_edf,
    )?;
    validate_gaussian_reml_forward_fit(x, y, penalty, weights, fit)?;
    let lambda = fit.lambda;
    let n = x.nrows();
    let p = x.ncols();
    let d = y.ncols();
    // The implicit-function channel dλ̂/d(inputs) = −V_ρθ/V_ρρ is the derivative
    // of an INTERIOR stationary root only. Two selections break its premise:
    //  * ρ̂ railed at a box endpoint (±RHO bound): the selection is locally the
    //    constant projection, so dλ̂/d(inputs) = 0 exactly — applying the
    //    interior formula there emits enormous wrong gradients;
    //  * unusable ρ-curvature (flat or rank-zero penalty): λ̂ is not identified.
    // Neither invalidates the FIXED-ρ explicit VJPs — coefficients/fitted still
    // depend on X, y, W at the selected λ — so only the λ̂-root channel is
    // suppressed below. (The old gate zeroed the WHOLE backward here, silently
    // dropping real coefficient gradients on unpenalized/flat-penalty fits.)
    let rho_hat = lambda.ln();
    let rho_at_bound =
        (rho_hat - RHO_UPPER).abs() <= 1.0e-9 || (rho_hat - RHO_LOWER).abs() <= 1.0e-9;
    let implicit_rho_usable =
        fit.reml_hess_rho.is_finite() && fit.reml_hess_rho.abs() > 1.0e-14 && !rho_at_bound;
    let weight = gaussian_reml_weights(n, weights)?;
    let inverse_hessian = match gaussian_reml_inverse_hessian_from_cache(&fit.cache, lambda) {
        Ok(inv) => inv,
        Err(EstimationError::ModelIsIllConditioned { condition_number }) => {
            warn_ill_conditioned_backward_once(p, d, condition_number);
            return Ok(zero_backward_result(n, p, d));
        }
        Err(err) => return Err(err),
    };
    gaussian_reml_multi_closed_form_backward_from_fit_with_inverse_hessian_impl(
        x,
        y,
        penalty,
        weight,
        fit,
        inverse_hessian,
        upstream_lambda,
        upstream_coefficients,
        upstream_fitted,
        upstream_reml_score,
        upstream_edf,
        implicit_rho_usable,
        n,
        p,
        d,
    )
}

fn gaussian_reml_multi_closed_form_backward_from_fit_with_inverse_hessian_impl(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weight: Array1<f64>,
    fit: &GaussianRemlMultiResult,
    inverse_hessian: Array2<f64>,
    upstream_lambda: f64,
    upstream_coefficients: Option<ArrayView2<'_, f64>>,
    upstream_fitted: Option<ArrayView2<'_, f64>>,
    upstream_reml_score: f64,
    upstream_edf: f64,
    implicit_rho_usable: bool,
    n: usize,
    p: usize,
    d: usize,
) -> Result<GaussianRemlBackwardResult, EstimationError> {
    // Backward sees the same symmetric S the forward used. Canonicalize on
    // entry so an asymmetric input (e.g. a single-entry gradcheck perturbation
    // around a symmetric base) cannot leak into the per-helper VJPs.
    let penalty_owned = canonicalize_penalty(penalty);
    let penalty = penalty_owned.view();
    let lambda = fit.lambda;
    let beta = &fit.coefficients;
    let residual = y.to_owned() - &fit.fitted;
    // Match the forward's REML residual DoF: zero prior-weight rows are excluded
    // from the effective sample size (see `effective_observation_count`), so the
    // adjoint of `ν` uses the same count the forward used.
    let nu = effective_observation_count(weight.view()) as f64 - fit.cache.nullity as f64;

    let mut grad_x = Array2::<f64>::zeros((n, p));
    let mut grad_y = Array2::<f64>::zeros((n, d));
    let mut grad_penalty = Array2::<f64>::zeros((p, p));
    let mut grad_weights = Array1::<f64>::zeros(n);

    let mut upstream_beta = Array2::<f64>::zeros((p, d));
    if let Some(upstream_coefficients) = upstream_coefficients {
        upstream_beta += &upstream_coefficients;
    }
    if let Some(upstream_fitted) = upstream_fitted {
        upstream_beta += &dense_atb(x, upstream_fitted);
        grad_x += &dense_ab(upstream_fitted, beta.t());
    }

    let mut lambda_adjoint = upstream_lambda;
    if upstream_beta.iter().any(|value| *value != 0.0) {
        // A downstream loss that explicitly uses beta_hat or fitted = X beta_hat
        // cannot use the REML envelope shortcut.  Route those seeds through
        // the fixed-rho KKT adjoint M u = upstream_beta, then differentiate
        // X, y, weights, and S through the ridge solve.
        add_ridge_profile_vjp_with_lambda_grad(
            1.0,
            x,
            y,
            penalty,
            &weight,
            lambda,
            &inverse_hessian,
            beta,
            upstream_beta.view(),
            &mut grad_x,
            &mut grad_y,
            &mut grad_penalty,
            &mut grad_weights,
            &mut lambda_adjoint,
        );
    }

    if upstream_reml_score != 0.0 {
        add_reml_score_vjp(
            upstream_reml_score,
            x,
            &weight,
            &inverse_hessian,
            beta,
            &residual,
            &fit.sigma2,
            nu,
            lambda,
            &fit.cache,
            &mut grad_x,
            &mut grad_y,
            &mut grad_penalty,
            &mut grad_weights,
        )?;
        lambda_adjoint += upstream_reml_score * fit.reml_grad_lambda;
    }

    if upstream_edf != 0.0 {
        lambda_adjoint += add_edf_vjp(
            upstream_edf,
            x,
            penalty,
            &weight,
            lambda,
            &inverse_hessian,
            &mut grad_x,
            &mut grad_penalty,
            &mut grad_weights,
        );
    }

    if lambda_adjoint != 0.0 && implicit_rho_usable {
        let root_scale = -lambda_adjoint * lambda / fit.reml_hess_rho;
        add_reml_rho_gradient_vjp(
            root_scale,
            x,
            y,
            penalty,
            &weight,
            lambda,
            &inverse_hessian,
            beta,
            &residual,
            &fit.sigma2,
            nu,
            &mut grad_x,
            &mut grad_y,
            &mut grad_penalty,
            &mut grad_weights,
        );
    }

    // The forward consumes `S` only through the canonicalization
    // `S_canon = 0.5 (S + Sᵀ)`. By the chain rule, the gradient w.r.t. an
    // input `S_input` is `0.5 (G + Gᵀ)` where `G = ∂L/∂S_canon` is what the
    // per-helper VJPs accumulate. Symmetrize the full matrix here so a
    // single-entry perturbation `δS = ε E_{i,j}` (asymmetric, as
    // `torch.autograd.gradcheck` produces) sees the gradient component
    // `0.5 (G[i,j] + G[j,i])` it expects from FD — no caller-side
    // bookkeeping required.
    let p = grad_penalty.nrows();
    for i in 0..p {
        for j in (i + 1)..p {
            let avg = 0.5 * (grad_penalty[[i, j]] + grad_penalty[[j, i]]);
            grad_penalty[[i, j]] = avg;
            grad_penalty[[j, i]] = avg;
        }
    }
    Ok(GaussianRemlBackwardResult {
        grad_x,
        grad_y,
        grad_penalty,
        grad_weights,
    })
}

pub fn gaussian_reml_multi_closed_form_backward_batch<'a>(
    problems: &[GaussianRemlMultiBackwardProblem<'a>],
    penalty: ArrayView2<'a, f64>,
) -> Vec<Result<GaussianRemlBackwardResult, EstimationError>> {
    let inverse_hessians = batched_inverse_hessians_from_caches(problems);
    let results: Vec<Result<GaussianRemlBackwardResult, EstimationError>> = problems
        .par_iter()
        .zip(inverse_hessians.into_par_iter())
        .map(|(problem, inverse_hessian_result)| {
            validate_gaussian_reml_backward_upstreams(
                problem.x.view(),
                problem.y.view(),
                penalty,
                problem.grad_lambda,
                problem.grad_coefficients.as_ref().map(|g| g.view()),
                problem.grad_fitted.as_ref().map(|g| g.view()),
                problem.grad_reml_score,
                problem.grad_edf,
            )?;
            validate_gaussian_reml_forward_fit(
                problem.x.view(),
                problem.y.view(),
                penalty,
                problem.weights.as_ref().map(|w| w.view()),
                problem.fit,
            )?;
            let n = problem.x.nrows();
            let p = problem.x.ncols();
            let d = problem.y.ncols();
            if !(problem.fit.reml_hess_rho.is_finite() && problem.fit.reml_hess_rho.abs() > 1.0e-14)
            {
                // Graceful degradation — see `gaussian_reml_multi_closed_form_backward_from_fit`.
                warn_ill_conditioned_backward_once(p, d, f64::INFINITY);
                return Ok(zero_backward_result(n, p, d));
            }
            let weight = gaussian_reml_weights(n, problem.weights.as_ref().map(|w| w.view()))?;
            let inverse_hessian = match inverse_hessian_result {
                Ok(inv) => inv,
                Err(EstimationError::ModelIsIllConditioned { condition_number }) => {
                    warn_ill_conditioned_backward_once(p, d, condition_number);
                    return Ok(zero_backward_result(n, p, d));
                }
                Err(err) => return Err(err),
            };
            // Same selection-validity rule as the single-problem entry above:
            // the implicit λ̂-root channel is usable only for an INTERIOR
            // stationary root with usable ρ-curvature (a ρ̂ railed at a box
            // endpoint is locally the constant projection — its channel is 0).
            let rho_hat = problem.fit.lambda.ln();
            let rho_at_bound =
                (rho_hat - RHO_UPPER).abs() <= 1.0e-9 || (rho_hat - RHO_LOWER).abs() <= 1.0e-9;
            let implicit_rho_usable = problem.fit.reml_hess_rho.is_finite()
                && problem.fit.reml_hess_rho.abs() > 1.0e-14
                && !rho_at_bound;
            gaussian_reml_multi_closed_form_backward_from_fit_with_inverse_hessian_impl(
                problem.x.view(),
                problem.y.view(),
                penalty,
                weight,
                problem.fit,
                inverse_hessian,
                problem.grad_lambda,
                problem.grad_coefficients.as_ref().map(|g| g.view()),
                problem.grad_fitted.as_ref().map(|g| g.view()),
                problem.grad_reml_score,
                problem.grad_edf,
                implicit_rho_usable,
                n,
                p,
                d,
            )
        })
        .collect();
    results
}

fn rho_derivatives_to_lambda(lambda: f64, grad_rho: f64, hess_rho: f64) -> (f64, f64) {
    (grad_rho / lambda, (hess_rho - grad_rho) / (lambda * lambda))
}

fn validate_gaussian_reml_backward_upstreams(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    upstream_lambda: f64,
    upstream_coefficients: Option<ArrayView2<'_, f64>>,
    upstream_fitted: Option<ArrayView2<'_, f64>>,
    upstream_reml_score: f64,
    upstream_edf: f64,
) -> Result<(), EstimationError> {
    if !(upstream_lambda.is_finite() && upstream_reml_score.is_finite() && upstream_edf.is_finite())
    {
        crate::bail_invalid_estim!("Gaussian REML backward upstream scalars must be finite");
    }
    if let Some(upstream_coefficients) = upstream_coefficients {
        if upstream_coefficients.dim() != (x.ncols(), y.ncols()) {
            crate::bail_invalid_estim!(
                "Gaussian REML backward coefficient upstream shape mismatch: expected {}x{}, got {}x{}",
                x.ncols(),
                y.ncols(),
                upstream_coefficients.nrows(),
                upstream_coefficients.ncols()
            );
        }
        if upstream_coefficients.iter().any(|value| !value.is_finite()) {
            crate::bail_invalid_estim!(
                "Gaussian REML backward coefficient upstream must be finite"
            );
        }
    }
    if let Some(upstream_fitted) = upstream_fitted {
        if upstream_fitted.dim() != y.dim() {
            crate::bail_invalid_estim!(
                "Gaussian REML backward fitted upstream shape mismatch: expected {}x{}, got {}x{}",
                y.nrows(),
                y.ncols(),
                upstream_fitted.nrows(),
                upstream_fitted.ncols()
            );
        }
        if upstream_fitted.iter().any(|value| !value.is_finite()) {
            crate::bail_invalid_estim!("Gaussian REML backward fitted upstream must be finite");
        }
    }
    validate_gaussian_reml_design(x, penalty, None)?;
    Ok(())
}

fn validate_gaussian_reml_forward_fit(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    fit: &GaussianRemlMultiResult,
) -> Result<(), EstimationError> {
    // Fingerprint the canonicalized penalty: caches are keyed on the
    // symmetric average, and the caller may hand us a raw input (e.g. a
    // single-entry-perturbed matrix produced by ``torch.autograd.gradcheck``).
    let penalty_owned = canonicalize_penalty(penalty);
    let penalty = penalty_owned.view();
    let n = x.nrows();
    let p = x.ncols();
    let d = y.ncols();
    validate_gaussian_reml_design(x, penalty, weights)?;
    validate_gaussian_reml_eigen_cache(&fit.cache, p)?;
    if y.nrows() != n
        || fit.coefficients.dim() != (p, d)
        || fit.fitted.dim() != (n, d)
        || fit.sigma2.len() != d
    {
        crate::bail_invalid_estim!(
            "Gaussian REML backward forward-state shape mismatch: expected coefficients=({p},{d}), fitted=({n},{d}), sigma2={d}"
        );
    }
    if !(fit.lambda.is_finite()
        && fit.lambda > 0.0
        && fit.rho.is_finite()
        && fit.reml_score.is_finite()
        && fit.reml_hess_rho.is_finite()
        && fit.edf.is_finite())
        || fit.coefficients.iter().any(|value| !value.is_finite())
        || fit.fitted.iter().any(|value| !value.is_finite())
        || fit.sigma2.iter().any(|value| !value.is_finite())
    {
        crate::bail_invalid_estim!("Gaussian REML backward forward state must be finite");
    }
    let penalty_fingerprint = matrix_fingerprint(penalty);
    if fit.cache.penalty_fingerprint != penalty_fingerprint {
        crate::bail_invalid_estim!("Gaussian REML backward forward-state penalty mismatch");
    }
    let weight = gaussian_reml_weights(n, weights)?;
    let xtwx = dense_xt_diag_x(x, weight.view());
    if fit.cache.xtwx_fingerprint != matrix_fingerprint(xtwx.view()) {
        crate::bail_invalid_estim!("Gaussian REML backward forward-state X'WX mismatch");
    }
    Ok(())
}

fn gaussian_reml_inverse_hessian_from_cache(
    cache: &GaussianRemlEigenCache,
    lambda: f64,
) -> Result<Array2<f64>, EstimationError> {
    if !(lambda.is_finite() && lambda > 0.0) {
        crate::bail_invalid_estim!(
            "Gaussian REML lambda must be finite and positive; got {lambda}"
        );
    }
    let p = cache.penalty_eigenvalues.len();
    let spectrum = PenaltyRangeSpectrum::of(cache);
    let mut scaled_basis = cache.coefficient_basis.clone();
    for eig in 0..p {
        // `H = XᵀWX + λS` must be assembled from the same `S` the objective
        // scores; a direction the range predicate calls null carries no `λδ`
        // here either (#2740).
        let scale = 1.0 / (1.0 + lambda * spectrum.get(eig));
        for row in 0..p {
            scaled_basis[[row, eig]] *= scale;
        }
    }
    let inverse = dense_ab(scaled_basis.view(), cache.coefficient_basis.t());
    if inverse.iter().any(|value| !value.is_finite()) {
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }
    Ok(inverse)
}

fn batched_inverse_hessians_from_caches(
    problems: &[GaussianRemlMultiBackwardProblem<'_>],
) -> Vec<Result<Array2<f64>, EstimationError>> {
    if problems.is_empty() {
        return Vec::new();
    }
    let p = problems[0].fit.cache.coefficient_basis.nrows();
    let uniform = p > 0
        && problems.iter().all(|problem| {
            let cache = &problem.fit.cache;
            cache.coefficient_basis.dim() == (p, p) && cache.penalty_eigenvalues.len() == p
        });
    if uniform && problems.len() > 1 {
        let mut scaled_basis = Array3::<f64>::zeros((problems.len(), p, p));
        let mut basis = Array3::<f64>::zeros((problems.len(), p, p));
        let mut valid = true;
        for (idx, problem) in problems.iter().enumerate() {
            let lambda = problem.fit.lambda;
            if !(lambda.is_finite() && lambda > 0.0) {
                valid = false;
                break;
            }
            let cache = &problem.fit.cache;
            let spectrum = PenaltyRangeSpectrum::of(cache);
            basis
                .slice_mut(s![idx, .., ..])
                .assign(&cache.coefficient_basis);
            for eig in 0..p {
                let scale = 1.0 / (1.0 + lambda * spectrum.get(eig));
                for row in 0..p {
                    scaled_basis[[idx, row, eig]] = cache.coefficient_basis[[row, eig]] * scale;
                }
            }
        }
        if valid
            && let Some(inverses) =
                gam_gpu::try_fast_abt_strided_batched(scaled_basis.view(), basis.view())
        {
            return inverses
                .axis_iter(Axis(0))
                .map(|inverse| Ok(inverse.to_owned()))
                .collect();
        }
    }
    problems
        .iter()
        .map(|problem| {
            gaussian_reml_inverse_hessian_from_cache(&problem.fit.cache, problem.fit.lambda)
        })
        .collect()
}

/// Side-effects of the ridge-profile VJP that are independent of λ.
///
/// Computes the KKT adjoint `m = M^{-1} u` for `u = upstream_beta` and accumulates
/// the partials w.r.t. `X`, `y`, `S`, and `w` into the provided gradient buffers.
/// Returns `m` so callers that also need `∂L/∂λ` can fold in the λ-adjoint dot
/// product `−scale · ⟨m, S β⟩` without recomputing the adjoint solve.
fn ridge_profile_vjp_data_partials(
    scale: f64,
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    lambda: f64,
    inverse_hessian: &Array2<f64>,
    beta: &Array2<f64>,
    upstream_beta: ArrayView2<'_, f64>,
    grad_x: &mut Array2<f64>,
    grad_y: &mut Array2<f64>,
    grad_penalty: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
) -> Array2<f64> {
    let m = dense_ab(inverse_hessian.view(), upstream_beta);
    let c = dense_ab(m.view(), beta.t());
    let c_sym = &c + &c.t();
    let ymt = dense_ab(y, m.t());
    let xcs = dense_ab(x, c_sym.view());
    for i in 0..x.nrows() {
        let wi = weights[i] * scale;
        for k in 0..x.ncols() {
            grad_x[[i, k]] += wi * (ymt[[i, k]] - xcs[[i, k]]);
        }
    }

    let xm = dense_ab(x, m.view());
    for i in 0..x.nrows() {
        let wi = weights[i] * scale;
        for j in 0..y.ncols() {
            grad_y[[i, j]] += wi * xm[[i, j]];
        }
    }

    let xc = dense_ab(x, c.view());
    for i in 0..x.nrows() {
        let mut from_b = 0.0;
        for j in 0..y.ncols() {
            from_b += y[[i, j]] * xm[[i, j]];
        }
        let mut from_a = 0.0;
        for k in 0..x.ncols() {
            from_a += x[[i, k]] * xc[[i, k]];
        }
        grad_weights[i] += scale * (from_b - from_a);
    }

    for row in 0..penalty.nrows() {
        for col in 0..penalty.ncols() {
            let mut value = 0.0;
            for output in 0..beta.ncols() {
                value += m[[row, output]] * beta[[col, output]];
            }
            grad_penalty[[row, col]] -= scale * lambda * value;
        }
    }
    m
}

/// Ridge-profile VJP for callers that also need `∂L/∂λ`.
///
/// Accumulates the data/penalty/weight partials and adds the implicit-function
/// λ-adjoint contribution `−scale · ⟨M^{-1} u, S β⟩` into `lambda_adjoint_out`.
fn add_ridge_profile_vjp_with_lambda_grad(
    scale: f64,
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    lambda: f64,
    inverse_hessian: &Array2<f64>,
    beta: &Array2<f64>,
    upstream_beta: ArrayView2<'_, f64>,
    grad_x: &mut Array2<f64>,
    grad_y: &mut Array2<f64>,
    grad_penalty: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
    lambda_adjoint_out: &mut f64,
) {
    let m = ridge_profile_vjp_data_partials(
        scale,
        x,
        y,
        penalty,
        weights,
        lambda,
        inverse_hessian,
        beta,
        upstream_beta,
        grad_x,
        grad_y,
        grad_penalty,
        grad_weights,
    );
    let penalty_beta = dense_ab(penalty, beta.view());
    let dot = m
        .iter()
        .zip(penalty_beta.iter())
        .map(|(left, right)| left * right)
        .sum::<f64>();
    *lambda_adjoint_out += -scale * dot;
}

/// Ridge-profile VJP for callers that hold λ fixed (e.g. the implicit-root
/// partial inside `add_reml_rho_gradient_vjp`). The λ-adjoint dot product is
/// skipped entirely — it would be unused work in this branch.
fn add_ridge_profile_vjp_fixed_lambda(
    scale: f64,
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    lambda: f64,
    inverse_hessian: &Array2<f64>,
    beta: &Array2<f64>,
    upstream_beta: ArrayView2<'_, f64>,
    grad_x: &mut Array2<f64>,
    grad_y: &mut Array2<f64>,
    grad_penalty: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
) {
    ridge_profile_vjp_data_partials(
        scale,
        x,
        y,
        penalty,
        weights,
        lambda,
        inverse_hessian,
        beta,
        upstream_beta,
        grad_x,
        grad_y,
        grad_penalty,
        grad_weights,
    );
}

fn add_reml_score_vjp(
    scale: f64,
    x: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    inverse_hessian: &Array2<f64>,
    beta: &Array2<f64>,
    residual: &Array2<f64>,
    sigma2: &Array1<f64>,
    nu: f64,
    lambda: f64,
    cache: &GaussianRemlEigenCache,
    grad_x: &mut Array2<f64>,
    grad_y: &mut Array2<f64>,
    grad_penalty: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
) -> Result<(), EstimationError> {
    let d = beta.ncols() as f64;
    let xp = dense_ab(x, inverse_hessian.view());
    let penalty_pinv = gaussian_reml_penalty_pseudoinverse_from_cache(cache)?;
    for row in 0..grad_penalty.nrows() {
        for col in 0..grad_penalty.ncols() {
            grad_penalty[[row, col]] +=
                scale * 0.5 * d * (lambda * inverse_hessian[[col, row]] - penalty_pinv[[col, row]]);
        }
    }
    for i in 0..x.nrows() {
        let wi = weights[i] * scale * d;
        for k in 0..x.ncols() {
            grad_x[[i, k]] += wi * xp[[i, k]];
        }
        let mut leverage = 0.0;
        for k in 0..x.ncols() {
            leverage += x[[i, k]] * xp[[i, k]];
        }
        grad_weights[i] += scale * 0.5 * d * leverage;
    }

    for j in 0..beta.ncols() {
        let dp = (sigma2[j] * nu).max(MIN_DEVIANCE);
        let coef = scale * 0.5 * nu / dp;
        add_deviance_profile_vjp(
            coef,
            j,
            x,
            weights,
            beta,
            residual,
            grad_x,
            grad_y,
            grad_weights,
        );
        add_rank_one_penalty_vjp(coef * lambda, beta.column(j), grad_penalty);
    }
    Ok(())
}

/// VJP contribution from an upstream gradient on `edf`.
///
/// With `M = X^T W X + λ S`, `edf = trace(M^{-1} · X^T W X) = p - λ trace(M^{-1} S)`.
/// Holding `λ` fixed, the direct partials are
///   ∂edf/∂A = λ M^{-1} S M^{-1}      (A = X^T W X, symmetric)
///   ∂edf/∂S = −λ M^{-1} A M^{-1} = −λ M^{-1} + λ² M^{-1} S M^{-1}
///   ∂edf/∂λ = −trace(M^{-1} S) + λ trace((M^{-1} S)²)
/// The λ-component is returned as the lambda_adjoint contribution and routed
/// through the implicit-function chain by the caller (same path as
/// `upstream_lambda` and `upstream_reml_score`).
fn add_edf_vjp(
    scale: f64,
    x: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    lambda: f64,
    inverse_hessian: &Array2<f64>,
    grad_x: &mut Array2<f64>,
    grad_penalty: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
) -> f64 {
    // m_inv_s = M^{-1} S, then g_a = λ M^{-1} S M^{-1} = ∂edf/∂A.
    let m_inv_s = dense_ab(inverse_hessian.view(), penalty);
    let mut g_a = dense_ab(m_inv_s.view(), inverse_hessian.view());
    g_a.mapv_inplace(|v| v * lambda);

    // Chain ∂edf/∂A through A = X^T W X.
    //   grad_X += scale · 2 · (W X) · G_A
    //   grad_w_i += scale · (X G_A X^T)_{ii}
    let xg = dense_ab(x, g_a.view());
    // Row-scaled dense accumulate: grad_x[i,:] += (2·scale·weights[i]) · xg[i,:].
    // (Inlined here — the former `assembly::add_row_scaled_dense_into` helper was
    // removed as "unused" by 0cb722d, which missed this gam-pyffi-reachable caller.)
    let leading_scale = 2.0 * scale;
    for i in 0..xg.nrows() {
        let row_scale = leading_scale * weights[i];
        for k in 0..xg.ncols() {
            grad_x[[i, k]] += row_scale * xg[[i, k]];
        }
    }
    for i in 0..x.nrows() {
        let mut quad = 0.0;
        for k in 0..x.ncols() {
            quad += x[[i, k]] * xg[[i, k]];
        }
        grad_weights[i] += scale * quad;
    }

    // ∂edf/∂S = -λ M^{-1} + λ² M^{-1} S M^{-1} = -λ M^{-1} + λ · g_a
    // (since g_a = λ M^{-1} S M^{-1}, so λ · g_a = λ² M^{-1} S M^{-1}).
    for row in 0..grad_penalty.nrows() {
        for col in 0..grad_penalty.ncols() {
            grad_penalty[[row, col]] +=
                scale * (-lambda * inverse_hessian[[row, col]] + lambda * g_a[[row, col]]);
        }
    }

    // ∂edf/∂λ (with A, S fixed) = -tr(M^{-1} S) + λ tr((M^{-1} S)²).
    let p_dim = m_inv_s.nrows();
    let mut tr_m_inv_s = 0.0;
    for i in 0..p_dim {
        tr_m_inv_s += m_inv_s[[i, i]];
    }
    let mut tr_squared = 0.0;
    for i in 0..p_dim {
        for j in 0..p_dim {
            tr_squared += m_inv_s[[i, j]] * m_inv_s[[j, i]];
        }
    }
    scale * (-tr_m_inv_s + lambda * tr_squared)
}

fn add_reml_rho_gradient_vjp(
    scale: f64,
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    lambda: f64,
    inverse_hessian: &Array2<f64>,
    beta: &Array2<f64>,
    residual: &Array2<f64>,
    sigma2: &Array1<f64>,
    nu: f64,
    grad_x: &mut Array2<f64>,
    grad_y: &mut Array2<f64>,
    grad_penalty: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
) {
    let d = beta.ncols() as f64;
    let inverse_s = dense_ab(inverse_hessian.view(), penalty);
    let trace_kernel = dense_ab(inverse_s.view(), inverse_hessian.view());
    for row in 0..grad_penalty.nrows() {
        for col in 0..grad_penalty.ncols() {
            grad_penalty[[row, col]] += scale
                * 0.5
                * d
                * lambda
                * (inverse_hessian[[col, row]] - lambda * trace_kernel[[col, row]]);
        }
    }
    let xt = dense_ab(x, trace_kernel.view());
    for i in 0..x.nrows() {
        let wi = -scale * d * lambda * weights[i];
        for k in 0..x.ncols() {
            grad_x[[i, k]] += wi * xt[[i, k]];
        }
        let mut quad = 0.0;
        for k in 0..x.ncols() {
            quad += x[[i, k]] * xt[[i, k]];
        }
        grad_weights[i] -= scale * 0.5 * d * lambda * quad;
    }

    let s_beta = dense_ab(penalty, beta.view());
    let mut upstream_beta = Array2::<f64>::zeros(beta.dim());
    for j in 0..beta.ncols() {
        let dp = (sigma2[j] * nu).max(MIN_DEVIANCE);
        let q = lambda * beta.column(j).dot(&s_beta.column(j));
        let q_coef = scale * nu / dp;
        for row in 0..beta.nrows() {
            upstream_beta[[row, j]] = q_coef * lambda * s_beta[[row, j]];
        }
        let dp_coef = -scale * 0.5 * nu * q / (dp * dp);
        add_rank_one_penalty_vjp(
            (0.5 * q_coef + dp_coef) * lambda,
            beta.column(j),
            grad_penalty,
        );
        add_deviance_profile_vjp(
            dp_coef,
            j,
            x,
            weights,
            beta,
            residual,
            grad_x,
            grad_y,
            grad_weights,
        );
    }
    // The implicit-root VJP holds lambda fixed inside this partial; only the
    // data, penalty, and weight side effects from the ridge solve are needed.
    add_ridge_profile_vjp_fixed_lambda(
        1.0,
        x,
        y,
        penalty,
        weights,
        lambda,
        inverse_hessian,
        beta,
        upstream_beta.view(),
        grad_x,
        grad_y,
        grad_penalty,
        grad_weights,
    );
}

fn add_rank_one_penalty_vjp(
    scale: f64,
    beta_col: ArrayView1<'_, f64>,
    grad_penalty: &mut Array2<f64>,
) {
    for row in 0..beta_col.len() {
        for col in 0..beta_col.len() {
            grad_penalty[[row, col]] += scale * beta_col[row] * beta_col[col];
        }
    }
}

/// The one range/null threshold for a cached penalty spectrum.
///
/// `GaussianRemlEigenCache::penalty_rank` is *defined* as the number of
/// eigenvalues strictly above this value, so any consumer that asks "is this
/// direction in the range of `S`?" with a different predicate is answering a
/// different question about the same matrix, and the two answers disagree on
/// exactly the directions whose reciprocal is `1/roundoff`.
///
/// The threshold is relative to `max|δ|` and never floored at an absolute
/// value, for the reason documented at the cache builder: an absolute floor
/// breaks REML's invariance under `S → c·S`.
///
/// Evaluating this on the STORED eigenvalues gives the same number the cache
/// builder computed before its sign cleanup: that loop only zeroes eigenvalues
/// that are negative AND within tolerance, and such a value cannot have carried
/// `max|δ|`.
fn penalty_range_tolerance(eigenvalues: ArrayView1<'_, f64>) -> f64 {
    let max_abs = eigenvalues
        .iter()
        .fold(0.0_f64, |acc, &value| acc.max(value.abs()));
    max_abs * EIGEN_REL_TOL
}

/// The cached penalty spectrum read through the ONE range/null predicate.
///
/// [`penalty_range_tolerance`] defines the threshold; this is the single place
/// that APPLIES it, and every consumer of `cache.penalty_eigenvalues` goes
/// through here. A direction that fails the test is reported as EXACTLY `0.0`,
/// so a downstream `δ > 0.0` or `δ == 0.0` on a classified value re-reads that
/// one predicate instead of introducing a second and a third.
///
/// #2740: before this existed the same array was partitioned three ways —
/// `δ > EIGEN_REL_TOL·max|δ|` (which DEFINES `penalty_rank`), an absolute
/// `δ > 0.0`, and an absolute `δ == 0.0`. A numerically null direction the
/// eigensolver returns as a small POSITIVE number (measured at
/// `3.20001575162645240e-18` on an ordinary second-difference penalty) was then
/// simultaneously in the range set by one test and out of it by another. That
/// is not a rounding difference but a POPULATION mismatch: the compactified limit cost
/// summed `ln δ` over `count(δ > 0.0)` directions and subtracted
/// `logdet_penalty_positive`, which is reconciled to exactly `penalty_rank`
/// directions, so the ρ→+∞ limit cost the profile search compares against was
/// wrong by `ln(3.2e-18) = −40.3` per disputed direction — and the same
/// mismatch offset `Σ t/(1+t)` by `penalty_rank` in the gradient and in its
/// interval enclosure.
///
/// Classifying rather than only counting also removes the disputed direction
/// from `log|H| = Σ log(1 + λδ)`: keeping it there while `log|S|₊` counts only
/// `penalty_rank` directions makes `V(ρ)` diverge like `(count − rank)·ρ/2`
/// instead of approaching the finite `ρ→+∞` limit the compactified endpoint
/// claims. Value, gradient, enclosure, limit, coefficients and dispersion all
/// therefore score the SAME matrix.
#[derive(Clone, Copy)]
struct PenaltyRangeSpectrum<'a> {
    eigenvalues: &'a Array1<f64>,
    tolerance: f64,
}

impl<'a> PenaltyRangeSpectrum<'a> {
    fn of(cache: &'a GaussianRemlEigenCache) -> Self {
        Self {
            eigenvalues: &cache.penalty_eigenvalues,
            tolerance: penalty_range_tolerance(cache.penalty_eigenvalues.view()),
        }
    }

    fn len(&self) -> usize {
        self.eigenvalues.len()
    }

    /// `δ_i` when direction `i` is in the range of `S`, exactly `0.0` when it is
    /// not.
    #[inline]
    fn get(&self, index: usize) -> f64 {
        let delta = self.eigenvalues[index];
        if delta > self.tolerance { delta } else { 0.0 }
    }

    fn iter(&self) -> impl Iterator<Item = f64> + '_ {
        (0..self.len()).map(move |index| self.get(index))
    }

    /// The number of range directions under this same predicate — the quantity
    /// `GaussianRemlEigenCache::penalty_rank` is defined to be, recomputed here
    /// so a sum and the count it is differenced against can never be populated
    /// by two different rules.
    fn rank(&self) -> usize {
        self.eigenvalues
            .iter()
            .filter(|&&delta| delta > self.tolerance)
            .count()
    }
}

fn gaussian_reml_penalty_pseudoinverse_from_cache(
    cache: &GaussianRemlEigenCache,
) -> Result<Array2<f64>, EstimationError> {
    let p = cache.penalty_eigenvalues.len();
    // Ask the range/null question with the SAME predicate that defined
    // `cache.penalty_rank`.  `δ > 0.0` is a different question: the cache's
    // cleanup loop zeroes only NEGATIVE eigenvalues inside the tolerance, so a
    // numerically null direction the eigensolver returned as `+3.2e-18` is
    // classified null by `penalty_rank` and positive here — and this is the one
    // consumer that divides by it.  Measured at `p = 8` on a second-difference
    // penalty: `penalty_rank = 6`, seven eigenvalues pass `δ > 0.0`, and the
    // seventh contributes `1/3.20001575162645240e-18 = 3.125e17`, which lands in
    // the returned penalty gradient as entries of `1.618287e15` — fifteen orders
    // above every legitimate term, on healthy and near-interpolating charts
    // alike.  See [`penalty_range_tolerance`].
    // The shared predicate makes the selected count equal `penalty_rank` by
    // construction only when `penalty_rank` was derived from THIS array.  A
    // cache supplied through `GaussianRemlWarmStart` or `prepare_gaussian_reml`'s
    // `Some(eigen_cache)` can carry a rank computed under another rule, and
    // `validate_gaussian_reml_eigen_cache` checks only
    // `penalty_rank + nullity == p` — never the rank against the spectrum.  So
    // the agreement is checked rather than assumed.
    //
    // [`gaussian_penalty_positive_logdet`] reconciles the same disagreement by
    // taking the `penalty_rank` largest, and this site deliberately does NOT
    // copy that.  There the selected values are consumed as `ln(δ)`, which is
    // bounded; here they are consumed as `1/δ`, so re-admitting a direction that
    // failed the relative test reintroduces exactly the `1/roundoff` term this
    // function was repaired to exclude — the reconciliation would restore the
    // defect through its own fallback.  A dividing consumer has no safe
    // reconstruction of a rank it cannot verify, so it refuses and says which
    // two numbers disagreed.
    let spectrum = PenaltyRangeSpectrum::of(cache);
    let tolerance = spectrum.tolerance;
    let selected: Vec<usize> = (0..p).filter(|eig| spectrum.get(*eig) > 0.0).collect();
    if selected.len() != cache.penalty_rank {
        crate::bail_invalid_estim!(
            "Gaussian REML penalty pseudoinverse: the cache reports penalty_rank={} but {} of its \
             {p} eigenvalues exceed the range tolerance {tolerance:e}; the pseudoinverse divides by \
             each selected eigenvalue, so it cannot reconcile a rank it did not derive",
            cache.penalty_rank,
            selected.len()
        );
    }
    let mut scaled_basis = Array2::<f64>::zeros((p, p));
    for eig in selected {
        let delta = spectrum.get(eig);
        for row in 0..p {
            scaled_basis[[row, eig]] = cache.coefficient_basis[[row, eig]] / delta;
        }
    }
    Ok(dense_ab(scaled_basis.view(), cache.coefficient_basis.t()))
}

fn add_deviance_profile_vjp(
    scale: f64,
    output: usize,
    x: ArrayView2<'_, f64>,
    weights: &Array1<f64>,
    beta: &Array2<f64>,
    residual: &Array2<f64>,
    grad_x: &mut Array2<f64>,
    grad_y: &mut Array2<f64>,
    grad_weights: &mut Array1<f64>,
) {
    for i in 0..x.nrows() {
        let r = residual[[i, output]];
        let wr_scale = scale * weights[i] * r;
        grad_y[[i, output]] += 2.0 * wr_scale;
        for k in 0..x.ncols() {
            grad_x[[i, k]] -= 2.0 * wr_scale * beta[[k, output]];
        }
        grad_weights[i] += scale * r * r;
    }
}

fn validate_initial_lambda(lambda: f64) -> Result<f64, EstimationError> {
    if lambda.is_finite() && lambda > 0.0 {
        Ok(lambda)
    } else {
        Err(EstimationError::InvalidInput(format!(
            "Gaussian REML initial lambda must be finite and positive; got {lambda}"
        )))
    }
}

fn dense_ab(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Array2<f64> {
    fast_ab(&a, &b)
}

fn dense_atb(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Array2<f64> {
    fast_atb(&a, &b)
}

fn dense_xt_diag_x(x: ArrayView2<'_, f64>, w: ArrayView1<'_, f64>) -> Array2<f64> {
    fast_xt_diag_x(&x, &w)
}

fn dense_xt_diag_y(
    x: ArrayView2<'_, f64>,
    w: ArrayView1<'_, f64>,
    y: ArrayView2<'_, f64>,
) -> Array2<f64> {
    fast_xt_diag_y(&x, &w, &y)
}

fn matrix_fingerprint(matrix: ArrayView2<'_, f64>) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    hash = fnv1a_mix(hash, matrix.nrows() as u64);
    hash = fnv1a_mix(hash, matrix.ncols() as u64);
    for &value in matrix {
        hash = fnv1a_mix(hash, value.to_bits());
    }
    hash
}

fn fnv1a_mix(hash: u64, value: u64) -> u64 {
    (hash ^ value).wrapping_mul(0x100000001b3)
}

/// Build eigen caches for K problems that share the same penalty matrix in a
/// single phased pipeline. X'WX construction is batched by the caller; each
/// cache then uses the same Cholesky/eigendecomposition implementation as the
/// single-fit path.
pub fn build_gaussian_reml_eigen_cache_batched(
    xtwx_matrices: Vec<Array2<f64>>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
) -> Vec<Result<GaussianRemlEigenCache, EstimationError>> {
    let penalty_owned = canonicalize_penalty(penalty);
    let penalty = penalty_owned.view();
    let k = xtwx_matrices.len();
    if k == 0 {
        return Vec::new();
    }
    let fingerprints: Vec<u64> = xtwx_matrices
        .iter()
        .map(|m| matrix_fingerprint(m.view()))
        .collect();

    let p = xtwx_matrices[0].nrows();
    let uniform_square = p > 0 && xtwx_matrices.iter().all(|matrix| matrix.dim() == (p, p));
    if uniform_square && k > 1 {
        let mut lower_matrices = xtwx_matrices.clone();
        if gam_gpu::try_cholesky_batched_lower_inplace(&mut lower_matrices).is_some() {
            // The batched penalty transform is an optional accelerator. On
            // failure we must NOT fabricate an empty Vec (indexing it per-block
            // would silently drop the transform for every block and could index
            // out of range) — instead route every block through the same
            // no-GPU-transform path used when the batched transform is
            // unavailable, which recomputes the whitened penalty on CPU from the
            // already-valid Cholesky factor `lower`.
            let transforms = batched_whitened_penalty_transforms(&lower_matrices, penalty);
            return lower_matrices
                .into_iter()
                .enumerate()
                .map(|(b, lower)| {
                    let precomputed_transform = transforms.as_ref().map(|t| t[b].clone());
                    gaussian_reml_eigen_cache_from_lower_with_transform(
                        lower,
                        penalty,
                        nullspace_dim,
                        fingerprints[b],
                        precomputed_transform,
                    )
                })
                .collect();
        }
    }

    let mut results = Vec::with_capacity(k);
    for (b, xtwx) in xtwx_matrices.into_iter().enumerate() {
        let lower = match gaussian_reml_cholesky_lower(xtwx) {
            Ok(l) => l,
            Err(err) => {
                results.push(Err(err));
                continue;
            }
        };
        results.push(gaussian_reml_eigen_cache_from_lower_with_transform(
            lower,
            penalty,
            nullspace_dim,
            fingerprints[b],
            None,
        ));
    }
    results
}

fn batched_whitened_penalty_transforms(
    lowers: &[Array2<f64>],
    penalty: ArrayView2<'_, f64>,
) -> Option<Vec<Array2<f64>>> {
    let first = lowers.first()?;
    let p = first.nrows();
    if p == 0 || first.ncols() != p || lowers.iter().any(|lower| lower.dim() != (p, p)) {
        return None;
    }
    let mut linv_stack = Array3::<f64>::zeros((lowers.len(), p, p));
    for (idx, lower) in lowers.iter().enumerate() {
        let l_inv = invert_lower_triangular(lower).ok()?;
        linv_stack.slice_mut(s![idx, .., ..]).assign(&l_inv);
    }
    let penalty_in_metric = gam_gpu::try_fast_ab_broadcast_b_batched(linv_stack.view(), penalty)?;
    let transformed =
        gam_gpu::try_fast_abt_strided_batched(penalty_in_metric.view(), linv_stack.view())?;
    Some(
        transformed
            .axis_iter(Axis(0))
            .map(|matrix| matrix.to_owned())
            .collect(),
    )
}

pub fn build_gaussian_reml_eigen_cache_with_nullspace_dim(
    x: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<GaussianRemlEigenCache, EstimationError> {
    let penalty_owned = canonicalize_penalty(penalty);
    let penalty = penalty_owned.view();
    let n = x.nrows();
    validate_gaussian_reml_design(x, penalty, weights)?;
    let weight = gaussian_reml_weights(n, weights)?;

    let xtwx = dense_xt_diag_x(x, weight.view());
    gaussian_reml_eigen_cache_from_xtwx(xtwx, penalty, nullspace_dim)
}

fn validate_gaussian_reml_design(
    x: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<(), EstimationError> {
    let n = x.nrows();
    let p = x.ncols();
    if penalty.nrows() != p || penalty.ncols() != p {
        crate::bail_invalid_estim!(
            "Gaussian REML penalty shape mismatch: expected {p}x{p}, got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        );
    }
    if x.iter().chain(penalty.iter()).any(|v| !v.is_finite()) {
        crate::bail_invalid_estim!("Gaussian REML inputs must be finite");
    }
    if let Some(w) = weights {
        if w.len() != n {
            crate::bail_invalid_estim!(
                "Gaussian REML weights length mismatch: expected {n}, got {}",
                w.len()
            );
        }
        if w.iter().any(|value| !value.is_finite() || *value < 0.0) {
            crate::bail_invalid_estim!("Gaussian REML weights must be finite and non-negative");
        }
    }
    Ok(())
}

/// Effective observation count for the REML residual degrees of freedom.
///
/// A prior weight of exactly `0` is the universal "excluded / infinite-variance"
/// convention (mgcv, statsmodels): such a row must be equivalent to omitting it
/// entirely. The weighted response energy already handles this (`weight[row] *
/// y² = 0` for a zero-weight row), and a zero-weight row likewise contributes
/// nothing to `XᵀWX` / `XᵀWy`, so it cannot move the coefficients at a fixed
/// smoothing parameter. The one place a zero-weight row used to leak in was the
/// residual degrees of freedom `ν = n − nullity`, which counted the raw row
/// count `n`. That deflated `σ²`, under-smoothed `λ`, and (through `λ`) biased
/// the coefficients — growing with the number of zero-weight rows. The residual
/// DoF must instead be built from the number of rows that actually enter the
/// likelihood, i.e. those with a strictly positive weight.
fn effective_observation_count(weight: ArrayView1<'_, f64>) -> usize {
    weight.iter().filter(|&&w| w > 0.0).count()
}

fn gaussian_reml_weights(
    n: usize,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array1<f64>, EstimationError> {
    match weights {
        Some(w) => {
            if w.len() != n {
                crate::bail_invalid_estim!(
                    "Gaussian REML weights length mismatch: expected {n}, got {}",
                    w.len()
                );
            }
            if w.iter().any(|value| !value.is_finite() || *value < 0.0) {
                crate::bail_invalid_estim!("Gaussian REML weights must be finite and non-negative");
            }
            Ok(w.to_owned())
        }
        None => Ok(Array1::ones(n)),
    }
}

fn gaussian_reml_eigen_cache_from_xtwx(
    xtwx: Array2<f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
) -> Result<GaussianRemlEigenCache, EstimationError> {
    let xtwx_fingerprint = matrix_fingerprint(xtwx.view());
    let lower = gaussian_reml_cholesky_lower(xtwx)?;
    gaussian_reml_eigen_cache_from_lower(lower, penalty, nullspace_dim, xtwx_fingerprint)
}

/// Cache-build entry point for callers that have already computed `L =
/// chol(X'WX, lower)`. Used by the batched K-way fit path so a single
/// `cusolverDnDpotrfBatched` call factors all K matrices, then each cache
/// finishes per-fit without re-doing the Cholesky.
fn gaussian_reml_eigen_cache_from_lower(
    lower: Array2<f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    xtwx_fingerprint: u64,
) -> Result<GaussianRemlEigenCache, EstimationError> {
    gaussian_reml_eigen_cache_from_lower_with_transform(
        lower,
        penalty,
        nullspace_dim,
        xtwx_fingerprint,
        None,
    )
}

/// Cache-build variant that accepts a pre-computed whitened penalty
/// `L⁻¹·S·L⁻ᵀ`. Callers pass `None` to compute it from the Cholesky factor.
fn gaussian_reml_eigen_cache_from_lower_with_transform(
    lower: Array2<f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    xtwx_fingerprint: u64,
    precomputed_transform: Option<Array2<f64>>,
) -> Result<GaussianRemlEigenCache, EstimationError> {
    let p = lower.nrows();
    if lower.ncols() != p {
        crate::bail_invalid_estim!("Gaussian REML Cholesky factor must be square");
    }
    let penalty_fingerprint = matrix_fingerprint(penalty);
    let logdet_xtwx = 2.0 * lower.diag().iter().map(|v| v.ln()).sum::<f64>();
    let transformed_penalty = match precomputed_transform {
        Some(transformed) => transformed,
        None => {
            let l_inv = invert_lower_triangular(&lower)?;
            let penalty_in_metric = dense_ab(l_inv.view(), penalty);
            dense_ab(penalty_in_metric.view(), l_inv.t())
        }
    };
    let (mut penalty_eigenvalues, eigenvectors) =
        transformed_penalty.eigh(Side::Lower).map_err(|_| {
            EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            }
        })?;
    // Rank tolerance must be RELATIVE to the largest eigenvalue — never
    // floored at an absolute value. The old `.max(1.0)` clamped the
    // tolerance up whenever max|eig| < 1, classifying genuine modes as
    // null for small-scale penalties (e.g. Wahba pseudo-spline `m=4`
    // with `K(p,p) ≈ 3e-4`). That broke REML's invariance under
    // `S → c·S` — the optimum λ rescales but the score landscape
    // diverges from the true marginal likelihood, and the smooth
    // contribution collapsed to ~0 on smooth truths.
    // Fully scale-invariant form: `safety · max|eig| · eps`.
    let eig_tol = penalty_range_tolerance(penalty_eigenvalues.view());
    for value in &mut penalty_eigenvalues {
        if *value < 0.0 && value.abs() <= eig_tol {
            *value = 0.0;
        }
        if *value < 0.0 {
            crate::bail_invalid_estim!(
                "Gaussian REML penalty is not positive semidefinite; eigenvalue={value:.3e}"
            );
        }
    }
    let penalty_rank = penalty_eigenvalues
        .iter()
        .filter(|&&value| value > eig_tol)
        .count();
    let nullity = p - penalty_rank;
    if let Some(expected_nullity) = nullspace_dim
        && expected_nullity != nullity
    {
        crate::bail_invalid_estim!(
            "Gaussian REML penalty nullspace mismatch: expected {expected_nullity}, inferred {nullity}"
        );
    }
    let logdet_penalty_positive = gaussian_penalty_positive_logdet(penalty, penalty_rank)?;
    let coefficient_basis = solve_upper_triangular_matrix(&lower.t().to_owned(), &eigenvectors)?;

    Ok(GaussianRemlEigenCache {
        penalty_eigenvalues,
        eigenvectors,
        coefficient_basis,
        xtwx_fingerprint,
        penalty_fingerprint,
        logdet_xtwx,
        logdet_penalty_positive,
        penalty_rank,
        nullity,
    })
}

fn gaussian_reml_cholesky_lower(xtwx: Array2<f64>) -> Result<Array2<f64>, EstimationError> {
    // Attempt Cholesky directly; on failure, retry with a tiny diagonal jitter
    // proportional to the matrix trace. X'WX is symmetric positive semidefinite
    // by construction, but FP noise (e.g. in a basis whose kernel block is only
    // FP-orthogonal to its explicit polynomial nullspace columns, as the
    // periodic Duchon basis is) can push the smallest eigenvalue slightly
    // negative on adversarial inputs, intermittently failing Cholesky. A
    // jitter of 1e-12 * trace/p shifts every eigenvalue up by an amount well
    // below the natural scale of the well-conditioned eigenvalues but well
    // above f64 FP noise, eliminating the spurious-failure regime.
    let mut gpu_candidate = xtwx.clone();
    if gam_gpu::try_cholesky_lower_inplace(&mut gpu_candidate).is_some() {
        return Ok(gpu_candidate);
    }
    if let Ok(chol) = xtwx.cholesky(Side::Lower) {
        return Ok(chol.lower_triangular());
    }
    let p = xtwx.nrows();
    let trace: f64 = (0..p).map(|i| xtwx[[i, i]]).sum();
    if !trace.is_finite() || trace <= 0.0 {
        return Err(EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        });
    }
    let schedule = RidgeSchedule::geometric(1e-12 * trace / (p as f64), 6);
    escalate_ridge(
        schedule,
        |jitter| {
            let mut jittered = xtwx.clone();
            for i in 0..p {
                jittered[[i, i]] += jitter;
            }
            let mut gpu_candidate = jittered.clone();
            if gam_gpu::try_cholesky_lower_inplace(&mut gpu_candidate).is_some() {
                return Some(gpu_candidate);
            }
            jittered
                .cholesky(Side::Lower)
                .ok()
                .map(|chol| chol.lower_triangular())
        },
    )
    .map(|success| success.value)
    .map_err(|exhausted| {
        // Cholesky failed at every escalation. The largest shift actually tried
        // is one growth factor below the one the schedule would try next, and
        // X'WX is still not numerically PSD there, so `trace / last_attempted`
        // is a measured lower bound on the conditioning rather than a blanket
        // `INFINITY`.
        let last_attempted = exhausted.next_ridge / schedule.growth;
        EstimationError::ModelIsIllConditioned {
            condition_number: if last_attempted > 0.0 && last_attempted.is_finite() {
                trace / last_attempted
            } else {
                f64::INFINITY
            },
        }
    })
}

fn gaussian_penalty_positive_logdet(
    penalty: ArrayView2<'_, f64>,
    penalty_rank: usize,
) -> Result<f64, EstimationError> {
    if penalty_rank == 0 {
        return Ok(0.0);
    }
    let (pen_eigs, _) = penalty.to_owned().eigh(Side::Lower).map_err(|_| {
        EstimationError::ModelIsIllConditioned {
            condition_number: f64::INFINITY,
        }
    })?;
    // Scale-invariant relative tolerance — see the cousin site for the
    // rationale. Same `.max(1.0)` floor used to live here and corrupted
    // the positive-eigenvalue count for small-scale penalties. This is a
    // DIFFERENT array from the cache's (raw `S`, not `L⁻¹SL⁻ᵀ`), but it is the
    // SAME criterion, so it is read from the one definition rather than
    // re-derived here (#2740).
    let pen_tol = penalty_range_tolerance(pen_eigs.view());
    let mut positive_eigs: Vec<f64> = pen_eigs
        .iter()
        .copied()
        .filter(|&value| value > pen_tol)
        .collect();
    if positive_eigs.len() != penalty_rank {
        positive_eigs = pen_eigs
            .iter()
            .copied()
            .filter(|&value| value > 0.0)
            .collect();
        positive_eigs.sort_by(|a, b| b.total_cmp(a));
        if positive_eigs.len() < penalty_rank {
            return Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            });
        }
        positive_eigs.truncate(penalty_rank);
    }
    Ok(positive_eigs.iter().map(|value| value.ln()).sum())
}

fn validate_gaussian_reml_eigen_cache(
    cache: &GaussianRemlEigenCache,
    p: usize,
) -> Result<(), EstimationError> {
    if cache.penalty_eigenvalues.len() != p
        || cache.eigenvectors.dim() != (p, p)
        || cache.coefficient_basis.dim() != (p, p)
    {
        crate::bail_invalid_estim!(
            "Gaussian REML eigen cache dimension mismatch: expected {p} coefficients"
        );
    }
    if cache.penalty_rank > p || cache.nullity > p || cache.penalty_rank + cache.nullity != p {
        crate::bail_invalid_estim!(
            "Gaussian REML eigen cache rank/nullity mismatch: rank={}, nullity={}, p={p}",
            cache.penalty_rank,
            cache.nullity
        );
    }
    if !(cache.logdet_xtwx.is_finite() && cache.logdet_penalty_positive.is_finite()) {
        crate::bail_invalid_estim!("Gaussian REML eigen cache log-determinants must be finite");
    }
    if cache
        .penalty_eigenvalues
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
        || cache.eigenvectors.iter().any(|value| !value.is_finite())
        || cache
            .coefficient_basis
            .iter()
            .any(|value| !value.is_finite())
    {
        crate::bail_invalid_estim!(
            "Gaussian REML eigen cache entries must be finite with non-negative eigenvalues"
                .to_string(),
        );
    }
    // #2740: `penalty_rank` is DEFINED as the number of eigenvalues clearing
    // `penalty_range_tolerance`, and `logdet_penalty_positive` is reconciled to
    // exactly that many directions. Every consumer reads the spectrum through
    // `PenaltyRangeSpectrum`, which applies the same test — but a cache handed in
    // through `GaussianRemlWarmStart` or `prepare_gaussian_reml`'s
    // `Some(eigen_cache)` can carry a rank counted under some other rule, and the
    // shape check above never compares the rank against the spectrum. Then the
    // objective's Σ over the range and the `penalty_rank` it is differenced
    // against are populated by two different rules again, which is the whole
    // defect. Check it here rather than assume it.
    let spectrum = PenaltyRangeSpectrum::of(cache);
    let classified_rank = spectrum.rank();
    if classified_rank != cache.penalty_rank {
        crate::bail_invalid_estim!(
            "Gaussian REML eigen cache reports penalty_rank={} but {classified_rank} of its {p} \
             eigenvalues clear the range tolerance {:e}; the log-determinant sums run over the \
             directions that clear it while log|S|₊ and the gradient offset are denominated in \
             penalty_rank, so the two must be the same count",
            cache.penalty_rank,
            spectrum.tolerance
        );
    }
    Ok::<(), _>(())
}

fn prepare_gaussian_reml(
    x: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    nullspace_dim: Option<usize>,
    weights: Option<ArrayView1<'_, f64>>,
    eigen_cache: Option<&GaussianRemlEigenCache>,
) -> Result<GaussianRemlPrepared, EstimationError> {
    // Enforce the symmetric-S contract once at the central forward chokepoint;
    // every closed-form forward path funnels through here.
    let penalty_owned = canonicalize_penalty(penalty);
    let penalty = penalty_owned.view();
    let n = x.nrows();
    let p = x.ncols();
    let d = y.ncols();
    validate_gaussian_reml_design(x, penalty, weights)?;
    if y.nrows() != n {
        crate::bail_invalid_estim!(
            "Gaussian REML row mismatch: X has {n} rows but Y has {}",
            y.nrows()
        );
    }
    if y.iter().any(|v| !v.is_finite()) {
        crate::bail_invalid_estim!("Gaussian REML inputs must be finite");
    }
    let weight = gaussian_reml_weights(n, weights)?;
    let n_effective = effective_observation_count(weight.view());

    let xtwy = dense_xt_diag_y(x, weight.view(), y);
    let ywy = Array1::from_iter((0..d).map(|j| {
        let mut value = 0.0;
        for row in 0..n {
            value += weight[row] * y[[row, j]] * y[[row, j]];
        }
        value
    }));
    let xtwx = dense_xt_diag_x(x, weight.view());

    if let Some(cache) = eigen_cache {
        validate_gaussian_reml_eigen_cache(cache, p)?;
        let xtwx_fingerprint = matrix_fingerprint(xtwx.view());
        if cache.xtwx_fingerprint != xtwx_fingerprint {
            crate::bail_invalid_estim!("Gaussian REML eigen cache X'WX mismatch");
        }
        let penalty_fingerprint = matrix_fingerprint(penalty);
        if cache.penalty_fingerprint != penalty_fingerprint {
            crate::bail_invalid_estim!("Gaussian REML eigen cache penalty mismatch");
        }
        if let Some(expected_nullity) = nullspace_dim
            && expected_nullity != cache.nullity
        {
            crate::bail_invalid_estim!(
                "Gaussian REML eigen cache nullspace mismatch: expected {expected_nullity}, got {}",
                cache.nullity
            );
        }
        if n_effective <= cache.nullity {
            crate::bail_invalid_estim!(
                "Gaussian REML requires more positive-weight rows than the nullspace dimension; got n_effective={n_effective}, nullity={}",
                cache.nullity
            );
        }
        let projected_rhs = dense_atb(cache.coefficient_basis.view(), xtwy.view());
        let projected_rhs_squared = projected_rhs.mapv(|value| value * value);
        return Ok(GaussianRemlPrepared {
            cache: cache.clone(),
            ywy,
            projected_rhs_squared,
            projected_rhs,
            n_effective,
            n_outputs: d,
        });
    }

    let cache = gaussian_reml_eigen_cache_from_xtwx(xtwx, penalty, nullspace_dim)?;
    if n_effective <= cache.nullity {
        crate::bail_invalid_estim!(
            "Gaussian REML requires more positive-weight rows than the nullspace dimension; got n_effective={n_effective}, nullity={}",
            cache.nullity
        );
    }
    let projected_rhs = dense_atb(cache.coefficient_basis.view(), xtwy.view());
    let projected_rhs_squared = projected_rhs.mapv(|value| value * value);

    Ok(GaussianRemlPrepared {
        cache,
        ywy,
        projected_rhs_squared,
        projected_rhs,
        n_effective,
        n_outputs: d,
    })
}

impl GaussianRemlPrepared {
    fn nu(&self) -> f64 {
        self.n_effective as f64 - self.cache.nullity as f64
    }

    fn evaluate(&self, rho: f64) -> ObjectiveEval {
        evaluate_reml_parts(
            &self.cache,
            self.ywy.view(),
            self.projected_rhs_squared.view(),
            self.n_effective,
            self.n_outputs,
            rho,
        )
    }

    fn coefficients(&self, lambda: f64) -> Array2<f64> {
        let mut scaled = self.projected_rhs.clone();
        let spectrum = PenaltyRangeSpectrum::of(&self.cache);
        for i in 0..spectrum.len() {
            let scale = 1.0 / (1.0 + lambda * spectrum.get(i));
            for value in scaled.row_mut(i) {
                *value *= scale;
            }
        }
        dense_ab(self.cache.coefficient_basis.view(), scaled.view())
    }

    /// Profiled dispersion `σ̂²_j = dp_j(ρ̂)/ν`, through the same cancellation-free
    /// decomposition the objective, the domain check and the enclosure use (see
    /// [`dispersion_residual_parts`]). Computing it as `ywy − Σ c²/(1+λδ)`
    /// returns exactly `0` on a design that interpolates its response, which
    /// would propagate a zero scale into every downstream covariance.
    fn sigma2(&self, rho: f64) -> Array1<f64> {
        let nu = self.nu();
        Array1::from_iter((0..self.n_outputs).map(|j| {
            let DispersionResidualParts {
                unpenalized_residual,
                penalized_residual,
                ..
            } = dispersion_residual_parts(
                &self.cache,
                self.ywy.view(),
                self.projected_rhs_squared.view(),
                j,
                rho,
            );
            (unpenalized_residual + penalized_residual) / nu
        }))
    }
}

/// Roundoff resolution of the profiled residual deviance `dp_j` for one output:
/// the magnitude at or below which `dp_j` carries no significant digit and is
/// indistinguishable from exactly zero.
///
/// `dp_j = (ywy_j − Σ_i c²_ij) + Σ_i c²_ij·u_i` is accumulated in
/// nearest-rounded arithmetic. The absolute sum of its contributing terms is
/// `ywy_j + Σ_i c²_ij ≤ 2·ywy_j`, because `Σ_i c²_ij ≤ ywy_j` — the discarded
/// remainder `r0_j` is a squared weighted residual norm and therefore
/// non-negative (see [`dispersion_residual_parts`]). Under the standard
/// `γ_m = m·eps/(1 − m·eps)` model the accumulated error is bounded by
/// `γ_m · 2·ywy_j`, so that product is the resolution.
///
/// Both inputs are derived, not chosen: `eps` is a machine constant and the
/// operation count is this file's own convention, single-sourced with
/// `reml_deriv_enclosure_profile` (`64 + 32·(n_eig · n_out)`, here at
/// `n_out = 1` because the check is per output) and consumed by
/// [`conservative_interval`] under the same error model. The result is
/// proportional to `ywy_j`, so the derived bar is scale-invariant: rescaling
/// `y` by `α` scales both `dp_j` and the resolution by `α²`.
fn profile_residual_resolution(cache: &GaussianRemlEigenCache, ywy_output: f64) -> f64 {
    let operations =
        64usize.saturating_add(32usize.saturating_mul(cache.penalty_eigenvalues.len()));
    let n_eps = (operations as f64) * f64::EPSILON;
    if !(ywy_output.is_finite() && ywy_output >= 0.0) || n_eps >= 1.0 {
        return f64::INFINITY;
    }
    (n_eps / (1.0 - n_eps)) * 2.0 * ywy_output
}

/// Certify that every profiled residual is RESOLVABLY positive at `rho`.
/// Residual deviance is monotone increasing in rho, so validating the lower
/// search boundary certifies the log-dispersion domain on the entire window.
/// A zero/perfect-fit residual has no finite profiled Gaussian scale and must be
/// refused; replacing it with a tiny constant would change both the objective
/// and its derivatives.
///
/// #2723: the bar used to be the absolute `residual > 0.0`, and on a perfect fit
/// that predicate reads the sign of the last rounding rather than the design.
/// `dp = max(ywy − Σc², 0) + Σc²·u` is a sum of clamped non-negatives, so it can
/// only reach exactly `0.0` when the cancellation `ywy − Σc²` happens to land
/// non-positive AND no penalized direction carries any mass. Measured on four
/// designs whose true residual is EXACTLY zero, that bar refused two and
/// accepted two — the discriminator being whether the debris landed at `+1.8e-15`
/// or at `−3.6e-15`, and whether the basis was irrational or integral. Evidence
/// about one rounding, generalised to the design.
///
/// The verdict a perfect fit must get is REFUSAL, and the accepted side is the
/// wrong one: a profiled Gaussian likelihood genuinely cannot score an
/// exactly-interpolated response — `σ̂² → 0`, `V = ½ν·log(2π·dp/ν) → −∞`, and
/// every smoothing candidate ties at `−∞`. Accepting instead carries a `σ̂²` of
/// pure roundoff (measured at `1.6e-16` and `6.8e-14`) into the dominant term of
/// the score that model selection then ranks by, so the ranking is decided by
/// debris. Abstention is the only defensible outcome, and it is the one the two
/// already-refusing designs get.
///
/// So the bar is re-denominated in the quantity it means to test: refuse unless
/// `dp` exceeds its own arithmetic resolution, [`profile_residual_resolution`]
/// (`γ_m·2·ywy_j`, derived from `eps` and the measured response scale). All four
/// designs then agree on refusal, and the verdict is a property of the design
/// rather than of the last rounding. Note this bar answers RESOLVABILITY of
/// `dp` — not reliability of the score built on it, which is a wider band
/// (`≈ ½ν·eps/τ`) and a separate question.
fn validate_reml_profile_residuals(
    cache: &GaussianRemlEigenCache,
    ywy: ArrayView1<'_, f64>,
    projected_rhs_squared: ArrayView2<'_, f64>,
    rho: f64,
) -> Result<(), EstimationError> {
    for output in 0..ywy.len() {
        // Same `r0 + Σ c²·u` decomposition the evaluator and the enclosure use.
        // Checking the domain through the cancelling form while the search
        // evaluates the stable one lets a fit be refused for a residual that is
        // strictly positive, or admitted for one that is not.
        let DispersionResidualParts {
            unpenalized_residual,
            penalized_residual,
            ..
        } = dispersion_residual_parts(cache, ywy, projected_rhs_squared, output, rho);
        let residual = unpenalized_residual + penalized_residual;
        let resolution = profile_residual_resolution(cache, ywy[output]);
        if !(residual.is_finite() && residual > resolution) {
            return Err(EstimationError::InvalidInput(format!(
                "Gaussian REML profiled residual {output} is not resolvably positive at rho={rho}: {residual} against its own arithmetic resolution {resolution} (gamma_m * 2 * ywy, ywy={}); the design interpolates its response, so the profiled dispersion has no finite value",
                ywy[output]
            )));
        }
    }
    Ok(())
}

// ============================================================================
// Grid-free stationary-point certification for the profiled Gaussian-REML
// ρ-objective `V(ρ)` (ρ = ln λ).
// ============================================================================
//
// The previous optimizer sampled `V′` on a fixed 96-point ρ grid and refined
// the sign-change cells. A grid can only see stationary points it happens to
// bracket: two roots inside one 0.625-wide cell (or a root pair narrower than
// the sample spacing) are invisible, so the selected λ̂ was grid-resolution
// limited. This replaces the grid with analytic kernel enclosures plus
// operation-count roundoff padding. A successful return isolates the stationary
// structure to the stated finite-window resolution; an ambiguous cell refuses
// the fit through a typed convergence error.
//
// ---- Analytic structure of V′ (single-sourced with the evaluator) ----------
//
// With λ = e^ρ and t_i = λ·δ_i (δ_i = `cache.penalty_eigenvalues` ≥ 0), the two
// contributions of `gaussian_reml_logdet_term` / `gaussian_reml_dispersion_term`
// give, using dt_i/dρ = t_i,
//
//   V′(ρ) = ½d·( Σ_i t_i/(1+t_i) − rank )                                 (g1)
//         + ½ν·Σ_j [ Σ_i c²_ij · t_i/(1+t_i)² ] / dp_j(ρ)                 (g2)
//
//   dp_j(ρ) = ywy_j − Σ_i c²_ij/(1+t_i)   (residual deviance, strictly > 0,
//                                          strictly increasing in ρ).
//
//   g1: each kernel t/(1+t) ∈ [0,1) is monotone ↑; the sum minus `rank`
//       positive eigenvalues is strictly negative and rises to 0⁻ — g1 is
//       monotone increasing.
//   g2 ≥ 0: numerator kernel t/(1+t)² is a unimodal bump peaking at ¼ when t=1.
//
// V has poles only at λ = −1/δ_i < 0, i.e. outside the real ρ window, so V is
// real-analytic on [RHO_LOWER, RHO_UPPER] ⇒ V′ has finitely many isolated roots
// there. That finiteness is what makes exhaustive enumeration well-posed.
//
// ---- V″ and its kernel critical points -------------------------------------
//
//   V″(ρ) = ½d·Σ_i t_i/(1+t_i)²
//         + ½ν·Σ_j [ dp″_j/dp_j − (dp′_j/dp_j)² ],
//   dp′_j = Σ_i c²_ij·t_i/(1+t_i)²,   dp″_j = Σ_i c²_ij·t_i(1−t_i)/(1+t_i)³.
//
// The only non-monotone / non-unimodal kernel is k(t) = t(1−t)/(1+t)³ in dp″.
// Differentiating and clearing (1+t)⁴ (documented derivation):
//
//   k′(t) = [ (1−2t)(1+t) − 3(t−t²) ] / (1+t)⁴
//         = ( 1 − 4t + t² ) / (1+t)⁴.
//
// So the interior extrema of k are the roots of the fixed quadratic
//
//        t² − 4t + 1 = 0   ⇒   t = 2 ± √3,
//
// giving the analytic range for k over any t-window by testing the two endpoints
// and whichever of {2−√3, 2+√3} lies strictly inside. Every other kernel is
// monotone (t/(1+t), 1/(1+t)) or unimodal with a known peak (t/(1+t)²), so each
// admits an endpoint-plus-critical-point range.
//
// ---- Interval enclosure of (V′, V″) over [a,b] -----------------------------
//
// log(t_i) ∈ [a+log δ_i, b+log δ_i] (monotone in ρ). Per kernel:
//   t/(1+t)   ↑   → endpoint range.
//   1/(1+t)   ↓   → endpoint range ⇒ dp endpoints bound dp(a),
//                    dp(b) (dp monotone), both > 0.
//   t/(1+t)²  unimodal → endpoint range, max replaced by ¼ iff 1∈[t_lo,t_hi].
//   k(t)      → endpoints + interior roots 2±√3 (above).
// g2 ratio enclosure ½ν·[ Σ num_lo/dp_hi , Σ num_hi/dp_lo ] is conservative
// in the ratio. The accumulated bounds are widened by a gamma_n roundoff budget
// and checked against both endpoint jets before a cell may be pruned.
//
// ---- Branch-and-bound (DFS, fixed stack, no heap in the shared core) --------
//
// For [a,b]: (1) enclose V′; if 0 ∉ enclosure, prune. (2) else enclose V″; if
// 0 ∉ enclosure then V′ is monotone on [a,b] (≤ 1 root) — isolate by the shared
// refinement iff the evaluated V′(a),V′(b) straddle 0. (3) else split at the
// midpoint. Children are pushed right-then-left so the leftmost interval is
// processed first and isolated roots are therefore EMITTED IN ASCENDING ρ with
// no sort and no heap. Recursion is bounded at MAX_DEPTH =
// ⌈log₂((RHO_UPPER−RHO_LOWER)/RHO_BRACKET_RESOLUTION)⌉, where the resolution is
// the same ρ-bracket width the safeguarded Newton stop uses. Reaching it without
// a monotonicity certificate returns `RemlDidNotConverge`; no best-effort fit is
// minted.

/// ρ-bracket resolution shared by the enumeration recursion depth and the
/// safeguarded-Newton stop: a bracket narrower than `RHO_BRACKET_RESOLUTION·
/// (1+|ρ|)` is treated as converged. ρ = ln λ is O(1)–O(10), so 1e-12 pins λ̂ to
/// ~12 significant figures — the floor below which cost ordering between two ρ
/// candidates is pure rounding noise (the non-smoothness that used to wreck the
/// closed-form REML reverse-mode VJP against finite differences).
const RHO_BRACKET_RESOLUTION: f64 = 1.0e-12;

/// ⌈log₂(range/resolution)⌉ computed at compile time: the smallest depth `d`
/// with `resolution·2^d ≥ range`, i.e. the number of midpoint bisections needed
/// to drive the window down to the ρ-bracket resolution. `const fn` so the DFS
/// stack is a fixed-size array with no heap.
const fn dfs_max_depth(range: f64, resolution: f64) -> usize {
    let mut width = range;
    let mut depth = 0usize;
    while width > resolution {
        width *= 0.5;
        depth += 1;
    }
    depth
}

/// Maximum branch-and-bound recursion depth (= 46 for the ±30 window at 1e-12).
const MAX_DEPTH: usize = dfs_max_depth(RHO_UPPER - RHO_LOWER, RHO_BRACKET_RESOLUTION);

/// A closed real interval `[lo, hi]` used to enclose `V′`/`V″` over a ρ-cell.
#[derive(Clone, Copy)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    fn entire() -> Self {
        Self {
            lo: f64::NEG_INFINITY,
            hi: f64::INFINITY,
        }
    }
}

/// Next representable f64 strictly below `x` (toward −∞): outward rounding for
/// an enclosure lower bound, so the rounded value is provably ≤ the exact one.
fn round_down(x: f64) -> f64 {
    if x.is_nan() || x == f64::NEG_INFINITY {
        return x;
    }
    if x == 0.0 {
        return -f64::from_bits(1);
    }
    let bits = x.to_bits();
    let next = if x > 0.0 { bits - 1 } else { bits + 1 };
    f64::from_bits(next)
}

/// Next representable f64 strictly above `x` (toward +∞): outward rounding for
/// an enclosure upper bound, so the rounded value is provably ≥ the exact one.
fn round_up(x: f64) -> f64 {
    if x.is_nan() || x == f64::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f64::from_bits(1);
    }
    let bits = x.to_bits();
    let next = if x > 0.0 { bits + 1 } else { bits - 1 };
    f64::from_bits(next)
}

fn add_down(lhs: f64, rhs: f64) -> f64 {
    round_down(lhs + rhs)
}

fn add_up(lhs: f64, rhs: f64) -> f64 {
    round_up(lhs + rhs)
}

/// Outward product of a non-negative scalar and a non-negative interval.
/// Invalid signs/order are not a recoverable numerical perturbation: callers
/// must refuse certification rather than silently clamp the interval.
fn nonnegative_product_interval(lhs: f64, rhs: Interval) -> Option<Interval> {
    if !(lhs.is_finite()
        && lhs >= 0.0
        && rhs.lo.is_finite()
        && rhs.hi.is_finite()
        && rhs.lo >= 0.0
        && rhs.hi >= rhs.lo)
    {
        return None;
    }
    Some(Interval {
        lo: round_down(lhs * rhs.lo).max(0.0),
        hi: round_up(lhs * rhs.hi),
    })
}

/// Outward square of a non-negative interval.
fn nonnegative_square_interval(bounds: Interval) -> Option<Interval> {
    if !(bounds.lo.is_finite()
        && bounds.hi.is_finite()
        && bounds.lo >= 0.0
        && bounds.hi >= bounds.lo)
    {
        return None;
    }
    Some(Interval {
        lo: round_down(bounds.lo * bounds.lo).max(0.0),
        hi: round_up(bounds.hi * bounds.hi),
    })
}

/// Enclose accumulated nearest-rounded arithmetic under the standard
/// `gamma_n = n*eps/(1-n*eps)` model, then step both endpoints outward once.
/// `magnitude` is an absolute sum of the contributing terms, so cancellation
/// in the final bound cannot erase its roundoff allowance. Non-finite
/// arithmetic refuses pruning by returning the entire real line.
fn conservative_interval(lo: f64, hi: f64, magnitude: f64, operations: usize) -> Interval {
    if !(lo.is_finite() && hi.is_finite() && magnitude.is_finite() && lo <= hi) {
        return Interval::entire();
    }
    let n_eps = (operations as f64) * f64::EPSILON;
    if n_eps >= 1.0 {
        return Interval::entire();
    }
    let pad =
        (n_eps / (1.0 - n_eps)) * magnitude.max(lo.abs()).max(hi.abs()).max(f64::MIN_POSITIVE);
    Interval {
        lo: round_down(lo - pad),
        hi: round_up(hi + pad),
    }
}

/// Analytic per-eigenvalue ranges of the `V′`/`V″` kernels over a monotone
/// log-`t` window. See the module derivation above:
/// `u=t/(1+t)` ↑, `w=t/(1+t)²` unimodal (peak ¼ at t=1),
/// `k=t(1−t)/(1+t)³` with interior extrema at t = 2 ± √3.
///
/// `v=1/(1+t)` is deliberately absent: the residual deviance is enclosed through
/// `dp = r0 + Σ c²·u`, never through the cancelling `dp = ywy − Σ c²·v`, so no
/// consumer needs the `v` range and carrying it would invite the cancelling form
/// back in.
#[derive(Clone, Copy)]
struct KernelRange {
    u_lo: f64,
    u_hi: f64,
    w_lo: f64,
    w_hi: f64,
    k_lo: f64,
    k_hi: f64,
}

fn kernel_ranges(log_t_lo: f64, log_t_hi: f64) -> KernelRange {
    let kernels = |log_t: f64| modal_kernels(log_t, 1.0);
    let left = kernels(log_t_lo);
    let right = kernels(log_t_hi);

    // t/(1+t) increasing; 1/(1+t) decreasing. Evaluating in log-t space
    // retains the finite limiting values when exp(log_t) is not representable.
    let u_lo = left.u;
    let u_hi = right.u;

    // t/(1+t)² unimodal, single interior peak ¼ at t=1.
    let w_a = left.w;
    let w_b = right.w;
    let w_lo = w_a.min(w_b);
    let w_hi = if log_t_lo <= 0.0 && 0.0 <= log_t_hi {
        0.25
    } else {
        w_a.max(w_b)
    };

    // k(t)=t(1−t)/(1+t)³: interior extrema are the roots t = 2 ± √3 of the fixed
    // quadratic t²−4t+1 (derived in the module comment).
    let sqrt3 = 3.0_f64.sqrt();
    let cp_lo = (2.0 - sqrt3).ln();
    let cp_hi = (2.0 + sqrt3).ln();
    let mut k_lo = left.k.min(right.k);
    let mut k_hi = left.k.max(right.k);
    if log_t_lo < cp_lo && cp_lo < log_t_hi {
        let kc = kernels(cp_lo).k;
        k_lo = k_lo.min(kc);
        k_hi = k_hi.max(kc);
    }
    if log_t_lo < cp_hi && cp_hi < log_t_hi {
        let kc = kernels(cp_hi).k;
        k_lo = k_lo.min(kc);
        k_hi = k_hi.max(kc);
    }

    KernelRange {
        u_lo: round_down(u_lo).max(0.0),
        u_hi: round_up(u_hi),
        w_lo: round_down(w_lo).max(0.0),
        w_hi: round_up(w_hi),
        k_lo: round_down(k_lo),
        k_hi: round_up(k_hi),
    }
}

/// Outward-rounded interval enclosure of `(V′([a,b]), V″([a,b]))` for the DFS.
/// Both intervals are conservative bounds for the analytic profiled derivative
/// range over the ρ-cell. Kernel extrema are included explicitly and the final
/// accumulated arithmetic is padded by an operation-count roundoff bound. A
/// non-finite or non-positive residual bound returns the entire line, which
/// prevents pruning and therefore ends in a typed unresolved-search refusal if
/// tighter children cannot certify the cell.
fn reml_deriv_enclosure(
    cache: &GaussianRemlEigenCache,
    ywy: ArrayView1<'_, f64>,
    projected_rhs_squared: ArrayView2<'_, f64>,
    n_effective: usize,
    n_outputs: usize,
    a: f64,
    b: f64,
) -> (Interval, Interval) {
    reml_deriv_enclosure_profile(
        cache,
        ywy,
        projected_rhs_squared,
        n_outputs,
        n_effective as f64 - cache.nullity as f64,
        a,
        b,
    )
}

/// Derivative enclosure for an arbitrary response-dispersion profile.
/// `logdet_output_count` prices the independent coefficient columns, while
/// `dispersion_dof` is the degrees of freedom of each pooled deviance column in
/// `ywy` / `projected_rhs_squared`.  The ordinary multi-response objective uses
/// `d` separate columns each with `n-q` degrees of freedom; shared-dispersion
/// REML supplies one pooled column with `d(n-q)` degrees of freedom.
fn reml_deriv_enclosure_profile(
    cache: &GaussianRemlEigenCache,
    ywy: ArrayView1<'_, f64>,
    projected_rhs_squared: ArrayView2<'_, f64>,
    logdet_output_count: usize,
    dispersion_dof: f64,
    a: f64,
    b: f64,
) -> (Interval, Interval) {
    let d = logdet_output_count as f64;
    let spectrum = PenaltyRangeSpectrum::of(cache);
    let rank = cache.penalty_rank as f64;
    let half_d = 0.5 * d;
    let half_nu = 0.5 * dispersion_dof;
    // g1 = ½d(Σ t/(1+t) − rank) and the logdet part of V″ = ½d·Σ t/(1+t)²,
    // both summing only over the strictly positive penalty eigenvalues.
    let mut sum_u_lo = 0.0;
    let mut sum_u_hi = 0.0;
    let mut sum_w_lo = 0.0;
    let mut sum_w_hi = 0.0;
    // The enclosure must bound the expression the evaluator computes, so it
    // classifies through the same predicate `gaussian_reml_logdet_term` uses and
    // the population of this sum is again exactly `rank` (#2740).
    for delta in spectrum.iter() {
        if delta > 0.0 {
            let log_delta = delta.ln();
            let kr = kernel_ranges(a + log_delta, b + log_delta);
            sum_u_lo = add_down(sum_u_lo, kr.u_lo);
            sum_u_hi = add_up(sum_u_hi, kr.u_hi);
            sum_w_lo = add_down(sum_w_lo, kr.w_lo);
            sum_w_hi = add_up(sum_w_hi, kr.w_hi);
        }
    }
    let g1_lo = round_down(half_d * round_down(sum_u_lo - rank));
    let g1_hi = round_up(half_d * round_up(sum_u_hi - rank));

    // Dispersion contributions to V′ (g2) and V″, folded per output so no
    // per-output heap buffer is needed (the shared core is Vec-free).
    let mut g2_lo = 0.0;
    let mut g2_hi = 0.0;
    let mut vpp_disp_lo = 0.0;
    let mut vpp_disp_hi = 0.0;
    for j in 0..ywy.len() {
        let mut num_lo = 0.0; // Σ c² · w   (= dp′, ≥ 0)
        let mut num_hi = 0.0;
        let mut su_lo = 0.0; // Σ c² · u   (the ρ-dependent part of dp, ≥ 0)
        let mut su_hi = 0.0;
        // Σ c², the ρ-INDEPENDENT half of dp's decomposition. Accumulated as a
        // PLAIN sum in the same order as `dispersion_residual_parts`' `total_c2`,
        // so `r0` below is bit-identical to the evaluator's — see the comment at
        // `r0` for why this term must be a point rather than an interval.
        let mut c2_point = 0.0;
        let mut dph_lo = 0.0; // Σ c² · k   (= dp″, sign-indefinite)
        let mut dph_hi = 0.0;
        for eig in 0..spectrum.len() {
            let delta = spectrum.get(eig);
            let c2 = projected_rhs_squared[[eig, j]];
            let log_delta = if delta == 0.0 {
                f64::NEG_INFINITY
            } else {
                delta.ln()
            };
            let kr = kernel_ranges(a + log_delta, b + log_delta);
            let Some(w_product) = nonnegative_product_interval(
                c2,
                Interval {
                    lo: kr.w_lo,
                    hi: kr.w_hi,
                },
            ) else {
                return (Interval::entire(), Interval::entire());
            };
            let Some(u_product) = nonnegative_product_interval(
                c2,
                Interval {
                    lo: kr.u_lo,
                    hi: kr.u_hi,
                },
            ) else {
                return (Interval::entire(), Interval::entire());
            };
            num_lo = add_down(num_lo, w_product.lo);
            num_hi = add_up(num_hi, w_product.hi);
            su_lo = add_down(su_lo, u_product.lo);
            su_hi = add_up(su_hi, u_product.hi);
            c2_point += c2;
            dph_lo = add_down(dph_lo, round_down(c2 * kr.k_lo));
            dph_hi = add_up(dph_hi, round_up(c2 * kr.k_hi));
        }
        // dp is monotone increasing, and it is enclosed through the SAME
        // `r0 + Σ c²·u` decomposition the evaluator uses (see
        // `dispersion_residual_parts`). Bounding the cancelling form
        // `ywy − Σ c²·v` instead put the whole quantity below the cancellation
        // floor on a saturated design: the outward-rounded `Σ c²·v` reaches
        // `ywy` near the small-λ end even though the true `dp` is positive
        // there, the bound goes non-positive, and the enclosure collapses to the
        // entire line — a cell that can be neither pruned nor certified monotone
        // and therefore must split. In the summed form the ρ-dependent part is a
        // sum of non-negatives and the only cancellation left sits in `r0`,
        // which is ρ-independent and therefore identical in every cell.
        // `r0` is a KNOWN CONSTANT here, not an unknown to be bracketed.
        //
        // #2694/#2703. Bracketing it as `[max(ywy − c2_hi, 0), ywy − c2_lo]`
        // treated the cancellation's lost digits as uncertainty in the quantity
        // being enclosed. On a design that reproduces its response the bracket
        // becomes `[0, ~eps·ywy]`, `dp_lo` collapses onto `Σc²·u` — the ratio
        // NUMERATOR's own scale — and `num_hi/dp_lo` reads `1.0` whatever the
        // data, putting `V′`'s upper bound at `g1 + half_nu`. Measured: a
        // ZERO-WIDTH enclosure of a `V′` of `−1.0` came back `[−1.0, +4.5]`,
        // width `5.5 = half_nu` exactly, at three separate ρ.
        //
        // The width equalling a STRUCTURAL CONSTANT regardless of design is the
        // tell: the residual term contributed nothing to the bound. And it could
        // not be bisected away — the file's own comment says why, offered as
        // reassurance: `r0` "is ρ-independent and therefore identical in every
        // cell". A quantity identical in every cell is a CONSTANT, and a
        // constant belongs in an enclosure as a point.
        //
        // The search certifies the stationary structure of the objective AS
        // EVALUATED: the DFS audits every cell against the computed endpoint
        // jets (`interval_contains(dv, ea.grad)`), `refine_stationary_rho_core`
        // brackets sign changes of the computed gradient, and the returned ρ̂
        // builds the computed fit. So the enclosure owes a bound on the
        // evaluator's `V′`, and for that `r0` is the single value
        // `dispersion_residual_parts` uses — same plain accumulation, same
        // order, same clamp, hence bit-identical. Forming it any other way is
        // precisely the objective↔enclosure desync this file exists to prevent.
        //
        // Roundoff is still priced: `conservative_interval` at the end of this
        // function pads the accumulated bounds by the operation-count budget.
        let r0 = (ywy[j] - c2_point).max(0.0);
        let dp_lo = add_down(r0, su_lo);
        let dp_hi = add_up(r0, su_hi);
        if !(dp_lo.is_finite() && dp_hi.is_finite() && dp_lo > 0.0 && dp_hi >= dp_lo) {
            return (Interval::entire(), Interval::entire());
        }

        // g2_j = num_j / dp_j  (num ≥ 0, dp > 0).
        let ratio_lo = round_down(num_lo / dp_hi).max(0.0);
        let ratio_hi = round_up(num_hi / dp_lo);
        g2_lo = add_down(g2_lo, ratio_lo);
        g2_hi = add_up(g2_hi, ratio_hi);

        // dp″/dp with dp″ sign-indefinite: exact four-corner range over the
        // strictly positive denominator interval.
        let quotients = [
            dph_lo / dp_lo,
            dph_lo / dp_hi,
            dph_hi / dp_lo,
            dph_hi / dp_hi,
        ];
        let adp_lo = round_down(quotients.iter().copied().fold(f64::INFINITY, f64::min));
        let adp_hi = round_up(quotients.iter().copied().fold(f64::NEG_INFINITY, f64::max));

        // (dp′/dp)² with dp′ ≥ 0, dp > 0.
        let bl = round_down(num_lo / dp_hi).max(0.0);
        let bh = round_up(num_hi / dp_lo);
        let Some(squared_ratio) = nonnegative_square_interval(Interval { lo: bl, hi: bh }) else {
            return (Interval::entire(), Interval::entire());
        };

        // term_j = dp″/dp − (dp′/dp)².
        vpp_disp_lo = add_down(vpp_disp_lo, round_down(adp_lo - squared_ratio.hi));
        vpp_disp_hi = add_up(vpp_disp_hi, round_up(adp_hi - squared_ratio.lo));
    }

    let vp_lo = add_down(g1_lo, round_down(half_nu * g2_lo));
    let vp_hi = add_up(g1_hi, round_up(half_nu * g2_hi));
    let vpp_lo = add_down(
        round_down(half_d * sum_w_lo),
        round_down(half_nu * vpp_disp_lo),
    );
    let vpp_hi = add_up(round_up(half_d * sum_w_hi), round_up(half_nu * vpp_disp_hi));

    let operations = 64usize.saturating_add(
        32usize.saturating_mul(
            cache
                .penalty_eigenvalues
                .len()
                .saturating_mul(ywy.len().max(1)),
        ),
    );
    let vp_magnitude = g1_lo.abs() + g1_hi.abs() + half_nu.abs() * (g2_lo.abs() + g2_hi.abs());
    let vpp_magnitude = half_d.abs() * (sum_w_lo.abs() + sum_w_hi.abs())
        + half_nu.abs() * (vpp_disp_lo.abs() + vpp_disp_hi.abs());
    (
        conservative_interval(vp_lo, vp_hi, vp_magnitude, operations),
        conservative_interval(vpp_lo, vpp_hi, vpp_magnitude, operations),
    )
}

#[derive(Clone, Copy, Debug)]
struct StationaryRoot {
    rho: f64,
    bracket: [f64; 2],
}

#[derive(Clone, Copy, Debug)]
struct ProfileSelection {
    rho: f64,
}

#[derive(Clone, Copy)]
struct ProfileSearchControls {
    lower: f64,
    upper: f64,
    resolution: f64,
    max_depth: usize,
}

impl ProfileSearchControls {
    const PRODUCTION: Self = Self {
        lower: RHO_LOWER,
        upper: RHO_UPPER,
        resolution: RHO_BRACKET_RESOLUTION,
        max_depth: MAX_DEPTH,
    };
}

fn profile_search_refusal(
    eval: &impl Fn(f64) -> ObjectiveEval,
    checkpoint: f64,
    reason: String,
) -> EstimationError {
    let e = eval(checkpoint);
    EstimationError::RemlDidNotConverge {
        context: "closed-form Gaussian profiled REML stationary search".to_string(),
        reason,
        iterations: 0,
        final_value: e.cost,
        projected_grad_norm: e.grad.is_finite().then_some(e.grad.abs()),
        // This route makes NO stationarity comparison, so it reports no
        // bound (#2458/#2530).
        //
        // It used to report `GRAD_TOL·(1 + |V|)`, and I labelled that a
        // gradient band of its own. Measuring instead of reading settles it:
        // `GRAD_TOL` occurs exactly twice in this file — its definition and
        // that message — so nothing ever compared against it. The acceptance
        // criterion here is `width <= resolution * scale`, a BRACKET-WIDTH test
        // in rho, and every refusal above is a bracket, enclosure or
        // representability failure rather than a residual weighed against a
        // band. Naming that number a rung made a false sentence more
        // confident, which is the defect this pair of issues exists to remove.
        stationarity_standard: StationarityStandard::NoComparison,
        rho_checkpoint: vec![checkpoint],
    }
}

/// Isolate one unique derivative root to a geometric rho bracket. Newton is
/// accepted only in the central half of the maintained sign bracket, so every
/// iteration contracts it by at least one quarter. There is no iteration cap:
/// termination follows from geometric contraction, and loss of a representable
/// interior point is a typed refusal rather than a best-effort root.
fn refine_stationary_rho_core(
    eval: &impl Fn(f64) -> ObjectiveEval,
    mut lo: f64,
    mut hi: f64,
    resolution: f64,
    mut hint: Option<f64>,
) -> Result<StationaryRoot, EstimationError> {
    let mut left = eval(lo);
    let mut right = eval(hi);
    if left.grad == 0.0 {
        return Ok(StationaryRoot {
            rho: lo,
            bracket: [lo, lo],
        });
    }
    if right.grad == 0.0 {
        return Ok(StationaryRoot {
            rho: hi,
            bracket: [hi, hi],
        });
    }
    if left.grad.is_sign_positive() == right.grad.is_sign_positive() {
        return Err(profile_search_refusal(
            eval,
            0.5 * (lo + hi),
            format!("stationary refinement received a non-bracketing cell [{lo}, {hi}]"),
        ));
    }

    loop {
        let width = hi - lo;
        let scale = 1.0 + lo.abs().max(hi.abs());
        if width <= resolution * scale {
            let midpoint = lo + 0.5 * width;
            let middle = if midpoint > lo && midpoint < hi {
                Some((midpoint, eval(midpoint)))
            } else {
                None
            };
            let mut representative = (lo, left);
            if right.grad.abs() < representative.1.grad.abs() {
                representative = (hi, right);
            }
            if let Some(candidate) = middle
                && candidate.1.grad.abs() < representative.1.grad.abs()
            {
                representative = candidate;
            }
            return Ok(StationaryRoot {
                rho: representative.0,
                bracket: [lo, hi],
            });
        }

        let midpoint = lo + 0.5 * width;
        if !(midpoint > lo && midpoint < hi) {
            return Err(profile_search_refusal(
                eval,
                midpoint,
                format!(
                    "stationary root on [{lo}, {hi}] reached floating-point spacing before rho resolution {resolution}"
                ),
            ));
        }
        let guard = 0.25 * width;
        let base = if left.grad.abs() <= right.grad.abs() {
            (lo, left)
        } else {
            (hi, right)
        };
        let newton = if base.1.hess != 0.0 {
            base.0 - base.1.grad / base.1.hess
        } else {
            f64::NAN
        };
        let candidate = hint
            .take()
            .filter(|&rho| rho >= lo + guard && rho <= hi - guard)
            .or_else(|| {
                (newton.is_finite() && newton >= lo + guard && newton <= hi - guard)
                    .then_some(newton)
            })
            .unwrap_or(midpoint);
        if !(candidate > lo && candidate < hi) {
            return Err(profile_search_refusal(
                eval,
                midpoint,
                format!(
                    "stationary refinement could not represent an interior point on [{lo}, {hi}]"
                ),
            ));
        }
        let current = eval(candidate);
        if current.grad == 0.0 {
            return Ok(StationaryRoot {
                rho: candidate,
                bracket: [candidate, candidate],
            });
        }
        if current.grad.is_sign_positive() == left.grad.is_sign_positive() {
            lo = candidate;
            left = current;
        } else {
            hi = candidate;
            right = current;
        }
    }
}

/// Intersection of two sound enclosures of the same quantity.
///
/// Both arguments contain the true range, so the intersection does too. A
/// non-finite endpoint on one side simply lets the other side govern.
fn intersect_intervals(left: Interval, right: Interval) -> Interval {
    let lo = if right.lo.is_nan() { left.lo } else { left.lo.max(right.lo) };
    let hi = if right.hi.is_nan() { left.hi } else { left.hi.min(right.hi) };
    if lo > hi { left } else { Interval { lo, hi } }
}

/// Mean-value enclosure of `V′` over a cell of width `h`, from the EXACT
/// endpoint derivatives and a sound enclosure of `V″`.
///
/// For any `ρ ∈ [a, b]`, `V′(ρ) = V′(a) + V″(ξ)·(ρ − a)` for some `ξ` in the
/// cell, and `ρ − a ∈ [0, h]`, so
/// `V′(ρ) ∈ [V′(a) + min(0, curvature.lo·h), V′(a) + max(0, curvature.hi·h)]`.
/// The same holds anchored at `b` with `ρ − b ∈ [−h, 0]`. Taking the tighter of
/// the two anchors costs nothing and both are rounded outward.
///
/// Its width is `(curvature.hi − curvature.lo)·h`, which vanishes with the cell.
/// That is the property the direct ratio enclosure lacks (gam#2585).
fn mean_value_derivative_enclosure(
    at_a: Interval,
    at_b: Interval,
    curvature: Interval,
    h: f64,
) -> Interval {
    if !(h.is_finite()
        && h >= 0.0
        && curvature.lo.is_finite()
        && curvature.hi.is_finite()
        && at_a.lo.is_finite()
        && at_a.hi.is_finite()
        && at_b.lo.is_finite()
        && at_b.hi.is_finite())
    {
        return Interval::entire();
    }
    let down = round_down(curvature.lo * h).min(0.0);
    let up = round_up(curvature.hi * h).max(0.0);
    let from_a = Interval {
        lo: round_down(at_a.lo + down),
        hi: round_up(at_a.hi + up),
    };
    let from_b = Interval {
        lo: round_down(at_b.lo - up),
        hi: round_up(at_b.hi - down),
    };
    intersect_intervals(from_a, from_b)
}

/// Widen an enclosure until it also covers the evaluator's own endpoint values.
///
/// A superset of a sound enclosure is still sound, and the DFS audits every cell
/// by requiring the computed endpoint jets to lie inside the derivative
/// enclosure. Tightening can only make that audit harder to satisfy, so the
/// tightened interval is extended to cover them: the pruning decision then needs
/// the true range AND both computed endpoints to share a sign, which is strictly
/// more conservative than either alone.
fn widen_to_include(interval: Interval, first: f64, second: f64) -> Interval {
    let mut out = interval;
    for value in [first, second] {
        if value.is_finite() {
            out.lo = out.lo.min(value);
            out.hi = out.hi.max(value);
        }
    }
    out
}

fn interval_contains(interval: Interval, value: f64) -> bool {
    value.is_finite() && interval.lo <= value && value <= interval.hi
}

/// Certify the stationary structure of the actual profiled objective on the
/// finite rho window, then compare one representative of every isolated root
/// with both boundaries. `init_rho` is only a refinement hint; an arbitrary
/// nonstationary seed is never eligible to become the estimator.
fn enumerate_and_select_rho_with_controls(
    eval: impl Fn(f64) -> ObjectiveEval,
    enclose: impl Fn(f64, f64) -> (Interval, Interval),
    init_rho: Option<f64>,
    controls: ProfileSearchControls,
    mut visit: Option<&mut dyn FnMut(StationaryRoot, &ObjectiveEval)>,
) -> Result<ProfileSelection, EstimationError> {
    const CAP: usize = MAX_DEPTH + 4;
    let lower_eval = eval(controls.lower);
    let upper_eval = eval(controls.upper);
    // Every cell carries the objective jets of BOTH its endpoints. A bisection
    // shares three of its four child endpoints with the parent — (a, mid) and
    // (mid, b) reuse `a`, `b` and the ONE newly evaluated midpoint — so the DFS
    // evaluates each ρ point exactly once instead of re-evaluating both
    // endpoints of every popped cell. Each evaluation is an O(p) sweep over the
    // penalty spectrum, and on the saturated p ≈ n designs where this search
    // subdivides hardest that redundancy was two thirds of the evaluation work.
    // `eval` is a deterministic function of ρ over borrowed data, so the reused
    // jets are bit-identical to the recomputed ones (#2585).
    // Each endpoint also carries a POINT enclosure of `V′` there — `enclose(x, x)`
    // — which has no cell-width looseness at all, only the roundoff budget. It is
    // the anchor of the mean-value tightening below, and is cached on the stack
    // for the same reason the jets are: a bisection introduces exactly one new ρ.
    let lower_point = enclose(controls.lower, controls.lower).0;
    let upper_point = enclose(controls.upper, controls.upper).0;
    let mut stack = [(
        controls.lower,
        lower_eval,
        lower_point,
        controls.upper,
        upper_eval,
        upper_point,
        0usize,
    ); CAP];
    let mut top = 1usize;

    let (mut best_rho, mut best_eval) = if upper_eval.cost < lower_eval.cost {
        (controls.upper, upper_eval)
    } else {
        (controls.lower, lower_eval)
    };
    let mut last_root: Option<StationaryRoot> = None;
    // Search-effort tally. The branch-and-bound's cost is its CELL COUNT, which
    // is a property of the data (how tight the outward enclosure is on this
    // spectrum), not of `p` alone — so it has to be measured rather than
    // predicted. Reported once per search at `info` (#2585).
    let mut cells_visited = 0usize;
    let mut evaluations = 2usize;
    let mut deepest = 0usize;
    let mut unbounded_enclosures = 0usize;

    while top > 0 {
        top -= 1;
        let (a, ea, pa, b, eb, pb, depth) = stack[top];
        cells_visited += 1;
        deepest = deepest.max(depth);
        let (direct_dv, dvv) = enclose(a, b);
        // Mean-value tightening of the FIRST-derivative enclosure.
        //
        // The direct enclosure bounds `V′ = g1 + ½ν·(dp′/dp)` by bounding the
        // ratio's numerator and denominator independently. Each is tight to the
        // cell width `h`, but their ratio then inherits `≈ 2h` of RELATIVE
        // slack, and `½ν` multiplies it: the enclosure's width floors at `≈ ν·h`
        // no matter how accurate the pieces are. Where `V′` itself is far
        // smaller than `ν·h` — the whole small-λ stretch of a saturated design,
        // whose `V′` decays like `e^ρ` — no cell can be pruned until
        // `h ≲ |V′|/ν`, so the search bisects a wide band of ρ down to `~1e−6`
        // and visits `∫ν/|V′| dρ ≈ 10⁷–10⁸` cells (gam#2585: measured 10⁶ cells
        // per 164 s, all at depth 25, marching left to right across the window).
        //
        // But `V′` is differentiable on the cell with `V″` inside `dvv`, and
        // both exact endpoint jets are already in hand, so the mean value
        // theorem gives a second sound enclosure whose width is
        // `(dvv.hi − dvv.lo)·h` — proportional to the CURVATURE spread rather
        // than to `ν`, and therefore vanishing with the cell. Intersecting two
        // sound enclosures is sound, and both endpoint jets lie in the
        // intersection by construction, so the containment audit below is
        // unaffected. Nothing new is computed: `ea`, `eb` and `dvv` are already
        // on hand at this point.
        let dv = widen_to_include(
            intersect_intervals(
                direct_dv,
                mean_value_derivative_enclosure(pa, pb, dvv, b - a),
            ),
            ea.grad,
            eb.grad,
        );
        if !(dv.lo.is_finite() && dv.hi.is_finite()) {
            unbounded_enclosures += 1;
        }
        if !(interval_contains(dv, ea.grad)
            && interval_contains(dv, eb.grad)
            && interval_contains(dvv, ea.hess)
            && interval_contains(dvv, eb.hess))
        {
            return Err(profile_search_refusal(
                &eval,
                0.5 * (a + b),
                format!(
                    "analytic derivative enclosure [{}, {}] / curvature enclosure [{}, {}] missed an endpoint jet on [{a}, {b}]",
                    dv.lo, dv.hi, dvv.lo, dvv.hi
                ),
            ));
        }
        if dv.lo > 0.0 || dv.hi < 0.0 {
            continue;
        }

        let monotone = dvv.lo > 0.0 || dvv.hi < 0.0;
        let at_floor = depth >= controls.max_depth
            || (b - a) <= controls.resolution * (1.0 + a.abs().max(b.abs()));
        if !monotone && at_floor {
            return Err(profile_search_refusal(
                &eval,
                0.5 * (a + b),
                format!(
                    "stationary structure remained non-monotone on [{a}, {b}] at rho resolution {} \
                     after {cells_visited} branch-and-bound cells ({evaluations} objective \
                     evaluations, deepest bisection {deepest}, {unbounded_enclosures} cells whose \
                     derivative enclosure was unbounded)",
                    controls.resolution
                ),
            ));
        }

        if monotone {
            let crosses = (ea.grad <= 0.0 && eb.grad >= 0.0) || (ea.grad >= 0.0 && eb.grad <= 0.0);
            if crosses {
                let hint = init_rho.filter(|rho| rho.is_finite() && *rho >= a && *rho <= b);
                let root = refine_stationary_rho_core(&eval, a, b, controls.resolution, hint)?;
                let duplicate = last_root.is_some_and(|previous| {
                    root.rho.to_bits() == previous.rho.to_bits()
                        || (root.bracket[0] <= previous.bracket[1]
                            && previous.bracket[0] <= root.bracket[1])
                });
                if !duplicate {
                    let e = eval(root.rho);
                    if e.cost < best_eval.cost {
                        best_rho = root.rho;
                        best_eval = e;
                    }
                    if let Some(observer) = visit.as_deref_mut() {
                        observer(root, &e);
                    }
                    last_root = Some(root);
                }
            }
            continue;
        }

        let mid = a + 0.5 * (b - a);
        if !(mid > a && mid < b) || top + 2 > CAP {
            return Err(profile_search_refusal(
                &eval,
                mid,
                format!("stationary subdivision could not continue on [{a}, {b}]"),
            ));
        }
        let emid = eval(mid);
        let pmid = enclose(mid, mid).0;
        evaluations += 1;
        stack[top] = (mid, emid, pmid, b, eb, pb, depth + 1);
        top += 1;
        stack[top] = (a, ea, pa, mid, emid, pmid, depth + 1);
        top += 1;
    }
    log::info!(
        "[REML-BNB] certified 1-D rho search over [{}, {}]: {cells_visited} cells, \
         {evaluations} objective evaluations, deepest bisection {deepest}/{}, \
         {unbounded_enclosures} unbounded enclosures",
        controls.lower,
        controls.upper,
        controls.max_depth,
    );

    if !(best_eval.cost.is_finite() && best_eval.grad.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "Gaussian REML profiled search produced no finite candidate".to_string(),
        ));
    }
    Ok(ProfileSelection { rho: best_rho })
}

fn enumerate_and_select_rho(
    eval: impl Fn(f64) -> ObjectiveEval,
    enclose: impl Fn(f64, f64) -> (Interval, Interval),
    init_rho: Option<f64>,
    visit: Option<&mut dyn FnMut(StationaryRoot, &ObjectiveEval)>,
) -> Result<ProfileSelection, EstimationError> {
    enumerate_and_select_rho_with_controls(
        eval,
        enclose,
        init_rho,
        ProfileSearchControls::PRODUCTION,
        visit,
    )
}

/// Select ρ̂ = ln λ̂ by grid-free stationary-point enumeration (allocating path).
fn optimize_rho(
    prepared: &GaussianRemlPrepared,
    init_rho: Option<f64>,
) -> Result<f64, EstimationError> {
    validate_reml_profile_residuals(
        &prepared.cache,
        prepared.ywy.view(),
        prepared.projected_rhs_squared.view(),
        RHO_LOWER,
    )?;
    if prepared.cache.penalty_rank == 0 {
        return Ok(init_rho.unwrap_or(0.0).clamp(RHO_LOWER, RHO_UPPER));
    }
    let eval = |rho: f64| prepared.evaluate(rho);
    let enclose = |a: f64, b: f64| {
        reml_deriv_enclosure(
            &prepared.cache,
            prepared.ywy.view(),
            prepared.projected_rhs_squared.view(),
            prepared.n_effective,
            prepared.n_outputs,
            a,
            b,
        )
    };
    Ok(enumerate_and_select_rho(eval, enclose, init_rho, None)?.rho)
}

fn evaluate_reml_parts(
    cache: &GaussianRemlEigenCache,
    ywy: ArrayView1<'_, f64>,
    projected_rhs_squared: ArrayView2<'_, f64>,
    n_effective: usize,
    n_outputs: usize,
    rho: f64,
) -> ObjectiveEval {
    evaluate_reml_profile(
        cache,
        ywy,
        projected_rhs_squared,
        n_outputs,
        n_effective as f64 - cache.nullity as f64,
        rho,
    )
}

/// Evaluate the REML objective under either separate or pooled response
/// dispersions.  See [`reml_deriv_enclosure_profile`] for the two independent
/// dimensions of the profile contract.
fn evaluate_reml_profile(
    cache: &GaussianRemlEigenCache,
    ywy: ArrayView1<'_, f64>,
    projected_rhs_squared: ArrayView2<'_, f64>,
    logdet_output_count: usize,
    dispersion_dof: f64,
    rho: f64,
) -> ObjectiveEval {
    let d = logdet_output_count as f64;

    // Each term's value and its ρ-derivatives come back from ONE function so
    // they cannot be edited independently; `+=` folds the triple in lock-step.
    let (logdet_term, edf) = gaussian_reml_logdet_term(cache, rho, d);
    let mut eval = ObjectiveEval {
        cost: 0.0,
        grad: 0.0,
        hess: 0.0,
        edf,
        cost_roundoff: 0.0,
    };
    eval += logdet_term;
    for output in 0..ywy.len() {
        eval += gaussian_reml_dispersion_term(
            cache,
            ywy,
            projected_rhs_squared,
            output,
            dispersion_dof,
            rho,
        );
    }
    eval
}

fn invert_lower_triangular(lower: &Array2<f64>) -> Result<Array2<f64>, EstimationError> {
    let n = lower.nrows();
    if lower.ncols() != n {
        crate::bail_invalid_estim!("lower-triangular solve requires a square matrix");
    }
    let eye = Array2::eye(n);
    solve_lower_triangular_matrix(lower, &eye)
}

fn solve_lower_triangular_matrix(
    lower: &Array2<f64>,
    rhs: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    let n = lower.nrows();
    if lower.ncols() != n || rhs.nrows() != n {
        crate::bail_invalid_estim!("lower-triangular solve dimension mismatch");
    }
    if let Some(out) = gam_gpu::try_solve_lower_triangular_matrix(lower.view(), rhs.view()) {
        return Ok(out);
    }
    let mut out = Array2::<f64>::zeros(rhs.dim());
    for col in 0..rhs.ncols() {
        for i in 0..n {
            let mut value = rhs[[i, col]];
            for k in 0..i {
                value -= lower[[i, k]] * out[[k, col]];
            }
            let diag = lower[[i, i]];
            if !(diag.is_finite() && diag.abs() > 0.0) {
                return Err(EstimationError::ModelIsIllConditioned {
                    condition_number: f64::INFINITY,
                });
            }
            out[[i, col]] = value / diag;
        }
    }
    Ok(out)
}

/// Solve the SPD system `L Lᵀ X = rhs` for `X` given the lower Cholesky factor
/// `L` (as returned by [`gaussian_reml_cholesky_lower`]): a forward solve
/// against `L` followed by a back solve against `Lᵀ`.
fn solve_spd_from_lower_factor(
    lower: &Array2<f64>,
    rhs: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    let forward = solve_lower_triangular_matrix(lower, rhs)?;
    solve_upper_triangular_matrix(&lower.t().to_owned(), &forward)
}

fn solve_upper_triangular_matrix(
    upper: &Array2<f64>,
    rhs: &Array2<f64>,
) -> Result<Array2<f64>, EstimationError> {
    let n = upper.nrows();
    if upper.ncols() != n || rhs.nrows() != n {
        crate::bail_invalid_estim!("upper-triangular solve dimension mismatch");
    }
    if let Some(out) = gam_gpu::try_solve_upper_triangular_matrix(upper.view(), rhs.view()) {
        return Ok(out);
    }
    let mut out = Array2::<f64>::zeros(rhs.dim());
    for col in 0..rhs.ncols() {
        for i_rev in 0..n {
            let i = n - 1 - i_rev;
            let mut value = rhs[[i, col]];
            for k in (i + 1)..n {
                value -= upper[[i, k]] * out[[k, col]];
            }
            let diag = upper[[i, i]];
            if !(diag.is_finite() && diag.abs() > 0.0) {
                return Err(EstimationError::ModelIsIllConditioned {
                    condition_number: f64::INFINITY,
                });
            }
            out[[i, col]] = value / diag;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    /// #2694 / #2703 — a ZERO-WIDTH enclosure must SIGN an order-one `V′`.
    ///
    /// `enumerate_and_select_rho_with_controls` anchors its mean-value tightening
    /// on `enclose(x, x)` and states, at the `lower_point` / `upper_point`
    /// bindings, that such a point enclosure "has no cell-width looseness at all,
    /// only the roundoff budget". Both witnesses measured that false: at ZERO
    /// cell width the enclosure of a `V′` whose exact value was `-4.0` came back
    /// `[-4.0, +18.0]`.
    ///
    /// Cause. `dp = max(ywy − Σc², 0) + Σc²·u` (see `dispersion_residual_parts`).
    /// On a design that reproduces its response exactly the subtraction cancels;
    /// the enclosure used to BRACKET `r0` over the digits that cancellation
    /// destroyed, so its lower bound clamped to `0`, `dp_lo` collapsed onto
    /// `Σc²·u` — the ratio NUMERATOR's own scale — and `num_hi / dp_lo` read
    /// `1.0` regardless of the data, putting `V′`'s upper bound at
    /// `g1 + half_nu`. `r0` is ρ-INDEPENDENT, so that width survived every
    /// bisection and was present at zero width: a formula defect, not a
    /// tightening one, which is why no amount of subdivision removed it and why
    /// the repair is to carry `r0` as the point the evaluator already uses.
    ///
    /// Two fixtures, bounding the regime from BOTH sides:
    ///
    /// * CONTROL — a perfect fit whose cancellation is EXACT (small integer
    ///   design, response exactly in the column span). `r0` is then `[0, 0]`:
    ///   zero WIDTH, `dp_hi` collapses together with `dp_lo`, and the enclosure
    ///   stays sharp. This side is measured green on #2703 and must keep passing;
    ///   if it ever fails, the gate is broken rather than the code.
    /// * WITNESS — the #2694 harvest regime rebuilt without gam-sae: a CONSTANT
    ///   response, lying in the span of the basis AND in the null space of the
    ///   penalty, on an irrational (periodic-harmonic) basis so the cancellation
    ///   is INEXACT. This is the side the defect fired on; it is green since the
    ///   `r0`-as-a-point repair in `reml_deriv_enclosure_profile`.
    ///
    /// What the sharpness clause asserts carries no invented constant: an
    /// enclosure whose WIDTH exceeds the magnitude of the value it encloses
    /// cannot sign that value, and `dv.lo > 0 || dv.hi < 0` is precisely what the
    /// branch-and-bound prunes on. `ORDER_ONE_DERIVATIVE` selects WHICH ρ the
    /// property is asserted at — near a genuine stationary point straddling zero
    /// is the correct answer — it is not a looseness allowance.
    ///
    /// NOT asserted: `r0_hi > 0` numerically. `r0_lo` / `r0_hi` are locals of
    /// `reml_deriv_enclosure_profile`, and recomputing them here would make the
    /// gate a copy of the rule it guards. The regime is pinned instead by the
    /// perfect fit itself — reported through the evaluator's own
    /// `dispersion_residual_parts` — together with the exact-cancellation CONTROL
    /// carried as the other side of the cliff. A fixture that drifts to a
    /// non-interpolating design trips the regime clause rather than going green.
    #[test]
    fn point_enclosure_must_sign_an_order_one_derivative_2694_2703() {
        // The smallest `|V′|` at which the gate demands a SIGN. It selects which
        // ρ the sharpness property is asserted at — near a genuine stationary
        // point an enclosure straddling zero is the correct answer — and it is
        // not a looseness allowance: the property itself compares the enclosure's
        // width against `|V′|` and carries no constant. Sits well above the
        // roundoff floor and well below every `|V′|` either fixture produces
        // (CONTROL ≈ 0.5, WITNESS ≈ 1.0).
        const ORDER_ONE_DERIVATIVE: f64 = 0.25;

        fn point_check(
            tag: &str,
            prepared: &GaussianRemlPrepared,
            rho: f64,
            failures: &mut Vec<String>,
        ) -> bool {
            let exact = prepared.evaluate(rho);
            let (dv, _) = reml_deriv_enclosure(
                &prepared.cache,
                prepared.ywy.view(),
                prepared.projected_rhs_squared.view(),
                prepared.n_effective,
                prepared.n_outputs,
                rho,
                rho,
            );
            // Soundness first: a bound that does not contain the value it bounds
            // is a different defect and would make the sharpness reading
            // meaningless.
            if !(dv.lo <= exact.grad && exact.grad <= dv.hi) {
                failures.push(format!(
                    "{tag} rho={rho}: SOUNDNESS — the zero-width enclosure \
                     [{:.9e}, {:.9e}] does not contain the evaluator's own \
                     V'={:.9e}",
                    dv.lo, dv.hi, exact.grad
                ));
                return false;
            }
            if exact.grad.abs() < ORDER_ONE_DERIVATIVE {
                return false;
            }
            if dv.hi - dv.lo > exact.grad.abs() {
                failures.push(format!(
                    "{tag} rho={rho}: SHARPNESS — V'={:.9e} but the ZERO-WIDTH \
                     enclosure is [{:.9e}, {:.9e}], width {:.9e}. A width larger \
                     than the value it encloses cannot sign that value, so no \
                     cell containing this point can ever satisfy the \
                     branch-and-bound's `dv.lo > 0 || dv.hi < 0` prune test, at \
                     any subdivision depth.",
                    exact.grad,
                    dv.lo,
                    dv.hi,
                    dv.hi - dv.lo
                ));
            }
            true
        }

        let mut failures: Vec<String> = Vec::new();

        // ---- CONTROL: a perfect fit whose cancellation is EXACT -------------
        // `y` is exactly `x · [1, 2]` in integers, so `ywy` and `Σc²` agree to
        // the bit, `r0 = [0, 0]` has zero WIDTH, and the enclosure is tight.
        // This is the neighbouring regime in which the defect provably cannot
        // appear, and it is what makes a green on the witness meaningful.
        let control_x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
        let control_y = array![[1.0], [3.0], [5.0], [7.0], [9.0]];
        let control_penalty = array![[0.0, 0.0], [0.0, 1.0]];
        let control = prepare_gaussian_reml(
            control_x.view(),
            control_y.view(),
            control_penalty.view(),
            None,
            None,
            None,
        )
        .expect("the control design is finite and full rank");
        let mut control_asserted = 0usize;
        for rho in [RHO_LOWER, -10.0, 0.0, 10.0] {
            if point_check("CONTROL", &control, rho, &mut failures) {
                control_asserted += 1;
            }
        }
        if control_asserted == 0 {
            failures.push(
                "CONTROL: no rho carried an order-one V', so the sharpness \
                 property was never asserted on the passing side — the gate's \
                 instrument did not engage"
                    .to_string(),
            );
        }

        // ---- WITNESS: the #2694 harvest regime ------------------------------
        // A constant response on a periodic-harmonic basis. The constant is the
        // basis's own first column (so the fit is exact) AND spans the penalty's
        // null space (so no penalized direction carries any mass). The basis
        // values are irrational, so `Σc²` reaches `ywy` through a different
        // rounding path than the direct `ywy` sum and the cancellation is
        // INEXACT — which is the part a small integer design cannot produce.
        let n = 12usize;
        let witness_x = Array2::<f64>::from_shape_fn((n, 3), |(row, col)| {
            let t = 2.0 * std::f64::consts::PI * (row as f64) / (n as f64);
            match col {
                0 => 1.0,
                1 => t.sin(),
                _ => t.cos(),
            }
        });
        let witness_y = Array2::<f64>::from_elem((n, 1), 0.7);
        let witness_penalty = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let witness = prepare_gaussian_reml(
            witness_x.view(),
            witness_y.view(),
            witness_penalty.view(),
            None,
            None,
            None,
        )
        .expect("the witness design is finite and full rank");

        // Regime clause. The necessary condition is the PERFECT FIT: the
        // rho-dependent part of the deviance must be vanishing relative to
        // `ywy`, which is what drives `ywy − Σc²` into cancellation. Reported
        // through the evaluator's own decomposition rather than recomputed.
        let DispersionResidualParts {
                unpenalized_residual,
                penalized_residual,
                ..
            } = dispersion_residual_parts(
            &witness.cache,
            witness.ywy.view(),
            witness.projected_rhs_squared.view(),
            0,
            RHO_LOWER,
        );
        let ywy = witness.ywy[0];
        if !(penalized_residual >= 0.0 && penalized_residual < 1.0e-25 * ywy) {
            failures.push(format!(
                "WITNESS regime: the rho-dependent deviance at rho={RHO_LOWER} is \
                 {penalized_residual:.9e} against ywy={ywy:.9e}; this design does \
                 not interpolate its response, so `ywy − Σc²` never cancels and \
                 the fixture has drifted OUT of the regime under test — a pass \
                 below would mean nothing"
            ));
        }
        let mut witness_asserted = 0usize;
        for rho in [RHO_LOWER, -25.0, -20.0] {
            if point_check("WITNESS", &witness, rho, &mut failures) {
                witness_asserted += 1;
            }
        }
        if witness_asserted == 0 {
            failures.push(
                "WITNESS: no rho carried an order-one V', so the sharpness \
                 property was never asserted on the failing side — the gate's \
                 instrument did not engage"
                    .to_string(),
            );
        }

        // Engagement report. A green here is only meaningful if the sharpness
        // property was actually asserted, so the counts are printed rather than
        // left to be inferred from the absence of a panic.
        println!(
            "[2694-gate] CONTROL asserted at {control_asserted} rho, WITNESS \
             asserted at {witness_asserted} rho, failed clauses {}, witness \
             ywy={ywy:.9e} unpenalized_residual={unpenalized_residual:.9e} \
             penalized_residual={penalized_residual:.9e}",
            failures.len()
        );

        assert!(
            failures.is_empty(),
            "#2703/#2694 REGRESSION — the profiled-REML derivative enclosure has \
             lost its sharpness on an exactly-interpolating design.\n\
             This gate was red when it landed and went green with the repair in \
             `reml_deriv_enclosure_profile`: `r0` is ρ-INDEPENDENT, so it enters \
             the enclosure as the single value the evaluator uses, not as a \
             bracket over the digits its cancellation destroyed. Bracketing it \
             put `dp_lo` on the ratio numerator's own scale, pinned \
             `num_hi/dp_lo` at `1.0` whatever the data, and gave a ZERO-WIDTH \
             enclosure of width `half_nu`. If you are seeing this, check that \
             change first.\n\
             witness ywy={ywy:.9e} unpenalized_residual={unpenalized_residual:.9e} \
             penalized_residual={penalized_residual:.9e}\n{}",
            failures.join("\n")
        );
    }

    /// #2703 — the ρ enumerator must RESOLVE an objective that is flat toward the
    /// small-λ box endpoint, and must still REFUSE structure it genuinely cannot
    /// resolve at its own resolution.
    ///
    /// The six `gam-sae inference::` failures this issue was filed on all carried
    /// ONE byte-identical refusal — "stationary structure remained non-monotone on
    /// `[-30, -29.999999999972715]` … deepest bisection 41" — and `60/2^41` is
    /// exactly that interval's width, so the search descended a SINGLE branch to
    /// the resolution floor at the lower rail: at every level the right half was
    /// pruned and the left half could not be.
    ///
    /// The cause was not a missing corner case in the enumerator. It was an
    /// enclosure that could not tighten. `r0 = ywy − Σc²` was BRACKETED over the
    /// digits its cancellation destroys; `r0` is ρ-INDEPENDENT, so that width
    /// survived every bisection — it was present at ZERO cell width — `dp_lo`
    /// collapsed onto the ratio numerator's own scale, `num_hi/dp_lo` read `1.0`
    /// whatever the data, and `V′`'s enclosure came out `half_nu` wide. An
    /// enclosure wider than the value it encloses can never satisfy
    /// `dv.lo > 0 || dv.hi < 0`, at any depth, so nothing could prune and the
    /// walk to the floor was structurally forced. The repair carries `r0` as the
    /// point the evaluator itself uses (see the comment at `r0` in
    /// `reml_deriv_enclosure_profile`).
    ///
    /// The gate beside this one pins the enclosure's sharpness directly. This one
    /// pins the CONSEQUENCE, at the enumerator, because that is where #2703 was
    /// observed and where a future regression would surface:
    ///
    /// * WITNESS — an interpolating design in the same regime as the six fixtures:
    ///   a constant response lying in the span of the basis AND in the null space
    ///   of the penalty, on an irrational (periodic-harmonic) basis so the
    ///   cancellation is INEXACT rather than bit-exact. The full production
    ///   branch-and-bound must return a SELECTION on it rather than the
    ///   unresolvable-structure refusal. A regime clause asserts the design still
    ///   interpolates, so a fixture that drifts out of the regime fails loudly
    ///   instead of going silently green. Measured: it selects `ρ = RHO_UPPER`,
    ///   a RAIL answer — reached rather than refused, which is the whole verdict
    ///   #2703 was denied. The clause does not assert WHICH rail: the search
    ///   certifies no interior stationary point, not a direction.
    /// * POSITIVE CONTROL — the refusal must survive. A guard that can no longer
    ///   fire is the defect one level up, and the enclosure repair is only safe
    ///   because it changes the enclosure's TIGHTNESS and not when the enumerator
    ///   returns `Ok`. An objective whose stationary points are spaced BELOW the
    ///   search's own bracket resolution genuinely cannot be enumerated at that
    ///   resolution, and must still be refused with that verdict.
    /// * NEGATIVE CONTROL — the same synthetic family with the oscillation removed
    ///   is smooth and unimodal, and must be ACCEPTED at its analytic root. Without
    ///   it, a harness that refused everything would read as a passing positive
    ///   control.
    ///
    /// Nothing here is a wall-clock budget or an invented tolerance: the positive
    /// control's frequency is derived from `RHO_BRACKET_RESOLUTION` — the search's
    /// own certified resolution — its amplitude from that frequency, and the
    /// negative control's admissible offset from the search's own bracket-width
    /// acceptance rule.
    #[test]
    fn rho_enumeration_resolves_the_small_lambda_rail_and_still_refuses_unresolvable_structure_2703()
    {
        let mut failures: Vec<String> = Vec::new();

        // ---- WITNESS: the #2703 interpolating regime, at the ENUMERATOR ------
        let n = 12usize;
        let witness_x = Array2::<f64>::from_shape_fn((n, 3), |(row, col)| {
            let t = 2.0 * std::f64::consts::PI * (row as f64) / (n as f64);
            match col {
                0 => 1.0,
                1 => t.sin(),
                _ => t.cos(),
            }
        });
        let witness_y = Array2::<f64>::from_elem((n, 1), 0.7);
        let witness_penalty = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let witness = prepare_gaussian_reml(
            witness_x.view(),
            witness_y.view(),
            witness_penalty.view(),
            None,
            None,
            None,
        )
        .expect("the witness design is finite and full rank");

        // Regime clause, reported through the evaluator's own decomposition: the
        // ρ-dependent part of the deviance must be vanishing against `ywy`, which
        // is what drives `ywy − Σc²` into cancellation in the first place.
        let DispersionResidualParts {
            unpenalized_residual,
            penalized_residual,
            ..
        } = dispersion_residual_parts(
            &witness.cache,
            witness.ywy.view(),
            witness.projected_rhs_squared.view(),
            0,
            RHO_LOWER,
        );
        let ywy = witness.ywy[0];
        if !(penalized_residual >= 0.0 && penalized_residual < 1.0e-25 * ywy) {
            failures.push(format!(
                "WITNESS regime: the rho-dependent deviance at rho={RHO_LOWER} is \
                 {penalized_residual:.9e} against ywy={ywy:.9e}; this design does not \
                 interpolate its response, so `ywy − Σc²` never cancels and the \
                 fixture has drifted OUT of the regime under test — a pass below \
                 would mean nothing"
            ));
        }

        let witness_eval = |rho: f64| witness.evaluate(rho);
        let witness_enclose = |a: f64, b: f64| {
            reml_deriv_enclosure(
                &witness.cache,
                witness.ywy.view(),
                witness.projected_rhs_squared.view(),
                witness.n_effective,
                witness.n_outputs,
                a,
                b,
            )
        };
        let mut witness_rho = f64::NAN;
        match enumerate_and_select_rho_with_controls(
            &witness_eval,
            &witness_enclose,
            None,
            ProfileSearchControls::PRODUCTION,
            None,
        ) {
            Ok(selection) => {
                witness_rho = selection.rho;
                let at_selected = witness_eval(selection.rho).cost;
                let at_lower = witness_eval(RHO_LOWER).cost;
                let at_upper = witness_eval(RHO_UPPER).cost;
                if !(selection.rho.is_finite()
                    && selection.rho >= RHO_LOWER
                    && selection.rho <= RHO_UPPER)
                {
                    failures.push(format!(
                        "WITNESS: the selected rho={} is not inside the search window \
                         [{RHO_LOWER}, {RHO_UPPER}]",
                        selection.rho
                    ));
                }
                if !(at_selected <= at_lower && at_selected <= at_upper) {
                    failures.push(format!(
                        "WITNESS: the selection is not the best candidate the search \
                         saw — cost {at_selected:.9e} at rho={} against {at_lower:.9e} \
                         at the lower rail and {at_upper:.9e} at the upper rail",
                        selection.rho
                    ));
                }
            }
            Err(error) => failures.push(format!(
                "WITNESS: the production branch-and-bound REFUSED an interpolating \
                 design — this is the #2703 symptom itself: {error}"
            )),
        }

        // ---- the synthetic family shared by both controls -------------------
        // `V(ρ) = ½(ρ − CENTRE)² + amplitude·sin(wavenumber·ρ)` with SOUND
        // (superset) enclosures of `V′` and `V″` over any cell: `V′` is bounded by
        // its linear part over `[a, b]` widened by the oscillation's own swing
        // `|amplitude·wavenumber|`, and `V″` by `1 ± |amplitude·wavenumber²|`.
        // Passing the enumerator an objective directly is what lets the controls
        // state the STRUCTURE under test rather than hunt for a design that
        // happens to have it.
        const CENTRE: f64 = 0.5;
        let objective = |amplitude: f64, wavenumber: f64| {
            move |rho: f64| {
                let phase = wavenumber * rho;
                ObjectiveEval {
                    cost: 0.5 * (rho - CENTRE) * (rho - CENTRE) + amplitude * phase.sin(),
                    grad: (rho - CENTRE) + amplitude * wavenumber * phase.cos(),
                    hess: 1.0 - amplitude * wavenumber * wavenumber * phase.sin(),
                    edf: 0.0,
                    // The synthetic objective is closed-form and exactly
                    // representable, so its cost carries no accumulated
                    // forward error to declare (#2729).
                    cost_roundoff: 0.0,
                }
            }
        };
        let enclosure = |amplitude: f64, wavenumber: f64| {
            move |a: f64, b: f64| {
                let grad_swing = (amplitude * wavenumber).abs();
                let hess_swing = (amplitude * wavenumber * wavenumber).abs();
                (
                    Interval {
                        lo: (a - CENTRE) - grad_swing,
                        hi: (b - CENTRE) + grad_swing,
                    },
                    Interval {
                        lo: 1.0 - hess_swing,
                        hi: 1.0 + hess_swing,
                    },
                )
            }
        };

        // ---- POSITIVE CONTROL: the refusal must still fire -------------------
        // One oscillation per QUARTER of the search's own bracket resolution, so
        // consecutive stationary points are closer together than the finest
        // bracket the search is permitted to certify — "unresolvable" stated in
        // the search's own units. The amplitude then follows from
        // `amplitude·wavenumber = 1`: an order-one swing in `V′`, so the cells
        // near `CENTRE` genuinely cannot be signed.
        let unresolvable_wavenumber = std::f64::consts::TAU / (0.25 * RHO_BRACKET_RESOLUTION);
        let unresolvable_amplitude = 1.0 / unresolvable_wavenumber;
        let mut positive_control_verdict = String::new();
        match enumerate_and_select_rho_with_controls(
            objective(unresolvable_amplitude, unresolvable_wavenumber),
            enclosure(unresolvable_amplitude, unresolvable_wavenumber),
            None,
            ProfileSearchControls::PRODUCTION,
            None,
        ) {
            Ok(selection) => failures.push(format!(
                "POSITIVE CONTROL: the enumerator MINTED rho={} on an objective whose \
                 stationary points are spaced below its own bracket resolution \
                 ({RHO_BRACKET_RESOLUTION:e}). The unresolvable-structure refusal can \
                 no longer fire, which is a worse defect than the one #2703 reported",
                selection.rho
            )),
            Err(error) => {
                positive_control_verdict = error.to_string();
                if !positive_control_verdict.contains("remained non-monotone") {
                    failures.push(format!(
                        "POSITIVE CONTROL: refused, but not with the \
                         unresolvable-structure verdict: {positive_control_verdict}"
                    ));
                }
            }
        }

        // ---- NEGATIVE CONTROL: the same harness must be able to accept -------
        let mut negative_control_rho = f64::NAN;
        match enumerate_and_select_rho_with_controls(
            objective(0.0, unresolvable_wavenumber),
            enclosure(0.0, unresolvable_wavenumber),
            None,
            ProfileSearchControls::PRODUCTION,
            None,
        ) {
            Ok(selection) => {
                negative_control_rho = selection.rho;
                // The search's OWN acceptance rule: it stops when the bracket is
                // narrower than `resolution · (1 + max|endpoint|)` and returns a
                // point of that bracket, which also contains the analytic root.
                // The bracket's endpoints exceed the two points it contains by at
                // most its own width, hence the `+ RHO_BRACKET_RESOLUTION`.
                let scale =
                    1.0 + selection.rho.abs().max(CENTRE.abs()) + RHO_BRACKET_RESOLUTION;
                let admissible = RHO_BRACKET_RESOLUTION * scale;
                if (selection.rho - CENTRE).abs() > admissible {
                    failures.push(format!(
                        "NEGATIVE CONTROL: selected rho={} against the analytic root \
                         {CENTRE}, off by {:e} which exceeds the search's own bracket \
                         acceptance {admissible:e}",
                        selection.rho,
                        (selection.rho - CENTRE).abs()
                    ));
                }
            }
            Err(error) => failures.push(format!(
                "NEGATIVE CONTROL: the enumerator refused a smooth unimodal objective \
                 with an interior root at {CENTRE}, so the positive control's refusal \
                 above is attributable to the harness rather than to the structure: \
                 {error}"
            )),
        }

        // Engagement report: a green is only readable if each arm actually reached
        // its verdict, so the verdicts are printed rather than inferred from the
        // absence of a panic.
        println!(
            "[2703-gate] WITNESS selected rho={witness_rho:.9e} \
             (ywy={ywy:.9e} unpenalized_residual={unpenalized_residual:.9e} \
             penalized_residual={penalized_residual:.9e}), \
             POSITIVE CONTROL refusal={:?}, NEGATIVE CONTROL rho={negative_control_rho:.9e}, \
             failed clauses {}",
            positive_control_verdict
                .split(':')
                .next_back()
                .unwrap_or("")
                .trim(),
            failures.len()
        );

        assert!(
            failures.is_empty(),
            "#2703 REGRESSION — the 1-D REML rho enumerator no longer resolves a \
             small-lambda-flat objective, or no longer refuses one it cannot \
             resolve.\n\
             The six `gam-sae inference::` failures this gate stands for were ONE \
             cause: `r0` entered the derivative enclosure as a BRACKET over the \
             digits its cancellation destroys, the width was rho-INDEPENDENT and so \
             survived every bisection, and no cell could ever be pruned. If the \
             WITNESS clause is red, check `reml_deriv_enclosure_profile`'s `r0` \
             first. If a CONTROL clause is red, the guard's ability to fire has \
             moved, which is the more serious direction.\n{}",
            failures.join("\n")
        );
    }

    #[test]
    fn edf_does_not_double_count_penalty_nullspace() {
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0],];
        let y = array![[0.0], [1.0], [1.8], [3.2], [4.1]];
        let penalty = array![[0.0, 0.0], [0.0, 1.0]];
        let result =
            gaussian_reml_multi_closed_form(x.view(), y.view(), penalty.view(), None, Some(0.0))
                .expect("small full-rank Gaussian REML fit");

        assert!(result.edf >= result.cache.nullity as f64);
        assert!(result.edf <= x.ncols() as f64 + 1.0e-10);
    }

    /// #2496: this one-mode problem has an analytic interior optimum at λ=1.
    ///
    /// With `X=(1,0,0)'`, `y=(1,1,0)'`, and `S=(1)`, the fitted coefficient is
    /// `1/(1+λ)` and the profiled scale is
    /// `dp = ||y-Xβ||² + λβ'Sβ = 2 - 1/(1+λ)`. Differentiating the full REML
    /// objective gives its unique finite stationary minimum at λ=1. A profile
    /// that substitutes plain RSS for `dp` does not satisfy this identity.
    #[test]
    fn profiled_gaussian_reml_penalized_scale_selects_analytic_lambda_one_2496() {
        let x = array![[1.0], [0.0], [0.0]];
        let y = array![1.0, 1.0, 0.0];
        let penalty = array![[1.0]];
        let fit = gaussian_reml_closed_form_with_nullspace_dim(
            x.view(),
            y.view(),
            penalty.view(),
            Some(0),
            None,
            None,
        )
        .expect("analytic one-mode Gaussian REML profile");

        eprintln!(
            "[#2496] analytic profile: lambda={:.12e} rho={:.12e} sigma2={:.12e}",
            fit.lambda, fit.rho, fit.sigma2,
        );
        assert!(
            fit.rho.abs() <= 1.0e-9,
            "analytic optimum is rho=log(lambda)=0, got {}",
            fit.rho,
        );
        assert!((fit.lambda - 1.0).abs() <= 1.0e-9);
        assert!((fit.coefficients[0] - 0.5).abs() <= 1.0e-9);
        assert!((fit.sigma2 - 0.5).abs() <= 1.0e-9);
    }

    /// gam#2585: a SATURATED design (`p = n`, zero residual degrees of freedom)
    /// must not lose its profiled residual to cancellation.
    ///
    /// With `X = I_n` the unpenalized fit interpolates, so `r0 = ywy − Σ c²` is
    /// exactly `0` and the whole profiled residual is `dp(ρ) = Σ c²·u(ρ)`. At the
    /// ρ window's small-λ end that is tiny but strictly positive — here
    /// `u ≈ 9.4e−17`, small enough that `v = 1 − u` rounds to exactly `1.0`.
    /// Computing `dp` as the DIFFERENCE `ywy − Σ c²·v` therefore returns `0` (or
    /// a negative rounding artefact) and destroys the quantity outright: the
    /// domain check refuses the fit, and the interval enclosure collapses to the
    /// entire line, which the branch-and-bound can neither prune nor certify
    /// monotone and must therefore split.
    ///
    /// The summed decomposition `dp = r0 + Σ c²·u` has no cancellation in its
    /// ρ-dependent part, so both survive. Pinned here on both halves: the fit
    /// completes, and the leftmost cell's enclosure is finite AND contains the
    /// endpoint jets the evaluator actually produces — a tight enclosure that
    /// excluded them would be worse than a wide one.
    #[test]
    fn saturated_design_keeps_a_finite_small_lambda_enclosure_2585() {
        let n = 8usize;
        let mut x = Array2::<f64>::zeros((n, n));
        for i in 0..n {
            x[[i, i]] = 1.0;
        }
        let y =
            Array2::from_shape_vec((n, 1), vec![0.7, -1.3, 2.1, 0.4, -0.9, 1.6, -0.2, 1.1])
                .expect("saturated response");
        // Small penalty eigenvalues put the small-λ end of the window deep
        // enough that `1 − u` is not representable: `u ≈ e^(−30)·1e−3`.
        let mut penalty = Array2::<f64>::zeros((n, n));
        for i in 0..n - 1 {
            penalty[[i, i]] = 1.0e-3;
        }

        let prepared =
            prepare_gaussian_reml(x.view(), y.view(), penalty.view(), None, None, None)
                .expect("saturated design must still prepare");

        let a = RHO_LOWER;
        let b = RHO_LOWER + 1.0e-3;
        let (dv, dvv) = reml_deriv_enclosure(
            &prepared.cache,
            prepared.ywy.view(),
            prepared.projected_rhs_squared.view(),
            prepared.n_effective,
            prepared.n_outputs,
            a,
            b,
        );
        assert!(
            dv.lo.is_finite() && dv.hi.is_finite(),
            "saturated small-lambda cell produced an unbounded V' enclosure [{}, {}]",
            dv.lo,
            dv.hi
        );
        assert!(
            dvv.lo.is_finite() && dvv.hi.is_finite(),
            "saturated small-lambda cell produced an unbounded V'' enclosure [{}, {}]",
            dvv.lo,
            dvv.hi
        );
        for rho in [a, b] {
            let jet = prepared.evaluate(rho);
            assert!(
                interval_contains(dv, jet.grad),
                "V' enclosure [{}, {}] missed the endpoint gradient {} at rho={rho}",
                dv.lo,
                dv.hi,
                jet.grad
            );
            assert!(
                interval_contains(dvv, jet.hess),
                "V'' enclosure [{}, {}] missed the endpoint curvature {} at rho={rho}",
                dvv.lo,
                dvv.hi,
                jet.hess
            );
        }

        // The profiled objective itself must stay defined at the window edge.
        // Through the cancelling difference this is exactly `0` — `v` rounds to
        // `1.0`, `Σ c²·v` reaches `ywy`, and `log(dp)` is `-inf`; through the
        // summed decomposition it is small but finite, so the cost, gradient and
        // curvature are all real numbers.
        let edge = prepared.evaluate(RHO_LOWER);
        assert!(
            edge.cost.is_finite() && edge.grad.is_finite() && edge.hess.is_finite(),
            "saturated small-lambda jet is not finite: cost={} grad={} hess={}",
            edge.cost,
            edge.grad,
            edge.hess
        );
        let sigma2 = prepared.sigma2(RHO_LOWER);
        assert!(
            sigma2.iter().all(|v| v.is_finite() && *v > 0.0),
            "saturated profiled dispersion collapsed to {sigma2:?}"
        );

        // Deliberately NOT asserted: that the certified search returns a ρ̂ here.
        // `p = n` forces `penalty_rank = p − nullity = n − nullity = ν`, and with
        // that equality `V′(ρ) → ½(ν − rank) = 0` as ρ → −∞ identically. On this
        // 8×8 fixture `V′(−30)` is 4e−16 — the profile is stationary at the
        // window edge to machine precision, so no enclosure can certify a sign
        // and a typed refusal is the honest verdict. That is a statement about
        // the estimand, not about this file's arithmetic, and it is the reason
        // the assertions above are about the ENCLOSURE and the JET rather than
        // about a returned λ̂.
    }

    /// Profiling must be invariant to both gauges of the same penalized
    /// function: scaling `S -> αS` translates `rho -> rho-log(α)`, while the
    /// coefficient-chart change `X -> X/c`, `S -> S/c²`, `β -> cβ` leaves rho
    /// unchanged. In both cases the fitted function and REML evidence are the
    /// same. This is a regression on the canonical certified solver, not a
    /// second implementation of its objective or λ search.
    #[test]
    fn profiled_gaussian_reml_is_penalty_scale_and_coefficient_chart_invariant_2496() {
        let x = array![[1.0], [0.0], [0.0]];
        let y = array![1.0, 1.0, 0.0];
        let penalty = array![[1.0]];
        let baseline = gaussian_reml_closed_form_with_nullspace_dim(
            x.view(),
            y.view(),
            penalty.view(),
            Some(0),
            None,
            None,
        )
        .expect("baseline analytic Gaussian REML profile");

        for alpha in [1.0e-3_f64, 37.0, 1.0e4] {
            let scaled_penalty = penalty.mapv(|value| alpha * value);
            let scaled = gaussian_reml_closed_form_with_nullspace_dim(
                x.view(),
                y.view(),
                scaled_penalty.view(),
                Some(0),
                None,
                Some(baseline.rho - alpha.ln()),
            )
            .expect("penalty-scaled Gaussian REML profile");
            let score_tolerance = 1.0e-9 * (1.0 + baseline.reml_score.abs());
            assert!(
                (scaled.reml_score - baseline.reml_score).abs() <= score_tolerance,
                "S -> alpha S changed profiled evidence at alpha={alpha}: baseline={}, scaled={}",
                baseline.reml_score,
                scaled.reml_score,
            );
            assert!(
                (scaled.rho - (baseline.rho - alpha.ln())).abs() <= 1.0e-9,
                "S -> alpha S must shift rho by -log(alpha) at alpha={alpha}: baseline={}, scaled={}",
                baseline.rho,
                scaled.rho,
            );
            assert!(
                (alpha * scaled.lambda - baseline.lambda).abs() <= 1.0e-9,
                "physical lambda*S changed at alpha={alpha}",
            );
            for row in 0..x.nrows() {
                assert!((scaled.fitted[row] - baseline.fitted[row]).abs() <= 1.0e-9);
            }
        }

        let coefficient_scale = 7.0_f64;
        let reparameterized_x = x.mapv(|value| value / coefficient_scale);
        let reparameterized_penalty =
            penalty.mapv(|value| value / coefficient_scale.powi(2));
        let reparameterized = gaussian_reml_closed_form_with_nullspace_dim(
            reparameterized_x.view(),
            y.view(),
            reparameterized_penalty.view(),
            Some(0),
            None,
            Some(baseline.rho),
        )
        .expect("coefficient-reparameterized Gaussian REML profile");
        let score_tolerance = 1.0e-9 * (1.0 + baseline.reml_score.abs());
        assert!(
            (reparameterized.reml_score - baseline.reml_score).abs() <= score_tolerance
        );
        assert!((reparameterized.rho - baseline.rho).abs() <= 1.0e-9);
        assert!(
            (reparameterized.coefficients[0]
                - coefficient_scale * baseline.coefficients[0])
                .abs()
                <= 1.0e-9
        );
        for row in 0..x.nrows() {
            assert!((reparameterized.fitted[row] - baseline.fitted[row]).abs() <= 1.0e-9);
        }
        eprintln!(
            "[#2496] gauges: base_rho={:.12e}, chart_rho={:.12e}, score={:.12e}",
            baseline.rho, reparameterized.rho, baseline.reml_score,
        );
    }

    #[test]
    fn shared_dispersion_pools_projection_exact_and_missed_outputs() {
        let n = 12usize;
        let mut x = Array2::<f64>::zeros((n, 2));
        let mut y = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            let t = row as f64 - 5.5;
            x[[row, 0]] = 1.0;
            x[[row, 1]] = t;
            // The first ambient output is exactly the chart coordinate: this is
            // the tautological zero-residual channel a PCA chart creates.
            y[[row, 0]] = t;
            // The second output is deliberately outside the linear chart.
            y[[row, 1]] = if row % 2 == 0 { -2.0 } else { 3.0 };
        }
        let penalty = Array2::<f64>::zeros((2, 2));
        let fit = gaussian_reml_multi_shared_dispersion_closed_form(
            x.view(),
            y.view(),
            penalty.view(),
            None,
            None,
        )
        .expect("shared-dispersion vector REML fit");

        assert_eq!(fit.sigma2[0].to_bits(), fit.sigma2[1].to_bits());
        let mut pooled_rss = 0.0_f64;
        for row in 0..n {
            for output in 0..2 {
                let residual = y[[row, output]] - fit.fitted[[row, output]];
                pooled_rss += residual * residual;
            }
        }
        let shared_nu = (2 * (n - fit.cache.nullity)) as f64;
        let expected_sigma2 = pooled_rss / shared_nu;
        assert!(expected_sigma2 > 0.0);
        assert!(
            (fit.sigma2[0] - expected_sigma2).abs()
                <= f64::EPSILON.sqrt() * expected_sigma2.max(1.0),
            "shared sigma2 {} must equal pooled vector deviance / shared dof {}",
            fit.sigma2[0],
            expected_sigma2
        );
    }

    #[test]
    fn shared_dispersion_penalty_envelope_gradient_matches_refitted_direction() {
        let n = 24usize;
        let mut x = Array2::<f64>::zeros((n, 3));
        let mut y = Array2::<f64>::zeros((n, 2));
        for row in 0..n {
            let t = -1.0 + 2.0 * row as f64 / (n - 1) as f64;
            x[[row, 0]] = 1.0;
            x[[row, 1]] = t;
            x[[row, 2]] = t * t;
            y[[row, 0]] = 0.3 + 1.2 * t - 0.8 * t * t + 0.04 * (7.0 * t).sin();
            y[[row, 1]] = -0.2 + 0.5 * t + 0.4 * t * t + 0.03 * (5.0 * t).cos();
        }
        let penalty = array![[0.0, 0.0, 0.0], [0.0, 0.7, 0.1], [0.0, 0.1, 1.4]];
        let direction = array![[0.0, 0.0, 0.0], [0.0, 0.3, -0.08], [0.0, -0.08, 0.6]];
        let fit = gaussian_reml_multi_shared_dispersion_closed_form(
            x.view(),
            y.view(),
            penalty.view(),
            None,
            None,
        )
        .unwrap();
        let gradient = gaussian_reml_multi_shared_dispersion_penalty_gradient_from_fit(
            x.view(),
            y.view(),
            penalty.view(),
            None,
            &fit,
        )
        .unwrap();
        let analytic = gradient
            .iter()
            .zip(direction.iter())
            .map(|(gradient, direction)| gradient * direction)
            .sum::<f64>();

        let step = f64::EPSILON.cbrt();
        let plus_penalty = &penalty + &(direction.mapv(|value| step * value));
        let minus_penalty = &penalty - &(direction.mapv(|value| step * value));
        let plus = gaussian_reml_multi_shared_dispersion_closed_form(
            x.view(),
            y.view(),
            plus_penalty.view(),
            None,
            Some(fit.rho),
        )
        .unwrap();
        let minus = gaussian_reml_multi_shared_dispersion_closed_form(
            x.view(),
            y.view(),
            minus_penalty.view(),
            None,
            Some(fit.rho),
        )
        .unwrap();
        let numerical = (plus.reml_score - minus.reml_score) / (2.0 * step);
        let scale = analytic.abs().max(numerical.abs()).max(1.0);
        assert!(
            (analytic - numerical).abs() <= 2.0e-5 * scale,
            "shared-dispersion penalty envelope derivative mismatch: analytic={analytic}, refitted={numerical}"
        );
    }

    #[test]
    fn block_orthogonal_score_matches_the_objective_derivative() {
        let gram = array![[3.0, 0.4], [0.4, 2.0]];
        let rhs = array![[1.2, -0.3], [0.6, 0.9]];
        let penalty = array![[1.0, 0.2], [0.2, 0.8]];
        let scale = array![1.3, 0.8];
        let rho = 0.37;
        let step = 1.0e-6;
        let eval = block_orthogonal_eval(&gram, &rhs, &penalty, rho).unwrap();
        let analytic = block_orthogonal_scale_objective(&eval, rho, scale.view(), 2).grad;
        let value_at = |candidate_rho: f64| {
            let candidate = block_orthogonal_eval(&gram, &rhs, &penalty, candidate_rho).unwrap();
            block_orthogonal_scale_objective(&candidate, candidate_rho, scale.view(), 2).value
        };
        let numerical = (value_at(rho + step) - value_at(rho - step)) / (2.0 * step);
        assert!(
            (analytic - numerical).abs() <= 1.0e-7 * analytic.abs().max(1.0),
            "analytic score {analytic:.12e} != objective derivative {numerical:.12e}"
        );
    }

    #[test]
    fn block_orthogonal_profile_hessian_matches_the_profiled_objective() {
        let grams = [
            array![[3.0, 0.4], [0.4, 2.0]],
            array![[2.5, -0.2], [-0.2, 1.8]],
        ];
        let rhs = [
            array![[1.2, -0.3], [0.6, 0.9]],
            array![[0.5, 0.8], [-0.4, 0.7]],
        ];
        let penalties = [
            array![[1.0, 0.2], [0.2, 0.8]],
            array![[0.9, -0.1], [-0.1, 1.1]],
        ];
        let ranks = [2_usize, 2_usize];
        let ywy = array![8.0, 9.0];
        let nu = 7.0;
        let rhos = array![0.37, -0.21];
        let profile_value = |candidate_rhos: ArrayView1<'_, f64>| {
            let evals = (0..2)
                .map(|block| {
                    block_orthogonal_eval(
                        &grams[block],
                        &rhs[block],
                        &penalties[block],
                        candidate_rhos[block],
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let mut q = ywy.clone();
            for eval in &evals {
                q -= &eval.fitted_energy;
            }
            let determinant_term = evals
                .iter()
                .enumerate()
                .map(|(block, eval)| eval.logdet - ranks[block] as f64 * candidate_rhos[block])
                .sum::<f64>();
            0.5 * 2.0 * determinant_term + 0.5 * nu * q.iter().map(|value| value.ln()).sum::<f64>()
        };
        let evals = (0..2)
            .map(|block| {
                block_orthogonal_eval(&grams[block], &rhs[block], &penalties[block], rhos[block])
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let scale = block_orthogonal_conditional_scale(&evals, ywy.view(), nu).unwrap();
        let analytic =
            block_orthogonal_profile_hessian(&evals, rhos.view(), scale.view(), &ranks, nu)
                .unwrap();
        let step = 1.0e-4;
        let center = profile_value(rhos.view());
        let mut numerical = Array2::<f64>::zeros((2, 2));
        for coordinate in 0..2 {
            let mut plus = rhos.clone();
            let mut minus = rhos.clone();
            plus[coordinate] += step;
            minus[coordinate] -= step;
            numerical[[coordinate, coordinate]] = (profile_value(plus.view()) - 2.0 * center
                + profile_value(minus.view()))
                / (step * step);
        }
        let mut plus_plus = rhos.clone();
        let mut plus_minus = rhos.clone();
        let mut minus_plus = rhos.clone();
        let mut minus_minus = rhos.clone();
        plus_plus[0] += step;
        plus_plus[1] += step;
        plus_minus[0] += step;
        plus_minus[1] -= step;
        minus_plus[0] -= step;
        minus_plus[1] += step;
        minus_minus[0] -= step;
        minus_minus[1] -= step;
        let cross = (profile_value(plus_plus.view())
            - profile_value(plus_minus.view())
            - profile_value(minus_plus.view())
            + profile_value(minus_minus.view()))
            / (4.0 * step * step);
        numerical[[0, 1]] = cross;
        numerical[[1, 0]] = cross;
        for row in 0..2 {
            for col in 0..2 {
                assert!(
                    (analytic[[row, col]] - numerical[[row, col]]).abs()
                        <= 2.0e-6 * analytic[[row, col]].abs().max(1.0),
                    "profile Hessian ({row}, {col}) analytic {:.12e} != numerical {:.12e}",
                    analytic[[row, col]],
                    numerical[[row, col]]
                );
            }
        }
    }

    #[test]
    fn block_orthogonal_shared_scale_fit_carries_a_score_certificate() {
        // Two mutually orthogonal ±1 blocks (Hadamard columns) with full-rank
        // penalties. A minted fit must satisfy the joint first-order REML
        // score certificate at its own returned iterate — re-derived here from
        // the same production primitives the solver certifies with, so a
        // regression that lets an iteration cap select the estimator fails.
        let c0 = [1.0_f64; 8];
        let c1 = [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0];
        let c2 = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0];
        let c3 = [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];
        let mut d1 = Array2::<f64>::zeros((8, 2));
        let mut d2 = Array2::<f64>::zeros((8, 2));
        for i in 0..8 {
            d1[[i, 0]] = c0[i];
            d1[[i, 1]] = c1[i];
            d2[[i, 0]] = c2[i];
            d2[[i, 1]] = c3[i];
        }
        let penalties = vec![Array2::<f64>::eye(2), Array2::<f64>::eye(2)];
        let bumps = [0.03, -0.05, 0.02, 0.01, -0.02, 0.04, -0.01, -0.02];
        let mut y = Array2::<f64>::zeros((8, 1));
        for i in 0..8 {
            y[[i, 0]] = c0[i] + 0.5 * c1[i] + 0.25 * c2[i] + bumps[i];
        }

        let result = gaussian_reml_blocks_orthogonal_shared_scale(
            &[d1.clone(), d2.clone()],
            &penalties,
            y.view(),
            None,
            None,
        )
        .expect("well-posed orthogonal-block fit must certify and mint");

        let weight = Array1::<f64>::ones(8);
        let ywy = (0..8).map(|i| y[[i, 0]] * y[[i, 0]]).sum::<f64>();
        // Full-rank penalties: zero total nullity, so nu = n.
        let nu = 8.0_f64;
        let mut evals = Vec::new();
        for (block, design) in [&d1, &d2].into_iter().enumerate() {
            let gram = canonicalize_penalty(dense_xt_diag_x(design.view(), weight.view()).view());
            let rhs = dense_xt_diag_y(design.view(), weight.view(), y.view());
            let pen = canonicalize_penalty(penalties[block].view());
            evals.push(
                block_orthogonal_eval(&gram, &rhs, &pen, result.log_lambdas[block])
                    .expect("block eval at the minted rho"),
            );
        }
        let explained: f64 = evals.iter().map(|eval| eval.fitted_energy[0]).sum();
        let q = ywy - explained;
        assert!(q > 0.0);
        let scale = Array1::from_vec(vec![nu / q]);
        for (block, eval) in evals.iter().enumerate() {
            let derivs =
                block_orthogonal_scale_objective(eval, result.log_lambdas[block], scale.view(), 2);
            let residual = derivs.grad.abs() / 2.0;
            assert!(
                residual <= BLOCK_ORTHOGONAL_SCORE_TOL,
                "block {block} score residual {residual:.3e} exceeds the certificate tolerance"
            );
        }
        let curvature = block_orthogonal_profile_spectrum(
            &block_orthogonal_profile_hessian(
                &evals,
                result.log_lambdas.view(),
                scale.view(),
                &[2, 2],
                nu,
            )
            .unwrap(),
        )
        .unwrap()
        .curvature;
        assert!(
            curvature.min_eigenvalue >= -curvature.roundoff,
            "minted fit has negative profiled curvature {:.6e} beyond roundoff {:.3e}",
            curvature.min_eigenvalue,
            curvature.roundoff
        );

        let err = gaussian_reml_blocks_orthogonal_shared_scale_with_controls(
            &[d1, d2],
            &penalties,
            y.view(),
            None,
            None,
            BlockOrthogonalControls {
                max_outer_passes: 0,
                ..BlockOrthogonalControls::default()
            },
        )
        .unwrap_err();
        match err {
            EstimationError::BlockOrthogonalRemlDidNotConverge {
                iterations,
                max_score_residual,
                rho_checkpoint,
                ..
            } => {
                assert_eq!(iterations, 0);
                assert!(max_score_residual.is_infinite());
                assert_eq!(rho_checkpoint, vec![0.0, 0.0]);
            }
            other => panic!("expected typed block-orthogonal exhaustion, got {other}"),
        }
    }

    #[test]
    fn block_orthogonal_solver_rejects_cross_block_signal() {
        let first = array![[1.0], [1.0], [1.0], [1.0], [1.0], [1.0]];
        let second = array![[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]];
        let penalties = vec![Array2::<f64>::eye(1), Array2::<f64>::eye(1)];
        let y = array![[0.2], [0.8], [1.7], [3.1], [3.9], [5.2]];
        let err = gaussian_reml_blocks_orthogonal_shared_scale(
            &[first, second],
            &penalties,
            y.view(),
            None,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(&err, EstimationError::InvalidInput(_)),
            "nonorthogonal blocks must fail the decomposed-objective contract: {err}"
        );
        assert!(err.to_string().contains("weighted cross-product"));
    }

    #[test]
    fn multi_output_duplicate_columns_match_scalar_fit() {
        let x = array![
            [1.0, -1.0],
            [1.0, -0.5],
            [1.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.0],
            [1.0, 1.5],
        ];
        let y1 = array![0.5, 0.2, 0.0, 0.3, 1.1, 2.0];
        let y = Array2::from_shape_fn(
            (y1.len(), 2),
            |(i, j)| if j == 0 { y1[i] } else { 2.0 * y1[i] },
        );
        let penalty = array![[0.0, 0.0], [0.0, 1.0]];

        let scalar =
            gaussian_reml_closed_form(x.view(), y1.view(), penalty.view(), None, Some(0.0))
                .expect("scalar Gaussian REML fit");
        let multi =
            gaussian_reml_multi_closed_form(x.view(), y.view(), penalty.view(), None, Some(0.0))
                .expect("multi-output Gaussian REML fit");

        assert!((multi.rho - scalar.rho).abs() <= 1.0e-8);
        for i in 0..x.ncols() {
            assert!((multi.coefficients[[i, 0]] - scalar.coefficients[i]).abs() <= 1.0e-8);
            assert!((multi.coefficients[[i, 1]] - 2.0 * scalar.coefficients[i]).abs() <= 1.0e-8);
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum ForwardScalar {
        Lambda,
        RemlScore,
        Coefficient(usize, usize),
        Fitted(usize, usize),
        Edf,
    }

    fn finite_difference_design() -> Array2<f64> {
        Array2::from_shape_fn((20, 5), |(row, col)| {
            let t = (row as f64 - 9.5) / 10.0;
            match col {
                0 => 1.0,
                1 => t,
                2 => 0.5 * (3.0 * t * t - 1.0),
                3 => 0.5 * (5.0 * t * t * t - 3.0 * t),
                4 => (35.0 * t.powi(4) - 30.0 * t * t + 3.0) / 8.0,
                _ => unreachable!(),
            }
        })
    }

    fn finite_difference_response(outputs: usize) -> Array2<f64> {
        // The truth must NOT lie (essentially) in span(X). The 5-column design
        // is Legendre P_0..P_4, so a low-order polynomial + low-frequency sin
        // would be fit to near machine precision — driving σ² → 0, dp → 0,
        // and ∂score/∂y ≈ ν w r / dp → ∞. Central finite differences with
        // Richardson extrapolation cannot resolve such steep, highly-nonlinear
        // surfaces at 1e-6 relative because the truncation term scales with
        // f^(5)(y), which explodes in that regime. The high-frequency sin
        // below is well outside span(P_0..P_4) on t ∈ [-0.95, 0.95], leaving
        // a genuine residual (σ² ≈ 1e-3) and an interior REML optimum
        // (ρ ≈ -3) at which the analytic-vs-FD comparison is meaningful.
        Array2::from_shape_fn((20, outputs), |(row, output)| {
            let t = (row as f64 - 9.5) / 10.0;
            let phase = output as f64 + 1.0;
            0.2 + 0.25 * phase * t - 0.12 * t * t
                + (0.08 + 0.03 * phase) * (1.1 * t + 0.3 * phase).sin()
                + 0.05 * (7.0 * t + 0.5 * phase).sin()
        })
    }

    fn finite_difference_penalty() -> Array2<f64> {
        Array2::from_diag(&array![0.0, 0.8, 1.2, 1.7, 2.3])
    }

    fn finite_difference_weights() -> Array1<f64> {
        Array1::from_shape_fn(20, |row| {
            let t = (row as f64 - 9.5) / 10.0;
            1.0 + 0.025 * (1.1 * t).sin() + 0.01 * t
        })
    }

    /// Fallible forward-scalar probe. Returns `None` when the closed-form fit
    /// rejects the inputs — the relevant case being a penalty perturbation that
    /// pushes `S` out of the PSD cone (a single-entry central bump on a
    /// null-direction entry drives one eigenvalue slightly negative). Such a
    /// point has no well-defined REML objective, so the caller skips it rather
    /// than panicking.
    fn one_hot_objective_try(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        penalty: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
        target: ForwardScalar,
    ) -> Option<f64> {
        let fit = gaussian_reml_multi_closed_form_with_cache(
            x,
            y,
            penalty,
            Some(weights),
            Some(0.85),
            None,
        )
        .ok()?;
        Some(match target {
            ForwardScalar::Lambda => fit.lambda,
            ForwardScalar::RemlScore => fit.reml_score,
            ForwardScalar::Coefficient(row, col) => fit.coefficients[[row, col]],
            ForwardScalar::Fitted(row, col) => fit.fitted[[row, col]],
            ForwardScalar::Edf => fit.edf,
        })
    }

    fn one_hot_objective(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        penalty: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
        target: ForwardScalar,
    ) -> f64 {
        one_hot_objective_try(x, y, penalty, weights, target)
            .expect("finite-difference forward fit")
    }

    fn one_hot_backward(
        x: ArrayView2<'_, f64>,
        y: ArrayView2<'_, f64>,
        penalty: ArrayView2<'_, f64>,
        weights: ArrayView1<'_, f64>,
        target: ForwardScalar,
    ) -> GaussianRemlBackwardResult {
        let mut grad_coefficients = Array2::<f64>::zeros((x.ncols(), y.ncols()));
        let mut grad_fitted = Array2::<f64>::zeros(y.dim());
        let (grad_lambda, grad_score, grad_edf, coefficient_upstream, fitted_upstream) =
            match target {
                ForwardScalar::Lambda => (1.0, 0.0, 0.0, None, None),
                ForwardScalar::RemlScore => (0.0, 1.0, 0.0, None, None),
                ForwardScalar::Coefficient(row, col) => {
                    grad_coefficients[[row, col]] = 1.0;
                    (0.0, 0.0, 0.0, Some(grad_coefficients.view()), None)
                }
                ForwardScalar::Fitted(row, col) => {
                    grad_fitted[[row, col]] = 1.0;
                    (0.0, 0.0, 0.0, None, Some(grad_fitted.view()))
                }
                ForwardScalar::Edf => (0.0, 0.0, 1.0, None, None),
            };
        gaussian_reml_multi_closed_form_backward(
            x,
            y,
            penalty,
            Some(weights),
            Some(0.85),
            grad_lambda,
            coefficient_upstream,
            fitted_upstream,
            grad_score,
            grad_edf,
        )
        .expect("analytic backward VJP")
    }

    fn assert_fd_close(label: &str, analytic: f64, finite_difference: f64) {
        let rel_tol = 1.0e-6_f64;
        let abs_tol = 1.0e-6_f64;
        let tol = abs_tol.max(rel_tol * analytic.abs().max(finite_difference.abs()));
        let diff = (analytic - finite_difference).abs();
        assert!(
            diff <= tol,
            "{label}: analytic={analytic:.12e}, finite_difference={finite_difference:.12e}, diff={diff:.3e}, tol={tol:.3e}"
        );
    }

    fn adaptive_central_difference(mut eval: impl FnMut(f64) -> f64) -> f64 {
        let steps: [f64; 5] = [1.0e-3, 5.0e-4, 2.5e-4, 1.25e-4, 6.25e-5];
        let mut best = f64::NAN;
        let mut best_delta = f64::INFINITY;
        let mut previous: Option<f64> = None;
        for h in steps {
            let d1 = (eval(h) - eval(-h)) / (2.0 * h);
            let half_h = 0.5 * h;
            let d2 = (eval(half_h) - eval(-half_h)) / (2.0 * half_h);
            let estimate: f64 = d2 + (d2 - d1) / 3.0;
            if let Some(prev) = previous {
                let delta = (estimate - prev).abs();
                if delta < best_delta {
                    best_delta = delta;
                    best = estimate;
                }
            } else {
                best = estimate;
            }
            previous = Some(estimate);
        }
        best
    }

    fn assert_backward_matches_forward_finite_difference(outputs: usize) {
        let x = finite_difference_design();
        let y = finite_difference_response(outputs);
        let penalty = finite_difference_penalty();
        let weights = finite_difference_weights();
        let targets = [
            ForwardScalar::Lambda,
            ForwardScalar::RemlScore,
            ForwardScalar::Coefficient(3, outputs - 1),
            ForwardScalar::Fitted(12, outputs - 1),
            ForwardScalar::Edf,
        ];
        for target in targets {
            let backward =
                one_hot_backward(x.view(), y.view(), penalty.view(), weights.view(), target);

            for row in 0..x.nrows() {
                for col in 0..x.ncols() {
                    let eval = |delta: f64| {
                        let mut candidate = x.clone();
                        candidate[[row, col]] += delta;
                        one_hot_objective(
                            candidate.view(),
                            y.view(),
                            penalty.view(),
                            weights.view(),
                            target,
                        )
                    };
                    let fd = adaptive_central_difference(eval);
                    assert_fd_close(
                        &format!("target={target:?} x[{row},{col}]"),
                        backward.grad_x[[row, col]],
                        fd,
                    );
                }
            }

            for row in 0..y.nrows() {
                for col in 0..y.ncols() {
                    let eval = |delta: f64| {
                        let mut candidate = y.clone();
                        candidate[[row, col]] += delta;
                        one_hot_objective(
                            x.view(),
                            candidate.view(),
                            penalty.view(),
                            weights.view(),
                            target,
                        )
                    };
                    let fd = adaptive_central_difference(eval);
                    assert_fd_close(
                        &format!("target={target:?} y[{row},{col}]"),
                        backward.grad_y[[row, col]],
                        fd,
                    );
                }
            }

            for row in 0..weights.len() {
                let eval = |delta: f64| {
                    let mut candidate = weights.clone();
                    candidate[row] += delta;
                    one_hot_objective(x.view(), y.view(), penalty.view(), candidate.view(), target)
                };
                let fd = adaptive_central_difference(eval);
                assert_fd_close(
                    &format!("target={target:?} weights[{row}]"),
                    backward.grad_weights[row],
                    fd,
                );
            }

            // ∂L/∂S over the RANGE-SPACE penalty entries. The REML objective
            // carries −½d·log|S|₊ (the pseudo-determinant over the NONZERO
            // eigenvalues), so ∂L/∂S is only a finite, FD-verifiable derivative
            // where a central ±h bump keeps S inside the PSD cone WITHOUT
            // changing its rank. A single-entry bump touching the null
            // direction violates both: the −h side drives an eigenvalue
            // slightly negative (leaves the cone → fit Err) and the +h side
            // turns the zero eigenvalue into a tiny positive one that joins
            // log|S|₊ as a −log(ε) term (a rank-change discontinuity in L).
            // The null-direction component of the analytic S-gradient is a
            // gauge convention for the null space (the L-metric pseudoinverse
            // `penalty_pinv` = L⁻ᵀ T⁺ L⁻¹), validated by algebra/consumer, not
            // FD. So restrict to the strictly-positive diagonal block (both
            // indices in 1..p for the diag([0, 0.8, 1.2, 1.7, 2.3]) fixture,
            // where S_rr > 0 and ±h stays PSD at full rank). The forward
            // consumes only `S_canon = 0.5(S + Sᵀ)` and the backward returns
            // the symmetrized gradient, so a single-entry bump of S[r, c]
            // (asymmetric) compares directly against `grad_penalty[r, c]` =
            // 0.5(G[r, c] + G[c, r]). Defensively, any entry whose largest ±h
            // probe leaves the cone is skipped (cone membership is monotone in
            // |h| here, so probing the largest step suffices).
            let null_index = 0usize; // diag([0.0, ...]) ⇒ coordinate 0 is the null direction.
            let probe_h = 1.0e-3_f64; // matches the largest adaptive_central_difference step.
            for r in 0..penalty.nrows() {
                for c in 0..penalty.ncols() {
                    if r == null_index || c == null_index {
                        continue;
                    }
                    let eval = |delta: f64| {
                        let mut candidate = penalty.clone();
                        candidate[[r, c]] += delta;
                        one_hot_objective(
                            x.view(),
                            y.view(),
                            candidate.view(),
                            weights.view(),
                            target,
                        )
                    };
                    let cone_safe = {
                        let mut s_plus = penalty.clone();
                        let mut s_minus = penalty.clone();
                        s_plus[[r, c]] += probe_h;
                        s_minus[[r, c]] -= probe_h;
                        one_hot_objective_try(
                            x.view(),
                            y.view(),
                            s_plus.view(),
                            weights.view(),
                            target,
                        )
                        .is_some()
                            && one_hot_objective_try(
                                x.view(),
                                y.view(),
                                s_minus.view(),
                                weights.view(),
                                target,
                            )
                            .is_some()
                    };
                    if !cone_safe {
                        continue;
                    }
                    let fd = adaptive_central_difference(eval);
                    assert_fd_close(
                        &format!("target={target:?} penalty[{r},{c}]"),
                        backward.grad_penalty[[r, c]],
                        fd,
                    );
                }
            }
        }
    }

    #[test]
    fn scalar_backward_matches_forward_finite_difference_for_all_x_y_and_weight_entries() {
        assert_backward_matches_forward_finite_difference(1);
    }

    #[test]
    fn multi_output_backward_matches_forward_finite_difference_for_all_x_y_and_weight_entries() {
        assert_backward_matches_forward_finite_difference(3);
    }

    #[test]
    fn backward_vjp_matches_finite_difference() {
        let x = array![
            [1.0, -1.0, 0.2],
            [1.0, -0.3, -0.1],
            [1.0, 0.2, 0.4],
            [1.0, 0.8, 0.1],
            [1.0, 1.4, 0.5],
            [1.0, 2.0, 0.9],
        ];
        let y = array![
            [0.1, -0.2],
            [0.2, 0.1],
            [0.7, 0.0],
            [1.1, 0.3],
            [1.8, 0.9],
            [2.4, 1.4],
        ];
        let weights = array![1.0, 0.9, 1.1, 1.2, 0.8, 1.3];
        let penalty = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.2], [0.0, 0.2, 1.7]];
        let upstream_coefficients = array![[0.2, -0.1], [0.05, 0.03], [-0.04, 0.07]];
        let upstream_fitted = array![
            [0.01, -0.02],
            [0.03, 0.01],
            [-0.01, 0.02],
            [0.04, -0.03],
            [0.02, 0.05],
            [-0.02, 0.01],
        ];
        let upstream_lambda = 0.17;
        let upstream_score = -0.11;

        let backward = gaussian_reml_multi_closed_form_backward(
            x.view(),
            y.view(),
            penalty.view(),
            Some(weights.view()),
            Some(0.8),
            upstream_lambda,
            Some(upstream_coefficients.view()),
            Some(upstream_fitted.view()),
            upstream_score,
            0.0,
        )
        .expect("backward VJP");

        let objective = |x_eval: &Array2<f64>, y_eval: &Array2<f64>, w_eval: &Array1<f64>| {
            let fit = gaussian_reml_multi_closed_form_with_cache(
                x_eval.view(),
                y_eval.view(),
                penalty.view(),
                Some(w_eval.view()),
                Some(0.8),
                None,
            )
            .expect("fit for objective");
            upstream_lambda * fit.lambda
                + upstream_score * fit.reml_score
                + (&fit.coefficients * &upstream_coefficients).sum()
                + (&fit.fitted * &upstream_fitted).sum()
        };
        let eps = 1.0e-6;
        assert!(objective(&x, &y, &weights).is_finite());

        let mut x_plus = x.clone();
        let mut x_minus = x.clone();
        x_plus[[3, 2]] += eps;
        x_minus[[3, 2]] -= eps;
        let fd_x =
            (objective(&x_plus, &y, &weights) - objective(&x_minus, &y, &weights)) / (2.0 * eps);
        assert!(
            (fd_x - backward.grad_x[[3, 2]]).abs() <= 2.0e-4,
            "grad_x mismatch: analytic={} fd={}",
            backward.grad_x[[3, 2]],
            fd_x
        );

        let mut y_plus = y.clone();
        let mut y_minus = y.clone();
        y_plus[[4, 1]] += eps;
        y_minus[[4, 1]] -= eps;
        let fd_y =
            (objective(&x, &y_plus, &weights) - objective(&x, &y_minus, &weights)) / (2.0 * eps);
        assert!(
            (fd_y - backward.grad_y[[4, 1]]).abs() <= 2.0e-4,
            "grad_y mismatch: analytic={} fd={}",
            backward.grad_y[[4, 1]],
            fd_y
        );

        let mut w_plus = weights.clone();
        let mut w_minus = weights.clone();
        w_plus[2] += eps;
        w_minus[2] -= eps;
        let fd_w = (objective(&x, &y, &w_plus) - objective(&x, &y, &w_minus)) / (2.0 * eps);
        assert!(
            (fd_w - backward.grad_weights[2]).abs() <= 2.0e-4,
            "grad_weight mismatch: analytic={} fd={}",
            backward.grad_weights[2],
            fd_w
        );

        // Combined-seed ∂L/∂S spot-check: perturb individual penalty entries with
        // x/y/w held at base, under mixed (λ, score, β, fitted) seeds. The penalty
        // [[0,0,0],[0,1,0.2],[0,0.2,1.7]] is nullity 1 (coordinate 0 is the null
        // direction); ∂L/∂S is FD-verifiable only on the strictly-positive
        // RANGE block (indices 1,2), where a central ±h bump keeps S PSD at full
        // rank. Null-touching entries (any index 0) are non-FD-verifiable — the
        // −½d·log|S|₊ pseudo-determinant term makes L either cone-leaving or
        // rank-change-discontinuous there (see the exhaustive S loop above). A
        // single-entry asymmetric bump of S[r, c] compares directly to
        // grad_penalty[[r, c]] = 0.5(G[r,c] + G[c,r]), exercising the backward
        // symmetrization.
        let objective_s = |s_eval: &Array2<f64>| {
            let fit = gaussian_reml_multi_closed_form_with_cache(
                x.view(),
                y.view(),
                s_eval.view(),
                Some(weights.view()),
                Some(0.8),
                None,
            )
            .expect("fit for penalty objective");
            upstream_lambda * fit.lambda
                + upstream_score * fit.reml_score
                + (&fit.coefficients * &upstream_coefficients).sum()
                + (&fit.fitted * &upstream_fitted).sum()
        };
        // (1,1) full-rank diagonal; (1,2) pure off-diagonal between two penalized
        // directions; (2,2) full-rank diagonal. All in the strictly-positive
        // range block, so ±h stays PSD at full rank.
        for (r, c) in [(1usize, 1usize), (1, 2), (2, 2)] {
            let mut s_plus = penalty.clone();
            let mut s_minus = penalty.clone();
            s_plus[[r, c]] += eps;
            s_minus[[r, c]] -= eps;
            let fd_s = (objective_s(&s_plus) - objective_s(&s_minus)) / (2.0 * eps);
            assert!(
                (fd_s - backward.grad_penalty[[r, c]]).abs() <= 2.0e-4,
                "grad_penalty[{r},{c}] mismatch: analytic={} fd={}",
                backward.grad_penalty[[r, c]],
                fd_s
            );
        }
    }

    #[test]
    fn batched_eigen_cache_matches_per_fit_build() {
        // Three K=3 problems sharing the same penalty matrix. The batched
        // pipeline must produce caches that are bit-exact identical to what
        // the per-fit `gaussian_reml_eigen_cache_from_xtwx` builder produces,
        // regardless of whether the GPU batched Cholesky kicks in or the
        // helper falls through to per-fit Cholesky.
        let xtwx_a = array![[4.0, 1.0], [1.0, 3.0]];
        let xtwx_b = array![[2.5, -0.5], [-0.5, 1.7]];
        let xtwx_c = array![[7.2, 0.3], [0.3, 5.1]];
        let penalty = array![[0.0, 0.0], [0.0, 1.0]];

        let batched = build_gaussian_reml_eigen_cache_batched(
            vec![xtwx_a.clone(), xtwx_b.clone(), xtwx_c.clone()],
            penalty.view(),
            None,
        );
        assert_eq!(batched.len(), 3);

        for (xtwx, batched_cache) in [&xtwx_a, &xtwx_b, &xtwx_c].into_iter().zip(batched.iter()) {
            let single = gaussian_reml_eigen_cache_from_xtwx(xtwx.clone(), penalty.view(), None)
                .expect("per-fit cache");
            let batched_cache = batched_cache.as_ref().expect("batched cache");
            assert_eq!(batched_cache.penalty_rank, single.penalty_rank);
            assert_eq!(batched_cache.nullity, single.nullity);
            assert_eq!(batched_cache.xtwx_fingerprint, single.xtwx_fingerprint);
            assert_eq!(
                batched_cache.penalty_fingerprint,
                single.penalty_fingerprint
            );
            assert!((batched_cache.logdet_xtwx - single.logdet_xtwx).abs() <= 1.0e-12);
            assert!(
                (batched_cache.logdet_penalty_positive - single.logdet_penalty_positive).abs()
                    <= 1.0e-12
            );
            for (a, b) in batched_cache
                .penalty_eigenvalues
                .iter()
                .zip(single.penalty_eigenvalues.iter())
            {
                assert!((a - b).abs() <= 1.0e-12);
            }
            for ((a, b), _) in batched_cache
                .coefficient_basis
                .iter()
                .zip(single.coefficient_basis.iter())
                .zip(0..)
            {
                assert!((a - b).abs() <= 1.0e-12);
            }
        }
    }

    /// Deterministic linear-congruential generator (Knuth/MMIX constants) so the
    /// enumeration stress tests are fully reproducible — no time/thread seeding.
    struct Lcg(u64);
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0
        }
        fn unit(&mut self) -> f64 {
            (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
        }
        fn range(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }
    }

    /// Synthetic eigen-cache with identity bases (the enumerator only reads
    /// `penalty_eigenvalues`, `penalty_rank`, `nullity` and the additive logdet
    /// constants), so tests can drive `evaluate_reml_parts` /
    /// `reml_deriv_enclosure` directly from a spectrum.
    fn synthetic_cache(eigs: &[f64]) -> GaussianRemlEigenCache {
        let n = eigs.len();
        // The SAME range/null predicate the cache builder uses to define
        // `penalty_rank` and every consumer uses to read the spectrum (#2740);
        // a fixture that counted its own rank under `δ > 0.0` would hand the
        // objective a sum and an offset populated by two different rules.
        let tolerance = penalty_range_tolerance(ArrayView1::from(eigs));
        let rank = eigs.iter().filter(|&&delta| delta > tolerance).count();
        GaussianRemlEigenCache {
            penalty_eigenvalues: Array1::from(eigs.to_vec()),
            eigenvectors: Array2::eye(n),
            coefficient_basis: Array2::eye(n),
            xtwx_fingerprint: 0,
            penalty_fingerprint: 0,
            logdet_xtwx: 0.0,
            logdet_penalty_positive: 0.0,
            penalty_rank: rank,
            nullity: n - rank,
        }
    }

    /// One-mode profiled REML has an analytic stationary point. With
    /// `q = c²`, irreducible residual `r`, residual dof `n`, and `t = λδ`,
    ///
    /// `dp(t) = r + q t/(1+t)` and `V'(rho)=0`
    /// iff `t = r / ((n-1)q-r)`.
    ///
    /// This pins the objective actually implemented here (dispersion profiled
    /// at every rho), not the fixed-sigma surrogate proposed in #2312.
    #[test]
    fn profiled_one_mode_certificate_matches_analytic_root_and_ignores_seed_as_candidate() {
        let delta = 4.0;
        let q = 2.0;
        let irreducible_residual = 3.0;
        let n_effective = 10usize;
        let cache = synthetic_cache(&[delta]);
        let ywy = array![q + irreducible_residual];
        let projected = array![[q]];
        let eval = |rho: f64| {
            evaluate_reml_parts(&cache, ywy.view(), projected.view(), n_effective, 1, rho)
        };
        let enclose = |a: f64, b: f64| {
            reml_deriv_enclosure(&cache, ywy.view(), projected.view(), n_effective, 1, a, b)
        };
        let expected_t =
            irreducible_residual / (((n_effective - 1) as f64) * q - irreducible_residual);
        let expected_rho = (expected_t / delta).ln();
        let mut roots = Vec::new();
        // Scoped so the visitor's mutable borrow of `roots` ends before the
        // assertions below read it.
        let selection = {
            let mut collect_root = |root: StationaryRoot, _: &ObjectiveEval| roots.push(root);
            enumerate_and_select_rho(&eval, &enclose, Some(-20.0), Some(&mut collect_root))
                .expect("profile certificate")
        };

        assert_eq!(roots.len(), 1, "unexpected stationary set");
        assert!(
            roots[0].bracket[0] <= expected_rho && expected_rho <= roots[0].bracket[1],
            "analytic root {expected_rho} outside certified bracket {:?}",
            roots[0].bracket
        );
        assert!(
            (selection.rho - expected_rho).abs()
                <= RHO_BRACKET_RESOLUTION * (1.0 + expected_rho.abs()),
            "selected rho {} differs from analytic profiled root {expected_rho}",
            selection.rho
        );
        assert_ne!(
            selection.rho.to_bits(),
            (-20.0_f64).to_bits(),
            "a nonstationary warm hint must never enter the objective argmin"
        );
    }

    #[test]
    fn unresolved_stationary_structure_is_a_typed_refusal() {
        let eval = |rho: f64| ObjectiveEval {
            cost: rho * rho,
            grad: 2.0 * rho,
            hess: 2.0,
            edf: 0.0,
            // An analytic fixture: its cost is exact by construction.
            cost_roundoff: 0.0,
        };
        // Deliberately uninformative but endpoint-valid enclosures force the
        // resolution-floor branch without an expensive production-depth tree.
        let enclose = |_: f64, _: f64| (Interval::entire(), Interval::entire());
        let error = enumerate_and_select_rho_with_controls(
            eval,
            enclose,
            None,
            ProfileSearchControls {
                lower: -1.0,
                upper: 1.0,
                resolution: 0.25,
                max_depth: 0,
            },
            None,
        )
        .expect_err("ambiguous stationary structure must refuse");
        assert!(matches!(error, EstimationError::RemlDidNotConverge { .. }));
    }

    #[test]
    fn profiled_modal_evaluation_is_finite_beyond_exp_range() {
        let cache = synthetic_cache(&[4.0]);
        let ywy = array![5.0];
        let projected = array![[2.0]];
        for rho in [-1_000.0, 1_000.0] {
            let mode = modal_kernels(rho, 4.0);
            assert!(mode.log_one_plus_t.is_finite());
            assert!(mode.u.is_finite());
            assert!(mode.v.is_finite());
            assert!(mode.w.is_finite());
            assert!(mode.k.is_finite());
            let value = evaluate_reml_parts(&cache, ywy.view(), projected.view(), 10, 1, rho);
            assert!(value.cost.is_finite(), "non-finite cost at rho={rho}");
            assert!(value.grad.is_finite(), "non-finite gradient at rho={rho}");
            assert!(value.hess.is_finite(), "non-finite Hessian at rho={rho}");
        }
    }

    /// The selected representative must have cost no larger than every isolated
    /// stationary representative and both finite-window endpoints.
    #[test]
    fn selected_rho_beats_every_certified_profile_candidate() {
        let mut rng = Lcg::new(0x9911_7733_5522_0044);
        for _case in 0..40 {
            let n_eig = 2 + (rng.next_u64() % 4) as usize;
            let eigs: Vec<f64> = (0..n_eig).map(|_| rng.range(-5.0, 6.0).exp()).collect();
            let cache = synthetic_cache(&eigs);
            let c2: Vec<f64> = (0..n_eig)
                .map(|_| {
                    let v = rng.range(0.0, 2.5);
                    v * v
                })
                .collect();
            let sum_c2: f64 = c2.iter().sum();
            let prs = Array2::from_shape_vec((n_eig, 1), c2).unwrap();
            let ywy = Array1::from(vec![sum_c2 + rng.range(0.05, 2.0)]);
            let n_eff = 80usize;
            let n_out = 1usize;

            let eval =
                |rho: f64| evaluate_reml_parts(&cache, ywy.view(), prs.view(), n_eff, n_out, rho);
            let enclose = |a: f64, b: f64| {
                reml_deriv_enclosure(&cache, ywy.view(), prs.view(), n_eff, n_out, a, b)
            };
            let mut roots = Vec::new();
            let selection = {
                let mut collect_rho = |root: StationaryRoot, _: &ObjectiveEval| roots.push(root.rho);
                enumerate_and_select_rho(&eval, &enclose, None, Some(&mut collect_rho)).unwrap()
            };
            let selected = selection.rho;
            let selected_cost = eval(selected).cost;
            let tol = 1.0e-8 * (1.0 + selected_cost.abs());

            for &r in &roots {
                assert!(selected_cost <= eval(r).cost + tol);
            }
            assert!(selected_cost <= eval(RHO_LOWER).cost + tol);
            assert!(selected_cost <= eval(RHO_UPPER).cost + tol);
        }
    }

    #[test]
    fn backward_from_fit_matches_backward_with_refit() {
        // The Task 3 state round-trip in pyffi calls `_from_fit`; that path
        // must be numerically identical to the refitting `_backward` entry
        // when fed the same forward result. This guards the optimization
        // against drift when either path is touched.
        let x = array![[1.0, -0.9], [1.0, -0.4], [1.0, 0.1], [1.0, 0.6], [1.0, 1.1],];
        let y = array![[0.2, -0.1], [0.4, 0.1], [0.7, 0.3], [1.0, 0.5], [1.5, 0.8]];
        let penalty = array![[0.0, 0.0], [0.0, 1.5]];
        let weights = array![1.05, 0.95, 1.01, 0.99, 1.03];

        let refit = gaussian_reml_multi_closed_form_backward(
            x.view(),
            y.view(),
            penalty.view(),
            Some(weights.view()),
            Some(0.85),
            0.2,
            None,
            None,
            -0.1,
            0.0,
        )
        .expect("refit backward");

        let fit = gaussian_reml_multi_closed_form_with_cache(
            x.view(),
            y.view(),
            penalty.view(),
            Some(weights.view()),
            Some(0.85),
            None,
        )
        .expect("forward fit");
        let from_fit = gaussian_reml_multi_closed_form_backward_from_fit(
            x.view(),
            y.view(),
            penalty.view(),
            Some(weights.view()),
            &fit,
            0.2,
            None,
            None,
            -0.1,
            0.0,
        )
        .expect("from_fit backward");

        for (a, b) in refit.grad_x.iter().zip(from_fit.grad_x.iter()) {
            assert!((a - b).abs() <= 1.0e-12);
        }
        for (a, b) in refit.grad_y.iter().zip(from_fit.grad_y.iter()) {
            assert!((a - b).abs() <= 1.0e-12);
        }
        for (a, b) in refit.grad_weights.iter().zip(from_fit.grad_weights.iter()) {
            assert!((a - b).abs() <= 1.0e-12);
        }
    }

    /// Regression: when `K = XᵀWX + λS` is effectively rank-deficient (e.g.
    /// `λ` has saturated very large), the backward must NOT error — it must
    /// degrade gracefully and return zero gradients of the correct shape.
    /// This is the production-training scenario where individual atoms can
    /// saturate `λ_k` in early batches; raising here would crash an entire
    /// step. We construct the degenerate state by running a real forward
    /// fit and then corrupting `reml_hess_rho` to 0 (the gate variable the
    /// backward checks). We assert: (a) no error, (b) all gradients finite,
    /// (c) shapes match the inputs.
    #[test]
    fn backward_degrades_gracefully_when_k_is_near_singular() {
        // Small, full-rank S with a moderately-conditioned X. The exact
        // numbers don't matter; what matters is that we then force the
        // ill-conditioned gate to fire.
        let x = array![
            [1.0, -1.0, 0.5],
            [1.0, -0.5, 0.2],
            [1.0, 0.0, -0.1],
            [1.0, 0.5, 0.3],
            [1.0, 1.0, 0.8],
            [1.0, 1.5, 1.1],
            [1.0, 2.0, 1.5],
            [1.0, 2.5, 2.0],
            [1.0, 3.0, 2.6],
            [1.0, 3.5, 3.1],
        ];
        let y = array![
            [0.1],
            [0.3],
            [0.4],
            [0.7],
            [1.0],
            [1.5],
            [2.0],
            [2.7],
            [3.3],
            [4.0]
        ];
        // Full-rank S to keep the forward well-posed.
        let penalty = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        let mut fit =
            gaussian_reml_multi_closed_form(x.view(), y.view(), penalty.view(), None, Some(0.0))
                .expect("forward fit must succeed for well-posed input");
        // Force the ill-conditioned gate to fire by zeroing the REML
        // Hessian w.r.t. rho — this is exactly what happens in production
        // when `λ` saturates to 1e10+ and `d²ℓ/dρ² → 0`.
        fit.reml_hess_rho = 0.0;

        let result = gaussian_reml_multi_closed_form_backward_from_fit(
            x.view(),
            y.view(),
            penalty.view(),
            None,
            &fit,
            // Nonzero upstreams to force the backward to actually try to
            // populate gradients (rather than short-circuit on zero seeds).
            1.0,
            None,
            None,
            1.0,
            1.0,
        )
        .expect("backward must NOT error on near-singular K");

        assert_eq!(result.grad_x.dim(), (x.nrows(), x.ncols()));
        assert_eq!(result.grad_y.dim(), (y.nrows(), y.ncols()));
        assert_eq!(result.grad_penalty.dim(), (x.ncols(), x.ncols()));
        assert_eq!(result.grad_weights.dim(), x.nrows());
        for v in result.grad_x.iter() {
            assert!(v.is_finite(), "grad_x must be finite, got {v}");
        }
        for v in result.grad_y.iter() {
            assert!(v.is_finite(), "grad_y must be finite, got {v}");
        }
        for v in result.grad_penalty.iter() {
            assert!(v.is_finite(), "grad_penalty must be finite, got {v}");
        }
        for v in result.grad_weights.iter() {
            assert!(v.is_finite(), "grad_weights must be finite, got {v}");
        }
    }
}

/// Vector–Jacobian products of the multi-block per-smooth-λ Gaussian REML
/// forward fit ([`gaussian_reml_blocks_orthogonal_shared_scale`]), back to the
/// design blocks, penalty blocks, response, and weights.
pub struct GaussianRemlBlocksBackwardAnalytic {
    pub grad_designs: Vec<Array2<f64>>,
    pub grad_penalties: Vec<Array2<f64>>,
    pub grad_y: Array2<f64>,
    pub grad_weights: Array1<f64>,
}

/// Analytic backward for the multi-block per-smooth-λ Gaussian REML forward.
///
/// Computes VJPs of (coefficients, fitted, lambdas, log_lambdas, reml_score,
/// edf) back to (design_blocks, penalty_blocks, y, weights). The VJP is
/// assembled at the converged log-λ vector: fixed-ρ β/fitted/profiled-REML/EDF
/// terms are accumulated first, then the smoothing-parameter sensitivity is
/// routed through the F×F profiled REML score Hessian from the implicit optimum.
/// Pairs with the forward [`gaussian_reml_blocks_orthogonal_shared_scale`].
pub fn gaussian_reml_fit_blocks_backward_analytic(
    designs: &[Array2<f64>],
    penalties_raw: &[Array2<f64>],
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    rhos: &[f64],
    grad_coefficients: Option<ArrayView2<'_, f64>>,
    grad_fitted: Option<ArrayView2<'_, f64>>,
    grad_lambdas: Option<ArrayView1<'_, f64>>,
    grad_log_lambdas: Option<ArrayView1<'_, f64>>,
    grad_reml_score: f64,
    grad_edf: Option<ArrayView1<'_, f64>>,
) -> Result<GaussianRemlBlocksBackwardAnalytic, EstimationError> {
    let n = y.len();
    let f_blocks = designs.len();
    if f_blocks == 0 || penalties_raw.len() != f_blocks {
        return Err(EstimationError::InvalidInput(format!(
            "gaussian_reml_fit_blocks_backward requires equal non-zero design and penalty \
             block counts; got designs={}, penalties={}",
            f_blocks,
            penalties_raw.len()
        )));
    }
    let mut offsets = Vec::with_capacity(f_blocks + 1);
    let mut cursor = 0_usize;
    offsets.push(cursor);
    for (block, design) in designs.iter().enumerate() {
        if design.nrows() != n {
            return Err(EstimationError::InvalidInput(format!(
                "designs[{block}] has {} rows, expected {n}",
                design.nrows()
            )));
        }
        if penalties_raw[block].dim() != (design.ncols(), design.ncols()) {
            return Err(EstimationError::InvalidInput(format!(
                "penalties[{block}] has shape {}x{}, expected {}x{}",
                penalties_raw[block].nrows(),
                penalties_raw[block].ncols(),
                design.ncols(),
                design.ncols()
            )));
        }
        cursor += design.ncols();
        offsets.push(cursor);
    }
    // The fold's running total IS the last offset, so there is nothing to
    // re-read and no emptiness to assert.
    let p_total = cursor;
    if n == 0 || p_total == 0 {
        return Err(EstimationError::InvalidInput(
            "gaussian_reml_fit_blocks_backward requires non-empty rows and at least one coefficient column"
                .to_string(),
        ));
    }

    if rhos.len() != f_blocks {
        return Err(EstimationError::InvalidInput(format!(
            "log_lambdas length mismatch: expected {f_blocks}, got {}",
            rhos.len()
        )));
    }
    if let Some(gc) = grad_coefficients {
        if gc.dim() != (p_total, 1) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_coefficients shape mismatch: expected {}x1, got {}x{}",
                p_total,
                gc.nrows(),
                gc.ncols()
            )));
        }
    }
    if let Some(gf) = grad_fitted {
        if gf.dim() != (n, 1) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_fitted shape mismatch: expected {}x1, got {}x{}",
                n,
                gf.nrows(),
                gf.ncols()
            )));
        }
    }
    if !grad_reml_score.is_finite() {
        return Err(EstimationError::InvalidInput(format!(
            "grad_reml_score must be finite; got {grad_reml_score}"
        )));
    }
    if let Some(vec) = grad_lambdas {
        if vec.len() != f_blocks {
            return Err(EstimationError::InvalidInput(format!(
                "grad_lambdas length mismatch: expected {f_blocks}, got {}",
                vec.len()
            )));
        }
    }
    if let Some(vec) = grad_log_lambdas {
        if vec.len() != f_blocks {
            return Err(EstimationError::InvalidInput(format!(
                "grad_log_lambdas length mismatch: expected {f_blocks}, got {}",
                vec.len()
            )));
        }
    }
    if let Some(vec) = grad_edf {
        if vec.len() != f_blocks {
            return Err(EstimationError::InvalidInput(format!(
                "grad_edf length mismatch: expected {f_blocks}, got {}",
                vec.len()
            )));
        }
    }
    if let Some(gc) = grad_coefficients {
        if let Some(((row, col), value)) = gc.indexed_iter().find(|(_, value)| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_coefficients[{row},{col}] must be finite; got {value}"
            )));
        }
    }
    if let Some(gf) = grad_fitted {
        if let Some(((row, col), value)) = gf.indexed_iter().find(|(_, value)| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_fitted[{row},{col}] must be finite; got {value}"
            )));
        }
    }
    if let Some(vec) = grad_lambdas {
        if let Some((block, value)) = vec.iter().enumerate().find(|(_, value)| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_lambdas[{block}] must be finite; got {value}"
            )));
        }
    }
    if let Some(vec) = grad_log_lambdas {
        if let Some((block, value)) = vec.iter().enumerate().find(|(_, value)| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_log_lambdas[{block}] must be finite; got {value}"
            )));
        }
    }
    if let Some(vec) = grad_edf {
        if let Some((block, value)) = vec.iter().enumerate().find(|(_, value)| !value.is_finite()) {
            return Err(EstimationError::InvalidInput(format!(
                "grad_edf[{block}] must be finite; got {value}"
            )));
        }
    }
    for (block, design) in designs.iter().enumerate() {
        if let Some(((row, col), value)) =
            design.indexed_iter().find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "designs[{block}][{row},{col}] must be finite; got {value}"
            )));
        }
    }
    for (block, penalty) in penalties_raw.iter().enumerate() {
        if let Some(((row, col), value)) =
            penalty.indexed_iter().find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "penalties[{block}][{row},{col}] must be finite; got {value}"
            )));
        }
    }
    if let Some((row, value)) = y.iter().enumerate().find(|(_, value)| !value.is_finite()) {
        return Err(EstimationError::InvalidInput(format!(
            "y[{row}] must be finite; got {value}"
        )));
    }
    if let Some((row, value)) = weights
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || **value < 0.0)
    {
        return Err(EstimationError::InvalidInput(format!(
            "weights[{row}] must be finite and non-negative; got {value}"
        )));
    }

    let mut z = Array2::<f64>::zeros((n, p_total));
    for k in 0..f_blocks {
        z.slice_mut(s![.., offsets[k]..offsets[k + 1]])
            .assign(&designs[k]);
    }

    let blockwise_penalties: Vec<BlockwisePenalty> = penalties_raw
        .iter()
        .enumerate()
        .map(|(block, penalty)| {
            BlockwisePenalty::new(offsets[block]..offsets[block + 1], penalty.clone())
        })
        .collect();
    let domain = GaussianRemlBlocksDomain::from_blockwise_penalties(p_total, &blockwise_penalties)?;
    let lambdas = Array1::from_vec(
        gam_problem::checked_exp_log_strengths(rhos.iter().copied())
            .map_err(|error| EstimationError::InvalidInput(error.to_string()))?,
    );
    let k_matrix = domain.certify_joint_coefficient_map(z.view(), weights, lambdas.view())?;

    // The one-block forward is an exact algebraic reduction through the
    // established scalar closed-form solver.  Its VJP must reduce through the
    // same implementation as well: doing so preserves the scalar solver's
    // grid-free stationary-root selection and all of its boundary semantics,
    // rather than asking a nominally equivalent multi-block derivation to
    // reproduce them to roundoff.
    if f_blocks == 1 {
        let mut upstream_lambda = grad_lambdas.map_or(0.0, |gradient| gradient[0]);
        if let Some(gradient) = grad_log_lambdas {
            upstream_lambda += gradient[0] / lambdas[0];
        }
        let y_owned = y.to_owned().insert_axis(Axis(1));
        let weights_owned = weights.to_owned();
        let fit = gaussian_reml_multi_closed_form_with_cache(
            z.view(),
            y_owned.view(),
            penalties_raw[0].view(),
            Some(weights_owned.view()),
            Some(lambdas[0]),
            None,
        )?;
        let backward = gaussian_reml_multi_closed_form_backward_from_fit(
            z.view(),
            y_owned.view(),
            penalties_raw[0].view(),
            Some(weights_owned.view()),
            &fit,
            upstream_lambda,
            grad_coefficients,
            grad_fitted,
            grad_reml_score,
            grad_edf.map_or(0.0, |gradient| gradient[0]),
        )?;
        return Ok(GaussianRemlBlocksBackwardAnalytic {
            grad_designs: vec![backward.grad_x],
            grad_penalties: vec![backward.grad_penalty],
            grad_y: backward.grad_y,
            grad_weights: backward.grad_weights,
        });
    }

    let penalties = domain.local_penalties();
    let pinvs = domain.penalty_pseudoinverses()?;
    let r = gam_linalg::utils::certified_spd_inverse(
        &k_matrix,
        "block Gaussian REML penalized normal matrix",
    )
    .map(gam_linalg::utils::CertifiedSpdInverse::into_inverse)
    .map_err(|error| {
        EstimationError::InvalidInput(format!(
            "block Gaussian REML requires an exact SPD penalized normal matrix: {error}"
        ))
    })?;

    let mut xtwy = Array1::<f64>::zeros(p_total);
    for row in 0..n {
        let wy = weights[row] * y[row];
        for col in 0..p_total {
            xtwy[col] += z[[row, col]] * wy;
        }
    }
    let beta = r.dot(&xtwy);
    let fitted = z.dot(&beta);
    if let Some((col, value)) = beta
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(EstimationError::InvalidInput(format!(
            "solved coefficient {col} is non-finite: {value}"
        )));
    }
    let residual = &y.to_owned() - &fitted;
    let weighted_residual = &residual * &weights.to_owned();
    let mut q = residual
        .iter()
        .zip(weights.iter())
        .map(|(&value, &weight)| weight * value * value)
        .sum::<f64>();
    for block in 0..f_blocks {
        let start = offsets[block];
        let end = offsets[block + 1];
        let beta_block = beta.slice(s![start..end]);
        q += lambdas[block] * beta_block.dot(&penalties[block].dot(&beta_block));
    }
    if !q.is_finite() || q <= 0.0 {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML residual quadratic form must be finite and positive; got {q}"
        )));
    }
    let nullity = domain.nullspace_dims().iter().sum::<usize>();
    // Match the block-orthogonal forward's effective sample size: zero
    // prior-weight rows are excluded from the residual degrees of freedom.
    let nu = effective_observation_count(weights) as f64 - nullity as f64;
    if !(nu.is_finite() && nu > 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML residual degrees of freedom must be positive; got {nu}"
        )));
    }
    let tau = nu / q;
    let tau_q = -nu / (q * q);
    if !(tau.is_finite() && tau_q.is_finite()) {
        return Err(EstimationError::InvalidInput(format!(
            "Gaussian REML scale derivatives are non-finite: tau={tau}, tau_q={tau_q}"
        )));
    }

    let mut grad_z = Array2::<f64>::zeros((n, p_total));
    let mut g_kernel = Array2::<f64>::zeros((p_total, p_total));
    let mut h_kernel = Array1::<f64>::zeros(p_total);
    let mut q_kernel = 0.0_f64;
    let mut j_blocks: Vec<Array2<f64>> = penalties
        .iter()
        .map(|p| Array2::<f64>::zeros(p.dim()))
        .collect();

    let mut beta_tilde = Array1::<f64>::zeros(p_total);
    if let Some(gc) = grad_coefficients {
        beta_tilde += &gc.column(0).to_owned();
    }
    if let Some(gf) = grad_fitted {
        let gf_col = gf.column(0).to_owned();
        beta_tilde += &z.t().dot(&gf_col);
        for row in 0..n {
            for col in 0..p_total {
                grad_z[[row, col]] += gf_col[row] * beta[col];
            }
        }
    }

    // Generic downstream losses that explicitly seed beta_hat or fitted
    // values cannot use the REML envelope shortcut. Route those seeds through
    // the fixed-rho KKT adjoint K u = beta_tilde before differentiating
    // designs, penalties, y, weights, and rho.
    let u = r.dot(&beta_tilde);
    h_kernel += &u;
    for i in 0..p_total {
        for j in 0..p_total {
            g_kernel[[i, j]] -= 0.5 * (beta[i] * u[j] + u[i] * beta[j]);
        }
    }

    let mut alpha = Array1::<f64>::zeros(f_blocks);
    if let Some(gl) = grad_lambdas {
        for block in 0..f_blocks {
            alpha[block] += gl[block] * lambdas[block];
        }
    }
    if let Some(grho) = grad_log_lambdas {
        alpha += &grho.to_owned();
    }

    let mut p_betas = Vec::with_capacity(f_blocks);
    let mut m_vectors = Vec::with_capacity(f_blocks);
    let mut rp_matrices = Vec::with_capacity(f_blocks);
    let mut rpr_matrices = Vec::with_capacity(f_blocks);
    let mut b_values = Array1::<f64>::zeros(f_blocks);
    let mut t_values = Array1::<f64>::zeros(f_blocks);

    for block in 0..f_blocks {
        let start = offsets[block];
        let end = offsets[block + 1];
        let beta_k = beta.slice(s![start..end]).to_owned();
        let s_beta = penalties[block].dot(&beta_k);
        let lambda = lambdas[block];
        let lambda_s_beta = s_beta.mapv(|value| lambda * value);
        let mut p_beta = Array1::<f64>::zeros(p_total);
        for local_i in 0..(end - start) {
            p_beta[start + local_i] = lambda_s_beta[local_i];
        }
        let weighted_penalty = penalties[block].mapv(|value| lambda * value);
        let rp_block = r.slice(s![.., start..end]).dot(&weighted_penalty);
        let mut rp = Array2::<f64>::zeros((p_total, p_total));
        rp.slice_mut(s![.., start..end]).assign(&rp_block);
        let rpr = rp_block.dot(&r.slice(s![start..end, ..]));
        let m = r.slice(s![.., start..end]).dot(&lambda_s_beta);
        b_values[block] = beta.dot(&p_beta);
        t_values[block] = (0..(end - start))
            .map(|local_i| rp_block[[start + local_i, local_i]])
            .sum::<f64>();
        alpha[block] -= u.dot(&p_beta);
        p_betas.push(p_beta);
        m_vectors.push(m);
        rp_matrices.push(rp);
        rpr_matrices.push(rpr);
    }

    if grad_reml_score != 0.0 {
        q_kernel += 0.5 * grad_reml_score * tau;
        g_kernel += &(r.clone() * (0.5 * grad_reml_score));
        for block in 0..f_blocks {
            j_blocks[block] -= &(pinvs[block].clone() * (0.5 * grad_reml_score / lambdas[block]));
        }
    }

    let mut trace_pairs = Array2::<f64>::zeros((f_blocks, f_blocks));
    for i in 0..f_blocks {
        for j in 0..f_blocks {
            trace_pairs[[i, j]] =
                gam_linalg::utils::trace_of_product(rp_matrices[i].view(), rp_matrices[j].view());
        }
    }

    if let Some(ge) = grad_edf {
        for edf_block in 0..f_blocks {
            let scale = ge[edf_block];
            if scale == 0.0 {
                continue;
            }
            let start = offsets[edf_block];
            let end = offsets[edf_block + 1];
            g_kernel += &(rpr_matrices[edf_block].clone() * scale);
            j_blocks[edf_block] -= &(r.slice(s![start..end, start..end]).to_owned() * scale);
            for rho_block in 0..f_blocks {
                alpha[rho_block] += scale * trace_pairs[[edf_block, rho_block]];
                if rho_block == edf_block {
                    alpha[rho_block] -= scale * t_values[edf_block];
                }
            }
        }
    }

    if let Some((block, value)) = alpha
        .iter()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(EstimationError::InvalidInput(format!(
            "rho adjoint seed for block {block} is non-finite: {value}"
        )));
    }

    if alpha.iter().any(|value| *value != 0.0) {
        let mut outer_h = Array2::<f64>::zeros((f_blocks, f_blocks));
        for k in 0..f_blocks {
            for j in 0..f_blocks {
                let beta_pk_r_pj_beta = p_betas[k].dot(&m_vectors[j]);
                outer_h[[k, j]] = 0.5 * trace_pairs[[k, j]] + tau * beta_pk_r_pj_beta
                    - if k == j {
                        0.5 * (t_values[k] + tau * b_values[k])
                    } else {
                        0.0
                    }
                    - 0.5 * tau_q * b_values[k] * b_values[j];
            }
        }
        // `outer_h` is the Jacobian of the negative profiled REML estimating
        // equation. Preserve every signed curvature direction exactly; a
        // singular Jacobian means this VJP is not identified and must fail,
        // rather than silently replacing its spectrum with a floored one.
        gam_linalg::matrix::symmetrize_in_place(&mut outer_h);
        if let Some(((row, col), value)) =
            outer_h.indexed_iter().find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "outer rho curvature entry ({row},{col}) is non-finite: {value}"
            )));
        }
        let rho_adj = gam_linalg::utils::certified_symmetric_solve(
            &outer_h,
            &alpha,
            "block Gaussian REML outer-rho adjoint",
        )
        .map(gam_linalg::utils::CertifiedSymmetricSolution::into_solution)
        .map_err(|error| {
            EstimationError::InvalidInput(format!(
                "block Gaussian REML outer-rho adjoint is not exactly solvable: {error}"
            ))
        })?;
        if let Some((block, value)) = rho_adj
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(EstimationError::InvalidInput(format!(
                "outer rho adjoint for block {block} is non-finite: {value}"
            )));
        }
        let weighted_b_sum = rho_adj
            .iter()
            .zip(b_values.iter())
            .map(|(&zk, &bk)| zk * bk)
            .sum::<f64>();
        q_kernel += 0.5 * tau_q * weighted_b_sum;
        for block in 0..f_blocks {
            let zk = rho_adj[block];
            if zk == 0.0 {
                continue;
            }
            g_kernel -= &(rpr_matrices[block].clone() * (0.5 * zk));
            let m = &m_vectors[block];
            for i in 0..p_total {
                h_kernel[i] += tau * zk * m[i];
                for j in 0..p_total {
                    g_kernel[[i, j]] -= 0.5 * tau * zk * (beta[i] * m[j] + m[i] * beta[j]);
                }
            }
            let start = offsets[block];
            let end = offsets[block + 1];
            j_blocks[block] += &(r.slice(s![start..end, start..end]).to_owned() * (0.5 * zk));
            for i in 0..(end - start) {
                for j in 0..(end - start) {
                    j_blocks[block][[i, j]] += 0.5 * tau * zk * beta[start + i] * beta[start + j];
                }
            }
        }
    }

    for row in 0..n {
        for col in 0..p_total {
            grad_z[[row, col]] += -2.0 * q_kernel * weighted_residual[row] * beta[col];
        }
    }
    let zg = z.dot(&g_kernel);
    for row in 0..n {
        for col in 0..p_total {
            grad_z[[row, col]] += 2.0 * weights[row] * zg[[row, col]];
        }
    }
    let wy = y.to_owned() * &weights.to_owned();
    for row in 0..n {
        for col in 0..p_total {
            grad_z[[row, col]] += wy[row] * h_kernel[col];
        }
    }

    let mut grad_y = Array2::<f64>::zeros((n, 1));
    let zh = z.dot(&h_kernel);
    for row in 0..n {
        grad_y[[row, 0]] = 2.0 * q_kernel * weighted_residual[row] + weights[row] * zh[row];
    }

    let mut grad_weights = Array1::<f64>::zeros(n);
    for row in 0..n {
        let diag_zgz = (0..p_total)
            .map(|col| z[[row, col]] * zg[[row, col]])
            .sum::<f64>();
        grad_weights[row] = q_kernel * residual[row] * residual[row] + diag_zgz + y[row] * zh[row];
    }

    let mut grad_penalties = Vec::with_capacity(f_blocks);
    for block in 0..f_blocks {
        let start = offsets[block];
        let end = offsets[block + 1];
        let mut local = g_kernel.slice(s![start..end, start..end]).to_owned();
        for i in 0..(end - start) {
            for j in 0..(end - start) {
                local[[i, j]] += q_kernel * beta[start + i] * beta[start + j];
            }
        }
        local += &j_blocks[block];
        local *= lambdas[block];
        gam_linalg::matrix::symmetrize_in_place(&mut local);
        grad_penalties.push(local);
    }

    let mut grad_designs = Vec::with_capacity(f_blocks);
    for block in 0..f_blocks {
        grad_designs.push(
            grad_z
                .slice(s![.., offsets[block]..offsets[block + 1]])
                .to_owned(),
        );
    }

    Ok(GaussianRemlBlocksBackwardAnalytic {
        grad_designs,
        grad_penalties,
        grad_y,
        grad_weights,
    })
}

/// Fixed-λ multi-output Gaussian fit under a per-row dense Fisher–Rao precision
/// metric: coefficients, fitted values, per-output residual scale, and the
/// penalized Fisher-weighted objective.
pub struct DenseFisherGaussianFit {
    pub coefficients: Array2<f64>,
    pub fitted: Array2<f64>,
    pub sigma2: Array1<f64>,
    pub objective: f64,
}

/// Add a block-diagonal `λ·S` penalty (one `S` block per output) into a stacked
/// `(k·n_outputs)` Hessian in place, symmetrizing `S`.
pub fn add_block_diagonal_penalty(
    hessian: &mut Array2<f64>,
    penalty: ArrayView2<'_, f64>,
    lambda: f64,
    n_outputs: usize,
) -> Result<(), EstimationError> {
    let k = penalty.ncols();
    if penalty.nrows() != k {
        return Err(EstimationError::InvalidInput(format!(
            "penalty must be square for dense Fisher fit; got {}x{}",
            penalty.nrows(),
            penalty.ncols()
        )));
    }
    if hessian.dim() != (k * n_outputs, k * n_outputs) {
        return Err(EstimationError::InvalidInput(
            "dense Fisher Hessian shape mismatch while adding penalty".to_string(),
        ));
    }
    for output in 0..n_outputs {
        let offset = output * k;
        for row in 0..k {
            for col in 0..k {
                let s_sym = 0.5 * (penalty[[row, col]] + penalty[[col, row]]);
                hessian[[offset + row, offset + col]] += lambda * s_sym;
            }
        }
    }
    Ok(())
}

/// Closed-form fixed-λ multi-output Gaussian fit with a per-row dense Fisher–Rao
/// precision metric. Assembles the block `XᵀWX` (+ block-diagonal `λS`) and
/// `XᵀWY` via the dense Fisher block kernels, solves, then forms fitted values,
/// per-output residual scale `sigma2`, and the penalized Fisher-weighted
/// objective seeded by `latent_prior_score`. `row_weights` are the (already
/// resolved) per-observation likelihood weights.
pub fn dense_fisher_gaussian_fit(
    design: ArrayView2<'_, f64>,
    y: ArrayView2<'_, f64>,
    penalty: ArrayView2<'_, f64>,
    row_weights: ArrayView1<'_, f64>,
    fisher_w: ArrayView3<'_, f64>,
    lambda: f64,
    latent_prior_score: f64,
) -> Result<DenseFisherGaussianFit, EstimationError> {
    let n_obs = design.nrows();
    let k = design.ncols();
    let n_outputs = y.ncols();
    let mut hessian = crate::pirls::dense_block_xtwx(design, fisher_w, Some(row_weights))?;
    add_block_diagonal_penalty(&mut hessian, penalty, lambda, n_outputs)?;
    let rhs = crate::pirls::dense_block_xtwy(design, fisher_w, y, Some(row_weights))?;
    let beta_vec =
        gam_linalg::utils::solve_dense_block_system(&hessian, &rhs, "dense Fisher Gaussian")
            .map_err(EstimationError::InvalidInput)?;
    let mut coefficients = Array2::<f64>::zeros((k, n_outputs));
    for output in 0..n_outputs {
        for col in 0..k {
            coefficients[[col, output]] = beta_vec[output * k + col];
        }
    }
    let fitted = design.dot(&coefficients);
    let mut sigma2 = Array1::<f64>::zeros(n_outputs);
    let mut objective = latent_prior_score;
    for row in 0..n_obs {
        for a in 0..n_outputs {
            let ra = y[[row, a]] - fitted[[row, a]];
            sigma2[a] += row_weights[row] * ra * ra;
            for b in 0..n_outputs {
                objective += 0.5
                    * row_weights[row]
                    * ra
                    * fisher_w[[row, a, b]]
                    * (y[[row, b]] - fitted[[row, b]]);
            }
        }
    }
    for output in 0..n_outputs {
        sigma2[output] /= (n_obs.saturating_sub(k).max(1)) as f64;
        let beta_col = coefficients.column(output);
        let s_beta = penalty.dot(&beta_col);
        objective += 0.5 * lambda * beta_col.dot(&s_beta);
    }
    Ok(DenseFisherGaussianFit {
        coefficients,
        fitted,
        sigma2,
        objective,
    })
}

/// #2723 — the perfect-fit refusal must be a property of the DESIGN, not of the
/// sign of the last rounding.
///
/// Positive control, MEASURED rather than argued: with the bar reverted to the
/// bare `residual > 0.0` and nothing else changed, two of these three tests go
/// red, and the failure names the two designs the issue measured as wrongly
/// accepted —
///
/// ```text
///   A irrational basis, constant response:      residual 1.776357e-15,
///                                               ywy 5.880000e0,   resolution 4.177991e-13
///   B integer basis, penalized mass present:    residual 3.743049e-13,
///                                               ywy 1.650000e2,   resolution 9.379164e-12
/// ```
///
/// while `a_genuine_residual_is_accepted_at_every_scale` stays green, so the
/// reverted bar fails for the reason under test rather than by refusing (or
/// accepting) everything. Both accepted residuals sit two to four orders BELOW
/// their own resolution — the old predicate was reading debris, and the margin
/// by which it was doing so is what these numbers record.
#[cfg(test)]
mod perfect_fit_refusal_tests {
    use super::*;
    use ndarray::array;

    /// Build the four designs of #2723. Every one of them has a residual that is
    /// EXACTLY zero: each response lies exactly in its design's column span.
    /// They differ only in how the floating-point debris of `ywy − Σc²` lands
    /// and in whether any penalized direction carries mass — the two accidents
    /// the old `residual > 0.0` bar was actually reading.
    fn zero_residual_designs() -> Vec<(&'static str, Array2<f64>, Array2<f64>, Array2<f64>)> {
        // A: constant response on an irrational (periodic-harmonic) basis. The
        // cancellation is INEXACT and landed POSITIVE (+1.776e-15), which is the
        // only reason the old bar accepted it.
        let n = 12usize;
        let a_x = Array2::<f64>::from_shape_fn((n, 3), |(row, col)| {
            let t = 2.0 * std::f64::consts::PI * (row as f64) / (n as f64);
            match col {
                0 => 1.0,
                1 => t.sin(),
                _ => t.cos(),
            }
        });
        let a_y = Array2::<f64>::from_elem((n, 1), 0.7);
        let a_penalty = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

        // B: `y = X·[1, 2]` in integers, with mass on the penalized direction.
        // The cancellation landed negative and was clamped to `0`, but
        // `Σc²·u(RHO_LOWER)` is strictly positive for ANY design carrying
        // penalized mass — the generic case — so the old bar accepted it too.
        let b_x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
        let b_y = array![[1.0], [3.0], [5.0], [7.0], [9.0]];
        let b_penalty = array![[0.0, 0.0], [0.0, 1.0]];

        // C: constant response on an exact integer basis, all mass in `null(S)`.
        // Statistically identical to A; the old bar refused it purely because the
        // integer basis put the debris on the other side and left `Σc²·u = 0`.
        let c_x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
        let c_y = array![[1.0], [1.0], [1.0], [1.0], [1.0]];
        let c_penalty = array![[0.0, 0.0], [0.0, 1.0]];

        // D: identically zero response. Every term is exactly `0`; refused.
        let d_x = c_x.clone();
        let d_y = Array2::<f64>::zeros((5, 1));
        let d_penalty = c_penalty.clone();

        vec![
            ("A irrational basis, constant response", a_x, a_y, a_penalty),
            ("B integer basis, penalized mass present", b_x, b_y, b_penalty),
            ("C integer basis, all mass in null(S)", c_x, c_y, c_penalty),
            ("D identically zero response", d_x, d_y, d_penalty),
        ]
    }

    /// All four zero-residual designs must reach the SAME verdict, and that
    /// verdict must be refusal: a profiled Gaussian likelihood has no finite
    /// scale for an exactly-interpolated response, so every candidate ties at
    /// `−∞` and abstention is the only defensible outcome.
    #[test]
    fn every_zero_residual_design_is_refused_alike() {
        let mut failures: Vec<String> = Vec::new();
        let mut verdicts: Vec<(&'static str, bool)> = Vec::new();

        for (name, x, y, penalty) in zero_residual_designs() {
            let prepared =
                prepare_gaussian_reml(x.view(), y.view(), penalty.view(), None, None, None)
                    .unwrap_or_else(|error| panic!("{name}: preparation failed: {error}"));

            // Regime clause. A pass below is meaningless unless the design is
            // actually in the perfect-fit regime, so report the decomposition
            // the validator itself reads rather than trusting the construction.
            let DispersionResidualParts {
                unpenalized_residual,
                penalized_residual,
                ..
            } = dispersion_residual_parts(
                &prepared.cache,
                prepared.ywy.view(),
                prepared.projected_rhs_squared.view(),
                0,
                RHO_LOWER,
            );
            let residual = unpenalized_residual + penalized_residual;
            let ywy = prepared.ywy[0];
            let resolution = profile_residual_resolution(&prepared.cache, ywy);
            if !(residual <= f64::EPSILON.sqrt() * ywy.max(1.0)) {
                failures.push(format!(
                    "{name}: REGIME — residual {residual:.6e} against ywy {ywy:.6e} is far above \
                     the cancellation scale, so this design does NOT interpolate its response and \
                     the fixture has drifted out of the regime under test"
                ));
            }

            let verdict = validate_reml_profile_residuals(
                &prepared.cache,
                prepared.ywy.view(),
                prepared.projected_rhs_squared.view(),
                RHO_LOWER,
            );
            verdicts.push((name, verdict.is_ok()));
            if verdict.is_ok() {
                failures.push(format!(
                    "{name}: ACCEPTED a residual of {residual:.6e} (ywy {ywy:.6e}, resolution \
                     {resolution:.6e}) whose true value is exactly zero; the profiled dispersion \
                     it carries is pure roundoff"
                ));
            }
        }

        let accepted: Vec<&str> = verdicts
            .iter()
            .filter(|(_, ok)| *ok)
            .map(|(name, _)| *name)
            .collect();
        assert!(
            failures.is_empty(),
            "#2723: the four exactly-zero-residual designs did not agree on refusal. \
             Accepted: {accepted:?}. Details:\n  - {}",
            failures.join("\n  - ")
        );
    }

    /// Non-vacuity control: the same bar must ACCEPT a design with a genuine
    /// residual, at both a small and a large response scale. Without this, a
    /// validator that refuses everything would pass the test above.
    #[test]
    fn a_genuine_residual_is_accepted_at_every_scale() {
        let x = array![[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [1.0, 4.0]];
        let base_y = array![[1.1], [2.9], [5.2], [6.8], [9.1]];
        let penalty = array![[0.0, 0.0], [0.0, 1.0]];

        for scale in [1.0e-6, 1.0, 1.0e6] {
            let y = base_y.mapv(|value| value * scale);
            let prepared =
                prepare_gaussian_reml(x.view(), y.view(), penalty.view(), None, None, None)
                    .expect("the control design is finite and full rank");
            let DispersionResidualParts {
                unpenalized_residual,
                penalized_residual,
                ..
            } = dispersion_residual_parts(
                &prepared.cache,
                prepared.ywy.view(),
                prepared.projected_rhs_squared.view(),
                0,
                RHO_LOWER,
            );
            let residual = unpenalized_residual + penalized_residual;
            let ywy = prepared.ywy[0];
            assert!(
                residual > f64::EPSILON.sqrt() * ywy,
                "control at scale {scale:e}: residual {residual:.6e} is at cancellation scale \
                 against ywy {ywy:.6e}, so this is not a genuine-residual control"
            );
            let verdict = validate_reml_profile_residuals(
                &prepared.cache,
                prepared.ywy.view(),
                prepared.projected_rhs_squared.view(),
                RHO_LOWER,
            );
            assert!(
                verdict.is_ok(),
                "control at scale {scale:e}: a genuine residual {residual:.6e} (ywy {ywy:.6e}) was \
                 refused: {:?}",
                verdict.err()
            );
        }
    }

    /// The bar is scale-invariant by construction (it is proportional to `ywy`),
    /// so rescaling a zero-residual response must not move the verdict.
    #[test]
    fn the_refusal_is_invariant_to_the_response_scale() {
        for (name, x, y, penalty) in zero_residual_designs() {
            for scale in [1.0e-8, 1.0, 1.0e8] {
                let scaled = y.mapv(|value| value * scale);
                let prepared =
                    prepare_gaussian_reml(x.view(), scaled.view(), penalty.view(), None, None, None)
                        .unwrap_or_else(|error| panic!("{name} at {scale:e}: {error}"));
                let verdict = validate_reml_profile_residuals(
                    &prepared.cache,
                    prepared.ywy.view(),
                    prepared.projected_rhs_squared.view(),
                    RHO_LOWER,
                );
                assert!(
                    verdict.is_err(),
                    "{name}: rescaling the response by {scale:e} flipped the perfect-fit verdict \
                     to ACCEPTED; the bar is not scale-invariant"
                );
            }
        }
    }
}

/// #2740: one eigenvalue array, one range/null predicate.
///
/// `cache.penalty_eigenvalues` used to be partitioned three different ways in
/// this file — the relative `δ > EIGEN_REL_TOL·max|δ|` that DEFINES
/// `penalty_rank`, an absolute `δ > 0.0`, and an absolute `δ == 0.0`. The
/// eigensolver returns a numerically null direction as a small POSITIVE number,
/// so those three answers disagree on exactly that direction: it is in the range
/// set by one test, out of the null set by another, and out of the range set by
/// the third. A `ln`-sum taken over one population and then differenced against
/// a count taken under another is wrong by precisely the terms the two
/// populations disagree about.
///
/// Every test below is built on a spectrum that carries such a DISPUTED band —
/// eigenvalues strictly positive and strictly below the range tolerance — so a
/// green here cannot come from a fixture on which the predicates happen to
/// agree; each test states its own non-vacuity control.
#[cfg(test)]
mod eigenvalue_range_predicate_agreement_2740_tests {
    use super::*;
    use ndarray::array;

    const LARGEST: f64 = 4.0;

    /// Range: `4.0` and `1.0`. Disputed (positive, below `4.0·1e-10 = 4e-10`):
    /// `5.0e-11` and `3.2e-18` — the second is the magnitude #2739 MEASURED on an
    /// ordinary second-difference penalty. Null: an exact `0.0`.
    ///
    /// `logdet_penalty_positive` is set to the log-determinant over the range
    /// directions, which is what `gaussian_penalty_positive_logdet` reconciles it
    /// to on a real cache: it is the quantity the compactified limit differences
    /// the eigenvalue sum against, so the fixture must denominate it the same way.
    fn disputed_band_cache() -> GaussianRemlEigenCache {
        let eigenvalues = array![LARGEST, 1.0, 5.0e-11, 3.2e-18, 0.0];
        let p = eigenvalues.len();
        GaussianRemlEigenCache {
            penalty_eigenvalues: eigenvalues,
            eigenvectors: Array2::eye(p),
            coefficient_basis: Array2::eye(p),
            xtwx_fingerprint: 0,
            penalty_fingerprint: 0,
            logdet_xtwx: 0.0,
            logdet_penalty_positive: LARGEST.ln() + 1.0_f64.ln(),
            penalty_rank: 2,
            nullity: 3,
        }
    }

    /// The classification and the rank are the same question asked once.
    #[test]
    fn the_range_count_the_null_count_and_penalty_rank_are_one_predicate() {
        let cache = disputed_band_cache();
        let spectrum = PenaltyRangeSpectrum::of(&cache);

        // The threshold is derived from the spectrum and the file's relative
        // rank constant, not chosen here.
        assert_eq!(
            spectrum.tolerance,
            LARGEST * EIGEN_REL_TOL,
            "the range threshold must be the relative one that defines penalty_rank"
        );

        // NON-VACUITY: without a disputed band `δ > 0.0` and the relative test
        // coincide and every assertion below would pass on a fixture that could
        // never have exhibited the defect.
        let absolute_positive = cache
            .penalty_eigenvalues
            .iter()
            .filter(|delta| **delta > 0.0)
            .count();
        assert!(
            absolute_positive > cache.penalty_rank,
            "precondition unmet: the fixture carries no positive-but-null direction \
             (penalty_rank={}, eigenvalues passing `> 0.0`={absolute_positive})",
            cache.penalty_rank
        );

        assert_eq!(
            spectrum.rank(),
            cache.penalty_rank,
            "the classified range count must be the rank the cache reports"
        );
        assert_eq!(
            spectrum.iter().filter(|delta| *delta > 0.0).count(),
            cache.penalty_rank,
            "a `δ > 0.0` read of the CLASSIFIED spectrum must select exactly the \
             directions penalty_rank counted"
        );
        assert_eq!(
            spectrum.iter().filter(|delta| *delta == 0.0).count(),
            cache.nullity,
            "the null set must be the exact complement of the range set"
        );
    }

    /// `V′` from the log-determinant term is `½d·(Σ t/(1+t) − penalty_rank)`. The
    /// sum and the offset are the same population, so as `λ→∞` the difference goes
    /// to zero: `t/(1+t) = 1 − 1/(1+t)` leaves exactly `½d·Σ_range 1/(1+λδ)`.
    /// Under the abandoned `δ > 0.0` the sum has more terms than the offset and
    /// the residual saturates at half a mode per disputed direction instead.
    #[test]
    fn the_large_rho_logdet_gradient_vanishes_because_sum_and_offset_share_a_population() {
        let cache = disputed_band_cache();
        let spectrum = PenaltyRangeSpectrum::of(&cache);
        let lambda = RHO_UPPER.exp();
        let n_outputs = 1.0_f64;

        let (term, _edf) = gaussian_reml_logdet_term(&cache, RHO_UPPER, n_outputs);

        // The bound is the analytic residual of the SAME sum, not a chosen
        // tolerance, widened by the accumulation of `rank` additions.
        let residual: f64 = spectrum
            .iter()
            .filter(|delta| *delta > 0.0)
            .map(|delta| 1.0 / (1.0 + lambda * delta))
            .sum();
        let bound = 0.5 * n_outputs * residual
            + ((spectrum.len() + 4) as f64) * f64::EPSILON * (cache.penalty_rank as f64);
        assert!(
            term.grad.abs() <= bound,
            "the large-λ log-determinant gradient is {} but the range-populated \
             residual bounds it by {bound:e}",
            term.grad
        );

        // NON-VACUITY: at ρ = RHO_UPPER the disputed band is saturated, so the
        // `δ > 0.0` population would have left most of a whole mode in the
        // gradient — the two predicates are separated here by far more than the
        // bound above.
        let disputed_trace: f64 = (0..spectrum.len())
            .filter(|index| spectrum.get(*index) == 0.0)
            .map(|index| {
                let t = lambda * cache.penalty_eigenvalues[index];
                t / (1.0 + t)
            })
            .sum();
        assert!(
            disputed_trace > 0.5,
            "precondition unmet: the disputed band contributes only {disputed_trace} to \
             the trace at rho={RHO_UPPER}, so an absolute `> 0.0` sum would barely differ \
             from the classified one and this test would be mute"
        );
        assert!(
            0.5 * n_outputs * disputed_trace > bound,
            "the disputed band's contribution {disputed_trace} does not clear the bound \
             {bound:e}; the assertion above cannot distinguish the two predicates"
        );
    }
}
