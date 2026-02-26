//! # Model Estimation via Penalized Likelihood and REML
//!
//! This module orchestrates the core model fitting procedure for Generalized Additive
//! Models (GAMs). It determines optimal smoothing parameters directly from the data,
//! moving beyond simple hyperparameter-driven models. This is achieved through a
//! nested optimization scheme, a standard approach for this class of models:
//!
//! 1.  Outer Loop (BFGS): Optimizes the log-smoothing parameters (`rho`) by
//!     maximizing a marginal likelihood criterion. For non-Gaussian models (e.g., Logit),
//!     this is the Laplace Approximate Marginal Likelihood (LAML). This advanced strategy
//!     is detailed in Wood (2011), upon which this implementation is heavily based. The
//!     BFGS algorithm itself is a classic quasi-Newton method, with our implementation
//!     following the standard described in Nocedal & Wright (2006).
//!
//! 2.  Inner Loop (P-IRLS): For each set of trial smoothing parameters from the
//!     outer loop, this routine finds the corresponding model coefficients (`beta`) by
//!     running a Penalized Iteratively Reweighted Least Squares (P-IRLS) algorithm
//!     to convergence.
//!
//! This two-tiered structure allows the model to learn the appropriate complexity for
//! each smooth term directly from the data.

// External Crate for Optimization
use wolfe_bfgs::{Bfgs, BfgsSolution};

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use self::internal::RemlState;

// Crate-level imports
use crate::construction::{
    ReparamInvariant, calculate_condition_number, compute_penalty_square_roots,
    create_balanced_penalty_root, precompute_reparam_invariant,
};
use crate::matrix::DesignMatrix;
use crate::pirls::{self, PirlsResult};
use crate::probability::{inverse_link_array, standard_normal_quantile};
use crate::seeding::{SeedConfig, SeedStrategy, generate_rho_candidates};
use crate::types::{
    Coefficients, LinkFunction, LogSmoothingParams, LogSmoothingParamsView, RidgeDeterminantMode,
    RidgePassport,
};

// Ndarray and faer linear algebra helpers
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, Axis, Zip, s};
// faer: high-performance dense solvers
use crate::faer_ndarray::{
    FaerArrayView, FaerCholesky, FaerEigh, FaerLinalgError, array2_to_mat_mut, fast_ab, fast_ata,
    fast_atb,
};
use faer::Mat as FaerMat;
use faer::Side;
use faer::linalg::solvers::{
    Lblt as FaerLblt, Ldlt as FaerLdlt, Llt as FaerLlt, Solve as FaerSolve,
};

use crate::diagnostics::{
    approx_f64, format_compact_series, format_cond, format_range, quantize_value, quantize_vec,
    should_emit_h_min_eig_diag,
};

// Note: deflate_weights_by_se was removed. We now use integrated (GHQ) likelihood
// instead of weight deflation. See update_glm_vectors_integrated in pirls.rs.
// The SE is passed through to PIRLS which properly integrates over uncertainty
// in the likelihood, rather than using ad-hoc weight adjustment.

fn faer_frob_inner(a: faer::MatRef<'_, f64>, b: faer::MatRef<'_, f64>) -> f64 {
    let (m, n) = (a.nrows(), a.ncols());
    // Hot-path optimization:
    // - small matrices: plain accumulation is faster and sufficiently accurate
    // - larger matrices: compensated (Kahan) sum protects trace/EDF numerics
    const KAHAN_SWITCH_ELEMS: usize = 10_000;
    let elem_count = m.saturating_mul(n);
    if elem_count < KAHAN_SWITCH_ELEMS {
        let mut sum = 0.0_f64;
        for j in 0..n {
            for i in 0..m {
                sum += a[(i, j)] * b[(i, j)];
            }
        }
        sum
    } else {
        let mut sum = KahanSum::default();
        for j in 0..n {
            for i in 0..m {
                sum.add(a[(i, j)] * b[(i, j)]);
            }
        }
        sum.sum()
    }
}

#[derive(Default, Clone, Copy)]
struct KahanSum {
    sum: f64,
    c: f64,
}

impl KahanSum {
    fn add(&mut self, value: f64) {
        let y = value - self.c;
        let t = self.sum + y;
        self.c = (t - self.sum) - y;
        self.sum = t;
    }

    fn sum(self) -> f64 {
        self.sum
    }
}

fn kahan_sum<I>(iter: I) -> f64
where
    I: IntoIterator<Item = f64>,
{
    let mut acc = KahanSum::default();
    for value in iter {
        acc.add(value);
    }
    acc.sum()
}

fn map_hessian_to_original_basis(
    pirls: &crate::pirls::PirlsResult,
) -> Result<Array2<f64>, EstimationError> {
    let qs = &pirls.reparam_result.qs;
    let h_t = &pirls.penalized_hessian_transformed;
    let tmp = qs.dot(h_t);
    Ok(tmp.dot(&qs.t()))
}

fn matrix_inverse_with_regularization(matrix: &Array2<f64>, label: &str) -> Option<Array2<f64>> {
    let p = matrix.nrows();
    if p == 0 || matrix.ncols() != p {
        return None;
    }

    enum Fact {
        Llt(FaerLlt<f64>),
        Ldlt(FaerLdlt<f64>),
        Lblt(FaerLblt<f64>),
    }
    impl Fact {
        fn solve_in_place(&self, rhs: faer::MatMut<'_, f64>) {
            match self {
                Fact::Llt(f) => f.solve_in_place(rhs),
                Fact::Ldlt(f) => f.solve_in_place(rhs),
                Fact::Lblt(f) => f.solve_in_place(rhs),
            }
        }
    }

    let mut planner = RidgePlanner::new(matrix);
    let factor = loop {
        let ridge = planner.ridge();
        let h_eff = if ridge > 0.0 {
            add_ridge(matrix, ridge)
        } else {
            matrix.clone()
        };
        let h_view = FaerArrayView::new(&h_eff);
        if let Ok(chol) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
            break Fact::Llt(chol);
        }
        if let Ok(ldlt) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
            break Fact::Ldlt(ldlt);
        }
        if planner.attempts() >= MAX_FACTORIZATION_ATTEMPTS {
            log::warn!(
                "Falling back to LDLT pivoted inverse for {} after ridge {:.3e}",
                label,
                ridge
            );
            if let Ok(h_lblt) = std::panic::catch_unwind({
                let h_view = FaerArrayView::new(&h_eff);
                move || FaerLblt::new(h_view.as_ref(), Side::Lower)
            }) {
                break Fact::Lblt(h_lblt);
            }
            log::warn!("Failed to factorize {} for covariance", label);
            return None;
        }
        planner.bump_with_matrix(matrix);
    };

    let mut inv = Array2::<f64>::eye(p);
    let mut inv_view = array2_to_mat_mut(&mut inv);
    factor.solve_in_place(inv_view.as_mut());

    // Numerical solves can leave tiny asymmetry; enforce symmetry explicitly.
    for i in 0..p {
        for j in (i + 1)..p {
            let avg = 0.5 * (inv[[i, j]] + inv[[j, i]]);
            inv[[i, j]] = avg;
            inv[[j, i]] = avg;
        }
    }
    Some(inv)
}

fn se_from_covariance(cov: &Array2<f64>) -> Array1<f64> {
    let p = cov.nrows().min(cov.ncols());
    let mut se = Array1::<f64>::zeros(p);
    for i in 0..p {
        se[i] = cov[[i, i]].max(0.0).sqrt();
    }
    se
}

fn apply_family_inverse_link(
    eta: &Array1<f64>,
    family: crate::types::LikelihoodFamily,
) -> Result<Array1<f64>, EstimationError> {
    if matches!(family, crate::types::LikelihoodFamily::RoystonParmar) {
        return Err(EstimationError::InvalidInput(
            "prediction uncertainty for RoystonParmar is not available in predict_gam".to_string(),
        ));
    }
    Ok(inverse_link_array(family, eta.view()))
}

fn linear_predictor_variance(x: &DesignMatrix, cov: &Array2<f64>) -> Array1<f64> {
    match x {
        DesignMatrix::Dense(xd) => {
            let xc = xd.dot(cov);
            let mut out = Array1::<f64>::zeros(xd.nrows());
            for i in 0..xd.nrows() {
                out[i] = xd.row(i).dot(&xc.row(i)).max(0.0);
            }
            out
        }
        DesignMatrix::Sparse(xs) => {
            let mut out = Array1::<f64>::zeros(xs.nrows());
            if let Ok(csr) = xs.as_ref().to_row_major() {
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                for i in 0..xs.nrows() {
                    let start = row_ptr[i];
                    let end = row_ptr[i + 1];
                    let mut acc = 0.0_f64;
                    for a in start..end {
                        let j = col_idx[a];
                        let xij = vals[a];
                        for b in start..end {
                            let k = col_idx[b];
                            let xik = vals[b];
                            acc += xij * cov[[j, k]] * xik;
                        }
                    }
                    out[i] = acc.max(0.0);
                }
            } else {
                let dense_arc = x.to_dense_arc();
                let dense = dense_arc.as_ref();
                let xc = dense.dot(cov);
                for i in 0..dense.nrows() {
                    out[i] = dense.row(i).dot(&xc.row(i)).max(0.0);
                }
            }
            out
        }
    }
}

#[derive(Clone, Copy)]
struct RemlConfig {
    link_function: LinkFunction,
    convergence_tolerance: f64,
    max_iterations: usize,
    reml_convergence_tolerance: f64,
    reml_max_iterations: u64,
    firth_bias_reduction: bool,
    objective_consistent_fd_gradient: bool,
}

impl RemlConfig {
    fn external(
        link_function: LinkFunction,
        reml_tol: f64,
        reml_max_iter: usize,
        firth_bias_reduction: bool,
    ) -> Self {
        Self {
            link_function,
            convergence_tolerance: reml_tol,
            max_iterations: 500,
            reml_convergence_tolerance: reml_tol,
            reml_max_iterations: reml_max_iter as u64,
            firth_bias_reduction,
            // Use analytic outer gradients for external fits to avoid
            // expensive FD sweeps that repeatedly re-run PIRLS.
            objective_consistent_fd_gradient: false,
        }
    }

    fn link_function(&self) -> LinkFunction {
        self.link_function
    }

    fn as_pirls_config(&self) -> pirls::PirlsConfig {
        pirls::PirlsConfig {
            link_function: self.link_function,
            max_iterations: self.max_iterations,
            convergence_tolerance: self.convergence_tolerance,
            firth_bias_reduction: self.firth_bias_reduction,
        }
    }
}
const HESSIAN_CONDITION_TARGET: f64 = 1e10;

fn max_abs_diag(matrix: &Array2<f64>) -> f64 {
    matrix
        .diag()
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max)
        .max(1.0)
}

fn add_ridge(matrix: &Array2<f64>, ridge: f64) -> Array2<f64> {
    if ridge <= 0.0 {
        return matrix.clone();
    }
    let mut regularized = matrix.clone();
    let n = regularized.nrows();
    for i in 0..n {
        regularized[[i, i]] += ridge;
    }
    regularized
}

#[derive(Clone)]
struct RidgePlanner {
    cond_estimate: Option<f64>,
    ridge: f64,
    attempts: usize,
    scale: f64,
}

impl RidgePlanner {
    fn new(matrix: &Array2<f64>) -> Self {
        let scale = max_abs_diag(matrix);
        let min_step = scale * 1e-10;
        let cond_estimate = calculate_condition_number(matrix)
            .ok()
            .filter(|c| c.is_finite() && *c > 0.0);
        let mut ridge = min_step;
        if let Some(cond) = cond_estimate {
            if !cond.is_finite() {
                ridge = scale * 1e-8;
            } else if cond > HESSIAN_CONDITION_TARGET {
                // If initial condition estimate is already above target, seed ridge
                // proportional to the excess so the first retry is meaningful.
                ridge = min_step * (cond / HESSIAN_CONDITION_TARGET);
            }
        } else {
            ridge = scale * 1e-8;
        }
        ridge = ridge.max(min_step);
        Self {
            cond_estimate,
            ridge,
            attempts: 0,
            scale,
        }
    }

    fn ridge(&self) -> f64 {
        self.ridge
    }

    fn cond_estimate(&self) -> Option<f64> {
        self.cond_estimate
    }

    #[inline]
    fn estimate_condition_with_ridge(&self, matrix: &Array2<f64>, ridge: f64) -> Option<f64> {
        let regularized = if ridge > 0.0 {
            add_ridge(matrix, ridge)
        } else {
            matrix.clone()
        };
        calculate_condition_number(&regularized)
            .ok()
            .filter(|c| c.is_finite() && *c > 0.0)
    }

    fn bump_with_matrix(&mut self, matrix: &Array2<f64>) {
        self.attempts += 1;
        let min_step = self.scale * 1e-10;
        let base = self.ridge.max(min_step);

        // Estimate conditioning at the current ridge level.
        let cond_now = self.estimate_condition_with_ridge(matrix, base);
        self.cond_estimate = cond_now;

        self.ridge = if let Some(cond) = cond_now {
            let ratio = cond / HESSIAN_CONDITION_TARGET;
            // Primary update from condition feedback.
            // sqrt-ratio avoids wild overshoot while still scaling with severity.
            let mut multiplier = if ratio > 1.0 {
                ratio.sqrt().clamp(1.5, 10.0)
            } else {
                // Factorization failed despite "acceptable" condition number.
                // This usually indicates indefiniteness/numerical fragility, so
                // use a stronger fallback than ×2, increasing with attempts.
                (2.0 + self.attempts as f64).clamp(3.0, 10.0)
            };

            let mut proposal = base * multiplier;
            // Verify whether the proposal actually improves condition enough.
            // If not, escalate once more before returning.
            if let Some(cond_next) = self.estimate_condition_with_ridge(matrix, proposal) {
                if cond_next > cond * 0.9 && ratio > 1.0 {
                    multiplier = (multiplier * 1.8).clamp(2.0, 10.0);
                    proposal = base * multiplier;
                }
            }
            proposal.max(min_step)
        } else if self.ridge <= 0.0 {
            min_step
        } else {
            // Condition estimate unavailable: geometric fallback.
            (base * 10.0).max(min_step)
        };

        if !self.ridge.is_finite() || self.ridge <= 0.0 {
            self.ridge = self.scale;
        }
    }

    fn attempts(&self) -> usize {
        self.attempts
    }
}

const MAX_FACTORIZATION_ATTEMPTS: usize = 4;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use thiserror::Error;

const LAML_RIDGE: f64 = 1e-8;
/// Smallest penalized deviance value we allow when profiling the Gaussian scale.
/// Prevents logarithms and divisions by nearly-zero D_p from destabilizing the
/// REML objective and its gradient in near-perfect-fit regimes.
const DP_FLOOR: f64 = 1e-12;
/// Width for the smooth deviance floor transition.
///
/// Kept generous (1e-8) so that finite-difference probes cannot straddle a
/// sharp kink when the penalized deviance is near zero, yet still tiny relative
/// to the typical residual sums of squares encountered during estimation.
const DP_FLOOR_SMOOTH_WIDTH: f64 = 1e-8;
// Unified rho bound corresponding to lambda in [exp(-RHO_BOUND), exp(RHO_BOUND)].
// Additional headroom reduces frequent contact with the hard box constraints.
const RHO_BOUND: f64 = 30.0;
// Soft interior prior on rho near the box boundaries.
const RHO_SOFT_PRIOR_WEIGHT: f64 = 1e-6;
const RHO_SOFT_PRIOR_SHARPNESS: f64 = 4.0;
const MAX_CONSECUTIVE_INNER_ERRORS: usize = 3;
// Adaptive cubature guardrails for bounded correction latency.
const AUTO_CUBATURE_MAX_RHO_DIM: usize = 12;
const AUTO_CUBATURE_MAX_EIGENVECTORS: usize = 4;
const AUTO_CUBATURE_TARGET_VAR_FRAC: f64 = 0.95;
const AUTO_CUBATURE_MAX_BETA_DIM: usize = 1600;
const AUTO_CUBATURE_BOUNDARY_MARGIN: f64 = 2.0;

#[inline]
fn stable_atanh(x: f64) -> f64 {
    // Use a formulation that remains accurate for |x| close to 1 while
    // avoiding spurious infinities from catastrophic cancellation.
    //
    // atanh(x) = 0.5 * [ln(1 + x) - ln(1 - x)]
    0.5 * ((1.0 + x).ln() - (1.0 - x).ln())
}

#[inline]
fn next_toward_zero(x: f64) -> f64 {
    if x == 0.0 {
        0.0
    } else if x > 0.0 {
        f64::from_bits(x.to_bits() - 1)
    } else {
        // For negative values, decreasing the bit pattern moves toward +0.0.
        f64::from_bits(x.to_bits() - 1)
    }
}

#[inline]
fn to_z_from_rho(rho: &Array1<f64>) -> Array1<f64> {
    rho.mapv(|r| {
        // Map bounded rho ∈ [-RHO_BOUND, RHO_BOUND] to unbounded z via z = RHO_BOUND * atanh(r/RHO_BOUND)
        let ratio = r / RHO_BOUND;
        let xr = if ratio <= -1.0 {
            next_toward_zero(-1.0)
        } else if ratio >= 1.0 {
            next_toward_zero(1.0)
        } else {
            ratio
        };
        let z = RHO_BOUND * stable_atanh(xr);
        z.clamp(-1e6, 1e6)
    })
}

#[inline]
fn to_rho_from_z(z: &Array1<f64>) -> Array1<f64> {
    z.mapv(|v| {
        let scaled = v / RHO_BOUND;
        RHO_BOUND * scaled.tanh()
    })
}

#[inline]
fn jacobian_drho_dz_from_rho(rho: &Array1<f64>) -> Array1<f64> {
    rho.mapv(|r| {
        // Numerical guard: can be slightly negative near the walls; clamp to [0, 1].
        (1.0 - (r / RHO_BOUND).powi(2)).max(0.0)
    })
}

#[cfg(test)]
mod rho_mapping_tests {
    use super::{RHO_BOUND, next_toward_zero, to_rho_from_z, to_z_from_rho};
    use ndarray::arr1;

    #[test]
    fn next_toward_zero_moves_toward_zero_for_both_signs() {
        let p = next_toward_zero(1.0);
        let n = next_toward_zero(-1.0);
        assert!(p.is_finite() && n.is_finite());
        assert!(p < 1.0 && p > 0.0, "positive side should move inward");
        assert!(n > -1.0 && n < 0.0, "negative side should move inward");
    }

    #[test]
    fn rho_to_z_is_finite_at_box_boundaries() {
        let rho = arr1(&[-RHO_BOUND, RHO_BOUND]);
        let z = to_z_from_rho(&rho);
        assert!(z.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn rho_z_roundtrip_stays_finite_and_within_bounds() {
        let rho = arr1(&[-RHO_BOUND, 0.0, RHO_BOUND]);
        let z = to_z_from_rho(&rho);
        let rho_back = to_rho_from_z(&z);
        assert!(rho_back.iter().all(|v| v.is_finite()));
        assert!(rho_back.iter().all(|v| v.abs() <= RHO_BOUND));
    }
}

#[inline]
fn project_rho_gradient(rho: &Array1<f64>, grad: &mut Array1<f64>) {
    let tol = 1e-8;
    for i in 0..rho.len() {
        if rho[i] <= -RHO_BOUND + tol && grad[i] > 0.0 {
            grad[i] = 0.0;
        }
        if rho[i] >= RHO_BOUND - tol && grad[i] < 0.0 {
            grad[i] = 0.0;
        }
    }
}

#[inline]
fn grad_norm_in_z_space(rho: &Array1<f64>, grad_rho: &Array1<f64>) -> f64 {
    let jac = jacobian_drho_dz_from_rho(rho);
    let grad_z = grad_rho * &jac;
    grad_z.dot(&grad_z).sqrt()
}

/// Smooth approximation of `max(dp, DP_FLOOR)` that is differentiable.
///
/// Returns the smoothed value and its derivative with respect to `dp`.
fn smooth_floor_dp(dp: f64) -> (f64, f64) {
    // Degenerate tau would reduce to the original hard max; guard against it.
    let tau = DP_FLOOR_SMOOTH_WIDTH.max(f64::EPSILON);
    let scaled = (dp - DP_FLOOR) / tau;

    // Stable softplus implementation.
    let softplus = if scaled > 20.0 {
        scaled + (-scaled).exp()
    } else if scaled < -20.0 {
        scaled.exp()
    } else {
        (1.0 + scaled.exp()).ln()
    };

    // Logistic function (softplus derivative) evaluated stably.
    let sigma = if scaled >= 0.0 {
        let exp_neg = (-scaled).exp();
        1.0 / (1.0 + exp_neg)
    } else {
        let exp_pos = scaled.exp();
        exp_pos / (1.0 + exp_pos)
    };

    let dp_c = DP_FLOOR + tau * softplus;
    (dp_c, sigma)
}

/// Compute the smoothing parameter uncertainty correction matrix `V_corr = J * V_rho * J^T`.
///
/// This implements the Wood et al. (2016) correction for smoothing parameter uncertainty.
/// The corrected covariance for `beta` is: `V*_beta = V_beta + J * V_rho * J^T`.
/// where:
/// - `V_beta = H^{-1}` (conditional covariance treating `lambda` as fixed)
/// - `J = d(beta)/d(rho)` (Jacobian wrt log-smoothing parameters)
/// - `V_rho = (d^2 LAML / d rho^2)^{-1}` (outer covariance)
///
/// Returns the correction matrix in the ORIGINAL coefficient basis.
///
/// FULL CORRECTION REFERENCE
/// -------------------------
/// Let `rho ~ N(mu, Sigma)` with `mu = rho_hat`, `Sigma = V_rho`,
/// and define:
/// - `A(rho) = H_rho^{-1}`
/// - `b(rho) = beta_hat_rho`
///
/// The exact Gaussian-mixture identity is:
///   `Var(beta) = E[A(rho)] + Var(b(rho))`.
///
/// Around `mu`, this routine keeps the first-order terms:
///
///   `E[A(rho)]      ~= A(mu) = H_mu^{-1}`
///   `Var(b(rho))    ~= J Sigma J^T`
///   `Var(beta)      ~= H_mu^{-1} + J V_rho J^T`.
///
/// Equivalent first-order propagation around the outer optimum `rho*`:
///
///   `Var(beta_hat) ~= Var(beta_hat | rho_hat) + (d beta_hat / d rho) Var(rho_hat) (d beta_hat / d rho)^T`
///                  `= V_beta + J V_rho J^T`.
///
/// Components:
///   `J[:,k] = d(beta_hat)/d(rho_k) = -H^{-1}(A_k beta_hat),  A_k = exp(rho_k) S_k`
///   `V_rho  = (d^2 V / d rho^2 at rho*)^{-1}`
///
/// Exact non-Gaussian V_ρ^{-1} requires the full Hessian with:
///   - tr(H^{-1}H_{kℓ})
///   - tr(H^{-1}H_k H^{-1}H_ℓ)
///   - pseudo-det second derivatives in S
///   - and H_{kℓ} terms containing fourth-likelihood derivatives.
///
/// This routine estimates V_ρ^{-1} by finite-differencing
/// the implemented analytic gradient and then regularizing before inversion.
///
/// Notes on omitted higher-order terms:
/// - The exact `E[A(rho)]` and `Var(b(rho))` can be written with the Gaussian
///   smoothing/heat operator `exp(0.5 * Delta_Sigma)` (equivalently Wick/Isserlis
///   contractions of high-order derivatives).
/// - Those infinite-series corrections are not expanded in this routine.
pub(crate) fn compute_smoothing_correction(
    reml_state: &internal::RemlState<'_>,
    final_rho: &Array1<f64>,
    final_fit: &pirls::PirlsResult,
) -> Option<Array2<f64>> {
    use crate::faer_ndarray::{FaerCholesky, FaerEigh};
    use faer::Side;

    let n_rho = final_rho.len();
    if n_rho == 0 {
        return None;
    }

    let n_coeffs_trans = final_fit.beta_transformed.len();
    let n_coeffs_orig = final_fit.reparam_result.qs.nrows();
    let lambdas: Array1<f64> = final_rho.mapv(f64::exp);

    // Step 1: Compute the Jacobian J = d(beta)/d(rho) in transformed space.
    //
    // Exact implicit-function identity at the inner optimum:
    //   dβ̂/dρ_k = -H^{-1}(S_k^ρ β̂),   S_k^ρ = λ_k S_k, λ_k = exp(ρ_k).
    //
    // In transformed coordinates with root penalties S_k = R_kᵀR_k:
    //   S_k β̂ = R_kᵀ(R_k β̂),
    // so each Jacobian column is one linear solve with H.

    // Get the effective Hessian from the fit - use stabilized version for consistency
    let h_trans = &final_fit.stabilized_hessian_transformed;

    // Factor the Hessian for solving
    let h_chol = match h_trans.clone().cholesky(Side::Lower) {
        Ok(c) => c,
        Err(_) => {
            log::warn!("Cholesky decomposition failed for smoothing correction; skipping.");
            return None;
        }
    };

    let beta_trans = final_fit.beta_transformed.as_ref();
    let rs_transformed = &final_fit.reparam_result.rs_transformed;

    // Build Jacobian matrix J where column k is dβ/dρ_k
    let mut jacobian_trans = Array2::<f64>::zeros((n_coeffs_trans, n_rho));
    for k in 0..n_rho {
        if k >= rs_transformed.len() {
            continue;
        }
        let r_k = &rs_transformed[k];
        if r_k.ncols() == 0 {
            continue;
        }
        // S_k β = R_k^T (R_k β)
        let r_beta = r_k.dot(beta_trans);
        let s_k_beta = r_k.t().dot(&r_beta);

        // dβ/dρ_k = -H^{-1}(λ_k S_k β)
        let rhs = s_k_beta.mapv(|v| -lambdas[k] * v);
        let delta = h_chol.solve_vec(&rhs);

        jacobian_trans.column_mut(k).assign(&delta);
    }

    // Step 2: Build V_rho by inverting the LAML Hessian in rho-space.
    // Prefer the exact analytic Hessian; fallback to finite differences.
    let mut hessian_rho = match reml_state.compute_laml_hessian_exact(&final_rho) {
        Ok(h) => h,
        Err(err) => {
            log::warn!(
                "Exact LAML Hessian unavailable ({}); falling back to FD Hessian for smoothing correction.",
                err
            );
            let h_step = 1e-4;
            let mut hessian_fd = Array2::<f64>::zeros((n_rho, n_rho));
            for k in 0..n_rho {
                let mut rho_plus = final_rho.clone();
                rho_plus[k] += h_step;
                let mut rho_minus = final_rho.clone();
                rho_minus[k] -= h_step;

                let grad_plus = match reml_state.compute_gradient(&rho_plus) {
                    Ok(g) => g,
                    Err(_) => continue,
                };
                let grad_minus = match reml_state.compute_gradient(&rho_minus) {
                    Ok(g) => g,
                    Err(_) => continue,
                };

                for j in 0..n_rho {
                    hessian_fd[[k, j]] = (grad_plus[j] - grad_minus[j]) / (2.0 * h_step);
                }
            }
            hessian_fd
        }
    };

    // Symmetrize the Hessian
    for i in 0..n_rho {
        for j in (i + 1)..n_rho {
            let avg = 0.5 * (hessian_rho[[i, j]] + hessian_rho[[j, i]]);
            hessian_rho[[i, j]] = avg;
            hessian_rho[[j, i]] = avg;
        }
    }

    // Step 3: Invert Hessian to get V_rho.
    // Add a small ridge before factorization to regularize weakly identified ρ directions.
    let ridge = 1e-8
        * hessian_rho
            .diag()
            .iter()
            .map(|&v| v.abs())
            .fold(0.0, f64::max)
            .max(1e-8);
    for i in 0..n_rho {
        hessian_rho[[i, i]] += ridge;
    }

    let v_rho = match hessian_rho.cholesky(Side::Lower) {
        Ok(chol) => {
            let mut eye = Array2::<f64>::eye(n_rho);
            for col in 0..n_rho {
                let col_vec = eye.column(col).to_owned();
                let solved = chol.solve_vec(&col_vec);
                eye.column_mut(col).assign(&solved);
            }
            eye
        }
        Err(_) => {
            log::warn!("Failed to invert LAML Hessian for smoothing correction; skipping.");
            return None;
        }
    };

    // Step 4: Compute V_corr = J * V_rho * J^T in transformed space.
    //
    // This is the first-order smoothing-parameter uncertainty inflation:
    //   Var(β̂) ≈ Var(β̂|ρ̂) + (dβ̂/dρ) Var(ρ̂) (dβ̂/dρ)ᵀ.
    //
    // Here:
    //   J = dβ̂/dρ,  J[:,k] = -H^{-1}(A_k β̂),
    //   V_ρ = (∇²_{ρρ}V)^{-1} evaluated at the final ρ.
    let j_v_rho = jacobian_trans.dot(&v_rho); // (n_coeffs_trans x n_rho)
    let v_corr_trans = j_v_rho.dot(&jacobian_trans.t()); // (n_coeffs_trans x n_coeffs_trans)

    // Step 5: Transform back to original coefficient basis:
    // V_corr_orig = Qs * V_corr_trans * Qs^T
    let qs = &final_fit.reparam_result.qs;
    let qs_v = qs.dot(&v_corr_trans);
    let v_corr_orig = qs_v.dot(&qs.t());

    // Validate the result
    if !v_corr_orig.iter().all(|v| v.is_finite()) {
        log::warn!("Non-finite values in smoothing correction matrix; skipping.");
        return None;
    }

    // Ensure positive semi-definiteness by clamping negative eigenvalues
    // (can happen due to numerical noise)
    match v_corr_orig.clone().eigh(Side::Lower) {
        Ok((eigenvalues, eigenvectors)) => {
            let min_eig = eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            if min_eig < -1e-10 {
                log::debug!(
                    "Smoothing correction has negative eigenvalue {:.3e}; clamping to zero.",
                    min_eig
                );
                // Reconstruct with clamped eigenvalues
                let mut result = Array2::<f64>::zeros((n_coeffs_orig, n_coeffs_orig));
                for i in 0..n_coeffs_orig {
                    let eig = eigenvalues[i].max(0.0);
                    let v = eigenvectors.column(i);
                    for j in 0..n_coeffs_orig {
                        for k in 0..n_coeffs_orig {
                            result[[j, k]] += eig * v[j] * v[k];
                        }
                    }
                }
                return Some(result);
            }
        }
        Err(_) => {
            log::warn!("Eigendecomposition failed for smoothing correction validation.");
        }
    }

    Some(v_corr_orig)
}

fn run_bfgs_for_candidate(
    label: &str,
    reml_state: &RemlState<'_>,
    config: &RemlConfig,
    initial_z: Array1<f64>,
) -> Result<(BfgsSolution, f64, bool), EstimationError> {
    log::debug!("[Candidate {label}] Running BFGS optimization from queued seed");
    let mut solver = Bfgs::new(initial_z, |z| reml_state.cost_and_grad(z))
        .with_tolerance(config.reml_convergence_tolerance)
        .with_max_iterations(config.reml_max_iterations as usize)
        .with_fp_tolerances(1e2, 1e2)
        .with_no_improve_stop(1e-8, 5)
        .with_rng_seed(0xC0FFEE_u64);

    let solution = match solver.run() {
        Ok(solution) => {
            log::debug!("[Candidate {label}] BFGS converged successfully according to tolerance.");
            solution
        }
        Err(wolfe_bfgs::BfgsError::LineSearchFailed { last_solution, .. }) => {
            log::debug!(
                "[Candidate {label}] Line search stopped early; using best-so-far parameters."
            );
            *last_solution
        }
        Err(wolfe_bfgs::BfgsError::MaxIterationsReached { last_solution }) => {
            log::warn!(
                "[Candidate {label}] BFGS hit the iteration cap; using best-so-far parameters."
            );
            log::debug!(
                "[Candidate {label}] Last recorded gradient norm: {:.2e}",
                last_solution.final_gradient_norm
            );
            *last_solution
        }
        Err(e) => {
            return Err(EstimationError::RemlOptimizationFailed(format!(
                "Candidate {label} failed with a critical BFGS error: {e:?}"
            )));
        }
    };

    if reml_state.consecutive_cost_error_count() >= MAX_CONSECUTIVE_INNER_ERRORS {
        let last_msg = reml_state
            .last_cost_error_string()
            .unwrap_or_else(|| "unknown error".to_string());
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "Candidate {label} aborted due to repeated inner-loop failures ({} consecutive). Last error: {}",
            reml_state.consecutive_cost_error_count(),
            last_msg
        )));
    }

    if !solution.final_value.is_finite() {
        return Err(EstimationError::RemlOptimizationFailed(format!(
            "Candidate {label} produced a non-finite final value: {}",
            solution.final_value
        )));
    }

    let grad_norm_rho = solution.final_gradient_norm;
    let is_stationary = grad_norm_rho <= config.reml_convergence_tolerance.max(1e-12);
    log::debug!(
        "[Candidate {label}] BFGS final gradient norm {:.3e} (tol {:.3e}); stationary={}",
        grad_norm_rho,
        config.reml_convergence_tolerance,
        is_stationary
    );

    Ok((solution, grad_norm_rho, is_stationary))
}

struct OuterSolveResult {
    final_rho: Array1<f64>,
    final_value: f64,
    iterations: usize,
    grad_norm_rho: f64,
    stationary: bool,
}

fn run_newton_for_candidate(
    label: &str,
    reml_state: &RemlState<'_>,
    config: &RemlConfig,
    initial_z: Array1<f64>,
) -> Result<OuterSolveResult, EstimationError> {
    log::debug!("[Candidate {label}] Running exact Newton optimization from queued seed");
    let mut rho = to_rho_from_z(&initial_z);
    let max_iter = config.reml_max_iterations as usize;
    let tol = config.reml_convergence_tolerance.max(1e-8);

    let mut best_rho = rho.clone();
    let mut best_cost = f64::INFINITY;
    let mut iter_done = 0usize;
    let mut last_grad_norm = f64::INFINITY;

    for iter in 0..max_iter {
        iter_done = iter + 1;
        let cost = reml_state.compute_cost(&rho)?;
        let mut grad = reml_state.compute_gradient(&rho)?;
        maybe_audit_gradient(reml_state, &rho, iter_done as u64, &mut grad);
        project_rho_gradient(&rho, &mut grad);
        last_grad_norm = grad_norm_in_z_space(&rho, &grad);
        if cost.is_finite() && (cost < best_cost || !best_cost.is_finite()) {
            best_cost = cost;
            best_rho = rho.clone();
        }
        if last_grad_norm <= tol {
            break;
        }

        let hess = reml_state.compute_laml_hessian_exact(&rho)?;
        let mut hreg = hess.clone();
        let mut ridge = 0.0_f64;
        let step = loop {
            if ridge > 0.0 {
                for i in 0..hreg.nrows() {
                    hreg[[i, i]] = hess[[i, i]] + ridge;
                }
            }
            let h_view = FaerArrayView::new(&hreg);
            if let Ok(ch) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
                let mut rhs = grad.mapv(|v| -v).insert_axis(Axis(1));
                let mut rhs_view = array2_to_mat_mut(&mut rhs);
                ch.solve_in_place(rhs_view.as_mut());
                break rhs.column(0).to_owned();
            }
            if let Ok(ld) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
                let mut rhs = grad.mapv(|v| -v).insert_axis(Axis(1));
                let mut rhs_view = array2_to_mat_mut(&mut rhs);
                ld.solve_in_place(rhs_view.as_mut());
                break rhs.column(0).to_owned();
            }
            ridge = if ridge == 0.0 { 1e-8 } else { ridge * 10.0 };
            if ridge > 1e8 {
                return Err(EstimationError::RemlOptimizationFailed(
                    "Newton Hessian factorization failed repeatedly".to_string(),
                ));
            }
            hreg.assign(&hess);
        };

        let armijo_c = 1e-4;
        let descent = grad.dot(&step);
        let mut alpha = 1.0_f64;
        let mut accepted = false;
        for _ in 0..30 {
            let mut trial = &rho + &(step.mapv(|v| alpha * v));
            for i in 0..trial.len() {
                trial[i] = trial[i].clamp(-RHO_BOUND, RHO_BOUND);
            }
            let trial_cost = reml_state.compute_cost(&trial)?;
            if trial_cost.is_finite() && trial_cost <= cost + armijo_c * alpha * descent {
                rho = trial;
                accepted = true;
                break;
            }
            alpha *= 0.5;
        }
        if !accepted {
            break;
        }
        if alpha * step.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-10 {
            break;
        }
    }

    let verified_grad_norm = last_grad_norm;
    let verified_stationary = verified_grad_norm <= tol;
    log::debug!(
        "[Candidate {label}] Newton final gradient norm {:.3e} (tol {:.3e}); stationary={}",
        verified_grad_norm,
        tol,
        verified_stationary
    );

    Ok(OuterSolveResult {
        final_rho: best_rho,
        final_value: best_cost,
        iterations: iter_done,
        grad_norm_rho: verified_grad_norm,
        stationary: verified_stationary,
    })
}

/// A comprehensive error type for the model estimation process.
#[derive(Error)]
pub enum EstimationError {
    #[error("Underlying basis function generation failed: {0}")]
    BasisError(#[from] crate::basis::BasisError),

    #[error("A linear system solve failed. The penalized Hessian may be singular. Error: {0}")]
    LinearSystemSolveFailed(FaerLinalgError),

    #[error("Eigendecomposition failed: {0}")]
    EigendecompositionFailed(FaerLinalgError),

    #[error("Parameter constraint violation: {0}")]
    ParameterConstraintViolation(String),

    #[error(
        "The P-IRLS inner loop did not converge within {max_iterations} iterations. Last gradient norm was {last_change:.6e}."
    )]
    PirlsDidNotConverge {
        max_iterations: usize,
        last_change: f64,
    },

    #[error(
        "Perfect or quasi-perfect separation detected during model fitting at iteration {iteration}. \
        The model cannot converge because a predictor perfectly separates the binary outcomes. \
        (Diagnostic: max|eta| = {max_abs_eta:.2e})."
    )]
    PerfectSeparationDetected { iteration: usize, max_abs_eta: f64 },

    #[error(
        "Hessian matrix is not positive definite (minimum eigenvalue: {min_eigenvalue:.4e}). This indicates a numerical instability."
    )]
    HessianNotPositiveDefinite { min_eigenvalue: f64 },

    #[error("REML/BFGS optimization failed to converge: {0}")]
    RemlOptimizationFailed(String),

    #[error("An internal error occurred during model layout or coefficient mapping: {0}")]
    LayoutError(String),

    #[error(
        "Model is over-parameterized: {num_coeffs} coefficients for {num_samples} samples.\n\n\
        Coefficient Breakdown:\n\
          - Intercept:                     {intercept_coeffs}\n\
          - Binary Main Effects:           {binary_main_coeffs}\n\
          - Primary Smooth Effects:        {primary_smooth_coeffs}\n\
          - Binary×Primary Interactions:   {binary_primary_interaction_coeffs}\n\
          - Auxiliary Main Effects:        {aux_main_coeffs}\n\
          - Auxiliary Interactions:        {aux_interaction_coeffs}"
    )]
    ModelOverparameterized {
        num_coeffs: usize,
        num_samples: usize,
        intercept_coeffs: usize,
        binary_main_coeffs: usize,
        primary_smooth_coeffs: usize,
        aux_main_coeffs: usize,
        binary_primary_interaction_coeffs: usize,
        aux_interaction_coeffs: usize,
    },

    #[error(
        "Model is ill-conditioned with condition number {condition_number:.2e}. This typically occurs when the model is over-parameterized (too many knots relative to data points). Consider reducing the number of knots or increasing regularization."
    )]
    ModelIsIllConditioned { condition_number: f64 },

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Calibrator training failed: {0}")]
    CalibratorTrainingFailed(String),

    #[error("Invalid specification: {0}")]
    InvalidSpecification(String),

    #[error("Prediction error")]
    PredictionError,
}

// Ensure Debug prints with actual line breaks by delegating to Display
impl core::fmt::Debug for EstimationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self)
    }
}

//
// This uses the joint model architecture where the base predictor and
// flexible link are fitted together in one optimization with REML.
//
// The model is: η = g(Xβ) where g is a learned flexible link function.
// Domain-specific training orchestration is handled by caller adapters.
// The gam engine exposes matrix/family-based APIs: fit_gam / optimize_external_design.

pub struct ExternalOptimResult {
    pub beta: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub scale: f64,
    pub edf_by_block: Vec<f64>,
    pub edf_total: f64,
    pub iterations: usize,
    pub final_grad_norm: f64,
    pub pirls_status: crate::pirls::PirlsStatus,
    pub smoothing_correction: Option<Array2<f64>>,
    pub penalized_hessian: Array2<f64>,
    pub working_weights: Array1<f64>,
    pub working_response: Array1<f64>,
    pub reparam_qs: Array2<f64>,
    pub artifacts: FitArtifacts,
    /// Conditional posterior covariance under fixed smoothing parameters:
    /// Var(β | λ) ≈ (X'WX + S)^(-1)
    pub beta_covariance: Option<Array2<f64>>,
    /// Marginal SEs from `beta_covariance`.
    pub beta_standard_errors: Option<Array1<f64>>,
    /// Optional smoothing-parameter-corrected covariance.
    /// Usually this is first-order:
    /// Var*(β) ≈ Var(β|λ) + J Var(ρ) J^T.
    /// In high-risk regimes the engine may use adaptive cubature for higher-order terms.
    pub beta_covariance_corrected: Option<Array2<f64>>,
    /// Marginal SEs from `beta_covariance_corrected`.
    pub beta_standard_errors_corrected: Option<Array1<f64>>,
}

#[derive(Clone)]
pub struct ExternalOptimOptions {
    pub family: crate::types::LikelihoodFamily,
    pub max_iter: usize,
    pub tol: f64,
    pub nullspace_dims: Vec<usize>,
    /// Optional explicit Firth override for binomial-logit external fitting.
    /// - `Some(true)`: force Firth on
    /// - `Some(false)`: force Firth off
    /// - `None`: use family default behavior
    pub firth_bias_reduction: Option<bool>,
}

fn resolve_external_family(
    family: crate::types::LikelihoodFamily,
    firth_override: Option<bool>,
) -> Result<(LinkFunction, bool), EstimationError> {
    match family {
        crate::types::LikelihoodFamily::GaussianIdentity => Ok((LinkFunction::Identity, false)),
        crate::types::LikelihoodFamily::BinomialLogit => {
            Ok((LinkFunction::Logit, firth_override.unwrap_or(true)))
        }
        crate::types::LikelihoodFamily::BinomialProbit => Ok((LinkFunction::Probit, false)),
        crate::types::LikelihoodFamily::BinomialCLogLog => Ok((LinkFunction::CLogLog, false)),
        crate::types::LikelihoodFamily::RoystonParmar => Err(EstimationError::InvalidInput(
            "optimize_external_design does not support RoystonParmar; use survival training APIs"
                .to_string(),
        )),
    }
}

fn validate_full_size_penalties(
    s_list: &[Array2<f64>],
    p: usize,
    context: &str,
) -> Result<(), EstimationError> {
    for (idx, s) in s_list.iter().enumerate() {
        let (rows, cols) = s.dim();
        if rows != p || cols != p {
            return Err(EstimationError::InvalidInput(format!(
                "{context}: penalty matrix {idx} must be {p}x{p}, got {rows}x{cols}"
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct SmoothingBfgsOptions {
    pub max_iter: usize,
    pub tol: f64,
    pub finite_diff_step: f64,
    pub seed_config: SeedConfig,
}

impl Default for SmoothingBfgsOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-5,
            finite_diff_step: 1e-3,
            seed_config: SeedConfig::default(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SmoothingBfgsResult {
    pub rho: Array1<f64>,
    pub final_value: f64,
    pub iterations: usize,
    pub final_grad_norm: f64,
    pub stationary: bool,
}

fn finite_diff_gradient_external<F>(
    rho: &Array1<f64>,
    step: f64,
    objective: &mut F,
) -> Result<Array1<f64>, EstimationError>
where
    F: FnMut(&Array1<f64>) -> Result<f64, EstimationError>,
{
    // Central-difference gradient in rho-space:
    //   g_k ≈ [V(rho + h e_k) - V(rho - h e_k)] / (2h).
    //
    // This is intentionally objective-level (black-box) differentiation, so the returned
    // gradient is exactly consistent with whatever nonlinearities the objective currently
    // includes (Laplace terms, truncation conventions, ridge policies, survival constraints,
    // etc.). That consistency is often more robust than brittle closed-form expressions.
    let mut grad = Array1::<f64>::zeros(rho.len());
    let mut rp = rho.clone();
    let mut rm = rho.clone();
    for i in 0..rho.len() {
        rp[i] += step;
        let fp = objective(&rp)?;
        rm[i] -= step;
        let fm = objective(&rm)?;
        grad[i] = (fp - fm) / (2.0 * step);
        rp[i] = rho[i];
        rm[i] = rho[i];
    }
    Ok(grad)
}

fn approx_same_rho_point(a: &Array1<f64>, b: &Array1<f64>) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if (a[i] - b[i]).abs() > 1e-12 {
            return false;
        }
    }
    true
}

fn should_replace_smoothing_candidate(
    best: &Option<SmoothingBfgsResult>,
    candidate: &SmoothingBfgsResult,
) -> bool {
    match best {
        None => true,
        Some(current) => {
            if candidate.stationary != current.stationary {
                candidate.stationary
            } else if candidate.stationary {
                candidate.final_value < current.final_value
            } else {
                candidate.final_grad_norm < current.final_grad_norm
            }
        }
    }
}

fn run_multistart_bfgs<C, Eval>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    context: &mut C,
    eval_cost_grad_rho: &mut Eval,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
where
    Eval: FnMut(&mut C, &Array1<f64>) -> Result<(f64, Array1<f64>), EstimationError>,
{
    let seeds = generate_rho_candidates(num_penalties, heuristic_lambdas, &options.seed_config);
    if seeds.is_empty() {
        return Err(EstimationError::RemlOptimizationFailed(
            "no smoothing seeds produced".to_string(),
        ));
    }

    let mut best: Option<SmoothingBfgsResult> = None;
    for (idx, rho_seed) in seeds.iter().enumerate() {
        let initial_z = to_z_from_rho(rho_seed);
        let mut last_eval: Option<(Array1<f64>, Array1<f64>)> = None;
        let mut optimizer = Bfgs::new(initial_z.clone(), |z| {
            let rho = to_rho_from_z(z);
            let (cost, grad_rho) = match eval_cost_grad_rho(context, &rho) {
                Ok(v) => v,
                Err(_) => (f64::INFINITY, Array1::<f64>::zeros(rho.len())),
            };
            last_eval = Some((rho.clone(), grad_rho.clone()));
            let jac = jacobian_drho_dz_from_rho(&rho);
            let mut grad_z = &grad_rho * &jac;
            for g in grad_z.iter_mut() {
                if !g.is_finite() {
                    *g = 0.0;
                }
            }
            (cost, grad_z)
        })
        .with_tolerance(options.tol)
        .with_max_iterations(options.max_iter)
        .with_fp_tolerances(1e2, 1e2)
        .with_no_improve_stop(1e-8, 5)
        .with_rng_seed(0x5EED_u64.wrapping_add(idx as u64));

        let solution = match optimizer.run() {
            Ok(sol) => sol,
            Err(wolfe_bfgs::BfgsError::LineSearchFailed { last_solution, .. }) => *last_solution,
            Err(wolfe_bfgs::BfgsError::MaxIterationsReached { last_solution }) => *last_solution,
            Err(_) => continue,
        };

        let rho = to_rho_from_z(&solution.final_point);
        let mut grad_rho = match &last_eval {
            Some((rho_cached, grad_cached)) if approx_same_rho_point(&rho, rho_cached) => {
                grad_cached.clone()
            }
            _ => match eval_cost_grad_rho(context, &rho) {
                Ok((_, grad)) => grad,
                Err(_) => Array1::<f64>::zeros(rho.len()),
            },
        };
        project_rho_gradient(&rho, &mut grad_rho);
        let grad_norm = grad_rho.dot(&grad_rho).sqrt();
        let candidate = SmoothingBfgsResult {
            rho,
            final_value: solution.final_value,
            iterations: solution.iterations,
            final_grad_norm: grad_norm,
            stationary: grad_norm <= options.tol.max(1e-6),
        };

        if should_replace_smoothing_candidate(&best, &candidate) {
            best = Some(candidate);
        }
    }

    best.ok_or_else(|| {
        EstimationError::RemlOptimizationFailed(
            "all smoothing BFGS starts failed before producing a candidate".to_string(),
        )
    })
}

/// Generic multi-start BFGS smoothing optimizer over log-smoothing parameters (`rho`).
///
/// This is intended for likelihoods whose outer objective is exposed as a scalar
/// function of `rho` (for example survival workflows built on working-model PIRLS).
///
/// Mathematically, this optimizer searches:
///   rho* = argmin_rho V(rho),
/// where `V` is supplied by the caller.
///
/// The gradient seen by BFGS is always computed by finite differences on `V`:
///   grad_k = dV/drho_k ≈ [V(rho+h e_k)-V(rho-h e_k)]/(2h).
/// This makes the direction field fully consistent with the exact scalar objective,
/// which is particularly useful for complicated non-Gaussian/survival objectives where
/// exact analytic outer derivatives are either expensive or error-prone.
pub fn optimize_log_smoothing_with_multistart<F>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    mut objective: F,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
where
    F: FnMut(&Array1<f64>) -> Result<f64, EstimationError>,
{
    if num_penalties == 0 {
        let rho = Array1::<f64>::zeros(0);
        return Ok(SmoothingBfgsResult {
            rho,
            final_value: objective(&Array1::<f64>::zeros(0))?,
            iterations: 0,
            final_grad_norm: 0.0,
            stationary: true,
        });
    }

    let mut eval_cost_grad_rho = |objective: &mut F, rho: &Array1<f64>| {
        let cost = objective(rho)?;
        let grad_rho = finite_diff_gradient_external(rho, options.finite_diff_step, objective)?;
        Ok((cost, grad_rho))
    };
    run_multistart_bfgs(
        num_penalties,
        heuristic_lambdas,
        &mut objective,
        &mut eval_cost_grad_rho,
        options,
    )
}

/// Generic multi-start BFGS smoothing optimizer over log-smoothing parameters (`rho`)
/// when the caller can provide an exact objective gradient in rho-space.
///
/// The callback must return:
/// - `value = V(rho)`
/// - `grad_rho = dV/drho` (same dimension/order as `rho`)
///
/// Internally we optimize in unconstrained `z` coordinates:
/// - `rho = to_rho_from_z(z)` (bounded/smoothed map used by this module)
/// - chain rule for BFGS objective gradient:
///   `grad_z = diag(drho_dz(rho)) * grad_rho`.
///
/// Why this exists:
/// - finite-difference outer gradients require repeated inner solves per coordinate,
/// - exact outer gradients can be injected directly here,
/// - multi-start seed handling and stationarity ranking remain identical to the
///   FD-based optimizer, so behavior is comparable while much faster when exact
///   gradients are available.
pub fn optimize_log_smoothing_with_multistart_with_gradient<F>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    mut objective_with_gradient: F,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
where
    F: FnMut(&Array1<f64>) -> Result<(f64, Array1<f64>), EstimationError>,
{
    if num_penalties == 0 {
        let rho = Array1::<f64>::zeros(0);
        let (value, grad) = objective_with_gradient(&rho)?;
        let grad_norm = grad.dot(&grad).sqrt();
        return Ok(SmoothingBfgsResult {
            rho,
            final_value: value,
            iterations: 0,
            final_grad_norm: grad_norm,
            stationary: grad_norm <= options.tol.max(1e-6),
        });
    }

    let mut eval_cost_grad_rho = |objective: &mut F, rho: &Array1<f64>| objective(rho);
    run_multistart_bfgs(
        num_penalties,
        heuristic_lambdas,
        &mut objective_with_gradient,
        &mut eval_cost_grad_rho,
        options,
    )
}

/// Optimize smoothing parameters for an external design using the same REML/LAML machinery.
/// Contract: likelihood dispatch is determined by `opts.family`.
pub fn optimize_external_design<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: Vec<Array2<f64>>,
    opts: &ExternalOptimOptions,
) -> Result<ExternalOptimResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if !(y.len() == w.len() && y.len() == x.nrows() && y.len() == offset.len()) {
        return Err(EstimationError::InvalidInput(format!(
            "Row mismatch: y={}, w={}, X.rows={}, offset={}",
            y.len(),
            w.len(),
            x.nrows(),
            offset.len()
        )));
    }

    use crate::construction::compute_penalty_square_roots;

    let p = x.ncols();
    validate_full_size_penalties(&s_list, p, "optimize_external_design")?;
    let k = s_list.len();
    let (link, firth_active) = resolve_external_family(opts.family, opts.firth_bias_reduction)?;
    let cfg = RemlConfig::external(link, opts.tol, opts.max_iter, firth_active);

    let rs_list = compute_penalty_square_roots(&s_list)?;

    // Clone inputs to own their storage and unify lifetimes inside this function
    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let x_o = x.clone();
    let offset_o = offset.to_owned();
    let reml_state = internal::RemlState::new_with_offset(
        y_o.view(),
        x_o.clone(),
        w_o.view(),
        offset_o.view(),
        s_list,
        p,
        &cfg,
        Some(opts.nullspace_dims.clone()),
    )?;
    let seed_strategy = if k >= 10 {
        SeedStrategy::Light
    } else {
        SeedStrategy::Exhaustive
    };
    let seed_config = SeedConfig {
        strategy: seed_strategy,
        bounds: (-12.0, 12.0),
    };
    let rho_seeds = generate_rho_candidates(k, None, &seed_config);
    let mut candidate_seeds: Vec<(String, Array1<f64>)> = rho_seeds
        .into_iter()
        .enumerate()
        .map(|(idx, rho)| (format!("seed_{idx}"), to_z_from_rho(&rho)))
        .collect();
    if candidate_seeds.is_empty() {
        let primary_rho = Array1::<f64>::zeros(k);
        candidate_seeds.push(("fallback_zero".to_string(), to_z_from_rho(&primary_rho)));
    }

    let mut best_stationary: Option<OuterSolveResult> = None;
    let mut best_nonstationary: Option<OuterSolveResult> = None;
    let mut best_grad_norm = f64::INFINITY;
    let mut candidate_failures: Vec<String> = Vec::new();

    // Two-stage seed handling:
    // 1) evaluate each seed once with compute_cost (cheap relative to a full outer solve),
    // 2) fully optimize only the most promising seeds.
    let screening_budget = if k <= 2 {
        4
    } else if k <= 6 {
        6
    } else {
        8
    };
    let mut screened: Vec<(String, Array1<f64>, f64)> = Vec::with_capacity(candidate_seeds.len());
    for (label, initial_z) in candidate_seeds.drain(..) {
        let rho0 = to_rho_from_z(&initial_z);
        match reml_state.compute_cost(&rho0) {
            Ok(cost0) if cost0.is_finite() => screened.push((label, initial_z, cost0)),
            Ok(cost0) => {
                candidate_failures.push(format!(
                    "[Seed screen {}] non-finite initial cost {}",
                    label, cost0
                ));
            }
            Err(err) => {
                candidate_failures.push(format!("[Seed screen {}] {}", label, err));
            }
        }
    }
    screened.sort_by(|a, b| a.2.total_cmp(&b.2));
    let candidate_seeds: Vec<(String, Array1<f64>)> = screened
        .into_iter()
        .take(screening_budget.max(1))
        .map(|(label, z, _)| (label, z))
        .collect();
    if candidate_seeds.is_empty() {
        candidate_failures.push(
            "[Seed screen] all candidate seeds failed initial screening; reverting to zero seed"
                .to_string(),
        );
    }
    let candidate_seeds = if candidate_seeds.is_empty() {
        vec![(
            "fallback_zero".to_string(),
            to_z_from_rho(&Array1::<f64>::zeros(k)),
        )]
    } else {
        candidate_seeds
    };
    let near_stationary_tol = (cfg.reml_convergence_tolerance.max(1e-12)) * 2.0;
    let use_newton = true;
    for (label, initial_z) in candidate_seeds {
        let solution_result: Result<OuterSolveResult, EstimationError> = if use_newton {
            match run_newton_for_candidate(&label, &reml_state, &cfg, initial_z.clone()) {
                Ok(sol) => {
                    if sol.stationary {
                        Ok(sol)
                    } else {
                        log::debug!(
                            "[Candidate {label}] Newton ended non-stationary (grad_norm={:.3e}); retrying with BFGS.",
                            sol.grad_norm_rho
                        );
                        match run_bfgs_for_candidate(&label, &reml_state, &cfg, initial_z) {
                            Ok((bfgs_solution, grad_norm_rho, stationary)) => {
                                let bfgs_outer = OuterSolveResult {
                                    final_rho: to_rho_from_z(&bfgs_solution.final_point),
                                    final_value: bfgs_solution.final_value,
                                    iterations: bfgs_solution.iterations,
                                    grad_norm_rho,
                                    stationary,
                                };
                                if bfgs_outer.stationary
                                    || bfgs_outer.final_value <= sol.final_value
                                {
                                    Ok(bfgs_outer)
                                } else {
                                    Ok(sol)
                                }
                            }
                            Err(err) => {
                                log::warn!(
                                    "[Candidate {label}] BFGS fallback failed ({err}); keeping non-stationary Newton solution."
                                );
                                Ok(sol)
                            }
                        }
                    }
                }
                Err(err) => {
                    log::warn!("[Candidate {label}] Newton failed ({err}); falling back to BFGS.");
                    match run_bfgs_for_candidate(&label, &reml_state, &cfg, initial_z) {
                        Ok((bfgs_solution, grad_norm_rho, stationary)) => Ok(OuterSolveResult {
                            final_rho: to_rho_from_z(&bfgs_solution.final_point),
                            final_value: bfgs_solution.final_value,
                            iterations: bfgs_solution.iterations,
                            grad_norm_rho,
                            stationary,
                        }),
                        Err(bfgs_err) => Err(EstimationError::RemlOptimizationFailed(format!(
                            "Candidate {label}: Newton failed ({err}); BFGS fallback failed ({bfgs_err})"
                        ))),
                    }
                }
            }
        } else {
            run_bfgs_for_candidate(&label, &reml_state, &cfg, initial_z).map(
                |(bfgs_solution, grad_norm_rho, stationary)| OuterSolveResult {
                    final_rho: to_rho_from_z(&bfgs_solution.final_point),
                    final_value: bfgs_solution.final_value,
                    iterations: bfgs_solution.iterations,
                    grad_norm_rho,
                    stationary,
                },
            )
        };
        let solution = match solution_result {
            Ok(sol) => sol,
            Err(err) => {
                candidate_failures.push(format!("[Candidate {label}] {err}"));
                continue;
            }
        };
        let grad_norm = solution.grad_norm_rho;
        if solution.stationary {
            let better = match &best_stationary {
                None => true,
                Some(current) => solution.final_value < current.final_value,
            };
            if better {
                best_stationary = Some(solution);
            }
        } else {
            let better = match &best_nonstationary {
                None => true,
                Some(current) => solution.final_value < current.final_value,
            };
            if better {
                best_nonstationary = Some(solution);
            }
        }

        best_grad_norm = best_grad_norm.min(grad_norm);
        if best_stationary.is_some() && best_grad_norm <= near_stationary_tol {
            log::debug!(
                "[external] early stop on near-stationary candidate (grad_norm={:.3e}, tol={:.3e})",
                best_grad_norm,
                near_stationary_tol
            );
            break;
        }
    }
    let mut found_stationary = best_stationary.is_some();
    let chosen_solution = if let Some(sol) = best_stationary.or(best_nonstationary) {
        sol
    } else {
        log::warn!(
            "[external] all candidate seeds failed; using emergency fixed-rho fallback (rho=0)."
        );
        if !candidate_failures.is_empty() {
            log::warn!(
                "[external] candidate failures summary ({}):\n{}",
                candidate_failures.len(),
                candidate_failures.join("\n")
            );
        }
        found_stationary = false;
        let fallback_rho = Array1::<f64>::zeros(k);
        let fallback_value = reml_state
            .compute_cost(&fallback_rho)
            .unwrap_or(f64::INFINITY);
        OuterSolveResult {
            final_rho: fallback_rho,
            final_value: fallback_value,
            iterations: 1,
            grad_norm_rho: f64::INFINITY,
            stationary: false,
        }
    };
    if !found_stationary {
        log::debug!(
            "[external] no stationary candidate found; using best non-stationary solution with grad_norm={:.3e}",
            best_grad_norm
        );
    }
    let final_rho = chosen_solution.final_rho.clone();
    // Ensure we don't report 0 iterations to the caller; at least 1 is more meaningful.
    let iters = std::cmp::max(1, chosen_solution.iterations);
    let (pirls_res, _) = pirls::fit_model_for_fixed_rho_matrix(
        LogSmoothingParamsView::new(final_rho.view()),
        &x_o,
        offset_o.view(),
        y_o.view(),
        w_o.view(),
        &rs_list,
        Some(reml_state.balanced_penalty_root()),
        None,
        p,
        &cfg.as_pirls_config(),
        None,
        None, // No SE for base external optimization
    )?;

    // Map beta back to original basis
    let beta_orig = pirls_res
        .reparam_result
        .qs
        .dot(pirls_res.beta_transformed.as_ref());

    // Weighted residual sum of squares for Gaussian models
    let n = y_o.len() as f64;
    let weighted_rss = if matches!(link, LinkFunction::Identity) {
        let fitted = {
            let mut eta = offset_o.clone();
            eta += &x_o.matrix_vector_multiply(&beta_orig);
            eta
        };
        let resid = y_o.to_owned() - &fitted;
        w_o.iter()
            .zip(resid.iter())
            .map(|(&wi, &ri)| wi * ri * ri)
            .sum()
    } else {
        0.0
    };

    // EDF by block using stabilized H and penalty roots in transformed basis
    let lambdas = final_rho.mapv(f64::exp);
    let h = &pirls_res.stabilized_hessian_transformed;
    let h_view = FaerArrayView::new(h);
    enum Fact {
        Llt(FaerLlt<f64>),
        Ldlt(FaerLdlt<f64>),
        Lblt(FaerLblt<f64>),
    }
    impl Fact {
        fn solve(&self, rhs: faer::MatRef<'_, f64>) -> FaerMat<f64> {
            match self {
                Fact::Llt(f) => f.solve(rhs),
                Fact::Ldlt(f) => f.solve(rhs),
                Fact::Lblt(f) => f.solve(rhs),
            }
        }
    }
    let mut planner = RidgePlanner::new(h);
    let cond_display = planner
        .cond_estimate()
        .map(|c| format!("{c:.2e}"))
        .unwrap_or_else(|| "unavailable".to_string());
    let fact = loop {
        let ridge = planner.ridge();
        if ridge > 0.0 {
            let regularized = add_ridge(h, ridge);
            let view = FaerArrayView::new(&regularized);
            if let Ok(ch) = FaerLlt::new(view.as_ref(), Side::Lower) {
                log::warn!(
                    "LLᵀ succeeded after adding ridge {:.3e} (cond ≈ {})",
                    ridge,
                    cond_display
                );
                break Fact::Llt(ch);
            }
            if let Ok(ld) = FaerLdlt::new(view.as_ref(), Side::Lower) {
                log::warn!(
                    "LLᵀ failed; LDLᵀ succeeded with ridge {:.3e} (cond ≈ {})",
                    ridge,
                    cond_display
                );
                break Fact::Ldlt(ld);
            }
            if planner.attempts() >= MAX_FACTORIZATION_ATTEMPTS {
                log::warn!(
                    "LLᵀ/LDLᵀ failed even after ridge {:.3e}; falling back to LBLᵀ (cond ≈ {})",
                    ridge,
                    cond_display
                );
                let f = FaerLblt::new(view.as_ref(), Side::Lower);
                break Fact::Lblt(f);
            }
        } else {
            if let Ok(ch) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
                break Fact::Llt(ch);
            }
            if let Ok(ld) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
                log::warn!(
                    "LLᵀ failed for Hessian (cond ≈ {}); using LDLᵀ without ridge",
                    cond_display
                );
                break Fact::Ldlt(ld);
            }
        }
        planner.bump_with_matrix(h);
    };
    let mut traces = vec![0.0f64; k];
    for (kk, rs) in pirls_res.reparam_result.rs_transformed.iter().enumerate() {
        let ekt_arr = rs.t().to_owned();
        let ekt_view = FaerArrayView::new(&ekt_arr);
        let x_sol = fact.solve(ekt_view.as_ref());
        let frob = faer_frob_inner(x_sol.as_ref(), ekt_view.as_ref());
        traces[kk] = lambdas[kk] * frob;
    }
    let p_dim = pirls_res.beta_transformed.len();
    let penalty_rank = pirls_res.reparam_result.e_transformed.nrows();
    let mp = (p_dim as f64 - penalty_rank as f64).max(0.0);
    let edf_total = (p_dim as f64 - kahan_sum(traces.iter().copied())).clamp(mp, p_dim as f64);
    // Per-block EDF: use block range dimension (rank of R_k) minus λ tr(H^{-1} S_k)
    // This better reflects penalized coefficients in the transformed basis
    let mut edf_by_block: Vec<f64> = Vec::with_capacity(k);
    for (kk, rs_k) in pirls_res.reparam_result.rs_transformed.iter().enumerate() {
        let p_k = rs_k.nrows() as f64;
        let edf_k = (p_k - traces[kk]).clamp(0.0, p_k);
        edf_by_block.push(edf_k);
    }

    // Persist residual-based scale for Gaussian identity models
    let scale = match link {
        LinkFunction::Identity => {
            let denom = (n - edf_total).max(1.0);
            weighted_rss / denom
        }
        LinkFunction::Logit | LinkFunction::Probit | LinkFunction::CLogLog => 1.0,
    };

    // Compute gradient norm at final rho for reporting
    let mut final_grad = reml_state
        .compute_gradient(&final_rho)
        .unwrap_or_else(|_| Array1::from_elem(final_rho.len(), f64::NAN));
    project_rho_gradient(&final_rho, &mut final_grad);
    let final_grad_norm_z = grad_norm_in_z_space(&final_rho, &final_grad);
    let final_grad_norm = if final_grad_norm_z.is_finite() {
        final_grad_norm_z
    } else {
        best_grad_norm
    };

    let penalized_hessian = map_hessian_to_original_basis(&pirls_res)?;
    let beta_covariance =
        matrix_inverse_with_regularization(&penalized_hessian, "posterior covariance");
    let smoothing_correction = reml_state.compute_smoothing_correction_auto(
        &final_rho,
        &pirls_res,
        beta_covariance.as_ref(),
        final_grad_norm,
    );
    let beta_standard_errors = beta_covariance.as_ref().map(se_from_covariance);
    let beta_covariance_corrected = match (&beta_covariance, &smoothing_correction) {
        (Some(base_cov), Some(corr)) if base_cov.dim() == corr.dim() => {
            // First-order total covariance assembly:
            //   Var(beta) ~= Var(beta | rho_hat) + J Var(rho_hat) J^T
            //             ~= base_cov + corr.
            let mut corrected = base_cov.clone();
            corrected += corr;
            // Keep covariance symmetric after numerical addition.
            for i in 0..corrected.nrows() {
                for j in (i + 1)..corrected.ncols() {
                    let avg = 0.5 * (corrected[[i, j]] + corrected[[j, i]]);
                    corrected[[i, j]] = avg;
                    corrected[[j, i]] = avg;
                }
            }
            Some(corrected)
        }
        (Some(_), Some(corr)) => {
            log::warn!(
                "Skipping corrected covariance: dimension mismatch (base {:?}, corr {:?})",
                beta_covariance.as_ref().map(Array2::dim),
                Some(corr.dim())
            );
            None
        }
        _ => None,
    };
    let beta_standard_errors_corrected = beta_covariance_corrected.as_ref().map(se_from_covariance);
    let working_weights = pirls_res.solve_weights.clone();
    let working_response = pirls_res.solve_working_response.clone();
    let reparam_qs = pirls_res.reparam_result.qs.clone();

    let pirls_status = pirls_res.status.clone();

    Ok(ExternalOptimResult {
        beta: beta_orig,
        lambdas: lambdas.to_owned(),
        scale,
        edf_by_block,
        edf_total,
        iterations: iters,
        final_grad_norm,
        pirls_status,
        smoothing_correction,
        penalized_hessian,
        working_weights,
        working_response,
        reparam_qs,
        artifacts: FitArtifacts { pirls: pirls_res },
        beta_covariance,
        beta_standard_errors,
        beta_covariance_corrected,
        beta_standard_errors_corrected,
    })
}

#[derive(Clone)]
pub struct FitOptions {
    pub max_iter: usize,
    pub tol: f64,
    pub nullspace_dims: Vec<usize>,
}

/// Post-fit artifacts needed by downstream diagnostics/inference without
/// re-running PIRLS.
pub struct FitArtifacts {
    pub pirls: crate::pirls::PirlsResult,
}

pub struct FitResult {
    pub beta: Array1<f64>,
    pub lambdas: Array1<f64>,
    pub scale: f64,
    pub edf_by_block: Vec<f64>,
    pub edf_total: f64,
    pub iterations: usize,
    pub final_grad_norm: f64,
    pub pirls_status: crate::pirls::PirlsStatus,
    pub smoothing_correction: Option<Array2<f64>>,
    pub penalized_hessian: Array2<f64>,
    pub working_weights: Array1<f64>,
    pub working_response: Array1<f64>,
    pub reparam_qs: Array2<f64>,
    pub artifacts: FitArtifacts,
    /// Conditional posterior covariance under fixed smoothing parameters:
    /// Var(β | λ) ≈ (X'WX + S)^(-1)
    pub beta_covariance: Option<Array2<f64>>,
    /// Marginal SEs from `beta_covariance`.
    pub beta_standard_errors: Option<Array1<f64>>,
    /// Optional smoothing-parameter-corrected covariance.
    /// Usually this is first-order:
    /// Var*(β) ≈ Var(β|λ) + J Var(ρ) J^T.
    /// In high-risk regimes the engine may use adaptive cubature for higher-order terms.
    pub beta_covariance_corrected: Option<Array2<f64>>,
    /// Marginal SEs from `beta_covariance_corrected`.
    pub beta_standard_errors_corrected: Option<Array1<f64>>,
}

pub struct PredictResult {
    pub eta: Array1<f64>,
    pub mean: Array1<f64>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InferenceCovarianceMode {
    /// Use conditional posterior covariance only:
    ///   Var(beta | lambda_hat) ~= H_{rho_hat}^{-1}.
    Conditional,
    /// Prefer first-order smoothing-corrected covariance when available:
    ///   Var(beta) ~= H_{rho_hat}^{-1} + J Var(rho_hat) J^T.
    /// Falls back to conditional if correction is unavailable.
    ConditionalPlusSmoothingPreferred,
    /// Require the first-order smoothing-corrected covariance; error if unavailable.
    ConditionalPlusSmoothingRequired,
}

pub struct PredictUncertaintyOptions {
    /// Central interval level in (0, 1), e.g. 0.95.
    pub confidence_level: f64,
    /// Covariance mode used for eta/mean intervals.
    pub covariance_mode: InferenceCovarianceMode,
    /// Mean-scale interval construction method.
    pub mean_interval_method: MeanIntervalMethod,
    /// For Gaussian identity, also return observation intervals using
    /// Var(y_new | x) = Var(eta_hat) + scale.
    pub include_observation_interval: bool,
}

impl Default for PredictUncertaintyOptions {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            covariance_mode: InferenceCovarianceMode::ConditionalPlusSmoothingPreferred,
            mean_interval_method: MeanIntervalMethod::TransformEta,
            include_observation_interval: true,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MeanIntervalMethod {
    /// Interval on mean scale from delta-method SEs.
    Delta,
    /// Transform eta interval endpoints through inverse link.
    /// This is usually better behaved for nonlinear links.
    TransformEta,
}

pub struct PredictUncertaintyResult {
    pub eta: Array1<f64>,
    pub mean: Array1<f64>,
    pub eta_standard_error: Array1<f64>,
    pub mean_standard_error: Array1<f64>,
    pub eta_lower: Array1<f64>,
    pub eta_upper: Array1<f64>,
    pub mean_lower: Array1<f64>,
    pub mean_upper: Array1<f64>,
    /// Optional Gaussian observation interval bounds.
    pub observation_lower: Option<Array1<f64>>,
    pub observation_upper: Option<Array1<f64>>,
    /// Covariance mode requested by caller.
    pub covariance_mode_requested: InferenceCovarianceMode,
    /// True if smoothing-corrected covariance was used.
    pub covariance_corrected_used: bool,
}

pub struct CoefficientUncertaintyResult {
    pub estimate: Array1<f64>,
    pub standard_error: Array1<f64>,
    pub lower: Array1<f64>,
    pub upper: Array1<f64>,
    pub corrected: bool,
    pub covariance_mode_requested: InferenceCovarianceMode,
}

/// Canonical engine entrypoint for external designs.
/// Likelihood dispatch is determined exclusively by `family`.
pub fn fit_gam<X>(
    x: X,
    y: ArrayView1<'_, f64>,
    weights: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    s_list: &[Array2<f64>],
    family: crate::types::LikelihoodFamily,
    opts: &FitOptions,
) -> Result<FitResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if matches!(family, crate::types::LikelihoodFamily::RoystonParmar) {
        return Err(EstimationError::InvalidInput(
            "fit_gam external design path does not support RoystonParmar; use survival training APIs".to_string(),
        ));
    }
    validate_full_size_penalties(s_list, x.ncols(), "fit_gam")?;
    let mut ext_opts = ExternalOptimOptions {
        family,
        max_iter: opts.max_iter,
        tol: opts.tol,
        nullspace_dims: opts.nullspace_dims.clone(),
        firth_bias_reduction: None,
    };

    let result = if matches!(family, crate::types::LikelihoodFamily::BinomialLogit) {
        let weighted_events = y
            .iter()
            .zip(weights.iter())
            .map(|(&yy, &ww)| yy.max(0.0).min(1.0) * ww.max(0.0))
            .sum::<f64>();
        let weighted_total = weights.iter().map(|w| w.max(0.0)).sum::<f64>();
        let weighted_nonevents = (weighted_total - weighted_events).max(0.0);
        let low_event_support = weighted_events.min(weighted_nonevents) < 20.0;

        // Start without Firth unless support is clearly too small.
        ext_opts.firth_bias_reduction = Some(low_event_support);
        let first_try =
            optimize_external_design(y, weights, &x, offset, s_list.to_vec(), &ext_opts);

        match first_try {
            Ok(res) => {
                let unstable_status = matches!(
                    res.pirls_status,
                    crate::pirls::PirlsStatus::MaxIterationsReached
                        | crate::pirls::PirlsStatus::Unstable
                );
                let extreme_eta = res.artifacts.pirls.max_abs_eta > 15.0;
                if !low_event_support && (unstable_status || extreme_eta) {
                    ext_opts.firth_bias_reduction = Some(true);
                    optimize_external_design(y, weights, &x, offset, s_list.to_vec(), &ext_opts)?
                } else {
                    res
                }
            }
            Err(err) => {
                if low_event_support {
                    return Err(err);
                }
                ext_opts.firth_bias_reduction = Some(true);
                optimize_external_design(y, weights, &x, offset, s_list.to_vec(), &ext_opts)?
            }
        }
    } else {
        optimize_external_design(y, weights, &x, offset, s_list.to_vec(), &ext_opts)?
    };
    Ok(FitResult {
        beta: result.beta,
        lambdas: result.lambdas,
        scale: result.scale,
        edf_by_block: result.edf_by_block,
        edf_total: result.edf_total,
        iterations: result.iterations,
        final_grad_norm: result.final_grad_norm,
        pirls_status: result.pirls_status,
        smoothing_correction: result.smoothing_correction,
        penalized_hessian: result.penalized_hessian,
        working_weights: result.working_weights,
        working_response: result.working_response,
        reparam_qs: result.reparam_qs,
        artifacts: result.artifacts,
        beta_covariance: result.beta_covariance,
        beta_standard_errors: result.beta_standard_errors,
        beta_covariance_corrected: result.beta_covariance_corrected,
        beta_standard_errors_corrected: result.beta_standard_errors_corrected,
    })
}

/// Generic engine prediction for external designs.
/// This API is domain-agnostic: callers provide only design matrix, coefficients, offset, and family.
pub fn predict_gam<X>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: crate::types::LikelihoodFamily,
) -> Result<PredictResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if x.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam dimension mismatch: X has {} columns but beta has length {}",
            x.ncols(),
            beta.len()
        )));
    }
    if x.nrows() != offset.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam dimension mismatch: X has {} rows but offset has length {}",
            x.nrows(),
            offset.len()
        )));
    }
    if matches!(family, crate::types::LikelihoodFamily::RoystonParmar) {
        return Err(EstimationError::InvalidInput(
            "predict_gam does not support RoystonParmar; use survival prediction APIs".to_string(),
        ));
    }

    let mut eta = x.matrix_vector_multiply(&beta.to_owned());
    eta += &offset;

    let mean = apply_family_inverse_link(&eta, family)?;

    Ok(PredictResult { eta, mean })
}

/// Prediction with coefficient uncertainty propagation.
///
/// The linear predictor variance uses:
/// Var(η_i) = x_i^T Var(β) x_i
///
/// Mean-scale SEs are delta-method approximations:
/// Var(μ_i) ≈ (dμ/dη)^2 Var(η_i)
///
/// Math note (logit family, Gaussian η posterior):
///
/// If η_i | D ≈ N(m_i, v_i), then the exact posterior predictive mean on the
/// probability scale is the logistic-normal integral
///
///   E[sigmoid(η_i)] = ∫ sigmoid(x) N(x; m_i, v_i) dx.
///
/// This does not reduce to an elementary closed form. Two exact representations
/// often used in the literature are:
///
/// 1) Theta/Appell-Lerch style representations (via Poisson summation / Mordell integrals).
/// 2) Absolutely convergent complex-error-function (Faddeeva) series obtained from
///    partial-fraction expansions of tanh/logistic.
///
/// A practical exact series form is:
///
///   E[sigmoid(η)] = 1/2
///                   - (sqrt(2π)/σ) * Σ_{n>=1} Im[ w((i a_n - μ)/(sqrt(2)σ)) ],
///   where a_n = (2n-1)π, σ = sqrt(v), and w is the Faddeeva function
///   w(z) = exp(-z^2) erfc(-i z).
///
/// The formulas above define the exact logistic-normal target moments under
/// Gaussian η uncertainty.
///
/// CLogLog note (exact target):
/// If p = 1 - exp(-exp(η)) and η ~ N(μ,σ²), then
///   E[p] = 1 - I(1),  E[p²] = 1 - 2I(1) + I(2),  Var(p) = I(2) - I(1)²
/// where I(λ) = E[exp(-λ exp(η))] is the lognormal Laplace transform.
/// This identity is exact, and highlights that the moments are determined by
/// the lognormal Laplace transform values at λ=1 and λ=2.
///
/// Exact analytic representation (Mellin-Barnes) for I(λ):
///   I(λ) = (1/(2πi)) ∫_{c-i∞}^{c+i∞} Γ(z) λ^{-z} exp(-μ z + 0.5 σ² z²) dz, c>0.
/// This Mellin-Barnes integral is mathematically exact.
pub fn predict_gam_with_uncertainty<X>(
    x: X,
    beta: ArrayView1<'_, f64>,
    offset: ArrayView1<'_, f64>,
    family: crate::types::LikelihoodFamily,
    fit: &FitResult,
    options: &PredictUncertaintyOptions,
) -> Result<PredictUncertaintyResult, EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if x.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam_with_uncertainty dimension mismatch: X has {} columns but beta has length {}",
            x.ncols(),
            beta.len()
        )));
    }
    if x.nrows() != offset.len() {
        return Err(EstimationError::InvalidInput(format!(
            "predict_gam_with_uncertainty dimension mismatch: X has {} rows but offset has length {}",
            x.nrows(),
            offset.len()
        )));
    }
    if !(options.confidence_level.is_finite()
        && options.confidence_level > 0.0
        && options.confidence_level < 1.0)
    {
        return Err(EstimationError::InvalidInput(format!(
            "confidence_level must be in (0,1), got {}",
            options.confidence_level
        )));
    }

    let requested_mode = options.covariance_mode;
    // Covariance selection corresponds to approximation order:
    // - Conditional: uses only A(mu) = H_mu^{-1}
    // - Corrected: adds first-order Var(b(rho)) term J V_rho J^T
    let (cov, covariance_corrected_used) = match requested_mode {
        InferenceCovarianceMode::Conditional => (
            fit.beta_covariance.as_ref().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "fit result does not contain conditional covariance".to_string(),
                )
            })?,
            false,
        ),
        InferenceCovarianceMode::ConditionalPlusSmoothingPreferred => {
            if let Some(cov_corr) = fit.beta_covariance_corrected.as_ref() {
                (cov_corr, true)
            } else if let Some(cov_base) = fit.beta_covariance.as_ref() {
                (cov_base, false)
            } else {
                return Err(EstimationError::InvalidInput(
                    "fit result does not contain a usable posterior covariance".to_string(),
                ));
            }
        }
        InferenceCovarianceMode::ConditionalPlusSmoothingRequired => (
            fit.beta_covariance_corrected.as_ref().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "fit result does not contain smoothing-corrected covariance".to_string(),
                )
            })?,
            true,
        ),
    };

    if cov.nrows() != beta.len() || cov.ncols() != beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "covariance dimension mismatch: expected {}x{}, got {}x{}",
            beta.len(),
            beta.len(),
            cov.nrows(),
            cov.ncols()
        )));
    }

    let mut eta = x.matrix_vector_multiply(&beta.to_owned());
    eta += &offset;
    let mean = apply_family_inverse_link(&eta, family)?;

    let eta_var = linear_predictor_variance(&x, cov);
    let eta_standard_error = eta_var.mapv(|v| v.max(0.0).sqrt());

    let z = standard_normal_quantile(0.5 + 0.5 * options.confidence_level)
        .map_err(EstimationError::InvalidInput)?;
    let eta_lower = &eta - &eta_standard_error.mapv(|s| z * s);
    let eta_upper = &eta + &eta_standard_error.mapv(|s| z * s);

    // Derivative of inverse link g^{-1}(η) used for delta-method:
    //   Var(μ_i) ≈ [d g^{-1}(η_i)/dη]^2 Var(η_i).
    //
    // For logit:
    //   g^{-1}(η)=sigmoid(η), dμ/dη=μ(1-μ).
    // If η itself is uncertain (η ~ N(m,v)), the exact predictive mean is
    // E[sigmoid(η)] (logistic-normal integral) as documented above.
    //
    // For cloglog:
    //   g^{-1}(η)=1-exp(-exp(η)), dμ/dη=exp(η)exp(-exp(η)).
    // With uncertain η the exact moments can be written via I(λ)=E[exp(-λexp(η))],
    // and:
    //   E[μ]   = 1 - I(1),
    //   E[μ²]  = 1 - 2I(1) + I(2),
    //   Var(μ) = I(2) - I(1)^2.
    // These identities characterize the exact cloglog moments under Gaussian η uncertainty.
    let mut mean_standard_error = Array1::<f64>::zeros(eta.len());
    for i in 0..eta.len() {
        let dmu_deta = match family {
            crate::types::LikelihoodFamily::GaussianIdentity => 1.0,
            crate::types::LikelihoodFamily::BinomialLogit => {
                let mu = mean[i];
                mu * (1.0 - mu)
            }
            crate::types::LikelihoodFamily::BinomialProbit => {
                crate::probability::normal_pdf(eta[i])
            }
            crate::types::LikelihoodFamily::BinomialCLogLog => {
                let z = eta[i].clamp(-30.0, 30.0);
                let exp_eta = z.exp();
                let surv = (-exp_eta).exp();
                exp_eta * surv
            }
            crate::types::LikelihoodFamily::RoystonParmar => unreachable!(),
        };
        mean_standard_error[i] = (dmu_deta.abs() * eta_standard_error[i]).max(0.0);
    }

    let (mut mean_lower, mut mean_upper) = match options.mean_interval_method {
        MeanIntervalMethod::Delta => (
            &mean - &mean_standard_error.mapv(|s| z * s),
            &mean + &mean_standard_error.mapv(|s| z * s),
        ),
        MeanIntervalMethod::TransformEta => (
            apply_family_inverse_link(&eta_lower, family)?,
            apply_family_inverse_link(&eta_upper, family)?,
        ),
    };

    if matches!(
        family,
        crate::types::LikelihoodFamily::BinomialLogit
            | crate::types::LikelihoodFamily::BinomialProbit
            | crate::types::LikelihoodFamily::BinomialCLogLog
    ) {
        mean_lower.mapv_inplace(|v| v.clamp(0.0, 1.0));
        mean_upper.mapv_inplace(|v| v.clamp(0.0, 1.0));
    }

    let (observation_lower, observation_upper) = if options.include_observation_interval
        && matches!(family, crate::types::LikelihoodFamily::GaussianIdentity)
    {
        let obs_se = eta_var.mapv(|v| (v + fit.scale.max(0.0)).max(0.0).sqrt());
        let lower = &eta - &obs_se.mapv(|s| z * s);
        let upper = &eta + &obs_se.mapv(|s| z * s);
        (Some(lower), Some(upper))
    } else {
        (None, None)
    };

    Ok(PredictUncertaintyResult {
        eta,
        mean,
        eta_standard_error,
        mean_standard_error,
        eta_lower,
        eta_upper,
        mean_lower,
        mean_upper,
        observation_lower,
        observation_upper,
        covariance_mode_requested: requested_mode,
        covariance_corrected_used,
    })
}

/// Coefficient-level uncertainty and confidence intervals.
pub fn coefficient_uncertainty(
    fit: &FitResult,
    confidence_level: f64,
    covariance_mode: InferenceCovarianceMode,
) -> Result<CoefficientUncertaintyResult, EstimationError> {
    coefficient_uncertainty_with_mode(fit, confidence_level, covariance_mode)
}

/// Coefficient-level uncertainty and confidence intervals with explicit covariance mode.
pub fn coefficient_uncertainty_with_mode(
    fit: &FitResult,
    confidence_level: f64,
    covariance_mode: InferenceCovarianceMode,
) -> Result<CoefficientUncertaintyResult, EstimationError> {
    if !(confidence_level.is_finite() && confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(EstimationError::InvalidInput(format!(
            "confidence_level must be in (0,1), got {}",
            confidence_level
        )));
    }
    // Coefficient SEs are extracted from either:
    // - conditional covariance H^{-1}, or
    // - first-order corrected covariance H^{-1} + J V_rho J^T.
    let (se, corrected) = match covariance_mode {
        InferenceCovarianceMode::Conditional => (
            fit.beta_standard_errors.as_ref().cloned().ok_or_else(|| {
                EstimationError::InvalidInput(
                    "fit result does not contain conditional coefficient standard errors"
                        .to_string(),
                )
            })?,
            false,
        ),
        InferenceCovarianceMode::ConditionalPlusSmoothingPreferred => {
            if let Some(se_corr) = fit.beta_standard_errors_corrected.as_ref() {
                (se_corr.clone(), true)
            } else if let Some(se_base) = fit.beta_standard_errors.as_ref() {
                (se_base.clone(), false)
            } else {
                return Err(EstimationError::InvalidInput(
                    "fit result does not contain coefficient standard errors".to_string(),
                ));
            }
        }
        InferenceCovarianceMode::ConditionalPlusSmoothingRequired => (
            fit.beta_standard_errors_corrected
                .as_ref()
                .cloned()
                .ok_or_else(|| {
                    EstimationError::InvalidInput(
                        "fit result does not contain smoothing-corrected coefficient standard errors"
                            .to_string(),
                    )
                })?,
            true,
        ),
    };

    if se.len() != fit.beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "standard error length mismatch: beta has {}, se has {}",
            fit.beta.len(),
            se.len()
        )));
    }

    let z = standard_normal_quantile(0.5 + 0.5 * confidence_level)
        .map_err(EstimationError::InvalidInput)?;
    let lower = &fit.beta - &se.mapv(|s| z * s);
    let upper = &fit.beta + &se.mapv(|s| z * s);
    Ok(CoefficientUncertaintyResult {
        estimate: fit.beta.clone(),
        standard_error: se,
        lower,
        upper,
        corrected,
        covariance_mode_requested: covariance_mode,
    })
}

/// Computes the gradient of the LAML cost function using the central finite-difference method.
const FD_REL_GAP_THRESHOLD: f64 = 0.2;
const FD_MIN_BASE_STEP: f64 = 1e-6;
const FD_MAX_REFINEMENTS: usize = 4;
const FD_RIDGE_REL_JITTER_THRESHOLD: f64 = 1e-3;
const FD_RIDGE_ABS_JITTER_THRESHOLD: f64 = 1e-12;

#[derive(Clone, Copy, Debug)]
struct GradAuditConfig {
    every: usize,
    rel_tol: f64,
    abs_tol: f64,
    fallback_to_fd: bool,
}

#[derive(Clone, Copy, Debug)]
enum TraceBackend {
    Exact,
    Hutchinson { probes: usize },
    HutchPP { probes: usize, sketch: usize },
}

fn grad_audit_config() -> GradAuditConfig {
    GradAuditConfig {
        every: 10,
        rel_tol: 5e-2,
        abs_tol: 1e-4,
        fallback_to_fd: true,
    }
}

fn gradient_audit_stats(
    analytic: &Array1<f64>,
    reference: &Array1<f64>,
) -> Option<(f64, f64, f64, f64)> {
    if analytic.len() != reference.len() {
        return None;
    }
    let mut sq_diff = 0.0_f64;
    let mut sq_ref = 0.0_f64;
    let mut max_abs = 0.0_f64;
    let mut max_ref_abs = 0.0_f64;
    for (&a, &r) in analytic.iter().zip(reference.iter()) {
        if !(a.is_finite() && r.is_finite()) {
            return None;
        }
        let d = a - r;
        sq_diff += d * d;
        sq_ref += r * r;
        max_abs = max_abs.max(d.abs());
        max_ref_abs = max_ref_abs.max(r.abs());
    }
    let rel_l2 = sq_diff.sqrt() / sq_ref.sqrt().max(1e-12);
    Some((rel_l2, max_abs, sq_ref.sqrt(), max_ref_abs))
}

fn maybe_audit_gradient(
    reml_state: &internal::RemlState<'_>,
    rho: &Array1<f64>,
    eval_num: u64,
    grad: &mut Array1<f64>,
) {
    let audit_cfg = grad_audit_config();
    if audit_cfg.every == 0 || rho.is_empty() || eval_num % (audit_cfg.every as u64) != 0 {
        return;
    }
    match compute_fd_gradient_internal(reml_state, rho, false, true) {
        Ok(fd_grad) => {
            if let Some((rel_l2, max_abs, ref_l2, ref_max_abs)) =
                gradient_audit_stats(grad, &fd_grad)
            {
                let mismatch = rel_l2 > audit_cfg.rel_tol || max_abs > audit_cfg.abs_tol;
                if mismatch {
                    log::warn!(
                        "[GRAD AUDIT] eval={} rel_l2={:.3e} (tol {:.3e}) max_abs={:.3e} (tol {:.3e}) ref_l2={:.3e} ref_max={:.3e}",
                        eval_num,
                        rel_l2,
                        audit_cfg.rel_tol,
                        max_abs,
                        audit_cfg.abs_tol,
                        ref_l2,
                        ref_max_abs,
                    );
                    if audit_cfg.fallback_to_fd {
                        log::warn!(
                            "[GRAD AUDIT] eval={} replacing analytic gradient with FD gradient for this step",
                            eval_num
                        );
                        *grad = fd_grad;
                    }
                }
            } else {
                log::warn!(
                    "[GRAD AUDIT] eval={} produced non-finite comparison stats; skipping audit decision",
                    eval_num
                );
            }
        }
        Err(e) => {
            log::warn!(
                "[GRAD AUDIT] eval={} failed FD gradient computation: {:?}",
                eval_num,
                e
            );
        }
    }
}

struct FdEval {
    f_p: f64,
    f_m: f64,
    f_p2: f64,
    f_m2: f64,
    d_small: f64,
    d_big: f64,
    ridge_min: f64,
    ridge_max: f64,
    ridge_rel_span: f64,
    ridge_jitter: bool,
}

fn evaluate_fd_pair(
    reml_state: &internal::RemlState,
    rho: &Array1<f64>,
    coord: usize,
    base_h: f64,
) -> Result<FdEval, EstimationError> {
    let mut rho_p = rho.clone();
    rho_p[coord] += 0.5 * base_h;
    let mut rho_m = rho.clone();
    rho_m[coord] -= 0.5 * base_h;
    let f_p = reml_state.compute_cost(&rho_p)?;
    let ridge_p = reml_state.last_ridge_used().unwrap_or(f64::NAN);

    let f_m = reml_state.compute_cost(&rho_m)?;
    let ridge_m = reml_state.last_ridge_used().unwrap_or(f64::NAN);
    let d_small = (f_p - f_m) / base_h;

    let h2 = 2.0 * base_h;
    let mut rho_p2 = rho.clone();
    rho_p2[coord] += 0.5 * h2;
    let mut rho_m2 = rho.clone();
    rho_m2[coord] -= 0.5 * h2;
    let f_p2 = reml_state.compute_cost(&rho_p2)?;
    let ridge_p2 = reml_state.last_ridge_used().unwrap_or(f64::NAN);

    let f_m2 = reml_state.compute_cost(&rho_m2)?;
    let ridge_m2 = reml_state.last_ridge_used().unwrap_or(f64::NAN);
    let d_big = (f_p2 - f_m2) / h2;

    let finite_ridges: Vec<f64> = [ridge_p, ridge_m, ridge_p2, ridge_m2]
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect();
    let (ridge_min, ridge_max, ridge_span, ridge_rel_span) = if finite_ridges.is_empty() {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    } else {
        let mut min_v = f64::INFINITY;
        let mut max_v = f64::NEG_INFINITY;
        for v in finite_ridges {
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        let span = max_v - min_v;
        let rel = span / max_v.abs().max(1e-12);
        (min_v, max_v, span, rel)
    };
    let ridge_jitter = ridge_span.is_finite()
        && ridge_rel_span.is_finite()
        && (ridge_span > FD_RIDGE_ABS_JITTER_THRESHOLD
            && ridge_rel_span > FD_RIDGE_REL_JITTER_THRESHOLD);

    Ok(FdEval {
        f_p,
        f_m,
        f_p2,
        f_m2,
        d_small,
        d_big,
        ridge_min,
        ridge_max,
        ridge_rel_span,
        ridge_jitter,
    })
}

fn fd_same_sign(d_small: f64, d_big: f64) -> bool {
    if !d_small.is_finite() || !d_big.is_finite() {
        false
    } else {
        (d_small >= 0.0 && d_big >= 0.0) || (d_small <= 0.0 && d_big <= 0.0)
    }
}

fn select_fd_derivative(d_small: f64, d_big: f64, same_sign: bool) -> f64 {
    match (d_small.is_finite(), d_big.is_finite()) {
        (true, true) => {
            if same_sign {
                d_small
            } else {
                d_big
            }
        }
        (true, false) => d_small,
        (false, true) => d_big,
        (false, false) => 0.0,
    }
}

fn compute_fd_gradient_internal(
    reml_state: &internal::RemlState,
    rho: &Array1<f64>,
    emit_logs: bool,
    allow_analytic_fallback: bool,
) -> Result<Array1<f64>, EstimationError> {
    let mut fd_grad = Array1::zeros(rho.len());
    let mut analytic_fallback: Option<Array1<f64>> = None;

    let mut log_lines: Vec<String> = Vec::new();
    let (rho_min, rho_max) = rho
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &v| {
            (min.min(v), max.max(v))
        });
    let rho_summary = format!("len={} range=[{:.3e},{:.3e}]", rho.len(), rho_min, rho_max);
    match reml_state.last_ridge_used() {
        Some(ridge) => log_lines.push(format!(
            "[FD RIDGE] Baseline cached ridge: {ridge:.3e} for rho {rho_summary}",
        )),
        None => log_lines.push(format!(
            "[FD RIDGE] No cached baseline ridge available for rho {rho_summary}",
        )),
    }

    for i in 0..rho.len() {
        let h_rel = 1e-4_f64 * (1.0 + rho[i].abs());
        let h_abs = 1e-5_f64;
        let mut base_h = h_rel.max(h_abs);

        log_lines.push(format!("[FD RIDGE] coord {i} rho={:+.6e}", rho[i]));

        let mut d_small = 0.0;
        let mut d_big = 0.0;
        let mut derivative: Option<f64> = None;
        let mut best_rel_gap = f64::INFINITY;
        let mut best_derivative: Option<f64> = None;
        let mut last_rel_gap = f64::INFINITY;
        let mut refine_steps = 0usize;
        let mut rel_gap_first = None;
        let mut rel_gap_max = 0.0;
        let mut ridge_jitter_seen = false;
        let mut ridge_rel_span_max = 0.0;
        let h_start = base_h;

        for attempt in 0..=FD_MAX_REFINEMENTS {
            let eval = evaluate_fd_pair(reml_state, rho, i, base_h)?;
            d_small = eval.d_small;
            d_big = eval.d_big;
            ridge_jitter_seen |= eval.ridge_jitter;
            if eval.ridge_rel_span.is_finite() && eval.ridge_rel_span > ridge_rel_span_max {
                ridge_rel_span_max = eval.ridge_rel_span;
            }

            let denom = d_small.abs().max(d_big.abs()).max(1e-12);
            let rel_gap = (d_small - d_big).abs() / denom;
            let same_sign = fd_same_sign(d_small, d_big);

            if same_sign && !eval.ridge_jitter {
                if rel_gap <= best_rel_gap {
                    best_rel_gap = rel_gap;
                    best_derivative = Some(select_fd_derivative(d_small, d_big, same_sign));
                }
                if rel_gap > last_rel_gap {
                    // Smaller steps are worsening the agreement; keep the best seen.
                    derivative = best_derivative;
                    break;
                }
                last_rel_gap = rel_gap;
            }

            let refine_for_rel_gap =
                same_sign && rel_gap > FD_REL_GAP_THRESHOLD && base_h * 0.5 >= FD_MIN_BASE_STEP;
            let refine_for_ridge = eval.ridge_jitter && base_h * 0.5 >= FD_MIN_BASE_STEP;
            let refining = refine_for_rel_gap || refine_for_ridge;
            if attempt == 0 {
                rel_gap_first = Some(rel_gap);
            }
            if rel_gap.is_finite() && rel_gap > rel_gap_max {
                rel_gap_max = rel_gap;
            }
            let last_attempt = attempt == FD_MAX_REFINEMENTS || !refining;
            if attempt == 0 || last_attempt {
                if attempt == 0 {
                    log_lines.push(format!(
                        "[FD RIDGE]   attempt {} h={:.3e} f(+/-0.5h)={:+.9e}/{:+.9e} \
f(+/-1h)={:+.9e}/{:+.9e} d_small={:+.9e} d_big={:+.9e} ridge=[{:.3e},{:.3e}]",
                        attempt + 1,
                        base_h,
                        eval.f_p,
                        eval.f_m,
                        eval.f_p2,
                        eval.f_m2,
                        d_small,
                        d_big,
                        eval.ridge_min,
                        eval.ridge_max,
                    ));
                } else {
                    log_lines.push(format!(
                        "[FD RIDGE]   attempt {} h={:.3e} d_small={:+.9e} d_big={:+.9e} \
rel_gap={:.3e} ridge=[{:.3e},{:.3e}] ridge_rel_span={:.3e}",
                        attempt + 1,
                        base_h,
                        d_small,
                        d_big,
                        rel_gap,
                        eval.ridge_min,
                        eval.ridge_max,
                        eval.ridge_rel_span
                    ));
                }
            }

            if refining {
                base_h *= 0.5;
                refine_steps += 1;
                continue;
            }

            if eval.ridge_jitter {
                derivative = None;
            } else {
                derivative = Some(select_fd_derivative(d_small, d_big, same_sign));
            }
            break;
        }

        if derivative.is_none() {
            let same_sign = fd_same_sign(d_small, d_big);
            if same_sign && !ridge_jitter_seen {
                derivative = best_derivative
                    .or_else(|| Some(select_fd_derivative(d_small, d_big, same_sign)));
            } else if !ridge_jitter_seen {
                derivative = Some(select_fd_derivative(d_small, d_big, same_sign));
            }
        }

        if derivative.is_none() && allow_analytic_fallback {
            if analytic_fallback.is_none() {
                analytic_fallback = Some(reml_state.compute_gradient(rho)?);
            }
            derivative = analytic_fallback.as_ref().map(|g| g[i]);
            log_lines.push(format!(
                "[FD RIDGE]   coord {} fallback to analytic gradient due to ridge jitter (max rel span {:.3e})",
                i, ridge_rel_span_max
            ));
        }

        fd_grad[i] = derivative.unwrap_or(f64::NAN);
        let rel_gap_first = rel_gap_first.unwrap_or(f64::NAN);
        log_lines.push(format!(
            "[FD RIDGE]   refine steps={} h_start={:.3e} h_final={:.3e} rel_gap_first={:.3e} rel_gap_max={:.3e} ridge_jitter_seen={} ridge_rel_span_max={:.3e}",
            refine_steps,
            h_start,
            base_h,
            rel_gap_first,
            rel_gap_max,
            ridge_jitter_seen,
            ridge_rel_span_max
        ));
        log_lines.push(format!(
            "[FD RIDGE]   chosen derivative = {:+.9e}",
            fd_grad[i]
        ));
    }

    if emit_logs && !log_lines.is_empty() {
        println!("{}", log_lines.join("\n"));
    }

    Ok(fd_grad)
}

fn compute_fd_gradient(
    reml_state: &internal::RemlState,
    rho: &Array1<f64>,
) -> Result<Array1<f64>, EstimationError> {
    compute_fd_gradient_internal(reml_state, rho, true, true)
}

/// Evaluate both analytic and finite-difference gradients for the external REML objective.
pub fn evaluate_external_gradients<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: &[Array2<f64>],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
) -> Result<(Array1<f64>, Array1<f64>), EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if !(y.len() == w.len() && y.len() == x.nrows() && y.len() == offset.len()) {
        return Err(EstimationError::InvalidInput(format!(
            "Row mismatch: y={}, w={}, X.rows={}, offset={}",
            y.len(),
            w.len(),
            x.nrows(),
            offset.len()
        )));
    }

    let p = x.ncols();
    validate_full_size_penalties(s_list, p, "evaluate_external_gradients")?;

    let (link, firth_active) = resolve_external_family(opts.family, opts.firth_bias_reduction)?;
    let cfg = RemlConfig::external(link, opts.tol, opts.max_iter, firth_active);

    let s_vec: Vec<Array2<f64>> = s_list.to_vec();
    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let x_o = x.clone();
    let offset_o = offset.to_owned();

    let reml_state = internal::RemlState::new_with_offset(
        y_o.view(),
        x_o,
        w_o.view(),
        offset_o.view(),
        s_vec,
        p,
        &cfg,
        Some(opts.nullspace_dims.clone()),
    )?;

    let analytic_grad = reml_state.compute_gradient(rho)?;
    let fd_grad = compute_fd_gradient(&reml_state, rho)?;

    Ok((analytic_grad, fd_grad))
}

/// Evaluate the external cost and report the stabilization ridge used.
/// This is a diagnostic helper for tests that need to detect ridge jitter.
pub fn evaluate_external_cost_and_ridge<X>(
    y: ArrayView1<'_, f64>,
    w: ArrayView1<'_, f64>,
    x: X,
    offset: ArrayView1<'_, f64>,
    s_list: &[Array2<f64>],
    opts: &ExternalOptimOptions,
    rho: &Array1<f64>,
) -> Result<(f64, f64), EstimationError>
where
    X: Into<DesignMatrix>,
{
    let x = x.into();
    if !(y.len() == w.len() && y.len() == x.nrows() && y.len() == offset.len()) {
        return Err(EstimationError::InvalidInput(format!(
            "Row mismatch: y={}, w={}, X.rows={}, offset={}",
            y.len(),
            w.len(),
            x.nrows(),
            offset.len()
        )));
    }

    let p = x.ncols();
    validate_full_size_penalties(s_list, p, "evaluate_external_cost_and_ridge")?;

    let (link, firth_active) = resolve_external_family(opts.family, opts.firth_bias_reduction)?;
    let cfg = RemlConfig::external(link, opts.tol, opts.max_iter, firth_active);

    let s_vec: Vec<Array2<f64>> = s_list.to_vec();
    let y_o = y.to_owned();
    let w_o = w.to_owned();
    let x_o = x.clone();
    let offset_o = offset.to_owned();

    let reml_state = internal::RemlState::new_with_offset(
        y_o.view(),
        x_o,
        w_o.view(),
        offset_o.view(),
        s_vec,
        p,
        &cfg,
        Some(opts.nullspace_dims.clone()),
    )?;

    let cost = reml_state.compute_cost(rho)?;
    let ridge = reml_state.last_ridge_used().unwrap_or(0.0);
    Ok((cost, ridge))
}

/// Internal module for estimation logic.
// Make internal module public for tests
pub mod internal {
    use super::*;
    use faer::Side;

    enum FaerFactor {
        Llt(FaerLlt<f64>),
        Lblt(FaerLblt<f64>),
        Ldlt(FaerLdlt<f64>),
    }

    impl FaerFactor {
        fn solve(&self, rhs: faer::MatRef<'_, f64>) -> FaerMat<f64> {
            match self {
                FaerFactor::Llt(f) => f.solve(rhs),
                FaerFactor::Lblt(f) => f.solve(rhs),
                FaerFactor::Ldlt(f) => f.solve(rhs),
            }
        }

        fn solve_in_place(&self, rhs: faer::MatMut<'_, f64>) {
            match self {
                FaerFactor::Llt(f) => f.solve_in_place(rhs),
                FaerFactor::Lblt(f) => f.solve_in_place(rhs),
                FaerFactor::Ldlt(f) => f.solve_in_place(rhs),
            }
        }
    }

    /// Holds the state for the outer REML optimization and supplies cost and
    /// gradient evaluations to the `wolfe_bfgs` optimizer.
    ///
    /// The `cache` field uses `RefCell` to enable interior mutability. This is a crucial
    /// performance optimization. The `cost_and_grad` closure required by the BFGS
    /// optimizer takes an immutable reference `&self`. However, we want to cache the
    /// results of the expensive P-IRLS computation to avoid re-calculating the fit
    /// for the same `rho` vector, which can happen during the line search.
    /// `RefCell` allows us to mutate the cache through a `&self` reference,
    /// making this optimization possible while adhering to the optimizer's API.

    #[derive(Clone)]
    struct EvalShared {
        key: Option<Vec<u64>>,
        pirls_result: Arc<PirlsResult>,
        h_eff: Arc<Array2<f64>>,
        ridge_passport: RidgePassport,
        /// The exact H_total matrix used for LAML cost computation.
        /// For Firth: h_eff - h_phi. For non-Firth: h_eff.
        h_total: Arc<Array2<f64>>,

        // ══════════════════════════════════════════════════════════════════════
        // WHY TWO INVERSES? (Hybrid Approach for Indefinite Hessians)
        // ══════════════════════════════════════════════════════════════════════
        //
        // The LAML gradient has two terms requiring DIFFERENT matrix inverses:
        //
        // 1. TRACE TERM (∂/∂ρ log|H|): Uses PSEUDOINVERSE H₊†
        //    - Cost defines log|H| = Σᵢ log(λᵢ) for λᵢ > ε only (truncated)
        //    - Derivative: ∂J/∂ρ = ½ tr(H₊† ∂H/∂ρ)
        //    - H₊† = Σᵢ (1/λᵢ) uᵢuᵢᵀ for positive λᵢ only
        //    - Negative eigenvalues contribute 0 to cost, so their derivative contribution is 0
        //
        // 2. IMPLICIT TERM (dβ/dρ): Uses RIDGED FACTOR (H + δI)⁻¹
        //    - PIRLS stabilizes indefinite H by adding ridge: solves (H + δI)β = ...
        //    - Stationarity condition: G(β,ρ) = ∇L + δβ = 0
        //    - By Implicit Function Theorem: dβ/dρ = (H + δI)⁻¹ (λₖ Sₖ β)
        //    - Must use ridged inverse because β moves on the RIDGED surface
        //
        // EXAMPLE: H = -5 (indefinite), ridge δ = 10
        //   Trace term: Pseudoinverse → 0 (correct: truncated eigenvalue)
        //               Ridged inverse → 0.2 (WRONG: gradient of non-existent curve)
        //   Implicit term: Ridged inverse → 1/5 (correct: solver sees stiffness +5)
        //                  Pseudoinverse → 0 or ∞ (WRONG: ignores ridge physics)
        //
        // ══════════════════════════════════════════════════════════════════════
        /// Positive-spectrum factor W = U_+ diag(1/sqrt(lambda_+)).
        /// This avoids materializing H₊† = W Wᵀ in hot paths.
        ///
        /// We use identities:
        ///   H₊† v = W (Wᵀ v)
        ///   tr(H₊† S_k) = ||R_k W||_F², where S_k = R_kᵀ R_k.
        h_pos_factor_w: Arc<Array2<f64>>,

        /// Log determinant via truncation: Σᵢ log(λᵢ) for λᵢ > ε only.
        h_total_log_det: f64,
    }

    impl EvalShared {
        fn matches(&self, key: &Option<Vec<u64>>) -> bool {
            match (&self.key, key) {
                (None, None) => true,
                (Some(a), Some(b)) => a == b,
                _ => false,
            }
        }
    }

    struct RemlWorkspace {
        rho_plus: Array1<f64>,
        rho_minus: Array1<f64>,
        lambda_values: Array1<f64>,
        grad_primary: Array1<f64>,
        grad_secondary: Array1<f64>,
        cost_gradient: Array1<f64>,
        prior_gradient: Array1<f64>,
        concat: Array2<f64>,
        solved: Array2<f64>,
        block_ranges: Vec<(usize, usize)>,
        solved_rows: usize,
    }

    impl RemlWorkspace {
        fn new(max_penalties: usize, coeffs: usize, total_rank: usize) -> Self {
            RemlWorkspace {
                rho_plus: Array1::zeros(max_penalties),
                rho_minus: Array1::zeros(max_penalties),
                lambda_values: Array1::zeros(max_penalties),
                grad_primary: Array1::zeros(max_penalties),
                grad_secondary: Array1::zeros(max_penalties),
                cost_gradient: Array1::zeros(max_penalties),
                prior_gradient: Array1::zeros(max_penalties),
                concat: Array2::zeros((coeffs, total_rank)),
                solved: Array2::zeros((coeffs, total_rank)),
                block_ranges: Vec::with_capacity(max_penalties),
                solved_rows: coeffs,
            }
        }

        fn reset_for_eval(&mut self, penalties: usize) {
            self.block_ranges.clear();
            self.solved_rows = 0;
            if penalties == 0 {
                return;
            }
            self.grad_primary.slice_mut(s![..penalties]).fill(0.0);
            self.grad_secondary.slice_mut(s![..penalties]).fill(0.0);
            self.cost_gradient.slice_mut(s![..penalties]).fill(0.0);
            self.prior_gradient.slice_mut(s![..penalties]).fill(0.0);
        }

        fn reset_block_ranges(&mut self) {
            self.block_ranges.clear();
            self.solved_rows = 0;
        }

        fn set_lambda_values(&mut self, rho: &Array1<f64>) {
            let len = rho.len();
            if len == 0 {
                return;
            }
            let mut view = self.lambda_values.slice_mut(s![..len]);
            for (dst, &src) in view.iter_mut().zip(rho.iter()) {
                *dst = src.exp();
            }
        }

        fn lambda_view(&self, len: usize) -> ArrayView1<'_, f64> {
            self.lambda_values.slice(s![..len])
        }

        fn cost_gradient_view(&mut self, len: usize) -> ArrayViewMut1<'_, f64> {
            self.cost_gradient.slice_mut(s![..len])
        }

        fn zero_cost_gradient(&mut self, len: usize) {
            self.cost_gradient.slice_mut(s![..len]).fill(0.0);
        }

        fn cost_gradient_view_const(&self, len: usize) -> ArrayView1<'_, f64> {
            self.cost_gradient.slice(s![..len])
        }

        fn soft_prior_cost_and_grad<'a>(
            &'a mut self,
            rho: &Array1<f64>,
        ) -> (f64, ArrayView1<'a, f64>) {
            let len = rho.len();
            let mut grad_view = self.prior_gradient.slice_mut(s![..len]);
            grad_view.fill(0.0);

            if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
                return (0.0, self.prior_gradient.slice(s![..len]));
            }

            let inv_bound = 1.0 / RHO_BOUND;
            let sharp = RHO_SOFT_PRIOR_SHARPNESS;
            let mut cost = 0.0;
            for (grad, &ri) in grad_view.iter_mut().zip(rho.iter()) {
                let scaled = sharp * ri * inv_bound;
                cost += scaled.cosh().ln();
                *grad = sharp * inv_bound * scaled.tanh();
            }

            if RHO_SOFT_PRIOR_WEIGHT != 1.0 {
                for grad in grad_view.iter_mut() {
                    *grad *= RHO_SOFT_PRIOR_WEIGHT;
                }
                cost *= RHO_SOFT_PRIOR_WEIGHT;
            }

            (cost, self.prior_gradient.slice(s![..len]))
        }
    }

    pub(crate) struct RemlState<'a> {
        y: ArrayView1<'a, f64>,
        x: DesignMatrix,
        weights: ArrayView1<'a, f64>,
        offset: Array1<f64>,
        // Original penalty matrices S_k (p × p), ρ-independent basis
        s_full_list: Vec<Array2<f64>>,
        pub(super) rs_list: Vec<Array2<f64>>, // Pre-computed penalty square roots
        balanced_penalty_root: Array2<f64>,
        reparam_invariant: ReparamInvariant,
        p: usize,
        config: &'a RemlConfig,
        nullspace_dims: Vec<usize>,

        cache: RwLock<HashMap<Vec<u64>, Arc<PirlsResult>>>,
        faer_factor_cache: RwLock<HashMap<Vec<u64>, Arc<FaerFactor>>>,
        eval_count: RwLock<u64>,
        last_cost: RwLock<f64>,
        last_grad_norm: RwLock<f64>,
        consecutive_cost_errors: RwLock<usize>,
        last_cost_error_msg: RwLock<Option<String>>,
        current_eval_bundle: RwLock<Option<EvalShared>>,
        cost_last: RwLock<Option<CostAgg>>,
        cost_repeat: RwLock<u64>,
        cost_last_emit: RwLock<u64>,
        cost_eval_count: RwLock<u64>,
        raw_cond_snapshot: RwLock<f64>,
        gaussian_cond_snapshot: RwLock<f64>,
        workspace: Mutex<RemlWorkspace>,
        pub(super) warm_start_beta: RwLock<Option<Coefficients>>,
        warm_start_enabled: AtomicBool,
    }

    #[derive(Clone)]
    struct CostKey {
        compact: String,
    }

    #[derive(Clone)]
    struct CostAgg {
        key: CostKey,
        count: u64,
        stab_cond_min: f64,
        stab_cond_max: f64,
        stab_cond_last: f64,
        raw_cond_min: f64,
        raw_cond_max: f64,
        raw_cond_last: f64,
        laml_min: f64,
        laml_max: f64,
        laml_last: f64,
        edf_min: f64,
        edf_max: f64,
        edf_last: f64,
        trace_min: f64,
        trace_max: f64,
        trace_last: f64,
    }

    impl CostKey {
        fn new(rho: &[f64], smooth: &[f64], stab_cond: f64, raw_cond: f64) -> Self {
            let rho_compact = format_compact_series(rho, |v| format!("{:.3}", v));
            let smooth_compact = format_compact_series(smooth, |v| format!("{:.2e}", v));
            let compact = format!(
                "rho={} | smooth={} | κ(stable/raw)={:.3e}/{:.3e}",
                rho_compact, smooth_compact, stab_cond, raw_cond
            );
            let compact = compact.replace("-0.000", "0.000");
            Self { compact }
        }

        fn approx_eq(&self, other: &Self) -> bool {
            self.compact == other.compact
        }

        fn format_compact(&self) -> String {
            self.compact.clone()
        }
    }

    impl CostAgg {
        fn new(
            key: CostKey,
            laml: f64,
            edf: f64,
            trace: f64,
            stab_cond: f64,
            raw_cond: f64,
        ) -> Self {
            Self {
                key,
                count: 1,
                stab_cond_min: stab_cond,
                stab_cond_max: stab_cond,
                stab_cond_last: stab_cond,
                raw_cond_min: raw_cond,
                raw_cond_max: raw_cond,
                raw_cond_last: raw_cond,
                laml_min: laml,
                laml_max: laml,
                laml_last: laml,
                edf_min: edf,
                edf_max: edf,
                edf_last: edf,
                trace_min: trace,
                trace_max: trace,
                trace_last: trace,
            }
        }

        fn update(&mut self, laml: f64, edf: f64, trace: f64, stab_cond: f64, raw_cond: f64) {
            self.count += 1;
            self.laml_last = laml;
            self.edf_last = edf;
            self.trace_last = trace;
            self.stab_cond_last = stab_cond;
            self.raw_cond_last = raw_cond;
            if stab_cond < self.stab_cond_min {
                self.stab_cond_min = stab_cond;
            }
            if stab_cond > self.stab_cond_max {
                self.stab_cond_max = stab_cond;
            }
            if raw_cond < self.raw_cond_min {
                self.raw_cond_min = raw_cond;
            }
            if raw_cond > self.raw_cond_max {
                self.raw_cond_max = raw_cond;
            }
            if laml < self.laml_min {
                self.laml_min = laml;
            }
            if laml > self.laml_max {
                self.laml_max = laml;
            }
            if edf < self.edf_min {
                self.edf_min = edf;
            }
            if edf > self.edf_max {
                self.edf_max = edf;
            }
            if trace < self.trace_min {
                self.trace_min = trace;
            }
            if trace > self.trace_max {
                self.trace_max = trace;
            }
        }

        fn format_summary(&self) -> String {
            let key = self.key.format_compact();
            let metric =
                |label: &str, min: f64, max: f64, last: f64, fmt: &dyn Fn(f64) -> String| {
                    if approx_f64(min, max, 1e-6, 1e-9) && approx_f64(min, last, 1e-6, 1e-9) {
                        format!("{label}={}", fmt(min))
                    } else {
                        let range = format_range(min, max, |v| fmt(v));
                        format!("{label}={range} last={}", fmt(last))
                    }
                };
            let kappa = if approx_f64(self.stab_cond_min, self.stab_cond_max, 1e-6, 1e-9)
                && approx_f64(self.raw_cond_min, self.raw_cond_max, 1e-6, 1e-9)
                && approx_f64(self.stab_cond_min, self.stab_cond_last, 1e-6, 1e-9)
                && approx_f64(self.raw_cond_min, self.raw_cond_last, 1e-6, 1e-9)
            {
                format!(
                    "κ(stable/raw)={}/{}",
                    format_cond(self.stab_cond_min),
                    format_cond(self.raw_cond_min)
                )
            } else {
                let stable = format_range(self.stab_cond_min, self.stab_cond_max, format_cond);
                let raw = format_range(self.raw_cond_min, self.raw_cond_max, format_cond);
                format!(
                    "κ(stable/raw)={stable}/{raw} last={}/{}",
                    format_cond(self.stab_cond_last),
                    format_cond(self.raw_cond_last)
                )
            };
            let laml = metric("LAML", self.laml_min, self.laml_max, self.laml_last, &|v| {
                format!("{:.6e}", v)
            });
            let edf = metric("EDF", self.edf_min, self.edf_max, self.edf_last, &|v| {
                format!("{:.6}", v)
            });
            let trace = metric(
                "tr(H^-1 Sλ)",
                self.trace_min,
                self.trace_max,
                self.trace_last,
                &|v| format!("{:.6}", v),
            );
            let count = if self.count > 1 {
                format!(" | count={}", self.count)
            } else {
                String::new()
            };
            format!("{key}{count} | {kappa} | {laml} | {edf} | {trace}",)
        }
    }

    // Formatting utilities moved to crate::diagnostics
    impl<'a> RemlState<'a> {
        #[inline]
        fn should_compute_hot_diagnostics(&self, eval_idx: u64) -> bool {
            // Keep expensive diagnostics out of the hot path unless they can
            // be surfaced. This has zero effect on optimization math.
            (log::log_enabled!(log::Level::Info) || log::log_enabled!(log::Level::Warn))
                && (eval_idx == 1 || eval_idx % 200 == 0)
        }

        fn log_det_s_with_ridge(
            s_transformed: &Array2<f64>,
            ridge_passport: RidgePassport,
            base_log_det: f64,
        ) -> Result<f64, EstimationError> {
            // Penalty determinant convention:
            //   S(ρ) = Σ_k λ_k S_k + δI,   λ_k = exp(ρ_k).
            //
            // For outer derivatives we require a rank-stable interpretation of log|S|_+:
            //   log|S|_+ = sum_{i in penalized subspace} log(σ_i(S)),
            // with a fixed structural nullspace convention.
            //
            // In this code path, `s_transformed` is already in the stabilized transformed basis
            // produced by reparameterization. When ridge participates in determinant terms
            // (`penalty_logdet_ridge > 0`), we evaluate log|S + ridge I| according to policy:
            //   - Full: ordinary SPD logdet (Cholesky).
            //   - PositivePart: positive-eigenvalue pseudo-logdet.
            //
            // Matching this convention between cost and gradient is mandatory because
            // det1[k] represents:
            //   ∂/∂ρ_k log|S(ρ)|_+ = tr(S^+ S_k^ρ),  S_k^ρ = λ_k S_k.
            let ridge = ridge_passport.penalty_logdet_ridge();
            if ridge <= 0.0 {
                return Ok(base_log_det);
            }

            let p = s_transformed.nrows();
            let mut s_ridge = s_transformed.clone();
            for i in 0..p {
                s_ridge[[i, i]] += ridge;
            }
            match ridge_passport.policy.determinant_mode {
                RidgeDeterminantMode::Full => {
                    let chol = s_ridge.clone().cholesky(Side::Lower).map_err(|_| {
                        EstimationError::ModelIsIllConditioned {
                            condition_number: f64::INFINITY,
                        }
                    })?;
                    Ok(2.0 * chol.diag().mapv(f64::ln).sum())
                }
                RidgeDeterminantMode::PositivePart => {
                    let (evals, _) = s_ridge
                        .eigh(Side::Lower)
                        .map_err(EstimationError::EigendecompositionFailed)?;
                    let floor = ridge.max(1e-14);
                    Ok(evals.iter().filter(|&&v| v > floor).map(|&v| v.ln()).sum())
                }
            }
        }

        fn log_gam_cost(
            &self,
            rho: &Array1<f64>,
            lambdas: &[f64],
            laml: f64,
            stab_cond: f64,
            raw_cond: f64,
            edf: f64,
            trace_h_inv_s_lambda: f64,
        ) {
            const GAM_REPEAT_EMIT: u64 = 50;
            const GAM_MIN_EMIT_GAP: u64 = 200;
            let rho_q = quantize_vec(rho.as_slice().unwrap_or_default(), 5e-3, 1e-6);
            let smooth_q = quantize_vec(lambdas, 5e-3, 1e-6);
            let stab_q = quantize_value(stab_cond, 5e-3, 1e-6);
            let raw_q = quantize_value(raw_cond, 5e-3, 1e-6);
            let key = CostKey::new(&rho_q, &smooth_q, stab_q, raw_q);

            let mut last_opt = self.cost_last.write().unwrap();
            let mut repeat = self.cost_repeat.write().unwrap();
            let mut last_emit = self.cost_last_emit.write().unwrap();
            let eval_idx = *self.eval_count.read().unwrap();

            if let Some(last) = last_opt.as_mut() {
                if last.key.approx_eq(&key) {
                    last.update(laml, edf, trace_h_inv_s_lambda, stab_q, raw_q);
                    *repeat += 1;
                    if *repeat >= GAM_REPEAT_EMIT
                        && eval_idx.saturating_sub(*last_emit) >= GAM_MIN_EMIT_GAP
                    {
                        println!("[GAM COST] {}", last.format_summary());
                        *repeat = 0;
                        *last_emit = eval_idx;
                    }
                    return;
                }

                let emit_prev =
                    last.count > 1 && eval_idx.saturating_sub(*last_emit) >= GAM_MIN_EMIT_GAP;
                if emit_prev {
                    println!("[GAM COST] {}", last.format_summary());
                    *last_emit = eval_idx;
                }
            }

            let new_agg = CostAgg::new(key, laml, edf, trace_h_inv_s_lambda, stab_q, raw_q);
            if eval_idx.saturating_sub(*last_emit) >= GAM_MIN_EMIT_GAP {
                println!("[GAM COST] {}", new_agg.format_summary());
                *last_emit = eval_idx;
            }
            *last_opt = Some(new_agg);
            *repeat = 0;
        }

        #[allow(dead_code)]
        pub fn reset_optimizer_tracking(&self) {
            *self.eval_count.write().unwrap() = 0;
            *self.last_cost.write().unwrap() = f64::INFINITY;
            *self.last_grad_norm.write().unwrap() = f64::INFINITY;
            *self.consecutive_cost_errors.write().unwrap() = 0;
            *self.last_cost_error_msg.write().unwrap() = None;
            self.current_eval_bundle.write().unwrap().take();
            self.cost_last.write().unwrap().take();
            *self.cost_repeat.write().unwrap() = 0;
            *self.cost_last_emit.write().unwrap() = 0;
            *self.cost_eval_count.write().unwrap() = 0;
            *self.raw_cond_snapshot.write().unwrap() = f64::NAN;
            *self.gaussian_cond_snapshot.write().unwrap() = f64::NAN;
        }

        /// Compute soft prior cost without needing workspace
        fn compute_soft_prior_cost(&self, rho: &Array1<f64>) -> f64 {
            let len = rho.len();
            if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
                return 0.0;
            }

            let inv_bound = 1.0 / RHO_BOUND;
            let sharp = RHO_SOFT_PRIOR_SHARPNESS;
            let mut cost = 0.0;
            for &ri in rho.iter() {
                let scaled = sharp * ri * inv_bound;
                cost += scaled.cosh().ln();
            }

            cost * RHO_SOFT_PRIOR_WEIGHT
        }

        /// Compute soft prior gradient without workspace mutation.
        fn compute_soft_prior_grad(&self, rho: &Array1<f64>) -> Array1<f64> {
            let len = rho.len();
            let mut grad = Array1::<f64>::zeros(len);
            if len == 0 || RHO_SOFT_PRIOR_WEIGHT == 0.0 {
                return grad;
            }
            let inv_bound = 1.0 / RHO_BOUND;
            let sharp = RHO_SOFT_PRIOR_SHARPNESS;
            for (g, &ri) in grad.iter_mut().zip(rho.iter()) {
                let scaled = sharp * ri * inv_bound;
                *g = sharp * inv_bound * scaled.tanh() * RHO_SOFT_PRIOR_WEIGHT;
            }
            grad
        }

        /// Returns the effective Hessian and the ridge value used (if any).
        /// Uses the same Hessian matrix in both cost and gradient calculations.
        ///
        /// PIRLS folds any stabilization ridge directly into the penalized objective:
        ///   l_p(β; ρ) = l(β) - 0.5 * βᵀ (S_λ + ridge I) β.
        /// Therefore the curvature used in LAML is
        ///   H_eff = X'WX + S_λ + ridge I,
        /// and adding another ridge here places the Laplace expansion on a different surface.
        fn effective_hessian(
            &self,
            pr: &PirlsResult,
        ) -> Result<(Array2<f64>, RidgePassport), EstimationError> {
            let base = pr.stabilized_hessian_transformed.clone();

            if base.cholesky(Side::Lower).is_ok() {
                return Ok((base, pr.ridge_passport));
            }

            Err(EstimationError::ModelIsIllConditioned {
                condition_number: f64::INFINITY,
            })
        }

        #[allow(dead_code)]
        pub(super) fn new<X>(
            y: ArrayView1<'a, f64>,
            x: X,
            weights: ArrayView1<'a, f64>,
            s_list: Vec<Array2<f64>>,
            p: usize,
            config: &'a RemlConfig,
            nullspace_dims: Option<Vec<usize>>,
        ) -> Result<Self, EstimationError>
        where
            X: Into<DesignMatrix>,
        {
            let zero_offset = Array1::<f64>::zeros(y.len());
            Self::new_with_offset(
                y,
                x,
                weights,
                zero_offset.view(),
                s_list,
                p,
                config,
                nullspace_dims,
            )
        }

        pub(super) fn new_with_offset<X>(
            y: ArrayView1<'a, f64>,
            x: X,
            weights: ArrayView1<'a, f64>,
            offset: ArrayView1<'_, f64>,
            s_list: Vec<Array2<f64>>,
            p: usize,
            config: &'a RemlConfig,
            nullspace_dims: Option<Vec<usize>>,
        ) -> Result<Self, EstimationError>
        where
            X: Into<DesignMatrix>,
        {
            // Pre-compute penalty square roots once
            let rs_list = compute_penalty_square_roots(&s_list)?;
            let x = x.into();

            let expected_len = s_list.len();
            let nullspace_dims = match nullspace_dims {
                Some(dims) => {
                    if dims.len() != expected_len {
                        return Err(EstimationError::InvalidInput(format!(
                            "nullspace_dims length {} does not match penalties {}",
                            dims.len(),
                            expected_len
                        )));
                    }
                    dims
                }
                None => vec![0; expected_len],
            };

            let penalty_count = rs_list.len();
            let total_rank: usize = rs_list.iter().map(|rk| rk.nrows()).sum();
            let workspace = RemlWorkspace::new(penalty_count, p, total_rank);

            let balanced_penalty_root = create_balanced_penalty_root(&s_list, p)?;
            let reparam_invariant = precompute_reparam_invariant(&rs_list, p)?;

            Ok(Self {
                y,
                x,
                weights,
                offset: offset.to_owned(),
                s_full_list: s_list,
                rs_list,
                balanced_penalty_root,
                reparam_invariant,
                p,
                config,
                nullspace_dims,
                cache: RwLock::new(HashMap::new()),
                faer_factor_cache: RwLock::new(HashMap::new()),
                eval_count: RwLock::new(0),
                last_cost: RwLock::new(f64::INFINITY),
                last_grad_norm: RwLock::new(f64::INFINITY),
                consecutive_cost_errors: RwLock::new(0),
                last_cost_error_msg: RwLock::new(None),
                current_eval_bundle: RwLock::new(None),
                cost_last: RwLock::new(None),
                cost_repeat: RwLock::new(0),
                cost_last_emit: RwLock::new(0),
                cost_eval_count: RwLock::new(0),
                raw_cond_snapshot: RwLock::new(f64::NAN),
                gaussian_cond_snapshot: RwLock::new(f64::NAN),
                workspace: Mutex::new(workspace),
                warm_start_beta: RwLock::new(None),
                warm_start_enabled: AtomicBool::new(true),
            })
        }

        /// Creates a sanitized cache key from rho values.
        /// Returns None if any component is NaN, in which case caching is skipped.
        /// Maps -0.0 to 0.0 to ensure consistency in caching.
        fn rho_key_sanitized(&self, rho: &Array1<f64>) -> Option<Vec<u64>> {
            let mut key = Vec::with_capacity(rho.len());
            for &v in rho.iter() {
                if v.is_nan() {
                    return None; // Don't cache NaN values
                }
                if v == 0.0 {
                    // This handles both +0.0 and -0.0
                    key.push(0.0f64.to_bits());
                } else {
                    key.push(v.to_bits());
                }
            }
            Some(key)
        }

        fn prepare_eval_bundle_with_key(
            &self,
            rho: &Array1<f64>,
            key: Option<Vec<u64>>,
        ) -> Result<EvalShared, EstimationError> {
            let pirls_result = self.execute_pirls_if_needed(rho)?;
            let (h_eff, ridge_passport) = self.effective_hessian(pirls_result.as_ref())?;

            // Spectral consistency threshold for eigenvalue truncation.
            //
            // Root-cause fix:
            // An absolute cutoff is scale-dependent and can misclassify near-null
            // modes when ||H|| varies by orders of magnitude. Use a relative rule
            // anchored to the dominant eigenvalue so pseudoinverse support and
            // log|H|_+ are stable across problem scales.
            const EIG_REL_THRESHOLD: f64 = 1e-10;
            const EIG_ABS_FLOOR: f64 = 1e-14;

            let dim = h_eff.nrows();

            // Compute spectral quantities from the same curvature used by inner PIRLS.
            // This path stays on H_eff for cost/gradient consistency.
            let h_total = h_eff.clone();
            let (eigvals, eigvecs) = h_total
                .eigh(Side::Lower)
                .map_err(|e| EstimationError::EigendecompositionFailed(e))?;
            let max_eig = eigvals.iter().copied().fold(0.0_f64, f64::max);
            let eig_threshold = (max_eig * EIG_REL_THRESHOLD).max(EIG_ABS_FLOOR);

            // Positive-part Hessian log-determinant convention:
            //   log|H|_+ = Σ_{λ_i(H) > τ} log λ_i(H),
            // where τ is a relative+absolute cutoff tied to spectrum scale.
            //
            // This avoids unstable rank flips from tiny signed eigenvalues and keeps
            // logdet/traces/pseudoinverse operations on the same effective subspace.
            let h_total_log_det: f64 = eigvals
                .iter()
                .filter(|&&v| v > eig_threshold)
                .map(|&v| v.ln())
                .sum();

            if !h_total_log_det.is_finite() {
                return Err(EstimationError::ModelIsIllConditioned {
                    condition_number: f64::INFINITY,
                });
            }

            // Build factor W for the Moore-Penrose pseudoinverse on the kept subspace:
            //   H_+^† = U_+ diag(1/λ_+) U_+ᵀ = W Wᵀ,
            //   W := U_+ diag(1/sqrt(λ_+)).
            //
            // Later trace terms use this representation directly, e.g.
            //   tr(H_+^† S_k) = ||R_k W||_F^2
            // without materializing H_+^† as a dense matrix.
            let valid_indices: Vec<usize> = eigvals
                .iter()
                .enumerate()
                .filter_map(|(i, &v)| if v > eig_threshold { Some(i) } else { None })
                .collect();

            let valid_count = valid_indices.len();
            let mut w = Array2::<f64>::zeros((dim, valid_count));

            for (w_col_idx, &eig_idx) in valid_indices.iter().enumerate() {
                let val = eigvals[eig_idx];
                let scale = 1.0 / val.sqrt();
                let u_col = eigvecs.column(eig_idx);

                let mut w_col = w.column_mut(w_col_idx);
                Zip::from(&mut w_col)
                    .and(&u_col)
                    .for_each(|w_elem, &u_elem| {
                        *w_elem = u_elem * scale;
                    });
            }

            Ok(EvalShared {
                key,
                pirls_result,
                h_eff: Arc::new(h_eff),
                ridge_passport,
                h_total: Arc::new(h_total),
                h_pos_factor_w: Arc::new(w),
                h_total_log_det,
            })
        }

        fn obtain_eval_bundle(&self, rho: &Array1<f64>) -> Result<EvalShared, EstimationError> {
            let key = self.rho_key_sanitized(rho);
            if let Some(existing) = self.current_eval_bundle.read().unwrap().as_ref()
                && existing.matches(&key)
            {
                return Ok(existing.clone());
            }
            let bundle = self.prepare_eval_bundle_with_key(rho, key)?;
            *self.current_eval_bundle.write().unwrap() = Some(bundle.clone());
            Ok(bundle)
        }

        pub(super) fn last_ridge_used(&self) -> Option<f64> {
            self.current_eval_bundle
                .read()
                .unwrap()
                .as_ref()
                .map(|bundle| bundle.ridge_passport.delta)
        }

        /// Calculate effective degrees of freedom (EDF) using a consistent approach
        /// for both cost and gradient calculations, ensuring identical values.
        ///
        /// # Arguments
        /// * `pr` - PIRLS result containing the penalty matrices
        /// * `lambdas` - Smoothing parameters (lambda values)
        /// * `h_eff` - Effective Hessian matrix
        ///
        /// # Returns
        /// * Effective degrees of freedom value
        fn edf_from_h_and_rk(
            &self,
            pr: &PirlsResult,
            lambdas: ArrayView1<'_, f64>,
            h_eff: &Array2<f64>,
        ) -> Result<f64, EstimationError> {
            // Why caching by ρ is sound:
            // The effective degrees of freedom (EDF) calculation is one of only two places where
            // we ask for a Faer factorization through `get_faer_factor`.  The cache inside that
            // helper uses only the vector of log smoothing parameters (ρ) as the key.  At first
            // glance that can look risky—two different Hessians with the same ρ might appear to be
            // conflated.  The surrounding call graph prevents that situation:
            //   • Identity / Gaussian models call `edf_from_h_and_rk` with the stabilized Hessian
            //     `pirls_result.stabilized_hessian_transformed`.
            //   • Non-Gaussian (logit / LAML) models call it with the effective / ridged Hessian
            //     returned by `effective_hessian(pr)`.
            // Within a given `RemlState` we never switch between those two flavours—the state is
            // constructed for a single link function, so the cost/gradient pathways stay aligned.
            // Because of that design, a given ρ vector corresponds to exactly one Hessian type in
            // practice, and the cache cannot hand back a factorization of an unintended matrix.

            // Prefer an un-ridged factorization when the stabilized Hessian is already PD.
            // Only fall back to the RidgePlanner path if direct factorization fails.
            let rho_like = lambdas.mapv(|lam| lam.ln());
            let factor = {
                let h_view = FaerArrayView::new(h_eff);
                if let Ok(f) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
                    Arc::new(FaerFactor::Llt(f))
                } else if let Ok(f) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
                    Arc::new(FaerFactor::Ldlt(f))
                } else {
                    self.get_faer_factor(&rho_like, h_eff)
                }
            };

            // Use the single λ-weighted penalty root E for S_λ = Eᵀ E to compute
            // trace(H⁻¹ S_λ) = ⟨H⁻¹ Eᵀ, Eᵀ⟩_F directly.
            let e_t = pr.reparam_result.e_transformed.t().to_owned(); // (p × rank_total)
            let e_view = FaerArrayView::new(&e_t);
            let x = factor.solve(e_view.as_ref());
            let trace_h_inv_s_lambda = faer_frob_inner(x.as_ref(), e_view.as_ref());

            // Calculate EDF as p - trace, clamped to the penalty nullspace dimension
            let p = pr.beta_transformed.len() as f64;
            let rank_s = pr.reparam_result.e_transformed.nrows() as f64;
            let mp = (p - rank_s).max(0.0);
            let edf = (p - trace_h_inv_s_lambda).clamp(mp, p);

            Ok(edf)
        }

        fn update_warm_start_from(&self, pr: &PirlsResult) {
            if !self.warm_start_enabled.load(Ordering::Relaxed) {
                return;
            }
            match pr.status {
                pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                    let beta_original = pr.reparam_result.qs.dot(pr.beta_transformed.as_ref());
                    self.warm_start_beta
                        .write()
                        .unwrap()
                        .replace(Coefficients::new(beta_original));
                }
                _ => {
                    self.warm_start_beta.write().unwrap().take();
                }
            }
        }

        /// Clear warm-start state. Used in tests to ensure consistent starting points
        /// when comparing different gradient computation paths.
        #[cfg(test)]
        #[allow(dead_code)]
        pub fn clear_warm_start(&self) {
            self.warm_start_beta.write().unwrap().take();
            self.current_eval_bundle.write().unwrap().take();
        }

        /// Returns the per-penalty square-root matrices in the transformed coefficient basis
        /// without any λ weighting. Each returned R_k satisfies S_k = R_kᵀ R_k in that basis.
        /// Using these avoids accidental double counting of λ when forming derivatives.
        ///
        /// # Arguments
        /// * `pr` - The PIRLS result with the transformation matrix Qs
        ///
        /// # Returns
        fn factorize_faer(&self, h: &Array2<f64>) -> FaerFactor {
            let mut planner = RidgePlanner::new(h);
            loop {
                let ridge = planner.ridge();
                if ridge > 0.0 {
                    let regularized = add_ridge(h, ridge);
                    let view = FaerArrayView::new(&regularized);
                    if let Ok(f) = FaerLlt::new(view.as_ref(), Side::Lower) {
                        return FaerFactor::Llt(f);
                    }
                    if let Ok(f) = FaerLdlt::new(view.as_ref(), Side::Lower) {
                        return FaerFactor::Ldlt(f);
                    }
                    if planner.attempts() >= MAX_FACTORIZATION_ATTEMPTS {
                        let f = FaerLblt::new(view.as_ref(), Side::Lower);
                        return FaerFactor::Lblt(f);
                    }
                } else {
                    let h_view = FaerArrayView::new(h);
                    if let Ok(f) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
                        return FaerFactor::Llt(f);
                    }
                    if let Ok(f) = FaerLdlt::new(h_view.as_ref(), Side::Lower) {
                        return FaerFactor::Ldlt(f);
                    }
                }
                planner.bump_with_matrix(h);
            }
        }

        fn get_faer_factor(&self, rho: &Array1<f64>, h: &Array2<f64>) -> Arc<FaerFactor> {
            // Cache strategy: ρ alone is the key.
            // The cache deliberately ignores which Hessian matrix we are factoring.  Today this is
            // sound because every caller obeys a single rule:
            //   • Identity/Gaussian REML cost & gradient only ever request factors of the
            //     stabilized Hessian.
            //   • Non-Gaussian (logit/LAML) cost and gradient request factors of the effective/ridged Hessian.
            // Consequently each ρ corresponds to exactly one matrix within the lifetime of a
            // `RemlState`, so returning the cached factorization is correct.
            // This design is still brittle: adding a new code path that calls `get_faer_factor`
            // with a different H for the same ρ would silently reuse the wrong factor.  If such a
            // path ever appears, extend the key (for example by tagging the Hessian variant) or
            // split the cache.  The current key maximizes cache
            // hits across repeated EDF/gradient evaluations for the same smoothing parameters.
            let key_opt = self.rho_key_sanitized(rho);
            if let Some(key) = &key_opt
                && let Some(f) = self.faer_factor_cache.read().unwrap().get(key)
            {
                return Arc::clone(f);
            }
            let fact = Arc::new(self.factorize_faer(h));

            if let Some(key) = key_opt {
                let mut cache = self.faer_factor_cache.write().unwrap();
                if cache.len() > 64 {
                    cache.clear();
                }
                cache.insert(key, Arc::clone(&fact));
            }
            fact
        }

        /// Numerical gradient of the penalized log-likelihood part w.r.t. rho via central differences.
        /// Returns g_pll where g_pll[k] = - d/d rho_k penalised_ll(rho), suitable for COST gradient assembly.
        #[cfg(test)]
        #[allow(dead_code)]
        fn numeric_penalised_ll_grad(
            &self,
            rho: &Array1<f64>,
        ) -> Result<Array1<f64>, EstimationError> {
            let mut workspace = self.workspace.lock().unwrap();
            self.numeric_penalised_ll_grad_with_workspace(rho, &mut workspace)
        }

        fn numeric_penalised_ll_grad_with_workspace(
            &self,
            rho: &Array1<f64>,
            workspace: &mut RemlWorkspace,
        ) -> Result<Array1<f64>, EstimationError> {
            let len = rho.len();
            if len == 0 {
                return Ok(Array1::zeros(0));
            }

            let x = &self.x;
            let offset_view = self.offset.view();
            let y = self.y;
            let weights = self.weights;
            let rs_list = &self.rs_list;
            let p_dim = self.p;
            let config = self.config;
            let firth_bias = config.firth_bias_reduction;
            let link_is_logit = matches!(config.link_function(), LinkFunction::Logit);
            let balanced_root = &self.balanced_penalty_root;
            let reparam_invariant = &self.reparam_invariant;

            // Capture the current best beta to warm-start the gradient probes.
            // This is crucial for stability: if we start from zero, P-IRLS might converge
            // to a different local optimum (or stall differently) than the main cost evaluation,
            // creating huge phantom gradients that violate the envelope theorem.
            let warm_start_initial = if self.warm_start_enabled.load(Ordering::Relaxed) {
                self.warm_start_beta.read().unwrap().clone()
            } else {
                None
            };

            // Run a fresh PIRLS solve for each perturbed smoothing vector.  We avoid the
            // `execute_pirls_if_needed` cache here because these evaluations happen in parallel
            // and never reuse the same ρ, so the cache would not help and would require
            // synchronization across threads.
            let evaluate_penalised_ll = |rho_vec: &Array1<f64>| -> Result<f64, EstimationError> {
                let (pirls_result, _) = pirls::fit_model_for_fixed_rho_matrix(
                    LogSmoothingParamsView::new(rho_vec.view()),
                    x,
                    offset_view,
                    y,
                    weights,
                    rs_list,
                    Some(balanced_root),
                    Some(reparam_invariant),
                    p_dim,
                    &config.as_pirls_config(),
                    warm_start_initial.as_ref(),
                    None, // No SE for base model
                )?;

                match pirls_result.status {
                    pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                        let penalty = pirls_result.stable_penalty_term;
                        let mut penalised = -0.5 * pirls_result.deviance - 0.5 * penalty;
                        // Include Firth log-det term in LAML for consistency with inner PIRLS
                        if firth_bias && link_is_logit {
                            if let Some(firth_log_det) = pirls_result.firth_log_det {
                                penalised += firth_log_det; // Jeffreys prior contribution
                            }
                        }
                        Ok(penalised)
                    }
                    pirls::PirlsStatus::Unstable => {
                        Err(EstimationError::PerfectSeparationDetected {
                            iteration: pirls_result.iteration,
                            max_abs_eta: pirls_result.max_abs_eta,
                        })
                    }
                    pirls::PirlsStatus::MaxIterationsReached => {
                        if pirls_result.last_gradient_norm > 1.0 {
                            Err(EstimationError::PirlsDidNotConverge {
                                max_iterations: pirls_result.iteration,
                                last_change: pirls_result.last_gradient_norm,
                            })
                        } else {
                            let penalty = pirls_result.stable_penalty_term;
                            let mut penalised = -0.5 * pirls_result.deviance - 0.5 * penalty;
                            // Include Firth log-det term in LAML for consistency with inner PIRLS
                            if firth_bias && link_is_logit {
                                if let Some(firth_log_det) = pirls_result.firth_log_det {
                                    penalised += firth_log_det; // Jeffreys prior contribution
                                }
                            }
                            Ok(penalised)
                        }
                    }
                }
            };

            let grad_values = (0..len)
                .into_par_iter()
                .map(|k| -> Result<f64, EstimationError> {
                    let h_rel = 1e-4_f64 * (1.0 + rho[k].abs());
                    let h_abs = 1e-5_f64;
                    let h = h_rel.max(h_abs);

                    let mut rho_plus = rho.clone();
                    rho_plus[k] += 0.5 * h;
                    let mut rho_minus = rho.clone();
                    rho_minus[k] -= 0.5 * h;

                    let fp = evaluate_penalised_ll(&rho_plus)?;
                    let fm = evaluate_penalised_ll(&rho_minus)?;
                    Ok(-(fp - fm) / h)
                })
                .collect::<Result<Vec<_>, _>>()?;

            let grad_array = Array1::from_vec(grad_values);
            let mut g_view = workspace.grad_secondary.slice_mut(s![..len]);
            g_view.assign(&grad_array);

            Ok(grad_array)
        }

        /// Compute 0.5 * log|H_eff(rho)| using the SAME stabilized Hessian and logdet path as compute_cost.
        fn half_logh_at(&self, rho: &Array1<f64>) -> Result<f64, EstimationError> {
            let pr = self.execute_pirls_if_needed(rho)?;
            let (h_eff, _) = self.effective_hessian(&pr)?;
            let chol = h_eff.clone().cholesky(Side::Lower).map_err(|_| {
                let min_eig = h_eff
                    .clone()
                    .eigh(Side::Lower)
                    .ok()
                    .and_then(|(eigs, _)| eigs.iter().cloned().reduce(f64::min))
                    .unwrap_or(f64::NAN);
                EstimationError::HessianNotPositiveDefinite {
                    min_eigenvalue: min_eig,
                }
            })?;
            let log_det_h = 2.0 * chol.diag().mapv(f64::ln).sum();
            Ok(0.5 * log_det_h)
        }

        /// Numerical gradient of 0.5 * log|H_eff(rho)| with respect to rho via central differences.
        fn numeric_half_logh_grad_with_workspace(
            &self,
            rho: &Array1<f64>,
            workspace: &mut RemlWorkspace,
        ) -> Result<Array1<f64>, EstimationError> {
            let len = rho.len();
            if len == 0 {
                return Ok(Array1::zeros(0));
            }

            let mut g_view = workspace.grad_primary.slice_mut(s![..len]);
            g_view.fill(0.0);

            for k in 0..len {
                let h_rel = 1e-4_f64 * (1.0 + rho[k].abs());
                let h_abs = 1e-5_f64;
                let h = h_rel.max(h_abs);

                workspace.rho_plus.assign(rho);
                workspace.rho_plus[k] += 0.5 * h;
                workspace.rho_minus.assign(rho);
                workspace.rho_minus[k] -= 0.5 * h;

                let fp = self.half_logh_at(&workspace.rho_plus)?;
                let fm = self.half_logh_at(&workspace.rho_minus)?;
                g_view[k] = (fp - fm) / h;
            }

            Ok(g_view.to_owned())
        }

        const MIN_DMU_DETA: f64 = 1e-6;

        // Accessor methods for private fields
        pub(super) fn x(&self) -> &DesignMatrix {
            &self.x
        }

        #[allow(dead_code)]
        pub(super) fn y(&self) -> ArrayView1<'a, f64> {
            self.y
        }

        #[allow(dead_code)]
        pub(super) fn rs_list_ref(&self) -> &Vec<Array2<f64>> {
            &self.rs_list
        }

        pub(super) fn balanced_penalty_root(&self) -> &Array2<f64> {
            &self.balanced_penalty_root
        }

        pub(super) fn weights(&self) -> ArrayView1<'a, f64> {
            self.weights
        }

        #[allow(dead_code)]
        pub(super) fn offset(&self) -> ArrayView1<'_, f64> {
            self.offset.view()
        }

        // Expose error tracking state to parent module
        pub(super) fn consecutive_cost_error_count(&self) -> usize {
            *self.consecutive_cost_errors.read().unwrap()
        }

        pub(super) fn last_cost_error_string(&self) -> Option<String> {
            self.last_cost_error_msg.read().unwrap().clone()
        }

        /// Runs the inner P-IRLS loop, caching the result.
        fn execute_pirls_if_needed(
            &self,
            rho: &Array1<f64>,
        ) -> Result<Arc<PirlsResult>, EstimationError> {
            // Use sanitized key to handle NaN and -0.0 vs 0.0 issues
            let key_opt = self.rho_key_sanitized(rho);
            if let Some(key) = &key_opt
                && let Some(cached) = {
                    let cache_ref = self.cache.read().unwrap();
                    cache_ref.get(key).cloned()
                }
            {
                if self.warm_start_enabled.load(Ordering::Relaxed) {
                    self.update_warm_start_from(cached.as_ref());
                }
                return Ok(cached);
            }

            // Run P-IRLS with original matrices to perform fresh reparameterization
            // The returned result will include the transformation matrix qs
            let pirls_result = {
                let warm_start_holder = self.warm_start_beta.read().unwrap();
                let warm_start_ref = if self.warm_start_enabled.load(Ordering::Relaxed) {
                    warm_start_holder.as_ref()
                } else {
                    None
                };
                pirls::fit_model_for_fixed_rho_matrix(
                    LogSmoothingParamsView::new(rho.view()),
                    &self.x,
                    self.offset.view(),
                    self.y,
                    self.weights,
                    &self.rs_list,
                    Some(&self.balanced_penalty_root),
                    Some(&self.reparam_invariant),
                    self.p,
                    &self.config.as_pirls_config(),
                    warm_start_ref,
                    None, // No SE for base model
                )
            };

            if let Err(e) = &pirls_result {
                println!("[GAM COST]   -> P-IRLS INNER LOOP FAILED. Error: {e:?}");
                if self.warm_start_enabled.load(Ordering::Relaxed) {
                    self.warm_start_beta.write().unwrap().take();
                }
            }

            let (pirls_result, _) = pirls_result?; // Propagate error if it occurred
            let pirls_result = Arc::new(pirls_result);

            // Check the status returned by the P-IRLS routine.
            match pirls_result.status {
                pirls::PirlsStatus::Converged | pirls::PirlsStatus::StalledAtValidMinimum => {
                    self.update_warm_start_from(pirls_result.as_ref());
                    // This is a successful fit. Cache only if key is valid (not NaN).
                    if let Some(key) = key_opt {
                        self.cache
                            .write()
                            .unwrap()
                            .insert(key, Arc::clone(&pirls_result));
                    }
                    Ok(pirls_result)
                }
                pirls::PirlsStatus::Unstable => {
                    if self.warm_start_enabled.load(Ordering::Relaxed) {
                        self.warm_start_beta.write().unwrap().take();
                    }
                    // The fit was unstable. This is where we throw our specific, user-friendly error.
                    // Pass the diagnostic info into the error
                    Err(EstimationError::PerfectSeparationDetected {
                        iteration: pirls_result.iteration,
                        max_abs_eta: pirls_result.max_abs_eta,
                    })
                }
                pirls::PirlsStatus::MaxIterationsReached => {
                    if self.warm_start_enabled.load(Ordering::Relaxed) {
                        self.warm_start_beta.write().unwrap().take();
                    }
                    if pirls_result.last_gradient_norm > 1.0 {
                        // The fit timed out and gradient is large.
                        log::error!(
                            "P-IRLS failed convergence check: gradient norm {} > 1.0 (iter {})",
                            pirls_result.last_gradient_norm,
                            pirls_result.iteration
                        );
                        Err(EstimationError::PirlsDidNotConverge {
                            max_iterations: pirls_result.iteration,
                            last_change: pirls_result.last_gradient_norm,
                        })
                    } else {
                        // Gradient is acceptable, treat as converged but with warning if needed
                        log::warn!(
                            "P-IRLS reached max iterations but gradient norm {:.3e} is acceptable.",
                            pirls_result.last_gradient_norm
                        );
                        Ok(pirls_result)
                    }
                }
            }
        }
    }
    impl<'a> RemlState<'a> {
        /// Compute the objective function for BFGS optimization.
        ///
        /// FULL OBJECTIVE REFERENCE
        /// ------------------------
        /// This function returns the scalar outer cost minimized over ρ.
        ///
        /// Non-Gaussian branch (negative LAML form used by optimizer):
        ///   V_LAML(ρ) =
        ///      -ℓ(β̂(ρ))
        ///      + 0.5 β̂(ρ)ᵀ S(ρ) β̂(ρ)
        ///      + 0.5 log|H(ρ)|
        ///      - 0.5 log|S(ρ)|_+
        ///      + const
        ///
        /// where:
        ///   S(ρ) = Σ_k exp(ρ_k) S_k + δI
        ///   H(ρ) = -∇²ℓ(β̂(ρ)) + S(ρ)
        ///
        /// Gaussian identity-link REML branch:
        ///   V_REML(ρ, φ) =
        ///      D_p(ρ)/(2φ)
        ///      + (n_r/2) log φ
        ///      + 0.5 log|H(ρ)|
        ///      - 0.5 log|S(ρ)|_+
        ///      + const
        ///
        /// with profiled φ:
        ///   φ̂(ρ) = D_p(ρ)/n_r
        ///   V_REML,prof(ρ) =
        ///      (n_r/2) log D_p(ρ)
        ///      + 0.5 log|H(ρ)|
        ///      - 0.5 log|S(ρ)|_+
        ///      + const.
        ///
        /// Consistency rule enforced throughout:
        ///   The same stabilized matrices/factorizations are used for
        ///   objective and gradient/Hessian terms. Mixing different H/S variants
        ///   causes deterministic gradient mismatch and unstable outer optimization.
        ///
        /// Determinant conventions:
        ///   - log|H| may use positive-part/stabilized spectrum conventions when needed.
        ///   - log|S|_+ follows fixed-rank pseudo-determinant conventions in the
        ///     transformed penalty basis, optionally including ridge policy.
        /// These conventions are mirrored in gradient code via corresponding trace terms.
        pub fn compute_cost(&self, p: &Array1<f64>) -> Result<f64, EstimationError> {
            let cost_call_idx = {
                let mut calls = self.cost_eval_count.write().unwrap();
                *calls += 1;
                *calls
            };
            let bundle = match self.obtain_eval_bundle(p) {
                Ok(bundle) => bundle,
                Err(EstimationError::ModelIsIllConditioned { .. }) => {
                    self.current_eval_bundle.write().unwrap().take();
                    // Inner linear algebra says "too singular" — treat as barrier.
                    log::warn!(
                        "P-IRLS flagged ill-conditioning for current rho; returning +inf cost to retreat."
                    );
                    // Diagnostics: which rho are at bounds
                    let at_lower: Vec<usize> = p
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &v)| {
                            if v <= -RHO_BOUND + 1e-8 {
                                Some(i)
                            } else {
                                None
                            }
                        })
                        .collect();
                    let at_upper: Vec<usize> = p
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &v)| if v >= RHO_BOUND - 1e-8 { Some(i) } else { None })
                        .collect();
                    if !(at_lower.is_empty() && at_upper.is_empty()) {
                        eprintln!(
                            "[Diag] rho bounds: lower={:?} upper={:?}",
                            at_lower, at_upper
                        );
                    }
                    return Ok(f64::INFINITY);
                }
                Err(e) => {
                    self.current_eval_bundle.write().unwrap().take();
                    // Other errors still bubble up
                    // Provide bounds diagnostics here too
                    let at_lower: Vec<usize> = p
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &v)| {
                            if v <= -RHO_BOUND + 1e-8 {
                                Some(i)
                            } else {
                                None
                            }
                        })
                        .collect();
                    let at_upper: Vec<usize> = p
                        .iter()
                        .enumerate()
                        .filter_map(|(i, &v)| if v >= RHO_BOUND - 1e-8 { Some(i) } else { None })
                        .collect();
                    if !(at_lower.is_empty() && at_upper.is_empty()) {
                        eprintln!(
                            "[Diag] rho bounds: lower={:?} upper={:?}",
                            at_lower, at_upper
                        );
                    }
                    return Err(e);
                }
            };
            let pirls_result = bundle.pirls_result.as_ref();
            let h_eff = bundle.h_eff.as_ref();
            let ridge_used = bundle.ridge_passport.delta;

            let lambdas = p.mapv(f64::exp);

            // Sanity check: penalty dimension consistency across lambdas, R_k, and det1.
            if !p.is_empty() {
                let k_lambda = p.len();
                let k_r = pirls_result.reparam_result.rs_transformed.len();
                let k_d = pirls_result.reparam_result.det1.len();
                if !(k_lambda == k_r && k_r == k_d) {
                    return Err(EstimationError::LayoutError(format!(
                        "Penalty dimension mismatch: lambdas={}, R={}, det1={}",
                        k_lambda, k_r, k_d
                    )));
                }
                if self.nullspace_dims.len() != k_lambda {
                    return Err(EstimationError::LayoutError(format!(
                        "Nullspace dimension mismatch: expected {} entries, got {}",
                        k_lambda,
                        self.nullspace_dims.len()
                    )));
                }
            }

            // Don't barrier on non-PD; we'll stabilize and continue like mgcv
            // Only check eigenvalues if we needed to add a ridge
            const MIN_ACCEPTABLE_HESSIAN_EIGENVALUE: f64 = 1e-12;
            let want_hot_diag = self.should_compute_hot_diagnostics(cost_call_idx);
            if ridge_used > 0.0
                && let Ok((eigs, _)) = pirls_result.penalized_hessian_transformed.eigh(Side::Lower)
                && let Some(min_eig) = eigs.iter().cloned().reduce(f64::min)
            {
                if should_emit_h_min_eig_diag(min_eig) {
                    eprintln!(
                        "[Diag] H min_eig={:.3e} (ridge={:.3e})",
                        min_eig, ridge_used
                    );
                }

                if min_eig <= 0.0 {
                    log::warn!(
                        "Penalized Hessian not PD (min eig <= 0) before stabilization; proceeding with ridge {:.3e}.",
                        ridge_used
                    );
                }

                if want_hot_diag
                    && (!min_eig.is_finite() || min_eig <= MIN_ACCEPTABLE_HESSIAN_EIGENVALUE)
                {
                    let condition_number =
                        calculate_condition_number(&pirls_result.penalized_hessian_transformed)
                            .ok()
                            .unwrap_or(f64::INFINITY);

                    log::warn!(
                        "Penalized Hessian extremely ill-conditioned (cond={:.3e}); continuing with stabilized Hessian.",
                        condition_number
                    );
                }
            }
            // Use stable penalty calculation - no need to reconstruct matrices
            // The penalty term is already calculated stably in the P-IRLS loop

            match self.config.link_function() {
                LinkFunction::Identity => {
                    // From Wood (2017), Chapter 6, Eq. 6.24:
                    // V_r(λ) = D_p/(2φ) + (r/2φ) + ½log|X'X/φ + S_λ/φ| - ½log|S_λ/φ|_+
                    // where D_p = ||y - Xβ̂||² + β̂'S_λβ̂ is the PENALIZED deviance
                    //
                    // With profiled dispersion φ̂ = D_p/(n-M_p), this becomes:
                    //   V_REML(ρ) =
                    //     D_p/(2φ̂)
                    //   + 0.5 log|H|
                    //   - 0.5 log|S|_+
                    //   + ((n-M_p)/2) log(2πφ̂),
                    // where H = XᵀW0X + S(ρ), S(ρ)=Σ_k exp(ρ_k) S_k + δI.
                    //
                    // Because Gaussian identity has c=d=0, there is no third/fourth derivative
                    // correction in H_k: ∂H/∂ρ_k = S_k^ρ exactly.

                    // Check condition number with improved thresholds per Wood (2011)
                    const MAX_CONDITION_NUMBER: f64 = 1e12;
                    if want_hot_diag {
                        let cond = pirls_result
                            .penalized_hessian_transformed
                            .eigh(Side::Lower)
                            .ok()
                            .map(|(evals, _)| {
                                let max_ev = evals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                                let min_ev = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                                if min_ev <= 1e-12 {
                                    f64::INFINITY
                                } else {
                                    max_ev / min_ev
                                }
                            })
                            .unwrap_or(f64::NAN);
                        *self.gaussian_cond_snapshot.write().unwrap() = cond;
                    }
                    let condition_number = *self.gaussian_cond_snapshot.read().unwrap();
                    if condition_number.is_finite() {
                        if condition_number > MAX_CONDITION_NUMBER {
                            log::warn!(
                                "Penalized Hessian very ill-conditioned (cond={:.2e}); proceeding despite poor conditioning.",
                                condition_number
                            );
                        } else if condition_number > 1e8 {
                            log::warn!(
                                "Penalized Hessian is ill-conditioned but proceeding: condition number = {condition_number:.2e}"
                            );
                        }
                    }

                    // STRATEGIC DESIGN DECISION: Use unweighted sample count for mgcv parity
                    // In standard WLS theory, one might use sum(weights) as effective sample size.
                    // However, mgcv deliberately uses the unweighted count 'n.true' in gam.fit3.
                    let n = self.y.len() as f64;
                    // Number of coefficients (transformed basis)

                    // Calculate PENALIZED deviance D_p = ||y - Xβ̂||² + β̂'S_λβ̂
                    let rss = pirls_result.deviance; // Unpenalized ||y - μ||²
                    // Use stable penalty term calculated in P-IRLS
                    let penalty = pirls_result.stable_penalty_term;

                    let dp = rss + penalty;

                    // Calculate EDF = p - tr((X'X + S_λ)⁻¹S_λ)
                    // Work directly in the transformed basis for efficiency and numerical stability
                    // This avoids transforming matrices back to the original basis unnecessarily
                    // Penalty roots are available in reparam_result if needed

                    // Nullspace dimension M_p is constant with respect to ρ.  Use it to profile φ
                    // following the standard REML identity φ = D_p / (n - M_p).
                    let penalty_rank = pirls_result.reparam_result.e_transformed.nrows();
                    let mp = self.p.saturating_sub(penalty_rank) as f64;

                    // EDF diagnostics are expensive; compute only when diagnostics are enabled.
                    if want_hot_diag {
                        let edf = self.edf_from_h_and_rk(pirls_result, lambdas.view(), h_eff)?;
                        log::debug!("[Diag] EDF total={:.3}", edf);
                        if n - edf < 1.0 {
                            log::warn!("Effective DoF exceeds samples; model may be overfit.");
                        }
                    }

                    let denom = (n - mp).max(LAML_RIDGE);
                    let (dp_c, _) = smooth_floor_dp(dp);
                    if dp < DP_FLOOR {
                        log::warn!(
                            "Penalized deviance {:.3e} fell below DP_FLOOR; clamping to maintain REML stability.",
                            dp
                        );
                    }
                    let phi = dp_c / denom;

                    // log |H| = log |X'X + S_λ + ridge I| using the single effective
                    // Hessian shared with the gradient. Ridge is already baked into h_eff.
                    //
                    // This is the same stabilized H used in compute_gradient;
                    // otherwise the chain-rule pieces and determinant pieces are taken on
                    // different objective surfaces and the optimizer sees inconsistent derivatives.
                    let h_for_det = h_eff.clone();

                    let chol = h_for_det.cholesky(Side::Lower).map_err(|_| {
                        let min_eig = h_eff
                            .clone()
                            .eigh(Side::Lower)
                            .ok()
                            .and_then(|(eigs, _)| eigs.iter().cloned().reduce(f64::min))
                            .unwrap_or(f64::NAN);
                        EstimationError::HessianNotPositiveDefinite {
                            min_eigenvalue: min_eig,
                        }
                    })?;
                    let log_det_h = 2.0 * chol.diag().mapv(f64::ln).sum();

                    // log |S_λ + ridge I|_+ (pseudo-determinant) to match the
                    // stabilized penalty used by PIRLS.
                    //
                    // Fixed-rank rule: unpenalized/null directions do not contribute to the
                    // pseudo-logdet. This keeps the objective continuous in ρ when S is singular
                    // (or near-singular before ridge augmentation).
                    let ridge_passport = pirls_result.ridge_passport;
                    let log_det_s_plus = Self::log_det_s_with_ridge(
                        &pirls_result.reparam_result.s_transformed,
                        ridge_passport,
                        pirls_result.reparam_result.log_det,
                    )?;

                    // Standard REML expression from Wood (2017), Section 6.5.1
                    // V = (n/2)log(2πσ²) + D_p/(2σ²) + ½log|H| - ½log|S_λ|_+ + (M_p-1)/2 log(2πσ²)
                    // Simplifying: V = D_p/(2φ) + ½log|H| - ½log|S_λ|_+ + ((n-M_p)/2) log(2πφ)
                    let reml = dp_c / (2.0 * phi)
                        + 0.5 * (log_det_h - log_det_s_plus)
                        + ((n - mp) / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();

                    let prior_cost = self.compute_soft_prior_cost(p);

                    Ok(reml + prior_cost)
                }
                _ => {
                    // For non-Gaussian GLMs, use the LAML approximation
                    // Note: Deviance = -2 * log-likelihood + C. So -0.5 * Deviance = log-likelihood - C/2.
                    // Use stable penalty term calculated in P-IRLS
                    let mut penalised_ll =
                        -0.5 * pirls_result.deviance - 0.5 * pirls_result.stable_penalty_term;

                    let ridge_passport = pirls_result.ridge_passport;
                    // Include Firth log-det term in LAML for consistency with inner PIRLS
                    if self.config.firth_bias_reduction
                        && matches!(self.config.link_function(), LinkFunction::Logit)
                    {
                        if let Some(firth_log_det) = pirls_result.firth_log_det {
                            penalised_ll += firth_log_det; // Jeffreys prior contribution
                        }
                    }

                    // Use the stabilized log|Sλ|_+ from the reparameterization (consistent with gradient)
                    let log_det_s = Self::log_det_s_with_ridge(
                        &pirls_result.reparam_result.s_transformed,
                        ridge_passport,
                        pirls_result.reparam_result.log_det,
                    )?;

                    // Log-determinant of the effective Hessian.
                    // HESSIAN PASSPORT: Use the pre-computed h_total and its factorization
                    // from the bundle to ensure exact consistency with gradient computation.
                    // For Firth: h_total = h_eff - h_phi (computed in prepare_eval_bundle)
                    // For non-Firth: h_total = h_eff
                    //
                    // LAML objective:
                    //   V_LAML(ρ) =
                    //      -ℓ(β̂) + 0.5 β̂ᵀSβ̂
                    //    - 0.5 log|S|_+
                    //    + 0.5 log|H|
                    //    + const.
                    //
                    // For non-Gaussian families, H depends on ρ both directly through S and
                    // indirectly through β̂(ρ), which induces the dH/dρ_k third-derivative term in
                    // the exact gradient path (documented in compute_gradient).
                    let log_det_h = bundle.h_total_log_det;

                    // Mp is null space dimension (number of unpenalized coefficients)
                    // For logit, scale parameter is typically fixed at 1.0, but include for completeness
                    let phi = 1.0; // Logit family typically has dispersion parameter = 1

                    // Compute null space dimension using the TRANSFORMED, STABLE basis
                    // Use the rank of the lambda-weighted transformed penalty root (e_transformed)
                    // to determine M_p with the transformed penalty basis.
                    let penalty_rank = pirls_result.reparam_result.e_transformed.nrows();
                    let mp = self.p.saturating_sub(penalty_rank) as f64;

                    let laml = penalised_ll + 0.5 * log_det_s - 0.5 * log_det_h
                        + (mp / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();

                    // Diagnostics below are expensive and not needed for objective value.
                    let (edf, trace_h_inv_s_lambda, stab_cond) = if want_hot_diag {
                        let p_eff = pirls_result.beta_transformed.len() as f64;
                        let edf = self.edf_from_h_and_rk(pirls_result, lambdas.view(), h_eff)?;
                        let trace_h_inv_s_lambda = (p_eff - edf).max(0.0);
                        let stab_cond = pirls_result
                            .penalized_hessian_transformed
                            .eigh(Side::Lower)
                            .ok()
                            .map(|(evals, _)| {
                                let min = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                                let max = evals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                                max / min.max(1e-12)
                            })
                            .unwrap_or(f64::NAN);
                        (edf, trace_h_inv_s_lambda, stab_cond)
                    } else {
                        (f64::NAN, f64::NAN, f64::NAN)
                    };

                    // Raw-condition diagnostics are rate-limited in this loop.
                    // We only refresh occasionally, and keep the last snapshot otherwise.
                    let raw_cond = if matches!(self.x(), DesignMatrix::Dense(_)) && want_hot_diag {
                        let x_orig_arc = self.x().to_dense_arc();
                        let x_orig = x_orig_arc.as_ref();
                        let w_orig = self.weights();
                        let sqrt_w = w_orig.mapv(|w| w.max(0.0).sqrt());
                        let wx = x_orig * &sqrt_w.insert_axis(Axis(1));
                        let mut h_raw = fast_ata(&wx);
                        for (k, &lambda) in lambdas.iter().enumerate() {
                            let s_k = &self.s_full_list[k];
                            if lambda != 0.0 {
                                h_raw.scaled_add(lambda, s_k);
                            }
                        }
                        let raw = h_raw
                            .eigh(Side::Lower)
                            .ok()
                            .map(|(evals, _)| {
                                let min = evals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                                let max = evals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                                max / min.max(1e-12)
                            })
                            .unwrap_or(f64::NAN);
                        *self.raw_cond_snapshot.write().unwrap() = raw;
                        raw
                    } else {
                        *self.raw_cond_snapshot.read().unwrap()
                    };
                    if want_hot_diag {
                        self.log_gam_cost(
                            &p,
                            lambdas.as_slice().unwrap_or(&[]),
                            laml,
                            stab_cond,
                            raw_cond,
                            edf,
                            trace_h_inv_s_lambda,
                        );
                    }

                    let prior_cost = self.compute_soft_prior_cost(p);

                    Ok(-laml + prior_cost)
                }
            }
        }

        /// The state-aware closure method for the BFGS optimizer.
        /// Accepts unconstrained parameters `z`, maps to bounded `rho = RHO_BOUND * tanh(z / RHO_BOUND)`.
        pub fn cost_and_grad(&self, z: &Array1<f64>) -> (f64, Array1<f64>) {
            let eval_num = {
                let mut count = self.eval_count.write().unwrap();
                *count += 1;
                *count
            };
            let verbose_opt = false;

            // Map from unbounded z to bounded rho via rho = RHO_BOUND * tanh(z / RHO_BOUND)
            let rho = LogSmoothingParams::new(z.mapv(|v| {
                if v.is_finite() {
                    let scaled = v / RHO_BOUND;
                    RHO_BOUND * scaled.tanh()
                } else {
                    0.0
                }
            }));

            // Attempt to compute the cost and gradient.
            let cost_result = self.compute_cost(&rho);

            match cost_result {
                Ok(cost) if cost.is_finite() => {
                    // Reset consecutive error counter on successful finite cost
                    *self.consecutive_cost_errors.write().unwrap() = 0;
                    match self.compute_gradient(&rho) {
                        Ok(mut grad) => {
                            maybe_audit_gradient(self, &rho, eval_num, &mut grad);
                            // Projected/KKT handling at active bounds in rho-space
                            project_rho_gradient(&rho, &mut grad);
                            // Chain rule: dCost/dz = dCost/drho * drho/dz, where drho/dz|_{z=0} = 1
                            let jac = jacobian_drho_dz_from_rho(&rho);
                            let grad_z = &grad * &jac;
                            let grad_norm = grad_z.dot(&grad_z).sqrt();
                            let last_cost_before = *self.last_cost.read().unwrap();
                            let status = if eval_num == 1 {
                                "Initializing"
                            } else if cost < last_cost_before {
                                "Improving"
                            } else {
                                "Exploring"
                            };
                            let eval_state = if eval_num == 1 {
                                "initial"
                            } else if cost < last_cost_before {
                                "accepted"
                            } else {
                                "trial"
                            };
                            if verbose_opt {
                                crate::visualizer::update(
                                    cost,
                                    grad_norm,
                                    status,
                                    eval_num as f64,
                                    eval_state,
                                );
                            }

                            // --- Correct State Management: Only Update on Actual Improvement ---
                            // Print summary every 50 steps to avoid spam (graph shows real-time anyway)
                            const PRINT_INTERVAL: u64 = 50;
                            let should_print = eval_num == 1 || eval_num % PRINT_INTERVAL == 0;

                            if eval_num == 1 {
                                if verbose_opt {
                                    println!("\n[BFGS] Starting optimization...");
                                    println!(
                                        "  -> Initial Cost: {cost:.7} | Grad Norm: {grad_norm:.6e}"
                                    );
                                }
                                *self.last_cost.write().unwrap() = cost;
                                *self.last_grad_norm.write().unwrap() = grad_norm;
                            } else if cost < *self.last_cost.read().unwrap() {
                                let improvement = *self.last_cost.read().unwrap() - cost;
                                if should_print && verbose_opt {
                                    println!(
                                        "[BFGS Step {eval_num}] Cost: {cost:.7} (Δ={improvement:.2e}) | Grad: {grad_norm:.6e}"
                                    );
                                }
                                *self.last_cost.write().unwrap() = cost;
                                *self.last_grad_norm.write().unwrap() = grad_norm;
                            } else {
                                // Trial step that didn't improve - only log every PRINT_INTERVAL
                                if should_print && verbose_opt {
                                    println!(
                                        "[BFGS Step {eval_num}] Trial (no improvement) | Best: {:.7}",
                                        *self.last_cost.read().unwrap()
                                    );
                                }
                            }

                            (cost, grad_z)
                        }
                        Err(e) => {
                            if verbose_opt {
                                println!(
                                    "\n[BFGS FAILED Step #{eval_num}] -> Gradient calculation error: {e:?}"
                                );
                            }
                            // Generate retreat gradient toward heavier smoothing in rho-space
                            let retreat_rho_grad = Array1::from_elem(rho.len(), -1.0);
                            let jac = jacobian_drho_dz_from_rho(&rho);
                            let retreat_gradient = &retreat_rho_grad * &jac;
                            (f64::INFINITY, retreat_gradient)
                        }
                    }
                }
                // Special handling for infinite costs
                Ok(cost) if cost.is_infinite() => {
                    if verbose_opt {
                        println!(
                            "\n[BFGS Step #{eval_num}] -> Cost is infinite, computing retreat gradient"
                        );
                    }
                    // Diagnostics: report which rho are at bounds
                    if !rho.is_empty() {
                        let at_lower: Vec<usize> = rho
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &v)| {
                                if v <= -RHO_BOUND + 1e-8 {
                                    Some(i)
                                } else {
                                    None
                                }
                            })
                            .collect();
                        let at_upper: Vec<usize> = rho
                            .iter()
                            .enumerate()
                            .filter_map(
                                |(i, &v)| if v >= RHO_BOUND - 1e-8 { Some(i) } else { None },
                            )
                            .collect();
                        if verbose_opt {
                            eprintln!("  -> Rho bounds: lower={:?} upper={:?}", at_lower, at_upper);
                        }
                    }
                    // Try to get a useful gradient direction to move away from problematic region
                    let gradient = match self.compute_gradient(&rho) {
                        Ok(grad) => grad,
                        Err(_) => z.mapv(|v| {
                            if v.is_finite() {
                                v.signum().max(0.0) + 1.0
                            } else {
                                1.0
                            }
                        }),
                    };
                    let jac = jacobian_drho_dz_from_rho(&rho);
                    let gradient = &gradient * &jac;
                    let grad_norm = gradient.dot(&gradient).sqrt();
                    if verbose_opt {
                        println!("  -> Retreat gradient norm: {grad_norm:.6e}");
                    }

                    (cost, gradient)
                }
                // Explicitly handle underlying error to avoid swallowing details
                Err(e) => {
                    log::warn!(
                        "[BFGS Step #{eval_num}] Underlying cost computation failed: {:?}. Retreating.",
                        e
                    );
                    // Track consecutive errors so we can abort after repeated failures
                    {
                        let mut cnt = self.consecutive_cost_errors.write().unwrap();
                        *cnt += 1;
                    }
                    *self.last_cost_error_msg.write().unwrap() = Some(format!("{:?}", e));
                    if verbose_opt {
                        println!(
                            "\n[BFGS FAILED Step #{eval_num}] -> Cost computation failed. Optimizer will backtrack."
                        );
                    }
                    // Generate retreat gradient toward heavier smoothing in rho-space
                    let retreat_rho_grad = Array1::from_elem(rho.len(), -1.0);
                    let jac = jacobian_drho_dz_from_rho(&rho);
                    let retreat_gradient = &retreat_rho_grad * &jac;
                    (f64::INFINITY, retreat_gradient)
                }
                // Cost was non-finite or an error occurred.
                _ => {
                    if verbose_opt {
                        println!(
                            "\n[BFGS FAILED Step #{eval_num}] -> Cost is non-finite or errored. Optimizer will backtrack."
                        );
                    }

                    // For infinite costs, compute a more informed gradient instead of zeros
                    // Generate retreat gradient toward heavier smoothing in rho-space
                    let retreat_rho_grad = Array1::from_elem(rho.len(), -1.0);
                    let jac = jacobian_drho_dz_from_rho(&rho);
                    let retreat_gradient = &retreat_rho_grad * &jac;
                    (f64::INFINITY, retreat_gradient)
                }
            }
        }
        ///
        /// -------------------------------------------------------------------------
        /// Exact non-Laplace evidence identities (reference comments; not runtime path)
        /// -------------------------------------------------------------------------
        /// We optimize a Laplace-style outer objective for scalability, but the exact
        /// marginal likelihood for non-Gaussian models can be written analytically as:
        ///
        ///   L(ρ) = ∫ exp(l(β) - 0.5 βᵀ S(ρ) β) dβ,   S(ρ)=Σ_k exp(ρ_k) S_k.
        ///
        /// Universal exact gradient identity (when differentiation under the integral
        /// is justified and L(ρ) < ∞):
        ///
        ///   ∂_{ρ_k} log L(ρ)
        ///   = -0.5 * exp(ρ_k) * E_{π(β|y,ρ)}[ βᵀ S_k β ].
        ///
        /// Laplace bridge to implemented terms:
        /// - If π(β|y,ρ) is approximated locally by N(β̂, H^{-1}), then
        ///     E[βᵀ S_k β] ≈ β̂ᵀ S_k β̂ + tr(H^{-1} S_k),
        ///   giving the familiar quadratic + trace structure.
        /// - In this code those appear as:
        ///     0.5 * β̂ᵀ S_k^ρ β̂,
        ///     -0.5 * tr(S^+ S_k^ρ),
        ///     +0.5 * tr(H^{-1} H_k).
        ///
        /// Why this does NOT collapse to only tr(H^{-1}S_k):
        /// - The exact identity differentiates the true integral measure.
        /// - LAML differentiates a moving approximation:
        ///     V_LAML(ρ) = -ℓ(β̂(ρ)) + 0.5 β̂(ρ)ᵀ S(ρ) β̂(ρ)
        ///                 + 0.5 log|H(ρ)| - 0.5 log|S(ρ)|_+.
        /// - Here both center β̂(ρ) and curvature H(ρ) move with ρ.
        /// - For non-Gaussian families, H_k includes the third-derivative tensor path
        ///   through β̂(ρ), i.e. H_k != S_k^ρ. These are the explicit dH/dρ_k terms
        ///   retained below to differentiate the Laplace objective exactly.
        ///
        /// For Bernoulli-logit, an exact Pólya-Gamma augmentation gives:
        ///
        ///   L(ρ) = 2^{-n} (2π)^{p/2}
        ///          E_{ω_i ~ PG(1,0)} [ |Q(ω,ρ)|^{-1/2} exp(0.5 bᵀ Q^{-1} b) ],
        ///   Q(ω,ρ)=S(ρ)+XᵀΩX, b=Xᵀ(y-1/2).
        ///
        /// and
        ///
        ///   ∂_{ρ_k} log L
        ///   = -0.5 * exp(ρ_k) *
        ///     E_{ω|y,ρ}[ tr(S_k Q^{-1}) + μᵀ S_k μ ],  μ=Q^{-1}b.
        /// Equivalently, since β|ω,y,ρ ~ N(μ,Q^{-1}):
        ///   E[βᵀS_kβ | ω,y,ρ] = tr(S_k Q^{-1}) + μᵀS_kμ.
        ///
        /// yielding exact (but high-dimensional) contour integrals / series after
        /// analytically integrating β.
        ///
        /// Practical note:
        /// - These are exact equalities but generally not polynomial-time tractable
        ///   for arbitrary dense (X, n, p).
        /// - This code therefore uses deterministic Laplace/implicit-differentiation
        ///   machinery for the main optimizer path, with exact tensor terms where
        ///   feasible (H_k, H_{kℓ}, c/d arrays), and scalable trace backends.
        ///
        /// FULL OUTER-DERIVATIVE REFERENCE (exact system, sign convention used here)
        /// -------------------------------------------------------------------------
        /// This optimizer minimizes an outer cost V(ρ).
        ///
        /// Common definitions:
        ///   λ_k = exp(ρ_k)
        ///   S(ρ) = Σ_k λ_k S_k + δI
        ///   A_k = ∂S/∂ρ_k = λ_k S_k
        ///   A_{kℓ} = ∂²S/(∂ρ_k∂ρ_ℓ) = δ_{kℓ} A_k
        ///
        /// Inner mode (β̂):
        ///   ∇_β ℓ(β̂) - S(ρ) β̂ = 0
        ///
        /// Curvature:
        ///   H(ρ) = -∇²_β ℓ(β̂(ρ)) + S(ρ)
        ///
        ///   w_i = -∂²ℓ_i/∂η_i²
        ///   d_i = -∂³ℓ_i/∂η_i³
        ///   e_i = -∂⁴ℓ_i/∂η_i⁴
        ///
        /// Then:
        ///   H_k = A_k + Xᵀ diag(d ⊙ u_k) X,     u_k := X B_k
        ///   H_{kℓ} = δ_{kℓ}A_k + Xᵀ diag(e ⊙ u_k ⊙ u_ℓ + d ⊙ u_{kℓ}) X
        ///
        /// with implicit derivatives:
        ///   B_k := ∂β̂/∂ρ_k = -H^{-1}(A_k β̂)
        ///   H B_{kℓ} = -(H_ℓ B_k + δ_{kℓ}A_k β̂ + A_k B_ℓ)
        ///
        /// Non-Gaussian negative LAML cost:
        ///   V(ρ) = -ℓ(β̂) + 0.5 β̂ᵀSβ̂ + 0.5 log|H| - 0.5 log|S|_+
        ///
        /// Exact gradient:
        ///   g_k = 0.5 β̂ᵀA_kβ̂ + 0.5 tr(H^{-1}H_k) - 0.5 ∂_k log|S|_+
        ///
        /// Exact Hessian decomposition:
        ///   ∂²V/(∂ρ_k∂ρ_ℓ) = Q_{kℓ} + L_{kℓ} + P_{kℓ}
        ///
        ///   Q_{kℓ} = 0.5 δ_{kℓ} β̂ᵀA_kβ̂ - B_ℓᵀ H B_k
        ///
        ///   L_{kℓ} = 0.5 [ tr(H^{-1}H_{kℓ}) - tr(H^{-1}H_k H^{-1}H_ℓ) ]
        ///
        ///   P_{kℓ} = -0.5 ∂²_{kℓ} log|S|_+
        ///
        /// Here, this function computes the exact gradient terms (including dH/dρ_k via d_i).
        /// The full exact Hessian is not assembled in this loop because it requires B_{kℓ}
        /// solves and fourth-derivative terms for every (k,ℓ) pair.
        ///
        /// Gaussian REML note:
        ///   In identity-link Gaussian, d=e=0 so H_k=A_k and H_{kℓ}=δ_{kℓ}A_k.
        ///   With profiled φ, use either:
        ///   - explicit profiled objective derivatives, or
        ///   - Schur complement in (ρ, log φ):
        ///       H_prof = H_{ρρ} - H_{ρα} H_{αα}^{-1} H_{αρ}.
        ///
        /// Pseudo-determinant note:
        ///   The code uses fixed-rank/stabilized conventions for log|S|_+ to keep objective
        ///   derivatives smooth and consistent with the transformed penalty basis used by PIRLS.
        ///
        /// This is the core of the outer optimization loop and provides the search direction for the BFGS algorithm.
        /// The calculation differs significantly between the Gaussian (REML) and non-Gaussian (LAML) cases.
        ///
        /// # Mathematical Basis (Gaussian/REML Case)
        ///
        /// For Gaussian models (Identity link), we minimize the negative REML log-likelihood, which serves as our cost function.
        /// From Wood (2011, JRSSB, Eq. 4), the cost function to minimize is:
        ///
        ///   Cost(ρ) = -l_r(ρ) = D_p / (2φ) + (1/2)log|XᵀWX + S(ρ)| - (1/2)log|S(ρ)|_+
        ///
        /// where D_p is the penalized deviance, H = XᵀWX + S(ρ) is the penalized Hessian, S(ρ) is the total
        /// penalty matrix, and |S(ρ)|_+ is the pseudo-determinant.
        ///
        /// The gradient ∇Cost(ρ) is computed term-by-term. A key simplification for the Gaussian case is the
        /// **envelope theorem**: at the P-IRLS optimum for β̂, the derivative of the cost function with respect to β̂ is zero.
        /// This means we only need the *partial* derivatives with respect to ρ, and the complex indirect derivatives
        /// involving ∂β̂/∂ρ can be ignored.
        ///
        /// # Mathematical Basis (Non-Gaussian/LAML Case)
        ///
        /// For non-Gaussian models, the envelope theorem does not apply because the weight matrix W depends on β̂.
        /// The gradient requires calculating the full derivative, including the indirect term (∂V/∂β̂)ᵀ(∂β̂/∂ρ).
        /// This leads to a different final formula involving derivatives of the weight matrix, as detailed in
        /// Wood (2011, Appendix D).
        ///
        /// This method handles two distinct statistical criteria for marginal likelihood optimization:
        ///
        /// - For Gaussian models (Identity link), this calculates the exact REML gradient
        ///   (Restricted Maximum Likelihood).
        /// - For non-Gaussian GLMs, this calculates the LAML gradient (Laplace Approximate
        ///   Marginal Likelihood) as derived in Wood (2011, Appendix C & D).
        ///
        /// # Mathematical Theory
        ///
        /// The gradient calculation requires careful application of the chain rule and envelope theorem
        /// due to the nested optimization structure of GAMs:
        ///
        /// - The inner loop (P-IRLS) finds coefficients β̂ that maximize the penalized log-likelihood
        ///   for a fixed set of smoothing parameters ρ.
        /// - The outer loop (BFGS) finds smoothing parameters ρ that maximize the marginal likelihood.
        ///
        /// Since β̂ is an implicit function of ρ, the total derivative is:
        ///
        ///    dV_R/dρ_k = (∂V_R/∂β̂)ᵀ(∂β̂/∂ρ_k) + ∂V_R/∂ρ_k
        ///
        /// By the envelope theorem, (∂V_R/∂β̂) = 0 at the optimum β̂, so the first term vanishes.
        ///
        /// # Key Distinction Between REML and LAML Gradients
        ///
        /// - Gaussian (REML): by the envelope theorem the indirect β̂ terms vanish. The deviance
        ///   contribution reduces to the penalty-only derivative, yielding the familiar
        ///   (β̂ᵀS_kβ̂)/σ² piece in the gradient.
        /// - Non-Gaussian (LAML): there is no cancellation of the penalty derivative within the
        ///   deviance component. The derivative of the penalized deviance contains both
        ///   d(D)/dρ_k and d(βᵀSβ)/dρ_k. Our implementation follows mgcv’s gdi1: we add the penalty
        ///   derivative to the deviance derivative before applying the 1/2 factor.
        // Stage: Start with the chain rule for any λₖ,
        //     dV/dλₖ = ∂V/∂λₖ  (holding β̂ fixed)  +  (∂V/∂β̂)ᵀ · (∂β̂/∂λₖ).
        //     The first summand is called the direct part, the second the indirect part.
        //
        // Stage: Note the two outer criteria—Gaussian likelihood maximizes REML, while non-Gaussian likelihood
        //     maximizes a Laplace approximation to the marginal likelihood (LAML). These objectives respond differently to β̂.
        //
        //     2.1  Gaussian case, REML.
        //          The REML construction integrates the fixed effects out of the likelihood.  At the optimum
        //          the partial derivative ∂V/∂β̂ is exactly zero.  The indirect part therefore vanishes.
        //          What remains is the direct derivative of the penalty and determinant terms.  The penalty
        //          contribution is found by differentiating −½ β̂ᵀ S_λ β̂ / σ² with respect to λₖ; this yields
        //          −½ β̂ᵀ Sₖ β̂ / σ².  No opposing term exists, so the quantity stays in the REML gradient.
        //          The code path selected by LinkFunction::Identity therefore computes
        //          beta_term = β̂ᵀ Sₖ β̂ and places it inside
        //          gradient[k] = 0.5 * λₖ * (beta_term / σ² − trace_term).
        //
        //     2.2  Non-Gaussian case, LAML.
        //          The Laplace objective contains −½ log |H_p| with H_p = Xᵀ W(β̂) X + S_λ.  Because W
        //          depends on β̂, the total derivative includes dW/dλₖ via β̂.  Differentiating the
        //          optimality condition for β̂ gives
        //          ∂β̂/∂λₖ = −λₖ H_p⁻¹ Sₖ β̂.  The penalized log-likelihood L(β̂, λ) still obeys the
        //          envelope theorem, so dL/dλₖ = −½ β̂ᵀ Sₖ β̂ (no implicit term).
        //          The resulting cost gradient combines four pieces:
        //            +½ λₖ β̂ᵀ Sₖ β̂
        //            +½ λₖ tr(H_p⁻¹ Sₖ)
        //            +½ tr(H_p⁻¹ Xᵀ ∂W/∂λₖ X)
        //            −½ λₖ tr(S_λ⁺ Sₖ)
        //
        // Stage: Remember that the sign of ∂β̂/∂λₖ matters; from the implicit-function theorem the linear solve reads
        //     −H_p (∂β̂/∂λₖ) = λₖ Sₖ β̂, giving the minus sign used above.  With that sign the indirect and
        //     direct quadratic pieces are exact negatives, which is what the algebra requires.
        pub fn compute_gradient(&self, p: &Array1<f64>) -> Result<Array1<f64>, EstimationError> {
            if self.config.link_function() != LinkFunction::Identity
                && (self.config.objective_consistent_fd_gradient || p.len() == 1)
            {
                // Single-penalty non-Gaussian problems can violate the local
                // objective-trend sign relation under the current analytic
                // gradient approximation. Use objective-consistent FD there.
                return compute_fd_gradient_internal(self, p, false, false);
            }
            // Get the converged P-IRLS result for the current rho (`p`)
            let bundle = match self.obtain_eval_bundle(p) {
                Ok(bundle) => bundle,
                Err(EstimationError::ModelIsIllConditioned { .. }) => {
                    self.current_eval_bundle.write().unwrap().take();
                    // Push toward heavier smoothing: larger rho
                    // Minimizer steps along -grad, so use negative values
                    let grad = p.mapv(|rho| -(rho.abs() + 1.0));
                    return Ok(grad);
                }
                Err(e) => {
                    self.current_eval_bundle.write().unwrap().take();
                    return Err(e);
                }
            };
            self.compute_gradient_with_bundle(p, &bundle)
        }

        /// Helper function that computes gradient using a shared evaluation bundle
        /// so cost and gradient reuse the identical stabilized Hessian and PIRLS state.
        ///
        /// # Exact Outer-Gradient Identity Used by This Function
        ///
        /// Notation:
        /// - `rho[k]` are log-smoothing parameters; `lambda[k] = exp(rho[k])`.
        /// - `S(rho) = Σ_k lambda[k] S_k`.
        /// - `A_k = ∂S/∂rho_k = lambda[k] S_k`.
        /// - `beta_hat(rho)` is the inner PIRLS mode.
        /// - `H(rho)` is the Laplace curvature matrix used by this objective path.
        ///
        /// Outer objective:
        ///   V(rho) = [penalized data-fit at beta_hat]
        ///          + 0.5 log|H(rho)| - 0.5 log|S(rho)|_+.
        ///
        /// Exact derivative form:
        ///   dV/drho_k
        ///   = 0.5 * beta_hat^T A_k beta_hat
        ///   + 0.5 * tr(H^{-1} H_k)
        ///   - 0.5 * tr(S^+ A_k),
        /// where H_k = dH/drho_k is the *total* derivative (includes beta_hat movement).
        ///
        /// Important implementation point:
        /// - We do NOT add a separate `(∇_beta V)^T (d beta_hat / d rho_k)` term on top of
        ///   `tr(H^{-1} H_k)`. That dependence is already inside `H_k`.
        ///
        /// Variable mapping in this function:
        /// - `beta_terms[k]`     => beta_hat^T A_k beta_hat
        /// - `det1_values[k]`    => tr(S^+ A_k)
        /// - `trace_terms[k]`    => tr(H^{-1} H_k) / lambda[k] (before the outer lambda factor)
        /// - final assembly       => 0.5*beta_terms + 0.5*lambda*trace_terms - 0.5*det1
        ///
        /// ## Exact non-Gaussian Hessian system (reference for this implementation)
        ///
        /// For outer parameters ρ with λ_k = exp(ρ_k), A_k = ∂S/∂ρ_k = λ_k S_k, and
        /// H = -∇²ℓ(β̂(ρ)) + S(ρ), exact derivatives are:
        ///
        ///   B_k := ∂β̂/∂ρ_k = -H^{-1}(A_k β̂)
        ///
        ///   H_k := ∂H/∂ρ_k = A_k + D(-∇²ℓ)[B_k]
        ///
        ///   B_{kℓ} solves:
        ///     H B_{kℓ} = -(H_ℓ B_k + δ_{kℓ} A_k β̂ + A_k B_ℓ)
        ///
        ///   H_{kℓ} := ∂²H/(∂ρ_k∂ρ_ℓ)
        ///     = δ_{kℓ}A_k + D²(-∇²ℓ)[B_k,B_ℓ] + D(-∇²ℓ)[B_{kℓ}]
        ///
        /// Then the exact outer Hessian for V(ρ) = -ℓ(β̂)+0.5β̂ᵀSβ̂+0.5log|H|-0.5log|S|_+ is:
        ///
        ///   ∂²V/(∂ρ_k∂ρ_ℓ)
        ///     = 0.5 δ_{kℓ} β̂ᵀA_kβ̂ - B_ℓᵀ H B_k
        ///       + 0.5[ tr(H^{-1}H_{kℓ}) - tr(H^{-1}H_k H^{-1}H_ℓ) ]
        ///       - 0.5 ∂² log|S|_+ /(∂ρ_k∂ρ_ℓ)
        ///
        /// This function computes the exact gradient terms (including the third-derivative
        /// contribution in H_k for logit). Full explicit H_{kℓ} assembly is not
        /// performed in the hot optimization loop because it requires B_{kℓ} solves and
        /// fourth-derivative likelihood terms for every (k,ℓ) pair.
        fn compute_gradient_with_bundle(
            &self,
            p: &Array1<f64>,
            bundle: &EvalShared,
        ) -> Result<Array1<f64>, EstimationError> {
            // If there are no penalties (zero-length rho), the gradient in rho-space is empty.
            if p.is_empty() {
                return Ok(Array1::zeros(0));
            }

            let pirls_result = bundle.pirls_result.as_ref();
            let h_eff = bundle.h_eff.as_ref();
            let ridge_passport = bundle.ridge_passport;

            // Sanity check: penalty dimension consistency across lambdas, R_k, and det1.
            let k_lambda = p.len();
            let k_r = pirls_result.reparam_result.rs_transformed.len();
            let k_d = pirls_result.reparam_result.det1.len();
            if !(k_lambda == k_r && k_r == k_d) {
                return Err(EstimationError::LayoutError(format!(
                    "Penalty dimension mismatch: lambdas={}, R={}, det1={}",
                    k_lambda, k_r, k_d
                )));
            }
            if self.nullspace_dims.len() != k_lambda {
                return Err(EstimationError::LayoutError(format!(
                    "Nullspace dimension mismatch: expected {} entries, got {}",
                    k_lambda,
                    self.nullspace_dims.len()
                )));
            }

            // --- Extract stable transformed quantities ---
            let beta_transformed = pirls_result.beta_transformed.as_ref();
            let reparam_result = &pirls_result.reparam_result;
            // Use cached X·Qs from PIRLS
            let rs_transformed = &reparam_result.rs_transformed;
            let rs_transposed = &reparam_result.rs_transposed;

            let includes_prior = false;
            let (gradient_result, gradient_snapshot, _) = {
                let mut workspace_ref = self.workspace.lock().unwrap();
                let workspace = &mut *workspace_ref;
                let len = p.len();
                workspace.reset_for_eval(len);
                workspace.set_lambda_values(p);
                workspace.zero_cost_gradient(len);
                let lambdas = workspace.lambda_view(len).to_owned();

                // Fixed structural-rank pseudo-determinant derivatives:
                // d/dρ_k log|S|_+ and d²/(dρ_k dρ_ℓ) log|S|_+ are evaluated on a
                // reduced structural subspace (rank = e_transformed.nrows()) with a
                // smooth floor in that reduced block. This avoids adaptive rank flips.
                let (det1_values, _) = self.structural_penalty_logdet_derivatives(
                    rs_transformed,
                    &lambdas,
                    reparam_result.e_transformed.nrows(),
                    ridge_passport.penalty_logdet_ridge(),
                )?;

                // --- Use Single Stabilized Hessian from P-IRLS ---
                // Use the same effective Hessian as the cost function for consistency.
                if ridge_passport.laplace_hessian_ridge() > 0.0 {
                    log::debug!(
                        "Gradient path using PIRLS-stabilized Hessian (ridge {:.3e})",
                        ridge_passport.laplace_hessian_ridge()
                    );
                }

                // Check that the stabilized effective Hessian is still numerically valid.
                // If even the ridged matrix is indefinite, the PIRLS fit is unreliable and we retreat.
                if let Ok((eigenvalues, _)) = h_eff.eigh(Side::Lower) {
                    let min_eig = eigenvalues.iter().fold(f64::INFINITY, |a, &b| a.min(b));
                    const SEVERE_INDEFINITENESS: f64 = -1e-4; // Threshold for severe problems
                    if min_eig < SEVERE_INDEFINITENESS {
                        // The matrix was severely indefinite - signal a need to retreat
                        log::warn!(
                            "Severely indefinite Hessian detected in gradient (min_eig={:.2e}); returning robust retreat gradient.",
                            min_eig
                        );
                        // Generate an informed retreat direction based on current parameters
                        let retreat_grad = p.mapv(|v| -(v.abs() + 1.0));
                        return Ok(retreat_grad);
                    }
                }

                // --- Extract common components ---

                let n = self.y.len() as f64;

                // Implement Wood (2011) exact REML/LAML gradient formulas
                // Reference: gam.fit3.R line 778: REML1 <- oo$D1/(2*scale*gamma) + oo$trA1/2 - rp$det1/2

                match self.config.link_function() {
                    LinkFunction::Identity => {
                        // GAUSSIAN REML GRADIENT - Wood (2011) Section 6.6.1

                        // Calculate scale parameter using the regular REML profiling
                        // φ = D_p / (n - M_p), where M_p is the penalty nullspace dimension.
                        let rss = pirls_result.deviance;

                        // Use stable penalty term calculated in P-IRLS
                        let penalty = pirls_result.stable_penalty_term;
                        let dp = rss + penalty; // Penalized deviance (a.k.a. D_p)
                        let (dp_c, dp_c_grad) = smooth_floor_dp(dp);

                        let factor_g = self.get_faer_factor(p, h_eff);
                        let penalty_rank = pirls_result.reparam_result.e_transformed.nrows();
                        let mp = self.p.saturating_sub(penalty_rank) as f64;
                        let scale = dp_c / (n - mp).max(LAML_RIDGE);
                        // Gaussian profiled-scale identity used by this branch:
                        //   φ̂(ρ) = D_p(ρ)/(n-M_p), with D_p = rss + β̂ᵀSβ̂.
                        // The gradient therefore includes the profiled contribution
                        //   (n-M_p)/2 * D_k / D_p
                        // which is exactly represented by `deviance_grad_term` below.
                        // (Equivalent to Schur-complement profiling in (ρ, log φ).)

                        if dp_c <= DP_FLOOR + DP_FLOOR_SMOOTH_WIDTH {
                            eprintln!(
                                "[REML WARNING] Penalized deviance {:.3e} near DP_FLOOR; using central differences for entire gradient.",
                                dp_c
                            );
                            let mut grad_total_view =
                                workspace.grad_secondary.slice_mut(s![..lambdas.len()]);
                            grad_total_view.fill(0.0);
                            for k in 0..lambdas.len() {
                                let h = 1e-3_f64 * (1.0 + p[k].abs());
                                if h == 0.0 {
                                    continue;
                                }
                                workspace.rho_plus.assign(p);
                                workspace.rho_plus[k] += h;
                                workspace.rho_minus.assign(p);
                                workspace.rho_minus[k] -= h;
                                let cost_plus = self.compute_cost(&workspace.rho_plus)?;
                                let cost_minus = self.compute_cost(&workspace.rho_minus)?;
                                grad_total_view[k] = (cost_plus - cost_minus) / (2.0 * h);
                            }
                            return Ok(grad_total_view.to_owned());
                        }

                        // Three-term gradient computation following mgcv gdi1
                        // for k in 0..lambdas.len() {
                        //   We'll calculate s_k_beta for all cases, as it's needed for both paths
                        //   For Identity link, this is all we need due to envelope theorem
                        //   For other links, we'll use it to compute dβ/dρ_k

                        //   Use transformed penalty matrix for consistent gradient calculation
                        //   let s_k_beta = reparam_result.rs_transformed[k].dot(beta);

                        // For the Gaussian/REML case, the Envelope Theorem applies: at the P-IRLS optimum,
                        // the indirect derivative through β cancels out for the deviance part, leaving only
                        // the direct penalty term derivative. This simplification is not available for
                        // non-Gaussian models where the weight matrix depends on β.

                        // factor_g already computed above; reuse it for trace terms

                        // When the penalized deviance collapses to the numerical floor, the Hessian
                        // can become so ill-conditioned that the analytic ½·log|H| derivative loses
                        // fidelity.  Switch to an exact finite-difference evaluation in that regime
                        // to match the cost function.
                        let use_numeric_logh = dp_c <= DP_FLOOR + DP_FLOOR_SMOOTH_WIDTH;
                        let numeric_logh_grad = if use_numeric_logh {
                            eprintln!(
                                "[REML WARNING] Switching ½·log|H| gradient to numeric finite differences; dp_c={:.3e}.",
                                dp_c
                            );
                            Some(self.numeric_half_logh_grad_with_workspace(p, workspace)?)
                        } else {
                            None
                        };

                        workspace.reset_block_ranges();
                        let mut total_rank = 0;
                        for rt in rs_transposed {
                            let cols = rt.ncols();
                            workspace.block_ranges.push((total_rank, total_rank + cols));
                            total_rank += cols;
                        }
                        workspace.solved_rows = h_eff.nrows();

                        if numeric_logh_grad.is_none() && total_rank > 0 {
                            workspace.concat.fill(0.0);
                            let rows = h_eff.nrows();
                            for ((start, end), rt) in
                                workspace.block_ranges.iter().zip(rs_transposed.iter())
                            {
                                if *end > *start {
                                    workspace
                                        .concat
                                        .slice_mut(s![..rows, *start..*end])
                                        .assign(rt);
                                }
                            }
                            let rows = h_eff.nrows();
                            let cols = total_rank;
                            {
                                let mut solved_slice =
                                    workspace.solved.slice_mut(s![..rows, ..cols]);
                                solved_slice.assign(&workspace.concat.slice(s![..rows, ..cols]));
                                if let Some(slice) = solved_slice.as_slice_mut() {
                                    let mut solved_view =
                                        faer::MatMut::from_row_major_slice_mut(slice, rows, cols);
                                    factor_g.solve_in_place(solved_view.as_mut());
                                } else {
                                    let mut temp =
                                        faer::Mat::from_fn(rows, cols, |i, j| solved_slice[(i, j)]);
                                    factor_g.solve_in_place(temp.as_mut());
                                    for j in 0..cols {
                                        for i in 0..rows {
                                            solved_slice[(i, j)] = temp[(i, j)];
                                        }
                                    }
                                }
                            }
                            workspace.solved_rows = rows;
                        } else {
                            workspace.solved_rows = 0;
                        }

                        // Gradient correction for spectral truncation (same as Logit path).
                        // Error_k = 0.5 * λ_k * tr(M_⊥ * (U_⊥^T S_k U_⊥)) where M_⊥ = U_⊥^T H⁻¹ U_⊥.
                        let u_truncated_gauss = &reparam_result.u_truncated;
                        let truncated_count_gauss = u_truncated_gauss.ncols();

                        let gaussian_corrections: Vec<f64> =
                            if truncated_count_gauss > 0 && workspace.solved_rows > 0 {
                                let rows = h_eff.nrows();
                                let mut h_inv_u_perp =
                                    faer::Mat::<f64>::zeros(rows, truncated_count_gauss);

                                for i in 0..rows.min(u_truncated_gauss.nrows()) {
                                    for j in 0..truncated_count_gauss {
                                        h_inv_u_perp[(i, j)] = u_truncated_gauss[(i, j)];
                                    }
                                }

                                factor_g.solve_in_place(h_inv_u_perp.as_mut());

                                let mut m_perp = faer::Mat::<f64>::zeros(
                                    truncated_count_gauss,
                                    truncated_count_gauss,
                                );
                                for i in 0..truncated_count_gauss {
                                    for j in 0..truncated_count_gauss {
                                        let mut sum = 0.0;
                                        for r in 0..rows.min(u_truncated_gauss.nrows()) {
                                            sum += u_truncated_gauss[(r, i)] * h_inv_u_perp[(r, j)];
                                        }
                                        m_perp[(i, j)] = sum;
                                    }
                                }

                                let mut corrections = vec![0.0; lambdas.len()];
                                for k_idx in 0..lambdas.len() {
                                    let r_k = &rs_transformed[k_idx];
                                    let rank_k = r_k.nrows();

                                    let mut w_k =
                                        faer::Mat::<f64>::zeros(rank_k, truncated_count_gauss);
                                    for i in 0..rank_k {
                                        for j in 0..truncated_count_gauss {
                                            let mut sum = 0.0;
                                            for l in 0..r_k.ncols().min(u_truncated_gauss.nrows()) {
                                                sum += r_k[(i, l)] * u_truncated_gauss[(l, j)];
                                            }
                                            w_k[(i, j)] = sum;
                                        }
                                    }

                                    let mut trace_error = 0.0;
                                    for i in 0..truncated_count_gauss {
                                        for j in 0..truncated_count_gauss {
                                            let mut wtw_ij = 0.0;
                                            for l in 0..rank_k {
                                                wtw_ij += w_k[(l, i)] * w_k[(l, j)];
                                            }
                                            trace_error += m_perp[(i, j)] * wtw_ij;
                                        }
                                    }

                                    corrections[k_idx] = 0.5 * lambdas[k_idx] * trace_error;
                                }
                                corrections
                            } else {
                                vec![0.0; lambdas.len()]
                            };

                        let numeric_logh_grad_ref = numeric_logh_grad.as_ref();
                        let det1_values = &det1_values;
                        let beta_ref = beta_transformed;
                        let solved_rows = workspace.solved_rows;
                        let block_ranges_ref = &workspace.block_ranges;
                        let solved_ref = &workspace.solved;
                        let concat_ref = &workspace.concat;
                        let gaussian_corrections_ref = &gaussian_corrections;
                        // Exact Gaussian identity REML gradient (profiled scale) in log-smoothing coordinates:
                        //
                        //   V_REML(ρ) =
                        //     0.5 * log|H|
                        //   - 0.5 * log|S|_+
                        //   + ((n - M_p)/2) * log(2π φ̂)
                        //   + const,
                        //
                        // where H = Xᵀ W0 X + S(ρ), S(ρ) = Σ_k λ_k S_k + δI, λ_k = exp(ρ_k),
                        // and φ̂ = D_p / (n - M_p), D_p = ||W0^(1/2)(y - Xβ̂ - o)||² + β̂ᵀ S β̂.
                        //
                        // Because Gaussian identity has c_i = d_i = 0, we have:
                        //   H_k := ∂H/∂ρ_k = S_k^ρ = λ_k S_k.
                        // Envelope theorem at β̂(ρ) gives:
                        //   ∂D_p/∂ρ_k = β̂ᵀ S_k^ρ β̂.
                        // Therefore:
                        //   ∂V_REML/∂ρ_k =
                        //     0.5 * tr(H^{-1} S_k^ρ)
                        //   - 0.5 * tr(S^+ S_k^ρ)
                        //   + (1/(2 φ̂)) * β̂ᵀ S_k^ρ β̂.
                        //
                        // Mapping to variables below:
                        //   d1 / (2*scale)                     -> (1/(2 φ̂)) * β̂ᵀ S_k^ρ β̂
                        //   log_det_h_grad_term (or numeric)   -> 0.5 * tr(H^{-1} S_k^ρ)
                        //   0.5 * det1_values[k]               -> 0.5 * tr(S^+ S_k^ρ)
                        let compute_gaussian_grad = |k: usize| -> f64 {
                            let r_k = &rs_transformed[k];
                            // Avoid forming S_k: compute S_k β = Rᵀ (R β)
                            let r_beta = r_k.dot(beta_ref);
                            let s_k_beta_transformed = r_k.t().dot(&r_beta);

                            // Component 1 derivation (profiled Gaussian REML):
                            //
                            //   V_prof includes (n-M_p)/2 * log D_p(ρ), so
                            //   ∂V_prof/∂ρ_k contributes (n-M_p)/2 * D_k / D_p = D_k/(2φ̂),
                            //   φ̂ = D_p/(n-M_p).
                            //
                            // At β̂, envelope cancellation gives:
                            //   D_k = β̂ᵀ A_k β̂ = λ_k β̂ᵀ S_k β̂.
                            //
                            // `d1` stores D_k, and the expression below is D_k/(2φ̂)
                            // with the smooth-floor derivative factor `dp_c_grad`.
                            let d1 = lambdas[k] * beta_ref.dot(&s_k_beta_transformed);
                            let deviance_grad_term = dp_c_grad * (d1 / (2.0 * scale));

                            // Component 2 derivation:
                            //   ∂/∂ρ_k [0.5 log|H|] = 0.5 tr(H^{-1} H_k),
                            // and for Gaussian identity H_k = A_k = λ_k S_k.
                            //
                            // Root form gives:
                            //   tr(H^{-1}A_k) = λ_k tr(H^{-1}R_kᵀR_k)
                            //                = λ_k tr(R_k H^{-1} R_kᵀ),
                            // computed by solving H X = R_kᵀ and taking tr(R_k X).
                            let log_det_h_grad_term = if let Some(g) = numeric_logh_grad_ref {
                                g[k]
                            } else if solved_rows > 0 {
                                let (start, end) = block_ranges_ref[k];
                                if end > start {
                                    let solved_block =
                                        solved_ref.slice(s![..solved_rows, start..end]);
                                    let rt_block = concat_ref.slice(s![..solved_rows, start..end]);
                                    let trace_h_inv_s_k = kahan_sum(
                                        solved_block
                                            .iter()
                                            .zip(rt_block.iter())
                                            .map(|(&x, &y)| x * y),
                                    );
                                    let tra1 = lambdas[k] * trace_h_inv_s_k;
                                    tra1 / 2.0
                                } else {
                                    0.0
                                }
                            } else {
                                0.0
                            };

                            // Apply truncation correction to match truncated cost function
                            let corrected_log_det_h =
                                log_det_h_grad_term - gaussian_corrections_ref[k];

                            // Component 3 derivation:
                            //   -0.5 * ∂/∂ρ_k log|S|_+,
                            // with `det1_values[k]` already equal to ∂ log|S|_+ / ∂ρ_k.
                            let log_det_s_grad_term = 0.5 * det1_values[k];

                            deviance_grad_term + corrected_log_det_h - log_det_s_grad_term
                        };

                        let mut gaussian_grad = Vec::with_capacity(lambdas.len());
                        for k in 0..lambdas.len() {
                            gaussian_grad.push(compute_gaussian_grad(k));
                        }
                        workspace
                            .cost_gradient_view(len)
                            .assign(&Array1::from_vec(gaussian_grad));
                    }
                    _ => {
                        // NON-GAUSSIAN LAML GRADIENT (exact in ρ, including dH/dρ third-derivative term)
                        //
                        // Objective:
                        //   V_LAML(ρ) =
                        //     -ℓ(β̂) + 0.5 β̂ᵀ S β̂
                        //   - 0.5 log|S|_+
                        //   + 0.5 log|H|
                        //   + const
                        //
                        // with H(ρ) = J(β̂(ρ)) + S(ρ), J = Xᵀ diag(b) X.
                        //
                        // Exact gradient:
                        //   ∂V/∂ρ_k =
                        //     0.5 β̂ᵀ S_k^ρ β̂
                        //   - 0.5 tr(S^+ S_k^ρ)
                        //   + 0.5 tr(H^{-1} H_k)
                        //
                        // where:
                        //   S_k^ρ = λ_k S_k, λ_k = exp(ρ_k),
                        //   v_k   = H^{-1}(S_k^ρ β̂),
                        //   H_k   = S_k^ρ - Xᵀ diag(c ⊙ (X v_k)) X,
                        // and c_i = -∂^3 ℓ_i / ∂η_i^3.
                        //
                        // The second term inside H_k is the exact "missing tensor term":
                        //   ∂H/∂ρ_k ≠ S_k^ρ
                        // for non-Gaussian families; dropping it yields the usual approximation.
                        //
                        // Implementation strategy here (logit path):
                        //   1) build S_k β̂ in transformed basis via penalty roots R_k,
                        //   2) solve/apply H_+^† to get v_k and leverage terms,
                        //   3) evaluate tr(H_+^† H_k) as
                        //        tr(H_+^† S_k) - tr(H_+^† Xᵀ diag(c ⊙ X v_k) X),
                        //   4) assemble
                        //        0.5*β̂ᵀA_kβ̂ + 0.5*tr(H_+^†H_k) - 0.5*tr(S^+A_k).
                        //
                        // There is intentionally no extra "(∇_β V)^T dβ/dρ" add-on here:
                        // the beta-dependence path is already encoded in H_k through the
                        // third-derivative contraction term.
                        // Replace FD with implicit differentiation for logit models.
                        // When Firth bias reduction is enabled, the inner objective is:
                        //   L*(beta, rho) = l(beta) - 0.5 * beta' S_lambda beta
                        //                 + 0.5 * log|X' W(beta) X|
                        // with W depending on beta (logit: w_i = mu_i (1 - mu_i)).
                        // Stationarity: grad_beta L* = 0, so the implicit derivative uses
                        // H_total = X' W X + S_lambda - d^2/d beta^2 (0.5 * log|X' W X|).
                        //
                        // Exact Firth derivatives (let K = (X' W X)^{-1}):
                        //   Phi(beta) = 0.5 * log|X' W X|
                        //   grad Phi_j = 0.5 * tr(K X' (dW/d beta_j) X)
                        //             = 0.5 * sum_i h_i * (d w_i / d eta_i) * x_ij
                        //   where h_i = x_i' K x_i (leverages in weighted space).
                        //
                        //   Hessian:
                        //     d^2 Phi / (d beta_j d beta_l) =
                        //       -0.5 * tr(K X' (dW/d beta_l) X K X' (dW/d beta_j) X)
                        //       +0.5 * sum_i h_i * (d^2 w_i / d eta_i^2) * x_ij * x_il
                        //
                        // This curvature enters H_total and therefore d beta_hat / d rho_k.
                        // Our analytic LAML gradient uses H_pen = X' W X + S_lambda only,
                        // so it is inconsistent with the Firth-adjusted objective unless
                        // we add H_phi. Below we compute H_phi and use H_total for the
                        // implicit solve (d beta_hat / d rho). If that fails, we fall
                        // back to H_pen for stability.
                        let w_prime = pirls_result.solve_c_array.clone();
                        if !w_prime.iter().all(|v| v.is_finite()) {
                            let g_pll =
                                self.numeric_penalised_ll_grad_with_workspace(p, workspace)?;
                            let g_half_logh =
                                self.numeric_half_logh_grad_with_workspace(p, workspace)?;
                            let det1_full = det1_values.clone();
                            let mut laml_grad = Vec::with_capacity(lambdas.len());
                            for k in 0..lambdas.len() {
                                let gradient_value = g_pll[k] + g_half_logh[k] - 0.5 * det1_full[k];
                                laml_grad.push(gradient_value);
                            }
                            workspace
                                .cost_gradient_view(len)
                                .assign(&Array1::from_vec(laml_grad));
                            // Continue to prior-gradient adjustment below.
                        } else {
                            let clamp_nonsmooth = self.config.firth_bias_reduction
                                && pirls_result
                                    .solve_mu
                                    .iter()
                                    .any(|&mu| mu * (1.0 - mu) < Self::MIN_DMU_DETA);
                            if clamp_nonsmooth {
                                // Keep analytic gradient as the optimizer default even when IRLS
                                // weights are clamped, to avoid FD ridge-jitter artifacts in
                                // line-search/BFGS updates.
                                log::debug!(
                                    "[REML] IRLS weight clamp detected; continuing with analytic gradient"
                                );
                            }
                            let k_count = lambdas.len();
                            let det1_values = &det1_values;
                            let mut laml_grad = Vec::with_capacity(k_count);
                            let beta_ref = beta_transformed;
                            let mut beta_terms = Array1::<f64>::zeros(k_count);
                            let mut s_k_beta_all: Vec<Array1<f64>> = Vec::with_capacity(k_count);
                            for k in 0..k_count {
                                let r_k = &rs_transformed[k];
                                let r_beta = r_k.dot(beta_ref);
                                let s_k_beta = r_k.t().dot(&r_beta);
                                beta_terms[k] = lambdas[k] * beta_ref.dot(&s_k_beta);
                                s_k_beta_all.push(s_k_beta);
                            }

                            // Keep outer gradient on the same Hessian surface as PIRLS.
                            // The outer loop uses H_eff consistently (no H_phi subtraction).

                            // P-IRLS already folded any stabilization ridge into h_eff.

                            // TRACE TERM COMPUTATION (exact non-Gaussian/logit dH term):
                            //   tr(H_+^\dagger H_k), with
                            //   H_k = S_k - X^T diag(c ⊙ (X v_k)) X,  v_k = H_+^\dagger (S_k beta).
                            //
                            // We evaluate this without explicit third-derivative tensors:
                            //   tr(H_+^\dagger S_k) = ||R_k W||_F^2
                            //   tr(H_+^\dagger X^T diag(t_k) X) = Σ_i t_k[i] * h_i,
                            // where t_k = c ⊙ (X v_k), h_i = x_i^T H_+^\dagger x_i, and H_+^\dagger = W W^T.
                            //
                            // This is the matrix-free realization of the exact identity:
                            //   tr(H^{-1}H_k) = tr(H^{-1}A_k) + tr(H^{-1}D(-∇²ℓ)[B_k]),
                            // with B_k = -H^{-1}(A_kβ̂).
                            //
                            //   D(-∇²ℓ)[B_k] = Xᵀ diag(d ⊙ (X B_k)) X,
                            // where d_i = -∂³ℓ_i/∂η_i³. Here `c_vec` stores this per-observation
                            // third derivative quantity in the stabilized logit path.
                            let w_pos = bundle.h_pos_factor_w.as_ref();
                            let n_obs = pirls_result.solve_mu.len();

                            // c_i = dW_ii/deta_i for H = Xᵀ W X + S (exact for this objective surface).
                            let c_vec = w_prime.clone();

                            // h_i = x_i^T H_+^\dagger x_i = ||(XW)_{i,*}||^2.
                            let mut leverage_h_pos = Array1::<f64>::zeros(n_obs);
                            if w_pos.ncols() > 0 {
                                match &pirls_result.x_transformed {
                                    DesignMatrix::Dense(x_dense) => {
                                        let xw = x_dense.dot(w_pos);
                                        for i in 0..xw.nrows() {
                                            leverage_h_pos[i] =
                                                xw.row(i).iter().map(|v| v * v).sum();
                                        }
                                    }
                                    DesignMatrix::Sparse(_) => {
                                        for col in 0..w_pos.ncols() {
                                            let w_col = w_pos.column(col).to_owned();
                                            let xw_col = pirls_result
                                                .x_transformed
                                                .matrix_vector_multiply(&w_col);
                                            Zip::from(&mut leverage_h_pos)
                                                .and(&xw_col)
                                                .for_each(|h, &v| *h += v * v);
                                        }
                                    }
                                }
                            }

                            // Precompute r = X^T (c ⊙ h) once:
                            //   trace_third_k = (c ⊙ h)^T (X v_k) = r^T v_k.
                            // This removes the per-k O(np) multiply X*v_k from the hot loop.
                            let c_times_h = &c_vec * &leverage_h_pos;
                            let r_third = pirls_result
                                .x_transformed
                                .transpose_vector_multiply(&c_times_h);

                            // Batch all v_k = H_+^† (S_k beta) into one BLAS-3 path:
                            //   V = W (W^T [S_1 beta, ..., S_K beta]).
                            let mut s_k_beta_mat = Array2::<f64>::zeros((beta_ref.len(), k_count));
                            for (k_idx, s_k_beta) in s_k_beta_all.iter().enumerate() {
                                s_k_beta_mat.column_mut(k_idx).assign(s_k_beta);
                            }
                            let v_all = if w_pos.ncols() > 0 && k_count > 0 {
                                let wt_sk_beta_all = w_pos.t().dot(&s_k_beta_mat);
                                w_pos.dot(&wt_sk_beta_all)
                            } else {
                                Array2::<f64>::zeros((beta_ref.len(), k_count))
                            };

                            let mut trace_terms: Vec<f64> = Vec::with_capacity(k_count);
                            for k_idx in 0..k_count {
                                let r_k = &rs_transformed[k_idx];
                                if r_k.ncols() == 0 || w_pos.ncols() == 0 {
                                    trace_terms.push(0.0);
                                    continue;
                                }

                                // First piece:
                                //   tr(H_+^† S_k) = ||R_k W||_F^2, with H_+^† = W W^T.
                                let rkw = r_k.dot(w_pos);
                                let trace_h_inv_s_k: f64 = rkw.iter().map(|v| v * v).sum();

                                // Exact third-derivative contraction:
                                //   tr(H_+^† X^T diag(c ⊙ X v_k) X) = r^T v_k.
                                let v_k = v_all.column(k_idx);
                                let trace_third = r_third.dot(&v_k);

                                trace_terms.push(trace_h_inv_s_k - trace_third);
                            }

                            // We do NOT need to set workspace.solved_rows as we aren't using the workspace solver.
                            workspace.solved_rows = 0;

                            // Implicit Truncation Correction:
                            // By using H_+^\dagger essentially constructed from U_R D_R^{-1} U_R^T,
                            // we automatically project dS onto the valid subspace P_R.
                            // The phantom spectral bleed term (tr(H^-1 P_N dS P_N)) is identically zero
                            // because P_N H_+^\dagger = 0.
                            let truncation_corrections = vec![0.0; k_count];

                            for k in 0..k_count {
                                let log_det_h_grad_term = 0.5 * lambdas[k] * trace_terms[k];
                                let corrected_log_det_h =
                                    log_det_h_grad_term - truncation_corrections[k];
                                let log_det_s_grad_term = 0.5 * det1_values[k];

                                // Exact LAML gradient assembly for the implemented objective:
                                //   g_k = 0.5 * β̂ᵀ A_k β̂ - 0.5 * tr(S^+ A_k) + 0.5 * tr(H^{-1} H_k)
                                // where A_k = ∂S/∂ρ_k = λ_k S_k and H_k is the total derivative.
                                let gradient_value =
                                    0.5 * beta_terms[k] + corrected_log_det_h - log_det_s_grad_term;
                                laml_grad.push(gradient_value);
                            }
                            workspace
                                .cost_gradient_view(len)
                                .assign(&Array1::from_vec(laml_grad));
                        }
                    }
                }

                if !includes_prior {
                    let (_, prior_grad_view) = workspace.soft_prior_cost_and_grad(p);
                    let prior_grad = prior_grad_view.to_owned();
                    {
                        let mut cost_gradient_view = workspace.cost_gradient_view(len);
                        cost_gradient_view += &prior_grad;
                    }
                }

                // Capture the gradient snapshot before releasing the workspace borrow so
                // that diagnostics can continue without holding the RefCell borrow.
                let gradient_result = workspace.cost_gradient_view_const(len).to_owned();
                let gradient_snapshot = if p.is_empty() {
                    None
                } else {
                    Some(gradient_result.clone())
                };

                (gradient_result, gradient_snapshot, None::<Vec<f64>>)
            };

            // The gradient buffer stored in the workspace already holds -∇V(ρ),
            // which is exactly what the optimizer needs.
            // No final negation is needed.

            // Comprehensive gradient diagnostics (all four strategies)
            if let Some(gradient_snapshot) = gradient_snapshot
                && !p.is_empty()
            {
                // Run all diagnostics and emit a single summary if issues found
                self.run_gradient_diagnostics(p, bundle, &gradient_snapshot, None);
            }

            if self.should_use_stochastic_exact_gradient(bundle, &gradient_result) {
                match self.compute_logit_stochastic_exact_gradient(p, bundle) {
                    Ok(stochastic_grad) => {
                        log::warn!(
                            "[REML] using stochastic exact log-marginal gradient fallback (posterior-sampled expectation)"
                        );
                        return Ok(stochastic_grad);
                    }
                    Err(err) => {
                        log::warn!(
                            "[REML] stochastic exact gradient fallback failed; keeping analytic gradient: {:?}",
                            err
                        );
                    }
                }
            }

            Ok(gradient_result)
        }

        fn should_use_stochastic_exact_gradient(
            &self,
            bundle: &EvalShared,
            gradient: &Array1<f64>,
        ) -> bool {
            // Gate for the posterior-sampled gradient path.
            // This predicate checks for non-finite or unstable analytic states.
            if self.config.link_function() != LinkFunction::Logit {
                return false;
            }
            if self.config.firth_bias_reduction {
                // Firth-adjusted inner objective does not match the plain PG/NUTS posterior target here.
                return false;
            }
            if gradient.is_empty() {
                return false;
            }
            if !gradient.iter().all(|g| g.is_finite()) {
                return true;
            }
            let pirls = bundle.pirls_result.as_ref();
            if matches!(pirls.status, pirls::PirlsStatus::Unstable) {
                return true;
            }
            let kkt_like = pirls.last_gradient_norm;
            if !kkt_like.is_finite() || kkt_like > 1e2 {
                return true;
            }
            let grad_inf = gradient.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
            !grad_inf.is_finite() || grad_inf > 1e9
        }

        fn compute_logit_stochastic_exact_gradient(
            &self,
            p: &Array1<f64>,
            bundle: &EvalShared,
        ) -> Result<Array1<f64>, EstimationError> {
            // Derivation sketch (sign convention used by this minimization objective):
            //
            // 1) Penalized evidence identity (logit):
            //      Z(ρ) = ∫ exp(l(β) - 0.5 βᵀS(ρ)β) dβ,   S(ρ)=Σ_j exp(ρ_j) S_j.
            //
            // 2) Fisher/PG identity for each coordinate:
            //      ∂/∂ρ_k log Z(ρ) = -0.5 * λ_k * E_{π(β|y,ρ)}[βᵀ S_k β],   λ_k=exp(ρ_k).
            //
            // 3) This code optimizes a cost that includes the pseudo-determinant
            //    normalization of the improper Gaussian penalty, yielding:
            //      g_k = ∂Cost/∂ρ_k
            //          = 0.5 * λ_k * E[βᵀS_kβ] - 0.5 * λ_k * tr(S(ρ)^+ S_k).
            //
            // 4) Root-factor rewrite used numerically:
            //      S_k = R_kᵀR_k  =>  βᵀS_kβ = ||R_kβ||².
            //
            // 5) Implementation mapping:
            //      PG-Rao-Blackwell average of tr(S_kQ^{-1})+μᵀS_kμ -> E[βᵀS_kβ],
            //      det1_values[k]                                 -> λ_k tr(S(ρ)^+S_k),
            //      grad[k]                                        -> g_k.
            // Equation-to-code map for this fallback path (logit, fixed ρ):
            //   g_k := ∂Cost/∂ρ_k
            //      = 0.5 * λ_k * E_{π(β|y,ρ)}[βᵀ S_k β]
            //        - 0.5 * λ_k * tr(S(ρ)^+ S_k),
            //   λ_k = exp(ρ_k).
            //
            // The first expectation is evaluated by PG Gibbs + Rao-Blackwellization.
            // The second term is deterministic via structural pseudo-logdet derivatives.
            let pirls_result = bundle.pirls_result.as_ref();
            let beta_mode = pirls_result.beta_transformed.as_ref();
            let s_transformed = &pirls_result.reparam_result.s_transformed;
            let x_arc = pirls_result.x_transformed.to_dense_arc();
            let x_dense = x_arc.as_ref();
            let y = self.y;
            let weights = self.weights;
            let h_eff = bundle.h_eff.as_ref();

            // PG-Gibbs Rao-Blackwell fallback: fewer samples are needed than β-NUTS
            // because each retained ω state contributes the analytic conditional moment
            // tr(S_k Q^{-1}) + μᵀ S_k μ instead of a raw quadratic draw.
            let pg_cfg = crate::hmc::NutsConfig {
                n_samples: 24,
                n_warmup: 48,
                n_chains: 2,
                target_accept: 0.85,
                seed: 17_391,
            };

            let len = p.len();
            let mut lambda = Array1::<f64>::zeros(len);
            for k in 0..len {
                // Outer parameters are ρ; penalties are λ = exp(ρ).
                lambda[k] = p[k].exp();
            }

            let (det1_values, _) = self.structural_penalty_logdet_derivatives(
                &pirls_result.reparam_result.rs_transformed,
                &lambda,
                pirls_result.reparam_result.e_transformed.nrows(),
                bundle.ridge_passport.penalty_logdet_ridge(),
            )?;
            // det1_values[k] = ∂ log|S(ρ)|_+ / ∂ρ_k = λ_k tr(S(ρ)^+ S_k).

            let rb_terms_result = crate::hmc::estimate_logit_pg_rao_blackwell_terms(
                x_dense.view(),
                y,
                weights,
                s_transformed.view(),
                beta_mode.view(),
                &pirls_result.reparam_result.rs_transformed,
                &pg_cfg,
            );

            let mut grad = Array1::<f64>::zeros(len);
            match rb_terms_result {
                Ok(rb_terms) => {
                    for k in 0..len {
                        // Rao-Blackwellized exact identity:
                        //   g_k = 0.5 * λ_k * E_ω[ tr(S_k Q^{-1}) + μᵀ S_k μ ] - 0.5 * det1_values[k].
                        grad[k] = 0.5 * lambda[k] * rb_terms[k] - 0.5 * det1_values[k];
                    }
                }
                Err(err) => {
                    log::warn!(
                        "[REML] PG Rao-Blackwell fallback failed ({}); reverting to NUTS beta averaging",
                        err
                    );

                    let nuts_cfg = crate::hmc::NutsConfig {
                        n_samples: 120,
                        n_warmup: 160,
                        n_chains: 2,
                        target_accept: 0.85,
                        seed: 17_391,
                    };

                    let nuts_result = crate::hmc::run_nuts_sampling_flattened_family(
                        crate::types::LikelihoodFamily::BinomialLogit,
                        crate::hmc::FamilyNutsInputs::Glm(crate::hmc::GlmFlatInputs {
                            x: x_dense.view(),
                            y,
                            weights,
                            penalty_matrix: s_transformed.view(),
                            mode: beta_mode.view(),
                            hessian: h_eff.view(),
                            firth_bias_reduction: self.config.firth_bias_reduction,
                        }),
                        &nuts_cfg,
                    )
                    .map_err(EstimationError::InvalidInput)?;

                    let samples = &nuts_result.samples;
                    let n_draws = samples.nrows().max(1);
                    let mut expected_quad = vec![0.0_f64; len];
                    for draw in 0..samples.nrows() {
                        let beta_draw = samples.row(draw).to_owned();
                        for k in 0..len {
                            let r_k = &pirls_result.reparam_result.rs_transformed[k];
                            let r_beta = r_k.dot(&beta_draw);
                            expected_quad[k] += r_beta.dot(&r_beta);
                        }
                    }
                    let inv_draws = 1.0 / (n_draws as f64);
                    for v in &mut expected_quad {
                        *v *= inv_draws;
                    }
                    for k in 0..len {
                        grad[k] = 0.5 * lambda[k] * expected_quad[k] - 0.5 * det1_values[k];
                    }
                }
            }
            grad += &self.compute_soft_prior_grad(p);
            Ok(grad)
        }

        fn xt_diag_x_dense_into(
            x: &Array2<f64>,
            diag: &Array1<f64>,
            weighted: &mut Array2<f64>,
        ) -> Array2<f64> {
            let n = x.nrows();
            weighted.assign(x);
            for i in 0..n {
                let w = diag[i];
                for j in 0..x.ncols() {
                    weighted[[i, j]] *= w;
                }
            }
            fast_atb(x, weighted)
        }

        fn trace_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
            debug_assert_eq!(a.nrows(), b.ncols());
            debug_assert_eq!(a.ncols(), b.nrows());
            let elems = a.nrows().saturating_mul(a.ncols());
            if elems >= 32 * 32 {
                let a_view = FaerArrayView::new(a);
                let b_view = FaerArrayView::new(b);
                return faer_frob_inner(a_view.as_ref(), b_view.as_ref().transpose());
            }
            let m = a.nrows();
            let n = a.ncols();
            kahan_sum((0..m).map(|i| {
                let mut acc = 0.0_f64;
                for j in 0..n {
                    acc += a[[i, j]] * b[[j, i]];
                }
                acc
            }))
        }

        fn bilinear_form(
            mat: &Array2<f64>,
            left: ndarray::ArrayView1<'_, f64>,
            right: ndarray::ArrayView1<'_, f64>,
        ) -> f64 {
            let n = mat.nrows();
            debug_assert_eq!(mat.ncols(), n);
            debug_assert_eq!(left.len(), n);
            debug_assert_eq!(right.len(), n);
            let mut acc = KahanSum::default();
            for i in 0..n {
                let mut row_dot = 0.0_f64;
                for j in 0..n {
                    row_dot += mat[[i, j]] * right[j];
                }
                acc.add(left[i] * row_dot);
            }
            acc.sum()
        }

        fn select_trace_backend(n_obs: usize, p_dim: usize, k_count: usize) -> TraceBackend {
            // Workload-aware policy driven by (n, p, K):
            // - Exact for moderate total complexity.
            // - Hutchinson/Hutch++ as n·p·K and p²·K² costs grow.
            //
            // Proxies:
            //   w_npk   ~ n*p*K   (X/Xᵀ + diagonal contractions)
            //   w_pk2   ~ p*K²    (pairwise rho-Hessian assembly)
            let k = k_count.max(1);
            let w_npk = (n_obs as u128)
                .saturating_mul(p_dim as u128)
                .saturating_mul(k as u128);
            let w_pk2 = (p_dim as u128).saturating_mul((k as u128).saturating_mul(k as u128));

            if p_dim <= 700 && k <= 20 && w_npk <= 220_000_000 && w_pk2 <= 20_000_000 {
                return TraceBackend::Exact;
            }

            let very_large =
                p_dim >= 1_800 || k >= 28 || w_npk >= 1_100_000_000 || w_pk2 >= 85_000_000;
            if very_large {
                let sketch = if p_dim >= 3_500 || w_npk >= 2_500_000_000 {
                    12
                } else {
                    8
                };
                let probes = if k >= 36 || w_pk2 >= 150_000_000 {
                    28
                } else {
                    22
                };
                return TraceBackend::HutchPP { probes, sketch };
            }

            let probes = if w_npk >= 700_000_000 || k >= 24 {
                34
            } else if w_npk >= 350_000_000 {
                28
            } else {
                22
            };
            TraceBackend::Hutchinson { probes }
        }

        #[inline]
        fn splitmix64(mut x: u64) -> u64 {
            x = x.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }

        fn rademacher_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
            let mut out = Array2::<f64>::zeros((rows, cols));
            for j in 0..cols {
                for i in 0..rows {
                    let h = Self::splitmix64(
                        seed ^ ((i as u64).wrapping_mul(0x9E37))
                            ^ ((j as u64).wrapping_mul(0x85EB)),
                    );
                    out[[i, j]] = if (h & 1) == 0 { -1.0 } else { 1.0 };
                }
            }
            out
        }

        fn orthonormalize_columns(a: &Array2<f64>, tol: f64) -> Array2<f64> {
            let p = a.nrows();
            let c = a.ncols();
            let mut q = Array2::<f64>::zeros((p, c));
            let mut kept = 0usize;
            for j in 0..c {
                let mut v = a.column(j).to_owned();
                for t in 0..kept {
                    let qt = q.column(t);
                    let proj = qt.dot(&v);
                    v -= &qt.mapv(|x| x * proj);
                }
                let nrm = v.dot(&v).sqrt();
                if nrm > tol {
                    q.column_mut(kept).assign(&v.mapv(|x| x / nrm));
                    kept += 1;
                }
            }
            if kept == c {
                q
            } else {
                q.slice(ndarray::s![.., 0..kept]).to_owned()
            }
        }

        fn structural_penalty_logdet_derivatives(
            &self,
            rs_transformed: &[Array2<f64>],
            lambdas: &Array1<f64>,
            structural_rank: usize,
            ridge: f64,
        ) -> Result<(Array1<f64>, Array2<f64>), EstimationError> {
            let k_count = lambdas.len();
            let p_dim = self.p;
            if k_count == 0 || structural_rank == 0 {
                return Ok((Array1::zeros(k_count), Array2::zeros((k_count, k_count))));
            }
            let rank = structural_rank.min(p_dim);
            if rank == 0 {
                return Ok((Array1::zeros(k_count), Array2::zeros((k_count, k_count))));
            }

            let mut s_k_full = Vec::with_capacity(k_count);
            let mut s_lambda = Array2::<f64>::zeros((p_dim, p_dim));
            for k in 0..k_count {
                let r_k = &rs_transformed[k];
                let s_k = r_k.t().dot(r_k);
                s_lambda += &s_k.mapv(|v| lambdas[k] * v);
                s_k_full.push(s_k);
            }
            if ridge > 0.0 {
                for i in 0..p_dim {
                    s_lambda[[i, i]] += ridge;
                }
            }

            let (evals, evecs) = s_lambda
                .eigh(Side::Lower)
                .map_err(EstimationError::EigendecompositionFailed)?;
            let mut order: Vec<usize> = (0..p_dim).collect();
            order.sort_by(|&a, &b| {
                evals[b]
                    .partial_cmp(&evals[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then(a.cmp(&b))
            });

            let mut u1 = Array2::<f64>::zeros((p_dim, rank));
            for (col_out, &col_in) in order.iter().take(rank).enumerate() {
                u1.column_mut(col_out).assign(&evecs.column(col_in));
            }
            let mut s_r = u1.t().dot(&s_lambda).dot(&u1);
            let max_diag = s_r
                .diag()
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max)
                .max(1.0);
            let eps = 1e-12 * max_diag;
            for i in 0..rank {
                s_r[[i, i]] += eps;
            }
            let s_r_inv = matrix_inverse_with_regularization(&s_r, "structural penalty block")
                .ok_or_else(|| EstimationError::ModelIsIllConditioned {
                    condition_number: f64::INFINITY,
                })?;

            let mut s_k_reduced = Vec::with_capacity(k_count);
            let mut det1 = Array1::<f64>::zeros(k_count);
            for k in 0..k_count {
                let s_kr = u1.t().dot(&s_k_full[k]).dot(&u1);
                let tr = kahan_sum((0..rank).map(|i| {
                    let mut acc = 0.0;
                    for j in 0..rank {
                        acc += s_r_inv[[i, j]] * s_kr[[j, i]];
                    }
                    acc
                }));
                det1[k] = lambdas[k] * tr;
                s_k_reduced.push(s_kr);
            }

            let mut det2 = Array2::<f64>::zeros((k_count, k_count));
            for k in 0..k_count {
                for l in 0..=k {
                    let a = s_r_inv.dot(&s_k_reduced[k]);
                    let b = s_r_inv.dot(&s_k_reduced[l]);
                    let tr_ab = kahan_sum((0..rank).map(|i| {
                        let mut acc = 0.0;
                        for j in 0..rank {
                            acc += a[[i, j]] * b[[j, i]];
                        }
                        acc
                    }));
                    let mut val = -lambdas[k] * lambdas[l] * tr_ab;
                    if k == l {
                        val += det1[k];
                    }
                    det2[[k, l]] = val;
                    det2[[l, k]] = val;
                }
            }
            Ok((det1, det2))
        }

        pub(super) fn compute_laml_hessian_exact(
            &self,
            rho: &Array1<f64>,
        ) -> Result<Array2<f64>, EstimationError> {
            // Exact non-Gaussian outer Hessian components (ρ-space):
            //
            //   B_k   = ∂β̂/∂ρ_k = -H^{-1}(A_k β̂)
            //   B_{kℓ}= ∂²β̂/(∂ρ_k∂ρ_ℓ) from
            //          H B_{kℓ} = -(H_ℓ B_k + A_k B_ℓ + δ_{kℓ} A_k β̂)
            //
            //   H_k   = A_k + Xᵀ diag(c ⊙ u_k) X,        u_k   = X B_k
            //   H_{kℓ}= δ_{kℓ}A_k + Xᵀ diag(d ⊙ u_k ⊙ u_ℓ + c ⊙ u_{kℓ}) X,
            //           where u_{kℓ}=X B_{kℓ}
            //
            // Here `c` and `d` are the per-observation 3rd/4th eta-derivative arrays
            // prepared by PIRLS (`solve_c_array`, `solve_d_array`).
            //
            // Full exact Hessian entry used below:
            //
            //   ∂²V/(∂ρ_k∂ρ_ℓ) = Q_{kℓ} + L_{kℓ} + P_{kℓ}
            //
            // with
            //   Q_{kℓ} = B_ℓᵀ A_k β̂ + 0.5 δ_{kℓ} β̂ᵀ A_k β̂
            //   L_{kℓ} = 0.5 [ -tr(H^{-1}H_ℓ H^{-1}H_k) + tr(H^{-1}H_{kℓ}) ]
            //   P_{kℓ} = -0.5 ∂² log|S|_+ /(∂ρ_k∂ρ_ℓ)
            //
            // Numerically, this function computes:
            // - Q exactly from B_k solves,
            // - P exactly from reduced-penalty logdet derivatives,
            // - L either exactly or stochastically, depending on workload.
            //
            // Stochastic trace identities used when backend != Exact:
            //   tr(A) = E[zᵀAz],  z_i∈{±1}.
            //   tr(H^{-1}H_ℓH^{-1}H_k) estimated by shared-probe contractions.
            //   tr(H^{-1}H_{kℓ}) estimated by probe bilinear forms.
            // Hutch++ augments this with a low-rank deflation subspace Q to reduce
            // variance before Hutchinson residual estimation.
            let bundle = self.obtain_eval_bundle(rho)?;
            let pirls_result = bundle.pirls_result.as_ref();
            let beta = pirls_result.beta_transformed.as_ref();
            let reparam_result = &pirls_result.reparam_result;
            let rs_transformed = &reparam_result.rs_transformed;
            let h_total = bundle.h_total.as_ref();
            let h_factor = self.get_faer_factor(rho, h_total);
            let solve_h = |rhs: &Array2<f64>| -> Array2<f64> {
                let mut out = rhs.clone();
                let mut out_view = array2_to_mat_mut(&mut out);
                h_factor.solve_in_place(out_view.as_mut());
                out
            };

            let k_count = rho.len();
            if k_count == 0 {
                return Ok(Array2::zeros((0, 0)));
            }
            let lambdas = rho.mapv(f64::exp);
            let x_dense_arc = pirls_result.x_transformed.to_dense_arc();
            let x_dense = x_dense_arc.as_ref();
            let n = x_dense.nrows();
            let p_dim = x_dense.ncols();
            let c = &pirls_result.solve_c_array;
            let d = &pirls_result.solve_d_array;
            if c.len() != n || d.len() != n {
                return Err(EstimationError::InvalidInput(format!(
                    "Exact Hessian derivative arrays size mismatch: n={}, c.len()={}, d.len()={}",
                    n,
                    c.len(),
                    d.len()
                )));
            }

            let mut a_k_mats = Vec::with_capacity(k_count);
            let mut a_k_beta = Vec::with_capacity(k_count);
            let mut rhs_bk = Array2::<f64>::zeros((p_dim, k_count));
            let mut q_diag = vec![0.0; k_count];
            for k in 0..k_count {
                let r_k = &rs_transformed[k];
                let s_k = r_k.t().dot(r_k);
                let r_beta = r_k.dot(beta);
                let s_k_beta = r_k.t().dot(&r_beta);
                let a_k = s_k.mapv(|v| lambdas[k] * v);
                let a_kb = s_k_beta.mapv(|v| lambdas[k] * v);
                q_diag[k] = beta.dot(&a_kb);
                rhs_bk.column_mut(k).assign(&a_kb.mapv(|v| -v));
                a_k_mats.push(a_k);
                a_k_beta.push(a_kb);
            }

            let b_mat = solve_h(&rhs_bk);
            let u_mat = fast_ab(x_dense, &b_mat);

            let mut h_k = Vec::with_capacity(k_count);
            let mut weighted_xtdx = Array2::<f64>::zeros(x_dense.raw_dim());
            for k in 0..k_count {
                let mut diag = Array1::<f64>::zeros(n);
                for i in 0..n {
                    diag[i] = c[i] * u_mat[[i, k]];
                }
                let mut hk = a_k_mats[k].clone();
                hk += &Self::xt_diag_x_dense_into(x_dense, &diag, &mut weighted_xtdx);
                h_k.push(hk);
            }
            let s_cols: Vec<Array1<f64>> = (0..k_count)
                .map(|k| {
                    let mut s = Array1::<f64>::zeros(n);
                    for i in 0..n {
                        s[i] = c[i] * u_mat[[i, k]];
                    }
                    s
                })
                .collect();

            let trace_backend = Self::select_trace_backend(n, p_dim, k_count);
            let (exact_trace_mode, n_probe, n_sketch) = match trace_backend {
                TraceBackend::Exact => (true, 0usize, 0usize),
                TraceBackend::Hutchinson { probes } => (false, probes.max(1), 0usize),
                TraceBackend::HutchPP { probes, sketch } => (false, probes.max(1), sketch.max(1)),
            };
            let use_hutchpp = matches!(trace_backend, TraceBackend::HutchPP { .. });
            // Backend semantics:
            // - Exact: deterministic traces via explicit H^{-1} contractions.
            // - Hutchinson/Hutch++: Monte-Carlo trace estimators (unbiased/low-bias in
            //   expectation) trading tiny stochastic noise for major scaling gains.

            let h_inv = if exact_trace_mode {
                Some(solve_h(&Array2::<f64>::eye(p_dim)))
            } else {
                None
            };
            let m_k: Option<Vec<Array2<f64>>> = h_inv
                .as_ref()
                .map(|hinv| h_k.iter().map(|hk| hinv.dot(hk)).collect());

            let mut probe_z: Option<Array2<f64>> = None;
            let mut probe_u: Option<Array2<f64>> = None;
            let mut probe_xz: Option<Array2<f64>> = None;
            let mut probe_xu: Option<Array2<f64>> = None;
            let mut sketch_q: Option<Array2<f64>> = None;
            let mut sketch_uq: Option<Array2<f64>> = None;
            let mut sketch_xq: Option<Array2<f64>> = None;
            let mut sketch_xuq: Option<Array2<f64>> = None;

            if !exact_trace_mode {
                let mut z = Self::rademacher_matrix(p_dim, n_probe, 0xC0DEC0DE5EEDu64);
                if use_hutchpp && n_sketch > 0 {
                    let g = Self::rademacher_matrix(p_dim, n_sketch, 0xBADC0FFEE0DDF00Du64);
                    let y = solve_h(&g);
                    let q = Self::orthonormalize_columns(&y, 1e-10);
                    if q.ncols() > 0 {
                        for r in 0..n_probe {
                            let mut zr = z.column(r).to_owned();
                            let qt_z = q.t().dot(&zr);
                            let proj = q.dot(&qt_z);
                            zr -= &proj;
                            z.column_mut(r).assign(&zr);
                        }
                        let uq = solve_h(&q);
                        let xq = fast_ab(x_dense, &q);
                        let xuq = fast_ab(x_dense, &uq);
                        sketch_q = Some(q);
                        sketch_uq = Some(uq);
                        sketch_xq = Some(xq);
                        sketch_xuq = Some(xuq);
                    }
                }
                let u = solve_h(&z);
                let xz = fast_ab(x_dense, &z);
                let xu = fast_ab(x_dense, &u);
                probe_z = Some(z);
                probe_u = Some(u);
                probe_xz = Some(xz);
                probe_xu = Some(xu);
            }

            let mut t1_mat = Array2::<f64>::zeros((k_count, k_count));
            if exact_trace_mode {
                let mk = m_k.as_ref().expect("m_k present in exact mode");
                for l in 0..k_count {
                    for k in 0..k_count {
                        t1_mat[[l, k]] = Self::trace_product(&mk[l], &mk[k]);
                    }
                }
            } else {
                if let (Some(q), Some(uq), Some(xq), Some(xuq)) = (
                    sketch_q.as_ref(),
                    sketch_uq.as_ref(),
                    sketch_xq.as_ref(),
                    sketch_xuq.as_ref(),
                ) {
                    let rdim = q.ncols();
                    for j in 0..rdim {
                        let qj = q.column(j).to_owned();
                        let uqj = uq.column(j).to_owned();
                        let xqj = xq.column(j).to_owned();
                        let xuqj = xuq.column(j).to_owned();
                        let mut bq = Array2::<f64>::zeros((p_dim, k_count));
                        for k in 0..k_count {
                            let mut hkq = a_k_mats[k].dot(&qj);
                            let weighted = &s_cols[k] * &xqj;
                            hkq += &x_dense.t().dot(&weighted);
                            bq.column_mut(k).assign(&hkq);
                        }
                        let wq = solve_h(&bq);
                        let xwq = fast_ab(x_dense, &wq);
                        for l in 0..k_count {
                            let alu = a_k_mats[l].dot(&uqj);
                            let sxu = &s_cols[l] * &xuqj;
                            for k in 0..k_count {
                                let val = alu.dot(&wq.column(k)) + sxu.dot(&xwq.column(k));
                                t1_mat[[l, k]] += val;
                            }
                        }
                    }
                }
                let z = probe_z.as_ref().expect("probes present in stochastic mode");
                let u = probe_u
                    .as_ref()
                    .expect("solved probes present in stochastic mode");
                let xz = probe_xz
                    .as_ref()
                    .expect("X probes present in stochastic mode");
                let xu = probe_xu
                    .as_ref()
                    .expect("X solved probes present in stochastic mode");
                for r in 0..n_probe {
                    let zr = z.column(r).to_owned();
                    let ur = u.column(r).to_owned();
                    let xzr = xz.column(r).to_owned();
                    let xur = xu.column(r).to_owned();
                    let mut bz = Array2::<f64>::zeros((p_dim, k_count));
                    for k in 0..k_count {
                        let mut hkz = a_k_mats[k].dot(&zr);
                        let weighted = &s_cols[k] * &xzr;
                        hkz += &x_dense.t().dot(&weighted);
                        bz.column_mut(k).assign(&hkz);
                    }
                    let wz = solve_h(&bz);
                    let xwz = fast_ab(x_dense, &wz);
                    for l in 0..k_count {
                        let alu = a_k_mats[l].dot(&ur);
                        let sxu = &s_cols[l] * &xur;
                        for k in 0..k_count {
                            let val = alu.dot(&wz.column(k)) + sxu.dot(&xwz.column(k));
                            t1_mat[[l, k]] += val / (n_probe as f64);
                        }
                    }
                }
            }
            for i in 0..k_count {
                for j in 0..i {
                    let avg = 0.5 * (t1_mat[[i, j]] + t1_mat[[j, i]]);
                    t1_mat[[i, j]] = avg;
                    t1_mat[[j, i]] = avg;
                }
            }

            let (_, d2logs) = self.structural_penalty_logdet_derivatives(
                rs_transformed,
                &lambdas,
                reparam_result.e_transformed.nrows(),
                bundle.ridge_passport.penalty_logdet_ridge(),
            )?;

            let mut hess = Array2::<f64>::zeros((k_count, k_count));
            for l in 0..k_count {
                let bl = b_mat.column(l).to_owned();
                let mut rhs_kl_all = Array2::<f64>::zeros((p_dim, k_count));
                for k in l..k_count {
                    let bk = b_mat.column(k).to_owned();
                    let mut rhs_kl = -h_k[l].dot(&bk);
                    rhs_kl -= &a_k_mats[k].dot(&bl);
                    if k == l {
                        rhs_kl -= &a_k_beta[k];
                    }
                    rhs_kl_all.column_mut(k).assign(&rhs_kl);
                }
                let b_kl_all = solve_h(&rhs_kl_all);
                let u_kl_all = fast_ab(x_dense, &b_kl_all);

                let mut weighted_xtdx_kl = Array2::<f64>::zeros(x_dense.raw_dim());
                for k in l..k_count {
                    let mut diag = Array1::<f64>::zeros(n);
                    for i in 0..n {
                        diag[i] = d[i] * u_mat[[i, k]] * u_mat[[i, l]] + c[i] * u_kl_all[[i, k]];
                    }

                    let q = bl.dot(&a_k_beta[k]) + if k == l { 0.5 * q_diag[k] } else { 0.0 };
                    let t1 = t1_mat[[l, k]];
                    let t2 = if exact_trace_mode {
                        let mut h_kl = if k == l {
                            a_k_mats[k].clone()
                        } else {
                            Array2::<f64>::zeros((p_dim, p_dim))
                        };
                        h_kl += &Self::xt_diag_x_dense_into(x_dense, &diag, &mut weighted_xtdx_kl);
                        let h_inv_ref = h_inv.as_ref().expect("h_inv present in exact mode");
                        Self::trace_product(h_inv_ref, &h_kl)
                    } else {
                        let mut t2_acc = 0.0_f64;
                        if let (Some(q), Some(uq), Some(xq), Some(xuq)) = (
                            sketch_q.as_ref(),
                            sketch_uq.as_ref(),
                            sketch_xq.as_ref(),
                            sketch_xuq.as_ref(),
                        ) {
                            for j in 0..q.ncols() {
                                let qj = q.column(j);
                                let uqj = uq.column(j);
                                let xqj = xq.column(j);
                                let xuqj = xuq.column(j);
                                let mut term = 0.0_f64;
                                if k == l {
                                    term += Self::bilinear_form(&a_k_mats[k], uqj, qj);
                                }
                                let mut quad = 0.0_f64;
                                for i in 0..n {
                                    quad += xuqj[i] * diag[i] * xqj[i];
                                }
                                term += quad;
                                t2_acc += term;
                            }
                        }
                        let z = probe_z.as_ref().expect("probes present in stochastic mode");
                        let u = probe_u
                            .as_ref()
                            .expect("solved probes present in stochastic mode");
                        let xz = probe_xz
                            .as_ref()
                            .expect("X probes present in stochastic mode");
                        let xu = probe_xu
                            .as_ref()
                            .expect("X solved probes present in stochastic mode");
                        let mut res = 0.0_f64;
                        for r in 0..n_probe {
                            let zr = z.column(r);
                            let ur = u.column(r);
                            let xzr = xz.column(r);
                            let xur = xu.column(r);
                            let mut term = 0.0_f64;
                            if k == l {
                                term += Self::bilinear_form(&a_k_mats[k], ur, zr);
                            }
                            let mut quad = 0.0_f64;
                            for i in 0..n {
                                quad += xur[i] * diag[i] * xzr[i];
                            }
                            term += quad;
                            res += term;
                        }
                        t2_acc + res / (n_probe as f64)
                    };
                    let l_term = 0.5 * (-t1 + t2);
                    let p_term = -0.5 * d2logs[[k, l]];
                    let val = q + l_term + p_term;
                    hess[[k, l]] = val;
                    hess[[l, k]] = val;
                }
            }
            Ok(hess)
        }

        pub(super) fn compute_smoothing_correction_auto(
            &self,
            final_rho: &Array1<f64>,
            final_fit: &PirlsResult,
            base_covariance: Option<&Array2<f64>>,
            final_grad_norm: f64,
        ) -> Option<Array2<f64>> {
            // Always compute the fast first-order correction first.
            let first_order = super::compute_smoothing_correction(self, final_rho, final_fit);
            let n_rho = final_rho.len();
            if n_rho == 0 {
                return first_order;
            }
            if n_rho > AUTO_CUBATURE_MAX_RHO_DIM {
                return first_order;
            }
            if final_fit.beta_transformed.len() > AUTO_CUBATURE_MAX_BETA_DIM {
                return first_order;
            }

            let near_boundary = final_rho
                .iter()
                .any(|&v| (RHO_BOUND - v.abs()) <= AUTO_CUBATURE_BOUNDARY_MARGIN);
            let grad_norm = if final_grad_norm.is_finite() {
                final_grad_norm
            } else {
                0.0
            };
            let high_grad = grad_norm > 1e-3;
            if !near_boundary && !high_grad {
                // Keep the hot path cheap when the local linearization is likely sufficient.
                return first_order;
            }

            // Build V_rho from the outer Hessian around rho_hat.
            let mut hessian_rho = match self.compute_laml_hessian_exact(final_rho) {
                Ok(h) => h,
                Err(err) => {
                    log::debug!(
                        "Auto cubature skipped: exact rho Hessian unavailable ({}).",
                        err
                    );
                    return first_order;
                }
            };
            for i in 0..n_rho {
                for j in (i + 1)..n_rho {
                    let avg = 0.5 * (hessian_rho[[i, j]] + hessian_rho[[j, i]]);
                    hessian_rho[[i, j]] = avg;
                    hessian_rho[[j, i]] = avg;
                }
            }
            let ridge = 1e-8
                * hessian_rho
                    .diag()
                    .iter()
                    .map(|&v| v.abs())
                    .fold(0.0, f64::max)
                    .max(1e-8);
            for i in 0..n_rho {
                hessian_rho[[i, i]] += ridge;
            }
            let hessian_rho_inv =
                match matrix_inverse_with_regularization(&hessian_rho, "auto cubature rho Hessian")
                {
                    Some(v) => v,
                    None => return first_order,
                };

            let max_rho_var = hessian_rho_inv
                .diag()
                .iter()
                .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
            if !near_boundary && !high_grad && max_rho_var < 0.1 {
                return first_order;
            }

            use crate::faer_ndarray::FaerEigh;
            use faer::Side;
            let (evals, evecs) = match hessian_rho_inv.eigh(Side::Lower) {
                Ok(x) => x,
                Err(_) => return first_order,
            };
            let mut eig_pairs: Vec<(usize, f64)> = evals
                .iter()
                .copied()
                .enumerate()
                .filter(|(_, v)| v.is_finite() && *v > 1e-12)
                .collect();
            if eig_pairs.is_empty() {
                return first_order;
            }
            eig_pairs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let total_var: f64 = eig_pairs.iter().map(|(_, v)| *v).sum();
            if !total_var.is_finite() || total_var <= 0.0 {
                return first_order;
            }

            let mut rank = 0usize;
            let mut captured = 0.0_f64;
            for (_, eig) in eig_pairs
                .iter()
                .take(AUTO_CUBATURE_MAX_EIGENVECTORS.min(eig_pairs.len()))
            {
                captured += *eig;
                rank += 1;
                if captured / total_var >= AUTO_CUBATURE_TARGET_VAR_FRAC {
                    break;
                }
            }
            if rank == 0 {
                return first_order;
            }

            let base_cov = match base_covariance {
                Some(v) => v,
                None => return first_order,
            };
            let p = base_cov.nrows();
            let radius = (rank as f64).sqrt();
            let mut sigma_points: Vec<Array1<f64>> = Vec::with_capacity(2 * rank);
            for (eig_idx, eig_val) in eig_pairs.iter().take(rank) {
                let axis = evecs.column(*eig_idx).to_owned();
                let scale = radius * eig_val.sqrt();
                let delta = axis.mapv(|v| v * scale);

                for sign in [1.0_f64, -1.0_f64] {
                    let mut rho_point = final_rho.clone();
                    for i in 0..n_rho {
                        rho_point[i] = (rho_point[i] + sign * delta[i])
                            .clamp(-RHO_BOUND + 1e-8, RHO_BOUND - 1e-8);
                    }
                    sigma_points.push(rho_point);
                }
            }
            if sigma_points.is_empty() {
                return first_order;
            }

            // Disable warm-start coupling while evaluating sigma points in parallel.
            // This keeps point evaluations independent and deterministic.
            struct WarmStartRestoreGuard<'a> {
                flag: &'a AtomicBool,
                prev: bool,
            }
            impl Drop for WarmStartRestoreGuard<'_> {
                fn drop(&mut self) {
                    self.flag.store(self.prev, Ordering::SeqCst);
                }
            }
            let prev_warm_start = self.warm_start_enabled.swap(false, Ordering::SeqCst);
            let _warm_start_guard = WarmStartRestoreGuard {
                flag: &self.warm_start_enabled,
                prev: prev_warm_start,
            };
            let point_results: Vec<Option<(Array2<f64>, Array1<f64>)>> = (0..sigma_points.len())
                .into_par_iter()
                .map(|idx| {
                    let fit_point = self.execute_pirls_if_needed(&sigma_points[idx]).ok()?;
                    let h_point = map_hessian_to_original_basis(fit_point.as_ref()).ok()?;
                    let cov_point =
                        matrix_inverse_with_regularization(&h_point, "auto cubature point")?;
                    let beta_point = fit_point
                        .reparam_result
                        .qs
                        .dot(fit_point.beta_transformed.as_ref());
                    Some((cov_point, beta_point))
                })
                .collect();

            if point_results.iter().any(|r| r.is_none()) {
                return first_order;
            }

            let w = 1.0 / (sigma_points.len() as f64);
            let mut mean_hinv = Array2::<f64>::zeros((p, p));
            let mut mean_beta = Array1::<f64>::zeros(p);
            let mut second_beta = Array2::<f64>::zeros((p, p));
            for (cov_point, beta_point) in point_results.into_iter().flatten() {
                mean_hinv += &cov_point.mapv(|v| w * v);
                mean_beta += &beta_point.mapv(|v| w * v);
                for i in 0..p {
                    let bi = beta_point[i];
                    for j in 0..p {
                        second_beta[[i, j]] += w * bi * beta_point[j];
                    }
                }
            }

            let mut var_beta = second_beta;
            for i in 0..p {
                for j in 0..p {
                    var_beta[[i, j]] -= mean_beta[i] * mean_beta[j];
                }
            }

            let mut total_cov = mean_hinv + var_beta;
            for i in 0..p {
                for j in (i + 1)..p {
                    let avg = 0.5 * (total_cov[[i, j]] + total_cov[[j, i]]);
                    total_cov[[i, j]] = avg;
                    total_cov[[j, i]] = avg;
                }
            }
            if !total_cov.iter().all(|v| v.is_finite()) {
                return first_order;
            }

            let mut corr = total_cov - base_cov;
            for i in 0..p {
                for j in (i + 1)..p {
                    let avg = 0.5 * (corr[[i, j]] + corr[[j, i]]);
                    corr[[i, j]] = avg;
                    corr[[j, i]] = avg;
                }
            }

            log::info!(
                "Using adaptive cubature smoothing correction (rank={}, points={}, near_boundary={}, grad_norm={:.2e}, max_var={:.2e})",
                rank,
                2 * rank,
                near_boundary,
                grad_norm,
                max_rho_var
            );
            Some(corr)
        }

        /// Run comprehensive gradient diagnostics implementing four strategies:
        /// 1. KKT/Envelope Theorem Audit
        /// 2. Component-wise Finite Difference
        /// 3. Spectral Bleed Trace
        /// 4. Dual-Ridge Consistency
        ///
        /// Only prints a summary when issues are detected.
        fn run_gradient_diagnostics(
            &self,
            rho: &Array1<f64>,
            bundle: &EvalShared,
            analytic_grad: &Array1<f64>,
            applied_truncation_corrections: Option<&[f64]>,
        ) {
            use crate::diagnostics::{
                DiagnosticConfig, GradientDiagnosticReport, compute_dual_ridge_check,
                compute_envelope_audit, compute_spectral_bleed,
            };

            let config = DiagnosticConfig::default();
            let mut report = GradientDiagnosticReport::new();

            let pirls_result = bundle.pirls_result.as_ref();
            let ridge_used = bundle.ridge_passport.delta;
            let beta = pirls_result.beta_transformed.as_ref();
            let lambdas: Array1<f64> = rho.mapv(f64::exp);

            // === Strategy 4: Dual-Ridge Consistency Check ===
            // Compare the PIRLS ridge with the ridge used by cost/gradient paths.
            let dual_ridge = compute_dual_ridge_check(
                pirls_result.ridge_passport.delta, // Ridge from PIRLS passport
                ridge_used,                        // Ridge passed to cost
                ridge_used,                        // Ridge passed to gradient (same bundle)
                beta,
            );
            report.dual_ridge = Some(dual_ridge);

            // === Strategy 1: KKT/Envelope Theorem Audit ===
            // Check if the inner solver actually reached stationarity
            let reparam = &pirls_result.reparam_result;
            let penalty_grad = reparam.s_transformed.dot(beta);

            let envelope_audit = compute_envelope_audit(
                pirls_result.last_gradient_norm,
                &penalty_grad,
                pirls_result.ridge_passport.delta,
                ridge_used, // What gradient assumes
                beta,
                config.kkt_tolerance,
                config.rel_error_threshold,
            );
            report.envelope_audit = Some(envelope_audit);

            // === Strategy 3: Spectral Bleed Trace ===
            // Check if truncated eigenspace corrections are adequate
            let u_truncated = &reparam.u_truncated;
            let truncated_count = u_truncated.ncols();

            if truncated_count > 0 {
                let h_eff = bundle.h_eff.as_ref();

                // Solve H⁻¹ U_⊥ for spectral bleed calculation
                let h_view = FaerArrayView::new(h_eff);
                if let Ok(chol) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
                    let mut h_inv_u = u_truncated.clone();
                    let mut rhs_view = array2_to_mat_mut(&mut h_inv_u);
                    chol.solve_in_place(rhs_view.as_mut());

                    for (k, r_k) in reparam.rs_transformed.iter().enumerate() {
                        let applied_correction = applied_truncation_corrections
                            .and_then(|values| values.get(k).copied())
                            .unwrap_or(0.0);
                        let bleed = compute_spectral_bleed(
                            k,
                            r_k.view(),
                            u_truncated.view(),
                            h_inv_u.view(),
                            lambdas[k],
                            applied_correction,
                            config.rel_error_threshold,
                        );
                        if bleed.has_bleed || bleed.truncated_energy.abs() > 1e-4 {
                            report.spectral_bleed.push(bleed);
                        }
                    }
                }
            }

            // === Strategy 2: Component-wise FD (only if we detected other issues) ===
            // This is expensive, so only do it when other diagnostics flag problems
            if report.has_issues() {
                let h = config.fd_step_size;
                let mut numeric_grad = Array1::<f64>::zeros(rho.len());

                for k in 0..rho.len() {
                    let mut rho_plus = rho.clone();
                    rho_plus[k] += h;
                    let mut rho_minus = rho.clone();
                    rho_minus[k] -= h;

                    let fp = self.compute_cost(&rho_plus).unwrap_or(f64::INFINITY);
                    let fm = self.compute_cost(&rho_minus).unwrap_or(f64::INFINITY);
                    numeric_grad[k] = (fp - fm) / (2.0 * h);
                }

                report.analytic_gradient = Some(analytic_grad.clone());
                report.numeric_gradient = Some(numeric_grad.clone());

                // Compute per-component relative errors
                let mut rel_errors = Array1::<f64>::zeros(rho.len());
                for k in 0..rho.len() {
                    let denom = analytic_grad[k].abs().max(numeric_grad[k].abs()).max(1e-8);
                    rel_errors[k] = (analytic_grad[k] - numeric_grad[k]).abs() / denom;
                }
                report.component_rel_errors = Some(rel_errors);
            }

            // === Output Summary (single print, not in a loop) ===
            if report.has_issues() {
                println!("\n[GRADIENT DIAGNOSTICS] Issues detected:");
                println!("{}", report.summary());

                // Also log total gradient comparison
                if let (Some(analytic), Some(numeric)) =
                    (&report.analytic_gradient, &report.numeric_gradient)
                {
                    let diff = analytic - numeric;
                    let rel_l2 = diff.dot(&diff).sqrt() / numeric.dot(numeric).sqrt().max(1e-8);
                    println!(
                        "[GRADIENT DIAGNOSTICS] Total gradient rel. L2 error: {:.2e}",
                        rel_l2
                    );
                }
            }
        }

        /// Implements the stable re-parameterization algorithm from Wood (2011) Appendix B
        /// This replaces naive summation S_λ = Σ λᵢSᵢ with similarity transforms
        /// to avoid "dominant machine zero leakage" between penalty components
        ///
        // Helper for boundary perturbation
        // Returns (perturbed_rho, optional_corrected_covariance_in_transformed_basis)
        // The covariance is V'_beta_trans
        #[allow(dead_code)]
        pub(super) fn perform_boundary_perturbation_correction(
            &self,
            initial_rho: &Array1<f64>,
        ) -> Result<(Array1<f64>, Option<Array2<f64>>), EstimationError> {
            // 1. Identify boundary parameters and perturb
            let mut current_rho = initial_rho.clone();
            let mut perturbed = false;

            // Target cost increase: 0.01 log-likelihood units (statistically insignificant)
            let target_diff = 0.01;

            for k in 0..current_rho.len() {
                // Check if at upper boundary (high smoothing -> linear)
                // RHO_BOUND is 30.0.
                if current_rho[k] > RHO_BOUND - 1.0 {
                    // Compute base_cost fresh for each parameter to handle multiple boundary cases
                    let base_cost = self.compute_cost(&current_rho)?;

                    log::info!(
                        "[Boundary] rho[{}] = {:.2} is at boundary. Perturbing...",
                        k,
                        current_rho[k]
                    );

                    // Search inwards (decreasing rho)
                    // We want delta > 0 such that Cost(rho - delta) approx Base + 0.01
                    let mut lower = 0.0;
                    let mut upper = 15.0;
                    let mut best_delta = 0.0;

                    // Initial check: if upper is not enough, just take upper
                    let mut rho_test = current_rho.clone();
                    rho_test[k] -= upper;
                    if let Ok(c) = self.compute_cost(&rho_test) {
                        if (c - base_cost).abs() < target_diff {
                            // Even big change doesn't change cost much?
                            // This implies extremely flat surface. Just move away from boundary significantly.
                            best_delta = upper;
                        }
                    }

                    if best_delta == 0.0 {
                        // Bisection
                        for _ in 0..15 {
                            let mid = (lower + upper) * 0.5;
                            rho_test[k] = current_rho[k] - mid;
                            if let Ok(c) = self.compute_cost(&rho_test) {
                                let diff = c - base_cost;
                                if diff < target_diff {
                                    // Need more change -> larger delta
                                    lower = mid;
                                } else {
                                    // Too much change -> smaller delta
                                    upper = mid;
                                }
                            } else {
                                // Error computing cost, assume strictly worse (too far?)
                                upper = mid;
                            }
                        }
                        best_delta = (lower + upper) * 0.5;
                    }

                    current_rho[k] -= best_delta;
                    perturbed = true;
                    log::info!(
                        "[Boundary] rho[{}] moved to {:.2} (delta={:.3})",
                        k,
                        current_rho[k],
                        best_delta
                    );
                }
            }

            if !perturbed {
                return Ok((current_rho, None));
            }

            let n_rho = current_rho.len();
            let mut laml_hessian = match self.compute_laml_hessian_exact(&current_rho) {
                Ok(h) => h,
                Err(err) => {
                    log::warn!(
                        "Exact boundary Hessian unavailable ({}); falling back to FD Hessian.",
                        err
                    );
                    let h_step = 1e-4;
                    let mut h_fd = Array2::<f64>::zeros((n_rho, n_rho));
                    let grad_center = self.compute_gradient(&current_rho)?;
                    for j in 0..n_rho {
                        let mut rho_plus = current_rho.clone();
                        rho_plus[j] += h_step;
                        let grad_plus = self.compute_gradient(&rho_plus)?;
                        let col_diff = (&grad_plus - &grad_center) / h_step;
                        for i in 0..n_rho {
                            h_fd[[i, j]] = col_diff[i];
                        }
                    }
                    for i in 0..n_rho {
                        for j in 0..i {
                            let avg = 0.5 * (h_fd[[i, j]] + h_fd[[j, i]]);
                            h_fd[[i, j]] = avg;
                            h_fd[[j, i]] = avg;
                        }
                    }
                    h_fd
                }
            };

            // Invert local Hessian to obtain V_ρ.
            // Stabilization ridge is applied before Cholesky to control near-singularity
            // in weakly identified smoothing directions.
            let mut v_rho = Array2::<f64>::zeros((n_rho, n_rho));
            {
                use crate::faer_ndarray::{FaerArrayView, array2_to_mat_mut};
                use faer::Side;

                // Ensure PD
                crate::pirls::ensure_positive_definite_with_label(
                    &mut laml_hessian,
                    "LAML Hessian",
                )?;

                let h_view = FaerArrayView::new(&laml_hessian);
                if let Ok(chol) = faer::linalg::solvers::Llt::new(h_view.as_ref(), Side::Lower) {
                    let mut eye = Array2::<f64>::eye(n_rho);
                    let mut eye_view = array2_to_mat_mut(&mut eye);
                    chol.solve_in_place(eye_view.as_mut());
                    v_rho.assign(&eye);
                } else {
                    // Fallback: SVD or pseudoinverse? Or just fail correction.
                    log::warn!(
                        "LAML Hessian not invertible even after stabilization. Skipping correction."
                    );
                    return Ok((current_rho, None));
                }
            }

            // 3. Compute smoothing-parameter uncertainty correction: J * V_rho * J^T.
            //
            // Notation mapping to the exact Gaussian-mixture identity:
            //   rho ~ N(mu, Sigma),  mu = rho_hat,  Sigma = V_rho
            //   A(rho) = H_rho^{-1},  b(rho) = beta_hat_rho
            //   Var(beta) = E[A(rho)] + Var(b(rho))   (exact, no truncation)
            //
            // This implementation uses the standard first-order truncation around mu:
            //   E[A(rho)]      ≈ A(mu) = H_p^{-1} = V_beta_cond
            //   Var(b(rho))    ≈ J * V_rho * J^T,  J = dbeta_hat/drho |_{rho=mu}
            // so:
            //   V_total ≈ V_beta_cond + J * V_rho * J^T.
            //
            // Exact higher-order terms from the heat-operator / Wick expansion are
            // not included here.
            //
            // Jacobian identity used here:
            //   d(beta_hat)/d(rho_k) = -H_p^{-1}(S_k^rho * beta_hat), S_k^rho = lambda_k S_k.
            // This is the same implicit derivative used in the main gradient code.

            // We need H_p and beta at the perturbed rho.
            let pirls_res = self.execute_pirls_if_needed(&current_rho)?;

            let beta = pirls_res.beta_transformed.as_ref();
            let h_p = &pirls_res.penalized_hessian_transformed;
            let lambdas = current_rho.mapv(f64::exp);
            let rs = &pirls_res.reparam_result.rs_transformed;

            let p_dim = beta.len();

            // Invert H_p to get V_beta_cond = H_p^{-1}, i.e. A(mu) in the
            // first-order approximation above.
            let mut v_beta_cond = Array2::<f64>::zeros((p_dim, p_dim));
            {
                use crate::faer_ndarray::{FaerArrayView, array2_to_mat_mut};
                use faer::Side;
                let h_view = FaerArrayView::new(h_p);
                // At convergence H_p is typically PD.
                if let Ok(chol) = faer::linalg::solvers::Llt::new(h_view.as_ref(), Side::Lower) {
                    let mut eye = Array2::<f64>::eye(p_dim);
                    let mut eye_view = array2_to_mat_mut(&mut eye);
                    chol.solve_in_place(eye_view.as_mut());
                    v_beta_cond.assign(&eye);
                } else {
                    // Use LDLT if LLT fails
                    if let Ok(ldlt) = faer::linalg::solvers::Ldlt::new(h_view.as_ref(), Side::Lower)
                    {
                        let mut eye = Array2::<f64>::eye(p_dim);
                        let mut eye_view = array2_to_mat_mut(&mut eye);
                        ldlt.solve_in_place(eye_view.as_mut());
                        v_beta_cond.assign(&eye);
                    } else {
                        log::warn!("Penalized Hessian not invertible. Skipping correction.");
                        return Ok((current_rho, None));
                    }
                }
            }

            // Compute Jacobian columns:
            //   J[:,k] = -H_p^{-1}(S_k^ρ β̂)
            //          = -V_beta_cond * (S_k β̂ * λ_k)
            // with S_k β̂ assembled as R_kᵀ(R_k β̂).
            // S_k = R_k^T R_k.
            let mut jacobian = Array2::<f64>::zeros((p_dim, n_rho));

            for k in 0..n_rho {
                let r_k = &rs[k];
                if r_k.ncols() == 0 {
                    continue;
                }

                let lambda = lambdas[k];
                // S_k beta = R_k^T (R_k beta)
                let r_beta = r_k.dot(beta);
                let s_beta = r_k.t().dot(&r_beta);

                let term = s_beta.mapv(|v| v * lambda);

                // col = - V_beta_cond * term
                let col = v_beta_cond.dot(&term).mapv(|v| -v);

                jacobian.column_mut(k).assign(&col);
            }

            // V_corr approximates Var(b(rho)) under first-order linearization.
            // V_corr = J * V_rho * J^T.
            let temp = jacobian.dot(&v_rho); // (p, k) * (k, k) -> (p, k)
            let v_corr = temp.dot(&jacobian.t()); // (p, k) * (k, p) -> (p, p)

            log::info!(
                "[Boundary] Correction computed. Max element in V_corr: {:.3e}",
                v_corr.iter().fold(0.0_f64, |a, &b| a.max(b.abs()))
            );

            // First-order total covariance approximation to Var(beta).
            let v_total = v_beta_cond + v_corr;

            Ok((current_rho, Some(v_total)))
        }
    }
}
