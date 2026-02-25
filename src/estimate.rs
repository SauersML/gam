//! # Model Estimation via Penalized Likelihood and REML
//!
//! This module orchestrates the core model fitting procedure for Generalized Additive
//! Models (GAMs). It determines optimal smoothing parameters directly from the data,
//! moving beyond simple hyperparameter-driven models. This is achieved through a
//! nested optimization scheme, a standard and robust approach for this class of models:
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
use crate::probability::normal_cdf_approx;
use crate::seeding::{SeedConfig, SeedStrategy, generate_rho_candidates};
use crate::types::{Coefficients, LinkFunction, LogSmoothingParams, LogSmoothingParamsView};

// Ndarray and faer linear algebra helpers
use ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1, Axis, Zip, s};
// faer: high-performance dense solvers
use crate::faer_ndarray::{
    FaerArrayView, FaerCholesky, FaerEigh, FaerLinalgError, array2_to_mat_mut, fast_ata,
};
use faer::Mat as FaerMat;
use faer::Side;
use faer::linalg::solvers::{
    Lblt as FaerLblt, Ldlt as FaerLdlt, Llt as FaerLlt, Solve as FaerSolve,
};

fn logit_from_prob(p: f64) -> f64 {
    let p = p.clamp(1e-8, 1.0 - 1e-8);
    (p / (1.0 - p)).ln()
}

use crate::diagnostics::{
    GRAD_DIAG_BETA_COLLAPSE_COUNT, GRAD_DIAG_DELTA_ZERO_COUNT, GRAD_DIAG_KKT_SKIP_COUNT,
    GRAD_DIAG_LOGH_CLAMPED_COUNT, approx_f64, format_compact_series, format_cond, format_range,
    quantize_value, quantize_vec, should_emit_grad_diag, should_emit_h_min_eig_diag,
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
    Ok(match family {
        crate::types::LikelihoodFamily::GaussianIdentity => eta.clone(),
        crate::types::LikelihoodFamily::BinomialLogit => eta.mapv(|v| {
            let z = v.clamp(-30.0, 30.0);
            1.0 / (1.0 + (-z).exp())
        }),
        crate::types::LikelihoodFamily::BinomialProbit => eta.mapv(normal_cdf_approx),
        crate::types::LikelihoodFamily::BinomialCLogLog => eta.mapv(|v| {
            let z = v.clamp(-30.0, 30.0);
            1.0 - (-(z.exp())).exp()
        }),
        crate::types::LikelihoodFamily::RoystonParmar => unreachable!(),
    })
}

fn standard_normal_quantile(p: f64) -> Result<f64, EstimationError> {
    if !(p.is_finite() && p > 0.0 && p < 1.0) {
        return Err(EstimationError::InvalidInput(format!(
            "normal quantile requires p in (0,1), got {p}"
        )));
    }
    // Acklam rational approximation.
    const A: [f64; 6] = [
        -3.969_683_028_665_376e1,
        2.209_460_984_245_205e2,
        -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2,
        -3.066_479_806_614_716e1,
        2.506_628_277_459_239,
    ];
    const B: [f64; 5] = [
        -5.447_609_879_822_406e1,
        1.615_858_368_580_409e2,
        -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1,
        -1.328_068_155_288_572e1,
    ];
    const C: [f64; 6] = [
        -7.784_894_002_430_293e-3,
        -3.223_964_580_411_365e-1,
        -2.400_758_277_161_838,
        -2.549_732_539_343_734,
        4.374_664_141_464_968,
        2.938_163_982_698_783,
    ];
    const D: [f64; 4] = [
        7.784_695_709_041_462e-3,
        3.224_671_290_700_398e-1,
        2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];

    const P_LOW: f64 = 0.02425;
    const P_HIGH: f64 = 1.0 - P_LOW;

    let x = if p < P_LOW {
        let q = (-2.0 * p.ln()).sqrt();
        (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    } else if p <= P_HIGH {
        let q = p - 0.5;
        let r = q * q;
        (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
            / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0)
    };
    Ok(x)
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
                let dense = x.to_dense();
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
use std::sync::{Mutex, RwLock};
use std::sync::atomic::{AtomicBool, Ordering};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
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
// Use a unified rho bound corresponding to lambda in [exp(-RHO_BOUND), exp(RHO_BOUND)].
// Allow additional headroom so the optimizer rarely collides with the hard box even
// when the likelihood prefers effectively infinite smoothing.
const RHO_BOUND: f64 = 30.0;
// Soft interior prior that nudges rho away from the hard walls without meaningfully
// affecting the optimum when the data are informative.
const RHO_SOFT_PRIOR_WEIGHT: f64 = 1e-6;
const RHO_SOFT_PRIOR_SHARPNESS: f64 = 4.0;
const MAX_CONSECUTIVE_INNER_ERRORS: usize = 3;

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
        f64::from_bits(x.to_bits() + 1)
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

/// Compute the smoothing parameter uncertainty correction matrix V_corr = J * V_ρ * J^T.
///
/// This implements the Wood et al. (2016) correction for smoothing parameter uncertainty.
/// The corrected covariance for β is: V*_β = V_β + J * V_ρ * J^T
/// where:
/// - V_β = H^{-1} (the conditional covariance treating λ as fixed)
/// - J = dβ/dρ (the Jacobian of coefficients w.r.t. log-smoothing parameters)
/// - V_ρ = inverse Hessian of LAML w.r.t. ρ (smoothing parameter covariance)
///
/// Returns the correction matrix in the ORIGINAL coefficient basis.
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

    // Step 1: Compute the Jacobian J = dβ/dρ in transformed space
    // For each k: dβ/dρ_k = -H^{-1}(λ_k S_k β)
    // where H is the penalized Hessian and S_k is the k-th penalty matrix.

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

    // Step 2: Compute V_ρ via finite differences of the LAML gradient
    // V_ρ^{-1} = d²LAML/dρ² (Hessian of LAML w.r.t. ρ)
    let h_step = 1e-4;
    let mut hessian_rho = Array2::<f64>::zeros((n_rho, n_rho));

    // Compute Hessian via central differences of gradient
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

        // Central difference: d²f/dρ_k dρ_j ≈ (∂f/∂ρ_j|ρ_k+h - ∂f/∂ρ_j|ρ_k-h) / (2h)
        for j in 0..n_rho {
            hessian_rho[[k, j]] = (grad_plus[j] - grad_minus[j]) / (2.0 * h_step);
        }
    }

    // Symmetrize the Hessian
    for i in 0..n_rho {
        for j in (i + 1)..n_rho {
            let avg = 0.5 * (hessian_rho[[i, j]] + hessian_rho[[j, i]]);
            hessian_rho[[i, j]] = avg;
            hessian_rho[[j, i]] = avg;
        }
    }

    // Step 3: Invert Hessian to get V_ρ
    // Add small ridge for numerical stability
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

    // Step 4: Compute V_corr = J * V_ρ * J^T in transformed space
    let j_v_rho = jacobian_trans.dot(&v_rho); // (n_coeffs_trans x n_rho)
    let v_corr_trans = j_v_rho.dot(&jacobian_trans.t()); // (n_coeffs_trans x n_coeffs_trans)

    // Step 5: Transform back to original coefficient basis
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

fn check_rho_gradient_stationarity(
    label: &str,
    reml_state: &RemlState<'_>,
    final_z: &Array1<f64>,
    tol_z: f64,
) -> Result<(f64, bool), EstimationError> {
    let rho = to_rho_from_z(final_z);
    let mut grad_rho = reml_state.compute_gradient(&rho)?;
    let grad_rho_raw = grad_rho.clone();
    project_rho_gradient(&rho, &mut grad_rho);
    let grad_norm_rho = grad_rho.dot(&grad_rho).sqrt();
    let max_abs_grad = grad_rho_raw
        .iter()
        .fold(0.0_f64, |acc, &val| acc.max(val.abs()));
    let max_abs_rho = rho.iter().fold(0.0_f64, |acc, &val| acc.max(val.abs()));

    let tol_rho = tol_z.max(1e-12);
    let mut is_stationary = grad_norm_rho <= tol_rho;

    let boundary_margin = 1.0_f64;
    let mut boundary_push = false;
    for (&rho_i, &grad_i) in rho.iter().zip(grad_rho_raw.iter()) {
        let dist_to_bound = RHO_BOUND - rho_i.abs();
        if dist_to_bound <= boundary_margin {
            if rho_i > 0.0 && grad_i < -tol_rho {
                boundary_push = true;
                break;
            }
            if rho_i < 0.0 && grad_i > tol_rho {
                boundary_push = true;
                break;
            }
        }
    }

    if boundary_push {
        is_stationary = false;
        eprintln!(
            "[Candidate {label}] Gradient pushes outside rho bound (max|rho|={:.2}, max|∇ρ|={:.3e}); marking as non-stationary",
            max_abs_rho, max_abs_grad
        );
    }

    if !boundary_push && grad_norm_rho > tol_rho {
        eprintln!(
            "[Candidate {label}] projected rho-space gradient norm {:.3e} exceeds tolerance {:.3e}; marking as non-stationary",
            grad_norm_rho, tol_rho
        );
        is_stationary = false;
    }

    eprintln!(
        "[Candidate {label}] rho-space gradient norm {:.3e} (tol {:.3e}); max|∇ρ| {:.3e}; max|ρ| {:.2}; stationary = {}",
        grad_norm_rho, tol_rho, max_abs_grad, max_abs_rho, is_stationary
    );

    Ok((grad_norm_rho, is_stationary))
}

fn run_bfgs_for_candidate(
    label: &str,
    reml_state: &RemlState<'_>,
    config: &RemlConfig,
    initial_z: Array1<f64>,
) -> Result<(BfgsSolution, f64, bool), EstimationError> {
    eprintln!("\n[Candidate {label}] Running BFGS optimization from queued seed");
    let mut solver = Bfgs::new(initial_z, |z| reml_state.cost_and_grad(z))
        .with_tolerance(config.reml_convergence_tolerance)
        .with_max_iterations(config.reml_max_iterations as usize)
        .with_fp_tolerances(1e2, 1e2)
        .with_no_improve_stop(1e-8, 5)
        .with_rng_seed(0xC0FFEE_u64);

    let solution = match solver.run() {
        Ok(solution) => {
            eprintln!("\n[Candidate {label}] BFGS converged successfully according to tolerance.");
            solution
        }
        Err(wolfe_bfgs::BfgsError::LineSearchFailed { last_solution, .. }) => {
            eprintln!(
                "[Candidate {label}] Line search stopped early; using best-so-far parameters."
            );
            *last_solution
        }
        Err(wolfe_bfgs::BfgsError::MaxIterationsReached { last_solution }) => {
            eprintln!(
                "\n[Candidate {label}] WARNING: BFGS hit the iteration cap; using best-so-far parameters."
            );
            eprintln!(
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

    let (grad_norm_rho, is_stationary) = check_rho_gradient_stationarity(
        label,
        reml_state,
        &solution.final_point,
        config.reml_convergence_tolerance,
    )?;

    Ok((solution, grad_norm_rho, is_stationary))
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

/// Train a joint single-index model with flexible link calibration
///
/// This uses the joint model architecture where the base predictor and
/// flexible link are fitted together in one optimization with REML.
///
/// The model is: η = g(Xβ) where g is a learned flexible link function.

// Domain-specific training orchestration is intentionally owned by caller adapters.
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
    /// Optional smoothing-parameter-corrected covariance:
    /// Var*(β) ≈ Var(β|λ) + J Var(ρ) J^T
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
}

fn resolve_external_family(
    family: crate::types::LikelihoodFamily,
) -> Result<(LinkFunction, bool), EstimationError> {
    match family {
        crate::types::LikelihoodFamily::GaussianIdentity => Ok((LinkFunction::Identity, false)),
        crate::types::LikelihoodFamily::BinomialLogit => Ok((LinkFunction::Logit, true)),
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
    objective: &F,
) -> Result<Array1<f64>, EstimationError>
where
    F: Fn(&Array1<f64>) -> Result<f64, EstimationError>,
{
    let mut grad = Array1::<f64>::zeros(rho.len());
    for i in 0..rho.len() {
        let mut rp = rho.clone();
        rp[i] += step;
        let fp = objective(&rp)?;
        let mut rm = rho.clone();
        rm[i] -= step;
        let fm = objective(&rm)?;
        grad[i] = (fp - fm) / (2.0 * step);
    }
    Ok(grad)
}

/// Generic multi-start BFGS smoothing optimizer over log-smoothing parameters (`rho`).
///
/// This is intended for likelihoods whose outer objective is exposed as a scalar
/// function of `rho` (for example survival workflows built on working-model PIRLS).
pub fn optimize_log_smoothing_with_multistart<F>(
    num_penalties: usize,
    heuristic_lambdas: Option<&[f64]>,
    objective: F,
    options: &SmoothingBfgsOptions,
) -> Result<SmoothingBfgsResult, EstimationError>
where
    F: Fn(&Array1<f64>) -> Result<f64, EstimationError>,
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

    let seeds = generate_rho_candidates(num_penalties, heuristic_lambdas, &options.seed_config);
    if seeds.is_empty() {
        return Err(EstimationError::RemlOptimizationFailed(
            "no smoothing seeds produced".to_string(),
        ));
    }

    let mut best: Option<SmoothingBfgsResult> = None;
    for (idx, rho_seed) in seeds.iter().enumerate() {
        let initial_z = to_z_from_rho(rho_seed);
        let mut optimizer = Bfgs::new(initial_z.clone(), |z| {
            let rho = to_rho_from_z(z);
            let cost = objective(&rho).unwrap_or(f64::INFINITY);
            let grad_rho =
                finite_diff_gradient_external(&rho, options.finite_diff_step, &objective)
                    .unwrap_or_else(|_| Array1::<f64>::zeros(rho.len()));
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
            Err(err) => {
                let _ = err;
                continue;
            }
        };

        let rho = to_rho_from_z(&solution.final_point);
        let mut grad_rho =
            finite_diff_gradient_external(&rho, options.finite_diff_step, &objective)?;
        project_rho_gradient(&rho, &mut grad_rho);
        let grad_norm = grad_rho.dot(&grad_rho).sqrt();
        let stationary = grad_norm <= options.tol.max(1e-6);
        let candidate = SmoothingBfgsResult {
            rho,
            final_value: solution.final_value,
            iterations: solution.iterations,
            final_grad_norm: grad_norm,
            stationary,
        };

        let replace = match &best {
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
        };
        if replace {
            best = Some(candidate);
        }
    }

    best.ok_or_else(|| {
        EstimationError::RemlOptimizationFailed(
            "all smoothing BFGS starts failed before producing a candidate".to_string(),
        )
    })
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
    let (link, firth_active) = resolve_external_family(opts.family)?;
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

    let mut best_solution: Option<BfgsSolution> = None;
    let mut best_grad_norm = f64::INFINITY;
    let mut found_stationary = false;
    for (label, initial_z) in candidate_seeds {
        let (solution, grad_norm_rho, stationary) =
            run_bfgs_for_candidate(&label, &reml_state, &cfg, initial_z)?;
        if stationary {
            best_solution = Some(solution);
            best_grad_norm = grad_norm_rho;
            found_stationary = true;
            break;
        }
        let better = match &best_solution {
            None => true,
            Some(current) => solution.final_value < current.final_value,
        };
        if better {
            best_grad_norm = grad_norm_rho;
            best_solution = Some(solution);
        }
    }
    let chosen_solution = best_solution.ok_or_else(|| {
        EstimationError::RemlOptimizationFailed(
            "no valid BFGS solution produced by candidate seeds".to_string(),
        )
    })?;
    if !found_stationary {
        eprintln!(
            "[external] no stationary candidate found; using best non-stationary solution with grad_norm={:.3e}",
            best_grad_norm
        );
    }
    let final_point = chosen_solution.final_point.clone();
    // Ensure we don't report 0 iterations to the caller; at least 1 is more meaningful.
    let iters = std::cmp::max(1, chosen_solution.iterations);
    let final_rho = to_rho_from_z(&final_point);
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
    let final_grad = reml_state
        .compute_gradient(&final_rho)
        .unwrap_or_else(|_| Array1::from_elem(final_rho.len(), f64::NAN));
    let final_grad_norm_rho = final_grad.dot(&final_grad).sqrt();
    let final_grad_norm = if final_grad_norm_rho.is_finite() {
        final_grad_norm_rho
    } else {
        best_grad_norm
    };

    let smoothing_correction = compute_smoothing_correction(&reml_state, &final_rho, &pirls_res);
    let penalized_hessian = map_hessian_to_original_basis(&pirls_res)?;
    let beta_covariance =
        matrix_inverse_with_regularization(&penalized_hessian, "posterior covariance");
    let beta_standard_errors = beta_covariance.as_ref().map(se_from_covariance);
    let beta_covariance_corrected = match (&beta_covariance, &smoothing_correction) {
        (Some(base_cov), Some(corr)) if base_cov.dim() == corr.dim() => {
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
    /// Optional smoothing-parameter-corrected covariance:
    /// Var*(β) ≈ Var(β|λ) + J Var(ρ) J^T
    pub beta_covariance_corrected: Option<Array2<f64>>,
    /// Marginal SEs from `beta_covariance_corrected`.
    pub beta_standard_errors_corrected: Option<Array1<f64>>,
}

pub struct PredictResult {
    pub eta: Array1<f64>,
    pub mean: Array1<f64>,
}

pub struct PredictUncertaintyOptions {
    /// Central interval level in (0, 1), e.g. 0.95.
    pub confidence_level: f64,
    /// If true, use smoothing-parameter-corrected covariance when available.
    pub prefer_corrected_covariance: bool,
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
            prefer_corrected_covariance: true,
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
}

pub struct CoefficientUncertaintyResult {
    pub estimate: Array1<f64>,
    pub standard_error: Array1<f64>,
    pub lower: Array1<f64>,
    pub upper: Array1<f64>,
    pub corrected: bool,
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
    let ext_opts = ExternalOptimOptions {
        family,
        max_iter: opts.max_iter,
        tol: opts.tol,
        nullspace_dims: opts.nullspace_dims.clone(),
    };
    let result = optimize_external_design(y, weights, &x, offset, s_list.to_vec(), &ext_opts)?;
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

    let cov = if options.prefer_corrected_covariance {
        fit.beta_covariance_corrected
            .as_ref()
            .or(fit.beta_covariance.as_ref())
    } else {
        fit.beta_covariance.as_ref()
    }
    .ok_or_else(|| {
        EstimationError::InvalidInput(
            "fit result does not contain a usable posterior covariance".to_string(),
        )
    })?;

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

    let z = standard_normal_quantile(0.5 + 0.5 * options.confidence_level)?;
    let eta_lower = &eta - &eta_standard_error.mapv(|s| z * s);
    let eta_upper = &eta + &eta_standard_error.mapv(|s| z * s);

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
    })
}

/// Coefficient-level uncertainty and confidence intervals.
pub fn coefficient_uncertainty(
    fit: &FitResult,
    confidence_level: f64,
    prefer_corrected_covariance: bool,
) -> Result<CoefficientUncertaintyResult, EstimationError> {
    if !(confidence_level.is_finite() && confidence_level > 0.0 && confidence_level < 1.0) {
        return Err(EstimationError::InvalidInput(format!(
            "confidence_level must be in (0,1), got {}",
            confidence_level
        )));
    }
    let (se, corrected) = if prefer_corrected_covariance {
        if let Some(se_corr) = fit.beta_standard_errors_corrected.as_ref() {
            (se_corr.clone(), true)
        } else if let Some(se_base) = fit.beta_standard_errors.as_ref() {
            (se_base.clone(), false)
        } else {
            return Err(EstimationError::InvalidInput(
                "fit result does not contain coefficient standard errors".to_string(),
            ));
        }
    } else if let Some(se_base) = fit.beta_standard_errors.as_ref() {
        (se_base.clone(), false)
    } else {
        return Err(EstimationError::InvalidInput(
            "fit result does not contain coefficient standard errors".to_string(),
        ));
    };

    if se.len() != fit.beta.len() {
        return Err(EstimationError::InvalidInput(format!(
            "standard error length mismatch: beta has {}, se has {}",
            fit.beta.len(),
            se.len()
        )));
    }

    let z = standard_normal_quantile(0.5 + 0.5 * confidence_level)?;
    let lower = &fit.beta - &se.mapv(|s| z * s);
    let upper = &fit.beta + &se.mapv(|s| z * s);
    Ok(CoefficientUncertaintyResult {
        estimate: fit.beta.clone(),
        standard_error: se,
        lower,
        upper,
        corrected,
    })
}

/// Computes the gradient of the LAML cost function using the central finite-difference method.
const FD_REL_GAP_THRESHOLD: f64 = 0.2;
const FD_MIN_BASE_STEP: f64 = 1e-6;
const FD_MAX_REFINEMENTS: usize = 4;
const FD_RIDGE_REL_JITTER_THRESHOLD: f64 = 1e-3;
const FD_RIDGE_ABS_JITTER_THRESHOLD: f64 = 1e-12;

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

fn compute_fd_gradient(
    reml_state: &internal::RemlState,
    rho: &Array1<f64>,
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

        if derivative.is_none() {
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

    if !log_lines.is_empty() {
        println!("{}", log_lines.join("\n"));
    }

    Ok(fd_grad)
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

    let (link, firth_active) = resolve_external_family(opts.family)?;
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

    let (link, firth_active) = resolve_external_family(opts.family)?;
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
        ridge_used: f64,
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
        //    - Negative eigenvalues contribute 0 to cost, so derivative must be 0
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
            ridge: f64,
            base_log_det: f64,
        ) -> Result<f64, EstimationError> {
            if ridge <= 0.0 {
                return Ok(base_log_det);
            }

            // When a stabilization ridge is treated as an explicit penalty term,
            // the penalty matrix becomes S_λ + ridge * I. The LAML cost must use
            // log|S_λ + ridge I|_+ for exact consistency. Without this, the cost
            // would be evaluating a different prior than the one implied by the
            // PIRLS stationarity condition and the stabilized Hessian.
            let p = s_transformed.nrows();
            let mut s_ridge = s_transformed.clone();
            for i in 0..p {
                s_ridge[[i, i]] += ridge;
            }
            let chol = s_ridge.clone().cholesky(Side::Lower).map_err(|_| {
                EstimationError::ModelIsIllConditioned {
                    condition_number: f64::INFINITY,
                }
            })?;
            let log_det = 2.0 * chol.diag().mapv(f64::ln).sum();
            Ok(log_det)
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

        /// Returns the effective Hessian and the ridge value used (if any).
        /// This ensures we use the same Hessian matrix in both cost and gradient calculations.
        ///
        /// PIRLS folds any stabilization ridge directly into the penalized objective:
        ///   l_p(β; ρ) = l(β) - 0.5 * βᵀ (S_λ + ridge I) β.
        /// Therefore the curvature used in LAML must be
        ///   H_eff = X'WX + S_λ + ridge I,
        /// and we must not add another ridge here or the Laplace expansion
        /// would be centered on a different surface.
        fn effective_hessian(
            &self,
            pr: &PirlsResult,
        ) -> Result<(Array2<f64>, f64), EstimationError> {
            let base = pr.stabilized_hessian_transformed.clone();

            if base.cholesky(Side::Lower).is_ok() {
                return Ok((base, pr.ridge_used));
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
                cache: RefCell::new(HashMap::new()),
                faer_factor_cache: RefCell::new(HashMap::new()),
                eval_count: RefCell::new(0),
                last_cost: RefCell::new(f64::INFINITY),
                last_grad_norm: RefCell::new(f64::INFINITY),
                consecutive_cost_errors: RefCell::new(0),
                last_cost_error_msg: RefCell::new(None),
                current_eval_bundle: RefCell::new(None),
                cost_last: RefCell::new(None),
                cost_repeat: RefCell::new(0),
                cost_last_emit: RefCell::new(0),
                cost_eval_count: RefCell::new(0),
                raw_cond_snapshot: RefCell::new(f64::NAN),
                gaussian_cond_snapshot: RefCell::new(f64::NAN),
                workspace: Mutex::new(workspace),
                warm_start_beta: RefCell::new(None),
                warm_start_enabled: Cell::new(true),
            })
        }

        /// Creates a sanitized cache key from rho values.
        /// Returns None if any component is NaN, which indicates that caching should be skipped.
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
            let (h_eff, ridge_used) = self.effective_hessian(pirls_result.as_ref())?;

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
            // We intentionally stay on H_eff here for cost/gradient consistency.
            let h_total = h_eff.clone();
            let (eigvals, eigvecs) = h_total
                .eigh(Side::Lower)
                .map_err(|e| EstimationError::EigendecompositionFailed(e))?;
            let max_eig = eigvals.iter().copied().fold(0.0_f64, f64::max);
            let eig_threshold = (max_eig * EIG_REL_THRESHOLD).max(EIG_ABS_FLOOR);

            // Sum log(lambda) for valid eigenvalues
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

            // Filter valid eigenvalues and construct W = U_+ diag(1/sqrt(lambda_+)).
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
                ridge_used,
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
                .read().unwrap()
                .as_ref()
                .map(|bundle| bundle.ridge_used)
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
            // trace(H⁻¹ S_λ) = ⟨H⁻¹ Eᵀ, Eᵀ⟩_F directly (numerically robust)
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
                        .write().unwrap()
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
            // split the cache.  Until then we prefer the cheaper key because it maximizes cache
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

        /// Compute ∂log|H|/∂β for logit GLM.
        /// Uses the penalized Hessian factorization for leverage computation.
        fn logh_beta_grad_logit(
            &self,
            x_transformed: &DesignMatrix,
            mu: &Array1<f64>,
            weights: &Array1<f64>,
            factor: &Arc<FaerFactor>,
        ) -> Option<Array1<f64>> {
            let n = mu.len();
            if n == 0 {
                return None;
            }

            // Match the GLM probability clamp in update_glm_vectors.
            // The clamp makes w(eta) constant in saturated regions, so dw/deta = 0.
            // Using the unclamped derivative there would create a phantom gradient.
            const PROB_EPS: f64 = 1e-8;
            // Match the GLM weight clamp in update_glm_vectors:
            // if dmu is clamped, weights are constant and w' = 0.
            const MIN_WEIGHT: f64 = 1e-12;
            let mut w_prime = Array1::<f64>::zeros(n);
            let mut clamped = 0usize;
            for i in 0..n {
                let mu_i = mu[i];
                let w_base = mu_i * (1.0 - mu_i);
                if mu_i <= PROB_EPS || mu_i >= 1.0 - PROB_EPS || w_base < MIN_WEIGHT {
                    clamped += 1;
                    w_prime[i] = 0.0;
                    continue;
                }
                let one_minus2 = 1.0 - 2.0 * mu_i;
                w_prime[i] = weights[i] * one_minus2;
            }

            // Always use full rank path (Cholesky solve).
            // This is consistent with compute_cost which now uses full log|H| via Cholesky.
            // Previously, truncation was used here but it caused gradient mismatch.

            let mut leverage = Array1::<f64>::zeros(n);
            let chunk_cols = 1024usize;
            match x_transformed {
                DesignMatrix::Dense(x_dense) => {
                    let p_dim = x_dense.ncols();
                    for chunk_start in (0..n).step_by(chunk_cols) {
                        let chunk_end = (chunk_start + chunk_cols).min(n);
                        let width = chunk_end - chunk_start;

                        // Full rank path (standard Cholesky solve)
                        let mut rhs = Array2::<f64>::zeros((p_dim, width));
                        for (local, row_idx) in (chunk_start..chunk_end).enumerate() {
                            rhs.column_mut(local).assign(&x_dense.row(row_idx));
                        }
                        let rhs_view = FaerArrayView::new(&rhs);
                        let sol = factor.solve(rhs_view.as_ref());
                        for local in 0..width {
                            let row_idx = chunk_start + local;
                            let mut acc = 0.0;
                            for j in 0..p_dim {
                                acc += x_dense[[row_idx, j]] * sol[(j, local)];
                            }
                            leverage[row_idx] = acc;
                        }
                    }
                }
                DesignMatrix::Sparse(x_sparse) => {
                    let p_dim = x_sparse.ncols();
                    let csr_opt = x_sparse.as_ref().to_row_major().ok();
                    if let Some(x_csr) = csr_opt {
                        let symbolic = x_csr.symbolic();
                        let values = x_csr.val();
                        let row_ptr = symbolic.row_ptr();
                        let col_idx = symbolic.col_idx();
                        for chunk_start in (0..n).step_by(chunk_cols) {
                            let chunk_end = (chunk_start + chunk_cols).min(n);
                            let width = chunk_end - chunk_start;

                            // Full rank sparse path
                            let mut rhs = Array2::<f64>::zeros((p_dim, width));
                            for (local, row_idx) in (chunk_start..chunk_end).enumerate() {
                                let start = row_ptr[row_idx];
                                let end = row_ptr[row_idx + 1];
                                for idx in start..end {
                                    rhs[[col_idx[idx], local]] = values[idx];
                                }
                            }
                            let rhs_view = FaerArrayView::new(&rhs);
                            let sol = factor.solve(rhs_view.as_ref());
                            for (local, row_idx) in (chunk_start..chunk_end).enumerate() {
                                let mut acc = 0.0;
                                let start = row_ptr[row_idx];
                                let end = row_ptr[row_idx + 1];
                                for idx in start..end {
                                    let col = col_idx[idx];
                                    acc += values[idx] * sol[(col, local)];
                                }
                                leverage[row_idx] = acc;
                            }
                        }
                    } else {
                        // Fallback for non-CSR sparse (convert to dense)
                        let mut x_dense =
                            Array2::<f64>::zeros((x_sparse.nrows(), x_sparse.ncols()));
                        let (symbolic, values) = x_sparse.parts();
                        let col_ptr = symbolic.col_ptr();
                        let row_idx = symbolic.row_idx();
                        for col in 0..x_sparse.ncols() {
                            let start = col_ptr[col];
                            let end = col_ptr[col + 1];
                            for idx in start..end {
                                x_dense[[row_idx[idx], col]] = values[idx];
                            }
                        }
                        let p_dim = x_dense.ncols();
                        for chunk_start in (0..n).step_by(chunk_cols) {
                            let chunk_end = (chunk_start + chunk_cols).min(n);
                            let width = chunk_end - chunk_start;

                            let mut rhs = Array2::<f64>::zeros((p_dim, width));
                            for (local, row_idx) in (chunk_start..chunk_end).enumerate() {
                                rhs.column_mut(local).assign(&x_dense.row(row_idx));
                            }
                            let rhs_view = FaerArrayView::new(&rhs);
                            let sol = factor.solve(rhs_view.as_ref());
                            for (local, row_idx) in (chunk_start..chunk_end).enumerate() {
                                let mut acc = 0.0;
                                for j in 0..p_dim {
                                    acc += x_dense[[row_idx, j]] * sol[(j, local)];
                                }
                                leverage[row_idx] = acc;
                            }
                        }
                    }
                }
            }

            let mut weight_vec = Array1::<f64>::zeros(n);
            for i in 0..n {
                weight_vec[i] = leverage[i] * w_prime[i];
            }
            let logh_grad = x_transformed.transpose_vector_multiply(&weight_vec);
            let logh_norm = logh_grad
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, |a, b| a.max(b));
            if clamped > 0 && logh_norm < 1e-8 {
                let (should_print, count) = should_emit_grad_diag(&GRAD_DIAG_LOGH_CLAMPED_COUNT);
                if should_print {
                    eprintln!(
                        "[GRAD DIAG #{count}] logh_beta_grad ~0 with clamped weights: clamped={}/{}, max|logh_beta_grad|={:.3e}",
                        clamped, n, logh_norm
                    );
                }
            }
            Some(logh_grad)
        }

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
                            .write().unwrap()
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
        /// For Gaussian models (Identity link), this is the exact REML score.
        /// For non-Gaussian GLMs, this is the LAML (Laplace Approximate Marginal Likelihood) score.
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
                    eprintln!(
                        "[Diag] rho bounds: lower={:?} upper={:?}",
                        at_lower, at_upper
                    );
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
                    eprintln!(
                        "[Diag] rho bounds: lower={:?} upper={:?}",
                        at_lower, at_upper
                    );
                    return Err(e);
                }
            };
            let pirls_result = bundle.pirls_result.as_ref();
            let h_eff = bundle.h_eff.as_ref();
            let ridge_used = bundle.ridge_used;

            let lambdas = p.mapv(f64::exp);

            // Sanity check: penalty dimension consistency across lambdas, R_k, and det1.
            if !p.is_empty() {
                let kλ = p.len();
                let kR = pirls_result.reparam_result.rs_transformed.len();
                let kD = pirls_result.reparam_result.det1.len();
                if !(kλ == kR && kR == kD) {
                    return Err(EstimationError::LayoutError(format!(
                        "Penalty dimension mismatch: lambdas={}, R={}, det1={}",
                        kλ, kR, kD
                    )));
                }
                if self.nullspace_dims.len() != kλ {
                    return Err(EstimationError::LayoutError(format!(
                        "Nullspace dimension mismatch: expected {} entries, got {}",
                        kλ,
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
                    // For Gaussian models, use the exact REML score
                    // From Wood (2017), Chapter 6, Eq. 6.24:
                    // V_r(λ) = D_p/(2φ) + (r/2φ) + ½log|X'X/φ + S_λ/φ| - ½log|S_λ/φ|_+
                    // where D_p = ||y - Xβ̂||² + β̂'S_λβ̂ is the PENALIZED deviance

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
                    let ridge_used = pirls_result.ridge_used;
                    let log_det_s_plus = Self::log_det_s_with_ridge(
                        &pirls_result.reparam_result.s_transformed,
                        ridge_used,
                        pirls_result.reparam_result.log_det,
                    )?;

                    // Standard REML expression from Wood (2017), Section 6.5.1
                    // V = (n/2)log(2πσ²) + D_p/(2σ²) + ½log|H| - ½log|S_λ|_+ + (M_p-1)/2 log(2πσ²)
                    // Simplifying: V = D_p/(2φ) + ½log|H| - ½log|S_λ|_+ + ((n-M_p)/2) log(2πφ)
                    let reml = dp_c / (2.0 * phi)
                        + 0.5 * (log_det_h - log_det_s_plus)
                        + ((n - mp) / 2.0) * (2.0 * std::f64::consts::PI * phi).ln();

                    let prior_cost = self.compute_soft_prior_cost(p);

                    // Return the REML score (which is a negative log-likelihood, i.e., a cost to be minimized)
                    Ok(reml + prior_cost)
                }
                _ => {
                    // For non-Gaussian GLMs, use the LAML approximation
                    // Penalized log-likelihood part of the score.
                    // Note: Deviance = -2 * log-likelihood + C. So -0.5 * Deviance = log-likelihood - C/2.
                    // Use stable penalty term calculated in P-IRLS
                    let mut penalised_ll =
                        -0.5 * pirls_result.deviance - 0.5 * pirls_result.stable_penalty_term;

                    let ridge_used = pirls_result.ridge_used;
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
                        ridge_used,
                        pirls_result.reparam_result.log_det,
                    )?;

                    // Log-determinant of the effective Hessian.
                    // HESSIAN PASSPORT: Use the pre-computed h_total and its factorization
                    // from the bundle to ensure exact consistency with gradient computation.
                    // For Firth: h_total = h_eff - h_phi (computed in prepare_eval_bundle)
                    // For non-Firth: h_total = h_eff
                    let log_det_h = bundle.h_total_log_det;

                    // The LAML score is Lp + 0.5*log|S| - 0.5*log|H| + Mp/2*log(2πφ)
                    // Mp is null space dimension (number of unpenalized coefficients)
                    // For logit, scale parameter is typically fixed at 1.0, but include for completeness
                    let phi = 1.0; // Logit family typically has dispersion parameter = 1

                    // Compute null space dimension using the TRANSFORMED, STABLE basis
                    // Use the rank of the lambda-weighted transformed penalty root (e_transformed)
                    // to determine M_p robustly, avoiding contamination from dominant penalties.
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

                    // Expensive raw-condition diagnostics are rate-limited in the hot loop.
                    // We only refresh occasionally, and keep the last snapshot otherwise.
                    let raw_cond = if matches!(self.x(), DesignMatrix::Dense(_)) && want_hot_diag {
                        let x_orig = self.x().to_dense();
                        let w_orig = self.weights();
                        let sqrt_w = w_orig.mapv(|w| w.max(0.0).sqrt());
                        let wx = &x_orig * &sqrt_w.insert_axis(Axis(1));
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
            let verbose_opt = std::env::var("GAM_VERBOSE_OPT")
                .map(|v| v == "1")
                .unwrap_or(false);

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
        /// Compute the gradient of the REML/LAML score with respect to the log-smoothing parameters (ρ).
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
        /// Since β̂ is an implicit function of ρ, we must use the total derivative:
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
        ///   deviance component. The derivative of the penalized deviance must include both
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
        /// # Derivation of the Analytic Gradient for Firth-Adjusted LAML
        ///
        /// This function implements the exact gradient of the Laplace Approximate Marginal Likelihood (LAML)
        /// with respect to the smoothing parameters $\rho$.
        ///
        /// The Outer Objective (LAML) is:
        /// $$ V(\rho) = - \mathcal{L}(\hat{\beta}, \rho) + \frac{1}{2} \log |H_{total}| - \frac{1}{2} \log |S_\lambda|_+ $$
        ///
        /// The gradient is computed via the Total Derivative:
        /// $$ \frac{d V}{d \rho_k} = \frac{\partial V}{\partial \rho_k} \bigg|_{\hat{\beta}} + \left( \nabla_\beta V \right)^\top \frac{d \hat{\beta}}{d \rho_k} $$
        ///
        /// ## Term 1: Direct Partial Derivative $\frac{\partial V}{\partial \rho_k}$
        /// $$ \frac{\partial V}{\partial \rho_k} = \frac{1}{2} \lambda_k \hat{\beta}^\top S_k \hat{\beta} + \frac{1}{2} \lambda_k \text{tr}(H_{total}^{-1} S_k) - \frac{1}{2} \lambda_k \text{tr}(S_\lambda^+ S_k) $$
        /// - **Beta Quadratic:** $0.5 \lambda_k \beta^\top S_k \beta$ (`0.5 * beta_terms`)
        /// - **Log-Det Hessian:** $0.5 \lambda_k \text{tr}(H_{total}^{-1} S_k)$ (`log_det_h_grad_term`)
        /// - **Log-Det Penalty:** $-0.5 \lambda_k \text{tr}(S^+ S_k)$ (`-0.5 * det1_values`)
        ///
        /// ## Term 2: Implicit Correction
        /// The implicit derivative of the coefficients $\frac{d \hat{\beta}}{d \rho_k}$ accounts for the fact that
        /// $\hat{\beta}$ moves as $\rho$ changes to maintain the stationarity condition $\nabla_\beta \mathcal{L} = 0$.
        ///
        /// $$ \frac{d \hat{\beta}}{d \rho_k} = - H_{total}^{-1} (\lambda_k S_k \hat{\beta}) $$
        ///
        /// The correction term is:
        /// $$ (\nabla_\beta V)^\top \frac{d \hat{\beta}}{d \rho_k} = - (\nabla_\beta V)^\top H_{total}^{-1} (\lambda_k S_k \hat{\beta}) $$
        ///
        /// Where $\nabla_\beta V = -\nabla_\beta \mathcal{L} + \frac{1}{2} \nabla_\beta \log |H_{total}|$.
        /// - At a perfect optimum, $\nabla_\beta \mathcal{L} = 0$, but we include `residual_grad` for robustness.
        /// - $\frac{1}{2} \nabla_\beta \log |H_{total}|$ is computed via `firth_logh_total_grad`.
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
            let ridge_used = bundle.ridge_used;

            // Sanity check: penalty dimension consistency across lambdas, R_k, and det1.
            let kλ = p.len();
            let kR = pirls_result.reparam_result.rs_transformed.len();
            let kD = pirls_result.reparam_result.det1.len();
            if !(kλ == kR && kR == kD) {
                return Err(EstimationError::LayoutError(format!(
                    "Penalty dimension mismatch: lambdas={}, R={}, det1={}",
                    kλ, kR, kD
                )));
            }
            if self.nullspace_dims.len() != kλ {
                return Err(EstimationError::LayoutError(format!(
                    "Nullspace dimension mismatch: expected {} entries, got {}",
                    kλ,
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

                // When we treat a stabilization ridge as a true penalty term, the
                // penalty matrix becomes S_λ + ridge * I. For exactness, both the
                // log|S| term in the cost and the derivative d/dρ_k log|S| must be
                // computed using this ridged matrix. The derivative follows from
                // Jacobi's formula:
                //   d/dρ_k log|S_λ + δI|
                //     = tr((S_λ + δI)^{-1} dS_λ/dρ_k)
                //     = λ_k tr((S_λ + δI)^{-1} S_k).
                //
                //   det1[k] = d/dρ_k log|S_λ + ridge I|
                //           = λ_k * tr((S_λ + ridge I)^{-1} S_k)
                //
                // which can be evaluated without explicitly forming S_k by using
                // the penalty roots R_k (S_k = R_kᵀ R_k).
                let det1_values = if ridge_used > 0.0 {
                    // If a stabilization ridge is treated as an explicit penalty term,
                    // the penalty matrix becomes S_λ + ridge * I. The gradient term
                    // d/dρ_k log|S_λ + ridge I| uses:
                    //   det1[k] = λ_k * tr((S_λ + ridge I)^{-1} S_k)
                    let p_dim = reparam_result.s_transformed.nrows();
                    let mut s_ridge = reparam_result.s_transformed.clone();
                    for i in 0..p_dim {
                        s_ridge[[i, i]] += ridge_used;
                    }
                    let s_view = FaerArrayView::new(&s_ridge);
                    let chol = FaerLlt::new(s_view.as_ref(), Side::Lower).map_err(|_| {
                        EstimationError::ModelIsIllConditioned {
                            condition_number: f64::INFINITY,
                        }
                    })?;

                    let mut det1 = Array1::<f64>::zeros(len);
                    for (k, rt) in rs_transposed.iter().enumerate() {
                        if rt.ncols() == 0 {
                            continue;
                        }
                        let mut rhs = rt.to_owned();
                        let mut rhs_view = array2_to_mat_mut(&mut rhs);
                        chol.solve_in_place(rhs_view.as_mut());
                        let trace = kahan_sum(rhs.iter().zip(rt.iter()).map(|(&x, &y)| x * y));
                        det1[k] = lambdas[k] * trace;
                    }
                    det1
                } else {
                    reparam_result.det1.clone()
                };

                // --- Use Single Stabilized Hessian from P-IRLS ---
                // Use the same effective Hessian as the cost function for consistency.
                if ridge_used > 0.0 {
                    log::debug!(
                        "Gradient path using PIRLS-stabilized Hessian (ridge {:.3e})",
                        ridge_used
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
                        let compute_gaussian_grad = |k: usize| -> f64 {
                            let r_k = &rs_transformed[k];
                            // Avoid forming S_k: compute S_k β = Rᵀ (R β)
                            let r_beta = r_k.dot(beta_ref);
                            let s_k_beta_transformed = r_k.t().dot(&r_beta);

                            // Component 1: derivative of the penalized deviance.
                            // For Gaussian models, the Envelope Theorem simplifies this to only the penalty term.
                            let d1 = lambdas[k] * beta_ref.dot(&s_k_beta_transformed);
                            let deviance_grad_term = dp_c_grad * (d1 / (2.0 * scale));

                            // Component 2: derivative of the penalized Hessian determinant.
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

                            // Component 3: derivative of the penalty pseudo-determinant.
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
                        // NON-GAUSSIAN LAML GRADIENT - Wood (2011) Appendix D
                        // This branch follows the common practical LAML strategy:
                        // keep the tractable implicit-differentiation terms and avoid
                        // explicit third-derivative tensor construction for dH/dtheta.
                        // This is the standard GAM approximation: drop the explicit
                        // dH/dtheta term while retaining the dominant mgcv-style
                        // implicit-beta sensitivity terms.
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
                        if !matches!(self.config.link_function(), LinkFunction::Logit) {
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
                            for k in 0..k_count {
                                let r_k = &rs_transformed[k];
                                let r_beta = r_k.dot(beta_ref);
                                let s_k_beta = r_k.t().dot(&r_beta);
                                beta_terms[k] = lambdas[k] * beta_ref.dot(&s_k_beta);
                            }

                            // Keep outer gradient on the same Hessian surface as PIRLS.
                            // The outer loop uses H_eff consistently (no H_phi subtraction).

                            // P-IRLS already folded any stabilization ridge into h_eff.

                            // Create local factor_g for the non-Firth path and Firth fallback.
                            // The non-Firth path intentionally uses full-rank Cholesky (not pseudoinverse)
                            // because truncation caused gradient mismatch (see logh_beta_grad_logit).
                            let factor_g = {
                                let h_total = bundle.h_total.as_ref();
                                let h_view = FaerArrayView::new(h_total);
                                if let Ok(f) = FaerLlt::new(h_view.as_ref(), Side::Lower) {
                                    Arc::new(FaerFactor::Llt(f))
                                } else {
                                    // Fallback to LDLT
                                    match FaerLdlt::new(h_view.as_ref(), Side::Lower) {
                                        Ok(f) => Arc::new(FaerFactor::Ldlt(f)),
                                        Err(_) => {
                                            // Last resort: use the RidgePlanner
                                            // But we don't have easy access to self.get_faer_factor here without rho.
                                            // We'll panic or return error if this fails, which is rare for h_total.
                                            // Or better, use get_faer_factor since we have rho.
                                            self.get_faer_factor(p, h_total)
                                        }
                                    }
                                }
                            };

                            // TRACE TERM COMPUTATION: tr(H_+^\dagger S_k)
                            // Use factor form to avoid materializing H_+^\dagger = W W^T:
                            // tr(H_+^\dagger S_k) = ||R_k W||_F^2.
                            let w_pos = bundle.h_pos_factor_w.as_ref();

                            let mut trace_terms = vec![0.0; k_count];
                            for k_idx in 0..k_count {
                                let r_k = &rs_transformed[k_idx];
                                if r_k.ncols() == 0 || w_pos.ncols() == 0 {
                                    continue;
                                }
                                let rkw = r_k.dot(w_pos);
                                trace_terms[k_idx] = rkw.iter().map(|v| v * v).sum();
                            }

                            // We do NOT need to set workspace.solved_rows as we aren't using the workspace solver.
                            workspace.solved_rows = 0;

                            // Implicit Truncation Correction:
                            // By using H_+^\dagger essentially constructed from U_R D_R^{-1} U_R^T,
                            // we automatically project dS onto the valid subspace P_R.
                            // The phantom spectral bleed term (tr(H^-1 P_N dS P_N)) is identically zero
                            // because P_N H_+^\dagger = 0.
                            let truncation_corrections = vec![0.0; k_count];

                            let residual_grad = {
                                let eta = pirls_result.solve_mu.mapv(|m| logit_from_prob(m));
                                let working_residual = &eta - &pirls_result.solve_working_response;
                                let weighted_residual =
                                    &pirls_result.solve_weights * &working_residual;
                                let gradient_data = pirls_result
                                    .x_transformed
                                    .transpose_vector_multiply(&weighted_residual);
                                let s_beta = reparam_result.s_transformed.dot(beta_ref);
                                // When Firth bias reduction is active, the working response already
                                // includes the Jeffreys adjustment via the hat diagonal. That means
                                // the Firth score term is embedded in this residual gradient; do not
                                // add any extra ∂log|I|/∂β term here or it will be double-counted.
                                // If PIRLS added a stabilization ridge, the objective being
                                // optimized is l_p(β) - 0.5 * ridge * ||β||². The gradient
                                // therefore gains + ridge * β, which must be included here
                                // so the implicit correction matches the stabilized objective.
                                if ridge_used > 0.0 {
                                    gradient_data + s_beta + beta_ref.mapv(|v| ridge_used * v)
                                } else {
                                    gradient_data + s_beta
                                }
                            };

                            // LAML adds 0.5 * ∂log|H₊|/∂β via Jacobi's formula:
                            //   ∂/∂β_j log|H₊| = tr(H₊† ∂H/∂β_j)
                            // For logit in this path, H = Xᵀ W X + S.
                            let logh_beta_grad: Option<Array1<f64>> =
                                if let LinkFunction::Logit = self.config.link_function() {
                                    self.logh_beta_grad_logit(
                                        &pirls_result.x_transformed,
                                        &pirls_result.solve_mu,
                                        &pirls_result.solve_weights,
                                        &factor_g,
                                    )
                                } else {
                                    None
                                };

                            let mut grad_beta = if self.config.firth_bias_reduction {
                                // Chain-rule term for Firth-LAML:
                                //
                                //   ∂V/∂β = -∂l_p^*/∂β + 0.5 * ∂log|H_total|/∂β
                                //
                                // where l_p^* is the *actual* inner objective optimized by PIRLS
                                // (log-likelihood + Jeffreys adjustment - 0.5 βᵀ S β - 0.5 ridge ||β||²).
                                //
                                // At a perfect optimum, ∂l_p^*/∂β = 0 and the residual term vanishes.
                                // In practice, PIRLS stops at a tolerance and may add a stabilization ridge,
                                // so ∂l_p^*/∂β can be non-zero. Dropping it breaks the chain rule and makes
                                // the implicit correction term collapse (exactly the observed failure mode).
                                //
                                // The working response already includes the Jeffreys (Firth) score, so
                                // residual_grad is the correct score of the *inner* objective. Therefore
                                // the exact ∂V/∂β is:
                                //
                                //   residual_grad + 0.5 * ∂log|H_total|/∂β
                                //
                                // which is what we construct here.

                                // ## 3. The Full Gradient Expression
                                // Combining into the total derivative:
                                // dV/drho = Direct Terms + Implicit Correction
                                // Direct Terms = 0.5 * beta_quad + 0.5 * log|H| - 0.5 * log|S|
                                // Implicit Correction = (grad_beta)^T * (-H_total^-1 * lambda * S_k * beta)
                                let mut g = residual_grad.clone();

                                if let Some(logh_grad) = logh_beta_grad.as_ref() {
                                    g += &(0.5 * logh_grad);
                                }
                                g
                            } else {
                                // Non-Firth case matches standard LAML
                                residual_grad.clone()
                            };
                            if !self.config.firth_bias_reduction {
                                if let Some(logh_grad) = logh_beta_grad {
                                    // At the PIRLS optimum (with or without Firth), the
                                    // residual term cancels, leaving +0.5 * ∂log|H|/∂β.
                                    grad_beta += &(0.5 * &logh_grad);
                                    let res_inf = residual_grad
                                        .iter()
                                        .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
                                    let logh_inf =
                                        logh_grad.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
                                    let grad_inf =
                                        grad_beta.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
                                    if logh_inf < 1e-8 || grad_inf < 1e-8 {
                                        let (should_print, count) =
                                            should_emit_grad_diag(&GRAD_DIAG_BETA_COLLAPSE_COUNT);
                                        if should_print {
                                            eprintln!(
                                                "[GRAD DIAG #{count}] beta-grad collapse: max|residual|={:.3e} max|logh|={:.3e} max|grad_beta|={:.3e}",
                                                res_inf, logh_inf, grad_inf
                                            );
                                        }
                                    }
                                }
                            }

                            // Compute KKT residual norm to check if envelope theorem applies.
                            // The Implicit Function Theorem (used for delta_opt) assumes that β moves
                            // to maintain ∇V = 0 as ρ changes. If P-IRLS hasn't converged (large residual),
                            // β is effectively "stuck" on a ledge and doesn't move as predicted by IFT.
                            // In that case, we MUST skip the implicit correction to match reality.
                            let kkt_norm = residual_grad
                                .iter()
                                .fold(0.0_f64, |acc, &v| acc + v * v)
                                .sqrt();
                            let kkt_tol = self.config.convergence_tolerance.max(1e-4);
                            let kkt_ok = kkt_norm <= kkt_tol;

                            if !grad_beta.iter().all(|v| v.is_finite()) {
                                log::warn!(
                                    "Skipping IFT correction: non-finite gradient entries (kkt_norm={:.2e}).",
                                    kkt_norm
                                );
                            }
                            if !kkt_ok {
                                let (should_print, count) =
                                    should_emit_grad_diag(&GRAD_DIAG_KKT_SKIP_COUNT);
                                if should_print {
                                    eprintln!(
                                        "[GRAD DIAG #{count}] skipping IFT correction: kkt_norm={:.3e} tol={:.3e}",
                                        kkt_norm, kkt_tol
                                    );
                                }
                            }

                            let delta_opt = if grad_beta.iter().all(|v| v.is_finite()) && kkt_ok {
                                // IMPLICIT DERIVATIVE: d/dρ beta_hat = -H^-1 S_k beta.
                                // For spectral consistency with truncated log|H| we use H_+^\dagger.
                                // Apply in factor form to avoid dense H_+^\dagger materialization:
                                //   H_+^\dagger v = W (W^T v), W = U_+ diag(1/sqrt(lambda_+)).
                                let delta: Array1<f64> = if w_pos.ncols() == 0 {
                                    Array1::zeros(grad_beta.len())
                                } else {
                                    let wtg = w_pos.t().dot(&grad_beta);
                                    w_pos.dot(&wtg)
                                };

                                let delta_inf = delta
                                    .iter()
                                    .fold(0.0_f64, |acc: f64, &v: &f64| acc.max(v.abs()));
                                if delta_inf < 1e-8 {
                                    let (should_print, count) =
                                        should_emit_grad_diag(&GRAD_DIAG_DELTA_ZERO_COUNT);
                                    if should_print {
                                        eprintln!(
                                            "[GRAD DIAG #{count}] delta ~0: max|delta|={:.3e} max|grad_beta|={:.3e}",
                                            delta_inf,
                                            grad_beta
                                                .iter()
                                                .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
                                        );
                                    }
                                }
                                Some(delta)
                            } else {
                                None
                            };

                            for k in 0..k_count {
                                let log_det_h_grad_term = 0.5 * lambdas[k] * trace_terms[k];
                                let corrected_log_det_h =
                                    log_det_h_grad_term - truncation_corrections[k];
                                let log_det_s_grad_term = 0.5 * det1_values[k];

                                // REML gradient formula (Wood 2017, Section 6.5) / User Derivation Section 2.2:
                                //   ∂V/∂ρ_k = 0.5 * λ_k * β'S_k β   (penalty on coefficients)
                                //           + 0.5 * λ_k * tr(H⁻¹ S_k)  (Hessian log-det derivative)
                                //           - 0.5 * det1[k]            (penalty log-det derivative)
                                //
                                // Note: log_det_h_grad_term already contains the 0.5 factor and λ_k
                                // Note: det1_values[k] already contains λ_k * tr(S^{-1} S_k)
                                let mut gradient_value =
                                    0.5 * beta_terms[k] + corrected_log_det_h - log_det_s_grad_term;

                                // Add Implicit Correction (Section 2.1 & 4.3):
                                // term = (nabla_beta V)^T * (d_beta / d_rho)
                                //      = (grad_beta)^T * (-H^-1 * lambda * S_k * beta)
                                //      = - (H^-1 grad_beta)^T * (lambda * S_k * beta)
                                //      = - delta_opt^T * u_k

                                if let Some(delta_ref) = delta_opt.as_ref() {
                                    let r_k = &rs_transformed[k];
                                    let r_beta = r_k.dot(beta_ref);
                                    let s_k_beta = r_k.t().dot(&r_beta);
                                    let u_k: Array1<f64> = s_k_beta.mapv(|v| v * lambdas[k]);
                                    // Indirect term from chain rule:
                                    // dV/dρ_k = ∂V/∂ρ_k + (∇β V)ᵀ dβ/dρ_k.
                                    // Differentiate stationarity g = score - Sβ (+ Firth): ∂g/∂β = -H,
                                    // ∂g/∂ρ_k = -S_k β, so dβ/dρ_k = -H^{-1} S_k β and
                                    // the implicit correction is -(∇β V)ᵀ H^{-1} (S_k β) = -δᵀ u_k.
                                    let correction = -delta_ref.dot(&u_k);
                                    gradient_value += correction;
                                }
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

            // The optimizer MINIMIZES a cost function. The score is MAXIMIZED.
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

            Ok(gradient_result)
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
            let ridge_used = bundle.ridge_used;
            let beta = pirls_result.beta_transformed.as_ref();
            let lambdas: Array1<f64> = rho.mapv(f64::exp);

            // === Strategy 4: Dual-Ridge Consistency Check ===
            // The ridge used by PIRLS must match what gradient/cost assume
            let dual_ridge = compute_dual_ridge_check(
                pirls_result.ridge_used, // Ridge from PIRLS
                ridge_used,              // Ridge passed to cost
                ridge_used,              // Ridge passed to gradient (same bundle)
                beta,
            );
            report.dual_ridge = Some(dual_ridge);

            // === Strategy 1: KKT/Envelope Theorem Audit ===
            // Check if the inner solver actually reached stationarity
            // Compute score gradient (X'W(y-μ) for GLM) and penalty gradient (S_λ β)
            let reparam = &pirls_result.reparam_result;
            let penalty_grad = reparam.s_transformed.dot(beta);

            // Approximate score gradient using working residuals from PIRLS
            let eta = pirls_result.solve_mu.mapv(|m| {
                if m <= 1e-10 {
                    (-700.0_f64).max((m / (1.0 - m + 1e-10)).ln())
                } else if m >= 1.0 - 1e-10 {
                    (700.0_f64).min((m / (1.0 - m)).ln())
                } else {
                    (m / (1.0 - m)).ln()
                }
            });
            let working_residual = &pirls_result.solve_working_response - &eta;
            let weighted_residual = &pirls_result.solve_weights * &working_residual;
            let score_grad = pirls_result
                .x_transformed
                .transpose_vector_multiply(&weighted_residual);

            let envelope_audit = compute_envelope_audit(
                &score_grad,
                &penalty_grad,
                pirls_result.ridge_used,
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

            // 2. Compute LAML Hessian at perturbed rho
            // Finite difference on gradient
            let h_step = 1e-4;
            let n_rho = current_rho.len();
            let mut laml_hessian = Array2::<f64>::zeros((n_rho, n_rho));

            // We need the gradient at the perturbed point
            let grad_center = self.compute_gradient(&current_rho)?;

            for j in 0..n_rho {
                let mut rho_plus = current_rho.clone();
                rho_plus[j] += h_step;
                let grad_plus = self.compute_gradient(&rho_plus)?;

                // Use forward difference for Hessian columns: H_j approx (g(rho+h) - g(rho)) / h
                let col_diff = (&grad_plus - &grad_center) / h_step;
                for i in 0..n_rho {
                    laml_hessian[[i, j]] = col_diff[i];
                }
            }

            // Symmetrize
            for i in 0..n_rho {
                for j in 0..i {
                    let avg = 0.5 * (laml_hessian[[i, j]] + laml_hessian[[j, i]]);
                    laml_hessian[[i, j]] = avg;
                    laml_hessian[[j, i]] = avg;
                }
            }

            // Invert LAML Hessian to get V_rho
            // Use faer for robust inversion
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

            // 3. Compute Correction: J * V_rho * J^T
            // J = d beta / d rho = - H_p^-1 * [S_1 beta lambda_1, ..., S_k beta lambda_k]

            // We need H_p and beta at the perturbed rho.
            let pirls_res = self.execute_pirls_if_needed(&current_rho)?;

            let beta = pirls_res.beta_transformed.as_ref();
            let h_p = &pirls_res.penalized_hessian_transformed;
            let lambdas = current_rho.mapv(f64::exp);
            let rs = &pirls_res.reparam_result.rs_transformed;

            let p_dim = beta.len();

            // Invert H_p to get V_beta_cond (conditional covariance)
            let mut v_beta_cond = Array2::<f64>::zeros((p_dim, p_dim));
            {
                use crate::faer_ndarray::{FaerArrayView, array2_to_mat_mut};
                use faer::Side;
                let h_view = FaerArrayView::new(h_p);
                // H_p should be PD at convergence
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

            // Compute Jacobian columns: u_k = - V_beta_cond * (S_k * beta * lambda_k)
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

            // V_corr = J * V_rho * J^T
            let temp = jacobian.dot(&v_rho); // (p, k) * (k, k) -> (p, k)
            let v_corr = temp.dot(&jacobian.t()); // (p, k) * (k, p) -> (p, p)

            log::info!(
                "[Boundary] Correction computed. Max element in V_corr: {:.3e}",
                v_corr.iter().fold(0.0_f64, |a, &b| a.max(b.abs()))
            );

            // Total Covariance
            let v_total = v_beta_cond + v_corr;

            Ok((current_rho, Some(v_total)))
        }
    }
}
