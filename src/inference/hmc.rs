//! NUTS Sampler using general-mcmc
//!
//! This module provides NUTS (No-U-Turn Sampler) for honest uncertainty
//! quantification after PIRLS convergence.
//!
//! # Design
//!
//! Since general-mcmc's NUTS uses an identity mass matrix, we whiten the
//! parameter space using the Cholesky decomposition of the inverse Hessian:
//!
//! - Transform: β = μ + L @ z  (where L L^T = H^{-1})
//! - The whitened space has unit covariance, so NUTS mixes efficiently
//! - Samples are un-transformed back to the original space
//!
//! # Analytical Gradients
//!
//! We override `unnorm_logp_and_grad` to compute gradients analytically using
//! ndarray, avoiding burn's autodiff overhead. The gradient computation mirrors
//! the true log-posterior gradient (not the PIRLS working gradient).
//!
//! # Memory Efficiency
//!
//! Large data (design matrix, response, etc.) is wrapped in `Arc` to allow
//! sharing across chains without duplication when general-mcmc clones the target.

use super::polya_gamma::PolyaGamma;
use crate::construction::CanonicalPenalty;
use crate::estimate::reml::FirthDenseOperator;
use crate::estimate::reml::penalty_logdet::PenaltyPseudologdet;
use crate::faer_ndarray::{FaerCholesky, FaerEigh, fast_ata_into, fast_atv};
use crate::families::gamlss::monotone_wiggle_basis_with_derivative_order;
use crate::matrix::DesignMatrix;
use crate::solver::mixture_link::inverse_link_jet_for_family;
use crate::types::{InverseLink, LikelihoodFamily, LinkFunction, RhoPrior};
use crate::visualizer::VisualizerSession;
use faer::Side;
use general_mcmc::generic_hmc::HamiltonianTarget;
use general_mcmc::generic_nuts::{GenericNUTS, MassMatrixAdaptation, NUTSMassMatrixConfig};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

struct SamplingVisualizer {
    session: VisualizerSession,
    chain: usize,
    total_chains: usize,
    warmup: usize,
    samples: usize,
}

impl SamplingVisualizer {
    fn new(stage: &str, total_chains: usize, warmup: usize, samples: usize) -> Self {
        let mut session = VisualizerSession::new(true);
        session.set_stage("sample", stage);
        session.start_workflow(&format!("chains completed ({stage})"), total_chains.max(1));
        Self {
            session,
            chain: 0,
            total_chains,
            warmup,
            samples,
        }
    }

    fn begin_chain(&mut self, chain: usize, label: &str) {
        self.chain = chain;
        self.session
            .set_stage("sample", &format!("[Chain {}] {label}", chain + 1));
        self.session
            .start_workflow("chains completed", self.total_chains.max(1));
        self.session.advance_workflow(chain);
        self.session
            .start_secondary_workflow(&format!("[Chain {}] Warmup", chain + 1), self.warmup);
    }

    fn warmup_step(&mut self, iter: usize) {
        self.session
            .advance_secondary_workflow(iter.min(self.warmup));
    }

    fn start_sampling(&mut self) {
        self.session
            .finish_secondary_progress(&format!("chain {} warmup complete", self.chain + 1));
        self.session
            .start_secondary_workflow(&format!("[Chain {}] Sample", self.chain + 1), self.samples);
    }

    fn sample_step(&mut self, iter: usize) {
        self.session
            .advance_secondary_workflow(iter.min(self.samples));
    }

    fn finish_chain(&mut self, accept_rate: f64) {
        self.session.finish_secondary_progress(&format!(
            "chain {} sample complete | accepted {:.1}%",
            self.chain + 1,
            accept_rate * 100.0
        ));
        self.session
            .advance_workflow((self.chain + 1).min(self.total_chains));
    }

    fn finish_all(&mut self, rhat: f64, ess: f64) {
        self.session.push_diagnostic(&format!(
            "sampling diagnostics | rhat={rhat:.3} | ess={ess:.1}"
        ));
        self.session.finish_progress("sampling complete");
    }
}

/// Compute split-chain R-hat and ESS using the Gelman-Rubin diagnostic.
///
/// This is the standard split-chain formulation (no rank normalization).
/// Returns (max_rhat, min_ess) across dimensions.
fn compute_split_rhat_and_ess(samples: &Array3<f64>) -> (f64, f64) {
    let n_chains = samples.shape()[0];
    let n_samples = samples.shape()[1];
    let dim = samples.shape()[2];

    if n_chains < 2 || n_samples < 4 {
        return (1.0, n_chains as f64 * n_samples as f64 * 0.5);
    }

    // Split each chain in half to detect non-stationarity
    let half = n_samples / 2;
    let n_split_chains = n_chains * 2;
    let n_split_samples = half;

    let mut max_rhat = 0.0f64;
    let mut min_ess = f64::INFINITY;

    #[inline]
    fn splitvalue(
        samples: &Array3<f64>,
        n_chains: usize,
        half: usize,
        dim: usize,
        sc: usize,
        t: usize,
    ) -> f64 {
        let chain = sc % n_chains;
        if sc < n_chains {
            samples[[chain, t, dim]]
        } else {
            samples[[chain, half + t, dim]]
        }
    }

    fn ess_from_split_dimension(
        samples: &Array3<f64>,
        n_chains: usize,
        half: usize,
        dim: usize,
    ) -> f64 {
        let m = n_chains * 2;
        let n = half;
        if m == 0 || n < 4 {
            return (m * n).max(1) as f64;
        }

        let mut means = vec![0.0_f64; m];
        let mut gamma0 = vec![0.0_f64; m];
        for sc in 0..m {
            let mut sum = 0.0;
            for t in 0..n {
                sum += splitvalue(samples, n_chains, half, dim, sc, t);
            }
            let mean = sum / n as f64;
            means[sc] = mean;
            let mut g0 = 0.0;
            for t in 0..n {
                let d = splitvalue(samples, n_chains, half, dim, sc, t) - mean;
                g0 += d * d;
            }
            gamma0[sc] = (g0 / n as f64).max(1e-16);
        }

        let max_lag = (n - 1).min(1000);
        let mut tau = 1.0_f64;
        let mut lag = 1usize;
        while lag < max_lag {
            let mut pair = 0.0_f64;
            for l in [lag, lag + 1] {
                if l > max_lag {
                    continue;
                }
                let mut rho_l = 0.0;
                for sc in 0..m {
                    let mu = means[sc];
                    let mut cov = 0.0;
                    let denom = (n - l) as f64;
                    for t in 0..(n - l) {
                        let x0 = splitvalue(samples, n_chains, half, dim, sc, t);
                        let x1 = splitvalue(samples, n_chains, half, dim, sc, t + l);
                        cov += (x0 - mu) * (x1 - mu);
                    }
                    cov /= denom;
                    rho_l += cov / gamma0[sc];
                }
                rho_l /= m as f64;
                pair += rho_l;
            }
            if !pair.is_finite() || pair <= 0.0 {
                break;
            }
            tau += 2.0 * pair;
            lag += 2;
        }
        if !tau.is_finite() || tau <= 0.0 {
            return 1.0;
        }
        let total = (m * n) as f64;
        (total / tau).clamp(1.0, total)
    }

    let mut chain_means = vec![0.0_f64; n_split_chains];
    let mut chainvars = vec![0.0_f64; n_split_chains];
    for d in 0..dim {
        for chain in 0..n_chains {
            // First half
            let mut sum1 = 0.0;
            for i in 0..half {
                sum1 += samples[[chain, i, d]];
            }
            let mean1 = sum1 / half as f64;
            let mut var1 = 0.0;
            for i in 0..half {
                let diff = samples[[chain, i, d]] - mean1;
                var1 += diff * diff;
            }
            var1 /= (half - 1).max(1) as f64;
            let first_idx = chain;
            chain_means[first_idx] = mean1;
            chainvars[first_idx] = var1;

            // Second half
            let mut sum2 = 0.0;
            for i in half..(2 * half) {
                sum2 += samples[[chain, i, d]];
            }
            let mean2 = sum2 / half as f64;
            let mut var2 = 0.0;
            for i in half..(2 * half) {
                let diff = samples[[chain, i, d]] - mean2;
                var2 += diff * diff;
            }
            var2 /= (half - 1).max(1) as f64;
            let second_idx = n_chains + chain;
            chain_means[second_idx] = mean2;
            chainvars[second_idx] = var2;
        }

        // Within-chain variance W
        let w: f64 = chainvars.iter().copied().sum::<f64>() / n_split_chains as f64;

        // Between-chain variance B
        let overall_mean: f64 = chain_means.iter().copied().sum::<f64>() / n_split_chains as f64;
        let b: f64 = chain_means
            .iter()
            .map(|m| (m - overall_mean).powi(2))
            .sum::<f64>()
            * n_split_samples as f64
            / (n_split_chains - 1) as f64;

        // Estimated variance
        let var_hat = (n_split_samples as f64 - 1.0) / n_split_samples as f64 * w
            + b / n_split_samples as f64;

        // R-hat
        let rhat_d = if w > 1e-10 { (var_hat / w).sqrt() } else { 1.0 };
        max_rhat = max_rhat.max(rhat_d);

        // Real ESS via split-chain autocorrelation with Geyer IPS truncation.
        let ess_d = ess_from_split_dimension(samples, n_chains, half, d);
        min_ess = min_ess.min(ess_d);
    }

    (max_rhat, min_ess.max(1.0))
}

/// Solve L^T * X = I where L is lower triangular.
///
/// Returns X = L^{-T} (the inverse transpose of L).
/// Uses back-substitution since L^T is upper triangular.
///
/// This is the correct way to compute the whitening transform matrix:
/// Given H = L L^T (Cholesky), we need W where W W^T = H^{-1}
/// Since H^{-1} = L^{-T} L^{-1}, we have W = L^{-T}
fn solve_upper_triangular_transpose(l: &Array2<f64>, dim: usize) -> Array2<f64> {
    // L^T is upper triangular, so we solve L^T * X = I via back-substitution
    let mut result = Array2::<f64>::zeros((dim, dim));

    // For each column of the identity (each column of result)
    for col in 0..dim {
        // Solve L^T * x = e_col (unit vector)
        // Back-substitution: start from last row, work up
        for i in (0..dim).rev() {
            let mut sum = if i == col { 1.0 } else { 0.0 }; // e_col[i]

            // Subtract contributions from already-solved entries
            for j in (i + 1)..dim {
                sum -= l[[j, i]] * result[[j, col]]; // L^T[i,j] = L[j,i]
            }

            // Divide by diagonal (L^T[i,i] = L[i,i])
            let diag = l[[i, i]];
            if diag.abs() < 1e-15 {
                result[[i, col]] = 0.0; // Regularize near-zero diagonal
            } else {
                result[[i, col]] = sum / diag;
            }
        }
    }

    result
}

/// Shared data for NUTS posterior (wrapped in Arc to prevent cloning).
///
/// This struct holds read-only data that is shared across all chains.
/// Using Arc prevents memory explosion when general-mcmc clones the target.
#[derive(Clone)]
struct SharedData {
    /// Design matrix X [n_samples, dim]
    x: Arc<Array2<f64>>,
    /// Response vector y [n_samples]
    y: Arc<Array1<f64>>,
    /// Observation/case weights [n_samples]
    weights: Arc<Array1<f64>>,
    /// Combined penalty matrix S [dim, dim]
    penalty: Arc<Array2<f64>>,
    /// MAP estimate (mode) μ [dim]
    mode: Arc<Array1<f64>>,
    /// Fitted Gamma shape (used only for Gamma-log likelihoods).
    gamma_shape: f64,
    /// Number of samples
    n_samples: usize,
    /// Number of coefficients
    dim: usize,
}

/// Whitened log-posterior target with analytical gradients.
///
/// Uses Arc for shared data to prevent memory explosion when cloned for chains.
/// Uses faer for numerically stable Cholesky decomposition.
/// Family mode for NUTS log-likelihood computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NutsFamily {
    Gaussian,
    BinomialLogit,
    BinomialProbit,
    BinomialCLogLog,
    PoissonLog,
    GammaLog,
}

impl NutsFamily {
    #[inline]
    fn likelihood_family(self) -> LikelihoodFamily {
        match self {
            Self::Gaussian => LikelihoodFamily::GaussianIdentity,
            Self::BinomialLogit => LikelihoodFamily::BinomialLogit,
            Self::BinomialProbit => LikelihoodFamily::BinomialProbit,
            Self::BinomialCLogLog => LikelihoodFamily::BinomialCLogLog,
            Self::PoissonLog => LikelihoodFamily::PoissonLog,
            Self::GammaLog => LikelihoodFamily::GammaLog,
        }
    }
}

pub struct NutsPosterior {
    /// Shared read-only data (Arc prevents duplication)
    data: SharedData,
    /// Transform: L where L L^T = H^{-1} (computed from Hessian)
    /// This is the inverse-transpose of the Cholesky of H.
    chol: Array2<f64>,
    /// L^T for gradient chain rule: ∇z = L^T @ ∇_β
    chol_t: Array2<f64>,
    /// Family for log-likelihood computation
    nuts_family: NutsFamily,
    /// Whether to add the identifiable-subspace Jeffreys/Firth term to the
    /// target
    firth_enabled: bool,
}

impl NutsPosterior {
    #[inline]
    fn log1pexp(eta: f64) -> f64 {
        if eta > 0.0 {
            eta + (-eta).exp().ln_1p()
        } else {
            eta.exp().ln_1p()
        }
    }

    #[inline]
    fn sigmoid_stable(eta: f64) -> f64 {
        if eta >= 0.0 {
            1.0 / (1.0 + (-eta).exp())
        } else {
            let e = eta.exp();
            e / (1.0 + e)
        }
    }

    /// Creates a new posterior target from ndarray data.
    ///
    /// # Arguments
    /// * `x` - Design matrix [n_samples, dim]
    /// * `y` - Response vector [n_samples]
    /// * `weights` - Observation/case weights [n_samples]
    /// * `penalty_matrix` - Combined penalty S [dim, dim]
    /// * `mode` - MAP estimate μ [dim]
    /// * `hessian` - Hessian H [dim, dim] (NOT the inverse!)
    /// * `nuts_family` - Family for log-likelihood computation
    ///
    /// # Numerical Stability
    /// Accepts the Hessian directly and computes L = (chol(H))^{-T} via
    /// triangular solves, which is more stable than explicitly inverting H.
    pub fn new(
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        weights: ArrayView1<f64>,
        penalty_matrix: ArrayView2<f64>,
        mode: ArrayView1<f64>,
        hessian: ArrayView2<f64>,
        nuts_family: NutsFamily,
        gamma_shape: f64,
        firth_enabled: bool,
    ) -> Result<Self, String> {
        let n_samples = x.nrows();
        let dim = x.ncols();

        // Validate inputs are finite
        if !penalty_matrix.iter().all(|x| x.is_finite()) {
            return Err("Penalty matrix contains NaN or Inf values".to_string());
        }
        if !hessian.iter().all(|x| x.is_finite()) {
            return Err("Hessian matrix contains NaN or Inf values".to_string());
        }
        if !mode.iter().all(|x| x.is_finite()) {
            return Err("Mode vector contains NaN or Inf values".to_string());
        }

        validate_firth_support(nuts_family, firth_enabled)?;

        // Use faer for numerically stable Cholesky decomposition of H
        // H = L_H L_H^T where L_H is lower triangular
        let hessian_owned = hessian.to_owned();
        let chol_factor = hessian_owned
            .cholesky(Side::Lower)
            .map_err(|e| format!("Hessian Cholesky decomposition failed: {:?}", e))?;

        // We need L where L L^T = H^{-1}
        // Since H = L_H L_H^T, we have H^{-1} = L_H^{-T} L_H^{-1}
        // So L = L_H^{-T} (the inverse transpose of the Cholesky factor)
        //
        // To get L_H^{-T}, we solve L_H^T * X = I using back-substitution
        // Since L_H is lower triangular, L_H^T is upper triangular
        let l_h = chol_factor.lower_triangular();
        let chol = solve_upper_triangular_transpose(&l_h, dim);
        let chol_t = chol.t().to_owned();

        let data = SharedData {
            x: Arc::new(x.to_owned()),
            y: Arc::new(y.to_owned()),
            weights: Arc::new(weights.to_owned()),
            penalty: Arc::new(penalty_matrix.to_owned()),
            mode: Arc::new(mode.to_owned()),
            gamma_shape,
            n_samples,
            dim,
        };

        Ok(Self {
            data,
            chol,
            chol_t,
            nuts_family,
            firth_enabled,
        })
    }

    /// Compute log-posterior and gradient analytically using ndarray.
    ///
    /// Returns (log_posterior, gradientz) where gradientz is the gradient
    /// with respect to the whitened parameters z.
    fn compute_logp_and_grad_nd(&self, z: &Array1<f64>) -> (f64, Array1<f64>) {
        // === Step 1: Transform z (whitened) -> β (original) ===
        // β = μ + L @ z
        let beta = self.data.mode.as_ref() + &self.chol.dot(z);

        // === Step 2: Compute η = X @ β ===
        let eta = self.data.x.dot(&beta);

        // === Step 3: Compute log-likelihood and gradient ===
        let (ll, mut grad_ll_beta) = self.family_logp_and_grad(&eta);

        let mut firth_logdet = 0.0;
        if self.firth_enabled {
            match firth_jeffreys_logp_and_grad(self.nuts_family, &self.data, &eta) {
                Ok((value, grad_beta_firth)) => {
                    firth_logdet = value;
                    grad_ll_beta += &grad_beta_firth;
                }
                Err(err) => {
                    log::warn!(
                        "[NUTS/Firth] Jeffreys target became invalid at the current state: {}",
                        err
                    );
                    return (f64::NEG_INFINITY, Array1::zeros(z.len()));
                }
            }
        }

        // === Step 4: Compute penalty and its gradient ===
        // penalty = 0.5 * β^T @ S @ β
        let s_beta = self.data.penalty.dot(&beta);
        let penalty = 0.5 * beta.dot(&s_beta);

        // ∇_β penalty = S @ β
        let grad_penalty_beta = s_beta;

        // === Step 5: Combined gradient in β space ===
        // ∇_β log p = ∇_β ll - ∇_β penalty
        let grad_beta = &grad_ll_beta - &grad_penalty_beta;

        // === Step 6: Chain rule to get gradient in z space ===
        // ∇z = L^T @ ∇_β
        let gradz = self.chol_t.dot(&grad_beta);

        let logp = ll + firth_logdet - penalty;

        (logp, gradz)
    }

    /// Dispatch log-likelihood and β-gradient computation to the appropriate family.
    fn family_logp_and_grad(&self, eta: &Array1<f64>) -> (f64, Array1<f64>) {
        nuts_family_logp_and_grad(self.nuts_family, &self.data, eta)
    }

    /// Get the Cholesky factor L for un-whitening samples
    pub fn chol(&self) -> &Array2<f64> {
        &self.chol
    }

    /// Get the mode
    pub fn mode(&self) -> &Array1<f64> {
        &self.data.mode
    }

    /// Get dimension
    pub fn dim(&self) -> usize {
        self.data.dim
    }
}

const HALF_LOG_2PI: f64 = 0.918_938_533_204_672_7;

#[inline]
fn standard_normal_log_pdf(x: f64) -> f64 {
    -0.5 * x * x - HALF_LOG_2PI
}

/// Stable log Φ(x) for the standard normal CDF.
#[inline]
fn log_ndtr(x: f64) -> f64 {
    let arg = -x * std::f64::consts::FRAC_1_SQRT_2;
    let erfc_val = statrs::function::erf::erfc(arg);
    if erfc_val > 0.0 {
        erfc_val.ln() - std::f64::consts::LN_2
    } else {
        -0.5 * x * x - (-x).ln() - HALF_LOG_2PI
    }
}

#[inline]
fn validate_firth_support(family: NutsFamily, firth_enabled: bool) -> Result<(), String> {
    if firth_enabled && !family.likelihood_family().supports_firth() {
        return Err(format!(
            "NUTS with Firth is only supported for {}; {} does not support it",
            LikelihoodFamily::BinomialLogit.pretty_name(),
            family.likelihood_family().pretty_name()
        ));
    }
    Ok(())
}

#[inline]
fn validate_firth_likelihood_support(
    family: LikelihoodFamily,
    firth_enabled: bool,
) -> Result<(), String> {
    if firth_enabled && !family.supports_firth() {
        return Err(format!(
            "Joint HMC with Firth is only supported for {}; {} does not support it",
            LikelihoodFamily::BinomialLogit.pretty_name(),
            family.pretty_name()
        ));
    }
    Ok(())
}

/// Compute the identifiable-subspace Jeffreys/Firth contribution and its
/// β-gradient.
///
/// HMC uses the same `FirthDenseOperator` as the REML exact-gradient path.
/// The operator owns the reduced identifiable Fisher factorization, the
/// Jeffreys log-determinant, and the analytic β-gradient.
fn firth_jeffreys_logp_and_grad(
    family: NutsFamily,
    data: &SharedData,
    eta: &Array1<f64>,
) -> Result<(f64, Array1<f64>), String> {
    if eta.len() != data.n_samples {
        return Err(format!(
            "Firth Jeffreys term eta length {} != number of samples {}",
            eta.len(),
            data.n_samples
        ));
    }
    if data.dim == 0 || data.n_samples == 0 {
        return Ok((0.0, Array1::zeros(data.dim)));
    }
    validate_firth_support(family, true)?;
    if data.weights.iter().all(|w| *w == 0.0) {
        return Ok((0.0, Array1::zeros(data.dim)));
    }

    let op = if data.weights.iter().all(|&w| w == 1.0) {
        FirthDenseOperator::build(data.x.as_ref(), eta)
    } else {
        FirthDenseOperator::build_with_observation_weights(
            data.x.as_ref(),
            eta,
            data.weights.view(),
        )
    }
    .map_err(|e| format!("Firth Jeffreys operator failed: {e}"))?;
    Ok(op.jeffreys_logdet_and_beta_gradient())
}

// ============================================================================
// Shared family log-likelihood helpers
// ============================================================================
//
// Freestanding functions for computing ℓ(y|β) and ∇_β ℓ for each supported
// family. Used by both `NutsPosterior` (fixed-ρ β-only sampling) and
// `JointBetaRhoPosterior` (joint β+ρ sampling).

/// Dispatch log-likelihood and ∇_β computation to the appropriate family.
fn nuts_family_logp_and_grad(
    family: NutsFamily,
    data: &SharedData,
    eta: &Array1<f64>,
) -> (f64, Array1<f64>) {
    match family {
        NutsFamily::BinomialLogit => logit_logp_and_grad(data, eta),
        NutsFamily::BinomialProbit => probit_logp_and_grad(data, eta),
        NutsFamily::BinomialCLogLog => cloglog_logp_and_grad(data, eta),
        NutsFamily::Gaussian => gaussian_logp_and_grad(data, eta),
        NutsFamily::PoissonLog => poisson_log_logp_and_grad(data, eta),
        NutsFamily::GammaLog => gamma_log_logp_and_grad(data, eta),
    }
}

fn joint_binomial_logp_and_grad(
    family: LikelihoodFamily,
    inverse_link: &InverseLink,
    data: &SharedData,
    eta: &Array1<f64>,
) -> Result<(f64, Array1<f64>), String> {
    let n = data.n_samples;
    let mut ll = 0.0;
    let mut residual = Array1::<f64>::zeros(n);

    for i in 0..n {
        let jet = inverse_link_jet_for_family(
            family,
            eta[i],
            inverse_link.mixture_state(),
            inverse_link.sas_state(),
        )
        .map_err(|err| err.to_string())?;
        let mu = jet.mu.clamp(1.0e-15, 1.0 - 1.0e-15);
        let dmu_deta = jet.d1;
        let y_i = data.y[i];
        let w_i = data.weights[i];
        ll += w_i * (y_i * mu.ln() + (1.0 - y_i) * (1.0 - mu).ln());
        residual[i] = w_i * (y_i - mu) * dmu_deta / (mu * (1.0 - mu)).max(1.0e-30);
    }

    Ok((ll, fast_atv(&data.x, &residual)))
}

fn joint_family_logp_and_grad(
    family: LikelihoodFamily,
    inverse_link: &InverseLink,
    data: &SharedData,
    eta: &Array1<f64>,
) -> Result<(f64, Array1<f64>), String> {
    match family {
        LikelihoodFamily::BinomialLogit
        | LikelihoodFamily::BinomialProbit
        | LikelihoodFamily::BinomialCLogLog
        | LikelihoodFamily::BinomialLatentCLogLog
        | LikelihoodFamily::BinomialSas
        | LikelihoodFamily::BinomialBetaLogistic
        | LikelihoodFamily::BinomialMixture => {
            joint_binomial_logp_and_grad(family, inverse_link, data, eta)
        }
        LikelihoodFamily::GaussianIdentity => Ok(gaussian_logp_and_grad(data, eta)),
        LikelihoodFamily::PoissonLog => Ok(poisson_log_logp_and_grad(data, eta)),
        LikelihoodFamily::GammaLog => Ok(gamma_log_logp_and_grad(data, eta)),
        LikelihoodFamily::RoystonParmar => {
            Err("Joint HMC fallback is not implemented for RoystonParmar".to_string())
        }
    }
}

/// Logistic regression log-likelihood and gradient.
///
/// log p(y|η) = y·η − log(1 + exp(η)), gradient = X'(w ⊙ (y − μ))
fn logit_logp_and_grad(data: &SharedData, eta: &Array1<f64>) -> (f64, Array1<f64>) {
    let n = data.n_samples;
    let mut ll = 0.0;
    let mut residual = Array1::<f64>::zeros(n);

    for i in 0..n {
        let eta_i = eta[i];
        let y_i = data.y[i];
        let w_i = data.weights[i];
        ll += w_i * (y_i * eta_i - NutsPosterior::log1pexp(eta_i));
        let mu = NutsPosterior::sigmoid_stable(eta_i);
        residual[i] = w_i * (y_i - mu);
    }

    let grad_ll = fast_atv(&data.x, &residual);
    (ll, grad_ll)
}

/// Probit regression log-likelihood and gradient.
///
/// log p(y|η) = Σ [y·log Φ(η) + (1-y)·log(1-Φ(η))],
/// gradient_i = w_i · [y_i · φ(η_i)/Φ(η_i) − (1-y_i) · φ(η_i)/(1−Φ(η_i))]
///
/// Uses erfc-based log Φ for numerical stability.
fn probit_logp_and_grad(data: &SharedData, eta: &Array1<f64>) -> (f64, Array1<f64>) {
    let n = data.n_samples;
    let mut ll = 0.0;
    let mut residual = Array1::<f64>::zeros(n);

    for i in 0..n {
        let eta_i = eta[i];
        let y_i = data.y[i];
        let w_i = data.weights[i];

        // log Φ(η) and log(1−Φ(η)) = log Φ(−η)
        let log_phi_pos = log_ndtr(eta_i);
        let log_phi_neg = log_ndtr(-eta_i);

        ll += w_i * (y_i * log_phi_pos + (1.0 - y_i) * log_phi_neg);

        // Gradient: y · φ/Φ − (1−y) · φ/(1−Φ)
        // Compute ratios via exp(log φ − log Φ) for stability.
        // log φ(x) = −x²/2 − 0.5·ln(2π)
        let log_phi_val = standard_normal_log_pdf(eta_i);

        // Inverse Mills ratio: φ(η)/Φ(η) = exp(log φ − log Φ)
        let ratio_pos = (log_phi_val - log_phi_pos).exp();
        // φ(η)/(1−Φ(η)) = exp(log φ − log Φ(−η))
        let ratio_neg = (log_phi_val - log_phi_neg).exp();

        let grad_i = y_i * ratio_pos - (1.0 - y_i) * ratio_neg;
        residual[i] = w_i * grad_i;
    }

    let grad_ll = fast_atv(&data.x, &residual);
    (ll, grad_ll)
}

/// Complementary log-log regression log-likelihood and gradient.
///
/// CLogLog link: μ = 1 − exp(−exp(η))
/// log p(y|η) = Σ [y·log(1−exp(−exp(η))) + (1−y)·(−exp(η))]
/// gradient_i = w_i · [y_i · exp(η_i)·exp(−exp(η_i)) / (1−exp(−exp(η_i))) − (1−y_i)·exp(η_i)]
fn cloglog_logp_and_grad(data: &SharedData, eta: &Array1<f64>) -> (f64, Array1<f64>) {
    let n = data.n_samples;
    let mut ll = 0.0;
    let mut residual = Array1::<f64>::zeros(n);

    for i in 0..n {
        let eta_i = eta[i].clamp(-700.0, 700.0);
        let y_i = data.y[i];
        let w_i = data.weights[i];

        let exp_eta = eta_i.exp();
        // −exp(η) clamped to avoid overflow in exp(−exp(η))
        let neg_exp_eta = (-exp_eta).clamp(-700.0, 0.0);

        // log(1 − exp(−exp(η))): use log1p(−exp(−exp(η))) = log1p(exp(−exp(η))·(−1))
        // For small exp(η), exp(−exp(η)) ≈ 1 − exp(η), so 1−exp(−exp(η)) ≈ exp(η)
        // and log(1−exp(−exp(η))) ≈ η
        let exp_neg_exp_eta = neg_exp_eta.exp(); // exp(−exp(η))
        let log_mu = if exp_eta < 1e-6 {
            // Small exp(η): log(1 - exp(-exp(η))) ≈ log(exp(η) - exp(η)²/2)
            // ≈ η + log(1 - exp(η)/2) ≈ η for very small
            (-exp_neg_exp_eta).ln_1p()
        } else {
            (-exp_neg_exp_eta).ln_1p()
        };

        // log(1−μ) = log(exp(−exp(η))) = −exp(η)
        let log_one_minus_mu = -exp_eta;

        ll += w_i * (y_i * log_mu + (1.0 - y_i) * log_one_minus_mu);

        // Gradient: d/dη [y·log(μ) + (1-y)·log(1-μ)]
        // = y · exp(η)·exp(−exp(η)) / (1−exp(−exp(η))) + (1-y)·(−exp(η))
        //
        // The ratio exp(η)·exp(−exp(η)) / (1−exp(−exp(η))) can be computed as
        // exp(η) · exp(log(exp(−exp(η))) − log(1−exp(−exp(η))))
        // = exp(η) · exp(−exp(η) − log_mu)
        let grad_y1 = if log_mu.is_finite() && log_mu > -700.0 {
            exp_eta * (neg_exp_eta - log_mu).exp()
        } else {
            // Fallback for extreme values: μ ≈ 0, ratio → 1
            1.0
        };
        let grad_y0 = -exp_eta;

        residual[i] = w_i * (y_i * grad_y1 + (1.0 - y_i) * grad_y0);
    }

    let grad_ll = fast_atv(&data.x, &residual);
    (ll, grad_ll)
}

/// Gaussian log-likelihood and gradient.
///
/// log p(y|η) = −½ w·(y − η)², gradient = X'(w ⊙ (y − η))
fn gaussian_logp_and_grad(data: &SharedData, eta: &Array1<f64>) -> (f64, Array1<f64>) {
    let n = data.n_samples;
    let mut ll = 0.0;
    let mut weighted_residual = Array1::<f64>::zeros(n);

    for i in 0..n {
        let residual = data.y[i] - eta[i];
        let w_i = data.weights[i];
        ll -= 0.5 * w_i * residual * residual;
        weighted_residual[i] = w_i * residual;
    }

    let grad_ll = fast_atv(&data.x, &weighted_residual);
    (ll, grad_ll)
}

/// Poisson(log) log-likelihood and gradient.
///
/// log p(y|η) = y·η − exp(η), gradient = X'(w ⊙ (y − μ))
fn poisson_log_logp_and_grad(data: &SharedData, eta: &Array1<f64>) -> (f64, Array1<f64>) {
    let n = data.n_samples;
    let mut ll = 0.0;
    let mut residual = Array1::<f64>::zeros(n);

    for i in 0..n {
        let eta_i = eta[i].clamp(-700.0, 700.0);
        let mu_i = eta_i.exp();
        let y_i = data.y[i];
        let w_i = data.weights[i];
        ll += w_i * (y_i * eta_i - mu_i);
        residual[i] = w_i * (y_i - mu_i);
    }

    let grad_ll = fast_atv(&data.x, &residual);
    (ll, grad_ll)
}

fn gamma_log_logp_and_grad(data: &SharedData, eta: &Array1<f64>) -> (f64, Array1<f64>) {
    let n = data.n_samples;
    let mut ll = 0.0;
    let mut residual = Array1::<f64>::zeros(n);
    let shape = data.gamma_shape.max(1e-10);

    for i in 0..n {
        let eta_i = eta[i].clamp(-700.0, 700.0);
        let mu_i = eta_i.exp();
        let y_i = data.y[i];
        let w_i = data.weights[i];
        ll += w_i
            * (shape * shape.ln() - statrs::function::gamma::ln_gamma(shape) - shape * eta_i
                + (shape - 1.0) * y_i.max(1e-12).ln()
                - shape * y_i / mu_i);
        residual[i] = w_i * shape * (y_i / mu_i - 1.0);
    }

    let grad_ll = fast_atv(&data.x, &residual);
    (ll, grad_ll)
}

#[cfg(test)]
mod tests {
    use super::{
        FamilyNutsInputs, GlmFlatInputs, JointBetaRhoInputs, JointBetaRhoPosterior, NutsConfig,
        NutsFamily, NutsPosterior, SharedData, firth_jeffreys_logp_and_grad,
        joint_family_logp_and_grad, laplace_directional_cubic_diagnostic,
        run_joint_beta_rho_sampling, run_logit_polya_gamma_gibbs,
        run_nuts_sampling_flattened_family,
    };
    use crate::construction::CanonicalPenalty;
    use crate::matrix::DesignMatrix;
    use crate::survival::{MonotonicityPenalty, PenaltyBlocks, SurvivalSpec};
    use crate::types::{InverseLink, LikelihoodFamily, LinkFunction, RhoPrior};
    use general_mcmc::generic_hmc::HamiltonianTarget;
    use ndarray::{Array1, Array2, array};
    use std::sync::Arc;

    #[test]
    fn log1pexp_is_finite_for_extreme_eta() {
        assert!(NutsPosterior::log1pexp(1000.0).is_finite());
        assert!(NutsPosterior::log1pexp(-1000.0).is_finite());
        assert!((NutsPosterior::log1pexp(-1000.0) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn sigmoid_stable_behaves_at_extremes() {
        let hi = NutsPosterior::sigmoid_stable(1000.0);
        let lo = NutsPosterior::sigmoid_stable(-1000.0);
        assert!(hi <= 1.0 && hi >= 1.0 - 1e-12);
        assert!(lo >= 0.0 && lo <= 1e-12);
    }

    #[test]
    fn nuts_logitgradient_matches_finite_difference() {
        let x = array![[1.0, -0.5], [0.2, 0.7], [-1.0, 0.3], [0.5, -1.2]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let w = array![1.0, 1.5, 0.8, 1.2];
        let penalty = array![[0.4, 0.0], [0.0, 0.6]];
        let mode = array![0.1, -0.2];
        let hessian = array![[2.0, 0.2], [0.2, 1.7]]; // SPD

        let posterior = NutsPosterior::new(
            x.view(),
            y.view(),
            w.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            NutsFamily::BinomialLogit,
            1.0,
            true,
        )
        .expect("posterior");

        let z = array![0.15, -0.35];
        let (_, grad) = posterior.compute_logp_and_grad_nd(&z);

        let eps = 1e-6;
        for j in 0..z.len() {
            let mut z_plus = z.clone();
            let mut z_minus = z.clone();
            z_plus[j] += eps;
            z_minus[j] -= eps;
            let (lp, _) = posterior.compute_logp_and_grad_nd(&z_plus);
            let (lm, _) = posterior.compute_logp_and_grad_nd(&z_minus);
            let fd = (lp - lm) / (2.0 * eps);
            assert_eq!(
                grad[j].signum(),
                fd.signum(),
                "gradient sign mismatch at {}: analytic={}, fd={}",
                j,
                grad[j],
                fd
            );
            assert!(
                (grad[j] - fd).abs() < 1e-5,
                "gradient mismatch at {}: analytic={}, fd={}",
                j,
                grad[j],
                fd
            );
        }
    }

    #[test]
    fn gamma_log_logp_and_grad_uses_fitted_shape() {
        let x = array![[1.0_f64], [1.0_f64]];
        let y = array![1.5_f64, 2.5_f64];
        let weights = array![1.0_f64, 2.0_f64];
        let eta = array![0.2_f64, 0.4_f64];
        let shape = 3.5_f64;
        let data = SharedData {
            x: Arc::new(x.clone()),
            y: Arc::new(y.clone()),
            weights: Arc::new(weights.clone()),
            penalty: Arc::new(Array2::zeros((1, 1))),
            mode: Arc::new(Array1::zeros(1)),
            gamma_shape: shape,
            n_samples: x.nrows(),
            dim: x.ncols(),
        };

        let (ll, grad) = super::gamma_log_logp_and_grad(&data, &eta);

        let mut expected_ll = 0.0;
        let mut expected_score = 0.0;
        for i in 0..eta.len() {
            let mu = eta[i].exp();
            expected_ll += weights[i]
                * (shape * shape.ln() - statrs::function::gamma::ln_gamma(shape) - shape * eta[i]
                    + (shape - 1.0) * y[i].ln()
                    - shape * y[i] / mu);
            expected_score += weights[i] * shape * (y[i] / mu - 1.0);
        }

        assert!((ll - expected_ll).abs() < 1e-12);
        assert_eq!(grad.len(), 1);
        assert!((grad[0] - expected_score).abs() < 1e-12);
    }

    #[test]
    fn firth_jeffreys_logit_is_finite_for_rank_deficient_design() {
        let x = array![
            [1.0, -0.5, 1.0],
            [1.0, 0.3, 1.0],
            [1.0, 0.8, 1.0],
            [1.0, -1.2, 1.0],
        ];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let weights = array![1.0, 2.0, 0.5, 1.5];
        let eta = array![0.2, -0.1, 0.4, -0.3];

        let data = SharedData {
            x: Arc::new(x.clone()),
            y: Arc::new(y),
            weights: Arc::new(weights.clone()),
            penalty: Arc::new(Array2::zeros((x.ncols(), x.ncols()))),
            mode: Arc::new(Array1::zeros(x.ncols())),
            gamma_shape: 1.0,
            n_samples: x.nrows(),
            dim: x.ncols(),
        };

        let (value, grad) =
            firth_jeffreys_logp_and_grad(NutsFamily::BinomialLogit, &data, &eta).expect("firth");

        assert!(value.is_finite());
        assert_eq!(grad.len(), x.ncols());
        assert!(grad.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn logit_pg_gibbs_returns_finite_samples() {
        let x = array![[1.0, 0.2], [1.0, -0.1], [1.0, 1.2], [1.0, -0.7]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let penalty = array![[0.2, 0.0], [0.0, 0.4]];
        let mode = array![0.0, 0.0];
        let cfg = NutsConfig {
            n_samples: 30,
            nwarmup: 30,
            n_chains: 2,
            target_accept: 0.8,
            seed: 123,
        };
        let out = run_logit_polya_gamma_gibbs(
            x.view(),
            y.view(),
            w.view(),
            penalty.view(),
            mode.view(),
            &cfg,
        )
        .expect("pg gibbs should run");
        assert_eq!(out.samples.ncols(), 2);
        assert_eq!(out.samples.nrows(), cfg.n_samples * cfg.n_chains);
        assert!(out.samples.iter().all(|v| v.is_finite()));
        assert!(out.posterior_mean.iter().all(|v| v.is_finite()));
        assert!(out.posterior_std.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn family_dispatch_uses_pg_gibbs_for_standard_logit() {
        let x = array![[1.0, 0.2], [1.0, -0.1], [1.0, 1.2], [1.0, -0.7]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let penalty = array![[0.2, 0.0], [0.0, 0.4]];
        let mode = array![0.0, 0.0];
        let non_spdhessian = array![[0.0, 0.0], [0.0, 0.0]];
        let cfg = NutsConfig {
            n_samples: 20,
            nwarmup: 20,
            n_chains: 2,
            target_accept: 0.8,
            seed: 456,
        };
        let out = run_nuts_sampling_flattened_family(
            LikelihoodFamily::BinomialLogit,
            FamilyNutsInputs::Glm(GlmFlatInputs {
                x: x.view(),
                y: y.view(),
                weights: w.view(),
                penalty_matrix: penalty.view(),
                mode: mode.view(),
                hessian: non_spdhessian.view(),
                gamma_shape: None,
                firth_bias_reduction: false,
            }),
            &cfg,
        )
        .expect("dispatch should use PG Gibbs and not require Hessian factorization");
        assert_eq!(out.samples.nrows(), cfg.n_samples * cfg.n_chains);
        assert!(out.samples.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn family_dispatch_routes_probit_to_nuts_path() {
        let x = array![[1.0, 0.2], [1.0, -0.1], [1.0, 1.2], [1.0, -0.7]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let penalty = array![[0.2, 0.0], [0.0, 0.4]];
        let mode = array![0.0, 0.0];
        let non_spdhessian = array![[0.0, 0.0], [0.0, 0.0]];
        let cfg = NutsConfig {
            n_samples: 20,
            nwarmup: 20,
            n_chains: 2,
            target_accept: 0.8,
            seed: 654,
        };

        let err = match run_nuts_sampling_flattened_family(
            LikelihoodFamily::BinomialProbit,
            FamilyNutsInputs::Glm(GlmFlatInputs {
                x: x.view(),
                y: y.view(),
                weights: w.view(),
                penalty_matrix: penalty.view(),
                mode: mode.view(),
                hessian: non_spdhessian.view(),
                gamma_shape: None,
                firth_bias_reduction: false,
            }),
            &cfg,
        ) {
            Ok(_) => panic!("non-SPD Hessian should fail after probit routes to the NUTS path"),
            Err(err) => err,
        };

        assert!(
            err.contains("Hessian Cholesky decomposition failed"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn family_dispatch_rejects_nonlogit_firth_family() {
        let x = array![[1.0, 0.2], [1.0, -0.1], [1.0, 1.2], [1.0, -0.7]];
        let y = array![1.0, 2.0, 0.0, 3.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let penalty = array![[0.2, 0.0], [0.0, 0.4]];
        let mode = array![0.0, 0.0];
        let hessian = array![[1.5, 0.1], [0.1, 1.2]];
        let cfg = NutsConfig {
            n_samples: 20,
            nwarmup: 20,
            n_chains: 2,
            target_accept: 0.8,
            seed: 111,
        };

        let err = match run_nuts_sampling_flattened_family(
            LikelihoodFamily::PoissonLog,
            FamilyNutsInputs::Glm(GlmFlatInputs {
                x: x.view(),
                y: y.view(),
                weights: w.view(),
                penalty_matrix: penalty.view(),
                mode: mode.view(),
                hessian: hessian.view(),
                gamma_shape: None,
                firth_bias_reduction: true,
            }),
            &cfg,
        ) {
            Ok(_) => panic!("Poisson Firth should be rejected explicitly"),
            Err(err) => err,
        };

        assert!(
            err.contains("NUTS with Firth is only supported for Binomial Logit"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn joint_hmc_boundary_rejects_nonlogit_firth_family() {
        let x = array![[1.0, 0.2], [1.0, -0.1], [1.0, 1.2], [1.0, -0.7]];
        let y = array![1.0, 2.0, 0.0, 3.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let hessian = array![[1.5, 0.1], [0.1, 1.2]];
        let penalty_root = array![[0.4, 0.0], [0.0, 0.6]];
        let mode = array![0.0, 0.0];
        let rho_mode = array![0.0];
        let cfg = NutsConfig {
            n_samples: 20,
            nwarmup: 20,
            n_chains: 2,
            target_accept: 0.8,
            seed: 111,
        };

        let inputs = JointBetaRhoInputs {
            x: x.view(),
            y: y.view(),
            weights: w.view(),
            likelihood_family: LikelihoodFamily::PoissonLog,
            inverse_link: InverseLink::Standard(LinkFunction::Log),
            gamma_shape: None,
            mode: mode.view(),
            hessian: hessian.view(),
            penalty_roots: vec![CanonicalPenalty::from_dense_root(
                penalty_root.clone(),
                penalty_root.ncols(),
            )],
            rho_mode: rho_mode.view(),
            rho_prior: RhoPrior::default(),
            firth_bias_reduction: true,
            trigger_skewness: 0.75,
        };

        let err = match run_joint_beta_rho_sampling(&inputs, &cfg) {
            Ok(_) => panic!("Poisson joint HMC Firth should be rejected explicitly"),
            Err(err) => err,
        };

        assert!(
            err.contains("Joint HMC with Firth is only supported for Binomial Logit"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn joint_hmc_uses_combined_penalty_logdet_for_overlapping_penalties() {
        let x = array![[0.0, 0.0]];
        let y = array![0.0];
        let w = array![0.0];
        let mode = array![0.0, 0.0];
        let hessian = array![[1.0, 0.0], [0.0, 1.0]];
        let rho_mode = array![0.0, 0.0];
        let penalty_1 = array![[1.0, 0.0], [0.0, 1.0]];
        let penalty_2 = array![[2.0_f64.sqrt(), 0.0], [0.0, 1.0]];
        let target = JointBetaRhoPosterior::new(
            x.view(),
            y.view(),
            w.view(),
            mode.view(),
            hessian.view(),
            vec![
                CanonicalPenalty::from_dense_root(penalty_1, 2),
                CanonicalPenalty::from_dense_root(penalty_2, 2),
            ],
            rho_mode.view(),
            LikelihoodFamily::GaussianIdentity,
            InverseLink::Standard(LinkFunction::Identity),
            None,
            RhoPrior::Flat,
            false,
        )
        .expect("joint target");

        let params = array![0.0, 0.0, 0.0, 0.0];
        let (_, grad) = target.compute_joint_logp_and_grad(&params);
        assert!(
            (grad[2] - 5.0 / 12.0).abs() < 1.0e-10,
            "expected overlapping-penalty gradient 5/12, got {}",
            grad[2]
        );
        assert!(
            (grad[3] - 7.0 / 12.0).abs() < 1.0e-10,
            "expected overlapping-penalty gradient 7/12, got {}",
            grad[3]
        );
    }

    #[test]
    fn joint_hmc_target_does_not_depend_on_rho_mode_when_prior_is_fixed() {
        let x = array![[0.0]];
        let y = array![0.0];
        let w = array![0.0];
        let mode = array![0.0];
        let hessian = array![[1.0]];
        let penalty = CanonicalPenalty::from_dense_root(array![[1.0]], 1);
        let prior = RhoPrior::Normal {
            mean: 0.25,
            sd: 1.7,
        };

        let target_a = JointBetaRhoPosterior::new(
            x.view(),
            y.view(),
            w.view(),
            mode.view(),
            hessian.view(),
            vec![penalty.clone()],
            array![0.0].view(),
            LikelihoodFamily::GaussianIdentity,
            InverseLink::Standard(LinkFunction::Identity),
            None,
            prior.clone(),
            false,
        )
        .expect("target a");
        let target_b = JointBetaRhoPosterior::new(
            x.view(),
            y.view(),
            w.view(),
            mode.view(),
            hessian.view(),
            vec![penalty],
            array![2.5].view(),
            LikelihoodFamily::GaussianIdentity,
            InverseLink::Standard(LinkFunction::Identity),
            None,
            prior,
            false,
        )
        .expect("target b");

        let params = array![0.0, -0.4];
        let (lp_a, grad_a) = target_a.compute_joint_logp_and_grad(&params);
        let (lp_b, grad_b) = target_b.compute_joint_logp_and_grad(&params);
        assert!((lp_a - lp_b).abs() < 1.0e-12);
        for i in 0..grad_a.len() {
            assert!(
                (grad_a[i] - grad_b[i]).abs() < 1.0e-12,
                "rho_mode leaked into target gradient at {}: {} vs {}",
                i,
                grad_a[i],
                grad_b[i]
            );
        }
    }

    #[test]
    fn joint_hmc_binomial_sas_uses_runtime_link_state() {
        let x = array![[1.0], [1.0]];
        let y = array![1.0, 0.0];
        let weights = array![1.0, 1.0];
        let eta = array![0.3, -0.2];
        let sas_state = crate::mixture_link::state_from_sasspec(crate::types::SasLinkSpec {
            initial_epsilon: 0.4,
            initial_log_delta: -0.2,
        })
        .expect("sas state");
        let data = SharedData {
            x: Arc::new(x),
            y: Arc::new(y),
            weights: Arc::new(weights),
            penalty: Arc::new(Array2::zeros((1, 1))),
            mode: Arc::new(Array1::zeros(1)),
            gamma_shape: 1.0,
            n_samples: 2,
            dim: 1,
        };

        let (ll_sas, _) = joint_family_logp_and_grad(
            LikelihoodFamily::BinomialSas,
            &InverseLink::Sas(sas_state),
            &data,
            &eta,
        )
        .expect("sas joint logp");
        let (ll_logit, _) = joint_family_logp_and_grad(
            LikelihoodFamily::BinomialLogit,
            &InverseLink::Standard(LinkFunction::Logit),
            &data,
            &eta,
        )
        .expect("logit joint logp");

        assert!(
            (ll_sas - ll_logit).abs() > 1.0e-6,
            "adaptive SAS link should not collapse to the logit likelihood"
        );
    }

    #[test]
    fn directional_cubic_diagnostic_is_rotation_invariant_for_hessian_eigenvectors() {
        let x = array![[1.0, 0.5], [-0.3, 1.4], [0.8, -1.1]];
        let c = array![0.7, -0.5, 0.2];
        let h = array![[4.0, 0.0], [0.0, 1.0]];
        let theta = std::f64::consts::FRAC_PI_4;
        let q = array![[theta.cos(), -theta.sin()], [theta.sin(), theta.cos()],];
        let x_rot = x.dot(&q);
        let h_rot = q.t().dot(&h).dot(&q);

        let (base_max, base_vals) = laplace_directional_cubic_diagnostic(
            &h,
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
            &c,
        )
        .expect("base diagnostic");
        let (rot_max, rot_vals) = laplace_directional_cubic_diagnostic(
            &h_rot,
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x_rot)),
            &c,
        )
        .expect("rotated diagnostic");

        let mut base_abs: Vec<f64> = base_vals.iter().map(|v| v.abs()).collect();
        let mut rot_abs: Vec<f64> = rot_vals.iter().map(|v| v.abs()).collect();
        base_abs.sort_by(|a, b| a.partial_cmp(b).expect("finite compare"));
        rot_abs.sort_by(|a, b| a.partial_cmp(b).expect("finite compare"));

        assert!((base_max - rot_max).abs() < 1.0e-10);
        for i in 0..base_abs.len() {
            assert!(
                (base_abs[i] - rot_abs[i]).abs() < 1.0e-10,
                "directional diagnostic changed under rotation at {}: {} vs {}",
                i,
                base_abs[i],
                rot_abs[i]
            );
        }
    }

    /// Verify that joint HMC and REML compute identical penalty logdet
    /// derivatives for the same penalty system. This catches any divergence
    /// between the two code paths.
    #[test]
    fn joint_hmc_penalty_logdet_agrees_with_reml_path() {
        use crate::estimate::reml::penalty_logdet::PenaltyPseudologdet;

        // Two overlapping 3x3 penalties with non-trivial lambdas.
        let root_1 = array![[1.0, 0.5, 0.0], [0.0, 0.8, 0.3]];
        let root_2 = array![[0.0, 0.7, 0.0], [0.0, 0.0, 1.2]];
        let cp1 = CanonicalPenalty::from_dense_root(root_1, 3);
        let cp2 = CanonicalPenalty::from_dense_root(root_2, 3);
        let lambdas = [2.5_f64, 0.8];
        let penalties = [cp1.clone(), cp2.clone()];

        // REML path: PenaltyPseudologdet directly.
        let pld =
            PenaltyPseudologdet::from_penalties(&penalties, &lambdas, 0.0, 3).expect("reml pld");
        let reml_value = pld.value();
        let (reml_d1, reml_d2) = pld.rho_derivatives_from_penalties(&penalties, &lambdas);

        // Joint HMC path: build a JointBetaRhoPosterior and extract the
        // penalty logdet contribution. We isolate it by using zero data
        // (so likelihood = 0, penalty quadratic = 0) and Flat rho prior.
        let x = Array2::<f64>::zeros((1, 3));
        let y = array![0.0];
        let w = array![0.0];
        let mode = Array1::<f64>::zeros(3);
        let hessian = Array2::<f64>::eye(3);
        let rho = Array1::from_vec(lambdas.iter().map(|l| l.ln()).collect());
        let target = JointBetaRhoPosterior::new(
            x.view(),
            y.view(),
            w.view(),
            mode.view(),
            hessian.view(),
            vec![cp1, cp2],
            rho.view(),
            LikelihoodFamily::GaussianIdentity,
            InverseLink::Standard(LinkFunction::Identity),
            None,
            RhoPrior::Flat,
            false,
        )
        .expect("joint target");

        // Evaluate at beta=0, rho=ln(lambdas).
        let mut params = Array1::<f64>::zeros(3 + 2);
        params[3] = rho[0];
        params[4] = rho[1];
        let (logp, grad) = target.compute_joint_logp_and_grad(&params);

        // logp should be 0.5 * reml_value (likelihood=0, prior=0, quadratic=0).
        assert!(
            (logp - 0.5 * reml_value).abs() < 1.0e-8,
            "joint HMC logdet value {} vs REML 0.5*{} = {}",
            logp,
            reml_value,
            0.5 * reml_value,
        );

        // grad[3..5] should be 0.5 * reml_d1.
        for k in 0..2 {
            assert!(
                (grad[3 + k] - 0.5 * reml_d1[k]).abs() < 1.0e-8,
                "joint HMC logdet gradient[{}] = {} vs REML 0.5*{} = {}",
                k,
                grad[3 + k],
                reml_d1[k],
                0.5 * reml_d1[k],
            );
        }

        // Sanity: second derivatives are available from REML but not directly
        // from a single HMC gradient call; just verify they're symmetric.
        assert!(
            (reml_d2[[0, 1]] - reml_d2[[1, 0]]).abs() < 1.0e-12,
            "REML penalty logdet Hessian not symmetric"
        );
    }

    /// Verify the family-gating invariant: every LikelihoodFamily that
    /// joint_family_logp_and_grad accepts produces a result (not an error
    /// about missing implementation). Every family it rejects returns an
    /// explicit error. No family is silently remapped to a different one.
    #[test]
    fn joint_hmc_family_gating_never_remaps() {
        let data = SharedData {
            x: Arc::new(array![[1.0], [1.0]]),
            y: Arc::new(array![1.0, 0.0]),
            weights: Arc::new(array![1.0, 1.0]),
            penalty: Arc::new(Array2::zeros((1, 1))),
            mode: Arc::new(Array1::zeros(1)),
            gamma_shape: 1.0,
            n_samples: 2,
            dim: 1,
        };
        let eta = array![0.1, -0.1];

        // These families must succeed with their own inverse link.
        let accepted = [
            (
                LikelihoodFamily::BinomialLogit,
                InverseLink::Standard(LinkFunction::Logit),
            ),
            (
                LikelihoodFamily::BinomialProbit,
                InverseLink::Standard(LinkFunction::Probit),
            ),
            (
                LikelihoodFamily::BinomialCLogLog,
                InverseLink::Standard(LinkFunction::CLogLog),
            ),
            (
                LikelihoodFamily::GaussianIdentity,
                InverseLink::Standard(LinkFunction::Identity),
            ),
            (
                LikelihoodFamily::PoissonLog,
                InverseLink::Standard(LinkFunction::Log),
            ),
            (
                LikelihoodFamily::GammaLog,
                InverseLink::Standard(LinkFunction::Log),
            ),
        ];
        for (family, link) in &accepted {
            let result = joint_family_logp_and_grad(*family, link, &data, &eta);
            assert!(
                result.is_ok(),
                "family {:?} should be accepted but got error: {:?}",
                family,
                result.err(),
            );
        }

        // SAS/BetaLogistic/Mixture must succeed with their real link state,
        // NOT be remapped to logit.
        let sas_state = crate::mixture_link::state_from_sasspec(crate::types::SasLinkSpec {
            initial_epsilon: 0.0,
            initial_log_delta: 0.0,
        })
        .expect("sas state");
        let adaptive = [
            (LikelihoodFamily::BinomialSas, InverseLink::Sas(sas_state)),
            (
                LikelihoodFamily::BinomialBetaLogistic,
                InverseLink::BetaLogistic(
                    crate::mixture_link::state_from_sasspec(crate::types::SasLinkSpec {
                        initial_epsilon: 0.0,
                        initial_log_delta: 0.0,
                    })
                    .expect("bl state"),
                ),
            ),
        ];
        for (family, link) in &adaptive {
            let result = joint_family_logp_and_grad(*family, link, &data, &eta);
            assert!(
                result.is_ok(),
                "adaptive family {:?} should be accepted with its real link",
                family,
            );
        }

        // RoystonParmar must be explicitly rejected (not silently remapped).
        let rp_result = joint_family_logp_and_grad(
            LikelihoodFamily::RoystonParmar,
            &InverseLink::Standard(LinkFunction::Logit),
            &data,
            &eta,
        );
        assert!(
            rp_result.is_err(),
            "RoystonParmar should be rejected, not silently accepted"
        );
    }

    /// The power-iteration refinement should find non-Gaussianity at least
    /// as large as the eigenvector-only pass (it's a supremum search).
    #[test]
    fn directional_cubic_power_iteration_finds_larger_or_equal_skewness() {
        // Construct a design where the maximum |gamma| occurs off-axis.
        // A single row with asymmetric structure makes the cubic form
        // peak between eigenvectors.
        let x = array![
            [2.0, 1.0],
            [-1.0, 2.0],
            [0.5, -0.5],
            [1.5, 0.3],
            [-0.8, 1.7],
        ];
        let c = array![1.0, -0.5, 0.3, -0.7, 0.4];
        let h = array![[3.0, 1.0], [1.0, 2.0]];

        let (max_val, eigenvector_vals) = laplace_directional_cubic_diagnostic(
            &h,
            &DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(x)),
            &c,
        )
        .expect("diagnostic");

        // max_val should be >= max of eigenvector-only values.
        let eig_max = eigenvector_vals
            .iter()
            .fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        assert!(
            max_val >= eig_max - 1.0e-12,
            "power iteration result {} should be >= eigenvector max {}",
            max_val,
            eig_max,
        );
    }

    #[test]
    fn logit_pg_rao_blackwell_returns_finite_terms() {
        let x = array![[1.0, 0.2], [1.0, -0.1], [1.0, 1.2], [1.0, -0.7]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let penalty = array![[0.2, 0.0], [0.0, 0.4]];
        let mode = array![0.0, 0.0];
        let roots = vec![array![[0.2_f64.sqrt(), 0.0], [0.0, 0.4_f64.sqrt()]]];
        let cfg = NutsConfig {
            n_samples: 30,
            nwarmup: 30,
            n_chains: 2,
            target_accept: 0.8,
            seed: 789,
        };

        let rb = super::estimate_logit_pg_rao_blackwell_terms(
            x.view(),
            y.view(),
            w.view(),
            penalty.view(),
            mode.view(),
            &roots,
            &cfg,
        )
        .expect("rao-blackwell PG should run");

        assert_eq!(rb.len(), 1);
        assert!(rb[0].is_finite());
        assert!(rb[0] >= 0.0);
    }

    #[test]
    fn logit_pg_rao_blackwell_matches_beta_quadratic_moment_sanity() {
        let x = array![[1.0, 0.2], [1.0, -0.1], [1.0, 1.2], [1.0, -0.7]];
        let y = array![1.0, 0.0, 1.0, 0.0];
        let w = array![1.0, 1.0, 1.0, 1.0];
        let penalty = array![[0.2, 0.0], [0.0, 0.4]];
        let mode = array![0.0, 0.0];
        let roots = vec![array![[0.2_f64.sqrt(), 0.0], [0.0, 0.4_f64.sqrt()]]];
        let cfg = NutsConfig {
            n_samples: 120,
            nwarmup: 80,
            n_chains: 2,
            target_accept: 0.8,
            seed: 901,
        };

        let gibbs = run_logit_polya_gamma_gibbs(
            x.view(),
            y.view(),
            w.view(),
            penalty.view(),
            mode.view(),
            &cfg,
        )
        .expect("pg gibbs should run");
        let mc_quad = gibbs
            .samples
            .rows()
            .into_iter()
            .map(|beta| {
                let sb = penalty.dot(&beta.to_owned());
                beta.dot(&sb)
            })
            .sum::<f64>()
            / (gibbs.samples.nrows() as f64);

        let rb = super::estimate_logit_pg_rao_blackwell_terms(
            x.view(),
            y.view(),
            w.view(),
            penalty.view(),
            mode.view(),
            &roots,
            &cfg,
        )
        .expect("rao-blackwell PG should run");

        let diff = (rb[0] - mc_quad).abs();
        assert!(
            diff < 0.35,
            "Rao-Blackwell vs beta-moment mismatch too large: rb={}, mc={}, diff={}",
            rb[0],
            mc_quad,
            diff
        );
    }

    #[test]
    fn survival_hmc_structural_monotonic_returns_finitevalues() {
        let age_entry = array![1.0];
        let age_exit = array![2.0];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[1.0, 0.2]];
        let x_exit = array![[1.0, 0.6]];
        let x_derivative = array![[0.0, 1.0]];
        let penalties = PenaltyBlocks::new(Vec::new());
        let monotonicity = MonotonicityPenalty { tolerance: 3.0 };
        let mode = array![0.0, 0.0];
        let hessian = Array2::<f64>::eye(2);

        let posterior = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            None,
            penalties,
            monotonicity,
            SurvivalSpec::Net,
            true,
            2,
            mode.view(),
            hessian.view(),
        )
        .expect("construct survival posterior");

        let position = array![0.0, 0.0];
        let mut grad = Array1::<f64>::zeros(2);
        let logp = HamiltonianTarget::logp_and_grad(&posterior, &position, &mut grad);
        assert!(logp.is_finite());
        assert!(grad.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn survival_hmc_structural_monotonic_differs_from_linear_geometry() {
        let age_entry = array![1.0];
        let age_exit = array![2.0];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[0.2, 0.1]];
        let x_exit = array![[0.6, 0.3]];
        let x_derivative = array![[1.0, 0.0]];
        let monotonicity = MonotonicityPenalty { tolerance: 3.0 };
        let mode = array![0.0, 0.0];
        let hessian = Array2::<f64>::eye(2);
        let z = array![std::f64::consts::LN_2, 0.0];

        let posterior_linear = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            None,
            PenaltyBlocks::new(Vec::new()),
            monotonicity,
            SurvivalSpec::Net,
            false,
            0,
            mode.view(),
            hessian.view(),
        )
        .expect("construct linear posterior");
        let mut grad_linear = Array1::<f64>::zeros(2);
        HamiltonianTarget::logp_and_grad(&posterior_linear, &z, &mut grad_linear);

        let posterior_struct = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            None,
            PenaltyBlocks::new(Vec::new()),
            monotonicity,
            SurvivalSpec::Net,
            true,
            2,
            mode.view(),
            hessian.view(),
        )
        .expect("construct structural posterior");
        let mut grad_struct = Array1::<f64>::zeros(2);
        HamiltonianTarget::logp_and_grad(&posterior_struct, &z, &mut grad_struct);

        assert!(
            (grad_struct[0] - grad_linear[0]).abs() > 1e-6,
            "expected structural and linear fallback gradients to differ"
        );
        assert!(grad_struct[0].is_finite());
        assert!(grad_linear[0].is_finite());
    }

    #[test]
    fn survival_hmc_fallback_barrier_rejects_offsets_below_monotonicity_threshold() {
        let age_entry = array![1.0];
        let age_exit = array![2.0];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[1.0, 0.0]];
        let x_exit = array![[1.0, 0.0]];
        // Zero derivative design so derivative_offset_exit drives d_eta/dt.
        let x_derivative = array![[0.0, 0.0]];
        let penalties = PenaltyBlocks::new(Vec::new());
        let monotonicity = MonotonicityPenalty { tolerance: 3.0 };
        let mode = array![0.0, 0.0];
        let hessian = Array2::<f64>::eye(2);
        let z = array![0.0, 0.0];

        let posterior_no_offset = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            Some(array![0.0].view()),
            penalties.clone(),
            monotonicity.clone(),
            SurvivalSpec::Net,
            false,
            0,
            mode.view(),
            hessian.view(),
        )
        .expect("construct posterior without derivative offset");
        let mut grad_no_offset = Array1::<f64>::zeros(2);
        let logp_no_offset =
            HamiltonianTarget::logp_and_grad(&posterior_no_offset, &z, &mut grad_no_offset);

        let posteriorwith_offset = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            Some(array![2.0].view()),
            penalties,
            monotonicity,
            SurvivalSpec::Net,
            false,
            0,
            mode.view(),
            hessian.view(),
        )
        .expect("construct posterior with derivative offset");
        let mut gradwith_offset = Array1::<f64>::zeros(2);
        let logpwith_offset =
            HamiltonianTarget::logp_and_grad(&posteriorwith_offset, &z, &mut gradwith_offset);

        assert!(!logp_no_offset.is_finite());
        assert!(!logpwith_offset.is_finite());
        assert!(grad_no_offset.iter().all(|v| *v == 0.0));
        assert!(gradwith_offset.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn survival_hmc_fallback_barrier_becomes_finite_once_offset_clears_guard() {
        let age_entry = array![1.0];
        let age_exit = array![2.0];
        let event_target = array![1u8];
        let event_competing = array![0u8];
        let sampleweight = array![1.0];
        let x_entry = array![[1.0, 0.0]];
        let x_exit = array![[1.0, 0.0]];
        let x_derivative = array![[0.0, 0.0]];
        let penalties = PenaltyBlocks::new(Vec::new());
        let monotonicity = MonotonicityPenalty { tolerance: 3.0 };
        let mode = array![0.0, 0.0];
        let hessian = Array2::<f64>::eye(2);
        let z = array![0.0, 0.0];

        let posterior_below_guard = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            Some(array![2.0].view()),
            penalties.clone(),
            monotonicity.clone(),
            SurvivalSpec::Net,
            false,
            0,
            mode.view(),
            hessian.view(),
        )
        .expect("construct posterior below derivative guard");
        let mut grad_below_guard = Array1::<f64>::zeros(2);
        let logp_below_guard =
            HamiltonianTarget::logp_and_grad(&posterior_below_guard, &z, &mut grad_below_guard);

        let posterior_above_guard = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            Some(array![3.1].view()),
            penalties,
            monotonicity,
            SurvivalSpec::Net,
            false,
            0,
            mode.view(),
            hessian.view(),
        )
        .expect("construct posterior above derivative guard");
        let mut grad_above_guard = Array1::<f64>::zeros(2);
        let logp_above_guard =
            HamiltonianTarget::logp_and_grad(&posterior_above_guard, &z, &mut grad_above_guard);

        assert!(!logp_below_guard.is_finite());
        assert!(logp_above_guard.is_finite());
        assert!(grad_below_guard.iter().all(|v| *v == 0.0));
        assert!(grad_above_guard.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn survival_hmc_structural_monotonic_handles_sparse_multirow_geometry() {
        let age_entry = array![1.0, 1.2];
        let age_exit = array![2.0, 2.4];
        let event_target = array![1u8, 1u8];
        let event_competing = array![0u8, 0u8];
        let sampleweight = array![1.0, 1.0];
        let x_entry = array![[0.1, 0.0, 0.2], [0.2, 0.1, 0.2]];
        let x_exit = array![[0.4, 0.2, 0.3], [0.6, 0.1, 0.3]];
        // First row constrains only column 0, second row constrains columns 0 and 1.
        let x_derivative = array![[1.0, 0.0, 0.0], [0.5, 1.0, 0.0]];
        let monotonicity = MonotonicityPenalty { tolerance: 3.0 };
        let mode = array![4.0, 2.0, 0.0];
        let hessian = Array2::<f64>::eye(3);
        let z = array![0.05, -0.1, 0.15];

        let posterior = super::survival_hmc::SurvivalPosterior::new(
            age_entry.view(),
            age_exit.view(),
            event_target.view(),
            event_competing.view(),
            sampleweight.view(),
            x_entry.view(),
            x_exit.view(),
            x_derivative.view(),
            None,
            None,
            None,
            PenaltyBlocks::new(Vec::new()),
            monotonicity,
            SurvivalSpec::Net,
            true,
            2,
            mode.view(),
            hessian.view(),
        )
        .expect("construct structural posterior");

        let mut grad = Array1::<f64>::zeros(3);
        let logp = HamiltonianTarget::logp_and_grad(&posterior, &z, &mut grad);
        assert!(logp.is_finite());
        assert!(grad.iter().all(|v| v.is_finite()));
    }
}

/// Implement HamiltonianTarget for NUTS with analytical gradients.
impl HamiltonianTarget<Array1<f64>> for NutsPosterior {
    fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
        let (logp, gradz) = self.compute_logp_and_grad_nd(position);
        grad.assign(&gradz);
        logp
    }
}

/// Configuration for NUTS sampling.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NutsConfig {
    /// Number of samples to collect (after warmup)
    pub n_samples: usize,
    /// Number of warmup samples to discard
    pub nwarmup: usize,
    /// Number of parallel chains
    pub n_chains: usize,
    /// Target acceptance probability (0.6-0.9 recommended)
    pub target_accept: f64,
    /// Seed for deterministic chain initialization
    #[serde(default = "default_nuts_seed")]
    pub seed: u64,
}

fn default_nuts_seed() -> u64 {
    42
}

fn robust_mass_matrix_config(dim: usize, nwarmup: usize) -> NUTSMassMatrixConfig {
    if nwarmup < 80 {
        return NUTSMassMatrixConfig::disabled();
    }
    let start_buffer = (nwarmup / 10).clamp(25, 150);
    let end_buffer = (nwarmup / 8).clamp(25, 150);
    let initial_window = (nwarmup / 12).clamp(20, 120);
    let dense_allowed = dim <= 50;
    NUTSMassMatrixConfig {
        adaptation: if dense_allowed {
            MassMatrixAdaptation::Dense
        } else {
            MassMatrixAdaptation::Diagonal
        },
        start_buffer,
        end_buffer,
        initial_window,
        regularize: if dense_allowed { 0.03 } else { 0.08 },
        jitter: 1e-6,
        dense_max_dim: 75,
    }
}

fn robust_survival_mass_matrix_config(dim: usize, nwarmup: usize) -> NUTSMassMatrixConfig {
    if nwarmup < 80 {
        return NUTSMassMatrixConfig::disabled();
    }
    // Survival posteriors with censoring/rare events are often skewed; this
    // configuration uses diagonal adaptation.
    let start_buffer = (nwarmup / 8).clamp(30, 180);
    let end_buffer = (nwarmup / 6).clamp(30, 180);
    let initial_window = (nwarmup / 10).clamp(25, 140);
    NUTSMassMatrixConfig {
        adaptation: MassMatrixAdaptation::Diagonal,
        start_buffer,
        end_buffer,
        initial_window,
        regularize: if dim > 50 { 0.12 } else { 0.08 },
        jitter: 1e-6,
        dense_max_dim: 75,
    }
}

impl Default for NutsConfig {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            nwarmup: 500,
            n_chains: 4,
            target_accept: 0.8,
            seed: 42,
        }
    }
}

impl NutsConfig {
    /// Create a config with sample counts tuned for the model dimension.
    ///
    /// Higher dimensions need more samples because:
    /// - ESS decreases with dimension (autocorrelation grows)
    /// - Split R-hat needs enough samples per chain to be meaningful
    ///
    /// Rule of thumb: target 100 effective samples per parameter.
    pub fn for_dimension(n_params: usize) -> Self {
        // ESS ≈ n_samples / (1 + 2τ) where τ ≈ sqrt(dim) for well-tuned NUTS
        let effective_autocorr = (n_params as f64).sqrt().max(1.0);

        // Target: at least 100 effective samples per parameter
        let target_ess = 100 * n_params;

        // Samples needed = ESS * (1 + 2τ), with 1.5x safety factor
        let raw_samples = (target_ess as f64 * (1.0 + 2.0 * effective_autocorr) * 1.5) as usize;

        // Clamp to reasonable range [500, 10000]
        let n_samples = raw_samples.clamp(500, 10_000);

        // Warmup ≈ samples (standard practice for adaptation)
        let nwarmup = n_samples;

        // More chains for higher dims (better R-hat estimation)
        let n_chains = if n_params > 50 { 4 } else { 2 };

        Self {
            n_samples,
            nwarmup,
            n_chains,
            target_accept: 0.8,
            seed: 42,
        }
    }
}

/// Result of NUTS sampling.
#[derive(Clone, Debug)]
pub struct NutsResult {
    /// Coefficient samples in ORIGINAL space: shape (n_total_samples, n_coeffs)
    pub samples: Array2<f64>,
    /// Posterior mean
    pub posterior_mean: Array1<f64>,
    /// Posterior standard deviation
    pub posterior_std: Array1<f64>,
    /// R-hat convergence diagnostic
    pub rhat: f64,
    /// Effective sample size
    pub ess: f64,
    /// Whether sampling converged (R-hat < 1.1)
    pub converged: bool,
}

impl NutsResult {
    /// Computes the posterior mean of a function applied to coefficients.
    /// Returns 0.0 if samples is empty to avoid divide-by-zero.
    pub fn posterior_mean_of<F>(&self, f: F) -> f64
    where
        F: Fn(ArrayView1<f64>) -> f64,
    {
        let n = self.samples.nrows();
        if n == 0 {
            return 0.0;
        }
        let mut sum = 0.0;
        for i in 0..n {
            sum += f(self.samples.row(i));
        }
        sum / n as f64
    }

    /// Computes percentiles of a function applied to coefficients.
    pub fn posterior_interval_of<F>(&self, f: F, lower_pct: f64, upper_pct: f64) -> (f64, f64)
    where
        F: Fn(ArrayView1<f64>) -> f64,
    {
        let n = self.samples.nrows();
        if n == 0 {
            return (0.0, 0.0);
        }
        let mut values: Vec<f64> = (0..n).map(|i| f(self.samples.row(i))).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let lower_idx = ((lower_pct / 100.0) * n as f64).floor() as usize;
        let upper_idx = ((upper_pct / 100.0) * n as f64).ceil() as usize;

        (
            values[lower_idx.min(n.saturating_sub(1))],
            values[upper_idx.min(n.saturating_sub(1))],
        )
    }
}

#[inline]
fn sample_standard_normal<R: rand::Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1 = rng.random::<f64>().max(1e-16);
    let u2 = rng.random::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[inline]
fn forward_solve_lower_triangular(l: &Array2<f64>, rhs: &Array1<f64>, out: &mut Array1<f64>) {
    let p = rhs.len();
    debug_assert_eq!(l.nrows(), p);
    debug_assert_eq!(l.ncols(), p);
    debug_assert_eq!(out.len(), p);
    for i in 0..p {
        let mut v = rhs[i];
        for j in 0..i {
            v -= l[[i, j]] * out[j];
        }
        let d = l[[i, i]];
        out[i] = if d.abs() > 1e-14 { v / d } else { 0.0 };
    }
}

/// Runs a Pólya-Gamma Gibbs sampler for Bernoulli-logit models.
///
/// This sampler is gradient-free: each iteration alternates
/// 1) ω_i | β, y ~ PG(1, x_i^T β), and
/// 2) β | ω, y ~ N(Q^{-1} b, Q^{-1}), with Q = S + X^T diag(ω) X, b = X^T(y - 1/2).
///
/// For weighted data, this implementation is defined for weights ≈ 1.0 because it
/// samples PG(1,·) latent variables.
pub fn run_logit_polya_gamma_gibbs(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    penalty_matrix: ArrayView2<f64>,
    mode: ArrayView1<f64>,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    let n = x.nrows();
    let p = x.ncols();
    if y.len() != n || weights.len() != n {
        return Err("run_logit_polya_gamma_gibbs: input length mismatch".to_string());
    }
    if mode.len() != p || penalty_matrix.nrows() != p || penalty_matrix.ncols() != p {
        return Err(
            "run_logit_polya_gamma_gibbs: coefficient/penalty dimension mismatch".to_string(),
        );
    }
    if !weights.iter().all(|w| (*w - 1.0).abs() <= 1e-10) {
        return Err(
            "run_logit_polya_gamma_gibbs requires unit weights (PG(1,·)); use NUTS for non-unit weights".to_string(),
        );
    }

    let n_iter = config.nwarmup + config.n_samples;
    let mut rng = StdRng::seed_from_u64(config.seed);

    // b = X^T (y - 1/2), constant across iterations.
    let mut kappa = Array1::<f64>::zeros(n);
    for i in 0..n {
        kappa[i] = y[i] - 0.5;
    }
    let rhs_b = fast_atv(&x, &kappa);

    let mut samples_array = Array3::<f64>::zeros((config.n_chains, config.n_samples, p));
    let mut eta = Array1::<f64>::zeros(n);
    let mut omega = Array1::<f64>::ones(n);
    let mut xw = x.to_owned();
    let mut xt_omega_x = Array2::<f64>::zeros((p, p));
    let penalty = penalty_matrix.to_owned();
    let mut q = Array2::<f64>::zeros((p, p));
    let mut mean = Array1::<f64>::zeros(p);
    let mut z = Array1::<f64>::zeros(p);
    let mut noise = Array1::<f64>::zeros(p);
    let mut progress = SamplingVisualizer::new(
        "polya-gamma gibbs",
        config.n_chains,
        config.nwarmup,
        config.n_samples,
    );

    for chain in 0..config.n_chains {
        progress.begin_chain(chain, "polya-gamma gibbs");
        let mut pg_rng = StdRng::seed_from_u64(
            config.seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul((chain as u64) + 1)),
        );
        let pg = PolyaGamma::new();
        let mut beta = mode.to_owned();
        // Small jitter so chains are not perfectly coupled.
        for j in 0..p {
            beta[j] += 0.05 * sample_standard_normal(&mut rng);
        }

        for iter in 0..n_iter {
            eta.assign(&x.dot(&beta));
            for i in 0..n {
                omega[i] = pg.draw(&mut pg_rng, eta[i]).max(1e-12);
            }

            // Build Xweighted = diag(sqrt(ω)) X and compute X^T Ω X via faer GEMM.
            for i in 0..n {
                let s = omega[i].sqrt();
                for j in 0..p {
                    xw[[i, j]] = x[[i, j]] * s;
                }
            }
            fast_ata_into(&xw, &mut xt_omega_x);

            q.assign(&penalty);
            q += &xt_omega_x;

            // β | ω,y ~ N(Q^{-1} b, Q^{-1})
            let factor = q
                .cholesky(Side::Lower)
                .map_err(|e| format!("PG Gibbs failed to factor Q: {:?}", e))?;
            mean.assign(&factor.solvevec(&rhs_b));

            for j in 0..p {
                z[j] = sample_standard_normal(&mut rng);
            }
            let l = factor.lower_triangular();
            forward_solve_lower_triangular(&l, &z, &mut noise);
            beta.assign(&(&mean + &noise));

            if iter < config.nwarmup {
                progress.warmup_step(iter + 1);
            } else {
                if iter == config.nwarmup {
                    progress.start_sampling();
                }
                let keep_idx = iter - config.nwarmup;
                progress.sample_step(keep_idx + 1);
                samples_array
                    .slice_mut(ndarray::s![chain, keep_idx, ..])
                    .assign(&beta);
            }
        }
        progress.finish_chain(1.0);
    }

    let total_samples = config.n_chains * config.n_samples;
    let mut samples = Array2::<f64>::zeros((total_samples, p));
    for chain in 0..config.n_chains {
        for s in 0..config.n_samples {
            let idx = chain * config.n_samples + s;
            samples
                .row_mut(idx)
                .assign(&samples_array.slice(ndarray::s![chain, s, ..]));
        }
    }

    let posterior_mean = samples
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(p));
    let posterior_std = samples.std_axis(Axis(0), 0.0);
    let (rhat, ess) = if config.n_chains >= 2 && config.n_samples >= 4 {
        compute_split_rhat_and_ess(&samples_array)
    } else {
        (1.0, (total_samples as f64) * 0.5)
    };
    let converged = rhat < 1.1 && ess > 100.0;
    progress.finish_all(rhat, ess);

    Ok(NutsResult {
        samples,
        posterior_mean,
        posterior_std,
        rhat,
        ess,
        converged,
    })
}

/// Estimate E_{ω|y,ρ}[ tr(S_k Q^{-1}) + μᵀ S_k μ ] with PG Gibbs + Rao-Blackwellization.
///
/// For each retained Gibbs state ω:
///   Q = S + Xᵀ diag(ω) X,  μ = Q^{-1} Xᵀ(y-1/2),
/// and with S_k = R_kᵀ R_k:
///   tr(S_k Q^{-1}) + μᵀ S_k μ
/// = tr(R_k Q^{-1} R_kᵀ) + ||R_k μ||².
///
/// Returns one expectation per penalty block k, averaged over retained draws.
pub fn estimate_logit_pg_rao_blackwell_terms(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    penalty_matrix: ArrayView2<f64>,
    mode: ArrayView1<f64>,
    penalty_roots: &[Array2<f64>],
    config: &NutsConfig,
) -> Result<Array1<f64>, String> {
    let n = x.nrows();
    let p = x.ncols();
    if y.len() != n || weights.len() != n {
        return Err("estimate_logit_pg_rao_blackwell_terms: input length mismatch".to_string());
    }
    if mode.len() != p || penalty_matrix.nrows() != p || penalty_matrix.ncols() != p {
        return Err(
            "estimate_logit_pg_rao_blackwell_terms: coefficient/penalty dimension mismatch"
                .to_string(),
        );
    }
    if !weights.iter().all(|w| (*w - 1.0).abs() <= 1e-10) {
        return Err(
            "estimate_logit_pg_rao_blackwell_terms requires unit weights (PG(1,·))".to_string(),
        );
    }
    if penalty_roots.iter().any(|r| r.ncols() != p) {
        return Err("estimate_logit_pg_rao_blackwell_terms: root width mismatch".to_string());
    }
    // Precompute transposed root blocks once:
    //   R_k^T is the RHS used for batched solves Q X = R_k^T.
    let penalty_roots_t: Vec<Array2<f64>> =
        penalty_roots.iter().map(|r| r.t().to_owned()).collect();

    let mut rng = StdRng::seed_from_u64(config.seed);
    let n_iter = config.nwarmup + config.n_samples;

    // Logistic PG identity uses kappa_i = y_i - 1/2 so that
    // b = X^T kappa in the Gaussian conditional for beta|omega.
    let mut kappa = Array1::<f64>::zeros(n);
    for i in 0..n {
        kappa[i] = y[i] - 0.5;
    }
    let rhs_b = fast_atv(&x, &kappa);

    let penalty = penalty_matrix.to_owned();
    let mut eta = Array1::<f64>::zeros(n);
    let mut omega = Array1::<f64>::ones(n);
    let mut xw = x.to_owned();
    let mut xt_omega_x = Array2::<f64>::zeros((p, p));
    let mut q = Array2::<f64>::zeros((p, p));
    let mut mean = Array1::<f64>::zeros(p);
    let mut rb_sum = Array1::<f64>::zeros(penalty_roots.len());
    let mut z = Array1::<f64>::zeros(p);
    let mut noise = Array1::<f64>::zeros(p);

    let mut kept = 0usize;
    for chain in 0..config.n_chains {
        let mut pg_rng = StdRng::seed_from_u64(
            config.seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul((chain as u64) + 1)),
        );
        let pg = PolyaGamma::new();
        let mut beta = mode.to_owned();
        for j in 0..p {
            beta[j] += 0.05 * sample_standard_normal(&mut rng);
        }

        for iter in 0..n_iter {
            eta.assign(&x.dot(&beta));
            for i in 0..n {
                omega[i] = pg.draw(&mut pg_rng, eta[i]).max(1e-12);
            }

            for i in 0..n {
                let s = omega[i].sqrt();
                for j in 0..p {
                    xw[[i, j]] = x[[i, j]] * s;
                }
            }
            fast_ata_into(&xw, &mut xt_omega_x);

            // Conditional precision:
            //   Q = S + X^T diag(omega) X.
            q.assign(&penalty);
            q += &xt_omega_x;

            let factor = q
                .cholesky(Side::Lower)
                .map_err(|e| format!("PG Rao-Blackwell failed to factor Q: {:?}", e))?;
            // Conditional mean:
            //   mu = Q^{-1} b,  b = X^T(y - 1/2).
            mean.assign(&factor.solvevec(&rhs_b));

            // Draw beta for the next Gibbs state.
            for j in 0..p {
                z[j] = sample_standard_normal(&mut rng);
            }
            let l = factor.lower_triangular();
            forward_solve_lower_triangular(&l, &z, &mut noise);
            beta.assign(&(&mean + &noise));

            if iter < config.nwarmup {
                continue;
            }
            kept += 1;

            for (k, r_k) in penalty_roots.iter().enumerate() {
                if r_k.nrows() == 0 {
                    continue;
                }

                // mu^T S_k mu via root form S_k = R_k^T R_k.
                let rmu = r_k.dot(&mean);
                let mu_quad = rmu.dot(&rmu);

                // Batched trace solve:
                //   V_k = Q^{-1} R_k^T  (single multi-RHS solve)
                // then tr(R_k Q^{-1} R_k^T) = <R_k, V_k^T>_F.
                let solved_mat = factor.solve_mat(&penalty_roots_t[k]); // (p, r_k)
                let solved_t = solved_mat.t();
                let mut trace_term = 0.0_f64;
                for (&a, &b) in r_k.iter().zip(solved_t.iter()) {
                    trace_term += a * b;
                }

                rb_sum[k] += trace_term + mu_quad;
            }
        }
    }

    if kept == 0 {
        return Err("estimate_logit_pg_rao_blackwell_terms: no retained samples".to_string());
    }
    let out = rb_sum.mapv(|v| v / (kept as f64));
    if !out.iter().all(|v| v.is_finite()) {
        return Err("estimate_logit_pg_rao_blackwell_terms: non-finite expectation".to_string());
    }
    Ok(out)
}

/// Runs NUTS sampling using general-mcmc with whitened parameter space.
///
/// # Arguments
/// * `x` - Design matrix [n_samples, dim]
/// * `y` - Response vector [n_samples]
/// * `weights` - Observation/case weights [n_samples]
/// * `penalty_matrix` - Combined penalty S [dim, dim]
/// * `mode` - MAP estimate μ [dim]
/// * `hessian` - Penalized Hessian H [dim, dim] (NOT the inverse!)
/// * `nuts_family` - Family for log-likelihood computation
/// * `firth_bias_reduction` - Whether Firth bias reduction was used in training
/// * `config` - NUTS configuration
pub(crate) fn run_nuts_sampling(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    penalty_matrix: ArrayView2<f64>,
    mode: ArrayView1<f64>,
    hessian: ArrayView2<f64>,
    nuts_family: NutsFamily,
    gamma_shape: f64,
    firth_bias_reduction: bool,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    validate_firth_support(nuts_family, firth_bias_reduction)?;
    let dim = mode.len();

    // Create posterior target with analytical gradients. When Firth is enabled,
    // this target includes the identifiable-subspace Jeffreys term.
    let target = NutsPosterior::new(
        x,
        y,
        weights,
        penalty_matrix,
        mode,
        hessian,
        nuts_family,
        gamma_shape,
        firth_bias_reduction,
    )?;

    // Get Cholesky factor for un-whitening samples later
    let chol = target.chol().clone();
    let mode_arr = target.mode().clone();

    // Initialize chains at z=0 with small jitter
    let mut rng = StdRng::seed_from_u64(config.seed);
    let initial_positions: Vec<Array1<f64>> = (0..config.n_chains)
        .map(|_| {
            Array1::from_shape_fn(dim, |_| {
                let u1: f64 = rng.random::<f64>().max(1e-10); // Prevent ln(0) = -inf
                let u2: f64 = rng.random();
                let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                z * 0.1
            })
        })
        .collect();

    // Create GenericNUTS sampler - it auto-tunes step size!
    let mass_cfg = robust_mass_matrix_config(dim, config.nwarmup);
    let mut sampler = GenericNUTS::new_with_mass_matrix(
        target,
        initial_positions,
        config.target_accept,
        mass_cfg,
    );

    let (samples_array, run_stats) = sampler
        .run_progress(config.n_samples, config.nwarmup)
        .map_err(|e| format!("NUTS sampling failed: {e}"))?;
    log::info!("NUTS sampling complete: {}", run_stats);

    // Convert samples from whitened space back to original space
    // samples_array has shape [n_chains, n_samples, dim]
    let shape = samples_array.shape();
    let n_chains = shape[0];
    let n_samples_out = shape[1];
    let total_samples = n_chains * n_samples_out;

    let mut samples = Array2::<f64>::zeros((total_samples, dim));
    let mut z_buffer = Array1::<f64>::zeros(dim);
    for chain in 0..n_chains {
        for sample_i in 0..n_samples_out {
            let zview = samples_array.slice(ndarray::s![chain, sample_i, ..]);
            z_buffer.assign(&zview);
            let beta = &mode_arr + &chol.dot(&z_buffer);
            let sample_idx = chain * n_samples_out + sample_i;
            samples.row_mut(sample_idx).assign(&beta);
        }
    }

    // Compute split-chain R-hat and ESS for proper convergence diagnostics
    let posterior_mean = samples
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(dim));
    let posterior_std = samples.std_axis(Axis(0), 0.0);

    // Split-chain R-hat: compare variance within vs between chains
    // Gelman-Rubin diagnostic with split chains
    let (rhat, ess) = compute_split_rhat_and_ess(&samples_array);

    let converged = rhat < 1.1 && ess > 100.0;

    Ok(NutsResult {
        samples,
        posterior_mean,
        posterior_std,
        rhat,
        ess,
        converged,
    })
}

/// Flattened numeric inputs for GLM-family NUTS sampling.
pub struct GlmFlatInputs<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub weights: ArrayView1<'a, f64>,
    pub penalty_matrix: ArrayView2<'a, f64>,
    pub mode: ArrayView1<'a, f64>,
    pub hessian: ArrayView2<'a, f64>,
    pub gamma_shape: Option<f64>,
    pub firth_bias_reduction: bool,
}

/// Flat survival inputs for engine-facing HMC APIs.
pub struct SurvivalFlatInputs<'a> {
    pub age_entry: ArrayView1<'a, f64>,
    pub age_exit: ArrayView1<'a, f64>,
    pub event_target: ArrayView1<'a, u8>,
    pub event_competing: ArrayView1<'a, u8>,
    pub weights: ArrayView1<'a, f64>,
    pub x_entry: ArrayView2<'a, f64>,
    pub x_exit: ArrayView2<'a, f64>,
    pub x_derivative: ArrayView2<'a, f64>,
    pub eta_offset_entry: Option<ArrayView1<'a, f64>>,
    pub eta_offset_exit: Option<ArrayView1<'a, f64>>,
    pub derivative_offset_exit: Option<ArrayView1<'a, f64>>,
}

/// Flattened numeric inputs for Royston-Parmar NUTS sampling.
pub struct SurvivalNutsInputs<'a> {
    pub flat: SurvivalFlatInputs<'a>,
    pub penalties: crate::survival::PenaltyBlocks,
    pub monotonicity: crate::survival::MonotonicityPenalty,
    pub spec: crate::survival::SurvivalSpec,
    pub structurally_monotonic: bool,
    pub structural_time_columns: usize,
    pub mode: ArrayView1<'a, f64>,
    pub hessian: ArrayView2<'a, f64>,
}

/// Family-dispatched flattened NUTS inputs.
pub enum FamilyNutsInputs<'a> {
    Glm(GlmFlatInputs<'a>),
    Survival(Box<SurvivalNutsInputs<'a>>),
}

/// Family-agnostic flattened NUTS entrypoint across all supported likelihood families.
pub fn run_nuts_sampling_flattened_family(
    family: LikelihoodFamily,
    inputs: FamilyNutsInputs<'_>,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    if let FamilyNutsInputs::Glm(glm) = &inputs {
        if glm.firth_bias_reduction && !family.supports_firth() {
            return Err(format!(
                "NUTS with Firth is only supported for {}; {} does not support it",
                LikelihoodFamily::BinomialLogit.pretty_name(),
                family.pretty_name()
            ));
        }
    }

    match (family, inputs) {
        (LikelihoodFamily::GaussianIdentity, FamilyNutsInputs::Glm(glm)) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            NutsFamily::Gaussian,
            1.0,
            glm.firth_bias_reduction,
            config,
        ),
        (LikelihoodFamily::BinomialLogit, FamilyNutsInputs::Glm(glm)) => {
            // Auto-select PG Gibbs when assumptions hold; otherwise fall back to NUTS.
            // This gives gradient-free posterior draws for standard Bernoulli logit GAMs.
            if !glm.firth_bias_reduction && glm.weights.iter().all(|w| (*w - 1.0).abs() <= 1e-10) {
                run_logit_polya_gamma_gibbs(
                    glm.x,
                    glm.y,
                    glm.weights,
                    glm.penalty_matrix,
                    glm.mode,
                    config,
                )
            } else {
                run_nuts_sampling(
                    glm.x,
                    glm.y,
                    glm.weights,
                    glm.penalty_matrix,
                    glm.mode,
                    glm.hessian,
                    NutsFamily::BinomialLogit,
                    1.0,
                    glm.firth_bias_reduction,
                    config,
                )
            }
        }
        (LikelihoodFamily::BinomialProbit, FamilyNutsInputs::Glm(glm)) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            NutsFamily::BinomialProbit,
            1.0,
            glm.firth_bias_reduction,
            config,
        ),
        (LikelihoodFamily::BinomialCLogLog, FamilyNutsInputs::Glm(glm)) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            NutsFamily::BinomialCLogLog,
            1.0,
            glm.firth_bias_reduction,
            config,
        ),
        (LikelihoodFamily::BinomialLatentCLogLog, FamilyNutsInputs::Glm(glm)) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            NutsFamily::BinomialCLogLog,
            1.0,
            glm.firth_bias_reduction,
            config,
        ),
        (LikelihoodFamily::BinomialMixture, FamilyNutsInputs::Glm(_)) => Err(
            "BinomialMixture NUTS is not implemented yet; use fit_gam/predict_gam for blended inverse-link models"
                .to_string(),
        ),
        (LikelihoodFamily::BinomialSas, FamilyNutsInputs::Glm(_)) => Err(
            "BinomialSas NUTS is not implemented yet; use fit_gam/predict_gam for SAS-link models"
                .to_string(),
        ),
        (LikelihoodFamily::BinomialBetaLogistic, FamilyNutsInputs::Glm(_)) => Err(
            "BinomialBetaLogistic NUTS is not implemented yet; use fit_gam/predict_gam for beta-logistic-link models"
                .to_string(),
        ),
        (LikelihoodFamily::RoystonParmar, FamilyNutsInputs::Survival(survival)) => {
            survival_hmc::run_survival_nuts_sampling(
                survival.flat.age_entry,
                survival.flat.age_exit,
                survival.flat.event_target,
                survival.flat.event_competing,
                survival.flat.weights,
                survival.flat.x_entry,
                survival.flat.x_exit,
                survival.flat.x_derivative,
                survival.flat.eta_offset_entry,
                survival.flat.eta_offset_exit,
                survival.flat.derivative_offset_exit,
                survival.penalties,
                survival.monotonicity,
                survival.spec,
                survival.structurally_monotonic,
                survival.structural_time_columns,
                survival.mode,
                survival.hessian,
                config,
            )
        }
        (LikelihoodFamily::RoystonParmar, FamilyNutsInputs::Glm(_)) => Err(
            "RoystonParmar family requires FamilyNutsInputs::Survival flattened inputs".to_string(),
        ),
        (_, FamilyNutsInputs::Survival(_)) => Err(
            "Survival flattened inputs are only valid for LikelihoodFamily::RoystonParmar"
                .to_string(),
        ),
        (LikelihoodFamily::PoissonLog, FamilyNutsInputs::Glm(glm)) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            NutsFamily::PoissonLog,
            1.0,
            glm.firth_bias_reduction,
            config,
        ),
        (LikelihoodFamily::GammaLog, FamilyNutsInputs::Glm(glm)) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            NutsFamily::GammaLog,
            glm.gamma_shape.unwrap_or(1.0),
            glm.firth_bias_reduction,
            config,
        ),
    }
}

// ============================================================================
// Joint (β, θ) Link-Wiggle HMC
// ============================================================================
//
// NUTS sampling over the joint parameter space [β_eta; β_wiggle] for models
// with a structurally monotone I-spline link wiggle. The wiggle introduces a
// nonlinear coupling:
//
//   η(β_eta, β_wiggle) = q₀(β_eta) + B(q₀(β_eta)) · β_wiggle
//
// where B is the shared monotone wiggle basis evaluated at the base linear
// predictor q₀ = X · β_eta. The gradient of log p(y|β_eta, β_wiggle) w.r.t.
// β_eta picks up a chain-rule factor g'(q₀) = 1 + B'(q₀) · β_wiggle / range_width
// from the dependence of B on q₀.
//
// Whitening uses the Cholesky of the joint Hessian at the mode, exactly as for
// the standard NutsPosterior. C^1 linear extension outside the training knot
// range prevents basis evaluation discontinuities.

/// Fixed spline artifacts for link-wiggle posterior sampling.
#[derive(Clone)]
pub struct LinkWiggleSplineArtifacts {
    /// Knot range (min, max) from training (in standardized [0,1] space of q₀)
    pub knot_range: (f64, f64),
    /// Full knot vector for the shared monotone I-spline basis
    pub knot_vector: Array1<f64>,
    /// I-spline degree
    pub degree: usize,
}

/// Whitened log-posterior target for joint (β_eta, β_wiggle) with analytical gradients.
#[derive(Clone)]
pub struct LinkWigglePosterior {
    /// Main design matrix X (n × p_main)
    x: Arc<Array2<f64>>,
    y: Arc<Array1<f64>>,
    weights: Arc<Array1<f64>>,
    /// Penalty for main coefficients (p_main × p_main)
    penalty_base: Arc<Array2<f64>>,
    /// Penalty for wiggle coefficients (p_wiggle × p_wiggle)
    penalty_link: Arc<Array2<f64>>,
    mode_beta: Arc<Array1<f64>>,
    mode_theta: Arc<Array1<f64>>,
    spline: LinkWiggleSplineArtifacts,
    /// L where LL^T = H^{-1} (joint Hessian)
    chol: Array2<f64>,
    /// L^T for gradient chain rule
    chol_t: Array2<f64>,
    p_base: usize,
    p_link: usize,
    n_samples: usize,
    nuts_family: NutsFamily,
    /// Family-specific noise parameter: Gaussian sigma or Gamma shape.
    scale: f64,
}

impl LinkWigglePosterior {
    /// Standardize q₀ values to [0,1] range using training knot bounds.
    #[inline]
    fn standardized_z(&self, u: &Array1<f64>) -> (Array1<f64>, Array1<f64>, f64) {
        let (min_u, max_u) = self.spline.knot_range;
        let rw = (max_u - min_u).max(1e-6);
        let z_raw: Array1<f64> = u.mapv(|v| (v - min_u) / rw);
        let z_c: Array1<f64> = z_raw.mapv(|z| z.clamp(0.0, 1.0));
        (z_raw, z_c, rw)
    }

    /// Creates a new link-wiggle posterior target.
    pub fn new(
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        weights: ArrayView1<f64>,
        penalty_base: ArrayView2<f64>,
        penalty_link: ArrayView2<f64>,
        mode_beta: ArrayView1<f64>,
        mode_theta: ArrayView1<f64>,
        hessian: ArrayView2<f64>,
        spline: LinkWiggleSplineArtifacts,
        nuts_family: NutsFamily,
        scale: f64,
    ) -> Result<Self, String> {
        let n_samples = x.nrows();
        let p_base = x.ncols();
        let p_link = mode_theta.len();
        let dim = p_base + p_link;
        if hessian.nrows() != dim || hessian.ncols() != dim {
            return Err(format!(
                "LinkWigglePosterior: Hessian dim mismatch: {}x{} vs expected {}x{}",
                hessian.nrows(),
                hessian.ncols(),
                dim,
                dim,
            ));
        }
        let hessian_owned = hessian.to_owned();
        let chol_factor = hessian_owned
            .cholesky(Side::Lower)
            .map_err(|e| format!("LinkWigglePosterior Cholesky failed: {:?}", e))?;
        let l_h = chol_factor.lower_triangular();
        let chol = solve_upper_triangular_transpose(&l_h, dim);
        let chol_t = chol.t().to_owned();
        Ok(Self {
            x: Arc::new(x.to_owned()),
            y: Arc::new(y.to_owned()),
            weights: Arc::new(weights.to_owned()),
            penalty_base: Arc::new(penalty_base.to_owned()),
            penalty_link: Arc::new(penalty_link.to_owned()),
            mode_beta: Arc::new(mode_beta.to_owned()),
            mode_theta: Arc::new(mode_theta.to_owned()),
            spline,
            chol,
            chol_t,
            p_base,
            p_link,
            n_samples,
            nuts_family,
            scale,
        })
    }

    /// Evaluate the wiggle basis and compute η = q₀ + B(q₀)·θ with C^1 linear extension.
    fn evaluate_link(&self, u: &Array1<f64>, theta: &Array1<f64>) -> (Array2<f64>, Array1<f64>) {
        let n = u.len();
        if theta.is_empty() {
            return (Array2::zeros((n, 0)), u.clone());
        }

        let (z_raw, z_c, _) = self.standardized_z(u);
        let Ok(mut basis) = monotone_wiggle_basis_with_derivative_order(
            z_c.view(),
            &self.spline.knot_vector,
            self.spline.degree,
            0,
        ) else {
            return (Array2::zeros((n, theta.len())), u.clone());
        };
        if basis.ncols() != theta.len() {
            return (Array2::zeros((n, theta.len())), u.clone());
        }

        // C^1 linear extension outside [0, 1]:
        // B_ext(z_raw) = B(z_c) + (z_raw - z_c) * B'(z_c)
        let mut needs_ext = false;
        for i in 0..n {
            if (z_raw[i] - z_c[i]).abs() > 1e-12 {
                needs_ext = true;
                break;
            }
        }
        if needs_ext {
            if let Ok(b_prime) = monotone_wiggle_basis_with_derivative_order(
                z_c.view(),
                &self.spline.knot_vector,
                self.spline.degree,
                1,
            ) {
                for i in 0..n {
                    let dz = z_raw[i] - z_c[i];
                    if dz.abs() <= 1e-12 {
                        continue;
                    }
                    for j in 0..basis.ncols().min(b_prime.ncols()) {
                        basis[[i, j]] += dz * b_prime[[i, j]];
                    }
                }
            }
        }
        (basis.clone(), u + &basis.dot(theta))
    }

    /// Compute dη/dq₀ = 1 + B'(q₀)·θ / range_width (chain-rule factor for β_eta gradient).
    fn compute_g_prime(&self, u: &Array1<f64>, theta: &Array1<f64>) -> Array1<f64> {
        let n = u.len();
        let mut g = Array1::<f64>::ones(n);
        let (_, z_c, rw) = self.standardized_z(u);
        if theta.is_empty() {
            return g;
        }

        let Ok(b_prime_constrained) = monotone_wiggle_basis_with_derivative_order(
            z_c.view(),
            &self.spline.knot_vector,
            self.spline.degree,
            1,
        ) else {
            return g;
        };
        if b_prime_constrained.ncols() != theta.len() {
            return g;
        }
        let dwiggle_dz = b_prime_constrained.dot(theta);
        for i in 0..n {
            g[i] = 1.0 + dwiggle_dz[i] / rw;
        }
        g
    }

    /// Compute log-posterior and gradient in whitened coordinates.
    fn compute_logp_and_grad(&self, z: &Array1<f64>) -> (f64, Array1<f64>) {
        let dim = self.p_base + self.p_link;

        // Un-whiten: q = mode + L·z
        let mut mode = Array1::<f64>::zeros(dim);
        mode.slice_mut(ndarray::s![0..self.p_base])
            .assign(&self.mode_beta);
        mode.slice_mut(ndarray::s![self.p_base..])
            .assign(&self.mode_theta);
        let q = &mode + &self.chol.dot(z);
        let beta = q.slice(ndarray::s![0..self.p_base]).to_owned();
        let theta = q.slice(ndarray::s![self.p_base..]).to_owned();

        // Compute η = q₀ + B(q₀)·θ where q₀ = X·β
        let u = self.x.dot(&beta);
        let (bwiggle, eta) = self.evaluate_link(&u, &theta);

        // Log-likelihood and residuals via family dispatch
        let ll;
        let mut residual = Array1::<f64>::zeros(self.n_samples);
        match self.nuts_family {
            NutsFamily::Gaussian => {
                let inv_scale_sq = 1.0 / (self.scale * self.scale).max(1e-10);
                let mut ll_acc = 0.0;
                for i in 0..self.n_samples {
                    let r = self.y[i] - eta[i];
                    let w = self.weights[i];
                    ll_acc -= 0.5 * w * r * r * inv_scale_sq;
                    residual[i] = w * r * inv_scale_sq;
                }
                ll = ll_acc;
            }
            NutsFamily::BinomialLogit => {
                let mut ll_acc = 0.0;
                for i in 0..self.n_samples {
                    let eta_i = eta[i];
                    let (y_i, w_i) = (self.y[i], self.weights[i]);
                    ll_acc += w_i * (y_i * eta_i - NutsPosterior::log1pexp(eta_i));
                    let mu = NutsPosterior::sigmoid_stable(eta_i);
                    residual[i] = w_i * (y_i - mu);
                }
                ll = ll_acc;
            }
            NutsFamily::BinomialProbit => {
                let mut ll_acc = 0.0;
                for i in 0..self.n_samples {
                    let eta_i = eta[i];
                    let (y_i, w_i) = (self.y[i], self.weights[i]);
                    let log_phi_pos = log_ndtr(eta_i);
                    let log_phi_neg = log_ndtr(-eta_i);
                    ll_acc += w_i * (y_i * log_phi_pos + (1.0 - y_i) * log_phi_neg);
                    let log_phi = standard_normal_log_pdf(eta_i);
                    let ratio_pos = (log_phi - log_phi_pos).exp();
                    let ratio_neg = (log_phi - log_phi_neg).exp();
                    residual[i] = w_i * (y_i * ratio_pos - (1.0 - y_i) * ratio_neg);
                }
                ll = ll_acc;
            }
            NutsFamily::BinomialCLogLog => {
                let mut ll_acc = 0.0;
                for i in 0..self.n_samples {
                    let eta_i = eta[i];
                    let (y_i, w_i) = (self.y[i], self.weights[i]);
                    let neg_exp_eta = (-eta_i.exp()).max(-700.0);
                    let log_mu = neg_exp_eta.ln_1p().min(0.0).max(-700.0);
                    let log_1m_mu = neg_exp_eta.min(0.0).max(-700.0);
                    ll_acc += w_i * (y_i * log_mu + (1.0 - y_i) * log_1m_mu);
                    let exp_eta = eta_i.exp().min(1e300);
                    let exp_neg_exp_eta = neg_exp_eta.exp();
                    let mu = (1.0 - exp_neg_exp_eta).clamp(1e-15, 1.0 - 1e-15);
                    let dmudeta = exp_eta * exp_neg_exp_eta;
                    residual[i] = w_i * (y_i - mu) * dmudeta / (mu * (1.0 - mu)).max(1e-30);
                }
                ll = ll_acc;
            }
            NutsFamily::PoissonLog => {
                let mut ll_acc = 0.0;
                for i in 0..self.n_samples {
                    let eta_i = eta[i].clamp(-30.0, 30.0);
                    let (y_i, w_i) = (self.y[i], self.weights[i]);
                    let mu = eta_i.exp();
                    ll_acc += w_i * (y_i * eta_i - mu);
                    residual[i] = w_i * (y_i - mu);
                }
                ll = ll_acc;
            }
            NutsFamily::GammaLog => {
                let mut ll_acc = 0.0;
                let shape = self.scale.max(1e-10);
                for i in 0..self.n_samples {
                    let eta_i = eta[i].clamp(-30.0, 30.0);
                    let (y_i, w_i) = (self.y[i], self.weights[i]);
                    let mu = eta_i.exp();
                    ll_acc += w_i * shape * (-y_i / mu - eta_i);
                    residual[i] = w_i * shape * (y_i / mu - 1.0);
                }
                ll = ll_acc;
            }
        }

        // Gradient w.r.t. θ (wiggle): ∂ℓ/∂θ = B(q₀)^T · residual − S_link · θ
        let grad_theta = &bwiggle.t().dot(&residual) - &self.penalty_link.dot(&theta);

        // Gradient w.r.t. β_eta: ∂ℓ/∂β = X^T · (residual ⊙ g'(q₀)) − S_base · β
        // where g'(q₀) = dη/dq₀ is the chain-rule factor
        let g_prime = self.compute_g_prime(&u, &theta);
        let r_scaled: Array1<f64> = residual
            .iter()
            .zip(g_prime.iter())
            .map(|(&r, &g)| r * g)
            .collect();
        let grad_beta = &fast_atv(&self.x, &r_scaled) - &self.penalty_base.dot(&beta);

        // Penalty
        let penalty = 0.5 * beta.dot(&self.penalty_base.dot(&beta))
            + 0.5 * theta.dot(&self.penalty_link.dot(&theta));

        // Assemble joint gradient and transform to whitened space
        let mut grad_q = Array1::<f64>::zeros(dim);
        grad_q
            .slice_mut(ndarray::s![0..self.p_base])
            .assign(&grad_beta);
        grad_q
            .slice_mut(ndarray::s![self.p_base..])
            .assign(&grad_theta);
        (ll - penalty, self.chol_t.dot(&grad_q))
    }

    /// Get the Cholesky factor L for un-whitening samples.
    pub fn chol(&self) -> &Array2<f64> {
        &self.chol
    }

    /// Get the mode [β_eta; β_wiggle].
    pub fn mode_joint(&self) -> Array1<f64> {
        let dim = self.p_base + self.p_link;
        let mut mode = Array1::<f64>::zeros(dim);
        mode.slice_mut(ndarray::s![0..self.p_base])
            .assign(&self.mode_beta);
        mode.slice_mut(ndarray::s![self.p_base..])
            .assign(&self.mode_theta);
        mode
    }
}

impl HamiltonianTarget<Array1<f64>> for LinkWigglePosterior {
    fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
        let (logp, g) = self.compute_logp_and_grad(position);
        grad.assign(&g);
        logp
    }
}

/// Runs NUTS sampling for joint (β_eta, β_wiggle) in a link-wiggle model.
pub fn run_link_wiggle_nuts_sampling(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    penalty_base: ArrayView2<f64>,
    penalty_link: ArrayView2<f64>,
    mode_beta: ArrayView1<f64>,
    mode_theta: ArrayView1<f64>,
    hessian: ArrayView2<f64>,
    spline: LinkWiggleSplineArtifacts,
    nuts_family: NutsFamily,
    scale: f64,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    let dim = mode_beta.len() + mode_theta.len();
    let target = LinkWigglePosterior::new(
        x,
        y,
        weights,
        penalty_base,
        penalty_link,
        mode_beta,
        mode_theta,
        hessian,
        spline,
        nuts_family,
        scale,
    )?;
    let chol = target.chol().clone();
    let mode_arr = target.mode_joint();

    let mut rng = StdRng::seed_from_u64(config.seed);
    let initial_positions: Vec<Array1<f64>> = (0..config.n_chains)
        .map(|_| {
            Array1::from_shape_fn(dim, |_| {
                let u1: f64 = rng.random::<f64>().max(1e-10);
                let u2: f64 = rng.random();
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos() * 0.1
            })
        })
        .collect();

    let mass_cfg = robust_mass_matrix_config(dim, config.nwarmup);
    let mut sampler = GenericNUTS::new_with_mass_matrix(
        target,
        initial_positions,
        config.target_accept,
        mass_cfg,
    );

    let (samples_array, run_stats) = sampler
        .run_progress(config.n_samples, config.nwarmup)
        .map_err(|e| format!("Link-wiggle NUTS sampling failed: {e}"))?;
    log::info!("Link-wiggle NUTS sampling complete: {}", run_stats);

    // Un-whiten samples: β = mode + L·z
    let shape = samples_array.shape();
    let n_chains = shape[0];
    let n_samples_out = shape[1];
    let total_samples = n_chains * n_samples_out;

    let mut samples = Array2::<f64>::zeros((total_samples, dim));
    let mut z_buffer = Array1::<f64>::zeros(dim);
    for chain in 0..n_chains {
        for sample_i in 0..n_samples_out {
            z_buffer.assign(&samples_array.slice(ndarray::s![chain, sample_i, ..]));
            samples
                .row_mut(chain * n_samples_out + sample_i)
                .assign(&(&mode_arr + &chol.dot(&z_buffer)));
        }
    }

    let posterior_mean = samples
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(dim));
    let posterior_std = samples.std_axis(Axis(0), 0.0);
    let (rhat, ess) = compute_split_rhat_and_ess(&samples_array);
    let converged = rhat < 1.1 && ess > 100.0;

    Ok(NutsResult {
        samples,
        posterior_mean,
        posterior_std,
        rhat,
        ess,
        converged,
    })
}

// ============================================================================
// Joint (β, ρ) HMC for Skewed Posteriors
// ============================================================================
//
// When the Laplace approximation to the marginal likelihood is unreliable
// (high posterior skewness), we bypass LAML entirely and sample from the
// joint posterior p(β, ρ | y) ∝ p(y|β) p(β|ρ) p(ρ).
//
// The joint log-posterior is:
//   log p(β, ρ | y) = ℓ(y|β) + Φ(β) [if Firth]
//                    - 0.5 β'S(ρ)β + 0.5 log|S(ρ)|_+ + log p(ρ) + const
//
// Gradients:
//   ∇_β: ∇_β ℓ + ∇_β Φ(β) [if Firth] - S(ρ) β
//   ∂/∂ρ_k: -0.5 λ_k β'S_k β + 0.5 tr(S_+⁻¹ A_k) + ∂log p(ρ)/∂ρ_k
//
// This completely avoids the Laplace approximation. When Firth bias reduction
// is active, the sampled target also includes the Jeffreys term Φ(β) in
// addition to the smoothing-parameter prior.

/// Directional cubic non-Gaussianity diagnostic for the Laplace approximation.
///
/// For each positive-curvature Hessian eigenpair `(lambda_r, v_r)`, this computes
///
///   gamma_r = T[v_r, v_r, v_r] / lambda_r^(3/2)
///            = Σ_i c_i (x_i^T v_r)^3 / lambda_r^(3/2),
///
/// and reports `max_r |gamma_r|`. This is invariant to arbitrary coordinate
/// relabeling and uses the full directional cubic contraction rather than only
/// diagonal tensor entries.
pub fn laplace_directional_cubic_diagnostic(
    hessian: &Array2<f64>,
    design: &DesignMatrix,
    c_weights: &Array1<f64>,
) -> Result<(f64, Array1<f64>), String> {
    let p = hessian.nrows();
    if p == 0 || hessian.ncols() != p {
        return Ok((0.0, Array1::zeros(0)));
    }

    let sym_h = (hessian + &hessian.t()) * 0.5;
    let (evals, evecs) = sym_h
        .eigh(Side::Lower)
        .map_err(|e| format!("directional cubic diagnostic eigendecomposition failed: {e}"))?;
    let max_eval = evals.iter().fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
    let tol = (max_eval * 1.0e-12).max(1.0e-14);
    let mut directional = Array1::<f64>::zeros(p);
    let mut max_abs = 0.0_f64;

    // Build the whitening transform L^{-1} where H = L L^T, so that
    // the standardized cubic along whitened direction u is:
    //   gamma(u) = T[L^{-T}u, L^{-T}u, L^{-T}u]  for ||u||=1
    // Eigenvector directions v_r satisfy u_r = lambda_r^{1/2} v_r (after
    // appropriate normalization), so gamma_r = T[v_r,v_r,v_r] / lambda_r^{3/2}.

    // Phase 1: evaluate gamma_r for all positive-curvature eigenvectors.
    for r in 0..p {
        let lambda = evals[r];
        if lambda <= tol {
            continue;
        }
        let v = evecs.column(r);
        let gamma = directional_cubic_contraction(design, c_weights, &v) / lambda.powf(1.5);
        directional[r] = if gamma.is_finite() { gamma } else { 0.0 };
        max_abs = max_abs.max(directional[r].abs());
    }

    // Phase 2: power-iteration refinement in whitened space.
    //
    // The supremum of |gamma(u)| over ||u||_H=1 can exceed the max over
    // eigenvectors. We approximate it with a few rounds of cubic power
    // iteration: given current direction v, the gradient of T[v,v,v] w.r.t.
    // v on the H-unit sphere is 3 T[·,v,v] projected onto the tangent space.
    // Since T[·,v,v] = X^T diag(c_i (x_i^T v)^2) which is a matrix-vector
    // product, each iteration is O(np).
    //
    // We seed from the eigenvector with largest |gamma_r| and also from a
    // few random probe directions.
    if p >= 2 {
        // Build H^{-1/2} columns for whitening: H^{-1/2} = V diag(1/sqrt(lam)) V^T
        // We need it to map whitened u -> original v = H^{-1/2} u, and
        // H^{1/2} to project back: H^{1/2} v = V diag(sqrt(lam)) V^T v.
        let positive_mask: Vec<bool> = evals.iter().map(|&ev| ev > tol).collect();
        let n_pos = positive_mask.iter().filter(|&&m| m).count();
        if n_pos >= 2 {
            let max_abs_from_probes = cubic_power_iteration_refinement(
                design,
                c_weights,
                &evals,
                &evecs,
                &positive_mask,
                n_pos,
            );
            if max_abs_from_probes > max_abs {
                max_abs = max_abs_from_probes;
            }
        }
    }

    Ok((max_abs, directional))
}

/// Compute T[v,v,v] = Σ_i c_i (x_i^T v)^3 for a given direction v.
fn directional_cubic_contraction(
    design: &DesignMatrix,
    c_weights: &Array1<f64>,
    v: &ArrayView1<f64>,
) -> f64 {
    match design.as_sparse() {
        Some(x_sparse) => {
            let (symbolic, values) = x_sparse.as_ref().parts();
            let col_ptr = symbolic.col_ptr();
            let row_idx = symbolic.row_idx();
            let mut row_scores = vec![0.0_f64; x_sparse.nrows()];
            for col in 0..x_sparse.ncols() {
                let coeff = v[col];
                for ptr in col_ptr[col]..col_ptr[col + 1] {
                    row_scores[row_idx[ptr]] += values[ptr] * coeff;
                }
            }
            let mut cubic = 0.0_f64;
            for i in 0..row_scores.len().min(c_weights.len()) {
                cubic += c_weights[i] * row_scores[i].powi(3);
            }
            cubic
        }
        None => {
            let x_dense = design.as_dense_cow();
            let x_dense = x_dense.as_ref();
            let mut cubic = 0.0_f64;
            for i in 0..x_dense.nrows().min(c_weights.len()) {
                let proj = x_dense.row(i).dot(v);
                cubic += c_weights[i] * proj.powi(3);
            }
            cubic
        }
    }
}

/// Compute the gradient of T[v,v,v] w.r.t. v:  3 X^T diag(c_i (x_i^T v)^2) 1.
/// More precisely: ∂/∂v T[v,v,v] = 3 Σ_i c_i (x_i^T v)^2 x_i.
fn directional_cubic_gradient(
    design: &DesignMatrix,
    c_weights: &Array1<f64>,
    v: &Array1<f64>,
) -> Array1<f64> {
    let p = v.len();
    match design.as_sparse() {
        Some(x_sparse) => {
            let (symbolic, values) = x_sparse.as_ref().parts();
            let col_ptr = symbolic.col_ptr();
            let row_idx = symbolic.row_idx();
            let n = x_sparse.nrows();
            let mut row_scores = vec![0.0_f64; n];
            for col in 0..x_sparse.ncols() {
                let coeff = v[col];
                for ptr in col_ptr[col]..col_ptr[col + 1] {
                    row_scores[row_idx[ptr]] += values[ptr] * coeff;
                }
            }
            // quadratic weights: 3 c_i (x_i^T v)^2
            let mut quad_weights = vec![0.0_f64; n];
            for i in 0..n.min(c_weights.len()) {
                quad_weights[i] = 3.0 * c_weights[i] * row_scores[i] * row_scores[i];
            }
            // X^T quad_weights
            let mut grad = Array1::<f64>::zeros(p);
            for col in 0..x_sparse.ncols() {
                let mut acc = 0.0_f64;
                for ptr in col_ptr[col]..col_ptr[col + 1] {
                    acc += values[ptr] * quad_weights[row_idx[ptr]];
                }
                grad[col] = acc;
            }
            grad
        }
        None => {
            let x_dense = design.as_dense_cow();
            let x_dense = x_dense.as_ref();
            let n = x_dense.nrows();
            let mut grad = Array1::<f64>::zeros(p);
            for i in 0..n.min(c_weights.len()) {
                let proj = x_dense.row(i).dot(v);
                let w = 3.0 * c_weights[i] * proj * proj;
                // scaled_add works with any ArrayBase reference.
                let row = x_dense.row(i);
                for j in 0..p {
                    grad[j] += w * row[j];
                }
            }
            grad
        }
    }
}

/// Power-iteration refinement for the supremum of |gamma(u)| over ||u||_H = 1.
///
/// Seeds from the best eigenvector direction plus deterministic probe
/// directions constructed from pairs of eigenvectors. Runs a few Riemannian
/// gradient ascent steps on the whitened unit sphere.
fn cubic_power_iteration_refinement(
    design: &DesignMatrix,
    c_weights: &Array1<f64>,
    evals: &Array1<f64>,
    evecs: &Array2<f64>,
    positive_mask: &[bool],
    n_pos: usize,
) -> f64 {
    let p = evals.len();
    let max_probes = 8;
    let max_iters = 5;

    // Helper: convert whitened u -> original v = Σ_r (u_r / sqrt(lam_r)) * evec_r
    // (only over positive eigenspace).
    let to_original = |u: &Array1<f64>| -> Array1<f64> {
        let mut v = Array1::<f64>::zeros(p);
        let mut idx = 0;
        for r in 0..p {
            if positive_mask[r] {
                let scale = u[idx] / evals[r].sqrt();
                let col = evecs.column(r);
                for j in 0..p {
                    v[j] += scale * col[j];
                }
                idx += 1;
            }
        }
        v
    };

    // Helper: project original-space vector to whitened: u_j = sqrt(lam_r) (evec_r^T g)
    let to_whitened = |g: &Array1<f64>| -> Array1<f64> {
        let mut u = Array1::<f64>::zeros(n_pos);
        let mut idx = 0;
        for r in 0..p {
            if positive_mask[r] {
                u[idx] = evals[r].sqrt() * evecs.column(r).dot(g);
                idx += 1;
            }
        }
        u
    };

    // Evaluate |gamma(u)| for whitened direction u.
    let eval_gamma = |u: &Array1<f64>| -> f64 {
        let norm = u.dot(u).sqrt();
        if norm < 1e-30 {
            return 0.0;
        }
        let u_normed: Array1<f64> = u / norm;
        let v = to_original(&u_normed);
        // gamma = T[v,v,v] since v already has ||v||_H = 1
        let cubic = directional_cubic_contraction(design, c_weights, &v.view());
        if cubic.is_finite() { cubic.abs() } else { 0.0 }
    };

    // One step of Riemannian gradient ascent on the whitened sphere for |T[v,v,v]|.
    let refine_step = |u: &Array1<f64>| -> Array1<f64> {
        let norm = u.dot(u).sqrt();
        if norm < 1e-30 {
            return u.clone();
        }
        let u_normed: Array1<f64> = u / norm;
        let v = to_original(&u_normed);
        // Gradient of T[v,v,v] w.r.t. v in original space
        let grad_v = directional_cubic_gradient(design, c_weights, &v);
        // Map to whitened space
        let mut grad_u = to_whitened(&grad_v);
        // Project onto tangent plane of sphere: grad - (grad . u) u
        let dot = grad_u.dot(&u_normed);
        grad_u.scaled_add(-dot, &u_normed);
        // Sign: we want to maximize |T|, so follow sign(T) * grad
        let cubic_val = directional_cubic_contraction(design, c_weights, &v.view());
        let sign = if cubic_val >= 0.0 { 1.0 } else { -1.0 };
        let step_size = 0.3;
        let mut u_new = &u_normed + &(&grad_u * (sign * step_size));
        let new_norm = u_new.dot(&u_new).sqrt();
        if new_norm > 1e-30 {
            u_new /= new_norm;
        }
        u_new
    };

    let mut best = 0.0_f64;

    // Build seed directions:
    // (a) The eigenvector with largest |gamma_r| (already computed by caller,
    //     but we re-derive the whitened form here).
    // (b) Deterministic probe directions from pairs of top eigenvectors:
    //     (e_i + e_j) / sqrt(2) and (e_i - e_j) / sqrt(2) in whitened space.
    let mut seeds: Vec<Array1<f64>> = Vec::with_capacity(max_probes);

    // Seed (a): each eigenvector is a standard basis vector in whitened space.
    // Find the one with largest |gamma|.
    let mut best_eig_idx = 0;
    let mut best_eig_gamma = 0.0_f64;
    for j in 0..n_pos {
        let mut u = Array1::<f64>::zeros(n_pos);
        u[j] = 1.0;
        let g = eval_gamma(&u);
        if g > best_eig_gamma {
            best_eig_gamma = g;
            best_eig_idx = j;
        }
    }
    best = best.max(best_eig_gamma);
    let mut u_best = Array1::<f64>::zeros(n_pos);
    u_best[best_eig_idx] = 1.0;
    seeds.push(u_best);

    // Seed (b): pairwise combinations of the top few eigenvectors.
    let n_top = n_pos.min(4);
    for i in 0..n_top {
        for j in (i + 1)..n_top {
            if seeds.len() >= max_probes {
                break;
            }
            let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
            let mut u_plus = Array1::<f64>::zeros(n_pos);
            u_plus[i] = inv_sqrt2;
            u_plus[j] = inv_sqrt2;
            seeds.push(u_plus);
            if seeds.len() < max_probes {
                let mut u_minus = Array1::<f64>::zeros(n_pos);
                u_minus[i] = inv_sqrt2;
                u_minus[j] = -inv_sqrt2;
                seeds.push(u_minus);
            }
        }
    }

    // Run power iteration from each seed.
    for seed in &seeds {
        let mut u = seed.clone();
        for _ in 0..max_iters {
            u = refine_step(&u);
        }
        let g = eval_gamma(&u);
        best = best.max(g);
    }

    best
}

/// Result of joint (β, ρ) sampling.
#[derive(Clone, Debug)]
pub struct JointBetaRhoResult {
    /// Coefficient samples: shape (n_total_samples, n_beta)
    pub beta_samples: Array2<f64>,
    /// Log-smoothing parameter samples: shape (n_total_samples, n_rho)
    pub rho_samples: Array2<f64>,
    /// Posterior mean of β
    pub beta_mean: Array1<f64>,
    /// Posterior mean of ρ
    pub rho_mean: Array1<f64>,
    /// R-hat diagnostic
    pub rhat: f64,
    /// Effective sample size
    pub ess: f64,
    /// Whether sampling converged
    pub converged: bool,
    /// Max skewness that triggered this sampling
    pub trigger_skewness: f64,
}

/// Joint (β, ρ) posterior target for NUTS.
///
/// Samples from p(β, ρ | y) ∝ p(y|β) p(β|ρ) p(ρ) directly,
/// completely bypassing the Laplace approximation.
///
/// The parameter vector is [z_β; ρ] where z_β = L⁻¹(β - μ) is the
/// whitened β and ρ is the raw log-smoothing parameters.
struct JointBetaRhoPosterior {
    data: SharedData,
    /// L where LL' = H⁻¹ (whitening for β block)
    chol: Array2<f64>,
    /// L' for chain rule
    chol_t: Array2<f64>,
    /// Family for log-likelihood computation
    likelihood_family: LikelihoodFamily,
    /// Exact runtime inverse-link state for adaptive binomial links.
    inverse_link: InverseLink,
    /// Dimension of β
    n_beta: usize,
    /// Dimension of ρ
    n_rho: usize,
    /// Canonical penalties in the transformed basis.
    penalty_canonical: Vec<crate::construction::CanonicalPenalty>,
    /// Fixed prior on rho used by the sampled target.
    rho_prior: RhoPrior,
    /// LAML-converged ρ (used only to initialize chains)
    rho_mode: Array1<f64>,
    /// Whether to add the identifiable-subspace Jeffreys/Firth term to the
    /// target
    firth_enabled: bool,
}

impl JointBetaRhoPosterior {
    fn new(
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        weights: ArrayView1<f64>,
        mode: ArrayView1<f64>,
        hessian: ArrayView2<f64>,
        penalty_canonical: Vec<crate::construction::CanonicalPenalty>,
        rho_mode: ArrayView1<f64>,
        likelihood_family: LikelihoodFamily,
        inverse_link: InverseLink,
        gamma_shape: Option<f64>,
        rho_prior: RhoPrior,
        firth_enabled: bool,
    ) -> Result<Self, String> {
        let n_samples = x.nrows();
        let n_beta = x.ncols();
        let n_rho = penalty_canonical.len();

        if rho_mode.len() != n_rho {
            return Err(format!(
                "rho_mode length {} != penalty count {}",
                rho_mode.len(),
                n_rho
            ));
        }

        match likelihood_family {
            LikelihoodFamily::BinomialLogit => {
                if !matches!(&inverse_link, InverseLink::Standard(LinkFunction::Logit)) {
                    return Err("Joint HMC BinomialLogit requires a logit inverse link".to_string());
                }
            }
            LikelihoodFamily::BinomialProbit => {
                if !matches!(&inverse_link, InverseLink::Standard(LinkFunction::Probit)) {
                    return Err(
                        "Joint HMC BinomialProbit requires a probit inverse link".to_string()
                    );
                }
            }
            LikelihoodFamily::BinomialCLogLog => {
                if !matches!(&inverse_link, InverseLink::Standard(LinkFunction::CLogLog)) {
                    return Err(
                        "Joint HMC BinomialCLogLog requires a cloglog inverse link".to_string()
                    );
                }
            }
            LikelihoodFamily::BinomialLatentCLogLog => {
                if !matches!(&inverse_link, InverseLink::LatentCLogLog(_)) {
                    return Err(
                        "Joint HMC BinomialLatentCLogLog requires latent cloglog link state"
                            .to_string(),
                    );
                }
            }
            LikelihoodFamily::BinomialSas => {
                if !matches!(&inverse_link, InverseLink::Sas(_)) {
                    return Err("Joint HMC BinomialSas requires SAS link state".to_string());
                }
            }
            LikelihoodFamily::BinomialBetaLogistic => {
                if !matches!(&inverse_link, InverseLink::BetaLogistic(_)) {
                    return Err(
                        "Joint HMC BinomialBetaLogistic requires Beta-Logistic link state"
                            .to_string(),
                    );
                }
            }
            LikelihoodFamily::BinomialMixture => {
                if !matches!(&inverse_link, InverseLink::Mixture(_)) {
                    return Err("Joint HMC BinomialMixture requires mixture link state".to_string());
                }
            }
            LikelihoodFamily::GaussianIdentity => {
                if !matches!(&inverse_link, InverseLink::Standard(LinkFunction::Identity)) {
                    return Err(
                        "Joint HMC GaussianIdentity requires an identity inverse link".to_string(),
                    );
                }
            }
            LikelihoodFamily::PoissonLog | LikelihoodFamily::GammaLog => {
                if !matches!(&inverse_link, InverseLink::Standard(LinkFunction::Log)) {
                    return Err("Joint HMC log-link family requires a log inverse link".to_string());
                }
            }
            LikelihoodFamily::RoystonParmar => {
                return Err("Joint HMC fallback is not implemented for RoystonParmar".to_string());
            }
        }

        validate_firth_likelihood_support(likelihood_family, firth_enabled)?;

        // Cholesky of H for β-whitening (same as NutsPosterior)
        let hessian_owned = hessian.to_owned();
        let chol_factor = hessian_owned
            .cholesky(Side::Lower)
            .map_err(|e| format!("Joint HMC: Hessian Cholesky failed: {:?}", e))?;
        let l_h = chol_factor.lower_triangular();
        let chol = solve_upper_triangular_transpose(&l_h, n_beta);
        let chol_t = chol.t().to_owned();

        // Build combined penalty at the LAML mode (for SharedData)
        let lambdas_mode: Array1<f64> = rho_mode.mapv(f64::exp);
        let mut s_combined = Array2::<f64>::zeros((n_beta, n_beta));
        for (k, cp) in penalty_canonical.iter().enumerate() {
            cp.accumulate_weighted(&mut s_combined, lambdas_mode[k]);
        }

        let data = SharedData {
            x: Arc::new(x.to_owned()),
            y: Arc::new(y.to_owned()),
            weights: Arc::new(weights.to_owned()),
            penalty: Arc::new(s_combined),
            mode: Arc::new(mode.to_owned()),
            gamma_shape: gamma_shape.unwrap_or(1.0),
            n_samples,
            dim: n_beta,
        };

        Ok(Self {
            data,
            chol,
            chol_t,
            likelihood_family,
            inverse_link,
            n_beta,
            n_rho,
            penalty_canonical,
            rho_prior,
            rho_mode: rho_mode.to_owned(),
            firth_enabled,
        })
    }

    /// Compute the joint log-posterior and gradient.
    ///
    /// The joint log-posterior is:
    ///   log p(β, ρ | y) = ℓ(y|β) + ½ log|I(β)| [if Firth]
    ///                    − ½β'S(ρ)β + ½ log|S(ρ)|₊ + log p(ρ) + const
    ///
    /// This is NOT the REML/LAML objective (which integrates out β). Here β is
    /// an explicit parameter being sampled, evaluated at arbitrary values — not
    /// just at the mode β̂(ρ).
    ///
    /// Parameter vector layout: [z_β (whitened, length n_beta); ρ (length n_rho)]
    fn compute_joint_logp_and_grad(&self, params: &Array1<f64>) -> (f64, Array1<f64>) {
        let n_beta = self.n_beta;
        let n_rho = self.n_rho;

        // Split parameter vector
        let z = params.slice(ndarray::s![..n_beta]).to_owned();
        let rho = params.slice(ndarray::s![n_beta..]).to_owned();
        let lambdas: Array1<f64> = rho.mapv(f64::exp);

        // Un-whiten: β = μ + L z
        let beta = self.data.mode.as_ref() + &self.chol.dot(&z);

        // η = X β
        let eta = self.data.x.dot(&beta);

        // ---- Log-likelihood ℓ(y|β) and ∇_β ℓ ----
        let (ll, mut grad_ll_beta) = match joint_family_logp_and_grad(
            self.likelihood_family,
            &self.inverse_link,
            &self.data,
            &eta,
        ) {
            Ok(value) => value,
            Err(err) => {
                log::warn!(
                    "[Joint HMC] likelihood target became invalid at the current state: {}",
                    err
                );
                return (f64::NEG_INFINITY, Array1::zeros(n_beta + n_rho));
            }
        };

        let mut firth_logdet = 0.0;
        if self.firth_enabled {
            match firth_jeffreys_logp_and_grad(NutsFamily::BinomialLogit, &self.data, &eta) {
                Ok((value, grad_beta_firth)) => {
                    firth_logdet = value;
                    grad_ll_beta += &grad_beta_firth;
                }
                Err(err) => {
                    log::warn!(
                        "[Joint HMC/Firth] Jeffreys target became invalid at the current state: {}",
                        err
                    );
                    return (f64::NEG_INFINITY, Array1::zeros(n_beta + n_rho));
                }
            }
        }

        // ---- Penalty: -0.5 β'S(ρ)β ----
        // S(ρ) = Σ_k λ_k S_k where S_k = R_k'R_k (precomputed in penalty_matrices).
        // Uses penalty_roots for the efficient ||R_k β||² form.
        let mut penalty_val = 0.0;
        let mut s_beta = Array1::<f64>::zeros(n_beta);
        let mut grad_rho = Array1::<f64>::zeros(n_rho);

        for (k, cp) in self.penalty_canonical.iter().enumerate() {
            // Block-local quadratic: β'S_k β via root
            let r = &cp.col_range;
            let beta_block = beta.slice(ndarray::s![r.start..r.end]);
            let r_beta: Array1<f64> = cp.root.dot(&beta_block);
            let quad_k = r_beta.dot(&r_beta);
            penalty_val += 0.5 * lambdas[k] * quad_k;

            // Accumulate S(ρ)β for β-gradient — block-local
            for a in 0..cp.block_dim() {
                let val: f64 = (0..cp.rank())
                    .map(|row| cp.root[[row, a]] * r_beta[row])
                    .sum();
                s_beta[r.start + a] += lambdas[k] * val;
            }

            // ρ_k gradient from penalty
            grad_rho[k] = -0.5 * lambdas[k] * quad_k;
        }

        // ---- Structural penalty log-determinant: +0.5 log|S(ρ)|₊ and ρ-derivatives ----
        let (log_det_s, logdet_grad) = if self.penalty_canonical.is_empty() {
            (0.0, Array1::zeros(n_rho))
        } else {
            match PenaltyPseudologdet::from_penalties(
                &self.penalty_canonical,
                lambdas.as_slice().unwrap_or(&[]),
                0.0,
                n_beta,
            ) {
                Ok(pld) => {
                    let (det1, _) = pld.rho_derivatives_from_penalties(
                        &self.penalty_canonical,
                        lambdas.as_slice().unwrap_or(&[]),
                    );
                    (pld.value(), det1)
                }
                Err(err) => {
                    log::warn!(
                        "[Joint HMC] structural penalty logdet became invalid at the current state: {}",
                        err
                    );
                    return (f64::NEG_INFINITY, Array1::zeros(n_beta + n_rho));
                }
            }
        };

        for k in 0..n_rho {
            grad_rho[k] += 0.5 * logdet_grad[k];
        }

        // ---- Prior on ρ ----
        let mut rho_prior = 0.0;
        match self.rho_prior {
            RhoPrior::Flat => {}
            RhoPrior::Normal { mean, sd } => {
                let inv_var = 1.0 / (sd * sd);
                for k in 0..n_rho {
                    let d = rho[k] - mean;
                    rho_prior -= 0.5 * inv_var * d * d;
                    grad_rho[k] -= inv_var * d;
                }
            }
        }

        // ---- Assemble ----
        let logp = ll + firth_logdet - penalty_val + 0.5 * log_det_s + rho_prior;

        // β-gradient in original space: ∇_β ℓ - S(ρ)β
        let grad_beta = &grad_ll_beta - &s_beta;

        // Chain rule to whitened space: ∇_z = L' ∇_β
        let grad_z = self.chol_t.dot(&grad_beta);

        // Combined gradient: [∇_z; ∇_ρ]
        let mut grad = Array1::<f64>::zeros(n_beta + n_rho);
        grad.slice_mut(ndarray::s![..n_beta]).assign(&grad_z);
        grad.slice_mut(ndarray::s![n_beta..]).assign(&grad_rho);

        (logp, grad)
    }
}

impl HamiltonianTarget<Array1<f64>> for JointBetaRhoPosterior {
    fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
        let (logp, g) = self.compute_joint_logp_and_grad(position);
        grad.assign(&g);
        logp
    }
}

/// Inputs for joint (β, ρ) sampling.
pub struct JointBetaRhoInputs<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub weights: ArrayView1<'a, f64>,
    pub likelihood_family: LikelihoodFamily,
    pub inverse_link: InverseLink,
    pub gamma_shape: Option<f64>,
    pub mode: ArrayView1<'a, f64>,
    pub hessian: ArrayView2<'a, f64>,
    pub penalty_roots: Vec<CanonicalPenalty>,
    pub rho_mode: ArrayView1<'a, f64>,
    pub rho_prior: RhoPrior,
    pub firth_bias_reduction: bool,
    /// Max posterior skewness that triggered this sampling
    pub trigger_skewness: f64,
}

/// Run joint (β, ρ) NUTS sampling.
///
/// This is the automatic fallback when the Laplace approximation has high
/// skewness. It samples from the true joint posterior, completely bypassing
/// the Laplace approximation for smoothing parameter selection.
pub fn run_joint_beta_rho_sampling(
    inputs: &JointBetaRhoInputs<'_>,
    config: &NutsConfig,
) -> Result<JointBetaRhoResult, String> {
    validate_firth_likelihood_support(inputs.likelihood_family, inputs.firth_bias_reduction)?;
    let n_beta = inputs.mode.len();
    let n_rho = inputs.penalty_roots.len();
    let total_dim = n_beta + n_rho;

    log::info!(
        "[Joint HMC] Sampling (β, ρ) jointly: {} β-params + {} ρ-params = {} total (triggered by skewness {:.3})",
        n_beta,
        n_rho,
        total_dim,
        inputs.trigger_skewness,
    );

    let target = JointBetaRhoPosterior::new(
        inputs.x,
        inputs.y,
        inputs.weights,
        inputs.mode,
        inputs.hessian,
        inputs.penalty_roots.clone(),
        inputs.rho_mode,
        inputs.likelihood_family,
        inputs.inverse_link.clone(),
        inputs.gamma_shape,
        inputs.rho_prior.clone(),
        inputs.firth_bias_reduction,
    )?;

    let chol = target.chol.clone();
    let mode_arr = target.data.mode.clone();
    let rho_mode = target.rho_mode.clone();

    // Initialize chains: z_β at 0 (= mode), ρ at rho_mode
    let mut rng = StdRng::seed_from_u64(config.seed);
    let initial_positions: Vec<Array1<f64>> = (0..config.n_chains)
        .map(|_| {
            let mut pos = Array1::<f64>::zeros(total_dim);
            // Small jitter for β (whitened space)
            for j in 0..n_beta {
                pos[j] = sample_standard_normal(&mut rng) * 0.1;
            }
            // Small jitter for ρ around mode
            for k in 0..n_rho {
                pos[n_beta + k] = rho_mode[k] + sample_standard_normal(&mut rng) * 0.2;
            }
            pos
        })
        .collect();

    // Auto-select dense mass matrix when dimension is small enough.
    // The joint (β, ρ) posterior has strong cross-block correlations —
    // changing ρ shifts the entire β posterior through the penalty —
    // so dense adaptation is critical for efficient sampling.
    let mass_cfg = robust_mass_matrix_config(total_dim, config.nwarmup);

    let mut sampler = GenericNUTS::new_with_mass_matrix(
        target,
        initial_positions,
        config.target_accept,
        mass_cfg,
    );

    let (samples_array, run_stats) = sampler
        .run_progress(config.n_samples, config.nwarmup)
        .map_err(|e| format!("Joint (β,ρ) NUTS sampling failed: {e}"))?;
    log::info!("[Joint HMC] Sampling complete: {}", run_stats);

    // Unpack samples
    let shape = samples_array.shape();
    let n_chains = shape[0];
    let n_samples_out = shape[1];
    let total_samples = n_chains * n_samples_out;

    let mut beta_samples = Array2::<f64>::zeros((total_samples, n_beta));
    let mut rho_samples = Array2::<f64>::zeros((total_samples, n_rho));

    for chain in 0..n_chains {
        for sample_i in 0..n_samples_out {
            let sample_idx = chain * n_samples_out + sample_i;
            let zview = samples_array.slice(ndarray::s![chain, sample_i, ..]);

            // Un-whiten β: β = μ + L z
            let z_beta = zview.slice(ndarray::s![..n_beta]).to_owned();
            let beta = mode_arr.as_ref() + &chol.dot(&z_beta);
            beta_samples.row_mut(sample_idx).assign(&beta);

            // ρ is stored directly
            let rho_slice = zview.slice(ndarray::s![n_beta..]);
            rho_samples.row_mut(sample_idx).assign(&rho_slice);
        }
    }

    let beta_mean = beta_samples
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(n_beta));
    let rho_mean = rho_samples
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(n_rho));

    let (rhat, ess) = compute_split_rhat_and_ess(&samples_array);

    let converged = rhat < 1.1 && ess > 50.0;
    if !converged {
        log::warn!(
            "[Joint HMC] Convergence warning: R-hat={:.3}, ESS={:.1}",
            rhat,
            ess,
        );
    }

    Ok(JointBetaRhoResult {
        beta_samples,
        rho_samples,
        beta_mean,
        rho_mean,
        rhat,
        ess,
        converged,
        trigger_skewness: inputs.trigger_skewness,
    })
}

// ============================================================================
// Survival Model HMC Support
// ============================================================================

mod survival_hmc {
    use super::*;
    use crate::survival::{
        MonotonicityPenalty, PenaltyBlocks, SurvivalEngineInputs, SurvivalSpec,
        WorkingModelSurvival,
    };

    /// Shared data for survival NUTS posterior (wrapped in Arc to prevent cloning).
    #[derive(Clone)]
    struct SharedSurvivalData {
        /// Exact survival model in original spline coordinates.
        base_model: Arc<WorkingModelSurvival>,
        /// MAP estimate in coefficient coordinates.
        mode: Arc<Array1<f64>>,
    }

    /// Whitened log-posterior target for survival models with analytical gradients.
    #[derive(Clone)]
    pub struct SurvivalPosterior {
        /// Shared read-only data (Arc prevents duplication)
        data: SharedSurvivalData,
        /// Transform: L where L L^T = H^{-1}
        chol: Array2<f64>,
        /// L^T for gradient chain rule: ∇z = L^T @ ∇_β
        chol_t: Array2<f64>,
    }

    impl SurvivalPosterior {
        /// Creates a new survival posterior target.
        pub fn new(
            age_entry: ArrayView1<'_, f64>,
            age_exit: ArrayView1<'_, f64>,
            event_target: ArrayView1<'_, u8>,
            event_competing: ArrayView1<'_, u8>,
            sampleweight: ArrayView1<'_, f64>,
            x_entry: ArrayView2<'_, f64>,
            x_exit: ArrayView2<'_, f64>,
            x_derivative: ArrayView2<'_, f64>,
            offset_eta_entry: Option<ArrayView1<'_, f64>>,
            offset_eta_exit: Option<ArrayView1<'_, f64>>,
            offset_derivative_exit: Option<ArrayView1<'_, f64>>,
            penalties: PenaltyBlocks,
            monotonicity: MonotonicityPenalty,
            spec: SurvivalSpec,
            structurally_monotonic: bool,
            structural_time_columns: usize,
            mode: ArrayView1<f64>,
            hessian: ArrayView2<f64>,
        ) -> Result<Self, String> {
            let n = age_entry.len();
            let off_eta_entry = offset_eta_entry
                .map(|v| v.to_owned())
                .unwrap_or_else(|| Array1::zeros(n));
            let off_eta_exit = offset_eta_exit
                .map(|v| v.to_owned())
                .unwrap_or_else(|| Array1::zeros(n));
            let off_deriv_exit = offset_derivative_exit
                .map(|v| v.to_owned())
                .unwrap_or_else(|| Array1::zeros(n));

            let mut base_model = WorkingModelSurvival::from_engine_inputswith_offsets(
                SurvivalEngineInputs {
                    age_entry,
                    age_exit,
                    event_target,
                    event_competing,
                    sampleweight,
                    x_entry,
                    x_exit,
                    x_derivative,
                    monotonicity_constraint_rows: None,
                    monotonicity_constraint_offsets: None,
                },
                Some(crate::survival::SurvivalBaselineOffsets {
                    eta_entry: off_eta_entry.view(),
                    eta_exit: off_eta_exit.view(),
                    derivative_exit: off_deriv_exit.view(),
                }),
                penalties,
                monotonicity,
                spec,
            )
            .map_err(|e| format!("Survival state construction failed: {:?}", e))?;
            if structurally_monotonic {
                base_model
                    .set_structural_monotonicity(true, structural_time_columns)
                    .map_err(|e| {
                        format!("Failed to enable structural monotonicity in survival HMC: {e}")
                    })?;
            }

            let sampler_mode = mode.to_owned();
            let dim = sampler_mode.len();

            // Compute whitening transform via Cholesky of Hessian
            let chol_factor = hessian
                .to_owned()
                .cholesky(Side::Lower)
                .map_err(|e| format!("Hessian Cholesky decomposition failed: {:?}", e))?;
            let l_h = chol_factor.lower_triangular();
            let chol = solve_upper_triangular_transpose(&l_h, dim);
            let chol_t = chol.t().to_owned();

            let data = SharedSurvivalData {
                base_model: Arc::new(base_model),
                mode: Arc::new(sampler_mode),
            };

            Ok(Self { data, chol, chol_t })
        }

        /// Compute log-posterior and gradient analytically.
        fn compute_logp_and_grad(&self, z: &Array1<f64>) -> Result<(f64, Array1<f64>), String> {
            let sampler_position = self.data.mode.as_ref() + &self.chol.dot(z);
            let state = self
                .data
                .base_model
                .update_state(&sampler_position)
                .map_err(|e| format!("Survival state update failed: {:?}", e))?;
            let logp = state.log_likelihood - 0.5 * state.penalty_term;
            let grad_beta = state.gradient.mapv(|g| -g);
            let gradz = self.chol_t.dot(&grad_beta);
            Ok((logp, gradz))
        }

        /// Get the Cholesky factor L for un-whitening samples
        pub fn chol(&self) -> &Array2<f64> {
            &self.chol
        }

        /// Get the mode
        pub fn mode(&self) -> &Array1<f64> {
            &self.data.mode
        }
    }

    impl HamiltonianTarget<Array1<f64>> for SurvivalPosterior {
        fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
            match self.compute_logp_and_grad(position) {
                Ok((logp, gradz)) => {
                    grad.assign(&gradz);
                    logp
                }
                Err(e) => {
                    log::warn!("Survival posterior evaluation failed: {}", e);
                    grad.fill(0.0);
                    f64::NEG_INFINITY
                }
            }
        }
    }

    /// Runs NUTS sampling for survival models with whitened parameter space.
    pub(crate) fn run_survival_nuts_sampling(
        age_entry: ArrayView1<'_, f64>,
        age_exit: ArrayView1<'_, f64>,
        event_target: ArrayView1<'_, u8>,
        event_competing: ArrayView1<'_, u8>,
        sampleweight: ArrayView1<'_, f64>,
        x_entry: ArrayView2<'_, f64>,
        x_exit: ArrayView2<'_, f64>,
        x_derivative: ArrayView2<'_, f64>,
        eta_offset_entry: Option<ArrayView1<'_, f64>>,
        eta_offset_exit: Option<ArrayView1<'_, f64>>,
        derivative_offset_exit: Option<ArrayView1<'_, f64>>,
        penalties: PenaltyBlocks,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
        structurally_monotonic: bool,
        structural_time_columns: usize,
        mode: ArrayView1<f64>,
        hessian: ArrayView2<f64>,
        config: &NutsConfig,
    ) -> Result<NutsResult, String> {
        // Create posterior target
        let target = SurvivalPosterior::new(
            age_entry,
            age_exit,
            event_target,
            event_competing,
            sampleweight,
            x_entry,
            x_exit,
            x_derivative,
            eta_offset_entry,
            eta_offset_exit,
            derivative_offset_exit,
            penalties,
            monotonicity,
            spec,
            structurally_monotonic,
            structural_time_columns,
            mode,
            hessian,
        )?;

        // Get Cholesky factor for un-whitening samples later
        let chol = target.chol().clone();
        let mode_arr = target.mode().clone();
        let dim = mode_arr.len();

        // Initialize chains at z=0 with small jitter
        let mut rng = StdRng::seed_from_u64(config.seed);
        let initial_positions: Vec<Array1<f64>> = (0..config.n_chains)
            .map(|_| {
                Array1::from_shape_fn(dim, |_| {
                    let u1: f64 = rng.random::<f64>().max(1e-10); // Prevent ln(0) = -inf
                    let u2: f64 = rng.random();
                    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    z * 0.1
                })
            })
            .collect();

        // Create GenericNUTS sampler
        let mass_cfg = robust_survival_mass_matrix_config(dim, config.nwarmup);
        let mut sampler = GenericNUTS::new_with_mass_matrix(
            target,
            initial_positions,
            config.target_accept,
            mass_cfg,
        );

        // Run sampling with progress bar
        let (samples_array, run_stats) = sampler
            .run_progress(config.n_samples, config.nwarmup)
            .map_err(|e| format!("NUTS sampling failed: {}", e))?;

        log::info!("Survival NUTS sampling complete: {}", run_stats);

        // Convert samples from whitened space back to original space
        let shape = samples_array.shape();
        let n_chains = shape[0];
        let n_samples_out = shape[1];
        let total_samples = n_chains * n_samples_out;

        let mut samples = Array2::<f64>::zeros((total_samples, dim));
        let mut z_buffer = Array1::<f64>::zeros(dim);
        for chain in 0..n_chains {
            for sample_i in 0..n_samples_out {
                let zview = samples_array.slice(ndarray::s![chain, sample_i, ..]);
                z_buffer.assign(&zview);

                let beta = &mode_arr + &chol.dot(&z_buffer);

                let sample_idx = chain * n_samples_out + sample_i;
                samples.row_mut(sample_idx).assign(&beta);
            }
        }

        // Compute statistics
        let posterior_mean = samples
            .mean_axis(Axis(0))
            .unwrap_or_else(|| Array1::zeros(dim));
        let posterior_std = samples.std_axis(Axis(0), 0.0);
        let (rhat, ess) = compute_split_rhat_and_ess(&samples_array);
        let converged = rhat < 1.1;

        Ok(NutsResult {
            samples,
            posterior_mean,
            posterior_std,
            rhat,
            ess,
            converged,
        })
    }
}

/// Engine-facing flattened survival NUTS entrypoint.
pub fn run_survival_nuts_sampling_flattened<'a>(
    flat: SurvivalFlatInputs<'a>,
    penalties: crate::survival::PenaltyBlocks,
    monotonicity: crate::survival::MonotonicityPenalty,
    spec: crate::survival::SurvivalSpec,
    structurally_monotonic: bool,
    structural_time_columns: usize,
    mode: ArrayView1<'a, f64>,
    hessian: ArrayView2<'a, f64>,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    run_nuts_sampling_flattened_family(
        LikelihoodFamily::RoystonParmar,
        FamilyNutsInputs::Survival(Box::new(SurvivalNutsInputs {
            flat,
            penalties,
            monotonicity,
            spec,
            structurally_monotonic,
            structural_time_columns,
            mode,
            hessian,
        })),
        config,
    )
}
