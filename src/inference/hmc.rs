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

use crate::faer_ndarray::{FaerCholesky, fast_ata_into, fast_atv};
use crate::solver::reml::unified::compute_block_penalty_logdet_derivs;
use crate::types::LikelihoodFamily;
use crate::visualizer::VisualizerSession;
use faer::Side;
use general_mcmc::generic_hmc::HamiltonianTarget;
use general_mcmc::generic_nuts::{GenericNUTS, MassMatrixAdaptation, NUTSMassMatrixConfig};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use polya_gamma::PolyaGamma;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand08::{SeedableRng as SeedableRng08, rngs::StdRng as StdRng08};
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
    /// Prior weights [n_samples]
    weights: Arc<Array1<f64>>,
    /// Combined penalty matrix S [dim, dim]
    penalty: Arc<Array2<f64>>,
    /// MAP estimate (mode) μ [dim]
    mode: Arc<Array1<f64>>,
    /// Number of samples
    n_samples: usize,
    /// Number of coefficients
    dim: usize,
}

/// Whitened log-posterior target with analytical gradients.
///
/// Uses Arc for shared data to prevent memory explosion when cloned for chains.
/// Uses faer for numerically stable Cholesky decomposition.
#[derive(Clone)]
/// Family mode for NUTS log-likelihood computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NutsFamily {
    Gaussian,
    BinomialLogit,
    PoissonLog,
    GammaLog,
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
    /// * `weights` - Prior weights [n_samples]
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
            n_samples,
            dim,
        };

        Ok(Self {
            data,
            chol,
            chol_t,
            nuts_family,
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
        let (ll, grad_ll_beta) = self.family_logp_and_grad(&eta);

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

        let logp = ll - penalty;

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
        NutsFamily::Gaussian => gaussian_logp_and_grad(data, eta),
        NutsFamily::PoissonLog => poisson_log_logp_and_grad(data, eta),
        NutsFamily::GammaLog => gamma_log_logp_and_grad(data, eta),
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

/// Gamma(log) log-likelihood and gradient (shape=1).
///
/// log p(y|η) = −η − y·exp(−η), gradient = X'(w ⊙ (y/μ − 1))
fn gamma_log_logp_and_grad(data: &SharedData, eta: &Array1<f64>) -> (f64, Array1<f64>) {
    let n = data.n_samples;
    let mut ll = 0.0;
    let mut residual = Array1::<f64>::zeros(n);

    for i in 0..n {
        let eta_i = eta[i].clamp(-700.0, 700.0);
        let mu_i = eta_i.exp();
        let y_i = data.y[i];
        let w_i = data.weights[i];
        ll += w_i * (-eta_i - y_i / mu_i);
        residual[i] = w_i * (y_i / mu_i - 1.0);
    }

    let grad_ll = fast_atv(&data.x, &residual);
    (ll, grad_ll)
}

#[cfg(test)]
mod tests {
    use super::{
        FamilyNutsInputs, GlmFlatInputs, JointLinkPosterior, JointSplineArtifacts, NutsConfig,
        NutsPosterior, run_logit_polya_gamma_gibbs, run_nuts_sampling_flattened_family,
    };
    use crate::basis::{BasisOptions, Dense, KnotSource, create_basis};
    use crate::survival::{MonotonicityPenalty, PenaltyBlocks, SurvivalSpec};
    use crate::types::LikelihoodFamily;
    use general_mcmc::generic_hmc::HamiltonianTarget;
    use ndarray::{Array1, Array2, array};

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
            true,
        )
        .expect("posterior");

        let z = array![0.15, -0.35];
        let (logp, grad) = posterior.compute_logp_and_grad_nd(&z);
        let _ = logp;

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
    fn joint_link_usesc1_extension_outside_knot_range() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let degree = 3usize;
        let n_raw = knots.len() - degree - 1;
        let z = Array2::<f64>::eye(n_raw);
        let spline = JointSplineArtifacts {
            knot_range: (0.0, 1.0),
            knot_vector: knots.clone(),
            link_transform: z.clone(),
            degree,
        };
        let x = array![[1.0], [1.0], [1.0]];
        let y = array![0.0, 1.0, 0.0];
        let w = array![1.0, 1.0, 1.0];
        let penalty_base = array![[1.0]];
        let penalty_link = Array2::<f64>::eye(n_raw);
        let mode_beta = array![0.0];
        let mode_theta = array![0.1, -0.2, 0.3, -0.1, 0.2];
        let hessian = Array2::<f64>::eye(1 + n_raw);
        let posterior = JointLinkPosterior::new(
            x.view(),
            y.view(),
            w.view(),
            penalty_base.view(),
            penalty_link.view(),
            mode_beta.view(),
            mode_theta.view(),
            hessian.view(),
            spline,
            true,
            1.0,
        )
        .expect("joint posterior");

        // Outside [0,1] for first and third entries; middle is interior.
        let u = array![-0.25, 0.5, 1.2];
        let theta = mode_theta.clone();
        let (b_eval, eta) = posterior.evaluate_link(&u, &theta);
        let _ = eta;

        let rw = 1.0;
        let z_raw = u.mapv(|ui| ui / rw);
        let z_c = z_raw.mapv(|zi| zi.clamp(0.0, 1.0));
        let (b_arc, _) = create_basis::<Dense>(
            z_c.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::value(),
        )
        .expect("basis");
        let (bp_arc, _) = create_basis::<Dense>(
            z_c.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::first_derivative(),
        )
        .expect("basis derivative");
        let mut b_expected = b_arc.as_ref().clone();
        let bp = bp_arc.as_ref();
        for i in 0..u.len() {
            let dz = z_raw[i] - z_c[i];
            if dz.abs() <= 1e-12 {
                continue;
            }
            for j in 0..b_expected.ncols() {
                b_expected[[i, j]] += dz * bp[[i, j]];
            }
        }

        for i in 0..u.len() {
            for j in 0..n_raw {
                assert!(
                    (b_eval[[i, j]] - b_expected[[i, j]]).abs() < 1e-10,
                    "C1 extension mismatch at ({}, {}): eval={}, expected={}",
                    i,
                    j,
                    b_eval[[i, j]],
                    b_expected[[i, j]]
                );
            }
        }
    }

    #[test]
    fn joint_link_g_prime_uses_extension_derivative_outside_range() {
        let knots = array![0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0];
        let degree = 3usize;
        let n_raw = knots.len() - degree - 1;
        let z = Array2::<f64>::eye(n_raw);
        let spline = JointSplineArtifacts {
            knot_range: (0.0, 1.0),
            knot_vector: knots.clone(),
            link_transform: z.clone(),
            degree,
        };
        let x = array![[1.0], [1.0], [1.0]];
        let y = array![0.0, 1.0, 0.0];
        let w = array![1.0, 1.0, 1.0];
        let penalty_base = array![[1.0]];
        let penalty_link = Array2::<f64>::eye(n_raw);
        let mode_beta = array![0.0];
        let mode_theta = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let hessian = Array2::<f64>::eye(1 + n_raw);
        let posterior = JointLinkPosterior::new(
            x.view(),
            y.view(),
            w.view(),
            penalty_base.view(),
            penalty_link.view(),
            mode_beta.view(),
            mode_theta.view(),
            hessian.view(),
            spline,
            true,
            1.0,
        )
        .expect("joint posterior");

        let u = array![-0.25, 0.5, 1.2];
        let g = posterior.compute_g_prime(&u, &mode_theta);
        let (z_raw, z_c, rw) = posterior.standardized_z(&u);
        let _ = z_raw;
        let (bp_arc, _) = create_basis::<Dense>(
            z_c.view(),
            KnotSource::Provided(knots.view()),
            degree,
            BasisOptions::first_derivative(),
        )
        .expect("basis derivative");
        let bp = bp_arc.as_ref();
        for i in 0..u.len() {
            let dwdz: f64 = (0..n_raw).map(|j| bp[[i, j]] * mode_theta[j]).sum();
            let expected = 1.0 + dwdz / rw;
            assert!(
                (g[i] - expected).abs() < 1e-10,
                "g' mismatch at {}: got {}, expected {}",
                i,
                g[i],
                expected
            );
        }
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
                firth_bias_reduction: false,
            }),
            &cfg,
        )
        .expect("dispatch should use PG Gibbs and not require Hessian factorization");
        assert_eq!(out.samples.nrows(), cfg.n_samples * cfg.n_chains);
        assert!(out.samples.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn family_dispatch_rejects_probit_for_nuts_entrypoint() {
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

        let err = run_nuts_sampling_flattened_family(
            LikelihoodFamily::BinomialProbit,
            FamilyNutsInputs::Glm(GlmFlatInputs {
                x: x.view(),
                y: y.view(),
                weights: w.view(),
                penalty_matrix: penalty.view(),
                mode: mode.view(),
                hessian: non_spdhessian.view(),
                firth_bias_reduction: false,
            }),
            &cfg,
        )
        .expect_err("probit family should not route through logit PG/NUTS dispatch");

        assert!(
            err.contains("BinomialProbit NUTS is not implemented yet"),
            "unexpected error: {err}"
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
        let _ = HamiltonianTarget::logp_and_grad(&posterior_linear, &z, &mut grad_linear);

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
        let _ = HamiltonianTarget::logp_and_grad(&posterior_struct, &z, &mut grad_struct);

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
        let mut pg_rng = StdRng08::seed_from_u64(
            config.seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul((chain as u64) + 1)),
        );
        let pg = PolyaGamma::new(1.0);
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
        let mut pg_rng = StdRng08::seed_from_u64(
            config.seed ^ (0x9E37_79B9_7F4A_7C15u64.wrapping_mul((chain as u64) + 1)),
        );
        let pg = PolyaGamma::new(1.0);
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
/// * `weights` - Prior weights [n_samples]
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
    firth_bias_reduction: bool,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    if nuts_family == NutsFamily::BinomialLogit && firth_bias_reduction {
        return Err(
            "NUTS with Firth bias reduction is disabled: posterior target mismatch. Refit with firth_bias_reduction=false for consistent HMC uncertainty."
                .to_string(),
        );
    }
    let dim = mode.len();

    // Create posterior target with analytical gradients.
    let target = NutsPosterior::new(x, y, weights, penalty_matrix, mode, hessian, nuts_family)?;

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

/// Flattened numeric inputs for Gaussian/Logit NUTS sampling.
pub struct GlmFlatInputs<'a> {
    pub x: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub weights: ArrayView1<'a, f64>,
    pub penalty_matrix: ArrayView2<'a, f64>,
    pub mode: ArrayView1<'a, f64>,
    pub hessian: ArrayView2<'a, f64>,
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
    match (family, inputs) {
        (LikelihoodFamily::GaussianIdentity, FamilyNutsInputs::Glm(glm)) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            NutsFamily::Gaussian,
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
                    glm.firth_bias_reduction,
                    config,
                )
            }
        }
        (LikelihoodFamily::BinomialProbit, FamilyNutsInputs::Glm(_)) => Err(
            "BinomialProbit NUTS is not implemented yet; use fit_gam/predict_gam for probit models"
                .to_string(),
        ),
        (LikelihoodFamily::BinomialCLogLog, FamilyNutsInputs::Glm(_)) => Err(
            "BinomialCLogLog NUTS is not implemented yet; use fit_gam/predict_gam for cloglog models"
                .to_string(),
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
            false,
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
            false,
            config,
        ),
    }
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
//   log p(β, ρ | y) = ℓ(y|β) - 0.5 β'S(ρ)β + 0.5 log|S(ρ)|_+ + log p(ρ) + const
//
// Gradients:
//   ∇_β: ∇_β ℓ - S(ρ) β
//   ∂/∂ρ_k: -0.5 λ_k β'S_k β + 0.5 tr(S_+⁻¹ A_k) + ∂log p(ρ)/∂ρ_k
//
// This completely avoids the Laplace approximation — no log|H| term needed.
// The sampler explores the full posterior including smoothing parameters.

/// Skewness diagnostic for the Laplace approximation.
///
/// Computes per-coefficient posterior skewness s_j = (H⁻¹)_{jj}^{3/2} · T_j
/// where T_j = Σ_i c_i x_{ij}³ is the third-derivative projection.
///
/// Returns (max_abs_skewness, per_coefficient_skewness).
pub fn laplace_skewness_diagnostic(
    h_inv_diag: &Array1<f64>,
    third_deriv: &Array1<f64>,
) -> (f64, Array1<f64>) {
    let p = h_inv_diag.len().min(third_deriv.len());
    let mut skewness = Array1::<f64>::zeros(p);
    let mut max_abs = 0.0_f64;
    for j in 0..p {
        let s = h_inv_diag[j];
        let s_j = s.sqrt() * s * third_deriv[j]; // s^{3/2} * T_j
        skewness[j] = if s_j.is_finite() { s_j } else { 0.0 };
        max_abs = max_abs.max(skewness[j].abs());
    }
    (max_abs, skewness)
}

/// Threshold for triggering joint (β, ρ) HMC.
/// When max|skewness_j| exceeds this, the Laplace approximation is unreliable
/// and NUTS on the joint space is used to refine smoothing parameters.
pub const SKEWNESS_HMC_THRESHOLD: f64 = 0.5;

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
    nuts_family: NutsFamily,
    /// Dimension of β
    n_beta: usize,
    /// Dimension of ρ
    n_rho: usize,
    /// Penalty root matrices R_k where S_k = R_k' R_k (in transformed basis)
    penalty_roots: Vec<Array2<f64>>,
    /// Precomputed penalty matrices S_k = R_k' R_k (avoids recomputation each gradient eval)
    penalty_matrices: Vec<Array2<f64>>,
    /// Prior std dev on ρ (weakly informative Gaussian)
    rho_prior_sd: f64,
    /// LAML-converged ρ (used as centering for ρ prior)
    rho_mode: Array1<f64>,
}

impl JointBetaRhoPosterior {
    fn new(
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        weights: ArrayView1<f64>,
        mode: ArrayView1<f64>,
        hessian: ArrayView2<f64>,
        penalty_roots: Vec<Array2<f64>>,
        rho_mode: ArrayView1<f64>,
        nuts_family: NutsFamily,
    ) -> Result<Self, String> {
        let n_samples = x.nrows();
        let n_beta = x.ncols();
        let n_rho = penalty_roots.len();

        if rho_mode.len() != n_rho {
            return Err(format!(
                "rho_mode length {} != penalty_roots count {}",
                rho_mode.len(),
                n_rho
            ));
        }

        // Cholesky of H for β-whitening (same as NutsPosterior)
        let hessian_owned = hessian.to_owned();
        let chol_factor = hessian_owned
            .cholesky(Side::Lower)
            .map_err(|e| format!("Joint HMC: Hessian Cholesky failed: {:?}", e))?;
        let l_h = chol_factor.lower_triangular();
        let chol = solve_upper_triangular_transpose(&l_h, n_beta);
        let chol_t = chol.t().to_owned();

        // Precompute S_k = R_k' R_k once; reused in every gradient evaluation.
        let penalty_matrices: Vec<Array2<f64>> = penalty_roots
            .iter()
            .map(|r_k| r_k.t().dot(r_k))
            .collect();

        // Build combined penalty at the LAML mode (for SharedData)
        let lambdas_mode: Array1<f64> = rho_mode.mapv(f64::exp);
        let mut s_combined = Array2::<f64>::zeros((n_beta, n_beta));
        for (k, s_k) in penalty_matrices.iter().enumerate() {
            s_combined.scaled_add(lambdas_mode[k], s_k);
        }

        let data = SharedData {
            x: Arc::new(x.to_owned()),
            y: Arc::new(y.to_owned()),
            weights: Arc::new(weights.to_owned()),
            penalty: Arc::new(s_combined),
            mode: Arc::new(mode.to_owned()),
            n_samples,
            dim: n_beta,
        };

        // Weakly informative prior on ρ: N(rho_mode, 3²)
        // This is broad enough to let the data speak but prevents
        // runaway smoothing parameters.
        let rho_prior_sd = 3.0;

        Ok(Self {
            data,
            chol,
            chol_t,
            nuts_family,
            n_beta,
            n_rho,
            penalty_roots,
            penalty_matrices,
            rho_prior_sd,
            rho_mode: rho_mode.to_owned(),
        })
    }

    /// Compute the joint log-posterior and gradient.
    ///
    /// The joint log-posterior is:
    ///   log p(β, ρ | y) = ℓ(y|β) − ½β'S(ρ)β + ½ log|S(ρ)|₊ + log p(ρ) + const
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
        // Delegates to shared family helpers (same code as NutsPosterior).
        let (ll, grad_ll_beta) = nuts_family_logp_and_grad(self.nuts_family, &self.data, &eta);

        // ---- Penalty: -0.5 β'S(ρ)β ----
        // S(ρ) = Σ_k λ_k S_k where S_k = R_k'R_k (precomputed in penalty_matrices).
        // Uses penalty_roots for the efficient ||R_k β||² form.
        let mut penalty_val = 0.0;
        let mut s_beta = Array1::<f64>::zeros(n_beta);
        let mut grad_rho = Array1::<f64>::zeros(n_rho);

        for (k, r_k) in self.penalty_roots.iter().enumerate() {
            let r_beta = r_k.dot(&beta);
            let quad_k = r_beta.dot(&r_beta); // β'S_k β
            penalty_val += 0.5 * lambdas[k] * quad_k;

            // Accumulate S(ρ)β for β-gradient
            let s_k_beta = r_k.t().dot(&r_beta);
            s_beta.scaled_add(lambdas[k], &s_k_beta);

            // ρ_k gradient from penalty: ∂/∂ρ_k [-0.5 λ_k β'S_k β] = -0.5 λ_k β'S_k β
            // (since ∂λ_k/∂ρ_k = λ_k for ρ = log λ)
            grad_rho[k] = -0.5 * lambdas[k] * quad_k;
        }

        // ---- Structural penalty log-determinant: +0.5 log|S(ρ)|₊ and ρ-derivatives ----
        // Delegates to the shared utility from the unified REML module, which
        // uses eigendecomposition with adaptive tolerance and computes both
        // log|S|₊ and ∂/∂ρ_k log|S|₊ = tr(S₊⁻¹ A_k) from the same decomposition.
        //
        // All penalty matrices live in a single block (no multi-block structure).
        let penalty_logdet = compute_block_penalty_logdet_derivs(
            &[rho.clone()],
            &[self.penalty_matrices.as_slice()],
            0.0,
        );
        let (log_det_s, logdet_grad) = match penalty_logdet {
            Ok(pld) => (pld.value, pld.first),
            Err(_) => (0.0, Array1::zeros(n_rho)),
        };

        // Add logdet ρ-gradient: +0.5 ∂/∂ρ_k log|S|₊
        for k in 0..n_rho {
            grad_rho[k] += 0.5 * logdet_grad[k];
        }

        // ---- Prior on ρ: N(rho_mode, σ²I) ----
        let mut rho_prior = 0.0;
        let inv_var = 1.0 / (self.rho_prior_sd * self.rho_prior_sd);
        for k in 0..n_rho {
            let d = rho[k] - self.rho_mode[k];
            rho_prior -= 0.5 * inv_var * d * d;
            grad_rho[k] -= inv_var * d;
        }

        // ---- Assemble ----
        let logp = ll - penalty_val + 0.5 * log_det_s + rho_prior;

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
    pub mode: ArrayView1<'a, f64>,
    pub hessian: ArrayView2<'a, f64>,
    pub penalty_roots: Vec<Array2<f64>>,
    pub rho_mode: ArrayView1<'a, f64>,
    pub nuts_family: NutsFamily,
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
        inputs.nuts_family,
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
            let logp = -0.5 * (state.deviance + state.penalty_term);
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

// ============================================================================
// Joint Link Model HMC Support
// ============================================================================

/// Fixed spline artifacts for joint link (frozen from REML fit).
#[derive(Clone)]
pub struct JointSplineArtifacts {
    /// Knot range (min, max) from training
    pub knot_range: (f64, f64),
    /// Knot vector for B-splines
    pub knot_vector: Array1<f64>,
    /// Constraint transform Z (raw basis → constrained basis)
    pub link_transform: Array2<f64>,
    /// B-spline degree
    pub degree: usize,
}

/// Whitened log-posterior target for joint (β, θ) with analytical gradients.
#[derive(Clone)]
pub struct JointLinkPosterior {
    x: Arc<Array2<f64>>,
    y: Arc<Array1<f64>>,
    weights: Arc<Array1<f64>>,
    penalty_base: Arc<Array2<f64>>,
    penalty_link: Arc<Array2<f64>>,
    mode_beta: Arc<Array1<f64>>,
    mode_theta: Arc<Array1<f64>>,
    spline: JointSplineArtifacts,
    chol: Array2<f64>,
    chol_t: Array2<f64>,
    p_base: usize,
    p_link: usize,
    n_samples: usize,
    is_logit: bool, // true=logit, false=identity
    scale: f64,     // dispersion parameter for identity link
}

impl JointLinkPosterior {
    #[inline]
    fn standardized_z(&self, u: &Array1<f64>) -> (Array1<f64>, Array1<f64>, f64) {
        let (min_u, max_u) = self.spline.knot_range;
        let rw = (max_u - min_u).max(1e-6);
        let z_raw: Array1<f64> = u.mapv(|v| (v - min_u) / rw);
        let z_c: Array1<f64> = z_raw.mapv(|z| z.clamp(0.0, 1.0));
        (z_raw, z_c, rw)
    }

    /// Creates a new joint posterior target.
    /// `is_logit`: true for Bernoulli-logit, false for Gaussian-identity
    /// `scale`: dispersion parameter (ignored if is_logit=true)
    pub fn new(
        x: ArrayView2<f64>,
        y: ArrayView1<f64>,
        weights: ArrayView1<f64>,
        penalty_base: ArrayView2<f64>,
        penalty_link: ArrayView2<f64>,
        mode_beta: ArrayView1<f64>,
        mode_theta: ArrayView1<f64>,
        hessian: ArrayView2<f64>,
        spline: JointSplineArtifacts,
        is_logit: bool,
        scale: f64,
    ) -> Result<Self, String> {
        let n_samples = x.nrows();
        let p_base = x.ncols();
        let p_link = mode_theta.len();
        let dim = p_base + p_link;
        if hessian.nrows() != dim || hessian.ncols() != dim {
            return Err(format!(
                "Hessian dim mismatch: {}x{} vs {}x{}",
                dim,
                dim,
                hessian.nrows(),
                hessian.ncols()
            ));
        }
        let hessian_owned = hessian.to_owned();
        let chol_factor = hessian_owned
            .cholesky(Side::Lower)
            .map_err(|e| format!("Cholesky failed: {:?}", e))?;
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
            is_logit,
            scale,
        })
    }

    fn compute_logp_and_grad(&self, z: &Array1<f64>) -> (f64, Array1<f64>) {
        let dim = self.p_base + self.p_link;
        let mut mode = Array1::<f64>::zeros(dim);
        mode.slice_mut(ndarray::s![0..self.p_base])
            .assign(&self.mode_beta);
        mode.slice_mut(ndarray::s![self.p_base..])
            .assign(&self.mode_theta);
        let q = &mode + &self.chol.dot(z);
        let beta = q.slice(ndarray::s![0..self.p_base]).to_owned();
        let theta = q.slice(ndarray::s![self.p_base..]).to_owned();
        let u = self.x.dot(&beta);
        let (bwiggle, eta) = self.evaluate_link(&u, &theta);
        let mut ll = 0.0;
        let mut residual = Array1::<f64>::zeros(self.n_samples);

        if self.is_logit {
            // Bernoulli-logit log-likelihood
            for i in 0..self.n_samples {
                let eta_i = eta[i];
                let (y_i, w_i) = (self.y[i], self.weights[i]);
                let log1pexp = if eta_i > 0.0 {
                    eta_i + (-eta_i).exp().ln_1p()
                } else {
                    eta_i.exp().ln_1p()
                };
                ll += w_i * (y_i * eta_i - log1pexp);
                let mu = if eta_i > 0.0 {
                    1.0 / (1.0 + (-eta_i).exp())
                } else {
                    let e = eta_i.exp();
                    e / (1.0 + e)
                };
                residual[i] = w_i * (y_i - mu);
            }
        } else {
            // Gaussian identity log-likelihood: -0.5 * w * (y - η)² / σ²
            let inv_scale_sq = 1.0 / (self.scale * self.scale).max(1e-10);
            for i in 0..self.n_samples {
                let eta_i = eta[i];
                let (y_i, w_i) = (self.y[i], self.weights[i]);
                let r = y_i - eta_i;
                ll -= 0.5 * w_i * r * r * inv_scale_sq;
                residual[i] = w_i * r * inv_scale_sq; // grad of ll w.r.t. η
            }
        }

        let g_prime = self.compute_g_prime(&u, &theta);
        let grad_theta = &bwiggle.t().dot(&residual) - &self.penalty_link.dot(&theta);
        let r_scaled: Array1<f64> = residual
            .iter()
            .zip(g_prime.iter())
            .map(|(&r, &g)| r * g)
            .collect();
        let grad_beta = &fast_atv(&self.x, &r_scaled) - &self.penalty_base.dot(&beta);
        let penalty = 0.5 * beta.dot(&self.penalty_base.dot(&beta))
            + 0.5 * theta.dot(&self.penalty_link.dot(&theta));
        let mut grad_q = Array1::<f64>::zeros(dim);
        grad_q
            .slice_mut(ndarray::s![0..self.p_base])
            .assign(&grad_beta);
        grad_q
            .slice_mut(ndarray::s![self.p_base..])
            .assign(&grad_theta);
        (ll - penalty, self.chol_t.dot(&grad_q))
    }

    fn evaluate_link(&self, u: &Array1<f64>, theta: &Array1<f64>) -> (Array2<f64>, Array1<f64>) {
        use crate::basis::{BasisOptions, Dense, KnotSource, create_basis};
        let n = u.len();
        let n_raw = self
            .spline
            .knot_vector
            .len()
            .saturating_sub(self.spline.degree + 1);
        let n_c = self.spline.link_transform.ncols();
        if n_raw == 0 || n_c == 0 || theta.len() != n_c {
            // Return (n, 0) matrix when no link basis - avoids dimension mismatch downstream
            return (Array2::zeros((n, 0)), u.clone());
        }

        let (z_raw, z_c, _) = self.standardized_z(u);
        let Ok((b_raw_arc, _)) = create_basis::<Dense>(
            z_c.view(),
            KnotSource::Provided(self.spline.knot_vector.view()),
            self.spline.degree,
            BasisOptions::value(),
        ) else {
            return (Array2::zeros((n, n_c)), u.clone());
        };
        let mut b_raw = b_raw_arc.as_ref().clone();

        // Match joint training/prediction model: C^1 linear extension outside [0, 1]
        // B_ext(z_raw) = B(z_c) + (z_raw - z_c) * B'(z_c)
        let mut needs_ext = false;
        for i in 0..n {
            if (z_raw[i] - z_c[i]).abs() > 1e-12 {
                needs_ext = true;
                break;
            }
        }
        if needs_ext
            && let Ok((b_prime_arc, _)) = create_basis::<Dense>(
                z_c.view(),
                KnotSource::Provided(self.spline.knot_vector.view()),
                self.spline.degree,
                BasisOptions::first_derivative(),
            )
        {
            let b_prime = b_prime_arc.as_ref();
            for i in 0..n {
                let dz = z_raw[i] - z_c[i];
                if dz.abs() <= 1e-12 {
                    continue;
                }
                for j in 0..b_raw.ncols() {
                    b_raw[[i, j]] += dz * b_prime[[i, j]];
                }
            }
        }

        let b = if self.spline.link_transform.nrows() == n_raw {
            b_raw.dot(&self.spline.link_transform)
        } else {
            Array2::zeros((n, n_c))
        };
        (b.clone(), u + &b.dot(theta))
    }

    fn compute_g_prime(&self, u: &Array1<f64>, theta: &Array1<f64>) -> Array1<f64> {
        use crate::basis::{BasisOptions, Dense, KnotSource, create_basis};
        let n = u.len();
        let mut g = Array1::<f64>::ones(n);
        let (_, z_c, rw) = self.standardized_z(u);
        let n_raw = self
            .spline
            .knot_vector
            .len()
            .saturating_sub(self.spline.degree + 1);
        let n_c = self.spline.link_transform.ncols();
        if n_raw == 0 || n_c == 0 || theta.len() != n_c {
            return g;
        }

        // For the C^1 extension B_ext(z_raw)=B(z_c)+(z_raw-z_c)B'(z_c),
        // dB_ext/dz_raw = B'(z_c) everywhere (including outside [0,1]).
        let Ok((b_prime_raw_arc, _)) = create_basis::<Dense>(
            z_c.view(),
            KnotSource::Provided(self.spline.knot_vector.view()),
            self.spline.degree,
            BasisOptions::first_derivative(),
        ) else {
            return g;
        };
        let b_prime_raw = b_prime_raw_arc.as_ref();
        if self.spline.link_transform.nrows() != n_raw {
            return g;
        }
        let b_prime_constrained = b_prime_raw.dot(&self.spline.link_transform);
        let dwiggle_dz = b_prime_constrained.dot(theta);
        for i in 0..n {
            g[i] = 1.0 + dwiggle_dz[i] / rw;
        }
        g
    }

    pub fn chol(&self) -> &Array2<f64> {
        &self.chol
    }
    pub fn mode(&self) -> (Array1<f64>, Array1<f64>) {
        (
            self.mode_beta.as_ref().clone(),
            self.mode_theta.as_ref().clone(),
        )
    }
}

impl HamiltonianTarget<Array1<f64>> for JointLinkPosterior {
    fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
        let (logp, g) = self.compute_logp_and_grad(position);
        grad.assign(&g);
        logp
    }
}

/// Runs NUTS sampling for joint (β, θ).
/// `is_logit`: true for Bernoulli-logit, false for Gaussian-identity
/// `scale`: dispersion parameter for identity link (ignored if is_logit=true)
pub fn run_joint_nuts_sampling(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    penalty_base: ArrayView2<f64>,
    penalty_link: ArrayView2<f64>,
    mode_beta: ArrayView1<f64>,
    mode_theta: ArrayView1<f64>,
    hessian: ArrayView2<f64>,
    spline: JointSplineArtifacts,
    config: &NutsConfig,
    is_logit: bool,
    scale: f64,
) -> Result<NutsResult, String> {
    let (p_base, dim) = (mode_beta.len(), mode_beta.len() + mode_theta.len());
    let target = JointLinkPosterior::new(
        x,
        y,
        weights,
        penalty_base,
        penalty_link,
        mode_beta,
        mode_theta,
        hessian,
        spline,
        is_logit,
        scale,
    )?;
    let chol = target.chol().clone();
    let (mb, mt) = target.mode();
    let mut mode_arr = Array1::<f64>::zeros(dim);
    mode_arr.slice_mut(ndarray::s![0..p_base]).assign(&mb);
    mode_arr.slice_mut(ndarray::s![p_base..]).assign(&mt);
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
    let samples_array = sampler.run(config.n_samples, config.nwarmup);
    let (n_chains, n_samples_out) = (samples_array.shape()[0], samples_array.shape()[1]);
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

    // Compute split R-hat and autocorrelation-based ESS diagnostics.
    let (rhat, ess) = compute_rhat_ess(&samples_array, n_chains, n_samples_out, dim);
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

/// Compute split R-hat and autocorrelation-based ESS diagnostics.
fn compute_rhat_ess(samples: &Array3<f64>, _: usize, _: usize, _: usize) -> (f64, f64) {
    compute_split_rhat_and_ess(samples)
}
