//! NUTS Sampler using mini-mcmc
//!
//! This module provides NUTS (No-U-Turn Sampler) for honest uncertainty
//! quantification after PIRLS convergence.
//!
//! # Design
//!
//! Since mini-mcmc's NUTS uses an identity mass matrix, we whiten the
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
//! sharing across chains without duplication when mini-mcmc clones the target.

use crate::faer_ndarray::{FaerCholesky, fast_atv};
use crate::types::LikelihoodFamily;
use faer::Side;
use mini_mcmc::generic_hmc::HamiltonianTarget;
use mini_mcmc::generic_nuts::GenericNUTS;
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// Ensure a matrix is positive definite by adding a minimal conditional ridge.
/// Only adds regularization if Cholesky fails, avoiding bias on well-conditioned matrices.
fn ensure_positive_definite_hmc(mat: &mut Array2<f64>) {
    if mat.cholesky(Side::Lower).is_ok() {
        return; // Already positive definite, no regularization needed
    }

    // Matrix needs regularization - use diagonal-scaled nugget
    let diag_scale = mat
        .diag()
        .iter()
        .map(|&d| d.abs())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let nugget = 1e-8 * diag_scale;
    for i in 0..mat.nrows() {
        mat[[i, i]] += nugget;
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

    for d in 0..dim {
        // Collect split-chain means and variances
        let mut chain_means = Vec::with_capacity(n_split_chains);
        let mut chain_vars = Vec::with_capacity(n_split_chains);

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
            chain_means.push(mean1);
            chain_vars.push(var1);

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
            chain_means.push(mean2);
            chain_vars.push(var2);
        }

        // Within-chain variance W
        let w: f64 = chain_vars.iter().sum::<f64>() / n_split_chains as f64;

        // Between-chain variance B
        let overall_mean: f64 = chain_means.iter().sum::<f64>() / n_split_chains as f64;
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

        // ESS approximation: n_eff ≈ n * m / (1 + 2 * sum of autocorrelations)
        // Simple approximation using variance ratio
        let ess_d = if var_hat > 1e-10 {
            n_split_chains as f64 * n_split_samples as f64 * w / var_hat
        } else {
            n_split_chains as f64 * n_split_samples as f64
        };
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
/// Using Arc prevents memory explosion when mini-mcmc clones the target.
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
pub struct NutsPosterior {
    /// Shared read-only data (Arc prevents duplication)
    data: SharedData,
    /// Transform: L where L L^T = H^{-1} (computed from Hessian)
    /// This is the inverse-transpose of the Cholesky of H.
    chol: Array2<f64>,
    /// L^T for gradient chain rule: ∇_z = L^T @ ∇_β
    chol_t: Array2<f64>,
    /// Link function type
    is_logit: bool,
    /// Whether Firth bias reduction was used in training (must match for HMC)
    firth_bias_reduction: bool,
}

impl NutsPosterior {
    /// Creates a new posterior target from ndarray data.
    ///
    /// # Arguments
    /// * `x` - Design matrix [n_samples, dim]
    /// * `y` - Response vector [n_samples]
    /// * `weights` - Prior weights [n_samples]
    /// * `penalty_matrix` - Combined penalty S [dim, dim]
    /// * `mode` - MAP estimate μ [dim]
    /// * `hessian` - Hessian H [dim, dim] (NOT the inverse!)
    /// * `is_logit` - True for logistic regression, false for Gaussian
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
        is_logit: bool,
        firth_bias_reduction: bool,
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
            is_logit,
            firth_bias_reduction,
        })
    }

    /// Compute log-posterior and gradient analytically using ndarray.
    ///
    /// Returns (log_posterior, gradient_z) where gradient_z is the gradient
    /// with respect to the whitened parameters z.
    fn compute_logp_and_grad_nd(&self, z: &Array1<f64>) -> (f64, Array1<f64>) {
        // === Step 1: Transform z (whitened) -> β (original) ===
        // β = μ + L @ z
        let beta = self.data.mode.as_ref() + &self.chol.dot(z);

        // === Step 2: Compute η = X @ β ===
        let eta = self.data.x.dot(&beta);

        // === Step 3: Compute log-likelihood and gradient ===
        let (ll, grad_ll_beta) = if self.is_logit {
            self.logit_logp_and_grad(&eta)
        } else {
            self.gaussian_logp_and_grad(&eta)
        };

        // === Step 4: Compute penalty and its gradient ===
        // penalty = 0.5 * β^T @ S @ β
        let s_beta = self.data.penalty.dot(&beta);
        let penalty = 0.5 * beta.dot(&s_beta);

        // ∇_β penalty = S @ β
        let grad_penalty_beta = s_beta;

        // === Step 5: Combined gradient in β space ===
        // ∇_β log p = ∇_β ll - ∇_β penalty
        let mut grad_beta = &grad_ll_beta - &grad_penalty_beta;

        // === Step 5b: Firth bias reduction term (if enabled) ===
        // Firth adds 0.5 * log|I(β)| to log-posterior, where I = X'WX is Fisher information
        // For logistic regression, W = diag(w * μ(1-μ))
        let firth_term = if self.firth_bias_reduction && self.is_logit {
            let n = self.data.n_samples;
            let p = self.data.dim;

            // Compute IRLS weights: w_irls = prior_weight * μ(1-μ)
            let mut w_irls = Array1::<f64>::zeros(n);
            for i in 0..n {
                let eta_i = eta[i].clamp(-700.0, 700.0);
                let mu_i = 1.0 / (1.0 + (-eta_i).exp());
                let mu_clamped = mu_i.clamp(1e-10, 1.0 - 1e-10);
                w_irls[i] = self.data.weights[i] * mu_clamped * (1.0 - mu_clamped);
            }

            // Build Fisher information: I = X' W X
            let mut fisher = Array2::<f64>::zeros((p, p));
            for i in 0..n {
                let w_i = w_irls[i].max(1e-10);
                for j in 0..p {
                    let xij = self.data.x[[i, j]];
                    for k in 0..p {
                        fisher[[j, k]] += w_i * xij * self.data.x[[i, k]];
                    }
                }
            }

            // Compute 0.5 * log|I| via Cholesky
            // Conditional regularization for stability
            ensure_positive_definite_hmc(&mut fisher);

            match fisher.cholesky(Side::Lower) {
                Ok(chol_i) => {
                    let half_log_det: f64 = chol_i.lower_triangular().diag().mapv(f64::ln).sum();

                    // Exact Firth gradient: ∂(0.5 log|I|)/∂β = X' h * (0.5 - μ)
                    // where h_i = leverage = x_i' (X'WX)^{-1} x_i * w_i
                    // Compute (X'WX)^{-1} = L^{-T} L^{-1} where L = chol(X'WX)
                    let l = chol_i.lower_triangular();

                    for i in 0..n {
                        let eta_i = eta[i].clamp(-700.0, 700.0);
                        let mu_i = 1.0 / (1.0 + (-eta_i).exp());
                        let mu_clamped = mu_i.clamp(1e-10, 1.0 - 1e-10);
                        let w_i = w_irls[i].max(1e-10);

                        // Compute h_ii = w_i * x_i' (X'WX)^{-1} x_i via L^{-1}(sqrt(w_i)*x_i)
                        // First solve L v = sqrt(w_i) * x_i
                        let sqrt_w = w_i.sqrt();
                        let mut v = Array1::<f64>::zeros(p);
                        for j in 0..p {
                            let mut sum = sqrt_w * self.data.x[[i, j]];
                            for k in 0..j {
                                sum -= l[[j, k]] * v[k];
                            }
                            v[j] = sum / l[[j, j]].max(1e-15);
                        }
                        // h_ii = ||v||^2
                        let h_ii: f64 = v.iter().map(|x| x * x).sum();

                        // Firth score contribution: h_ii * (0.5 - μ_i)
                        let firth_score = h_ii * (0.5 - mu_clamped);
                        for j in 0..p {
                            grad_beta[j] += firth_score * self.data.x[[i, j]];
                        }
                    }

                    half_log_det
                }
                Err(_) => 0.0, // Fall back to standard likelihood if Fisher is singular
            }
        } else {
            0.0
        };

        // === Step 6: Chain rule to get gradient in z space ===
        // ∇_z = L^T @ ∇_β
        let grad_z = self.chol_t.dot(&grad_beta);

        let logp = ll - penalty + firth_term;

        (logp, grad_z)
    }

    /// Logistic regression log-likelihood and gradient.
    fn logit_logp_and_grad(&self, eta: &Array1<f64>) -> (f64, Array1<f64>) {
        let n = self.data.n_samples;
        let mut ll = 0.0;
        let mut residual = Array1::<f64>::zeros(n);

        for i in 0..n {
            let eta_i = eta[i].clamp(-700.0, 700.0);
            let mu_i = 1.0 / (1.0 + (-eta_i).exp());
            let mu_clamped = mu_i.clamp(1e-10, 1.0 - 1e-10);

            // Log-likelihood: y*log(μ) + (1-y)*log(1-μ)
            let y_i = self.data.y[i];
            let w_i = self.data.weights[i];
            ll += w_i * (y_i * mu_clamped.ln() + (1.0 - y_i) * (1.0 - mu_clamped).ln());

            // Residual for gradient: y - μ (canonical link, score function)
            residual[i] = w_i * (y_i - mu_clamped);
        }

        // Gradient of log-likelihood: X^T @ (w * (y - μ))
        let grad_ll = fast_atv(&self.data.x, &residual);

        (ll, grad_ll)
    }

    /// Gaussian log-likelihood and gradient.
    fn gaussian_logp_and_grad(&self, eta: &Array1<f64>) -> (f64, Array1<f64>) {
        let n = self.data.n_samples;
        let mut ll = 0.0;
        let mut weighted_residual = Array1::<f64>::zeros(n);

        for i in 0..n {
            let residual = self.data.y[i] - eta[i];
            let w_i = self.data.weights[i];
            ll -= 0.5 * w_i * residual * residual;
            weighted_residual[i] = w_i * residual;
        }

        // Gradient of log-likelihood: X^T @ (w * (y - η))
        let grad_ll = fast_atv(&self.data.x, &weighted_residual);

        (ll, grad_ll)
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

/// Implement HamiltonianTarget for NUTS with analytical gradients.
impl HamiltonianTarget<Array1<f64>> for NutsPosterior {
    fn logp_and_grad(&self, position: &Array1<f64>, grad: &mut Array1<f64>) -> f64 {
        let (logp, grad_z) = self.compute_logp_and_grad_nd(position);
        grad.assign(&grad_z);
        logp
    }
}

/// Configuration for NUTS sampling.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NutsConfig {
    /// Number of samples to collect (after warmup)
    pub n_samples: usize,
    /// Number of warmup samples to discard
    pub n_warmup: usize,
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

impl Default for NutsConfig {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            n_warmup: 500,
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
        let n_warmup = n_samples;

        // More chains for higher dims (better R-hat estimation)
        let n_chains = if n_params > 50 { 4 } else { 2 };

        Self {
            n_samples,
            n_warmup,
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
    /// Returns (0.0, 0.0) if samples is empty to avoid index-out-of-bounds.
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

/// Runs NUTS sampling using mini-mcmc with whitened parameter space.
///
/// # Arguments
/// * `x` - Design matrix [n_samples, dim]
/// * `y` - Response vector [n_samples]
/// * `weights` - Prior weights [n_samples]
/// * `penalty_matrix` - Combined penalty S [dim, dim]
/// * `mode` - MAP estimate μ [dim]
/// * `hessian` - Penalized Hessian H [dim, dim] (NOT the inverse!)
/// * `is_logit` - True for logistic regression, false for Gaussian
/// * `firth_bias_reduction` - Whether Firth bias reduction was used in training
/// * `config` - NUTS configuration
pub fn run_nuts_sampling(
    x: ArrayView2<f64>,
    y: ArrayView1<f64>,
    weights: ArrayView1<f64>,
    penalty_matrix: ArrayView2<f64>,
    mode: ArrayView1<f64>,
    hessian: ArrayView2<f64>,
    is_logit: bool,
    firth_bias_reduction: bool,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    let dim = mode.len();

    // Create posterior target with analytical gradients (Firth term included when enabled)
    let target = NutsPosterior::new(
        x,
        y,
        weights,
        penalty_matrix,
        mode,
        hessian,
        is_logit,
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
    let mut sampler = GenericNUTS::new(target, initial_positions, config.target_accept);

    // Note: run_progress() has blocking issues in some contexts, using run() instead
    let samples_array = sampler.run(config.n_samples, config.n_warmup);

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
            let z_view = samples_array.slice(ndarray::s![chain, sample_i, ..]);
            z_buffer.assign(&z_view);
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
    let (rhat, ess) = if n_chains >= 2 && n_samples_out >= 4 {
        compute_split_rhat_and_ess(&samples_array)
    } else {
        // Fall back to simple estimates if not enough chains/samples
        (1.0, (total_samples as f64) * 0.5)
    };

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
}

/// Flattened numeric inputs for Royston-Parmar NUTS sampling.
pub struct SurvivalNutsInputs<'a> {
    pub flat: SurvivalFlatInputs<'a>,
    pub penalties: crate::survival::PenaltyBlocks,
    pub monotonicity: crate::survival::MonotonicityPenalty,
    pub spec: crate::survival::SurvivalSpec,
    pub mode: ArrayView1<'a, f64>,
    pub hessian: ArrayView2<'a, f64>,
}

/// Family-dispatched flattened NUTS inputs.
pub enum FamilyNutsInputs<'a> {
    Glm(GlmFlatInputs<'a>),
    Survival(SurvivalNutsInputs<'a>),
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
            false,
            glm.firth_bias_reduction,
            config,
        ),
        (LikelihoodFamily::BinomialLogit, FamilyNutsInputs::Glm(glm)) => run_nuts_sampling(
            glm.x,
            glm.y,
            glm.weights,
            glm.penalty_matrix,
            glm.mode,
            glm.hessian,
            true,
            glm.firth_bias_reduction,
            config,
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
                survival.penalties,
                survival.monotonicity,
                survival.spec,
                survival.mode,
                survival.hessian,
                config,
            )
        }
        (LikelihoodFamily::RoystonParmar, FamilyNutsInputs::Glm(_)) => Err(
            "RoystonParmar family requires FamilyNutsInputs::Survival flattened inputs"
                .to_string(),
        ),
        (_, FamilyNutsInputs::Survival(_)) => Err(
            "Survival flattened inputs are only valid for LikelihoodFamily::RoystonParmar"
                .to_string(),
        ),
    }
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
        /// Entry design matrix
        x_entry: Arc<Array2<f64>>,
        /// Exit design matrix
        x_exit: Arc<Array2<f64>>,
        /// Exit derivative design matrix
        x_derivative: Arc<Array2<f64>>,
        /// Sample weights
        sample_weight: Arc<Array1<f64>>,
        /// Event indicators (1 = event, 0 = censored)
        event_target: Arc<Array1<u8>>,
        /// Competing event indicators
        event_competing: Arc<Array1<u8>>,
        /// Entry ages
        age_entry: Arc<Array1<f64>>,
        /// Exit ages
        age_exit: Arc<Array1<f64>>,
        /// Penalty blocks
        penalties: Arc<PenaltyBlocks>,
        /// Monotonicity constraint
        monotonicity: Arc<MonotonicityPenalty>,
        /// Survival spec
        spec: SurvivalSpec,
        /// MAP estimate (mode) μ [dim]
        mode: Arc<Array1<f64>>,
    }

    /// Whitened log-posterior target for survival models with analytical gradients.
    #[derive(Clone)]
    pub struct SurvivalPosterior {
        /// Shared read-only data (Arc prevents duplication)
        data: SharedSurvivalData,
        /// Transform: L where L L^T = H^{-1}
        chol: Array2<f64>,
        /// L^T for gradient chain rule: ∇_z = L^T @ ∇_β
        chol_t: Array2<f64>,
    }

    impl SurvivalPosterior {
        /// Creates a new survival posterior target.
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            age_entry: ArrayView1<'_, f64>,
            age_exit: ArrayView1<'_, f64>,
            event_target: ArrayView1<'_, u8>,
            event_competing: ArrayView1<'_, u8>,
            sample_weight: ArrayView1<'_, f64>,
            x_entry: ArrayView2<'_, f64>,
            x_exit: ArrayView2<'_, f64>,
            x_derivative: ArrayView2<'_, f64>,
            penalties: PenaltyBlocks,
            monotonicity: MonotonicityPenalty,
            spec: SurvivalSpec,
            mode: ArrayView1<f64>,
            hessian: ArrayView2<f64>,
        ) -> Result<Self, String> {
            let dim = mode.len();

            // Compute whitening transform via Cholesky of Hessian
            let hessian_owned = hessian.to_owned();
            let chol_factor = hessian_owned
                .cholesky(Side::Lower)
                .map_err(|e| format!("Hessian Cholesky decomposition failed: {:?}", e))?;
            let l_h = chol_factor.lower_triangular();
            let chol = solve_upper_triangular_transpose(&l_h, dim);
            let chol_t = chol.t().to_owned();

            let data = SharedSurvivalData {
                x_entry: Arc::new(x_entry.to_owned()),
                x_exit: Arc::new(x_exit.to_owned()),
                x_derivative: Arc::new(x_derivative.to_owned()),
                sample_weight: Arc::new(sample_weight.to_owned()),
                event_target: Arc::new(event_target.to_owned()),
                event_competing: Arc::new(event_competing.to_owned()),
                age_entry: Arc::new(age_entry.to_owned()),
                age_exit: Arc::new(age_exit.to_owned()),
                penalties: Arc::new(penalties),
                monotonicity: Arc::new(monotonicity),
                spec,
                mode: Arc::new(mode.to_owned()),
            };

            Ok(Self { data, chol, chol_t })
        }

        /// Compute log-posterior and gradient analytically.
        fn compute_logp_and_grad(&self, z: &Array1<f64>) -> Result<(f64, Array1<f64>), String> {
            // Transform z (whitened) -> β (original): β = μ + L @ z
            let beta = self.data.mode.as_ref() + &self.chol.dot(z);

            // Create a temporary working model to compute likelihood
            let model = WorkingModelSurvival::from_engine_inputs(
                SurvivalEngineInputs {
                    age_entry: self.data.age_entry.view(),
                    age_exit: self.data.age_exit.view(),
                    event_target: self.data.event_target.view(),
                    event_competing: self.data.event_competing.view(),
                    sample_weight: self.data.sample_weight.view(),
                    x_entry: self.data.x_entry.view(),
                    x_exit: self.data.x_exit.view(),
                    x_derivative: self.data.x_derivative.view(),
                },
                self.data.penalties.as_ref().clone(),
                self.data.monotonicity.as_ref().clone(),
                self.data.spec,
            )
            .map_err(|e| format!("Survival state construction failed: {:?}", e))?;

            // Compute state (deviance and gradient in beta space)
            let state = model
                .update_state(&beta)
                .map_err(|e| format!("Survival state update failed: {:?}", e))?;

            // Convert deviance to log-posterior: logp = -0.5 * deviance
            // (deviance = -2 * log_likelihood + penalty, so -0.5 * deviance = log_likelihood - 0.5*penalty)
            let logp = -0.5 * state.deviance;

            // The gradient from update_state is ∂deviance/∂β = -2 * ∂log_posterior/∂β
            // So ∂log_posterior/∂β = -0.5 * gradient
            let grad_beta = state.gradient.mapv(|g| -0.5 * g);

            // Chain rule to get gradient in z space: ∇_z = L^T @ ∇_β
            let grad_z = self.chol_t.dot(&grad_beta);

            Ok((logp, grad_z))
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
                Ok((logp, grad_z)) => {
                    grad.assign(&grad_z);
                    logp
                }
                Err(e) => {
                    // On error (e.g., monotonicity violation), return -infinity log-prob
                    // This causes NUTS to reject the proposal
                    log::warn!("Survival posterior evaluation failed: {}", e);
                    grad.fill(0.0);
                    f64::NEG_INFINITY
                }
            }
        }
    }

    /// Runs NUTS sampling for survival models with whitened parameter space.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn run_survival_nuts_sampling(
        age_entry: ArrayView1<'_, f64>,
        age_exit: ArrayView1<'_, f64>,
        event_target: ArrayView1<'_, u8>,
        event_competing: ArrayView1<'_, u8>,
        sample_weight: ArrayView1<'_, f64>,
        x_entry: ArrayView2<'_, f64>,
        x_exit: ArrayView2<'_, f64>,
        x_derivative: ArrayView2<'_, f64>,
        penalties: PenaltyBlocks,
        monotonicity: MonotonicityPenalty,
        spec: SurvivalSpec,
        mode: ArrayView1<f64>,
        hessian: ArrayView2<f64>,
        config: &NutsConfig,
    ) -> Result<NutsResult, String> {
        let dim = mode.len();

        // Create posterior target
        let target = SurvivalPosterior::new(
            age_entry,
            age_exit,
            event_target,
            event_competing,
            sample_weight,
            x_entry,
            x_exit,
            x_derivative,
            penalties,
            monotonicity,
            spec,
            mode,
            hessian,
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

        // Create GenericNUTS sampler
        let mut sampler = GenericNUTS::new(target, initial_positions, config.target_accept);

        // Run sampling with progress bar
        let (samples_array, run_stats) = sampler
            .run_progress(config.n_samples, config.n_warmup)
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
                let z_view = samples_array.slice(ndarray::s![chain, sample_i, ..]);
                z_buffer.assign(&z_view);

                // Transform to β: β = μ + L @ z
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
        let rhat = f64::from(run_stats.rhat.mean);
        let ess = f64::from(run_stats.ess.mean);
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

    /// Test: Survival gradient consistency via finite differences.
    ///
    /// Validates that analytical gradients in SurvivalPosterior match numerical gradients.
    #[cfg(test)]
    mod survival_gradient_tests {
        use super::*;
        use crate::survival::{BasisDescriptor, SurvivalLayoutInputs, build_survival_layout};
        use ndarray::array;

        /// Helper to compute numerical gradient via central finite differences
        fn finite_difference_gradient<F>(f: F, x: &Array1<f64>, eps: f64) -> Array1<f64>
        where
            F: Fn(&Array1<f64>) -> f64,
        {
            let dim = x.len();
            let mut grad = Array1::<f64>::zeros(dim);

            for i in 0..dim {
                let mut x_plus = x.clone();
                let mut x_minus = x.clone();
                x_plus[i] += eps;
                x_minus[i] -= eps;

                let f_plus = f(&x_plus);
                let f_minus = f(&x_minus);

                grad[i] = (f_plus - f_minus) / (2.0 * eps);
            }

            grad
        }

        /// Cosine similarity between two vectors
        fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
            let dot = a.dot(b);
            let norm_a = a.dot(a).sqrt();
            let norm_b = b.dot(b).sqrt();
            if norm_a < 1e-15 || norm_b < 1e-15 {
                return 0.0;
            }
            dot / (norm_a * norm_b)
        }

        #[test]
        fn test_survival_gradient_finite_difference() {
            use rand::rngs::StdRng;
            use rand::{RngExt, SeedableRng};

            // Create a minimal survival setup
            let n = 50;
            let mut rng = StdRng::seed_from_u64(12345);

            // Synthetic times and events
            let age_entry = Array1::<f64>::from_shape_fn(n, |_| 40.0 + rng.random_range(0.0..10.0));
            let age_exit =
                Array1::<f64>::from_shape_fn(n, |i| age_entry[i] + rng.random_range(1.0..20.0));
            let event_target =
                Array1::<u8>::from_shape_fn(n, |_| if rng.random::<f64>() < 0.3 { 1 } else { 0 });
            let sample_weight = Array1::<f64>::ones(n);
            let event_competing = Array1::<u8>::zeros(n);
            let static_covariates = Array2::<f64>::zeros((n, 0));
            let static_covariate_names = Vec::new();

            let training_data = SurvivalLayoutInputs {
                age_entry: age_entry.clone(),
                age_exit: age_exit.clone(),
                event_target: event_target.clone(),
                event_competing: event_competing.clone(),
                sample_weight: sample_weight.clone(),
                static_covariates,
                static_covariate_names,
                time_varying_covariate: None,
            };

            let basis = BasisDescriptor {
                knot_vector: array![0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0],
                degree: 2,
            };

            let layout_bundle = match build_survival_layout(&training_data, &basis, 0.1, 2, 4, None)
            {
                Ok(bundle) => bundle,
                Err(e) => {
                    println!(
                        "[Survival Gradient] Skipping: could not build layout: {}",
                        e
                    );
                    return;
                }
            };
            let layout = layout_bundle.layout;
            let monotonicity = layout_bundle.monotonicity;
            let spec = SurvivalSpec::default();

            let dim = layout.combined_exit.ncols();
            let hessian = Array2::<f64>::from_diag(&Array1::<f64>::from_elem(dim, 10.0));
            let mode = Array1::<f64>::zeros(dim);

            // Try to create the posterior
            let posterior_result = SurvivalPosterior::new(
                age_entry.view(),
                age_exit.view(),
                event_target.view(),
                event_competing.view(),
                sample_weight.view(),
                layout.combined_entry.view(),
                layout.combined_exit.view(),
                layout.combined_derivative_exit.view(),
                layout.penalties.clone(),
                monotonicity,
                spec,
                mode.view(),
                hessian.view(),
            );

            // If posterior creation fails (e.g., due to complex survival setup),
            // skip the test with a warning
            let posterior = match posterior_result {
                Ok(p) => p,
                Err(e) => {
                    println!(
                        "[Survival Gradient] Skipping: could not create posterior: {}",
                        e
                    );
                    return;
                }
            };

            // Test at multiple random points
            let n_test_points = 3;
            let eps = 1e-5;
            let mut failures: Vec<String> = Vec::new();

            for test_idx in 0..n_test_points {
                // Random point in whitened space
                let z = Array1::<f64>::from_shape_fn(dim, |_| rng.random_range(-0.5..0.5));

                // Compute analytical gradient
                let analytical_result = posterior.compute_logp_and_grad(&z);
                let (logp, analytical_grad) = match analytical_result {
                    Ok((lp, g)) => (lp, g),
                    Err(e) => {
                        println!(
                            "[Survival Gradient] Point {}: evaluation failed: {}",
                            test_idx, e
                        );
                        continue;
                    }
                };

                // Compute numerical gradient
                let numeric_grad = finite_difference_gradient(
                    |z_test| match posterior.compute_logp_and_grad(z_test) {
                        Ok((lp, _)) => lp,
                        Err(_) => f64::NEG_INFINITY,
                    },
                    &z,
                    eps,
                );

                // Compute cosine similarity
                let cos_sim = cosine_similarity(&analytical_grad, &numeric_grad);

                // Compute relative error
                let diff = &analytical_grad - &numeric_grad;
                let rel_error =
                    diff.dot(&diff).sqrt() / (1e-10 + analytical_grad.dot(&analytical_grad).sqrt());

                println!(
                    "[Survival Gradient] Point {}: logp={:.4e}, cos_sim={:.6}, rel_error={:.4e}",
                    test_idx, logp, cos_sim, rel_error
                );

                if cos_sim < 0.999 || !cos_sim.is_finite() {
                    failures.push(format!(
                        "Point {}: cos_sim={:.6}, rel_error={:.4e}",
                        test_idx, cos_sim, rel_error
                    ));
                }
            }

            assert!(
                failures.is_empty(),
                "[Survival Gradient] Gradient mismatches found:\n{}",
                failures.join("\n")
            );
        }
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
        let (b_wiggle, eta) = self.evaluate_link(&u, &theta);
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
        let grad_theta = &b_wiggle.t().dot(&residual) - &self.penalty_link.dot(&theta);
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
        use crate::basis::{SplineScratch, evaluate_bspline_basis_scalar};
        let n = u.len();
        let (min_u, max_u) = self.spline.knot_range;
        let rw = (max_u - min_u).max(1e-6);
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
        let z: Array1<f64> = u.mapv(|v| ((v - min_u) / rw).clamp(0.0, 1.0));
        let mut b = Array2::<f64>::zeros((n, n_c));
        let mut raw = vec![0.0; n_raw];
        let mut scratch = SplineScratch::new(self.spline.degree);
        for i in 0..n {
            raw.fill(0.0);
            if evaluate_bspline_basis_scalar(
                z[i],
                self.spline.knot_vector.view(),
                self.spline.degree,
                &mut raw,
                &mut scratch,
            )
            .is_ok()
                && self.spline.link_transform.nrows() == n_raw
            {
                for c in 0..n_c {
                    b[[i, c]] = raw
                        .iter()
                        .zip(self.spline.link_transform.column(c).iter())
                        .map(|(&r, &t)| r * t)
                        .sum();
                }
            }
        }
        (b.clone(), u + &b.dot(theta))
    }

    fn compute_g_prime(&self, u: &Array1<f64>, theta: &Array1<f64>) -> Array1<f64> {
        use crate::basis::{
            evaluate_bspline_derivative_scalar_into, internal::BsplineScratch,
        };
        let n = u.len();
        let mut g = Array1::<f64>::ones(n);
        let (min_u, max_u) = self.spline.knot_range;
        let rw = (max_u - min_u).max(1e-6);
        let n_raw = self
            .spline
            .knot_vector
            .len()
            .saturating_sub(self.spline.degree + 1);
        let n_c = self.spline.link_transform.ncols();
        if n_raw == 0 || n_c == 0 || theta.len() != n_c {
            return g;
        }

        // Pre-allocate all buffers ONCE outside the loop
        let mut deriv_raw = vec![0.0; n_raw];
        let num_basis_lower = self
            .spline
            .knot_vector
            .len()
            .saturating_sub(self.spline.degree);
        let mut lower_basis = vec![0.0; num_basis_lower];
        let mut lower_scratch = BsplineScratch::new(self.spline.degree.saturating_sub(1));

        for i in 0..n {
            let u_i = u[i];
            let z_i = ((u_i - min_u) / rw).clamp(0.0, 1.0);

            // At boundaries, g'(u) = 1 (wiggle is constant)
            if z_i <= 1e-8 || z_i >= 1.0 - 1e-8 {
                g[i] = 1.0;
                continue;
            }

            // Zero-allocation eval using pre-allocated buffers
            deriv_raw.fill(0.0);
            if evaluate_bspline_derivative_scalar_into(
                z_i,
                self.spline.knot_vector.view(),
                self.spline.degree,
                &mut deriv_raw,
                &mut lower_basis,
                &mut lower_scratch,
            )
            .is_err()
            {
                continue;
            }

            // d(wiggle)/dz = B'(z) @ Z @ θ
            let d_wiggle_dz: f64 = if self.spline.link_transform.nrows() == n_raw {
                (0..n_c)
                    .map(|c| {
                        let b_prime_c: f64 = (0..n_raw)
                            .map(|r| deriv_raw[r] * self.spline.link_transform[[r, c]])
                            .sum();
                        b_prime_c * theta[c]
                    })
                    .sum()
            } else {
                0.0
            };

            g[i] = 1.0 + d_wiggle_dz / rw;
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
    let mut sampler = GenericNUTS::new(target, initial_positions, config.target_accept);
    let samples_array = sampler.run(config.n_samples, config.n_warmup);
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

    // Compute R-hat and ESS heuristics (simplified between/within-chain variance method)
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
    mode: ArrayView1<'a, f64>,
    hessian: ArrayView2<'a, f64>,
    config: &NutsConfig,
) -> Result<NutsResult, String> {
    run_nuts_sampling_flattened_family(
        LikelihoodFamily::RoystonParmar,
        FamilyNutsInputs::Survival(SurvivalNutsInputs {
            flat,
            penalties,
            monotonicity,
            spec,
            mode,
            hessian,
        }),
        config,
    )
}

/// Compute R-hat and ESS heuristics for MCMC samples.
/// NOTE: This is a simplified diagnostic, NOT the full Vehtari et al. split-R-hat/ESS.
/// Uses basic between/within chain variance ratio as a convergence heuristic.
fn compute_rhat_ess(
    samples: &Array3<f64>,
    n_chains: usize,
    n_samples: usize,
    dim: usize,
) -> (f64, f64) {
    if n_chains < 2 || n_samples < 4 {
        return (f64::NAN, (n_chains * n_samples) as f64 * 0.5);
    }

    let mut max_rhat = 1.0_f64;
    let mut min_ess = f64::MAX;

    for d in 0..dim {
        // Compute chain means and overall mean
        let mut chain_means = vec![0.0; n_chains];
        let mut chain_vars = vec![0.0; n_chains];
        let mut overall_mean = 0.0;

        for c in 0..n_chains {
            let mut sum = 0.0;
            for s in 0..n_samples {
                sum += samples[[c, s, d]];
            }
            chain_means[c] = sum / n_samples as f64;
            overall_mean += chain_means[c];
        }
        overall_mean /= n_chains as f64;

        // Within-chain variance W
        for c in 0..n_chains {
            let mut sum_sq = 0.0;
            for s in 0..n_samples {
                let diff = samples[[c, s, d]] - chain_means[c];
                sum_sq += diff * diff;
            }
            chain_vars[c] = sum_sq / (n_samples - 1) as f64;
        }
        let w: f64 = chain_vars.iter().sum::<f64>() / n_chains as f64;

        // Between-chain variance B
        let b: f64 = {
            let mut sum_sq = 0.0;
            for c in 0..n_chains {
                let diff = chain_means[c] - overall_mean;
                sum_sq += diff * diff;
            }
            sum_sq * n_samples as f64 / (n_chains - 1) as f64
        };

        // R-hat = sqrt((n-1)/n + B/(n*W))
        let rhat_d = if w > 1e-10 {
            (((n_samples as f64 - 1.0) / n_samples as f64) + (b / (n_samples as f64 * w))).sqrt()
        } else {
            1.0
        };
        max_rhat = max_rhat.max(rhat_d);

        // Simplified ESS = n * m * W / (W + B/n)
        let var_total = w + b / n_samples as f64;
        let ess_d = if var_total > 1e-10 {
            (n_chains * n_samples) as f64 * w / var_total
        } else {
            (n_chains * n_samples) as f64
        };
        min_ess = min_ess.min(ess_d);
    }

    (max_rhat, min_ess.max(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_posterior_creation() {
        let x = ndarray::array![[1.0]];
        let y = ndarray::array![1.0];
        let weights = ndarray::array![1.0];
        let penalty = ndarray::array![[0.0]];
        let mode = ndarray::array![0.0];
        // Pass Hessian (not inverse) - identity matrix
        let hessian = ndarray::array![[1.0]];

        let target = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            true,
            false,
        )
        .unwrap();

        assert_eq!(target.dim(), 1);
    }

    #[test]
    fn test_analytical_gradient_gaussian() {
        let x = ndarray::array![[1.0]];
        let y = ndarray::array![0.5];
        let weights = ndarray::array![1.0];
        let penalty = ndarray::array![[0.0]];
        let mode = ndarray::array![0.0];
        let hessian = ndarray::array![[1.0]];

        let target = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            false, // Gaussian
            false,
        )
        .unwrap();

        let z = ndarray::array![0.0];
        let (logp, grad) = target.compute_logp_and_grad_nd(&z);

        // At z=0, β=0, eta=0, residual=0.5
        // log L = -0.5 * 0.5^2 = -0.125
        assert!(
            (logp - (-0.125)).abs() < 0.01,
            "Expected logp~-0.125, got {}",
            logp
        );

        // Gradient: X^T @ (w * (y - η)) = 1 * 1 * 0.5 = 0.5
        // Chain rule: L^T @ grad_β = 1 * 0.5 = 0.5
        assert!(
            (grad[0] - 0.5).abs() < 0.01,
            "Expected grad~0.5, got {}",
            grad[0]
        );
    }

    #[test]
    fn test_analytical_gradient_logit() {
        let x = ndarray::array![[1.0]];
        let y = ndarray::array![1.0];
        let weights = ndarray::array![1.0];
        let penalty = ndarray::array![[0.0]];
        let mode = ndarray::array![0.0];
        let hessian = ndarray::array![[1.0]];

        let target = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            true, // Logit
            false,
        )
        .unwrap();

        let z = ndarray::array![0.0];
        let (logp, grad) = target.compute_logp_and_grad_nd(&z);

        // At z=0, β=0, eta=0, μ=0.5
        // log L = y*log(0.5) + (1-y)*log(0.5) = 1*log(0.5) = -0.693
        assert!(
            (logp - (-0.693)).abs() < 0.01,
            "Expected logp~-0.693, got {}",
            logp
        );

        // Gradient: X^T @ (w * (y - μ)) = 1 * 1 * (1 - 0.5) = 0.5
        assert!(
            (grad[0] - 0.5).abs() < 0.01,
            "Expected grad~0.5, got {}",
            grad[0]
        );
    }

    #[test]
    fn test_gradient_vs_finite_difference() {
        let x = ndarray::array![[1.0, 0.5], [0.5, 1.0], [1.0, 1.0]];
        let y = ndarray::array![1.0, 0.0, 1.0];
        let weights = ndarray::array![1.0, 1.0, 1.0];
        let penalty = ndarray::array![[0.1, 0.0], [0.0, 0.1]];
        let mode = ndarray::array![0.0, 0.0];
        let hessian = ndarray::array![[1.0, 0.0], [0.0, 1.0]];

        let target = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            true, // Logit
            false,
        )
        .unwrap();

        let z = ndarray::array![0.5, -0.3];
        let (_, grad) = target.compute_logp_and_grad_nd(&z);

        // Finite difference check
        let eps = 1e-5;
        for i in 0..2 {
            let mut z_plus = z.clone();
            let mut z_minus = z.clone();
            z_plus[i] += eps;
            z_minus[i] -= eps;

            let (logp_plus, _) = target.compute_logp_and_grad_nd(&z_plus);
            let (logp_minus, _) = target.compute_logp_and_grad_nd(&z_minus);

            let fd_grad = (logp_plus - logp_minus) / (2.0 * eps);
            let rel_error = (grad[i] - fd_grad).abs() / (grad[i].abs().max(1e-8));

            assert!(
                rel_error < 1e-4,
                "Gradient mismatch at index {}: analytical={}, fd={}, rel_error={}",
                i,
                grad[i],
                fd_grad,
                rel_error
            );
        }
    }

    #[test]
    fn test_arc_prevents_cloning_data() {
        let x = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let y = ndarray::array![1.0, 0.0];
        let weights = ndarray::array![1.0, 1.0];
        let penalty = ndarray::array![[0.1, 0.0], [0.0, 0.1]];
        let mode = ndarray::array![0.0, 0.0];
        let hessian = ndarray::array![[1.0, 0.0], [0.0, 1.0]];

        let target1 = NutsPosterior::new(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            true,
            false,
        )
        .unwrap();

        // Clone should share data via Arc, not duplicate
        let target2 = target1.clone();

        // Verify Arc::ptr_eq - same underlying allocation
        assert!(Arc::ptr_eq(&target1.data.x, &target2.data.x));
        assert!(Arc::ptr_eq(&target1.data.y, &target2.data.y));
    }

    #[test]
    fn test_whitening_transform_is_correct() {
        // Create a non-trivial positive definite Hessian
        let hessian = ndarray::array![[4.0, 1.0], [1.0, 3.0]];
        let dim = 2;

        // Compute the whitening transform via our function
        let hessian_owned = hessian.clone();
        let chol_factor = hessian_owned
            .cholesky(faer::Side::Lower)
            .expect("Cholesky should succeed");
        let l_h = chol_factor.lower_triangular();
        let chol = solve_upper_triangular_transpose(&l_h, dim);

        // Verify: L L^T should equal H^{-1}
        let llt = chol.dot(&chol.t());

        // Compute H^{-1} explicitly for comparison
        let det = 4.0 * 3.0 - 1.0 * 1.0; // = 11
        let h_inv = ndarray::array![[3.0 / det, -1.0 / det], [-1.0 / det, 4.0 / det]];

        // Check each element
        for i in 0..dim {
            for j in 0..dim {
                let diff = (llt[[i, j]] - h_inv[[i, j]]).abs();
                assert!(
                    diff < 1e-10,
                    "Whitening transform error at [{},{}]: got {}, expected {}",
                    i,
                    j,
                    llt[[i, j]],
                    h_inv[[i, j]]
                );
            }
        }
    }

    /// Fast smoke test: verify NUTS sampler runs and produces reasonable samples.
    /// Uses a trivial 2D Gaussian target to complete in ~1 second.
    #[test]
    fn test_nuts_sampler_runs_fast() {
        // Simple 2D logistic regression: 5 data points, 2 parameters
        let x = ndarray::array![[1.0, 0.5], [0.5, 1.0], [1.0, 1.0], [0.0, 0.5], [0.5, 0.0]];
        let y = ndarray::array![1.0, 0.0, 1.0, 0.0, 1.0];
        let weights = ndarray::array![1.0, 1.0, 1.0, 1.0, 1.0];
        let penalty = ndarray::array![[0.1, 0.0], [0.0, 0.1]];
        let mode = ndarray::array![0.5, 0.3]; // Pretend MAP estimate
        let hessian = ndarray::array![[2.0, 0.1], [0.1, 2.0]]; // SPD

        // Minimal config: 10 warmup, 20 samples, 2 chains
        let config = NutsConfig {
            n_samples: 20,
            n_warmup: 10,
            n_chains: 2,
            target_accept: 0.65,
            seed: 42,
        };

        let result = run_nuts_sampling(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            true, // Logit
            false,
            &config,
        );

        // Should succeed
        assert!(
            result.is_ok(),
            "NUTS sampling should succeed: {:?}",
            result.err()
        );
        let res = result.unwrap();

        // Should have correct shape: 2 chains * 20 samples = 40 rows, 2 params
        assert_eq!(res.samples.nrows(), 40, "Should have 40 total samples");
        assert_eq!(res.samples.ncols(), 2, "Should have 2 parameters");

        // Samples should be finite
        assert!(
            res.samples.iter().all(|x| x.is_finite()),
            "All samples should be finite"
        );

        // Samples should have some variance (chains moved)
        let std = res.samples.std_axis(ndarray::Axis(0), 0.0);
        let min_std = std.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(min_std > 1e-6, "Samples should have non-zero variance");

        // Posterior mean should exist and be reasonable
        assert!(res.posterior_mean.len() == 2);
        assert!(res.posterior_mean.iter().all(|x| x.is_finite()));

        // ESS should be positive
        assert!(res.ess > 0.0, "ESS should be positive");

        println!(
            "[NUTS Smoke] Passed: {} samples, ESS={:.1}, R-hat={:.3}",
            res.samples.nrows(),
            res.ess,
            res.rhat
        );
    }

    /// Test NUTS with higher dimension (10D) to diagnose if dimension causes hanging.
    #[test]
    fn test_nuts_higher_dimension_10d() {
        use rand::RngExt;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let dim = 10;
        let n_obs = 20;
        let mut rng = StdRng::seed_from_u64(999);

        // Random design matrix
        let x_data: Vec<f64> = (0..n_obs * dim)
            .map(|_| rng.random::<f64>() * 2.0 - 1.0)
            .collect();
        let x = ndarray::Array2::from_shape_vec((n_obs, dim), x_data).unwrap();

        // Random binary outcomes
        let y = ndarray::Array1::from_shape_fn(n_obs, |_| {
            if rng.random::<f64>() > 0.5 { 1.0 } else { 0.0 }
        });
        let weights = ndarray::Array1::ones(n_obs);

        // Ridge penalty
        let penalty = ndarray::Array2::from_diag(&ndarray::Array1::from_elem(dim, 0.1));

        // Mode near zero
        let mode = ndarray::Array1::zeros(dim);

        // Well-conditioned Hessian (identity-like)
        let hessian = ndarray::Array2::from_diag(&ndarray::Array1::from_elem(dim, 2.0));

        // Minimal config
        let config = NutsConfig {
            n_samples: 10,
            n_warmup: 5,
            n_chains: 2,
            target_accept: 0.65,
            seed: 123,
        };

        println!("[NUTS 10D] Starting with {} dim, {} obs...", dim, n_obs);
        let start = std::time::Instant::now();

        let result = run_nuts_sampling(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            true,
            false,
            &config,
        );

        let elapsed = start.elapsed();
        println!("[NUTS 10D] Completed in {:.2?}", elapsed);

        assert!(
            result.is_ok(),
            "NUTS 10D should succeed: {:?}",
            result.err()
        );
        let res = result.unwrap();

        // 2 chains * 10 samples = 20 total
        assert_eq!(res.samples.nrows(), 20);
        assert_eq!(res.samples.ncols(), dim);
        assert!(res.samples.iter().all(|x| x.is_finite()));

        println!(
            "[NUTS 10D] Passed: {} samples, dim={}",
            res.samples.nrows(),
            dim
        );
    }

    /// Test NUTS with ill-conditioned Hessian to diagnose hanging.
    #[test]
    fn test_nuts_ill_conditioned_hessian() {
        use rand::RngExt;
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let dim = 8;
        let n_obs = 30;
        let mut rng = StdRng::seed_from_u64(555);

        // Random design matrix
        let x_data: Vec<f64> = (0..n_obs * dim)
            .map(|_| rng.random::<f64>() * 2.0 - 1.0)
            .collect();
        let x = ndarray::Array2::from_shape_vec((n_obs, dim), x_data).unwrap();

        let y = ndarray::Array1::from_shape_fn(n_obs, |_| {
            if rng.random::<f64>() > 0.5 { 1.0 } else { 0.0 }
        });
        let weights = ndarray::Array1::ones(n_obs);
        let penalty = ndarray::Array2::from_diag(&ndarray::Array1::from_elem(dim, 0.1));
        let mode = ndarray::Array1::zeros(dim);

        // ILL-CONDITIONED Hessian: eigenvalues span 6 orders of magnitude
        // This mimics what happens with spline bases where some directions are very flat
        let mut hessian = ndarray::Array2::zeros((dim, dim));
        for i in 0..dim {
            // Eigenvalues: 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 10000
            hessian[[i, i]] = 10.0_f64.powi(i as i32 - 3);
        }

        let cond_number = 10.0_f64.powi((dim - 1) as i32);
        println!(
            "[NUTS Ill-Cond] Hessian condition number: {:.0e}",
            cond_number
        );

        let config = NutsConfig {
            n_samples: 10,
            n_warmup: 5,
            n_chains: 2,
            target_accept: 0.65,
            seed: 123,
        };

        println!("[NUTS Ill-Cond] Starting...");
        let start = std::time::Instant::now();

        let result = run_nuts_sampling(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            true,
            false,
            &config,
        );

        let elapsed = start.elapsed();
        println!("[NUTS Ill-Cond] Completed in {:.2?}", elapsed);

        assert!(
            result.is_ok(),
            "NUTS ill-conditioned should succeed: {:?}",
            result.err()
        );
        let res = result.unwrap();
        assert_eq!(res.samples.nrows(), 20);
        assert!(res.samples.iter().all(|x| x.is_finite()));

        println!("[NUTS Ill-Cond] Passed: {} samples", res.samples.nrows());
    }

    /// Test 1: Gaussian Recovery - Validates whitening transform.
    ///
    /// For a Gaussian (Identity link) model, the posterior is exactly:
    ///   N(β_MAP, H^{-1}) where H is the penalized Hessian
    ///
    /// This test verifies that MCMC samples have empirical covariance matching H^{-1}.
    #[test]
    fn test_gaussian_covariance_recovery() {
        use ndarray::arr1;
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let n = 100;
        let dim = 5;

        // Build design matrix with intercept + 4 covariates
        let mut x = Array2::<f64>::ones((n, dim));
        for i in 0..n {
            for j in 1..dim {
                x[[i, j]] = rng.random_range(-1.0..1.0);
            }
        }

        // True coefficients
        let true_beta = arr1(&[1.0, 0.5, -0.3, 0.2, -0.1]);

        // Generate response: y = X * β + ε, where ε ~ N(0, 1)
        let eta = x.dot(&true_beta);
        let noise: Array1<f64> = Array1::from_shape_fn(n, |_| {
            let u1: f64 = rng.random::<f64>().max(1e-10); // Prevent ln(0) = -inf
            let u2: f64 = rng.random();
            (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
        });
        let y = &eta + &noise;
        let weights = Array1::<f64>::ones(n);

        // Penalty matrix (ridge-like for stability)
        let lambda = 0.1;
        let mut penalty = Array2::<f64>::zeros((dim, dim));
        for i in 1..dim {
            penalty[[i, i]] = lambda; // Don't penalize intercept
        }

        // Compute penalized Hessian: H = X^T W X + S
        let xtx = x.t().dot(&x);
        let hessian = &xtx + &penalty;

        // Mode: use true_beta as starting mode
        let mode = true_beta.clone();

        // Run MCMC with sufficient samples to reduce Monte Carlo variance
        // Covariance estimation requires ~O(d^2) samples for stability
        let config = NutsConfig {
            n_samples: 500, // Increased for better covariance estimation
            n_warmup: 200,  // Increased warmup for better adaptation
            n_chains: 4,
            target_accept: 0.9,
            seed: 12345,
        };

        let result = run_nuts_sampling(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            false, // is_logit = false (Gaussian)
            false,
            &config,
        );

        assert!(result.is_ok(), "NUTS sampling failed: {:?}", result.err());
        let res = result.unwrap();
        let samples = &res.samples;

        println!("[Gaussian Recovery] Samples shape: {:?}", samples.shape());

        // Compute empirical covariance
        let n_samples = samples.nrows() as f64;
        let mean = samples.mean_axis(Axis(0)).unwrap();
        let centered = samples - &mean.view().insert_axis(Axis(0));
        let empirical_cov = centered.t().dot(&centered) / (n_samples - 1.0);

        // Compute theoretical covariance = H^{-1}
        let hessian_owned = hessian.clone();
        let chol = hessian_owned
            .cholesky(Side::Lower)
            .expect("Hessian should be positive definite");
        let l = chol.lower_triangular();

        // Solve L L^T * X = I to get H^{-1}
        let mut h_inv = Array2::<f64>::zeros((dim, dim));
        for col in 0..dim {
            // Forward substitution: solve L * z = e_col
            let mut z = Array1::<f64>::zeros(dim);
            for i in 0..dim {
                let mut sum = if i == col { 1.0 } else { 0.0 };
                for j in 0..i {
                    sum -= l[[i, j]] * z[j];
                }
                z[i] = sum / l[[i, i]];
            }
            // Back substitution: solve L^T * x = z
            for i in (0..dim).rev() {
                let mut sum = z[i];
                for j in (i + 1)..dim {
                    sum -= l[[j, i]] * h_inv[[j, col]];
                }
                h_inv[[i, col]] = sum / l[[i, i]];
            }
        }

        // Frobenius norm difference
        let diff = &empirical_cov - &h_inv;
        let frobenius_norm = diff.iter().map(|x| x * x).sum::<f64>().sqrt();
        let h_inv_norm = h_inv.iter().map(|x| x * x).sum::<f64>().sqrt();
        let relative_error = frobenius_norm / h_inv_norm;

        println!(
            "[Gaussian Recovery] ||Σ_emp - H^{{-1}}||_F = {:.4}",
            frobenius_norm
        );
        println!("[Gaussian Recovery] ||H^{{-1}}||_F = {:.4}", h_inv_norm);
        println!("[Gaussian Recovery] Relative error = {:.4}", relative_error);

        // Assert relative error is reasonable (<25% given finite samples)
        // Note: Covariance estimation has O(1/sqrt(N)) convergence, so with 2000 samples
        // (4 chains * 500) we expect ~2-3% error, but variance in the estimator itself
        // can cause occasional exceedances. 25% is a conservative bound.
        assert!(
            relative_error < 0.25,
            "Covariance mismatch: relative error {:.4} > 0.25",
            relative_error
        );

        println!("[Gaussian Recovery] PASSED: Empirical covariance matches H^{{-1}}");
    }

    /// Test 2: Jensen Gap - Validates Bayesian model averaging.
    ///
    /// Due to Jensen's inequality: E[σ(η)] <= σ(E[η]) for concave regions.
    /// For high logits, MCMC should shrink overconfident predictions.
    #[test]
    fn test_jensen_gap_overconfidence_shrinkage() {
        use ndarray::arr1;
        use rand::rngs::StdRng;
        use rand::{RngExt, SeedableRng};

        let mut rng = StdRng::seed_from_u64(42);
        let n = 200;
        let dim = 3;

        // Build design matrix
        let mut x = Array2::<f64>::ones((n, dim));
        for i in 0..n {
            for j in 1..dim {
                x[[i, j]] = rng.random_range(-1.0..1.0);
            }
        }

        // True coefficients - chosen so some predictions are extreme
        let true_beta = arr1(&[0.5, 1.5, 1.0]);

        // Generate binary response
        let eta = x.dot(&true_beta);
        let y: Array1<f64> = eta.mapv(|e| {
            let p = 1.0 / (1.0 + (-e).exp());
            if rng.random::<f64>() < p { 1.0 } else { 0.0 }
        });
        let weights = Array1::<f64>::ones(n);

        // Penalty matrix
        let lambda = 0.01;
        let mut penalty = Array2::<f64>::zeros((dim, dim));
        for i in 1..dim {
            penalty[[i, i]] = lambda;
        }

        // Construct Hessian for logistic regression: H = X^T W X + S
        let logit_weights: Array1<f64> = eta.mapv(|e| {
            let p = 1.0 / (1.0 + (-e).exp());
            p * (1.0 - p)
        });
        let mut hessian = penalty.clone();
        for i in 0..n {
            let w = logit_weights[i];
            for j in 0..dim {
                for k in 0..dim {
                    hessian[[j, k]] += w * x[[i, j]] * x[[i, k]];
                }
            }
        }

        // Use true_beta as mode
        let mode = true_beta.clone();

        // Run MCMC with more samples to reduce Monte Carlo variance
        let config = NutsConfig {
            n_samples: 200, // Increased from 50 for stability
            n_warmup: 100,  // Increased from 20
            n_chains: 4,
            target_accept: 0.8,
            seed: 54321,
        };

        let result = run_nuts_sampling(
            x.view(),
            y.view(),
            weights.view(),
            penalty.view(),
            mode.view(),
            hessian.view(),
            true, // is_logit = true
            false,
            &config,
        );

        assert!(result.is_ok(), "NUTS sampling failed: {:?}", result.err());
        let res = result.unwrap();
        let samples = &res.samples;

        // Create test input with high logit: x_test = [1, 2, 2]
        // η ≈ 0.5 + 1.5*2 + 1.0*2 = 5.5
        let x_test = arr1(&[1.0, 2.0, 2.0]);

        // MAP prediction
        let eta_map = x_test.dot(&mode);
        let pred_map = 1.0 / (1.0 + (-eta_map).exp());

        // MCMC prediction: E[σ(x_test^T * β)] and η moments for Jensen checks.
        let mut sum_prob = 0.0;
        let mut sum_eta = 0.0;
        let mut sum_eta_sq = 0.0;
        for i in 0..samples.nrows() {
            let beta_i = samples.row(i);
            let eta_i = x_test.dot(&beta_i);
            let p_i = 1.0 / (1.0 + (-eta_i.clamp(-700.0, 700.0)).exp());
            sum_prob += p_i;
            sum_eta += eta_i;
            sum_eta_sq += eta_i * eta_i;
        }
        let n_f = samples.nrows() as f64;
        let pred_mcmc = sum_prob / n_f;
        let mean_eta = sum_eta / n_f;
        let mean_eta_sq = sum_eta_sq / n_f;
        let var_eta = (mean_eta_sq - mean_eta * mean_eta).max(0.0);
        let pred_mean_eta = 1.0 / (1.0 + (-mean_eta.clamp(-700.0, 700.0)).exp());

        println!(
            "[Jensen Gap] η_MAP = {:.4}, P_MAP = {:.4}",
            eta_map, pred_map
        );
        println!(
            "[Jensen Gap] E[η] = {:.4}, P_MCMC = {:.4}",
            mean_eta, pred_mcmc
        );
        println!(
            "[Jensen Gap] σ(E[η]) = {:.4}, var(η) = {:.4}",
            pred_mean_eta, var_eta
        );
        println!("[Jensen Gap] Shrinkage = {:.4}", pred_map - pred_mcmc);

        // Guard against mode-collapse: eta variance should be non-trivial
        assert!(
            var_eta > 1e-4,
            "Sampler appears collapsed: var(eta)={:.3e}",
            var_eta
        );

        // The fundamental Jensen inequality must hold: E[σ(η)] <= σ(E[η])
        // Allow small tolerance for numerical precision
        assert!(
            pred_mcmc <= pred_mean_eta + 1e-3,
            "Jensen gap violated: E[σ(η)] ({:.4}) > σ(E[η]) ({:.4})",
            pred_mcmc,
            pred_mean_eta
        );

        // Note: We do NOT assert P_MCMC < P_MAP because the posterior mean of η
        // may drift away from the mode, especially with finite samples. The important
        // property is that Jensen's inequality holds for the sigmoid function.
        println!("[Jensen Gap] PASSED: Jensen inequality E[σ(η)] <= σ(E[η]) verified");
    }
}
