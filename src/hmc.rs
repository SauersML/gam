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
use crate::types::LikelihoodFamily;
use faer::Side;
use general_mcmc::generic_hmc::HamiltonianTarget;
use general_mcmc::generic_nuts::{GenericNUTS, MassMatrixAdaptation, NUTSMassMatrixConfig};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};
use polya_gamma::PolyaGamma;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use rand08::{SeedableRng as SeedableRng08, rngs::StdRng as StdRng08};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
    fn split_value(
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
                sum += split_value(samples, n_chains, half, dim, sc, t);
            }
            let mean = sum / n as f64;
            means[sc] = mean;
            let mut g0 = 0.0;
            for t in 0..n {
                let d = split_value(samples, n_chains, half, dim, sc, t) - mean;
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
                        let x0 = split_value(samples, n_chains, half, dim, sc, t);
                        let x1 = split_value(samples, n_chains, half, dim, sc, t + l);
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
    let mut chain_vars = vec![0.0_f64; n_split_chains];
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
            chain_vars[first_idx] = var1;

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
            chain_vars[second_idx] = var2;
        }

        // Within-chain variance W
        let w: f64 = chain_vars.iter().copied().sum::<f64>() / n_split_chains as f64;

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
        let grad_beta = &grad_ll_beta - &grad_penalty_beta;

        // === Step 6: Chain rule to get gradient in z space ===
        // ∇_z = L^T @ ∇_β
        let grad_z = self.chol_t.dot(&grad_beta);

        let logp = ll - penalty;

        (logp, grad_z)
    }

    /// Logistic regression log-likelihood and gradient.
    fn logit_logp_and_grad(&self, eta: &Array1<f64>) -> (f64, Array1<f64>) {
        let n = self.data.n_samples;
        let mut ll = 0.0;
        let mut residual = Array1::<f64>::zeros(n);

        for i in 0..n {
            let eta_i = eta[i];
            let y_i = self.data.y[i];
            let w_i = self.data.weights[i];
            // Use stable Bernoulli-logit log-likelihood directly:
            //   log p(y|eta) = y*eta - log(1 + exp(eta))
            // This keeps the target smooth and consistent with its gradient.
            ll += w_i * (y_i * eta_i - Self::log1pexp(eta_i));

            // Residual for gradient: y - μ (canonical link, score function)
            let mu = Self::sigmoid_stable(eta_i);
            residual[i] = w_i * (y_i - mu);
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

#[cfg(test)]
mod tests {
    use super::{
        FamilyNutsInputs, GlmFlatInputs, JointLinkPosterior, JointSplineArtifacts, NutsConfig,
        NutsPosterior, run_logit_polya_gamma_gibbs, run_nuts_sampling_flattened_family,
    };
    use crate::basis::{BasisOptions, Dense, KnotSource, create_basis};
    use crate::types::LikelihoodFamily;
    use ndarray::{Array2, array};

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
    fn nuts_logit_gradient_matches_finite_difference() {
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
        let (_logp, grad) = posterior.compute_logp_and_grad_nd(&z);

        let eps = 1e-6;
        for j in 0..z.len() {
            let mut z_plus = z.clone();
            let mut z_minus = z.clone();
            z_plus[j] += eps;
            z_minus[j] -= eps;
            let (lp, _) = posterior.compute_logp_and_grad_nd(&z_plus);
            let (lm, _) = posterior.compute_logp_and_grad_nd(&z_minus);
            let fd = (lp - lm) / (2.0 * eps);
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
    fn joint_link_uses_c1_extension_outside_knot_range() {
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
        let (b_eval, _eta) = posterior.evaluate_link(&u, &theta);

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
        let (_z_raw, z_c, rw) = posterior.standardized_z(&u);
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
            n_warmup: 30,
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
        let non_spd_hessian = array![[0.0, 0.0], [0.0, 0.0]];
        let cfg = NutsConfig {
            n_samples: 20,
            n_warmup: 20,
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
                hessian: non_spd_hessian.view(),
                firth_bias_reduction: false,
            }),
            &cfg,
        )
        .expect("dispatch should use PG Gibbs and not require Hessian factorization");
        assert_eq!(out.samples.nrows(), cfg.n_samples * cfg.n_chains);
        assert!(out.samples.iter().all(|v| v.is_finite()));
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

fn robust_mass_matrix_config(dim: usize, n_warmup: usize) -> NUTSMassMatrixConfig {
    if n_warmup < 80 {
        return NUTSMassMatrixConfig::disabled();
    }
    let start_buffer = (n_warmup / 10).clamp(25, 150);
    let end_buffer = (n_warmup / 8).clamp(25, 150);
    let initial_window = (n_warmup / 12).clamp(20, 120);
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

fn robust_survival_mass_matrix_config(dim: usize, n_warmup: usize) -> NUTSMassMatrixConfig {
    if n_warmup < 80 {
        return NUTSMassMatrixConfig::disabled();
    }
    // Survival posteriors with censoring/rare events are often skewed; this
    // configuration uses diagonal adaptation.
    let start_buffer = (n_warmup / 8).clamp(30, 180);
    let end_buffer = (n_warmup / 6).clamp(30, 180);
    let initial_window = (n_warmup / 10).clamp(25, 140);
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

    let n_iter = config.n_warmup + config.n_samples;
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

    for chain in 0..config.n_chains {
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

            // Build X_weighted = diag(sqrt(ω)) X and compute X^T Ω X via faer GEMM.
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
            mean.assign(&factor.solve_vec(&rhs_b));

            for j in 0..p {
                z[j] = sample_standard_normal(&mut rng);
            }
            let l = factor.lower_triangular();
            forward_solve_lower_triangular(&l, &z, &mut noise);
            beta.assign(&(&mean + &noise));

            if iter >= config.n_warmup {
                let keep_idx = iter - config.n_warmup;
                samples_array
                    .slice_mut(ndarray::s![chain, keep_idx, ..])
                    .assign(&beta);
            }
        }
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

    Ok(NutsResult {
        samples,
        posterior_mean,
        posterior_std,
        rhat,
        ess,
        converged,
    })
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
    if is_logit && firth_bias_reduction {
        return Err(
            "NUTS with Firth bias reduction is disabled: posterior target mismatch. Refit with firth_bias_reduction=false for consistent HMC uncertainty."
                .to_string(),
        );
    }
    let dim = mode.len();

    // Create posterior target with analytical gradients.
    let target = NutsPosterior::new(x, y, weights, penalty_matrix, mode, hessian, is_logit)?;

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
    let mass_cfg = robust_mass_matrix_config(dim, config.n_warmup);
    let mut sampler = GenericNUTS::new_with_mass_matrix(
        target,
        initial_positions,
        config.target_accept,
        mass_cfg,
    );

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
                    true,
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
            "RoystonParmar family requires FamilyNutsInputs::Survival flattened inputs".to_string(),
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

            // Survival WorkingState follows the same engine contract as GAM:
            //   deviance = -2 log-likelihood (without quadratic penalties)
            //   penalty_term = beta' S beta (+ stabilization ridge term)
            //   gradient = d/dβ [0.5 * (deviance + penalty_term)].
            let logp = -0.5 * (state.deviance + state.penalty_term);
            let grad_beta = state.gradient.mapv(|g| -g);

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
        let mass_cfg = robust_survival_mass_matrix_config(dim, config.n_warmup);
        let mut sampler = GenericNUTS::new_with_mass_matrix(
            target,
            initial_positions,
            config.target_accept,
            mass_cfg,
        );

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

        let (z_raw, z_c, _rw) = self.standardized_z(u);
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
        let (_z_raw, z_c, rw) = self.standardized_z(u);
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
        let d_wiggle_dz = b_prime_constrained.dot(theta);
        for i in 0..n {
            g[i] = 1.0 + d_wiggle_dz[i] / rw;
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
    let mass_cfg = robust_mass_matrix_config(dim, config.n_warmup);
    let mut sampler = GenericNUTS::new_with_mass_matrix(
        target,
        initial_positions,
        config.target_accept,
        mass_cfg,
    );
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

/// Compute split R-hat and autocorrelation-based ESS diagnostics.
fn compute_rhat_ess(
    samples: &Array3<f64>,
    _n_chains: usize,
    _n_samples: usize,
    _dim: usize,
) -> (f64, f64) {
    compute_split_rhat_and_ess(samples)
}
