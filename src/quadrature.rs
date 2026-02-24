//! Gauss-Hermite Quadrature for Posterior Mean Predictions
//!
//! This module provides functions to compute the posterior mean of predictions
//! by integrating over the uncertainty in the linear predictor using
//! Gauss-Hermite quadrature.
//!
//! # Background
//!
//! Standard predictions return `g⁻¹(η̂)` where `η̂` is the point estimate (mode).
//! For curved link functions like logit or survival transforms, this differs from
//! the posterior mean `E[g⁻¹(η)]` where `η ~ N(η̂, σ²)`.
//!
//! The posterior mean:
//! - Minimizes squared prediction error (Brier score)
//! - Is more conservative at extreme predictions
//! - Accounts for parameter uncertainty in the final probability
//!
//! # Implementation
//!
//! We use 7-point Gauss-Hermite quadrature, which is exact for polynomials up
//! to degree 13. For the smooth link functions used in practice, this provides
//! excellent accuracy with minimal computational cost.
//!
//! The nodes and weights are computed at compile time using the Golub-Welsch
//! algorithm, which finds eigenvalues of the symmetric tridiagonal Jacobi matrix.
//!
//! # Key Assumptions and Limitations
//!
//! Gaussian linear predictor: GHQ assumes the linear predictor η follows a
//! Gaussian distribution. Under a multivariate normal posterior for β (from the
//! Hessian), any linear combination η = Xβ is exactly Gaussian. This assumption
//! is consistent with LAML (Laplace Approximate Marginal Likelihood) used for
//! smoothing parameter selection.
//!
//! Non-Gaussian risk output: GHQ does NOT assume the risk is Gaussian. It
//! correctly integrates through nonlinear link functions (sigmoid, survival
//! transforms) to capture skewed risk distributions.
//!
//! Survival sensitivity: For survival models with double-exponential transforms
//! (e.g., 1 - exp(-exp(η))), small differences in η are amplified in the tails.
//! At extreme horizons, this tail sensitivity means GHQ-based intervals may be
//! slightly underconfident. HMC would provide marginally more accurate tail
//! quantiles at significant computational cost.
//!
//! B-spline local support: At any evaluation point, only ~k+1 spline basis
//! functions are nonzero (typically 4 for cubic splines). However, the linear
//! predictor can include main and interaction effects, so
//! the total remains a sum of many terms.
//!
//! # Alternative: HMC
//!
//! For cases where the Gaussian assumption on η is questionable (very rare
//! diseases with <500 cases, extreme non-Gaussianity in the coefficient
//! posterior), Hamiltonian Monte Carlo could sample β directly and compute
//! risk for each sample. This is 100-1000x more expensive but makes no
//! distributional assumptions.

use std::sync::OnceLock;

/// Number of quadrature points (7-point rule is exact for polynomials up to degree 13)
const N_POINTS: usize = 7;

/// Quadrature context that owns Gauss-Hermite caches.
pub struct QuadratureContext {
    gh_cache: OnceLock<GaussHermiteRule>,
}

impl QuadratureContext {
    pub fn new() -> Self {
        Self {
            gh_cache: OnceLock::new(),
        }
    }

    fn gauss_hermite(&self) -> &GaussHermiteRule {
        self.gh_cache.get_or_init(compute_gauss_hermite)
    }
}

/// Gauss-Hermite quadrature rule: nodes and weights.
struct GaussHermiteRule {
    /// Quadrature nodes (roots of Hermite polynomial)
    nodes: [f64; N_POINTS],
    /// Quadrature weights (for physicist's Hermite, sum to sqrt(π))
    weights: [f64; N_POINTS],
}

/// Compute Gauss-Hermite quadrature nodes and weights using the Golub-Welsch algorithm.
///
/// The Golub-Welsch algorithm computes quadrature rules by finding the eigenvalues
/// and eigenvectors of the symmetric tridiagonal Jacobi matrix associated with
/// the orthogonal polynomial recurrence relation.
///
/// For physicist's Hermite polynomials Hₙ(x) with weight exp(-x²):
/// - Recurrence: Hₙ₊₁(x) = 2x·Hₙ(x) - 2n·Hₙ₋₁(x)
/// - Jacobi matrix has: diagonal = 0, off-diagonal[i] = sqrt(i/2) for i = 1..n
///
/// The nodes are the eigenvalues, and weights are derived from the first
/// component of each eigenvector.
fn compute_gauss_hermite() -> GaussHermiteRule {
    // Build symmetric tridiagonal Jacobi matrix for physicist's Hermite polynomials
    // For the recurrence aₙHₙ₊₁ = (x - bₙ)Hₙ - cₙHₙ₋₁ where cₙ = n/(2aₙ₋₁)
    // The Jacobi matrix has: J[i,i] = 0, J[i,i+1] = J[i+1,i] = sqrt((i+1)/2)

    let mut diag = [0.0f64; N_POINTS]; // All zeros for Hermite
    let mut off_diag = [0.0f64; N_POINTS - 1];

    for i in 0..(N_POINTS - 1) {
        // Off-diagonal: sqrt((i+1)/2) for physicist's Hermite
        off_diag[i] = (((i + 1) as f64) / 2.0).sqrt();
    }

    // Find eigenvalues and eigenvectors using symmetric tridiagonal QR algorithm
    // This is the implicit symmetric QR algorithm with Wilkinson shifts
    let (eigenvalues, eigenvectors) = symmetric_tridiagonal_eigen(&mut diag, &mut off_diag);

    // Nodes are the eigenvalues (sorted)
    let nodes = eigenvalues;
    let mut weights = [0.0f64; N_POINTS];

    // Weights: wᵢ = μ₀ * (first component of eigenvector)²
    // For physicist's Hermite: μ₀ = ∫exp(-x²)dx = sqrt(π)
    let mu0 = std::f64::consts::PI.sqrt();
    for i in 0..N_POINTS {
        let v0 = eigenvectors[i][0];
        weights[i] = mu0 * v0 * v0;
    }

    // Sort nodes (and corresponding weights) in ascending order
    let mut indices: [usize; N_POINTS] = [0, 1, 2, 3, 4, 5, 6];
    indices.sort_by(|&a, &b| nodes[a].partial_cmp(&nodes[b]).unwrap());

    let sorted_nodes: [f64; N_POINTS] = std::array::from_fn(|i| nodes[indices[i]]);
    let sorted_weights: [f64; N_POINTS] = std::array::from_fn(|i| weights[indices[i]]);

    GaussHermiteRule {
        nodes: sorted_nodes,
        weights: sorted_weights,
    }
}

/// Symmetric tridiagonal eigenvalue decomposition using implicit QR with Wilkinson shifts.
///
/// Returns (eigenvalues, eigenvectors) where eigenvectors[i] is the i-th eigenvector.
fn symmetric_tridiagonal_eigen(
    diag: &mut [f64; N_POINTS],
    off_diag: &mut [f64; N_POINTS - 1],
) -> ([f64; N_POINTS], [[f64; N_POINTS]; N_POINTS]) {
    // Initialize eigenvector matrix as identity
    let mut z: [[f64; N_POINTS]; N_POINTS] = [[0.0; N_POINTS]; N_POINTS];
    for i in 0..N_POINTS {
        z[i][i] = 1.0;
    }

    let eps = 1e-15;
    let max_iter = 100;

    // Work on successively smaller submatrices
    let mut n = N_POINTS;
    while n > 1 {
        // Check for convergence of last off-diagonal element
        for _ in 0..max_iter {
            // Find the largest unreduced block
            let mut m = n - 1;
            while m > 0 {
                if off_diag[m - 1].abs() <= eps * (diag[m - 1].abs() + diag[m].abs()) {
                    off_diag[m - 1] = 0.0;
                    break;
                }
                m -= 1;
            }

            if m == n - 1 {
                // Last element converged
                n -= 1;
                break;
            }

            // Wilkinson shift: eigenvalue of trailing 2x2 closer to diag[n-1]
            let d = (diag[n - 2] - diag[n - 1]) / 2.0;
            let e = off_diag[n - 2];
            let shift = diag[n - 1] - e * e / (d + d.signum() * (d * d + e * e).sqrt());

            // Implicit QR step with shift
            let mut x = diag[m] - shift;
            let mut y = off_diag[m];

            for k in m..(n - 1) {
                // Givens rotation to zero out y
                let (c, s) = if y.abs() > eps {
                    let r = (x * x + y * y).sqrt();
                    (x / r, -y / r)
                } else {
                    (1.0, 0.0)
                };

                // Apply rotation to tridiagonal matrix
                if k > m {
                    off_diag[k - 1] = (x * x + y * y).sqrt();
                }

                let d1 = diag[k];
                let d2 = diag[k + 1];
                let e_k = off_diag[k];

                diag[k] = c * c * d1 + s * s * d2 - 2.0 * c * s * e_k;
                diag[k + 1] = s * s * d1 + c * c * d2 + 2.0 * c * s * e_k;
                off_diag[k] = c * s * (d1 - d2) + (c * c - s * s) * e_k;

                if k < n - 2 {
                    x = off_diag[k];
                    y = -s * off_diag[k + 1];
                    off_diag[k + 1] *= c;
                }

                // Accumulate rotation into eigenvector matrix
                for i in 0..N_POINTS {
                    let t = z[k][i];
                    z[k][i] = c * t - s * z[k + 1][i];
                    z[k + 1][i] = s * t + c * z[k + 1][i];
                }
            }
        }
    }

    (*diag, z)
}

/// Computes the posterior mean probability for a logistic model using
/// Gauss-Hermite quadrature.
///
/// Given:
/// - `eta`: point estimate of linear predictor (log-odds)
/// - `se_eta`: standard error of eta (from Hessian)
///
/// Returns: E[sigmoid(η)] where η ~ N(eta, se_eta²)
///
/// When `se_eta` is zero or very small, this reduces to `sigmoid(eta)`.
#[inline]
pub fn logit_posterior_mean(ctx: &QuadratureContext, eta: f64, se_eta: f64) -> f64 {
    // If SE is negligible, return the mode (standard sigmoid)
    if se_eta < 1e-10 {
        return sigmoid(eta);
    }

    let gh = ctx.gauss_hermite();

    // Gauss-Hermite integration: E[f(η)] = ∫ f(η) φ(η) dη
    // Transform: η = eta + sqrt(2) * se_eta * x, where x ~ standard Hermite measure
    // This gives: E[f(η)] ≈ (1/sqrt(π)) Σᵢ wᵢ f(eta + sqrt(2) * se_eta * xᵢ)
    let scale = std::f64::consts::SQRT_2 * se_eta;
    let mut sum = 0.0;

    for i in 0..N_POINTS {
        let eta_i = eta + scale * gh.nodes[i];
        let prob_i = sigmoid(eta_i);
        sum += gh.weights[i] * prob_i;
    }

    // Normalize by sqrt(pi) since Hermite weights are for exp(-x²) measure
    let mean_prob = sum / std::f64::consts::PI.sqrt();

    // Clamp to valid probability range
    mean_prob.clamp(1e-10, 1.0 - 1e-10)
}

/// Computes the integrated probability AND its derivative with respect to eta.
///
/// For IRLS, we need both:
/// - μ = ∫ σ(η) × N(η; m, SE²) dη
/// - dμ/dm = ∫ σ'(η) × N(η; m, SE²) dη = ∫ σ(η)(1-σ(η)) × N(η; m, SE²) dη
///
/// Returns: (μ, dμ/dm)
#[inline]
pub fn logit_posterior_mean_with_deriv(
    ctx: &QuadratureContext,
    eta: f64,
    se_eta: f64,
) -> (f64, f64) {
    const MIN_DERIV: f64 = 1e-6;

    // If SE is negligible, return standard sigmoid and its derivative
    if se_eta < 1e-10 {
        let mu = sigmoid(eta);
        let dmu = (mu * (1.0 - mu)).max(MIN_DERIV);
        return (mu, dmu);
    }

    let gh = ctx.gauss_hermite();
    let scale = std::f64::consts::SQRT_2 * se_eta;
    let norm = 1.0 / std::f64::consts::PI.sqrt();

    let mut sum_mu = 0.0;
    let mut sum_dmu = 0.0;

    for i in 0..N_POINTS {
        let eta_i = eta + scale * gh.nodes[i];
        let prob_i = sigmoid(eta_i);
        let deriv_i = prob_i * (1.0 - prob_i); // σ'(η) = σ(η)(1-σ(η))

        sum_mu += gh.weights[i] * prob_i;
        sum_dmu += gh.weights[i] * deriv_i;
    }

    let mu = (sum_mu * norm).clamp(1e-10, 1.0 - 1e-10);
    let dmu = (sum_dmu * norm).max(MIN_DERIV);

    (mu, dmu)
}

/// Batch version of logit_posterior_mean_with_deriv.
/// Returns (mu_array, dmu_array)
pub fn logit_posterior_mean_with_deriv_batch(
    ctx: &QuadratureContext,
    eta: &ndarray::Array1<f64>,
    se_eta: &ndarray::Array1<f64>,
) -> (ndarray::Array1<f64>, ndarray::Array1<f64>) {
    let n = eta.len();
    let mut mu = ndarray::Array1::<f64>::zeros(n);
    let mut dmu = ndarray::Array1::<f64>::zeros(n);

    for i in 0..n {
        let (m, d) = logit_posterior_mean_with_deriv(ctx, eta[i], se_eta[i]);
        mu[i] = m;
        dmu[i] = d;
    }

    (mu, dmu)
}

/// Computes posterior mean probabilities for a batch of predictions.
///
/// This is the vectorized version of `logit_posterior_mean`.
pub fn logit_posterior_mean_batch(
    ctx: &QuadratureContext,
    eta: &ndarray::Array1<f64>,
    se_eta: &ndarray::Array1<f64>,
) -> ndarray::Array1<f64> {
    ndarray::Zip::from(eta)
        .and(se_eta)
        .map_collect(|&e, &se| logit_posterior_mean(ctx, e, se))
}

/// Standard sigmoid function with numerical stability.
#[inline]
fn sigmoid(x: f64) -> f64 {
    let x_clamped = x.clamp(-700.0, 700.0);
    1.0 / (1.0 + f64::exp(-x_clamped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_computed_nodes_symmetric() {
        // Verify computed nodes are symmetric around zero
        let ctx = QuadratureContext::new();
        let gh = ctx.gauss_hermite();
        for i in 0..N_POINTS / 2 {
            let j = N_POINTS - 1 - i;
            assert_relative_eq!(gh.nodes[i], -gh.nodes[j], epsilon = 1e-12);
        }
        // Middle node should be zero
        assert_relative_eq!(gh.nodes[N_POINTS / 2], 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_computed_weights_symmetric() {
        // Verify computed weights are symmetric
        let ctx = QuadratureContext::new();
        let gh = ctx.gauss_hermite();
        for i in 0..N_POINTS / 2 {
            let j = N_POINTS - 1 - i;
            assert_relative_eq!(gh.weights[i], gh.weights[j], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_weights_sum_to_sqrt_pi() {
        // Verify weights sum to sqrt(pi) for physicist's Hermite
        let ctx = QuadratureContext::new();
        let gh = ctx.gauss_hermite();
        let sum: f64 = gh.weights.iter().sum();
        assert_relative_eq!(sum, std::f64::consts::PI.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_zero_se_returns_mode() {
        // When SE is zero, posterior mean should equal mode
        let eta = 1.5;
        let se = 0.0;
        let ctx = QuadratureContext::new();
        let mean = logit_posterior_mean(&ctx, eta, se);
        let mode = sigmoid(eta);
        assert_relative_eq!(mean, mode, epsilon = 1e-10);
    }

    #[test]
    fn test_symmetric_at_zero() {
        // At eta=0 (50% probability), mean should still be ~50%
        let eta = 0.0;
        let se = 1.0;
        let ctx = QuadratureContext::new();
        let mean = logit_posterior_mean(&ctx, eta, se);
        // Due to symmetry of sigmoid around 0, mean ≈ mode
        assert_relative_eq!(mean, 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_shrinkage_at_extremes() {
        // At extreme eta, mean should be pulled toward 0.5
        let eta = 3.0; // mode = sigmoid(3) ≈ 0.953
        let se = 1.0;
        let ctx = QuadratureContext::new();
        let mean = logit_posterior_mean(&ctx, eta, se);
        let mode = sigmoid(eta);

        // Mean should be less than mode (shrunk toward 0.5)
        assert!(mean < mode, "Expected mean {} < mode {}", mean, mode);
        // But still reasonably high
        assert!(mean > 0.8, "Mean {} should still be high", mean);
    }

    #[test]
    fn test_matches_monte_carlo() {
        // Compare quadrature to Monte Carlo with many samples
        let eta = 2.0;
        let se = 0.8;

        let ctx = QuadratureContext::new();
        let quad_mean = logit_posterior_mean(&ctx, eta, se);

        // Monte Carlo with 100,000 samples
        let n_samples = 100_000;
        let mut mc_sum = 0.0;
        let mut rng_state = 12345u64; // Simple LCG for reproducibility
        for _ in 0..n_samples {
            // Box-Muller for normal samples
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((rng_state as f64) / (u64::MAX as f64)).max(1e-10); // Prevent ln(0)
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (rng_state as f64) / (u64::MAX as f64);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let eta_sample = eta + se * z;
            mc_sum += sigmoid(eta_sample);
        }
        let mc_mean = mc_sum / (n_samples as f64);

        // Should match within Monte Carlo sampling error (~0.01)
        assert_relative_eq!(quad_mean, mc_mean, epsilon = 0.01);
    }

    #[test]
    fn test_quadrature_integrates_x_squared() {
        // The quadrature should exactly integrate x² against exp(-x²)
        // ∫ x² exp(-x²) dx = sqrt(π)/2
        let ctx = QuadratureContext::new();
        let gh = ctx.gauss_hermite();
        let mut sum = 0.0;
        for i in 0..N_POINTS {
            sum += gh.weights[i] * gh.nodes[i] * gh.nodes[i];
        }
        let expected = std::f64::consts::PI.sqrt() / 2.0;
        assert_relative_eq!(sum, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_quadrature_integrates_x_fourth() {
        // The quadrature should exactly integrate x⁴ against exp(-x²)
        // ∫ x⁴ exp(-x²) dx = 3*sqrt(π)/4
        let ctx = QuadratureContext::new();
        let gh = ctx.gauss_hermite();
        let mut sum = 0.0;
        for i in 0..N_POINTS {
            let x = gh.nodes[i];
            sum += gh.weights[i] * x * x * x * x;
        }
        let expected = 3.0 * std::f64::consts::PI.sqrt() / 4.0;
        assert_relative_eq!(sum, expected, epsilon = 1e-10);
    }
}
