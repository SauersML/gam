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
//! - Is more conservative at extreme predictions
//! - Accounts for parameter uncertainty in the final probability
//!
//! # Implementation
//!
//! We use Gauss-Hermite quadrature with adaptive node counts (7/15/21 points)
//! based on latent uncertainty scale. This preserves speed in well-identified
//! regions and improves tail accuracy for high-variance nonlinear transforms.
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
//!
//! Practical scope of "exact" special-function formulas:
//! - Logistic-normal mean/variance can be written exactly with Faddeeva-series
//!   representations and are useful as oracle references.
//! - These formulas are mathematically exact representations distinct from the
//!   GHQ-based moment computations used elsewhere in this module.

use std::sync::OnceLock;

/// Number of quadrature points (7-point rule is exact for polynomials up to degree 13)
const N_POINTS: usize = 7;
const SQRT_2: f64 = std::f64::consts::SQRT_2;

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
}

/// Quadrature context that owns Gauss-Hermite caches.
pub struct QuadratureContext {
    gh_cache: OnceLock<GaussHermiteRule>,
    gh15_cache: OnceLock<GaussHermiteRuleDynamic>,
    gh21_cache: OnceLock<GaussHermiteRuleDynamic>,
    gh31_cache: OnceLock<GaussHermiteRuleDynamic>,
}

impl QuadratureContext {
    pub fn new() -> Self {
        Self {
            gh_cache: OnceLock::new(),
            gh15_cache: OnceLock::new(),
            gh21_cache: OnceLock::new(),
            gh31_cache: OnceLock::new(),
        }
    }

    fn gauss_hermite(&self) -> &GaussHermiteRule {
        self.gh_cache.get_or_init(compute_gauss_hermite)
    }

    fn gauss_hermite_n(&self, n: usize) -> &GaussHermiteRuleDynamic {
        match n {
            7 => unreachable!("7-point rule uses fixed cache"),
            15 => self.gh15_cache.get_or_init(|| compute_gauss_hermite_n(15)),
            21 => self.gh21_cache.get_or_init(|| compute_gauss_hermite_n(21)),
            31 => self.gh31_cache.get_or_init(|| compute_gauss_hermite_n(31)),
            _ => self.gh21_cache.get_or_init(|| compute_gauss_hermite_n(21)),
        }
    }
}

impl Default for QuadratureContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Gauss-Hermite quadrature rule: nodes and weights.
struct GaussHermiteRule {
    /// Quadrature nodes (roots of Hermite polynomial)
    nodes: [f64; N_POINTS],
    /// Quadrature weights (for physicist's Hermite, sum to sqrt(π))
    weights: [f64; N_POINTS],
}

struct GaussHermiteRuleDynamic {
    nodes: Vec<f64>,
    weights: Vec<f64>,
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

    // Weights: wᵢ = μ₀ * (first component of eigenvector)².
    // `symmetric_tridiagonal_eigen` accumulates left rotations and returns Z = Q^T,
    // so q_{0i} is stored at eigenvectors[i][0].
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

fn compute_gauss_hermite_n(n: usize) -> GaussHermiteRuleDynamic {
    let mut diag = vec![0.0f64; n];
    let mut off_diag = vec![0.0f64; n.saturating_sub(1)];
    for (i, od) in off_diag.iter_mut().enumerate() {
        *od = (((i + 1) as f64) / 2.0).sqrt();
    }
    let (nodes, eigenvectors) = symmetric_tridiagonal_eigen_dynamic(&mut diag, &mut off_diag);
    let mu0 = std::f64::consts::PI.sqrt();
    let mut pairs = (0..n)
        .map(|i| {
            let v0 = eigenvectors[i][0];
            (nodes[i], mu0 * v0 * v0)
        })
        .collect::<Vec<_>>();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    GaussHermiteRuleDynamic {
        nodes: pairs.iter().map(|p| p.0).collect(),
        weights: pairs.iter().map(|p| p.1).collect(),
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
        let mut converged = false;
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
                converged = true;
                break;
            }

            // Wilkinson shift: eigenvalue of trailing 2x2 closer to diag[n-1].
            // Use sign(0)=+1 (not f64::signum) to avoid zero denominator when d=0.
            let shift = wilkinson_shift(diag[n - 2], diag[n - 1], off_diag[n - 2]);

            // Implicit QR step with shift
            let mut x = diag[m] - shift;
            let mut y = off_diag[m];

            for k in m..(n - 1) {
                // Givens rotation to zero out y
                let (c, s) = if y.abs() > eps {
                    let r = x.hypot(y);
                    if r > 0.0 && r.is_finite() {
                        (x / r, -y / r)
                    } else {
                        (1.0, 0.0)
                    }
                } else {
                    (1.0, 0.0)
                };

                // Apply rotation to tridiagonal matrix
                if k > m {
                    off_diag[k - 1] = x.hypot(y);
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
        if !converged {
            // Guaranteed progress fallback: force trailing deflation if QR did not
            // converge within max_iter. For our tiny fixed-size Jacobi matrices this
            // is extremely rare and avoids a potential infinite loop.
            off_diag[n - 2] = 0.0;
            n -= 1;
        }
    }

    (*diag, z)
}

#[allow(dead_code)]
fn symmetric_tridiagonal_eigen_dynamic(
    diag: &mut [f64],
    off_diag: &mut [f64],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let dim = diag.len();
    let mut z = vec![vec![0.0_f64; dim]; dim];
    for (i, row) in z.iter_mut().enumerate().take(dim) {
        row[i] = 1.0;
    }
    let eps = 1e-15;
    let max_iter = 200usize;
    let mut n = dim;
    while n > 1 {
        let mut converged = false;
        for _ in 0..max_iter {
            let mut m = n - 1;
            while m > 0 {
                if off_diag[m - 1].abs() <= eps * (diag[m - 1].abs() + diag[m].abs()) {
                    off_diag[m - 1] = 0.0;
                    break;
                }
                m -= 1;
            }
            if m == n - 1 {
                n -= 1;
                converged = true;
                break;
            }
            let shift = wilkinson_shift(diag[n - 2], diag[n - 1], off_diag[n - 2]);
            let mut x = diag[m] - shift;
            let mut y = off_diag[m];
            for k in m..(n - 1) {
                let (c, s) = if y.abs() > eps {
                    let r = x.hypot(y);
                    if r > 0.0 && r.is_finite() {
                        (x / r, -y / r)
                    } else {
                        (1.0, 0.0)
                    }
                } else {
                    (1.0, 0.0)
                };
                if k > m {
                    off_diag[k - 1] = x.hypot(y);
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
                for i in 0..dim {
                    let t = z[k][i];
                    z[k][i] = c * t - s * z[k + 1][i];
                    z[k + 1][i] = s * t + c * z[k + 1][i];
                }
            }
        }
        if !converged {
            off_diag[n - 2] = 0.0;
            n -= 1;
        }
    }
    (diag.to_vec(), z)
}

#[inline]
fn wilkinson_shift(a: f64, c: f64, b: f64) -> f64 {
    let d = (a - c) * 0.5;
    let t = d.hypot(b);
    let sgn = if d >= 0.0 { 1.0 } else { -1.0 }; // sign(0)=+1
    let denom = d + sgn * t;

    if denom.abs() > f64::EPSILON * t.max(1.0) {
        c - (b * b) / denom
    } else {
        // Degenerate fallback: equivalent limiting shift when denominator collapses.
        c - t
    }
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
    integrate_normal_ghq_adaptive(ctx, eta, se_eta, sigmoid).clamp(1e-10, 1.0 - 1e-10)
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
    // If SE is negligible, return standard sigmoid and its derivative
    if se_eta < 1e-10 {
        let mu = sigmoid(eta);
        let dmu = mu * (1.0 - mu);
        return (mu, dmu);
    }

    let n = adaptive_point_count_from_sd(se_eta.abs());
    with_gh_nodes_weights(ctx, n, |nodes, weights| {
        let scale = std::f64::consts::SQRT_2 * se_eta;
        let norm = 1.0 / std::f64::consts::PI.sqrt();
        let mut sum_mu = 0.0;
        let mut sum_dmu = 0.0;
        for i in 0..n {
            let eta_i = eta + scale * nodes[i];
            let prob_i = sigmoid(eta_i);
            let deriv_i = prob_i * (1.0 - prob_i);
            sum_mu += weights[i] * prob_i;
            sum_dmu += weights[i] * deriv_i;
        }
        let mu = (sum_mu * norm).clamp(1e-10, 1.0 - 1e-10);
        // dmu can be arbitrarily close to 0 in the tails; do not floor it.
        let dmu = (sum_dmu * norm).max(0.0);
        (mu, dmu)
    })
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

#[inline]
fn integrate_normal_ghq_adaptive<F>(ctx: &QuadratureContext, eta: f64, se_eta: f64, f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    if se_eta < 1e-10 {
        return f(eta);
    }
    let n = adaptive_point_count_from_sd(se_eta.abs());
    with_gh_nodes_weights(ctx, n, |nodes, weights| {
        let scale = SQRT_2 * se_eta;
        let mut sum = 0.0;
        for i in 0..n {
            sum += weights[i] * f(eta + scale * nodes[i]);
        }
        sum / std::f64::consts::PI.sqrt()
    })
}

#[inline]
pub fn normal_expectation_1d_adaptive<F>(
    ctx: &QuadratureContext,
    eta: f64,
    se_eta: f64,
    f: F,
) -> f64
where
    F: Fn(f64) -> f64,
{
    integrate_normal_ghq_adaptive(ctx, eta, se_eta, f)
}

fn adaptive_point_count_from_sd(max_sd: f64) -> usize {
    // Use a more aggressive schedule for nonlinear tail-sensitive transforms.
    // 7 points stays for very well-identified rows, 15/21 kick in earlier for
    // location-scale and rare-event regimes where MC checks showed larger error.
    if max_sd.is_finite() && max_sd > 2.5 {
        31
    } else if max_sd.is_finite() && max_sd > 1.0 {
        21
    } else if max_sd.is_finite() && max_sd > 0.35 {
        15
    } else {
        7
    }
}

#[inline]
fn with_gh_nodes_weights<R>(
    ctx: &QuadratureContext,
    n: usize,
    f: impl FnOnce(&[f64], &[f64]) -> R,
) -> R {
    if n == 7 {
        let gh = ctx.gauss_hermite();
        f(&gh.nodes, &gh.weights)
    } else {
        let gh = ctx.gauss_hermite_n(n);
        f(&gh.nodes, &gh.weights)
    }
}

fn cholesky_with_jitter(cov: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = cov.len();
    if n == 0 || cov.iter().any(|r| r.len() != n) {
        return None;
    }
    let mut base = cov.to_vec();
    for retry in 0..8 {
        let jitter = if retry == 0 {
            0.0
        } else {
            1e-12 * 10f64.powi(retry - 1)
        };
        if jitter > 0.0 {
            for i in 0..n {
                base[i][i] = cov[i][i] + jitter;
            }
        }
        let mut l = vec![vec![0.0_f64; n]; n];
        let mut ok = true;
        for i in 0..n {
            for j in 0..=i {
                let mut sum = base[i][j];
                for k in 0..j {
                    sum -= l[i][k] * l[j][k];
                }
                if i == j {
                    if !sum.is_finite() || sum <= 0.0 {
                        ok = false;
                        break;
                    }
                    l[i][j] = sum.sqrt();
                } else {
                    l[i][j] = sum / l[j][j];
                }
            }
            if !ok {
                break;
            }
        }
        if ok {
            return Some(l);
        }
    }
    None
}

/// Adaptive 2D GHQ expectation for correlated Gaussian latents.
pub fn normal_expectation_2d_adaptive<F>(
    ctx: &QuadratureContext,
    mu: [f64; 2],
    cov: [[f64; 2]; 2],
    f: F,
) -> f64
where
    F: Fn(f64, f64) -> f64,
{
    let max_sd = cov[0][0].max(cov[1][1]).max(0.0).sqrt();
    let n = adaptive_point_count_from_sd(max_sd).min(21);
    let cov_vec = vec![
        vec![cov[0][0].max(0.0), cov[0][1]],
        vec![cov[1][0], cov[1][1].max(0.0)],
    ];
    let l = match cholesky_with_jitter(&cov_vec) {
        Some(v) => v,
        None => return f(mu[0], mu[1]),
    };
    let norm = 1.0 / std::f64::consts::PI;
    with_gh_nodes_weights(ctx, n, |nodes, weights| {
        let mut acc = 0.0;
        for i in 0..n {
            for j in 0..n {
                let z1 = SQRT_2 * nodes[i];
                let z2 = SQRT_2 * nodes[j];
                let x0 = mu[0] + l[0][0] * z1;
                let x1 = mu[1] + l[1][0] * z1 + l[1][1] * z2;
                acc += weights[i] * weights[j] * f(x0, x1);
            }
        }
        acc * norm
    })
}

/// Adaptive 2D GHQ expectation for correlated Gaussian latents with a fallible integrand.
pub fn normal_expectation_2d_adaptive_result<F, E>(
    ctx: &QuadratureContext,
    mu: [f64; 2],
    cov: [[f64; 2]; 2],
    f: F,
) -> Result<f64, E>
where
    F: Fn(f64, f64) -> Result<f64, E>,
{
    let max_sd = cov[0][0].max(cov[1][1]).max(0.0).sqrt();
    let n = adaptive_point_count_from_sd(max_sd).min(21);
    let cov_vec = vec![
        vec![cov[0][0].max(0.0), cov[0][1]],
        vec![cov[1][0], cov[1][1].max(0.0)],
    ];
    let l = match cholesky_with_jitter(&cov_vec) {
        Some(v) => v,
        None => return f(mu[0], mu[1]),
    };
    let norm = 1.0 / std::f64::consts::PI;
    with_gh_nodes_weights(ctx, n, |nodes, weights| {
        let mut acc = 0.0;
        for i in 0..n {
            for j in 0..n {
                let z1 = SQRT_2 * nodes[i];
                let z2 = SQRT_2 * nodes[j];
                let x0 = mu[0] + l[0][0] * z1;
                let x1 = mu[1] + l[1][0] * z1 + l[1][1] * z2;
                acc += weights[i] * weights[j] * f(x0, x1)?;
            }
        }
        Ok(acc * norm)
    })
}

/// Adaptive 3D GHQ expectation for correlated Gaussian latents.
pub fn normal_expectation_3d_adaptive<F>(
    ctx: &QuadratureContext,
    mu: [f64; 3],
    cov: [[f64; 3]; 3],
    f: F,
) -> f64
where
    F: Fn(f64, f64, f64) -> f64,
{
    let max_sd = cov[0][0].max(cov[1][1]).max(cov[2][2]).max(0.0).sqrt();
    // 3D tensor GHQ grows cubically; cap nodes per axis for throughput.
    let n = adaptive_point_count_from_sd(max_sd).min(15);
    let cov_vec = vec![
        vec![cov[0][0].max(0.0), cov[0][1], cov[0][2]],
        vec![cov[1][0], cov[1][1].max(0.0), cov[1][2]],
        vec![cov[2][0], cov[2][1], cov[2][2].max(0.0)],
    ];
    let l = match cholesky_with_jitter(&cov_vec) {
        Some(v) => v,
        None => return f(mu[0], mu[1], mu[2]),
    };
    let norm = 1.0 / std::f64::consts::PI.powf(1.5);
    with_gh_nodes_weights(ctx, n, |nodes, weights| {
        let mut acc = 0.0;
        for i in 0..n {
            for j in 0..n {
                for k in 0..n {
                    let z0 = SQRT_2 * nodes[i];
                    let z1 = SQRT_2 * nodes[j];
                    let z2 = SQRT_2 * nodes[k];
                    let x0 = mu[0] + l[0][0] * z0;
                    let x1 = mu[1] + l[1][0] * z0 + l[1][1] * z1;
                    let x2 = mu[2] + l[2][0] * z0 + l[2][1] * z1 + l[2][2] * z2;
                    acc += weights[i] * weights[j] * weights[k] * f(x0, x1, x2);
                }
            }
        }
        acc * norm
    })
}

/// Closed-form posterior mean under probit link when eta is Gaussian:
/// E[Phi(Z)] for Z ~ N(eta, se_eta^2) = Phi(eta / sqrt(1 + se_eta^2)).
#[inline]
pub fn probit_posterior_mean(eta: f64, se_eta: f64) -> f64 {
    if se_eta < 1e-10 {
        return crate::probability::normal_cdf_approx(eta).clamp(1e-10, 1.0 - 1e-10);
    }
    let denom = (1.0 + se_eta * se_eta).sqrt();
    crate::probability::normal_cdf_approx(eta / denom).clamp(1e-10, 1.0 - 1e-10)
}

#[inline]
pub fn logit_posterior_mean_variance(ctx: &QuadratureContext, eta: f64, se_eta: f64) -> (f64, f64) {
    let m1 = integrate_normal_ghq_adaptive(ctx, eta, se_eta, sigmoid).clamp(1e-10, 1.0 - 1e-10);
    let m2 = integrate_normal_ghq_adaptive(ctx, eta, se_eta, |x| {
        let p = sigmoid(x);
        p * p
    })
    .clamp(0.0, 1.0);
    (m1, (m2 - m1 * m1).max(0.0))
}

#[inline]
pub fn probit_posterior_mean_variance(
    ctx: &QuadratureContext,
    eta: f64,
    se_eta: f64,
) -> (f64, f64) {
    let m1 = probit_posterior_mean(eta, se_eta);
    let m2 = integrate_normal_ghq_adaptive(ctx, eta, se_eta, |x| {
        let p = crate::probability::normal_cdf_approx(x).clamp(1e-10, 1.0 - 1e-10);
        p * p
    })
    .clamp(0.0, 1.0);
    (m1, (m2 - m1 * m1).max(0.0))
}

#[inline]
pub fn cloglog_posterior_mean_variance(
    ctx: &QuadratureContext,
    eta: f64,
    se_eta: f64,
) -> (f64, f64) {
    let m1 = cloglog_posterior_mean(ctx, eta, se_eta);
    let m2 = integrate_normal_ghq_adaptive(ctx, eta, se_eta, |x| {
        let z = x.clamp(-30.0, 30.0);
        let p = (1.0 - (-(z.exp())).exp()).clamp(1e-10, 1.0 - 1e-10);
        p * p
    })
    .clamp(0.0, 1.0);
    (m1, (m2 - m1 * m1).max(0.0))
}

/// Posterior mean under cloglog inverse link:
/// g^{-1}(x) = 1 - exp(-exp(x)).
#[inline]
pub fn cloglog_posterior_mean(ctx: &QuadratureContext, eta: f64, se_eta: f64) -> f64 {
    integrate_normal_ghq_adaptive(ctx, eta, se_eta, |x| {
        let z = x.clamp(-30.0, 30.0);
        1.0 - (-(z.exp())).exp()
    })
    .clamp(1e-10, 1.0 - 1e-10)
}

/// Posterior mean under the Royston-Parmar survival transform:
/// S(x) = exp(-exp(x)).
#[inline]
pub fn survival_posterior_mean(ctx: &QuadratureContext, eta: f64, se_eta: f64) -> f64 {
    integrate_normal_ghq_adaptive(ctx, eta, se_eta, |x| {
        let z = x.clamp(-30.0, 30.0);
        (-(z.exp())).exp()
    })
    .clamp(0.0, 1.0)
}

#[inline]
pub fn survival_posterior_mean_variance(
    ctx: &QuadratureContext,
    eta: f64,
    se_eta: f64,
) -> (f64, f64) {
    let m1 = survival_posterior_mean(ctx, eta, se_eta).clamp(0.0, 1.0);
    let m2 = integrate_normal_ghq_adaptive(ctx, eta, se_eta, |x| {
        let z = x.clamp(-30.0, 30.0);
        let p = (-(z.exp())).exp().clamp(0.0, 1.0);
        p * p
    })
    .clamp(0.0, 1.0);
    (m1, (m2 - m1 * m1).max(0.0))
}

/// Oracle-only exact logistic-normal mean using the Faddeeva-series representation.
///
/// For η ~ N(mu, sigma^2):
///   E[sigmoid(η)] = 1/2 - (sqrt(2π)/sigma) * Σ_{n>=1} Im[w((i a_n - mu)/(sqrt(2)sigma))]
/// where a_n = (2n-1)π and w is the Faddeeva function.
///
/// Deterministic integration-free equivalent representation:
/// Let m=mu, s=sigma, and erfcx(x)=exp(x^2)erfc(x). Then an exact convergent form is
///
///   E[sigmoid(η)]
///   = Φ(m/s)
///     + 0.5 * exp(-m^2/(2s^2))
///       * Σ_{k>=1} (-1)^{k-1}
///         [ erfcx((k s^2 + m)/(sqrt(2)s)) - erfcx((k s^2 - m)/(sqrt(2)s)) ].
///
/// A matching exact second-moment representation uses the same erfcx building
/// blocks U_k/V_k plus the boundary term -φ_{m,s}(0):
///
///   U_k(m,s) = 0.5 * exp(-m^2/(2s^2)) * erfcx((k s^2 - m)/(sqrt(2)s))
///   V_k(m,s) = 0.5 * exp(-m^2/(2s^2)) * erfcx((k s^2 + m)/(sqrt(2)s))
///
///   E[sigmoid(η)^2]
///     = Φ(m/s)
///       + Σ_{k>=1} (k+1)(-1)^k U_k
///       + Σ_{k>=2} (k-1)(-1)^k V_k
///       - φ_{m,s}(0),
///
/// and therefore
///
///   Var(sigmoid(η)) = E[sigmoid(η)^2] - E[sigmoid(η)]^2.
///
/// Derivation sketch:
/// 1) sigmoid(t) = 1/2 + 1/2 tanh(t/2)
/// 2) tanh has a partial-fraction expansion over odd poles ±i(2n-1)π
/// 3) Taking Gaussian expectations termwise yields rational expectations of the form
///    E[ 1 / (Z - i a_n) ], Z~N(mu,sigma^2)
/// 4) Those are exactly representable by the Faddeeva function:
///    E[ 1 / (Z - i a) ] = i*sqrt(pi)/(sqrt(2)*sigma) * w((i a - mu)/(sqrt(2)*sigma))
/// 5) Taking imaginary parts and summing odd a_n gives the stated series.
///
/// Therefore this routine is mathematically exact up to numerical truncation and
/// numerical evaluation error of w(z).
pub fn logit_posterior_mean_exact(mu: f64, sigma: f64) -> f64 {
    if !(mu.is_finite() && sigma.is_finite()) || sigma <= 0.0 {
        return sigmoid(mu).clamp(1e-12, 1.0 - 1e-12);
    }
    if sigma < 1e-10 {
        return sigmoid(mu).clamp(1e-12, 1.0 - 1e-12);
    }

    let sqrt2_sigma = SQRT_2 * sigma;
    let coeff = (2.0_f64 * std::f64::consts::PI).sqrt() / sigma;
    let mut sum_im = 0.0_f64;
    let mut stable_small_terms = 0usize;

    for n in 1..=4096usize {
        let a_n = (2.0 * (n as f64) - 1.0) * std::f64::consts::PI;
        let z = Complex {
            re: -mu / sqrt2_sigma,
            im: a_n / sqrt2_sigma, // strictly positive
        };
        let w = faddeeva_upper_halfplane(z);
        if !w.im.is_finite() {
            break;
        }
        let term = w.im;
        sum_im += term;

        let contrib = (coeff * term).abs();
        if contrib < 1e-14 {
            stable_small_terms += 1;
            if stable_small_terms >= 8 {
                break;
            }
        } else {
            stable_small_terms = 0;
        }
    }

    (0.5 - coeff * sum_im).clamp(1e-12, 1.0 - 1e-12)
}

/// Faddeeva function w(z)=exp(-z^2)erfc(-iz) for Im(z)>0.
///
/// Uses the Cauchy-type integral representation:
///   w(z) = (i/π) ∫ exp(-t^2)/(z-t) dt,  t in R, Im(z)>0.
///
/// Writing z=x+iy (y>0), this gives:
///   Re w(z) = (1/π) ∫ exp(-t^2) * y / ((x-t)^2 + y^2) dt
///   Im w(z) = (1/π) ∫ exp(-t^2) * (x-t) / ((x-t)^2 + y^2) dt
///
/// We evaluate both with composite Simpson's rule.
fn faddeeva_upper_halfplane(z: Complex) -> Complex {
    let x = z.re;
    let y = z.im.max(1e-12);
    let span = (x.abs() + 10.0).max(14.0);
    let a = -span;
    let b = span;
    let n = 4000usize; // even
    let h = (b - a) / (n as f64);

    let eval = |t: f64| -> Complex {
        let u = x - t;
        let den = (u * u + y * y).max(1e-300);
        let e = (-t * t).exp();
        // i/(z-t) = y/den + i*u/den
        Complex {
            re: e * y / den,
            im: e * u / den,
        }
    };

    let mut s_re = 0.0_f64;
    let mut s_im = 0.0_f64;
    let f0 = eval(a);
    let fn_ = eval(b);
    s_re += f0.re + fn_.re;
    s_im += f0.im + fn_.im;

    for i in 1..n {
        let t = a + (i as f64) * h;
        let f = eval(t);
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        s_re += w * f.re;
        s_im += w * f.im;
    }
    let scale = (h / 3.0) / std::f64::consts::PI;
    Complex {
        re: s_re * scale,
        im: s_im * scale,
    }
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

    fn even_moment_exp_neg_x2(power: usize) -> f64 {
        debug_assert!(power.is_multiple_of(2));
        let m = power / 2;
        let mut odd_double_factorial = 1.0_f64;
        for k in 0..m {
            odd_double_factorial *= (2 * k + 1) as f64;
        }
        odd_double_factorial * std::f64::consts::PI.sqrt() / 2.0_f64.powi(m as i32)
    }

    fn normal_pdf(z: f64) -> f64 {
        (-(z * z) * 0.5).exp() / (2.0 * std::f64::consts::PI).sqrt()
    }

    // Test-only real erfcx approximation based on A&S erf polynomial.
    // For x >= 0:
    //   erfcx(x) = exp(x^2) erfc(x) ≈ P(t), t = 1/(1+p x),
    // where erfc(x) ≈ exp(-x^2) P(t). This avoids large exp(x^2) factors.
    // For x < 0 use the reflection identity:
    //   erfcx(-x) = 2 exp(x^2) - erfcx(x).
    fn erfcx_test(x: f64) -> f64 {
        // Coefficients from the common A&S 7.1.26 erf approximation.
        const P: f64 = 0.3275911;
        const A1: f64 = 0.254_829_592;
        const A2: f64 = -0.284_496_736;
        const A3: f64 = 1.421_413_741;
        const A4: f64 = -1.453_152_027;
        const A5: f64 = 1.061_405_429;

        if x >= 0.0 {
            let t = 1.0 / (1.0 + P * x);
            (((((A5 * t + A4) * t + A3) * t + A2) * t + A1) * t).max(0.0)
        } else {
            let xp = -x;
            // Reflection: erfcx(-xp) = 2*exp(xp^2) - erfcx(xp).
            // In this test helper, cap exponent to avoid inf in extreme synthetic inputs.
            let exp_term = (xp * xp).min(700.0).exp();
            2.0 * exp_term - erfcx_test(xp)
        }
    }

    // Test-only deterministic erfcx-series oracle for E[sigmoid(N(m,s^2))].
    // Formula:
    // E = Phi(m/s) + 0.5*exp(-m^2/(2s^2)) * Σ_{k>=1} (-1)^{k-1}
    //     [erfcx((k s^2 + m)/(sqrt(2)s)) - erfcx((k s^2 - m)/(sqrt(2)s))].
    fn exact_logistic_expectation_erfcx_series(m: f64, s: f64) -> f64 {
        if !(m.is_finite() && s.is_finite()) || s <= 0.0 {
            return sigmoid(m).clamp(1e-12, 1.0 - 1e-12);
        }
        if s < 1e-10 {
            return sigmoid(m).clamp(1e-12, 1.0 - 1e-12);
        }

        let pref = 0.5 * (-(m * m) / (2.0 * s * s)).exp();
        let z = std::f64::consts::SQRT_2 * s;
        let mut sum = 0.0_f64;
        let mut stable_pairs = 0usize;

        // Pair terms (k,k+1) for faster alternating-series stabilization.
        let mut k = 1usize;
        while k <= 4096 {
            let kf = k as f64;
            let a1 = (kf * s * s + m) / z;
            let b1 = (kf * s * s - m) / z;
            let t1 = erfcx_test(a1) - erfcx_test(b1);
            let signed_t1 = if k % 2 == 1 { t1 } else { -t1 };
            sum += signed_t1;

            if k < 4096 {
                let k2f = (k + 1) as f64;
                let a2 = (k2f * s * s + m) / z;
                let b2 = (k2f * s * s - m) / z;
                let t2 = erfcx_test(a2) - erfcx_test(b2);
                let signed_t2 = if (k + 1) % 2 == 1 { t2 } else { -t2 };
                sum += signed_t2;

                let pair_mag = (signed_t1 + signed_t2).abs() * pref;
                if pair_mag < 1e-13 {
                    stable_pairs += 1;
                    if stable_pairs >= 6 {
                        break;
                    }
                } else {
                    stable_pairs = 0;
                }
            }
            k += 2;
        }

        let phi_term = crate::probability::normal_cdf_approx(m / s);
        (phi_term + pref * sum).clamp(1e-12, 1.0 - 1e-12)
    }

    fn high_res_sigmoid_integral(eta: f64, se: f64) -> f64 {
        // Composite Simpson rule over a wide finite interval under N(0,1).
        let a = -12.0_f64;
        let b = 12.0_f64;
        let n = 20_000usize; // even
        let h = (b - a) / n as f64;

        let integrand = |z: f64| -> f64 { sigmoid(eta + se * z) * normal_pdf(z) };

        let mut sum = integrand(a) + integrand(b);
        for i in 1..n {
            let x = a + (i as f64) * h;
            if i % 2 == 0 {
                sum += 2.0 * integrand(x);
            } else {
                sum += 4.0 * integrand(x);
            }
        }
        (sum * h / 3.0).clamp(1e-10, 1.0 - 1e-10)
    }

    #[test]
    fn test_computed_nodes_symmetric() {
        // Verify computed nodes are symmetric around zero
        let ctx = QuadratureContext::new();
        let gh = ctx.gauss_hermite();
        for i in 0..N_POINTS / 2 {
            let j = N_POINTS - 1 - i;
            assert_relative_eq!(gh.nodes[i], -gh.nodes[j], epsilon = 1e-12);
        }
        // Middle node is expected to be zero
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
    fn test_wilkinson_shift_finite_when_d_is_zero() {
        // Trailing 2x2 with equal diagonal entries => d=0.
        // Regression: using f64::signum() would produce denominator 0 here.
        let shift = wilkinson_shift(0.0, 0.0, 1.25);
        assert!(shift.is_finite());
        assert_relative_eq!(shift, -1.25, epsilon = 1e-14);
    }

    #[test]
    fn test_matches_known_7_point_constants() {
        let known_nodes = [
            -2.651_961_356_835_233_4,
            -1.673_551_628_767_471_4,
            -0.816_287_882_858_964_7,
            0.0,
            0.816_287_882_858_964_7,
            1.673_551_628_767_471_4,
            2.651_961_356_835_233_4,
        ];
        let known_weights = [
            0.000_971_781_245_099_519_1,
            0.054_515_582_819_127_03,
            0.425_607_252_610_127_8,
            0.810_264_617_556_807_3,
            0.425_607_252_610_127_8,
            0.054_515_582_819_127_03,
            0.000_971_781_245_099_519_1,
        ];

        let ctx = QuadratureContext::new();
        let gh = ctx.gauss_hermite();
        for i in 0..N_POINTS {
            assert_relative_eq!(gh.nodes[i], known_nodes[i], epsilon = 1e-12);
            assert_relative_eq!(gh.weights[i], known_weights[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn test_zero_se_returns_mode() {
        // When SE is zero, posterior mean is expected to equal mode
        let eta = 1.5;
        let se = 0.0;
        let ctx = QuadratureContext::new();
        let mean = logit_posterior_mean(&ctx, eta, se);
        let mode = sigmoid(eta);
        assert_relative_eq!(mean, mode, epsilon = 1e-10);
    }

    #[test]
    fn test_symmetric_at_zero() {
        // At eta=0 (50% probability), mean is expected to be ~50%
        let eta = 0.0;
        let se = 1.0;
        let ctx = QuadratureContext::new();
        let mean = logit_posterior_mean(&ctx, eta, se);
        // Due to symmetry of sigmoid around 0, mean ≈ mode
        assert_relative_eq!(mean, 0.5, epsilon = 0.01);
    }

    #[test]
    fn test_shrinkage_at_extremes() {
        // At extreme eta, mean is expected to be pulled toward 0.5
        let eta = 3.0; // mode = sigmoid(3) ≈ 0.953
        let se = 1.0;
        let ctx = QuadratureContext::new();
        let mean = logit_posterior_mean(&ctx, eta, se);
        let mode = sigmoid(eta);

        // Mean is expected to be less than mode (shrunk toward 0.5)
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
        // The quadrature exactly integrates x² against exp(-x²)
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
        // The quadrature exactly integrates x⁴ against exp(-x²)
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

    #[test]
    fn test_moment_exactness_up_to_degree_13() {
        let ctx = QuadratureContext::new();
        let gh = ctx.gauss_hermite();

        for degree in 0..=13usize {
            let approx: f64 = (0..N_POINTS)
                .map(|i| gh.weights[i] * gh.nodes[i].powi(degree as i32))
                .sum();

            let expected = if degree % 2 == 1 {
                0.0
            } else {
                even_moment_exp_neg_x2(degree)
            };

            let err = (approx - expected).abs();
            let rel_scale = approx.abs().max(expected.abs()).max(1.0);
            assert!(
                err <= 1e-10 || err / rel_scale <= 1e-10,
                "degree={} approx={} expected={} abs_err={}",
                degree,
                approx,
                expected,
                err
            );
        }
    }

    #[test]
    fn test_integrated_sigmoid_matches_high_res_integral_random_pairs() {
        let ctx = QuadratureContext::new();
        let mut rng_state = 0x4d595df4d0f33173u64;

        for _ in 0..20 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u_eta = (rng_state as f64) / (u64::MAX as f64);
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u_se = (rng_state as f64) / (u64::MAX as f64);

            let eta = -6.0 + 12.0 * u_eta;
            let se = 0.02 + 1.5 * u_se;

            let ghq = logit_posterior_mean(&ctx, eta, se);
            let numeric = high_res_sigmoid_integral(eta, se);
            assert_relative_eq!(ghq, numeric, epsilon = 2e-3);
        }
    }

    #[test]
    fn test_logit_posterior_derivative_not_hard_floored() {
        let ctx = QuadratureContext::new();
        let eta = 20.0;
        let se = 0.0;
        let (_, dmu) = logit_posterior_mean_with_deriv(&ctx, eta, se);
        assert!(dmu > 0.0);
        assert!(
            dmu < 1e-6,
            "tail derivative should be below legacy floor, got {dmu}"
        );
    }

    #[test]
    fn test_logit_posterior_derivative_matches_central_difference() {
        let ctx = QuadratureContext::new();
        let eta = 1.7;
        let se = 0.9;
        let h = 1e-5;

        let (_, dmu) = logit_posterior_mean_with_deriv(&ctx, eta, se);
        let mu_plus = logit_posterior_mean(&ctx, eta + h, se);
        let mu_minus = logit_posterior_mean(&ctx, eta - h, se);
        let dmu_fd = (mu_plus - mu_minus) / (2.0 * h);

        assert_relative_eq!(dmu, dmu_fd, epsilon = 5e-6, max_relative = 2e-4);
    }

    #[test]
    fn test_logit_posterior_mean_exact_symmetry_identity() {
        let cases = [(-3.0, 0.5), (-1.2, 1.7), (0.0, 2.2), (2.3, 0.8)];
        for (mu, sigma) in cases {
            let p = logit_posterior_mean_exact(mu, sigma);
            let q = logit_posterior_mean_exact(-mu, sigma);
            assert_relative_eq!(p + q, 1.0, epsilon = 3e-5);
        }
    }

    #[test]
    fn test_logit_posterior_mean_exact_matches_high_res_integral() {
        let cases = [(-2.0, 0.4), (-0.7, 1.1), (0.8, 0.9), (2.4, 1.7)];
        for (mu, sigma) in cases {
            let exact = logit_posterior_mean_exact(mu, sigma);
            let numeric = high_res_sigmoid_integral(mu, sigma);
            assert_relative_eq!(exact, numeric, epsilon = 2e-4);
        }
    }

    #[test]
    fn test_ghq7_close_to_exact_oracle() {
        let ctx = QuadratureContext::new();
        let cases = [(-3.0, 0.3), (-1.0, 0.8), (0.5, 1.2), (2.8, 1.0)];
        for (eta, se) in cases {
            let ghq = logit_posterior_mean(&ctx, eta, se);
            let exact = logit_posterior_mean_exact(eta, se);
            assert_relative_eq!(ghq, exact, epsilon = 2.5e-3);
        }
    }

    #[test]
    fn test_ghq7_matches_real_erfcx_series_oracle() {
        let ctx = QuadratureContext::new();
        let m_values = [-3.0, -1.0, 0.0, 1.0, 3.0];
        let s_values = [0.1, 0.5, 1.0, 2.0];

        for &m in &m_values {
            for &s in &s_values {
                let ghq = logit_posterior_mean(&ctx, m, s);
                let oracle = exact_logistic_expectation_erfcx_series(m, s);
                assert_relative_eq!(ghq, oracle, epsilon = 2e-3, max_relative = 3e-3);
            }
        }
    }

    #[test]
    fn test_probit_posterior_mean_reduces_to_map_at_zero_se() {
        let eta = 1.25;
        let p = probit_posterior_mean(eta, 0.0);
        let map = crate::probability::normal_cdf_approx(eta);
        assert_relative_eq!(p, map, epsilon = 1e-12);
    }

    #[test]
    fn test_probit_posterior_mean_shrinks_extremes_with_uncertainty() {
        let hi_eta = 3.0;
        let lo_eta = -3.0;
        let p_hi_map = probit_posterior_mean(hi_eta, 0.0);
        let p_hi_unc = probit_posterior_mean(hi_eta, 2.0);
        let p_lo_map = probit_posterior_mean(lo_eta, 0.0);
        let p_lo_unc = probit_posterior_mean(lo_eta, 2.0);
        assert!(p_hi_unc < p_hi_map);
        assert!(p_lo_unc > p_lo_map);
    }

    #[test]
    fn test_survival_posterior_mean_is_bounded_and_shrinks_tail() {
        let ctx = QuadratureContext::new();
        let eta: f64 = 3.0;
        let map = (-(eta.exp())).exp();
        let pm = survival_posterior_mean(&ctx, eta, 1.5);
        assert!((0.0..=1.0).contains(&pm));
        assert!(pm > map);
    }
}
