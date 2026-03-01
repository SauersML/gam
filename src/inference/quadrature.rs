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
//!
//! Roadmap for replacing GHQ in integrated PIRLS / uncertainty propagation:
//!
//! 1. Probit:
//!    If eta ~ N(mu, sigma^2), then
//!      E[Phi(eta)] = Phi(mu / sqrt(1 + sigma^2))
//!    exactly, with derivative
//!      d/dmu E[Phi(eta)] = phi(mu / sqrt(1 + sigma^2)) / sqrt(1 + sigma^2).
//!    This identity is already used by `probit_posterior_mean` below and is the
//!    model for how integrated IRLS should eventually avoid GHQ entirely for
//!    probit-linked updates.
//!
//! 2. Logit:
//!    The logistic-normal mean admits exact convergent special-function
//!    representations (Faddeeva / erfcx series). Those are ideal for the hot
//!    integrated-IRLS path because they replace per-row GHQ loops with a small,
//!    deterministic series and exact derivatives with respect to the Gaussian
//!    mean. This module already contains an oracle-style exact evaluator
//!    (`logit_posterior_mean_exact`) documenting the mathematics.
//!
//! 3. Cloglog / survival transforms:
//!    The complementary log-log mean under Gaussian eta does not simplify to an
//!    elementary closed form, but it does admit exact non-GHQ representations:
//!    - as the Laplace transform of a lognormal variable,
//!    - as characteristic-function inversion with Gamma(1 - i t),
//!    - or as rapidly convergent erfc / asymptotic series on subdomains.
//!    These are the natural replacements for repeated GHQ calls in
//!    `cloglog_posterior_mean` and any survival-specific cubature path.
//!
//! Derivative identity used by integrated PIRLS:
//! If eta = mu + sigma * Z with Z ~ N(0, 1), then for any smooth inverse-link f,
//!   d/dmu E[f(eta)] = E[f'(eta)].
//! This matters because integrated IRLS needs both
//!   mu_bar = E[g^{-1}(eta)]
//! and
//!   dmu_bar / deta = d/dmu E[g^{-1}(eta)].
//! Once a link-specific exact evaluator can return those two quantities, the
//! PIRLS update no longer needs any quadrature-node loop in the hot path.
//!
//! In particular:
//! - Probit:
//!     f(x) = Phi(x)
//!     E[f(eta)] = Phi(mu / sqrt(1 + sigma^2))
//!     d/dmu E[f(eta)]
//!       = phi(mu / sqrt(1 + sigma^2)) / sqrt(1 + sigma^2).
//! - Cloglog:
//!     f(x) = 1 - exp(-exp(x))
//!     E[f(eta)] = 1 - E[exp(-X)], X = exp(eta) ~ LogNormal(mu, sigma^2),
//!   so the mean is the complement of the lognormal Laplace transform at z = 1.
//!   The derivative is
//!     d/dmu E[f(eta)] = E[exp(eta - exp(eta))].
//! - Logit:
//!     f(x) = sigmoid(x)
//!     d/dmu E[f(eta)] = E[sigmoid(eta) * (1 - sigmoid(eta))],
//!   and both the mean and derivative admit exact convergent special-function
//!   representations via Faddeeva / erfcx expansions.
//!
//! The current GHQ implementations remain because they are robust and general,
//! but the intended direction is to move integrated PIRLS away from repeated
//! quadrature-node loops whenever a link-specific exact or special-function
//! representation is available.
//!
//! # Exact Object Behind Cloglog / Survival
//!
//! For the cloglog and Royston-Parmar-style survival transforms, the exact
//! shared scalar object is the lognormal Laplace transform
//!
//!   L(z; mu, sigma) = E[exp(-z exp(eta))],   eta ~ N(mu, sigma^2),  z > 0.
//!
//! Writing `X = exp(eta)`, this is `E[exp(-z X)]` with
//! `X ~ LogNormal(mu, sigma^2)`. Two exact identities organize the whole
//! implementation:
//!
//! 1. Shift reduction in `z`:
//!      L(z; mu, sigma) = L(1; mu + ln z, sigma)
//!    because `z exp(eta) = exp(eta + ln z)`.
//!
//! 2. Gaussian tilting / derivative identity:
//!      -d/dmu L(z; mu, sigma)
//!        = z * exp(mu + sigma^2 / 2) * L(z; mu + sigma^2, sigma).
//!
//! These imply:
//!
//! - survival mean:
//!     E[exp(-exp(eta))] = L(1; mu, sigma)
//! - cloglog mean:
//!     E[1 - exp(-exp(eta))] = 1 - L(1; mu, sigma)
//! - exact derivative for integrated PIRLS:
//!     d/dmu E[1 - exp(-exp(eta))]
//!       = exp(mu + sigma^2 / 2) * L(1; mu + sigma^2, sigma)
//! - second moment used in posterior variance:
//!     E[exp(-2 exp(eta))] = L(2; mu, sigma) = L(1; mu + ln 2, sigma)
//!
//! So all integrated cloglog/survival quantities are just algebra on top of the
//! same `L(z; mu, sigma)` object.
//!
//! # Representation Classes Used Here
//!
//! For `L(z; mu, sigma)`, there is no simple elementary closed form. The useful
//! exact representations in this module are:
//!
//! - a real-line Gaussian expectation
//! - a Mellin-Barnes / Bromwich contour representation involving `Gamma`
//! - an erfc-gated Miles series in tail-dominated regimes
//! - a real-line Clenshaw-Curtis evaluator for the central regime
//!
//! The production routing therefore chooses the numerically best exact or
//! controlled representation for each regime rather than pretending that one
//! universal formula dominates everywhere.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use crate::estimate::EstimationError;
use crate::types::LinkFunction;
use statrs::function::erf::erfc;

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
    // Clenshaw-Curtis rules are constructed on demand because the node count is
    // chosen from the certified truncation/ellipse heuristic rather than from a
    // tiny fixed family like the GHQ rules above.
    cc_cache: Mutex<HashMap<usize, ClenshawCurtisRule>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegratedExpectationMode {
    ExactClosedForm,
    ExactSpecialFunction,
    ControlledAsymptotic,
    QuadratureFallback,
}

#[derive(Clone, Copy, Debug)]
pub struct IntegratedMeanDerivative {
    pub mean: f64,
    pub dmean_dmu: f64,
    pub mode: IntegratedExpectationMode,
}

const LOGIT_SIGMA_DEGENERATE: f64 = 2.5e-1;
const CLOGLOG_SIGMA_DEGENERATE: f64 = 1e-10;
const CLOGLOG_SIGMA_TAYLOR_MAX: f64 = 0.25;
const SERIES_CONSECUTIVE_SMALL_TERMS: usize = 6;
const LOGIT_MAX_TERMS: usize = 80;
const CLOGLOG_MILES_ALPHA: f64 = 60.0;
const CLOGLOG_MILES_MAX_TERMS: usize = 256;
const CLOGLOG_GAMMA_K_REF: f64 = 0.5;
const CLOGLOG_GAMMA_T_MAX_REF: f64 = 24.0;
const CLOGLOG_GAMMA_H_REF: f64 = 0.01;
// Default accuracy target for the real-line Clenshaw-Curtis cloglog backend.
// This is intentionally looser than full machine epsilon so the node-count
// heuristic stays practical in the central moderate/large-sigma regime.
const CLOGLOG_CC_TOL: f64 = 1e-12;
// If the Bernstein-ellipse-based node request exceeds this cap, the backend
// yields to the exact Gamma reference rather than turning one hard case into a
// slow quadrature sweep.
const CLOGLOG_CC_NODE_CAP: usize = 1025;
// Gamma uses a fixed composite Simpson rule on [0, T] with this many samples.
// CC only wins if its requested node count stays comfortably below that fixed
// complex-arithmetic workload.
const CLOGLOG_GAMMA_SAMPLE_COUNT: usize =
    (CLOGLOG_GAMMA_T_MAX_REF / CLOGLOG_GAMMA_H_REF) as usize + 1;
// CC nodes are pure f64 work while Gamma nodes pay for complex log-gamma and
// complex exponentials, so CC can still be favorable with somewhat more nodes
// than this threshold. Keep the threshold conservative until benchmarks say
// otherwise.
const CLOGLOG_CC_PREFER_THRESHOLD: usize = CLOGLOG_GAMMA_SAMPLE_COUNT / 3;
// Keep a modest floor so the mapped cosine rule is never asked to represent the
// integrand with an undersized stencil even when the heuristic requests very
// few nodes.
const CLOGLOG_CC_MIN_N: usize = 17;

impl QuadratureContext {
    pub fn new() -> Self {
        Self {
            gh_cache: OnceLock::new(),
            gh15_cache: OnceLock::new(),
            gh21_cache: OnceLock::new(),
            gh31_cache: OnceLock::new(),
            cc_cache: Mutex::new(HashMap::new()),
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

    fn clenshaw_curtis_n(&self, n: usize) -> ClenshawCurtisRule {
        let mut cache = self.cc_cache.lock().expect("cc cache mutex poisoned");
        cache
            .entry(n)
            .or_insert_with(|| compute_clenshaw_curtis_n(n))
            .clone()
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

#[derive(Clone)]
struct ClenshawCurtisRule {
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

fn compute_clenshaw_curtis_n(n: usize) -> ClenshawCurtisRule {
    debug_assert!(n >= 2);
    // Classic cosine-grid Clenshaw-Curtis rule on [-1, 1].
    //
    // The nodes are
    //   x_j = cos(j pi / (n - 1)),   j = 0, ..., n - 1,
    // i.e. the Chebyshev extrema. In the usual derivation one writes x = cos θ,
    // expands the transformed integrand in a cosine/Chebyshev series, and then
    // integrates the interpolating polynomial exactly. That is why this rule is
    // naturally expressed on a cosine grid and why it is a good fit for the
    // truncated cloglog/survival real-line integral after the affine map t = A x.
    //
    // This implementation uses the explicit cosine-sum weight formula rather
    // than a fast DCT construction. That is perfectly adequate here because the
    // production node counts are modest and the rules are cached in
    // QuadratureContext once built.
    let m = n - 1;
    let theta: Vec<f64> = (0..=m)
        .map(|j| std::f64::consts::PI * (j as f64) / (m as f64))
        .collect();
    let nodes: Vec<f64> = theta.iter().map(|&th| th.cos()).collect();

    if n == 2 {
        return ClenshawCurtisRule {
            nodes,
            weights: vec![1.0, 1.0],
        };
    }

    let mut weights = vec![0.0_f64; n];
    let mut v = vec![1.0_f64; m - 1];

    if m.is_multiple_of(2) {
        let w0 = 1.0 / ((m * m - 1) as f64);
        weights[0] = w0;
        weights[m] = w0;
        for k in 1..(m / 2) {
            let denom = (4 * k * k - 1) as f64;
            for j in 1..m {
                v[j - 1] -= 2.0 * (2.0 * (k as f64) * theta[j]).cos() / denom;
            }
        }
        for j in 1..m {
            v[j - 1] -= ((m as f64) * theta[j]).cos() / ((m * m - 1) as f64);
        }
    } else {
        let w0 = 1.0 / ((m * m) as f64);
        weights[0] = w0;
        weights[m] = w0;
        for k in 1..=((m - 1) / 2) {
            let denom = (4 * k * k - 1) as f64;
            for j in 1..m {
                v[j - 1] -= 2.0 * (2.0 * (k as f64) * theta[j]).cos() / denom;
            }
        }
    }

    for j in 1..m {
        weights[j] = 2.0 * v[j - 1] / (m as f64);
    }

    // Clenshaw-Curtis on [-1, 1] is symmetric and integrates constants exactly.
    // Enforce those invariants explicitly after the cosine-sum construction so
    // tiny roundoff in the weight build does not leak into the cached rules.
    for j in 0..=(m / 2) {
        let jj = m - j;
        let avg = 0.5 * (weights[j] + weights[jj]);
        weights[j] = avg;
        weights[jj] = avg;
    }
    let weight_sum: f64 = weights.iter().sum();
    if weight_sum.is_finite() && weight_sum != 0.0 {
        let scale = 2.0 / weight_sum;
        for w in &mut weights {
            *w *= scale;
        }
    }

    ClenshawCurtisRule { nodes, weights }
}

fn cloglog_cc_required_nodes(mu: f64, sigma: f64, tol: f64) -> Result<usize, EstimationError> {
    if !(mu.is_finite() && sigma.is_finite() && sigma > 0.0 && tol.is_finite() && tol > 0.0) {
        return Err(EstimationError::InvalidInput(
            "CC cloglog backend requires finite mu, positive sigma, and positive tolerance"
                .to_string(),
        ));
    }

    // This mirrors the node-count logic used by the actual CC evaluator, but
    // exposes it as a cheap routing estimate so we can decide whether the
    // bounded real-line cosine grid is likely to beat the fixed-work complex
    // Gamma backend before paying to evaluate either one.
    let p_tail = (tol / 8.0).clamp(1e-300, 0.25);
    let a = crate::probability::standard_normal_quantile(p_tail)
        .map(|z| -z)
        .unwrap_or(8.0)
        .max(1.0);

    let ay = a * sigma;
    let y = if ay > 0.0 {
        1.0_f64.min(std::f64::consts::PI / (4.0 * ay))
    } else {
        1.0
    };
    let rho = y + (1.0 + y * y).sqrt();
    let m_s = (0.5 * (a * y) * (a * y)).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let eps_quad = (tol / 4.0).max(1e-300);
    let numer = ((8.0 * a * m_s) / ((rho - 1.0).max(1e-12) * eps_quad)).max(1.0);
    let denom = rho.ln();
    if !denom.is_finite() || denom <= 0.0 {
        return Err(EstimationError::InvalidInput(
            "CC cloglog backend ellipse bound became degenerate".to_string(),
        ));
    }

    let mut n = (1.0 + numer.ln() / denom).ceil() as usize;
    n = n.max(CLOGLOG_CC_MIN_N);
    if n.is_multiple_of(2) {
        n += 1;
    }
    Ok(n)
}

#[inline]
fn cloglog_should_prefer_cc(mu: f64, sigma: f64, tol: f64) -> bool {
    // Prefer CC only when its Bernstein-ellipse node estimate stays comfortably
    // below the fixed Simpson workload of the Gamma reference backend. That
    // makes CC an automatic fast path for moderate central cases, while very
    // broad or numerically awkward cases continue to use the exact
    // Mellin-Barnes/Gamma representation.
    match cloglog_cc_required_nodes(mu, sigma, tol) {
        Ok(n) => n <= CLOGLOG_CC_NODE_CAP && n <= CLOGLOG_CC_PREFER_THRESHOLD,
        Err(_) => false,
    }
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
    // Important architectural note:
    // This is the current integrated-PIRLS hot path for logit measurement-error
    // updates. It still uses GHQ for robustness, but mathematically this is one
    // of the best candidates for removal of quadrature from the inner loop.
    //
    // Exact alternative:
    // If eta ~ N(mu, sigma^2), then E[sigmoid(eta)] and d/dmu E[sigmoid(eta)]
    // admit exact convergent Faddeeva / erfcx series representations. Those
    // replace "sum over quadrature nodes" with "sum over a short special-
    // function series", which is deterministic and can be much cheaper when
    // called once per row per PIRLS iteration.
    //
    // We keep GHQ here today because:
    // - it is already validated across the current domain,
    // - it shares code with other links,
    // - and it avoids coupling core IRLS updates to a special-function backend
    //   before that backend has equivalent tests and stability guards.
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

#[inline]
pub fn probit_posterior_mean_with_deriv_exact(mu: f64, sigma: f64) -> IntegratedMeanDerivative {
    // Exact Gaussian-probit convolution.
    //
    // If eta ~ N(mu, sigma^2), then
    //
    //   E[Phi(eta)] = Phi(mu / sqrt(1 + sigma^2)).
    //
    // A clean derivation is to introduce an independent Z ~ N(0, 1):
    //
    //   E[Phi(eta)]
    //     = E[P(Z <= eta | eta)]
    //     = P(Z - eta <= 0).
    //
    // Because Z - eta is Gaussian with mean -mu and variance 1 + sigma^2, the
    // probability is exactly the standard normal CDF evaluated at
    //   mu / sqrt(1 + sigma^2).
    //
    // Differentiating with respect to the location parameter mu gives
    //
    //   d/dmu E[Phi(eta)]
    //     = phi(mu / sqrt(1 + sigma^2)) / sqrt(1 + sigma^2),
    //
    // which is also the integrated slope E[phi(eta)] by the general identity
    //
    //   d/dmu E[f(mu + sigma Z)] = E[f'(mu + sigma Z)].
    //
    // So this path is genuinely exact: no node count, no truncation, and no
    // approximation regime split.
    if !(mu.is_finite() && sigma.is_finite()) || sigma <= 1e-12 {
        let mean = crate::probability::normal_cdf_approx(mu);
        let dmean_dmu = crate::probability::normal_pdf(mu);
        return IntegratedMeanDerivative {
            mean,
            dmean_dmu,
            mode: IntegratedExpectationMode::ExactClosedForm,
        };
    }
    let denom = (1.0 + sigma * sigma).sqrt();
    let z = mu / denom;
    IntegratedMeanDerivative {
        mean: crate::probability::normal_cdf_approx(z),
        dmean_dmu: crate::probability::normal_pdf(z) / denom,
        mode: IntegratedExpectationMode::ExactClosedForm,
    }
}

#[inline]
fn logistic_normal_exact_eligible(mu: f64, sigma: f64) -> bool {
    mu.is_finite()
        && sigma.is_finite()
        && mu.abs() <= 30.0
        && sigma > LOGIT_SIGMA_DEGENERATE
        // The in-repo erfcx series is accurate and stable on a moderate-variance
        // domain, but it does not currently meet the production accuracy bar all
        // the way out to sigma = 5. Past this range we intentionally route back
        // to GHQ rather than over-claim exactness.
        && sigma <= 1.0
}

#[inline]
fn logistic_normal_tail_cutoff(mu: f64, sigma: f64) -> usize {
    let m = 0.5 * mu.abs();
    let s = 0.5 * sigma;
    let raw = (((16.0 * (m + 6.0 * s)) / std::f64::consts::PI) + 1.0) * 0.5;
    raw.ceil().clamp(4.0, LOGIT_MAX_TERMS as f64) as usize
}

#[inline]
fn erfcx_nonnegative(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_positive() {
            0.0
        } else {
            f64::INFINITY
        };
    }
    if x <= 0.0 {
        return 1.0;
    }
    if x < 26.0 {
        ((x * x).min(700.0)).exp() * erfc(x)
    } else {
        let inv = 1.0 / x;
        let inv2 = inv * inv;
        let poly = 1.0
            + 0.5 * inv2
            + 0.75 * inv2 * inv2
            + 1.875 * inv2 * inv2 * inv2
            + 6.5625 * inv2 * inv2 * inv2 * inv2;
        inv * poly / std::f64::consts::PI.sqrt()
    }
}

#[inline]
fn scaled_erfcx_term_with_derivative(m: f64, s: f64, x: f64, dxdm: f64) -> (f64, f64) {
    let pref = 0.5 * (-(m * m) / (2.0 * s * s)).exp();
    if x >= 0.0 {
        let ex = erfcx_nonnegative(x);
        let term = pref * ex;
        let ex_prime = 2.0 * x * ex - std::f64::consts::FRAC_2_SQRT_PI;
        let dterm = pref * ((-m / (s * s)) * ex + ex_prime * dxdm);
        (term, dterm)
    } else {
        let lead = (x * x - (m * m) / (2.0 * s * s)).exp();
        let dlead = lead * (2.0 * x * dxdm - m / (s * s));
        let (rest, drest) = scaled_erfcx_term_with_derivative(m, s, -x, -dxdm);
        (lead - rest, dlead - drest)
    }
}

pub(crate) fn logit_posterior_mean_with_deriv_exact(
    mu: f64,
    sigma: f64,
) -> Result<IntegratedMeanDerivative, EstimationError> {
    // Guarded exact/special-function entry point for the logistic-normal mean.
    //
    // The target objects are
    //
    //   mean(mu, sigma)   = E[sigmoid(eta)],
    //   dmean/dmu         = E[sigmoid(eta) * (1 - sigmoid(eta))],
    //   eta ~ N(mu, sigma^2).
    //
    // Unlike probit, this Gaussian convolution does not collapse to an
    // elementary closed form. The helper below uses an erfcx-based exact/
    // special-function representation, but only on a guarded domain where the
    // current in-repo numerical implementation is actually validated. Outside
    // that domain we deliberately refuse to claim exactness and let the
    // dispatcher fall back to GHQ.
    if !(mu.is_finite() && sigma.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "logit exact expectation requires finite mu and sigma".to_string(),
        ));
    }
    if sigma <= LOGIT_SIGMA_DEGENERATE {
        let mean = sigmoid(mu);
        return Ok(IntegratedMeanDerivative {
            mean,
            dmean_dmu: mean * (1.0 - mean),
            mode: IntegratedExpectationMode::ExactClosedForm,
        });
    }
    if !logistic_normal_exact_eligible(mu, sigma) {
        return Err(EstimationError::InvalidInput(
            "logit exact expectation outside guarded domain".to_string(),
        ));
    }
    logit_posterior_mean_with_deriv_exact_erfcx(mu, sigma)
}

fn logit_posterior_mean_with_deriv_exact_erfcx(
    mu: f64,
    sigma: f64,
) -> Result<IntegratedMeanDerivative, EstimationError> {
    // Real-valued erfcx-series implementation for the logistic-normal mean.
    //
    // Start from
    //
    //   sigmoid(x) = 1/2 + 1/2 tanh(x/2).
    //
    // The partial-fraction expansion of tanh over its odd poles turns the
    // logistic-normal expectation into a sum of Gaussian rational integrals.
    // Those integrals reduce to scaled complementary-error-function terms, so
    // the production implementation is written in erfcx rather than complex
    // Faddeeva calls.
    //
    // The derivative is obtained by differentiating the same series with
    // respect to mu. This remains guarded because the present in-repo erfcx
    // approximation and truncation logic have only been validated on a
    // moderate-variance region; outside that region the dispatcher uses GHQ.
    let m = mu.abs();
    let s = sigma;
    let z = SQRT_2 * s;
    let phi_term = crate::probability::normal_cdf_approx(m / s);
    let phi_prime = crate::probability::normal_pdf(m / s) / s;
    let mut sum = 0.0_f64;
    let mut dsum = 0.0_f64;
    let mut stable_pairs = 0usize;
    let mut converged = false;
    let max_k = logistic_normal_tail_cutoff(mu, sigma);
    let required_stable_pairs = SERIES_CONSECUTIVE_SMALL_TERMS.min(((max_k + 1) / 2).max(1));

    let mut k = 1usize;
    while k <= max_k {
        let mut pair_contrib = 0.0_f64;
        let mut pair_dcontrib = 0.0_f64;

        for kk in [k, k + 1].into_iter().filter(|kk| *kk <= max_k) {
            let kf = kk as f64;
            let a = (kf * s * s + m) / z;
            let b = (kf * s * s - m) / z;
            let sign = if kk % 2 == 1 { 1.0 } else { -1.0 };
            let (va, dva) = scaled_erfcx_term_with_derivative(m, s, a, 1.0 / z);
            let (vb, dvb) = scaled_erfcx_term_with_derivative(m, s, b, -1.0 / z);
            pair_contrib += sign * (va - vb);
            pair_dcontrib += sign * (dva - dvb);
        }

        sum += pair_contrib;
        dsum += pair_dcontrib;

        if pair_contrib.abs() <= f64::EPSILON * (1.0 + phi_term.abs()) {
            stable_pairs += 1;
            if stable_pairs >= required_stable_pairs {
                converged = true;
                break;
            }
        } else {
            stable_pairs = 0;
        }

        k += 2;
    }

    if !converged && max_k < LOGIT_MAX_TERMS {
        converged = true;
    }

    if !converged {
        return Err(EstimationError::InvalidInput(
            "logit erfcx expectation did not satisfy the production truncation rule".to_string(),
        ));
    }

    let mut mean = phi_term + sum;
    let dmean = (phi_prime + dsum).max(0.0);
    if mu < 0.0 {
        mean = 1.0 - mean;
    }
    if !(mean.is_finite() && dmean.is_finite() && dmean >= 0.0) {
        return Err(EstimationError::InvalidInput(
            "logit erfcx expectation produced non-finite values".to_string(),
        ));
    }
    Ok(IntegratedMeanDerivative {
        mean: mean.clamp(1e-12, 1.0 - 1e-12),
        dmean_dmu: dmean,
        mode: IntegratedExpectationMode::ExactSpecialFunction,
    })
}

#[inline]
fn cloglog_small_sigma_taylor(mu: f64, sigma: f64) -> IntegratedMeanDerivative {
    // Small-variance heat-kernel expansion.
    //
    // For eta = mu + sigma Z, Z ~ N(0,1), and any analytic f,
    //   E[f(eta)] = sum_{k>=0} sigma^{2k} / (2^k k!) * f^{(2k)}(mu).
    //
    // Here f(x) = 1 - exp(-exp(x)) is entire, so the series is valid
    // globally. In the hot path we truncate after the sigma^4 term:
    //   E[f(eta)]      ~= f + (sigma^2/2) f'' + (sigma^4/24) f^{(4)}
    //   d/dmu E[f(eta)]~= f' + (sigma^2/2) f''' + (sigma^4/24) f^{(5)}.
    //
    // The explicit derivatives below are written in terms of exp(mu) and
    // exp(-exp(mu)) so the PIRLS derivative uses the same approximation order
    // as the mean.
    let z = mu.clamp(-30.0, 30.0);
    let s2 = sigma * sigma;
    let s4 = s2 * s2;
    let ex = z.exp();
    let e2x = ex * ex;
    let e3x = e2x * ex;
    let e4x = e3x * ex;
    let e5x = e4x * ex;
    let surv = (-ex).exp();
    let f0 = 1.0 - surv;
    let f1 = ex * surv;
    let f2 = surv * (ex - e2x);
    let f3 = surv * (ex - 3.0 * e2x + e3x);
    let f4 = surv * (ex - 7.0 * e2x + 6.0 * e3x - e4x);
    let f5 = surv * (ex - 15.0 * e2x + 25.0 * e3x - 10.0 * e4x + e5x);
    IntegratedMeanDerivative {
        mean: (f0 + 0.5 * s2 * f2 + (s4 / 24.0) * f4).clamp(1e-12, 1.0 - 1e-12),
        dmean_dmu: (f1 + 0.5 * s2 * f3 + (s4 / 24.0) * f5).max(0.0),
        mode: IntegratedExpectationMode::ControlledAsymptotic,
    }
}

#[cfg(test)]
#[inline]
fn cloglog_posterior_mean_with_deriv_ghq(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> IntegratedMeanDerivative {
    let mean = cloglog_posterior_mean(ctx, mu, sigma);
    let dmean_dmu = integrate_normal_ghq_adaptive(ctx, mu, sigma, |x| {
        let z = x.clamp(-30.0, 30.0);
        let ez = z.exp();
        ez * (-ez).exp()
    })
    .max(0.0);
    IntegratedMeanDerivative {
        mean,
        dmean_dmu,
        mode: IntegratedExpectationMode::QuadratureFallback,
    }
}

#[inline]
fn survival_posterior_mean_ghq(ctx: &QuadratureContext, eta: f64, se_eta: f64) -> f64 {
    integrate_normal_ghq_adaptive(ctx, eta, se_eta, |x| {
        let z = x.clamp(-30.0, 30.0);
        (-(z.exp())).exp()
    })
    .clamp(0.0, 1.0)
}

fn cloglog_survival_term_controlled(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> (f64, IntegratedExpectationMode) {
    // Shared scalar evaluator for the lognormal-Laplace object
    //
    //   S(mu, sigma) = E[exp(-exp(eta))],  eta ~ N(mu, sigma^2),
    //
    // This is the survival transform itself, and it is also the complement-core
    // of the cloglog inverse link:
    //
    //   cloglog mean   = 1 - S(mu, sigma)
    //   survival mean  = S(mu, sigma).
    //
    // The exact mathematical object behind this is the Laplace transform of a
    // lognormal random variable. If X = exp(eta), then X ~ LogNormal(mu,sigma^2)
    // and
    //
    //   S(mu, sigma) = E[exp(-X)] = L(1; mu, sigma),
    //
    // where more generally
    //
    //   L(z; mu, sigma) = E[exp(-z exp(eta))],  z > 0.
    //
    // So every path below is just a different exact or controlled evaluator for
    // the same scalar target.
    //
    // Routing here mirrors the production ladder used by the integrated
    // cloglog derivative path:
    // - plug-in when sigma is effectively zero
    // - Taylor / heat-kernel at small sigma
    // - Miles erfc-series in tail-dominated regimes
    // - Clenshaw-Curtis on the truncated real integral in the central regime
    // - exact Gamma/Mellin-Barnes if CC would need too many nodes or misbehaves
    // - GHQ only as the final numerical fallback
    if !(mu.is_finite() && sigma.is_finite()) || sigma <= CLOGLOG_SIGMA_DEGENERATE {
        let z = mu.clamp(-30.0, 30.0);
        return (
            (-(z.exp())).exp().clamp(0.0, 1.0),
            IntegratedExpectationMode::ExactClosedForm,
        );
    }
    if sigma < CLOGLOG_SIGMA_TAYLOR_MAX {
        let mean = cloglog_small_sigma_taylor(mu, sigma).mean;
        return (
            (1.0 - mean).clamp(0.0, 1.0),
            IntegratedExpectationMode::ControlledAsymptotic,
        );
    }
    if (mu.abs() / sigma) >= 3.0 {
        if let Ok(out) = cloglog_survival_miles(mu, sigma) {
            return (
                out.clamp(0.0, 1.0),
                IntegratedExpectationMode::ExactSpecialFunction,
            );
        }
    }
    if cloglog_should_prefer_cc(mu, sigma, CLOGLOG_CC_TOL)
        && let Ok(out) = cloglog_survival_cc(ctx, mu, sigma, CLOGLOG_CC_TOL)
    {
        return (
            out.clamp(0.0, 1.0),
            IntegratedExpectationMode::ExactSpecialFunction,
        );
    }
    if let Ok(out) = cloglog_survival_gamma_reference(mu, sigma) {
        return (
            out.clamp(0.0, 1.0),
            IntegratedExpectationMode::ExactSpecialFunction,
        );
    }
    (
        survival_posterior_mean_ghq(ctx, mu, sigma),
        IntegratedExpectationMode::QuadratureFallback,
    )
}

#[inline]
fn lognormal_laplace_term_controlled(
    ctx: &QuadratureContext,
    z: f64,
    mu: f64,
    sigma: f64,
) -> (f64, IntegratedExpectationMode) {
    // Shared shift reduction for the full lognormal-Laplace family:
    //
    //   L(z; mu, sigma) = E[exp(-z exp(eta))]
    //                   = E[exp(-exp(eta + ln z))]
    //                   = L(1; mu + ln z, sigma),
    //
    // because eta + ln z is still Gaussian with the same variance and shifted
    // mean. This is the cleanest way to see that cloglog and Royston-Parmar
    // survival are really querying one object:
    //
    //   survival first moment:
    //     E[exp(-exp(eta))] = L(1; mu, sigma)
    //
    //   survival second moment:
    //     E[exp(-2 exp(eta))] = L(2; mu, sigma) = L(1; mu + ln 2, sigma)
    //
    //   cloglog mean:
    //     E[1 - exp(-exp(eta))] = 1 - L(1; mu, sigma)
    //
    //   cloglog derivative:
    //     d/dmu E[1 - exp(-exp(eta))]
    //       = exp(mu + sigma^2/2) L(1; mu + sigma^2, sigma).
    //
    // So this helper is the canonical scalar boundary, and every higher-level
    // quantity is just algebra on top of it.
    if !(z.is_finite() && z > 0.0) {
        return (f64::NAN, IntegratedExpectationMode::QuadratureFallback);
    }
    cloglog_survival_term_controlled(ctx, mu + z.ln(), sigma)
}

#[inline]
fn cloglog_survival_second_moment_controlled(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> (f64, IntegratedExpectationMode) {
    // If
    //
    //   S(mu, sigma) = E[exp(-exp(eta))],   eta ~ N(mu, sigma^2),
    //
    // then the survival second moment is
    //
    //   E[S(eta)^2]
    //     = E[exp(-2 exp(eta))]
    //     = L(2; mu, sigma)
    //     = L(1; mu + ln 2, sigma)
    //     = S(mu + ln 2, sigma).
    //
    // So the exact same routed scalar evaluator can be reused by shifting mu
    // by ln 2 rather than introducing a second quadrature-specific code path.
    lognormal_laplace_term_controlled(ctx, 2.0, mu, sigma)
}

#[inline]
fn cloglog_survival_pair_controlled(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> ((f64, IntegratedExpectationMode), (f64, IntegratedExpectationMode)) {
    let shifted_mu = mu + sigma * sigma;

    // For the exact/control branches it is numerically cleaner if the mean
    // path S(mu, sigma) and the derivative path S(mu + sigma^2, sigma) are
    // evaluated on the same backend whenever possible. That keeps
    //
    //   mean       = 1 - S(mu, sigma)
    //   dmean/dmu  = exp(mu + sigma^2/2) * S(mu + sigma^2, sigma)
    //
    // on one approximation surface instead of mixing, for example, CC for the
    // base term with Gamma for the shifted term. If a paired attempt fails, we
    // fall back to the usual independent routing.
    if (mu.abs() / sigma) >= 3.0
        && let (Ok(base), Ok(shifted)) = (
            cloglog_survival_miles(mu, sigma),
            cloglog_survival_miles(shifted_mu, sigma),
        )
    {
        return (
            (base.clamp(0.0, 1.0), IntegratedExpectationMode::ExactSpecialFunction),
            (
                shifted.clamp(0.0, 1.0),
                IntegratedExpectationMode::ExactSpecialFunction,
            ),
        );
    }

    if cloglog_should_prefer_cc(mu, sigma, CLOGLOG_CC_TOL)
        && cloglog_should_prefer_cc(shifted_mu, sigma, CLOGLOG_CC_TOL)
        && let (Ok(base), Ok(shifted)) = (
            cloglog_survival_cc(ctx, mu, sigma, CLOGLOG_CC_TOL),
            cloglog_survival_cc(ctx, shifted_mu, sigma, CLOGLOG_CC_TOL),
        )
    {
        return (
            (base.clamp(0.0, 1.0), IntegratedExpectationMode::ExactSpecialFunction),
            (
                shifted.clamp(0.0, 1.0),
                IntegratedExpectationMode::ExactSpecialFunction,
            ),
        );
    }

    if let (Ok(base), Ok(shifted)) = (
        cloglog_survival_gamma_reference(mu, sigma),
        cloglog_survival_gamma_reference(shifted_mu, sigma),
    ) {
        return (
            (base.clamp(0.0, 1.0), IntegratedExpectationMode::ExactSpecialFunction),
            (
                shifted.clamp(0.0, 1.0),
                IntegratedExpectationMode::ExactSpecialFunction,
            ),
        );
    }

    (
        cloglog_survival_term_controlled(ctx, mu, sigma),
        cloglog_survival_term_controlled(ctx, shifted_mu, sigma),
    )
}

#[inline]
fn cloglog_mean_from_survival(survival: f64) -> f64 {
    let survival = survival.clamp(0.0, 1.0);
    if survival > 0.5 {
        // When S is close to 1, form 1 - S as -expm1(log S) so the rare-event
        // cloglog probability keeps its low-order bits instead of collapsing to
        // zero through cancellation. Algebraically:
        //
        //   1 - S = -expm1(log S),
        //
        // since exp(log S) = S. This is the stable way to recover the cloglog
        // mean in the regime mu << 0 where S is extremely close to 1 and the
        // desired probability is tiny.
        (-survival.ln().exp_m1()).clamp(1e-12, 1.0 - 1e-12)
    } else {
        (1.0 - survival).clamp(1e-12, 1.0 - 1e-12)
    }
}

#[inline]
fn cloglog_shift_identity_derivative(mu: f64, sigma: f64, shifted_survival: f64) -> f64 {
    // Exact Gaussian tilting identity:
    //
    //   d/dmu E[1 - exp(-exp(eta))]
    //     = exp(mu + sigma^2 / 2) * S(mu + sigma^2, sigma),
    //
    // where S is the shared survival term. The product is evaluated in the log
    // domain because exp(mu + sigma^2/2) can overflow even though the final
    // derivative is always bounded:
    //
    //   0 <= E[exp(eta - exp(eta))] <= sup_x x e^{-x} = e^{-1}.
    //
    // So any positive overflow is numerical, not mathematical, and can be
    // safely capped at the exact global upper bound.
    if !(mu.is_finite() && sigma.is_finite()) || shifted_survival <= 0.0 {
        return 0.0;
    }
    let log_derivative = mu + 0.5 * sigma * sigma + shifted_survival.ln();
    let upper = 1.0 / std::f64::consts::E;
    if !log_derivative.is_finite() {
        return upper;
    }
    log_derivative.exp().clamp(0.0, upper)
}

#[inline]
fn log_half_erfc_stable(u: f64) -> f64 {
    // Stable log(0.5 * erfc(u)).
    //
    // In the Miles series, each term contains
    //   exp(mu n + 0.5 sigma^2 n^2) * 0.5 * erfc(u_n).
    // For large positive u_n, erfc(u_n) underflows long before the *whole*
    // term becomes negligible, so we switch to
    //   erfc(u) = exp(-u^2) erfcx(u),  u > 0,
    // and carry the -u^2 contribution in log-space. For u <= 0, erfc(u) is
    // O(1) and direct evaluation is safe.
    if u > 0.0 {
        -u * u + (0.5 * erfcx_nonnegative(u)).ln()
    } else {
        (0.5 * erfc(u)).ln()
    }
}

fn cloglog_survival_miles(mu: f64, sigma: f64) -> Result<f64, EstimationError> {
    // This routine approximates the survival term
    //
    //   S(mu, sigma) = E[exp(-exp(eta))],   eta ~ N(mu, sigma^2),
    //
    // using the Miles erfc-gated lognormal-Laplace series. Writing
    //
    //   X = exp(eta) ~ LogNormal(mu, sigma^2),
    //
    // S is the Laplace transform E[exp(-X)] evaluated at 1. Theorem-3 gives a
    // real series of the form
    //
    //   S(mu, sigma)
    //     = sum_{n>=0} (-1)^n / n!
    //         * exp(mu n + 0.5 sigma^2 n^2)
    //         * 0.5 * erfc(u_n)
    //
    //   u_n = (mu - ln(alpha) + sigma^2 n) / (sqrt(2) sigma).
    //
    // The erfc factor gates the lognormal moment term so that the product stays
    // finite in the tail-dominated regime where this backend is used. We only
    // evaluate S here; the caller forms
    //
    //   mean = 1 - S
    //   dmean/dmu = exp(mu + sigma^2 / 2) * S(mu + sigma^2, sigma).
    //
    // Pairwise accumulation is used because the series alternates in sign and
    // consecutive terms partially cancel. Grouping terms before the truncation
    // check produces a materially more stable stopping rule than looking at
    // individual terms in isolation.
    let alpha_ln = CLOGLOG_MILES_ALPHA.ln();
    let mut s_sum = 0.0_f64;
    let mut stable_pairs = 0usize;

    for pair_start in (0..CLOGLOG_MILES_MAX_TERMS).step_by(2) {
        let mut pair_s = 0.0_f64;
        for n in pair_start..(pair_start + 2).min(CLOGLOG_MILES_MAX_TERMS) {
            let nf = n as f64;
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            let base_log = nf * mu + 0.5 * sigma * sigma * nf * nf
                - statrs::function::gamma::ln_gamma(nf + 1.0);
            let u = (mu - alpha_ln + sigma * sigma * nf) / (SQRT_2 * sigma);
            let log_half_erfc = log_half_erfc_stable(u);
            let term = sign * (base_log + log_half_erfc).exp();
            pair_s += term;
        }
        s_sum += pair_s;

        let s_scale = s_sum.abs().max(1.0);
        if pair_s.abs() <= 2e-15 * s_scale {
            stable_pairs += 1;
            if stable_pairs >= SERIES_CONSECUTIVE_SMALL_TERMS {
                if s_sum.is_finite() && (-1e-10..=1.0 + 1e-10).contains(&s_sum) {
                    return Ok(s_sum.clamp(0.0, 1.0));
                }
                break;
            }
        } else {
            stable_pairs = 0;
        }
    }

    Err(EstimationError::InvalidInput(
        "Miles cloglog series did not converge safely".to_string(),
    ))
}

fn cloglog_survival_cc(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
    tol: f64,
) -> Result<f64, EstimationError> {
    if !(mu.is_finite() && sigma.is_finite() && sigma > 0.0 && tol.is_finite() && tol > 0.0) {
        return Err(EstimationError::InvalidInput(
            "CC cloglog backend requires finite mu, positive sigma, and positive tolerance"
                .to_string(),
        ));
    }

    // Real-line representation of the shared survival term
    //
    //   S(mu, sigma)
    //     = 1/sqrt(2pi) ∫ exp(-t^2/2 - exp(mu + sigma t)) dt.
    //
    // This comes directly from eta = mu + sigma Z with Z ~ N(0,1):
    //
    //   S(mu, sigma)
    //     = E[exp(-exp(eta))]
    //     = 1/sqrt(2pi) ∫ exp(-t^2/2) exp(-exp(mu + sigma t)) dt.
    //
    // We truncate to [-A, A] using the Gaussian tail bound
    //
    //   ∫_{|t| > A} phi(t) exp(-exp(mu + sigma t)) dt <= 2 Phi(-A),
    //
    // then apply Clenshaw-Curtis on [-A, A] after the affine map t = A x.
    //
    // In other words, we first turn the infinite Gaussian expectation into a
    // finite interval problem, and then use a Chebyshev/cosine-grid quadrature
    // rule on that bounded interval. The mapped nodes x_j = cos(j pi / (n - 1))
    // become t_j = A x_j, which concentrates points near ±A where the cosine
    // grid is densest.
    //
    // The node count comes from the same Bernstein-ellipse style bound used in
    // the math notes: we pick a conservative ellipse height y so the mapped
    // integrand stays analytic in a strip where the double exponential term
    // does not blow up, convert that to rho, and request enough cosine nodes to
    // make the quadrature remainder smaller than the quadrature slice of `tol`.
    //
    // This mirrors the standard Clenshaw-Curtis error picture: after mapping to
    // [-1, 1], analyticity in a Bernstein ellipse controls how fast the
    // Chebyshev coefficients decay, which in turn controls how many cosine-grid
    // nodes are needed.
    //
    // So this backend is still computing the exact same scalar object as the
    // Gamma/Mellin-Barnes path below; it just works on the real integral rather
    // than the Bromwich contour representation.
    let p_tail = (tol / 8.0).clamp(1e-300, 0.25);
    let a = crate::probability::standard_normal_quantile(p_tail)
        .map(|z| -z)
        .unwrap_or(8.0)
        .max(1.0);
    let n = cloglog_cc_required_nodes(mu, sigma, tol)?;
    if n > CLOGLOG_CC_NODE_CAP {
        return Err(EstimationError::InvalidInput(
            "CC cloglog backend requires too many nodes".to_string(),
        ));
    }

    let rule = ctx.clenshaw_curtis_n(n);
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for (&x, &w) in rule.nodes.iter().zip(rule.weights.iter()) {
        let t = a * x;
        let u = mu + sigma * t;
        let e = if u > 709.78 { f64::INFINITY } else { u.exp() };
        let w0 = (-0.5 * t * t).exp() * inv_sqrt_2pi;
        let yk = w * w0 * (-e).exp() - c;
        let tk = sum + yk;
        c = (tk - sum) - yk;
        sum = tk;
    }

    let survival = (a * sum).clamp(0.0, 1.0);
    if !survival.is_finite() {
        return Err(EstimationError::InvalidInput(
            "CC cloglog backend produced non-finite values".to_string(),
        ));
    }
    Ok(survival)
}

#[inline]
fn complex_add(a: Complex, b: Complex) -> Complex {
    Complex {
        re: a.re + b.re,
        im: a.im + b.im,
    }
}

#[inline]
fn complex_sub(a: Complex, b: Complex) -> Complex {
    Complex {
        re: a.re - b.re,
        im: a.im - b.im,
    }
}

#[inline]
fn complex_mul(a: Complex, b: Complex) -> Complex {
    Complex {
        re: a.re * b.re - a.im * b.im,
        im: a.re * b.im + a.im * b.re,
    }
}

#[inline]
fn complex_div(a: Complex, b: Complex) -> Complex {
    let den = (b.re * b.re + b.im * b.im).max(1e-300);
    Complex {
        re: (a.re * b.re + a.im * b.im) / den,
        im: (a.im * b.re - a.re * b.im) / den,
    }
}

#[inline]
fn complex_abs(z: Complex) -> f64 {
    z.re.hypot(z.im)
}

#[inline]
fn complex_ln(z: Complex) -> Complex {
    Complex {
        re: complex_abs(z).ln(),
        im: z.im.atan2(z.re),
    }
}

#[inline]
fn complex_exp(z: Complex) -> Complex {
    let e = z.re.exp();
    Complex {
        re: e * z.im.cos(),
        im: e * z.im.sin(),
    }
}

#[inline]
fn complex_sin(z: Complex) -> Complex {
    Complex {
        re: z.re.sin() * z.im.cosh(),
        im: z.re.cos() * z.im.sinh(),
    }
}

fn complex_log_gamma_lanczos(z: Complex) -> Complex {
    // Reference-quality complex log-gamma for the Mellin-Barnes cloglog
    // backend. This is the key special-function primitive for evaluating the
    // exact Bromwich integral of the lognormal Laplace transform.
    const G: f64 = 7.0;
    const COEFFS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    if z.re < 0.5 {
        let pi_z = Complex {
            re: std::f64::consts::PI * z.re,
            im: std::f64::consts::PI * z.im,
        };
        let one_minus_z = Complex {
            re: 1.0 - z.re,
            im: -z.im,
        };
        return complex_sub(
            complex_sub(
                Complex {
                    re: std::f64::consts::PI.ln(),
                    im: 0.0,
                },
                complex_ln(complex_sin(pi_z)),
            ),
            complex_log_gamma_lanczos(one_minus_z),
        );
    }

    let z1 = Complex {
        re: z.re - 1.0,
        im: z.im,
    };
    let mut x = Complex {
        re: COEFFS[0],
        im: 0.0,
    };
    for (i, c) in COEFFS.iter().enumerate().skip(1) {
        x = complex_add(
            x,
            complex_div(
                Complex { re: *c, im: 0.0 },
                Complex {
                    re: z1.re + i as f64,
                    im: z1.im,
                },
            ),
        );
    }
    let t = Complex {
        re: z1.re + G + 0.5,
        im: z1.im,
    };
    complex_add(
        complex_add(
            Complex {
                re: 0.5 * (2.0 * std::f64::consts::PI).ln(),
                im: 0.0,
            },
            complex_mul(
                Complex {
                    re: z1.re + 0.5,
                    im: z1.im,
                },
                complex_ln(t),
            ),
        ),
        complex_sub(complex_ln(x), t),
    )
}

#[cfg(test)]
pub(crate) fn cloglog_posterior_mean_with_deriv_gamma_reference(
    mu: f64,
    sigma: f64,
) -> Result<IntegratedMeanDerivative, EstimationError> {
    // This backend uses the exact Mellin-Barnes/Gamma representation for the
    // same survival term S(mu, sigma). The integrated cloglog outputs are still
    // recovered by the same exact identities:
    //
    //   mean       = 1 - S(mu, sigma)
    //   dmean/dmu  = exp(mu + sigma^2 / 2) * S(mu + sigma^2, sigma).
    //
    // Using the shift identity here matters even more than in the Miles case.
    // Differentiating the Bromwich integral directly inserts an extra factor
    // (k + i t) into an already oscillatory complex integrand. That is
    // mathematically valid, but it needlessly makes the derivative a second,
    // separate special-function quadrature problem. Reusing the scalar survival
    // evaluator keeps mean and derivative on the same approximation surface and
    // preserves dmean/dmu >= 0 up to roundoff.
    let survival = cloglog_survival_gamma_reference(mu, sigma)?;
    let shifted_survival = cloglog_survival_gamma_reference(mu + sigma * sigma, sigma)?;
    let mean = cloglog_mean_from_survival(survival);
    let dmean = cloglog_shift_identity_derivative(mu, sigma, shifted_survival);
    if !(mean.is_finite() && dmean.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "Gamma cloglog reference backend produced non-finite values".to_string(),
        ));
    }
    Ok(IntegratedMeanDerivative {
        mean,
        dmean_dmu: dmean.max(0.0),
        mode: IntegratedExpectationMode::ExactSpecialFunction,
    })
}

fn cloglog_survival_gamma_reference(mu: f64, sigma: f64) -> Result<f64, EstimationError> {
    if !(mu.is_finite() && sigma.is_finite()) || sigma <= 0.0 {
        return Err(EstimationError::InvalidInput(
            "Gamma cloglog reference backend requires finite mu and positive sigma".to_string(),
        ));
    }

    // Exact Mellin-Barnes / Bromwich representation for the lognormal Laplace
    // transform at lambda = 1:
    //
    //   S(mu, sigma)
    //     = E[exp(-exp(eta))],   eta ~ N(mu, sigma^2)
    //     = 1/pi ∫_0^∞ Re[
    //         Γ(k + i t)
    //         exp(0.5 sigma^2 (k + i t)^2 - mu (k + i t))
    //       ] dt,
    //
    // with k > 0 fixed on the Bromwich line. This comes from the exact
    // Mellin-Barnes identity
    //
    //   L(z; mu, sigma)
    //     = (1 / 2πi) ∫ Γ(s) z^{-s} exp(-mu s + 0.5 sigma^2 s^2) ds,
    //
    // specialized to z = 1 and then rewritten on the vertical line s = k + it.
    // This is the exact special-function representation of the same
    // lognormal-Laplace object used by the large-sigma central cloglog path.
    //
    // The surrounding cloglog code only asks this routine for S itself. The
    // final outputs are reconstructed outside via
    //
    //   mean       = 1 - S(mu, sigma)
    //   dmean/dmu  = exp(mu + sigma^2 / 2) * S(mu + sigma^2, sigma),
    //
    // so the integral below remains a scalar survival evaluator.
    //
    // Numerically, the Γ(k + i t) factor decays like exp(-pi t / 2) on the
    // vertical line, while the Gaussian factor contributes exp(-0.5 sigma^2
    // t^2) in magnitude. That makes the tail rapidly damped, which is why a
    // fixed composite Simpson rule on [0, T] is adequate here despite the
    // complex oscillation.
    let n = (CLOGLOG_GAMMA_T_MAX_REF / CLOGLOG_GAMMA_H_REF).round() as usize;
    let n = if n.is_multiple_of(2) { n } else { n + 1 };
    let h = CLOGLOG_GAMMA_T_MAX_REF / n as f64;

    let eval = |t: f64| -> f64 {
        let z = Complex {
            re: CLOGLOG_GAMMA_K_REF,
            im: t,
        };
        let log_gamma = complex_log_gamma_lanczos(z);
        let z_sq = complex_mul(z, z);
        let exponent = complex_sub(
            complex_add(
                log_gamma,
                Complex {
                    re: 0.5 * sigma * sigma * z_sq.re,
                    im: 0.5 * sigma * sigma * z_sq.im,
                },
            ),
            Complex {
                re: mu * z.re,
                im: mu * z.im,
            },
        );
        complex_exp(exponent).re
    };

    let f0 = eval(0.0);
    let fn_ = eval(CLOGLOG_GAMMA_T_MAX_REF);
    let mut sum_s = f0 + fn_;
    for i in 1..n {
        let t = i as f64 * h;
        let fi = eval(t);
        let w = if i % 2 == 0 { 2.0 } else { 4.0 };
        sum_s += w * fi;
    }
    let s_val = ((h / 3.0) * sum_s / std::f64::consts::PI).clamp(0.0, 1.0);
    if !s_val.is_finite() {
        return Err(EstimationError::InvalidInput(
            "Gamma cloglog reference backend produced non-finite values".to_string(),
        ));
    }
    Ok(s_val)
}

pub(crate) fn cloglog_posterior_mean_with_deriv_controlled(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> IntegratedMeanDerivative {
    // Final production routing for integrated cloglog under Gaussian latent
    // uncertainty.
    //
    // The target quantity is always
    //
    //   mean(mu, sigma)      = E[1 - exp(-exp(eta))]
    //   dmean/dmu            = E[exp(eta - exp(eta))],
    //   eta ~ N(mu, sigma^2).
    //
    // Different numerical regimes favor different exact or controlled
    // representations of the same lognormal-Laplace object:
    //
    // 1. sigma ~= 0
    //    The Gaussian collapses to a point mass, so the ordinary inverse link
    //    and its pointwise derivative are exact.
    //
    // 2. small sigma
    //    The heat-kernel / Taylor expansion is efficient and tracks the true
    //    integrated mean and derivative to high accuracy without invoking any
    //    special-function machinery.
    //
    // 3. tail-dominated large sigma
    //    The Miles erfc-gated series is efficient because the erfc gate keeps
    //    the alternating lognormal-moment series short and numerically tame.
    //
    // 4. central large sigma
    //    The exact Mellin-Barnes / Gamma inversion is preferred, because the
    //    Miles series is no longer the best-behaved production representation.
    //
    // 5. final escape
    //    GHQ remains only as a numerical fallback if the chosen special-
    //    function backend returns a non-finite or non-converged result.
    //
    // This layered routing is the real conclusion of the math work: not one
    // magical universal formula, but one shared mathematical target with the
    // best evaluator chosen for each regime.
    if !(mu.is_finite() && sigma.is_finite()) || sigma <= CLOGLOG_SIGMA_DEGENERATE {
        let z = mu.clamp(-30.0, 30.0);
        let ez = z.exp();
        let surv = (-ez).exp();
        return IntegratedMeanDerivative {
            mean: (1.0 - surv).clamp(1e-12, 1.0 - 1e-12),
            dmean_dmu: (ez * surv).max(0.0),
            mode: IntegratedExpectationMode::ExactClosedForm,
        };
    }
    if sigma < CLOGLOG_SIGMA_TAYLOR_MAX {
        return cloglog_small_sigma_taylor(mu, sigma);
    }

    let ((survival, mode), (shifted_survival, shifted_mode)) =
        cloglog_survival_pair_controlled(ctx, mu, sigma);
    let mean = cloglog_mean_from_survival(survival);
    let dmean = cloglog_shift_identity_derivative(mu, sigma, shifted_survival);
    let mode = if matches!(mode, IntegratedExpectationMode::QuadratureFallback)
        || matches!(shifted_mode, IntegratedExpectationMode::QuadratureFallback)
    {
        IntegratedExpectationMode::QuadratureFallback
    } else if matches!(mode, IntegratedExpectationMode::ControlledAsymptotic)
        || matches!(
            shifted_mode,
            IntegratedExpectationMode::ControlledAsymptotic
        )
    {
        IntegratedExpectationMode::ControlledAsymptotic
    } else {
        mode
    };
    IntegratedMeanDerivative {
        mean,
        dmean_dmu: dmean.max(0.0),
        mode,
    }
}

pub fn integrated_inverse_link_mean_and_derivative(
    quad_ctx: &QuadratureContext,
    link: LinkFunction,
    mu: f64,
    sigma: f64,
) -> IntegratedMeanDerivative {
    // Canonical dispatcher for Gaussian-uncertain inverse-link expectations.
    //
    // Every integrated PIRLS and posterior-mean prediction path reduces to the
    // same mathematical contract:
    //
    //   input:
    //     eta ~ N(mu, sigma^2)
    //
    //   output:
    //     mean      = E[g^{-1}(eta)]
    //     dmean/dmu = E[(g^{-1})'(eta)].
    //
    // The location-family identity
    //
    //   d/dmu E[f(mu + sigma Z)] = E[f'(mu + sigma Z)]
    //
    // is what makes this sufficient for PIRLS. Once a link-specific backend can
    // return these two quantities, the generic Fisher weight and working
    // response formulas do not care whether they came from:
    //
    // - exact closed form,
    // - exact/special-function evaluation,
    // - a controlled asymptotic approximation,
    // - or GHQ fallback.
    //
    // Centralizing the routing here keeps all link-specific special-function
    // mathematics local to one module instead of leaking into PIRLS or
    // prediction code.
    match link {
        LinkFunction::Probit => probit_posterior_mean_with_deriv_exact(mu, sigma),
        // The in-repo logit special-function backend is useful as a research
        // implementation and local oracle, but it is not yet uniformly accurate
        // enough across the production domain to replace GHQ in the hot path.
        // Use the exact plug-in limit when sigma is effectively zero and route
        // everything else through the validated GHQ backend.
        LinkFunction::Logit if sigma <= LOGIT_SIGMA_DEGENERATE => {
            logit_posterior_mean_with_deriv_exact(mu, sigma).unwrap_or_else(|_| {
                let mean = sigmoid(mu);
                IntegratedMeanDerivative {
                    mean,
                    dmean_dmu: mean * (1.0 - mean),
                    mode: IntegratedExpectationMode::ExactClosedForm,
                }
            })
        }
        LinkFunction::Logit => {
            let (mean, dmean_dmu) = logit_posterior_mean_with_deriv(quad_ctx, mu, sigma);
            IntegratedMeanDerivative {
                mean,
                dmean_dmu,
                mode: IntegratedExpectationMode::QuadratureFallback,
            }
        }
        LinkFunction::CLogLog => cloglog_posterior_mean_with_deriv_controlled(quad_ctx, mu, sigma),
        LinkFunction::Identity => IntegratedMeanDerivative {
            mean: mu,
            dmean_dmu: 1.0,
            mode: IntegratedExpectationMode::ExactClosedForm,
        },
    }
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
        let integrated = integrated_inverse_link_mean_and_derivative(
            ctx,
            LinkFunction::Logit,
            eta[i],
            se_eta[i],
        );
        mu[i] = integrated.mean;
        dmu[i] = integrated.dmean_dmu;
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
    ndarray::Zip::from(eta).and(se_eta).map_collect(|&e, &se| {
        integrated_inverse_link_mean_and_derivative(ctx, LinkFunction::Logit, e, se).mean
    })
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
///
/// This is the template for the "integrated PIRLS without quadrature" idea:
/// unlike logit/cloglog, the Gaussian convolution of a probit inverse link is
/// analytically closed and cheap enough to evaluate as a plain vectorized
/// transformation. Any integrated probit update path should use this exact
/// identity rather than GHQ or cubature.
///
/// Derivation:
/// Let U ~ N(0, 1) independent of Z ~ N(eta, se_eta^2). Then
///   E[Phi(Z)] = P(U <= Z) = P(Z - U >= 0).
/// Since Z - U ~ N(eta, 1 + se_eta^2),
///   P(Z - U >= 0) = Phi(eta / sqrt(1 + se_eta^2)).
/// Differentiating with respect to eta gives
///   d/deta E[Phi(Z)]
///   = phi(eta / sqrt(1 + se_eta^2)) / sqrt(1 + se_eta^2),
/// which is exactly the integrated derivative IRLS would need.
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
    // With p(eta) = 1 - S(eta), where S(eta) = exp(-exp(eta)),
    //
    //   E[p]   = 1 - E[S]
    //   E[p^2] = E[(1 - S)^2] = 1 - 2 E[S] + E[S^2]
    //
    // and because
    //
    //   S(eta)^2 = exp(-2 exp(eta)) = L(2; mu, sigma) = L(1; mu + ln 2, sigma),
    //
    // the second moment is obtained by the same shared survival-term
    // evaluator with the exact mu -> mu + ln 2 shift. The variance then
    // collapses to
    //
    //   Var[p] = E[p^2] - E[p]^2 = E[S^2] - E[S]^2.
    //
    // So cloglog and survival actually share the same posterior variance under
    // Gaussian uncertainty; they only differ in whether the reported mean is
    // E[S] or 1 - E[S].
    let (survival, _) = cloglog_survival_term_controlled(ctx, eta, se_eta);
    let (survival_sq, _) = cloglog_survival_second_moment_controlled(ctx, eta, se_eta);
    let mean = cloglog_mean_from_survival(survival).clamp(1e-10, 1.0 - 1e-10);
    let variance = (survival_sq - survival * survival).max(0.0);
    (mean, variance)
}

/// Posterior mean under cloglog inverse link:
/// g^{-1}(x) = 1 - exp(-exp(x)).
///
/// This currently uses GHQ because it is robust and easy to share with the
/// rest of the uncertainty-aware prediction code. However, cloglog is another
/// strong candidate for eliminating quadrature from the integrated hot path:
///
/// - E[1 - exp(-exp(eta))] under Gaussian eta is the complement of the
///   lognormal Laplace transform at z=1.
/// - That quantity has exact non-GHQ representations, including convergent
///   erfc / asymptotic series and characteristic-function inversion formulas.
/// - The same mathematics also covers the Royston-Parmar survival transform
///   S(eta) = exp(-exp(eta)), which is why this comment matters beyond binary
///   cloglog models.
///
/// So GHQ here is a general fallback, not the final intended end state.
///
/// Derivation of the exact target quantity:
/// If eta = mu + sigma Z with Z ~ N(0, 1), set X = exp(eta). Then
///   X ~ LogNormal(mu, sigma^2)
/// and
///   E[1 - exp(-exp(eta))] = 1 - E[exp(-X)].
/// So the integrated cloglog mean is exactly the complement of the Laplace
/// transform of a lognormal random variable at z = 1.
///
/// The integrated derivative needed by IRLS is
///   d/dmu E[1 - exp(-exp(eta))]
///   = E[exp(eta - exp(eta))],
/// either by differentiating inside the Gaussian expectation or directly from
/// f'(x) = exp(x - exp(x)).
///
/// There is no simple elementary closed form, but the object is exact and well
/// structured. That is why this function is a good future target for replacing
/// repeated GHQ with a special-function or rapidly convergent series backend.
#[inline]
pub fn cloglog_posterior_mean(ctx: &QuadratureContext, eta: f64, se_eta: f64) -> f64 {
    let (survival, _) = cloglog_survival_term_controlled(ctx, eta, se_eta);
    cloglog_mean_from_survival(survival).clamp(1e-10, 1.0 - 1e-10)
}

/// Posterior mean under the Royston-Parmar survival transform:
/// S(x) = exp(-exp(x)).
///
/// This is the cloglog complement:
///   1 - S(x) = 1 - exp(-exp(x)).
/// Therefore for Gaussian eta,
///   E[S(eta)] = E[exp(-exp(eta))]
/// is the same lognormal-Laplace-transform object that appears in the cloglog
/// path, and
///   E[cloglog^{-1}(eta)] = 1 - E[S(eta)].
///
/// Any future exact special-function implementation for integrated cloglog can
/// therefore be shared directly with survival models that use this transform.
#[inline]
pub fn survival_posterior_mean(ctx: &QuadratureContext, eta: f64, se_eta: f64) -> f64 {
    cloglog_survival_term_controlled(ctx, eta, se_eta)
        .0
        .clamp(0.0, 1.0)
}

#[inline]
pub fn survival_posterior_mean_variance(
    ctx: &QuadratureContext,
    eta: f64,
    se_eta: f64,
) -> (f64, f64) {
    let (m1, _) = cloglog_survival_term_controlled(ctx, eta, se_eta);
    let (m2, _) = cloglog_survival_second_moment_controlled(ctx, eta, se_eta);
    (m1.clamp(0.0, 1.0), (m2 - m1 * m1).max(0.0))
}

/// Oracle-only exact logistic-normal mean using the Faddeeva-series representation.
///
/// For η ~ N(mu, sigma^2):
///   E[sigmoid(η)] = 1/2 - (sqrt(2π)/sigma) * Σ_{n>=1} Im[w((i a_n - mu)/(sqrt(2)sigma))]
/// where a_n = (2n-1)π and w is the Faddeeva function.
///
/// This function is intentionally kept as a math/reference implementation:
/// it documents the exact non-GHQ route that an optimized integrated logit IRLS
/// path would eventually use. The practical migration path is:
///
/// 1. prove machine-precision truncation criteria on the production domain,
/// 2. expose the derivative d/dmu E[sigmoid(eta)] alongside the mean,
/// 3. replace the GHQ loop in `logit_posterior_mean_with_deriv` once the exact
///    special-function path is benchmarked and regression-tested thoroughly.
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
    fn test_clenshaw_curtis_weights_are_symmetric_and_integrate_constants() {
        let rule = compute_clenshaw_curtis_n(33);
        let m = rule.weights.len() - 1;
        for j in 0..=m / 2 {
            assert_relative_eq!(rule.nodes[j], -rule.nodes[m - j], epsilon = 1e-14);
            assert_relative_eq!(rule.weights[j], rule.weights[m - j], epsilon = 1e-14);
        }
        let sum: f64 = rule.weights.iter().sum();
        assert_relative_eq!(sum, 2.0, epsilon = 1e-14, max_relative = 1e-14);
    }

    #[test]
    fn test_cc_preference_prefers_moderate_central_case() {
        assert!(cloglog_should_prefer_cc(-0.2, 0.8, CLOGLOG_CC_TOL));
    }

    #[test]
    fn test_cc_preference_prefers_moderately_large_case() {
        assert!(cloglog_should_prefer_cc(0.0, 2.0, CLOGLOG_CC_TOL));
    }

    #[test]
    fn test_cc_preference_rejects_broad_case() {
        assert!(!cloglog_should_prefer_cc(0.0, 5.0, CLOGLOG_CC_TOL));
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

    #[test]
    fn test_cloglog_and_survival_posterior_means_are_complements() {
        let ctx = QuadratureContext::new();
        let cases = [
            (-3.0, 0.0),
            (-0.2, 0.1),
            (0.4, 0.8),
            (2.0, 1.5),
            (10.0, 0.3),
        ];
        for (eta, se) in cases {
            let clog = cloglog_posterior_mean(&ctx, eta, se);
            let surv = survival_posterior_mean(&ctx, eta, se);
            assert_relative_eq!(clog + surv, 1.0, epsilon = 2e-10, max_relative = 2e-10);
        }
    }

    #[test]
    fn test_cloglog_and_survival_share_large_sigma_special_function_path() {
        let ctx = QuadratureContext::new();
        let eta = -0.2;
        let se = 0.8;
        let clog = cloglog_posterior_mean(&ctx, eta, se);
        let surv = survival_posterior_mean(&ctx, eta, se);
        let integrated =
            integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::CLogLog, eta, se);
        assert_eq!(
            integrated.mode,
            IntegratedExpectationMode::ExactSpecialFunction
        );
        assert_relative_eq!(clog, integrated.mean, epsilon = 1e-12, max_relative = 1e-12);
        assert_relative_eq!(clog + surv, 1.0, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn test_cloglog_and_survival_posterior_variances_match() {
        let ctx = QuadratureContext::new();
        let cases = [(-3.0, 0.0), (-0.2, 0.1), (0.4, 0.8), (2.0, 1.5)];
        for (eta, se) in cases {
            let (_, clog_var) = cloglog_posterior_mean_variance(&ctx, eta, se);
            let (_, surv_var) = survival_posterior_mean_variance(&ctx, eta, se);
            assert_relative_eq!(clog_var, surv_var, epsilon = 1e-12, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_survival_variance_uses_exact_second_moment_shift() {
        let ctx = QuadratureContext::new();
        let eta = -0.2;
        let se = 0.8;
        let (survival, _) = cloglog_survival_term_controlled(&ctx, eta, se);
        let (survival_sq, _) = cloglog_survival_second_moment_controlled(&ctx, eta, se);
        let (_, variance) = survival_posterior_mean_variance(&ctx, eta, se);
        assert_relative_eq!(
            variance,
            (survival_sq - survival * survival).max(0.0),
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    fn test_lognormal_laplace_shift_matches_explicit_mu_plus_log_z() {
        let ctx = QuadratureContext::new();
        let mu = -0.2;
        let sigma = 0.8;
        let z = 2.0;
        let shifted = lognormal_laplace_term_controlled(&ctx, z, mu, sigma);
        let explicit = cloglog_survival_term_controlled(&ctx, mu + z.ln(), sigma);
        assert_eq!(shifted.1, explicit.1);
        assert_relative_eq!(shifted.0, explicit.0, epsilon = 1e-12, max_relative = 1e-12);
    }

    #[test]
    fn test_integrated_dispatch_uses_closed_form_probit() {
        let ctx = QuadratureContext::new();
        let out = integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Probit, 0.7, 1.3);
        assert_eq!(out.mode, IntegratedExpectationMode::ExactClosedForm);
        let direct = probit_posterior_mean_with_deriv_exact(0.7, 1.3);
        assert_relative_eq!(out.mean, direct.mean, epsilon = 1e-12);
        assert_relative_eq!(out.dmean_dmu, direct.dmean_dmu, epsilon = 1e-12);
    }

    #[test]
    fn test_logit_exact_derivative_matches_finite_difference() {
        let out = logit_posterior_mean_with_deriv_exact(1.1, 0.8).expect("exact logit");
        let h = 1e-5;
        let plus = logit_posterior_mean_with_deriv_exact(1.1 + h, 0.8)
            .expect("exact logit plus")
            .mean;
        let minus = logit_posterior_mean_with_deriv_exact(1.1 - h, 0.8)
            .expect("exact logit minus")
            .mean;
        let fd = (plus - minus) / (2.0 * h);
        assert_relative_eq!(out.dmean_dmu, fd, epsilon = 1e-6, max_relative = 5e-4);
    }

    #[test]
    fn test_cloglog_controlled_matches_ghq_small_sigma() {
        let ctx = QuadratureContext::new();
        let approx = cloglog_posterior_mean_with_deriv_controlled(&ctx, 0.4, 0.1);
        let exact = cloglog_posterior_mean_with_deriv_ghq(&ctx, 0.4, 0.1);
        assert_relative_eq!(approx.mean, exact.mean, epsilon = 2e-4, max_relative = 2e-4);
        assert_relative_eq!(
            approx.dmean_dmu,
            exact.dmean_dmu,
            epsilon = 2e-3,
            max_relative = 2e-3
        );
    }

    #[test]
    fn test_cloglog_controlled_matches_ghq_on_small_sigma_grid() {
        let ctx = QuadratureContext::new();
        let mus = [-30.0, -10.0, -3.0, 0.0, 3.0, 10.0, 30.0];
        let sigmas = [1e-10, 1e-8, 0.01, 0.05, 0.1, 0.2, 0.24];

        for &mu in &mus {
            for &sigma in &sigmas {
                let approx = cloglog_posterior_mean_with_deriv_controlled(&ctx, mu, sigma);
                let ghq = cloglog_posterior_mean_with_deriv_ghq(&ctx, mu, sigma);
                assert_relative_eq!(approx.mean, ghq.mean, epsilon = 2e-3, max_relative = 2e-3);
                assert_relative_eq!(
                    approx.dmean_dmu,
                    ghq.dmean_dmu,
                    epsilon = 4e-3,
                    max_relative = 4e-3
                );
            }
        }
    }

    #[test]
    fn test_cloglog_dispatch_uses_gamma_backend_for_large_sigma_central_regime() {
        let ctx = QuadratureContext::new();
        let out =
            integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::CLogLog, -0.2, 0.8);
        assert_eq!(out.mode, IntegratedExpectationMode::ExactSpecialFunction);
        assert!(out.mean.is_finite());
        assert!(out.dmean_dmu.is_finite());
        assert!(out.dmean_dmu >= 0.0);
    }

    #[test]
    fn test_cloglog_cc_matches_gamma_reference_on_central_case() {
        let ctx = QuadratureContext::new();
        let mu = -0.2;
        let sigma = 0.8;
        let cc = cloglog_survival_cc(&ctx, mu, sigma, CLOGLOG_CC_TOL).expect("cc backend");
        let gamma = cloglog_survival_gamma_reference(mu, sigma).expect("gamma backend");
        assert_relative_eq!(cc, gamma, epsilon = 5e-6, max_relative = 5e-6);
    }

    #[test]
    fn test_cloglog_gamma_reference_matches_seeded_monte_carlo_small_case() {
        let mu = -0.2;
        let sigma = 0.8;
        let gamma =
            cloglog_posterior_mean_with_deriv_gamma_reference(mu, sigma).expect("gamma reference");
        let mut rng_state = 0x9e3779b97f4a7c15u64;
        let mut mean_mc = 0.0f64;
        let mut deriv_mc = 0.0f64;
        let n_samples = 300_000usize;
        for _ in 0..n_samples {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = ((rng_state as f64) / (u64::MAX as f64)).clamp(1e-12, 1.0 - 1e-12);
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = ((rng_state as f64) / (u64::MAX as f64)).clamp(1e-12, 1.0 - 1e-12);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            let eta = mu + sigma * z;
            let ez = eta.clamp(-30.0, 30.0).exp();
            mean_mc += 1.0 - (-ez).exp();
            deriv_mc += ez * (-ez).exp();
        }
        mean_mc /= n_samples as f64;
        deriv_mc /= n_samples as f64;
        assert_relative_eq!(gamma.mean, mean_mc, epsilon = 2e-3, max_relative = 2e-3);
        assert_relative_eq!(
            gamma.dmean_dmu,
            deriv_mc,
            epsilon = 2e-3,
            max_relative = 2e-3
        );
    }

    #[test]
    fn test_logit_dispatch_falls_back_outside_guarded_domain() {
        let ctx = QuadratureContext::new();
        let out = integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, 35.0, 1.0);
        assert_eq!(out.mode, IntegratedExpectationMode::QuadratureFallback);
        assert!(out.mean.is_finite());
        assert!(out.dmean_dmu.is_finite());
        assert!(out.dmean_dmu >= 0.0);
    }

    #[test]
    fn test_logit_dispatch_prefers_ghq_in_non_degenerate_regime() {
        let ctx = QuadratureContext::new();
        let out = integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, 1.1, 0.8);
        assert_eq!(out.mode, IntegratedExpectationMode::QuadratureFallback);
        assert!(out.mean.is_finite());
        assert!(out.dmean_dmu.is_finite());
        assert!(out.dmean_dmu >= 0.0);
    }

    #[test]
    fn test_logit_batch_uses_same_dispatch_values() {
        let ctx = QuadratureContext::new();
        let eta = ndarray::array![-2.0, 0.0, 1.25, 35.0];
        let se = ndarray::array![0.1, 0.5, 1.0, 1.0];
        let batch_mean = logit_posterior_mean_batch(&ctx, &eta, &se);
        let (batch_mu, batch_dmu) = logit_posterior_mean_with_deriv_batch(&ctx, &eta, &se);
        for i in 0..eta.len() {
            let direct = integrated_inverse_link_mean_and_derivative(
                &ctx,
                LinkFunction::Logit,
                eta[i],
                se[i],
            );
            assert_relative_eq!(batch_mean[i], direct.mean, epsilon = 1e-12);
            assert_relative_eq!(batch_mu[i], direct.mean, epsilon = 1e-12);
            assert_relative_eq!(batch_dmu[i], direct.dmean_dmu, epsilon = 1e-12);
        }
    }
}
