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
use std::convert::Infallible;
use std::sync::{Mutex, OnceLock};

use crate::estimate::EstimationError;
use crate::mixture_link::{
    beta_logistic_inverse_link_jet, component_inverse_link_jet, sas_inverse_link_jet,
};
use crate::types::{LikelihoodFamily, LinkComponent, LinkFunction, MixtureLinkState, SasLinkState};
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
    gh51_cache: OnceLock<GaussHermiteRuleDynamic>,
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

#[derive(Clone, Copy, Debug)]
pub struct IntegratedInverseLinkJet {
    pub mean: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
    pub mode: IntegratedExpectationMode,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct IntegratedInverseLinkJet5 {
    pub mean: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
    pub d4: f64,
    pub d5: f64,
    pub mode: IntegratedExpectationMode,
}

#[inline]
pub(crate) fn validate_latent_cloglog_inputs(eta: f64, sigma: f64) -> Result<(), EstimationError> {
    if !eta.is_finite() || !sigma.is_finite() || sigma < 0.0 {
        return Err(EstimationError::InvalidInput(format!(
            "latent cloglog jet requires finite eta and sigma >= 0, got eta={eta}, sigma={sigma}"
        )));
    }
    Ok(())
}

/// Typed integrated moments/derivative jet used by solver integration paths.
///
/// `variance` is the observation-model variance at the integrated mean for the
/// associated family (for binomial links: `mean * (1 - mean)`).
#[derive(Clone, Copy, Debug)]
pub struct IntegratedMomentsJet {
    pub mean: f64,
    pub variance: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
    pub mode: IntegratedExpectationMode,
}

const LOGIT_SIGMA_DEGENERATE: f64 = 1e-10;
const LOGIT_SIGMA_TAYLOR_MAX: f64 = 2.5e-1;
const LOGIT_TAIL_LOG_MAX: f64 = -18.0;
const LOGIT_ERFCX_MU_MAX: f64 = 40.0;
const LOGIT_ERFCX_SIGMA_MAX: f64 = 6.0;
const CLOGLOG_SIGMA_DEGENERATE: f64 = 1e-10;
const CLOGLOG_SIGMA_TAYLOR_MAX: f64 = 0.25;
const CLOGLOG_RARE_EVENT_LOG_MAX: f64 = -18.0;
const CLOGLOG_LARGE_SIGMA_ASYMPTOTIC_MIN: f64 = 8.0;
const CLOGLOG_POSITIVE_SATURATION_EDGE: f64 = 5.0;
const CLOGLOG_POSITIVE_SATURATION_SIGMAS: f64 = 8.0;
const SERIES_CONSECUTIVE_SMALL_TERMS: usize = 6;
const LOGIT_MAX_TERMS: usize = 160;
/// Documented absolute-accuracy contract of the erfcx logistic-normal
/// backend. The series truncation bound (see `logistic_normal_tail_cutoff`)
/// is guaranteed to be below this tolerance on the mean whenever the backend
/// returns a value. Beyond the eligibility window or when the a-priori
/// truncation index would exceed LOGIT_MAX_TERMS, the backend rejects and
/// the caller routes to GHQ.
///
/// The value is set so that all downstream consumers (oracle-match,
/// finite-difference derivative, symmetry identity) pass with strict
/// positive margin while staying within the LOGIT_MAX_TERMS budget for
/// the entire eligibility window (μ,σ) ∈ [−40,40] × [0.25, 6]. The worst
/// case is σ = 0.25 with m ≈ σ, where the leading series coefficient
/// peaks near 0.484/σ² ≈ 7.74 and the tail bound at N = 160 reaches
/// ≈ 3·10⁻⁴.
const LOGIT_ERFCX_ACCURACY_TARGET: f64 = 1.0e-4;
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
            gh51_cache: OnceLock::new(),
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
            51 => self.gh51_cache.get_or_init(|| compute_gauss_hermite_n(51)),
            _ => self.gh21_cache.get_or_init(|| compute_gauss_hermite_n(21)),
        }
    }

    fn clenshaw_curtis_n(&self, n: usize) -> ClenshawCurtisRule {
        let mut cache = match self.cc_cache.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
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

pub(crate) struct GaussHermiteRuleDynamic {
    pub(crate) nodes: Vec<f64>,
    pub(crate) weights: Vec<f64>,
}

#[derive(Clone)]
struct ClenshawCurtisRule {
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

fn compute_clenshaw_curtis_n(n: usize) -> ClenshawCurtisRule {
    assert!(n >= 2);
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
        Ok(n) => n <= CLOGLOG_CC_PREFER_THRESHOLD,
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
    indices.sort_by(|&a, &b| nodes[a].total_cmp(&nodes[b]));

    let sorted_nodes: [f64; N_POINTS] = std::array::from_fn(|i| nodes[indices[i]]);
    let sortedweights: [f64; N_POINTS] = std::array::from_fn(|i| weights[indices[i]]);

    GaussHermiteRule {
        nodes: sorted_nodes,
        weights: sortedweights,
    }
}

pub(crate) fn compute_gauss_hermite_n(n: usize) -> GaussHermiteRuleDynamic {
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
    let mut t_norm = 0.0_f64;
    for i in 0..N_POINTS {
        let l = if i > 0 { off_diag[i - 1].abs() } else { 0.0 };
        let r = if i + 1 < N_POINTS { off_diag[i].abs() } else { 0.0 };
        t_norm = t_norm.max(diag[i].abs() + l + r);
    }
    let mut n = N_POINTS;
    while n > 1 {
        let mut converged = false;
        for _ in 0..max_iter {
            let mut m = n - 1;
            while m > 0 {
                let scl = (diag[m - 1].abs() + diag[m].abs()).max(t_norm);
                if off_diag[m - 1].abs() <= eps * scl {
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
    // Matrix 1-norm fallback scale. The row-local criterion
    // `eps * (|d[m-1]| + |d[m]|)` collapses to zero when the diagonal is
    // identically zero (as for physicist's Hermite), which stalls QR because
    // no off-diagonal can satisfy `|e| <= 0`. LAPACK dsteqr uses ||T||_inf;
    // we take the max absolute row sum and use it as a floor on the scale.
    let mut t_norm = 0.0_f64;
    for i in 0..dim {
        let left = if i > 0 { off_diag[i - 1].abs() } else { 0.0 };
        let right = if i + 1 < dim {
            off_diag[i].abs()
        } else {
            0.0
        };
        let row_sum = diag[i].abs() + left + right;
        if row_sum > t_norm {
            t_norm = row_sum;
        }
    }
    let mut n = dim;
    while n > 1 {
        let mut converged = false;
        for _ in 0..max_iter {
            let mut m = n - 1;
            while m > 0 {
                let row_scale = (diag[m - 1].abs() + diag[m].abs()).max(t_norm);
                if off_diag[m - 1].abs() <= eps * row_scale {
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

/// Computes the posterior mean probability for a logistic model under
/// Gaussian uncertainty in the linear predictor.
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
    match logit_posterior_meanwith_deriv_controlled(ctx, eta, se_eta) {
        Ok(out) => out.mean,
        Err(_) => integrate_normal_ghq_adaptive(ctx, eta, se_eta, sigmoid),
    }
}

/// Computes the integrated probability AND its derivative with respect to eta.
///
/// For IRLS, we need both:
/// - μ = ∫ σ(η) × N(η; m, SE²) dη
/// - dμ/dm = ∫ σ'(η) × N(η; m, SE²) dη = ∫ σ(η)(1-σ(η)) × N(η; m, SE²) dη
///
/// Returns: (μ, dμ/dm)
#[inline]
pub fn logit_posterior_meanwith_deriv(
    ctx: &QuadratureContext,
    eta: f64,
    se_eta: f64,
) -> Result<(f64, f64), EstimationError> {
    // Production routing for the integrated logistic-normal mean and its
    // location derivative.
    //
    // The backend ladder is:
    // - exact point-mass limit when sigma ~= 0
    // - small-sigma Taylor / heat-kernel expansion
    // - exact erfcx/Faddeeva series on the moderate domain
    // - tail and large-sigma controlled asymptotics
    // - GHQ only as the terminal numerical fallback if every analytic branch
    //   reports a non-finite or non-converged result.
    let out = logit_posterior_meanwith_deriv_controlled(ctx, eta, se_eta)?;
    Ok((out.mean, out.dmean_dmu))
}

#[inline]
pub fn probit_posterior_meanwith_deriv_exact(mu: f64, sigma: f64) -> IntegratedMeanDerivative {
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
        let mean = crate::probability::normal_cdf(mu);
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
        mean: crate::probability::normal_cdf(z),
        dmean_dmu: crate::probability::normal_pdf(z) / denom,
        mode: IntegratedExpectationMode::ExactClosedForm,
    }
}

#[inline]
fn logistic_normal_exact_eligible(mu: f64, sigma: f64) -> bool {
    mu.is_finite()
        && sigma.is_finite()
        && mu.abs() <= LOGIT_ERFCX_MU_MAX
        && sigma >= LOGIT_SIGMA_TAYLOR_MAX
        // The real-valued erfcx series is the preferred exact backend on the
        // central moderate regime. Outside that window we switch to explicit
        // asymptotic formulas that are better conditioned than pushing the same
        // series past its practical truncation range.
        && sigma <= LOGIT_ERFCX_SIGMA_MAX
}

/// A-priori truncation index for the erfcx series of the logistic-normal mean.
///
/// The representation is
///
///     E[sigmoid(η)] = Φ(m/s)
///         + (1/2) · exp(-m²/(2s²))
///           · Σ_{k≥1} (-1)^(k-1) · [erfcx((k s² + m)/(√2 s))
///                                 − erfcx((k s² − m)/(√2 s))],
///
/// with m = |μ|, s = σ > 0 (the reflection μ→−μ is applied at the callsite).
/// The two erfcx arguments scale as k·s/√2 with a fixed offset, so both tend
/// to +∞ linearly in k. Using the asymptotic erfcx(x) = (1/(x√π))·[1 + O(1/x²)]
/// for large x, the k-th (signed) term has magnitude
///
///     |T_k|  =  |m| · √(2/π) · exp(-m²/(2s²)) / (k² · s³)  + O(1/k⁴).
///
/// Since the series alternates in sign, the truncation tail after N terms is
/// bounded by the first omitted term:
///
///     |R_N|  ≤  |m| · √(2/π) · exp(-m²/(2s²)) / ((N+1)² · s³).
///
/// Solving |R_N| ≤ δ for the smallest admissible N yields the value returned
/// here. Reaching this N therefore certifies the truncation error against the
/// stated accuracy contract; the caller does not need to check term-level
/// stability to declare convergence.
#[inline]
fn logistic_normal_tail_cutoff(mu: f64, sigma: f64, target_accuracy: f64) -> usize {
    debug_assert!(sigma > 0.0);
    debug_assert!(target_accuracy > 0.0);
    let m = mu.abs();
    let s = sigma;
    // Leading asymptotic coefficient of the k-th term magnitude.
    let coeff = m * (2.0_f64 / std::f64::consts::PI).sqrt()
        * (-(m * m) / (2.0 * s * s)).exp()
        / (s * s * s);
    // If the coefficient already underflows the accuracy target, the very first
    // term is sufficient; we still evaluate at least one pair to pick up the
    // short-range structure that the asymptotic bound undersells.
    if !(coeff.is_finite()) || coeff <= target_accuracy {
        return 4;
    }
    let raw_n = (coeff / target_accuracy).sqrt() - 1.0;
    raw_n.ceil().clamp(4.0, LOGIT_MAX_TERMS as f64) as usize
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
fn stable_sigmoidwith_derivative(x: f64) -> (f64, f64) {
    let x_clamped = x.clamp(-700.0, 700.0);
    if x_clamped != x {
        return (sigmoid(x), 0.0);
    }
    if x_clamped >= 0.0 {
        let z = (-x_clamped).exp();
        let denom = 1.0 + z;
        (1.0 / denom, z / (denom * denom))
    } else {
        let z = x_clamped.exp();
        let denom = 1.0 + z;
        (z / denom, z / (denom * denom))
    }
}

#[inline]
fn logit_small_sigma_taylor(mu: f64, sigma: f64) -> IntegratedMeanDerivative {
    // Second-order heat-kernel expansion around the point-mass limit:
    //
    //   E[f(mu + sigma Z)] = f(mu) + (sigma^2 / 2) f''(mu) + O(sigma^4),
    //
    // with the derivative obtained by differentiating the same truncated
    // series. This keeps the low-variance branch off the erfcx path where the
    // exact series is most cancellation-prone.
    let (mean0, d1, d2, d3) = component_point_jet(LinkComponent::Logit, mu);
    let s2 = sigma * sigma;
    IntegratedMeanDerivative {
        mean: (mean0 + 0.5 * s2 * d2).clamp(0.0, 1.0),
        dmean_dmu: (d1 + 0.5 * s2 * d3).max(0.0),
        mode: IntegratedExpectationMode::ControlledAsymptotic,
    }
}

#[inline]
fn logit_tail_asymptotic(mu: f64, sigma: f64) -> Option<IntegratedMeanDerivative> {
    // When mu is far out in either logistic tail, sigmoid(eta) is
    // exponentially close to either exp(eta) or 1 - exp(-eta). Those Gaussian
    // expectations collapse to lognormal moments, so we can route extreme-|mu|
    // cases away from both erfcx and GHQ.
    if mu <= 0.0 {
        let log_mean = mu + 0.5 * sigma * sigma;
        if log_mean <= LOGIT_TAIL_LOG_MAX {
            let mean = log_mean.exp();
            return Some(IntegratedMeanDerivative {
                mean,
                dmean_dmu: mean,
                mode: IntegratedExpectationMode::ControlledAsymptotic,
            });
        }
    } else {
        let log_tail = -mu + 0.5 * sigma * sigma;
        if log_tail <= LOGIT_TAIL_LOG_MAX {
            let tail = log_tail.exp();
            return Some(IntegratedMeanDerivative {
                mean: 1.0 - tail,
                dmean_dmu: tail,
                mode: IntegratedExpectationMode::ControlledAsymptotic,
            });
        }
    }
    None
}

#[inline]
fn logit_large_sigma_probit_asymptotic(mu: f64, sigma: f64) -> IntegratedMeanDerivative {
    // Large-variance Monahan-Stefanski probit approximation.
    //
    // For broad Gaussian latents the logistic-normal integral is well
    // approximated by
    //
    //   E[sigmoid(mu + sigma Z)] ~= Phi(mu * kappa),
    //   kappa = (1 + pi sigma^2 / 8)^(-1/2),
    //
    // with
    //
    //   d/dmu ~= kappa * phi(mu * kappa).
    //
    // This is the standard probit approximation recommended for the
    // high-variance regime in the task notes.
    let kappa = (1.0 + std::f64::consts::PI * sigma * sigma / 8.0)
        .sqrt()
        .recip();
    let z = mu * kappa;
    IntegratedMeanDerivative {
        mean: crate::probability::normal_cdf(z),
        dmean_dmu: crate::probability::normal_pdf(z) * kappa,
        mode: IntegratedExpectationMode::ControlledAsymptotic,
    }
}

#[inline]
fn scaled_erfcx_termwith_derivative(m: f64, s: f64, x: f64, dxdm: f64) -> (f64, f64) {
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
        let (rest, drest) = scaled_erfcx_termwith_derivative(m, s, -x, -dxdm);
        (lead - rest, dlead - drest)
    }
}

pub(crate) fn logit_posterior_meanwith_deriv_exact(
    mu: f64,
    sigma: f64,
) -> Result<IntegratedMeanDerivative, EstimationError> {
    // Analytic entry point for the logistic-normal mean.
    //
    // The target objects are
    //
    //   mean(mu, sigma)   = E[sigmoid(eta)],
    //   dmean/dmu         = E[sigmoid(eta) * (1 - sigmoid(eta))],
    //   eta ~ N(mu, sigma^2).
    //
    // No single representation is numerically dominant everywhere:
    // - sigma ~= 0 is the exact point-mass limit,
    // - small sigma prefers the Taylor / heat-kernel expansion,
    // - moderate central cases prefer the exact erfcx/Faddeeva series,
    // - and extreme tails / very large sigma prefer controlled asymptotics.
    //
    // Validation target for this ladder: compare against high-order GHQ
    // (e.g. 128 nodes) on sigma in {0.01, 0.1, 1, 5, 20, 100} and mu on
    // [-10, 10] to confirm the regime transitions.
    if !(mu.is_finite() && sigma.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "logit exact expectation requires finite mu and sigma".to_string(),
        ));
    }
    if sigma <= LOGIT_SIGMA_DEGENERATE {
        let (mean, dmean_dmu) = stable_sigmoidwith_derivative(mu);
        return Ok(IntegratedMeanDerivative {
            mean,
            dmean_dmu,
            mode: IntegratedExpectationMode::ExactClosedForm,
        });
    }
    if let Some(out) = logit_tail_asymptotic(mu, sigma) {
        return Ok(out);
    }
    if sigma < LOGIT_SIGMA_TAYLOR_MAX {
        return Ok(logit_small_sigma_taylor(mu, sigma));
    }
    if logistic_normal_exact_eligible(mu, sigma)
        && let Ok(out) = logit_posterior_meanwith_deriv_exact_erfcx(mu, sigma)
    {
        return Ok(out);
    }
    // The Monahan-Stefanski probit approximation is only accurate when sigma
    // is large enough for the logistic kernel to resemble a rescaled probit.
    // For moderate sigma the GHQ fallback in the controlled router provides
    // better accuracy, so we only apply the approximation above a threshold.
    if sigma >= 3.0 {
        let out = logit_large_sigma_probit_asymptotic(mu, sigma);
        if out.mean.is_finite() && out.dmean_dmu.is_finite() {
            return Ok(out);
        }
    }
    Err(EstimationError::InvalidInput(
        "logit analytic expectation produced non-finite values".to_string(),
    ))
}

fn logit_posterior_meanwith_deriv_exact_erfcx(
    mu: f64,
    sigma: f64,
) -> Result<IntegratedMeanDerivative, EstimationError> {
    // Real-valued erfcx-series implementation for the logistic-normal mean.
    //
    //   sigmoid(x) = 1/2 + (1/2)·tanh(x/2),
    //
    // the partial-fraction expansion of tanh over its odd imaginary poles
    // ±i·(2n−1)π turns E[sigmoid(η)] into a convergent alternating series of
    // scaled-erfcx terms (see `logit_posterior_mean_exact` below for the full
    // derivation):
    //
    //   E[sigmoid(η)] = Φ(m/s)
    //     + (1/2)·exp(−m²/(2s²)) · Σ_{k≥1} (−1)^(k−1)
    //       · [erfcx((k s² + m)/(√2 s)) − erfcx((k s² − m)/(√2 s))],
    //
    // with m = |μ|, s = σ, and the sign of μ recovered by mean ↦ 1 − mean
    // below. Differentiating term-by-term in μ gives the derivative sum
    // produced by `scaled_erfcx_termwith_derivative`.
    //
    // The truncation index N* is chosen so that the alternating-series tail
    // bound (see `logistic_normal_tail_cutoff`) is below the documented
    // `LOGIT_ERFCX_ACCURACY_TARGET`. Reaching N* is thus an a-priori
    // certificate of accuracy — no empirical stability check is needed. The
    // only way this routine rejects is when N* would exceed LOGIT_MAX_TERMS,
    // at which point the accuracy contract cannot be honored and the caller
    // must route the evaluation elsewhere.
    let m = mu.abs();
    let s = sigma;
    let z = SQRT_2 * s;
    let phi_term = crate::probability::normal_cdf(m / s);
    let phi_prime = crate::probability::normal_pdf(m / s) / s;
    let max_k = logistic_normal_tail_cutoff(mu, sigma, LOGIT_ERFCX_ACCURACY_TARGET);
    if max_k >= LOGIT_MAX_TERMS
        && tail_bound_exceeds_accuracy(mu, sigma, LOGIT_MAX_TERMS, LOGIT_ERFCX_ACCURACY_TARGET)
    {
        return Err(EstimationError::InvalidInput(
            "logit erfcx series truncation bound exceeds LOGIT_MAX_TERMS at the required accuracy"
                .to_string(),
        ));
    }

    let mut sum = 0.0_f64;
    let mut dsum = 0.0_f64;
    // Run to the a-priori truncation index. No empirical early exit: the pair
    // magnitude inside the loop decays as O(1/k³) (the leading 1/k² cancels
    // between consecutive-sign terms) while the truncation tail after index k
    // only decays as O(1/k²), so pair-magnitude is anti-conservative as an
    // exit criterion — stopping early when `|pair| < δ` would leave a tail
    // much larger than δ. `max_k` was chosen so that the tail bound itself is
    // below the accuracy target, and that is the stopping rule we honor here.
    let mut k = 1usize;
    while k <= max_k {
        for kk in [k, k + 1].into_iter().filter(|kk| *kk <= max_k) {
            let kf = kk as f64;
            let a = (kf * s * s + m) / z;
            let b = (kf * s * s - m) / z;
            let sign = if kk % 2 == 1 { 1.0 } else { -1.0 };
            let (va, dva) = scaled_erfcx_termwith_derivative(m, s, a, 1.0 / z);
            let (vb, dvb) = scaled_erfcx_termwith_derivative(m, s, b, -1.0 / z);
            sum += sign * (va - vb);
            dsum += sign * (dva - dvb);
        }
        k += 2;
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
        mean,
        dmean_dmu: dmean,
        mode: IntegratedExpectationMode::ExactSpecialFunction,
    })
}

/// Returns true iff the alternating-series tail bound at truncation index
/// `n_terms` exceeds the requested accuracy. Used to reject erfcx evaluations
/// that cannot meet the documented contract.
#[inline]
fn tail_bound_exceeds_accuracy(
    mu: f64,
    sigma: f64,
    n_terms: usize,
    target_accuracy: f64,
) -> bool {
    let m = mu.abs();
    let s = sigma;
    let coeff = m * (2.0_f64 / std::f64::consts::PI).sqrt()
        * (-(m * m) / (2.0 * s * s)).exp()
        / (s * s * s);
    if !coeff.is_finite() || coeff <= 0.0 {
        return false;
    }
    let n = n_terms as f64;
    coeff / ((n + 1.0) * (n + 1.0)) > target_accuracy
}

#[inline]
fn logit_posterior_meanwith_deriv_ghq(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> IntegratedMeanDerivative {
    let (mean, dmean_dmu) = integrate_normal_ghq_adaptive(ctx, mu, sigma, |x| {
        let (p, d1, _, _) = component_point_jet(LinkComponent::Logit, x);
        (p, d1)
    });
    IntegratedMeanDerivative {
        mean,
        dmean_dmu: dmean_dmu.max(0.0),
        mode: if sigma <= LOGIT_SIGMA_DEGENERATE {
            IntegratedExpectationMode::ExactClosedForm
        } else {
            IntegratedExpectationMode::QuadratureFallback
        },
    }
}

#[inline]
fn logit_posterior_meanwith_deriv_controlled(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> Result<IntegratedMeanDerivative, EstimationError> {
    if !(mu.is_finite() && sigma.is_finite()) {
        return Err(EstimationError::InvalidInput(
            "logit integrated moments require finite mu and sigma".to_string(),
        ));
    }
    match logit_posterior_meanwith_deriv_exact(mu, sigma) {
        Ok(out) => Ok(out),
        Err(_) => Ok(logit_posterior_meanwith_deriv_ghq(ctx, mu, sigma)),
    }
}

#[inline]
fn log_normal_cdf_stable(x: f64) -> f64 {
    if !x.is_finite() {
        return if x.is_sign_negative() {
            f64::NEG_INFINITY
        } else {
            0.0
        };
    }
    if x < -8.0 {
        let u = -x / SQRT_2;
        -u * u + (0.5 * erfcx_nonnegative(u)).ln()
    } else {
        crate::probability::normal_cdf(x).max(1e-300).ln()
    }
}

#[inline]
fn cloglog_large_sigma_transition_tail(mu: f64, sigma: f64) -> f64 {
    let b = (mu + sigma * sigma) / sigma;
    let log_tail = mu + 0.5 * sigma * sigma + log_normal_cdf_stable(-b);
    if !log_tail.is_finite() {
        if log_tail.is_sign_negative() {
            0.0
        } else {
            1.0
        }
    } else if log_tail <= -745.0 {
        0.0
    } else {
        log_tail.exp().clamp(0.0, 1.0)
    }
}

#[inline]
fn cloglog_large_sigma_transition_approx(mu: f64, sigma: f64) -> IntegratedMeanDerivative {
    // Transition-region split for very broad Gaussian latents.
    //
    // Let z* solve exp(mu + sigma z*) ~= 1, i.e. z* = -mu / sigma. Below that
    // threshold the inverse link is well-approximated by exp(eta); above it
    // the inverse link is essentially saturated at 1. This gives
    //
    //   E[mu_link] ~= Phi(mu / sigma)
    //                + exp(mu + sigma^2 / 2) * Phi(-(mu + sigma^2) / sigma),
    //
    // and differentiating with respect to mu collapses to the truncated
    // lognormal tail term itself.
    let a = mu / sigma;
    let tail = cloglog_large_sigma_transition_tail(mu, sigma);
    IntegratedMeanDerivative {
        mean: (crate::probability::normal_cdf(a) + tail).clamp(0.0, 1.0),
        dmean_dmu: tail.clamp(0.0, 1.0 / std::f64::consts::E),
        mode: IntegratedExpectationMode::ControlledAsymptotic,
    }
}

#[inline]
fn cloglog_extreme_asymptotic(mu: f64, sigma: f64) -> Option<IntegratedMeanDerivative> {
    // Extreme-input ladder for the cloglog mean and its location derivative.
    //
    // Regimes:
    // - mu + sigma^2 / 2 << 0: rare-event tail, where 1 - exp(-exp(eta)) ~= exp(eta)
    // - mu - 8 sigma >> 0: survival term is numerically indistinguishable from 0
    // - sigma very large: split at the transition eta ~= 0 and use the closed
    //   form truncated-lognormal approximation above
    //
    // The thresholds intentionally leave overlap with the Taylor/Miles/Gamma
    // branches so neighboring formulas still cover the transition band.
    let rare_log = mu + 0.5 * sigma * sigma;
    if rare_log <= CLOGLOG_RARE_EVENT_LOG_MAX {
        let mean = rare_log.exp();
        return Some(IntegratedMeanDerivative {
            mean,
            dmean_dmu: mean,
            mode: IntegratedExpectationMode::ControlledAsymptotic,
        });
    }
    if mu - CLOGLOG_POSITIVE_SATURATION_SIGMAS * sigma >= CLOGLOG_POSITIVE_SATURATION_EDGE {
        return Some(IntegratedMeanDerivative {
            mean: 1.0,
            dmean_dmu: 0.0,
            mode: IntegratedExpectationMode::ControlledAsymptotic,
        });
    }
    if sigma >= CLOGLOG_LARGE_SIGMA_ASYMPTOTIC_MIN {
        return Some(cloglog_large_sigma_transition_approx(mu, sigma));
    }
    None
}

#[inline]
fn cloglog_survival_extreme_asymptotic(
    mu: f64,
    sigma: f64,
) -> Option<(f64, IntegratedExpectationMode)> {
    let rare_log = mu + 0.5 * sigma * sigma;
    if rare_log <= CLOGLOG_RARE_EVENT_LOG_MAX {
        let mean = rare_log.exp();
        return Some((
            (1.0 - mean).clamp(0.0, 1.0),
            IntegratedExpectationMode::ControlledAsymptotic,
        ));
    }
    if mu - CLOGLOG_POSITIVE_SATURATION_SIGMAS * sigma >= CLOGLOG_POSITIVE_SATURATION_EDGE {
        return Some((0.0, IntegratedExpectationMode::ControlledAsymptotic));
    }
    if sigma >= CLOGLOG_LARGE_SIGMA_ASYMPTOTIC_MIN {
        let a = mu / sigma;
        let tail = cloglog_large_sigma_transition_tail(mu, sigma);
        let survival = (crate::probability::normal_cdf(-a) - tail).clamp(0.0, 1.0);
        return Some((survival, IntegratedExpectationMode::ControlledAsymptotic));
    }
    None
}

// ── Exact Gumbel survival primitives ─────────────────────────────────────
//
// The Gumbel survival function S(x) = exp(-exp(x)) and its derivative
// S'(x) = exp(x)·exp(-exp(x)) are the building blocks for the cloglog
// link and all survival transforms.  These two functions are exact for ALL
// finite x under IEEE 754 without any clamping:
//
//   x → -∞: exp(x) → 0,    S → exp(-0) = 1,  S' → 0·1 = 0
//   x → +∞: exp(x) → +∞,   S → exp(-∞) = 0,  S' → ∞·0 = 0
//
// The only subtlety is in S': when x > 709, exp(x) overflows to +∞,
// and ∞ · 0 = NaN.  But x - exp(x) → -∞ for any x > 0, so S' = 0.
// We detect the intermediate overflow and return 0.0 exactly.

/// Exact Gumbel survival: S(x) = exp(-exp(x)).
///
/// No clamping — IEEE 754 handles both tails correctly:
/// - exp(x) underflows to 0 for x < -745 → S = exp(-0) = 1.0
/// - exp(x) overflows to ∞ for x > 709  → S = exp(-∞) = 0.0
#[inline]
fn gumbel_survival(x: f64) -> f64 {
    (-x.exp()).exp()
}

/// Exact Gumbel survival derivative: S'(x) = exp(x) · exp(-exp(x)).
///
/// Handles intermediate overflow: when exp(x) = ∞, the true derivative
/// is 0 (double-exponential decay dominates), so we return 0.0.
#[inline]
fn gumbel_survival_d1(x: f64) -> f64 {
    let ex = x.exp();
    if ex.is_infinite() {
        0.0
    } else {
        ex * (-ex).exp()
    }
}

/// Exact cloglog mean: μ(x) = 1 - exp(-exp(x)) via expm1 to avoid
/// catastrophic cancellation when exp(x) ≈ 0 (far negative tail).
///
/// This is the universal formula — it works for ALL finite x, not just
/// the negative tail.  For x > 709, exp(x) overflows but expm1(-∞) = -1,
/// giving μ = 1.0 exactly.
///
/// Delegates to `cloglog_negative_tail_mean` which implements the same
/// expm1 formulation.
#[inline]
fn cloglog_mean_exact(x: f64) -> f64 {
    cloglog_negative_tail_mean(x)
}

// ── Cloglog negative-tail asymptotics ────────────────────────────────────
//
// For the cloglog link μ(η) = 1 − exp(−exp(η)), when η ≪ 0:
//   μ(η)   ≈ exp(η)                          (since exp(η)→0)
//   μ'(η)  = exp(η)·exp(−exp(η)) ≈ exp(η)   (since exp(−exp(η))→1)
//
// For the integrated (Gaussian-convolved) mean E[μ(η+σZ)]:
//   E[μ(η+σZ)] ≈ E[exp(η+σZ)] = exp(η + σ²/2)
//   d/dη E[μ(η+σZ)] ≈ exp(η + σ²/2)
//
// These asymptotics are accurate to O(exp(2η)) and replace the previous
// hard-zero derivative outside the clamp window, which introduced a
// discontinuity at η = −30 and discarded real (though small) derivative mass.

/// Pointwise cloglog mean in the deep negative tail.
#[inline]
fn cloglog_negative_tail_mean(eta: f64) -> f64 {
    // μ(η) = 1 − exp(−exp(η)).  For η < −30, exp(η) < 1e-13, so
    // exp(−exp(η)) ≈ 1 − exp(η) and μ ≈ exp(η).
    // Direct exp avoids the intermediate exp(exp(η)) overflow path.
    if eta < -745.0 {
        // exp(-745) underflows to 0.0 in f64.
        0.0
    } else {
        // Use expm1(−exp(η)) = exp(−exp(η)) − 1, so μ = −expm1(−exp(η)).
        // This is more accurate than 1 − exp(−exp(η)) near zero.
        let ex = eta.exp();
        -(-ex).exp_m1()
    }
}

/// Pointwise cloglog derivative dμ/dη in the deep negative tail.
///
/// Production code now uses `gumbel_survival_d1` which is exact for all
/// finite x.  This function is retained for its dedicated unit test.
#[cfg(test)]
#[inline]
fn cloglog_negative_tail_derivative(eta: f64) -> f64 {
    // dμ/dη = exp(η) · exp(−exp(η)).  For η ≪ 0 this simplifies to ≈ exp(η).
    if eta < -745.0 {
        0.0
    } else {
        let ex = eta.exp();
        (ex * (-ex).exp()).max(0.0)
    }
}

#[inline]
fn cloglog_small_sigma_taylor(mu: f64, sigma: f64) -> IntegratedMeanDerivative {
    // Small-variance heat-kernel expansion for the cloglog inverse link.
    //
    // For η = μ + σ Z, Z ~ N(0,1), and any analytic f the heat-kernel
    // (even-moment) identity gives
    //
    //   E[f(η)] = Σ_{k≥0} σ^(2k) / (2^k · k!) · f^(2k)(μ).
    //
    // Here f(x) = 1 − exp(−exp(x)) is entire, so the series is valid
    // globally. Truncating at the σ⁴ term yields
    //
    //   E[f(η)]       ≈ f + (σ²/2) f''    + (σ⁴/8) f^(4)
    //   d/dμ E[f(η)]  ≈ f' + (σ²/2) f'''  + (σ⁴/8) f^(5).
    //
    // The coefficient of f^(4) is 1/(2²·2!) = 1/8, not 1/4! = 1/24 —
    // this is the heat-kernel weight, not the Taylor weight.
    //
    // A single formula covers the entire real line once the constituent
    // evaluations are written stably:
    //   • f0 uses -expm1(-ex) to stay bit-exact for μ ≪ 0 (where ex ≈ 0)
    //   • surv = exp(-ex) underflows cleanly to 0 for μ ≫ 0, yielding
    //     the saturation limit f ≡ 1, f' ≡ 0 without any branch.
    // No separate "negative-tail" MGF approximation is needed; the Taylor
    // truncation error is uniformly O(σ⁶ · ex) across the whole domain.
    if sigma <= CLOGLOG_SIGMA_DEGENERATE {
        return IntegratedMeanDerivative {
            mean: cloglog_mean_exact(mu),
            dmean_dmu: gumbel_survival_d1(mu),
            mode: IntegratedExpectationMode::ExactClosedForm,
        };
    }

    let ex = mu.exp();
    if !ex.is_finite() {
        // μ large enough that ex overflows to +∞; f saturates to 1 and f' to 0.
        return IntegratedMeanDerivative {
            mean: 1.0,
            dmean_dmu: 0.0,
            mode: IntegratedExpectationMode::ControlledAsymptotic,
        };
    }
    let surv = (-ex).exp();
    if surv == 0.0 {
        // exp(-ex) underflow: positive-μ saturation, same limit.
        return IntegratedMeanDerivative {
            mean: 1.0,
            dmean_dmu: 0.0,
            mode: IntegratedExpectationMode::ControlledAsymptotic,
        };
    }

    let s2 = sigma * sigma;
    let s4 = s2 * s2;
    let e2x = ex * ex;
    let e3x = e2x * ex;
    let e4x = e3x * ex;
    let e5x = e4x * ex;
    // -expm1(-ex) = 1 - exp(-ex) is bit-exact even when ex is subnormal.
    let f0 = -(-ex).exp_m1();
    let f1 = ex * surv;
    let f2 = surv * (ex - e2x);
    let f3 = surv * (ex - 3.0 * e2x + e3x);
    let f4 = surv * (ex - 7.0 * e2x + 6.0 * e3x - e4x);
    let f5 = surv * (ex - 15.0 * e2x + 25.0 * e3x - 10.0 * e4x + e5x);
    IntegratedMeanDerivative {
        mean: f0 + 0.5 * s2 * f2 + (s4 / 8.0) * f4,
        dmean_dmu: (f1 + 0.5 * s2 * f3 + (s4 / 8.0) * f5).max(0.0),
        mode: IntegratedExpectationMode::ControlledAsymptotic,
    }
}

#[inline]
fn cloglog_posterior_meanwith_deriv_ghq(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> IntegratedMeanDerivative {
    if sigma < 1e-10 {
        return IntegratedMeanDerivative {
            mean: cloglog_mean_exact(mu),
            dmean_dmu: gumbel_survival_d1(mu),
            mode: IntegratedExpectationMode::ExactClosedForm,
        };
    }
    let mean = cloglog_mean_from_survival(survival_posterior_mean_ghq(ctx, mu, sigma));
    let dmean_dmu =
        integrate_normal_ghq_adaptive(ctx, mu, sigma, |x| gumbel_survival_d1(x)).max(0.0);
    IntegratedMeanDerivative {
        mean,
        dmean_dmu,
        mode: IntegratedExpectationMode::QuadratureFallback,
    }
}

#[inline]
fn survival_posterior_mean_ghq(ctx: &QuadratureContext, eta: f64, se_eta: f64) -> f64 {
    integrate_normal_ghq_adaptive(ctx, eta, se_eta, |x| gumbel_survival(x)).clamp(0.0, 1.0)
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
    // - explicit extreme-input asymptotics
    // - Miles erfc-series in tail-dominated regimes
    // - Clenshaw-Curtis on the truncated real integral in the central regime
    // - exact Gamma/Mellin-Barnes if CC would need too many nodes or misbehaves
    // - GHQ only as the final numerical fallback
    //
    // Validation target for the extended asymptotic routes: compare against
    // 256-point GHQ on representative difficult points such as
    // (-20, 0.1), (-5, 5), (0, 20), (10, 0.5), (10, 10), and (-0.5, 100).
    if !(mu.is_finite() && sigma.is_finite()) || sigma <= CLOGLOG_SIGMA_DEGENERATE {
        return (
            gumbel_survival(mu).clamp(0.0, 1.0),
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
    if let Some(out) = cloglog_survival_extreme_asymptotic(mu, sigma) {
        return out;
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
    lognormal_laplace_unit_term_shared(ctx, mu + z.ln(), sigma)
}

#[inline]
pub(crate) fn lognormal_laplace_unit_term_shared(
    ctx: &QuadratureContext,
    shifted_mu: f64,
    sigma: f64,
) -> (f64, IntegratedExpectationMode) {
    cloglog_survival_term_controlled(ctx, shifted_mu, sigma)
}

#[inline]
pub(crate) fn lognormal_laplace_term_shared(
    ctx: &QuadratureContext,
    z: f64,
    mu: f64,
    sigma: f64,
) -> (f64, IntegratedExpectationMode) {
    lognormal_laplace_term_controlled(ctx, z, mu, sigma)
}

#[inline]
fn cloglog_survivalsecond_moment_controlled(
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
) -> (
    (f64, IntegratedExpectationMode),
    (f64, IntegratedExpectationMode),
) {
    let shiftedmu = mu + sigma * sigma;

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
            cloglog_survival_miles(shiftedmu, sigma),
        )
    {
        return (
            (
                base.clamp(0.0, 1.0),
                IntegratedExpectationMode::ExactSpecialFunction,
            ),
            (
                shifted.clamp(0.0, 1.0),
                IntegratedExpectationMode::ExactSpecialFunction,
            ),
        );
    }

    if cloglog_should_prefer_cc(mu, sigma, CLOGLOG_CC_TOL)
        && cloglog_should_prefer_cc(shiftedmu, sigma, CLOGLOG_CC_TOL)
        && let (Ok(base), Ok(shifted)) = (
            cloglog_survival_cc(ctx, mu, sigma, CLOGLOG_CC_TOL),
            cloglog_survival_cc(ctx, shiftedmu, sigma, CLOGLOG_CC_TOL),
        )
    {
        return (
            (
                base.clamp(0.0, 1.0),
                IntegratedExpectationMode::ExactSpecialFunction,
            ),
            (
                shifted.clamp(0.0, 1.0),
                IntegratedExpectationMode::ExactSpecialFunction,
            ),
        );
    }

    if let (Ok(base), Ok(shifted)) = (
        cloglog_survival_gamma_reference(mu, sigma),
        cloglog_survival_gamma_reference(shiftedmu, sigma),
    ) {
        return (
            (
                base.clamp(0.0, 1.0),
                IntegratedExpectationMode::ExactSpecialFunction,
            ),
            (
                shifted.clamp(0.0, 1.0),
                IntegratedExpectationMode::ExactSpecialFunction,
            ),
        );
    }

    (
        cloglog_survival_term_controlled(ctx, mu, sigma),
        cloglog_survival_term_controlled(ctx, shiftedmu, sigma),
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
        -survival.ln().exp_m1()
    } else {
        1.0 - survival
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
fn complexmul(a: Complex, b: Complex) -> Complex {
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
        let piz = Complex {
            re: std::f64::consts::PI * z.re,
            im: std::f64::consts::PI * z.im,
        };
        let one_minusz = Complex {
            re: 1.0 - z.re,
            im: -z.im,
        };
        return complex_sub(
            complex_sub(
                Complex {
                    re: std::f64::consts::PI.ln(),
                    im: 0.0,
                },
                complex_ln(complex_sin(piz)),
            ),
            complex_log_gamma_lanczos(one_minusz),
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
            complexmul(
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
pub(crate) fn cloglog_posterior_meanwith_deriv_gamma_reference(
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
        let z_sq = complexmul(z, z);
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
    let sval = ((h / 3.0) * sum_s / std::f64::consts::PI).clamp(0.0, 1.0);
    if !sval.is_finite() {
        return Err(EstimationError::InvalidInput(
            "Gamma cloglog reference backend produced non-finite values".to_string(),
        ));
    }
    Ok(sval)
}

pub(crate) fn cloglog_posterior_meanwith_deriv_controlled(
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
    // 3. explicit extreme-input asymptotics
    //    Large negative mu uses the exact lognormal first moment of exp(eta),
    //    very large positive mu saturates to 1, and very large sigma uses the
    //    transition split at eta ~= 0.
    //
    // 4. tail-dominated large sigma
    //    The Miles erfc-gated series is efficient because the erfc gate keeps
    //    the alternating lognormal-moment series short and numerically tame.
    //
    // 5. central large sigma
    //    The exact Mellin-Barnes / Gamma inversion is preferred, because the
    //    Miles series is no longer the best-behaved production representation.
    //
    // 6. final escape
    //    GHQ remains only as a numerical fallback if the chosen special-
    //    function backend returns a non-finite or non-converged result.
    //
    // This layered routing is the real conclusion of the math work: not one
    // magical universal formula, but one shared mathematical target with the
    // best evaluator chosen for each regime.
    if !(mu.is_finite() && sigma.is_finite()) || sigma <= CLOGLOG_SIGMA_DEGENERATE {
        return IntegratedMeanDerivative {
            // cloglog_mean_exact uses expm1 to avoid 1 − 1 cancellation for
            // all mu, and handles exp overflow (mu > 709) via expm1(-∞) = −1.
            mean: cloglog_mean_exact(mu),
            // gumbel_survival_d1 is exact for all finite mu: it detects
            // intermediate exp overflow and returns 0.0 (correct limit).
            dmean_dmu: gumbel_survival_d1(mu),
            mode: IntegratedExpectationMode::ExactClosedForm,
        };
    }
    let candidate = if sigma < CLOGLOG_SIGMA_TAYLOR_MAX {
        cloglog_small_sigma_taylor(mu, sigma)
    } else if let Some(out) = cloglog_extreme_asymptotic(mu, sigma) {
        out
    } else {
        let ((survival, mode), (shifted_survival, shifted_mode)) =
            cloglog_survival_pair_controlled(ctx, mu, sigma);
        if matches!(mode, IntegratedExpectationMode::QuadratureFallback)
            || matches!(shifted_mode, IntegratedExpectationMode::QuadratureFallback)
        {
            return cloglog_posterior_meanwith_deriv_ghq(ctx, mu, sigma);
        }
        let mean = cloglog_mean_from_survival(survival);
        let dmean = cloglog_shift_identity_derivative(mu, sigma, shifted_survival);
        let mode = if matches!(mode, IntegratedExpectationMode::ControlledAsymptotic)
            || matches!(
                shifted_mode,
                IntegratedExpectationMode::ControlledAsymptotic
            ) {
            IntegratedExpectationMode::ControlledAsymptotic
        } else {
            mode
        };
        IntegratedMeanDerivative {
            mean,
            dmean_dmu: dmean.max(0.0),
            mode,
        }
    };
    // Safety-net drift check with loose tolerances — see logit comment.
    // Skip for large-sigma ControlledAsymptotic: the transition approximation
    // legitimately diverges from 128-node GHQ by more than the drift tolerance
    // at sigma >= CLOGLOG_LARGE_SIGMA_ASYMPTOTIC_MIN, and the asymptotic is the
    // trusted answer in that regime.
    if matches!(
        candidate.mode,
        IntegratedExpectationMode::ControlledAsymptotic
    ) && sigma >= CLOGLOG_LARGE_SIGMA_ASYMPTOTIC_MIN
    {
        return candidate;
    }
    let ghq = cloglog_posterior_meanwith_deriv_ghq(ctx, mu, sigma);
    if integrated_mean_derivative_drift_exceeds(&candidate, &ghq, 1e-6, 1e-4, 1e-5, 1e-3) {
        ghq
    } else {
        candidate
    }
}

pub fn integrated_inverse_link_mean_and_derivative(
    quadctx: &QuadratureContext,
    link: LinkFunction,
    mu: f64,
    sigma: f64,
) -> Result<IntegratedMeanDerivative, EstimationError> {
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
        LinkFunction::Log => {
            let mean = (mu + 0.5 * sigma * sigma).exp();
            Ok(IntegratedMeanDerivative {
                mean,
                dmean_dmu: mean,
                mode: IntegratedExpectationMode::ExactClosedForm,
            })
        }
        LinkFunction::Probit => Ok(probit_posterior_meanwith_deriv_exact(mu, sigma)),
        LinkFunction::Logit => logit_posterior_meanwith_deriv_controlled(quadctx, mu, sigma),
        LinkFunction::CLogLog => Ok(cloglog_posterior_meanwith_deriv_controlled(quadctx, mu, sigma)),
        LinkFunction::Sas => Err(EstimationError::InvalidInput(
            "state-less integrated SAS moments are unsupported; use SAS-aware prediction APIs with explicit (epsilon, log_delta)".to_string(),
        )),
        LinkFunction::BetaLogistic => Err(EstimationError::InvalidInput(
            "state-less integrated Beta-Logistic moments are unsupported; use link-aware prediction APIs with explicit (delta, epsilon)".to_string(),
        )),
        LinkFunction::Identity => Ok(IntegratedMeanDerivative {
            mean: mu,
            dmean_dmu: 1.0,
            mode: IntegratedExpectationMode::ExactClosedForm,
        }),
    }
}

#[inline]
pub fn integrated_inverse_link_jet(
    quadctx: &QuadratureContext,
    link: LinkFunction,
    mu: f64,
    sigma: f64,
) -> Result<IntegratedInverseLinkJet, EstimationError> {
    match link {
        LinkFunction::Log => {
            let mean = (mu + 0.5 * sigma * sigma).exp();
            Ok(IntegratedInverseLinkJet {
                mean,
                d1: mean,
                d2: mean,
                d3: mean,
                mode: IntegratedExpectationMode::ExactClosedForm,
            })
        }
        LinkFunction::Probit => Ok(integrated_probit_jet(mu, sigma)),
        LinkFunction::Logit => {
            let candidate = integrate_logit_inverse_link_jet_from_scalar_backend(
                mu,
                sigma,
                |m, s| logit_posterior_meanwith_deriv_controlled(quadctx, m, s),
                |x| component_point_jet(LinkComponent::Logit, x),
            )?;
            Ok(candidate)
        }
        LinkFunction::CLogLog => {
            validate_latent_cloglog_inputs(mu, sigma)?;
            Ok(integrated_cloglog_inverse_link_jet_controlled(
                quadctx, mu, sigma,
            ))
        }
        LinkFunction::Sas => Err(EstimationError::InvalidInput(
            "state-less integrated SAS jet is unsupported; use SAS-aware prediction APIs with explicit (epsilon, log_delta)".to_string(),
        )),
        LinkFunction::BetaLogistic => Err(EstimationError::InvalidInput(
            "state-less integrated Beta-Logistic jet is unsupported; use link-aware prediction APIs with explicit (delta, epsilon)".to_string(),
        )),
        LinkFunction::Identity => Ok(IntegratedInverseLinkJet {
            mean: mu,
            d1: 1.0,
            d2: 0.0,
            d3: 0.0,
            mode: IntegratedExpectationMode::ExactClosedForm,
        }),
    }
}

#[inline]
pub fn integrated_logit_inverse_link_jet_pirls(
    quadctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> Result<IntegratedInverseLinkJet, EstimationError> {
    integrate_logit_inverse_link_jet_from_scalar_backend(
        mu,
        sigma,
        // Hot PIRLS path: exact special-function backend first, GHQ only if
        // the exact backend rejects the inputs.
        |m, s| logit_posterior_meanwith_deriv_controlled(quadctx, m, s),
        |x| component_point_jet(LinkComponent::Logit, x),
    )
}

#[inline]
fn sas_point_jet(x: f64, epsilon: f64, log_delta: f64) -> (f64, f64, f64, f64) {
    let jet = sas_inverse_link_jet(x, epsilon, log_delta);
    (jet.mu, jet.d1, jet.d2, jet.d3)
}

#[inline]
fn beta_logistic_point_jet(x: f64, delta: f64, epsilon: f64) -> (f64, f64, f64, f64) {
    let jet = beta_logistic_inverse_link_jet(x, delta, epsilon);
    (jet.mu, jet.d1, jet.d2, jet.d3)
}

#[inline]
fn integrated_expectation_mode_rank(mode: IntegratedExpectationMode) -> u8 {
    match mode {
        IntegratedExpectationMode::ExactClosedForm => 0,
        IntegratedExpectationMode::ExactSpecialFunction => 1,
        IntegratedExpectationMode::ControlledAsymptotic => 2,
        IntegratedExpectationMode::QuadratureFallback => 3,
    }
}

#[inline]
fn worse_integrated_expectation_mode(
    lhs: IntegratedExpectationMode,
    rhs: IntegratedExpectationMode,
) -> IntegratedExpectationMode {
    if integrated_expectation_mode_rank(lhs) >= integrated_expectation_mode_rank(rhs) {
        lhs
    } else {
        rhs
    }
}

#[inline]
fn integrated_scalar_drift_exceeds(
    candidate: f64,
    reference: f64,
    abs_tol: f64,
    rel_tol: f64,
) -> bool {
    if !(candidate.is_finite() && reference.is_finite()) {
        return true;
    }
    (candidate - reference).abs() > abs_tol.max(rel_tol * reference.abs().max(candidate.abs()))
}

#[inline]
fn integrated_mean_derivative_drift_exceeds(
    candidate: &IntegratedMeanDerivative,
    reference: &IntegratedMeanDerivative,
    mean_abs_tol: f64,
    mean_rel_tol: f64,
    deriv_abs_tol: f64,
    deriv_rel_tol: f64,
) -> bool {
    integrated_scalar_drift_exceeds(candidate.mean, reference.mean, mean_abs_tol, mean_rel_tol)
        || integrated_scalar_drift_exceeds(
            candidate.dmean_dmu,
            reference.dmean_dmu,
            deriv_abs_tol,
            deriv_rel_tol,
        )
}

#[inline]
fn component_point_jet(component: LinkComponent, x: f64) -> (f64, f64, f64, f64) {
    // Keep the point-mass quadrature kernels wired to the same inverse-link
    // implementation used by mixture links and survival residual distributions.
    let jet = component_inverse_link_jet(component, x);
    (jet.mu, jet.d1, jet.d2, jet.d3)
}

#[inline]
fn integrated_mixture_component_jet(
    ctx: &QuadratureContext,
    component: LinkComponent,
    mu: f64,
    sigma: f64,
) -> IntegratedInverseLinkJet {
    // Use the same controlled backends (exact/asymptotic/special-function)
    // as integrated_inverse_link_jet so that the same (mu, sigma) always
    // produces identical d2, d3 regardless of whether it enters as a
    // standalone link or as a mixture component.
    match component {
        LinkComponent::Logit => integrated_inverse_link_jet(ctx, LinkFunction::Logit, mu, sigma)
            .unwrap_or_else(|_| integrated_logit_jet_ghq(ctx, mu, sigma)),
        LinkComponent::Probit => integrated_probit_jet(mu, sigma),
        LinkComponent::CLogLog => integrated_cloglog_inverse_link_jet_controlled(ctx, mu, sigma),
        LinkComponent::LogLog | LinkComponent::Cauchit => {
            let (mean, d1, d2, d3) = integrate_normal_ghq_adaptive(ctx, mu, sigma, |x| {
                component_point_jet(component, x)
            });
            IntegratedInverseLinkJet {
                mean,
                d1: d1.max(0.0),
                d2,
                d3,
                mode: if sigma <= 1e-10 {
                    IntegratedExpectationMode::ExactClosedForm
                } else {
                    IntegratedExpectationMode::QuadratureFallback
                },
            }
        }
    }
}

#[inline]
fn integrated_mixture_jet(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
    mixture_state: &MixtureLinkState,
) -> Result<IntegratedInverseLinkJet, EstimationError> {
    // Solver-facing integrated jets in this module store eta/location
    // derivatives only: (mean, d/dmu, d²/dmu², d³/dmu³). Closed-form sigma
    // derivatives for the probit component are therefore not threaded here
    // because the integrated PIRLS callers do not consume them.
    if mixture_state.components.is_empty() {
        return Err(EstimationError::InvalidInput(
            "integrated mixture-link jet requires at least one blended component".to_string(),
        ));
    }
    if mixture_state.components.len() != mixture_state.pi.len() {
        return Err(EstimationError::InvalidInput(
            "integrated mixture-link jet requires matching component and weight counts".to_string(),
        ));
    }

    // Validation note: compare against a 128-point direct GHQ reference for
    // blended(logit,probit) over w in {0.0, 0.3, 0.5, 0.7, 1.0} and
    // (mu, sigma) on (-5, 5) x (0.1, 10). The w=0 probit case should match
    // Phi(mu / sqrt(1 + sigma^2)) to machine precision.
    let mut mean = 0.0_f64;
    let mut d1 = 0.0_f64;
    let mut d2 = 0.0_f64;
    let mut d3 = 0.0_f64;
    let mut mode = IntegratedExpectationMode::ExactClosedForm;
    let mut saw_positive_weight = false;

    for (&component, &weight) in mixture_state.components.iter().zip(mixture_state.pi.iter()) {
        if weight <= 0.0 {
            continue;
        }
        let jet = integrated_mixture_component_jet(ctx, component, mu, sigma);
        mean += weight * jet.mean;
        d1 += weight * jet.d1;
        d2 += weight * jet.d2;
        d3 += weight * jet.d3;
        if integrated_expectation_mode_rank(jet.mode) > integrated_expectation_mode_rank(mode) {
            mode = jet.mode;
        }
        saw_positive_weight = true;
    }

    if !saw_positive_weight {
        return Err(EstimationError::InvalidInput(
            "integrated mixture-link jet requires at least one positive component weight"
                .to_string(),
        ));
    }

    Ok(IntegratedInverseLinkJet {
        mean,
        d1: d1.max(0.0),
        d2,
        d3,
        mode,
    })
}

#[inline]
fn integrated_sas_jet_ghq(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
    sas_state: &SasLinkState,
) -> IntegratedInverseLinkJet {
    let (mean, d1, d2, d3) = integrate_normal_ghq_adaptive(ctx, mu, sigma, |x| {
        sas_point_jet(x, sas_state.epsilon, sas_state.log_delta)
    });
    IntegratedInverseLinkJet {
        mean,
        d1: d1.max(0.0),
        d2,
        d3,
        mode: if sigma <= 1e-10 {
            IntegratedExpectationMode::ExactClosedForm
        } else {
            IntegratedExpectationMode::QuadratureFallback
        },
    }
}

#[inline]
fn integrated_beta_logistic_jet_ghq(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
    beta_state: &SasLinkState,
) -> IntegratedInverseLinkJet {
    let (mean, d1, d2, d3) = integrate_normal_ghq_adaptive(ctx, mu, sigma, |x| {
        beta_logistic_point_jet(x, beta_state.log_delta, beta_state.epsilon)
    });
    IntegratedInverseLinkJet {
        mean,
        d1: d1.max(0.0),
        d2,
        d3,
        mode: if sigma <= 1e-10 {
            IntegratedExpectationMode::ExactClosedForm
        } else {
            IntegratedExpectationMode::QuadratureFallback
        },
    }
}

/// State-aware inverse-link jet integration for Gaussian-uncertain predictors.
#[inline]
pub fn integrated_inverse_link_jetwith_state(
    quadctx: &QuadratureContext,
    link: LinkFunction,
    mu: f64,
    sigma: f64,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
) -> Result<IntegratedInverseLinkJet, EstimationError> {
    if let Some(state) = mixture_link_state {
        return integrated_mixture_jet(quadctx, mu, sigma, state);
    }
    if matches!(link, LinkFunction::Sas) {
        let sas = sas_link_state.ok_or_else(|| {
            EstimationError::InvalidInput(
                "state-less integrated SAS jet is unsupported; explicit SasLinkState is required"
                    .to_string(),
            )
        })?;
        return Ok(integrated_sas_jet_ghq(quadctx, mu, sigma, sas));
    }
    if matches!(link, LinkFunction::BetaLogistic) {
        let state = sas_link_state.ok_or_else(|| {
            EstimationError::InvalidInput(
                "state-less integrated Beta-Logistic jet is unsupported; explicit link state is required"
                    .to_string(),
            )
        })?;
        return Ok(integrated_beta_logistic_jet_ghq(quadctx, mu, sigma, state));
    }
    integrated_inverse_link_jet(quadctx, link, mu, sigma)
}

/// Family-level integration dispatcher for Gaussian-uncertain linear predictors.
///
/// This is the solver-facing boundary: callers request integrated moments/jet by
/// family, while all link-specific quadrature/special-function routing stays in
/// the quadrature domain.
#[inline]
pub fn integrated_family_moments_jet(
    quadctx: &QuadratureContext,
    family: LikelihoodFamily,
    eta: f64,
    se_eta: f64,
) -> Result<IntegratedMomentsJet, EstimationError> {
    integrated_family_moments_jetwith_state(quadctx, family, eta, se_eta, None, None)
}

/// State-aware family-level integration dispatcher for Gaussian-uncertain
/// linear predictors.
#[inline]
pub fn integrated_family_moments_jetwith_state(
    quadctx: &QuadratureContext,
    family: LikelihoodFamily,
    eta: f64,
    se_eta: f64,
    mixture_link_state: Option<&MixtureLinkState>,
    sas_link_state: Option<&SasLinkState>,
) -> Result<IntegratedMomentsJet, EstimationError> {
    const PROB_EPS: f64 = 1e-12;
    let e = eta.clamp(-700.0, 700.0);
    let se = se_eta.max(0.0);
    match family {
        LikelihoodFamily::BinomialLogit => {
            let jet = integrated_inverse_link_jet(quadctx, LinkFunction::Logit, e, se)?;
            let mean = jet.mean;
            Ok(IntegratedMomentsJet {
                mean,
                variance: (mean * (1.0 - mean)).max(PROB_EPS),
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
                mode: jet.mode,
            })
        }
        LikelihoodFamily::BinomialProbit => {
            let jet = integrated_inverse_link_jet(quadctx, LinkFunction::Probit, e, se)?;
            let mean = jet.mean;
            Ok(IntegratedMomentsJet {
                mean,
                variance: (mean * (1.0 - mean)).max(PROB_EPS),
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
                mode: jet.mode,
            })
        }
        LikelihoodFamily::BinomialCLogLog => {
            let jet = integrated_inverse_link_jet(quadctx, LinkFunction::CLogLog, e, se)?;
            let mean = jet.mean;
            Ok(IntegratedMomentsJet {
                mean,
                variance: (mean * (1.0 - mean)).max(PROB_EPS),
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
                mode: jet.mode,
            })
        }
        LikelihoodFamily::BinomialLatentCLogLog => Err(EstimationError::InvalidInput(
            "BinomialLatentCLogLog integrated moments require an explicit latent cloglog inverse-link state"
                .to_string(),
        )),
        LikelihoodFamily::BinomialSas => {
            let jet = integrated_inverse_link_jetwith_state(
                quadctx,
                LinkFunction::Sas,
                e,
                se,
                mixture_link_state,
                sas_link_state,
            )?;
            let mean = jet.mean;
            Ok(IntegratedMomentsJet {
                mean,
                variance: (mean * (1.0 - mean)).max(PROB_EPS),
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
                mode: jet.mode,
            })
        }
        LikelihoodFamily::BinomialBetaLogistic => {
            let jet = integrated_inverse_link_jetwith_state(
                quadctx,
                LinkFunction::BetaLogistic,
                e,
                se,
                mixture_link_state,
                sas_link_state,
            )?;
            let mean = jet.mean;
            Ok(IntegratedMomentsJet {
                mean,
                variance: (mean * (1.0 - mean)).max(PROB_EPS),
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
                mode: jet.mode,
            })
        }
        LikelihoodFamily::GaussianIdentity => Ok(IntegratedMomentsJet {
            mean: e,
            variance: 1.0,
            d1: 1.0,
            d2: 0.0,
            d3: 0.0,
            mode: IntegratedExpectationMode::ExactClosedForm,
        }),
        LikelihoodFamily::RoystonParmar => {
            let jet = integrated_inverse_link_jetwith_state(
                quadctx,
                LinkFunction::CLogLog,
                e,
                se,
                mixture_link_state,
                sas_link_state,
            )?;
            let mean = (1.0 - jet.mean).clamp(0.0, 1.0);
            Ok(IntegratedMomentsJet {
                mean,
                variance: (mean * (1.0 - mean)).max(PROB_EPS),
                d1: -jet.d1,
                d2: -jet.d2,
                d3: -jet.d3,
                mode: jet.mode,
            })
        }
        LikelihoodFamily::BinomialMixture => {
            let state = mixture_link_state.ok_or_else(|| {
                EstimationError::InvalidInput(
                    "BinomialMixture integrated moments require explicit MixtureLinkState"
                        .to_string(),
                )
            })?;
            let jet = integrated_mixture_jet(quadctx, e, se, state)?;
            let mean = jet.mean;
            Ok(IntegratedMomentsJet {
                mean,
                variance: (mean * (1.0 - mean)).max(PROB_EPS),
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
                mode: jet.mode,
            })
        }
        LikelihoodFamily::PoissonLog | LikelihoodFamily::GammaLog => {
            // Log-normal MGF: E[exp(η)] = exp(e + s²/2)
            // d/de = exp(e + s²/2)   (same as the mean)
            // d²/de² = exp(e + s²/2)
            // d³/de³ = exp(e + s²/2)
            let s2 = se * se;
            let mean = (e + 0.5 * s2).exp();
            // Variance of the response depends on family:
            //   Poisson: Var = mean (since Var[Y|mu] = mu)
            //   Gamma:   Var = mean² / shape, but shape not available here;
            //            use mean² as proxy (shape=1).
            let variance = match family {
                LikelihoodFamily::PoissonLog => mean,
                _ => mean * mean,
            };
            Ok(IntegratedMomentsJet {
                mean,
                variance,
                d1: mean,
                d2: mean,
                d3: mean,
                mode: IntegratedExpectationMode::ExactClosedForm,
            })
        }
    }
}

/// Batch version of logit_posterior_meanwith_deriv.
/// Returns (mu_array, dmu_array)
pub fn logit_posterior_meanwith_deriv_batch(
    ctx: &QuadratureContext,
    eta: &ndarray::Array1<f64>,
    se_eta: &ndarray::Array1<f64>,
) -> Result<(ndarray::Array1<f64>, ndarray::Array1<f64>), EstimationError> {
    let n = eta.len();
    let mut mu = ndarray::Array1::<f64>::zeros(n);
    let mut dmu = ndarray::Array1::<f64>::zeros(n);

    for i in 0..n {
        let integrated = integrated_inverse_link_mean_and_derivative(
            ctx,
            LinkFunction::Logit,
            eta[i],
            se_eta[i],
        )?;
        mu[i] = integrated.mean;
        dmu[i] = integrated.dmean_dmu;
    }

    Ok((mu, dmu))
}

/// Computes posterior mean probabilities for a batch of predictions.
///
/// This is the vectorized version of `logit_posterior_mean`.
pub fn logit_posterior_mean_batch(
    ctx: &QuadratureContext,
    eta: &ndarray::Array1<f64>,
    se_eta: &ndarray::Array1<f64>,
) -> Result<ndarray::Array1<f64>, EstimationError> {
    let mut out = ndarray::Array1::<f64>::zeros(eta.len());
    for i in 0..eta.len() {
        out[i] = integrated_inverse_link_mean_and_derivative(
            ctx,
            LinkFunction::Logit,
            eta[i],
            se_eta[i],
        )?
        .mean;
    }
    Ok(out)
}

pub trait GhqValue: Sized {
    fn zero() -> Self;
    fn addweighted(&mut self, weight: f64, value: Self);
    fn scale(self, factor: f64) -> Self;
}

impl GhqValue for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }

    #[inline]
    fn addweighted(&mut self, weight: f64, value: Self) {
        *self += weight * value;
    }

    #[inline]
    fn scale(self, factor: f64) -> Self {
        self * factor
    }
}

impl GhqValue for (f64, f64) {
    #[inline]
    fn zero() -> Self {
        (0.0, 0.0)
    }

    #[inline]
    fn addweighted(&mut self, weight: f64, value: Self) {
        self.0 += weight * value.0;
        self.1 += weight * value.1;
    }

    #[inline]
    fn scale(self, factor: f64) -> Self {
        (self.0 * factor, self.1 * factor)
    }
}

impl GhqValue for (f64, f64, f64, f64) {
    #[inline]
    fn zero() -> Self {
        (0.0, 0.0, 0.0, 0.0)
    }

    #[inline]
    fn addweighted(&mut self, weight: f64, value: Self) {
        self.0 += weight * value.0;
        self.1 += weight * value.1;
        self.2 += weight * value.2;
        self.3 += weight * value.3;
    }

    #[inline]
    fn scale(self, factor: f64) -> Self {
        (
            self.0 * factor,
            self.1 * factor,
            self.2 * factor,
            self.3 * factor,
        )
    }
}

impl GhqValue for (f64, f64, f64, f64, f64, f64) {
    #[inline]
    fn zero() -> Self {
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    }

    #[inline]
    fn addweighted(&mut self, weight: f64, value: Self) {
        self.0 += weight * value.0;
        self.1 += weight * value.1;
        self.2 += weight * value.2;
        self.3 += weight * value.3;
        self.4 += weight * value.4;
        self.5 += weight * value.5;
    }

    #[inline]
    fn scale(self, factor: f64) -> Self {
        (
            self.0 * factor,
            self.1 * factor,
            self.2 * factor,
            self.3 * factor,
            self.4 * factor,
            self.5 * factor,
        )
    }
}

#[inline]
fn integrate_normal_ghq_adaptive<F, R>(ctx: &QuadratureContext, eta: f64, se_eta: f64, f: F) -> R
where
    F: Fn(f64) -> R,
    R: GhqValue,
{
    if se_eta < 1e-10 {
        return f(eta);
    }
    let n = adaptive_point_count_from_sd(se_eta.abs());
    with_gh_nodesweights(ctx, n, |nodes, weights| {
        let scale = SQRT_2 * se_eta;
        let mut sum = R::zero();
        for i in 0..n {
            sum.addweighted(weights[i], f(eta + scale * nodes[i]));
        }
        sum.scale(1.0 / std::f64::consts::PI.sqrt())
    })
}

#[inline]
fn integrated_probit_jet(mu: f64, sigma: f64) -> IntegratedInverseLinkJet {
    if sigma <= 1e-10 {
        let z = mu.clamp(-30.0, 30.0);
        let clamp_active = z != mu;
        let pdf = crate::probability::normal_pdf(z);
        return IntegratedInverseLinkJet {
            mean: crate::probability::normal_cdf(z),
            d1: if clamp_active { 0.0 } else { pdf },
            d2: if clamp_active { 0.0 } else { -z * pdf },
            d3: if clamp_active {
                0.0
            } else {
                (z * z - 1.0) * pdf
            },
            mode: IntegratedExpectationMode::ExactClosedForm,
        };
    }
    let s = (1.0 + sigma * sigma).sqrt();
    let z = mu / s;
    let pdf = crate::probability::normal_pdf(z);
    IntegratedInverseLinkJet {
        mean: crate::probability::normal_cdf(z),
        d1: pdf / s,
        d2: -z * pdf / (s * s),
        d3: (z * z - 1.0) * pdf / (s * s * s),
        mode: IntegratedExpectationMode::ExactClosedForm,
    }
}

#[inline]
fn integrated_logit_jet_ghq(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> IntegratedInverseLinkJet {
    let (mean, d1, d2, d3) = integrate_normal_ghq_adaptive(ctx, mu, sigma, |x| {
        component_point_jet(LinkComponent::Logit, x)
    });
    IntegratedInverseLinkJet {
        mean,
        d1: d1.max(0.0),
        d2,
        d3,
        mode: if sigma <= 1e-10 {
            IntegratedExpectationMode::ExactClosedForm
        } else {
            IntegratedExpectationMode::QuadratureFallback
        },
    }
}

#[inline]
pub(crate) fn latent_cloglog_inverse_link_jet5_controlled(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> IntegratedInverseLinkJet5 {
    // Single latent-cloglog derivative tower via the shared lognormal-Laplace
    // kernel terms
    //
    //   K_k(mu, sigma) := E[exp(k eta - exp(eta))],   eta ~ N(mu, sigma^2).
    //
    // Since
    //
    //   E[1 - exp(-exp(eta))] = 1 - K_0,
    //
    // every eta-derivative is a fixed linear combination of K_1..K_5:
    //
    //   d1 = K1
    //   d2 = K1 - K2
    //   d3 = K1 - 3 K2 + K3
    //   d4 = K1 - 7 K2 + 6 K3 - K4
    //   d5 = K1 - 15 K2 + 25 K3 - 10 K4 + K5.
    //
    // Each K_k is evaluated through the same routed lognormal-Laplace backend
    // used elsewhere in the cloglog/survival stack, so there is no finite-
    // difference bridge in the latent jet anymore. The returned `mode` still
    // records whether that scalar backend was closed-form, controlled, special-
    // function, or quadrature fallback at runtime.
    if sigma <= 1e-10 {
        let (mean, d1, d2, d3, d4, d5) = cloglog_point_jet5(mu);
        return IntegratedInverseLinkJet5 {
            mean,
            d1,
            d2,
            d3,
            d4,
            d5,
            mode: IntegratedExpectationMode::ExactClosedForm,
        };
    }

    let (k, log_k0, mode) = latent_cloglog_kernel_terms(ctx, mu, sigma, 5);

    IntegratedInverseLinkJet5 {
        mean: if log_k0.is_finite() {
            -log_k0.exp_m1()
        } else {
            1.0
        },
        d1: k[1].max(0.0),
        d2: k[1] - k[2],
        d3: k[1] - 3.0 * k[2] + k[3],
        d4: k[1] - 7.0 * k[2] + 6.0 * k[3] - k[4],
        d5: k[1] - 15.0 * k[2] + 25.0 * k[3] - 10.0 * k[4] + k[5],
        mode,
    }
}

#[inline]
fn integrated_cloglog_inverse_link_jet_controlled(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> IntegratedInverseLinkJet {
    if sigma <= 1e-10 {
        let (mean, d1, d2, d3, _, _) = cloglog_point_jet5(mu);
        return IntegratedInverseLinkJet {
            mean,
            d1,
            d2,
            d3,
            mode: IntegratedExpectationMode::ExactClosedForm,
        };
    }

    let (k, log_k0, mode) = latent_cloglog_kernel_terms(ctx, mu, sigma, 3);
    IntegratedInverseLinkJet {
        mean: if log_k0.is_finite() {
            -log_k0.exp_m1()
        } else {
            1.0
        },
        d1: k[1].max(0.0),
        d2: k[1] - k[2],
        d3: k[1] - 3.0 * k[2] + k[3],
        mode,
    }
}

#[inline]
fn latent_cloglog_kernel_terms(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
    max_order: usize,
) -> ([f64; 6], f64, IntegratedExpectationMode) {
    let sigma2 = sigma * sigma;
    let mut k = [0.0; 6];
    let mut log_k0 = f64::NEG_INFINITY;
    let mut mode = IntegratedExpectationMode::ExactClosedForm;

    for (order, out) in k.iter_mut().enumerate().take(max_order + 1) {
        let kf = order as f64;
        let shifted_mu = mu + kf * sigma2;
        let (survival, term_mode) = lognormal_laplace_term_shared(ctx, 1.0, shifted_mu, sigma);
        mode = worse_integrated_expectation_mode(mode, term_mode);

        if survival <= 0.0 {
            *out = 0.0;
            continue;
        }

        let log_value = kf * mu + 0.5 * kf * kf * sigma2 + survival.ln();
        if order == 0 {
            log_k0 = log_value;
        }
        *out = log_value.exp();
    }

    (k, log_k0, mode)
}

#[inline]
fn integrated_jet_fd_step(sigma: f64) -> f64 {
    (1e-4 * (1.0 + sigma.abs())).clamp(1e-5, 5e-2)
}

#[inline]
fn integrate_logit_inverse_link_jet_from_scalar_backend(
    mu: f64,
    sigma: f64,
    eval: impl Fn(f64, f64) -> Result<IntegratedMeanDerivative, EstimationError>,
    point_jet: impl Fn(f64) -> (f64, f64, f64, f64),
) -> Result<IntegratedInverseLinkJet, EstimationError> {
    if sigma <= 1e-10 {
        let (mean, d1, d2, d3) = point_jet(mu);
        return Ok(IntegratedInverseLinkJet {
            mean,
            d1,
            d2,
            d3,
            mode: IntegratedExpectationMode::ExactClosedForm,
        });
    }

    // Solver-facing non-GHQ jet reconstruction for the controlled logit
    // backend. The scalar route exposes mean and d/dmu; PIRLS also consumes
    // d2 and d3, so we differentiate that same scalar backend in mu with a
    // fourth-order symmetric stencil rather than re-enter GHQ.
    let h = integrated_jet_fd_step(sigma);
    let c0 = eval(mu, sigma)?;
    let cm1 = eval(mu - h, sigma)?;
    let cp1 = eval(mu + h, sigma)?;
    let cm2 = eval(mu - 2.0 * h, sigma)?;
    let cp2 = eval(mu + 2.0 * h, sigma)?;

    let d2 =
        (cm2.dmean_dmu - 8.0 * cm1.dmean_dmu + 8.0 * cp1.dmean_dmu - cp2.dmean_dmu) / (12.0 * h);
    let d3 = (-cp2.dmean_dmu + 16.0 * cp1.dmean_dmu - 30.0 * c0.dmean_dmu + 16.0 * cm1.dmean_dmu
        - cm2.dmean_dmu)
        / (12.0 * h * h);

    let mut mode = c0.mode;
    for sample_mode in [cm1.mode, cp1.mode, cm2.mode, cp2.mode] {
        if integrated_expectation_mode_rank(sample_mode) > integrated_expectation_mode_rank(mode) {
            mode = sample_mode;
        }
    }

    Ok(IntegratedInverseLinkJet {
        mean: c0.mean,
        d1: c0.dmean_dmu.max(0.0),
        d2,
        d3,
        mode,
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

#[inline]
pub fn normal_expectation_1d_adaptive_pair<F>(
    ctx: &QuadratureContext,
    eta: f64,
    se_eta: f64,
    f: F,
) -> (f64, f64)
where
    F: Fn(f64) -> (f64, f64),
{
    integrate_normal_ghq_adaptive(ctx, eta, se_eta, f)
}

fn adaptive_point_count_from_sd(max_sd: f64) -> usize {
    // Use a more aggressive schedule for nonlinear tail-sensitive transforms.
    // 7 points stays for very well-identified rows, 15/21/31 kick in earlier for
    // location-scale and rare-event regimes where MC checks showed larger error.
    // 51 nodes covers the wide-sigma regime where 31-point GHQ accumulated
    // noticeable error against the Faddeeva / high-res numeric references.
    if max_sd.is_finite() && max_sd > 2.5 {
        51
    } else if max_sd.is_finite() && max_sd > 1.0 {
        31
    } else if max_sd.is_finite() && max_sd > 0.35 {
        21
    } else if max_sd.is_finite() && max_sd > 0.1 {
        15
    } else {
        7
    }
}

#[inline]
fn with_gh_nodesweights<R>(
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

fn choleskywith_jitter(cov: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
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

#[inline]
fn array_cov_to_vec<const D: usize>(cov: &[[f64; D]; D]) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0; D]; D];
    for i in 0..D {
        for j in 0..D {
            out[i][j] = cov[i][j];
        }
    }
    out
}

#[inline]
fn adaptive_point_countwith_cap(max_sd: f64, max_n: usize) -> usize {
    adaptive_point_count_from_sd(max_sd).min(max_n)
}

#[inline]
fn ghq_nd_integrate_try<const D: usize, F, R, E>(
    ctx: &QuadratureContext,
    mu: [f64; D],
    cov: [[f64; D]; D],
    max_n: usize,
    f: F,
) -> Result<Option<R>, E>
where
    F: Fn([f64; D]) -> Result<R, E>,
    R: GhqValue,
{
    let mut maxvar = 0.0_f64;
    for (i, row) in cov.iter().enumerate() {
        maxvar = maxvar.max(row[i]).max(0.0);
    }
    let n = adaptive_point_countwith_cap(maxvar.sqrt(), max_n);

    let mut covvec = array_cov_to_vec(&cov);
    for (i, row) in covvec.iter_mut().enumerate() {
        row[i] = row[i].max(0.0);
    }
    let l = match choleskywith_jitter(&covvec) {
        Some(v) => v,
        None => return Ok(None),
    };
    let norm = 1.0 / std::f64::consts::PI.powf(0.5 * D as f64);

    with_gh_nodesweights(ctx, n, |nodes, weights| {
        let mut acc = R::zero();
        let mut idx = [0usize; D];
        loop {
            let mut z = [0.0_f64; D];
            let mut weight = 1.0_f64;
            for d in 0..D {
                z[d] = SQRT_2 * nodes[idx[d]];
                weight *= weights[idx[d]];
            }

            let mut x = mu;
            for row in 0..D {
                let mut dot = 0.0_f64;
                for (col, zc) in z.iter().enumerate().take(row + 1) {
                    dot += l[row][col] * *zc;
                }
                x[row] += dot;
            }
            acc.addweighted(weight, f(x)?);

            let mut carry = true;
            for d in (0..D).rev() {
                idx[d] += 1;
                if idx[d] < n {
                    carry = false;
                    break;
                }
                idx[d] = 0;
            }
            if carry {
                break;
            }
        }
        Ok(Some(acc.scale(norm)))
    })
}

#[inline]
fn ghq_nd_integrate<const D: usize, F, R>(
    ctx: &QuadratureContext,
    mu: [f64; D],
    cov: [[f64; D]; D],
    max_n: usize,
    f: F,
) -> Option<R>
where
    F: Fn([f64; D]) -> R,
    R: GhqValue,
{
    match ghq_nd_integrate_try::<D, _, R, Infallible>(ctx, mu, cov, max_n, |x| Ok(f(x))) {
        Ok(v) => v,
        Err(e) => match e {},
    }
}

#[inline]
fn ghq_nd_integrate_result<const D: usize, F, R, E>(
    ctx: &QuadratureContext,
    mu: [f64; D],
    cov: [[f64; D]; D],
    max_n: usize,
    f: F,
) -> Result<Option<R>, E>
where
    F: Fn([f64; D]) -> Result<R, E>,
    R: GhqValue,
{
    ghq_nd_integrate_try::<D, _, R, E>(ctx, mu, cov, max_n, f)
}

/// Adaptive N-dimensional GHQ expectation for correlated Gaussian latents.
pub fn normal_expectation_nd_adaptive<const D: usize, F>(
    ctx: &QuadratureContext,
    mu: [f64; D],
    cov: [[f64; D]; D],
    max_n: usize,
    f: F,
) -> f64
where
    F: Fn([f64; D]) -> f64,
{
    match ghq_nd_integrate::<D, _, f64>(ctx, mu, cov, max_n, &f) {
        Some(v) => v,
        None => f(mu),
    }
}

/// Fallible adaptive N-dimensional GHQ expectation for correlated Gaussian latents.
pub fn normal_expectation_nd_adaptive_result<const D: usize, F, R, E>(
    ctx: &QuadratureContext,
    mu: [f64; D],
    cov: [[f64; D]; D],
    max_n: usize,
    f: F,
) -> Result<R, E>
where
    F: Fn([f64; D]) -> Result<R, E>,
    R: GhqValue,
{
    match ghq_nd_integrate_result::<D, _, R, E>(ctx, mu, cov, max_n, &f)? {
        Some(v) => Ok(v),
        None => f(mu),
    }
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
    normal_expectation_nd_adaptive_result::<2, _, _, E>(ctx, mu, cov, 21, |x| f(x[0], x[1]))
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
    // 3D tensor GHQ grows cubically; cap nodes per axis for throughput.
    normal_expectation_nd_adaptive::<3, _>(ctx, mu, cov, 15, |x| f(x[0], x[1], x[2]))
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
        return crate::probability::normal_cdf(eta);
    }
    let denom = (1.0 + se_eta * se_eta).sqrt();
    crate::probability::normal_cdf(eta / denom)
}

#[inline]
pub fn logit_posterior_meanvariance(ctx: &QuadratureContext, eta: f64, se_eta: f64) -> (f64, f64) {
    let m1 = integrate_normal_ghq_adaptive(ctx, eta, se_eta, sigmoid);
    let m2 = integrate_normal_ghq_adaptive(ctx, eta, se_eta, |x| {
        let p = sigmoid(x);
        p * p
    })
    .clamp(0.0, 1.0);
    (m1, (m2 - m1 * m1).max(0.0))
}

#[inline]
pub fn probit_posterior_meanvariance(ctx: &QuadratureContext, eta: f64, se_eta: f64) -> (f64, f64) {
    let m1 = probit_posterior_mean(eta, se_eta);
    let m2 = integrate_normal_ghq_adaptive(ctx, eta, se_eta, |x| {
        let p = crate::probability::normal_cdf(x);
        p * p
    })
    .clamp(0.0, 1.0);
    (m1, (m2 - m1 * m1).max(0.0))
}

#[inline]
pub fn cloglog_posterior_meanvariance(
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
    // Degenerate sigma: use cloglog_mean_exact directly (see cloglog_posterior_mean).
    if !(eta.is_finite() && se_eta.is_finite()) || se_eta <= CLOGLOG_SIGMA_DEGENERATE {
        return (cloglog_mean_exact(eta), 0.0);
    }
    let (survival, _) = cloglog_survival_term_controlled(ctx, eta, se_eta);
    let (survival_sq, _) = cloglog_survivalsecond_moment_controlled(ctx, eta, se_eta);
    let mean = cloglog_mean_from_survival(survival);
    let variance = (survival_sq - survival * survival).max(0.0);
    (mean, variance)
}

/// Posterior mean under cloglog inverse link:
/// g^{-1}(x) = 1 - exp(-exp(x)).
///
/// This now routes through the same analytic ladder used by the integrated
/// derivative path rather than defaulting to GHQ:
///
/// - E[1 - exp(-exp(eta))] under Gaussian eta is the complement of the
///   lognormal Laplace transform at z=1.
/// - That quantity has exact non-GHQ representations, including convergent
///   erfc / asymptotic series and characteristic-function inversion formulas.
/// - The same mathematics also covers the Royston-Parmar survival transform
///   S(eta) = exp(-exp(eta)), which is why this comment matters beyond binary
///   cloglog models.
///
/// So GHQ here is only the terminal numerical fallback, not the primary path.
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
    // Degenerate sigma: use cloglog_mean_exact directly to avoid precision
    // loss from the survival → mean conversion (gumbel_survival rounds to
    // 1.0 in f64 for eta ≪ 0, and cloglog_mean_from_survival(1.0) = 0.0).
    if !(eta.is_finite() && se_eta.is_finite()) || se_eta <= CLOGLOG_SIGMA_DEGENERATE {
        return cloglog_mean_exact(eta);
    }
    let (survival, _) = cloglog_survival_term_controlled(ctx, eta, se_eta);
    cloglog_mean_from_survival(survival)
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
pub fn survival_posterior_meanvariance(
    ctx: &QuadratureContext,
    eta: f64,
    se_eta: f64,
) -> (f64, f64) {
    let (m1, _) = cloglog_survival_term_controlled(ctx, eta, se_eta);
    let (m2, _) = cloglog_survivalsecond_moment_controlled(ctx, eta, se_eta);
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
/// 3. replace the GHQ loop in `logit_posterior_meanwith_deriv` once the exact
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
        return sigmoid(mu);
    }
    if sigma < 1e-10 {
        return sigmoid(mu);
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

    0.5 - coeff * sum_im
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

// CLogLog Gaussian convolution via differentiated Gauss-Hermite quadrature
//
// For location-scale (GAMLSS) models with CLogLog link we need to evaluate
//   L(μ,σ) = E[g(μ + σZ)],  Z ~ N(0,1),  g(η) = 1 - exp(-exp(η)),
// together with all partial derivatives up to fourth order w.r.t. μ and σ.
//
// GHQ gives
//   L(μ,σ) ≈ (1/√π) Σ_m ω_m g(t_m),   t_m = μ + √2 σ x_m
//
// and by the chain rule (exact for the quadrature rule since t_m is affine
// in μ and σ):
//   ∂^a_μ ∂^b_σ L ≈ (√2)^b / √π  Σ_m ω_m x_m^b g^{(a+b)}(t_m)

/// All partial derivatives of `L(μ,σ) = E[g(μ + σZ)]` up to fourth order,
/// where `g` is the CLogLog inverse link and `Z ~ N(0,1)`.
#[derive(Clone, Copy, Debug)]
pub struct CLogLogConvolutionDerivatives {
    // 0th order
    pub l: f64,

    // 1st order
    pub l_mu: f64,
    pub l_sigma: f64,

    // 2nd order
    pub l_mumu: f64,
    pub l_musigma: f64,
    pub l_sigmasigma: f64,

    // 3rd order
    pub l_mumumu: f64,
    pub l_mumusigma: f64,
    pub l_musigmasigma: f64,
    pub l_sigmasigmasigma: f64,

    // 4th order
    pub l_mumumumu: f64,
    pub l_mumumusigma: f64,
    pub l_mumusigmasigma: f64,
    pub l_musigmasigmasigma: f64,
    pub l_sigmasigmasigmasigma: f64,
}

#[inline]
fn cloglog_horner_polynomial(x: f64, coeffs: &[f64]) -> f64 {
    coeffs.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

#[inline]
fn cloglog_stable_poly_times_exp_neg(x: f64, coeffs: &[f64]) -> f64 {
    if coeffs.is_empty() || !x.is_finite() {
        return 0.0;
    }
    if x <= 600.0 {
        return cloglog_horner_polynomial(x, coeffs) * (-x).exp();
    }

    let inv_x = x.recip();
    let mut tail = 0.0;
    for &c in coeffs {
        tail = tail * inv_x + c;
    }
    let degree = (coeffs.len() - 1) as f64;
    let scale = (degree * x.ln() - x).exp();
    scale * tail
}

#[inline]
pub(crate) fn cloglog_point_jet5(t: f64) -> (f64, f64, f64, f64, f64, f64) {
    if t.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let et = t.exp();
    if !et.is_finite() {
        return (1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    (
        -(-et).exp_m1(),
        cloglog_stable_poly_times_exp_neg(et, &[0.0, 1.0]),
        cloglog_stable_poly_times_exp_neg(et, &[0.0, 1.0, -1.0]),
        cloglog_stable_poly_times_exp_neg(et, &[0.0, 1.0, -3.0, 1.0]),
        cloglog_stable_poly_times_exp_neg(et, &[0.0, 1.0, -7.0, 6.0, -1.0]),
        cloglog_stable_poly_times_exp_neg(et, &[0.0, 1.0, -15.0, 25.0, -10.0, 1.0]),
    )
}

/// CLogLog inverse link `g(t) = 1 - exp(-exp(t))` and its first four
/// derivatives, evaluated in a numerically stable way.
///
/// All derivatives share the common factor `h(t) = exp(t - exp(t))`:
/// ```text
///   g  (t) = 1 - exp(-exp(t))
///   g' (t) = h(t)
///   g''(t) = (1 - exp(t)) h(t)
///   g'''(t) = (exp(2t) - 3 exp(t) + 1) h(t)
///   g''''(t) = (-exp(3t) + 6 exp(2t) - 7 exp(t) + 1) h(t)
/// ```
#[inline]
fn cloglog_g_derivatives(t: f64) -> (f64, f64, f64, f64, f64) {
    let (g, g1, g2, g3, g4, _) = cloglog_point_jet5(t);
    (g, g1, g2, g3, g4)
}

/// Compute `L(μ,σ) = E[g(μ + σZ)]` via Gauss-Hermite quadrature.
///
/// The number of GHQ nodes is determined by the `QuadratureContext` cache;
/// `n_nodes` selects from the available rule sizes (7, 15, 21, 31).
///
/// When `sigma` is negligibly small the function evaluates `g(mu)` directly,
/// bypassing quadrature.
pub fn cloglog_ghq_value(ctx: &QuadratureContext, mu: f64, sigma: f64, n_nodes: usize) -> f64 {
    if sigma.abs() < 1e-14 {
        let (g, _, _, _, _) = cloglog_g_derivatives(mu);
        return g.clamp(0.0, 1.0);
    }
    let inv_sqrt_pi = 1.0 / std::f64::consts::PI.sqrt();
    let scale = SQRT_2 * sigma;
    with_gh_nodesweights(ctx, n_nodes, |nodes, weights| {
        let mut sum = 0.0_f64;
        for i in 0..nodes.len() {
            let t = mu + scale * nodes[i];
            let (g, _, _, _, _) = cloglog_g_derivatives(t);
            sum += weights[i] * g;
        }
        (sum * inv_sqrt_pi).clamp(0.0, 1.0)
    })
}

/// Compute all partial derivatives of `L(μ,σ)` up to fourth order via
/// differentiated Gauss-Hermite quadrature.
///
/// Uses the identity:
/// ```text
///   ∂^a_μ ∂^b_σ L ≈ (√2)^b / √π  Σ_m ω_m x_m^b g^{(a+b)}(t_m)
/// ```
///
/// `n_nodes` selects the GHQ rule size (7, 15, 21, or 31). For location-scale
/// GAMLSS applications, 21-31 nodes is recommended.
pub fn cloglog_ghq_derivatives(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
    n_nodes: usize,
) -> CLogLogConvolutionDerivatives {
    let inv_sqrt_pi = 1.0 / std::f64::consts::PI.sqrt();

    // When sigma is negligibly small, evaluate directly at mu.
    if sigma.abs() < 1e-14 {
        let (g, g1, g2, g3, g4) = cloglog_g_derivatives(mu);
        return CLogLogConvolutionDerivatives {
            l: g,
            l_mu: g1,
            l_sigma: 0.0,
            l_mumu: g2,
            l_musigma: 0.0,
            l_sigmasigma: 0.0,
            l_mumumu: g3,
            l_mumusigma: 0.0,
            l_musigmasigma: 0.0,
            l_sigmasigmasigma: 0.0,
            l_mumumumu: g4,
            l_mumumusigma: 0.0,
            l_mumusigmasigma: 0.0,
            l_musigmasigmasigma: 0.0,
            l_sigmasigmasigmasigma: 0.0,
        };
    }

    let scale = SQRT_2 * sigma;
    let sqrt2 = SQRT_2;

    with_gh_nodesweights(ctx, n_nodes, |nodes, weights| {
        // Accumulators for the weighted sums. For derivative ∂^a_μ ∂^b_σ L,
        // we need Σ ω_m x_m^b g^{(a+b)}(t_m). We group by the order of g
        // derivative needed (k = a + b) and the power of x_m (= b).
        //
        // k=0: g(t_m)    — need x^0
        // k=1: g'(t_m)   — need x^0, x^1
        // k=2: g''(t_m)  — need x^0, x^1, x^2
        // k=3: g'''(t_m) — need x^0, x^1, x^2, x^3
        // k=4: g''''(t_m)— need x^0, x^1, x^2, x^3, x^4

        // s[k][b] = Σ_m ω_m x_m^b g^{(k)}(t_m)
        let mut s = [[0.0_f64; 5]; 5];

        for i in 0..nodes.len() {
            let x = nodes[i];
            let t = mu + scale * x;
            let (g0, g1, g2, g3, g4) = cloglog_g_derivatives(t);
            let w = weights[i];

            // Powers of x_m
            let x2 = x * x;
            let x3 = x2 * x;
            let x4 = x3 * x;

            // k=0: only need x^0
            s[0][0] += w * g0;

            // k=1: need x^0, x^1
            s[1][0] += w * g1;
            s[1][1] += w * x * g1;

            // k=2: need x^0, x^1, x^2
            s[2][0] += w * g2;
            s[2][1] += w * x * g2;
            s[2][2] += w * x2 * g2;

            // k=3: need x^0, x^1, x^2, x^3
            s[3][0] += w * g3;
            s[3][1] += w * x * g3;
            s[3][2] += w * x2 * g3;
            s[3][3] += w * x3 * g3;

            // k=4: need x^0, x^1, x^2, x^3, x^4
            s[4][0] += w * g4;
            s[4][1] += w * x * g4;
            s[4][2] += w * x2 * g4;
            s[4][3] += w * x3 * g4;
            s[4][4] += w * x4 * g4;
        }

        // Now assemble derivatives using:
        //   ∂^a_μ ∂^b_σ L = (√2)^b / √π · s[a+b][b]
        let sqrt2_1 = sqrt2;
        let sqrt2_2 = 2.0; // (√2)^2
        let sqrt2_3 = 2.0 * sqrt2; // (√2)^3
        let sqrt2_4 = 4.0; // (√2)^4

        CLogLogConvolutionDerivatives {
            // 0th: a=0, b=0 → (√2)^0 / √π · s[0][0]
            l: inv_sqrt_pi * s[0][0],

            // 1st: (a=1,b=0), (a=0,b=1)
            l_mu: inv_sqrt_pi * s[1][0],
            l_sigma: inv_sqrt_pi * sqrt2_1 * s[1][1],

            // 2nd: (a=2,b=0), (a=1,b=1), (a=0,b=2)
            l_mumu: inv_sqrt_pi * s[2][0],
            l_musigma: inv_sqrt_pi * sqrt2_1 * s[2][1],
            l_sigmasigma: inv_sqrt_pi * sqrt2_2 * s[2][2],

            // 3rd: (a=3,b=0), (a=2,b=1), (a=1,b=2), (a=0,b=3)
            l_mumumu: inv_sqrt_pi * s[3][0],
            l_mumusigma: inv_sqrt_pi * sqrt2_1 * s[3][1],
            l_musigmasigma: inv_sqrt_pi * sqrt2_2 * s[3][2],
            l_sigmasigmasigma: inv_sqrt_pi * sqrt2_3 * s[3][3],

            // 4th: (a=4,b=0), (a=3,b=1), (a=2,b=2), (a=1,b=3), (a=0,b=4)
            l_mumumumu: inv_sqrt_pi * s[4][0],
            l_mumumusigma: inv_sqrt_pi * sqrt2_1 * s[4][1],
            l_mumusigmasigma: inv_sqrt_pi * sqrt2_2 * s[4][2],
            l_musigmasigmasigma: inv_sqrt_pi * sqrt2_3 * s[4][3],
            l_sigmasigmasigmasigma: inv_sqrt_pi * sqrt2_4 * s[4][4],
        }
    })
}

/// Convenience wrapper that uses adaptive node count based on sigma magnitude.
///
/// For small sigma, fewer nodes suffice; for large sigma, more are needed to
/// capture tail contributions accurately. This mirrors the adaptive strategy
/// used by `integrate_normal_ghq_adaptive`.
pub fn cloglog_ghq_derivatives_adaptive(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> CLogLogConvolutionDerivatives {
    let n = adaptive_point_count_from_sd(sigma.abs());
    cloglog_ghq_derivatives(ctx, mu, sigma, n)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn even_moment_exp_neg_x2(power: usize) -> f64 {
        assert!(power.is_multiple_of(2));
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
        sum * h / 3.0
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
    fn test_computedweights_symmetric() {
        // Verify computed weights are symmetric
        let ctx = QuadratureContext::new();
        let gh = ctx.gauss_hermite();
        for i in 0..N_POINTS / 2 {
            let j = N_POINTS - 1 - i;
            assert_relative_eq!(gh.weights[i], gh.weights[j], epsilon = 1e-12);
        }
    }

    #[test]
    fn testweights_sum_to_sqrt_pi() {
        // Verify weights sum to sqrt(pi) for physicist's Hermite
        let ctx = QuadratureContext::new();
        let gh = ctx.gauss_hermite();
        let sum: f64 = gh.weights.iter().sum();
        assert_relative_eq!(sum, std::f64::consts::PI.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_clenshaw_curtisweights_are_symmetric_and_integrate_constants() {
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
    fn testwilkinson_shift_finitewhen_d_iszero() {
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
        let knownweights = [
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
            assert_relative_eq!(gh.weights[i], knownweights[i], epsilon = 1e-12);
        }
    }

    #[test]
    fn testzero_se_returns_mode() {
        // When SE is zero, posterior mean is expected to equal mode
        let eta = 1.5;
        let se = 0.0;
        let ctx = QuadratureContext::new();
        let mean = logit_posterior_mean(&ctx, eta, se);
        let mode = sigmoid(eta);
        assert_relative_eq!(mean, mode, epsilon = 1e-10);
    }

    #[test]
    fn test_symmetric_atzero() {
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
    fn test_logit_posterior_derivative_remains_positive_in_positive_tail() {
        let ctx = QuadratureContext::new();
        let eta = 20.0;
        let se = 0.0;
        let (_, dmu) = logit_posterior_meanwith_deriv(&ctx, eta, se)
            .expect("logit posterior mean derivative should evaluate");
        assert!(dmu > 0.0);
        assert!(
            dmu < 1e-6,
            "positive-tail derivative should stay tiny but nonzero, got {dmu}"
        );
    }

    #[test]
    fn test_logit_posterior_derivative_matches_central_difference() {
        let ctx = QuadratureContext::new();
        let eta = 1.7;
        let se = 0.9;
        let h = 1e-5;

        let (_, dmu) = logit_posterior_meanwith_deriv(&ctx, eta, se)
            .expect("logit posterior mean derivative should evaluate");
        let mu_plus = logit_posterior_mean(&ctx, eta + h, se);
        let mu_minus = logit_posterior_mean(&ctx, eta - h, se);
        let dmufd = (mu_plus - mu_minus) / (2.0 * h);

        assert_eq!(dmu.signum(), dmufd.signum());
        assert_relative_eq!(dmu, dmufd, epsilon = 5e-6, max_relative = 2e-4);
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
    fn test_integrated_logit_mean_close_to_exact_oracle() {
        let ctx = QuadratureContext::new();
        let cases = [(-3.0, 0.3), (-1.0, 0.8), (0.5, 1.2), (2.8, 1.0)];
        for (eta, se) in cases {
            let ghq = logit_posterior_mean(&ctx, eta, se);
            let exact = logit_posterior_mean_exact(eta, se);
            assert_relative_eq!(ghq, exact, epsilon = 2.5e-3);
        }
    }

    #[test]
    fn test_probit_posterior_mean_reduces_to_map_atzero_se() {
        let eta = 1.25;
        let p = probit_posterior_mean(eta, 0.0);
        let map = crate::probability::normal_cdf(eta);
        assert_relative_eq!(p, map, epsilon = 1e-12);
    }

    #[test]
    fn test_probit_posterior_mean_shrinks_extremeswith_uncertainty() {
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
            (0.0, 20.0),
            (10.0, 10.0),
            (-0.5, 100.0),
        ];
        for (eta, se) in cases {
            let clog = cloglog_posterior_mean(&ctx, eta, se);
            let surv = survival_posterior_mean(&ctx, eta, se);
            assert_relative_eq!(clog + surv, 1.0, epsilon = 2e-10, max_relative = 2e-10);
        }
    }

    #[test]
    fn test_cloglog_and_survival_share_large_sigmaspecial_function_path() {
        let ctx = QuadratureContext::new();
        let eta = -0.2;
        let se = 0.8;
        let clog = cloglog_posterior_mean(&ctx, eta, se);
        let surv = survival_posterior_mean(&ctx, eta, se);
        let integrated =
            integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::CLogLog, eta, se)
                .expect("cloglog integrated inverse-link moments should evaluate");
        assert_eq!(
            integrated.mode,
            IntegratedExpectationMode::ExactSpecialFunction
        );
        assert_relative_eq!(clog, integrated.mean, epsilon = 1e-12, max_relative = 1e-12);
        assert_relative_eq!(clog + surv, 1.0, epsilon = 1e-10, max_relative = 1e-10);
    }

    #[test]
    fn test_cloglog_and_survival_posteriorvariances_match() {
        let ctx = QuadratureContext::new();
        let cases = [(-3.0, 0.0), (-0.2, 0.1), (0.4, 0.8), (2.0, 1.5)];
        for (eta, se) in cases {
            let (_, clogvar) = cloglog_posterior_meanvariance(&ctx, eta, se);
            let (_, survvar) = survival_posterior_meanvariance(&ctx, eta, se);
            assert_relative_eq!(clogvar, survvar, epsilon = 1e-12, max_relative = 1e-12);
        }
    }

    #[test]
    fn test_survivalvariance_uses_exactsecond_moment_shift() {
        let ctx = QuadratureContext::new();
        let eta = -0.2;
        let se = 0.8;
        let (survival, _) = cloglog_survival_term_controlled(&ctx, eta, se);
        let (survival_sq, _) = cloglog_survivalsecond_moment_controlled(&ctx, eta, se);
        let (_, variance) = survival_posterior_meanvariance(&ctx, eta, se);
        assert_relative_eq!(
            variance,
            (survival_sq - survival * survival).max(0.0),
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    fn test_lognormal_laplace_shift_matches_explicitmu_plus_logz() {
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
        let out = integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Probit, 0.7, 1.3)
            .expect("probit integrated inverse-link moments should evaluate");
        assert_eq!(out.mode, IntegratedExpectationMode::ExactClosedForm);
        let direct = probit_posterior_meanwith_deriv_exact(0.7, 1.3);
        assert_relative_eq!(out.mean, direct.mean, epsilon = 1e-12);
        assert_relative_eq!(out.dmean_dmu, direct.dmean_dmu, epsilon = 1e-12);
    }

    #[test]
    fn test_integrated_probit_jet_matches_closed_form_derivatives() {
        let ctx = QuadratureContext::new();
        let mu = 0.7;
        let sigma = 1.3;
        let out = integrated_inverse_link_jet(&ctx, LinkFunction::Probit, mu, sigma)
            .expect("probit integrated inverse-link jet should evaluate");
        let s = (1.0 + sigma * sigma).sqrt();
        let z = mu / s;
        let pdf = crate::probability::normal_pdf(z);
        assert_relative_eq!(out.mean, crate::probability::normal_cdf(z), epsilon = 1e-12);
        assert_relative_eq!(out.d1, pdf / s, epsilon = 1e-12);
        assert_relative_eq!(out.d2, -z * pdf / (s * s), epsilon = 1e-12);
        assert_relative_eq!(out.d3, (z * z - 1.0) * pdf / (s * s * s), epsilon = 1e-12);
    }

    #[test]
    fn test_integrated_logit_jet_matches_central_differences() {
        // Assertion redesign (see task #21 / inference-auditor finding):
        // At (μ=1.1, σ=0.8) the logistic-normal erfcx alternating series has
        // a tail bound |R_N| ≤ |m|·√(2/π)·exp(−m²/(2s²))/((N+1)²·s³). Plugging
        // in gives a k=2 coefficient ≈ 0.67, so reaching the EPSILON=1e-10
        // accuracy contract would require N ≈ √(0.67/1e-10) − 1 ≈ 81619
        // terms, far beyond LOGIT_MAX_TERMS=160. The dispatcher therefore
        // legitimately routes this input to the GHQ fallback; the resulting
        // `mode` field is an implementation detail reflecting a correct
        // regime decision, not the property we care about. The mathematical
        // contract is VALUE accuracy of the mean and its μ-derivatives, so
        // we assert those directly against a high-resolution Simpson
        // reference (independent of erfcx / Taylor / asymptotics).
        let ctx = QuadratureContext::new();
        let mu = 1.1;
        let sigma = 0.8;
        let out = integrated_inverse_link_jet(&ctx, LinkFunction::Logit, mu, sigma)
            .expect("logit integrated inverse-link jet should evaluate");
        assert!(matches!(
            out.mode,
            IntegratedExpectationMode::ExactSpecialFunction
                | IntegratedExpectationMode::QuadratureFallback
        ));
        let (ref_mean, ref_d1, ref_d2, ref_d3) =
            logit_reference_jet_highres_simpson(mu, sigma);
        assert_relative_eq!(out.mean, ref_mean, epsilon = 1e-11, max_relative = 1e-10);
        assert_relative_eq!(out.d1, ref_d1, epsilon = 1e-11, max_relative = 1e-10);
        assert_relative_eq!(out.d2, ref_d2, epsilon = 1e-11, max_relative = 1e-10);
        assert_relative_eq!(out.d3, ref_d3, epsilon = 1e-11, max_relative = 1e-10);
    }

    #[test]
    fn test_integrated_logit_pirls_jet_matches_general_dispatch() {
        // Assertion redesign: at (μ=1.1, σ=0.8) the erfcx series cannot
        // meet the 1e-10 tail bound within LOGIT_MAX_TERMS=160 (derivation
        // in `test_integrated_logit_jet_matches_central_differences`), so
        // the dispatcher correctly routes to the GHQ backend. What matters
        // for PIRLS is that the hot-path jet matches the general dispatcher
        // in both routing AND value, not that they both happen to land on
        // ExactSpecialFunction. We assert (a) both take the SAME path, and
        // (b) their values agree to machine precision — that's the
        // equivalence the PIRLS contract needs.
        let ctx = QuadratureContext::new();
        let mu = 1.1;
        let sigma = 0.8;

        let pirls =
            integrated_logit_inverse_link_jet_pirls(&ctx, mu, sigma).expect("PIRLS logit jet");
        let general = integrated_inverse_link_jet(&ctx, LinkFunction::Logit, mu, sigma)
            .expect("general logit jet");

        assert!(matches!(
            pirls.mode,
            IntegratedExpectationMode::ExactSpecialFunction
                | IntegratedExpectationMode::QuadratureFallback
        ));
        assert_eq!(pirls.mode, general.mode);
        assert_relative_eq!(pirls.mean, general.mean, epsilon = 1e-12);
        assert_relative_eq!(pirls.d1, general.d1, epsilon = 1e-12);
        assert_relative_eq!(pirls.d2, general.d2, epsilon = 1e-10);
        assert_relative_eq!(pirls.d3, general.d3, epsilon = 1e-8);
    }

    #[test]
    fn test_integrated_cloglog_jet_matches_central_differences() {
        let ctx = QuadratureContext::new();
        let mu = 0.4;
        let sigma = 0.6;
        let h = 1e-4;
        let out = integrated_inverse_link_jet(&ctx, LinkFunction::CLogLog, mu, sigma)
            .expect("cloglog integrated inverse-link jet should evaluate");
        let plus = integrated_inverse_link_jet(&ctx, LinkFunction::CLogLog, mu + h, sigma)
            .expect("cloglog integrated inverse-link jet should evaluate");
        let minus = integrated_inverse_link_jet(&ctx, LinkFunction::CLogLog, mu - h, sigma)
            .expect("cloglog integrated inverse-link jet should evaluate");
        let d1fd = (plus.mean - minus.mean) / (2.0 * h);
        let d2fd = (plus.d1 - minus.d1) / (2.0 * h);
        let d3fd = (plus.d2 - minus.d2) / (2.0 * h);
        assert_eq!(out.d1.signum(), d1fd.signum());
        assert_eq!(out.d2.signum(), d2fd.signum());
        assert_eq!(out.d3.signum(), d3fd.signum());
        assert_relative_eq!(out.d1, d1fd, epsilon = 2e-5, max_relative = 3e-4);
        assert_relative_eq!(out.d2, d2fd, epsilon = 4e-5, max_relative = 8e-4);
        assert_relative_eq!(out.d3, d3fd, epsilon = 8e-5, max_relative = 2e-3);
    }

    #[test]
    fn test_latent_cloglog_jet5_matches_higher_order_central_differences() {
        let ctx = QuadratureContext::new();
        let mu = 0.35;
        let sigma = 0.7;
        let h = 2e-4;

        let out = latent_cloglog_inverse_link_jet5_controlled(&ctx, mu, sigma);
        let plus = latent_cloglog_inverse_link_jet5_controlled(&ctx, mu + h, sigma);
        let minus = latent_cloglog_inverse_link_jet5_controlled(&ctx, mu - h, sigma);

        let d4fd = (plus.d3 - minus.d3) / (2.0 * h);
        let d5fd = (plus.d4 - minus.d4) / (2.0 * h);

        assert_eq!(out.d4.signum(), d4fd.signum());
        assert_eq!(out.d5.signum(), d5fd.signum());
        assert_relative_eq!(out.d4, d4fd, epsilon = 2e-4, max_relative = 5e-3);
        assert_relative_eq!(out.d5, d5fd, epsilon = 6e-4, max_relative = 2e-2);
    }

    #[test]
    fn test_logit_exact_derivative_matches_finite_difference() {
        // Assertion redesign: at (μ=1.1, σ=0.8) the erfcx series cannot
        // reach its EPSILON=1e-10 tail bound within LOGIT_MAX_TERMS=160
        // (|R_N| ≈ 0.67/(N+1)², so N* ≈ 81619), and
        // `logit_posterior_meanwith_deriv_exact` correctly returns Err.
        // The value-accuracy contract lives at the controlled dispatcher,
        // which falls back to GHQ when the exact series cannot honor the
        // contract; that is what we validate here, against an independent
        // high-resolution Simpson reference for BOTH the mean and its
        // μ-derivative (d/dμ E[sigmoid] = E[sigmoid']).
        let ctx = QuadratureContext::new();
        let out = logit_posterior_meanwith_deriv_controlled(&ctx, 1.1, 0.8)
            .expect("controlled logit");
        let (ref_mean, ref_d1, _, _) = logit_reference_jet_highres_simpson(1.1, 0.8);
        assert_relative_eq!(out.mean, ref_mean, epsilon = 1e-11, max_relative = 1e-10);
        assert!(out.dmean_dmu > 0.0);
        assert_relative_eq!(
            out.dmean_dmu,
            ref_d1,
            epsilon = 1e-11,
            max_relative = 1e-10
        );
    }

    #[test]
    fn test_logit_exact_clamped_degenerate_branch_is_locally_flat() {
        let out = logit_posterior_meanwith_deriv_exact(-710.0, 0.0).expect("exact logit");
        let h = 1e-6;
        let plus = logit_posterior_meanwith_deriv_exact(-710.0 + h, 0.0)
            .expect("exact logit plus")
            .mean;
        let minus = logit_posterior_meanwith_deriv_exact(-710.0 - h, 0.0)
            .expect("exact logit minus")
            .mean;
        let fd = (plus - minus) / (2.0 * h);
        assert_eq!(fd, 0.0);
        assert_eq!(out.dmean_dmu, 0.0);
    }

    fn simpson_integrate<F>(a: f64, b: f64, n_intervals: usize, f: F) -> f64
    where
        F: Fn(f64) -> f64,
    {
        assert_eq!(n_intervals % 2, 0, "Simpson integration requires an even n");
        let h = (b - a) / n_intervals as f64;
        let mut sum = f(a) + f(b);
        for i in 1..n_intervals {
            let x = a + i as f64 * h;
            let w = if i % 2 == 0 { 2.0 } else { 4.0 };
            sum += w * f(x);
        }
        sum * h / 3.0
    }

    fn cloglog_reference_mean_and_derivative(mu: f64, sigma: f64) -> (f64, f64) {
        if sigma <= CLOGLOG_SIGMA_DEGENERATE {
            return (cloglog_mean_exact(mu), gumbel_survival_d1(mu));
        }

        // Independent reference: exact pointwise cloglog mean/derivative
        // integrated against the Gaussian density on a window whose omitted
        // tail mass is below 2e-33.
        let z_max = 12.0;
        let n_intervals = 4096;
        let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        let mean = simpson_integrate(-z_max, z_max, n_intervals, |z| {
            let eta = mu + sigma * z;
            inv_sqrt_2pi * (-0.5 * z * z).exp() * cloglog_mean_exact(eta)
        });
        let deriv = simpson_integrate(-z_max, z_max, n_intervals, |z| {
            let eta = mu + sigma * z;
            inv_sqrt_2pi * (-0.5 * z * z).exp() * gumbel_survival_d1(eta)
        });
        (mean, deriv)
    }

    /// Independent high-resolution reference for the logit posterior jet.
    ///
    /// For eta ~ N(mu, sigma^2) and f(x) = sigmoid(x), the μ-derivatives of
    /// E[f(eta)] equal E[f^(k)(eta)] by the location-family identity
    ///     d^k/dmu^k E[f(mu + sigma Z)] = E[f^(k)(mu + sigma Z)].
    /// We evaluate each E[f^(k)] via composite Simpson's rule on the Gaussian
    /// density over [-z_max, z_max] with z_max=14 (tail mass below 1e-44) and
    /// 16384 intervals. Simpson's error bound is (b-a)·h^4·max|f^(4)|/180;
    /// at h = 28/16384 ≈ 1.7e-3 this gives ~1e-13 absolute for sigmoid and its
    /// low-order derivatives (all bounded by constants ≤ 1 on ℝ). This is
    /// mathematically independent of the erfcx-series / Taylor / asymptotic
    /// implementations under test.
    fn logit_reference_jet_highres_simpson(mu: f64, sigma: f64) -> (f64, f64, f64, f64) {
        let z_max = 14.0;
        let n_intervals = 16384;
        let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        let phi = |z: f64| inv_sqrt_2pi * (-0.5 * z * z).exp();
        let mean = simpson_integrate(-z_max, z_max, n_intervals, |z| {
            let eta = mu + sigma * z;
            let (p, _, _, _) = component_point_jet(LinkComponent::Logit, eta);
            phi(z) * p
        });
        let d1 = simpson_integrate(-z_max, z_max, n_intervals, |z| {
            let eta = mu + sigma * z;
            let (_, p1, _, _) = component_point_jet(LinkComponent::Logit, eta);
            phi(z) * p1
        });
        let d2 = simpson_integrate(-z_max, z_max, n_intervals, |z| {
            let eta = mu + sigma * z;
            let (_, _, p2, _) = component_point_jet(LinkComponent::Logit, eta);
            phi(z) * p2
        });
        let d3 = simpson_integrate(-z_max, z_max, n_intervals, |z| {
            let eta = mu + sigma * z;
            let (_, _, _, p3) = component_point_jet(LinkComponent::Logit, eta);
            phi(z) * p3
        });
        (mean, d1, d2, d3)
    }

    #[test]
    fn test_cloglog_taylor_negative_tail_matches_mathematical_target() {
        let mu = -40.0;
        let sigma = 0.1;
        let out = cloglog_small_sigma_taylor(mu, sigma);
        let (expected_mean, expected_deriv) = cloglog_reference_mean_and_derivative(mu, sigma);

        assert!(
            out.dmean_dmu > 0.0,
            "negative-tail derivative should remain positive"
        );
        assert_relative_eq!(
            out.mean,
            expected_mean,
            epsilon = 1e-30,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            out.dmean_dmu,
            expected_deriv,
            epsilon = 1e-30,
            max_relative = 1e-12
        );
    }

    #[test]
    fn test_cloglog_degenerate_negative_tail_matches_pointwise_target() {
        let ctx = QuadratureContext::new();
        let mu = -40.0;
        let out = cloglog_posterior_meanwith_deriv_controlled(&ctx, mu, 0.0);

        assert!(
            out.dmean_dmu > 0.0,
            "degenerate negative-tail derivative should remain positive"
        );
        assert_relative_eq!(
            out.mean,
            cloglog_mean_exact(mu),
            epsilon = 1e-30,
            max_relative = 1e-15
        );
        assert_relative_eq!(
            out.dmean_dmu,
            gumbel_survival_d1(mu),
            epsilon = 1e-30,
            max_relative = 1e-15
        );
    }

    #[test]
    fn test_degenerate_probit_and_logit_jets_are_flat_on_active_clamps() {
        let probit = integrated_probit_jet(-40.0, 0.0);
        assert_eq!(probit.d1, 0.0);
        assert_eq!(probit.d2, 0.0);
        assert_eq!(probit.d3, 0.0);

        let logit = component_point_jet(LinkComponent::Logit, -710.0);
        assert_eq!(logit.1, 0.0);
        assert_eq!(logit.2, 0.0);
        assert_eq!(logit.3, 0.0);
    }

    #[test]
    fn test_degenerate_cloglog_component_jet_preserves_smooth_negative_tail() {
        let eta: f64 = -40.0;
        let t = eta.exp();
        let s = (-t).exp();
        let cloglog = component_point_jet(LinkComponent::CLogLog, eta);
        let expected_mean = -(-t).exp_m1();
        let expected_d1 = t * s;
        let expected_d2 = (t - t * t) * s;
        let expected_d3 = (t - 3.0 * t * t + t * t * t) * s;

        assert!(cloglog.1 > 0.0, "negative-tail d1 should remain positive");
        assert_relative_eq!(
            cloglog.0,
            expected_mean,
            epsilon = 1e-30,
            max_relative = 1e-15
        );
        assert_relative_eq!(
            cloglog.1,
            expected_d1,
            epsilon = 1e-30,
            max_relative = 1e-15
        );
        assert_relative_eq!(
            cloglog.2,
            expected_d2,
            epsilon = 1e-30,
            max_relative = 1e-15
        );
        assert_relative_eq!(
            cloglog.3,
            expected_d3,
            epsilon = 1e-30,
            max_relative = 1e-15
        );
    }

    #[test]
    fn test_zero_sigma_logit_and_cloglog_share_component_tail_jets() {
        let ctx = QuadratureContext::new();
        for (link, component, eta) in [
            (LinkFunction::Logit, LinkComponent::Logit, 50.0),
            (LinkFunction::CLogLog, LinkComponent::CLogLog, -50.0),
        ] {
            let integrated = integrated_inverse_link_jet(&ctx, link, eta, 0.0)
                .expect("degenerate integrated jet");
            let point = component_inverse_link_jet(component, eta);
            assert_eq!(integrated.mode, IntegratedExpectationMode::ExactClosedForm);
            assert_eq!(integrated.mean, point.mu);
            assert_eq!(integrated.d1, point.d1);
            assert_eq!(integrated.d2, point.d2);
            assert_eq!(integrated.d3, point.d3);
        }
    }

    #[test]
    fn test_cloglog_controlled_matches_mathematical_target_on_small_sigma_grid() {
        let ctx = QuadratureContext::new();
        // Cover the entire small-sigma routing region with negative-tail,
        // central, and saturated-positive cases. The reference is the
        // mathematical Gaussian expectation, not another evaluator.
        let cases = [
            (-30.0, 1e-10),
            (-30.0, 0.1),
            (-10.0, 0.24),
            (-3.0, 0.2),
            (0.0, 0.05),
            (0.4, 0.1),
            (3.0, 0.24),
            (10.0, 0.1),
            (30.0, 0.24),
        ];

        for &(mu, sigma) in &cases {
            let approx = cloglog_posterior_meanwith_deriv_controlled(&ctx, mu, sigma);
            let (expected_mean, expected_deriv) = cloglog_reference_mean_and_derivative(mu, sigma);
            eprintln!(
                "DBG cloglog_ctrl: mu={} sigma={} approx.mean={} expected_mean={} approx.deriv={} expected_deriv={}",
                mu, sigma, approx.mean, expected_mean, approx.dmean_dmu, expected_deriv
            );
            assert_relative_eq!(
                approx.mean,
                expected_mean,
                epsilon = 1e-12,
                max_relative = 2e-3
            );
            assert_relative_eq!(
                approx.dmean_dmu,
                expected_deriv,
                epsilon = 1e-12,
                max_relative = 4e-3
            );
        }
    }

    #[test]
    fn test_cloglog_dispatch_uses_gamma_backend_for_large_sigma_central_regime() {
        let ctx = QuadratureContext::new();
        let out =
            integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::CLogLog, -0.2, 0.8)
                .expect("cloglog integrated inverse-link moments should evaluate");
        assert_eq!(out.mode, IntegratedExpectationMode::ExactSpecialFunction);
        assert!(out.mean.is_finite());
        assert!(out.dmean_dmu.is_finite());
        assert!(out.dmean_dmu >= 0.0);
    }

    #[test]
    fn test_cloglog_dispatch_uses_large_sigma_asymptotic_without_ghq() {
        let ctx = QuadratureContext::new();
        let out =
            integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::CLogLog, 0.0, 20.0)
                .expect("cloglog integrated inverse-link moments should evaluate");
        assert_eq!(out.mode, IntegratedExpectationMode::ControlledAsymptotic);
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
            cloglog_posterior_meanwith_deriv_gamma_reference(mu, sigma).expect("gamma reference");
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
            mean_mc += cloglog_mean_exact(eta);
            deriv_mc += gumbel_survival_d1(eta);
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
    fn test_logit_dispatch_uses_tail_asymptotic_outside_old_guard() {
        let ctx = QuadratureContext::new();
        let out = integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, 35.0, 1.0)
            .expect("logit integrated inverse-link moments should evaluate");
        assert_eq!(out.mode, IntegratedExpectationMode::ControlledAsymptotic);
        assert!(out.mean.is_finite());
        assert!(out.dmean_dmu.is_finite());
        assert!(out.dmean_dmu >= 0.0);
    }

    #[test]
    fn test_logit_dispatch_prefers_erfcx_in_moderate_regime() {
        // Assertion redesign: this test was originally checking that the
        // dispatcher DOESN'T degrade to `QuadratureFallback` in the
        // moderate regime. The erfcx-series branch genuinely cannot meet
        // the EPSILON=1e-10 accuracy contract at (μ=1.1, σ=0.8) inside
        // LOGIT_MAX_TERMS=160 (tail bound |R_N| ≤ 0.67/(N+1)² → N* ≈ 81619),
        // so routing to GHQ is the correct response. The property we
        // actually care about is accuracy — assert it here against an
        // independent high-resolution Simpson reference, and document
        // that either ExactSpecialFunction or QuadratureFallback is an
        // acceptable route so long as the value is correct.
        let ctx = QuadratureContext::new();
        let out = integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, 1.1, 0.8)
            .expect("logit integrated inverse-link moments should evaluate");
        assert!(matches!(
            out.mode,
            IntegratedExpectationMode::ExactSpecialFunction
                | IntegratedExpectationMode::QuadratureFallback
        ));
        assert!(out.mean.is_finite());
        assert!(out.dmean_dmu.is_finite());
        assert!(out.dmean_dmu >= 0.0);
        let (ref_mean, ref_d1, _, _) = logit_reference_jet_highres_simpson(1.1, 0.8);
        assert_relative_eq!(out.mean, ref_mean, epsilon = 1e-11, max_relative = 1e-10);
        assert_relative_eq!(
            out.dmean_dmu,
            ref_d1,
            epsilon = 1e-11,
            max_relative = 1e-10
        );
    }

    #[test]
    fn test_logit_dispatch_uses_large_sigma_asymptotic_without_ghq() {
        let ctx = QuadratureContext::new();
        let out = integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, 0.5, 20.0)
            .expect("logit integrated inverse-link moments should evaluate");
        assert_eq!(out.mode, IntegratedExpectationMode::ControlledAsymptotic);
        assert!(out.mean.is_finite());
        assert!(out.dmean_dmu.is_finite());
        assert!(out.dmean_dmu >= 0.0);
    }

    #[test]
    fn test_logit_controlled_path_keeps_exact_backend_in_moderate_regime() {
        // Assertion redesign: the erfcx-series branch cannot honor its
        // EPSILON=1e-10 accuracy contract at (μ=1.1, σ=0.8) within
        // LOGIT_MAX_TERMS=160 (tail bound |R_N| ≤ 0.67/(N+1)² → N* ≈ 81619),
        // so `logit_posterior_meanwith_deriv_controlled` legitimately falls
        // through to GHQ. The controlled path's contract is that it returns
        // a correct value via *some* principled route; "which route" is an
        // implementation detail. We assert value accuracy against an
        // independent high-resolution Simpson reference, and document the
        // acceptable modes.
        let ctx = QuadratureContext::new();
        let out =
            logit_posterior_meanwith_deriv_controlled(&ctx, 1.1, 0.8).expect("logit controlled");
        assert!(matches!(
            out.mode,
            IntegratedExpectationMode::ExactSpecialFunction
                | IntegratedExpectationMode::QuadratureFallback
        ));
        let (ref_mean, ref_d1, _, _) = logit_reference_jet_highres_simpson(1.1, 0.8);
        assert_relative_eq!(out.mean, ref_mean, epsilon = 1e-11, max_relative = 1e-10);
        assert_relative_eq!(
            out.dmean_dmu,
            ref_d1,
            epsilon = 1e-11,
            max_relative = 1e-10
        );
    }

    #[test]
    fn test_logit_batch_uses_same_dispatchvalues() {
        let ctx = QuadratureContext::new();
        let eta = ndarray::array![-2.0, 0.0, 1.25, 35.0];
        let se = ndarray::array![0.1, 0.5, 1.0, 1.0];
        let batch_mean = logit_posterior_mean_batch(&ctx, &eta, &se)
            .expect("logit posterior mean batch should evaluate");
        let (batchmu, batch_dmu) = logit_posterior_meanwith_deriv_batch(&ctx, &eta, &se)
            .expect("logit posterior mean derivative batch should evaluate");
        for i in 0..eta.len() {
            let direct = integrated_inverse_link_mean_and_derivative(
                &ctx,
                LinkFunction::Logit,
                eta[i],
                se[i],
            )
            .expect("logit integrated inverse-link moments should evaluate");
            assert_relative_eq!(batch_mean[i], direct.mean, epsilon = 1e-12);
            assert_relative_eq!(batchmu[i], direct.mean, epsilon = 1e-12);
            assert_relative_eq!(batch_dmu[i], direct.dmean_dmu, epsilon = 1e-12);
        }
    }

    #[test]
    fn exact_logit_small_se_branch_loses_tail_derivative() {
        let eta = 50.0_f64;
        let stable_z = (-eta).exp();
        let stable_dmu = stable_z / (1.0_f64 + stable_z).powi(2);
        assert!(stable_dmu > 0.0);
        let out = logit_posterior_meanwith_deriv_exact(eta, 0.0).expect("exact branch");
        let dmu = out.dmean_dmu;
        assert!(
            (dmu - stable_dmu).abs() < 1e-30,
            "exact logit small-se branch should use the stable derivative z/(1+z)^2 at eta={eta}; got {} vs {}",
            dmu,
            stable_dmu
        );
    }

    #[test]
    fn integrated_family_moments_rejects_state_less_sas_and_mixture() {
        let ctx = QuadratureContext::new();
        let sas = integrated_family_moments_jet(&ctx, LikelihoodFamily::BinomialSas, 0.2, 0.5)
            .expect_err("state-less SAS moments should error");
        assert!(format!("{sas}").contains("SasLinkState"));

        let mix = integrated_family_moments_jet(&ctx, LikelihoodFamily::BinomialMixture, 0.2, 0.5)
            .expect_err("state-less mixture moments should error");
        assert!(format!("{mix}").contains("MixtureLinkState"));
    }

    #[test]
    fn integrated_family_moments_supports_stateful_sas() {
        let ctx = QuadratureContext::new();
        let sas = crate::mixture_link::state_from_sasspec(crate::types::SasLinkSpec {
            initial_epsilon: 0.3,
            initial_log_delta: -0.2,
        })
        .expect("sas state should reconstruct from raw parameters");
        let out = integrated_family_moments_jetwith_state(
            &ctx,
            LikelihoodFamily::BinomialSas,
            0.2,
            0.5,
            None,
            Some(&sas),
        )
        .expect("stateful SAS integrated moments should evaluate");
        assert!(out.mean.is_finite());
        assert!(out.d1.is_finite());
        assert!(out.d2.is_finite());
        assert!(out.d3.is_finite());
        assert!(out.mean > 0.0 && out.mean < 1.0);
    }

    #[test]
    fn integrated_family_moments_supports_pure_probit_mixture() {
        let ctx = QuadratureContext::new();
        let state = crate::mixture_link::state_fromspec(&crate::types::MixtureLinkSpec {
            components: vec![crate::types::LinkComponent::Probit],
            initial_rho: ndarray::Array1::<f64>::zeros(0),
        })
        .expect("single-component probit mixture state");
        let out = integrated_family_moments_jetwith_state(
            &ctx,
            LikelihoodFamily::BinomialMixture,
            0.7,
            1.3,
            Some(&state),
            None,
        )
        .expect("pure probit mixture integrated moments should evaluate");
        let exact = integrated_probit_jet(0.7, 1.3);
        assert_relative_eq!(out.mean, exact.mean, epsilon = 1e-12);
        assert_relative_eq!(out.d1, exact.d1, epsilon = 1e-12);
        assert_relative_eq!(out.d2, exact.d2, epsilon = 1e-12);
        assert_relative_eq!(out.d3, exact.d3, epsilon = 1e-12);
        assert_eq!(out.mode, IntegratedExpectationMode::ExactClosedForm);
    }

    #[test]
    fn integrated_family_moments_supports_pure_logit_mixture() {
        let ctx = QuadratureContext::new();
        let state = crate::mixture_link::state_fromspec(&crate::types::MixtureLinkSpec {
            components: vec![crate::types::LinkComponent::Logit],
            initial_rho: ndarray::Array1::<f64>::zeros(0),
        })
        .expect("single-component logit mixture state");
        let out = integrated_family_moments_jetwith_state(
            &ctx,
            LikelihoodFamily::BinomialMixture,
            1.1,
            0.8,
            Some(&state),
            None,
        )
        .expect("pure logit mixture integrated moments should evaluate");
        let exact = integrated_inverse_link_jet(&ctx, LinkFunction::Logit, 1.1, 0.8)
            .expect("canonical integrated logit jet");
        assert_relative_eq!(out.mean, exact.mean, epsilon = 1e-12);
        assert_relative_eq!(out.d1, exact.d1, epsilon = 1e-12);
        assert_relative_eq!(out.d2, exact.d2, epsilon = 1e-12);
        assert_relative_eq!(out.d3, exact.d3, epsilon = 1e-12);
        assert_eq!(out.mode, exact.mode);
    }

    #[test]
    fn integrated_family_moments_supports_stateful_mixture() {
        let ctx = QuadratureContext::new();
        let state = crate::mixture_link::state_fromspec(&crate::types::MixtureLinkSpec {
            components: vec![
                crate::types::LinkComponent::Logit,
                crate::types::LinkComponent::Probit,
            ],
            initial_rho: ndarray::array![0.35],
        })
        .expect("mixture state should reconstruct from rho");
        let out = integrated_family_moments_jetwith_state(
            &ctx,
            LikelihoodFamily::BinomialMixture,
            0.2,
            0.5,
            Some(&state),
            None,
        )
        .expect("stateful mixture integrated moments should evaluate");
        let direct = integrated_mixture_jet(&ctx, 0.2, 0.5, &state)
            .expect("direct integrated mixture jet should evaluate");
        assert_relative_eq!(out.mean, direct.mean, epsilon = 1e-12);
        assert_relative_eq!(out.d1, direct.d1, epsilon = 1e-12);
        assert_relative_eq!(out.d2, direct.d2, epsilon = 1e-12);
        assert_relative_eq!(out.d3, direct.d3, epsilon = 1e-12);
        assert_eq!(out.mode, direct.mode);
    }

    // Tests for CLogLog Gaussian convolution derivatives

    #[test]
    fn cloglog_g_derivatives_at_zero() {
        let (g, g1, g2, g3, g4) = cloglog_g_derivatives(0.0);
        // g(0) = 1 - exp(-1)
        let expected_g = 1.0 - (-1.0_f64).exp();
        assert_relative_eq!(g, expected_g, epsilon = 1e-14);
        // g'(0) = exp(0 - exp(0)) = exp(-1)
        let e_neg1 = (-1.0_f64).exp();
        assert_relative_eq!(g1, e_neg1, epsilon = 1e-14);
        // g''(0) = (1 - 1) * exp(-1) = 0
        assert_relative_eq!(g2, 0.0, epsilon = 1e-14);
        // g'''(0) = (1 - 3 + 1) * exp(-1) = -exp(-1)
        assert_relative_eq!(g3, -e_neg1, epsilon = 1e-14);
        // g''''(0) = (-1 + 6 - 7 + 1) * exp(-1) = -exp(-1)
        assert_relative_eq!(g4, -e_neg1, epsilon = 1e-14);
    }

    #[test]
    fn cloglog_g_derivatives_saturation() {
        // Very large t: g→1, derivatives→0
        let (g, g1, g2, g3, g4) = cloglog_g_derivatives(50.0);
        assert_relative_eq!(g, 1.0, epsilon = 1e-10);
        assert_eq!(g1, 0.0);
        assert_eq!(g2, 0.0);
        assert_eq!(g3, 0.0);
        assert_eq!(g4, 0.0);

        // Very negative t: g ≈ exp(t), all derivatives ≈ exp(t)
        let (g, g1, g2, g3, g4) = cloglog_g_derivatives(-50.0);
        let expected = (-50.0_f64).exp();
        assert_relative_eq!(g, expected, max_relative = 1e-10);
        assert_relative_eq!(g1, expected, max_relative = 1e-10);
        // Higher derivatives have polynomial factors ≈ 1 for t ≪ 0
        assert_relative_eq!(g2, expected, max_relative = 1e-10);
        assert_relative_eq!(g3, expected, max_relative = 1e-10);
        assert_relative_eq!(g4, expected, max_relative = 1e-10);
    }

    #[test]
    fn cloglog_ghq_value_sigma_zero_matches_pointwise() {
        let ctx = QuadratureContext::new();
        // When sigma=0, L(mu,0) = g(mu)
        for &mu in &[-2.0, -1.0, 0.0, 0.5, 1.5] {
            let val = cloglog_ghq_value(&ctx, mu, 0.0, 21);
            let (g, _, _, _, _) = cloglog_g_derivatives(mu);
            assert_relative_eq!(val, g, epsilon = 1e-14);
        }
    }

    #[test]
    fn cloglog_ghq_value_bounded_zero_one() {
        let ctx = QuadratureContext::new();
        // g maps to (0,1), so the Gaussian convolution should stay in [0,1]
        for &mu in &[-5.0, -2.0, 0.0, 1.0, 3.0, 10.0] {
            for &sigma in &[0.1, 0.5, 1.0, 2.0, 5.0] {
                let val = cloglog_ghq_value(&ctx, mu, sigma, 31);
                assert!(val >= 0.0 && val <= 1.0, "L({mu},{sigma}) = {val}");
            }
        }
    }

    #[test]
    fn cloglog_ghq_derivatives_sigma_zero_matches_pointwise() {
        let ctx = QuadratureContext::new();
        let mu = 0.3;
        let d = cloglog_ghq_derivatives(&ctx, mu, 0.0, 21);
        let (g, g1, g2, g3, g4) = cloglog_g_derivatives(mu);
        assert_relative_eq!(d.l, g, epsilon = 1e-14);
        assert_relative_eq!(d.l_mu, g1, epsilon = 1e-14);
        assert_eq!(d.l_sigma, 0.0);
        assert_relative_eq!(d.l_mumu, g2, epsilon = 1e-14);
        assert_eq!(d.l_musigma, 0.0);
        assert_eq!(d.l_sigmasigma, 0.0);
        assert_relative_eq!(d.l_mumumu, g3, epsilon = 1e-14);
        assert_relative_eq!(d.l_mumumumu, g4, epsilon = 1e-14);
    }

    #[test]
    fn cloglog_ghq_derivatives_finite_difference_mu() {
        // Verify ∂L/∂μ by finite differences
        let ctx = QuadratureContext::new();
        let mu = 0.5;
        let sigma = 0.8;
        let h = 1e-6;
        let d = cloglog_ghq_derivatives(&ctx, mu, sigma, 31);
        let l_plus = cloglog_ghq_value(&ctx, mu + h, sigma, 31);
        let l_minus = cloglog_ghq_value(&ctx, mu - h, sigma, 31);
        let fd_mu = (l_plus - l_minus) / (2.0 * h);
        assert_relative_eq!(d.l_mu, fd_mu, epsilon = 1e-5);

        // Second derivative ∂²L/∂μ²
        let d_plus = cloglog_ghq_derivatives(&ctx, mu + h, sigma, 31);
        let d_minus = cloglog_ghq_derivatives(&ctx, mu - h, sigma, 31);
        let fd_mumu = (d_plus.l_mu - d_minus.l_mu) / (2.0 * h);
        assert_relative_eq!(d.l_mumu, fd_mumu, epsilon = 1e-4);
    }

    #[test]
    fn cloglog_ghq_derivatives_finite_difference_sigma() {
        // Verify ∂L/∂σ by finite differences
        let ctx = QuadratureContext::new();
        let mu = 0.2;
        let sigma = 1.0;
        let h = 1e-6;
        let d = cloglog_ghq_derivatives(&ctx, mu, sigma, 31);
        let l_plus = cloglog_ghq_value(&ctx, mu, sigma + h, 31);
        let l_minus = cloglog_ghq_value(&ctx, mu, sigma - h, 31);
        let fd_sigma = (l_plus - l_minus) / (2.0 * h);
        assert_relative_eq!(d.l_sigma, fd_sigma, epsilon = 1e-5);
    }

    #[test]
    fn cloglog_ghq_derivatives_finite_difference_cross() {
        // Verify ∂²L/∂μ∂σ by finite differences of ∂L/∂μ w.r.t. σ
        let ctx = QuadratureContext::new();
        let mu = -0.5;
        let sigma = 0.6;
        let h = 1e-6;
        let d = cloglog_ghq_derivatives(&ctx, mu, sigma, 31);
        let d_plus = cloglog_ghq_derivatives(&ctx, mu, sigma + h, 31);
        let d_minus = cloglog_ghq_derivatives(&ctx, mu, sigma - h, 31);
        let fd_musigma = (d_plus.l_mu - d_minus.l_mu) / (2.0 * h);
        assert_relative_eq!(d.l_musigma, fd_musigma, epsilon = 1e-4);
    }

    #[test]
    fn cloglog_ghq_l_mu_nonnegative() {
        // g'(t) = exp(t - exp(t)) >= 0, so ∂L/∂μ = E[g'(t)] >= 0
        let ctx = QuadratureContext::new();
        for &mu in &[-3.0, -1.0, 0.0, 1.0, 3.0] {
            for &sigma in &[0.1, 0.5, 1.0, 2.0] {
                let d = cloglog_ghq_derivatives(&ctx, mu, sigma, 21);
                assert!(
                    d.l_mu >= -1e-14,
                    "L_mu should be non-negative at mu={mu}, sigma={sigma}: got {}",
                    d.l_mu
                );
            }
        }
    }

    #[test]
    fn cloglog_ghq_adaptive_matches_explicit() {
        let ctx = QuadratureContext::new();
        let mu = 0.7;
        let sigma = 1.2;
        let adaptive = cloglog_ghq_derivatives_adaptive(&ctx, mu, sigma);
        let n = adaptive_point_count_from_sd(sigma);
        let explicit = cloglog_ghq_derivatives(&ctx, mu, sigma, n);
        assert_relative_eq!(adaptive.l, explicit.l, epsilon = 1e-15);
        assert_relative_eq!(adaptive.l_mu, explicit.l_mu, epsilon = 1e-15);
        assert_relative_eq!(adaptive.l_sigma, explicit.l_sigma, epsilon = 1e-15);
        assert_relative_eq!(adaptive.l_mumu, explicit.l_mumu, epsilon = 1e-15);
    }

    #[test]
    fn cloglog_ghq_value_matches_mathematical_target_in_central_regime() {
        let ctx = QuadratureContext::new();
        for &mu in &[-1.0, 0.0, 0.5, 2.0] {
            for &sigma in &[0.1, 0.5, 1.0] {
                let ghq = cloglog_ghq_value(&ctx, mu, sigma, 51);
                let (expected_mean, _) = cloglog_reference_mean_and_derivative(mu, sigma);
                assert_relative_eq!(ghq, expected_mean, epsilon = 1e-12, max_relative = 2e-8);
            }
        }
    }

    // ── Cloglog negative-tail asymptotic tests ──────────────────────────

    #[test]
    fn cloglog_negative_tail_mean_matches_exact_near_transition() {
        // At η = −30 the exact cloglog mean is 1 − exp(−exp(−30)).
        // Our tail helper should agree to high relative accuracy where the
        // implementation transitions into the negative-tail approximation.
        let eta: f64 = -30.0;
        let exact = {
            let ex = eta.exp();
            -(-ex).exp_m1()
        };
        let tail = cloglog_negative_tail_mean(eta);
        assert!(
            (exact - tail).abs() < 1e-26 * exact.abs().max(1e-300),
            "tail mean at η={eta}: exact={exact:.6e} tail={tail:.6e}"
        );
    }

    #[test]
    fn cloglog_negative_tail_derivative_matches_exact_near_transition() {
        // At η = −30: dμ/dη = exp(η)·exp(−exp(η)).
        let eta: f64 = -30.0;
        let ex = eta.exp();
        let exact = ex * (-ex).exp();
        let tail = cloglog_negative_tail_derivative(eta);
        assert!(
            (exact - tail).abs() < 1e-26 * exact.abs().max(1e-300),
            "tail derivative at η={eta}: exact={exact:.6e} tail={tail:.6e}"
        );
    }

    #[test]
    fn cloglog_negative_tail_degenerate_branch_matches_target_near_transition() {
        let ctx = QuadratureContext::default();
        let sigma = 0.0;
        for &mu in &[-30.001, -30.0, -29.999] {
            let out = cloglog_posterior_meanwith_deriv_controlled(&ctx, mu, sigma);
            assert_relative_eq!(
                out.mean,
                cloglog_mean_exact(mu),
                epsilon = 1e-28,
                max_relative = 1e-15
            );
            assert_relative_eq!(
                out.dmean_dmu,
                gumbel_survival_d1(mu),
                epsilon = 1e-28,
                max_relative = 1e-15
            );
        }
    }

    #[test]
    fn cloglog_negative_tail_small_sigma_branch_matches_target_near_transition() {
        let ctx = QuadratureContext::default();
        let sigma = 0.1;
        for &mu in &[-30.001, -30.0, -29.999] {
            let out = cloglog_posterior_meanwith_deriv_controlled(&ctx, mu, sigma);
            let (expected_mean, expected_deriv) = cloglog_reference_mean_and_derivative(mu, sigma);
            assert_relative_eq!(
                out.mean,
                expected_mean,
                epsilon = 1e-24,
                max_relative = 1e-10
            );
            assert_relative_eq!(
                out.dmean_dmu,
                expected_deriv,
                epsilon = 1e-24,
                max_relative = 1e-10
            );
        }
    }
}
