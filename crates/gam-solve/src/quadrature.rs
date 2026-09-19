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
//!    mean.
//!
//! 3. Cloglog / survival transforms:
//!    The complementary log-log mean under Gaussian eta does not simplify to an
//!    elementary closed form, but it does admit exact non-GHQ representations:
//!    - as the Laplace transform of a lognormal variable,
//!    - as characteristic-function inversion with Gamma(1 - i t),
//!    - or as rapidly convergent erfc / asymptotic series on subdomains.
//!    These are the natural replacements for repeated GHQ calls in
//!    cloglog and survival-specific cubature paths.
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
use std::sync::{Arc, Mutex, OnceLock};

use crate::estimate::EstimationError;
use crate::mixture_link::{
    beta_logistic_inverse_link_jet, component_inverse_link_jet, sas_inverse_link_jet,
};
use gam_math::probability::erfcx_nonnegative;
use gam_math::quadrature::{GaussHermiteRule, gauss_hermite_rule};
use gam_math::special::stable_polynomial_times_exp_neg as cloglog_stable_poly_times_exp_neg;
use gam_problem::types::{
    GlmLikelihoodSpec, InverseLink, LinkComponent, LinkFunction, MixtureLinkState, ResponseFamily,
    SasLinkState, StandardLink,
};
/// Number of quadrature points (7-point rule is exact for polynomials up to degree 13)
const N_POINTS: usize = 7;
const SQRT_2: f64 = std::f64::consts::SQRT_2;
const QUADRATURE_EXP_LOG_MAX: f64 = 700.0;

// Convention: finite moments saturate exp arguments at 700 and use ControlledAsymptotic mode on saturation.
// Probability/tail kernels stay in log-space or bounded envelopes so overflow cannot turn finite targets into NaN.
#[inline]
fn safe_exp(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else {
        x.min(QUADRATURE_EXP_LOG_MAX).exp()
    }
}

#[inline]
fn safe_expwith_saturation(x: f64) -> (f64, bool) {
    (safe_exp(x), x > QUADRATURE_EXP_LOG_MAX)
}

/// Quadrature context that owns Gauss-Hermite caches.
pub struct QuadratureContext {
    gh_cache: OnceLock<GaussHermiteRule>,
    gh15_cache: OnceLock<GaussHermiteRule>,
    gh21_cache: OnceLock<GaussHermiteRule>,
    gh31_cache: OnceLock<GaussHermiteRule>,
    gh51_cache: OnceLock<GaussHermiteRule>,
    // Clenshaw-Curtis rules are constructed on demand because the node count is
    // chosen from the certified truncation/ellipse heuristic rather than from a
    // tiny fixed family like the GHQ rules above.
    cc_cache: Mutex<HashMap<usize, Arc<ClenshawCurtisRule>>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum IntegratedExpectationMode {
    ExactClosedForm,
    ExactSpecialFunction,
    ControlledAsymptotic,
    QuadratureFallback,
}

impl IntegratedExpectationMode {
    /// Ordinal rank where higher = lower-fidelity / further from exact closed
    /// form. Lets callers fold over a stream of modes and keep the *worst*
    /// one with `a.rank().max(b.rank())`.
    #[inline]
    pub const fn rank(self) -> u8 {
        match self {
            Self::ExactClosedForm => 0,
            Self::ExactSpecialFunction => 1,
            Self::ControlledAsymptotic => 2,
            Self::QuadratureFallback => 3,
        }
    }
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
        crate::bail_invalid_estim!(
            "latent cloglog jet requires finite eta and sigma >= 0, got eta={eta}, sigma={sigma}"
        );
    }
    Ok::<(), _>(())
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

/// Latent SD above which the logistic-normal *jet* stops trusting Gauss–Hermite
/// quadrature. The jet integrands are the localized inverse-link derivatives
/// `sigmoid^(k)` (bumps of characteristic width O(1) in η, hence width O(1/σ) in
/// the standardized GH coordinate). Once σ grows past ~1, GH can no longer
/// resolve the higher derivatives. Measured 31/51-node GH relative error vs a
/// 16384-interval Simpson reference (μ≈σ) shows the knee precisely:
///
/// ```text
///   σ     d1        d2        d3
///   0.8   2.8e-16   5.4e-13   2.1e-12
///   1.0   4.7e-12   1.4e-10   2.1e-9     ← still excellent
///   1.2   4.8e-10   4.5e-9    1.9e-7
///   1.5   4.8e-8    5.9e-8    1.8e-5
///   2.5   6.9e-5    9.2e-4    3.0e-2
///   5.0   1.7e-3    7.8e-2    2.1e+0     ← d3 209% wrong
/// ```
///
/// Adaptive Simpson, by contrast, holds ~1e-12 on every component at every σ.
/// So at σ ≤ 1 GH is both accurate (≤ ~2e-9 on all four components) and cheap
/// (31 nodes); beyond σ = 1 the jet is integrated by adaptive Simpson instead,
/// with `mean`/`d1` reused verbatim from the scalar controlled backend so the
/// scalar dispatcher and the jet agree by construction (#571 — the GH jet used
/// to drift ~4e-3 from the scalar value at (μ=3, σ=3)).
const LOGIT_JET_GHQ_SIGMA_MAX: f64 = 1.0;
const CLOGLOG_SIGMA_DEGENERATE: f64 = 1e-10;
/// Latent SD above which the cloglog integrated jet stops trusting shifted
/// lognormal-Laplace moment reconstruction for higher derivatives. The moments
/// `E[u^m exp(-u)]`, `u=exp(eta)`, evaluate the survival term at
/// `mu + m*sigma^2`; for d3 at sigma=4 that asks for a shift of 48 and loses
/// the small k3 contribution to cancellation. Directly integrating the stable
/// pointwise derivatives keeps the location-family jet identity intact.
const CLOGLOG_JET_MOMENT_SIGMA_MAX: f64 = 1.0;
// ── Log-space survival panel (#2714) ────────────────────────────────────────
//
// `ln S(μ,σ)`, `S(μ,σ) = E[exp(−e^η)]`, `η ~ N(μ,σ²)`, is evaluated on ONE
// Laplace-localized Clenshaw–Curtis panel in the standardized variable
// `z = (η−μ)/σ`, accumulated by log-sum-exp. The panel is placed from the
// integrand's own geometry rather than from a fixed truncation, which is what
// makes a single rule accurate over the whole `(μ,σ)` plane.
//
// The log-integrand of the survival branch,
//
// ```text
//   L(z) = −z²/2 − ln√(2π) − e^{μ+σz},
//   L''(z) = −(1 + σ² e^{μ+σz}) ≤ −1,
// ```
//
// is STRICTLY concave, so it is unimodal, its maximizer is the unique root of a
// monotone equation, and the two points where it falls a requested number of
// e-folds below that maximum bracket every part of the integral that can matter
// at f64 precision. The complement branch `1 − S = E[1−exp(−e^η)]` has the same
// structure (`ln(1−e^{−u})` is concave in `z`), and is used where `S → 1` so
// `ln S` stays RELATIVELY accurate instead of merely absolutely accurate.
//
// What this replaces, and why it had to be replaced rather than re-thresholded:
// `ln S` used to be assembled from a value-space ladder (Taylor / extreme
// asymptotic / Miles erfc-series / CC / Gamma / GHQ) and then logged, with a
// fixed-window Gumbel-mixing quadrature on `η ∈ [−40, 6]` as an underflow
// escape hatch. Graded against a 60-digit reference on
// `μ ∈ [−20, 12] × σ ∈ [0.002, 60]`, the worst absolute error of that composite
// in `ln S` was **4.4e+06**, and the three worst rows came off three DIFFERENT
// routes — the escape hatch (4.4e+06 at `(12, 0.002)`, its `Φ((η−μ)/σ)`
// transition unresolved by a pinned 513-node grid), the Miles/CC value route
// (82.0 at `(12, 1.0)`), and the rare-event asymptotic (`ln1p(−e^{μ+σ²/2})`,
// 20.8× wrong at `(−50, 8)`, where `rare_log = −18` is on the gate but the
// higher cumulants are not small). No threshold between those routes can work,
// because no two of them are accurate on either side of a common cut; and an
// analytic derivative cannot be the derivative of a surface whose error is a
// step function of `(μ,σ)`, which is the #2714 stall.
//
// The panel rule's worst absolute error in `ln S` on the same grid is
// **9.3e-10** — at `ln S = −5.7e6`, i.e. `1.6e-16` relative, the f64
// representation floor of the answer itself.
//
/// Number of e-folds below its own maximum at which the log-integrand is cut.
///
/// The neglected tails are bounded by (local scale) × `exp(−DROP)` relative to
/// the peak; at 60 that is `8.8e-27`, ten orders below the `~1e-16` relative
/// accuracy the rule targets, so the truncation is never the binding error.
/// Enlarging it is cheap (the panel grows like `√DROP`), which is why the
/// margin is taken here rather than defended.
const LOG_SURVIVAL_PANEL_LOG_DROP: f64 = 60.0;
/// Extra e-folds per μ-derivative order, in units of `ln(2 + |z⋆|)`.
///
/// Order `j` multiplies the integrand by `He_j(z)`, which grows like `|z|^j`, so
/// the point where the product falls `DROP` e-folds below its peak moves
/// outward by `j·ln|z|`. Padding the cut by exactly that keeps the truncation
/// bound of the tower equal to the truncation bound of the value.
const LOG_SURVIVAL_PANEL_ORDER_LOG_DROP: f64 = 2.0;
/// Clenshaw–Curtis nodes per unit of local-scale arclength
/// `T = ∫ √(1 + σ² e^{μ+σz}) dz` across the panel.
///
/// `T` counts how many local scales the panel spans — the resolution the rule
/// needs on the real axis. Measured node counts for `1e-13` relative accuracy
/// over `μ ∈ [−30, 20] × σ ∈ [5e-4, 200]` never exceed `4.6·T` once the σ term
/// below is added; the observed minimum sufficient ratio at small σ is 2.2, so
/// this carries a 2.1× margin.
const LOG_SURVIVAL_PANEL_ARCLENGTH_NODE_DENSITY: f64 = 4.6;
/// Clenshaw–Curtis nodes per `√σ`.
///
/// `T` alone does not price large σ: `exp(−e^{μ+σz})` is entire but its
/// modulus grows off the real axis with period `2π/σ`, so the Bernstein
/// ellipse on which the interpolant converges shrinks as σ grows and the node
/// count rises even though the panel does not. Measured requirement at `1e-13`:
/// `n = 65, 97, 193, 385, 769` at `σ = 0.5, 2, 8, 60, 200`, i.e. `n − 65` is
/// `≈ 50·√σ`; 70 carries a 1.4× margin on the fitted slope.
const LOG_SURVIVAL_PANEL_SIGMA_NODE_SCALE: f64 = 70.0;
/// Extra nodes per μ-derivative order (the `He_j` factor raises the polynomial
/// degree of the integrand by `j`, and the panel widens with `j`).
const LOG_SURVIVAL_PANEL_ORDER_NODES: f64 = 8.0;
/// Node floor. The panel spans ~20 local scales at small σ and needs 49–65
/// nodes there; the floor is the measured maximum of that regime.
const LOG_SURVIVAL_PANEL_MIN_NODES: usize = 65;
/// Node ceiling. `4.6·T + 70·√σ` reaches this at `σ ≈ 3.3e3`; beyond that the
/// rule is capped and reports its own conditioning, which is what the
/// derivative-tower gate reads.
const LOG_SURVIVAL_PANEL_MAX_NODES: usize = 4097;
/// Highest μ-derivative order the panel tower will produce.
///
/// `log_kernel_bundle` asks for `k + 4` rungs at most, so 8 covers every
/// shipped consumer with room; the bound exists so the tower is a fixed-size
/// array rather than a heap allocation on a per-row path.
pub(crate) const LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER: usize = 8;
/// Largest tolerated cancellation, as `ln(Σ|terms| / |Σ terms|)`, before the
/// direct Hermite μ-derivative tower is refused.
///
/// The tower is a signed log-sum-exp, so its relative error is
/// `≈ ε · Σ|terms|/|Σ terms|`; the quadrature therefore MEASURES its own
/// conditioning and does not have to be gated on a proxy for it.
///
/// The value is the solve of `ε · e^cond = 1e-13`, i.e. `ln(1e-13/ε) = 6.1`:
/// admit the tower exactly while it is at the working floor of everything
/// around it, so switching bases can never LOSE accuracy relative to the rung
/// basis it displaces. Verified on `μ ∈ [−20, 12] × σ ∈ [0.002, 60] ×
/// `j ∈ [1, 4]` against a 60-digit reference: the achieved error tracks
/// `ε·e^cond` as the model says (`≤ 1.1e-7` below 20, `≤ 1.8e-5` below 25), and
/// every admitted row lands at `2.9e-11` or better — which is itself the f64
/// ulp of the largest `ln|·|` on the grid, not a quadrature error.
///
/// This replaces a `σ ≥ 8` gate, which asked a chosen constant to stand in for
/// exactly this quantity. The replacement is a strict superset of it: all 132
/// grid rows with `σ ≥ 8` are admitted here, plus the moderate-σ / positive-μ
/// region where the tower is equally well conditioned and the rung basis was
/// being used only because `σ < 8`.
const LOG_SURVIVAL_TOWER_MAX_LOG_CANCELLATION: f64 = 6.1;

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

    fn gauss_hermite_n(&self, n: usize) -> &GaussHermiteRule {
        match n {
            // The fixed 7-point cache is served via `gauss_hermite()`. If a caller
            // ends up here with n=7 anyway, fall back to the 15-point rule.
            7 => self.gh15_cache.get_or_init(|| compute_gauss_hermite_n(15)),
            15 => self.gh15_cache.get_or_init(|| compute_gauss_hermite_n(15)),
            21 => self.gh21_cache.get_or_init(|| compute_gauss_hermite_n(21)),
            31 => self.gh31_cache.get_or_init(|| compute_gauss_hermite_n(31)),
            51 => self.gh51_cache.get_or_init(|| compute_gauss_hermite_n(51)),
            _ => self.gh21_cache.get_or_init(|| compute_gauss_hermite_n(21)),
        }
    }

    fn clenshaw_curtis_n(&self, n: usize) -> Arc<ClenshawCurtisRule> {
        let mut cache = match self.cc_cache.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        cache
            .entry(n)
            .or_insert_with(|| Arc::new(compute_clenshaw_curtis_n(n)))
            .clone()
    }
}

impl Default for QuadratureContext {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
struct ClenshawCurtisRule {
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

fn compute_clenshaw_curtis_n(n: usize) -> ClenshawCurtisRule {
    assert!(
        n >= 2,
        "Clenshaw-Curtis rule requires at least two nodes: n={n}"
    );
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

/// Fixed production rule built by the shared `O(n²)`-time, `O(n)`-space
/// Golub-Welsch implementation.
fn compute_gauss_hermite() -> GaussHermiteRule {
    compute_gauss_hermite_n(N_POINTS)
}

pub(crate) fn compute_gauss_hermite_n(n: usize) -> GaussHermiteRule {
    // SAFETY: every production caller selects n from the positive, hard-coded
    // GHQ family {7,15,21,31,51}. A failure here means the shared deterministic
    // eigensolver violated that construction invariant; no alternate numerical
    // rule has been certified for the caller to recover with.
    gauss_hermite_rule(n)
        .unwrap_or_else(|error| panic!("shared Gauss-Hermite construction failed: {error}"))
}

/// Computes the integrated probability AND its derivative with respect to eta.
///
/// For IRLS, we need both:
/// - μ = ∫ σ(η) × N(η; m, SE²) dη
/// - dμ/dm = ∫ σ'(η) × N(η; m, SE²) dη = ∫ σ(η)(1-σ(η)) × N(η; m, SE²) dη
///
/// Both come from one pass of the accelerated moment series in
/// [`logit_posterior_meanwith_deriv_exact`], accurate to working precision at
/// every finite `(eta, se_eta)`.
///
/// Returns: (μ, dμ/dm)
#[inline]
pub fn logit_posterior_meanwith_deriv(
    eta: f64,
    se_eta: f64,
) -> Result<(f64, f64), EstimationError> {
    let out = logit_posterior_meanwith_deriv_exact(eta, se_eta)?;
    Ok((out.mean, out.dmean_dmu))
}

#[inline]
pub(crate) fn probit_posterior_meanwith_deriv_exact(mu: f64, sigma: f64) -> IntegratedMeanDerivative {
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
    if !(mu.is_finite() && sigma.is_finite()) {
        let mean = gam_math::probability::normal_cdf(mu);
        let dmean_dmu = gam_math::probability::normal_pdf(mu);
        return IntegratedMeanDerivative {
            mean,
            dmean_dmu,
            mode: IntegratedExpectationMode::ExactClosedForm,
        };
    }
    let denom = (1.0 + sigma * sigma).sqrt();
    let z = mu / denom;
    IntegratedMeanDerivative {
        mean: gam_math::probability::normal_cdf(z),
        dmean_dmu: gam_math::probability::normal_pdf(z) / denom,
        mode: IntegratedExpectationMode::ExactClosedForm,
    }
}

#[inline]
fn stable_sigmoidwith_derivative(x: f64) -> (f64, f64) {
    let x_clamped = x.clamp(-QUADRATURE_EXP_LOG_MAX, QUADRATURE_EXP_LOG_MAX);
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

// ── Logistic-normal integral (F7) ───────────────────────────────────────────
//
// `E[σ(η)]` and `E[σ'(η)]`, `η ~ N(μ, s²)`, from ONE accelerated series whose
// error is bounded a priori, uniformly over `(μ, s)`.
//
// Split the real line at the kink-free point `η = 0` and write `t = e^{−|η|}`,
// so `t ∈ (0, 1]` on both halves. There the logistic and its slope are
//
// ```text
//   η ≤ 0:  σ(η)  = t/(1+t)       η > 0:  σ(η) = 1 − t/(1+t)
//   both:   σ'(η) = t/(1+t)²
// ```
//
// and every Gaussian expectation of a power `t^j` on a half-line is closed form:
//
// ```text
//   m_j⁻(μ) = E[e^{jη}; η ≤ 0] = e^{jμ + j²s²/2} · Φ(−(μ + j s²)/s),
//   m_j⁺(μ) = E[e^{−jη}; η > 0] = m_j⁻(−μ).
// ```
//
// The Cohen–Rodriguez Villegas–Zagier weights `w_k` (Experimental Math. 9,
// 2000, Algorithm 1) define a degree-`n` polynomial `Q(t) = Σ w_k t^k` with
//
// ```text
//   1/(1+t) − Q(t) = P_n(t) / (T_n(3) (1+t)),   |P_n| ≤ 1 on [0, 1],
// ```
//
// `T_n(3) = cosh(n·ln(3+√8))` the Chebyshev polynomial. Integrating against
// the POSITIVE measures of `t` on each half-line gives
//
// ```text
//   E[σ(η)]  = Φ(μ/s) − Σ_k w_k m⁺_{k+1} + Σ_k w_k m⁻_{k+1},
//   E[σ'(η)] = −Σ_{k≥1} k w_k (m⁺_k + m⁻_k)          (t/(1+t)² = −t·d/dt 1/(1+t)),
// ```
//
// with RELATIVE errors at most `1/T_n(3)` for the mean (both truncated sums are
// bounded by the mean itself, since `t/(1+t) ≤ σ` pointwise) and
// `(4n²+1)/T_n(3)` for the slope (Markov's inequality `|P_n'| ≤ 2n²` on the
// unit interval). Both are pointwise-in-`t` bounds, so they hold for every
// `(μ, s)` — there is no regime split, no eligibility window, and nothing for a
// second evaluator to cross-check. The series this replaces (a heat-kernel /
// erfcx / tail-asymptotic ladder behind an adaptive-Simpson drift check) spent
// ~70 µs per row on the check alone.

/// Geometric convergence rate `3 + √8` of the CRVZ acceleration: the degree-`n`
/// error denominator is `T_n(3) = ((3+√8)^n + (3+√8)^{−n}) / 2`.
const LOGISTIC_NORMAL_CRVZ_RATE: f64 = 3.0 + 2.0 * SQRT_2;

/// Smallest degree whose a-priori slope bound `(4n²+1)/T_n(3)` (which also
/// dominates the mean bound `1/T_n(3)`) is at or below the unit roundoff.
const LOGISTIC_NORMAL_SERIES_TERMS: usize = logistic_normal_series_terms();

const fn logistic_normal_series_terms() -> usize {
    let unit_roundoff = 0.5 * f64::EPSILON;
    let mut n = 1usize;
    let mut rate_pow = LOGISTIC_NORMAL_CRVZ_RATE;
    loop {
        let chebyshev = 0.5 * (rate_pow + 1.0 / rate_pow);
        let nf = n as f64;
        if (4.0 * nf * nf + 1.0) / chebyshev <= unit_roundoff {
            return n;
        }
        n += 1;
        rate_pow *= LOGISTIC_NORMAL_CRVZ_RATE;
    }
}

/// CRVZ weights `w_k`, plus the suffix sums `Σ_{i≥k} |w_i|` and
/// `Σ_{i≥k} i·|w_i|` that bound the part of either series not yet summed.
struct LogisticNormalSeriesWeights {
    weight: [f64; LOGISTIC_NORMAL_SERIES_TERMS],
    abs_tail: [f64; LOGISTIC_NORMAL_SERIES_TERMS + 1],
    index_abs_tail: [f64; LOGISTIC_NORMAL_SERIES_TERMS + 1],
}

const LOGISTIC_NORMAL_SERIES_WEIGHTS: LogisticNormalSeriesWeights =
    logistic_normal_series_weights();

const fn logistic_normal_series_weights() -> LogisticNormalSeriesWeights {
    const N: usize = LOGISTIC_NORMAL_SERIES_TERMS;
    let nf = N as f64;
    let mut d = 1.0;
    let mut i = 0;
    while i < N {
        d *= LOGISTIC_NORMAL_CRVZ_RATE;
        i += 1;
    }
    d = 0.5 * (d + 1.0 / d);
    let mut b = -1.0;
    let mut c = -d;
    let mut weight = [0.0; N];
    let mut k = 0;
    while k < N {
        c = b - c;
        weight[k] = c / d;
        let kf = k as f64;
        b = (kf + nf) * (kf - nf) * b / ((kf + 0.5) * (kf + 1.0));
        k += 1;
    }
    let mut abs_tail = [0.0; N + 1];
    let mut index_abs_tail = [0.0; N + 1];
    let mut k = N;
    while k > 0 {
        k -= 1;
        abs_tail[k] = abs_tail[k + 1] + weight[k].abs();
        index_abs_tail[k] = index_abs_tail[k + 1] + (k as f64) * weight[k].abs();
    }
    LogisticNormalSeriesWeights {
        weight,
        abs_tail,
        index_abs_tail,
    }
}

/// Half-line lognormal moment `m_j⁻(μ) = E[e^{jη}; η ≤ 0]`, `η ~ N(μ, s²)`.
///
/// With `x = (μ + j s²)/(√2 s)`, `m = ½ e^{jμ + j²s²/2} erfc(x)`. For `x ≥ 0`
/// the exponent collapses exactly, `jμ + j²s²/2 − x² = −μ²/(2s²)`, so the
/// moment is `½ e^{−μ²/(2s²)} erfcx(x)`: no overflow and full relative
/// accuracy however deep the tail. For `x < 0` the exponent `j(μ + j s²/2)`
/// is negative and `1 − ½ erfc(|x|) ∈ [½, 1]`, so the direct form loses at most
/// one bit. `half_gauss = ½ e^{−μ²/(2s²)}` is shared by every `j` and both
/// halves.
#[inline]
fn logistic_normal_half_moment(mu: f64, j: f64, s2: f64, sqrt2_s: f64, half_gauss: f64) -> f64 {
    let x = (mu + j * s2) / sqrt2_s;
    if x >= 0.0 {
        half_gauss * erfcx_nonnegative(x)
    } else {
        (j * (mu + 0.5 * j * s2)).exp() - half_gauss * erfcx_nonnegative(-x)
    }
}

/// Logistic-normal mean and location derivative, exact to working precision.
///
/// The CRVZ series above, summed until the unsummed remainder — bounded by the
/// suffix weight sums times the current (largest remaining) moment, since
/// `m_j` decreases in `j` — falls below the unit roundoff of a lower bound of
/// each output (`E[σ] ≥ ½(Φ(μ/s) + m_1⁻)` and `E[σ'] ≥ ¼(m_1⁺ + m_1⁻)`, from
/// `σ ≥ ½` on `η > 0`, `σ ≥ t/2` on `η ≤ 0`, and `σ' ≥ t/4`). Central rows use
/// all `LOGISTIC_NORMAL_SERIES_TERMS` pairs; tail rows stop early. No
/// allocation: the weights are compile-time constants.
pub(crate) fn logit_posterior_meanwith_deriv_exact(
    mu: f64,
    sigma: f64,
) -> Result<IntegratedMeanDerivative, EstimationError> {
    if !(mu.is_finite() && sigma.is_finite()) {
        crate::bail_invalid_estim!("logit integrated moments require finite mu and sigma");
    }
    // The point-mass limit `σ(μ)` differs from `E[σ(μ + σZ)]` by `½σ²σ''(μ) + O(σ⁴)`,
    // and the logistic has `|σ''/σ| ≤ 1` (and `|σ'''/σ'| ≤ 1` for the derivative),
    // so below `σ = √(2u) = √ε` the limit is exact to the rounding of `σ(μ)` (#2469).
    if sigma <= f64::EPSILON.sqrt() {
        let (mean, dmean_dmu) = stable_sigmoidwith_derivative(mu);
        return Ok(IntegratedMeanDerivative {
            mean,
            dmean_dmu,
            mode: IntegratedExpectationMode::ExactClosedForm,
        });
    }
    const N: usize = LOGISTIC_NORMAL_SERIES_TERMS;
    let weights = &LOGISTIC_NORMAL_SERIES_WEIGHTS;
    let unit_roundoff = 0.5 * f64::EPSILON;
    let s2 = sigma * sigma;
    let sqrt2_s = SQRT_2 * sigma;
    let standardized = mu / sigma;
    let half_gauss = 0.5 * (-0.5 * standardized * standardized).exp();
    let upper_mass = gam_math::probability::normal_cdf(standardized);

    let mut mean_series = 0.0;
    let mut slope_series = 0.0;
    let mut mean_floor = 0.0;
    let mut slope_floor = 0.0;
    for j in 1..=N {
        let jf = j as f64;
        let lower = logistic_normal_half_moment(mu, jf, s2, sqrt2_s, half_gauss);
        let upper = logistic_normal_half_moment(-mu, jf, s2, sqrt2_s, half_gauss);
        // m_j enters the mean with w_{j−1} and the slope with j·w_j.
        mean_series += weights.weight[j - 1] * (lower - upper);
        if j < N {
            slope_series += jf * weights.weight[j] * (lower + upper);
        }
        if j == 1 {
            mean_floor = 0.5 * (upper_mass + lower);
            slope_floor = 0.25 * (lower + upper);
        }
        let largest_remaining = lower + upper;
        if largest_remaining * weights.abs_tail[j] <= unit_roundoff * mean_floor
            && largest_remaining * weights.index_abs_tail[(j + 1).min(N)]
                <= unit_roundoff * slope_floor
        {
            break;
        }
    }
    let mean = (upper_mass + mean_series).clamp(0.0, 1.0);
    let dmean_dmu = (-slope_series).max(0.0);
    Ok(IntegratedMeanDerivative {
        mean,
        dmean_dmu,
        mode: IntegratedExpectationMode::ExactSpecialFunction,
    })
}

/// One Laplace-localized Clenshaw–Curtis panel for the log-space survival
/// integrals at a given `(μ, σ)` and μ-derivative order.
///
/// Placement is the whole point: `z_lo`/`z_hi` are where the log-integrand
/// falls [`LOG_SURVIVAL_PANEL_LOG_DROP`] e-folds below its own maximum, and
/// `nodes` comes from the panel's local-scale arclength plus the σ-dependent
/// analyticity term. Value and every derivative order are evaluated on THIS
/// panel, so they are the same approximation surface by construction.
#[derive(Clone, Copy, Debug)]
struct LogSurvivalPanel {
    z_lo: f64,
    z_hi: f64,
    nodes: usize,
}

/// Which branch of `S ↔ 1−S` a panel integrates.
///
/// Both are cancellation-free log-sum-exps of positive terms; the choice is
/// only about which one keeps `ln S` RELATIVELY accurate. Integrating `S` and
/// taking its log loses relative precision as `S → 1` (where `ln S → 0` is a
/// difference of near-equal quantities); integrating the complement and using
/// `ln1p(−(1−S))` does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LogSurvivalBranch {
    /// `S = E[exp(−e^η)]`, integrand `φ(z)·exp(−e^{μ+σz})`.
    Survival,
    /// `1 − S = E[1−exp(−e^η)]`, integrand `φ(z)·(−expm1(−e^{μ+σz}))`.
    Complement,
}

impl LogSurvivalBranch {
    /// `ln` of the branch's integrand without the `−ln√(2π) − z²/2` Gaussian
    /// head, i.e. the part that carries `(μ, σ)`.
    #[inline]
    fn log_tilt(self, mu: f64, sigma: f64, z: f64) -> f64 {
        let u = safe_exp(mu + sigma * z);
        match self {
            Self::Survival => -u,
            // ln(1 − e^{−u}) = ln(−expm1(−u)); for small u this is ln u = μ+σz
            // to full precision, and for large u it is ≈ 0.
            Self::Complement => {
                let m = (-u).exp_m1();
                if m == 0.0 {
                    // u underflowed: 1 − e^{−u} = u to f64.
                    mu + sigma * z
                } else {
                    (-m).ln()
                }
            }
        }
    }

    /// The branch's log-integrand `L(z)` (Gaussian head included, normalizing
    /// constant omitted — it cancels out of every root solve).
    #[inline]
    fn log_integrand(self, mu: f64, sigma: f64, z: f64) -> f64 {
        -0.5 * z * z + self.log_tilt(mu, sigma, z)
    }

    /// `d/dz` of [`Self::log_integrand`], used only to place the peak.
    ///
    /// `Survival`: `−z − σu`. `Complement`: `−z + σu/(e^u − 1)`. Both are
    /// strictly decreasing in `z` (the tilt is concave), so each has a unique
    /// root.
    #[inline]
    fn log_integrand_slope(self, mu: f64, sigma: f64, z: f64) -> f64 {
        let u = safe_exp(mu + sigma * z);
        match self {
            Self::Survival => -z - sigma * u,
            Self::Complement => {
                let em1 = u.exp_m1();
                let tilt_slope = if em1.is_finite() && em1 > 0.0 {
                    sigma * u / em1
                } else if em1 == 0.0 {
                    // u underflowed: u/(e^u − 1) → 1.
                    sigma
                } else {
                    0.0
                };
                -z + tilt_slope
            }
        }
    }
}

/// Maximizer of the survival branch's log-integrand: the unique root of
/// `z + σ e^{μ+σz} = 0`.
///
/// Solved in log form — `ln σ + μ + σz − ln(−z) = 0` on `z < 0` — so no
/// intermediate `e^{μ+σz}` is ever formed. That matters: the root can sit at
/// `z ≈ −μ/σ` with `μ` in the hundreds, where the direct form overflows on the
/// first Newton step and the value form of this solve simply cannot start.
/// The left side is strictly increasing in `z`, so a bracketed Newton is
/// unconditionally convergent.
fn log_survival_peak_z(mu: f64, sigma: f64) -> f64 {
    let log_sigma = sigma.ln();
    let residual = |z: f64| log_sigma + mu + sigma * z - (-z).ln();
    let mut hi = -f64::MIN_POSITIVE;
    let mut lo = -1.0;
    // Doubling a finite `lo` leaves the exponent range within its 1024 binades,
    // so the finiteness check ends this search (#2469).
    while residual(lo) > 0.0 {
        lo *= 2.0;
        if !lo.is_finite() {
            return f64::MIN;
        }
    }
    // Safeguarded Newton on the bracket. A Newton step is taken only when it lands
    // strictly inside the bracket and the bracket halved on the previous update;
    // otherwise the step bisects. So the bracket halves at least every second
    // step, and the loop ends at the bracket's own resolution: no double lies
    // strictly inside it, or the step no longer moves `z` (#2469).
    let mut z = 0.5 * (lo + hi);
    let mut width = hi - lo;
    loop {
        let r = residual(z);
        if r > 0.0 {
            hi = z;
        } else {
            lo = z;
        }
        let midpoint = 0.5 * (lo + hi);
        if !(midpoint > lo && midpoint < hi) {
            return z;
        }
        let halved = hi - lo <= 0.5 * width;
        width = hi - lo;
        // d/dz [ln σ + μ + σz − ln(−z)] = σ + 1/(−z) > 0.
        let newton = z - r / (sigma + 1.0 / (-z));
        let next = if halved && newton > lo && newton < hi { newton } else { midpoint };
        if next == z {
            return next;
        }
        z = next;
    }
}

/// Maximizer of the complement branch's log-integrand: the unique root of
/// `−z + σ u/(e^u − 1) = 0`, `u = e^{μ+σz}`.
///
/// The tilt slope `σ u/(e^u−1)` falls monotonically from `σ` (at `u → 0`) to
/// `0`, so the root is in `(0, σ]` and plain bisection with a Newton-free
/// contraction is both safe and fast — no derivative of the tilt slope is
/// needed.
fn log_complement_peak_z(mu: f64, sigma: f64) -> f64 {
    let branch = LogSurvivalBranch::Complement;
    let (mut lo, mut hi) = (0.0_f64, sigma.max(f64::MIN_POSITIVE));
    if branch.log_integrand_slope(mu, sigma, hi) > 0.0 {
        return hi;
    }
    // Bisect until no double lies strictly inside the bracket, so the peak is
    // located to the last bit at any scale of `σ` (#2469).
    loop {
        let mid = 0.5 * (lo + hi);
        if !(mid > lo && mid < hi) {
            break;
        }
        if branch.log_integrand_slope(mu, sigma, mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// The point on one side of `z_peak` where the log-integrand has fallen `drop`
/// e-folds below its maximum.
///
/// The log-integrand is strictly concave, hence strictly monotone on each side
/// of its peak, so doubling outward until the drop is exceeded and then
/// bisecting is exact to the last bit and cannot be fooled by a second mode.
fn log_survival_panel_edge(
    branch: LogSurvivalBranch,
    mu: f64,
    sigma: f64,
    z_peak: f64,
    drop: f64,
    direction: f64,
) -> f64 {
    let peak_log = branch.log_integrand(mu, sigma, z_peak);
    let fallen = |z: f64| peak_log - branch.log_integrand(mu, sigma, z) >= drop;
    let mut step = 1.0_f64;
    let mut inner = z_peak;
    let mut outer = z_peak + direction * step;
    // Doubling a finite step leaves the exponent range within its 1024 binades,
    // so the finiteness check ends this search (#2469).
    while !fallen(outer) {
        inner = outer;
        step *= 2.0;
        outer = z_peak + direction * step;
        if !outer.is_finite() {
            return inner;
        }
    }
    let (mut lo, mut hi) = if direction > 0.0 {
        (inner, outer)
    } else {
        (outer, inner)
    };
    // Bisect until no double lies strictly inside the bracket (#2469).
    loop {
        let mid = 0.5 * (lo + hi);
        if !(mid > lo && mid < hi) {
            break;
        }
        if fallen(mid) == (direction > 0.0) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    if direction > 0.0 { hi } else { lo }
}

/// `T = ∫ √(1 + σ² e^{μ+σz}) dz` across the panel, in closed form.
///
/// This is the panel's length measured in units of the log-integrand's own
/// local scale `1/√(−L'')`, i.e. how many resolution elements the rule has to
/// cover. With `r = √(1+σ²e^{μ+σz})` the antiderivative is
/// `(2r + ln((r−1)/(r+1)))/σ`; written as below it stays exact in the
/// `σ²e^{μ+σz} → 0` limit, where it must reduce to the plain length.
fn log_survival_panel_arclength(mu: f64, sigma: f64, z_lo: f64, z_hi: f64) -> f64 {
    let scale_root = |z: f64| (1.0 + safe_exp(2.0 * sigma.ln() + mu + sigma * z)).sqrt();
    let (r_lo, r_hi) = (scale_root(z_lo), scale_root(z_hi));
    if !(r_lo.is_finite() && r_hi.is_finite()) {
        return f64::INFINITY;
    }
    (z_hi - z_lo) + 2.0 / sigma * ((r_hi - r_lo) - ((1.0 + r_hi) / (1.0 + r_lo)).ln())
}

/// The order the DERIVATIVE-TOWER panel is placed at, for every request that
/// asks for one (#2714).
///
/// The placement below widens the window (`drop`) and adds nodes with the order
/// it is placed for. Reading that order off whatever the caller happened to ask
/// for makes `σ^j ∂_μ^j S` a function of THE REQUEST as well as of `(μ, σ)` —
/// and the consumers ask for different orders at the same `(μ, σ)` inside one
/// fit. `log_kernel_bundle` passes `max_k`, which the latent row program derives
/// from `base_max_k + 2·max_primary_increment + max_suffix_increment`, i.e. from
/// whether the caller wanted a value, a gradient, a Hessian or a contracted
/// third: `4`, `5`, `6` and `7` on the same row. So the gradient and the Hessian
/// paired with it were reading the same integral off four different rules, and
/// the Hessian was not exactly the derivative of the gradient — the fault class
/// this issue is about, one level below where it was found.
///
/// Pinning the placement makes entry `j` a property of `(μ, σ)` alone.
///
/// The VALUE route pays nothing for it. `order = 0` requests keep their own
/// order-0 placement, which is already a function of `(μ, σ)` alone, and that is
/// the placement every `ln S` in the tree comes off — a bundle evaluates
/// `max_k + 1` of them against one tower, so the tower's `~1.2×` node count is
/// a few percent of the bundle and the hot path is byte-identical.
const LOG_SURVIVAL_TOWER_PANEL_ORDER: usize = LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER;

/// The order a panel is PLACED at, given the order a caller asked to be
/// filled in.
///
/// Two placements, each a pure function of `(branch, μ, σ)`: the value surface
/// (`order = 0`) and the tower surface. See [`LOG_SURVIVAL_TOWER_PANEL_ORDER`].
/// They agree about `S` itself wherever it matters, because the one consumer of
/// the tower — `log_scaled_a_derivative_tower` — takes entry 0 from the value
/// route verbatim rather than from the tower panel.
#[inline]
fn log_survival_panel_placement_order(requested_order: usize) -> usize {
    if requested_order == 0 {
        0
    } else {
        LOG_SURVIVAL_TOWER_PANEL_ORDER
    }
}

/// Place the panel for `(branch, μ, σ)`, given the μ-derivative order the caller
/// asked to have filled in.
///
/// The `order` argument reaches the geometry only through
/// [`log_survival_panel_placement_order`], so the panel is one of exactly two
/// surfaces and never a per-request one.
fn log_survival_panel(
    branch: LogSurvivalBranch,
    mu: f64,
    sigma: f64,
    requested_order: usize,
) -> LogSurvivalPanel {
    let order = log_survival_panel_placement_order(requested_order);
    let z_peak = match branch {
        LogSurvivalBranch::Survival => log_survival_peak_z(mu, sigma),
        LogSurvivalBranch::Complement => log_complement_peak_z(mu, sigma),
    };
    let drop = LOG_SURVIVAL_PANEL_LOG_DROP
        + LOG_SURVIVAL_PANEL_ORDER_LOG_DROP * (order as f64) * (2.0 + z_peak.abs()).ln();
    let z_lo = log_survival_panel_edge(branch, mu, sigma, z_peak, drop, -1.0);
    let z_hi = log_survival_panel_edge(branch, mu, sigma, z_peak, drop, 1.0);
    let arclength = log_survival_panel_arclength(mu, sigma, z_lo, z_hi);
    let requested = LOG_SURVIVAL_PANEL_ARCLENGTH_NODE_DENSITY * arclength
        + LOG_SURVIVAL_PANEL_SIGMA_NODE_SCALE * sigma.sqrt()
        + LOG_SURVIVAL_PANEL_ORDER_NODES * (order as f64);
    let mut nodes = if requested.is_finite() {
        (requested.ceil() as usize).clamp(
            LOG_SURVIVAL_PANEL_MIN_NODES,
            LOG_SURVIVAL_PANEL_MAX_NODES,
        )
    } else {
        LOG_SURVIVAL_PANEL_MAX_NODES
    };
    if nodes.is_multiple_of(2) {
        nodes += 1;
    }
    LogSurvivalPanel { z_lo, z_hi, nodes }
}

/// One entry of the log-space survival tower: `ln|·|`, its sign, and the
/// cancellation the signed log-sum-exp actually suffered.
///
/// `log_cancellation = ln(Σ|terms| / |Σ terms|)` is not diagnostic decoration —
/// it is the rule's own error bar. A signed log-sum-exp returns a result whose
/// relative error is `≈ ε · Σ|terms|/|Σ terms|`, so a consumer that needs a
/// certified derivative reads this instead of guessing a `σ` threshold.
#[derive(Clone, Copy, Debug)]
pub struct LogSurvivalSignedValue {
    pub log_abs: f64,
    pub sign: f64,
    pub log_cancellation: f64,
}

impl LogSurvivalSignedValue {
    const ZERO: Self = Self {
        log_abs: f64::NEG_INFINITY,
        sign: 0.0,
        log_cancellation: f64::INFINITY,
    };
}

/// `ln S` together with `σ^j ∂_μ^j S` for `j = 0..=order`, all off ONE panel.
#[derive(Clone, Debug)]
pub struct LogSurvivalJet {
    /// `ln S(μ,σ)`.
    pub log_survival: f64,
    /// Entry `j` is `σ^j ∂_μ^j S` in signed-log form; entry 0 is `(ln S, +1)`.
    pub scaled_mu_derivatives: [LogSurvivalSignedValue; LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER + 1],
    /// Highest order actually filled in.
    pub order: usize,
    pub mode: IntegratedExpectationMode,
}

impl LogSurvivalJet {
    /// The scaled μ-derivatives `σ^j ∂_μ^j S`, `j = 1..=order`, or `None` when
    /// the quadrature's own measured cancellation says they are not accurate
    /// enough to displace the rung (Touchard) basis.
    ///
    /// This asks the quadrature what cancellation it suffered rather than
    /// asking `σ` to stand in for it. The two derivative bases are
    /// complementary and both are needed: the direct tower is one integral per
    /// order and degrades only where that integral cancels (small σ, where
    /// `σ^j ∂_μ^j S` is genuinely `O(σ^j)` while the summands are `O(1)`); the
    /// rung basis is an alternating combination of `K_k` and degrades where
    /// THOSE cancel (large σ, #2610). Before #2714 the split was a `σ ≥ 8`
    /// constant. It is now the measured conditioning of the thing admitted.
    ///
    /// All-or-nothing on purpose: a caller either gets the whole tower on one
    /// surface or falls back to the rung basis, never a partial mix.
    pub fn certified_scaled_mu_derivatives(
        &self,
        order: usize,
    ) -> Option<&[LogSurvivalSignedValue]> {
        let prefix = self.certified_prefix_order(order)?;
        (prefix == order).then(|| &self.scaled_mu_derivatives[..=order])
    }

    /// The largest `r ≤ order` for which entries `0..=r` are ALL certified, or
    /// `None` when even entry 0 is not (#2714).
    ///
    /// Certification is a prefix property — every entry is judged on its own
    /// finiteness, sign and measured cancellation — so "the whole tower up to
    /// `order`" and "the longest usable tower" are the same test read to two
    /// different lengths, and asking for the longest is what makes the answer
    /// independent of how much the caller asked for.
    ///
    /// That independence is the point. [`Self::certified_scaled_mu_derivatives`]
    /// refuses the whole tower when the LAST rung fails, so a consumer that
    /// needed rungs `0..=1` and asked for `0..=5` was denied a basis that is
    /// perfectly well conditioned at the rungs it actually reads — and, worse,
    /// two consumers reading the SAME term list at the same `(μ, σ)` could be
    /// routed to different bases purely because one of them also wanted a
    /// Hessian. Handing back the certified prefix moves that decision onto the
    /// term list, where it belongs.
    ///
    /// It does not weaken the "never a partial mix" rule: a tower truncated at
    /// `r` makes every term list needing a rung above `r` fall back WHOLE to the
    /// rung basis (see `latent_kernel_evaluate_terms_in_a_basis`, which refuses
    /// the list rather than the term). What is admitted is still one surface.
    pub fn certified_prefix_order(&self, order: usize) -> Option<usize> {
        let order = order.min(self.order);
        let mut certified = None;
        for (index, entry) in self.scaled_mu_derivatives[..=order].iter().enumerate() {
            let usable = entry.log_abs.is_finite()
                && (index == 0 || entry.sign != 0.0)
                && entry.log_cancellation <= LOG_SURVIVAL_TOWER_MAX_LOG_CANCELLATION;
            if !usable {
                break;
            }
            certified = Some(index);
        }
        certified
    }
}

/// Streaming signed log-sum-exp that also accumulates `ln Σ|terms|`.
struct SignedLogAccumulator {
    running_max: f64,
    signed_sum: f64,
    abs_sum: f64,
}

impl SignedLogAccumulator {
    #[inline]
    fn new() -> Self {
        Self {
            running_max: f64::NEG_INFINITY,
            signed_sum: 0.0,
            abs_sum: 0.0,
        }
    }

    #[inline]
    fn push(&mut self, log_abs: f64, sign: f64) {
        if !log_abs.is_finite() {
            return;
        }
        if log_abs > self.running_max {
            let rescale = (self.running_max - log_abs).exp();
            self.signed_sum = self.signed_sum * rescale + sign;
            self.abs_sum = self.abs_sum * rescale + 1.0;
            self.running_max = log_abs;
        } else {
            let weight = (log_abs - self.running_max).exp();
            self.signed_sum += sign * weight;
            self.abs_sum += weight;
        }
    }

    #[inline]
    fn finish(self) -> LogSurvivalSignedValue {
        if self.running_max == f64::NEG_INFINITY || self.signed_sum == 0.0 {
            return LogSurvivalSignedValue::ZERO;
        }
        LogSurvivalSignedValue {
            log_abs: self.running_max + self.signed_sum.abs().ln(),
            sign: self.signed_sum.signum(),
            log_cancellation: (self.abs_sum / self.signed_sum.abs()).ln().max(0.0),
        }
    }
}

/// Evaluate `∫ He_j(z) φ(z) · tilt(z) dz` for `j = 0..=order` on one panel.
///
/// Every order reads the SAME nodes and the SAME tilt values; only the Hermite
/// factor differs. That is what makes the value and its derivative tower one
/// approximation surface rather than two that happen to agree in the middle.
fn log_survival_panel_moments(
    ctx: &QuadratureContext,
    branch: LogSurvivalBranch,
    mu: f64,
    sigma: f64,
    order: usize,
) -> ([LogSurvivalSignedValue; LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER + 1], usize) {
    let panel = log_survival_panel(branch, mu, sigma, order);
    let rule = ctx.clenshaw_curtis_n(panel.nodes);
    let half = 0.5 * (panel.z_hi - panel.z_lo);
    let mid = 0.5 * (panel.z_hi + panel.z_lo);
    let log_gaussian_norm = -0.5 * (2.0 * std::f64::consts::PI).ln();
    let mut accumulators: Vec<SignedLogAccumulator> =
        (0..=order).map(|_| SignedLogAccumulator::new()).collect();
    for (&node, &weight) in rule.nodes.iter().zip(rule.weights.iter()) {
        let z = half * node + mid;
        let base = (weight * half).ln() + log_gaussian_norm - 0.5 * z * z
            + branch.log_tilt(mu, sigma, z);
        if !base.is_finite() {
            continue;
        }
        for (j, accumulator) in accumulators.iter_mut().enumerate() {
            let hermite = hermite_he(j, z);
            if hermite == 0.0 {
                continue;
            }
            accumulator.push(base + hermite.abs().ln(), hermite.signum());
        }
    }
    let mut out = [LogSurvivalSignedValue::ZERO; LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER + 1];
    for (slot, accumulator) in out.iter_mut().zip(accumulators.into_iter()) {
        *slot = accumulator.finish();
    }
    (out, panel.nodes)
}

/// The canonical log-space survival object: `ln S(μ,σ)` and its scaled
/// μ-derivative tower, on one panel, for every `(μ, σ)`.
///
/// The tower identity is Gaussian integration by parts in the standardized
/// variable: with `f(z) = exp(−e^{μ+σz})`, `∂_μ f = σ^{-1} ∂_z f`, so
///
/// ```text
///   σ^j ∂_μ^j S = ∫ He_j(z) φ(z) f(z) dz,
/// ```
///
/// because `(−∂_z)^j φ = He_j φ`. Order 0 is `S` itself, which is why the
/// value and the tower cannot come off different surfaces here even in
/// principle — they are the same sum with different Hermite weights.
///
/// Sign convention: entries are `σ^j ∂_μ^j S`, so odd orders are negative for
/// the survival branch (`S` decreases in `μ`).
pub fn log_survival_jet(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
    order: usize,
) -> LogSurvivalJet {
    let order = order.min(LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER);
    if !(mu.is_finite() && sigma.is_finite()) || sigma <= CLOGLOG_SIGMA_DEGENERATE {
        // σ = 0: S = exp(−e^μ) and σ^j ∂_μ^j S = 0 for j ≥ 1 (the SCALED
        // derivative carries a σ^j that annihilates the finite ∂_μ^j S).
        let mut scaled = [LogSurvivalSignedValue::ZERO;
            LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER + 1];
        let log_survival = -safe_exp(mu);
        scaled[0] = LogSurvivalSignedValue {
            log_abs: log_survival,
            sign: 1.0,
            log_cancellation: 0.0,
        };
        return LogSurvivalJet {
            log_survival,
            scaled_mu_derivatives: scaled,
            order,
            mode: IntegratedExpectationMode::ExactClosedForm,
        };
    }

    let (survival, _) =
        log_survival_panel_moments(ctx, LogSurvivalBranch::Survival, mu, sigma, order);
    // `ln S` from the survival branch is accurate ABSOLUTELY; as S → 1 that is
    // not the same as accurate relatively, because ln S → 0 is then the log of
    // a number whose distance from 1 carries the information. Past S ≈ 0.6 the
    // complement branch — an independent, equally cancellation-free panel —
    // supplies 1 − S directly and `ln1p` keeps every digit. Both branches are
    // at the f64 floor where they meet, so the handover is smooth to ~1e-16.
    let use_complement = survival[0].sign > 0.0 && survival[0].log_abs > -0.5;
    if !use_complement {
        let log_survival = if survival[0].sign > 0.0 {
            survival[0].log_abs
        } else {
            f64::NEG_INFINITY
        };
        return LogSurvivalJet {
            log_survival,
            scaled_mu_derivatives: survival,
            order,
            mode: IntegratedExpectationMode::ControlledAsymptotic,
        };
    }

    let (complement, _) =
        log_survival_panel_moments(ctx, LogSurvivalBranch::Complement, mu, sigma, order);
    let mut scaled = survival;
    // `log_abs < 0` is guaranteed by the handover test above (it fires only for
    // S > 0.6, hence 1 − S < 0.4) and is checked rather than assumed, because
    // `ln1p` of anything ≤ −1 is not a number.
    let log_survival = if complement[0].sign > 0.0 && complement[0].log_abs < 0.0 {
        (-safe_exp(complement[0].log_abs)).ln_1p()
    } else {
        survival[0].log_abs
    };
    scaled[0] = LogSurvivalSignedValue {
        log_abs: log_survival,
        sign: 1.0,
        log_cancellation: complement[0].log_cancellation,
    };
    // For j ≥ 1, ∂_μ^j S = −∂_μ^j (1−S) exactly, and the complement integrand
    // is the one that stays away from a constant in this regime, so it is the
    // better-conditioned of the two. Take whichever reports less cancellation;
    // both are the same quantity to their own accuracy.
    for j in 1..=order {
        let flipped = LogSurvivalSignedValue {
            log_abs: complement[j].log_abs,
            sign: -complement[j].sign,
            log_cancellation: complement[j].log_cancellation,
        };
        if flipped.sign != 0.0 && flipped.log_cancellation < scaled[j].log_cancellation {
            scaled[j] = flipped;
        }
    }
    LogSurvivalJet {
        log_survival,
        scaled_mu_derivatives: scaled,
        order,
        mode: IntegratedExpectationMode::ControlledAsymptotic,
    }
}

#[inline]
fn hermite_he(n: usize, x: f64) -> f64 {
    let mut previous = 1.0_f64;
    if n == 0 {
        return previous;
    }
    let mut current = x;
    for order in 1..n {
        let next = x * current - (order as f64) * previous;
        previous = current;
        current = next;
    }
    current
}


/// Canonical log-space survival evaluator: returns `ln S(μ,σ)` with its routing
/// mode. This is the log-domain twin of [`cloglog_survival_term_controlled`].
///
/// Every kernel quantity that multiplies `S` by an `exp(kμ + ½k²σ²)` prefix —
/// the integrated cloglog derivative, the latent-cloglog jet, the lognormal
/// kernel bundle — must form the product in log space through this function, so
/// the genuine (but f64-unrepresentable) magnitude of `S` is preserved instead
/// of underflowing to a hard zero that zeroes the derivative (#798).
///
/// It is ONE surface (#2714). There is deliberately no `.ln()` of a value-space
/// result here and no threshold between competing approximations: `ln S` is
/// accumulated in log space on the panel of [`log_survival_jet`] at every
/// `(μ, σ)` with `σ > 0`, and the only other branch is the exact `σ = 0` closed
/// form. What used to live here — a rare-event asymptotic, a value-route
/// logarithm, and a fixed-window quadrature as an underflow escape — disagreed
/// with each other by up to `4.4e+06` in `ln S`, and their disagreement was a
/// step function of `(μ, σ)`, which no analytic derivative can follow.
pub(crate) fn cloglog_log_survival_term_controlled(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> (f64, IntegratedExpectationMode) {
    let jet = log_survival_jet(ctx, mu, sigma, 0);
    (jet.log_survival, jet.mode)
}

// ── Exact Gumbel survival primitives ─────────────────────────────────────
//
// The Gumbel survival function S(x) = exp(-exp(x)) is the complement of the
// cloglog mean μ(x) = 1 - S(x). Both are exact for ALL finite x under IEEE 754
// without any clamping:
//
//   x → -∞: exp(x) → 0,    S → exp(-0) = 1,  μ' → 0·1 = 0
//   x → +∞: exp(x) → +∞,   S → exp(-∞) = 0,  μ' → ∞·0 = 0
//
// The only subtlety is in μ': when x > 709, exp(x) overflows to +∞,
// and ∞ · 0 = NaN. But x - exp(x) → -∞ for any x > 0, so μ' = 0.
// We detect the intermediate overflow and return 0.0 exactly.

/// Exact Gumbel survival: S(x) = exp(-exp(x)).
///
/// No clamping — IEEE 754 handles both tails correctly:
/// - exp(x) underflows to 0 for x < -745 → S = exp(-0) = 1.0
/// - exp(x) overflows to ∞ for x > 709  → S = exp(-∞) = 0.0
#[inline]
fn gumbel_survival(x: f64) -> f64 {
    (-safe_exp(x)).exp()
}

/// Exact cloglog mean derivative: μ'(x) = exp(x) · exp(-exp(x)) = -S'(x).
///
/// Saturates the intermediate exp in the positive tail; double-exponential
/// decay still drives the returned derivative to 0.0.
#[inline]
fn cloglog_mean_d1_exact(x: f64) -> f64 {
    let ex = safe_exp(x);
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
        let ex = safe_exp(eta);
        -(-ex).exp_m1()
    }
}

// Pointwise cloglog derivative dμ/dη in the deep negative tail:
// `cloglog_negative_tail_derivative` (a reference implementation retained
// solely for its unit test) lives inside `mod tests` below.

#[inline]
/// Panelized adaptive-Simpson refinement with Richardson extrapolation on a
/// single panel `[a, b]`. `whole` is the one-panel Simpson estimate; the panel
/// is bisected until the two-panel estimate agrees to `tol` (or `depth` is
/// exhausted), then the extrapolated value is returned.
fn adaptive_simpson_refine(
    g: &impl Fn(f64) -> f64,
    a: f64,
    b: f64,
    fa: f64,
    fb: f64,
    fm: f64,
    whole: f64,
    tol: f64,
    depth: i32,
) -> f64 {
    let m = 0.5 * (a + b);
    let lm = 0.5 * (a + m);
    let rm = 0.5 * (m + b);
    let flm = g(lm);
    let frm = g(rm);
    let left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
    let right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
    let est = left + right;
    if depth <= 0 || (est - whole).abs() <= 15.0 * tol {
        return est + (est - whole) / 15.0;
    }
    adaptive_simpson_refine(g, a, m, fa, fm, flm, left, 0.5 * tol, depth - 1)
        + adaptive_simpson_refine(g, m, b, fm, fb, frm, right, 0.5 * tol, depth - 1)
}

/// Accurate Gaussian expectation `E[f(mu + sigma·Z)]`, `Z ~ N(0,1)`, via
/// panelized adaptive Simpson over the standardized window `u ∈ [-K, K]`.
///
/// This is the trusted fallback when the controlled special-function backends
/// decline. Fixed Gauss-Hermite quadrature undersamples integrands whose
/// features are narrow in standardized coordinates — the cloglog transition
/// `1 − exp(−exp(η))` has width `~1/sigma` in `u`, so once `sigma` is large it
/// collapses below the GHQ node spacing and most of the fixed nodes scatter
/// into the flat dead zone, leaving only a handful to resolve the transition
/// (the ~1e-3 mean / ~3.5e-3 derivative error observed at `sigma = 4`).
/// Adaptive Simpson instead refines panels only where the integrand curves,
/// resolving the transition to tolerance regardless of `sigma`. The
/// standard-normal density kills the tails (`φ(15) ~ 1e-49`), so the finite
/// window `K = 15` captures the whole integral with no analytic tail term.
fn integrate_normal_adaptive(mu: f64, sigma: f64, f: impl Fn(f64) -> f64) -> f64 {
    // A Gaussian collapses to a point mass only at `sigma <= 0`. Any positive
    // `sigma` is integrated in standardized coordinates, where the panels resolve
    // `f(mu + sigma u)` to tolerance however narrow the density (#2469).
    if !(sigma.is_finite()) || sigma <= 0.0 {
        return f(mu);
    }
    const K: f64 = 15.0;
    const INITIAL_PANELS: usize = 24;
    const TOL: f64 = 1e-12;
    const MAX_DEPTH: i32 = 40;
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    // Integrand in standardized coordinates: f(mu + sigma·u) · φ(u). A coarse
    // initial panel grid guarantees the transition cannot fall entirely
    // between sampled points before adaptive refinement engages.
    let g = |u: f64| f(mu + sigma * u) * inv_sqrt_2pi * (-0.5 * u * u).exp();
    let panel = 2.0 * K / INITIAL_PANELS as f64;
    let mut total = 0.0;
    for p in 0..INITIAL_PANELS {
        let a = -K + p as f64 * panel;
        let b = a + panel;
        let fa = g(a);
        let fb = g(b);
        let fm = g(0.5 * (a + b));
        let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);
        total += adaptive_simpson_refine(&g, a, b, fa, fb, fm, whole, TOL, MAX_DEPTH);
    }
    total
}

fn cloglog_survival_term_controlled(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> (f64, IntegratedExpectationMode) {
    // The lognormal-Laplace object
    //
    //   S(mu, sigma) = E[exp(-exp(eta))],  eta ~ N(mu, sigma^2),
    //
    // is the survival transform itself and the complement-core of the cloglog
    // inverse link (cloglog mean = 1 - S, survival mean = S). If X = exp(eta),
    // then X ~ LogNormal(mu, sigma^2) and S = E[exp(-X)] = L(1; mu, sigma), with
    // L(z; mu, sigma) = E[exp(-z exp(eta))] in general.
    //
    // Its value is the exponential of the one log-space survival surface of the
    // #2714 panel at every sigma > 0; the only other branch is the exact sigma = 0
    // point mass. S is consumed here as a probability, so exponentiating ln S
    // loses nothing; a consumer that needs its magnitude below the f64 floor reads
    // `cloglog_log_survival_term_controlled` instead.
    if !(mu.is_finite() && sigma.is_finite()) || sigma <= CLOGLOG_SIGMA_DEGENERATE {
        return (
            gumbel_survival(mu).clamp(0.0, 1.0),
            IntegratedExpectationMode::ExactClosedForm,
        );
    }
    let jet = log_survival_jet(ctx, mu, sigma, 0);
    (safe_exp(jet.log_survival).clamp(0.0, 1.0), jet.mode)
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

/// Log-space twin of `lognormal_laplace_unit_term_shared`: returns
/// `ln L(1; shifted_mu, σ) = ln S(shifted_mu, σ)`. Used by the lognormal kernel
/// bundle so kernel log-magnitudes survive value-space underflow (#798).
#[inline]
pub fn lognormal_laplace_unit_log_term_shared(
    ctx: &QuadratureContext,
    shifted_mu: f64,
    sigma: f64,
) -> (f64, IntegratedExpectationMode) {
    cloglog_log_survival_term_controlled(ctx, shifted_mu, sigma)
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

/// The integrated cloglog location derivative by the exact Gaussian tilting
/// identity
///
/// ```text
///   d/dmu E[1 - exp(-exp(eta))] = exp(mu + sigma^2/2) * S(mu + sigma^2, sigma),
/// ```
///
/// from `ln S(mu + sigma^2, sigma)`. The product is formed in log space: the
/// shifted survival can sit below the f64 floor (large σ, #798) while the
/// derivative is finite and O(1), because the huge prefix exactly compensates
/// the tiny survival, and it is bounded by `sup_x x·e^{−x} = e^{−1}`.
#[inline]
fn cloglog_shift_identity_derivative_log(mu: f64, sigma: f64, log_shifted_survival: f64) -> f64 {
    if !(mu.is_finite() && sigma.is_finite()) || log_shifted_survival == f64::NEG_INFINITY {
        return 0.0;
    }
    let log_derivative = mu + 0.5 * sigma * sigma + log_shifted_survival;
    let upper = 1.0 / std::f64::consts::E;
    if !log_derivative.is_finite() {
        // Mathematically bounded by sup_x x·e^{−x} = e^{−1}; any overflow here
        // is purely numerical (the exp(mu+σ²/2) prefix), so cap at the bound.
        return upper;
    }
    safe_exp(log_derivative).clamp(0.0, upper)
}

pub(crate) fn cloglog_posterior_meanwith_deriv_controlled(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> IntegratedMeanDerivative {
    // Integrated cloglog under Gaussian latent uncertainty:
    //
    //   mean(mu, sigma)  = E[1 - exp(-exp(eta))] = 1 - S(mu, sigma),
    //   dmean/dmu        = E[exp(eta - exp(eta))]
    //                    = exp(mu + sigma^2/2) * S(mu + sigma^2, sigma),
    //   eta ~ N(mu, sigma^2),
    //
    // the second line by Gaussian tilting. Both read the one log-space survival
    // surface `ln S` of the #2714 panel at every sigma > 0: the mean as
    // `-expm1(ln S)`, the derivative with its `exp(mu + sigma^2/2)` prefix added
    // in log space. There is no second evaluator to route to or to check against.
    // The value-space ladder this replaces (heat-kernel Taylor, extreme-input
    // asymptotics, Miles / Clenshaw-Curtis / Gamma series, adaptive Simpson, and
    // a drift check between them) was wrong by 336% relative in the mean at
    // (mu, sigma) = (-35, 5.9) and by 26% in the derivative at (8, 1) against a
    // 50-digit reference (#2469), and its absolute drift tolerances saw neither.
    if !(mu.is_finite() && sigma.is_finite()) || sigma <= CLOGLOG_SIGMA_DEGENERATE {
        return IntegratedMeanDerivative {
            // cloglog_mean_exact uses expm1 to avoid 1 − 1 cancellation for
            // all mu, and handles exp overflow (mu > 709) via expm1(-∞) = −1.
            mean: cloglog_mean_exact(mu),
            // cloglog_mean_d1_exact is exact for all finite mu: it detects
            // intermediate exp overflow and returns 0.0 (correct limit).
            dmean_dmu: cloglog_mean_d1_exact(mu),
            mode: IntegratedExpectationMode::ExactClosedForm,
        };
    }
    let (log_base, base_mode) = cloglog_log_survival_term_controlled(ctx, mu, sigma);
    let (log_shift, shift_mode) =
        cloglog_log_survival_term_controlled(ctx, mu + sigma * sigma, sigma);
    // mean = 1 − S = −expm1(ln S), stable for S near both 0 and 1.
    let mean = (-log_base.exp_m1()).clamp(0.0, 1.0);
    let dmean = cloglog_shift_identity_derivative_log(mu, sigma, log_shift);
    IntegratedMeanDerivative {
        mean,
        dmean_dmu: dmean.max(0.0),
        mode: worse_integrated_expectation_mode(base_mode, shift_mode),
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
            let (mean, saturated) = safe_expwith_saturation(mu + 0.5 * sigma * sigma);
            Ok(IntegratedMeanDerivative {
                mean,
                dmean_dmu: mean,
                mode: if saturated {
                    IntegratedExpectationMode::ControlledAsymptotic
                } else {
                    IntegratedExpectationMode::ExactClosedForm
                },
            })
        }
        LinkFunction::Probit => Ok(probit_posterior_meanwith_deriv_exact(mu, sigma)),
        LinkFunction::Logit => logit_posterior_meanwith_deriv_exact(mu, sigma),
        LinkFunction::CLogLog => Ok(cloglog_posterior_meanwith_deriv_controlled(quadctx, mu, sigma)),
        LinkFunction::LogLog | LinkFunction::Cauchit => {
            // The outer arm restricts `link` to exactly these two variants.
            let component = if matches!(link, LinkFunction::LogLog) {
                LinkComponent::LogLog
            } else {
                LinkComponent::Cauchit
            };
            let (mean, dmean_dmu, _, _) = integrate_normal_ghq_adaptive(quadctx, mu, sigma, |x| {
                component_point_jet(component, x)
            });
            Ok(IntegratedMeanDerivative {
                mean,
                dmean_dmu,
                mode: if sigma <= 0.0 {
                    IntegratedExpectationMode::ExactClosedForm
                } else {
                    IntegratedExpectationMode::QuadratureFallback
                },
            })
        }
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
        LinkFunction::Inverse | LinkFunction::InverseSquared => {
            let jet = reciprocal_link_posterior_jet(link, mu, sigma)?;
            Ok(IntegratedMeanDerivative {
                mean: jet.mean,
                dmean_dmu: jet.d1,
                mode: jet.mode,
            })
        }
    }
}

/// Posterior mean of a reciprocal-power inverse link, `μ = η^{-1}` or
/// `μ = η^{-1/2}`, under `η ~ N(mu, sigma²)`, with its `mu`-derivatives.
///
/// The Gaussian has mass on `η ≤ 0`, where `1/η` is not integrable and
/// `η^{-1/2}` is not real, so the mean is the real part of `E[g⁻¹(η + i0)]`:
/// the Cauchy principal value (via Dawson's integral) for `1/η` and the
/// positive-half integral for `η^{-1/2}`. Both reduce to the plug-in at
/// `sigma = 0` and match the moment expansion of `g⁻¹` about `mu` to every
/// order (see `gam_math::gaussian_reciprocal`).
///
/// The linear predictor itself must lie in the link's domain `η > 0`: a
/// posterior centred on or below zero describes no positive mean.
fn reciprocal_link_posterior_jet(
    link: LinkFunction,
    mu: f64,
    sigma: f64,
) -> Result<IntegratedInverseLinkJet, EstimationError> {
    if !(mu.is_finite() && mu > 0.0 && sigma.is_finite() && sigma >= 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "{} link posterior mean requires a finite linear predictor inside its domain eta > 0 \
             and a finite nonnegative standard error; got eta = {mu}, se = {sigma}",
            link.name()
        )));
    }
    let [mean, d1, d2, d3] = match link {
        LinkFunction::Inverse => gam_math::gaussian_reciprocal::principal_value_inverse_normal_jet(mu, sigma),
        LinkFunction::InverseSquared => {
            // `η^{-1/2}` is the inverse of the `1/μ²` link.
            gam_math::gaussian_reciprocal::positive_part_inverse_sqrt_normal_jet(mu, sigma)
        }
        other => {
            return Err(EstimationError::InvalidInput(format!(
                "reciprocal-link posterior mean reached non-reciprocal link {other:?}"
            )));
        }
    };
    if ![mean, d1, d2, d3].iter().all(|value| value.is_finite()) {
        return Err(EstimationError::InvalidInput(format!(
            "{} link posterior mean is not representable at eta = {mu}, se = {sigma}",
            link.name()
        )));
    }
    Ok(IntegratedInverseLinkJet {
        mean,
        d1,
        d2,
        d3,
        mode: if sigma == 0.0 {
            IntegratedExpectationMode::ExactClosedForm
        } else {
            IntegratedExpectationMode::ExactSpecialFunction
        },
    })
}

/// Posterior mean and variance of a reciprocal-power inverse link under
/// `η ~ N(mu, sigma²)`, on the same analytic continuation `η + i0` that
/// defines the posterior mean (`reciprocal_link_posterior_jet`).
///
/// The second moment is `Re E[g⁻¹(η + i0)²]`:
///
/// - inverse link: `(η + i0)^{-2}`, whose real expectation is the Hadamard
///   finite part `−d/dm PV E[1/η]`, i.e. minus the first derivative of the
///   principal-value jet;
/// - inverse-squared link: `((η + i0)^{-1/2})² = (η + i0)^{-1}`, whose real
///   expectation is the principal value `PV E[1/η]`.
///
/// Both match the moment expansion of `g⁻¹(η)²` about `mu` to every order, so
/// the variance agrees with the exact moments of any posterior that keeps its
/// mass away from the pole. A negative difference means the posterior reaches
/// the pole so closely that no response-scale variance exists; that is
/// reported as an error rather than clamped.
pub fn reciprocal_link_posterior_meanvariance(
    link: LinkFunction,
    mu: f64,
    sigma: f64,
) -> Result<(f64, f64), EstimationError> {
    let mean = reciprocal_link_posterior_jet(link, mu, sigma)?.mean;
    let second_moment = match link {
        LinkFunction::Inverse => {
            -gam_math::gaussian_reciprocal::principal_value_inverse_normal_jet(mu, sigma)[1]
        }
        LinkFunction::InverseSquared => {
            gam_math::gaussian_reciprocal::principal_value_inverse_normal_jet(mu, sigma)[0]
        }
        other => {
            return Err(EstimationError::InvalidInput(format!(
                "reciprocal-link posterior variance reached non-reciprocal link {other:?}"
            )));
        }
    };
    if sigma == 0.0 {
        return Ok((mean, 0.0));
    }
    // `second_moment − mean²` cancels when `sigma ≪ mu`; a difference below
    // the rounding of its two operands is a zero variance, not a missing one.
    let rounding = 4.0 * f64::EPSILON * second_moment.abs().max(mean * mean);
    let variance = second_moment - mean * mean;
    let variance = if variance < 0.0 && -variance <= rounding { 0.0 } else { variance };
    if !(variance.is_finite() && variance >= 0.0) {
        return Err(EstimationError::InvalidInput(format!(
            "{} link posterior variance does not exist at eta = {mu}, se = {sigma}: the \
             linear-predictor posterior reaches the link's pole at eta = 0",
            link.name()
        )));
    }
    Ok((mean, variance))
}

#[inline]
pub(crate) fn integrated_inverse_link_jet(
    quadctx: &QuadratureContext,
    link: LinkFunction,
    mu: f64,
    sigma: f64,
) -> Result<IntegratedInverseLinkJet, EstimationError> {
    match link {
        LinkFunction::Log => {
            let (mean, saturated) = safe_expwith_saturation(mu + 0.5 * sigma * sigma);
            Ok(IntegratedInverseLinkJet {
                mean,
                d1: mean,
                d2: mean,
                d3: mean,
                mode: if saturated {
                    IntegratedExpectationMode::ControlledAsymptotic
                } else {
                    IntegratedExpectationMode::ExactClosedForm
                },
            })
        }
        LinkFunction::Probit => Ok(integrated_probit_jet(mu, sigma)),
        LinkFunction::Logit => {
            if sigma > LOGIT_JET_GHQ_SIGMA_MAX {
                // Wide σ: Gauss-Hermite under-resolves the localized
                // sigmoid^(k) integrands. Integrate accurately and reuse the
                // scalar backend's mean/d1 so the two entry points agree (#571).
                return logit_wide_sigma_jet(mu, sigma);
            }
            // Integrate the full pointwise jet directly: the same
            // Gauss-Hermite nodes evaluate component_point_jet, so mean/d1
            // retain their scalar-backend values to rounding and d2/d3 are
            // recovered analytically from the node-level jet.
            let (mean, d1, d2, d3) = integrate_normal_ghq_adaptive(quadctx, mu, sigma, |x| {
                component_point_jet(LinkComponent::Logit, x)
            });
            let mode = if sigma <= 0.0 {
                IntegratedExpectationMode::ExactClosedForm
            } else {
                IntegratedExpectationMode::QuadratureFallback
            };
            Ok(IntegratedInverseLinkJet {
                mean,
                d1: d1.max(0.0),
                d2,
                d3,
                mode,
            })
        }
        LinkFunction::CLogLog => {
            validate_latent_cloglog_inputs(mu, sigma)?;
            Ok(integrated_cloglog_inverse_link_jet_controlled(
                quadctx, mu, sigma,
            ))
        }
        LinkFunction::LogLog | LinkFunction::Cauchit => {
            // The outer arm restricts `link` to exactly these two variants.
            let component = if matches!(link, LinkFunction::LogLog) {
                LinkComponent::LogLog
            } else {
                LinkComponent::Cauchit
            };
            let (mean, d1, d2, d3) = integrate_normal_ghq_adaptive(quadctx, mu, sigma, |x| {
                component_point_jet(component, x)
            });
            Ok(IntegratedInverseLinkJet {
                mean,
                d1,
                d2,
                d3,
                mode: if sigma <= 0.0 {
                    IntegratedExpectationMode::ExactClosedForm
                } else {
                    IntegratedExpectationMode::QuadratureFallback
                },
            })
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
        LinkFunction::Inverse | LinkFunction::InverseSquared => {
            reciprocal_link_posterior_jet(link, mu, sigma)
        }
    }
}

/// Accurate logistic-normal jet for the wide-σ regime (σ > `LOGIT_JET_GHQ_SIGMA_MAX`)
/// where Gauss–Hermite can no longer resolve the localized inverse-link
/// derivatives. `mean` and `d1` are taken verbatim from the scalar controlled
/// backend, so the scalar dispatcher and the jet return identical values at
/// wide σ (the #571 scalar-vs-jet disagreement is closed by construction rather
/// than by two independent quadratures merely agreeing to a tolerance). `d2`
/// and `d3` are the location-derivatives `E[sigmoid''(η)]`, `E[sigmoid'''(η)]`,
/// integrated by the same adaptive-Simpson rule the scalar path trusts as its
/// reference (resolved to ~1e-12 at every σ). The returned `mode` mirrors the
/// scalar backend's mode for the regime.
#[inline]
fn logit_wide_sigma_jet(mu: f64, sigma: f64) -> Result<IntegratedInverseLinkJet, EstimationError> {
    let scalar = logit_posterior_meanwith_deriv_exact(mu, sigma)?;
    let d2 = integrate_normal_adaptive(mu, sigma, |x| {
        component_point_jet(LinkComponent::Logit, x).2
    });
    let d3 = integrate_normal_adaptive(mu, sigma, |x| {
        component_point_jet(LinkComponent::Logit, x).3
    });
    Ok(IntegratedInverseLinkJet {
        mean: scalar.mean,
        d1: scalar.dmean_dmu.max(0.0),
        d2,
        d3,
        mode: scalar.mode,
    })
}

#[inline]
fn sas_point_jet(x: f64, epsilon: f64, log_delta: f64) -> (f64, f64, f64, f64) {
    let jet = sas_inverse_link_jet(x, epsilon, log_delta)
        .expect("normal quadrature nodes must be finite");
    (jet.mu, jet.d1, jet.d2, jet.d3)
}

#[inline]
fn beta_logistic_point_jet(x: f64, log_shape_center: f64, epsilon: f64) -> (f64, f64, f64, f64) {
    let jet = beta_logistic_inverse_link_jet(x, log_shape_center, epsilon);
    (jet.mu, jet.d1, jet.d2, jet.d3)
}

#[inline]
fn worse_integrated_expectation_mode(
    lhs: IntegratedExpectationMode,
    rhs: IntegratedExpectationMode,
) -> IntegratedExpectationMode {
    if lhs.rank() >= rhs.rank() { lhs } else { rhs }
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
            .unwrap_or_else(|error| {
                log::trace!(
                    "integrated logit jet at (mu={mu}, sigma={sigma}) fell back to GHQ: {error}"
                );
                integrated_logit_jet_ghq(ctx, mu, sigma)
            }),
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
                mode: if sigma <= 0.0 {
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
        crate::bail_invalid_estim!(
            "integrated mixture-link jet requires at least one blended component"
        );
    }
    if mixture_state.components.len() != mixture_state.pi.len() {
        crate::bail_invalid_estim!(
            "integrated mixture-link jet requires matching component and weight counts"
        );
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
        if jet.mode.rank() > mode.rank() {
            mode = jet.mode;
        }
        saw_positive_weight = true;
    }

    if !saw_positive_weight {
        crate::bail_invalid_estim!(
            "integrated mixture-link jet requires at least one positive component weight"
                .to_string(),
        );
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
        mode: if sigma <= 0.0 {
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
        mode: if sigma <= 0.0 {
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
///
/// Family and scale metadata are resolved atomically from `likelihood`; a
/// Gamma/Tweedie response without its required scalar, or any duplicated
/// family/metadata scalar that disagrees, is rejected before integration.
#[inline]
pub fn integrated_family_moments_jet(
    quadctx: &QuadratureContext,
    likelihood: &GlmLikelihoodSpec,
    eta: f64,
    se_eta: f64,
) -> Result<IntegratedMomentsJet, EstimationError> {
    if !(eta.is_finite() && (-700.0..=700.0).contains(&eta)) {
        crate::bail_invalid_estim!(
            "integrated moments eta must be finite and within [-700, 700]; got {eta}"
        );
    }
    let e = eta;
    let se = se_eta.max(0.0);
    // Pull parameterized link state from the spec itself; these helpers return
    // `None` for `InverseLink::Standard`, which is what every non-parameterized
    // dispatch arm expects.
    let resolved_scale = likelihood
        .resolved_scale()
        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
    let spec = &likelihood.spec;
    let mixture_link_state: Option<&MixtureLinkState> = spec.link.mixture_state();
    let sas_link_state: Option<&SasLinkState> = spec.link.sas_state();
    match &spec.response {
        ResponseFamily::Binomial => match &spec.link {
            InverseLink::Standard(StandardLink::Logit) => {
                let jet = integrated_inverse_link_jet(quadctx, LinkFunction::Logit, e, se)?;
                let mean = jet.mean;
                Ok(IntegratedMomentsJet {
                    mean,
                    variance: mean * (1.0 - mean),
                    d1: jet.d1,
                    d2: jet.d2,
                    d3: jet.d3,
                    mode: jet.mode,
                })
            }
            InverseLink::Standard(StandardLink::Probit) => {
                let jet = integrated_inverse_link_jet(quadctx, LinkFunction::Probit, e, se)?;
                let mean = jet.mean;
                Ok(IntegratedMomentsJet {
                    mean,
                    variance: mean * (1.0 - mean),
                    d1: jet.d1,
                    d2: jet.d2,
                    d3: jet.d3,
                    mode: jet.mode,
                })
            }
            InverseLink::Standard(StandardLink::CLogLog) => {
                let jet = integrated_inverse_link_jet(quadctx, LinkFunction::CLogLog, e, se)?;
                let mean = jet.mean;
                Ok(IntegratedMomentsJet {
                    mean,
                    variance: mean * (1.0 - mean),
                    d1: jet.d1,
                    d2: jet.d2,
                    d3: jet.d3,
                    mode: jet.mode,
                })
            }
            InverseLink::LatentCLogLog(_) => Err(EstimationError::InvalidInput(
                "Binomial+LatentCLogLog integrated moments require an explicit latent cloglog inverse-link state"
                    .to_string(),
            )),
            InverseLink::Sas(_) => {
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
                    variance: mean * (1.0 - mean),
                    d1: jet.d1,
                    d2: jet.d2,
                    d3: jet.d3,
                    mode: jet.mode,
                })
            }
            InverseLink::BetaLogistic(_) => {
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
                    variance: mean * (1.0 - mean),
                    d1: jet.d1,
                    d2: jet.d2,
                    d3: jet.d3,
                    mode: jet.mode,
                })
            }
            InverseLink::Mixture(state) => {
                let jet = integrated_mixture_jet(quadctx, e, se, &state)?;
                let mean = jet.mean;
                Ok(IntegratedMomentsJet {
                    mean,
                    variance: mean * (1.0 - mean),
                    d1: jet.d1,
                    d2: jet.d2,
                    d3: jet.d3,
                    mode: jet.mode,
                })
            }
            InverseLink::Standard(other) => Err(EstimationError::InvalidInput(format!(
                "Binomial response paired with unsupported standard link {other:?} for integrated moments"
            ))),
        },
        ResponseFamily::Gaussian => {
            let variance = resolved_scale
                .gaussian_phi()
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
            // Identity or the reciprocal link; the legality table admits no other.
            let jet = integrated_inverse_link_jet(quadctx, spec.link.link_function(), e, se)?;
            Ok(IntegratedMomentsJet {
                mean: jet.mean,
                variance,
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
                mode: jet.mode,
            })
        }
        ResponseFamily::Gamma | ResponseFamily::InverseGaussian
            if !matches!(spec.link, InverseLink::Standard(StandardLink::Log)) =>
        {
            // Reciprocal links: `1/μ` (Gamma) or `1/μ²` (inverse Gaussian).
            let jet = integrated_inverse_link_jet(quadctx, spec.link.link_function(), e, se)?;
            let mean = jet.mean;
            let variance = if matches!(spec.response, ResponseFamily::Gamma) {
                resolved_scale
                    .gamma_phi()
                    .map_err(|error| EstimationError::InvalidInput(error.to_string()))?
                    * mean
                    * mean
            } else {
                resolved_scale
                    .dispersion_phi()
                    .map_err(|error| EstimationError::InvalidInput(error.to_string()))?
                    * mean
                    * mean
                    * mean
            };
            if !(variance.is_finite() && variance >= 0.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "integrated {} variance is not representable: {variance:?}",
                    spec.response.name()
                )));
            }
            Ok(IntegratedMomentsJet {
                mean,
                variance,
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
                mode: jet.mode,
            })
        }
        ResponseFamily::StudentT { sigma, nu } => {
            // Identity link: the mean is exact; the response variance
            // `σ²ν/(ν−2)` exists only for `ν > 2`.
            if !(*nu > 2.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "Student-t response variance does not exist at nu={nu} (requires nu > 2)"
                )));
            }
            Ok(IntegratedMomentsJet {
                mean: e,
                variance: sigma * sigma * nu / (nu - 2.0),
                d1: 1.0,
                d2: 0.0,
                d3: 0.0,
                mode: IntegratedExpectationMode::ExactClosedForm,
            })
        }
        ResponseFamily::RoystonParmar => {
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
                variance: mean * (1.0 - mean),
                d1: -jet.d1,
                d2: -jet.d2,
                d3: -jet.d3,
                mode: jet.mode,
            })
        }
        ResponseFamily::Beta { .. } => {
            let precision = resolved_scale
                .beta_precision()
                .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
            let jet = integrated_inverse_link_jet(quadctx, LinkFunction::Logit, e, se)?;
            let mean = jet.mean;
            Ok(IntegratedMomentsJet {
                mean,
                variance: mean * (1.0 - mean) / (1.0 + precision),
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
                mode: jet.mode,
            })
        }
        ResponseFamily::Poisson
        | ResponseFamily::Tweedie { .. }
        | ResponseFamily::NegativeBinomial { .. }
        | ResponseFamily::Gamma
        | ResponseFamily::InverseGaussian => {
            // Log link. Log-normal MGF: E[exp(η)] = exp(e + s²/2)
            // d/de = exp(e + s²/2)   (same as the mean)
            // d²/de² = exp(e + s²/2)
            // d³/de³ = exp(e + s²/2)
            let s2 = se * se;
            let (mean, saturated) = safe_expwith_saturation(e + 0.5 * s2);
            // Observation-model variance at the integrated mean `m`, by family:
            //   Poisson:           Var = m                 (φ ≡ 1, pinned by mean)
            //   Tweedie(p):        Var = φ · m^p           (φ from `scale`)
            //   NegativeBinomial:  Var = m + m² / theta    (φ ≡ 1, overdispersion in theta)
            //   Gamma (shape k):   Var = m² / k = φ · m²   (k from `scale`, φ = 1/k)
            //   InverseGaussian:   Var = φ · m³            (φ from `scale`)
            // The Tweedie φ and Gamma shape are genuine free dispersion parameters
            // (see `LikelihoodScaleMetadata`), so they are read from `scale` rather
            // than assumed unit. A Gamma/Tweedie response whose `scale` does not
            // carry the dispersion is a metadata bug and is rejected, not silently
            // collapsed to φ = 1 (issue #953).
            let variance = match &spec.response {
                ResponseFamily::Poisson => mean,
                ResponseFamily::Tweedie { p } => {
                    let phi = resolved_scale
                        .tweedie_phi()
                        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
                    phi * mean.powf(*p)
                }
                ResponseFamily::NegativeBinomial { .. } => {
                    let theta = resolved_scale.negative_binomial_theta().map_err(|error| {
                        EstimationError::InvalidInput(error.to_string())
                    })?;
                    mean + mean * mean / theta
                }
                ResponseFamily::Gamma => {
                    let phi = resolved_scale
                        .gamma_phi()
                        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
                    phi * mean * mean
                }
                ResponseFamily::InverseGaussian => {
                    let phi = resolved_scale
                        .dispersion_phi()
                        .map_err(|error| EstimationError::InvalidInput(error.to_string()))?;
                    phi * mean * mean * mean
                }
                // Unreachable: this match arm is only entered for the five families
                // in the enclosing `Poisson | Tweedie | NegativeBinomial | Gamma |
                // InverseGaussian` pattern, all handled above.
                other => {
                    return Err(EstimationError::InvalidInput(format!(
                        "integrated log-normal moments reached unexpected family {other:?}"
                    )));
                }
            };
            if !(variance.is_finite() && variance >= 0.0) {
                return Err(EstimationError::InvalidInput(format!(
                    "integrated {} variance is not representable: {variance:?}",
                    spec.response.name()
                )));
            }
            Ok(IntegratedMomentsJet {
                mean,
                variance,
                d1: mean,
                d2: mean,
                d3: mean,
                mode: if saturated {
                    IntegratedExpectationMode::ControlledAsymptotic
                } else {
                    IntegratedExpectationMode::ExactClosedForm
                },
            })
        }
    }
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
    // Only `sigma <= 0` is a point mass for every integrand. A positive `sigma`
    // keeps the node sum, which carries the `sigma^2 f'' / 2` term the point mass
    // drops (#2469).
    if se_eta <= 0.0 {
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
    // If Z ~ N(mu, sigma^2), E[Phi(Z)] = Phi(mu / sqrt(1+sigma^2)).
    // This identity is exact at sigma=0 too, so there is no degenerate branch
    // and no reason to project mu. `hypot` keeps the scale finite for every
    // finite sigma. Once the Gaussian density underflows, all represented
    // derivatives are the exact zero tail limit; return before forming z^2.
    let s = sigma.hypot(1.0);
    let z = mu / s;
    let mean = gam_math::probability::normal_cdf(z);
    let pdf = gam_math::probability::normal_pdf(z);
    if pdf == 0.0 {
        return IntegratedInverseLinkJet {
            mean,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
            mode: IntegratedExpectationMode::ExactClosedForm,
        };
    }
    IntegratedInverseLinkJet {
        mean,
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
        mode: if sigma <= 0.0 {
            IntegratedExpectationMode::ExactClosedForm
        } else {
            IntegratedExpectationMode::QuadratureFallback
        },
    }
}

#[inline]
fn cloglog_inverse_link_controlled_values(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
    max_order: usize,
) -> ([f64; 6], IntegratedExpectationMode) {
    assert!(max_order <= 5);
    if sigma <= 1e-10 {
        let (mean, d1, d2, d3, d4, d5) = cloglog_point_jet5(mu);
        return (
            [mean, d1, d2, d3, d4, d5],
            IntegratedExpectationMode::ExactClosedForm,
        );
    }

    let (k, log_k0, mode) = latent_cloglog_kernel_terms(ctx, mu, sigma, max_order);
    let mut values = [0.0; 6];
    values[0] = if log_k0.is_finite() {
        -log_k0.exp_m1()
    } else {
        1.0
    };
    values[1] = k[1].max(0.0);
    if sigma > CLOGLOG_JET_MOMENT_SIGMA_MAX {
        if max_order >= 2 {
            values[2] = integrate_normal_adaptive(mu, sigma, |x| cloglog_point_jet5(x).2);
        }
        if max_order >= 3 {
            values[3] = integrate_normal_adaptive(mu, sigma, |x| cloglog_point_jet5(x).3);
        }
        if max_order >= 4 {
            values[4] = integrate_normal_adaptive(mu, sigma, |x| cloglog_point_jet5(x).4);
        }
        if max_order >= 5 {
            values[5] = integrate_normal_adaptive(mu, sigma, |x| cloglog_point_jet5(x).5);
        }
        return (
            values,
            worse_integrated_expectation_mode(mode, IntegratedExpectationMode::QuadratureFallback),
        );
    }
    if max_order >= 2 {
        values[2] = k[1] - k[2];
    }
    if max_order >= 3 {
        values[3] = k[1] - 3.0 * k[2] + k[3];
    }
    if max_order >= 4 {
        values[4] = k[1] - 7.0 * k[2] + 6.0 * k[3] - k[4];
    }
    if max_order >= 5 {
        values[5] = k[1] - 15.0 * k[2] + 25.0 * k[3] - 10.0 * k[4] + k[5];
    }
    (values, mode)
}

#[inline]
pub(crate) fn latent_cloglog_inverse_link_jet5_controlled(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> IntegratedInverseLinkJet5 {
    let (values, mode) = cloglog_inverse_link_controlled_values(ctx, mu, sigma, 5);
    IntegratedInverseLinkJet5 {
        mean: values[0],
        d1: values[1],
        d2: values[2],
        d3: values[3],
        d4: values[4],
        d5: values[5],
        mode,
    }
}

/// Fifth-order latent-cloglog inverse-link jet.
///
/// Relocated here from `families::survival::lognormal_kernel` (#1135): this is
/// the public face of the latent-cloglog link jet, and its analytic backend
/// (`latent_cloglog_inverse_link_jet5_controlled`) already lives in this
/// quadrature module. Hosting the wrapper here lets the `solver` link layer
/// (`mixture_link`, `pirls`) name it via `crate::quadrature::*` instead of
/// importing *up* into `families::survival`. `lognormal_kernel` re-exports these
/// names so the in-family callers keep working.
#[derive(Clone, Copy, Debug)]
pub struct LatentCLogLogJet5 {
    pub mean: f64,
    pub d1: f64,
    pub d2: f64,
    pub d3: f64,
    pub d4: f64,
    pub d5: f64,
    pub mode: IntegratedExpectationMode,
}

pub fn latent_cloglog_jet5(
    quadctx: &QuadratureContext,
    eta: f64,
    sigma: f64,
) -> Result<LatentCLogLogJet5, EstimationError> {
    validate_latent_cloglog_inputs(eta, sigma)?;
    // Authoritative latent cloglog backend:
    //
    // - mean through d5 are all derived from the same lognormal-Laplace kernel
    //   terms K_{k,1}(eta, sigma),
    // - every derivative order uses the same routed analytic kernel backend.
    let jet = latent_cloglog_inverse_link_jet5_controlled(quadctx, eta, sigma);
    Ok(LatentCLogLogJet5 {
        mean: jet.mean,
        d1: jet.d1,
        d2: jet.d2,
        d3: jet.d3,
        d4: jet.d4,
        d5: jet.d5,
        mode: jet.mode,
    })
}

#[inline]
pub fn latent_cloglog_inverse_link_jet(
    quadctx: &QuadratureContext,
    eta: f64,
    sigma: f64,
) -> Result<IntegratedInverseLinkJet, EstimationError> {
    let jet = latent_cloglog_jet5(quadctx, eta, sigma)?;
    Ok(IntegratedInverseLinkJet {
        mean: jet.mean,
        d1: jet.d1,
        d2: jet.d2,
        d3: jet.d3,
        mode: jet.mode,
    })
}

#[inline]
fn integrated_cloglog_inverse_link_jet_controlled(
    ctx: &QuadratureContext,
    mu: f64,
    sigma: f64,
) -> IntegratedInverseLinkJet {
    let (values, mode) = cloglog_inverse_link_controlled_values(ctx, mu, sigma, 3);
    IntegratedInverseLinkJet {
        mean: values[0],
        d1: values[1],
        d2: values[2],
        d3: values[3],
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
        // Carry the survival S(μ + kσ², σ) as a log so the kernel
        //   K_{k,1} = exp(kμ + ½k²σ²) · S(μ + kσ², σ)
        // keeps its true magnitude when S underflows in value space: at large σ
        // the k=1 shifted location μ + σ² drives S below the f64 floor, and the
        // old value-space `survival <= 0.0 → 0` collapse zeroed K_{1,1} (the
        // IRLS working slope), even though the huge exp(½σ²) prefix makes the
        // product finite and O(1) (#798).
        let (log_survival, term_mode) =
            cloglog_log_survival_term_controlled(ctx, shifted_mu, sigma);
        mode = worse_integrated_expectation_mode(mode, term_mode);

        let log_value = kf * mu + 0.5 * kf * kf * sigma2 + log_survival;
        if order == 0 {
            log_k0 = log_value;
        }
        if !log_value.is_finite() {
            *out = 0.0;
            continue;
        }
        let upper = if order == 0 {
            1.0
        } else {
            let k_over_e = kf / std::f64::consts::E;
            k_over_e.powf(kf)
        };
        *out = safe_exp(log_value).clamp(0.0, upper);
    }

    (k, log_k0, mode)
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
    // The moderate-sigma 31-point band was widened (1.0 → 0.5) after the
    // Logit jet started feeding d2/d3 through the same Hermite rule: at
    // σ ≈ 0.8, 21-pt d2 on the logistic-normal reaches only ~2.5e-10 rel
    // vs 31-pt at ~3e-13, and several downstream tests pin to 1e-10.
    if max_sd.is_finite() && max_sd > 2.5 {
        51
    } else if max_sd.is_finite() && max_sd > 0.5 {
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

/// Stack-allocated Cholesky factor of a `D x D` symmetric positive semidefinite
/// covariance, with no heap allocation: per-row GHQ runs it once per observation.
///
/// Returns the lower-triangular `L` (strict upper triangle zero) with
/// `L L^T = cov`, or `None` when `cov` is non-finite or indefinite beyond
/// rounding. Every entry is the accumulation `cov_ij − Σ_{k<j} l_ik·l_jk` of
/// `j + 1` rounded terms, known only to `gam_linalg::roundoff::accumulation_band`
/// of their absolute sum. A pivot inside its band is the exact zero of a rank-deficient
/// covariance: its column stays zero and the Gaussian has no spread along it.
/// An exact PSD matrix has a zero column below a zero pivot, so a Schur entry
/// there outside its own band, or a pivot below minus its band, is refused as
/// indefinite. No jitter is added, and a positive definite `cov` factors
/// bit-for-bit as the textbook inner loop does (#2469).
#[inline]
fn cholesky_static<const D: usize>(cov: &[[f64; D]; D]) -> Option<[[f64; D]; D]> {
    if D == 0 {
        return None;
    }
    let mut l = [[0.0_f64; D]; D];
    for i in 0..D {
        for j in 0..=i {
            let mut sum = cov[i][j];
            let mut absolute_sum = cov[i][j].abs();
            for k in 0..j {
                let product = l[i][k] * l[j][k];
                sum -= product;
                absolute_sum += product.abs();
            }
            let band = gam_linalg::roundoff::accumulation_band(j + 1, absolute_sum);
            if i == j {
                if !sum.is_finite() || sum < -band {
                    return None;
                }
                l[i][j] = if sum > band { sum.sqrt() } else { 0.0 };
            } else if l[j][j] > 0.0 {
                l[i][j] = sum / l[j][j];
            } else if !(sum.abs() <= band) {
                return None;
            }
        }
    }
    Some(l)
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

    // Sanitize variances on the stack (clamp negative diagonal to 0),
    // then run the stack-allocated semidefinite Cholesky. This avoids the
    // `Vec<Vec<f64>>` per-row allocation that previously serialized
    // through the global allocator inside parallel workers.
    let mut cov_arr = cov;
    for i in 0..D {
        cov_arr[i][i] = cov_arr[i][i].max(0.0);
    }
    let Some(l) = cholesky_static::<D>(&cov_arr) else {
        return Ok(None);
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

/// Where a bivariate normal `N(mu, cov)` puts its mass when `cov` may be
/// singular in floating point, as [`normal_expectation_2d_projected_result`]
/// integrates it.
///
/// The 2-D rule runs exactly when its Cholesky factor exists without jitter:
/// the pivots `a` and `b − (c/√a)²` of `cov = [[a, c], [c, b]]`, formed as
/// `cholesky_static` forms them, are both positive. Any other `cov` is rank one
/// in floating point or indefinite, and the nearest positive semidefinite
/// covariance keeps only its major eigenpair: variance `m + r` with
/// `m = (a + b)/2` and `r = hypot((a − b)/2, c)`, along the major eigenvector.
/// A covariance with no positive eigenvalue is a point mass.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BivariateNormalSupport {
    /// Both Cholesky pivots are positive: the law spreads over the plane.
    Plane,
    /// Only the major eigenpair carries variance: `mu + t·axis` with
    /// `t ~ N(0, variance)` and `axis` a unit vector.
    Axis { axis: [f64; 2], variance: f64 },
    /// No positive eigenvalue: all mass at `mu`.
    Point,
}

impl BivariateNormalSupport {
    /// The support of `N(·, cov)` for a finite symmetric `cov`.
    pub fn of(cov: [[f64; 2]; 2]) -> Self {
        let (a, b, c) = (cov[0][0], cov[1][1], cov[1][0]);
        if a > 0.0 {
            let below = c / a.sqrt();
            if b - below * below > 0.0 {
                return Self::Plane;
            }
        }
        let half_difference = 0.5 * a - 0.5 * b;
        let radius = half_difference.hypot(c);
        let major = 0.5 * a + 0.5 * b + radius;
        if major <= 0.0 {
            return Self::Point;
        }
        // The major eigenvector is `(r + h, c)` or `(c, r − h)` with `h = (a − b)/2`;
        // take whichever adds rather than cancels. It is nonzero here: `r + h = 0`
        // with `h ≥ 0` forces `h = c = 0`, so `a = b = m > 0` and both pivots were
        // positive.
        let (u0, u1) = if half_difference >= 0.0 {
            (radius + half_difference, c)
        } else {
            (c, radius - half_difference)
        };
        let length = u0.hypot(u1);
        Self::Axis {
            axis: [u0 / length, u1 / length],
            variance: major,
        }
    }
}

/// `E[f(x₀, x₁)]` under `N(mu, cov)` for a finite symmetric `cov`, integrated
/// over every direction in which `cov` carries variance: adaptive 2-D
/// Gauss–Hermite on [`BivariateNormalSupport::Plane`], the 1-D rule along the
/// major axis on [`BivariateNormalSupport::Axis`], and `f(mu)` on a point mass.
pub fn normal_expectation_2d_projected_result<F, R, E>(
    ctx: &QuadratureContext,
    mu: [f64; 2],
    cov: [[f64; 2]; 2],
    f: F,
) -> Result<R, E>
where
    F: Fn(f64, f64) -> Result<R, E>,
    R: GhqValue,
{
    match BivariateNormalSupport::of(cov) {
        BivariateNormalSupport::Plane => {
            normal_expectation_nd_adaptive_result::<2, _, R, E>(ctx, mu, cov, 21, |x| f(x[0], x[1]))
        }
        BivariateNormalSupport::Axis { axis, variance } => {
            normal_expectation_nd_adaptive_result::<1, _, R, E>(ctx, [0.0], [[variance]], 21, |t| {
                f(mu[0] + axis[0] * t[0], mu[1] + axis[1] * t[0])
            })
        }
        BivariateNormalSupport::Point => f(mu[0], mu[1]),
    }
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
pub(crate) fn probit_posterior_mean(eta: f64, se_eta: f64) -> f64 {
    let denom = (1.0 + se_eta * se_eta).sqrt();
    gam_math::probability::normal_cdf(eta / denom)
}

#[inline]
pub fn logit_posterior_meanvariance(ctx: &QuadratureContext, eta: f64, se_eta: f64) -> (f64, f64) {
    let (m1, m2) = integrate_normal_ghq_adaptive(ctx, eta, se_eta, |x| {
        let p = sigmoid(x);
        (p, p * p)
    });
    let m1 = m1.clamp(0.0, 1.0);
    let m2 = m2.clamp(0.0, 1.0);
    (m1, (m2 - m1 * m1).max(0.0))
}

#[inline]
pub fn probit_posterior_meanvariance(ctx: &QuadratureContext, eta: f64, se_eta: f64) -> (f64, f64) {
    let m1 = probit_posterior_mean(eta, se_eta);
    let m2 = integrate_normal_ghq_adaptive(ctx, eta, se_eta, |x| {
        let p = gam_math::probability::normal_cdf(x);
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
    // Degenerate sigma: use cloglog_mean_exact directly.
    if !(eta.is_finite() && se_eta.is_finite()) || se_eta <= CLOGLOG_SIGMA_DEGENERATE {
        return (cloglog_mean_exact(eta), 0.0);
    }
    let (survival, _) = cloglog_survival_term_controlled(ctx, eta, se_eta);
    let (survival_sq, _) = cloglog_survivalsecond_moment_controlled(ctx, eta, se_eta);
    let mean = cloglog_mean_from_survival(survival);
    let variance = (survival_sq - survival * survival).max(0.0);
    (mean, variance)
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

/// Standard sigmoid function with numerical stability.
#[inline]
fn sigmoid(x: f64) -> f64 {
    let x_clamped = x.clamp(-QUADRATURE_EXP_LOG_MAX, QUADRATURE_EXP_LOG_MAX);
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

#[inline]
pub(crate) fn cloglog_point_jet5(t: f64) -> (f64, f64, f64, f64, f64, f64) {
    if t.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    let et = safe_exp(t);

    (
        -(-et).exp_m1(),
        cloglog_stable_poly_times_exp_neg(et, &[0.0, 1.0]),
        cloglog_stable_poly_times_exp_neg(et, &[0.0, 1.0, -1.0]),
        cloglog_stable_poly_times_exp_neg(et, &[0.0, 1.0, -3.0, 1.0]),
        cloglog_stable_poly_times_exp_neg(et, &[0.0, 1.0, -7.0, 6.0, -1.0]),
        cloglog_stable_poly_times_exp_neg(et, &[0.0, 1.0, -15.0, 25.0, -10.0, 1.0]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use gam_spec::LikelihoodSpec;

    fn even_moment_exp_neg_x2(power: usize) -> f64 {
        assert!(power.is_multiple_of(2));
        let m = power / 2;
        let mut odd_double_factorial = 1.0_f64;
        for k in 0..m {
            odd_double_factorial *= (2 * k + 1) as f64;
        }
        odd_double_factorial * std::f64::consts::PI.sqrt() / 2.0_f64.powi(m as i32)
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
    fn test_matches_abramowitz_stegun_7_point_gauss_hermite_constants() {
        // Abramowitz & Stegun 25.4, 7-point Gauss-Hermite rule for the
        // physicist's weight exp(-x^2). This pins both the Jacobi matrix and
        // the eigenvector orientation used for Golub-Welsch weights.
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
    fn test_logit_posterior_derivative_remains_positive_in_positive_tail() {
        let eta = 20.0;
        let se = 0.0;
        let (_, dmu) = logit_posterior_meanwith_deriv(eta, se)
            .expect("logit posterior mean derivative should evaluate");
        assert!(dmu > 0.0);
        assert!(
            dmu < 1e-6,
            "positive-tail derivative should stay tiny but nonzero, got {dmu}"
        );
    }

    #[test]
    fn test_probit_posterior_mean_reduces_to_map_atzero_se() {
        let eta = 1.25;
        let p = probit_posterior_mean(eta, 0.0);
        let map = gam_math::probability::normal_cdf(eta);
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
        let pdf = gam_math::probability::normal_pdf(z);
        assert_relative_eq!(
            out.mean,
            gam_math::probability::normal_cdf(z),
            epsilon = 1e-12
        );
        assert_relative_eq!(out.d1, pdf / s, epsilon = 1e-12);
        assert_relative_eq!(out.d2, -z * pdf / (s * s), epsilon = 1e-12);
        assert_relative_eq!(out.d3, (z * z - 1.0) * pdf / (s * s * s), epsilon = 1e-12);
    }

    #[test]
    fn test_integrated_logit_jet_matches_central_differences() {
        // At σ ≤ 1 the logit jet integrates all four orders by Gauss-Hermite.
        // The contract is VALUE accuracy of the mean and its μ-derivatives, so
        // assert those directly against an independent high-resolution Simpson
        // reference.
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
        let (ref_mean, ref_d1, ref_d2, ref_d3) = logit_reference_jet_highres_simpson(mu, sigma);
        assert_relative_eq!(out.mean, ref_mean, epsilon = 1e-11, max_relative = 1e-10);
        assert_relative_eq!(out.d1, ref_d1, epsilon = 1e-11, max_relative = 1e-10);
        assert_relative_eq!(out.d2, ref_d2, epsilon = 1e-11, max_relative = 1e-10);
        assert_relative_eq!(out.d3, ref_d3, epsilon = 1e-11, max_relative = 1e-10);
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
    fn test_integrated_cloglog_wide_sigma_d3_matches_simpson_and_d2_slope() {
        let ctx = QuadratureContext::new();
        let cases = [(0.0, 4.0), (-1.0, 4.0), (2.0, 3.0), (3.0, 3.0)];
        let h = 1e-4;

        for (mu, sigma) in cases {
            let out = integrated_inverse_link_jet(&ctx, LinkFunction::CLogLog, mu, sigma)
                .expect("wide-sigma cloglog integrated jet should evaluate");
            let reference = cloglog_reference_jet_highres_simpson(mu, sigma);
            let plus = integrated_inverse_link_jet(&ctx, LinkFunction::CLogLog, mu + h, sigma)
                .expect("wide-sigma cloglog integrated jet should evaluate");
            let minus = integrated_inverse_link_jet(&ctx, LinkFunction::CLogLog, mu - h, sigma)
                .expect("wide-sigma cloglog integrated jet should evaluate");
            let d3fd = (plus.d2 - minus.d2) / (2.0 * h);

            assert_eq!(out.mode, IntegratedExpectationMode::QuadratureFallback);
            assert_relative_eq!(out.mean, reference.0, epsilon = 4e-8, max_relative = 4e-8);
            assert_relative_eq!(out.d1, reference.1, epsilon = 4e-8, max_relative = 4e-8);
            assert_relative_eq!(out.d2, reference.2, epsilon = 2e-9, max_relative = 2e-7);
            assert_relative_eq!(out.d3, reference.3, epsilon = 2e-9, max_relative = 2e-7);
            assert_relative_eq!(out.d3, d3fd, epsilon = 2e-7, max_relative = 4e-5);
        }
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
        // Value and μ-derivative (d/dμ E[sigmoid] = E[sigmoid']) of the series
        // against an independent high-resolution Simpson reference.
        let out = logit_posterior_meanwith_deriv_exact(1.1, 0.8).expect("exact logit");
        let (ref_mean, ref_d1, _, _) = logit_reference_jet_highres_simpson(1.1, 0.8);
        assert_relative_eq!(out.mean, ref_mean, epsilon = 1e-11, max_relative = 1e-10);
        assert!(out.dmean_dmu > 0.0);
        assert_relative_eq!(out.dmean_dmu, ref_d1, epsilon = 1e-11, max_relative = 1e-10);
    }

    #[test]
    fn test_logit_small_sigma_returns_quadrature_truth_2623() {
        // A second-order heat-kernel truncation has O(sigma^4) error, enough to
        // reverse a near-tied posterior-probability pair in issue #2623. The
        // series carries no small-sigma truncation: it must match an
        // independent quadrature to working precision here too.
        for &(mu, sigma) in &[
            (-3.0, 0.05),
            (-0.5, 0.10),
            (0.0, 0.20),
            (0.5, 0.24),
            (3.0, 0.15),
        ] {
            let out = logit_posterior_meanwith_deriv_exact(mu, sigma).expect("exact logit");
            let (ref_mean, ref_d1, _, _) = logit_reference_jet_highres_simpson(mu, sigma);
            assert_eq!(out.mode, IntegratedExpectationMode::ExactSpecialFunction);
            assert_relative_eq!(out.mean, ref_mean, epsilon = 1e-12, max_relative = 1e-11);
            assert_relative_eq!(out.dmean_dmu, ref_d1, epsilon = 1e-12, max_relative = 1e-11);
        }
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
            return (cloglog_mean_exact(mu), cloglog_mean_d1_exact(mu));
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
            inv_sqrt_2pi * (-0.5 * z * z).exp() * cloglog_mean_d1_exact(eta)
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
    /// mathematically independent of the moment series under test.
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

    fn cloglog_reference_jet_highres_simpson(mu: f64, sigma: f64) -> (f64, f64, f64, f64) {
        let z_max = 14.0;
        let n_intervals = 16384;
        let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
        let phi = |z: f64| inv_sqrt_2pi * (-0.5 * z * z).exp();
        let mean = simpson_integrate(-z_max, z_max, n_intervals, |z| {
            let eta = mu + sigma * z;
            let (g, _, _, _, _, _) = cloglog_point_jet5(eta);
            phi(z) * g
        });
        let d1 = simpson_integrate(-z_max, z_max, n_intervals, |z| {
            let eta = mu + sigma * z;
            let (_, g1, _, _, _, _) = cloglog_point_jet5(eta);
            phi(z) * g1
        });
        let d2 = simpson_integrate(-z_max, z_max, n_intervals, |z| {
            let eta = mu + sigma * z;
            let (_, _, g2, _, _, _) = cloglog_point_jet5(eta);
            phi(z) * g2
        });
        let d3 = simpson_integrate(-z_max, z_max, n_intervals, |z| {
            let eta = mu + sigma * z;
            let (_, _, _, g3, _, _) = cloglog_point_jet5(eta);
            phi(z) * g3
        });
        (mean, d1, d2, d3)
    }

    #[test]
    fn test_cloglog_negative_tail_matches_mathematical_target() {
        let ctx = QuadratureContext::new();
        let mu = -40.0;
        let sigma = 0.1;
        let out = cloglog_posterior_meanwith_deriv_controlled(&ctx, mu, sigma);
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
            cloglog_mean_d1_exact(mu),
            epsilon = 1e-30,
            max_relative = 1e-15
        );
    }

    #[test]
    fn test_degenerate_probit_jet_is_exact_beyond_former_clamp() {
        let mu = -30.1;
        let probit = integrated_probit_jet(mu, 0.0);
        let pdf = gam_math::probability::normal_pdf(mu);
        assert!(
            pdf > 0.0,
            "test point must have a represented Gaussian tail"
        );
        assert_eq!(probit.mean, gam_math::probability::normal_cdf(mu));
        assert_eq!(probit.d1, pdf);
        assert_eq!(probit.d2, -mu * pdf);
        assert_eq!(probit.d3, (mu * mu - 1.0) * pdf);

        // eta = -710 is where the NAIVE `1/(1 + exp(-eta))` would overflow (`exp`
        // tops out near +709.78, so `exp(710)` is `inf` and the naive quotient
        // collapses to a flat zero jet). It is NOT where the logistic tail stops
        // being representable: `exp(-710) = 4.476e-309` is a perfectly good
        // subnormal, and the f64 tail survives to roughly eta = -745. The stable
        // implementation returns that exact tail, and `canonicalzero` keeps it on
        // purpose — "a nonzero subnormal is still a representable derivative and
        // must survive: replacing it by zero creates an artificial constant tail
        // and a kink at MIN_POSITIVE". Asserting zero here would have pinned the
        // overflow artifact this function exists to avoid.
        //
        // At this eta, `t = exp(eta)` is far below machine epsilon, so `1 + t == 1`
        // exactly and the whole jet collapses onto `t`:
        //     mu = t/(1+t) = t,  d1 = mu(1-mu) = t,
        //     d2 = d1(1-2mu)   = t,  d3 = d1(1-6mu+6mu^2) = t.
        let tail = (-710.0_f64).exp();
        assert!(
            tail > 0.0 && tail < f64::MIN_POSITIVE,
            "eta=-710 must sit in the subnormal tail, not underflow"
        );
        let logit = component_point_jet(LinkComponent::Logit, -710.0);
        assert_eq!(logit.0, tail);
        assert_eq!(logit.1, tail);
        assert_eq!(logit.2, tail);
        assert_eq!(logit.3, tail);

        // Only PAST the representable tail may the jet legitimately vanish.
        assert_eq!(
            (-750.0_f64).exp(),
            0.0,
            "eta=-750 must underflow f64 for this arm to mean anything"
        );
        let underflowed = component_point_jet(LinkComponent::Logit, -750.0);
        assert_eq!(underflowed.1, 0.0);
        assert_eq!(underflowed.2, 0.0);
        assert_eq!(underflowed.3, 0.0);
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

    /// `(mu, sigma, ln E[1 - exp(-exp(eta))], ln E[exp(eta - exp(eta))])` for
    /// `eta ~ N(mu, sigma^2)`: the log of the integrated cloglog mean and of its
    /// location derivative, from 50-digit mpmath quadrature. Each integrand is
    /// log-concave, so the reference places 121 breakpoints at +-60 local widths
    /// around its mode (spec-lead's truth3.py, #2469).
    const CLOGLOG_INTEGRATED_REFERENCE: &[(f64, f64, f64, f64)] = &[
        (8.0, 1.0, -3.989_793_784_967_362_000_2e-12, -24.416_460_656_908_228),
        (-35.0, 5.9, -18.211_122_410_899_579, -18.339_200_225_246_829),
        (6.0, 0.025, 0.0, -356.570_584_783_525_03),
        (2.5, 0.24, -6.057_951_710_870_456_1e-5, -7.646_194_685_271_555_2),
        (-60.0, 7.9, -29.704_544_253_336_292, -29.831_468_173_064_908),
        (-45.0, 7.9, -17.835_241_966_258_719, -18.200_130_943_040_836),
        (12.0, 0.4, 0.0, -251.599_626_124_600_76),
        (0.0, 1.0, -0.480_872_829_174_680_3, -1.351_482_882_134_652_8),
        (-3.0, 0.01, -3.024_743_917_485_727_3, -3.049_744_413_007_874_4),
        (3.0, 0.26, -9.593_995_695_714_014_7e-7, -11.529_830_057_475_451),
        (-35.0, 0.005, -34.999_987_500_000_000, -34.999_987_500_000_001),
    ];

    /// The integrated cloglog mean and its location derivative come off the one
    /// log-survival surface at every `(mu, sigma)`, at that surface's own
    /// accuracy: `1e-13` of the log's magnitude (floored at 1), the bar
    /// `log_survival_matches_a_high_precision_reference_2714` holds `ln S` to. The
    /// log is the consumed axis: the mean and derivative enter a log likelihood
    /// and the Fisher weight `d1^2 / (mean (1 - mean))`, so each must be right
    /// against its own magnitude, however far below 1 it sits.
    ///
    /// Rows the replaced value-space ladder got wrong (#2469, probe 1336291 at
    /// 6cceb4e517): `(8, 1)` its derivative by 26%; `(-35, 5.9)` its mean by
    /// 336%; `(6, 0.025)` its derivative by seven orders; `(12, 0.4)` its
    /// derivative as exactly zero; `(-60, 7.9)` / `(-45, 7.9)` its tail mean and
    /// derivative; `(2.5, 0.24)` / `(3, 0.26)` the positive-saturation band; and
    /// `(0, 1)` at 2e-13 of the mean. `(-3, 0.01)` and `(-35, 0.005)` are rows it
    /// had right.
    #[test]
    fn integrated_cloglog_reads_the_one_log_survival_surface_2469() {
        let ctx = QuadratureContext::new();
        let mut worst = 0.0_f64;
        for &(mu, sigma, ln_mean, ln_d1) in CLOGLOG_INTEGRATED_REFERENCE {
            let out = cloglog_posterior_meanwith_deriv_controlled(&ctx, mu, sigma);
            assert!(
                out.mean > 0.0 && out.dmean_dmu > 0.0,
                "cloglog({mu}, {sigma}): mean {:e} and derivative {:e} must be positive",
                out.mean,
                out.dmean_dmu
            );
            let mean_error = (out.mean.ln() - ln_mean).abs() / ln_mean.abs().max(1.0);
            let d1_error = (out.dmean_dmu.ln() - ln_d1).abs() / ln_d1.abs().max(1.0);
            worst = worst.max(mean_error).max(d1_error);
            assert!(
                mean_error <= 1.0e-13 && d1_error <= 1.0e-13,
                "cloglog({mu}, {sigma}): ln mean {:.17e} against {ln_mean:.17e} (error \
                 {mean_error:.3e}), ln d1 {:.17e} against {ln_d1:.17e} (error {d1_error:.3e})",
                out.mean.ln(),
                out.dmean_dmu.ln()
            );
        }
        assert!(worst > 0.0, "every row reproduced bit-exactly: the table tests nothing");
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
    fn test_logit_dispatch_is_exact_deep_in_the_tail() {
        // (35, 1) used to leave the series for a tail asymptotic. The CRVZ
        // series has no regime split: the same pass is exact here, and the
        // complement is the leading moment `E[e^{−η}] = e^{−μ+σ²/2}` to within
        // the next term `e^{−2μ+2σ²}`.
        let ctx = QuadratureContext::new();
        let out = integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, 35.0, 1.0)
            .expect("logit integrated inverse-link moments should evaluate");
        assert_eq!(out.mode, IntegratedExpectationMode::ExactSpecialFunction);
        assert_relative_eq!(out.mean, 1.0 - (-34.5_f64).exp(), max_relative = 1e-15);
        assert_relative_eq!(out.dmean_dmu, (-34.5_f64).exp(), max_relative = 1e-13);
    }

    #[test]
    fn test_logit_dispatch_is_exact_in_moderate_regime() {
        // (1.1, 0.8) is where the old erfcx series could not certify and the
        // dispatcher routed to quadrature. The CRVZ series needs no fallback.
        let ctx = QuadratureContext::new();
        let out = integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, 1.1, 0.8)
            .expect("logit integrated inverse-link moments should evaluate");
        assert_eq!(out.mode, IntegratedExpectationMode::ExactSpecialFunction);
        assert!(out.mean.is_finite());
        assert!(out.dmean_dmu.is_finite());
        assert!(out.dmean_dmu >= 0.0);
        let (ref_mean, ref_d1, _, _) = logit_reference_jet_highres_simpson(1.1, 0.8);
        assert_relative_eq!(out.mean, ref_mean, epsilon = 1e-11, max_relative = 1e-10);
        assert_relative_eq!(out.dmean_dmu, ref_d1, epsilon = 1e-11, max_relative = 1e-10);
    }

    #[test]
    fn test_logit_dispatch_large_sigma_is_exact_not_monahan() {
        // Regression for #571. At (μ=0.5, σ=20) the old code returned the
        // Monahan–Stefanski probit Φ(μκ), wrong by ~6e-3 absolute, as a
        // trusted `Ok`. The CRVZ series is exact at every σ; assert the value
        // against an independent reference and that it is not Monahan's.
        let ctx = QuadratureContext::new();
        let out = integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, 0.5, 20.0)
            .expect("logit integrated inverse-link moments should evaluate");
        assert_eq!(out.mode, IntegratedExpectationMode::ExactSpecialFunction);
        let (ref_mean, ref_d1, _, _) = logit_reference_jet_highres_simpson(0.5, 20.0);
        assert_relative_eq!(out.mean, ref_mean, epsilon = 1e-9, max_relative = 1e-7);
        assert_relative_eq!(out.dmean_dmu, ref_d1, epsilon = 1e-9, max_relative = 1e-7);
        // The discarded Monahan value differs in the third decimal place; pin
        // that the dispatcher is NOT returning it.
        let kappa = (1.0 + std::f64::consts::PI * 20.0 * 20.0 / 8.0)
            .sqrt()
            .recip();
        let monahan_mean = gam_math::probability::normal_cdf(0.5 * kappa);
        assert!(
            (out.mean - monahan_mean).abs() > 1e-3,
            "dispatcher must not return the inaccurate Monahan mean {monahan_mean}; got {}",
            out.mean
        );
    }

    #[test]
    fn test_logit_public_entry_point_is_the_exact_series() {
        // `logit_posterior_meanwith_deriv` (used by the multinomial posterior)
        // and the link dispatcher are the same single pass, bit for bit.
        let ctx = QuadratureContext::new();
        for &(mu, sigma) in &[(1.1, 0.8), (-4.0, 0.2), (0.0, 7.0), (25.0, 3.0)] {
            let public = logit_posterior_meanwith_deriv(mu, sigma).expect("public logit moments");
            let dispatched =
                integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, mu, sigma)
                    .expect("dispatched logit moments");
            assert_eq!(public.0.to_bits(), dispatched.mean.to_bits());
            assert_eq!(public.1.to_bits(), dispatched.dmean_dmu.to_bits());
        }
    }

    #[test]
    fn test_logit_dispatch_derivative_correct_at_mu_zero_small_sigma() {
        // Regression for #572. An old erfcx branch at μ=0 truncated on a
        // mean-only cutoff and reported dmean_dmu ≈ 0.58 at (0, 0.3), a factor
        // ~2.4 too large and physically impossible, since sigmoid'(0)=0.25 and
        // averaging over a Gaussian can only shrink it.
        let ctx = QuadratureContext::new();
        for &(mu, sigma) in &[(0.0, 0.3), (0.0, 0.4), (0.0, 0.5)] {
            let out =
                integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, mu, sigma)
                    .expect("logit integrated inverse-link moments should evaluate");
            // Mean is exactly 0.5 by symmetry at μ=0.
            assert_relative_eq!(out.mean, 0.5, epsilon = 1e-10);
            // Hard physical ceiling: E[sigmoid'(η)] ≤ sigmoid'(0) = 0.25.
            assert!(
                out.dmean_dmu <= 0.25 + 1e-9,
                "E[sigmoid'] must not exceed 0.25 at (μ={mu}, σ={sigma}); got {}",
                out.dmean_dmu
            );
            let (_, ref_d1, _, _) = logit_reference_jet_highres_simpson(mu, sigma);
            assert_relative_eq!(out.dmean_dmu, ref_d1, epsilon = 1e-9, max_relative = 1e-6);
        }
    }

    #[test]
    fn test_logit_exact_series_is_self_certified() {
        // The production path has no second evaluator behind it, so the
        // series must be accurate by itself everywhere, including (0, 0.3),
        // the #572 point the old erfcx branch had to reject.
        for &(mu, sigma) in &[
            (8.0, 1.0),
            (10.0, 1.0),
            (15.0, 2.0),
            (0.0, 0.3),
            (0.4, 0.05),
        ] {
            let out = logit_posterior_meanwith_deriv_exact(mu, sigma).expect("series evaluates");
            assert_eq!(out.mode, IntegratedExpectationMode::ExactSpecialFunction);
            let (ref_mean, ref_d1, _, _) = logit_reference_jet_highres_simpson(mu, sigma);
            assert_relative_eq!(out.mean, ref_mean, epsilon = 1e-12, max_relative = 1e-11);
            assert_relative_eq!(out.dmean_dmu, ref_d1, epsilon = 1e-12, max_relative = 1e-11);
        }
    }

    #[test]
    fn test_logistic_normal_series_degree_is_the_smallest_meeting_its_bound() {
        // The degree is the smallest n whose a-priori relative slope bound
        // (4n²+1)/T_n(3) reaches the unit roundoff; one less must miss it.
        let bound = |n: usize| {
            let nf = n as f64;
            let rate_pow = LOGISTIC_NORMAL_CRVZ_RATE.powi(n as i32);
            (4.0 * nf * nf + 1.0) / (0.5 * (rate_pow + rate_pow.recip()))
        };
        let n = LOGISTIC_NORMAL_SERIES_TERMS;
        assert!(bound(n) <= 0.5 * f64::EPSILON);
        assert!(bound(n - 1) > 0.5 * f64::EPSILON);
        // The weights are the coefficients of a polynomial Σ_k w_k t^k equal to
        // 1/(1+t) on [0, 1] to relative error 1/T_n(3), far below roundoff.
        for i in 0..=64 {
            let t = f64::from(i) / 64.0;
            let poly = LOGISTIC_NORMAL_SERIES_WEIGHTS
                .weight
                .iter()
                .rev()
                .fold(0.0, |acc, w| acc * t + w);
            assert_relative_eq!(poly, 1.0 / (1.0 + t), max_relative = 1e-13);
        }
    }

    #[test]
    fn test_logit_exact_series_matches_high_precision_reference() {
        // Reference values of E[σ(η)] and E[σ'(η)], η ~ N(μ, s²), computed
        // with mpmath at 40 digits by tanh-sinh quadrature in η with
        // breakpoints resolving both the Gaussian (every s/2 around μ) and the
        // logistic transition (every ½ around 0); a 50-digit rerun with a
        // denser breakpoint grid agrees to 5e-16 relative on every entry.
        // Rows span s from 1e-6 to 100 and η means out to e^{−700}.
        let table: &[(f64, &[(f64, f64, f64)])] = &[
            (
                1e-6,
                &[
                    (-700.0, 9.8596765437650614e-305, 9.8596765437650614e-305),
                    (-300.0, 5.1482002224147762e-131, 5.1482002224147762e-131),
                    (-40.0, 4.2483542552937132e-18, 4.2483542552937131e-18),
                    (-3.0, 4.7425873177587227e-2, 4.5176659730928598e-2),
                    (0.0, 5.0e-1, 2.499999999999375e-1),
                    (0.7, 6.6818777216812882e-1, 2.2171287329307244e-1),
                    (8.0, 9.9966464986953335e-1, 3.352376707566415e-4),
                    (300.0, 1.0, 5.1482002224147762e-131),
                ],
            ),
            (
                1e-3,
                &[
                    (-700.0, 9.8596814735996359e-305, 9.8596814735996359e-305),
                    (-300.0, 5.1482027965129569e-131, 5.1482027965129569e-131),
                    (-40.0, 4.2483563794692477e-18, 4.2483563794692476e-18),
                    (-3.0, 4.7425893623356452e-2, 4.5176676196449621e-2),
                    (0.0, 5.0e-1, 2.4999993750003125e-1),
                    (0.7, 6.6818773487878737e-1, 2.21712836679758e-1),
                    (8.0, 9.9966464970202707e-1, 3.3523783803819819e-4),
                    (300.0, 1.0, 5.1482027965129569e-131),
                ],
            ),
            (
                0.05,
                &[
                    (-700.0, 9.8720088455226649e-305, 9.8720088455226649e-305),
                    (-300.0, 5.1546394963980114e-131, 5.1546394963980114e-131),
                    (-40.0, 4.2536680185208255e-18, 4.2536680185208255e-18),
                    (-3.0, 4.7477002260580676e-2, 4.5217819654717117e-2),
                    (0.0, 5.0e-1, 2.498439449674203e-1),
                    (0.7, 6.6809464530355158e-1, 2.2162138280372003e-1),
                    (8.0, 9.996642308427173e-1, 3.3565613434122228e-4),
                    (300.0, 1.0, 5.1546394963980114e-131),
                ],
            ),
            (
                0.3,
                &[
                    (-700.0, 1.0313496354461585e-304, 1.0313496354461585e-304),
                    (-300.0, 5.3851608610314164e-131, 5.3851608610314164e-131),
                    (-40.0, 4.4438969097967517e-18, 4.4438969097967517e-18),
                    (-3.0, 4.9284332741628111e-2, 4.6652337487348473e-2),
                    (0.0, 5.0e-1, 2.446131934965273e-1),
                    (0.7, 6.6495133115389394e-1, 2.1847509919777044e-1),
                    (8.0, 9.9964923141774624e-1, 3.5063396631200147e-4),
                    (300.0, 1.0, 5.3851608610314164e-131),
                ],
            ),
            (
                1.0,
                &[
                    (-700.0, 1.6255858439920452e-304, 1.6255858439920452e-304),
                    (-300.0, 8.4879472125141282e-131, 8.4879472125141282e-131),
                    (-40.0, 7.0043520261686451e-18, 7.004352026168645e-18),
                    (-3.0, 6.9323858004285768e-2, 5.9821864078413161e-2),
                    (0.0, 5.0e-1, 2.0662096414190704e-1),
                    (0.7, 6.4115588112722935e-1, 1.919583592786463e-1),
                    (8.0, 9.9944774379699383e-1, 5.5143136174432282e-4),
                    (300.0, 1.0, 8.4879472125141282e-131),
                ],
            ),
            (
                3.0,
                &[
                    (-700.0, 8.8753979802033087e-303, 8.8753979802033087e-303),
                    (-300.0, 4.634262153822548e-129, 4.634262153822548e-129),
                    (-40.0, 3.8242466280852847e-16, 3.8242466280734341e-16),
                    (-3.0, 1.9438573608839076e-1, 7.8734608818823313e-2),
                    (0.0, 5.0e-1, 1.1483790477391054e-1),
                    (0.7, 5.7983742037389368e-1, 1.124944095508057e-1),
                    (8.0, 9.883210832830183e-1, 8.3539748503936299e-3),
                    (300.0, 1.0, 4.634262153822548e-129),
                ],
            ),
            (
                10.0,
                &[
                    (-700.0, 5.1119519486513433e-283, 5.1119519486513433e-283),
                    (-300.0, 2.669190215541374e-109, 2.669190215541374e-109),
                    (-40.0, 4.1914830581518058e-5, 1.7123017772409959e-5),
                    (-3.0, 3.8391056883627619e-1, 3.7584931138428607e-2),
                    (0.0, 5.0e-1, 3.9259560109364077e-2),
                    (0.7, 5.2745996633197458e-1, 3.9166493970108451e-2),
                    (8.0, 7.8443209202037574e-1, 2.8795653215573714e-2),
                    (300.0, 1.0, 2.669190215541374e-109),
                ],
            ),
            (
                30.0,
                &[
                    (-700.0, 3.7982149710920399e-120, 2.9391251210845267e-120),
                    (-300.0, 9.2215466208430222e-24, 3.0919158374860069e-24),
                    (-40.0, 9.1610277587177178e-2, 5.4747168305465195e-3),
                    (-3.0, 4.6024443796996282e-1, 1.3207900139307798e-2),
                    (0.0, 5.0e-1, 1.3273863811395113e-2),
                    (0.7, 5.0929086466429572e-1, 1.3270263990900267e-2),
                    (8.0, 6.0495014107041604e-1, 1.2811851472565363e-2),
                    (300.0, 1.0, 3.0919158374860069e-24),
                ],
            ),
            (
                100.0,
                &[
                    (-700.0, 1.2903867081808531e-12, 9.2072118515853982e-14),
                    (-300.0, 1.3520865722746964e-3, 4.4376830081080204e-5),
                    (-40.0, 3.4460248167363889e-1, 3.6821926917715808e-3),
                    (-3.0, 4.8803549372221685e-1, 3.9869728455162543e-3),
                    (0.0, 5.0e-1, 3.9887667968355803e-3),
                    (0.7, 5.0279211396299996e-1, 3.9886691053786453e-3),
                    (8.0, 5.3187614071964734e-1, 3.9760273273959502e-3),
                    (300.0, 9.986479134277253e-1, 4.4376830081080204e-5),
                ],
            ),
        ];
        let mut worst_mean = 0.0_f64;
        let mut worst_slope = 0.0_f64;
        for &(sigma, rows) in table {
            for &(mu, ref_mean, ref_slope) in rows {
                let out =
                    logit_posterior_meanwith_deriv_exact(mu, sigma).expect("series evaluates");
                assert_eq!(out.mode, IntegratedExpectationMode::ExactSpecialFunction);
                let mean_error = ((out.mean - ref_mean) / ref_mean).abs();
                let slope_error = ((out.dmean_dmu - ref_slope) / ref_slope).abs();
                worst_mean = worst_mean.max(mean_error);
                worst_slope = worst_slope.max(slope_error);
                assert!(
                    mean_error <= 1e-13 && slope_error <= 1e-12,
                    "logit({mu}, {sigma}): mean {:.17e} against {ref_mean:.17e} (error \
                     {mean_error:.3e}), slope {:.17e} against {ref_slope:.17e} (error \
                     {slope_error:.3e})",
                    out.mean,
                    out.dmean_dmu
                );
            }
        }
        assert!(
            worst_mean > 0.0 || worst_slope > 0.0,
            "every row reproduced bit-exactly: the table tests nothing"
        );
    }

    #[test]
    fn test_logit_integrated_derivative_is_even_in_mu() {
        // d/dμ E[sigmoid(η)] = E[sigmoid'(η)] and sigmoid' is even, so the
        // location-derivative is even in μ. #572 originated in a botched
        // sign/reflection of that derivative. Pin the symmetry from central
        // rows out to the deep tail.
        let ctx = QuadratureContext::new();
        for &(mu, sigma) in &[(0.3, 0.3), (1.1, 0.8), (10.0, 1.0), (3.0, 3.0), (35.0, 1.0)] {
            let pos =
                integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, mu, sigma)
                    .expect("logit moments (+μ)");
            let neg =
                integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, -mu, sigma)
                    .expect("logit moments (-μ)");
            assert_relative_eq!(pos.dmean_dmu, neg.dmean_dmu, max_relative = 1e-13);
            // And the mean reflects: E[sigmoid] at -μ equals 1 - E[sigmoid] at μ.
            assert_relative_eq!(
                neg.mean,
                1.0 - pos.mean,
                epsilon = 1e-15,
                max_relative = 1e-13
            );
        }
    }

    #[test]
    fn test_logit_dmean_dmu_equals_fd_of_mean_across_regimes() {
        // Regression for #571/#572 from the contract angle: the dispatcher's
        // returned `dmean_dmu` MUST equal d/dμ of the dispatcher's own `mean`
        // (the location-family identity the integrated-PIRLS Fisher weight and
        // working response depend on). A central finite difference of the
        // public `mean` is an end-to-end check that is blind to *which* internal
        // branch produced the value — it would have caught the #572 erfcx
        // derivative (2.4× too large) and any future formula that returns a
        // derivative inconsistent with its own mean. The grid keeps the rows
        // of the regimes the retired ladder used to split into.
        let ctx = QuadratureContext::new();
        let h = 1e-4;
        let cases = [
            (0.0, 0.8),  // μ=0 (the #572 failure family)
            (0.7, 0.8),  // off-center
            (1.5, 1.2),  // moderate
            (-1.1, 0.9), // μ<0
            (8.0, 1.0),  // tail, series exits early
            (10.0, 1.5), // tail, series exits early
            (-9.0, 1.0), // tail, μ<0
            (0.5, 0.05), // small σ
            (0.5, 20.0), // large σ
        ];
        for &(mu, sigma) in &cases {
            let at = |m: f64| {
                integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, m, sigma)
                    .expect("logit moments")
            };
            let out = at(mu);
            let fd = (at(mu + h).mean - at(mu - h).mean) / (2.0 * h);
            assert!(
                (out.dmean_dmu - fd).abs() <= 1e-5,
                "dmean_dmu must equal d/dμ of mean at (μ={mu}, σ={sigma}): \
                 returned {}, FD of mean {} (mode {:?})",
                out.dmean_dmu,
                fd,
                out.mode
            );
            // Physical ceiling: E[sigmoid'(η)] ≤ sigmoid'(0) = 0.25 for every
            // (μ, σ); a Gaussian average of sigmoid' (max 0.25) can never exceed
            // it. The #572 bug returned 0.58 here, violating this hard bound.
            assert!(
                out.dmean_dmu <= 0.25 + 1e-9 && out.dmean_dmu >= 0.0,
                "dmean_dmu out of [0, 0.25] at (μ={mu}, σ={sigma}): {}",
                out.dmean_dmu
            );
        }
    }

    #[test]
    fn test_logit_scalar_matches_jet_at_large_sigma() {
        // Regression for #571: the scalar dispatcher used to return the
        // Monahan probit mean (e.g. 0.9206 at (3,3)) while the jet path
        // integrated by GHQ returned the truth (0.8056) — two public entry
        // points disagreeing in the first decimal. With Monahan removed the
        // scalar path routes to the same quadrature, so the two must agree.
        let ctx = QuadratureContext::new();
        for &(mu, sigma) in &[(3.0, 3.0), (4.0, 4.0), (2.0, 5.0), (5.0, 5.0)] {
            let scalar =
                integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, mu, sigma)
                    .expect("scalar logit moments");
            let jet = integrated_inverse_link_jet(&ctx, LinkFunction::Logit, mu, sigma)
                .expect("jet logit moments");
            // The scalar path is the exact CRVZ series, matching the independent
            // high-resolution Simpson reference (truth); the Monahan ~0.11 error
            // is gone.
            let (ref_mean, ref_d1, _, _) = logit_reference_jet_highres_simpson(mu, sigma);
            assert_relative_eq!(scalar.mean, ref_mean, epsilon = 1e-9, max_relative = 1e-8);
            assert_relative_eq!(
                scalar.dmean_dmu,
                ref_d1,
                epsilon = 1e-9,
                max_relative = 1e-8
            );
            // At wide σ the jet no longer integrates mean/d1 by Gauss-Hermite
            // (which under-resolves the localized sigmoid^(k) integrands and
            // drifted ~4e-3 from the scalar adaptive-Simpson value — the
            // residual #571 symptom). The jet now *reuses* the scalar backend's
            // mean/d1 (see `logit_wide_sigma_jet`), so the two public entry
            // points are identical to the bit, not merely close. Pin that
            // strong invariant.
            assert_relative_eq!(scalar.mean, jet.mean, epsilon = 1e-12, max_relative = 1e-12);
            assert_relative_eq!(
                scalar.dmean_dmu,
                jet.d1,
                epsilon = 1e-12,
                max_relative = 1e-12
            );
        }
    }

    #[test]
    fn test_logit_jet_accurate_at_wide_sigma() {
        // Regression for the residual #571 root cause: at wide σ the 51-node
        // Gauss-Hermite jet under-resolves the localized sigmoid^(k) integrands
        // and drifts from the truth (e.g. d1 ≈ 0.0702 vs 0.0700 at (3,3)). The
        // jet now routes σ > LOGIT_JET_GHQ_SIGMA_MAX through adaptive Simpson.
        // Pin ALL FOUR jet components (mean, d1, d2, d3) to an independent
        // high-resolution Simpson reference across the broad-σ band. PIRLS
        // consumes this dispatcher directly, so there is no second jet to
        // synchronize.
        let ctx = QuadratureContext::new();
        for &(mu, sigma) in &[(3.0, 3.0), (4.0, 4.0), (2.0, 5.0), (5.0, 5.0), (0.5, 20.0)] {
            let jet = integrated_inverse_link_jet(&ctx, LinkFunction::Logit, mu, sigma)
                .expect("wide-σ logit jet");
            let (rm, rd1, rd2, rd3) = logit_reference_jet_highres_simpson(mu, sigma);
            assert_relative_eq!(jet.mean, rm, epsilon = 1e-8, max_relative = 1e-7);
            assert_relative_eq!(jet.d1, rd1, epsilon = 1e-8, max_relative = 1e-6);
            assert_relative_eq!(jet.d2, rd2, epsilon = 1e-8, max_relative = 1e-6);
            assert_relative_eq!(jet.d3, rd3, epsilon = 1e-8, max_relative = 1e-6);
            // d1 is the scalar backend's derivative verbatim (consistency #571).
            let scalar =
                integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, mu, sigma)
                    .expect("scalar logit moments");
            assert_relative_eq!(jet.d1, scalar.dmean_dmu, epsilon = 1e-12);
            assert_relative_eq!(jet.mean, scalar.mean, epsilon = 1e-12);
        }
    }

    #[test]
    fn test_logit_jet_continuous_across_ghq_simpson_seam() {
        // The jet switches integrators at σ = LOGIT_JET_GHQ_SIGMA_MAX (GHQ at or
        // below, adaptive Simpson above). Both sides are accurate, so the seam
        // must not introduce a visible jump that would perturb PIRLS. The seam
        // jump is exactly (GHQ value − Simpson value) at the threshold σ, so we
        // evaluate BOTH integrators at the same σ to isolate that jump from the
        // jet's genuine σ-dependence (a 1e-6 step in σ alone moves the mean by
        // ~∂M/∂σ·1e-6 ≈ 6e-8, which would otherwise masquerade as a seam jump).
        let ctx = QuadratureContext::new();
        let sigma = LOGIT_JET_GHQ_SIGMA_MAX;
        for mu in [-2.0, -0.5, 0.0, 0.7, 1.3, 3.0] {
            // Dispatch path at the threshold uses GHQ (σ is not > the cutoff).
            let ghq = integrated_inverse_link_jet(&ctx, LinkFunction::Logit, mu, sigma)
                .expect("jet at seam (GHQ dispatch)");
            // Same σ, but forced through the adaptive-Simpson backend.
            let simpson = logit_wide_sigma_jet(mu, sigma).expect("jet at seam (Simpson)");
            // GHQ at σ=1 holds to ≤ ~2e-9 on all four components (Simpson is
            // ~1e-12), so the seam jump is bounded by GHQ's residual error.
            assert_relative_eq!(ghq.mean, simpson.mean, epsilon = 1e-9, max_relative = 1e-8);
            assert_relative_eq!(ghq.d1, simpson.d1, epsilon = 1e-9, max_relative = 1e-7);
            assert_relative_eq!(ghq.d2, simpson.d2, epsilon = 1e-9, max_relative = 1e-7);
            assert_relative_eq!(ghq.d3, simpson.d3, epsilon = 1e-8, max_relative = 1e-6);
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
    fn integrated_family_moments_rejects_latent_cloglog_without_concrete_handler() {
        // With the LikelihoodSpec migration, SAS and Mixture parameterized binomial
        // variants carry their state through `InverseLink`, so the type system
        // already prevents constructing a state-less call. The only remaining
        // explicit error path here is `Binomial + LatentCLogLog`, which this
        // dispatcher reports as needing an explicit latent-cloglog state handler.
        let ctx = QuadratureContext::new();
        let latent =
            gam_problem::types::LatentCLogLogState::new(0.4).expect("valid latent cloglog state");
        let spec =
            LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::LatentCLogLog(latent));
        let likelihood = GlmLikelihoodSpec::canonical(spec);
        let err = integrated_family_moments_jet(&ctx, &likelihood, 0.2, 0.5)
            .expect_err("latent cloglog moments should error in this dispatcher");
        assert!(format!("{err}").contains("LatentCLogLog"));
    }

    #[test]
    fn integrated_family_moments_supports_stateful_sas() {
        let ctx = QuadratureContext::new();
        let sas = crate::mixture_link::state_from_sasspec(gam_problem::types::SasLinkSpec {
            initial_epsilon: 0.3,
            initial_log_delta: -0.2,
        })
        .expect("sas state should reconstruct from raw parameters");
        let spec = LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Sas(sas));
        let likelihood = GlmLikelihoodSpec::canonical(spec);
        let out = integrated_family_moments_jet(&ctx, &likelihood, 0.2, 0.5)
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
        let state = crate::mixture_link::state_fromspec(&gam_problem::types::MixtureLinkSpec {
            components: vec![gam_problem::types::LinkComponent::Probit],
            initial_rho: ndarray::Array1::<f64>::zeros(0),
        })
        .expect("single-component probit mixture state");
        let spec = LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Mixture(state));
        let likelihood = GlmLikelihoodSpec::canonical(spec);
        let out = integrated_family_moments_jet(&ctx, &likelihood, 0.7, 1.3)
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
        let state = crate::mixture_link::state_fromspec(&gam_problem::types::MixtureLinkSpec {
            components: vec![gam_problem::types::LinkComponent::Logit],
            initial_rho: ndarray::Array1::<f64>::zeros(0),
        })
        .expect("single-component logit mixture state");
        let spec = LikelihoodSpec::new(ResponseFamily::Binomial, InverseLink::Mixture(state));
        let likelihood = GlmLikelihoodSpec::canonical(spec);
        let out = integrated_family_moments_jet(&ctx, &likelihood, 1.1, 0.8)
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
        let state = crate::mixture_link::state_fromspec(&gam_problem::types::MixtureLinkSpec {
            components: vec![
                gam_problem::types::LinkComponent::Logit,
                gam_problem::types::LinkComponent::Probit,
            ],
            initial_rho: ndarray::array![0.35],
        })
        .expect("mixture state should reconstruct from rho");
        let spec = LikelihoodSpec::new(
            ResponseFamily::Binomial,
            InverseLink::Mixture(state.clone()),
        );
        let likelihood = GlmLikelihoodSpec::canonical(spec);
        let out = integrated_family_moments_jet(&ctx, &likelihood, 0.2, 0.5)
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

    #[inline]
    fn cloglog_negative_tail_derivative(eta: f64) -> f64 {
        // dμ/dη = exp(η) · exp(−exp(η)).
        if eta < -745.0 {
            0.0
        } else {
            let ex = safe_exp(eta);
            (ex * (-ex).exp()).max(0.0)
        }
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
                cloglog_mean_d1_exact(mu),
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

    /// Reference heap-based Cholesky-with-jitter, kept here as a test oracle
    /// so we can confirm that the new stack-allocated variant matches it
    /// bit-for-bit (modulo the bit-identical scalar math, which is by design).
    fn ref_cholesky_heap(cov: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
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

    #[test]
    fn cholesky_static_matches_heap_d2() {
        // A handful of deterministic PSD 2x2 cases generated from
        // randomized factors: cov = A A^T + diag(eps).
        let cases: &[[[f64; 2]; 2]] = &[
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.5, 0.3], [0.3, 0.75]],
            [[1.0, 0.9999], [0.9999, 1.0]],
            [[1e-10, 0.0], [0.0, 1e-10]],
            [[4.0, -1.5], [-1.5, 2.25]],
        ];
        for cov in cases {
            let stack = cholesky_static::<2>(cov).expect("stack cholesky");
            let heap_in: Vec<Vec<f64>> = cov.iter().map(|r| r.to_vec()).collect();
            let heap = ref_cholesky_heap(&heap_in).expect("heap cholesky");
            for i in 0..2 {
                for j in 0..2 {
                    assert_eq!(
                        stack[i][j].to_bits(),
                        heap[i][j].to_bits(),
                        "mismatch at ({i},{j}) for cov={cov:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn cholesky_static_matches_heap_d3() {
        let cases: &[[[f64; 3]; 3]] = &[
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            [[2.0, 0.5, 0.1], [0.5, 1.5, -0.2], [0.1, -0.2, 0.8]],
            [[4.0, 1.0, 0.5], [1.0, 3.0, 0.25], [0.5, 0.25, 2.0]],
        ];
        for cov in cases {
            let stack = cholesky_static::<3>(cov).expect("stack cholesky");
            let heap_in: Vec<Vec<f64>> = cov.iter().map(|r| r.to_vec()).collect();
            let heap = ref_cholesky_heap(&heap_in).expect("heap cholesky");
            for i in 0..3 {
                for j in 0..3 {
                    assert_eq!(
                        stack[i][j].to_bits(),
                        heap[i][j].to_bits(),
                        "mismatch at ({i},{j}) for cov={cov:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn cholesky_static_d1() {
        let l = cholesky_static::<1>(&[[2.25]]).expect("d=1");
        assert_eq!(l[0][0], 1.5);
        // A zero variance is a point mass along its axis: the factor column is zero.
        let point = cholesky_static::<1>(&[[0.0]]).expect("zero variance");
        assert_eq!(point[0][0], 0.0);
        // A negative variance is indefinite at every magnitude, because the band
        // a pivot is judged against scales with the pivot's own terms.
        assert!(cholesky_static::<1>(&[[-1.0e-13]]).is_none());
        assert!(cholesky_static::<1>(&[[-1.0e3]]).is_none());
    }

    #[test]
    fn cholesky_static_factors_a_rank_one_covariance_and_refuses_an_indefinite_one() {
        // [[1, 1], [1, 1]] has pivots 1 and exactly 0: the second column is zero,
        // so the Gaussian integrates along the major axis only.
        let rank_one = cholesky_static::<2>(&[[1.0, 1.0], [1.0, 1.0]]).expect("rank one");
        assert_eq!(rank_one, [[1.0, 0.0], [1.0, 0.0]]);
        // [[0, 1], [1, 0]] has a zero pivot above a unit Schur entry, which no
        // positive semidefinite matrix has.
        assert!(cholesky_static::<2>(&[[0.0, 1.0], [1.0, 0.0]]).is_none());
    }

    /// `normal_expectation_2d_projected_result` reproduces the closed-form
    /// Gaussian moments on every support it integrates over (gam#2931).
    ///
    /// Under the law the rule integrates, `E[1] = 1`, `E[xₖ] = μₖ` and
    /// `E[xₖxₗ] = μₖμₗ + Σₖₗ`, where `Σ` is the covariance itself on the plane
    /// and its major eigenpair `variance·axis·axisᵀ` on a rank-one or indefinite
    /// covariance. An n-point Gauss–Hermite rule integrates every polynomial of
    /// degree ≤ 2n − 1 exactly and the adaptive rule never takes fewer than 7
    /// points, so what is left is the rounding of the rule's own weighted sum.
    /// Each of its `nodes` terms is formed by at most `2·D` node and weight
    /// products, `D·(D + 1)` Cholesky or axis products and sums, one monomial
    /// product and the weight times the value; the terms are then summed, scaled
    /// by the rule's normalisation and compared with a closed form of at most two
    /// rounded operations. So the bar is
    /// `accumulation_band(nodes·(2·D + D·(D + 1) + 2) + 3, Σ w·|f|)`, with the
    /// node count taken from the uncapped schedule (never fewer nodes than the
    /// rule ran) and `Σ w·|f|` read off the same rule. A point mass is `f(μ)` bit
    /// for bit.
    #[test]
    fn projected_bivariate_expectation_reproduces_closed_form_moments_2931() {
        type Moments = (f64, f64, f64, f64, f64, f64);
        let ctx = QuadratureContext::new();
        let mu = [0.3, -1.2];
        let monomials =
            |x0: f64, x1: f64| -> Result<Moments, String> { Ok((1.0, x0, x1, x0 * x0, x0 * x1, x1 * x1)) };
        let magnitudes = |x0: f64, x1: f64| -> Result<Moments, String> {
            Ok((1.0, x0.abs(), x1.abs(), x0 * x0, (x0 * x1).abs(), x1 * x1))
        };
        let entries = |m: Moments| [m.0, m.1, m.2, m.3, m.4, m.5];
        let check = |label: &str, cov: [[f64; 2]; 2], law: [[f64; 2]; 2], dims: usize, max_sd: f64| {
            let got = entries(
                normal_expectation_2d_projected_result(&ctx, mu, cov, &monomials).expect("moments"),
            );
            let absolute = entries(
                normal_expectation_2d_projected_result(&ctx, mu, cov, &magnitudes)
                    .expect("magnitudes"),
            );
            let exact = [
                1.0,
                mu[0],
                mu[1],
                mu[0] * mu[0] + law[0][0],
                mu[0] * mu[1] + law[0][1],
                mu[1] * mu[1] + law[1][1],
            ];
            let nodes = adaptive_point_count_from_sd(max_sd).pow(dims as u32);
            let terms = nodes * (2 * dims + dims * (dims + 1) + 2) + 3;
            for k in 0..6 {
                let gap = (got[k] - exact[k]).abs();
                let bar = gam_linalg::roundoff::accumulation_band(terms, absolute[k]);
                assert!(
                    gap <= bar,
                    "[{label}] moment {k}: rule {:.17e}, closed form {:.17e}, gap {gap:.3e} above \
                     the rounding bar {bar:.3e}",
                    got[k],
                    exact[k]
                );
            }
        };

        let plane = [[0.04, 0.012], [0.012, 0.09]];
        assert_eq!(BivariateNormalSupport::of(plane), BivariateNormalSupport::Plane);
        check("plane", plane, plane, 2, plane[1][1].sqrt());

        // Exactly rank one in floating point: the second pivot 1 − (0.5/0.5)² is 0.
        let rank_one = [[0.25, 0.5], [0.5, 1.0]];
        let BivariateNormalSupport::Axis { variance, .. } = BivariateNormalSupport::of(rank_one) else {
            panic!("a rank-one covariance integrates along its major axis");
        };
        assert_eq!(variance, 1.25);
        check("rank-one axis", rank_one, rank_one, 1, variance.sqrt());

        // Indefinite: eigenvalues ±1. The rule integrates the nearest positive
        // semidefinite covariance, the eigenpair 1 along (1, 1)/√2.
        let indefinite = [[0.0, 1.0], [1.0, 0.0]];
        let BivariateNormalSupport::Axis { variance, .. } = BivariateNormalSupport::of(indefinite) else {
            panic!("an indefinite covariance integrates along its major axis");
        };
        assert_eq!(variance, 1.0);
        check("indefinite axis", indefinite, [[0.5, 0.5], [0.5, 0.5]], 1, variance.sqrt());

        let point = [[0.0, 0.0], [0.0, 0.0]];
        assert_eq!(BivariateNormalSupport::of(point), BivariateNormalSupport::Point);
        let at_point = entries(
            normal_expectation_2d_projected_result(&ctx, mu, point, &monomials).expect("point mass"),
        );
        let at_mean = entries(monomials(mu[0], mu[1]).expect("monomials at the mean"));
        assert_eq!(at_point.map(f64::to_bits), at_mean.map(f64::to_bits));
    }
}

#[cfg(test)]
mod log_survival_panel_2714_tests {
    use super::{
        LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER, LOG_SURVIVAL_PANEL_MAX_NODES,
        LOG_SURVIVAL_PANEL_MIN_NODES, LOG_SURVIVAL_TOWER_MAX_LOG_CANCELLATION, LogSurvivalBranch,
        QuadratureContext, log_complement_peak_z, log_survival_jet, log_survival_panel,
        log_survival_panel_edge,
    };

    /// #2469: the log-survival root brackets stop at their own floating-point
    /// resolution, not after 200 steps.
    /// - With `σ = 1e60` the complement peak sits near `5e-60` inside `(0, 1e60]`.
    ///   Bisection needs about 450 halvings to reach adjacent doubles. The replaced
    ///   cap returned a point about `0.3` away, where the slope is long negative. The
    ///   returned peak brackets the slope's sign change to one ulp.
    /// - Each panel edge is the first double past the peak at which the drop is
    ///   reached.
    #[test]
    fn log_survival_roots_stop_at_their_brackets_resolution_2469() {
        let complement = LogSurvivalBranch::Complement;
        let (mu, sigma) = (0.0_f64, 1.0e60_f64);
        let peak = log_complement_peak_z(mu, sigma);
        assert!(peak > 0.0 && peak.is_finite(), "peak {peak:e}");
        assert!(
            complement.log_integrand_slope(mu, sigma, peak.next_down()) > 0.0
                && complement.log_integrand_slope(mu, sigma, peak.next_up()) <= 0.0,
            "the slope changes sign within one ulp of {peak:e}"
        );

        for (branch, mu, sigma) in [
            (LogSurvivalBranch::Survival, 3.2_f64, 0.15_f64),
            (LogSurvivalBranch::Complement, -30.0, 0.05),
            (LogSurvivalBranch::Survival, 12.0, 0.002),
        ] {
            let peak = match branch {
                LogSurvivalBranch::Survival => super::log_survival_peak_z(mu, sigma),
                LogSurvivalBranch::Complement => log_complement_peak_z(mu, sigma),
            };
            let peak_log = branch.log_integrand(mu, sigma, peak);
            let drop = 40.0;
            let fallen = |z: f64| peak_log - branch.log_integrand(mu, sigma, z) >= drop;
            let upper = log_survival_panel_edge(branch, mu, sigma, peak, drop, 1.0);
            let lower = log_survival_panel_edge(branch, mu, sigma, peak, drop, -1.0);
            assert!(fallen(upper) && !fallen(upper.next_down()), "upper edge {upper:e}");
            assert!(fallen(lower) && !fallen(lower.next_up()), "lower edge {lower:e}");
        }
    }

    /// `(mu, sigma, ln S)` — 60-digit mpmath reference for
    /// `S(mu,sigma) = E[exp(-e^eta)]`, `eta ~ N(mu, sigma^2)`.
    ///
    /// Computed with the Laplace peak factored out (integrate
    /// `exp(L(z) - L(z*))`, add `L(z*)` back), which is what makes the
    /// reference accurate ABSOLUTELY in `ln S` even where `ln S = -1.3e5`.
    /// The naive un-shifted high-precision integral is itself wrong by `8.7e-3`
    /// at `(8, 0.005)` and must not be used as an oracle here.
    const LOG_SURVIVAL_REFERENCE: &[(f64, f64, f64)] = &[
        (-30.0, 0.05, -9.369_327_311_241_221e-14),
        (-20.0, 0.002, -2.061_157_744_749_916_5e-9),
        (-20.0, 2.0, -1.522_997_352_870_166_6e-8),
        (-8.0, 0.15, -3.392_565_812_965_201_6e-4),
        (-8.0, 8.0, -1.981_213_453_495_813_4e-1),
        (-3.0, 0.02, -4.979_653_073_925_083_3e-2),
        (-3.0, 1.0, -7.722_156_176_750_611e-2),
        (-1.0, 0.005, -3.678_823_479_542_769_3e-1),
        (0.0, 0.05, -9.999_992_216_748_937e-1),
        (0.0, 20.0, -7.163_521_358_485_562e-1),
        (1.0, 0.15, -2.668_055_285_814_174_4e0),
        (1.8, 0.15, -5.742_720_404_435_083e0),
        (1.8, 0.5, -4.257_254_029_226_836e0),
        (3.2, 0.005, -2.452_531_812_111_085_3e1),
        (3.2, 0.15, -2.014_631_867_969_775_7e1),
        (3.2, 4.0, -1.691_825_149_278_477e0),
        (5.0, 0.02, -1.442_777_351_752_632_2e2),
        (5.0, 1.0, -1.128_111_355_370_178_3e1),
        (8.0, 0.002, -2.963_400_229_027_156e3),
        (8.0, 0.05, -1.113_594_437_164_973_2e3),
        (8.0, 0.5, -7.097_988_851_759_84e1),
        (8.0, 60.0, -8.137_859_923_376_214e-1),
        (12.0, 0.002, -1.289_824_339_085_843_7e5),
        (12.0, 0.15, -1.181_338_131_091_633_2e3),
        (12.0, 1.0, -5.819_669_561_598_102e1),
        (20.0, 0.05, -3.135_653_820_340_577_3e4),
        (-50.0, 8.0, -7.328_008_793_232_462e-10),
        (-18.0, 0.001, -1.522_998_735_970_428_8e-8),
    ];

    /// #2714, the accuracy half: `ln S` on one panel is at the f64 relative
    /// floor everywhere, including the corners where the ladder it replaced was
    /// wrong by `4.4e+06`.
    ///
    /// The bar is RELATIVE (`1e-13` of `|ln S|`, floored at `1e-13` absolute so
    /// the near-1 rows are not graded on a vanishing denominator). That is the
    /// right axis: `ln S` is consumed as an additive contribution to a log
    /// likelihood and as the argument of exponentiated differences, so what
    /// must be small is its error against its own magnitude.
    ///
    /// Rows carried deliberately, with what each one would catch:
    /// * `(12, 0.002)` / `(8, 0.002)` — the fixed-window Gumbel escape hatch,
    ///   whose `Phi((eta-mu)/sigma)` transition was unresolved by 513 nodes.
    /// * `(12, 1.0)` / `(8, 0.5)` — the Miles/CC value route, `.ln()`-ed.
    /// * `(3.2, 0.15)` — the exact point #2714's diagnosis names.
    /// * `(-50, 8)` — the rare-event asymptotic `ln1p(-e^{mu+sigma^2/2})` at
    ///   its own gate `rare_log = -18`, where it is 20.8x wrong because the
    ///   higher cumulants are not small when sigma is large.
    /// * `(-30, 0.05)` / `(-18, 0.001)` — `S -> 1`, which is the complement
    ///   branch, i.e. relative accuracy where `ln S -> 0`.
    #[test]
    fn log_survival_matches_a_high_precision_reference_2714() {
        let ctx = QuadratureContext::new();
        let mut worst = 0.0_f64;
        let mut worst_at = (f64::NAN, f64::NAN);
        for &(mu, sigma, expected) in LOG_SURVIVAL_REFERENCE {
            let got = log_survival_jet(&ctx, mu, sigma, 0).log_survival;
            let error = (got - expected).abs() / expected.abs().max(1.0);
            if error > worst {
                worst = error;
                worst_at = (mu, sigma);
            }
            assert!(
                error <= 1.0e-13,
                "#2714: ln S({mu}, {sigma}) = {got:.17e} against reference \
                 {expected:.17e} (relative {error:.3e}); one log-space panel has \
                 to hold the f64 floor at every (mu, sigma), because a routed \
                 surface whose error is a step function of (mu, sigma) is the \
                 defect this replaced"
            );
        }
        // NON-VACUITY: the reference must actually be exercised, i.e. the grid
        // has to contain rows the evaluator does not reproduce for free.
        assert!(
            worst > 0.0,
            "#2714: every reference row reproduced bit-exactly, which means the \
             table is not testing the quadrature"
        );
        let (mu, sigma) = worst_at;
        println!("[2714] worst relative ln S error {worst:.3e} at (mu={mu}, sigma={sigma})");
    }

    /// #2714, the consistency half: the analytic `sigma^j d^j S/d mu^j` tower is
    /// the derivative of the `ln S` this module returns.
    ///
    /// This is the assertion whose absence let the defect live. Every earlier
    /// gate scored an analytic derivative against ITS OWN value function; the
    /// stall came from the derivative being right about the true `S` while the
    /// value came off a different approximation. A central difference of the
    /// shipped `ln S` against the shipped tower cannot be satisfied by two
    /// implementations that agree with each other and not with the truth.
    #[test]
    fn log_survival_tower_is_the_derivative_of_the_value_2714() {
        let ctx = QuadratureContext::new();
        let mut worst = 0.0_f64;
        for &(mu, sigma) in &[
            (-8.0, 0.15),
            (-3.0, 0.5),
            (0.0, 1.0),
            (1.8, 0.15),
            (3.2, 0.15),
            (3.2, 2.0),
            (5.0, 1.0),
            (8.0, 4.0),
            (0.0, 8.0),
            (-3.0, 20.0),
        ] {
            let jet = log_survival_jet(&ctx, mu, sigma, 1);
            let first = jet.scaled_mu_derivatives[1];
            // d/dmu ln S = (d S/d mu)/S = sign * exp(log|sigma dS/dmu| - ln S)/sigma
            let analytic = first.sign * (first.log_abs - jet.log_survival).exp() / sigma;
            let step = (f64::EPSILON.cbrt()) * (1.0 + mu.abs());
            let up = log_survival_jet(&ctx, mu + step, sigma, 0).log_survival;
            let down = log_survival_jet(&ctx, mu - step, sigma, 0).log_survival;
            let numeric = (up - down) / (2.0 * step);
            let error = (analytic - numeric).abs() / analytic.abs().max(1.0e-3);
            worst = worst.max(error);
            assert!(
                error <= 1.0e-7,
                "#2714: d/dmu ln S at (mu={mu}, sigma={sigma}) is {analytic:.12e} \
                 analytically and {numeric:.12e} by central difference of the \
                 shipped value (relative {error:.3e}). A tower that is not the \
                 derivative of the value it ships with is what collapses the \
                 joint-Newton trust region."
            );
        }
        println!("[2714] worst |analytic - FD| / |analytic| = {worst:.3e}");
    }

    /// #2714: `σ^j ∂_μ^j S` is a property of `(μ, σ)`, not of how many orders
    /// the caller asked for.
    ///
    /// The panel's window and node count are chosen from the order it is placed
    /// for, so reading that order off the REQUEST made the same integral come
    /// off a different rule per consumer. That is not hypothetical here: on one
    /// latent-survival row, `log_kernel_bundle` is asked for `max_k = 4`, `5`,
    /// `6` and `7` by the value, gradient, Hessian and contracted-third paths
    /// respectively, and every one of them reads entry `1` of the same tower.
    /// A Hessian assembled from a different `∂_a K₀` than the gradient it is
    /// paired with is not the derivative of that gradient.
    ///
    /// Bit-equality, not a tolerance: two placements of one rule differ by
    /// round-off, and round-off is exactly what a trust ratio divides by near
    /// its fixed point. There is nothing to be tolerant of — the numbers are
    /// either the same object or they are not.
    #[test]
    fn log_survival_tower_is_independent_of_the_requested_order_2714() {
        let ctx = QuadratureContext::new();
        let mut compared = 0usize;
        for &(mu, sigma) in &[
            (-20.0, 0.002),
            (-8.0, 0.15),
            (-3.0, 0.5),
            (0.0, 1.0),
            (1.8, 0.15),
            (3.2, 0.15),
            (3.2, 2.0),
            (5.0, 1.0),
            (8.0, 4.0),
            (0.0, 8.0),
            (12.0, 0.02),
            (-3.0, 20.0),
        ] {
            let reference = log_survival_jet(&ctx, mu, sigma, LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER);
            for requested in 1..=LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER {
                let jet = log_survival_jet(&ctx, mu, sigma, requested);
                for order in 1..=requested {
                    let got = jet.scaled_mu_derivatives[order];
                    let want = reference.scaled_mu_derivatives[order];
                    assert_eq!(
                        (got.log_abs.to_bits(), got.sign.to_bits()),
                        (want.log_abs.to_bits(), want.sign.to_bits()),
                        "#2714: sigma^{order} d^{order} S/d mu^{order} at (mu={mu}, \
                         sigma={sigma}) is {:.17e} (sign {}) when {requested} orders \
                         are requested and {:.17e} (sign {}) when \
                         {LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER} are. The tower must \
                         not depend on how much of it the caller wants.",
                        got.log_abs, got.sign, want.log_abs, want.sign
                    );
                    compared += 1;
                }
            }
        }
        // Non-vacuity: an assertion loop that never ran is a green that means
        // nothing, and this one is nested three deep.
        assert!(
            compared >= 12 * (LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER
                * (LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER + 1)
                / 2),
            "#2714: only {compared} tower entries were compared"
        );
        println!("[2714] tower order-independence: {compared} entries bit-identical");
    }

    /// #2714: certification is a PREFIX, so the basis a consumer is routed to
    /// does not depend on how many rungs it asked for.
    ///
    /// The all-or-nothing form refused the whole tower when its last rung
    /// cancelled, so a consumer needing rungs `0..=1` and asking for `0..=5` was
    /// denied a basis that is well conditioned at every rung it reads. The
    /// prefix form gives it exactly the rungs the quadrature vouches for.
    #[test]
    fn log_survival_tower_certification_is_a_prefix_2714() {
        let ctx = QuadratureContext::new();
        let mut saw_truncation = false;
        for &(mu, sigma, ..) in LOG_SURVIVAL_TOWER_REFERENCE {
            let jet = log_survival_jet(&ctx, mu, sigma, LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER);
            let Some(prefix) = jet.certified_prefix_order(LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER)
            else {
                continue;
            };
            // Monotone in the request: asking for less can never certify less.
            for requested in 0..=LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER {
                assert_eq!(
                    jet.certified_prefix_order(requested),
                    Some(prefix.min(requested)),
                    "#2714: the certified prefix at (mu={mu}, sigma={sigma}) is \
                     {prefix} but reading it against a request of {requested} \
                     disagrees — the prefix must be a property of the tower, \
                     truncated by the request and not decided by it"
                );
                // And it agrees with the all-or-nothing reading exactly where
                // that reading succeeds, so nothing that used to be admitted
                // has been silently widened.
                assert_eq!(
                    jet.certified_scaled_mu_derivatives(requested).is_some(),
                    requested <= prefix,
                    "#2714: the whole-tower and prefix readings disagree at \
                     (mu={mu}, sigma={sigma}, requested={requested})"
                );
            }
            if prefix < LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER {
                saw_truncation = true;
            }
        }
        // The whole point is the rows where the prefix is SHORTER than the
        // request; a table on which every tower certifies to the top says
        // nothing about the change.
        assert!(
            saw_truncation,
            "#2714: no reference row produced a truncated certified prefix, so \
             this test did not exercise the behaviour it exists for"
        );
    }

    /// `(mu, sigma, ln|σ^j ∂_μ^j S| for j = 0..=4, sign of each)` — 50-digit
    /// mpmath reference for the SAME object the panel tower produces,
    /// `σ^j ∂_μ^j S = ∫ He_j(z) φ(z) exp(−e^{μ+σz}) dz`.
    ///
    /// Computed peak-shifted, on a window carrying 95 e-folds plus the
    /// order's own `j·ln(2+|z⋆|)` padding, and cross-checked at two working
    /// precisions and two quadrature degrees.
    ///
    /// This is the oracle the tower has never had. Its existing gate is a
    /// central difference of the shipped `ln S` against the shipped order-1
    /// entry — necessary (it is the assertion whose absence let #2714 live)
    /// but self-referential past order 1 and blind to orders 2–4 entirely,
    /// which are exactly the orders `log_kernel_bundle` asks for.
    const LOG_SURVIVAL_TOWER_REFERENCE: &[(f64, f64, [f64; 5], [f64; 5])] = &[
        (-20.0, 0.002, [-2.0611577447499165e-9, -26.214606100483358, -32.429214200966715, -38.643822303511239, -44.858430410178095], [1.0, -1.0, -1.0, -1.0, -1.0]),
        (-8.0, 0.15, [-0.00033925658129652018, -9.8862169612318004, -11.783683981303588, -13.681498274811226, -15.580007954561913], [1.0, -1.0, -1.0, -1.0, -1.0]),
        (-8.0, 8.0, [-0.19812134534958133, -1.3528211702547069, -1.452147542328205, -3.240571743395306, -0.71024921821004234], [1.0, -1.0, -1.0, 1.0, 1.0]),
        (-3.0, 0.02, [-0.049796530739250832, -6.9616394585654159, -10.92476204736755, -14.944640460549261, -19.104083600790619], [1.0, -1.0, -1.0, -1.0, -1.0]),
        (-3.0, 0.5, [-0.055971877835218903, -3.6398579829672956, -4.4066491238893504, -5.2575250298329575, -6.3338354346007994], [1.0, -1.0, -1.0, -1.0, -1.0]),
        (-3.0, 20.0, [-0.60124121574363753, -0.9283108111246211, -3.0420009372801215, -0.94719003039898553, -1.9515751519096107], [1.0, -1.0, -1.0, 1.0, 1.0]),
        (-1.0, 0.005, [-0.36788234795427693, -6.666196411633682, -12.423205395414024, -20.715002885443461, -22.768239058212055], [1.0, -1.0, -1.0, -1.0, 1.0]),
        (0.0, 0.05, [-0.99999922167489372, -3.9969814880950871, -13.681704026824476, -9.9897069128682089, -12.994203213800263], [1.0, -1.0, 1.0, 1.0, 1.0]),
        (0.0, 1.0, [-0.96297240050030377, -1.3514828821346528, -3.4776689251068834, -2.0721333282379262, -4.077937287249247], [1.0, -1.0, 1.0, 1.0, -1.0]),
        (0.0, 8.0, [-0.75101079589424262, -0.93383523622026295, -3.6184199118831959, -0.96301212066547202, -2.5659122387330754], [1.0, -1.0, 1.0, 1.0, -1.0]),
        (1.0, 0.15, [-2.6680552858141743, -3.6128621614846425, -5.0108306306587141, -8.9368697272493613, -7.5728553818569553], [1.0, -1.0, 1.0, -1.0, -1.0]),
        (1.8, 0.15, [-5.7427204044350835, -5.9515476490674363, -6.340239096657947, -7.0035313750765859, -8.252090685678664], [1.0, -1.0, 1.0, -1.0, 1.0]),
        (1.8, 0.5, [-4.2572540292268353, -3.8392878811891043, -3.6246169834513785, -3.7636817578612526, -5.1584862006168726], [1.0, -1.0, 1.0, -1.0, 1.0]),
        (3.2, 0.005, [-24.525318121110852, -26.624235940415906, -28.764769508877318, -30.95061498411088, -33.186272517159154], [1.0, -1.0, 1.0, -1.0, 1.0]),
        (3.2, 0.15, [-20.146318679697758, -19.21570557965608, -18.328812156432285, -17.490308387278464, -16.706077981177801], [1.0, -1.0, 1.0, -1.0, 1.0]),
        (3.2, 2.0, [-2.9809436264637528, -2.3631698967287561, -1.9865287099827521, -2.1204144743340792, -2.9045009851726548], [1.0, -1.0, 1.0, -1.0, -1.0]),
        (3.2, 4.0, [-1.691825149278477, -1.3630396002384465, -1.5160548511506286, -2.9955755434621241, -0.80260762836904534], [1.0, -1.0, 1.0, 1.0, -1.0]),
        (5.0, 1.0, [-11.281113553701783, -9.9522520507276806, -8.6794206852298702, -7.4721966945023752, -6.3443308739944037], [1.0, -1.0, 1.0, -1.0, 1.0]),
        (8.0, 0.05, [-1113.5944371649733, -1110.1523256471932, -1106.7108385495186, -1103.2699768904941, -1099.8297416923265], [1.0, -1.0, 1.0, -1.0, 1.0]),
        (8.0, 0.5, [-70.979888517598404, -68.673129437259724, -66.374650578770375, -64.084648631557826, -61.803330805347407], [1.0, -1.0, 1.0, -1.0, 1.0]),
        (8.0, 4.0, [-3.9201018255002301, -3.0676533191912935, -2.3815942308869494, -1.9700438387194749, -2.2785027569363032], [1.0, -1.0, 1.0, -1.0, 1.0]),
        (8.0, 60.0, [-0.8137859923376214, -0.92937929368933511, -2.8751091738859909, -0.95047056774657423, -1.7838093439866154], [1.0, -1.0, 1.0, 1.0, -1.0]),
        (12.0, 0.002, [-128982.43390858436, -128977.07394828311, -128971.7139945779, -128966.35404746883, -128960.99410695599], [1.0, -1.0, 1.0, -1.0, 1.0]),
        (12.0, 1.0, [-58.196695615981018, -55.917604210289579, -53.648028066976042, -51.388233991426026, -49.138505716496612], [1.0, -1.0, 1.0, -1.0, 1.0]),
        (-50.0, 8.0, [-7.3280087932324622e-10, -19.248520695018656, -17.486423137639009, -15.749919102771783, -14.041433160139017], [1.0, -1.0, -1.0, -1.0, -1.0]),
        (20.0, 0.05, [-31356.538203405772, -31351.094833958202, -31345.651481726192, -31340.208146710605, -31334.764828912308], [1.0, -1.0, 1.0, -1.0, 1.0]),
    ];

    /// #2714 / #2610: the whole μ-derivative tower is graded against an
    /// EXTERNAL oracle, at every order `log_kernel_bundle` actually asks for.
    ///
    /// The tower's shipped gate finite-differences `ln S` against the order-1
    /// entry. That catches the defect this issue names, and it cannot catch a
    /// tower that is self-consistent and wrong at order 2, 3 or 4 — which are
    /// the orders `LogKernelSumJet` reads (`max_k = k + 4`) and which the
    /// log-σ curvature is built out of. A sign flip at order 3 would leave the
    /// FD gate green and turn every latent-survival Hessian upside down.
    ///
    /// Certification is graded together with accuracy on purpose: the point of
    /// [`LogSurvivalJet::certified_scaled_mu_derivatives`] is that admitted
    /// entries are AT the working floor, so "admitted" and "accurate to 1e-13"
    /// must be the same statement about the same rows.
    #[test]
    fn log_survival_tower_matches_a_high_precision_reference_2714() {
        let ctx = QuadratureContext::new();
        let mut worst_certified = 0.0_f64;
        let mut worst_certified_at = (f64::NAN, f64::NAN, 0usize);
        let mut certified_rows = 0usize;
        let mut refused_rows = 0usize;
        for &(mu, sigma, expected_log, expected_sign) in LOG_SURVIVAL_TOWER_REFERENCE {
            let jet = log_survival_jet(&ctx, mu, sigma, 4);
            let certified = jet.certified_scaled_mu_derivatives(4).is_some();
            if certified {
                certified_rows += 1;
            } else {
                refused_rows += 1;
            }
            if !certified {
                // A refused row is refused precisely because its signed sum
                // cancelled past the point where f64 resolves it — at
                // `(mu=-20, sigma=0.002)` order 4 the cancellation is ~45
                // nats, so neither the magnitude NOR the sign of that entry is
                // a number. Grading it would be grading noise; what the
                // refusal path owes is in the companion test below, which
                // checks that the cancellation it reports actually bounds the
                // error it made.
                continue;
            }
            for order in 0..=4 {
                let entry = jet.scaled_mu_derivatives[order];
                assert_eq!(
                    entry.sign, expected_sign[order],
                    "#2714: sigma^{order} d^{order} S/d mu^{order} at (mu={mu}, \
                     sigma={sigma}) is certified and has sign {} against the \
                     high-precision {}. A sign error in the tower is not an \
                     accuracy question — it reverses the curvature every \
                     latent-survival row hands the joint Newton.",
                    entry.sign, expected_sign[order]
                );
                // The entries are log-magnitudes, so a difference of logs IS
                // the relative error of the magnitude they encode. The bar is
                // the same one the `ln S` table uses: `1e-13` of the entry's
                // own magnitude, floored at `1e-13` absolute. An entry
                // reported as a logarithm cannot be closer to the truth than
                // the ulp of that logarithm, and `|ln| = 1.3e5` occurs here.
                let error = (entry.log_abs - expected_log[order]).abs();
                let bar = 1.0e-13 * expected_log[order].abs().max(1.0);
                assert!(
                    error <= bar,
                    "#2714: sigma^{order} d^{order} S/d mu^{order} at (mu={mu}, \
                     sigma={sigma}) is CERTIFIED by its own measured \
                     cancellation ({:.3} nats) and yet lands {error:.3e} from \
                     the reference against a bar of {bar:.3e}. Certification \
                     means 'this entry is at the working floor'; if it is not, \
                     the gate that replaced the sigma >= 8 constant is weaker \
                     than the constant was.",
                    entry.log_cancellation
                );
                let relative = error / expected_log[order].abs().max(1.0);
                if relative > worst_certified {
                    worst_certified = relative;
                    worst_certified_at = (mu, sigma, order);
                }
            }
        }
        // NON-VACUITY, both ways: the gate has to admit and it has to refuse.
        // A table on which everything is certified says nothing about the
        // refusal path, and one on which nothing is says nothing at all.
        assert!(
            certified_rows > 0 && refused_rows > 0,
            "#2714: the tower reference must straddle the certification gate — \
             got {certified_rows} certified and {refused_rows} refused rows"
        );
        let (mu, sigma, order) = worst_certified_at;
        println!(
            "[2714] tower: {certified_rows} certified / {refused_rows} refused; \
             worst certified error {worst_certified:.3e} at (mu={mu}, sigma={sigma}, j={order})"
        );
    }

    /// #2714: the cancellation the tower reports is an HONEST error bar, not a
    /// label.
    ///
    /// `certified_scaled_mu_derivatives` replaced a `sigma >= 8` constant with
    /// a measured quantity, and the whole argument for that replacement is one
    /// error model: a signed log-sum-exp returns a result whose relative error
    /// is `≈ ε · Σ|terms| / |Σ terms|`, i.e. `ε · e^cancellation`. The
    /// admission bar `6.1` is the solve of `ε · e^cond = 1e-13` — so if the
    /// model over-promises, the bar is calibrated against a fiction and the
    /// refusal path is wrong in both directions at once.
    ///
    /// This grades the model where it is checkable, INCLUDING at rows the gate
    /// refuses (which the accuracy test above cannot assert on): the achieved
    /// error must sit under the predicted bar across the whole table, at every
    /// order, with a slack that covers the `√n` accumulation the single-`ε`
    /// model omits.
    #[test]
    fn log_survival_tower_cancellation_bounds_its_own_error_2714() {
        /// Slack over `ε · e^cancellation`. The model prices one rounding per
        /// term; an `n`-term compensationless sum accumulates like `√n · ε`,
        /// and `n` reaches [`LOG_SURVIVAL_PANEL_MAX_NODES`] = 4097, so `√n ≈
        /// 64`. Rounded up to 128 — one doubling of margin over the bound the
        /// model itself implies, which is the point at which a violation is a
        /// statement about the model rather than about the last two bits.
        const ERROR_MODEL_SLACK: f64 = 128.0;
        // Floor on the predicted bar. At zero cancellation the model predicts
        // `ε`, but the entries are LOGARITHMS: an entry whose `|ln|·||` is
        // 1.3e5 cannot be closer to the reference than the ulp of its own
        // magnitude, so the bar is `max(e^cancellation, |log_abs|)` — the
        // larger of the cancellation the sum suffered and the resolution of
        // the number it is reported in.
        let ctx = QuadratureContext::new();
        let mut worst_ratio = 0.0_f64;
        let mut worst_at = (f64::NAN, f64::NAN, 0usize);
        let mut observed_max_cancellation = 0.0_f64;
        for &(mu, sigma, expected_log, _) in LOG_SURVIVAL_TOWER_REFERENCE {
            let jet = log_survival_jet(&ctx, mu, sigma, 4);
            for order in 0..=4 {
                let entry = jet.scaled_mu_derivatives[order];
                if !entry.log_cancellation.is_finite() {
                    continue;
                }
                observed_max_cancellation =
                    observed_max_cancellation.max(entry.log_cancellation);
                let predicted = ERROR_MODEL_SLACK
                    * f64::EPSILON
                    * entry.log_cancellation.exp().max(entry.log_abs.abs().max(1.0));
                let error = (entry.log_abs - expected_log[order]).abs();
                let ratio = error / predicted;
                if ratio > worst_ratio {
                    worst_ratio = ratio;
                    worst_at = (mu, sigma, order);
                }
                assert!(
                    error <= predicted,
                    "#2714: the tower's own cancellation bar is not honest at \
                     (mu={mu}, sigma={sigma}, j={order}): it reports \
                     {:.4} nats of cancellation, which the signed-log-sum-exp \
                     error model prices at {predicted:.3e}, and the achieved \
                     error against the high-precision reference is \
                     {error:.3e}. The admission bar \
                     LOG_SURVIVAL_TOWER_MAX_LOG_CANCELLATION is derived FROM \
                     that model, so a model that under-predicts makes the bar \
                     meaningless.",
                    entry.log_cancellation
                );
            }
        }
        assert!(
            observed_max_cancellation > LOG_SURVIVAL_TOWER_MAX_LOG_CANCELLATION,
            "#2714: no row of the reference reaches the admission bar \
             ({LOG_SURVIVAL_TOWER_MAX_LOG_CANCELLATION} nats), so this test \
             never grades the model in the regime the bar exists for — worst \
             observed cancellation was {observed_max_cancellation:.3}"
        );
        let (mu, sigma, order) = worst_at;
        println!(
            "[2714] cancellation bar: worst achieved/predicted = {worst_ratio:.3e} \
             at (mu={mu}, sigma={sigma}, j={order}); max cancellation \
             {observed_max_cancellation:.3} nats"
        );
    }

    /// #2714, the smoothness half: `ln S` has no step in it.
    ///
    /// The stall was not caused by `ln S` being inaccurate in the abstract — it
    /// was caused by the error being a DISCONTINUOUS function of `(mu, sigma)`,
    /// so that a step the model predicted from a gradient was scored by an
    /// objective that had jumped onto a different approximation in between.
    /// The instrument is Richardson agreement of the second difference at `h`
    /// and `2h`. For a smooth `f` both estimate `f''` and differ by
    /// `-h^2 f''''/4`, which for this family is a fixed `h^2/4 = 2.5e-7` of
    /// `f''` regardless of `mu` (`ln S ~ -e^mu`, so every derivative has the
    /// same magnitude). A routing step of size `d` instead moves the `h`
    /// estimate by `d/h^2` and the `2h` estimate by `d/4h^2`, so it cannot
    /// cancel: the pair disagrees by `0.75 d/h^2 = 7.5e5 d`. That is why the
    /// test compares two stencils rather than bounding one — the bound would
    /// have to know `f''`, and the disagreement does not.
    #[test]
    fn log_survival_has_no_routing_step_in_it_2714() {
        let ctx = QuadratureContext::new();
        let step = 1.0e-3_f64;
        for &sigma in &[0.05_f64, 0.15, 0.5, 1.0, 4.0, 8.0, 20.0] {
            let mut worst = 0.0_f64;
            let mut worst_mu = f64::NAN;
            let value = |mu: f64| log_survival_jet(&ctx, mu, sigma, 0).log_survival;
            let mut index = -100_i32;
            while index <= 100 {
                let mu = f64::from(index) * 0.1;
                let mid = value(mu);
                let fine =
                    (value(mu - step) - 2.0 * mid + value(mu + step)) / (step * step);
                let coarse = (value(mu - 2.0 * step) - 2.0 * mid + value(mu + 2.0 * step))
                    / (4.0 * step * step);
                let disagreement = (fine - coarse).abs() / (1.0 + fine.abs());
                if disagreement > worst {
                    worst = disagreement;
                    worst_mu = mu;
                }
                index += 1;
            }
            assert!(
                worst <= 2.0e-6,
                "#2714: the h and 2h second differences of ln S(., sigma={sigma}) \
                 disagree by {worst:.3e} (relative) near mu={worst_mu}, i.e. a jump \
                 of about {:.3e} in the value. The surface must be smooth across \
                 every internal routing decision, because that is precisely what \
                 the joint-Newton accept test differentiates.",
                worst * step * step / 0.75
            );
            println!("[2714] sigma={sigma}: worst Richardson disagreement {worst:.3e}");
        }
    }

    /// #2714 / #2469: the node ladder is LIVE, and bounded.
    ///
    /// This is the successor to `..._node_ladder_is_inert_at_a_constant_513`,
    /// which pinned the opposite property: floor and ceiling coincided at 513,
    /// so `cloglog_gumbel_quad_nodes` returned a constant and the `SCALE/sigma`
    /// arithmetic around it reached no output. That inertness is exactly why the
    /// escape-hatch quadrature was wrong by `4.4e+06` at small sigma — it could
    /// not buy the resolution its own transition needed.
    ///
    /// The panel rule has no such suppressed dial: the count follows the
    /// integrand's local-scale arclength plus a `sqrt(sigma)` analyticity term,
    /// and both halves are load-bearing. The assertion is two-sided so a future
    /// change that re-pins the count fails on the mechanism.
    #[test]
    fn log_survival_panel_node_count_is_live_and_bounded_2714() {
        let mut counts = Vec::new();
        for &sigma in &[1.0e-4_f64, 1.0e-2, 0.15, 1.0, 8.0, 60.0, 1.0e3] {
            let panel = log_survival_panel(LogSurvivalBranch::Survival, 1.8, sigma, 0);
            assert!(
                panel.nodes >= LOG_SURVIVAL_PANEL_MIN_NODES
                    && panel.nodes <= LOG_SURVIVAL_PANEL_MAX_NODES,
                "#2714: node count {} at sigma={sigma} is outside \
                 [{LOG_SURVIVAL_PANEL_MIN_NODES}, {LOG_SURVIVAL_PANEL_MAX_NODES}]",
                panel.nodes
            );
            assert!(
                !panel.nodes.is_multiple_of(2),
                "#2714: the Clenshaw-Curtis panel needs an odd node count so the \
                 grid is symmetric and gap-free (sigma={sigma} gave {})",
                panel.nodes
            );
            assert!(
                panel.z_lo < panel.z_hi,
                "#2714: degenerate panel at sigma={sigma}"
            );
            counts.push(panel.nodes);
        }
        let (low, high) = (
            *counts.iter().min().expect("non-empty"),
            *counts.iter().max().expect("non-empty"),
        );
        assert!(
            high >= 4 * low,
            "#2714: the node ladder must actually MOVE with sigma — it spans \
             {low}..{high} over sigma in [1e-4, 1e3], and a ladder that does not \
             move is the #2469 inertness that made the escape-hatch quadrature \
             unable to resolve its own transition"
        );
    }

    /// The panel is placed by the integrand, so it must bracket the peak and
    /// carry the requested number of e-folds on both sides. If it did not, the
    /// accuracy gate above could still pass by luck at the sampled points.
    #[test]
    fn log_survival_panel_brackets_its_own_peak_2714() {
        for &(mu, sigma) in &[
            (-20.0, 0.002),
            (-3.0, 0.15),
            (0.0, 1.0),
            (3.2, 0.15),
            (12.0, 0.002),
            (0.0, 60.0),
        ] {
            for branch in [LogSurvivalBranch::Survival, LogSurvivalBranch::Complement] {
                let panel = log_survival_panel(branch, mu, sigma, 0);
                let mid = 0.5 * (panel.z_lo + panel.z_hi);
                let at_mid = branch.log_integrand(mu, sigma, mid);
                let at_lo = branch.log_integrand(mu, sigma, panel.z_lo);
                let at_hi = branch.log_integrand(mu, sigma, panel.z_hi);
                assert!(
                    at_lo < at_mid && at_hi < at_mid,
                    "#2714: {branch:?} panel at (mu={mu}, sigma={sigma}) does not \
                     bracket its own maximum: L(lo)={at_lo:.6e}, L(mid)={at_mid:.6e}, \
                     L(hi)={at_hi:.6e}"
                );
            }
        }
    }
}
