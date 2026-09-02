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
use std::sync::{Arc, Mutex, OnceLock};

use crate::estimate::EstimationError;
use crate::mixture_link::{
    beta_logistic_inverse_link_jet, component_inverse_link_jet, sas_inverse_link_jet,
};
use gam_math::probability::{erfcx_nonnegative, normal_logcdf};
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

#[derive(Clone, Copy, Debug, Default)]
struct Complex {
    re: f64,
    im: f64,
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

const LOGIT_SIGMA_DEGENERATE: f64 = 1e-10;
const LOGIT_ERFCX_SIGMA_MIN: f64 = 2.5e-1;
const LOGIT_TAIL_LOG_MAX: f64 = -18.0;
const LOGIT_ERFCX_MU_MAX: f64 = 40.0;
const LOGIT_ERFCX_SIGMA_MAX: f64 = 6.0;
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
const CLOGLOG_SIGMA_TAYLOR_MAX: f64 = 0.25;
/// Latent SD above which the cloglog integrated jet stops trusting shifted
/// lognormal-Laplace moment reconstruction for higher derivatives. The moments
/// `E[u^m exp(-u)]`, `u=exp(eta)`, evaluate the survival term at
/// `mu + m*sigma^2`; for d3 at sigma=4 that asks for a shift of 48 and loses
/// the small k3 contribution to cancellation. Directly integrating the stable
/// pointwise derivatives keeps the location-family jet identity intact.
const CLOGLOG_JET_MOMENT_SIGMA_MAX: f64 = 1.0;
const CLOGLOG_RARE_EVENT_LOG_MAX: f64 = -18.0;
const CLOGLOG_LARGE_SIGMA_ASYMPTOTIC_MIN: f64 = 8.0;
const CLOGLOG_POSITIVE_SATURATION_EDGE: f64 = 5.0;
const CLOGLOG_POSITIVE_SATURATION_SIGMAS: f64 = 8.0;
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
pub const LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER: usize = 8;
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
const SERIES_CONSECUTIVE_SMALL_TERMS: usize = 6;
const LOGIT_MAX_TERMS: usize = 160;
/// Documented absolute-accuracy contract of the erfcx logistic-normal
/// backend. The series truncation bound (see `logistic_normal_series_cutoff`)
/// is guaranteed to be below this tolerance on the mean and its μ-derivative
/// whenever the backend
/// returns a value. Beyond the eligibility window or when the a-priori
/// truncation index would exceed LOGIT_MAX_TERMS, the backend rejects and
/// the caller routes to GHQ.
///
/// Set to 1e-11 so that the erfcx branch only commits to a value when it
/// can honor the sharp tolerances used by downstream consumers. Oracle and
/// jet-match tests pin to `max_relative = 1e-10`; at 1e-11 the series rejects
/// in the central
/// band near (μ=1.1, σ=0.8) where the tail bound reaches ~2.6e-5 at
/// N=160, correctly deferring to GHQ which is accurate to ~1e-13 in that
/// regime after the QR eigenvector fix.
const LOGIT_ERFCX_ACCURACY_TARGET: f64 = 1.0e-11;
const CLOGLOG_MILES_ALPHA: f64 = 60.0;
const CLOGLOG_MILES_MAX_TERMS: usize = 256;
// Upper bound on the (log of the) peak Miles-series term magnitude under which
// the alternating cancellation still leaves a usable result in f64.
//
// The Miles series for S(mu, sigma) has term magnitudes whose log peaks at
// `peak_log(mu, sigma) ≈ α − (mu − ln α)² / (2 σ²)` near n = α. The final S is
// O(1), so a peak of `exp(peak_log)` is summed with alternating signs and must
// cancel down to ~1. f64 has ~53 bits, so after losing roughly peak_log/ln(2)
// bits to cancellation, the residual carries `53 − peak_log/ln(2)` bits of
// precision. Setting the cap at 0 means the peak term magnitude is bounded by
// 1, so the alternating sum never reaches into regions where bits get spent on
// cancellation at all. Outside this gate the caller drops down to CC / Gamma /
// GHQ, which evaluate the same survival object on numerically stable grids and
// do not depend on telescoping huge cancellations. The exit from "Miles
// reliable" to "fall back" therefore happens at peak terms of size 1, so the
// two backends already agree on the boundary at full f64 precision and the
// integrated cloglog mean remains monotone in `mu` as the routing switches.
const CLOGLOG_MILES_PEAK_LOG_MAX: f64 = 0.0;
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

fn cloglog_cc_required_nodes(mu: f64, sigma: f64, tol: f64) -> Result<usize, EstimationError> {
    if !(mu.is_finite() && sigma.is_finite() && sigma > 0.0 && tol.is_finite() && tol > 0.0) {
        crate::bail_invalid_estim!(
            "CC cloglog backend requires finite mu, positive sigma, and positive tolerance"
                .to_string(),
        );
    }

    // This mirrors the node-count logic used by the actual CC evaluator, but
    // exposes it as a cheap routing estimate so we can decide whether the
    // bounded real-line cosine grid is likely to beat the fixed-work complex
    // Gamma backend before paying to evaluate either one.
    let p_tail = (tol / 8.0).clamp(1e-300, 0.25);
    let a = gam_math::probability::standard_normal_quantile(p_tail)
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
        crate::bail_invalid_estim!("CC cloglog backend ellipse bound became degenerate");
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
/// Returns: (μ, dμ/dm)
#[inline]
pub fn logit_posterior_meanwith_deriv(
    eta: f64,
    se_eta: f64,
) -> Result<(f64, f64), EstimationError> {
    // Production routing for the integrated logistic-normal mean and its
    // location derivative.
    //
    // The backend ladder is:
    // - exact point-mass limit when sigma ~= 0
    // - adaptive quadrature at small sigma, where the erfcx series is
    //   cancellation-prone
    // - exact erfcx/Faddeeva series on the moderate domain
    // - a certified extreme-tail asymptotic
    // - adaptive quadrature wherever no analytic representation carries the
    //   required accuracy certificate.
    let out = logit_posterior_meanwith_deriv_controlled(eta, se_eta)?;
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
fn logistic_normal_exact_eligible(mu: f64, sigma: f64) -> bool {
    mu.is_finite()
        && sigma.is_finite()
        && mu.abs() <= LOGIT_ERFCX_MU_MAX
        && (LOGIT_ERFCX_SIGMA_MIN..=LOGIT_ERFCX_SIGMA_MAX).contains(&sigma)
}

/// A-priori truncation index for the erfcx series of the logistic-normal mean
/// **and its μ-derivative**, or `None` when no index ≤ `LOGIT_MAX_TERMS` can
/// certify both to `target_accuracy`.
///
/// The representation is
///
/// ```text
/// E[sigmoid(η)] = Φ(m/s)
///     + (1/2) · exp(-m²/(2s²))
///       · Σ_{k≥1} (-1)^(k-1) · [erfcx((k s² + m)/(√2 s))
///                             − erfcx((k s² − m)/(√2 s))]
/// ```
///
/// with m = |μ|, s = σ > 0 (the reflection μ→−μ is applied at the callsite).
/// The two erfcx arguments scale as k·s/√2 with a fixed offset, so both tend
/// to +∞ linearly in k. Using the asymptotic erfcx(x) = (1/(x√π))·[1 + O(1/x²)]
/// for large x, the k-th (signed) term and its μ-derivative have magnitudes
///
/// ```text
/// |T_k|  = m · √(2/π) · exp(-m²/(2s²)) / (k² · s³)        + O(1/k⁴)
/// |T_k'| = 2 · exp(-m²/(2s²)) · |m²−s²| / (√(2π) · s⁵ · k²) + O(1/k⁴)
/// ```
///
/// Because the series alternates in sign, the truncation tail after N terms is
/// bounded by the first omitted term — **but only once the terms are past their
/// magnitude peak**, which sits near k ≈ m/s² (where the erfcx argument
/// `(k s² − m)/(√2 s)` crosses zero). Below the peak the term magnitudes can
/// *grow* with k, so the alternating-series remainder bound is invalid there;
/// truncating before the peak would silently undersell the tail. We therefore
/// require N to exceed the peak in addition to satisfying both tail bounds:
///
/// ```text
/// |R_N(mean)|  ≤ coeff_mean  / (N+1)²   with coeff_mean  = m·√(2/π)·e^{-m²/2s²}/s³
/// |R_N(deriv)| ≤ coeff_deriv / (N+1)²   with coeff_deriv = 2·|m²−s²|·e^{-m²/2s²}/(√(2π)·s⁵)
/// N ≥ ⌈m/s²⌉ + 1                         (past the magnitude peak)
/// ```
///
/// Solving each tail bound for the smallest admissible N and taking the maximum
/// (also with the peak floor) yields the returned index. Reaching it bounds the
/// leading-order truncation error of *both* outputs; the adaptive-Simpson
/// drift-check in `logit_posterior_meanwith_deriv_controlled` remains the hard
/// backstop for the residual higher-order terms (notably near m ≈ s, where the
/// `|m²−s²|` derivative coefficient vanishes and the next order dominates).
#[inline]
fn logistic_normal_series_cutoff(mu: f64, sigma: f64, target_accuracy: f64) -> Option<usize> {
    assert!(sigma > 0.0);
    assert!(target_accuracy > 0.0);
    let m = mu.abs();
    let s = sigma;
    let gauss = (-(m * m) / (2.0 * s * s)).exp();
    let coeff_mean = m * (2.0_f64 / std::f64::consts::PI).sqrt() * gauss / (s * s * s);
    let coeff_deriv =
        2.0 * gauss * (m * m - s * s).abs() / ((2.0 * std::f64::consts::PI).sqrt() * s.powi(5));
    // Index past which the first-omitted-term bound for a given leading
    // coefficient drops to `target_accuracy`. A non-finite or already-tiny
    // coefficient imposes no constraint (returns 0).
    let asymptotic_index = |coeff: f64| -> f64 {
        if !coeff.is_finite() || coeff <= target_accuracy {
            0.0
        } else {
            (coeff / target_accuracy).sqrt() - 1.0
        }
    };
    // The alternating-tail bound is only valid past the magnitude peak at
    // k ≈ m/s²; enforce N strictly beyond it so the remainder ≤ first-omitted
    // term argument holds for both the mean and the derivative series.
    let peak_floor = m / (s * s) + 1.0;
    let required = asymptotic_index(coeff_mean)
        .max(asymptotic_index(coeff_deriv))
        .max(peak_floor);
    if !required.is_finite() || required > LOGIT_MAX_TERMS as f64 {
        return None;
    }
    // Evaluate at least a few pairs to pick up short-range structure the
    // asymptotic bound undersells; this only ever runs extra certified terms.
    Some((required.ceil() as usize).max(4))
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

#[inline]
fn logit_tail_asymptotic(mu: f64, sigma: f64) -> Option<IntegratedMeanDerivative> {
    // When mu is far out in either logistic tail, sigmoid(eta) is
    // exponentially close to either exp(eta) or 1 - exp(-eta). Those Gaussian
    // expectations collapse to lognormal moments, so we can route extreme-|mu|
    // cases away from both erfcx and GHQ.
    if mu <= 0.0 {
        let log_mean = mu + 0.5 * sigma * sigma;
        if log_mean <= LOGIT_TAIL_LOG_MAX {
            let mean = safe_exp(log_mean);
            return Some(IntegratedMeanDerivative {
                mean,
                dmean_dmu: mean,
                mode: IntegratedExpectationMode::ControlledAsymptotic,
            });
        }
    } else {
        let log_tail = -mu + 0.5 * sigma * sigma;
        if log_tail <= LOGIT_TAIL_LOG_MAX {
            let tail = safe_exp(log_tail);
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
    // - small sigma uses adaptive quadrature because the erfcx series is
    //   cancellation-prone and a finite heat-kernel truncation is not exact,
    // - moderate central cases prefer the exact erfcx/Faddeeva series,
    // - and extreme tails / very large sigma prefer controlled asymptotics.
    //
    // Validation target for this ladder: compare against high-order GHQ
    // (e.g. 128 nodes) on sigma in {0.01, 0.1, 1, 5, 20, 100} and mu on
    // [-10, 10] to confirm the regime transitions.
    if !(mu.is_finite() && sigma.is_finite()) {
        crate::bail_invalid_estim!("logit exact expectation requires finite mu and sigma");
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
    if logistic_normal_exact_eligible(mu, sigma)
        && let Ok(out) = logit_posterior_meanwith_deriv_exact_erfcx(mu, sigma)
    {
        return Ok(out);
    }
    // No analytic representation carries an accuracy certificate here: the
    // erfcx series was ineligible or could not certify its truncation within
    // LOGIT_MAX_TERMS. We deliberately return Err rather than fall back to the
    // Monahan-Stefanski probit approximation (Φ(μκ)), which carries ~1e-1
    // absolute error at moderate σ and, being returned as `Ok`, would bypass
    // the controlled router's drift-check and corrupt the posterior mean
    // (#571). The router maps this Err to the accurate adaptive-Simpson
    // fallback instead.
    Err(EstimationError::InvalidInput(
        "logit analytic expectation has no certified representation in this regime".to_string(),
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
    // bound for BOTH the mean and its μ-derivative, evaluated past the series
    // magnitude peak (see `logistic_normal_series_cutoff`), is below the
    // documented `LOGIT_ERFCX_ACCURACY_TARGET`. Reaching N* is thus an a-priori
    // estimate of accuracy for both outputs; the adaptive-Simpson drift-check
    // in the controlled router is the hard backstop. The only way this routine
    // rejects is when N* would exceed LOGIT_MAX_TERMS, at which point the
    // accuracy contract cannot be honored and the caller routes elsewhere.
    let m = mu.abs();
    let s = sigma;
    let z = SQRT_2 * s;
    let phi_term = gam_math::probability::normal_cdf(m / s);
    let phi_prime = gam_math::probability::normal_pdf(m / s) / s;
    let Some(max_k) = logistic_normal_series_cutoff(mu, sigma, LOGIT_ERFCX_ACCURACY_TARGET) else {
        crate::bail_invalid_estim!(
            "logit erfcx series truncation bound exceeds LOGIT_MAX_TERMS at the required accuracy"
                .to_string(),
        );
    };

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
        crate::bail_invalid_estim!("logit erfcx expectation produced non-finite values");
    }
    Ok(IntegratedMeanDerivative {
        mean,
        dmean_dmu: dmean,
        mode: IntegratedExpectationMode::ExactSpecialFunction,
    })
}

/// Accurate logistic-normal mean and location-derivative via adaptive Simpson.
/// `sigmoid` and `sigmoid' = sigmoid·(1−sigmoid)` are smooth and bounded, so
/// `integrate_normal_adaptive` resolves both to ~1e-12 at every sigma — the
/// trusted reference / fallback when the closed-form ladder is out of regime.
#[inline]
fn logit_posterior_meanwith_deriv_quadrature(mu: f64, sigma: f64) -> IntegratedMeanDerivative {
    let mean = integrate_normal_adaptive(mu, sigma, |x| stable_sigmoidwith_derivative(x).0);
    let dmean_dmu =
        integrate_normal_adaptive(mu, sigma, |x| stable_sigmoidwith_derivative(x).1).max(0.0);
    IntegratedMeanDerivative {
        mean,
        dmean_dmu,
        mode: IntegratedExpectationMode::QuadratureFallback,
    }
}

#[inline]
fn logit_posterior_meanwith_deriv_controlled(
    mu: f64,
    sigma: f64,
) -> Result<IntegratedMeanDerivative, EstimationError> {
    if !(mu.is_finite() && sigma.is_finite()) {
        crate::bail_invalid_estim!("logit integrated moments require finite mu and sigma");
    }
    let candidate = match logit_posterior_meanwith_deriv_exact(mu, sigma) {
        Ok(out) => out,
        Err(_) => return Ok(logit_posterior_meanwith_deriv_quadrature(mu, sigma)),
    };
    // Defense-in-depth drift-check. The erfcx series now sizes its truncation
    // from the per-output tail bounds past the magnitude peak (mean AND
    // derivative — see `logistic_normal_series_cutoff`), so the
    // `ExactSpecialFunction` candidate is accurate by construction; the
    // adaptive-Simpson reference confirms it and absorbs the residual
    // higher-order terms (e.g. near m ≈ s where the derivative coefficient
    // vanishes). `ControlledAsymptotic` covers only the extreme-|μ|
    // lognormal-collapse approximation, which is likewise confirmed against
    // the reference. Small-σ, exact point-mass, and erfcx-ineligible regimes
    // route to their mathematically appropriate implementations rather than
    // returning a tolerance-accepted finite Taylor truncation (#571, #2623).
    match candidate.mode {
        IntegratedExpectationMode::ExactSpecialFunction
        | IntegratedExpectationMode::ControlledAsymptotic => {
            let reference = logit_posterior_meanwith_deriv_quadrature(mu, sigma);
            if integrated_mean_derivative_drift_exceeds(
                &candidate, &reference, 1e-6, 1e-4, 1e-7, 1e-3,
            ) {
                Ok(reference)
            } else {
                Ok(candidate)
            }
        }
        _ => Ok(candidate),
    }
}

#[inline]
fn cloglog_extreme_asymptotic(mu: f64, sigma: f64) -> Option<IntegratedMeanDerivative> {
    // Extreme-input ladder for the cloglog mean and its location derivative.
    //
    // Regimes:
    // - mu + sigma^2 / 2 << 0: rare-event tail, where 1 - exp(-exp(eta)) ~= exp(eta)
    // - mu - 8 sigma >> 0: survival term is numerically indistinguishable from 0
    //
    // The large-σ regime is intentionally NOT handled here (see the trailing
    // comment); the thresholds otherwise leave overlap with the Taylor/Miles/
    // Gamma branches so neighboring formulas still cover the transition band.
    let rare_log = mu + 0.5 * sigma * sigma;
    if rare_log <= CLOGLOG_RARE_EVENT_LOG_MAX {
        let mean = safe_exp(rare_log);
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
    // The large-σ regime (σ ≥ CLOGLOG_LARGE_SIGMA_ASYMPTOTIC_MIN) is handled
    // upstream in cloglog_posterior_meanwith_deriv_controlled via the accurate
    // log-space Gumbel survival quadrature, so this ladder no longer carries the
    // leading-order "sharp transition" split (it was biased low by 2–7%, #799,
    // and zeroed the derivative through value-space underflow, #798).
    None
}

#[inline]
fn cloglog_survival_extreme_asymptotic(
    mu: f64,
    sigma: f64,
) -> Option<(f64, IntegratedExpectationMode)> {
    let rare_log = mu + 0.5 * sigma * sigma;
    if rare_log <= CLOGLOG_RARE_EVENT_LOG_MAX {
        let mean = safe_exp(rare_log);
        return Some((
            (1.0 - mean).clamp(0.0, 1.0),
            IntegratedExpectationMode::ControlledAsymptotic,
        ));
    }
    if mu - CLOGLOG_POSITIVE_SATURATION_SIGMAS * sigma >= CLOGLOG_POSITIVE_SATURATION_EDGE {
        // For σ < CLOGLOG_LARGE_SIGMA_ASYMPTOTIC_MIN this deep in the positive
        // tail S is below ~1e-300, so the value path's hard zero is exact to
        // f64. The genuine log-magnitude (needed by the kernel derivative path)
        // is recovered separately by cloglog_log_survival_term_controlled.
        return Some((0.0, IntegratedExpectationMode::ControlledAsymptotic));
    }
    // σ ≥ CLOGLOG_LARGE_SIGMA_ASYMPTOTIC_MIN is handled by the caller via the
    // accurate log-space Gumbel quadrature (replaces the biased step-model
    // split, #799).
    None
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
    let mut widen = 0;
    while residual(lo) > 0.0 && widen < 4096 {
        lo *= 2.0;
        widen += 1;
        if !lo.is_finite() {
            return f64::MIN;
        }
    }
    let mut z = if lo > -1.0e6 { 0.5 * (lo + hi) } else { lo };
    for _ in 0..200 {
        let r = residual(z);
        if r > 0.0 {
            hi = z;
        } else {
            lo = z;
        }
        // d/dz [ln σ + μ + σz − ln(−z)] = σ + 1/(−z) > 0.
        let slope = sigma + 1.0 / (-z);
        let mut next = z - r / slope;
        if !(next > lo && next < hi) {
            next = 0.5 * (lo + hi);
        }
        if (next - z).abs() <= f64::EPSILON * (1.0 + z.abs()) {
            return next;
        }
        z = next;
    }
    z
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
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if branch.log_integrand_slope(mu, sigma, mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo <= f64::EPSILON * (1.0 + hi.abs()) {
            break;
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
    let mut widen = 0;
    while !fallen(outer) && widen < 4096 {
        inner = outer;
        step *= 2.0;
        outer = z_peak + direction * step;
        widen += 1;
        if !outer.is_finite() {
            return inner;
        }
    }
    let (mut lo, mut hi) = if direction > 0.0 {
        (inner, outer)
    } else {
        (outer, inner)
    };
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if fallen(mid) == (direction > 0.0) {
            hi = mid;
        } else {
            lo = mid;
        }
        if hi - lo <= f64::EPSILON * (1.0 + mid.abs()) {
            break;
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
fn cloglog_small_sigma_taylor(mu: f64, sigma: f64) -> IntegratedMeanDerivative {
    // Small-variance heat-kernel expansion for the cloglog inverse link.
    //
    // For η = μ + σ Z, Z ~ N(0,1), and any analytic f the heat-kernel
    // (even-moment) identity gives
    //
    //   E[f(η)] = Σ_{k≥0} σ^(2k) / (2^k · k!) · f^(2k)(μ).
    //
    // Here f(x) = 1 − exp(−exp(x)) is entire, so the series is valid
    // globally. Truncating at the σ⁶ term yields
    //
    //   E[f(η)]       ≈ f + (σ²/2) f'' + (σ⁴/8) f^(4) + (σ⁶/48) f^(6)
    //   d/dμ E[f(η)]  ≈ f' + (σ²/2) f''' + (σ⁴/8) f^(5) + (σ⁶/48) f^(7).
    //
    // Coefficients are heat-kernel weights 1/(2^k k!), not Taylor 1/(2k)!:
    // 1/(2²·2!) = 1/8 (not 1/4! = 1/24), 1/(2³·3!) = 1/48 (not 1/6! = 1/720).
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
            dmean_dmu: cloglog_mean_d1_exact(mu),
            mode: IntegratedExpectationMode::ExactClosedForm,
        };
    }

    let ex = safe_exp(mu);
    if !ex.is_finite() {
        // Non-finite μ in the positive direction saturates f to 1 and f' to 0.
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
    let s6 = s4 * s2;
    let s8 = s4 * s4;
    let e2x = ex * ex;
    let e3x = e2x * ex;
    let e4x = e3x * ex;
    let e5x = e4x * ex;
    let e6x = e5x * ex;
    let e7x = e6x * ex;
    let e8x = e7x * ex;
    let e9x = e8x * ex;
    // -expm1(-ex) = 1 - exp(-ex) is bit-exact even when ex is subnormal.
    //
    // Derivatives of f(x) = 1 - exp(-exp(x)) follow the Stirling-second-kind
    // pattern: f^(n)(x) = exp(-exp(x)) * sum_{k=1..n} (-1)^(k+1) S(n,k) u^k,
    // where u = exp(x) and S(n,k) are Stirling numbers of the second kind:
    //   S(2,.) = {1, 1}
    //   S(3,.) = {1, 3, 1}
    //   S(4,.) = {1, 7, 6, 1}
    //   S(5,.) = {1, 15, 25, 10, 1}
    //   S(6,.) = {1, 31, 90, 65, 15, 1}
    //   S(7,.) = {1, 63, 301, 350, 140, 21, 1}
    //   S(8,.) = {1, 127, 966, 1701, 1050, 266, 28, 1}
    //   S(9,.) = {1, 255, 3025, 7770, 6951, 2646, 462, 36, 1}
    let f0 = -(-ex).exp_m1();
    let f1 = ex * surv;
    let f2 = surv * (ex - e2x);
    let f3 = surv * (ex - 3.0 * e2x + e3x);
    let f4 = surv * (ex - 7.0 * e2x + 6.0 * e3x - e4x);
    let f5 = surv * (ex - 15.0 * e2x + 25.0 * e3x - 10.0 * e4x + e5x);
    let f6 = surv * (ex - 31.0 * e2x + 90.0 * e3x - 65.0 * e4x + 15.0 * e5x - e6x);
    let f7 = surv * (ex - 63.0 * e2x + 301.0 * e3x - 350.0 * e4x + 140.0 * e5x - 21.0 * e6x + e7x);
    let f8 = surv
        * (ex - 127.0 * e2x + 966.0 * e3x - 1701.0 * e4x + 1050.0 * e5x - 266.0 * e6x + 28.0 * e7x
            - e8x);
    let f9 = surv
        * (ex - 255.0 * e2x + 3025.0 * e3x - 7770.0 * e4x + 6951.0 * e5x - 2646.0 * e6x
            + 462.0 * e7x
            - 36.0 * e8x
            + e9x);
    // Heat-kernel coefficients 1/(2^k k!): k=1: 1/2, k=2: 1/8, k=3: 1/48,
    // k=4: 1/384. Truncation after sigma^8 leaves O(sigma^10) remainder,
    // comfortably below 1e-12 rel at sigma = 0.1 even in the negative tail.
    IntegratedMeanDerivative {
        mean: f0 + 0.5 * s2 * f2 + (s4 / 8.0) * f4 + (s6 / 48.0) * f6 + (s8 / 384.0) * f8,
        dmean_dmu: (f1 + 0.5 * s2 * f3 + (s4 / 8.0) * f5 + (s6 / 48.0) * f7 + (s8 / 384.0) * f9)
            .max(0.0),
        mode: IntegratedExpectationMode::ControlledAsymptotic,
    }
}

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
    if !(sigma.is_finite()) || sigma < 1e-10 {
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

fn cloglog_posterior_meanwith_deriv_quadrature(mu: f64, sigma: f64) -> IntegratedMeanDerivative {
    if sigma < 1e-10 {
        return IntegratedMeanDerivative {
            mean: cloglog_mean_exact(mu),
            dmean_dmu: cloglog_mean_d1_exact(mu),
            mode: IntegratedExpectationMode::ExactClosedForm,
        };
    }
    let mean = cloglog_mean_from_survival(survival_posterior_mean_quadrature(mu, sigma));
    let dmean_dmu = integrate_normal_adaptive(mu, sigma, cloglog_mean_d1_exact).max(0.0);
    IntegratedMeanDerivative {
        mean,
        dmean_dmu,
        mode: IntegratedExpectationMode::QuadratureFallback,
    }
}

#[inline]
fn survival_posterior_mean_quadrature(eta: f64, se_eta: f64) -> f64 {
    integrate_normal_adaptive(eta, se_eta, gumbel_survival).clamp(0.0, 1.0)
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
    if sigma >= CLOGLOG_LARGE_SIGMA_ASYMPTOTIC_MIN {
        // Accurate large-σ survival from the log-space survival panel,
        // replacing the leading-order "sharp transition" split that was biased
        // low by 2–7% across σ ∈ [8, 20] (#799). Exponentiating ln S here loses
        // nothing on the value path (S is consumed as a probability); the
        // log-magnitude that the kernel derivative path needs is taken straight
        // from cloglog_log_survival_term_controlled, which reads the same panel.
        let log_s = log_survival_jet(ctx, mu, sigma, 0).log_survival;
        return (
            safe_exp(log_s).clamp(0.0, 1.0),
            IntegratedExpectationMode::ControlledAsymptotic,
        );
    }
    if cloglog_survival_miles_is_reliable(mu, sigma)
        && let Ok(out) = cloglog_survival_miles(mu, sigma)
    {
        return (
            out.clamp(0.0, 1.0),
            IntegratedExpectationMode::ExactSpecialFunction,
        );
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
        survival_posterior_mean_quadrature(mu, sigma),
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
    if cloglog_survival_miles_is_reliable(mu, sigma)
        && cloglog_survival_miles_is_reliable(shiftedmu, sigma)
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
    cloglog_shift_identity_derivative_log(mu, sigma, shifted_survival.ln())
}

/// Log-domain form of [`cloglog_shift_identity_derivative`] that takes
/// `ln S(mu + sigma^2, sigma)` directly.
///
/// This is the underflow-safe path: when the shifted survival `S(mu+σ²,σ)` is
/// below the f64 floor (large σ, #798), the value form above sees
/// `shifted_survival == 0` and returns a spurious zero slope, whereas the
/// genuine derivative `exp(mu + σ²/2) · S(mu+σ², σ)` is finite and O(1) because
/// the huge prefix exactly compensates the tiny survival. Carrying the survival
/// as a log keeps that cancellation exact.
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
    // O(1): 0.5*erfc(u) = Phi(-u*sqrt(2)), so log(0.5*erfc(u)) is exactly
    // normal_logcdf(-u*sqrt(2)) — reusing the full-precision (libm-erfc) primitive
    // instead of statrs::erfc (which carried ~1e-10 relative error, #932).
    if u > 0.0 {
        -u * u + (0.5 * erfcx_nonnegative(u)).ln()
    } else {
        normal_logcdf(-u * SQRT_2)
    }
}

/// True when the Miles erfc-gated lognormal-Laplace series can be summed in
/// f64 without the alternating-cancellation transient destroying the result.
///
/// Background. The Miles representation of `S(mu, sigma) = E[exp(-exp(eta))]`
/// is a real series
///
/// ```text
///   S = Σ_{n≥0} (-1)^n / n! · exp(mu n + ½ σ² n²) · ½ erfc(u_n)
///   u_n = (mu − ln α + σ² n) / (√2 σ).
/// ```
///
/// For large positive `u_n`, `½ erfc(u_n)` decays like `exp(-u_n²) / (√(2π) u_n)`.
/// Substituting the asymptotic and using Stirling on `ln n!`, the log of the
/// `n`-th term magnitude reduces to
///
/// ```text
///   log|t_n| ≈ n ln(α / n) + n − ½ (mu − ln α)² / σ²,
/// ```
///
/// which is maximised at `n = α` with peak value
///
/// ```text
///   peak_log(mu, sigma) = α − ½ (mu − ln α)² / σ².
/// ```
///
/// The series telescopes down to `S ∈ [0, 1]`, so once `peak_log` exceeds
/// `CLOGLOG_MILES_PEAK_LOG_MAX` the partial sums sweep through magnitudes
/// `exp(peak_log)` and the residual after cancellation no longer carries enough
/// f64 precision to be a reliable answer. The fixed `|mu|/σ ≥ 3` gate that
/// used to guard the Miles call was a proxy for "tail-dominated", not for
/// "series reliable" — it misses precisely the band `mu ∈ (ln α − √(2 α σ²),
/// ln α + √(2 α σ²))` where the peak term blows up, and several values of mu in
/// that band were already empirically returning a clamped-but-wrong S
/// (e.g. the latent cloglog inverse link was producing μ = 0.94 instead of
/// μ ≈ 0.07 for `mu ≈ −3.2, σ = 1`).
///
/// This predicate replaces the proxy with the actual reliability condition.
/// Callers should drop to the CC / Gamma / GHQ branches when it returns false,
/// which all evaluate the same survival object on numerically stable grids.
#[inline]
fn cloglog_survival_miles_is_reliable(mu: f64, sigma: f64) -> bool {
    if !(mu.is_finite() && sigma.is_finite() && sigma > 0.0) {
        return false;
    }
    let alpha_ln = CLOGLOG_MILES_ALPHA.ln();
    let shifted = mu - alpha_ln;
    let peak_log = CLOGLOG_MILES_ALPHA - 0.5 * shifted * shifted / (sigma * sigma);
    peak_log.is_finite() && peak_log <= CLOGLOG_MILES_PEAK_LOG_MAX
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
            let term_log = base_log + log_half_erfc;
            if term_log > QUADRATURE_EXP_LOG_MAX {
                crate::bail_invalid_estim!("Miles cloglog series term exceeded finite exp range");
            }
            let term = sign * safe_exp(term_log);
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
        crate::bail_invalid_estim!(
            "CC cloglog backend requires finite mu, positive sigma, and positive tolerance"
                .to_string(),
        );
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
    let a = gam_math::probability::standard_normal_quantile(p_tail)
        .map(|z| -z)
        .unwrap_or(8.0)
        .max(1.0);
    let n = cloglog_cc_required_nodes(mu, sigma, tol)?;
    if n > CLOGLOG_CC_NODE_CAP {
        crate::bail_invalid_estim!("CC cloglog backend requires too many nodes");
    }

    let rule = ctx.clenshaw_curtis_n(n);
    let inv_sqrt_2pi = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    let mut sum = 0.0_f64;
    let mut c = 0.0_f64;
    for (&x, &w) in rule.nodes.iter().zip(rule.weights.iter()) {
        let t = a * x;
        let u = mu + sigma * t;
        let e = safe_exp(u);
        let w0 = (-0.5 * t * t).exp() * inv_sqrt_2pi;
        let yk = w * w0 * (-e).exp() - c;
        let tk = sum + yk;
        c = (tk - sum) - yk;
        sum = tk;
    }

    let survival = (a * sum).clamp(0.0, 1.0);
    if !survival.is_finite() {
        crate::bail_invalid_estim!("CC cloglog backend produced non-finite values");
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

// `cloglog_posterior_meanwith_deriv_gamma_reference` is a test reference
// implementation; it lives inside `mod tests` below.

fn cloglog_survival_gamma_reference(mu: f64, sigma: f64) -> Result<f64, EstimationError> {
    if !(mu.is_finite() && sigma.is_finite()) || sigma <= 0.0 {
        crate::bail_invalid_estim!(
            "Gamma cloglog reference backend requires finite mu and positive sigma"
        );
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
        crate::bail_invalid_estim!("Gamma cloglog reference backend produced non-finite values");
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
    // ApproxKind: NumericalApproximation — each branch carries its own
    // backward error bound (special-function or GHQ tail), composed so the
    // worst-case error is the max across regimes; documented at each branch.
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
    if sigma >= CLOGLOG_LARGE_SIGMA_ASYMPTOTIC_MIN {
        // Large-σ regime: form both the mean and the location derivative from the
        // accurate, underflow-safe log-space survival. This replaces the biased
        // step-model split (mean too low by 2–7%, #799) and, because the
        // derivative is exp(μ + σ²/2)·S(μ+σ², σ) carried entirely in log space,
        // it no longer collapses to zero when S(μ+σ², σ) underflows (#798).
        let (log_base, base_mode) = cloglog_log_survival_term_controlled(ctx, mu, sigma);
        let (log_shift, shift_mode) =
            cloglog_log_survival_term_controlled(ctx, mu + sigma * sigma, sigma);
        // mean = 1 − S = −expm1(ln S), stable for S near both 0 and 1.
        let mean = (-log_base.exp_m1()).clamp(0.0, 1.0);
        let dmean = cloglog_shift_identity_derivative_log(mu, sigma, log_shift);
        return IntegratedMeanDerivative {
            mean,
            dmean_dmu: dmean.max(0.0),
            mode: worse_integrated_expectation_mode(base_mode, shift_mode),
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
            return cloglog_posterior_meanwith_deriv_quadrature(mu, sigma);
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
    let ghq = cloglog_posterior_meanwith_deriv_quadrature(mu, sigma);
    // Drift tolerances tightened on the derivative absolute floor: the
    // Taylor truncation diverges in the positive-saturation band
    // (e.g. mu ~ 3, sigma ~ 0.24) because f^(n) grow near the saturation
    // transition, and a 1e-5 absolute floor hid 1e-6-scale errors there.
    // GHQ is the trusted evaluator in that regime because f' has negligible
    // probability outside the Gaussian 3-sigma window.
    if integrated_mean_derivative_drift_exceeds(&candidate, &ghq, 1e-6, 1e-4, 1e-7, 1e-3) {
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
        LinkFunction::Logit => logit_posterior_meanwith_deriv_controlled(mu, sigma),
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
                mode: if sigma <= 1e-10 {
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
            let mode = if sigma <= 1e-10 {
                IntegratedExpectationMode::ExactClosedForm
            } else {
                // Mirror the scalar controlled-path mode when it accepts the
                // exact erfcx backend; otherwise the node-sum above is a
                // quadrature fallback.
                match logit_posterior_meanwith_deriv_controlled(mu, sigma) {
                    Ok(scalar) => scalar.mode,
                    Err(_) => IntegratedExpectationMode::QuadratureFallback,
                }
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
                mode: if sigma <= 1e-10 {
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
    let scalar = logit_posterior_meanwith_deriv_controlled(mu, sigma)?;
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
            .unwrap_or_else(|error| {
                log::debug!(
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
    const PROB_EPS: f64 = 1e-12;
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
                    variance: (mean * (1.0 - mean)).max(PROB_EPS),
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
                    variance: (mean * (1.0 - mean)).max(PROB_EPS),
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
                    variance: (mean * (1.0 - mean)).max(PROB_EPS),
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
                    variance: (mean * (1.0 - mean)).max(PROB_EPS),
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
                    variance: (mean * (1.0 - mean)).max(PROB_EPS),
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
                    variance: (mean * (1.0 - mean)).max(PROB_EPS),
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
            Ok(IntegratedMomentsJet {
                mean: e,
                variance,
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
                variance: (mean * (1.0 - mean)).max(PROB_EPS),
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
            let mean = jet.mean.clamp(PROB_EPS, 1.0 - PROB_EPS);
            Ok(IntegratedMomentsJet {
                mean,
                variance: (mean * (1.0 - mean) / (1.0 + precision)).max(PROB_EPS),
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
                mode: jet.mode,
            })
        }
        ResponseFamily::Poisson
        | ResponseFamily::Tweedie { .. }
        | ResponseFamily::NegativeBinomial { .. }
        | ResponseFamily::Gamma => {
            // Log-normal MGF: E[exp(η)] = exp(e + s²/2)
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
                // Unreachable: this match arm is only entered for the four families
                // in the enclosing `Poisson | Tweedie | NegativeBinomial | Gamma`
                // pattern, all handled above.
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
        mode: if sigma <= 1e-10 {
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

/// Stack-allocated Cholesky factor for `D x D` symmetric PSD matrices.
///
/// Returns the lower-triangular factor `L` (with strict upper triangle = 0)
/// such that `L L^T = cov`, or `None` if `cov` is not positive definite
/// (non-finite or non-positive pivot encountered).
///
/// This mirrors a standard textbook Cholesky inner loop bit-for-bit at a
/// single jitter level, but avoids any heap allocation — critical for
/// per-row GHQ where this runs once per observation.
#[inline]
fn cholesky_static<const D: usize>(cov: &[[f64; D]; D]) -> Option<[[f64; D]; D]> {
    let mut l = [[0.0_f64; D]; D];
    for i in 0..D {
        for j in 0..=i {
            let mut sum = cov[i][j];
            for k in 0..j {
                sum -= l[i][k] * l[j][k];
            }
            if i == j {
                if !sum.is_finite() || sum <= 0.0 {
                    return None;
                }
                l[i][j] = sum.sqrt();
            } else {
                l[i][j] = sum / l[j][j];
            }
        }
    }
    Some(l)
}

/// Stack-allocated Cholesky with a jitter-retry ladder
/// (0, 1e-12, 1e-11, …, 1e-6 added to diagonal).
#[inline]
fn cholesky_static_with_jitter<const D: usize>(cov: &[[f64; D]; D]) -> Option<[[f64; D]; D]> {
    if D == 0 {
        return None;
    }
    for retry in 0..8 {
        let jitter = if retry == 0 {
            0.0
        } else {
            1e-12 * 10f64.powi(retry - 1)
        };
        if jitter == 0.0 {
            if let Some(l) = cholesky_static::<D>(cov) {
                return Some(l);
            }
        } else {
            let mut base = *cov;
            for i in 0..D {
                base[i][i] = cov[i][i] + jitter;
            }
            if let Some(l) = cholesky_static::<D>(&base) {
                return Some(l);
            }
        }
    }
    None
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
    // then run a stack-allocated Cholesky-with-jitter. This avoids the
    // `Vec<Vec<f64>>` per-row allocation that previously serialized
    // through the global allocator inside parallel workers.
    let mut cov_arr = cov;
    for i in 0..D {
        cov_arr[i][i] = cov_arr[i][i].max(0.0);
    }
    let Some(l) = cholesky_static_with_jitter::<D>(&cov_arr) else {
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
        return gam_math::probability::normal_cdf(eta);
    }
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
    use gam_problem::LikelihoodScaleMetadata;
    use gam_spec::LikelihoodSpec;

    /// Pins `log_half_erfc_stable` (both the `u > 0` erfcx branch and the
    /// `u <= 0` `normal_logcdf` branch) against an external high-precision
    /// reference (mpmath, dps=50) for `log(0.5·erfc(u))`. Guards the #932
    /// root-cause fix: the `u <= 0` branch previously routed through
    /// `statrs::erfc` (~1e-10 relative error); the 1e-12 tolerance here fails
    /// on any regression to a low-accuracy complementary error function.
    #[test]
    fn log_half_erfc_stable_matches_high_precision_reference() {
        let refs: &[(f64, f64)] = &[
            (-3.0, -1.1045309498499094e-5),
            (-1.5, -0.017092677825984745),
            (-0.5, -0.27410803278438573),
            (0.0, -0.69314718055994531),
            (0.7, -1.8257336940742865),
            (2.0, -6.0580884451765829),
            (5.0, -27.89403672609738),
            (12.0, -147.75386135854695),
        ];
        for &(u, reference) in refs {
            let got = log_half_erfc_stable(u);
            let rel = (got - reference).abs() / reference.abs().max(1.0e-6);
            assert!(
                rel < 1.0e-12,
                "log_half_erfc_stable({u}) = {got:.17e}, reference {reference:.17e}, \
                 rel {rel:.3e} >= 1e-12"
            );
        }
    }

    pub(crate) fn cloglog_posterior_meanwith_deriv_gamma_reference(
        mu: f64,
        sigma: f64,
    ) -> Result<IntegratedMeanDerivative, EstimationError> {
        // Reference: mean = 1 - S(mu, sigma), dmean/dmu = exp(mu + sigma^2/2) *
        // S(mu + sigma^2, sigma).
        let survival = cloglog_survival_gamma_reference(mu, sigma)?;
        let shifted_survival = cloglog_survival_gamma_reference(mu + sigma * sigma, sigma)?;
        let mean = cloglog_mean_from_survival(survival);
        let dmean = cloglog_shift_identity_derivative(mu, sigma, shifted_survival);
        if !(mean.is_finite() && dmean.is_finite()) {
            crate::bail_invalid_estim!(
                "Gamma cloglog reference backend produced non-finite values"
            );
        }
        Ok(IntegratedMeanDerivative {
            mean,
            dmean_dmu: dmean.max(0.0),
            mode: IntegratedExpectationMode::ExactSpecialFunction,
        })
    }

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
    fn test_logit_posterior_derivative_matches_central_difference() {
        let ctx = QuadratureContext::new();
        let eta = 1.7;
        let se = 0.9;
        let h = 1e-5;

        let (_, dmu) = logit_posterior_meanwith_deriv(eta, se)
            .expect("logit posterior mean derivative should evaluate");
        let mu_plus = logit_posterior_mean(&ctx, eta + h, se);
        let mu_minus = logit_posterior_mean(&ctx, eta - h, se);
        let dmufd = (mu_plus - mu_minus) / (2.0 * h);

        assert_eq!(dmu.signum(), dmufd.signum());
        assert_relative_eq!(dmu, dmufd, epsilon = 5e-6, max_relative = 2e-4);
    }

    /// Independent dense reference for `E[sigmoid(η)]`, `η ~ N(mu, sigma²)`,
    /// using composite Simpson on a wide grid under N(0,1). No Gauss–Hermite,
    /// no Faddeeva, no erfcx — a method-independent arbiter accurate to
    /// ~1e-13 for the smooth bounded integrand. This is the test the #1459
    /// oracle is held to.
    fn dense_sigmoid_normal_mean(mu: f64, sigma: f64) -> f64 {
        let a = -18.0_f64;
        let b = 18.0_f64;
        let n = 400_000usize; // even
        let h = (b - a) / n as f64;
        let integrand = |z: f64| -> f64 { sigmoid(mu + sigma * z) * normal_pdf(z) };
        let mut sum = integrand(a) + integrand(b);
        for i in 1..n {
            let z = a + (i as f64) * h;
            sum += if i % 2 == 0 { 2.0 } else { 4.0 } * integrand(z);
        }
        sum * h / 3.0
    }

    #[test]
    fn test_logit_posterior_mean_exact_symmetry_identity() {
        // sigmoid is odd-symmetric about 1/2, so E[sigmoid(η;μ)] +
        // E[sigmoid(η;−μ)] = 1 exactly; the oracle must honor it to ~f64.
        let cases = [
            (-3.0, 0.5),
            (-1.2, 1.7),
            (0.0, 2.2),
            (2.3, 0.8),
            (3.0, 0.05),
        ];
        for (mu, sigma) in cases {
            let p = logit_posterior_mean_exact(mu, sigma);
            let q = logit_posterior_mean_exact(-mu, sigma);
            assert!(
                (p + q - 1.0).abs() < 1e-12,
                "symmetry broken at mu={mu} sigma={sigma}: p+q-1 = {:.3e}",
                p + q - 1.0
            );
        }
    }

    #[test]
    fn test_logit_posterior_mean_exact_matches_high_res_integral() {
        // Spans small σ (where erfcx-style schemes underflow), moderate σ, and
        // both signs of μ. The pre-#1459 4096-term truncation failed these by
        // 1e-5 (μ-linear); the accelerated oracle holds 1e-10.
        let cases = [
            (-2.0, 0.4),
            (-0.7, 1.1),
            (0.8, 0.9),
            (2.4, 1.7),
            (3.0, 0.05),
            (3.0, 0.5),
            (-2.0, 2.0),
            (5.0, 3.0),
        ];
        for (mu, sigma) in cases {
            let exact = logit_posterior_mean_exact(mu, sigma);
            let numeric = dense_sigmoid_normal_mean(mu, sigma);
            assert!(
                (exact - numeric).abs() < 1e-10,
                "oracle ≠ dense reference at mu={mu} sigma={sigma}: \
                 exact={exact:.13} ref={numeric:.13} err={:.3e}",
                (exact - numeric).abs()
            );
        }
    }

    /// Regression for #1459: the Faddeeva-pole oracle carried a μ-linear,
    /// σ-independent bias toward 1/2 of `μ/(2π²·4096) ≈ 1.236e-5·μ` because it
    /// hard-truncated an O(1/N) series at 4096 terms. This reproduces the exact
    /// table from the bug report and demands the oracle resolve `E[sigmoid(η)]`
    /// to 1e-10 — four orders tighter than the bug — including the diagnostic
    /// structure (the error was *identical* across σ at fixed μ).
    #[test]
    fn test_logit_posterior_mean_exact_no_truncation_bias_1459() {
        // Full Cartesian grid {1,3,-2} x {0.02,0.05,0.5,2.0} (12 cases; salvaged
        // from PR #1462 by HomunculusLabs — was an 8-case hand-picked subset).
        let table = [
            (1.0, 0.02),
            (1.0, 0.05),
            (1.0, 0.5),
            (1.0, 2.0),
            (3.0, 0.02),
            (3.0, 0.05),
            (3.0, 0.5),
            (3.0, 2.0),
            (-2.0, 0.02),
            (-2.0, 0.05),
            (-2.0, 0.5),
            (-2.0, 2.0),
        ];
        for (mu, sigma) in table {
            let exact = logit_posterior_mean_exact(mu, sigma);
            let reference = dense_sigmoid_normal_mean(mu, sigma);
            let err = (exact - reference).abs();
            assert!(
                err < 1e-10,
                "#1459 truncation bias resurfaced at mu={mu} sigma={sigma}: \
                 err={err:.3e} (pre-fix bias here was ~{:.2e})",
                mu.abs() / (2.0 * std::f64::consts::PI.powi(2) * 4096.0)
            );
        }

        // The defining symptom: at fixed μ the old bias was constant in σ. The
        // fixed oracle must have *no* such σ-independent residual — the spread
        // of (oracle − reference) across σ at μ=3 must be ~round-off, not the
        // old 3.71e-5 plateau.
        let mu = 3.0;
        let errs: Vec<f64> = [0.05, 0.5, 2.0]
            .iter()
            .map(|&s| logit_posterior_mean_exact(mu, s) - dense_sigmoid_normal_mean(mu, s))
            .collect();
        for e in &errs {
            assert!(
                e.abs() < 1e-10,
                "residual {e:.3e} at mu=3 — old σ-independent plateau was 3.71e-5"
            );
        }
    }

    /// The new Weideman Faddeeva evaluator must match known `w(z)` values to
    /// near machine precision on the upper half-plane. References are
    /// machine-precision values of `w(z)` (SciPy `wofz` / mpmath), NOT the
    /// crate's `erfcx_nonnegative` — which this very check revealed to be only
    /// ~6e-11 accurate (it inherits `statrs::erfc`'s rational-approx error),
    /// so using it as the reference would both slacken the bound and certify
    /// against a wrong value.
    #[test]
    fn test_faddeeva_weideman_matches_known_values() {
        // w(0) = 1.
        let w0 = faddeeva_upper_halfplane(Complex { re: 0.0, im: 0.0 });
        assert!(
            (w0.re - 1.0).abs() < 1e-13 && w0.im.abs() < 1e-13,
            "w(0)={w0:?}"
        );
        // w(i·y) is purely real and equals erfcx(y) for y>0 (reference: wofz).
        let on_axis = [
            (0.1, 0.8964569799691268),
            (0.5, 0.6156903441929258),
            (1.0, 0.427583576155807),
            (2.0, 0.2553956763105058),
            (5.0, 0.11070463773306861),
            (9.0, 0.06230772403777468),
        ];
        for (y, want) in on_axis {
            let w = faddeeva_upper_halfplane(Complex { re: 0.0, im: y });
            assert!(
                (w.re - want).abs() < 1e-13 && w.im.abs() < 1e-13,
                "w(i·{y}): got {w:?}, want re={want}, err={:.2e}",
                (w.re - want).abs()
            );
        }
        // Off-axis values across the upper half-plane (reference: wofz).
        let off_axis = [
            ((0.7, 1.3), (0.31327301971562715, 0.12443489420104513)),
            ((-1.5, 0.8), (0.21066359024766423, -0.27001624496296617)),
            ((3.0, 0.4), (0.030278754646989155, 0.1957320888774461)),
        ];
        for ((re, im), (wre, wim)) in off_axis {
            let w = faddeeva_upper_halfplane(Complex { re, im });
            assert!(
                (w.re - wre).abs() < 1e-13 && (w.im - wim).abs() < 1e-13,
                "w({re}+{im}i): got {w:?}, want ({wre},{wim})"
            );
        }
        // Large |z| (deep in the series tail, |z|≈40): must stay machine-precise,
        // not merely match the leading i/(√π z) asymptotic (which is only ~4e-6
        // accurate there). Reference: wofz(3+40i).
        let w = faddeeva_upper_halfplane(Complex { re: 3.0, im: 40.0 });
        assert!(
            (w.re - 0.01402158696172506).abs() < 1e-13
                && (w.im - 0.0010509664408184546).abs() < 1e-13,
            "tail value mismatch: w={w:?}"
        );
    }

    #[test]
    fn test_integrated_logit_mean_close_to_exact_oracle() {
        // The production integrated-logit path (erfcx series + Simpson
        // drift-check) is ~1e-8 accurate; the oracle is now ~1e-13, so it can
        // certify the production path far more tightly than the old 2.5e-3.
        let ctx = QuadratureContext::new();
        let cases = [(-3.0, 0.3), (-1.0, 0.8), (0.5, 1.2), (2.8, 1.0)];
        for (eta, se) in cases {
            let ghq = logit_posterior_mean(&ctx, eta, se);
            let exact = logit_posterior_mean_exact(eta, se);
            assert!(
                (ghq - exact).abs() < 1e-6,
                "production path drifts from oracle at eta={eta} se={se}: \
                 ghq={ghq:.12} oracle={exact:.12} gap={:.3e}",
                (ghq - exact).abs()
            );
        }
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
        // Assertion redesign: at (μ=1.1, σ=0.8) the erfcx series cannot
        // reach its EPSILON=1e-10 tail bound within LOGIT_MAX_TERMS=160
        // (|R_N| ≈ 0.67/(N+1)², so N* ≈ 81619), and
        // `logit_posterior_meanwith_deriv_exact` correctly returns Err.
        // The value-accuracy contract lives at the controlled dispatcher,
        // which falls back to GHQ when the exact series cannot honor the
        // contract; that is what we validate here, against an independent
        // high-resolution Simpson reference for BOTH the mean and its
        // μ-derivative (d/dμ E[sigmoid] = E[sigmoid']).
        let out = logit_posterior_meanwith_deriv_controlled(1.1, 0.8).expect("controlled logit");
        let (ref_mean, ref_d1, _, _) = logit_reference_jet_highres_simpson(1.1, 0.8);
        assert_relative_eq!(out.mean, ref_mean, epsilon = 1e-11, max_relative = 1e-10);
        assert!(out.dmean_dmu > 0.0);
        assert_relative_eq!(out.dmean_dmu, ref_d1, epsilon = 1e-11, max_relative = 1e-10);
    }

    #[test]
    fn test_logit_small_sigma_returns_quadrature_truth_2623() {
        // A second-order heat-kernel truncation has O(sigma^4) error. The old
        // dispatcher computed this accurate reference but returned the
        // truncation whenever it was merely within 1e-6, enough to reverse a
        // near-tied posterior-probability pair in issue #2623. Small-sigma
        // logistic-normal means now return the quadrature value itself.
        for &(mu, sigma) in &[
            (-3.0, 0.05),
            (-0.5, 0.10),
            (0.0, 0.20),
            (0.5, 0.24),
            (3.0, 0.15),
        ] {
            let out =
                logit_posterior_meanwith_deriv_controlled(mu, sigma).expect("controlled logit");
            let (ref_mean, ref_d1, _, _) = logit_reference_jet_highres_simpson(mu, sigma);
            assert_eq!(out.mode, IntegratedExpectationMode::QuadratureFallback);
            assert_relative_eq!(out.mean, ref_mean, epsilon = 1e-12, max_relative = 1e-11);
            assert_relative_eq!(
                out.dmean_dmu,
                ref_d1,
                epsilon = 1e-12,
                max_relative = 1e-11
            );
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
            deriv_mc += cloglog_mean_d1_exact(eta);
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
        assert_relative_eq!(out.dmean_dmu, ref_d1, epsilon = 1e-11, max_relative = 1e-10);
    }

    #[test]
    fn test_logit_dispatch_large_sigma_uses_accurate_quadrature_not_monahan() {
        // Regression for #571. At (μ=0.5, σ=20) the case is erfcx-ineligible
        // (σ > LOGIT_ERFCX_SIGMA_MAX) and not in any tail/Taylor regime. The
        // old code returned the Monahan–Stefanski probit Φ(μκ) here — wrong by
        // ~6e-3 absolute — as a trusted `Ok`, bypassing the drift-check. The
        // corrected path returns `Err` from the analytic ladder, so the
        // controlled router routes straight to accurate adaptive-Simpson
        // quadrature. Assert the route is GHQ/quadrature (NOT a trusted
        // asymptotic) and that the value matches an independent reference.
        let ctx = QuadratureContext::new();
        let out = integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, 0.5, 20.0)
            .expect("logit integrated inverse-link moments should evaluate");
        assert_eq!(out.mode, IntegratedExpectationMode::QuadratureFallback);
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
        let out = logit_posterior_meanwith_deriv_controlled(1.1, 0.8).expect("logit controlled");
        assert!(matches!(
            out.mode,
            IntegratedExpectationMode::ExactSpecialFunction
                | IntegratedExpectationMode::QuadratureFallback
        ));
        let (ref_mean, ref_d1, _, _) = logit_reference_jet_highres_simpson(1.1, 0.8);
        assert_relative_eq!(out.mean, ref_mean, epsilon = 1e-11, max_relative = 1e-10);
        assert_relative_eq!(out.dmean_dmu, ref_d1, epsilon = 1e-11, max_relative = 1e-10);
    }

    #[test]
    fn test_logit_dispatch_derivative_correct_at_mu_zero_small_sigma() {
        // Regression for #572. On the erfcx branch at μ=0 the old mean-only
        // truncation cutoff returned the clamp floor (4 terms), leaving the
        // derivative series uncancelled: it reported dmean_dmu ≈ 0.58 at
        // (0, 0.3) — a factor ~2.4 too large and physically impossible, since
        // sigmoid'(0)=0.25 and averaging over a Gaussian can only shrink it.
        // The corrected cutoff sizes the truncation from the derivative tail
        // bound past the series peak; at small σ this exceeds LOGIT_MAX_TERMS,
        // so the branch honestly bails to accurate quadrature.
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
    fn test_logit_erfcx_exact_branch_is_self_certified() {
        // Regression for #572: the `ExactSpecialFunction` branch must be
        // accurate *by itself*, not merely rescued by the controlled router's
        // drift-check. Call `logit_posterior_meanwith_deriv_exact` directly
        // (no quadrature net) in the large-|μ| band where the erfcx series
        // certifies within LOGIT_MAX_TERMS, and require both the mean and the
        // μ-derivative to match an independent high-resolution reference.
        for &(mu, sigma) in &[(8.0, 1.0), (10.0, 1.0), (15.0, 2.0)] {
            let out = logit_posterior_meanwith_deriv_exact(mu, sigma)
                .expect("erfcx branch should certify");
            assert_eq!(out.mode, IntegratedExpectationMode::ExactSpecialFunction);
            let (ref_mean, ref_d1, _, _) = logit_reference_jet_highres_simpson(mu, sigma);
            assert_relative_eq!(out.mean, ref_mean, epsilon = 1e-9, max_relative = 1e-7);
            assert_relative_eq!(out.dmean_dmu, ref_d1, epsilon = 1e-9, max_relative = 1e-7);
        }
        // Where the series cannot certify the derivative within LOGIT_MAX_TERMS
        // it must reject (Err) rather than return a wrong "exact" value — the
        // router then routes to quadrature. (0, 0.3) is the #572 point.
        assert!(
            logit_posterior_meanwith_deriv_exact(0.0, 0.3).is_err(),
            "erfcx branch must not claim ExactSpecialFunction when it cannot certify the derivative"
        );
    }

    #[test]
    fn test_logit_integrated_derivative_is_even_in_mu() {
        // d/dμ E[sigmoid(η)] = E[sigmoid'(η)] and sigmoid' is even, so the
        // location-derivative is even in μ. The erfcx series works in m=|μ|;
        // #572 originated in a botched sign/reflection of that derivative.
        // Pin exact symmetry across regimes (erfcx-success, erfcx-bail/GHQ,
        // and tail-asymptotic).
        let ctx = QuadratureContext::new();
        for &(mu, sigma) in &[(0.3, 0.3), (1.1, 0.8), (10.0, 1.0), (3.0, 3.0), (35.0, 1.0)] {
            let pos =
                integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, mu, sigma)
                    .expect("logit moments (+μ)");
            let neg =
                integrated_inverse_link_mean_and_derivative(&ctx, LinkFunction::Logit, -mu, sigma)
                    .expect("logit moments (-μ)");
            assert_relative_eq!(
                pos.dmean_dmu,
                neg.dmean_dmu,
                epsilon = 1e-9,
                max_relative = 1e-7
            );
            // And the mean reflects: E[sigmoid] at -μ equals 1 - E[sigmoid] at μ.
            assert_relative_eq!(
                neg.mean,
                1.0 - pos.mean,
                epsilon = 1e-9,
                max_relative = 1e-7
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
        // derivative inconsistent with its own mean. Grid points are chosen well
        // inside single regimes (away from the σ∈{0.25,6} and |μ|=40 branch
        // seams) so the mean is locally smooth and a tight FD is meaningful:
        //   - quadrature-fallback band (erfcx-eligible but un-certifiable),
        //   - erfcx self-certified band (large |μ|),
        //   - small-σ quadrature band,
        //   - large-σ (erfcx-ineligible) band.
        let ctx = QuadratureContext::new();
        let h = 1e-4;
        let cases = [
            (0.0, 0.8),  // quadrature fallback, μ=0 (the #572 failure family)
            (0.7, 0.8),  // quadrature fallback, off-center
            (1.5, 1.2),  // quadrature fallback
            (-1.1, 0.9), // quadrature fallback, μ<0 (reflection path)
            (8.0, 1.0),  // erfcx self-certified
            (10.0, 1.5), // erfcx self-certified
            (-9.0, 1.0), // erfcx self-certified, μ<0
            (0.5, 0.05), // small-σ quadrature
            (0.5, 20.0), // large-σ, erfcx-ineligible → quadrature
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
            // The scalar path now routes to accurate adaptive-Simpson, matching
            // the independent high-resolution Simpson reference (truth) to ~1e-10
            // — the Monahan ~0.11 error is gone.
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

    #[test]
    fn integrated_family_moments_use_scale_dispersion_for_tweedie_and_gamma() {
        // Regression for #953: the log-normal arm's observation-model variance
        // must read the Tweedie dispersion φ / Gamma shape k from the supplied
        // `LikelihoodScaleMetadata`, not assume φ = 1 (Tweedie) / k = 1 (Gamma).
        let ctx = QuadratureContext::new();
        // Deterministic small inputs; integrated mean m = exp(e + s²/2).
        let e = 0.3_f64;
        let se = 0.5_f64;
        let m = (e + 0.5 * se * se).exp();

        // Tweedie p = 1.5, φ = 2: Var = φ · m^p (the old code returned m^p, i.e. φ = 1).
        let p = 1.5_f64;
        let phi = 2.0_f64;
        let tweedie = LikelihoodSpec::tweedie_log(p);
        let tweedie_likelihood = GlmLikelihoodSpec {
            spec: tweedie.clone(),
            scale: LikelihoodScaleMetadata::EstimatedTweediePhi { phi },
        };
        let out = integrated_family_moments_jet(&ctx, &tweedie_likelihood, e, se)
            .expect("tweedie integrated moments should evaluate");
        let expected = phi * m.powf(p);
        assert_relative_eq!(out.variance, expected, epsilon = 1e-12);
        // Guard against the φ = 1 regression: the corrected value is φ× the old one.
        assert_relative_eq!(out.variance / m.powf(p), phi, epsilon = 1e-12);

        // Gamma shape k = 4: Var = m² / k = φ·m² with φ = 1/k (old code: m², i.e. k = 1).
        let shape = 4.0_f64;
        let gamma = LikelihoodSpec::gamma_log();
        let gamma_likelihood = GlmLikelihoodSpec {
            spec: gamma.clone(),
            scale: LikelihoodScaleMetadata::EstimatedGammaShape { shape },
        };
        let out = integrated_family_moments_jet(&ctx, &gamma_likelihood, e, se)
            .expect("gamma integrated moments should evaluate");
        let expected = m * m / shape;
        assert_relative_eq!(out.variance, expected, epsilon = 1e-12);
        // Guard against the k = 1 regression: the corrected value is (1/k)× the old one.
        assert_relative_eq!(out.variance / (m * m), 1.0 / shape, epsilon = 1e-12);

        // Poisson is φ ≡ 1, Var = m, independent of the (unit) scale label.
        let poisson = LikelihoodSpec::poisson_log();
        let poisson_likelihood = GlmLikelihoodSpec::canonical(poisson);
        let out = integrated_family_moments_jet(&ctx, &poisson_likelihood, e, se)
            .expect("poisson integrated moments should evaluate");
        assert_relative_eq!(out.variance, m, epsilon = 1e-12);

        // NB2 with theta = 3: Var = m + m²/θ, unchanged by this fix.
        let theta = 3.0_f64;
        let nb = LikelihoodSpec::negative_binomial_log(theta);
        let nb_likelihood = GlmLikelihoodSpec::canonical(nb);
        let out = integrated_family_moments_jet(&ctx, &nb_likelihood, e, se)
            .expect("negative-binomial integrated moments should evaluate");
        assert_relative_eq!(out.variance, m + m * m / theta, epsilon = 1e-12);

        // Missing Gamma dispersion metadata is rejected, not silently φ = 1.
        let missing_gamma = GlmLikelihoodSpec {
            spec: gamma,
            scale: LikelihoodScaleMetadata::Unspecified,
        };
        let err = integrated_family_moments_jet(&ctx, &missing_gamma, e, se)
            .expect_err("gamma without a shape in the scale metadata must error");
        assert!(
            format!("{err}").contains("GammaShape"),
            "unexpected error message: {err}"
        );

        // Likewise a Tweedie response with no dispersion φ in the metadata.
        let missing_tweedie = GlmLikelihoodSpec {
            spec: tweedie,
            scale: LikelihoodScaleMetadata::Unspecified,
        };
        let err = integrated_family_moments_jet(&ctx, &missing_tweedie, e, se)
            .expect_err("tweedie without a φ in the scale metadata must error");
        assert!(
            format!("{err}").contains("EstimatedTweediePhi"),
            "unexpected error message: {err}"
        );
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
                assert!((0.0..=1.0).contains(&val), "L({mu},{sigma}) = {val}");
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
        assert_relative_eq!(d.l_mumu, g2, epsilon = 1e-14);
        assert_relative_eq!(d.l_mumumu, g3, epsilon = 1e-14);
        assert_relative_eq!(d.l_mumumumu, g4, epsilon = 1e-14);

        // Odd sigma-derivatives vanish at sigma=0 (odd Gaussian moments are 0).
        assert_eq!(d.l_sigma, 0.0);
        assert_eq!(d.l_musigma, 0.0);
        assert_eq!(d.l_mumusigma, 0.0);
        assert_eq!(d.l_mumumusigma, 0.0);
        assert_eq!(d.l_sigmasigmasigma, 0.0);
        assert_eq!(d.l_musigmasigmasigma, 0.0);

        // Even sigma-derivatives carry the surviving moments E[Z^2]=1, E[Z^4]=3:
        //   L_σσ = g'', L_μσσ = g''', L_μμσσ = g'''', L_σσσσ = 3 g''''.
        assert_relative_eq!(d.l_sigmasigma, g2, epsilon = 1e-14);
        assert_relative_eq!(d.l_musigmasigma, g3, epsilon = 1e-14);
        assert_relative_eq!(d.l_mumusigmasigma, g4, epsilon = 1e-14);
        assert_relative_eq!(d.l_sigmasigmasigmasigma, 3.0 * g4, epsilon = 1e-14);
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
            let stack = cholesky_static_with_jitter::<2>(cov).expect("stack cholesky");
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
            let stack = cholesky_static_with_jitter::<3>(cov).expect("stack cholesky");
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
        let l = cholesky_static_with_jitter::<1>(&[[2.25]]).expect("d=1");
        assert_eq!(l[0][0], 1.5);
        // Tiny negative diagonal (roundoff-scale) is rescued by the
        // additive jitter ladder (1e-12 … 1e-6). At retry 1 the diagonal
        // becomes -1e-13 + 1e-12 ≈ 9e-13 > 0, so Cholesky succeeds.
        // The original assertion here used `-1.0`, but additive jitter
        // capped at 1e-6 cannot recover a diagonal of -1.0 → -1.0+1e-6
        // < 0 for every retry, so that assertion was unsatisfiable under
        // the function's documented jitter ladder. The intent of the
        // assertion was clearly to cover the "rescued by jitter" path,
        // which is what a roundoff-scale negative diagonal exercises.
        assert!(cholesky_static_with_jitter::<1>(&[[-1.0e-13]]).is_some());
        // A negative variance triggers jitter; with jitter <= 1e-6 it still
        // can't reach positive — should return None.
        assert!(cholesky_static_with_jitter::<1>(&[[-1.0e3]]).is_none());
    }
}

#[cfg(test)]
mod log_survival_panel_2714_tests {
    use super::{
        LOG_SURVIVAL_MAX_MU_DERIVATIVE_ORDER, LOG_SURVIVAL_PANEL_MAX_NODES,
        LOG_SURVIVAL_PANEL_MIN_NODES, LOG_SURVIVAL_TOWER_MAX_LOG_CANCELLATION, LogSurvivalBranch,
        QuadratureContext, log_survival_jet, log_survival_panel,
    };

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
