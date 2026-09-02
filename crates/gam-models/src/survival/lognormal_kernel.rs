//! Shared analytic kernel for latent-variable families with lognormal structure.
//!
//! The kernel object `K_{k,m}(μ, σ) := E[exp(k·U − m·exp(U))]`, where
//! `U ~ N(μ, σ²)`, is the only special function required by all latent families.
//!
//! It satisfies exact μ-recurrences (see [`kernel_ratio_jet`]) and the
//! corresponding heat-equation σ-identities, so fixed-σ latent families reduce
//! to evaluating kernel bundles at shifted arguments.
//!
//! Row likelihoods for binary and survival models are small signed sums of
//! kernel terms; [`LogKernelSumJet`] evaluates their log-derivatives from
//! log-space kernel bundles and treats non-positive signed sums as invalid rows.

use crate::model_types::EstimationError;
use crate::probability::{log1mexp_positive, signed_log_sum_exp};
use crate::quadrature::{
    IntegratedExpectationMode, QuadratureContext, log_survival_jet,
    lognormal_laplace_unit_log_term_shared,
};
use serde::{Deserialize, Serialize};
use std::fmt;

// ─── Typed errors ────────────────────────────────────────────────────────────

/// Errors produced by the lognormal-kernel frailty/marginal-slope validators.
///
/// Public boundaries that historically returned `Result<_, String>` continue to
/// do so via `.map_err(|e| e.to_string())`; the `Display` impl reproduces the
/// original error strings byte-for-byte.
#[derive(Debug, Clone)]
pub enum LognormalKernelError {
    /// The chosen frailty modifier is not finite-state exact with the
    /// requested marginal-slope family.
    InvalidSpec { reason: String },
}

impl_reason_error_boilerplate! {
    LognormalKernelError {
        InvalidSpec,
    }
}

// ─── Frailty specification ───────────────────────────────────────────────────

/// How the hazard multiplier frailty loads onto the hazard components.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum HazardLoading {
    /// Frailty multiplies the entire hazard: h(t|U) = exp(U) · h_0(t).
    Full,
    /// Frailty multiplies only the disease-like component; an exogenous
    /// background ("Makeham") component is unloaded:
    ///   h(t|U) = exp(U) · h_loaded(t) + h_unloaded(t).
    /// This is the faithful model for Gompertz-Makeham.
    LoadedVsUnloaded,
}

/// Frailty modifier specification at the family level.
///
/// Two structurally different exact modifiers exist:
///
/// 1. **GaussianShift**: additive Gaussian on the final transformation index.
///    Exact for probit families — the existing sextic microcell kernel survives
///    unchanged (just scale denested cell coefficients by 1/√(1+σ²)).
///
/// 2. **HazardMultiplier**: lognormal multiplier on the loaded cumulative hazard.
///    Exact for PH/cloglog families — row likelihoods are finite sums of
///    K_{k,m}(μ, σ) kernel terms.
///
/// These are mathematically distinct families.  Do not mix them.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "scale_kind", rename_all = "kebab-case")]
pub enum FrailtyScale {
    Fixed { sigma: f64 },
    Learned { initial_sigma: f64 },
}

impl FrailtyScale {
    fn validate(self, kind: &str) -> Result<(), LognormalKernelError> {
        match self {
            Self::Fixed { sigma } if sigma.is_finite() && sigma >= 0.0 => Ok(()),
            Self::Fixed { sigma } => Err(LognormalKernelError::InvalidSpec {
                reason: format!(
                    "{kind} frailty Fixed scale requires finite sigma >= 0, got {sigma}"
                ),
            }),
            Self::Learned { initial_sigma }
                if initial_sigma.is_finite() && initial_sigma > 0.0 =>
            {
                Ok(())
            }
            Self::Learned { initial_sigma } => Err(LognormalKernelError::InvalidSpec {
                reason: format!(
                    "{kind} frailty Learned scale requires finite initial_sigma > 0, got {initial_sigma}"
                ),
            }),
        }
    }

    /// Exact log-sigma coordinate and declared finite chart domain for a
    /// learned scale. Fixed scales have no optimizer coordinate.
    pub(crate) fn learned_log_sigma_coordinate(self) -> Option<(f64, f64, f64)> {
        match self {
            Self::Fixed { .. } => None,
            Self::Learned { initial_sigma } => Some((initial_sigma.ln(), -12.0, 6.0)),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "frailty_kind", rename_all = "kebab-case")]
pub enum FrailtySpec {
    /// No frailty modifier.
    #[default]
    None,
    /// Gaussian shift on the final scalar index: U ~ N(0, σ²) added to η.
    /// Exact for probit: E[Φ(η + U)] = Φ(η / √(1+σ²)).
    /// The existing sextic microcell kernel is preserved.
    GaussianShift {
        scale: FrailtyScale,
    },
    /// Lognormal hazard multiplier: conditional hazard h(t|U) involves exp(U).
    /// Exact for PH/cloglog/survival via K_{k,m} kernel.
    HazardMultiplier {
        scale: FrailtyScale,
        /// How the multiplier loads onto hazard components.
        loading: HazardLoading,
    },
}

impl FrailtySpec {
    /// Whether this spec requests an actual frailty modifier.
    ///
    /// [`FrailtySpec::None`] is the sole "no frailty" value. Family/mode guards
    /// that only support the no-frailty case must reject on this predicate, or they
    /// misclassify every ordinary CLI fit as a frailty request.
    #[inline]
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// Validate the frailty scale domain independently of a model family.
    pub fn validate(&self) -> Result<(), LognormalKernelError> {
        let (kind, scale) = match self {
            Self::None => return Ok(()),
            Self::GaussianShift { scale } => ("GaussianShift", *scale),
            Self::HazardMultiplier { scale, .. } => ("HazardMultiplier", *scale),
        };
        scale.validate(kind)
    }

    /// Validate that this frailty spec is compatible with score_warp/linkwiggle
    /// cubic marginal-slope families.
    ///
    /// - `GaussianShift` is exact: the sextic microcell kernel is preserved
    ///   (probit scaling by 1/τ, τ = √(1+σ²)).
    /// - `HazardMultiplier` is exact only for PH/cloglog rowwise families.
    ///   It is NOT finite-state exact with score_warp/linkwiggle cubic
    ///   marginal-slope, because the multiplicative frailty breaks the
    ///   polynomial kernel closure that the cubic cell derivatives require.
    ///
    /// Returns an error if the combination is not exactly integrable.
    pub fn validate_for_marginal_slope(&self) -> Result<(), String> {
        self.validate_for_marginal_slope_typed()
            .map_err(|e| e.to_string())
    }

    /// Typed variant of [`Self::validate_for_marginal_slope`] used internally;
    /// the `String`-returning entry point above is preserved as a one-line
    /// shim for external callers.
    pub fn validate_for_marginal_slope_typed(&self) -> Result<(), LognormalKernelError> {
        self.validate()?;
        match self {
            Self::None | Self::GaussianShift { .. } => Ok(()),
            Self::HazardMultiplier { .. } => Err(LognormalKernelError::InvalidSpec {
                reason:
                    "HazardMultiplier frailty is not finite-state exact with score_warp/linkwiggle \
                     cubic marginal-slope families. Use GaussianShift frailty (exact probit scaling) \
                     or use the standalone latent-cloglog/latent-survival families instead."
                        .to_string(),
            }),
        }
    }
}

// ─── Probit frailty scaling ──────────────────────────────────────────────────

#[inline]
fn probit_frailty_scale_components(sigma: f64) -> (f64, f64) {
    let abs_sigma = sigma.abs();
    if abs_sigma > 1.0 {
        let inv = 1.0 / abs_sigma;
        let denom = 1.0 + inv * inv;
        (inv / denom.sqrt(), 1.0 / denom)
    } else {
        let sigma2 = sigma * sigma;
        let denom = 1.0 + sigma2;
        (1.0 / denom.sqrt(), sigma2 / denom)
    }
}

/// Probit frailty scaling factor **with** t-derivatives (t = log σ).
///
/// Provides exact closed-form derivatives of s = 1/√(1+σ²) with respect to
/// t = log(σ) for learnable Gaussian-shift frailty in the marginal-slope
/// families.  For Gaussian frailty on the final probit index
/// E[Φ(η + U)] = Φ(η · s) with s = 1/√(1+σ²); writing α = σ²/(1+σ²) the
/// derivatives are ∂_t s = −α·s and ∂_{tt} s = α(3α−2)·s.
#[derive(Clone, Copy, Debug)]
pub struct ProbitFrailtyScaleJet {
    /// s = 1/√(1+σ²)
    pub s: f64,
    /// α = σ²/(1+σ²)  — shared auxiliary for all derivative levels.
    pub alpha: f64,
    /// ∂_t s = -α·s
    pub ds: f64,
    /// ∂_{tt} s = α(3α−2)·s
    pub d2s: f64,
}

impl ProbitFrailtyScaleJet {
    /// Build the jet from σ (not from t = log σ).
    ///
    /// At σ = 0 the jet degenerates to (s=1, α=0, ds=0, d2s=0), which is
    /// correct: zero frailty means s ≡ 1 independent of t.
    pub fn new(sigma: f64) -> Self {
        let (s, alpha) = probit_frailty_scale_components(sigma);
        Self {
            s,
            alpha,
            ds: -alpha * s,
            d2s: alpha * (3.0 * alpha - 2.0) * s,
        }
    }

    /// Build the jet from t = log(σ) directly.
    pub fn from_log_sigma(log_sigma: f64) -> Self {
        Self::new(log_sigma.exp())
    }
}

#[inline]
fn worst_mode(
    a: IntegratedExpectationMode,
    b: IntegratedExpectationMode,
) -> IntegratedExpectationMode {
    if a.rank() >= b.rank() { a } else { b }
}

// ─── Log-space kernel infrastructure ──────────────────────────────────────────
//
// The runtime kernel path stays in log-space until the final ratios are formed,
// avoiding the overflow/underflow and cancellation problems that come from
// exponentiating individual terms too early.

/// Returns `log K_{k,m}(μ,σ)` directly, without exponentiation.
///
/// The value is always finite (or `NEG_INFINITY` when the kernel is zero), so
/// it cannot overflow or underflow.
#[inline]
fn validate_kernel_inputs(m: f64, mu: f64, sigma: f64) -> Result<(), EstimationError> {
    if !m.is_finite() || m < 0.0 {
        crate::bail_invalid_estim!("lognormal kernel requires finite m >= 0, got {m}");
    }
    if !mu.is_finite() || !sigma.is_finite() || sigma < 0.0 {
        crate::bail_invalid_estim!(
            "lognormal kernel requires finite mu and sigma >= 0, got mu={mu}, sigma={sigma}"
        );
    }
    Ok::<(), _>(())
}

/// Kernel bundle storing `log K_{k,m}` values instead of `K_{k,m}`.
#[derive(Clone, Debug)]
pub struct LogLognormalKernelBundle {
    pub log_values: Vec<f64>,
    /// The Laplace half of each `log_values` entry, kept apart from the
    /// analytic prefix (#2610).
    ///
    /// `log K_k = prefix_k + laplace_k` with `prefix_k = k·μ + σ²k²/2` known in
    /// closed form. Storing the two halves summed is enough to READ a kernel
    /// value but not enough to difference one accurately, because the prefix
    /// grows quadratically in `k` and swamps the Laplace part it is added to.
    /// Retaining `laplace_k` costs one `f64` per rung and is what lets
    /// [`Self::second_cumulant_ratio`] recover the exact part of a second
    /// difference instead of subtracting it away.
    pub log_laplace: Vec<f64>,
    /// `σ^j · ∂_a^j K_0` in signed-log coordinates for `j = 0..=max_k`, where
    /// `a = μ + ln m` is the SINGLE location the `k = 0` kernel depends on
    /// (#2610), or `None` when the analytic branch does not apply.
    ///
    /// `K_0(m,μ,σ) = S(μ + ln m, σ)` because `m` and `μ` enter only through the
    /// product `m·e^U`. That makes `∂_a` the one derivative the rung ladder is
    /// built out of:
    ///
    /// ```text
    /// m^k K_k = (−1)^k · ∂_a(∂_a − 1)···(∂_a − k + 1) K_0,
    /// ```
    ///
    /// so a term list in the `K_k` basis and one in the `∂_a^j K_0` basis carry
    /// the SAME information — but not the same conditioning. Reaching
    /// `∂_a^4 K_0` through the rungs means evaluating `−mK_1 + 7m²K_2 − 6m³K_3 +
    /// m⁴K_4`, whose summands are `O(1/σ)` while the sum is `O(1/σ^5)`; reading
    /// it from here is one quadrature over an integrand that is already small.
    /// The scaling by `σ^j` is what keeps every entry inside the representable
    /// range: `∂_a^4 K_0` itself is `~1e-17` at `log σ = 7`.
    ///
    /// Entry `j = 0` is `log_values[0]` verbatim, so a term list that only
    /// reaches `k = 0` evaluates bit-identically on either basis.
    pub log_scaled_a_derivatives: Option<Vec<KernelSignedLog>>,
    pub mode: IntegratedExpectationMode,
}

/// One signed magnitude in logarithmic coordinates.
///
/// `sign` is exactly `-1.0`, `0.0`, or `1.0`; `log_abs` is `-∞` when and only
/// when `sign` is zero.
#[derive(Clone, Copy, Debug)]
pub struct KernelSignedLog {
    pub log_abs: f64,
    pub sign: f64,
}

impl LogLognormalKernelBundle {
    #[inline]
    pub fn get(&self, k: usize) -> f64 {
        self.log_values[k]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.log_values.len()
    }

    /// `K_{k+2}/K_k − (K_{k+1}/K_k)²`, formed without the cancelling
    /// subtraction (#2610).
    ///
    /// The naive route evaluates both ratios and subtracts. In the large-σ
    /// regime they agree to `~1/(120σ²)` of their own size, so the difference
    /// keeps only the bits they do NOT share and the result is noise past
    /// `log σ ≈ 5.4` — no working precision repairs that, because the cancelled
    /// quantity keeps shrinking while the roundoff floor does not (#2566).
    ///
    /// Factoring the common ratio out first turns the subtraction into one
    /// `expm1`:
    ///
    /// ```text
    /// R₂ − R₁² = R₂ · (1 − e^Δ) = −R₂ · expm1(Δ),   Δ = 2L_{k+1} − L_{k+2} − L_k
    /// ```
    ///
    /// `expm1` is exact to full relative precision as `Δ → 0`, which is
    /// precisely where the difference form fails. That alone would only move
    /// the problem into `Δ`, a second difference of large log-values — except
    /// that the prefix's second difference is available in closed form:
    ///
    /// ```text
    /// prefix_{k+2} − 2·prefix_{k+1} + prefix_k = σ²   exactly, for every k and μ
    /// ```
    ///
    /// So `Δ = −(σ² + D²laplace)`: the dominant term is exact, and only the
    /// slowly-varying Laplace half is differenced numerically. For `m = 0` the
    /// Laplace half is identically zero and `Δ = −σ²` is exact outright.
    ///
    /// Returns `None` when the rung is missing or any input is non-finite —
    /// a caller that cannot form this must fall back rather than receive a
    /// silently degraded number.
    pub fn second_cumulant_ratio(&self, k: usize, sigma: f64) -> Option<f64> {
        if k + 2 >= self.log_values.len() || k + 2 >= self.log_laplace.len() {
            return None;
        }
        let log_k0 = self.log_values[k];
        let log_k2 = self.log_values[k + 2];
        if !log_k0.is_finite() || !log_k2.is_finite() {
            return None;
        }
        let (lap0, lap1, lap2) = (
            self.log_laplace[k],
            self.log_laplace[k + 1],
            self.log_laplace[k + 2],
        );
        if !(lap0.is_finite() && lap1.is_finite() && lap2.is_finite()) {
            return None;
        }
        let sigma2 = sigma * sigma;
        if !sigma2.is_finite() {
            return None;
        }
        // Differencing as (lap2 - lap1) - (lap1 - lap0) rather than
        // lap2 - 2*lap1 + lap0: the two first differences are each small, so
        // neither intermediate is a large quantity waiting to cancel.
        let second_difference_laplace = (lap2 - lap1) - (lap1 - lap0);
        let delta = -(sigma2 + second_difference_laplace);
        let ratio2 = (log_k2 - log_k0).exp();
        let value = -ratio2 * delta.exp_m1();
        if value.is_finite() { Some(value) } else { None }
    }
}

/// Builds a log-space kernel bundle for `k = 0, 1, …, max_k` at fixed
/// `(m, μ, σ)`.
pub fn log_kernel_bundle(
    quadctx: &QuadratureContext,
    m: f64,
    mu: f64,
    sigma: f64,
    max_k: usize,
) -> Result<LogLognormalKernelBundle, EstimationError> {
    validate_kernel_inputs(m, mu, sigma)?;
    let mut log_values = Vec::with_capacity(max_k + 1);
    let sigma2 = sigma * sigma;
    if !sigma2.is_finite() {
        crate::bail_invalid_estim!(
            "lognormal kernel sigma is outside the finite exact-derivative range: sigma={sigma}"
        );
    }
    let max_kf = max_k as f64;
    let prefix_bound = max_kf * mu.abs() + 0.5 * max_kf * max_kf * sigma2;
    if !prefix_bound.is_finite() {
        crate::bail_invalid_estim!(
            "lognormal kernel bundle prefix is outside the finite exact-derivative range: max_k={max_k}, mu={mu}, sigma={sigma}"
        );
    }
    if m == 0.0 {
        let mut prefix = 0.0;
        for k in 0..=max_k {
            log_values.push(prefix);
            prefix += mu + (k as f64 + 0.5) * sigma2;
        }
        // No Laplace factor on this branch: the kernel IS its prefix, so the
        // second difference of the Laplace half is exactly zero (#2610).
        let log_laplace = vec![0.0; max_k + 1];
        return Ok(LogLognormalKernelBundle {
            log_values,
            log_laplace,
            // `a = μ + ln m` does not exist at `m = 0`, and it is not needed:
            // this branch IS closed form, so no rung difference can cancel.
            log_scaled_a_derivatives: None,
            mode: IntegratedExpectationMode::ExactClosedForm,
        });
    }

    let log_m = m.ln();
    let shifted_bound = mu.abs() + max_kf * sigma2 + log_m.abs();
    if !shifted_bound.is_finite() {
        crate::bail_invalid_estim!(
            "lognormal kernel bundle shifted location is outside the finite exact-derivative range: max_k={max_k}, m={m}, mu={mu}, sigma={sigma}"
        );
    }
    let mut shifted_mu = mu + log_m;
    let mut prefix = 0.0;
    let mut mode = IntegratedExpectationMode::ExactClosedForm;
    let mut log_laplace_values = Vec::with_capacity(max_k + 1);
    for k in 0..=max_k {
        let (log_laplace, val_mode) =
            lognormal_laplace_unit_log_term_shared(quadctx, shifted_mu, sigma);
        log_values.push(if log_laplace.is_finite() {
            prefix + log_laplace
        } else {
            f64::NEG_INFINITY
        });
        // Kept unclamped on purpose: the summed entry above collapses a
        // non-finite Laplace term to −∞, which is the right reading for a
        // kernel VALUE but would erase the information a second difference
        // needs. `second_cumulant_ratio` refuses on non-finite instead (#2610).
        log_laplace_values.push(log_laplace);
        mode = worst_mode(mode, val_mode);
        prefix += mu + (k as f64 + 0.5) * sigma2;
        shifted_mu += sigma2;
    }
    // `log K_0` read out before the vector moves into the bundle; the loop above
    // always pushes at least the `k = 0` rung.
    let log_values_first = log_values[0];
    Ok(LogLognormalKernelBundle {
        log_values,
        log_laplace: log_laplace_values,
        log_scaled_a_derivatives: log_scaled_a_derivative_tower(
            quadctx,
            mu + log_m,
            sigma,
            max_k,
            log_values_first,
        ),
        mode,
    })
}

/// The `σ^j ∂_a^j K_0` tower for one bundle, truncated at the highest rung the
/// quadrature certifies, or `None` when it certifies none.
///
/// The tower and the value `ln S(a,σ)` come off the same log-space survival
/// panel by construction (#2714), so there is no longer any question of a
/// derivative and a value living on two approximation surfaces — that used to
/// be gated by "is the value routed through the Gumbel-mixing quadrature",
/// i.e. by `σ ≥ 8`. What is left to decide is only whether the direct tower is
/// better conditioned than the rung basis here, and the quadrature answers that
/// itself: [`LogSurvivalJet::certified_prefix_order`] admits each entry exactly
/// when its measured signed-sum cancellation says it is accurate.
///
/// **The length is the certified PREFIX, not `max_k`.** Refusing the whole
/// tower because its last rung cancelled made the basis a function of how many
/// rungs the CALLER asked for, and the callers differ: a value request, a
/// gradient, a Hessian and a contracted third derive `max_k` as
/// `base + 2·max_primary_increment + max_suffix_increment`, so they can reach
/// `4`, `5`, `6` and `7` on one row. Two of them evaluating the same term list
/// at the same point would then disagree about which basis to use, and their
/// answers would differ — which is the same fault as the one #2714 is filed on,
/// one level down.
///
/// This is not a partial mix: a term list that needs a rung past the truncation
/// falls back to the rung basis WHOLE (see
/// `latent_kernel_evaluate_terms_in_a_basis`, which refuses the list, not the
/// term). `None` also covers a non-finite `log_k0`.
fn log_scaled_a_derivative_tower(
    quadctx: &QuadratureContext,
    a: f64,
    sigma: f64,
    max_k: usize,
    log_k0: f64,
) -> Option<Vec<KernelSignedLog>> {
    if !log_k0.is_finite() {
        return None;
    }
    let jet = log_survival_jet(quadctx, a, sigma, max_k);
    let certified_order = jet.certified_prefix_order(max_k)?;
    let certified = jet.certified_scaled_mu_derivatives(certified_order)?;
    let mut tower = Vec::with_capacity(certified_order + 1);
    // Entry 0 is the kernel itself, taken verbatim from the value path so a
    // `k = 0` term list evaluates identically on either basis.
    tower.push(KernelSignedLog {
        log_abs: log_k0,
        sign: 1.0,
    });
    for entry in &certified[1..] {
        tower.push(KernelSignedLog {
            log_abs: entry.log_abs,
            sign: entry.sign,
        });
    }
    Some(tower)
}

/// Computes the value-space derivative ratios `∂ⁿ_μ K_{k,m} / K_{k,m}`
/// from a log-space bundle.
///
/// Returns `[1, K'/K, K''/K, K'''/K, K''''/K]` where only the first
/// `order + 1` entries are valid.
///
/// The recurrences are applied in ratio form, with each `K_{k+r}/K_k`
/// computed as `exp(log K_{k+r} − log K_k)`, which remains finite even when
/// the individual kernel values would overflow or underflow.
pub fn kernel_ratio_jet(
    log_bundle: &LogLognormalKernelBundle,
    k: usize,
    m: f64,
    order: usize,
) -> [f64; 5] {
    let kf = k as f64;
    let log_k0 = log_bundle.get(k);

    // Precompute ratios K_{k+r}/K_k for r = 1..=order, each from a single
    // log-difference.  This avoids redundant exp() calls when the same ratio
    // appears in multiple derivative orders.
    let mut rk = [0.0f64; 5]; // rk[0] unused; rk[r] = K_{k+r}/K_k
    for r in 1..=order.min(4) {
        let delta = log_bundle.get(k + r) - log_k0;
        rk[r] = if delta.is_finite() {
            delta.exp()
        } else if delta > 0.0 {
            f64::INFINITY
        } else {
            0.0
        };
    }

    let mut jet = [0.0; 5];
    jet[0] = 1.0;

    if order >= 1 {
        jet[1] = kf - m * rk[1];
    }
    if order >= 2 {
        jet[2] = kf * kf - (2.0 * kf + 1.0) * m * rk[1] + m * m * rk[2];
    }
    if order >= 3 {
        jet[3] = kf * kf * kf - (3.0 * kf * kf + 3.0 * kf + 1.0) * m * rk[1]
            + 3.0 * (kf + 1.0) * m * m * rk[2]
            - m * m * m * rk[3];
    }
    if order >= 4 {
        let k2 = kf * kf;
        let k3 = k2 * kf;
        let k4 = k3 * kf;
        let m2 = m * m;
        let m3 = m2 * m;
        let m4 = m3 * m;
        jet[4] = k4 - (4.0 * k3 + 6.0 * k2 + 4.0 * kf + 1.0) * m * rk[1]
            + (6.0 * k2 + 12.0 * kf + 7.0) * m2 * rk[2]
            - (4.0 * kf + 6.0) * m3 * rk[3]
            + m4 * rk[4];
    }

    jet
}

// `LatentCLogLogJet5` + `latent_cloglog_jet5` / `latent_cloglog_inverse_link_jet`
// moved DOWN to `crate::quadrature` (#1135), co-located with their analytic
// backend, so the `solver` link layer names them without importing up into
// `families::survival`. Re-exported here so the in-family callers (e.g.
// `family_runtime`) keep resolving.
pub use crate::quadrature::{
    LatentCLogLogJet5, latent_cloglog_inverse_link_jet, latent_cloglog_jet5,
};

// ─── LogKernelSumJet: log-sum derivatives from log-space bundles ─────────────

/// A single signed term in a kernel sum: coefficient × K_{k,m}.
#[derive(Clone, Copy, Debug)]
pub struct KernelSumTerm {
    /// Multiplicative coefficient (can be negative for difference terms).
    pub coeff: f64,
    /// Kernel order parameter k.
    pub k: usize,
    /// Kernel mass parameter m (≥ 0).
    pub m: f64,
}

/// Log-mass separation below which a same-rung pair is differenced
/// ANALYTICALLY rather than numerically (see
/// [`LogKernelSumJet::analytic_log_mag_gap`]).
///
/// The two routes have opposite error behaviour in `dv`: the direct difference
/// carries relative error `ε/|dv|`, the expansion `~dv³/24` from its first
/// dropped order. They cross at `dv⁴ ≈ 24ε`, i.e. `dv ≈ 8e-4`. `1e-4` sits
/// safely on the expansion's side of that crossing (`4e-14` expansion error
/// against `2e-12` for the difference) and leaves every pair a runtime row
/// actually forms — interval widths of order the observation scale — on the
/// unchanged numerical path.
const ANALYTIC_LOG_MASS_GAP_THRESHOLD: f64 = 1e-4;

/// Derivatives of `log(Σ_j a_j · K_{k_j, m_j}(μ, σ))` with respect to μ.
///
/// This is the workhorse for row-level log-likelihood derivatives in all
/// latent families.  The numerator and denominator of a row likelihood are
/// each a small signed sum of kernel terms.
///
/// The value path is assembled from log-space kernel bundles and ratio jets,
/// so individual kernel terms are never exponentiated before the final signed
/// sum. That avoids the old overflow/underflow problems from value-space
/// kernels. When the signed sum is zero or negative, this returns an invalid
/// row (`value = -∞`) instead of trying to continue with a floored surrogate.
/// Signed two-term differences (e.g. interval censoring `K_{0,M_L} − K_{0,M_R}`)
/// are still combined through the shared sign-aware log-sum path.
#[derive(Clone, Copy, Debug)]
pub struct LogKernelSumJet {
    /// log(Σ a_j K_j)
    pub value: f64,
    /// d/dμ log(Σ a_j K_j)
    pub d1: f64,
    /// d²/dμ² log(Σ a_j K_j)
    pub d2: f64,
    /// d³/dμ³ log(Σ a_j K_j)
    pub d3: f64,
    /// d⁴/dμ⁴ log(Σ a_j K_j)
    pub d4: f64,
    pub mode: IntegratedExpectationMode,
}

impl LogKernelSumJet {
    #[inline]
    fn non_positive(mode: IntegratedExpectationMode) -> Self {
        Self {
            value: f64::NEG_INFINITY,
            d1: 0.0,
            d2: 0.0,
            d3: 0.0,
            d4: 0.0,
            mode,
        }
    }

    #[inline]
    fn from_log_value_and_ratios(
        value: f64,
        ratio: [f64; 5],
        mode: IntegratedExpectationMode,
    ) -> Self {
        let r1 = ratio[1];
        let r2 = ratio[2];
        let r3 = ratio[3];
        let r4 = ratio[4];
        Self {
            value,
            d1: r1,
            d2: r2 - r1 * r1,
            d3: r3 - 3.0 * r1 * r2 + 2.0 * r1 * r1 * r1,
            d4: r4 - 4.0 * r1 * r3 - 3.0 * r2 * r2 + 12.0 * r1 * r1 * r2 - 6.0 * r1.powi(4),
            mode,
        }
    }

    #[inline]
    fn term_log_mag_and_ratio(
        bundle: &LogLognormalKernelBundle,
        term: KernelSumTerm,
    ) -> (f64, [f64; 5]) {
        (
            term.coeff.abs().ln() + bundle.get(term.k),
            // d4 is used by the exact log-sigma curvature, so this must carry
            // ratios through order 4 rather than truncating at order 3.
            kernel_ratio_jet(bundle, term.k, term.m, 4),
        )
    }

    /// `log|a₁K₁| − log|a₀K₀|` for a same-rung pair, without differencing two
    /// `O(1)` logs (#2277).
    ///
    /// `log K_{k,m} = kμ + σ²k²/2 + Λ(v)` with `v = μ + kσ² + ln m`, so a pair
    /// sharing `k` differs only through `dv = ln(m₁/m₀)` and the analytic
    /// prefix. Subtracting the two assembled logs costs an ABSOLUTE `ε ≈
    /// 2.2e-16`, hence a RELATIVE error `ε/|dv|` in the separation — `2e-4` at
    /// `dv = 1e-12`. That is the whole of the #2277 narrow-interval failure:
    /// the interval value is `log|a₀K₀| + log1mexp(δ) ≈ log|a₀K₀| + ln|δ|`, so
    /// a relative error in `δ` lands as an absolute error in the value.
    ///
    /// Expanding `Λ` instead makes the leading order analytic:
    ///
    /// ```text
    /// Λ(v₀ + dv) − Λ(v₀) = Λ'·dv + Λ''·dv²/2 + Λ'''·dv³/6 + O(dv⁴)
    /// ```
    ///
    /// `Λ', Λ'', Λ'''` are exactly the μ-log-derivatives the ratio jet already
    /// carries, because `∂_μ` and `∂_{ln m}` act identically on `Λ`; only the
    /// `kμ` prefix distinguishes them, which is the `− kf` on the first order.
    /// `dv` is formed as `ln1p((m₁ − m₀)/m₀)`: for nearby masses `m₁ − m₀` is
    /// exact by Sterbenz and `ln1p` is accurate at small argument, so `dv`
    /// keeps full RELATIVE precision where `ln m₁ − ln m₀` would not.
    ///
    /// Returns `None` unless the pair shares its rung, both masses are
    /// strictly positive, and `|dv|` is inside
    /// [`ANALYTIC_LOG_MASS_GAP_THRESHOLD`]; outside that range the direct
    /// difference is both valid and more accurate and the caller must use it.
    ///
    /// LIMIT: the coefficient half, `ln|a₁| − ln|a₀|`, is still a numerical
    /// difference. Interval censoring passes `a = ±exp(−unloaded mass)`, so a
    /// narrow interval whose UNLOADED masses also nearly coincide reintroduces
    /// the same amplification through that half; removing it needs the caller
    /// to pass the unloaded masses rather than their exponentials.
    fn analytic_log_mag_gap(
        t0: KernelSumTerm,
        t1: KernelSumTerm,
        ratio0: &[f64; 5],
    ) -> Option<f64> {
        if t0.k != t1.k || t0.m <= 0.0 || t1.m <= 0.0 {
            return None;
        }
        let dv = ((t1.m - t0.m) / t0.m).ln_1p();
        if !dv.is_finite() || dv.abs() > ANALYTIC_LOG_MASS_GAP_THRESHOLD {
            return None;
        }
        let kf = t0.k as f64;
        let (r1, r2, r3) = (ratio0[1], ratio0[2], ratio0[3]);
        let lambda1 = r1 - kf;
        let lambda2 = r2 - r1 * r1;
        let lambda3 = r3 - 3.0 * r1 * r2 + 2.0 * r1 * r1 * r1;
        let kernel_gap = dv * (lambda1 + dv * (0.5 * lambda2 + dv * (lambda3 / 6.0)));
        let coeff_gap = t1.coeff.abs().ln() - t0.coeff.abs().ln();
        let delta = coeff_gap + kernel_gap;
        if delta.is_finite() { Some(delta) } else { None }
    }

    /// Reduces `sign₀·e^{L₀} + sign₁·e^{L₀+δ}` to `(log|u|, sign(u))` for
    /// `u = sign₀ + sign₁·e^δ`, so the pair's magnitude is `L₀ + log|u|`.
    ///
    /// The opposite-sign branch is where cancellation lives and it is exactly
    /// `log|1 − e^δ|`, i.e. Måchler's `log1mexp` on `|δ|`, which is accurate
    /// across the whole range of `δ` PROVIDED `δ` is itself accurate. That
    /// proviso is why [`Self::analytic_log_mag_gap`] exists.
    ///
    /// Returns `None` when either sign is zero (nothing cancels; the general
    /// path already drops a zero term correctly) or when the sum vanishes.
    fn reduce_signed_pair(sign0: f64, sign1: f64, delta: f64) -> Option<(f64, f64)> {
        if sign0 == 0.0 || sign1 == 0.0 || !delta.is_finite() {
            return None;
        }
        if sign0 * sign1 > 0.0 {
            // No cancellation; factor out whichever exponential is larger.
            let log_u = if delta > 0.0 {
                delta + (-delta).exp().ln_1p()
            } else {
                delta.exp().ln_1p()
            };
            Some((log_u, sign0.signum()))
        } else if delta == 0.0 {
            // Exact cancellation: the signed sum is zero, not a small positive.
            None
        } else {
            // |1 − e^δ| = e^{max(δ,0)}·(1 − e^{−|δ|}).
            let log_u = delta.max(0.0) + log1mexp_positive(delta.abs());
            Some((log_u, sign0.signum() * -delta.signum()))
        }
    }

    fn evaluate_two_terms(
        quadctx: &QuadratureContext,
        t0: KernelSumTerm,
        t1: KernelSumTerm,
        mu: f64,
        sigma: f64,
    ) -> Result<Self, EstimationError> {
        let max_k_needed = t0.k.max(t1.k) + 4;
        let bundle0 = log_kernel_bundle(quadctx, t0.m, mu, sigma, max_k_needed)?;
        let mut overall_mode = bundle0.mode;
        let bundle1_owned = if t0.m == t1.m {
            None
        } else {
            let bundle1 = log_kernel_bundle(quadctx, t1.m, mu, sigma, max_k_needed)?;
            overall_mode = worst_mode(overall_mode, bundle1.mode);
            Some(bundle1)
        };
        let bundle1 = bundle1_owned.as_ref().unwrap_or(&bundle0);

        let (log_mag0, ratio0) = Self::term_log_mag_and_ratio(&bundle0, t0);
        let (log_mag1, ratio1) = Self::term_log_mag_and_ratio(bundle1, t1);
        let log_mags = [log_mag0, log_mag1];
        let signs = [t0.coeff.signum(), t1.coeff.signum()];

        // Narrow same-rung pairs go through the analytic separation; everything
        // else keeps the general signed reduction, including the infinite
        // log-magnitudes only `signed_log_sum_exp` resolves.
        let analytic = if log_mag0.is_finite() && log_mag1.is_finite() {
            Self::analytic_log_mag_gap(t0, t1, &ratio0).and_then(|delta| {
                Self::reduce_signed_pair(signs[0], signs[1], delta)
                    .map(|(log_u, sign_u)| (log_u, sign_u, delta))
            })
        } else {
            None
        };
        let (log_s, sign_s, log_w0, log_w1) = match analytic {
            Some((log_u, sign_u, delta)) => (log_mag0 + log_u, sign_u, -log_u, delta - log_u),
            None => {
                let (log_s, sign_s) = signed_log_sum_exp(&log_mags, &signs);
                (log_s, sign_s, log_mag0 - log_s, log_mag1 - log_s)
            }
        };
        if !log_s.is_finite() || sign_s <= 0.0 {
            return Ok(Self::non_positive(overall_mode));
        }

        let w0 = sign_s * signs[0] * log_w0.exp();
        let w1 = sign_s * signs[1] * log_w1.exp();
        let wr1 = w0 * ratio0[1] + w1 * ratio1[1];
        let wr2 = w0 * ratio0[2] + w1 * ratio1[2];
        let wr3 = w0 * ratio0[3] + w1 * ratio1[3];
        let wr4 = w0 * ratio0[4] + w1 * ratio1[4];

        Ok(Self {
            value: log_s,
            d1: wr1,
            d2: wr2 - wr1 * wr1,
            d3: wr3 - 3.0 * wr1 * wr2 + 2.0 * wr1 * wr1 * wr1,
            d4: wr4 - 4.0 * wr1 * wr3 - 3.0 * wr2 * wr2 + 12.0 * wr1 * wr1 * wr2
                - 6.0 * wr1.powi(4),
            mode: overall_mode,
        })
    }

    /// Evaluate for a single positive kernel term (fast path).
    ///
    /// Computes `log(K_{k,m})` and its μ-derivatives from exact recurrences,
    /// entirely in log-space.
    pub fn single_term(
        quadctx: &QuadratureContext,
        k: usize,
        m: f64,
        mu: f64,
        sigma: f64,
    ) -> Result<Self, EstimationError> {
        let max_k_needed = k + 4;
        let lb = log_kernel_bundle(quadctx, m, mu, sigma, max_k_needed)?;
        Ok(Self::from_log_value_and_ratios(
            lb.get(k),
            kernel_ratio_jet(&lb, k, m, 4),
            lb.mode,
        ))
    }

    /// Evaluate `log(Σ a_j K_j)` and its μ-derivatives for a small signed sum.
    ///
    /// All terms share the same `(μ, σ)`.  Both the value and derivative
    /// ratios are computed entirely in log-space.  The runtime latent-survival
    /// rows in this repo are almost always one-term or two-term sums, so those
    /// cases stay on dedicated stack paths; the heap-backed logic below is only
    /// for genuinely longer symbolic sums:
    ///
    /// 1. Per-term log-magnitudes `log|a_j| + log K_{k_j,m_j}` and signs.
    /// 2. Sign-aware log-sum-exp to get `log|S|` and `sign(S)`.
    /// 3. Importance weights `w_j = a_j K_j / S` formed in log-space.
    /// 4. Weighted ratio sums `R_n = Σ w_j · (∂ⁿK_j / K_j)` for the
    ///    final log-derivatives.
    pub fn evaluate(
        quadctx: &QuadratureContext,
        terms: &[KernelSumTerm],
        mu: f64,
        sigma: f64,
    ) -> Result<Self, EstimationError> {
        if terms.is_empty() {
            // Empty sums are a caller-contract violation, not a degenerate row.
            // Return an input error so callers can report the malformed kernel sum.
            crate::bail_invalid_estim!("KernelSumJet requires at least one term");
        }

        // Fast path for single term.
        if terms.len() == 1 {
            let t = &terms[0];
            if t.coeff <= 0.0 {
                // Negative or zero coefficient: the sum is non-positive, so
                // log(sum) is undefined.  Return −∞ (impossible observation),
                // matching the general path's sign_s ≤ 0 branch.
                return Ok(Self::non_positive(
                    IntegratedExpectationMode::ExactClosedForm,
                ));
            }
            let jet = Self::single_term(quadctx, t.k, t.m, mu, sigma)?;
            return Ok(Self {
                value: t.coeff.ln() + jet.value,
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
                d4: jet.d4,
                mode: jet.mode,
            });
        }
        if terms.len() == 2 {
            return Self::evaluate_two_terms(quadctx, terms[0], terms[1], mu, sigma);
        }

        let max_k_needed = terms.iter().map(|t| t.k).max().unwrap_or(0) + 4;

        // Build log-bundles for each unique mass.
        let mut log_bundles: Vec<(f64, LogLognormalKernelBundle)> = Vec::with_capacity(2);
        let mut overall_mode = IntegratedExpectationMode::ExactClosedForm;
        for term in terms {
            if !log_bundles.iter().any(|(m, _)| *m == term.m) {
                let b = log_kernel_bundle(quadctx, term.m, mu, sigma, max_k_needed)?;
                overall_mode = worst_mode(overall_mode, b.mode);
                log_bundles.push((term.m, b));
            }
        }

        let get_lb = |m: f64| -> &LogLognormalKernelBundle {
            &log_bundles
                .iter()
                .find(|(bm, _)| *bm == m)
                .expect("the loop above pushes a bundle for every distinct term.m before any lookup")
                .1
        };

        // Per-term: log magnitude, sign, and ratio jet.
        let mut log_mags: Vec<f64> = Vec::with_capacity(terms.len());
        let mut signs: Vec<f64> = Vec::with_capacity(terms.len());
        let mut ratios: Vec<[f64; 5]> = Vec::with_capacity(terms.len());
        for term in terms {
            let lb = get_lb(term.m);
            log_mags.push(term.coeff.abs().ln() + lb.get(term.k));
            signs.push(term.coeff.signum());
            ratios.push(kernel_ratio_jet(lb, term.k, term.m, 4));
        }

        // Sign-aware log-sum-exp: compute log|S| and sign(S).
        let (log_s, sign_s) = signed_log_sum_exp(&log_mags, &signs);

        if !log_s.is_finite() || sign_s <= 0.0 {
            // Sum is zero or negative — degenerate row.
            return Ok(Self::non_positive(overall_mode));
        }

        // Importance weights w_j = sign(S) · sign(a_j) · exp(log|a_j K_j| − log|S|).
        // When S > 0 and all terms have well-defined kernels, Σ w_j = 1.
        let mut wr1 = 0.0;
        let mut wr2 = 0.0;
        let mut wr3 = 0.0;
        let mut wr4 = 0.0;
        for i in 0..terms.len() {
            let w = sign_s * signs[i] * (log_mags[i] - log_s).exp();
            wr1 += w * ratios[i][1];
            wr2 += w * ratios[i][2];
            wr3 += w * ratios[i][3];
            wr4 += w * ratios[i][4];
        }

        Ok(Self {
            value: log_s,
            d1: wr1,
            d2: wr2 - wr1 * wr1,
            d3: wr3 - 3.0 * wr1 * wr2 + 2.0 * wr1 * wr1 * wr1,
            d4: wr4 - 4.0 * wr1 * wr3 - 3.0 * wr2 * wr2 + 12.0 * wr1 * wr1 * wr2
                - 6.0 * wr1.powi(4),
            mode: overall_mode,
        })
    }
}

// ─── Latent survival sufficient statistics ───────────────────────────────────

/// Event type for compiled survival sufficient statistics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LatentSurvivalEventType {
    /// Right-censored: observed alive in the observation window.
    RightCensored,
    /// Exact event: event observed at a known time.
    ExactEvent,
    /// Interval-censored: event known to occur in (t_left, t_right].
    IntervalCensored,
}

impl fmt::Display for LatentSurvivalEventType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RightCensored => write!(f, "right_censored"),
            Self::ExactEvent => write!(f, "exact_event"),
            Self::IntervalCensored => write!(f, "interval_censored"),
        }
    }
}

/// Row-level sufficient statistics for one latent survival observation.
///
/// This is the canonical row representation used by both fitted-family
/// evaluation and saved-model prediction.
///
/// For the full-loading model (frailty multiplies entire hazard):
///   mass_loaded = total cumulative hazard mass
///   mass_unloaded = 0
///
/// For the loaded-vs-unloaded model (Gompertz-Makeham):
///   mass_loaded = integrated disease hazard component
///   mass_unloaded = integrated background hazard component (not frailty-modified)
///
/// The unloaded mass contributes a simple exp(-M_U) prefactor.
#[derive(Clone, Copy, Debug)]
pub struct LatentSurvivalRow {
    pub event_type: LatentSurvivalEventType,
    /// Cumulative nuisance mass at entry: B(a_in).
    /// Zero if there is no left truncation.
    pub mass_entry: f64,
    /// Cumulative nuisance mass at exit/event: B(a_out) or B(a_event).
    pub mass_exit: f64,
    /// For interval censoring: mass at left boundary B(a_L).
    pub mass_left: f64,
    /// For interval censoring: mass at right boundary B(a_R).
    pub mass_right: f64,
    /// For interval censoring: unloaded mass at left boundary.
    pub mass_unloaded_left: f64,
    /// For interval censoring: unloaded mass at right boundary.
    pub mass_unloaded_right: f64,
    /// Unloaded (background) cumulative mass at entry (0 for full loading).
    pub mass_unloaded_entry: f64,
    /// Unloaded (background) cumulative mass at exit.
    pub mass_unloaded_exit: f64,
    /// Loaded instantaneous hazard at event time (for exact events).
    pub hazard_loaded: f64,
    /// Unloaded instantaneous hazard at event time (for exact events).
    pub hazard_unloaded: f64,
}

impl LatentSurvivalRow {
    /// Delayed-entry right-censored row with explicit loaded/unloaded masses.
    ///
    /// `mass_entry` and `mass_exit` are cumulative loaded masses `B_L(a_in)`
    /// and `B_L(a_out)` for this row object. They are not an increment over
    /// `(a_in, a_out]`.
    pub fn right_censored(
        mass_entry: f64,
        mass_exit: f64,
        mass_unloaded_entry: f64,
        mass_unloaded_exit: f64,
    ) -> Self {
        Self {
            event_type: LatentSurvivalEventType::RightCensored,
            mass_entry,
            mass_exit,
            mass_left: 0.0,
            mass_right: 0.0,
            mass_unloaded_left: 0.0,
            mass_unloaded_right: 0.0,
            mass_unloaded_entry,
            mass_unloaded_exit,
            hazard_loaded: 0.0,
            hazard_unloaded: 0.0,
        }
    }

    /// Delayed-entry exact-event row with explicit loaded/unloaded hazard parts.
    pub fn exact_event(
        mass_entry: f64,
        mass_exit: f64,
        mass_unloaded_entry: f64,
        mass_unloaded_exit: f64,
        hazard_loaded: f64,
        hazard_unloaded: f64,
    ) -> Self {
        Self {
            event_type: LatentSurvivalEventType::ExactEvent,
            mass_entry,
            mass_exit,
            mass_left: 0.0,
            mass_right: 0.0,
            mass_unloaded_left: 0.0,
            mass_unloaded_right: 0.0,
            mass_unloaded_entry,
            mass_unloaded_exit,
            hazard_loaded,
            hazard_unloaded,
        }
    }

    /// Delayed-entry interval-censored row with explicit loaded/unloaded masses.
    pub fn interval_censored(
        mass_entry: f64,
        mass_left: f64,
        mass_right: f64,
        mass_unloaded_entry: f64,
        mass_unloaded_left: f64,
        mass_unloaded_right: f64,
    ) -> Self {
        Self {
            event_type: LatentSurvivalEventType::IntervalCensored,
            mass_entry,
            mass_exit: 0.0,
            mass_left,
            mass_right,
            mass_unloaded_left,
            mass_unloaded_right,
            mass_unloaded_entry,
            mass_unloaded_exit: 0.0,
            hazard_loaded: 0.0,
            hazard_unloaded: 0.0,
        }
    }

    pub fn validate(&self) -> Result<(), EstimationError> {
        let fields = [
            ("mass_entry", self.mass_entry),
            ("mass_exit", self.mass_exit),
            ("mass_left", self.mass_left),
            ("mass_right", self.mass_right),
            ("mass_unloaded_left", self.mass_unloaded_left),
            ("mass_unloaded_right", self.mass_unloaded_right),
            ("mass_unloaded_entry", self.mass_unloaded_entry),
            ("mass_unloaded_exit", self.mass_unloaded_exit),
            ("hazard_loaded", self.hazard_loaded),
            ("hazard_unloaded", self.hazard_unloaded),
        ];
        for (name, value) in fields {
            if !value.is_finite() || value < 0.0 {
                crate::bail_invalid_estim!(
                    "latent survival row has invalid {name}={value}; expected a finite non-negative value"
                );
            }
        }

        match self.event_type {
            LatentSurvivalEventType::RightCensored => {
                if self.mass_exit < self.mass_entry {
                    crate::bail_invalid_estim!(
                        "latent survival right-censored row requires mass_exit >= mass_entry, got {} < {}",
                        self.mass_exit,
                        self.mass_entry
                    );
                }
                if self.mass_unloaded_exit < self.mass_unloaded_entry {
                    crate::bail_invalid_estim!(
                        "latent survival right-censored row requires unloaded exit mass >= unloaded entry mass, got {} < {}",
                        self.mass_unloaded_exit,
                        self.mass_unloaded_entry
                    );
                }
                if self.mass_left > 0.0
                    || self.mass_right > 0.0
                    || self.mass_unloaded_left > 0.0
                    || self.mass_unloaded_right > 0.0
                    || self.hazard_loaded > 0.0
                    || self.hazard_unloaded > 0.0
                {
                    crate::bail_invalid_estim!("latent survival right-censored row cannot carry interval masses or event hazards"
                            .to_string(),);
                }
            }
            LatentSurvivalEventType::ExactEvent => {
                if self.mass_exit < self.mass_entry {
                    crate::bail_invalid_estim!(
                        "latent survival exact-event row requires mass_exit >= mass_entry, got {} < {}",
                        self.mass_exit,
                        self.mass_entry
                    );
                }
                if self.mass_unloaded_exit < self.mass_unloaded_entry {
                    crate::bail_invalid_estim!(
                        "latent survival exact-event row requires unloaded exit mass >= unloaded entry mass, got {} < {}",
                        self.mass_unloaded_exit,
                        self.mass_unloaded_entry
                    );
                }
                if self.mass_left > 0.0
                    || self.mass_right > 0.0
                    || self.mass_unloaded_left > 0.0
                    || self.mass_unloaded_right > 0.0
                {
                    crate::bail_invalid_estim!(
                        "latent survival exact-event row cannot carry interval masses"
                    );
                }
                if self.hazard_loaded == 0.0 && self.hazard_unloaded == 0.0 {
                    crate::bail_invalid_estim!("latent survival exact-event row requires a positive loaded or unloaded hazard"
                            .to_string(),);
                }
            }
            LatentSurvivalEventType::IntervalCensored => {
                if self.mass_left < self.mass_entry || self.mass_right < self.mass_left {
                    crate::bail_invalid_estim!(
                        "latent survival interval row requires mass_entry <= mass_left <= mass_right, got entry={}, left={}, right={}",
                        self.mass_entry,
                        self.mass_left,
                        self.mass_right
                    );
                }
                if self.mass_unloaded_left < self.mass_unloaded_entry
                    || self.mass_unloaded_right < self.mass_unloaded_left
                {
                    crate::bail_invalid_estim!(
                        "latent survival interval row requires unloaded_entry <= unloaded_left <= unloaded_right, got entry={}, left={}, right={}",
                        self.mass_unloaded_entry,
                        self.mass_unloaded_left,
                        self.mass_unloaded_right
                    );
                }
                if self.mass_exit > 0.0
                    || self.mass_unloaded_exit > 0.0
                    || self.hazard_loaded > 0.0
                    || self.hazard_unloaded > 0.0
                {
                    crate::bail_invalid_estim!(
                        "latent survival interval row cannot carry exit masses or event hazards"
                            .to_string(),
                    );
                }
            }
        }

        Ok(())
    }
}

fn exact_event_kernel_jet(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    mu: f64,
    sigma: f64,
) -> Result<LogKernelSumJet, EstimationError> {
    if row.hazard_loaded < 0.0 || row.hazard_unloaded < 0.0 {
        crate::bail_invalid_estim!(
            "latent survival exact-event hazards must be non-negative, got loaded={} unloaded={}",
            row.hazard_loaded,
            row.hazard_unloaded
        );
    }
    match (row.hazard_unloaded > 0.0, row.hazard_loaded > 0.0) {
        (true, true) => {
            let terms = [
                KernelSumTerm {
                    coeff: row.hazard_unloaded,
                    k: 0,
                    m: row.mass_exit,
                },
                KernelSumTerm {
                    coeff: row.hazard_loaded,
                    k: 1,
                    m: row.mass_exit,
                },
            ];
            LogKernelSumJet::evaluate(quadctx, &terms, mu, sigma)
        }
        (true, false) => {
            let jet = LogKernelSumJet::single_term(quadctx, 0, row.mass_exit, mu, sigma)?;
            Ok(LogKernelSumJet {
                value: row.hazard_unloaded.ln() + jet.value,
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
                d4: jet.d4,
                mode: jet.mode,
            })
        }
        (false, true) => {
            let jet = LogKernelSumJet::single_term(quadctx, 1, row.mass_exit, mu, sigma)?;
            Ok(LogKernelSumJet {
                value: row.hazard_loaded.ln() + jet.value,
                d1: jet.d1,
                d2: jet.d2,
                d3: jet.d3,
                d4: jet.d4,
                mode: jet.mode,
            })
        }
        (false, false) => Err(EstimationError::InvalidInput(
            "latent survival exact-event row requires a positive loaded or unloaded hazard"
                .to_string(),
        )),
    }
}

/// Row-level log-likelihood and μ-derivatives for the latent survival model.
///
/// The conditional model is:
///   `Λ(a | U) = B(a) · exp(U)`,  `U ~ N(μ, σ²)`
///
/// All likelihoods reduce to algebra on `K_{k,m}(μ, σ)`.
#[derive(Clone, Copy, Debug)]
pub struct LatentSurvivalRowJet {
    pub log_lik: f64,
    pub score: f64,
    pub neg_hessian: f64,
    pub d3: f64,
    pub score_log_sigma: f64,
    pub neg_hessian_log_sigma: f64,
}

#[inline]
fn log_sigma_score_from_log_sum(jet: &LogKernelSumJet, sigma: f64) -> f64 {
    let sigma2 = sigma * sigma;
    sigma2 * (jet.d2 + jet.d1 * jet.d1)
}

#[inline]
fn log_sigma_neg_hessian_from_log_sum(jet: &LogKernelSumJet, sigma: f64) -> f64 {
    let sigma2 = sigma * sigma;
    let sigma4 = sigma2 * sigma2;
    let d1 = jet.d1;
    let d2 = jet.d2;
    let d3 = jet.d3;
    let d4 = jet.d4;
    let s2_over_s = d2 + d1 * d1;
    // For S = Σ a_j K_j, D = σ ∂_σ, and D S = σ² S_μμ:
    // D² log S = 2σ² (S''/S) + σ⁴ (S''''/S - (S''/S)²).
    // Express the final parenthesized term directly in log-derivatives to
    // avoid the larger cancellation in `r4 - r2²`.
    let s4_over_s_minus_s2_sq = d4 + 4.0 * d1 * d3 + 2.0 * d2 * d2 + 4.0 * d1 * d1 * d2;
    -(2.0 * sigma2 * s2_over_s + sigma4 * s4_over_s_minus_s2_sq)
}

impl LatentSurvivalRowJet {
    pub fn evaluate(
        quadctx: &QuadratureContext,
        row: &LatentSurvivalRow,
        mu: f64,
        sigma: f64,
    ) -> Result<Self, EstimationError> {
        row.validate()?;
        match row.event_type {
            LatentSurvivalEventType::RightCensored => Self::right_censored(quadctx, mu, sigma, row),
            LatentSurvivalEventType::ExactEvent => Self::exact_event(quadctx, mu, sigma, row),
            LatentSurvivalEventType::IntervalCensored => {
                Self::interval_censored(quadctx, mu, sigma, row)
            }
        }
    }

    /// Right-censoring with loaded/unloaded mass decomposition.
    ///
    /// Full formula:
    ///   `ℓ = -M_U_exit + log K_{0,M_L_exit} + M_U_entry - log K_{0,M_L_entry}`
    ///
    /// When `mass_unloaded_exit == 0` and `mass_unloaded_entry == 0`, this
    /// falls back to the original formula using `mass_exit` / `mass_entry`.
    fn right_censored(
        quadctx: &QuadratureContext,
        mu: f64,
        sigma: f64,
        row: &LatentSurvivalRow,
    ) -> Result<Self, EstimationError> {
        let has_unloaded = row.mass_unloaded_exit != 0.0 || row.mass_unloaded_entry != 0.0;

        // Loaded mass for the kernel terms: when unloaded mass is present,
        // mass_exit contains only the loaded component; otherwise it is the
        // total mass.
        let mass_exit_loaded = row.mass_exit;
        let mass_entry_loaded = row.mass_entry;

        // Unloaded mass contributes a simple additive constant to log-lik
        let unloaded_offset = if has_unloaded {
            -row.mass_unloaded_exit + row.mass_unloaded_entry
        } else {
            0.0
        };

        let num = LogKernelSumJet::single_term(quadctx, 0, mass_exit_loaded, mu, sigma)?;
        if mass_entry_loaded > 0.0 {
            let den = LogKernelSumJet::single_term(quadctx, 0, mass_entry_loaded, mu, sigma)?;
            Ok(Self {
                log_lik: unloaded_offset + num.value - den.value,
                score: num.d1 - den.d1,
                neg_hessian: -(num.d2 - den.d2),
                d3: num.d3 - den.d3,
                score_log_sigma: log_sigma_score_from_log_sum(&num, sigma)
                    - log_sigma_score_from_log_sum(&den, sigma),
                neg_hessian_log_sigma: log_sigma_neg_hessian_from_log_sum(&num, sigma)
                    - log_sigma_neg_hessian_from_log_sum(&den, sigma),
            })
        } else {
            Ok(Self {
                log_lik: unloaded_offset + num.value,
                score: num.d1,
                neg_hessian: -num.d2,
                d3: num.d3,
                score_log_sigma: log_sigma_score_from_log_sum(&num, sigma),
                neg_hessian_log_sigma: log_sigma_neg_hessian_from_log_sum(&num, sigma),
            })
        }
    }

    /// Exact event with loaded/unloaded hazard decomposition.
    ///
    /// `ℓ = log(h_U · K_{0,M_L} + h_L · K_{1,M_L}) - M_U_event + M_U_entry - log K_{0,M_L_entry}`
    fn exact_event(
        quadctx: &QuadratureContext,
        mu: f64,
        sigma: f64,
        row: &LatentSurvivalRow,
    ) -> Result<Self, EstimationError> {
        let unloaded_offset = if row.mass_unloaded_exit != 0.0 || row.mass_unloaded_entry != 0.0 {
            -row.mass_unloaded_exit + row.mass_unloaded_entry
        } else {
            0.0
        };
        let num = exact_event_kernel_jet(quadctx, row, mu, sigma)?;

        if row.mass_entry > 0.0 {
            let den = LogKernelSumJet::single_term(quadctx, 0, row.mass_entry, mu, sigma)?;
            Ok(Self {
                log_lik: unloaded_offset + num.value - den.value,
                score: num.d1 - den.d1,
                neg_hessian: -(num.d2 - den.d2),
                d3: num.d3 - den.d3,
                score_log_sigma: log_sigma_score_from_log_sum(&num, sigma)
                    - log_sigma_score_from_log_sum(&den, sigma),
                neg_hessian_log_sigma: log_sigma_neg_hessian_from_log_sum(&num, sigma)
                    - log_sigma_neg_hessian_from_log_sum(&den, sigma),
            })
        } else {
            Ok(Self {
                log_lik: unloaded_offset + num.value,
                score: num.d1,
                neg_hessian: -num.d2,
                d3: num.d3,
                score_log_sigma: log_sigma_score_from_log_sum(&num, sigma),
                neg_hessian_log_sigma: log_sigma_neg_hessian_from_log_sum(&num, sigma),
            })
        }
    }

    /// Interval event: `ℓ = log(K_{0,M_L} − K_{0,M_R}) − log K_{0,M_in}`.
    fn interval_censored(
        quadctx: &QuadratureContext,
        mu: f64,
        sigma: f64,
        row: &LatentSurvivalRow,
    ) -> Result<Self, EstimationError> {
        let num_terms = [
            KernelSumTerm {
                coeff: (-row.mass_unloaded_left).exp(),
                k: 0,
                m: row.mass_left,
            },
            KernelSumTerm {
                coeff: -(-row.mass_unloaded_right).exp(),
                k: 0,
                m: row.mass_right,
            },
        ];
        let num = LogKernelSumJet::evaluate(quadctx, &num_terms, mu, sigma)?;

        if row.mass_entry > 0.0 {
            let den = LogKernelSumJet::single_term(quadctx, 0, row.mass_entry, mu, sigma)?;
            Ok(Self {
                log_lik: num.value + row.mass_unloaded_entry - den.value,
                score: num.d1 - den.d1,
                neg_hessian: -(num.d2 - den.d2),
                d3: num.d3 - den.d3,
                score_log_sigma: log_sigma_score_from_log_sum(&num, sigma)
                    - log_sigma_score_from_log_sum(&den, sigma),
                neg_hessian_log_sigma: log_sigma_neg_hessian_from_log_sum(&num, sigma)
                    - log_sigma_neg_hessian_from_log_sum(&den, sigma),
            })
        } else {
            Ok(Self {
                log_lik: num.value + row.mass_unloaded_entry,
                score: num.d1,
                neg_hessian: -num.d2,
                d3: num.d3,
                score_log_sigma: log_sigma_score_from_log_sum(&num, sigma),
                neg_hessian_log_sigma: log_sigma_neg_hessian_from_log_sum(&num, sigma),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #2610: the reformulation is an IDENTITY, so where the differenced form is
    /// trustworthy the two must agree to near machine epsilon.
    ///
    /// #2566 measured the differenced channel's step-to-step relative jump
    /// growing `0.05, 0.07, 0.19, 0.47, 0.84, 2.89, 9.78` and changing SIGN
    /// between adjacent samples past `log σ ≈ 5.45`. Sign flips between
    /// neighbouring points of a smooth function are the signature of
    /// cancellation, not of a branch. This walks the same ladder and compares
    /// the two formations on identical bundles, so the only difference is the
    /// arithmetic.
    ///
    /// SCOPE: on this fixture (`m = 1`, `mu = 0`, rung `k = 0`) the differenced
    /// form does NOT degrade -- worst step jump `0.095`, no sign flips -- because
    /// `rk2/rk1^2` grows like `e^(sigma^2)` here, so the two moments diverge
    /// instead of colliding. The regime #2566 reported therefore lives in the
    /// jet combination or the two-term mixture, not in the raw ratios, and
    /// locating it in terms of `(m, mu, k)` is still open. What this test
    /// establishes is that the reformulation is the SAME NUMBER as the thing it
    /// replaces; that it also repairs the bad regime is not yet demonstrated.
    #[test]
    fn zz_measure_2610_reformulation_reproduces_the_differenced_value_it_replaces() {
        let quadctx = QuadratureContext::new();
        let (mass, mu) = (1.0_f64, 0.0_f64);

        let mut rows: Vec<(f64, f64, f64)> = Vec::new();
        let mut log_sigma = 4.0_f64;
        while log_sigma <= 7.0001 {
            let sigma = log_sigma.exp();
            let bundle = log_kernel_bundle(&quadctx, mass, mu, sigma, 4)
                .expect("bundle over the measured range");
            // Differenced form: exactly what a consumer writes today.
            let ratio1 = (bundle.get(1) - bundle.get(0)).exp();
            let ratio2 = (bundle.get(2) - bundle.get(0)).exp();
            let differenced = ratio2 - ratio1 * ratio1;
            let stable = bundle
                .second_cumulant_ratio(0, sigma)
                .expect("cancellation-free form is defined on this range");
            println!(
                "[2610] log_sigma={log_sigma:.2} differenced={differenced:.12e} stable={stable:.12e}"
            );
            rows.push((log_sigma, differenced, stable));
            log_sigma += 0.1;
        }

        // Step-to-step relative jump, the quantity #2566 reported.
        let jump = |a: f64, b: f64| -> f64 {
            let scale = a.abs().max(b.abs());
            if scale > 0.0 { (b - a).abs() / scale } else { 0.0 }
        };
        let mut worst_differenced = 0.0_f64;
        let mut sign_flips_differenced = 0usize;
        for pair in rows.windows(2) {
            let (d0, d1) = (pair[0].1, pair[1].1);
            let differenced_jump = jump(d0, d1);
            if differenced_jump > worst_differenced {
                worst_differenced = differenced_jump;
            }
            if d0 * d1 < 0.0 {
                sign_flips_differenced += 1;
            }
        }

        let mut worst_disagreement = 0.0_f64;
        for row in &rows {
            let scale = row.1.abs().max(row.2.abs());
            if scale > 0.0 {
                let relative = (row.1 - row.2).abs() / scale;
                if relative > worst_disagreement {
                    worst_disagreement = relative;
                }
            }
        }
        println!(
            "[2610] worst differenced step jump={worst_differenced:.6} \
             sign flips={sign_flips_differenced} worst disagreement={worst_disagreement:.3e}"
        );

        // Measured, not assumed: this ladder is WELL CONDITIONED. The differenced
        // form is smooth here and never changes sign, which is exactly what makes
        // the equivalence check below a statement about the algebra rather than
        // about which of two noisy channels is noisier.
        //
        // It also bounds what this test may be read as saying. It does NOT reach
        // the regime #2566 reported. An earlier revision asserted only that the
        // reformulated channel behaved, and passed -- on a ladder where the
        // channel it replaces was already fine. That green meant nothing, and
        // this assertion exists so the same mistake cannot be made silently.
        assert!(
            worst_differenced < 1.0 && sign_flips_differenced == 0,
            "#2610: this ladder is supposed to be well conditioned, so that agreement \
             between the two formations means something (worst jump {worst_differenced}, \
             {sign_flips_differenced} sign flips)"
        );
        assert!(
            rows.iter().all(|row| row.2.is_finite()),
            "#2610: the cancellation-free second cumulant must be finite across the ladder"
        );
        // `-R2 * expm1(D)` is an IDENTITY for `R2 - R1^2`, so where the
        // subtraction is trustworthy the two must agree to near machine epsilon.
        // The tightness is the point: a sign slip, or a wrong closed form for the
        // prefix's second difference, would miss this by orders rather than
        // slightly.
        assert!(
            worst_disagreement < 1.0e-12,
            "#2610: the reformulation must reproduce the differenced value wherever that \
             value is trustworthy; worst relative disagreement {worst_disagreement:e}"
        );
    }

    #[test]
    fn frailty_scale_validation_distinguishes_fixed_and_learned_domains() {
        assert!(FrailtySpec::None.validate().is_ok());
        assert!(
            FrailtySpec::GaussianShift {
                scale: FrailtyScale::Fixed { sigma: 0.75 },
            }
            .validate()
            .is_ok()
        );
        assert!(
            FrailtySpec::HazardMultiplier {
                scale: FrailtyScale::Learned { initial_sigma: 0.5 },
                loading: HazardLoading::Full,
            }
            .validate()
            .is_ok()
        );
        assert!(
            FrailtySpec::GaussianShift {
                scale: FrailtyScale::Fixed { sigma: -0.1 },
            }
            .validate()
            .is_err()
        );
        assert!(
            FrailtySpec::GaussianShift {
                scale: FrailtyScale::Fixed { sigma: f64::NAN },
            }
            .validate()
            .is_err()
        );
        assert!(
            FrailtySpec::GaussianShift {
                scale: FrailtyScale::Learned { initial_sigma: 0.0 },
            }
            .validate()
            .is_err()
        );
        assert!(
            FrailtySpec::GaussianShift {
                scale: FrailtyScale::Learned {
                    initial_sigma: f64::INFINITY,
                },
            }
            .validate()
            .is_err()
        );
    }

    fn latent_binomial_row_log_lik(
        ctx: &QuadratureContext,
        eta: f64,
        sigma: f64,
        y: f64,
        weight: f64,
    ) -> f64 {
        let mu = latent_cloglog_jet5(ctx, eta, sigma)
            .expect("latent jet")
            .mean;
        let mu = mu.clamp(1e-12, 1.0 - 1e-12);
        weight * (y * mu.ln() + (1.0 - y) * (1.0 - mu).ln())
    }

    #[test]
    fn survival_right_censored_score_fd() {
        let ctx = QuadratureContext::new();
        let mu = -0.5;
        let sigma = 0.3;
        let h = 1e-6;
        let row = LatentSurvivalRow::right_censored(0.0, 2.0, 0.0, 0.0);
        let ll_p = LatentSurvivalRowJet::evaluate(&ctx, &row, mu + h, sigma)
            .unwrap()
            .log_lik;
        let ll_m = LatentSurvivalRowJet::evaluate(&ctx, &row, mu - h, sigma)
            .unwrap()
            .log_lik;
        let fd_score = (ll_p - ll_m) / (2.0 * h);
        let jet = LatentSurvivalRowJet::evaluate(&ctx, &row, mu, sigma).unwrap();
        assert!(
            (jet.score - fd_score).abs() / fd_score.abs().max(1e-15) < 1e-3,
            "score={}, fd={fd_score}",
            jet.score
        );
    }

    #[test]
    fn survival_exact_event_score_fd() {
        let ctx = QuadratureContext::new();
        let mu = 0.2;
        let sigma = 0.5;
        let h = 1e-6;
        let row = LatentSurvivalRow::exact_event(0.0, 1.5, 0.0, 0.0, (-0.3f64).exp(), 0.0);
        let ll_p = LatentSurvivalRowJet::evaluate(&ctx, &row, mu + h, sigma)
            .unwrap()
            .log_lik;
        let ll_m = LatentSurvivalRowJet::evaluate(&ctx, &row, mu - h, sigma)
            .unwrap()
            .log_lik;
        let fd_score = (ll_p - ll_m) / (2.0 * h);
        let jet = LatentSurvivalRowJet::evaluate(&ctx, &row, mu, sigma).unwrap();
        assert!(
            (jet.score - fd_score).abs() / fd_score.abs().max(1e-15) < 1e-3,
            "score={}, fd={fd_score}",
            jet.score
        );
    }

    #[test]
    fn survival_exact_event_loaded_vs_unloaded_score_fd() {
        let ctx = QuadratureContext::new();
        let mu = -0.1;
        let sigma = 0.4;
        let h = 1e-6;
        let row = LatentSurvivalRow::exact_event(0.3, 1.2, 0.2, 0.6, 0.9, 0.15);
        let ll_p = LatentSurvivalRowJet::evaluate(&ctx, &row, mu + h, sigma)
            .unwrap()
            .log_lik;
        let ll_m = LatentSurvivalRowJet::evaluate(&ctx, &row, mu - h, sigma)
            .unwrap()
            .log_lik;
        let fd_score = (ll_p - ll_m) / (2.0 * h);
        let jet = LatentSurvivalRowJet::evaluate(&ctx, &row, mu, sigma).unwrap();
        assert!(
            (jet.score - fd_score).abs() / fd_score.abs().max(1e-15) < 1e-3,
            "score={}, fd={fd_score}",
            jet.score
        );
    }

    #[test]
    fn survival_right_censored_loaded_vs_unloaded_score_fd() {
        let ctx = QuadratureContext::new();
        let mu = 0.15;
        let sigma: f64 = 0.35;
        let h = 1e-6;
        let row = LatentSurvivalRow::right_censored(0.4, 1.7, 0.1, 0.5);
        let ll_p = LatentSurvivalRowJet::evaluate(&ctx, &row, mu + h, sigma)
            .unwrap()
            .log_lik;
        let ll_m = LatentSurvivalRowJet::evaluate(&ctx, &row, mu - h, sigma)
            .unwrap()
            .log_lik;
        let fd_score = (ll_p - ll_m) / (2.0 * h);
        let jet = LatentSurvivalRowJet::evaluate(&ctx, &row, mu, sigma).unwrap();
        assert!(
            (jet.score - fd_score).abs() / fd_score.abs().max(1e-15) < 1e-3,
            "score={}, fd={fd_score}",
            jet.score
        );
    }

    #[test]
    fn survival_interval_censored_score_fd() {
        let ctx = QuadratureContext::new();
        let mu = 0.0;
        let sigma = 0.6;
        let h = 1e-6;
        let row = LatentSurvivalRow::interval_censored(0.0, 1.0, 2.0, 0.0, 0.0, 0.0);
        let ll_p = LatentSurvivalRowJet::evaluate(&ctx, &row, mu + h, sigma)
            .unwrap()
            .log_lik;
        let ll_m = LatentSurvivalRowJet::evaluate(&ctx, &row, mu - h, sigma)
            .unwrap()
            .log_lik;
        let fd_score = (ll_p - ll_m) / (2.0 * h);
        let jet = LatentSurvivalRowJet::evaluate(&ctx, &row, mu, sigma).unwrap();
        assert!(
            (jet.score - fd_score).abs() / fd_score.abs().max(1e-15) < 1e-3,
            "score={}, fd={fd_score}",
            jet.score
        );
    }

    /// #2277 hardening: a NARROW interval-censored window (S(L) ≈ S(R)) must
    /// stay numerically stable. The interval contribution `log[S(L) − S(R)]` is
    /// evaluated in the log domain (sign-aware log-sum-exp + `log1mexp`), never
    /// as a probability-space `S(L) − S(R)` subtraction, so for a small gap
    /// `Δ = M_R − M_L` the interval mass is `≈ |S'(M_L)|·Δ` and the
    /// log-likelihood behaves as `const + log Δ` — finite and accurate down to
    /// gaps where subtracting two nearly-equal survival probabilities would
    /// catastrophically cancel. Pin the log-linear-in-Δ law: the shared,
    /// cancellation-prone `const` drops out of the difference, leaving exactly
    /// `log(Δ₁/Δ₂)`.
    #[test]
    fn survival_narrow_interval_is_log_domain_stable_issue_2277() {
        let ctx = QuadratureContext::new();
        let (mu, sigma, m_l) = (0.0_f64, 0.6_f64, 1.0_f64);
        // Returns the row log-likelihood together with the interval width the
        // row ACTUALLY holds.
        //
        // A nominal gap this narrow is not representable as `m_l + gap`: the
        // double spacing at 1.0 is 2.22e-16, so `1e-12` is 4504.5 ulps and the
        // constructed right mass carries the ROUNDED width — up to 1.1e-4
        // relative away from the nominal one, and 1.1e-2 at `1e-14`. Asserting
        // `ll(g1) − ll(g2) = ln(g1_nominal/g2_nominal)` therefore charges the
        // kernel arithmetic for the fixture's own quantization. The law is
        // `ll = const + log Δ` in the width the row was built with, so that is
        // what it is asserted against.
        let ll_and_width = |gap: f64| -> (f64, f64) {
            let m_r = m_l + gap;
            let row = LatentSurvivalRow::interval_censored(0.0, m_l, m_r, 0.0, 0.0, 0.0);
            let log_lik = LatentSurvivalRowJet::evaluate(&ctx, &row, mu, sigma)
                .unwrap()
                .log_lik;
            (log_lik, m_r - m_l)
        };
        // REPRESENTABILITY and ACCURACY are different claims, and a previous
        // revision of this fixture correctly observed that the differenced
        // two-term path delivered only the first at an arbitrarily narrow gap:
        // `evaluate_two_terms` subtracted two INDEPENDENTLY evaluated
        // log-kernels, cancelling as many digits as a probability-space
        // `S(L) − S(R)` would, so at `Δ = 1e-12` the separation carried ~2e-4
        // RELATIVE error. It therefore narrowed the law to `(1e-6, 1e-9)` and
        // kept only finiteness at `1e-12`.
        //
        // `analytic_log_mag_gap` removes that subtraction — the separation is now
        // a Taylor expansion of `Λ` in `dv = ln(M_R/M_L)`, whose leading order is
        // analytic — so both claims hold at the extreme gap and the original
        // `(1e-8, 1e-12)` pair is restored below. Finiteness is still asserted
        // separately at `1e-12`, because it is a different property from the law
        // and is the one the log domain buys on its own.
        assert!(
            ll_and_width(1e-12).0.is_finite(),
            "narrow-interval log-lik must stay finite at the extreme gap: {}",
            ll_and_width(1e-12).0
        );
        let (ll1, width1) = ll_and_width(1e-8);
        let (ll2, width2) = ll_and_width(1e-12);
        assert!(
            ll1.is_finite() && ll2.is_finite(),
            "narrow-interval log-lik must stay finite: ll1={ll1}, ll2={ll2}"
        );
        // const + log Δ ⇒ ll(Δ₁) − ll(Δ₂) = log(Δ₁/Δ₂), independent of the
        // shared const a probability-space subtraction would destroy here.
        let expected = (width1 / width2).ln();
        assert!(
            (ll1 - ll2 - expected).abs() < 1e-5,
            "narrow interval must follow the log-domain log(Δ) law: \
             ll(Δ₁)-ll(Δ₂)={}, expected {expected} (Δ₁={width1:e}, Δ₂={width2:e})",
            ll1 - ll2
        );

        // The law is asserted over a LADDER, not one pair, because the failure
        // mode it guards is an accuracy loss `∝ 1/Δ`: a single pair cannot tell
        // "accurate everywhere" from "accurate at one point". The differenced
        // form missed the pair above by 2.2557e-4 — 22× the tolerance — and the
        // miss GREW as the gap shrank.
        //
        // `ll(Δ) = const + log Δ + O(Δ)`; the `O(Δ)` remainder is real, from the
        // curvature of `log1mexp` and of `ln1p(Δ/M_L)`, with a coefficient set
        // by `Λ''/Λ'` at this `(μ, σ, m)`. The bound is therefore an `O(Δ)` term
        // with a deliberately generous constant PLUS a floor. The floor is the
        // claim that matters: at `Δ = 1e-14` the bound is `1e-9`, five orders
        // below what the differenced formulation delivered, so it asserts that
        // the accuracy does not degrade as `1/Δ`.
        let mut gap = 1e-6_f64;
        let (mut previous, mut previous_width) = ll_and_width(gap);
        while gap > 1e-13 {
            let next_gap = gap * 1e-2;
            let (next, next_width) = ll_and_width(next_gap);
            let step_expected = (previous_width / next_width).ln();
            let residual = previous - next - step_expected;
            assert!(
                residual.abs() < 1e-9 + 100.0 * gap,
                "log(Δ) law must hold at every rung: Δ {previous_width:e}->{next_width:e} \
                 gave {}, expected {step_expected}, residual {residual:e}",
                previous - next
            );
            gap = next_gap;
            previous = next;
            previous_width = next_width;
        }
    }
    #[test]
    fn survival_interval_censored_neg_hessian_fd() {
        // Second μ-derivative of ℓ = log[S(L) − S(R)] for the interval kernel,
        // FD-checked. `neg_hessian` stores −d²ℓ/dμ², so compare against the
        // negated central second difference.
        let ctx = QuadratureContext::new();
        let mu = -0.2;
        let sigma = 0.55;
        let h = 2e-4;
        let row = LatentSurvivalRow::interval_censored(0.0, 0.7, 1.9, 0.0, 0.0, 0.0);
        let ll = |m: f64| {
            LatentSurvivalRowJet::evaluate(&ctx, &row, m, sigma)
                .unwrap()
                .log_lik
        };
        let fd_d2 = (ll(mu + h) - 2.0 * ll(mu) + ll(mu - h)) / (h * h);
        let jet = LatentSurvivalRowJet::evaluate(&ctx, &row, mu, sigma).unwrap();
        assert!(
            (jet.neg_hessian - (-fd_d2)).abs() / fd_d2.abs().max(1e-12) < 1e-2,
            "interval neg_hessian={}, fd(-d2)={}",
            jet.neg_hessian,
            -fd_d2
        );
    }

    #[test]
    fn survival_interval_censored_log_sigma_score_fd() {
        // σ-recovery for interval data is driven by `score_log_sigma`, the
        // derivative of ℓ = log[S(L) − S(R)] w.r.t. log σ. FD-check it directly
        // against the row log-likelihood (this is the channel the interval fit's
        // latent_sd estimate moves along, the test's primary metric).
        let ctx = QuadratureContext::new();
        let mu = 0.1;
        let sigma: f64 = 0.6;
        let h = 1e-5;
        let row = LatentSurvivalRow::interval_censored(0.0, 0.8, 2.1, 0.0, 0.0, 0.0);
        let ll_at = |s: f64| {
            LatentSurvivalRowJet::evaluate(&ctx, &row, mu, s)
                .unwrap()
                .log_lik
        };
        // d/d(log σ) = σ · d/dσ, so FD over log σ directly.
        let fd_dlogsigma =
            (ll_at((sigma.ln() + h).exp()) - ll_at((sigma.ln() - h).exp())) / (2.0 * h);
        let jet = LatentSurvivalRowJet::evaluate(&ctx, &row, mu, sigma).unwrap();
        assert!(
            (jet.score_log_sigma - fd_dlogsigma).abs() / fd_dlogsigma.abs().max(1e-12) < 1e-3,
            "interval score_log_sigma={}, fd={fd_dlogsigma}",
            jet.score_log_sigma
        );
    }

    #[test]
    fn latent_cloglog_jet_matches_point_limit_at_zero_sigma() {
        let ctx = QuadratureContext::new();
        let eta = -0.4;
        let jet = latent_cloglog_jet5(&ctx, eta, 0.0).expect("latent jet");
        let t = eta.exp();
        let d1 = (eta - t).exp();
        let d2 = (1.0 - t) * d1;
        let d3 = (t * t - 3.0 * t + 1.0) * d1;
        let d4 = (-t * t * t + 6.0 * t * t - 7.0 * t + 1.0) * d1;
        let d5 = (t.powi(4) - 10.0 * t.powi(3) + 25.0 * t * t - 15.0 * t + 1.0) * d1;
        assert!((jet.mean - (1.0 - (-t).exp())).abs() < 1e-12);
        assert!((jet.d1 - d1).abs() < 1e-12);
        assert!((jet.d2 - d2).abs() < 1e-12);
        assert!((jet.d3 - d3).abs() < 1e-12);
        assert!((jet.d4 - d4).abs() < 1e-12);
        assert!((jet.d5 - d5).abs() < 1e-12);
    }

    #[test]
    fn latent_cloglog_jet_matches_exact_kernel_recurrence() {
        let ctx = QuadratureContext::new();
        let cases = [(-4.0, 0.15), (-1.2, 0.35), (0.4, 0.6), (1.3, 0.9)];

        for (eta, sigma) in cases {
            let jet = latent_cloglog_jet5(&ctx, eta, sigma).expect("latent jet");
            let bundle = log_kernel_bundle(&ctx, 1.0, eta, sigma, 5).expect("kernel bundle");
            let k0 = bundle.get(0);
            let k1 = bundle.get(1).exp();
            let k2 = bundle.get(2).exp();
            let k3 = bundle.get(3).exp();
            let k4 = bundle.get(4).exp();
            let k5 = bundle.get(5).exp();

            let mean = if k0.is_finite() { -k0.exp_m1() } else { 1.0 };
            let d1 = k1;
            let d2 = k1 - k2;
            let d3 = k1 - 3.0 * k2 + k3;
            let d4 = k1 - 7.0 * k2 + 6.0 * k3 - k4;
            let d5 = k1 - 15.0 * k2 + 25.0 * k3 - 10.0 * k4 + k5;

            assert!((jet.mean - mean).abs() < 1e-12);
            assert!((jet.d1 - d1).abs() < 1e-12);
            assert!((jet.d2 - d2).abs() < 1e-12);
            assert!((jet.d3 - d3).abs() < 1e-12);
            assert!((jet.d4 - d4).abs() < 1e-12);
            assert!((jet.d5 - d5).abs() < 1e-12);
        }
    }

    #[test]
    fn latent_cloglog_binomial_row_neg_hessian_matches_fd() {
        let ctx = QuadratureContext::new();
        let eta = 0.4;
        let sigma = 0.6;
        let y = 0.35;
        let weight = 2.0;
        let h = 1e-4;

        let jet = latent_cloglog_jet5(&ctx, eta, sigma).expect("latent jet");
        let mu = jet.mean.clamp(1e-12, 1.0 - 1e-12);
        let ellmu = y / mu - (1.0 - y) / (1.0 - mu);
        let ellmumu = -y / (mu * mu) - (1.0 - y) / ((1.0 - mu) * (1.0 - mu));
        let neg_hessian = -weight * (ellmumu * jet.d1 * jet.d1 + ellmu * jet.d2);

        let ll_minus = latent_binomial_row_log_lik(&ctx, eta - h, sigma, y, weight);
        let ll0 = latent_binomial_row_log_lik(&ctx, eta, sigma, y, weight);
        let ll_plus = latent_binomial_row_log_lik(&ctx, eta + h, sigma, y, weight);
        let neg_hessian_fd = -(ll_plus - 2.0 * ll0 + ll_minus) / (h * h);

        let err = (neg_hessian - neg_hessian_fd).abs();
        let tol = 2e-5_f64.max(3e-3 * neg_hessian_fd.abs());
        assert!(
            err <= tol,
            "latent cloglog Bernoulli row curvature mismatch: analytic={} fd={}",
            neg_hessian,
            neg_hessian_fd
        );
    }
}
