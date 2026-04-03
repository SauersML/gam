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

use crate::estimate::EstimationError;
use crate::quadrature::{
    IntegratedExpectationMode, IntegratedInverseLinkJet, QuadratureContext,
    latent_cloglog_inverse_link_jet5_controlled, lognormal_laplace_unit_term_shared,
    validate_latent_cloglog_inputs,
};
use serde::{Deserialize, Serialize};
use std::fmt;

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
    /// This is the biologically faithful model for Gompertz-Makeham.
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
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "frailty_kind", rename_all = "kebab-case")]
pub enum FrailtySpec {
    /// No frailty modifier.
    None,
    /// Gaussian shift on the final scalar index: U ~ N(0, σ²) added to η.
    /// Exact for probit: E[Φ(η + U)] = Φ(η / √(1+σ²)).
    /// The existing sextic microcell kernel is preserved.
    GaussianShift {
        /// Fixed σ, or None if learnable.
        sigma_fixed: Option<f64>,
    },
    /// Lognormal hazard multiplier: conditional hazard h(t|U) involves exp(U).
    /// Exact for PH/cloglog/survival via K_{k,m} kernel.
    HazardMultiplier {
        /// Fixed σ, or None if learnable.
        sigma_fixed: Option<f64>,
        /// How the multiplier loads onto hazard components.
        loading: HazardLoading,
    },
}

impl FrailtySpec {
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
        match self {
            Self::None | Self::GaussianShift { .. } => Ok(()),
            Self::HazardMultiplier { .. } => Err(
                "HazardMultiplier frailty is not finite-state exact with score_warp/linkwiggle \
                 cubic marginal-slope families. Use GaussianShift frailty (exact probit scaling) \
                 or use the standalone latent-cloglog/latent-survival families instead."
                    .to_string(),
            ),
        }
    }
}

// ─── Probit frailty scaling ──────────────────────────────────────────────────

/// Probit frailty scaling factor s = 1/√(1+σ²) and its derivatives.
///
/// For Gaussian frailty on the final probit index:
///   E[Φ(η + U)] = Φ(η · s)  where s = 1/√(1+σ²)
///
/// With t = log(σ), define α = σ²/(1+σ²).  Then:
///   ∂_t s = -α·s
///   ∂_{tt} s = α(3α-2)·s
///   ∂_{ttt} s = α(-15α²+18α-4)·s
///   ∂_{tttt} s = α(105α³-180α²+84α-8)·s
#[derive(Clone, Copy, Debug)]
pub struct ProbitFrailtyScale {
    /// s = 1/√(1+σ²)
    pub s: f64,
}

impl ProbitFrailtyScale {
    pub fn new(sigma: f64) -> Self {
        let sigma2 = sigma * sigma;
        Self {
            s: 1.0 / (1.0 + sigma2).sqrt(),
        }
    }
}

/// Probit frailty scaling factor **with** t-derivatives (t = log σ).
///
/// Provides exact closed-form derivatives of s = 1/√(1+σ²) with respect to
/// t = log(σ) for learnable Gaussian-shift frailty in the marginal-slope
/// families.  All formulas follow from the recurrence documented on
/// [`ProbitFrailtyScale`]; the auxiliary quantity α = σ²/(1+σ²) keeps each
/// expression compact.
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
        let sigma2 = sigma * sigma;
        let one_plus_sigma2 = 1.0 + sigma2;
        let s = 1.0 / one_plus_sigma2.sqrt();
        let alpha = sigma2 / one_plus_sigma2;
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
fn mode_rank(mode: IntegratedExpectationMode) -> u8 {
    match mode {
        IntegratedExpectationMode::ExactClosedForm => 0,
        IntegratedExpectationMode::ExactSpecialFunction => 1,
        IntegratedExpectationMode::ControlledAsymptotic => 2,
        IntegratedExpectationMode::QuadratureFallback => 3,
    }
}

#[inline]
fn worst_mode(
    a: IntegratedExpectationMode,
    b: IntegratedExpectationMode,
) -> IntegratedExpectationMode {
    if mode_rank(a) >= mode_rank(b) { a } else { b }
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
        return Err(EstimationError::InvalidInput(format!(
            "lognormal kernel requires finite m >= 0, got {m}"
        )));
    }
    if !mu.is_finite() || !sigma.is_finite() || sigma < 0.0 {
        return Err(EstimationError::InvalidInput(format!(
            "lognormal kernel requires finite mu and sigma >= 0, got mu={mu}, sigma={sigma}"
        )));
    }
    Ok(())
}

#[inline]
pub fn log_kernel_term(
    quadctx: &QuadratureContext,
    k: usize,
    m: f64,
    mu: f64,
    sigma: f64,
) -> Result<(f64, IntegratedExpectationMode), EstimationError> {
    validate_kernel_inputs(m, mu, sigma)?;
    let kf = k as f64;
    let prefix = kf * mu + 0.5 * kf * kf * sigma * sigma;
    if m == 0.0 {
        return Ok((prefix, IntegratedExpectationMode::ExactClosedForm));
    }
    let shifted_mu = mu + kf * sigma * sigma + m.ln();
    let (laplace, mode) = lognormal_laplace_unit_term_shared(quadctx, shifted_mu, sigma);
    if laplace <= 0.0 {
        return Ok((f64::NEG_INFINITY, mode));
    }
    Ok((prefix + laplace.ln(), mode))
}

/// Kernel bundle storing `log K_{k,m}` values instead of `K_{k,m}`.
#[derive(Clone, Debug)]
pub struct LogLognormalKernelBundle {
    pub log_values: Vec<f64>,
    pub mode: IntegratedExpectationMode,
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
    if m == 0.0 {
        let sigma2 = sigma * sigma;
        let mut prefix = 0.0;
        for k in 0..=max_k {
            log_values.push(prefix);
            prefix += mu + (k as f64 + 0.5) * sigma2;
        }
        return Ok(LogLognormalKernelBundle {
            log_values,
            mode: IntegratedExpectationMode::ExactClosedForm,
        });
    }

    let sigma2 = sigma * sigma;
    let log_m = m.ln();
    let mut shifted_mu = mu + log_m;
    let mut prefix = 0.0;
    let mut mode = IntegratedExpectationMode::ExactClosedForm;
    for k in 0..=max_k {
        let (laplace, val_mode) = lognormal_laplace_unit_term_shared(quadctx, shifted_mu, sigma);
        log_values.push(if laplace > 0.0 {
            prefix + laplace.ln()
        } else {
            f64::NEG_INFINITY
        });
        mode = worst_mode(mode, val_mode);
        prefix += mu + (k as f64 + 0.5) * sigma2;
        shifted_mu += sigma2;
    }
    Ok(LogLognormalKernelBundle { log_values, mode })
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

/// Computes `log(1 − exp(−a))` for `a > 0`, numerically stable.
///
/// Uses `ln_1p(−exp(−a))` for large `a` (avoids catastrophic cancellation)
/// and `ln(−expm1(−a))` for small `a` (avoids loss of significance).
#[inline]
fn log1mexp(a: f64) -> f64 {
    assert!(a >= 0.0);
    if a > core::f64::consts::LN_2 {
        // exp(-a) < 0.5, so 1 - exp(-a) > 0.5: safe for ln_1p.
        (-(-a).exp()).ln_1p()
    } else if a > 0.0 {
        // exp(-a) close to 1: use expm1 for precision.
        // expm1(-a) = exp(-a) - 1, negate to get 1 - exp(-a).
        (-(-a).exp_m1()).ln()
    } else {
        f64::NEG_INFINITY
    }
}

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
    // Authoritative latent cloglog backend via `quadrature.rs`:
    //
    // - mean through d5 are all derived from the same lognormal-Laplace kernel
    //   terms K_{k,1}(eta, sigma),
    // - there is no finite-difference bridge for d2 / d3,
    // - and every derivative order uses the same routed kernel backend.
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
            d4: r4 - 4.0 * r1 * r3 - 3.0 * r2 * r2 + 12.0 * r1 * r1 * r2
                - 6.0 * r1.powi(4),
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
            kernel_ratio_jet(bundle, term.k, term.m, 3),
        )
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
        let bundle1_owned = if (t0.m - t1.m).abs() < 1e-300 {
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
        let (log_s, sign_s) = signed_log_sum_exp(&log_mags, &signs);
        if !log_s.is_finite() || sign_s <= 0.0 {
            return Ok(Self::non_positive(overall_mode));
        }

        let w0 = sign_s * signs[0] * (log_mag0 - log_s).exp();
        let w1 = sign_s * signs[1] * (log_mag1 - log_s).exp();
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
            kernel_ratio_jet(&lb, k, m, 3),
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
        assert!(!terms.is_empty(), "KernelSumJet requires at least one term");

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
            if !log_bundles
                .iter()
                .any(|(m, _)| (*m - term.m).abs() < 1e-300)
            {
                let b = log_kernel_bundle(quadctx, term.m, mu, sigma, max_k_needed)?;
                overall_mode = worst_mode(overall_mode, b.mode);
                log_bundles.push((term.m, b));
            }
        }

        let get_lb = |m: f64| -> &LogLognormalKernelBundle {
            &log_bundles
                .iter()
                .find(|(bm, _)| (*bm - m).abs() < 1e-300)
                .unwrap()
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
            ratios.push(kernel_ratio_jet(lb, term.k, term.m, 3));
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

/// Sign-aware log-sum-exp: computes `log|Σ s_j exp(l_j)|` and the sign.
///
/// Separates positive and negative contributions, applies log-sum-exp within
/// each group, then uses [`log1mexp`] for the group difference.
fn signed_log_sum_exp(log_mags: &[f64], signs: &[f64]) -> (f64, f64) {
    // Separate into positive and negative buckets.
    let mut pos_max = f64::NEG_INFINITY;
    let mut neg_max = f64::NEG_INFINITY;
    for (i, &lm) in log_mags.iter().enumerate() {
        if signs[i] > 0.0 {
            pos_max = pos_max.max(lm);
        } else if signs[i] < 0.0 {
            neg_max = neg_max.max(lm);
        }
    }

    // Accumulate exp(l − max) within each bucket.
    let mut pos_sum = 0.0f64;
    let mut neg_sum = 0.0f64;
    for (i, &lm) in log_mags.iter().enumerate() {
        if !lm.is_finite() {
            continue;
        }
        if signs[i] > 0.0 {
            pos_sum += (lm - pos_max).exp();
        } else if signs[i] < 0.0 {
            neg_sum += (lm - neg_max).exp();
        }
    }

    let log_pos = if pos_sum > 0.0 {
        pos_max + pos_sum.ln()
    } else {
        f64::NEG_INFINITY
    };
    let log_neg = if neg_sum > 0.0 {
        neg_max + neg_sum.ln()
    } else {
        f64::NEG_INFINITY
    };

    if log_neg == f64::NEG_INFINITY {
        // All positive (or all zero).
        return (log_pos, 1.0);
    }
    if log_pos == f64::NEG_INFINITY {
        // All negative.
        return (log_neg, -1.0);
    }

    // Both groups present: compute log|pos − neg| and its sign.
    if log_pos > log_neg {
        // Positive dominates: log(pos − neg) = log_pos + log(1 − exp(−(log_pos − log_neg)))
        let gap = log_pos - log_neg;
        (log_pos + log1mexp(gap), 1.0)
    } else if log_neg > log_pos {
        // Negative dominates.
        let gap = log_neg - log_pos;
        (log_neg + log1mexp(gap), -1.0)
    } else {
        // Exact cancellation.
        (f64::NEG_INFINITY, 0.0)
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
                return Err(EstimationError::InvalidInput(format!(
                    "latent survival row has invalid {name}={value}; expected a finite non-negative value"
                )));
            }
        }

        match self.event_type {
            LatentSurvivalEventType::RightCensored => {
                if self.mass_exit < self.mass_entry {
                    return Err(EstimationError::InvalidInput(format!(
                        "latent survival right-censored row requires mass_exit >= mass_entry, got {} < {}",
                        self.mass_exit, self.mass_entry
                    )));
                }
                if self.mass_unloaded_exit < self.mass_unloaded_entry {
                    return Err(EstimationError::InvalidInput(format!(
                        "latent survival right-censored row requires unloaded exit mass >= unloaded entry mass, got {} < {}",
                        self.mass_unloaded_exit, self.mass_unloaded_entry
                    )));
                }
                if self.mass_left > 0.0
                    || self.mass_right > 0.0
                    || self.mass_unloaded_left > 0.0
                    || self.mass_unloaded_right > 0.0
                    || self.hazard_loaded > 0.0
                    || self.hazard_unloaded > 0.0
                {
                    return Err(EstimationError::InvalidInput(
                        "latent survival right-censored row cannot carry interval masses or event hazards"
                            .to_string(),
                    ));
                }
            }
            LatentSurvivalEventType::ExactEvent => {
                if self.mass_exit < self.mass_entry {
                    return Err(EstimationError::InvalidInput(format!(
                        "latent survival exact-event row requires mass_exit >= mass_entry, got {} < {}",
                        self.mass_exit, self.mass_entry
                    )));
                }
                if self.mass_unloaded_exit < self.mass_unloaded_entry {
                    return Err(EstimationError::InvalidInput(format!(
                        "latent survival exact-event row requires unloaded exit mass >= unloaded entry mass, got {} < {}",
                        self.mass_unloaded_exit, self.mass_unloaded_entry
                    )));
                }
                if self.mass_left > 0.0
                    || self.mass_right > 0.0
                    || self.mass_unloaded_left > 0.0
                    || self.mass_unloaded_right > 0.0
                {
                    return Err(EstimationError::InvalidInput(
                        "latent survival exact-event row cannot carry interval masses".to_string(),
                    ));
                }
                if self.hazard_loaded == 0.0 && self.hazard_unloaded == 0.0 {
                    return Err(EstimationError::InvalidInput(
                        "latent survival exact-event row requires a positive loaded or unloaded hazard"
                            .to_string(),
                    ));
                }
            }
            LatentSurvivalEventType::IntervalCensored => {
                if self.mass_left < self.mass_entry || self.mass_right < self.mass_left {
                    return Err(EstimationError::InvalidInput(format!(
                        "latent survival interval row requires mass_entry <= mass_left <= mass_right, got entry={}, left={}, right={}",
                        self.mass_entry, self.mass_left, self.mass_right
                    )));
                }
                if self.mass_unloaded_left < self.mass_unloaded_entry
                    || self.mass_unloaded_right < self.mass_unloaded_left
                {
                    return Err(EstimationError::InvalidInput(format!(
                        "latent survival interval row requires unloaded_entry <= unloaded_left <= unloaded_right, got entry={}, left={}, right={}",
                        self.mass_unloaded_entry, self.mass_unloaded_left, self.mass_unloaded_right
                    )));
                }
                if self.mass_exit > 0.0
                    || self.mass_unloaded_exit > 0.0
                    || self.hazard_loaded > 0.0
                    || self.hazard_unloaded > 0.0
                {
                    return Err(EstimationError::InvalidInput(
                        "latent survival interval row cannot carry exit masses or event hazards"
                            .to_string(),
                    ));
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
        return Err(EstimationError::InvalidInput(format!(
            "latent survival exact-event hazards must be non-negative, got loaded={} unloaded={}",
            row.hazard_loaded, row.hazard_unloaded
        )));
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
    let r2 = jet.d2 + jet.d1 * jet.d1;
    let r4 = jet.d4
        + 4.0 * jet.d1 * jet.d3
        + 3.0 * jet.d2 * jet.d2
        + 6.0 * jet.d1 * jet.d1 * jet.d2
        + jet.d1.powi(4);
    -(2.0 * sigma2 * r2 + sigma4 * (r4 - r2 * r2))
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
        let has_unloaded =
            row.mass_unloaded_exit.abs() > 1e-300 || row.mass_unloaded_entry.abs() > 1e-300;

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
        if mass_entry_loaded > 1e-300 {
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
        let unloaded_offset =
            if row.mass_unloaded_exit.abs() > 1e-300 || row.mass_unloaded_entry.abs() > 1e-300 {
                -row.mass_unloaded_exit + row.mass_unloaded_entry
            } else {
                0.0
            };
        let num = exact_event_kernel_jet(quadctx, row, mu, sigma)?;

        if row.mass_entry > 1e-300 {
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

        if row.mass_entry > 1e-300 {
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
    fn kernel_ratio_jet_d1_fd_check() {
        let ctx = QuadratureContext::new();
        let mu = 0.3;
        let sigma = 0.5;
        let m = 1.0;
        let k = 0usize;
        let h = 1e-5;

        let bundle = log_kernel_bundle(&ctx, m, mu, sigma, k + 4).unwrap();
        let log_k = bundle.get(k);
        let ratios = kernel_ratio_jet(&bundle, k, m, 2);
        let kc = log_k.exp();
        let d1 = kc * ratios[1];
        let d2 = kc * ratios[2];

        let kp = log_kernel_term(&ctx, k, m, mu + h, sigma).unwrap().0.exp();
        let km = log_kernel_term(&ctx, k, m, mu - h, sigma).unwrap().0.exp();
        let fd_d1 = (kp - km) / (2.0 * h);
        assert!(
            (d1 - fd_d1).abs() / fd_d1.abs().max(1e-15) < 1e-4,
            "d1: jet={d1}, fd={fd_d1}",
        );

        let fd_d2 = (kp - 2.0 * kc + km) / (h * h);
        assert!(
            (d2 - fd_d2).abs() / fd_d2.abs().max(1e-15) < 1e-3,
            "d2: jet={d2}, fd={fd_d2}",
        );
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
        let sigma = 0.35;
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

    #[test]
    fn log_kernel_sum_jet_single_term_d1_fd() {
        let ctx = QuadratureContext::new();
        let mu = 0.5;
        let sigma = 0.4;
        let m = 1.0;
        let k = 0usize;
        let h = 1e-6;

        let jet = LogKernelSumJet::single_term(&ctx, k, m, mu, sigma).unwrap();
        let val_p = log_kernel_term(&ctx, k, m, mu + h, sigma).unwrap().0;
        let val_m = log_kernel_term(&ctx, k, m, mu - h, sigma).unwrap().0;
        let fd_d1 = (val_p - val_m) / (2.0 * h);
        assert!(
            (jet.d1 - fd_d1).abs() / fd_d1.abs().max(1e-15) < 1e-3,
            "d1={}, fd={fd_d1}",
            jet.d1
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
