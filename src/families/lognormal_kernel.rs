//! Shared analytic kernel for latent-variable families with lognormal structure.
//!
//! The kernel object `K_{k,m}(μ, σ) := E[exp(k·U − m·exp(U))]`, where
//! `U ~ N(μ, σ²)`, is the only special function required by all latent families.
//!
//! It satisfies exact μ-recurrences (see [`kernel_mu_jet`]) and heat-equation
//! σ-identities, so the full derivative calculus for PIRLS,
//! LAML outer, and learnable σ reduces to evaluating kernel bundles at shifted
//! arguments.
//!
//! Row likelihoods for binary and survival models are small signed sums of
//! kernel terms; [`LogKernelSumJet`] provides numerically stable log-space
//! derivatives of such sums.

use crate::estimate::EstimationError;
use crate::quadrature::{
    IntegratedExpectationMode, IntegratedInverseLinkJet, QuadratureContext,
    cloglog_ghq_jet5_adaptive, cloglog_point_jet5, cloglog_posterior_meanwith_deriv_controlled,
    lognormal_laplace_term_shared,
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

#[derive(Clone, Debug)]
pub struct LognormalKernelBundle {
    values: Vec<f64>,
    pub mode: IntegratedExpectationMode,
}

impl LognormalKernelBundle {
    #[inline]
    pub fn get(&self, k: usize) -> f64 {
        self.values[k]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.values.len()
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

#[inline]
pub fn kernel_term(
    quadctx: &QuadratureContext,
    k: usize,
    m: f64,
    mu: f64,
    sigma: f64,
) -> Result<(f64, IntegratedExpectationMode), EstimationError> {
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
    if m == 0.0 {
        let kf = k as f64;
        let log_value = kf * mu + 0.5 * kf * kf * sigma * sigma;
        return Ok((
            if log_value > 709.0 {
                f64::INFINITY
            } else {
                log_value.exp()
            },
            IntegratedExpectationMode::ExactClosedForm,
        ));
    }

    let kf = k as f64;
    let shifted_mu = mu + kf * sigma * sigma;
    let (laplace, mode) = lognormal_laplace_term_shared(quadctx, m, shifted_mu, sigma);
    if laplace <= 0.0 {
        return Ok((0.0, mode));
    }
    let log_value = kf * mu + 0.5 * kf * kf * sigma * sigma + laplace.ln();
    Ok((
        if log_value > 709.0 {
            f64::INFINITY
        } else {
            log_value.exp()
        },
        mode,
    ))
}

pub fn kernel_bundle(
    quadctx: &QuadratureContext,
    m: f64,
    mu: f64,
    sigma: f64,
    max_k: usize,
) -> Result<LognormalKernelBundle, EstimationError> {
    let mut values = Vec::with_capacity(max_k + 1);
    let mut mode = IntegratedExpectationMode::ExactClosedForm;
    for k in 0..=max_k {
        let (value, value_mode) = kernel_term(quadctx, k, m, mu, sigma)?;
        values.push(value);
        mode = worst_mode(mode, value_mode);
    }
    Ok(LognormalKernelBundle { values, mode })
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
    let sigma = sigma.max(0.0);
    if sigma <= 1e-10 {
        let (mean, d1, d2, d3, d4, d5) = cloglog_point_jet5(eta);
        return Ok(LatentCLogLogJet5 {
            mean,
            d1,
            d2,
            d3,
            d4,
            d5,
            mode: IntegratedExpectationMode::ExactClosedForm,
        });
    }

    let base = cloglog_posterior_meanwith_deriv_controlled(quadctx, eta, sigma);
    let (_, _, d2, d3, d4, d5) = cloglog_ghq_jet5_adaptive(quadctx, eta, sigma);
    Ok(LatentCLogLogJet5 {
        mean: base.mean,
        d1: base.dmean_dmu.max(0.0),
        d2,
        d3,
        d4,
        d5,
        mode: base.mode,
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

// ─── μ-derivative jet from kernel recurrences ────────────────────────────────

/// Exact μ-derivatives of `K_{k,m}` up to the specified order, computed from
/// the recurrence relations.
///
/// Returns `[K, ∂_μ K, ∂_{μμ} K, ∂_{μμμ} K, ∂_{μμμμ} K]` for the base order k.
/// Only `order + 1` entries are valid.
///
/// Recurrences:
///   ∂_μ K_{k,m} = k·K_{k,m} − m·K_{k+1,m}
///   ∂_{μμ} K_{k,m} = k²·K_{k,m} − (2k+1)·m·K_{k+1,m} + m²·K_{k+2,m}
///   ∂_{μμμ} K_{k,m} = k³·K − (3k²+3k+1)·m·K_{k+1} + 3(k+1)·m²·K_{k+2} − m³·K_{k+3}
///   ∂_{μμμμ} K_{k,m} = k⁴·K − (4k³+6k²+4k+1)·m·K_{k+1} + (6k²+12k+7)·m²·K_{k+2}
///                        − (4k+6)·m³·K_{k+3} + m⁴·K_{k+4}
pub fn kernel_mu_jet(bundle: &LognormalKernelBundle, k: usize, m: f64, order: usize) -> [f64; 5] {
    let kf = k as f64;
    let a = |r: usize| -> f64 { bundle.get(k + r) };

    let mut jet = [0.0; 5];
    jet[0] = a(0);

    if order >= 1 {
        jet[1] = kf * a(0) - m * a(1);
    }
    if order >= 2 {
        jet[2] = kf * kf * a(0) - (2.0 * kf + 1.0) * m * a(1) + m * m * a(2);
    }
    if order >= 3 {
        jet[3] = kf * kf * kf * a(0) - (3.0 * kf * kf + 3.0 * kf + 1.0) * m * a(1)
            + 3.0 * (kf + 1.0) * m * m * a(2)
            - m * m * m * a(3);
    }
    if order >= 4 {
        let k2 = kf * kf;
        let k3 = k2 * kf;
        let k4 = k3 * kf;
        let m2 = m * m;
        let m3 = m2 * m;
        let m4 = m3 * m;
        jet[4] = k4 * a(0) - (4.0 * k3 + 6.0 * k2 + 4.0 * kf + 1.0) * m * a(1)
            + (6.0 * k2 + 12.0 * kf + 7.0) * m2 * a(2)
            - (4.0 * kf + 6.0) * m3 * a(3)
            + m4 * a(4);
    }

    jet
}

// ─── LogKernelSumJet: log-sum derivatives ────────────────────────────────────

/// Floor for kernel values to avoid log(0).
const LOG_FLOOR: f64 = 1e-300;

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
    pub mode: IntegratedExpectationMode,
}

impl LogKernelSumJet {
    /// Evaluate for a single positive kernel term (fast path).
    ///
    /// Computes `log(K_{k,m})` and its μ-derivatives from exact recurrences.
    pub fn single_term(
        quadctx: &QuadratureContext,
        k: usize,
        m: f64,
        mu: f64,
        sigma: f64,
    ) -> Result<Self, EstimationError> {
        let max_k_needed = k + 4;
        let bundle = kernel_bundle(quadctx, m, mu, sigma, max_k_needed)?;
        let jet = kernel_mu_jet(&bundle, k, m, 3);
        let val = jet[0].max(LOG_FLOOR);
        let log_val = val.ln();
        let r1 = jet[1] / val;
        let r2 = jet[2] / val;
        let r3 = jet[3] / val;

        Ok(Self {
            value: log_val,
            d1: r1,
            d2: r2 - r1 * r1,
            d3: r3 - 3.0 * r1 * r2 + 2.0 * r1 * r1 * r1,
            mode: bundle.mode,
        })
    }

    /// Evaluate `log(Σ a_j K_j)` and its μ-derivatives for a small signed sum.
    ///
    /// All terms share the same (μ, σ).  For differences (e.g. interval
    /// censoring), numerical stability is handled in log-space.
    pub fn evaluate(
        quadctx: &QuadratureContext,
        terms: &[KernelSumTerm],
        mu: f64,
        sigma: f64,
    ) -> Result<Self, EstimationError> {
        assert!(!terms.is_empty(), "KernelSumJet requires at least one term");

        // Find the maximum k needed across all terms (plus 4 for recurrence jets).
        let max_k_needed = terms.iter().map(|t| t.k).max().unwrap_or(0) + 4;

        // Group terms by mass parameter and evaluate bundles.
        let mut bundles: Vec<(f64, LognormalKernelBundle)> = Vec::with_capacity(2);
        let mut overall_mode = IntegratedExpectationMode::ExactClosedForm;
        for term in terms {
            if !bundles.iter().any(|(m, _)| (*m - term.m).abs() < 1e-300) {
                let b = kernel_bundle(quadctx, term.m, mu, sigma, max_k_needed)?;
                overall_mode = worst_mode(overall_mode, b.mode);
                bundles.push((term.m, b));
            }
        }

        let get_bundle = |m: f64| -> &LognormalKernelBundle {
            &bundles
                .iter()
                .find(|(bm, _)| (*bm - m).abs() < 1e-300)
                .unwrap()
                .1
        };

        // Accumulate S = Σ a_j K_{k_j} and its μ-derivatives.
        let mut s0 = 0.0;
        let mut s1 = 0.0;
        let mut s2 = 0.0;
        let mut s3 = 0.0;

        for term in terms {
            let bundle = get_bundle(term.m);
            let jet = kernel_mu_jet(bundle, term.k, term.m, 3);
            s0 += term.coeff * jet[0];
            s1 += term.coeff * jet[1];
            s2 += term.coeff * jet[2];
            s3 += term.coeff * jet[3];
        }

        let s_safe = s0.max(LOG_FLOOR);
        let log_s = s_safe.ln();
        let r1 = s1 / s_safe;
        let r2 = s2 / s_safe;
        let r3 = s3 / s_safe;

        Ok(Self {
            value: log_s,
            d1: r1,
            d2: r2 - r1 * r1,
            d3: r3 - 3.0 * r1 * r2 + 2.0 * r1 * r1 * r1,
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

/// Compiled sufficient statistics for one row of a latent survival model.
///
/// These are produced by upstream data compilation (not the fitting engine)
/// and represent the integrated nuisance structure of the observation.
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
    /// Log baseline hazard at event time (for exact events only).
    pub log_baseline_hazard: f64,
    /// Unloaded (background) cumulative mass at entry (0 for full loading).
    pub mass_unloaded_entry: f64,
    /// Unloaded (background) cumulative mass at exit.
    pub mass_unloaded_exit: f64,
    /// Loaded instantaneous hazard at event time (for exact events).
    pub hazard_loaded: f64,
    /// Unloaded instantaneous hazard at event time (for exact events).
    pub hazard_unloaded: f64,
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
}

impl LatentSurvivalRowJet {
    pub fn evaluate(
        quadctx: &QuadratureContext,
        row: &LatentSurvivalRow,
        mu: f64,
        sigma: f64,
    ) -> Result<Self, EstimationError> {
        match row.event_type {
            LatentSurvivalEventType::RightCensored => Self::right_censored(quadctx, mu, sigma, row),
            LatentSurvivalEventType::ExactEvent => Self::exact_event(quadctx, mu, sigma, row),
            LatentSurvivalEventType::IntervalCensored => Self::interval_censored(
                quadctx,
                mu,
                sigma,
                row.mass_entry,
                row.mass_left,
                row.mass_right,
            ),
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
            })
        } else {
            Ok(Self {
                log_lik: unloaded_offset + num.value,
                score: num.d1,
                neg_hessian: -num.d2,
                d3: num.d3,
            })
        }
    }

    /// Exact event with loaded/unloaded hazard decomposition.
    ///
    /// When `hazard_unloaded > 0` (Gompertz-Makeham style):
    ///   `ℓ = log(h_U · K_{0,M_L} + h_L · K_{1,M_L}) - M_U_event + M_U_entry - log K_{0,M_L_entry}`
    ///
    /// When `hazard_unloaded == 0` (full loading, the common case):
    ///   `ℓ = log(h_0) + log K_{1,M} - log K_{0,M_entry}`
    ///   which is the original formula with `log_baseline_hazard = log(h_L)`.
    fn exact_event(
        quadctx: &QuadratureContext,
        mu: f64,
        sigma: f64,
        row: &LatentSurvivalRow,
    ) -> Result<Self, EstimationError> {
        let has_unloaded_hazard = row.hazard_unloaded.abs() > 1e-300;

        let unloaded_offset =
            if row.mass_unloaded_exit.abs() > 1e-300 || row.mass_unloaded_entry.abs() > 1e-300 {
                -row.mass_unloaded_exit + row.mass_unloaded_entry
            } else {
                0.0
            };

        if has_unloaded_hazard {
            // Two-term numerator: h_U · K_{0,M_L} + h_L · K_{1,M_L}
            let mass_event_loaded = row.mass_exit;
            let terms = [
                KernelSumTerm {
                    coeff: row.hazard_unloaded,
                    k: 0,
                    m: mass_event_loaded,
                },
                KernelSumTerm {
                    coeff: row.hazard_loaded,
                    k: 1,
                    m: mass_event_loaded,
                },
            ];
            let num = LogKernelSumJet::evaluate(quadctx, &terms, mu, sigma)?;

            if row.mass_entry > 1e-300 {
                let den = LogKernelSumJet::single_term(quadctx, 0, row.mass_entry, mu, sigma)?;
                Ok(Self {
                    log_lik: unloaded_offset + num.value - den.value,
                    score: num.d1 - den.d1,
                    neg_hessian: -(num.d2 - den.d2),
                    d3: num.d3 - den.d3,
                })
            } else {
                Ok(Self {
                    log_lik: unloaded_offset + num.value,
                    score: num.d1,
                    neg_hessian: -num.d2,
                    d3: num.d3,
                })
            }
        } else {
            // Full-loading path: original formula
            let num = LogKernelSumJet::single_term(quadctx, 1, row.mass_exit, mu, sigma)?;
            if row.mass_entry > 1e-300 {
                let den = LogKernelSumJet::single_term(quadctx, 0, row.mass_entry, mu, sigma)?;
                Ok(Self {
                    log_lik: unloaded_offset + row.log_baseline_hazard + num.value - den.value,
                    score: num.d1 - den.d1,
                    neg_hessian: -(num.d2 - den.d2),
                    d3: num.d3 - den.d3,
                })
            } else {
                Ok(Self {
                    log_lik: unloaded_offset + row.log_baseline_hazard + num.value,
                    score: num.d1,
                    neg_hessian: -num.d2,
                    d3: num.d3,
                })
            }
        }
    }

    /// Interval event: `ℓ = log(K_{0,M_L} − K_{0,M_R}) − log K_{0,M_in}`.
    fn interval_censored(
        quadctx: &QuadratureContext,
        mu: f64,
        sigma: f64,
        mass_entry: f64,
        mass_left: f64,
        mass_right: f64,
    ) -> Result<Self, EstimationError> {
        let num_terms = [
            KernelSumTerm {
                coeff: 1.0,
                k: 0,
                m: mass_left,
            },
            KernelSumTerm {
                coeff: -1.0,
                k: 0,
                m: mass_right,
            },
        ];
        let num = LogKernelSumJet::evaluate(quadctx, &num_terms, mu, sigma)?;

        if mass_entry > 1e-300 {
            let den = LogKernelSumJet::single_term(quadctx, 0, mass_entry, mu, sigma)?;
            Ok(Self {
                log_lik: num.value - den.value,
                score: num.d1 - den.d1,
                neg_hessian: -(num.d2 - den.d2),
                d3: num.d3 - den.d3,
            })
        } else {
            Ok(Self {
                log_lik: num.value,
                score: num.d1,
                neg_hessian: -num.d2,
                d3: num.d3,
            })
        }
    }
}

// ─── 2-block row jet for learnable σ ─────────────────────────────────────────

/// 2-block row jet: μ-block and t = log(σ) block.
///
/// Heat-equation identities (t = log σ):
///   ∂_t ℓ = σ² · ∂²ℓ/∂μ²
///   ∂²_t ℓ = 2σ² · ∂²ℓ/∂μ² + σ⁴ · ∂⁴ℓ/∂μ⁴
#[derive(Clone, Copy, Debug)]
pub struct RowJet2Block {
    pub log_lik: f64,
    pub score_mu: f64,
    pub neg_hessian_mu: f64,
    pub score_t: f64,
    pub neg_hessian_t: f64,
}

impl RowJet2Block {
    pub fn from_mu_jet(
        log_lik: f64,
        score_mu: f64,
        d2_ll_mu: f64,
        d4_ll_mu: f64,
        sigma: f64,
    ) -> Self {
        let s2 = sigma * sigma;
        let s4 = s2 * s2;
        Self {
            log_lik,
            score_mu,
            neg_hessian_mu: -d2_ll_mu,
            score_t: s2 * d2_ll_mu,
            neg_hessian_t: -(2.0 * s2 * d2_ll_mu + s4 * d4_ll_mu),
        }
    }

    pub fn latent_survival(
        quadctx: &QuadratureContext,
        row: &LatentSurvivalRow,
        mu: f64,
        sigma: f64,
    ) -> Result<Self, EstimationError> {
        let jet1 = LatentSurvivalRowJet::evaluate(quadctx, row, mu, sigma)?;
        let d2_ll_mu = -jet1.neg_hessian;
        let d4_ll_mu = survival_row_d4_ll(quadctx, row, mu, sigma)?;
        Ok(Self::from_mu_jet(
            jet1.log_lik,
            jet1.score,
            d2_ll_mu,
            d4_ll_mu,
            sigma,
        ))
    }
}

#[inline]
fn log_derivative_4th(r1: f64, r2: f64, r3: f64, r4: f64) -> f64 {
    r4 - 4.0 * r1 * r3 - 3.0 * r2 * r2 + 12.0 * r1 * r1 * r2 - 6.0 * r1 * r1 * r1 * r1
}

fn single_term_d4_log(bundle: &LognormalKernelBundle, k: usize, m: f64) -> f64 {
    let jet = kernel_mu_jet(bundle, k, m, 4);
    let val = jet[0].max(LOG_FLOOR);
    log_derivative_4th(jet[1] / val, jet[2] / val, jet[3] / val, jet[4] / val)
}

fn survival_row_d4_ll(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    mu: f64,
    sigma: f64,
) -> Result<f64, EstimationError> {
    let max_k = 5;
    match row.event_type {
        LatentSurvivalEventType::RightCensored => {
            let b = kernel_bundle(quadctx, row.mass_exit, mu, sigma, max_k)?;
            let mut d4 = single_term_d4_log(&b, 0, row.mass_exit);
            if row.mass_entry > 1e-300 {
                let be = kernel_bundle(quadctx, row.mass_entry, mu, sigma, max_k)?;
                d4 -= single_term_d4_log(&be, 0, row.mass_entry);
            }
            Ok(d4)
        }
        LatentSurvivalEventType::ExactEvent => {
            let mk = 1 + 4;
            let b = kernel_bundle(quadctx, row.mass_exit, mu, sigma, mk)?;
            let mut d4 = single_term_d4_log(&b, 1, row.mass_exit);
            if row.mass_entry > 1e-300 {
                let be = kernel_bundle(quadctx, row.mass_entry, mu, sigma, max_k)?;
                d4 -= single_term_d4_log(&be, 0, row.mass_entry);
            }
            Ok(d4)
        }
        LatentSurvivalEventType::IntervalCensored => {
            let bl = kernel_bundle(quadctx, row.mass_left, mu, sigma, max_k)?;
            let br = kernel_bundle(quadctx, row.mass_right, mu, sigma, max_k)?;
            let terms = [
                KernelSumTerm {
                    coeff: 1.0,
                    k: 0,
                    m: row.mass_left,
                },
                KernelSumTerm {
                    coeff: -1.0,
                    k: 0,
                    m: row.mass_right,
                },
            ];
            // Accumulate S⁽⁰⁾..S⁽⁴⁾ over the two terms.
            let mut s = [0.0; 5];
            for term in &terms {
                let bnd = if (term.m - row.mass_left).abs() < 1e-300 {
                    &bl
                } else {
                    &br
                };
                let jet = kernel_mu_jet(bnd, term.k, term.m, 4);
                for r in 0..5 {
                    s[r] += term.coeff * jet[r];
                }
            }
            let sv = s[0].max(LOG_FLOOR);
            let mut d4 = log_derivative_4th(s[1] / sv, s[2] / sv, s[3] / sv, s[4] / sv);
            if row.mass_entry > 1e-300 {
                let be = kernel_bundle(quadctx, row.mass_entry, mu, sigma, max_k)?;
                d4 -= single_term_d4_log(&be, 0, row.mass_entry);
            }
            Ok(d4)
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
    fn kernel_mu_jet_d1_fd_check() {
        let ctx = QuadratureContext::new();
        let mu = 0.3;
        let sigma = 0.5;
        let m = 1.0;
        let k = 0usize;
        let h = 1e-5;

        let bundle = kernel_bundle(&ctx, m, mu, sigma, 6).unwrap();
        let jet = kernel_mu_jet(&bundle, k, m, 2);

        let kp = kernel_term(&ctx, k, m, mu + h, sigma).unwrap().0;
        let km = kernel_term(&ctx, k, m, mu - h, sigma).unwrap().0;
        let fd_d1 = (kp - km) / (2.0 * h);
        assert!(
            (jet[1] - fd_d1).abs() / fd_d1.abs().max(1e-15) < 1e-4,
            "d1: jet={}, fd={fd_d1}",
            jet[1]
        );

        let kc = kernel_term(&ctx, k, m, mu, sigma).unwrap().0;
        let fd_d2 = (kp - 2.0 * kc + km) / (h * h);
        assert!(
            (jet[2] - fd_d2).abs() / fd_d2.abs().max(1e-15) < 1e-3,
            "d2: jet={}, fd={fd_d2}",
            jet[2]
        );
    }

    #[test]
    fn survival_right_censored_score_fd() {
        let ctx = QuadratureContext::new();
        let mu = -0.5;
        let sigma = 0.3;
        let h = 1e-6;
        let row = LatentSurvivalRow {
            event_type: LatentSurvivalEventType::RightCensored,
            mass_entry: 0.0,
            mass_exit: 2.0,
            mass_left: 0.0,
            mass_right: 0.0,
            log_baseline_hazard: 0.0,
            mass_unloaded_entry: 0.0,
            mass_unloaded_exit: 0.0,
            hazard_loaded: 0.0,
            hazard_unloaded: 0.0,
        };
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
        let row = LatentSurvivalRow {
            event_type: LatentSurvivalEventType::ExactEvent,
            mass_entry: 0.0,
            mass_exit: 1.5,
            mass_left: 0.0,
            mass_right: 0.0,
            log_baseline_hazard: -0.3,
            mass_unloaded_entry: 0.0,
            mass_unloaded_exit: 0.0,
            hazard_loaded: 0.0,
            hazard_unloaded: 0.0,
        };
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
        let row = LatentSurvivalRow {
            event_type: LatentSurvivalEventType::IntervalCensored,
            mass_entry: 0.0,
            mass_exit: 0.0,
            mass_left: 1.0,
            mass_right: 2.0,
            log_baseline_hazard: 0.0,
            mass_unloaded_entry: 0.0,
            mass_unloaded_exit: 0.0,
            hazard_loaded: 0.0,
            hazard_unloaded: 0.0,
        };
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
        let val_p = kernel_term(&ctx, k, m, mu + h, sigma).unwrap().0.ln();
        let val_m = kernel_term(&ctx, k, m, mu - h, sigma).unwrap().0.ln();
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
    fn latent_cloglog_jet_curvature_envelope_matches_scalar_backend_differences() {
        const CURVATURE_ABS_EPS: f64 = 2e-5;
        const CURVATURE_REL_EPS: f64 = 2e-3;
        const THIRD_ABS_EPS: f64 = 8e-5;
        const THIRD_REL_EPS: f64 = 8e-3;

        let ctx = QuadratureContext::new();
        let cases = [(-4.0, 0.15), (-1.2, 0.35), (0.4, 0.6), (1.3, 0.9)];
        let h = 2e-4;

        for (eta, sigma) in cases {
            let jet = latent_cloglog_jet5(&ctx, eta, sigma).expect("latent jet");
            let dm2 = crate::quadrature::cloglog_posterior_meanwith_deriv_controlled(
                &ctx,
                eta - 2.0 * h,
                sigma,
            );
            let dm1 = crate::quadrature::cloglog_posterior_meanwith_deriv_controlled(
                &ctx,
                eta - h,
                sigma,
            );
            let d0 =
                crate::quadrature::cloglog_posterior_meanwith_deriv_controlled(&ctx, eta, sigma);
            let dp1 = crate::quadrature::cloglog_posterior_meanwith_deriv_controlled(
                &ctx,
                eta + h,
                sigma,
            );
            let dp2 = crate::quadrature::cloglog_posterior_meanwith_deriv_controlled(
                &ctx,
                eta + 2.0 * h,
                sigma,
            );

            let d2fd = (dm2.dmean_dmu - 8.0 * dm1.dmean_dmu + 8.0 * dp1.dmean_dmu - dp2.dmean_dmu)
                / (12.0 * h);
            let d3fd = (-dp2.dmean_dmu + 16.0 * dp1.dmean_dmu - 30.0 * d0.dmean_dmu
                + 16.0 * dm1.dmean_dmu
                - dm2.dmean_dmu)
                / (12.0 * h * h);

            let d2_err = (jet.d2 - d2fd).abs();
            let d3_err = (jet.d3 - d3fd).abs();

            assert!(
                d2_err <= CURVATURE_ABS_EPS.max(CURVATURE_REL_EPS * d2fd.abs()),
                "latent cloglog curvature envelope failed at eta={eta}, sigma={sigma}: analytic={} fd={}",
                jet.d2,
                d2fd
            );
            assert!(
                d3_err <= THIRD_ABS_EPS.max(THIRD_REL_EPS * d3fd.abs()),
                "latent cloglog third-derivative envelope failed at eta={eta}, sigma={sigma}: analytic={} fd={}",
                jet.d3,
                d3fd
            );
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
