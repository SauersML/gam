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
    lognormal_laplace_term_shared,
};
use std::fmt;

// ─── Frailty specification ───────────────────────────────────────────────────

/// How the hazard multiplier frailty loads onto the hazard components.
#[derive(Clone, Copy, Debug, PartialEq)]
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
#[derive(Clone, Debug)]
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
    /// Whether this spec includes any frailty.
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::None)
    }

    /// The fixed σ value, if any.
    pub fn fixed_sigma(&self) -> Option<f64> {
        match self {
            Self::None => None,
            Self::GaussianShift { sigma_fixed } => *sigma_fixed,
            Self::HazardMultiplier { sigma_fixed, .. } => *sigma_fixed,
        }
    }

    /// Whether σ is a learnable parameter (needs its own block).
    pub fn sigma_is_learnable(&self) -> bool {
        match self {
            Self::None => false,
            Self::GaussianShift { sigma_fixed } => sigma_fixed.is_none(),
            Self::HazardMultiplier { sigma_fixed, .. } => sigma_fixed.is_none(),
        }
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
    /// α = σ²/(1+σ²)
    pub alpha: f64,
    /// τ = √(1+σ²) = 1/s
    pub tau: f64,
}

impl ProbitFrailtyScale {
    pub fn new(sigma: f64) -> Self {
        let sigma2 = sigma * sigma;
        let tau = (1.0 + sigma2).sqrt();
        Self {
            s: 1.0 / tau,
            alpha: sigma2 / (1.0 + sigma2),
            tau,
        }
    }

    /// ∂_t s where t = log(σ)
    pub fn ds_dt(&self) -> f64 {
        -self.alpha * self.s
    }

    /// ∂_{tt} s
    pub fn d2s_dt2(&self) -> f64 {
        let a = self.alpha;
        a * (3.0 * a - 2.0) * self.s
    }

    /// ∂_{ttt} s
    pub fn d3s_dt3(&self) -> f64 {
        let a = self.alpha;
        a * (-15.0 * a * a + 18.0 * a - 4.0) * self.s
    }

    /// ∂_{tttt} s
    pub fn d4s_dt4(&self) -> f64 {
        let a = self.alpha;
        a * (105.0 * a * a * a - 180.0 * a * a + 84.0 * a - 8.0) * self.s
    }

    /// Scale a denested cubic cell's coefficients by s.
    /// Input: [c0, c1, c2, c3].  Output: [s*c0, s*c1, s*c2, s*c3].
    pub fn scale_cell(&self, coeffs: &[f64; 4]) -> [f64; 4] {
        [
            self.s * coeffs[0],
            self.s * coeffs[1],
            self.s * coeffs[2],
            self.s * coeffs[3],
        ]
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
fn worst_mode(a: IntegratedExpectationMode, b: IntegratedExpectationMode) -> IntegratedExpectationMode {
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
    let bundle = kernel_bundle(quadctx, 1.0, eta, sigma.max(0.0), 5)?;
    let a1 = bundle.get(1);
    let a2 = bundle.get(2);
    let a3 = bundle.get(3);
    let a4 = bundle.get(4);
    let a5 = bundle.get(5);
    Ok(LatentCLogLogJet5 {
        mean: (1.0 - bundle.get(0)).clamp(0.0, 1.0),
        d1: a1.max(0.0),
        d2: a1 - a2,
        d3: a1 - 3.0 * a2 + a3,
        d4: a1 - 7.0 * a2 + 6.0 * a3 - a4,
        d5: a1 - 15.0 * a2 + 25.0 * a3 - 10.0 * a4 + a5,
        mode: bundle.mode,
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
        jet[3] = kf * kf * kf * a(0)
            - (3.0 * kf * kf + 3.0 * kf + 1.0) * m * a(1)
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
        jet[4] = k4 * a(0)
            - (4.0 * k3 + 6.0 * k2 + 4.0 * kf + 1.0) * m * a(1)
            + (6.0 * k2 + 12.0 * kf + 7.0) * m2 * a(2)
            - (4.0 * kf + 6.0) * m3 * a(3)
            + m4 * a(4);
    }

    jet
}

// ─── Full partial-derivative jet (μ, t = log σ, m) ───────────────────────────

/// Full partial-derivative jet of K_{k,m}(μ, σ) with respect to all three
/// primary variables: μ, t = log σ, and m.
///
/// This is the object needed when the baseline parameters move the mass m,
/// e.g. in Royston-Parmar or Gompertz-Makeham with learned baseline.
///
/// Returns a struct with all partials needed for exact score and Hessian.
#[derive(Clone, Copy, Debug)]
pub struct KernelFullJet {
    /// K_{k,m}
    pub value: f64,
    /// ∂_μ K
    pub d_mu: f64,
    /// ∂_{μμ} K
    pub d_mu_mu: f64,
    /// ∂_t K = σ² · ∂_{μμ} K
    pub d_t: f64,
    /// ∂_{μt} K = σ² · ∂_{μμμ} K
    pub d_mu_t: f64,
    /// ∂_{tt} K = 2σ² · ∂_{μμ} K + σ⁴ · ∂_{μμμμ} K
    pub d_tt: f64,
    /// ∂_m K = -K_{k+1,m}
    pub d_m: f64,
    /// ∂_{mm} K = K_{k+2,m}
    pub d_mm: f64,
    /// ∂_{μm} K = -(k+1)·K_{k+1,m} + m·K_{k+2,m}
    pub d_mu_m: f64,
    /// ∂_{tm} K = σ²·(-(k+1)²·K_{k+1} + (2k+3)·m·K_{k+2} - m²·K_{k+3})
    pub d_t_m: f64,
}

impl KernelFullJet {
    /// Compute all partials from a kernel bundle.
    ///
    /// The bundle must contain at least k + 5 entries.
    pub fn from_bundle(bundle: &LognormalKernelBundle, k: usize, m: f64, sigma: f64) -> Self {
        let kf = k as f64;
        let a = |r: usize| -> f64 { bundle.get(k + r) };
        let s2 = sigma * sigma;
        let s4 = s2 * s2;

        // μ-derivatives from recurrence
        let value = a(0);
        let d_mu = kf * a(0) - m * a(1);
        let d_mu_mu = kf * kf * a(0) - (2.0 * kf + 1.0) * m * a(1) + m * m * a(2);
        let d_mu_mu_mu = kf * kf * kf * a(0)
            - (3.0 * kf * kf + 3.0 * kf + 1.0) * m * a(1)
            + 3.0 * (kf + 1.0) * m * m * a(2)
            - m * m * m * a(3);
        let k2 = kf * kf;
        let k3 = k2 * kf;
        let k4 = k3 * kf;
        let m2 = m * m;
        let m3 = m2 * m;
        let m4 = m3 * m;
        let d_mu4 = k4 * a(0)
            - (4.0 * k3 + 6.0 * k2 + 4.0 * kf + 1.0) * m * a(1)
            + (6.0 * k2 + 12.0 * kf + 7.0) * m2 * a(2)
            - (4.0 * kf + 6.0) * m3 * a(3)
            + m4 * a(4);

        // t-derivatives via heat equation
        let d_t = s2 * d_mu_mu;
        let d_mu_t = s2 * d_mu_mu_mu;
        let d_tt = 2.0 * s2 * d_mu_mu + s4 * d_mu4;

        // m-derivatives
        let d_m = -a(1);
        let d_mm = a(2);
        let d_mu_m = -(kf + 1.0) * a(1) + m * a(2);

        // ∂_{tm} K = σ²·(-(k+1)²·K_{k+1} + (2k+3)·m·K_{k+2} - m²·K_{k+3})
        let kp1 = kf + 1.0;
        let d_t_m = s2 * (-kp1 * kp1 * a(1) + (2.0 * kf + 3.0) * m * a(2) - m2 * a(3));

        Self {
            value,
            d_mu,
            d_mu_mu,
            d_t,
            d_mu_t,
            d_tt,
            d_m,
            d_mm,
            d_mu_m,
            d_t_m,
        }
    }
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

// ─── LogKernelSumFullJet: full (μ, t, m) log-sum derivatives ─────────────────

/// Full log-kernel-sum jet with all (μ, t, m) partial derivatives.
///
/// Used when the mass m depends on baseline parameters that are being learned,
/// e.g. Royston-Parmar spline coefficients or Gompertz-Makeham time block.
///
/// If `S = Σ a_j K_{k_j, m_j}(μ, σ)` then `ℓ = log(S)` and we store all
/// first and second partial derivatives of ℓ with respect to μ, t = log σ,
/// and m.
#[derive(Clone, Copy, Debug)]
pub struct LogKernelSumFullJet {
    /// log(S)
    pub value: f64,
    /// ∂_μ log(S)
    pub d_mu: f64,
    /// ∂_t log(S)
    pub d_t: f64,
    /// ∂_m log(S)
    pub d_m: f64,
    /// ∂²_{μμ} log(S)
    pub d_mu_mu: f64,
    /// ∂²_{μt} log(S)
    pub d_mu_t: f64,
    /// ∂²_{tt} log(S)
    pub d_tt: f64,
    /// ∂²_{μm} log(S)
    pub d_mu_m: f64,
    /// ∂²_{tm} log(S)
    pub d_t_m: f64,
    /// ∂²_{mm} log(S)
    pub d_mm: f64,
    pub mode: IntegratedExpectationMode,
}

impl LogKernelSumFullJet {
    /// Evaluate for a single positive kernel term (fast path).
    ///
    /// Computes `log(K_{k,m})` and all first/second partial derivatives with
    /// respect to (μ, t = log σ, m) from the exact `KernelFullJet` machinery.
    pub fn single_term(
        quadctx: &QuadratureContext,
        k: usize,
        m: f64,
        mu: f64,
        sigma: f64,
    ) -> Result<Self, EstimationError> {
        let max_k = k + 5;
        let bundle = kernel_bundle(quadctx, m, mu, sigma, max_k)?;
        let jet = KernelFullJet::from_bundle(&bundle, k, m, sigma);

        let s = jet.value.max(LOG_FLOOR);
        let log_s = s.ln();

        // Score ratios r_x = S_x / S
        let r_mu = jet.d_mu / s;
        let r_t = jet.d_t / s;
        let r_m = jet.d_m / s;

        Ok(Self {
            value: log_s,
            d_mu: r_mu,
            d_t: r_t,
            d_m: r_m,
            d_mu_mu: jet.d_mu_mu / s - r_mu * r_mu,
            d_mu_t: jet.d_mu_t / s - r_mu * r_t,
            d_tt: jet.d_tt / s - r_t * r_t,
            d_mu_m: jet.d_mu_m / s - r_mu * r_m,
            d_t_m: jet.d_t_m / s - r_t * r_m,
            d_mm: jet.d_mm / s - r_m * r_m,
            mode: bundle.mode,
        })
    }

    /// Evaluate `log(Σ a_j K_j)` and all (μ, t, m) partials for a signed sum.
    ///
    /// All terms share the same (μ, σ) and mass m.  For multi-mass sums each
    /// term carries its own `KernelSumTerm::m`, but the m-derivatives are with
    /// respect to a *shared* perturbation δm applied to every term (chain rule
    /// by the caller when masses differ).
    pub fn evaluate(
        quadctx: &QuadratureContext,
        terms: &[KernelSumTerm],
        mu: f64,
        sigma: f64,
    ) -> Result<Self, EstimationError> {
        assert!(!terms.is_empty(), "LogKernelSumFullJet requires at least one term");

        // Max k across all terms, plus headroom for full-jet recurrence.
        let max_k_needed = terms.iter().map(|t| t.k).max().unwrap_or(0) + 5;

        // Deduplicate bundles by mass.
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

        // Accumulate S and all its raw partial sums.
        let mut s_val = 0.0;
        let mut s_mu = 0.0;
        let mut s_t = 0.0;
        let mut s_m = 0.0;
        let mut s_mu_mu = 0.0;
        let mut s_mu_t = 0.0;
        let mut s_tt = 0.0;
        let mut s_mu_m = 0.0;
        let mut s_t_m = 0.0;
        let mut s_mm = 0.0;

        for term in terms {
            let bundle = get_bundle(term.m);
            let jet = KernelFullJet::from_bundle(bundle, term.k, term.m, sigma);
            let c = term.coeff;
            s_val += c * jet.value;
            s_mu += c * jet.d_mu;
            s_t += c * jet.d_t;
            s_m += c * jet.d_m;
            s_mu_mu += c * jet.d_mu_mu;
            s_mu_t += c * jet.d_mu_t;
            s_tt += c * jet.d_tt;
            s_mu_m += c * jet.d_mu_m;
            s_t_m += c * jet.d_t_m;
            s_mm += c * jet.d_mm;
        }

        let s = s_val.max(LOG_FLOOR);
        let log_s = s.ln();

        let r_mu = s_mu / s;
        let r_t = s_t / s;
        let r_m = s_m / s;

        Ok(Self {
            value: log_s,
            d_mu: r_mu,
            d_t: r_t,
            d_m: r_m,
            d_mu_mu: s_mu_mu / s - r_mu * r_mu,
            d_mu_t: s_mu_t / s - r_mu * r_t,
            d_tt: s_tt / s - r_t * r_t,
            d_mu_m: s_mu_m / s - r_mu * r_m,
            d_t_m: s_t_m / s - r_t * r_m,
            d_mm: s_mm / s - r_m * r_m,
            mode: overall_mode,
        })
    }
}

// ─── Row-level binary latent-cloglog likelihood ──────────────────────────────

/// Row-level log-likelihood and μ-derivatives for the binary latent-cloglog model.
///
/// Model:
///   `U_i ~ N(μ_i, σ²)`,
///   `P(y=1 | U, m) = 1 − exp(−m·exp(U))`
///
/// Marginal:
///   `p = 1 − K_{0,m}(μ, σ)`
///
/// Row log-likelihood:
///   `ℓ = y·log(p) + (1−y)·log(1−p)`
#[derive(Clone, Copy, Debug)]
pub struct BinaryCloglogRowJet {
    pub log_lik: f64,
    pub score: f64,
    pub neg_hessian: f64,
    pub d3: f64,
}

impl BinaryCloglogRowJet {
    /// Evaluate the row jet for the binary latent-cloglog model.
    ///
    /// - `y`: binary response (0 or 1).
    /// - `eta`: linear predictor value (without offset).
    /// - `log_mass`: log of the exposure mass (added as offset to η).
    /// - `sigma`: latent standard deviation (≥ 0).
    pub fn evaluate(
        quadctx: &QuadratureContext,
        y: f64,
        eta: f64,
        log_mass: f64,
        sigma: f64,
    ) -> Result<Self, EstimationError> {
        let alpha = eta + log_mass;
        let jet = latent_cloglog_jet5(quadctx, alpha, sigma)?;

        let p = jet.mean.clamp(LOG_FLOOR, 1.0 - LOG_FLOOR);
        let q = 1.0 - p;

        let log_lik = if y > 0.5 { p.ln() } else { q.ln() };
        let dp = jet.d1.max(0.0);
        let ddp = jet.d2;
        let dddp = jet.d3;

        // Score: dℓ/dμ via chain rule on y·log(p) + (1−y)·log(q).
        let w = if y > 0.5 { 1.0 / p } else { -1.0 / q };
        let score = w * dp;

        // Second derivative:
        // d²ℓ/dμ² = w·p'' + dw/dμ·p'
        // where dw/dμ = −y·p'/p² − (1−y)·p'/q²
        let dw = -y * dp / (p * p) - (1.0 - y) * dp / (q * q);
        let d2_ll = w * ddp + dw * dp;

        // Third derivative via d³(log f)/dμ³:
        let r1_p = dp / p;
        let r2_p = ddp / p;
        let r3_p = dddp / p;
        let d3_log_p = r3_p - 3.0 * r1_p * r2_p + 2.0 * r1_p * r1_p * r1_p;

        let r1_q = -dp / q;
        let r2_q = -ddp / q;
        let r3_q = -dddp / q;
        let d3_log_q = r3_q - 3.0 * r1_q * r2_q + 2.0 * r1_q * r1_q * r1_q;

        let d3_ll = y * d3_log_p + (1.0 - y) * d3_log_q;

        Ok(Self {
            log_lik,
            score,
            neg_hessian: -d2_ll,
            d3: d3_ll,
        })
    }
}

// ─── Generic 2-block row jet for learnable σ ─────────────────────────────────

/// Generic 2-block row jet: μ-block and t = log(σ) block.
///
/// Any latent family that depends on (μ, σ) through Gaussian smoothing can
/// use this struct.  The σ-derivatives are obtained from the heat-equation
/// identities applied to the μ-jet of the row log-likelihood, so the
/// underlying row kernel does not need to know about σ at all.
///
/// Heat-equation identities (t = log σ):
///   `∂_t ℓ = σ² · ∂²ℓ/∂μ²`
///   `∂²_t ℓ = 2σ² · ∂²ℓ/∂μ² + σ⁴ · ∂⁴ℓ/∂μ⁴`
#[derive(Clone, Copy, Debug)]
pub struct RowJet2Block {
    pub log_lik: f64,
    /// dℓ/dμ
    pub score_mu: f64,
    /// −d²ℓ/dμ² (observed curvature for μ block)
    pub neg_hessian_mu: f64,
    /// dℓ/dt where t = log σ
    pub score_t: f64,
    /// −d²ℓ/dt² (observed curvature for t block)
    pub neg_hessian_t: f64,
}

impl RowJet2Block {
    /// Construct from μ-derivatives of the row log-likelihood and σ.
    ///
    /// `d2_ll_mu`: d²ℓ/dμ² (negative of neg_hessian)
    /// `d4_ll_mu`: d⁴ℓ/dμ⁴
    ///
    /// The heat-equation identities give:
    ///   score_t = σ² · d²ℓ/dμ²
    ///   d²ℓ/dt² = 2σ² · d²ℓ/dμ² + σ⁴ · d⁴ℓ/dμ⁴
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

    /// Evaluate for a binary latent-cloglog row with learnable σ.
    pub fn binary_cloglog(
        quadctx: &QuadratureContext,
        y: f64,
        eta: f64,
        log_mass: f64,
        sigma: f64,
    ) -> Result<Self, EstimationError> {
        let alpha = eta + log_mass;
        let jet = latent_cloglog_jet5(quadctx, alpha, sigma)?;

        let p = jet.mean.clamp(LOG_FLOOR, 1.0 - LOG_FLOOR);
        let q = 1.0 - p;

        let log_lik = if y > 0.5 { p.ln() } else { q.ln() };
        let dp = jet.d1.max(0.0);
        let ddp = jet.d2;
        let dddp = jet.d3;
        let ddddp = jet.d4;

        // μ-derivatives of log-likelihood
        let w = if y > 0.5 { 1.0 / p } else { -1.0 / q };
        let score_mu = w * dp;
        let dw = -y * dp / (p * p) - (1.0 - y) * dp / (q * q);
        let d2_ll_mu = w * ddp + dw * dp;

        // 4th log-derivative formula for d⁴ℓ/dμ⁴
        let (d4_log_p, d4_log_q) = log_4th_derivative_pair(dp, ddp, dddp, ddddp, p, q);
        let d4_ll_mu = y * d4_log_p + (1.0 - y) * d4_log_q;

        Ok(Self::from_mu_jet(log_lik, score_mu, d2_ll_mu, d4_ll_mu, sigma))
    }

    /// Evaluate for a latent survival row with learnable σ.
    ///
    /// Uses exact 4th-order kernel recurrences for d⁴ℓ/dμ⁴ via
    /// [`survival_row_d4_ll`], combined with the standard 3rd-order jet
    /// for score and Hessian.
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

/// Helper: compute d⁴(log f)/dμ⁴ for both f = p and f = q = 1-p.
///
/// d⁴(log f)/dμ⁴ = f''''/f − 4(f'/f)(f'''/f) − 3(f''/f)² + 12(f'/f)²(f''/f) − 6(f'/f)⁴
fn log_4th_derivative_pair(dp: f64, ddp: f64, dddp: f64, ddddp: f64, p: f64, q: f64) -> (f64, f64) {
    let r1_p = dp / p;
    let r2_p = ddp / p;
    let r3_p = dddp / p;
    let r4_p = ddddp / p;
    let d4_log_p = r4_p - 4.0 * r1_p * r3_p - 3.0 * r2_p * r2_p
        + 12.0 * r1_p * r1_p * r2_p
        - 6.0 * r1_p * r1_p * r1_p * r1_p;

    let r1_q = -dp / q;
    let r2_q = -ddp / q;
    let r3_q = -dddp / q;
    let r4_q = -ddddp / q;
    let d4_log_q = r4_q - 4.0 * r1_q * r3_q - 3.0 * r2_q * r2_q
        + 12.0 * r1_q * r1_q * r2_q
        - 6.0 * r1_q * r1_q * r1_q * r1_q;

    (d4_log_p, d4_log_q)
}

/// Compute d⁴(log f)/dμ⁴ from the ratios rₙ = f⁽ⁿ⁾/f.
///
/// Formula: r₄ − 4·r₁·r₃ − 3·r₂² + 12·r₁²·r₂ − 6·r₁⁴
#[inline]
fn log_derivative_4th(r1: f64, r2: f64, r3: f64, r4: f64) -> f64 {
    r4 - 4.0 * r1 * r3 - 3.0 * r2 * r2 + 12.0 * r1 * r1 * r2
        - 6.0 * r1 * r1 * r1 * r1
}

/// Compute d⁴ℓ/dμ⁴ for a single positive kernel term log K_{k,m}.
///
/// Uses exact 4th-order μ-recurrences via [`kernel_mu_jet`].
fn single_term_d4_log(bundle: &LognormalKernelBundle, k: usize, m: f64) -> f64 {
    let jet = kernel_mu_jet(bundle, k, m, 4);
    let val = jet[0].max(LOG_FLOOR);
    let r1 = jet[1] / val;
    let r2 = jet[2] / val;
    let r3 = jet[3] / val;
    let r4 = jet[4] / val;
    log_derivative_4th(r1, r2, r3, r4)
}

/// Compute d⁴(log S)/dμ⁴ for a signed sum S = Σ aⱼ K_{k_j, m_j}.
///
/// Accumulates S and its first four μ-derivatives from kernel recurrences,
/// then applies the 4th log-derivative formula.
fn kernel_sum_d4_log(
    bundles: &[(f64, &LognormalKernelBundle)],
    terms: &[KernelSumTerm],
) -> f64 {
    let get_bundle = |m: f64| -> &LognormalKernelBundle {
        bundles
            .iter()
            .find(|(bm, _)| (*bm - m).abs() < 1e-300)
            .unwrap()
            .1
    };

    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2 = 0.0;
    let mut s3 = 0.0;
    let mut s4 = 0.0;

    for term in terms {
        let bundle = get_bundle(term.m);
        let jet = kernel_mu_jet(bundle, term.k, term.m, 4);
        s0 += term.coeff * jet[0];
        s1 += term.coeff * jet[1];
        s2 += term.coeff * jet[2];
        s3 += term.coeff * jet[3];
        s4 += term.coeff * jet[4];
    }

    let s_safe = s0.max(LOG_FLOOR);
    let r1 = s1 / s_safe;
    let r2 = s2 / s_safe;
    let r3 = s3 / s_safe;
    let r4 = s4 / s_safe;
    log_derivative_4th(r1, r2, r3, r4)
}

/// Compute d⁴ℓ/dμ⁴ for a latent survival row using exact kernel recurrences.
///
/// This mirrors the event-type dispatch of [`LatentSurvivalRowJet::evaluate`]
/// but only computes the 4th log-derivative of the log-likelihood.
fn survival_row_d4_ll(
    quadctx: &QuadratureContext,
    row: &LatentSurvivalRow,
    mu: f64,
    sigma: f64,
) -> Result<f64, EstimationError> {
    match row.event_type {
        LatentSurvivalEventType::RightCensored => {
            // ℓ = log K_{0, M_exit} [- log K_{0, M_entry}]
            // d⁴ℓ/dμ⁴ = d⁴(log K_{0, M_exit})/dμ⁴ [- d⁴(log K_{0, M_entry})/dμ⁴]
            let mass_exit = row.mass_exit;
            let max_k = 0 + 5; // k=0, need up to k+4 in bundle
            let bundle_exit = kernel_bundle(quadctx, mass_exit, mu, sigma, max_k)?;
            let mut d4 = single_term_d4_log(&bundle_exit, 0, mass_exit);

            if row.mass_entry > 1e-300 {
                let bundle_entry =
                    kernel_bundle(quadctx, row.mass_entry, mu, sigma, max_k)?;
                d4 -= single_term_d4_log(&bundle_entry, 0, row.mass_entry);
            }
            Ok(d4)
        }
        LatentSurvivalEventType::ExactEvent => {
            let has_unloaded_hazard = row.hazard_unloaded.abs() > 1e-300;

            if has_unloaded_hazard {
                // Numerator: S = h_U · K_{0,M} + h_L · K_{1,M}
                let mass_event = row.mass_exit;
                let max_k = 1 + 5; // k goes up to 1, need k+4
                let bundle_event =
                    kernel_bundle(quadctx, mass_event, mu, sigma, max_k)?;
                let terms = [
                    KernelSumTerm {
                        coeff: row.hazard_unloaded,
                        k: 0,
                        m: mass_event,
                    },
                    KernelSumTerm {
                        coeff: row.hazard_loaded,
                        k: 1,
                        m: mass_event,
                    },
                ];
                let bundles = [(mass_event, &bundle_event)];
                let mut d4 = kernel_sum_d4_log(&bundles, &terms);

                if row.mass_entry > 1e-300 {
                    let max_k_den = 0 + 5;
                    let bundle_entry =
                        kernel_bundle(quadctx, row.mass_entry, mu, sigma, max_k_den)?;
                    d4 -= single_term_d4_log(&bundle_entry, 0, row.mass_entry);
                }
                Ok(d4)
            } else {
                // Full loading: ℓ = const + log K_{1,M} [- log K_{0,M_entry}]
                let max_k = 1 + 5;
                let bundle_exit =
                    kernel_bundle(quadctx, row.mass_exit, mu, sigma, max_k)?;
                let mut d4 = single_term_d4_log(&bundle_exit, 1, row.mass_exit);

                if row.mass_entry > 1e-300 {
                    let max_k_den = 0 + 5;
                    let bundle_entry =
                        kernel_bundle(quadctx, row.mass_entry, mu, sigma, max_k_den)?;
                    d4 -= single_term_d4_log(&bundle_entry, 0, row.mass_entry);
                }
                Ok(d4)
            }
        }
        LatentSurvivalEventType::IntervalCensored => {
            // ℓ = log(K_{0,M_L} − K_{0,M_R}) [- log K_{0,M_entry}]
            let max_k = 0 + 5;
            let bundle_left =
                kernel_bundle(quadctx, row.mass_left, mu, sigma, max_k)?;
            let bundle_right =
                kernel_bundle(quadctx, row.mass_right, mu, sigma, max_k)?;

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
            let bundles = [
                (row.mass_left, &bundle_left),
                (row.mass_right, &bundle_right),
            ];
            let mut d4 = kernel_sum_d4_log(&bundles, &terms);

            if row.mass_entry > 1e-300 {
                let bundle_entry =
                    kernel_bundle(quadctx, row.mass_entry, mu, sigma, max_k)?;
                d4 -= single_term_d4_log(&bundle_entry, 0, row.mass_entry);
            }
            Ok(d4)
        }
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
            LatentSurvivalEventType::RightCensored => {
                Self::right_censored(quadctx, mu, sigma, row)
            }
            LatentSurvivalEventType::ExactEvent => {
                Self::exact_event(quadctx, mu, sigma, row)
            }
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
        let has_unloaded = row.mass_unloaded_exit.abs() > 1e-300
            || row.mass_unloaded_entry.abs() > 1e-300;

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

        let unloaded_offset = if row.mass_unloaded_exit.abs() > 1e-300
            || row.mass_unloaded_entry.abs() > 1e-300
        {
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
                let den =
                    LogKernelSumJet::single_term(quadctx, 0, row.mass_entry, mu, sigma)?;
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
            let num =
                LogKernelSumJet::single_term(quadctx, 1, row.mass_exit, mu, sigma)?;
            if row.mass_entry > 1e-300 {
                let den =
                    LogKernelSumJet::single_term(quadctx, 0, row.mass_entry, mu, sigma)?;
                Ok(Self {
                    log_lik: unloaded_offset + row.log_baseline_hazard + num.value
                        - den.value,
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

// ─── Prediction semantics ────────────────────────────────────────────────────

/// Prediction mode for latent families.
///
/// The model produces a latent linear predictor η = X·β.  How that predictor
/// is turned into a user-facing prediction depends on the stored reference
/// information.
///
/// This prevents the library from returning an unlabeled "probability" when
/// the reference mass/horizon is unknown.
#[derive(Clone, Debug)]
pub enum LatentPredictionMode {
    /// Raw latent linear predictor η (always available).
    LatentEta,
    /// Relative rate: exp(η).  Interpretable as a hazard ratio relative to
    /// a reference population at η = 0.
    RelativeRate,
    /// Standardized risk at a reference horizon.  Only available when the
    /// model stores a reference mass (or mass grid) so that the absolute
    /// probability can be computed.
    StandardizedRisk {
        /// Reference cumulative mass at the prediction horizon.
        reference_mass: f64,
    },
    /// Full risk curve over an age/mass grid.  Available when the model
    /// stores a baseline mass curve.
    ReferenceCurve {
        /// Grid of evaluation points.
        mass_grid: Vec<f64>,
    },
}

/// Compute predictions from a latent-family model.
///
/// Given the fitted linear predictor η and a prediction mode, returns the
/// appropriate user-facing prediction.
pub fn latent_predict(
    quadctx: &QuadratureContext,
    eta: &[f64],
    sigma: f64,
    mode: &LatentPredictionMode,
) -> Result<Vec<f64>, EstimationError> {
    match mode {
        LatentPredictionMode::LatentEta => Ok(eta.to_vec()),
        LatentPredictionMode::RelativeRate => Ok(eta.iter().map(|&e| e.exp()).collect()),
        LatentPredictionMode::StandardizedRisk { reference_mass } => {
            let log_m = reference_mass.ln();
            eta.iter()
                .map(|&e| {
                    let alpha = e + log_m;
                    let jet = latent_cloglog_jet5(quadctx, alpha, sigma)?;
                    Ok(jet.mean)
                })
                .collect()
        }
        LatentPredictionMode::ReferenceCurve { mass_grid } => {
            if mass_grid.is_empty() {
                return Err(EstimationError::InvalidInput(
                    "ReferenceCurve prediction requires a non-empty mass grid".to_string(),
                ));
            }
            Err(EstimationError::InvalidInput(
                "ReferenceCurve prediction requires a dedicated curve-valued API".to_string(),
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn binary_cloglog_row_jet_score_fd() {
        let ctx = QuadratureContext::new();
        let mu = 0.3;
        let sigma = 0.4;
        let log_mass = 0.5;
        let h = 1e-6;

        for y in [0.0, 1.0] {
            let ll_p = BinaryCloglogRowJet::evaluate(&ctx, y, mu + h, log_mass, sigma)
                .unwrap()
                .log_lik;
            let ll_m = BinaryCloglogRowJet::evaluate(&ctx, y, mu - h, log_mass, sigma)
                .unwrap()
                .log_lik;
            let fd_score = (ll_p - ll_m) / (2.0 * h);
            let jet = BinaryCloglogRowJet::evaluate(&ctx, y, mu, log_mass, sigma).unwrap();
            assert!(
                (jet.score - fd_score).abs() / fd_score.abs().max(1e-15) < 1e-3,
                "y={y}: score={}, fd={fd_score}",
                jet.score
            );
        }
    }

    #[test]
    fn binary_cloglog_row_jet_neg_hessian_fd() {
        let ctx = QuadratureContext::new();
        let mu = -0.2;
        let sigma = 0.3;
        let log_mass = 0.0;
        let h = 1e-5;

        for y in [0.0, 1.0] {
            let ll_p = BinaryCloglogRowJet::evaluate(&ctx, y, mu + h, log_mass, sigma)
                .unwrap()
                .log_lik;
            let ll_c = BinaryCloglogRowJet::evaluate(&ctx, y, mu, log_mass, sigma)
                .unwrap()
                .log_lik;
            let ll_m = BinaryCloglogRowJet::evaluate(&ctx, y, mu - h, log_mass, sigma)
                .unwrap()
                .log_lik;
            let fd_d2 = (ll_p - 2.0 * ll_c + ll_m) / (h * h);
            let jet = BinaryCloglogRowJet::evaluate(&ctx, y, mu, log_mass, sigma).unwrap();
            assert!(
                (jet.neg_hessian - (-fd_d2)).abs() / jet.neg_hessian.abs().max(1e-15) < 1e-3,
                "y={y}: neg_hessian={}, fd_neg_d2={}",
                jet.neg_hessian,
                -fd_d2
            );
        }
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
}
