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
        // where dw/dμ = −y·(p')²/p² − (1−y)·(p')²/q²
        let dw = -y * dp * dp / (p * p) - (1.0 - y) * dp * dp / (q * q);
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
                Self::right_censored(quadctx, mu, sigma, row.mass_entry, row.mass_exit)
            }
            LatentSurvivalEventType::ExactEvent => Self::exact_event(
                quadctx,
                mu,
                sigma,
                row.mass_entry,
                row.mass_exit,
                row.log_baseline_hazard,
            ),
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

    /// Right-censoring: `ℓ = log K_{0,M_out}(μ,σ) − log K_{0,M_in}(μ,σ)`.
    fn right_censored(
        quadctx: &QuadratureContext,
        mu: f64,
        sigma: f64,
        mass_entry: f64,
        mass_exit: f64,
    ) -> Result<Self, EstimationError> {
        let num = LogKernelSumJet::single_term(quadctx, 0, mass_exit, mu, sigma)?;
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

    /// Exact event: `ℓ = log(h_0) + log K_{1,M}(μ,σ) − log K_{0,M_in}(μ,σ)`.
    fn exact_event(
        quadctx: &QuadratureContext,
        mu: f64,
        sigma: f64,
        mass_entry: f64,
        mass_event: f64,
        log_baseline_hazard: f64,
    ) -> Result<Self, EstimationError> {
        let num = LogKernelSumJet::single_term(quadctx, 1, mass_event, mu, sigma)?;
        if mass_entry > 1e-300 {
            let den = LogKernelSumJet::single_term(quadctx, 0, mass_entry, mu, sigma)?;
            Ok(Self {
                log_lik: log_baseline_hazard + num.value - den.value,
                score: num.d1 - den.d1,
                neg_hessian: -(num.d2 - den.d2),
                d3: num.d3 - den.d3,
            })
        } else {
            Ok(Self {
                log_lik: log_baseline_hazard + num.value,
                score: num.d1,
                neg_hessian: -num.d2,
                d3: num.d3,
            })
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
            // Return risk at the first grid point for now; a full curve
            // would need a matrix output which is outside this scalar API.
            if mass_grid.is_empty() {
                return Err(EstimationError::InvalidInput(
                    "ReferenceCurve prediction requires a non-empty mass grid".to_string(),
                ));
            }
            let log_m = mass_grid.last().unwrap().ln();
            eta.iter()
                .map(|&e| {
                    let alpha = e + log_m;
                    let jet = latent_cloglog_jet5(quadctx, alpha, sigma)?;
                    Ok(jet.mean)
                })
                .collect()
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
