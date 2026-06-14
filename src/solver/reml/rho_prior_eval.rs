//! Shared evaluation of the configured deterministic smoothing-parameter (ρ)
//! prior objective.
//!
//! The cost / gradient / Hessian of the configured [`RhoPrior`] is the same
//! math whether it is consumed by the custom-family outer objective or by the
//! REML/LAML runtime; only the *invalid-prior policy* differs between callers.
//! Custom-family handling surfaces malformed priors as hard errors, while the
//! REML runtime folds them into the objective as `+inf` cost (and `NaN`
//! gradient/Hessian) so the outer optimizer steps away from the offending
//! region. This module owns the single source of truth for the prior math and
//! lets each caller pick its policy explicitly via [`InvalidPriorPolicy`].
//!
//! For a single coordinate `r` (a log-precision, `λ = exp(r)`):
//!
//! * `Flat`           — cost 0, grad 0, Hessian 0.
//! * `Normal{mean,sd}`— with `inv_var = 1 / sd²`,
//!   cost `0.5 · (r − mean)² · inv_var`, grad `(r − mean) · inv_var`,
//!   Hessian `inv_var`.
//! * `GammaPrecision{shape,rate}` — Gamma(shape, rate) hyperprior on the
//!   precision `λ = exp(r)`, using the REML/LAML MAP-in-lambda convention and
//!   contributing (up to an additive constant) cost `rate · λ − (shape − 1) · r`,
//!   grad `rate · λ − (shape − 1)`, Hessian `rate · λ`. Samplers over `r`
//!   include the `+r` Jacobian and therefore use the transformed density instead.
//! * `Independent`    — one prior per coordinate; the same per-coordinate math
//!   summed/stacked, with nested `Independent` priors rejected as invalid.
//!
//! The Hessian is diagonal and is returned as `Some` only when at least one
//! diagonal entry is non-zero; an all-zero curvature contribution collapses to
//! `None` so callers can skip adding an empty block.

use crate::types::RhoPrior;
use ndarray::{Array1, Array2};

/// Calibrated exponential rate `θ = −ln(tail_prob) / upper` of a
/// penalized-complexity prior, from the tail statement `P(d > upper) =
/// tail_prob` on the distance scale `d = exp(-ρ/2)`. Caller must validate the
/// hyperparameters first; with `0 < tail_prob < 1` and `upper > 0` the result
/// is finite and strictly positive.
pub(crate) fn pc_prior_rate(upper: f64, tail_prob: f64) -> f64 {
    -tail_prob.ln() / upper
}

/// Negative-log penalized-complexity prior contribution and its first/second
/// derivatives in the log-precision `r`, for calibrated rate `θ`. The objective
/// is minimized, so this returns the cost (up to the ρ-independent additive
/// constant `−ln(θ/2)`), gradient and Hessian:
///
/// ```text
/// cost = r/2 + θ exp(-r/2),  grad = 1/2 − (θ/2) exp(-r/2),  hess = (θ/4) exp(-r/2).
/// ```
///
/// The curvature is strictly positive (`θ > 0`), so the contribution is convex
/// and always supplies usable outer-Hessian information.
pub(crate) fn pc_prior_terms(theta: f64, r: f64) -> (f64, f64, f64) {
    let e = (-0.5 * r).exp();
    (0.5 * r + theta * e, 0.5 - 0.5 * theta * e, 0.25 * theta * e)
}

/// Self-gated, *one-sided* penalized-complexity barrier used as the firth-general
/// DEFAULT outer ρ prior on smoothing coordinates the caller left unset. It walls
/// off the `λ → 0` / `ρ → −∞` under-smoothing degeneracy WITHOUT perturbing a
/// well-identified `λ` — restoring the strict zero-downside guarantee (a clean
/// fit stays byte-identical to plain REML), exactly the way the Jeffreys
/// conditioning gate returns a byte-identical-zero contribution on a clean fit.
///
/// MOTIVATION. The plain PC prior contributes `cost = ρ/2 + θ e^{-ρ/2}` whose
/// gradient `1/2 − (θ/2) e^{-ρ/2}` tends to the CONSTANT `+1/2` as `ρ → +∞`. That
/// residual `O(1)` Occam pull shifts the REML optimum of EVERY identified `λ` by
/// `Δρ ≈ (1/2)/H_reml = O(1/n)` — small, but never byte-zero, so it is a
/// (tiny) downside on every fit including well-conditioned ones. The firth
/// default exists only to remove the `λ → 0` degeneracy, not to bias an
/// identified `λ`, so the default should be EXACTLY flat on the identified side.
///
/// SHAPE. Work on the distance scale `b(ρ) = e^{-ρ/2}` (the marginal-SD scale of
/// the penalized component; `b → ∞` is the under-smoothing degeneracy). Gate at
/// `ρ_gate` with `b_gate = e^{-ρ_gate/2}`:
///   * `ρ ≥ ρ_gate` (identified side, distance `b ≤ b_gate`): cost = grad = hess
///     = 0, BYTE-IDENTICALLY (a clean fit pays nothing — strict zero-downside).
///   * `ρ < ρ_gate` (degenerate side, distance `b > b_gate`): a convex barrier
///     that is C¹ at the gate,
///       cost = θ [ b(ρ) − b_gate − (1/2) b_gate (ρ_gate − ρ) ],
///       grad = θ [ −(1/2) b(ρ) + (1/2) b_gate ]   (= 0 at the gate, < 0 below),
///       hess = (θ/4) b(ρ)  (> 0, always usable curvature).
///     The gradient is negative below the gate (pushing `ρ` UP, away from the
///     `λ → 0` wall) and decays continuously to zero AT the gate, so there is no
///     persistent pull on the identified side.
///
/// GATE CHOICE. `ρ_gate = −2 ln(upper)`, i.e. the barrier engages only once the
/// marginal-SD distance `b = e^{-ρ/2}` exceeds the SAME interpretable `upper`
/// bound that calibrates `θ` (`P(b > upper) = tail_prob`). Any `λ` whose
/// distance scale is within the plausible identified range `b ≤ upper` is left
/// exactly untouched; the wall only asserts itself in the genuinely
/// under-smoothed tail the prior was introduced to bound.
pub(crate) fn firth_default_barrier_terms(theta: f64, upper: f64, r: f64) -> (f64, f64, f64) {
    let rho_gate = -2.0 * upper.ln();
    if r >= rho_gate {
        // Identified side: byte-identically flat (strict zero-downside).
        return (0.0, 0.0, 0.0);
    }
    let b = (-0.5 * r).exp();
    let b_gate = (-0.5 * rho_gate).exp(); // == upper, but kept symbolic for clarity.
    let cost = theta * (b - b_gate - 0.5 * b_gate * (rho_gate - r));
    let grad = theta * (-0.5 * b + 0.5 * b_gate);
    let hess = 0.25 * theta * b;
    (cost, grad, hess)
}

/// What a caller wants done when the configured prior is malformed (e.g. a
/// `Normal` with non-positive `sd`, a `GammaPrecision` with non-positive
/// `shape`, an `Independent` whose length disagrees with `ρ`, or a nested
/// `Independent`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InvalidPriorPolicy {
    /// Return a descriptive [`RhoPriorError`]. Used by custom-family handling,
    /// which validates configuration up front and reports problems precisely.
    HardError,
    /// Saturate the contribution so the outer optimizer is repelled: cost is
    /// `+inf` and gradient/Hessian entries are `NaN`. Used by the REML/LAML
    /// runtime, where the prior is one additive term inside a search.
    Saturate,
}

/// Structured malformed-prior diagnostic produced under
/// [`InvalidPriorPolicy::HardError`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum RhoPriorError {
    /// A length / shape disagreement (e.g. an `Independent` prior whose number
    /// of coordinates does not match `ρ`).
    DimensionMismatch { reason: String },
    /// A structurally invalid prior, such as a nested `Independent` where a
    /// scalar prior is required.
    ConstraintViolation { reason: String },
}

impl RhoPriorError {
    pub(crate) fn dimension_mismatch(reason: String) -> Self {
        RhoPriorError::DimensionMismatch { reason }
    }

    pub(crate) fn constraint_violation(reason: String) -> Self {
        RhoPriorError::ConstraintViolation { reason }
    }
}

/// Cost, gradient, and (optional) diagonal Hessian of the configured ρ prior.
#[derive(Debug, Clone)]
pub(crate) struct RhoPriorEval {
    pub cost: f64,
    pub gradient: Array1<f64>,
    /// Diagonal Hessian, `Some` only when at least one entry is non-zero.
    pub hessian: Option<Array2<f64>>,
}

/// Per-coordinate scalar contribution `(cost, grad, hess)` of one *scalar*
/// prior at `r`, or a structured error when the scalar prior is malformed.
/// Nested `Independent` priors are not scalar and are rejected here.
pub(crate) fn scalar_terms(prior: &RhoPrior, r: f64, context: &str) -> Result<(f64, f64, f64), RhoPriorError> {
    match prior {
        RhoPrior::Flat => Ok((0.0, 0.0, 0.0)),
        RhoPrior::Normal { mean, sd } => {
            if !mean.is_finite() || !sd.is_finite() || *sd <= 0.0 {
                return Err(RhoPriorError::constraint_violation(format!(
                    "{context} Normal log-precision prior requires finite mean and sd > 0"
                )));
            }
            let inv_var = 1.0 / (*sd * *sd);
            let delta = r - *mean;
            Ok((0.5 * delta * delta * inv_var, delta * inv_var, inv_var))
        }
        RhoPrior::GammaPrecision { shape, rate } => {
            if !shape.is_finite() || *shape <= 0.0 || !rate.is_finite() || *rate < 0.0 {
                return Err(RhoPriorError::constraint_violation(format!(
                    "{context} Gamma precision prior requires shape > 0 and rate >= 0"
                )));
            }
            let lambda = r.exp();
            // Deterministic REML/LAML uses the MAP-in-lambda convention; rho samplers add the Jacobian.
            Ok((
                *rate * lambda - (*shape - 1.0) * r,
                *rate * lambda - (*shape - 1.0),
                *rate * lambda,
            ))
        }
        RhoPrior::PenalizedComplexity { upper, tail_prob } => {
            if !upper.is_finite() || *upper <= 0.0 {
                return Err(RhoPriorError::constraint_violation(format!(
                    "{context} penalized-complexity prior requires a finite upper > 0"
                )));
            }
            if !tail_prob.is_finite() || *tail_prob <= 0.0 || *tail_prob >= 1.0 {
                return Err(RhoPriorError::constraint_violation(format!(
                    "{context} penalized-complexity prior requires tail probability in (0, 1)"
                )));
            }
            let theta = pc_prior_rate(*upper, *tail_prob);
            Ok(pc_prior_terms(theta, r))
        }
        RhoPrior::Independent(_) => Err(RhoPriorError::constraint_violation(format!(
            "{context} must be a scalar rho prior, not a nested Independent prior"
        ))),
    }
}

/// Saturated contribution returned (under [`InvalidPriorPolicy::Saturate`])
/// when the prior is malformed: `+inf` cost, `NaN` gradient, `NaN` Hessian.
pub(crate) fn saturated(len: usize) -> RhoPriorEval {
    RhoPriorEval {
        cost: f64::INFINITY,
        gradient: Array1::from_elem(len, f64::NAN),
        hessian: Some(Array2::from_elem((len, len), f64::NAN)),
    }
}

/// Evaluate the configured ρ prior under the given invalid-prior `policy`.
///
/// On success returns the prior `cost`, `gradient`, and diagonal `hessian`
/// (the latter `Some` only when non-zero). Under [`InvalidPriorPolicy::Saturate`]
/// a malformed prior yields the saturated `+inf`/`NaN` contribution and never
/// errors; under [`InvalidPriorPolicy::HardError`] it returns the structured
/// [`RhoPriorError`].
pub(crate) fn evaluate(
    prior: &RhoPrior,
    rho: &Array1<f64>,
    policy: InvalidPriorPolicy,
) -> Result<RhoPriorEval, RhoPriorError> {
    match evaluate_strict(prior, rho) {
        Ok(eval) => Ok(eval),
        Err(err) => match policy {
            InvalidPriorPolicy::HardError => Err(err),
            InvalidPriorPolicy::Saturate => Ok(saturated(rho.len())),
        },
    }
}

/// Strict evaluation: always errors on a malformed prior. Policy mapping lives
/// in [`evaluate`].
pub(crate) fn evaluate_strict(prior: &RhoPrior, rho: &Array1<f64>) -> Result<RhoPriorEval, RhoPriorError> {
    let len = rho.len();
    match prior {
        RhoPrior::Flat => Ok(RhoPriorEval {
            cost: 0.0,
            gradient: Array1::zeros(len),
            hessian: None,
        }),
        RhoPrior::Normal { .. }
        | RhoPrior::GammaPrecision { .. }
        | RhoPrior::PenalizedComplexity { .. } => {
            let mut cost = 0.0;
            let mut gradient = Array1::<f64>::zeros(len);
            let mut hessian = Array2::<f64>::zeros((len, len));
            let mut any_hessian = false;
            for (idx, &r) in rho.iter().enumerate() {
                let (c, g, h) = scalar_terms(prior, r, "rho prior")?;
                cost += c;
                gradient[idx] = g;
                hessian[[idx, idx]] = h;
                any_hessian |= h != 0.0;
            }
            Ok(RhoPriorEval {
                cost,
                gradient,
                hessian: any_hessian.then_some(hessian),
            })
        }
        RhoPrior::Independent(priors) => {
            if priors.len() != len {
                return Err(RhoPriorError::dimension_mismatch(format!(
                    "Independent rho prior length mismatch: got {}, expected {}",
                    priors.len(),
                    len
                )));
            }
            let mut cost = 0.0;
            let mut gradient = Array1::<f64>::zeros(len);
            let mut hessian = Array2::<f64>::zeros((len, len));
            let mut any_hessian = false;
            for (idx, (prior, &r)) in priors.iter().zip(rho.iter()).enumerate() {
                let (c, g, h) = scalar_terms(prior, r, &format!("rho prior coordinate {idx}"))?;
                cost += c;
                gradient[idx] = g;
                hessian[[idx, idx]] = h;
                any_hessian |= h != 0.0;
            }
            Ok(RhoPriorEval {
                cost,
                gradient,
                hessian: any_hessian.then_some(hessian),
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn approx(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-12, "expected {a} ~= {b}");
    }

    /// The shared engine and the saturating policy must agree term-for-term on
    /// every valid prior variant — this is the parity guarantee the two former
    /// duplicate call sites relied on.
    #[test]
    pub(crate) fn cost_grad_hess_parity_across_valid_priors() {
        let rho = Array1::from_vec(vec![-0.5, 0.25, 1.5, 0.7]);
        let priors = vec![
            RhoPrior::Flat,
            RhoPrior::Normal { mean: 0.2, sd: 0.8 },
            RhoPrior::GammaPrecision {
                shape: 2.0,
                rate: 0.5,
            },
            RhoPrior::PenalizedComplexity {
                upper: 0.5,
                tail_prob: 0.05,
            },
            RhoPrior::Independent(vec![
                RhoPrior::Flat,
                RhoPrior::Normal {
                    mean: -0.1,
                    sd: 1.3,
                },
                RhoPrior::GammaPrecision {
                    shape: 1.5,
                    rate: 0.0,
                },
                RhoPrior::PenalizedComplexity {
                    upper: 1.2,
                    tail_prob: 0.01,
                },
            ]),
        ];
        for prior in &priors {
            // Both policies must produce identical results when the prior is
            // valid: the policy only changes the malformed branch.
            let hard = evaluate(prior, &rho, InvalidPriorPolicy::HardError)
                .expect("valid prior must not error under HardError");
            let sat =
                evaluate(prior, &rho, InvalidPriorPolicy::Saturate).expect("Saturate never errors");
            approx(hard.cost, sat.cost);
            assert_eq!(hard.gradient, sat.gradient);
            assert_eq!(hard.hessian, sat.hessian);

            // Finite-difference check of gradient and (diagonal) Hessian. The
            // gradient uses a small step; the Hessian needs a larger one
            // because a central second difference amplifies roundoff like
            // macheps/h² — the optimal step is ≈ macheps^¼ ≈ 1e-4.
            let base = evaluate(prior, &rho, InvalidPriorPolicy::HardError).unwrap();
            let cost_at = |k: usize, delta: f64| -> f64 {
                let mut r = rho.clone();
                r[k] += delta;
                evaluate(prior, &r, InvalidPriorPolicy::HardError)
                    .unwrap()
                    .cost
            };
            let (h_grad, h_hess) = (1e-6, 1e-4);
            for k in 0..rho.len() {
                let fd_grad = (cost_at(k, h_grad) - cost_at(k, -h_grad)) / (2.0 * h_grad);
                assert!(
                    (fd_grad - base.gradient[k]).abs() <= 1e-5,
                    "gradient mismatch at {k}: fd {fd_grad} vs {}",
                    base.gradient[k]
                );
                let fd_hess = (cost_at(k, h_hess) - 2.0 * base.cost + cost_at(k, -h_hess))
                    / (h_hess * h_hess);
                let analytic_hess = base.hessian.as_ref().map_or(0.0, |h| h[[k, k]]);
                assert!(
                    (fd_hess - analytic_hess).abs() <= 1e-4,
                    "hessian mismatch at {k}: fd {fd_hess} vs {analytic_hess}"
                );
            }
        }
    }

    #[test]
    pub(crate) fn invalid_prior_policy_branches() {
        let rho = Array1::from_vec(vec![0.0, 0.0]);
        let bad_normal = RhoPrior::Normal {
            mean: 0.0,
            sd: -1.0,
        };
        // HardError surfaces a structured error.
        assert!(matches!(
            evaluate(&bad_normal, &rho, InvalidPriorPolicy::HardError),
            Err(RhoPriorError::ConstraintViolation { .. })
        ));
        // Saturate folds it into +inf / NaN.
        let sat = evaluate(&bad_normal, &rho, InvalidPriorPolicy::Saturate).unwrap();
        assert!(sat.cost.is_infinite() && sat.cost > 0.0);
        assert!(sat.gradient.iter().all(|v| v.is_nan()));
        assert!(sat.hessian.unwrap().iter().all(|v| v.is_nan()));

        // Length-mismatched Independent.
        let bad_len = RhoPrior::Independent(vec![RhoPrior::Flat]);
        assert!(matches!(
            evaluate(&bad_len, &rho, InvalidPriorPolicy::HardError),
            Err(RhoPriorError::DimensionMismatch { .. })
        ));
        // Nested Independent.
        let nested = RhoPrior::Independent(vec![
            RhoPrior::Independent(vec![RhoPrior::Flat]),
            RhoPrior::Flat,
        ]);
        assert!(matches!(
            evaluate(&nested, &rho, InvalidPriorPolicy::HardError),
            Err(RhoPriorError::ConstraintViolation { .. })
        ));
    }

    // ---- Penalized-complexity prior ---------------------------------------

    /// Normalized PC-prior log-density on ρ (includes the additive constant
    /// `ln(θ/2)` the optimizer cost drops). `log p(ρ) = ln(θ/2) − ρ/2 − θ
    /// exp(−ρ/2)`, the change-of-variables image of `d ~ Exp(θ)` under
    /// `d = exp(−ρ/2)`.
    pub(crate) fn pc_log_pdf(upper: f64, tail_prob: f64, r: f64) -> f64 {
        let theta = pc_prior_rate(upper, tail_prob);
        (0.5 * theta).ln() - 0.5 * r - theta * (-0.5 * r).exp()
    }

    #[test]
    pub(crate) fn pc_rate_calibrates_to_tail_statement() {
        // θ = −ln(α)/U solves P(d > U) = exp(−θU) = α exactly.
        for &(upper, alpha) in &[(0.5_f64, 0.05_f64), (1.2, 0.01), (3.0, 0.25)] {
            let theta = pc_prior_rate(upper, alpha);
            let tail = (-theta * upper).exp();
            assert!(
                (tail - alpha).abs() < 1e-12,
                "P(d>U)={tail} vs α={alpha} (U={upper})"
            );
        }
    }

    #[test]
    pub(crate) fn pc_density_integrates_to_one_and_matches_tail() {
        // Trapezoidal integration of the normalized ρ-density. d = exp(−ρ/2)
        // ranges over (0, ∞); the density decays at both ends, so a wide grid
        // captures essentially all the mass.
        let upper = 0.5_f64;
        let alpha = 0.05_f64;
        let (lo, hi, n) = (-60.0_f64, 80.0_f64, 2_000_000usize);
        let h = (hi - lo) / n as f64;
        // ρ < −2 ln U  ⇔  exp(−ρ/2) > U  ⇔  d > U: the tail region.
        let tail_boundary = -2.0 * upper.ln();
        let mut total = 0.0;
        let mut tail = 0.0;
        for i in 0..=n {
            let r = lo + i as f64 * h;
            let w = if i == 0 || i == n { 0.5 } else { 1.0 };
            let p = pc_log_pdf(upper, alpha, r).exp();
            total += w * p;
            if r <= tail_boundary {
                tail += w * p;
            }
        }
        total *= h;
        tail *= h;
        assert!((total - 1.0).abs() < 1e-4, "∫ p(ρ) dρ = {total}");
        // P(d > U) recovered from the ρ-density must equal the calibration α.
        assert!(
            (tail - alpha).abs() < 1e-3,
            "P(d>U) = {tail} vs α = {alpha}"
        );
    }

    #[test]
    pub(crate) fn pc_terms_are_negative_log_density_derivatives() {
        // cost = −log p (up to the dropped constant); grad/hess are its ρ
        // derivatives, cross-checked against a finite difference of pc_log_pdf.
        let (upper, alpha) = (0.8_f64, 0.02_f64);
        let theta = pc_prior_rate(upper, alpha);
        // Small step for the first derivative; larger step for the second,
        // whose central difference amplifies roundoff like macheps/h².
        let (h1, h2) = (1e-6, 1e-4);
        for &r in &[-2.0_f64, -0.3, 0.0, 1.7, 4.0] {
            let (cost, grad, hess) = pc_prior_terms(theta, r);
            // cost + log p(ρ) is the ρ-independent constant ln(θ/2).
            approx(cost + pc_log_pdf(upper, alpha, r), (0.5 * theta).ln());
            // grad = −d/dρ log p  (FD on the log-density).
            let dlp =
                (pc_log_pdf(upper, alpha, r + h1) - pc_log_pdf(upper, alpha, r - h1)) / (2.0 * h1);
            let neg_dlp = -dlp;
            assert!(
                (grad - neg_dlp).abs() < 1e-5,
                "grad {grad} vs {neg_dlp} at r={r}"
            );
            // hess = −d²/dρ² log p (FD), and is strictly positive (convex).
            let d2lp = (pc_log_pdf(upper, alpha, r + h2) - 2.0 * pc_log_pdf(upper, alpha, r)
                + pc_log_pdf(upper, alpha, r - h2))
                / (h2 * h2);
            let neg_d2lp = -d2lp;
            assert!(
                (hess - neg_d2lp).abs() < 1e-4,
                "hess {hess} vs {neg_d2lp} at r={r}"
            );
            assert!(hess > 0.0, "PC curvature must be positive, got {hess}");
        }
    }

    #[test]
    pub(crate) fn pc_prior_pulls_toward_simpler_model() {
        // The simpler (base) model is more smoothing: larger ρ ⇒ larger
        // precision λ = exp(ρ) ⇒ the penalized component collapses. The PC cost
        // must therefore make *under*-smoothing (small ρ, wiggly) more expensive
        // than over-smoothing of the same magnitude — an asymmetric, convex
        // bowl whose gradient at the base side is bounded by 1/2.
        let prior = RhoPrior::PenalizedComplexity {
            upper: 1.0,
            tail_prob: 0.05,
        };
        let cost = |r: f64| {
            evaluate(
                &prior,
                &Array1::from_vec(vec![r]),
                InvalidPriorPolicy::HardError,
            )
            .unwrap()
            .cost
        };
        // Wiggly (ρ = −4) is penalized far more than smooth (ρ = +4).
        assert!(
            cost(-4.0) > cost(4.0),
            "under-smoothing must cost more: {} vs {}",
            cost(-4.0),
            cost(4.0)
        );
        // As ρ → +∞ the cost grows only linearly (slope 1/2): a gentle pull, not
        // a wall — the data can still buy complexity.
        let g_far = evaluate(
            &prior,
            &Array1::from_vec(vec![25.0]),
            InvalidPriorPolicy::HardError,
        )
        .unwrap()
        .gradient[0];
        assert!(
            (g_far - 0.5).abs() < 1e-3,
            "far over-smoothing slope {g_far}"
        );
    }

    #[test]
    pub(crate) fn pc_prior_rejects_invalid_hyperparameters() {
        let rho = Array1::from_vec(vec![0.0]);
        for bad in [
            RhoPrior::PenalizedComplexity {
                upper: 0.0,
                tail_prob: 0.05,
            },
            RhoPrior::PenalizedComplexity {
                upper: -1.0,
                tail_prob: 0.05,
            },
            RhoPrior::PenalizedComplexity {
                upper: 1.0,
                tail_prob: 0.0,
            },
            RhoPrior::PenalizedComplexity {
                upper: 1.0,
                tail_prob: 1.0,
            },
            RhoPrior::PenalizedComplexity {
                upper: 1.0,
                tail_prob: f64::NAN,
            },
        ] {
            assert!(matches!(
                evaluate(&bad, &rho, InvalidPriorPolicy::HardError),
                Err(RhoPriorError::ConstraintViolation { .. })
            ));
            let sat = evaluate(&bad, &rho, InvalidPriorPolicy::Saturate).unwrap();
            assert!(sat.cost.is_infinite() && sat.cost > 0.0);
        }
    }
}
