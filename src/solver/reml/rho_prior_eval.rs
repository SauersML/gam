//! Shared evaluation of the configured smoothing-parameter (ρ) prior.
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
//!   precision `λ = exp(r)`, contributing (up to an additive constant)
//!   cost `rate · λ − (shape − 1) · r`, grad `rate · λ − (shape − 1)`,
//!   Hessian `rate · λ`.
//! * `Independent`    — one prior per coordinate; the same per-coordinate math
//!   summed/stacked, with nested `Independent` priors rejected as invalid.
//!
//! The Hessian is diagonal and is returned as `Some` only when at least one
//! diagonal entry is non-zero; an all-zero curvature contribution collapses to
//! `None` so callers can skip adding an empty block.

use crate::types::RhoPrior;
use ndarray::{Array1, Array2};

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
    fn dimension_mismatch(reason: String) -> Self {
        RhoPriorError::DimensionMismatch { reason }
    }

    fn constraint_violation(reason: String) -> Self {
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
fn scalar_terms(prior: &RhoPrior, r: f64, context: &str) -> Result<(f64, f64, f64), RhoPriorError> {
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
            Ok((
                *rate * lambda - (*shape - 1.0) * r,
                *rate * lambda - (*shape - 1.0),
                *rate * lambda,
            ))
        }
        RhoPrior::Independent(_) => Err(RhoPriorError::constraint_violation(format!(
            "{context} must be a scalar rho prior, not a nested Independent prior"
        ))),
    }
}

/// Saturated contribution returned (under [`InvalidPriorPolicy::Saturate`])
/// when the prior is malformed: `+inf` cost, `NaN` gradient, `NaN` Hessian.
fn saturated(len: usize) -> RhoPriorEval {
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
fn evaluate_strict(prior: &RhoPrior, rho: &Array1<f64>) -> Result<RhoPriorEval, RhoPriorError> {
    let len = rho.len();
    match prior {
        RhoPrior::Flat => Ok(RhoPriorEval {
            cost: 0.0,
            gradient: Array1::zeros(len),
            hessian: None,
        }),
        RhoPrior::Normal { .. } | RhoPrior::GammaPrecision { .. } => {
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

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-12, "expected {a} ~= {b}");
    }

    /// The shared engine and the saturating policy must agree term-for-term on
    /// every valid prior variant — this is the parity guarantee the two former
    /// duplicate call sites relied on.
    #[test]
    fn cost_grad_hess_parity_across_valid_priors() {
        let rho = Array1::from_vec(vec![-0.5, 0.25, 1.5]);
        let priors = vec![
            RhoPrior::Flat,
            RhoPrior::Normal { mean: 0.2, sd: 0.8 },
            RhoPrior::GammaPrecision {
                shape: 2.0,
                rate: 0.5,
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

            // Finite-difference check of gradient and (diagonal) Hessian.
            let eps = 1e-6;
            let base = evaluate(prior, &rho, InvalidPriorPolicy::HardError).unwrap();
            for k in 0..rho.len() {
                let mut rp = rho.clone();
                rp[k] += eps;
                let mut rm = rho.clone();
                rm[k] -= eps;
                let cp = evaluate(prior, &rp, InvalidPriorPolicy::HardError)
                    .unwrap()
                    .cost;
                let cm = evaluate(prior, &rm, InvalidPriorPolicy::HardError)
                    .unwrap()
                    .cost;
                let fd_grad = (cp - cm) / (2.0 * eps);
                assert!(
                    (fd_grad - base.gradient[k]).abs() <= 1e-5,
                    "gradient mismatch at {k}: fd {fd_grad} vs {}",
                    base.gradient[k]
                );
                let fd_hess = (cp - 2.0 * base.cost + cm) / (eps * eps);
                let analytic_hess = base.hessian.as_ref().map_or(0.0, |h| h[[k, k]]);
                assert!(
                    (fd_hess - analytic_hess).abs() <= 1e-3,
                    "hessian mismatch at {k}: fd {fd_hess} vs {analytic_hess}"
                );
            }
        }
    }

    #[test]
    fn invalid_prior_policy_branches() {
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
}
