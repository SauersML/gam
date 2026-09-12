//! Judgement of a Ridders finite-difference measurement against an analytic
//! derivative, for gradient tests.
//!
//! These methods decide agreement and render the ladder for a diagnostic line.
//! Only tests call them, so they live here rather than on the production
//! [`FdDerivative`] type, whose measurement API stays in `gam-linalg`.

use gam_linalg::numeric_derivative::{FdDerivative, FdVerdict};

/// Test-side judgement of an [`FdDerivative`] measurement.
pub trait FdDerivativeJudgement {
    /// The single place that decides whether an analytic derivative component
    /// agrees with this measurement. Callers should route every comparison
    /// through it rather than re-deriving the three-way rule, which is easy to
    /// state as two-way and thereby convert every unmeasurable component into a
    /// false violation.
    fn judge(&self, analytic: f64, rel_tol: f64, abs_floor: f64) -> FdVerdict;

    /// The ladder rendered for a diagnostic line: `h=… D=…` coarsest first.
    fn ladder_report(&self) -> String;
}

impl FdDerivativeJudgement for FdDerivative {
    fn judge(&self, analytic: f64, rel_tol: f64, abs_floor: f64) -> FdVerdict {
        if !self.value.is_finite() || !self.resolved(rel_tol, abs_floor) {
            return FdVerdict::Unresolved;
        }
        if (analytic - self.value).abs() > self.agreement_bound(analytic, rel_tol, abs_floor) {
            FdVerdict::Disagree
        } else {
            FdVerdict::Agree
        }
    }

    fn ladder_report(&self) -> String {
        self.ladder
            .iter()
            .map(|(h, d)| format!("h={h:.2e} D={d:+.10e}"))
            .collect::<Vec<_>>()
            .join("  ")
    }
}
