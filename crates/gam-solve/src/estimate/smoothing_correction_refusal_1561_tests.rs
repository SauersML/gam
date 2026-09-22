//! One reason an unavailable smoothing correction does NOT refuse the fit, and
//! every other one still does.
//!
//! The optimizer used to refuse on any `SmoothingCorrectionOutcome::Unavailable`,
//! justified in place by "Every Firth link carries its analytic outer ρ-Hessian
//! (#3203), so an unavailable correction is a real defect." That premise is
//! about a matrix that EXISTS and was not produced. It is false for one reason:
//! a criterion that DECLARES it has no outer ρ-Hessian, which a term priced on
//! a profiled posterior does (gam#3234) — every shape-constrained smooth at
//! estimated scale. Refusing there killed fits whose point estimate was
//! converged and whose uncorrected covariance is exactly what mgcv publishes
//! with `unconditional = FALSE` and what scam publishes for the same models
//! (gam#1561).
//!
//! Nothing pinned the Firth premise, so nothing would have caught the opposite
//! mistake either: widening the exemption until a genuine assembly failure was
//! published as a fit. `unavailable_correction_refuses_fit` is the one place
//! that decides, and this is its gate.
//!
//! The list below is written out by name rather than derived, so a tenth
//! variant does not join either side by default: adding one leaves it out of
//! `every_reason`, and `every_reason_is_classified_1561` stops compiling.

use crate::estimate::smoothing_correction::{
    SmoothingCorrectionUnavailable, unavailable_correction_refuses_fit,
};

/// Every reason the correction can be unavailable, one of each, by name.
fn every_reason() -> Vec<SmoothingCorrectionUnavailable> {
    vec![
        SmoothingCorrectionUnavailable::ObjectiveInnerHessian {
            error: "inner Hessian assembly failed".to_string(),
        },
        SmoothingCorrectionUnavailable::InnerHessianDimension {
            rows: 3,
            cols: 4,
            coefficients: 3,
        },
        SmoothingCorrectionUnavailable::InnerHessianNotPositiveDefinite,
        SmoothingCorrectionUnavailable::SensitivitySolve,
        SmoothingCorrectionUnavailable::OuterHessian {
            error: "the assembly reported a failure".to_string(),
        },
        SmoothingCorrectionUnavailable::OuterHessianDeclaredAbsent,
        SmoothingCorrectionUnavailable::OuterHessianInverse {
            error: "the inverse refused".to_string(),
        },
        SmoothingCorrectionUnavailable::PenaltyDimension {
            rho: 2,
            lambdas: 2,
            canonical_penalties: 3,
        },
        SmoothingCorrectionUnavailable::PenaltyStructure {
            error: "the penalty map disagrees".to_string(),
        },
        SmoothingCorrectionUnavailable::NonFiniteCorrection,
    ]
}

/// The declaration is the ONLY reason that publishes the fit.
#[test]
fn only_a_declared_absent_outer_hessian_publishes_the_fit_1561() {
    let published: Vec<SmoothingCorrectionUnavailable> = every_reason()
        .into_iter()
        .filter(|reason| !unavailable_correction_refuses_fit(reason))
        .collect();
    assert_eq!(
        published,
        vec![SmoothingCorrectionUnavailable::OuterHessianDeclaredAbsent],
        "exactly one reason may publish the fit instead of refusing it, and it is the \
         criterion's own declaration"
    );
}

/// Every FAILURE still refuses, which is the Firth premise (#3203) this rule
/// must not have widened: a Firth link's outer ρ-Hessian exists, so an
/// unavailable correction there is a defect and the fit does not publish.
#[test]
fn every_assembly_failure_still_refuses_the_fit_1561() {
    for reason in every_reason() {
        if reason == SmoothingCorrectionUnavailable::OuterHessianDeclaredAbsent {
            continue;
        }
        assert!(
            unavailable_correction_refuses_fit(&reason),
            "{reason:?} is a failure to produce a matrix that exists, so the fit must refuse"
        );
    }
}

/// The list this file classifies covers the enum.
///
/// `SmoothingCorrectionUnavailable` is not enumerable at runtime, so the cover
/// is checked the way the compiler can check it: an exhaustive match with no
/// wildcard, which fails to build the moment a variant is added without a
/// decision here about which side it falls on.
#[test]
fn every_reason_is_classified_1561() {
    for reason in every_reason() {
        let refuses = match &reason {
            SmoothingCorrectionUnavailable::ObjectiveInnerHessian { .. }
            | SmoothingCorrectionUnavailable::InnerHessianDimension { .. }
            | SmoothingCorrectionUnavailable::InnerHessianNotPositiveDefinite
            | SmoothingCorrectionUnavailable::SensitivitySolve
            | SmoothingCorrectionUnavailable::OuterHessian { .. }
            | SmoothingCorrectionUnavailable::OuterHessianInverse { .. }
            | SmoothingCorrectionUnavailable::PenaltyDimension { .. }
            | SmoothingCorrectionUnavailable::PenaltyStructure { .. }
            | SmoothingCorrectionUnavailable::NonFiniteCorrection => true,
            SmoothingCorrectionUnavailable::OuterHessianDeclaredAbsent => false,
        };
        assert_eq!(
            refuses,
            unavailable_correction_refuses_fit(&reason),
            "{reason:?} is classified one way here and another by the rule"
        );
    }
}
