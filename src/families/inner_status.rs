//! Structured return type for the custom-family inner blockwise solver.
//!
//! The legacy contract was `Result<BlockwiseInnerResult, String>` — the
//! outer optimiser could read it as success / failure but could not
//! discriminate *why* a failure occurred without parsing the error
//! string. Three classes of failure all looked the same to the outer
//! seed-validation cascade:
//!
//! - The joint Newton certificate refused (rank-deficient H_pen, phantom
//!   multiplier, incomplete active set). This is a *structural* failure:
//!   the same seed will fail the same way every time, and so will every
//!   sibling seed that sees the same penalized Hessian rank.
//! - The likelihood evaluator returned NaN / domain error. This is a
//!   *seed-local* failure: another seed at a different ρ is likely fine.
//! - Budget exhaustion / trust-region floor without certifying. This is
//!   between the two — sometimes structural, sometimes just unlucky.
//!
//! [`InnerStatus`] preserves the existing data while exposing this
//! structure. It is produced from a legacy `Result<BlockwiseInnerResult,
//! String>` by [`classify_inner_result`], which inspects the bubbled
//! error string for the sentinels emitted by `inner_blockwise_fit` and
//! the diagnostician's `KktRefusalReport`. Once the inner solver is
//! reworked to emit `InnerStatus` natively, the classifier becomes the
//! identity on the structured branch and the string path is removed.

use super::custom_family::{BlockwiseInnerResult, KktRefusalDiagnosis};

/// Lightweight KKT/cert summary attached to every inner status that
/// terminated at a (real or constrained) stationary point. Carries
/// enough information for the outer optimiser to log seed convergence
/// without holding a reference to the full [`BlockwiseInnerResult`].
#[derive(Clone, Debug)]
pub(crate) struct KktCertificate {
    pub projected_residual_inf: f64,
    pub residual_tol: f64,
    pub active_set_size: usize,
    pub free_rank: Option<usize>,
    pub accepted_step_inf: f64,
    pub obj_change: f64,
}

/// Structured failure modes for the inner solver. Each variant carries
/// the original message for backwards-compatible logging plus structured
/// fields that the outer cascade can match against without reparsing.
#[derive(Clone, Debug)]
pub(crate) enum InnerFailure {
    /// The joint Newton constrained-stationary certificate refused. The
    /// projected KKT residual exceeded 4× the residual tolerance, and the
    /// underlying H_pen spectrum / active-set inspection classified the
    /// reason via [`KktRefusalDiagnosis`].
    ///
    /// `carrying_block` names the block (by spec name) that holds the
    /// largest unresolved residual mass when the bubbled error embeds
    /// the diagnostician's structured payload. Older error strings (the
    /// legacy `block_residual_diagnostic_string` shape) populate this via
    /// the `block '<name>' carries the dominant unresolved KKT gradient`
    /// sentinel. The pair `(diagnosis, carrying_block)` is the equality
    /// key the seed-screening loop uses for structural early-exit.
    CertRefused {
        diagnosis: KktRefusalDiagnosis,
        carrying_block: Option<String>,
        message: String,
    },
    /// Joint Newton ran out of cycle budget without ever certifying and
    /// without crossing into "structural rank deficiency" territory.
    BudgetExhausted { message: String },
    /// Trust-region radius collapsed to its floor; the proposal is
    /// rejected but no certificate could be formed at the floor.
    TrustRegionFloor { message: String },
    /// The likelihood / penalty evaluator returned NaN, ±∞, or a
    /// family-domain error. The seed sits outside the feasible region
    /// for the parameterisation.
    LikelihoodFailure(String),
    /// Catch-all for legacy error strings that do not match any of the
    /// structured sentinels above. These are still rejected; the outer
    /// cascade just cannot classify them.
    Other(String),
}

impl InnerFailure {
    pub(crate) fn message(&self) -> &str {
        match self {
            InnerFailure::CertRefused { message, .. }
            | InnerFailure::BudgetExhausted { message }
            | InnerFailure::TrustRegionFloor { message }
            | InnerFailure::LikelihoodFailure(message)
            | InnerFailure::Other(message) => message.as_str(),
        }
    }
}

/// Structured replacement for `Result<BlockwiseInnerResult, String>`.
/// `Converged` and `ConstrainedStationary` both carry a successful
/// result; the split keeps the certificate provenance explicit so the
/// outer optimiser can apply different IFT projections without
/// re-deriving the certificate at the call site.
#[derive(Debug)]
pub(crate) enum InnerStatus {
    Converged {
        result: BlockwiseInnerResult,
        certificate: KktCertificate,
    },
    ConstrainedStationary {
        result: BlockwiseInnerResult,
        certificate: KktCertificate,
    },
    Failed(InnerFailure),
}

/// Promote a legacy inner-solver result into [`InnerStatus`]. Inspects
/// the bubbled error string for the structured sentinels emitted by
/// `inner_blockwise_fit` and `KktRefusalReport::format_bubbled_error`.
/// On success, packages the `BlockwiseInnerResult.kkt_residual` (if
/// any) into a [`KktCertificate`] and splits `Converged` vs
/// `ConstrainedStationary` on the `converged` flag.
pub(crate) fn classify_inner_result(
    raw: Result<BlockwiseInnerResult, String>,
) -> InnerStatus {
    match raw {
        Ok(result) => {
            let certificate = certificate_from_result(&result);
            if result.converged {
                InnerStatus::Converged {
                    result,
                    certificate,
                }
            } else {
                InnerStatus::ConstrainedStationary {
                    result,
                    certificate,
                }
            }
        }
        Err(message) => InnerStatus::Failed(classify_inner_error(message)),
    }
}

fn certificate_from_result(result: &BlockwiseInnerResult) -> KktCertificate {
    let projected_residual_inf = result
        .kkt_residual
        .as_ref()
        .map(|kkt| {
            kkt.as_array()
                .iter()
                .map(|x: &f64| x.abs())
                .fold(0.0_f64, f64::max)
        })
        .unwrap_or(f64::NAN);
    let active_set_size: usize = result
        .active_sets
        .iter()
        .map(|maybe| maybe.as_ref().map(|v| v.len()).unwrap_or(0))
        .sum();
    KktCertificate {
        projected_residual_inf,
        residual_tol: f64::NAN,
        active_set_size,
        free_rank: None,
        accepted_step_inf: f64::NAN,
        obj_change: f64::NAN,
    }
}

pub(crate) fn classify_inner_error(message: String) -> InnerFailure {
    // The diagnostician's structured cert-refusal bubbled error carries
    // `diagnosis: <label>` near the end. Look for that first.
    if let Some(diagnosis) = KktRefusalDiagnosis::parse_from_error(&message) {
        let carrying_block = parse_carrying_block(&message);
        return InnerFailure::CertRefused {
            diagnosis,
            carrying_block,
            message,
        };
    }
    // Legacy cert-refusal path: the inner solver bubbled the error via
    // `block_residual_diagnostic_string`. We can recover the carrying
    // block name but not the H_pen-spectrum diagnosis, so default to
    // PhantomMultiplier when the sentinel matches and otherwise leave
    // the failure unclassified.
    if message.contains("coupled exact-joint inner solve exited the joint Newton path")
        || message.contains("carries the dominant unresolved KKT gradient")
    {
        let carrying_block = parse_legacy_carrying_block(&message);
        return InnerFailure::CertRefused {
            diagnosis: KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH,
            carrying_block,
            message,
        };
    }
    if message.contains("inner_max_cycles") || message.contains("budget exhausted") {
        return InnerFailure::BudgetExhausted { message };
    }
    if message.contains("trust-region floor") || message.contains("trust region floor") {
        return InnerFailure::TrustRegionFloor { message };
    }
    if message.contains("NaN")
        || message.contains("non-finite")
        || message.contains("not finite")
        || message.contains("DomainError")
        || message.contains("likelihood evaluation failed")
    {
        return InnerFailure::LikelihoodFailure(message);
    }
    InnerFailure::Other(message)
}

/// Extract `<name>` from the diagnostician's structured `carrying-block:
/// <name> (idx=...` payload. Returns `None` if the sentinel is absent or
/// the name slice is empty.
fn parse_carrying_block(message: &str) -> Option<String> {
    let marker = "carrying-block: ";
    let start = message.find(marker)? + marker.len();
    let tail = &message[start..];
    let end = tail.find(" (idx=").unwrap_or_else(|| {
        tail.find(|c: char| c == ';' || c == '\n')
            .unwrap_or(tail.len())
    });
    let name = tail[..end].trim();
    if name.is_empty() || name == "<no block carries finite residual>" {
        None
    } else {
        Some(name.to_string())
    }
}

/// Legacy parser: `block '<name>' carries the dominant unresolved KKT
/// gradient (...)`. The single-quoted block name lets us recover the
/// carrying block even when the structured diagnosis label is missing.
fn parse_legacy_carrying_block(message: &str) -> Option<String> {
    let marker = "block '";
    let start = message.find(marker)? + marker.len();
    let tail = &message[start..];
    let end = tail.find('\'')?;
    let name = &tail[..end];
    if name.is_empty() { None } else { Some(name.to_string()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_structured_diagnosis_and_carrying_block() {
        let message = "cycle=7 cert REFUSED: residual=5.0e+05 > 4·tol=4.0e+03; \
            carrying-block: time_surface (idx=0, |g|=5.0e+05, |Sβ|=1.0e-03, \
            |∇L-Sβ|=5.0e+05, |β|=1.0e+00, width=12); diagnosis: rank_deficient_H_pen; \
            check whether the named block's penalty has a polynomial null space"
            .to_string();
        match classify_inner_error(message.clone()) {
            InnerFailure::CertRefused {
                diagnosis,
                carrying_block,
                ..
            } => {
                assert_eq!(diagnosis, KktRefusalDiagnosis::RankDeficientHPen);
                assert_eq!(carrying_block.as_deref(), Some("time_surface"));
            }
            other => panic!("expected CertRefused, got {other:?}"),
        }
    }

    #[test]
    fn parses_legacy_carrying_block_when_diagnosis_absent() {
        let message = "coupled exact-joint inner solve exited the joint Newton path before \
            convergence — block 'time_surface' carries the dominant unresolved KKT gradient \
            (|g_block|∞ = 5.000e+05); |∇L − Sβ|∞ = 5.000e+05"
            .to_string();
        match classify_inner_error(message) {
            InnerFailure::CertRefused {
                diagnosis,
                carrying_block,
                ..
            } => {
                assert_eq!(
                    diagnosis,
                    KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH
                );
                assert_eq!(carrying_block.as_deref(), Some("time_surface"));
            }
            other => panic!("expected CertRefused, got {other:?}"),
        }
    }

    #[test]
    fn classifies_likelihood_failures() {
        let message = "likelihood evaluation failed: NaN response".to_string();
        assert!(matches!(
            classify_inner_error(message),
            InnerFailure::LikelihoodFailure(_)
        ));
    }

    #[test]
    fn classifies_budget_exhaustion() {
        let message = "inner_max_cycles reached without certification".to_string();
        assert!(matches!(
            classify_inner_error(message),
            InnerFailure::BudgetExhausted { .. }
        ));
    }

    #[test]
    fn falls_back_to_other_for_unknown_strings() {
        let message = "some completely unrecognised legacy error".to_string();
        assert!(matches!(
            classify_inner_error(message),
            InnerFailure::Other(_)
        ));
    }
}
