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
//! [`InnerFailure`] captures these classes. Startup accounting classifies
//! bubbled inner errors through [`classify_inner_error`], which inspects the
//! structured labels emitted by `inner_blockwise_fit` and the diagnostician's
//! `KktRefusalReport`.

use gam_problem::diagnostics::KktRefusalDiagnosis;

/// Structured failure modes for the inner solver. Each variant carries the
/// original message for logging plus structured fields that the outer cascade
/// can match against without reparsing.
#[derive(Clone, Debug)]
pub(crate) enum InnerFailure {
    /// The joint Newton constrained-stationary certificate refused. The
    /// projected KKT residual exceeded 4× the residual tolerance, and the
    /// underlying H_pen spectrum / active-set inspection classified the
    /// reason via [`KktRefusalDiagnosis`].
    ///
    /// `carrying_block` names the block (by spec name) that holds the
    /// largest unresolved residual mass when the bubbled error embeds the
    /// diagnostician's structured payload. The pair `(diagnosis,
    /// carrying_block)` is the equality key the seed-screening loop uses for
    /// structural early-exit.
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
    /// Pre-fit cross-block identifiability audit refused the fit. The
    /// joint design across `ParameterBlockSpec`s carries a rank
    /// deficiency that post-`joint_null_rotation` absorption did not
    /// resolve. Held as a structured failure (not `Other`) so the
    /// continuation driver and `StartupStats` can distinguish "the
    /// solver couldn't classify this" from "the design is provably
    /// unfittable in its current shape". No rho-anneal recovers this;
    /// the structural fix is to reparameterise the aliased block.
    IdentifiabilityFailure { message: String },
    /// Catch-all for strings that do not match any of the structured sentinels
    /// above. These are still rejected; the outer cascade just cannot classify
    /// them.
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
            InnerFailure::IdentifiabilityFailure { message } => message.as_str(),
        }
    }
}

pub(crate) fn classify_inner_error(message: String) -> InnerFailure {
    if message.contains("IdentifiabilityFailure") || message.contains("identifiability audit") {
        return InnerFailure::IdentifiabilityFailure { message };
    }
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
    if message.contains("inner_max_cycles")
        || message.contains("budget exhausted")
        || message.contains("exhausted the joint Newton budget")
        || message.contains("did not converge after")
    {
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
    fn classifies_joint_newton_budget_exhaustion() {
        let message = "coupled exact-joint inner solve exhausted the joint Newton budget \
            without KKT convergence after 1200 cycle(s)"
            .to_string();
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

    /// Round-trip test against the diagnostician's
    /// `KktRefusalReport::format_bubbled_error` output. The report
    /// formatter lives in `src/families/custom_family.rs` and is private; we
    /// reproduce its exact text shape here so the parser stays pinned to the
    /// single structured report format.
    ///
    /// The fixture below is a verbatim copy of the diagnostician's
    /// format string with concrete values plugged in. Any change to
    /// the report's text shape that breaks this test must be
    /// accompanied by a matching update to the parser.
    #[test]
    fn round_trip_matches_kkt_refusal_report_bubbled_format() {
        for (diagnosis_label, expected) in [
            (
                "rank_deficient_H_pen",
                KktRefusalDiagnosis::RankDeficientHPen,
            ),
            (
                "phantom_multiplier_with_well_conditioned_H",
                KktRefusalDiagnosis::PhantomMultiplierWithWellConditionedH,
            ),
            (
                "active_set_incomplete",
                KktRefusalDiagnosis::ActiveSetIncomplete,
            ),
        ] {
            let message = format!(
                "cycle=7 cert REFUSED: residual=5.000e+05 > 4·tol=4.000e+03; \
                 carrying-block: time_surface (idx=0, |g|=5.000e+05, |Sβ|=1.000e-03, \
                 |∇L-Sβ|=5.000e+05, |β|=1.000e+00, width=12); \
                 block_names=[\"time_surface\", \"marginal\", \"logslope\"], \
                 block_widths=[12, 11, 10], block_grad_inf=[5.0e+05, 1.0e-03, 1.0e-03], \
                 block_penalty_grad_inf=[1.0e-03, 1.0e-03, 1.0e-03], \
                 block_residual_inf=[5.0e+05, 1.0e-03, 1.0e-03]; \
                 H_pen spectrum: λ_max=1.000e+03, λ_min=1.000e-12, cond=1.000e+15, \
                 nullity@1e-10=2/33; cert math: linearized_rel=9.99e-01, \
                 scalar_relerr=1.0e-10, |Δobj|=1.0e-09, accepted_step_inf=1.0e+00, \
                 proposal_step_inf=1.0e+06, trust_radius=1.0e-03, |β|∞=1.0e+00, \
                 active_set_rows_total=0; diagnosis: {diagnosis_label}; \
                 check whether the named block's penalty has a polynomial null space"
            );
            match classify_inner_error(message) {
                InnerFailure::CertRefused {
                    diagnosis,
                    carrying_block,
                    ..
                } => {
                    assert_eq!(
                        diagnosis, expected,
                        "diagnosis label {diagnosis_label} must round-trip"
                    );
                    assert_eq!(
                        carrying_block.as_deref(),
                        Some("time_surface"),
                        "carrying-block must survive the round-trip for {diagnosis_label}"
                    );
                }
                other => panic!("expected CertRefused for label {diagnosis_label}, got {other:?}"),
            }
        }
    }
}
