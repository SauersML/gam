//! Margin-resolved [`Verdict`] mappings for the gam-solve-tier certificates
//! (descended #1521).
//!
//! The monolith `inference::certificate_impls` mixes two unrelated concerns in
//! one file: (a) the `impl Certificate for …` blocks for the gam-sae certificate
//! zoo (`EncodeResult`, `ResidualGaugeReport`, `CertificateInputs`, …), which
//! genuinely pull `gam_sae` and therefore stay above gam-solve, and (b) the two
//! pure margin-resolution helpers below, whose only inputs are gam-solve-tier
//! types ([`LogdetEnclosure`]/[`MarginVerdict`] and
//! [`CoresetMarginVerdict`](crate::row_sampling_measure::CoresetMarginVerdict))
//! plus the contracted-down [`Verdict`] ladder. Only (b) is consumed by
//! [`crate::topology_selector`], so it is descended here unchanged; (a) is left
//! in the monolith. Bodies are byte-identical to the monolith originals so there
//! remains exactly one decision rule per verdict.

use crate::logdet_bounds::{LogdetEnclosure, MarginVerdict};
use crate::row_sampling_measure::CoresetMarginVerdict;
use gam_problem::topology_certificates::Verdict;

/// Map a coreset race outcome (the certificate's own
/// [`CoresetCertificate::certify_margin`](crate::row_sampling_measure::CoresetCertificate::certify_margin)
/// rule, evaluated against a consumer's
/// `decision_margin`) onto the shared [`Verdict`] ladder. This is the
/// margin-resolved entry point a race consumer uses to obtain a unified verdict
/// without re-deriving the mapping.
pub fn coreset_race_verdict(verdict: CoresetMarginVerdict) -> Verdict {
    match verdict {
        CoresetMarginVerdict::Certified { .. } => Verdict::Certified,
        CoresetMarginVerdict::InsufficientMargin { .. } => Verdict::Insufficient,
    }
}

/// Verdict for an enclosure resolved against a concrete consumer
/// `decision_margin`, reusing [`LogdetEnclosure::decide_within_margin`].
pub fn enclosure_margin_verdict(enclosure: &LogdetEnclosure, decision_margin: f64) -> Verdict {
    match enclosure.decide_within_margin(decision_margin) {
        MarginVerdict::Decided { .. } => Verdict::Certified,
        MarginVerdict::InsufficientMargin { .. } => Verdict::Insufficient,
    }
}
