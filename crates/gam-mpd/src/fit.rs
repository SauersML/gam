//! Structural proposals of a program decomposition (#2951), decided on decoded code.
//!
//! A proposal (share, split, refine, reduce, expose) is decided on decoded
//! artifacts only:
//! * the decoded reference must meet the declared tolerance;
//! * the candidate's fidelity status over the declared domain must be a bound on
//!   the supremum, not an estimate, and must not refute `sup d ≤ ε`;
//! * the decoded candidate must meet the tolerance with a strictly shorter code.
//!
//! Fidelity alone never accepts. An operator that interpolates the teacher still
//! loses when its code is longer.

use std::fmt;

use super::codec::code_saving_at_proven_fidelity;
use super::precision::DecodedFidelity;
use super::supports::{EvidenceStatus, Extremum};

/// A structural proposal on the current artifact.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProposalKind {
    /// Two families of one shape class and label-manifold type become one field, or
    /// tied uses call one shared body.
    Share,
    /// A component becomes two whose sum is that component at the start.
    Split,
    /// One more basis function or label dimension, starting at its prior mean.
    Refine,
    /// A component, a rank or a basis function is removed, or a tensor returns to its
    /// native primitive.
    Reduce,
    /// A component is born from the residual at a violating witness, or a recovered
    /// structured coordinate becomes a program node.
    Expose,
}

/// An accepted structural proposal.
#[derive(Clone, Debug, PartialEq)]
pub struct ProposalAcceptance<W, D> {
    pub kind: ProposalKind,
    /// `L(reference) − L(candidate)` in bits. Always strictly positive.
    pub saving_bits: i128,
    /// Whether the fidelity status proves `sup d ≤ ε` over its domain. When false, no
    /// violation was proved either, and the status is carried as it was proved.
    pub fidelity_certified: bool,
    /// The decoded candidate's fidelity status over the declared mask domain.
    pub fidelity: EvidenceStatus<W, D>,
}

/// Why a structural proposal was not accepted.
#[derive(Clone, Debug, PartialEq)]
pub enum ProposalRejection<W, D> {
    /// The decoded reference misses the declared tolerance. The declared precision
    /// and tolerance are then inconsistent, so the loop is refused, not only this
    /// proposal.
    ReferenceMissesTolerance(String),
    /// The decoded candidate does not prove it meets the reference's declared tolerance:
    /// its verdict is not `Meets`, or its tolerance differs bitwise.
    CandidateMissesTolerance(String),
    /// The candidate's code is not strictly shorter.
    NoShorterCode { saving_bits: i128 },
    /// The fidelity status proves the supremum exceeds the tolerance.
    FidelityRefuted(EvidenceStatus<W, D>),
    /// A statistical estimate is about a mean, not the supremum the fidelity check
    /// needs (A6).
    EstimateIsNotAFidelityBound(EvidenceStatus<W, D>),
    /// The status brackets an infimum. Bounding an infimum from above says nothing
    /// about the supremum.
    NotASupremum(EvidenceStatus<W, D>),
}

impl<W: fmt::Debug, D: fmt::Debug> fmt::Display for ProposalRejection<W, D> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ReferenceMissesTolerance(message) => write!(
                f,
                "structural loop refused: the decoded reference misses the declared tolerance, so \
                 the declared precision and tolerance are inconsistent ({message})"
            ),
            Self::CandidateMissesTolerance(message) => {
                write!(f, "proposal rejected: the decoded candidate misses the tolerance ({message})")
            }
            Self::NoShorterCode { saving_bits } => write!(
                f,
                "proposal rejected: the candidate's code is not strictly shorter (saving \
                 {saving_bits} bits)"
            ),
            Self::FidelityRefuted(status) => {
                write!(f, "proposal rejected: the fidelity status refutes the tolerance: {status:?}")
            }
            Self::EstimateIsNotAFidelityBound(status) => write!(
                f,
                "proposal refused: a statistical estimate bounds no supremum: {status:?}"
            ),
            Self::NotASupremum(status) => write!(
                f,
                "proposal refused: the fidelity status brackets an infimum, not the supremum: \
                 {status:?}"
            ),
        }
    }
}

impl<W: fmt::Debug, D: fmt::Debug> std::error::Error for ProposalRejection<W, D> {}

/// Decides one structural proposal.
///
/// `reference` and `candidate` pair each decoded artifact's exact code length in bits
/// with its decoded distortion evidence under the declared tolerance, as the precision
/// owner states it (`precision::decode_then_evaluate`). `fidelity` is the separation
/// oracle's status for `sup d` over the declared mask domain, on the decoded candidate.
/// It is a different quantity from the decoded distortion, read at the reference's
/// declared tolerance.
///
/// The rule reuses the owners' predicates (`code_saving_at_proven_fidelity`,
/// `EvidenceStatus::refutes_at_most` and `certifies_at_most`) and writes no second
/// comparison:
/// 1. The decoded reference must prove it meets its tolerance; otherwise the loop is
///    refused.
/// 2. An estimate or an infimum bracket is refused. A status that refutes
///    `sup d ≤ tolerance` rejects the candidate.
/// 3. The decoded candidate must prove it meets the same tolerance, bitwise, and its
///    code must be strictly shorter.
pub fn decide_proposal<W, D, V, E>(
    kind: ProposalKind,
    reference: (u64, &DecodedFidelity<V, E>),
    candidate: (u64, &DecodedFidelity<V, E>),
    fidelity: EvidenceStatus<W, D>,
) -> Result<ProposalAcceptance<W, D>, ProposalRejection<W, D>> {
    code_saving_at_proven_fidelity(reference, reference)
        .map_err(ProposalRejection::ReferenceMissesTolerance)?;
    let tolerance = reference.1.tolerance();
    match fidelity {
        EvidenceStatus::StatisticalEstimate { .. } => {
            return Err(ProposalRejection::EstimateIsNotAFidelityBound(fidelity));
        }
        EvidenceStatus::Unresolved {
            extremum: Extremum::Infimum,
            ..
        } => return Err(ProposalRejection::NotASupremum(fidelity)),
        EvidenceStatus::Exact { .. }
        | EvidenceStatus::UniformBound { .. }
        | EvidenceStatus::Counterexample { .. }
        | EvidenceStatus::Unresolved { .. } => {}
    }
    if fidelity.refutes_at_most(tolerance) {
        return Err(ProposalRejection::FidelityRefuted(fidelity));
    }
    let saving_bits = code_saving_at_proven_fidelity(reference, candidate)
        .map_err(ProposalRejection::CandidateMissesTolerance)?;
    if saving_bits <= 0 {
        return Err(ProposalRejection::NoShorterCode { saving_bits });
    }
    Ok(ProposalAcceptance {
        kind,
        saving_bits,
        fidelity_certified: fidelity.certifies_at_most(tolerance),
        fidelity,
    })
}

#[cfg(test)]
mod proposal_tests {
    use super::*;
    use crate::precision::{DecodableArtifact, decode_then_evaluate};
    use crate::supports::ExactBasis;

    /// The declared fidelity tolerance of these fixtures.
    const TOLERANCE: f64 = 0.1;

    type Status = EvidenceStatus<Vec<f64>, &'static str>;

    type Fidelity = DecodedFidelity<Vec<f64>, &'static str>;

    /// A decoded figure whose artifact is its own output, so its evidence is exact.
    struct Figure(f64);

    impl DecodableArtifact for Figure {
        type Decoded = f64;

        fn decode(&self) -> Result<f64, String> {
            Ok(self.0)
        }
    }

    /// Decoded distortion evidence under `tolerance`, built through the precision owner:
    /// the output's exact distance from a native reference of zero, with a stated rounding
    /// bound, over a one-member input family.
    fn decoded(distortion: f64, numerical_error: f64, tolerance: f64) -> Fidelity {
        decode_then_evaluate(
            &Figure(distortion),
            |value: &f64| Ok(*value),
            &0.0,
            |outputs: &f64, native: &f64| {
                EvidenceStatus::exact(
                    *outputs - *native,
                    numerical_error,
                    ExactBasis::Exhaustive { cardinality: 1 },
                    None,
                    "declared input family",
                )
                .map_err(|error| error.to_string())
            },
            tolerance,
        )
        .expect("a valid declared distortion")
    }

    fn certified() -> Status {
        EvidenceStatus::uniform_bound(0.09, 0.001, "declared mask box").expect("a valid uniform bound")
    }

    #[test]
    fn a_shorter_certified_candidate_is_accepted_and_code_that_is_not_shorter_is_rejected() {
        let reference = decoded(0.05, 0.001, TOLERANCE);
        let candidate = decoded(0.08, 0.001, TOLERANCE);
        let accepted =
            decide_proposal(ProposalKind::Split, (1000, &reference), (900, &candidate), certified());
        assert!(
            matches!(
                &accepted,
                Ok(ProposalAcceptance { kind: ProposalKind::Split, saving_bits: 100, fidelity_certified: true, .. })
            ),
            "a 100-bit shorter certified candidate must be accepted, got {accepted:?}"
        );
        let equal =
            decide_proposal(ProposalKind::Share, (1000, &reference), (1000, &candidate), certified());
        assert!(
            matches!(equal, Err(ProposalRejection::NoShorterCode { saving_bits: 0 })),
            "equal code is not a strict decrease, got {equal:?}"
        );
        // An operator that interpolates the teacher at equal fidelity with a longer code
        // (mpd-modadd's rank caveat) loses on code alone.
        let interpolating_fidelity = decoded(0.05, 0.001, TOLERANCE);
        let interpolating = decide_proposal(
            ProposalKind::Expose,
            (1000, &reference),
            (1200, &interpolating_fidelity),
            certified(),
        );
        assert!(
            matches!(interpolating, Err(ProposalRejection::NoShorterCode { saving_bits: -200 })),
            "a longer interpolating operator must lose on code, got {interpolating:?}"
        );
    }

    #[test]
    fn a_refuting_estimated_or_infimum_fidelity_status_rejects_even_a_shorter_candidate() {
        let reference = decoded(0.05, 0.001, TOLERANCE);
        let shorter = decoded(0.08, 0.001, TOLERANCE);
        let mask = vec![1.0, 0.0, 1.0];
        let refuting: [Status; 3] = [
            EvidenceStatus::counterexample(0.3, 0.001, TOLERANCE, mask.clone()).expect("a violation"),
            EvidenceStatus::exact(0.2, 0.0, ExactBasis::Exhaustive { cardinality: 8 }, None, "binary masks")
                .expect("a valid exact value"),
            EvidenceStatus::unresolved(0.12, f64::INFINITY, Extremum::Supremum, Some(mask.clone()), "mask box")
                .expect("a valid bracket"),
        ];
        for status in refuting {
            let decision =
                decide_proposal(ProposalKind::Reduce, (1000, &reference), (900, &shorter), status);
            assert!(
                matches!(decision, Err(ProposalRejection::FidelityRefuted(..))),
                "a status whose lower bound exceeds the tolerance must reject, got {decision:?}"
            );
        }

        let estimate: Status =
            EvidenceStatus::statistical_estimate(0.02, 0.001, 64, "iid uniform masks").expect("a valid estimate");
        let estimated =
            decide_proposal(ProposalKind::Refine, (1000, &reference), (900, &shorter), estimate);
        assert!(
            matches!(estimated, Err(ProposalRejection::EstimateIsNotAFidelityBound(..))),
            "a stochastic-mask mean must never stand in for the supremum, got {estimated:?}"
        );
        // Positive control: the same figure as a uniform bound is accepted.
        let bound: Status =
            EvidenceStatus::uniform_bound(0.02, 0.001, "declared mask box").expect("a valid uniform bound");
        let bounded = decide_proposal(ProposalKind::Refine, (1000, &reference), (900, &shorter), bound);
        assert!(bounded.is_ok(), "the same figure as a uniform bound must be accepted, got {bounded:?}");

        let infimum =
            EvidenceStatus::unresolved(0.04, 0.09, Extremum::Infimum, Some(mask.clone()), "mask box").expect("a valid bracket");
        let infimum_decision =
            decide_proposal(ProposalKind::Expose, (1000, &reference), (900, &shorter), infimum);
        assert!(
            matches!(infimum_decision, Err(ProposalRejection::NotASupremum(..))),
            "an infimum bracket certifies nothing about the supremum, got {infimum_decision:?}"
        );
        // Positive control: the same bracket about the supremum certifies.
        let supremum = decide_proposal(
            ProposalKind::Expose,
            (1000, &reference),
            (900, &shorter),
            EvidenceStatus::unresolved(0.04, 0.09, Extremum::Supremum, Some(mask), "mask box").expect("a valid bracket"),
        );
        assert!(
            matches!(&supremum, Ok(acceptance) if acceptance.fidelity_certified),
            "the same bracket about the supremum must certify, got {supremum:?}"
        );
    }

    #[test]
    fn an_unrefuted_uncertified_fidelity_is_accepted_without_a_certificate() {
        let reference = decoded(0.05, 0.001, TOLERANCE);
        let candidate = decoded(0.08, 0.001, TOLERANCE);
        let decision = decide_proposal(
            ProposalKind::Split,
            (1000, &reference),
            (900, &candidate),
            EvidenceStatus::unresolved(0.04, f64::INFINITY, Extremum::Supremum, Some(vec![1.0, 1.0]), "mask box")
                .expect("a valid bracket"),
        );
        assert!(
            matches!(
                &decision,
                Ok(acceptance) if !acceptance.fidelity_certified
                    && matches!(acceptance.fidelity, EvidenceStatus::Unresolved { .. })
            ),
            "a lower witness below the tolerance with no derived upper bound is accepted as \
             uncertified, got {decision:?}"
        );
    }

    #[test]
    fn the_decoded_reference_and_candidate_must_meet_the_tolerance() {
        let shorter = decoded(0.08, 0.001, TOLERANCE);
        let violating_reference = decoded(0.2, 0.001, TOLERANCE);
        let inconsistent = decide_proposal(
            ProposalKind::Split,
            (1000, &violating_reference),
            (900, &shorter),
            certified(),
        );
        assert!(
            matches!(inconsistent, Err(ProposalRejection::ReferenceMissesTolerance(..))),
            "a decoded reference that misses the tolerance must refuse the loop, got {inconsistent:?}"
        );
        let reference = decoded(0.05, 0.001, TOLERANCE);
        // 0.099 ± 0.002 brackets the tolerance, so its verdict is Unresolved, never a pass.
        let unresolved = decoded(0.099, 0.002, TOLERANCE);
        let missing =
            decide_proposal(ProposalKind::Split, (1000, &reference), (900, &unresolved), certified());
        assert!(
            matches!(missing, Err(ProposalRejection::CandidateMissesTolerance(..))),
            "a decoded candidate that does not prove it meets the tolerance must be rejected, got {missing:?}"
        );
        // Positive control: 0.097 + 0.002 = 0.099 is inside the tolerance by far more than one
        // ulp, so the outcome does not rest on rounding.
        let meeting_fidelity = decoded(0.097, 0.002, TOLERANCE);
        let meeting =
            decide_proposal(ProposalKind::Split, (1000, &reference), (900, &meeting_fidelity), certified());
        assert!(meeting.is_ok(), "a candidate at the tolerance must be accepted, got {meeting:?}");

        // A candidate declared at a different tolerance is refused, even though it meets its own.
        let other_tolerance = decoded(0.05, 0.001, 0.2);
        let mismatched =
            decide_proposal(ProposalKind::Split, (1000, &reference), (900, &other_tolerance), certified());
        assert!(
            matches!(mismatched, Err(ProposalRejection::CandidateMissesTolerance(..))),
            "a candidate scored at another tolerance must be refused, got {mismatched:?}"
        );
        // Positive control: the same figure at the reference's tolerance is accepted.
        let same_tolerance = decoded(0.05, 0.001, TOLERANCE);
        let matched =
            decide_proposal(ProposalKind::Split, (1000, &reference), (900, &same_tolerance), certified());
        assert!(matched.is_ok(), "the same figure at the reference's tolerance must be accepted, got {matched:?}");
    }
}
