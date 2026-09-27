//! `code_lengths` and `decide_proposal`: the `codec`, `precision` and
//! `fit::decide_proposal` owners on the wire, so a caller's search loop prices its
//! items with Rust's codes and decides each proposal with Rust's rule.

use std::collections::BTreeMap;

use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, ArrayD};
use serde::{Deserialize, Serialize};

use super::{MpdOutput, MpdResult, MpdSurfaceError, finite, input, output, reserve};
use crate::parameter_decomposition::codec::{
    BitString, CodecError, fixed_index_len_bits, prefix_integer_len_bits,
    signed_prefix_integer_len_bits, subset_code_len_bits,
};
use crate::parameter_decomposition::fit::{
    ProposalAcceptance, ProposalKind, ProposalRejection, decide_proposal,
};
use crate::parameter_decomposition::precision::{
    DecodableArtifact, DecodedFidelity, DeclaredPrecision, FidelityVerdict, LatticeCode,
    decode_then_evaluate,
};
use crate::parameter_decomposition::supports::{
    EvidenceStatus, EvidenceStatusError, ExactBasis, Extremum,
};

/// The items to price, in order.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CodeLengthsRequest {
    pub items: Vec<CodeItem>,
}

/// One item of a message, in the code its owner implements.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CodeItem {
    /// An integer `value ≥ 1` in the Elias omega code (`codec::prefix_integer_len_bits`).
    PrefixInteger { value: u64 },
    /// A signed integer in the zigzag prefix code
    /// (`codec::signed_prefix_integer_len_bits`).
    SignedPrefixInteger { value: i64 },
    /// An index into an alphabet of `alphabet_size ≥ 1` symbols the decoder knows
    /// (`codec::fixed_index_len_bits`).
    FixedIndex { alphabet_size: usize },
    /// A `cardinality`-subset of `universe` elements in the enumerative code
    /// (`codec::subset_code_len_bits`).
    Subset { universe: usize, cardinality: usize },
    /// The reals of an input array, in row-major order, as one lattice message at the
    /// precision `2^-fraction_bits` (`precision::LatticeCode::write`).
    Lattice { tensor: String, fraction_bits: i32 },
}

/// The exact bit count of every item, and their sum.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CodeLengthsReport {
    pub items: Vec<CodeLengthReport>,
    pub total_bits: u64,
}

/// One item's exact length.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CodeLengthReport {
    pub bits: u64,
    /// For a lattice item: what the decoder rebuilds.
    pub lattice: Option<LatticeReport>,
}

/// A lattice message's parts and its decoded reals.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct LatticeReport {
    /// The index codewords alone (`LatticeCode::index_bits`); `bits` adds the count and
    /// precision header.
    pub index_bits: u64,
    /// Id of the decoded reals, in the input array's shape: what fidelity must be
    /// measured on.
    pub decoded: String,
}

fn codec(error: CodecError) -> MpdSurfaceError {
    MpdSurfaceError::Code(error.to_string())
}

pub(super) fn run_lengths(
    request: CodeLengthsRequest,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let mut arrays = BTreeMap::new();
    let mut items = Vec::with_capacity(request.items.len());
    let mut total_bits = 0_u64;
    for (index, item) in request.items.iter().enumerate() {
        let (bits, lattice) = match item {
            CodeItem::PrefixInteger { value } => (prefix_integer_len_bits(*value).map_err(codec)?, None),
            CodeItem::SignedPrefixInteger { value } => {
                (signed_prefix_integer_len_bits(*value).map_err(codec)?, None)
            }
            CodeItem::FixedIndex { alphabet_size } => {
                (u64::from(fixed_index_len_bits(*alphabet_size).map_err(codec)?), None)
            }
            CodeItem::Subset {
                universe,
                cardinality,
            } => (subset_code_len_bits(*universe, *cardinality).map_err(codec)?, None),
            CodeItem::Lattice {
                tensor,
                fraction_bits,
            } => {
                let reals = input(tensors, tensor)?;
                let precision = DeclaredPrecision::new(*fraction_bits).map_err(MpdSurfaceError::Code)?;
                // The row-major reals, the indices and the decoded reals.
                let formed = reserve(governor, reals.len(), 1, 3, "lattice code")?;
                let values: Vec<f64> = reals.iter().copied().collect();
                let code = LatticeCode::encode(&values, precision).map_err(MpdSurfaceError::Code)?;
                let mut message = BitString::new();
                code.write(&mut message).map_err(MpdSurfaceError::Code)?;
                let index_bits = code.index_bits().map_err(MpdSurfaceError::Code)?;
                let decoded = Array1::from_vec(code.decode().map_err(MpdSurfaceError::Code)?)
                    .into_shape_with_order(reals.shape())
                    .map_err(|error| MpdSurfaceError::TensorShape {
                        tensor: tensor.clone(),
                        reason: error.to_string(),
                    })?;
                drop(formed);
                let id = format!("items/{index}/decoded");
                arrays.insert(id.clone(), decoded);
                (
                    message.len_bits(),
                    Some(LatticeReport {
                        index_bits,
                        decoded: id,
                    }),
                )
            }
        };
        total_bits = total_bits.checked_add(bits).ok_or_else(|| {
            MpdSurfaceError::Code("the total code length exceeds u64 bits".to_string())
        })?;
        items.push(CodeLengthReport { bits, lattice });
    }
    Ok(output(
        MpdResult::CodeLengths(CodeLengthsReport { items, total_bits }),
        arrays,
    ))
}

/// One structural proposal, with every length and evidence status supplied.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DecideProposalRequest {
    pub proposal: ProposalKindWire,
    /// The declared fidelity tolerance both decoded artifacts are read at.
    pub tolerance: f64,
    pub reference: StatedArtifact,
    pub candidate: StatedArtifact,
    /// The separation oracle's status for `sup d` over the declared mask domain, on
    /// the decoded candidate.
    pub fidelity: EvidenceStatusWire,
}

/// A decoded artifact's exact code length and the distortion evidence of its decoded
/// outputs. The caller decoded and executed the artifact; the status is its
/// statement, validated by the owner's constructor and never strengthened here.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StatedArtifact {
    pub bits: u64,
    pub distortion: EvidenceStatusWire,
}

/// [`ProposalKind`] on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ProposalKindWire {
    Share,
    Split,
    Refine,
    Reduce,
    Expose,
}

impl From<ProposalKindWire> for ProposalKind {
    fn from(kind: ProposalKindWire) -> Self {
        match kind {
            ProposalKindWire::Share => Self::Share,
            ProposalKindWire::Split => Self::Split,
            ProposalKindWire::Refine => Self::Refine,
            ProposalKindWire::Reduce => Self::Reduce,
            ProposalKindWire::Expose => Self::Expose,
        }
    }
}

impl From<ProposalKind> for ProposalKindWire {
    fn from(kind: ProposalKind) -> Self {
        match kind {
            ProposalKind::Share => Self::Share,
            ProposalKind::Split => Self::Split,
            ProposalKind::Refine => Self::Refine,
            ProposalKind::Reduce => Self::Reduce,
            ProposalKind::Expose => Self::Expose,
        }
    }
}

/// [`EvidenceStatus`] on the wire, with a named witness and a named domain. An
/// unresolved side that is not derived is absent (the owner's infinite side).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EvidenceStatusWire {
    Exact {
        value: f64,
        numerical_error: f64,
        basis: ExactBasisWire,
        witness: Option<String>,
        domain: String,
    },
    UniformBound {
        upper: f64,
        numerical_error: f64,
        region: String,
    },
    StatisticalEstimate {
        estimate: f64,
        standard_error: f64,
        samples: u64,
        law: String,
    },
    Counterexample {
        value: f64,
        numerical_error: f64,
        threshold: f64,
        witness: String,
    },
    Unresolved {
        lower: Option<f64>,
        upper: Option<f64>,
        extremum: ExtremumWire,
        witness: Option<String>,
        domain: String,
    },
}

/// [`ExactBasis`] on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ExactBasisWire {
    Algebraic {},
    Exhaustive { cardinality: u64 },
}

/// [`Extremum`] on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExtremumWire {
    Supremum,
    Infimum,
}

type Status = EvidenceStatus<String, String>;

impl EvidenceStatusWire {
    /// Builds the status through its owner's validating constructor.
    pub fn into_status(self) -> Result<Status, EvidenceStatusError> {
        match self {
            Self::Exact {
                value,
                numerical_error,
                basis,
                witness,
                domain,
            } => {
                let basis = match basis {
                    ExactBasisWire::Algebraic {} => ExactBasis::Algebraic,
                    ExactBasisWire::Exhaustive { cardinality } => ExactBasis::Exhaustive { cardinality },
                };
                EvidenceStatus::exact(value, numerical_error, basis, witness, domain)
            }
            Self::UniformBound {
                upper,
                numerical_error,
                region,
            } => EvidenceStatus::uniform_bound(upper, numerical_error, region),
            Self::StatisticalEstimate {
                estimate,
                standard_error,
                samples,
                law,
            } => EvidenceStatus::statistical_estimate(estimate, standard_error, samples, law),
            Self::Counterexample {
                value,
                numerical_error,
                threshold,
                witness,
            } => EvidenceStatus::counterexample(value, numerical_error, threshold, witness),
            Self::Unresolved {
                lower,
                upper,
                extremum,
                witness,
                domain,
            } => {
                let extremum = match extremum {
                    ExtremumWire::Supremum => Extremum::Supremum,
                    ExtremumWire::Infimum => Extremum::Infimum,
                };
                EvidenceStatus::unresolved(
                    lower.unwrap_or(f64::NEG_INFINITY),
                    upper.unwrap_or(f64::INFINITY),
                    extremum,
                    witness,
                    domain,
                )
            }
        }
    }

    /// The owner's status, with an underived (infinite) side absent.
    pub fn from_status(status: Status) -> Result<Self, MpdSurfaceError> {
        Ok(match status {
            EvidenceStatus::Exact {
                value,
                numerical_error,
                basis,
                witness,
                domain,
                ..
            } => Self::Exact {
                value,
                numerical_error,
                basis: match basis {
                    ExactBasis::Algebraic => ExactBasisWire::Algebraic {},
                    ExactBasis::Exhaustive { cardinality } => ExactBasisWire::Exhaustive { cardinality },
                },
                witness,
                domain,
            },
            EvidenceStatus::UniformBound {
                upper,
                numerical_error,
                region,
                ..
            } => Self::UniformBound {
                upper,
                numerical_error,
                region,
            },
            EvidenceStatus::StatisticalEstimate {
                estimate,
                standard_error,
                samples,
                law,
                ..
            } => Self::StatisticalEstimate {
                estimate,
                standard_error,
                samples,
                law,
            },
            EvidenceStatus::Counterexample {
                value,
                numerical_error,
                threshold,
                witness,
                ..
            } => Self::Counterexample {
                value,
                numerical_error,
                threshold,
                witness,
            },
            EvidenceStatus::Unresolved {
                lower,
                upper,
                extremum,
                witness,
                domain,
                ..
            } => Self::Unresolved {
                lower: if lower == f64::NEG_INFINITY {
                    None
                } else {
                    Some(finite("fidelity.lower", lower)?)
                },
                upper: if upper == f64::INFINITY {
                    None
                } else {
                    Some(finite("fidelity.upper", upper)?)
                },
                extremum: match extremum {
                    Extremum::Supremum => ExtremumWire::Supremum,
                    Extremum::Infimum => ExtremumWire::Infimum,
                },
                witness,
                domain,
            },
        })
    }
}

/// [`FidelityVerdict`] on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FidelityVerdictWire {
    Meets,
    Violates,
    Unresolved,
}

impl From<FidelityVerdict> for FidelityVerdictWire {
    fn from(verdict: FidelityVerdict) -> Self {
        match verdict {
            FidelityVerdict::Meets => Self::Meets,
            FidelityVerdict::Violates => Self::Violates,
            FidelityVerdict::Unresolved => Self::Unresolved,
        }
    }
}

/// The owner's decision, with each decoded artifact's verdict at the tolerance.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct DecideProposalReport {
    pub reference_verdict: FidelityVerdictWire,
    pub candidate_verdict: FidelityVerdictWire,
    pub decision: ProposalDecision,
}

/// [`ProposalAcceptance`] or [`ProposalRejection`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProposalDecision {
    Accepted {
        proposal: ProposalKindWire,
        /// `L(reference) − L(candidate)`, always strictly positive.
        saving_bits: i128,
        /// Whether the fidelity status proves `sup d ≤ tolerance`.
        fidelity_certified: bool,
        fidelity: EvidenceStatusWire,
    },
    /// The decoded reference misses the tolerance, so the loop itself is refused.
    ReferenceMissesTolerance { reason: String },
    CandidateMissesTolerance { reason: String },
    NoShorterCode { saving_bits: i128 },
    FidelityRefuted { fidelity: EvidenceStatusWire },
    EstimateIsNotAFidelityBound { fidelity: EvidenceStatusWire },
    NotASupremum { fidelity: EvidenceStatusWire },
}

/// A decoded artifact the caller executed itself: nothing is rebuilt here, and its
/// distortion is the status the caller states.
struct ExternallyDecoded;

impl DecodableArtifact for ExternallyDecoded {
    type Decoded = ();

    fn decode(&self) -> Result<(), String> {
        Ok(())
    }
}

/// The owner's `DecodedFidelity` of a stated status at the declared tolerance; the
/// owner refuses a tolerance that is not finite and non-negative.
fn stated_fidelity(
    status: Status,
    tolerance: f64,
) -> Result<DecodedFidelity<String, String>, MpdSurfaceError> {
    decode_then_evaluate(
        &ExternallyDecoded,
        |_| Ok(()),
        &(),
        |_, _| Ok(status),
        tolerance,
    )
    .map_err(MpdSurfaceError::InvalidRequest)
}

pub(super) fn run_decide(request: DecideProposalRequest) -> Result<MpdOutput, MpdSurfaceError> {
    let status = |wire: EvidenceStatusWire| wire.into_status().map_err(MpdSurfaceError::Evidence);
    let reference = stated_fidelity(status(request.reference.distortion)?, request.tolerance)?;
    let candidate = stated_fidelity(status(request.candidate.distortion)?, request.tolerance)?;
    let fidelity = status(request.fidelity)?;
    let decision = decide_proposal(
        request.proposal.into(),
        (request.reference.bits, &reference),
        (request.candidate.bits, &candidate),
        fidelity,
    );
    project_decision(reference.verdict(), candidate.verdict(), decision)
}

pub(super) fn project_decision(
    reference: FidelityVerdict,
    candidate: FidelityVerdict,
    decision: Result<ProposalAcceptance<String, String>, ProposalRejection<String, String>>,
) -> Result<MpdOutput, MpdSurfaceError> {
    let decision = match decision {
        Ok(accepted) => ProposalDecision::Accepted {
            proposal: accepted.kind.into(),
            saving_bits: accepted.saving_bits,
            fidelity_certified: accepted.fidelity_certified,
            fidelity: EvidenceStatusWire::from_status(accepted.fidelity)?,
        },
        Err(ProposalRejection::ReferenceMissesTolerance(reason)) => {
            ProposalDecision::ReferenceMissesTolerance { reason }
        }
        Err(ProposalRejection::CandidateMissesTolerance(reason)) => {
            ProposalDecision::CandidateMissesTolerance { reason }
        }
        Err(ProposalRejection::NoShorterCode { saving_bits }) => {
            ProposalDecision::NoShorterCode { saving_bits }
        }
        Err(ProposalRejection::FidelityRefuted(status)) => ProposalDecision::FidelityRefuted {
            fidelity: EvidenceStatusWire::from_status(status)?,
        },
        Err(ProposalRejection::EstimateIsNotAFidelityBound(status)) => {
            ProposalDecision::EstimateIsNotAFidelityBound {
                fidelity: EvidenceStatusWire::from_status(status)?,
            }
        }
        Err(ProposalRejection::NotASupremum(status)) => ProposalDecision::NotASupremum {
            fidelity: EvidenceStatusWire::from_status(status)?,
        },
    };
    Ok(output(
        MpdResult::DecideProposal(DecideProposalReport {
            reference_verdict: reference.into(),
            candidate_verdict: candidate.into(),
            decision,
        }),
        BTreeMap::new(),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::run_parameter_decomposition;
    use super::super::tests::request_json;
    use super::*;
    use crate::parameter_decomposition::codec::{encode_prefix_integer, encode_subset};
    use crate::parameter_decomposition::test_support::test_governor;
    use ndarray::array;

    fn lengths(items: &str, tensors: &BTreeMap<String, ArrayD<f64>>) -> Result<MpdOutput, MpdSurfaceError> {
        run_parameter_decomposition(
            &request_json(&format!(r#"{{"kind": "code_lengths", "items": {items}}}"#)),
            tensors,
            test_governor(),
        )
    }

    #[test]
    fn code_lengths_report_is_the_owners_lengths_item_for_item() {
        let reals = array![[0.3, -1.7], [2.25, 0.0]];
        let tensors = BTreeMap::from([("x".to_string(), reals.clone().into_dyn())]);
        let output = lengths(
            r#"[{"kind": "prefix_integer", "value": 1000},
                {"kind": "signed_prefix_integer", "value": -5},
                {"kind": "fixed_index", "alphabet_size": 113},
                {"kind": "subset", "universe": 56, "cardinality": 5},
                {"kind": "lattice", "tensor": "x", "fraction_bits": 4}]"#,
            &tensors,
        )
        .expect("surface run");
        let MpdResult::CodeLengths(report) = &output.report.result else {
            panic!("expected code lengths, got {:?}", output.report.result);
        };
        let precision = DeclaredPrecision::new(4).expect("precision");
        let code = LatticeCode::encode(&[0.3, -1.7, 2.25, 0.0], precision).expect("encode");
        let mut message = BitString::new();
        code.write(&mut message).expect("write");
        let expected = [
            prefix_integer_len_bits(1000).expect("prefix"),
            signed_prefix_integer_len_bits(-5).expect("signed"),
            u64::from(fixed_index_len_bits(113).expect("index")),
            subset_code_len_bits(56, 5).expect("subset"),
            message.len_bits(),
        ];
        assert_eq!(
            report.items.iter().map(|item| item.bits).collect::<Vec<_>>(),
            expected
        );
        assert_eq!(report.total_bits, expected.iter().sum::<u64>());
        // The lengths are those of messages the encoders write.
        let mut written = BitString::new();
        encode_prefix_integer(&mut written, 1000).expect("encode prefix");
        assert_eq!(written.len_bits(), expected[0]);
        let mut subset = BitString::new();
        encode_subset(&mut subset, 56, &[1, 7, 20, 33, 55]).expect("encode subset");
        assert_eq!(subset.len_bits(), expected[3]);
        let lattice = report.items[4].lattice.as_ref().expect("lattice report");
        assert_eq!(lattice.index_bits, code.index_bits().expect("index bits"));
        let decoded = code.decode().expect("decode");
        assert_eq!(
            output.arrays[&lattice.decoded],
            Array1::from_vec(decoded).into_shape_with_order(vec![2, 2]).expect("shape")
        );
        assert!(report.items[..4].iter().all(|item| item.lattice.is_none()));
        assert_eq!(output.arrays.len(), 1);
    }

    #[test]
    fn a_code_item_its_owner_refuses_reaches_the_caller() {
        let tensors = BTreeMap::from([("x".to_string(), array![1.0].into_dyn())]);
        assert!(lengths(r#"[{"kind": "prefix_integer", "value": 1}]"#, &tensors).is_ok());
        for refused in [
            r#"[{"kind": "prefix_integer", "value": 0}]"#,
            r#"[{"kind": "fixed_index", "alphabet_size": 0}]"#,
            r#"[{"kind": "subset", "universe": 3, "cardinality": 4}]"#,
            r#"[{"kind": "lattice", "tensor": "x", "fraction_bits": 4000}]"#,
            r#"[{"kind": "signed_prefix_integer", "value": -9223372036854775808}]"#,
        ] {
            assert!(
                matches!(lengths(refused, &tensors), Err(MpdSurfaceError::Code(_))),
                "{refused}"
            );
        }
        assert!(matches!(
            lengths(r#"[{"kind": "prefix_integer", "value": 3, "bits": 2}]"#, &tensors),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
        assert!(matches!(
            lengths(r#"[{"kind": "lattice", "tensor": "y", "fraction_bits": 2}]"#, &tensors),
            Err(MpdSurfaceError::MissingTensor { .. })
        ));
    }

    fn exact(value: f64) -> String {
        format!(
            r#"{{"kind": "exact", "value": {value}, "numerical_error": 0.0, "basis": {{"kind": "exhaustive", "cardinality": 4}}, "witness": null, "domain": "all inputs"}}"#
        )
    }

    fn decide(reference_bits: u64, candidate_bits: u64, candidate: &str, fidelity: &str) -> String {
        request_json(&format!(
            r#"{{"kind": "decide_proposal", "proposal": "expose", "tolerance": 0.5,
                "reference": {{"bits": {reference_bits}, "distortion": {}}},
                "candidate": {{"bits": {candidate_bits}, "distortion": {candidate}}},
                "fidelity": {fidelity}}}"#,
            exact(0.1)
        ))
    }

    fn decision(json: &str) -> DecideProposalReport {
        let output = run_parameter_decomposition(json, &BTreeMap::new(), test_governor()).expect("surface run");
        let MpdResult::DecideProposal(report) = output.report.result else {
            panic!("expected a decision, got {:?}", output.report.result);
        };
        report
    }

    fn owner_fidelity(value: f64) -> DecodedFidelity<String, String> {
        stated_fidelity(
            EvidenceStatus::exact(value, 0.0, ExactBasis::Exhaustive { cardinality: 4 }, None, "all inputs".to_string())
                .expect("status"),
            0.5,
        )
        .expect("fidelity")
    }

    #[test]
    fn decide_proposal_report_is_the_owner_decision_field_for_field() {
        let owner_status = |value| {
            EvidenceStatus::exact(value, 0.0, ExactBasis::Exhaustive { cardinality: 4 }, None, "all inputs".to_string())
                .expect("status")
        };
        let cases = [
            (100, 60, 0.2, 0.3),
            (100, 100, 0.2, 0.3),
            (100, 60, 0.7, 0.3),
            (100, 60, 0.2, 0.9),
        ];
        for (reference_bits, candidate_bits, candidate_value, fidelity_value) in cases {
            let owner = decide_proposal(
                ProposalKind::Expose,
                (reference_bits, &owner_fidelity(0.1)),
                (candidate_bits, &owner_fidelity(candidate_value)),
                owner_status(fidelity_value),
            );
            let expected = match project_decision(
                owner_fidelity(0.1).verdict(),
                owner_fidelity(candidate_value).verdict(),
                owner,
            )
            .expect("projection")
            .report
            .result
            {
                MpdResult::DecideProposal(report) => report,
                other => panic!("expected a decision, got {other:?}"),
            };
            let wire = decision(&decide(
                reference_bits,
                candidate_bits,
                &exact(candidate_value),
                &exact(fidelity_value),
            ));
            assert_eq!(wire, expected);
        }
        // The four cases are the four outcomes, so the projection was exercised on each.
        let accepted = decision(&decide(100, 60, &exact(0.2), &exact(0.3)));
        assert_eq!(accepted.candidate_verdict, FidelityVerdictWire::Meets);
        assert!(matches!(
            accepted.decision,
            ProposalDecision::Accepted { saving_bits: 40, fidelity_certified: true, proposal: ProposalKindWire::Expose, .. }
        ));
        assert!(matches!(
            decision(&decide(100, 100, &exact(0.2), &exact(0.3))).decision,
            ProposalDecision::NoShorterCode { saving_bits: 0 }
        ));
        let missed = decision(&decide(100, 60, &exact(0.7), &exact(0.3)));
        assert_eq!(missed.candidate_verdict, FidelityVerdictWire::Violates);
        assert!(matches!(missed.decision, ProposalDecision::CandidateMissesTolerance { .. }));
        assert!(matches!(
            decision(&decide(100, 60, &exact(0.2), &exact(0.9))).decision,
            ProposalDecision::FidelityRefuted { .. }
        ));
        let json: serde_json::Value = serde_json::from_str(
            &run_parameter_decomposition(&decide(100, 60, &exact(0.2), &exact(0.3)), &BTreeMap::new(), test_governor())
                .expect("surface run")
                .report_json()
                .expect("json"),
        )
        .expect("parse");
        assert_eq!(json["result"]["decision"]["kind"], "accepted");
        assert_eq!(json["result"]["decision"]["saving_bits"], 40);
    }

    #[test]
    fn statuses_that_bound_no_supremum_are_refused_by_the_owner_rule() {
        let estimate = r#"{"kind": "statistical_estimate", "estimate": 0.1, "standard_error": 0.01, "samples": 10, "law": "uniform masks"}"#;
        assert!(matches!(
            decision(&decide(100, 60, &exact(0.2), estimate)).decision,
            ProposalDecision::EstimateIsNotAFidelityBound { .. }
        ));
        let infimum = r#"{"kind": "unresolved", "lower": null, "upper": 0.2, "extremum": "infimum", "witness": null, "domain": "masks"}"#;
        assert!(matches!(
            decision(&decide(100, 60, &exact(0.2), infimum)).decision,
            ProposalDecision::NotASupremum { .. }
        ));
        // An unresolved supremum below the tolerance on its upper side is certified; its
        // underived lower side stays absent on the way back.
        let bracket = r#"{"kind": "unresolved", "lower": null, "upper": 0.4, "extremum": "supremum", "witness": null, "domain": "masks"}"#;
        let ProposalDecision::Accepted { fidelity, fidelity_certified, .. } =
            decision(&decide(100, 60, &exact(0.2), bracket)).decision
        else {
            panic!("an upper bound below the tolerance is accepted");
        };
        assert!(fidelity_certified);
        assert!(matches!(fidelity, EvidenceStatusWire::Unresolved { lower: None, upper: Some(0.4), .. }));
        // The loop is refused when the reference misses its tolerance.
        let reference_misses = decide(100, 60, &exact(0.2), &exact(0.3)).replacen(&exact(0.1), &exact(0.8), 1);
        assert!(matches!(
            decision(&reference_misses).decision,
            ProposalDecision::ReferenceMissesTolerance { .. }
        ));
    }

    #[test]
    fn a_decision_request_the_owners_refuse_reaches_the_caller() {
        let good = decide(100, 60, &exact(0.2), &exact(0.3));
        assert!(run_parameter_decomposition(&good, &BTreeMap::new(), test_governor()).is_ok());
        // A counterexample that roundoff explains, an inverted bracket, a negative
        // tolerance and a stray field.
        let not_violation = r#"{"kind": "counterexample", "value": 0.5, "numerical_error": 0.1, "threshold": 0.5, "witness": "mask 3"}"#;
        assert!(matches!(
            run_parameter_decomposition(&decide(100, 60, &exact(0.2), not_violation), &BTreeMap::new(), test_governor()),
            Err(MpdSurfaceError::Evidence(EvidenceStatusError::NotAViolation { .. }))
        ));
        let inverted = r#"{"kind": "unresolved", "lower": 0.4, "upper": 0.3, "extremum": "supremum", "witness": null, "domain": "masks"}"#;
        assert!(matches!(
            run_parameter_decomposition(&decide(100, 60, &exact(0.2), inverted), &BTreeMap::new(), test_governor()),
            Err(MpdSurfaceError::Evidence(EvidenceStatusError::InvertedInterval { .. }))
        ));
        let negative = good.replacen("\"tolerance\": 0.5", "\"tolerance\": -0.5", 1);
        assert!(matches!(
            run_parameter_decomposition(&negative, &BTreeMap::new(), test_governor()),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
        let stray = good.replacen("\"tolerance\"", "\"saving_bits\": 3, \"tolerance\"", 1);
        assert!(matches!(
            run_parameter_decomposition(&stray, &BTreeMap::new(), test_governor()),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
        let algebraic_with_field = good.replace(
            r#"{"kind": "exhaustive", "cardinality": 4}"#,
            r#"{"kind": "algebraic", "cardinality": 4}"#,
        );
        assert!(matches!(
            run_parameter_decomposition(&algebraic_with_field, &BTreeMap::new(), test_governor()),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
    }
}
