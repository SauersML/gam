//! `verify_logits`: the exhaustive verification and counterfactual contract of
//! [`crate::parameter_decomposition::verify`] over logits the caller executed, each with
//! its per-entry forward-error radius.

use std::collections::BTreeMap;
use std::convert::Infallible;

use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array2, ArrayD, ArrayView4, Axis, Ix4, s};
use serde::{Deserialize, Serialize};

use super::code::EvidenceStatusWire;
use super::{MpdOutput, MpdResult, MpdSurfaceError, finite, input, output, reserve};
use crate::parameter_decomposition::secant::BandedMatrix;
use crate::parameter_decomposition::verify::{
    CounterfactualContract, FamilyDomain, FamilyVerification, FamilyWitness, Tolerance,
    verify_counterfactual_contract,
};

/// Reference and candidate logits over a declared finite family under every declared
/// intervention. Each array is `interventions × inputs × rows × classes`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct VerifyLogitsRequest {
    pub reference: BandedLogits,
    pub candidate: BandedLogits,
    pub tolerance: ToleranceRequest,
}

/// Logits and their per-entry radius against the exact values, as the executor derived it.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BandedLogits {
    pub logits: String,
    pub radius: String,
}

/// [`Tolerance`] on the wire: experiment declarations with no default.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToleranceRequest {
    pub kl: f64,
    pub centred_logit_gap: f64,
}

/// [`CounterfactualContract`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct VerifyLogitsReport {
    /// Every intervention's family is proven within the tolerance.
    pub certified: bool,
    pub interventions: Vec<FamilyVerificationReport>,
}

/// [`FamilyWitness`] on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct FamilyWitnessReport {
    pub input: usize,
    pub row: usize,
}

/// [`FamilyDomain`] on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct FamilyDomainReport {
    pub intervention: Option<usize>,
    pub inputs: usize,
    pub rows: usize,
}

pub type FamilyStatusWire = EvidenceStatusWire<FamilyWitnessReport, FamilyDomainReport>;

/// [`FamilyVerification`] on the wire; its per-row comparisons are arrays.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FamilyVerificationReport {
    pub certified: bool,
    pub forward_kl: FamilyStatusWire,
    pub reverse_kl: FamilyStatusWire,
    pub centred_logit_gap: FamilyStatusWire,
    pub argmax_rows: u64,
    pub argmax_disagreeing: u64,
    pub argmax_certified_rows: u64,
    pub argmax_disagreeing_fraction: FamilyStatusWire,
    /// Ids of `inputs × rows` arrays: each row's computed argmaxes, whether both are
    /// certified (1 or 0), and its largest centred-logit gap with its band.
    pub reference_argmax: String,
    pub candidate_argmax: String,
    pub argmax_certified: String,
    pub centred_gap: String,
    pub centred_gap_band: String,
}

fn witness(witness: FamilyWitness) -> FamilyWitnessReport {
    FamilyWitnessReport {
        input: witness.input,
        row: witness.row,
    }
}

fn domain(domain: FamilyDomain) -> FamilyDomainReport {
    FamilyDomainReport {
        intervention: domain.intervention,
        inputs: domain.inputs,
        rows: domain.rows,
    }
}

/// The caller's logits at one intervention and input, with their radius: an executable
/// that only reads what was supplied.
pub(super) fn supplied<'a>(
    logits: ArrayView4<'a, f64>,
    radius: ArrayView4<'a, f64>,
) -> impl Fn(&MemoryGovernor, &usize, &usize) -> Result<BandedMatrix, Infallible> + 'a {
    move |_, intervention, input| {
        Ok(BandedMatrix {
            values: logits.slice(s![*intervention, *input, .., ..]).to_owned(),
            bands: radius.slice(s![*intervention, *input, .., ..]).to_owned(),
        })
    }
}

fn four<'a>(
    tensors: &'a BTreeMap<String, ArrayD<f64>>,
    id: &str,
) -> Result<ArrayView4<'a, f64>, MpdSurfaceError> {
    let array = input(tensors, id)?;
    array
        .view()
        .into_dimensionality::<Ix4>()
        .map_err(|error| MpdSurfaceError::TensorShape {
            tensor: id.to_string(),
            reason: format!(
                "expected interventions x inputs x rows x classes, got shape {:?}: {error}",
                array.shape()
            ),
        })
}

pub(super) fn run(
    request: VerifyLogitsRequest,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let reference = (four(tensors, &request.reference.logits)?, four(tensors, &request.reference.radius)?);
    let candidate = (four(tensors, &request.candidate.logits)?, four(tensors, &request.candidate.radius)?);
    let (interventions, inputs) = (reference.0.len_of(Axis(0)), reference.0.len_of(Axis(1)));
    for (id, array) in [
        (&request.reference.radius, reference.1),
        (&request.candidate.logits, candidate.0),
        (&request.candidate.radius, candidate.1),
    ] {
        if array.len_of(Axis(0)) != interventions || array.len_of(Axis(1)) != inputs {
            return Err(MpdSurfaceError::TensorShape {
                tensor: id.clone(),
                reason: format!(
                    "{interventions} interventions and {inputs} inputs expected, got shape {:?}",
                    array.shape()
                ),
            });
        }
    }
    // One input's four logit blocks at a time.
    let (rows, classes) = (reference.0.len_of(Axis(2)), reference.0.len_of(Axis(3)));
    let working = reserve(governor, rows.max(1), classes.max(1), 4, "verify logits: one input")?;
    let tolerance = Tolerance {
        kl: request.tolerance.kl,
        centred_logit_gap: request.tolerance.centred_logit_gap,
    };
    let intervention_indices: Vec<usize> = (0..interventions).collect();
    let family: Vec<usize> = (0..inputs).collect();
    let contract = verify_counterfactual_contract(
        governor,
        &supplied(reference.0, reference.1),
        &supplied(candidate.0, candidate.1),
        &intervention_indices,
        &family,
        &tolerance,
    )
    .map_err(|error| MpdSurfaceError::Verify(error.to_string()))?;
    drop(working);
    project(contract, &tolerance, inputs)
}

pub(super) fn project(
    contract: CounterfactualContract,
    tolerance: &Tolerance,
    inputs: usize,
) -> Result<MpdOutput, MpdSurfaceError> {
    let certified = contract.certified_within(tolerance);
    let mut arrays = BTreeMap::new();
    let mut reports = Vec::with_capacity(contract.interventions.len());
    for (k, family) in contract.interventions.into_iter().enumerate() {
        let family_certified = family.certified_within(tolerance);
        let FamilyVerification {
            forward_kl,
            reverse_kl,
            centred_logit_gap,
            argmax,
            rows,
        } = family;
        let per_input = rows.len() / inputs.max(1);
        let mut columns: [Array2<f64>; 5] = std::array::from_fn(|_| Array2::zeros((inputs, per_input)));
        for (row_witness, comparison) in &rows {
            let at = [row_witness.input, row_witness.row];
            columns[0][at] = comparison.reference_argmax as f64;
            columns[1][at] = comparison.candidate_argmax as f64;
            columns[2][at] = f64::from(u8::from(comparison.argmax_certified));
            columns[3][at] = finite("centred_gap", comparison.centred_gap)?;
            columns[4][at] = finite("centred_gap_band", comparison.centred_gap_band)?;
        }
        let names = ["reference_argmax", "candidate_argmax", "argmax_certified", "centred_gap", "centred_gap_band"];
        let ids: Vec<String> = names
            .iter()
            .zip(columns)
            .map(|(name, column)| {
                let id = format!("interventions/{k}/{name}");
                arrays.insert(id.clone(), column.into_dyn());
                id
            })
            .collect();
        let status = |status| EvidenceStatusWire::from_status_with(status, witness, domain);
        reports.push(FamilyVerificationReport {
            certified: family_certified,
            forward_kl: status(forward_kl)?,
            reverse_kl: status(reverse_kl)?,
            centred_logit_gap: status(centred_logit_gap)?,
            argmax_rows: argmax.rows,
            argmax_disagreeing: argmax.disagreeing,
            argmax_certified_rows: argmax.certified_rows,
            argmax_disagreeing_fraction: status(argmax.fraction)?,
            reference_argmax: ids[0].clone(),
            candidate_argmax: ids[1].clone(),
            argmax_certified: ids[2].clone(),
            centred_gap: ids[3].clone(),
            centred_gap_band: ids[4].clone(),
        });
    }
    Ok(output(
        MpdResult::VerifyLogits(VerifyLogitsReport {
            certified,
            interventions: reports,
        }),
        arrays,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::run_parameter_decomposition;
    use super::super::tests::request_json;
    use super::*;
    use crate::parameter_decomposition::test_support::test_governor;
    use ndarray::Array4;

    /// Two interventions over three inputs of two rows of four classes, dyadic logits.
    fn logits(offset: f64) -> Array4<f64> {
        Array4::from_shape_fn((2, 3, 2, 4), |(k, input, row, class)| {
            0.25 * ((k + 2 * input + row + class * 3) % 5) as f64 + if class == 0 { offset } else { 0.0 }
        })
    }

    fn tensors(offset: f64) -> BTreeMap<String, ArrayD<f64>> {
        let radius = Array4::from_elem((2, 3, 2, 4), 1e-12);
        BTreeMap::from([
            ("r".to_string(), logits(0.0).into_dyn()),
            ("rr".to_string(), radius.clone().into_dyn()),
            ("c".to_string(), logits(offset).into_dyn()),
            ("cr".to_string(), radius.into_dyn()),
        ])
    }

    fn request(kl: f64) -> String {
        request_json(&format!(
            r#"{{"kind": "verify_logits", "reference": {{"logits": "r", "radius": "rr"}},
                "candidate": {{"logits": "c", "radius": "cr"}}, "tolerance": {{"kl": {kl}, "centred_logit_gap": 0.5}}}}"#
        ))
    }

    fn owner(tensors: &BTreeMap<String, ArrayD<f64>>, tolerance: &Tolerance) -> MpdOutput {
        let four = |id: &str| tensors[id].view().into_dimensionality::<Ix4>().expect("four axes");
        let contract = verify_counterfactual_contract(
            test_governor(),
            &supplied(four("r"), four("rr")),
            &supplied(four("c"), four("cr")),
            &[0, 1],
            &[0, 1, 2],
            tolerance,
        )
        .expect("owner contract");
        project(contract, tolerance, 3).expect("projection")
    }

    #[test]
    fn verify_report_is_the_owner_contract_field_for_field() {
        for (offset, kl) in [(0.0, 0.1), (0.5, 0.1), (0.5, 1e-6)] {
            let tensors = tensors(offset);
            let tolerance = Tolerance { kl, centred_logit_gap: 0.5 };
            let output = run_parameter_decomposition(&request(kl), &tensors, test_governor()).expect("surface run");
            assert_eq!(output, owner(&tensors, &tolerance));
            let MpdResult::VerifyLogits(report) = &output.report.result else {
                panic!("expected a verification, got {:?}", output.report.result);
            };
            assert_eq!(report.interventions.len(), 2);
            if offset == 0.0 {
                // Identical logits: certified, no disagreement.
                assert!(report.certified);
                assert_eq!(report.interventions[0].argmax_disagreeing, 0);
            }
            if kl == 1e-6 {
                // A shifted class-0 logit exceeds a tight KL tolerance: a counterexample.
                assert!(!report.certified);
                assert!(matches!(report.interventions[0].forward_kl, EvidenceStatusWire::Counterexample { .. }));
            }
        }
    }

    #[test]
    fn a_verification_the_owner_refuses_reaches_the_caller() {
        let good = tensors(0.0);
        assert!(run_parameter_decomposition(&request(0.1), &good, test_governor()).is_ok());
        assert!(matches!(
            run_parameter_decomposition(&request(-1.0), &good, test_governor()),
            Err(MpdSurfaceError::Verify(_))
        ));
        let mut short = good.clone();
        short.insert("c".to_string(), Array4::<f64>::zeros((2, 3, 2, 3)).into_dyn());
        assert!(matches!(
            run_parameter_decomposition(&request(0.1), &short, test_governor()),
            Err(MpdSurfaceError::Verify(_))
        ));
        let mut fewer = good.clone();
        fewer.insert("cr".to_string(), Array4::<f64>::zeros((1, 3, 2, 4)).into_dyn());
        assert!(matches!(
            run_parameter_decomposition(&request(0.1), &fewer, test_governor()),
            Err(MpdSurfaceError::TensorShape { .. })
        ));
        let mut flat = good.clone();
        flat.insert("r".to_string(), Array2::<f64>::zeros((2, 4)).into_dyn());
        assert!(matches!(
            run_parameter_decomposition(&request(0.1), &flat, test_governor()),
            Err(MpdSurfaceError::TensorShape { .. })
        ));
        let stray = request(0.1).replacen("\"kl\"", "\"js\": 1, \"kl\"", 1);
        assert!(matches!(
            run_parameter_decomposition(&stray, &good, test_governor()),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
    }
}
