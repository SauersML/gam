//! `weighted_observability`: the weighted observability Gramian of readouts pulled
//! back through declared steps (`state::WeightedObservability`), and the capture of
//! declared candidate subspaces.

use std::collections::BTreeMap;

use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, ArrayD};
use serde::{Deserialize, Serialize};

use super::code::EvidenceStatusWire;
use super::module_split::{MlpBlockRequest, normal_form};
use super::{MpdOutput, MpdResult, MpdSurfaceError, finite, matrix, output};
use crate::parameter_decomposition::module_split::MlpNormalForm;
use crate::parameter_decomposition::state::{
    ObservabilityLetter, ObservabilitySpectrum, ObservabilityStep, StateDomain, SubspaceCapture,
    WeightedObservability,
};
use crate::parameter_decomposition::supports::EvidenceStatus;

/// Forward steps and declared candidate subspaces. Every string is the id of an input
/// array.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WeightedObservabilityRequest {
    pub steps: Vec<StepRequest>,
    /// Candidate subspaces (`k × n_0`, full row rank) whose capture is wanted.
    pub candidates: Vec<String>,
}

/// One forward step: its letters and the readouts of its output.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct StepRequest {
    pub letters: Vec<LetterRequest>,
    pub readouts: Vec<String>,
}

/// One declared letter.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum LetterRequest {
    /// A linear letter `T` (`n_{l+1} × n_l`).
    Linear { tensor: String },
    /// Rank-one unit letters: reads `a_j` as rows (`n × n_l`), writes `u_j` as rows
    /// (`n × n_{l+1}`). They must be a merged normal form.
    Units { reads: String, writes: String },
    /// An MLP block's letters from its merged normal form
    /// (`module_split::MlpNormalForm::observability_letters`).
    Mlp { block: MlpBlockRequest },
}

/// [`WeightedObservability`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct WeightedObservabilityReport {
    /// Ids of `F = diag(σ) Vᵀ` and of `V`'s rows.
    pub factor: String,
    pub directions: String,
    pub formation: f64,
    /// The spectrum at the input of every step; `step_spectra[0]` is `G`'s.
    pub step_spectra: Vec<SpectrumReport>,
    pub captures: Vec<CaptureReport>,
}

/// [`StateDomain`] on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StateDomainReport {
    States { count: usize },
    Codes { count: usize },
    FiberPairs { tested_states: usize, vacuous_states: usize, futures: usize },
    UnitStates { dimension: usize },
}

fn state_domain(domain: StateDomain) -> StateDomainReport {
    match domain {
        StateDomain::States { count } => StateDomainReport::States { count },
        StateDomain::Codes { count } => StateDomainReport::Codes { count },
        StateDomain::FiberPairs { tested_states, vacuous_states, futures } => {
            StateDomainReport::FiberPairs { tested_states, vacuous_states, futures }
        }
        StateDomain::UnitStates { dimension } => StateDomainReport::UnitStates { dimension },
    }
}

pub type StateStatusWire = EvidenceStatusWire<(), StateDomainReport>;

fn status(status: EvidenceStatus<(), StateDomain>) -> Result<StateStatusWire, MpdSurfaceError> {
    EvidenceStatusWire::from_status_with(status, |witness| witness, state_domain)
}

/// [`ObservabilitySpectrum`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SpectrumReport {
    /// Id of `σ₁ ≥ σ₂ ≥ …`.
    pub singular_values: String,
    pub band: f64,
    pub resolved_rank: usize,
    pub rank_ceiling: usize,
    pub rank: StateStatusWire,
    pub energy: f64,
    pub participation_ratio: f64,
}

/// [`SubspaceCapture`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CaptureReport {
    pub dimension: usize,
    pub energy_fraction: StateStatusWire,
    pub principal_cosines: Vec<f64>,
    pub eigengap: f64,
    pub angle_perturbation: f64,
}

pub(super) fn run(
    request: WeightedObservabilityRequest,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    // The normal forms the MLP letters borrow, in declaration order.
    let mut forms: Vec<MlpNormalForm> = Vec::new();
    for step in &request.steps {
        for letter in &step.letters {
            if let LetterRequest::Mlp { block } = letter {
                forms.push(normal_form(tensors, block)?);
            }
        }
    }
    let mut next_form = forms.iter();
    let mut steps = Vec::with_capacity(request.steps.len());
    for step in &request.steps {
        let mut letters = Vec::with_capacity(step.letters.len());
        for letter in &step.letters {
            match letter {
                LetterRequest::Linear { tensor } => letters.push(ObservabilityLetter::Linear(matrix(tensors, tensor)?)),
                LetterRequest::Units { reads, writes } => letters.push(ObservabilityLetter::Units {
                    reads: matrix(tensors, reads)?,
                    writes: matrix(tensors, writes)?,
                }),
                LetterRequest::Mlp { .. } => {
                    let form = next_form.next().ok_or_else(|| {
                        MpdSurfaceError::InvalidRequest("an MLP letter has no normal form".to_string())
                    })?;
                    letters.extend(form.observability_letters());
                }
            }
        }
        let readouts = step
            .readouts
            .iter()
            .map(|id| matrix(tensors, id))
            .collect::<Result<Vec<_>, _>>()?;
        steps.push(ObservabilityStep { letters, readouts });
    }
    let observability = WeightedObservability::pull_back(governor, &steps).map_err(MpdSurfaceError::State)?;
    let captures = request
        .candidates
        .iter()
        .map(|id| {
            observability
                .capture(governor, matrix(tensors, id)?)
                .map_err(MpdSurfaceError::State)
        })
        .collect::<Result<Vec<_>, _>>()?;
    project(observability, captures)
}

pub(super) fn project(
    observability: WeightedObservability,
    captures: Vec<SubspaceCapture>,
) -> Result<MpdOutput, MpdSurfaceError> {
    let mut arrays = BTreeMap::new();
    let mut step_spectra = Vec::with_capacity(observability.step_spectra.len());
    for (index, spectrum) in observability.step_spectra.into_iter().enumerate() {
        let rank = status(spectrum.rank_evidence().map_err(MpdSurfaceError::Evidence)?)?;
        let ObservabilitySpectrum {
            singular_values,
            band,
            resolved_rank,
            rank_ceiling,
            energy,
            participation_ratio,
        } = spectrum;
        let id = format!("step_spectra/{index}/singular_values");
        arrays.insert(id.clone(), Array1::from(singular_values).into_dyn());
        step_spectra.push(SpectrumReport {
            singular_values: id,
            band: finite("band", band)?,
            resolved_rank,
            rank_ceiling,
            rank,
            energy: finite("energy", energy)?,
            participation_ratio: finite("participation_ratio", participation_ratio)?,
        });
    }
    let captures = captures
        .into_iter()
        .map(|capture| {
            Ok(CaptureReport {
                dimension: capture.dimension,
                energy_fraction: status(capture.energy_fraction)?,
                principal_cosines: capture
                    .principal_cosines
                    .iter()
                    .map(|&value| finite("principal_cosines", value))
                    .collect::<Result<Vec<_>, _>>()?,
                eigengap: finite("eigengap", capture.eigengap)?,
                angle_perturbation: finite("angle_perturbation", capture.angle_perturbation)?,
            })
        })
        .collect::<Result<Vec<_>, MpdSurfaceError>>()?;
    arrays.insert("factor".to_string(), observability.factor.into_dyn());
    arrays.insert("directions".to_string(), observability.directions.into_dyn());
    Ok(output(
        MpdResult::WeightedObservability(WeightedObservabilityReport {
            factor: "factor".to_string(),
            directions: "directions".to_string(),
            formation: finite("formation", observability.formation)?,
            step_spectra,
            captures,
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
    use ndarray::{Array2, array};

    fn tensors() -> BTreeMap<String, ArrayD<f64>> {
        BTreeMap::from([
            ("identity".to_string(), Array2::<f64>::eye(3).into_dyn()),
            ("shear".to_string(), array![[1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.5]].into_dyn()),
            ("r".to_string(), array![[1.0, 0.0, 0.0]].into_dyn()),
            ("w_in".to_string(), array![[0.0, 1.0, 0.5], [1.0, 0.0, -1.0]].into_dyn()),
            ("b_in".to_string(), array![0.1, -0.2].into_dyn()),
            ("w_out".to_string(), array![[1.0, 0.5], [0.0, 1.0], [0.25, 0.0]].into_dyn()),
            ("b_out".to_string(), array![0.0, 0.0, 0.0].into_dyn()),
            ("candidate".to_string(), array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]].into_dyn()),
        ])
    }

    const REQUEST: &str = r#"{"kind": "weighted_observability", "steps": [
        {"letters": [{"kind": "linear", "tensor": "identity"}, {"kind": "linear", "tensor": "shear"}], "readouts": []},
        {"letters": [{"kind": "mlp", "block": {"hidden_act": "gelu", "w_in": "w_in", "b_in": "b_in", "w_out": "w_out",
            "b_out": "b_out", "skip": "identity"}}], "readouts": ["r"]}],
        "candidates": ["candidate"]}"#;

    #[test]
    fn weighted_observability_report_is_the_owner_result_field_for_field() {
        let tensors = tensors();
        let m = |id: &str| matrix(&tensors, id).expect("matrix");
        let form = MlpNormalForm::new(
            gam_math::gaussian_activation::GaussianActivation::ExactGelu,
            m("w_in"),
            super::super::vector(&tensors, "b_in").expect("b_in"),
            m("w_out"),
            super::super::vector(&tensors, "b_out").expect("b_out"),
            Some(m("identity")),
        )
        .expect("owner normal form");
        let steps = vec![
            ObservabilityStep {
                letters: vec![ObservabilityLetter::Linear(m("identity")), ObservabilityLetter::Linear(m("shear"))],
                readouts: vec![],
            },
            ObservabilityStep { letters: form.observability_letters(), readouts: vec![m("r")] },
        ];
        let owner = WeightedObservability::pull_back(test_governor(), &steps).expect("owner pull-back");
        let capture = owner.capture(test_governor(), m("candidate")).expect("owner capture");
        let expected = project(owner, vec![capture]).expect("projection");
        let output = run_parameter_decomposition(&request_json(REQUEST), &tensors, test_governor()).expect("surface run");
        assert_eq!(output, expected);
        let MpdResult::WeightedObservability(report) = &output.report.result else {
            panic!("expected weighted observability, got {:?}", output.report.result);
        };
        assert_eq!(report.step_spectra.len(), 2);
        assert_eq!(report.captures[0].dimension, 2);
    }

    #[test]
    fn a_pull_back_the_owner_refuses_reaches_the_caller() {
        let tensors = tensors();
        assert!(run_parameter_decomposition(&request_json(REQUEST), &tensors, test_governor()).is_ok());
        let no_readouts = REQUEST.replace(r#""readouts": ["r"]"#, r#""readouts": []"#);
        assert!(matches!(
            run_parameter_decomposition(&request_json(&no_readouts), &tensors, test_governor()),
            Err(MpdSurfaceError::State(_))
        ));
        let narrow = REQUEST.replace(r#""readouts": ["r"]"#, r#""readouts": ["b_in"]"#);
        assert!(matches!(
            run_parameter_decomposition(&request_json(&narrow), &tensors, test_governor()),
            Err(MpdSurfaceError::TensorShape { .. })
        ));
        let wrong_candidate = REQUEST.replace(r#""candidates": ["candidate"]"#, r#""candidates": ["w_in"]"#);
        assert!(run_parameter_decomposition(&request_json(&wrong_candidate), &tensors, test_governor()).is_ok());
        let wrong_width = REQUEST.replace(r#""candidates": ["candidate"]"#, r#""candidates": ["w_out"]"#);
        assert!(matches!(
            run_parameter_decomposition(&request_json(&wrong_width), &tensors, test_governor()),
            Err(MpdSurfaceError::State(_))
        ));
        let stray = REQUEST.replace(r#""kind": "linear", "tensor": "shear""#, r#""kind": "linear", "tensor": "shear", "weight": 2"#);
        assert!(matches!(
            run_parameter_decomposition(&request_json(&stray), &tensors, test_governor()),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
    }
}
