//! `weighted_observability`: the weighted observability Gramian of readouts pulled
//! back through declared steps (`state::WeightedObservability`), and the capture of
//! declared candidate subspaces.

use std::collections::BTreeMap;

use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, ArrayD};
use serde::{Deserialize, Serialize};

use super::code::EvidenceStatusWire;
use super::layer::{AttentionRequest, native_attention};
use super::module_split::{MlpBlockRequest, normal_form};
use super::{MpdOutput, MpdResult, MpdSurfaceError, finite, matrix, output, vector};
use crate::joint_operators::{RoutingLawLetters, attention_letters, head_transport_of};
use crate::module_split::MlpNormalForm;
use crate::state::{
    ObservabilityLetter, ObservabilitySpectrum, ObservabilityStep, StateDomain, SubspaceCapture,
    WeightedObservability,
};
use crate::supports::EvidenceStatus;

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
    /// A linear letter `T` (`n_{l+1} × n_l`). A head's value/output transport of an attention
    /// block declared in the same step is refused (`joint_operators::head_transport_of`): the
    /// block enters only as its `attention` letter.
    Linear { tensor: String },
    /// Rank-one unit letters: reads `a_j` as rows (`n × n_l`), writes `u_j` as rows
    /// (`n × n_{l+1}`). They must be a merged normal form.
    Units { reads: String, writes: String },
    /// An MLP block's letters from its merged normal form
    /// (`module_split::MlpNormalForm::observability_letters`).
    Mlp { block: MlpBlockRequest },
    /// An attention block's letters, one per routing law
    /// (`joint_operators::attention_letters`): heads with equal score operators share one
    /// letter, their summed value/output transport. With `input_gain` the preceding
    /// residual norm's gain is folded in (pre-norm); with `output_gain` the gain of the norm
    /// on the attention output (post-norm) scales the transports' rows. This is the only way
    /// an attention block enters: per-head value/output maps are not letters, because one
    /// letter per head over-counts and is not invariant under the cross-head `GL` of heads
    /// that share a law.
    Attention { attention: AttentionRequest, input_gain: Option<String>, output_gain: Option<String> },
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
    /// The routing laws of every attention letter, in declaration order: each law's query
    /// heads (`joint_operators::RoutingLaws::laws`).
    pub attention_laws: Vec<Vec<Vec<usize>>>,
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
    // The routing-law letters the attention letters borrow, in declaration order.
    let mut attentions: Vec<RoutingLawLetters> = Vec::new();
    for step in &request.steps {
        for letter in &step.letters {
            if let LetterRequest::Attention { attention, input_gain, output_gain } = letter {
                let native = native_attention(tensors, attention)?;
                let gain = |id: &Option<String>| match id {
                    Some(id) => vector(tensors, id).map(Some),
                    None => Ok(None),
                };
                let (input, output) = (gain(input_gain)?, gain(output_gain)?);
                for other in &step.letters {
                    if let LetterRequest::Linear { tensor } = other {
                        let linear = matrix(tensors, tensor)?;
                        if linear.dim() == (native.geometry().model_dim, native.geometry().model_dim)
                            && let Some(head) = head_transport_of(&native, input, output, linear).map_err(MpdSurfaceError::Joint)?
                        {
                            return Err(MpdSurfaceError::InvalidRequest(format!(
                                "linear letter {tensor:?} is head {head}'s value/output transport: an attention block \
                                 enters only as its routing-law letters"
                            )));
                        }
                    }
                }
                let derived = attention_letters(governor, &native, input, output).map_err(MpdSurfaceError::Joint)?;
                attentions.push(derived);
            }
        }
    }
    let mut next_attention = attentions.iter();
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
                LetterRequest::Attention { .. } => {
                    let derived = next_attention.next().ok_or_else(|| {
                        MpdSurfaceError::InvalidRequest("an attention letter has no routing laws".to_string())
                    })?;
                    letters.push(ObservabilityLetter::RoutingLaws(derived));
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
    let laws = attentions.iter().map(|letters| letters.laws.laws.clone()).collect();
    project(observability, captures, laws)
}

pub(super) fn project(
    observability: WeightedObservability,
    captures: Vec<SubspaceCapture>,
    attention_laws: Vec<Vec<Vec<usize>>>,
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
            attention_laws,
        }),
        arrays,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::run_parameter_decomposition;
    use super::super::tests::request_json;
    use super::*;
    use crate::test_support::test_governor;
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
        let expected = project(owner, vec![capture], Vec::new()).expect("projection");
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

    /// An attention letter is derived by the joint-operator owner: the surface result is the
    /// owner's pull-back through `attention_letters`, and heads 0 and 1 (equal query rows on
    /// one key/value head) share one law.
    #[test]
    fn an_attention_letter_is_the_owners_routing_law_letters() {
        use crate::attention::{
            AffineProjection, AttentionGeometry, NativeAttention, RotaryEmbedding, RotaryPairing,
        };
        use crate::joint_operators::attention_letters;
        let mut state: u64 = 17;
        let mut draw = |rows: usize, cols: usize| {
            Array2::from_shape_simple_fn((rows, cols), || {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                // Dyadic `k/16`: every product and short sum the test forms is exact.
                (((state >> 40) as f64 / (1u64 << 24) as f64 - 0.5) * 16.0).round() / 16.0
            })
        };
        let mut query = draw(8, 4);
        let first = query.slice(ndarray::s![..2, ..]).to_owned();
        query.slice_mut(ndarray::s![2..4, ..]).assign(&first);
        let tensors = BTreeMap::from([
            ("q".to_string(), query.into_dyn()),
            ("k".to_string(), draw(4, 4).into_dyn()),
            ("v".to_string(), draw(4, 4).into_dyn()),
            ("o".to_string(), draw(4, 8).into_dyn()),
            ("identity".to_string(), Array2::<f64>::eye(4).into_dyn()),
            ("r".to_string(), array![[1.0, -0.5, 0.25, 0.75]].into_dyn()),
        ]);
        let request = request_json(
            r#"{"kind": "weighted_observability", "steps": [{"letters": [{"kind": "linear", "tensor": "identity"},
                {"kind": "attention", "input_gain": null, "output_gain": null, "attention": {
                    "geometry": {"model_dim": 4, "n_heads": 4, "n_kv_heads": 2, "head_dim": 2},
                    "rotary": {"pairing": "half_split", "inverse_frequencies": [1.0], "attention_scaling": 1.0},
                    "score_scale": 0.5, "query": {"weight": "q", "bias": null}, "key": {"weight": "k", "bias": null},
                    "value": {"weight": "v", "bias": null}, "output": {"weight": "o", "bias": null},
                    "query_key_norm": null}}],
                "readouts": ["r"]}], "candidates": []}"#,
        );
        let m = |id: &str| tensors[id].view().into_dimensionality::<ndarray::Ix2>().expect("matrix").to_owned();
        let affine = |weight: Array2<f64>| AffineProjection { bias: ndarray::Array1::zeros(weight.nrows()), weight };
        let native = NativeAttention::new(
            AttentionGeometry { model_dim: 4, n_heads: 4, n_kv_heads: 2, head_dim: 2 },
            RotaryEmbedding { pairing: RotaryPairing::HalfSplit, inverse_frequencies: vec![1.0], attention_scaling: 1.0 },
            0.5,
            affine(m("q")),
            affine(m("k")),
            affine(m("v")),
            affine(m("o")),
        )
        .expect("native attention");
        let letters = attention_letters(test_governor(), &native, None, None).expect("owner letters");
        assert_eq!(letters.laws.laws, vec![vec![0, 1], vec![2], vec![3]]);
        let identity = m("identity");
        let readout = m("r");
        let owner = WeightedObservability::pull_back(
            test_governor(),
            &[ObservabilityStep {
                letters: vec![ObservabilityLetter::Linear(identity.view()), ObservabilityLetter::RoutingLaws(&letters)],
                readouts: vec![readout.view()],
            }],
        )
        .expect("owner pull-back");
        let expected = project(owner, Vec::new(), vec![letters.laws.laws.clone()]).expect("projection");
        let output = run_parameter_decomposition(&request, &tensors, test_governor()).expect("surface run");
        assert_eq!(output, expected);

        // A head's transport declared beside the block is a per-head letter: refused.
        let mut with_head = tensors.clone();
        let head = m("o").slice(ndarray::s![.., 4..6]).dot(&m("v").slice(ndarray::s![2..4, ..]));
        with_head.insert("head".to_string(), head.into_dyn());
        let per_head = request.replace(
            r#"{"kind": "linear", "tensor": "identity"},"#,
            r#"{"kind": "linear", "tensor": "identity"}, {"kind": "linear", "tensor": "head"},"#,
        );
        assert_ne!(per_head, request);
        assert!(matches!(
            run_parameter_decomposition(&per_head, &with_head, test_governor()),
            Err(MpdSurfaceError::InvalidRequest(message)) if message.contains("head 2")
        ));
    }
}
