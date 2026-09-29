//! `joint_operators`: the factored gauge-invariant query/key and value/output operators
//! of an attention block ([`crate::joint_operators`]), their
//! energies over a declared context, family Grams and pairwise comparisons.

use std::collections::BTreeMap;

use gam_runtime::resource::MemoryGovernor;
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

use super::code::EvidenceStatusWire;
use super::layer::{AttentionRequest, native_attention};
use super::{MpdOutput, MpdResult, MpdSurfaceError, finite, output, vector};
use crate::joint_operators::{
    FactoredOperator, OperatorComparison, OperatorPair, QueryKeyOperators, ValueOutputOperators,
    compare_operators, family_gram, query_key_operators, value_output_operators,
};

/// An attention block and what to read off its joint operators.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct JointOperatorsRequest {
    pub attention: AttentionRequest,
    /// The preceding residual norm's gain, folded into the columns, or null.
    pub input_gain: Option<String>,
    /// The declared context length over which planes split into content and position.
    pub context_length: f64,
    /// Families whose Frobenius Gram is wanted.
    pub grams: Vec<Vec<OperatorRef>>,
    /// Pairs to compare (equality and proportionality).
    pub comparisons: Vec<[OperatorRef; 2]>,
}

/// One joint operator of the block.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OperatorRef {
    /// `A_hj`, the cosine operator of query head `head` on rotary plane `plane`.
    Cosine { head: usize, plane: usize },
    /// `B_hj`, the sine operator.
    Sine { head: usize, plane: usize },
    /// `Π_h`, the pass-through operator.
    PassThrough { head: usize },
    /// `C_h = O_h V_g diag(γ)`.
    HeadValueOutput { head: usize },
    /// `Σ_{h∈g} C_h`.
    GroupValueOutput { group: usize },
}

/// Every operator's factors (as array ids), the energies, Grams and comparisons.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct JointOperatorsReport {
    pub planes: usize,
    /// `σα²` on the rotary operators and `σ` on the pass-through operators.
    pub rotary_multiplier: f64,
    pub pass_through_multiplier: f64,
    pub query_key_formation_defect: f64,
    pub value_output_formation_defect: f64,
    /// Per query head: `[cosine, sine]` factors per plane, and the pass-through factors.
    pub query_key: Vec<HeadOperatorsReport>,
    pub head_value_output: Vec<FactorsReport>,
    pub group_value_output: Vec<FactorsReport>,
    pub energies: Vec<HeadEnergyReport>,
    /// Ids of each family's Gram and band.
    pub grams: Vec<GramReport>,
    pub comparisons: Vec<ComparisonReport>,
}

/// One query head's operators.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HeadOperatorsReport {
    pub cosine: Vec<FactorsReport>,
    pub sine: Vec<FactorsReport>,
    pub pass_through: Option<FactorsReport>,
}

/// `M = L Rᵀ`: ids of `L` and `R` (`width × rank` each).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FactorsReport {
    pub left: String,
    pub right: String,
}

/// [`crate::joint_operators::HeadEnergy`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct HeadEnergyReport {
    pub content: f64,
    pub positional: f64,
    pub pass_through: f64,
    pub band: f64,
    pub slow_planes: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GramReport {
    pub gram: String,
    pub band: String,
}

/// [`OperatorPair`] on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
pub struct OperatorPairReport {
    pub width: usize,
}

pub type PairStatusWire = EvidenceStatusWire<(), OperatorPairReport>;

/// [`OperatorComparison`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ComparisonReport {
    pub pair: [OperatorRef; 2],
    pub difference: PairStatusWire,
    pub first_norm: PairStatusWire,
    pub second_norm: PairStatusWire,
    pub scale: f64,
    pub proportionality_residual: PairStatusWire,
    /// `M₁ ≠ M₂` is certified.
    pub proven_distinct: bool,
}

fn pair(pair: OperatorPair) -> OperatorPairReport {
    OperatorPairReport { width: pair.width }
}

fn comparison(refs: [OperatorRef; 2], compared: OperatorComparison) -> Result<ComparisonReport, MpdSurfaceError> {
    let status = |status| EvidenceStatusWire::from_status_with(status, |witness| witness, pair);
    Ok(ComparisonReport {
        pair: refs,
        proven_distinct: compared.proven_distinct(),
        difference: status(compared.difference)?,
        first_norm: status(compared.first_norm)?,
        second_norm: status(compared.second_norm)?,
        scale: finite("scale", compared.scale)?,
        proportionality_residual: status(compared.proportionality_residual)?,
    })
}

fn operator<'a>(
    reference: OperatorRef,
    query_key: &'a QueryKeyOperators,
    value_output: &'a ValueOutputOperators,
) -> Result<&'a FactoredOperator, MpdSurfaceError> {
    let geometry = query_key.geometry();
    let (heads, planes, groups) = (geometry.n_heads, query_key.planes(), geometry.n_kv_heads);
    let refused = || MpdSurfaceError::InvalidRequest(format!("the block has no operator {reference:?}"));
    match reference {
        OperatorRef::Cosine { head, plane } if head < heads && plane < planes => Ok(query_key.cosine(head, plane)),
        OperatorRef::Sine { head, plane } if head < heads && plane < planes => Ok(query_key.sine(head, plane)),
        OperatorRef::PassThrough { head } if head < heads => query_key.pass_through(head).ok_or_else(refused),
        OperatorRef::HeadValueOutput { head } if head < heads => Ok(value_output.head(head)),
        OperatorRef::GroupValueOutput { group } if group < groups => Ok(value_output.group(group)),
        _ => Err(refused()),
    }
}

pub(super) fn run(
    request: JointOperatorsRequest,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let joint = MpdSurfaceError::Joint;
    let native = native_attention(tensors, &request.attention)?;
    let input_gain = match &request.input_gain {
        Some(id) => Some(vector(tensors, id)?),
        None => None,
    };
    let query_key = query_key_operators(&native, input_gain).map_err(joint)?;
    let value_output = value_output_operators(&native, input_gain).map_err(joint)?;
    drop(native);
    let energies = query_key.energies(request.context_length).map_err(joint)?;
    let mut grams = Vec::with_capacity(request.grams.len());
    for family in &request.grams {
        let members = family
            .iter()
            .map(|&reference| operator(reference, &query_key, &value_output))
            .collect::<Result<Vec<_>, _>>()?;
        let gram = family_gram(governor, &members).map_err(joint)?;
        grams.push((gram.gram.clone(), gram.band.clone()));
    }
    let mut comparisons = Vec::with_capacity(request.comparisons.len());
    for refs in &request.comparisons {
        let first = operator(refs[0], &query_key, &value_output)?;
        let second = operator(refs[1], &query_key, &value_output)?;
        comparisons.push((*refs, compare_operators(governor, first, second).map_err(joint)?));
    }
    project(&query_key, &value_output, energies, grams, comparisons)
}

pub(super) fn project(
    query_key: &QueryKeyOperators,
    value_output: &ValueOutputOperators,
    energies: Vec<crate::joint_operators::HeadEnergy>,
    grams: Vec<(ndarray::Array2<f64>, ndarray::Array2<f64>)>,
    comparisons: Vec<([OperatorRef; 2], OperatorComparison)>,
) -> Result<MpdOutput, MpdSurfaceError> {
    let mut arrays = BTreeMap::new();
    let mut factors = |prefix: String, operator: &FactoredOperator| {
        let (left, right) = (format!("{prefix}/left"), format!("{prefix}/right"));
        arrays.insert(left.clone(), operator.left().to_owned().into_dyn());
        arrays.insert(right.clone(), operator.right().to_owned().into_dyn());
        FactorsReport { left, right }
    };
    let geometry = query_key.geometry();
    let planes = query_key.planes();
    let heads: Vec<HeadOperatorsReport> = (0..geometry.n_heads)
        .map(|head| HeadOperatorsReport {
            cosine: (0..planes)
                .map(|plane| factors(format!("query_key/{head}/cosine/{plane}"), query_key.cosine(head, plane)))
                .collect(),
            sine: (0..planes)
                .map(|plane| factors(format!("query_key/{head}/sine/{plane}"), query_key.sine(head, plane)))
                .collect(),
            pass_through: query_key
                .pass_through(head)
                .map(|operator| factors(format!("query_key/{head}/pass_through"), operator)),
        })
        .collect();
    let head_value_output = (0..geometry.n_heads)
        .map(|head| factors(format!("value_output/head/{head}"), value_output.head(head)))
        .collect();
    let group_value_output = (0..geometry.n_kv_heads)
        .map(|group| factors(format!("value_output/group/{group}"), value_output.group(group)))
        .collect();
    let grams = grams
        .into_iter()
        .enumerate()
        .map(|(index, (gram, band))| {
            let (gram_id, band_id) = (format!("grams/{index}/gram"), format!("grams/{index}/band"));
            arrays.insert(gram_id.clone(), gram.into_dyn());
            arrays.insert(band_id.clone(), band.into_dyn());
            GramReport { gram: gram_id, band: band_id }
        })
        .collect();
    let energies = energies
        .into_iter()
        .map(|energy| {
            Ok(HeadEnergyReport {
                content: finite("content", energy.content)?,
                positional: finite("positional", energy.positional)?,
                pass_through: finite("pass_through", energy.pass_through)?,
                band: finite("band", energy.band)?,
                slow_planes: energy.slow_planes,
            })
        })
        .collect::<Result<Vec<_>, MpdSurfaceError>>()?;
    let comparisons = comparisons
        .into_iter()
        .map(|(refs, compared)| comparison(refs, compared))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(output(
        MpdResult::JointOperators(Box::new(JointOperatorsReport {
            planes,
            rotary_multiplier: finite("rotary_multiplier", query_key.rotary_multiplier())?,
            pass_through_multiplier: finite("pass_through_multiplier", query_key.pass_through_multiplier())?,
            query_key_formation_defect: finite("formation_defect", query_key.formation_defect())?,
            value_output_formation_defect: finite("formation_defect", value_output.formation_defect())?,
            query_key: heads,
            head_value_output,
            group_value_output,
            energies,
            grams,
            comparisons,
        })),
        arrays,
    ))
}

#[cfg(test)]
mod tests {
    use super::super::run_parameter_decomposition;
    use super::super::tests::request_json;
    use super::*;
    use crate::attention::{
        AffineProjection, AttentionGeometry, NativeAttention, RotaryEmbedding, RotaryPairing,
    };
    use crate::test_support::test_governor;
    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    const WIDTH: usize = 8;

    fn uniform(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || rng.random_range(-1.0..1.0))
    }

    /// Four query heads over two key/value heads of width 4: one rotary plane of
    /// frequency 1, one of 0.001, and no pass-through.
    fn tensors() -> BTreeMap<String, ArrayD<f64>> {
        let mut rng = StdRng::seed_from_u64(31);
        let mut query = uniform(&mut rng, 16, WIDTH);
        // Head 1 repeats head 0's rows, so their operators are equal.
        for column in 0..WIDTH {
            for row in 0..4 {
                query[[4 + row, column]] = query[[row, column]];
            }
        }
        BTreeMap::from([
            ("q".to_string(), query.into_dyn()),
            ("k".to_string(), uniform(&mut rng, 8, WIDTH).into_dyn()),
            ("v".to_string(), uniform(&mut rng, 8, WIDTH).into_dyn()),
            ("o".to_string(), uniform(&mut rng, WIDTH, 16).into_dyn()),
            ("gain".to_string(), Array1::from_shape_simple_fn(WIDTH, || rng.random_range(0.5..1.5)).into_dyn()),
        ])
    }

    fn request(comparisons: &str) -> String {
        request_json(&format!(
            r#"{{"kind": "joint_operators",
                "attention": {{"geometry": {{"model_dim": 8, "n_heads": 4, "n_kv_heads": 2, "head_dim": 4}},
                    "rotary": {{"pairing": "half_split", "inverse_frequencies": [1.0, 0.001], "attention_scaling": 1.0}},
                    "score_scale": 0.5, "query": {{"weight": "q", "bias": null}}, "key": {{"weight": "k", "bias": null}},
                    "value": {{"weight": "v", "bias": null}}, "output": {{"weight": "o", "bias": null}},
                    "query_key_norm": null}},
                "input_gain": "gain", "context_length": 64.0,
                "grams": [[{{"kind": "cosine", "head": 0, "plane": 0}}, {{"kind": "sine", "head": 0, "plane": 0}}, {{"kind": "group_value_output", "group": 1}}]],
                "comparisons": {comparisons}}}"#
        ))
    }

    fn owner(tensors: &BTreeMap<String, ArrayD<f64>>) -> MpdOutput {
        let m = |id: &str| tensors[id].view().into_dimensionality::<ndarray::Ix2>().expect("matrix").to_owned();
        let affine = |weight: Array2<f64>| AffineProjection { bias: Array1::zeros(weight.nrows()), weight };
        let native = NativeAttention::new(
            AttentionGeometry { model_dim: WIDTH, n_heads: 4, n_kv_heads: 2, head_dim: 4 },
            RotaryEmbedding { pairing: RotaryPairing::HalfSplit, inverse_frequencies: vec![1.0, 0.001], attention_scaling: 1.0 },
            0.5,
            affine(m("q")),
            affine(m("k")),
            affine(m("v")),
            affine(m("o")),
        )
        .expect("native attention");
        let gain = tensors["gain"].view().into_dimensionality::<ndarray::Ix1>().expect("gain");
        let query_key = query_key_operators(&native, Some(gain)).expect("owner query/key");
        let value_output = value_output_operators(&native, Some(gain)).expect("owner value/output");
        let energies = query_key.energies(64.0).expect("owner energies");
        let family = [query_key.cosine(0, 0), query_key.sine(0, 0), value_output.group(1)];
        let gram = family_gram(test_governor(), &family).expect("owner gram");
        let pairs = [
            [OperatorRef::Cosine { head: 0, plane: 0 }, OperatorRef::Cosine { head: 1, plane: 0 }],
            [OperatorRef::Cosine { head: 0, plane: 0 }, OperatorRef::Cosine { head: 2, plane: 0 }],
        ];
        let comparisons = pairs
            .iter()
            .map(|refs| {
                let first = operator(refs[0], &query_key, &value_output).expect("first");
                let second = operator(refs[1], &query_key, &value_output).expect("second");
                (*refs, compare_operators(test_governor(), first, second).expect("owner comparison"))
            })
            .collect();
        project(&query_key, &value_output, energies, vec![(gram.gram.clone(), gram.band.clone())], comparisons)
            .expect("projection")
    }

    const PAIRS: &str = r#"[[{"kind": "cosine", "head": 0, "plane": 0}, {"kind": "cosine", "head": 1, "plane": 0}],
                            [{"kind": "cosine", "head": 0, "plane": 0}, {"kind": "cosine", "head": 2, "plane": 0}]]"#;

    #[test]
    fn joint_operators_report_is_the_owner_result_field_for_field() {
        let tensors = tensors();
        let output = run_parameter_decomposition(&request(PAIRS), &tensors, test_governor()).expect("surface run");
        assert_eq!(output, owner(&tensors));
        let MpdResult::JointOperators(report) = &output.report.result else {
            panic!("expected joint operators, got {:?}", output.report.result);
        };
        // Heads 0 and 1 share query rows and one key head: equal operators; head 2 differs.
        assert!(!report.comparisons[0].proven_distinct);
        assert!(report.comparisons[1].proven_distinct);
        // The slow plane (wavelength 2π/0.001 > 64) is content.
        assert_eq!(report.energies[0].slow_planes, vec![1]);
    }

    #[test]
    fn a_joint_request_the_owner_refuses_reaches_the_caller() {
        let mut tensors = tensors();
        assert!(run_parameter_decomposition(&request("[]"), &tensors, test_governor()).is_ok());
        assert!(matches!(
            run_parameter_decomposition(
                &request(r#"[[{"kind": "cosine", "head": 9, "plane": 0}, {"kind": "sine", "head": 0, "plane": 0}]]"#),
                &tensors,
                test_governor()
            ),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
        assert!(matches!(
            run_parameter_decomposition(
                &request(r#"[[{"kind": "pass_through", "head": 0}, {"kind": "sine", "head": 0, "plane": 0}]]"#),
                &tensors,
                test_governor()
            ),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
        tensors.insert("gain".to_string(), Array1::<f64>::ones(3).into_dyn());
        assert!(matches!(
            run_parameter_decomposition(&request("[]"), &tensors, test_governor()),
            Err(MpdSurfaceError::Joint(_))
        ));
    }
}
