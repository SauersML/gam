//! `canonical_layer`: the canonical gauge form of a native decoder layer
//! ([`crate::parameter_decomposition::canonical`]) on the wire, optionally with both
//! layers executed on declared residual rows.

use std::collections::BTreeMap;

use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, Array2, ArrayD};
use serde::{Deserialize, Serialize};

use super::layer::{AttentionRequest, RmsNormRequest, native_attention};
use super::{MpdOutput, MpdResult, MpdSurfaceError, finite, input, matrix, output, reserve, vector};
use crate::parameter_decomposition::attention::ProjectedRows;
use crate::parameter_decomposition::canonical::{
    CanonicalLayer, DecoderLayer, LayerExecution, LayerProjection, LayerRmsNorm, TensorDefect,
};
use crate::parameter_decomposition::gauge::SwigluUnits;

/// A bias-free-MLP sequential pre-norm decoder layer (Qwen3, Llama). Every string is
/// the id of an input array.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CanonicalLayerRequest {
    pub attention: AttentionRequest,
    pub input_norm: RmsNormRequest,
    pub post_norm: RmsNormRequest,
    pub gate: String,
    pub up: String,
    pub down: String,
    /// Residual rows to execute the native and the canonical layer on
    /// (`DecoderLayer::execute`), or null.
    pub execute: Option<ExecuteRequest>,
}

/// Exact residual rows (`positions × d`) at their token positions.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecuteRequest {
    pub residual: String,
    pub positions: Vec<i64>,
}

/// [`CanonicalLayer`] on the wire: the canonical tensors with their representation
/// defects, and the group elements that took the native layer there.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CanonicalLayerReport {
    pub input_norm: NormReport,
    pub query: ProjectionReport,
    pub key: ProjectionReport,
    pub value: ProjectionReport,
    pub output: ProjectionReport,
    pub query_key_norm: Option<QueryKeyNormReport>,
    pub post_norm: NormReport,
    pub gate: ProjectionReport,
    pub up: ProjectionReport,
    pub down: ProjectionReport,
    pub elements: ElementsReport,
    pub execution: Option<ExecutionReport>,
}

/// [`LayerRmsNorm`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NormReport {
    pub epsilon: f64,
    /// Id of the stored gain.
    pub gain: String,
    /// `|stored − represented| ≤ gain_defect · |stored|` entrywise.
    pub gain_defect: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QueryKeyNormReport {
    pub query: NormReport,
    pub key: NormReport,
}

/// [`LayerProjection`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ProjectionReport {
    pub weight: String,
    pub bias: String,
    pub weight_defect: DefectReport,
    /// Id of `|stored bias − represented bias|` entrywise.
    pub bias_defect: String,
}

/// [`TensorDefect`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DefectReport {
    Exact,
    /// `|stored − represented| ≤ bound · |stored|` entrywise.
    Relative { bound: f64 },
    /// Id of the entrywise bound.
    Entrywise { bound: String },
}

/// [`crate::parameter_decomposition::canonical::LayerGaugeElements`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ElementsReport {
    /// Id of `D⁻¹` of the input norm's element (the native gain).
    pub input_norm: String,
    pub post_norm: String,
    /// Ids of `T_g` per key/value group.
    pub value_output: Vec<String>,
    pub value_output_separated: Vec<bool>,
    pub swiglu_permutation: Vec<usize>,
    /// Id of the SwiGLU unit scales.
    pub swiglu_scales: String,
    pub query_key: Option<QueryKeyElementReport>,
}

/// [`crate::parameter_decomposition::gauge::NormedRotaryChange`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct QueryKeyElementReport {
    /// Id of `ρ_p` per rotated plane.
    pub plane_scales: String,
    /// `n_kv_heads × planes` quarter turns in `0..4`.
    pub quarter_turns: Vec<Vec<u8>>,
    pub query_gain_flips: Vec<bool>,
    pub key_gain_flips: Vec<bool>,
}

/// Both layers executed on the declared rows.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExecutionReport {
    pub native: ExecutionIds,
    pub canonical: ExecutionIds,
    /// `max |native − canonical| − (r_native + r_canonical)` over the output
    /// (`LayerExecution::largest_excess`): positive certifies different functions at
    /// these rows. Absent when the output has no entries.
    pub largest_excess: Option<f64>,
    pub at: [usize; 2],
}

/// Ids of one [`LayerExecution`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ExecutionIds {
    pub attention_residual: String,
    pub attention_radius: String,
    pub output: String,
    pub output_radius: String,
}

struct Arrays(BTreeMap<String, ArrayD<f64>>);

impl Arrays {
    fn matrix(&mut self, id: String, array: Array2<f64>) -> String {
        self.0.insert(id.clone(), array.into_dyn());
        id
    }

    fn vector(&mut self, id: String, array: Array1<f64>) -> String {
        self.0.insert(id.clone(), array.into_dyn());
        id
    }

    fn norm(&mut self, prefix: &str, norm: LayerRmsNorm) -> Result<NormReport, MpdSurfaceError> {
        Ok(NormReport {
            epsilon: norm.epsilon,
            gain: self.vector(format!("{prefix}/gain"), norm.gain),
            gain_defect: finite("gain_defect", norm.gain_defect)?,
        })
    }

    fn projection(&mut self, prefix: &str, projection: LayerProjection) -> Result<ProjectionReport, MpdSurfaceError> {
        let weight_defect = match projection.weight_defect {
            TensorDefect::Exact => DefectReport::Exact,
            TensorDefect::Relative(bound) => DefectReport::Relative {
                bound: finite("weight_defect", bound)?,
            },
            TensorDefect::Entrywise(bound) => DefectReport::Entrywise {
                bound: self.matrix(format!("{prefix}/weight_defect"), bound),
            },
        };
        Ok(ProjectionReport {
            weight: self.matrix(format!("{prefix}/weight"), projection.weight),
            bias: self.vector(format!("{prefix}/bias"), projection.bias),
            weight_defect,
            bias_defect: self.vector(format!("{prefix}/bias_defect"), projection.bias_defect),
        })
    }

    fn execution(&mut self, prefix: &str, execution: LayerExecution) -> ExecutionIds {
        ExecutionIds {
            attention_residual: self.matrix(format!("{prefix}/attention_residual"), execution.attention_residual),
            attention_radius: self.matrix(format!("{prefix}/attention_radius"), execution.attention_radius),
            output: self.matrix(format!("{prefix}/output"), execution.output),
            output_radius: self.matrix(format!("{prefix}/output_radius"), execution.output_radius),
        }
    }
}

pub(super) fn run(
    request: CanonicalLayerRequest,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    // The owned layer, its canonical copy, and each owner's working copies.
    let mut entries = 0usize;
    for id in [&request.gate, &request.up, &request.down] {
        entries = entries.saturating_add(input(tensors, id)?.len());
    }
    for projection in [
        &request.attention.query,
        &request.attention.key,
        &request.attention.value,
        &request.attention.output,
    ] {
        entries = entries.saturating_add(input(tensors, &projection.weight)?.len());
    }
    let copies = reserve(governor, entries, 1, 4, "canonical layer: owned copies")?;
    let attention = native_attention(tensors, &request.attention)?;
    let mlp = SwigluUnits::new(
        matrix(tensors, &request.gate)?.to_owned(),
        matrix(tensors, &request.up)?.to_owned(),
        matrix(tensors, &request.down)?.to_owned(),
    )
    .map_err(MpdSurfaceError::Gauge)?;
    let native = DecoderLayer::native(
        LayerRmsNorm::native(request.input_norm.epsilon, vector(tensors, &request.input_norm.gain)?.to_owned()),
        &attention,
        LayerRmsNorm::native(request.post_norm.epsilon, vector(tensors, &request.post_norm.gain)?.to_owned()),
        &mlp,
    )
    .map_err(|refusal| MpdSurfaceError::Canonical(Box::new(refusal)))?;
    drop(attention);
    drop(mlp);
    let canonical = native
        .canonical()
        .map_err(|refusal| MpdSurfaceError::Canonical(Box::new(refusal)))?;
    let executions = match &request.execute {
        None => None,
        Some(execute) => {
            let residual = matrix(tensors, &execute.residual)?;
            let run = |layer: &DecoderLayer| {
                layer
                    .execute(governor, ProjectedRows::exact(residual), &execute.positions)
                    .map_err(|refusal| MpdSurfaceError::Canonical(Box::new(refusal)))
            };
            Some((run(&native)?, run(&canonical.layer)?))
        }
    };
    drop(native);
    let projected = project(canonical, executions);
    drop(copies);
    projected
}

pub(super) fn project(
    canonical: CanonicalLayer,
    executions: Option<(LayerExecution, LayerExecution)>,
) -> Result<MpdOutput, MpdSurfaceError> {
    let mut arrays = Arrays(BTreeMap::new());
    let CanonicalLayer { layer, elements } = canonical;
    let query_key_norm = match layer.query_key_norm {
        None => None,
        Some(norm) => Some(QueryKeyNormReport {
            query: arrays.norm("query_key_norm/query", norm.query)?,
            key: arrays.norm("query_key_norm/key", norm.key)?,
        }),
    };
    let value_output = elements
        .value_output
        .into_iter()
        .enumerate()
        .map(|(group, change)| arrays.matrix(format!("elements/value_output/{group}"), change))
        .collect();
    let query_key = elements.query_key.map(|change| QueryKeyElementReport {
        plane_scales: arrays.vector("elements/query_key/plane_scales".to_string(), change.plane_scales),
        quarter_turns: change.quarter_turns.outer_iter().map(|row| row.to_vec()).collect(),
        query_gain_flips: change.query_gain_flips,
        key_gain_flips: change.key_gain_flips,
    });
    let execution = match executions {
        None => None,
        Some((before, after)) => {
            let (excess, at) = before.largest_excess(&after);
            let largest_excess = if excess == f64::NEG_INFINITY {
                None
            } else {
                Some(finite("largest_excess", excess)?)
            };
            Some(ExecutionReport {
                native: arrays.execution("execution/native", before),
                canonical: arrays.execution("execution/canonical", after),
                largest_excess,
                at: [at.0, at.1],
            })
        }
    };
    let report = CanonicalLayerReport {
        input_norm: arrays.norm("input_norm", layer.input_norm)?,
        query: arrays.projection("query", layer.query)?,
        key: arrays.projection("key", layer.key)?,
        value: arrays.projection("value", layer.value)?,
        output: arrays.projection("output", layer.output)?,
        query_key_norm,
        post_norm: arrays.norm("post_norm", layer.post_norm)?,
        gate: arrays.projection("gate", layer.gate)?,
        up: arrays.projection("up", layer.up)?,
        down: arrays.projection("down", layer.down)?,
        elements: ElementsReport {
            input_norm: arrays.vector("elements/input_norm".to_string(), elements.input_norm),
            post_norm: arrays.vector("elements/post_norm".to_string(), elements.post_norm),
            value_output,
            value_output_separated: elements.value_output_separated,
            swiglu_permutation: elements.swiglu.permutation,
            swiglu_scales: arrays.vector("elements/swiglu_scales".to_string(), elements.swiglu.scales),
            query_key,
        },
        execution,
    };
    Ok(output(MpdResult::CanonicalLayer(Box::new(report)), arrays.0))
}

#[cfg(test)]
mod tests {
    use super::super::run_parameter_decomposition;
    use super::super::tests::request_json;
    use super::*;
    use crate::parameter_decomposition::attention::{
        AffineProjection, AttentionGeometry, NativeAttention, RotaryEmbedding, RotaryPairing,
    };
    use crate::parameter_decomposition::canonical::CanonicalRefusal;
    use crate::parameter_decomposition::test_support::test_governor;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    const WIDTH: usize = 8;
    const HIDDEN: usize = 6;
    const POSITIONS: [i64; 3] = [0, 1, 2];

    fn uniform(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || rng.random_range(-1.0..1.0))
    }

    /// Gains in `[1/2, 3/2]` with one coordinate sign-flipped.
    fn gains(rng: &mut StdRng, len: usize, negative: usize) -> Array1<f64> {
        let mut gain = Array1::from_shape_simple_fn(len, || rng.random_range(0.5..1.5));
        gain[negative] = -gain[negative];
        gain
    }

    fn tensors(seed: u64) -> BTreeMap<String, ArrayD<f64>> {
        let mut rng = StdRng::seed_from_u64(seed);
        BTreeMap::from([
            ("q".to_string(), uniform(&mut rng, 16, WIDTH).into_dyn()),
            ("k".to_string(), uniform(&mut rng, 8, WIDTH).into_dyn()),
            ("v".to_string(), uniform(&mut rng, 8, WIDTH).into_dyn()),
            ("o".to_string(), uniform(&mut rng, WIDTH, 16).into_dyn()),
            ("qn".to_string(), gains(&mut rng, 4, 1).into_dyn()),
            ("kn".to_string(), gains(&mut rng, 4, 2).into_dyn()),
            ("gate".to_string(), uniform(&mut rng, HIDDEN, WIDTH).into_dyn()),
            ("up".to_string(), uniform(&mut rng, HIDDEN, WIDTH).into_dyn()),
            ("down".to_string(), uniform(&mut rng, WIDTH, HIDDEN).into_dyn()),
            ("n1".to_string(), gains(&mut rng, WIDTH, 3).into_dyn()),
            ("n2".to_string(), gains(&mut rng, WIDTH, 5).into_dyn()),
            ("h".to_string(), (uniform(&mut rng, POSITIONS.len(), WIDTH) * 2.0).into_dyn()),
        ])
    }

    fn request(execute: &str) -> String {
        request_json(&format!(
            r#"{{"kind": "canonical_layer",
                "attention": {{"geometry": {{"model_dim": 8, "n_heads": 4, "n_kv_heads": 2, "head_dim": 4}},
                    "rotary": {{"pairing": "half_split", "inverse_frequencies": [1.0, 0.1], "attention_scaling": 1.0}},
                    "score_scale": 0.5, "query": {{"weight": "q", "bias": null}}, "key": {{"weight": "k", "bias": null}},
                    "value": {{"weight": "v", "bias": null}}, "output": {{"weight": "o", "bias": null}},
                    "query_key_norm": {{"epsilon": 1e-6, "query_gain": "qn", "key_gain": "kn"}}}},
                "input_norm": {{"epsilon": 1e-6, "gain": "n1"}}, "post_norm": {{"epsilon": 1e-6, "gain": "n2"}},
                "gate": "gate", "up": "up", "down": "down", "execute": {execute}}}"#
        ))
    }

    fn owner(tensors: &BTreeMap<String, ArrayD<f64>>) -> DecoderLayer {
        let m = |id: &str| matrix(tensors, id).expect("matrix").to_owned();
        let v = |id: &str| vector(tensors, id).expect("vector").to_owned();
        let affine = |weight: Array2<f64>| AffineProjection { bias: Array1::zeros(weight.nrows()), weight };
        let attention = NativeAttention::new(
            AttentionGeometry { model_dim: WIDTH, n_heads: 4, n_kv_heads: 2, head_dim: 4 },
            RotaryEmbedding { pairing: RotaryPairing::HalfSplit, inverse_frequencies: vec![1.0, 0.1], attention_scaling: 1.0 },
            0.5,
            affine(m("q")),
            affine(m("k")),
            affine(m("v")),
            affine(m("o")),
        )
        .and_then(|native| native.with_query_key_norm(1e-6, v("qn"), v("kn")))
        .expect("native attention");
        let mlp = SwigluUnits::new(m("gate"), m("up"), m("down")).expect("mlp");
        DecoderLayer::native(LayerRmsNorm::native(1e-6, v("n1")), &attention, LayerRmsNorm::native(1e-6, v("n2")), &mlp)
            .expect("native layer")
    }

    #[test]
    fn canonical_layer_report_is_the_owner_result_field_for_field() {
        let tensors = tensors(21);
        let native = owner(&tensors);
        let canonical = native.canonical().expect("owner canonical form");
        let residual = matrix(&tensors, "h").expect("h");
        let execute = |layer: &DecoderLayer| {
            layer.execute(test_governor(), ProjectedRows::exact(residual), &POSITIONS).expect("owner execution")
        };
        let executions = (execute(&native), execute(&canonical.layer));
        let expected = project(canonical, Some(executions)).expect("owner projection");
        let output = run_parameter_decomposition(
            &request(r#"{"residual": "h", "positions": [0, 1, 2]}"#),
            &tensors,
            test_governor(),
        )
        .expect("surface run");
        assert_eq!(output, expected);
        let MpdResult::CanonicalLayer(report) = &output.report.result else {
            panic!("expected a canonical layer, got {:?}", output.report.result);
        };
        // The canonical layer executes the native function within the bands, and its
        // gains are the unit representative.
        let execution = report.execution.as_ref().expect("execution");
        assert!(execution.largest_excess.expect("entries") <= 0.0, "{execution:?}");
        assert!(output.arrays[&report.input_norm.gain].iter().all(|&gain| gain == 1.0));
        assert_eq!(output.arrays[&report.elements.input_norm], tensors["n1"]);
        assert!(report.elements.query_key.is_some());
        assert_eq!(report.elements.value_output.len(), 2);
        assert!(matches!(report.value.weight_defect, DefectReport::Entrywise { .. }));
        let json: serde_json::Value =
            serde_json::from_str(&output.report_json().expect("report json")).expect("parse report");
        assert_eq!(json["result"]["kind"], "canonical_layer");
        assert_eq!(json["result"]["query"]["weight_defect"]["kind"], "relative");
    }

    #[test]
    fn a_canonical_request_the_owners_refuse_reaches_the_caller() {
        let good = tensors(22);
        assert!(run_parameter_decomposition(&request("null"), &good, test_governor()).is_ok());
        // A zero norm gain is a null coordinate, not an orbit coordinate.
        let mut zero_gain = good.clone();
        zero_gain.get_mut("n1").expect("n1")[[0]] = 0.0;
        assert!(matches!(
            run_parameter_decomposition(&request("null"), &zero_gain, test_governor()),
            Err(MpdSurfaceError::Canonical(refusal)) if matches!(*refusal, CanonicalRefusal::ZeroNormGain { coordinate: 0 })
        ));
        // Residual rows of the wrong width, and positions that do not match the rows.
        let mut narrow = good.clone();
        narrow.insert("h".to_string(), Array2::<f64>::zeros((3, 4)).into_dyn());
        assert!(matches!(
            run_parameter_decomposition(&request(r#"{"residual": "h", "positions": [0, 1, 2]}"#), &narrow, test_governor()),
            Err(MpdSurfaceError::Canonical(_))
        ));
        assert!(matches!(
            run_parameter_decomposition(&request(r#"{"residual": "h", "positions": [0, 1]}"#), &good, test_governor()),
            Err(MpdSurfaceError::Canonical(_))
        ));
        let stray = request("null").replacen("\"gate\": \"gate\"", "\"gate\": \"gate\", \"balance\": true", 1);
        assert!(matches!(
            run_parameter_decomposition(&stray, &good, test_governor()),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
    }
}
