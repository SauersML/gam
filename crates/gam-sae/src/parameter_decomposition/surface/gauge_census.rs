//! `gauge_census`: the [`crate::parameter_decomposition::gauge`] families and the
//! [`crate::parameter_decomposition::gauge_census`] owner over supplied tensors.

use std::collections::BTreeMap;

use gam_math::gaussian_activation::GaussianActivation;
use gam_runtime::resource::{MemoryGovernor, MemoryReservation};
use ndarray::ArrayD;
use serde::{Deserialize, Serialize};

use super::layer::{AttentionRequest, native_attention};
use super::{MpdOutput, MpdResult, MpdSurfaceError, input, matrix, output, reserve, vector};
use crate::parameter_decomposition::gauge::{
    GaugeFamily, GaugeRefusal, HiddenUnits, LinearPassthrough, NormGain, QueryKeyGauge, SwigluUnits,
};
use crate::parameter_decomposition::gauge_census::{
    CensusCharge, DecoderLayerTensors, TiedResidualCensus, decoder_layer_census, tied_residual_census,
};

/// The blocks to census, in order. Every string field is the id of an input array.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GaugeCensusRequest {
    pub blocks: Vec<CensusBlock>,
}

/// One block and the family (or families) that act on it.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum CensusBlock {
    /// A read `A` (`r × d_in`) and a write `B` (`d_out × r`) with nothing between them
    /// (`gauge::LinearPassthrough`, `GL(r)`).
    LinearPassthrough { read: String, write: String },
    /// `W₂ σ(W₁ x + b₁)` for an elementwise activation named by its Hugging Face tag
    /// (`gauge::HiddenUnits`).
    HiddenUnits {
        read_in: String,
        bias_in: String,
        write_out: String,
        hidden_act: String,
    },
    /// `W_d [s(W_g x) ⊙ W_u x]` (`gauge::SwigluUnits`).
    SwigluUnits { gate: String, up: String, down: String },
    /// A norm output `w ⊙ n + β` read only by linear reads (`gauge::NormGain`).
    NormGain {
        gain: String,
        bias: Option<String>,
        reads: Vec<String>,
    },
    /// The query/key gauge of a native attention block (`gauge::QueryKeyGauge`).
    QueryKey { attention: AttentionRequest },
    /// Every family of one bias-free sequential pre-norm decoder layer
    /// (`gauge_census::decoder_layer_census`).
    DecoderLayer {
        attention: AttentionRequest,
        gate: String,
        up: String,
        down: String,
        input_norm: String,
        post_norm: String,
    },
    /// A residual stream under a tied embedding/unembedding
    /// (`gauge_census::tied_residual_census`): the final norm gain, embedding rows that
    /// resolve its rank from below, and the vocabulary size.
    TiedResidual {
        final_norm: String,
        embedding_rows: String,
        vocab: usize,
    },
}

/// Every block's families and charges, and the total.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GaugeCensusReport {
    pub blocks: Vec<CensusBlockReport>,
    /// The blocks' charges summed; they add when the blocks move disjoint coordinates.
    pub total: CensusChargeReport,
}

/// One block's families, each charge, and the block's total charge.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CensusBlockReport {
    pub families: Vec<NamedFamily>,
    pub charges: Vec<NamedCharge>,
    pub charge: CensusChargeReport,
    /// For a tied residual stream.
    pub residual: Option<TiedResidualReport>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NamedFamily {
    pub name: String,
    pub family: GaugeFamilyReport,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct NamedCharge {
    pub name: String,
    pub charge: CensusChargeReport,
}

/// [`GaugeFamily`] on the wire. The group names are the owner's type names.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct GaugeFamilyReport {
    pub kind: String,
    pub continuous: String,
    pub continuous_dimension: usize,
    pub discrete: String,
    pub parameter_coordinates: usize,
    pub orbit_resolved: usize,
    pub orbit_at_most: usize,
    pub null_coordinates: usize,
    pub real_coordinates_at_most: usize,
}

/// [`CensusCharge`] on the wire.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct CensusChargeReport {
    pub parameters: usize,
    pub orbit_resolved: usize,
    pub orbit_at_most: usize,
    pub null: usize,
    pub real_coordinates_at_most: usize,
    /// Absent for a block of no stored coordinates.
    pub convention_fraction: Option<f64>,
}

/// [`TiedResidualCensus`] on the wire.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TiedResidualReport {
    pub untied: String,
    pub untied_dimension: usize,
    pub equal_gain_groups: Vec<usize>,
    pub tied_dimension: usize,
    pub embedding_rank: usize,
    pub embedding_rows: usize,
}

pub(super) fn family_report(family: &GaugeFamily) -> GaugeFamilyReport {
    GaugeFamilyReport {
        kind: format!("{:?}", family.kind),
        continuous: format!("{:?}", family.continuous),
        continuous_dimension: family.continuous.dimension(),
        discrete: format!("{:?}", family.discrete),
        parameter_coordinates: family.parameter_coordinates,
        orbit_resolved: family.orbit_dimension.resolved,
        orbit_at_most: family.orbit_dimension.at_most,
        null_coordinates: family.null_coordinates,
        real_coordinates_at_most: family.real_coordinates_at_most(),
    }
}

pub(super) fn charge_report(charge: CensusCharge) -> CensusChargeReport {
    CensusChargeReport {
        parameters: charge.parameters,
        orbit_resolved: charge.orbit_resolved,
        orbit_at_most: charge.orbit_at_most,
        null: charge.null,
        real_coordinates_at_most: charge.real_coordinates_at_most(),
        convention_fraction: (charge.parameters > 0).then(|| charge.convention_fraction()),
    }
}

/// Reserves the owned copies the gauge owners take of `ids`.
fn reserve_copies(
    governor: &MemoryGovernor,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    ids: &[&String],
) -> Result<MemoryReservation, MpdSurfaceError> {
    let mut entries = 0usize;
    for id in ids {
        entries = entries.saturating_add(input(tensors, id)?.len());
    }
    // Twice: the owners copy some reads again while forming stacked blocks.
    reserve(governor, entries, 1, 2, "gauge census: owned copies")
}

fn gauge(refusal: GaugeRefusal) -> MpdSurfaceError {
    MpdSurfaceError::Gauge(refusal)
}

fn single(name: &str, family: GaugeFamily) -> CensusBlockReport {
    let charge = CensusCharge::of(&family, family.parameter_coordinates);
    CensusBlockReport {
        families: vec![NamedFamily {
            name: name.to_string(),
            family: family_report(&family),
        }],
        charges: vec![NamedCharge {
            name: name.to_string(),
            charge: charge_report(charge),
        }],
        charge: charge_report(charge),
        residual: None,
    }
}

fn attention_ids(attention: &AttentionRequest) -> Vec<&String> {
    let mut ids = Vec::new();
    for projection in [&attention.query, &attention.key, &attention.value, &attention.output] {
        ids.push(&projection.weight);
        ids.extend(projection.bias.iter());
    }
    if let Some(norm) = &attention.query_key_norm {
        ids.push(&norm.query_gain);
        ids.push(&norm.key_gain);
    }
    ids
}

fn census_block(
    block: &CensusBlock,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<(CensusBlockReport, CensusCharge), MpdSurfaceError> {
    let owned = |ids: &[&String]| reserve_copies(governor, tensors, ids);
    let report = match block {
        CensusBlock::LinearPassthrough { read, write } => {
            let copies = owned(&[read, write])?;
            let family = LinearPassthrough::new(matrix(tensors, read)?.to_owned(), matrix(tensors, write)?.to_owned())
                .and_then(|block| block.family())
                .map_err(gauge)?;
            drop(copies);
            single("linear_passthrough", family)
        }
        CensusBlock::HiddenUnits {
            read_in,
            bias_in,
            write_out,
            hidden_act,
        } => {
            let activation = GaussianActivation::from_hidden_act(hidden_act)
                .map_err(|error| MpdSurfaceError::InvalidRequest(format!("{error:?}")))?;
            let copies = owned(&[read_in, bias_in, write_out])?;
            let family = HiddenUnits::new(
                matrix(tensors, read_in)?.to_owned(),
                vector(tensors, bias_in)?.to_owned(),
                matrix(tensors, write_out)?.to_owned(),
                activation,
            )
            .map_err(gauge)?
            .family();
            drop(copies);
            single("hidden_units", family)
        }
        CensusBlock::SwigluUnits { gate, up, down } => {
            let copies = owned(&[gate, up, down])?;
            let family = SwigluUnits::new(
                matrix(tensors, gate)?.to_owned(),
                matrix(tensors, up)?.to_owned(),
                matrix(tensors, down)?.to_owned(),
            )
            .map_err(gauge)?
            .family();
            drop(copies);
            single("swiglu", family)
        }
        CensusBlock::NormGain { gain, bias, reads } => {
            let mut ids: Vec<&String> = reads.iter().collect();
            ids.push(gain);
            ids.extend(bias.iter());
            let copies = owned(&ids)?;
            let bias = match bias {
                Some(id) => Some(vector(tensors, id)?.to_owned()),
                None => None,
            };
            let reads = reads
                .iter()
                .map(|id| matrix(tensors, id).map(|read| read.to_owned()))
                .collect::<Result<Vec<_>, _>>()?;
            let family = NormGain::new(vector(tensors, gain)?.to_owned(), bias, reads)
                .map_err(gauge)?
                .family();
            drop(copies);
            single("norm_gain", family)
        }
        CensusBlock::QueryKey { attention } => {
            let copies = owned(&attention_ids(attention))?;
            let native = native_attention(tensors, attention)?;
            let family = QueryKeyGauge::new(&native)
                .and_then(|gauge| gauge.family())
                .map_err(gauge)?;
            drop(native);
            drop(copies);
            single("qk", family)
        }
        CensusBlock::DecoderLayer {
            attention,
            gate,
            up,
            down,
            input_norm,
            post_norm,
        } => {
            if [&attention.query, &attention.key, &attention.value, &attention.output]
                .iter()
                .any(|projection| projection.bias.is_some())
            {
                return Err(MpdSurfaceError::InvalidRequest(
                    "the decoder-layer census reads bias-free projections".to_string(),
                ));
            }
            let mut ids = attention_ids(attention);
            ids.extend([gate, up, down, input_norm, post_norm]);
            let copies = owned(&ids)?;
            let query_key_norm = match &attention.query_key_norm {
                Some(norm) => Some((vector(tensors, &norm.query_gain)?, vector(tensors, &norm.key_gain)?)),
                None => None,
            };
            let census = decoder_layer_census(
                attention.geometry.into(),
                &(&attention.rotary).into(),
                attention.score_scale,
                attention.query_key_norm.as_ref().map_or(0.0, |norm| norm.epsilon),
                DecoderLayerTensors {
                    query: matrix(tensors, &attention.query.weight)?,
                    key: matrix(tensors, &attention.key.weight)?,
                    value: matrix(tensors, &attention.value.weight)?,
                    output: matrix(tensors, &attention.output.weight)?,
                    query_key_norm,
                    gate: matrix(tensors, gate)?,
                    up: matrix(tensors, up)?,
                    down: matrix(tensors, down)?,
                    input_norm: vector(tensors, input_norm)?,
                    post_norm: vector(tensors, post_norm)?,
                },
            )
            .map_err(MpdSurfaceError::Census)?;
            drop(copies);
            let mut families: Vec<NamedFamily> = census
                .ov
                .iter()
                .enumerate()
                .map(|(group, family)| NamedFamily {
                    name: format!("ov/{group}"),
                    family: family_report(family),
                })
                .collect();
            for (name, family) in [
                ("qk", &census.qk),
                ("input_norm", &census.input_norm),
                ("post_norm", &census.post_norm),
                ("swiglu", &census.swiglu),
            ] {
                families.push(NamedFamily {
                    name: name.to_string(),
                    family: family_report(family),
                });
            }
            CensusBlockReport {
                families,
                charges: census
                    .charges()
                    .into_iter()
                    .map(|(name, charge)| NamedCharge {
                        name: name.to_string(),
                        charge: charge_report(charge),
                    })
                    .collect(),
                charge: charge_report(census.charge()),
                residual: None,
            }
        }
        CensusBlock::TiedResidual {
            final_norm,
            embedding_rows,
            vocab,
        } => {
            let copies = owned(&[embedding_rows])?;
            let census: TiedResidualCensus =
                tied_residual_census(vector(tensors, final_norm)?, matrix(tensors, embedding_rows)?, *vocab)
                    .map_err(MpdSurfaceError::Census)?;
            drop(copies);
            CensusBlockReport {
                families: Vec::new(),
                charges: vec![NamedCharge {
                    name: "residual".to_string(),
                    charge: charge_report(census.charge),
                }],
                charge: charge_report(census.charge),
                residual: Some(TiedResidualReport {
                    untied: format!("{:?}", census.untied),
                    untied_dimension: census.untied.dimension(),
                    equal_gain_groups: census.equal_gain_groups,
                    tied_dimension: census.tied_dimension,
                    embedding_rank: census.embedding_rank,
                    embedding_rows: census.embedding_rows,
                }),
            }
        }
    };
    let charge = CensusCharge {
        parameters: report.charge.parameters,
        orbit_resolved: report.charge.orbit_resolved,
        orbit_at_most: report.charge.orbit_at_most,
        null: report.charge.null,
    };
    Ok((report, charge))
}

pub(super) fn run(
    request: GaugeCensusRequest,
    tensors: &BTreeMap<String, ArrayD<f64>>,
    governor: &MemoryGovernor,
) -> Result<MpdOutput, MpdSurfaceError> {
    let mut blocks = Vec::with_capacity(request.blocks.len());
    let mut total = CensusCharge::default();
    for block in &request.blocks {
        let (report, charge) = census_block(block, tensors, governor)?;
        total.add(charge);
        blocks.push(report);
    }
    Ok(output(
        MpdResult::GaugeCensus(GaugeCensusReport {
            blocks,
            total: charge_report(total),
        }),
        BTreeMap::new(),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::run_parameter_decomposition;
    use super::super::tests::request_json;
    use super::*;
    use crate::parameter_decomposition::attention::{
        AffineProjection, AttentionGeometry, NativeAttention, RotaryEmbedding, RotaryPairing,
    };
    use crate::parameter_decomposition::test_support::test_governor;
    use ndarray::{Array1, Array2};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn uniform(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || rng.random_range(-1.0..1.0))
    }

    fn gains(rng: &mut StdRng, len: usize) -> Array1<f64> {
        Array1::from_shape_simple_fn(len, || rng.random_range(0.5..1.5))
    }

    const WIDTH: usize = 8;
    const HIDDEN: usize = 6;

    fn geometry() -> AttentionGeometry {
        AttentionGeometry {
            model_dim: WIDTH,
            n_heads: 4,
            n_kv_heads: 2,
            head_dim: 4,
        }
    }

    fn rotary() -> RotaryEmbedding {
        RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: vec![1.0, 0.1],
            attention_scaling: 1.0,
        }
    }

    /// A Qwen3-shaped layer's tensors (GQA 4/2, two rotary planes, a query/key norm).
    fn layer_tensors() -> BTreeMap<String, ArrayD<f64>> {
        let mut rng = StdRng::seed_from_u64(7);
        let g = geometry();
        BTreeMap::from([
            ("q".to_string(), uniform(&mut rng, g.query_dim(), WIDTH).into_dyn()),
            ("k".to_string(), uniform(&mut rng, g.key_value_dim(), WIDTH).into_dyn()),
            ("v".to_string(), uniform(&mut rng, g.key_value_dim(), WIDTH).into_dyn()),
            ("o".to_string(), uniform(&mut rng, WIDTH, g.query_dim()).into_dyn()),
            ("qn".to_string(), gains(&mut rng, g.head_dim).into_dyn()),
            ("kn".to_string(), gains(&mut rng, g.head_dim).into_dyn()),
            ("gate".to_string(), uniform(&mut rng, HIDDEN, WIDTH).into_dyn()),
            ("up".to_string(), uniform(&mut rng, HIDDEN, WIDTH).into_dyn()),
            ("down".to_string(), uniform(&mut rng, WIDTH, HIDDEN).into_dyn()),
            ("n1".to_string(), gains(&mut rng, WIDTH).into_dyn()),
            ("n2".to_string(), gains(&mut rng, WIDTH).into_dyn()),
            ("embed".to_string(), uniform(&mut rng, 20, WIDTH).into_dyn()),
        ])
    }

    const ATTENTION: &str = r#"{"geometry": {"model_dim": 8, "n_heads": 4, "n_kv_heads": 2, "head_dim": 4},
        "rotary": {"pairing": "half_split", "inverse_frequencies": [1.0, 0.1], "attention_scaling": 1.0},
        "score_scale": 0.5, "query": {"weight": "q", "bias": null}, "key": {"weight": "k", "bias": null},
        "value": {"weight": "v", "bias": null}, "output": {"weight": "o", "bias": null},
        "query_key_norm": {"epsilon": 1e-6, "query_gain": "qn", "key_gain": "kn"}}"#;

    fn census(blocks: &str, tensors: &BTreeMap<String, ArrayD<f64>>) -> Result<MpdOutput, MpdSurfaceError> {
        run_parameter_decomposition(
            &request_json(&format!(r#"{{"kind": "gauge_census", "blocks": {blocks}}}"#)),
            tensors,
            test_governor(),
        )
    }

    fn report(output: &MpdOutput) -> &GaugeCensusReport {
        let MpdResult::GaugeCensus(report) = &output.report.result else {
            panic!("expected a gauge census, got {:?}", output.report.result);
        };
        report
    }

    #[test]
    fn decoder_layer_and_residual_census_is_the_owner_result_field_for_field() {
        let tensors = layer_tensors();
        let at = |id: &str| tensors[id].view();
        let m = |id: &str| at(id).into_dimensionality::<ndarray::Ix2>().expect("matrix");
        let v = |id: &str| at(id).into_dimensionality::<ndarray::Ix1>().expect("vector");
        let owner = decoder_layer_census(
            geometry(),
            &rotary(),
            0.5,
            1e-6,
            DecoderLayerTensors {
                query: m("q"),
                key: m("k"),
                value: m("v"),
                output: m("o"),
                query_key_norm: Some((v("qn"), v("kn"))),
                gate: m("gate"),
                up: m("up"),
                down: m("down"),
                input_norm: v("n1"),
                post_norm: v("n2"),
            },
        )
        .expect("owner census");
        let residual = tied_residual_census(v("n1"), m("embed"), 100).expect("owner residual");
        let output = census(
            &format!(
                r#"[{{"kind": "decoder_layer", "attention": {ATTENTION}, "gate": "gate", "up": "up", "down": "down",
                     "input_norm": "n1", "post_norm": "n2"}},
                   {{"kind": "tied_residual", "final_norm": "n1", "embedding_rows": "embed", "vocab": 100}}]"#
            ),
            &tensors,
        )
        .expect("surface run");
        let report = report(&output);
        let layer = &report.blocks[0];
        let mut expected_families: Vec<(String, GaugeFamilyReport)> = owner
            .ov
            .iter()
            .enumerate()
            .map(|(group, family)| (format!("ov/{group}"), family_report(family)))
            .collect();
        for (name, family) in [("qk", &owner.qk), ("input_norm", &owner.input_norm), ("post_norm", &owner.post_norm), ("swiglu", &owner.swiglu)] {
            expected_families.push((name.to_string(), family_report(family)));
        }
        assert_eq!(
            layer.families.iter().map(|named| (named.name.clone(), named.family.clone())).collect::<Vec<_>>(),
            expected_families
        );
        assert_eq!(
            layer.charges.iter().map(|named| (named.name.clone(), named.charge)).collect::<Vec<_>>(),
            owner.charges().into_iter().map(|(name, charge)| (name.to_string(), charge_report(charge))).collect::<Vec<_>>()
        );
        assert_eq!(layer.charge, charge_report(owner.charge()));
        // GL(4) per key/value group resolves exactly at generic tensors.
        assert!(owner.ov_exact());
        assert_eq!(layer.families[0].family.orbit_resolved, 16);
        let wire = report.blocks[1].residual.as_ref().expect("residual");
        assert_eq!(wire.equal_gain_groups, residual.equal_gain_groups);
        assert_eq!((wire.tied_dimension, wire.embedding_rank, wire.embedding_rows), (residual.tied_dimension, residual.embedding_rank, residual.embedding_rows));
        assert_eq!(wire.untied_dimension, WIDTH * (WIDTH - 1) / 2);
        assert_eq!(report.blocks[1].charge, charge_report(residual.charge));
        let mut total = owner.charge();
        total.add(residual.charge);
        assert_eq!(report.total, charge_report(total));
    }

    #[test]
    fn single_family_blocks_are_the_owner_families() {
        let mut tensors = layer_tensors();
        let mut rng = StdRng::seed_from_u64(9);
        let (read, bias, write) = (uniform(&mut rng, 5, WIDTH), Array1::zeros(5), uniform(&mut rng, WIDTH, 5));
        tensors.insert("w1".to_string(), read.clone().into_dyn());
        tensors.insert("b1".to_string(), bias.clone().into_dyn());
        tensors.insert("w2".to_string(), write.clone().into_dyn());
        tensors.insert("a".to_string(), uniform(&mut rng, 2, WIDTH).into_dyn());
        tensors.insert("b".to_string(), uniform(&mut rng, WIDTH, 2).into_dyn());
        let unnormed = ATTENTION.replace(r#""query_key_norm": {"epsilon": 1e-6, "query_gain": "qn", "key_gain": "kn"}"#, r#""query_key_norm": null"#);
        let output = census(
            &format!(
                r#"[{{"kind": "linear_passthrough", "read": "a", "write": "b"}},
                   {{"kind": "hidden_units", "read_in": "w1", "bias_in": "b1", "write_out": "w2", "hidden_act": "gelu"}},
                   {{"kind": "hidden_units", "read_in": "w1", "bias_in": "b1", "write_out": "w2", "hidden_act": "relu"}},
                   {{"kind": "swiglu_units", "gate": "gate", "up": "up", "down": "down"}},
                   {{"kind": "norm_gain", "gain": "n1", "bias": null, "reads": ["q", "k"]}},
                   {{"kind": "query_key", "attention": {unnormed}}}]"#
            ),
            &tensors,
        )
        .expect("surface run");
        let report = report(&output);
        let m = |id: &str| tensors[id].view().into_dimensionality::<ndarray::Ix2>().expect("matrix").to_owned();
        let passthrough = LinearPassthrough::new(m("a"), m("b")).and_then(|block| block.family()).expect("owner");
        let gelu = HiddenUnits::new(read.clone(), bias.clone(), write.clone(), GaussianActivation::ExactGelu).expect("owner").family();
        let relu = HiddenUnits::new(read, bias, write, GaussianActivation::Relu).expect("owner").family();
        let swiglu = SwigluUnits::new(m("gate"), m("up"), m("down")).expect("owner").family();
        let gains = tensors["n1"].view().into_dimensionality::<ndarray::Ix1>().expect("vector").to_owned();
        let norm = NormGain::new(gains, None, vec![m("q"), m("k")]).expect("owner").family();
        let bias_free = |weight: Array2<f64>| AffineProjection { bias: Array1::zeros(weight.nrows()), weight };
        let native = NativeAttention::new(geometry(), rotary(), 0.5, bias_free(m("q")), bias_free(m("k")), bias_free(m("v")), bias_free(m("o")))
            .expect("native");
        let qk = QueryKeyGauge::new(&native).and_then(|gauge| gauge.family()).expect("owner");
        for (block, family) in report.blocks.iter().zip([passthrough, gelu, relu, swiglu, norm, qk]) {
            assert_eq!(block.families[0].family, family_report(&family));
            assert_eq!(block.charge, charge_report(CensusCharge::of(&family, family.parameter_coordinates)));
        }
        // GL(2) for the pass-through; permutation only for exact GELU; positive scales for ReLU.
        assert_eq!(report.blocks[0].families[0].family.orbit_resolved, 4);
        assert_eq!(report.blocks[1].families[0].family.continuous_dimension, 0);
        assert_eq!(report.blocks[2].families[0].family.continuous_dimension, 5);
    }

    #[test]
    fn a_census_the_owners_refuse_reaches_the_caller() {
        let tensors = layer_tensors();
        assert!(census(r#"[{"kind": "swiglu_units", "gate": "gate", "up": "up", "down": "down"}]"#, &tensors).is_ok());
        assert!(matches!(
            census(r#"[{"kind": "swiglu_units", "gate": "gate", "up": "down", "down": "down"}]"#, &tensors),
            Err(MpdSurfaceError::Gauge(_))
        ));
        assert!(matches!(
            census(r#"[{"kind": "hidden_units", "read_in": "gate", "bias_in": "n1", "write_out": "down", "hidden_act": "gelu_new"}]"#, &tensors),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
        let biased = format!(
            r#"[{{"kind": "decoder_layer", "attention": {}, "gate": "gate", "up": "up", "down": "down", "input_norm": "n1", "post_norm": "n2"}}]"#,
            ATTENTION.replacen(r#"{"weight": "q", "bias": null}"#, r#"{"weight": "q", "bias": "n1"}"#, 1)
        );
        assert!(matches!(census(&biased, &tensors), Err(MpdSurfaceError::InvalidRequest(_))));
        let bad_geometry = format!(
            r#"[{{"kind": "query_key", "attention": {}}}]"#,
            ATTENTION.replace(r#""n_kv_heads": 2"#, r#""n_kv_heads": 3"#)
        );
        assert!(matches!(census(&bad_geometry, &tensors), Err(MpdSurfaceError::Attention(_))));
        assert!(matches!(
            census(r#"[{"kind": "tied_residual", "final_norm": "n1", "embedding_rows": "down", "vocab": 10}]"#, &tensors),
            Err(MpdSurfaceError::Census(_))
        ));
        assert!(matches!(
            census(r#"[{"kind": "swiglu_units", "gate": "gate", "up": "up", "down": "down", "units": 6}]"#, &tensors),
            Err(MpdSurfaceError::InvalidRequest(_))
        ));
    }
}
