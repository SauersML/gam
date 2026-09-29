//! The implementation-gauge census of a decoder layer and of a tied residual stream
//! (#2951): which [`super::gauge`] family acts on which stored coordinates, and how much
//! of the stored parameters is convention.
//!
//! # Per layer (a sequential pre-norm layer: Qwen3, Llama)
//!
//! - `ov`: one [`LinearPassthrough`] per key/value group, reading the group's value head
//!   and writing through the output columns of every query head sharing it (`GL(hd)`).
//! - `qk`: [`QueryKeyGauge`] on the native attention (the rotary commutant, or the normed
//!   rotary family behind a per-head query/key norm).
//! - `swiglu`: [`SwigluUnits`] (one nonzero up/down scale per unit).
//! - `input_norm`, `post_norm`: [`NormGain`] with their linear reads. Their reads are
//!   charged to `qk`, `ov` and `swiglu`, so each norm is charged its `d` gains only.
//!
//! Additivity: every family moves a tensor no other family moves (each norm gain, the
//! query/key norm gains), or acts on coordinates disjoint from the other families' (value
//! rows and output columns, up rows and down columns, the rows of coincident query/key
//! planes). So the joint orbit's tangent is the direct sum of the families' tangents, and
//! the resolved orbit dimensions add.
//!
//! # A tied residual stream
//!
//! [`ResidualStreamGauge`] gives `O(d)` for RMSNorm reads. A tied embedding/unembedding
//! restricts it: the embedding write forces `E ↦ E Qᵀ` and the final-norm read
//! `E diag(w) ↦ E diag(w) Qᵀ`, so `Q diag(w) Qᵀ` must stay diagonal. The identity component
//! is the block-orthogonal group over the groups of bitwise-equal final gains, of dimension
//! `Σ m_v (m_v − 1)/2`, and its orbit has that dimension when `E` has full column rank
//! (resolved on a row subset, which bounds the rank from below). The final norm's own
//! coordinate scales are not a gauge under the tie: they would rescale the embedding's
//! writes.
//!
//! # Evidence
//!
//! `orbit_resolved` counts ranks above the SVD's backward-error band (a Weyl lower bound)
//! and exact bit tests; `null` counts exact zero tests on the stored bits.
//! `real_coordinates_at_most = parameters − orbit_resolved − null` bounds the coordinates a
//! code must carry from above; it never claims that no further invariance exists.

use std::collections::BTreeMap;

use gam_linalg::faer_ndarray::FaerSvd;
use gam_linalg::roundoff::factor_singular_band;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, s};

use super::attention::{AffineProjection, AttentionGeometry, AttentionProgramError, NativeAttention, RotaryEmbedding};
use super::gauge::{
    ContinuousGauge, GaugeFamily, GaugeRefusal, LinearPassthrough, NormGain, QueryKeyGauge, ResidualRead,
    ResidualStreamGauge, SwigluUnits,
};

/// Why a census was declined.
#[derive(Clone, Debug, PartialEq)]
pub enum CensusRefusal {
    /// A gauge owner refused the family named by `family`.
    Gauge { family: &'static str, refusal: GaugeRefusal },
    /// The attention owner refused the layer's attention tensors.
    Attention(AttentionProgramError),
    /// The embedding rows' singular values did not converge.
    Decomposition { detail: String },
}

/// One census row: stored coordinates, the resolved and largest orbit dimension over
/// them, and the exact null coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CensusCharge {
    pub parameters: usize,
    pub orbit_resolved: usize,
    pub orbit_at_most: usize,
    pub null: usize,
}

impl CensusCharge {
    /// A family's orbit and null coordinates, charged against `parameters` stored
    /// coordinates (reads shared with another family are charged there).
    pub fn of(family: &GaugeFamily, parameters: usize) -> Self {
        Self {
            parameters,
            orbit_resolved: family.orbit_dimension.resolved,
            orbit_at_most: family.orbit_dimension.at_most,
            null: family.null_coordinates,
        }
    }

    pub fn add(&mut self, other: CensusCharge) {
        self.parameters += other.parameters;
        self.orbit_resolved += other.orbit_resolved;
        self.orbit_at_most += other.orbit_at_most;
        self.null += other.null;
    }

    /// `parameters − orbit_resolved − null`: at most this many real coordinates carry
    /// the function.
    pub fn real_coordinates_at_most(&self) -> usize {
        self.parameters - self.orbit_resolved - self.null
    }

    /// The share of stored coordinates that is certified convention.
    pub fn convention_fraction(&self) -> f64 {
        (self.orbit_resolved + self.null) as f64 / self.parameters as f64
    }
}

/// The stored tensors of one sequential pre-norm decoder layer.
#[derive(Clone, Copy, Debug)]
pub struct DecoderLayerTensors<'a> {
    pub query: ArrayView2<'a, f64>,
    pub key: ArrayView2<'a, f64>,
    pub value: ArrayView2<'a, f64>,
    pub output: ArrayView2<'a, f64>,
    /// Qwen3's per-head query and key norm gains, when the layer has them.
    pub query_key_norm: Option<(ArrayView1<'a, f64>, ArrayView1<'a, f64>)>,
    pub gate: ArrayView2<'a, f64>,
    pub up: ArrayView2<'a, f64>,
    pub down: ArrayView2<'a, f64>,
    pub input_norm: ArrayView1<'a, f64>,
    pub post_norm: ArrayView1<'a, f64>,
}

/// The families of one decoder layer, each with the stored coordinates it is charged.
#[derive(Clone, Debug)]
pub struct DecoderLayerCensus {
    /// One `GL(hd)` pass-through per key/value group.
    pub ov: Vec<GaugeFamily>,
    pub qk: GaugeFamily,
    /// `|W_Q| + |W_K|`, plus the two query/key norm gains when present.
    pub qk_parameters: usize,
    pub input_norm: GaugeFamily,
    pub post_norm: GaugeFamily,
    pub swiglu: GaugeFamily,
    /// `d`, the gains of each residual norm.
    pub norm_gains: usize,
}

impl DecoderLayerCensus {
    /// `(family, charge)` in the order `ov`, `qk`, `input_norm`, `post_norm`, `swiglu`.
    /// Each norm is charged its gains only; its reads are charged to `qk`, `ov` and
    /// `swiglu`.
    pub fn charges(&self) -> Vec<(&'static str, CensusCharge)> {
        let mut ov = CensusCharge::default();
        for family in &self.ov {
            ov.add(CensusCharge::of(family, family.parameter_coordinates));
        }
        vec![
            ("ov", ov),
            ("qk", CensusCharge::of(&self.qk, self.qk_parameters)),
            ("input_norm", CensusCharge::of(&self.input_norm, self.norm_gains)),
            ("post_norm", CensusCharge::of(&self.post_norm, self.norm_gains)),
            ("swiglu", CensusCharge::of(&self.swiglu, self.swiglu.parameter_coordinates)),
        ]
    }

    /// The layer's total charge.
    pub fn charge(&self) -> CensusCharge {
        let mut total = CensusCharge::default();
        for (_, charge) in self.charges() {
            total.add(charge);
        }
        total
    }

    /// Whether every OV group's orbit dimension is resolved exactly.
    pub fn ov_exact(&self) -> bool {
        self.ov.iter().all(|family| family.orbit_dimension.is_exact())
    }
}

/// The census of one decoder layer (module docs).
pub fn decoder_layer_census(
    geometry: AttentionGeometry,
    rotary: &RotaryEmbedding,
    score_scale: f64,
    query_key_norm_epsilon: f64,
    tensors: DecoderLayerTensors<'_>,
) -> Result<DecoderLayerCensus, CensusRefusal> {
    let gauge = |family: &'static str| move |refusal| CensusRefusal::Gauge { family, refusal };
    let (d, hd, n_kv) = (geometry.model_dim, geometry.head_dim, geometry.n_kv_heads);
    let group = geometry.n_heads / n_kv.max(1);
    let mut ov = Vec::with_capacity(n_kv);
    for g in 0..n_kv {
        let read = tensors.value.slice(s![g * hd..(g + 1) * hd, ..]).to_owned();
        let mut write = Array2::<f64>::zeros((group * d, hd));
        for member in 0..group {
            let h = g * group + member;
            write
                .slice_mut(s![member * d..(member + 1) * d, ..])
                .assign(&tensors.output.slice(s![.., h * hd..(h + 1) * hd]));
        }
        ov.push(
            LinearPassthrough::new(read, write)
                .and_then(|block| block.family())
                .map_err(gauge("ov"))?,
        );
    }
    let bias_free = |weight: ArrayView2<'_, f64>| AffineProjection {
        bias: Array1::zeros(weight.nrows()),
        weight: weight.to_owned(),
    };
    let native = NativeAttention::new(
        geometry,
        rotary.clone(),
        score_scale,
        bias_free(tensors.query),
        bias_free(tensors.key),
        bias_free(tensors.value),
        bias_free(tensors.output),
    )
    .map_err(CensusRefusal::Attention)?;
    let native = match tensors.query_key_norm {
        None => native,
        Some((query_gain, key_gain)) => native
            .with_query_key_norm(query_key_norm_epsilon, query_gain.to_owned(), key_gain.to_owned())
            .map_err(CensusRefusal::Attention)?,
    };
    let qk = QueryKeyGauge::new(&native)
        .and_then(|gauge| gauge.family())
        .map_err(gauge("qk"))?;
    drop(native);
    let qk_gains = tensors.query_key_norm.map_or(0, |(query, key)| query.len() + key.len());
    let qk_parameters = tensors.query.len() + tensors.key.len() + qk_gains;
    let input_norm = NormGain::new(
        tensors.input_norm.to_owned(),
        None,
        vec![tensors.query.to_owned(), tensors.key.to_owned(), tensors.value.to_owned()],
    )
    .map(|norm| norm.family())
    .map_err(gauge("input_norm"))?;
    let post_norm = NormGain::new(
        tensors.post_norm.to_owned(),
        None,
        vec![tensors.gate.to_owned(), tensors.up.to_owned()],
    )
    .map(|norm| norm.family())
    .map_err(gauge("post_norm"))?;
    let swiglu = SwigluUnits::new(tensors.gate.to_owned(), tensors.up.to_owned(), tensors.down.to_owned())
        .map(|units| units.family())
        .map_err(gauge("swiglu"))?;
    Ok(DecoderLayerCensus {
        ov,
        qk,
        qk_parameters,
        input_norm,
        post_norm,
        swiglu,
        norm_gains: d,
    })
}

/// The residual stream's gauge under a tied embedding/unembedding (module docs).
#[derive(Clone, Debug, PartialEq)]
pub struct TiedResidualCensus {
    /// The untied stream's group, `O(d)` for RMSNorm reads.
    pub untied: ContinuousGauge,
    /// The sizes of the groups of bitwise-equal final gains, largest first.
    pub equal_gain_groups: Vec<usize>,
    /// `Σ m_v (m_v − 1)/2`, the tied group's dimension.
    pub tied_dimension: usize,
    /// The embedding rank resolved on the supplied rows, and the row count.
    pub embedding_rank: usize,
    pub embedding_rows: usize,
    /// `vocab · d + d` coordinates; the orbit is resolved only when the rows resolve rank `d`.
    pub charge: CensusCharge,
}

/// The tied residual census from the final norm gain, a subset of the embedding rows, and
/// the vocabulary size.
pub fn tied_residual_census(
    final_norm: ArrayView1<'_, f64>,
    embedding_rows: ArrayView2<'_, f64>,
    vocab: usize,
) -> Result<TiedResidualCensus, CensusRefusal> {
    let width = final_norm.len();
    let stream = ResidualStreamGauge::new(width, &[ResidualRead::RmsNorm])
        .map_err(|refusal| CensusRefusal::Gauge { family: "residual", refusal })?;
    if embedding_rows.ncols() != width {
        return Err(CensusRefusal::Gauge {
            family: "residual",
            refusal: GaugeRefusal::DimensionMismatch {
                what: "embedding row width",
                expected: width,
                found: embedding_rows.ncols(),
            },
        });
    }
    let sigma = embedding_rows
        .to_owned()
        .svd(false, false)
        .map_err(|error| CensusRefusal::Decomposition { detail: format!("{error:?}") })?
        .1;
    let sigma_max = sigma.iter().fold(0.0_f64, |largest, &value| largest.max(value));
    let band = factor_singular_band(embedding_rows.nrows(), width, sigma_max);
    let embedding_rank = sigma.iter().filter(|&&value| value > band).count();
    let mut equal: BTreeMap<u64, usize> = BTreeMap::new();
    for &gain in final_norm.iter() {
        *equal.entry(gain.to_bits()).or_default() += 1;
    }
    let mut equal_gain_groups: Vec<usize> = equal.into_values().collect();
    equal_gain_groups.sort_unstable_by(|left, right| right.cmp(left));
    let tied_dimension = equal_gain_groups.iter().map(|&m| m * (m - 1) / 2).sum();
    Ok(TiedResidualCensus {
        untied: stream.continuous(),
        equal_gain_groups,
        tied_dimension,
        embedding_rank,
        embedding_rows: embedding_rows.nrows(),
        charge: CensusCharge {
            parameters: vocab * width + width,
            orbit_resolved: if embedding_rank == width { tied_dimension } else { 0 },
            orbit_at_most: tied_dimension,
            null: 0,
        },
    })
}
