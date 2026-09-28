//! A native attention block and a decoder layer's norms and MLP on the wire, shared by
//! the operations that read a layer's tensors (`gauge_census`, `canonical_layer`).

use std::collections::BTreeMap;

use ndarray::{Array1, ArrayD};
use serde::{Deserialize, Serialize};

use super::{MpdSurfaceError, matrix, vector};
use crate::parameter_decomposition::attention::{
    AffineProjection, AttentionGeometry, NativeAttention, RotaryEmbedding, RotaryPairing,
};

/// [`AttentionGeometry`] on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct GeometryRequest {
    pub model_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl From<GeometryRequest> for AttentionGeometry {
    fn from(geometry: GeometryRequest) -> Self {
        Self {
            model_dim: geometry.model_dim,
            n_heads: geometry.n_heads,
            n_kv_heads: geometry.n_kv_heads,
            head_dim: geometry.head_dim,
        }
    }
}

/// [`RotaryPairing`] on the wire.
#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RotaryPairingRequest {
    HalfSplit,
    Interleaved,
}

/// [`RotaryEmbedding`] on the wire: the source's exported rotary.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RotaryRequest {
    pub pairing: RotaryPairingRequest,
    pub inverse_frequencies: Vec<f64>,
    pub attention_scaling: f64,
}

impl From<&RotaryRequest> for RotaryEmbedding {
    fn from(rotary: &RotaryRequest) -> Self {
        Self {
            pairing: match rotary.pairing {
                RotaryPairingRequest::HalfSplit => RotaryPairing::HalfSplit,
                RotaryPairingRequest::Interleaved => RotaryPairing::Interleaved,
            },
            inverse_frequencies: rotary.inverse_frequencies.clone(),
            attention_scaling: rotary.attention_scaling,
        }
    }
}

/// A projection `W x + b`: ids of its weight and, when the source has one, its bias.
/// A null bias is the bias-free projection (a zero bias).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectionRequest {
    pub weight: String,
    pub bias: Option<String>,
}

/// Qwen3's per-head query/key norm.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QueryKeyNormRequest {
    pub epsilon: f64,
    pub query_gain: String,
    pub key_gain: String,
}

/// A native rotary attention block (`attention::NativeAttention`).
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AttentionRequest {
    pub geometry: GeometryRequest,
    pub rotary: RotaryRequest,
    /// The multiplier on each query-key inner product.
    pub score_scale: f64,
    pub query: ProjectionRequest,
    pub key: ProjectionRequest,
    pub value: ProjectionRequest,
    pub output: ProjectionRequest,
    pub query_key_norm: Option<QueryKeyNormRequest>,
}

/// An RMSNorm `w ⊙ h (mean h² + ε)^{-1/2}`.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RmsNormRequest {
    pub epsilon: f64,
    pub gain: String,
}

fn projection(
    tensors: &BTreeMap<String, ArrayD<f64>>,
    request: &ProjectionRequest,
) -> Result<AffineProjection, MpdSurfaceError> {
    let weight = matrix(tensors, &request.weight)?.to_owned();
    let bias = match &request.bias {
        Some(id) => vector(tensors, id)?.to_owned(),
        None => Array1::zeros(weight.nrows()),
    };
    Ok(AffineProjection { weight, bias })
}

/// The owner's native attention block from its wire tensors; the owner validates every
/// shape.
pub(super) fn native_attention(
    tensors: &BTreeMap<String, ArrayD<f64>>,
    request: &AttentionRequest,
) -> Result<NativeAttention, MpdSurfaceError> {
    let native = NativeAttention::new(
        request.geometry.into(),
        (&request.rotary).into(),
        request.score_scale,
        projection(tensors, &request.query)?,
        projection(tensors, &request.key)?,
        projection(tensors, &request.value)?,
        projection(tensors, &request.output)?,
    )
    .map_err(MpdSurfaceError::Attention)?;
    match &request.query_key_norm {
        None => Ok(native),
        Some(norm) => native
            .with_query_key_norm(
                norm.epsilon,
                vector(tensors, &norm.query_gain)?.to_owned(),
                vector(tensors, &norm.key_gain)?.to_owned(),
            )
            .map_err(MpdSurfaceError::Attention),
    }
}
