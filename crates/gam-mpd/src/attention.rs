//! Exact execution of the source's rotary causal self-attention (#2951).
//!
//! With the source's rotary embedding `R_p` at absolute position `p`, the score of
//! query token `t` on key token `s` in head `h` is
//!
//! ```text
//! S_ts = σ (R_{p_t} q_{t,h})ᵀ (R_{p_s} k_{s,g(h)}),
//! ```
//!
//! `σ` the source's score multiplier (its `scaling`) and `g(h)` the key/value head
//! the source's `repeat_kv` gives query head `h`. The scores go through the source's
//! causal mask and one joint softmax per query row, and the weights read the source's
//! value projection and output projection. A mask on a projection is a mask on its
//! summed read ([`super::block`]): the softmax sees the masked rows, never a sum of
//! separately normalized per-component patterns (`softmax(Σ_c ℓ_c) ≠ Σ_c softmax(ℓ_c)`).
//!
//! # The source's rotary convention
//!
//! The source rotates plane `(a, b)` of a head by `+p θ_p`:
//! `(x_a, x_b) ↦ (x_a cos − x_b sin, x_b cos + x_a sin)`, with `cos` and `sin`
//! both multiplied by the source's `attention_scaling`. Two pairings exist in
//! transformers:
//! - [`RotaryPairing::HalfSplit`]: `rotate_half` (`models/llama`, `models/qwen3`,
//!   `models/gpt_neox`), planes `(p, p + P)` over the first `2P` head coordinates;
//! - [`RotaryPairing::Interleaved`]: `rotate_every_two` (`models/gptj`), planes
//!   `(2p, 2p + 1)`.
//!
//! Head coordinates past the rotary dimension pass through unrotated and unscaled
//! (gpt_neox `rotary_ndims`). The frequencies are the source's `inv_freq` buffer
//! as exported, so every rope-scaling rule the source applied is already in them.
//!
//! # Query/key norm
//!
//! A per-head norm between the projection and the rotary embedding (Qwen3's
//! `q_norm`/`k_norm`, [`NativeAttention::with_query_key_norm`]) is a native
//! nonlinearity. It runs on each token's summed head rows through gam-sae's
//! gated-rewrite owner.
//!
//! # Roundoff
//!
//! Every result carries a forward-error radius against the exact value of its own
//! program for inputs anywhere within their own radii. It collects:
//! - Higham's `γ_k · Σ|monomials|` over the rounded operations at the computed
//!   inputs (`gam_linalg::roundoff`);
//! - each input's own radius, through every product of two inexact factors as the
//!   box bound `|â| r_b + r_a |b̂| + r_a r_b`, whose last term a first-order
//!   propagation drops;
//! - the trigonometric input error `u·|φ| + ulp` of each angle, as one of those
//!   inexact factors;
//! - the log-softmax evaluation radius from `gam_math::categorical`, which holds
//!   over the whole logit box;
//! - through the per-head query/key RMS norm
//!   ([`NativeAttention::with_query_key_norm`]), a mean value bound. It takes the
//!   normalizer and its slope at the smallest mean square the input box admits.
//!
//! Two programs with the same exact value agree within the sum of their radii.
//!
//! The libm terms rest on the platform's documented accuracy. glibc x86_64's "Known
//! Maximum Errors" puts `sin`, `cos` and `exp` within one ulp, which covers the MSI
//! build targets. The tests measure that bound against a double-double reference at
//! every point the fixtures evaluate. That checks the platform the tests run on; it
//! does not certify production inputs.
//!
//! # Memory
//!
//! The `heads × tokens × tokens` scores, weights and their radii are reserved on
//! gam-runtime's process memory governor before they are allocated. Each result
//! holds its reservation for as long as its arrays live, so none is `Clone`. A
//! footprint beyond the budget is a typed refusal, [`AttentionProgramError::Memory`].

use std::fmt;

use gam_linalg::roundoff::accumulation_growth;
use gam_math::categorical::{CategoricalError, log_softmax_with_error};
use gam_runtime::resource::{MemoryGovernor, MemoryReservation, MemoryReservationError};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis, CowArray, Ix2, ShapeBuilder};
use serde::{Deserialize, Serialize};

use crate::block::{down, rms_norm_band, up};
use crate::gated_rewrite::{GatedRewriteError, MaskedNorm};

/// Largest position magnitude whose differences stay exact in `f64`:
/// `|p_s − p_t| ≤ 2^53` is representable, so `Δ as f64` rounds nothing.
const EXACT_POSITION_LIMIT: i64 = 1 << 52;

#[derive(Clone, Debug, PartialEq)]
pub enum AttentionProgramError {
    /// A tensor, input or mask whose shape does not match the declared geometry.
    Shape {
        tensor: &'static str,
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// A `heads × tokens × tokens` score or weight array whose shape does not
    /// match the geometry's head count and the token count.
    HeadShape {
        tensor: &'static str,
        expected: (usize, usize, usize),
        found: (usize, usize, usize),
    },
    /// Query heads share key/value heads in contiguous groups, so `n_kv_heads`
    /// must divide `n_heads`.
    KeyValueHeadsDoNotDivide { n_heads: usize, n_kv_heads: usize },
    /// The rotary planes need more coordinates than one head has.
    RotaryExceedsHead { rotary_dim: usize, head_dim: usize },
    /// A position whose differences are not exact in `f64`.
    PositionNotExact { position: i64 },
    /// A score row that names no categorical distribution.
    Categorical(CategoricalError),
    /// The source's per-head query/key norm refused its rows.
    Norm(GatedRewriteError),
    /// The `heads × tokens × tokens` scores and weights do not fit the process
    /// memory budget. Nothing of that size was allocated.
    Memory(MemoryReservationError),
}

impl fmt::Display for AttentionProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape {
                tensor,
                expected,
                found,
            } => write!(f, "{tensor} has shape {found:?}, the geometry needs {expected:?}"),
            Self::HeadShape {
                tensor,
                expected,
                found,
            } => write!(f, "{tensor} has shape {found:?}, the heads and tokens need {expected:?}"),
            Self::KeyValueHeadsDoNotDivide {
                n_heads,
                n_kv_heads,
            } => write!(f, "{n_kv_heads} key/value heads do not divide {n_heads} query heads"),
            Self::RotaryExceedsHead {
                rotary_dim,
                head_dim,
            } => write!(f, "rotary dimension {rotary_dim} exceeds head dimension {head_dim}"),
            Self::PositionNotExact { position } => write!(
                f,
                "position {position} exceeds 2^52, so position differences are not exact in f64"
            ),
            Self::Categorical(error) => write!(f, "attention score row: {error}"),
            Self::Norm(error) => write!(f, "attention query/key norm: {error}"),
            Self::Memory(error) => write!(f, "attention scores and weights: {error}"),
        }
    }
}

impl std::error::Error for AttentionProgramError {}

/// The source attention block's dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttentionGeometry {
    pub model_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl AttentionGeometry {
    pub fn query_dim(&self) -> usize {
        self.n_heads * self.head_dim
    }

    pub fn key_value_dim(&self) -> usize {
        self.n_kv_heads * self.head_dim
    }

    /// The key/value head the source's `repeat_kv` gives query head `head`.
    pub fn key_value_head(&self, head: usize) -> usize {
        head / (self.n_heads / self.n_kv_heads)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RotaryPairing {
    HalfSplit,
    Interleaved,
}

/// The source's rotary embedding, as exported from the source module.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RotaryEmbedding {
    pub pairing: RotaryPairing,
    /// The source's `inv_freq` buffer, one frequency per rotated plane.
    pub inverse_frequencies: Vec<f64>,
    /// The source's `attention_scaling`, multiplying both `cos` and `sin`.
    pub attention_scaling: f64,
}

impl RotaryEmbedding {
    pub fn rotary_dim(&self) -> usize {
        2 * self.inverse_frequencies.len()
    }

    /// Head coordinates `(a, b)` of rotated plane `plane`.
    pub fn plane(&self, plane: usize) -> (usize, usize) {
        match self.pairing {
            RotaryPairing::HalfSplit => (plane, plane + self.inverse_frequencies.len()),
            RotaryPairing::Interleaved => (2 * plane, 2 * plane + 1),
        }
    }
}

/// `cos φ`, `sin φ` at `φ = multiplier · θ`, with the error bound `η` on each:
/// the product rounds by at most `γ_1 |φ̂|` and libm adds at most one ulp, which
/// is at most `ε |value|`.
fn plane_trig(multiplier: f64, frequency: f64) -> (f64, f64, f64) {
    let angle = multiplier * frequency;
    let (sin, cos) = angle.sin_cos();
    let eta = accumulation_growth(1) * angle.abs() + f64::EPSILON * cos.abs().max(sin.abs());
    (cos, sin, eta)
}

/// The largest `|a b − â b̂|` over `|a − â| ≤ r_a` and `|b − b̂| ≤ r_b`:
/// `|â| r_b + r_a |b̂| + r_a r_b`. Every term is nonnegative, so the sum rounds
/// without cancellation.
fn box_product(a: f64, radius_a: f64, b: f64, radius_b: f64) -> f64 {
    a.abs() * radius_b + radius_a * b.abs() + radius_a * radius_b
}

/// One sequential inner product and the absolute sum of its terms.
fn inner_with_abs(row: ArrayView1<f64>, x: ArrayView1<f64>) -> (f64, f64) {
    row.iter()
        .zip(x.iter())
        .fold((0.0, 0.0), |(value, abs), (w, v)| (value + w * v, abs + (w * v).abs()))
}

/// The source's joint softmax over one causally admissible query row, with a
/// radius on each weight.
///
/// The log weights and their evaluation radii `e_s` come from gam-math's
/// categorical owner, [`log_softmax_with_error`]. A logit within `r_s` of exact
/// moves `ℓ_s` by `r_s` and `lse ℓ` by at most `max_t r_t`, so each log weight is
/// within `b_s = r_s + max_t r_t + e_s` of exact. `exp` is monotone, so the exact
/// weight lies in `[exp(ℓ̂_s − b_s), exp(ℓ̂_s + b_s)]`. Each endpoint's argument is
/// rounded outward and libm's `exp` is within one ulp, so one ulp step outward on each
/// endpoint encloses it. The radius is the farther endpoint's distance from `ŵ_s`,
/// rounded up. The ulp is absolute, not relative, and that matters: in the subnormal
/// range it is `2^-1074`, and a weight that underflowed to `ŵ_s = 0` still has a
/// positive exact weight up to `exp(ℓ̂_s + b_s)`. A relative bound `ŵ_s (expm1(b_s) + ε)`
/// would give that weight a radius of zero. For a normal weight in a narrow box, the
/// endpoints agree with that relative bound to first order.
///
/// Every exact weight lies in `[0, 1]` whatever the logits, so it is also within
/// `max(ŵ_s, 1 − ŵ_s)` of the computed one, and the radius is the smaller bound. A
/// narrow box keeps the endpoint bound. A wide one (a box of component masks) makes the
/// upper endpoint overflow to `∞`, and the range bound keeps the radius finite.
fn attention_weights(logits: &[f64], logit_radius: &[f64]) -> Result<(Vec<f64>, Vec<f64>), CategoricalError> {
    let (log_weights, evaluation_radius) = log_softmax_with_error(logits)?;
    let widest = logit_radius.iter().copied().fold(0.0, f64::max);
    let mut weights = Vec::with_capacity(logits.len());
    let mut radius = Vec::with_capacity(logits.len());
    for ((&log_weight, &evaluation), &own) in log_weights.iter().zip(&evaluation_radius).zip(logit_radius) {
        let weight = log_weight.exp();
        weights.push(weight);
        let band = ((own + widest).next_up() + evaluation).next_up();
        let highest = (log_weight + band).next_up().exp().next_up();
        let lowest = (log_weight - band).next_down().exp().next_down().max(0.0);
        let reach = (highest - weight).next_up().max((weight - lowest).next_up());
        let range = (1.0 - weight).next_up().max(weight);
        radius.push(reach.min(range));
    }
    Ok((weights, radius))
}

/// A source projection `W x + b`, as the source module stores it. A bias-free
/// source carries a zero bias, and adding `0.0` is exact.
#[derive(Clone, Debug)]
pub struct AffineProjection {
    pub weight: Array2<f64>,
    pub bias: Array1<f64>,
}

/// The source's causal rotary attention without its projections: the geometry,
/// the rotary embedding and the score multiplier, validated once at construction.
/// It holds no tensor, so a program node can carry it by serializing its three
/// inputs.
#[derive(Clone, Debug)]
pub struct RotaryCausalAttention {
    geometry: AttentionGeometry,
    rotary: RotaryEmbedding,
    score_scale: f64,
}

/// The source's per-head query/key RMS norm: its declared epsilon and the
/// `head_dim` gains of `q_norm` and `k_norm`.
#[derive(Clone, Debug)]
pub struct QueryKeyNorm {
    epsilon: f64,
    query_gain: Array1<f64>,
    key_gain: Array1<f64>,
}

impl QueryKeyNorm {
    /// The source's declared `ε` in `(mean(h²) + ε)^{-1/2}`.
    pub fn epsilon(&self) -> f64 {
        self.epsilon
    }

    /// The `head_dim` gains of `q_norm`, shared by every query head.
    pub fn query_gain(&self) -> ArrayView1<'_, f64> {
        self.query_gain.view()
    }

    /// The `head_dim` gains of `k_norm`, shared by every key head.
    pub fn key_gain(&self) -> ArrayView1<'_, f64> {
        self.key_gain.view()
    }
}

/// The source's causal self-attention block on its original tensors.
#[derive(Clone, Debug)]
pub struct NativeAttention {
    attention: RotaryCausalAttention,
    query: AffineProjection,
    key: AffineProjection,
    value: AffineProjection,
    output: AffineProjection,
    query_key_norm: Option<QueryKeyNorm>,
}

fn expect_shape(
    tensor: &'static str,
    found: (usize, usize),
    expected: (usize, usize),
) -> Result<(), AttentionProgramError> {
    if found == expected {
        Ok(())
    } else {
        Err(AttentionProgramError::Shape {
            tensor,
            expected,
            found,
        })
    }
}

fn expect_heads(
    tensor: &'static str,
    found: (usize, usize, usize),
    expected: (usize, usize, usize),
) -> Result<(), AttentionProgramError> {
    if found == expected {
        Ok(())
    } else {
        Err(AttentionProgramError::HeadShape {
            tensor,
            expected,
            found,
        })
    }
}

/// Reserve `copies` `heads × tokens × tokens` `f64` arrays on `governor`
/// before any of them is allocated. The caller holds the reservation beside the
/// arrays for as long as they live. A shape whose cell count overflows `usize`
/// is refused by the governor as a size overflow.
fn reserve_heads(
    governor: &MemoryGovernor,
    shape: (usize, usize, usize),
    copies: usize,
) -> Result<MemoryReservation, AttentionProgramError> {
    let (heads, tokens, keys) = shape;
    governor
        .try_reserve_dense_f64_copies(heads.saturating_mul(tokens), keys, copies, "attention scores and weights")
        .map_err(AttentionProgramError::Memory)
}

/// A pre-softmax score matrix per head with its radius; causally masked entries
/// are `−∞` with radius zero.
struct ScoredHeads {
    scores: Array3<f64>,
    radius: Array3<f64>,
}

/// One program's executed attention block, each quantity with its forward-error
/// radius against the program's exact value. It holds the memory reservation of
/// its four `heads × tokens × tokens` arrays for as long as they live, so it is
/// not `Clone`: a copy would be memory the governor never admitted.
#[derive(Debug)]
pub struct AttentionExecution {
    /// `heads × tokens × tokens` scores; `s > t` is causally masked to `−∞`.
    pub scores: Array3<f64>,
    pub score_radius: Array3<f64>,
    pub weights: Array3<f64>,
    pub weight_radius: Array3<f64>,
    /// `tokens × model_dim`, after the source's output projection.
    pub output: Array2<f64>,
    pub output_radius: Array2<f64>,
    footprint: MemoryReservation,
}

impl AttentionExecution {
    /// Bytes this execution holds on the process memory governor.
    pub fn reserved_bytes(&self) -> usize {
        self.footprint.bytes()
    }
}

/// Already-projected rows, `tokens × width`, with a per-entry forward-error
/// radius against their exact values.
#[derive(Clone, Copy, Debug)]
pub struct ProjectedRows<'a> {
    pub values: ArrayView2<'a, f64>,
    pub radius: ArrayView2<'a, f64>,
}

/// The one zero that exact rows view as their radius.
static EXACT_RADIUS: [f64; 1] = [0.0];

impl<'a> ProjectedRows<'a> {
    /// Rows taken as exact. The radius is a zero-stride view of one zero, so no
    /// `tokens × width` matrix is allocated.
    pub fn exact(values: ArrayView2<'a, f64>) -> Self {
        let radius = ArrayView2::from_shape(values.dim().strides((0, 0)), &EXACT_RADIUS[..])
            .expect("a zero-stride view reads only its one entry");
        Self { values, radius }
    }
}

/// The attention block before the output projection, each quantity with its
/// forward-error radius. Like [`AttentionExecution`], it holds the reservation of
/// its four `heads × tokens × tokens` arrays for as long as they live.
#[derive(Debug)]
pub struct ProjectedAttention {
    /// `heads × tokens × tokens` scores; `s > t` is causally masked to `−∞`.
    pub scores: Array3<f64>,
    pub score_radius: Array3<f64>,
    pub weights: Array3<f64>,
    pub weight_radius: Array3<f64>,
    /// `tokens × n_heads·head_dim`: each head's weighted value read, before the
    /// output projection.
    pub mixed: Array2<f64>,
    pub mixed_radius: Array2<f64>,
    footprint: MemoryReservation,
}

impl ProjectedAttention {
    /// Bytes this result holds on the process memory governor.
    pub fn reserved_bytes(&self) -> usize {
        self.footprint.bytes()
    }
}

/// The causal attention weights at given scores, each with its radius, holding
/// the reservation of both `heads × tokens × tokens` arrays for as long as they
/// live.
#[derive(Debug)]
pub struct AttentionWeights {
    pub weights: Array3<f64>,
    pub weight_radius: Array3<f64>,
    footprint: MemoryReservation,
}

impl AttentionWeights {
    /// Bytes these weights hold on the process memory governor.
    pub fn reserved_bytes(&self) -> usize {
        self.footprint.bytes()
    }
}

/// Per-token head-space rows with a per-entry forward-error radius.
struct HeadRows {
    value: Array2<f64>,
    radius: Array2<f64>,
}

/// `W x_t + b` for every token. The `model_dim`-term inner product and the bias
/// addition give the radius `γ_{d+1}` times the absolute sum of the terms.
fn project_affine(projection: &AffineProjection, x: ArrayView2<f64>) -> (Array2<f64>, Array2<f64>) {
    let growth = accumulation_growth(x.ncols() + 1);
    let mut value = Array2::zeros((x.nrows(), projection.weight.nrows()));
    let mut radius = Array2::zeros((x.nrows(), projection.weight.nrows()));
    for (t, token) in x.outer_iter().enumerate() {
        for (row, w) in projection.weight.outer_iter().enumerate() {
            let (inner, abs) = inner_with_abs(w, token);
            let bias = projection.bias[row];
            value[[t, row]] = inner + bias;
            radius[[t, row]] = growth * (abs + bias.abs());
        }
    }
    (value, radius)
}

/// `tokens × width` rows as `tokens·heads × head_dim` head rows, one per contiguous
/// `head_dim` block. A width that is not a whole, nonzero number of heads is refused.
/// A contiguous view is reshaped in place; any other view (e.g. a zero-stride exact radius) is copied.
fn head_rows<'a>(rows: &'a ArrayView2<'_, f64>, head_dim: usize) -> Result<CowArray<'a, f64, Ix2>, GatedRewriteError> {
    let (tokens, width) = rows.dim();
    let heads = width.checked_div(head_dim).unwrap_or(0);
    let mismatch = GatedRewriteError::ShapeMismatch {
        what: "per-head query/key rows",
        expected: (tokens * heads, head_dim),
        found: (tokens, width),
    };
    if heads == 0 || heads * head_dim != width {
        return Err(mismatch);
    }
    rows.to_shape((tokens * heads, head_dim)).map_err(|_| mismatch)
}

/// The source's per-head RMS norm (Qwen3 `q_norm`, `k_norm`) of `tokens × heads·head_dim`
/// rows: `w ⊙ x (mean x² + ε)^{-1/2}` on each contiguous `head_dim` block of a row, with one
/// gain `w` of length `head_dim` shared by every head, evaluated by the gated-rewrite owner.
/// [`NativeAttention::with_query_key_norm`] and the mechanism program's `HeadRmsNorm` node
/// both run it, so their rows agree bit for bit.
pub fn head_rms_norm(
    rows: ArrayView2<'_, f64>,
    head_dim: usize,
    epsilon: f64,
    gain: ArrayView1<'_, f64>,
) -> Result<Array2<f64>, GatedRewriteError> {
    let (tokens, width) = rows.dim();
    let per_head = head_rows(&rows, head_dim)?;
    let normalized = MaskedNorm::Rms { epsilon, gain }.apply(per_head.view())?;
    normalized
        .into_shape_with_order((tokens, width))
        .map_err(|_| GatedRewriteError::ShapeMismatch {
            what: "per-head query/key rows",
            expected: (tokens, width),
            found: (tokens * width / head_dim, head_dim),
        })
}

/// [`head_rms_norm`] of rows given with a per-entry radius, and the radius of the result.
///
/// For one head row `x̂` with radius `r`, `y(x) = w ⊙ x ν(x)` with
/// `ν = s^{-1/2}`, `s(x) = mean x² + ε`. The radius bounds `|y_c(x) − fl(y_c(x̂))|` for
/// every `x` in the box by the smaller of two bounds.
///
/// The derived bound is `|y_c(x) − y_c(x̂)| + |y_c(x̂) − fl(y_c(x̂))|`:
/// - The second term is the head row's [`rms_norm_band`]. That is the band of this
///   program, and it carries binary64's underflow.
/// - The first term uses the identity
///   `y_c(x) − y_c(x̂) = w_c [(x_c − x̂_c) ν(x) + x̂_c (ν(x) − ν(x̂))]`.
///   - Each square's box bound gives `|s(x) − s(x̂)| ≤ Δs = mean(r_d (2|x̂_d| + r_d))`,
///     so `s(x) ≥ s_lo = max(s(x̂) − Δs, ε)`.
///   - `ν` and `|ν'| = ½ s^{-3/2}` both decrease in `s`. So `ν(x) ≤ s_lo^{-1/2}`, and by
///     the mean value inequality `|ν(x) − ν(x̂)| ≤ ½ s_lo^{-3/2} Δs`.
///   - Together the term is at most `|w_c| (r_c s_lo^{-1/2} + |x̂_c| ½ s_lo^{-3/2} Δs)`.
///
/// The range bound is `|w_c| √d + |fl(y_c(x̂))|`. Every `x` has `x_c² ≤ d · mean x²`, so
/// `|y_c(x)| ≤ |w_c| √d`, and this bound holds on any box. It alone is the radius where:
/// - `s_lo = 0`, a box that reaches `x = 0` at `ε = 0`;
/// - [`rms_norm_band`] refuses the head row;
/// - the derived bound overflows.
///
/// Every operation rounds to nearest and then steps one float outward, down for `s_lo` and
/// up elsewhere. So `s_lo` is a lower bound and the radius an upper bound, both computed in
/// binary64. A square that rounds into the subnormal range moves by at most an absolute
/// `2^-1075`, and the one-float step covers that too.
pub fn head_rms_norm_with_radius(
    rows: ProjectedRows<'_>,
    head_dim: usize,
    epsilon: f64,
    gain: ArrayView1<'_, f64>,
) -> Result<(Array2<f64>, Array2<f64>), GatedRewriteError> {
    let (tokens, width) = rows.values.dim();
    if rows.radius.dim() != (tokens, width) {
        return Err(GatedRewriteError::ShapeMismatch {
            what: "per-head query/key radius",
            expected: (tokens, width),
            found: rows.radius.dim(),
        });
    }
    let values = head_rms_norm(rows.values, head_dim, epsilon, gain)?;
    let per_head = head_rows(&rows.values, head_dim)?;
    let per_head_radius = head_rows(&rows.radius, head_dim)?;
    let head_width = head_dim as f64;
    let root_width = up(head_width.sqrt());
    let values_view = values.view();
    let normalized = head_rows(&values_view, head_dim)?;
    let mut normalized_radius = Array2::zeros(normalized.dim());
    for (row, ((x, r), y)) in per_head
        .rows()
        .into_iter()
        .zip(per_head_radius.rows())
        .zip(normalized.rows())
        .enumerate()
    {
        let squares = x.iter().fold(0.0, |sum, &value| down(sum + down(value * value)));
        let spread = up(x
            .iter()
            .zip(r.iter())
            .fold(0.0, |sum, (&value, &radius)| up(sum + up(radius * up(2.0 * value.abs() + radius))))
            / head_width);
        let floor = down(down(down(squares / head_width) + epsilon) - spread).max(epsilon);
        let band = if floor > 0.0 {
            rms_norm_band(epsilon, gain, x.insert_axis(Axis(0)), y.insert_axis(Axis(0))).ok()
        } else {
            None
        };
        let inverse_root = up(down(floor.sqrt()).recip());
        let slope_spread = up(0.5 * up(inverse_root * up(inverse_root * up(inverse_root * spread))));
        for c in 0..head_dim {
            let weight = gain[c].abs();
            let range = up(up(weight * root_width) + y[c].abs());
            let derived = band.as_ref().map(|band| {
                up(up(weight * up(up(r[c] * inverse_root) + up(x[c].abs() * slope_spread))) + band[[0, c]])
            });
            // A derived bound that overflowed, or is NaN from `0 · ∞`, is not below the range bound.
            normalized_radius[[row, c]] = match derived {
                Some(bound) if bound < range => bound,
                _ => range,
            };
        }
    }
    let radius = normalized_radius
        .into_shape_with_order((tokens, width))
        .map_err(|_| GatedRewriteError::ShapeMismatch {
            what: "per-head query/key radius",
            expected: (tokens, width),
            found: (tokens * width / head_dim, head_dim),
        })?;
    Ok((values, radius))
}

impl RotaryCausalAttention {
    /// Query heads share key/value heads in contiguous groups, and the rotated
    /// planes must fit in one head. Only these checks run; nothing is allocated.
    pub fn new(
        geometry: AttentionGeometry,
        rotary: RotaryEmbedding,
        score_scale: f64,
    ) -> Result<Self, AttentionProgramError> {
        if geometry.n_kv_heads == 0 || !geometry.n_heads.is_multiple_of(geometry.n_kv_heads) {
            return Err(AttentionProgramError::KeyValueHeadsDoNotDivide {
                n_heads: geometry.n_heads,
                n_kv_heads: geometry.n_kv_heads,
            });
        }
        if rotary.rotary_dim() > geometry.head_dim {
            return Err(AttentionProgramError::RotaryExceedsHead {
                rotary_dim: rotary.rotary_dim(),
                head_dim: geometry.head_dim,
            });
        }
        Ok(Self {
            geometry,
            rotary,
            score_scale,
        })
    }

    fn check_positions(&self, positions: &[i64]) -> Result<(), AttentionProgramError> {
        match positions
            .iter()
            .find(|p| p.unsigned_abs() > EXACT_POSITION_LIMIT as u64)
        {
            Some(&position) => Err(AttentionProgramError::PositionNotExact { position }),
            None => Ok(()),
        }
    }

    /// The source's attention on already-projected rows: rotate queries and keys
    /// at their absolute positions, score, mask causally, softmax jointly and read
    /// the values. Queries are `tokens × n_heads·head_dim`; keys and values are
    /// `tokens × n_kv_heads·head_dim`. Each input radius enters as a box bound (see the module docs).
    pub fn attend_projected(
        &self,
        governor: &MemoryGovernor,
        queries: ProjectedRows<'_>,
        keys: ProjectedRows<'_>,
        values: ProjectedRows<'_>,
        positions: &[i64],
    ) -> Result<ProjectedAttention, AttentionProgramError> {
        self.check_positions(positions)?;
        let g = self.geometry;
        let (heads, tokens, hd) = (g.n_heads, positions.len(), g.head_dim);
        // Dimensions only: an array of `&ProjectedRows` would tie the three
        // arguments' lifetimes together, and `ArrayView` is invariant in its lifetime.
        for (name, values_dim, radius_dim, width) in [
            ("query rows", queries.values.dim(), queries.radius.dim(), g.query_dim()),
            ("key rows", keys.values.dim(), keys.radius.dim(), g.key_value_dim()),
            ("value rows", values.values.dim(), values.radius.dim(), g.key_value_dim()),
        ] {
            expect_shape(name, values_dim, (tokens, width))?;
            expect_shape(name, radius_dim, (tokens, width))?;
        }
        // Scores, score radius, weights and weight radius, before anything is allocated.
        let footprint = reserve_heads(governor, (heads, tokens, tokens), 4)?;
        let queries = self.rotate(queries, positions, heads);
        let keys = self.rotate(keys, positions, g.n_kv_heads);
        // The `head_dim` products and their additions, then σ.
        let growth = accumulation_growth(hd + 1);
        let mut scores = Array3::from_elem((heads, tokens, tokens), f64::NEG_INFINITY);
        let mut radius = Array3::zeros((heads, tokens, tokens));
        for head in 0..heads {
            let (qo, ko) = (head * hd, g.key_value_head(head) * hd);
            for t in 0..tokens {
                for s in 0..=t {
                    let (mut value, mut abs, mut propagated) = (0.0, 0.0, 0.0);
                    for c in 0..hd {
                        let (q, k) = ((t, qo + c), (s, ko + c));
                        value += queries.value[q] * keys.value[k];
                        abs += (queries.value[q] * keys.value[k]).abs();
                        propagated += box_product(queries.value[q], queries.radius[q], keys.value[k], keys.radius[k]);
                    }
                    scores[[head, t, s]] = self.score_scale * value;
                    radius[[head, t, s]] = self.score_scale.abs() * (propagated + growth * abs);
                }
            }
        }
        self.mix(ScoredHeads { scores, radius }, values, footprint)
    }

    /// The source's rotation of every head of `rows` at each token's absolute
    /// position. The radius carries the rows' own radius and the trigonometric
    /// input error `|α| η` of `α·cos` and `α·sin` as the box bound of each product,
    /// and the rotation's rounding (`α·cos`, the product and the in-plane sum).
    fn rotate(&self, rows: ProjectedRows<'_>, positions: &[i64], heads: usize) -> HeadRows {
        let alpha = self.rotary.attention_scaling;
        let growth = accumulation_growth(3);
        let mut value = rows.values.to_owned();
        let mut radius = rows.radius.to_owned();
        for (t, &position) in positions.iter().enumerate() {
            for (plane, &frequency) in self.rotary.inverse_frequencies.iter().enumerate() {
                let (cos, sin, eta) = plane_trig(position as f64, frequency);
                let (cos, sin) = (alpha * cos, alpha * sin);
                let (a, b) = self.rotary.plane(plane);
                for head in 0..heads {
                    let offset = head * self.geometry.head_dim;
                    let (a, b) = ((t, offset + a), (t, offset + b));
                    let (xa, xb, ra, rb) = (value[a], value[b], radius[a], radius[b]);
                    value[a] = xa * cos - xb * sin;
                    value[b] = xb * cos + xa * sin;
                    let trig = alpha.abs() * eta;
                    radius[a] = box_product(xa, ra, cos, trig)
                        + box_product(xb, rb, sin, trig)
                        + growth * (xa.abs() * cos.abs() + xb.abs() * sin.abs());
                    radius[b] = box_product(xb, rb, cos, trig)
                        + box_product(xa, ra, sin, trig)
                        + growth * (xb.abs() * cos.abs() + xa.abs() * sin.abs());
                }
            }
        }
        HeadRows { value, radius }
    }

    /// The source's rotation of every head of `rows` (`tokens × heads·head_dim`) at each
    /// token's absolute position, with the radius the attention carries for it: the
    /// rows' own radius, the trigonometric input error and the rotation's rounding. The
    /// rotation is linear, so a change of rows rotates as rows do.
    pub fn rotate_heads(
        &self,
        rows: ProjectedRows<'_>,
        positions: &[i64],
        heads: usize,
    ) -> Result<(Array2<f64>, Array2<f64>), AttentionProgramError> {
        self.check_positions(positions)?;
        let shape = (positions.len(), heads * self.geometry.head_dim);
        expect_shape("rotated rows", rows.values.dim(), shape)?;
        expect_shape("rotated rows", rows.radius.dim(), shape)?;
        let rotated = self.rotate(rows, positions, heads);
        Ok((rotated.value, rotated.radius))
    }

    /// The source's causal mask and joint softmax at given scores: for each head
    /// and query token `t`, one categorical distribution over the keys `s ≤ t`,
    /// with the radius [`RotaryCausalAttention::attend_projected`] carries. Scores
    /// and their radii are `n_heads × tokens × tokens`. Entries with `s > t` are
    /// never read; their weights are zero with radius zero. Both weight arrays are
    /// reserved on the process memory governor before they are allocated.
    pub fn weights_at_scores(
        &self,
        governor: &MemoryGovernor,
        scores: ArrayView3<'_, f64>,
        score_radius: ArrayView3<'_, f64>,
    ) -> Result<AttentionWeights, AttentionProgramError> {
        let tokens = scores.dim().1;
        let shape = (self.geometry.n_heads, tokens, tokens);
        expect_heads("scores", scores.dim(), shape)?;
        expect_heads("score radius", score_radius.dim(), shape)?;
        let footprint = reserve_heads(governor, shape, 2)?;
        let (weights, weight_radius) = self.causal_softmax(scores, score_radius)?;
        Ok(AttentionWeights {
            weights,
            weight_radius,
            footprint,
        })
    }

    /// The per-row causal softmax behind [`RotaryCausalAttention::weights_at_scores`],
    /// on `n_heads × tokens × tokens` scores whose shape the caller has checked and
    /// whose weights it has reserved.
    fn causal_softmax(
        &self,
        scores: ArrayView3<'_, f64>,
        score_radius: ArrayView3<'_, f64>,
    ) -> Result<(Array3<f64>, Array3<f64>), AttentionProgramError> {
        let (shape, tokens) = (scores.dim(), scores.dim().1);
        let mut weights = Array3::zeros(shape);
        let mut weight_radius = Array3::zeros(shape);
        for head in 0..self.geometry.n_heads {
            for t in 0..tokens {
                let logits: Vec<f64> = (0..=t).map(|s| scores[[head, t, s]]).collect();
                let logit_radius: Vec<f64> = (0..=t).map(|s| score_radius[[head, t, s]]).collect();
                let (row, row_radius) =
                    attention_weights(&logits, &logit_radius).map_err(AttentionProgramError::Categorical)?;
                for s in 0..=t {
                    weights[[head, t, s]] = row[s];
                    weight_radius[[head, t, s]] = row_radius[s];
                }
            }
        }
        Ok((weights, weight_radius))
    }

    /// Each head's read of the values at given causal weights,
    /// `mixed[t, h] = Σ_{s ≤ t} w_{hts} v_{s, g(h)}`, `tokens × n_heads·head_dim`,
    /// with the radius [`RotaryCausalAttention::attend_projected`] carries:
    /// `Σ_s (r^w_{hts} |v_s| + w_{hts} r^v_s + r^w_{hts} r^v_s) + γ_{t+1} Σ_s |w_{hts} v_s|`,
    /// which bounds the exact read for weights and values anywhere within their radii.
    /// Weights and their radii are `n_heads × tokens × tokens`, values
    /// `tokens × n_kv_heads·head_dim`. Entries with `s > t` are never read.
    pub fn mix_at_weights(
        &self,
        weights: ArrayView3<'_, f64>,
        weight_radius: ArrayView3<'_, f64>,
        values: ProjectedRows<'_>,
    ) -> Result<(Array2<f64>, Array2<f64>), AttentionProgramError> {
        let g = self.geometry;
        let (tokens, hd) = (values.values.nrows(), g.head_dim);
        expect_shape("value rows", values.values.dim(), (tokens, g.key_value_dim()))?;
        expect_shape("value rows", values.radius.dim(), (tokens, g.key_value_dim()))?;
        let shape = (g.n_heads, tokens, tokens);
        expect_heads("weights", weights.dim(), shape)?;
        expect_heads("weight radius", weight_radius.dim(), shape)?;
        let mut mixed = Array2::zeros((tokens, g.query_dim()));
        let mut mixed_radius = Array2::zeros((tokens, g.query_dim()));
        for head in 0..g.n_heads {
            let (qo, vo) = (head * hd, g.key_value_head(head) * hd);
            for t in 0..tokens {
                let mix_growth = accumulation_growth(t + 1);
                for c in 0..hd {
                    let (mut value, mut abs, mut propagated) = (0.0, 0.0, 0.0);
                    for s in 0..=t {
                        let (w, v) = (weights[[head, t, s]], values.values[[s, vo + c]]);
                        value += w * v;
                        abs += (w * v).abs();
                        propagated += box_product(w, weight_radius[[head, t, s]], v, values.radius[[s, vo + c]]);
                    }
                    mixed[[t, qo + c]] = value;
                    mixed_radius[[t, qo + c]] = propagated + mix_growth * abs;
                }
            }
        }
        Ok((mixed, mixed_radius))
    }

    /// Causal mask, joint softmax per query row and the value read, propagating
    /// the score and value radii. `footprint` is the reservation the caller took
    /// for the scores, the weights and their radii.
    fn mix(
        &self,
        scored: ScoredHeads,
        values: ProjectedRows<'_>,
        footprint: MemoryReservation,
    ) -> Result<ProjectedAttention, AttentionProgramError> {
        let (weights, weight_radius) = self.causal_softmax(scored.scores.view(), scored.radius.view())?;
        let (mixed, mixed_radius) = self.mix_at_weights(weights.view(), weight_radius.view(), values)?;
        Ok(ProjectedAttention {
            scores: scored.scores,
            score_radius: scored.radius,
            weights,
            weight_radius,
            mixed,
            mixed_radius,
            footprint,
        })
    }

}

impl NativeAttention {
    pub fn new(
        geometry: AttentionGeometry,
        rotary: RotaryEmbedding,
        score_scale: f64,
        query: AffineProjection,
        key: AffineProjection,
        value: AffineProjection,
        output: AffineProjection,
    ) -> Result<Self, AttentionProgramError> {
        let attention = RotaryCausalAttention::new(geometry, rotary, score_scale)?;
        let (model, query_dim, kv_dim) = (
            geometry.model_dim,
            geometry.query_dim(),
            geometry.key_value_dim(),
        );
        for (weight_name, bias_name, projection, rows, cols) in [
            ("query weight", "query bias", &query, query_dim, model),
            ("key weight", "key bias", &key, kv_dim, model),
            ("value weight", "value bias", &value, kv_dim, model),
            ("output weight", "output bias", &output, model, query_dim),
        ] {
            expect_shape(weight_name, projection.weight.dim(), (rows, cols))?;
            expect_shape(bias_name, (projection.bias.len(), 1), (rows, 1))?;
        }
        Ok(Self {
            attention,
            query,
            key,
            value,
            output,
            query_key_norm: None,
        })
    }

    /// The source block's dimensions.
    pub fn geometry(&self) -> AttentionGeometry {
        self.attention.geometry
    }

    /// The source's rotary embedding.
    pub fn rotary(&self) -> &RotaryEmbedding {
        &self.attention.rotary
    }

    /// The source's query projection.
    pub fn query(&self) -> &AffineProjection {
        &self.query
    }

    /// The source's key projection.
    pub fn key(&self) -> &AffineProjection {
        &self.key
    }

    /// The source's value projection.
    pub fn value(&self) -> &AffineProjection {
        &self.value
    }

    /// The source's output projection.
    pub fn output(&self) -> &AffineProjection {
        &self.output
    }

    /// The source's multiplier on each query-key inner product.
    pub fn score_scale(&self) -> f64 {
        self.attention.score_scale
    }

    /// Whether the source normalizes each head's queries and keys before the rotary embedding
    /// ([`NativeAttention::with_query_key_norm`]). With a norm, a map that preserves every score
    /// must also commute with that norm, and a caller that projects rows itself would skip it.
    pub fn has_query_key_norm(&self) -> bool {
        self.query_key_norm.is_some()
    }

    /// The source's per-head query/key norm, if it has one
    /// ([`NativeAttention::with_query_key_norm`]).
    pub fn query_key_norm(&self) -> Option<&QueryKeyNorm> {
        self.query_key_norm.as_ref()
    }

    /// The source's per-head query/key RMS norm between the projections and the
    /// rotary embedding (Qwen3 `q_norm`, `k_norm`): `w ⊙ h (mean(h²) + ε)^{-1/2}` on
    /// each head's rows, with the source's declared `ε` and gains.
    pub fn with_query_key_norm(
        self,
        epsilon: f64,
        query_gain: Array1<f64>,
        key_gain: Array1<f64>,
    ) -> Result<Self, AttentionProgramError> {
        let head_dim = self.attention.geometry.head_dim;
        expect_shape("query norm gain", (query_gain.len(), 1), (head_dim, 1))?;
        expect_shape("key norm gain", (key_gain.len(), 1), (head_dim, 1))?;
        Ok(Self {
            query_key_norm: Some(QueryKeyNorm {
                epsilon,
                query_gain,
                key_gain,
            }),
            ..self
        })
    }

    /// The source's per-head query norm, when the source has one.
    fn normalize_queries(
        &self,
        rows: (Array2<f64>, Array2<f64>),
    ) -> Result<(Array2<f64>, Array2<f64>), AttentionProgramError> {
        match &self.query_key_norm {
            None => Ok(rows),
            Some(norm) => head_rms_norm_with_radius(
                ProjectedRows {
                    values: rows.0.view(),
                    radius: rows.1.view(),
                },
                self.attention.geometry.head_dim,
                norm.epsilon,
                norm.query_gain.view(),
            )
            .map_err(AttentionProgramError::Norm),
        }
    }

    /// The source's per-head key norm, when the source has one.
    fn normalize_keys(
        &self,
        rows: (Array2<f64>, Array2<f64>),
    ) -> Result<(Array2<f64>, Array2<f64>), AttentionProgramError> {
        match &self.query_key_norm {
            None => Ok(rows),
            Some(norm) => head_rms_norm_with_radius(
                ProjectedRows {
                    values: rows.0.view(),
                    radius: rows.1.view(),
                },
                self.attention.geometry.head_dim,
                norm.epsilon,
                norm.key_gain.view(),
            )
            .map_err(AttentionProgramError::Norm),
        }
    }

    fn check_input(&self, x: ArrayView2<f64>, positions: &[i64]) -> Result<(), AttentionProgramError> {
        expect_shape(
            "input",
            x.dim(),
            (positions.len(), self.attention.geometry.model_dim),
        )?;
        self.attention.check_positions(positions)
    }

    /// The source block on its original tensors: project queries, keys and
    /// values, attend, and project the output.
    pub fn execute(
        &self,
        governor: &MemoryGovernor,
        x: ArrayView2<f64>,
        positions: &[i64],
    ) -> Result<AttentionExecution, AttentionProgramError> {
        self.check_input(x, positions)?;
        let (queries, query_radius) = self.normalize_queries(project_affine(&self.query, x))?;
        let (keys, key_radius) = self.normalize_keys(project_affine(&self.key, x))?;
        let (values, value_radius) = project_affine(&self.value, x);
        let projected = self.attention.attend_projected(
            governor,
            ProjectedRows {
                values: queries.view(),
                radius: query_radius.view(),
            },
            ProjectedRows {
                values: keys.view(),
                radius: key_radius.view(),
            },
            ProjectedRows {
                values: values.view(),
                radius: value_radius.view(),
            },
            positions,
        )?;
        Ok(self.project_output(projected))
    }

    /// The source's output projection `W_O h_t + b_O` of the head-mixed rows.
    fn project_output(&self, projected: ProjectedAttention) -> AttentionExecution {
        let g = self.attention.geometry;
        let tokens = projected.mixed.nrows();
        // The `n_heads·head_dim`-term inner product and the bias addition.
        let growth = accumulation_growth(g.query_dim() + 1);
        let mut output = Array2::zeros((tokens, g.model_dim));
        let mut output_radius = Array2::zeros((tokens, g.model_dim));
        for t in 0..tokens {
            for (d, w) in self.output.weight.outer_iter().enumerate() {
                let (inner, abs) = inner_with_abs(w, projected.mixed.row(t));
                let bias = self.output.bias[d];
                let propagated: f64 = w
                    .iter()
                    .zip(projected.mixed_radius.row(t).iter())
                    .map(|(o, r)| o.abs() * r)
                    .sum();
                output[[t, d]] = inner + bias;
                output_radius[[t, d]] = propagated + growth * (abs + bias.abs());
            }
        }
        AttentionExecution {
            scores: projected.scores,
            score_radius: projected.score_radius,
            weights: projected.weights,
            weight_radius: projected.weight_radius,
            output,
            output_radius,
            footprint: projected.footprint,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::test_governor;
    use ndarray::array;
    use crate::gated_rewrite::rms_normalizers;
    use qd::Quad;

    /// Small dyadic rationals. Every product and sum the fixture forms stays exact
    /// in f64, so the tensors `U R` and their head-space rows carry no rounding.
    fn dyadic(rows: usize, cols: usize, salt: usize, denominator: f64) -> Array2<f64> {
        Array2::from_shape_fn((rows, cols), |(i, j)| {
            ((i * 7 + j * 3 + salt * 5 + i * j) % 9) as f64 / denominator - 4.0 / denominator
        })
    }

    /// `U R`, exact on the fixture's dyadics.
    fn product(outputs: &Array2<f64>, readins: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_fn((outputs.nrows(), readins.ncols()), |(c, j)| {
            (0..outputs.ncols()).map(|i| outputs[[c, i]] * readins[[i, j]]).sum()
        })
    }

    struct Fixture {
        geometry: AttentionGeometry,
        rotary: RotaryEmbedding,
        score_scale: f64,
        query_outputs: Array2<f64>,
        query_readins: Array2<f64>,
        key_outputs: Array2<f64>,
        key_readins: Array2<f64>,
        value: Array2<f64>,
        output: Array2<f64>,
        x: Array2<f64>,
        positions: Vec<i64>,
        query_bias: Array1<f64>,
        key_bias: Array1<f64>,
        value_bias: Array1<f64>,
        output_bias: Array1<f64>,
        query_key_norm: Option<(f64, Array1<f64>, Array1<f64>)>,
    }

    const QUERY_COMPONENTS: usize = 5;
    const KEY_COMPONENTS: usize = 4;

    impl Fixture {
        /// Two query heads sharing one key/value head, two rotated planes and two
        /// pass-through coordinates per head, positions offset and with gaps.
        fn new(pairing: RotaryPairing) -> Self {
            let geometry = AttentionGeometry {
                model_dim: 4,
                n_heads: 2,
                n_kv_heads: 1,
                head_dim: 6,
            };
            Self {
                geometry,
                rotary: RotaryEmbedding {
                    pairing,
                    inverse_frequencies: vec![1.0, 0.25],
                    attention_scaling: 1.25,
                },
                score_scale: 1.0 / (geometry.head_dim as f64).sqrt(),
                query_outputs: dyadic(12, QUERY_COMPONENTS, 1, 8.0),
                query_readins: dyadic(QUERY_COMPONENTS, 4, 2, 8.0),
                key_outputs: dyadic(6, KEY_COMPONENTS, 3, 8.0),
                key_readins: dyadic(KEY_COMPONENTS, 4, 4, 8.0),
                value: dyadic(6, 4, 5, 8.0),
                output: dyadic(4, 12, 6, 8.0),
                x: dyadic(5, 4, 7, 4.0),
                positions: vec![2, 3, 5, 6, 9],
                query_bias: Array1::zeros(12),
                key_bias: Array1::zeros(6),
                value_bias: Array1::zeros(6),
                output_bias: Array1::zeros(4),
                query_key_norm: None,
            }
        }

        /// Qwen3's per-head query/key RMS norm, with dyadic gains in `[1/2, 3/2]`
        /// and a declared epsilon of `1/64`.
        fn normalized(self) -> Self {
            Self {
                query_key_norm: Some((
                    0.015625,
                    dyadic(6, 1, 12, 8.0).column(0).mapv(|gain| gain + 1.0),
                    dyadic(6, 1, 13, 8.0).column(0).mapv(|gain| gain + 1.0),
                )),
                ..self
            }
        }

        /// Source projection biases, as on Pythia's `query_key_value` and `dense`.
        /// They are dyadic, so the edited-tensor route stays exact.
        fn biased(self) -> Self {
            Self {
                query_bias: dyadic(12, 1, 8, 8.0).column(0).to_owned(),
                key_bias: dyadic(6, 1, 9, 8.0).column(0).to_owned(),
                value_bias: dyadic(6, 1, 10, 8.0).column(0).to_owned(),
                output_bias: dyadic(4, 1, 11, 8.0).column(0).to_owned(),
                ..self
            }
        }

        fn native(&self) -> NativeAttention {
            let (query, key) = (
                product(&self.query_outputs, &self.query_readins),
                product(&self.key_outputs, &self.key_readins),
            );
            let affine = |weight: Array2<f64>, bias: &Array1<f64>| AffineProjection {
                weight,
                bias: bias.clone(),
            };
            let native = NativeAttention::new(
                self.geometry,
                self.rotary.clone(),
                self.score_scale,
                affine(query, &self.query_bias),
                affine(key, &self.key_bias),
                affine(self.value.clone(), &self.value_bias),
                affine(self.output.clone(), &self.output_bias),
            )
            .expect("fixture tensors match the geometry");
            match &self.query_key_norm {
                None => native,
                Some((epsilon, query_gain, key_gain)) => native
                    .with_query_key_norm(*epsilon, query_gain.clone(), key_gain.clone())
                    .expect("norm gains match the head dimension"),
            }
        }
    }

    /// The source's pairing and sign on a hand-computed score: the query token at
    /// position 1 reads `e_0`, the key token at position 0 reads `e_1`. gptj's
    /// `rotate_every_two` rotates plane (0, 1), so `e_0 ↦ α(cos θ, sin θ)` and the
    /// score is `+σα² sin θ`; llama's `rotate_half` puts `e_0` and `e_1` in the
    /// different planes (0, 2) and (1, 3), so the score is zero.
    #[test]
    fn rotary_pairing_and_sign_follow_the_source() {
        let geometry = AttentionGeometry {
            model_dim: 4,
            n_heads: 1,
            n_kv_heads: 1,
            head_dim: 4,
        };
        let (frequency, score_scale, attention_scaling) = (0.75_f64, 0.5, 1.25);
        let x = array![[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]];
        let positions = [0_i64, 1];
        let identity = || AffineProjection {
            weight: Array2::eye(4),
            bias: Array1::zeros(4),
        };
        let score = |pairing: RotaryPairing| -> (f64, f64) {
            let executed = NativeAttention::new(
                geometry,
                RotaryEmbedding {
                    pairing,
                    inverse_frequencies: vec![frequency, frequency],
                    attention_scaling,
                },
                score_scale,
                identity(),
                identity(),
                identity(),
                identity(),
            )
            .expect("identity block")
            .execute(test_governor(), x.view(), &positions)
            .expect("execution");
            (executed.scores[[0, 1, 0]], executed.score_radius[[0, 1, 0]])
        };
        let expected = score_scale * attention_scaling * attention_scaling * frequency.sin();
        // Three products and one libm ulp in the reference itself.
        let expected_error = (accumulation_growth(3) + f64::EPSILON) * expected.abs();
        let (interleaved, interleaved_radius) = score(RotaryPairing::Interleaved);
        let (half_split, half_split_radius) = score(RotaryPairing::HalfSplit);
        let tolerance = interleaved_radius + expected_error;
        assert!(
            (interleaved - expected).abs() <= tolerance,
            "interleaved score {interleaved}, expected {expected} within {tolerance}"
        );
        assert!(
            (interleaved + expected).abs() > tolerance,
            "the opposite rotation sign must be distinguishable from the source's"
        );
        assert!(
            half_split.abs() <= half_split_radius,
            "half-split score {half_split} must be zero within {half_split_radius}"
        );
        assert!(
            (half_split - expected).abs() > half_split_radius + expected_error,
            "the two pairings must be distinguishable on this fixture"
        );
    }

    /// Exact rows give bit for bit what explicit zero radii give. A nonzero query
    /// radius changes the bits, so the comparison sees the radius channel.
    #[test]
    fn exact_rows_match_explicit_zero_radii() {
        let fixture = Fixture::new(RotaryPairing::HalfSplit).biased();
        let native = fixture.native();
        let queries = project_affine(&native.query, fixture.x.view()).0;
        let keys = project_affine(&native.key, fixture.x.view()).0;
        let values = project_affine(&native.value, fixture.x.view()).0;
        let (zero_queries, zero_keys, zero_values) = (
            Array2::zeros(queries.dim()),
            Array2::zeros(keys.dim()),
            Array2::zeros(values.dim()),
        );
        let bits = |p: &ProjectedAttention| -> Vec<u64> {
            p.scores
                .iter()
                .chain(p.score_radius.iter())
                .chain(p.weights.iter())
                .chain(p.weight_radius.iter())
                .chain(p.mixed.iter())
                .chain(p.mixed_radius.iter())
                .map(|v| v.to_bits())
                .collect()
        };
        let explicit = native
            .attention
            .attend_projected(
                test_governor(),
                ProjectedRows {
                    values: queries.view(),
                    radius: zero_queries.view(),
                },
                ProjectedRows {
                    values: keys.view(),
                    radius: zero_keys.view(),
                },
                ProjectedRows {
                    values: values.view(),
                    radius: zero_values.view(),
                },
                &fixture.positions,
            )
            .expect("explicit zero radii");
        let exact = native
            .attention
            .attend_projected(
                test_governor(),
                ProjectedRows::exact(queries.view()),
                ProjectedRows::exact(keys.view()),
                ProjectedRows::exact(values.view()),
                &fixture.positions,
            )
            .expect("exact rows");
        assert_eq!(bits(&exact), bits(&explicit), "exact rows must equal explicit zero radii");
        let widened_queries = Array2::from_elem(queries.dim(), 0.125);
        let widened = native
            .attention
            .attend_projected(
                test_governor(),
                ProjectedRows {
                    values: queries.view(),
                    radius: widened_queries.view(),
                },
                ProjectedRows::exact(keys.view()),
                ProjectedRows::exact(values.view()),
                &fixture.positions,
            )
            .expect("widened query radius");
        assert_ne!(bits(&widened), bits(&explicit), "a nonzero query radius must change the radii");
    }

    /// The softmax at given scores and the value read at given weights, chained on
    /// `attend_projected`'s own scores, reproduce its weights, mixed rows and radii
    /// bit for bit. Controls:
    /// - a moved admissible score changes its row's weights and leaves the other rows alone;
    /// - a NaN in a masked entry (`s > t`) is never read;
    /// - an array with the wrong head count is refused.
    /// A logit box too wide for `expm1` keeps every weight's radius a finite number within the
    /// weight's `[0, 1]` range: a weight near one, one that underflows to zero (where the box bound
    /// alone is `0 · ∞`), and one in between. The exact weights at the box's corners stay within
    /// the radii. Control: a narrow box's radius is its first-order box bound `w (expm1(b) + ε)`
    /// up to the endpoints' outward rounding.
    #[test]
    fn a_wide_logit_box_keeps_every_weight_radius_a_finite_number_in_the_unit_range() {
        let logits = [0.0, -800.0, -0.5];
        let wide = [1.0e3; 3];
        let (weights, radius) = attention_weights(&logits, &wide).expect("finite logits");
        assert_eq!(weights[1], 0.0, "the middle weight underflows");
        for (s, (&weight, &bound)) in weights.iter().zip(&radius).enumerate() {
            assert!(bound.is_finite(), "weight {s}: radius {bound}");
            assert!(bound <= (1.0 - weight).next_up().max(weight), "weight {s}: radius {bound} past its range");
        }
        // The corners of the box: raise one logit by the radius and lower the others.
        for raised in 0..logits.len() {
            let corner: Vec<f64> = logits
                .iter()
                .enumerate()
                .map(|(t, &logit)| if t == raised { logit + wide[t] } else { logit - wide[t] })
                .collect();
            let (exact, _) = attention_weights(&corner, &[0.0; 3]).expect("finite corner logits");
            for s in 0..logits.len() {
                assert!(
                    (exact[s] - weights[s]).abs() <= radius[s],
                    "corner {raised}, weight {s}: {} moved past {}",
                    exact[s],
                    radius[s]
                );
            }
        }
        let narrow = [1.0e-3; 3];
        let (weights, radius) = attention_weights(&logits, &narrow).expect("finite logits");
        let (log_weights, evaluation) = log_softmax_with_error(&logits).expect("finite logits");
        for s in 0..logits.len() {
            let boxed = weights[s] * ((narrow[s] + 1.0e-3 + evaluation[s]).exp_m1() + f64::EPSILON);
            assert_eq!(weights[s].to_bits(), log_weights[s].exp().to_bits());
            // The endpoints' outward roundings add a few ε to the relative bound, which is
            // below 1e-9 of `expm1(2e-3)`. An underflowed weight keeps its two subnormal ulps.
            assert!(
                radius[s] >= boxed * (1.0 - 1.0e-9) && radius[s] <= boxed * (1.0 + 1.0e-9) + f64::from_bits(2),
                "weight {s}: a narrow box keeps its first-order box bound {boxed}, got {}",
                radius[s]
            );
        }
    }

    /// A weight that underflows to zero in a logit box that is finite but wide enough to lift
    /// it back above zero keeps a radius that encloses the exact weights at the box's corners.
    /// The relative box bound `ŵ (expm1(b) + ε)` is `0` there, but the exact weight at the
    /// raised corner is `exp(-720)`, a positive subnormal. Control: the weight that stays near
    /// one keeps a finite radius within its range.
    #[test]
    fn an_underflowed_weight_in_a_finite_box_keeps_a_radius_enclosing_the_exact_weight() {
        let logits = [0.0, -760.0];
        let finite = [20.0; 2];
        let (weights, radius) = attention_weights(&logits, &finite).expect("finite logits");
        assert_eq!(weights[1], 0.0, "the second weight underflows");
        assert!(radius[1] > 0.0, "an underflowed weight's exact value is positive");
        for raised in 0..logits.len() {
            let corner: Vec<f64> = logits
                .iter()
                .enumerate()
                .map(|(t, &logit)| if t == raised { logit + finite[t] } else { logit - finite[t] })
                .collect();
            let (exact, _) = attention_weights(&corner, &[0.0; 2]).expect("finite corner logits");
            if raised == 1 {
                assert!(exact[1] > 0.0, "the raised corner lifts the weight out of underflow");
            }
            for s in 0..logits.len() {
                assert!(
                    (exact[s] - weights[s]).abs() <= radius[s],
                    "corner {raised}, weight {s}: {} moved past {}",
                    exact[s],
                    radius[s]
                );
            }
        }
        assert!(radius[0].is_finite() && radius[0] <= (1.0 - weights[0]).next_up().max(weights[0]));
    }

    #[test]
    fn softmax_and_value_read_at_external_inputs_chain_to_attend_projected() {
        let fixture = Fixture::new(RotaryPairing::HalfSplit).biased();
        let native = fixture.native();
        let queries = project_affine(&native.query, fixture.x.view()).0;
        let keys = project_affine(&native.key, fixture.x.view()).0;
        let values = project_affine(&native.value, fixture.x.view()).0;
        let value_radius = Array2::from_elem(values.dim(), 0.0625);
        let value_rows = ProjectedRows {
            values: values.view(),
            radius: value_radius.view(),
        };
        let attention = &native.attention;
        let projected = attention
            .attend_projected(
                test_governor(),
                ProjectedRows::exact(queries.view()),
                ProjectedRows::exact(keys.view()),
                value_rows,
                &fixture.positions,
            )
            .expect("projected attention");
        fn bits<D: ndarray::Dimension>(arrays: &[&ndarray::Array<f64, D>]) -> Vec<u64> {
            arrays.iter().flat_map(|a| a.iter().map(|v| v.to_bits())).collect()
        }
        let stage = attention
            .weights_at_scores(test_governor(), projected.scores.view(), projected.score_radius.view())
            .expect("weights at attend_projected's scores");
        assert_eq!(
            bits(&[&stage.weights, &stage.weight_radius]),
            bits(&[&projected.weights, &projected.weight_radius]),
            "the softmax stage must reproduce attend_projected's weights and radii"
        );
        let (mixed, mixed_radius) = attention
            .mix_at_weights(stage.weights.view(), stage.weight_radius.view(), value_rows)
            .expect("value read at those weights");
        assert_eq!(
            bits(&[&mixed, &mixed_radius]),
            bits(&[&projected.mixed, &projected.mixed_radius]),
            "the value-read stage must reproduce attend_projected's mixed rows and radii"
        );

        let row = |w: &Array3<f64>, head: usize, t: usize| -> Vec<u64> {
            w.slice(ndarray::s![head, t, ..]).iter().map(|v| v.to_bits()).collect()
        };
        let mut moved = projected.scores.clone();
        moved[[0, 3, 1]] += 0.25;
        let moved_weights = attention
            .weights_at_scores(test_governor(), moved.view(), projected.score_radius.view())
            .expect("moved scores")
            .weights;
        assert_ne!(row(&moved_weights, 0, 3), row(&stage.weights, 0, 3), "a moved score must move its row");
        assert_eq!(row(&moved_weights, 1, 3), row(&stage.weights, 1, 3), "a moved score must leave other heads alone");
        assert_eq!(row(&moved_weights, 0, 4), row(&stage.weights, 0, 4), "a moved score must leave other rows alone");

        let mut masked = projected.scores.clone();
        masked[[0, 1, 3]] = f64::NAN;
        let masked_weights = attention
            .weights_at_scores(test_governor(), masked.view(), projected.score_radius.view())
            .expect("a NaN past the causal mask is never read")
            .weights;
        assert_eq!(
            bits(&[&masked_weights]),
            bits(&[&stage.weights]),
            "entries with s > t must not be read"
        );
        let mut masked_mix = stage.weights.clone();
        masked_mix[[1, 2, 4]] = f64::NAN;
        let masked_mixed = attention
            .mix_at_weights(masked_mix.view(), stage.weight_radius.view(), value_rows)
            .expect("a NaN weight past the causal mask is never read")
            .0;
        assert_eq!(
            bits(&[&masked_mixed]),
            bits(&[&mixed]),
            "weights with s > t must not be read"
        );

        let one_head = projected.scores.slice(ndarray::s![..1, .., ..]);
        assert_eq!(
            attention
                .weights_at_scores(test_governor(), one_head, projected.score_radius.slice(ndarray::s![..1, .., ..]))
                .expect_err("one head for a two-head geometry"),
            AttentionProgramError::HeadShape {
                tensor: "scores",
                expected: (2, 5, 5),
                found: (1, 5, 5),
            }
        );
        assert_eq!(
            attention
                .mix_at_weights(stage.weights.slice(ndarray::s![..1, .., ..]), stage.weight_radius.view(), value_rows)
                .expect_err("one head of weights for a two-head geometry"),
            AttentionProgramError::HeadShape {
                tensor: "weights",
                expected: (2, 5, 5),
                found: (1, 5, 5),
            }
        );
    }

    /// A score footprint beyond the process memory budget is refused, typed, before
    /// anything of that size is allocated. `2^20` tokens on two heads ask for four
    /// `2 × 2^20 × 2^20` arrays, 64 TiB. Every input is a zero-stride view of one
    /// zero, so the only large allocation left is the one the reservation guards, and
    /// making it would abort the test process instead of returning.
    #[test]
    fn a_score_footprint_beyond_the_memory_budget_is_refused_before_allocating() {
        let fixture = Fixture::new(RotaryPairing::HalfSplit);
        let native = fixture.native();
        let g = fixture.geometry;
        let tokens = 1_usize << 20;
        let positions: Vec<i64> = (0..tokens as i64).collect();
        let zeros = |width: usize| {
            ArrayView2::from_shape((tokens, width).strides((0, 0)), &EXACT_RADIUS[..])
                .expect("a zero-stride view reads only its one entry")
        };
        let requested = |copies: usize| {
            gam_runtime::resource::dense_f64_bytes(g.n_heads * tokens, tokens).expect("fits in usize") * copies
        };
        fn refused<T>(result: &Result<T, AttentionProgramError>, bytes: usize) -> bool {
            matches!(
                result,
                Err(AttentionProgramError::Memory(MemoryReservationError::BudgetExceeded { requested_bytes, .. }))
                    if *requested_bytes == bytes
            )
        }
        let projected = native.attention.attend_projected(
            test_governor(),
            ProjectedRows::exact(zeros(g.query_dim())),
            ProjectedRows::exact(zeros(g.key_value_dim())),
            ProjectedRows::exact(zeros(g.key_value_dim())),
            &positions,
        );
        assert!(
            refused(&projected, requested(4)),
            "attend_projected must refuse its four score arrays, got {projected:?}"
        );
        let executed = native.execute(test_governor(), zeros(g.model_dim), &positions);
        assert!(
            refused(&executed, requested(4)),
            "the native execution must refuse its four score arrays, got {executed:?}"
        );
        let scores = ArrayView3::from_shape((g.n_heads, tokens, tokens).strides((0, 0, 0)), &EXACT_RADIUS[..])
            .expect("a zero-stride view reads only its one entry");
        let weights = native.attention.weights_at_scores(test_governor(), scores, scores);
        assert!(
            refused(&weights, requested(2)),
            "the softmax stage must refuse its two weight arrays, got {weights:?}"
        );
    }

    /// While a result lives, the governor's ledger holds exactly its
    /// `heads × tokens × tokens` footprint, and dropping it releases the charge. The
    /// test reserves on a private governor, so its ledger holds only these results.
    #[test]
    fn the_score_footprint_is_held_while_a_result_lives_and_released_on_drop() {
        let fixture = Fixture::new(RotaryPairing::HalfSplit).biased();
        let native = fixture.native();
        let g = fixture.geometry;
        let tokens = fixture.positions.len();
        let array = gam_runtime::resource::dense_f64_bytes(g.n_heads * tokens, tokens).expect("fits in usize");
        let governor = MemoryGovernor::with_budget_bytes(1 << 30);
        let idle = governor.remaining_bytes();
        let execution = native
            .execute(&governor, fixture.x.view(), &fixture.positions)
            .expect("native execution");
        assert_eq!(execution.reserved_bytes(), 4 * array, "an execution reserves its four arrays");
        assert_eq!(idle - governor.remaining_bytes(), 4 * array, "the ledger holds them while it lives");
        let stage = native
            .attention
            .weights_at_scores(&governor, execution.scores.view(), execution.score_radius.view())
            .expect("weights at the execution's scores");
        assert_eq!(stage.reserved_bytes(), 2 * array, "the softmax stage reserves its two arrays");
        assert_eq!(idle - governor.remaining_bytes(), 6 * array, "the ledger holds both results");
        drop(execution);
        assert_eq!(idle - governor.remaining_bytes(), 2 * array, "dropping an execution releases its four arrays");
        drop(stage);
        assert_eq!(governor.remaining_bytes(), idle, "nothing stays reserved once every result is dropped");
    }

    /// Double-double unit roundoff: the low word's last place relative to the high
    /// word, `2^-104`.
    const QUAD_UNIT: f64 = f64::EPSILON * f64::EPSILON;

    fn ulp(value: f64) -> f64 {
        value.abs().next_up() - value.abs()
    }

    /// `sin x` or `cos x` by the alternating Taylor series in double-double. Its
    /// derived error is `QUAD_UNIT` times three rounded operations per term times the
    /// largest term, plus the first omitted term: the series alternates, and its terms
    /// decrease once the order exceeds `|x|`.
    fn quad_trig(x: f64, sine: bool) -> (Quad, f64) {
        let square = Quad::from_f64(x) * Quad::from_f64(x);
        let mut term = if sine {
            Quad::from_f64(x)
        } else {
            Quad::from_f64(1.0)
        };
        let mut order = if sine { 1.0_f64 } else { 0.0 };
        let mut sum = term;
        let mut largest = term.0.abs();
        let mut terms = 1usize;
        loop {
            let next = Quad::from_f64(0.0) - term * square / Quad::from_f64((order + 1.0) * (order + 2.0));
            order += 2.0;
            if order > x.abs() && next.0.abs() <= QUAD_UNIT * largest {
                return (sum, 3.0 * terms as f64 * QUAD_UNIT * largest + next.0.abs());
            }
            term = next;
            sum += term;
            largest = largest.max(term.0.abs());
            terms += 1;
        }
    }

    /// `e^x` in double-double through the positive series; for `x < 0` it computes
    /// `1/e^{-x}`, so no term cancels. Once the order exceeds `2|x|` consecutive terms
    /// shrink by at least half, so the omitted tail is at most twice the first
    /// omitted term.
    fn quad_exp(x: f64) -> (Quad, f64) {
        let magnitude = x.abs();
        let mut term = Quad::from_f64(1.0);
        let mut sum = term;
        let mut order = 0.0_f64;
        let mut terms = 1usize;
        loop {
            order += 1.0;
            term = term * Quad::from_f64(magnitude) / Quad::from_f64(order);
            if order + 1.0 > 2.0 * magnitude && term.0 <= QUAD_UNIT * sum.0 {
                let relative = 3.0 * terms as f64 * QUAD_UNIT + 2.0 * term.0 / sum.0;
                return if x < 0.0 {
                    let inverse = Quad::from_f64(1.0) / sum;
                    (inverse, inverse.0 * (relative + QUAD_UNIT))
                } else {
                    (sum, sum.0 * relative)
                };
            }
            sum += term;
            terms += 1;
        }
    }

    /// `|f_libm − f_ref| ≤ ulp(f_libm) + e_ref`, evaluated in double-double.
    fn within_one_ulp(computed: f64, (reference, reference_error): (Quad, f64)) -> bool {
        (Quad::from_f64(computed) - reference).0.abs() <= ulp(computed) + reference_error
    }

    /// The positive control: `computed` moved 2 ulp further from the reference,
    /// so its error is at least 2 ulp whichever side libm rounded to.
    fn moved_away(computed: f64, reference: Quad) -> f64 {
        if (Quad::from_f64(computed) - reference).0 >= 0.0 {
            computed.next_up().next_up()
        } else {
            computed.next_down().next_down()
        }
    }

    /// The radii's libm assumption, checked at every angle and exponent the fixtures
    /// evaluate: `sin` and `cos` at `p·θ` for `|p| ≤ 9` and `θ ∈ {1, 1/4, 3/4}`, and
    /// `exp` at every executed log weight. This checks the documented platform bound
    /// on the platform the tests run on. A value moved 2 ulp away must fail.
    #[test]
    fn libm_sin_cos_exp_are_within_one_ulp_at_the_fixture_points() {
        let mut checked = 0usize;
        for multiplier in -9..=9 {
            for frequency in [1.0, 0.25, 0.75] {
                let angle = f64::from(multiplier) * frequency;
                let (sin, cos) = angle.sin_cos();
                for (value, reference) in [(sin, quad_trig(angle, true)), (cos, quad_trig(angle, false))] {
                    assert!(
                        within_one_ulp(value, reference),
                        "libm trig at {angle}: {value} is beyond one ulp"
                    );
                    assert!(
                        !within_one_ulp(moved_away(value, reference.0), reference),
                        "a value moved 2 ulp away at {angle} must fail"
                    );
                    checked += 1;
                }
            }
        }
        let fixture = Fixture::new(RotaryPairing::HalfSplit).biased();
        let executed = fixture
            .native()
            .execute(test_governor(), fixture.x.view(), &fixture.positions)
            .expect("native execution");
        for head in 0..fixture.geometry.n_heads {
            for t in 0..fixture.positions.len() {
                let row: Vec<f64> = (0..=t).map(|s| executed.scores[[head, t, s]]).collect();
                let (log_weights, evaluation_radius) = log_softmax_with_error(&row).expect("finite row");
                assert_eq!(log_weights.len(), evaluation_radius.len(), "one radius per log weight");
                for &log_weight in &log_weights {
                    let value = log_weight.exp();
                    let reference = quad_exp(log_weight);
                    assert!(
                        within_one_ulp(value, reference),
                        "libm exp at {log_weight}: {value} is beyond one ulp"
                    );
                    assert!(
                        !within_one_ulp(moved_away(value, reference.0), reference),
                        "a value moved 2 ulp away at {log_weight} must fail"
                    );
                    checked += 1;
                }
            }
        }
        assert!(checked > 0, "the libm check must evaluate at least one point");
    }

    /// The read accessors return the parts the block was built from (geometry, rotary embedding,
    /// score scale and all four projections), and `has_query_key_norm` and `query_key_norm`
    /// follow `with_query_key_norm`, whose epsilon and gains the norm returns. Positive controls:
    /// an edited query tensor compares unequal to the accessor's, and the fixture's query and key
    /// gains differ.
    #[test]
    fn accessors_return_the_source_parts() {
        let fixture = Fixture::new(RotaryPairing::HalfSplit).biased();
        let native = fixture.native();
        assert_eq!(native.geometry(), fixture.geometry, "geometry");
        assert_eq!(native.rotary(), &fixture.rotary, "rotary embedding");
        assert_eq!(native.query().bias, fixture.query_bias, "query bias");
        assert_eq!(native.key().bias, fixture.key_bias, "key bias");
        let source_query = product(&fixture.query_outputs, &fixture.query_readins);
        assert_eq!(native.query().weight, source_query, "query weight");
        let source_key = product(&fixture.key_outputs, &fixture.key_readins);
        assert_eq!(native.key().weight, source_key, "key weight");
        assert_eq!(native.value().weight, fixture.value, "value weight");
        assert_eq!(native.value().bias, fixture.value_bias, "value bias");
        assert_eq!(native.output().weight, fixture.output, "output weight");
        assert_eq!(native.output().bias, fixture.output_bias, "output bias");
        assert_eq!(native.score_scale(), fixture.score_scale, "score scale");
        assert!(
            !native.has_query_key_norm(),
            "a block built without q_norm/k_norm has no query/key norm"
        );
        let normalized = Fixture::new(RotaryPairing::HalfSplit).normalized().native();
        assert!(
            normalized.has_query_key_norm(),
            "with_query_key_norm must set the query/key norm"
        );
        assert!(
            native.query_key_norm().is_none(),
            "a block built without q_norm/k_norm exposes no norm"
        );
        let (epsilon, query_gain, key_gain) = Fixture::new(RotaryPairing::HalfSplit)
            .normalized()
            .query_key_norm
            .expect("the normalized fixture declares a norm");
        assert_ne!(query_gain, key_gain, "distinct gains, so a swapped accessor fails");
        let norm = normalized
            .query_key_norm()
            .expect("with_query_key_norm must expose the norm");
        assert_eq!(norm.epsilon(), epsilon, "norm epsilon");
        assert_eq!(norm.query_gain(), query_gain.view(), "query norm gain");
        assert_eq!(norm.key_gain(), key_gain.view(), "key norm gain");
        let mut edited_query = source_query.clone();
        edited_query[[0, 1]] += 0.125;
        assert_ne!(
            native.query().weight,
            edited_query,
            "the accessor must return the source query tensor, not an edited one"
        );
    }

    /// Offsets sampled across a radius-`r` box, `−r … r` in quarters of `r`. With a
    /// dyadic `r` every sampled point stays dyadic.
    fn box_offsets(radius: f64) -> [f64; 5] {
        [-radius, -0.5 * radius, 0.0, 0.5 * radius, radius]
    }

    /// The score radius encloses the exact score's deviation for queries and keys
    /// sampled across their radius boxes, including the far corner, where the
    /// `r_q r_k` term that a first-order propagation drops is the margin.
    /// - The block is rotary-free (`rotary_dim` 0) with `σ = 1/2`.
    /// - Rows and radius are positive dyadics, so every sampled score
    ///   `σ Σ (q + δ_q)(k + δ_k)` is exact in `f64`. With `q, k > 0` the deviation is
    ///   largest at the all-`+r` corner, so the sampled diagonal contains the worst case.
    /// - Control: the far corner `δ_q = δ_k = r` lies outside the radius less its
    ///   `σ Σ r_q r_k`.
    #[test]
    fn score_radius_encloses_the_exact_score_across_the_input_boxes() {
        let geometry = AttentionGeometry {
            model_dim: 4,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 6,
        };
        let rotary = RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: vec![],
            attention_scaling: 1.0,
        };
        let attention = RotaryCausalAttention::new(geometry, rotary, 0.5).expect("rotary-free geometry");
        let positions = [0_i64, 1, 2];
        let radius = 0.125;
        let queries = dyadic(3, geometry.query_dim(), 1, 8.0).mapv(|v| v.abs() + radius);
        let keys = dyadic(3, geometry.key_value_dim(), 2, 8.0).mapv(|v| v.abs() + radius);
        let values = dyadic(3, geometry.key_value_dim(), 3, 8.0);
        let (query_radius, key_radius) = (
            Array2::from_elem(queries.dim(), radius),
            Array2::from_elem(keys.dim(), radius),
        );
        let projected = attention
            .attend_projected(
                test_governor(),
                ProjectedRows {
                    values: queries.view(),
                    radius: query_radius.view(),
                },
                ProjectedRows {
                    values: keys.view(),
                    radius: key_radius.view(),
                },
                ProjectedRows::exact(values.view()),
                &positions,
            )
            .expect("rotary-free attention");
        let hd = geometry.head_dim;
        for head in 0..geometry.n_heads {
            let (qo, ko) = (head * hd, geometry.key_value_head(head) * hd);
            for t in 0..positions.len() {
                for s in 0..=t {
                    let bound = projected.score_radius[[head, t, s]];
                    let reach_at = |dq: f64, dk: f64| {
                        let exact = 0.5
                            * (0..hd)
                                .map(|c| (queries[[t, qo + c]] + dq) * (keys[[s, ko + c]] + dk))
                                .sum::<f64>();
                        (exact - projected.scores[[head, t, s]]).abs()
                    };
                    for dq in box_offsets(radius) {
                        for dk in box_offsets(radius) {
                            let reach = reach_at(dq, dk);
                            assert!(
                                reach <= bound,
                                "score ({head}, {t}, {s}) at offsets ({dq}, {dk}): {reach} beyond the radius {bound}"
                            );
                        }
                    }
                    let first_order = bound - 0.5 * hd as f64 * radius * radius;
                    assert!(
                        reach_at(radius, radius) > first_order,
                        "score ({head}, {t}, {s}): the radius without r_q r_k, {first_order}, still covers the corner"
                    );
                }
            }
        }
    }

    /// The mixed-row radius encloses the exact value read's deviation for weights and
    /// values sampled across their radius boxes, including the far corner, where the
    /// `r_w r_v` term is the margin. Weights, values and radius are positive
    /// dyadics, so every sampled read `Σ_s (w + δ_w)(v + δ_v)` is exact in `f64`, and with
    /// `w, v > 0` the all-`+r` corner, which is sampled, is the worst case.
    /// Control: the far corner lies outside the radius less its `Σ_s r_w r_v`.
    #[test]
    fn mixed_radius_encloses_the_exact_read_across_the_input_boxes() {
        let fixture = Fixture::new(RotaryPairing::HalfSplit);
        let g = fixture.geometry;
        let attention =
            RotaryCausalAttention::new(g, fixture.rotary.clone(), fixture.score_scale).expect("fixture geometry");
        let tokens = 4;
        let radius = 0.125;
        let weights = Array3::from_shape_fn((g.n_heads, tokens, tokens), |(h, t, s)| {
            if s <= t {
                ((h + 2 * t + 3 * s) % 4 + 1) as f64 / 8.0
            } else {
                0.0
            }
        });
        let values = dyadic(tokens, g.key_value_dim(), 5, 8.0).mapv(|v| v.abs() + radius);
        let weight_radius = Array3::from_elem(weights.dim(), radius);
        let value_radius = Array2::from_elem(values.dim(), radius);
        let (mixed, mixed_radius) = attention
            .mix_at_weights(
                weights.view(),
                weight_radius.view(),
                ProjectedRows {
                    values: values.view(),
                    radius: value_radius.view(),
                },
            )
            .expect("value read");
        let hd = g.head_dim;
        for head in 0..g.n_heads {
            let (qo, vo) = (head * hd, g.key_value_head(head) * hd);
            for t in 0..tokens {
                for c in 0..hd {
                    let bound = mixed_radius[[t, qo + c]];
                    let reach_at = |dw: f64, dv: f64| {
                        let exact = (0..=t)
                            .map(|s| (weights[[head, t, s]] + dw) * (values[[s, vo + c]] + dv))
                            .sum::<f64>();
                        (exact - mixed[[t, qo + c]]).abs()
                    };
                    for dw in box_offsets(radius) {
                        for dv in box_offsets(radius) {
                            let reach = reach_at(dw, dv);
                            assert!(
                                reach <= bound,
                                "mixed ({t}, {head}, {c}) at offsets ({dw}, {dv}): {reach} beyond the radius {bound}"
                            );
                        }
                    }
                    let first_order = bound - (t + 1) as f64 * radius * radius;
                    assert!(
                        reach_at(radius, radius) > first_order,
                        "mixed ({t}, {head}, {c}): the radius without r_w r_v, {first_order}, still covers the corner"
                    );
                }
            }
        }
    }

    /// The per-head RMS norm's radius encloses the exact norm for rows sampled across
    /// their radius box, where a first-order propagation does not.
    /// - Setup: one head of `head_dim` 2, rows `x̂ = (1/8, 1/2)` with radius
    ///   `(0, 1/4)`, unit gains and `ε = 1/64`.
    /// - Why first order fails: moving `x_2` down by its radius raises `ν` convexly,
    ///   so `y_1` moves by more than the linearization's bound.
    /// - How it is compared: the exact norm at each dyadic sample is checked by
    ///   [`encloses_unit_rms_norm`].
    /// - Control: the first-order radius misses the sample `x_2 = 1/4`.
    #[test]
    fn query_key_norm_radius_encloses_the_exact_norm_across_the_input_box() {
        let epsilon = 0.015625;
        let rows = array![[0.125, 0.5]];
        let radius = array![[0.0, 0.25]];
        let gain = array![1.0, 1.0];
        let (normalized, bound) =
            head_rms_norm_with_radius(ProjectedRows { values: rows.view(), radius: radius.view() }, 2, epsilon, gain.view())
                .expect("finite head row");
        let encloses = |x: [f64; 2], c: usize, center: f64, reach: f64| encloses_unit_rms_norm(&x, epsilon, c, center, reach);
        for x2 in [0.25, 0.375, 0.5, 0.625, 0.75] {
            for c in 0..2 {
                assert!(
                    encloses([0.125, x2], c, normalized[[0, c]], bound[[0, c]]),
                    "coordinate {c} at x_2 = {x2}: the exact norm lies outside {} ± {}",
                    normalized[[0, c]],
                    bound[[0, c]]
                );
            }
        }
        let nu = rms_normalizers(rows.view(), epsilon).expect("finite head row")[0];
        let weighted = (rows[[0, 0]] * radius[[0, 0]] + rows[[0, 1]] * radius[[0, 1]]) / 2.0;
        let first_order = nu * (radius[[0, 0]] + nu * nu * rows[[0, 0]] * weighted)
            + accumulation_growth(8) * normalized[[0, 0]].abs();
        assert!(
            !encloses([0.125, 0.25], 0, normalized[[0, 0]], first_order),
            "the first-order radius {first_order} must miss the exact norm at x_2 = 1/4"
        );
    }

    /// Whether the exact unit-gain RMS norm `y_c = x_c / √s`, `s = mean x² + ε`, lies in
    /// `[center − reach, center + reach]`, checked in double-double without a square root.
    /// `y_c` has the sign of `x_c`, so for `x_c < 0` the interval is reflected through zero.
    /// For `y ≥ 0`, `y ∈ [low, high]` holds exactly when `high ≥ 0` and
    /// `max(low, 0)² s ≤ x_c² ≤ high² s`.
    fn encloses_unit_rms_norm(x: &[f64], epsilon: f64, c: usize, center: f64, reach: f64) -> bool {
        let quad = Quad::from_f64;
        let s = x.iter().fold(quad(0.0), |sum, &value| sum + quad(value) * quad(value)) / quad(x.len() as f64)
            + quad(epsilon);
        let center = if x[c] < 0.0 { -center } else { center };
        let square = quad(x[c]) * quad(x[c]);
        let low = quad(center) - quad(reach);
        let low = if low.0 < 0.0 { quad(0.0) } else { low };
        let high = quad(center) + quad(reach);
        high.0 >= 0.0 && (low * low * s - square).0 <= 0.0 && (square - high * high * s).0 <= 0.0
    }

    /// A zero-epsilon head row whose squares round into the subnormal range keeps its
    /// radius around the exact norm, where the relative-only band misses by ten orders.
    /// - Setup: the head row `x̂ = (10^-160, 0)` with zero radius, unit gains and `ε = 0`.
    ///   Its square `10^-320` is subnormal, so `ŷ_1` is off `√2` by a relative `5.6·10^-6`.
    /// - How it is compared: with `ε = 0` the exact norm is scale invariant, so
    ///   `y(x̂) = y(1, 0) = (√2, 0)` exactly, checked at `(1, 0)` where double-double is exact
    ///   enough.
    /// - Control: the relative-only band `γ_(d+6) |ŷ_1|` misses `√2`.
    #[test]
    fn query_key_norm_radius_encloses_a_zero_epsilon_row_whose_squares_underflow() {
        let rows = array![[1e-160, 0.0]];
        let radius = array![[0.0, 0.0]];
        let gain = array![1.0, 1.0];
        let (normalized, bound) =
            head_rms_norm_with_radius(ProjectedRows { values: rows.view(), radius: radius.view() }, 2, 0.0, gain.view())
                .expect("finite head row");
        for c in 0..2 {
            assert!(
                encloses_unit_rms_norm(&[1.0, 0.0], 0.0, c, normalized[[0, c]], bound[[0, c]]),
                "coordinate {c}: the exact norm lies outside {} ± {}",
                normalized[[0, c]],
                bound[[0, c]]
            );
        }
        assert!(
            bound[[0, 0]] < 0.01,
            "the radius {} must come from the underflow-aware band, not the range bound √2 + |ŷ_1|",
            bound[[0, 0]]
        );
        let relative = accumulation_growth(2 + 6) * normalized[[0, 0]].abs();
        assert!(
            !encloses_unit_rms_norm(&[1.0, 0.0], 0.0, 0, normalized[[0, 0]], relative),
            "the relative-only band {relative} must miss the exact norm √2 of {}",
            normalized[[0, 0]]
        );
    }

    /// A zero-epsilon head row whose radius box reaches `x = 0` has no lower bound on its
    /// mean square, so its radius is the range bound `|w_c| √d + |ŷ_c|`: finite, and it
    /// encloses the exact norm at every sample of the box, of either sign.
    /// - Setup: `x̂ = (1, 0)` with radius `(2, 0)`, unit gains and `ε = 0`.
    #[test]
    fn query_key_norm_radius_of_a_box_that_reaches_zero_is_the_range_bound() {
        let rows = array![[1.0, 0.0]];
        let radius = array![[2.0, 0.0]];
        let gain = array![1.0, 1.0];
        let (normalized, bound) =
            head_rms_norm_with_radius(ProjectedRows { values: rows.view(), radius: radius.view() }, 2, 0.0, gain.view())
                .expect("finite head row");
        assert!(bound.iter().all(|reach| reach.is_finite()), "the radius {bound} must be finite");
        for x1 in [-1.0, -0.5, 0.5, 1.5, 3.0] {
            for c in 0..2 {
                assert!(
                    encloses_unit_rms_norm(&[x1, 0.0], 0.0, c, normalized[[0, c]], bound[[0, c]]),
                    "coordinate {c} at x_1 = {x1}: the exact norm lies outside {} ± {}",
                    normalized[[0, c]],
                    bound[[0, c]]
                );
            }
        }
    }

    /// Control: a normal-scale head row with zero radius keeps the relative band of its
    /// `d + 6` roundings, within `γ_(d+8) |ŷ_c|`, so the underflow terms and directed
    /// rounding do not inflate it.
    #[test]
    fn query_key_norm_radius_of_an_exact_normal_row_is_its_relative_band() {
        let rows = array![[0.125, 0.5]];
        let radius = array![[0.0, 0.0]];
        let gain = array![1.0, 1.0];
        let (normalized, bound) = head_rms_norm_with_radius(
            ProjectedRows { values: rows.view(), radius: radius.view() },
            2,
            0.015625,
            gain.view(),
        )
        .expect("finite head row");
        for c in 0..2 {
            let relative = accumulation_growth(2 + 8) * normalized[[0, c]].abs();
            assert!(
                bound[[0, c]] <= relative,
                "coordinate {c}: the radius {} exceeds the relative band {relative}",
                bound[[0, c]]
            );
        }
    }
}
