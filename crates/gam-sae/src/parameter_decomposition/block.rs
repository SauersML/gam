//! A whole pre-norm transformer block executed under component masks (#2951).
//!
//! # The block
//!
//! A decoder block moves the residual stream rows `h` through two sublayers,
//! each reading its own normalization of the current stream:
//!
//! ```text
//! Sequential (Qwen3, Llama):  h₁ = h + A(N₁(h)),   h' = h₁ + F(N₂(h₁))
//! Parallel (GPT-NeoX):        h' = (F(N₂(h)) + A(N₁(h))) + h
//! ```
//!
//! `A` is the source's causal self-attention through its output projection
//! ([`super::attention`]), `F(x) = W₂ σ(W₁ x + b₁) + b₂` is the MLP write
//! ([`super::rewrite`]), and `N₁`, `N₂` are the source's RMSNorm or LayerNorm
//! ([`MaskedNorm`]). The residual additions, in the source's association, are
//! [`decoder_layer`]'s. This module owns no primitive and no residual order,
//! only the composition.
//!
//! # Exact replay under masks
//!
//! [`ComponentBlock`] executes the query and key projections through masked
//! component coordinates (P6) and both MLP weights through masked exact factors
//! (P5). Each sublayer is its owner's identity: for every real mask
//! (continuous, binary or signed) it computes, on the input it is handed, the
//! sublayer with the edited tensors `U diag(m) R`. The block hands each sublayer
//! the current stream, after every upstream masked write, never a cached clean
//! one, and each normalization reads its whole current row. So the block under
//! any mask is the source block with the edited tensors: an identity, not a
//! linearization. In the sequential layout a query or key mask reaches the MLP
//! only through `N₂` and `σ` of the summed stream; in the parallel layout it does
//! not reach the MLP input at all. Every block-level mask reaches the logits
//! through at least the final normalization (mpd-gated, #2951 comment
//! 5716738723), so the exact affine-logit adversary (P9) does not apply to it.
//!
//! With every mask at one both sublayers run the original tensors on their
//! original paths, so the block is bit-identical to [`NativeBlock::execute`].
//!
//! # Stages
//!
//! [`BlockExecution`] and [`GatedBlockExecution`] keep each stage that a torch
//! block exports as a module output: the two normalized inputs, the attention
//! execution with its forward-error radius, the MLP's stages (the pre-activation,
//! activation and write of an activated MLP, or the `gate_proj`, `up_proj`,
//! hidden and `down_proj` rows of a gated one), and the layer output. A receipt compares an external executor with the native block
//! stage by stage (A12). The tests propagate the stage radii into a derived
//! replay band over the whole block.
//!
//! # Validity domain
//!
//! The block covers what its sublayers cover: query and key projections feeding
//! the rotary embedding with the source's projection biases, and normalizations
//! with the source's own gain and bias. A per-head query/key norm, when the source
//! attention carries one ([`NativeAttention::with_query_key_norm`]), runs inside
//! the attention sublayer on both paths, and [`super::attention`] certifies it; the
//! replay bands below exercise attention without it.
//! The MLP is either an activated residual MLP with a ReLU or exact GELU
//! activation ([`NativeBlock`], [`ComponentBlock`]) or a gated SwiGLU MLP with a
//! SiLU gate and no biases ([`NativeGatedBlock`], [`ComponentGatedBlock`]). The
//! two kinds are separate types, so a mask of one kind can never reach the
//! other. Masks on the value and output projections and on normalization gains
//! are not carried by either component block.
//!
//! # Attention-only layers
//!
//! An attention-only transformer's layer has no normalization, no MLP and no
//! bias: `h' = h + concat_h(z_h) W_Oᵀ`, with `z_h` the joint-softmax value read of
//! head `h` ([`NativeAttentionLayer`]). Its learned absolute positions enter the
//! embedding, so its rotary embedding is empty and the source's rotation leaves
//! the rows unchanged. Each of the four linear reads executes one of three ways
//! ([`ProjectionRead`]), all through the owner of matrix-free edits
//! ([`super::apply`]):
//!
//! * `Native`: the stored tensor on its original path.
//! * `Components(m)`: `U diag(m) R` over the component coordinates of the exact
//!   factor `W = U R` ([`ComponentAttentionLayer`]), with the native anchor at
//!   zero, so `U M R` is never formed. The mask is one for every row or one per row
//!   (a position-scoped control), and it may be a box of masks ([`ComponentMasks`]):
//!   the read executes the box's center and its radius covers every mask of the box.
//! * `Edited`: the stored tensor plus a factored edit `Δ` on exactly the rows whose
//!   absolute positions a [`PositionScope`] reaches. The other rows come from the
//!   same native product as the unedited layer, so they are its bits.
//!
//! Queries, keys and values go through the tensor-free attention core
//! ([`RotaryCausalAttention::attend_projected`]). Its head-mixed rows feed the
//! output read. Every mask on Q, K, V and O is a real mask on the summed read: the
//! softmax and the value mix see the masked rows, never a sum of per-component
//! patterns.
//!
//! # Radii
//!
//! The residual rows come with their own radius against the exact rows ([`ProjectedRows`];
//! [`ProjectedRows::exact`] for exact rows), and every stage of [`AttentionLayerExecution`]
//! carries a forward-error radius against the exact layer at the exact rows under the same
//! reads, so a receipt against another executor can use it as that side's band. Every read
//! takes one route: its band is its rounding band at the computed rows (`read_band`:
//! `γ_n |A| |x̂| + 𝒜 · 2^-1074` for apply.rs's kernel, with `n` the operations of the read's
//! product and `𝒜` its allowance for products that round into the subnormal range) plus
//! `|A| r`, the most the exact read can move between the computed rows and the exact ones,
//! with `r` the rows' radius. Exact rows carry nothing, so their band is the rounding band
//! itself. The query, key and value bands enter the attention core as its input radii, so
//! the core's score, pattern and mixed radii cover the exact attention of the exact reads
//! (first order in the core's trigonometric error). The output read takes the mixed rows with
//! the mixed radius, and the output adds the residual addition's rounding and the rows' own
//! radius. A stack of layers, each fed the previous output with its radius and ending in
//! [`linear_read`], so carries one radius from its input rows to its logits. Every magnitude
//! is computed one output column at a time, so no `|A|` is formed, and each computed bound
//! adds the underflow its own products could have lost and is divided by the factor its own
//! arithmetic could have lost, every operation rounded up.

use super::apply::{ApplyError, FactorView, apply_anchored_linear, native_linear};
use super::attention::{
    AttentionExecution, AttentionGeometry, AttentionProgramError, ComponentAttention,
    ComponentProjection, NativeAttention, ProjectedAttention, ProjectedRows, QueryKeyMasks,
    RotaryCausalAttention, RotaryEmbedding,
};
use super::gated_rewrite::{
    ComponentSwiglu, GatedRewriteError, MaskedNorm, NativeSwiglu, ResidualLayout, SwigluError,
    SwigluMask, SwigluStages, decoder_layer,
};
use super::occurrence::PositionScope;
use super::rewrite::{
    ComponentMlp, ComponentRead, ExactFactor, FactorRefusal, MlpMask, NativeMlp, RewriteError,
};
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use gam_runtime::resource::Governed;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
use std::fmt;

/// A source model's input normalization on its own tensors.
#[derive(Clone, Debug, PartialEq)]
pub enum NativeNorm {
    /// `Qwen3RMSNorm`: `w ⊙ h (mean(h²) + ε)^(-1/2)`.
    Rms { epsilon: f64, gain: Array1<f64> },
    /// `torch.nn.LayerNorm`: the RMSNorm of the centred row, plus `β`.
    Layer {
        epsilon: f64,
        gain: Array1<f64>,
        bias: Array1<f64>,
    },
}

impl NativeNorm {
    /// The normalization as its owner's [`MaskedNorm`], with the source's own
    /// gain and bias as the edited tensors.
    pub fn as_masked_norm(&self) -> MaskedNorm<'_> {
        match self {
            Self::Rms { epsilon, gain } => MaskedNorm::Rms {
                epsilon: *epsilon,
                gain: gain.view(),
            },
            Self::Layer {
                epsilon,
                gain,
                bias,
            } => MaskedNorm::Layer {
                epsilon: *epsilon,
                gain: gain.view(),
                bias: bias.view(),
            },
        }
    }

    /// The normalized rows, as [`MaskedNorm::apply`] computes them, and their rounding
    /// band against the exact normalization of the same rows: `|fl(N(h)) − N(h)|`
    /// entrywise, for a receipt that compares this stage at its own input.
    ///
    /// - **RMSNorm.** [`rms_norm_band`].
    /// - **LayerNorm.** The centred row `ĉ = fl(h − fl(mean h))` carries an absolute
    ///   error `e_j ≤ γ_d mean|h| + 2^-1074 + u |ĉ_j| / (1 − u)` (the `2^-1074` covers the
    ///   mean's division rounding into the subnormal range, [`SUBNORMAL_SPACING`]), which no
    ///   relative bound covers near a constant row. It passes through the normalization, whose
    ///   Jacobian has spectral norm `ρ(x) = (‖x‖²/d + ε)^(-1/2)`, so it adds `|w_i| sup ρ ‖e‖₂`
    ///   by the mean value inequality. On the segment `sup ρ ≤ ((‖ĉ‖/2)²/d + ε)^(-1/2)` once
    ///   `4 ‖e‖₂ ≤ ‖ĉ‖`, and `sup ρ ≤ ε^(-1/2)` always. `‖e‖₂` is scaled by its largest entry,
    ///   so no square of an error underflows. The row's own rounding at `ĉ` is
    ///   [`rms_norm_band`]'s at `x = ĉ` with `γ_(d+7)` for the bias addition:
    ///   `λ (|w_i ĉ_i ν̂| + |b_i| + a_i) / (1 − λ) + a_i`.
    ///
    /// A LayerNorm band is divided by `1 − γ_(2d+23)` for its own arithmetic: its slope and
    /// error norm each take a sum of `d` squares and a square root before their product, and
    /// the scaled error norm adds a division and a product on its path and scaled squares
    /// whose underflow stays below one more rounding. A row with `ε = 0` whose error is not
    /// below a quarter of its norm, or whose mean square is not resolved above binary64's
    /// underflow ([`rms_norm_band`]), has no bound and is refused.
    pub fn apply_with_band(&self, rows: ArrayView2<'_, f64>) -> Result<(Array2<f64>, Array2<f64>), BlockError> {
        let normalized = self.as_masked_norm().apply(rows)?;
        let width = rows.ncols();
        let band = match self {
            Self::Rms { epsilon, gain } => rms_norm_band(*epsilon, gain.view(), rows, normalized.view())?,
            Self::Layer { epsilon, gain, bias } => {
                let mean_growth = accumulation_growth(width);
                let dominance = 1.0 - accumulation_growth(2 * width + 23);
                let mut band = Array2::<f64>::zeros(rows.raw_dim());
                for (row, (values, mut band_row)) in rows.rows().into_iter().zip(band.rows_mut()).enumerate() {
                    let mean = values.sum() / width as f64;
                    let centred = values.mapv(|value| value - mean);
                    let mean_abs = values.iter().map(|value| value.abs()).sum::<f64>() / width as f64;
                    let errors = centred.mapv(|value| {
                        mean_growth * mean_abs + SUBNORMAL_SPACING + UNIT_ROUNDOFF * value.abs() / (1.0 - UNIT_ROUNDOFF)
                    });
                    let error_norm = scaled_norm(errors.view());
                    let centred_norm = centred.iter().map(|value| value * value).sum::<f64>().sqrt();
                    let slope = if 4.0 * error_norm <= centred_norm {
                        let half = centred_norm / 2.0;
                        (half * half / width as f64 + epsilon).sqrt().recip()
                    } else if *epsilon > 0.0 {
                        epsilon.sqrt().recip()
                    } else {
                        return Err(BlockError::NormBandUnbounded { row });
                    };
                    let growth = normalization_growth(centred.view(), *epsilon, width + 7)
                        .ok_or(BlockError::NormBandUnbounded { row })?;
                    let inverse_root = (centred.iter().map(|value| value * value).sum::<f64>() / width as f64
                        + epsilon)
                        .sqrt()
                        .recip();
                    for (((slot, &weight), &offset), &value) in
                        band_row.iter_mut().zip(gain.iter()).zip(bias.iter()).zip(centred.iter())
                    {
                        let absolute = underflow_reach(weight);
                        let magnitude = up(up((weight * (value * inverse_root)).abs() + offset.abs()) + absolute);
                        let own = up(up(up(growth * magnitude) / down(1.0 - growth)) + absolute);
                        *slot = (own + weight.abs() * slope * error_norm) / dominance;
                    }
                }
                band
            }
        };
        Ok((normalized, band))
    }
}

/// binary64's subnormal spacing `2^-1074`. A product or quotient whose result rounds into
/// the subnormal range moves by an absolute amount of at most half of it, not relatively;
/// additions and subtractions are exact there. Half the spacing, `2^-1075`, is not a
/// binary64 number (`f64::MIN_POSITIVE * UNIT_ROUNDOFF` rounds to zero), so a band carries
/// the whole spacing.
pub const SUBNORMAL_SPACING: f64 = f64::MIN_POSITIVE * f64::EPSILON;

/// `value` stepped one float up: an upper bound on the exact result of an operation that
/// rounded to nearest to `value`.
pub(super) fn up(value: f64) -> f64 {
    value.next_up()
}

/// `value` stepped one float down, clamped at zero: a lower bound on a nonnegative exact
/// result that rounded to nearest to `value`.
pub(super) fn down(value: f64) -> f64 {
    value.next_down().max(0.0)
}

/// An upper bound on the absolute underflow of one normalized entry `fl(w fl(x ν̂))` beyond
/// its relative rounding: `|w| 2^-1075 (1 + u)` from `x ν̂` and `2^-1075` from the gain's
/// product, carried as `|w| 2^-1074 + 2^-1074`, which also covers a following bias
/// addition's `1 + u`.
fn underflow_reach(weight: f64) -> f64 {
    up(up(weight.abs() * SUBNORMAL_SPACING) + SUBNORMAL_SPACING)
}

/// The relative growth `λ` of one normalized entry `fl(w fl(x ν̂))` of the row `values`
/// (`x`, `d` wide) against its exact value `w x ν` beyond [`underflow_reach`], with
/// `operations = k` relative roundings on the entry's path, or `None` when no finite `λ < 1`
/// bounds it.
///
/// The computed argument of the square root is `X (1 + θ) + A`, with `X = mean(x²) + ε`,
/// `|θ| ≤ γ_(d+2)` and `|A| ≤ 2^-1074 (1 + γ_(d+1))`: each square and the mean's division
/// round by at most `2^-1075` absolute into the subnormal range, while the sum and `+ ε`
/// stay relative. With `X ≥ X₀ = max(ε, max_j x_j² / d)`, the absolute part is a relative
/// `α`, `|α| ≤ ω = 2 · 2^-1074 / (X₀ (1 − γ_k))`, which moves `ν̂` by `(1 + α)^(-1/2)`, within
/// `ρ = ω / (2 (1 − ω))` of one for `ω < 1`: `(1 − ω)^(-1/2) − 1 = ω / (√(1 − ω) (1 + √(1 − ω)))`
/// and `1 − ω ≤ √(1 − ω)`. So `λ = (1 + γ_k)(1 + ρ) − 1 = γ_k + ρ + γ_k ρ`. A row with `ε = 0`
/// whose mean square is not resolved above the underflow (`ω ≥ 1`) has no bound. Every
/// operation rounds to nearest and then steps one float up (down where it divides), so `λ`
/// is an upper bound computed in binary64.
fn normalization_growth(values: ArrayView1<'_, f64>, epsilon: f64, operations: usize) -> Option<f64> {
    let largest = values.iter().fold(0.0_f64, |largest, value| largest.max(value.abs()));
    let floor = epsilon.max(down(down(largest * largest) / values.len() as f64));
    let growth = up(accumulation_growth(operations));
    let omega = up(2.0 * SUBNORMAL_SPACING / down(floor * down(1.0 - growth)));
    if !(omega < 1.0) {
        return None;
    }
    let rho = up(omega / down(2.0 * down(1.0 - omega)));
    let lambda = up(up(growth + rho) + up(growth * rho));
    (lambda < 1.0).then_some(lambda)
}

/// `‖e‖₂` of the nonnegative `errors`, scaled by the largest entry so that no square
/// underflows.
fn scaled_norm(errors: ArrayView1<'_, f64>) -> f64 {
    let largest = errors.iter().fold(0.0_f64, |largest, &error| largest.max(error));
    if largest == 0.0 {
        return 0.0;
    }
    largest
        * errors
            .iter()
            .map(|&error| {
                let scaled = error / largest;
                scaled * scaled
            })
            .sum::<f64>()
            .sqrt()
}

/// The rounding band of one binary64 evaluation of an RMSNorm, `MaskedNorm::Rms`'s program
/// `fl(w_j fl(x_j ν̂))`, against the exact RMSNorm `y = w x (mean(x²) + ε)^(-1/2)` of the input
/// rows `inputs` (`d` wide, with the gain `gain`), from its computed rows `normalized`.
///
/// The squares, `d − 1` additions, the mean, `+ ε`, the square root, the reciprocal and the
/// products with the row and the gain are `γ_(d+6)` relative while their results stay
/// normal. A square, the mean or a product whose result is subnormal rounds by an absolute
/// `2^-1075` instead: through `ν̂` that is the relative `ρ` of [`normalization_growth`], and on
/// the entry it is the absolute `a_j = |w_j| 2^-1074 + 2^-1074` of [`underflow_reach`]. So
/// `|ŷ − y| ≤ λ |y| + a` with `λ = (1 + γ_(d+6))(1 + ρ) − 1`, and `|y| ≤ (|ŷ| + a)/(1 − λ)`,
/// so the band is `λ (|ŷ| + a)/(1 − λ) + a`. Every operation rounds to nearest and then steps
/// one float up (down where it divides), so the band is an upper bound computed in binary64.
///
/// `inputs` and `normalized` share a shape and `gain` is as wide; a row with `ε = 0` whose
/// mean square is not resolved above binary64's underflow has no band and is refused.
pub fn rms_norm_band(
    epsilon: f64,
    gain: ArrayView1<'_, f64>,
    inputs: ArrayView2<'_, f64>,
    normalized: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, NormBandUnbounded> {
    let width = inputs.ncols();
    let mut band = Array2::<f64>::zeros(normalized.raw_dim());
    for (row, (values, (computed, mut band_row))) in inputs
        .rows()
        .into_iter()
        .zip(normalized.rows().into_iter().zip(band.rows_mut()))
        .enumerate()
    {
        let growth = normalization_growth(values, epsilon, width + 6).ok_or(NormBandUnbounded { row })?;
        let complement = down(1.0 - growth);
        Zip::from(&mut band_row)
            .and(gain)
            .and(computed)
            .for_each(|slot, &weight, &value| {
                let absolute = underflow_reach(weight);
                *slot = up(up(up(growth * up(value.abs() + absolute)) / complement) + absolute);
            });
    }
    Ok(band)
}

/// A normalization row whose rounding band has no finite bound: a row with `ε = 0` whose mean
/// square is not resolved above binary64's underflow ([`rms_norm_band`]), or a LayerNorm row
/// with `ε = 0` whose centring error is not below a quarter of its norm
/// ([`NativeNorm::apply_with_band`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NormBandUnbounded {
    pub row: usize,
}

impl From<NormBandUnbounded> for BlockError {
    fn from(NormBandUnbounded { row }: NormBandUnbounded) -> Self {
        Self::NormBandUnbounded { row }
    }
}

impl fmt::Display for NormBandUnbounded {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "normalization row {} with zero epsilon has no finite rounding band: its mean square is not \
             resolved above binary64's underflow, or its centring error is too large for a slope bound",
            self.row
        )
    }
}

/// One of a decoder block's two sublayers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sublayer {
    Attention,
    Mlp,
}

impl fmt::Display for Sublayer {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Attention => "attention",
            Self::Mlp => "MLP",
        })
    }
}

/// A refused block execution.
#[derive(Debug)]
pub enum BlockError {
    /// A sublayer's input normalization refused the current stream.
    Norm {
        sublayer: Sublayer,
        error: GatedRewriteError,
    },
    /// The decoder layer refused a sublayer's write.
    Layer(GatedRewriteError),
    Attention(AttentionProgramError),
    Mlp(RewriteError),
    /// The gated MLP sublayer refused.
    Swiglu(SwigluError),
    /// The decoder layer returned a stream without running this sublayer: an
    /// executor invariant.
    SublayerNotExecuted { sublayer: Sublayer },
    /// An attention-only layer's residual rows do not match its positions and its
    /// model width.
    ResidualShape {
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// A projection weight whose shape is not the layer geometry's.
    ProjectionShape {
        projection: AttentionProjection,
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// A projection weight has no exact factor through its declared read.
    Factor {
        projection: AttentionProjection,
        refusal: FactorRefusal,
    },
    /// A matrix-free read refused its operands or its memory footprint.
    Apply {
        projection: AttentionProjection,
        error: ApplyError,
    },
    /// A component read on a layer that holds no component factors.
    NoComponentFactors { projection: AttentionProjection },
    /// An edited read at a negative absolute position, which no position scope can
    /// name.
    NegativePosition {
        projection: AttentionProjection,
        position: i64,
    },
    /// An edit declares a position that no row of the layer holds.
    EditPositionAbsent {
        projection: AttentionProjection,
        position: usize,
    },
    /// A normalization row with `ε = 0` whose rounding band has no finite bound
    /// ([`NormBandUnbounded`]).
    NormBandUnbounded { row: usize },
}

impl From<SwigluError> for BlockError {
    fn from(error: SwigluError) -> Self {
        Self::Swiglu(error)
    }
}

impl From<GatedRewriteError> for BlockError {
    fn from(error: GatedRewriteError) -> Self {
        Self::Layer(error)
    }
}

impl From<AttentionProgramError> for BlockError {
    fn from(error: AttentionProgramError) -> Self {
        Self::Attention(error)
    }
}

impl From<RewriteError> for BlockError {
    fn from(error: RewriteError) -> Self {
        Self::Mlp(error)
    }
}

impl fmt::Display for BlockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Norm { sublayer, error } => {
                write!(formatter, "the {sublayer} sublayer's normalization refused: {error}")
            }
            Self::Layer(error) => write!(formatter, "the decoder layer refused a write: {error}"),
            Self::Attention(error) => write!(formatter, "the attention sublayer refused: {error}"),
            Self::Mlp(error) => write!(formatter, "the MLP sublayer refused: {error}"),
            Self::Swiglu(error) => write!(formatter, "the gated MLP sublayer refused: {error}"),
            Self::SublayerNotExecuted { sublayer } => write!(
                formatter,
                "executor invariant: the decoder layer returned without running the {sublayer} sublayer"
            ),
            Self::ResidualShape { expected, found } => write!(
                formatter,
                "the attention-only layer's residual rows have shape {found:?}, its positions and width need {expected:?}"
            ),
            Self::ProjectionShape {
                projection,
                expected,
                found,
            } => write!(
                formatter,
                "the {projection} weight has shape {found:?}, the layer geometry needs {expected:?}"
            ),
            Self::Factor { projection, refusal } => {
                write!(formatter, "the {projection} weight has no exact factor: {refusal}")
            }
            Self::Apply { projection, error } => {
                write!(formatter, "the {projection} read refused: {error}")
            }
            Self::NoComponentFactors { projection } => write!(
                formatter,
                "a component read of the {projection} projection needs a layer with component factors"
            ),
            Self::NegativePosition {
                projection,
                position,
            } => write!(
                formatter,
                "an edited {projection} read at negative position {position}, which no position scope names"
            ),
            Self::EditPositionAbsent {
                projection,
                position,
            } => write!(
                formatter,
                "the {projection} edit declares position {position}, which no row of the layer holds"
            ),
            Self::NormBandUnbounded { row } => write!(formatter, "{}", NormBandUnbounded { row: *row }),
        }
    }
}

impl std::error::Error for BlockError {}

/// Every stage of one executed block. Rows are positions throughout. It holds its
/// attention sublayer's memory reservation, so it is not `Clone`.
#[derive(Debug)]
pub struct BlockExecution {
    /// `N₁(h)`, the attention sublayer's input.
    pub attention_input: Array2<f64>,
    /// The attention sublayer on [`Self::attention_input`]. Its output, after the
    /// output projection, is the sublayer's write, with its forward-error radius.
    pub attention: AttentionExecution,
    /// `N₂` of the stream the MLP reads: `h + A` in the sequential layout, `h` in
    /// the parallel one.
    pub mlp_input: Array2<f64>,
    /// `W₁ x + b₁` on [`Self::mlp_input`], under the read-in mask.
    pub mlp_summed_input: Array2<f64>,
    /// `σ` of [`Self::mlp_summed_input`].
    pub mlp_activations: Array2<f64>,
    /// `W₂ a + b₂` on [`Self::mlp_activations`], under the write-out mask.
    pub mlp_write: Array2<f64>,
    /// The residual stream after both writes.
    pub output: Array2<f64>,
}

/// The MLP sublayer's stages on its normalized input.
struct MlpStage {
    summed_input: Array2<f64>,
    activations: Array2<f64>,
    write: Array2<f64>,
}

/// Every stage of one executed decoder layer, with the MLP sublayer's own stages.
struct LayerStages<Stages> {
    attention_input: Array2<f64>,
    attention: AttentionExecution,
    mlp_input: Array2<f64>,
    mlp: Stages,
    output: Array2<f64>,
}

/// The decoder layer over `residual`, with each sublayer normalizing the stream
/// the layer hands it, and every stage kept. The MLP closure returns its stages
/// and its write.
fn execute_layer<Attend, Mlp, Stages>(
    layout: ResidualLayout,
    attention_norm: &NativeNorm,
    mlp_norm: &NativeNorm,
    residual: ArrayView2<'_, f64>,
    attend: Attend,
    mlp: Mlp,
) -> Result<LayerStages<Stages>, BlockError>
where
    Attend: FnOnce(ArrayView2<'_, f64>) -> Result<AttentionExecution, BlockError>,
    Mlp: FnOnce(ArrayView2<'_, f64>) -> Result<(Stages, Array2<f64>), BlockError>,
{
    let mut attention_stage = None;
    let mut mlp_stage = None;
    let output = decoder_layer::<BlockError, _, _>(
        layout,
        residual,
        |stream| {
            let input = attention_norm
                .as_masked_norm()
                .apply(stream)
                .map_err(|error| BlockError::Norm {
                    sublayer: Sublayer::Attention,
                    error,
                })?;
            let execution = attend(input.view())?;
            let write = execution.output.clone();
            attention_stage = Some((input, execution));
            Ok(write)
        },
        |stream| {
            let input = mlp_norm
                .as_masked_norm()
                .apply(stream)
                .map_err(|error| BlockError::Norm {
                    sublayer: Sublayer::Mlp,
                    error,
                })?;
            let (stages, write) = mlp(input.view())?;
            mlp_stage = Some((input, stages));
            Ok(write)
        },
    )?;
    let (attention_input, attention) = attention_stage.ok_or(BlockError::SublayerNotExecuted {
        sublayer: Sublayer::Attention,
    })?;
    let (mlp_input, mlp) = mlp_stage.ok_or(BlockError::SublayerNotExecuted {
        sublayer: Sublayer::Mlp,
    })?;
    Ok(LayerStages {
        attention_input,
        attention,
        mlp_input,
        mlp,
        output,
    })
}

/// The execution record of a block whose MLP is an activated residual MLP.
fn activated_execution(layer: LayerStages<MlpStage>) -> BlockExecution {
    BlockExecution {
        attention_input: layer.attention_input,
        attention: layer.attention,
        mlp_input: layer.mlp_input,
        mlp_summed_input: layer.mlp.summed_input,
        mlp_activations: layer.mlp.activations,
        mlp_write: layer.mlp.write,
        output: layer.output,
    }
}

/// A decoder block on its original tensors.
#[derive(Clone, Debug)]
pub struct NativeBlock {
    layout: ResidualLayout,
    attention_norm: NativeNorm,
    attention: NativeAttention,
    mlp_norm: NativeNorm,
    mlp: NativeMlp,
}

impl NativeBlock {
    /// A block of the source's sublayers. Each width is refused where its owner
    /// reads it: the normalizations at their gain, the attention at its input, the
    /// MLP at its read-in, and the layer at each write.
    pub fn new(
        layout: ResidualLayout,
        attention_norm: NativeNorm,
        attention: NativeAttention,
        mlp_norm: NativeNorm,
        mlp: NativeMlp,
    ) -> Self {
        Self {
            layout,
            attention_norm,
            attention,
            mlp_norm,
            mlp,
        }
    }

    pub fn layout(&self) -> ResidualLayout {
        self.layout
    }

    /// The source block on its original tensors, over the residual rows at their
    /// absolute positions.
    pub fn execute(
        &self,
        residual: ArrayView2<'_, f64>,
        positions: &[i64],
    ) -> Result<BlockExecution, BlockError> {
        let layer = execute_layer(
            self.layout,
            &self.attention_norm,
            &self.mlp_norm,
            residual,
            |input| Ok(self.attention.execute(input, positions)?),
            |input| {
                let summed_input = self.mlp.summed_input(input).map_err(RewriteError::from)?;
                let activations = self
                    .mlp
                    .activate(summed_input.view())
                    .map_err(RewriteError::from)?;
                let write = self
                    .mlp
                    .written(activations.view())
                    .map_err(RewriteError::from)?;
                let stage_write = write.clone();
                Ok((
                    MlpStage {
                        summed_input,
                        activations,
                        write,
                    },
                    stage_write,
                ))
            },
        )?;
        Ok(activated_execution(layer))
    }
}

/// The masks of a [`ComponentBlock`]: the query and key component masks of its
/// attention and the factor masks of its MLP.
#[derive(Clone, Copy, Debug)]
pub struct BlockMasks<'a> {
    pub attention: &'a QueryKeyMasks,
    pub mlp: MlpMask<'a>,
}

/// A decoder block with its query and key projections and both MLP weights
/// executed through masked component coordinates (see the module documentation).
#[derive(Clone, Debug)]
pub struct ComponentBlock {
    layout: ResidualLayout,
    attention_norm: NativeNorm,
    attention: ComponentAttention,
    mlp_norm: NativeNorm,
    mlp: ComponentMlp,
}

impl ComponentBlock {
    /// Factors `native`'s sublayers: `query` and `key` must factor the attention's
    /// query and key projections exactly (as [`ComponentAttention::new`] requires),
    /// and both MLP weights are solved for their exact writes through `read_in` and
    /// `write_out` ([`ComponentMlp::new`]).
    pub fn new(
        native: NativeBlock,
        query: ComponentProjection,
        key: ComponentProjection,
        read_in: ComponentRead<'_>,
        write_out: ComponentRead<'_>,
    ) -> Result<Self, BlockError> {
        let NativeBlock {
            layout,
            attention_norm,
            attention,
            mlp_norm,
            mlp,
        } = native;
        Ok(Self {
            layout,
            attention_norm,
            attention: ComponentAttention::new(attention, query, key)?,
            mlp_norm,
            mlp: ComponentMlp::new(mlp, read_in, write_out)?,
        })
    }

    pub fn layout(&self) -> ResidualLayout {
        self.layout
    }

    /// The MLP sublayer and its exact factors.
    pub fn mlp(&self) -> &ComponentMlp {
        &self.mlp
    }

    /// The block under `masks`, over the residual rows at their absolute
    /// positions. All-on masks execute the original tensors on their original
    /// paths.
    pub fn execute(
        &self,
        masks: BlockMasks<'_>,
        residual: ArrayView2<'_, f64>,
        positions: &[i64],
    ) -> Result<BlockExecution, BlockError> {
        let layer = execute_layer(
            self.layout,
            &self.attention_norm,
            &self.mlp_norm,
            residual,
            |input| Ok(self.attention.execute(masks.attention, input, positions)?),
            |input| {
                let summed_input = self
                    .mlp
                    .summed_input(input, masks.mlp.read_in)
                    .map_err(RewriteError::from)?;
                let activations = self
                    .mlp
                    .native()
                    .activate(summed_input.view())
                    .map_err(RewriteError::from)?;
                let write = self
                    .mlp
                    .written(activations.view(), masks.mlp.write_out)
                    .map_err(RewriteError::from)?;
                let stage_write = write.clone();
                Ok((
                    MlpStage {
                        summed_input,
                        activations,
                        write,
                    },
                    stage_write,
                ))
            },
        )?;
        Ok(activated_execution(layer))
    }
}

/// Every stage of one executed gated block. Rows are positions throughout. It holds
/// its attention sublayer's memory reservation, so it is not `Clone`.
#[derive(Debug)]
pub struct GatedBlockExecution {
    /// `N₁(h)`, the attention sublayer's input.
    pub attention_input: Array2<f64>,
    /// The attention sublayer on [`Self::attention_input`], with its radius.
    pub attention: AttentionExecution,
    /// `N₂` of the stream the gated MLP reads.
    pub mlp_input: Array2<f64>,
    /// The gated MLP's `gate_proj`, `up_proj`, hidden `s(gate) ⊙ up` and
    /// `down_proj` rows on [`Self::mlp_input`]; its write carries no residual.
    pub mlp: SwigluStages,
    /// The residual stream after both writes.
    pub output: Array2<f64>,
}

/// The execution record of a block whose MLP is a gated (SwiGLU) MLP.
fn gated_execution(layer: LayerStages<SwigluStages>) -> GatedBlockExecution {
    GatedBlockExecution {
        attention_input: layer.attention_input,
        attention: layer.attention,
        mlp_input: layer.mlp_input,
        mlp: layer.mlp,
        output: layer.output,
    }
}

/// A decoder block with a gated (SwiGLU) MLP on its original tensors: Qwen3 or
/// Llama, with a SiLU gate and no MLP biases.
#[derive(Clone, Debug)]
pub struct NativeGatedBlock {
    layout: ResidualLayout,
    attention_norm: NativeNorm,
    attention: NativeAttention,
    mlp_norm: NativeNorm,
    mlp: NativeSwiglu,
}

impl NativeGatedBlock {
    /// The source's attention normalization (`input_layernorm`).
    pub fn attention_norm(&self) -> &NativeNorm {
        &self.attention_norm
    }

    /// The source's attention sublayer on its own tensors.
    pub fn attention(&self) -> &NativeAttention {
        &self.attention
    }

    /// The source's MLP normalization (`post_attention_layernorm`).
    pub fn mlp_norm(&self) -> &NativeNorm {
        &self.mlp_norm
    }

    /// The source's gated MLP on its own tensors.
    pub fn mlp(&self) -> &NativeSwiglu {
        &self.mlp
    }

    /// A block of the source's sublayers. Each width is refused where its owner
    /// reads it.
    pub fn new(
        layout: ResidualLayout,
        attention_norm: NativeNorm,
        attention: NativeAttention,
        mlp_norm: NativeNorm,
        mlp: NativeSwiglu,
    ) -> Self {
        Self {
            layout,
            attention_norm,
            attention,
            mlp_norm,
            mlp,
        }
    }

    pub fn layout(&self) -> ResidualLayout {
        self.layout
    }

    /// The source block on its original tensors, over the residual rows at their
    /// absolute positions.
    pub fn execute(
        &self,
        residual: ArrayView2<'_, f64>,
        positions: &[i64],
    ) -> Result<GatedBlockExecution, BlockError> {
        let layer = execute_layer(
            self.layout,
            &self.attention_norm,
            &self.mlp_norm,
            residual,
            |input| Ok(self.attention.execute(input, positions)?),
            |input| {
                let stages = self.mlp.execute_stages(input).map_err(SwigluError::from)?;
                let write = stages.write.clone();
                Ok((stages, write))
            },
        )?;
        Ok(gated_execution(layer))
    }
}

/// The masks of a [`ComponentGatedBlock`]: the query and key component masks of
/// its attention and the factor masks of its gated MLP.
#[derive(Clone, Copy, Debug)]
pub struct GatedBlockMasks<'a> {
    pub attention: &'a QueryKeyMasks,
    pub mlp: SwigluMask<'a>,
}

/// A decoder block with its query and key projections and the three gated-MLP
/// weights executed through masked component coordinates.
#[derive(Clone, Debug)]
pub struct ComponentGatedBlock {
    layout: ResidualLayout,
    attention_norm: NativeNorm,
    attention: ComponentAttention,
    mlp_norm: NativeNorm,
    mlp: ComponentSwiglu,
}

impl ComponentGatedBlock {
    /// Factors `native`'s sublayers: `query` and `key` must factor the attention's
    /// query and key projections exactly, and the gate, up and down weights are
    /// solved for their exact writes ([`ComponentSwiglu::new`]).
    pub fn new(
        native: NativeGatedBlock,
        query: ComponentProjection,
        key: ComponentProjection,
        gate: ComponentRead<'_>,
        up: ComponentRead<'_>,
        down: ComponentRead<'_>,
    ) -> Result<Self, BlockError> {
        let NativeGatedBlock {
            layout,
            attention_norm,
            attention,
            mlp_norm,
            mlp,
        } = native;
        Ok(Self {
            layout,
            attention_norm,
            attention: ComponentAttention::new(attention, query, key)?,
            mlp_norm,
            mlp: ComponentSwiglu::new(mlp, gate, up, down)?,
        })
    }

    pub fn layout(&self) -> ResidualLayout {
        self.layout
    }

    /// The gated MLP sublayer and its exact factors.
    pub fn mlp(&self) -> &ComponentSwiglu {
        &self.mlp
    }

    /// The block under `masks`, over the residual rows at their absolute
    /// positions. All-on masks execute the original tensors on their original
    /// paths.
    pub fn execute(
        &self,
        masks: GatedBlockMasks<'_>,
        residual: ArrayView2<'_, f64>,
        positions: &[i64],
    ) -> Result<GatedBlockExecution, BlockError> {
        let layer = execute_layer(
            self.layout,
            &self.attention_norm,
            &self.mlp_norm,
            residual,
            |input| Ok(self.attention.execute(masks.attention, input, positions)?),
            |input| {
                let stages = self.mlp.execute_stages(input, masks.mlp)?;
                let write = stages.write.clone();
                Ok((stages, write))
            },
        )?;
        Ok(gated_execution(layer))
    }
}

/// One of an attention-only layer's four linear reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttentionProjection {
    Query,
    Key,
    Value,
    Output,
}

impl AttentionProjection {
    fn index(self) -> usize {
        match self {
            Self::Query => 0,
            Self::Key => 1,
            Self::Value => 2,
            Self::Output => 3,
        }
    }
}

impl fmt::Display for AttentionProjection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Query => "query",
            Self::Key => "key",
            Self::Value => "value",
            Self::Output => "output",
        })
    }
}

/// A parameter edit read at declared positions: `Θ + Δ` on every row whose
/// absolute position `positions` reaches, and the stored tensor on every other row.
#[derive(Clone, Copy, Debug)]
pub struct ScopedEdit<'a> {
    /// `Δ = Σ_k u_k v_kᵀ` in the read's orientation (`occurrence`'s
    /// `ParameterEditRecord::delta_read_at`).
    pub edit: FactorView<'a>,
    pub positions: &'a PositionScope,
}

/// The component controls of one read: every mask `m` with `|m − center| ≤ half_width`
/// entrywise. `center` is `1 × C`, one mask for every row, or `rows × C`, one mask per row
/// (a position-scoped control); `half_width` has `center`'s shape and is read as a
/// magnitude. `None` is the exact mask `center`, a box of width zero.
///
/// A read under a box executes its center, `U diag(c) R x̂`. Its radius adds
/// `|U| diag|w| |R| (|x̂| + r)` to the center's band, which bounds `|U diag(m − c) R x*|` for
/// every mask in the box and exact rows `x*` within `r` of `x̂`. So one execution encloses the
/// read, and every stage after it, at every mask of the box.
#[derive(Clone, Copy, Debug)]
pub struct ComponentMasks<'a> {
    pub center: ArrayView2<'a, f64>,
    pub half_width: Option<ArrayView2<'a, f64>>,
}

impl<'a> ComponentMasks<'a> {
    /// One exact mask for every row.
    pub fn uniform(mask: ArrayView1<'a, f64>) -> Self {
        Self {
            center: mask.insert_axis(Axis(0)),
            half_width: None,
        }
    }
}

/// How one linear read of an attention-only layer executes (see the module
/// documentation).
#[derive(Clone, Copy, Debug)]
pub enum ProjectionRead<'a> {
    /// The stored tensor on its original path.
    Native,
    /// `U diag(m) R` over the component coordinates of a [`ComponentAttentionLayer`], for
    /// every mask `m` of a [`ComponentMasks`] box. Entries are any real numbers:
    /// continuous, binary or signed.
    Components(ComponentMasks<'a>),
    /// The stored tensor plus a factored edit at declared positions.
    Edited(ScopedEdit<'a>),
}

/// The reads of an attention-only layer's four projections.
#[derive(Clone, Copy, Debug)]
pub struct AttentionLayerReads<'a> {
    pub query: ProjectionRead<'a>,
    pub key: ProjectionRead<'a>,
    pub value: ProjectionRead<'a>,
    pub output: ProjectionRead<'a>,
}

impl AttentionLayerReads<'_> {
    /// Every projection on its stored tensor.
    pub fn native() -> Self {
        Self {
            query: ProjectionRead::Native,
            key: ProjectionRead::Native,
            value: ProjectionRead::Native,
            output: ProjectionRead::Native,
        }
    }
}

/// Every stage of one executed attention-only layer, each with its forward-error
/// radius against the exact layer at the exact residual rows under the same reads.
/// Rows are positions.
#[derive(Debug)]
pub struct AttentionLayerExecution {
    /// `x W_Qᵀ` under the query read, `tokens × n_heads·head_dim`.
    pub queries: Governed<Array2<f64>>,
    /// The query read's rounding band ([`NativeAttentionLayer::read_band`]) plus `|A_Q| r`
    /// for the residual rows' radius `r` (nothing for exact rows).
    pub query_radius: Array2<f64>,
    /// `x W_Kᵀ` under the key read, `tokens × n_kv_heads·head_dim`.
    pub keys: Governed<Array2<f64>>,
    pub key_radius: Array2<f64>,
    /// `x W_Vᵀ` under the value read, `tokens × n_kv_heads·head_dim`.
    pub values: Governed<Array2<f64>>,
    pub value_radius: Array2<f64>,
    /// The scores, the attention pattern and the head-mixed rows `concat_h z_h`. The
    /// three reads' radii enter the owner's attention as its input radii, so each
    /// radius here is against the exact attention of the exact reads.
    pub attention: ProjectedAttention,
    /// `concat_h(z_h) W_Oᵀ` under the output read: the layer's write, no residual.
    pub write: Governed<Array2<f64>>,
    /// The output read's rounding band plus `|W_O|` (under the read) times the mixed
    /// radius.
    pub write_radius: Array2<f64>,
    /// `h + write`, the residual stream after the layer.
    pub output: Array2<f64>,
    /// The write radius plus the residual addition's rounding and the residual rows' radius.
    pub output_radius: Array2<f64>,
}

/// A norm-free, MLP-free, bias-free decoder layer `h' = h + concat_h(z_h) W_Oᵀ` on
/// its original tensors.
#[derive(Clone, Debug)]
pub struct NativeAttentionLayer {
    geometry: AttentionGeometry,
    rotary: RotaryEmbedding,
    score_scale: f64,
    attention: RotaryCausalAttention,
    query: Array2<f64>,
    key: Array2<f64>,
    value: Array2<f64>,
    output: Array2<f64>,
}

impl NativeAttentionLayer {
    /// The layer's attention core and its four weights in torch `Linear` layout:
    /// `query` is `n_heads·head_dim × model_dim`, `key` and `value` are
    /// `n_kv_heads·head_dim × model_dim`, `output` is `model_dim × n_heads·head_dim`.
    /// A source with learned absolute positions passes an empty rotary embedding.
    pub fn new(
        geometry: AttentionGeometry,
        rotary: RotaryEmbedding,
        score_scale: f64,
        query: Array2<f64>,
        key: Array2<f64>,
        value: Array2<f64>,
        output: Array2<f64>,
    ) -> Result<Self, BlockError> {
        let attention = RotaryCausalAttention::new(geometry, rotary.clone(), score_scale)?;
        let (model, query_dim, kv_dim) = (
            geometry.model_dim,
            geometry.query_dim(),
            geometry.key_value_dim(),
        );
        for (projection, weight, expected) in [
            (AttentionProjection::Query, &query, (query_dim, model)),
            (AttentionProjection::Key, &key, (kv_dim, model)),
            (AttentionProjection::Value, &value, (kv_dim, model)),
            (AttentionProjection::Output, &output, (model, query_dim)),
        ] {
            if weight.dim() != expected {
                return Err(BlockError::ProjectionShape {
                    projection,
                    expected,
                    found: weight.dim(),
                });
            }
        }
        Ok(Self {
            geometry,
            rotary,
            score_scale,
            attention,
            query,
            key,
            value,
            output,
        })
    }

    pub fn geometry(&self) -> AttentionGeometry {
        self.geometry
    }

    /// The source's rotary embedding: empty for learned absolute positions.
    pub fn rotary(&self) -> &RotaryEmbedding {
        &self.rotary
    }

    /// The source's score scale, `1/sqrt(head_dim)` for a standard head.
    pub fn score_scale(&self) -> f64 {
        self.score_scale
    }

    /// The stored weight of one projection.
    pub fn weight(&self, projection: AttentionProjection) -> ArrayView2<'_, f64> {
        match projection {
            AttentionProjection::Query => self.query.view(),
            AttentionProjection::Key => self.key.view(),
            AttentionProjection::Value => self.value.view(),
            AttentionProjection::Output => self.output.view(),
        }
    }

    /// The layer under `reads`, over the residual rows at their absolute positions, with
    /// their radius against the exact rows ([`ProjectedRows::exact`] for exact rows). A
    /// component read is refused: this layer holds no component factors.
    pub fn execute(
        &self,
        reads: AttentionLayerReads<'_>,
        residual: ProjectedRows<'_>,
        positions: &[i64],
    ) -> Result<AttentionLayerExecution, BlockError> {
        execute_attention_layer(self, None, reads, residual, positions)
    }

    /// The rounding band of one linear read of `rows` at their absolute positions:
    /// `|fl(A x) − A x|` for the map `A` that `read` applies, entrywise, where `fl`
    /// is apply.rs's kernel ([`read_band`]). A component read is refused.
    pub fn read_band(
        &self,
        projection: AttentionProjection,
        read: ProjectionRead<'_>,
        rows: ArrayView2<'_, f64>,
        positions: &[i64],
    ) -> Result<Array2<f64>, BlockError> {
        read_band(self.weight(projection), None, projection, read, rows, positions)
    }
}

/// The exact factor `W = U R` of one projection, with `Rᵀ` held in the column layout
/// the matrix-free kernels read.
#[derive(Clone, Debug)]
struct ProjectionFactor {
    factor: ExactFactor,
    read_transpose: Array2<f64>,
}

impl ProjectionFactor {
    fn view(&self) -> Result<FactorView<'_>, ApplyError> {
        FactorView::new(self.factor.write(), self.read_transpose.view())
    }
}

/// An attention-only layer whose four projections can execute through masked
/// component coordinates.
#[derive(Clone, Debug)]
pub struct ComponentAttentionLayer {
    native: NativeAttentionLayer,
    factors: [ProjectionFactor; 4],
}

impl ComponentAttentionLayer {
    /// Solves each projection's exact write through its declared read (rewrite's
    /// overcomplete factor), refusing an uncovered or unresolved read.
    pub fn new(
        native: NativeAttentionLayer,
        query: ComponentRead<'_>,
        key: ComponentRead<'_>,
        value: ComponentRead<'_>,
        output: ComponentRead<'_>,
    ) -> Result<Self, BlockError> {
        let solve = |projection: AttentionProjection, read: ComponentRead<'_>| {
            ExactFactor::solve_write(native.weight(projection), read.read, read.candidate_write)
                .map(|factor| ProjectionFactor {
                    read_transpose: factor.read().t().to_owned(),
                    factor,
                })
                .map_err(|refusal| BlockError::Factor { projection, refusal })
        };
        let factors = [
            solve(AttentionProjection::Query, query)?,
            solve(AttentionProjection::Key, key)?,
            solve(AttentionProjection::Value, value)?,
            solve(AttentionProjection::Output, output)?,
        ];
        Ok(Self { native, factors })
    }

    /// The layer on its stored tensors.
    pub fn native(&self) -> &NativeAttentionLayer {
        &self.native
    }

    /// The exact factor `W = U R` of one projection.
    pub fn factor(&self, projection: AttentionProjection) -> &ExactFactor {
        &self.factors[projection.index()].factor
    }

    /// One projection's factors as apply.rs reads them: left `U`, right `Rᵀ`.
    pub fn components(&self, projection: AttentionProjection) -> Result<FactorView<'_>, BlockError> {
        self.factors[projection.index()]
            .view()
            .map_err(|error| BlockError::Apply { projection, error })
    }

    /// The layer under `reads`, over the residual rows at their absolute positions, with
    /// their radius against the exact rows ([`ProjectedRows::exact`] for exact rows).
    pub fn execute(
        &self,
        reads: AttentionLayerReads<'_>,
        residual: ProjectedRows<'_>,
        positions: &[i64],
    ) -> Result<AttentionLayerExecution, BlockError> {
        execute_attention_layer(&self.native, Some(&self.factors), reads, residual, positions)
    }

    /// The rounding band of one linear read of `rows` at their absolute positions
    /// ([`read_band`]).
    pub fn read_band(
        &self,
        projection: AttentionProjection,
        read: ProjectionRead<'_>,
        rows: ArrayView2<'_, f64>,
        positions: &[i64],
    ) -> Result<Array2<f64>, BlockError> {
        read_band(
            self.native.weight(projection),
            Some(&self.factors[projection.index()]),
            projection,
            read,
            rows,
            positions,
        )
    }
}

fn execute_attention_layer(
    layer: &NativeAttentionLayer,
    factors: Option<&[ProjectionFactor; 4]>,
    reads: AttentionLayerReads<'_>,
    residual: ProjectedRows<'_>,
    positions: &[i64],
) -> Result<AttentionLayerExecution, BlockError> {
    let expected = (positions.len(), layer.geometry.model_dim);
    for found in [residual.values.dim(), residual.radius.dim()] {
        if found != expected {
            return Err(BlockError::ResidualShape { expected, found });
        }
    }
    let factor = |projection: AttentionProjection| factors.map(|all| &all[projection.index()]);
    // Every read takes one route: the exact read `A x*` of the exact rows differs from the
    // computed read by the rounding band at the computed rows plus `|A| |x̂ − x*| ≤ |A| r`,
    // with `r` the rows' radius.
    let read_rows = |projection: AttentionProjection, read: ProjectionRead<'_>, rows: ProjectedRows<'_>| {
        let (weight, factor) = (layer.weight(projection), factor(projection));
        let values = read_projection(weight, factor, projection, read, rows.values, positions)?;
        let rounding = read_band(weight, factor, projection, read, rows.values, positions)?;
        let radius = if is_exact(rows.radius) {
            rounding
        } else {
            let carried = read_magnitude(weight, factor, projection, read, rows.radius, positions)?;
            sum_of_bounds(rounding, carried.dominated())
        };
        // A mask box adds `|U| diag|w| |R| (|x̂| + r)`, the most any mask of the box moves the
        // exact read of the exact rows away from its center's, since `|x*| ≤ |x̂| + r`.
        let radius = match read {
            ProjectionRead::Components(ComponentMasks {
                half_width: Some(half_width),
                ..
            }) => {
                let reach = sum_of_bounds(rows.values.mapv(f64::abs), rows.radius.to_owned());
                let widths = ProjectionRead::Components(ComponentMasks {
                    center: half_width,
                    half_width: None,
                });
                let spread = read_magnitude(weight, factor, projection, widths, reach.view(), positions)?;
                sum_of_bounds(radius, spread.dominated())
            }
            ProjectionRead::Native | ProjectionRead::Components(_) | ProjectionRead::Edited(_) => radius,
        };
        Ok::<_, BlockError>((values, radius))
    };
    let (queries, query_radius) = read_rows(AttentionProjection::Query, reads.query, residual)?;
    let (keys, key_radius) = read_rows(AttentionProjection::Key, reads.key, residual)?;
    let (values, value_radius) = read_rows(AttentionProjection::Value, reads.value, residual)?;
    let attention = layer.attention.attend_projected(
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
    // The output read takes the head-mixed rows with their radius against the exact mixed
    // rows `z*` of the exact reads.
    let (write, write_radius) = read_rows(
        AttentionProjection::Output,
        reads.output,
        ProjectedRows {
            values: attention.mixed.view(),
            radius: attention.mixed_radius.view(),
        },
    )?;
    let mut output = residual.values.to_owned();
    output += &*write;
    // `|fl(h + w) − (h + w)| ≤ u |fl(h + w)| / (1 − u)`, and `|ĥ − h*| ≤ r` for the stream.
    let addition = output.mapv(|value| UNIT_ROUNDOFF * value.abs() / (1.0 - UNIT_ROUNDOFF));
    let output_radius = sum_of_bounds(write_radius.clone(), addition);
    let output_radius = if is_exact(residual.radius) {
        output_radius
    } else {
        sum_of_bounds(output_radius, residual.radius.to_owned())
    };
    Ok(AttentionLayerExecution {
        queries,
        query_radius,
        keys,
        key_radius,
        values,
        value_radius,
        attention,
        write,
        write_radius,
        output,
        output_radius,
    })
}

/// The rows of `positions` that an edit scoped at `scoped` reaches, refusing a
/// negative position and a declared position no row holds.
fn reached_rows(
    projection: AttentionProjection,
    scoped: &ScopedEdit<'_>,
    positions: &[i64],
) -> Result<Vec<usize>, BlockError> {
    let mut row_positions = Vec::with_capacity(positions.len());
    for &position in positions {
        row_positions.push(
            usize::try_from(position)
                .ok()
                .ok_or(BlockError::NegativePosition {
                    projection,
                    position,
                })?,
        );
    }
    if let Some(&absent) = scoped
        .positions
        .positions()
        .and_then(|declared| declared.iter().find(|position| !row_positions.contains(position)))
    {
        return Err(BlockError::EditPositionAbsent {
            projection,
            position: absent,
        });
    }
    Ok(row_positions
        .iter()
        .enumerate()
        .filter_map(|(row, position)| scoped.positions.reaches(*position).then_some(row))
        .collect())
}

/// One linear read of `rows`, whose row `r` sits at absolute position `positions[r]`.
fn read_projection(
    weight: ArrayView2<'_, f64>,
    factor: Option<&ProjectionFactor>,
    projection: AttentionProjection,
    read: ProjectionRead<'_>,
    rows: ArrayView2<'_, f64>,
    positions: &[i64],
) -> Result<Governed<Array2<f64>>, BlockError> {
    let refused = |error: ApplyError| BlockError::Apply { projection, error };
    match read {
        ProjectionRead::Native => native_linear(weight, rows).map_err(refused),
        ProjectionRead::Components(masks) => {
            let factor = factor.ok_or(BlockError::NoComponentFactors { projection })?;
            let view = factor.view().map_err(refused)?;
            let groups = mask_groups(projection, &masks, rows.nrows())?;
            // The rows of the first mask read every row, so one mask for every row is one
            // product; each other mask's rows are read on their own and written over it.
            let first = groups.first().map_or(0, |group| group.0);
            let mut written =
                apply_anchored_linear(weight, 0.0, view, masks.center.row(first), rows).map_err(refused)?;
            for (mask_row, members) in groups.iter().skip(1) {
                let selected = rows.select(Axis(0), members);
                let read = apply_anchored_linear(weight, 0.0, view, masks.center.row(*mask_row), selected.view())
                    .map_err(refused)?;
                for (index, &row) in members.iter().enumerate() {
                    written.row_mut(row).assign(&read.row(index));
                }
            }
            Ok(written)
        }
        ProjectionRead::Edited(scoped) => {
            let reached = reached_rows(projection, &scoped, positions)?;
            let mut written = native_linear(weight, rows).map_err(refused)?;
            if reached.is_empty() {
                return Ok(written);
            }
            let selected = rows.select(Axis(0), &reached);
            let every_term = Array1::<f64>::ones(scoped.edit.term_count());
            let edited = apply_anchored_linear(weight, 1.0, scoped.edit, every_term.view(), selected.view())
                .map_err(refused)?;
            for (index, &row) in reached.iter().enumerate() {
                written.row_mut(row).assign(&edited.row(index));
            }
            Ok(written)
        }
    }
}

/// The rows that read each distinct mask of `masks`, as `(mask row, rows)` in order of
/// first appearance: one group of every row for a `1 × C` center. Refuses a center with
/// neither one row nor one per input row, and a half-width of another shape than the center
/// or with a non-finite entry.
fn mask_groups(
    projection: AttentionProjection,
    masks: &ComponentMasks<'_>,
    tokens: usize,
) -> Result<Vec<(usize, Vec<usize>)>, BlockError> {
    let refused = |error: ApplyError| BlockError::Apply { projection, error };
    let (mask_rows, components) = masks.center.dim();
    if mask_rows != 1 && (mask_rows != tokens || tokens == 0) {
        return Err(refused(ApplyError::Shape {
            operand: "component mask rows",
            expected: (tokens.max(1), components),
            found: (mask_rows, components),
        }));
    }
    if let Some(half_width) = masks.half_width {
        if half_width.dim() != masks.center.dim() {
            return Err(refused(ApplyError::Shape {
                operand: "mask half-width",
                expected: masks.center.dim(),
                found: half_width.dim(),
            }));
        }
        if !half_width.iter().all(|width| width.is_finite()) {
            return Err(refused(ApplyError::NonFinite {
                operand: "mask half-width",
            }));
        }
    }
    if mask_rows == 1 {
        return Ok(vec![(0, (0..tokens).collect())]);
    }
    let mut groups: Vec<(usize, Vec<usize>)> = Vec::new();
    for row in 0..tokens {
        let same = |other: usize| {
            masks
                .center
                .row(other)
                .iter()
                .zip(masks.center.row(row))
                .all(|(a, b)| a.to_bits() == b.to_bits())
        };
        match groups.iter_mut().find(|(mask_row, _)| same(*mask_row)) {
            Some((_, members)) => members.push(row),
            None => groups.push((row, vec![row])),
        }
    }
    Ok(groups)
}

/// `|M| r` for nonnegative rows `r` (`tokens × n`) and a matrix `M` (`outputs × n`), one
/// output column at a time, so `|M|` is never formed.
fn abs_map(matrix: ArrayView2<'_, f64>, rows: ArrayView2<'_, f64>) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((rows.nrows(), matrix.nrows()));
    for (column, entries) in matrix.outer_iter().enumerate() {
        out.column_mut(column).assign(&rows.dot(&entries.mapv(f64::abs)));
    }
    out
}

/// `|A| r` for nonnegative rows `r`, with `A` the map a read applies to each row, the rounded
/// operations of that read's product (`n`) and of this magnitude (`k`), per row, and the
/// read's underflow reach per entry.
///
/// Rounding in binary64 is `fl(a ∘ b) = (a ∘ b)(1 + δ) + η` with `|δ| ≤ u`, where `|η| ≤ 2^-1075`
/// for a product or quotient that rounds into the subnormal range and `η = 0` for a sum, which is
/// exact there (a fused multiply-add rounds once and adds one `η`). Higham's `γ_n |A| |x|` covers
/// the `δ` terms only. Each product's `η` passes the later sums with a factor of at most
/// `1 + γ_n ≤ 2`, and a later product scales an earlier stage's `η` by the magnitude it
/// multiplies. So the entry's absolute term is at most `𝒜 · 2^-1074`, with the allowance `𝒜` of
/// [`read_magnitude`]. The magnitude program runs the read's products on nonnegative operands,
/// so `𝒜` bounds its underflow as well. `underflow` holds `𝒜 · 2^-1074`, rounded up.
struct ReadMagnitude {
    magnitude: Array2<f64>,
    underflow: Array2<f64>,
    read_operations: Vec<usize>,
    magnitude_operations: Vec<usize>,
}

/// `𝒜 · 2^-1074` rounded up, for an allowance `𝒜` computed as `computed` in `operations` rounded
/// operations on nonnegative terms.
///
/// `𝒜` is either exactly zero (a read with no products) or at least one, since it counts the
/// read's own last products. So the underflow inside its own computation, below
/// `operations · 2^-1074`, lies far inside the step one float up from a value of at least one, and
/// `𝒜 ≤ up(𝒜̂ / (1 − γ_k))`.
fn underflow_allowance(computed: f64, operations: usize) -> f64 {
    let dominance = down(1.0 - up(accumulation_growth(operations)));
    up(up(computed / dominance) * SUBNORMAL_SPACING)
}

impl ReadMagnitude {
    /// The magnitude itself as a bound on `|A| r`. Its `k` operations on nonnegative terms each
    /// scale by `1 + δ`, `|δ| ≤ u`, and its products add at most the underflow reach `a`, so
    /// `M̂ ≥ (1 − γ_k) |A| r − a` and the exact value is at most `(M̂ + a) / (1 − γ_k)`. Every
    /// operation of that bound rounds to nearest and steps one float up (down for the
    /// divisor), so a magnitude that underflowed to zero still carries `a`.
    fn dominated(self) -> Array2<f64> {
        dominate(self.magnitude, self.underflow.view(), &self.magnitude_operations)
    }

    /// `|fl(A x) − A x| ≤ γ_n |A| |x| + a` per entry: the forward error of a product with `n`
    /// rounded operations per entry in any summation order (Higham ASNA Lemma 3.1), plus the
    /// read's underflow reach `a`, with `|A| |x|` bounded as in [`Self::dominated`] from a
    /// magnitude of `k` operations. Evaluated with every operation stepped one float up, so a
    /// subnormal band does not round to zero.
    fn rounding(self) -> Array2<f64> {
        let mut band = dominate(self.magnitude, self.underflow.view(), &self.magnitude_operations);
        for ((mut row, reach), &read) in band
            .outer_iter_mut()
            .zip(self.underflow.outer_iter())
            .zip(&self.read_operations)
        {
            let growth = up(accumulation_growth(read));
            Zip::from(&mut row)
                .and(&reach)
                .for_each(|value, &reach| *value = up(up(growth * *value) + reach));
        }
        band
    }
}

/// `(M̂ + a) / (1 − γ_k)` per entry, each operation rounded to nearest and stepped one float up
/// (down for the divisor): the upper bound [`ReadMagnitude::dominated`] derives.
fn dominate(magnitude: Array2<f64>, underflow: ArrayView2<'_, f64>, operations: &[usize]) -> Array2<f64> {
    let mut bound = magnitude;
    for ((mut row, reach), &operations) in bound.outer_iter_mut().zip(underflow.outer_iter()).zip(operations) {
        let dominance = down(1.0 - up(accumulation_growth(operations)));
        Zip::from(&mut row)
            .and(&reach)
            .for_each(|value, &reach| *value = up(up(*value + reach) / dominance));
    }
    bound
}

/// `a + b` for two nonnegative bounds, rounded up: the exact sum is at most the computed
/// one divided by `1 − u`.
fn sum_of_bounds(first: Array2<f64>, second: Array2<f64>) -> Array2<f64> {
    (first + &second).mapv(|value| value / (1.0 - UNIT_ROUNDOFF))
}

/// Whether rows with this radius are exact. Their input radius then carries nothing into a
/// read, and a bound plus exact zeros is the bound itself, with no addition to round.
fn is_exact(radius: ArrayView2<'_, f64>) -> bool {
    radius.iter().all(|&entry| entry == 0.0)
}

/// `|W| r` for a native read of the nonnegative rows `r`: the inner product takes `n = d`
/// rounded operations, and so does this magnitude. Its `d` products give the allowance
/// `𝒜 = d`, and `d · 2^-1074` is exact for `d < 2^53`.
fn native_magnitude(weight: ArrayView2<'_, f64>, rows: ArrayView2<'_, f64>) -> ReadMagnitude {
    let (tokens, width) = rows.dim();
    ReadMagnitude {
        magnitude: abs_map(weight, rows),
        underflow: Array2::from_elem((tokens, weight.nrows()), width as f64 * SUBNORMAL_SPACING),
        read_operations: vec![width; tokens],
        magnitude_operations: vec![width; tokens],
    }
}

/// A native linear read `x Wᵀ` of rows that carry a radius against their exact values, the
/// way an attention-only layer reads a projection under [`ProjectionRead::Native`]:
/// apply.rs's `native_linear`, its rounding band `γ_d |W| |x̂| + d · 2^-1074` (the second term
/// for its products that round into the subnormal range), and `|W| r`, the most the exact read
/// can move between the computed rows and the exact ones. A stack of layers ends in such a
/// read, its unembedding, so the logits carry the whole stack's radius. Returns the read and
/// its radius against the exact read of the exact rows.
pub fn linear_read(
    weight: ArrayView2<'_, f64>,
    rows: ProjectedRows<'_>,
) -> Result<(Governed<Array2<f64>>, Array2<f64>), ApplyError> {
    if rows.radius.dim() != rows.values.dim() {
        return Err(ApplyError::Shape {
            operand: "input radius",
            expected: rows.values.dim(),
            found: rows.radius.dim(),
        });
    }
    let values = native_linear(weight, rows.values)?;
    let rounding = native_magnitude(weight, rows.values.mapv(f64::abs).view()).rounding();
    let radius = if is_exact(rows.radius) {
        rounding
    } else {
        sum_of_bounds(rounding, native_magnitude(weight, rows.radius).dominated())
    };
    Ok((values, radius))
}

fn expect_read_shape(
    projection: AttentionProjection,
    operand: &'static str,
    expected: (usize, usize),
    found: (usize, usize),
) -> Result<(), BlockError> {
    if expected == found {
        Ok(())
    } else {
        Err(BlockError::Apply {
            projection,
            error: ApplyError::Shape {
                operand,
                expected,
                found,
            },
        })
    }
}

/// `|A| r` for the map `read` applies ([`ReadMagnitude`]), never forming `|A|`:
/// - `Native`: `|W| r`, the read taking `n = d` operations (the inner product);
/// - `Components(m)`: `|U| diag|m| |R| r` with `m` the box's center (each row's own for a
///   per-row center), the read taking `n = C + d + 2` (the products `R x`, the scale, the
///   `C`-term product with `U` and the accumulation into the tile);
/// - `Edited`: `|W| r` on every row, plus `|L| |Rᵀ| r` on the rows the edit reaches, whose
///   read takes `n = T + d + 2` (the native product, then the `T` edit terms the same way).
///
/// Each arm also carries the read's underflow allowance `𝒜` per entry ([`ReadMagnitude`]):
/// `d` for `Native`, `C + |U| (𝟙 + d |m|)` for `Components(m)`, and `(d + T) + d |L| 𝟙` on the
/// rows an edit reaches.
fn read_magnitude(
    weight: ArrayView2<'_, f64>,
    factor: Option<&ProjectionFactor>,
    projection: AttentionProjection,
    read: ProjectionRead<'_>,
    rows: ArrayView2<'_, f64>,
    positions: &[i64],
) -> Result<ReadMagnitude, BlockError> {
    let (tokens, width) = rows.dim();
    expect_read_shape(projection, "band input rows", (tokens, weight.ncols()), (tokens, width))?;
    match read {
        ProjectionRead::Native => Ok(native_magnitude(weight, rows)),
        ProjectionRead::Components(masks) => {
            let factor = &factor.ok_or(BlockError::NoComponentFactors { projection })?.factor;
            let components = factor.components();
            expect_read_shape(projection, "band component mask", (components, 1), (masks.center.ncols(), 1))?;
            mask_groups(projection, &masks, tokens)?;
            // A `1 × C` center scales every row; a `rows × C` center scales its own row.
            let coordinates = abs_map(factor.read(), rows) * &masks.center.mapv(f64::abs);
            // `𝒜 = C + |U| (1 + d |m|)`: the `C` products with `U`, and each coordinate's `d`
            // products scaled by `|m_c|` plus the scale's own product, scaled by `|u_ic|`. It
            // takes `C + 3` roundings: `d |m_c|`, `1 +`, the `C`-term product with `|U|`, `C +`.
            let scaled = masks.center.mapv(|center| 1.0 + width as f64 * center.abs());
            let allowance = abs_map(factor.write(), scaled.view())
                .mapv(|reach| underflow_allowance(components as f64 + reach, components + 3));
            let per_row = allowance.nrows() == tokens;
            let underflow = Array2::from_shape_fn((tokens, allowance.ncols()), |(row, output)| {
                allowance[[if per_row { row } else { 0 }, output]]
            });
            Ok(ReadMagnitude {
                magnitude: abs_map(factor.write(), coordinates.view()),
                underflow,
                read_operations: vec![components + width + 2; tokens],
                magnitude_operations: vec![components + width + 1; tokens],
            })
        }
        ProjectionRead::Edited(scoped) => {
            let terms = scoped.edit.term_count();
            expect_read_shape(
                projection,
                "band edit factors",
                (weight.nrows(), width),
                (scoped.edit.output_dim(), scoped.edit.input_dim()),
            )?;
            let reached = reached_rows(projection, &scoped, positions)?;
            let mut magnitude = native_magnitude(weight, rows);
            if !reached.is_empty() {
                let selected = rows.select(Axis(0), &reached);
                let edit = abs_map(scoped.edit.left(), abs_map(scoped.edit.right().t(), selected.view()).view());
                // `𝒜 = (d + T) + d |L| 𝟙` on a reached row: the native and edit products, and
                // each edit coordinate's `d` products scaled by `|l_it|` (the unit term scales
                // multiply exactly). It takes `T + 1` roundings: the `T`-term product of `d`
                // with `|L|`, then `(d + T) +`.
                let spread = Array2::from_elem((1, terms), width as f64);
                let allowance = abs_map(scoped.edit.left(), spread.view())
                    .mapv(|reach| underflow_allowance((width + terms) as f64 + reach, terms + 1));
                for (index, &row) in reached.iter().enumerate() {
                    magnitude.underflow.row_mut(row).assign(&allowance.row(0));
                    let mut target = magnitude.magnitude.row_mut(row);
                    target += &edit.row(index);
                    magnitude.read_operations[row] = terms + width + 2;
                    magnitude.magnitude_operations[row] = terms + width + 1;
                }
            }
            Ok(magnitude)
        }
    }
}

/// The rounding band of one read of `rows`: `|fl(A x) − A x| ≤ γ_n |A| |x| + 𝒜 · 2^-1074`
/// entrywise, with `fl` apply.rs's kernel and `n`, `|A|` and the underflow allowance `𝒜` as in
/// [`read_magnitude`]. Rows the read leaves unedited get the native band.
fn read_band(
    weight: ArrayView2<'_, f64>,
    factor: Option<&ProjectionFactor>,
    projection: AttentionProjection,
    read: ProjectionRead<'_>,
    rows: ArrayView2<'_, f64>,
    positions: &[i64],
) -> Result<Array2<f64>, BlockError> {
    let magnitude = rows.mapv(f64::abs);
    Ok(read_magnitude(weight, factor, projection, read, magnitude.view(), positions)?.rounding())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter_decomposition::attention::{AffineProjection, RotaryPairing};
    use crate::parameter_decomposition::rewrite::ComponentMask;
    use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
    use gam_math::gaussian_activation::GaussianActivation;
    use ndarray::ArrayView1;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    const WIDTH: usize = 4;
    const HIDDEN: usize = 6;
    const READ_IN_COMPONENTS: usize = 7;
    const WRITE_OUT_COMPONENTS: usize = 8;
    const QUERY_COMPONENTS: usize = 5;
    const KEY_COMPONENTS: usize = 4;
    const TOKENS: usize = 5;
    const GATE_COMPONENTS: usize = 5;
    const UP_COMPONENTS: usize = 5;
    const DOWN_COMPONENTS: usize = 7;
    /// Qwen3's declared `rms_norm_eps`.
    const RMS_EPSILON: f64 = 1.0e-6;
    /// Pythia's declared `layer_norm_eps`.
    const LAYER_EPSILON: f64 = 1.0e-5;

    /// Two query heads sharing one key/value head, two rotated planes and two
    /// pass-through coordinates per head.
    fn geometry() -> AttentionGeometry {
        AttentionGeometry {
            model_dim: WIDTH,
            n_heads: 2,
            n_kv_heads: 1,
            head_dim: 6,
        }
    }

    fn rotary() -> RotaryEmbedding {
        RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: vec![1.0, 0.25],
            attention_scaling: 1.25,
        }
    }

    /// Eighths in `[-1, 1]`. A product of three and a sum of a few dozen stay
    /// exact in f64, so every product of fixture tensors and masks, including each
    /// edited tensor `A diag(m) B`, is formed without rounding.
    fn eighths(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || rng.random_range(-8..=8) as f64 / 8.0)
    }

    fn eighths_vector(rng: &mut StdRng, len: usize) -> Array1<f64> {
        Array1::from_shape_simple_fn(len, || rng.random_range(-8..=8) as f64 / 8.0)
    }

    /// A bias-free source projection: adding the zero bias is exact, so the
    /// block's arithmetic is that of the unbiased source.
    fn unbiased(weight: Array2<f64>) -> AffineProjection {
        let bias = Array1::zeros(weight.nrows());
        AffineProjection { weight, bias }
    }

    /// `A diag(m) B`.
    fn masked_product(write: ArrayView2<'_, f64>, mask: ArrayView1<'_, f64>, read: ArrayView2<'_, f64>) -> Array2<f64> {
        (&write * &mask).dot(&read)
    }

    struct Fixture {
        layout: ResidualLayout,
        activation: GaussianActivation,
        attention_gain: Array1<f64>,
        attention_bias: Array1<f64>,
        mlp_gain: Array1<f64>,
        mlp_bias: Array1<f64>,
        query_outputs: Array2<f64>,
        query_readins: Array2<f64>,
        key_outputs: Array2<f64>,
        key_readins: Array2<f64>,
        value: Array2<f64>,
        output: Array2<f64>,
        read_in_candidate: Array2<f64>,
        read_in: Array2<f64>,
        bias_in: Array1<f64>,
        write_out_candidate: Array2<f64>,
        write_out: Array2<f64>,
        bias_out: Array1<f64>,
        residual: Array2<f64>,
        positions: Vec<i64>,
        gate_candidate: Array2<f64>,
        gate_read: Array2<f64>,
        up_candidate: Array2<f64>,
        up_read: Array2<f64>,
        down_candidate: Array2<f64>,
        down_read: Array2<f64>,
    }

    impl Fixture {
        /// A block whose weights factor exactly through overcomplete reads,
        /// `W₁ = N R` and `W₂ = N̄ R̄`, so each exact write is its candidate. The
        /// sequential layout normalizes with RMSNorm (Qwen3), the parallel one with
        /// LayerNorm (GPT-NeoX).
        fn new(layout: ResidualLayout, activation: GaussianActivation, seed: u64) -> Self {
            let mut rng = StdRng::seed_from_u64(seed);
            let g = geometry();
            Self {
                layout,
                activation,
                attention_gain: eighths_vector(&mut rng, WIDTH),
                attention_bias: eighths_vector(&mut rng, WIDTH),
                mlp_gain: eighths_vector(&mut rng, WIDTH),
                mlp_bias: eighths_vector(&mut rng, WIDTH),
                query_outputs: eighths(&mut rng, g.query_dim(), QUERY_COMPONENTS),
                query_readins: eighths(&mut rng, QUERY_COMPONENTS, WIDTH),
                key_outputs: eighths(&mut rng, g.key_value_dim(), KEY_COMPONENTS),
                key_readins: eighths(&mut rng, KEY_COMPONENTS, WIDTH),
                value: eighths(&mut rng, g.key_value_dim(), WIDTH),
                output: eighths(&mut rng, WIDTH, g.query_dim()),
                read_in_candidate: eighths(&mut rng, HIDDEN, READ_IN_COMPONENTS),
                read_in: eighths(&mut rng, READ_IN_COMPONENTS, WIDTH),
                bias_in: eighths_vector(&mut rng, HIDDEN),
                write_out_candidate: eighths(&mut rng, WIDTH, WRITE_OUT_COMPONENTS),
                write_out: eighths(&mut rng, WRITE_OUT_COMPONENTS, HIDDEN),
                bias_out: eighths_vector(&mut rng, WIDTH),
                residual: Array2::from_shape_simple_fn((TOKENS, WIDTH), || {
                    rng.random_range(-16..=16) as f64 / 8.0
                }),
                positions: vec![2, 3, 5, 6, 9],
                gate_candidate: eighths(&mut rng, HIDDEN, GATE_COMPONENTS),
                gate_read: eighths(&mut rng, GATE_COMPONENTS, WIDTH),
                up_candidate: eighths(&mut rng, HIDDEN, UP_COMPONENTS),
                up_read: eighths(&mut rng, UP_COMPONENTS, WIDTH),
                down_candidate: eighths(&mut rng, WIDTH, DOWN_COMPONENTS),
                down_read: eighths(&mut rng, DOWN_COMPONENTS, HIDDEN),
            }
        }

        fn native_attention(&self, query: Array2<f64>, key: Array2<f64>) -> NativeAttention {
            NativeAttention::new(
                geometry(),
                rotary(),
                1.0 / (geometry().head_dim as f64).sqrt(),
                unbiased(query),
                unbiased(key),
                unbiased(self.value.clone()),
                unbiased(self.output.clone()),
            )
            .expect("fixture attention tensors match the geometry")
        }

        /// The source gated block: every weight is its factors' product.
        fn native_gated_block(&self) -> NativeGatedBlock {
            NativeGatedBlock::new(
                self.layout,
                self.norm(&self.attention_gain, &self.attention_bias),
                self.native_attention(
                    self.query_outputs.dot(&self.query_readins),
                    self.key_outputs.dot(&self.key_readins),
                ),
                self.norm(&self.mlp_gain, &self.mlp_bias),
                NativeSwiglu::new(
                    self.gate_candidate.dot(&self.gate_read),
                    self.up_candidate.dot(&self.up_read),
                    self.down_candidate.dot(&self.down_read),
                )
                .expect("fixture SwiGLU shapes compose"),
            )
        }

        fn component_gated_block(&self) -> ComponentGatedBlock {
            ComponentGatedBlock::new(
                self.native_gated_block(),
                ComponentProjection::new(self.query_outputs.clone(), self.query_readins.clone())
                    .expect("query factor shapes agree"),
                ComponentProjection::new(self.key_outputs.clone(), self.key_readins.clone())
                    .expect("key factor shapes agree"),
                ComponentRead {
                    read: self.gate_read.view(),
                    candidate_write: self.gate_candidate.view(),
                },
                ComponentRead {
                    read: self.up_read.view(),
                    candidate_write: self.up_candidate.view(),
                },
                ComponentRead {
                    read: self.down_read.view(),
                    candidate_write: self.down_candidate.view(),
                },
            )
            .expect("random overcomplete reads are resolved")
        }

        fn norm(&self, gain: &Array1<f64>, bias: &Array1<f64>) -> NativeNorm {
            match self.layout {
                ResidualLayout::Sequential => NativeNorm::Rms {
                    epsilon: RMS_EPSILON,
                    gain: gain.clone(),
                },
                ResidualLayout::Parallel => NativeNorm::Layer {
                    epsilon: LAYER_EPSILON,
                    gain: gain.clone(),
                    bias: bias.clone(),
                },
            }
        }

        fn native_block(
            &self,
            query: Array2<f64>,
            key: Array2<f64>,
            read_in_weight: Array2<f64>,
            write_out_weight: Array2<f64>,
        ) -> NativeBlock {
            NativeBlock::new(
                self.layout,
                self.norm(&self.attention_gain, &self.attention_bias),
                self.native_attention(query, key),
                self.norm(&self.mlp_gain, &self.mlp_bias),
                NativeMlp::new(
                    read_in_weight,
                    self.bias_in.clone(),
                    write_out_weight,
                    self.bias_out.clone(),
                    self.activation,
                )
                .expect("fixture MLP shapes compose"),
            )
        }

        /// The source block: every weight is its factors' product.
        fn native(&self) -> NativeBlock {
            self.native_block(
                self.query_outputs.dot(&self.query_readins),
                self.key_outputs.dot(&self.key_readins),
                self.read_in_candidate.dot(&self.read_in),
                self.write_out_candidate.dot(&self.write_out),
            )
        }

        fn component_block(&self) -> ComponentBlock {
            ComponentBlock::new(
                self.native(),
                ComponentProjection::new(self.query_outputs.clone(), self.query_readins.clone())
                    .expect("query factor shapes agree"),
                ComponentProjection::new(self.key_outputs.clone(), self.key_readins.clone())
                    .expect("key factor shapes agree"),
                ComponentRead {
                    read: self.read_in.view(),
                    candidate_write: self.read_in_candidate.view(),
                },
                ComponentRead {
                    read: self.write_out.view(),
                    candidate_write: self.write_out_candidate.view(),
                },
            )
            .expect("random overcomplete reads are resolved")
        }

        /// Direct execution with the edited tensors `U diag(m) R`.
        fn edited_block(&self, block: &ComponentBlock, masks: &FixtureMasks) -> NativeBlock {
            let (read_in, write_out) = (block.mlp().read_in(), block.mlp().write_out());
            self.native_block(
                masked_product(self.query_outputs.view(), masks.query.view(), self.query_readins.view()),
                masked_product(self.key_outputs.view(), masks.key.view(), self.key_readins.view()),
                masked_product(read_in.write(), masks.read_in.view(), read_in.read()),
                masked_product(write_out.write(), masks.write_out.view(), write_out.read()),
            )
        }
    }

    #[derive(Clone)]
    struct FixtureMasks {
        query: Array1<f64>,
        key: Array1<f64>,
        read_in: Array1<f64>,
        write_out: Array1<f64>,
    }

    impl FixtureMasks {
        /// Eighths: continuous in `[0, 1]`, binary, and signed in `[-3/2, 3/2]`.
        fn family(rng: &mut StdRng) -> [(&'static str, Self); 3] {
            let mut draw = |low: i32, high: i32, denominator: f64| {
                let mut vector = |len: usize| {
                    Array1::from_shape_simple_fn(len, || rng.random_range(low..=high) as f64 / denominator)
                };
                Self {
                    query: vector(QUERY_COMPONENTS),
                    key: vector(KEY_COMPONENTS),
                    read_in: vector(READ_IN_COMPONENTS),
                    write_out: vector(WRITE_OUT_COMPONENTS),
                }
            };
            [
                ("continuous", draw(0, 8, 8.0)),
                ("binary", draw(0, 1, 1.0)),
                ("signed", draw(-12, 12, 8.0)),
            ]
        }

        fn ones() -> Self {
            Self {
                query: Array1::ones(QUERY_COMPONENTS),
                key: Array1::ones(KEY_COMPONENTS),
                read_in: Array1::ones(READ_IN_COMPONENTS),
                write_out: Array1::ones(WRITE_OUT_COMPONENTS),
            }
        }

        fn attention(&self) -> QueryKeyMasks {
            QueryKeyMasks {
                query: self.query.clone(),
                key: self.key.clone(),
            }
        }

        fn block<'a>(&'a self, attention: &'a QueryKeyMasks) -> BlockMasks<'a> {
            BlockMasks {
                attention,
                mlp: MlpMask {
                    read_in: ComponentMask::Components(self.read_in.view()),
                    write_out: ComponentMask::Components(self.write_out.view()),
                },
            }
        }
    }

    /// `(|x| |B|ᵀ ⊙ |m|) |A|ᵀ` over nonnegative rows `x`: it bounds
    /// `|A diag(m) B| x` entrywise, and on `|x|` it is the absolute sum of the terms
    /// of `A diag(m) B x`.
    fn magnitude(factor: &ExactFactor, mask: ArrayView1<'_, f64>, rows: ArrayView2<'_, f64>) -> Array2<f64> {
        (rows.dot(&factor.read().mapv(f64::abs).t()) * &mask.mapv(f64::abs))
            .dot(&factor.write().mapv(f64::abs).t())
    }

    /// `|fl(a + b) − (a + b)| ≤ u |a + b| ≤ u |fl(a + b)| / (1 − u)`, entrywise.
    fn addition_rounding(sum: &Array2<f64>) -> Array2<f64> {
        sum.mapv(|value| UNIT_ROUNDOFF * value.abs() / (1.0 - UNIT_ROUNDOFF))
    }

    /// Rounding of `A diag(m) B x + b` on either route, for rows `x` of `rows`.
    ///
    /// The factored route passes each term `a_ic m_c b_cj x_j` through the product
    /// `b x`, `d − 1` additions, the mask, the product with `a`, `C − 1` additions and
    /// the bias: `C + d + 2` rounded operations (Higham ASNA Lemma 3.1, in any
    /// summation order). The edited tensor's route passes `d + 1` over terms bounded
    /// entrywise by the same magnitudes, since `|A diag(m) B| ≤ |A| |m| |B|`.
    fn affine_rounding(
        factor: &ExactFactor,
        mask: ArrayView1<'_, f64>,
        rows: ArrayView2<'_, f64>,
        bias: ArrayView1<'_, f64>,
    ) -> Array2<f64> {
        let growth = accumulation_growth(factor.components() + factor.read().ncols() + 2);
        (magnitude(factor, mask, rows.mapv(f64::abs).view()) + &bias.mapv(f64::abs))
            .mapv(|value| growth * value / (1.0 - growth))
    }

    /// The longest chain of rounded operations in either route band below, all
    /// on nonnegative operands:
    /// - the stream error takes 3;
    /// - its row norm takes `d + 1`, and the normalizer slope `d + 7`;
    /// - the MLP input error takes 3 more;
    /// - the read-in propagation takes `C + d + 1`, then 1;
    /// - the write-out propagation takes `C̄ + H + 1`, then 3;
    /// - the sum of the two routes takes 1.
    const BAND_OPERATIONS: usize =
        2 * WIDTH + READ_IN_COMPONENTS + HIDDEN + WRITE_OUT_COMPONENTS + 18;

    /// Divides a computed band by `1 − γ_(k+1)`, with `k` = [`BAND_OPERATIONS`] and
    /// the extra operation this division. Every operation of the band arithmetic
    /// scales its chain by a factor `(1 + δ)^(±1)`, or `(1 + δ)^(1/2)` for a square
    /// root, with `|δ| ≤ u`. So the computed band dominates its real value.
    fn dominate(band: Array2<f64>) -> Array2<f64> {
        let factor = 1.0 - accumulation_growth(BAND_OPERATIONS + 1);
        band.mapv(|value| value / factor)
    }

    /// Per output entry, a bound on the distance of one route's computed sequential
    /// RMSNorm ReLU block from the exact block with the edited tensors.
    ///
    /// Both routes normalize the same `h` with the same code, so they hand the
    /// attention identical bits `x = fl(N₁(h))`. Let `o*` be the exact sublayer at
    /// `x`, the same for both routes (P6). The reference is
    /// `s* = h + o*`, `h'* = s* + W₂(m̄) relu(W₁(m) N₂(s*) + b₁) + b₂`.
    /// - **Stream.** `|ŝ − s*| ≤ r̂ + u|ŝ|/(1 − u)`, with `r̂` the attention's
    ///   output radius (first order in its trigonometric input error).
    /// - **MLP input.** `|fl(N₂(ŝ)) − N₂(ŝ)| ≤ γ_(d+6) |N₂(ŝ)|`: the squares, `d − 1`
    ///   additions, the mean, `+ ε`, the square root, the reciprocal, the product
    ///   with the row and the gain. The Jacobian of `h ↦ h (‖h‖²/d + ε)^(-1/2)` has
    ///   spectral norm `ρ(h) = (‖h‖²/d + ε)^(-1/2)`: its eigenvalues are `ρ` on
    ///   `h^⊥` and `ρ ε/(‖h‖²/d + ε)` along `h`. So the mean value inequality bounds
    ///   `|N₂(ŝ)_i − N₂(s*)_i|` by `|w_i| sup ρ · ‖ŝ − s*‖₂`. The supremum is over
    ///   the segment, whose norms stay at least `‖ŝ‖/2` once `4‖ŝ − s*‖ ≤ ‖ŝ‖`,
    ///   which the band asserts.
    /// - **Pre-activation.** The route's own rounding ([`affine_rounding`]) plus
    ///   `|U| |m| |R|` times the MLP input error.
    /// - **Activation.** `relu` is exact and 1-Lipschitz, so the pre-activation
    ///   error carries through unchanged.
    /// - **Output.** The route's rounding of the write, `|Ū| |m̄| |R̄|` times the
    ///   activation error, the stream error, and the rounding of the last
    ///   addition.
    fn sequential_route_band(
        fixture: &Fixture,
        block: &ComponentBlock,
        masks: &FixtureMasks,
        execution: &BlockExecution,
    ) -> Array2<f64> {
        let stream = &fixture.residual + &execution.attention.output;
        let stream_error = &execution.attention.output_radius + &addition_rounding(&stream);
        let norm_growth = accumulation_growth(WIDTH + 6);
        let mut input_error = execution
            .mlp_input
            .mapv(|value| norm_growth * value.abs() / (1.0 - norm_growth));
        for token in 0..TOKENS {
            let displacement = stream_error.row(token).iter().map(|error| error * error).sum::<f64>().sqrt();
            let stream_norm = stream.row(token).iter().map(|value| value * value).sum::<f64>().sqrt();
            assert!(
                4.0 * displacement <= stream_norm,
                "row {token}: the stream error {displacement:e} must stay below a quarter of the row norm {stream_norm:e} for the normalizer slope bound"
            );
            let half = stream_norm / 2.0;
            let slope = (half * half / WIDTH as f64 + RMS_EPSILON).sqrt().recip();
            for (slot, gain) in input_error.row_mut(token).iter_mut().zip(fixture.mlp_gain.iter()) {
                *slot += gain.abs() * slope * displacement;
            }
        }
        let mlp = block.mlp();
        let hidden_error = affine_rounding(
            mlp.read_in(),
            masks.read_in.view(),
            execution.mlp_input.view(),
            fixture.bias_in.view(),
        ) + &magnitude(mlp.read_in(), masks.read_in.view(), input_error.view());
        affine_rounding(
            mlp.write_out(),
            masks.write_out.view(),
            execution.mlp_activations.view(),
            fixture.bias_out.view(),
        ) + &magnitude(mlp.write_out(), masks.write_out.view(), hidden_error.view())
            + &stream_error
            + &addition_rounding(&execution.output)
    }

    /// The parallel layout's route band. Both routes normalize the same `h` for
    /// both sublayers, so the MLP input carries no upstream error. The output
    /// `fl(fl(ŵ + ô) + h)` adds the write's error, the attention radius, and the
    /// rounding of the two additions.
    fn parallel_route_band(
        fixture: &Fixture,
        block: &ComponentBlock,
        masks: &FixtureMasks,
        execution: &BlockExecution,
    ) -> Array2<f64> {
        let mlp = block.mlp();
        let hidden_error = affine_rounding(
            mlp.read_in(),
            masks.read_in.view(),
            execution.mlp_input.view(),
            fixture.bias_in.view(),
        );
        let writes = &execution.mlp_write + &execution.attention.output;
        affine_rounding(
            mlp.write_out(),
            masks.write_out.view(),
            execution.mlp_activations.view(),
            fixture.bias_out.view(),
        ) + &magnitude(mlp.write_out(), masks.write_out.view(), hidden_error.view())
            + &execution.attention.output_radius
            + &addition_rounding(&writes)
            + &addition_rounding(&execution.output)
    }

    fn route_band(
        fixture: &Fixture,
        block: &ComponentBlock,
        masks: &FixtureMasks,
        execution: &BlockExecution,
    ) -> Array2<f64> {
        match fixture.layout {
            ResidualLayout::Sequential => sequential_route_band(fixture, block, masks, execution),
            ResidualLayout::Parallel => parallel_route_band(fixture, block, masks, execution),
        }
    }

    fn violations(left: &Array2<f64>, right: &Array2<f64>, band: &Array2<f64>) -> usize {
        left.iter()
            .zip(right.iter())
            .zip(band.iter())
            .filter(|((a, b), bound)| (*a - *b).abs() > **bound)
            .count()
    }

    fn stage_bits(execution: &BlockExecution) -> Vec<u64> {
        execution
            .attention_input
            .iter()
            .chain(execution.attention.output.iter())
            .chain(execution.attention.weights.iter())
            .chain(execution.mlp_input.iter())
            .chain(execution.mlp_summed_input.iter())
            .chain(execution.mlp_activations.iter())
            .chain(execution.mlp_write.iter())
            .chain(execution.output.iter())
            .map(|value| value.to_bits())
            .collect()
    }

    fn bits(values: &Array2<f64>) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// A1 over a whole block: for continuous, binary and signed masks, the
    /// component block equals direct execution with the edited tensors within the
    /// sum of the two routes' derived bands, in both layouts.
    #[test]
    fn a_masked_block_equals_the_edited_tensor_block_within_the_derived_band() {
        for (layout, seed) in [(ResidualLayout::Sequential, 2961), (ResidualLayout::Parallel, 2962)] {
            let fixture = Fixture::new(layout, GaussianActivation::Relu, seed);
            let block = fixture.component_block();
            assert_eq!(
                (block.mlp().read_in().write(), block.mlp().write_out().write()),
                (fixture.read_in_candidate.view(), fixture.write_out_candidate.view()),
                "{layout:?}: an exactly factored weight's exact write is its candidate, so the edited tensors are formed without rounding"
            );
            let mut rng = StdRng::seed_from_u64(seed + 100);
            for (kind, masks) in FixtureMasks::family(&mut rng) {
                let attention_masks = masks.attention();
                let component = block
                    .execute(masks.block(&attention_masks), fixture.residual.view(), &fixture.positions)
                    .expect("masked block");
                let edited = fixture
                    .edited_block(&block, &masks)
                    .execute(fixture.residual.view(), &fixture.positions)
                    .expect("edited-tensor block");
                let band = dominate(
                    route_band(&fixture, &block, &masks, &component) + &route_band(&fixture, &block, &masks, &edited),
                );
                assert_eq!(
                    violations(&component.output, &edited.output, &band),
                    0,
                    "{layout:?}, {kind} masks: the component block left the derived band around the edited-tensor block"
                );

                // Positive controls: a write-out mask moved by 1e-9, and a key mask
                // moved by 1/8 upstream of the second normalization, both leave the band.
                let mut moved_write_out = masks.clone();
                moved_write_out.write_out[3] += 1.0e-9;
                let mut moved_key = masks.clone();
                moved_key.key[1] += 0.125;
                for (control, moved) in [("write-out 1e-9", moved_write_out), ("key 1/8", moved_key)] {
                    let perturbed = fixture
                        .edited_block(&block, &moved)
                        .execute(fixture.residual.view(), &fixture.positions)
                        .expect("perturbed edited-tensor block");
                    let control_band = dominate(
                        route_band(&fixture, &block, &masks, &component)
                            + &route_band(&fixture, &block, &moved, &perturbed),
                    );
                    assert!(
                        violations(&component.output, &perturbed.output, &control_band) > 0,
                        "{layout:?}, {kind} masks: the band must resolve a {control} mask move"
                    );
                }
            }
        }
    }

    /// All-on masks run both sublayers' original tensors on their original paths,
    /// so every stage is bit-identical to the native block.
    #[test]
    fn all_on_masks_execute_the_native_block_bit_for_bit() {
        for (layout, seed) in [(ResidualLayout::Sequential, 2963), (ResidualLayout::Parallel, 2964)] {
            let fixture = Fixture::new(layout, GaussianActivation::ExactGelu, seed);
            let block = fixture.component_block();
            let native = fixture
                .native()
                .execute(fixture.residual.view(), &fixture.positions)
                .expect("native block");
            let ones = FixtureMasks::ones();
            let attention_masks = ones.attention();
            let all_on = BlockMasks {
                attention: &attention_masks,
                mlp: MlpMask {
                    read_in: ComponentMask::AllOn,
                    write_out: ComponentMask::AllOn,
                },
            };
            let executed = block
                .execute(all_on, fixture.residual.view(), &fixture.positions)
                .expect("all-on block");
            assert!(
                stage_bits(&executed) == stage_bits(&native),
                "{layout:?}: the all-on block must execute the original tensors on their original paths"
            );

            // Positive control: the factored MLP at the all-ones mask is algebraically
            // the same block but not the same bits.
            let factored = block
                .execute(ones.block(&attention_masks), fixture.residual.view(), &fixture.positions)
                .expect("all-ones factored block");
            assert!(
                stage_bits(&factored) != stage_bits(&native),
                "{layout:?}: the bit-identity check must distinguish the factored all-ones path"
            );
        }
    }

    /// The block's stages are its owners' stages on the current stream, composed in
    /// the source's order, bit for bit. A sequential MLP reads `N₂(h + A)`, and a
    /// parallel MLP reads `N₂(h)`. So a query/key mask moves a sequential block's MLP
    /// input and leaves a parallel block's bit-identical.
    #[test]
    fn each_sublayer_reads_the_current_stream_in_the_source_order() {
        for (layout, seed) in [(ResidualLayout::Sequential, 2965), (ResidualLayout::Parallel, 2966)] {
            let fixture = Fixture::new(layout, GaussianActivation::ExactGelu, seed);
            let block = fixture.component_block();
            let mut rng = StdRng::seed_from_u64(seed + 100);
            let [(kind, masks), ..] = FixtureMasks::family(&mut rng);
            let attention_masks = masks.attention();
            let execution = block
                .execute(masks.block(&attention_masks), fixture.residual.view(), &fixture.positions)
                .expect("masked block");

            let attention_input = fixture
                .norm(&fixture.attention_gain, &fixture.attention_bias)
                .as_masked_norm()
                .apply(fixture.residual.view())
                .expect("attention norm");
            let attention = ComponentAttention::new(
                fixture.native().attention,
                ComponentProjection::new(fixture.query_outputs.clone(), fixture.query_readins.clone())
                    .expect("query factor shapes agree"),
                ComponentProjection::new(fixture.key_outputs.clone(), fixture.key_readins.clone())
                    .expect("key factor shapes agree"),
            )
            .expect("attention factors match the geometry")
            .execute(&attention_masks, attention_input.view(), &fixture.positions)
            .expect("masked attention");
            let stream = match layout {
                ResidualLayout::Sequential => &fixture.residual + &attention.output,
                ResidualLayout::Parallel => fixture.residual.clone(),
            };
            let mlp_input = fixture
                .norm(&fixture.mlp_gain, &fixture.mlp_bias)
                .as_masked_norm()
                .apply(stream.view())
                .expect("MLP norm");
            let summed_input = block
                .mlp()
                .summed_input(mlp_input.view(), ComponentMask::Components(masks.read_in.view()))
                .expect("masked summed input");
            let activations = block
                .mlp()
                .native()
                .activate(summed_input.view())
                .expect("activations");
            let write = block
                .mlp()
                .written(activations.view(), ComponentMask::Components(masks.write_out.view()))
                .expect("masked write");
            let output = match layout {
                ResidualLayout::Sequential => &stream + &write,
                ResidualLayout::Parallel => &(&write + &attention.output) + &fixture.residual,
            };
            let composed = BlockExecution {
                attention_input,
                attention,
                mlp_input,
                mlp_summed_input: summed_input,
                mlp_activations: activations,
                mlp_write: write,
                output,
            };
            assert!(
                stage_bits(&execution) == stage_bits(&composed),
                "{layout:?}, {kind} masks: the block must be its owners' stages composed on the current stream"
            );

            let mut moved = masks.clone();
            moved.key[1] += 0.125;
            let moved_attention = moved.attention();
            let moved_execution = block
                .execute(moved.block(&moved_attention), fixture.residual.view(), &fixture.positions)
                .expect("moved-mask block");
            assert!(
                bits(&moved_execution.attention.output) != bits(&execution.attention.output),
                "{layout:?}: positive control: the moved key mask must reach the attention write"
            );
            let mlp_input_moved = bits(&moved_execution.mlp_input) != bits(&execution.mlp_input);
            assert_eq!(
                mlp_input_moved,
                layout == ResidualLayout::Sequential,
                "{layout:?}: a key mask must reach the MLP input exactly in the sequential layout"
            );
        }
    }

    /// `NativeNorm::apply_with_band` returns `MaskedNorm::apply`'s rows, and its band covers
    /// the double-double normalization of the same rows, for RMSNorm and for LayerNorm,
    /// including a LayerNorm row near a constant, where the centring error dominates.
    /// Positive control: on that row the norm's own rounding alone, without the centring
    /// term, is exceeded. A zero-epsilon LayerNorm row whose centring error reaches a quarter
    /// of its norm is refused, typed.
    #[test]
    fn a_normalization_band_covers_the_double_double_normalization() {
        use qd::Quad;
        let mut rng = StdRng::seed_from_u64(2997);
        let gain = eighths_vector(&mut rng, LAYER_WIDTH);
        let bias = eighths_vector(&mut rng, LAYER_WIDTH);
        let mut rows = Array2::from_shape_simple_fn((LAYER_TOKENS, LAYER_WIDTH), || {
            rng.random_range(-48..=48) as f64 / 24.0
        });
        // The last row sits near 1024: its centred values are about 1e-6, so fl(mean h) has an
        // error far above the centred row's own rounding.
        let near_constant = LAYER_TOKENS - 1;
        for (column, slot) in rows.row_mut(near_constant).iter_mut().enumerate() {
            *slot = 1024.0 + (column as f64 - 3.5) * 1.0e-6;
        }
        let quad = Quad::from_f64;
        let reference = |norm: &NativeNorm, row: usize| -> Vec<Quad> {
            let values: Vec<Quad> = rows.row(row).iter().map(|&value| quad(value)).collect();
            let width = quad(LAYER_WIDTH as f64);
            let (epsilon, centred): (f64, Vec<Quad>) = match norm {
                NativeNorm::Rms { epsilon, .. } => (*epsilon, values.clone()),
                NativeNorm::Layer { epsilon, .. } => {
                    let mean = values.iter().fold(quad(0.0), |sum, &value| sum + value) / width;
                    (*epsilon, values.iter().map(|&value| value - mean).collect())
                }
            };
            let square = centred.iter().fold(quad(0.0), |sum, &value| sum + value * value);
            let inverse_root = quad(1.0) / (square / width + quad(epsilon)).sqrt();
            centred
                .iter()
                .enumerate()
                .map(|(column, &value)| match norm {
                    NativeNorm::Rms { gain, .. } => quad(gain[column]) * value * inverse_root,
                    NativeNorm::Layer { gain, bias, .. } => {
                        quad(gain[column]) * value * inverse_root + quad(bias[column])
                    }
                })
                .collect()
        };
        let excess = |computed: &Array2<f64>, band: &Array2<f64>, norm: &NativeNorm, row: usize| {
            let exact = reference(norm, row);
            (0..LAYER_WIDTH)
                .filter(|&column| {
                    (quad(computed[[row, column]]) - exact[column]).0.abs() > band[[row, column]]
                })
                .count()
        };
        let norms = [
            NativeNorm::Rms {
                epsilon: RMS_EPSILON,
                gain: gain.clone(),
            },
            NativeNorm::Layer {
                epsilon: LAYER_EPSILON,
                gain: gain.clone(),
                bias: bias.clone(),
            },
        ];
        for norm in &norms {
            let (normalized, band) = norm.apply_with_band(rows.view()).expect("finite rows");
            let owner = norm.as_masked_norm().apply(rows.view()).expect("finite rows");
            assert!(bits(&normalized) == bits(&owner), "the band's rows must be the owner's normalized rows");
            for row in 0..LAYER_TOKENS {
                assert_eq!(
                    excess(&normalized, &band, norm, row),
                    0,
                    "{norm:?}: row {row} left its rounding band around the double-double normalization"
                );
            }
        }

        // Positive control: the LayerNorm row's own rounding at the computed centred row,
        // without the centring term, does not cover the near-constant row.
        let layer = &norms[1];
        let normalized = layer.apply_with_band(rows.view()).expect("finite rows").0;
        let own_growth = accumulation_growth(LAYER_WIDTH + 7);
        let own_only = Array2::from_shape_fn(normalized.dim(), |(row, column)| {
            own_growth * (normalized[[row, column]].abs() + bias[column].abs()) / (1.0 - own_growth)
        });
        assert!(
            excess(&normalized, &own_only, layer, near_constant) > 0,
            "positive control: the centring error must exceed the norm's own rounding on the near-constant row"
        );

        // Refusal: with zero epsilon, a row within a few ulps of a constant has a centring error
        // above a quarter of its norm, and no slope bound.
        let mut flat = Array2::from_elem((1, LAYER_WIDTH), 3.0);
        for (column, slot) in flat.row_mut(0).iter_mut().enumerate() {
            *slot += column as f64 * 2.0_f64.powi(-51);
        }
        let zero_epsilon = NativeNorm::Layer {
            epsilon: 0.0,
            gain: gain.clone(),
            bias: bias.clone(),
        };
        assert!(
            matches!(zero_epsilon.apply_with_band(flat.view()), Err(BlockError::NormBandUnbounded { row: 0 })),
            "a zero-epsilon row near a constant must be refused"
        );
        assert!(
            zero_epsilon.apply_with_band(rows.slice(ndarray::s![0..1, ..])).is_ok(),
            "control: a zero-epsilon row far from a constant has a band"
        );
    }

    /// An RMSNorm row whose product `x ν̂` falls in the subnormal range rounds it by an
    /// absolute `2^-1075`, not relatively: `[1000, 1e-310]` with gain `[1, 1e10]` has
    /// `x ν̂ ≈ 1.4e-313`, whose rounding the gain lifts to about `9e-315` on a normal output
    /// of `1.4e-303`. The band covers the double-double normalization there. Positive
    /// control: the relative band alone, `γ_(d+6) |ŷ| / (1 − γ_(d+6))`, about `1e-318`, is
    /// exceeded. A zero-epsilon row whose mean square is the smallest subnormal has no
    /// band and is refused, typed, though the owner normalizes it.
    #[test]
    fn a_normalization_band_covers_a_row_whose_products_underflow() {
        use qd::Quad;
        let quad = Quad::from_f64;
        let rows = ndarray::array![[1000.0, 1.0e-310]];
        let gain = ndarray::array![1.0, 1.0e10];
        let norm = NativeNorm::Rms {
            epsilon: RMS_EPSILON,
            gain: gain.clone(),
        };
        let (normalized, band) = norm.apply_with_band(rows.view()).expect("finite rows");
        let square = quad(rows[[0, 0]]) * quad(rows[[0, 0]]) + quad(rows[[0, 1]]) * quad(rows[[0, 1]]);
        let inverse_root = quad(1.0) / (square / quad(2.0) + quad(RMS_EPSILON)).sqrt();
        // `w x` is normal, so the reference's products keep their accuracy.
        let exact = quad(gain[1]) * quad(rows[[0, 1]]) * inverse_root;
        let error = (quad(normalized[[0, 1]]) - exact).0.abs();
        assert!(
            error <= band[[0, 1]],
            "the band {:e} must cover the underflowed product's error {error:e}",
            band[[0, 1]]
        );
        let growth = accumulation_growth(2 + 6);
        let relative_only = growth * normalized[[0, 1]].abs() / (1.0 - growth);
        assert!(
            error > relative_only,
            "positive control: the relative band {relative_only:e} must not cover the error {error:e}"
        );

        let unresolved = ndarray::array![[2.0e-162, 2.0e-162]];
        let zero_epsilon = NativeNorm::Rms {
            epsilon: 0.0,
            gain: ndarray::array![1.0, 1.0],
        };
        assert!(
            zero_epsilon.as_masked_norm().apply(unresolved.view()).is_ok(),
            "control: the owner normalizes a row whose mean square is the smallest subnormal"
        );
        assert!(
            matches!(
                zero_epsilon.apply_with_band(unresolved.view()),
                Err(BlockError::NormBandUnbounded { row: 0 })
            ),
            "a zero-epsilon row whose mean square is not resolved above the underflow must be refused"
        );
    }

    /// Shapes are refused by the owner that reads them, typed by sublayer.
    #[test]
    fn a_block_refuses_a_normalization_of_another_width_and_positions_of_another_length() {
        let fixture = Fixture::new(ResidualLayout::Sequential, GaussianActivation::Relu, 2967);
        let masks = FixtureMasks::ones();
        let attention_masks = masks.attention();
        assert!(
            fixture
                .component_block()
                .execute(masks.block(&attention_masks), fixture.residual.view(), &fixture.positions)
                .is_ok(),
            "positive control: the fixture block executes"
        );

        let narrow_positions = &fixture.positions[..TOKENS - 1];
        assert!(
            matches!(
                fixture
                    .component_block()
                    .execute(masks.block(&attention_masks), fixture.residual.view(), narrow_positions),
                Err(BlockError::Attention(AttentionProgramError::Shape { tensor: "input", .. }))
            ),
            "positions of another length must be refused by the attention sublayer"
        );

        let wide = Fixture {
            mlp_gain: Array1::ones(WIDTH + 1),
            ..Fixture::new(ResidualLayout::Sequential, GaussianActivation::Relu, 2967)
        };
        assert!(
            matches!(
                wide.component_block()
                    .execute(masks.block(&attention_masks), wide.residual.view(), &wide.positions),
                Err(BlockError::Norm {
                    sublayer: Sublayer::Mlp,
                    error: GatedRewriteError::ShapeMismatch { .. },
                })
            ),
            "an MLP normalization gain of another width must be refused, typed by sublayer"
        );
    }

    const LAYER_WIDTH: usize = 8;
    const LAYER_HEAD_DIM: usize = 4;
    const LAYER_TOKENS: usize = 6;
    const LAYER_COMPONENTS: usize = 9;
    /// `1/sqrt(head_dim)` at `head_dim = 4`, exactly.
    const LAYER_SCORE_SCALE: f64 = 0.5;

    fn layer_geometry() -> AttentionGeometry {
        AttentionGeometry {
            model_dim: LAYER_WIDTH,
            n_heads: 2,
            n_kv_heads: 2,
            head_dim: LAYER_HEAD_DIM,
        }
    }

    /// Learned absolute positions: the source rotates no plane.
    fn no_rotary() -> RotaryEmbedding {
        RotaryEmbedding {
            pairing: RotaryPairing::HalfSplit,
            inverse_frequencies: Vec::new(),
            attention_scaling: 1.0,
        }
    }

    const PROJECTIONS: [AttentionProjection; 4] = [
        AttentionProjection::Query,
        AttentionProjection::Key,
        AttentionProjection::Value,
        AttentionProjection::Output,
    ];

    /// An attention-only layer whose four weights factor exactly through overcomplete
    /// reads in eighths, `W = N R`, so each exact write is its candidate and every edited
    /// tensor `U diag(m) R` is formed without rounding.
    struct LayerFixture {
        candidates: [Array2<f64>; 4],
        reads: [Array2<f64>; 4],
        residual: Array2<f64>,
        positions: Vec<i64>,
    }

    impl LayerFixture {
        fn new(seed: u64) -> Self {
            let mut rng = StdRng::seed_from_u64(seed);
            let candidates = [
                eighths(&mut rng, LAYER_WIDTH, LAYER_COMPONENTS),
                eighths(&mut rng, LAYER_WIDTH, LAYER_COMPONENTS),
                eighths(&mut rng, LAYER_WIDTH, LAYER_COMPONENTS),
                eighths(&mut rng, LAYER_WIDTH, LAYER_COMPONENTS),
            ];
            let reads = [
                eighths(&mut rng, LAYER_COMPONENTS, LAYER_WIDTH),
                eighths(&mut rng, LAYER_COMPONENTS, LAYER_WIDTH),
                eighths(&mut rng, LAYER_COMPONENTS, LAYER_WIDTH),
                eighths(&mut rng, LAYER_COMPONENTS, LAYER_WIDTH),
            ];
            Self {
                candidates,
                reads,
                // Thirds of eighths in `[-2, 2]`. With eighth-valued rows every linear read is exact
                // in f64, so two routes with the same algebra would share their bits: the bitwise
                // checks could not see a route change, and the derived bands would never meet rounding.
                residual: Array2::from_shape_simple_fn((LAYER_TOKENS, LAYER_WIDTH), || {
                    rng.random_range(-48..=48) as f64 / 24.0
                }),
                positions: (0..LAYER_TOKENS as i64).collect(),
            }
        }

        fn native_with(&self, weights: [Array2<f64>; 4]) -> NativeAttentionLayer {
            let [query, key, value, output] = weights;
            NativeAttentionLayer::new(
                layer_geometry(),
                no_rotary(),
                LAYER_SCORE_SCALE,
                query,
                key,
                value,
                output,
            )
            .expect("fixture weights match the geometry")
        }

        /// The source layer: every weight is its factors' product.
        fn native(&self) -> NativeAttentionLayer {
            self.native_with([0, 1, 2, 3].map(|index| self.candidates[index].dot(&self.reads[index])))
        }

        fn component(&self) -> ComponentAttentionLayer {
            let read = |index: usize| ComponentRead {
                read: self.reads[index].view(),
                candidate_write: self.candidates[index].view(),
            };
            ComponentAttentionLayer::new(self.native(), read(0), read(1), read(2), read(3))
                .expect("random overcomplete reads are resolved")
        }

        /// Direct execution with the edited tensors `U diag(m) R`.
        fn edited(&self, layer: &ComponentAttentionLayer, masks: &LayerMasks) -> NativeAttentionLayer {
            self.native_with(PROJECTIONS.map(|projection| {
                let factor = layer.factor(projection);
                masked_product(factor.write(), masks.of(projection), factor.read())
            }))
        }
    }

    #[derive(Clone)]
    struct LayerMasks {
        query: Array1<f64>,
        key: Array1<f64>,
        value: Array1<f64>,
        output: Array1<f64>,
    }

    impl LayerMasks {
        /// Eighths: continuous in `[0, 1]`, binary, and signed in `[-3/2, 3/2]`.
        fn family(rng: &mut StdRng) -> [(&'static str, Self); 3] {
            let mut draw = |low: i32, high: i32, denominator: f64| {
                let mut vector = || {
                    Array1::from_shape_simple_fn(LAYER_COMPONENTS, || rng.random_range(low..=high) as f64 / denominator)
                };
                Self {
                    query: vector(),
                    key: vector(),
                    value: vector(),
                    output: vector(),
                }
            };
            [
                ("continuous", draw(0, 8, 8.0)),
                ("binary", draw(0, 1, 1.0)),
                ("signed", draw(-12, 12, 8.0)),
            ]
        }

        fn of(&self, projection: AttentionProjection) -> ArrayView1<'_, f64> {
            match projection {
                AttentionProjection::Query => self.query.view(),
                AttentionProjection::Key => self.key.view(),
                AttentionProjection::Value => self.value.view(),
                AttentionProjection::Output => self.output.view(),
            }
        }

        fn reads(&self) -> AttentionLayerReads<'_> {
            AttentionLayerReads {
                query: ProjectionRead::Components(ComponentMasks::uniform(self.query.view())),
                key: ProjectionRead::Components(ComponentMasks::uniform(self.key.view())),
                value: ProjectionRead::Components(ComponentMasks::uniform(self.value.view())),
                output: ProjectionRead::Components(ComponentMasks::uniform(self.output.view())),
            }
        }
    }

    /// Each stage of two executions with its radius, paired, for the checks that two routes
    /// to one exact layer agree within the sum of their radii.
    fn stage_radii<'a>(
        left: &'a AttentionLayerExecution,
        right: &'a AttentionLayerExecution,
    ) -> Vec<(&'static str, (&'a Array2<f64>, &'a Array2<f64>), (&'a Array2<f64>, &'a Array2<f64>))> {
        let stages = |execution: &'a AttentionLayerExecution| {
            [
                ("queries", (&*execution.queries, &execution.query_radius)),
                ("keys", (&*execution.keys, &execution.key_radius)),
                ("values", (&*execution.values, &execution.value_radius)),
                ("mixed rows", (&execution.attention.mixed, &execution.attention.mixed_radius)),
                ("write", (&*execution.write, &execution.write_radius)),
                ("output", (&execution.output, &execution.output_radius)),
            ]
        };
        let (first, second) = (stages(left), stages(right));
        (0..first.len()).map(|stage| (first[stage].0, first[stage].1, second[stage].1)).collect()
    }

    fn layer_stage_bits(execution: &AttentionLayerExecution) -> Vec<u64> {
        execution
            .queries
            .iter()
            .chain(execution.keys.iter())
            .chain(execution.values.iter())
            .chain(execution.attention.weights.iter())
            .chain(execution.attention.mixed.iter())
            .chain(execution.write.iter())
            .chain(execution.output.iter())
            .map(|value| value.to_bits())
            .collect()
    }

    fn row_bits(values: &Array2<f64>, row: usize) -> Vec<u64> {
        values.row(row).iter().map(|value| value.to_bits()).collect()
    }

    /// A1 for an attention-only layer: for continuous, binary and signed masks on Q, K, V and
    /// O, the component layer equals direct execution with the edited tensors within the sum of
    /// the two routes' derived bands.
    #[test]
    fn a_masked_attention_layer_equals_the_edited_tensor_layer_within_the_derived_band() {
        let fixture = LayerFixture::new(2981);
        let layer = fixture.component();
        for projection in PROJECTIONS {
            assert_eq!(
                layer.factor(projection).write(),
                fixture.candidates[projection.index()].view(),
                "the {projection} weight factors exactly through its candidate, so the edited tensors carry no rounding"
            );
        }
        let mut rng = StdRng::seed_from_u64(2982);
        for (kind, masks) in LayerMasks::family(&mut rng) {
            let component = layer
                .execute(masks.reads(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
                .expect("masked layer");
            let edited = fixture
                .edited(&layer, &masks)
                .execute(AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
                .expect("edited-tensor layer");
            // Both routes compute the same exact layer, so at every stage they differ by at most
            // the sum of their production radii.
            for (stage, (left, left_radius), (right, right_radius)) in stage_radii(&component, &edited) {
                assert_eq!(
                    violations(left, right, &sum_of_bounds(left_radius.clone(), right_radius.clone())),
                    0,
                    "{kind} masks: the component layer's {stage} left the sum of the two routes' radii"
                );
            }

            // Positive controls: a value mask and an output mask moved by 1e-9 each leave the
            // output band, and the value move leaves the band at the values stage already.
            let mut moved_value = masks.clone();
            moved_value.value[2] += 1.0e-9;
            let mut moved_output = masks.clone();
            moved_output.output[5] += 1.0e-9;
            for (control, moved) in [("value 1e-9", moved_value), ("output 1e-9", moved_output)] {
                let perturbed = fixture
                    .edited(&layer, &moved)
                    .execute(AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
                    .expect("perturbed edited-tensor layer");
                let band = sum_of_bounds(component.output_radius.clone(), perturbed.output_radius.clone());
                assert!(
                    violations(&component.output, &perturbed.output, &band) > 0,
                    "{kind} masks: the output radii must resolve a {control} mask move"
                );
                if control.starts_with("value") {
                    let band = sum_of_bounds(component.value_radius.clone(), perturbed.value_radius.clone());
                    assert!(
                        violations(&component.values, &perturbed.values, &band) > 0,
                        "{kind} masks: the value radii must resolve a {control} mask move"
                    );
                }
            }
        }
    }

    /// All-on reads run every projection on its stored tensor, so each stage is bit-identical
    /// to the native layer's.
    #[test]
    fn all_on_reads_execute_the_native_attention_layer_bit_for_bit() {
        let fixture = LayerFixture::new(2983);
        let layer = fixture.component();
        let native = fixture
            .native()
            .execute(AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
            .expect("native layer");
        let all_on = layer
            .execute(AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
            .expect("all-on component layer");
        assert!(
            layer_stage_bits(&all_on) == layer_stage_bits(&native),
            "all-on reads must execute the stored tensors on their original paths"
        );

        // Positive control: the factored all-ones value read is algebraically the same layer but
        // not the same bits.
        let ones = Array1::<f64>::ones(LAYER_COMPONENTS);
        let factored = layer
            .execute(
                AttentionLayerReads {
                    value: ProjectionRead::Components(ComponentMasks::uniform(ones.view())),
                    ..AttentionLayerReads::native()
                },
                ProjectedRows::exact(fixture.residual.view()),
                &fixture.positions,
            )
            .expect("all-ones factored value read");
        assert!(
            layer_stage_bits(&factored) != layer_stage_bits(&native),
            "the bit-identity check must distinguish the factored all-ones value path"
        );
    }

    /// The occurrence test's shape: head 1 of layer 0 ablated in the output write, `Δ = −W_O[:,
    /// 4..8] e_{4..7}ᵀ`. Scoped at one declared row, it moves exactly that row of layer 0's write,
    /// every earlier logits row of a 2-layer stack keeps its bits, and the declared row equals
    /// the dense edited layer within the derived band. Scoped at every position, every row
    /// does.
    #[test]
    fn a_scoped_output_edit_moves_exactly_its_declared_rows_through_a_two_layer_stack() {
        let (first, second) = (LayerFixture::new(2984), LayerFixture::new(2985));
        let (layer0, layer1) = (first.native(), second.native());
        let mut rng = StdRng::seed_from_u64(2986);
        let unembed = eighths(&mut rng, 5, LAYER_WIDTH);
        let head_columns: Vec<usize> = (LAYER_HEAD_DIM..2 * LAYER_HEAD_DIM).collect();
        let output_weight = layer0.weight(AttentionProjection::Output).to_owned();
        let left = output_weight.select(Axis(1), &head_columns).mapv(|value| -value);
        let mut right = Array2::<f64>::zeros((LAYER_WIDTH, LAYER_HEAD_DIM));
        for (term, &column) in head_columns.iter().enumerate() {
            right[[column, term]] = 1.0;
        }
        let edit = FactorView::new(left.view(), right.view()).expect("finite edit factors");
        let stack = |reads: AttentionLayerReads<'_>| {
            let hidden = layer0
                .execute(reads, ProjectedRows::exact(first.residual.view()), &first.positions)
                .expect("layer 0");
            // Layer 1 reads layer 0's output with its radius, as a stack does.
            let top = layer1
                .execute(
                    AttentionLayerReads::native(),
                    ProjectedRows {
                        values: hidden.output.view(),
                        radius: hidden.output_radius.view(),
                    },
                    &first.positions,
                )
                .expect("layer 1");
            let logits = native_linear(unembed.view(), top.output.view())
                .expect("unembedding")
                .to_owned();
            (hidden, logits)
        };
        let (native_hidden, native_logits) = stack(AttentionLayerReads::native());
        let declared = 3_usize;
        let scope = PositionScope::declared(vec![declared]).expect("one declared position");
        let (scoped_hidden, scoped_logits) = stack(AttentionLayerReads {
            output: ProjectionRead::Edited(ScopedEdit {
                edit,
                positions: &scope,
            }),
            ..AttentionLayerReads::native()
        });
        for row in 0..LAYER_TOKENS {
            assert_eq!(
                row_bits(&native_hidden.write, row) == row_bits(&scoped_hidden.write, row),
                row != declared,
                "row {row}: an output edit scoped at {declared} must move exactly its declared row of the write"
            );
            if row < declared {
                assert!(
                    row_bits(&native_logits, row) == row_bits(&scoped_logits, row),
                    "logits row {row} precedes the edited row, so it must keep its bits"
                );
            }
        }
        assert!(
            row_bits(&native_logits, declared) != row_bits(&scoped_logits, declared),
            "positive control: the logits at the declared row move"
        );

        // The dense edited layer: `W_O + left rightᵀ` zeroes head 1's columns exactly.
        let ablated = &output_weight + &left.dot(&right.t());
        let dense_layer = first.native_with([
            layer0.weight(AttentionProjection::Query).to_owned(),
            layer0.weight(AttentionProjection::Key).to_owned(),
            layer0.weight(AttentionProjection::Value).to_owned(),
            ablated.clone(),
        ]);
        let dense = dense_layer
            .execute(AttentionLayerReads::native(), ProjectedRows::exact(first.residual.view()), &first.positions)
            .expect("dense edited layer");
        assert!(
            bits(&dense.attention.mixed) == bits(&scoped_hidden.attention.mixed),
            "the edit reaches only the output read, so the mixed rows keep their bits"
        );
        // Both routes reach the same exact write at the same mixed rows, so they differ by at
        // most the sum of their production write radii: the scoped route's `W_O x` plus the
        // terms `left (rightᵀ x)` on the declared row, the dense route's `W_O' x`.
        let band = sum_of_bounds(scoped_hidden.write_radius.clone(), dense.write_radius.clone());
        let row_violations = |left_rows: &Array2<f64>, right_rows: &Array2<f64>, row: usize| {
            (0..LAYER_WIDTH)
                .filter(|&column| (left_rows[[row, column]] - right_rows[[row, column]]).abs() > band[[row, column]])
                .count()
        };
        assert_eq!(
            row_violations(&scoped_hidden.write, &dense.write, declared),
            0,
            "the declared row of the scoped write must equal the dense edited layer within the derived band"
        );
        assert!(
            row_violations(&native_hidden.write, &dense.write, declared) > 0,
            "positive control: the unedited write at the declared row must leave the band"
        );

        let every = PositionScope::every();
        let global = layer0
            .execute(
                AttentionLayerReads {
                    output: ProjectionRead::Edited(ScopedEdit {
                        edit,
                        positions: &every,
                    }),
                    ..AttentionLayerReads::native()
                },
                ProjectedRows::exact(first.residual.view()),
                &first.positions,
            )
            .expect("globally edited layer 0");
        assert_eq!(
            violations(
                &global.write,
                &dense.write,
                &sum_of_bounds(global.write_radius.clone(), dense.write_radius.clone())
            ),
            0,
            "an output edit at every position must equal the dense edited layer within the sum of the write radii"
        );
    }

    /// Reads outside the layer's domain are refused, typed by projection.
    #[test]
    fn an_attention_layer_refuses_reads_outside_its_domain() {
        let fixture = LayerFixture::new(2987);
        let native = fixture.native();
        let layer = fixture.component();
        assert!(
            native
                .execute(AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
                .is_ok(),
            "positive control: the fixture layer executes"
        );

        let ones = Array1::<f64>::ones(LAYER_COMPONENTS);
        assert!(
            matches!(
                native.execute(
                    AttentionLayerReads {
                        value: ProjectionRead::Components(ComponentMasks::uniform(ones.view())),
                        ..AttentionLayerReads::native()
                    },
                    ProjectedRows::exact(fixture.residual.view()),
                    &fixture.positions,
                ),
                Err(BlockError::NoComponentFactors {
                    projection: AttentionProjection::Value
                })
            ),
            "a component read on a layer without factors must be refused"
        );

        let short = Array1::<f64>::ones(LAYER_COMPONENTS - 1);
        assert!(
            matches!(
                layer.execute(
                    AttentionLayerReads {
                        key: ProjectionRead::Components(ComponentMasks::uniform(short.view())),
                        ..AttentionLayerReads::native()
                    },
                    ProjectedRows::exact(fixture.residual.view()),
                    &fixture.positions,
                ),
                Err(BlockError::Apply {
                    projection: AttentionProjection::Key,
                    error: ApplyError::Shape { .. }
                })
            ),
            "a mask of another length must be refused by the matrix-free read"
        );

        let (left, right) = (Array2::<f64>::ones((LAYER_WIDTH, 1)), Array2::<f64>::ones((LAYER_WIDTH, 1)));
        let edit = FactorView::new(left.view(), right.view()).expect("finite edit factors");
        let absent = PositionScope::declared(vec![LAYER_TOKENS]).expect("one declared position");
        assert!(
            matches!(
                native.execute(
                    AttentionLayerReads {
                        output: ProjectionRead::Edited(ScopedEdit {
                            edit,
                            positions: &absent,
                        }),
                        ..AttentionLayerReads::native()
                    },
                    ProjectedRows::exact(fixture.residual.view()),
                    &fixture.positions,
                ),
                Err(BlockError::EditPositionAbsent {
                    projection: AttentionProjection::Output,
                    position,
                }) if position == LAYER_TOKENS
            ),
            "an edit declaring a position no row holds must be refused"
        );

        let first = PositionScope::declared(vec![0]).expect("one declared position");
        let shifted: Vec<i64> = fixture.positions.iter().map(|position| position - 1).collect();
        assert!(
            matches!(
                native.execute(
                    AttentionLayerReads {
                        output: ProjectionRead::Edited(ScopedEdit {
                            edit,
                            positions: &first,
                        }),
                        ..AttentionLayerReads::native()
                    },
                    ProjectedRows::exact(fixture.residual.view()),
                    &shifted,
                ),
                Err(BlockError::NegativePosition {
                    projection: AttentionProjection::Output,
                    position: -1
                })
            ),
            "an edited read at a negative position must be refused"
        );

        let [query, key, value, output] =
            [0, 1, 2, 3].map(|index| fixture.candidates[index].dot(&fixture.reads[index]));
        assert!(
            matches!(
                NativeAttentionLayer::new(
                    layer_geometry(),
                    no_rotary(),
                    LAYER_SCORE_SCALE,
                    query,
                    key,
                    value,
                    output.select(Axis(1), &[0, 1, 2]),
                ),
                Err(BlockError::ProjectionShape {
                    projection: AttentionProjection::Output,
                    ..
                })
            ),
            "an output weight of another shape must be refused at construction"
        );
        assert_eq!(output.dim(), (LAYER_WIDTH, LAYER_WIDTH), "control: the fixture weight has the geometry's shape");

        // The pub read band is the radius an execution carries, and it refuses what the read
        // refuses.
        let components = layer
            .execute(
                AttentionLayerReads {
                    query: ProjectionRead::Components(ComponentMasks::uniform(ones.view())),
                    ..AttentionLayerReads::native()
                },
                ProjectedRows::exact(fixture.residual.view()),
                &fixture.positions,
            )
            .expect("component query read");
        let band = layer
            .read_band(
                AttentionProjection::Query,
                ProjectionRead::Components(ComponentMasks::uniform(ones.view())),
                fixture.residual.view(),
                &fixture.positions,
            )
            .expect("component query band");
        assert!(
            bits(&band) == bits(&components.query_radius),
            "the execution's query radius must be the pub read band"
        );
        assert!(
            matches!(
                native.read_band(
                    AttentionProjection::Query,
                    ProjectionRead::Components(ComponentMasks::uniform(ones.view())),
                    fixture.residual.view(),
                    &fixture.positions,
                ),
                Err(BlockError::NoComponentFactors {
                    projection: AttentionProjection::Query
                })
            ),
            "a component band on a layer without factors must be refused"
        );
        assert!(
            matches!(
                layer.read_band(
                    AttentionProjection::Key,
                    ProjectionRead::Components(ComponentMasks::uniform(short.view())),
                    fixture.residual.view(),
                    &fixture.positions,
                ),
                Err(BlockError::Apply {
                    projection: AttentionProjection::Key,
                    error: ApplyError::Shape { operand: "band component mask", .. }
                })
            ),
            "a band mask of another length must be refused"
        );
        let narrow = fixture.residual.select(Axis(1), &[0, 1, 2]);
        assert!(
            matches!(
                native.read_band(
                    AttentionProjection::Value,
                    ProjectionRead::Native,
                    narrow.view(),
                    &fixture.positions,
                ),
                Err(BlockError::Apply {
                    projection: AttentionProjection::Value,
                    error: ApplyError::Shape { operand: "band input rows", .. }
                })
            ),
            "band rows of another width must be refused"
        );
    }

    /// Every stage radius of one layer execution, in stage order.
    fn layer_radius_bits(execution: &AttentionLayerExecution) -> Vec<u64> {
        execution
            .query_radius
            .iter()
            .chain(execution.key_radius.iter())
            .chain(execution.value_radius.iter())
            .chain(execution.attention.score_radius.iter())
            .chain(execution.attention.weight_radius.iter())
            .chain(execution.attention.mixed_radius.iter())
            .chain(execution.write_radius.iter())
            .chain(execution.output_radius.iter())
            .map(|value| value.to_bits())
            .collect()
    }

    /// The layer fixture's reads for one run of the radius tests: native reads on the native
    /// layer, then continuous, binary and signed component masks on the component layer.
    fn radius_cases(seed: u64) -> Vec<(&'static str, bool, LayerMasks)> {
        let mut rng = StdRng::seed_from_u64(seed);
        let ones = Array1::<f64>::ones(LAYER_COMPONENTS);
        let native = LayerMasks {
            query: ones.clone(),
            key: ones.clone(),
            value: ones.clone(),
            output: ones,
        };
        let mut cases = vec![("native reads", false, native)];
        cases.extend(LayerMasks::family(&mut rng).into_iter().map(|(kind, masks)| (kind, true, masks)));
        cases
    }

    /// Exact residual rows carry nothing: every stage and every radius is bit-identical
    /// whether the zero radius is `ProjectedRows::exact`'s zero-stride view or a dense array
    /// of zeros, each query, key and value radius is the pub read band, the write radius is the
    /// output band plus `|A_O|` times the mixed radius, and the output radius is the write
    /// radius plus the residual addition's rounding: the radii an execution carried before the
    /// residual rows had a radius.
    #[test]
    fn exact_residual_rows_carry_only_the_rounding_bands_bit_for_bit() {
        let fixture = LayerFixture::new(2988);
        let (native, component) = (fixture.native(), fixture.component());
        let zeros = Array2::<f64>::zeros(fixture.residual.dim());
        for (kind, factored, masks) in radius_cases(2989) {
            let reads = if factored { masks.reads() } else { AttentionLayerReads::native() };
            let run = |rows: ProjectedRows<'_>| {
                let executed = if factored {
                    component.execute(reads, rows, &fixture.positions)
                } else {
                    native.execute(reads, rows, &fixture.positions)
                };
                executed.expect("fixture layer")
            };
            let exact = run(ProjectedRows::exact(fixture.residual.view()));
            let dense = run(ProjectedRows {
                values: fixture.residual.view(),
                radius: zeros.view(),
            });
            assert!(
                layer_stage_bits(&exact) == layer_stage_bits(&dense)
                    && layer_radius_bits(&exact) == layer_radius_bits(&dense),
                "{kind}: a dense zero radius must execute as exact rows, bit for bit"
            );
            let band = |projection: AttentionProjection, read: ProjectionRead<'_>, rows: ArrayView2<'_, f64>| {
                let banded = if factored {
                    component.read_band(projection, read, rows, &fixture.positions)
                } else {
                    native.read_band(projection, read, rows, &fixture.positions)
                };
                banded.expect("fixture band")
            };
            for (projection, read, radius) in [
                (AttentionProjection::Query, reads.query, &exact.query_radius),
                (AttentionProjection::Key, reads.key, &exact.key_radius),
                (AttentionProjection::Value, reads.value, &exact.value_radius),
            ] {
                assert!(
                    bits(&band(projection, read, fixture.residual.view())) == bits(radius),
                    "{kind}: the {projection} radius of exact rows must be the read band itself"
                );
            }
            let output_factor = if factored {
                Some(&component.factors[AttentionProjection::Output.index()])
            } else {
                None
            };
            let carried = read_magnitude(
                native.weight(AttentionProjection::Output),
                output_factor,
                AttentionProjection::Output,
                reads.output,
                exact.attention.mixed_radius.view(),
                &fixture.positions,
            )
            .expect("mixed-radius magnitude");
            let write_radius = sum_of_bounds(
                band(AttentionProjection::Output, reads.output, exact.attention.mixed.view()),
                carried.dominated(),
            );
            assert!(
                bits(&write_radius) == bits(&exact.write_radius),
                "{kind}: the write radius must be the output band plus |A_O| times the mixed radius"
            );
            let addition = exact.output.mapv(|value| UNIT_ROUNDOFF * value.abs() / (1.0 - UNIT_ROUNDOFF));
            assert!(
                bits(&sum_of_bounds(write_radius, addition)) == bits(&exact.output_radius),
                "{kind}: the output radius of exact rows must be the write radius plus the addition's rounding"
            );
        }
    }

    /// Residual rows `x̂` within `r` of exact rows `x*` carry `r` through every stage. The
    /// execution at `x̂` with radius `r` and the execution at `x*` taken as exact both enclose
    /// the exact layer at `x*`, so at every stage they differ by at most the sum of their radii.
    /// Positive control, the mutant that drops the input radius: `x̂` executed as exact rows
    /// leaves that sum at the reads and at the output. The same holds through a second layer
    /// that reads the first layer's output with its radius, and for the unembedding read.
    #[test]
    fn a_residual_radius_carries_through_every_stage_of_a_layer_stack() {
        let (fixture, upper) = (LayerFixture::new(2990), LayerFixture::new(2991));
        let (native, component, top) = (fixture.native(), fixture.component(), upper.native());
        let mut rng = StdRng::seed_from_u64(2992);
        let unembed = eighths(&mut rng, 5, LAYER_WIDTH);
        // `|fl(x* + δ) − x*| ≤ |δ| + u|x* + δ| < 2|δ|` for `|δ| = 2^-30` and `|x*| ≤ 2`, so `2|δ|`
        // is a radius of `x̂` against `x*`.
        let step = 2.0_f64.powi(-30);
        let moved = fixture
            .residual
            .mapv(|value| if rng.random_range(0..2) == 0 { value + step } else { value - step });
        let radius = Array2::<f64>::from_elem(fixture.residual.dim(), 2.0 * step);
        let carried_rows = ProjectedRows {
            values: moved.view(),
            radius: radius.view(),
        };
        for (kind, factored, masks) in radius_cases(2993) {
            let reads = if factored { masks.reads() } else { AttentionLayerReads::native() };
            let run = |rows: ProjectedRows<'_>| {
                let executed = if factored {
                    component.execute(reads, rows, &fixture.positions)
                } else {
                    native.execute(reads, rows, &fixture.positions)
                };
                executed.expect("fixture layer")
            };
            let carried = run(carried_rows);
            let reference = run(ProjectedRows::exact(fixture.residual.view()));
            let dropped = run(ProjectedRows::exact(moved.view()));
            for (stage, (left, left_radius), (right, right_radius)) in stage_radii(&carried, &reference) {
                assert_eq!(
                    violations(left, right, &sum_of_bounds(left_radius.clone(), right_radius.clone())),
                    0,
                    "{kind}: the {stage} of rows within r of the exact rows must stay within the sum of the radii"
                );
            }
            for (stage, (left, left_radius), (right, right_radius)) in stage_radii(&dropped, &reference) {
                if stage == "queries" || stage == "output" {
                    assert!(
                        violations(left, right, &sum_of_bounds(left_radius.clone(), right_radius.clone())) > 0,
                        "{kind}: positive control: without the input radius the {stage} must leave the sum of the radii"
                    );
                }
            }

            // A second layer reads the first layer's output with its radius, and the unembedding
            // reads the second layer's.
            let stacked = |first: &AttentionLayerExecution| {
                let second = top
                    .execute(
                        AttentionLayerReads::native(),
                        ProjectedRows {
                            values: first.output.view(),
                            radius: first.output_radius.view(),
                        },
                        &fixture.positions,
                    )
                    .expect("second layer");
                let (logits, logit_radius) = linear_read(
                    unembed.view(),
                    ProjectedRows {
                        values: second.output.view(),
                        radius: second.output_radius.view(),
                    },
                )
                .expect("unembedding");
                ((*logits).to_owned(), logit_radius)
            };
            let ((logits, logit_radius), (reference_logits, reference_radius)) =
                (stacked(&carried), stacked(&reference));
            assert_eq!(
                violations(&logits, &reference_logits, &sum_of_bounds(logit_radius, reference_radius)),
                0,
                "{kind}: the logits of rows within r of the exact rows must stay within the sum of the radii"
            );
        }
    }

    /// A layer that writes nothing passes its residual's radius on to its output. With a zero output
    /// weight the write is exactly `0` with radius `0`, so the output is the residual plus one exact
    /// addition, and only the residual's own radius can cover rows moved within it. Rows within
    /// `r = 2^-29` of the exact ones must stay within the sum of the output radii, and every output
    /// radius must be at least `r`. That isolates the output radius's residual term, which the stack
    /// test above cannot see behind the write's carried radius.
    #[test]
    fn a_layer_that_writes_nothing_passes_its_residual_radius_to_its_output() {
        let fixture = LayerFixture::new(2995);
        let mut weights = [0, 1, 2, 3].map(|index| fixture.candidates[index].dot(&fixture.reads[index]));
        weights[AttentionProjection::Output.index()].fill(0.0);
        let silent = fixture.native_with(weights);
        let mut rng = StdRng::seed_from_u64(2996);
        let step = 2.0_f64.powi(-30);
        let moved = fixture
            .residual
            .mapv(|value| if rng.random_range(0..2) == 0 { value + step } else { value - step });
        let radius = Array2::<f64>::from_elem(fixture.residual.dim(), 2.0 * step);
        let carried = silent
            .execute(
                AttentionLayerReads::native(),
                ProjectedRows {
                    values: moved.view(),
                    radius: radius.view(),
                },
                &fixture.positions,
            )
            .expect("silent layer");
        let reference = silent
            .execute(AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
            .expect("silent layer");
        assert!(carried.write.iter().all(|&value| value == 0.0), "a zero output weight writes exactly zero");
        assert!(
            carried.output_radius.iter().all(|&bound| bound >= 2.0 * step),
            "the output radius must carry the residual's own radius"
        );
        assert_eq!(
            violations(
                &carried.output,
                &reference.output,
                &sum_of_bounds(carried.output_radius.clone(), reference.output_radius.clone())
            ),
            0,
            "rows within r of the exact rows must give an output within the sum of the radii"
        );
    }

    /// The unembedding read and a layer's native read are one route: `linear_read` returns a
    /// native layer's queries and query radius bit for bit, for exact rows and for rows with a
    /// radius, and refuses a radius of another shape.
    #[test]
    fn linear_read_is_the_layers_native_read() {
        let fixture = LayerFixture::new(2994);
        let native = fixture.native();
        let radius = fixture.residual.mapv(|value| value.abs() * 2.0_f64.powi(-20));
        for rows in [
            ProjectedRows::exact(fixture.residual.view()),
            ProjectedRows {
                values: fixture.residual.view(),
                radius: radius.view(),
            },
        ] {
            let layer = native
                .execute(AttentionLayerReads::native(), rows, &fixture.positions)
                .expect("native layer");
            let (read, read_radius) =
                linear_read(native.weight(AttentionProjection::Query), rows).expect("linear read");
            assert!(
                bits(&read) == bits(&layer.queries) && bits(&read_radius) == bits(&layer.query_radius),
                "linear_read must be the layer's native query read, values and radius"
            );
        }
        let short = fixture.residual.select(Axis(1), &[0, 1, 2]);
        assert!(
            matches!(
                linear_read(
                    native.weight(AttentionProjection::Query),
                    ProjectedRows {
                        values: fixture.residual.view(),
                        radius: short.view(),
                    },
                ),
                Err(ApplyError::Shape { operand: "input radius", .. })
            ),
            "a radius of another shape than its rows must be refused"
        );
        assert!(
            matches!(
                native.execute(
                    AttentionLayerReads::native(),
                    ProjectedRows {
                        values: fixture.residual.view(),
                        radius: short.view(),
                    },
                    &fixture.positions,
                ),
                Err(BlockError::ResidualShape { .. })
            ),
            "a residual radius of another shape than the residual rows must be refused"
        );
    }

    /// A mask box: one execution at its center, whose radii cover every mask of the box. Two
    /// key components and two output components range over `[0, 1]` (center 1/2, half-width
    /// 1/2), and every other control is a fixed binary value. At every stage the box execution
    /// agrees with each of the 16 vertex executions, and with an interior mask, within the sum
    /// of their radii. Positive control: the center executed as an exact mask leaves that sum
    /// at a vertex.
    #[test]
    fn a_mask_box_execution_encloses_every_mask_of_the_box() {
        let fixture = LayerFixture::new(2995);
        let layer = fixture.component();
        let mut rng = StdRng::seed_from_u64(2996);
        let binary = |rng: &mut StdRng| {
            Array1::from_shape_simple_fn(LAYER_COMPONENTS, || rng.random_range(0..=1) as f64)
        };
        let fixed = LayerMasks {
            query: binary(&mut rng),
            key: binary(&mut rng),
            value: binary(&mut rng),
            output: binary(&mut rng),
        };
        let free = [
            (AttentionProjection::Key, 1_usize),
            (AttentionProjection::Key, 6),
            (AttentionProjection::Output, 0),
            (AttentionProjection::Output, 4),
        ];
        let setting = |levels: [f64; 4]| {
            let mut masks = fixed.clone();
            for (&(projection, component), level) in free.iter().zip(levels) {
                match projection {
                    AttentionProjection::Key => masks.key[component] = level,
                    _ => masks.output[component] = level,
                }
            }
            masks
        };
        let center = setting([0.5; 4]);
        let mut widths = [(); 4].map(|_| Array2::<f64>::zeros((1, LAYER_COMPONENTS)));
        for &(projection, component) in &free {
            widths[projection.index()][[0, component]] = 0.5;
        }
        let centers = PROJECTIONS.map(|projection| center.of(projection).insert_axis(Axis(0)).to_owned());
        let boxed = |index: usize| ComponentMasks {
            center: centers[index].view(),
            half_width: Some(widths[index].view()),
        };
        let reads = AttentionLayerReads {
            query: ProjectionRead::Components(boxed(0)),
            key: ProjectionRead::Components(boxed(1)),
            value: ProjectionRead::Components(boxed(2)),
            output: ProjectionRead::Components(boxed(3)),
        };
        let execute = |reads: AttentionLayerReads<'_>| {
            layer
                .execute(reads, ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
                .expect("fixture layer")
        };
        let enclosing = execute(reads);
        let mut vertices = Vec::new();
        for corner in 0..16_u32 {
            vertices.push(setting([0, 1, 2, 3].map(|bit| f64::from((corner >> bit) & 1))));
        }
        vertices.push(setting([0.25, 0.75, 0.125, 1.0]));
        for (index, masks) in vertices.iter().enumerate() {
            let at = execute(masks.reads());
            for (stage, (left, left_radius), (right, right_radius)) in stage_radii(&enclosing, &at) {
                assert_eq!(
                    violations(left, right, &sum_of_bounds(left_radius.clone(), right_radius.clone())),
                    0,
                    "mask {index} of the box: the box execution's {stage} must enclose it"
                );
            }
        }
        let exact_center = execute(center.reads());
        let escapes = vertices.iter().any(|masks| {
            let at = execute(masks.reads());
            violations(
                &exact_center.output,
                &at.output,
                &sum_of_bounds(exact_center.output_radius.clone(), at.output_radius.clone()),
            ) > 0
        });
        assert!(escapes, "positive control: without the box radius the center must miss a vertex");
    }

    /// A per-row output mask reads each row under its own mask: rows that share a mask share
    /// one product, a per-row center whose rows all agree executes the uniform mask bit for bit,
    /// and each row of the write agrees with the uniform execution of that row's mask within the
    /// sum of the write radii. Malformed boxes are refused, typed.
    #[test]
    fn a_per_row_output_mask_reads_each_row_under_its_own_mask() {
        let fixture = LayerFixture::new(2997);
        let layer = fixture.component();
        let mut rng = StdRng::seed_from_u64(2998);
        let row_masks: Vec<Array1<f64>> = (0..LAYER_TOKENS)
            .map(|_| Array1::from_shape_simple_fn(LAYER_COMPONENTS, || rng.random_range(0..=1) as f64))
            .collect();
        let mut centers = Array2::<f64>::zeros((LAYER_TOKENS, LAYER_COMPONENTS));
        for (row, mask) in row_masks.iter().enumerate() {
            // Rows 0 and 2 share row 0's mask.
            centers.row_mut(row).assign(if row == 2 { &row_masks[0] } else { mask });
        }
        fn output_read(masks: ComponentMasks<'_>) -> AttentionLayerReads<'_> {
            AttentionLayerReads {
                output: ProjectionRead::Components(masks),
                ..AttentionLayerReads::native()
            }
        }
        let execute = |reads: AttentionLayerReads<'_>| {
            layer.execute(reads, ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
        };
        let per_row = execute(output_read(ComponentMasks {
            center: centers.view(),
            half_width: None,
        }))
        .expect("per-row output mask");
        for row in 0..LAYER_TOKENS {
            let mask = if row == 2 { &row_masks[0] } else { &row_masks[row] };
            let uniform = execute(output_read(ComponentMasks::uniform(mask.view()))).expect("uniform output mask");
            assert!(
                bits(&uniform.attention.mixed) == bits(&per_row.attention.mixed),
                "row {row}: the output mask reaches only the output read"
            );
            let band = sum_of_bounds(per_row.write_radius.clone(), uniform.write_radius.clone());
            let row_violations = (0..LAYER_WIDTH)
                .filter(|&column| (per_row.write[[row, column]] - uniform.write[[row, column]]).abs() > band[[row, column]])
                .count();
            assert_eq!(row_violations, 0, "row {row}: the write must be the row's own mask's write");
        }
        let other_row = (1..LAYER_TOKENS)
            .find(|&row| row != 2 && row_masks[row] != row_masks[0])
            .expect("a random row mask differs from row 0's");
        let first = execute(output_read(ComponentMasks::uniform(row_masks[0].view()))).expect("row 0's mask");
        assert!(
            row_bits(&per_row.write, other_row) != row_bits(&first.write, other_row),
            "positive control: a row under another mask must not read row 0's"
        );
        let same = Array2::from_shape_fn((LAYER_TOKENS, LAYER_COMPONENTS), |(_, column)| row_masks[0][column]);
        let repeated = execute(output_read(ComponentMasks {
            center: same.view(),
            half_width: None,
        }))
        .expect("repeated row mask");
        assert!(
            layer_stage_bits(&repeated) == layer_stage_bits(&first)
                && layer_radius_bits(&repeated) == layer_radius_bits(&first),
            "a per-row center whose rows agree must execute the uniform mask bit for bit"
        );

        let two_rows = centers.select(Axis(0), &[0, 1]);
        assert!(
            matches!(
                execute(output_read(ComponentMasks {
                    center: two_rows.view(),
                    half_width: None,
                })),
                Err(BlockError::Apply {
                    projection: AttentionProjection::Output,
                    error: ApplyError::Shape { operand: "component mask rows", .. }
                })
            ),
            "a center with neither one row nor one per input row must be refused"
        );
        let narrow = Array2::<f64>::zeros((LAYER_TOKENS, LAYER_COMPONENTS - 1));
        assert!(
            matches!(
                execute(output_read(ComponentMasks {
                    center: centers.view(),
                    half_width: Some(narrow.view()),
                })),
                Err(BlockError::Apply {
                    error: ApplyError::Shape { operand: "mask half-width", .. },
                    ..
                })
            ),
            "a half-width of another shape than its center must be refused"
        );
        let mut undefined = Array2::<f64>::zeros((LAYER_TOKENS, LAYER_COMPONENTS));
        undefined[[1, 3]] = f64::NAN;
        assert!(
            matches!(
                execute(output_read(ComponentMasks {
                    center: centers.view(),
                    half_width: Some(undefined.view()),
                })),
                Err(BlockError::Apply {
                    error: ApplyError::NonFinite { operand: "mask half-width" },
                    ..
                })
            ),
            "a non-finite half-width must be refused"
        );
    }

    /// Eighths in `[-3/2, 3/2]` for each gated-MLP factor.
    struct GatedMasks {
        gate: Array1<f64>,
        up: Array1<f64>,
        down: Array1<f64>,
    }

    impl GatedMasks {
        fn draw(rng: &mut StdRng) -> Self {
            let mut vector =
                |len: usize| Array1::from_shape_simple_fn(len, || rng.random_range(-12..=12) as f64 / 8.0);
            Self {
                gate: vector(GATE_COMPONENTS),
                up: vector(UP_COMPONENTS),
                down: vector(DOWN_COMPONENTS),
            }
        }

        fn swiglu(&self) -> SwigluMask<'_> {
            SwigluMask {
                gate: ComponentMask::Components(self.gate.view()),
                up: ComponentMask::Components(self.up.view()),
                down: ComponentMask::Components(self.down.view()),
            }
        }
    }

    fn gated_stage_bits(execution: &GatedBlockExecution) -> Vec<u64> {
        execution
            .attention_input
            .iter()
            .chain(execution.attention.output.iter())
            .chain(execution.attention.weights.iter())
            .chain(execution.mlp_input.iter())
            .chain(execution.mlp.gate.iter())
            .chain(execution.mlp.up.iter())
            .chain(execution.mlp.hidden.iter())
            .chain(execution.mlp.write.iter())
            .chain(execution.output.iter())
            .map(|value| value.to_bits())
            .collect()
    }

    /// All-on masks run both sublayers of a gated block on their original tensors
    /// and paths, so every stage is bit-identical to the native gated block.
    #[test]
    fn all_on_masks_execute_the_native_gated_block_bit_for_bit() {
        for (layout, seed) in [(ResidualLayout::Sequential, 2968), (ResidualLayout::Parallel, 2969)] {
            let fixture = Fixture::new(layout, GaussianActivation::Relu, seed);
            let block = fixture.component_gated_block();
            let native = fixture
                .native_gated_block()
                .execute(fixture.residual.view(), &fixture.positions)
                .expect("native gated block");
            let ones = FixtureMasks::ones();
            let attention_masks = ones.attention();
            let all_on = GatedBlockMasks {
                attention: &attention_masks,
                mlp: SwigluMask {
                    gate: ComponentMask::AllOn,
                    up: ComponentMask::AllOn,
                    down: ComponentMask::AllOn,
                },
            };
            let executed = block
                .execute(all_on, fixture.residual.view(), &fixture.positions)
                .expect("all-on gated block");
            assert!(
                gated_stage_bits(&executed) == gated_stage_bits(&native),
                "{layout:?}: the all-on gated block must execute the original tensors on their original paths"
            );

            // Positive control: the factored all-ones gate read is algebraically the
            // same block but not the same bits.
            let gate_ones = Array1::ones(GATE_COMPONENTS);
            let factored = block
                .execute(
                    GatedBlockMasks {
                        attention: &attention_masks,
                        mlp: SwigluMask {
                            gate: ComponentMask::Components(gate_ones.view()),
                            up: ComponentMask::AllOn,
                            down: ComponentMask::AllOn,
                        },
                    },
                    fixture.residual.view(),
                    &fixture.positions,
                )
                .expect("all-ones factored gate");
            assert!(
                gated_stage_bits(&factored) != gated_stage_bits(&native),
                "{layout:?}: the bit-identity check must distinguish the factored all-ones gate path"
            );
        }
    }

    /// A gated block's stages are its owners' stages on the current stream,
    /// composed in the source's order, bit for bit. A key mask reaches the gated MLP
    /// input exactly in the sequential layout.
    #[test]
    fn a_gated_block_is_its_owners_stages_on_the_current_stream() {
        for (layout, seed) in [(ResidualLayout::Sequential, 2970), (ResidualLayout::Parallel, 2971)] {
            let fixture = Fixture::new(layout, GaussianActivation::Relu, seed);
            let block = fixture.component_gated_block();
            let mut rng = StdRng::seed_from_u64(seed + 100);
            let [(kind, masks), ..] = FixtureMasks::family(&mut rng);
            let gated_masks = GatedMasks::draw(&mut rng);
            let attention_masks = masks.attention();
            let block_masks = GatedBlockMasks {
                attention: &attention_masks,
                mlp: gated_masks.swiglu(),
            };
            let execution = block
                .execute(block_masks, fixture.residual.view(), &fixture.positions)
                .expect("masked gated block");

            let attention_input = fixture
                .norm(&fixture.attention_gain, &fixture.attention_bias)
                .as_masked_norm()
                .apply(fixture.residual.view())
                .expect("attention norm");
            let attention = ComponentAttention::new(
                fixture.native_gated_block().attention,
                ComponentProjection::new(fixture.query_outputs.clone(), fixture.query_readins.clone())
                    .expect("query factor shapes agree"),
                ComponentProjection::new(fixture.key_outputs.clone(), fixture.key_readins.clone())
                    .expect("key factor shapes agree"),
            )
            .expect("attention factors match the geometry")
            .execute(&attention_masks, attention_input.view(), &fixture.positions)
            .expect("masked attention");
            let stream = match layout {
                ResidualLayout::Sequential => &fixture.residual + &attention.output,
                ResidualLayout::Parallel => fixture.residual.clone(),
            };
            let mlp_input = fixture
                .norm(&fixture.mlp_gain, &fixture.mlp_bias)
                .as_masked_norm()
                .apply(stream.view())
                .expect("MLP norm");
            let stages = block
                .mlp()
                .execute_stages(mlp_input.view(), gated_masks.swiglu())
                .expect("masked gated MLP");
            let output = match layout {
                ResidualLayout::Sequential => &stream + &stages.write,
                ResidualLayout::Parallel => &(&stages.write + &attention.output) + &fixture.residual,
            };
            let composed = GatedBlockExecution {
                attention_input,
                attention,
                mlp_input,
                mlp: stages,
                output,
            };
            assert!(
                gated_stage_bits(&execution) == gated_stage_bits(&composed),
                "{layout:?}, {kind} masks: the gated block must be its owners' stages composed on the current stream"
            );

            let mut moved = masks.clone();
            moved.key[1] += 0.125;
            let moved_attention = moved.attention();
            let moved_execution = block
                .execute(
                    GatedBlockMasks {
                        attention: &moved_attention,
                        mlp: gated_masks.swiglu(),
                    },
                    fixture.residual.view(),
                    &fixture.positions,
                )
                .expect("moved-mask gated block");
            assert!(
                bits(&moved_execution.attention.output) != bits(&execution.attention.output),
                "{layout:?}: positive control: the moved key mask must reach the attention write"
            );
            assert_eq!(
                bits(&moved_execution.mlp_input) != bits(&execution.mlp_input),
                layout == ResidualLayout::Sequential,
                "{layout:?}: a key mask must reach the gated MLP input exactly in the sequential layout"
            );
        }
    }

    /// A read whose products round into the subnormal range (#4005). Each of eight products
    /// `3e-161 · 7e-162 ≈ 42.504 · 2^-1074` rounds to a multiple of `2^-1074` and loses about
    /// `0.496 · 2^-1074`, which no relative band sees: the read's error is near
    /// `3.96 · 2^-1074`. Every quantity here is measured in units of `2^-1074`, so each
    /// comparison is between plain normal numbers.
    ///
    /// The units matter and are not cosmetic (#2822). Written as `3.5 · SUBNORMAL_SPACING`
    /// the bar is not the bar: the subnormal grid holds only integer multiples of `2^-1074`,
    /// so that product rounds to `4 · 2^-1074` and demands a whole spacing more than it says,
    /// which the measured `3.964` does not clear. Scaling the comparison into the normal range
    /// by `scale · scale` compounds it, because `2^600 · 2^600` overflows: the bar saturates
    /// at `f64::MAX`, and a diagnostic dividing by it reports an error of zero for an error of
    /// four spacings. The operands are still scaled by `2^600` to read each product's rounding
    /// exactly through a fused multiply-add, but the scaling is undone before anything is
    /// compared, and `scale · (scale · SUBNORMAL_SPACING)` is associated so no factor leaves
    /// the normal range.
    ///
    /// Positive control: the relative band alone, `γ_d |W| |x̂| / (1 − γ_(d+3))`, misses the
    /// error entirely — at this magnitude it underflows to exactly zero, so the whole band is
    /// the read's underflow reach.
    #[test]
    fn a_read_band_encloses_products_that_round_into_the_subnormal_range() {
        const PRODUCTS: usize = 8;
        let scale = 2.0_f64.powi(600);
        let (weight_entry, row_entry) = (3.0e-161, 7.0e-162);
        let weight = Array2::from_elem((1, PRODUCTS), weight_entry);
        let rows = Array2::from_elem((1, PRODUCTS), row_entry);
        let (read, band) = linear_read(weight.view(), ProjectedRows::exact(rows.view())).expect("subnormal read");
        let (weight_scaled, row_scaled) = (weight_entry * scale, row_entry * scale);
        let product = weight_scaled * row_scaled;
        let residual = weight_scaled.mul_add(row_scaled, -product);
        // One spacing at the scaled magnitude, `2^600 · 2^600 · 2^-1074 = 2^126`. Associated
        // so the inner factor is `2^-474`: no intermediate leaves the normal range.
        let scaled_spacing = scale * (scale * SUBNORMAL_SPACING);
        // Every division below is by a power of two and so is exact. The computed read is an
        // integer multiple of the spacing (subnormal sums are exact), and the exact read is
        // `8 (product + residual)` at the scaled magnitude.
        let computed_spacings = read[[0, 0]] / SUBNORMAL_SPACING;
        let terms = PRODUCTS as f64;
        let exact_spacings = terms * product / scaled_spacing + terms * residual / scaled_spacing;
        let error_spacings = (computed_spacings - exact_spacings).abs();
        assert!(
            error_spacings > 3.5,
            "the fixture must lose most of four subnormal spacings to underflow, \
             lost {error_spacings:.4} (computed {computed_spacings}, exact {exact_spacings:.6})"
        );
        let band_spacings = band[[0, 0]] / SUBNORMAL_SPACING;
        assert!(
            error_spacings <= band_spacings,
            "the read band of {band_spacings:.4} spacings must enclose the underflow error of \
             {error_spacings:.4} spacings"
        );
        let magnitude = abs_map(weight.view(), rows.view())[[0, 0]];
        let relative = accumulation_growth(PRODUCTS) * magnitude / (1.0 - accumulation_growth(PRODUCTS + 3));
        assert!(
            relative / SUBNORMAL_SPACING < error_spacings,
            "positive control: the relative band of {:.4} spacings alone must miss the \
             underflow error of {error_spacings:.4} spacings",
            relative / SUBNORMAL_SPACING
        );
    }

    /// A radius read `|W| r` whose products underflow to zero still carries their allowance
    /// (#4005): `1e-170 · 1e-170` rounds to zero, yet rows within `r = 1e-170` of the computed
    /// ones can move the exact read by `1e-340`. The carried radius is at least one subnormal
    /// spacing and exceeds the radius of the same read of exact rows. Positive control: the
    /// computed magnitude `|W| r` is zero.
    #[test]
    fn an_underflowed_radius_read_carries_its_allowance() {
        let weight = Array2::from_elem((1, 1), 1.0e-170);
        let rows = Array2::from_elem((1, 1), 1.0e-170);
        let radius = Array2::from_elem((1, 1), 1.0e-170);
        assert_eq!(
            native_magnitude(weight.view(), radius.view()).magnitude[[0, 0]],
            0.0,
            "positive control: the magnitude of the radius read underflows to zero"
        );
        let (_, exact) = linear_read(weight.view(), ProjectedRows::exact(rows.view())).expect("exact rows");
        let (_, carried) = linear_read(
            weight.view(),
            ProjectedRows {
                values: rows.view(),
                radius: radius.view(),
            },
        )
        .expect("rows with a radius");
        assert!(
            carried[[0, 0]] >= SUBNORMAL_SPACING && carried[[0, 0]] > exact[[0, 0]],
            "an underflowed radius read must carry its allowance: carried {:e}, exact rows {:e}",
            carried[[0, 0]],
            exact[[0, 0]]
        );
    }

    /// At normal scale the underflow allowance and the upward steps leave a read band at its
    /// relative size `γ_d |W| |x̂| / (1 − γ_(d+3))`, plus at most `2 d · 2^-1074`. The band's five
    /// stepped operations and its stepped divisor each scale by at most `(1 + u)(1 + 2u)`, and
    /// this test's relative band rounds four times, so the ratio stays below
    /// `(1 + 3u)^6 (1 + u)^4 < 1 + 16 ε`.
    #[test]
    fn the_underflow_allowance_leaves_a_normal_read_band_at_its_relative_size() {
        let fixture = LayerFixture::new(4005);
        let native = fixture.native();
        let weight = native.weight(AttentionProjection::Query);
        let width = weight.ncols();
        let (_, band) = linear_read(weight, ProjectedRows::exact(fixture.residual.view())).expect("native read");
        let magnitude = abs_map(weight, fixture.residual.mapv(f64::abs).view());
        let relative = magnitude
            .mapv(|value| accumulation_growth(width) * value / (1.0 - accumulation_growth(width + 3)));
        Zip::from(&band).and(&relative).for_each(|&band, &relative| {
            assert!(
                band <= relative * (1.0 + 16.0 * f64::EPSILON) + 2.0 * width as f64 * SUBNORMAL_SPACING,
                "a normal read band {band:e} must stay at its relative size {relative:e}"
            );
        });
        assert!(
            relative.iter().any(|&value| value > 0.0),
            "the fixture's reads must round"
        );
    }
}
