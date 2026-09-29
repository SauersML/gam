//! Transformer layers executed under component masks, with forward-error radii (#2951).
//!
//! This module owns the attention-only layer ([`NativeAttentionLayer`],
//! [`ComponentAttentionLayer`]), the matrix-free linear read with its rounding band
//! ([`linear_read`]), and the rounding band of the source's RMSNorm ([`rms_norm_band`]) that a
//! receipt compares an external executor against.
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
    AttentionGeometry, AttentionProgramError, ProjectedAttention, ProjectedRows, RotaryCausalAttention,
    RotaryEmbedding,
};
use super::gated_rewrite::GatedRewriteError;
use super::occurrence::PositionScope;
use super::rewrite::{
    ComponentRead, ExactFactor, FactorRefusal,
};
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use gam_runtime::resource::{Governed, MemoryGovernor};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
use std::fmt;

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

/// The rounding band of one binary64 evaluation of an RMSNorm, `MaskedNorm::Rms`'s program
/// `fl(w_j fl(x_j ν̂))`, against the exact RMSNorm `y = w x (mean(x²) + ε)^(-1/2)` of the input
/// rows `inputs` (`d` wide, with the gain `gain`), from its computed rows `normalized`.
///
/// The squares, `d − 1` additions, the mean, `+ ε`, the square root, the reciprocal and the
/// products with the row and the gain are `γ_(d+6)` relative while their results stay
/// normal. A square, the mean or a product whose result is subnormal rounds by an absolute
/// `2^-1075` instead: through `ν̂` that is the relative `ρ` of `normalization_growth`, and on
/// the entry it is the absolute `a_j = |w_j| 2^-1074 + 2^-1074` of `underflow_reach`. So
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
/// square is not resolved above binary64's underflow ([`rms_norm_band`]).
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

/// A refused block execution.
#[derive(Debug)]
pub enum BlockError {
    /// A normalization refused its rows.
    Norm(GatedRewriteError),
    Attention(AttentionProgramError),
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

impl From<GatedRewriteError> for BlockError {
    fn from(error: GatedRewriteError) -> Self {
        Self::Norm(error)
    }
}

impl From<AttentionProgramError> for BlockError {
    fn from(error: AttentionProgramError) -> Self {
        Self::Attention(error)
    }
}

impl fmt::Display for BlockError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Norm(error) => write!(formatter, "the normalization refused: {error}"),
            Self::Attention(error) => write!(formatter, "the attention sublayer refused: {error}"),
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
        governor: &MemoryGovernor,
        reads: AttentionLayerReads<'_>,
        residual: ProjectedRows<'_>,
        positions: &[i64],
    ) -> Result<AttentionLayerExecution, BlockError> {
        execute_attention_layer(governor, self, None, reads, residual, positions)
    }

    /// The rounding band of one linear read of `rows` at their absolute positions:
    /// `|fl(A x) − A x|` for the map `A` that `read` applies, entrywise, where `fl`
    /// is apply.rs's kernel (`read_band`). A component read is refused.
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
        governor: &MemoryGovernor,
        reads: AttentionLayerReads<'_>,
        residual: ProjectedRows<'_>,
        positions: &[i64],
    ) -> Result<AttentionLayerExecution, BlockError> {
        execute_attention_layer(governor, &self.native, Some(&self.factors), reads, residual, positions)
    }

    /// The rounding band of one linear read of `rows` at their absolute positions
    /// (`read_band`).
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
    governor: &MemoryGovernor,
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
        let values = read_projection(governor, weight, factor, projection, read, rows.values, positions)?;
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
    governor: &MemoryGovernor,
    weight: ArrayView2<'_, f64>,
    factor: Option<&ProjectionFactor>,
    projection: AttentionProjection,
    read: ProjectionRead<'_>,
    rows: ArrayView2<'_, f64>,
    positions: &[i64],
) -> Result<Governed<Array2<f64>>, BlockError> {
    let refused = |error: ApplyError| BlockError::Apply { projection, error };
    match read {
        ProjectionRead::Native => native_linear(governor, weight, rows).map_err(refused),
        ProjectionRead::Components(masks) => {
            let factor = factor.ok_or(BlockError::NoComponentFactors { projection })?;
            let view = factor.view().map_err(refused)?;
            let groups = mask_groups(projection, &masks, rows.nrows())?;
            // The rows of the first mask read every row, so one mask for every row is one
            // product; each other mask's rows are read on their own and written over it.
            let first = groups.first().map_or(0, |group| group.0);
            let mut written =
                apply_anchored_linear(governor, weight, 0.0, view, masks.center.row(first), rows).map_err(refused)?;
            for (mask_row, members) in groups.iter().skip(1) {
                let selected = rows.select(Axis(0), members);
                let read = apply_anchored_linear(governor, weight, 0.0, view, masks.center.row(*mask_row), selected.view())
                    .map_err(refused)?;
                for (index, &row) in members.iter().enumerate() {
                    written.row_mut(row).assign(&read.row(index));
                }
            }
            Ok(written)
        }
        ProjectionRead::Edited(scoped) => {
            let reached = reached_rows(projection, &scoped, positions)?;
            let mut written = native_linear(governor, weight, rows).map_err(refused)?;
            if reached.is_empty() {
                return Ok(written);
            }
            let selected = rows.select(Axis(0), &reached);
            let every_term = Array1::<f64>::ones(scoped.edit.term_count());
            let edited = apply_anchored_linear(governor, weight, 1.0, scoped.edit, every_term.view(), selected.view())
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
    governor: &MemoryGovernor,
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
    let values = native_linear(governor, weight, rows.values)?;
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
    use crate::test_support::test_governor;
    use crate::attention::RotaryPairing;
    use crate::gated_rewrite::MaskedNorm;
    use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
    use ndarray::ArrayView1;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// Qwen3's declared `rms_norm_eps`.
    const RMS_EPSILON: f64 = 1.0e-6;

    /// Eighths in `[-1, 1]`. A product of three and a sum of a few dozen stay
    /// exact in f64, so every product of fixture tensors and masks, including each
    /// edited tensor `A diag(m) B`, is formed without rounding.
    fn eighths(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || rng.random_range(-8..=8) as f64 / 8.0)
    }

    fn eighths_vector(rng: &mut StdRng, len: usize) -> Array1<f64> {
        Array1::from_shape_simple_fn(len, || rng.random_range(-8..=8) as f64 / 8.0)
    }

    /// `A diag(m) B`.
    fn masked_product(write: ArrayView2<'_, f64>, mask: ArrayView1<'_, f64>, read: ArrayView2<'_, f64>) -> Array2<f64> {
        (&write * &mask).dot(&read)
    }

    fn violations(left: &Array2<f64>, right: &Array2<f64>, band: &Array2<f64>) -> usize {
        left.iter()
            .zip(right.iter())
            .zip(band.iter())
            .filter(|((a, b), bound)| (*a - *b).abs() > **bound)
            .count()
    }

    fn bits(values: &Array2<f64>) -> Vec<u64> {
        values.iter().map(|value| value.to_bits()).collect()
    }

    /// `rms_norm_band` covers the double-double RMSNorm of the rows `MaskedNorm::apply`
    /// normalized. Positive control: the computed rows are not the double-double ones, so a
    /// zero band is exceeded and the band's coverage is not vacuous.
    #[test]
    fn a_normalization_band_covers_the_double_double_normalization() {
        use qd::Quad;
        let mut rng = StdRng::seed_from_u64(2997);
        let gain = eighths_vector(&mut rng, LAYER_WIDTH);
        let rows = Array2::from_shape_simple_fn((LAYER_TOKENS, LAYER_WIDTH), || {
            rng.random_range(-48..=48) as f64 / 24.0
        });
        let quad = Quad::from_f64;
        let reference = |row: usize| -> Vec<Quad> {
            let values: Vec<Quad> = rows.row(row).iter().map(|&value| quad(value)).collect();
            let square = values.iter().fold(quad(0.0), |sum, &value| sum + value * value);
            let inverse_root = quad(1.0) / (square / quad(LAYER_WIDTH as f64) + quad(RMS_EPSILON)).sqrt();
            values
                .iter()
                .enumerate()
                .map(|(column, &value)| quad(gain[column]) * value * inverse_root)
                .collect()
        };
        let excess = |computed: &Array2<f64>, band: &Array2<f64>| {
            (0..LAYER_TOKENS)
                .map(|row| {
                    let exact = reference(row);
                    (0..LAYER_WIDTH)
                        .filter(|&column| (quad(computed[[row, column]]) - exact[column]).0.abs() > band[[row, column]])
                        .count()
                })
                .sum::<usize>()
        };
        let normalized = MaskedNorm::Rms { epsilon: RMS_EPSILON, gain: gain.view() }
            .apply(rows.view())
            .expect("finite rows");
        let band = rms_norm_band(RMS_EPSILON, gain.view(), rows.view(), normalized.view()).expect("finite rows");
        assert_eq!(excess(&normalized, &band), 0, "a row left its rounding band around the double-double RMSNorm");
        let zero = Array2::<f64>::zeros(normalized.raw_dim());
        assert!(
            excess(&normalized, &zero) > 0,
            "positive control: the computed rows must differ from the double-double normalization somewhere"
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
        let normalized = MaskedNorm::Rms { epsilon: RMS_EPSILON, gain: gain.view() }
            .apply(rows.view())
            .expect("finite rows");
        let band = rms_norm_band(RMS_EPSILON, gain.view(), rows.view(), normalized.view()).expect("finite rows");
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
        let unit_gain = ndarray::array![1.0, 1.0];
        let owner = MaskedNorm::Rms { epsilon: 0.0, gain: unit_gain.view() }.apply(unresolved.view());
        assert!(owner.is_ok(), "control: the owner normalizes a row whose mean square is the smallest subnormal");
        let owner = owner.expect("checked above");
        assert!(
            matches!(
                rms_norm_band(0.0, unit_gain.view(), unresolved.view(), owner.view()),
                Err(NormBandUnbounded { row: 0 })
            ),
            "a zero-epsilon row whose mean square is not resolved above the underflow must be refused"
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
                .execute(test_governor(), masks.reads(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
                .expect("masked layer");
            let edited = fixture
                .edited(&layer, &masks)
                .execute(test_governor(), AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
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
                    .execute(test_governor(), AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
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
            .execute(test_governor(), AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
            .expect("native layer");
        let all_on = layer
            .execute(test_governor(), AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
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
                test_governor(),
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
                .execute(test_governor(), reads, ProjectedRows::exact(first.residual.view()), &first.positions)
                .expect("layer 0");
            // Layer 1 reads layer 0's output with its radius, as a stack does.
            let top = layer1
                .execute(
                    test_governor(),
                    AttentionLayerReads::native(),
                    ProjectedRows {
                        values: hidden.output.view(),
                        radius: hidden.output_radius.view(),
                    },
                    &first.positions,
                )
                .expect("layer 1");
            let logits = native_linear(test_governor(), unembed.view(), top.output.view())
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
            .execute(test_governor(), AttentionLayerReads::native(), ProjectedRows::exact(first.residual.view()), &first.positions)
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
                test_governor(),
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
                .execute(test_governor(), AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
                .is_ok(),
            "positive control: the fixture layer executes"
        );

        let ones = Array1::<f64>::ones(LAYER_COMPONENTS);
        assert!(
            matches!(
                native.execute(
                    test_governor(),
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
                    test_governor(),
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
                    test_governor(),
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
                    test_governor(),
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
                test_governor(),
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
                    component.execute(test_governor(), reads, rows, &fixture.positions)
                } else {
                    native.execute(test_governor(), reads, rows, &fixture.positions)
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
                    component.execute(test_governor(), reads, rows, &fixture.positions)
                } else {
                    native.execute(test_governor(), reads, rows, &fixture.positions)
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
                        test_governor(),
                        AttentionLayerReads::native(),
                        ProjectedRows {
                            values: first.output.view(),
                            radius: first.output_radius.view(),
                        },
                        &fixture.positions,
                    )
                    .expect("second layer");
                let (logits, logit_radius) = linear_read(
                    test_governor(),
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
                test_governor(),
                AttentionLayerReads::native(),
                ProjectedRows {
                    values: moved.view(),
                    radius: radius.view(),
                },
                &fixture.positions,
            )
            .expect("silent layer");
        let reference = silent
            .execute(test_governor(), AttentionLayerReads::native(), ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
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
                .execute(test_governor(), AttentionLayerReads::native(), rows, &fixture.positions)
                .expect("native layer");
            let (read, read_radius) =
                linear_read(test_governor(), native.weight(AttentionProjection::Query), rows).expect("linear read");
            assert!(
                bits(&read) == bits(&layer.queries) && bits(&read_radius) == bits(&layer.query_radius),
                "linear_read must be the layer's native query read, values and radius"
            );
        }
        let short = fixture.residual.select(Axis(1), &[0, 1, 2]);
        assert!(
            matches!(
                linear_read(
                    test_governor(),
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
                    test_governor(),
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
                .execute(test_governor(), reads, ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
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
            layer.execute(test_governor(), reads, ProjectedRows::exact(fixture.residual.view()), &fixture.positions)
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
        let (read, band) = linear_read(test_governor(), weight.view(), ProjectedRows::exact(rows.view())).expect("subnormal read");
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
        let (_, exact) = linear_read(test_governor(), weight.view(), ProjectedRows::exact(rows.view())).expect("exact rows");
        let (_, carried) = linear_read(
            test_governor(),
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
        let (_, band) = linear_read(test_governor(), weight, ProjectedRows::exact(fixture.residual.view())).expect("native read");
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
