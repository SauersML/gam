//! Exact canonical gauge forms of a native decoder layer (#2951).
//!
//! [`super::gauge`] detects the implementation gauges of a layer's tensors. This module
//! applies them: it moves a layer along its declared gauge orbits to one stated
//! representative, returns the group element it applied, and executes the layer before and
//! after with derived forward-error radii, so function equality is checked on executed
//! outputs rather than asserted.
//!
//! # The layer
//!
//! [`DecoderLayer`] is the sequential pre-norm layer of Qwen3 and Llama:
//!
//! ```text
//! h₁ = h + Attn(N₁(h)),   h₂ = h₁ + W_d [s(W_g N₂(h₁)) ⊙ W_u N₂(h₁)],
//! ```
//!
//! with `N_k(h) = w_k ⊙ h ν(h)`, `ν(h) = (mean h² + ε)^{-1/2}`, the SiLU `s(t) = t σ(t)`, and
//! the source's rotary attention ([`super::attention`]), with or without Qwen3's per-head
//! query/key norm. GPT-NeoX's LayerNorm gain is folded by [`fold_norm_gain`] exactly as an
//! RMSNorm gain is (below); its parallel-residual GELU layer is not a [`DecoderLayer`].
//!
//! # The canonical form
//!
//! Each transform is an element of a family [`super::gauge`] certifies, so in exact
//! arithmetic the layer's function does not move.
//!
//! * **(a) Norm gains** ([`fold_norm_gain`]). A norm output `w ⊙ n + β` read only by linear
//!   reads `W_k` is invariant under `(w, β, W_k) ↦ (D w, D β, W_k D⁻¹)`. The representative is
//!   `D = diag(w)⁻¹`: unit gains, bias `β ⊘ w`, reads `W_k diag(w)`. The normalizer `n(h)` is
//!   formed before the gain, so it never sees `D`; this holds for RMSNorm and for LayerNorm
//!   alike. A zero gain is a null coordinate, not an orbit coordinate, and is refused.
//! * **(b) SwiGLU unit scales** ([`balance_swiglu`]). The block is bilinear in the up read,
//!   so `(up_n, down_n) ↦ (s_n up_n, down_n / s_n)` is exact for every `s_n ≠ 0`. The
//!   representative has `‖up_n‖ = ‖down_n‖` (`|s_n| = (‖down_n‖/‖up_n‖)^{1/2}`), and the sign of
//!   `s_n` makes the up row's first largest-magnitude entry positive. A unit with a zero gate
//!   row, up row or down column writes nothing and keeps `s_n = 1`.
//! * **(c) OV per key/value group** ([`canonical_value_output`]). Group `g`'s value rows `V_g`
//!   and the output columns `O_h` of its query heads `h ∈ g` carry `GL(r)`:
//!   `(V_g, O_h) ↦ (T⁻¹ V_g, O_h T)`. The orbit's invariant is the stacked operator
//!   `C_g = [O_h V_g]_{h∈g}`, and the representative takes `V_g′` to be `C_g`'s right singular
//!   vectors (orthonormal rows, ordered by singular value, each row's first
//!   largest-magnitude entry positive) and `O_h′ = O_h T`. It is computed without forming
//!   `C_g`: with `V_gᵀ = Q R` (thin QR) and `[O_h]_h Rᵀ = U Σ Wᵀ` (thin SVD),
//!   `C_g = U Σ (Q W)ᵀ`, so `T = Rᵀ W` and `V_g′ = Wᵀ Qᵀ`. The representative is unique when
//!   `C_g`'s singular values are distinct and nonzero; [`CanonicalValueOutput::separated`]
//!   reports whether the computed gaps resolve that. A value block without full row rank has
//!   a smaller orbit and is refused.
//! * **(d) Normed rotary query/key** ([`balance_query_key`]). Behind Qwen3's per-head norm
//!   only the group [`super::gauge::NormedRotaryQueryKey`] derives acts. Its plane scales
//!   `ρ_p` multiply the plane's query gains and divide its key gains; the representative takes
//!   `ρ_p = (|v_a v_b| / |w_a w_b|)^{1/4}`, so both gain pairs have one geometric mean, and
//!   flips every negative gain (a certified sign flip carried by its rows). The quarter turns
//!   and the per-head rotations of coincident planes are left at the identity. An unnormed
//!   rotary block carries the whole rotary commutant, whose representative is not derived
//!   here, so [`DecoderLayer::canonical`] leaves its query/key tensors as they are.
//!
//! # Scope: declared families only
//!
//! The canonical form quotients exactly the families [`super::gauge`] declares, and those are
//! a lower bound on a layer's implementation redundancy. The planted toys of #2951 found
//! function-preserving freedom outside them: a `GL(d)` hidden in a pair of cancelling MLP
//! branches, and a cross-head `GL` between heads whose attention patterns are identical,
//! whose value/output transports can be recombined. Two layers with different canonical forms
//! can therefore still execute one function; equal canonical forms (within the stated bands)
//! do imply equal functions.
//!
//! # Representation defects
//!
//! Each element is applied exactly as a group element, but the canonical tensors are stored
//! in binary64, so each lies within a derived entrywise defect ([`TensorDefect`]) of the tensor
//! the element maps it to exactly. A unit gain is stored exactly. [`DecoderLayer::execute`]
//! carries each defect into its forward-error radius, so a canonical layer's radius covers the
//! exact layer at the tensors it represents, which is the native layer's function. Native and
//! canonical outputs then agree within the sum of their radii, and a difference beyond it is
//! a certified counterexample to function equality.

use std::ops::Range;

use gam_linalg::faer_ndarray::{FaerLinalgError, FaerQr, FaerSvd, fast_ab, fast_abt, fast_av};
use gam_linalg::roundoff::{accumulation_growth, factor_singular_band, householder_qr_backward_band};
use gam_runtime::resource::MemoryGovernor;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip, concatenate, s};

use super::apply::ApplyError;
use super::attention::{
    AffineProjection, AttentionGeometry, AttentionProgramError, NativeAttention, ProjectedRows, RotaryCausalAttention,
    RotaryEmbedding, head_rms_norm_with_radius,
};
use super::block::{SUBNORMAL_SPACING, linear_read, up};
use super::gated_rewrite::{GatedRewriteError, swiglu_hidden};
use super::gauge::{GaugeRefusal, NormGain, NormedRotaryChange, NormedRotaryQueryKey, SwigluUnits, UnitGaugeChange};

/// `sup_t |s′(t)|` for the SiLU `s(t) = t σ(t)`, bounded above by `1 + 1/e`.
///
/// `s′ = σ + t σ(1 − σ)` with `0 < σ < 1`, and `σ(1 − σ) = e^{−|t|}/(1 + e^{−|t|})² ≤ e^{−|t|}`, so
/// `|t σ(1 − σ)| ≤ |t| e^{−|t|} ≤ 1/e`. The constant is rounded up.
fn silu_slope_bound() -> f64 {
    up(1.0 + up((-1.0_f64).exp()))
}

/// Why a canonical form or a layer execution was declined.
#[derive(Clone, Debug, PartialEq)]
pub enum CanonicalRefusal {
    /// The gauge owner refused a family or an element.
    Gauge(Box<GaugeRefusal>),
    /// The attention owner refused its tensors or rows.
    Attention(Box<AttentionProgramError>),
    /// The gated-rewrite owner refused a norm or a gate.
    Norm(Box<GatedRewriteError>),
    /// A matrix-free read was refused.
    Apply(Box<ApplyError>),
    /// A tensor or row block disagrees with the layer.
    Shape {
        what: &'static str,
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// A zero norm gain: its coordinate is null, and no scale takes it to one.
    ZeroNormGain { coordinate: usize },
    /// A value block whose group element `T` does not resolve full rank: its smallest singular
    /// value is not above the SVD's band, so `T⁻¹` is not certified.
    ValueRank { group: usize, smallest: f64, band: f64 },
    /// [`DecoderLayer::canonical`] on a layer whose tensors already carry a defect.
    NotNative { tensor: &'static str },
    /// The linear-algebra backend failed.
    Decomposition { what: &'static str, detail: String },
}

impl From<GaugeRefusal> for CanonicalRefusal {
    fn from(error: GaugeRefusal) -> Self {
        Self::Gauge(Box::new(error))
    }
}

impl From<AttentionProgramError> for CanonicalRefusal {
    fn from(error: AttentionProgramError) -> Self {
        Self::Attention(Box::new(error))
    }
}

impl From<GatedRewriteError> for CanonicalRefusal {
    fn from(error: GatedRewriteError) -> Self {
        Self::Norm(Box::new(error))
    }
}

impl From<ApplyError> for CanonicalRefusal {
    fn from(error: ApplyError) -> Self {
        Self::Apply(Box::new(error))
    }
}

fn decomposition(what: &'static str, failure: &FaerLinalgError) -> CanonicalRefusal {
    CanonicalRefusal::Decomposition {
        what,
        detail: format!("{failure:?}"),
    }
}

fn expect_shape(what: &'static str, expected: (usize, usize), found: (usize, usize)) -> Result<(), CanonicalRefusal> {
    if expected == found {
        Ok(())
    } else {
        Err(CanonicalRefusal::Shape { what, expected, found })
    }
}

/// An entrywise bound on `|stored − represented|` for a stored tensor, where `represented`
/// is the tensor its canonical form stands for exactly.
#[derive(Clone, Debug, PartialEq)]
pub enum TensorDefect {
    /// The stored tensor is the represented one.
    Exact,
    /// `|stored − represented| ≤ ρ |stored|` entrywise.
    Relative(f64),
    /// `|stored − represented| ≤ E` entrywise, `E` of the tensor's shape.
    Entrywise(Array2<f64>),
}

impl TensorDefect {
    pub fn is_exact(&self) -> bool {
        matches!(self, Self::Exact)
    }

    /// `|δW| m` for nonnegative rows `m` (`tokens × in`), `tokens × out`: the most the defect
    /// moves a read of any rows within `m` of zero entrywise. `None` when exact.
    fn reach(&self, weight: ArrayView2<'_, f64>, magnitude: ArrayView2<'_, f64>) -> Option<Array2<f64>> {
        match self {
            Self::Exact => None,
            Self::Relative(relative) => {
                let reach = nonnegative_product(magnitude, weight.mapv(f64::abs).view());
                Some(reach.mapv(|entry| up(entry * relative)))
            }
            Self::Entrywise(bound) => Some(nonnegative_product(magnitude, bound.view())),
        }
    }

    /// An upper bound on `‖δW‖_F` over the rows `rows` of `weight`.
    fn frobenius_rows(&self, weight: ArrayView2<'_, f64>, rows: Range<usize>) -> f64 {
        match self {
            Self::Exact => 0.0,
            Self::Relative(relative) => up(relative * upper_frobenius(weight.slice(s![rows, ..]))),
            Self::Entrywise(bound) => upper_frobenius(bound.slice(s![rows, ..])),
        }
    }
}

/// `m Bᵀ` for nonnegative `m` (`t × n`) and `B` (`o × n`), rounded up. Each entry is an
/// `n`-term sum of nonnegative products, within `γ_n` of its exact value, so the exact value is
/// at most `(1 + 2γ_n)` times the computed one while `γ_n ≤ 1/2`. Each product that rounds into
/// the subnormal range adds at most `2^-1074` absolute, `n 2^-1074` in all.
fn nonnegative_product(magnitude: ArrayView2<'_, f64>, bound: ArrayView2<'_, f64>) -> Array2<f64> {
    let terms = bound.ncols();
    let factor = up(1.0 + up(2.0 * accumulation_growth(terms)));
    let underflow = up(terms as f64 * SUBNORMAL_SPACING);
    fast_abt(&magnitude, &bound).mapv(|entry| up(up(entry * factor) + underflow))
}

/// An upper bound on `‖M‖_F`: the sum of squares within `γ_n`, the square root correctly
/// rounded, each rounded up.
fn upper_frobenius(matrix: ArrayView2<'_, f64>) -> f64 {
    let terms = matrix.len().max(1);
    let squares = matrix.iter().fold(0.0, |sum, &entry| up(sum + up(entry * entry)));
    up(up(squares * up(1.0 + up(2.0 * accumulation_growth(terms)))).sqrt())
}

/// A linear read `W x + b` of a layer, with the defects of its stored weight and bias.
#[derive(Clone, Debug)]
pub struct LayerProjection {
    pub weight: Array2<f64>,
    pub bias: Array1<f64>,
    pub weight_defect: TensorDefect,
    /// `|stored bias − represented bias|` entrywise.
    pub bias_defect: Array1<f64>,
}

impl LayerProjection {
    fn native(projection: &AffineProjection) -> Self {
        Self {
            weight: projection.weight.clone(),
            bias: projection.bias.clone(),
            weight_defect: TensorDefect::Exact,
            bias_defect: Array1::zeros(projection.bias.len()),
        }
    }

    fn bias_free(weight: Array2<f64>, weight_defect: TensorDefect) -> Self {
        let rows = weight.nrows();
        Self {
            weight,
            bias: Array1::zeros(rows),
            weight_defect,
            bias_defect: Array1::zeros(rows),
        }
    }

    fn is_exact(&self) -> bool {
        self.weight_defect.is_exact() && self.bias_defect.iter().all(|&entry| entry == 0.0)
    }

    fn affine(&self) -> AffineProjection {
        AffineProjection {
            weight: self.weight.clone(),
            bias: self.bias.clone(),
        }
    }

    /// `W x + b` of rows `x` given within `r`, and its radius against the represented read of
    /// the exact rows:
    /// - [`linear_read`]'s band, `γ_d |Ŵ| |x̂| + |Ŵ| r` with its underflow allowance;
    /// - the weight defect over the box, `|δW| (|x̂| + r)`;
    /// - the bias defect, and the bias addition's rounding `γ_1 |fl(y + b)|`.
    fn read(&self, governor: &MemoryGovernor, rows: &Banded) -> Result<Banded, CanonicalRefusal> {
        let (product, band) = linear_read(
            governor,
            self.weight.view(),
            ProjectedRows {
                values: rows.values.view(),
                radius: rows.radius.view(),
            },
        )?;
        let mut values = product.to_owned();
        let mut radius = band;
        if let Some(reach) = self.weight_defect.reach(self.weight.view(), rows.magnitude().view()) {
            Zip::from(&mut radius).and(&reach).for_each(|slot, &extra| *slot = up(*slot + extra));
        }
        let rounding = accumulation_growth(1);
        for (mut value_row, mut radius_row) in values.rows_mut().into_iter().zip(radius.rows_mut()) {
            Zip::from(&mut value_row)
                .and(&mut radius_row)
                .and(&self.bias)
                .and(&self.bias_defect)
                .for_each(|value, slot, &bias, &defect| {
                    *value += bias;
                    *slot = up(up(*slot + defect) + up(rounding * value.abs()));
                });
        }
        Ok(Banded { values, radius })
    }
}

/// Rows with a per-entry forward-error radius.
#[derive(Clone, Debug)]
struct Banded {
    values: Array2<f64>,
    radius: Array2<f64>,
}

impl Banded {
    /// `|x̂| + r`, rounded up: every exact row of the box lies within it entrywise.
    fn magnitude(&self) -> Array2<f64> {
        Zip::from(&self.values)
            .and(&self.radius)
            .map_collect(|&value, &radius| up(value.abs() + radius))
    }

    /// `x + y` and its radius `r_x + r_y + γ_1 |fl(x + y)|`.
    fn add(&self, other: &Banded) -> Banded {
        let rounding = accumulation_growth(1);
        let values = &self.values + &other.values;
        let radius = Zip::from(&self.radius)
            .and(&other.radius)
            .and(&values)
            .map_collect(|&first, &second, &sum| up(up(first + second) + up(rounding * sum.abs())));
        Banded { values, radius }
    }
}

/// A sequential RMSNorm `w ⊙ h (mean h² + ε)^{-1/2}` with the relative defect of its stored
/// gain.
#[derive(Clone, Debug)]
pub struct LayerRmsNorm {
    pub epsilon: f64,
    pub gain: Array1<f64>,
    /// `|stored gain − represented gain| ≤ gain_defect · |stored gain|` entrywise.
    pub gain_defect: f64,
}

impl LayerRmsNorm {
    pub fn native(epsilon: f64, gain: Array1<f64>) -> Self {
        Self {
            epsilon,
            gain,
            gain_defect: 0.0,
        }
    }

    /// The norm of every `head_dim`-wide block of rows given within their radius
    /// ([`head_rms_norm_with_radius`]), plus the gain defect. The represented gain is
    /// `ŵ (1 + θ)` with `|θ| ≤ ρ`, so its output is `(1 + θ)` times the stored gain's, and the
    /// defect adds `ρ (|ŷ| + r)`.
    /// The norm over whole rows given within their radius: [`Self::apply`] with one head
    /// the width of a row. Returns the normalized rows and their radius.
    pub(super) fn apply_rows(&self, rows: ProjectedRows<'_>) -> Result<(Array2<f64>, Array2<f64>), CanonicalRefusal> {
        let banded = Banded {
            values: rows.values.to_owned(),
            radius: rows.radius.to_owned(),
        };
        let normed = self.apply(&banded, rows.values.ncols())?;
        Ok((normed.values, normed.radius))
    }

    fn apply(&self, rows: &Banded, head_dim: usize) -> Result<Banded, CanonicalRefusal> {
        let (values, mut radius) = head_rms_norm_with_radius(
            ProjectedRows {
                values: rows.values.view(),
                radius: rows.radius.view(),
            },
            head_dim,
            self.epsilon,
            self.gain.view(),
        )?;
        if self.gain_defect > 0.0 {
            Zip::from(&mut radius)
                .and(&values)
                .for_each(|slot, &value| *slot = up(*slot + up(self.gain_defect * up(value.abs() + *slot))));
        }
        Ok(Banded { values, radius })
    }
}

/// Qwen3's per-head query/key norm with the relative defect of its stored gains.
#[derive(Clone, Debug)]
pub struct LayerQueryKeyNorm {
    pub query: LayerRmsNorm,
    pub key: LayerRmsNorm,
}

/// One executed layer: the residual after attention and after the MLP, each with its
/// forward-error radius against the exact layer at the represented tensors.
#[derive(Clone, Debug)]
pub struct LayerExecution {
    pub attention_residual: Array2<f64>,
    pub attention_radius: Array2<f64>,
    pub output: Array2<f64>,
    pub output_radius: Array2<f64>,
}

impl LayerExecution {
    /// The largest `|self − other| − (r_self + r_other)` over the output, and where. A
    /// positive value certifies that the two layers execute different functions at these
    /// rows; a nonpositive one is agreement within the bands.
    pub fn largest_excess(&self, other: &LayerExecution) -> (f64, (usize, usize)) {
        let mut largest = (f64::NEG_INFINITY, (0, 0));
        for ((index, &first), (&second, (&first_radius, &second_radius))) in self
            .output
            .indexed_iter()
            .zip(other.output.iter().zip(self.output_radius.iter().zip(other.output_radius.iter())))
        {
            let excess = (first - second).abs() - up(first_radius + second_radius);
            if excess > largest.0 {
                largest = (excess, index);
            }
        }
        largest
    }
}

/// A native sequential pre-norm decoder layer (Qwen3, Llama): RMSNorm, rotary attention,
/// residual, RMSNorm, SwiGLU, residual. Every tensor carries its representation defect.
#[derive(Clone, Debug)]
pub struct DecoderLayer {
    geometry: AttentionGeometry,
    rotary: RotaryEmbedding,
    score_scale: f64,
    pub input_norm: LayerRmsNorm,
    pub query: LayerProjection,
    pub key: LayerProjection,
    pub value: LayerProjection,
    pub output: LayerProjection,
    pub query_key_norm: Option<LayerQueryKeyNorm>,
    pub post_norm: LayerRmsNorm,
    pub gate: LayerProjection,
    pub up: LayerProjection,
    pub down: LayerProjection,
}

impl DecoderLayer {
    /// The layer on its native tensors. The MLP is bias-free, as in Qwen3 and Llama.
    pub fn native(
        input_norm: LayerRmsNorm,
        attention: &NativeAttention,
        post_norm: LayerRmsNorm,
        mlp: &SwigluUnits,
    ) -> Result<Self, CanonicalRefusal> {
        let geometry = attention.geometry();
        let width = geometry.model_dim;
        expect_shape("input norm gain", (width, 1), (input_norm.gain.len(), 1))?;
        expect_shape("post-attention norm gain", (width, 1), (post_norm.gain.len(), 1))?;
        expect_shape("SwiGLU gate", (mlp.units(), width), mlp.gate().dim())?;
        expect_shape("SwiGLU down", (width, mlp.units()), mlp.down().dim())?;
        let query_key_norm = attention.query_key_norm().map(|norm| LayerQueryKeyNorm {
            query: LayerRmsNorm::native(norm.epsilon(), norm.query_gain().to_owned()),
            key: LayerRmsNorm::native(norm.epsilon(), norm.key_gain().to_owned()),
        });
        Ok(Self {
            geometry,
            rotary: attention.rotary().clone(),
            score_scale: attention.score_scale(),
            input_norm,
            query: LayerProjection::native(attention.query()),
            key: LayerProjection::native(attention.key()),
            value: LayerProjection::native(attention.value()),
            output: LayerProjection::native(attention.output()),
            query_key_norm,
            post_norm,
            gate: LayerProjection::bias_free(mlp.gate().to_owned(), TensorDefect::Exact),
            up: LayerProjection::bias_free(mlp.up().to_owned(), TensorDefect::Exact),
            down: LayerProjection::bias_free(mlp.down().to_owned(), TensorDefect::Exact),
        })
    }

    pub fn geometry(&self) -> AttentionGeometry {
        self.geometry
    }

    pub fn rotary(&self) -> &RotaryEmbedding {
        &self.rotary
    }

    pub fn score_scale(&self) -> f64 {
        self.score_scale
    }

    /// The attention block on the stored tensors, with the stored query/key gains.
    pub fn attention(&self) -> Result<NativeAttention, CanonicalRefusal> {
        let native = NativeAttention::new(
            self.geometry,
            self.rotary.clone(),
            self.score_scale,
            self.query.affine(),
            self.key.affine(),
            self.value.affine(),
            self.output.affine(),
        )?;
        Ok(match &self.query_key_norm {
            None => native,
            Some(norm) => native.with_query_key_norm(norm.query.epsilon, norm.query.gain.clone(), norm.key.gain.clone())?,
        })
    }

    /// The SwiGLU block on the stored tensors.
    pub fn swiglu(&self) -> Result<SwigluUnits, CanonicalRefusal> {
        Ok(SwigluUnits::new(self.gate.weight.clone(), self.up.weight.clone(), self.down.weight.clone())?)
    }

    fn is_native(&self) -> Result<(), CanonicalRefusal> {
        let norms = [("input norm", &self.input_norm), ("post-attention norm", &self.post_norm)];
        for (tensor, norm) in norms {
            if norm.gain_defect != 0.0 {
                return Err(CanonicalRefusal::NotNative { tensor });
            }
        }
        if let Some(norm) = &self.query_key_norm
            && (norm.query.gain_defect != 0.0 || norm.key.gain_defect != 0.0)
        {
            return Err(CanonicalRefusal::NotNative { tensor: "query/key norm" });
        }
        let projections = [
            ("query", &self.query),
            ("key", &self.key),
            ("value", &self.value),
            ("output", &self.output),
            ("gate", &self.gate),
            ("up", &self.up),
            ("down", &self.down),
        ];
        for (tensor, projection) in projections {
            if !projection.is_exact() {
                return Err(CanonicalRefusal::NotNative { tensor });
            }
        }
        Ok(())
    }

    /// The layer at residual rows given within their radius, at absolute `positions`. Every
    /// stage's radius covers the exact layer at the represented tensors for every exact row
    /// of the box:
    /// - each norm through [`head_rms_norm_with_radius`] (the residual norms as one head per
    ///   row), plus its gain defect;
    /// - each read through its [`LayerProjection`]'s banded read;
    /// - the attention core through [`RotaryCausalAttention::attend_projected`];
    /// - the SwiGLU gate as in `swiglu_band`;
    /// - each residual addition with `γ_1` of its sum.
    pub fn execute(
        &self,
        governor: &MemoryGovernor,
        residual: ProjectedRows<'_>,
        positions: &[i64],
    ) -> Result<LayerExecution, CanonicalRefusal> {
        let width = self.geometry.model_dim;
        expect_shape("residual rows", (positions.len(), width), residual.values.dim())?;
        expect_shape("residual radius", residual.values.dim(), residual.radius.dim())?;
        let input = Banded {
            values: residual.values.to_owned(),
            radius: residual.radius.to_owned(),
        };
        let normed = self.input_norm.apply(&input, width)?;
        let mut queries = self.query.read(governor, &normed)?;
        let mut keys = self.key.read(governor, &normed)?;
        let values = self.value.read(governor, &normed)?;
        if let Some(norm) = &self.query_key_norm {
            queries = norm.query.apply(&queries, self.geometry.head_dim)?;
            keys = norm.key.apply(&keys, self.geometry.head_dim)?;
        }
        let core = RotaryCausalAttention::new(self.geometry, self.rotary.clone(), self.score_scale)?;
        let mixed = {
            let projected = core.attend_projected(
                governor,
                ProjectedRows {
                    values: queries.values.view(),
                    radius: queries.radius.view(),
                },
                ProjectedRows {
                    values: keys.values.view(),
                    radius: keys.radius.view(),
                },
                ProjectedRows {
                    values: values.values.view(),
                    radius: values.radius.view(),
                },
                positions,
            )?;
            Banded {
                values: projected.mixed.clone(),
                radius: projected.mixed_radius.clone(),
            }
        };
        let written = self.output.read(governor, &mixed)?;
        let attended = input.add(&written);
        let normed = self.post_norm.apply(&attended, width)?;
        let gate = self.gate.read(governor, &normed)?;
        let up_rows = self.up.read(governor, &normed)?;
        let hidden = swiglu_band(&gate, &up_rows)?;
        let mlp = self.down.read(governor, &hidden)?;
        let output = attended.add(&mlp);
        Ok(LayerExecution {
            attention_residual: attended.values,
            attention_radius: attended.radius,
            output: output.values,
            output_radius: output.radius,
        })
    }

    /// The canonical form of a native layer (module docs, (a)–(d)), with the group elements
    /// applied and the defects of the canonical tensors.
    ///
    /// Order: the input norm gain folds into the query, key and value reads, and the
    /// post-attention gain into the gate and up reads (each `γ_1` relative). The query/key
    /// balance permutes and flips whole rows exactly and scales the gains (`γ_1`). The OV form
    /// acts on the folded value rows, whose defect it composes. The SwiGLU balance scales the
    /// folded up rows (`γ_2` in all) and the down columns (`γ_1`).
    pub fn canonical(&self) -> Result<CanonicalLayer, CanonicalRefusal> {
        self.is_native()?;
        let rounding = accumulation_growth(1);
        let input = fold_norm_gain(&NormGain::new(
            self.input_norm.gain.clone(),
            None,
            vec![self.query.weight.clone(), self.key.weight.clone(), self.value.weight.clone()],
        )?)?;
        let post = fold_norm_gain(&NormGain::new(
            self.post_norm.gain.clone(),
            None,
            vec![self.gate.weight.clone(), self.up.weight.clone()],
        )?)?;
        let folded = input.canonical().reads();
        let mut layer = self.clone();
        layer.input_norm = LayerRmsNorm {
            epsilon: self.input_norm.epsilon,
            gain: input.canonical().gain().to_owned(),
            gain_defect: 0.0,
        };
        layer.query.weight = folded[0].clone();
        layer.query.weight_defect = TensorDefect::Relative(rounding);
        layer.key.weight = folded[1].clone();
        layer.key.weight_defect = TensorDefect::Relative(rounding);
        layer.value.weight = folded[2].clone();
        layer.value.weight_defect = TensorDefect::Relative(rounding);

        let query_key = match &self.query_key_norm {
            None => None,
            Some(norm) => {
                let balanced = balance_query_key(&layer.attention()?)?;
                let canonical = balanced.canonical();
                layer.query.weight = canonical.query_weight().to_owned();
                layer.query.bias = canonical.query_bias().to_owned();
                layer.key.weight = canonical.key_weight().to_owned();
                layer.key.bias = canonical.key_bias().to_owned();
                layer.query_key_norm = Some(LayerQueryKeyNorm {
                    query: LayerRmsNorm {
                        epsilon: norm.query.epsilon,
                        gain: canonical.query_gain().to_owned(),
                        gain_defect: balanced.gain_defect(),
                    },
                    key: LayerRmsNorm {
                        epsilon: norm.key.epsilon,
                        gain: canonical.key_gain().to_owned(),
                        gain_defect: balanced.gain_defect(),
                    },
                });
                Some(balanced.change().clone())
            }
        };

        let value_output = canonical_value_output(
            self.geometry,
            &layer.value.affine(),
            &self.output.affine(),
            &layer.value.weight_defect,
        )?;
        layer.value = LayerProjection {
            weight: value_output.value.weight.clone(),
            bias: value_output.value.bias.clone(),
            weight_defect: TensorDefect::Entrywise(value_output.value_defect.clone()),
            bias_defect: value_output.value_bias_defect.clone(),
        };
        layer.output = LayerProjection {
            weight: value_output.output.weight.clone(),
            bias: value_output.output.bias.clone(),
            weight_defect: TensorDefect::Entrywise(value_output.output_defect.clone()),
            bias_defect: Array1::zeros(self.output.bias.len()),
        };

        let post_reads = post.canonical().reads();
        layer.post_norm = LayerRmsNorm {
            epsilon: self.post_norm.epsilon,
            gain: post.canonical().gain().to_owned(),
            gain_defect: 0.0,
        };
        let swiglu = balance_swiglu(&SwigluUnits::new(
            post_reads[0].clone(),
            post_reads[1].clone(),
            self.down.weight.clone(),
        )?)?;
        let units = swiglu.canonical();
        layer.gate = LayerProjection::bias_free(units.gate().to_owned(), TensorDefect::Relative(rounding));
        layer.up = LayerProjection::bias_free(units.up().to_owned(), TensorDefect::Relative(accumulation_growth(2)));
        layer.down = LayerProjection::bias_free(units.down().to_owned(), TensorDefect::Relative(rounding));

        Ok(CanonicalLayer {
            layer,
            elements: LayerGaugeElements {
                input_norm: input.inverse_scales().to_owned(),
                post_norm: post.inverse_scales().to_owned(),
                value_output: value_output.changes.clone(),
                value_output_separated: value_output.separated.clone(),
                swiglu: swiglu.change().clone(),
                query_key,
            },
        })
    }
}

/// [`swiglu_band`] of gate and up rows given as values and radius: the SwiGLU hidden rows
/// and their radius.
pub(super) fn swiglu_rows(
    gate: ProjectedRows<'_>,
    up_rows: ProjectedRows<'_>,
) -> Result<(Array2<f64>, Array2<f64>), CanonicalRefusal> {
    let banded = |rows: ProjectedRows<'_>| Banded {
        values: rows.values.to_owned(),
        radius: rows.radius.to_owned(),
    };
    let hidden = swiglu_band(&banded(gate), &banded(up_rows))?;
    Ok((hidden.values, hidden.radius))
}

/// The SwiGLU hidden rows `s(g) ⊙ u` of gate and up rows given within their radii, and the
/// radius against the exact `s(g_e) u_e` over the box.
///
/// - **Propagation.** `|s(g_e) u_e − s(ĝ) û| ≤ |s(g_e) − s(ĝ)| |u_e| + |s(ĝ)| |u_e − û|
///   ≤ L r_g (|û| + r_u) + (|ŝ| + e_s) r_u`, with `L` = [`silu_slope_bound`].
/// - **Evaluation of `ŝ = fl(g σ̂(g))`.** gam-math's `σ` forms `exp` (one ulp, `2u`
///   relative), one addition and one division, so `σ̂` is within `γ_5` relative; the product
///   adds `u`. An `exp` that underflows is off by one subnormal ulp, which reaches `ŝ` as
///   `2|g| 2^-1074`, and the product can round by one more. So
///   `e_s = γ_7 |ŝ|/(1 − γ_7) + (2|g| + 2) 2^-1074 ≤ γ_8 |ŝ| + (2|g| + 2) 2^-1074`.
/// - **The product `fl(ŝ û)`**: `e_s |û| + γ_1 |fl(ŝ û)| + 2^-1074`.
fn swiglu_band(gate: &Banded, up_rows: &Banded) -> Result<Banded, CanonicalRefusal> {
    let values = swiglu_hidden(gate.values.view(), up_rows.values.view())?;
    let slope = silu_slope_bound();
    let (evaluation, product) = (accumulation_growth(8), accumulation_growth(1));
    let mut radius = Array2::zeros(values.raw_dim());
    Zip::from(&mut radius)
        .and(&values)
        .and(&gate.values)
        .and(&gate.radius)
        .and(&up_rows.values)
        .and(&up_rows.radius)
        .for_each(|slot, &hidden, &g, &gate_radius, &u, &up_radius| {
            let silu = g * gam_math::special::logistic(g);
            let silu_error = up(up(evaluation * silu.abs()) + up(up(2.0 * g.abs() + 2.0) * SUBNORMAL_SPACING));
            let propagated = up(up(slope * up(gate_radius * up(u.abs() + up_radius))) + up(up(silu.abs() + silu_error) * up_radius));
            let evaluated = up(up(silu_error * u.abs()) + up(product * hidden.abs()) + SUBNORMAL_SPACING);
            *slot = up(propagated + evaluated);
        });
    Ok(Banded { values, radius })
}

/// A canonical layer and the group elements that took the native layer to it.
#[derive(Clone, Debug)]
pub struct CanonicalLayer {
    pub layer: DecoderLayer,
    pub elements: LayerGaugeElements,
}

/// The gauge elements of a [`CanonicalLayer`], each in its family's coordinates.
#[derive(Clone, Debug)]
pub struct LayerGaugeElements {
    /// `D⁻¹` of the input norm's element, as its diagonal: the native gain.
    pub input_norm: Array1<f64>,
    /// `D⁻¹` of the post-attention norm's element: the native gain.
    pub post_norm: Array1<f64>,
    /// `T_g` per key/value group: `V_g′ = T_g⁻¹ V_g`, `O_h′ = O_h T_g`.
    pub value_output: Vec<Array2<f64>>,
    /// Whether each group's representative is certified unique ([`CanonicalValueOutput::separated`]).
    pub value_output_separated: Vec<bool>,
    /// The SwiGLU unit scales (the permutation is the identity).
    pub swiglu: UnitGaugeChange,
    /// The normed query/key element, when the layer has a query/key norm.
    pub query_key: Option<NormedRotaryChange>,
}

/// A norm with its gain folded into its reads ((a) in the module docs).
#[derive(Clone, Debug)]
pub struct FoldedNorm {
    canonical: NormGain,
    native_gain: Array1<f64>,
}

impl FoldedNorm {
    /// Unit gains, bias `β ⊘ w`, reads `W_k diag(w)`.
    pub fn canonical(&self) -> &NormGain {
        &self.canonical
    }

    /// The applied element is `D = diag(w)⁻¹`; this is `D⁻¹`'s diagonal, the native gain,
    /// stored exactly. [`NormGain::apply`] with `1 ⊘ w` is the same element up to the rounding
    /// of its reciprocals.
    pub fn inverse_scales(&self) -> ArrayView1<'_, f64> {
        self.native_gain.view()
    }

    /// The relative defect of every canonical read entry and bias entry: one rounded
    /// product or quotient, `γ_1`. The unit gains are exact.
    pub fn defect(&self) -> f64 {
        accumulation_growth(1)
    }
}

/// (a) Fold a norm's gain into every read of its output. It refuses a zero gain.
pub fn fold_norm_gain(norm: &NormGain) -> Result<FoldedNorm, CanonicalRefusal> {
    let gain = norm.gain();
    if let Some(coordinate) = gain.iter().position(|&entry| entry == 0.0) {
        return Err(CanonicalRefusal::ZeroNormGain { coordinate });
    }
    let reads = norm.reads().iter().map(|read| read * &gain).collect();
    let bias = norm.bias().map(|bias| &bias / &gain);
    let canonical = NormGain::new(Array1::ones(gain.len()), bias, reads)?;
    Ok(FoldedNorm {
        canonical,
        native_gain: gain.to_owned(),
    })
}

/// A SwiGLU block at its balanced unit scales ((b) in the module docs).
#[derive(Clone, Debug)]
pub struct BalancedSwiglu {
    canonical: SwigluUnits,
    change: UnitGaugeChange,
}

impl BalancedSwiglu {
    pub fn canonical(&self) -> &SwigluUnits {
        &self.canonical
    }

    /// The element applied: identity relabelling, scales `s_n`.
    pub fn change(&self) -> &UnitGaugeChange {
        &self.change
    }

    /// The relative defect of every canonical up and down entry: one rounded product or
    /// quotient, `γ_1`.
    pub fn defect(&self) -> f64 {
        accumulation_growth(1)
    }

    /// A relative band on `‖up_n′‖/‖down_n′‖ − 1`, each norm computed as a rounded sum of
    /// squares and a square root.
    ///
    /// The computed norms of an `n`-vector are within `γ_{n+2}` relative. So the ratio behind
    /// `s_n` is within `γ_{w+o+5}` and `s_n` within `γ_{w+o+8}` of the balancing scale `s*`,
    /// with `w` the up row's and `o` the down column's length. The stored rows round once more
    /// each, `‖up′‖/‖down′‖ = (s/s*)² (1 + θ₁)/(1 + θ₂)`, and recomputing the two norms adds
    /// `γ_{w+2} + γ_{o+2}`. With `1/(1 − γ_1) ≤ 1 + γ_2` and `(1 + γ_a)(1 + γ_b) ≤ 1 + γ_{a+b}`,
    /// the ratio is within `γ_{3(w+o)+26}` of one.
    pub fn balance_band(&self) -> f64 {
        accumulation_growth(3 * (self.canonical.gate().ncols() + self.canonical.down().nrows()) + 26)
    }
}

/// (b) Balance every SwiGLU unit's up row against its down column.
pub fn balance_swiglu(units: &SwigluUnits) -> Result<BalancedSwiglu, CanonicalRefusal> {
    let count = units.units();
    let mut scales = Array1::ones(count);
    let (up_rows, down_columns) = (units.up(), units.down());
    for (unit, scale) in scales.iter_mut().enumerate() {
        let up_row = up_rows.row(unit);
        let gate_zero = units.gate().row(unit).iter().all(|&entry| entry == 0.0);
        let up_norm = up_row.dot(&up_row).sqrt();
        let down_column = down_columns.column(unit);
        let down_norm = down_column.dot(&down_column).sqrt();
        if gate_zero || up_norm == 0.0 || down_norm == 0.0 {
            continue;
        }
        let leading = leading_index(up_row);
        let magnitude = (down_norm / up_norm).sqrt();
        *scale = if up_row[leading] < 0.0 { -magnitude } else { magnitude };
    }
    let change = UnitGaugeChange {
        permutation: (0..count).collect(),
        scales,
    };
    let canonical = units.apply(&change)?;
    Ok(BalancedSwiglu { canonical, change })
}

/// The first index of the largest `|entry|`, compared exactly.
fn leading_index(values: ArrayView1<'_, f64>) -> usize {
    let mut best = 0;
    for (index, &entry) in values.iter().enumerate() {
        if entry.abs() > values[best].abs() {
            best = index;
        }
    }
    best
}

/// A canonical value/output pair ((c) in the module docs).
#[derive(Clone, Debug)]
pub struct CanonicalValueOutput {
    pub value: AffineProjection,
    pub output: AffineProjection,
    /// `T_g` per key/value group, `head_dim × head_dim`.
    pub changes: Vec<Array2<f64>>,
    /// The computed singular values of each group's stacked operator `C_g`, descending.
    pub singular_values: Vec<Array1<f64>>,
    /// Whether each group's singular values are distinct and nonzero beyond their band, so
    /// its representative is unique.
    pub separated: Vec<bool>,
    /// Entrywise bound on `|V′ − T⁻¹ V|` (the represented value rows), per entry of `value`.
    pub value_defect: Array2<f64>,
    /// Entrywise bound on `|b′ − T⁻¹ b|`.
    pub value_bias_defect: Array1<f64>,
    /// Entrywise bound on `|O′ − O T|`.
    pub output_defect: Array2<f64>,
}

/// (c) The canonical OV representative of every key/value group. `value_defect` is the defect
/// the value weight already carries (the input norm fold); the output weight is native.
///
/// Defects, per group with `T = T_g`:
/// - `O_h′ = fl(O_h T)`: `γ_r |O_h| |T|` entrywise;
/// - `V_g′`: the represented rows are `T⁻¹ V_g`. With the computed residual
///   `E = T V_g′ − V̂_g` (within `γ_{r+1} (|T| |V_g′| + |V̂_g|)` of its exact value) and the stored
///   value rows within `δ` of the represented ones, `‖V_g′ − T⁻¹ V_g‖₂ ≤ (‖E‖_F + ‖δ‖_F)/σ_min(T)`,
///   which bounds every entry. `σ_min(T)` is taken below its computed value by the SVD's band,
///   and a `T` whose bound is not positive is refused.
/// - the bias `b′ = T⁻¹ b` by the same residual argument.
pub fn canonical_value_output(
    geometry: AttentionGeometry,
    value: &AffineProjection,
    output: &AffineProjection,
    value_defect: &TensorDefect,
) -> Result<CanonicalValueOutput, CanonicalRefusal> {
    let (width, rank) = (geometry.model_dim, geometry.head_dim);
    expect_shape("value weight", (geometry.key_value_dim(), width), value.weight.dim())?;
    expect_shape("output weight", (width, geometry.query_dim()), output.weight.dim())?;
    let group_size = geometry.n_heads / geometry.n_kv_heads;
    let mut canonical_value = value.clone();
    let mut canonical_output = output.clone();
    let mut value_bound = Array2::zeros(value.weight.raw_dim());
    let mut bias_bound = Array1::zeros(value.bias.len());
    let mut output_bound = Array2::zeros(output.weight.raw_dim());
    let (mut changes, mut singular_values, mut separated) = (Vec::new(), Vec::new(), Vec::new());
    for group in 0..geometry.n_kv_heads {
        let rows = group * rank..(group + 1) * rank;
        let block = value.weight.slice(s![rows.clone(), ..]);
        if width < rank {
            return Err(CanonicalRefusal::ValueRank {
                group,
                smallest: 0.0,
                band: 0.0,
            });
        }
        let (basis, triangle) = block.t().qr().map_err(|failure| decomposition("value rows", &failure))?;
        let basis = basis.slice(s![.., ..rank]).to_owned();
        let triangle = triangle.slice(s![..rank, ..rank]).to_owned();
        let heads: Vec<usize> = (group * group_size..(group + 1) * group_size).collect();
        let columns: Vec<_> = heads
            .iter()
            .map(|&head| output.weight.slice(s![.., head * rank..(head + 1) * rank]))
            .collect();
        let stacked = concatenate(Axis(0), &columns).map_err(|failure| CanonicalRefusal::Decomposition {
            what: "stacked output columns",
            detail: failure.to_string(),
        })?;
        let core = fast_abt(&stacked, &triangle);
        let (spectrum, right) = match core.svd(false, true) {
            Ok((_, spectrum, Some(right))) => (spectrum, right),
            Ok(_) => {
                return Err(CanonicalRefusal::Decomposition {
                    what: "stacked OV core",
                    detail: "right singular vectors were requested but not returned".to_string(),
                });
            }
            Err(failure) => return Err(decomposition("stacked OV core", &failure)),
        };
        // V′ = Wᵀ Qᵀ, then each row's first largest-magnitude entry made positive.
        let mut rows_prime = fast_abt(&right, &basis);
        let mut change = fast_abt(&triangle.t(), &right);
        let mut flipped = vec![false; rank];
        for (row, flip) in flipped.iter_mut().enumerate() {
            let lead = leading_index(rows_prime.row(row));
            if rows_prime[[row, lead]] < 0.0 {
                *flip = true;
                rows_prime.row_mut(row).mapv_inplace(|entry| -entry);
                change.column_mut(row).mapv_inplace(|entry| -entry);
            }
        }
        let change_spectrum = change.svd(false, false).map_err(|failure| decomposition("OV change", &failure))?.1;
        let largest = change_spectrum.iter().copied().fold(0.0, f64::max);
        let smallest = change_spectrum.iter().copied().fold(f64::INFINITY, f64::min);
        let band = factor_singular_band(rank, rank, largest);
        let floor = smallest - band;
        if floor.is_nan() || floor <= 0.0 {
            return Err(CanonicalRefusal::ValueRank { group, smallest, band });
        }
        let floor = floor.next_down();

        // Value defect.
        let absolute_change = change.mapv(f64::abs);
        let residual = fast_ab(&change, &rows_prime) - block;
        let residual_band = (fast_ab(&absolute_change, &rows_prime.mapv(f64::abs)) + &block.mapv(f64::abs))
            .mapv(|entry| up(entry * accumulation_growth(rank + 1)));
        let residual_norm = up(upper_frobenius(residual.view()) + upper_frobenius(residual_band.view()));
        let incoming = value_defect.frobenius_rows(value.weight.view(), rows.clone());
        let row_bound = up(up(residual_norm + incoming) / floor);
        value_bound.slice_mut(s![rows.clone(), ..]).fill(row_bound);
        canonical_value.weight.slice_mut(s![rows.clone(), ..]).assign(&rows_prime);

        // Value bias: b′ = D Wᵀ R⁻ᵀ b, with R⁻ᵀ b by forward substitution on the lower triangle Rᵀ.
        let bias = value.bias.slice(s![rows.clone()]);
        if bias.iter().any(|&entry| entry != 0.0) {
            let mut solved = Array1::<f64>::zeros(rank);
            for i in 0..rank {
                let partial: f64 = (0..i).map(|k| triangle[[k, i]] * solved[k]).sum();
                solved[i] = (bias[i] - partial) / triangle[[i, i]];
            }
            let mut canonical_bias = fast_av(&right, &solved);
            for (entry, &flip) in canonical_bias.iter_mut().zip(&flipped) {
                if flip {
                    *entry = -*entry;
                }
            }
            let residual = fast_av(&change, &canonical_bias) - bias;
            let residual_band =
                (fast_av(&absolute_change, &canonical_bias.mapv(f64::abs)) + &bias.mapv(f64::abs)).mapv(|entry| up(entry * accumulation_growth(rank + 1)));
            let norm = up(upper_frobenius(residual.view().insert_axis(Axis(1))) + upper_frobenius(residual_band.view().insert_axis(Axis(1))));
            bias_bound.slice_mut(s![rows.clone()]).fill(up(norm / floor));
            canonical_value.bias.slice_mut(s![rows.clone()]).assign(&canonical_bias);
        }

        // Output columns.
        for &head in &heads {
            let columns = head * rank..(head + 1) * rank;
            let native = output.weight.slice(s![.., columns.clone()]);
            canonical_output.weight.slice_mut(s![.., columns.clone()]).assign(&fast_ab(&native, &change));
            let reach = nonnegative_product(native.mapv(f64::abs).view(), absolute_change.t());
            let rounding = accumulation_growth(rank);
            output_bound
                .slice_mut(s![.., columns])
                .assign(&reach.mapv(|entry| up(up(rounding * entry) + up(rank as f64 * SUBNORMAL_SPACING))));
        }

        // Separation of C_g's singular values: the core's SVD band, the core's formation
        // `γ_r ‖|O| |Rᵀ|‖_F`, and the QR's backward error carried through `‖O‖_F`.
        let spectrum_top = spectrum.iter().copied().fold(0.0, f64::max);
        let formation = up(accumulation_growth(rank) * upper_frobenius(fast_abt(&stacked.mapv(f64::abs), &triangle.mapv(f64::abs)).view()));
        let qr_band = householder_qr_backward_band(width, rank, upper_frobenius(triangle.view()));
        let spectrum_band = up(up(factor_singular_band(core.nrows(), rank, spectrum_top) + formation) + up(upper_frobenius(stacked.view()) * qr_band));
        let distinct = spectrum.windows(2).into_iter().all(|pair| pair[0] - pair[1] > up(2.0 * spectrum_band));
        let nonzero = spectrum.iter().all(|&sigma| sigma > spectrum_band);
        separated.push(distinct && nonzero && spectrum.len() == rank);
        singular_values.push(spectrum);
        changes.push(change);
    }
    Ok(CanonicalValueOutput {
        value: canonical_value,
        output: canonical_output,
        changes,
        singular_values,
        separated,
        value_defect: value_bound,
        value_bias_defect: bias_bound,
        output_defect: output_bound,
    })
}

/// A normed rotary query/key block at its balanced plane scales ((d) in the module docs).
#[derive(Clone, Debug)]
pub struct BalancedQueryKey {
    canonical: NormedRotaryQueryKey,
    change: NormedRotaryChange,
}

impl BalancedQueryKey {
    pub fn canonical(&self) -> &NormedRotaryQueryKey {
        &self.canonical
    }

    /// The element applied: the plane scales `ρ_p`, no quarter turns, and a flip of every
    /// negative gain.
    pub fn change(&self) -> &NormedRotaryChange {
        &self.change
    }

    /// The relative defect of every canonical gain: one rounded product or quotient. The
    /// projection rows are signed permutations, so they carry no defect.
    pub fn gain_defect(&self) -> f64 {
        accumulation_growth(1)
    }
}

/// (d) Balance a normed rotary query/key block's plane scales and make its gains positive.
/// It refuses every block [`NormedRotaryQueryKey::new`] refuses.
pub fn balance_query_key(native: &NativeAttention) -> Result<BalancedQueryKey, CanonicalRefusal> {
    let block = NormedRotaryQueryKey::new(native)?;
    let rotary = native.rotary();
    let planes = rotary.inverse_frequencies.len();
    let (query_gain, key_gain) = (block.query_gain(), block.key_gain());
    let plane_scales = Array1::from_shape_fn(planes, |plane| {
        let (a, b) = rotary.plane(plane);
        let ratio = (key_gain[a] * key_gain[b]).abs() / (query_gain[a] * query_gain[b]).abs();
        ratio.sqrt().sqrt()
    });
    let change = NormedRotaryChange {
        plane_scales,
        quarter_turns: Array2::zeros((native.geometry().n_kv_heads, planes)),
        query_gain_flips: query_gain.iter().map(|&gain| gain < 0.0).collect(),
        key_gain_flips: key_gain.iter().map(|&gain| gain < 0.0).collect(),
    };
    let canonical = block.apply(&change)?;
    Ok(BalancedQueryKey { canonical, change })
}

#[cfg(test)]
#[path = "canonical_tests.rs"]
mod tests;
