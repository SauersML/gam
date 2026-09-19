//! Exact execution of gated activations, normalizations, biases and residual
//! writes under component masks (#2951).
//!
//! A masked linear tensor `W(m) = U diag(m) R` (with mpd-lift's anchor adding
//! `m_Delta (W_* - U R)`) is applied to input rows by the rewrite owners. This
//! module owns the native primitives that are not linear in their input, and it
//! applies each one to the SUMMED masked vector, never per component: the native
//! nonlinearity sees the same summed input under every mask, so the rewrite
//! equals direct execution with the edited tensors for every real mask
//! (continuous, binary and signed). Inputs are rows of observations, as in the
//! lift and rewrite kernels. The derivation is on #2951 (mpd-gated, R1-R5); the
//! native forwards were read in the installed `transformers` source (`Qwen3MLP`,
//! `Qwen3RMSNorm`, `Qwen3Model`, `GPTNeoXLayer`).
//!
//! * **Biases (R1).** An affine map `(W, b)` on `x` is the linear map `[W | b]`
//!   on the homogeneous read `(x, 1)` ([`homogeneous_read`]). A component with
//!   read `(r_c, rho_c)` writes `u_c rho_c` into the bias, so bias components are
//!   ordinary components of a `(d + 1)`-column tensor and their moment vectors
//!   carry the bias coordinates. The constant `1` is never masked.
//! * **SwiGLU (R2).** `F_m(x) = W_d(m_d) [s(W_g(m_g) x) ⊙ W_u(m_u) x]` with the
//!   SiLU gate `s(t) = t sigma(t)` (Qwen3 `hidden_act = "silu"`)
//!   ([`swiglu_hidden`], [`ComponentSwiglu`]). The gate is read from its one
//!   owner, [`gam_math::gaussian_gated::silu_derivatives`], and every weight
//!   factors through mpd-rewrite's [`ExactFactor`]. At fixed `x` the block is
//!   affine in each of `m_u` and `m_d` (multilinear, degree one per control) and
//!   nonlinear in `m_g`. `W_u -> D W_u`, `W_d -> W_d D^-1` for an invertible
//!   diagonal `D` is a gauge of the block; the gate rows have none.
//! * **Norms (R3, R4).** `Qwen3RMSNorm` is `w ⊙ h (mean(h^2) + eps)^(-1/2)` and
//!   `torch.nn.LayerNorm` is `w ⊙ (P h)(mean((P h)^2) + eps)^(-1/2) + beta` with
//!   the centring `P = I - 1 1^T / d` ([`MaskedNorm`], [`rms_normalizers`]). The
//!   gain and bias are masked tensors that enter linearly; the normalizer is
//!   computed on each whole current residual row. `N_eps(c h) = sign(c)
//!   N_(eps/c^2)(h)`, so a uniform scale of the stream reaches the next RMSNorm
//!   only through `eps`, and a LayerNorm annihilates every write along `1`.
//! * **Residual writes (R5).** The identity skip carries no parameter, so no
//!   control reaches it. An edge mask `e` on a block is the edit
//!   `(W_out, b_out) -> e (W_out, b_out)`, the tied mask group
//!   `m_c <- e m_c, m_Delta <- e m_Delta` over the write tensor's components: a
//!   `cI` mask, invariant under every change of component basis. The next block
//!   reads the summed stream ([`decoder_layer`]).
//!
//! The analytic pullbacks of the gate, the component layer and the norms
//! ([`swiglu_hidden_pullback`], [`ComponentSwiglu::pullback`],
//! [`MaskedNorm::pullback`]) carry cotangents through the same summed inputs, so
//! a parameter gradient through a masked block (P16) never differentiates a
//! per-component copy. The component layer also returns the cotangent of every
//! masked factor's mask.
//!
//! Every downstream readout passes a final norm (`z = W_U N(h_L)`), so a mask on
//! any residual write reaches the logits through `(mean(h_L^2) + eps)^(-1/2)`,
//! a nonlinearity between the masked factors and the logits.

use super::rewrite::{ComponentMask, ComponentRead, ExactFactor, FactorRefusal, ShapeMismatch};
use gam_math::gaussian_gated::silu_derivatives;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip, s};
use std::fmt;

/// A refused gated rewrite.
#[derive(Clone, Debug, PartialEq)]
pub enum GatedRewriteError {
    /// An operand's shape disagrees with the rows it combines with. A vector's
    /// shape is `(len, 1)`.
    ShapeMismatch {
        what: &'static str,
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// Every input of a native primitive must be finite.
    NonFinite {
        what: &'static str,
        row: usize,
        column: usize,
        value: f64,
    },
    /// A normalization epsilon, declared by the source model, must be finite and
    /// nonnegative.
    InvalidEpsilon { epsilon: f64 },
    /// A normalization reads rows of width zero.
    EmptyResidual,
    /// `mean(h^2) + eps` of a row is zero or not finite, so the source's
    /// normalizer has no finite value: a zero row under `eps = 0`, or a row whose
    /// squares overflow.
    DegenerateNormalizer {
        row: usize,
        mean_square: f64,
        epsilon: f64,
    },
}

impl fmt::Display for GatedRewriteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch {
                what,
                expected,
                found,
            } => write!(
                formatter,
                "{what} has shape {found:?}, but the rows it combines with need {expected:?}"
            ),
            Self::NonFinite {
                what,
                row,
                column,
                value,
            } => write!(formatter, "{what}[{row}, {column}] is not finite: {value}"),
            Self::InvalidEpsilon { epsilon } => write!(
                formatter,
                "a normalization epsilon must be finite and nonnegative, got {epsilon}"
            ),
            Self::EmptyResidual => write!(formatter, "a normalization reads rows of width zero"),
            Self::DegenerateNormalizer {
                row,
                mean_square,
                epsilon,
            } => write!(
                formatter,
                "row {row}: the normalizer mean(h^2) + eps = {mean_square} + {epsilon} has no finite inverse square root"
            ),
        }
    }
}

impl std::error::Error for GatedRewriteError {}

fn require_shape(
    what: &'static str,
    expected: (usize, usize),
    found: (usize, usize),
) -> Result<(), GatedRewriteError> {
    if expected == found {
        Ok(())
    } else {
        Err(GatedRewriteError::ShapeMismatch {
            what,
            expected,
            found,
        })
    }
}

fn require_finite(what: &'static str, values: ArrayView2<'_, f64>) -> Result<(), GatedRewriteError> {
    match values.indexed_iter().find(|(_, value)| !value.is_finite()) {
        Some(((row, column), &value)) => Err(GatedRewriteError::NonFinite {
            what,
            row,
            column,
            value,
        }),
        None => Ok(()),
    }
}

/// A finite vector of one entry per residual column.
fn require_vector(what: &'static str, width: usize, vector: ArrayView1<'_, f64>) -> Result<(), GatedRewriteError> {
    require_shape(what, (width, 1), (vector.len(), 1))?;
    require_finite(what, vector.insert_axis(Axis(1)))
}

/// The gated hidden rows of a SwiGLU layer, `g = s(a_g) ⊙ a_u`, with the SiLU
/// gate `s(t) = t sigma(t)` read from its owner's jet.
///
/// `gate` and `up` are the summed masked pre-activation rows `x W_g(m_g)^T` and
/// `x W_u(m_u)^T` of the same normalized input rows. The down projection of `g`
/// is the block's write. Qwen3 carries no MLP bias; a source with biases passes
/// the homogeneous read.
pub fn swiglu_hidden(
    gate: ArrayView2<'_, f64>,
    up: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, GatedRewriteError> {
    require_shape("SwiGLU up pre-activations", gate.dim(), up.dim())?;
    require_finite("SwiGLU gate pre-activations", gate)?;
    require_finite("SwiGLU up pre-activations", up)?;
    Ok(Zip::from(gate)
        .and(up)
        .map_collect(|&gate_value, &up_value| silu_derivatives(gate_value)[0] * up_value))
}

/// Cotangents of a SwiGLU layer's two pre-activation rows.
#[derive(Clone, Debug, PartialEq)]
pub struct SwigluCotangents {
    /// `dl/da_g = g_bar ⊙ s'(a_g) ⊙ a_u`.
    pub gate: Array2<f64>,
    /// `dl/da_u = g_bar ⊙ s(a_g)`.
    pub up: Array2<f64>,
}

/// The analytic pullback of [`swiglu_hidden`] at the pre-activation rows for the
/// hidden cotangent rows `g_bar`, with `s` and `s'` read from the SiLU jet's owner.
pub fn swiglu_hidden_pullback(
    gate: ArrayView2<'_, f64>,
    up: ArrayView2<'_, f64>,
    hidden_cotangent: ArrayView2<'_, f64>,
) -> Result<SwigluCotangents, GatedRewriteError> {
    require_shape("SwiGLU up pre-activations", gate.dim(), up.dim())?;
    require_shape("SwiGLU hidden cotangent", gate.dim(), hidden_cotangent.dim())?;
    require_finite("SwiGLU gate pre-activations", gate)?;
    require_finite("SwiGLU up pre-activations", up)?;
    require_finite("SwiGLU hidden cotangent", hidden_cotangent)?;
    let mut gate_cotangent = Array2::zeros(gate.raw_dim());
    let mut up_cotangent = Array2::zeros(gate.raw_dim());
    Zip::from(&mut gate_cotangent)
        .and(&mut up_cotangent)
        .and(gate)
        .and(up)
        .and(hidden_cotangent)
        .for_each(|gate_slot, up_slot, &gate_value, &up_value, &cotangent| {
            let jet = silu_derivatives(gate_value);
            *gate_slot = cotangent * jet[1] * up_value;
            *up_slot = cotangent * jet[0];
        });
    Ok(SwigluCotangents {
        gate: gate_cotangent,
        up: up_cotangent,
    })
}

/// The three weights of a SwiGLU layer, in torch `Linear` layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwigluFactor {
    /// `W_g`, `H x d`: the SiLU gate read.
    Gate,
    /// `W_u`, `H x d`: the up read.
    Up,
    /// `W_d`, `d x H`: the write back into the residual stream.
    Down,
}

impl fmt::Display for SwigluFactor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Gate => "gate",
            Self::Up => "up",
            Self::Down => "down",
        })
    }
}

/// A refused SwiGLU block construction or execution.
#[derive(Debug)]
pub enum SwigluError {
    /// The input rows or a read do not compose with the weights.
    Shape(ShapeMismatch),
    /// A native primitive refused its summed input.
    Primitive(GatedRewriteError),
    /// A weight has no exact factor through its declared read.
    Factor {
        factor: SwigluFactor,
        refusal: FactorRefusal,
    },
}

impl From<ShapeMismatch> for SwigluError {
    fn from(mismatch: ShapeMismatch) -> Self {
        Self::Shape(mismatch)
    }
}

impl From<GatedRewriteError> for SwigluError {
    fn from(error: GatedRewriteError) -> Self {
        Self::Primitive(error)
    }
}

impl fmt::Display for SwigluError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(mismatch) => write!(formatter, "SwiGLU block shapes do not compose: {mismatch}"),
            Self::Primitive(error) => write!(formatter, "a SwiGLU native primitive refused: {error}"),
            Self::Factor { factor, refusal } => {
                write!(formatter, "the SwiGLU {factor} weight has no exact factor: {refusal}")
            }
        }
    }
}

impl std::error::Error for SwigluError {}

/// A bias-free SwiGLU layer on its original tensors (`Qwen3MLP`): the write of
/// the normalized input rows `x` is `(s(x W_g^T) ⊙ (x W_u^T)) W_d^T`, with no
/// residual added.
#[derive(Clone, Debug)]
pub struct NativeSwiglu {
    gate: Array2<f64>,
    up: Array2<f64>,
    down: Array2<f64>,
}

impl NativeSwiglu {
    /// The layer on `gate` and `up` (`H x d`) and `down` (`d x H`).
    pub fn new(gate: Array2<f64>, up: Array2<f64>, down: Array2<f64>) -> Result<Self, GatedRewriteError> {
        require_shape("SwiGLU up weight", gate.dim(), up.dim())?;
        require_shape("SwiGLU down weight", (gate.ncols(), gate.nrows()), down.dim())?;
        require_finite("SwiGLU gate weight", gate.view())?;
        require_finite("SwiGLU up weight", up.view())?;
        require_finite("SwiGLU down weight", down.view())?;
        Ok(Self { gate, up, down })
    }

    /// The residual width `d`.
    pub fn width(&self) -> usize {
        self.gate.ncols()
    }

    /// The source's `gate_proj` weight, `H x d`.
    pub fn gate(&self) -> ArrayView2<'_, f64> {
        self.gate.view()
    }

    /// The source's `up_proj` weight, `H x d`.
    pub fn up(&self) -> ArrayView2<'_, f64> {
        self.up.view()
    }

    /// The source's `down_proj` weight, `d x H`.
    pub fn down(&self) -> ArrayView2<'_, f64> {
        self.down.view()
    }

    /// Every stage of the layer on the original tensors.
    pub fn execute_stages(&self, inputs: ArrayView2<'_, f64>) -> Result<SwigluStages, GatedRewriteError> {
        require_shape("SwiGLU input rows", (inputs.nrows(), self.width()), inputs.dim())?;
        let gate = inputs.dot(&self.gate.t());
        let up = inputs.dot(&self.up.t());
        let hidden = swiglu_hidden(gate.view(), up.view())?;
        let write = hidden.dot(&self.down.t());
        Ok(SwigluStages {
            gate,
            up,
            hidden,
            write,
        })
    }

    /// The layer's write for every normalized input row, on the original tensors.
    pub fn execute(&self, inputs: ArrayView2<'_, f64>) -> Result<Array2<f64>, GatedRewriteError> {
        Ok(self.execute_stages(inputs)?.write)
    }
}

/// Every stage of a SwiGLU layer that torch exports as a module output: `gate_proj`,
/// `up_proj`, the `down_proj` input `s(gate) ⊙ up`, and `down_proj`, the write with no
/// residual added.
#[derive(Clone, Debug, PartialEq)]
pub struct SwigluStages {
    pub gate: Array2<f64>,
    pub up: Array2<f64>,
    pub hidden: Array2<f64>,
    pub write: Array2<f64>,
}

/// How the three factors of a [`ComponentSwiglu`] execute.
#[derive(Clone, Copy, Debug)]
pub struct SwigluMask<'a> {
    pub gate: ComponentMask<'a>,
    pub up: ComponentMask<'a>,
    pub down: ComponentMask<'a>,
}

/// A SwiGLU layer executed in component coordinates under masks.
///
/// Every weight factors exactly as `W = U R` through mpd-rewrite's
/// [`ExactFactor`], and a masked factor reads `((x R^T) ⊙ m) U^T`, so the SiLU
/// gate and the Hadamard product receive the same summed pre-activations as the
/// edited tensors `U diag(m) R`. A factor left all on executes its original
/// tensor on its original path, so a layer with every factor all on is
/// bit-identical to [`NativeSwiglu::execute`].
#[derive(Clone, Debug)]
pub struct ComponentSwiglu {
    native: NativeSwiglu,
    gate: ExactFactor,
    up: ExactFactor,
    down: ExactFactor,
}

impl ComponentSwiglu {
    /// Factors every weight of `native` exactly through its declared read.
    pub fn new(
        native: NativeSwiglu,
        gate: ComponentRead<'_>,
        up: ComponentRead<'_>,
        down: ComponentRead<'_>,
    ) -> Result<Self, SwigluError> {
        let gate = solve_factor(SwigluFactor::Gate, native.gate.view(), gate)?;
        let up = solve_factor(SwigluFactor::Up, native.up.view(), up)?;
        let down = solve_factor(SwigluFactor::Down, native.down.view(), down)?;
        Ok(Self {
            native,
            gate,
            up,
            down,
        })
    }

    pub fn native(&self) -> &NativeSwiglu {
        &self.native
    }

    /// The exact factor of one weight.
    pub fn factor(&self, which: SwigluFactor) -> &ExactFactor {
        match which {
            SwigluFactor::Gate => &self.gate,
            SwigluFactor::Up => &self.up,
            SwigluFactor::Down => &self.down,
        }
    }

    /// Every stage of the layer under `mask`.
    pub fn execute_stages(
        &self,
        inputs: ArrayView2<'_, f64>,
        mask: SwigluMask<'_>,
    ) -> Result<SwigluStages, SwigluError> {
        let gate = read_through(self.native.gate.view(), &self.gate, inputs, mask.gate)?;
        let up = read_through(self.native.up.view(), &self.up, inputs, mask.up)?;
        let hidden = swiglu_hidden(gate.view(), up.view())?;
        let write = read_through(self.native.down.view(), &self.down, hidden.view(), mask.down)?;
        Ok(SwigluStages {
            gate,
            up,
            hidden,
            write,
        })
    }

    /// The layer's write for every normalized input row under `mask`.
    pub fn execute(&self, inputs: ArrayView2<'_, f64>, mask: SwigluMask<'_>) -> Result<Array2<f64>, SwigluError> {
        Ok(self.execute_stages(inputs, mask)?.write)
    }

    /// The analytic pullback of [`ComponentSwiglu::execute`] at `inputs` and `mask` for the write cotangent
    /// rows `y_bar`: the down factor, then the SiLU gate and the Hadamard product
    /// ([`swiglu_hidden_pullback`]), then both reads, with the input cotangents of the two reads summed.
    ///
    /// A masked factor pulls back through mpd-rewrite's [`ExactFactor::apply_masked_pullback`], which returns
    /// its input cotangent and its mask cotangent without forming `U diag(m) R`. A factor left all on pulls back
    /// as `y_bar W` on its original tensor and has no mask to differentiate.
    pub fn pullback(
        &self,
        inputs: ArrayView2<'_, f64>,
        mask: SwigluMask<'_>,
        write_cotangent: ArrayView2<'_, f64>,
    ) -> Result<SwigluPullback, SwigluError> {
        require_finite("SwiGLU write cotangent", write_cotangent)?;
        let stages = self.execute_stages(inputs, mask)?;
        let (hidden_cotangent, down_mask) =
            pull_through(self.native.down.view(), &self.down, stages.hidden.view(), mask.down, write_cotangent)?;
        let hidden = swiglu_hidden_pullback(stages.gate.view(), stages.up.view(), hidden_cotangent.view())?;
        let (gate_inputs, gate_mask) =
            pull_through(self.native.gate.view(), &self.gate, inputs, mask.gate, hidden.gate.view())?;
        let (up_inputs, up_mask) = pull_through(self.native.up.view(), &self.up, inputs, mask.up, hidden.up.view())?;
        Ok(SwigluPullback {
            inputs: gate_inputs + &up_inputs,
            gate_mask,
            up_mask,
            down_mask,
        })
    }
}

/// Cotangents of a [`ComponentSwiglu`] layer's input rows and of every masked factor's mask.
#[derive(Clone, Debug, PartialEq)]
pub struct SwigluPullback {
    /// `dl/dx` for every input row, through both reads.
    pub inputs: Array2<f64>,
    /// `dl/dm_g`, or `None` when the gate factor executed all on.
    pub gate_mask: Option<Array1<f64>>,
    /// `dl/dm_u`, or `None` when the up factor executed all on.
    pub up_mask: Option<Array1<f64>>,
    /// `dl/dm_d`, or `None` when the down factor executed all on.
    pub down_mask: Option<Array1<f64>>,
}

fn solve_factor(
    which: SwigluFactor,
    weight: ArrayView2<'_, f64>,
    read: ComponentRead<'_>,
) -> Result<ExactFactor, SwigluError> {
    ExactFactor::solve_write(weight, read.read, read.candidate_write).map_err(|refusal| SwigluError::Factor {
        factor: which,
        refusal,
    })
}

/// `x W^T` on the original tensor with the factor all on, else `U M R x`
/// matrix-free through the factor.
fn read_through(
    native: ArrayView2<'_, f64>,
    factor: &ExactFactor,
    inputs: ArrayView2<'_, f64>,
    mask: ComponentMask<'_>,
) -> Result<Array2<f64>, ShapeMismatch> {
    match mask {
        ComponentMask::AllOn => {
            if inputs.ncols() == native.ncols() {
                Ok(inputs.dot(&native.t()))
            } else {
                Err(ShapeMismatch {
                    what: "SwiGLU read input width",
                    expected: native.ncols(),
                    found: inputs.ncols(),
                })
            }
        }
        ComponentMask::Components(mask) => factor.apply_masked(inputs, mask),
    }
}

/// The pullback of [`read_through`] at `inputs` for the output cotangent rows `y_bar`: `y_bar W` on the original
/// tensor with the factor all on, else the factor's masked-apply pullback with its mask cotangent.
fn pull_through(
    native: ArrayView2<'_, f64>,
    factor: &ExactFactor,
    inputs: ArrayView2<'_, f64>,
    mask: ComponentMask<'_>,
    output_cotangent: ArrayView2<'_, f64>,
) -> Result<(Array2<f64>, Option<Array1<f64>>), ShapeMismatch> {
    match mask {
        ComponentMask::AllOn => {
            if output_cotangent.ncols() != native.nrows() {
                Err(ShapeMismatch {
                    what: "SwiGLU read cotangent width",
                    expected: native.nrows(),
                    found: output_cotangent.ncols(),
                })
            } else if output_cotangent.nrows() != inputs.nrows() {
                Err(ShapeMismatch {
                    what: "SwiGLU read cotangent rows",
                    expected: inputs.nrows(),
                    found: output_cotangent.nrows(),
                })
            } else {
                Ok((output_cotangent.dot(&native), None))
            }
        }
        ComponentMask::Components(mask) => {
            let cotangents = factor.apply_masked_pullback(inputs, mask, output_cotangent)?;
            Ok((cotangents.inputs, Some(cotangents.mask)))
        }
    }
}

/// The homogeneous read `(x, 1)` of every input row: an affine map `(W, b)` on
/// `x` is the linear map `[W | b]` on it, so a bias is one more read column of a
/// component factor.
pub fn homogeneous_read(inputs: ArrayView2<'_, f64>) -> Array2<f64> {
    let mut read = Array2::ones((inputs.nrows(), inputs.ncols() + 1));
    read.slice_mut(s![.., ..inputs.ncols()]).assign(&inputs);
    read
}

/// The RMS normalizer `nu_r = (mean(x_r^2) + eps)^(-1/2)` of every row.
///
/// Each normalizer reads its whole current row, so a sum of masked
/// contributions is normalized once, never per contribution. It is formed only
/// from `+`, `*`, `/` and `sqrt`, which IEEE 754 rounds correctly, so a computed
/// `nu_r` lies within `gamma_(d + 4)` relative of its real value for rows of width
/// `d`: the squares, `d - 1` additions, the mean, `+ eps`, the square root and the
/// reciprocal. A per-head norm (Qwen3 `q_norm`, `k_norm`) reads the per-head rows.
pub fn rms_normalizers(rows: ArrayView2<'_, f64>, epsilon: f64) -> Result<Array1<f64>, GatedRewriteError> {
    if rows.ncols() == 0 {
        return Err(GatedRewriteError::EmptyResidual);
    }
    require_finite("normalized rows", rows)?;
    require_epsilon(epsilon)?;
    rows.rows()
        .into_iter()
        .enumerate()
        .map(|(row, values)| inverse_root_mean_square(row, values, epsilon))
        .collect()
}

/// A source model's native normalization, with the masked gain and bias it
/// applies after normalizing and the epsilon its configuration declares.
///
/// `gain` and `bias` are the edited tensors `w(m)` and `beta(m)`. The
/// normalizer of each row is computed on that whole row of the residual passed
/// to [`MaskedNorm::apply`], which is the current masked stream, never a cached
/// clean one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MaskedNorm<'a> {
    /// `Qwen3RMSNorm`: `w ⊙ (h (mean(h^2) + eps)^(-1/2))`.
    Rms {
        epsilon: f64,
        gain: ArrayView1<'a, f64>,
    },
    /// `torch.nn.LayerNorm`: `w ⊙ (x (mean(x^2) + eps)^(-1/2)) + beta` with
    /// `x = h - mean(h)`, the population variance.
    Layer {
        epsilon: f64,
        gain: ArrayView1<'a, f64>,
        bias: ArrayView1<'a, f64>,
    },
}

/// Cotangents of a normalization's residual rows and its tied parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct NormCotangents {
    /// `dl/dh` for every residual row.
    pub residual: Array2<f64>,
    /// `dl/dw`, summed over rows: every row reads the same gain.
    pub gain: Array1<f64>,
    /// `dl/dbeta` of a LayerNorm, summed over rows; an RMSNorm has no bias.
    pub bias: Option<Array1<f64>>,
}

impl MaskedNorm<'_> {
    /// The normalized residual rows, in the source's order of operations:
    /// normalize, multiply by the gain, then add the bias.
    pub fn apply(&self, residual: ArrayView2<'_, f64>) -> Result<Array2<f64>, GatedRewriteError> {
        let width = residual.ncols();
        if width == 0 {
            return Err(GatedRewriteError::EmptyResidual);
        }
        require_finite("normalized residual rows", residual)?;
        let mut output = Array2::zeros(residual.raw_dim());
        match *self {
            Self::Rms { epsilon, gain } => {
                require_vector("RMSNorm gain", width, gain)?;
                let normalizers = rms_normalizers(residual, epsilon)?;
                for ((output_row, input_row), &inverse_root) in output
                    .rows_mut()
                    .into_iter()
                    .zip(residual.rows())
                    .zip(normalizers.iter())
                {
                    Zip::from(output_row)
                        .and(gain)
                        .and(input_row)
                        .for_each(|slot, &weight, &value| *slot = weight * (value * inverse_root));
                }
            }
            Self::Layer {
                epsilon,
                gain,
                bias,
            } => {
                require_vector("LayerNorm gain", width, gain)?;
                require_vector("LayerNorm bias", width, bias)?;
                let centred = centred_rows(residual);
                let normalizers = rms_normalizers(centred.view(), epsilon)?;
                for ((output_row, centred_row), &inverse_root) in output
                    .rows_mut()
                    .into_iter()
                    .zip(centred.rows())
                    .zip(normalizers.iter())
                {
                    Zip::from(output_row)
                        .and(gain)
                        .and(centred_row)
                        .and(bias)
                        .for_each(|slot, &weight, &value, &offset| {
                            *slot = weight * (value * inverse_root) + offset
                        });
                }
            }
        }
        Ok(output)
    }

    /// The analytic pullback of [`MaskedNorm::apply`] at `residual` for the output
    /// cotangent rows `y_bar`.
    ///
    /// Per row, with `z = w ⊙ y_bar` and `nu = (mean(x^2) + eps)^(-1/2)`, the
    /// normalization `n = x nu` pulls back as `x_bar = nu z - (nu^3 / d)(z^T x) x`.
    /// An RMSNorm reads `x = h`, so `h_bar = x_bar`. A LayerNorm reads `x = P h`,
    /// and the centring projector is symmetric, so `h_bar = x_bar - mean(x_bar)`.
    /// The gain cotangent sums `y_bar ⊙ n` over rows, and a LayerNorm's bias
    /// cotangent sums `y_bar`.
    pub fn pullback(
        &self,
        residual: ArrayView2<'_, f64>,
        output_cotangent: ArrayView2<'_, f64>,
    ) -> Result<NormCotangents, GatedRewriteError> {
        let width = residual.ncols();
        if width == 0 {
            return Err(GatedRewriteError::EmptyResidual);
        }
        require_shape("normalization output cotangent", residual.dim(), output_cotangent.dim())?;
        require_finite("normalized residual rows", residual)?;
        require_finite("normalization output cotangent", output_cotangent)?;
        let (epsilon, gain, centring) = match *self {
            Self::Rms { epsilon, gain } => {
                require_vector("RMSNorm gain", width, gain)?;
                (epsilon, gain, false)
            }
            Self::Layer {
                epsilon,
                gain,
                bias,
            } => {
                require_vector("LayerNorm gain", width, gain)?;
                require_vector("LayerNorm bias", width, bias)?;
                (epsilon, gain, true)
            }
        };
        let read = if centring {
            centred_rows(residual)
        } else {
            residual.to_owned()
        };
        let normalizers = rms_normalizers(read.view(), epsilon)?;
        let mut residual_cotangent = Array2::zeros(residual.raw_dim());
        let mut gain_cotangent = Array1::zeros(width);
        for (((cotangent_row, read_row), output_row), &inverse_root) in residual_cotangent
            .rows_mut()
            .into_iter()
            .zip(read.rows())
            .zip(output_cotangent.rows())
            .zip(normalizers.iter())
        {
            let scaled = Zip::from(gain)
                .and(output_row)
                .map_collect(|&weight, &cotangent| weight * cotangent);
            let radial = inverse_root * inverse_root * inverse_root * scaled.dot(&read_row) / width as f64;
            let mut read_cotangent = Zip::from(&scaled)
                .and(read_row)
                .map_collect(|&direction, &value| inverse_root * direction - radial * value);
            if centring {
                let cotangent_mean = read_cotangent.sum() / width as f64;
                read_cotangent.mapv_inplace(|value| value - cotangent_mean);
            }
            Zip::from(cotangent_row)
                .and(&read_cotangent)
                .for_each(|slot, &value| *slot = value);
            Zip::from(&mut gain_cotangent)
                .and(output_row)
                .and(read_row)
                .for_each(|slot, &cotangent, &value| *slot += cotangent * (value * inverse_root));
        }
        Ok(NormCotangents {
            residual: residual_cotangent,
            gain: gain_cotangent,
            bias: centring.then(|| output_cotangent.sum_axis(Axis(0))),
        })
    }
}

/// Every row minus its own mean, the LayerNorm read `P h`.
fn centred_rows(residual: ArrayView2<'_, f64>) -> Array2<f64> {
    let width = residual.ncols() as f64;
    let mut centred = residual.to_owned();
    for mut row in centred.rows_mut() {
        let mean = row.sum() / width;
        row.mapv_inplace(|value| value - mean);
    }
    centred
}

fn require_epsilon(epsilon: f64) -> Result<(), GatedRewriteError> {
    if epsilon.is_finite() && epsilon >= 0.0 {
        Ok(())
    } else {
        Err(GatedRewriteError::InvalidEpsilon { epsilon })
    }
}

/// `(mean(x^2) + eps)^(-1/2)` of one finite row under a valid epsilon.
fn inverse_root_mean_square(
    row: usize,
    values: ArrayView1<'_, f64>,
    epsilon: f64,
) -> Result<f64, GatedRewriteError> {
    let mean_square = values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64;
    let denominator = mean_square + epsilon;
    if denominator.is_finite() && denominator > 0.0 {
        Ok(denominator.sqrt().recip())
    } else {
        Err(GatedRewriteError::DegenerateNormalizer {
            row,
            mean_square,
            epsilon,
        })
    }
}

/// How a decoder layer's two blocks read and write the residual stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidualLayout {
    /// Qwen3: `h_1 = h + attention(h)`, then `h_2 = h_1 + mlp(h_1)`. The MLP reads
    /// the stream the attention write already moved.
    Sequential,
    /// GPT-NeoX `use_parallel_residual`: `h_2 = mlp(h) + attention(h) + h`. Both
    /// blocks read the layer input, so at a fixed input the layer is affine in
    /// the two write edges.
    Parallel,
}

/// One decoder layer of the residual stream rows, with each block evaluated on
/// the current masked stream and its write summed into the stream in the
/// source's order.
///
/// Each block owns its input normalization and its masked write tensors,
/// including a write bias inside its linear map as torch's `Linear` computes it,
/// so an edge mask is carried by the block's write (R5), not by this function.
pub fn decoder_layer<E, Attention, Mlp>(
    layout: ResidualLayout,
    residual: ArrayView2<'_, f64>,
    attention: Attention,
    mlp: Mlp,
) -> Result<Array2<f64>, E>
where
    E: From<GatedRewriteError>,
    Attention: FnOnce(ArrayView2<'_, f64>) -> Result<Array2<f64>, E>,
    Mlp: FnOnce(ArrayView2<'_, f64>) -> Result<Array2<f64>, E>,
{
    match layout {
        ResidualLayout::Sequential => {
            let attention_write = attention(residual)?;
            require_shape("attention write", residual.dim(), attention_write.dim())?;
            let after_attention = &residual + &attention_write;
            let mlp_write = mlp(after_attention.view())?;
            require_shape("MLP write", residual.dim(), mlp_write.dim())?;
            Ok(after_attention + &mlp_write)
        }
        ResidualLayout::Parallel => {
            let attention_write = attention(residual)?;
            require_shape("attention write", residual.dim(), attention_write.dim())?;
            let mlp_write = mlp(residual)?;
            require_shape("MLP write", residual.dim(), mlp_write.dim())?;
            Ok(mlp_write + &attention_write + &residual)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ComponentSwiglu, GatedRewriteError, MaskedNorm, NativeSwiglu, ResidualLayout, SwigluError, SwigluFactor,
        SwigluMask, decoder_layer, homogeneous_read, rms_normalizers, swiglu_hidden, swiglu_hidden_pullback,
    };
    use crate::parameter_decomposition::rewrite::{ComponentMask, ComponentRead, FactorRefusal};
    use gam_math::gaussian_gated::silu_derivatives;
    use ndarray::{Array, Array1, Array2, ArrayView2, Axis, Dimension, Zip, arr0, array, s};
    use qd::Quad;
    use rand::rngs::StdRng;
    use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
    use rand::{RngExt, SeedableRng};

    /// A Lipschitz constant of the SiLU gate: `s' = sigma + t sigma (1 - sigma)`
    /// and `sigma (1 - sigma) = e^-|t| / (1 + e^-|t|)^2 <= e^-|t|`, so
    /// `|s'| <= 1 + sup_t t e^-t = 1 + 1/e < 1 + 3/8`, because `e > 8/3`.
    const SILU_LIPSCHITZ: f64 = 1.375;

    /// `|s''| = |2 sigma' + t sigma''| <= 1/2 + 1/e < 7/8`: `sigma' <= 1/4`, and
    /// `|sigma''| = sigma' |1 - 2 sigma| <= sigma' <= e^-|t|` with `sup t e^-t = 1/e < 3/8`.
    const SILU_SECOND_BOUND: f64 = 0.875;

    /// `|s'''| = |3 sigma'' + t sigma'''| <= 3/4 + 1/e < 9/8`: `|sigma''| <= sigma' <= 1/4`,
    /// and `|sigma'''| = sigma' |1 - 6 sigma'| <= sigma' <= e^-|t|` because `0 <= sigma' <= 1/4`.
    const SILU_THIRD_BOUND: f64 = 1.125;

    const ROWS: usize = 3;
    const INPUT_DIM: usize = 6;
    const HIDDEN_DIM: usize = 5;
    const GATE_COMPONENTS: usize = 9;
    const UP_COMPONENTS: usize = 9;
    const DOWN_COMPONENTS: usize = 7;

    fn silu(t: f64) -> f64 {
        silu_derivatives(t)[0]
    }

    /// Divides a computed tolerance by `1 - gamma_(k + 1)`, where `k` bounds the
    /// rounded operations along any chain of the tolerance arithmetic and the
    /// extra operation is this division, so the computed tolerance dominates its
    /// real-arithmetic value.
    fn inflate<D: Dimension>(tolerance: Array<f64, D>, operations: usize) -> Array<f64, D> {
        let factor = 1.0 - accumulation_growth(operations + 1);
        tolerance.mapv(|value| value / factor)
    }

    /// `2^-100`: qd's double-double resolves about `2^-104` relative per operation,
    /// so this per-operation allowance covers the oracle's own error.
    fn quad_relative() -> f64 {
        0.5_f64.powi(100)
    }

    fn quad(value: f64) -> Quad {
        Quad::from_f64(value)
    }

    fn quad_to_f64(value: Quad) -> f64 {
        value.0 + value.1
    }

    fn quad_logistic(t: Quad) -> Quad {
        quad(1.0) / (quad(1.0) + (-t).exp())
    }

    fn quad_silu(t: Quad) -> Quad {
        t * quad_logistic(t)
    }

    /// `|fl(s(t)) - s(t)|` at a floating-point `t`, measured against the
    /// double-double `t sigma(t)`. IEEE 754 does not bound the error of libm's
    /// `exp`, so this is measured rather than assumed; rounding the difference to
    /// f64 costs at most `u` of it, and the oracle's own error, about 106 bits
    /// down, lies below the added `u |fl(s(t))|`.
    fn silu_evaluation_error(t: f64) -> f64 {
        let difference = quad(silu(t)) - quad_silu(quad(t));
        quad_to_f64(difference).abs() / (1.0 - UNIT_ROUNDOFF) + UNIT_ROUNDOFF * silu(t).abs()
    }

    /// `|fl(s'(t)) - s'(t)|`, measured against the double-double `sigma + t sigma (1 - sigma)`.
    fn silu_slope_evaluation_error(t: f64) -> f64 {
        let sigma = quad_logistic(quad(t));
        let oracle = sigma + quad(t) * sigma * (quad(1.0) - sigma);
        let slope = silu_derivatives(t)[1];
        quad_to_f64(quad(slope) - oracle).abs() / (1.0 - UNIT_ROUNDOFF) + UNIT_ROUNDOFF * slope.abs()
    }

    fn uniform_rows(rng: &mut StdRng, rows: usize, cols: usize, low: f64, high: f64) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || rng.random_range(low..high))
    }

    #[derive(Clone, Copy)]
    enum Route {
        /// The rewrite: `((x R^T) ⊙ m) U^T`, never forming the edited tensor.
        Factored,
        /// Direct execution: materialize `W(m) = U diag(m) R`, then apply it.
        Direct,
    }

    struct Factor {
        writes: Array2<f64>,
        reads: Array2<f64>,
    }

    impl Factor {
        fn random(rng: &mut StdRng, outputs: usize, components: usize, inputs: usize) -> Self {
            Self {
                writes: uniform_rows(rng, outputs, components, -1.0, 1.0),
                reads: uniform_rows(rng, components, inputs, -1.0, 1.0),
            }
        }

        fn apply(&self, route: Route, mask: &Array1<f64>, inputs: ArrayView2<'_, f64>) -> Array2<f64> {
            match route {
                Route::Factored => (inputs.dot(&self.reads.t()) * mask).dot(&self.writes.t()),
                Route::Direct => {
                    let edited = self.writes.dot(&Array2::from_diag(mask)).dot(&self.reads);
                    inputs.dot(&edited.t())
                }
            }
        }

        /// On either route every term `u_ic m_c r_ck x_k` of an output entry passes
        /// two products, at most `C - 1` additions over components, one product
        /// with `x_k` and at most `d - 1` additions over inputs: `C + d + 1`
        /// rounded operations.
        fn operations_per_term(&self) -> usize {
            self.writes.ncols() + self.reads.ncols() + 1
        }

        /// `((|x| |R|^T) ⊙ |m|) |U|^T`, the sum of `|u_ic m_c r_ck x_k|` over the
        /// terms of each output entry.
        fn term_magnitudes(&self, mask: &Array1<f64>, inputs: ArrayView2<'_, f64>) -> Array2<f64> {
            (inputs.mapv(f64::abs).dot(&self.reads.mapv(f64::abs).t()) * &mask.mapv(f64::abs))
                .dot(&self.writes.mapv(f64::abs).t())
        }

        /// A bound on `|fl(x W(m)^T) - x W(m)^T|` on either route. The magnitudes
        /// are computed with the same structure, so dividing by `1 - gamma_n`
        /// makes them dominate their real values.
        fn roundoff_radius(&self, mask: &Array1<f64>, inputs: ArrayView2<'_, f64>) -> Array2<f64> {
            let bound = accumulation_growth(self.operations_per_term());
            self.term_magnitudes(mask, inputs)
                .mapv(|magnitude| bound * magnitude / (1.0 - bound))
        }

        /// Every term `y_ri u_ic m_c r_cj` of an adjoint entry `((y U) ⊙ m) R` passes the products with `u`, `m`
        /// and `r`, at most `p - 1` additions over outputs and at most `C - 1` additions over components.
        fn adjoint_operations_per_term(&self) -> usize {
            self.writes.nrows() + self.writes.ncols() + 1
        }

        /// `((|y| |U|) ⊙ |m|) |R|`, the sum of `|y_ri u_ic m_c r_cj|` over the terms of each adjoint entry.
        fn adjoint_term_magnitudes(&self, mask: &Array1<f64>, cotangent: ArrayView2<'_, f64>) -> Array2<f64> {
            (cotangent.mapv(f64::abs).dot(&self.writes.mapv(f64::abs)) * &mask.mapv(f64::abs))
                .dot(&self.reads.mapv(f64::abs))
        }

        /// A radius on `fl(((y U) ⊙ m) R)` against the real adjoint at the real cotangent, given the computed
        /// cotangent and its radius: the adjoint's own rounding plus the cotangent's radius carried through
        /// `|U| |m| |R|`.
        fn adjoint_radius(
            &self,
            mask: &Array1<f64>,
            cotangent: ArrayView2<'_, f64>,
            cotangent_radius: ArrayView2<'_, f64>,
        ) -> Array2<f64> {
            let bound = accumulation_growth(self.adjoint_operations_per_term());
            self.adjoint_term_magnitudes(mask, cotangent)
                .mapv(|magnitude| bound * magnitude / (1.0 - bound))
                + &self.adjoint_term_magnitudes(mask, cotangent_radius)
        }

        /// A radius on the mask cotangent `m_bar_c = sum_r (y U)_rc (x R^T)_rc` from a computed cotangent and
        /// input with their radii. `P = y U` lies within `gamma_p |y| |U| + |e_y| |U|` and `G = x R^T` within
        /// `gamma_d |x| |R|^T + e_x |R|^T`; then `|P G - P* G*| <= e_P |G| + (|P| + e_P) e_G` per row, and the
        /// product and the row sum add `gamma_rows` of `sum_r |P G|`.
        fn mask_cotangent_radius(
            &self,
            inputs: ArrayView2<'_, f64>,
            input_radius: ArrayView2<'_, f64>,
            cotangent: ArrayView2<'_, f64>,
            cotangent_radius: ArrayView2<'_, f64>,
        ) -> Array1<f64> {
            let absolute_writes = self.writes.mapv(f64::abs);
            let absolute_reads = self.reads.mapv(f64::abs);
            let output_growth = accumulation_growth(self.writes.nrows());
            let input_growth = accumulation_growth(self.reads.ncols());
            let row_growth = accumulation_growth(inputs.nrows());
            let projected = cotangent.dot(&self.writes);
            let coordinates = inputs.dot(&self.reads.t());
            let projected_radius = cotangent
                .mapv(f64::abs)
                .dot(&absolute_writes)
                .mapv(|value| output_growth * value / (1.0 - output_growth))
                + &cotangent_radius.dot(&absolute_writes);
            let coordinates_radius = inputs
                .mapv(f64::abs)
                .dot(&absolute_reads.t())
                .mapv(|value| input_growth * value / (1.0 - input_growth))
                + &input_radius.dot(&absolute_reads.t());
            Zip::from(&projected)
                .and(&coordinates)
                .and(&projected_radius)
                .and(&coordinates_radius)
                .map_collect(|&p, &g, &e_p, &e_g| {
                    e_p * g.abs() + (p.abs() + e_p) * e_g + row_growth * (p * g).abs() / (1.0 - row_growth)
                })
                .sum_axis(Axis(0))
        }
    }

    struct SwigluFixture {
        gate: Factor,
        up: Factor,
        down: Factor,
    }

    impl SwigluFixture {
        fn random(rng: &mut StdRng) -> Self {
            Self {
                gate: Factor::random(rng, HIDDEN_DIM, GATE_COMPONENTS, INPUT_DIM),
                up: Factor::random(rng, HIDDEN_DIM, UP_COMPONENTS, INPUT_DIM),
                down: Factor::random(rng, INPUT_DIM, DOWN_COMPONENTS, HIDDEN_DIM),
            }
        }

        /// The longest chain of rounded operations in [`swiglu_block`]'s radius:
        /// a read radius takes `n + 2` (its magnitudes, `gamma`, `1 - gamma`), the
        /// SiLU error `+ 2`, the Hadamard stage's three terms `+ 4`, the down read
        /// over the hidden radius `n_d`, its own term `+ 1`, and two routes `+ 1`.
        fn tolerance_operations(&self) -> usize {
            self.gate
                .operations_per_term()
                .max(self.up.operations_per_term())
                + self.down.operations_per_term()
                + 10
        }
    }

    struct SwigluMasks {
        gate: Array1<f64>,
        up: Array1<f64>,
        down: Array1<f64>,
    }

    impl SwigluMasks {
        fn all_on() -> Self {
            Self {
                gate: Array1::ones(GATE_COMPONENTS),
                up: Array1::ones(UP_COMPONENTS),
                down: Array1::ones(DOWN_COMPONENTS),
            }
        }
    }

    #[derive(Clone, Copy, Debug)]
    enum MaskKind {
        AllOn,
        Continuous,
        Binary,
        Signed,
    }

    impl MaskKind {
        const ALL: [Self; 4] = [Self::AllOn, Self::Continuous, Self::Binary, Self::Signed];

        fn draw(self, rng: &mut StdRng, components: usize) -> Array1<f64> {
            match self {
                Self::AllOn => Array1::ones(components),
                Self::Continuous => {
                    Array1::from_shape_simple_fn(components, || rng.random_range(0.0..1.0))
                }
                Self::Binary => Array1::from_shape_simple_fn(components, || {
                    if rng.random_range(0..2) == 0 { 0.0 } else { 1.0 }
                }),
                Self::Signed => {
                    Array1::from_shape_simple_fn(components, || rng.random_range(-1.0..1.0))
                }
            }
        }
    }

    fn one_hot(components: usize, index: usize) -> Array1<f64> {
        Array1::from_shape_fn(components, |component| if component == index { 1.0 } else { 0.0 })
    }

    /// A radius on the computed hidden rows `g = fl(fl(s(a_g)) a_u)` against the real ones, from the computed
    /// pre-activations and their radii `e_g`, `e_u`.
    ///
    /// With `rho` the measured SiLU error, `|fl(s(a_g)) - s(a_g)| <= rho + L e_g`, so the Hadamard stage has
    /// radius `u |g| / (1 - u) + (rho + L e_g) |a_u| + (|s(a_g)| + rho + L e_g) e_u`.
    fn hidden_radius(
        gate: &Array2<f64>,
        up: &Array2<f64>,
        hidden: &Array2<f64>,
        gate_radius: &Array2<f64>,
        up_radius: &Array2<f64>,
    ) -> Array2<f64> {
        Zip::from(gate)
            .and(up)
            .and(hidden)
            .and(gate_radius)
            .and(up_radius)
            .map_collect(|&gate_value, &up_value, &hidden_value, &gate_error, &up_error| {
                let silu_error = silu_evaluation_error(gate_value) + SILU_LIPSCHITZ * gate_error;
                UNIT_ROUNDOFF * hidden_value.abs() / (1.0 - UNIT_ROUNDOFF)
                    + silu_error * up_value.abs()
                    + (silu(gate_value).abs() + silu_error) * up_error
            })
    }

    /// `|x y z - x* y* z*| <= e_x |y| |z| + (|x| + e_x) e_y |z| + (|x| + e_x)(|y| + e_y) e_z` from computed
    /// factors and their radii.
    fn triple_product_radius(x: f64, e_x: f64, y: f64, e_y: f64, z: f64, e_z: f64) -> f64 {
        e_x * y.abs() * z.abs() + (x.abs() + e_x) * e_y * z.abs() + (x.abs() + e_x) * (y.abs() + e_y) * e_z
    }

    /// `fl(F_m(x))` for every input row on one route, with a radius bounding
    /// `|fl(F_m(x)) - F_m(x)|` entrywise: the read radii, the Hadamard stage's
    /// [`hidden_radius`], and the down read's magnitudes over that radius plus its own.
    fn swiglu_block(
        fixture: &SwigluFixture,
        masks: &SwigluMasks,
        inputs: ArrayView2<'_, f64>,
        route: Route,
    ) -> (Array2<f64>, Array2<f64>) {
        let gate = fixture.gate.apply(route, &masks.gate, inputs);
        let up = fixture.up.apply(route, &masks.up, inputs);
        let hidden = swiglu_hidden(gate.view(), up.view()).expect("the fixture pre-activations are finite");
        let gate_radius = fixture.gate.roundoff_radius(&masks.gate, inputs);
        let up_radius = fixture.up.roundoff_radius(&masks.up, inputs);
        let hidden_error = hidden_radius(&gate, &up, &hidden, &gate_radius, &up_radius);
        let write = fixture.down.apply(route, &masks.down, hidden.view());
        let radius = fixture.down.term_magnitudes(&masks.down, hidden_error.view())
            + &fixture.down.roundoff_radius(&masks.down, hidden.view());
        (write, radius)
    }

    /// `sum_c fl(F_(m_c)(x))` over `mask_at(c)` on the factored route, with a radius
    /// bounding its distance from the real sum: the blocks' radii plus
    /// `gamma_(components - 1)` of the summed magnitudes.
    fn separately_masked_sum(
        fixture: &SwigluFixture,
        inputs: ArrayView2<'_, f64>,
        components: usize,
        mask_at: impl Fn(usize) -> SwigluMasks,
    ) -> (Array2<f64>, Array2<f64>) {
        let mut sum = Array2::<f64>::zeros((inputs.nrows(), INPUT_DIM));
        let mut radius = Array2::<f64>::zeros((inputs.nrows(), INPUT_DIM));
        let mut magnitude = Array2::<f64>::zeros((inputs.nrows(), INPUT_DIM));
        for component in 0..components {
            let (write, write_radius) = swiglu_block(fixture, &mask_at(component), inputs, Route::Factored);
            sum += &write;
            radius += &write_radius;
            magnitude += &write.mapv(f64::abs);
        }
        let summation = accumulation_growth(components - 1);
        let summation_radius = magnitude.mapv(|value| summation * value / (1.0 - summation));
        (sum, radius + &summation_radius)
    }

    /// A random bias-free SwiGLU layer with overcomplete reads on every weight, its
    /// exact component factorization and input rows.
    fn random_component_swiglu(seed: u64) -> (ComponentSwiglu, Array2<f64>) {
        let mut rng = StdRng::seed_from_u64(seed);
        let native = NativeSwiglu::new(
            uniform_rows(&mut rng, HIDDEN_DIM, INPUT_DIM, -1.0, 1.0),
            uniform_rows(&mut rng, HIDDEN_DIM, INPUT_DIM, -1.0, 1.0),
            uniform_rows(&mut rng, INPUT_DIM, HIDDEN_DIM, -1.0, 1.0),
        )
        .expect("finite weights of composing shapes");
        let gate_read = uniform_rows(&mut rng, GATE_COMPONENTS, INPUT_DIM, -1.0, 1.0);
        let gate_candidate = uniform_rows(&mut rng, HIDDEN_DIM, GATE_COMPONENTS, -1.0, 1.0);
        let up_read = uniform_rows(&mut rng, UP_COMPONENTS, INPUT_DIM, -1.0, 1.0);
        let up_candidate = uniform_rows(&mut rng, HIDDEN_DIM, UP_COMPONENTS, -1.0, 1.0);
        let down_read = uniform_rows(&mut rng, DOWN_COMPONENTS, HIDDEN_DIM, -1.0, 1.0);
        let down_candidate = uniform_rows(&mut rng, INPUT_DIM, DOWN_COMPONENTS, -1.0, 1.0);
        let component = ComponentSwiglu::new(
            native,
            ComponentRead {
                read: gate_read.view(),
                candidate_write: gate_candidate.view(),
            },
            ComponentRead {
                read: up_read.view(),
                candidate_write: up_candidate.view(),
            },
            ComponentRead {
                read: down_read.view(),
                candidate_write: down_candidate.view(),
            },
        )
        .expect("overcomplete random reads resolve");
        let inputs = uniform_rows(&mut rng, ROWS, INPUT_DIM, -2.0, 2.0);
        (component, inputs)
    }

    #[test]
    fn swiglu_rewrite_equals_direct_edited_tensor_execution_for_every_mask_kind() {
        let mut rng = StdRng::seed_from_u64(2951);
        let fixture = SwigluFixture::random(&mut rng);
        let inputs = uniform_rows(&mut rng, ROWS, INPUT_DIM, -2.0, 2.0);
        for kind in MaskKind::ALL {
            let masks = SwigluMasks {
                gate: kind.draw(&mut rng, GATE_COMPONENTS),
                up: kind.draw(&mut rng, UP_COMPONENTS),
                down: kind.draw(&mut rng, DOWN_COMPONENTS),
            };
            let (rewritten, rewritten_radius) = swiglu_block(&fixture, &masks, inputs.view(), Route::Factored);
            let (executed, executed_radius) = swiglu_block(&fixture, &masks, inputs.view(), Route::Direct);
            let tolerance = inflate(rewritten_radius + &executed_radius, fixture.tolerance_operations());
            for (((row, column), &rewrite_value), (&direct_value, &bound)) in rewritten
                .indexed_iter()
                .zip(executed.iter().zip(tolerance.iter()))
            {
                assert!(
                    (rewrite_value - direct_value).abs() <= bound,
                    "{kind:?} masks, row {row}, output {column}: rewrite {rewrite_value:e}, direct \
                     execution {direct_value:e}, derived roundoff bound {bound:e}"
                );
            }
        }
    }

    #[test]
    fn swiglu_is_linear_in_up_masks_and_not_in_gate_masks() {
        let mut rng = StdRng::seed_from_u64(2952);
        let fixture = SwigluFixture::random(&mut rng);
        let inputs = uniform_rows(&mut rng, ROWS, INPUT_DIM, -2.0, 2.0);
        let (whole, whole_radius) = swiglu_block(&fixture, &SwigluMasks::all_on(), inputs.view(), Route::Factored);
        let operations = fixture.tolerance_operations() + UP_COMPONENTS.max(GATE_COMPONENTS) + 3;

        let (up_sum, up_radius) = separately_masked_sum(&fixture, inputs.view(), UP_COMPONENTS, |component| {
            SwigluMasks {
                up: one_hot(UP_COMPONENTS, component),
                ..SwigluMasks::all_on()
            }
        });
        let up_tolerance = inflate(up_radius + &whole_radius, operations);
        for (((row, column), &summed), (&whole_value, &bound)) in
            up_sum.indexed_iter().zip(whole.iter().zip(up_tolerance.iter()))
        {
            assert!(
                (summed - whole_value).abs() <= bound,
                "row {row}, output {column}: the up-mask split {summed:e} must equal the whole block \
                 {whole_value:e} within the derived bound {bound:e}"
            );
        }

        let (gate_sum, gate_radius) = separately_masked_sum(&fixture, inputs.view(), GATE_COMPONENTS, |component| {
            SwigluMasks {
                gate: one_hot(GATE_COMPONENTS, component),
                ..SwigluMasks::all_on()
            }
        });
        let gate_tolerance = inflate(gate_radius + &whole_radius, operations);
        let refuted = gate_sum
            .iter()
            .zip(whole.iter())
            .zip(gate_tolerance.iter())
            .filter(|&((&summed, &whole_value), &bound)| (summed - whole_value).abs() > bound)
            .count();
        assert!(
            refuted > 0,
            "the separately gated sum {gate_sum} must differ from the whole block {whole} beyond \
             the derived bound {gate_tolerance} somewhere; the same instrument accepts the up split"
        );
    }

    #[test]
    fn component_swiglu_all_on_is_the_native_layer_bit_for_bit() {
        let (component, inputs) = random_component_swiglu(2959);
        let all_on = SwigluMask {
            gate: ComponentMask::AllOn,
            up: ComponentMask::AllOn,
            down: ComponentMask::AllOn,
        };
        let native_stages = component.native().execute_stages(inputs.view()).expect("finite rows");
        let component_stages = component.execute_stages(inputs.view(), all_on).expect("finite rows");
        assert_eq!(
            component_stages, native_stages,
            "every factor all on must execute the original tensors on their original path, stage by stage"
        );
        let native = component.native().execute(inputs.view()).expect("finite rows");
        assert_eq!(native, native_stages.write, "execute must return the stages' write bit for bit");
        assert_eq!(
            component.execute(inputs.view(), all_on).expect("finite rows"),
            native,
            "every factor all on must execute the original tensors on their original path"
        );
        let ones = Array1::ones(GATE_COMPONENTS);
        let components_on = SwigluMask {
            gate: ComponentMask::Components(ones.view()),
            ..all_on
        };
        // The factored gate read at an all-ones mask is algebraically the same layer
        // but a different floating-point program, so bit identity is not automatic.
        let factored = component.execute(inputs.view(), components_on).expect("finite rows");
        assert!(
            factored.iter().zip(native.iter()).any(|(&rewritten, &original)| rewritten != original),
            "positive control: the factored gate read must differ from the native read in some bit"
        );
    }

    #[test]
    fn component_swiglu_equals_its_edited_tensors_under_every_mask_kind() {
        let (component, inputs) = random_component_swiglu(2960);
        let mut rng = StdRng::seed_from_u64(2961);
        let as_fixture = |which: SwigluFactor| Factor {
            writes: component.factor(which).write().to_owned(),
            reads: component.factor(which).read().to_owned(),
        };
        let fixture = SwigluFixture {
            gate: as_fixture(SwigluFactor::Gate),
            up: as_fixture(SwigluFactor::Up),
            down: as_fixture(SwigluFactor::Down),
        };
        for kind in MaskKind::ALL {
            let masks = SwigluMasks {
                gate: kind.draw(&mut rng, GATE_COMPONENTS),
                up: kind.draw(&mut rng, UP_COMPONENTS),
                down: kind.draw(&mut rng, DOWN_COMPONENTS),
            };
            let rewritten = component
                .execute(
                    inputs.view(),
                    SwigluMask {
                        gate: ComponentMask::Components(masks.gate.view()),
                        up: ComponentMask::Components(masks.up.view()),
                        down: ComponentMask::Components(masks.down.view()),
                    },
                )
                .expect("finite rows");
            let (factored, factored_radius) = swiglu_block(&fixture, &masks, inputs.view(), Route::Factored);
            let (executed, executed_radius) = swiglu_block(&fixture, &masks, inputs.view(), Route::Direct);
            assert_eq!(
                rewritten, factored,
                "{kind:?}: ComponentSwiglu must run the factored program ((x R^T) ⊙ m) U^T bit for bit"
            );
            let tolerance = inflate(factored_radius + &executed_radius, fixture.tolerance_operations());
            for (((row, column), &rewrite_value), (&direct_value, &bound)) in rewritten
                .indexed_iter()
                .zip(executed.iter().zip(tolerance.iter()))
            {
                assert!(
                    (rewrite_value - direct_value).abs() <= bound,
                    "{kind:?} masks, row {row}, output {column}: component layer {rewrite_value:e}, \
                     edited tensors {direct_value:e}, derived roundoff bound {bound:e}"
                );
            }
        }
    }

    #[test]
    fn component_swiglu_refuses_an_uncovering_read_and_mismatched_weights() {
        let mut rng = StdRng::seed_from_u64(2962);
        let native = || {
            NativeSwiglu::new(
                Array2::from_elem((HIDDEN_DIM, INPUT_DIM), 0.5),
                Array2::from_elem((HIDDEN_DIM, INPUT_DIM), -0.25),
                Array2::from_elem((INPUT_DIM, HIDDEN_DIM), 0.75),
            )
            .expect("composing shapes")
        };
        let narrow_read = uniform_rows(&mut rng, INPUT_DIM - 2, INPUT_DIM, -1.0, 1.0);
        let narrow_candidate = uniform_rows(&mut rng, HIDDEN_DIM, INPUT_DIM - 2, -1.0, 1.0);
        let wide_read = uniform_rows(&mut rng, GATE_COMPONENTS, INPUT_DIM, -1.0, 1.0);
        let wide_candidate = uniform_rows(&mut rng, HIDDEN_DIM, GATE_COMPONENTS, -1.0, 1.0);
        let down_read = uniform_rows(&mut rng, DOWN_COMPONENTS, HIDDEN_DIM, -1.0, 1.0);
        let down_candidate = uniform_rows(&mut rng, INPUT_DIM, DOWN_COMPONENTS, -1.0, 1.0);
        let narrow = ComponentRead {
            read: narrow_read.view(),
            candidate_write: narrow_candidate.view(),
        };
        let wide = ComponentRead {
            read: wide_read.view(),
            candidate_write: wide_candidate.view(),
        };
        let down = ComponentRead {
            read: down_read.view(),
            candidate_write: down_candidate.view(),
        };
        assert!(
            matches!(
                ComponentSwiglu::new(native(), narrow, wide, down),
                Err(SwigluError::Factor {
                    factor: SwigluFactor::Gate,
                    refusal: FactorRefusal::Uncovered { .. },
                })
            ),
            "a gate read with fewer components than input directions must be refused"
        );
        assert!(
            ComponentSwiglu::new(native(), wide, wide, down).is_ok(),
            "positive control: overcomplete reads on every weight are accepted"
        );
        assert!(matches!(
            NativeSwiglu::new(
                Array2::zeros((HIDDEN_DIM, INPUT_DIM)),
                Array2::zeros((HIDDEN_DIM, INPUT_DIM)),
                Array2::zeros((HIDDEN_DIM, INPUT_DIM)),
            ),
            Err(GatedRewriteError::ShapeMismatch {
                what: "SwiGLU down weight",
                ..
            })
        ));
    }

    #[test]
    fn swiglu_hidden_pullback_matches_a_double_double_central_difference() {
        let mut rng = StdRng::seed_from_u64(2957);
        let width = 7;
        let gate = uniform_rows(&mut rng, ROWS, width, -4.0, 4.0);
        let up = uniform_rows(&mut rng, ROWS, width, -2.0, 2.0);
        let cotangent = uniform_rows(&mut rng, ROWS, width, -1.0, 1.0);
        let gate_direction = uniform_rows(&mut rng, ROWS, width, -1.0, 1.0);
        let up_direction = uniform_rows(&mut rng, ROWS, width, -1.0, 1.0);
        let cotangents =
            swiglu_hidden_pullback(gate.view(), up.view(), cotangent.view()).expect("finite fixture");

        // `<c_g, v_g> + <c_u, v_u>`, summed in double-double so only the f64 entries round.
        let directional = |gate_cotangent: &Array2<f64>, up_cotangent: &Array2<f64>| {
            gate_direction.indexed_iter().fold(quad(0.0), |sum, (index, &direction)| {
                sum + quad(gate_cotangent[index]) * quad(direction)
                    + quad(up_cotangent[index]) * quad(up_direction[index])
            })
        };
        // `f(t) = <g_bar, s(a_g + t v_g) ⊙ (a_u + t v_u)>` in double-double.
        let block = |step: f64| {
            gate.indexed_iter().fold(quad(0.0), |sum, (index, &gate_value)| {
                let gate_point = quad(gate_value) + quad(step) * quad(gate_direction[index]);
                let up_point = quad(up[index]) + quad(step) * quad(up_direction[index]);
                sum + quad(cotangent[index]) * quad_silu(gate_point) * up_point
            })
        };
        let entries = ROWS * width;
        let step = 0.5_f64.powi(20);
        let central = (block(step) - block(-step)) / quad(2.0 * step);

        // f64 entry radii: `c_g = fl(fl(g_bar fl(s')) a_u)` and `c_u = fl(g_bar fl(s))`.
        let mut entry_radius = 0.0;
        let mut third_derivative = 0.0;
        let mut oracle_magnitude = 0.0;
        let mut predicted_magnitude = 0.0;
        for (index, &t) in gate.indexed_iter() {
            let gate_entry = cotangents.gate[index];
            let up_entry = cotangents.up[index];
            let hidden = cotangent[index];
            let gate_radius = accumulation_growth(2) * gate_entry.abs() / (1.0 - accumulation_growth(2))
                + (hidden * up[index]).abs() * silu_slope_evaluation_error(t);
            let up_radius =
                UNIT_ROUNDOFF * up_entry.abs() / (1.0 - UNIT_ROUNDOFF) + hidden.abs() * silu_evaluation_error(t);
            entry_radius += gate_radius * gate_direction[index].abs() + up_radius * up_direction[index].abs();
            // `f_i''' = g_bar_i [v_g^3 s''' (a_u + t v_u) + 3 v_g^2 v_u s'']` on `|t| <= step`.
            third_derivative += hidden.abs()
                * (gate_direction[index].abs().powi(3)
                    * SILU_THIRD_BOUND
                    * (up[index].abs() + step * up_direction[index].abs())
                    + 3.0 * gate_direction[index].powi(2) * up_direction[index].abs() * SILU_SECOND_BOUND);
            oracle_magnitude += hidden.abs()
                * (silu(t).abs() + silu_evaluation_error(t) + SILU_LIPSCHITZ * step * gate_direction[index].abs())
                * (up[index].abs() + step * up_direction[index].abs());
            predicted_magnitude +=
                (gate_entry * gate_direction[index]).abs() + (up_entry * up_direction[index]).abs();
        }
        // Central difference: `|D(step) - f'(0)| <= step^2 / 6 sup |f'''|`. The oracle's
        // own error: at most `entries + 8` operations per evaluation at `2^-100`, divided
        // by the step, and the predicted sum's `2 entries` operations.
        let tolerance = entry_radius
            + step * step / 6.0 * third_derivative
            + quad_relative() * (entries + 8) as f64 * oracle_magnitude / step
            + quad_relative() * (2 * entries) as f64 * predicted_magnitude;
        let tolerance = inflate(arr0(tolerance), 6 * entries + 16).into_scalar();
        let difference = quad_to_f64(directional(&cotangents.gate, &cotangents.up) - central).abs();
        assert!(
            difference / (1.0 - UNIT_ROUNDOFF) <= tolerance,
            "the SwiGLU pullback's directional derivative differs from the double-double central \
             difference by {difference:e}, beyond the derived bound {tolerance:e}"
        );
        let without_up_factor = Zip::from(&cotangent)
            .and(&gate)
            .map_collect(|&hidden, &gate_value| hidden * silu_derivatives(gate_value)[1]);
        let wrong_difference = quad_to_f64(directional(&without_up_factor, &cotangents.up) - central).abs();
        assert!(
            wrong_difference > tolerance,
            "positive control: a gate cotangent missing the up factor must differ beyond the bound \
             {tolerance:e}, got {wrong_difference:e}"
        );
    }

    #[test]
    fn component_swiglu_pullback_matches_a_double_double_central_difference() {
        let (component, inputs) = random_component_swiglu(2963);
        let mut rng = StdRng::seed_from_u64(2964);
        let as_fixture = |which: SwigluFactor| Factor {
            writes: component.factor(which).write().to_owned(),
            reads: component.factor(which).read().to_owned(),
        };
        let gate_factor = as_fixture(SwigluFactor::Gate);
        let up_factor = as_fixture(SwigluFactor::Up);
        let down_factor = as_fixture(SwigluFactor::Down);
        let masks = SwigluMasks {
            gate: MaskKind::Signed.draw(&mut rng, GATE_COMPONENTS),
            up: MaskKind::Signed.draw(&mut rng, UP_COMPONENTS),
            down: MaskKind::Signed.draw(&mut rng, DOWN_COMPONENTS),
        };
        let input_direction = uniform_rows(&mut rng, ROWS, INPUT_DIM, -1.0, 1.0);
        let gate_direction = Array1::from_shape_simple_fn(GATE_COMPONENTS, || rng.random_range(-1.0..1.0));
        let up_direction = Array1::from_shape_simple_fn(UP_COMPONENTS, || rng.random_range(-1.0..1.0));
        let down_direction = Array1::from_shape_simple_fn(DOWN_COMPONENTS, || rng.random_range(-1.0..1.0));
        let write_cotangent = uniform_rows(&mut rng, ROWS, INPUT_DIM, -1.0, 1.0);
        let mask = SwigluMask {
            gate: ComponentMask::Components(masks.gate.view()),
            up: ComponentMask::Components(masks.up.view()),
            down: ComponentMask::Components(masks.down.view()),
        };
        let pullback = component
            .pullback(inputs.view(), mask, write_cotangent.view())
            .expect("finite fixture");
        let gate_mask_cotangent = pullback.gate_mask.as_ref().expect("the gate factor ran masked");
        let up_mask_cotangent = pullback.up_mask.as_ref().expect("the up factor ran masked");
        let down_mask_cotangent = pullback.down_mask.as_ref().expect("the down factor ran masked");

        // `<x_bar, dx> + sum_f <m_bar_f, dm_f>`, summed in double-double so only the f64 entries round.
        let directional = |down_mask: &Array1<f64>| {
            let dot = |cotangent: &Array1<f64>, direction: &Array1<f64>| {
                Zip::from(cotangent)
                    .and(direction)
                    .fold(quad(0.0), |sum, &value, &step_direction| sum + quad(value) * quad(step_direction))
            };
            Zip::from(&pullback.inputs)
                .and(&input_direction)
                .fold(quad(0.0), |sum, &value, &direction| sum + quad(value) * quad(direction))
                + dot(gate_mask_cotangent, &gate_direction)
                + dot(up_mask_cotangent, &up_direction)
                + dot(down_mask, &down_direction)
        };

        // `U diag(m + t dm) R` applied to every row in double-double.
        let quad_read = |factor: &Factor, factor_mask: &Array1<f64>, direction: &Array1<f64>, rows: &[Vec<Quad>], step: f64| {
            rows.iter()
                .map(|row| {
                    let coordinates: Vec<Quad> = (0..factor.reads.nrows())
                        .map(|component| {
                            let read = (0..factor.reads.ncols())
                                .fold(quad(0.0), |sum, column| sum + quad(factor.reads[[component, column]]) * row[column]);
                            (quad(factor_mask[component]) + quad(step) * quad(direction[component])) * read
                        })
                        .collect();
                    (0..factor.writes.nrows())
                        .map(|output| {
                            coordinates
                                .iter()
                                .enumerate()
                                .fold(quad(0.0), |sum, (component, &coordinate)| {
                                    sum + quad(factor.writes[[output, component]]) * coordinate
                                })
                        })
                        .collect::<Vec<Quad>>()
                })
                .collect::<Vec<Vec<Quad>>>()
        };
        // `f(t) = <y_bar, F(x + t dx; m + t dm)>` in double-double.
        let evaluate = |step: f64| {
            let rows: Vec<Vec<Quad>> = (0..ROWS)
                .map(|row| {
                    (0..INPUT_DIM)
                        .map(|column| quad(inputs[[row, column]]) + quad(step) * quad(input_direction[[row, column]]))
                        .collect()
                })
                .collect();
            let gate = quad_read(&gate_factor, &masks.gate, &gate_direction, &rows, step);
            let up = quad_read(&up_factor, &masks.up, &up_direction, &rows, step);
            let hidden: Vec<Vec<Quad>> = gate
                .iter()
                .zip(&up)
                .map(|(gate_row, up_row)| gate_row.iter().zip(up_row).map(|(&g, &u)| quad_silu(g) * u).collect())
                .collect();
            let write = quad_read(&down_factor, &masks.down, &down_direction, &hidden, step);
            write.iter().enumerate().fold(quad(0.0), |total, (row, write_row)| {
                write_row
                    .iter()
                    .enumerate()
                    .fold(total, |sum, (column, &value)| sum + quad(write_cotangent[[row, column]]) * value)
            })
        };

        // Forward radii at the computed stages, which the pullback recomputes with the same operations.
        let gate = gate_factor.apply(Route::Factored, &masks.gate, inputs.view());
        let up = up_factor.apply(Route::Factored, &masks.up, inputs.view());
        let hidden = swiglu_hidden(gate.view(), up.view()).expect("finite rows");
        let zero_inputs = Array2::<f64>::zeros(inputs.raw_dim());
        let gate_radius = gate_factor.roundoff_radius(&masks.gate, inputs.view());
        let up_radius = up_factor.roundoff_radius(&masks.up, inputs.view());
        let hidden_error = hidden_radius(&gate, &up, &hidden, &gate_radius, &up_radius);

        // Backward radii: the down adjoint of the exact write cotangent, the gate and Hadamard pullback, both reads.
        let zero_write_cotangent = Array2::<f64>::zeros(write_cotangent.raw_dim());
        let hidden_cotangent = ((write_cotangent.dot(&down_factor.writes)) * &masks.down).dot(&down_factor.reads);
        let hidden_cotangent_error =
            down_factor.adjoint_radius(&masks.down, write_cotangent.view(), zero_write_cotangent.view());
        let down_mask_radius = down_factor.mask_cotangent_radius(
            hidden.view(),
            hidden_error.view(),
            write_cotangent.view(),
            zero_write_cotangent.view(),
        );
        let mut gate_cotangent = Array2::<f64>::zeros(gate.raw_dim());
        let mut gate_cotangent_error = Array2::<f64>::zeros(gate.raw_dim());
        let mut up_cotangent = Array2::<f64>::zeros(gate.raw_dim());
        let mut up_cotangent_error = Array2::<f64>::zeros(gate.raw_dim());
        let two_products = accumulation_growth(2);
        for (index, &gate_value) in gate.indexed_iter() {
            let jet = silu_derivatives(gate_value);
            let slope_error = silu_slope_evaluation_error(gate_value) + SILU_SECOND_BOUND * gate_radius[index];
            let value_error = silu_evaluation_error(gate_value) + SILU_LIPSCHITZ * gate_radius[index];
            let cotangent = hidden_cotangent[index];
            let cotangent_error = hidden_cotangent_error[index];
            gate_cotangent[index] = cotangent * jet[1] * up[index];
            up_cotangent[index] = cotangent * jet[0];
            gate_cotangent_error[index] = two_products * gate_cotangent[index].abs() / (1.0 - two_products)
                + triple_product_radius(cotangent, cotangent_error, jet[1], slope_error, up[index], up_radius[index]);
            up_cotangent_error[index] = UNIT_ROUNDOFF * up_cotangent[index].abs() / (1.0 - UNIT_ROUNDOFF)
                + triple_product_radius(cotangent, cotangent_error, jet[0], value_error, 1.0, 0.0);
        }
        // The two read cotangents are summed once: `u / (1 - u)` of each summed entry.
        let sum_growth = UNIT_ROUNDOFF / (1.0 - UNIT_ROUNDOFF);
        let input_radius = gate_factor.adjoint_radius(&masks.gate, gate_cotangent.view(), gate_cotangent_error.view())
            + &up_factor.adjoint_radius(&masks.up, up_cotangent.view(), up_cotangent_error.view())
            + &pullback.inputs.mapv(|value| 2.0 * sum_growth * value.abs());
        let gate_mask_radius = gate_factor.mask_cotangent_radius(
            inputs.view(),
            zero_inputs.view(),
            gate_cotangent.view(),
            gate_cotangent_error.view(),
        );
        let up_mask_radius =
            up_factor.mask_cotangent_radius(inputs.view(), zero_inputs.view(), up_cotangent.view(), up_cotangent_error.view());
        let weighted = |radius: &Array1<f64>, direction: &Array1<f64>| {
            Zip::from(radius).and(direction).fold(0.0, |sum, &e, &d| sum + e * d.abs())
        };
        let entry_radius = Zip::from(&input_radius).and(&input_direction).fold(0.0, |sum, &e, &d| sum + e * d.abs())
            + weighted(&gate_mask_radius, &gate_direction)
            + weighted(&up_mask_radius, &up_direction)
            + weighted(&down_mask_radius, &down_direction);

        // Truncation. Along the path each read `a(t) = ((x + t dx) R^T ⊙ (m + t dm)) U^T` is quadratic in `t`:
        // `|a| <= A0 + tau A1 + tau^2 A2`, `|a'| <= A1 + 2 tau A2`, `|a''| <= 2 A2` and `a''' = 0`. With `|s(z)| <= |z|`,
        // `|s'| < 11/8`, `|s''| < 7/8` and `|s'''| < 9/8`, `(s∘a)'' = s'' a'^2 + s' a''` and
        // `(s∘a)''' = s''' a'^3 + 3 s'' a' a''`. Leibniz gives the hidden rows' bounds, and the down stage
        // `y = U_d ((m_d + t dm_d) ⊙ R_d h)` gives `|y'''| <= ((|h'''| |R_d|^T) ⊙ (|m_d| + tau |dm_d|)
        // + 3 (|h''| |R_d|^T) ⊙ |dm_d|) |U_d|^T`.
        let step = 0.5_f64.powi(20);
        let path_bounds = |factor: &Factor, factor_mask: &Array1<f64>, direction: &Array1<f64>, value: &Array2<f64>, radius: &Array2<f64>| {
            let absolute_writes = factor.writes.mapv(f64::abs);
            let absolute_reads = factor.reads.mapv(f64::abs);
            let input_reads = inputs.mapv(f64::abs).dot(&absolute_reads.t());
            let direction_reads = input_direction.mapv(f64::abs).dot(&absolute_reads.t());
            let first = (&input_reads * &direction.mapv(f64::abs) + &(&direction_reads * &factor_mask.mapv(f64::abs)))
                .dot(&absolute_writes.t());
            let second = (&direction_reads * &direction.mapv(f64::abs)).dot(&absolute_writes.t());
            let maximum = value.mapv(f64::abs) + radius + &first.mapv(|entry| step * entry)
                + &second.mapv(|entry| step * step * entry);
            let slope = &first + &second.mapv(|entry| 2.0 * step * entry);
            (maximum, slope, second.mapv(|entry| 2.0 * entry))
        };
        let (gate_maximum, gate_slope, gate_curvature) =
            path_bounds(&gate_factor, &masks.gate, &gate_direction, &gate, &gate_radius);
        let (up_maximum, up_slope, up_curvature) = path_bounds(&up_factor, &masks.up, &up_direction, &up, &up_radius);
        let s1 = gate_slope.mapv(|entry| SILU_LIPSCHITZ * entry);
        let s2 = Zip::from(&gate_slope)
            .and(&gate_curvature)
            .map_collect(|&slope, &curvature| SILU_SECOND_BOUND * slope * slope + SILU_LIPSCHITZ * curvature);
        let s3 = Zip::from(&gate_slope).and(&gate_curvature).map_collect(|&slope, &curvature| {
            SILU_THIRD_BOUND * slope.powi(3) + 3.0 * SILU_SECOND_BOUND * slope * curvature
        });
        let hidden_second = &s2 * &up_maximum + &(2.0 * &s1 * &up_slope) + &(&gate_maximum * &up_curvature);
        let hidden_third = &s3 * &up_maximum + &(3.0 * &s2 * &up_slope) + &(3.0 * &s1 * &up_curvature);
        let absolute_down_reads = down_factor.reads.mapv(f64::abs);
        let absolute_down_writes = down_factor.writes.mapv(f64::abs);
        let down_mask_reach = masks.down.mapv(f64::abs) + &down_direction.mapv(|entry| step * entry.abs());
        let write_third = (hidden_third.dot(&absolute_down_reads.t()) * &down_mask_reach
            + &(hidden_second.dot(&absolute_down_reads.t()) * &down_direction.mapv(|entry| 3.0 * entry.abs())))
            .dot(&absolute_down_writes.t());
        let third_derivative =
            Zip::from(&write_cotangent).and(&write_third).fold(0.0, |sum, &y, &bound| sum + y.abs() * bound);
        let write_maximum = ((&gate_maximum * &up_maximum).dot(&absolute_down_reads.t()) * &down_mask_reach)
            .dot(&absolute_down_writes.t());
        let oracle_magnitude =
            Zip::from(&write_cotangent).and(&write_maximum).fold(0.0, |sum, &y, &bound| sum + y.abs() * bound);
        // The oracle's own error: every term of one evaluation passes at most this many double-double
        // operations, each within `2^-100` relative, and the difference quotient divides by the step.
        let oracle_operations =
            4 * (GATE_COMPONENTS * INPUT_DIM + UP_COMPONENTS * INPUT_DIM + HIDDEN_DIM + DOWN_COMPONENTS * HIDDEN_DIM) + 16;
        let predicted_magnitude = Zip::from(&pullback.inputs)
            .and(&input_direction)
            .fold(0.0, |sum, &v, &d| sum + (v * d).abs())
            + Zip::from(gate_mask_cotangent).and(&gate_direction).fold(0.0, |sum, &v, &d| sum + (v * d).abs())
            + Zip::from(up_mask_cotangent).and(&up_direction).fold(0.0, |sum, &v, &d| sum + (v * d).abs())
            + Zip::from(down_mask_cotangent).and(&down_direction).fold(0.0, |sum, &v, &d| sum + (v * d).abs());
        let entries = ROWS * INPUT_DIM + GATE_COMPONENTS + UP_COMPONENTS + DOWN_COMPONENTS;
        let tolerance = entry_radius
            + step * step / 6.0 * third_derivative
            + quad_relative() * oracle_operations as f64 * oracle_magnitude / step
            + quad_relative() * (2 * entries) as f64 * predicted_magnitude;
        let tolerance = inflate(arr0(tolerance), 8 * entries + 64).into_scalar();
        let central = (evaluate(step) - evaluate(-step)) / quad(2.0 * step);
        let difference = quad_to_f64(directional(down_mask_cotangent) - central).abs();
        assert!(
            difference / (1.0 - UNIT_ROUNDOFF) <= tolerance,
            "the component SwiGLU pullback's directional derivative differs from the double-double central \
             difference by {difference:e}, beyond the derived bound {tolerance:e}"
        );
        let without_down_mask = Array1::<f64>::zeros(DOWN_COMPONENTS);
        let wrong_difference = quad_to_f64(directional(&without_down_mask) - central).abs();
        assert!(
            wrong_difference > tolerance,
            "positive control: a pullback dropping the down mask cotangent must differ beyond the bound \
             {tolerance:e}, got {wrong_difference:e}"
        );

        // All on: no mask cotangents, and every read pulls back on its original tensor.
        let all_on = SwigluMask {
            gate: ComponentMask::AllOn,
            up: ComponentMask::AllOn,
            down: ComponentMask::AllOn,
        };
        let native_pullback = component
            .pullback(inputs.view(), all_on, write_cotangent.view())
            .expect("finite fixture");
        assert!(
            native_pullback.gate_mask.is_none() && native_pullback.up_mask.is_none() && native_pullback.down_mask.is_none(),
            "an all-on factor has no mask to differentiate"
        );
        let native = component.native();
        let native_stages = native.execute_stages(inputs.view()).expect("finite rows");
        let native_hidden = swiglu_hidden_pullback(
            native_stages.gate.view(),
            native_stages.up.view(),
            write_cotangent.dot(&native.down).view(),
        )
        .expect("finite fixture");
        let native_inputs = native_hidden.gate.dot(&native.gate) + &native_hidden.up.dot(&native.up);
        assert_eq!(
            native_pullback.inputs, native_inputs,
            "every factor all on must pull back on the original tensors bit for bit"
        );
        let ones = Array1::ones(GATE_COMPONENTS);
        let factored_gate = component
            .pullback(
                inputs.view(),
                SwigluMask {
                    gate: ComponentMask::Components(ones.view()),
                    ..all_on
                },
                write_cotangent.view(),
            )
            .expect("finite fixture");
        assert!(
            factored_gate.inputs.iter().zip(native_inputs.iter()).any(|(&factored, &original)| factored != original),
            "positive control: the factored gate pullback at an all-ones mask must differ from the native one in some bit"
        );
    }

    #[test]
    fn norm_pullbacks_match_a_double_double_central_difference() {
        let mut rng = StdRng::seed_from_u64(2958);
        let dim = 8;
        // Eighths below 3 in magnitude: every row's LayerNorm centring is exact.
        let residual = Array2::from_shape_simple_fn((ROWS, dim), || rng.random_range(-24..24) as f64 / 8.0);
        let gain = Array1::from_shape_simple_fn(dim, || rng.random_range(-2.0..2.0));
        let bias = Array1::from_shape_simple_fn(dim, || rng.random_range(-2.0..2.0));
        let output_cotangent = uniform_rows(&mut rng, ROWS, dim, -1.0, 1.0);
        let residual_direction = uniform_rows(&mut rng, ROWS, dim, -1.0, 1.0);
        let gain_direction = Array1::from_shape_simple_fn(dim, || rng.random_range(-1.0..1.0));
        let bias_direction = Array1::from_shape_simple_fn(dim, || rng.random_range(-1.0..1.0));
        let epsilon = 1.0e-5;
        for centring in [false, true] {
            let norm = if centring {
                MaskedNorm::Layer {
                    epsilon,
                    gain: gain.view(),
                    bias: bias.view(),
                }
            } else {
                MaskedNorm::Rms {
                    epsilon,
                    gain: gain.view(),
                }
            };
            let cotangents = norm
                .pullback(residual.view(), output_cotangent.view())
                .expect("finite fixture");
            let read = if centring {
                let means = residual.sum_axis(Axis(1)) / dim as f64;
                &residual - &means.insert_axis(Axis(1))
            } else {
                residual.clone()
            };
            let direction_means: Vec<Quad> = (0..ROWS)
                .map(|row| {
                    (0..dim).fold(quad(0.0), |sum, column| sum + quad(residual_direction[[row, column]]))
                        / quad(dim as f64)
                })
                .collect();
            let read_direction = Array2::from_shape_fn((ROWS, dim), |(row, column)| {
                let direction = quad(residual_direction[[row, column]]);
                if centring { direction - direction_means[row] } else { direction }
            });

            // `f(t) = sum_r <y_bar_r, N(h_r + t v_r; w + t omega, beta + t b)>` in double-double.
            let evaluate = |step: f64| {
                (0..ROWS).fold(quad(0.0), |total, row| {
                    let points: Vec<Quad> = (0..dim)
                        .map(|column| quad(read[[row, column]]) + quad(step) * read_direction[[row, column]])
                        .collect();
                    let square = points.iter().fold(quad(0.0), |sum, point| sum + *point * *point);
                    let inverse_root = quad(1.0) / (square / quad(dim as f64) + quad(epsilon)).sqrt();
                    (0..dim).fold(total, |sum, column| {
                        let cotangent = quad(output_cotangent[[row, column]]);
                        let weight = quad(gain[column]) + quad(step) * quad(gain_direction[column]);
                        let offset = if centring {
                            cotangent * (quad(bias[column]) + quad(step) * quad(bias_direction[column]))
                        } else {
                            quad(0.0)
                        };
                        sum + cotangent * weight * points[column] * inverse_root + offset
                    })
                })
            };
            let directional = |residual_cotangent: &Array2<f64>| {
                let residual_part = Zip::from(residual_cotangent)
                    .and(&residual_direction)
                    .fold(quad(0.0), |sum, &value, &direction| sum + quad(value) * quad(direction));
                let gain_part = Zip::from(&cotangents.gain)
                    .and(&gain_direction)
                    .fold(quad(0.0), |sum, &value, &direction| sum + quad(value) * quad(direction));
                let bias_part = match &cotangents.bias {
                    Some(bias_cotangent) => Zip::from(bias_cotangent)
                        .and(&bias_direction)
                        .fold(quad(0.0), |sum, &value, &direction| sum + quad(value) * quad(direction)),
                    None => quad(0.0),
                };
                residual_part + gain_part + bias_part
            };

            // Cauchy, per row: along `x_r + t v_r`, `q_r(t) + eps = a_r + b_r t + c_r t^2`. On
            // `|t| <= R_r` with `|b_r| R_r + c_r R_r^2 = a_r / 2`, `|q_r + eps| >= a_r / 2`. With
            // `R = min_r R_r` the normalized part of `f` is bounded on `|t| <= R` by
            // `M = sum_r P_r (2 / a_r)^(1/2)`, the bias part is linear, and for `|s| <= R / 2`,
            // `|f'''(s)| <= 3! M / (R / 2)^3`: the central difference is within `8 step^2 M / R^3`.
            let direction_f64 = read_direction.mapv(quad_to_f64);
            let quadratics: Vec<(f64, f64, f64)> = (0..ROWS)
                .map(|row| {
                    let x = read.row(row);
                    let v = direction_f64.row(row);
                    (x.dot(&x) / dim as f64 + epsilon, 2.0 * x.dot(&v) / dim as f64, v.dot(&v) / dim as f64)
                })
                .collect();
            let radius = quadratics
                .iter()
                .map(|&(a, b, c)| a / (b.abs() + (b * b + 2.0 * a * c).sqrt()))
                .fold(f64::INFINITY, f64::min);
            let bound_m = (0..ROWS)
                .map(|row| {
                    (0..dim)
                        .map(|column| {
                            output_cotangent[[row, column]].abs()
                                * (gain[column].abs() + radius * gain_direction[column].abs())
                                * (read[[row, column]].abs() + radius * direction_f64[[row, column]].abs())
                        })
                        .sum::<f64>()
                        * (2.0 / quadratics[row].0).sqrt()
                })
                .sum::<f64>();
            let step = radius * 0.5_f64.powi(20);
            let central = (evaluate(step) - evaluate(-step)) / quad(2.0 * step);

            // f64 entry radii. `nu` passes `d + 4` rounded operations; every term of
            // `x_bar = nu z - (nu^3 / d)(z^T x) x` passes at most `4 d + 20`. The LayerNorm
            // centring of the cotangent adds its mean's radius and `gamma_(d + 1)`. The gain
            // cotangent's terms pass `d + 6` operations and the row sum `rows`; the bias
            // cotangent's row sum `rows - 1`.
            let term_gamma = accumulation_growth(4 * dim + 20);
            let centring_gamma = accumulation_growth(dim + 1);
            let gain_gamma = accumulation_growth(dim + 6 + ROWS);
            let summation = accumulation_growth(ROWS - 1);
            let mut entry_radius = 0.0;
            let mut predicted_magnitude = 0.0;
            let mut gain_magnitude = Array1::<f64>::zeros(dim);
            let mut flipped_radial = cotangents.residual.clone();
            let mut uncentred = cotangents.residual.clone();
            for row in 0..ROWS {
                let x = read.row(row);
                let inverse_root = super::inverse_root_mean_square(row, x, epsilon).expect("finite row");
                let scaled = &gain * &output_cotangent.row(row);
                let projection = scaled.dot(&x);
                let absolute_projection = Zip::from(&scaled)
                    .and(x)
                    .fold(0.0, |sum, &direction, &value| sum + (direction * value).abs());
                let radial = inverse_root.powi(3) * projection / dim as f64;
                let radial_magnitude = inverse_root.powi(3) * (projection.abs() + absolute_projection) / dim as f64;
                let magnitudes = Zip::from(&scaled)
                    .and(x)
                    .map_collect(|&direction, &value| inverse_root * direction.abs() + radial_magnitude * value.abs());
                let read_radius = magnitudes.mapv(|magnitude| term_gamma * magnitude / (1.0 - term_gamma));
                let mean_radius = read_radius.sum() / dim as f64;
                let mean_magnitude = magnitudes.sum() / dim as f64;
                let scaled_mean = scaled.sum() / dim as f64;
                for column in 0..dim {
                    let residual_entry_radius = if centring {
                        read_radius[column]
                            + mean_radius
                            + centring_gamma * (magnitudes[column] + mean_magnitude) / (1.0 - centring_gamma)
                    } else {
                        read_radius[column]
                    };
                    entry_radius += residual_entry_radius * residual_direction[[row, column]].abs();
                    gain_magnitude[column] += (output_cotangent[[row, column]] * x[column] * inverse_root).abs();
                    predicted_magnitude +=
                        (cotangents.residual[[row, column]] * residual_direction[[row, column]]).abs();
                    flipped_radial[[row, column]] += 2.0 * radial * x[column];
                    uncentred[[row, column]] += inverse_root * scaled_mean;
                }
            }
            let mut bias_oracle_magnitude = 0.0;
            for column in 0..dim {
                entry_radius +=
                    gain_gamma * gain_magnitude[column] / (1.0 - gain_gamma) * gain_direction[column].abs();
                predicted_magnitude += (cotangents.gain[column] * gain_direction[column]).abs();
                if centring {
                    let bias_magnitude = (0..ROWS).map(|row| output_cotangent[[row, column]].abs()).sum::<f64>();
                    entry_radius += summation * bias_magnitude / (1.0 - summation) * bias_direction[column].abs();
                    predicted_magnitude += bias_magnitude * bias_direction[column].abs();
                    bias_oracle_magnitude +=
                        bias_magnitude * (bias[column].abs() + step * bias_direction[column].abs());
                }
            }
            let entries = ROWS * dim;
            let tolerance = entry_radius
                + 8.0 * step * step * bound_m / radius.powi(3)
                + quad_relative() * (4 * entries + 8) as f64 * (bound_m + bias_oracle_magnitude) / step
                + quad_relative() * (3 * entries) as f64 * predicted_magnitude;
            let tolerance = inflate(arr0(tolerance), 6 * entries + 30).into_scalar();
            let difference = quad_to_f64(directional(&cotangents.residual) - central).abs();
            assert!(
                difference / (1.0 - UNIT_ROUNDOFF) <= tolerance,
                "centring {centring}: the norm pullback's directional derivative differs from the \
                 double-double central difference by {difference:e}, beyond the derived bound {tolerance:e}"
            );
            let flipped_difference = quad_to_f64(directional(&flipped_radial) - central).abs();
            assert!(
                flipped_difference > tolerance,
                "positive control (centring {centring}): a sign-flipped radial term must differ beyond \
                 the bound {tolerance:e}, got {flipped_difference:e}"
            );
            if centring {
                let uncentred_difference = quad_to_f64(directional(&uncentred) - central).abs();
                assert!(
                    uncentred_difference > tolerance,
                    "positive control: a LayerNorm cotangent left uncentred must differ beyond the bound \
                     {tolerance:e}, got {uncentred_difference:e}"
                );
            }
        }
    }

    #[test]
    fn rms_norm_sees_a_power_of_two_scale_only_through_epsilon() {
        let mut rng = StdRng::seed_from_u64(2953);
        let residual = uniform_rows(&mut rng, ROWS, 12, -3.0, 3.0);
        let gain = Array1::from_shape_simple_fn(12, || rng.random_range(-2.0..2.0));
        let epsilon = 1.0e-6;
        let norm = MaskedNorm::Rms {
            epsilon,
            gain: gain.view(),
        };
        // Scaling by 2^k commutes with correctly rounded +, *, / and sqrt absent
        // overflow, so both identities hold bit for bit.
        for scale in [8.0, -8.0] {
            let scaled = residual.mapv(|value| scale * value);
            let normalized_scaled = norm.apply(scaled.view()).expect("finite residual");
            let rescaled_epsilon = MaskedNorm::Rms {
                epsilon: epsilon / (scale * scale),
                gain: gain.view(),
            }
            .apply(residual.view())
            .expect("finite residual")
            .mapv(|value| scale.signum() * value);
            assert_eq!(
                normalized_scaled, rescaled_epsilon,
                "N_eps({scale} h) must equal sign({scale}) N_(eps/{scale}^2)(h) bit for bit"
            );
            assert_eq!(
                rms_normalizers(scaled.view(), epsilon).expect("finite rows"),
                rms_normalizers(residual.view(), epsilon / (scale * scale))
                    .expect("finite rows")
                    .mapv(|value| value / scale.abs()),
                "nu_eps({scale} h) must equal nu_(eps/{scale}^2)(h) / |{scale}| bit for bit"
            );
        }
        let unscaled = norm.apply(residual.view()).expect("finite residual");
        let scaled_same_epsilon = norm
            .apply(residual.mapv(|value| 8.0 * value).view())
            .expect("finite residual");
        assert_ne!(
            unscaled, scaled_same_epsilon,
            "positive control: at a fixed epsilon the scale must reach the output"
        );
        assert_ne!(
            rms_normalizers(residual.mapv(|value| 8.0 * value).view(), epsilon).expect("finite rows"),
            rms_normalizers(residual.view(), epsilon)
                .expect("finite rows")
                .mapv(|value| value / 8.0),
            "positive control: at a fixed epsilon the scale must reach the normalizer"
        );
    }

    #[test]
    fn rms_norm_of_a_summed_residual_is_not_the_sum_of_normalized_parts() {
        let mut rng = StdRng::seed_from_u64(2954);
        let dim = 10;
        let parts = 4;
        let gain = Array1::from_shape_simple_fn(dim, || rng.random_range(-2.0..2.0));
        let norm = MaskedNorm::Rms {
            epsilon: 1.0e-6,
            gain: gain.view(),
        };
        // Sixteenths below 2 in magnitude add exactly, so the summed residual is the
        // real sum of the parts.
        let part_values: Vec<Array2<f64>> = std::iter::repeat_with(|| {
            Array2::from_shape_simple_fn((ROWS, dim), || rng.random_range(-32..32) as f64 / 16.0)
        })
        .take(parts)
        .collect();
        let summed = part_values
            .iter()
            .fold(Array2::<f64>::zeros((ROWS, dim)), |accumulated, part| accumulated + part);
        // fl(N(h)) passes the squares, d - 1 additions, the mean, + eps, sqrt, the
        // reciprocal, the product with h and the gain: gamma_(d + 6) relative.
        let norm_gamma = accumulation_growth(dim + 6);
        let whole = norm.apply(summed.view()).expect("finite residual");
        let mut separate = Array2::<f64>::zeros((ROWS, dim));
        let mut radius = whole.mapv(|value| norm_gamma * value.abs() / (1.0 - norm_gamma));
        let mut magnitude = Array2::<f64>::zeros((ROWS, dim));
        for part in &part_values {
            let normalized = norm.apply(part.view()).expect("finite residual");
            radius += &normalized.mapv(|value| norm_gamma * value.abs() / (1.0 - norm_gamma));
            magnitude += &normalized.mapv(f64::abs);
            separate += &normalized;
        }
        let summation = accumulation_growth(parts - 1);
        radius += &magnitude.mapv(|value| summation * value / (1.0 - summation));
        let tolerance = inflate(radius, dim + parts + 9);
        let refuted = separate
            .iter()
            .zip(whole.iter())
            .zip(tolerance.iter())
            .filter(|&((&separate_value, &whole_value), &bound)| (separate_value - whole_value).abs() > bound)
            .count();
        assert!(
            refuted > 0,
            "the separately normalized sum {separate} must differ from the normalized sum {whole} \
             beyond the derived bound {tolerance}"
        );
    }

    #[test]
    fn layer_norm_annihilates_a_write_along_the_constant_direction() {
        let mut rng = StdRng::seed_from_u64(2955);
        // Eighths: every sum, the mean and the centring are exact, so the centred
        // rows agree bit for bit and the identity must hold bit for bit.
        let residual = array![
            [0.5, -1.25, 2.0, 0.75, -0.375, 1.5, -2.25, 0.125],
            [1.0, 0.25, -0.5, -1.75, 2.5, 0.0, -0.625, 0.875]
        ];
        let shifted = residual.mapv(|value| value + 3.5);
        let gain = Array1::from_shape_simple_fn(8, || rng.random_range(-2.0..2.0));
        let bias = Array1::from_shape_simple_fn(8, || rng.random_range(-2.0..2.0));
        let layer = MaskedNorm::Layer {
            epsilon: 1.0e-5,
            gain: gain.view(),
            bias: bias.view(),
        };
        assert_eq!(
            layer.apply(shifted.view()),
            layer.apply(residual.view()),
            "a LayerNorm must not see a write along the constant direction"
        );
        let rms = MaskedNorm::Rms {
            epsilon: 1.0e-5,
            gain: gain.view(),
        };
        assert_ne!(
            rms.apply(shifted.view()),
            rms.apply(residual.view()),
            "positive control: an RMSNorm has no constant null direction"
        );
    }

    #[test]
    fn a_bias_is_the_homogeneous_read_column_of_a_component_factor() {
        let inputs = array![[0.5, -1.5, 2.0], [-1.0, 0.25, 0.75]];
        let read = homogeneous_read(inputs.view());
        assert_eq!(read, array![[0.5, -1.5, 2.0, 1.0], [-1.0, 0.25, 0.75, 1.0]]);
        // Dyadic entries with small denominators: every product and sum is exact.
        let writes = array![[0.5, -1.0, 0.25], [1.0, 0.5, -0.5]];
        let reads = array![[1.0, -0.5, 0.0, 0.25], [0.5, 0.25, -1.0, 0.0], [0.0, 1.0, 0.5, -0.75]];
        let mask = array![1.0, -0.5, 0.25];
        let rewritten = (read.dot(&reads.t()) * &mask).dot(&writes.t());
        let edited_weight = writes.dot(&Array2::from_diag(&mask)).dot(&reads.slice(s![.., ..3]));
        let edited_bias = writes.dot(&(&mask * &reads.column(3)));
        let executed = inputs.dot(&edited_weight.t()) + &edited_bias;
        assert_eq!(rewritten, executed, "[W | b](m) (x, 1) must equal W(m) x + b(m) on every row");
        let mut without_bias = reads.clone();
        without_bias.column_mut(3).fill(0.0);
        assert_ne!(
            rewritten,
            (read.dot(&without_bias.t()) * &mask).dot(&writes.t()),
            "positive control: the bias read column must reach the output"
        );
    }

    #[test]
    fn a_parallel_layer_is_affine_in_its_write_edges_and_a_sequential_layer_is_not() {
        let mut rng = StdRng::seed_from_u64(2956);
        let dim = 10;
        let epsilon = 1.0e-6;
        let residual = uniform_rows(&mut rng, ROWS, dim, -2.0, 2.0);
        let attention_gain = Array1::from_shape_simple_fn(dim, || rng.random_range(-2.0..2.0));
        let mlp_gain = Array1::from_shape_simple_fn(dim, || rng.random_range(-2.0..2.0));
        let attention_write = uniform_rows(&mut rng, dim, dim, -1.0, 1.0);
        let mlp_write = uniform_rows(&mut rng, dim, dim, -1.0, 1.0);
        let attention_norm = MaskedNorm::Rms {
            epsilon,
            gain: attention_gain.view(),
        };
        let mlp_norm = MaskedNorm::Rms {
            epsilon,
            gain: mlp_gain.view(),
        };
        let run = |layout: ResidualLayout, attention_edge: f64, mlp_edge: f64| {
            decoder_layer(
                layout,
                residual.view(),
                |stream| {
                    attention_norm
                        .apply(stream)
                        .map(|normalized| normalized.dot(&attention_write.t()).mapv(|value| attention_edge * value))
                },
                |stream| {
                    mlp_norm
                        .apply(stream)
                        .map(|normalized| normalized.dot(&mlp_write.t()).mapv(|value| mlp_edge * value))
                },
            )
            .expect("the fixture layer is finite")
        };
        let mixed_difference = |layout: ResidualLayout| {
            let both = run(layout, 1.0, 1.0);
            let attention_only = run(layout, 1.0, 0.0);
            let mlp_only = run(layout, 0.0, 1.0);
            let neither = run(layout, 0.0, 0.0);
            let difference = &(&(&both - &attention_only) - &mlp_only) + &neither;
            let magnitude = both.mapv(f64::abs)
                + &attention_only.mapv(f64::abs)
                + &mlp_only.mapv(f64::abs)
                + &neither.mapv(f64::abs);
            (difference, magnitude)
        };
        let attention_at_input = attention_norm
            .apply(residual.view())
            .expect("finite")
            .dot(&attention_write.t());
        let mlp_at_input = mlp_norm.apply(residual.view()).expect("finite").dot(&mlp_write.t());
        let after_attention = &residual + &attention_at_input;
        let mlp_after_attention = mlp_norm
            .apply(after_attention.view())
            .expect("finite")
            .dot(&mlp_write.t());
        // Were the layer affine in its binary edges, the real mixed difference would
        // be 0. The computed one then carries only the rounding of `fl(fl(M + A) + h)`
        // (two additions), `fl(A + h)`, `fl(M + h)` and its own three additions.
        let tolerance = |mlp_magnitude: &Array2<f64>, magnitude: &Array2<f64>| {
            let entries = Zip::from(mlp_magnitude)
                .and(&attention_at_input)
                .and(&residual)
                .and(magnitude)
                .map_collect(|&mlp_value, &attention_value, &input_value, &total| {
                    let attention_value = attention_value.abs();
                    let input_value = input_value.abs();
                    accumulation_growth(2) * (mlp_value + attention_value + input_value)
                        + UNIT_ROUNDOFF * (attention_value + input_value)
                        + UNIT_ROUNDOFF * (mlp_value + input_value)
                        + accumulation_growth(3) * total
                });
            inflate(entries, 6)
        };

        let (parallel_difference, parallel_magnitude) = mixed_difference(ResidualLayout::Parallel);
        let parallel_tolerance = tolerance(&mlp_at_input.mapv(f64::abs), &parallel_magnitude);
        for (((row, column), &difference), &bound) in
            parallel_difference.indexed_iter().zip(parallel_tolerance.iter())
        {
            assert!(
                difference.abs() <= bound,
                "row {row}, output {column}: a parallel layer's mixed edge difference {difference:e} \
                 exceeds the derived rounding bound {bound:e}"
            );
        }

        let (sequential_difference, sequential_magnitude) = mixed_difference(ResidualLayout::Sequential);
        let sequential_mlp_magnitude = Zip::from(&mlp_at_input)
            .and(&mlp_after_attention)
            .map_collect(|&at_input, &after| at_input.abs().max(after.abs()));
        let sequential_tolerance = tolerance(&sequential_mlp_magnitude, &sequential_magnitude);
        let refuted = sequential_difference
            .iter()
            .zip(sequential_tolerance.iter())
            .filter(|&(&difference, &bound)| difference.abs() > bound)
            .count();
        assert!(
            refuted > 0,
            "positive control: a sequential layer's mixed edge difference {sequential_difference} \
             must exceed the rounding bound {sequential_tolerance} somewhere"
        );
    }

    #[test]
    fn native_primitives_refuse_mismatched_nonfinite_and_degenerate_inputs() {
        assert_eq!(
            swiglu_hidden(array![[1.0, 2.0]].view(), array![[1.0]].view()),
            Err(GatedRewriteError::ShapeMismatch {
                what: "SwiGLU up pre-activations",
                expected: (1, 2),
                found: (1, 1),
            })
        );
        assert!(matches!(
            swiglu_hidden(array![[f64::NAN]].view(), array![[1.0]].view()),
            Err(GatedRewriteError::NonFinite { row: 0, column: 0, .. })
        ));
        assert_eq!(
            swiglu_hidden(array![[0.0]].view(), array![[3.0]].view()),
            Ok(array![[0.0]]),
            "positive control: a finite pair is accepted"
        );
        assert!(matches!(
            swiglu_hidden_pullback(array![[1.0]].view(), array![[1.0]].view(), array![[1.0, 2.0]].view()),
            Err(GatedRewriteError::ShapeMismatch {
                what: "SwiGLU hidden cotangent",
                ..
            })
        ));

        let gain = array![1.0, 1.0];
        let zero = array![[0.0, 0.0]];
        assert_eq!(
            MaskedNorm::Rms {
                epsilon: 0.0,
                gain: gain.view(),
            }
            .apply(zero.view()),
            Err(GatedRewriteError::DegenerateNormalizer {
                row: 0,
                mean_square: 0.0,
                epsilon: 0.0,
            })
        );
        assert_eq!(
            MaskedNorm::Rms {
                epsilon: 1.0e-6,
                gain: gain.view(),
            }
            .apply(zero.view()),
            Ok(array![[0.0, 0.0]]),
            "positive control: the declared epsilon normalizes a zero row"
        );
        assert!(matches!(
            MaskedNorm::Rms {
                epsilon: -1.0,
                gain: gain.view(),
            }
            .apply(zero.view()),
            Err(GatedRewriteError::InvalidEpsilon { .. })
        ));
        assert!(matches!(
            MaskedNorm::Rms {
                epsilon: 1.0e-6,
                gain: gain.view(),
            }
            .apply(array![[1.0, 2.0], [1.0e200, 1.0e200]].view()),
            Err(GatedRewriteError::DegenerateNormalizer { row: 1, .. })
        ));
        assert!(matches!(
            MaskedNorm::Rms {
                epsilon: 1.0e-6,
                gain: array![1.0].view(),
            }
            .apply(zero.view()),
            Err(GatedRewriteError::ShapeMismatch { what: "RMSNorm gain", .. })
        ));
        assert!(matches!(
            MaskedNorm::Rms {
                epsilon: 1.0e-6,
                gain: gain.view(),
            }
            .pullback(zero.view(), array![[1.0, 2.0, 3.0]].view()),
            Err(GatedRewriteError::ShapeMismatch {
                what: "normalization output cotangent",
                ..
            })
        ));
        let empty_rows = Array2::<f64>::zeros((1, 0));
        let empty = Array1::<f64>::zeros(0);
        assert_eq!(
            MaskedNorm::Layer {
                epsilon: 1.0e-5,
                gain: empty.view(),
                bias: empty.view(),
            }
            .apply(empty_rows.view()),
            Err(GatedRewriteError::EmptyResidual)
        );
        assert_eq!(
            rms_normalizers(empty_rows.view(), 1.0e-6),
            Err(GatedRewriteError::EmptyResidual)
        );
        // Nine plus sixteen over two is exactly 12.5, so the normalizer is its
        // correctly rounded inverse square root.
        assert_eq!(
            rms_normalizers(array![[3.0, 4.0]].view(), 0.0),
            Ok(array![12.5_f64.sqrt().recip()]),
            "positive control: a finite row is normalized"
        );

        let residual = array![[1.0, 2.0]];
        assert_eq!(
            decoder_layer::<GatedRewriteError, _, _>(
                ResidualLayout::Parallel,
                residual.view(),
                |stream| Ok(Array2::zeros((stream.nrows(), stream.ncols() + 1))),
                |stream| Ok(stream.to_owned()),
            ),
            Err(GatedRewriteError::ShapeMismatch {
                what: "attention write",
                expected: (1, 2),
                found: (1, 3),
            })
        );
        assert_eq!(
            decoder_layer::<GatedRewriteError, _, _>(
                ResidualLayout::Parallel,
                residual.view(),
                |stream| Ok(Array2::zeros(stream.raw_dim())),
                |stream| Ok(stream.to_owned()),
            ),
            Ok(array![[2.0, 4.0]]),
            "positive control: matching writes are summed into the stream"
        );
    }
}
