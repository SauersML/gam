//! Executed-stage receipts against the native lift (#2951 A12).
//!
//! A receipt compares one stage of a network executed by an external executor
//! (torch, through `gamfit.torch.parameter_interventions`) with the same stage
//! executed natively, entry by entry. Two IEEE evaluations `ŝ_ext` and `ŝ_nat` of
//! one exact stage value `s` satisfy `|ŝ_ext − ŝ_nat| ≤ |ŝ_ext − s| + |ŝ_nat − s|`,
//! so they agree when the discrepancy is at most the sum of two bands, each a
//! certified bound on its own evaluation's error.
//!
//! # The band of one evaluation
//!
//! A stage entry is a sum of terms, each a product of floats. Every rounded
//! operation on a term's path scales it by at most `1 ± u`, and a rounded product
//! applied to an accumulated sum scales every term inside it. With `k` the most
//! rounded operations on any term's path, the computed entry is within
//! `γ_k Σ|terms|` of the exact one, `γ_k = ku/(1 − ku)` (Higham, *Accuracy and
//! Stability of Numerical Algorithms*, 2nd ed., Lemma 3.1 and §3.1; the owner is
//! [`accumulation_growth`]). The order of additions inside one accumulation does
//! not change `k`: a binary tree over `n` leaves has depth at most `n − 1`, and a
//! fused multiply-add rounds once where the model counts two.
//!
//! `Σ|terms|` is accumulated under emulated upward rounding: each operation rounds
//! to nearest and then steps one float up, which lies at or above the exact result
//! of nonnegative operands. So the band never rests on a rounded-down magnitude.
//!
//! # Underflow
//!
//! The relative model holds only while no operation underflows. In binary64 a
//! rounded operation is `fl(a ∘ b) = (a ∘ b)(1 + δ) + η` with `|δ| ≤ u`, where
//! `|η| ≤ 2^-1075` for a product or quotient that rounds into the subnormal range and
//! `η = 0` for a sum, which is exact there (a fused multiply-add rounds once and adds
//! one `η`). A product's `η` passes the later sums with a factor of at most
//! `1 + γ_k ≤ 2`, and a later product scales it by the magnitude it multiplies. So an
//! entry's error is at most `γ_k Σ|terms| + 𝒜 · 2^-1074`, with the underflow
//! allowance `𝒜` counting each product once, scaled by the magnitudes that multiply
//! it afterwards: `𝒜 = d` for an inner product of length `d`. The band adds that
//! term ([`evaluation_band`]), so an entry whose products round into the subnormal
//! range is still enclosed ([`SUBNORMAL_SPACING`], #4005).
//!
//! # What a band does not cover
//!
//! The model counts rounded binary64 operations in round-to-nearest. So a band is valid
//! only for an executor that evaluates the stage in IEEE binary64 with one rounding per
//! basic operation. A float32 or bfloat16 execution, or TF32 matrix multiplication,
//! rounds at a coarser precision that no binary64 band covers. [`compare_stage`] takes
//! the external [`ExternalExecution`] record and refuses such an execution before any
//! verdict exists, so neither `agrees` nor `refutes` can be read without the check
//! (mpd-verify and mpd-spec on #2951). The native side is Rust `f64`, binary64 by type.
//!
//! A module can round at a coarser precision than its rows' dtype says. transformers'
//! `Qwen3RMSNorm` and `LlamaRMSNorm` cast binary64 rows to binary32, normalize there, and
//! cast back before the gain, so their stage output is binary64 while the norm ran in
//! binary32. The dtype record cannot see that; the caller declares the module's program
//! ([`ExternalRmsNormProgram`]), and [`rms_norm_stage`] takes the band of that program.
//!
//! Two implementations of a special function (torch's `erf` against gam-math's
//! `erfc`) share no derivation here. A receipt evaluates the native activation at
//! the external executor's exported pre-activation, which isolates that difference
//! as a measured discrepancy. The measured discrepancy is reported beside the
//! band and never folded into it (#2951 correction C2).

use super::apply::{ApplyError, FactorView, apply_anchored_linear, native_linear};
use super::block::{NormBandUnbounded, SUBNORMAL_SPACING, down, rms_norm_band, up};
use super::gated_rewrite::{GatedRewriteError, MaskedNorm, swiglu_hidden};
use super::occurrence::{OccurrenceError, PositionScope};
use super::rewrite::{NativeMlp, ShapeMismatch};
use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
use gam_math::gaussian_activation::{GaussianActivation, GaussianActivationError};
use gam_math::gaussian_gated::silu_derivatives;
use ndarray::{Array2, ArrayView1, ArrayView2, Zip};
use std::fmt;

/// What an external executor ran a stage in, as it reports it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalExecution<'a> {
    /// The executed dtype's name, e.g. torch's `"float64"`.
    pub dtype: &'a str,
    /// The executing device. It is recorded, not gated: float64 BLAS on a CPU or a GPU,
    /// with blocked sums or fused multiply-add, stays inside the rounding model.
    pub device: &'a str,
    /// Whether TF32 matrix multiplication was enabled.
    pub tf32_matmul: bool,
}

impl ExternalExecution<'_> {
    /// Refuses unless the stage ran in binary64 with TF32 off, the precondition every band
    /// here assumes.
    fn require_binary64_bands(&self) -> Result<(), ReceiptRefusal> {
        let float64 = self.dtype == "float64";
        if float64 && !self.tf32_matmul {
            Ok(())
        } else {
            Err(ReceiptRefusal::ExternalPrecision {
                float64,
                tf32_matmul: self.tf32_matmul,
            })
        }
    }
}

/// A refused receipt computation.
#[derive(Clone, Debug, PartialEq)]
pub enum ReceiptRefusal {
    Shape(ShapeMismatch),
    /// An executed or band array has a non-finite entry.
    NonFinite {
        array: &'static str,
        row: usize,
        column: usize,
    },
    /// The external stage was not executed in binary64 with TF32 off, so no band here
    /// covers it.
    ExternalPrecision { float64: bool, tf32_matmul: bool },
    /// The native factored-edit kernel refused.
    Apply(ApplyError),
    /// The native activation refused.
    Activation(GaussianActivationError),
    /// The native gated primitive refused.
    GatedRewrite(GatedRewriteError),
    /// An edit's declared rows do not fit the receipt's rows.
    Occurrence(OccurrenceError),
    /// A binary32-internal norm's band rests on torch's CPU kernels (a correctly rounded
    /// square root and division in `rsqrt`), and the stage ran elsewhere.
    ExternalDevice { program: ExternalRmsNormProgram },
    /// A row that the binary32 program cannot evaluate in range: its sum of squares reaches
    /// binary32's largest finite value, so torch's norm would overflow or read zero.
    Binary32Range { row: usize },
    /// A norm epsilon the binary32 program's band does not cover: not a positive binary32
    /// normal number, so `fl32(ε)` and `mean + ε` are not relative roundings.
    Binary32Epsilon { epsilon: f64 },
    /// A binary64 RMSNorm row with no finite band ([`rms_norm_band`]).
    NormBand(NormBandUnbounded),
}

impl From<ShapeMismatch> for ReceiptRefusal {
    fn from(mismatch: ShapeMismatch) -> Self {
        Self::Shape(mismatch)
    }
}

impl From<ApplyError> for ReceiptRefusal {
    fn from(error: ApplyError) -> Self {
        Self::Apply(error)
    }
}

impl From<GaussianActivationError> for ReceiptRefusal {
    fn from(error: GaussianActivationError) -> Self {
        Self::Activation(error)
    }
}

impl From<GatedRewriteError> for ReceiptRefusal {
    fn from(error: GatedRewriteError) -> Self {
        Self::GatedRewrite(error)
    }
}

impl From<NormBandUnbounded> for ReceiptRefusal {
    fn from(unbounded: NormBandUnbounded) -> Self {
        Self::NormBand(unbounded)
    }
}

impl From<OccurrenceError> for ReceiptRefusal {
    fn from(error: OccurrenceError) -> Self {
        Self::Occurrence(error)
    }
}

impl fmt::Display for ReceiptRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(mismatch) => write!(formatter, "receipt shapes do not compose: {mismatch}"),
            Self::NonFinite { array, row, column } => write!(
                formatter,
                "the receipt's {array} has a non-finite entry at ({row}, {column})"
            ),
            Self::ExternalPrecision {
                float64,
                tf32_matmul,
            } => write!(
                formatter,
                "the external stage was not executed in binary64 with TF32 off (float64: {float64}, \
                 tf32_matmul: {tf32_matmul}), so no binary64 band covers it"
            ),
            Self::Apply(error) => {
                write!(formatter, "the native factored-edit kernel refused: {error:?}")
            }
            Self::Activation(error) => write!(formatter, "the native activation refused: {error}"),
            Self::GatedRewrite(error) => {
                write!(formatter, "the native gated primitive refused: {error}")
            }
            Self::Occurrence(error) => {
                write!(formatter, "an edit's declared rows do not fit the receipt: {error}")
            }
            Self::ExternalDevice { program } => write!(
                formatter,
                "the {program:?} RMSNorm band covers torch's CPU kernels only, and the stage ran elsewhere"
            ),
            Self::Binary32Range { row } => write!(
                formatter,
                "row {row}'s sum of squares reaches binary32's largest finite value, outside the binary32 norm band"
            ),
            Self::Binary32Epsilon { epsilon } => write!(
                formatter,
                "the norm epsilon {epsilon:e} is not a positive binary32 normal number, outside the binary32 norm band"
            ),
            Self::NormBand(unbounded) => write!(formatter, "the binary64 norm band refused: {unbounded}"),
        }
    }
}

impl std::error::Error for ReceiptRefusal {}

fn check(what: &'static str, expected: usize, found: usize) -> Result<(), ShapeMismatch> {
    if expected == found {
        Ok(())
    } else {
        Err(ShapeMismatch {
            what,
            expected,
            found,
        })
    }
}

/// The certified band of one evaluation: `γ_k` rounded up times an upper bound on
/// the exact `Σ|terms|`, where `path_roundings = k` is the most rounded operations
/// on any term's path, plus `𝒜 · 2^-1074` for an upper bound `underflow_allowance`
/// on the entry's underflow allowance `𝒜` (see the module's *Underflow*). Every
/// operation rounds to nearest and steps one float up, so a subnormal band does not
/// round to zero. An entry with no products has `𝒜 = 0`, and the band is then the
/// relative term alone: adding an exact zero rounds nothing.
pub fn evaluation_band(path_roundings: usize, absolute_sum_upper: f64, underflow_allowance: f64) -> f64 {
    let relative = up(up(accumulation_growth(path_roundings)) * absolute_sum_upper);
    if underflow_allowance == 0.0 {
        relative
    } else {
        up(relative + up(underflow_allowance * SUBNORMAL_SPACING))
    }
}

/// Per-entry band of one evaluation of `x Wᵀ + b` for each row `x` of `inputs`,
/// rows by outputs.
///
/// A term `w_jk x_k` rounds once as a product and then passes at most `d − 1`
/// additions among the products and one for the bias, so `k = d + 1` (`d` without
/// a bias). Its `d` products give the underflow allowance `𝒜 = d`; the bias
/// addition adds none.
pub fn affine_stage_band(
    weight: ArrayView2<'_, f64>,
    bias: Option<ArrayView1<'_, f64>>,
    inputs: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, ShapeMismatch> {
    let (outputs, width) = weight.dim();
    check("affine input width", width, inputs.ncols())?;
    if let Some(bias) = bias {
        check("affine bias length", outputs, bias.len())?;
    }
    let path_roundings = width + usize::from(bias.is_some());
    let mut band = Array2::<f64>::zeros((inputs.nrows(), outputs));
    for (row, input) in inputs.outer_iter().enumerate() {
        for output in 0..outputs {
            let mut absolute_sum = bias.map_or(0.0, |bias| bias[output].abs());
            for (w, x) in weight.row(output).iter().zip(input.iter()) {
                absolute_sum = up(absolute_sum + up((w * x).abs()));
            }
            band[[row, output]] = evaluation_band(path_roundings, absolute_sum, width as f64);
        }
    }
    Ok(band)
}

/// Per-entry band of one evaluation of `x (W + L diag(s) Rᵀ)ᵀ + b` for each row `x`
/// of `inputs`: a weight `W` (`p × d`) with a factored edit of `C` components, left
/// factor `L` (`p × C`), coefficients `s` and right factor `R` (`d × C`).
///
/// It covers both programs an executor runs:
/// * **dense:** form `W' = W + L diag(s) Rᵀ`, then `x W'ᵀ + b`;
/// * **matrix-free:** `x Wᵀ + ((x R) ⊙ s) Lᵀ + b`.
///
/// On either path a term `l_jc s_c r_kc x_k` rounds at most three times as a
/// product, `C − 1` times accumulating over components, `d − 1` times over inputs,
/// once where the edit meets `W` and once for the bias: `k = C + d + 3`. The terms
/// `w_jk x_k` take shorter paths.
///
/// The underflow allowance is the larger of the two programs' (module *Underflow*):
/// * **matrix-free:** `d` for `x Wᵀ`; each coordinate `(x R)_c` carries `d`, which its
///   product with `s_c` scales by `|s_c|` and adds one, and its product with `l_jc`
///   scales by `|l_jc|` and adds one: `𝒜 = d + C + Σ_c (d |l_jc s_c| + |l_jc|)`;
/// * **dense:** each edit term `l_jc s_c r_kc` of `W'_jk` rounds twice as a product,
///   the second scaling the first's `η` by the factor it multiplies, in whichever
///   association: at most `1 + |l_jc| + |s_c| + |r_kc|`. The sums into `W'` add
///   none, and `x W'ᵀ` scales `W'_jk`'s allowance by `|x_k|` and adds one per
///   product: `𝒜 = d + Σ_k |x_k| (C + Σ_c (|l_jc| + |s_c| + |r_kc|))`, formed as
///   `d + Σ_k |x_k| · (C + Σ_c (|l_jc| + |s_c|)) + Σ_c Σ_k |r_kc x_k|`.
pub fn factored_edit_stage_band(
    weight: ArrayView2<'_, f64>,
    left: ArrayView2<'_, f64>,
    coefficients: ArrayView1<'_, f64>,
    right: ArrayView2<'_, f64>,
    bias: Option<ArrayView1<'_, f64>>,
    inputs: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, ShapeMismatch> {
    let (outputs, width) = weight.dim();
    let components = coefficients.len();
    check("edit left rows", outputs, left.nrows())?;
    check("edit left components", components, left.ncols())?;
    check("edit right rows", width, right.nrows())?;
    check("edit right components", components, right.ncols())?;
    check("edit input width", width, inputs.ncols())?;
    if let Some(bias) = bias {
        check("edit bias length", outputs, bias.len())?;
    }
    let mut band = Array2::<f64>::zeros((inputs.nrows(), outputs));
    let path_roundings = components + width + 3;
    let mut read = vec![0.0; components];
    let coefficient_sum = coefficients.iter().fold(0.0, |sum, coefficient| up(sum + coefficient.abs()));
    for (row, input) in inputs.outer_iter().enumerate() {
        for (component, slot) in read.iter_mut().enumerate() {
            let mut absolute_sum = 0.0;
            for (r, x) in right.column(component).iter().zip(input.iter()) {
                absolute_sum = up(absolute_sum + up((r * x).abs()));
            }
            *slot = absolute_sum;
        }
        let input_sum = input.iter().fold(0.0, |sum, x| up(sum + x.abs()));
        let read_sum = read.iter().fold(0.0, |sum, &value| up(sum + value));
        for output in 0..outputs {
            let mut absolute_sum = bias.map_or(0.0, |bias| bias[output].abs());
            for (w, x) in weight.row(output).iter().zip(input.iter()) {
                absolute_sum = up(absolute_sum + up((w * x).abs()));
            }
            let mut matrix_free = up(width as f64 + components as f64);
            let mut left_sum = 0.0;
            for component in 0..components {
                let magnitude = left[[output, component]].abs();
                let scale = up((left[[output, component]] * coefficients[component]).abs());
                absolute_sum = up(absolute_sum + up(scale * read[component]));
                matrix_free = up(matrix_free + up(up(width as f64 * scale) + magnitude));
                left_sum = up(left_sum + magnitude);
            }
            let per_input = up(up(components as f64 + left_sum) + coefficient_sum);
            let dense = up(up(width as f64 + up(input_sum * per_input)) + read_sum);
            band[[row, output]] = evaluation_band(path_roundings, absolute_sum, matrix_free.max(dense));
        }
    }
    Ok(band)
}

/// How one executed stage compares with its native execution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StageAgreement {
    /// Every entry has `|external − native| ≤ external_band + native_band`, decided
    /// with the discrepancy rounded up and the band sum rounded down: a certified
    /// agreement.
    pub agrees: bool,
    /// Some entry has `|external − native| > external_band + native_band`, decided
    /// with the discrepancy rounded down and the band sum rounded up: a certified
    /// violation. When neither `agrees` nor `refutes` holds, an entry sits inside the
    /// rounding of its own comparison and the stage is unresolved.
    pub refutes: bool,
    /// `(row, column)` of the entry with the largest discrepancy-to-band ratio.
    pub witness: (usize, usize),
    /// The discrepancy at the witness, rounded up.
    pub discrepancy: f64,
    /// The band sum at the witness.
    pub band: f64,
    /// `discrepancy / band` at the witness. It is for reports only: `agrees` is
    /// decided entry by entry without this division.
    pub ratio: f64,
}

fn require_finite(array: &'static str, values: ArrayView2<'_, f64>) -> Result<(), ReceiptRefusal> {
    match values.indexed_iter().find(|(_, value)| !value.is_finite()) {
        Some(((row, column), _)) => Err(ReceiptRefusal::NonFinite { array, row, column }),
        None => Ok(()),
    }
}

/// Compares an external executor's stage output with the native one against their
/// bands.
///
/// It refuses before comparing unless `external_execution` is binary64 with TF32 off,
/// the precondition every band here assumes.
pub fn compare_stage(
    external_execution: ExternalExecution<'_>,
    external: ArrayView2<'_, f64>,
    native: ArrayView2<'_, f64>,
    external_band: ArrayView2<'_, f64>,
    native_band: ArrayView2<'_, f64>,
) -> Result<StageAgreement, ReceiptRefusal> {
    external_execution.require_binary64_bands()?;
    let (rows, columns) = native.dim();
    for (what, array) in [
        ("external", external),
        ("external band", external_band),
        ("native band", native_band),
    ] {
        check(what, rows, array.nrows())?;
        check(what, columns, array.ncols())?;
    }
    for (what, array) in [
        ("external output", external),
        ("native output", native),
        ("external band", external_band),
        ("native band", native_band),
    ] {
        require_finite(what, array)?;
    }
    let mut agreement = StageAgreement {
        agrees: true,
        refutes: false,
        witness: (0, 0),
        discrepancy: 0.0,
        band: 0.0,
        ratio: f64::NEG_INFINITY,
    };
    for ((row, column), &native_value) in native.indexed_iter() {
        // A finite difference of floats is zero only when they are equal, and then it
        // is exact. Otherwise the exact difference lies between the rounded one's
        // neighbours, and so does the exact band sum.
        let difference = (external[[row, column]] - native_value).abs();
        let band_sum = external_band[[row, column]] + native_band[[row, column]];
        let (discrepancy, discrepancy_lower) = if difference == 0.0 {
            (0.0, 0.0)
        } else {
            (up(difference), down(difference))
        };
        let band = down(band_sum);
        agreement.agrees &= discrepancy <= band;
        agreement.refutes |= discrepancy_lower > up(band_sum);
        let ratio = if discrepancy == 0.0 { 0.0 } else { discrepancy / band };
        if ratio > agreement.ratio {
            agreement.witness = (row, column);
            agreement.discrepancy = discrepancy;
            agreement.band = band;
            agreement.ratio = ratio;
        }
    }
    Ok(agreement)
}

/// The largest `|external − native|` entry of one stage, measured. It compares two
/// implementations of a special function, so no band covers it: a receipt reports it and
/// never folds it into a band (#2951 correction C2).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeasuredDiscrepancy {
    pub largest: f64,
    /// `(row, column)` of the largest entry.
    pub witness: (usize, usize),
    /// The native value at the witness.
    pub native_at_witness: f64,
}

fn measured_discrepancy(
    external_name: &'static str,
    native_name: &'static str,
    external: ArrayView2<'_, f64>,
    native: ArrayView2<'_, f64>,
) -> Result<MeasuredDiscrepancy, ReceiptRefusal> {
    check(external_name, native.nrows(), external.nrows())?;
    check(external_name, native.ncols(), external.ncols())?;
    require_finite(external_name, external)?;
    require_finite(native_name, native)?;
    let mut measured = MeasuredDiscrepancy {
        largest: 0.0,
        witness: (0, 0),
        native_at_witness: native.iter().next().copied().unwrap_or(0.0),
    };
    for ((row, column), &native_value) in native.indexed_iter() {
        let difference = (external[[row, column]] - native_value).abs();
        if difference > measured.largest {
            measured = MeasuredDiscrepancy {
                largest: difference,
                witness: (row, column),
                native_at_witness: native_value,
            };
        }
    }
    Ok(measured)
}

/// One residual MLP block's tensors and the stages an external executor ran.
///
/// The block is `W₂ σ((W₁ + L diag(s) Rᵀ) x + b₁) + b₂` without the residual: the output
/// of a transformer MLP module. The `external_*` arrays are the executor's stage outputs,
/// with rows matching `inputs`.
#[derive(Clone, Copy, Debug)]
pub struct MlpBlockReceiptInputs<'a> {
    pub external_execution: ExternalExecution<'a>,
    pub activation: GaussianActivation,
    pub weight: ArrayView2<'a, f64>,
    pub bias: ArrayView1<'a, f64>,
    pub left: ArrayView2<'a, f64>,
    pub coefficients: ArrayView1<'a, f64>,
    pub right: ArrayView2<'a, f64>,
    pub weight_out: ArrayView2<'a, f64>,
    pub bias_out: ArrayView1<'a, f64>,
    pub inputs: ArrayView2<'a, f64>,
    pub external_pre_activation: ArrayView2<'a, f64>,
    pub external_activation: ArrayView2<'a, f64>,
    pub external_output: ArrayView2<'a, f64>,
}

/// A residual MLP block's receipt, stage by stage. Each stage compares from the external
/// executor's own stage input, so an upstream discrepancy never enters a downstream band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MlpBlockReceipt {
    /// `(W₁ + L diag(s) Rᵀ) x + b₁`, native through [`apply_anchored_linear`] with anchor
    /// one, against the factored-edit band on both sides.
    pub pre_activation: StageAgreement,
    /// The native activation at the external pre-activation against the external
    /// activation: measured only.
    pub activation_measured: MeasuredDiscrepancy,
    /// `W₂ a + b₂` at the external activation, native through [`NativeMlp::written`],
    /// against the affine band on both sides.
    pub output: StageAgreement,
    /// The native block run from `inputs` alone against the external output: measured only,
    /// since the activation discrepancy propagates into it.
    pub end_to_end_measured: MeasuredDiscrepancy,
}

/// The whole stage map of one residual MLP block ([`MlpBlockReceipt`]). It refuses before
/// executing anything unless `external_execution` is binary64 with TF32 off.
pub fn mlp_block_receipt(block: MlpBlockReceiptInputs<'_>) -> Result<MlpBlockReceipt, ReceiptRefusal> {
    block.external_execution.require_binary64_bands()?;
    check("read-in bias length", block.weight.nrows(), block.bias.len())?;
    let native = NativeMlp::new(
        block.weight.to_owned(),
        block.bias.to_owned(),
        block.weight_out.to_owned(),
        block.bias_out.to_owned(),
        block.activation,
    )?;

    // Anchor one rounds nothing, and the bias is added after the kernel: the path the
    // factored-edit band counts.
    let edited = apply_anchored_linear(
        block.weight,
        1.0,
        FactorView::new(block.left, block.right)?,
        block.coefficients,
        block.inputs,
    )?;
    let native_pre_activation = &*edited + &block.bias;
    let pre_band = factored_edit_stage_band(
        block.weight,
        block.left,
        block.coefficients,
        block.right,
        Some(block.bias),
        block.inputs,
    )?;
    let pre_activation = compare_stage(
        block.external_execution,
        block.external_pre_activation,
        native_pre_activation.view(),
        pre_band.view(),
        pre_band.view(),
    )?;

    let native_activation = native.activate(block.external_pre_activation)?;
    let activation_measured = measured_discrepancy(
        "external activation",
        "native activation",
        block.external_activation,
        native_activation.view(),
    )?;

    let native_output = native.written(block.external_activation)?;
    let output_band =
        affine_stage_band(block.weight_out, Some(block.bias_out), block.external_activation)?;
    let output = compare_stage(
        block.external_execution,
        block.external_output,
        native_output.view(),
        output_band.view(),
        output_band.view(),
    )?;

    let end_to_end = native.written(native.activate(native_pre_activation.view())?.view())?;
    let end_to_end_measured = measured_discrepancy(
        "external output",
        "native end-to-end output",
        block.external_output,
        end_to_end.view(),
    )?;
    Ok(MlpBlockReceipt {
        pre_activation,
        activation_measured,
        output,
        end_to_end_measured,
    })
}

/// A factored edit of one stored read, executed as `W + L diag(s) Rᵀ` at the rows it
/// reaches and as `W` at every other row.
#[derive(Clone, Copy, Debug)]
pub struct FactoredEditViews<'a> {
    pub factors: FactorView<'a>,
    pub coefficients: ArrayView1<'a, f64>,
    /// The rows of a receipt are the positions of one pass, so a declared position is a row
    /// index. [`PositionScope::every`] reaches every row.
    pub rows: &'a PositionScope,
}

/// One bias-free linear read of a block: its stored weight and an optional factored edit.
#[derive(Clone, Copy, Debug)]
pub struct EditedRead<'a> {
    pub weight: ArrayView2<'a, f64>,
    pub edit: Option<FactoredEditViews<'a>>,
}

/// `edited`'s rows where `scope` reaches, `unedited`'s rows elsewhere.
fn reached_rows(scope: &PositionScope, mut unedited: Array2<f64>, edited: &Array2<f64>) -> Array2<f64> {
    for row in (0..unedited.nrows()).filter(|&row| scope.reaches(row)) {
        unedited.row_mut(row).assign(&edited.row(row));
    }
    unedited
}

/// The native execution of one read at `inputs`: [`apply_anchored_linear`] at anchor one on
/// the rows the edit reaches, [`native_linear`] on every other row.
fn read_native(read: EditedRead<'_>, inputs: ArrayView2<'_, f64>) -> Result<Array2<f64>, ReceiptRefusal> {
    let unedited = native_linear(read.weight, inputs)?;
    let Some(edit) = read.edit else {
        return Ok((*unedited).clone());
    };
    edit.rows.check_within(inputs.nrows())?;
    let edited = apply_anchored_linear(read.weight, 1.0, edit.factors, edit.coefficients, inputs)?;
    Ok(reached_rows(edit.rows, (*unedited).clone(), &edited))
}

/// The band of one read at `inputs`: the factored-edit band on the rows the edit reaches, the
/// affine band on every other row. Without a bias an edited term rounds at most `C + d + 2`
/// times, inside the factored-edit band's `C + d + 3`.
fn read_band(read: EditedRead<'_>, inputs: ArrayView2<'_, f64>) -> Result<Array2<f64>, ReceiptRefusal> {
    let unedited = affine_stage_band(read.weight, None, inputs)?;
    let Some(edit) = read.edit else {
        return Ok(unedited);
    };
    edit.rows.check_within(inputs.nrows())?;
    let edited = factored_edit_stage_band(
        read.weight,
        edit.factors.left(),
        edit.coefficients,
        edit.factors.right(),
        None,
        inputs,
    )?;
    Ok(reached_rows(edit.rows, unedited, &edited))
}

/// One read executed natively at `inputs` and compared with the executor's rows against its
/// band on both sides.
fn banded_read_stage(
    external_execution: ExternalExecution<'_>,
    read: EditedRead<'_>,
    inputs: ArrayView2<'_, f64>,
    external: ArrayView2<'_, f64>,
) -> Result<StageAgreement, ReceiptRefusal> {
    let native = read_native(read, inputs)?;
    let band = read_band(read, inputs)?;
    compare_stage(external_execution, external, native.view(), band.view(), band.view())
}

/// Per-entry band of one evaluation of `a ⊙ u`: one rounded product, `k = 1`, and its
/// underflow allowance `𝒜 = 1`. The operands are floats, so `up(|fl(a·u)|)` lies at or
/// above the exact `|a·u|`.
fn product_stage_band(left: ArrayView2<'_, f64>, right: ArrayView2<'_, f64>) -> Array2<f64> {
    Zip::from(left)
        .and(right)
        .map_collect(|&a, &u| evaluation_band(1, up((a * u).abs()), 1.0))
}

/// The program an external executor's RMSNorm module runs on binary64 rows, as its source
/// declares it: `w ⊙ x (mean(x²) + ε)^{-1/2}` computed in one of two precisions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalRmsNormProgram {
    /// Every operation in binary64, [`MaskedNorm::Rms`]'s program, with its band
    /// [`rms_norm_band`].
    Binary64,
    /// transformers' `Qwen3RMSNorm` and `LlamaRMSNorm` on torch's CPU kernels:
    /// `h = x.to(float32)`, `v = h.pow(2).mean(-1)`, `h * torch.rsqrt(v + ε)` in binary32, cast
    /// back to binary64, then `w *` that in binary64 ([`binary32_internal_rms_norm_band`]).
    Binary32Internal,
}

/// The binary32 unit roundoff `2^-24`.
const BINARY32_UNIT_ROUNDOFF: f64 = f32::EPSILON as f64 / 2.0;

/// The largest rounding error of a binary32 operation whose result is subnormal:
/// half the subnormal spacing, `2^-150`.
const BINARY32_UNDERFLOW: f64 = f32::MIN_POSITIVE as f64 * BINARY32_UNIT_ROUNDOFF;

/// `γ_k` at the binary32 unit roundoff, rounded up; infinite when `k u ≥ 1`.
fn binary32_growth(operations: usize) -> f64 {
    let scaled = up(operations as f64 * BINARY32_UNIT_ROUNDOFF);
    if scaled < 1.0 { up(scaled / down(1.0 - scaled)) } else { f64::INFINITY }
}

/// Per-entry band of the external rows `external` of the [`ExternalRmsNormProgram::Binary32Internal`]
/// program at `inputs` (binary64 rows, `d` wide), against the exact binary64 RMSNorm
/// `y = w x (mean(x²) + ε)^{-1/2}` of the same rows.
///
/// # The program's roundings
/// With `u = 2^-24` and `U = 2^-53`, for an entry whose binary32 operands stay normal:
/// - `h = fl32(x)`: one rounding. `h²`: one more (ATen's `pow(2)` is `x * x`);
/// - the sum over the row in any order and any association: at most `d − 1` roundings,
///   plus one for a final conversion when the accumulator is wider than binary32;
/// - the mean's scaling, a division or a reciprocal and a product: two;
/// - `+ ε`: one, and `fl32(ε)` rounds once on its own branch;
/// so the argument `W` of the square root is `X (1 + θ)`, `|θ| ≤ γ_(d+6)(u)`, with
/// `X = mean(x²) + ε`, all terms being nonnegative.
/// - `torch.rsqrt` on the CPU is `1 / sqrt` (`Vectorized<float>::rsqrt` and the scalar kernel in
///   ATen's `UnaryOpsKernel.cpp`): a correctly rounded square root and a division, two roundings.
///   CUDA's `rsqrtf` is not correctly rounded, so any other device is refused.
/// So `R = r (1 + ρ')` with `r = X^{-1/2}` and `|ρ'| ≤ ρ = (1 + u)/((1 − u) √(1 − ω)) − 1`, where
/// `ω = γ_(d+6)(u) + A/ε` bounds `|W − X|/X` (below).
/// - `z = fl32(h R)`: one rounding; the cast back is exact; `ŷ = fl64(w z)`: `U`.
///
/// Then `|ŷ − y| ≤ λ |y| + a`, `λ = (1 + u)² (1 + ρ)(1 + U) − 1`, and `|y| ≤ (|ŷ| + a)/(1 − λ)`,
/// so the band is `λ (|ŷ| + a)/(1 − λ) + a`.
///
/// # Underflow
/// A binary32 result in the subnormal range rounds by an absolute `μ = 2^-150`, not relatively.
/// - An entry with `x² < 2^-126` enters the sum with `|fl32(h²) − x²| ≤ 2^-126`, so `n` such
///   entries move `W` by at most `A = (1 + γ_(d+6)) n 2^-126 / d + μ` (the last `μ` for a subnormal
///   mean), and `A/X ≤ A/ε`.
/// - The cast of an entry below `2^-126` and the product `z` in the subnormal range add
///   `a = |w| μ (1 + (1 + ρ)(1 + u)/√ε)(1 + U) + 2^-1074` to each entry, the last term
///   bounding the binary64 product's `2^-1075` by [`SUBNORMAL_SPACING`].
///
/// Every operation here rounds to nearest and then steps one float up (or down where the value
/// is subtracted), so the band is an upper bound computed in binary64.
///
/// It refuses a row whose sum of squares, grown by `γ_(d+6)`, reaches binary32's largest finite
/// value (the program would overflow, or read `rsqrt(inf) = 0` into a finite row), and an `ε`
/// that is not a positive binary32 normal number.
pub fn binary32_internal_rms_norm_band(
    epsilon: f64,
    gain: ArrayView1<'_, f64>,
    inputs: ArrayView2<'_, f64>,
    external: ArrayView2<'_, f64>,
) -> Result<Array2<f64>, ReceiptRefusal> {
    let (rows, width) = inputs.dim();
    check("binary32 norm gain", width, gain.len())?;
    check("binary32 norm external rows", rows, external.nrows())?;
    check("binary32 norm external width", width, external.ncols())?;
    require_finite("binary32 norm inputs", inputs)?;
    require_finite("external output", external)?;
    if !(epsilon >= f32::MIN_POSITIVE as f64 && epsilon <= f32::MAX as f64) {
        return Err(ReceiptRefusal::Binary32Epsilon { epsilon });
    }
    let u = BINARY32_UNIT_ROUNDOFF;
    let growth = binary32_growth(width + 6);
    let square_floor = f32::MIN_POSITIVE as f64;
    let root_epsilon = down(epsilon.sqrt());
    let mut band = Array2::<f64>::zeros((rows, width));
    for (row, (input, (executed, mut band_row))) in inputs
        .rows()
        .into_iter()
        .zip(external.rows().into_iter().zip(band.rows_mut()))
        .enumerate()
    {
        let (mut sum_of_squares, mut tiny) = (0.0_f64, 0_usize);
        for &value in input {
            let square = up(value * value);
            sum_of_squares = up(sum_of_squares + square);
            if value * value < square_floor {
                tiny += 1;
            }
        }
        if !(up(sum_of_squares * up(1.0 + growth)) < f32::MAX as f64) {
            return Err(ReceiptRefusal::Binary32Range { row });
        }
        let absolute_sum = up(up(up(up(1.0 + growth) * up(tiny as f64 * square_floor)) / width as f64)
            + BINARY32_UNDERFLOW);
        let relative = up(growth + up(absolute_sum / epsilon));
        if !(relative < 1.0) {
            return Err(ReceiptRefusal::Binary32Range { row });
        }
        let root_relative = up(up(up(1.0 + u) / down(1.0 - u)) / down(down(1.0 - relative).sqrt())) - 1.0;
        let root_relative = up(root_relative);
        let lambda = up(up(up(up(up(1.0 + u) * up(1.0 + u)) * up(1.0 + root_relative)) * up(1.0 + UNIT_ROUNDOFF)) - 1.0);
        let dominance = down(1.0 - lambda);
        let cast_reach = up(up(up(1.0 + root_relative) * up(1.0 + u)) / root_epsilon);
        for ((slot, &weight), &value) in band_row.iter_mut().zip(gain.iter()).zip(executed.iter()) {
            let absolute = up(up(up(up(weight.abs() * BINARY32_UNDERFLOW) * up(1.0 + cast_reach))
                * up(1.0 + UNIT_ROUNDOFF))
                + SUBNORMAL_SPACING);
            *slot = up(up(up(lambda * up(value.abs() + absolute)) / dominance) + absolute);
        }
    }
    Ok(band)
}

/// One RMSNorm stage: the native [`MaskedNorm::Rms`] at `inputs` with its band
/// [`rms_norm_band`], against the external module's rows with the band of its declared
/// `program`. A `Binary32Internal` stage must have run on the CPU.
pub fn rms_norm_stage(
    external_execution: ExternalExecution<'_>,
    program: ExternalRmsNormProgram,
    epsilon: f64,
    gain: ArrayView1<'_, f64>,
    inputs: ArrayView2<'_, f64>,
    external: ArrayView2<'_, f64>,
) -> Result<StageAgreement, ReceiptRefusal> {
    external_execution.require_binary64_bands()?;
    let native = MaskedNorm::Rms { epsilon, gain }.apply(inputs)?;
    let native_band = rms_norm_band(epsilon, gain, inputs, native.view())?;
    let external_band = match program {
        ExternalRmsNormProgram::Binary64 => {
            check("external norm rows", native.nrows(), external.nrows())?;
            check("external norm width", native.ncols(), external.ncols())?;
            require_finite("external output", external)?;
            rms_norm_band(epsilon, gain, inputs, external)?
        }
        ExternalRmsNormProgram::Binary32Internal => {
            if external_execution.device != "cpu" {
                return Err(ReceiptRefusal::ExternalDevice { program });
            }
            binary32_internal_rms_norm_band(epsilon, gain, inputs, external)?
        }
    };
    compare_stage(
        external_execution,
        external,
        native.view(),
        external_band.view(),
        native_band.view(),
    )
}

/// One SwiGLU MLP block's tensors and the stages an external executor ran.
///
/// The block writes `g W_dᵀ` with `g = SiLU(x W_gᵀ) ⊙ x W_uᵀ`, without the residual and without
/// MLP biases, as in Qwen3. `inputs` are the MLP module's input rows (the normalized stream),
/// and the `external_*` arrays are the executor's stage outputs, with rows matching `inputs`.
#[derive(Clone, Copy, Debug)]
pub struct SwigluBlockReceiptInputs<'a> {
    pub external_execution: ExternalExecution<'a>,
    pub inputs: ArrayView2<'a, f64>,
    pub gate: EditedRead<'a>,
    pub up: EditedRead<'a>,
    pub down: EditedRead<'a>,
    pub external_gate: ArrayView2<'a, f64>,
    pub external_up: ArrayView2<'a, f64>,
    pub external_activation: ArrayView2<'a, f64>,
    pub external_hidden: ArrayView2<'a, f64>,
    pub external_output: ArrayView2<'a, f64>,
}

/// A SwiGLU block's receipt, stage by stage. Each stage compares from the external executor's
/// own stage input, so an upstream discrepancy never enters a downstream band.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SwigluBlockReceipt {
    /// The gate read `x W_gᵀ` against its band on both sides.
    pub gate: StageAgreement,
    /// The up read `x W_uᵀ` against its band on both sides.
    pub up: StageAgreement,
    /// SiLU at the external gate rows against the external activation: measured only.
    pub activation_measured: MeasuredDiscrepancy,
    /// The external activation times the external up rows, entry by entry, against the
    /// external hidden rows: one rounded product on each side.
    pub hidden: StageAgreement,
    /// [`swiglu_hidden`] at the external gate and up rows against the external hidden rows:
    /// measured only, since it carries the SiLU implementation discrepancy.
    pub hidden_measured_propagation: MeasuredDiscrepancy,
    /// The down read at the external hidden rows against its band on both sides.
    pub output: StageAgreement,
    /// The native block run from `inputs` alone against the external output: measured only.
    pub end_to_end_measured: MeasuredDiscrepancy,
}

/// The whole stage map of one SwiGLU block ([`SwigluBlockReceipt`]). It refuses before executing
/// anything unless `external_execution` is binary64 with TF32 off, and refuses an edit whose
/// declared rows fall outside the rows it reads.
pub fn swiglu_block_receipt(
    block: SwigluBlockReceiptInputs<'_>,
) -> Result<SwigluBlockReceipt, ReceiptRefusal> {
    block.external_execution.require_binary64_bands()?;
    let gate = banded_read_stage(block.external_execution, block.gate, block.inputs, block.external_gate)?;
    let up_stage = banded_read_stage(block.external_execution, block.up, block.inputs, block.external_up)?;

    let native_activation = block.external_gate.mapv(|value| silu_derivatives(value)[0]);
    let activation_measured = measured_discrepancy(
        "external activation",
        "native activation",
        block.external_activation,
        native_activation.view(),
    )?;

    // The gate, up and activation stages fixed both operands' rows to the inputs' rows and
    // refused a non-finite entry; the gate and up widths must agree too.
    check("up read width", block.external_activation.ncols(), block.external_up.ncols())?;
    let native_hidden = Zip::from(block.external_activation)
        .and(block.external_up)
        .map_collect(|&a, &u| a * u);
    let hidden_band = product_stage_band(block.external_activation, block.external_up);
    let hidden = compare_stage(
        block.external_execution,
        block.external_hidden,
        native_hidden.view(),
        hidden_band.view(),
        hidden_band.view(),
    )?;
    let hidden_measured_propagation = measured_discrepancy(
        "external hidden",
        "native hidden",
        block.external_hidden,
        swiglu_hidden(block.external_gate, block.external_up)?.view(),
    )?;

    let output = banded_read_stage(
        block.external_execution,
        block.down,
        block.external_hidden,
        block.external_output,
    )?;

    let native_hidden_from_inputs = swiglu_hidden(
        read_native(block.gate, block.inputs)?.view(),
        read_native(block.up, block.inputs)?.view(),
    )?;
    let end_to_end = read_native(block.down, native_hidden_from_inputs.view())?;
    let end_to_end_measured = measured_discrepancy(
        "external output",
        "native end-to-end output",
        block.external_output,
        end_to_end.view(),
    )?;
    Ok(SwigluBlockReceipt {
        gate,
        up: up_stage,
        activation_measured,
        hidden,
        hidden_measured_propagation,
        output,
        end_to_end_measured,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, array};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    /// The execution record of a Rust-native "external" program in these tests.
    const BINARY64_CPU: ExternalExecution<'static> = ExternalExecution {
        dtype: "float64",
        device: "cpu",
        tf32_matmul: false,
    };

    fn uniform(rng: &mut StdRng, rows: usize, cols: usize) -> Array2<f64> {
        Array2::from_shape_simple_fn((rows, cols), || rng.random_range(-1.0..1.0))
    }

    /// `x Wᵀ + b` accumulated left to right.
    fn affine_forward(weight: &Array2<f64>, inputs: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_fn((inputs.nrows(), weight.nrows()), |(row, output)| {
            let mut sum = 0.0;
            for input in 0..weight.ncols() {
                sum += weight[[output, input]] * inputs[[row, input]];
            }
            sum
        })
    }

    /// `x Wᵀ + b` accumulated right to left.
    fn affine_reverse(weight: &Array2<f64>, inputs: &Array2<f64>) -> Array2<f64> {
        Array2::from_shape_fn((inputs.nrows(), weight.nrows()), |(row, output)| {
            let mut sum = 0.0;
            for input in (0..weight.ncols()).rev() {
                sum += weight[[output, input]] * inputs[[row, input]];
            }
            sum
        })
    }

    /// `W + L diag(s) Rᵀ` formed densely, each entry's edit accumulated left to right.
    fn dense_edit(
        weight: &Array2<f64>,
        left: &Array2<f64>,
        coefficients: &Array1<f64>,
        right: &Array2<f64>,
    ) -> Array2<f64> {
        let mut edited = weight.clone();
        for output in 0..weight.nrows() {
            for input in 0..weight.ncols() {
                let mut edit = 0.0;
                for component in 0..coefficients.len() {
                    edit += left[[output, component]] * coefficients[component] * right[[input, component]];
                }
                edited[[output, input]] += edit;
            }
        }
        edited
    }

    /// Two summation orders of the row `[2⁵³, 1, …, 1, −2⁵³]` against ones differ by
    /// fourteen: left to right each `+1` ties to even and is lost, right to left every
    /// partial sum is exact. The depth-`d` band covers it. Positive control: a band
    /// with the path depth dropped to one does not, so the depth count is load-bearing.
    #[test]
    fn affine_band_encloses_two_summation_orders_of_a_cancelling_row() {
        let width = 16;
        let mut weight = Array2::<f64>::ones((1, width));
        weight[[0, 0]] = 2.0_f64.powi(53);
        weight[[0, width - 1]] = -2.0_f64.powi(53);
        let inputs = Array2::<f64>::ones((1, width));
        let forward = affine_forward(&weight, &inputs);
        let reverse = affine_reverse(&weight, &inputs);
        assert_eq!((forward[[0, 0]], reverse[[0, 0]]), (0.0, 14.0));

        let band = affine_stage_band(weight.view(), None, inputs.view()).expect("shapes compose");
        let agreement = compare_stage(
            BINARY64_CPU,
            forward.view(),
            reverse.view(),
            band.view(),
            band.view(),
        )
        .expect("finite arrays");
        assert!(agreement.agrees && !agreement.refutes, "{agreement:?}");

        let absolute_sum = 2.0_f64.powi(54) + 14.0;
        let shallow = Array2::from_elem((1, 1), evaluation_band(1, absolute_sum, width as f64));
        let refuted = compare_stage(
            BINARY64_CPU,
            forward.view(),
            reverse.view(),
            shallow.view(),
            shallow.view(),
        )
        .expect("finite arrays");
        assert!(
            !refuted.agrees && refuted.refutes,
            "a depth-one band must be refuted by fourteen: {refuted:?}"
        );
    }

    /// Two correct binary64 evaluations of an affine entry whose products round into the
    /// subnormal range agree within the band (#4005). Each of 32 products
    /// `3e-161 · 7e-162 ≈ 42.504 · 2^-1074` rounds up to `43 · 2^-1074`, so the native
    /// evaluation lands on `1376 · 2^-1074`, while the correctly rounded exact sum, the most
    /// accurate an executor can return, is `1360 · 2^-1074`. The exact error is measured with
    /// both operands scaled by `2^600` into the normal range, where a fused multiply-add gives
    /// each product's rounding error exactly and every power-of-two scaling is exact. The band
    /// with its allowance `𝒜 = d` encloses the native error, and the two evaluations agree.
    /// Positive control: the relative band alone, `γ_d Σ|terms|` rounded up to one spacing,
    /// refutes the two correct evaluations.
    #[test]
    fn affine_band_encloses_products_that_round_into_the_subnormal_range() {
        let width = 32;
        let scale = 2.0_f64.powi(600);
        let (weight_entry, input_entry) = (3.0e-161, 7.0e-162);
        let weight = Array2::from_elem((1, width), weight_entry);
        let inputs = Array2::from_elem((1, width), input_entry);
        let native = affine_forward(&weight, &inputs);
        let (weight_scaled, input_scaled) = (weight_entry * scale, input_entry * scale);
        let product = weight_scaled * input_scaled;
        let residual = weight_scaled.mul_add(input_scaled, -product);
        let count = width as f64;
        // The exact entry times 2^1200 is `32 (product + residual)`, 32 a power of two. The
        // native entry times 2^1200 is within a factor two of `32 product`, so their
        // difference is exact.
        let exact_scaled = count * product + count * residual;
        let error = ((native[[0, 0]] * scale * scale - count * product) - count * residual).abs();
        assert!(
            error > 15.0 * SUBNORMAL_SPACING * scale * scale,
            "the fixture must lose most of sixteen subnormal spacings to underflow, lost {:e}",
            error / (scale * scale)
        );
        let band = affine_stage_band(weight.view(), None, inputs.view()).expect("shapes compose");
        assert!(
            error <= band[[0, 0]] * scale * scale,
            "the band {:e} must enclose the native evaluation's error {:e}",
            band[[0, 0]],
            error / (scale * scale)
        );

        let correctly_rounded = Array2::from_elem((1, 1), exact_scaled / scale / scale);
        let agreement = compare_stage(
            BINARY64_CPU,
            correctly_rounded.view(),
            native.view(),
            band.view(),
            band.view(),
        )
        .expect("finite arrays");
        assert!(agreement.agrees && !agreement.refutes, "{agreement:?}");

        let absolute_sum = (0..width).fold(0.0, |sum, _| up(sum + up((weight_entry * input_entry).abs())));
        let relative = Array2::from_elem((1, 1), up(up(accumulation_growth(width)) * absolute_sum));
        let refuted = compare_stage(
            BINARY64_CPU,
            correctly_rounded.view(),
            native.view(),
            relative.view(),
            relative.view(),
        )
        .expect("finite arrays");
        assert!(
            !refuted.agrees && refuted.refutes,
            "positive control: the relative band alone must refute two correct evaluations: {refuted:?}"
        );
    }

    /// A factored edit executed densely and matrix-free, under signed and continuous
    /// coefficients, agrees within the factored-edit band. Positive control: the native
    /// output displaced at one entry by twice its band sum is refuted at that entry.
    #[test]
    fn factored_edit_band_encloses_dense_and_matrix_free_programs() {
        let (outputs, width, components, rows) = (5, 7, 3, 4);
        let mut rng = StdRng::seed_from_u64(2951);
        let weight = uniform(&mut rng, outputs, width);
        let left = uniform(&mut rng, outputs, components);
        let right = uniform(&mut rng, width, components);
        let inputs = uniform(&mut rng, rows, width);
        let bias: Array1<f64> = Array1::from_shape_simple_fn(outputs, || rng.random_range(-1.0..1.0));
        let coefficients = array![-1.0, -0.37, 0.5];

        let dense = affine_forward(&dense_edit(&weight, &left, &coefficients, &right), &inputs) + &bias;

        let read = affine_forward(&right.t().to_owned(), &inputs) * &coefficients;
        let matrix_free = affine_forward(&weight, &inputs) + affine_forward(&left, &read) + &bias;

        let band = factored_edit_stage_band(
            weight.view(),
            left.view(),
            coefficients.view(),
            right.view(),
            Some(bias.view()),
            inputs.view(),
        )
        .expect("shapes compose");
        let agreement = compare_stage(
            BINARY64_CPU,
            dense.view(),
            matrix_free.view(),
            band.view(),
            band.view(),
        )
        .expect("finite arrays");
        assert!(agreement.agrees, "{agreement:?}");

        let mut displaced = matrix_free.clone();
        displaced[[2, 1]] += 2.0 * (band[[2, 1]] + band[[2, 1]]);
        let refuted = compare_stage(
            BINARY64_CPU,
            dense.view(),
            displaced.view(),
            band.view(),
            band.view(),
        )
        .expect("finite arrays");
        assert!(!refuted.agrees && refuted.refutes);
        assert_eq!(refuted.witness, (2, 1));
    }

    /// A discrepancy of exactly its band sum is decided by rounding alone, so neither
    /// agreement nor refutation is certified. Positive controls: a band sum of twice the
    /// discrepancy agrees, and one of half the discrepancy refutes.
    #[test]
    fn compare_stage_leaves_a_boundary_entry_unresolved() {
        let external = Array2::from_elem((1, 1), 0.0);
        let native = Array2::from_elem((1, 1), 1.0);
        let verdict = |half_band: f64| {
            let band = Array2::from_elem((1, 1), half_band);
            compare_stage(
                BINARY64_CPU,
                external.view(),
                native.view(),
                band.view(),
                band.view(),
            )
            .expect("finite arrays")
        };

        let boundary = verdict(0.5);
        assert!(!boundary.agrees && !boundary.refutes, "{boundary:?}");

        let wide = verdict(1.0);
        assert!(wide.agrees && !wide.refutes, "{wide:?}");

        let narrow = verdict(0.25);
        assert!(!narrow.agrees && narrow.refutes, "{narrow:?}");
    }

    /// Mismatched shapes and non-finite entries are refused, never compared.
    /// Positive control: the same arrays with matching shapes and finite entries compare.
    #[test]
    fn compare_stage_refuses_mismatched_shapes_and_non_finite_entries() {
        let values = Array2::<f64>::zeros((2, 3));
        let band = Array2::<f64>::zeros((2, 3));
        assert!(
            compare_stage(BINARY64_CPU, values.view(), values.view(), band.view(), band.view())
                .is_ok()
        );

        let narrow = Array2::<f64>::zeros((2, 2));
        assert!(matches!(
            compare_stage(BINARY64_CPU, narrow.view(), values.view(), band.view(), band.view()),
            Err(ReceiptRefusal::Shape(..))
        ));

        let mut poisoned = values.clone();
        poisoned[[1, 2]] = f64::NAN;
        assert_eq!(
            compare_stage(BINARY64_CPU, poisoned.view(), values.view(), band.view(), band.view()),
            Err(ReceiptRefusal::NonFinite {
                array: "external output",
                row: 1,
                column: 2,
            })
        );
    }

    /// An external stage that no binary64 band covers is refused before any verdict:
    /// a float32 execution, and a float64 execution with TF32 matmul on. Positive
    /// control: the same arrays under a binary64 record compare and agree.
    #[test]
    fn compare_stage_refuses_an_execution_no_binary64_band_covers() {
        let values = Array2::<f64>::zeros((2, 3));
        let band = Array2::<f64>::zeros((2, 3));
        let agreement =
            compare_stage(BINARY64_CPU, values.view(), values.view(), band.view(), band.view())
                .expect("a binary64 execution compares");
        assert!(agreement.agrees && !agreement.refutes, "{agreement:?}");

        let float32 = ExternalExecution {
            dtype: "float32",
            device: "cpu",
            tf32_matmul: false,
        };
        assert_eq!(
            compare_stage(float32, values.view(), values.view(), band.view(), band.view()),
            Err(ReceiptRefusal::ExternalPrecision {
                float64: false,
                tf32_matmul: false,
            })
        );

        let tf32 = ExternalExecution {
            dtype: "float64",
            device: "cuda",
            tf32_matmul: true,
        };
        assert_eq!(
            compare_stage(tf32, values.view(), values.view(), band.view(), band.view()),
            Err(ReceiptRefusal::ExternalPrecision {
                float64: true,
                tf32_matmul: true,
            })
        );
    }

    /// transformers' `Qwen3RMSNorm` on binary64 rows, emulated in binary32 as its source runs it:
    /// `h = x as f32`, the squares summed over the row in `order`, the mean, `+ ε as f32`,
    /// `1 / sqrt`, `h * r`, cast back, and the binary64 gain.
    fn binary32_rms_norm(inputs: &Array2<f64>, epsilon: f64, gain: &Array1<f64>, order: &[usize]) -> Array2<f64> {
        let width = inputs.ncols();
        let mut output = Array2::<f64>::zeros(inputs.raw_dim());
        for (input, mut output_row) in inputs.rows().into_iter().zip(output.rows_mut()) {
            let h: Vec<f32> = input.iter().map(|&value| value as f32).collect();
            let sum = order.iter().fold(0.0_f32, |sum, &column| sum + h[column] * h[column]);
            let inverse_root = 1.0_f32 / (sum / width as f32 + epsilon as f32).sqrt();
            for ((slot, &value), &weight) in output_row.iter_mut().zip(h.iter()).zip(gain.iter()) {
                *slot = weight * f64::from(value * inverse_root);
            }
        }
        output
    }

    /// The binary32 program's band covers its evaluation in every summation order: left to right,
    /// right to left and interleaved halves, on rows spanning 2^-30 to 2^20 with a zero and an entry
    /// whose binary32 square is subnormal, all with the native binary64 norm on the other side.
    ///
    /// Positive controls:
    /// - the same rows declared `Binary64` are refuted: binary64 bands cannot hold a binary32
    ///   evaluation, so the declared program is load-bearing;
    /// - an external entry displaced by twice its band sum is refuted there;
    /// - a CUDA record, a row whose squares overflow binary32 and a subnormal binary32 epsilon are
    ///   refused, typed;
    /// - rows the binary64 owner itself computed agree under `Binary64`.
    #[test]
    fn a_binary32_internal_norm_band_holds_every_summation_order_and_the_binary64_band_does_not() {
        let (rows, width) = (5, 64);
        let mut rng = StdRng::seed_from_u64(2951);
        let mut inputs = Array2::from_shape_simple_fn((rows, width), || {
            rng.random_range(-8.0..8.0) * 2.0_f64.powi(rng.random_range(-30..=20))
        });
        inputs[[1, 3]] = 0.0;
        inputs[[2, 7]] = 1.0e-25;
        let gain: Array1<f64> = Array1::from_shape_simple_fn(width, || rng.random_range(-2.0..2.0));
        // Qwen3's rms_norm_eps.
        let epsilon = 1.0e-6;
        let forward: Vec<usize> = (0..width).collect();
        let reverse: Vec<usize> = (0..width).rev().collect();
        let interleaved: Vec<usize> = (0..width / 2).flat_map(|c| [c, c + width / 2]).collect();
        let stage = |program, execution: ExternalExecution<'static>, external: &Array2<f64>| {
            rms_norm_stage(execution, program, epsilon, gain.view(), inputs.view(), external.view())
        };
        for order in [&forward, &reverse, &interleaved] {
            let external = binary32_rms_norm(&inputs, epsilon, &gain, order);
            let agreement = stage(ExternalRmsNormProgram::Binary32Internal, BINARY64_CPU, &external)
                .expect("a binary32 CPU stage compares");
            assert!(agreement.agrees && !agreement.refutes, "{agreement:?}");
            let refuted = stage(ExternalRmsNormProgram::Binary64, BINARY64_CPU, &external)
                .expect("a declared binary64 stage compares");
            assert!(
                !refuted.agrees && refuted.refutes,
                "a binary64 band must not hold a binary32 evaluation: {refuted:?}"
            );
        }

        let external = binary32_rms_norm(&inputs, epsilon, &gain, &forward);
        let band = binary32_internal_rms_norm_band(epsilon, gain.view(), inputs.view(), external.view())
            .expect("rows in binary32 range");
        let native = MaskedNorm::Rms { epsilon, gain: gain.view() }.apply(inputs.view()).expect("finite rows");
        let native_band = rms_norm_band(epsilon, gain.view(), inputs.view(), native.view()).expect("resolved rows");
        let mut displaced = external.clone();
        displaced[[3, 5]] += 2.0 * (band[[3, 5]] + native_band[[3, 5]]);
        let refuted = stage(ExternalRmsNormProgram::Binary32Internal, BINARY64_CPU, &displaced)
            .expect("a binary32 CPU stage compares");
        assert!(refuted.refutes && refuted.witness == (3, 5), "{refuted:?}");

        let cuda = ExternalExecution {
            dtype: "float64",
            device: "cuda",
            tf32_matmul: false,
        };
        assert_eq!(
            stage(ExternalRmsNormProgram::Binary32Internal, cuda, &external),
            Err(ReceiptRefusal::ExternalDevice {
                program: ExternalRmsNormProgram::Binary32Internal
            })
        );
        let mut overflowing = inputs.clone();
        overflowing[[4, 0]] = 2.0_f64.powi(70);
        assert_eq!(
            binary32_internal_rms_norm_band(epsilon, gain.view(), overflowing.view(), external.view()),
            Err(ReceiptRefusal::Binary32Range { row: 4 })
        );
        let subnormal = f64::from(f32::MIN_POSITIVE) / 4.0;
        assert_eq!(
            binary32_internal_rms_norm_band(subnormal, gain.view(), inputs.view(), external.view()),
            Err(ReceiptRefusal::Binary32Epsilon { epsilon: subnormal })
        );
        let binary64 = stage(ExternalRmsNormProgram::Binary64, BINARY64_CPU, &native)
            .expect("a binary64 stage compares");
        assert!(binary64.agrees && !binary64.refutes, "{binary64:?}");
    }

    /// A dense float64 program stands in for the external executor. It forms
    /// `W' = W + L diag(s) Rᵀ` densely, activates through the same gam-math owner, and writes
    /// left to right. The receipt agrees at the pre-activation and the output, and the
    /// measured activation discrepancy is zero because both sides call one implementation.
    ///
    /// Positive controls:
    /// - an external pre-activation displaced at one entry by twice its band sum is refuted
    ///   there, and moves the measured activation discrepancy off zero at that entry;
    /// - a float32 record refuses before anything executes.
    #[test]
    fn mlp_block_receipt_agrees_with_a_dense_float64_program_and_refutes_a_displaced_stage() {
        let (hidden, width, components, rows) = (6, 4, 2, 3);
        let mut rng = StdRng::seed_from_u64(2951);
        let weight = uniform(&mut rng, hidden, width);
        let left = uniform(&mut rng, hidden, components);
        let right = uniform(&mut rng, width, components);
        let weight_out = uniform(&mut rng, width, hidden);
        let inputs = uniform(&mut rng, rows, width);
        let bias: Array1<f64> = Array1::from_shape_simple_fn(hidden, || rng.random_range(-1.0..1.0));
        let bias_out: Array1<f64> =
            Array1::from_shape_simple_fn(width, || rng.random_range(-1.0..1.0));
        let coefficients = array![-1.0, 0.5];

        let external_pre_activation =
            affine_forward(&dense_edit(&weight, &left, &coefficients, &right), &inputs) + &bias;
        let executor = NativeMlp::new(
            weight.clone(),
            bias.clone(),
            weight_out.clone(),
            bias_out.clone(),
            GaussianActivation::ExactGelu,
        )
        .expect("the block's shapes compose");
        let external_activation = executor
            .activate(external_pre_activation.view())
            .expect("finite pre-activations");
        let external_output = affine_forward(&weight_out, &external_activation) + &bias_out;

        let receipt_for = |execution: ExternalExecution<'static>, pre_activation: &Array2<f64>| {
            mlp_block_receipt(MlpBlockReceiptInputs {
                external_execution: execution,
                activation: GaussianActivation::ExactGelu,
                weight: weight.view(),
                bias: bias.view(),
                left: left.view(),
                coefficients: coefficients.view(),
                right: right.view(),
                weight_out: weight_out.view(),
                bias_out: bias_out.view(),
                inputs: inputs.view(),
                external_pre_activation: pre_activation.view(),
                external_activation: external_activation.view(),
                external_output: external_output.view(),
            })
        };

        let receipt = receipt_for(BINARY64_CPU, &external_pre_activation)
            .expect("a binary64 execution compares");
        assert!(receipt.pre_activation.agrees && !receipt.pre_activation.refutes, "{receipt:?}");
        assert!(receipt.output.agrees && !receipt.output.refutes, "{receipt:?}");
        assert_eq!(receipt.activation_measured.largest, 0.0, "{receipt:?}");

        let band = factored_edit_stage_band(
            weight.view(),
            left.view(),
            coefficients.view(),
            right.view(),
            Some(bias.view()),
            inputs.view(),
        )
        .expect("shapes compose");
        let mut displaced = external_pre_activation.clone();
        displaced[[1, 2]] += 2.0 * (band[[1, 2]] + band[[1, 2]]);
        let refuted =
            receipt_for(BINARY64_CPU, &displaced).expect("a binary64 execution compares");
        assert!(
            refuted.pre_activation.refutes && refuted.pre_activation.witness == (1, 2),
            "{refuted:?}"
        );
        assert!(
            refuted.activation_measured.largest > 0.0 && refuted.activation_measured.witness == (1, 2),
            "{refuted:?}"
        );

        let float32 = ExternalExecution {
            dtype: "float32",
            device: "cpu",
            tf32_matmul: false,
        };
        assert_eq!(
            receipt_for(float32, &external_pre_activation),
            Err(ReceiptRefusal::ExternalPrecision {
                float64: false,
                tf32_matmul: false,
            })
        );
    }

    /// A dense float64 program stands in for the executor of a SwiGLU block writing `g W_dᵀ`,
    /// `g = SiLU(x W_gᵀ) ⊙ x W_uᵀ`. The gate read carries a factored edit declared at row 1
    /// only, the up read none, and the down read one reaching every row. The executor forms each
    /// edited weight densely, applies it at the rows it reaches, and sums left to right. The
    /// receipt agrees at every banded stage, and both measured discrepancies are zero because
    /// both sides call one SiLU owner.
    ///
    /// Positive controls:
    /// - an executor applying the gate edit at every row is refuted at the gate stage, at a row
    ///   the edit does not reach;
    /// - an external hidden entry displaced by four times its band sum is refuted there;
    /// - a declared position past the last row refuses;
    /// - a float32 record refuses before anything executes.
    #[test]
    fn swiglu_block_receipt_agrees_with_a_row_scoped_program_and_refutes_an_unscoped_edit() {
        let (hidden, width, components, rows) = (6, 4, 2, 3);
        let mut rng = StdRng::seed_from_u64(2951);
        let gate_weight = uniform(&mut rng, hidden, width);
        let gate_left = uniform(&mut rng, hidden, components);
        let gate_right = uniform(&mut rng, width, components);
        let up_weight = uniform(&mut rng, hidden, width);
        let down_weight = uniform(&mut rng, width, hidden);
        let down_left = uniform(&mut rng, width, components);
        let down_right = uniform(&mut rng, hidden, components);
        let inputs = uniform(&mut rng, rows, width);
        let coefficients = array![-1.0, 0.5];
        let row_one = PositionScope::declared(vec![1]).expect("one position");
        let every_row = PositionScope::every();

        let unscoped_gate =
            affine_forward(&dense_edit(&gate_weight, &gate_left, &coefficients, &gate_right), &inputs);
        let mut external_gate = affine_forward(&gate_weight, &inputs);
        external_gate.row_mut(1).assign(&unscoped_gate.row(1));
        let external_up = affine_forward(&up_weight, &inputs);
        let external_activation = external_gate.mapv(|value| silu_derivatives(value)[0]);
        let external_hidden = &external_activation * &external_up;
        let external_output = affine_forward(
            &dense_edit(&down_weight, &down_left, &coefficients, &down_right),
            &external_hidden,
        );

        let receipt_for = |execution: ExternalExecution<'static>,
                           gate_rows: &PositionScope,
                           executed_gate: &Array2<f64>,
                           executed_hidden: &Array2<f64>| {
            swiglu_block_receipt(SwigluBlockReceiptInputs {
                external_execution: execution,
                inputs: inputs.view(),
                gate: EditedRead {
                    weight: gate_weight.view(),
                    edit: Some(FactoredEditViews {
                        factors: FactorView::new(gate_left.view(), gate_right.view())
                            .expect("the gate factors share components"),
                        coefficients: coefficients.view(),
                        rows: gate_rows,
                    }),
                },
                up: EditedRead {
                    weight: up_weight.view(),
                    edit: None,
                },
                down: EditedRead {
                    weight: down_weight.view(),
                    edit: Some(FactoredEditViews {
                        factors: FactorView::new(down_left.view(), down_right.view())
                            .expect("the down factors share components"),
                        coefficients: coefficients.view(),
                        rows: &every_row,
                    }),
                },
                external_gate: executed_gate.view(),
                external_up: external_up.view(),
                external_activation: external_activation.view(),
                external_hidden: executed_hidden.view(),
                external_output: external_output.view(),
            })
        };

        let receipt = receipt_for(BINARY64_CPU, &row_one, &external_gate, &external_hidden)
            .expect("a binary64 execution compares");
        for stage in [receipt.gate, receipt.up, receipt.hidden, receipt.output] {
            assert!(stage.agrees && !stage.refutes, "{receipt:?}");
        }
        assert_eq!(receipt.activation_measured.largest, 0.0, "{receipt:?}");
        assert_eq!(receipt.hidden_measured_propagation.largest, 0.0, "{receipt:?}");

        let unscoped = receipt_for(BINARY64_CPU, &row_one, &unscoped_gate, &external_hidden)
            .expect("a binary64 execution compares");
        assert!(unscoped.gate.refutes && unscoped.gate.witness.0 != 1, "{unscoped:?}");

        let hidden_band = product_stage_band(external_activation.view(), external_up.view());
        let mut displaced = external_hidden.clone();
        displaced[[2, 3]] += 4.0 * (hidden_band[[2, 3]] + hidden_band[[2, 3]]);
        let refuted = receipt_for(BINARY64_CPU, &row_one, &external_gate, &displaced)
            .expect("a binary64 execution compares");
        assert!(refuted.hidden.refutes && refuted.hidden.witness == (2, 3), "{refuted:?}");

        let past_the_last_row = PositionScope::declared(vec![rows]).expect("one position");
        assert!(matches!(
            receipt_for(BINARY64_CPU, &past_the_last_row, &external_gate, &external_hidden),
            Err(ReceiptRefusal::Occurrence(..))
        ));

        let float32 = ExternalExecution {
            dtype: "float32",
            device: "cpu",
            tf32_matmul: false,
        };
        assert_eq!(
            receipt_for(float32, &row_one, &external_gate, &external_hidden),
            Err(ReceiptRefusal::ExternalPrecision {
                float64: false,
                tf32_matmul: false,
            })
        );
    }
}
