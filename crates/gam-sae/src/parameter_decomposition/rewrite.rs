//! Exact component-coordinate execution of a residual MLP block under masks
//! (#2951 P5).
//!
//! # The rewrite
//!
//! A residual MLP block `h' = h + W₂ σ(W₁ h + b₁) + b₂` whose weights factor
//! through component coordinates, `W₁ = U R` with `R ∈ ℝ^{C×d}` and `W₂ = Ū R̄`
//! with `R̄ ∈ ℝ^{C̄×H}`, executes under the component masks `M = diag(m)` and
//! `M̄ = diag(m̄)` as
//!
//! ```text
//! h' = h + Ū M̄ R̄ σ(U M R h + b₁) + b₂
//! ```
//!
//! This is the block with the edited tensors `U M R` and `Ū M̄ R̄`, for every real
//! mask: continuous, binary or signed. It is an identity, not a linearization:
//! the native nonlinearity receives exactly the summed input the edited tensor
//! gives it. The summed input is affine in `m` and the block is not, so two
//! components together do not act as the sum of their separate actions,
//! `F_{P₁+P₂} ≠ F_{P₁} + F_{P₂}`.
//!
//! The rewrite is matrix-free: `R h` gives the `C` component coordinates, the
//! mask scales them and `U` writes them, so `U M R` is never formed. Algebraic
//! equality is not bitwise equality, so a factor left all on executes its
//! original tensor on its original path, and a block with both factors all on is
//! bit-identical to [`NativeMlp::execute`].
//!
//! # The exact overcomplete factor
//!
//! Given a weight `W ∈ ℝ^{p×d}` and a component read `R ∈ ℝ^{C×d}` of full
//! column rank, so `R⁺R = I_d`, every write
//!
//! ```text
//! U = W R⁺ + N (I_C − R R⁺)
//! ```
//!
//! satisfies `U R = W R⁺R + N (R − R R⁺R) = W`. The exact writes form the affine
//! set `{U : U R = W}`, whose directions are the `Δ` with `Δ R = 0`: rows
//! orthogonal to `range(R)`. `U − N = (W − N R) R⁺` has its rows in `range(R)`,
//! so `U` is the orthogonal projection of the candidate `N` onto that set, the
//! exact write nearest `N` in Frobenius norm. The candidate is not a gauge of
//! interventions: under a mask `M ≠ I`, `N (I − R R⁺) M R` is nonzero in general,
//! so two exact writes of one weight are two different decompositions.
//!
//! With the thin QR `R = Q T` (`Q ∈ ℝ^{C×d}` orthonormal, `T ∈ ℝ^{d×d}` upper
//! triangular), `R⁺ = T⁻¹ Qᵀ`, so `U = N + X Qᵀ` with `X T = W − N R`: one
//! triangular solve and no pseudoinverse. The factorization is a one-time
//! construction whose read already holds `C ≥ d` rows.
//!
//! The solve is faer's blocked `solve_lower_triangular_in_place` at
//! `decomposition_parallelism()`, not gam-linalg's `triangular` owner, for speed
//! at LLM width. That owner substitutes one right-hand side at a time in a scalar
//! loop, about `p·d²` flops without blocking or parallelism: ≈2.7·10¹¹ for
//! `p = 16384`, `d = 4096`.
//!
//! # Coverage refusal
//!
//! A read with fewer components than input directions, `C < d`, has a null
//! direction `z`, and no write reproduces a weight with `W z ≠ 0`. That is
//! [`FactorRefusal::Uncovered`].
//!
//! With `C ≥ d` the rank decision reads the singular values of `T`, which are
//! those of `R`. The thin QR is backward stable, `Q T = R + δR` with
//! `‖δR‖₂ ≤ max(C, d)·ε·‖R‖₂`, and the SVD of `T` adds `d·ε·‖T‖₂`; by Weyl each
//! computed singular value is within the sum of the two [`factor_singular_band`]s
//! of the exact one. A smallest singular value at or below that band is not
//! resolved from zero: `R` is indistinguishable from a rank-deficient read to
//! working precision. The same predicate says the first-order error of the
//! triangular solve, `κ₂(R)` times the band relative to `‖R‖₂`, is at least one,
//! so the computed write would carry no correct digit of `U R = W`. Both
//! readings are [`FactorRefusal::Unresolved`]. An accepted factor reports how far
//! above the band it sits, [`ExactFactor::resolution_margin`].

use gam_linalg::faer_ndarray::{
    FaerArrayView, FaerLinalgError, FaerQr, FaerSvd, array2_to_matmut, decomposition_parallelism,
};
use gam_linalg::roundoff::factor_singular_band;
use gam_math::gaussian_activation::{
    GaussianActivation, GaussianActivationError, gaussian_smoothing_derivatives,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use std::fmt;

/// Two tensor dimensions that must agree and do not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ShapeMismatch {
    /// The dimension being checked.
    pub what: &'static str,
    pub expected: usize,
    pub found: usize,
}

impl fmt::Display for ShapeMismatch {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{}: expected {}, found {}",
            self.what, self.expected, self.found
        )
    }
}

impl std::error::Error for ShapeMismatch {}

pub(super) fn check(what: &'static str, expected: usize, found: usize) -> Result<(), ShapeMismatch> {
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

/// A refused exact factor `U R = W`.
#[derive(Debug)]
pub enum FactorRefusal {
    /// The weight, read and candidate write do not compose as `p×C · C×d = p×d`.
    Shape(ShapeMismatch),
    /// A tensor of the factor has a non-finite entry.
    NonFinite { tensor: &'static str },
    /// A read with fewer components than input directions leaves an input
    /// direction uncovered.
    Uncovered {
        components: usize,
        input_width: usize,
    },
    /// The read's smallest singular value is not resolved from zero by the
    /// backward-error band of the QR and SVD that measured it.
    Unresolved {
        smallest_singular_value: f64,
        band: f64,
    },
    /// The QR of the read or the SVD of its triangular factor failed.
    Factorization(FaerLinalgError),
}

impl fmt::Display for FactorRefusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(mismatch) => {
                write!(formatter, "an exact factor's shapes do not compose: {mismatch}")
            }
            Self::NonFinite { tensor } => {
                write!(formatter, "the {tensor} of an exact factor has a non-finite entry")
            }
            Self::Uncovered {
                components,
                input_width,
            } => write!(
                formatter,
                "a read with {components} components cannot cover {input_width} input directions"
            ),
            Self::Unresolved {
                smallest_singular_value,
                band,
            } => write!(
                formatter,
                "the read's smallest singular value {smallest_singular_value:e} is not resolved above the QR and SVD rounding band {band:e}"
            ),
            Self::Factorization(error) => {
                write!(formatter, "the exact factor's QR or SVD failed: {error}")
            }
        }
    }
}

impl std::error::Error for FactorRefusal {}

/// A write `U` and a read `R` with `U R = W` for the weight the write was solved
/// against.
#[derive(Clone, Debug)]
pub struct ExactFactor {
    write: Array2<f64>,
    read: Array2<f64>,
    resolution_margin: f64,
}

impl ExactFactor {
    /// The exact write `U = N + (W − N R) R⁺` nearest `candidate_write = N`, for
    /// `weight = W` and `read = R` (see the module documentation).
    pub fn solve_write(
        weight: ArrayView2<'_, f64>,
        read: ArrayView2<'_, f64>,
        candidate_write: ArrayView2<'_, f64>,
    ) -> Result<Self, FactorRefusal> {
        let (components, input_width) = read.dim();
        check("weight input width", input_width, weight.ncols()).map_err(FactorRefusal::Shape)?;
        check("candidate write rows", weight.nrows(), candidate_write.nrows())
            .map_err(FactorRefusal::Shape)?;
        check("candidate write components", components, candidate_write.ncols())
            .map_err(FactorRefusal::Shape)?;
        for (tensor, entries) in [
            ("weight", weight),
            ("read", read),
            ("candidate write", candidate_write),
        ] {
            if !entries.iter().all(|value| value.is_finite()) {
                return Err(FactorRefusal::NonFinite { tensor });
            }
        }
        if components < input_width {
            return Err(FactorRefusal::Uncovered {
                components,
                input_width,
            });
        }
        let (orthonormal, triangular) = read.qr().map_err(FactorRefusal::Factorization)?;
        let (_, singular_values, _) = triangular
            .svd(false, false)
            .map_err(FactorRefusal::Factorization)?;
        let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
        let smallest = singular_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let band = factor_singular_band(components, input_width, largest)
            + factor_singular_band(input_width, input_width, largest);
        if !(smallest > band) {
            return Err(FactorRefusal::Unresolved {
                smallest_singular_value: smallest,
                band,
            });
        }
        let residual = &weight - &candidate_write.dot(&read);
        // X T = W − N R is Tᵀ Xᵀ = (W − N R)ᵀ: one blocked lower-triangular solve,
        // in place over the transposed residual.
        let mut solved = residual.t().to_owned();
        faer::linalg::triangular_solve::solve_lower_triangular_in_place(
            FaerArrayView::new(&triangular).as_ref().transpose(),
            array2_to_matmut(&mut solved),
            decomposition_parallelism(),
        );
        let write = &candidate_write + &orthonormal.dot(&solved).t();
        Ok(Self {
            write,
            read: read.to_owned(),
            resolution_margin: smallest / band,
        })
    }

    /// The write `U`, `p × C`.
    pub fn write(&self) -> ArrayView2<'_, f64> {
        self.write.view()
    }

    /// The read `R`, `C × d`.
    pub fn read(&self) -> ArrayView2<'_, f64> {
        self.read.view()
    }

    /// The number of components `C`.
    pub fn components(&self) -> usize {
        self.read.nrows()
    }

    /// The read's smallest singular value over the refusal band: how far above
    /// the working-precision rank cut this factor sits. Always greater than one.
    pub fn resolution_margin(&self) -> f64 {
        self.resolution_margin
    }

    /// `U M R x` for each row `x` of `inputs`, with `M = diag(mask)`, never
    /// forming `U M R`: `R x` gives the component coordinates, the mask scales
    /// them and `U` writes them.
    pub fn apply_masked(
        &self,
        inputs: ArrayView2<'_, f64>,
        mask: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, ShapeMismatch> {
        check("factor input width", self.read.ncols(), inputs.ncols())?;
        check("component mask length", self.components(), mask.len())?;
        let mut coordinates = inputs.dot(&self.read.t());
        coordinates *= &mask;
        Ok(coordinates.dot(&self.write.t()))
    }

    /// The pullback of [`Self::apply_masked`]. For a loss `ℓ` of `y = U M R x`
    /// with row cotangents `ȳ = ∂ℓ/∂y`, it returns each input row's cotangent
    /// `x̄ = Rᵀ (m ⊙ Uᵀ ȳ)` and the mask cotangent `m̄_c = Σ_rows (Uᵀ ȳ)_c (R x)_c`.
    /// The map is linear in `x` and in `m`, so both are exact adjoints. Like the
    /// forward, the pullback never forms `U M R`.
    pub fn apply_masked_pullback(
        &self,
        inputs: ArrayView2<'_, f64>,
        mask: ArrayView1<'_, f64>,
        output_cotangent: ArrayView2<'_, f64>,
    ) -> Result<MaskedFactorCotangents, ShapeMismatch> {
        check("factor input width", self.read.ncols(), inputs.ncols())?;
        check("component mask length", self.components(), mask.len())?;
        check("output cotangent width", self.write.nrows(), output_cotangent.ncols())?;
        check("output cotangent rows", inputs.nrows(), output_cotangent.nrows())?;
        let written = output_cotangent.dot(&self.write);
        let coordinates = inputs.dot(&self.read.t());
        let mask_cotangent = (&written * &coordinates).sum_axis(ndarray::Axis(0));
        let mut scaled = written;
        scaled *= &mask;
        Ok(MaskedFactorCotangents {
            inputs: scaled.dot(&self.read),
            mask: mask_cotangent,
        })
    }
}

/// The cotangents [`ExactFactor::apply_masked_pullback`] returns.
#[derive(Clone, Debug)]
pub struct MaskedFactorCotangents {
    /// `x̄ = Rᵀ (m ⊙ Uᵀ ȳ)` for each input row, `n × d`.
    pub inputs: Array2<f64>,
    /// `m̄_c = Σ_rows (Uᵀ ȳ)_c (R x)_c`, one entry per component.
    pub mask: Array1<f64>,
}

/// One of the two weights of a residual MLP block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlpFactor {
    /// `W₁`, from the residual stream into the hidden layer.
    ReadIn,
    /// `W₂`, from the hidden layer back into the residual stream.
    WriteOut,
}

impl fmt::Display for MlpFactor {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::ReadIn => "read-in",
            Self::WriteOut => "write-out",
        })
    }
}

/// A refused MLP rewrite or execution.
#[derive(Debug)]
pub enum RewriteError {
    Shape(ShapeMismatch),
    /// A weight of the block has no exact factor through its declared read.
    Factor {
        factor: MlpFactor,
        refusal: FactorRefusal,
    },
    Activation(GaussianActivationError),
}

impl From<ShapeMismatch> for RewriteError {
    fn from(mismatch: ShapeMismatch) -> Self {
        Self::Shape(mismatch)
    }
}

impl From<GaussianActivationError> for RewriteError {
    fn from(error: GaussianActivationError) -> Self {
        Self::Activation(error)
    }
}

impl fmt::Display for RewriteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shape(mismatch) => write!(formatter, "MLP block shapes do not compose: {mismatch}"),
            Self::Factor { factor, refusal } => {
                write!(formatter, "the {factor} weight has no exact factor: {refusal}")
            }
            Self::Activation(error) => write!(formatter, "the native activation refused: {error}"),
        }
    }
}

impl std::error::Error for RewriteError {}

/// A residual MLP block `h' = h + W₂ σ(W₁ h + b₁) + b₂` on its original tensors:
/// `W₁ ∈ ℝ^{H×d}`, `b₁ ∈ ℝ^H`, `W₂ ∈ ℝ^{d×H}`, `b₂ ∈ ℝ^d`.
#[derive(Clone, Debug)]
pub struct NativeMlp {
    read_in: Array2<f64>,
    bias_in: Array1<f64>,
    write_out: Array2<f64>,
    bias_out: Array1<f64>,
    activation: GaussianActivation,
}

impl NativeMlp {
    pub fn new(
        read_in: Array2<f64>,
        bias_in: Array1<f64>,
        write_out: Array2<f64>,
        bias_out: Array1<f64>,
        activation: GaussianActivation,
    ) -> Result<Self, ShapeMismatch> {
        let (hidden_width, width) = read_in.dim();
        check("read-in bias length", hidden_width, bias_in.len())?;
        check("write-out rows", width, write_out.nrows())?;
        check("write-out columns", hidden_width, write_out.ncols())?;
        check("write-out bias length", width, bias_out.len())?;
        Ok(Self {
            read_in,
            bias_in,
            write_out,
            bias_out,
            activation,
        })
    }

    /// The residual-stream width `d`.
    pub fn width(&self) -> usize {
        self.read_in.ncols()
    }

    /// The hidden width `H`.
    pub fn hidden_width(&self) -> usize {
        self.read_in.nrows()
    }

    pub fn activation(&self) -> GaussianActivation {
        self.activation
    }

    /// `W₁`, `H × d`.
    pub fn read_in(&self) -> ArrayView2<'_, f64> {
        self.read_in.view()
    }

    /// `b₁`, length `H`.
    pub fn bias_in(&self) -> ArrayView1<'_, f64> {
        self.bias_in.view()
    }

    /// `W₂`, `d × H`.
    pub fn write_out(&self) -> ArrayView2<'_, f64> {
        self.write_out.view()
    }

    /// `b₂`, length `d`.
    pub fn bias_out(&self) -> ArrayView1<'_, f64> {
        self.bias_out.view()
    }

    /// The summed input `W₁ x + b₁` for each row `x` of `inputs`.
    pub fn summed_input(&self, inputs: ArrayView2<'_, f64>) -> Result<Array2<f64>, ShapeMismatch> {
        check("input width", self.width(), inputs.ncols())?;
        Ok(inputs.dot(&self.read_in.t()) + &self.bias_in)
    }

    /// The native activation `σ`, entrywise.
    pub fn activate(
        &self,
        summed_input: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, GaussianActivationError> {
        let mut activations = Array2::<f64>::zeros(summed_input.raw_dim());
        let mut value = [0.0];
        for (slot, &point) in activations.iter_mut().zip(summed_input.iter()) {
            gaussian_smoothing_derivatives(self.activation, point, 0.0, &mut value)?;
            *slot = value[0];
        }
        Ok(activations)
    }

    /// The block's write `W₂ a + b₂` for each row `a` of `activations`, without
    /// the residual: what a transformer layer adds to its residual stream in the
    /// source's own order.
    pub fn written(&self, activations: ArrayView2<'_, f64>) -> Result<Array2<f64>, ShapeMismatch> {
        check("activation width", self.hidden_width(), activations.ncols())?;
        Ok(activations.dot(&self.write_out.t()) + &self.bias_out)
    }

    /// `x + (W₂ a + b₂)` for each row `x` of `inputs` and its row `a` of
    /// `activations`: [`Self::written`] with the residual added last.
    pub fn write(
        &self,
        inputs: ArrayView2<'_, f64>,
        activations: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, ShapeMismatch> {
        add_residual(inputs, self.written(activations)?)
    }

    /// The block on its original tensors: [`Self::summed_input`], then
    /// [`Self::activate`], then [`Self::write`].
    pub fn execute(&self, inputs: ArrayView2<'_, f64>) -> Result<Array2<f64>, RewriteError> {
        let summed_input = self.summed_input(inputs)?;
        let activations = self.activate(summed_input.view())?;
        Ok(self.write(inputs, activations.view())?)
    }
}

/// `x + written` for each row `x` of `inputs`, in torch's association
/// (`residual + hidden_states`, with the bias inside the linear map): the
/// residual is added last. Floating-point addition commutes bitwise but does not
/// associate, so this order is part of the native reference. The native and the
/// component write share it, so the all-on path stays bit-identical.
fn add_residual(
    inputs: ArrayView2<'_, f64>,
    mut written: Array2<f64>,
) -> Result<Array2<f64>, ShapeMismatch> {
    check("input width", written.ncols(), inputs.ncols())?;
    check("input rows", written.nrows(), inputs.nrows())?;
    written += &inputs;
    Ok(written)
}

/// The component read `R` of one weight and the candidate write `N` whose
/// nearest exact write the block executes.
#[derive(Clone, Copy, Debug)]
pub struct ComponentRead<'a> {
    pub read: ArrayView2<'a, f64>,
    pub candidate_write: ArrayView2<'a, f64>,
}

/// How one factor of a [`ComponentMlp`] executes.
#[derive(Clone, Copy, Debug)]
pub enum ComponentMask<'a> {
    /// Every component on: the original tensor executes on its original path.
    AllOn,
    /// `M = diag(m)` scales the component coordinates. Entries are any real
    /// numbers: continuous, binary or signed.
    Components(ArrayView1<'a, f64>),
}

/// The masks of both factors of a [`ComponentMlp`].
#[derive(Clone, Copy, Debug)]
pub struct MlpMask<'a> {
    pub read_in: ComponentMask<'a>,
    pub write_out: ComponentMask<'a>,
}

/// A residual MLP block executed in component coordinates under masks (see the
/// module documentation).
#[derive(Clone, Debug)]
pub struct ComponentMlp {
    native: NativeMlp,
    read_in: ExactFactor,
    write_out: ExactFactor,
}

impl ComponentMlp {
    /// Factors both weights of `native` exactly through their declared reads.
    pub fn new(
        native: NativeMlp,
        read_in: ComponentRead<'_>,
        write_out: ComponentRead<'_>,
    ) -> Result<Self, RewriteError> {
        let read_in = ExactFactor::solve_write(
            native.read_in.view(),
            read_in.read,
            read_in.candidate_write,
        )
        .map_err(|refusal| RewriteError::Factor {
            factor: MlpFactor::ReadIn,
            refusal,
        })?;
        let write_out = ExactFactor::solve_write(
            native.write_out.view(),
            write_out.read,
            write_out.candidate_write,
        )
        .map_err(|refusal| RewriteError::Factor {
            factor: MlpFactor::WriteOut,
            refusal,
        })?;
        Ok(Self {
            native,
            read_in,
            write_out,
        })
    }

    pub fn native(&self) -> &NativeMlp {
        &self.native
    }

    /// The exact factor `W₁ = U R`.
    pub fn read_in(&self) -> &ExactFactor {
        &self.read_in
    }

    /// The exact factor `W₂ = Ū R̄`.
    pub fn write_out(&self) -> &ExactFactor {
        &self.write_out
    }

    /// The summed input `U M R x + b₁`, or the native `W₁ x + b₁` with the
    /// read-in factor all on.
    pub fn summed_input(
        &self,
        inputs: ArrayView2<'_, f64>,
        mask: ComponentMask<'_>,
    ) -> Result<Array2<f64>, ShapeMismatch> {
        match mask {
            ComponentMask::AllOn => self.native.summed_input(inputs),
            ComponentMask::Components(mask) => {
                Ok(self.read_in.apply_masked(inputs, mask)? + &self.native.bias_in)
            }
        }
    }

    /// The write `Ū M̄ R̄ a + b₂`, or the native `W₂ a + b₂` with the write-out
    /// factor all on, without the residual.
    pub fn written(
        &self,
        activations: ArrayView2<'_, f64>,
        mask: ComponentMask<'_>,
    ) -> Result<Array2<f64>, ShapeMismatch> {
        match mask {
            ComponentMask::AllOn => self.native.written(activations),
            ComponentMask::Components(mask) => {
                Ok(self.write_out.apply_masked(activations, mask)? + &self.native.bias_out)
            }
        }
    }

    /// `x + (Ū M̄ R̄ a + b₂)`: [`Self::written`] with the residual added last.
    pub fn write(
        &self,
        inputs: ArrayView2<'_, f64>,
        activations: ArrayView2<'_, f64>,
        mask: ComponentMask<'_>,
    ) -> Result<Array2<f64>, ShapeMismatch> {
        add_residual(inputs, self.written(activations, mask)?)
    }

    /// `h + Ū M̄ R̄ σ(U M R h + b₁) + b₂` for each row `h` of `inputs`: the native
    /// activation applied to [`Self::summed_input`], written by [`Self::write`].
    pub fn execute(
        &self,
        inputs: ArrayView2<'_, f64>,
        mask: MlpMask<'_>,
    ) -> Result<Array2<f64>, RewriteError> {
        let summed_input = self.summed_input(inputs, mask.read_in)?;
        let activations = self.native.activate(summed_input.view())?;
        Ok(self.write(inputs, activations.view(), mask.write_out)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parameter_decomposition::test_support::component_mlp::{
        HIDDEN, READ_IN_COMPONENTS, ROWS, WIDTH, WRITE_OUT_COMPONENTS, bitwise_equal, mask_family,
        random_block, uniform, uniform_vector,
    };
    use gam_linalg::roundoff::accumulation_growth;
    use ndarray::{Axis, array};
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};

    fn edited_tensor(factor: &ExactFactor, mask: ArrayView1<'_, f64>) -> Array2<f64> {
        (&factor.write() * &mask).dot(&factor.read())
    }

    /// The block with the edited tensors `U M R` and `Ū M̄ R̄` formed explicitly:
    /// the native reference the rewrite must reproduce.
    fn edited_block(
        block: &ComponentMlp,
        read_in_mask: ArrayView1<'_, f64>,
        write_out_mask: ArrayView1<'_, f64>,
    ) -> NativeMlp {
        NativeMlp::new(
            edited_tensor(&block.read_in, read_in_mask),
            block.native.bias_in.clone(),
            edited_tensor(&block.write_out, write_out_mask),
            block.native.bias_out.clone(),
            block.native.activation,
        )
        .expect("edited tensors keep the block's shapes")
    }

    /// `|U| |M| |R| |x|` for each row `x` of `inputs`.
    fn magnitude(
        factor: &ExactFactor,
        mask: ArrayView1<'_, f64>,
        inputs: ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        (inputs.mapv(f64::abs).dot(&factor.read().mapv(f64::abs).t()) * &mask.mapv(f64::abs))
            .dot(&factor.write().mapv(f64::abs).t())
    }

    /// Per-coordinate distance of one path's `fl(U M R x)` plus `additions`
    /// rounded additions from the exact value, divided by `1 − γ` so it also
    /// covers the rounding of the magnitude itself.
    ///
    /// The rewrite forms `R x` (`d` rounded products and sums), scales by the
    /// mask (one), and writes with `U` (`C`): `γ_d ⊕ γ_1 ⊕ γ_C ≤ γ_{C+d+1}`
    /// (Higham ASNA Lemma 3.3), times `|U| |M| |R| |x|`. The edited tensor forms
    /// `(U M) R` (`C + 1` per entry) and multiplies by `x` (`d`): the same
    /// `γ_{C+d+1}`. Any summation order obeys these bounds, so they hold for any
    /// matrix-multiply kernel. Each later addition of a shift `s` adds at most
    /// `u·(|value| + |s|)`, absorbed as one more step of `γ` over `A + |s|`.
    fn single_path_band(
        factor: &ExactFactor,
        mask: ArrayView1<'_, f64>,
        inputs: ArrayView2<'_, f64>,
        shift: ArrayView2<'_, f64>,
        additions: usize,
    ) -> Array2<f64> {
        let growth =
            accumulation_growth(factor.components() + factor.read().ncols() + 1 + additions);
        (magnitude(factor, mask, inputs) + &shift) * (growth / (1.0 - growth))
    }

    fn violations(gap: &Array2<f64>, band: &Array2<f64>) -> usize {
        gap.iter()
            .zip(band.iter())
            .filter(|(difference, bound)| difference.abs() > **bound)
            .count()
    }

    fn frobenius(matrix: &Array2<f64>) -> f64 {
        matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
    }

    #[test]
    fn summed_input_under_every_mask_kind_equals_the_edited_tensor_within_the_derived_band() {
        let (block, inputs) = random_block(GaussianActivation::ExactGelu, 2951);
        let mut rng = StdRng::seed_from_u64(5);
        let write_out_ones = Array1::ones(WRITE_OUT_COMPONENTS);
        let shift = Array2::<f64>::zeros((ROWS, HIDDEN)) + &block.native.bias_in.mapv(f64::abs);
        for (kind, mask) in mask_family(&mut rng, READ_IN_COMPONENTS) {
            let rewritten = block
                .summed_input(inputs.view(), ComponentMask::Components(mask.view()))
                .expect("masked summed input");
            let edited = edited_block(&block, mask.view(), write_out_ones.view())
                .summed_input(inputs.view())
                .expect("edited summed input");
            let band =
                single_path_band(&block.read_in, mask.view(), inputs.view(), shift.view(), 1) * 2.0;
            assert_eq!(
                violations(&(&rewritten - &edited), &band),
                0,
                "{kind} mask: the rewrite's summed input left the derived band around the edited tensor's"
            );

            // Positive control: the band resolves a 1e-9 move of one mask entry.
            let mut moved = mask.clone();
            moved[1] += 1.0e-9;
            let perturbed = edited_block(&block, moved.view(), write_out_ones.view())
                .summed_input(inputs.view())
                .expect("perturbed summed input");
            assert!(
                violations(&(&rewritten - &perturbed), &band) > 0,
                "{kind} mask: the band must resolve a 1e-9 move of one mask entry"
            );
        }
    }

    #[test]
    fn write_under_every_mask_kind_equals_the_edited_tensor_within_the_derived_band() {
        let (block, inputs) = random_block(GaussianActivation::ExactGelu, 2952);
        let mut rng = StdRng::seed_from_u64(6);
        let read_in_ones = Array1::ones(READ_IN_COMPONENTS);
        let summed_input = block
            .native
            .summed_input(inputs.view())
            .expect("native summed input");
        let activations = block
            .native
            .activate(summed_input.view())
            .expect("native activations");
        let shift = inputs.mapv(f64::abs) + &block.native.bias_out.mapv(f64::abs);
        for (kind, mask) in mask_family(&mut rng, WRITE_OUT_COMPONENTS) {
            let rewritten = block
                .write(inputs.view(), activations.view(), ComponentMask::Components(mask.view()))
                .expect("masked write");
            let edited = edited_block(&block, read_in_ones.view(), mask.view())
                .write(inputs.view(), activations.view())
                .expect("edited write");
            let band = single_path_band(
                &block.write_out,
                mask.view(),
                activations.view(),
                shift.view(),
                2,
            ) * 2.0;
            assert_eq!(
                violations(&(&rewritten - &edited), &band),
                0,
                "{kind} mask: the rewrite's write left the derived band around the edited tensor's"
            );

            // Positive control: the band resolves a 1e-9 move of one mask entry.
            let mut moved = mask.clone();
            moved[2] += 1.0e-9;
            let perturbed = edited_block(&block, read_in_ones.view(), moved.view())
                .write(inputs.view(), activations.view())
                .expect("perturbed write");
            assert!(
                violations(&(&rewritten - &perturbed), &band) > 0,
                "{kind} mask: the band must resolve a 1e-9 move of one mask entry"
            );
        }
    }

    /// End to end with ReLU, which is `t.max(0.0)`: exact in floating point and
    /// 1-Lipschitz, so the two paths' activations differ by at most the
    /// summed-input band, and that gap reaches the output through
    /// `|Ū| |M̄| |R̄|`. The second path's own rounding is over activations at
    /// most that gap larger, one more `γ` on the propagated term.
    #[test]
    fn relu_block_under_every_mask_kind_equals_the_edited_block_within_the_derived_band() {
        let (block, inputs) = random_block(GaussianActivation::Relu, 2953);
        let mut rng = StdRng::seed_from_u64(7);
        let read_in_shift =
            Array2::<f64>::zeros((ROWS, HIDDEN)) + &block.native.bias_in.mapv(f64::abs);
        let write_out_shift = inputs.mapv(f64::abs) + &block.native.bias_out.mapv(f64::abs);
        let write_out_masks = mask_family(&mut rng, WRITE_OUT_COMPONENTS);
        for ((kind, read_in_mask), (write_out_kind, write_out_mask)) in
            mask_family(&mut rng, READ_IN_COMPONENTS)
                .into_iter()
                .zip(write_out_masks)
        {
            let mask = MlpMask {
                read_in: ComponentMask::Components(read_in_mask.view()),
                write_out: ComponentMask::Components(write_out_mask.view()),
            };
            let rewritten = block.execute(inputs.view(), mask).expect("masked block");
            let reference = edited_block(&block, read_in_mask.view(), write_out_mask.view())
                .execute(inputs.view())
                .expect("edited block");
            let read_in_band = single_path_band(
                &block.read_in,
                read_in_mask.view(),
                inputs.view(),
                read_in_shift.view(),
                1,
            ) * 2.0;
            let summed_input = block
                .summed_input(inputs.view(), mask.read_in)
                .expect("masked summed input");
            let activations = block
                .native
                .activate(summed_input.view())
                .expect("masked activations");
            let own = single_path_band(
                &block.write_out,
                write_out_mask.view(),
                activations.view(),
                write_out_shift.view(),
                2,
            ) * 2.0;
            let propagated = magnitude(&block.write_out, write_out_mask.view(), read_in_band.view());
            let growth = accumulation_growth(WRITE_OUT_COMPONENTS + HIDDEN + 3);
            let band = own + propagated * ((1.0 + growth) / (1.0 - growth));
            assert_eq!(
                violations(&(&rewritten - &reference), &band),
                0,
                "{kind} read-in and {write_out_kind} write-out masks: the rewritten block left the derived band around the edited block"
            );

            // Positive control: the band resolves a 1e-9 move of one write-out mask entry.
            let mut moved = write_out_mask.clone();
            moved[2] += 1.0e-9;
            let perturbed = edited_block(&block, read_in_mask.view(), moved.view())
                .execute(inputs.view())
                .expect("perturbed block");
            assert!(
                violations(&(&rewritten - &perturbed), &band) > 0,
                "{kind} read-in and {write_out_kind} write-out masks: the band must resolve a 1e-9 mask move"
            );
        }
    }

    #[test]
    fn execute_hands_the_native_nonlinearity_the_masked_summed_input() {
        let (block, inputs) = random_block(GaussianActivation::ExactGelu, 2954);
        let mut rng = StdRng::seed_from_u64(8);
        let read_in_mask =
            Array1::from_shape_simple_fn(READ_IN_COMPONENTS, || rng.random_range(-1.0..1.0));
        let write_out_mask =
            Array1::from_shape_simple_fn(WRITE_OUT_COMPONENTS, || rng.random_range(-1.0..1.0));
        let mask = MlpMask {
            read_in: ComponentMask::Components(read_in_mask.view()),
            write_out: ComponentMask::Components(write_out_mask.view()),
        };
        let executed = block.execute(inputs.view(), mask).expect("masked block");
        let summed_input = block
            .summed_input(inputs.view(), mask.read_in)
            .expect("masked summed input");
        let activations = block
            .native
            .activate(summed_input.view())
            .expect("masked activations");
        let composed = block
            .write(inputs.view(), activations.view(), mask.write_out)
            .expect("masked write");
        assert!(
            bitwise_equal(&executed, &composed),
            "execute must be the native activation of the masked summed input, written under the write-out mask"
        );

        // Positive control: the same activations written all on are different bits.
        let unmasked = block
            .write(inputs.view(), activations.view(), ComponentMask::AllOn)
            .expect("all-on write");
        assert!(
            !bitwise_equal(&executed, &unmasked),
            "the composition check must see the write-out mask"
        );
    }

    #[test]
    fn both_factors_all_on_execute_the_original_tensors_bit_for_bit() {
        let (block, inputs) = random_block(GaussianActivation::ExactGelu, 2955);
        let native = block.native.execute(inputs.view()).expect("native block");
        let all_on = MlpMask {
            read_in: ComponentMask::AllOn,
            write_out: ComponentMask::AllOn,
        };
        let rewritten = block.execute(inputs.view(), all_on).expect("all-on block");
        assert!(
            bitwise_equal(&rewritten, &native),
            "the all-on block must execute the original tensors on their original path"
        );

        // Positive control: the factored path at the all-ones mask is algebraically
        // the same block but not the same bits, which is why all-on routes natively.
        let read_in_ones = Array1::ones(READ_IN_COMPONENTS);
        let write_out_ones = Array1::ones(WRITE_OUT_COMPONENTS);
        let factored = block
            .execute(
                inputs.view(),
                MlpMask {
                    read_in: ComponentMask::Components(read_in_ones.view()),
                    write_out: ComponentMask::Components(write_out_ones.view()),
                },
            )
            .expect("all-ones block");
        assert!(
            !bitwise_equal(&factored, &native),
            "the bit-identity check must distinguish the factored all-ones path from the native one"
        );
    }

    #[test]
    fn written_is_the_write_before_its_residual() {
        let (block, inputs) = random_block(GaussianActivation::ExactGelu, 2960);
        let mut rng = StdRng::seed_from_u64(9);
        let summed_input = block
            .native
            .summed_input(inputs.view())
            .expect("native summed input");
        let activations = block
            .native
            .activate(summed_input.view())
            .expect("native activations");
        let write_out_mask =
            Array1::from_shape_simple_fn(WRITE_OUT_COMPONENTS, || rng.random_range(-1.0..1.0));
        for mask in [
            ComponentMask::AllOn,
            ComponentMask::Components(write_out_mask.view()),
        ] {
            let written = block
                .written(activations.view(), mask)
                .expect("write without the residual");
            let write = block
                .write(inputs.view(), activations.view(), mask)
                .expect("write with the residual");
            assert!(
                bitwise_equal(&(&written + &inputs), &write),
                "the write must be the written block with the residual added last, under {mask:?}"
            );

            // Positive control: the residual is not zero, so the written block alone is not the write.
            assert!(
                !bitwise_equal(&written, &write),
                "the comparison must see the residual, under {mask:?}"
            );
        }
    }

    /// A hand-built factor in small integers, so every product and sum below is
    /// exact in floating point. The masked apply is linear in its inputs and in its
    /// mask, so its pullback must be the exact adjoint: `⟨ȳ, U M R e_{rj}⟩ = x̄_{rj}`
    /// for every input entry, and `⟨ȳ, U (e_c ⊙ R x)⟩ = m̄_c` for every component.
    #[test]
    fn masked_factor_pullback_is_the_exact_adjoint_of_the_masked_apply() {
        let factor = ExactFactor {
            write: array![[1.0, 2.0, -1.0], [0.0, 3.0, 1.0]],
            read: array![[2.0, 1.0], [-1.0, 1.0], [1.0, 3.0]],
            resolution_margin: f64::INFINITY,
        };
        let mask = array![2.0, -1.0, 3.0];
        let inputs = array![[1.0, -2.0], [3.0, 1.0]];
        let output_cotangent = array![[1.0, -1.0], [2.0, 1.0]];
        let pairing = |left: &Array2<f64>, right: &Array2<f64>| (left * right).sum();

        let mut adjoint_inputs = Array2::<f64>::zeros(inputs.raw_dim());
        for row in 0..inputs.nrows() {
            for column in 0..inputs.ncols() {
                let mut direction = Array2::<f64>::zeros(inputs.raw_dim());
                direction[[row, column]] = 1.0;
                let pushed = factor
                    .apply_masked(direction.view(), mask.view())
                    .expect("forward along an input direction");
                adjoint_inputs[[row, column]] = pairing(&output_cotangent, &pushed);
            }
        }
        let mut adjoint_mask = Array1::<f64>::zeros(factor.components());
        for component in 0..factor.components() {
            let mut indicator = Array1::<f64>::zeros(factor.components());
            indicator[component] = 1.0;
            let pushed = factor
                .apply_masked(inputs.view(), indicator.view())
                .expect("forward along one component");
            adjoint_mask[component] = pairing(&output_cotangent, &pushed);
        }

        let cotangents = factor
            .apply_masked_pullback(inputs.view(), mask.view(), output_cotangent.view())
            .expect("pullback shapes compose");
        assert_eq!(
            cotangents.inputs, adjoint_inputs,
            "the input cotangent must be the exact adjoint of the masked apply"
        );
        assert_eq!(
            cotangents.mask, adjoint_mask,
            "the mask cotangent must be the exact derivative of the pairing along each component"
        );

        // Positive control: pulled back under the all-ones mask, the input cotangent
        // is not the adjoint under `m`, so the check sees the mask.
        let ones = Array1::<f64>::ones(factor.components());
        let unmasked = factor
            .apply_masked_pullback(inputs.view(), ones.view(), output_cotangent.view())
            .expect("all-ones pullback");
        assert_ne!(
            unmasked.inputs, adjoint_inputs,
            "the adjoint check must distinguish the mask"
        );
    }

    /// A hand-built exact factor in small integers, so every operation is exact in
    /// floating point: `U R = [2 2]·[1 1]ᵀ = [4] = W₁`, `b₁ = −1`, `W₂ = [1]`,
    /// `b₂ = 0`, at the input `h = 1`.
    #[test]
    fn the_block_is_not_additive_over_components_while_its_summed_input_is() {
        let native = NativeMlp::new(
            array![[4.0]],
            array![-1.0],
            array![[1.0]],
            array![0.0],
            GaussianActivation::Relu,
        )
        .expect("scalar block");
        let block = ComponentMlp {
            native,
            read_in: ExactFactor {
                write: array![[2.0, 2.0]],
                read: array![[1.0], [1.0]],
                resolution_margin: f64::INFINITY,
            },
            write_out: ExactFactor {
                write: array![[1.0]],
                read: array![[1.0]],
                resolution_margin: f64::INFINITY,
            },
        };
        let inputs = array![[1.0]];
        let masks = [
            array![0.0, 0.0],
            array![1.0, 0.0],
            array![0.0, 1.0],
            array![1.0, 1.0],
        ];
        let summed_inputs: Vec<Array2<f64>> = masks
            .iter()
            .map(|mask| {
                block
                    .summed_input(inputs.view(), ComponentMask::Components(mask.view()))
                    .expect("summed input")
            })
            .collect();
        let outputs: Vec<Array2<f64>> = masks
            .iter()
            .map(|mask| {
                block
                    .execute(
                        inputs.view(),
                        MlpMask {
                            read_in: ComponentMask::Components(mask.view()),
                            write_out: ComponentMask::AllOn,
                        },
                    )
                    .expect("block")
            })
            .collect();
        assert_eq!(
            second_difference(&summed_inputs),
            array![[0.0]],
            "the summed input is affine in the mask, so its second difference vanishes"
        );
        assert_eq!(
            second_difference(&outputs),
            array![[1.0]],
            "the block is not additive over components: F(P1 + P2) != F(P1) + F(P2)"
        );
    }

    /// `f(1, 1) − f(1, 0) − f(0, 1) + f(0, 0)`.
    fn second_difference(values: &[Array2<f64>]) -> Array2<f64> {
        &(&(&values[3] - &values[1]) - &values[2]) + &values[0]
    }

    /// `σ_min(R)` floored by twice the refusal band, and `σ_max(R)`. The floor
    /// covers the SVD of `R` read here and the perturbation of `T` in the solve.
    fn read_spectrum(read: &Array2<f64>) -> (f64, f64) {
        let (components, width) = read.dim();
        let (_, singular_values, _) = read.svd(false, false).expect("read singular values");
        let largest = singular_values.iter().copied().fold(0.0_f64, f64::max);
        let smallest = singular_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min);
        let band = factor_singular_band(components, width, largest)
            + factor_singular_band(width, width, largest);
        (smallest - 2.0 * band, largest)
    }

    /// First-order bound on `‖W − U R‖_F` for the exact product and the computed
    /// write. With `X T̂ = Ê = fl(W − N R)` solved with backward error `δT` and
    /// `Q̂ T̂ = R + δR`, `W − U R = (E − Ê) + X (δT + Q̂ᵀ δR) − e R`, where `e` is
    /// the rounding of `N + X Q̂ᵀ`. `‖X‖_F ≤ ‖Ê‖_F / σ_min`, `‖δR‖₂` is the QR
    /// band, `‖δT‖₂ ≤ γ_d ‖T‖_F ≤ γ ‖R‖_F`, `‖e‖_F ≤ γ (‖N‖_F + √d ‖X‖_F)` and
    /// `‖E − Ê‖_F ≤ γ (‖|W|‖_F + ‖|N| |R|‖_F)`, all with `γ = γ_{C+d+2}`.
    fn exact_write_residual_bound(
        weight: &Array2<f64>,
        read: &Array2<f64>,
        candidate: &Array2<f64>,
    ) -> f64 {
        let (components, width) = read.dim();
        let (floor, largest) = read_spectrum(read);
        let growth = accumulation_growth(components + width + 2);
        let solved = frobenius(&(weight - &candidate.dot(read))) / floor;
        solved
            * (factor_singular_band(components, width, largest)
                + growth * (frobenius(read) + (width as f64).sqrt() * largest))
            + growth
                * (frobenius(candidate) * largest
                    + frobenius(&weight.mapv(f64::abs))
                    + frobenius(&candidate.mapv(f64::abs).dot(&read.mapv(f64::abs))))
    }

    #[test]
    fn the_exact_write_reproduces_the_weight_from_any_candidate_and_the_candidate_shows_under_masks()
    {
        let mut rng = StdRng::seed_from_u64(2956);
        let weight = uniform(&mut rng, HIDDEN, WIDTH, 1.0);
        let read = uniform(&mut rng, READ_IN_COMPONENTS, WIDTH, 1.0);
        let inputs = uniform(&mut rng, ROWS, WIDTH, 2.0);
        let zero_candidate = Array2::<f64>::zeros((HIDDEN, READ_IN_COMPONENTS));
        let drawn_candidate = uniform(&mut rng, HIDDEN, READ_IN_COMPONENTS, 1.0);
        let growth = accumulation_growth(READ_IN_COMPONENTS + WIDTH + 2);
        let mut factors = Vec::new();
        let mut exact_bounds = Vec::new();
        for candidate in [&zero_candidate, &drawn_candidate] {
            let factor = ExactFactor::solve_write(weight.view(), read.view(), candidate.view())
                .expect("a random overcomplete read is resolved");
            let exact = exact_write_residual_bound(&weight, &read, candidate);
            let product_rounding = growth
                * (frobenius(&weight.mapv(f64::abs))
                    + frobenius(&factor.write().mapv(f64::abs).dot(&read.mapv(f64::abs))));
            let bound = exact + product_rounding;
            let residual = frobenius(&(&weight - &factor.write().dot(&read)));
            assert!(
                residual <= bound,
                "the exact write must reproduce the weight: residual {residual:e} over the derived bound {bound:e}"
            );

            // Positive control: a write moved off the exact set by 1e-9 per entry leaves the bound.
            let moved = &factor.write() + &uniform(&mut rng, HIDDEN, READ_IN_COMPONENTS, 1.0e-9);
            let moved_residual = frobenius(&(&weight - &moved.dot(&read)));
            assert!(
                moved_residual > bound,
                "the bound must resolve a 1e-9 move off the exact set: {moved_residual:e} within {bound:e}"
            );
            factors.push(factor);
            exact_bounds.push(exact);
        }

        let zero_shift = Array2::<f64>::zeros((ROWS, HIDDEN));
        let computation_band = |mask: &Array1<f64>| {
            single_path_band(&factors[0], mask.view(), inputs.view(), zero_shift.view(), 1)
                + single_path_band(&factors[1], mask.view(), inputs.view(), zero_shift.view(), 1)
        };

        // Control: at the all-ones mask both writes act as the weight.
        let ones = Array1::ones(READ_IN_COMPONENTS);
        let row_norms = inputs
            .map_axis(Axis(1), |row| row.dot(&row).sqrt())
            .insert_axis(Axis(1));
        let ones_band =
            computation_band(&ones) + &(row_norms * (exact_bounds[0] + exact_bounds[1]));
        let ones_gap = &factors[0]
            .apply_masked(inputs.view(), ones.view())
            .expect("all-ones write of the zero candidate")
            - &factors[1]
                .apply_masked(inputs.view(), ones.view())
                .expect("all-ones write of the drawn candidate");
        assert_eq!(
            violations(&ones_gap, &ones_band),
            0,
            "two exact writes of one weight must agree at the all-ones mask"
        );

        // Under a partial mask the candidate is a different intervention.
        let partial = Array1::from_shape_simple_fn(READ_IN_COMPONENTS, || rng.random_range(0.0..1.0));
        let partial_gap = &factors[0]
            .apply_masked(inputs.view(), partial.view())
            .expect("partial-mask write of the zero candidate")
            - &factors[1]
                .apply_masked(inputs.view(), partial.view())
                .expect("partial-mask write of the drawn candidate");
        assert!(
            violations(&partial_gap, &computation_band(&partial)) > 0,
            "two exact writes of one weight must differ under a partial mask"
        );
    }

    #[test]
    fn solving_from_an_exact_write_returns_that_write() {
        let mut rng = StdRng::seed_from_u64(2957);
        let weight = uniform(&mut rng, HIDDEN, WIDTH, 1.0);
        let read = uniform(&mut rng, READ_IN_COMPONENTS, WIDTH, 1.0);
        let drawn_candidate = uniform(&mut rng, HIDDEN, READ_IN_COMPONENTS, 1.0);
        let exact_write = ExactFactor::solve_write(weight.view(), read.view(), drawn_candidate.view())
            .expect("a random overcomplete read is resolved")
            .write()
            .to_owned();
        let again = ExactFactor::solve_write(weight.view(), read.view(), exact_write.view())
            .expect("the same read is resolved again");

        // `U′ − U = X Q̂ᵀ + e` with `‖X‖_F ≤ ‖fl(W − U R)‖_F / σ_min` and the
        // rounding `e` of `U + X Q̂ᵀ`; the comparison adds one subtraction.
        let (floor, largest) = read_spectrum(&read);
        let growth = accumulation_growth(READ_IN_COMPONENTS + WIDTH + 2);
        let solved = frobenius(&(&weight - &exact_write.dot(&read))) / floor;
        let bound = solved * (1.0 + growth * (WIDTH as f64).sqrt())
            + growth * (2.0 * frobenius(&exact_write) + frobenius(&again.write().to_owned()));
        let moved = frobenius(&(&again.write() - &exact_write));
        assert!(
            moved <= bound,
            "an exact candidate must be its own nearest exact write: moved {moved:e} over {bound:e} (largest singular value {largest:e})"
        );

        // Positive control: from the zero candidate the exact write is a different one.
        let zero_candidate = Array2::<f64>::zeros((HIDDEN, READ_IN_COMPONENTS));
        let minimum_norm = ExactFactor::solve_write(weight.view(), read.view(), zero_candidate.view())
            .expect("the same read is resolved from zero");
        let distance = frobenius(&(&minimum_norm.write() - &exact_write));
        assert!(
            distance > bound,
            "the idempotence bound must resolve a different candidate: {distance:e} within {bound:e}"
        );
    }

    #[test]
    fn a_read_with_fewer_components_than_input_directions_is_refused_as_uncovered() {
        let mut rng = StdRng::seed_from_u64(2958);
        let weight = uniform(&mut rng, HIDDEN, WIDTH, 1.0);
        let narrow = uniform(&mut rng, WIDTH - 1, WIDTH, 1.0);
        let refusal = ExactFactor::solve_write(
            weight.view(),
            narrow.view(),
            Array2::<f64>::zeros((HIDDEN, WIDTH - 1)).view(),
        );
        assert!(
            matches!(
                refusal,
                Err(FactorRefusal::Uncovered { components, input_width })
                    if components == WIDTH - 1 && input_width == WIDTH
            ),
            "a read with fewer components than input directions must be refused as uncovered, got {refusal:?}"
        );

        // Positive control: a square read covers the input.
        let square = uniform(&mut rng, WIDTH, WIDTH, 1.0);
        let accepted = ExactFactor::solve_write(
            weight.view(),
            square.view(),
            Array2::<f64>::zeros((HIDDEN, WIDTH)).view(),
        );
        assert!(accepted.is_ok(), "a random square read must be accepted, got {accepted:?}");
    }

    #[test]
    fn a_numerically_rank_deficient_read_is_refused_and_a_resolved_one_is_not() {
        let mut rng = StdRng::seed_from_u64(2959);
        let weight = uniform(&mut rng, HIDDEN, WIDTH, 1.0);
        let candidate = Array2::<f64>::zeros((HIDDEN, READ_IN_COMPONENTS));
        let mut read = uniform(&mut rng, READ_IN_COMPONENTS, WIDTH, 1.0);
        let repeated = read.column(0).to_owned();
        read.column_mut(WIDTH - 1).assign(&repeated);
        let refusal = ExactFactor::solve_write(weight.view(), read.view(), candidate.view());
        assert!(
            matches!(
                refusal,
                Err(FactorRefusal::Unresolved { smallest_singular_value, band })
                    if smallest_singular_value <= band
            ),
            "a read with a repeated column must be refused as unresolved, got {refusal:?}"
        );

        // Positive control: separating the repeated column by 1e-6 resolves the read.
        let offset = uniform_vector(&mut rng, READ_IN_COMPONENTS, 1.0e-6);
        read.column_mut(WIDTH - 1).assign(&(&repeated + &offset));
        let accepted = ExactFactor::solve_write(weight.view(), read.view(), candidate.view())
            .expect("a column separated by 1e-6 is resolved");
        assert!(
            accepted.resolution_margin() > 1.0,
            "an accepted factor sits above the refusal band"
        );
    }
}
