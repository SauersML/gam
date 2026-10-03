//! Native primitives that are not linear in their input: the SwiGLU gate and the
//! source's normalizations (#2951).
//!
//! Each one is applied to the SUMMED masked vector, never per component: the
//! native nonlinearity sees the same summed input under every mask, so executing
//! it after a masked read equals direct execution with the edited tensors for every
//! real mask (continuous, binary and signed). Inputs are rows of observations. The
//! derivation is on #2951 (mpd-gated, R2-R4); the native forwards were read in the
//! installed `transformers` source (`Qwen3MLP`, `Qwen3RMSNorm`, `GPTNeoXLayer`).
//!
//! * **SwiGLU (R2).** `F_m(x) = W_d(m_d) [s(W_g(m_g) x) ⊙ W_u(m_u) x]` with the
//!   SiLU gate `s(t) = t sigma(t)` (Qwen3 `hidden_act = "silu"`)
//!   ([`swiglu_hidden`]). The gate is read from its one owner,
//!   [`gam_math::gaussian_gated::silu_derivatives`]. At fixed `x` the block is
//!   affine in each of `m_u` and `m_d` (multilinear, degree one per control) and
//!   nonlinear in `m_g`.
//! * **Norms (R3, R4).** `Qwen3RMSNorm` is `w ⊙ h (mean(h^2) + eps)^(-1/2)` and
//!   `torch.nn.LayerNorm` is `w ⊙ (P h)(mean((P h)^2) + eps)^(-1/2) + beta` with
//!   the centring `P = I - 1 1^T / d` ([`MaskedNorm`], [`rms_normalizers`]). The
//!   gain and bias are masked tensors that enter linearly; the normalizer is
//!   computed on each whole current residual row. `N_eps(c h) = sign(c)
//!   N_(eps/c^2)(h)`, so a uniform scale of the stream reaches the next RMSNorm
//!   only through `eps`, and a LayerNorm annihilates every write along `1`.

use gam_math::gaussian_gated::silu_derivatives;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, Zip};
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

/// The epsilon at which `h` normalizes as `c h` does at `epsilon`:
/// `N_eps(c h) = sign(c) N_(eps/c^2)(h)` (module note).
pub fn rms_epsilon_under_input_scale(epsilon: f64, scale: f64) -> f64 {
    epsilon / (scale * scale)
}

/// Whether every positive scale of an RMSNorm's input is an exact symmetry of
/// the norm at `epsilon`: [`rms_epsilon_under_input_scale`] keeps `epsilon` for
/// every `c > 0` only when `epsilon = 0`. With `epsilon > 0` a scale of the
/// input is a change of `epsilon`, never a gauge.
pub fn rms_input_scale_is_symmetry(epsilon: f64) -> bool {
    [2.0, 0.5].into_iter().all(|scale| rms_epsilon_under_input_scale(epsilon, scale) == epsilon)
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

#[cfg(test)]
mod tests {
    use super::{GatedRewriteError, MaskedNorm, rms_epsilon_under_input_scale, rms_input_scale_is_symmetry, rms_normalizers, swiglu_hidden};
    use gam_math::gaussian_gated::silu_derivatives;
    use ndarray::{Array, Array1, Array2, ArrayView2, Dimension, Zip, array};
    use qd::Quad;
    use rand::rngs::StdRng;
    use gam_linalg::roundoff::{UNIT_ROUNDOFF, accumulation_growth};
    use rand::{RngExt, SeedableRng};

    /// A Lipschitz constant of the SiLU gate: `s' = sigma + t sigma (1 - sigma)`
    /// and `sigma (1 - sigma) = e^-|t| / (1 + e^-|t|)^2 <= e^-|t|`, so
    /// `|s'| <= 1 + sup_t t e^-t = 1 + 1/e < 1 + 3/8`, because `e > 8/3`.
    const SILU_LIPSCHITZ: f64 = 1.375;

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
                epsilon: rms_epsilon_under_input_scale(epsilon, scale),
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
                rms_normalizers(residual.view(), rms_epsilon_under_input_scale(epsilon, scale))
                    .expect("finite rows")
                    .mapv(|value| value / scale.abs()),
                "nu_eps({scale} h) must equal nu_(eps/{scale}^2)(h) / |{scale}| bit for bit"
            );
        }
        assert!(rms_input_scale_is_symmetry(0.0));
        assert!(!rms_input_scale_is_symmetry(epsilon), "with eps > 0 an input scale moves eps, so it is no gauge");
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
    }
}
