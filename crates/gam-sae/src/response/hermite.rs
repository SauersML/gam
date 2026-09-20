//! Hermite coefficients of a known block's response under the declared Gaussian law (#2946 R8).
//!
//! # Coefficients
//!
//! Under `Y ~ N(0, I_k)` the normalized probabilists' Hermite polynomials
//! `h_α(y) = Π_i He_{α_i}(y_i)/√(α_i!)` are orthonormal. A square-integrable response therefore
//! expands as `f = Σ_α c_α h_α` with `c_α = E[f(Y) h_α(Y)]`, its variance in the output metric `M`
//! splits as `Σ_{α≠0} c_αᵀ M c_α`, and grouping `α` by its support `S = {i : α_i > 0}` gives the
//! Hoeffding term energies `‖f_S‖²_M`.
//!
//! One unit `σ(b + wᵀz)` expands by Gaussian integration by parts, `E[g(Z) He_α(Z)] = E[∂^α g(Z)]`:
//! `c_α = w^α/√(α!) · μ_n` with `n = |α|` and `μ_n = E σ⁽ⁿ⁾(b + ‖w‖E) = ∂_tⁿ T_{‖w‖²}σ(b)`. The
//! retained response `F̄_P` of R1 in frame coordinates `y = Qᵀz` has readers `w̃ = Qᵀw` and
//! activation `T_{v⊥}σ`. Because `T_{v⊥} ∘ T_{‖w̃‖²} = T_{‖w‖²}`, its coefficients use the same
//! `μ_n` at the FULL reader norm. The block's output bias `c` enters `c_0` alone, and no energy.
//!
//! # Scaled columns
//!
//! The raw `μ_n` grows like `√(n!)/‖w‖ⁿ`, so each unit is carried as the scaled column
//! `a_n = ‖w‖ⁿ μ_n/√(n!) = E[σ(b + ‖w‖E) h_n(E)]` together with the direction `ũ = w̃/‖w‖`, where
//! `‖ũ‖ ≤ 1`. Then `c_α = Σ_j u_j a_{j,n} √(n!/α!) ũ_j^α`. Every factor stays bounded, because
//! `Σ_{|α|=n} (n!/α!) ũ^{2α} = ‖ũ‖^{2n} ≤ 1`. The closed-form columns and their per-coefficient
//! rounding bounds come from the smoothing owner, `gaussian_hermite_coefficients`.
//!
//! # Shells and truncation
//!
//! By the multinomial theorem the order-`n` shell is
//! `Σ_{|α|=n} ‖c_α‖²_M = Σ_jk D_jk a_{j,n} a_{k,n} (ũ_jᵀũ_k)ⁿ`, with `D_jk = u_jᵀ M u_k`. Mehler's
//! formula sums the shells `n ≥ 1` to R4's `V(P)`. So the energy an order-`N` proposal leaves out,
//! `V(P) − Σ_{n≤N} shell_n`, is exact and costs `O(h²N)` with no enumeration. The order is chosen
//! from a declared fidelity budget.
//!
//! # Bands
//!
//! Every coefficient and energy carries a derived bound on its absolute error: the scaled columns'
//! running error bounds carried through, plus `γ_k` times the absolute shadow of the accumulation
//! (Higham, *ASNA* §3.1), to first order in `u`. Directions enter through a shadow `S_j ≥ |ũ_j|`
//! with relative rounding `ε`, so an `n`-fold product of them moves by at most `nε` of its shadow.

use super::subspace::{
    BandedEnergy, KnownBlock, ResponseError, require_finite, require_length, signed_sum,
};
use gam_linalg::faer_ndarray::fast_ab;
use gam_linalg::roundoff::accumulation_growth;
use gam_math::gaussian_activation::{
    GaussianActivation, GaussianActivationError, PreactivationPair, gaussian_hermite_coefficients,
    pair_kernel,
};
use gam_math::probability::normal_pdf;
use gam_math::quadrature::{QuadratureError, gauss_hermite_rule};
use gam_runtime::resource::{Governed, MemoryGovernor, MemoryReservation, MemoryReservationError};
use ndarray::linalg::general_mat_vec_mul;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::fmt;

/// Cramér's inequality (Abramowitz & Stegun 22.14.17), `|He_n(x)| ≤ K e^{x²/4} √(n!)` with
/// `K ≈ 1.086435`, rounded up. Every entry of [`hermite_functions`] is bounded by `K (2π)^{-1/4}`.
const CRAMER_HERMITE_BOUND: f64 = 1.086_436;

/// Refusals of the Hermite expansion.
#[derive(Debug)]
pub enum HermiteError {
    /// A unit's bias or reader norm is not finite, or the norm is negative.
    InvalidUnit { bias: f64, scale: f64 },
    /// The smoothing owner refused a unit.
    Kernel(GaussianActivationError),
    /// The block refused a frame or a shape.
    Response(ResponseError),
    /// Moment columns that do not match the readers, or are not finite.
    InvalidMoments { reason: String },
    /// A retained reader longer than the full reader norm it is scaled by.
    RetainedNormExceedsFull {
        unit: usize,
        retained: f64,
        full: f64,
    },
    /// The columns came from moments, which carry no order to change and no tail envelope.
    NoClosedForm,
    /// An activation without closed-form Hermite coefficients, whose columns come from moments.
    NoClosedFormActivation { activation: GaussianActivation },
    /// A unit's energy less its captured part lies below zero by more than its band, so the column
    /// and the kernel's diagonal moment disagree.
    InconsistentTail {
        /// `"value"` or `"slope"`.
        tail: &'static str,
        order: usize,
        remaining: f64,
        band: f64,
    },
    /// A degree range outside the expansion's order.
    InvalidDegreeRange {
        first_degree: usize,
        last_degree: usize,
        order: usize,
    },
    /// The declared unexplained fraction is outside `(0, 1)`.
    InvalidBudget { unexplained_fraction: f64 },
    /// The retained variance handed in is not finite, has a negative band, or lies below zero by
    /// more than its band.
    InvalidTotalEnergy { total_energy: BandedEnergy },
    /// No order representable in `usize` brings the a-priori envelope within budget.
    EnvelopeOrderUnrepresentable { allowed_energy: f64 },
    /// At the envelope order the exact tail is within budget, but the computed tail lies above it
    /// by more than its band, so the handed-in `V(P)` and the shells disagree.
    TailAboveEnvelope {
        order: usize,
        computed_tail: BandedEnergy,
        allowed_energy: f64,
    },
    /// The `C(k + N, N) − 1` coefficients of a materialized proposal are not addressable.
    TooManyCoefficients {
        dimension: usize,
        order: usize,
        outputs: usize,
    },
    /// The process memory governor refused a materialization's live footprint.
    Memory(MemoryReservationError),
    /// A support that is not ascending, distinct and within the retained coordinates.
    InvalidSupport {
        support: Vec<usize>,
        retained_dim: usize,
    },
    /// The tensor grid `node_count^dimension × output_dimension` is not addressable.
    GridUnrepresentable {
        node_count: usize,
        dimension: usize,
        output_dimension: usize,
    },
    /// The Gauss-Hermite rule could not be built.
    Quadrature(QuadratureError),
}

impl fmt::Display for HermiteError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidUnit { bias, scale } => write!(
                formatter,
                "unit bias {bias} and reader norm {scale} must be finite, with a nonnegative norm"
            ),
            Self::Kernel(error) => write!(formatter, "Gaussian smoothing: {error}"),
            Self::Response(error) => write!(formatter, "{error}"),
            Self::InvalidMoments { reason } => write!(formatter, "Hermite moments: {reason}"),
            Self::RetainedNormExceedsFull {
                unit,
                retained,
                full,
            } => write!(
                formatter,
                "unit {unit}: retained reader norm {retained} exceeds the full reader norm {full}"
            ),
            Self::NoClosedForm => write!(
                formatter,
                "columns built from moments carry no closed form to reorder or bound"
            ),
            Self::NoClosedFormActivation { activation } => write!(
                formatter,
                "{activation:?} has no closed-form Hermite coefficients; build its columns from moments"
            ),
            Self::InconsistentTail {
                tail,
                order,
                remaining,
                band,
            } => write!(
                formatter,
                "the {tail} tail past order {order} is {remaining}, below zero by more than its band {band}"
            ),
            Self::InvalidDegreeRange {
                first_degree,
                last_degree,
                order,
            } => write!(
                formatter,
                "degree range {first_degree}..={last_degree} must lie within 1..={order}"
            ),
            Self::InvalidBudget {
                unexplained_fraction,
            } => write!(
                formatter,
                "the declared unexplained fraction {unexplained_fraction} must lie in (0, 1)"
            ),
            Self::InvalidTotalEnergy { total_energy } => write!(
                formatter,
                "the retained variance {total_energy:?} must be finite, banded and nonnegative"
            ),
            Self::EnvelopeOrderUnrepresentable { allowed_energy } => write!(
                formatter,
                "no representable order brings the Cramér tail envelope within {allowed_energy}"
            ),
            Self::TailAboveEnvelope {
                order,
                computed_tail,
                allowed_energy,
            } => write!(
                formatter,
                "at the envelope order {order} the computed tail {computed_tail:?} exceeds the \
                 allowed {allowed_energy}, so V(P) and the shells disagree"
            ),
            Self::TooManyCoefficients {
                dimension,
                order,
                outputs,
            } => write!(
                formatter,
                "a proposal over {dimension} coordinates to order {order} with {outputs} outputs \
                 is not addressable"
            ),
            Self::Memory(error) => write!(formatter, "Hermite materialization: {error}"),
            Self::InvalidSupport {
                support,
                retained_dim,
            } => write!(
                formatter,
                "support {support:?} must be ascending, distinct and within 0..{retained_dim}"
            ),
            Self::GridUnrepresentable {
                node_count,
                dimension,
                output_dimension,
            } => write!(
                formatter,
                "a {node_count}^{dimension} grid with {output_dimension} outputs is not addressable"
            ),
            Self::Quadrature(error) => write!(formatter, "Gauss-Hermite rule: {error}"),
        }
    }
}

impl std::error::Error for HermiteError {}

/// Fills `out[k] = start · h_k(x)` by `h_{k+1} = (x h_k − √k h_{k−1})/√(k+1)`.
fn scaled_hermite_recurrence(x: f64, start: f64, out: &mut [f64]) {
    let Some(first) = out.first_mut() else {
        return;
    };
    *first = start;
    if out.len() > 1 {
        out[1] = x * start;
    }
    for rank in 1..out.len().saturating_sub(1) {
        let index = rank as f64;
        out[rank + 1] = (x * out[rank] - index.sqrt() * out[rank - 1]) / (index + 1.0).sqrt();
    }
}

/// Normalized probabilists' Hermite polynomials, `out[k] = h_k(x) = He_k(x)/√(k!)`.
pub fn normalized_hermite_polynomials(x: f64, out: &mut [f64]) {
    scaled_hermite_recurrence(x, 1.0, out);
}

/// Normalized Hermite functions, `out[k] = ψ_k(x) = √φ(x) h_k(x)`.
///
/// Cramér's inequality bounds every entry by `K (2π)^{-1/4} < 0.69`. So the recurrence neither
/// overflows nor loses a saturated unit's coefficients to an underflowed `φ(x)`: `φ(x)` alone
/// underflows near `|x| = 38.6`, while `√φ(x)` lasts to `|x| = 54.6`.
pub fn hermite_functions(x: f64, out: &mut [f64]) {
    let root_density = (-0.25 * x * x).exp() * (2.0 * std::f64::consts::PI).powf(-0.25);
    scaled_hermite_recurrence(x, root_density, out);
}

fn integer_power(base: f64, exponent: usize) -> f64 {
    let mut result = 1.0;
    let mut square = base;
    let mut remaining = exponent;
    while remaining > 0 {
        if remaining & 1 == 1 {
            result *= square;
        }
        square *= square;
        remaining >>= 1;
    }
    result
}

/// An upper bound on `Σ_{n>N} a_n(b, s)²` for `N ≥ 1`, from `|ψ_k| ≤ K (2π)^{-1/4}`.
///
/// ReLU: `a_n² ≤ s² φ(β) K²/(√(2π) n(n−1))`, and `Σ_{n>N} 1/(n(n−1)) = 1/N`.
/// Exact GELU: for `n > N`, `n(n−1) ≥ N(N+1)`, so
/// `a_n² ≤ ρ^{2n} φ(x) K² (1 + A/√(N(N+1)))²/(A √(2π))`, and `Σ_{n>N} ρ^{2n} = A ρ^{2N+2}`.
/// At `N = 0` no bound is claimed.
pub fn unit_tail_envelope(
    activation: GaussianActivation,
    bias: f64,
    scale: f64,
    order: usize,
) -> Result<f64, HermiteError> {
    if !(bias.is_finite() && scale.is_finite() && scale >= 0.0) {
        return Err(HermiteError::InvalidUnit { bias, scale });
    }
    if order == 0 {
        return Ok(f64::INFINITY);
    }
    if scale == 0.0 {
        return Ok(0.0);
    }
    let cramer = CRAMER_HERMITE_BOUND * CRAMER_HERMITE_BOUND / (2.0 * std::f64::consts::PI).sqrt();
    let rank = order as f64;
    Ok(match activation {
        GaussianActivation::Relu => scale * scale * normal_pdf(bias / scale) * cramer / rank,
        GaussianActivation::ExactGelu => {
            let smoothed_variance = 1.0 + scale * scale;
            let standardized = bias / smoothed_variance.sqrt();
            let lead = 1.0 + smoothed_variance / (rank * (rank + 1.0)).sqrt();
            normal_pdf(standardized)
                * cramer
                * lead
                * lead
                * integer_power(scale * scale / smoothed_variance, order + 1)
        }
        GaussianActivation::Silu => {
            return Err(HermiteError::NoClosedFormActivation { activation });
        }
    })
}

/// An upper bound on the slope tail `Σ_{n>N} n·a_n(b, s)²` for `N ≥ 1`, whose sum over all `n ≥ 1`
/// is `s² E σ′(b + sE)²`.
///
/// Exact GELU: for `n > N`, `n a_n² ≤ n ρ^{2n} φ(x) K² (1 + A/√(N(N+1)))²/(A √(2π))`, and with
/// `q = ρ²`, `1 − q = 1/A`, `Σ_{n>N} n qⁿ = q^{N+1}((N + 1) − N q)/(1 − q)²`. So the tail is at most
/// `φ(x) K² (1 + A/√(N(N+1)))² A ρ^{2N+2} ((N + 1) − N ρ²)/√(2π)`.
///
/// ReLU: Cramér's inequality gives `n a_n² ≤ s² φ(β) K²/(√(2π)(n − 1))`, which sums harmonically, so
/// no a-priori slope envelope follows. The slope tail comes only from the exact subtraction
/// `s² Φ(β) − Σ_{n≤N} n a_n²`, and the envelope is `+∞`, the neutral element of that minimum.
/// At `N = 0` no bound is claimed.
pub fn unit_slope_tail_envelope(
    activation: GaussianActivation,
    bias: f64,
    scale: f64,
    order: usize,
) -> Result<f64, HermiteError> {
    if !(bias.is_finite() && scale.is_finite() && scale >= 0.0) {
        return Err(HermiteError::InvalidUnit { bias, scale });
    }
    if scale == 0.0 {
        return Ok(0.0);
    }
    if order == 0 {
        return Ok(f64::INFINITY);
    }
    Ok(match activation {
        GaussianActivation::Relu => f64::INFINITY,
        GaussianActivation::ExactGelu => {
            let cramer =
                CRAMER_HERMITE_BOUND * CRAMER_HERMITE_BOUND / (2.0 * std::f64::consts::PI).sqrt();
            let rank = order as f64;
            let smoothed_variance = 1.0 + scale * scale;
            let standardized = bias / smoothed_variance.sqrt();
            let contraction_square = scale * scale / smoothed_variance;
            let lead = 1.0 + smoothed_variance / (rank * (rank + 1.0)).sqrt();
            normal_pdf(standardized)
                * cramer
                * lead
                * lead
                * smoothed_variance
                * integer_power(contraction_square, order + 1)
                * ((rank + 1.0) - rank * contraction_square)
        }
        GaussianActivation::Silu => {
            return Err(HermiteError::NoClosedFormActivation { activation });
        }
    })
}

/// Upper bounds `t(n) ≥ √(Σ_{m>n} a_m²)` and `d(n) ≥ √(Σ_{m>n} m a_m²)` for `n = 0..=N`, from a
/// unit's banded scaled column `a_0..=a_N` and its envelopes.
///
/// Value tails take the smaller of the envelope at `n` and, when `n < N`, the next coefficient's
/// magnitude plus its band, squared, added to the envelope at `n + 1`. At `n = 0` no envelope is
/// claimed, so only the second form applies there.
///
/// Slope tails take the same two forms over [`unit_slope_tail_envelope`], with weight `n + 1` on the
/// next coefficient.
///
/// Both tails then take the exact subtraction from the kernel's diagonal moments and their rounding
/// bounds: `Σ_{m≥1} a_m² = E σ(X)² − a_0² = K(b, b; s², s², s²) − a_0²` and
/// `Σ_{m≥1} m a_m² = s² E σ′(X)² = s² ∂_r K(b, b; s², s², s²)`. For ReLU this subtraction is the only slope
/// bound, because Cramér's slope envelope sums harmonically.
fn unit_tails(
    activation: GaussianActivation,
    bias: f64,
    scale: f64,
    values: &[f64],
    bands: &[f64],
) -> Result<(Vec<f64>, Vec<f64>), HermiteError> {
    let order = values.len().saturating_sub(1);
    let mut value_tails = vec![0.0; order + 1];
    let mut slope_tails = vec![0.0; order + 1];
    if scale == 0.0 || values.is_empty() {
        return Ok((value_tails, slope_tails));
    }
    for degree in 0..=order {
        let mut value_bound = unit_tail_envelope(activation, bias, scale, degree)?;
        let mut slope_bound = unit_slope_tail_envelope(activation, bias, scale, degree)?;
        if degree < order {
            let next = values[degree + 1].abs() + bands[degree + 1];
            let next_square = next * next;
            value_bound = value_bound
                .min(next_square + unit_tail_envelope(activation, bias, scale, degree + 1)?);
            slope_bound = slope_bound.min(
                (degree + 1) as f64 * next_square
                    + unit_slope_tail_envelope(activation, bias, scale, degree + 1)?,
            );
        }
        value_tails[degree] = value_bound.sqrt();
        slope_tails[degree] = slope_bound.sqrt();
    }
    let variance = scale * scale;
    let diagonal = pair_kernel(
        activation,
        PreactivationPair {
            mean_x: bias,
            mean_y: bias,
            variance_x: variance,
            variance_y: variance,
            covariance: variance,
            covariance_rounding: 0.0,
        },
    )
    .map_err(HermiteError::Kernel)?;
    let mean = values[0];
    let mean_band = bands[0];
    let value_total = diagonal.value - mean * mean;
    let value_total_band = diagonal.value_rounding
        + 2.0 * mean.abs() * mean_band
        + mean_band * mean_band
        + accumulation_growth(2) * (diagonal.value.abs() + mean * mean);
    let slope_total = variance * diagonal.covariance_derivative;
    let slope_total_band = variance * diagonal.covariance_derivative_rounding
        + accumulation_growth(1) * slope_total.abs();
    subtract_captured(
        "value",
        value_total,
        value_total_band,
        false,
        values,
        bands,
        &mut value_tails,
    )?;
    subtract_captured(
        "slope",
        slope_total,
        slope_total_band,
        true,
        values,
        bands,
        &mut slope_tails,
    )?;
    Ok((value_tails, slope_tails))
}

/// Tightens each `tails[n]` to `√(remaining + band)`, where `remaining = total − Σ_{1≤m≤n} w_m a_m²`
/// with `w_m = m` when `weighted` and `1` otherwise. The band is the total's band and the column bands
/// carried through, plus `γ_{n+2}` of the absolute sum. A remainder below zero by more than its band
/// is refused.
fn subtract_captured(
    tail: &'static str,
    total: f64,
    total_band: f64,
    weighted: bool,
    values: &[f64],
    bands: &[f64],
    tails: &mut [f64],
) -> Result<(), HermiteError> {
    let mut captured = 0.0;
    let mut captured_band = 0.0;
    for degree in 0..values.len() {
        if degree >= 1 {
            let weight = if weighted { degree as f64 } else { 1.0 };
            let value = values[degree];
            let band = bands[degree];
            captured += weight * value * value;
            captured_band += weight * (2.0 * value.abs() * band + band * band);
        }
        let remaining = total - captured;
        let band = total_band
            + captured_band
            + accumulation_growth(degree + 2) * (total.abs() + captured);
        if remaining < -band {
            return Err(HermiteError::InconsistentTail {
                tail,
                order: degree,
                remaining,
                band,
            });
        }
        tails[degree] = tails[degree].min((remaining + band).sqrt());
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ClosedFormUnits {
    activation: GaussianActivation,
    biases: Array1<f64>,
}

/// Per-unit scaled columns `a_{j,n}` and directions `ũ_j` of a block read through a retained frame.
#[derive(Clone, Debug)]
pub struct HermiteColumns {
    /// `‖w_j‖`.
    scales: Array1<f64>,
    /// `h × k`: `ũ_j = w̃_j/‖w_j‖`, zero for a constant unit.
    directions: Array2<f64>,
    /// `h × k`: a shadow `S_j ≥ |ũ_j|`.
    direction_shadow: Array2<f64>,
    /// The relative rounding `ε` of every direction against its shadow.
    direction_rounding: f64,
    /// `h × (N + 1)`: `a_{j,n}`.
    coefficients: Array2<f64>,
    /// `h × (N + 1)`: bounds on each column entry's absolute error.
    coefficient_band: Array2<f64>,
    /// `h × (N + 1)`: `t_j(n) ≥ √(Σ_{m>n} a_{j,m}²)`, for closed-form columns.
    value_tails: Option<Array2<f64>>,
    /// `h × (N + 1)`: `d_j(n) ≥ √(Σ_{m>n} m a_{j,m}²)`, for closed-form columns.
    slope_tails: Option<Array2<f64>>,
    closed_form: Option<ClosedFormUnits>,
}

impl HermiteColumns {
    /// The closed-form columns of `block` through `frame` (`d × k`, orthonormal columns), to `order`.
    pub fn for_block(
        block: &KnownBlock,
        frame: ArrayView2<'_, f64>,
        order: usize,
    ) -> Result<Self, HermiteError> {
        require_length("Hermite frame rows", block.input_dim(), frame.nrows())
            .map_err(HermiteError::Response)?;
        require_finite("Hermite frame", frame.iter()).map_err(HermiteError::Response)?;
        let readers = block.readers();
        let units = block.width();
        let retained_dim = frame.ncols();
        let projected = fast_ab(&readers, &frame);
        let absolute_projected = fast_ab(&readers.mapv(f64::abs), &frame.mapv(f64::abs));
        let mut scales = Array1::<f64>::zeros(units);
        let mut directions = Array2::<f64>::zeros((units, retained_dim));
        let mut direction_shadow = Array2::<f64>::zeros((units, retained_dim));
        for unit in 0..units {
            let reader = readers.row(unit);
            let scale = reader.dot(&reader).sqrt();
            scales[unit] = scale;
            if scale > 0.0 {
                for axis in 0..retained_dim {
                    directions[[unit, axis]] = projected[[unit, axis]] / scale;
                    direction_shadow[[unit, axis]] = absolute_projected[[unit, axis]] / scale;
                }
            }
        }
        let mut columns = Self {
            scales,
            directions,
            direction_shadow,
            direction_rounding: accumulation_growth(2 * block.input_dim() + retained_dim + 4),
            coefficients: Array2::zeros((units, 0)),
            coefficient_band: Array2::zeros((units, 0)),
            value_tails: None,
            slope_tails: None,
            closed_form: Some(ClosedFormUnits {
                activation: block.activation(),
                biases: block.biases().to_owned(),
            }),
        };
        columns.set_order(order)?;
        Ok(columns)
    }

    /// Columns from any producer of the full-norm moments `μ_{j,n} = ∂_tⁿ T_{‖w_j‖²}σ_j(b_j)`.
    ///
    /// - `retained_readers` (`h × k`): `w̃_j = Qᵀw_j`.
    /// - `scales` (`h`): the full reader norms `‖w_j‖`.
    /// - `moments` (`h × (N + 1)`): `μ_{j,n}` for `n = 0..=N`, with `moment_band` their absolute
    ///   error bounds.
    ///
    /// Readers and norms are exact inputs. `a_{j,n} = ‖w_j‖ⁿ μ_{j,n}/√(n!)` is formed by `2n`
    /// roundings.
    pub fn from_moments(
        retained_readers: ArrayView2<'_, f64>,
        scales: ArrayView1<'_, f64>,
        moments: ArrayView2<'_, f64>,
        moment_band: ArrayView2<'_, f64>,
    ) -> Result<Self, HermiteError> {
        let (units, retained_dim) = retained_readers.dim();
        let reason = if scales.len() != units {
            Some(format!("{} reader norms for {units} units", scales.len()))
        } else if moments.nrows() != units || moments.ncols() == 0 {
            Some(format!(
                "the moments are {:?}; expected {units}×(N+1)",
                moments.dim()
            ))
        } else if moment_band.dim() != moments.dim() {
            Some(format!(
                "the moment band is {:?} but the moments are {:?}",
                moment_band.dim(),
                moments.dim()
            ))
        } else if !(retained_readers.iter().all(|value| value.is_finite())
            && scales.iter().all(|value| value.is_finite() && *value >= 0.0)
            && moments.iter().all(|value| value.is_finite())
            && moment_band
                .iter()
                .all(|value| value.is_finite() && *value >= 0.0))
        {
            Some(
                "readers, norms and moments must be finite, and norms and bands nonnegative"
                    .to_string(),
            )
        } else {
            None
        };
        if let Some(reason) = reason {
            return Err(HermiteError::InvalidMoments { reason });
        }
        let order = moments.ncols() - 1;
        let mut directions = Array2::<f64>::zeros((units, retained_dim));
        let mut coefficients = Array2::<f64>::zeros((units, order + 1));
        let mut coefficient_band = Array2::<f64>::zeros((units, order + 1));
        for unit in 0..units {
            let scale = scales[unit];
            let reader = retained_readers.row(unit);
            let retained = reader.dot(&reader);
            if retained > scale * scale * (1.0 + accumulation_growth(retained_dim + 2)) {
                return Err(HermiteError::RetainedNormExceedsFull {
                    unit,
                    retained: retained.sqrt(),
                    full: scale,
                });
            }
            if scale > 0.0 {
                directions.row_mut(unit).assign(&reader.mapv(|entry| entry / scale));
            }
            let mut factor = 1.0;
            for degree in 0..=order {
                if degree > 0 {
                    factor *= scale / (degree as f64).sqrt();
                }
                let value = factor * moments[[unit, degree]];
                coefficients[[unit, degree]] = value;
                coefficient_band[[unit, degree]] = factor * moment_band[[unit, degree]]
                    + accumulation_growth(2 * degree + 1) * value.abs();
            }
        }
        Ok(Self {
            scales: scales.to_owned(),
            direction_shadow: directions.mapv(f64::abs),
            directions,
            direction_rounding: accumulation_growth(1),
            coefficients,
            coefficient_band,
            value_tails: None,
            slope_tails: None,
            closed_form: None,
        })
    }

    /// Recomputes closed-form columns to `order`, keeping the directions.
    pub fn set_order(&mut self, order: usize) -> Result<(), HermiteError> {
        let Some(units) = &self.closed_form else {
            return Err(HermiteError::NoClosedForm);
        };
        let count = self.scales.len();
        let mut coefficients = Array2::<f64>::zeros((count, order + 1));
        let mut coefficient_band = Array2::<f64>::zeros((count, order + 1));
        let mut value_tails = Array2::<f64>::zeros((count, order + 1));
        let mut slope_tails = Array2::<f64>::zeros((count, order + 1));
        let mut values = vec![0.0; order + 1];
        let mut bands = vec![0.0; order + 1];
        for unit in 0..count {
            let bias = units.biases[unit];
            let scale = self.scales[unit];
            gaussian_hermite_coefficients(units.activation, bias, scale, &mut values, &mut bands)
                .map_err(HermiteError::Kernel)?;
            let (value_tail, slope_tail) =
                unit_tails(units.activation, bias, scale, &values, &bands)?;
            coefficients
                .row_mut(unit)
                .assign(&ArrayView1::from(values.as_slice()));
            coefficient_band
                .row_mut(unit)
                .assign(&ArrayView1::from(bands.as_slice()));
            value_tails
                .row_mut(unit)
                .assign(&ArrayView1::from(value_tail.as_slice()));
            slope_tails
                .row_mut(unit)
                .assign(&ArrayView1::from(slope_tail.as_slice()));
        }
        self.coefficients = coefficients;
        self.coefficient_band = coefficient_band;
        self.value_tails = Some(value_tails);
        self.slope_tails = Some(slope_tails);
        Ok(())
    }

    /// `t_j(n) ≥ √(Σ_{m>n} a_{j,m}²)`, `h × (N + 1)`, for closed-form columns.
    pub fn value_tails(&self) -> Result<ArrayView2<'_, f64>, HermiteError> {
        self.value_tails
            .as_ref()
            .map(|tails| tails.view())
            .ok_or(HermiteError::NoClosedForm)
    }

    /// `d_j(n) ≥ √(Σ_{m>n} m a_{j,m}²)`, `h × (N + 1)`, for closed-form columns.
    pub fn slope_tails(&self) -> Result<ArrayView2<'_, f64>, HermiteError> {
        self.slope_tails
            .as_ref()
            .map(|tails| tails.view())
            .ok_or(HermiteError::NoClosedForm)
    }

    /// The units' activation and biases, for closed-form columns.
    pub fn closed_form_units(&self) -> Result<(GaussianActivation, ArrayView1<'_, f64>), HermiteError> {
        self.closed_form
            .as_ref()
            .map(|units| (units.activation, units.biases.view()))
            .ok_or(HermiteError::NoClosedForm)
    }

    /// The highest degree `N` the columns carry.
    pub fn order(&self) -> usize {
        self.coefficients.ncols().saturating_sub(1)
    }

    pub fn width(&self) -> usize {
        self.scales.len()
    }

    pub fn retained_dim(&self) -> usize {
        self.directions.ncols()
    }

    /// `‖w_j‖`.
    pub fn scales(&self) -> ArrayView1<'_, f64> {
        self.scales.view()
    }

    /// `ũ_j`, `h × k`.
    pub fn directions(&self) -> ArrayView2<'_, f64> {
        self.directions.view()
    }

    /// `a_{j,n}`, `h × (N + 1)`.
    pub fn coefficients(&self) -> ArrayView2<'_, f64> {
        self.coefficients.view()
    }

    /// Bounds on each `a_{j,n}`'s absolute error, `h × (N + 1)`.
    pub fn coefficient_band(&self) -> ArrayView2<'_, f64> {
        self.coefficient_band.view()
    }

    /// Cramér's bound on each unit's tail `Σ_{n>N} a_{j,n}²`, for closed-form columns.
    pub fn tail_envelopes(&self, order: usize) -> Result<Vec<f64>, HermiteError> {
        let Some(units) = &self.closed_form else {
            return Err(HermiteError::NoClosedForm);
        };
        (0..self.scales.len())
            .map(|unit| {
                unit_tail_envelope(
                    units.activation,
                    units.biases[unit],
                    self.scales[unit],
                    order,
                )
            })
            .collect()
    }
}

/// `C(dimension + order, order)`, or `None` when it overflows `usize`. Each step
/// `C(d + s, s) = C(d + s − 1, s − 1)·(d + s)/s` is an exact integer division.
fn multi_index_count(dimension: usize, order: usize) -> Option<usize> {
    let mut count: usize = 1;
    for step in 1..=order {
        count = dimension
            .checked_add(step)
            .and_then(|factor| count.checked_mul(factor))?
            / step;
    }
    Some(count)
}

/// The energy an order-`N` proposal leaves out, `V(P) − Σ_{n=1}^N shell_n`, with `shells` the
/// degrees `1..=N` from [`HermiteResponse::degree_energies`].
pub fn hermite_truncation_energy(total: BandedEnergy, shells: &[BandedEnergy]) -> BandedEnergy {
    signed_sum(&[total], shells)
}

/// One coefficient `c_α` handed to a visitor.
pub struct CoefficientVisit<'a> {
    pub alpha: &'a [usize],
    /// `c_α`, one entry per output.
    pub value: ArrayView1<'a, f64>,
    /// A bound on each entry's absolute error.
    pub band: ArrayView1<'a, f64>,
    /// `c_αᵀ M c_α`.
    pub energy: BandedEnergy,
}

/// How the coefficient walk treats one retained coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Presence {
    /// Any exponent.
    Free,
    /// Exponent zero only.
    Absent,
    /// Exponents of at least one only.
    Present,
}

struct CoefficientWalk {
    alpha: Vec<usize>,
    /// Row `c` holds `√(D!/Π_{t<c} α_t!) Π_{t<c} ũ_{j,t}^{α_t}` per unit, `D = Σ_{t<c} α_t`.
    monomials: Array2<f64>,
    /// The same product over the direction shadows.
    shadows: Array2<f64>,
    value_weights: Array1<f64>,
    shadow_weights: Array1<f64>,
    band_weights: Array1<f64>,
    value: Array1<f64>,
    metric_value: Array1<f64>,
    shadow: Array1<f64>,
    metric_shadow: Array1<f64>,
    band: Array1<f64>,
    metric_band: Array1<f64>,
}

/// A block's Hermite expansion: scaled columns bound to the writers, the output bias and the
/// output metric.
pub struct HermiteResponse<'a> {
    columns: HermiteColumns,
    writers: ArrayView2<'a, f64>,
    /// `c`, length `p`.
    output_bias: ArrayView1<'a, f64>,
    /// `M U`, `p × h`.
    metric_writers: Array2<f64>,
    /// `|U|`, `p × h`.
    absolute_writers: Array2<f64>,
    /// `|M| |U|`, `p × h`.
    absolute_metric_writers: Array2<f64>,
}

impl<'a> HermiteResponse<'a> {
    /// `writers` (`p × h`), the `output_bias` (`p`; zeros for a layer without one) and a symmetric
    /// `metric` (`p × p`).
    pub fn new(
        columns: HermiteColumns,
        writers: ArrayView2<'a, f64>,
        output_bias: ArrayView1<'a, f64>,
        metric: ArrayView2<'a, f64>,
    ) -> Result<Self, HermiteError> {
        let outputs = writers.nrows();
        for (context, expected, got) in [
            ("writer columns per unit", columns.width(), writers.ncols()),
            ("output bias entries per output", outputs, output_bias.len()),
            ("metric rows per output", outputs, metric.nrows()),
            ("metric columns per output", outputs, metric.ncols()),
        ] {
            require_length(context, expected, got).map_err(HermiteError::Response)?;
        }
        require_finite("Hermite writers", writers.iter()).map_err(HermiteError::Response)?;
        require_finite("Hermite output bias", output_bias.iter())
            .map_err(HermiteError::Response)?;
        require_finite("Hermite metric", metric.iter()).map_err(HermiteError::Response)?;
        let metric_writers = fast_ab(&metric, &writers);
        let absolute_writers = writers.mapv(f64::abs);
        let absolute_metric_writers = fast_ab(&metric.mapv(f64::abs), &absolute_writers);
        Ok(Self {
            columns,
            writers,
            output_bias,
            metric_writers,
            absolute_writers,
            absolute_metric_writers,
        })
    }

    /// The closed-form expansion of `block` through `frame`, to `order`.
    pub fn for_block(
        block: &'a KnownBlock,
        frame: ArrayView2<'_, f64>,
        order: usize,
    ) -> Result<Self, HermiteError> {
        Self::new(
            HermiteColumns::for_block(block, frame, order)?,
            block.writers(),
            block.output_bias(),
            block.metric(),
        )
    }

    pub fn columns(&self) -> &HermiteColumns {
        &self.columns
    }

    pub fn order(&self) -> usize {
        self.columns.order()
    }

    /// Recomputes closed-form columns to `order`.
    pub fn set_order(&mut self, order: usize) -> Result<(), HermiteError> {
        self.columns.set_order(order)
    }

    /// `c_0 = E F = c + Σ_j u_j a_{j,0}` with a bound on each entry's absolute error: the columns'
    /// bounds carried through, plus `γ_{h+3}` of the absolute shadow `|c| + Σ_j |u_j||a_{j,0}|`.
    pub fn mean(&self) -> (Array1<f64>, Array1<f64>) {
        let columns = &self.columns;
        let value = self.writers.dot(&columns.coefficients.column(0)) + &self.output_bias;
        let shadow = self
            .absolute_writers
            .dot(&columns.coefficients.column(0).mapv(f64::abs))
            + self.output_bias.mapv(f64::abs);
        let band = self
            .absolute_writers
            .dot(&columns.coefficient_band.column(0))
            + shadow.mapv(|entry| accumulation_growth(columns.width() + 3) * entry);
        (value, band)
    }

    /// `(Σ_j ‖u_j‖_M √t_j(N))²`, an upper bound on `Σ_{n>N} shell_n` for every frame.
    ///
    /// `√shell_n` is the `L²_M` norm of `Σ_j u_j a_{j,n} H_{j,n}`, where
    /// `H_{j,n} = Σ_{|α|=n} √(n!/α!) ũ_j^α h_α` has norm `‖ũ_j‖ⁿ ≤ 1`. The triangle inequality
    /// bounds it by `Σ_j ‖u_j‖_M |a_{j,n}|`. Minkowski's inequality over `n > N` then moves the sum
    /// over units outside the tail, where [`unit_tail_envelope`] bounds each `Σ_{n>N} a_{j,n}²`.
    pub fn tail_envelope(&self, order: usize) -> Result<f64, HermiteError> {
        let tails = self.columns.tail_envelopes(order)?;
        let root = tails
            .iter()
            .enumerate()
            .map(|(unit, tail)| {
                self.writers
                    .column(unit)
                    .dot(&self.metric_writers.column(unit))
                    .sqrt()
                    * tail.sqrt()
            })
            .sum::<f64>();
        Ok(root * root)
    }

    /// The smallest `N ≥ 1` whose [`Self::tail_envelope`] is within `allowed_energy`.
    pub fn envelope_order(&self, allowed_energy: f64) -> Result<usize, HermiteError> {
        let mut upper = 1usize;
        while !(self.tail_envelope(upper)? <= allowed_energy) {
            upper = upper
                .checked_mul(2)
                .ok_or(HermiteError::EnvelopeOrderUnrepresentable { allowed_energy })?;
        }
        let mut lower = upper / 2;
        while upper - lower > 1 {
            let middle = lower + (upper - lower) / 2;
            if self.tail_envelope(middle)? <= allowed_energy {
                upper = middle;
            } else {
                lower = middle;
            }
        }
        Ok(upper)
    }

    /// The shells `Σ_{|α|=n} ‖c_α‖²_M = Σ_jk D_jk a_{j,n} a_{k,n} (ũ_jᵀũ_k)ⁿ` for
    /// `n = first_degree..=last_degree`.
    ///
    /// One unit row at a time forms `D_j·` and `ũ_j·Ũᵀ`, so no `h × h` matrix is held, and rows
    /// sum in unit order, so the value does not depend on the thread count. The band of shell `n`
    /// is the columns' bounds and the directions' `2nε` carried through the absolute shadow
    /// `|u_j|ᵀ|M||u_k| |a_{j,n} a_{k,n}| (S_jᵀS_k)ⁿ`. It adds `γ_{2p + nk + 2n + 6 + h²}` times that
    /// shadow for `D`, the reader products and the pair sum.
    pub fn degree_energies(
        &self,
        first_degree: usize,
        last_degree: usize,
    ) -> Result<Vec<BandedEnergy>, HermiteError> {
        let order = self.order();
        if first_degree == 0 || first_degree > last_degree || last_degree > order {
            return Err(HermiteError::InvalidDegreeRange {
                first_degree,
                last_degree,
                order,
            });
        }
        let count = last_degree - first_degree + 1;
        let columns = &self.columns;
        let units = columns.width();
        let epsilon = columns.direction_rounding;
        let rows = (0..units)
            .into_par_iter()
            .map(|left| {
                let mut value = vec![0.0; count];
                let mut absolute = vec![0.0; count];
                let mut propagated = vec![0.0; count];
                let left_metric_writer = self.metric_writers.column(left);
                let left_absolute_metric_writer = self.absolute_metric_writers.column(left);
                let left_direction = columns.directions.row(left);
                let left_shadow = columns.direction_shadow.row(left);
                for right in 0..units {
                    let coupling = self.writers.column(right).dot(&left_metric_writer);
                    let absolute_coupling = self
                        .absolute_writers
                        .column(right)
                        .dot(&left_absolute_metric_writer);
                    let correlation = left_direction.dot(&columns.directions.row(right));
                    let shadow_correlation = left_shadow.dot(&columns.direction_shadow.row(right));
                    let mut power = integer_power(correlation, first_degree);
                    let mut shadow_power = integer_power(shadow_correlation, first_degree);
                    for offset in 0..count {
                        let degree = first_degree + offset;
                        let left_value = columns.coefficients[[left, degree]];
                        let right_value = columns.coefficients[[right, degree]];
                        let product = (left_value * right_value).abs();
                        let shadow = absolute_coupling * shadow_power;
                        value[offset] += coupling * left_value * right_value * power;
                        absolute[offset] += shadow * product;
                        propagated[offset] += shadow
                            * (columns.coefficient_band[[left, degree]] * right_value.abs()
                                + left_value.abs() * columns.coefficient_band[[right, degree]]
                                + 2.0 * degree as f64 * epsilon * product);
                        power *= correlation;
                        shadow_power *= shadow_correlation;
                    }
                }
                (value, absolute, propagated)
            })
            .collect::<Vec<_>>();
        let outputs = self.writers.nrows();
        let retained_dim = columns.retained_dim();
        Ok((0..count)
            .map(|offset| {
                let degree = first_degree + offset;
                let mut value = 0.0;
                let mut absolute = 0.0;
                let mut propagated = 0.0;
                for row in &rows {
                    value += row.0[offset];
                    absolute += row.1[offset];
                    propagated += row.2[offset];
                }
                let operations = (2 * outputs + degree * retained_dim + 2 * degree + 6)
                    .saturating_add(units.saturating_mul(units));
                BandedEnergy {
                    value,
                    band: accumulation_growth(operations) * absolute + propagated,
                }
            })
            .collect())
    }

    /// Visits every `α` with `1 ≤ |α| ≤ N` over the retained coordinates, in lexicographic order,
    /// handing over `c_α = Σ_j u_j a_{j,|α|} √(|α|!/α!) ũ_j^α`, its band and `c_αᵀ M c_α`.
    ///
    /// The multinomial weight builds incrementally: raising `α_c` to `e` while the degree becomes
    /// `D + e` multiplies it by `√((D + e)/e) ũ_{j,c}`, so no factorial is formed. The entry band
    /// carries each column's bound and the directions' `nε` through the shadow weights, plus
    /// `γ_{3n + k + h + 4}` of the shadow. The energy band is `2|c|ᵀ|M|δc + δcᵀ|M|δc` plus
    /// `γ_{p² + p + h + 4}` of `|c|ᵀ|M||c|`.
    pub fn for_each_coefficient<F>(&self, visit: F)
    where
        F: FnMut(&CoefficientVisit<'_>),
    {
        self.walk_pattern(&vec![Presence::Free; self.columns.retained_dim()], visit);
    }

    fn walk_pattern<F>(&self, pattern: &[Presence], mut visit: F)
    where
        F: FnMut(&CoefficientVisit<'_>),
    {
        let units = self.columns.width();
        let outputs = self.writers.nrows();
        let retained_dim = self.columns.retained_dim();
        let mut walk = CoefficientWalk {
            alpha: vec![0; retained_dim],
            monomials: Array2::zeros((retained_dim + 1, units)),
            shadows: Array2::zeros((retained_dim + 1, units)),
            value_weights: Array1::zeros(units),
            shadow_weights: Array1::zeros(units),
            band_weights: Array1::zeros(units),
            value: Array1::zeros(outputs),
            metric_value: Array1::zeros(outputs),
            shadow: Array1::zeros(outputs),
            metric_shadow: Array1::zeros(outputs),
            band: Array1::zeros(outputs),
            metric_band: Array1::zeros(outputs),
        };
        walk.monomials.row_mut(0).fill(1.0);
        walk.shadows.row_mut(0).fill(1.0);
        self.walk_coordinate(0, 0, pattern, &mut walk, &mut visit);
    }

    fn walk_coordinate<F>(
        &self,
        coordinate: usize,
        degree: usize,
        pattern: &[Presence],
        walk: &mut CoefficientWalk,
        visit: &mut F,
    ) where
        F: FnMut(&CoefficientVisit<'_>),
    {
        let columns = &self.columns;
        let units = columns.width();
        let retained_dim = columns.retained_dim();
        if coordinate == retained_dim {
            if degree == 0 {
                return;
            }
            let growth = accumulation_growth(3 * degree + retained_dim + units + 4);
            let rounding = degree as f64 * columns.direction_rounding;
            for unit in 0..units {
                let value = columns.coefficients[[unit, degree]];
                let shadow = walk.shadows[[coordinate, unit]];
                walk.value_weights[unit] = walk.monomials[[coordinate, unit]] * value;
                walk.shadow_weights[unit] = shadow * value.abs();
                walk.band_weights[unit] = shadow
                    * (columns.coefficient_band[[unit, degree]] + rounding * value.abs())
                    + growth * shadow * value.abs();
            }
            general_mat_vec_mul(1.0, &self.writers, &walk.value_weights, 0.0, &mut walk.value);
            general_mat_vec_mul(
                1.0,
                &self.metric_writers,
                &walk.value_weights,
                0.0,
                &mut walk.metric_value,
            );
            general_mat_vec_mul(
                1.0,
                &self.absolute_writers,
                &walk.shadow_weights,
                0.0,
                &mut walk.shadow,
            );
            general_mat_vec_mul(
                1.0,
                &self.absolute_metric_writers,
                &walk.shadow_weights,
                0.0,
                &mut walk.metric_shadow,
            );
            general_mat_vec_mul(
                1.0,
                &self.absolute_writers,
                &walk.band_weights,
                0.0,
                &mut walk.band,
            );
            general_mat_vec_mul(
                1.0,
                &self.absolute_metric_writers,
                &walk.band_weights,
                0.0,
                &mut walk.metric_band,
            );
            let outputs = walk.value.len();
            let energy = BandedEnergy {
                value: walk.value.dot(&walk.metric_value),
                band: 2.0 * walk.shadow.dot(&walk.metric_band)
                    + walk.band.dot(&walk.metric_band)
                    + accumulation_growth(outputs * outputs + outputs + units + 4)
                        * walk.shadow.dot(&walk.metric_shadow),
            };
            visit(&CoefficientVisit {
                alpha: &walk.alpha,
                value: walk.value.view(),
                band: walk.band.view(),
                energy,
            });
            return;
        }
        for unit in 0..units {
            walk.monomials[[coordinate + 1, unit]] = walk.monomials[[coordinate, unit]];
            walk.shadows[[coordinate + 1, unit]] = walk.shadows[[coordinate, unit]];
        }
        walk.alpha[coordinate] = 0;
        if pattern[coordinate] != Presence::Present {
            self.walk_coordinate(coordinate + 1, degree, pattern, walk, visit);
        }
        if pattern[coordinate] != Presence::Absent {
            for exponent in 1..=(self.order() - degree) {
                let multinomial_step = ((degree + exponent) as f64 / exponent as f64).sqrt();
                for unit in 0..units {
                    walk.monomials[[coordinate + 1, unit]] *=
                        multinomial_step * columns.directions[[unit, coordinate]];
                    walk.shadows[[coordinate + 1, unit]] *=
                        multinomial_step * columns.direction_shadow[[unit, coordinate]];
                }
                walk.alpha[coordinate] = exponent;
                self.walk_coordinate(coordinate + 1, degree + exponent, pattern, walk, visit);
            }
        }
        walk.alpha[coordinate] = 0;
    }

    /// The live footprint of `count` stored coefficients, each two `p`-vectors and one `k`-index,
    /// reserved on the process memory governor. An uncountable footprint is refused as not
    /// addressable.
    fn reserve_coefficients(
        &self,
        count: Option<usize>,
        context: &str,
    ) -> Result<MemoryReservation, HermiteError> {
        let retained_dim = self.columns.retained_dim();
        let outputs = self.writers.nrows();
        let per_coefficient = outputs
            .checked_mul(2 * std::mem::size_of::<f64>())
            .and_then(|values| {
                retained_dim
                    .checked_mul(std::mem::size_of::<usize>())
                    .and_then(|index| values.checked_add(index))
            });
        let bytes = count
            .and_then(|count| per_coefficient.and_then(|per| count.checked_mul(per)))
            .ok_or(HermiteError::TooManyCoefficients {
                dimension: retained_dim,
                order: self.order(),
                outputs,
            })?;
        MemoryGovernor::global()
            .try_reserve(bytes, context)
            .map_err(HermiteError::Memory)
    }

    /// The coefficients whose support is exactly `support`: the `C(N, |S|)` multi-indices with
    /// `α_i ≥ 1` on `S` and `α_i = 0` elsewhere, reserved on the process memory governor.
    pub fn support_coefficients(
        &self,
        support: &[usize],
    ) -> Result<Governed<ProposalTerm>, HermiteError> {
        let retained_dim = self.columns.retained_dim();
        let ascending = support.windows(2).all(|pair| pair[0] < pair[1]);
        if !ascending || support.last().is_some_and(|last| *last >= retained_dim) {
            return Err(HermiteError::InvalidSupport {
                support: support.to_vec(),
                retained_dim,
            });
        }
        let count = match self.order().checked_sub(support.len()) {
            Some(free) if !support.is_empty() => multi_index_count(support.len(), free),
            Some(_) | None => Some(0),
        };
        let reservation = self.reserve_coefficients(count, "Hermite support coefficients")?;
        let mut pattern = vec![Presence::Absent; retained_dim];
        for coordinate in support {
            pattern[*coordinate] = Presence::Present;
        }
        let mut term = ProposalTerm {
            support: support.to_vec(),
            energy: BandedEnergy::ZERO,
            coefficients: Vec::new(),
        };
        let mut absolute = 0.0;
        self.walk_pattern(&pattern, |visit| {
            term.energy.value += visit.energy.value;
            term.energy.band += visit.energy.band;
            absolute += visit.energy.value.abs();
            term.coefficients.push(ProposalCoefficient {
                alpha: visit.alpha.to_vec(),
                value: visit.value.to_vec(),
                band: visit.band.to_vec(),
            });
        });
        term.energy.band +=
            accumulation_growth(term.coefficients.len().saturating_sub(1)) * absolute;
        Ok(reservation.bind(term))
    }

    /// Every coefficient, grouped into support terms largest first. The `C(k + N, N) − 1` stored
    /// coefficients are reserved on the process memory governor first.
    pub fn proposal_terms(&self) -> Result<Governed<Vec<ProposalTerm>>, HermiteError> {
        let count = multi_index_count(self.columns.retained_dim(), self.order())
            .map(|count| count - 1);
        let reservation = self.reserve_coefficients(count, "Hermite proposal coefficients")?;
        let mut grouped: BTreeMap<Vec<usize>, ProposalTerm> = BTreeMap::new();
        let mut counts: BTreeMap<Vec<usize>, (f64, usize)> = BTreeMap::new();
        self.for_each_coefficient(|visit| {
            let support = support_of(visit.alpha);
            let tally = counts.entry(support.clone()).or_insert((0.0, 0));
            tally.0 += visit.energy.value.abs();
            tally.1 += 1;
            let term = grouped.entry(support.clone()).or_insert(ProposalTerm {
                support,
                energy: BandedEnergy::ZERO,
                coefficients: Vec::new(),
            });
            term.energy.value += visit.energy.value;
            term.energy.band += visit.energy.band;
            term.coefficients.push(ProposalCoefficient {
                alpha: visit.alpha.to_vec(),
                value: visit.value.to_vec(),
                band: visit.band.to_vec(),
            });
        });
        let mut terms = grouped
            .into_iter()
            .map(|(support, mut term)| {
                if let Some((absolute, count)) = counts.get(&support) {
                    term.energy.band += accumulation_growth(count.saturating_sub(1)) * absolute;
                }
                term
            })
            .collect::<Vec<_>>();
        terms.sort_by(|left, right| right.energy.value.total_cmp(&left.energy.value));
        Ok(reservation.bind(terms))
    }
}

fn support_of(alpha: &[usize]) -> Vec<usize> {
    alpha
        .iter()
        .enumerate()
        .filter(|entry| *entry.1 > 0)
        .map(|entry| entry.0)
        .collect()
}

/// One stored coefficient of a proposal term.
#[derive(Clone, Debug)]
pub struct ProposalCoefficient {
    pub alpha: Vec<usize>,
    pub value: Vec<f64>,
    pub band: Vec<f64>,
}

/// The coefficients with one support, with their energy.
#[derive(Clone, Debug)]
pub struct ProposalTerm {
    pub support: Vec<usize>,
    pub energy: BandedEnergy,
    pub coefficients: Vec<ProposalCoefficient>,
}

/// The declared fraction of the retained variance an expansion may leave uncaptured. It has no
/// default: the caller declares it.
#[derive(Clone, Copy, Debug)]
pub struct FidelityBudget {
    pub unexplained_fraction: f64,
}

/// The order chosen from a [`FidelityBudget`], with its exact truncation error.
#[derive(Clone, Debug)]
pub struct OrderSelection {
    pub order: usize,
    /// The a-priori order at which the Cramér envelope alone guarantees the budget.
    pub envelope_order: usize,
    /// The shells `n = 1..=order`.
    pub degree_energies: Vec<BandedEnergy>,
    pub captured_energy: BandedEnergy,
    /// `V(P)`.
    pub total_energy: BandedEnergy,
    /// `V(P) − captured_energy`.
    pub truncation_error: BandedEnergy,
}

/// A retained response's Hermite basis proposal for a compact GAM.
#[derive(Debug)]
pub struct HermiteProposal {
    pub selection: OrderSelection,
    /// `E F̄_P = c_0`, with each entry's band.
    pub mean: Vec<f64>,
    pub mean_band: Vec<f64>,
    /// Support terms, largest energy first, holding their memory reservation.
    pub terms: Governed<Vec<ProposalTerm>>,
}

/// [`select_order`] with the whole proposal materialized.
pub fn propose(
    block: &KnownBlock,
    frame: ArrayView2<'_, f64>,
    budget: FidelityBudget,
) -> Result<HermiteProposal, HermiteError> {
    let (response, selection) = select_order(block, frame, budget)?;
    let (mean, mean_band) = response.mean();
    let terms = response.proposal_terms()?;
    Ok(HermiteProposal {
        selection,
        mean: mean.to_vec(),
        mean_band: mean_band.to_vec(),
        terms,
    })
}

/// The smallest order whose truncation error, band included, is within
/// `budget.unexplained_fraction · V(P)`, with `V(P)` from the block's kernels.
///
/// `V(P)` enters with the band its owner publishes ([`KnownBlock::explained_variance`]).
pub fn select_order<'a>(
    block: &'a KnownBlock,
    frame: ArrayView2<'_, f64>,
    budget: FidelityBudget,
) -> Result<(HermiteResponse<'a>, OrderSelection), HermiteError> {
    let total = block
        .explained_variance(frame)
        .map_err(HermiteError::Response)?;
    select_order_with_total_energy(block, frame, total, budget)
}

/// [`select_order`] for a caller holding `V(P)` with its band.
///
/// The scan doubles the order it computes and stops at the first degree whose truncation error
/// plus its band meets the budget. It never passes the envelope order, where the exact tail is
/// guaranteed within budget. There a computed tail consistent with the budget within its band is
/// accepted, and one above it beyond its band is refused as disagreement between `V(P)` and the
/// shells. A retained variance not resolved from zero is a constant response, of order zero.
pub fn select_order_with_total_energy<'a>(
    block: &'a KnownBlock,
    frame: ArrayView2<'_, f64>,
    total_energy: BandedEnergy,
    budget: FidelityBudget,
) -> Result<(HermiteResponse<'a>, OrderSelection), HermiteError> {
    let fraction = budget.unexplained_fraction;
    if !(fraction > 0.0 && fraction < 1.0) {
        return Err(HermiteError::InvalidBudget {
            unexplained_fraction: fraction,
        });
    }
    let admissible = total_energy.value.is_finite()
        && total_energy.band.is_finite()
        && total_energy.band >= 0.0
        && total_energy.value >= -total_energy.band;
    if !admissible {
        return Err(HermiteError::InvalidTotalEnergy { total_energy });
    }
    let mut response = HermiteResponse::for_block(block, frame, 0)?;
    let finish = |response: &mut HermiteResponse<'a>,
                  order: usize,
                  envelope_order: usize,
                  shells: Vec<BandedEnergy>|
     -> Result<OrderSelection, HermiteError> {
        response.set_order(order)?;
        Ok(OrderSelection {
            order,
            envelope_order,
            captured_energy: signed_sum(&shells, &[]),
            truncation_error: hermite_truncation_energy(total_energy, &shells),
            degree_energies: shells,
            total_energy,
        })
    };
    if !total_energy.resolved_positive() {
        let selection = finish(&mut response, 0, 0, Vec::new())?;
        return Ok((response, selection));
    }
    let allowed_energy = fraction * total_energy.value;
    let envelope_order = response.envelope_order(allowed_energy)?;
    let mut shells = Vec::new();
    let mut trial_order = 1usize;
    loop {
        let bounded_order = trial_order.min(envelope_order);
        response.set_order(bounded_order)?;
        let first_degree = shells.len() + 1;
        for shell in response.degree_energies(first_degree, bounded_order)? {
            shells.push(shell);
            let truncation = hermite_truncation_energy(total_energy, &shells);
            if truncation.value + truncation.band <= allowed_energy {
                let order = shells.len();
                let selection = finish(&mut response, order, envelope_order, shells)?;
                return Ok((response, selection));
            }
        }
        if bounded_order == envelope_order {
            let truncation = hermite_truncation_energy(total_energy, &shells);
            if truncation.value - truncation.band <= allowed_energy {
                let selection = finish(&mut response, envelope_order, envelope_order, shells)?;
                return Ok((response, selection));
            }
            return Err(HermiteError::TailAboveEnvelope {
                order: envelope_order,
                computed_tail: truncation,
                allowed_energy,
            });
        }
        trial_order = trial_order.saturating_mul(2);
    }
}

/// A probabilists' Gauss-Hermite rule for `E[g(E)]`, `E ~ N(0, 1)`, with its a-posteriori accuracy.
///
/// `m` nodes integrate every polynomial of degree `≤ 2m − 1` exactly.
#[derive(Clone, Debug)]
pub struct GaussianRule {
    nodes: Vec<f64>,
    weights: Vec<f64>,
    node_residual: f64,
    weight_residual: f64,
}

impl GaussianRule {
    /// Converts gam-math's physicists' rule (`x ↦ √2 x`, `w ↦ w/√π`) and certifies it against
    /// the zeros of `He_m` and the Christoffel numbers.
    pub fn with_node_count(node_count: usize) -> Result<Self, HermiteError> {
        let physicists = gauss_hermite_rule(node_count).map_err(HermiteError::Quadrature)?;
        let sqrt_pi = std::f64::consts::PI.sqrt();
        let nodes = physicists
            .nodes
            .iter()
            .map(|node| std::f64::consts::SQRT_2 * node)
            .collect::<Vec<_>>();
        let weights = physicists
            .weights
            .iter()
            .map(|weight| weight / sqrt_pi)
            .collect::<Vec<_>>();
        let mut functions = vec![0.0; node_count + 1];
        let mut node_residual = 0.0_f64;
        let mut weight_residual = 0.0_f64;
        for (node, weight) in nodes.iter().zip(&weights) {
            hermite_functions(*node, &mut functions);
            let newton_step =
                functions[node_count] / ((node_count as f64).sqrt() * functions[node_count - 1]);
            let energy = functions[..node_count]
                .iter()
                .map(|value| value * value)
                .sum::<f64>();
            let christoffel = functions[0] * functions[0] / energy;
            let christoffel_slope = 2.0 * christoffel / energy
                * (1..node_count)
                    .map(|rank| (rank as f64).sqrt() * functions[rank] * functions[rank - 1])
                    .sum::<f64>()
                    .abs();
            node_residual = node_residual.max(newton_step.abs());
            weight_residual = weight_residual
                .max((weight - christoffel).abs() + christoffel_slope * newton_step.abs());
        }
        Ok(Self {
            nodes,
            weights,
            node_residual,
            weight_residual,
        })
    }

    /// The smallest rule integrating every polynomial of degree `≤ degree` exactly.
    pub fn exact_for_degree(degree: usize) -> Result<Self, HermiteError> {
        Self::with_node_count(degree / 2 + 1)
    }

    pub fn nodes(&self) -> &[f64] {
        &self.nodes
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    /// `max_i |He_m(x_i)/He_m′(x_i)|`, the Newton step to the nearest zero of `He_m`. To first order
    /// it is the distance from each computed node to the exact one.
    pub fn node_residual(&self) -> f64 {
        self.node_residual
    }

    /// `max_i (|w_i − λ(x_i)| + |λ′(x_i)| δ_i)` against the Christoffel numbers
    /// `λ(x) = 1/Σ_{k<m} h_k(x)²`, with `λ′ = −2λ Σ_k √k h_k h_{k−1}/Σ_k h_k²` and `δ_i` the node's
    /// Newton step. To first order it bounds each weight's distance from `λ` at the exact node.
    pub fn weight_residual(&self) -> f64 {
        self.weight_residual
    }
}

/// Every multi-index `α ∈ ℕ^dimension` with `|α| ≤ order`, in lexicographic order.
pub fn multi_indices(dimension: usize, order: usize) -> Vec<Vec<usize>> {
    let mut indices = Vec::new();
    let mut alpha = vec![0; dimension];
    push_multi_indices(0, order, &mut alpha, &mut indices);
    indices
}

fn push_multi_indices(
    axis: usize,
    remaining: usize,
    alpha: &mut Vec<usize>,
    indices: &mut Vec<Vec<usize>>,
) {
    if axis == alpha.len() {
        indices.push(alpha.clone());
        return;
    }
    for exponent in 0..=remaining {
        alpha[axis] = exponent;
        push_multi_indices(axis + 1, remaining - exponent, alpha, indices);
    }
    alpha[axis] = 0;
}

/// One coefficient `c_α ∈ R^p` of a projected response.
#[derive(Clone, Debug)]
pub struct HermiteCoefficient {
    pub alpha: Vec<usize>,
    pub value: Vec<f64>,
}

/// Projects `f: R^k → R^p` onto every `h_α` with `|α| ≤ order`: `c_α = E[f(Y) h_α(Y)]`,
/// `Y ~ N(0, I_k)`, by the tensor rule.
///
/// When `f` is a polynomial of per-axis degree `D` and `rule` is exact for degree `D + order`,
/// every coefficient is exact up to roundoff and the rule's residuals. Otherwise the accuracy is
/// the caller's, checked against a doubled rule.
pub fn project_response<F>(
    dimension: usize,
    order: usize,
    output_dimension: usize,
    rule: &GaussianRule,
    mut response: F,
) -> Result<Vec<HermiteCoefficient>, HermiteError>
where
    F: FnMut(&[f64], &mut [f64]),
{
    let node_count = rule.nodes.len();
    let unrepresentable = HermiteError::GridUnrepresentable {
        node_count,
        dimension,
        output_dimension,
    };
    let points = match u32::try_from(dimension)
        .ok()
        .and_then(|exponent| node_count.checked_pow(exponent))
    {
        Some(points)
            if points.checked_mul(output_dimension).is_some()
                && points.checked_mul(dimension).is_some() =>
        {
            points
        }
        Some(_) | None => return Err(unrepresentable),
    };
    let columns = order + 1;
    let mut table = vec![0.0; node_count * columns];
    for (node_index, node) in rule.nodes.iter().enumerate() {
        normalized_hermite_polynomials(
            *node,
            &mut table[node_index * columns..(node_index + 1) * columns],
        );
    }
    let mut node_indices = vec![0usize; points * dimension];
    let mut point_weights = vec![0.0; points];
    let mut values = vec![0.0; points * output_dimension];
    let mut coordinates = vec![0.0; dimension];
    for point in 0..points {
        let mut remainder = point;
        let mut weight = 1.0;
        for axis in 0..dimension {
            let node_index = remainder % node_count;
            remainder /= node_count;
            node_indices[point * dimension + axis] = node_index;
            coordinates[axis] = rule.nodes[node_index];
            weight *= rule.weights[node_index];
        }
        point_weights[point] = weight;
        response(
            &coordinates,
            &mut values[point * output_dimension..(point + 1) * output_dimension],
        );
    }
    Ok(multi_indices(dimension, order)
        .into_iter()
        .map(|alpha| {
            let mut value = vec![0.0; output_dimension];
            for point in 0..points {
                let basis = alpha
                    .iter()
                    .enumerate()
                    .map(|(axis, exponent)| {
                        table[node_indices[point * dimension + axis] * columns + exponent]
                    })
                    .product::<f64>();
                let scaled = point_weights[point] * basis;
                for (total, sample) in value
                    .iter_mut()
                    .zip(&values[point * output_dimension..(point + 1) * output_dimension])
                {
                    *total += scaled * sample;
                }
            }
            HermiteCoefficient { alpha, value }
        })
        .collect())
}

/// `h_α(y) = Π_i h_{α_i}(y_i)`, the proposal basis function of `α` at retained coordinates `y`.
pub fn hermite_basis_value(alpha: &[usize], coordinates: &[f64]) -> Result<f64, HermiteError> {
    require_length("Hermite basis coordinates", alpha.len(), coordinates.len())
        .map_err(HermiteError::Response)?;
    let mut buffer = Vec::new();
    let mut value = 1.0;
    for (exponent, coordinate) in alpha.iter().zip(coordinates) {
        buffer.resize(exponent + 1, 0.0);
        normalized_hermite_polynomials(*coordinate, &mut buffer);
        value *= buffer[*exponent];
    }
    Ok(value)
}

#[cfg(test)]
#[path = "hermite_tests.rs"]
mod hermite_tests;
