//! The unit-pair loop of the retained-response operator in Mehler / Wiener chaos form (#2946).
//!
//! # Why a chaos route
//!
//! `V(P)` and its frame gradient read `K_σ − m_j m_k` and `∂_r K_σ` for every ordered unit pair: `1.5·10⁸` pairs per
//! frame at Qwen3-8B width (`h = 12288`). A closed-form pair costs transcendental calls, and a biased one a bivariate
//! normal CDF. The Gaussian chaos expansion writes each pair instead as a short polynomial in the pair's reader
//! correlation, whose coefficients belong to single units and do not depend on the frame. On Qwen3-8B layer 18 and
//! Pythia-70m layer 3 at least 99.998% of pairs certify by order 32 (#2946 comments 5716689638, 5716752180).
//!
//! # The expansion
//!
//! Write `X_j = b_j + s_j E_j` with `s_j = √v_j`, and `ρ_jk = w_jᵀ P w_k/(s_j s_k)`, the correlation of `(E_j, E_k)`.
//! With the orthonormal Hermite polynomials `h_n = He_n/√n!`, Mehler's formula and iterated Stein give
//!
//! ```text
//! K_σ − m_j m_k = Σ_{n≥1} ρ_jkⁿ a_{j,n} a_{k,n},    a_{j,n} = E σ(X_j) h_n(E_j) = s_jⁿ T_{v_j}σ⁽ⁿ⁾(b_j)/√n!,
//! s_j s_k ∂_r K_σ = Σ_{n≥1} n ρ_jkⁿ⁻¹ a_{j,n} a_{k,n},
//! ```
//!
//! the second line termwise, which is Price's theorem.
//!
//! # Columns come from their owner
//!
//! The coefficients, their error bounds and the tail bounds are one block's per-unit columns, produced by the
//! coefficient owner (the normalized coefficients of `gam_math::gaussian_activation`, wrapped per block by
//! `response::hermite`). Nothing here produces them. [`PairChaosTable::for_columns`] reads them from a block's
//! `HermiteColumns`; [`PairChaosTable::from_columns`] takes `a_{j,n}` for `n = 0..=N`
//! with `a_{j,0} = m_j`, a bound on each entry's absolute error, and banded upper bounds on the tails
//! `t_j(n)² ≥ Σ_{m>n} a_{j,m}²` and `d_j(n)² ≥ Σ_{m>n} m a_{j,m}²`. By Parseval `t_j(0) ≥ √Var σ(X_j)` and
//! `d_j(0) ≥ √(v_j E σ'(X_j)²)` bound the largest magnitudes a pair's value and slope can take, and they are the scales
//! the bounds below are stated in.
//!
//! # The truncation bound
//!
//! Cauchy–Schwarz on what order `N` leaves out gives
//!
//! ```text
//! |K_σ − m_j m_k − Σ_{n≤N} ρ_jkⁿ a_{j,n} a_{k,n}| ≤ |ρ_jk|^{N+1} t_j(N) t_k(N),
//! |s_j s_k ∂_r K_σ − Σ_{n≤N} n ρ_jkⁿ⁻¹ a_{j,n} a_{k,n}| ≤ |ρ_jk|^N d_j(N) d_k(N).
//! ```
//!
//! A row stops at the first order where both bounds, taken at the row's largest chaos correlation and against each
//! tail's envelope over units, are at most `γ` of the pair's chaos operations times its scale, `t_j(0) t_k(0)` for the
//! value and `d_j(0) d_k(0)` for the slope.
//!
//! # Routes
//!
//! The bound decays only through `|ρ|^{N+1}` as `|ρ| → 1`, and the diagonal has `ρ_jj = ‖P w_j‖²/v_j`, which is `1` at
//! `P = I`. A unit's route radius is the largest correlation whose bounds the table's order meets. An off-diagonal
//! pair takes the chaos route when `|ρ_jk|` lies within both units' radii, so the route is symmetric; every other pair
//! and every diagonal takes the closed-form [`pair_kernel`], with the covariance rounding the caller's formation
//! states. The order moves cost between the routes and nothing else: every chaos pair meets the same derived bound
//! whatever the order is.
//!
//! A zero reader is a constant unit. Its coefficients vanish, so its chaos pairs carry no value, and the slope they
//! write is never read: the frame gradient multiplies row `j` of `B` by `w_j = 0` and column `j` by `Qᵀ w_j = 0`.
//!
//! # The row band
//!
//! [`PairRowSum::band`] bounds a row value's error, given exact metric rows and covariances. Per chaos pair, the
//! truncation is at most `γ t_j(0) t_k(0)`, and Horner's rounding is at most `γ Σ_n |a_{j,n} a_{k,n}| |ρ|ⁿ` (Higham,
//! *Accuracy and Stability of Numerical Algorithms*, §5.1), which Cauchy–Schwarz bounds by the same `γ t_j(0) t_k(0)`;
//! the columns' errors `e_j` add at most `‖e_j‖ t_k(0) + t_j(0) ‖e_k‖ + ‖e_j‖ ‖e_k‖`. Per closed-form pair, the kernel
//! states its own evaluation rounding (`PairKernel::value_rounding`), forming `K − m_j m_k` adds
//! `γ_2 (|K| + |m_j m_k|)`, and the means' errors add `|m_j| e_{k,0} + e_{j,0} |m_k| + e_{j,0} e_{k,0}`. The fold adds
//! `γ` of its terms times its absolute sum.
//!
//! # Layout
//!
//! The coefficients are stored column-major by order, so one Horner step over a row is a straight loop over the
//! other units with no reduction, and it vectorizes. It multiplies and adds separately and never fuses, so the AVX2
//! copy the running CPU selects writes the same words as the portable body. The row sum is one fold in unit order,
//! with the operations of the closed-form loop. Scratch is per worker: nothing is allocated per pair.

use super::hermite::HermiteColumns;
use gam_linalg::roundoff::accumulation_growth;
use gam_math::gaussian_activation::{GaussianActivation, GaussianActivationError, PreactivationPair, pair_kernel};
use ndarray::ArrayView2;
use std::fmt;

/// The operations one chaos pair chains through order `order`, to first order in the unit roundoff: two products for
/// the correlation, per order the coefficient product and Horner's product and sum, and the closing product by the
/// correlation. The slope's Horner step takes the same product and sum. The coefficients' own errors are the
/// producer's bounds and enter the row band separately.
pub(crate) fn chaos_operations(order: usize) -> usize {
    3 * order + 3
}

/// A refusal of the chaos pair route.
#[derive(Debug, Clone, PartialEq)]
pub enum PairRouteError {
    DimensionMismatch {
        context: &'static str,
        expected: usize,
        got: usize,
    },
    /// A chaos table needs coefficients through at least order one.
    EmptyOrder,
    /// A column, bound, bias, variance or stated covariance rounding is not finite.
    NonFinite { context: &'static str },
    /// An error bound, a tail bound, a variance or a stated covariance rounding is negative.
    NegativeBound { context: &'static str },
    /// A pair on the closed-form route was refused.
    Pair {
        unit: usize,
        other: usize,
        error: GaussianActivationError,
    },
    /// The column owner refused the block's columns (`response::hermite`).
    Columns { reason: String },
}

impl fmt::Display for PairRouteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch {
                context,
                expected,
                got,
            } => write!(f, "{context}: expected {expected}, got {got}"),
            Self::EmptyOrder => write!(f, "a chaos pair table needs coefficients through at least order one"),
            Self::NonFinite { context } => write!(f, "{context} holds a non-finite entry"),
            Self::NegativeBound { context } => write!(f, "{context} holds a negative entry"),
            Self::Pair { unit, other, error } => {
                write!(f, "closed-form pair ({unit}, {other}) of the chaos pair route: {error}")
            }
            Self::Columns { reason } => write!(f, "the chaos table's Hermite columns were refused: {reason}"),
        }
    }
}

impl std::error::Error for PairRouteError {}

/// One unit's row of the pair sum and a bound on its absolute error.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PairRowSum {
    /// `Σ_k D_jk [K_σ − m_j m_k]`.
    pub value: f64,
    /// A bound on `|value − exact row|` given exact metric rows and covariances (module docs).
    pub band: f64,
}

/// The chaos coefficients, tails and route radii of one block's units, all independent of the frame.
#[derive(Debug, Clone)]
pub struct PairChaosTable {
    activation: GaussianActivation,
    width: usize,
    order: usize,
    biases: Vec<f64>,
    variances: Vec<f64>,
    /// `m_j = a_{j,0}`.
    means: Vec<f64>,
    /// The error bound of `a_{j,0}`.
    mean_bands: Vec<f64>,
    /// `1/s_j`, and `0` for a zero reader.
    inverse_scales: Vec<f64>,
    /// `a_{j,n}` for `n = 1..=order`, entry `(n − 1)·width + j`.
    coefficients: Vec<f64>,
    /// `‖(e_{j,1}, …, e_{j,order})‖₂`.
    coefficient_errors: Vec<f64>,
    /// `t_j(0)`.
    value_scales: Vec<f64>,
    /// `t_j(n)/t_j(0)` for `n = 0..=order`, entry `n·width + j`.
    value_tails: Vec<f64>,
    /// `d_j(n)/d_j(0)` for `n = 0..=order`, entry `n·width + j`.
    slope_tails: Vec<f64>,
    /// The largest value tail of each order over units.
    value_envelope: Vec<f64>,
    /// The largest slope tail of each order over units.
    slope_envelope: Vec<f64>,
    /// Each unit's route radius.
    routes: Vec<f64>,
}

/// Per-worker scratch for [`PairChaosTable::pair_row`].
#[derive(Debug, Clone)]
pub struct PairRowScratch {
    correlations: Vec<f64>,
    values: Vec<f64>,
    slopes: Vec<f64>,
    bands: Vec<f64>,
    unit_coefficients: Vec<f64>,
    direct: Vec<usize>,
}

impl PairChaosTable {
    /// The table for units `σ(b_j + w_jᵀ Z)` with biases `biases` and reader norms `scales = ‖w_j‖`, the norms the
    /// columns were scaled by (a closed-form pair reads the variance `s_j²`), from the coefficient owner's columns
    /// (`width × (N + 1)` each): `coefficients` holds `a_{j,n}` with `a_{j,0} = m_j`, `coefficient_band` a bound on
    /// each entry's absolute error, and `value_tails` and `slope_tails` the banded upper bounds `t_j(n)` and `d_j(n)`
    /// (module docs).
    pub fn from_columns(
        activation: GaussianActivation,
        biases: &[f64],
        scales: &[f64],
        coefficients: ArrayView2<'_, f64>,
        coefficient_band: ArrayView2<'_, f64>,
        value_tails: ArrayView2<'_, f64>,
        slope_tails: ArrayView2<'_, f64>,
    ) -> Result<Self, PairRouteError> {
        let width = biases.len();
        require_length("chaos table scales", width, scales.len())?;
        require_finite("chaos table biases", biases.iter())?;
        require_finite("chaos table scales", scales.iter())?;
        if scales.iter().any(|scale| *scale < 0.0) {
            return Err(PairRouteError::NegativeBound {
                context: "chaos table scales",
            });
        }
        let columns = coefficients.ncols();
        for (context, view) in [
            ("chaos coefficients", coefficients),
            ("chaos coefficient band", coefficient_band),
            ("chaos value tails", value_tails),
            ("chaos slope tails", slope_tails),
        ] {
            require_length(context, width, view.nrows())?;
            require_length(context, columns, view.ncols())?;
            require_finite(context, view.iter())?;
        }
        for (context, view) in [
            ("chaos coefficient band", coefficient_band),
            ("chaos value tails", value_tails),
            ("chaos slope tails", slope_tails),
        ] {
            if view.iter().any(|entry| *entry < 0.0) {
                return Err(PairRouteError::NegativeBound { context });
            }
        }
        if columns < 2 {
            return Err(PairRouteError::EmptyOrder);
        }
        let order = columns - 1;
        let mut means = vec![0.0; width];
        let mut mean_bands = vec![0.0; width];
        let mut inverse_scales = vec![0.0; width];
        let mut packed = vec![0.0; order * width];
        let mut coefficient_errors = vec![0.0; width];
        let mut value_scales = vec![0.0; width];
        let mut relative_value_tails = vec![0.0; columns * width];
        let mut relative_slope_tails = vec![0.0; columns * width];
        for unit in 0..width {
            means[unit] = coefficients[[unit, 0]];
            mean_bands[unit] = coefficient_band[[unit, 0]];
            if scales[unit] > 0.0 {
                inverse_scales[unit] = scales[unit].recip();
            }
            let mut error_square = 0.0;
            for degree in 1..=order {
                packed[(degree - 1) * width + unit] = coefficients[[unit, degree]];
                let error = coefficient_band[[unit, degree]];
                error_square += error * error;
            }
            coefficient_errors[unit] = error_square.sqrt();
            let value_scale = value_tails[[unit, 0]];
            let slope_scale = slope_tails[[unit, 0]];
            value_scales[unit] = value_scale;
            for degree in 0..=order {
                relative_value_tails[degree * width + unit] = relative(value_tails[[unit, degree]], value_scale);
                relative_slope_tails[degree * width + unit] = relative(slope_tails[[unit, degree]], slope_scale);
            }
        }
        let value_envelope = envelope(&relative_value_tails, width, order);
        let slope_envelope = envelope(&relative_slope_tails, width, order);
        let mut table = Self {
            activation,
            width,
            order,
            biases: biases.to_vec(),
            variances: scales.iter().map(|scale| scale * scale).collect(),
            means,
            mean_bands,
            inverse_scales,
            coefficients: packed,
            coefficient_errors,
            value_scales,
            value_tails: relative_value_tails,
            slope_tails: relative_slope_tails,
            value_envelope,
            slope_envelope,
            routes: vec![0.0; width],
        };
        for unit in 0..width {
            let radius = table.route_radius(unit);
            table.routes[unit] = radius;
        }
        Ok(table)
    }

    /// The table for a block's closed-form Hermite columns: the units' activation and biases, the reader norms the
    /// columns were scaled by, the coefficients with their band, and both banded tails, all read from the column owner.
    /// Moment columns carry no tails and are refused.
    ///
    /// The table's order is `min(N_sat, N)`, with `N` the order the caller built the columns to and `N_sat` the
    /// [`saturation order`](Self::saturation_order). Past `N_sat` no pair's route or value moves, since every row
    /// already stops at or below it, so the columns' further orders are only storage. Where the columns never reach
    /// it, as ReLU's slowly decaying tails do not, the caller's `N` is the table's. A higher order would only move more
    /// high-correlation pairs from the closed form to the chaos route: a cost choice, which the SpeedGate measures.
    pub fn for_columns(columns: &HermiteColumns) -> Result<Self, PairRouteError> {
        let refused = |error: super::hermite::HermiteError| PairRouteError::Columns {
            reason: format!("{error:?}"),
        };
        let (activation, biases) = columns.closed_form_units().map_err(refused)?;
        let biases = biases.to_vec();
        let scales = columns.scales().to_vec();
        let value_tails = columns.value_tails().map_err(refused)?;
        let slope_tails = columns.slope_tails().map_err(refused)?;
        let full = Self::from_columns(
            activation,
            &biases,
            &scales,
            columns.coefficients(),
            columns.coefficient_band(),
            value_tails,
            slope_tails,
        )?;
        let order = full.saturation_order();
        if order == full.order {
            return Ok(full);
        }
        let kept = ndarray::s![.., ..=order];
        Self::from_columns(
            activation,
            &biases,
            &scales,
            columns.coefficients().slice(kept),
            columns.coefficient_band().slice(kept),
            value_tails.slice(kept),
            slope_tails.slice(kept),
        )
    }

    /// The order to build a block's columns to so that [`Self::saturation_order`] finds a saturation order whenever the
    /// column owner's a-priori envelopes admit one: the smallest `N` at which every unit's value and slope tail
    /// envelopes (`response::hermite::unit_tail_envelope`, `unit_slope_tail_envelope`), taken relative to the unit's
    /// own scales `t_j(0)` and `d_j(0)`, meet both chaos bounds at correlation one. The actual tails are at most the
    /// envelopes, so a table built to `N` saturates at or below it. The scales are read from `columns` at any order.
    ///
    /// `None` when some unit's envelopes never do, as ReLU's slope envelope is `+∞`: such a block has no saturation
    /// order an a-priori bound can find. Both envelopes shrink with `N` and the bound grows with it, so the condition
    /// holds from its first order on and is found by doubling and bisection.
    pub fn covering_column_order(columns: &HermiteColumns) -> Result<Option<usize>, PairRouteError> {
        let refused = |error: super::hermite::HermiteError| PairRouteError::Columns {
            reason: format!("{error:?}"),
        };
        let (activation, biases) = columns.closed_form_units().map_err(refused)?;
        let scales = columns.scales();
        let value_scales = columns.value_tails().map_err(refused)?.column(0).to_owned();
        let slope_scales = columns.slope_tails().map_err(refused)?.column(0).to_owned();
        let covered = |order: usize| -> Result<bool, PairRouteError> {
            let band = accumulation_growth(chaos_operations(order)).sqrt();
            for unit in 0..scales.len() {
                let (bias, scale) = (biases[unit], scales[unit]);
                let value = super::hermite::unit_tail_envelope(activation, bias, scale, order).map_err(refused)?;
                let slope =
                    super::hermite::unit_slope_tail_envelope(activation, bias, scale, order).map_err(refused)?;
                if !(value.sqrt() <= band * value_scales[unit] && slope.sqrt() <= band * slope_scales[unit]) {
                    return Ok(false);
                }
            }
            Ok(true)
        };
        let mut upper = 1usize;
        while !covered(upper)? {
            // An envelope that is not finite at this order is not finite at any: the envelope forms have no order
            // at which `+∞` turns finite, and a doubling that overflows has no order left to try.
            if (0..scales.len()).any(|unit| {
                super::hermite::unit_slope_tail_envelope(activation, biases[unit], scales[unit], upper)
                    .is_ok_and(|slope| !slope.is_finite())
                    || super::hermite::unit_tail_envelope(activation, biases[unit], scales[unit], upper)
                        .is_ok_and(|value| !value.is_finite())
            }) {
                return Ok(None);
            }
            let Some(doubled) = upper.checked_mul(2) else {
                return Ok(None);
            };
            upper = doubled;
        }
        let mut lower = upper / 2;
        while lower + 1 < upper {
            let middle = lower + (upper - lower) / 2;
            if covered(middle)? {
                upper = middle;
            } else {
                lower = middle;
            }
        }
        Ok(Some(upper))
    }

    /// The highest order the table carries.
    pub fn order(&self) -> usize {
        self.order
    }

    /// The smallest order at which every unit meets both chaos bounds at correlation one, so every off-diagonal
    /// pair's route radius is one and no row needs a higher order; the table's own order when no order it carries
    /// does.
    pub fn saturation_order(&self) -> usize {
        (1..=self.order)
            .find(|&order| (0..self.width).all(|unit| self.meets_bounds(unit, order, 1.0, true)))
            .unwrap_or(self.order)
    }

    /// Scratch for one worker's rows of this table.
    pub fn row_scratch(&self) -> PairRowScratch {
        PairRowScratch {
            correlations: vec![0.0; self.width],
            values: vec![0.0; self.width],
            slopes: vec![0.0; self.width],
            bands: vec![0.0; self.width],
            unit_coefficients: Vec::with_capacity(self.order),
            direct: Vec::new(),
        }
    }

    /// One unit's row of the pair sum over the columns `k ≥ first_column`, `Σ_{k ≥ first_column} D_jk [K_σ − m_j m_k]`,
    /// with its error band. `covariances` holds `r_jk = w_jᵀ P w_k` and `weights` enters as `D_jk`, both for
    /// `k = first_column..width` in order; with `gradient` `weights` leaves as `B_jk = D_jk ∂_r K_σ`. A caller that
    /// assembles a symmetric sum from `k ≥ j` passes `first_column = j`, reading one packed upper row, and every pair
    /// it evaluates is the same function of its own correlation as in the full row.
    ///
    /// `covariance_rounding(k)`, for the column index `k` itself, is the `covariance_rounding` the caller's formation
    /// of `r_jk` states for the pair `(unit, k)` (`response::subspace::covariance_rounding_band`). Only closed-form
    /// pairs read it.
    pub fn pair_row(
        &self,
        unit: usize,
        first_column: usize,
        covariances: &[f64],
        covariance_rounding: impl Fn(usize) -> f64,
        weights: &mut [f64],
        gradient: bool,
        scratch: &mut PairRowScratch,
    ) -> Result<PairRowSum, PairRouteError> {
        let width = self.width;
        require_length("pair row scratch", width, scratch.values.len())?;
        if first_column > width {
            return Err(PairRouteError::DimensionMismatch {
                context: "pair row first column",
                expected: width,
                got: first_column,
            });
        }
        let count = width - first_column;
        require_length("pair row covariances", count, covariances.len())?;
        require_length("pair row weights", count, weights.len())?;
        let PairRowScratch {
            correlations,
            values,
            slopes,
            bands,
            unit_coefficients,
            direct,
        } = scratch;
        let (correlations, values, slopes, bands) = (
            &mut correlations[..count],
            &mut values[..count],
            &mut slopes[..count],
            &mut bands[..count],
        );
        let inverse_scales = &self.inverse_scales[first_column..];
        let inverse_unit = self.inverse_scales[unit];
        for ((correlation, &covariance), &inverse_scale) in
            correlations.iter_mut().zip(covariances).zip(inverse_scales)
        {
            *correlation = covariance * inverse_unit * inverse_scale;
        }
        let radius = self.routes[unit];
        direct.clear();
        let mut largest = 0.0f64;
        for (other, &correlation) in (first_column..width).zip(correlations.iter()) {
            let magnitude = correlation.abs();
            if other == unit || !(magnitude <= radius.min(self.routes[other])) {
                direct.push(other);
            } else {
                largest = largest.max(magnitude);
            }
        }
        let order = self.row_order(unit, largest, gradient);
        unit_coefficients.clear();
        for degree in 0..order {
            unit_coefficients.push(self.coefficients[degree * width + unit]);
        }
        chaos_orders(
            unit_coefficients.as_slice(),
            &self.coefficients,
            width,
            first_column,
            correlations,
            values,
            slopes,
            gradient,
        );
        if gradient {
            for (slope, &inverse_scale) in slopes.iter_mut().zip(inverse_scales) {
                *slope *= inverse_unit * inverse_scale;
            }
        }
        let unit_scale = self.value_scales[unit];
        let unit_error = self.coefficient_errors[unit];
        let series_band = 2.0 * accumulation_growth(chaos_operations(order)) * unit_scale;
        for ((band, &scale), &error) in bands
            .iter_mut()
            .zip(&self.value_scales[first_column..])
            .zip(&self.coefficient_errors[first_column..])
        {
            *band = scale * (series_band + unit_error) + error * (unit_scale + unit_error);
        }
        // Forming `K − m_j m_k`: the mean product and the difference, one rounding each.
        let difference_band = accumulation_growth(2);
        let unit_mean = self.means[unit];
        let unit_mean_band = self.mean_bands[unit];
        for &other in direct.iter() {
            let local = other - first_column;
            let rounding = covariance_rounding(other);
            if !rounding.is_finite() {
                return Err(PairRouteError::NonFinite {
                    context: "pair row covariance rounding",
                });
            }
            if rounding < 0.0 {
                return Err(PairRouteError::NegativeBound {
                    context: "pair row covariance rounding",
                });
            }
            let kernel = pair_kernel(
                self.activation,
                PreactivationPair {
                    mean_x: self.biases[unit],
                    mean_y: self.biases[other],
                    variance_x: self.variances[unit],
                    variance_y: self.variances[other],
                    covariance: covariances[local],
                    covariance_rounding: rounding,
                },
            )
            .map_err(|error| PairRouteError::Pair { unit, other, error })?;
            let other_mean = self.means[other];
            let other_mean_band = self.mean_bands[other];
            let mean_product = unit_mean * other_mean;
            values[local] = kernel.value - mean_product;
            slopes[local] = kernel.covariance_derivative;
            bands[local] = kernel.value_rounding
                + difference_band * (kernel.value.abs() + mean_product.abs())
                + unit_mean.abs() * other_mean_band
                + unit_mean_band * (other_mean.abs() + other_mean_band);
        }
        let mut value = 0.0;
        let mut absolute = 0.0;
        let mut band = 0.0;
        if gradient {
            for (((weight, &pair_value), &slope), &pair_band) in weights
                .iter_mut()
                .zip(values.iter())
                .zip(slopes.iter())
                .zip(bands.iter())
            {
                let metric_product = *weight;
                let term = metric_product * pair_value;
                value += term;
                absolute += term.abs();
                band += metric_product.abs() * pair_band;
                *weight = metric_product * slope;
            }
        } else {
            for ((&metric_product, &pair_value), &pair_band) in
                weights.iter().zip(values.iter()).zip(bands.iter())
            {
                let term = metric_product * pair_value;
                value += term;
                absolute += term.abs();
                band += metric_product.abs() * pair_band;
            }
        }
        Ok(PairRowSum {
            value,
            band: band + accumulation_growth(count) * absolute,
        })
    }

    fn value_tail(&self, unit: usize, order: usize) -> f64 {
        self.value_tails[order * self.width + unit]
    }

    fn slope_tail(&self, unit: usize, order: usize) -> f64 {
        self.slope_tails[order * self.width + unit]
    }

    /// Whether order `order` meets the value bound, and with `gradient` the slope bound, for every chaos pair of
    /// `unit`'s row whose correlation is at most `correlation`.
    fn meets_bounds(&self, unit: usize, order: usize, correlation: f64, gradient: bool) -> bool {
        let band = accumulation_growth(chaos_operations(order));
        let value =
            correlation.powi(order as i32 + 1) * self.value_tail(unit, order) * self.value_envelope[order];
        let slope = correlation.powi(order as i32) * self.slope_tail(unit, order) * self.slope_envelope[order];
        value <= band && (!gradient || slope <= band)
    }

    /// The first order whose bounds a row with largest chaos correlation `largest` meets. The route radius
    /// guarantees the table's own order does.
    fn row_order(&self, unit: usize, largest: f64, gradient: bool) -> usize {
        (1..self.order)
            .find(|&order| self.meets_bounds(unit, order, largest, gradient))
            .unwrap_or(self.order)
    }

    /// The largest correlation whose value and slope bounds the table's order meets for `unit`, checked after the
    /// root is rounded.
    fn route_radius(&self, unit: usize) -> f64 {
        let order = self.order;
        let band = accumulation_growth(chaos_operations(order));
        let value_product = self.value_tail(unit, order) * self.value_envelope[order];
        let slope_product = self.slope_tail(unit, order) * self.slope_envelope[order];
        let mut candidate = radius(band, value_product, order + 1).min(radius(band, slope_product, order));
        while candidate > 0.0 && !self.meets_bounds(unit, order, candidate, true) {
            candidate *= 1.0 - f64::EPSILON;
        }
        candidate
    }
}

fn relative(tail: f64, scale: f64) -> f64 {
    if scale > 0.0 { tail / scale } else { 0.0 }
}

/// The largest of each order's `width` tails.
fn envelope(tails: &[f64], width: usize, order: usize) -> Vec<f64> {
    (0..=order)
        .map(|degree| {
            tails[degree * width..(degree + 1) * width]
                .iter()
                .fold(0.0f64, |largest, &tail| largest.max(tail))
        })
        .collect()
}

/// The largest `ρ ≤ 1` with `ρ^exponent · product ≤ band`.
fn radius(band: f64, product: f64, exponent: usize) -> f64 {
    if product <= band {
        1.0
    } else {
        (band / product).powf((exponent as f64).recip())
    }
}

/// Every order of one row by Horner: `values[i] = Σ_{n≤N} ρ_iⁿ a_{j,n} a_{k,n}` and, with `gradient`,
/// `slopes[i] = Σ_{n≤N} n ρ_iⁿ⁻¹ a_{j,n} a_{k,n}` for `k = first + i`, where `N = unit_coefficients.len()` and `columns`
/// holds each order's coefficients over all units at `stride`, on the CPU-feature variant the running machine supports.
#[inline]
fn chaos_orders(
    unit_coefficients: &[f64],
    columns: &[f64],
    stride: usize,
    first: usize,
    correlations: &[f64],
    values: &mut [f64],
    slopes: &mut [f64],
    gradient: bool,
) {
    #[cfg(target_arch = "x86_64")]
    if avx2_available() {
        // SAFETY: `avx2_available` is the cached CPU probe for exactly the `avx2` feature this variant enables.
        unsafe {
            chaos_orders_avx2(unit_coefficients, columns, stride, first, correlations, values, slopes, gradient)
        };
        return;
    }
    chaos_orders_body(unit_coefficients, columns, stride, first, correlations, values, slopes, gradient);
}

/// Whether the running x86_64 CPU executes the `avx2` variant. `is_x86_feature_detected!` caches its probe in a
/// process-wide static, so this is a load and a bit test per row.
#[cfg(target_arch = "x86_64")]
#[inline]
fn avx2_available() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

/// [`chaos_orders_body`] compiled with the `avx2` target feature. The body never fuses a product into a sum, so the
/// wider lanes write the portable body's words.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn chaos_orders_avx2(
    unit_coefficients: &[f64],
    columns: &[f64],
    stride: usize,
    first: usize,
    correlations: &[f64],
    values: &mut [f64],
    slopes: &mut [f64],
    gradient: bool,
) {
    chaos_orders_body(unit_coefficients, columns, stride, first, correlations, values, slopes, gradient);
}

/// Horner from the highest order down. While it runs, `values` carries `Q = Σ_{n≤N} a_{j,n} a_{k,n} ρⁿ⁻¹` and `slopes`
/// its derivative `Q'`; the closing pass writes `P = ρ Q` and `P' = Q + ρ Q'`.
#[inline]
fn chaos_orders_body(
    unit_coefficients: &[f64],
    columns: &[f64],
    stride: usize,
    first: usize,
    correlations: &[f64],
    values: &mut [f64],
    slopes: &mut [f64],
    gradient: bool,
) {
    let width = correlations.len();
    values.fill(0.0);
    slopes.fill(0.0);
    if width == 0 {
        return;
    }
    let order = unit_coefficients.len();
    for (&unit_coefficient, column) in unit_coefficients
        .iter()
        .zip(columns[..order * stride].chunks_exact(stride))
        .rev()
    {
        let column = &column[first..first + width];
        if gradient {
            for (((value, slope), &coefficient), &correlation) in values
                .iter_mut()
                .zip(slopes.iter_mut())
                .zip(column)
                .zip(correlations)
            {
                *slope = *slope * correlation + *value;
                *value = *value * correlation + unit_coefficient * coefficient;
            }
        } else {
            for ((value, &coefficient), &correlation) in values.iter_mut().zip(column).zip(correlations) {
                *value = *value * correlation + unit_coefficient * coefficient;
            }
        }
    }
    if gradient {
        for ((value, slope), &correlation) in values.iter_mut().zip(slopes.iter_mut()).zip(correlations) {
            *slope = *value + correlation * *slope;
            *value *= correlation;
        }
    } else {
        for (value, &correlation) in values.iter_mut().zip(correlations) {
            *value *= correlation;
        }
    }
}

fn require_length(context: &'static str, expected: usize, got: usize) -> Result<(), PairRouteError> {
    if expected == got {
        Ok(())
    } else {
        Err(PairRouteError::DimensionMismatch {
            context,
            expected,
            got,
        })
    }
}

fn require_finite<'a>(
    context: &'static str,
    mut values: impl Iterator<Item = &'a f64>,
) -> Result<(), PairRouteError> {
    if values.all(|value| value.is_finite()) {
        Ok(())
    } else {
        Err(PairRouteError::NonFinite { context })
    }
}

#[cfg(test)]
#[path = "tiles_tests.rs"]
mod tiles_tests;
