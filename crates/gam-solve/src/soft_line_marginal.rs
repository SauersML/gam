//! The soft-line marginal: the Laplace criterion with its softest inner coordinate integrated exactly (#2979).
//!
//! # Why the Gaussian step fails at a fold
//! The single-mode Laplace criterion `V_L = f(β̂) + ½ln|H/2π|` treats every inner coordinate as Gaussian. At an
//! inner-mode fold (#2979), the tracked minimum meets a saddle and the pair annihilates. The softest curvature then
//! runs as `σ ∝ √(ρ* − ρ)`, so `∂_ρ(½ln σ) ∝ 1/σ²` diverges: the criterion and its gradient are not finite there.
//! Nothing about the likelihood is singular. Only the Gaussian step along the collapsing direction is wrong.
//!
//! # The criterion here
//! Along a line `s` in the inner space, the other coordinates keep their Laplace treatment. Each point of the line
//! (a *slice*) contributes a cost
//!
//! `q(s; ω) = f(β*(s)) + ½ln|K(s)/2π| + (the slice's cone term)`,
//!
//! where `β*(s)` minimizes `f` over the complement of the line through `s`, `K(s)` is the complement's curvature and
//! the cone term is the slice's own normalizer. The line itself is integrated exactly:
//!
//! `T(ω) = −ln ∫_{a(ω)}^{b(ω)} e^{−q(s; ω)} ds`.
//!
//! Away from folds `q` is sharply peaked, and `T = V_L + O(n⁻¹)`. Through a fold `T` stays finite and smooth,
//! because the integral keeps all the mass the Gaussian step loses.
//!
//! # Derivatives
//! With `p(s) = e^{−q}/∫e^{−q}` the slice density and `E` its expectation, the Leibniz rule gives
//!
//! - `T_i = E[q_i] − p(b)b_i + p(a)a_i`,
//! - `T_ij = E[q_ij] − E[q_iq_j] + T_iT_j − p(b)[b_ij − q_i(b)b_j − q_j(b)b_i − q_s(b)b_ib_j] + p(a)[the same at a]`.
//!
//! The kernel carries the integrals in scaled form, `g = e^{−(q − R)}` for a reference `R`:
//! `I_0 = ∫g`, `I_i = ∫q_ig` and `I_ij = ∫(q_iq_j − q_ij)g`. With
//! `Z_i = −I_i + g(b)b_i − g(a)a_i` and `Z_ij = I_ij + g(b)[b_ij − q_ib_j − q_jb_i − q_sb_ib_j] − g(a)[the same at a]`,
//!
//! `T = R − ln I_0`, `T_i = −Z_i/I_0` and `T_ij = −Z_ij/I_0 + T_iT_j`.
//!
//! These are the derivatives of `T` itself. They do not come from differentiating a quadrature rule.
//!
//! # Integration
//! Each piece of the interval is charted onto `x ∈ [0, 1]`:
//! - A bounded interval is affine: `s = a + (b − a)x`.
//! - A half-line from `e` is a ray: `s = e ± h(1 − x)/x`, with `|ds/dx| = h/x²`. The scale `h` sets only where the
//!   work goes. It does not change the answer. The open end sits at `x = 0`, so a node near it is resolved relative
//!   to its own distance from it: the chart coordinate rounds by `u·x`, not by `u`, and the band stays a rounding
//!   band however far the panels reach toward the open end.
//!
//! The chart is covered by panels, each integrated by the certified Gauss–Legendre rule
//! ([`gauss_legendre_certified`]) over the whole panel and over its two halves. A panel's discretization error is
//! estimated by the difference between the two. A child panel reuses its parent's half-rule as its whole rule.
//!
//! Integration stops when, for every component `c` of `(I_0, I_i, I_ij)`, the summed error estimates sit inside
//! the summed rounding bands: `Σ|err_c| ≤ Σ band_c`. This is the point past which bisection cannot improve the
//! result in `f64`. The stop has no tolerance of its own. Until it is reached, the panel holding the largest share
//! of a failing component's error is bisected.
//!
//! A panel that cannot be bisected in `f64` before it resolves is refused:
//! - [`SoftLineError::ImproperSoftLine`] when it touches the open end of a ray, so the integral does not converge.
//!   The same refusal covers a panel there whose chart point or contribution leaves the range of `f64`;
//! - [`SoftLineError::UnresolvedPanel`] otherwise.
//!
//! # Rounding bands
//! Each node's contribution carries a relative band made of:
//! - the slice's own cost band;
//! - `u(|q| + |R|)` for the shifted exponent, and `2u` for a faithfully rounded `exp`;
//! - the certified rule's weight error;
//! - the certified node error mapped through the log-integrand's slope, `(|q_s||ds/dx| + |dln|ds/dx|/dx|)`;
//! - the rounding of the chart point, `|q_s|·δs`;
//! - `γ_4` for the jacobian and `γ_4` for the weight product.
//!
//! Two absolute charges are added:
//! - each component's own rounding;
//! - its slope along the panel times the node's position error. The slope is estimated from neighbouring nodes. On
//!   a resolved panel that is as accurate as the whole-versus-halves error estimate the stop already relies on.
//!
//! Sums carry `γ_n·Σ|terms|`.
//!
//! # Stationary slicing
//! The line and the reference slicing may depend on coordinates `v` of `ω` that are not model hyperparameters. When
//! `T` is made stationary in `v`, the envelope theorem gives the gradient and the Schur complement gives the Hessian
//! in the remaining coordinates `θ`. [`reduce_to_stationary_slicing`] forms both.
//!
//! It refuses to form them when:
//! - `T_v` is resolved from zero, so the slicing was not stationary;
//! - an eigenvalue of `T_vv` is not resolved from zero. `T_vv` need not be definite; any resolved spectrum is
//!   accepted.
//!
//! How to settle a flat slicing direction is left to the caller. The refusal carries the eigenvalue and its band.

use faer::Side;
use gam_linalg::faer_ndarray::strict_symmetric_eigh;
use gam_math::roundoff::{UNIT_ROUNDOFF, accumulation_growth, inflated};
use gam_math::special::{CertifiedGaussLegendreRule, gauss_legendre_certified};
use ndarray::{Array1, Array2};

/// Order of the panel rule.
///
/// It sets the work per panel only. The certified value does not depend on it, because every panel is refined until
/// its discretization error sits inside its rounding band.
const PANEL_RULE_ORDER: usize = 8;

/// One slice of the soft line at `s`: its cost `q(s; ω)` and the derivatives the marginal needs.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftLineSlice {
    /// `q(s; ω)`.
    pub cost: f64,
    /// An absolute bound on the error in `cost`.
    pub cost_band: f64,
    /// `∂q/∂s` at fixed `ω`.
    pub cost_s: f64,
    /// `∂q/∂ω_i` at fixed `s`.
    pub cost_gradient: Array1<f64>,
    /// `∂²q/∂ω_i∂ω_j` at fixed `s`.
    pub cost_hessian: Array2<f64>,
    /// An absolute bound on the error in each entry of `cost_s`, `cost_gradient` and `cost_hessian`.
    pub derivative_band: f64,
}

/// A finite end of the soft line, which may move with `ω`.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftLineEnd {
    /// The end's position.
    pub at: f64,
    /// Its gradient in `ω`.
    pub gradient: Array1<f64>,
    /// Its Hessian in `ω`.
    pub hessian: Array2<f64>,
}

/// The extent of the soft line.
///
/// Each `scale` is the length scale of a ray chart. It must be positive and finite. It changes only where the panels
/// fall, not the certified value.
#[derive(Clone, Debug, PartialEq)]
pub enum SoftLineInterval {
    /// `[lower, upper]`.
    Bounded { lower: SoftLineEnd, upper: SoftLineEnd },
    /// `[lower, ∞)`.
    LowerBounded { lower: SoftLineEnd, scale: f64 },
    /// `(−∞, upper]`.
    UpperBounded { upper: SoftLineEnd, scale: f64 },
    /// `(−∞, ∞)`, charted as two rays from `centre`. `centre` is a fixed split and not an end.
    Unbounded { centre: f64, scale: f64 },
}

/// The soft-line marginal `T` with its gradient, its Hessian and their bands.
#[derive(Clone, Debug, PartialEq)]
pub struct SoftLineMarginal {
    pub value: f64,
    pub value_band: f64,
    pub gradient: Array1<f64>,
    pub gradient_band: Array1<f64>,
    pub hessian: Array2<f64>,
    pub hessian_band: Array2<f64>,
    /// The number of slices evaluated, the ends included.
    pub slice_evaluations: usize,
}

/// Why a soft-line marginal was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum SoftLineError<E> {
    /// The slice evaluation failed.
    Slice(E),
    /// A slice at `at` had a non-finite entry or a negative band.
    NonFiniteSlice { at: f64 },
    /// A slice or an end did not have `dimension` coordinates.
    DimensionMismatch { expected: usize, found: usize },
    /// A bounded interval with `upper ≤ lower`.
    EmptyInterval { lower: f64, upper: f64 },
    /// A ray scale that is not positive and finite.
    NonPositiveScale { scale: f64 },
    /// A panel on `[lower, upper]` in `s` that stopped bisecting in `f64` before it resolved.
    UnresolvedPanel { lower: f64, upper: f64 },
    /// A panel touching the open end of a ray, starting at `from` in `s`, that never resolved. The integral does not
    /// converge there.
    ImproperSoftLine { from: f64 },
}

impl<E: std::fmt::Display> std::fmt::Display for SoftLineError<E> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Slice(error) => write!(formatter, "soft-line slice evaluation failed: {error}"),
            Self::NonFiniteSlice { at } => write!(formatter, "soft-line slice at s = {at} is not finite"),
            Self::DimensionMismatch { expected, found } => {
                write!(formatter, "soft-line slice has {found} coordinates, expected {expected}")
            }
            Self::EmptyInterval { lower, upper } => {
                write!(formatter, "soft-line interval [{lower}, {upper}] is empty")
            }
            Self::NonPositiveScale { scale } => write!(formatter, "soft-line ray scale {scale} is not positive"),
            Self::UnresolvedPanel { lower, upper } => write!(
                formatter,
                "soft-line panel [{lower}, {upper}] stopped bisecting before its error reached its rounding band"
            ),
            Self::ImproperSoftLine { from } => {
                write!(formatter, "soft-line integral does not converge on the ray from s = {from}")
            }
        }
    }
}

impl<E: std::fmt::Debug + std::fmt::Display> std::error::Error for SoftLineError<E> {}

/// The number of integrated components for `dimension` coordinates: `I_0`, the `I_i` and the upper triangle of
/// `I_ij`.
fn component_count(dimension: usize) -> usize {
    1 + dimension + dimension * (dimension + 1) / 2
}

/// The component index of `I_ij`, in row-major order over the upper triangle.
fn pair_index(dimension: usize, i: usize, j: usize) -> usize {
    let (row, column) = if i <= j { (i, j) } else { (j, i) };
    1 + dimension + row * dimension - row * (row + 1) / 2 + column
}

#[derive(Clone, Copy, Debug)]
enum Chart {
    Affine { lower: f64, width: f64 },
    Ray { origin: f64, scale: f64, direction: f64 },
}

struct ChartPoint {
    s: f64,
    /// `|ds/dx|`.
    jacobian: f64,
    /// `|d ln|ds/dx| / dx|`.
    log_jacobian_slope: f64,
    /// A bound on the rounding error of `s`.
    position_rounding: f64,
}

impl Chart {
    fn point(self, x: f64) -> ChartPoint {
        match self {
            Chart::Affine { lower, width } => {
                let offset = width * x;
                ChartPoint {
                    s: lower + offset,
                    jacobian: width,
                    log_jacobian_slope: 0.0,
                    position_rounding: accumulation_growth(2) * (lower.abs() + offset.abs()),
                }
            }
            Chart::Ray { origin, scale, direction } => {
                let offset = scale * ((1.0 - x) / x);
                ChartPoint {
                    s: origin + direction * offset,
                    jacobian: scale / (x * x),
                    log_jacobian_slope: 2.0 / x,
                    position_rounding: accumulation_growth(4) * (origin.abs() + offset.abs()),
                }
            }
        }
    }

    fn s_at(self, x: f64) -> f64 {
        match self {
            Chart::Affine { lower, width } => lower + width * x,
            Chart::Ray { origin, scale, direction } if x > 0.0 => origin + direction * scale * ((1.0 - x) / x),
            Chart::Ray { direction, .. } => direction * f64::INFINITY,
        }
    }

    /// Whether a panel starting at `lower` touches the open end of a ray.
    fn open_below(self, lower: f64) -> bool {
        matches!(self, Chart::Ray { .. }) && lower == 0.0
    }

    /// The refusal for a panel on `[lower, upper]` that cannot be resolved in `f64`.
    fn refusal<E>(self, lower: f64, upper: f64) -> SoftLineError<E> {
        if self.open_below(lower) {
            SoftLineError::ImproperSoftLine { from: self.s_at(upper) }
        } else {
            SoftLineError::UnresolvedPanel { lower: self.s_at(lower), upper: self.s_at(upper) }
        }
    }
}

/// A rule's scaled sums `Σ w·g·jac·component` over one panel, with `g = e^{−(q − reference)}`.
#[derive(Clone, Debug)]
struct RuleSum {
    reference: f64,
    sums: Vec<f64>,
    bands: Vec<f64>,
}

impl RuleSum {
    /// The sums and bands rescaled to a reference `target ≤ self.reference`.
    fn rescaled(&self, target: f64) -> (Vec<f64>, Vec<f64>) {
        let factor = (-(self.reference - target)).exp();
        let factor_relative = UNIT_ROUNDOFF * (self.reference.abs() + target.abs()) + 3.0 * UNIT_ROUNDOFF;
        let sums: Vec<f64> = self.sums.iter().map(|&sum| sum * factor).collect();
        let bands = self
            .bands
            .iter()
            .zip(&sums)
            .map(|(&band, &sum)| inflated(band * factor + sum.abs() * factor_relative, 1))
            .collect();
        (sums, bands)
    }
}

/// A panel with its whole rule and its two half rules, compared at a common reference.
#[derive(Clone, Debug)]
struct Panel {
    chart: usize,
    lower: f64,
    upper: f64,
    left: RuleSum,
    right: RuleSum,
    reference: f64,
    /// The refined value, the sum of the halves.
    value: Vec<f64>,
    /// `|whole − halves|`, the discretization error estimate.
    error: Vec<f64>,
    /// The rounding band of the comparison.
    band: Vec<f64>,
}

impl Panel {
    fn settle(chart: usize, lower: f64, upper: f64, whole: &RuleSum, left: RuleSum, right: RuleSum) -> Self {
        let reference = whole.reference.min(left.reference).min(right.reference);
        let (whole_sums, whole_bands) = whole.rescaled(reference);
        let (left_sums, left_bands) = left.rescaled(reference);
        let (right_sums, right_bands) = right.rescaled(reference);
        let components = whole_sums.len();
        let mut value = Vec::with_capacity(components);
        let mut error = Vec::with_capacity(components);
        let mut band = Vec::with_capacity(components);
        for c in 0..components {
            let refined = left_sums[c] + right_sums[c];
            value.push(refined);
            error.push((whole_sums[c] - refined).abs());
            let magnitude = whole_sums[c].abs() + left_sums[c].abs() + right_sums[c].abs();
            band.push(inflated(
                whole_bands[c] + left_bands[c] + right_bands[c] + accumulation_growth(2) * magnitude,
                4,
            ));
        }
        Panel { chart, lower, upper, left, right, reference, value, error, band }
    }
}

struct Integrator<'a, F> {
    slice: &'a mut F,
    rule: CertifiedGaussLegendreRule,
    dimension: usize,
    evaluations: usize,
}

impl<F> Integrator<'_, F> {
    fn evaluate<E>(&mut self, s: f64) -> Result<SoftLineSlice, SoftLineError<E>>
    where
        F: FnMut(f64) -> Result<SoftLineSlice, E>,
    {
        self.evaluations += 1;
        let slice = (self.slice)(s).map_err(SoftLineError::Slice)?;
        let d = self.dimension;
        if slice.cost_gradient.len() != d {
            return Err(SoftLineError::DimensionMismatch { expected: d, found: slice.cost_gradient.len() });
        }
        if slice.cost_hessian.dim() != (d, d) {
            return Err(SoftLineError::DimensionMismatch { expected: d, found: slice.cost_hessian.nrows() });
        }
        let finite = slice.cost.is_finite()
            && slice.cost_s.is_finite()
            && slice.cost_band.is_finite()
            && slice.cost_band >= 0.0
            && slice.derivative_band.is_finite()
            && slice.derivative_band >= 0.0
            && slice.cost_gradient.iter().all(|value| value.is_finite())
            && slice.cost_hessian.iter().all(|value| value.is_finite());
        if !finite {
            return Err(SoftLineError::NonFiniteSlice { at: s });
        }
        Ok(slice)
    }

    /// The certified rule on `[lower, upper]` of `chart`.
    fn rule_sum<E>(&mut self, chart: Chart, lower: f64, upper: f64) -> Result<RuleSum, SoftLineError<E>>
    where
        F: FnMut(f64) -> Result<SoftLineSlice, E>,
    {
        let unit = UNIT_ROUNDOFF;
        let d = self.dimension;
        let components = component_count(d);
        let order = self.rule.nodes.len();
        let centre = 0.5 * (lower + upper);
        let half = 0.5 * (upper - lower);
        let mut xs = Vec::with_capacity(order);
        let mut position_errors = Vec::with_capacity(order);
        let mut points = Vec::with_capacity(order);
        let mut slices = Vec::with_capacity(order);
        for index in 0..order {
            let offset = half * self.rule.nodes[index];
            let x = centre + offset;
            xs.push(x);
            position_errors.push(half * self.rule.node_error + accumulation_growth(2) * (centre.abs() + offset.abs()));
            let point = chart.point(x);
            if !(point.s.is_finite() && point.jacobian.is_finite()) {
                return Err(chart.refusal(lower, upper));
            }
            slices.push(self.evaluate(point.s)?);
            points.push(point);
        }
        let reference = slices.iter().map(|slice| slice.cost).fold(f64::INFINITY, f64::min);
        // The integrand's components and their own rounding, node by node.
        let mut values = vec![vec![0.0; components]; order];
        let mut roundings = vec![vec![0.0; components]; order];
        for (node, slice) in slices.iter().enumerate() {
            values[node][0] = 1.0;
            for i in 0..d {
                values[node][1 + i] = slice.cost_gradient[i];
                roundings[node][1 + i] = slice.derivative_band;
                for j in i..d {
                    let product = slice.cost_gradient[i] * slice.cost_gradient[j];
                    let curvature = slice.cost_hessian[[i, j]];
                    let index = pair_index(d, i, j);
                    values[node][index] = product - curvature;
                    roundings[node][index] = accumulation_growth(2) * (product.abs() + curvature.abs())
                        + slice.derivative_band
                            * (slice.cost_gradient[i].abs() + slice.cost_gradient[j].abs() + 1.0);
                }
            }
        }
        let mut sums = vec![0.0; components];
        let mut magnitudes = vec![0.0; components];
        let mut bands = vec![0.0; components];
        for node in 0..order {
            let slice = &slices[node];
            let point = &points[node];
            let position_error = position_errors[node];
            let factor = self.rule.weights[node] * half * point.jacobian * (-(slice.cost - reference)).exp();
            let slope = slice.cost_s.abs() + slice.derivative_band;
            let relative = slice.cost_band
                + unit * (slice.cost.abs() + reference.abs())
                + 2.0 * unit
                + self.rule.weight_relative_error
                + (slope * point.jacobian + point.log_jacobian_slope) * position_error
                + slope * point.position_rounding
                + 2.0 * accumulation_growth(4);
            for c in 0..components {
                // The component's slope along x, from its neighbouring nodes.
                let mut component_slope: f64 = 0.0;
                for neighbour in [node.wrapping_sub(1), node + 1] {
                    if neighbour < order {
                        let run = (xs[neighbour] - xs[node]).abs();
                        component_slope = component_slope.max((values[neighbour][c] - values[node][c]).abs() / run);
                    }
                }
                let term = factor * values[node][c];
                sums[c] += term;
                magnitudes[c] += term.abs();
                bands[c] += term.abs() * relative + factor * (roundings[node][c] + component_slope * position_error);
            }
        }
        let bands: Vec<f64> = bands
            .iter()
            .zip(&magnitudes)
            .map(|(&band, &magnitude)| inflated(band + accumulation_growth(order) * magnitude, 4 * order))
            .collect();
        if !(sums.iter().all(|sum| sum.is_finite()) && bands.iter().all(|band| band.is_finite())) {
            return Err(chart.refusal(lower, upper));
        }
        Ok(RuleSum { reference, sums, bands })
    }

    fn panel<E>(&mut self, charts: &[Chart], chart: usize, lower: f64, upper: f64, whole: &RuleSum)
    -> Result<Panel, SoftLineError<E>>
    where
        F: FnMut(f64) -> Result<SoftLineSlice, E>,
    {
        let middle = 0.5 * (lower + upper);
        if !(lower < middle && middle < upper) {
            return Err(charts[chart].refusal(lower, upper));
        }
        let left = self.rule_sum(charts[chart], lower, middle)?;
        let right = self.rule_sum(charts[chart], middle, upper)?;
        Ok(Panel::settle(chart, lower, upper, whole, left, right))
    }
}

/// The value, error estimate and band of every component over all panels, at the lowest panel reference.
struct Totals {
    reference: f64,
    factors: Vec<f64>,
    value: Vec<f64>,
    error: Vec<f64>,
    band: Vec<f64>,
}

fn totals(panels: &[Panel], components: usize) -> Totals {
    let reference = panels.iter().map(|panel| panel.reference).fold(f64::INFINITY, f64::min);
    let mut factors = Vec::with_capacity(panels.len());
    let mut value = vec![0.0; components];
    let mut magnitude = vec![0.0; components];
    let mut error = vec![0.0; components];
    let mut band = vec![0.0; components];
    for panel in panels {
        let factor = (-(panel.reference - reference)).exp();
        let factor_relative = UNIT_ROUNDOFF * (panel.reference.abs() + reference.abs()) + 3.0 * UNIT_ROUNDOFF;
        factors.push(factor);
        for c in 0..components {
            let scaled = panel.value[c] * factor;
            value[c] += scaled;
            magnitude[c] += scaled.abs();
            error[c] += panel.error[c] * factor;
            band[c] += panel.band[c] * factor + scaled.abs() * factor_relative;
        }
    }
    let count = panels.len();
    for c in 0..components {
        band[c] = inflated(band[c] + accumulation_growth(count) * magnitude[c], 2 * count);
    }
    Totals { reference, factors, value, error, band }
}

/// The soft-line marginal `T(ω) = −ln ∫ e^{−q(s; ω)} ds` over `interval`, with its gradient and Hessian in the
/// `dimension` coordinates of `ω`. `slice(s)` evaluates the slice at `s`. See the module documentation.
pub fn soft_line_marginal<E, F>(
    interval: &SoftLineInterval,
    dimension: usize,
    mut slice: F,
) -> Result<SoftLineMarginal, SoftLineError<E>>
where
    F: FnMut(f64) -> Result<SoftLineSlice, E>,
{
    let d = dimension;
    let check_end = |end: &SoftLineEnd| -> Result<(), SoftLineError<E>> {
        if end.gradient.len() != d {
            return Err(SoftLineError::DimensionMismatch { expected: d, found: end.gradient.len() });
        }
        if end.hessian.dim() != (d, d) {
            return Err(SoftLineError::DimensionMismatch { expected: d, found: end.hessian.nrows() });
        }
        if !(end.at.is_finite()
            && end.gradient.iter().all(|value| value.is_finite())
            && end.hessian.iter().all(|value| value.is_finite()))
        {
            return Err(SoftLineError::NonFiniteSlice { at: end.at });
        }
        Ok(())
    };
    let check_scale = |scale: f64| -> Result<(), SoftLineError<E>> {
        if scale.is_finite() && scale > 0.0 { Ok(()) } else { Err(SoftLineError::NonPositiveScale { scale }) }
    };
    // Charts, and the moving ends with their signs: `+1` for an upper end and `−1` for a lower one.
    let (charts, ends): (Vec<Chart>, Vec<(&SoftLineEnd, f64)>) = match interval {
        SoftLineInterval::Bounded { lower, upper } => {
            check_end(lower)?;
            check_end(upper)?;
            if !(upper.at > lower.at) {
                return Err(SoftLineError::EmptyInterval { lower: lower.at, upper: upper.at });
            }
            let width = upper.at - lower.at;
            (vec![Chart::Affine { lower: lower.at, width }], vec![(lower, -1.0), (upper, 1.0)])
        }
        SoftLineInterval::LowerBounded { lower, scale } => {
            check_end(lower)?;
            check_scale(*scale)?;
            (vec![Chart::Ray { origin: lower.at, scale: *scale, direction: 1.0 }], vec![(lower, -1.0)])
        }
        SoftLineInterval::UpperBounded { upper, scale } => {
            check_end(upper)?;
            check_scale(*scale)?;
            (vec![Chart::Ray { origin: upper.at, scale: *scale, direction: -1.0 }], vec![(upper, 1.0)])
        }
        SoftLineInterval::Unbounded { centre, scale } => {
            check_scale(*scale)?;
            if !centre.is_finite() {
                return Err(SoftLineError::NonFiniteSlice { at: *centre });
            }
            (
                vec![
                    Chart::Ray { origin: *centre, scale: *scale, direction: 1.0 },
                    Chart::Ray { origin: *centre, scale: *scale, direction: -1.0 },
                ],
                Vec::new(),
            )
        }
    };
    let components = component_count(d);
    let mut integrator =
        Integrator { slice: &mut slice, rule: gauss_legendre_certified(PANEL_RULE_ORDER), dimension: d, evaluations: 0 };

    let mut panels = Vec::with_capacity(charts.len());
    for chart in 0..charts.len() {
        let whole = integrator.rule_sum(charts[chart], 0.0, 1.0)?;
        panels.push(integrator.panel(&charts, chart, 0.0, 1.0, &whole)?);
    }
    let totals = loop {
        let totals = totals(&panels, components);
        let failing: Vec<usize> = (0..components).filter(|&c| totals.error[c] > totals.band[c]).collect();
        if failing.is_empty() {
            break totals;
        }
        // The panel holding the largest share of a failing component's excess error.
        let mut worst = (0, f64::NEG_INFINITY);
        for (index, panel) in panels.iter().enumerate() {
            for &c in &failing {
                let share = (panel.error[c] - panel.band[c]) * totals.factors[index] / totals.error[c];
                if share > worst.1 {
                    worst = (index, share);
                }
            }
        }
        let parent = panels.swap_remove(worst.0);
        let middle = 0.5 * (parent.lower + parent.upper);
        let left = integrator.panel(&charts, parent.chart, parent.lower, middle, &parent.left)?;
        let right = integrator.panel(&charts, parent.chart, middle, parent.upper, &parent.right)?;
        panels.push(left);
        panels.push(right);
    };

    let reference = totals.reference;
    let unit = UNIT_ROUNDOFF;
    // The integrals' uncertainty: the discretization estimate plus the rounding band.
    let uncertainty: Vec<f64> = (0..components).map(|c| totals.error[c] + totals.band[c]).collect();
    let mass = totals.value[0];
    let mut z_gradient = Array1::<f64>::zeros(d);
    let mut z_gradient_band = Array1::<f64>::zeros(d);
    let mut z_hessian = Array2::<f64>::zeros((d, d));
    let mut z_hessian_band = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        z_gradient[i] = -totals.value[1 + i];
        z_gradient_band[i] = uncertainty[1 + i];
        for j in i..d {
            let index = pair_index(d, i, j);
            z_hessian[[i, j]] = totals.value[index];
            z_hessian_band[[i, j]] = uncertainty[index];
        }
    }
    for &(end, sign) in &ends {
        let at_end = integrator.evaluate(end.at)?;
        let weight = (-(at_end.cost - reference)).exp();
        let weight_relative = at_end.cost_band + unit * (at_end.cost.abs() + reference.abs()) + 2.0 * unit;
        let band = at_end.derivative_band;
        for i in 0..d {
            let term = weight * end.gradient[i];
            z_gradient[i] += sign * term;
            z_gradient_band[i] += term.abs() * (weight_relative + accumulation_growth(2));
            for j in i..d {
                let parts = [
                    end.hessian[[i, j]],
                    -at_end.cost_gradient[i] * end.gradient[j],
                    -at_end.cost_gradient[j] * end.gradient[i],
                    -at_end.cost_s * end.gradient[i] * end.gradient[j],
                ];
                let bracket: f64 = parts.iter().sum();
                let bracket_magnitude: f64 = parts.iter().map(|part| part.abs()).sum();
                let bracket_band = accumulation_growth(6) * bracket_magnitude
                    + band
                        * (end.gradient[i].abs()
                            + end.gradient[j].abs()
                            + (end.gradient[i] * end.gradient[j]).abs());
                z_hessian[[i, j]] += sign * weight * bracket;
                z_hessian_band[[i, j]] += (weight * bracket).abs() * (weight_relative + accumulation_growth(2))
                    + weight * bracket_band;
            }
        }
    }
    let mass_relative = uncertainty[0] / mass;
    let value = reference - mass.ln();
    let value_band = inflated(mass_relative + unit * (mass.ln().abs() + value.abs()), 2);
    let mut gradient = Array1::<f64>::zeros(d);
    let mut gradient_band = Array1::<f64>::zeros(d);
    for i in 0..d {
        gradient[i] = -z_gradient[i] / mass;
        gradient_band[i] =
            inflated((z_gradient_band[i] + gradient[i].abs() * uncertainty[0]) / mass + unit * gradient[i].abs(), 3);
    }
    let mut hessian = Array2::<f64>::zeros((d, d));
    let mut hessian_band = Array2::<f64>::zeros((d, d));
    for i in 0..d {
        for j in i..d {
            let scaled = z_hessian[[i, j]] / mass;
            let product = gradient[i] * gradient[j];
            let entry = -scaled + product;
            let band = inflated(
                (z_hessian_band[[i, j]] + scaled.abs() * uncertainty[0]) / mass
                    + gradient[i].abs() * gradient_band[j]
                    + gradient[j].abs() * gradient_band[i]
                    + accumulation_growth(3) * (scaled.abs() + product.abs()),
                6,
            );
            hessian[[i, j]] = entry;
            hessian[[j, i]] = entry;
            hessian_band[[i, j]] = band;
            hessian_band[[j, i]] = band;
        }
    }
    Ok(SoftLineMarginal {
        value,
        value_band,
        gradient,
        gradient_band,
        hessian,
        hessian_band,
        slice_evaluations: integrator.evaluations,
    })
}

/// The soft-line marginal reduced to the outer coordinates `θ`, at a slicing stationary in the trailing coordinates
/// `v` of `ω`.
#[derive(Clone, Debug, PartialEq)]
pub struct StationarySlicing {
    pub value: f64,
    pub value_band: f64,
    pub gradient: Array1<f64>,
    pub gradient_band: Array1<f64>,
    pub hessian: Array2<f64>,
    pub hessian_band: Array2<f64>,
}

/// Why a stationary-slicing reduction was refused.
#[derive(Clone, Debug, PartialEq)]
pub enum SlicingError {
    /// `outer_dimension` exceeds the marginal's dimension.
    DimensionMismatch { outer_dimension: usize, dimension: usize },
    /// `T_v` at slicing coordinate `coordinate` is resolved from zero.
    NotStationary { coordinate: usize, gradient: f64, band: f64 },
    /// An eigenvalue of `T_vv` is not resolved from zero. `coupling` is `‖T_θv u‖` along its eigenvector `u`.
    UnresolvedSlicingCurvature { eigenvalue: f64, band: f64, coupling: f64 },
    /// The eigendecomposition of `T_vv` failed, or its eigenvectors are not orthonormal to working precision.
    Eigendecomposition { reason: String },
}

impl std::fmt::Display for SlicingError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DimensionMismatch { outer_dimension, dimension } => write!(
                formatter,
                "stationary slicing: {outer_dimension} outer coordinates exceed the marginal's {dimension}"
            ),
            Self::NotStationary { coordinate, gradient, band } => write!(
                formatter,
                "stationary slicing: T_v[{coordinate}] = {gradient} is resolved from zero (band {band})"
            ),
            Self::UnresolvedSlicingCurvature { eigenvalue, band, coupling } => write!(
                formatter,
                "stationary slicing: T_vv eigenvalue {eigenvalue} is within its band {band} of zero (coupling \
                 {coupling})"
            ),
            Self::Eigendecomposition { reason } => write!(formatter, "stationary slicing: {reason}"),
        }
    }
}

impl std::error::Error for SlicingError {}

fn frobenius(matrix: &Array2<f64>) -> f64 {
    matrix.iter().map(|value| value * value).sum::<f64>().sqrt()
}

/// Reduces `marginal` to its first `outer_dimension` coordinates `θ`, with the trailing coordinates `v` a stationary
/// slicing: gradient `T_θ` (envelope) and Hessian `T_θθ − T_θv T_vv⁻¹ T_vθ` (Schur complement).
///
/// # Certificate for `T_vv`
/// With `Û, Λ̂` the computed eigendecomposition of the `k × k` block `M = T_vv`:
/// - `ρ ≥ ‖M − ÛΛ̂Ûᵀ‖_F`, measured, plus its own rounding;
/// - `ε ≥ ‖ÛᵀÛ − I‖_F`, measured, plus its own rounding.
///
/// By Ostrowski's theorem the eigenvalues of `ÛΛ̂Ûᵀ` are `λ̂_l·θ_l` with `θ_l ∈ [1 − ε, 1 + ε]`. By Weyl's
/// inequality those of the exact `M` lie within `ρ + ‖band_vv‖_F` of them, in sorted order. An eigenvalue is resolved
/// when `|λ̂_l|(1 − ε) > ρ + ‖band_vv‖_F`, and `λ_lo` is the smallest such lower bound.
///
/// # Bands
/// - The computed inverse `ÛΛ̂⁻¹Ûᵀ` differs from `M⁻¹` by at most
///   `δ/λ_lo² + (1 + ε)(2η + η²)/λ_lo`, with `δ = ρ + ‖band_vv‖_F` and `η = ε/(1 − ε)`, plus its rounding.
/// - The Schur complement's band adds `2‖C‖‖band_θv‖/λ_lo + ‖band_θv‖²/λ_lo + ‖C‖²·(that difference)` to
///   `band_θθ`, with `C = T_θv`, plus the rounding of the product.
/// - A slicing stationary only to its band leaves `|T_v| ≤ 2·band_v` in exact arithmetic. So the true stationary
///   point lies within `2‖band_v‖/λ_lo`, which moves the gradient by `|C_a|·2‖band_v‖/λ_lo` and the value by
///   `2‖band_v‖²/λ_lo`.
///
/// The Hessian's first-order response to that displacement needs `T`'s third derivative, which the marginal does not
/// carry. It is not in the band.
pub fn reduce_to_stationary_slicing(
    marginal: &SoftLineMarginal,
    outer_dimension: usize,
) -> Result<StationarySlicing, SlicingError> {
    let d = marginal.gradient.len();
    if outer_dimension > d {
        return Err(SlicingError::DimensionMismatch { outer_dimension, dimension: d });
    }
    let m = outer_dimension;
    let k = d - m;
    let slice = |array: &Array1<f64>| array.slice(ndarray::s![..m]).to_owned();
    let block = |array: &Array2<f64>, rows: std::ops::Range<usize>, columns: std::ops::Range<usize>| {
        array.slice(ndarray::s![rows, columns]).to_owned()
    };
    if k == 0 {
        return Ok(StationarySlicing {
            value: marginal.value,
            value_band: marginal.value_band,
            gradient: marginal.gradient.clone(),
            gradient_band: marginal.gradient_band.clone(),
            hessian: marginal.hessian.clone(),
            hessian_band: marginal.hessian_band.clone(),
        });
    }
    for coordinate in m..d {
        let (gradient, band) = (marginal.gradient[coordinate], marginal.gradient_band[coordinate]);
        if gradient.abs() > band {
            return Err(SlicingError::NotStationary { coordinate, gradient, band });
        }
    }
    let unit = UNIT_ROUNDOFF;
    let curvature = block(&marginal.hessian, m..d, m..d);
    let curvature_band = frobenius(&block(&marginal.hessian_band, m..d, m..d));
    let coupling = block(&marginal.hessian, 0..m, m..d);
    let coupling_band = frobenius(&block(&marginal.hessian_band, 0..m, m..d));
    let coupling_norm = frobenius(&coupling);
    // The marginal writes `hessian[[i, j]]` and `hessian[[j, i]]` from one
    // rounded entry, so a sub-block of it is bitwise symmetric too.
    let (eigenvalues, vectors) = strict_symmetric_eigh(
        &curvature,
        gam_linalg::roundoff::SymmetricAssembly::Mirrored,
        Side::Lower,
    )
        .map_err(|error| SlicingError::Eigendecomposition { reason: error.to_string() })?;
    // ρ and ε, each with the rounding of its own measurement.
    let mut residual = 0.0;
    let mut residual_rounding = 0.0;
    let mut orthogonality = 0.0;
    let mut orthogonality_rounding = 0.0;
    for a in 0..k {
        for b in 0..k {
            let mut reconstruction = 0.0;
            let mut reconstruction_magnitude = 0.0;
            let mut gram = 0.0;
            let mut gram_magnitude = 0.0;
            for l in 0..k {
                let term = vectors[[a, l]] * eigenvalues[l] * vectors[[b, l]];
                reconstruction += term;
                reconstruction_magnitude += term.abs();
                let product = vectors[[l, a]] * vectors[[l, b]];
                gram += product;
                gram_magnitude += product.abs();
            }
            let difference = curvature[[a, b]] - reconstruction;
            residual += difference * difference;
            let entry_rounding =
                accumulation_growth(k + 2) * reconstruction_magnitude + unit * (curvature[[a, b]].abs() + difference.abs());
            residual_rounding += entry_rounding * entry_rounding;
            let defect = gram - if a == b { 1.0 } else { 0.0 };
            orthogonality += defect * defect;
            let gram_rounding = accumulation_growth(k + 1) * gram_magnitude;
            orthogonality_rounding += gram_rounding * gram_rounding;
        }
    }
    let rho = inflated(residual.sqrt() + residual_rounding.sqrt(), 2 * k * k + 2);
    let epsilon = inflated(orthogonality.sqrt() + orthogonality_rounding.sqrt(), 2 * k * k + 2);
    if !(epsilon < 1.0) {
        return Err(SlicingError::Eigendecomposition {
            reason: format!("eigenvectors of T_vv are not orthonormal: ‖ÛᵀÛ − I‖ ≤ {epsilon}"),
        });
    }
    let delta = rho + curvature_band;
    let mut lowest = f64::INFINITY;
    for l in 0..k {
        let lower = eigenvalues[l].abs() * (1.0 - epsilon) - delta;
        if !(lower > 0.0) {
            let along: f64 =
                (0..m).map(|a| (0..k).map(|b| coupling[[a, b]] * vectors[[b, l]]).sum::<f64>().powi(2)).sum();
            return Err(SlicingError::UnresolvedSlicingCurvature {
                eigenvalue: eigenvalues[l],
                band: epsilon * eigenvalues[l].abs() + delta,
                coupling: along.sqrt(),
            });
        }
        lowest = lowest.min(lower);
    }
    // The computed inverse ÛΛ̂⁻¹Ûᵀ and its rounding.
    let mut inverse = Array2::<f64>::zeros((k, k));
    let mut inverse_rounding = 0.0;
    for a in 0..k {
        for b in 0..k {
            let mut entry = 0.0;
            let mut magnitude = 0.0;
            for l in 0..k {
                let term = vectors[[a, l]] * vectors[[b, l]] / eigenvalues[l];
                entry += term;
                magnitude += term.abs();
            }
            inverse[[a, b]] = entry;
            let rounding = accumulation_growth(k + 2) * magnitude;
            inverse_rounding += rounding * rounding;
        }
    }
    let eta = epsilon / (1.0 - epsilon);
    let inverse_error = inflated(
        delta / (lowest * lowest) + (1.0 + epsilon) * (2.0 * eta + eta * eta) / lowest + inverse_rounding.sqrt(),
        12,
    );
    let gain = coupling.dot(&inverse);
    let schur_spread = inflated(
        2.0 * coupling_norm * coupling_band / lowest
            + coupling_band * coupling_band / lowest
            + coupling_norm * coupling_norm * inverse_error,
        10,
    );
    let mut hessian = block(&marginal.hessian, 0..m, 0..m);
    let mut hessian_band = block(&marginal.hessian_band, 0..m, 0..m);
    for a in 0..m {
        for b in a..m {
            let mut correction = 0.0;
            let mut magnitude = 0.0;
            for l in 0..k {
                let term = gain[[a, l]] * coupling[[b, l]];
                correction += term;
                magnitude += term.abs();
            }
            let entry = hessian[[a, b]] - correction;
            let band = inflated(
                hessian_band[[a, b]]
                    + schur_spread
                    + accumulation_growth(2 * k + 1) * (magnitude + hessian[[a, b]].abs()),
                2,
            );
            hessian[[a, b]] = entry;
            hessian[[b, a]] = entry;
            hessian_band[[a, b]] = band;
            hessian_band[[b, a]] = band;
        }
    }
    let stationarity: f64 =
        marginal.gradient_band.slice(ndarray::s![m..]).iter().map(|band| band * band).sum::<f64>().sqrt();
    let displacement = 2.0 * stationarity / lowest;
    let mut gradient_band = slice(&marginal.gradient_band);
    for a in 0..m {
        let row: f64 = (0..k).map(|l| coupling[[a, l]] * coupling[[a, l]]).sum::<f64>().sqrt();
        gradient_band[a] = inflated(gradient_band[a] + row * displacement, 4);
    }
    Ok(StationarySlicing {
        value: marginal.value,
        value_band: inflated(marginal.value_band + 2.0 * stationarity * stationarity / lowest, 4),
        gradient: slice(&marginal.gradient),
        gradient_band,
        hessian,
        hessian_band,
    })
}

#[cfg(test)]
#[path = "soft_line_marginal_tests.rs"]
mod soft_line_marginal_tests;
