//! The fitted CTN conditional transform, tabulated — **with the derivative and
//! the affine tails it actually has**, so that inverting it is a statement about
//! the model rather than about the table.
//!
//! A conditional-transformation-normal model is `F(y | x) = Φ(h(y | x))` with
//! `h(·|x)` strictly increasing, so every response-scale question a fitted CTN
//! can be asked — the conditional mean `E[Y|x] = E_Z[h⁻¹(Z|x)]`, the predictive
//! quantile ladder `h⁻¹(Φ⁻¹(p)|x)`, an inverse-transform draw `h⁻¹(Z|x)` — is
//! one function, `h⁻¹`, evaluated at different latent arguments. This type is
//! that function, and it is the ONLY place it is implemented.
//!
//! # One rule, no special cases
//!
//! The table carries `(y_k, h(y_k), h'(y_k))` per row — the chart computes the
//! value and the derivative together at every node anyway — and the interpolant
//! is the cubic Hermite those three determine. The two exterior branches are
//! then not a separate convention: they are the same Hermite rule degenerating
//! to its end slopes, because since gam#2600 the CTN transformation really is
//! affine beyond the boundary knots at exactly `h'(y_lo)` and `h'(y_hi)`
//! (`ctn_response_bases_at` continues the I-spline value basis linearly there).
//!
//! # Why the tails are part of the object
//!
//! `h` is tabulated on the fitted response support `[y_lo, y_hi]`, a bounded
//! interval; the latent `Z` is not bounded. The tabulated rows therefore never
//! cover the whole latent axis: `h(y_lo|x)` and `h(y_hi|x)` are finite numbers
//! `L(x)`, `U(x)`, typically near `∓Φ⁻¹(1/(n+1))`, and every latent target
//! outside `[L, U]` — which is `Φ(L) + 1 − Φ(U)` of the predictive mass, *by the
//! model's own reckoning* — lands off the end of the table.
//!
//! Both inverters this type replaces answered such a target with the support
//! endpoint. That is a truncation the fitted likelihood does not perform: it
//! makes `y_lo` and `y_hi` atoms of the predictive law, pins every observation
//! band at the training range no matter how extreme the level, and biases
//! `E[Y|x]` inward. Measured on an intercept-only fit to `Y = exp(N(0,1))` at
//! `n = 256`, `Φ(L) + 1 − Φ(U) = 2.5e-2`, and the 2.3 %, 0.13 % and 0.003 %
//! predictive quantiles were the same number. gam#2600.
//!
//! Carrying the derivative alongside the values is what makes those tails
//! impossible for a consumer to forget: there is no way to hold this object and
//! not hold them.
//!
//! # Why cubic Hermite and not linear interpolation
//!
//! The interpolation error is not decoration either. A CTN transform is a
//! degree-`(response_degree + 1)` piecewise polynomial whose curvature is
//! largest exactly where the response is densest, so linear interpolation of a
//! `G`-node table carries an `O(Δy²·h'')` latent error — measured at `2.1e-3` on
//! the lognormal fixture above, against a reported ladder whose own step is
//! `0.125`. Matching the derivative as well as the value raises that to
//! `O(Δy⁴·h⁗)` at no extra evaluation cost, because `h'` was already computed at
//! every node and thrown away.
//!
//! The interpolant is used only where it is provably monotone: a cell whose end
//! slopes violate the Fritsch–Carlson bound relative to its own secant is
//! under-resolved for a cubic, and falls back to the linear chord on that cell
//! alone. Monotonicity is not a nicety here — it is what makes `invert` a
//! function.
//!
//! # Predictive expectations are integrals in `y`, not averages of `h⁻¹`
//!
//! The law this table defines is `F(y|x) = Φ(H(y|x))` with `H` the interpolant
//! above, so `E[g(Y)|x] = ∫ g(y)·φ(H(y))·H'(y) dy`, and that integral splits
//! along the table's own structure into pieces each of which can be done
//! exactly or to a certified accuracy:
//!
//! * **The two affine tails** in closed form. Below `y_lo` the latent is
//!   `L + γ_lo(y − y_lo)`, so the tail carries mass `Φ(L)` and first moment about
//!   its anchor `∫(y − y_lo) dF = −(φ(L) + L·Φ(L))/γ_lo`; above `y_hi`, mass
//!   `Q(U) = 1 − Φ(U)` and moment `(φ(U) − U·Q(U))/γ_hi`.
//! * **Every interior cell** by Gauss–Legendre in the cell's own coordinate. On
//!   a cell the integrand `y·φ(P(s))·P'(s)`, `P` the cubic (or chord) in
//!   `s ∈ [−1, 1]`, is entire, so the order is read off a bound rather than
//!   guessed: see [`CtnPredictiveQuadrature`].
//!
//! The rule this replaces averaged `h⁻¹` over 48 midpoint levels
//! `Φ⁻¹((k + ½)/48)`. `E[Y] = ∫₀¹ h⁻¹(Φ⁻¹(p)) dp` has an integrand that is
//! unbounded at both ends of `p`, so the midpoint rule is not even first-order
//! there: it dropped `−1.9 %` of the mean of a lognormal (gam#3520).

use gam_math::probability::{normal_cdf_and_pdf, normal_pdf};
use gam_math::special::gauss_legendre_certified;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// Fritsch–Carlson sufficient bound on `h'(y_k)/secant` for the cubic Hermite
/// interpolant of a cell to be monotone. The classical sufficient region is the
/// disc `α² + β² ≤ 9`; the box `α, β ∈ [0, 3]` is the standard conservative
/// inscription of it and is the one checked here, so a cell is only interpolated
/// by a cubic when that cubic is certainly increasing.
const HERMITE_MONOTONE_SLOPE_BOUND: f64 = 3.0;

/// A per-row tabulation of the fitted CTN transform `h(·|x_i)` and its
/// derivative on a shared response grid.
///
/// Invariants, all checked by [`CtnTransformTable::new`]:
/// * `grid_y` has `g ≥ 2` finite, strictly increasing entries;
/// * `h` is `n × g` and strictly increasing along every row;
/// * `h_prime` is `n × g`, finite and strictly positive everywhere.
///
/// Together these make [`CtnTransformTable::invert`] a total, strictly
/// increasing function of the latent argument on the whole real line: the
/// tabulated part is a monotone bracket-and-solve, and the two tails are exact
/// affine inverses at the end slopes.
#[derive(Clone, Debug)]
pub struct CtnTransformTable {
    grid_y: Array1<f64>,
    h: Array2<f64>,
    h_prime: Array2<f64>,
}

impl CtnTransformTable {
    /// Assemble a table, validating every invariant `invert` relies on.
    ///
    /// The validation is not defensive decoration: a non-monotone row makes the
    /// bracketing search meaningless, and a zero or negative slope makes the
    /// affine tail inverse point the wrong way (or to infinity). Both are
    /// structurally impossible for a feasible CTN fit — `h' = ε + Σ_k M_k α_k`
    /// with `α ≥ 0` on the monotonicity cone — so either one signals a corrupt
    /// coefficient block, and the caller should hear about it here rather than
    /// receive a silently wrong quantile.
    pub fn new(grid_y: Array1<f64>, h: Array2<f64>, h_prime: Array2<f64>) -> Result<Self, String> {
        let g = grid_y.len();
        if g < 2 {
            return Err(format!(
                "CTN transform table needs at least two response grid nodes, got {g}"
            ));
        }
        for k in 0..g {
            if !grid_y[k].is_finite() {
                return Err(format!(
                    "CTN transform table response grid node {k} is not finite: {}",
                    grid_y[k]
                ));
            }
            if k > 0 && !(grid_y[k] > grid_y[k - 1]) {
                return Err(format!(
                    "CTN transform table response grid is not strictly increasing at node {k}: \
                     {:.17e} -> {:.17e}",
                    grid_y[k - 1],
                    grid_y[k]
                ));
            }
        }
        let n = h.nrows();
        if h.ncols() != g {
            return Err(format!(
                "CTN transform table has {} latent columns but {g} response grid nodes",
                h.ncols()
            ));
        }
        if h_prime.dim() != h.dim() {
            return Err(format!(
                "CTN transform table derivative is {:?} but the latent is {:?}",
                h_prime.dim(),
                h.dim()
            ));
        }
        for i in 0..n {
            for k in 0..g {
                if !h[[i, k]].is_finite() {
                    return Err(format!(
                        "CTN transform table entry (row {i}, node {k}) is not finite: {}",
                        h[[i, k]]
                    ));
                }
                if k > 0 && !(h[[i, k]] > h[[i, k - 1]]) {
                    return Err(format!(
                        "CTN transform table row {i} is not strictly increasing between nodes \
                         {} and {k}: {:.17e} -> {:.17e}",
                        k - 1,
                        h[[i, k - 1]],
                        h[[i, k]]
                    ));
                }
                let slope = h_prime[[i, k]];
                if !(slope.is_finite() && slope > 0.0) {
                    return Err(format!(
                        "CTN transform table slope at (row {i}, node {k}) is {slope:.6e}; \
                         h' = ε + Σ_k M_k·α_k is structurally positive on the monotonicity cone, \
                         and the two END slopes are the slopes of the transform's affine tails"
                    ));
                }
            }
        }
        Ok(Self { grid_y, h, h_prime })
    }

    /// Number of covariate rows the table carries.
    pub fn nrows(&self) -> usize {
        self.h.nrows()
    }

    /// The shared, strictly increasing response grid the transform is tabulated
    /// on. Its first and last entries are the fitted support `[y_lo, y_hi]`, the
    /// two points the affine tails are anchored at.
    pub fn grid_y(&self) -> ArrayView1<'_, f64> {
        self.grid_y.view()
    }

    /// `h[[i, k]] = h(grid_y[k] | x_i)` — the model's own latent, on the scale
    /// the standard normal is compared against.
    pub fn latent(&self) -> ArrayView2<'_, f64> {
        self.h.view()
    }

    /// `h(y | x_row)` — the tabulated transform itself, by the same rule
    /// [`CtnTransformTable::invert`] inverts.
    ///
    /// The forward map is part of the contract, not a convenience: without it
    /// "the inverse is the inverse of the interpolant" is not a statement anyone
    /// can check, and the only available check would be against the exact chart,
    /// which conflates a solver bug with an interpolation error.
    pub fn evaluate(&self, row: usize, y: f64) -> f64 {
        let g = self.grid_y.len();
        let h = self.h.row(row);
        let slope = self.h_prime.row(row);
        if y <= self.grid_y[0] {
            return h[0] + (y - self.grid_y[0]) * slope[0];
        }
        if y >= self.grid_y[g - 1] {
            return h[g - 1] + (y - self.grid_y[g - 1]) * slope[g - 1];
        }
        let mut lo = 0usize;
        let mut hi = g - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if self.grid_y[mid] <= y {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let cell = self.cell(row, lo);
        cell.value((y - self.grid_y[lo]) / cell.width)
    }

    /// `h⁻¹(target | x_row)` — the response value whose latent is `target`.
    ///
    /// Outside the tabulated range the transform is affine, so the inverse is
    /// the exact affine inverse rather than the support endpoint. Inside, the
    /// bracketing cell is inverted by safeguarded Newton on its Hermite
    /// interpolant. The branches agree at the endpoints by construction
    /// (`target == h[0]` returns `grid_y[0]` from either side), so the returned
    /// quantile function is continuous and strictly increasing in `target` on
    /// the whole real line.
    pub fn invert(&self, row: usize, target: f64) -> f64 {
        self.invert_with_slope(row, target).0
    }

    /// `h⁻¹(target | x_row)` together with its exact latent slope
    /// `d h⁻¹ / d target = 1 / h'(y)`, taken on the same interpolant
    /// [`CtnTransformTable::invert`] inverts: `1 / h'` at the nearer end node on
    /// the affine exterior, and `width / slope(t)` of the bracketing Hermite
    /// cell inside. Consumers that interpolate the quantile function use this
    /// derivative instead of differencing tabulated quantiles.
    pub(crate) fn invert_with_slope(&self, row: usize, target: f64) -> (f64, f64) {
        let g = self.grid_y.len();
        let h = self.h.row(row);
        let slope = self.h_prime.row(row);
        if target <= h[0] {
            return (self.grid_y[0] + (target - h[0]) / slope[0], 1.0 / slope[0]);
        }
        if target >= h[g - 1] {
            return (
                self.grid_y[g - 1] + (target - h[g - 1]) / slope[g - 1],
                1.0 / slope[g - 1],
            );
        }
        let mut lo = 0usize;
        let mut hi = g - 1;
        while hi - lo > 1 {
            let mid = (lo + hi) / 2;
            if h[mid] <= target {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let cell = self.cell(row, lo);
        let t = cell.invert(target);
        (self.grid_y[lo] + cell.width * t, cell.width / cell.slope(t))
    }

    fn cell(&self, row: usize, index: usize) -> HermiteCell {
        HermiteCell::new(
            self.h[[row, index]],
            self.h[[row, index + 1]],
            self.h_prime[[row, index]],
            self.h_prime[[row, index + 1]],
            self.grid_y[index + 1] - self.grid_y[index],
        )
    }

    /// The quadrature every predictive expectation of this table is taken with:
    /// closed-form affine tails plus one Gauss–Legendre rule per interior cell,
    /// its order certified for every row (see [`CtnPredictiveQuadrature`]).
    ///
    /// Each cell's order is the largest any row needs. The certificate below
    /// decreases in the number of points, so the shared order is certified for
    /// every row, and sharing it makes the nodes — hence every basis evaluated at
    /// them — one set for the whole table instead of one per row.
    pub fn predictive_quadrature(&self) -> Result<CtnPredictiveQuadrature<'_>, String> {
        let cells = self.grid_y.len() - 1;
        let orders = (0..self.nrows())
            .into_par_iter()
            .try_fold(
                || vec![0usize; cells],
                |mut orders, row| {
                    // The optimal ellipse moves slowly from cell to cell, so each
                    // search starts from the previous cell's.
                    let mut rung = 0_i32;
                    for (index, order) in orders.iter_mut().enumerate() {
                        let needed = self
                            .cell(row, index)
                            .certified_gauss_order(self.grid_y[index], &mut rung)
                            .map_err(|reason| {
                                format!("CTN predictive quadrature, row {row}, cell {index}: {reason}")
                            })?;
                        *order = (*order).max(needed);
                    }
                    Ok::<_, String>(orders)
                },
            )
            .try_reduce(
                || vec![0usize; cells],
                |mut left, right| {
                    for (a, b) in left.iter_mut().zip(right) {
                        *a = (*a).max(b);
                    }
                    Ok(left)
                },
            )?;
        let mut rules = BTreeMap::new();
        for &order in orders.iter().filter(|&&order| order > 0) {
            rules.entry(order).or_insert_with(|| {
                let rule = gauss_legendre_certified(order);
                (rule.nodes, rule.weights)
            });
        }
        let total: usize = orders.iter().sum();
        let mut nodes = Vec::with_capacity(total);
        let mut positions = Vec::with_capacity(total);
        let mut half_weights = Vec::with_capacity(total);
        let mut cell_starts = Vec::with_capacity(cells + 1);
        cell_starts.push(0);
        for (index, &order) in orders.iter().enumerate() {
            if let Some((abscissae, weights)) = rules.get(&order) {
                let (left, right) = (self.grid_y[index], self.grid_y[index + 1]);
                let width = right - left;
                for (&s, &weight) in abscissae.iter().zip(weights) {
                    let t = 0.5 * (1.0 + s);
                    positions.push(t);
                    half_weights.push(0.5 * weight);
                    // `left + width·t` can round one ulp past the cell; the node
                    // belongs to it, and to the support, by construction.
                    nodes.push((left + width * t).min(right));
                }
            }
            cell_starts.push(nodes.len());
        }
        Ok(CtnPredictiveQuadrature {
            table: self,
            nodes: Array1::from_vec(nodes),
            cell_starts,
            positions,
            half_weights,
        })
    }
}

/// One affine tail of a row's predictive law: the probability it carries and
/// its first moment about the support endpoint it is anchored at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CtnAffineTail {
    /// The support endpoint `y_lo` or `y_hi` the tail continues from.
    pub anchor: f64,
    /// `Φ(L)` below the support, `Q(U)` above it.
    pub mass: f64,
    /// `∫_tail (y − anchor) dF(y)`: `−(φ(L) + L·Φ(L))/γ_lo` below the support,
    /// `(φ(U) − U·Q(U))/γ_hi` above it.
    pub offset_moment: f64,
}

impl CtnAffineTail {
    /// `∫_tail g dF` for `g(y) = level + slope·(y − anchor)` — exact, and the
    /// only form a tail expectation takes: on the tail the transform is affine,
    /// so anything built from its bases is affine there too.
    pub fn expectation(&self, level: f64, slope: f64) -> f64 {
        level * self.mass + slope * self.offset_moment
    }
}

/// One row's predictive law as quadrature weights:
/// `E[g(Y)] = Σ_j interior[j]·g(nodes[j]) + lower.expectation(…) + upper.expectation(…)`,
/// with `nodes` those of the [`CtnPredictiveQuadrature`] that produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct CtnRowMeasure {
    /// `(λ_j/2)·φ(H(t_j))·H_t(t_j)` at each shared interior node.
    pub interior: Vec<f64>,
    /// The affine tail below `y_lo`.
    pub lower: CtnAffineTail,
    /// The affine tail above `y_hi`.
    pub upper: CtnAffineTail,
}

/// The quadrature for `E[g(Y)|x_i] = ∫ g dΦ(H(·|x_i))` over a whole table.
///
/// # The certified order
///
/// On a cell of width `w` starting at `y_k`, write `y = c + (w/2)s` with
/// `s ∈ [−1, 1]` and the interpolant as `P(s) = b₀ + b₁s + b₂s² + b₃s³`. The
/// cell's share of the mean is `∫ f ds`, `f = y·φ(P)·P'`, an entire function, so
/// Trefethen's bound (*Approximation Theory and Approximation Practice*,
/// Thm 19.3) applies on every Bernstein ellipse `E_ρ`: the `(n + 1)`-point Gauss
/// rule errs by at most `(64/15)·M_ρ·ρ^{−2n}/(ρ² − 1)`, `M_ρ = max_{E_ρ}|f|`. On
/// `E_ρ`, `|s| ≤ a = (ρ + ρ⁻¹)/2` and `|Im s| ≤ β = (ρ − ρ⁻¹)/2`, so
///
/// * `|y| ≤ |c| + (w/2)·a` and `|P'| ≤ |b₁| + 2|b₂|a + 3|b₃|a²`;
/// * `|P − b₀| ≤ R = |b₁|a + |b₂|a² + |b₃|a³`, so `|Re P| ≥ max(0, |b₀| − R)`;
/// * `|Im P| ≤ V = β(|b₁| + 2|b₂|a + |b₃|(3a² + β²))`, from `Im s² = 2·Re s·Im s`
///   and `Im s³ = 3(Re s)²Im s − (Im s)³`;
/// * `|φ(P)| = exp(((Im P)² − (Re P)²)/2)/√(2π)`, bounded by those two.
///
/// The target is `ε·max(|y_k|, |y_{k+1}|)·m` with `m` a lower bound on the
/// cell's probability `Φ(h₁) − Φ(h₀)` — machine precision relative to the
/// largest thing the cell can contribute, which is not a tolerance anyone chose.
/// `ρ` runs over the ladder `1 + 2ʲ`, `j ∈ ℤ` (a cell whose latent span is large
/// needs `ρ` close to 1, where `V` stays small), descending from the previous
/// cell's rung to the best one; any rung gives a valid order, the descent only
/// makes it smaller.
///
/// The same certificate covers the mass integrand `φ(P)·P'`: its bound is
/// `M_ρ/(|c| + (w/2)a)` and `|c| + (w/2)a ≥ max(|y_k|, |y_{k+1}|)`, so each
/// cell's probability is within `ε·m` too.
pub struct CtnPredictiveQuadrature<'a> {
    table: &'a CtnTransformTable,
    /// The shared interior nodes, cell by cell, each inside its cell.
    nodes: Array1<f64>,
    /// Cell `k`'s nodes are `cell_starts[k]..cell_starts[k + 1]`.
    cell_starts: Vec<usize>,
    /// Each node's position `t ∈ (0, 1)` in its cell.
    positions: Vec<f64>,
    /// Each node's Gauss–Legendre weight mapped from `[−1, 1]` to `t ∈ [0, 1]`.
    half_weights: Vec<f64>,
}

impl<'a> CtnPredictiveQuadrature<'a> {
    /// The table this quadrature integrates.
    pub fn table(&self) -> &'a CtnTransformTable {
        self.table
    }

    /// The shared interior nodes, every one inside the support `[y_lo, y_hi]`.
    pub fn nodes(&self) -> ArrayView1<'_, f64> {
        self.nodes.view()
    }

    /// Row `row`'s predictive law as weights on [`CtnPredictiveQuadrature::nodes`]
    /// plus its two closed-form tails.
    pub fn row_measure(&self, row: usize) -> CtnRowMeasure {
        let table = self.table;
        let mut interior = Vec::with_capacity(self.nodes.len());
        for (index, bounds) in self.cell_starts.windows(2).enumerate() {
            if bounds[0] == bounds[1] {
                continue;
            }
            let cell = table.cell(row, index);
            for j in bounds[0]..bounds[1] {
                let t = self.positions[j];
                interior.push(self.half_weights[j] * normal_pdf(cell.value(t)) * cell.slope(t));
            }
        }
        let last = table.grid_y.len() - 1;
        let lower_latent = table.h[[row, 0]];
        let upper_latent = table.h[[row, last]];
        // `φ` is even, so the upper tail's density comes with `Q(U) = Φ(−U)`.
        let (lower_mass, lower_density) = normal_cdf_and_pdf(lower_latent);
        let (upper_mass, upper_density) = normal_cdf_and_pdf(-upper_latent);
        CtnRowMeasure {
            interior,
            lower: CtnAffineTail {
                anchor: table.grid_y[0],
                mass: lower_mass,
                offset_moment: -(lower_density + lower_latent * lower_mass)
                    / table.h_prime[[row, 0]],
            },
            upper: CtnAffineTail {
                anchor: table.grid_y[last],
                mass: upper_mass,
                offset_moment: (upper_density - upper_latent * upper_mass)
                    / table.h_prime[[row, last]],
            },
        }
    }

    /// `E[Y|x_row]` under the table's law.
    pub fn mean(&self, row: usize) -> f64 {
        let measure = self.row_measure(row);
        let interior: f64 = measure
            .interior
            .iter()
            .zip(self.nodes.iter())
            .map(|(weight, y)| weight * y)
            .sum();
        interior
            + measure.lower.expectation(measure.lower.anchor, 1.0)
            + measure.upper.expectation(measure.upper.anchor, 1.0)
    }
}

/// A lower bound on `Φ(h₁) − Φ(h₀)`, `h₀ < h₁`, with no cancellation in it.
///
/// `ln φ` is concave, so on an interval it lies above its chord, and `∫exp(chord)`
/// is the width times the logarithmic mean of the end densities. An interval
/// that straddles the mode is split there, so the chord never cuts across it.
fn latent_mass_lower_bound(h0: f64, h1: f64) -> f64 {
    if h0 < 0.0 && h1 > 0.0 {
        return log_concave_chord_mass(h0, 0.0) + log_concave_chord_mass(0.0, h1);
    }
    log_concave_chord_mass(h0, h1)
}

/// `(h₁ − h₀)·LogMean(φ(h₀), φ(h₁))` for an interval on one side of the mode,
/// written from the larger density with `expm1` so it neither cancels nor
/// underflows ahead of the true value.
fn log_concave_chord_mass(h0: f64, h1: f64) -> f64 {
    let (near, far) = if h0.abs() <= h1.abs() { (h0, h1) } else { (h1, h0) };
    // `ln φ(near) − ln φ(far)`, non-negative on one side of the mode.
    let drop = 0.5 * (far - near) * (far + near);
    let span = h1 - h0;
    let peak = normal_pdf(near);
    if drop > 0.0 {
        span * peak * (-(-drop).exp_m1() / drop)
    } else {
        span * peak
    }
}

/// One grid cell of a row, as the interpolant used on it.
///
/// `cubic` records whether the cell's own end slopes admit a monotone cubic; a
/// cell that does not is under-resolved for one, and the linear chord — which is
/// monotone whenever the tabulated values are — is used instead. Deciding this
/// per cell rather than per table keeps a single ill-conditioned interval from
/// coarsening the whole transform, and deciding it from the stored numbers keeps
/// it deterministic and storage-free.
struct HermiteCell {
    h0: f64,
    h1: f64,
    m0: f64,
    m1: f64,
    width: f64,
    cubic: bool,
}

impl HermiteCell {
    fn new(h0: f64, h1: f64, m0: f64, m1: f64, width: f64) -> Self {
        let secant = (h1 - h0) / width;
        let cubic = secant > 0.0
            && m0 <= HERMITE_MONOTONE_SLOPE_BOUND * secant
            && m1 <= HERMITE_MONOTONE_SLOPE_BOUND * secant;
        Self {
            h0,
            h1,
            m0,
            m1,
            width,
            cubic,
        }
    }

    /// The interpolant at `t ∈ [0, 1]`, `y = y_k + t·width`.
    fn value(&self, t: f64) -> f64 {
        if !self.cubic {
            return self.h0 + t * (self.h1 - self.h0);
        }
        let t2 = t * t;
        let t3 = t2 * t;
        (2.0 * t3 - 3.0 * t2 + 1.0) * self.h0
            + (t3 - 2.0 * t2 + t) * self.width * self.m0
            + (-2.0 * t3 + 3.0 * t2) * self.h1
            + (t3 - t2) * self.width * self.m1
    }

    /// `d/dt` of [`HermiteCell::value`].
    fn slope(&self, t: f64) -> f64 {
        if !self.cubic {
            return self.h1 - self.h0;
        }
        let t2 = t * t;
        (6.0 * t2 - 6.0 * t) * self.h0
            + (3.0 * t2 - 4.0 * t + 1.0) * self.width * self.m0
            + (-6.0 * t2 + 6.0 * t) * self.h1
            + (3.0 * t2 - 2.0 * t) * self.width * self.m1
    }

    /// `[c₀, c₁, c₂, c₃]` with `value(t) = Σ c_k t^k`.
    fn power_coefficients(&self) -> [f64; 4] {
        if !self.cubic {
            return [self.h0, self.h1 - self.h0, 0.0, 0.0];
        }
        let (d0, d1) = (self.width * self.m0, self.width * self.m1);
        [
            self.h0,
            d0,
            3.0 * (self.h1 - self.h0) - 2.0 * d0 - d1,
            2.0 * (self.h0 - self.h1) + d0 + d1,
        ]
    }

    /// `[b₀, b₁, b₂, b₃]` with `value((1 + s)/2) = Σ b_k s^k`, `s ∈ [−1, 1]`.
    fn centred_coefficients(&self) -> [f64; 4] {
        let [c0, c1, c2, c3] = self.power_coefficients();
        [
            c0 + 0.5 * c1 + 0.25 * c2 + 0.125 * c3,
            0.5 * c1 + 0.5 * c2 + 0.375 * c3,
            0.25 * c2 + 0.375 * c3,
            0.125 * c3,
        ]
    }

    /// The fewest Gauss–Legendre points that integrate `y·φ(H)·H'` over this
    /// cell (starting at `left`) to `ε·max|y|·m`, `m` a lower bound on the cell's
    /// probability — the certificate derived on [`CtnPredictiveQuadrature`].
    ///
    /// `rung` is the ellipse `ρ = 1 + 2^rung` to start the descent from; it is
    /// left at the best rung found, for the next cell to start from. A cell whose
    /// probability underflows needs no points: it cannot move the mean by more
    /// than `max|y|` times a subnormal.
    fn certified_gauss_order(&self, left: f64, rung: &mut i32) -> Result<usize, String> {
        let mass = latent_mass_lower_bound(self.h0, self.h1);
        if mass == 0.0 {
            return Ok(0);
        }
        let [b0, b1, b2, b3] = self.centred_coefficients().map(f64::abs);
        let half_width = 0.5 * self.width;
        let centre = (left + half_width).abs();
        let reach = left.abs().max((left + self.width).abs());
        let constant = (64.0 / 15.0_f64).ln()
            - 0.5 * (2.0 * PI).ln()
            - f64::EPSILON.ln()
            - reach.ln()
            - mass.ln();
        // Points beyond the first, as a real number, for the ellipse at `rung`:
        // `ln((64/15)·M_ρ/((ρ² − 1)·target)) / (2 ln ρ)`.
        let points_beyond_first = |rung: i32| {
            let excess = 2.0_f64.powi(rung);
            let rho = 1.0 + excess;
            let rho_sq_minus_one = excess * (2.0 + excess);
            let major = 0.5 * (rho + rho.recip());
            let minor = 0.5 * rho_sq_minus_one / rho;
            let modulus = centre + half_width * major;
            let slope = b1 + major * (2.0 * b2 + 3.0 * b3 * major);
            let reach_of_p = major * (b1 + major * (b2 + major * b3));
            let imaginary = minor * (b1 + 2.0 * b2 * major + b3 * (3.0 * major * major + minor * minor));
            let deficit = (b0 - reach_of_p).max(0.0);
            (constant + modulus.ln() + slope.ln() + 0.5 * (imaginary - deficit) * (imaginary + deficit)
                - rho_sq_minus_one.ln())
                / (2.0 * excess.ln_1p())
        };
        let mut best = points_beyond_first(*rung);
        if !best.is_finite() {
            *rung = 0;
            best = points_beyond_first(0);
        }
        for step in [1, -1] {
            let mut moved = false;
            loop {
                let next = points_beyond_first(*rung + step);
                if next < best {
                    best = next;
                    *rung += step;
                    moved = true;
                } else {
                    break;
                }
            }
            if moved {
                break;
            }
        }
        if !best.is_finite() {
            return Err(format!(
                "no Bernstein ellipse bounds the cell integrand (centred coefficients \
                 {b0:e}, {b1:e}, {b2:e}, {b3:e})"
            ));
        }
        Ok((best.ceil().max(0.0) as usize).saturating_add(1))
    }

    /// The `t ∈ [0, 1]` with `value(t) == target`, by Newton safeguarded inside
    /// a maintained bracket. The bracket exists because the caller only reaches
    /// here with `h0 ≤ target ≤ h1`, and every step that leaves it is replaced
    /// by a bisection, so the iteration cannot diverge on a cell whose cubic is
    /// flat somewhere in the interior.
    fn invert(&self, target: f64) -> f64 {
        if !self.cubic {
            return (target - self.h0) / (self.h1 - self.h0);
        }
        let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
        let mut t = ((target - self.h0) / (self.h1 - self.h0)).clamp(0.0, 1.0);
        // No budget: every step moves one bracket end onto `t`, and each next `t`
        // lies strictly inside the bracket (Newton only when strictly inside,
        // else the midpoint, which is strictly inside while the width exceeds ε).
        // So the bracket strictly shrinks, and the loop ends at width ε or at a
        // Newton fixed point.
        loop {
            let value = self.value(t);
            if value > target {
                hi = t;
            } else {
                lo = t;
            }
            if hi - lo <= f64::EPSILON {
                break;
            }
            let slope = self.slope(t);
            let newton = t - (value - target) / slope;
            let next = if slope > 0.0 && newton > lo && newton < hi {
                newton
            } else {
                0.5 * (lo + hi)
            };
            if next == t {
                break;
            }
            t = next;
        }
        t
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::probability::{
        NORMAL_CDF_RELATIVE_ERROR, NORMAL_CDF_UNDERFLOW_FLOOR, normal_pdf_bounded,
    };
    use gam_math::roundoff::{UNIT_ROUNDOFF, accumulation_growth, inflated};
    use gam_math::score_opt::certified_ln_positive;
    use gam_math::special::CertifiedGaussLegendreRule;

    /// `h(y) = slope·y` on `[-1, 1]` for two rows — exactly affine, so the
    /// Hermite interpolant, the linear chord and the tails all coincide and any
    /// deviation from `y = z/slope` is a bug rather than interpolation error.
    fn affine_table() -> CtnTransformTable {
        let grid_y = Array1::from_vec(vec![-1.0, -0.5, 0.0, 0.5, 1.0]);
        let slopes = [2.0_f64, 4.0];
        let h = Array2::from_shape_fn((2, 5), |(i, k)| slopes[i] * grid_y[k]);
        let h_prime = Array2::from_shape_fn((2, 5), |(i, _)| slopes[i]);
        CtnTransformTable::new(grid_y, h, h_prime).expect("valid affine table")
    }

    /// A curved transform sampled on a coarse grid: `h(y) = ln y` on `[0.1, 3]`,
    /// the shape a lognormal response actually produces, where the difference
    /// between matching the derivative and not matching it is the whole point.
    fn log_table(nodes: usize) -> CtnTransformTable {
        let (y_lo, y_hi) = (0.1_f64, 3.0_f64);
        let grid_y = Array1::from_shape_fn(nodes, |k| {
            y_lo + (y_hi - y_lo) * (k as f64) / ((nodes - 1) as f64)
        });
        let h = Array2::from_shape_fn((1, nodes), |(_, k)| grid_y[k].ln());
        let h_prime = Array2::from_shape_fn((1, nodes), |(_, k)| 1.0 / grid_y[k]);
        CtnTransformTable::new(grid_y, h, h_prime).expect("valid log table")
    }

    #[test]
    fn inverse_is_exact_inside_the_table() {
        let t = affine_table();
        for &z in &[-2.0_f64, -1.0, 0.0, 1.0, 2.0] {
            assert!((t.invert(0, z) - 0.5 * z).abs() < 1e-12, "z={z}");
            assert!((t.invert(1, z) - 0.25 * z).abs() < 1e-12, "z={z}");
        }
    }

    #[test]
    fn inverse_extends_through_the_affine_tails_rather_than_clamping() {
        // The whole of gam#2600's predictive residual in one assertion: a latent
        // target past the tabulated range is not the support endpoint.
        let t = affine_table();
        for &z in &[-8.0_f64, -4.0, 4.0, 8.0] {
            let y = t.invert(0, z);
            assert!(
                (y - 0.5 * z).abs() < 1e-12,
                "tail inverse at z={z} is {y}, expected {}",
                0.5 * z
            );
            assert!(
                y.abs() > 1.0,
                "tail inverse at z={z} clamped back inside the tabulated support"
            );
        }
    }

    #[test]
    fn inverse_is_continuous_at_both_table_ends() {
        let t = affine_table();
        for &(z, expected) in &[(-2.0_f64, -1.0_f64), (2.0, 1.0)] {
            let inside = t.invert(0, z * (1.0 - 1e-12));
            let outside = t.invert(0, z * (1.0 + 1e-12));
            assert!((inside - expected).abs() < 1e-9, "inside {inside}");
            assert!((outside - expected).abs() < 1e-9, "outside {outside}");
        }
    }

    /// `(hermite, chord)` worst-case inverse error of a `nodes`-point table for
    /// `h = ln y`, the chord being recomputed from the SAME table so the only
    /// thing the comparison isolates is the interpolation rule.
    fn inverse_errors(nodes: usize) -> (f64, f64) {
        let table = log_table(nodes);
        let grid = table.grid_y().to_owned();
        let h = table.latent();
        let (mut hermite, mut chord) = (0.0_f64, 0.0_f64);
        for step in 1..977 {
            let y = 0.1 + (3.0 - 0.1) * (step as f64) / 977.0;
            let z = y.ln();
            hermite = hermite.max((table.invert(0, z) - y).abs());
            let mut lo = 0usize;
            let mut hi = grid.len() - 1;
            while hi - lo > 1 {
                let mid = (lo + hi) / 2;
                if h[[0, mid]] <= z {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            let t = (z - h[[0, lo]]) / (h[[0, hi]] - h[[0, lo]]);
            chord = chord.max((grid[lo] + t * (grid[hi] - grid[lo]) - y).abs());
        }
        (hermite, chord)
    }

    #[test]
    fn matching_the_node_derivative_raises_the_interpolation_order_from_two_to_four() {
        // The structural claim behind spending an array on `h'`: it is not that
        // the error is smaller at some grid size, it is that the SCHEME is
        // fourth-order where the chord is second-order. That is what makes the
        // difference grow as the table is refined, and it is what turns the
        // `2.1e-3` chord error measured on the real fixture into `~2e-5`.
        //
        // `h = ln y` on `[0.1, 3]` is the shape a lognormal response produces —
        // the curvature is concentrated exactly where the responses are.
        let (coarse_hermite, coarse_chord) = inverse_errors(129);
        let (fine_hermite, fine_chord) = inverse_errors(257);
        let hermite_order = (coarse_hermite / fine_hermite).log2();
        let chord_order = (coarse_chord / fine_chord).log2();
        eprintln!(
            "#2600 table: max|h^-1(ln y) - y| hermite {coarse_hermite:.3e} -> {fine_hermite:.3e} \
             (order {hermite_order:.2})  chord {coarse_chord:.3e} -> {fine_chord:.3e} \
             (order {chord_order:.2})"
        );
        assert!(
            hermite_order > 3.0,
            "the Hermite inverse is converging at order {hermite_order:.2}, not the ~4 a scheme \
             that matches the node derivative must reach ({coarse_hermite:.6e} -> \
             {fine_hermite:.6e})"
        );
        assert!(
            chord_order < 2.5,
            "the chord is converging at order {chord_order:.2}; if it were fourth-order too \
             this comparison would not be measuring what it claims"
        );
        assert!(
            fine_hermite * 20.0 < fine_chord,
            "at the production table size the Hermite inverse is not decisively tighter than \
             the chord: hermite={fine_hermite:.6e} chord={fine_chord:.6e}"
        );
    }

    #[test]
    fn the_inverse_inverts_the_interpolant_it_is_built_from() {
        // Separates a solver bug from an interpolation error: whatever the
        // Hermite interpolant is, `invert` must return its argument's preimage
        // under exactly that interpolant, to round-off, on both tails and in
        // every interior cell — including the strongly curved ones where the
        // Newton iteration has to be safeguarded to stay in its bracket.
        let table = log_table(17);
        let mut worst = 0.0_f64;
        for step in 0..=2000 {
            let z = -9.0 + 18.0 * (step as f64) / 2000.0;
            let y = table.invert(0, z);
            worst = worst.max((table.evaluate(0, y) - z).abs());
        }
        eprintln!("#2600 table: max|H(H^-1(z)) - z| = {worst:.3e}");
        assert!(
            worst < 1.0e-12,
            "invert is not the inverse of the interpolant it evaluates: {worst:.6e}"
        );
    }

    #[test]
    fn invert_with_slope_returns_the_exact_derivative_of_the_inverse() {
        // Affine rows: `dy/dz = 1/slope` everywhere, both tails included.
        let affine = affine_table();
        for &z in &[-8.0_f64, -1.5, 0.0, 0.7, 8.0] {
            let (y0, s0) = affine.invert_with_slope(0, z);
            let (y1, s1) = affine.invert_with_slope(1, z);
            assert!(
                (y0 - 0.5 * z).abs() < 1e-12 && (s0 - 0.5).abs() < 1e-12,
                "row 0 at z={z}: ({y0}, {s0})"
            );
            assert!(
                (y1 - 0.25 * z).abs() < 1e-12 && (s1 - 0.25).abs() < 1e-12,
                "row 1 at z={z}: ({y1}, {s1})"
            );
        }

        // Curved row `h = ln y`: at every node the slope is `1/h' = y` there, and
        // everywhere else it matches a central difference of `invert` on the
        // tails and in every cell. Finite differences are sanctioned in tests.
        let table = log_table(17);
        let grid = table.grid_y().to_owned();
        let latent = table.latent().to_owned();
        for k in 0..grid.len() {
            let (y, slope) = table.invert_with_slope(0, latent[[0, k]]);
            assert!((y - grid[k]).abs() < 1e-12, "node {k}: value {y} vs {}", grid[k]);
            assert!(
                (slope - grid[k]).abs() < 1e-10 * grid[k].max(1.0),
                "node {k}: slope {slope} vs 1/h' = {}",
                grid[k]
            );
        }
        let h = 1e-6;
        let mut worst = 0.0_f64;
        for step in 0..=2000 {
            let z = -9.0 + 18.0 * (step as f64) / 2000.0;
            let (y, slope) = table.invert_with_slope(0, z);
            assert_eq!(
                y.to_bits(),
                table.invert(0, z).to_bits(),
                "invert must be the value half of invert_with_slope at z={z}"
            );
            let central = (table.invert(0, z + h) - table.invert(0, z - h)) / (2.0 * h);
            worst = worst.max((slope - central).abs() / central.abs().max(1e-12));
        }
        assert!(
            worst < 1e-5,
            "invert_with_slope disagrees with a central difference of invert: {worst:.3e}"
        );
    }

    #[test]
    fn the_inverse_is_strictly_increasing_across_the_whole_latent_axis() {
        let table = log_table(17);
        let mut previous = f64::NEG_INFINITY;
        for step in 0..=400 {
            let z = -12.0 + 24.0 * (step as f64) / 400.0;
            let y = table.invert(0, z);
            assert!(
                y > previous,
                "h^-1 is not strictly increasing at z={z}: {y} <= {previous}"
            );
            previous = y;
        }
    }

    #[test]
    fn a_non_monotone_row_is_refused() {
        let grid_y = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let h = Array2::from_shape_vec((1, 3), vec![0.0, 0.5, 0.5]).expect("shape");
        let h_prime = Array2::from_elem((1, 3), 1.0);
        let error =
            CtnTransformTable::new(grid_y, h, h_prime).expect_err("a flat row must be refused");
        assert!(error.contains("strictly increasing"), "{error}");
    }

    #[test]
    fn a_non_positive_slope_is_refused() {
        let grid_y = Array1::from_vec(vec![0.0, 1.0]);
        let h = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).expect("shape");
        let h_prime = Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).expect("shape");
        let error =
            CtnTransformTable::new(grid_y, h, h_prime).expect_err("a zero slope must be refused");
        assert!(error.contains("structurally positive"), "{error}");
    }

    #[test]
    fn an_under_resolved_cell_falls_back_to_its_chord_and_stays_monotone() {
        // End slopes far above the cell's own secant: the cubic through them is
        // not monotone, so the cell must use its chord instead of producing an
        // inverse that runs backwards.
        let grid_y = Array1::from_vec(vec![0.0, 1.0, 2.0]);
        let h = Array2::from_shape_vec((1, 3), vec![0.0, 1.0e-3, 1.0]).expect("shape");
        let h_prime = Array2::from_shape_vec((1, 3), vec![5.0, 5.0, 1.0]).expect("shape");
        let table = CtnTransformTable::new(grid_y, h, h_prime).expect("valid table");
        let mut previous = f64::NEG_INFINITY;
        for step in 0..=500 {
            let z = -2.0 + 4.0 * (step as f64) / 500.0;
            let y = table.invert(0, z);
            assert!(
                y > previous,
                "the under-resolved cell produced a non-monotone inverse at z={z}: \
                 {y} <= {previous}"
            );
            previous = y;
        }
    }

    /// `sup|φ'| = φ(1) = e^{−1/2}/√(2π) = 0.2420…`, bounded above by `1/4`.
    const NORMAL_PDF_SLOPE_BOUND: f64 = 0.25;
    /// `sup φ = 1/√(2π) = 0.3989…`, bounded above by `2/5`.
    const NORMAL_PDF_BOUND: f64 = 0.4;

    /// A bound on `|τ̂ − τ|` for one interior quadrature term as the production
    /// code computes it, `τ̂ = fl(fl(fl(λ̂/2·φ̂)·Ŝ)·ŷ)` (`moment`) or without the
    /// last factor, against `τ = (λ/2)·φ(H(t))·H_t(t)·y(t)` at the EXACT Gauss
    /// node `t` and weight `λ`, `H` the exact cubic through the stored data.
    ///
    /// First order in `u`, term by term:
    /// * `Ĥ` sums four basis products, each at most 11 rounded operations from
    ///   `t`, whose absolute-coefficient polynomials are at most `6, 4, 5, 2` on
    ///   `[0, 1]`: `|Ĥ − H(t̂)| ≤ 6γ₁₁·scale`, `scale = |h₀| + |h₁| + w(|m₀| + |m₁|)`.
    ///   The slope's are at most `12, 8, 12, 5`: `|Ŝ − S(t̂)| ≤ 12γ₁₁·scale`. The
    ///   chord's two operations are inside both.
    /// * `φ̂` errs by [`normal_pdf_bounded`]'s bound at `Ĥ`, plus `sup|φ'|·|Ĥ − H|`.
    /// * `ŷ = fl(y_k + fl(ŵ·t̂))`, `ŵ = fl(y_{k+1} − y_k)`: within `u(|ŷ| + 2w)` of
    ///   `y_k + w·t̂`.
    /// * The two or three products round by `γ₂` or `γ₃` of `|τ̂|`.
    /// * `t̂ = fl(½·fl(1 + ŝ))` is within `½·node_error + u·t̂` of `t`, and moves the
    ///   integrand by at most `sup|g'|` times that: `g = φ(H)·S·y` has
    ///   `|g'| ≤ sup|Hφ(H)|·S²|y| + φ·(|S_t||y| + |S|w)` with `sup|Hφ(H)| = φ(1)`.
    /// * `|λ̂ − λ| ≤ weight_relative_error·λ`.
    fn node_rounding_bound(
        cell: &HermiteCell,
        left: f64,
        t: f64,
        y: f64,
        half_weight: f64,
        rule: &CertifiedGaussLegendreRule,
        moment: bool,
    ) -> f64 {
        let u = UNIT_ROUNDOFF;
        let [_, c1, c2, c3] = cell.power_coefficients().map(f64::abs);
        let scale = cell.h0.abs() + cell.h1.abs() + cell.width * (cell.m0.abs() + cell.m1.abs());
        let value_error = 6.0 * accumulation_growth(11) * scale;
        let slope_error = 12.0 * accumulation_growth(11) * scale;
        let (density, density_rounding) = normal_pdf_bounded(cell.value(t));
        let density_error = density_rounding + NORMAL_PDF_SLOPE_BOUND * value_error;
        let density_bound = density + density_error;
        let slope = cell.slope(t).abs();
        let slope_bound = slope + slope_error;
        let slope_sup = c1 + 2.0 * c2 + 3.0 * c3;
        let curvature_sup = 2.0 * c2 + 6.0 * c3;
        let reach = left.abs().max((left + cell.width).abs());
        let (y_bound, y_error, derivative_sup, products) = if moment {
            let y_error = u * (y.abs() + 2.0 * cell.width);
            let derivative_sup = NORMAL_PDF_SLOPE_BOUND * slope_sup * slope_sup * reach
                + NORMAL_PDF_BOUND * (curvature_sup * reach + slope_sup * cell.width);
            (y.abs() + y_error, y_error, derivative_sup, 3)
        } else {
            let derivative_sup =
                NORMAL_PDF_SLOPE_BOUND * slope_sup * slope_sup + NORMAL_PDF_BOUND * curvature_sup;
            (1.0, 0.0, derivative_sup, 2)
        };
        let computed = half_weight * density * slope * y_bound;
        let evaluation = accumulation_growth(products) * computed
            + half_weight
                * (density_error * slope_bound * y_bound
                    + density_bound * slope_error * y_bound
                    + density_bound * slope_bound * y_error);
        let position_error = 0.5 * rule.node_error + u * t;
        let weight = rule.weight_relative_error;
        let integrand_bound = density_bound * slope_bound * y_bound + derivative_sup * position_error;
        evaluation
            + half_weight * (1.0 + weight) * derivative_sup * position_error
            + half_weight * weight * integrand_bound
    }

    /// `(Σ τ̂_j, bound on its distance from the exact-rule sum)` for one cell and
    /// one rule, computed exactly as [`CtnTransformTable::predictive_quadrature`]
    /// and [`CtnPredictiveQuadrature::mean`] compute it.
    fn cell_rule_sum(
        cell: &HermiteCell,
        left: f64,
        right: f64,
        rule: &CertifiedGaussLegendreRule,
        moment: bool,
    ) -> (f64, f64) {
        let width = right - left;
        let (mut sum, mut magnitude, mut rounding) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (&s, &weight) in rule.nodes.iter().zip(&rule.weights) {
            let t = 0.5 * (1.0 + s);
            let half_weight = 0.5 * weight;
            let y = (left + width * t).min(right);
            let mut term = half_weight * normal_pdf(cell.value(t)) * cell.slope(t);
            if moment {
                term *= y;
            }
            sum += term;
            magnitude += term.abs();
            rounding += node_rounding_bound(cell, left, t, y, half_weight, rule, moment);
        }
        let count = rule.nodes.len();
        (sum, inflated(rounding + accumulation_growth(count) * magnitude, count + 64))
    }

    /// `ln y` on `grid_y`, with the stored end latents `∓2.6` and end slopes the
    /// literals `grid_y[last]`, `grid_y[0]` (`≈ 1/y` at the far end), together
    /// with certified bounds on each stored value's and slope's distance from the
    /// exact `ln y` and `1/y` — the data error of the table against the law the
    /// reference integrates.
    fn lognormal_table(grid_y: Array1<f64>) -> (CtnTransformTable, Vec<f64>, Vec<f64>) {
        let g = grid_y.len();
        let (y_lo, y_hi) = (grid_y[0], grid_y[g - 1]);
        let mut h = Array2::zeros((1, g));
        let mut h_prime = Array2::zeros((1, g));
        let mut value_errors = Vec::with_capacity(g);
        let mut slope_errors = Vec::with_capacity(g);
        for k in 0..g {
            let y = grid_y[k];
            let value = match k {
                0 => -2.6,
                k if k == g - 1 => 2.6,
                _ => y.ln(),
            };
            let enclosure = certified_ln_positive(y).expect("a positive grid node has a logarithm");
            value_errors.push((value - enclosure.lo).abs().max((enclosure.hi - value).abs()));
            let (slope, slope_error) = match k {
                // `|1/y_lo − y_hi| = |y_lo·y_hi − 1|/y_lo`, the product minus one
                // carried by one fused rounding.
                0 => (y_hi, inflated(y_lo.mul_add(y_hi, -1.0).abs() / y_lo, 4)),
                k if k == g - 1 => (y_lo, inflated(y_lo.mul_add(y_hi, -1.0).abs() / y_hi, 4)),
                // A correctly rounded quotient is within `u` of itself.
                _ => (1.0 / y, UNIT_ROUNDOFF / y),
            };
            h[[0, k]] = value;
            h_prime[[0, k]] = slope;
            slope_errors.push(slope_error);
        }
        let table = CtnTransformTable::new(grid_y, h, h_prime).expect("valid lognormal table");
        (table, value_errors, slope_errors)
    }

    #[test]
    fn the_predictive_mean_of_a_lognormal_table_matches_its_high_precision_reference() {
        // gam#3520's shape: `h = ln y` on `[e^{−2.6}, e^{2.6}]` with the affine
        // tails the table carries. The reference is the mean of THAT law with the
        // exact `ln y` inside,
        //   e^{1/2}(Φ(ln y_hi − 1) − Φ(ln y_lo − 1))
        //   + y_lo·Φ(−2.6) − (φ(2.6) − 2.6·Φ(−2.6))/γ_lo
        //   + y_hi·Φ(−2.6) + (φ(2.6) − 2.6·Φ(−2.6))/γ_hi,
        // `γ_lo = Y_HI`, `γ_hi = Y_LO` the stored end slopes, evaluated in mpmath
        // at 50 digits from these exact doubles: 1.640813978945418449733113…
        // (independently confirmed by adaptive quadrature of both tails). The
        // 48-point midpoint average this replaces returned 1.6092 (−1.9 %).
        const Y_LO: f64 = 0.074_273_578_214_333_88;
        const Y_HI: f64 = 13.463_738_035_001_692;
        const REFERENCE_MEAN: f64 = 1.640_813_978_945_418_4;
        let nodes = 257;
        let last = (nodes - 1) as f64;
        let uniform = Array1::from_shape_fn(nodes, |k| match k {
            0 => Y_LO,
            k if k == nodes - 1 => Y_HI,
            _ => Y_LO + (Y_HI - Y_LO) * k as f64 / last,
        });
        let geometric = Array1::from_shape_fn(nodes, |k| match k {
            0 => Y_LO,
            k if k == nodes - 1 => Y_HI,
            _ => (-2.6 + 5.2 * k as f64 / last).exp(),
        });
        for (name, grid_y) in [("uniform", uniform), ("geometric", geometric)] {
            let (table, value_errors, slope_errors) = lognormal_table(grid_y);
            let quadrature = table.predictive_quadrature().expect("certified quadrature");
            let mean = quadrature.mean(0);
            let grid = table.grid_y();
            let g = grid.len();

            // Interpolation. The interior means of `Φ(H)` and `Φ(ln y)` differ, by
            // parts, by `[y(Φ(H) − Φ(ln y))]` at the ends minus `∫(Φ(H) − Φ(ln y))`,
            // and `|ΔΦ| ≤ |ΔH|/√(2π)`; the tails are the same law in both. On a
            // cubic cell `|f − H| ≤ max|f⁗|·(w²t(1 − t))²/24` for exact data, and
            // `∫₀¹(t(1 − t))² = 1/30`, with `max|(ln)⁗| = 6/y_k⁴`; the data errors
            // enter through the Hermite basis, whose magnitudes integrate to
            // `1/2, 1/2, 1/12, 1/12`.
            let mut interpolation = Y_LO * value_errors[0] + Y_HI * value_errors[g - 1];
            for k in 0..g - 1 {
                let cell = table.cell(0, k);
                assert!(cell.cubic, "{name}: cell {k} must be cubic for the w⁴ bound");
                let w = cell.width;
                interpolation += w
                    * (w.powi(4) * 6.0 / grid[k].powi(4) / 720.0
                        + 0.5 * (value_errors[k] + value_errors[k + 1])
                        + w * (slope_errors[k] + slope_errors[k + 1]) / 12.0);
            }
            interpolation /= (2.0 * PI).sqrt();

            // Truncation: each cell within `ε·max|y|·m` by the certificate.
            let mut truncation = 0.0;
            for k in 0..g - 1 {
                let cell = table.cell(0, k);
                truncation += f64::EPSILON
                    * grid[k].abs().max(grid[k + 1].abs())
                    * latent_mass_lower_bound(cell.h0, cell.h1);
            }

            // Rounding of the interior, node by node, and of its sum.
            let mut rules = BTreeMap::new();
            let (mut interior, mut magnitude, mut rounding) = (0.0_f64, 0.0_f64, 0.0_f64);
            let measure = quadrature.row_measure(0);
            for k in 0..g - 1 {
                let span = quadrature.cell_starts[k]..quadrature.cell_starts[k + 1];
                let order = span.len();
                if order == 0 {
                    continue;
                }
                let rule = rules.entry(order).or_insert_with(|| gauss_legendre_certified(order));
                let cell = table.cell(0, k);
                for j in span {
                    let y = quadrature.nodes[j];
                    let term = measure.interior[j] * y;
                    interior += term;
                    magnitude += term.abs();
                    rounding += node_rounding_bound(
                        &cell,
                        grid[k],
                        quadrature.positions[j],
                        y,
                        quadrature.half_weights[j],
                        rule,
                        true,
                    );
                }
            }
            let count = quadrature.nodes.len();
            rounding += accumulation_growth(count) * magnitude;

            // The tails: `Φ̂` within NORMAL_CDF_RELATIVE_ERROR of itself plus the
            // floor, `φ̂` within its certified bound, then `L·m`, a sum and a
            // quotient for the moment and two products and a sum for the
            // expectation.
            let tail_rounding = |tail: &CtnAffineTail, latent: f64, gamma: f64| {
                let mass_error = NORMAL_CDF_RELATIVE_ERROR * tail.mass + NORMAL_CDF_UNDERFLOW_FLOOR;
                let (density, density_error) = normal_pdf_bounded(latent);
                tail.anchor.abs() * mass_error
                    + accumulation_growth(2) * (tail.anchor.abs() * tail.mass + tail.offset_moment.abs())
                    + (density_error
                        + latent.abs() * mass_error
                        + accumulation_growth(3) * (density + latent.abs() * tail.mass))
                        / gamma
            };
            let lower = measure.lower.expectation(measure.lower.anchor, 1.0);
            let upper = measure.upper.expectation(measure.upper.anchor, 1.0);
            rounding += tail_rounding(&measure.lower, -2.6, Y_HI) + tail_rounding(&measure.upper, 2.6, Y_LO);
            rounding += accumulation_growth(2) * (interior.abs() + lower.abs() + upper.abs());
            // The reference, rounded once to the literal.
            rounding += UNIT_ROUNDOFF * REFERENCE_MEAN;

            let bound = inflated(interpolation + truncation + rounding, count + 64);
            let error = (mean - REFERENCE_MEAN).abs();
            println!(
                "{name}: mean {mean:.17} reference {REFERENCE_MEAN:.17} error {error:.3e} \
                 (relative {:.3e}); bound {bound:.3e} = interpolation {interpolation:.3e} \
                 + truncation {truncation:.3e} + rounding {rounding:.3e}; {count} nodes",
                error / REFERENCE_MEAN,
            );
            assert!(
                error <= bound,
                "{name}: E[Y] = {mean:.17} is {error:.3e} from the reference {REFERENCE_MEAN:.17}, \
                 beyond the derived bound {bound:.3e}"
            );
        }
    }

    #[test]
    fn every_cell_meets_its_certified_quadrature_target() {
        // Each cell's rule is compared with one of `2n + 16` points. Both are
        // certified to `target = ε·max|y|·m` (the certificate decreases in the
        // number of points), so they agree to twice that plus their rounding.
        let log_grid = Array1::from_shape_fn(257, |k| (-2.6 + 5.2 * k as f64 / 256.0).exp());
        let heavy_grid = Array1::from_shape_fn(257, |k| 3000.0 * (2.0 * k as f64 - 256.0) / 256.0);
        let cubic_grid = Array1::from_shape_fn(257, |k| 5.0 * (2.0 * k as f64 - 256.0) / 256.0);
        let tables = [
            (
                "log",
                CtnTransformTable::new(
                    log_grid.clone(),
                    Array2::from_shape_fn((1, 257), |(_, k)| log_grid[k].ln()),
                    Array2::from_shape_fn((1, 257), |(_, k)| 1.0 / log_grid[k]),
                ),
            ),
            (
                "heavy",
                CtnTransformTable::new(
                    heavy_grid.clone(),
                    Array2::from_shape_fn((1, 257), |(_, k)| 3.0 * (heavy_grid[k] / 3.0).atan()),
                    Array2::from_shape_fn((1, 257), |(_, k)| 1.0 / (1.0 + (heavy_grid[k] / 3.0).powi(2))),
                ),
            ),
            (
                "cubic",
                CtnTransformTable::new(
                    cubic_grid.clone(),
                    Array2::from_shape_fn((1, 257), |(_, k)| cubic_grid[k].powi(3) / 10.0 + cubic_grid[k]),
                    Array2::from_shape_fn((1, 257), |(_, k)| 0.3 * cubic_grid[k].powi(2) + 1.0),
                ),
            ),
        ];
        let mut rules = BTreeMap::new();
        for (name, table) in tables {
            let table = table.expect("valid table");
            let grid = table.grid_y();
            let mut rung = 0_i32;
            let mut worst = 0.0_f64;
            let mut largest = 0usize;
            for k in 0..grid.len() - 1 {
                let cell = table.cell(0, k);
                let order = cell
                    .certified_gauss_order(grid[k], &mut rung)
                    .expect("certified order");
                if order == 0 {
                    continue;
                }
                largest = largest.max(order);
                let reference_order = 2 * order + 16;
                for n in [order, reference_order] {
                    rules.entry(n).or_insert_with(|| gauss_legendre_certified(n));
                }
                let mass = latent_mass_lower_bound(cell.h0, cell.h1);
                let reach = grid[k].abs().max(grid[k + 1].abs());
                for (moment, scale) in [(true, reach), (false, 1.0)] {
                    let target = f64::EPSILON * scale * mass;
                    let (sum, rounding) = cell_rule_sum(&cell, grid[k], grid[k + 1], &rules[&order], moment);
                    let (reference, reference_rounding) =
                        cell_rule_sum(&cell, grid[k], grid[k + 1], &rules[&reference_order], moment);
                    let gap = (sum - reference).abs();
                    let allowed = 2.0 * target + rounding + reference_rounding;
                    assert!(
                        gap <= allowed,
                        "{name}: cell {k} ({order} points, moment {moment}) differs from \
                         {reference_order} points by {gap:.3e} > {allowed:.3e}"
                    );
                    worst = worst.max(gap / allowed);
                }
            }
            println!("{name}: largest order {largest}, worst gap/allowed {worst:.3e}");
        }
    }
}
