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

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

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
        let g = self.grid_y.len();
        let h = self.h.row(row);
        let slope = self.h_prime.row(row);
        if target <= h[0] {
            return self.grid_y[0] + (target - h[0]) / slope[0];
        }
        if target >= h[g - 1] {
            return self.grid_y[g - 1] + (target - h[g - 1]) / slope[g - 1];
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
        self.grid_y[lo] + cell.width * cell.invert(target)
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
        for _ in 0..64 {
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
}
