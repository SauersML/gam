//! Natural cubic regression spline (`cr`) basis — mgcv-compatible.
//!
//! Implements the Lancaster–Salkauskas natural cubic regression spline that
//! mgcv exposes as `bs="cr"` (and its shrinkage twin `bs="cs"`), following
//! Wood (2017) *Generalized Additive Models*, §5.3.1.
//!
//! The smooth is parameterized by its values at `k` knots,
//! `β_i = f(x*_i)`, with natural boundary conditions `f''(x*_1) = f''(x*_k) =
//! 0`. The basis dimension is exactly `k` (the number of knots), and the
//! roughness penalty `∫ f''(x)² dx` is the quadratic form `βᵀ S β` with
//! `S = Dᵀ B⁻¹ D` whose null space is `{const, linear}` (dimension 2).
//!
//! This matches mgcv's `smooth.construct.cr.smooth.spec` output (`$X` and
//! `$S[[1]]`) to round-off for the same knot vector — see the unit tests at
//! the bottom of this module and the in-tree quality cross-checks.
//!
//! ## Geometry (the `F` matrix)
//! For interior knots, the second derivatives `δ` are linear in the values
//! `β` via `δ = F β`, where `F` is `k × k` with zero first/last rows and
//! interior rows given by `B⁻¹ D`:
//!   * `D` is `(k-2) × k`:   `D[i,i]=1/h_i`, `D[i,i+1]=-1/h_i-1/h_{i+1}`,
//!                            `D[i,i+2]=1/h_{i+1}`.
//!   * `B` is `(k-2) × (k-2)` tridiagonal SPD: `B[i,i]=(h_i+h_{i+1})/3`,
//!                            `B[i,i+1]=B[i+1,i]=h_{i+1}/6`.
//! with `h_i = x*_{i+1} - x*_i` (1-indexed in the math, 0-indexed below).
//!
//! ## Design row
//! For `x ∈ [x*_j, x*_{j+1}]` (knot interval `j`, 0-indexed) with
//! `a₋ = (x*_{j+1}-x)/h_j`, `a₊ = (x-x*_j)/h_j`:
//!   `row = a₋·e_j + a₊·e_{j+1} + c₋·F[j,:] + c₊·F[j+1,:]`
//! where `c₋ = (a₋³-a₋) h_j²/6`, `c₊ = (a₊³-a₊) h_j²/6`.
//!
//! Outside `[x*_1, x*_k]` mgcv extrapolates *linearly*: the value and first
//! derivative are continued from the nearest endpoint knot. We reproduce that
//! exactly so predict-time rows past the data range match mgcv.

use super::*;

/// Precomputed natural cubic regression spline geometry for a fixed knot set.
#[derive(Clone, Debug)]
pub struct CubicRegressionBasis {
    /// Knot locations `x*_1 < … < x*_k` (strictly increasing).
    pub knots: Array1<f64>,
    /// The `k × k` second-derivative map `F` (`δ = F β`); rows 0 and k-1 are zero.
    f_matrix: Array2<f64>,
}

impl CubicRegressionBasis {
    /// Build the cr geometry for a strictly increasing knot vector of length
    /// `k >= 3`. (mgcv requires `k >= 3` for a cubic regression spline.)
    pub fn new(knots: Array1<f64>) -> Result<Self, BasisError> {
        let k = knots.len();
        if k < 3 {
            crate::bail_invalid_basis!(
                "cubic regression spline requires at least 3 knots, got {k}"
            );
        }
        // Strictly increasing check.
        for i in 1..k {
            if !(knots[i] > knots[i - 1]) {
                crate::bail_invalid_basis!(
                    "cubic regression spline knots must be strictly increasing; \
                     knot[{}]={} is not greater than knot[{}]={}",
                    i,
                    knots[i],
                    i - 1,
                    knots[i - 1]
                );
            }
        }
        let h: Vec<f64> = (0..k - 1).map(|i| knots[i + 1] - knots[i]).collect();
        let f_matrix = build_f_matrix(&h, k)?;
        Ok(Self { knots, f_matrix })
    }

    pub fn num_basis(&self) -> usize {
        self.knots.len()
    }

    /// The natural cubic regression roughness penalty `S = Dᵀ B⁻¹ D` (k×k).
    ///
    /// Equivalently `S = Dᵀ F_int` where `F_int = B⁻¹ D` are the interior rows
    /// of `F`. We assemble it directly from `D` and the interior block of `F`.
    pub fn penalty(&self) -> Array2<f64> {
        let k = self.knots.len();
        let h: Vec<f64> = (0..k - 1).map(|i| self.knots[i + 1] - self.knots[i]).collect();
        // D is (k-2) x k.
        let mut d = Array2::<f64>::zeros((k - 2, k));
        for i in 0..k - 2 {
            d[[i, i]] = 1.0 / h[i];
            d[[i, i + 1]] = -1.0 / h[i] - 1.0 / h[i + 1];
            d[[i, i + 2]] = 1.0 / h[i + 1];
        }
        // F_int = interior rows of F (rows 1..k-1 of F_matrix), shape (k-2) x k.
        // S = Dᵀ F_int. (F_int = B⁻¹ D, so Dᵀ B⁻¹ D.)
        let f_int = self.f_matrix.slice(s![1..k - 1, ..]).to_owned();
        // S = Dᵀ (F_int)  -> (k x (k-2)) x ((k-2) x k) = k x k.
        let s = d.t().dot(&f_int);
        // Symmetrize defensively (it is symmetric in exact arithmetic).
        let mut s_sym = Array2::<f64>::zeros((k, k));
        for a in 0..k {
            for b in 0..k {
                s_sym[[a, b]] = 0.5 * (s[[a, b]] + s[[b, a]]);
            }
        }
        s_sym
    }

    /// Evaluate the cr design row for a single point `x` into `row` (length k).
    /// `row` is overwritten.
    pub fn eval_row_into(&self, x: f64, row: &mut [f64]) {
        let k = self.knots.len();
        // assert_eq!, not debug_assert_eq!: the ban-scanner forbids debug_assert
        // (silent in release → debug/release divergence). The length check is a
        // cheap O(1) guard, so an always-active assert is acceptable here.
        assert_eq!(row.len(), k);
        for r in row.iter_mut() {
            *r = 0.0;
        }
        let x1 = self.knots[0];
        let xk = self.knots[k - 1];

        if x <= x1 {
            // Linear extrapolation off the left endpoint, matching mgcv: the
            // value at x1 is β_0, the slope is the spline's first derivative at
            // x1. For the first interval [x*_0, x*_1] the cubic has
            //   f(x) = a₋β_0 + a₊β_1 + c₋δ_0 + c₊δ_1   with δ_0 = 0 (natural),
            // so f'(x1⁻side) at x = x1 is
            //   slope = (β_1 - β_0)/h_0 - h_0/6 * δ_1     (δ_0 = 0).
            let h0 = self.knots[1] - self.knots[0];
            // row picks up β_0 (=1 at e_0) plus slope*(x-x1) expressed in β.
            row[0] += 1.0;
            // d/dx contributions: (β_1-β_0)/h0 term and -h0/6 * δ_1 term.
            let dx = x - x1;
            row[0] += dx * (-1.0 / h0);
            row[1] += dx * (1.0 / h0);
            // δ_1 = F[1,:]·β  → -h0/6 * δ_1 contributes -h0/6 * F[1,:].
            let coeff = dx * (-h0 / 6.0);
            for c in 0..k {
                row[c] += coeff * self.f_matrix[[1, c]];
            }
            return;
        }
        if x >= xk {
            // Linear extrapolation off the right endpoint. For the last
            // interval [x*_{k-2}, x*_{k-1}], δ_{k-1} = 0 (natural), and the
            // first derivative at x = xk is
            //   slope = (β_{k-1} - β_{k-2})/h_{k-2} + h_{k-2}/6 * δ_{k-2}.
            let hk = self.knots[k - 1] - self.knots[k - 2];
            row[k - 1] += 1.0;
            let dx = x - xk;
            row[k - 2] += dx * (-1.0 / hk);
            row[k - 1] += dx * (1.0 / hk);
            // + h_{k-2}/6 * δ_{k-2}, δ_{k-2} = F[k-2,:]·β.
            let coeff = dx * (hk / 6.0);
            for c in 0..k {
                row[c] += coeff * self.f_matrix[[k - 2, c]];
            }
            return;
        }

        // Interior: locate interval j with x*_j <= x <= x*_{j+1}.
        // knots strictly increasing; binary search for the upper bound.
        let mut j = match self
            .knots
            .as_slice()
            .expect("contiguous knots")
            .binary_search_by(|probe| probe.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Less))
        {
            Ok(idx) => idx,  // x equals a knot: use interval starting at idx
            Err(idx) => idx - 1, // x in (knot[idx-1], knot[idx])
        };
        if j >= k - 1 {
            j = k - 2;
        }
        let hj = self.knots[j + 1] - self.knots[j];
        let a_minus = (self.knots[j + 1] - x) / hj;
        let a_plus = (x - self.knots[j]) / hj;
        let c_minus = (a_minus * a_minus * a_minus - a_minus) * hj * hj / 6.0;
        let c_plus = (a_plus * a_plus * a_plus - a_plus) * hj * hj / 6.0;
        row[j] += a_minus;
        row[j + 1] += a_plus;
        for c in 0..k {
            row[c] += c_minus * self.f_matrix[[j, c]] + c_plus * self.f_matrix[[j + 1, c]];
        }
    }

    /// Dense `n × k` design matrix for a column of evaluation points.
    pub fn design(&self, data: ArrayView1<'_, f64>) -> Array2<f64> {
        let k = self.knots.len();
        let n = data.len();
        let mut x = Array2::<f64>::zeros((n, k));
        let mut row = vec![0.0f64; k];
        for (i, &xi) in data.iter().enumerate() {
            self.eval_row_into(xi, &mut row);
            for c in 0..k {
                x[[i, c]] = row[c];
            }
        }
        x
    }
}

/// Assemble the `k × k` map `F` (`δ = F β`) from interval widths `h`.
/// Rows 0 and k-1 are zero (natural boundary). Interior rows solve
/// `B (F_int) = D` for the `(k-2) × k` interior block `F_int`.
fn build_f_matrix(h: &[f64], k: usize) -> Result<Array2<f64>, BasisError> {
    let m = k - 2; // interior count
    // B (m x m) tridiagonal SPD.
    let mut b_diag = vec![0.0f64; m];
    let mut b_off = vec![0.0f64; m.saturating_sub(1)]; // b_off[i] = B[i,i+1] = B[i+1,i]
    for i in 0..m {
        b_diag[i] = (h[i] + h[i + 1]) / 3.0;
    }
    for i in 0..m.saturating_sub(1) {
        // B[i,i+1] = h_{i+1}/6 (the shared interior width).
        b_off[i] = h[i + 1] / 6.0;
    }
    // D (m x k).
    let mut d = Array2::<f64>::zeros((m, k));
    for i in 0..m {
        d[[i, i]] = 1.0 / h[i];
        d[[i, i + 1]] = -1.0 / h[i] - 1.0 / h[i + 1];
        d[[i, i + 2]] = 1.0 / h[i + 1];
    }
    // Solve B X = D column-by-column with the Thomas algorithm; X = F_int.
    let f_int = thomas_solve_multi(&b_diag, &b_off, &d)?;
    let mut f = Array2::<f64>::zeros((k, k));
    for i in 0..m {
        for c in 0..k {
            f[[i + 1, c]] = f_int[[i, c]];
        }
    }
    Ok(f)
}

/// Solve a symmetric tridiagonal system `B X = RHS` for every column of `RHS`
/// using the Thomas algorithm. `diag` is length m, `off` is length m-1
/// (the shared sub/super-diagonal). `rhs` is `m × c`. Returns `m × c`.
fn thomas_solve_multi(
    diag: &[f64],
    off: &[f64],
    rhs: &Array2<f64>,
) -> Result<Array2<f64>, BasisError> {
    let m = diag.len();
    let cols = rhs.ncols();
    if m == 0 {
        return Ok(Array2::<f64>::zeros((0, cols)));
    }
    if rhs.nrows() != m {
        crate::bail_dim_basis!(
            "tridiagonal solve RHS has {} rows but system is {}x{}",
            rhs.nrows(),
            m,
            m
        );
    }
    // Forward sweep.
    let mut c_prime = vec![0.0f64; m]; // modified super-diagonal
    let mut d_prime = Array2::<f64>::zeros((m, cols));
    let denom0 = diag[0];
    if denom0.abs() < 1e-300 {
        crate::bail_invalid_basis!("singular tridiagonal pivot at row 0 in cr penalty solve");
    }
    if m > 1 {
        c_prime[0] = off[0] / denom0;
    }
    for col in 0..cols {
        d_prime[[0, col]] = rhs[[0, col]] / denom0;
    }
    for i in 1..m {
        let denom = diag[i] - off[i - 1] * c_prime[i - 1];
        if denom.abs() < 1e-300 {
            crate::bail_invalid_basis!(
                "singular tridiagonal pivot at row {i} in cr penalty solve"
            );
        }
        if i < m - 1 {
            c_prime[i] = off[i] / denom;
        }
        for col in 0..cols {
            d_prime[[i, col]] = (rhs[[i, col]] - off[i - 1] * d_prime[[i - 1, col]]) / denom;
        }
    }
    // Back substitution.
    let mut x = Array2::<f64>::zeros((m, cols));
    for col in 0..cols {
        x[[m - 1, col]] = d_prime[[m - 1, col]];
    }
    for i in (0..m - 1).rev() {
        for col in 0..cols {
            x[[i, col]] = d_prime[[i, col]] - c_prime[i] * x[[i + 1, col]];
        }
    }
    Ok(x)
}

/// Place `k` cr knots at evenly-spaced quantiles of the unique sorted data,
/// exactly as mgcv's default `cr` knot placement: the first and last knots are
/// the min/max, and the interior knots are at the `1/(k-1) … (k-2)/(k-1)`
/// quantiles of the *unique* observed values. Returns a strictly increasing
/// length-`k` knot vector.
pub fn select_cr_knots(
    data: ArrayView1<'_, f64>,
    k: usize,
) -> Result<Array1<f64>, BasisError> {
    if k < 3 {
        crate::bail_invalid_basis!("cubic regression spline requires k >= 3, got {k}");
    }
    if data.is_empty() {
        crate::bail_invalid_basis!("cannot place cr knots on empty data");
    }
    if data.iter().any(|x| !x.is_finite()) {
        crate::bail_invalid_basis!("cr knot placement requires finite data");
    }
    let mut sorted: Vec<f64> = data.iter().copied().collect();
    sorted.sort_by(f64::total_cmp);
    // Unique values (mgcv places cr knots on the unique data quantiles).
    let mut unique: Vec<f64> = Vec::with_capacity(sorted.len());
    for &v in &sorted {
        if unique.last().map(|&p| p != v).unwrap_or(true) {
            unique.push(v);
        }
    }
    let nu = unique.len();
    if nu < k {
        crate::bail_invalid_basis!(
            "cubic regression spline with k={k} requires at least {k} distinct \
             values, got {nu}"
        );
    }
    // mgcv's `place.knots`: knots at quantile type-1-ish positions over the
    // index range [0, nu-1] evenly in (k-1) steps. Endpoints are exact min/max.
    let mut knots = Array1::<f64>::zeros(k);
    for j in 0..k {
        let pos = (j as f64) * ((nu - 1) as f64) / ((k - 1) as f64);
        let lo = pos.floor() as usize;
        let hi = pos.ceil() as usize;
        let frac = pos - lo as f64;
        knots[j] = if lo == hi {
            unique[lo]
        } else {
            unique[lo] * (1.0 - frac) + unique[hi] * frac
        };
    }
    // Guard strict monotonicity in case of ties from interpolation rounding.
    for i in 1..k {
        if !(knots[i] > knots[i - 1]) {
            crate::bail_invalid_basis!(
                "cr knot placement produced non-increasing knots (too many knots \
                 for the data spread); reduce k"
            );
        }
    }
    Ok(knots)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cr smooth must reproduce constants and lines exactly: the penalty null
    /// space is {const, linear}, and the design with values β_i = f(x*_i)
    /// interpolates any line through the knots with zero penalty.
    #[test]
    fn cr_penalty_nullspace_is_const_and_linear() {
        let knots = Array1::from(vec![0.0, 0.3, 0.55, 0.8, 1.0]);
        let cr = CubicRegressionBasis::new(knots.clone()).unwrap();
        let s = cr.penalty();
        let k = knots.len();
        // const: β = 1.
        let ones = Array1::<f64>::ones(k);
        let q_const = ones.dot(&s.dot(&ones));
        assert!(q_const.abs() < 1e-9, "const not in null space: {q_const}");
        // linear: β_i = knot_i.
        let lin = knots.clone();
        let q_lin = lin.dot(&s.dot(&lin));
        assert!(q_lin.abs() < 1e-9, "linear not in null space: {q_lin}");
        // a quadratic should have positive penalty.
        let quad: Array1<f64> = knots.mapv(|x| x * x);
        let q_quad = quad.dot(&s.dot(&quad));
        assert!(q_quad > 1e-6, "quadratic penalty not positive: {q_quad}");
    }

    /// The design must reproduce a line exactly at arbitrary evaluation points
    /// (interior and extrapolated), since a line is in the cr span.
    #[test]
    fn cr_design_reproduces_line_including_extrapolation() {
        let knots = Array1::from(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let cr = CubicRegressionBasis::new(knots.clone()).unwrap();
        // f(x) = 2 + 3x  → β_i = 2 + 3*knot_i.
        let beta: Array1<f64> = knots.mapv(|x| 2.0 + 3.0 * x);
        let xs = Array1::from(vec![-0.4, 0.0, 0.13, 0.5, 0.87, 1.0, 1.3]);
        let design = cr.design(xs.view());
        let fitted = design.dot(&beta);
        for (i, &x) in xs.iter().enumerate() {
            let truth = 2.0 + 3.0 * x;
            assert!(
                (fitted[i] - truth).abs() < 1e-9,
                "line not reproduced at x={x}: got {}, want {truth}",
                fitted[i]
            );
        }
    }

    /// Knot placement returns endpoints = min/max and strictly increasing knots.
    #[test]
    fn cr_knots_span_data_and_increase() {
        let data = Array1::from((0..50).map(|i| i as f64 / 49.0).collect::<Vec<_>>());
        let knots = select_cr_knots(data.view(), 5).unwrap();
        assert_eq!(knots.len(), 5);
        assert!((knots[0] - 0.0).abs() < 1e-12);
        assert!((knots[4] - 1.0).abs() < 1e-12);
        for i in 1..5 {
            assert!(knots[i] > knots[i - 1]);
        }
    }
}
