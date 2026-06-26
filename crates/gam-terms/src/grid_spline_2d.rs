//! Streaming scatter-add 2-D smoother: K×K tensor-product cubic B-splines
//! with the EXACT anisotropic biharmonic penalty and REML-selected λ.
//!
//! Basis. Each axis carries K equal-width cells over the data's bounding box
//! `[lo, hi]` with uniform extended knots `t_j = lo + (j−3)·h`, `h = (hi−lo)/K`,
//! giving `m = K+3` cubic B-splines per axis; the tensor product has
//! `p = (K+3)²` coefficients. A point in cell `i` activates exactly the four
//! splines `i..i+3` per axis, hence exactly 4×4 = 16 tensor basis entries per
//! data row.
//!
//! Streaming normal equations. ONE pass over the rows `(x1, x2, y_·, w)`
//! scatter-adds `X'WX` and `X'Wy_d` (any number of response dimensions share
//! the design, the penalty, and one REML λ — the multi-output "one surface
//! smoothness" contract of the ANOVA pair component): O(n·(16² + 16·D)) work,
//! no n×p design is ever materialized. Two tensor bases overlap only when both per-axis indices
//! differ by ≤ 3, so under the row-major coefficient index
//! `g = j1·(K+3) + j2` both `X'WX` and the penalty `S` are banded with
//! half-bandwidth `3(K+3)+3`; they are stored as upper bands — O(K³) numbers.
//!
//! Penalty. The FULL anisotropic biharmonic form for the diagonal metric
//! `A = diag(a1, a2)`,
//!   `J(f) = ∫∫ a1²·f_{x1x1}² + 2·a1·a2·f_{x1x2}² + a2²·f_{x2x2}²  dx1 dx2`,
//! INCLUDING the mixed `f_{x1x2}` term (the axis-wise P-spline difference
//! shortcut drops it), assembled per knot cell by 4-point Gauss–Legendre per
//! axis. Exactness degree arithmetic: on a knot cell every basis function is
//! a single cubic polynomial per axis, so each entry of `S` is a sum over
//! cells of integrands that factorize per axis as one of value·value
//! (degree 3+3 = 6), deriv·deriv (2+2 = 4) or 2nd-deriv·2nd-deriv (1+1 = 2);
//! every channel pairs a low-degree factor on one axis with at worst the
//! degree-6 value·value factor on the other. 4-point Gauss–Legendre is exact
//! through degree 2·4−1 = 7 ≥ 6, so the assembled `S` is the EXACT integral,
//! not a quadrature approximation.
//!
//! Solve and selection. For a trial λ the bands are expanded and
//! `(X'WX + λS)c = X'Wy` is solved by dense Cholesky with the exact
//! log-determinant read off the factor. `p ≤ (32+3)² = 1225`, so the O(p³)
//! factorization costs ≲ 1 Gflop per trial and the retained factor is ≤ 12 MB;
//! K is capped at 32 to keep that sizing contract honest. λ maximizes the
//! profiled-σ² restricted (REML) criterion
//!   `ℓ_R(λ) = −½[ log|X'WX+λS| − r·log λ + (n−3)·log σ̂²(λ) ] + const`,
//! where `r = p−3` is the penalty rank — the null space of `J` is
//! span{1, x1, x2} (the mixed term penalizes `x1·x2`, whose cross derivative
//! is 1 ≠ 0, so it is NOT in the null space), `σ̂²(λ) = (y'Wy − c'X'Wy)/(n−3)`
//! is the profiled scale, and the λ-free additive constants (`log|S|₊` on the
//! row space of S, `Σ log w`, 2π factors) are dropped: differences across λ
//! are exact REML criterion differences. Selection is the same deterministic
//! coarse-grid + golden-section scheme as `spline_scan` — no RNG, same data ⇒
//! same fit.
//!
//! Prediction. `predict(x1, x2)` builds the 16-entry basis row; the mean is
//! its dot with `c` and the variance is the Bayesian posterior
//! `σ̂²·x'(X'WX+λS)⁻¹x` through the retained Cholesky factor. Outside the
//! bounding box the boundary cell's cubic polynomial extends naturally (the
//! cell index clamps, the local coordinate does not).

/// Dimension of the penalty null space: span{1, x1, x2}. The mixed
/// `2·a1·a2·f_{x1x2}²` term excludes `x1·x2` (its cross derivative is 1).
const PENALTY_NULLITY: usize = 3;

/// Deterministic coarse-grid width for the log-λ search.
const LOG_LAMBDA_GRID: usize = 25;
/// Search interval for log λ (natural log), generous on both sides.
const LOG_LAMBDA_LO: f64 = -18.0;
const LOG_LAMBDA_HI: f64 = 18.0;
/// Golden-section refinement tolerance on log λ.
const LOG_LAMBDA_TOL: f64 = 1e-7;
/// Cholesky pivot floor below which the penalized system is declared singular.
const PIVOT_FLOOR: f64 = 1e-300;
/// Dense-Cholesky sizing contract documented in the module header.
const MAX_CELLS_PER_AXIS: usize = 32;

/// 4-point Gauss–Legendre nodes and weights on [−1, 1]. Exact through degree
/// 2·4−1 = 7, which dominates the degree-6 worst per-axis factor of the
/// penalty integrands (see the module header for the degree arithmetic).
const GL4_NODES: [f64; 4] = [
    -0.861_136_311_594_052_6,
    -0.339_981_043_584_856_26,
    0.339_981_043_584_856_26,
    0.861_136_311_594_052_6,
];
const GL4_WEIGHTS: [f64; 4] = [
    0.347_854_845_137_453_85,
    0.652_145_154_862_546_2,
    0.652_145_154_862_546_2,
    0.347_854_845_137_453_85,
];

/// Cubic B-spline segment values at local coordinate `u` within a cell.
/// Entry `m` weights basis `cell + m`: m = 0 is the spline ENDING in this
/// cell (`(1−u)³/6`), m = 3 the one STARTING (`u³/6`). The four entries sum
/// to 1 (partition of unity) for u ∈ [0, 1].
#[inline]
fn bspline_value(u: f64) -> [f64; 4] {
    let v = 1.0 - u;
    [
        v * v * v / 6.0,
        (3.0 * u * u * u - 6.0 * u * u + 4.0) / 6.0,
        (-3.0 * u * u * u + 3.0 * u * u + 3.0 * u + 1.0) / 6.0,
        u * u * u / 6.0,
    ]
}

/// d/du of `bspline_value` (caller scales by 1/h for d/dx). Entries sum to 0.
#[inline]
fn bspline_d1(u: f64) -> [f64; 4] {
    let v = 1.0 - u;
    [
        -0.5 * v * v,
        0.5 * (3.0 * u * u - 4.0 * u),
        0.5 * (-3.0 * u * u + 2.0 * u + 1.0),
        0.5 * u * u,
    ]
}

/// d²/du² of `bspline_value` (caller scales by 1/h²). Piecewise LINEAR in u —
/// the degree-1 factor in the quadrature-exactness argument. Entries sum to 0.
#[inline]
fn bspline_d2(u: f64) -> [f64; 4] {
    [1.0 - u, 3.0 * u - 2.0, 1.0 - 3.0 * u, u]
}

/// One uniform B-spline axis over `[lo, lo + cells·h]`.
#[derive(Clone, Copy, Debug)]
struct Axis {
    lo: f64,
    h: f64,
    cells: usize,
}

impl Axis {
    /// Cell index and local coordinate. Inside the box `u ∈ [0, 1]`; outside,
    /// the cell clamps and `u` leaves [0, 1], extending the boundary cell's
    /// cubic polynomial (deterministic extrapolation, no special casing).
    #[inline]
    fn locate(&self, x: f64) -> (usize, f64) {
        let t = (x - self.lo) / self.h;
        let cell = (t.floor().max(0.0) as usize).min(self.cells - 1);
        (cell, t - cell as f64)
    }
}

/// The four active cubic B-spline values of one uniform axis `(lo, h, cells)`
/// at `x`: `(first basis index, values)`, where `values[i]` weights basis
/// `first + i` of the `cells + 3` axis splines. Outside `[lo, lo + cells·h]`
/// the boundary cell's cubic polynomial extends — the single convention
/// shared by fitting, prediction, and every consumer-rebuilt basis row.
pub fn axis_basis_at(lo: f64, h: f64, cells: usize, x: f64) -> (usize, [f64; 4]) {
    let (cell, u) = Axis { lo, h, cells }.locate(x);
    (cell, bspline_value(u))
}

/// The 16 active tensor-basis entries `(flat index, value)` at `(x1, x2)`.
/// Flat indices are strictly increasing across the returned arrays.
#[inline]
fn basis_row(axes: &[Axis; 2], m_axis: usize, x1: f64, x2: f64) -> ([usize; 16], [f64; 16]) {
    let (c1, u1) = axes[0].locate(x1);
    let (c2, u2) = axes[1].locate(x2);
    let b1 = bspline_value(u1);
    let b2 = bspline_value(u2);
    let mut idx = [0usize; 16];
    let mut val = [0f64; 16];
    for i in 0..4 {
        for j in 0..4 {
            idx[4 * i + j] = (c1 + i) * m_axis + (c2 + j);
            val[4 * i + j] = b1[i] * b2[j];
        }
    }
    (idx, val)
}

/// Dense lower-Cholesky in place (row-major `p×p`); returns the exact
/// `log det` (twice the log of the pivot products). The strict upper triangle
/// is zeroed so the buffer is exactly `L` afterwards.
pub fn cholesky_logdet(a: &mut [f64], p: usize) -> Result<f64, String> {
    let mut logdet = 0.0;
    for j in 0..p {
        let mut s = a[j * p + j];
        for t in 0..j {
            s -= a[j * p + t] * a[j * p + t];
        }
        if !(s.is_finite() && s > PIVOT_FLOOR) {
            return Err(format!(
                "grid spline 2d: penalized system not positive definite at pivot {j} (value {s})"
            ));
        }
        let l = s.sqrt();
        a[j * p + j] = l;
        logdet += 2.0 * l.ln();
        for i in j + 1..p {
            let mut s2 = a[i * p + j];
            for t in 0..j {
                s2 -= a[i * p + t] * a[j * p + t];
            }
            a[i * p + j] = s2 / l;
        }
    }
    for i in 0..p {
        for j in i + 1..p {
            a[i * p + j] = 0.0;
        }
    }
    Ok(logdet)
}

/// Solve `L Lᵀ x = b` from the stored lower factor.
pub fn chol_solve(l: &[f64], p: usize, b: &[f64]) -> Vec<f64> {
    let mut z = b.to_vec();
    for i in 0..p {
        let mut s = z[i];
        for t in 0..i {
            s -= l[i * p + t] * z[t];
        }
        z[i] = s / l[i * p + i];
    }
    for i in (0..p).rev() {
        let mut s = z[i];
        for t in i + 1..p {
            s -= l[t * p + i] * z[t];
        }
        z[i] = s / l[i * p + i];
    }
    z
}

/// Banded sufficient statistics of one streaming pass plus the exact penalty:
/// everything needed to evaluate the REML criterion and solve at any λ.
pub struct GridSpline2dDesign {
    axes: [Axis; 2],
    /// Basis count per axis, `K + 3`.
    m_axis: usize,
    /// Total coefficients, `(K + 3)²`.
    p: usize,
    /// Upper half-bandwidth `3·(K+3) + 3` of both banded matrices.
    band_half: usize,
    /// Upper band of `X'WX`: entry `(g, g+d)` at `g·(band_half+1) + d`.
    gram_band: Vec<f64>,
    /// Upper band of the exact anisotropic biharmonic penalty `S`.
    pen_band: Vec<f64>,
    /// `X'Wy_d`, one length-`p` vector per response dimension. The design
    /// (gram and penalty bands) is shared across dimensions; only these
    /// right-hand sides and the response cross-moments are per-dimension.
    rhs: Vec<Vec<f64>>,
    /// Response cross-moments `y_d'W y_e` (`D × D` row-major), for the
    /// profiled-σ² residual quadratics and the residual cross-covariance.
    cross_moments: Vec<f64>,
    n_obs: usize,
}

/// Internal solve product at one λ (all response dimensions share the factor).
struct Solved {
    chol: Vec<f64>,
    logdet: f64,
    coeffs: Vec<Vec<f64>>,
    /// Per dimension: penalized residual quadratic `y'Wy − c'X'Wy` =
    /// `‖√W(y − Xc)‖² + λ c'Sc` at the minimizer.
    rss_pen: Vec<f64>,
}

impl GridSpline2dDesign {
    /// Single-response entry: see [`Self::build_multi`].
    pub fn build(
        x1: &[f64],
        x2: &[f64],
        y: &[f64],
        w: &[f64],
        k: usize,
        metric: [f64; 2],
    ) -> Result<Self, String> {
        Self::build_multi(x1, x2, &[y], w, k, metric)
    }

    /// One streaming pass over the rows plus the exact per-cell quadrature
    /// assembly of the penalty. `k` is the number of cells per axis;
    /// `metric = [a1, a2]` is the diagonal anisotropy of the biharmonic form.
    /// `responses` holds one length-`n` response per dimension; the design,
    /// penalty, and the REML-shared λ are common to all dimensions (one
    /// surface smoothness), only the right-hand sides differ.
    pub fn build_multi(
        x1: &[f64],
        x2: &[f64],
        responses: &[&[f64]],
        w: &[f64],
        k: usize,
        metric: [f64; 2],
    ) -> Result<Self, String> {
        let n = x1.len();
        if responses.is_empty() {
            return Err("grid spline 2d: no response dimensions supplied".to_string());
        }
        if x2.len() != n || w.len() != n {
            return Err(format!(
                "grid spline 2d: length mismatch x1={n}, x2={}, w={}",
                x2.len(),
                w.len()
            ));
        }
        for (d, y) in responses.iter().enumerate() {
            if y.len() != n {
                return Err(format!(
                    "grid spline 2d: response dimension {d} has length {} != {n}",
                    y.len()
                ));
            }
        }
        if n <= PENALTY_NULLITY {
            return Err(format!(
                "grid spline 2d: needs more than {PENALTY_NULLITY} rows for the profiled REML \
                 degrees of freedom, got {n}"
            ));
        }
        if k == 0 || k > MAX_CELLS_PER_AXIS {
            return Err(format!(
                "grid spline 2d: k must be in 1..={MAX_CELLS_PER_AXIS} (dense Cholesky on \
                 (k+3)² coefficients — see module sizing contract), got {k}"
            ));
        }
        if !(metric[0].is_finite() && metric[0] > 0.0 && metric[1].is_finite() && metric[1] > 0.0) {
            return Err(format!(
                "grid spline 2d: metric diagonal must be finite and positive, got [{}, {}]",
                metric[0], metric[1]
            ));
        }
        for i in 0..n {
            if !(x1[i].is_finite() && x2[i].is_finite()) || !(w[i] > 0.0) || !w[i].is_finite() {
                return Err(format!(
                    "grid spline 2d: non-finite or non-positive input at row {i} \
                     (x1={}, x2={}, w={})",
                    x1[i], x2[i], w[i]
                ));
            }
            for (d, y) in responses.iter().enumerate() {
                if !y[i].is_finite() {
                    return Err(format!(
                        "grid spline 2d: non-finite response at row {i}, dimension {d} ({})",
                        y[i]
                    ));
                }
            }
        }
        let mut axes = [Axis {
            lo: 0.0,
            h: 1.0,
            cells: k,
        }; 2];
        for (axis, xs) in axes.iter_mut().zip([x1, x2]) {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &v in xs {
                lo = lo.min(v);
                hi = hi.max(v);
            }
            if !(hi > lo) {
                return Err(format!(
                    "grid spline 2d: degenerate axis bounding box [{lo}, {hi}]"
                ));
            }
            axis.lo = lo;
            axis.h = (hi - lo) / k as f64;
        }
        let m_axis = k + 3;
        let p = m_axis * m_axis;
        let band_half = 3 * m_axis + 3;
        let stride = band_half + 1;
        let n_dims = responses.len();
        let mut gram_band = vec![0.0_f64; p * stride];
        let mut rhs = vec![vec![0.0_f64; p]; n_dims];
        let mut cross_moments = vec![0.0_f64; n_dims * n_dims];

        // ── ONE streaming pass: scatter-add X'WX (upper band) and X'Wy_d ──
        // Each row touches exactly 16 basis entries with strictly increasing
        // flat indices, so the in-row pair loop (a ≤ b) lands directly in the
        // upper band: O(n·(16² + 16·D)) total work.
        for i in 0..n {
            let (idx, val) = basis_row(&axes, m_axis, x1[i], x2[i]);
            let wi = w[i];
            for (d, y) in responses.iter().enumerate() {
                let wy = wi * y[i];
                for e in 0..16 {
                    rhs[d][idx[e]] += wy * val[e];
                }
                for (e, ye) in responses.iter().enumerate().skip(d) {
                    cross_moments[d * n_dims + e] += wy * ye[i];
                }
            }
            for a in 0..16 {
                let base = idx[a] * stride - idx[a];
                let wa = wi * val[a];
                for b in a..16 {
                    gram_band[base + idx[b]] += wa * val[b];
                }
            }
        }
        for d in 0..n_dims {
            for e in 0..d {
                cross_moments[d * n_dims + e] = cross_moments[e * n_dims + d];
            }
        }

        // ── Exact penalty assembly: 4-pt Gauss–Legendre per axis per cell ──
        // Per-axis quadrature tables (cell-independent on a uniform grid):
        // values, d/dx (scaled 1/h), d²/dx² (scaled 1/h²) at each GL node.
        let mut tab = [[[[0.0_f64; 4]; 4]; 3]; 2]; // [axis][channel][node][basis offset]
        for ax in 0..2 {
            let h = axes[ax].h;
            for q in 0..4 {
                let u = 0.5 * (1.0 + GL4_NODES[q]);
                let v0 = bspline_value(u);
                let v1 = bspline_d1(u);
                let v2 = bspline_d2(u);
                for e in 0..4 {
                    tab[ax][0][q][e] = v0[e];
                    tab[ax][1][q][e] = v1[e] / h;
                    tab[ax][2][q][e] = v2[e] / (h * h);
                }
            }
        }
        // Channel scales: J = ∫ a1²·f11² + 2·a1·a2·f12² + a2²·f22².
        let s11 = metric[0] * metric[0];
        let s12 = 2.0 * metric[0] * metric[1];
        let s22 = metric[1] * metric[1];
        let cell_area_jac = 0.25 * axes[0].h * axes[1].h; // d(x1,x2)/d(ξ1,ξ2) on [−1,1]²
        let mut pen_band = vec![0.0_f64; p * stride];
        let mut r11 = [0.0_f64; 16];
        let mut r12 = [0.0_f64; 16];
        let mut r22 = [0.0_f64; 16];
        let mut idx = [0usize; 16];
        for c1 in 0..k {
            for c2 in 0..k {
                for i in 0..4 {
                    for j in 0..4 {
                        idx[4 * i + j] = (c1 + i) * m_axis + (c2 + j);
                    }
                }
                for q1 in 0..4 {
                    for q2 in 0..4 {
                        let wq = cell_area_jac * GL4_WEIGHTS[q1] * GL4_WEIGHTS[q2];
                        for i in 0..4 {
                            for j in 0..4 {
                                let e = 4 * i + j;
                                r11[e] = tab[0][2][q1][i] * tab[1][0][q2][j];
                                r12[e] = tab[0][1][q1][i] * tab[1][1][q2][j];
                                r22[e] = tab[0][0][q1][i] * tab[1][2][q2][j];
                            }
                        }
                        for a in 0..16 {
                            let base = idx[a] * stride - idx[a];
                            let (pa11, pa12, pa22) =
                                (wq * s11 * r11[a], wq * s12 * r12[a], wq * s22 * r22[a]);
                            for b in a..16 {
                                pen_band[base + idx[b]] +=
                                    pa11 * r11[b] + pa12 * r12[b] + pa22 * r22[b];
                            }
                        }
                    }
                }
            }
        }

        Ok(GridSpline2dDesign {
            axes,
            m_axis,
            p,
            band_half,
            gram_band,
            pen_band,
            rhs,
            cross_moments,
            n_obs: n,
        })
    }

    /// Number of cells per axis (the caller-supplied K).
    pub fn num_cells(&self) -> usize {
        self.axes[0].cells
    }

    /// Basis functions per axis, `K + 3`.
    pub fn basis_per_axis(&self) -> usize {
        self.m_axis
    }

    /// Total coefficient count `(K + 3)²`.
    pub fn num_coeffs(&self) -> usize {
        self.p
    }

    /// Lower corner of the data bounding box per axis.
    pub fn lower_corner(&self) -> [f64; 2] {
        [self.axes[0].lo, self.axes[1].lo]
    }

    /// Knot-cell width per axis.
    pub fn cell_widths(&self) -> [f64; 2] {
        [self.axes[0].h, self.axes[1].h]
    }

    /// Number of data rows the design was streamed from.
    pub fn num_rows(&self) -> usize {
        self.n_obs
    }

    /// Number of response dimensions sharing the design.
    pub fn num_responses(&self) -> usize {
        self.rhs.len()
    }

    /// The four active cubic B-spline values of one AXIS at `x`: returns
    /// `(j0, values)` where `values[i]` weights basis `j0 + i` of that axis
    /// (`0..K+3`). The tensor flat index of `(j1, j2)` is `j1·(K+3) + j2` —
    /// row-major, axis 0 major. Outside the bounding box the boundary cell's
    /// cubic polynomial extends (same convention as fitting and prediction).
    pub fn axis_basis(&self, axis: usize, x: f64) -> Result<(usize, [f64; 4]), String> {
        if axis > 1 {
            return Err(format!("grid spline 2d: axis {axis} out of range"));
        }
        if !x.is_finite() {
            return Err(format!("grid spline 2d: non-finite axis-{axis} point {x}"));
        }
        let ax = self.axes[axis];
        Ok(axis_basis_at(ax.lo, ax.h, ax.cells, x))
    }

    /// Exact penalty quadratic form `J(f) = c'Sc` of a coefficient vector —
    /// the assembled anisotropic biharmonic energy of the spline it encodes.
    pub fn penalty_value(&self, coeff: &[f64]) -> Result<f64, String> {
        if coeff.len() != self.p {
            return Err(format!(
                "grid spline 2d: coefficient length {} != {}",
                coeff.len(),
                self.p
            ));
        }
        let stride = self.band_half + 1;
        let mut j = 0.0;
        for g in 0..self.p {
            let dmax = self.band_half.min(self.p - 1 - g);
            j += self.pen_band[g * stride] * coeff[g] * coeff[g];
            for d in 1..=dmax {
                j += 2.0 * self.pen_band[g * stride + d] * coeff[g] * coeff[g + d];
            }
        }
        Ok(j)
    }

    /// Expand `X'WX + λS` from the bands to a dense symmetric matrix.
    fn dense_system(&self, lambda: f64) -> Vec<f64> {
        let p = self.p;
        let stride = self.band_half + 1;
        let mut a = vec![0.0_f64; p * p];
        for g in 0..p {
            let dmax = self.band_half.min(p - 1 - g);
            for d in 0..=dmax {
                let v = self.gram_band[g * stride + d] + lambda * self.pen_band[g * stride + d];
                a[g * p + g + d] = v;
                a[(g + d) * p + g] = v;
            }
        }
        a
    }

    fn solve_at(&self, log_lambda: f64) -> Result<Solved, String> {
        if !log_lambda.is_finite() {
            return Err(format!(
                "grid spline 2d: non-finite log lambda {log_lambda}"
            ));
        }
        let mut a = self.dense_system(log_lambda.exp());
        let logdet = cholesky_logdet(&mut a, self.p)?;
        let n_dims = self.rhs.len();
        let mut coeffs = Vec::with_capacity(n_dims);
        let mut rss_pen = Vec::with_capacity(n_dims);
        for (d, rhs) in self.rhs.iter().enumerate() {
            let coeff = chol_solve(&a, self.p, rhs);
            let mut quad = 0.0;
            for g in 0..self.p {
                quad += rhs[g] * coeff[g];
            }
            rss_pen.push(self.cross_moments[d * n_dims + d] - quad);
            coeffs.push(coeff);
        }
        Ok(Solved {
            chol: a,
            logdet,
            coeffs,
            rss_pen,
        })
    }

    /// Profiled-σ² REML criterion at `log λ`, pooled across the response
    /// dimensions sharing the design and λ, up to λ- and data-independent
    /// additive constants (differences across λ are exact REML differences):
    ///   `−½ Σ_d [ log|X'WX+λS| − r·log λ + (n−3)·log σ̂²_d(λ) ]`.
    fn criterion(&self, log_lambda: f64) -> Result<f64, String> {
        let solved = self.solve_at(log_lambda)?;
        let dof = (self.n_obs - PENALTY_NULLITY) as f64;
        let r = (self.p - PENALTY_NULLITY) as f64;
        let shared = solved.logdet - r * log_lambda;
        let mut v = 0.0;
        for &rss in &solved.rss_pen {
            if !(rss > 0.0) {
                return Err(format!(
                    "grid spline 2d: degenerate penalized residual {rss}"
                ));
            }
            v += shared + dof * (rss / dof).ln();
        }
        Ok(-0.5 * v)
    }

    /// Fit at a FIXED `log λ`, with σ² either supplied (applied to every
    /// response dimension) or profiled per dimension.
    pub fn fit_at(&self, log_lambda: f64, sigma2: Option<f64>) -> Result<GridSpline2dFit, String> {
        let solved = self.solve_at(log_lambda)?;
        let dof = (self.n_obs - PENALTY_NULLITY) as f64;
        let mut sigma2_dims = Vec::with_capacity(solved.rss_pen.len());
        for &rss in &solved.rss_pen {
            match sigma2 {
                Some(s) => {
                    if !(s.is_finite() && s > 0.0) {
                        return Err(format!("grid spline 2d: invalid sigma2 {s}"));
                    }
                    sigma2_dims.push(s);
                }
                None => {
                    if !(rss > 0.0) {
                        return Err(format!(
                            "grid spline 2d: degenerate penalized residual {rss}"
                        ));
                    }
                    sigma2_dims.push(rss / dof);
                }
            }
        }
        // Full restricted log-likelihood at this (λ, σ²) up to λ- and σ-free
        // constants, pooled across dimensions: at the profiled σ̂²_d the
        // quadratic collapses to the λ-free constant `dof` per dimension,
        // matching `criterion` up to that constant.
        let r = (self.p - PENALTY_NULLITY) as f64;
        let mut restricted_loglik = 0.0;
        for (d, &rss) in solved.rss_pen.iter().enumerate() {
            restricted_loglik -= 0.5
                * (solved.logdet - r * log_lambda
                    + dof * sigma2_dims[d].ln()
                    + rss / sigma2_dims[d]);
        }
        Ok(GridSpline2dFit {
            coeffs: solved.coeffs,
            log_lambda,
            sigma2: sigma2_dims,
            restricted_loglik,
            chol: solved.chol,
            axes: self.axes,
            m_axis: self.m_axis,
        })
    }

    /// Fit with `log λ` selected by the profiled REML criterion: deterministic
    /// coarse grid then golden-section refinement (no RNG — same data, same fit).
    pub fn fit_reml(&self) -> Result<GridSpline2dFit, String> {
        let mut best_i = 0usize;
        let mut best_v = f64::NEG_INFINITY;
        let step = (LOG_LAMBDA_HI - LOG_LAMBDA_LO) / (LOG_LAMBDA_GRID - 1) as f64;
        for i in 0..LOG_LAMBDA_GRID {
            let ll = LOG_LAMBDA_LO + step * i as f64;
            let v = self.criterion(ll)?;
            if v > best_v {
                best_v = v;
                best_i = i;
            }
        }
        let mut lo = LOG_LAMBDA_LO + step * best_i.saturating_sub(1) as f64;
        let mut hi = (LOG_LAMBDA_LO + step * (best_i + 1) as f64).min(LOG_LAMBDA_HI);
        // Golden-section maximization on [lo, hi].
        let inv_phi = 0.618_033_988_749_894_9_f64;
        let mut x1 = hi - inv_phi * (hi - lo);
        let mut x2 = lo + inv_phi * (hi - lo);
        let mut f1 = self.criterion(x1)?;
        let mut f2 = self.criterion(x2)?;
        while hi - lo > LOG_LAMBDA_TOL {
            if f1 < f2 {
                lo = x1;
                x1 = x2;
                f1 = f2;
                x2 = lo + inv_phi * (hi - lo);
                f2 = self.criterion(x2)?;
            } else {
                hi = x2;
                x2 = x1;
                f2 = f1;
                x1 = hi - inv_phi * (hi - lo);
                f1 = self.criterion(x1)?;
            }
        }
        self.fit_at(0.5 * (lo + hi), None)
    }

    /// `a'(X'WX)b` through the retained upper band (exact, O(p·bandwidth)).
    fn gram_quadratic(&self, a: &[f64], b: &[f64]) -> f64 {
        let stride = self.band_half + 1;
        let mut q = 0.0;
        for g in 0..self.p {
            let dmax = self.band_half.min(self.p - 1 - g);
            q += self.gram_band[g * stride] * a[g] * b[g];
            for d in 1..=dmax {
                q += self.gram_band[g * stride + d] * (a[g] * b[g + d] + a[g + d] * b[g]);
            }
        }
        q
    }

    /// Posterior summary of a fit FROM THIS DESIGN, in the exact algebra of
    /// the solved system (no approximation):
    /// - `unit_covariance = (X'WX + λS)⁻¹` (scale-free Bayesian posterior
    ///   covariance of the row-major coefficient vec, shared by dimensions);
    /// - `edf = tr[(X'WX + λS)⁻¹ X'WX]` (the smoother's effective degrees of
    ///   freedom at the fitted λ);
    /// - `residual_cross_cov[d,e] = r_d'W r_e / (n − edf)` assembled from the
    ///   streamed sufficient statistics
    ///   (`y_d'Wy_e − c_d'X'Wy_e − c_e'X'Wy_d + c_d'X'WX c_e`).
    pub fn posterior(&self, fit: &GridSpline2dFit) -> Result<GridSpline2dPosterior, String> {
        let p = self.p;
        let n_dims = self.rhs.len();
        if fit.coeffs.len() != n_dims || fit.coeffs.iter().any(|c| c.len() != p) {
            return Err(format!(
                "grid spline 2d: posterior asked for a fit with {} dimensions of length {}, \
                 design has {n_dims} of {p}",
                fit.coeffs.len(),
                fit.coeffs.first().map_or(0, Vec::len)
            ));
        }
        // H⁻¹ column by column through the retained factor (symmetric, O(p³)).
        let mut unit_covariance = vec![0.0_f64; p * p];
        let mut e_g = vec![0.0_f64; p];
        for g in 0..p {
            e_g[g] = 1.0;
            let col = chol_solve(&fit.chol, p, &e_g);
            e_g[g] = 0.0;
            for (r, &v) in col.iter().enumerate() {
                unit_covariance[r * p + g] = v;
            }
        }
        // edf = tr(H⁻¹ X'WX) via the gram band (diagonal once, off-band twice).
        let stride = self.band_half + 1;
        let mut edf = 0.0;
        for g in 0..p {
            let dmax = self.band_half.min(p - 1 - g);
            edf += self.gram_band[g * stride] * unit_covariance[g * p + g];
            for d in 1..=dmax {
                edf += 2.0 * self.gram_band[g * stride + d] * unit_covariance[g * p + g + d];
            }
        }
        let residual_df = self.n_obs as f64 - edf;
        if !(residual_df >= 1.0) {
            return Err(format!(
                "grid spline 2d: too few rows for a scale estimate \
                 (n = {}, edf = {edf:.2}; need n − edf ≥ 1)",
                self.n_obs
            ));
        }
        let mut residual_cross_cov = vec![0.0_f64; n_dims * n_dims];
        for d in 0..n_dims {
            for e in d..n_dims {
                let mut cd_rhse = 0.0;
                let mut ce_rhsd = 0.0;
                for g in 0..p {
                    cd_rhse += fit.coeffs[d][g] * self.rhs[e][g];
                    ce_rhsd += fit.coeffs[e][g] * self.rhs[d][g];
                }
                let quad = self.gram_quadratic(&fit.coeffs[d], &fit.coeffs[e]);
                let v =
                    (self.cross_moments[d * n_dims + e] - cd_rhse - ce_rhsd + quad) / residual_df;
                residual_cross_cov[d * n_dims + e] = v;
                residual_cross_cov[e * n_dims + d] = v;
            }
        }
        Ok(GridSpline2dPosterior {
            unit_covariance,
            edf,
            residual_df,
            residual_cross_cov,
        })
    }
}

/// Exact posterior summary of a [`GridSpline2dFit`] (see
/// [`GridSpline2dDesign::posterior`]): the bridge from the streaming engine
/// to covariance-consuming clients (the ANOVA pair-component carve).
pub struct GridSpline2dPosterior {
    /// `(X'WX + λS)⁻¹`, `p × p` row-major — scale-free posterior covariance
    /// of the row-major coefficient vec, shared by all response dimensions.
    pub unit_covariance: Vec<f64>,
    /// `tr[(X'WX + λS)⁻¹ X'WX]`.
    pub edf: f64,
    /// `n − edf`.
    pub residual_df: f64,
    /// `D × D` row-major residual cross-covariance at `n − edf`.
    pub residual_cross_cov: Vec<f64>,
}

/// Fitted penalized tensor-product smoother with its factored covariance.
pub struct GridSpline2dFit {
    /// Per response dimension: coefficients in row-major flat order
    /// `g = j1·(K+3) + j2`.
    pub coeffs: Vec<Vec<f64>>,
    /// Selected (or supplied) log smoothing parameter, shared by all
    /// response dimensions.
    pub log_lambda: f64,
    /// Per response dimension: profiled (or supplied) observation variance σ².
    pub sigma2: Vec<f64>,
    /// Pooled restricted log-likelihood at the optimum, up to λ- and
    /// data-independent additive constants (exact REML differences across λ).
    pub restricted_loglik: f64,
    /// Lower Cholesky factor of `X'WX + λS` — the factored posterior precision
    /// (unit-σ² scale) used for prediction variances, shared by all dimensions.
    chol: Vec<f64>,
    axes: [Axis; 2],
    m_axis: usize,
}

/// Serializable snapshot of a [`GridSpline2dFit`] (#1031 persistence
/// prerequisite). The grid is deliberately NOT a formula fast path — it is an
/// ANOVA pair component (#975 carve) — so there is no `FitResult` variant; this
/// state is what the carve's persistence payload serializes and what
/// `from_state` replays for an exact predict.
///
/// Predict needs the MEAN (`coeffs` + the 16-entry tensor basis row, which is a
/// pure function of `axes`/`m_axis`) and the VARIANCE
/// (`σ²·x'(X'WX+λS)⁻¹x` through the retained Cholesky factor `chol`). All of
/// that — and nothing about the training rows — lives on the fit already, so the
/// state is a verbatim snapshot: no design CSR, no re-factor on load.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct GridSpline2dState {
    /// Per response dimension: row-major coefficients `g = j1·(K+3) + j2`.
    pub coeffs: Vec<Vec<f64>>,
    pub log_lambda: f64,
    /// Per response dimension: profiled (or supplied) observation variance σ².
    pub sigma2: Vec<f64>,
    pub restricted_loglik: f64,
    /// Lower Cholesky factor of `X'WX + λS` (unit-σ² scale), `p × p` row-major —
    /// the factored posterior precision the variance term solves against.
    pub chol: Vec<f64>,
    /// Per axis lower corner of the basis bounding box.
    pub axis_lo: [f64; 2],
    /// Per axis cell width `h = (hi − lo)/K`.
    pub axis_h: [f64; 2],
    /// Per axis cell count `K`.
    pub axis_cells: [u64; 2],
    /// Basis count per axis, `K + 3` (so `p = m_axis²`).
    pub m_axis: u64,
}

impl GridSpline2dFit {
    /// Snapshot the fit for persistence (#1031). Verbatim — every field
    /// `predict` reads is copied; the training design is not retained on the fit
    /// and is not needed for replay.
    pub fn to_state(&self) -> GridSpline2dState {
        GridSpline2dState {
            coeffs: self.coeffs.clone(),
            log_lambda: self.log_lambda,
            sigma2: self.sigma2.clone(),
            restricted_loglik: self.restricted_loglik,
            chol: self.chol.clone(),
            axis_lo: [self.axes[0].lo, self.axes[1].lo],
            axis_h: [self.axes[0].h, self.axes[1].h],
            axis_cells: [self.axes[0].cells as u64, self.axes[1].cells as u64],
            m_axis: self.m_axis as u64,
        }
    }

    /// Rebuild a predict-capable fit from a snapshot (#1031). Validates shape,
    /// finiteness, positive cell widths/counts, positive σ², and that the basis
    /// arithmetic is self-consistent (`m_axis = K + 3`, `chol` is `p × p`,
    /// `coeffs`/`sigma2` agree on `D`), so a corrupt payload fails here rather
    /// than inside a later `predict`. The restored fit replays the posterior
    /// mean+variance bit-for-bit: `predict` reads only the snapshotted fields.
    pub fn from_state(state: &GridSpline2dState) -> Result<Self, String> {
        let m_axis = state.m_axis as usize;
        let p = m_axis * m_axis;
        for a in 0..2 {
            let cells = state.axis_cells[a] as usize;
            if cells == 0 {
                return Err(format!(
                    "grid spline 2d state: axis {a} must have at least one cell"
                ));
            }
            if m_axis != cells + 3 {
                return Err(format!(
                    "grid spline 2d state: m_axis {m_axis} must equal K+3 = {} for axis {a}",
                    cells + 3
                ));
            }
            if !(state.axis_lo[a].is_finite()
                && state.axis_h[a].is_finite()
                && state.axis_h[a] > 0.0)
            {
                return Err(format!(
                    "grid spline 2d state: axis {a} must have finite lo and positive h, got lo={}, h={}",
                    state.axis_lo[a], state.axis_h[a]
                ));
            }
        }
        if state.chol.len() != p * p {
            return Err(format!(
                "grid spline 2d state: chol must be p×p = {p}² = {}, got {}",
                p * p,
                state.chol.len()
            ));
        }
        let d = state.coeffs.len();
        if d == 0 || state.sigma2.len() != d {
            return Err(format!(
                "grid spline 2d state: need ≥1 response dimension with matching σ² (coeffs D={d}, sigma2 D={})",
                state.sigma2.len()
            ));
        }
        for (dim, c) in state.coeffs.iter().enumerate() {
            if c.len() != p {
                return Err(format!(
                    "grid spline 2d state: response dimension {dim} has {} coeffs, expected p = {p}",
                    c.len()
                ));
            }
        }
        for (dim, &s2) in state.sigma2.iter().enumerate() {
            if !(s2.is_finite() && s2 > 0.0) {
                return Err(format!(
                    "grid spline 2d state: response dimension {dim} has non-positive σ² = {s2}"
                ));
            }
        }
        for (i, v) in state
            .chol
            .iter()
            .chain(state.coeffs.iter().flatten())
            .enumerate()
        {
            if !v.is_finite() {
                return Err(format!("grid spline 2d state: non-finite entry at {i}"));
            }
        }
        // The diagonal of a lower Cholesky factor is strictly positive; a
        // zero/negative pivot means the persisted factor is not a valid
        // precision factor and `chol_solve` would divide by it.
        for g in 0..p {
            let piv = state.chol[g * p + g];
            if !(piv.is_finite() && piv > 0.0) {
                return Err(format!(
                    "grid spline 2d state: non-positive Cholesky pivot {piv} at index {g}"
                ));
            }
        }
        if !(state.log_lambda.is_finite() && state.restricted_loglik.is_finite()) {
            return Err(format!(
                "grid spline 2d state: invalid scalars (log_lambda={}, restricted_loglik={})",
                state.log_lambda, state.restricted_loglik
            ));
        }
        let axes = [
            Axis {
                lo: state.axis_lo[0],
                h: state.axis_h[0],
                cells: state.axis_cells[0] as usize,
            },
            Axis {
                lo: state.axis_lo[1],
                h: state.axis_h[1],
                cells: state.axis_cells[1] as usize,
            },
        ];
        Ok(GridSpline2dFit {
            coeffs: state.coeffs.clone(),
            log_lambda: state.log_lambda,
            sigma2: state.sigma2.clone(),
            restricted_loglik: state.restricted_loglik,
            chol: state.chol.clone(),
            axes,
            m_axis,
        })
    }

    /// Posterior `(mean, variance)` of response dimension `dim` at an
    /// arbitrary point: the 16-entry basis row dotted with the coefficients,
    /// and `σ̂²_dim·x'(X'WX+λS)⁻¹x` through the retained Cholesky factor.
    /// Outside the bounding box the boundary cell's cubic polynomial extends.
    pub fn predict(&self, dim: usize, x1: f64, x2: f64) -> Result<(f64, f64), String> {
        if dim >= self.coeffs.len() {
            return Err(format!(
                "grid spline 2d: response dimension {dim} out of range (D = {})",
                self.coeffs.len()
            ));
        }
        if !(x1.is_finite() && x2.is_finite()) {
            return Err(format!(
                "grid spline 2d: non-finite prediction point ({x1}, {x2})"
            ));
        }
        let (idx, val) = basis_row(&self.axes, self.m_axis, x1, x2);
        let p = self.coeffs[dim].len();
        let mut mean = 0.0;
        let mut row = vec![0.0_f64; p];
        for e in 0..16 {
            mean += val[e] * self.coeffs[dim][idx[e]];
            row[idx[e]] += val[e];
        }
        let z = chol_solve(&self.chol, p, &row);
        let mut quad = 0.0;
        for g in 0..p {
            quad += row[g] * z[g];
        }
        Ok((mean, self.sigma2[dim] * quad))
    }
}

/// Build the streaming design and fit with REML-selected λ.
pub fn fit_grid_spline_2d(
    x1: &[f64],
    x2: &[f64],
    y: &[f64],
    w: &[f64],
    k: usize,
    metric: [f64; 2],
) -> Result<GridSpline2dFit, String> {
    GridSpline2dDesign::build(x1, x2, y, w, k, metric)?.fit_reml()
}

/// Build the streaming design and fit at a FIXED `log λ` (σ² supplied or profiled).
pub fn fit_grid_spline_2d_at(
    x1: &[f64],
    x2: &[f64],
    y: &[f64],
    w: &[f64],
    k: usize,
    metric: [f64; 2],
    log_lambda: f64,
    sigma2: Option<f64>,
) -> Result<GridSpline2dFit, String> {
    GridSpline2dDesign::build(x1, x2, y, w, k, metric)?.fit_at(log_lambda, sigma2)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// State → JSON → from_state replays the posterior mean+variance bit-for-bit
    /// at held-out points (the grid carries no training CSR, so the snapshot is
    /// the whole predict-capable object). This is the #1031 persistence
    /// prerequisite the ANOVA carve consumes.
    #[test]
    fn grid_spline_2d_state_roundtrip_reproduces_predict() {
        let k = 8usize;
        // A smooth multi-output surface on a scattered grid of points.
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        let mut y0 = Vec::new();
        let mut y1 = Vec::new();
        for i in 0..24 {
            for j in 0..24 {
                let a = i as f64 / 23.0;
                let b = j as f64 / 23.0;
                x1.push(a);
                x2.push(b);
                y0.push((2.5 * a).sin() * (1.7 * b).cos() + 0.3 * a * b);
                y1.push(a * a - 0.5 * b + 0.2 * (3.0 * a * b).cos());
            }
        }
        let n = x1.len();
        let w = vec![1.0_f64; n];
        let ys: Vec<&[f64]> = vec![&y0, &y1];
        let fit = GridSpline2dDesign::build_multi(&x1, &x2, &ys, &w, k, [1.0, 1.0])
            .expect("design")
            .fit_reml()
            .expect("fit");

        let json = serde_json::to_string(&fit.to_state()).expect("serialize");
        let state: GridSpline2dState = serde_json::from_str(&json).expect("deserialize");
        let restored = GridSpline2dFit::from_state(&state).expect("restore");

        // Held-out points, including one outside the box to exercise the
        // boundary-cell polynomial extension.
        let probes = [
            (0.13, 0.77),
            (0.41, 0.05),
            (0.66, 0.92),
            (0.99, 0.31),
            (1.20, -0.10),
        ];
        for dim in 0..2 {
            for &(p1, p2) in &probes {
                let (m0, v0) = fit.predict(dim, p1, p2).expect("orig predict");
                let (m1, v1) = restored.predict(dim, p1, p2).expect("restored predict");
                assert!(
                    (m0 - m1).abs() <= 1e-12 * (1.0 + m0.abs()),
                    "mean drift dim={dim} at ({p1},{p2}): {m0} vs {m1}"
                );
                assert!(
                    (v0 - v1).abs() <= 1e-12 * (1.0 + v0.abs()),
                    "variance drift dim={dim} at ({p1},{p2}): {v0} vs {v1}"
                );
            }
        }
        assert!((fit.log_lambda - restored.log_lambda).abs() <= 0.0);
        assert!((fit.restricted_loglik - restored.restricted_loglik).abs() <= 0.0);
    }

    /// Corrupt snapshots fail loudly in `from_state`, not inside a later predict.
    #[test]
    fn grid_spline_2d_state_rejects_corruption() {
        let k = 6usize;
        // A dense grid with n > p = (k+3)² so the fit is well-posed: this test
        // exercises `from_state` corruption rejection, not the small-n regime,
        // so the fit must succeed first (n=18 ≪ p=81 left the penalized design
        // rank-deficient and `fit_grid_spline_2d` refused before any assertion).
        let side = 12usize;
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for i in 0..side {
            for j in 0..side {
                x1.push(i as f64 / (side - 1) as f64);
                x2.push(j as f64 / (side - 1) as f64);
            }
        }
        let n = x1.len();
        let y: Vec<f64> = x1.iter().zip(&x2).map(|(&a, &b)| a + b).collect();
        let w = vec![1.0_f64; n];
        let fit = fit_grid_spline_2d(&x1, &x2, &y, &w, k, [1.0, 1.0]).expect("fit");

        let good = fit.to_state();
        let mut bad = good.clone();
        bad.chol.pop();
        assert!(
            GridSpline2dFit::from_state(&bad).is_err(),
            "chol length mismatch must error"
        );

        let mut bad = good.clone();
        bad.sigma2[0] = -1.0;
        assert!(
            GridSpline2dFit::from_state(&bad).is_err(),
            "non-positive σ² must error"
        );

        let mut bad = good.clone();
        bad.m_axis += 1;
        assert!(
            GridSpline2dFit::from_state(&bad).is_err(),
            "m_axis ≠ K+3 must error"
        );

        let mut bad = good.clone();
        bad.axis_h[0] = 0.0;
        assert!(
            GridSpline2dFit::from_state(&bad).is_err(),
            "non-positive cell width must error"
        );

        let mut bad = good;
        bad.chol[0] = 0.0;
        assert!(
            GridSpline2dFit::from_state(&bad).is_err(),
            "zero Cholesky pivot must error"
        );
    }
}
