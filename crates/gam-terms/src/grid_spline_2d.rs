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
//! Solve and selection. A single reference factorization of `H₀=X'WX+S`
//! produces the affine generalized-eigenvalue pencil
//! `H(λ)=L[I+(λ-1)U diag(μ)U']L'`. The profiled score and its first two
//! analytic log-λ derivatives are then O(pD), and outward interval enclosures
//! isolate every stationary interval without a grid. The selected system is
//! factored once more for coefficients/posterior covariance. `p ≤ (32+3)² =
//! 1225`; K is capped at 32 to keep the dense reference factor and eigensystem
//! sizing contract honest. λ maximizes the
//! profiled-σ² restricted (REML) criterion
//!   `ℓ_R(λ) = −½[ log|X'WX+λS| − r·log λ + (n−3)·log σ̂²(λ) ] + const`,
//! where `r = p−3` is the penalty rank — the null space of `J` is
//! span{1, x1, x2} (the mixed term penalizes `x1·x2`, whose cross derivative
//! is 1 ≠ 0, so it is NOT in the null space), `σ̂²(λ) = (y'Wy − c'X'Wy)/(n−3)`
//! is the profiled scale, and the λ-free additive constants (`log|S|₊` on the
//! row space of S, `Σ log w`, 2π factors) are dropped: differences across λ
//! are exact REML criterion differences. The exact bounded-domain endpoints
//! (including the null-recovery end) compete with every certified stationary
//! point; no RNG or lattice is involved, so the same data imply the same fit.
//!
//! Prediction. `predict(x1, x2)` builds the 16-entry basis row; the mean is
//! its dot with `c` and the variance is the Bayesian posterior
//! `σ̂²·x'(X'WX+λS)⁻¹x` through the retained Cholesky factor. Outside the
//! bounding box the boundary cell's cubic polynomial extends naturally (the
//! cell index clamps, the local coordinate does not).

use faer::{Mat, Side};
use gam_math::score_opt::{
    AffineRemlProfile, ClosedInterval, ScoreOptimumLocation, certified_ln_positive,
};

/// Dimension of the penalty null space: span{1, x1, x2}. The mixed
/// `2·a1·a2·f_{x1x2}²` term excludes `x1·x2` (its cross derivative is 1).
const PENALTY_NULLITY: usize = 3;

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

/// One uniform B-spline axis over `[lo, lo + cells·h]`.
#[derive(Clone, Copy, Debug)]
struct Axis {
    lo: f64,
    h: f64,
    cells: usize,
}

impl Axis {
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

/// Solve `L z = b` from a dense row-major lower-triangular factor.
fn lower_solve(l: &[f64], p: usize, b: &[f64]) -> Vec<f64> {
    let mut z = b.to_vec();
    for i in 0..p {
        let mut s = z[i];
        for t in 0..i {
            s -= l[i * p + t] * z[t];
        }
        z[i] = s / l[i * p + i];
    }
    z
}

/// Solve `L Lᵀ x = b` from the stored lower factor.
pub fn chol_solve(l: &[f64], p: usize, b: &[f64]) -> Vec<f64> {
    let mut z = lower_solve(l, p, b);
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

/// Owned spectral data for the shared affine REML profile.  Keeping the
/// eigensystem reduction separate from the search makes every score evaluation
/// O(pD) and ensures the final dense system is factored only at the selected λ.
struct RemlSpectrum {
    gram_modes: Vec<f64>,
    penalty_modes: Vec<f64>,
    projected_rhs_squared: Vec<f64>,
    response_energy: Vec<f64>,
    residual_dof: f64,
    logdet_constant: f64,
}

impl RemlSpectrum {
    fn profile(&self) -> Result<AffineRemlProfile<'_>, String> {
        AffineRemlProfile::new(
            &self.gram_modes,
            &self.penalty_modes,
            &self.projected_rhs_squared,
            &self.response_energy,
            self.residual_dof,
            self.penalty_modes.len() - PENALTY_NULLITY,
            self.logdet_constant,
        )
        .map_err(|error| format!("grid spline 2d: invalid REML spectrum: {error}"))
    }

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

    /// Number of data rows the design was streamed from.
    pub fn num_rows(&self) -> usize {
        self.n_obs
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn affine_reml_profile_matches_direct_factorizations() {
        let side = 10usize;
        let mut x1 = Vec::with_capacity(side * side);
        let mut x2 = Vec::with_capacity(side * side);
        let mut y0 = Vec::with_capacity(side * side);
        let mut y1 = Vec::with_capacity(side * side);
        for i in 0..side {
            for j in 0..side {
                let a = i as f64 / (side - 1) as f64;
                let b = j as f64 / (side - 1) as f64;
                x1.push(a);
                x2.push(b);
                y0.push((2.0 * a).sin() * (3.0 * b).cos() + a * b);
                y1.push(a * a - b * b + (a + 2.0 * b).sin());
            }
        }
        let weights = vec![1.0; x1.len()];
        let responses: [&[f64]; 2] = [&y0, &y1];
        let design = GridSpline2dDesign::build_multi(&x1, &x2, &responses, &weights, 3, [1.0, 1.5])
            .expect("design");
        let spectrum = design.reml_spectrum().expect("reference pencil");
        let profile = spectrum.profile().expect("affine profile");
        let dof = (design.n_obs - PENALTY_NULLITY) as f64;
        let rank = (design.p - PENALTY_NULLITY) as f64;

        for log_lambda in [-5.0, 0.0, 6.0] {
            let solved = design.solve_at(log_lambda).expect("direct solve");
            let shared = solved.logdet - rank * log_lambda;
            let direct = -0.5
                * solved
                    .rss_pen
                    .iter()
                    .map(|rss| shared + dof * (rss / dof).ln())
                    .sum::<f64>();
            let spectral = profile.evaluate(log_lambda).expect("spectral score").value;
            assert!(
                (direct - spectral).abs() <= f64::EPSILON.sqrt() * (1.0 + direct.abs()),
                "score mismatch at log lambda {log_lambda}: direct={direct}, spectral={spectral}"
            );
        }
    }

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
        // The response must carry genuine curvature: a purely affine `a + b`
        // lies entirely in the penalty NULL SPACE (the spline reproduces it
        // exactly at any λ), so the penalized residual is identically zero and
        // `fit_grid_spline_2d` correctly refuses with "degenerate penalized
        // residual 0" — there is no variance to estimate. Add a smooth
        // non-null-space (curved) component so the penalized fit leaves a
        // positive residual and the REML criterion is well-posed; this test is
        // about `from_state` corruption rejection, which needs a successful fit
        // first.
        let y: Vec<f64> = x1
            .iter()
            .zip(&x2)
            .map(|(&a, &b)| a + b + (3.0 * a).sin() * (2.5 * b).cos())
            .collect();
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
