//! Functorial inter-layer concept transport maps (issue #1013).
//!
//! For an atom whose layer-`l` chart assigns coordinates `t_l` to each row and
//! whose continuation at layer `l+1` assigns `t_{l+1}`, the estimand is the
//! smooth transport map
//!
//! ```text
//!     t_{l+1} = h_{l→l+1}(t_l)
//! ```
//!
//! fitted as a small penalized GAM with the engine's Gaussian REML machinery
//! (exact 1-D criterion, no GCV per policy). Three questions are answered with
//! evidence:
//!
//! 1. **Topology compatibility** — does `h` preserve the chart topology
//!    (circle→circle degree-±1 covering, i.e. a homeomorphism of `S¹`) or
//!    break it (circle→arcs, folds)? For circle charts the winding **degree**
//!    is estimated by maximizing the circular concentration (mean resultant
//!    length) of the de-wound residual `θ_to − d·θ_from` over candidate
//!    degrees `d ∈ {−2,−1,0,1,2}` — for a transport whose smooth residual
//!    stays inside half a turn this is the circular-correlation-maximizing
//!    degree, and it is exact in the noiseless limit. A fold check on a dense
//!    grid (`sign(d)·h′(t) > 0` everywhere) separates genuine degree-±1
//!    covers from degree-±1 maps with local back-tracking.
//! 2. **Isometry defect** — `∫ (|h′| − 1)² dP̂` under the empirical data
//!    density `P̂` (the integral is evaluated at the observed coordinates, so
//!    dense regions of the chart dominate, as the issue requires). A
//!    delta-method standard error is propagated from the coefficient
//!    covariance. Near-zero defect ⇒ TRANSPORT layer (the concept is carried
//!    isometrically); large defect ⇒ COMPUTE layer (the chart metric is
//!    reshaped).
//! 3. **Composition law** — `h_{l→l+2}` vs `h_{l+1→l+2} ∘ h_{l→l+1}`. The
//!    defect `d(t) = h_ac(t) ⊖ h_bc(h_ab(t))` (circular difference on circle
//!    charts) is evaluated on a grid, studentized by the composed
//!    delta-method bands, and tested with the existing
//!    [`wood_smooth_test`](crate::inference::smooth_test::wood_smooth_test)
//!    machinery applied to a REML smooth of the defect.
//!
//! # Gauge discipline
//!
//! Each chart coordinate is identified only up to the residual isometry gauge
//! of its chart, so a transport map is identified only up to the **double
//! coset** `[Isom(M_to)] · h · [Isom(M_from)]`. Two facts are used:
//!
//! * All three routes in a composition test consume the *same* source
//!   coordinates, so any isometry of the source chart acts identically on
//!   `h_ac` and on `h_bc ∘ h_ab`; the source gauge cancels in the defect and
//!   needs no explicit alignment.
//! * The target gauge does not cancel: before testing, the composed route is
//!   aligned to the direct route using ONLY the certified finite/1-parameter
//!   isometries of the target chart — for a circle, the rotation (fixed at
//!   the circular mean of the defect) and the reflection (the orientation
//!   with the smaller squared defect); for an interval, the reflection about
//!   its midpoint. No general reparameterization is ever fitted away.
//!
//! All smooths reuse the engine's existing periodic cardinal-B-spline basis
//! ([`build_periodic_bspline_basis_1d`]) with the cyclic difference penalty on
//! circular domains, and the open B-spline basis with the standard difference
//! penalty on interval domains — constructed directly, not via the string DSL.

use crate::faer_ndarray::FaerEigh;
use crate::inference::smooth_test::{SmoothTestInput, SmoothTestScale, wood_smooth_test};
use crate::terms::basis::{
    BasisOptions, Dense, KnotSource, PeriodicBSplineBasisSpec, build_periodic_bspline_basis_1d,
    create_basis, create_cyclic_difference_penalty_matrix, create_difference_penalty_matrix,
    periodic_bspline_first_derivative_nd,
};
use crate::terms::sae::chart_canonicalization::CanonicalChartTopology;
use faer::Side;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use statrs::distribution::{ContinuousCDF, Normal};
use std::f64::consts::{PI, TAU};

/// Cubic splines for every transport smooth.
const TRANSPORT_SPLINE_DEGREE: usize = 3;
/// Second-order (curvature) difference penalty: the cyclic variant leaves
/// constants unpenalized on a circle; the open variant leaves affine maps
/// unpenalized on an interval — exactly the isometry-adjacent null spaces.
const TRANSPORT_PENALTY_ORDER: usize = 2;
/// Minimum paired observations for a transport fit.
const MIN_TRANSPORT_OBS: usize = 16;
/// Target observations per basis function when auto-sizing the basis.
const OBS_PER_BASIS: usize = 8;
/// Periodic basis size bounds (auto-derived from `n`, never a caller knob).
const MIN_PERIODIC_BASIS: usize = 8;
const MAX_PERIODIC_BASIS: usize = 20;
/// Open-interval internal-knot bounds.
const MIN_OPEN_INTERNAL_KNOTS: usize = 4;
const MAX_OPEN_INTERNAL_KNOTS: usize = 12;
/// Candidate winding degrees scanned by the circular-concentration estimator.
const DEGREE_CANDIDATES: [i32; 5] = [-2, -1, 0, 1, 2];
/// Dense grid used for the fold / orientation check of `h′`.
const FOLD_CHECK_GRID: usize = 512;
/// Default evaluation grid for the composition-law defect.
pub const DEFAULT_COMPOSITION_GRID: usize = 256;
/// REML λ-profile: log-spaced grid points then golden-section refinement.
const REML_LAMBDA_GRID_POINTS: usize = 41;
const REML_GOLDEN_ITERATIONS: usize = 40;
const REML_LAMBDA_SPAN_DECADES: f64 = 8.0;

/// Topology of a one-dimensional concept chart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChartTopology {
    /// Circular chart; coordinates are angles in radians, identified mod 2π.
    Circle,
    /// Interval chart with the Euclidean metric on `[lo, hi]`.
    Interval { lo: f64, hi: f64 },
}

impl ChartTopology {
    /// Short stable name used by FFI payloads.
    pub fn name(&self) -> &'static str {
        match self {
            ChartTopology::Circle => "circle",
            ChartTopology::Interval { .. } => "interval",
        }
    }

    fn validate(&self) -> Result<(), String> {
        match *self {
            ChartTopology::Circle => Ok(()),
            ChartTopology::Interval { lo, hi } => {
                if !(lo.is_finite() && hi.is_finite()) || hi <= lo {
                    Err(format!(
                        "interval chart bounds must be finite and ordered; got [{lo}, {hi}]"
                    ))
                } else {
                    Ok(())
                }
            }
        }
    }
}

/// Bridge from the SAE canonicalization topology to the transport topology.
///
/// `CanonicalChartTopology::Circle { period }` becomes a `Circle` chart whose
/// coordinates are interpreted on `[0, period)` — the transport module's period
/// is fixed to `TAU` (angles in radians), so the conversion rescales by mapping
/// the period-normalized angle `t / period * TAU` at the call site. The caller
/// must apply this rescaling before handing coordinates to `fit_transport_map`.
///
/// `CanonicalChartTopology::Interval` becomes `Interval { lo: 0.0, hi: 1.0 }`
/// (the canonical unit-speed interval span set by the canonicalization step).
impl From<&CanonicalChartTopology> for ChartTopology {
    fn from(src: &CanonicalChartTopology) -> Self {
        match src {
            CanonicalChartTopology::Circle { .. } => ChartTopology::Circle,
            CanonicalChartTopology::Interval => ChartTopology::Interval { lo: 0.0, hi: 1.0 },
        }
    }
}

impl From<CanonicalChartTopology> for ChartTopology {
    fn from(src: CanonicalChartTopology) -> Self {
        ChartTopology::from(&src)
    }
}

/// Wrap an angle into `[0, 2π)`.
fn wrap_tau(x: f64) -> f64 {
    x.rem_euclid(TAU)
}

/// Wrap an angle into `(−π, π]`.
fn wrap_pi(x: f64) -> f64 {
    let w = (x + PI).rem_euclid(TAU) - PI;
    if w <= -PI { w + TAU } else { w }
}

/// Circular mean of a set of angles; `0` when the resultant degenerates.
fn circular_mean(angles: &[f64]) -> f64 {
    let mut s = 0.0_f64;
    let mut c = 0.0_f64;
    for &a in angles {
        s += a.sin();
        c += a.cos();
    }
    if s.hypot(c) <= f64::EPSILON * angles.len().max(1) as f64 {
        0.0
    } else {
        s.atan2(c)
    }
}

/// Mean resultant length `R ∈ [0, 1]` of a set of angles.
fn resultant_length(angles: &[f64]) -> f64 {
    if angles.is_empty() {
        return 0.0;
    }
    let mut s = 0.0_f64;
    let mut c = 0.0_f64;
    for &a in angles {
        s += a.sin();
        c += a.cos();
    }
    s.hypot(c) / angles.len() as f64
}

/// Domain-side basis carrier: periodic cardinal B-splines on a circle, open
/// B-splines on an interval. Both reuse the existing basis constructors
/// directly (no string DSL round-trip).
#[derive(Debug, Clone)]
enum DomainBasis {
    Periodic(PeriodicBSplineBasisSpec),
    Open { knots: Array1<f64>, degree: usize },
}

impl DomainBasis {
    fn build(topology: ChartTopology, coords: ArrayView1<'_, f64>) -> Result<Self, String> {
        let n = coords.len();
        match topology {
            ChartTopology::Circle => {
                let num_basis = (n / OBS_PER_BASIS).clamp(MIN_PERIODIC_BASIS, MAX_PERIODIC_BASIS);
                Ok(DomainBasis::Periodic(PeriodicBSplineBasisSpec {
                    degree: TRANSPORT_SPLINE_DEGREE,
                    num_basis,
                    period: TAU,
                    origin: 0.0,
                    penalty_order: TRANSPORT_PENALTY_ORDER,
                }))
            }
            ChartTopology::Interval { lo, hi } => {
                let num_internal =
                    (n / OBS_PER_BASIS).clamp(MIN_OPEN_INTERNAL_KNOTS, MAX_OPEN_INTERNAL_KNOTS);
                let (seed, knots) = create_basis::<Dense>(
                    coords.mapv(|v| v.clamp(lo, hi)).view(),
                    KnotSource::Generate {
                        data_range: (lo, hi),
                        num_internal_knots: num_internal,
                    },
                    TRANSPORT_SPLINE_DEGREE,
                    BasisOptions::value(),
                )
                .map_err(|e| format!("layer transport open basis construction failed: {e}"))?;
                if seed.nrows() != n {
                    return Err(format!(
                        "layer transport open basis returned {} rows for {n} inputs",
                        seed.nrows()
                    ));
                }
                Ok(DomainBasis::Open {
                    knots,
                    degree: TRANSPORT_SPLINE_DEGREE,
                })
            }
        }
    }

    fn num_basis(&self) -> usize {
        match self {
            DomainBasis::Periodic(spec) => spec.num_basis,
            DomainBasis::Open { knots, degree } => knots.len() - degree - 1,
        }
    }

    /// Rank of the smoothing penalty: the cyclic 2nd-difference penalty
    /// annihilates only constants (a linear map is not periodic), the open
    /// 2nd-difference penalty annihilates affine maps.
    fn penalty_rank(&self) -> usize {
        match self {
            DomainBasis::Periodic(spec) => spec.num_basis - 1,
            DomainBasis::Open { .. } => self.num_basis() - TRANSPORT_PENALTY_ORDER,
        }
    }

    fn penalty(&self) -> Result<Array2<f64>, String> {
        match self {
            DomainBasis::Periodic(spec) => {
                create_cyclic_difference_penalty_matrix(spec.num_basis, TRANSPORT_PENALTY_ORDER)
                    .map_err(|e| format!("cyclic transport penalty failed: {e}"))
            }
            DomainBasis::Open { .. } => {
                create_difference_penalty_matrix(self.num_basis(), TRANSPORT_PENALTY_ORDER, None)
                    .map_err(|e| format!("open transport penalty failed: {e}"))
            }
        }
    }

    /// Clamp/wrap an evaluation point into the basis domain.
    fn project(&self, t: f64) -> f64 {
        match self {
            DomainBasis::Periodic(_) => wrap_tau(t),
            DomainBasis::Open { knots, degree } => {
                let lo = knots[*degree];
                let hi = knots[knots.len() - 1 - degree];
                t.clamp(lo, hi)
            }
        }
    }

    fn value_rows(&self, t: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        let projected = t.mapv(|v| self.project(v));
        match self {
            DomainBasis::Periodic(spec) => build_periodic_bspline_basis_1d(projected.view(), spec)
                .map_err(|e| format!("periodic transport basis evaluation failed: {e}")),
            DomainBasis::Open { knots, degree } => {
                let (rows, used_knots) = create_basis::<Dense>(
                    projected.view(),
                    KnotSource::Provided(knots.view()),
                    *degree,
                    BasisOptions::value(),
                )
                .map_err(|e| format!("open transport basis evaluation failed: {e}"))?;
                if used_knots.len() != knots.len() {
                    return Err("open transport basis knot vector drifted".to_string());
                }
                Ok(rows.as_ref().to_owned())
            }
        }
    }

    fn derivative_rows(&self, t: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        let projected = t.mapv(|v| self.project(v));
        match self {
            DomainBasis::Periodic(spec) => {
                let n = projected.len();
                let mut col = Array2::<f64>::zeros((n, 1));
                for (i, &v) in projected.iter().enumerate() {
                    col[[i, 0]] = v;
                }
                let jet = periodic_bspline_first_derivative_nd(
                    col.view(),
                    (0.0, TAU),
                    spec.degree,
                    spec.num_basis,
                )
                .map_err(|e| format!("periodic transport derivative failed: {e}"))?;
                Ok(jet.index_axis(Axis(2), 0).to_owned())
            }
            DomainBasis::Open { knots, degree } => {
                let (rows, used_knots) = create_basis::<Dense>(
                    projected.view(),
                    KnotSource::Provided(knots.view()),
                    *degree,
                    BasisOptions::first_derivative(),
                )
                .map_err(|e| format!("open transport derivative failed: {e}"))?;
                if used_knots.len() != knots.len() {
                    return Err("open transport derivative knot vector drifted".to_string());
                }
                Ok(rows.as_ref().to_owned())
            }
        }
    }
}

/// One penalized 1-D smooth chosen by exact Gaussian REML (or known-scale
/// REML for the weighted defect fit), with everything downstream inference
/// needs: scale-included covariance, the influence block for trace-corrected
/// reference d.f., EDF, and the selected λ.
struct Penalized1dFit {
    beta: Array1<f64>,
    /// Scale-included posterior covariance `σ̂²(XᵀWX + λS)⁻¹` (φ̂ = 1 in the
    /// known-scale branch).
    covariance: Array2<f64>,
    /// Coefficient-space influence `F = (XᵀWX + λS)⁻¹ XᵀWX` for Wood's
    /// trace-corrected reference d.f.
    influence: Array2<f64>,
    lambda: f64,
    edf: f64,
    sigma2: f64,
    residual_rms: f64,
}

/// Exact 1-D Gaussian REML on a fixed design/penalty pair.
///
/// Estimated scale (`known_scale = false`): profile σ² out of Wood's REML,
/// `V(λ) = (n − M₀)·log PRSS(λ) + log|XᵀWX + λS| − rank(S)·log λ`, with
/// `M₀ = dim ker S` and `PRSS = yᵀWy − β̂ᵀXᵀWy`. Known scale (φ = 1, used for
/// the variance-weighted defect smooth): `V(λ) = PRSS + log|XᵀWX + λS| −
/// rank(S)·log λ`. λ is selected on a deterministic log grid spanning
/// ±[`REML_LAMBDA_SPAN_DECADES`] decades around the design's trace scale and
/// refined by golden section — no RNG, no caller knobs.
fn fit_penalized_1d(
    design: &Array2<f64>,
    penalty: &Array2<f64>,
    response: ArrayView1<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    penalty_rank: usize,
    known_scale: bool,
) -> Result<Penalized1dFit, String> {
    let n = design.nrows();
    let m = design.ncols();
    if response.len() != n || penalty.nrows() != m || penalty.ncols() != m {
        return Err(format!(
            "penalized 1-D fit shape mismatch: X is {n}×{m}, y has {}, S is {}×{}",
            response.len(),
            penalty.nrows(),
            penalty.ncols()
        ));
    }
    if let Some(w) = weights {
        if w.len() != n {
            return Err(format!(
                "penalized 1-D fit weight length {} does not match n = {n}",
                w.len()
            ));
        }
        if w.iter().any(|&v| !v.is_finite() || v <= 0.0) {
            return Err("penalized 1-D fit weights must be finite and positive".to_string());
        }
    }

    let mut xtwx = Array2::<f64>::zeros((m, m));
    let mut xtwy = Array1::<f64>::zeros(m);
    let mut ytwy = 0.0_f64;
    let mut sum_w = 0.0_f64;
    for r in 0..n {
        let w = weights.map_or(1.0, |wv| wv[r]);
        let y = response[r];
        ytwy += w * y * y;
        sum_w += w;
        for j in 0..m {
            let xj = design[[r, j]];
            if xj == 0.0 {
                continue;
            }
            xtwy[j] += w * xj * y;
            for k in j..m {
                xtwx[[j, k]] += w * xj * design[[r, k]];
            }
        }
    }
    for j in 0..m {
        for k in 0..j {
            xtwx[[j, k]] = xtwx[[k, j]];
        }
    }

    let trace_scale = (0..m).map(|i| xtwx[[i, i]]).sum::<f64>() / m as f64;
    let anchor = trace_scale.max(f64::MIN_POSITIVE);
    let nullspace_dim = m.saturating_sub(penalty_rank);
    let dof = ((n as f64) - nullspace_dim as f64).max(1.0);
    let rank_f = penalty_rank as f64;

    let solve_at = |lambda: f64| -> Result<(Array1<f64>, Array1<f64>, Array2<f64>), String> {
        let mut a = xtwx.clone();
        for j in 0..m {
            for k in 0..m {
                a[[j, k]] += lambda * penalty[[j, k]];
            }
        }
        // Representative-selecting micro-ridge for exactly aliased designs.
        let diag_scale = (0..m).map(|i| a[[i, i]].abs()).fold(1.0_f64, f64::max);
        for i in 0..m {
            a[[i, i]] += 1e-12 * diag_scale;
        }
        let (evals, evecs) = a
            .eigh(Side::Lower)
            .map_err(|e| format!("penalized 1-D fit eigendecomposition failed: {e:?}"))?;
        Ok((evals, evecs.t().dot(&xtwy), evecs))
    };

    let criterion = |lambda: f64| -> f64 {
        let Ok(parts) = solve_at(lambda) else {
            return f64::INFINITY;
        };
        let (evals, rotated) = (&parts.0, &parts.1);
        let floor = evals.iter().copied().fold(0.0_f64, f64::max) * 1e-14;
        let mut prss = ytwy;
        let mut logdet = 0.0_f64;
        for i in 0..m {
            let d = evals[i].max(floor).max(f64::MIN_POSITIVE);
            prss -= rotated[i] * rotated[i] / d;
            logdet += d.ln();
        }
        let prss = prss.max(f64::MIN_POSITIVE);
        let fit_term = if known_scale { prss } else { dof * prss.ln() };
        fit_term + logdet - rank_f * lambda.ln()
    };

    let lo = anchor * 10f64.powf(-REML_LAMBDA_SPAN_DECADES);
    let hi = anchor * 10f64.powf(REML_LAMBDA_SPAN_DECADES);
    let grid: Vec<f64> = (0..REML_LAMBDA_GRID_POINTS)
        .map(|i| {
            let t = i as f64 / (REML_LAMBDA_GRID_POINTS - 1) as f64;
            lo * (hi / lo).powf(t)
        })
        .collect();
    let mut best_idx = 0usize;
    let mut best_val = f64::INFINITY;
    for (i, &lam) in grid.iter().enumerate() {
        let v = criterion(lam);
        if v < best_val {
            best_val = v;
            best_idx = i;
        }
    }
    let mut a_log = grid[best_idx.saturating_sub(1)].ln();
    let mut c_log = grid[(best_idx + 1).min(REML_LAMBDA_GRID_POINTS - 1)].ln();
    let golden = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut x1 = c_log - golden * (c_log - a_log);
    let mut x2 = a_log + golden * (c_log - a_log);
    let mut f1 = criterion(x1.exp());
    let mut f2 = criterion(x2.exp());
    for _ in 0..REML_GOLDEN_ITERATIONS {
        if f1 <= f2 {
            c_log = x2;
            x2 = x1;
            f2 = f1;
            x1 = c_log - golden * (c_log - a_log);
            f1 = criterion(x1.exp());
        } else {
            a_log = x1;
            x1 = x2;
            f1 = f2;
            x2 = a_log + golden * (c_log - a_log);
            f2 = criterion(x2.exp());
        }
    }
    let lambda = (0.5 * (a_log + c_log)).exp();

    let (evals, rotated, evecs) = solve_at(lambda)?;
    let floor = evals.iter().copied().fold(0.0_f64, f64::max) * 1e-14;
    let mut a_inv = Array2::<f64>::zeros((m, m));
    let mut beta = Array1::<f64>::zeros(m);
    for i in 0..m {
        let d = evals[i].max(floor).max(f64::MIN_POSITIVE);
        let coeff = rotated[i] / d;
        for j in 0..m {
            beta[j] += evecs[[j, i]] * coeff;
            for k in 0..m {
                a_inv[[j, k]] += evecs[[j, i]] * evecs[[k, i]] / d;
            }
        }
    }
    let influence = a_inv.dot(&xtwx);
    let edf = (0..m).map(|i| influence[[i, i]]).sum::<f64>();

    let fitted = design.dot(&beta);
    let mut rss = 0.0_f64;
    for r in 0..n {
        let w = weights.map_or(1.0, |wv| wv[r]);
        let e = response[r] - fitted[r];
        rss += w * e * e;
    }
    let sigma2 = if known_scale {
        1.0
    } else {
        (rss / ((n as f64) - edf).max(1.0)).max(f64::MIN_POSITIVE)
    };
    let covariance = a_inv.mapv(|v| v * sigma2);
    let residual_rms = (rss / sum_w.max(f64::MIN_POSITIVE)).sqrt();

    if beta.iter().any(|v| !v.is_finite()) {
        return Err("penalized 1-D fit produced non-finite coefficients".to_string());
    }
    Ok(Penalized1dFit {
        beta,
        covariance,
        influence,
        lambda,
        edf,
        sigma2,
        residual_rms,
    })
}

/// A fitted inter-layer transport map with full posterior bookkeeping, ready
/// for evaluation, banding, and composition testing.
///
/// Representation: `h(t) = degree·t + rotation_offset + g(t)` on circle
/// targets (`g` the REML periodic/open spline; the result is read mod 2π) and
/// `h(t) = g(t)` on interval targets. The discrete winding `degree` and the
/// wrap-branch offset are treated as fixed (a discrete selection and a gauge
/// representative respectively); pointwise variances propagate the spline
/// coefficient covariance only.
#[derive(Debug, Clone)]
pub struct FittedTransport {
    pub topology_from: ChartTopology,
    pub topology_to: ChartTopology,
    /// Winding degree of the map (circle→circle charts only).
    pub degree: Option<i32>,
    /// Mean resultant length of the de-wound residual at the selected degree
    /// (circle→circle only): the concentration evidence behind `degree`.
    pub degree_concentration: Option<f64>,
    /// Rotation gauge representative used to pick the wrap branch of the
    /// angular response (circle targets; `0` for interval targets). The
    /// estimand is the double coset, so this constant carries no information
    /// on its own.
    pub rotation_offset: f64,
    /// Spline coefficients of the residual smooth `g`.
    pub beta: Array1<f64>,
    /// Scale-included posterior covariance of `beta` (mgcv `Vb` analogue).
    pub covariance: Array2<f64>,
    pub smoothing_lambda: f64,
    /// Effective degrees of freedom of the transport smooth.
    pub edf: f64,
    /// REML-profiled residual variance σ̂² of the (unwrapped) response.
    pub noise_variance: f64,
    pub n_obs: usize,
    /// Empirical-density-weighted isometry defect `mean((|h′(tᵢ)| − 1)²)`.
    pub isometry_defect: f64,
    /// Delta-method standard error of the isometry defect.
    pub isometry_defect_se: f64,
    /// Whether `h` is compatible with both chart topologies: a degree-±1
    /// circle cover without folds, or a fold-free interval homeomorphism.
    pub topology_preserved: bool,
    /// `min over a dense grid of orientation·h′(t)`; positive ⇔ no folds.
    pub min_directional_derivative: f64,
    /// RMS of the response residuals at the fitted map.
    pub residual_rms: f64,
    basis: DomainBasis,
}

impl FittedTransport {
    fn linear_slope(&self) -> f64 {
        self.degree.map_or(0.0, f64::from)
    }

    /// Evaluate `h` at `t` (wrapped to `[0, 2π)` on circle targets).
    pub fn eval(&self, t: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        let rows = self.basis.value_rows(t)?;
        let smooth = rows.dot(&self.beta);
        let slope = self.linear_slope();
        let mut out = Array1::<f64>::zeros(t.len());
        for i in 0..t.len() {
            let raw = slope * t[i] + self.rotation_offset + smooth[i];
            out[i] = match self.topology_to {
                ChartTopology::Circle => wrap_tau(raw),
                ChartTopology::Interval { .. } => raw,
            };
        }
        Ok(out)
    }

    /// Evaluate `h` and its pointwise delta-method variance.
    pub fn eval_with_variance(
        &self,
        t: ArrayView1<'_, f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let rows = self.basis.value_rows(t)?;
        let values = self.eval(t)?;
        let mut variances = Array1::<f64>::zeros(t.len());
        for i in 0..t.len() {
            let row = rows.row(i);
            variances[i] = row.dot(&self.covariance.dot(&row)).max(0.0);
        }
        Ok((values, variances))
    }

    /// Evaluate `h′(t)` (chart-coordinate derivative).
    pub fn derivative(&self, t: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        let rows = self.basis.derivative_rows(t)?;
        let slope = self.linear_slope();
        Ok(rows.dot(&self.beta).mapv(|v| v + slope))
    }

    /// Package the fit as a [`LayerTransportReport`] for the given layer pair
    /// (composition fields empty; see [`LayerTransportReport::with_composition`]).
    pub fn report(&self, layer_from: usize, layer_to: usize) -> LayerTransportReport {
        LayerTransportReport {
            layer_from,
            layer_to,
            topology_from: self.topology_from,
            topology_to: self.topology_to,
            topology_preserved: self.topology_preserved,
            degree: self.degree,
            degree_concentration: self.degree_concentration,
            rotation_offset: self.rotation_offset,
            isometry_defect: self.isometry_defect,
            isometry_defect_se: self.isometry_defect_se,
            min_directional_derivative: self.min_directional_derivative,
            transport_edf: self.edf,
            smoothing_lambda: self.smoothing_lambda,
            noise_variance: self.noise_variance,
            residual_rms: self.residual_rms,
            n_obs: self.n_obs,
            composition_defect: None,
            composition_max_studentized: None,
            composition_p_value: None,
            composition_gauge_reflected: None,
        }
    }
}

/// Evidence payload for one estimated inter-layer transport map.
#[derive(Debug, Clone)]
pub struct LayerTransportReport {
    pub layer_from: usize,
    pub layer_to: usize,
    pub topology_from: ChartTopology,
    pub topology_to: ChartTopology,
    /// Degree-±1 fold-free circle cover (or fold-free interval homeo).
    pub topology_preserved: bool,
    /// Estimated winding degree (circle→circle only).
    pub degree: Option<i32>,
    /// Circular concentration of the de-wound residual at `degree`.
    pub degree_concentration: Option<f64>,
    /// Rotation gauge representative (circle targets).
    pub rotation_offset: f64,
    /// `∫(|h′| − 1)² dP̂` under the empirical chart density.
    pub isometry_defect: f64,
    /// Delta-method SE of the isometry defect.
    pub isometry_defect_se: f64,
    /// Fold diagnostic: min of orientation·h′ over a dense grid.
    pub min_directional_derivative: f64,
    /// EDF of the REML transport smooth.
    pub transport_edf: f64,
    pub smoothing_lambda: f64,
    pub noise_variance: f64,
    pub residual_rms: f64,
    pub n_obs: usize,
    /// RMS composition defect of the triple ending at this two-hop map
    /// (populated by [`transport_ladder`] / [`LayerTransportReport::with_composition`]).
    pub composition_defect: Option<f64>,
    /// Max studentized composition defect against the composed bands.
    pub composition_max_studentized: Option<f64>,
    /// `wood_smooth_test` p-value of the defect smooth (H₀: defect ≡ 0 up to
    /// the target-chart gauge).
    pub composition_p_value: Option<f64>,
    /// Whether the gauge alignment chose the reflected target orientation.
    pub composition_gauge_reflected: Option<bool>,
}

impl LayerTransportReport {
    /// Merge a composition-law test into this (direct, two-hop) report.
    pub fn with_composition(mut self, composition: &CompositionDefectReport) -> Self {
        self.composition_defect = Some(composition.rms_defect);
        self.composition_max_studentized = Some(composition.max_studentized_defect);
        self.composition_p_value = Some(composition.p_value);
        self.composition_gauge_reflected = Some(composition.gauge_reflected);
        self
    }
}

/// Estimate the transport map `h: M_from → M_to` between two chart
/// coordinatizations of the same rows.
///
/// `coords_from[i]` and `coords_to[i]` must coordinatize the same observation
/// in the source and target charts. Circle coordinates are radians (any
/// branch; wrapped internally). See the module docs for the estimator.
pub fn fit_transport_map(
    coords_from: ArrayView1<'_, f64>,
    coords_to: ArrayView1<'_, f64>,
    topology_from: ChartTopology,
    topology_to: ChartTopology,
) -> Result<FittedTransport, String> {
    let n = coords_from.len();
    if coords_to.len() != n {
        return Err(format!(
            "layer transport coordinate lengths disagree: {} vs {}",
            n,
            coords_to.len()
        ));
    }
    if n < MIN_TRANSPORT_OBS {
        return Err(format!(
            "layer transport needs at least {MIN_TRANSPORT_OBS} paired observations, got {n}"
        ));
    }
    if coords_from
        .iter()
        .chain(coords_to.iter())
        .any(|v| !v.is_finite())
    {
        return Err("layer transport coordinates must all be finite".to_string());
    }
    topology_from.validate()?;
    topology_to.validate()?;

    // --- degree + rotation gauge + unwrapped response -----------------------
    let (degree, degree_concentration, rotation_offset, response): (
        Option<i32>,
        Option<f64>,
        f64,
        Array1<f64>,
    ) = match (topology_from, topology_to) {
        (ChartTopology::Circle, ChartTopology::Circle) => {
            // Winding degree by circular concentration: over candidate
            // degrees d, the de-wound residual r_i(d) = θ_to − d·θ_from is
            // tightest (largest mean resultant length R_d) at the true
            // degree whenever the smooth residual stays inside half a turn.
            // This is the circular-correlation-maximizing degree estimate
            // the issue specifies, in resultant form.
            let mut best_degree = DEGREE_CANDIDATES[0];
            let mut best_r = f64::NEG_INFINITY;
            for &d in DEGREE_CANDIDATES.iter() {
                let residual: Vec<f64> = (0..n)
                    .map(|i| coords_to[i] - f64::from(d) * coords_from[i])
                    .collect();
                let r = resultant_length(&residual);
                if r > best_r {
                    best_r = r;
                    best_degree = d;
                }
            }
            let residual: Vec<f64> = (0..n)
                .map(|i| coords_to[i] - f64::from(best_degree) * coords_from[i])
                .collect();
            let mu = circular_mean(&residual);
            let response = Array1::from_iter(residual.iter().map(|&r| wrap_pi(r - mu)));
            (Some(best_degree), Some(best_r), mu, response)
        }
        (_, ChartTopology::Circle) => {
            // Interval domain, circular target: the domain is contractible so
            // the map is null-homotopic — no winding term. Unwrap the angular
            // response about its circular mean.
            let angles: Vec<f64> = coords_to.iter().copied().collect();
            let mu = circular_mean(&angles);
            let response = Array1::from_iter(angles.iter().map(|&a| wrap_pi(a - mu)));
            (None, None, mu, response)
        }
        (_, ChartTopology::Interval { .. }) => (None, None, 0.0, coords_to.to_owned()),
    };

    // --- REML residual smooth on the source chart ---------------------------
    let basis = DomainBasis::build(topology_from, coords_from)?;
    let design = basis.value_rows(coords_from)?;
    let penalty = basis.penalty()?;
    let fit = fit_penalized_1d(
        &design,
        &penalty,
        response.view(),
        None,
        basis.penalty_rank(),
        false,
    )?;

    // --- isometry defect under the empirical density -------------------------
    let slope = degree.map_or(0.0, f64::from);
    let deriv_rows = basis.derivative_rows(coords_from)?;
    let deriv = deriv_rows.dot(&fit.beta).mapv(|v| v + slope);
    let m = basis.num_basis();
    let mut defect = 0.0_f64;
    let mut grad = Array1::<f64>::zeros(m);
    for i in 0..n {
        let speed = deriv[i].abs();
        let gap = speed - 1.0;
        defect += gap * gap;
        let sgn = if deriv[i] >= 0.0 { 1.0 } else { -1.0 };
        for j in 0..m {
            grad[j] += 2.0 * gap * sgn * deriv_rows[[i, j]];
        }
    }
    defect /= n as f64;
    grad.mapv_inplace(|v| v / n as f64);
    let isometry_defect_se = grad.dot(&fit.covariance.dot(&grad)).max(0.0).sqrt();

    // --- fold / orientation check on a dense grid ---------------------------
    let grid = domain_grid(topology_from, FOLD_CHECK_GRID);
    let grid_deriv = basis
        .derivative_rows(grid.view())?
        .dot(&fit.beta)
        .mapv(|v| v + slope);
    let orientation = if slope != 0.0 {
        slope.signum()
    } else {
        let mean = grid_deriv.iter().sum::<f64>() / grid_deriv.len() as f64;
        if mean < 0.0 { -1.0 } else { 1.0 }
    };
    let min_directional_derivative = grid_deriv
        .iter()
        .map(|&v| orientation * v)
        .fold(f64::INFINITY, f64::min);
    let topology_preserved = match (topology_from, topology_to) {
        (ChartTopology::Circle, ChartTopology::Circle) => {
            matches!(degree, Some(1) | Some(-1)) && min_directional_derivative > 0.0
        }
        (ChartTopology::Interval { .. }, ChartTopology::Interval { .. }) => {
            min_directional_derivative > 0.0
        }
        _ => false,
    };

    Ok(FittedTransport {
        topology_from,
        topology_to,
        degree,
        degree_concentration,
        rotation_offset,
        beta: fit.beta,
        covariance: fit.covariance,
        smoothing_lambda: fit.lambda,
        edf: fit.edf,
        noise_variance: fit.sigma2,
        n_obs: n,
        isometry_defect: defect,
        isometry_defect_se,
        topology_preserved,
        min_directional_derivative,
        residual_rms: fit.residual_rms,
        basis,
    })
}

/// Estimate the transport map between two layers and package the evidence.
pub fn fit_layer_transport(
    layer_from: usize,
    layer_to: usize,
    coords_from: ArrayView1<'_, f64>,
    coords_to: ArrayView1<'_, f64>,
    topology_from: ChartTopology,
    topology_to: ChartTopology,
) -> Result<LayerTransportReport, String> {
    Ok(
        fit_transport_map(coords_from, coords_to, topology_from, topology_to)?
            .report(layer_from, layer_to),
    )
}

/// Composition-law test report for one triple `(h_ab, h_bc, h_ac)`.
#[derive(Debug, Clone)]
pub struct CompositionDefectReport {
    pub n_grid: usize,
    /// Rotation gauge applied to the composed route (circle targets).
    pub gauge_rotation: f64,
    /// Whether the reflected target orientation minimized the defect.
    pub gauge_reflected: bool,
    pub mean_abs_defect: f64,
    pub rms_defect: f64,
    pub max_abs_defect: f64,
    /// `max_t |d(t)| / band(t)` against the composed pointwise bands.
    pub max_studentized_defect: f64,
    /// Bonferroni p-value bound for the max studentized defect over all tested
    /// grid points.
    pub max_studentized_p_value: f64,
    /// EDF of the variance-weighted REML defect smooth.
    pub defect_edf: f64,
    /// Wood rank-truncated Wald statistic of the defect smooth.
    pub statistic: f64,
    pub ref_df: f64,
    /// `wood_smooth_test` p-value for H₀: the gauge-aligned defect is zero.
    pub p_value: f64,
}

/// Uniform evaluation grid over a chart domain.
fn domain_grid(topology: ChartTopology, n: usize) -> Array1<f64> {
    match topology {
        ChartTopology::Circle => Array1::from_iter((0..n).map(|i| TAU * i as f64 / n as f64)),
        ChartTopology::Interval { lo, hi } => {
            Array1::from_iter((0..n).map(|i| lo + (hi - lo) * i as f64 / (n - 1).max(1) as f64))
        }
    }
}

/// Test the composition law `h_ac ≟ h_bc ∘ h_ab` on `n_grid` points.
///
/// The defect `d(t) = h_ac(t) ⊖ (h_bc ∘ h_ab)(t)` (circular difference on
/// circle targets) is first quotiented by the certified isometry gauge of the
/// TARGET chart only — the source gauge cancels because both routes consume
/// identical source coordinates (double-coset estimand; see module docs):
/// rotation fixed at the circular mean of the defect, reflection chosen as
/// the orientation with smaller squared defect. The aligned defect is then
/// (a) studentized pointwise against the composed delta-method bands
/// `var(h_ac) + var(h_bc) + h_bc′² var(h_ab)` (the three maps are fitted from
/// disjoint response pairs; cross-correlations through shared rows are
/// neglected), with the max studentized defect as the headline statistic, and
/// (b) smoothed by a variance-weighted known-scale REML fit whose coefficients
/// feed [`wood_smooth_test`] for the calibrated p-value.
pub fn composition_defect(
    h_ab: &FittedTransport,
    h_bc: &FittedTransport,
    h_ac: &FittedTransport,
    n_grid: usize,
) -> Result<CompositionDefectReport, String> {
    if h_ab.topology_from != h_ac.topology_from
        || h_ab.topology_to != h_bc.topology_from
        || h_bc.topology_to != h_ac.topology_to
    {
        return Err("composition defect requires chart-compatible transports: \
             h_ab: A→B, h_bc: B→C, h_ac: A→C"
            .to_string());
    }
    if n_grid < MIN_TRANSPORT_OBS {
        return Err(format!(
            "composition defect grid must have at least {MIN_TRANSPORT_OBS} points, got {n_grid}"
        ));
    }

    let grid = domain_grid(h_ab.topology_from, n_grid);
    let (direct, var_direct) = h_ac.eval_with_variance(grid.view())?;
    let (mid, var_mid) = h_ab.eval_with_variance(grid.view())?;
    let (composed, var_bc) = h_bc.eval_with_variance(mid.view())?;
    let mid_slope = h_bc.derivative(mid.view())?;
    let mut variance = Array1::<f64>::zeros(n_grid);
    for i in 0..n_grid {
        variance[i] = var_direct[i] + var_bc[i] + mid_slope[i] * mid_slope[i] * var_mid[i];
    }

    // --- target-chart gauge alignment (rotation + reflection only) ----------
    let circle_target = matches!(h_ac.topology_to, ChartTopology::Circle);
    let mut gauge_reflected = false;
    let mut gauge_rotation = 0.0_f64;
    let mut defect = Array1::<f64>::zeros(n_grid);
    let mut best_sse = f64::INFINITY;
    for reflected in [false, true] {
        let composed_oriented: Array1<f64> = match (h_ac.topology_to, reflected) {
            (_, false) => composed.clone(),
            (ChartTopology::Circle, true) => composed.mapv(|v| wrap_tau(-v)),
            (ChartTopology::Interval { lo, hi }, true) => composed.mapv(|v| lo + hi - v),
        };
        let (rotation, candidate): (f64, Array1<f64>) = if circle_target {
            let raw: Vec<f64> = (0..n_grid)
                .map(|i| wrap_pi(direct[i] - composed_oriented[i]))
                .collect();
            let rot = circular_mean(&raw);
            (
                rot,
                Array1::from_iter(raw.iter().map(|&d| wrap_pi(d - rot))),
            )
        } else {
            (
                0.0,
                Array1::from_iter((0..n_grid).map(|i| direct[i] - composed_oriented[i])),
            )
        };
        let sse = candidate.iter().map(|&d| d * d).sum::<f64>();
        if sse < best_sse {
            best_sse = sse;
            gauge_reflected = reflected;
            gauge_rotation = rotation;
            defect = candidate;
        }
    }

    // --- pointwise studentization against the composed bands ----------------
    let max_var = variance.iter().copied().fold(0.0_f64, f64::max);
    let var_floor = (max_var * 1e-10).max(f64::MIN_POSITIVE);
    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    let mut max_z = 0.0_f64;
    for i in 0..n_grid {
        let d = defect[i];
        let a = d.abs();
        max_abs = max_abs.max(a);
        sum_abs += a;
        sum_sq += d * d;
        let z = a / variance[i].max(var_floor).sqrt();
        max_z = max_z.max(z);
    }
    let mean_abs_defect = sum_abs / n_grid as f64;
    let rms_defect = (sum_sq / n_grid as f64).sqrt();

    // --- variance-weighted REML defect smooth + Wood Wald test ---------------
    let basis = DomainBasis::build(h_ab.topology_from, grid.view())?;
    let design = basis.value_rows(grid.view())?;
    let penalty = basis.penalty()?;
    let weights = variance.mapv(|v| 1.0 / v.max(var_floor));
    let fit = fit_penalized_1d(
        &design,
        &penalty,
        defect.view(),
        Some(weights.view()),
        basis.penalty_rank(),
        true,
    )?;
    let m = basis.num_basis();
    let test = wood_smooth_test(SmoothTestInput {
        beta: fit.beta.view(),
        covariance: &fit.covariance,
        influence_matrix: Some(&fit.influence),
        coeff_range: 0..m,
        edf: fit.edf,
        nullspace_dim: 0,
        residual_df: (n_grid as f64 - fit.edf).max(1.0),
        scale: SmoothTestScale::Known,
    })
    .ok_or_else(|| "composition defect smooth test degenerated".to_string())?;

    // Bonferroni bound for the max studentized defect over the actual grid:
    // valid for arbitrary dependence among the tested pointwise contrasts.
    let normal =
        Normal::new(0.0, 1.0).map_err(|e| format!("standard normal construction failed: {e}"))?;
    let pointwise = (2.0 * (1.0 - normal.cdf(max_z))).clamp(0.0, 1.0);
    let max_studentized_p_value = (n_grid as f64 * pointwise).min(1.0);

    Ok(CompositionDefectReport {
        n_grid,
        gauge_rotation,
        gauge_reflected,
        mean_abs_defect,
        rms_defect,
        max_abs_defect: max_abs,
        max_studentized_defect: max_z,
        max_studentized_p_value,
        defect_edf: fit.edf,
        statistic: test.statistic,
        ref_df: test.ref_df,
        p_value: test.p_value,
    })
}

/// Full transport report for a ladder of layers: every adjacent map plus
/// every two-hop map with its composition-law test attached.
#[derive(Debug, Clone)]
pub struct TransportLadderReport {
    /// `h_{l→l+1}` for each consecutive pair.
    pub adjacent: Vec<LayerTransportReport>,
    /// `h_{l→l+2}` with the composition test against the composed adjacent
    /// pair merged in.
    pub two_hop: Vec<LayerTransportReport>,
}

/// Fit the whole transport ladder: adjacent maps, two-hop maps, and the
/// composition law `h_{l→l+2} ≟ h_{l+1→l+2} ∘ h_{l→l+1}` per triple.
///
/// `layers[k]`, `coords[k]`, `topologies[k]` describe layer `k` of the
/// ladder; all coordinate vectors must index the same rows.
pub fn transport_ladder(
    layers: &[usize],
    coords: &[Array1<f64>],
    topologies: &[ChartTopology],
) -> Result<TransportLadderReport, String> {
    let depth = layers.len();
    if coords.len() != depth || topologies.len() != depth {
        return Err(format!(
            "transport ladder inputs disagree: {depth} layers, {} coordinate vectors, {} topologies",
            coords.len(),
            topologies.len()
        ));
    }
    if depth < 2 {
        return Err("transport ladder needs at least two layers".to_string());
    }

    let mut adjacent_fits: Vec<FittedTransport> = Vec::with_capacity(depth - 1);
    let mut adjacent: Vec<LayerTransportReport> = Vec::with_capacity(depth - 1);
    for k in 0..depth - 1 {
        let fit = fit_transport_map(
            coords[k].view(),
            coords[k + 1].view(),
            topologies[k],
            topologies[k + 1],
        )
        .map_err(|e| {
            format!(
                "adjacent transport {}→{} failed: {e}",
                layers[k],
                layers[k + 1]
            )
        })?;
        adjacent.push(fit.report(layers[k], layers[k + 1]));
        adjacent_fits.push(fit);
    }

    let mut two_hop: Vec<LayerTransportReport> = Vec::with_capacity(depth.saturating_sub(2));
    for k in 0..depth.saturating_sub(2) {
        let direct = fit_transport_map(
            coords[k].view(),
            coords[k + 2].view(),
            topologies[k],
            topologies[k + 2],
        )
        .map_err(|e| {
            format!(
                "two-hop transport {}→{} failed: {e}",
                layers[k],
                layers[k + 2]
            )
        })?;
        let composition = composition_defect(
            &adjacent_fits[k],
            &adjacent_fits[k + 1],
            &direct,
            DEFAULT_COMPOSITION_GRID,
        )
        .map_err(|e| {
            format!(
                "composition test {}→{}→{} failed: {e}",
                layers[k],
                layers[k + 1],
                layers[k + 2]
            )
        })?;
        two_hop.push(
            direct
                .report(layers[k], layers[k + 2])
                .with_composition(&composition),
        );
    }

    Ok(TransportLadderReport { adjacent, two_hop })
}
