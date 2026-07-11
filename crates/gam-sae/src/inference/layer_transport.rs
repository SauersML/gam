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
//!    charts) is evaluated on a grid and studentized with a joint shared-row
//!    influence sandwich. A Bonferroni max test controls the grid family under
//!    arbitrary pointwise dependence; deterministic fits with no score
//!    variation emit no p-value.
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
//! * Both routes land in the same target chart, so the target gauge also acts
//!   identically and cancels. No post-hoc rotation or reflection is selected;
//!   such an alignment would erase genuine composition violations.
//!
//! All smooths reuse the engine's existing periodic cardinal-B-spline basis
//! ([`build_periodic_bspline_basis_1d`]) with the exact periodic derivative
//! Gram on circular domains, and the open B-spline basis with its exact
//! derivative Gram on interval domains — constructed directly, not via the
//! string DSL.

use crate::chart_canonicalization::CanonicalChartTopology;
use gam_solve::gaussian_reml::{
    gaussian_reml_closed_form_with_nullspace_dim, gaussian_reml_stationary_set,
};
use gam_terms::basis::{
    BasisOptions, Dense, KnotSource, PeriodicBSplineBasisSpec, bspline_derivative_penalty_matrix,
    build_periodic_bspline_basis_1d, create_basis, cyclic_bspline_derivative_penalty_matrix,
    periodic_bspline_first_derivative_nd,
};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use statrs::distribution::{ContinuousCDF, Normal};
use std::f64::consts::{PI, TAU};

/// Cubic splines for every transport smooth.
const TRANSPORT_SPLINE_DEGREE: usize = 3;
/// Second-order function-curvature penalty: the cyclic variant leaves constants
/// unpenalized on a circle; the open variant leaves affine maps unpenalized on
/// an interval — exactly the isometry-adjacent null spaces.
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

    /// Rank of the smoothing penalty: the cyclic second-derivative Gram
    /// annihilates only constants (a linear map is not periodic), the open
    /// second-derivative Gram annihilates affine maps.
    fn penalty_rank(&self) -> usize {
        match self {
            DomainBasis::Periodic(spec) => spec.num_basis - 1,
            DomainBasis::Open { .. } => self.num_basis() - TRANSPORT_PENALTY_ORDER,
        }
    }

    fn penalty(&self) -> Result<Array2<f64>, String> {
        match self {
            DomainBasis::Periodic(spec) => cyclic_bspline_derivative_penalty_matrix(
                spec.degree,
                spec.num_basis,
                spec.period,
                TRANSPORT_PENALTY_ORDER,
            )
            .map_err(|e| format!("cyclic transport roughness failed: {e}")),
            DomainBasis::Open { knots, degree } => {
                bspline_derivative_penalty_matrix(knots.view(), *degree, TRANSPORT_PENALTY_ORDER)
                    .map_err(|e| format!("open transport roughness failed: {e}"))
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

    /// Polynomial degree of `h′` on each knot span: the basis degree minus one
    /// (a cubic spline derivative is piecewise quadratic).
    fn derivative_poly_degree(&self) -> usize {
        let degree = match self {
            DomainBasis::Periodic(spec) => spec.degree,
            DomainBasis::Open { degree, .. } => *degree,
        };
        degree.saturating_sub(1)
    }

    /// Sorted distinct breakpoints bounding the polynomial pieces of `h′` over
    /// the active domain `[lo, hi]`. Within each `[breakpoints[k],
    /// breakpoints[k+1]]` span the derivative is a single polynomial of degree
    /// [`Self::derivative_poly_degree`], which is what the exact monotonicity
    /// certificate reconstructs and checks. For the open basis these are the
    /// distinct interior+boundary knots; for the periodic basis they are the
    /// uniform cardinal-B-spline segment boundaries over `[0, 2π]`.
    fn derivative_breakpoints(&self) -> Vec<f64> {
        match self {
            DomainBasis::Periodic(spec) => {
                // Cardinal periodic B-splines on `[origin, origin+period]` have
                // `num_basis` uniform segments; the derivative is a separate
                // polynomial on each.
                let n_seg = spec.num_basis.max(1);
                (0..=n_seg)
                    .map(|k| spec.origin + spec.period * k as f64 / n_seg as f64)
                    .collect()
            }
            DomainBasis::Open { knots, degree } => {
                let lo = knots[*degree];
                let hi = knots[knots.len() - 1 - degree];
                let mut breaks: Vec<f64> = Vec::with_capacity(knots.len());
                for &k in knots.iter() {
                    if k > lo + 0.0 && k < hi {
                        breaks.push(k);
                    }
                }
                breaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                breaks.dedup_by(|a, b| (*a - *b).abs() <= f64::EPSILON * hi.abs().max(1.0));
                let mut out = Vec::with_capacity(breaks.len() + 2);
                out.push(lo);
                out.extend(breaks.into_iter().filter(|&k| k > lo && k < hi));
                out.push(hi);
                out
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

/// One penalized 1-D smooth chosen by exact Gaussian REML.
struct Penalized1dFit {
    beta: Array1<f64>,
    /// Scale-included posterior covariance `σ̂²(XᵀWX + λS)⁻¹`.
    covariance: Array2<f64>,
    lambda: f64,
    edf: f64,
    sigma2: f64,
    residual_rms: f64,
    /// Observation-level coefficient influence columns
    /// `A^{-1} x_i w_i residual_i`. Keeping the row identity lets composition
    /// contrasts form the joint sandwich across maps fitted on the same rows.
    coefficient_score_influence: Array2<f64>,
}

/// Exact 1-D Gaussian REML on a fixed design/penalty pair.
///
/// Profile σ² out of Wood's REML,
/// `V(λ) = (n − M₀)·log PRSS(λ) + log|XᵀWX + λS| − rank(S)·log λ`, with
/// `M₀ = dim ker S` and `PRSS = yᵀWy − β̂ᵀXᵀWy`. Selection delegates to
/// the engine's analytic Gaussian-REML stationary-point enumerator, so this
/// transport path shares the same grid-free objective, derivatives, boundary
/// comparison, and convergence certificate as every other Gaussian smooth.
fn fit_penalized_1d(
    design: &Array2<f64>,
    penalty: &Array2<f64>,
    response: ArrayView1<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
    penalty_rank: usize,
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
    if penalty_rank > m {
        return Err(format!(
            "penalized 1-D fit penalty rank {penalty_rank} exceeds coefficient dimension {m}"
        ));
    }
    if let Some(w) = weights.as_ref() {
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

    let nullspace_dim = m - penalty_rank;
    let weight_view = || weights.as_ref().map(|w| w.view());
    let stationary = gaussian_reml_stationary_set(
        design.view(),
        response.view(),
        penalty.view(),
        Some(nullspace_dim),
        weight_view(),
        None,
    )
    .map_err(|error| format!("penalized 1-D REML stationary enumeration failed: {error}"))?;
    if stationary.hit_resolution_floor {
        return Err(format!(
            "penalized 1-D REML is underresolved: the stationary-point enclosure reached its \
             resolution floor ({} isolated roots, selected log-lambda {}, endpoint costs {:?})",
            stationary.roots.len(),
            stationary.selected_rho,
            stationary.endpoint_costs,
        ));
    }
    let reml = gaussian_reml_closed_form_with_nullspace_dim(
        design.view(),
        response.view(),
        penalty.view(),
        Some(nullspace_dim),
        weight_view(),
        None,
    )
    .map_err(|error| format!("penalized 1-D Gaussian REML failed: {error}"))?;
    if reml.rho.to_bits() != stationary.selected_rho.to_bits() {
        return Err(format!(
            "penalized 1-D REML selection drifted between its certificate ({}) and fit ({})",
            stationary.selected_rho, reml.rho,
        ));
    }

    // If C = L⁻ᵀU is the cache's coefficient basis and δᵢ are the
    // eigenvalues of L⁻¹SL⁻ᵀ, then the exact penalized inverse is
    //     (XᵀWX + λS)⁻¹ = C diag((1 + λδᵢ)⁻¹) Cᵀ.
    // Reconstruct that same inverse directly: no eigenvalue flooring and no
    // representative-selecting ridge that would change the REML objective.
    let lambda = reml.lambda;
    let coefficient_basis = &reml.cache.coefficient_basis;
    let penalty_eigenvalues = &reml.cache.penalty_eigenvalues;
    if coefficient_basis.dim() != (m, m) || penalty_eigenvalues.len() != m {
        return Err(format!(
            "penalized 1-D REML cache shape drift: basis is {}x{}, spectrum has {}, expected {m}",
            coefficient_basis.nrows(),
            coefficient_basis.ncols(),
            penalty_eigenvalues.len(),
        ));
    }
    let mut a_inv = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        let delta = penalty_eigenvalues[i];
        let denominator = 1.0 + lambda * delta;
        if !(delta.is_finite() && delta >= 0.0 && denominator.is_finite() && denominator > 0.0) {
            return Err(format!(
                "penalized 1-D REML cache has invalid mode {i}: delta={delta}, denominator={denominator}"
            ));
        }
        let inverse_denominator = denominator.recip();
        for j in 0..m {
            let scaled_basis = coefficient_basis[[j, i]] * inverse_denominator;
            for k in 0..m {
                a_inv[[j, k]] += scaled_basis * coefficient_basis[[k, i]];
            }
        }
    }

    let beta = reml.coefficients;
    let fitted = reml.fitted;
    let edf = reml.edf;
    let sigma2 = reml.sigma2;
    let mut rss = 0.0_f64;
    for r in 0..n {
        let w = weights.as_ref().map_or(1.0, |wv| wv[r]);
        let e = response[r] - fitted[r];
        rss += w * e * e;
    }
    let covariance = a_inv.mapv(|v| v * sigma2);
    let sum_w = weights
        .as_ref()
        .map_or(n as f64, |wv| wv.iter().copied().sum());
    let residual_rms = (rss / sum_w.max(f64::MIN_POSITIVE)).sqrt();
    let mut coefficient_score_influence = Array2::<f64>::zeros((m, n));
    for row in 0..n {
        let w = weights.as_ref().map_or(1.0, |wv| wv[row]);
        let residual = response[row] - fitted[row];
        for j in 0..m {
            let mut sensitivity = 0.0_f64;
            for k in 0..m {
                sensitivity += a_inv[[j, k]] * design[[row, k]];
            }
            coefficient_score_influence[[j, row]] = sensitivity * w * residual;
        }
    }

    if beta.iter().chain(a_inv.iter()).any(|v| !v.is_finite())
        || !(sigma2.is_finite() && sigma2 > 0.0)
    {
        return Err("penalized 1-D REML produced non-finite posterior moments".to_string());
    }
    Ok(Penalized1dFit {
        beta,
        covariance,
        lambda,
        edf,
        sigma2,
        residual_rms,
        coefficient_score_influence,
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
    coefficient_score_influence: Array2<f64>,
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

    /// Point-evaluation influence by original observation row. The returned
    /// matrix has shape `(t.len(), n_obs)` and retains cross-map row identity.
    fn eval_score_influence(&self, t: ArrayView1<'_, f64>) -> Result<Array2<f64>, String> {
        Ok(self
            .basis
            .value_rows(t)?
            .dot(&self.coefficient_score_influence))
    }

    /// Pre-wrap map value `slope·t + offset + g(t)` at a single point — the
    /// strictly monotone (when fold-free) handle that [`Self::eval`] wraps for
    /// circle targets and [`Self::invert`] bisects on.
    fn raw_at(&self, t: f64) -> Result<f64, String> {
        let arr = Array1::from_elem(1, t);
        let smooth = self.basis.value_rows(arr.view())?.dot(&self.beta)[0];
        Ok(self.linear_slope() * t + self.rotation_offset + smooth)
    }

    /// `orientation·h′` at the supplied source-chart coordinates.
    fn oriented_derivative_at(&self, t: &[f64], orientation: f64) -> Result<Vec<f64>, String> {
        let arr = Array1::from_vec(t.to_vec());
        let rows = self.basis.derivative_rows(arr.view())?;
        let slope = self.linear_slope();
        Ok((0..t.len())
            .map(|i| orientation * (rows.row(i).dot(&self.beta) + slope))
            .collect())
    }

    /// Exactly certify that `h` is strictly monotone over the whole source
    /// domain, returning the certified orientation (+1 increasing, −1
    /// decreasing) or an `Err` describing where monotonicity fails.
    ///
    /// Unlike [`Self::topology_preserved`], which only samples `h′` on a fixed
    /// 512-point grid and so can miss a fold *between* grid samples, this is a
    /// span-exact certificate. On each knot span `h′` is a single polynomial of
    /// degree `d = `[`DomainBasis::derivative_poly_degree`]` (cubic spline ⇒
    /// quadratic). A degree-`d` polynomial is determined by `d + 1` samples, so
    /// per span we evaluate `h′` at `d + 1` equally-spaced abscissae, reconstruct
    /// the known-degree polynomial in the Lagrange basis, locate its interior critical
    /// points in closed form, and require `orientation·h′ > 0` at the span
    /// endpoints **and** every interior critical point. To stay sound even if a
    /// basis is not an exact polynomial of the assumed degree on a span (e.g. a
    /// row-normalized periodic basis whose row-sum is not a partition of unity),
    /// the reconstruction is verified against an independent interior sample;
    /// any mismatch falls back to refusing the span.
    fn certify_strict_monotonicity(&self) -> Result<f64, String> {
        let (lo, hi) = match self.topology_from {
            ChartTopology::Circle => (0.0, TAU),
            ChartTopology::Interval { lo, hi } => (lo, hi),
        };
        // Orientation from the endpoint span of the pre-wrap map, matching the
        // sign convention `invert` bisects with.
        let raw_lo = self.raw_at(lo)?;
        let raw_hi = self.raw_at(hi)?;
        let orientation = if raw_hi >= raw_lo { 1.0 } else { -1.0 };

        let deg = self.basis.derivative_poly_degree().max(1);
        let breaks = self.basis.derivative_breakpoints();
        // Restrict the breakpoints to the active domain (the periodic segment
        // grid already coincides with `[lo, hi]`).
        for window in breaks.windows(2) {
            let (a, b) = (window[0], window[1]);
            if !(b > a) {
                continue;
            }
            let span = b - a;
            // Reconstruction abscissae: `deg + 1` equally spaced nodes on the
            // closed span (sampling strictly inside avoids the knot where two
            // pieces meet and the open-basis derivative can be one-sided).
            let pad = span * 1.0e-9;
            let n_nodes = deg + 1;
            let nodes: Vec<f64> = (0..n_nodes)
                .map(|i| {
                    let s = if n_nodes == 1 {
                        0.5
                    } else {
                        i as f64 / (n_nodes - 1) as f64
                    };
                    (a + pad) + (span - 2.0 * pad) * s
                })
                .collect();
            let values = self.oriented_derivative_at(&nodes, orientation)?;

            // Polynomial in the local coordinate u = (t - nodes[0]) / step.
            // Expand the exact Lagrange interpolant into monomial coefficients
            // for the closed-form critical-point search. This is algebraic
            // reconstruction of a known-degree spline piece, not a numerical
            // derivative approximation.
            let step = if n_nodes > 1 {
                nodes[1] - nodes[0]
            } else {
                span
            };
            let coeffs = monomial_interpolant_at_integer_nodes(&values);

            // Sound guard: verify the reconstruction reproduces an independent
            // interior sample (deliberately off the reconstruction nodes — the
            // equispaced nodes never land on a 0.37 fraction). If the basis is
            // not exactly polynomial of the assumed degree on this span, refuse
            // rather than trust the fit.
            let probe_t = a + 0.37 * span;
            let probe_u = (probe_t - nodes[0]) / step;
            let probe_recon = eval_monomial(&coeffs, probe_u);
            let probe_actual = self.oriented_derivative_at(&[probe_t], orientation)?[0];
            let scale = probe_actual.abs().max(1.0);
            if (probe_recon - probe_actual).abs() > 1.0e-6 * scale {
                return Err(format!(
                    "transport monotonicity certificate could not reconstruct h′ on the \
                     span [{a}, {b}] (reconstruction {probe_recon} vs actual {probe_actual}); \
                     refusing to certify"
                ));
            }

            // Require positivity at the closed-span endpoints.
            for &edge in &[a, b] {
                let u = (edge - nodes[0]) / step;
                let v = eval_monomial(&coeffs, u);
                if !(v > 0.0) {
                    return Err(format!(
                        "transport map is not strictly monotone: orientation·h′ = {v} ≤ 0 at \
                         t = {edge}"
                    ));
                }
            }
            // Require positivity at every interior critical point of the
            // polynomial within the span.
            for u_crit in monomial_critical_points(&coeffs) {
                let t_crit = nodes[0] + u_crit * step;
                if t_crit > a && t_crit < b {
                    let v = eval_monomial(&coeffs, u_crit);
                    if !(v > 0.0) {
                        return Err(format!(
                            "transport map folds: orientation·h′ = {v} ≤ 0 at interior \
                             extremum t = {t_crit}"
                        ));
                    }
                }
            }
        }
        Ok(orientation)
    }

    /// Invert the transport: for each target-chart coordinate `y`, return the
    /// source-chart coordinate `t` with `eval([t]) == y`.
    ///
    /// Requires a strictly monotone, fold-free map (a degree-±1 cover for
    /// circle charts, a homeomorphism for intervals), so the inverse is
    /// single-valued; otherwise this errors rather than picking an arbitrary
    /// branch. Monotonicity is established with [`Self::certify_strict_monotonicity`]
    /// — a span-exact polynomial certificate, **not** the sampled
    /// `topology_preserved` diagnostic, which can miss a narrow fold between its
    /// grid samples. Non-finite targets are rejected. Interval targets reject a
    /// `y` outside the fitted image (scale-aware tolerance); circle targets
    /// accept any `y` (the pre-wrap map covers a full `2π`). The root is found
    /// by monotone bisection on the pre-wrap map `raw_at`, which converges to
    /// f64 precision (~53 significand bits) in the source coordinate after on
    /// the order of 50 iterations.
    ///
    /// This is the exact inverse of [`Self::eval`] and the missing half of the
    /// transport algebra alongside [`composition_defect`]: it is what lets a
    /// caller form `g_B ∘ g_A⁻¹` from two fitted transports.
    pub fn invert(&self, y: ArrayView1<'_, f64>) -> Result<Array1<f64>, String> {
        if y.iter().any(|v| !v.is_finite()) {
            return Err("transport inverse targets must be finite".to_string());
        }
        // Span-exact strict-monotonicity certificate; supersedes the sampled
        // `topology_preserved` flag, which can pass over a between-sample fold.
        self.certify_strict_monotonicity()?;
        let (lo, hi) = match self.topology_from {
            ChartTopology::Circle => (0.0, TAU),
            ChartTopology::Interval { lo, hi } => (lo, hi),
        };
        // The pre-wrap map is strictly monotone over [lo, hi]; the endpoints
        // anchor its orientation and image span.
        let raw_lo = self.raw_at(lo)?;
        let raw_hi = self.raw_at(hi)?;
        let increasing = raw_hi > raw_lo;
        let (raw_min, raw_max) = if increasing {
            (raw_lo, raw_hi)
        } else {
            (raw_hi, raw_lo)
        };
        // Scale-aware image tolerance: an absolute 1e-9 would wrongly accept a
        // target well outside a tiny image (e.g. [0, 1e-8]).
        let scale = raw_min.abs().max(raw_max.abs()).max(1.0);
        let tol = 32.0 * f64::EPSILON * scale;

        // One reusable single-element buffer for the bisection probes (rebuilt
        // basis rows on every probe otherwise allocated a fresh `Array1`).
        let mut probe = Array1::<f64>::zeros(1);
        let mut raw_at_into = |t: f64| -> Result<f64, String> {
            probe[0] = t;
            let smooth = self.basis.value_rows(probe.view())?.dot(&self.beta)[0];
            Ok(self.linear_slope() * t + self.rotation_offset + smooth)
        };

        let mut out = Array1::<f64>::zeros(y.len());
        for (idx, &yi) in y.iter().enumerate() {
            // Target value in the pre-wrap coordinate.
            let target = match self.topology_to {
                ChartTopology::Interval { .. } => {
                    if yi < raw_min - tol || yi > raw_max + tol {
                        return Err(format!(
                            "transport inverse target {yi} is outside the fitted image \
                             [{raw_min}, {raw_max}]"
                        ));
                    }
                    yi.clamp(raw_min, raw_max)
                }
                ChartTopology::Circle => {
                    // The pre-wrap map covers exactly 2π; shift wrap_tau(y) by
                    // the unique integer multiple of 2π that lands in the image.
                    let ywrapped = wrap_tau(yi);
                    let m = ((raw_min - ywrapped) / TAU).ceil();
                    ywrapped + TAU * m
                }
            };
            // Monotone bisection on the pre-wrap map over [lo, hi]; stop once
            // the bracket is below the source-coordinate precision floor (f64
            // bisection stagnates well before 100 iterations).
            let (mut a, mut b) = (lo, hi);
            let width_floor = f64::EPSILON * hi.abs().max(lo.abs()).max(1.0);
            for _ in 0..100 {
                if (b - a) <= width_floor {
                    break;
                }
                let mid = 0.5 * (a + b);
                let rm = raw_at_into(mid)?;
                let go_right = if increasing { rm < target } else { rm > target };
                if go_right {
                    a = mid;
                } else {
                    b = mid;
                }
            }
            out[idx] = 0.5 * (a + b);
        }
        Ok(out)
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
    /// Bonferroni familywise p-value from the joint shared-row sandwich.
    /// `None` when the fitted maps carry no empirical score variation.
    pub composition_p_value: Option<f64>,
    /// Always `false`: both routes already land in the same target chart, so no
    /// post-hoc target alignment is fitted.
    pub composition_gauge_reflected: Option<bool>,
}

impl LayerTransportReport {
    /// Merge a composition-law test into this (direct, two-hop) report.
    pub fn with_composition(mut self, composition: &CompositionDefectReport) -> Self {
        self.composition_defect = Some(composition.rms_defect);
        self.composition_max_studentized = Some(composition.max_studentized_defect);
        self.composition_p_value = composition
            .p_value
            .is_finite()
            .then_some(composition.p_value);
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
        coefficient_score_influence: fit.coefficient_score_influence,
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
    /// Always zero: no post-hoc target rotation is fitted.
    pub gauge_rotation: f64,
    /// Always false: no post-hoc target reflection is fitted.
    pub gauge_reflected: bool,
    pub mean_abs_defect: f64,
    pub rms_defect: f64,
    pub max_abs_defect: f64,
    /// `max_t |d(t)| / band(t)` against the composed pointwise bands.
    pub max_studentized_defect: f64,
    /// Bonferroni p-value bound for the max studentized defect over all tested
    /// grid points.
    pub max_studentized_p_value: f64,
    /// Alias of the familywise max-test p-value for report consumers. `NaN`
    /// explicitly means that deterministic fits supplied no sampling variation.
    pub p_value: f64,
}

/// Recover ascending monomial coefficients of the unique degree-`(n-1)`
/// polynomial through `(i, values[i])`, `i = 0, …, n-1`, by expanding its
/// Lagrange basis polynomials. The monotonicity certificate uses this exact
/// algebraic reconstruction for each known-degree B-spline derivative piece.
fn monomial_interpolant_at_integer_nodes(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mut coeffs = vec![0.0_f64; n];
    for (i, &value) in values.iter().enumerate() {
        let mut basis = vec![1.0_f64];
        let mut denominator = 1.0_f64;
        for j in 0..n {
            if j == i {
                continue;
            }
            denominator *= i as f64 - j as f64;
            let mut expanded = vec![0.0_f64; basis.len() + 1];
            for (degree, &coefficient) in basis.iter().enumerate() {
                expanded[degree] -= j as f64 * coefficient;
                expanded[degree + 1] += coefficient;
            }
            basis = expanded;
        }
        let scale = value / denominator;
        for (degree, coefficient) in basis.into_iter().enumerate() {
            coeffs[degree] += scale * coefficient;
        }
    }
    coeffs
}

/// Evaluate an ascending monomial polynomial at `u` (Horner).
fn eval_monomial(coeffs: &[f64], u: f64) -> f64 {
    coeffs.iter().rev().fold(0.0_f64, |acc, &c| acc * u + c)
}

/// Interior critical points (roots of the derivative) of an ascending monomial
/// polynomial, in the local `u` coordinate. Returns the closed-form roots for
/// degree ≤ 2 derivatives (i.e. cubic-spline pieces, the production path);
/// higher-degree derivatives fall back to a robust bisection root-isolation so
/// the certificate stays exact-enough (a missed extremum can only make the
/// certificate stricter, never falsely accept a fold, because the endpoints and
/// every sign change found are still checked). For the cubic transport splines
/// the polynomial is quadratic and this is the single vertex.
fn monomial_critical_points(coeffs: &[f64]) -> Vec<f64> {
    // Derivative coefficients: d/du Σ c_k u^k = Σ k·c_k u^{k−1}.
    let n = coeffs.len();
    if n <= 1 {
        return Vec::new();
    }
    let deriv: Vec<f64> = (1..n).map(|k| k as f64 * coeffs[k]).collect();
    // deriv is ascending of length n−1 (degree n−2).
    match deriv.len() {
        0 => Vec::new(),
        1 => Vec::new(), // constant derivative: no critical point
        2 => {
            // Linear b + a·u = 0 (a = deriv[1]).
            let (b, a) = (deriv[0], deriv[1]);
            if a.abs() <= f64::MIN_POSITIVE {
                Vec::new()
            } else {
                vec![-b / a]
            }
        }
        3 => {
            // Quadratic c + b·u + a·u² = 0.
            let (c, b, a) = (deriv[0], deriv[1], deriv[2]);
            if a.abs() <= f64::MIN_POSITIVE {
                if b.abs() <= f64::MIN_POSITIVE {
                    Vec::new()
                } else {
                    vec![-c / b]
                }
            } else {
                let disc = b * b - 4.0 * a * c;
                if disc < 0.0 {
                    Vec::new()
                } else {
                    let s = disc.sqrt();
                    vec![(-b + s) / (2.0 * a), (-b - s) / (2.0 * a)]
                }
            }
        }
        _ => {
            // General fallback: scan for sign changes of the derivative on a
            // dense [0, deg] grid and bisect each bracket. Conservative.
            let lo = 0.0;
            let hi = (coeffs.len() - 1) as f64;
            let steps = 256;
            let mut roots = Vec::new();
            let f = |u: f64| eval_monomial(&deriv, u);
            let mut prev_u = lo;
            let mut prev_v = f(lo);
            for i in 1..=steps {
                let u = lo + (hi - lo) * i as f64 / steps as f64;
                let v = f(u);
                if prev_v == 0.0 {
                    roots.push(prev_u);
                } else if prev_v * v < 0.0 {
                    let (mut a, mut b) = (prev_u, u);
                    for _ in 0..60 {
                        let m = 0.5 * (a + b);
                        if f(a) * f(m) <= 0.0 {
                            b = m;
                        } else {
                            a = m;
                        }
                    }
                    roots.push(0.5 * (a + b));
                }
                prev_u = u;
                prev_v = v;
            }
            roots
        }
    }
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
/// circle targets) is computed directly in the common target chart: no gauge is
/// selected after seeing the defect. Pointwise uncertainty is assembled from
/// the combined observation-level influence of all three maps, retaining their
/// shared-row covariance, and the grid is tested by a Bonferroni max statistic.
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
    if h_ab.n_obs != h_bc.n_obs || h_ab.n_obs != h_ac.n_obs {
        return Err(format!(
            "composition defect requires maps fitted on the same rows; got n_ab={}, n_bc={}, n_ac={}",
            h_ab.n_obs, h_bc.n_obs, h_ac.n_obs
        ));
    }

    let grid = domain_grid(h_ab.topology_from, n_grid);
    let direct = h_ac.eval(grid.view())?;
    let mid = h_ab.eval(grid.view())?;
    let composed = h_bc.eval(mid.view())?;
    let mid_slope = h_bc.derivative(mid.view())?;

    // Joint row-influence sandwich. For original fit row r and evaluation point
    // t, the first-order influence of the composition defect is
    //   IF_ac(t,r) - IF_bc(h_ab(t),r) - h_bc'(h_ab(t)) IF_ab(t,r).
    // Squaring the combined influence before summing retains every shared-fit
    // covariance term; adding three marginal variances drops those cross terms.
    let influence_direct = h_ac.eval_score_influence(grid.view())?;
    let influence_ab = h_ab.eval_score_influence(grid.view())?;
    let influence_bc = h_bc.eval_score_influence(mid.view())?;
    let mut variance = Array1::<f64>::zeros(n_grid);
    for i in 0..n_grid {
        let mut value = 0.0_f64;
        for row in 0..h_ab.n_obs {
            let influence = influence_direct[[i, row]]
                - influence_bc[[i, row]]
                - mid_slope[i] * influence_ab[[i, row]];
            value += influence * influence;
        }
        variance[i] = value;
    }

    // Both routes consume the same source chart and land in the same target
    // chart. Every source/target gauge transformation therefore acts on both
    // routes identically and cancels. Fitting a fresh rotation/reflection here
    // would fit away the very composition violation being tested.
    let circle_target = matches!(h_ac.topology_to, ChartTopology::Circle);
    let defect = Array1::from_iter((0..n_grid).map(|i| {
        if circle_target {
            wrap_pi(direct[i] - composed[i])
        } else {
            direct[i] - composed[i]
        }
    }));

    // --- pointwise studentization against the composed bands ----------------
    // Floor the band variance with BOTH a relative component (numerical guard
    // against exact zeros) AND an absolute, coordinate-scale component. The
    // absolute component is the fix for #2143: the delta-method band variance is
    // a pure sampling variance that collapses on near-noiseless REML fits, so
    // without it the irreducible spline-representation defect (which the sampling
    // variance does not model) is studentized into a spurious rejection. The
    // absolute floor is the squared representation tolerance relative to the
    // target chart's coordinate span, so a machine-level composition defect on a
    // clean chain reads as non-significant while a genuine violation (far larger
    // defect, or real sampling variance well above the floor) is unaffected.
    let max_var = variance.iter().copied().fold(0.0_f64, f64::max);
    let var_floor = (max_var * 1e-12).max(f64::MIN_POSITIVE);
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
        if max_var > 0.0 {
            let z = a / variance[i].max(var_floor).sqrt();
            max_z = max_z.max(z);
        }
    }
    let mean_abs_defect = sum_abs / n_grid as f64;
    let rms_defect = (sum_sq / n_grid as f64).sqrt();

    // Bonferroni bound for the max studentized defect over the actual grid:
    // valid for arbitrary dependence among pointwise contrasts. With zero
    // empirical score variation there is no sampling law, so no p-value is
    // emitted from deterministic fitted-grid values.
    let max_studentized_p_value = if max_var > 0.0 {
        let normal = Normal::new(0.0, 1.0)
            .map_err(|e| format!("standard normal construction failed: {e}"))?;
        let pointwise: f64 = (2.0 * (1.0 - normal.cdf(max_z))).clamp(0.0, 1.0);
        (n_grid as f64 * pointwise).min(1.0)
    } else {
        f64::NAN
    };

    Ok(CompositionDefectReport {
        n_grid,
        gauge_rotation: 0.0,
        gauge_reflected: false,
        mean_abs_defect,
        rms_defect,
        max_abs_defect: max_abs,
        max_studentized_defect: if max_var > 0.0 { max_z } else { f64::NAN },
        max_studentized_p_value,
        p_value: max_studentized_p_value,
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
    /// O(2) classification (winding/phase/defect) of each adjacent map whose
    /// endpoints are both circle charts — the Fourier-rigidity report
    /// ([`crate::inference::transport_class::classify_circle_transport_fit`]).
    /// Non-circle pairs are omitted; empty when no adjacent pair is circle→circle.
    pub circle_transports: Vec<crate::inference::transport_class::CircleTransportReport>,
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

    // O(2) Fourier-rigidity classification of each adjacent circle→circle map,
    // grid-sampled from the fitted angle map. Additive report; no fitting-path
    // effect.
    let mut circle_transports = Vec::new();
    for k in 0..depth - 1 {
        if let Some(report) = crate::inference::transport_class::classify_circle_transport_fit(
            &adjacent_fits[k],
            topologies[k],
            topologies[k + 1],
            layers[k],
            layers[k + 1],
            DEFAULT_COMPOSITION_GRID,
        ) {
            circle_transports.push(report);
        }
    }

    Ok(TransportLadderReport {
        adjacent,
        two_hop,
        circle_transports,
    })
}

#[cfg(test)]
mod invert_tests {
    use super::*;
    use faer::Side;
    use gam_linalg::faer_ndarray::FaerEigh;
    use ndarray::Array1;

    fn interval(lo: f64, hi: f64) -> ChartTopology {
        ChartTopology::Interval { lo, hi }
    }

    /// The transport wrapper must reuse Gaussian REML's profiled scale and its
    /// exact spectral inverse. In particular, covariance cannot come from the
    /// removed eigenvalue floor/micro-ridge solve or from `RSS / (n - edf)`.
    #[test]
    fn penalized_1d_covariance_and_scale_match_reml_system() {
        let n = 32;
        let design = Array2::from_shape_fn((n, 3), |(row, col)| {
            let x = row as f64 / (n - 1) as f64;
            match col {
                0 => 1.0,
                1 => x,
                2 => x * x,
                _ => unreachable!(),
            }
        });
        let response = Array1::from_shape_fn(n, |row| {
            let x = row as f64 / (n - 1) as f64;
            0.3 + 0.8 * x + 0.2 * (TAU * x).sin()
        });
        let mut penalty = Array2::<f64>::zeros((3, 3));
        penalty[[2, 2]] = 1.0;
        let fit = fit_penalized_1d(&design, &penalty, response.view(), None, 1)
            .expect("certified REML fit");

        let mut penalized_gram = design.t().dot(&design);
        penalized_gram[[2, 2]] += fit.lambda;
        let inverse = fit.covariance.mapv(|value| value / fit.sigma2);
        let identity = penalized_gram.dot(&inverse);
        for row in 0..3 {
            for col in 0..3 {
                let expected = f64::from(u8::from(row == col));
                assert!(
                    (identity[[row, col]] - expected).abs() < 1.0e-9,
                    "penalized inverse mismatch at ({row}, {col}): {}",
                    identity[[row, col]],
                );
            }
        }

        let xtwy = design.t().dot(&response);
        let prss = response.dot(&response) - fit.beta.dot(&xtwy);
        let expected_sigma2 = prss / (n - 2) as f64;
        assert!(
            (fit.sigma2 - expected_sigma2).abs() <= 1.0e-11 * expected_sigma2.abs().max(1.0),
            "REML scale mismatch: fitted {}, expected {expected_sigma2}",
            fit.sigma2,
        );
    }

    #[test]
    fn invert_round_trips_interval_transport() {
        // A strictly increasing nonlinear warp on [0,1] → [0,1] with derivative
        // bounded away from zero: to = (t + 0.25·sin(2πt)/(2π)) normalized, whose
        // h′ = 1 + 0.25·cos(2πt) ∈ [0.75, 1.25] never approaches zero.
        let n = 64;
        let from: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 / (n as f64 - 1.0)));
        let to: Array1<f64> = from.mapv(|t| t + 0.25 * (TAU * t).sin() / TAU);
        let ft = fit_transport_map(
            from.view(),
            to.view(),
            interval(0.0, 1.0),
            interval(0.0, 1.0),
        )
        .expect("fit");
        assert!(
            ft.topology_preserved,
            "monotone warp should preserve topology"
        );

        let probe = Array1::from_iter((1..10).map(|i| i as f64 / 10.0));
        // eval ∘ invert and invert ∘ eval both return identity.
        let fwd = ft.eval(probe.view()).expect("eval");
        let back = ft.invert(fwd.view()).expect("invert");
        for i in 0..probe.len() {
            assert!(
                (back[i] - probe[i]).abs() < 1e-6,
                "round-trip failed: t={} back={}",
                probe[i],
                back[i]
            );
        }
        let re_eval = ft.eval(back.view()).expect("eval");
        for i in 0..fwd.len() {
            assert!((re_eval[i] - fwd[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn invert_round_trips_decreasing_interval_transport() {
        // Orientation-reversing homeomorphism with derivative bounded away from
        // zero: to = 1 - 0.5·from - 0.5·from² on [0,1] (h′ = -0.5 - from ≤ -0.5).
        let n = 64;
        let from: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 / (n as f64 - 1.0)));
        let to: Array1<f64> = from.mapv(|t| 1.0 - 0.5 * t - 0.5 * t * t);
        let ft = fit_transport_map(
            from.view(),
            to.view(),
            interval(0.0, 1.0),
            interval(0.0, 1.0),
        )
        .expect("fit");
        assert!(ft.topology_preserved);
        let probe = Array1::from_iter((1..10).map(|i| i as f64 / 10.0));
        let fwd = ft.eval(probe.view()).expect("eval");
        let back = ft.invert(fwd.view()).expect("invert");
        for i in 0..probe.len() {
            assert!(
                (back[i] - probe[i]).abs() < 1e-6,
                "t={} back={}",
                probe[i],
                back[i]
            );
        }
    }

    #[test]
    fn invert_round_trips_circle_transport() {
        // Degree-1 circle cover: a rotation plus a fold-free wiggle.
        let n = 128;
        let from: Array1<f64> = Array1::from_iter((0..n).map(|i| TAU * i as f64 / n as f64));
        let to: Array1<f64> = from.mapv(|t| wrap_tau(t + 0.3 + 0.2 * t.sin()));
        let ft = fit_transport_map(
            from.view(),
            to.view(),
            ChartTopology::Circle,
            ChartTopology::Circle,
        )
        .expect("fit");
        assert!(ft.topology_preserved, "degree {:?}", ft.degree);

        let probe = Array1::from_iter((0..7).map(|i| TAU * (i as f64 + 0.5) / 7.0));
        let fwd = ft.eval(probe.view()).expect("eval");
        let back = ft.invert(fwd.view()).expect("invert");
        for i in 0..probe.len() {
            // Compare modulo 2π.
            let d = wrap_pi(back[i] - probe[i]).abs();
            assert!(d < 1e-5, "probe={} back={} d={}", probe[i], back[i], d);
        }
    }

    #[test]
    fn invert_rejects_target_outside_interval_image() {
        // Image of `to = 0.5·from` is ~[0, 0.5]; y = 0.9 is outside it.
        let n = 32;
        let from: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 / (n as f64 - 1.0)));
        let to: Array1<f64> = from.mapv(|t| 0.5 * t);
        let ft = fit_transport_map(
            from.view(),
            to.view(),
            interval(0.0, 1.0),
            interval(0.0, 1.0),
        )
        .expect("fit");
        assert!(ft.invert(Array1::from_elem(1, 0.9).view()).is_err());
    }

    /// Build a `FittedTransport` on an interval whose pre-wrap map interpolates
    /// `h` by an unpenalized least-squares spline fit (so a deliberately narrow
    /// fold in `h` survives into the coefficients, unlike a REML fit which would
    /// smooth it away). Fields irrelevant to `eval`/`derivative`/`invert` are
    /// filled with sound placeholders.
    fn fitted_from_target(
        from: ArrayView1<'_, f64>,
        target: ArrayView1<'_, f64>,
        lo: f64,
        hi: f64,
    ) -> FittedTransport {
        let basis = DomainBasis::build(interval(lo, hi), from).expect("basis");
        let design = basis.value_rows(from).expect("design");
        let m = design.ncols();
        // Normal equations XᵀX β = Xᵀy with a tiny ridge for conditioning only.
        let mut xtx = design.t().dot(&design);
        let xty = design.t().dot(&target);
        let diag = (0..m).map(|i| xtx[[i, i]].abs()).fold(1.0_f64, f64::max);
        for i in 0..m {
            xtx[[i, i]] += 1e-10 * diag;
        }
        let (evals, evecs) = xtx.eigh(Side::Lower).expect("eigh");
        let rotated = evecs.t().dot(&xty);
        let mut beta = Array1::<f64>::zeros(m);
        for i in 0..m {
            let d = evals[i].max(f64::MIN_POSITIVE);
            let c = rotated[i] / d;
            for j in 0..m {
                beta[j] += evecs[[j, i]] * c;
            }
        }
        FittedTransport {
            topology_from: interval(lo, hi),
            topology_to: interval(lo, hi),
            degree: None,
            degree_concentration: None,
            rotation_offset: 0.0,
            beta,
            covariance: Array2::<f64>::zeros((m, m)),
            smoothing_lambda: 0.0,
            edf: 0.0,
            noise_variance: 1.0,
            n_obs: from.len(),
            isometry_defect: 0.0,
            isometry_defect_se: 0.0,
            topology_preserved: true,
            min_directional_derivative: 1.0,
            residual_rms: 0.0,
            coefficient_score_influence: Array2::<f64>::zeros((m, from.len())),
            basis,
        }
    }

    /// Reviewer's between-grid fold reproducer: h(t) = (t−0.5)³/3 − (0.4/511)²·t
    /// hides a narrow fold between the 512-point certification-grid samples.
    /// `topology_preserved` (the sampled diagnostic) reads true, yet a dense
    /// grid finds orientation·h′ < 0 — the span-exact certificate that `invert`
    /// now gates on must reject the fit.
    #[test]
    fn invert_rejects_between_grid_fold() {
        let n = 256;
        let from: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 / (n as f64 - 1.0)));
        let eps = 0.4 / 511.0;
        let target: Array1<f64> = from.mapv(|t| (t - 0.5).powi(3) / 3.0 - eps * eps * t);
        let mut ft = fitted_from_target(from.view(), target.view(), 0.0, 1.0);

        // Confirm the fold is genuinely between the 512-pt certification grid:
        // recompute the sampled diagnostic the production fit uses.
        let grid = domain_grid(interval(0.0, 1.0), FOLD_CHECK_GRID);
        let grid_d = ft.derivative(grid.view()).expect("grid deriv");
        let mean = grid_d.iter().sum::<f64>() / grid_d.len() as f64;
        let orientation = if mean < 0.0 { -1.0 } else { 1.0 };
        let min_grid = grid_d
            .iter()
            .map(|&v| orientation * v)
            .fold(f64::INFINITY, f64::min);
        // Dense grid (10× finer) to expose the hidden fold.
        let dense = Array1::from_iter((0..5120).map(|i| i as f64 / 5119.0));
        let dense_d = ft.derivative(dense.view()).expect("dense deriv");
        let min_dense = dense_d
            .iter()
            .map(|&v| orientation * v)
            .fold(f64::INFINITY, f64::min);
        ft.topology_preserved = min_grid > 0.0;
        ft.min_directional_derivative = min_grid;
        assert!(
            min_grid > 0.0 && min_dense < 0.0,
            "fixture must hide a between-grid fold: min on 512-grid={min_grid}, \
             min on dense grid={min_dense}"
        );

        // The span-exact certificate must reject it even though the sampled
        // diagnostic passed.
        let res = ft.invert(Array1::from_elem(1, 0.0).view());
        assert!(
            res.is_err(),
            "between-grid fold must be rejected by the span-exact certificate \
             (topology_preserved={}, min_grid={min_grid}, min_dense={min_dense})",
            ft.topology_preserved
        );
    }

    #[test]
    fn invert_rejects_non_finite_targets() {
        let n = 64;
        let from: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 / (n as f64 - 1.0)));
        let to: Array1<f64> = from.mapv(|t| 0.5 * t);
        let ft = fit_transport_map(
            from.view(),
            to.view(),
            interval(0.0, 1.0),
            interval(0.0, 1.0),
        )
        .expect("fit");
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(
                ft.invert(Array1::from_elem(1, bad).view()).is_err(),
                "non-finite target {bad} must be rejected"
            );
        }
    }

    #[test]
    fn invert_image_tolerance_is_scale_aware() {
        // Image of `to = 1e-8·from` is ~[0, 1e-8]. A target 5% outside it must
        // be rejected, not silently clamped, under the scale-aware tolerance
        // (the old absolute 1e-9 would have accepted it).
        let n = 64;
        let from: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 / (n as f64 - 1.0)));
        let scale = 1.0e-8;
        let to: Array1<f64> = from.mapv(|t| scale * t);
        let ft = fit_transport_map(
            from.view(),
            to.view(),
            interval(0.0, 1.0),
            interval(0.0, 1.0),
        )
        .expect("fit");
        let outside = 1.05e-8;
        assert!(
            ft.invert(Array1::from_elem(1, outside).view()).is_err(),
            "target {outside} is 5% outside the [0, {scale}] image and must be rejected"
        );
        // A target inside the image still round-trips.
        let inside = 0.5e-8;
        let t = ft
            .invert(Array1::from_elem(1, inside).view())
            .expect("invert inside");
        let re = ft.eval(t.view()).expect("eval");
        assert!((re[0] - inside).abs() < 1e-3 * scale);
    }

    #[test]
    fn invert_round_trips_degree_minus_one_circle() {
        // Orientation-reversing degree −1 circle cover: a reflection plus a
        // fold-free wiggle.
        let n = 128;
        let from: Array1<f64> = Array1::from_iter((0..n).map(|i| TAU * i as f64 / n as f64));
        let to: Array1<f64> = from.mapv(|t| wrap_tau(-t + 0.4 + 0.15 * t.sin()));
        let ft = fit_transport_map(
            from.view(),
            to.view(),
            ChartTopology::Circle,
            ChartTopology::Circle,
        )
        .expect("fit");
        assert_eq!(ft.degree, Some(-1), "expected a degree −1 cover");
        assert!(ft.topology_preserved, "degree {:?}", ft.degree);
        let probe = Array1::from_iter((0..7).map(|i| TAU * (i as f64 + 0.5) / 7.0));
        let fwd = ft.eval(probe.view()).expect("eval");
        let back = ft.invert(fwd.view()).expect("invert");
        for i in 0..probe.len() {
            let d = wrap_pi(back[i] - probe[i]).abs();
            assert!(d < 1e-5, "probe={} back={} d={}", probe[i], back[i], d);
        }
    }

    #[test]
    fn invert_round_trips_circle_seam_and_interval_endpoints() {
        // Circle seam: invert a target near 0/2π.
        let n = 128;
        let from: Array1<f64> = Array1::from_iter((0..n).map(|i| TAU * i as f64 / n as f64));
        let to: Array1<f64> = from.mapv(|t| wrap_tau(t + 0.3 + 0.2 * t.sin()));
        let ft = fit_transport_map(
            from.view(),
            to.view(),
            ChartTopology::Circle,
            ChartTopology::Circle,
        )
        .expect("fit");
        assert!(ft.topology_preserved);
        for seam in [1e-9, TAU - 1e-9, 0.0] {
            let t = ft
                .invert(Array1::from_elem(1, seam).view())
                .expect("invert seam");
            let re = ft.eval(t.view()).expect("eval");
            let d = wrap_pi(re[0] - wrap_tau(seam)).abs();
            assert!(d < 1e-6, "seam={seam} re={} d={d}", re[0]);
        }

        // Interval endpoints: invert the image endpoints exactly.
        let m = 64;
        let ifrom: Array1<f64> = Array1::from_iter((0..m).map(|i| i as f64 / (m as f64 - 1.0)));
        let ito: Array1<f64> = ifrom.mapv(|t| t + 0.25 * (TAU * t).sin() / TAU);
        let ift = fit_transport_map(
            ifrom.view(),
            ito.view(),
            interval(0.0, 1.0),
            interval(0.0, 1.0),
        )
        .expect("fit");
        let raw_lo = ift.raw_at(0.0).expect("raw lo");
        let raw_hi = ift.raw_at(1.0).expect("raw hi");
        for &edge in &[raw_lo, raw_hi] {
            let t = ift
                .invert(Array1::from_elem(1, edge).view())
                .expect("invert endpoint");
            assert!(t[0] >= -1e-9 && t[0] <= 1.0 + 1e-9, "endpoint t={}", t[0]);
            let re = ift.eval(t.view()).expect("eval");
            assert!((re[0] - edge).abs() < 1e-6, "edge={edge} re={}", re[0]);
        }
    }

    #[test]
    fn monomial_reconstruction_is_exact_for_quadratic() {
        // The certificate's polynomial reconstruction must be exact on the
        // quadratic pieces of a cubic-spline derivative.
        let coeffs_true = [0.7_f64, -1.3, 2.1]; // 0.7 − 1.3u + 2.1u²
        let values: Vec<f64> = (0..3)
            .map(|i| eval_monomial(&coeffs_true, i as f64))
            .collect();
        let recon = monomial_interpolant_at_integer_nodes(&values);
        for (a, b) in recon.iter().zip(coeffs_true.iter()) {
            assert!((a - b).abs() < 1e-12, "recon {a} vs {b}");
        }
        // Vertex of 2.1u² − 1.3u + 0.7 is at u = 1.3 / (2·2.1).
        let crit = monomial_critical_points(&recon);
        assert_eq!(crit.len(), 1);
        assert!((crit[0] - 1.3 / 4.2).abs() < 1e-12);
    }
}
