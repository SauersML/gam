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
//! estimated under the pair law the caller declares ([`PairLaw`]):
//!
//! * **Deterministic** pairs (each target coordinate is one function of its
//!   source coordinate) get the minimum-curvature interpolant through the
//!   distinct sites. It has no smoothing parameter and no sampling law, so its
//!   bands are exact zeros.
//! * **Stochastic** pairs (the target scatters about the map) get a Gaussian
//!   REML smooth (exact 1-D criterion, no GCV per policy) on a resolution
//!   ladder of uniform spaces. Each rung doubles the segment count, and the
//!   ladder stops at the first rung whose spacing `h` resolves the fitted
//!   penalty, `max span count · h³ ≤ π⁴·λ̂`, or when the space has as many
//!   coefficients as distinct sites.
//!
//! Three questions are answered with evidence:
//!
//! 1. **Topology compatibility** — does `h` preserve the chart topology
//!    (circle→circle degree-±1 covering, i.e. a homeomorphism of `S¹`) or
//!    break it (circle→arcs, folds)? For circle charts the winding **degree**
//!    is read off the pairs themselves: order the rows by source angle, walk
//!    the closed loop once, and sum each target-angle step wrapped into
//!    `(−π, π]`. A closed loop's steps sum to an exact multiple of `2π`, and
//!    that multiple is the degree whenever neighbouring rows move the target
//!    by less than half a turn — no candidate set, no bound on `|d|`. A fold check on a dense
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
//!    charts) is evaluated on a grid that samples every segment of the finer
//!    source-domain spline with `degree + 1` points (the points that fix one
//!    polynomial piece, so the grid resolves the fits without a caller-chosen
//!    density), and is studentized with a joint shared-row
//!    influence sandwich, with resolution bounded below by the fitted maps'
//!    observed approximation error. A Bonferroni max test controls the grid
//!    family under arbitrary pointwise dependence; deterministic fits with
//!    neither score variation nor approximation error emit no p-value.
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
use gam_linalg::faer_ndarray::FaerLblt;
use gam_math::probability::normal_two_sided_probability;
use gam_solve::gaussian_reml::gaussian_reml_closed_form_with_nullspace_dim;
use gam_terms::basis::{
    BasisOptions, Dense, KnotSource, PeriodicBSplineBasisSpec, bspline_derivative_penalty_matrix,
    build_periodic_bspline_basis_1d, create_basis, cyclic_bspline_derivative_penalty_matrix,
    periodic_bspline_first_derivative_nd,
};
use ndarray::{Array1, Array2, ArrayView1, Axis};
use std::f64::consts::{PI, TAU};

/// Cubic splines for every transport smooth.
const TRANSPORT_SPLINE_DEGREE: usize = 3;
/// Second-order function-curvature penalty: the cyclic variant leaves constants
/// unpenalized on a circle; the open variant leaves affine maps unpenalized on
/// an interval — exactly the isometry-adjacent null spaces.
const TRANSPORT_PENALTY_ORDER: usize = 2;
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

    /// The source domain `[lo, hi]`: one turn `[0, 2π]` on a circle.
    fn domain(&self) -> (f64, f64) {
        match *self {
            ChartTopology::Circle => (0.0, TAU),
            ChartTopology::Interval { lo, hi } => (lo, hi),
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

/// How the paired coordinates handed to [`fit_transport_map`] were produced.
///
/// The caller declares it, because the pairs cannot: a smooth deterministic map
/// sampled at `n` points and the same map observed with small noise differ only
/// in what an estimator is allowed to assume about the residual.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairLaw {
    /// Each target coordinate is one deterministic function of its source
    /// coordinate (a held executed transport, a synthetic noise-free map). The
    /// estimator is the minimum-curvature interpolant `argmin ∫h″²` through the
    /// pairs; there is no observation noise, so no sampling covariance.
    Deterministic,
    /// The target coordinates scatter about a smooth map (estimated chart
    /// coordinates, resampled other state). The estimator is the Gaussian-REML
    /// penalized spline on a data-resolved knot spacing.
    Stochastic,
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
/// clamped B-splines on an interval or, seamed, on one turn of a circle. All
/// reuse the existing basis constructors directly (no string DSL round-trip).
#[derive(Debug, Clone)]
enum DomainBasis {
    Periodic(PeriodicBSplineBasisSpec),
    /// Clamped B-splines on `knots`. `seamed` marks a circle domain `[0, 2π]`
    /// whose coefficients the caller constrains to a `C^{degree−1}` seam, so an
    /// evaluation point is wrapped onto the turn rather than clamped.
    Open {
        knots: Array1<f64>,
        degree: usize,
        seamed: bool,
    },
}

impl DomainBasis {
    /// The uniform cubic space with `segments` equal polynomial pieces over the
    /// source domain: periodic cardinal B-splines on a circle (`segments ≥
    /// degree + 1` periodized shifts), clamped B-splines with `segments − 1`
    /// equally spaced interior knots on an interval.
    fn uniform(topology: ChartTopology, segments: usize) -> Self {
        match topology {
            ChartTopology::Circle => DomainBasis::Periodic(PeriodicBSplineBasisSpec {
                degree: TRANSPORT_SPLINE_DEGREE,
                num_basis: segments,
                period: TAU,
                origin: 0.0,
                penalty_order: TRANSPORT_PENALTY_ORDER,
            }),
            ChartTopology::Interval { lo, hi } => {
                let interior = (1..segments).map(|k| lo + (hi - lo) * k as f64 / segments as f64);
                DomainBasis::Open {
                    knots: clamped_knots(lo, hi, interior),
                    degree: TRANSPORT_SPLINE_DEGREE,
                    seamed: false,
                }
            }
        }
    }

    /// The cubic space with one knot at every distinct interior data site
    /// (`sites` sorted, distinct, already projected onto the domain). It holds
    /// the minimum-curvature interpolant through those sites: the natural cubic
    /// spline on an interval, and — under the seam constraints — the periodic
    /// cubic spline on a circle.
    fn at_sites(topology: ChartTopology, sites: &[f64]) -> Self {
        let (lo, hi) = topology.domain();
        let interior = sites.iter().copied().filter(|&site| site > lo && site < hi);
        DomainBasis::Open {
            knots: clamped_knots(lo, hi, interior),
            degree: TRANSPORT_SPLINE_DEGREE,
            seamed: matches!(topology, ChartTopology::Circle),
        }
    }

    fn num_basis(&self) -> usize {
        match self {
            DomainBasis::Periodic(spec) => spec.num_basis,
            DomainBasis::Open { knots, degree, .. } => knots.len() - degree - 1,
        }
    }

    /// Polynomial pieces of the spline over its domain: `num_basis` uniform
    /// segments on the periodic basis, `num_basis − degree` knot spans on the
    /// open basis.
    fn num_segments(&self) -> usize {
        match self {
            DomainBasis::Periodic(spec) => spec.num_basis,
            DomainBasis::Open { degree, .. } => self.num_basis() - degree,
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
            DomainBasis::Open { knots, degree, .. } => {
                bspline_derivative_penalty_matrix(knots.view(), *degree, TRANSPORT_PENALTY_ORDER)
                    .map_err(|e| format!("open transport roughness failed: {e}"))
            }
        }
    }

    /// Clamp/wrap an evaluation point into the basis domain.
    fn project(&self, t: f64) -> f64 {
        match self {
            DomainBasis::Periodic(_) | DomainBasis::Open { seamed: true, .. } => wrap_tau(t),
            DomainBasis::Open { knots, degree, .. } => {
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
            DomainBasis::Open { knots, degree, .. } => {
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
            DomainBasis::Open { knots, degree, .. } => {
                let lo = knots[*degree];
                let hi = knots[knots.len() - 1 - degree];
                let mut breaks: Vec<f64> = Vec::with_capacity(knots.len());
                for &k in knots.iter() {
                    if k > lo + 0.0 && k < hi {
                        breaks.push(k);
                    }
                }
                breaks.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                breaks.dedup();
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
            DomainBasis::Open { knots, degree, .. } => {
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

/// Clamped knot vector on `[lo, hi]`: each end repeated `degree + 1` times
/// around the supplied interior knots.
fn clamped_knots(lo: f64, hi: f64, interior: impl Iterator<Item = f64>) -> Array1<f64> {
    let ends = TRANSPORT_SPLINE_DEGREE + 1;
    let mut knots: Vec<f64> = std::iter::repeat_n(lo, ends).collect();
    knots.extend(interior);
    knots.extend(std::iter::repeat_n(hi, ends));
    Array1::from_vec(knots)
}

/// Rows `(left, right)` with `left·β = g^{(k)}(lo)` and `right·β = g^{(k)}(hi)`
/// for `k = 0, …, degree − 1`, for the clamped spline `g = Σ βⱼBⱼ` on `knots`.
///
/// The `k`-th derivative of a degree-`p` spline is the degree-`(p − k)` spline
/// on the inner knots whose coefficients follow by de Boor's differencing,
/// `c⁽ᵏ⁾ᵢ = (p − k + 1)(c⁽ᵏ⁻¹⁾ᵢ₊₁ − c⁽ᵏ⁻¹⁾ᵢ)/(t_{i+p+1} − t_{i+k})`, and a clamped
/// spline takes its first coefficient at `lo` and its last at `hi`. The jets are
/// therefore exact linear functionals of `β`, read without evaluating the basis
/// at an endpoint.
fn clamped_end_jets(knots: ArrayView1<'_, f64>, degree: usize) -> Vec<(Array1<f64>, Array1<f64>)> {
    let m = knots.len() - degree - 1;
    let mut coefficients: Vec<Array1<f64>> = (0..m)
        .map(|i| {
            let mut unit = Array1::<f64>::zeros(m);
            unit[i] = 1.0;
            unit
        })
        .collect();
    let mut jets = Vec::with_capacity(degree);
    jets.push((coefficients[0].clone(), coefficients[m - 1].clone()));
    for order in 1..degree {
        let factor = (degree - order + 1) as f64;
        coefficients = (0..coefficients.len() - 1)
            .map(|i| {
                let width = knots[i + degree + 1] - knots[i + order];
                (&coefficients[i + 1] - &coefficients[i]) * (factor / width)
            })
            .collect();
        jets.push((coefficients[0].clone(), coefficients[coefficients.len() - 1].clone()));
    }
    jets
}

/// Project source coordinates onto the domain: wrapped onto one turn on a
/// circle, clamped into `[lo, hi]` on an interval.
fn project_onto_domain(topology: ChartTopology, coords: ArrayView1<'_, f64>) -> Vec<f64> {
    let (lo, hi) = topology.domain();
    coords
        .iter()
        .map(|&t| match topology {
            ChartTopology::Circle => wrap_tau(t),
            ChartTopology::Interval { .. } => t.clamp(lo, hi),
        })
        .collect()
}

/// The minimum-curvature interpolant `argmin ∫g″²` through deterministic pairs.
///
/// Among all functions through the pairs, the one with least `∫g″²` is the
/// natural cubic spline on an interval and the periodic cubic spline on a
/// circle, with knots at the distinct data sites (Holladay). Both live in the
/// clamped cubic space of [`DomainBasis::at_sites`] — on a circle once its
/// coefficients carry the `C²` seam `g^{(k)}(0) = g^{(k)}(2π)`, `k < 3` — whose
/// exact `∫g″²` Gram `S` is the roughness. The interpolant is the solution of
/// the KKT system
///
/// ```text
///     [ S  Aᵀ ] [β]   [0]
///     [ A  0  ] [μ] = [b]
/// ```
///
/// with `A` the value rows at the sites stacked on the seam rows, and `b` the
/// responses stacked on zeros. `S` is positive definite on `ker A`: `S`
/// annihilates only affine maps, and an affine map through two sites, or a
/// periodic one through one, is zero. The system is then nonsingular with
/// inertia `(m, p, 0)`, which the Bunch–Kaufman factor certifies before its
/// solution is used. Scaling `S` by its largest entry and each constraint row
/// by its own changes neither the argmin nor the constraint set; it only
/// equilibrates the factor.
///
/// A deterministic map has no observation noise, so the fit carries no
/// sampling law: covariance, dispersion and row influence are exact zeros,
/// `λ = 0` is the interpolation limit of the smoothing family, and the
/// effective degrees of freedom are the distinct sites it reproduces. Pairs that
/// assign two responses to one source coordinate are not a function and are
/// refused.
fn fit_interpolant(
    topology: ChartTopology,
    coords: ArrayView1<'_, f64>,
    response: ArrayView1<'_, f64>,
) -> Result<(DomainBasis, Penalized1dFit), String> {
    let n = coords.len();
    let projected = project_onto_domain(topology, coords);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| projected[a].total_cmp(&projected[b]));
    let mut sites: Vec<f64> = Vec::with_capacity(n);
    let mut values: Vec<f64> = Vec::with_capacity(n);
    for &row in &order {
        match (sites.last(), values.last()) {
            (Some(&site), Some(&value)) if site == projected[row] => {
                if response[row] != value {
                    return Err(format!(
                        "deterministic transport pairs are not a function of the source \
                         coordinate: t = {site} carries responses {value} and {}; declare the \
                         pairs stochastic",
                        response[row]
                    ));
                }
            }
            _ => {
                sites.push(projected[row]);
                values.push(response[row]);
            }
        }
    }
    // The penalty's null space after the constraints the domain imposes:
    // affine maps on an interval, constants under a circle's seam.
    let pinned_null_dim = match topology {
        ChartTopology::Circle => 1,
        ChartTopology::Interval { .. } => TRANSPORT_PENALTY_ORDER,
    };
    if sites.len() < pinned_null_dim {
        return Err(format!(
            "the minimum-curvature interpolant needs {pinned_null_dim} distinct source \
             coordinates to pin the unpenalized maps, got {}",
            sites.len()
        ));
    }

    let basis = DomainBasis::at_sites(topology, &sites);
    let DomainBasis::Open {
        knots,
        degree,
        seamed,
    } = &basis
    else {
        return Err("the interpolating transport space must be clamped B-splines".to_string());
    };
    let m = basis.num_basis();
    let site_rows = basis.value_rows(ArrayView1::from(sites.as_slice()))?;
    let mut constraints: Vec<(Array1<f64>, f64)> = site_rows
        .outer_iter()
        .zip(values.iter())
        .map(|(row, &value)| (row.to_owned(), value))
        .collect();
    if *seamed {
        for (left, right) in clamped_end_jets(knots.view(), *degree) {
            constraints.push((&left - &right, 0.0));
        }
    }
    let p = constraints.len();

    let penalty = basis.penalty()?;
    let penalty_scale = penalty.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
    if !(penalty_scale.is_finite() && penalty_scale > 0.0) {
        return Err(format!(
            "the interpolating transport roughness Gram has no finite positive scale: \
             {penalty_scale}"
        ));
    }
    let mut kkt = faer::Mat::<f64>::zeros(m + p, m + p);
    let mut rhs = faer::Mat::<f64>::zeros(m + p, 1);
    for i in 0..m {
        for j in 0..m {
            kkt[(i, j)] = penalty[[i, j]] / penalty_scale;
        }
    }
    for (r, (row, value)) in constraints.iter().enumerate() {
        let row_scale = row.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs()));
        if !(row_scale.is_finite() && row_scale > 0.0) {
            return Err(format!(
                "interpolation constraint {r} has no finite nonzero entry: scale {row_scale}"
            ));
        }
        for j in 0..m {
            kkt[(m + r, j)] = row[j] / row_scale;
            kkt[(j, m + r)] = row[j] / row_scale;
        }
        rhs[(m + r, 0)] = value / row_scale;
    }
    let factor = FaerLblt::new(kkt.as_ref(), faer::Side::Lower);
    let inertia = factor.inertia();
    if inertia.positive != m || inertia.negative != p || inertia.zero != 0 {
        return Err(format!(
            "the minimum-curvature interpolant is not unique: the KKT system of {m} \
             coefficients and {p} constraints has inertia (+{}, −{}, 0×{}), not (+{m}, −{p}, 0×0)",
            inertia.positive, inertia.negative, inertia.zero
        ));
    }
    let solution = factor.solve(rhs.as_ref());
    let beta = Array1::from_iter((0..m).map(|j| solution[(j, 0)]));
    if beta.iter().any(|v| !v.is_finite()) {
        return Err("the minimum-curvature interpolant has non-finite coefficients".to_string());
    }

    let fitted = basis.value_rows(coords)?.dot(&beta);
    let rss: f64 = (0..n).map(|i| (response[i] - fitted[i]).powi(2)).sum();
    let fit = Penalized1dFit {
        beta,
        covariance: Array2::zeros((m, m)),
        lambda: 0.0,
        edf: sites.len() as f64,
        sigma2: 0.0,
        residual_rms: (rss / n as f64).sqrt(),
        coefficient_score_influence: Array2::zeros((m, n)),
    };
    Ok((basis, fit))
}

/// Gaussian-REML smooth of stochastic pairs on a data-resolved knot spacing.
///
/// The spaces are the uniform cubic splines of [`DomainBasis::uniform`], from the
/// coarsest (`m = degree + 1` coefficients) through halvings of the knot spacing
/// `h`; they nest, and each carries the exact `∫g″²` Gram. A penalized spline
/// with criterion `‖y − Xβ‖² + λ∫g″²` acts on data of local density `ρ` rows
/// per unit length as the equivalent kernel of bandwidth `b = (λ/ρ)^{1/4}`
/// (Silverman), whose transfer `1/(1 + (bω)⁴)` passes frequencies below `1/b`.
/// The knot grid resolves every frequency below its Nyquist `π/h`, so the space
/// resolves the smoother once `π/h ≥ 1/b` in every segment. With `c` rows in a
/// segment of width `h`, `ρ = c/h` and the condition is `c·h³ ≤ π⁴·λ̂`, taken at
/// the busiest segment. Refinement also stops once the space has as many
/// coefficients as distinct source coordinates: the hat matrix has rank at most
/// that count, so a finer space adds no resolution the data can use.
fn fit_resolved_smooth(
    topology: ChartTopology,
    coords: ArrayView1<'_, f64>,
    response: ArrayView1<'_, f64>,
) -> Result<(DomainBasis, Penalized1dFit), String> {
    let (lo, hi) = topology.domain();
    let projected = project_onto_domain(topology, coords);
    let distinct = {
        let mut sorted = projected.clone();
        sorted.sort_by(f64::total_cmp);
        sorted.dedup();
        sorted.len()
    };
    let mut segments = match topology {
        ChartTopology::Circle => TRANSPORT_SPLINE_DEGREE + 1,
        ChartTopology::Interval { .. } => 1,
    };
    loop {
        let basis = DomainBasis::uniform(topology, segments);
        let design = basis.value_rows(coords)?;
        let penalty = basis.penalty()?;
        let fit = fit_penalized_1d(&design, &penalty, response, None, basis.penalty_rank())?;
        let width = (hi - lo) / segments as f64;
        let mut counts = vec![0usize; segments];
        for &t in &projected {
            counts[(((t - lo) / width) as usize).min(segments - 1)] += 1;
        }
        let busiest = counts.iter().copied().max().unwrap_or(0) as f64;
        if basis.num_basis() >= distinct || busiest * width.powi(3) <= PI.powi(4) * fit.lambda {
            return Ok((basis, fit));
        }
        segments *= 2;
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
    let reml = gaussian_reml_closed_form_with_nullspace_dim(
        design.view(),
        response.view(),
        penalty.view(),
        Some(nullspace_dim),
        weights.as_ref().map(|w| w.view()),
        None,
    )
    .map_err(|error| format!("penalized 1-D Gaussian REML failed: {error}"))?;

    // The exact penalized inverse `(XᵀWX + λS)⁻¹` in the cache's spectral modes:
    // no eigenvalue flooring and no representative-selecting ridge that would
    // change the REML objective.
    let lambda = reml.lambda;
    let a_inv = reml
        .cache
        .inverse_hessian(lambda)
        .map_err(|error| format!("penalized 1-D REML inverse Hessian failed: {error}"))?;
    if a_inv.dim() != (m, m) {
        return Err(format!(
            "penalized 1-D REML inverse Hessian is {}x{}, expected {m}x{m}",
            a_inv.nrows(),
            a_inv.ncols(),
        ));
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
    // The weights were validated finite and positive, so `sum_w > 0`.
    let residual_rms = (rss / sum_w).sqrt();
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
    /// The law the pairs were declared under, which chose the estimator.
    pub pair_law: PairLaw,
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
    /// Rounding band of `isometry_defect` as evaluated: the derivative sums, the
    /// speed gaps and their mean. A defect inside it cannot be told from zero.
    isometry_defect_band: f64,
    /// Whether `h` is compatible with both chart topologies: a degree-±1
    /// circle cover without folds, or a fold-free interval homeomorphism.
    pub topology_preserved: bool,
    /// Exact minimum of `orientation·h′(t)` over the source domain, taken per
    /// knot span where `h′` is a known-degree polynomial; positive ⇔ no folds.
    pub min_directional_derivative: f64,
    /// RMS of the response residuals at the fitted map.
    pub residual_rms: f64,
    basis: DomainBasis,
    coefficient_score_influence: Array2<f64>,
}

impl FittedTransport {
    /// Rounding band of [`Self::isometry_defect`].
    pub(crate) fn isometry_defect_band(&self) -> f64 {
        self.isometry_defect_band
    }

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

    /// Standard error of the O(2) resultant `R_s = |mean_i e^{i(h(t_i) − s·t_i)}|`
    /// at the supplied source coordinates for winding `s = ±1`, by the delta
    /// method through the coefficient covariance. With `ψ_i = h(t_i) − s·t_i` and
    /// `φ_s` the resultant's argument, `∂R_s/∂h(t_i) = sin(φ_s − ψ_i)/n`, and `h`
    /// depends on the coefficients through the basis value rows.
    pub(crate) fn circle_resultant_se(
        &self,
        t: ArrayView1<'_, f64>,
        winding: i8,
    ) -> Result<f64, String> {
        let n = t.len();
        if n == 0 {
            return Err("circle resultant standard error needs at least one coordinate".to_string());
        }
        let rows = self.basis.value_rows(t)?;
        let values = self.eval(t)?;
        let sign = f64::from(winding);
        let (mut cos_sum, mut sin_sum) = (0.0_f64, 0.0_f64);
        for i in 0..n {
            let psi = values[i] - sign * t[i];
            cos_sum += psi.cos();
            sin_sum += psi.sin();
        }
        let phase = sin_sum.atan2(cos_sum);
        let mut gradient = Array1::<f64>::zeros(rows.ncols());
        for i in 0..n {
            let weight = (phase - (values[i] - sign * t[i])).sin() / n as f64;
            for j in 0..rows.ncols() {
                gradient[j] += weight * rows[[i, j]];
            }
        }
        Ok(gradient.dot(&self.covariance.dot(&gradient)).max(0.0).sqrt())
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
    /// This reads the same exact minimum the fit publishes as
    /// [`FittedTransport::min_directional_derivative`], and from which it derives
    /// [`FittedTransport::topology_preserved`]. On each knot span `h′` is a single polynomial of
    /// degree `d = `[`DomainBasis::derivative_poly_degree`]` (cubic spline ⇒
    /// quadratic). A degree-`d` polynomial is determined by `d + 1` samples, so
    /// per span we evaluate `h′` at the midpoints of `d + 1` equal cells, reconstruct
    /// the known-degree polynomial in the Lagrange basis, locate its interior critical
    /// points in closed form, and require `orientation·h′ > 0` at the span
    /// endpoints **and** every interior critical point. Both bases are exact
    /// piecewise polynomials on these spans: the open basis on the distinct knots
    /// its derivative rows are evaluated with, and the periodic basis on the
    /// cardinal lattice `k·2π/m`, whose `m` periodized shifts cover every integer
    /// shift once, so the rows are a partition of unity and the row normalization
    /// divides by one.
    fn certify_strict_monotonicity(&self) -> Result<f64, String> {
        let (orientation, minimum, argmin) = self.exact_minimum_oriented_derivative()?;
        if !(minimum > 0.0) {
            return Err(format!(
                "transport map is not strictly monotone: orientation·h′ = {minimum} ≤ 0 at \
                 t = {argmin}"
            ));
        }
        Ok(orientation)
    }

    /// The exact minimum of `orientation·h′` over the source domain, with the
    /// orientation (+1 increasing, −1 decreasing, read off the pre-wrap map's
    /// endpoints) and the coordinate where the minimum sits. Each knot span holds
    /// one known-degree polynomial piece, reconstructed from `d + 1` samples, so
    /// the span minimum is at a span endpoint or an interior critical point. Folds
    /// are reported through a non-positive minimum, not an error. An error means
    /// `h′` was not finite at a candidate, or its pieces are above cubic, where
    /// `monomial_critical_points` has no closed form.
    fn exact_minimum_oriented_derivative(&self) -> Result<(f64, f64, f64), String> {
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
        let mut minimum = f64::INFINITY;
        let mut argmin = lo;
        // Restrict the breakpoints to the active domain (the periodic segment
        // grid already coincides with `[lo, hi]`).
        for window in breaks.windows(2) {
            let (a, b) = (window[0], window[1]);
            if !(b > a) {
                continue;
            }
            let span = b - a;
            // Reconstruction abscissae: the midpoints of `deg + 1` equal cells of the
            // span. Each sits strictly inside, so no node lands on the knot where two
            // pieces meet and the open-basis derivative can be one-sided.
            let n_nodes = deg + 1;
            let step = span / n_nodes as f64;
            let nodes: Vec<f64> = (0..n_nodes)
                .map(|i| a + (i as f64 + 0.5) * step)
                .collect();
            let values = self.oriented_derivative_at(&nodes, orientation)?;

            // Polynomial in the local coordinate u = (t - nodes[0]) / step, which puts
            // the nodes on 0..=deg. Expand the exact Lagrange interpolant into monomial
            // coefficients for the closed-form critical-point search. This is algebraic
            // reconstruction of a known-degree spline piece, not a numerical
            // derivative approximation.
            let coeffs = monomial_interpolant_at_integer_nodes(&values);

            // The piece's minimum over the closed span is at an endpoint or at an
            // interior critical point.
            let mut candidates: Vec<(f64, f64)> = [a, b]
                .iter()
                .map(|&edge| (edge, eval_monomial(&coeffs, (edge - nodes[0]) / step)))
                .collect();
            for u_crit in monomial_critical_points(&coeffs)? {
                let t_crit = nodes[0] + u_crit * step;
                if t_crit > a && t_crit < b {
                    candidates.push((t_crit, eval_monomial(&coeffs, u_crit)));
                }
            }
            for (t_candidate, value) in candidates {
                if !value.is_finite() {
                    return Err(format!(
                        "transport monotonicity certificate met a non-finite orientation·h′ at \
                         t = {t_candidate}"
                    ));
                }
                if value < minimum {
                    minimum = value;
                    argmin = t_candidate;
                }
            }
        }
        if !minimum.is_finite() {
            return Err(
                "transport monotonicity certificate found no knot span to certify".to_string(),
            );
        }
        Ok((orientation, minimum, argmin))
    }

    /// Invert the transport: for each target-chart coordinate `y`, return the
    /// source-chart coordinate `t` with `eval([t]) == y`.
    ///
    /// Requires a strictly monotone, fold-free map (a degree-±1 cover for
    /// circle charts, a homeomorphism for intervals), so the inverse is
    /// single-valued; otherwise this errors rather than picking an arbitrary
    /// branch. Monotonicity is established with `Self::certify_strict_monotonicity`
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
        // An endpoint image is `slope·t + offset + Σ_j b_j(t)·β_j` over the `d + 1` live
        // B-splines. Each basis value costs at most `4d + 1` roundings (the `d`-level
        // recursion and the partition-of-unity normalization), the weighted sum two per
        // term, and the affine part three, so an endpoint rounds by at most `γ_{6d+6}`
        // of `|slope·t| + |offset| + Σ_j |b_j·β_j|`. A target beyond the larger
        // endpoint band is outside the fitted image.
        let endpoint_band = |t: f64| -> Result<f64, String> {
            let row = self.basis.value_rows(Array1::from_elem(1, t).view())?;
            let smooth_mass: f64 = row
                .row(0)
                .iter()
                .zip(self.beta.iter())
                .map(|(value, coefficient)| (value * coefficient).abs())
                .sum();
            Ok(gam_linalg::roundoff::accumulation_growth(6 * TRANSPORT_SPLINE_DEGREE + 6)
                * ((self.linear_slope() * t).abs() + self.rotation_offset.abs() + smooth_mass))
        };
        let tol = endpoint_band(lo)?.max(endpoint_band(hi)?);

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
            // Monotone bisection on the pre-wrap map over [lo, hi], down to the
            // source coordinate's relative precision `ε·max(|lo|, |hi|)`. Float
            // spacing inside the bracket is at most that, so each halving narrows it
            // and about `log₂(2/ε)` passes suffice; the adjacency test only guards a
            // collapsed midpoint.
            let (mut a, mut b) = (lo, hi);
            let width_floor = f64::EPSILON * hi.abs().max(lo.abs());
            while (b - a) > width_floor {
                let mid = 0.5 * (a + b);
                if mid <= a || mid >= b {
                    break;
                }
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
            pair_law: self.pair_law,
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
    /// The law the pairs were declared under.
    pub pair_law: PairLaw,
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
    /// Fold diagnostic: exact minimum of orientation·h′ over the source domain.
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
    pair_law: PairLaw,
) -> Result<FittedTransport, String> {
    let n = coords_from.len();
    if coords_to.len() != n {
        return Err(format!(
            "layer transport coordinate lengths disagree: {} vs {}",
            n,
            coords_to.len()
        ));
    }
    // Identifiability is the REML fit's own typed refusal: its residual degrees
    // of freedom `n − nullity` must be positive, where the nullity is the
    // penalty null space (constants on a circle, affine maps on an interval).
    // The only precondition this entry owns is a non-empty pairing, which the
    // empirical isometry-defect average divides by.
    if n == 0 {
        return Err("layer transport needs paired observations, got none".to_string());
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
            // Winding degree read off the pairs: order the rows by source
            // angle and walk the closed loop once, summing each target-angle
            // step wrapped into (−π, π]. The steps of a closed loop sum to an
            // exact multiple of 2π, so rounding only absorbs roundoff, and the
            // multiple is the degree whenever neighbouring rows move the target
            // by less than half a turn (the sampling condition the unwrapped
            // response below already needs). No candidate set bounds |d|.
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| wrap_tau(coords_from[a]).total_cmp(&wrap_tau(coords_from[b])));
            let mut turn = 0.0_f64;
            for k in 0..n {
                turn += wrap_pi(coords_to[order[(k + 1) % n]] - coords_to[order[k]]);
            }
            let degree = (turn / TAU).round() as i32;
            let residual: Vec<f64> = (0..n)
                .map(|i| coords_to[i] - f64::from(degree) * coords_from[i])
                .collect();
            let concentration = resultant_length(&residual);
            let mu = circular_mean(&residual);
            let response = Array1::from_iter(residual.iter().map(|&r| wrap_pi(r - mu)));
            (Some(degree), Some(concentration), mu, response)
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

    // --- residual map on the source chart, under the declared pair law -------
    let (basis, fit) = match pair_law {
        PairLaw::Deterministic => fit_interpolant(topology_from, coords_from, response.view())?,
        PairLaw::Stochastic => fit_resolved_smooth(topology_from, coords_from, response.view())?,
    };

    // --- isometry defect under the empirical density -------------------------
    let slope = degree.map_or(0.0, f64::from);
    let deriv_rows = basis.derivative_rows(coords_from)?;
    let deriv = deriv_rows.dot(&fit.beta).mapv(|v| v + slope);
    let m = basis.num_basis();
    // Each `h′(tᵢ) = Σ_j D_ij·β_j + slope` sums `m` products and one more term, so it is
    // off by at most `γ_{m+1}·(Σ_j |D_ij·β_j| + |slope|)`. The gap `|h′| − 1` adds
    // `γ_1·(|h′| + 1)`, and the mean of squared gaps carries
    // `Σ_i (2|gap|·δgap + δgap²)/n` plus `γ_{n+1}` of itself.
    let absolute_terms = deriv_rows.mapv(f64::abs).dot(&fit.beta.mapv(f64::abs));
    let derivative_growth = gam_linalg::roundoff::accumulation_growth(m + 1);
    let gap_growth = gam_linalg::roundoff::accumulation_growth(1);
    let mut defect = 0.0_f64;
    let mut defect_band = 0.0_f64;
    let mut grad = Array1::<f64>::zeros(m);
    for i in 0..n {
        let speed = deriv[i].abs();
        let gap = speed - 1.0;
        defect += gap * gap;
        let gap_band = derivative_growth * (absolute_terms[i] + slope.abs())
            + gap_growth * (speed + 1.0);
        defect_band += 2.0 * gap.abs() * gap_band + gap_band * gap_band;
        let sgn = if deriv[i] >= 0.0 { 1.0 } else { -1.0 };
        for j in 0..m {
            grad[j] += 2.0 * gap * sgn * deriv_rows[[i, j]];
        }
    }
    defect /= n as f64;
    let isometry_defect_band =
        defect_band / n as f64 + gam_linalg::roundoff::accumulation_growth(n + 1) * defect;
    grad.mapv_inplace(|v| v / n as f64);
    let isometry_defect_se = grad.dot(&fit.covariance.dot(&grad)).max(0.0).sqrt();

    // --- fold / orientation certificate --------------------------------------
    // `h′` is a known-degree polynomial on each knot span, so its minimum over
    // the source domain is exact. The published fold diagnostic and the
    // topology verdict both read that minimum; a sampled grid can pass over a
    // fold between its samples.
    let mut fitted = FittedTransport {
        topology_from,
        topology_to,
        pair_law,
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
        isometry_defect_band,
        topology_preserved: false,
        min_directional_derivative: f64::NAN,
        residual_rms: fit.residual_rms,
        basis,
        coefficient_score_influence: fit.coefficient_score_influence,
    };
    let min_directional_derivative = fitted.exact_minimum_oriented_derivative()?.1;
    fitted.min_directional_derivative = min_directional_derivative;
    fitted.topology_preserved = match (topology_from, topology_to) {
        (ChartTopology::Circle, ChartTopology::Circle) => {
            matches!(degree, Some(1) | Some(-1)) && min_directional_derivative > 0.0
        }
        (ChartTopology::Interval { .. }, ChartTopology::Interval { .. }) => {
            min_directional_derivative > 0.0
        }
        _ => false,
    };
    Ok(fitted)
}

/// Estimate the transport map between two layers and package the evidence.
pub fn fit_layer_transport(
    layer_from: usize,
    layer_to: usize,
    coords_from: ArrayView1<'_, f64>,
    coords_to: ArrayView1<'_, f64>,
    topology_from: ChartTopology,
    topology_to: ChartTopology,
    pair_law: PairLaw,
) -> Result<LayerTransportReport, String> {
    Ok(
        fit_transport_map(coords_from, coords_to, topology_from, topology_to, pair_law)?
            .report(layer_from, layer_to),
    )
}

/// Composition-law test report for one triple `(h_ab, h_bc, h_ac)`.
#[derive(Debug, Clone)]
pub struct CompositionDefectReport {
    /// Grid points tested: `degree + 1` per segment of the finer source-domain
    /// spline.
    pub n_grid: usize,
    /// Always zero: no post-hoc target rotation is fitted.
    pub gauge_rotation: f64,
    /// Always false: no post-hoc target reflection is fitted.
    pub gauge_reflected: bool,
    pub mean_abs_defect: f64,
    pub rms_defect: f64,
    pub max_abs_defect: f64,
    /// `max_t |d(t)| / se(t)`, with `se(t)` the joint influence-sandwich
    /// standard error of the defect at `t` — the sampling scale of the fitted
    /// curves themselves, not of a single observation (#3512).
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
/// polynomial, in the local `u` coordinate, in closed form for derivatives of
/// degree ≤ 2. For the cubic transport splines the polynomial is quadratic and
/// this is the single vertex. A higher-degree derivative has no closed-form root
/// list here and is refused rather than scanned on a grid.
fn monomial_critical_points(coeffs: &[f64]) -> Result<Vec<f64>, String> {
    // Derivative coefficients: d/du Σ c_k u^k = Σ k·c_k u^{k−1}.
    let n = coeffs.len();
    if n <= 1 {
        return Ok(Vec::new());
    }
    let deriv: Vec<f64> = (1..n).map(|k| k as f64 * coeffs[k]).collect();
    // deriv is ascending of length n−1 (degree n−2).
    match deriv.len() {
        0 => Ok(Vec::new()),
        1 => Ok(Vec::new()), // constant derivative: no critical point
        2 => {
            // Linear b + a·u = 0 (a = deriv[1]).
            let (b, a) = (deriv[0], deriv[1]);
            if a == 0.0 {
                Ok(Vec::new())
            } else {
                Ok(vec![-b / a])
            }
        }
        3 => {
            // Quadratic c + b·u + a·u² = 0.
            let (c, b, a) = (deriv[0], deriv[1], deriv[2]);
            if a == 0.0 {
                if b == 0.0 {
                    Ok(Vec::new())
                } else {
                    Ok(vec![-c / b])
                }
            } else {
                let disc = b * b - 4.0 * a * c;
                if disc < 0.0 {
                    Ok(Vec::new())
                } else {
                    let s = disc.sqrt();
                    Ok(vec![(-b + s) / (2.0 * a), (-b - s) / (2.0 * a)])
                }
            }
        }
        len => Err(format!(
            "transport monotonicity certificate has no closed-form critical points for a \
             degree-{} derivative piece",
            len - 1
        )),
    }
}

/// Evaluation grid that fixes every polynomial piece of `basis`: the midpoints
/// of `degree + 1` equal cells in each span between its breakpoints. A cubic
/// piece is determined by that many values, and a midpoint never lands on the
/// knot where two pieces meet.
fn piece_grid(basis: &DomainBasis) -> Array1<f64> {
    let cells = TRANSPORT_SPLINE_DEGREE + 1;
    let breaks = basis.derivative_breakpoints();
    let mut grid = Vec::with_capacity(cells * breaks.len());
    for window in breaks.windows(2) {
        let (a, b) = (window[0], window[1]);
        let step = (b - a) / cells as f64;
        grid.extend((0..cells).map(|i| a + (i as f64 + 0.5) * step));
    }
    Array1::from_vec(grid)
}

/// Test the composition law `h_ac ≟ h_bc ∘ h_ab` on a grid derived from the fits.
///
/// The defect `d(t) = h_ac(t) ⊖ (h_bc ∘ h_ab)(t)` (circular difference on
/// circle targets) is computed directly in the common target chart: no gauge is
/// selected after seeing the defect. Pointwise uncertainty is assembled from
/// the combined observation-level influence of all three maps, retaining their
/// shared-row covariance, and is the whole studentizer: the defect contrasts
/// fitted curves, whose standard error is `O(σ√(edf/n))`, so no observation-
/// noise scale floors it (#3512). The grid is tested by a Bonferroni max
/// statistic.
///
/// The grid samples every polynomial piece of the finer of the two
/// source-domain splines (`h_ab`, `h_ac`) at `degree + 1` cell midpoints, the
/// number that fixes one piece. A denser grid adds only dependent contrasts that inflate
/// the Bonferroni family; a sparser one cannot see a piece.
pub fn composition_defect(
    h_ab: &FittedTransport,
    h_bc: &FittedTransport,
    h_ac: &FittedTransport,
) -> Result<CompositionDefectReport, String> {
    if h_ab.topology_from != h_ac.topology_from
        || h_ab.topology_to != h_bc.topology_from
        || h_bc.topology_to != h_ac.topology_to
    {
        return Err("composition defect requires chart-compatible transports: \
             h_ab: A→B, h_bc: B→C, h_ac: A→C"
            .to_string());
    }
    if h_ab.n_obs != h_bc.n_obs || h_ab.n_obs != h_ac.n_obs {
        return Err(format!(
            "composition defect requires maps fitted on the same rows; got n_ab={}, n_bc={}, n_ac={}",
            h_ab.n_obs, h_bc.n_obs, h_ac.n_obs
        ));
    }

    let finer = if h_ab.basis.num_segments() >= h_ac.basis.num_segments() {
        &h_ab.basis
    } else {
        &h_ac.basis
    };
    let grid = piece_grid(finer);
    let n_grid = grid.len();
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

    // --- pointwise studentization by the fitted curves' own sampling law -----
    // The quantity tested is a contrast of FITTED CURVES, so the law it is read
    // against is the sampling law of those curves — exactly what the joint
    // influence sandwich above estimates, shared rows and all. At a grid point
    // that scale is O(σ√(edf/n)), not O(σ).
    //
    // #3512 — an earlier rule floored this variance at the three maps' observed
    // residual RMS, combined by Minkowski's inequality. `residual_rms` is the
    // in-sample OBSERVATION residual RMS, which estimates the per-observation
    // noise σ: on any noisy fit the floor exceeded the true variance by about
    // 9n/edf, deflated every z by about 3√(n/edf), and pinned the test at a
    // power that does not grow with n — a fixed violation δ scored z ≈ δ/(3σ)
    // at every sample size. It was not a bound on the representation error it
    // was named for either: an RMS over the data rows bounds nothing about the
    // approximation error at a grid point, so defect/residual_rms is a standard
    // normal deviate under no law. A conservative calibration is a calibration
    // defect exactly as an anti-conservative one is, so there is no floor.
    //
    // The representation defect of the finite spline family under composition
    // (#2143) is a bias, not a variance, and is not priced as one here; it is
    // the deterministic-map estimator of #3364. A deterministic pair carries no
    // observation law at all: every residual is rounding, its influence
    // collapses to zero, `max_var` is zero, and no p-value is emitted.
    let max_var = variance.iter().copied().fold(0.0_f64, f64::max);
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
            // Studentize by this point's own variance. With none, a nonzero
            // defect is infinitely many standard errors out and a zero one is none.
            let pointwise_variance = variance[i];
            let z = if pointwise_variance > 0.0 {
                a / pointwise_variance.sqrt()
            } else if a > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
            max_z = max_z.max(z);
        }
    }
    let mean_abs_defect = sum_abs / n_grid as f64;
    let rms_defect = (sum_sq / n_grid as f64).sqrt();

    // Bonferroni bound for the max studentized defect over the actual grid:
    // valid for arbitrary dependence among pointwise contrasts, and at most a
    // factor `n_grid` conservative, which scales the p-value but not the way it
    // moves with the sample. With no empirical score variation there is no
    // uncertainty law, so no p-value is emitted from deterministic fitted-grid
    // values.
    let max_studentized_p_value = if max_var > 0.0 {
        let pointwise = normal_two_sided_probability(max_z);
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
    pair_law: PairLaw,
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
            pair_law,
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
            pair_law,
        )
        .map_err(|e| {
            format!(
                "two-hop transport {}→{} failed: {e}",
                layers[k],
                layers[k + 2]
            )
        })?;
        let composition = composition_defect(&adjacent_fits[k], &adjacent_fits[k + 1], &direct)
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
    // evaluated at the map's own observed source coordinates. Additive report;
    // no fitting-path effect.
    let mut circle_transports = Vec::new();
    for k in 0..depth - 1 {
        if let Some(report) = crate::inference::transport_class::classify_circle_transport_fit(
            &adjacent_fits[k],
            coords[k].view(),
            topologies[k],
            topologies[k + 1],
            layers[k],
            layers[k + 1],
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
    use gam_linalg::faer_ndarray::FaerQr;
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
            PairLaw::Deterministic,
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
        let quadratic = fitted_from_target(from.view(), to.view(), 0.0, 1.0);
        // Keep a fitted, orientation-reversing arm as well: the minimum-curvature
        // interpolant of a nonpolynomial map with h′ in [-1.25, -0.75].
        let nonlinear = from.mapv(|t| 1.0 - t - 0.25 * (TAU * t).sin() / TAU);
        let fitted = fit_transport_map(
            from.view(),
            nonlinear.view(),
            interval(0.0, 1.0),
            interval(0.0, 1.0),
            PairLaw::Deterministic,
        )
        .expect("fit decreasing nonlinear map");
        let probe = Array1::from_iter((1..10).map(|i| i as f64 / 10.0));
        for (label, ft) in [("quadratic", quadratic), ("fitted", fitted)] {
            assert!(ft.topology_preserved, "{label}");
            let fwd = ft.eval(probe.view()).expect("eval");
            let back = ft.invert(fwd.view()).expect("invert");
            for i in 0..probe.len() {
                assert!(
                    (back[i] - probe[i]).abs() < 1e-6,
                    "{label}: t={} back={}",
                    probe[i],
                    back[i]
                );
            }
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
            PairLaw::Deterministic,
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
        let ft = fitted_from_target(from.view(), to.view(), 0.0, 1.0);
        assert!(ft.invert(Array1::from_elem(1, 0.9).view()).is_err());
    }

    /// Build a `FittedTransport` on an interval whose pre-wrap map reproduces a
    /// polynomial target of degree at most three by an unpenalized least-squares
    /// fit in the one-piece cubic space, which holds such a target exactly. These
    /// fixtures test the inverse geometry of a specified map (#2822), and the
    /// exact reproduction keeps a deliberately narrow fold where the target puts
    /// it. Inference-only fields are placeholders.
    fn fitted_from_target(
        from: ArrayView1<'_, f64>,
        target: ArrayView1<'_, f64>,
        lo: f64,
        hi: f64,
    ) -> FittedTransport {
        let basis = DomainBasis::uniform(interval(lo, hi), 1);
        let design = basis.value_rows(from).expect("design");
        let m = design.ncols();
        let (q, r) = design.qr().expect("fixture QR");
        let rhs = q.t().dot(&target);
        let mut beta = Array1::<f64>::zeros(m);
        for i in (0..m).rev() {
            assert!(
                r[[i, i]].is_finite() && r[[i, i]] != 0.0,
                "full-rank fixture"
            );
            let tail: f64 = ((i + 1)..m).map(|j| r[[i, j]] * beta[j]).sum();
            beta[i] = (rhs[i] - tail) / r[[i, i]];
        }
        let scale = target.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
        let tolerance = (from.len() + m) as f64 * f64::EPSILON * scale;
        for (actual, expected) in design.dot(&beta).iter().zip(target.iter()) {
            assert!((actual - expected).abs() <= tolerance, "fixture interpolation");
        }
        let mut fitted = FittedTransport {
            topology_from: interval(lo, hi),
            topology_to: interval(lo, hi),
            pair_law: PairLaw::Deterministic,
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
            isometry_defect_band: 0.0,
            topology_preserved: false,
            min_directional_derivative: f64::NAN,
            residual_rms: 0.0,
            coefficient_score_influence: Array2::<f64>::zeros((m, from.len())),
            basis,
        };
        let min_directional_derivative = fitted
            .exact_minimum_oriented_derivative()
            .expect("fixture exact fold minimum")
            .1;
        fitted.min_directional_derivative = min_directional_derivative;
        fitted.topology_preserved = min_directional_derivative > 0.0;
        fitted
    }

    /// Reviewer's between-grid fold reproducer: h(t) = (t−0.5)³/3 − (0.4/511)²·t
    /// hides a narrow fold between the samples of a 512-point grid, which is the
    /// grid the fold diagnostic used to be sampled on. That sampled minimum reads
    /// positive while a 10× denser grid finds orientation·h′ < 0. The published
    /// diagnostic is now the exact span minimum, so it must read the fold, the
    /// topology verdict must be false, and `invert` must still refuse.
    #[test]
    fn invert_rejects_between_grid_fold() {
        let n = 256;
        let from: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 / (n as f64 - 1.0)));
        let eps = 0.4 / 511.0;
        let target: Array1<f64> = from.mapv(|t| (t - 0.5).powi(3) / 3.0 - eps * eps * t);
        let ft = fitted_from_target(from.view(), target.view(), 0.0, 1.0);

        // Confirm the fold is genuinely between the samples of the historical
        // 512-point grid, and exposed by a 10× denser one.
        let grid = Array1::linspace(0.0, 1.0, 512);
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
        assert!(
            min_grid > 0.0 && min_dense < 0.0,
            "fixture must hide a between-grid fold: min on 512-grid={min_grid}, \
             min on dense grid={min_dense}"
        );
        // The published diagnostic is the exact minimum, so it cannot pass over
        // the fold the sampled grid missed.
        assert!(
            ft.min_directional_derivative < 0.0 && !ft.topology_preserved,
            "exact fold diagnostic must read the between-grid fold: \
             min_directional_derivative={}, topology_preserved={}, min on 512-grid={min_grid}, \
             min on dense grid={min_dense}",
            ft.min_directional_derivative,
            ft.topology_preserved
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
    fn transport_fit_rejects_unidentified_dispersion_for_exact_affine_data() {
        let from = Array1::linspace(0.0, 1.0, 64);
        let to = from.mapv(|t| 0.5 * t);
        let refusal = fit_transport_map(
            from.view(),
            to.view(),
            interval(0.0, 1.0),
            interval(0.0, 1.0),
            PairLaw::Stochastic,
        )
        .expect_err("exact interpolation cannot identify a positive Gaussian scale");
        assert!(
            refusal.contains("profiled residual") && refusal.contains("not resolvably positive"),
            "the refusal must identify the missing dispersion information: {refusal}"
        );
    }

    #[test]
    fn invert_rejects_non_finite_targets() {
        let n = 64;
        let from: Array1<f64> = Array1::from_iter((0..n).map(|i| i as f64 / (n as f64 - 1.0)));
        let to: Array1<f64> = from.mapv(|t| 0.5 * t);
        let ft = fitted_from_target(from.view(), to.view(), 0.0, 1.0);
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
        let ft = fitted_from_target(from.view(), to.view(), 0.0, 1.0);
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
            PairLaw::Deterministic,
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

    /// A degree-3 map lies outside any hand-picked candidate box such as
    /// `{−2,…,2}`, which could only return a wrong degree for it. The closed-loop
    /// walk reads the winding off the pairs, so it needs no box.
    #[test]
    fn winding_degree_three_is_read_from_the_closed_loop_walk() {
        let n = 128;
        let from: Array1<f64> = Array1::from_iter((0..n).map(|i| TAU * i as f64 / n as f64));
        let to: Array1<f64> = from.mapv(|t| wrap_tau(3.0 * t + 0.4 + 0.1 * t.sin()));
        let ft = fit_transport_map(
            from.view(),
            to.view(),
            ChartTopology::Circle,
            ChartTopology::Circle,
            PairLaw::Deterministic,
        )
        .expect("fit");
        assert_eq!(ft.degree, Some(3), "expected winding degree 3, got {:?}", ft.degree);
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
            PairLaw::Deterministic,
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
            PairLaw::Deterministic,
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
        let crit = monomial_critical_points(&recon).expect("closed-form vertex");
        assert_eq!(crit.len(), 1);
        assert!((crit[0] - 1.3 / 4.2).abs() < 1e-12);
    }
}
