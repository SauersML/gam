use super::*;

/// Basis/topology tag for one SAE manifold atom.
///
/// The evaluated basis and input-location jet live on [`SaeManifoldAtom`].
/// This enum records the user-facing topology choice so downstream diagnostics
/// and Python wrappers can round-trip whether the atom was a Duchon patch,
/// periodic curve, sphere, or a caller-supplied precomputed basis.
/// Integration measure `ω_i` for the intrinsic bending Gram
/// [`SaeManifoldAtom::refresh_intrinsic_smooth_penalty`].
///
/// The bending penalty is a discrete quadrature of the total squared second
/// fundamental form `∫_M ‖II‖²_g dμ` over the grid points `t_i`. The measure is
/// the quadrature weight `ω_i` attached to each shape-band grid point:
///
/// * [`Self::Volume`] — `ω_i = √det g_i`, the Riemannian volume element. This
///   is the DEFAULT: the penalty is the local-VOLUME-weighted bending sum
///   `Σ_i √det g_i ‖II(t_i)‖²_g`, weighting each sample's bending by the
///   Riemannian volume its chart cell carries. Two consequences, stated
///   honestly:
///   * A region the chart barely covers (small `√det g`) is CHEAP to bend, not
///     expensive — its contribution `ω_i ‖II_i‖²` is scaled toward zero. That
///     is the INTENDED prior: do not spend penalty budget keeping flat a region
///     that carries little volume (hence little data); reserve the roughness
///     budget for the high-volume, data-rich part of the manifold. (The volume
///     weight also bounds the cost of the near-boundary chart blow-ups the
///     raw-`t` measure would over-count.)
///   * This sum is NOT sampling-density invariant on its own: it approximates
///     `∫_M ‖II‖²_g dV_g` only up to the density `q(t_i)` of the grid points —
///     an unbiased estimate would need an importance weight `1/q(t_i)` or
///     explicit quadrature-cell widths. In practice the grid points ARE the
///     fitting-data locations, so `√det g` measures bending relative to the
///     data density: a deliberate data-adaptive prior, not a chart-independent
///     invariant.
/// * [`Self::Data`] — `ω_i = 1`, the raw counting measure over the sample
///   rows. Because `‖II(t_i)‖²_g` is a POINTWISE chart-invariant scalar (a
///   full `g`-contraction of a normal-bundle-valued `(0,2)`-tensor), summing it
///   with unit weights over *matched material points* is EXACTLY invariant
///   under any reparameterisation that keeps those same physical points — this
///   is the measure the gauge-invariance test uses. Offered as an option for
///   callers that supply their own (already volume-weighted or
///   importance-sampled) grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeBendingMeasure {
    Volume,
    Data,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SaeAtomBasisKind {
    Duchon,
    Periodic,
    Sphere,
    Torus,
    /// Cylinder `S¹ × ℝ` (`d = 2`): a periodic circle axis tensored with a flat
    /// (Duchon-polynomial) line axis, via [`CylinderHarmonicEvaluator`]. Axis 0
    /// is the circle (fraction-of-period convention, wrapped modulo `1.0`),
    /// axis 1 is the unbounded line (`Euclidean`). Completes the `d = 2`
    /// topology race (torus vs sphere vs euclidean-patch vs cylinder) so a
    /// periodic-times-linear feature is adjudicable on its true manifold instead
    /// of being forced into a torus or flat-patch stand-in.
    Cylinder,
    /// A genuinely LINEAR (affine) decoder atom: `γ(t) = b₀ + Σ_a t_a·b_a`, the
    /// degree-1 monomial patch `{1, t₁, …, t_d}` (#1221). This is the principled
    /// reconstruction-parity baseline — one straight decoder direction per latent
    /// axis plus an intercept — distinct from [`Self::EuclideanPatch`], which is
    /// the degree-2 QUADRATIC patch `{1, t, t²}`. It shares the
    /// [`crate::basis::EuclideanPatchEvaluator`] (at `max_degree = 1`)
    /// and the flat Euclidean latent manifold, so the only difference from the
    /// quadratic patch is the (smaller, linear) basis width — which is exactly
    /// what makes a "curved vs linear" comparison honest rather than
    /// "curved vs quadratic." Round-trips under the name `"linear"`.
    Linear,
    EuclideanPatch,
    /// Hyperbolic (Poincaré-ball) tangent patch at unit curvature `c = −1`.
    ///
    /// Shares the monomial decoder design of [`Self::EuclideanPatch`] — the
    /// latent coordinate `t` is read as a tangent vector at the ball origin
    /// (the wrapped / tangent parameterisation) and the decoder is the same
    /// polynomial-in-`t` expansion — but its smoothness penalty is measured in
    /// *hyperbolic* arc length rather than flat tangent length
    /// (`refresh_intrinsic_smooth_penalty`). For the `d = 1` tangent chart the
    /// coordinate runs at a constant multiple of arc length (geodesic distance
    /// `= 2|t|`), so the intrinsic reweighting is a *constant* — coinciding with
    /// the flat arc-length reweighting, since the chart is intrinsically flat in
    /// 1-D (see `poincare.rs::conformal_dirichlet_penalty`, whose `d = 1` metric
    /// weight is the constant `G ≡ 1/2`). The genuinely hyperbolic, curvature-
    /// dependent anisotropy is a `d ≥ 2` matrix effect carried by that pullback,
    /// not by this scalar `d = 1` path. The decoder is nonetheless the tangent-
    /// wrapped exp-map parameterisation, so an atom whose feature density grows
    /// toward the ball boundary (exponential-volume / tree-leaf hierarchy) still
    /// retracts on the hyperbolic manifold.
    Poincare,
    /// A FINITE-SET (discrete anchor) atom (F2): the latent `t` is CATEGORICAL —
    /// each sample is assigned to one of a finite set of anchors — and the basis
    /// is the indicator/one-hot design over those anchors
    /// ([`crate::basis::AnchorIndicatorEvaluator`]). Unlike every other kind here,
    /// which is a continuous manifold, this is a discrete measure: the honest
    /// model for cluster-like structure (weekdays as 7 points with cyclic
    /// adjacency, not an occupied circle). Its rank charge is `anchors − 1` (the
    /// categorical `t` has `anchors − 1` independent contrasts, one anchor being
    /// the reference) — see [`finite_set_rank_charge`]. The anchor count is carried
    /// by the evaluator (as harmonics/degree are for the periodic/patch kinds), so
    /// this stays a unit variant.
    ///
    /// PLANNED COMPLETION / OPT-IN: the topology race does NOT enrol this candidate
    /// by default (see [`crate::structure_harvest::finite_set_race_enrolled`]); the
    /// enum arm + evaluator land as inert scaffolding so unenrolled code cannot
    /// affect any birth, and the enrolment flag flips only after full-suite +
    /// real-data (weekday) verification. First-class integration into the
    /// continuous-latent optimizer is the remaining follow-up.
    FiniteSet,
    Precomputed(String),
}

/// The rank charge (effective latent dimension) of a finite-set atom with
/// `anchors` anchors: `anchors − 1`. A categorical coordinate over `k` anchors has
/// `k − 1` independent contrasts (one anchor is the reference level), so that — not
/// `k`, and not a continuous manifold's intrinsic `d` — is what the race must
/// charge the finite-set alternative. Returns `0` for the degenerate `anchors ≤ 1`
/// (a single anchor is the constant, no contrasts).
pub fn finite_set_rank_charge(anchors: usize) -> usize {
    anchors.saturating_sub(1)
}

impl SaeAtomBasisKind {
    pub(crate) fn latent_manifold(&self, latent_dim: usize) -> LatentManifold {
        match self {
            // `Periodic` uses [`PeriodicHarmonicEvaluator`], whose basis
            // functions are `cos(2π·h·t), sin(2π·h·t)` — i.e. `t` is a
            // fraction of one period, not radians. The latent manifold
            // wraps modulo `period = 1.0` to match this convention.
            // Wrapping modulo `2π` instead would scramble the
            // fraction-of-period interpretation and cause #174-style
            // failures where Newton updates push `t` outside `[0, 1)` and
            // the optimiser sees a discontinuous landscape.
            Self::Periodic => {
                if latent_dim == 1 {
                    LatentManifold::Circle { period: 1.0 }
                } else {
                    LatentManifold::Product(
                        (0..latent_dim)
                            .map(|_| LatentManifold::Circle { period: 1.0 })
                            .collect(),
                    )
                }
            }
            // `Sphere` is parameterised via a (lat, lon) *product* chart, NOT an
            // intrinsic / rotation-invariant `S²` parametrisation: the latent
            // optimiser sees a 2-D product manifold whose cos/sin terms (in
            // radians) embed the chart, where lat is a bounded interval
            // `[-π/2, π/2]` (enforced here by the `Interval` retraction — its
            // clamp + active-bound tangent projection — NOT by truncating the
            // chart jet) and lon is an `S^1` angle wrapped modulo `2π`. This chart
            // carries pole gauge singularities: at the poles `cos(lat) = 0`, all
            // longitudes collapse to the same physical point, so longitude is a
            // gauge coordinate there and the longitude jet vanishes; and the
            // `[xy, yz, xz]` quadratic block is not a rotation-invariant spherical-
            // harmonic basis (rotating `xy` yields `x² − y²`, outside its span).
            // Both caveats are documented in full at
            // `gam_sae::basis::sphere_chart_basis_jet`; do not read this chart as
            // artefact-free spherical geometry.
            // Treating it as `LatentManifold::Sphere { dim: 2 }` would
            // require ambient unit-vectors of length 2 (impossible for S^2).
            Self::Sphere => LatentManifold::Product(vec![
                LatentManifold::Interval {
                    lo: -std::f64::consts::FRAC_PI_2,
                    hi: std::f64::consts::FRAC_PI_2,
                },
                LatentManifold::Circle {
                    period: std::f64::consts::TAU,
                },
            ]),
            // `Torus` uses [`TorusHarmonicEvaluator`], which shares the
            // fraction-of-period convention with `PeriodicHarmonicEvaluator`
            // (basis is `cos(2π·h·t)`, `sin(2π·h·t)` on each axis). Each
            // per-axis latent wraps modulo `1.0`.
            Self::Torus => {
                if latent_dim == 1 {
                    LatentManifold::Circle { period: 1.0 }
                } else {
                    LatentManifold::Product(
                        (0..latent_dim)
                            .map(|_| LatentManifold::Circle { period: 1.0 })
                            .collect(),
                    )
                }
            }
            // `Cylinder` is `S¹ × ℝ`: axis 0 is the circle (fraction-of-period
            // convention, shared with `Periodic`/`Torus`, wrapped modulo `1.0`)
            // and axis 1 is the unbounded line (`Euclidean`). The product
            // latent manifold composes the two retractions blockwise.
            Self::Cylinder => LatentManifold::Product(vec![
                LatentManifold::Circle { period: 1.0 },
                LatentManifold::Euclidean,
            ]),
            // Poincaré tangent patch: the latent `t` is a tangent vector at the
            // ball origin, optimised in the unconstrained tangent chart (the
            // hyperbolic geometry enters through the penalty, not a constrained
            // retraction), so it shares the Euclidean latent manifold.
            // A finite-set atom's categorical assignment is carried as a flat
            // (Euclidean) coordinate: the anchor index. The discreteness lives in
            // the indicator basis, not the latent manifold, so the retraction is
            // the trivial Euclidean one.
            Self::Linear
            | Self::Duchon
            | Self::EuclideanPatch
            | Self::Poincare
            | Self::FiniteSet
            | Self::Precomputed(_) => LatentManifold::Euclidean,
        }
    }

    /// Dense candidate coordinates spanning compact latents for fixed-decoder
    /// out-of-sample projection. Unbounded/basis-linear latents return `None`
    /// because their PCA seed already lies in the convex training hull.
    pub(crate) fn projection_seed_grid(
        &self,
        latent_dim: usize,
        resolution: usize,
    ) -> Option<Array2<f64>> {
        match self {
            Self::Periodic => torus_projection_seed_grid(latent_dim, resolution),
            Self::Sphere if latent_dim == 2 => sphere_projection_seed_grid(resolution),
            Self::Sphere => None,
            Self::Torus => torus_projection_seed_grid(latent_dim, resolution),
            // `Cylinder` (`S¹ × ℝ`) has one compact (circle) axis that wraps and
            // one unbounded (line) axis whose PCA seed already lies in the
            // convex hull. A robust fixed-decoder projection therefore only
            // needs to sweep the *periodic* axis (the line axis is left at its
            // hull-centered seed `0`); a pure line offset is recovered by the
            // unconstrained Newton step.
            Self::Cylinder if latent_dim == 2 => cylinder_projection_seed_grid(resolution),
            Self::Cylinder => None,
            // The tangent latent of a Poincaré patch lies in the convex hull of
            // its PCA seed exactly like the Euclidean patch, so no compact
            // projection grid is needed.
            // A finite-set atom has no compact continuous latent to sweep for
            // fixed-decoder projection — its assignment is categorical, recovered
            // by nearest-anchor, not a projection grid.
            Self::Linear
            | Self::Duchon
            | Self::EuclideanPatch
            | Self::Poincare
            | Self::FiniteSet
            | Self::Precomputed(_) => None,
        }
    }
}

pub(crate) fn sphere_projection_seed_grid(resolution: usize) -> Option<Array2<f64>> {
    use std::f64::consts::PI;
    let r = resolution.max(2);
    let mut grid = Array2::<f64>::zeros((r * r, 2));
    for i in 0..r {
        let lat = -PI / 2.0 + PI * (i as f64 + 0.5) / r as f64;
        for j in 0..r {
            let lon = -PI + 2.0 * PI * (j as f64) / r as f64;
            grid[[i * r + j, 0]] = lat;
            grid[[i * r + j, 1]] = lon;
        }
    }
    Some(grid)
}

pub(crate) fn cylinder_projection_seed_grid(resolution: usize) -> Option<Array2<f64>> {
    // Sweep the periodic (circle) axis over one period in fraction-of-period
    // coordinates `[0, 1)`; hold the unbounded line axis at the hull-centered
    // seed `0`. The Newton retraction recovers any line offset from there.
    let r = resolution.max(2);
    let mut grid = Array2::<f64>::zeros((r, 2));
    for i in 0..r {
        grid[[i, 0]] = i as f64 / r as f64;
        grid[[i, 1]] = 0.0;
    }
    Some(grid)
}

pub(crate) fn torus_projection_seed_grid(
    latent_dim: usize,
    resolution: usize,
) -> Option<Array2<f64>> {
    if latent_dim == 0 || latent_dim >= usize::BITS as usize {
        return None;
    }
    const MAX_GRID_POINTS: usize = 4096;
    let min_points = 1usize << latent_dim;
    if min_points > MAX_GRID_POINTS {
        return None;
    }
    let requested = resolution.max(2);
    let mut per_axis = requested;
    while per_axis.saturating_pow(latent_dim as u32) > MAX_GRID_POINTS {
        per_axis -= 1;
        if per_axis < 2 {
            return None;
        }
    }
    let total: usize = (0..latent_dim).fold(1usize, |acc, _| acc.saturating_mul(per_axis));
    let mut grid = Array2::<f64>::zeros((total, latent_dim));
    let mut idx = vec![0usize; latent_dim];
    for flat in 0..total {
        for axis in 0..latent_dim {
            grid[[flat, axis]] = idx[axis] as f64 / per_axis as f64;
        }
        for axis in (0..latent_dim).rev() {
            idx[axis] += 1;
            if idx[axis] < per_axis {
                break;
            }
            idx[axis] = 0;
        }
    }
    Some(grid)
}

/// Per-axis ARD coordinate prior, evaluated as a smooth energy in the latent
/// coordinate `t` with precision `alpha = exp(log_ard)`.
///
/// On a *Euclidean* axis the prior is the usual Gaussian negative-log density
/// `½·α·t²`, with gradient `α·t` and curvature `α`.
///
/// On a *periodic* axis (a `Circle` factor of period `P`) the Euclidean `½α t²`
/// is geometrically ill-posed (it depends on the arbitrary choice of origin /
/// branch cut, so a Newton step crossing the cut makes the loss jump by
/// `½α P²` and breaks Armijo descent). We replace it with the von-Mises energy
///
/// ```text
///   V(t) = (α / κ²) · (1 − cos(κ t)),   κ = 2π / P
/// ```
///
/// which is the period-`P` periodic function whose Taylor expansion at the
/// origin is `½ α t² + O(t⁴)` — so it carries the *same* precision `α`
/// (curvature at the origin) as the Gaussian, matching the ARD interpretation,
/// but is globally smooth and continuous across the cut (`cos(κ·P)=cos 2π=1`).
/// Its derivatives are
///
/// ```text
///   V'(t)  = (α / κ) · sin(κ t)
///   V''(t) = α · cos(κ t)
/// ```
///
/// The value, gradient, and curvature returned here all come from this single
/// energy, so they are mutually FD-consistent. The *value* (`ard_value` /
/// `loss.ard`) and the *gradient* (the assembled `gt`) use the exact `V` and
/// `V'`. The curvature `V'' = α·cos(κt)` is INDEFINITE — it turns negative for
/// `|κt|` past `π/2` (a quarter period) — so it is NOT written raw into the
/// Newton/Schur `H_tt` diagonal: that would make the per-row coordinate block
/// indefinite and the Schur (and log-det) Cholesky would fail on a non-PD pivot
/// at `K ≥ 2`. The assembly accumulates the PSD majorizer `max(V'', 0)` into
/// `H_tt` instead (mirroring `add_sae_coord_penalty`'s `psd_majorizer_diag` for
/// the registry coord penalties). Majorizing the curvature of a *fixed* prior
/// only damps the Newton step; the stationary point is set by the exact gradient
/// `V'`, so it is unchanged. The Laplace `½ log|H|` is therefore evaluated on the
/// same PSD-majorized `H_tt` (a valid Cholesky requires a PD operator anyway).
///
/// `sq_equiv` is the Euclidean-equivalent `t²` such that `½·α·sq_equiv == V`,
/// i.e. `sq_equiv = 2V/α = (2/κ²)(1−cos κt)`. It is what the
/// Mackay/Fellner–Schall `α ← n / (Σ sq_equiv + tr H⁻¹)` fixed point must use so
/// that the prior energy it implies stays consistent with `ard_value`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ArdAxisPrior {
    pub(crate) value: f64,
    pub(crate) grad: f64,
    pub(crate) hess: f64,
    pub(crate) sq_equiv: f64,
}

impl ArdAxisPrior {
    /// Evaluate the per-axis prior at coordinate `t` with precision `alpha`.
    /// `period == None` selects the Euclidean Gaussian; `Some(p)` selects the
    /// von-Mises periodic energy with period `p`.
    pub(crate) fn eval(alpha: f64, t: f64, period: Option<f64>) -> Self {
        match period {
            None => Self {
                value: 0.5 * alpha * t * t,
                grad: alpha * t,
                hess: alpha,
                sq_equiv: t * t,
            },
            Some(p) => {
                let kappa = std::f64::consts::TAU / p;
                let (sin, cos) = (kappa * t).sin_cos();
                let one_minus_cos = 1.0 - cos;
                Self {
                    value: (alpha / (kappa * kappa)) * one_minus_cos,
                    grad: (alpha / kappa) * sin,
                    hess: alpha * cos,
                    sq_equiv: (2.0 / (kappa * kappa)) * one_minus_cos,
                }
            }
        }
    }
}

/// Large-argument (`|x| >= 3.75`) Abramowitz & Stegun 9.8.2 polynomial for the
/// *exponentially-scaled* `I0`: `√x · e^{−x} · I0(x) ≈ poly(3.75/x)`. Factoring
/// the `e^{x}/√x` envelope out lets the log-partition and the `I1/I0` ratio be
/// computed without ever materialising `e^{x}` (which overflows to `+inf` for
/// `x ≳ 709`, see [`bessel_i0_log_and_ratio`]).
pub(crate) fn bessel_i0_scaled_poly(ax: f64) -> f64 {
    let y = 3.75 / ax;
    0.39894228
        + y * (0.01328592
            + y * (0.00225319
                + y * (-0.00157565
                    + y * (0.00916281
                        + y * (-0.02057706
                            + y * (0.02635537 + y * (-0.01647633 + y * 0.00392377)))))))
}

/// Large-argument (`|x| >= 3.75`) Abramowitz & Stegun 9.8.4 polynomial for the
/// *exponentially-scaled* `I1`: `√x · e^{−x} · I1(x) ≈ poly(3.75/x)`. Pairs with
/// [`bessel_i0_scaled_poly`] so their shared `e^{x}/√x` envelope cancels exactly
/// in the `I1/I0` ratio.
pub(crate) fn bessel_i1_scaled_poly(ax: f64) -> f64 {
    let y = 3.75 / ax;
    0.39894228
        + y * (-0.03988024
            + y * (-0.00362018
                + y * (0.00163801
                    + y * (-0.01031555
                        + y * (0.02282967
                            + y * (-0.02895312 + y * (0.01787654 - y * 0.00420059)))))))
}

/// Modified Bessel function of the first kind, order zero, `I0(x)`.
///
/// Abramowitz & Stegun 9.8.1 (|x| <= 3.75) and 9.8.2 (|x| > 3.75) polynomial
/// approximations; relative error < 1.6e-7 / 1.9e-7 respectively, which is far
/// below the precision tolerance the ARD normaliser is read at. `I0` is even,
/// so only `|x|` enters. Used for the exact von-Mises precision log-partition.
pub(crate) fn bessel_i0(x: f64) -> f64 {
    let ax = x.abs();
    if ax < 3.75 {
        let t = x / 3.75;
        let t2 = t * t;
        1.0 + t2
            * (3.5156229
                + t2 * (3.0899424
                    + t2 * (1.2067492 + t2 * (0.2659732 + t2 * (0.0360768 + t2 * 0.0045813)))))
    } else {
        (ax.exp() / ax.sqrt()) * bessel_i0_scaled_poly(ax)
    }
}

/// Modified Bessel function of the first kind, order one, `I1(x)`.
///
/// Uses the Abramowitz & Stegun approximations paired with [`bessel_i0`]. This is
/// needed only for the derivative of the periodic ARD precision normalizer
/// `log I0(η)`, whose derivative is `I1(η) / I0(η)`.
pub(crate) fn bessel_i1(x: f64) -> f64 {
    let ax = x.abs();
    let value = if ax < 3.75 {
        let t = x / 3.75;
        let t2 = t * t;
        ax * (0.5
            + t2 * (0.87890594
                + t2 * (0.51498869
                    + t2 * (0.15084934 + t2 * (0.02658733 + t2 * (0.00301532 + t2 * 0.00032411))))))
    } else {
        (ax.exp() / ax.sqrt()) * bessel_i1_scaled_poly(ax)
    };
    if x < 0.0 { -value } else { value }
}

/// Overflow-free `(log I0(η), I1(η)/I0(η))` for `η >= 0`, the only two Bessel
/// quantities the von-Mises ARD precision normaliser and its ρ-gradient need.
///
/// The naive `bessel_i0(η).ln()` and `bessel_i1(η)/bessel_i0(η)` both route
/// through `e^{η}/√η`, which overflows to `+inf` once `η ≳ 709`. Two `+inf`s
/// then divide to `NaN`, poisoning the very first outer ρ-gradient on
/// large-norm / ill-conditioned checkpoints (issue #1113: a dispersion-inflated
/// ARD seed pushes `η = α/κ²` past the overflow threshold at iter 0). For a
/// periodic circle atom (`κ = 2π`) this fires for any seed precision
/// `α ≳ 2.8e4`, well inside the reachable seed range.
///
/// We never form `e^{η}`. For the small branch (`η < 3.75`) the A&S series are
/// finite, so we evaluate them directly. For the large branch the shared
/// `e^{η}/√η` envelope cancels in the *log* (`log I0 = η − ½ ln η + ln poly`)
/// and in the *ratio* (`I1/I0 = poly₁/poly₀`), so both are computed from the
/// bounded scaled polynomials alone — exact for non-degenerate η and finite for
/// every finite η.
pub(crate) fn bessel_i0_log_and_ratio(eta: f64) -> (f64, f64) {
    let ax = eta.abs();
    if ax < 3.75 {
        let i0 = bessel_i0(ax);
        let i1 = bessel_i1(ax);
        (i0.ln(), i1 / i0)
    } else {
        let poly0 = bessel_i0_scaled_poly(ax);
        let poly1 = bessel_i1_scaled_poly(ax);
        let log_i0 = ax - 0.5 * ax.ln() + poly0.ln();
        let ratio = poly1 / poly0;
        (log_i0, ratio)
    }
}
/// One manifold atom.
///
/// `basis_values` is `Phi_k(t_{ik})`, shape `(N, M_k)`.
/// `basis_jacobian` is `d Phi_k / d t_{ik}`, shape `(N, M_k, d_k)`.
/// `decoder_coefficients` is `B_k`, shape `(M_k, p)`.
/// `smooth_penalty` is `P_k`, shape `(M_k, M_k)`.
#[derive(Debug, Clone)]
pub struct SaeManifoldAtom {
    pub name: String,
    pub basis_kind: SaeAtomBasisKind,
    pub latent_dim: usize,
    pub basis_values: Array2<f64>,
    pub basis_jacobian: Array3<f64>,
    pub decoder_coefficients: Array2<f64>,
    /// Effective (intrinsic) roughness Gram `S̃_k` that every consumer reads
    /// (smoothness value, gradient, Kronecker Hessian op, REML rank/log-det).
    ///
    /// `S̃_k` is the raw coefficient-space Gram [`Self::smooth_penalty_raw`]
    /// reparameterized by the decoder pullback metric so the roughness — and
    /// hence the topology evidence — is gauge-invariant under reparameterization
    /// of the latent coordinate `t` (issue #673). It is recomputed from the
    /// current basis Jacobian and decoder coefficients by
    /// [`Self::refresh_intrinsic_smooth_penalty`] (lagged-diffusivity: the
    /// metric weight is frozen within each inner Newton/evidence assembly and
    /// refreshed between them, so at convergence the penalty is the true
    /// arc-length roughness). The metric weight is centered (geometric mean 1),
    /// so for constant-speed atoms (the periodic sin/cos basis on `S¹`) every
    /// weight is exactly `1` and `S̃_k = S_k` — periodic atoms are untouched
    /// and no overall magnitude leaks into the penalty.
    pub smooth_penalty: Array2<f64>,
    /// Canonical raw roughness Gram `S_k` in raw coefficient/`t` space (the
    /// finite-/cyclic-difference Reinsch Gram or the Duchon RKHS Gram). Never
    /// mutated after construction; [`Self::smooth_penalty`] is derived from it
    /// each assembly via the pullback-metric reweighting.
    pub smooth_penalty_raw: Array2<f64>,
    /// Roughness operator order `r` of [`Self::smooth_penalty_raw`], recovered
    /// once at construction as its null-space dimension (an order-`r`
    /// difference / Duchon penalty annihilates the degree-`<r` polynomials, so
    /// `nullity(S) = r`). Sets the arc-length reweighting exponent
    /// `β = ½ − r` (`β = −3/2` for the standard second-derivative penalty):
    /// the metric-speed power that converts raw-`t` roughness into intrinsic
    /// arc-length roughness. `0` when the raw Gram is empty/zero (no
    /// reweighting).
    pub smooth_penalty_order: usize,
    pub basis_evaluator: Option<Arc<dyn SaeBasisEvaluator>>,
    /// Same evaluator upcast to `dyn SaeBasisSecondJet` when the
    /// implementation provides a closed-form Hessian. `None` for
    /// evaluators that only implement the base [`SaeBasisEvaluator`]
    /// trait. Installed via [`Self::with_basis_second_jet`]; the base
    /// [`Self::with_basis_evaluator`] populates only the supertrait
    /// slot. Used by [`refresh_isometry_caches_from_atom`] to install
    /// the `H` cache on isometry penalties when the second jet is
    /// analytically available.
    pub basis_second_jet: Option<Arc<dyn SaeBasisSecondJet>>,
    /// Cached VALUES of the analytic second jet
    /// `H[n, m, a, c] = ∂²Φ_m(t_n) / (∂t_a ∂t_c)`, shape `(N, M, d, d)`, at the
    /// atom's CURRENT latent coordinates. The `d ≥ 2` intrinsic bending Gram
    /// ([`Self::refresh_intrinsic_smooth_penalty`]) needs `∂²Φ` — which, unlike
    /// the `d = 1` scalar-speed reweighting, is NOT recoverable from `Φ` and
    /// `∂Φ` alone — but `refresh_intrinsic_smooth_penalty` takes no coordinates,
    /// so the values are cached here alongside `basis_values` / `basis_jacobian`
    /// and refreshed on the same coordinate change. Populated by
    /// [`Self::refresh_basis`] (from [`Self::basis_second_jet`]), transported by
    /// [`Self::reduce_basis_to_subspace`] (the `M`-axis `Q` congruence), or
    /// installed directly via [`Self::install_bending_second_jet`]. `None` ⇒ no
    /// second jet available (a caller-managed or first-jet-only atom); the
    /// `d ≥ 2` bending path then falls back to the raw Gram, so no atom
    /// regresses relative to the historical flat-`d>1` fallback.
    ///
    /// Frozen at the current iterate (`∂²Φ` depends only on the coordinates, not
    /// the decoder), so it obeys the same lagged-diffusivity contract as the
    /// `(g, Γ)` metric freeze: constant within one inner solve, refreshed
    /// between assemblies so the CONVERGED penalty is the true intrinsic bending
    /// energy.
    pub basis_second_jet_values: Option<Array4<f64>>,
    /// Cached latent coordinates `t[n, a]` (shape `(N, d)`) at the atom's CURRENT
    /// chart, populated by [`Self::refresh_basis`] on every coordinate change.
    ///
    /// Needed by the Poincaré branch of
    /// [`Self::refresh_intrinsic_smooth_penalty`]: the hyperbolic conformal
    /// Dirichlet roughness Gram
    /// [`gam_geometry::manifolds::poincare::conformal_dirichlet_penalty`] is a
    /// function of the latent `t` (through the exp-map pullback metric), which
    /// `refresh_intrinsic_smooth_penalty` does not otherwise receive. Cached here
    /// alongside `basis_values` / `basis_jacobian` and refreshed on the same
    /// coordinate change (the lagged-diffusivity freeze). `None` for a
    /// caller-managed / never-refreshed atom (the Poincaré branch then falls back
    /// to the raw Gram). Unchanged by [`Self::reduce_basis_to_subspace`] (the
    /// `M`-axis congruence leaves the `(N, d)` coordinates untouched).
    pub latent_coords: Option<Array2<f64>>,
    /// Quadrature measure `ω_i` for the intrinsic bending Gram (every latent
    /// dim `d ≥ 1`); see [`SaeBendingMeasure`]. Default
    /// [`SaeBendingMeasure::Volume`] (the Riemannian volume element `√det g_i`;
    /// for `d = 1` this is the arc-length element `ds = ‖γ'‖ dt`, so the penalty
    /// is `∫ κ² ds`). Ignored for Poincaré atoms (which use the conformal
    /// Dirichlet Gram) and for atoms with no second-jet cache (raw-Gram
    /// fallback).
    pub bending_measure: SaeBendingMeasure,
    /// Profiled low-rank Grassmann decoder frame `U_k` (`p × r`), issue #972.
    ///
    /// `None` ⇒ the historical full-`B` path: the border carries the entire
    /// `M_k · p` decoder block and is bit-for-bit unchanged. `Some(frame)` ⇒ the
    /// decoder factors as `B_k = C_k · Uᵀ` with the `M_k · r` coordinate matrix
    /// `C_k = B_k · U` in the border and the frame `U` profiled out by streaming
    /// polar steps. [`Self::decoder_coefficients`] stays the authoritative
    /// reconstructed `B_k` (so every existing consumer is unchanged); the frame
    /// is the *representation* that shrinks the border and contributes the
    /// `r·(p − r)` Grassmann dimensions to the Laplace evidence normalizer.
    /// Activated automatically by [`Self::maybe_activate_decoder_frame`] when the
    /// decoder's effective column rank is materially below `p`; never a flag.
    pub decoder_frame: Option<GrassmannFrame>,
    /// Curvature-homotopy dial `η ∈ [0, 1]` (#1007). [`Self::refresh_basis`]
    /// scales every *curved* basis column (per
    /// [`SaeBasisEvaluator::phi_eta_split`]) by `η`, leaving the *base*
    /// (η-invariant) columns untouched, so `η = 0` is the base-topology
    /// relaxation — the atom on its base columns only — and `η = 1` is the full
    /// curved basis. The base endpoint is NOT in general a linear/affine model:
    /// for the harmonic and sphere-chart bases the base block already carries
    /// extrinsic curvature (a first-harmonic `[sin, cos]` traces a circle, the
    /// sphere chart's `[x, y, z]` traces the sphere). Its decoder sub-problem is
    /// still convex, and a genuine low-rank (Eckart-Young / PCA) residual ceiling
    /// is certified by [`linear_span_anchor`] — a rank bound on every `η`, not a
    /// claim that `η = 0` is curvature-free. The certified tracker walks `η`
    /// from `0 → 1`; every other caller sees the default `1.0`, which makes
    /// [`Self::refresh_basis`] bit-for-bit identical to the un-dialed `evaluate`
    /// path (`evaluate_phi_eta` at `η = 1` returns the unscaled basis).
    /// Caller-managed atoms (no installed evaluator) ignore the dial — there is
    /// no curved/base split without an evaluator to provide it.
    pub homotopy_eta: f64,
    /// #1019: `true` once the post-fit chart canonicalization has been
    /// applied to this atom — the latent chart is then the canonical
    /// representative of its `Diff(M)` orbit (the arc-length / unit-speed
    /// chart for `d = 1`, the minimum-isometry-defect flow chart for `d = 2`
    /// torus atoms) and the residual chart freedom is the finite isometry
    /// group of the reference manifold (rotation + reflection on `S¹`,
    /// reflection + translation on the interval, `Isom(T², flat)` on the
    /// torus). Read by the residual-gauge lowering so the certificate reports
    /// the downgrade with the `PinnedByCanonicalization` provenance. Only
    /// ever set for `latent_dim == 1` atoms and `latent_dim == 2` torus
    /// atoms; never a flag the user controls.
    pub chart_canonicalized: bool,
    /// #2022 — explicit per-atom log-amplitude `s_k` (positivity-free): the atom
    /// contributes `exp(s_k)·Φ_k(t)·B_k`, so `s_k` carries the atom's SCALE and
    /// the decoder `B_k` can be a pure (eventually unit-Frobenius) shape frame.
    /// Decoupling magnitude from the decoder frame removes the SCALE gauge
    /// flat-direction (gate = existence, amplitude = intensity). Default `0.0`
    /// ⇒ `exp(s_k) = 1`, i.e. the historical `Φ·B` contribution bit-for-bit; the
    /// decode primitives skip the scaling entirely when `s_k == 0.0`, so the
    /// hot loop is unchanged until an amplitude is set.
    ///
    /// STEP 1 (this change) installs the field, the decode-path threading, and
    /// the [`Self::absorb_decoder_norm_into_log_amplitude`] peel. Carrying `s_k`
    /// through the joint solve as a free arrow block AND pinning `‖B_k‖ = 1`
    /// (the Stiefel retract) land together in STEP 2: because `set_flat_beta`
    /// round-trips the magnitude-bearing β every inner iterate, `s_k` cannot
    /// carry scale THROUGH the solve until `‖B_k‖` is constrained, so the two
    /// are deliberately coupled.
    pub log_amplitude: f64,
    /// Row indices of the quadrature subsample the `d ≥ 2` bending Gram is
    /// assembled from at scale. `None` ⇒ the Gram uses the full active set and
    /// [`Self::basis_second_jet_values`] is the full `(n, M, d, d)` jet — the
    /// bitwise-legacy path, taken whenever `n ≤ BENDING_QUADRATURE_CAP`.
    /// `Some(rows)` ⇒ the second-jet cache is the COMPACT `(rows.len(), M, d, d)`
    /// jet evaluated ONLY on these rows, and the Gram loop reweights each by
    /// `n / rows.len()` so the penalty scale (hence λ selection) matches the
    /// full-set penalty. This is the memory fix: a torus atom's full second jet
    /// is ~1.5 GB at `n = 10⁶`; the `Q ≈ 4096` compact jet is ~6 MB and still
    /// oversamples the `≤ M(M+1)/2 ≲ 1.2k`-dof Gram 3–6×.
    pub bending_quadrature_rows: Option<Vec<usize>>,
}

/// Quadrature-subsample cap for the `d ≥ 2` bending Gram: above this many active
/// rows the covariant Gram is assembled from a leverage-stratified subsample of
/// this many rows rather than the full set (see
/// [`SaeManifoldAtom::bending_quadrature_rows`]). Below it, the full-set legacy
/// path is bitwise-preserved.
pub const BENDING_QUADRATURE_CAP: usize = 4096;

impl SaeManifoldAtom {
    #[must_use = "build error must be handled"]
    pub fn new(
        name: impl Into<String>,
        basis_kind: SaeAtomBasisKind,
        latent_dim: usize,
        basis_values: Array2<f64>,
        basis_jacobian: Array3<f64>,
        decoder_coefficients: Array2<f64>,
        smooth_penalty: Array2<f64>,
    ) -> Result<Self, String> {
        let n = basis_values.nrows();
        let m = basis_values.ncols();
        let p = decoder_coefficients.ncols();
        if basis_jacobian.dim() != (n, m, latent_dim) {
            return Err(format!(
                "SaeManifoldAtom::new: basis_jacobian must be ({n}, {m}, {latent_dim}); got {:?}",
                basis_jacobian.dim()
            ));
        }
        if decoder_coefficients.nrows() != m {
            return Err(format!(
                "SaeManifoldAtom::new: decoder rows {} must equal basis size {m}",
                decoder_coefficients.nrows()
            ));
        }
        if smooth_penalty.dim() != (m, m) {
            return Err(format!(
                "SaeManifoldAtom::new: smooth penalty must be ({m}, {m}); got {:?}",
                smooth_penalty.dim()
            ));
        }
        if p == 0 {
            return Err("SaeManifoldAtom::new: decoder output dimension must be positive".into());
        }
        // Recover the roughness operator order `r` from the raw Gram's
        // null-space dimension (`nullity(S) = r` for an order-`r` difference /
        // Duchon penalty). This pins the arc-length reweighting exponent
        // `β = ½ − r` once, so the per-assembly reweighting needs no
        // eigendecomposition in the hot loop.
        let smooth_penalty_order = smooth_penalty_nullity(&smooth_penalty)?;
        let mut atom = Self {
            name: name.into(),
            basis_kind,
            latent_dim,
            basis_values,
            decoder_coefficients,
            smooth_penalty_raw: smooth_penalty.clone(),
            smooth_penalty,
            smooth_penalty_order,
            basis_jacobian,
            basis_evaluator: None,
            basis_second_jet: None,
            basis_second_jet_values: None,
            latent_coords: None,
            bending_measure: SaeBendingMeasure::Volume,
            decoder_frame: None,
            homotopy_eta: 1.0,
            chart_canonicalized: false,
            // #2022 — default 0.0 ⇒ exp(s)=1 ⇒ historical `Φ·B` bit-for-bit.
            log_amplitude: 0.0,
            bending_quadrature_rows: None,
        };
        // Seed `smooth_penalty` with the intrinsic Gram at the initial
        // decoder/coordinates so the very first assembly already reads the
        // pullback-metric-reweighted penalty.
        atom.refresh_intrinsic_smooth_penalty();
        Ok(atom)
    }

    pub fn with_basis_evaluator(mut self, evaluator: Arc<dyn SaeBasisEvaluator>) -> Self {
        self.basis_evaluator = Some(evaluator);
        self.basis_second_jet = None;
        self
    }

    /// Install an evaluator that additionally exposes a closed-form
    /// second jet. Populates both the base [`SaeBasisEvaluator`] slot
    /// (used by [`Self::refresh_basis`] and the standard evaluate path)
    /// and the [`SaeBasisSecondJet`] slot (consumed by
    /// [`refresh_isometry_caches_from_atom`] for the `H` cache).
    pub fn with_basis_second_jet(mut self, evaluator: Arc<dyn SaeBasisSecondJet>) -> Self {
        let base: Arc<dyn SaeBasisEvaluator> = evaluator.clone();
        self.basis_evaluator = Some(base);
        self.basis_second_jet = Some(evaluator);
        self
    }

    /// Rank-revealing reduction of this atom's fixed-width basis onto the
    /// data-supported subspace `Q` (`M × r`, orthonormal columns, `r ≤ M`),
    /// the root-cause fix for issue #1117.
    ///
    /// A fixed-depth decoder basis (e.g. [`PeriodicHarmonicEvaluator`]) emits
    /// `M` columns whether or not the data excites them; on a near-degenerate
    /// checkpoint the unexcited columns make the design rank-deficient by
    /// construction, flattening the outer REML surface and stalling the solve.
    /// Here we replace the basis with its restriction to the data-identified
    /// subspace, so the design is **full-rank by construction** and the outer
    /// problem is well-posed. Everything transforms by the same `Q` congruence:
    ///
    /// * basis design `Φ̃ = Φ Q`  (`basis_values`, and on every refresh through
    ///   the wrapped [`SubspaceReducedEvaluator`]),
    /// * basis Jacobian `∂Φ̃ = (∂Φ) Q`  (`basis_jacobian`),
    /// * decoder `B̃ = Qᵀ B`  — the minimum-norm pre-image, dropping exactly the
    ///   data-null component that carries no curvature, so the reconstruction
    ///   `Φ̃ B̃ = Φ Q Qᵀ B = Φ B_range` is the rank-`r` oracle,
    /// * roughness Gram `S̃ = Qᵀ S Q` (`smooth_penalty`, `smooth_penalty_raw`),
    /// * evaluator → `SubspaceReducedEvaluator(inner, Q)` so the reduction
    ///   *survives* every `refresh_basis` re-evaluation.
    ///
    /// Requires an installed analytic second-jet evaluator (so the wrapper can
    /// compose the jets); a caller-managed atom (no evaluator) is left
    /// untouched. `Q` with `r == M` and `Q == I` is the well-conditioned case
    /// and the caller should skip the reduction entirely so that path stays
    /// byte-for-byte unchanged.
    pub fn reduce_basis_to_subspace(&mut self, q: &Array2<f64>) -> Result<(), String> {
        let m = self.basis_size();
        if q.nrows() != m {
            return Err(format!(
                "SaeManifoldAtom::reduce_basis_to_subspace: column map has {} rows, basis width {m}",
                q.nrows()
            ));
        }
        let r = q.ncols();
        if r == 0 || r > m {
            return Err(format!(
                "SaeManifoldAtom::reduce_basis_to_subspace: invalid retained rank {r} (basis width {m})"
            ));
        }
        let Some(inner) = self.basis_second_jet.clone() else {
            return Err(
                "SaeManifoldAtom::reduce_basis_to_subspace: requires an analytic second-jet \
                 evaluator to compose the reduced jets"
                    .to_string(),
            );
        };
        let p = self.output_dim();
        let d = self.latent_dim;
        // Φ̃ = Φ Q  (n × r).
        let phi_red = self.basis_values.dot(q);
        // ∂Φ̃[:, :, a] = (∂Φ[:, :, a]) Q  for each latent axis a.
        let n = self.n_obs();
        let mut jac_red = Array3::<f64>::zeros((n, r, d));
        for axis in 0..d {
            let slice = self.basis_jacobian.slice(s![.., .., axis]).to_owned();
            let reduced = slice.dot(q);
            for row in 0..n {
                for col in 0..r {
                    jac_red[[row, col, axis]] = reduced[[row, col]];
                }
            }
        }
        // B̃ = Qᵀ B  (r × p): the minimum-norm pre-image onto range(Q).
        let dec_red = q.t().dot(&self.decoder_coefficients);
        if dec_red.dim() != (r, p) {
            return Err(format!(
                "SaeManifoldAtom::reduce_basis_to_subspace: reduced decoder dim {:?} != ({r}, {p})",
                dec_red.dim()
            ));
        }
        // S̃ = Qᵀ S Q  (r × r) on both the raw and the (re-derived) effective Gram.
        let s_raw_red = q.t().dot(&self.smooth_penalty_raw).dot(q);
        let order = smooth_penalty_nullity(&s_raw_red)?;
        let reduced_eval = SubspaceReducedEvaluator::new(inner, q.clone())?;
        let reduced_arc: Arc<dyn SaeBasisSecondJet> = Arc::new(reduced_eval);
        let base: Arc<dyn SaeBasisEvaluator> = reduced_arc.clone();

        // Transport the cached second jet through the SAME `M`-axis congruence
        // as `Φ` / `∂Φ`: `H̃[n, :, a, c] = H[n, :, a, c] · Q`, so the reduced
        // atom carries the intrinsic bending penalty on the reduced basis
        // immediately (before the next `refresh_basis`), matching the full-width
        // path — otherwise the `(r, r)` `smooth_penalty` seeded below would be
        // recomputed against a stale `(·, M, ·, ·)` cache. Dropped when absent.
        let second_jet_red = self.basis_second_jet_values.as_ref().map(|hess| {
            let mut reduced = Array4::<f64>::zeros((n, r, d, d));
            for a in 0..d {
                for c in 0..d {
                    let slab = hess.slice(s![.., .., a, c]).dot(q);
                    reduced.slice_mut(s![.., .., a, c]).assign(&slab);
                }
            }
            reduced
        });

        self.basis_values = phi_red;
        self.basis_jacobian = jac_red;
        self.basis_second_jet_values = second_jet_red;
        self.decoder_coefficients = dec_red;
        self.smooth_penalty_raw = s_raw_red.clone();
        // Seed the effective penalty with the reduced raw Gram so the buffer is
        // the right `(r, r)` shape; the arc-length refresh below overwrites it.
        self.smooth_penalty = s_raw_red;
        self.smooth_penalty_order = order;
        self.basis_evaluator = Some(base);
        self.basis_second_jet = Some(reduced_arc);
        // The decoder frame is a profiled representation of the *previous* M×p
        // decoder; the column count just changed, so drop it and let the joint
        // fit re-activate it for the reduced block if still profitable.
        self.decoder_frame = None;
        // Re-derive the intrinsic (pullback-metric / arc-length) reweighted
        // effective penalty on the REDUCED basis — exactly as the constructor
        // does for the full-width atom. Without this the reduced atom would
        // carry the bare `S̃ = Qᵀ S Q` while the full-width path carries the
        // arc-length-reweighted `W^{½} S W^{½}`, so a `latent_dim == 1` atom
        // with a genuine order-`r ≥ 1` (difference / Duchon) penalty would be
        // smoothed under a DIFFERENT roughness metric after reduction than
        // before — biasing exactly the rank-deficient circle #1117 targets.
        // (For the constant-speed periodic basis and order-0 / `latent_dim != 1`
        // atoms this is `S̃ = S̃_raw`, so the eye-penalty reductions are
        // byte-for-byte unchanged.) All inputs the refresh reads
        // (`basis_values`, `decoder_coefficients`, `smooth_penalty_raw`,
        // `smooth_penalty_order`, `basis_kind`, `latent_dim`) are now set.
        self.refresh_intrinsic_smooth_penalty();
        Ok(())
    }

    pub fn refresh_basis(&mut self, coords: ArrayView2<'_, f64>) -> Result<(), String> {
        // No installed evaluator means the caller is managing the basis
        // out-of-band (the construction-time `phi` / `jet` are authoritative).
        // The contract for that mode is documented in the constructor: the
        // caller takes responsibility for rebuilding the term after a
        // coordinate change. We must NOT fail here, because driver entry
        // points (`run_joint_fit_arrow_schur`, the inner Newton loop, …)
        // unconditionally call `refresh_basis_from_current_coords` to keep
        // the auto-refresh path correct, and that prelude has to pass through
        // unchanged for caller-managed atoms.
        let Some(evaluator) = self.basis_evaluator.as_ref() else {
            return Ok(());
        };
        // Curvature-homotopy dial (#1007): at the default `η = 1` this is the
        // un-dialed basis (`evaluate_phi_eta` returns the unscaled Φ / jet
        // bit-for-bit), so the production path is unchanged. For `η < 1` the
        // tracker scales the curved columns toward the base-topology relaxation; the
        // `dphi_deta` / `djet_deta` channels are discarded here (the predictor
        // forms `∂g/∂η` separately from a dedicated evaluation).
        let (phi, jet) = if self.homotopy_eta == 1.0 {
            evaluator.evaluate(coords)?
        } else {
            let evaluated = evaluator.evaluate_phi_eta(coords, self.homotopy_eta)?;
            (evaluated.phi, evaluated.jet)
        };
        if phi.dim() != self.basis_values.dim() {
            return Err(format!(
                "SaeManifoldAtom::refresh_basis: evaluator returned Phi {:?}, expected {:?}",
                phi.dim(),
                self.basis_values.dim()
            ));
        }
        if jet.dim() != self.basis_jacobian.dim() {
            return Err(format!(
                "SaeManifoldAtom::refresh_basis: evaluator returned jet {:?}, expected {:?}",
                jet.dim(),
                self.basis_jacobian.dim()
            ));
        }
        self.basis_values = phi;
        self.basis_jacobian = jet;
        // Cache the latent coordinates on the same coordinate change: the
        // Poincaré conformal-Dirichlet roughness Gram is a function of `t`
        // (through the exp-map pullback metric) and
        // `refresh_intrinsic_smooth_penalty` receives no coordinates. Frozen at
        // the current iterate exactly like the second-jet cache below.
        self.latent_coords = Some(coords.to_owned());
        // Refresh the cached second-jet VALUES on the same coordinate change so
        // the `d ≥ 2` intrinsic bending Gram sees `∂²Φ` at the current chart
        // (the lagged-diffusivity freeze the metric obeys). Only an evaluator
        // that exposes a closed-form Hessian can supply it; a first-jet-only /
        // caller-managed atom invalidates the cache to `None` so a stale jet
        // from a prior coordinate can never survive a refresh, and the bending
        // path then falls back to the raw Gram.
        // Lenient by contract: the second-jet cache is an OPTIONAL accelerant
        // for the `d ≥ 2` bending Gram, never load-bearing for `refresh_basis`
        // itself. A failing / mis-shaped second jet invalidates the cache to
        // `None` (bending then falls back to the raw Gram — no regression)
        // rather than failing the whole basis refresh, so an evaluator that is
        // fine for `Φ`/`∂Φ` but degenerate for `∂²Φ` at some coordinate cannot
        // break the refresh path.
        let n = self.n_obs();
        let m = self.basis_size();
        let d = self.latent_dim;
        if d >= 2 && n > BENDING_QUADRATURE_CAP && self.basis_second_jet.is_some() {
            // At scale the full `(n, M, d, d)` second jet is the OOM risk
            // (~1.5 GB/atom at n=10⁶). Evaluate it ONLY on a leverage-stratified
            // quadrature subsample; the Gram reweights by `n/|rows|`.
            let rows = self.select_bending_quadrature_rows(BENDING_QUADRATURE_CAP);
            let sub = coords.select(ndarray::Axis(0), &rows);
            let expected = (rows.len(), m, d, d);
            let jet = self
                .basis_second_jet
                .as_ref()
                .and_then(|second| second.second_jet(sub.view()).ok())
                .filter(|hess| hess.dim() == expected);
            if jet.is_some() {
                self.basis_second_jet_values = jet;
                self.bending_quadrature_rows = Some(rows);
            } else {
                // Sub-eval failed / mis-shaped: fall back to the raw-Gram path
                // (no cache) rather than a stale or full-size jet.
                self.basis_second_jet_values = None;
                self.bending_quadrature_rows = None;
            }
        } else {
            // Small-`n` or `d = 1`: full jet, bitwise-identical legacy path.
            let expected = (n, m, d, d);
            self.basis_second_jet_values = self
                .basis_second_jet
                .as_ref()
                .and_then(|second| second.second_jet(coords).ok())
                .filter(|hess| hess.dim() == expected);
            self.bending_quadrature_rows = None;
        }
        Ok(())
    }

    /// Install the cached second-jet VALUES `H[n, m, a, c] = ∂²Φ_m(t_n)`
    /// directly (shape `(N, M, d, d)`), for atoms whose basis is managed
    /// out-of-band (no installed evaluator) but that still want the `d ≥ 2`
    /// intrinsic bending Gram. Validates the shape against the current basis /
    /// latent dimensions. Real evaluator-backed atoms populate the same cache
    /// automatically in [`Self::refresh_basis`]; this is the caller-managed
    /// counterpart.
    pub fn install_bending_second_jet(&mut self, values: Array4<f64>) -> Result<(), String> {
        let expected = (self.n_obs(), self.basis_size(), self.latent_dim, self.latent_dim);
        if values.dim() != expected {
            return Err(format!(
                "SaeManifoldAtom::install_bending_second_jet: values {:?}, expected {expected:?}",
                values.dim()
            ));
        }
        self.basis_second_jet_values = Some(values);
        // A caller-installed jet is full-`n` by shape contract, so the Gram uses
        // the full active set (no quadrature reweight).
        self.bending_quadrature_rows = None;
        Ok(())
    }

    /// Leverage-stratified quadrature subsample for the `d ≥ 2` bending Gram.
    ///
    /// Returns ≤ `cap` row indices (all rows when `n ≤ cap`). The stratification
    /// key is each row's tangent energy `Σ_{m,a} (∂_aΦ_m)²` (a cheap
    /// `O(n·M·d)` proxy for the pullback volume `√det g`, read straight off the
    /// resident first-jet cache — no decoder pass, no full second jet). Rows are
    /// ranked by that key and `cap` are taken at evenly-spaced ranks, so the
    /// subsample spans the whole leverage spectrum (flat AND high-curvature
    /// regions) rather than clustering — the coverage the covariant Gram needs to
    /// see every bent direction. Deterministic.
    fn select_bending_quadrature_rows(&self, cap: usize) -> Vec<usize> {
        let n = self.n_obs();
        if n <= cap || cap == 0 {
            return (0..n).collect();
        }
        let m = self.basis_size();
        let d = self.latent_dim;
        let mut leverage = vec![0.0_f64; n];
        for (row, lev) in leverage.iter_mut().enumerate() {
            let mut acc = 0.0_f64;
            for coeff in 0..m {
                for a in 0..d {
                    let g = self.basis_jacobian[[row, coeff, a]];
                    acc += g * g;
                }
            }
            *lev = acc;
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| {
            leverage[i]
                .partial_cmp(&leverage[j])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut sel = Vec::with_capacity(cap);
        for k in 0..cap {
            let pos = (((k as f64 + 0.5) * n as f64 / cap as f64) as usize).min(n - 1);
            sel.push(order[pos]);
        }
        sel.sort_unstable();
        sel.dedup();
        sel
    }

    pub fn n_obs(&self) -> usize {
        self.basis_values.nrows()
    }

    pub fn basis_size(&self) -> usize {
        self.basis_values.ncols()
    }

    pub fn output_dim(&self) -> usize {
        self.decoder_coefficients.ncols()
    }

    /// Effective profiled frame rank `r` of this atom's decoder block in the
    /// arrow-Schur border (issue #972). `r == p` (full output dim) when no
    /// Grassmann frame is active — the historical full-`B` border width. When a
    /// frame is active the border holds only `M_k · r` coordinates.
    pub fn border_frame_rank(&self) -> usize {
        match &self.decoder_frame {
            Some(frame) => frame.rank(),
            None => self.output_dim(),
        }
    }

    /// Per-atom arrow-Schur border coefficient count: `M_k · r` when a frame is
    /// active (the factored width), else the full `M_k · p` (issue #972).
    pub fn border_coeff_count(&self) -> usize {
        self.basis_size() * self.border_frame_rank()
    }

    /// Grassmann manifold dimension `r·(p − r)` profiled OUT of the border for
    /// this atom (issue #972). `0` when no frame is active. This is the number
    /// of frame degrees of freedom that must enter the Laplace evidence
    /// dimension accounting (evidence honesty).
    pub fn frame_manifold_dimension(&self) -> usize {
        match &self.decoder_frame {
            Some(frame) => frame.manifold_dimension(),
            None => 0,
        }
    }

    /// Effective numerical column rank of the decoder `B_k` (`M_k × p`) from its
    /// singular values, with the relative cutoff [`SAE_FRAME_RANK_CUTOFF`]. This
    /// is the smallest frame rank `r` that captures `B_k`'s span up to that
    /// energy floor; the auto-activation heuristic compares it against `p`.
    pub fn decoder_numerical_rank(&self) -> Result<usize, String> {
        let p = self.output_dim();
        if p == 0 || self.basis_size() == 0 {
            return Ok(0);
        }
        let (_u, sv, _vt) = self
            .decoder_coefficients
            .svd(false, false)
            .map_err(|e| format!("SaeManifoldAtom::decoder_numerical_rank: SVD failed: {e}"))?;
        let max_sv = sv.iter().copied().fold(0.0_f64, f64::max);
        if !(max_sv > 0.0) {
            // A zero decoder has rank 0 but still needs a rank-1 frame so the
            // border carries a non-degenerate coordinate column.
            return Ok(0);
        }
        let tol = SAE_FRAME_RANK_CUTOFF * max_sv;
        Ok(sv.iter().filter(|&&v| v > tol).count())
    }

    /// #1610 — the GEOMETRICALLY REACHABLE output rank of this atom's decoded
    /// image, used to calibrate the co-collapse acceptance bar against what the
    /// dictionary can actually span (instead of the nominal coefficient count).
    ///
    /// The decoded image of the atom over its active rows is `Φ_k B_k`
    /// (`n × p`), whose column space lies inside `colspan(Φ_k) · B_k`. Its
    /// dimension is therefore bounded by `rank(Φ_k)` — the number of linearly
    /// independent directions the CHART (the realized basis evaluations on the
    /// current latent coordinates) produces, capped by the output dimension `p`.
    /// Crucially this is computed from the chart design `Φ_k` ALONE, **not** the
    /// decoder magnitude `B_k`: a curved `latent_dim = d` atom whose chart image
    /// is rank-deficient on the actual sample (few distinct coordinates, a
    /// degenerate chart, or a genuinely lower-dimensional realized image) reaches
    /// fewer linear directions than its nominal `basis_size()`, while a
    /// co-collapsed atom whose decoder norm → 0 still reports its full geometric
    /// reach (so the collapse guard keyed on this rank does NOT silently disable
    /// itself at the very state it must catch). `rank(Φ_k) ≤ basis_size()`
    /// always, so this can only LOWER the (linearly biased-high) PCA ceiling the
    /// collapse bar uses — the #1610 "nonlinear dict vs linear PCA ceiling" fix.
    ///
    /// Returns `0` for an empty/degenerate chart (no rows, no columns, or a
    /// zero/non-finite design); SVD failure is surfaced as an error so the
    /// caller can fall back to the nominal count rather than silently key on a
    /// meaningless rank.
    pub fn realized_chart_image_rank(&self) -> Result<usize, String> {
        let m = self.basis_size();
        let n = self.n_obs();
        let p = self.output_dim();
        if m == 0 || n == 0 || p == 0 {
            return Ok(0);
        }
        if !self.basis_values.iter().all(|v| v.is_finite()) {
            return Ok(0);
        }
        let (_u, sv, _vt) = self
            .basis_values
            .svd(false, false)
            .map_err(|e| format!("SaeManifoldAtom::realized_chart_image_rank: SVD failed: {e}"))?;
        let max_sv = sv.iter().copied().fold(0.0_f64, f64::max);
        if !(max_sv > 0.0) {
            return Ok(0);
        }
        let tol = SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF * max_sv;
        // The reachable output rank is bounded by both the chart's realized rank
        // and the output dimension `p` (a chart can never span more output
        // directions than the decoder has columns).
        Ok(sv.iter().filter(|&&v| v > tol).count().min(p))
    }

    /// Rank that should be carried by the low-rank Grassmann decoder frame for
    /// the current decoder, or `None` when the full-`B` representation is still
    /// the intended path. This is the exact activation predicate:
    ///
    /// * `r = max(numerical_rank(B_k), 1)`;
    /// * `r <= p * (1 - SAE_FRAME_ACTIVATION_MARGIN)`;
    /// * `p - r > 0`.
    ///
    /// Because `rank(B_k) <= M_k`, a cold LSQ decoder with `p >= 896` and
    /// `M_k <= 16` always satisfies the shrink predicate (`16 << 0.75p`) unless
    /// the decoder has no output dimension or no basis columns.
    pub fn decoder_frame_activation_rank(&self) -> Result<Option<usize>, String> {
        let p = self.output_dim();
        if p == 0 || self.basis_size() == 0 {
            return Ok(None);
        }
        if p < SAE_FRAME_MIN_AUTO_OUTPUT_DIM {
            return Ok(None);
        }
        let numerical_rank = self.decoder_numerical_rank()?;
        // A degenerate all-zero decoder keeps a rank-1 frame so the coordinate
        // column is non-empty; otherwise use the numerical rank.
        let r = numerical_rank.max(1).min(p);
        // Beneficial only if the frame materially shrinks the border AND there
        // is a positive Grassmann dimension to profile out.
        let shrink_ok = (r as f64) <= (p as f64) * (1.0 - SAE_FRAME_ACTIVATION_MARGIN);
        if !shrink_ok || p.saturating_sub(r) == 0 {
            return Ok(None);
        }
        Ok(Some(r))
    }

    /// Auto-derive whether the low-rank Grassmann factorization is beneficial for
    /// this atom and, if so, activate it (issue #972) — magic-by-default, no
    /// flag. The frame is installed (decoder factored as `B_k = C_k Uᵀ`) only
    /// when the decoder's effective rank `r` shrinks the per-atom border
    /// `M_k · p → M_k · r` by at least [`SAE_FRAME_ACTIVATION_MARGIN`] AND leaves
    /// a positive Grassmann dimension (`p − r ≥ 1`). Otherwise the atom stays on
    /// the bit-for-bit full-`B` path (`decoder_frame == None`).
    ///
    /// `B_k` is unchanged numerically: the installed frame spans exactly
    /// `range(B_kᵀ)` (the column space of the decoder) up to the truncation
    /// floor, so [`Self::reconstruct_decoder_coefficients`] recovers `B_k` to
    /// machine precision when `r` equals the true rank. Returns the activated
    /// frame rank, or `None` if the full-`B` path was kept.
    pub fn maybe_activate_decoder_frame(&mut self) -> Result<Option<usize>, String> {
        let Some(r) = self.decoder_frame_activation_rank()? else {
            self.decoder_frame = None;
            return Ok(None);
        };
        let p = self.output_dim();
        // Build the canonical frame from the decoder's own column-span evidence:
        // the cross-moment `B_kᵀ B_k`-induced left subspace is exactly the top-`r`
        // right-singular subspace of `B_k`. We obtain it by polaring the rank-`r`
        // truncation of the column cross-moment `B_kᵀ · (B_k · Vr)` — equivalently
        // the top-`r` right singular vectors of `B_k`. Use the SVD of `B_k`
        // directly: `B_k = W Σ Vᵀ` (W: M×?, Vᵀ: ?×p) ⇒ frame = top-`r` rows of `Vᵀ`
        // transposed = top-`r` columns of `V` (`p × r`).
        let (_w, sv, vt_opt) = self.decoder_coefficients.svd(false, true).map_err(|e| {
            format!("SaeManifoldAtom::maybe_activate_decoder_frame: SVD failed: {e}")
        })?;
        let vt = vt_opt.ok_or_else(|| {
            "SaeManifoldAtom::maybe_activate_decoder_frame: SVD returned no right factor"
                .to_string()
        })?;
        // `vt` is `min(M,p) × p`; take its top-`r` rows as the frame columns.
        let available = vt.nrows();
        let r_eff = r.min(available);
        if r_eff == 0 || p.saturating_sub(r_eff) == 0 {
            self.decoder_frame = None;
            return Ok(None);
        }
        let mut frame = Array2::<f64>::zeros((p, r_eff));
        for col in 0..r_eff {
            for row in 0..p {
                frame[[row, col]] = vt[[col, row]];
            }
        }
        let mut gauge = Array1::<f64>::zeros(r_eff);
        for i in 0..r_eff {
            gauge[i] = sv.get(i).copied().unwrap_or(0.0);
        }
        self.decoder_frame = Some(GrassmannFrame::from_oriented(frame, gauge));
        // Project the decoder onto the activated frame so the authoritative
        // `B_k = C_k U_kᵀ` holds EXACTLY from the first factored assembly
        // (issue #972 / #977 T1). Without this, `B_k` keeps its off-frame
        // component while the factored C-block solve only moves within
        // `range(U_k)`, leaving an irreducible residual the solver cannot
        // reduce — the fit then never converges. `B ← (B U) Uᵀ` is a no-op in
        // span for a truly rank-`r` decoder (the common, beneficial case).
        let u_proj = self
            .decoder_frame
            .as_ref()
            .expect("frame just set")
            .frame()
            .to_owned();
        let c_proj = self.decoder_coefficients.dot(&u_proj);
        self.decoder_coefficients = c_proj.dot(&u_proj.t());
        Ok(Some(r_eff))
    }

    /// Deactivate the Grassmann frame, returning this atom to the full-`B`
    /// border path (issue #972). `decoder_coefficients` already holds the
    /// reconstructed `B_k`, so no numerical change occurs.
    pub fn deactivate_decoder_frame(&mut self) {
        self.decoder_frame = None;
    }

    /// Coordinate matrix `C_k = B_k · U` (`M_k × r`) that the border stores when
    /// a frame is active (issue #972). Returns `None` on the full-`B` path.
    pub fn factored_coordinates(&self) -> Result<Option<Array2<f64>>, String> {
        match &self.decoder_frame {
            Some(frame) => Ok(Some(
                frame.project_decoder(self.decoder_coefficients.view())?,
            )),
            None => Ok(None),
        }
    }

    /// Reconstruct the full decoder `B_k = C_k · Uᵀ` from a border coordinate
    /// matrix `C_k` (`M_k × r`) and the active frame (issue #972). Used when the
    /// border solver returns updated coordinates and the authoritative
    /// `decoder_coefficients` must be refreshed for the full-`B` consumers.
    pub fn reconstruct_decoder_coefficients(
        &self,
        coords: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let frame = self.decoder_frame.as_ref().ok_or_else(|| {
            "SaeManifoldAtom::reconstruct_decoder_coefficients: no active frame".to_string()
        })?;
        frame.reconstruct_decoder(coords)
    }

    /// Install border coordinates `C_k` (`M_k × r`) returned by the factored
    /// border solve, refreshing `decoder_coefficients = C_k · Uᵀ` so all
    /// full-`B` consumers stay consistent with the profiled frame (issue #972).
    pub fn set_factored_coordinates(&mut self, coords: ArrayView2<'_, f64>) -> Result<(), String> {
        let reconstructed = self.reconstruct_decoder_coefficients(coords)?;
        if reconstructed.dim() != self.decoder_coefficients.dim() {
            return Err(format!(
                "SaeManifoldAtom::set_factored_coordinates: reconstructed decoder {:?} \
                 must match {:?}",
                reconstructed.dim(),
                self.decoder_coefficients.dim()
            ));
        }
        self.decoder_coefficients = reconstructed;
        Ok(())
    }

    /// Closed-form streaming polar refresh of the active frame from an
    /// accumulated `p × r` cross-moment (issue #972): `U ← polar(Mcm)`, then
    /// re-project the coordinates so `B_k` is unchanged in span. The frame
    /// update happens OUTSIDE the border; the coordinate matrix is re-derived by
    /// projection onto the new frame. No-op (error) when no frame is active.
    pub fn refresh_frame_from_cross_moment(
        &mut self,
        cross_moment: ArrayView2<'_, f64>,
    ) -> Result<(), String> {
        if self.decoder_frame.is_none() {
            return Err("SaeManifoldAtom::refresh_frame_from_cross_moment: no active frame".into());
        }
        let new_frame = GrassmannFrame::polar_update(cross_moment)?;
        if new_frame.output_dim() != self.output_dim() {
            return Err(format!(
                "SaeManifoldAtom::refresh_frame_from_cross_moment: frame output dim {} \
                 must equal decoder output dim {}",
                new_frame.output_dim(),
                self.output_dim()
            ));
        }
        // Re-express the current decoder in the new frame's coordinates, then
        // reconstruct `B_k` so its in-span component is carried forward exactly
        // and the out-of-span residual (orthogonal to the refreshed span) is
        // dropped — the streaming-polar fixed point.
        let coords = new_frame.project_decoder(self.decoder_coefficients.view())?;
        self.decoder_coefficients = new_frame.reconstruct_decoder(coords.view())?;
        self.decoder_frame = Some(new_frame);
        Ok(())
    }

    /// `g_k(t_{ik}) = Phi_k(t_{ik}) B_k`.
    pub fn decoded_row(&self, row: usize) -> Array1<f64> {
        let p = self.output_dim();
        let mut out = Array1::<f64>::zeros(p);
        self.fill_decoded_row(row, out.as_slice_mut().expect("contiguous"));
        out
    }

    /// In-place fill of `g_k(t_{ik})` into a caller-supplied buffer of length `p`.
    /// Hot-loop variant used by the arrow-Schur assembly to avoid per-row
    /// allocations.
    pub fn fill_decoded_row(&self, row: usize, out: &mut [f64]) {
        let p = self.output_dim();
        let m = self.basis_size();
        assert_eq!(out.len(), p);
        for slot in out.iter_mut() {
            *slot = 0.0;
        }
        for basis_col in 0..m {
            let phi = self.basis_values[[row, basis_col]];
            if phi == 0.0 {
                continue;
            }
            // Row `basis_col` of the (M×p) decoder is contiguous; iterate it as a
            // slice-backed view so the axpy has no per-element 2-D index recompute
            // or bounds check and autovectorizes (hot: per-row × per-atom).
            let dec = self.decoder_coefficients.row(basis_col);
            for (o, &d) in out.iter_mut().zip(dec.iter()) {
                *o += phi * d;
            }
        }
        // #2022 — apply the explicit log-amplitude: contribution is exp(s)·Φ·B.
        // Skipped when s == 0.0 (default) so the historical hot loop is
        // bit-for-bit; a non-zero s scales the whole decoded row.
        if self.log_amplitude != 0.0 {
            let amp = self.log_amplitude.exp();
            for slot in out.iter_mut() {
                *slot *= amp;
            }
        }
    }

    /// `d g_k(t_{ik}) / d t_{ik,j}` for one row and latent axis.
    pub fn decoded_derivative_row(&self, row: usize, latent_axis: usize) -> Array1<f64> {
        let p = self.output_dim();
        let mut out = Array1::<f64>::zeros(p);
        self.fill_decoded_derivative_row(row, latent_axis, out.as_slice_mut().expect("contiguous"));
        out
    }

    /// In-place fill of `d g_k / d t_{ik,axis}` into a caller-supplied buffer of
    /// length `p`. Hot-loop variant used by the arrow-Schur assembly.
    pub fn fill_decoded_derivative_row(&self, row: usize, latent_axis: usize, out: &mut [f64]) {
        let p = self.output_dim();
        let m = self.basis_size();
        assert_eq!(out.len(), p);
        for slot in out.iter_mut() {
            *slot = 0.0;
        }
        for basis_col in 0..m {
            let dphi = self.basis_jacobian[[row, basis_col, latent_axis]];
            if dphi == 0.0 {
                continue;
            }
            let dec = self.decoder_coefficients.row(basis_col);
            for (o, &d) in out.iter_mut().zip(dec.iter()) {
                *o += dphi * d;
            }
        }
        // #2022 — the reconstruction is exp(s)·Φ·B, so its coordinate derivative
        // is exp(s)·(dΦ·B). Skipped when s == 0.0 (default) ⇒ bit-for-bit.
        if self.log_amplitude != 0.0 {
            let amp = self.log_amplitude.exp();
            for slot in out.iter_mut() {
                *slot *= amp;
            }
        }
    }

    /// #1026 — `∂²g_k/∂t_{ik,axis}∂η` for one row/axis, restricted to the curved
    /// basis columns. Because the η-dial scales exactly the curved columns
    /// (`∂Φ^η/∂η = Φ_curved`), the η-derivative of the coordinate Jacobian
    /// `∂(∂Φ/∂t·B)/∂η` is the SAME coordinate-Jacobian contraction summed over
    /// only the curved columns. This is the coordinate-channel analog of the
    /// β-predictor's `curvature_basis_eta_derivatives`, and supplies the missing
    /// `w_t = ∂g_t/∂η` forcing that lets the homotopy walk track onto the curved
    /// branch instead of riding the base-topology shadow. `curved_cols` are the
    /// atom's `phi_eta_split` curved column indices; a base-only atom (no dialed
    /// columns) writes zeros.
    pub fn fill_decoded_curved_derivative_row(
        &self,
        row: usize,
        latent_axis: usize,
        curved_cols: &[usize],
        out: &mut [f64],
    ) {
        let p = self.output_dim();
        assert_eq!(out.len(), p);
        for slot in out.iter_mut() {
            *slot = 0.0;
        }
        for &basis_col in curved_cols {
            let dphi = self.basis_jacobian[[row, basis_col, latent_axis]];
            if dphi == 0.0 {
                continue;
            }
            let dec = self.decoder_coefficients.row(basis_col);
            for (o, &d) in out.iter_mut().zip(dec.iter()) {
                *o += dphi * d;
            }
        }
        // #2022 — same exp(s) scaling as the value/derivative primitives so the
        // curved η-derivative stays consistent. No-op at s == 0.0 (default).
        if self.log_amplitude != 0.0 {
            let amp = self.log_amplitude.exp();
            for slot in out.iter_mut() {
                *slot *= amp;
            }
        }
    }

    /// #2022 — fold the decoder's Frobenius magnitude into the explicit
    /// log-amplitude: `s_k ← s_k + ln‖B_k‖_F` and `B_k ← B_k / ‖B_k‖_F`, leaving
    /// the atom's contribution `exp(s_k)·Φ·B_k` numerically UNCHANGED. This is
    /// the representation half of the SCALE-gauge removal — after peeling,
    /// magnitude lives only in `s_k` and `B_k` is a unit-Frobenius shape frame
    /// (the invariant the STEP 2 Stiefel retract maintains). A decoder at/under
    /// `floor` (or non-finite norm) is treated as collapsed and left untouched
    /// (no `ln 0`); pass e.g. `f64::MIN_POSITIVE` for "skip only an exactly-zero
    /// decoder".
    pub fn absorb_decoder_norm_into_log_amplitude(&mut self, floor: f64) {
        let norm = self
            .decoder_coefficients
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        if !(norm.is_finite() && norm > floor) {
            return;
        }
        self.log_amplitude += norm.ln();
        self.decoder_coefficients.mapv_inplace(|v| v / norm);
        // Keep the pullback-metric-reweighted roughness Gram consistent with the
        // normalized decoder. It is magnitude-blind by design (#673), so this is
        // a no-op in exact arithmetic — refreshed for defensive consistency.
        self.refresh_intrinsic_smooth_penalty();
    }

    /// #2022 — like [`Self::absorb_decoder_norm_into_log_amplitude`] but SKIPS the
    /// [`Self::refresh_intrinsic_smooth_penalty`] recompute, so a caller that has
    /// already installed a specific `smooth_penalty` keeps it. Used at the
    /// basis-change TRANSPORT / rank-reparam sites, whose penalty is set by
    /// [`transport_smooth_penalty_for_decoder`](crate::manifold::outer_objective::transport_smooth_penalty_for_decoder)
    /// = `T⁻ᵀ S_old T⁻¹` — a function of the basis transport and the OLD penalty
    /// ONLY, independent of the decoder's magnitude/values — so normalizing the
    /// decoder here cannot invalidate it, and refreshing would clobber it. A
    /// decoder at/under `floor` (or non-finite norm) is left untouched.
    pub fn absorb_decoder_norm_into_log_amplitude_without_refresh(&mut self, floor: f64) {
        let norm = self
            .decoder_coefficients
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        if !(norm.is_finite() && norm > floor) {
            return;
        }
        self.log_amplitude += norm.ln();
        self.decoder_coefficients.mapv_inplace(|v| v / norm);
        // Deliberately NOT refreshing smooth_penalty — the caller's transported
        // penalty is decoder-magnitude-independent and must survive the peel.
    }

    /// Recompute the intrinsic (gauge-invariant) roughness Gram
    /// [`Self::smooth_penalty`] from [`Self::smooth_penalty_raw`], the current
    /// basis Jacobian / second jet, and the current decoder coefficients
    /// (issue #673).
    ///
    /// The raw penalty `0.5·λ·tr(BᵀS B)` measures roughness per unit of the raw
    /// latent coordinate `t`, so it is *not* invariant under reparameterizing
    /// `t` — and the model evidence that ranks an atom's topology (circle vs
    /// line) inherits that gauge dependence. The gauge-invariant target is the
    /// roughness of the decoded FUNCTION itself, read off its intrinsic geometry
    /// rather than the chart (SPEC: penalise the final function, not the
    /// coefficients). Two closed forms realise it, dispatched on the manifold:
    ///
    /// * **Poincaré** — the hyperbolic conformal Dirichlet energy
    ///   `E_g[f] = ∫ gᵃᵇ ∂_a f ∂_b f dμ_g` measured in the ball metric pulled
    ///   back to the tangent chart, delegated to
    ///   [`gam_geometry::manifolds::poincare::conformal_dirichlet_penalty`]
    ///   (`curvature = −1`, the single source of truth for the hyperbolic
    ///   metric). Needs the cached [`Self::latent_coords`] and the first-jet
    ///   [`Self::basis_jacobian`]; absent coordinates fall back to the raw Gram.
    ///   This is an ORDER-1 (Dirichlet) operator — a different order than the
    ///   atom's second-derivative `smooth_penalty_raw` — but the effective Gram
    ///   is installed DIRECTLY into `smooth_penalty`, so every downstream
    ///   consumer (the REML rank / log-det Occam term, the Kronecker Hessian)
    ///   reads the correct order off the matrix itself; no separate accounting is
    ///   needed. See [`Self::try_poincare_conformal_gram`].
    ///
    /// * **All other manifolds** — the total squared SECOND FUNDAMENTAL FORM
    ///   `∫_M ‖II‖²_g dμ` of the decoded embedding `γ(t) = BᵀΦ(t)`, delegated to
    ///   [`Self::try_intrinsic_bending_gram`] for EVERY latent dim `d ≥ 1`. For
    ///   `d = 1` this is exactly the reparameterisation-invariant curve bending
    ///   `∫ κ² ds = Σ_i ‖P_N γ''(t_i)‖² / ‖γ'(t_i)‖³` — the NORMAL-projected
    ///   acceleration weighted by the arc-length element `ds = ‖γ'‖ dt`, which
    ///   zeroes the tangential part a chart reparameterisation manufactures (e.g.
    ///   `γ(u) = u²e₁` and `γ(s) = s e₁` are the SAME straight segment and both
    ///   score zero, whereas the old raw order-2 Gram charged `γ(u)'' = 2e₁ > 0`
    ///   spuriously). For `d ≥ 2` it is the Riemannian-volume-weighted `‖II‖²_g`.
    ///   It needs the cached analytic second jet
    ///   ([`Self::basis_second_jet_values`]); if that is absent (a caller-managed
    ///   / first-jet-only atom) the raw Gram is left untouched, so no atom
    ///   regresses relative to the old flat fallback.
    ///
    /// A degenerate (empty/zero) raw Gram leaves `S̃ = S` untouched.
    ///
    /// The intrinsic geometry `(g, Γ)` / latent coordinates are FROZEN at the
    /// current iterate (lagged-diffusivity / IRLS surrogate): within one inner
    /// solve the penalty stays a fixed quadratic Gram, and refreshing between
    /// assemblies makes the *converged* penalty the true intrinsic roughness. A
    /// geodesic (zero-bending) image — the periodic sin/cos basis on `S¹`, a
    /// straight decoder — carries no intrinsic roughness, so no overall magnitude
    /// (which `λ` already owns) leaks into the penalty.
    ///
    /// # Curvature is identifiability (Scott / Thm B)
    ///
    /// Superposition ambiguity is a FLATNESS disease and curvature is its cure.
    /// Two atoms that share a decoded image but differ by a chart
    /// reparameterisation are indistinguishable to a *flat* (raw-`t`) roughness
    /// measure; the intrinsic bending penalty is precisely what makes the atom's
    /// osculating flag — its second-order contact with the ambient space, hence
    /// its gauge rigidity — MEASURABLE. By Thm B (jet transversality) the
    /// second-order contact of generic embeddings is of infinite codimension:
    /// generic curvature breaks the flat degeneracy almost surely, so a penalty
    /// that reads genuine curvature (and refuses to charge the affine — zero-II
    /// — maps, the exact flat ambiguity it must NOT penalise; see the invariant
    /// `(d+1)·p` null space in [`Self::try_intrinsic_bending_gram`]) is what
    /// turns an unidentified superposition into an identified manifold.
    pub fn refresh_intrinsic_smooth_penalty(&mut self) {
        let m = self.basis_size();
        if m == 0 {
            self.smooth_penalty.assign(&self.smooth_penalty_raw);
            return;
        }
        // `d ≥ 2`: the gauge-invariant target is the total squared second
        // fundamental form, not the raw coefficient-`t` Gram. Delegate to the
        // bending builder; fall back to the raw Gram only when the second jet is
        // unavailable or the geometry is degenerate (no regression vs the old
        // flat-`d>1` fallback).
        if self.latent_dim >= 2 {
            match self.try_intrinsic_bending_gram() {
                Some(gram) => self.smooth_penalty.assign(&gram),
                None => self.smooth_penalty.assign(&self.smooth_penalty_raw),
            }
            return;
        }
        // `d = 1` scalar-speed arc-length reweighting (unchanged). No reweighting
        // when there is no penalty operator order to invert into arc length.
        if self.smooth_penalty_order == 0 {
            self.smooth_penalty.assign(&self.smooth_penalty_raw);
            return;
        }
        let n = self.n_obs();
        let p = self.output_dim();
        let beta = 0.5 - self.smooth_penalty_order as f64;

        // Per-sample squared speed m_n = ‖J(t_n)‖², J(t_n) = Φ'(t_n) B (axis 0,
        // the single latent axis), and the basis-activation accumulators
        // act_μ = Σ_n Φ_μ(t_n)² and num_μ = Σ_n Φ_μ(t_n)² m_n.
        let mut act = vec![0.0_f64; m];
        let mut num = vec![0.0_f64; m];
        let mut deriv = vec![0.0_f64; p];
        // Poincaré tangent patch (`d = 1`): the latent coordinate `t` is a
        // tangent vector at the ball origin, and it runs at a *constant* multiple
        // of hyperbolic arc length. The ball point is `p = exp₀(t)` with
        // `‖p‖ = tanh|t|` (curvature `c = −1`), so `dp/dt = sech²(t)` and the
        // arc-length rate is `λ(p)·|dp/dt| = 2cosh²(t)·sech²(t) = 2`, independent
        // of `t` (the geodesic distance from the origin is `2|t|`). The decoded
        // speed per unit arc length is therefore `‖J‖ / 2` and the intrinsic
        // squared speed is `‖J‖² / 4` — a *constant* multiple of the flat-`t`
        // squared speed. The *key* property — a metric weight that is CONSTANT in
        // `t`, not `t`-dependent — is the same one the authoritative pullback
        // `gam_geometry::manifolds::poincare::conformal_dirichlet_penalty` states
        // for `d = 1` (its constant `G ≡ 1/2`): the chart is intrinsically flat in
        // 1-D. (The two numeric constants differ — `1/4 = (dt/ds)²` here is the
        // scalar reparam factor for squared speed, whereas `G = √det h · h⁻¹` is
        // that module's matrix Dirichlet weight — but both are constant in `t`,
        // which is all that matters below.)
        //
        // The earlier code divided by `λ(p)² = 4cosh⁴(t)`, conflating the *ball*
        // conformal factor with the *tangent* arc-length rate. That is
        // `t`-dependent and under-penalised near-boundary roughness by `~cosh⁴(t)`
        // (~200× at `|t| = 2`, ~10⁴× at `|t| = 3`), manufacturing fake curvature
        // in a chart that is flat for `d = 1` and corrupting the topology /
        // evidence / smoothness selection for hyperbolic atoms.
        //
        // The geometric-mean centering below divides `speeds` by their center, so
        // the constant `1/4` cancels exactly and does not move the numbers; it is
        // applied so `speeds` carries genuine intrinsic squared speed and the
        // `d = 1` Poincaré reweighting is provably the flat arc-length reweighting
        // (no spurious `t`-dependence). A `latent_dim > 1` Poincaré atom needs the
        // non-constant matrix pullback and never reaches this scalar path — the
        // `latent_dim != 1` early return above leaves it at the raw Gram.
        //
        // Convention note (why `1/4` here, not `1/2`): the authoritative
        // `poincare.rs::conformal_dirichlet_penalty` reports `G ≡ 1/2` for `d = 1`
        // in the *energy-density* convention `∫ G ‖dg/dt‖² dt`, whose `G` absorbs
        // the arc-length measure `ds = 2 dt`. This path instead reweights the
        // *pointwise* squared speed, whose intrinsic factor is `(dt/ds)² = 1/4`.
        // Both are correct in their own convention, and — crucially — both are
        // numerically inert below because the geometric-mean centering cancels any
        // constant. The load-bearing fix is not the value of this constant but
        // that the reweighting is now CONSTANT in `t` (vs the old per-sample
        // `λ⁻²`).
        //
        // TRAILHEAD (Poincaré true hyperbolic Gram, secondary of the Superposed-
        // Geometry batch): this `d = 1` path reweights the EUCLIDEAN monomial
        // second-derivative raw Gram by a constant, which is correct for the
        // intrinsically-flat `d = 1` tangent chart but is NOT the documented
        // hyperbolic pullback roughness. The authoritative Gram is
        // `gam_geometry::manifolds::poincare::conformal_dirichlet_penalty(coords,
        // basis_jacobian, curvature = -1.0)` — an ORDER-1 (Dirichlet) energy
        // `S = Σ_n Φ'ᵀ G(t_n) Φ'`, `G = √det h · h⁻¹` (`G ≡ 1/2` for `d = 1`,
        // the anisotropic `Dexp₀ᵀλ²Dexp₀` matrix for `d ≥ 2`). Wiring it here is
        // deferred, not trivial, for two coupled reasons:
        //   1. It needs the latent `coords`, which `refresh_intrinsic_smooth_
        //      penalty` does not receive. Route them the SAME way the `d ≥ 2`
        //      bending path does — cache them alongside `basis_second_jet_values`
        //      in `refresh_basis` (add a `poincare_coords: Option<Array2<f64>>`
        //      field), so the frozen-metric contract is preserved.
        //   2. It is a DIFFERENT operator order (Dirichlet, `r = 1`) than the
        //      atom's second-derivative raw Gram (`r = 2`), so adopting it must
        //      also update `smooth_penalty_order` and re-derive the REML
        //      rank/log-det Occam accounting — otherwise the evidence term is
        //      inconsistent. Prefer installing it as a NEW `smooth_penalty_raw`
        //      at construction (so `smooth_penalty_nullity` recovers `r = 1`
        //      once) rather than reweighting inside this refresh.
        // Until then the constant reweighting below is the honest `d = 1`
        // limit (flat chart ⇒ constant `G`), and `d ≥ 2` Poincaré atoms are
        // handled by the bending Gram above (which reads the true metric `g`
        // from the decoder pullback, not from `conformal_dirichlet_penalty`).
        const POINCARE_D1_ARCLEN_RATE_SQ: f64 = 4.0;
        let hyperbolic = matches!(self.basis_kind, SaeAtomBasisKind::Poincare);
        for row in 0..n {
            self.fill_decoded_derivative_row(row, 0, &mut deriv);
            let mut speed_sq = 0.0_f64;
            for &d in deriv.iter() {
                speed_sq += d * d;
            }
            if hyperbolic {
                // Only reached for `d = 1` (the `latent_dim != 1` early return
                // above skips this whole routine for `d > 1`, leaving `S̃ = S_raw`
                // — the safe constant-speed limit, pending a proper non-constant
                // `Dexp₀ᵀ λ² Dexp₀` matrix pullback for `d > 1`). Constant `1/4`
                // from the `d = 1` arc-length rate `2` (see above): intrinsic
                // squared speed = `‖J‖² / rate²`.
                speed_sq /= POINCARE_D1_ARCLEN_RATE_SQ;
            }
            // Row `row` of the (N×M) basis design is contiguous; read it once as
            // a 1-D view so the per-coefficient accumulation below has no 2-D
            // index recompute (n-hot: one pass per sample × per atom).
            let phi_row = self.basis_values.row(row);
            for (col, &phi) in phi_row.iter().enumerate() {
                let w = phi * phi;
                if w == 0.0 {
                    continue;
                }
                act[col] += w;
                num[col] += w * speed_sq;
            }
        }

        // Representative squared speed per coefficient, and the geometric-mean
        // center of the finite positive speeds. Only finite positive speeds
        // enter the center so a degenerate (inf/NaN) sample cannot corrupt it.
        let mut speeds = vec![0.0_f64; m];
        let mut log_acc = 0.0_f64;
        let mut log_cnt = 0usize;
        for col in 0..m {
            let s = if act[col] > 0.0 {
                num[col] / act[col]
            } else {
                0.0
            };
            speeds[col] = s;
            if s > 0.0 && s.is_finite() {
                log_acc += s.ln();
                log_cnt += 1;
            }
        }
        let center = if log_cnt > 0 {
            (log_acc / log_cnt as f64).exp()
        } else {
            0.0
        };
        // Degenerate curve (no finite positive speed anywhere, or a non-finite
        // center): the pullback metric carries no usable scale, so leave the
        // penalty at its raw Gram — exactly `S̃ = S_raw`, matching the
        // constant-speed limit with no spurious magnitude inflation.
        if !(center > 0.0 && center.is_finite()) {
            self.smooth_penalty.assign(&self.smooth_penalty_raw);
            return;
        }

        // Reweight relative to the center so the congruence is a *scale-free*
        // shape reweighting: the geometric mean of `w_μ` is 1, so a
        // constant-speed atom (every `s_μ = center`) gives `w_μ ≡ 1` and hence
        // `S̃ = S_raw` exactly — periodic atoms are untouched and no overall
        // magnitude (which `λ` already owns) leaks in. The relative floor keeps
        // a vanishing-speed coefficient at a small fraction of the typical
        // speed rather than a singular negative power, and clamps any non-finite
        // ratio back to a finite weight.
        const RELATIVE_SPEED_FLOOR: f64 = 1.0e-6;
        const RELATIVE_SPEED_CEIL: f64 = 1.0e6;
        let mut root_w = vec![0.0_f64; m];
        for col in 0..m {
            // Normalised squared speed (ratio to the geometric-mean center),
            // clamped to `[1e-6, 1e6]` so a vanishing-/diverging-speed
            // coefficient is treated as a bounded fraction/multiple of the
            // typical speed rather than a singular negative power, and any
            // non-finite ratio (e.g. an overflowed speed) maps to the ceiling.
            // The symmetric clamp keeps every weight finite and centered near 1
            // so the REML numerical-rank eigencutoff cannot drift.
            let ratio = speeds[col] / center;
            let ratio = if ratio.is_finite() {
                ratio.clamp(RELATIVE_SPEED_FLOOR, RELATIVE_SPEED_CEIL)
            } else {
                RELATIVE_SPEED_CEIL
            };
            // w_μ = ratio^β; the congruence uses W^{½}, so store ratio^{β/2}.
            root_w[col] = ratio.powf(0.5 * beta);
        }

        // S̃ = W^{½} S_raw W^{½}: scale row i and column j by root_w.
        for i in 0..m {
            let ri = root_w[i];
            for j in 0..m {
                self.smooth_penalty[[i, j]] = ri * self.smooth_penalty_raw[[i, j]] * root_w[j];
            }
        }
    }

    /// Build the `d ≥ 2` gauge-invariant intrinsic bending Gram `S̃` (`M × M`),
    /// whose decoder trace is exactly the total squared second fundamental form
    ///   `tr(Bᵀ S̃ B) = Σ_i ω_i ‖II(t_i)‖²_g`.
    ///
    /// # Geometry (Gauss formula — no Christoffel-from-metric-derivatives)
    ///
    /// The atom's decoded embedding is `γ(t) = Bᵀ Φ(t) ∈ ℝ^p` (one physical
    /// point per latent `t`), with tangent frame `∂_a γ = Bᵀ ∂_a Φ`, pullback
    /// metric `g_{ab} = ⟨∂_a γ, ∂_b γ⟩` (`g = JᵀJ`, `J = Bᵀ ∂Φ`), and ambient
    /// Hessian `∂²_{ab} γ = Bᵀ ∂²_{ab} Φ`. The second fundamental form is the
    /// NORMAL part of that ambient Hessian — equivalently the metric covariant
    /// Hessian of the embedding, which for an isometric immersion is ALREADY
    /// normal:
    ///   `II_{ab} = (I − J g⁺ Jᵀ) ∂²_{ab} γ = ∂²_{ab} γ − Γ^c_{ab} ∂_c γ`,
    /// with the Levi-Civita symbols read straight off the embedding,
    ///   `Γ^c_{ab} = g^{cd} ⟨∂_d γ, ∂²_{ab} γ⟩`.
    /// The embedding hands us `(g, Γ)` directly — no differentiating the metric
    /// — and NO `∂³Φ` enters, so this is exactly a curvature (order-2) object.
    /// `(g, Γ)` are FROZEN at the current iterate (the same lagged-diffusivity
    /// contract as the `d = 1` scalar-speed path): the penalty is a fixed
    /// quadratic Gram within one inner solve, refreshed between assemblies so
    /// the CONVERGED penalty is the true intrinsic bending energy.
    ///
    /// # Coefficient-space realisation (exact, PSD by construction)
    ///
    /// Define the COVARIANT second-jet design in coefficient space
    ///   `Ψ_{ab} = ∂²_{ab} Φ − Γ^c_{ab} ∂_c Φ ∈ ℝ^M`,
    /// so that `Bᵀ Ψ_{ab} = II_{ab}` EXACTLY (the decoder pulls the coefficient
    /// design back to the ambient normal vector). Whitening the metric
    /// contraction via the eigen-square-root `g⁺ = R Rᵀ` (`R_{ak} = u_{ak}/√σ_k`
    /// on the numerically nonzero spectrum — a pseudo-inverse onto the REALISED
    /// tangent space, which is also the projector inside `II`) and stacking the
    /// whitened symmetric pairs `Ψ̃_{ef} = Σ_{ab} R_{ae} R_{bf} Ψ_{ab}` gives
    ///   `S̃ = Σ_i ω_i Σ_{e,f} Ψ̃_{i,ef} Ψ̃_{i,ef}ᵀ`,
    /// manifestly `⪰ 0` (a nonneg-weighted sum of rank-1 outer products). Since
    /// `Σ_e R_{ae} R_{ce} = g^{ac}`,
    ///   `tr(Bᵀ S̃ B) = Σ_i ω_i Σ_{abcd} g^{ac} g^{bd} ⟨II_ab, II_cd⟩
    ///               = Σ_i ω_i ‖II(t_i)‖²_g`,
    /// the target energy to machine precision (the whitening is an exact
    /// algebraic refactor of the `g⁻¹⊗g⁻¹` contraction, not an approximation).
    ///
    /// # Invariant null space — the flat ambiguity the penalty must NOT charge
    ///
    /// `S̃ v = 0` iff the scalar chart function `Φ v` has vanishing covariant
    /// Hessian at every grid point, i.e. it is AFFINE in the (frozen) intrinsic
    /// geometry — `γ = c + A t` in normal coordinates. That kernel has dimension
    /// `d + 1` in `M`-space (the constant plus `d` linear directions), so the
    /// decoder form `B ↦ tr(Bᵀ S̃ B)` has null space of dimension `(d+1)·p`: the
    /// penalty annihilates EXACTLY the affine (zero-curvature) embeddings in ANY
    /// chart and charges everything else its true bending energy. This is the
    /// correct invariant null space: the flat superposition ambiguity is free,
    /// curvature is not — curvature is identifiability.
    ///
    /// # Measure
    ///
    /// `ω_i` is [`Self::bending_measure`]: the DEFAULT Riemannian volume element
    /// `√det g_i` (a local-VOLUME-weighted quadrature — a chart region of
    /// near-zero volume is CHEAP to bend, so the penalty budget is not spent
    /// keeping flat a region that carries little volume/data; this weighting is
    /// NOT sampling-density invariant on its own, see [`SaeBendingMeasure`]), or
    /// the counting measure `ω_i = 1`. A grid point with a rank-deficient
    /// tangent frame has `det g = 0` and contributes nothing under the volume
    /// measure.
    ///
    /// Returns `None` (raw-Gram fallback) when the second-jet cache is absent or
    /// mis-shaped, or the accumulated Gram is non-finite — never a regression
    /// versus the historical flat-`d>1` fallback.
    fn try_intrinsic_bending_gram(&self) -> Option<Array2<f64>> {
        let d = self.latent_dim;
        let m = self.basis_size();
        let n = self.n_obs();
        let p = self.output_dim();
        if d < 2 || m == 0 || n == 0 || p == 0 {
            return None;
        }
        let hess = self.basis_second_jet_values.as_ref()?;
        // Quadrature-subsampled at scale: when `bending_quadrature_rows` is set
        // the cache is the COMPACT jet over those rows and each row's Gram
        // contribution is reweighted by `n / |rows|` so the penalty scale (hence
        // λ selection) matches the full-set penalty. Full-set path (`None`) is
        // bitwise-legacy: rows = 0..n, weight 1, hess shape (n, M, d, d).
        let rows_owned: Vec<usize>;
        let rows: &[usize] = match &self.bending_quadrature_rows {
            Some(r) => r.as_slice(),
            None => {
                rows_owned = (0..n).collect();
                &rows_owned
            }
        };
        if hess.dim() != (rows.len(), m, d, d) {
            return None;
        }
        let weight_scale = if self.bending_quadrature_rows.is_some() {
            n as f64 / rows.len().max(1) as f64
        } else {
            1.0
        };
        let b = &self.decoder_coefficients; // (M, p)
        let volume = matches!(self.bending_measure, SaeBendingMeasure::Volume);

        let mut gram = Array2::<f64>::zeros((m, m));
        for (jet_idx, &row) in rows.iter().enumerate() {
            // Ambient tangent frame Dγ (p × d): Dγ[:, a] = Bᵀ ∂_aΦ.
            let mut dgamma = Array2::<f64>::zeros((p, d));
            for a in 0..d {
                let dphi_a = self.basis_jacobian.slice(s![row, .., a]); // (M)
                let mut col = dgamma.column_mut(a);
                for coeff in 0..m {
                    let w = dphi_a[coeff];
                    if w == 0.0 {
                        continue;
                    }
                    let brow = b.row(coeff);
                    for l in 0..p {
                        col[l] += w * brow[l];
                    }
                }
            }
            // Pullback metric g = DγᵀDγ (d × d, symmetric PSD).
            let mut g = Array2::<f64>::zeros((d, d));
            for a in 0..d {
                for bb in 0..d {
                    let mut acc = 0.0_f64;
                    for l in 0..p {
                        acc += dgamma[[l, a]] * dgamma[[l, bb]];
                    }
                    g[[a, bb]] = acc;
                }
            }
            // Symmetric eigendecomposition (d is tiny). A degenerate frame
            // (no positive eigenvalue) has no tangent space / zero volume — it
            // charges no bending and is skipped.
            let (evals, evecs) = g.eigh(Side::Lower).ok()?;
            let max_eig = evals.iter().cloned().fold(0.0_f64, f64::max);
            if !(max_eig > 0.0 && max_eig.is_finite()) {
                continue;
            }
            let tol = 1.0e-12 * max_eig;
            // Whitening root R (RRᵀ = g⁺), pseudo-inverse g⁺, and log det g on
            // the numerically-nonzero spectrum. R spans exactly the realised
            // tangent space, so the null directions of g never enter II or the
            // energy (the correct pseudo-inverse projector).
            let mut rmat = Array2::<f64>::zeros((d, d));
            let mut ginv = Array2::<f64>::zeros((d, d));
            let mut logdet = 0.0_f64;
            let mut full_rank = true;
            for k in 0..d {
                let sigma = evals[k];
                if sigma > tol {
                    let inv_sqrt = 1.0 / sigma.sqrt();
                    let inv = 1.0 / sigma;
                    for a in 0..d {
                        rmat[[a, k]] = evecs[[a, k]] * inv_sqrt;
                    }
                    for a in 0..d {
                        for c in 0..d {
                            ginv[[a, c]] += evecs[[a, k]] * evecs[[c, k]] * inv;
                        }
                    }
                    logdet += sigma.ln();
                } else {
                    full_rank = false;
                }
            }
            // Volume weight ω = √det g (0 on a rank-deficient frame); or ω = 1.
            // The quadrature reweight `n/|rows|` folds in here so a subsampled
            // Gram estimates the full-set Gram (its trace = Σ_i ω_i‖II_i‖²).
            let omega = weight_scale
                * if volume {
                    if full_rank {
                        (0.5 * logdet).exp()
                    } else {
                        0.0
                    }
                } else {
                    1.0
                };
            if !(omega > 0.0 && omega.is_finite()) {
                continue;
            }
            let sqrt_omega = omega.sqrt();

            // Covariant second-jet design Ψ_{ab} ∈ ℝ^M for every ordered pair,
            // stored (d, d, M). Ψ_{ab} = ∂²Φ_ab − Σ_c Γ^c_ab ∂_cΦ with
            // Γ^c_ab = Σ_e g⁺_{ce} ⟨∂_eγ, ∂²γ_ab⟩ and ∂²γ_ab = Bᵀ ∂²Φ_ab.
            let mut psi = Array3::<f64>::zeros((d, d, m));
            let mut d2gamma = vec![0.0_f64; p];
            for a in 0..d {
                for bb in 0..d {
                    let d2phi = hess.slice(s![jet_idx, .., a, bb]); // (M)
                    for slot in d2gamma.iter_mut() {
                        *slot = 0.0;
                    }
                    for coeff in 0..m {
                        let w = d2phi[coeff];
                        if w == 0.0 {
                            continue;
                        }
                        let brow = b.row(coeff);
                        for l in 0..p {
                            d2gamma[l] += w * brow[l];
                        }
                    }
                    // proj_e = ⟨∂_eγ, ∂²γ_ab⟩, then Γ^c = Σ_e g⁺_{ce} proj_e.
                    let mut gamma_c = vec![0.0_f64; d];
                    for c in 0..d {
                        let mut acc = 0.0_f64;
                        for e in 0..d {
                            let mut proj_e = 0.0_f64;
                            for l in 0..p {
                                proj_e += dgamma[[l, e]] * d2gamma[l];
                            }
                            acc += ginv[[c, e]] * proj_e;
                        }
                        gamma_c[c] = acc;
                    }
                    let mut psi_ab = psi.slice_mut(s![a, bb, ..]);
                    for coeff in 0..m {
                        psi_ab[coeff] = d2phi[coeff];
                    }
                    for c in 0..d {
                        let gc = gamma_c[c];
                        if gc == 0.0 {
                            continue;
                        }
                        let dphi_c = self.basis_jacobian.slice(s![row, .., c]);
                        for coeff in 0..m {
                            psi_ab[coeff] -= gc * dphi_c[coeff];
                        }
                    }
                }
            }

            // Whitened symmetric-pair design W (M × d²), columns
            // √ω · Ψ̃_{ef} = √ω · Σ_ab R_ae R_bf Ψ_ab; one syrk accumulates
            // gram += W Wᵀ = ω Σ_ef Ψ̃_ef Ψ̃_efᵀ (⪰ 0 by construction).
            let mut wcols = Array2::<f64>::zeros((m, d * d));
            let mut pair = 0usize;
            for e in 0..d {
                for f in 0..d {
                    let mut wcol = wcols.column_mut(pair);
                    for a in 0..d {
                        let rae = rmat[[a, e]];
                        if rae == 0.0 {
                            continue;
                        }
                        for bb in 0..d {
                            let coef = sqrt_omega * rae * rmat[[bb, f]];
                            if coef == 0.0 {
                                continue;
                            }
                            let psi_ab = psi.slice(s![a, bb, ..]);
                            for coeff in 0..m {
                                wcol[coeff] += coef * psi_ab[coeff];
                            }
                        }
                    }
                    pair += 1;
                }
            }
            gram += &wcols.dot(&wcols.t());
        }
        if gram.iter().all(|v| v.is_finite()) {
            Some(gram)
        } else {
            None
        }
    }
}

/// Null-space dimension of the symmetric PSD roughness Gram `S` — the order
/// `r` of the difference / Duchon penalty it encodes (`nullity(S) = r`, since
/// the operator annihilates exactly the degree-`<r` polynomials). Used once at
/// atom construction to fix the arc-length reweighting exponent `β = ½ − r`.
///
/// Numerical null space: eigenvalues at or below `1e-9 · max_eig` (the same
/// conventional relative spectral cutoff [`SaeManifoldTerm::symmetric_rank`]
/// uses for `S`'s rank).
pub(crate) fn smooth_penalty_nullity(s: &Array2<f64>) -> Result<usize, String> {
    let m = s.ncols();
    if m == 0 {
        return Ok(0);
    }
    let mut sym = Array2::<f64>::zeros((m, m));
    for i in 0..m {
        for j in 0..m {
            sym[[i, j]] = 0.5 * (s[[i, j]] + s[[j, i]]);
        }
    }
    let (evals, _evecs) = sym
        .eigh(Side::Lower)
        .map_err(|e| format!("smooth_penalty_nullity: eigh failed: {e}"))?;
    let max_eig = evals.iter().fold(0.0_f64, |acc, &v| acc.max(v));
    if !(max_eig > 0.0) {
        // A zero (or negative-semidefinite) Gram carries no roughness; report a
        // zero operator order so the reweighting is skipped.
        return Ok(0);
    }
    let tol = SAE_MANIFOLD_SPECTRAL_RANK_CUTOFF * max_eig;
    Ok(evals.iter().filter(|&&v| v <= tol).count())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Build an atom over the degree-2 monomial basis `[1, t, t²]` at the given
    // latent coordinates, with decoder `γ(t) = t + t²` (decoded speed `1 + 2t`,
    // which varies across the samples so the arc-length reweighting is
    // non-trivial) and a pure second-derivative penalty `diag(0, 0, 1)` (order
    // `r = 2`, so `β = ½ − r = −3/2 ≠ 0`). The only thing that varies between the
    // Euclidean-patch and Poincaré cases below is `basis_kind`.
    fn monomial_atom(kind: SaeAtomBasisKind, ts: &[f64]) -> SaeManifoldAtom {
        let n = ts.len();
        let mut phi = Vec::with_capacity(n * 3);
        let mut jac = Vec::with_capacity(n * 3);
        for &t in ts {
            phi.extend_from_slice(&[1.0, t, t * t]);
            // ∂[1, t, t²]/∂t = [0, 1, 2t].
            jac.extend_from_slice(&[0.0, 1.0, 2.0 * t]);
        }
        let basis_values = Array2::from_shape_vec((n, 3), phi).unwrap();
        let basis_jacobian = Array3::from_shape_vec((n, 3, 1), jac).unwrap();
        let decoder = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 1.0]).unwrap();
        let mut penalty = Array2::<f64>::zeros((3, 3));
        penalty[[2, 2]] = 1.0;
        SaeManifoldAtom::new(
            "atom",
            kind,
            1,
            basis_values,
            basis_jacobian,
            decoder,
            penalty,
        )
        .unwrap()
    }

    // The Poincaré tangent chart is intrinsically flat for `d = 1`: the latent
    // coordinate runs at a constant multiple of hyperbolic arc length, so the
    // intrinsic arc-length reweighting is a *constant* in `t` and must reproduce
    // the flat Euclidean-patch reweighting exactly — even when the coordinates
    // reach large `|t|`. The previous code divided the squared speed by
    // `λ(p)² = 4cosh⁴(t)`, which at `t = 3` is `≈ cosh⁴(3) ≈ 1.0e4` times off from
    // the correct constant, so it would break this equality by orders of
    // magnitude. Coordinates are deliberately spread to `|t| = 3` to exercise that
    // regime.
    #[test]
    fn poincare_d1_reweighting_is_constant_and_matches_euclidean() {
        let ts = [0.1_f64, 1.5, 3.0];
        let poincare = monomial_atom(SaeAtomBasisKind::Poincare, &ts);

        // The reweighting actually did something: the second-derivative penalty
        // entry moved away from its raw value of 1.0 (so the equality below is
        // not vacuously true because reweighting was a no-op).
        let raw = poincare.smooth_penalty_raw[[2, 2]];
        assert!((raw - 1.0).abs() < 1e-12);
        let reweighted = poincare.smooth_penalty[[2, 2]];
        assert!(
            (reweighted - 1.0).abs() > 0.05,
            "arc-length reweighting should be non-trivial; got {reweighted}"
        );

        // Constant-in-`t` factor ⇒ Poincaré and Euclidean reweightings coincide
        // exactly. The old `1/λ²` divide (~1e4× at t=3) would make these differ.
        assert_poincare_tracks_euclidean(&ts);
    }

    // Assert the `d = 1` Poincaré intrinsic penalty equals its Euclidean-patch
    // twin (same design/decoder, only `basis_kind` differs) elementwise. Because
    // the metric factor is a constant in `t`, this must hold at *any* coordinate
    // concentration; the old per-sample `1/λ²` divide would break it once `|t|`
    // grows (the divide scales as `cosh⁴(t)`).
    fn assert_poincare_tracks_euclidean(ts: &[f64]) {
        let euclidean = monomial_atom(SaeAtomBasisKind::EuclideanPatch, ts);
        let poincare = monomial_atom(SaeAtomBasisKind::Poincare, ts);
        for i in 0..3 {
            for j in 0..3 {
                let e = euclidean.smooth_penalty[[i, j]];
                let p = poincare.smooth_penalty[[i, j]];
                assert!(
                    (e - p).abs() <= 1e-12 + 1e-9 * e.abs(),
                    "Poincaré d=1 penalty[{i},{j}]={p} must equal Euclidean {e}"
                );
            }
        }
    }

    // Small-`|t|` vs large-`|t|` concentration: the Poincaré reweighting tracks
    // the flat one at BOTH (that invariance across concentrations IS the
    // constant-in-`t` property). Under the old `1/λ²`, the large-`|t|` case
    // (`|t| ≈ 3`, `λ² ≈ 4cosh⁴(3) ≈ 1.6e4`) would diverge from its Euclidean twin
    // by orders of magnitude while the small-`|t|` case stayed close — so the
    // reweighting shape would depend on where `t` sits, which this test forbids.
    #[test]
    fn poincare_reweighting_invariant_to_t_concentration() {
        assert_poincare_tracks_euclidean(&[0.02_f64, 0.05, 0.1]);
        assert_poincare_tracks_euclidean(&[2.0_f64, 2.5, 3.0]);
    }

    // ---- d ≥ 2 intrinsic bending Gram (Contribution 1) --------------------
    //
    // These tests exercise `refresh_intrinsic_smooth_penalty` on the `d ≥ 2`
    // branch through the `B = I_p` "identity decoder" harness: an atom with
    // `M = p` and `decoder = I` has `γ(t) = Φ(t)` decoded coordinate-for-
    // coordinate, so the coefficient basis IS the ambient embedding and
    // `tr(Bᵀ S̃ B) = tr(S̃)` (the plain diagonal sum of `smooth_penalty`). The
    // atom then reads its tangent frame `∂γ` from `basis_jacobian` and its
    // ambient Hessian `∂²γ` from the installed second-jet cache — so the harness
    // can feed ANY analytic surface-through-a-chart, with EXACT derivatives, and
    // no basis-approximation error enters the comparison.

    // Build a `B = I_p` bending atom from an explicit tangent frame `∂γ`
    // (`n × p × d`) and ambient Hessian `∂²γ` (`n × p × d × d`).
    fn bending_atom(
        dgamma: Array3<f64>,
        d2gamma: Array4<f64>,
        measure: SaeBendingMeasure,
    ) -> SaeManifoldAtom {
        let (n, p, d) = dgamma.dim();
        let basis_values = Array2::<f64>::zeros((n, p));
        let decoder = Array2::<f64>::eye(p);
        let penalty = Array2::<f64>::zeros((p, p));
        let mut atom = SaeManifoldAtom::new(
            "bend",
            SaeAtomBasisKind::EuclideanPatch,
            d,
            basis_values,
            dgamma,
            decoder,
            penalty,
        )
        .unwrap();
        atom.bending_measure = measure;
        atom.install_bending_second_jet(d2gamma).unwrap();
        atom.refresh_intrinsic_smooth_penalty();
        atom
    }

    // Quadrature subsampling estimates the full-set bending energy: a paraboloid
    // over a 40×40 chart grid, full Gram vs a leverage-stratified Q=400 subsample
    // reweighted by n/Q, agree in total energy to a few percent. This is the
    // scale fix — the full second jet is never materialized at n≫Q.
    #[test]
    fn quadrature_subsample_matches_full_bending_energy() {
        let side = 40usize;
        let n = side * side;
        let p = 3usize;
        let d = 2usize;
        let mut dgamma = Array3::<f64>::zeros((n, p, d));
        let mut d2gamma = Array4::<f64>::zeros((n, p, d, d));
        for i in 0..side {
            for j in 0..side {
                let row = i * side + j;
                let t0 = -1.0 + 2.0 * (i as f64) / (side as f64 - 1.0);
                let t1 = -1.0 + 2.0 * (j as f64) / (side as f64 - 1.0);
                // γ = (t0, t1, ½(t0²+t1²)): ∂γ = [[1,0],[0,1],[t0,t1]], ∂²γ_z = I.
                dgamma[[row, 0, 0]] = 1.0;
                dgamma[[row, 1, 1]] = 1.0;
                dgamma[[row, 2, 0]] = t0;
                dgamma[[row, 2, 1]] = t1;
                d2gamma[[row, 2, 0, 0]] = 1.0;
                d2gamma[[row, 2, 1, 1]] = 1.0;
            }
        }
        // Full-set reference (Data measure ⇒ ω = 1, so energy = Σ_n ‖II‖²).
        let full = bending_atom(dgamma.clone(), d2gamma.clone(), SaeBendingMeasure::Data);
        let e_full = bending_energy_of(&full);
        assert!(e_full > 0.0 && e_full.is_finite());
        assert!(full.bending_quadrature_rows.is_none(), "small-n harness stays full");

        // Manually drive the subsample path: pick Q rows, keep only their jet
        // slices, tag the quadrature. try_intrinsic then reweights by n/Q.
        let q = 400usize;
        let rows = full.select_bending_quadrature_rows(q);
        assert!(rows.len() <= q && rows.len() >= q / 2, "got {} rows", rows.len());
        let mut compact = Array4::<f64>::zeros((rows.len(), p, d, d));
        for (jet_idx, &r) in rows.iter().enumerate() {
            compact
                .slice_mut(s![jet_idx, .., .., ..])
                .assign(&d2gamma.slice(s![r, .., .., ..]));
        }
        let mut sub = bending_atom(dgamma, d2gamma, SaeBendingMeasure::Data);
        sub.basis_second_jet_values = Some(compact);
        sub.bending_quadrature_rows = Some(rows.clone());
        // Memory footprint is O(Q·M·d²), NOT O(n·M·d²): the retained jet holds
        // only the subsampled rows (Q ≪ n), the OOM-avoiding invariant at scale.
        let jet_len = sub.basis_second_jet_values.as_ref().unwrap().len();
        assert_eq!(jet_len, rows.len() * p * d * d, "compact jet must be Q·p·d²");
        assert!(rows.len() * 3 < n, "subsample Q={} must be ≪ n={n}", rows.len());
        sub.refresh_intrinsic_smooth_penalty();
        let e_sub = bending_energy_of(&sub);
        assert!(e_sub > 0.0 && e_sub.is_finite());
        let rel = (e_sub - e_full).abs() / e_full;
        assert!(
            rel < 0.08,
            "subsampled bending energy {e_sub} vs full {e_full} (rel {rel:.3}) — reweight off"
        );
    }

    // `tr(Bᵀ S̃ B) = tr(S̃)` for the identity decoder — the total bending energy.
    fn bending_energy_of(atom: &SaeManifoldAtom) -> f64 {
        (0..atom.basis_size())
            .map(|i| atom.smooth_penalty[[i, i]])
            .sum()
    }

    // Independent ambient-space reference for `Σ_i ω_i ‖II(t_i)‖²_g`: at each
    // row form `g = ∂γᵀ∂γ`, project the ambient Hessian onto the NORMAL bundle
    // by explicit subtraction of its tangential part
    // `II_ab = ∂²γ_ab − ∂γ · g⁺ (∂γᵀ ∂²γ_ab)`, then contract
    // `Σ_abcd g^{ac} g^{bd} ⟨II_ab, II_cd⟩`. This shares NO code with the
    // coefficient-space whitening in `try_intrinsic_bending_gram` (which never
    // forms an ambient `II` vector), so agreement is a genuine cross-check.
    fn ambient_bending_energy(
        dgamma: &Array3<f64>,
        d2gamma: &Array4<f64>,
        volume: bool,
    ) -> f64 {
        let (n, p, d) = dgamma.dim();
        let mut total = 0.0_f64;
        for row in 0..n {
            let dgm = dgamma.slice(s![row, .., ..]).to_owned(); // (p, d)
            let mut g = Array2::<f64>::zeros((d, d));
            for a in 0..d {
                for b in 0..d {
                    let mut acc = 0.0;
                    for m in 0..p {
                        acc += dgm[[m, a]] * dgm[[m, b]];
                    }
                    g[[a, b]] = acc;
                }
            }
            let (ev, evec) = g.eigh(Side::Lower).unwrap();
            let maxe = ev.iter().cloned().fold(0.0_f64, f64::max);
            if !(maxe > 0.0) {
                continue;
            }
            let tol = 1.0e-12 * maxe;
            let mut ginv = Array2::<f64>::zeros((d, d));
            let mut logdet = 0.0_f64;
            let mut full = true;
            for k in 0..d {
                let s = ev[k];
                if s > tol {
                    let inv = 1.0 / s;
                    for a in 0..d {
                        for c in 0..d {
                            ginv[[a, c]] += evec[[a, k]] * evec[[c, k]] * inv;
                        }
                    }
                    logdet += s.ln();
                } else {
                    full = false;
                }
            }
            let omega = if volume {
                if full {
                    (0.5 * logdet).exp()
                } else {
                    0.0
                }
            } else {
                1.0
            };
            if omega == 0.0 {
                continue;
            }
            // II_ab (p) per ordered pair.
            let mut ii = vec![vec![vec![0.0_f64; p]; d]; d];
            for a in 0..d {
                for b in 0..d {
                    let mut v = vec![0.0_f64; p];
                    for m in 0..p {
                        v[m] = d2gamma[[row, m, a, b]];
                    }
                    let mut dtv = vec![0.0_f64; d];
                    for e in 0..d {
                        let mut acc = 0.0;
                        for m in 0..p {
                            acc += dgm[[m, e]] * v[m];
                        }
                        dtv[e] = acc;
                    }
                    let mut tc = vec![0.0_f64; d];
                    for c in 0..d {
                        let mut acc = 0.0;
                        for e in 0..d {
                            acc += ginv[[c, e]] * dtv[e];
                        }
                        tc[c] = acc;
                    }
                    for m in 0..p {
                        let mut proj = 0.0;
                        for c in 0..d {
                            proj += dgm[[m, c]] * tc[c];
                        }
                        ii[a][b][m] = v[m] - proj;
                    }
                }
            }
            let mut e = 0.0_f64;
            for a in 0..d {
                for b in 0..d {
                    for c in 0..d {
                        for dd in 0..d {
                            let mut dot = 0.0;
                            for m in 0..p {
                                dot += ii[a][b][m] * ii[c][dd][m];
                            }
                            e += ginv[[a, c]] * ginv[[b, dd]] * dot;
                        }
                    }
                }
            }
            total += omega * e;
        }
        total
    }

    // The raw (gauge-LEAKING) coefficient-`t` second-derivative Gram energy that
    // the bending penalty supersedes: `Σ_i Σ_ab ‖∂²γ_ab‖²` with neither a
    // tangential projection nor a metric contraction. It re-charges roughness on
    // a flat atom seen through a curved chart, so it is NOT chart-invariant.
    fn raw_hessian_energy(d2gamma: &Array4<f64>) -> f64 {
        d2gamma.iter().map(|v| v * v).sum()
    }

    // Paraboloid graph γ(x) = (x₁, x₂, ½(κ₁x₁² + κ₂x₂²)) in ℝ³ (p = 3, d = 2)
    // sampled at the base points `pts`, returning its exact (∂γ, ∂²γ).
    fn paraboloid_arrays(pts: &[(f64, f64)], k1: f64, k2: f64) -> (Array3<f64>, Array4<f64>) {
        let n = pts.len();
        let mut dg = Array3::<f64>::zeros((n, 3, 2));
        let mut d2 = Array4::<f64>::zeros((n, 3, 2, 2));
        for (i, &(x1, x2)) in pts.iter().enumerate() {
            dg[[i, 0, 0]] = 1.0;
            dg[[i, 2, 0]] = k1 * x1;
            dg[[i, 1, 1]] = 1.0;
            dg[[i, 2, 1]] = k2 * x2;
            d2[[i, 2, 0, 0]] = k1;
            d2[[i, 2, 1, 1]] = k2;
        }
        (dg, d2)
    }

    // (a) Paraboloid: the coefficient-space whitened Gram trace matches the
    //     independent ambient normal-projection reference to machine precision,
    //     under BOTH the volume and the data measure; and the single-point,
    //     unit-metric case reproduces the hand-checkable `κ₁² + κ₂²`.
    #[test]
    fn bending_paraboloid_matches_ambient_reference() {
        let (k1, k2) = (2.0_f64, 3.0_f64);

        // Origin, unit metric (g = I): ‖II‖² = κ₁² + κ₂² = 4 + 9 = 13 exactly.
        let (dg0, d20) = paraboloid_arrays(&[(0.0, 0.0)], k1, k2);
        let atom0 = bending_atom(dg0, d20, SaeBendingMeasure::Data);
        assert!(
            (bending_energy_of(&atom0) - 13.0).abs() < 1e-12,
            "origin bending energy {} != 13",
            bending_energy_of(&atom0)
        );

        // A genuine 2-D grid where the metric is non-trivial (∇f ≠ 0) so the
        // tangential projection and the g⁻¹⊗g⁻¹ contraction both bite.
        let pts: Vec<(f64, f64)> = [-0.7, -0.2, 0.3, 0.9]
            .iter()
            .flat_map(|&a| [-0.5, 0.1, 0.6, 1.1].iter().map(move |&b| (a, b)))
            .collect();
        let (dg, d2) = paraboloid_arrays(&pts, k1, k2);

        for &volume in &[false, true] {
            let measure = if volume {
                SaeBendingMeasure::Volume
            } else {
                SaeBendingMeasure::Data
            };
            let atom = bending_atom(dg.clone(), d2.clone(), measure);
            let got = bending_energy_of(&atom);
            let want = ambient_bending_energy(&dg, &d2, volume);
            assert!(want > 1.0, "reference energy should be sizeable: {want}");
            assert!(
                (got - want).abs() <= 1e-9 * want.max(1.0),
                "volume={volume}: coefficient-space bending {got} != ambient reference {want}"
            );
        }
    }

    // Warped paraboloid chart: parameter `u` with base point `x = φ(u)`,
    // `φ = (u₁ + a₁u₁², u₂ + a₂u₂²)` (a nonlinear, axis-decoupled diffeo).
    // Returns the chart-`u` (∂γ, ∂²γ) via the exact chain rule, plus the matched
    // base points `x` so the identity chart can hit the SAME physical points.
    fn paraboloid_warped_arrays(
        us: &[(f64, f64)],
        k1: f64,
        k2: f64,
        a1: f64,
        a2: f64,
    ) -> (Array3<f64>, Array4<f64>, Vec<(f64, f64)>) {
        let n = us.len();
        let mut dg = Array3::<f64>::zeros((n, 3, 2));
        let mut d2 = Array4::<f64>::zeros((n, 3, 2, 2));
        let mut xs = Vec::with_capacity(n);
        for (i, &(u1, u2)) in us.iter().enumerate() {
            let x1 = u1 + a1 * u1 * u1;
            let x2 = u2 + a2 * u2 * u2;
            xs.push((x1, x2));
            let s = [1.0 + 2.0 * a1 * u1, 1.0 + 2.0 * a2 * u2];
            let c = [2.0 * a1, 2.0 * a2];
            // Base-surface derivatives at x.
            let dpx = [[1.0, 0.0, k1 * x1], [0.0, 1.0, k2 * x2]]; // ∂_{x_a}P
            let d2px = [k1, k2]; // ∂²_{x_a x_a}P z-component; off-diagonal 0.
            for a in 0..2 {
                for m in 0..3 {
                    dg[[i, m, a]] = dpx[a][m] * s[a];
                }
            }
            for a in 0..2 {
                for b in 0..2 {
                    for m in 0..3 {
                        let mut val = 0.0;
                        if a == b && m == 2 {
                            val += d2px[a] * s[a] * s[b];
                        }
                        if a == b {
                            val += dpx[a][m] * c[a];
                        }
                        d2[[i, m, a, b]] = val;
                    }
                }
            }
        }
        (dg, d2, xs)
    }

    // (b) Gauge invariance: the SAME paraboloid through the identity chart and
    //     through a nonlinear-warped chart, evaluated at MATCHED material
    //     points, yields the SAME bending energy under the (pointwise-invariant)
    //     data measure — whereas the raw coefficient-`t` Hessian energy differs
    //     by an order of magnitude. Curvature is chart-honest; raw roughness is
    //     not.
    #[test]
    fn bending_gauge_invariant_under_nonlinear_chart() {
        let (k1, k2) = (1.3_f64, 0.8_f64);
        let us: Vec<(f64, f64)> = [-0.8, -0.3, 0.4, 1.0]
            .iter()
            .flat_map(|&a| [-0.6, 0.2, 0.7, 1.2].iter().map(move |&b| (a, b)))
            .collect();
        let (dg_u, d2_u, xs) = paraboloid_warped_arrays(&us, k1, k2, 0.35, 0.5);
        let (dg_x, d2_x) = paraboloid_arrays(&xs, k1, k2);

        let warped = bending_atom(dg_u.clone(), d2_u.clone(), SaeBendingMeasure::Data);
        let identity = bending_atom(dg_x.clone(), d2_x.clone(), SaeBendingMeasure::Data);

        let e_warped = bending_energy_of(&warped);
        let e_identity = bending_energy_of(&identity);
        assert!(e_identity > 1.0, "identity-chart energy sanity: {e_identity}");
        assert!(
            (e_warped - e_identity).abs() <= 1e-9 * e_identity,
            "bending energy must be chart-invariant: warped {e_warped} vs identity {e_identity}"
        );

        // The raw Gram is NOT invariant: the curved chart re-charges roughness.
        let raw_warped = raw_hessian_energy(&d2_u);
        let raw_identity = raw_hessian_energy(&d2_x);
        let ratio = raw_warped / raw_identity;
        assert!(
            ratio > 3.0,
            "raw (gauge-leaking) Hessian energy should diverge across charts: \
             warped {raw_warped} vs identity {raw_identity} (ratio {ratio})"
        );
    }

    // (c) Flat plane through a curved chart: γ(u) = A·φ(u) with A a 3×2
    //     orthonormal plane frame and φ the nonlinear warp — the IMAGE is a flat
    //     2-plane, so II ≡ 0 and the bending energy must vanish, even though the
    //     raw Hessian energy is sizeable (the exact gauge leak the penalty
    //     closes).
    #[test]
    fn bending_flat_plane_through_curved_chart_is_zero() {
        // Orthonormal plane frame in ℝ³ (a tilted, non-axis-aligned plane).
        let inv_sqrt2 = 1.0 / 2.0_f64.sqrt();
        let acol = [[inv_sqrt2, 0.0], [0.0, 1.0], [inv_sqrt2, 0.0]]; // A[m][axis]
        let (a1, a2) = (0.3_f64, 0.3_f64);
        let us: Vec<(f64, f64)> = [-0.9, -0.2, 0.5, 1.1]
            .iter()
            .flat_map(|&a| [-0.7, 0.1, 0.8].iter().map(move |&b| (a, b)))
            .collect();
        let n = us.len();
        let mut dg = Array3::<f64>::zeros((n, 3, 2));
        let mut d2 = Array4::<f64>::zeros((n, 3, 2, 2));
        for (i, &(u1, u2)) in us.iter().enumerate() {
            let s = [1.0 + 2.0 * a1 * u1, 1.0 + 2.0 * a2 * u2];
            let c = [2.0 * a1, 2.0 * a2];
            // γ = A φ(u): Dγ[:,axis] = A[:,axis]·s_axis; D²γ_ab = A[:,a]·c_a·[a==b]
            // (φ axis-decoupled: only i = a = b survives in ΣA[:,i]D²φ_i,ab).
            for m in 0..3 {
                for axis in 0..2 {
                    dg[[i, m, axis]] = acol[m][axis] * s[axis];
                }
                d2[[i, m, 0, 0]] = acol[m][0] * c[0];
                d2[[i, m, 1, 1]] = acol[m][1] * c[1];
            }
        }
        for &volume in &[false, true] {
            let measure = if volume {
                SaeBendingMeasure::Volume
            } else {
                SaeBendingMeasure::Data
            };
            let atom = bending_atom(dg.clone(), d2.clone(), measure);
            let energy = bending_energy_of(&atom);
            assert!(
                energy.abs() < 1e-16,
                "flat plane through a curved chart must have zero bending energy; \
                 got {energy} (volume={volume})"
            );
        }
        // The gauge leak was real: the raw Hessian energy is far from zero.
        let raw = raw_hessian_energy(&d2);
        assert!(
            raw > 0.1,
            "raw Hessian energy should be sizeable on the curved chart: {raw}"
        );
    }

    // The invariant null space (the affine, zero-II maps the penalty must NOT
    // charge) has dimension `d + 1` in coefficient space but is spanned by the
    // functions with vanishing COVARIANT Hessian — which are the geometry-affine
    // maps, not the ambient coordinate slots (a straight ambient axis restricted
    // to a curved surface has nonzero covariant Hessian). Constructing those
    // null vectors by hand needs the frozen connection, so the null space is
    // exercised structurally instead: `try_intrinsic_bending_gram` builds S̃ as
    // `Σ ω Ψ̃Ψ̃ᵀ`, a manifest sum of outer products, so it is PSD with the affine
    // kernel by construction; the exact-zero FLAT-plane test above is the
    // load-bearing check that the kernel is hit (a flat atom seen through a
    // curved chart is charged nothing).
    #[test]
    fn bending_gram_is_symmetric_psd() {
        let pts: Vec<(f64, f64)> = [-0.5, 0.4]
            .iter()
            .flat_map(|&a| [-0.3, 0.6].iter().map(move |&b| (a, b)))
            .collect();
        let (dg, d2) = paraboloid_arrays(&pts, 1.7, 2.2);
        let atom = bending_atom(dg, d2, SaeBendingMeasure::Volume);
        let s = &atom.smooth_penalty;
        let m = atom.basis_size();
        for i in 0..m {
            for j in 0..m {
                assert!(
                    (s[[i, j]] - s[[j, i]]).abs() < 1e-12,
                    "S̃ must be symmetric at [{i},{j}]"
                );
            }
        }
        let (ev, _evec) = s.eigh(Side::Lower).unwrap();
        let min_eig = ev.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(min_eig > -1e-10, "S̃ must be PSD; min eigenvalue {min_eig}");
        let max_eig = ev.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_eig > 1.0, "curved atom must carry real bending: {max_eig}");
    }
}
