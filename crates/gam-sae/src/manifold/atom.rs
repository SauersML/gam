use super::*;

/// Basis/topology tag for one SAE manifold atom.
///
/// The evaluated basis and input-location jet live on [`SaeManifoldAtom`].
/// This enum records the user-facing topology choice so downstream diagnostics
/// and Python wrappers can round-trip whether the atom was a Duchon patch,
/// periodic curve, sphere, or a caller-supplied precomputed basis.
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
    /// polynomial-in-`t` expansion — but its smoothness penalty is the
    /// conformal-reweighted Dirichlet energy of the Poincaré metric
    /// (`refresh_intrinsic_smooth_penalty` measures wiggle in *hyperbolic*
    /// arc length via the `λ(p)` conformal factor). This makes an atom whose
    /// feature density grows toward the ball boundary (exponential-volume /
    /// tree-leaf hierarchy) the regime where it differs from the flat patch.
    Poincare,
    Precomputed(String),
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
            // `Sphere` is parameterised via a (lat, lon) intrinsic chart; the
            // chart evaluator already enforces sphere geometry through its
            // cos/sin terms (in radians, multiplying lat/lon directly into
            // `sin`/`cos`), so the latent optimiser sees a 2-D product
            // manifold: lat is a bounded interval `[-π/2, π/2]` (enforced here
            // by the `Interval` retraction — its clamp + active-bound tangent
            // projection — NOT by truncating the chart jet) and lon is an `S^1`
            // angle wrapped modulo `2π`.
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
            Self::Linear
            | Self::Duchon
            | Self::EuclideanPatch
            | Self::Poincare
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
            Self::Linear
            | Self::Duchon
            | Self::EuclideanPatch
            | Self::Poincare
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
    /// [`SaeBasisEvaluator::phi_eta_split`]) by `η`, leaving the *linear*
    /// columns untouched, so `η = 0` is the Eckart-Young linear relaxation (a
    /// convex decoder problem whose global optimum [`linear_span_anchor`]
    /// certifies) and `η = 1` is the full curved basis. The certified tracker
    /// walks `η` from `0 → 1`; every other caller sees the default `1.0`, which
    /// makes [`Self::refresh_basis`] bit-for-bit identical to the un-dialed
    /// `evaluate` path (`evaluate_phi_eta` at `η = 1` returns the unscaled
    /// basis). Caller-managed atoms (no installed evaluator) ignore the dial —
    /// there is no curved/linear split without an evaluator to provide it.
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
}

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
            decoder_frame: None,
            homotopy_eta: 1.0,
            chart_canonicalized: false,
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

        self.basis_values = phi_red;
        self.basis_jacobian = jac_red;
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
        // tracker scales the curved columns toward the linear relaxation; the
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
        Ok(())
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
            for out_col in 0..p {
                out[out_col] += phi * self.decoder_coefficients[[basis_col, out_col]];
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
            for out_col in 0..p {
                out[out_col] += dphi * self.decoder_coefficients[[basis_col, out_col]];
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
    /// branch instead of riding the linear shadow. `curved_cols` are the atom's
    /// `phi_eta_split` curved column indices; a linear-only atom writes zeros.
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
            for out_col in 0..p {
                out[out_col] += dphi * self.decoder_coefficients[[basis_col, out_col]];
            }
        }
    }

    /// Recompute the intrinsic (arc-length) roughness Gram
    /// [`Self::smooth_penalty`] from [`Self::smooth_penalty_raw`], the current
    /// basis Jacobian, and the current decoder coefficients (issue #673).
    ///
    /// The raw penalty `0.5·λ·tr(BᵀS B)` measures roughness per unit of the raw
    /// latent coordinate `t`, so it is *not* invariant under reparameterizing
    /// `t` — and the model evidence that ranks an atom's topology (circle vs
    /// line) inherits that gauge dependence. The decoder curve is
    /// `g(t) = Φ(t) B` and its pulled-back metric is the scalar squared speed
    /// `m(t) = ‖g'(t)‖² = ‖J(t)‖²` with `J(t) = Φ'(t) B` (the decoder
    /// Jacobian, [`Self::fill_decoded_derivative_row`]). The arc-length
    /// roughness of an order-`r` operator reweights the raw-`t` derivative
    /// energy density by `m^{½−r}` (`= m^{−3/2}` for the standard
    /// second-derivative penalty), which removes the gauge dependence.
    ///
    /// Realised as a per-coefficient symmetric congruence
    /// `S̃ = W^{½} S W^{½}`, `W = diag(w_μ)`, `w_μ = m̄_μ^{β}`, `β = ½ − r`,
    /// where `m̄_μ` is the basis-activation-weighted average squared speed
    /// localised to coefficient `μ`,
    /// `m̄_μ = (Σ_n Φ_μ(t_n)² m_n) / Σ_n Φ_μ(t_n)²`, `m_n = ‖J(t_n)‖²`. The
    /// congruence keeps `S̃` symmetric PSD with the same rank as `S` (Sylvester
    /// inertia), so the Kronecker Hessian `S̃ ⊗ I_p` and the REML
    /// `rank(S)`-Occam term are structurally unchanged; only the metric-aware
    /// log-det / quadratic value move, which is exactly the gauge correction.
    ///
    /// The metric weight is frozen at the current `B` (lagged-diffusivity /
    /// IRLS surrogate): within one inner solve the penalty stays a quadratic
    /// Gram form, and refreshing `W` between assemblies makes the *converged*
    /// penalty the true arc-length roughness. The per-coefficient weight is
    /// centered (its geometric mean is 1), so constant-speed atoms (the
    /// periodic sin/cos basis, `m̄_μ ≡ c`) get `w_μ ≡ 1` and hence `S̃ = S`
    /// exactly — periodic atoms are unaffected and no overall magnitude (which
    /// `λ` already owns) leaks into the penalty.
    ///
    /// Conservative scope: the scalar-speed reweighting is the genuine
    /// arc-length normalisation only for a 1-D latent (the circle-vs-line case
    /// the issue is about). For `latent_dim != 1`, or a degenerate (empty/zero)
    /// raw Gram, `S̃ = S` is left untouched.
    pub fn refresh_intrinsic_smooth_penalty(&mut self) {
        let m = self.basis_size();
        // No reweighting when there is no penalty operator order to invert into
        // arc length, or for higher-dim latents where the metric is a matrix
        // (det(g) volume reweighting is deferred — see scope note above).
        if m == 0 || self.smooth_penalty_order == 0 || self.latent_dim != 1 {
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
        // Poincaré tangent patch: measure the decoded speed per unit of
        // *hyperbolic* latent length rather than flat tangent length. A unit
        // step in the tangent coordinate `t` covers hyperbolic distance
        // `λ(p(t))` (the conformal factor at the ball point `p = exp₀(t)`), so
        // the arc-length speed is `‖J‖ / λ` and the squared speed picks up a
        // `1/λ²` factor. For the monomial patch (`d = 1`) the tangent coordinate
        // is the linear monomial column (`Φ = [1, t, …]`, so column 1 is `t`).
        let hyperbolic = matches!(self.basis_kind, SaeAtomBasisKind::Poincare);
        let linear_col = if hyperbolic && m >= 2 {
            Some(1usize)
        } else {
            None
        };
        for row in 0..n {
            self.fill_decoded_derivative_row(row, 0, &mut deriv);
            let mut speed_sq = 0.0_f64;
            for &d in deriv.iter() {
                speed_sq += d * d;
            }
            if let Some(col) = linear_col {
                let t = self.basis_values[[row, col]];
                // p = exp₀(t) at unit curvature c = −1: ‖p‖ = tanh(|t|), and
                // λ(p) = 2 / (1 − ‖p‖²) = 2 / (1 − tanh²|t|) = 2·cosh²(t).
                // speed_sq ← speed_sq / λ².  (cosh is even, so the sign of t
                // does not matter.)
                let lambda = 2.0 * t.cosh() * t.cosh();
                if lambda.is_finite() && lambda > 0.0 {
                    speed_sq /= lambda * lambda;
                }
            }
            for col in 0..m {
                let phi = self.basis_values[[row, col]];
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
