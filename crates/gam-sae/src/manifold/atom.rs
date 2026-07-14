use super::*;

/// Declared function-space seminorm used by one atom's smoothing prior.
///
/// The declaration is consumed once, at atom construction or an explicit
/// structural reparameterization boundary, and materialized as a fixed
/// coefficient Gram `S_ref`. It is never inferred from the decoder and is never
/// refreshed during an inner or outer optimization step.
#[derive(Debug, Clone, PartialEq)]
pub enum SaeReferenceRoughness {
    /// A caller/topology supplied basis representation of a declared
    /// final-function seminorm:
    ///
    /// `S_ref[i,j] = <L phi_i, L phi_j>_(nu_ref)`.
    ///
    /// The matrix is validated as finite, symmetric, and positive
    /// semidefinite before it is installed. This is not a coefficient-size
    /// fallback: the caller is explicitly declaring the reference measure
    /// `nu_ref` and differential operator `L` represented by the matrix.
    ProvidedFunctionGram(Array2<f64>),
    /// Hyperbolic Dirichlet seminorm evaluated at a fixed set of Poincare
    /// tangent-chart reference coordinates. The coordinates are part of the
    /// declaration, not the fitted latent state. Construction fails if their
    /// shape or values are invalid or if the analytic geometry builder fails.
    PoincareConformalDirichlet { reference_coords: Array2<f64> },
}

/// Provenance of the frozen reference-function Gram retained by an atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SaeReferenceRoughnessKind {
    ProvidedFunctionGram,
    PoincareConformalDirichlet,
}

const POINCARE_REFERENCE_CURVATURE: f64 = -1.0;

/// Basis/topology tag for one SAE manifold atom.
///
/// The evaluated basis and input-location jet live on [`SaeManifoldAtom`].
/// This enum records the user-facing topology choice so downstream diagnostics
/// and Python wrappers can round-trip whether the atom was a Duchon patch,
/// periodic curve, sphere, or a caller-supplied precomputed basis.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SaeAtomBasisKind {
    Duchon,
    Periodic,
    Sphere,
    Torus,
    /// Real projective plane `RP² = S²/{u ~ -u}` on the round unit cover.
    /// The analytic basis is the even-degree spherical-harmonic restriction
    /// owned by [`crate::basis::QuotientSpectralEvaluator`]. Coordinates stay
    /// on the existing `(latitude, longitude)` sphere cover; antipodal rows are
    /// an exact discrete gauge because every emitted basis column is even.
    ProjectivePlane,
    /// Flat Klein bottle `T²/{(theta, phi) ~ (theta + 1/2, -phi)}`.
    /// The analytic basis is the diagonal-character restriction of the real
    /// tensor Fourier cover. Coordinates remain two unit-period circle phases;
    /// the deck twin is an exact discrete gauge of every decoder function.
    KleinBottle,
    /// Cylinder `S¹ × ℝ` (`d = 2`): a periodic circle axis tensored with a flat
    /// (Duchon-polynomial) line axis, via [`CylinderHarmonicEvaluator`]. Axis 0
    /// is the circle (fraction-of-period convention, wrapped modulo `1.0`),
    /// axis 1 is the unbounded line (`Euclidean`). Completes the `d = 2`
    /// topology race (torus vs sphere vs euclidean-patch vs cylinder) so a
    /// periodic-times-linear feature is adjudicable on its true manifold instead
    /// of being forced into a torus or flat-patch stand-in.
    Cylinder,
    /// Möbius band (`d = 2`, #2240): the double-cover chart
    /// `Circle{period 2} × Interval[-1, 1]` with the deck-invariant harmonic
    /// basis of [`crate::basis::MobiusHarmonicEvaluator`] (`trig(πks)·wᵐ`,
    /// `k + m` even). The half-twist lives in the parity culling of the
    /// basis, not in the retraction manifold, so the optimizer walks an
    /// ordinary smooth cylinder while the represented surface is genuinely
    /// non-orientable — the topology a torus wraps spuriously and a flat
    /// patch loses. Round-trips under the name `"mobius"`.
    Mobius,
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
    /// ([`SaeReferenceRoughness::PoincareConformalDirichlet`]). For the `d = 1`
    /// tangent chart the
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
            Self::Sphere | Self::ProjectivePlane => LatentManifold::Product(vec![
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
            Self::Torus | Self::KleinBottle => {
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
            // The Möbius basis lives on its smooth double cover. The deck
            // identification `(s, w) ~ (s + 1, -w)` is enforced by the basis
            // parity, while optimization retracts on the ordinary cylinder
            // `S¹(period 2) × [-1, 1]`.
            Self::Mobius => LatentManifold::Product(vec![
                LatentManifold::Circle { period: 2.0 },
                LatentManifold::Interval { lo: -1.0, hi: 1.0 },
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
                let phase = kappa * t;
                let (sin, cos) = phase.sin_cos();
                // `1 - cos(phase)` rounds to zero for nonzero
                // |phase| < sqrt(EPSILON). The half-angle identity preserves
                // the quadratic energy all the way to the subnormal range.
                let sin_half = (0.5 * phase).sin();
                let one_minus_cos = 2.0 * sin_half * sin_half;
                Self {
                    value: (alpha / (kappa * kappa)) * one_minus_cos,
                    grad: (alpha / kappa) * sin,
                    hess: alpha * cos,
                    sq_equiv: (2.0 / (kappa * kappa)) * one_minus_cos,
                }
            }
        }
    }

    /// Stable signed energy change `V(to) - V(from)`.
    ///
    /// Line searches need the change itself, which can be many orders of
    /// magnitude smaller than either endpoint energy near a stationary point.
    /// Subtracting two calls to [`Self::eval`] would erase that signal. These
    /// difference identities evaluate the increment directly for both supported
    /// geometries and remain valid across a periodic branch cut.
    pub(crate) fn value_delta(alpha: f64, from: f64, to: f64, period: Option<f64>) -> f64 {
        let delta = to - from;
        match period {
            None => alpha * delta * (from + 0.5 * delta),
            Some(p) => {
                let kappa = std::f64::consts::TAU / p;
                let midpoint = from + 0.5 * delta;
                let midpoint_phase = kappa * midpoint;
                let half_delta_phase = 0.5 * kappa * delta;
                (2.0 * alpha / (kappa * kappa))
                    * midpoint_phase.sin()
                    * half_delta_phase.sin()
            }
        }
    }

    /// Positive-semidefinite curvature used by the Newton/Schur majorizer.
    ///
    /// This is deliberately not the exact prior Hessian on a periodic axis:
    /// the exact `alpha*cos(kappa*t)` remains signed.  The factorized inner
    /// solver and its Laplace log determinant declare this positive-part
    /// operator, so every derivative of that operator must call this seam.
    #[inline]
    pub(crate) fn psd_majorizer_hess(self) -> f64 {
        self.hess.max(0.0)
    }

    /// Signed correction that turns the PSD majorizer back into the exact
    /// prior Hessian: `hess = psd_majorizer_hess + negative_hessian_remainder`.
    #[inline]
    pub(crate) fn negative_hessian_remainder(self) -> f64 {
        self.hess.min(0.0)
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
    basis_kind: SaeAtomBasisKind,
    latent_dim: usize,
    pub basis_values: Array2<f64>,
    pub basis_jacobian: Array3<f64>,
    pub decoder_coefficients: Array2<f64>,
    /// Frozen reference-function Gram `S_ref` read by every smoothing consumer
    /// (value, gradient, Kronecker Hessian, rank, and log determinant).
    ///
    /// If `f_B(t) = B^T phi(t)` and the declared seminorm has
    /// `S_ref[i,j] = <L phi_i,L phi_j>_(nu_ref)`, bilinearity gives
    ///
    /// `||L f_B||^2_(nu_ref) = sum_c b_c^T S_ref b_c = tr(B^T S_ref B)`.
    ///
    /// Therefore the implemented `0.5 * lambda * tr(B^T S_ref B)` is exactly
    /// the declared final-function seminorm, with exact gradient
    /// `lambda * S_ref * B` and Hessian `lambda * (S_ref tensor I)`. It is not
    /// the moving intrinsic bending energy of the current decoder.
    smooth_penalty: Array2<f64>,
    /// Which explicit declaration produced [`Self::smooth_penalty`].
    reference_roughness_kind: SaeReferenceRoughnessKind,
    /// Persisted analytic geometry authority for atoms built by the native
    /// lifecycle. Hand-assembled/precomputed atoms may omit it, but an atom
    /// without this plan cannot be serialized for analytic rebuild or OOS.
    geometry_plan: Option<SaeAtomGeometryPlan>,
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
    /// `r·(p − r)` Grassmann dimensions to the quasi-Laplace score normalizer.
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
    /// Orthonormal column map `Q` (`M × r`) frozen by the #1117 rank-revealing
    /// reduction [`Self::reduce_basis_to_subspace`]. `Some` iff this atom's
    /// fixed-width inner basis was reparametrized onto its data-supported
    /// subspace: [`Self::decoder_coefficients`] is then the REDUCED
    /// `B̃ = Qᵀ B` (`r × p`), [`Self::basis_values`] the reduced `Φ̃ = Φ Q`
    /// (`n × r`), and the live evaluator a [`SubspaceReducedEvaluator`] emitting
    /// `Φ̃` on every refresh — so the reduced design is full-rank BY
    /// CONSTRUCTION for the whole fit.
    ///
    /// The reduction is a purely INTERNAL fit-conditioning device and must not
    /// escape to a consumer that rebuilds the standard fixed-width inner basis
    /// from emitted metadata (out-of-sample predict / steer / reconstruction,
    /// issue #2135): such a consumer re-emits the full `M`-column inner design
    /// (`[1, sin, cos, …]`), against which the correct decoder is the full-width
    /// pre-image `B = Q B̃` (`M × p`), NOT the reduced `B̃`. Decoding the full
    /// design by the reduced block mismatches widths (`M` vs `r`) and is the
    /// #2135 "decoder_blocks[k] has M=2 but rebuilt basis has M=3" defect.
    /// [`Self::full_width_decoder`] / [`Self::full_basis_size`] re-expand the
    /// reduced frame at the emission boundary so it never leaks. `None` ⇒ the
    /// atom was never reduced and the stored decoder is already full-width.
    pub reduced_column_map: Option<Array2<f64>>,
    /// #F3 — the fitted per-axis ARD precision `α_a = exp(log_ard[k][a])` of the
    /// latent coordinate prior, length `latent_dim`. Stamped from the TERMINAL
    /// `rho.log_ard[k]` when the term finalizes each atom (alongside the other
    /// terminal-rho state), so the certified encode can add the SAME ARD /
    /// von-Mises coordinate prior the fit optimized `t` against
    /// ([`crate::encode::EncodeObjective`] `prior_alpha`) rather than certifying a
    /// prior-free objective. `None` ⇒ no coordinate prior was fitted for this atom
    /// (`rho.log_ard[k]` empty), in which case the encode objective is prior-free
    /// exactly as before.
    pub ard_precisions: Option<Array1<f64>>,
}

impl SaeManifoldAtom {
    pub fn basis_kind(&self) -> &SaeAtomBasisKind {
        &self.basis_kind
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    pub fn smooth_penalty(&self) -> &Array2<f64> {
        &self.smooth_penalty
    }

    pub fn reference_roughness_kind(&self) -> SaeReferenceRoughnessKind {
        self.reference_roughness_kind
    }

    pub fn geometry_plan(&self) -> Option<&SaeAtomGeometryPlan> {
        self.geometry_plan.as_ref()
    }

    /// Install the congruence-transformed representation of the already
    /// declared reference-function Gram after an exact basis change. This is
    /// the only mutation seam for the Gram outside this type; it validates the
    /// new matrix before making the atomic replacement and never changes the
    /// reference metric provenance or geometry plan.
    pub(crate) fn install_transported_smooth_penalty(
        &mut self,
        penalty: Array2<f64>,
    ) -> Result<(), String> {
        self.smooth_penalty =
            Self::validate_reference_function_gram(penalty, self.basis_size(), false)?;
        Ok(())
    }

    /// Restore an exact Gram captured from this same atom by the line-search
    /// snapshot. The shape check prevents a stale snapshot from crossing a
    /// structural basis change; no eigendecomposition is repeated on this hot
    /// rollback path because the snapshot was copied from validated state.
    pub(crate) fn restore_smooth_penalty_snapshot(
        &mut self,
        penalty: &Array2<f64>,
    ) -> Result<(), String> {
        if penalty.dim() != self.smooth_penalty.dim() {
            return Err(format!(
                "SaeManifoldAtom::restore_smooth_penalty_snapshot: snapshot shape {:?} != live shape {:?}",
                penalty.dim(),
                self.smooth_penalty.dim()
            ));
        }
        self.smooth_penalty.assign(penalty);
        Ok(())
    }

    #[must_use = "build error must be handled"]
    pub fn new(
        name: impl Into<String>,
        basis_kind: SaeAtomBasisKind,
        latent_dim: usize,
        basis_values: Array2<f64>,
        basis_jacobian: Array3<f64>,
        decoder_coefficients: Array2<f64>,
        reference_roughness: SaeReferenceRoughness,
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
        if m == 0 {
            return Err("SaeManifoldAtom::new: basis width must be positive".into());
        }
        if p == 0 {
            return Err("SaeManifoldAtom::new: decoder output dimension must be positive".into());
        }
        let (smooth_penalty, reference_roughness_kind) = Self::materialize_reference_roughness(
            &basis_kind,
            latent_dim,
            basis_jacobian.view(),
            reference_roughness,
        )?;
        Ok(Self {
            name: name.into(),
            basis_kind,
            latent_dim,
            basis_values,
            decoder_coefficients,
            smooth_penalty,
            reference_roughness_kind,
            basis_jacobian,
            geometry_plan: None,
            basis_evaluator: None,
            basis_second_jet: None,
            decoder_frame: None,
            homotopy_eta: 1.0,
            chart_canonicalized: false,
            // Set only by `reduce_basis_to_subspace`; a freshly-built atom is
            // full-width (its decoder is already the un-reduced `M × p` block).
            reduced_column_map: None,
            // Stamped from the terminal `rho.log_ard` at fit finalization; a
            // freshly-built atom carries no fitted coordinate prior yet.
            ard_precisions: None,
        })
    }

    /// Construct an atom whose topology/caller already supplied the exact
    /// basis Gram of its declared reference-function seminorm.
    #[must_use = "build error must be handled"]
    pub fn new_with_provided_function_gram(
        name: impl Into<String>,
        basis_kind: SaeAtomBasisKind,
        latent_dim: usize,
        basis_values: Array2<f64>,
        basis_jacobian: Array3<f64>,
        decoder_coefficients: Array2<f64>,
        function_gram: Array2<f64>,
    ) -> Result<Self, String> {
        Self::new(
            name,
            basis_kind,
            latent_dim,
            basis_values,
            basis_jacobian,
            decoder_coefficients,
            SaeReferenceRoughness::ProvidedFunctionGram(function_gram),
        )
    }

    fn materialize_reference_roughness(
        basis_kind: &SaeAtomBasisKind,
        latent_dim: usize,
        basis_jacobian: ArrayView3<'_, f64>,
        reference_roughness: SaeReferenceRoughness,
    ) -> Result<(Array2<f64>, SaeReferenceRoughnessKind), String> {
        let (n, m, d) = basis_jacobian.dim();
        if d != latent_dim {
            return Err(format!(
                "SaeManifoldAtom::materialize_reference_roughness: basis Jacobian latent dimension {d} != declared {latent_dim}"
            ));
        }
        match reference_roughness {
            SaeReferenceRoughness::ProvidedFunctionGram(gram) => Ok((
                Self::validate_reference_function_gram(gram, m, false)?,
                SaeReferenceRoughnessKind::ProvidedFunctionGram,
            )),
            SaeReferenceRoughness::PoincareConformalDirichlet { reference_coords } => {
                if !matches!(basis_kind, SaeAtomBasisKind::Poincare) {
                    return Err(
                        "SaeManifoldAtom::materialize_reference_roughness: Poincare conformal-Dirichlet norm requires a Poincare atom"
                            .into(),
                    );
                }
                if n == 0 || latent_dim == 0 {
                    return Err(
                        "SaeManifoldAtom::materialize_reference_roughness: Poincare reference coordinates must be non-empty"
                            .into(),
                    );
                }
                if reference_coords.dim() != (n, latent_dim) {
                    return Err(format!(
                        "SaeManifoldAtom::materialize_reference_roughness: Poincare reference coordinates {:?}, expected ({n}, {latent_dim})",
                        reference_coords.dim()
                    ));
                }
                if reference_coords.iter().any(|value| !value.is_finite()) {
                    return Err(
                        "SaeManifoldAtom::materialize_reference_roughness: Poincare reference coordinates must be finite"
                            .into(),
                    );
                }
                let gram = gam_geometry::manifolds::poincare::conformal_dirichlet_penalty(
                    reference_coords.view(),
                    basis_jacobian,
                    POINCARE_REFERENCE_CURVATURE,
                )
                .map_err(|error| {
                    format!(
                        "SaeManifoldAtom::materialize_reference_roughness: Poincare conformal-Dirichlet Gram failed: {error}"
                    )
                })?;
                Ok((
                    Self::validate_reference_function_gram(gram, m, true)?,
                    SaeReferenceRoughnessKind::PoincareConformalDirichlet,
                ))
            }
        }
    }

    fn validate_reference_function_gram(
        gram: Array2<f64>,
        m: usize,
        require_positive_rank: bool,
    ) -> Result<Array2<f64>, String> {
        if gram.dim() != (m, m) {
            return Err(format!(
                "SaeManifoldAtom::validate_reference_function_gram: Gram {:?}, expected ({m}, {m})",
                gram.dim()
            ));
        }
        if gram.iter().any(|value| !value.is_finite()) {
            return Err(
                "SaeManifoldAtom::validate_reference_function_gram: Gram must be finite".into(),
            );
        }
        let scale = gram
            .iter()
            .fold(1.0_f64, |current, value| current.max(value.abs()));
        let tolerance = f64::EPSILON.sqrt() * scale * m.max(1) as f64;
        for i in 0..m {
            for j in 0..m {
                if (gram[[i, j]] - gram[[j, i]]).abs() > tolerance {
                    return Err(format!(
                        "SaeManifoldAtom::validate_reference_function_gram: Gram is not symmetric at ({i}, {j})"
                    ));
                }
            }
        }
        let sym = (&gram + &gram.t()) * 0.5;
        let (eigenvalues, eigenvectors) = sym
            .eigh(Side::Lower)
            .map_err(|error| {
                format!(
                    "SaeManifoldAtom::validate_reference_function_gram: eigendecomposition failed: {error}"
                )
            })?;
        let min_eigenvalue = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        let max_eigenvalue = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
        if min_eigenvalue < -tolerance {
            return Err(format!(
                "SaeManifoldAtom::validate_reference_function_gram: Gram is not positive semidefinite (minimum eigenvalue {min_eigenvalue})"
            ));
        }
        if require_positive_rank && max_eigenvalue <= tolerance {
            return Err(
                "SaeManifoldAtom::validate_reference_function_gram: declared seminorm has zero numerical rank"
                    .into(),
            );
        }
        if min_eigenvalue >= 0.0 {
            return Ok(sym);
        }
        // Roundoff-sized negative eigenvalues are projected to zero so the
        // installed Hessian is genuinely PSD rather than merely PSD up to a
        // tolerance. Larger negative directions were rejected above.
        let clipped = Array2::from_diag(&eigenvalues.mapv(|value| value.max(0.0)));
        Ok(eigenvectors.dot(&clipped).dot(&eigenvectors.t()))
    }

    pub fn with_basis_evaluator(mut self, evaluator: Arc<dyn SaeBasisEvaluator>) -> Self {
        self.basis_evaluator = Some(evaluator);
        self.basis_second_jet = None;
        self
    }

    /// Attach the exact geometry plan used to build this atom. The plan's kind,
    /// latent dimension, derived full width, and reference-function Gram must
    /// agree. Attachment is one-shot: replacing a plan could otherwise change
    /// the declared metric independently of the frozen Gram. There is no width
    /// or harmonic-order inference from the realized arrays.
    pub fn with_geometry_plan(mut self, plan: SaeAtomGeometryPlan) -> Result<Self, String> {
        if self.geometry_plan.is_some() {
            return Err(
                "SaeManifoldAtom::with_geometry_plan: geometry plan is already installed; replacement is forbidden"
                    .to_string(),
            );
        }
        if plan.kind() != &self.basis_kind || plan.latent_dim() != self.latent_dim {
            return Err(format!(
                "SaeManifoldAtom::with_geometry_plan: plan ({:?}, dim={}) disagrees with atom ({:?}, dim={})",
                plan.kind(),
                plan.latent_dim(),
                self.basis_kind,
                self.latent_dim
            ));
        }
        let planned_width = plan.basis_size()?;
        if planned_width != self.full_basis_size() {
            return Err(format!(
                "SaeManifoldAtom::with_geometry_plan: plan width {planned_width} disagrees with atom full width {}",
                self.full_basis_size()
            ));
        }
        let planned_penalty = plan.build_reference_penalty()?;
        if planned_penalty.dim() != self.smooth_penalty.dim() {
            return Err(format!(
                "SaeManifoldAtom::with_geometry_plan: plan reference Gram shape {:?} disagrees with atom Gram shape {:?}",
                planned_penalty.dim(),
                self.smooth_penalty.dim()
            ));
        }
        let scale = planned_penalty
            .iter()
            .chain(self.smooth_penalty.iter())
            .fold(1.0_f64, |current, value| current.max(value.abs()));
        let tolerance = f64::EPSILON.sqrt() * scale * planned_width.max(1) as f64;
        let max_difference = planned_penalty
            .iter()
            .zip(self.smooth_penalty.iter())
            .map(|(planned, installed)| (planned - installed).abs())
            .fold(0.0_f64, f64::max);
        if max_difference > tolerance {
            return Err(format!(
                "SaeManifoldAtom::with_geometry_plan: installed reference Gram differs from plan by {max_difference}, tolerance {tolerance}"
            ));
        }
        self.reference_roughness_kind = plan.reference_roughness_kind();
        self.geometry_plan = Some(plan);
        Ok(self)
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
    /// construction, flattening the outer penalized quasi-Laplace surface and stalling the solve.
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
    /// * frozen reference-function Gram `S̃_ref = Qᵀ S_ref Q`
    ///   (`smooth_penalty`),
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
        // S̃_ref = Qᵀ S_ref Q (r × r). This is the exact basis-change law for
        // the already-declared function seminorm; no metric or decoder is
        // re-estimated.
        let s_ref_red = q.t().dot(&self.smooth_penalty).dot(q);
        let s_ref_red = Self::validate_reference_function_gram(s_ref_red, r, false)?;
        let reduced_eval = SubspaceReducedEvaluator::new(inner, q.clone())?;
        let reduced_arc: Arc<dyn SaeBasisSecondJet> = Arc::new(reduced_eval);
        let base: Arc<dyn SaeBasisEvaluator> = reduced_arc.clone();

        self.basis_values = phi_red;
        self.basis_jacobian = jac_red;
        self.decoder_coefficients = dec_red;
        self.smooth_penalty = s_ref_red;
        self.basis_evaluator = Some(base);
        self.basis_second_jet = Some(reduced_arc);
        // The decoder frame is a profiled representation of the *previous* M×p
        // decoder; the column count just changed, so drop it and let the joint
        // fit re-activate it for the reduced block if still profitable.
        self.decoder_frame = None;
        // Record the inner→reduced column map so the reduced frame can be
        // re-expanded at the emission boundary (#2135). If this atom was ALREADY
        // reduced (`prev`: `M × r_prev`) and is being reduced again against its
        // now-`r_prev`-wide inner basis (`q`: `r_prev × r`), compose so the
        // stored map stays the true inner-width `M × r` congruence
        // `Φ_inner (Q_prev Q) = Φ̃`. (In practice the reduced design is full-rank
        // by construction, so a second reduction is a no-op skip; the
        // composition is defensive.)
        self.reduced_column_map = Some(match self.reduced_column_map.take() {
            Some(prev) => prev.dot(q),
            None => q.clone(),
        });
        Ok(())
    }

    /// Full-width decoder `B = Q B̃` (`M × p`) on this atom's UN-reduced inner
    /// basis. After a #1117 rank reduction the stored
    /// [`Self::decoder_coefficients`] is the reduced `B̃ = Qᵀ B` (`r × p`);
    /// re-expanding by the frozen column map `Q` recovers the minimum-norm
    /// full-width pre-image, which reconstructs IDENTICALLY on the standard
    /// inner basis: `Φ_inner (Q B̃) = (Φ_inner Q) B̃ = Φ̃ B̃`. Consumers that
    /// rebuild the fixed-width inner basis from emitted metadata (out-of-sample
    /// predict / steer / reconstruction) must decode against THIS so the reduced
    /// fit-conditioning frame never escapes (issue #2135). Returns a clone of the
    /// stored decoder unchanged when the atom was not reduced.
    pub fn full_width_decoder(&self) -> Array2<f64> {
        match &self.reduced_column_map {
            Some(q) => q.dot(&self.decoder_coefficients),
            None => self.decoder_coefficients.clone(),
        }
    }

    /// Full inner-basis width `M` — the row count of
    /// [`Self::full_width_decoder`]. Equals [`Self::basis_size`] unless the atom
    /// was #1117 rank-reduced, in which case it is the un-reduced inner width
    /// `M = Q.nrows()` (`basis_size` is then the reduced `r`).
    pub fn full_basis_size(&self) -> usize {
        match &self.reduced_column_map {
            Some(q) => q.nrows(),
            None => self.basis_size(),
        }
    }

    /// Lift a reduced-frame decoder covariance `Cov(vec B̃)`
    /// (`(r·p) × (r·p)`, basis-major flat layout `b·p + c`, matching
    /// `assemble_shape_uncertainty`) to the full inner-basis frame
    /// `Cov(vec B) = (Q ⊗ I_p) Cov(vec B̃) (Q ⊗ I_p)ᵀ` (`(M·p) × (M·p)`) — the
    /// exact posterior covariance of [`Self::full_width_decoder`], so the emitted
    /// covariance stays width-consistent with the re-expanded decoder (#2135).
    /// Returns the input unchanged when the atom was not reduced. `p` is the
    /// ambient output dimension the covariance is laid out against.
    pub fn lift_reduced_decoder_covariance(
        &self,
        cov_reduced: &Array2<f64>,
        p: usize,
    ) -> Result<Array2<f64>, String> {
        let Some(q) = self.reduced_column_map.as_ref() else {
            return Ok(cov_reduced.clone());
        };
        let (m, r) = q.dim();
        if p == 0 {
            return Err(
                "SaeManifoldAtom::lift_reduced_decoder_covariance: p must be positive".to_string(),
            );
        }
        if cov_reduced.dim() != (r * p, r * p) {
            return Err(format!(
                "SaeManifoldAtom::lift_reduced_decoder_covariance: covariance dim {:?} != reduced ({}, {})",
                cov_reduced.dim(),
                r * p,
                r * p
            ));
        }
        // First contract the left basis index: T[m1·p+c1, r2·p+c2]
        //   = Σ_{r1} Q[m1,r1] · Cov[r1·p+c1, r2·p+c2].
        let mut tmp = Array2::<f64>::zeros((m * p, r * p));
        for m1 in 0..m {
            for c1 in 0..p {
                let out_row = m1 * p + c1;
                for col in 0..(r * p) {
                    let mut acc = 0.0_f64;
                    for r1 in 0..r {
                        acc += q[[m1, r1]] * cov_reduced[[r1 * p + c1, col]];
                    }
                    tmp[[out_row, col]] = acc;
                }
            }
        }
        // Then the right basis index: Cov_full[row, m2·p+c2]
        //   = Σ_{r2} T[row, r2·p+c2] · Q[m2,r2].
        let mut lifted = Array2::<f64>::zeros((m * p, m * p));
        for row in 0..(m * p) {
            for m2 in 0..m {
                for c2 in 0..p {
                    let mut acc = 0.0_f64;
                    for r2 in 0..r {
                        acc += tmp[[row, r2 * p + c2]] * q[[m2, r2]];
                    }
                    lifted[[row, m2 * p + c2]] = acc;
                }
            }
        }
        Ok(lifted)
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
        // Clone the `Arc` handle (a cheap refcount bump) so the evaluator is no
        // longer borrowed from `self`, freeing the mutable borrow the in-place
        // fill needs below.
        let Some(evaluator) = self.basis_evaluator.clone() else {
            return Ok(());
        };
        // Curvature-homotopy dial (#1007): at the default `η = 1` this is the
        // un-dialed basis (`evaluate_phi_eta` returns the unscaled Φ / jet
        // bit-for-bit), so the production path is unchanged. For `η < 1` the
        // tracker scales the curved columns toward the base-topology relaxation; the
        // `dphi_deta` / `djet_deta` channels are discarded here (the predictor
        // forms `∂g/∂η` separately from a dedicated evaluation).
        if self.homotopy_eta == 1.0 {
            // Hot path: fill the atom's already-correctly-shaped Φ / jet buffers
            // in place. This is called on EVERY β-Newton line-search trial (via
            // `apply_newton_step`), so avoiding the fresh `(N, M)` + `(N, M, d)`
            // allocation here removes the dominant per-trial allocation churn.
            // The evaluator validates the buffer shapes and errors on mismatch —
            // the same guard the freshly-allocated path applied.
            evaluator.evaluate_into(&mut self.basis_values, &mut self.basis_jacobian, coords)?;
        } else {
            let evaluated = evaluator.evaluate_phi_eta(coords, self.homotopy_eta)?;
            let (phi, jet) = (evaluated.phi, evaluated.jet);
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
        }
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
    /// of frame degrees of freedom that must enter the quasi-Laplace score
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
    }

    /// #2133 — the pure second coordinate derivative `∂²g_k/∂t_{ik,axis}²`
    /// contracted through the decoder, for one row/axis, given the atom's second
    /// jet `(n, M, d, d)` (from [`SaeManifoldTerm::atom_second_jets`]). This is the
    /// `f''` leg the Gauss-Newton `htt = J̃J̃ᵀ` omits; the SURE within-basin
    /// divergence correction contracts it against the metric residual `M·r`.
    /// Mirrors [`Self::fill_decoded_derivative_row`] exactly (same decoder axpy),
    /// only the basis weight is the diagonal
    /// second jet `∂²Φ/∂t_axis²` instead of the first jet `∂Φ/∂t_axis`.
    pub(crate) fn fill_decoded_second_derivative_row(
        &self,
        second_jet: &Array4<f64>,
        row: usize,
        latent_axis: usize,
        out: &mut [f64],
    ) {
        let m = self.basis_size();
        assert_eq!(out.len(), self.output_dim());
        for slot in out.iter_mut() {
            *slot = 0.0;
        }
        for basis_col in 0..m {
            let d2phi = second_jet[[row, basis_col, latent_axis, latent_axis]];
            if d2phi == 0.0 {
                continue;
            }
            let dec = self.decoder_coefficients.row(basis_col);
            for (o, &d) in out.iter_mut().zip(dec.iter()) {
                *o += d2phi * d;
            }
        }
    }

    /// Frobenius scale of the physical decoder contribution.
    pub(crate) fn contribution_frobenius_scale(&self) -> f64 {
        let decoder_norm = self
            .decoder_coefficients
            .iter()
            .map(|v| v * v)
            .sum::<f64>()
            .sqrt();
        if decoder_norm.is_finite() {
            decoder_norm
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gam_math::special::{bessel_i0_log_and_ratio, bessel_i0_log_minus_abs_and_ratio};

    /// The overflow-free `(log I0(η), I1(η)/I0(η))` must satisfy the exact Bessel
    /// identity `d/dη log I0(η) = I1(η)/I0(η)` on BOTH the small-argument series
    /// branch and the large-argument scaled-polynomial branch — including
    /// `η ≫ 709`, where the naive `bessel_i0(η).ln()` / `bessel_i1/bessel_i0`
    /// overflow to `+inf` and divide to `NaN` (the #1113 iter-0 ρ-gradient poison).
    /// The returned ratio is the ordinary first derivative of `log I0`, so a
    /// central difference must reproduce it. Periodic ARD uses the separate
    /// centered, log-scale derivative because `η·(ratio−1)` is numerically
    /// singular after this ordinary ratio rounds to one.
    #[test]
    fn bessel_log_i0_and_ratio_is_overflow_free_and_derivative_consistent() {
        // η spanning both branches and well past the e^η overflow threshold.
        for &eta in &[
            0.25_f64, 1.0, 3.0, 3.74, 3.76, 5.0, 12.0, 50.0, 400.0, 900.0,
        ] {
            let (log_i0, ratio) = bessel_i0_log_and_ratio(eta);
            assert!(
                log_i0.is_finite() && ratio.is_finite(),
                "bessel_i0_log_and_ratio({eta}) must be finite, got log_i0={log_i0}, ratio={ratio}"
            );
            // Central difference of log I0 must match the returned ratio.
            let h = 1.0e-4 * eta.max(1.0);
            let (lp, _) = bessel_i0_log_and_ratio(eta + h);
            let (lm, _) = bessel_i0_log_and_ratio(eta - h);
            let fd = (lp - lm) / (2.0 * h);
            let err = (fd - ratio).abs();
            let tol = 1.0e-6 + 1.0e-5 * ratio.abs();
            assert!(
                err <= tol,
                "d/dη log I0({eta}) mismatch: analytic ratio={ratio:.12e}, fd={fd:.12e}, err={err:.3e}"
            );
            // I1/I0 ∈ (0, 1) and → 1 as η → ∞.
            assert!(
                ratio > 0.0 && ratio < 1.0,
                "I1/I0({eta}) must lie in (0,1), got {ratio}"
            );
        }
        // Known reference value at η = 1: I0(1)=1.26606587..., I1(1)=0.56515910...
        let (log_i0_1, ratio_1) = bessel_i0_log_and_ratio(1.0);
        assert!(
            (log_i0_1 - 1.266_065_877_752_008_f64.ln()).abs() < 1.0e-6,
            "log I0(1) reference mismatch, got {log_i0_1}"
        );
        assert!(
            (ratio_1 - 0.446_389_221_869_1_f64).abs() < 1.0e-6,
            "I1/I0(1) reference mismatch, got {ratio_1}"
        );
    }

    #[test]
    fn centered_bessel_log_preserves_thin_ring_cancellation() {
        for &eta in &[0.0_f64, 1.0, 3.74, 3.76, 900.0] {
            let (log_i0, ratio) = bessel_i0_log_and_ratio(eta);
            let (centered, centered_ratio) = bessel_i0_log_minus_abs_and_ratio(eta);
            assert!((centered + eta.abs() - log_i0).abs() < 2.0e-13);
            assert_eq!(centered_ratio, ratio);
        }

        // Forming log(I0(eta))-eta cannot retain this O(log eta) remainder
        // once eta dwarfs the f64 mantissa. The centered branch never forms
        // either exponentially/linearly large term and remains informative.
        for &eta in &[1.0e20_f64, 1.0e100, 1.0e300] {
            let (centered, ratio) = bessel_i0_log_minus_abs_and_ratio(eta);
            let asymptotic = -0.5 * (std::f64::consts::TAU * eta).ln();
            assert!(centered.is_finite() && ratio.is_finite());
            assert!((centered - asymptotic).abs() < 2.0e-8);
            assert!((0.0..=1.0).contains(&ratio));
        }
    }

    #[test]
    fn provided_reference_function_gram_rejects_asymmetry_and_negative_energy() {
        let phi = Array2::<f64>::zeros((2, 2));
        let jet = Array3::<f64>::zeros((2, 2, 1));
        let decoder = Array2::<f64>::ones((2, 1));

        let asymmetric = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 0.0, 1.0]).unwrap();
        let err = SaeManifoldAtom::new_with_provided_function_gram(
            "asymmetric",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi.clone(),
            jet.clone(),
            decoder.clone(),
            asymmetric,
        )
        .expect_err("an asymmetric reference Gram must be rejected");
        assert!(err.contains("not symmetric"), "unexpected error: {err}");

        let indefinite = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, -1.0]).unwrap();
        let err = SaeManifoldAtom::new_with_provided_function_gram(
            "indefinite",
            SaeAtomBasisKind::EuclideanPatch,
            1,
            phi,
            jet,
            decoder,
            indefinite,
        )
        .expect_err("an indefinite reference Gram must be rejected");
        assert!(
            err.contains("not positive semidefinite"),
            "unexpected error: {err}"
        );
    }

    // Build an atom over the degree-2 monomial basis `[1, t, t²]` at the given
    // latent coordinates, with decoder `γ(t) = t + t²`. Poincare atoms declare
    // the conformal-Dirichlet reference norm at these coordinates; other atoms
    // declare the supplied second-derivative function Gram.
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
        let reference_roughness = if matches!(kind, SaeAtomBasisKind::Poincare) {
            SaeReferenceRoughness::PoincareConformalDirichlet {
                reference_coords: Array2::from_shape_vec((n, 1), ts.to_vec()).unwrap(),
            }
        } else {
            SaeReferenceRoughness::ProvidedFunctionGram(penalty)
        };
        SaeManifoldAtom::new(
            "atom",
            kind,
            1,
            basis_values,
            basis_jacobian,
            decoder,
            reference_roughness,
        )
        .unwrap()
    }

    // #2135 — a #1117 rank-reduced circle/periodic atom stores a REDUCED decoder
    // `B̃ = Qᵀ B` (`r × p`) in a fit-internal eigenvector frame `Q` (`M × r`),
    // while every out-of-sample / reconstruct consumer rebuilds the STANDARD
    // `M`-column inner design `[1, sin, cos]`. The reduced frame must be
    // re-expanded at the emission boundary (`full_width_decoder` / `full_basis_size`)
    // so held-out reconstruction on the full inner basis is IDENTICAL to the
    // reduced fit and the decoder width matches the rebuilt basis width — not the
    // "M=2 vs M=3" mismatch the raw reduced block produced.
    #[test]
    fn reduced_periodic_decoder_re_expands_for_full_basis_oos() {
        // Inner circle basis `[1, sin 2πt, cos 2πt]` (M = 3), evaluated on a
        // training grid; the atom is built directly on that design.
        let eval = PeriodicHarmonicEvaluator::new(3).unwrap();
        let train_t = Array2::from_shape_vec((5, 1), vec![0.05, 0.23, 0.41, 0.66, 0.88]).unwrap();
        let (phi_train, jac_train) = eval.evaluate(train_t.view()).unwrap();
        let p = 2usize;
        // Arbitrary full-width decoder `B` (M = 3 rows, p = 2 cols).
        let decoder = Array2::from_shape_vec((3, p), vec![0.5, -0.2, 1.3, 0.7, -0.9, 0.4]).unwrap();
        let mut penalty = Array2::<f64>::zeros((3, 3));
        penalty[[1, 1]] = 1.0;
        penalty[[2, 2]] = 1.0;
        let mut atom = SaeManifoldAtom::new_with_provided_function_gram(
            "circle",
            SaeAtomBasisKind::Periodic,
            1,
            phi_train,
            jac_train,
            decoder.clone(),
            penalty,
        )
        .unwrap()
        .with_basis_second_jet(Arc::new(eval.clone()));

        // A GENUINE non-axis-aligned orthonormal column map `Q` (M = 3, r = 2),
        // built by Gram–Schmidt on two non-basis directions — the analogue of the
        // eigenvector remix `reduce_atoms_to_data_supported_rank` freezes, NOT an
        // axis-aligned `[sin, cos]` selector (the fabricated frame #2218 was
        // rejected for).
        let v1 = [1.0_f64, 1.0, 0.0];
        let v2 = [0.0_f64, 1.0, 1.0];
        let n1 = (v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]).sqrt();
        let u1 = [v1[0] / n1, v1[1] / n1, v1[2] / n1];
        let dot = v2[0] * u1[0] + v2[1] * u1[1] + v2[2] * u1[2];
        let w2 = [
            v2[0] - dot * u1[0],
            v2[1] - dot * u1[1],
            v2[2] - dot * u1[2],
        ];
        let n2 = (w2[0] * w2[0] + w2[1] * w2[1] + w2[2] * w2[2]).sqrt();
        let u2 = [w2[0] / n2, w2[1] / n2, w2[2] / n2];
        let q =
            Array2::from_shape_vec((3, 2), vec![u1[0], u2[0], u1[1], u2[1], u1[2], u2[2]]).unwrap();

        atom.reduce_basis_to_subspace(&q).unwrap();

        // After reduction the stored decoder is the reduced `r = 2` block, but the
        // full inner width is still `M = 3`.
        assert_eq!(atom.basis_size(), 2, "reduced design width r");
        assert_eq!(atom.full_basis_size(), 3, "un-reduced inner width M");
        let full = atom.full_width_decoder();
        assert_eq!(full.dim(), (3, p), "full-width decoder is M×p");
        // `full == Q · B̃` exactly.
        let expected_full = q.dot(&atom.decoder_coefficients);
        for i in 0..3 {
            for c in 0..p {
                assert!(
                    (full[[i, c]] - expected_full[[i, c]]).abs() <= 1e-14,
                    "full_width_decoder[{i},{c}] must equal (Q·B̃)"
                );
            }
        }
        // The reduced block width (2) does NOT match the rebuilt inner basis width
        // (3): decoding the standard `[1,sin,cos]` design by `B̃` is the exact
        // "M=2 vs M=3" defect; the full-width decoder resolves it.
        assert_ne!(atom.decoder_coefficients.nrows(), 3);
        assert_eq!(full.nrows(), 3);

        // HELD-OUT reconstruction fidelity: on a fresh OOS grid, decoding the
        // rebuilt full inner design by `full_width_decoder` reproduces the reduced
        // fit's reconstruction `Φ̃ · B̃` bit-for-bit.
        let oos_t = Array2::from_shape_vec((4, 1), vec![0.13, 0.37, 0.59, 0.95]).unwrap();
        let (phi_oos, _) = eval.evaluate(oos_t.view()).unwrap();
        let recon_full = phi_oos.dot(&full); // (4 × p) on full inner basis
        let phi_tilde = phi_oos.dot(&q); // reduced design Φ̃ = Φ·Q
        let recon_reduced = phi_tilde.dot(&atom.decoder_coefficients); // (4 × p)
        for i in 0..4 {
            for c in 0..p {
                assert!(
                    (recon_full[[i, c]] - recon_reduced[[i, c]]).abs() <= 1e-12,
                    "OOS reconstruction[{i},{c}]: full-basis {} != reduced fit {}",
                    recon_full[[i, c]],
                    recon_reduced[[i, c]]
                );
            }
        }

        // Covariance lift `(Q ⊗ I_p)` keeps the emitted covariance width-consistent
        // with the full decoder and preserves the exact posterior band variance.
        // A fixed SPD reduced covariance `(r·p = 4)²`.
        let a = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 0.2, 0.0, 0.1, //
                0.2, 1.5, 0.3, 0.0, //
                0.0, 0.3, 2.0, 0.4, //
                0.1, 0.0, 0.4, 1.2, //
            ],
        )
        .unwrap();
        let cov_red = a.dot(&a.t()); // SPD (r·p × r·p)
        let lifted = atom.lift_reduced_decoder_covariance(&cov_red, p).unwrap();
        assert_eq!(lifted.dim(), (3 * p, 3 * p), "lifted covariance is (M·p)²");
        // Explicit `K = Q ⊗ I_p` congruence check.
        let mut k = Array2::<f64>::zeros((3 * p, 2 * p));
        for m in 0..3 {
            for r in 0..2 {
                for c in 0..p {
                    k[[m * p + c, r * p + c]] = q[[m, r]];
                }
            }
        }
        let expected_cov = k.dot(&cov_red).dot(&k.t());
        for i in 0..(3 * p) {
            for j in 0..(3 * p) {
                assert!(
                    (lifted[[i, j]] - expected_cov[[i, j]]).abs() <= 1e-12,
                    "lifted covariance[{i},{j}] must equal (Q⊗I)·cov·(Q⊗I)ᵀ"
                );
            }
        }
        // Band variance is invariant under the lift: for the first OOS row the
        // per-channel variance in the reduced frame (Φ̃, cov_red) equals the full
        // frame (Φ_inner, lifted).
        let phi_inner_row = phi_oos.row(0);
        let phi_tilde_row = phi_tilde.row(0);
        for c in 0..p {
            let mut var_red = 0.0_f64;
            for r1 in 0..2 {
                for r2 in 0..2 {
                    var_red +=
                        phi_tilde_row[r1] * phi_tilde_row[r2] * cov_red[[r1 * p + c, r2 * p + c]];
                }
            }
            let mut var_full = 0.0_f64;
            for m1 in 0..3 {
                for m2 in 0..3 {
                    var_full +=
                        phi_inner_row[m1] * phi_inner_row[m2] * lifted[[m1 * p + c, m2 * p + c]];
                }
            }
            assert!(
                (var_red - var_full).abs() <= 1e-12,
                "channel {c} band variance must be invariant under the lift: {var_red} vs {var_full}"
            );
        }
    }

    #[test]
    fn poincare_d1_uses_hyperbolic_conformal_dirichlet() {
        let ts = [0.1_f64, 1.5, 3.0];
        let mut poincare = monomial_atom(SaeAtomBasisKind::Poincare, &ts);
        let coords = Array2::from_shape_vec((ts.len(), 1), ts.to_vec()).unwrap();

        // Exact wiring: the effective Gram IS the geometry crate's hyperbolic
        // conformal Dirichlet Gram at unit curvature.
        let expected = gam_geometry::manifolds::poincare::conformal_dirichlet_penalty(
            coords.view(),
            poincare.basis_jacobian.view(),
            POINCARE_REFERENCE_CURVATURE,
        )
        .unwrap();
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (poincare.smooth_penalty[[i, j]] - expected[[i, j]]).abs() <= 1e-12,
                    "Poincaré penalty[{i},{j}]={} must equal conformal-Dirichlet {}",
                    poincare.smooth_penalty[[i, j]],
                    expected[[i, j]]
                );
            }
        }

        // The `d = 1` hyperbolic pullback is the ORDER-1 Dirichlet Gram of the
        // flat first jet scaled by the constant `G ≡ 1/2` (the tangent chart is
        // intrinsically flat but the coordinate runs at half arc length,
        // geodesic distance `= 2|t|`). Verify `S = ½ Σ_n ∂Φ(t_n) ∂Φ(t_n)ᵀ`
        // exactly — a precise check that the hyperbolic metric (not the raw
        // order-2 second-derivative Gram, whose only nonzero entry is [2,2]=1)
        // is what got installed.
        let jac = &poincare.basis_jacobian;
        let mut flat = Array2::<f64>::zeros((3, 3));
        for row in 0..ts.len() {
            for i in 0..3 {
                for j in 0..3 {
                    flat[[i, j]] += jac[[row, i, 0]] * jac[[row, j, 0]];
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                assert!(
                    (expected[[i, j]] - 0.5 * flat[[i, j]]).abs()
                        <= 1e-9 * (1.0 + flat[[i, j]].abs()),
                    "d=1 hyperbolic Dirichlet Gram[{i},{j}]={} must equal ½·flat {}",
                    expected[[i, j]],
                    0.5 * flat[[i, j]]
                );
            }
        }
        assert_eq!(
            poincare.reference_roughness_kind,
            SaeReferenceRoughnessKind::PoincareConformalDirichlet
        );
        assert!(
            poincare.smooth_penalty[[1, 1]] > 1e-6,
            "Dirichlet roughness must charge the linear column; got {}",
            poincare.smooth_penalty[[1, 1]]
        );

        // The declaration is frozen. Neither decoder rescaling nor changing
        // the live first jet silently rebuilds a moving metric Gram.
        let frozen = poincare.smooth_penalty.clone();
        poincare
            .decoder_coefficients
            .mapv_inplace(|value| value * 9.0);
        poincare.basis_jacobian.fill(17.0);
        assert_eq!(poincare.smooth_penalty, frozen);
    }
}
