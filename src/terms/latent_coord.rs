//! `LatentCoord` — per-row latent coordinates as a first-class gamfit parameter.
//!
//! See `proposals/latent_coord.md` for the full design.
//!
//! The Riemannian update path follows manifold GPLVM practice (mGPLVM;
//! Jensen/Kao/Tran/Stevenson 2020 and related head-direction / population
//! manifold work): angular, spherical, and product-topology latents are
//! updated on their natural manifold instead of as Euclidean coordinates
//! with basis-side periodic hacks. Retractions and Euclidean-to-Riemannian
//! Hessian conversion follow Absil/Mahony/Sepulchre (2008) and the Manopt /
//! Pymanopt implementation pattern. In the audit-revised gauge framing, the
//! Riemannian update is itself a gauge restriction: Circle/Sphere/Torus
//! structure identifies the latent up to the corresponding global isometry
//! (for example one rotation per cycle), not up to the full diffeomorphism
//! group of an unconstrained Euclidean latent chart.
//!
//! ## Summary
//!
//! `LatentCoordValues` is the structural sibling of [`SpatialLogKappaCoords`]
//! (see [`crate::terms::smooth`]). Both store a flat `Array1<f64>` that the
//! REML/IFT outer loop treats as *design-moving, non-penalty-like*
//! hyper-coordinates. `SpatialLogKappaCoords` holds one or more kernel-shape
//! coordinates per spatial term. `LatentCoordValues`
//! holds an `N × d` matrix of per-row latent coordinates `t_n ∈ ℝ^d`.
//!
//! For a Duchon (or any radial) basis:
//!
//! ```text
//! Φ_{n,k} = φ(‖t_n − c_k‖),
//! ∂Φ_{n,k}/∂t_n = φ'(r_{nk}) · (t_n − c_k) / r_{nk}.
//! ```
//!
//! The radial-gradient `φ'(r)` is the same scalar the kernel-shape machinery already
//! computes via [`crate::basis::duchon_radial_jets`]; the chain rule
//! `(t_n − c_k)/r_{nk}` is what differs between "differentiate against the
//! kernel scale" and "differentiate against the first kernel argument t".
//! Everything downstream of `HyperDesignDerivative::from_implicit` (matrix-free
//! Newton, IFT cache, persistent warm-start, REML/LAML evaluation) is reused
//! verbatim.
//!
//! ## Gauge fixing
//!
//! The bare data-fit `½‖y − Φ(t)β‖²` is invariant under any diffeomorphism
//! `t ↦ φ(t)` (absorb into a re-fit β), so the inner Hessian in the latent
//! block is singular and IFT breaks. [`LatentIdMode`] enumerates the
//! gauge-fix penalties exposed at the configuration layer:
//!
//! * [`LatentIdMode::AuxPrior`] — iVAE-style auxiliary-conditional prior
//!   `R_id(t,u) = ½ μ ‖t − ĥ(u)‖²` where `ĥ` is a small ridge / linear map
//!   fit internally against the auxiliary `u`. `μ` is REML-selectable like a
//!   smoothing parameter only when the marginal likelihood includes the
//!   log-`μ` normalizer, `ĥ` is at least C¹, and the conditional precision is
//!   positive-definite on the anchored subspace. Under those regularity
//!   conditions this is the principled identifiability fix (Khemakhem et al.
//!   2020).
//! * [`LatentIdMode::DimSelection`] — ARD on each latent axis. One ridge
//!   penalty per axis; REML drives unused axes' precision to infinity only
//!   after `AuxPrior` or a future isometry prior fixes the gauge.
//! * [`LatentIdMode::None`] — no gauge fix. Useful only as an explicit
//!   opt-out; the caller is responsible for separately providing a unique
//!   inner minimum (e.g. via a custom penalty).
//!
//! `IsometryToReference` is deferred to a follow-up (see proposal §4(b)).

use crate::terms::basis::{BasisError, RadialScalarKind};
use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3};
use std::sync::atomic::{AtomicU64, Ordering};

const TWO_PI: f64 = std::f64::consts::PI * 2.0;
const SPHERE_NORMAL_PIN: f64 = 1.0;
static NEXT_LATENT_COORD_ID: AtomicU64 = AtomicU64::new(1);

fn next_latent_coord_id() -> u64 {
    NEXT_LATENT_COORD_ID.fetch_add(1, Ordering::Relaxed)
}

/// Choice of auxiliary-prior conditional mean estimator `ĥ(u)`.
///
/// `Ridge` is the cheap default that closes form (one `K_u × K_u` solve);
/// `Linear` is equivalent to `Ridge` with zero ridge and is intended for
/// auxiliaries `u` that are already low-dimensional and well-conditioned.
#[derive(Debug, Clone, Copy)]
pub enum AuxPriorFamily {
    /// Ridge regression `t ≈ U · A` with a small diagonal regularizer.
    /// The default ridge strength is `1e-6 · trace(UᵀU)/p`, which is
    /// numerically benign and never under-constrains the fit when
    /// `n_obs > p`.
    Ridge,
    /// Plain linear projection (no ridge). Errors out at construction if
    /// `UᵀU` is singular.
    Linear,
}

/// Strength of the auxiliary-prior identifiability penalty.
///
/// `Auto` defers the choice to REML — the strength is added to the outer
/// vector as one extra `ρ`-axis (one log-precision per `LatentCoord`). When
/// the caller supplies an explicit `Fixed(μ)` the strength is held constant
/// throughout the fit; useful for warm-starts and reproducibility. The REML
/// path is valid only with the prior normalizer included, a C¹ conditional
/// mean map, and positive-definite precision on the anchored subspace.
#[derive(Debug, Clone, Copy)]
pub enum AuxPriorStrength {
    Auto,
    Fixed(f64),
}

/// Identifiability / gauge-fix mode for a [`LatentCoordValues`] block.
///
/// `AuxPrior` is currently the only standalone gauge-fixing mode; see the
/// module docstring. `DimSelection` must be paired with `AuxPrior` (or a
/// future isometry mode) by higher-level assembly before fitting.
#[derive(Debug, Clone)]
pub enum LatentIdMode {
    /// Conditional Gaussian prior `p(t | u)` with mean `ĥ(u)` fit by
    /// `family`. The penalty contribution is
    /// `R_id = ½ μ · ‖t − ĥ(u)‖²`. `u` has shape `(n_obs, p)`. If
    /// `strength == Auto`, REML selection of `μ` requires the log-`μ`
    /// normalizer, C¹ regularity of `ĥ`, and positive-definiteness on the
    /// subspace anchored by `u`.
    AuxPrior {
        u: Array2<f64>,
        family: AuxPriorFamily,
        strength: AuxPriorStrength,
    },
    /// Auxiliary prior plus ARD over latent axes. `AuxPrior` supplies the
    /// identifiability anchor; `init_log_precision` seeds the per-axis ARD
    /// coordinates.
    AuxPriorDimSelection {
        u: Array2<f64>,
        family: AuxPriorFamily,
        strength: AuxPriorStrength,
        init_log_precision: Option<Array1<f64>>,
    },
    /// ARD over latent axes. One ridge penalty per latent axis; the per-axis
    /// log-precision joins the outer ρ vector. `init_log_precision` seeds
    /// the per-axis ρ — a vector of length `d`. `None` defaults to a flat
    /// zero seed (precision = 1 on every axis).
    DimSelection {
        init_log_precision: Option<Array1<f64>>,
    },
    /// No gauge fix. Inner Hessian is rank-deficient; results are not
    /// uniquely defined. Intended only for the explicit "I supply my own
    /// gauge constraint via the smoothing penalty" pathway.
    None,
}

/// Natural manifold for per-row latent-coordinate updates.
///
/// `Euclidean` preserves the original additive update. `Circle` is a scalar
/// angular coordinate wrapped modulo `2π`. `Sphere { dim }` is the embedded
/// unit sphere in `R^dim`, with retraction `(t + ξ) / ||t + ξ||`. `Product`
/// composes these blockwise; inside a product, `Euclidean` denotes one
/// unconstrained scalar axis.
#[derive(Debug, Clone, PartialEq)]
pub enum LatentManifold {
    /// Unconstrained `R^d` — the current default.
    Euclidean,
    /// Scalar angular coordinate on `S^1`, represented in radians.
    Circle,
    /// Embedded unit sphere `S^(dim-1)`.
    Sphere { dim: usize },
    /// Closed interval in `R`; the retraction clamps to the boundary.
    Interval { lo: f64, hi: f64 },
    /// Product manifold, split block-by-block in row-major ambient storage.
    Product(Vec<LatentManifold>),
    /// Product manifold with explicit per-axis trust-region metric weights.
    ///
    /// Without per-axis weighting, a Product of Circle + Interval treats
    /// 1 radian as commensurate with the entire bounded range. With weights
    /// = 1/scale², the trust-region radius respects each axis's natural unit.
    ProductWithMetric {
        manifolds: Vec<LatentManifold>,
        weights: Vec<f64>,
    },
}

impl Default for LatentManifold {
    fn default() -> Self {
        Self::Euclidean
    }
}

impl LatentManifold {
    pub fn is_euclidean(&self) -> bool {
        matches!(self, Self::Euclidean)
    }

    pub fn ambient_dim(&self, fallback_dim: usize) -> usize {
        match self {
            Self::Euclidean => fallback_dim,
            Self::Circle | Self::Interval { .. } => 1,
            Self::Sphere { dim } => *dim,
            Self::Product(parts) | Self::ProductWithMetric { manifolds: parts, .. } => {
                parts.iter().map(|part| part.ambient_dim(1)).sum()
            }
        }
    }

    /// Per-axis weights for the Riemannian trust-region metric.
    ///
    /// Defaults use `1/scale²`: Circle scale is `2π`, Sphere scale is `π`,
    /// Interval scale is `hi - lo`, and Euclidean scale is `1`. Product
    /// manifolds recurse and concatenate; [`Self::ProductWithMetric`] uses
    /// the caller-supplied weights directly.
    pub fn metric_weights(&self) -> Vec<f64> {
        match self {
            Self::Euclidean => vec![1.0],
            Self::Circle => vec![1.0 / (TWO_PI * TWO_PI)],
            Self::Sphere { dim } => {
                let w = 1.0 / (std::f64::consts::PI * std::f64::consts::PI);
                vec![w; *dim]
            }
            Self::Interval { lo, hi } => {
                let scale = hi - lo;
                vec![1.0 / (scale * scale)]
            }
            Self::Product(parts) => {
                let mut out = Vec::with_capacity(self.ambient_dim(1));
                for part in parts {
                    out.extend(part.metric_weights());
                }
                out
            }
            Self::ProductWithMetric { manifolds, weights } => {
                let expected: usize = manifolds.iter().map(|part| part.ambient_dim(1)).sum();
                assert_eq!(
                    weights.len(),
                    expected,
                    "LatentManifold::ProductWithMetric weights length must match ambient dimension"
                );
                weights.clone()
            }
        }
    }

    /// Project an arbitrary ambient point back to the manifold.
    pub fn project_point(&self, t: ArrayView1<'_, f64>) -> Array1<f64> {
        match self {
            Self::Euclidean => t.to_owned(),
            Self::Circle => {
                let mut out = Array1::<f64>::zeros(1);
                out[0] = wrap_angle(t[0]);
                out
            }
            Self::Sphere { dim } => {
                debug_assert_eq!(t.len(), *dim);
                normalize_or_axis(t, *dim)
            }
            Self::Interval { lo, hi } => {
                let mut out = Array1::<f64>::zeros(1);
                out[0] = t[0].clamp(*lo, *hi);
                out
            }
            Self::Product(parts) | Self::ProductWithMetric { manifolds: parts, .. } => {
                let mut out = Array1::<f64>::zeros(t.len());
                let mut offset = 0_usize;
                for part in parts {
                    let dim = part.ambient_dim(1);
                    let projected = part.project_point(t.slice(ndarray::s![offset..offset + dim]));
                    for a in 0..dim {
                        out[offset + a] = projected[a];
                    }
                    offset += dim;
                }
                debug_assert_eq!(offset, t.len());
                out
            }
        }
    }

    /// Retraction `R_t(ξ)`, using closed-form analytic maps for every variant.
    pub fn retract(&self, t: ArrayView1<'_, f64>, xi: ArrayView1<'_, f64>) -> Array1<f64> {
        debug_assert_eq!(t.len(), xi.len());
        match self {
            Self::Euclidean => {
                let mut out = t.to_owned();
                for a in 0..out.len() {
                    out[a] += xi[a];
                }
                out
            }
            Self::Circle => {
                let mut out = Array1::<f64>::zeros(1);
                out[0] = wrap_angle(t[0] + xi[0]);
                out
            }
            Self::Sphere { dim } => {
                debug_assert_eq!(t.len(), *dim);
                let mut y = Array1::<f64>::zeros(*dim);
                for a in 0..*dim {
                    y[a] = t[a] + xi[a];
                }
                normalize_or_axis(y.view(), *dim)
            }
            Self::Interval { lo, hi } => {
                let mut out = Array1::<f64>::zeros(1);
                out[0] = (t[0] + xi[0]).clamp(*lo, *hi);
                out
            }
            Self::Product(parts) | Self::ProductWithMetric { manifolds: parts, .. } => {
                let mut out = Array1::<f64>::zeros(t.len());
                let mut offset = 0_usize;
                for part in parts {
                    let dim = part.ambient_dim(1);
                    let next = part.retract(
                        t.slice(ndarray::s![offset..offset + dim]),
                        xi.slice(ndarray::s![offset..offset + dim]),
                    );
                    for a in 0..dim {
                        out[offset + a] = next[a];
                    }
                    offset += dim;
                }
                debug_assert_eq!(offset, t.len());
                out
            }
        }
    }

    /// Orthogonal projection of an ambient vector onto `T_t M`.
    pub fn project_to_tangent(
        &self,
        t: ArrayView1<'_, f64>,
        v: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        debug_assert_eq!(t.len(), v.len());
        match self {
            Self::Euclidean | Self::Circle => v.to_owned(),
            Self::Sphere { dim } => {
                debug_assert_eq!(t.len(), *dim);
                let tv = dot_views(t.clone(), v.clone());
                let mut out = v.to_owned();
                for a in 0..*dim {
                    out[a] -= tv * t[a];
                }
                out
            }
            Self::Interval { lo, hi } => {
                let mut out = Array1::<f64>::zeros(1);
                let at_lo = t[0] <= *lo && v[0] < 0.0;
                let at_hi = t[0] >= *hi && v[0] > 0.0;
                out[0] = if at_lo || at_hi { 0.0 } else { v[0] };
                out
            }
            Self::Product(parts) | Self::ProductWithMetric { manifolds: parts, .. } => {
                let mut out = Array1::<f64>::zeros(v.len());
                let mut offset = 0_usize;
                for part in parts {
                    let dim = part.ambient_dim(1);
                    let projected = part.project_to_tangent(
                        t.slice(ndarray::s![offset..offset + dim]),
                        v.slice(ndarray::s![offset..offset + dim]),
                    );
                    for a in 0..dim {
                        out[offset + a] = projected[a];
                    }
                    offset += dim;
                }
                debug_assert_eq!(offset, v.len());
                out
            }
        }
    }

    /// Convert Euclidean Hessian action `eh · xi` to Riemannian Hessian action.
    ///
    /// For the sphere this is the Absil/Mahony/Sepulchre embedded-sphere
    /// conversion: differentiate the projected gradient and project back to
    /// the tangent space. The ambient derivative includes the normal
    /// curvature term `-<grad_R, ξ> t`; the tangent action is equivalent to
    /// `P_t(eh ξ) - <eg, t> ξ`.
    pub fn euclidean_to_riemannian_hessian(
        &self,
        t: ArrayView1<'_, f64>,
        eg: ArrayView1<'_, f64>,
        eh: ArrayView2<'_, f64>,
        xi: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        debug_assert_eq!(t.len(), eg.len());
        debug_assert_eq!(t.len(), xi.len());
        debug_assert_eq!(eh.nrows(), t.len());
        debug_assert_eq!(eh.ncols(), t.len());
        let eh_xi = matvec(eh, xi.clone());
        self.euclidean_hessian_action_to_riemannian(t, eg, xi, eh_xi.view())
    }

    fn euclidean_hessian_action_to_riemannian(
        &self,
        t: ArrayView1<'_, f64>,
        eg: ArrayView1<'_, f64>,
        xi: ArrayView1<'_, f64>,
        eh_xi: ArrayView1<'_, f64>,
    ) -> Array1<f64> {
        debug_assert_eq!(t.len(), eg.len());
        debug_assert_eq!(t.len(), xi.len());
        debug_assert_eq!(t.len(), eh_xi.len());
        match self {
            Self::Euclidean | Self::Circle | Self::Interval { .. } => {
                self.project_to_tangent(t, eh_xi)
            }
            Self::Sphere { dim } => {
                debug_assert_eq!(t.len(), *dim);
                let grad_r = self.project_to_tangent(t.clone(), eg.clone());
                let mut ambient = self.project_to_tangent(t.clone(), eh_xi);
                let eg_normal = dot_views(eg, t.clone());
                let normal_curve = dot_views(grad_r.view(), xi.clone());
                for a in 0..*dim {
                    ambient[a] -= eg_normal * xi[a];
                    ambient[a] -= normal_curve * t[a];
                }
                self.project_to_tangent(t, ambient.view())
            }
            Self::Product(parts) | Self::ProductWithMetric { manifolds: parts, .. } => {
                let mut out = Array1::<f64>::zeros(t.len());
                let mut offset = 0_usize;
                for part in parts {
                    let dim = part.ambient_dim(1);
                    let converted = part.euclidean_hessian_action_to_riemannian(
                        t.slice(ndarray::s![offset..offset + dim]),
                        eg.slice(ndarray::s![offset..offset + dim]),
                        xi.slice(ndarray::s![offset..offset + dim]),
                        eh_xi.slice(ndarray::s![offset..offset + dim]),
                    );
                    for a in 0..dim {
                        out[offset + a] = converted[a];
                    }
                    offset += dim;
                }
                debug_assert_eq!(offset, t.len());
                out
            }
        }
    }

    /// Dense ambient matrix representation of the tangent Hessian action.
    ///
    /// Normal directions are pinned with an identity block for embedded
    /// constrained factors so existing BA Cholesky code can factor the ambient
    /// matrix while RHS/cross blocks stay tangent-projected.
    pub fn riemannian_hessian_matrix(
        &self,
        t: ArrayView1<'_, f64>,
        eg: ArrayView1<'_, f64>,
        eh: ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        let d = t.len();
        let mut out = Array2::<f64>::zeros((d, d));
        let mut xi = Array1::<f64>::zeros(d);
        for a in 0..d {
            xi.fill(0.0);
            xi[a] = 1.0;
            let tangent_xi = self.project_to_tangent(t.clone(), xi.view());
            let col = self.euclidean_to_riemannian_hessian(
                t.clone(),
                eg.clone(),
                eh.clone(),
                tangent_xi.view(),
            );
            for b in 0..d {
                out[[b, a]] = col[b];
            }
        }
        self.add_normal_pinning(t, &mut out);
        symmetrize(&mut out);
        out
    }

    /// Project every column of an ambient matrix into `T_t M`.
    pub fn project_matrix_columns_to_tangent(
        &self,
        t: ArrayView1<'_, f64>,
        matrix: ArrayView2<'_, f64>,
    ) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros(matrix.dim());
        for col_idx in 0..matrix.ncols() {
            let col = self.project_to_tangent(t.clone(), matrix.column(col_idx));
            for row_idx in 0..matrix.nrows() {
                out[[row_idx, col_idx]] = col[row_idx];
            }
        }
        out
    }

    fn add_normal_pinning(&self, t: ArrayView1<'_, f64>, matrix: &mut Array2<f64>) {
        match self {
            Self::Sphere { dim } => {
                debug_assert_eq!(t.len(), *dim);
                for a in 0..*dim {
                    for b in 0..*dim {
                        matrix[[a, b]] += SPHERE_NORMAL_PIN * t[a] * t[b];
                    }
                }
            }
            Self::Product(parts) | Self::ProductWithMetric { manifolds: parts, .. } => {
                let mut offset = 0_usize;
                for part in parts {
                    let dim = part.ambient_dim(1);
                    let mut block = matrix
                        .slice_mut(ndarray::s![offset..offset + dim, offset..offset + dim]);
                    let mut owned = block.to_owned();
                    part.add_normal_pinning(
                        t.slice(ndarray::s![offset..offset + dim]),
                        &mut owned,
                    );
                    block.assign(&owned);
                    offset += dim;
                }
            }
            Self::Euclidean | Self::Circle | Self::Interval { .. } => {}
        }
    }
}

impl LatentIdMode {
    /// Fixes the audit finding that ARD/DimSelection alone is rotation
    /// symmetric and therefore not a standalone identifiability mode.
    pub fn is_identifiable(&self) -> bool {
        matches!(
            self,
            Self::AuxPrior { .. } | Self::AuxPriorDimSelection { .. }
        )
    }

    fn reject_dim_selection_alone(&self) {
        if matches!(self, Self::DimSelection { .. }) {
            // `DimSelection` alone is rotation-symmetric — not a valid
            // gauge fix; callers must pair ARD with `AuxPrior`/`Isometry`.
            // SAFETY: reaching this panic means the builder accepted an
            // unpaired `DimSelection`, violating the identifiability gate.
            panic!(
                "LatentIdMode::DimSelection is not a standalone gauge fix; pair ARD with AuxPrior or Isometry"
            );
        }
    }
}

/// Carrier for the `∂Φ/∂t` chain-rule input, dispatched on basis kind by
/// [`LatentCoordValues::design_gradient_wrt_t_dispatch`].
///
/// * [`InputLocationDerivative::Radial`] is the *radial-kernel* path: the
///   caller supplies the radial kernel family together with the center
///   coordinates, and the chain rule
///   `∂Φ/∂t = q(r) · (t − c)` is applied internally. This covers every
///   isotropic radial basis — Duchon (any nullspace order), Matérn (every
///   supported half-integer ν), and anything else whose pointwise
///   gradient is radial. Helpers:
///   [`crate::terms::basis::duchon_radial_first_derivative_nd`],
///   [`crate::terms::basis::matern_radial_first_derivative_nd`].
/// * [`InputLocationDerivative::Jet`] is the *pre-computed jet* path: the
///   caller has already assembled a closed-form `(N, K, d)` tensor for a
///   basis whose chain rule is not a simple radial scalar times a unit
///   vector. Sphere kernels carry the tangent-direction times `K'(cos γ)`;
///   periodic-cyclic B-splines carry the closed-form cardinal derivative;
///   tensor-product B-splines carry the product-rule mix. Helpers:
///   [`crate::terms::basis::sphere_first_derivative_nd`],
///   [`crate::terms::basis::periodic_bspline_first_derivative_nd`],
///   [`crate::terms::basis::bspline_tensor_first_derivative`].
///
/// The dispatch is an enum rather than a trait because each path's
/// arguments differ structurally (radial bases reuse scalar radial kernels shared with
/// the kernel-shape chain machinery; jet bases ship the full tensor). All chain rules
/// are analytic and closed-form; no autodiff, no finite differences.
pub(crate) enum InputLocationDerivative<'a> {
    /// Radial-kernel chain rule. The chain rule `(t − c)/r` is reconstructed
    /// internally from the finite `q = φ'(r)/r` scalar and the center coordinates.
    Radial {
        centers: ArrayView2<'a, f64>,
        radial_kind: &'a RadialScalarKind,
    },
    /// Pre-computed analytic `(n_obs, n_centers, latent_dim)` jet.
    Jet(ArrayView3<'a, f64>),
}

/// Per-row latent coordinates `t ∈ ℝ^{N × d}` stored as a flat
/// row-major `Array1<f64>` of length `n_obs * latent_dim`.
///
/// The flat-`Array1` layout mirrors [`crate::terms::smooth::SpatialLogKappaCoords`]
/// so the same `HyperDesignDerivative::from_implicit` / `DirectionalHyperParam`
/// outer plumbing can consume it without modification.
#[derive(Debug, Clone)]
pub struct LatentCoordValues {
    /// Stable process-local identity for this latent-coordinate block.
    id: u64,
    /// Flattened (n_obs, latent_dim) latent matrix, row-major
    /// (so `values[n * d + k] = t_n[k]`).
    values: Array1<f64>,
    /// Number of rows `N`.
    n_obs: usize,
    /// Number of latent dimensions `d`.
    latent_dim: usize,
    /// Identifiability / gauge-fix mode.
    id_mode: LatentIdMode,
    /// Manifold used for per-row Riemannian updates.
    manifold: LatentManifold,
}

impl LatentCoordValues {
    /// Construct from a dense `(n_obs, latent_dim)` matrix.
    pub fn from_matrix(matrix: ArrayView2<'_, f64>, id_mode: LatentIdMode) -> Self {
        Self::from_matrix_with_manifold(matrix, id_mode, LatentManifold::Euclidean)
    }

    /// Construct from a dense matrix and explicit latent manifold.
    pub fn from_matrix_with_manifold(
        matrix: ArrayView2<'_, f64>,
        id_mode: LatentIdMode,
        manifold: LatentManifold,
    ) -> Self {
        id_mode.reject_dim_selection_alone();
        let n_obs = matrix.nrows();
        let latent_dim = matrix.ncols();
        let mut values = Array1::<f64>::zeros(n_obs * latent_dim);
        for n in 0..n_obs {
            for k in 0..latent_dim {
                values[n * latent_dim + k] = matrix[[n, k]];
            }
        }
        let mut out = Self {
            id: next_latent_coord_id(),
            values,
            n_obs,
            latent_dim,
            id_mode,
            manifold,
        };
        out.project_all_rows_to_manifold();
        out
    }

    /// Construct directly from a flat (`n_obs * latent_dim`) array.
    pub fn from_flat(
        values: Array1<f64>,
        n_obs: usize,
        latent_dim: usize,
        id_mode: LatentIdMode,
    ) -> Self {
        Self::from_flat_with_manifold(values, n_obs, latent_dim, id_mode, LatentManifold::Euclidean)
    }

    /// Construct directly from a flat array and explicit latent manifold.
    pub fn from_flat_with_manifold(
        values: Array1<f64>,
        n_obs: usize,
        latent_dim: usize,
        id_mode: LatentIdMode,
        manifold: LatentManifold,
    ) -> Self {
        Self::from_flat_with_manifold_and_id(
            values,
            n_obs,
            latent_dim,
            id_mode,
            manifold,
            next_latent_coord_id(),
        )
    }

    pub(crate) fn from_flat_with_manifold_and_id(
        values: Array1<f64>,
        n_obs: usize,
        latent_dim: usize,
        id_mode: LatentIdMode,
        manifold: LatentManifold,
        id: u64,
    ) -> Self {
        id_mode.reject_dim_selection_alone();
        debug_assert_eq!(
            values.len(),
            n_obs * latent_dim,
            "LatentCoordValues::from_flat: length {} != n_obs * latent_dim = {}",
            values.len(),
            n_obs * latent_dim
        );
        let mut out = Self {
            id,
            values,
            n_obs,
            latent_dim,
            id_mode,
            manifold,
        };
        out.project_all_rows_to_manifold();
        out
    }

    pub fn latent_id(&self) -> u64 {
        self.id
    }

    pub fn n_obs(&self) -> usize {
        self.n_obs
    }

    pub fn latent_dim(&self) -> usize {
        self.latent_dim
    }

    /// Total length of the flat value array (= `n_obs * latent_dim`).
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn id_mode(&self) -> &LatentIdMode {
        &self.id_mode
    }

    pub fn manifold(&self) -> &LatentManifold {
        &self.manifold
    }

    pub fn with_manifold(&self, manifold: LatentManifold) -> Self {
        Self::from_flat_with_manifold_and_id(
            self.values.clone(),
            self.n_obs,
            self.latent_dim,
            self.id_mode.clone(),
            manifold,
            self.id,
        )
    }

    /// View the flat value array.
    pub fn as_flat(&self) -> &Array1<f64> {
        &self.values
    }

    /// View row `n` as a length-`d` slice.
    pub fn row(&self, n: usize) -> &[f64] {
        let start = n * self.latent_dim;
        let end = start + self.latent_dim;
        &self.values.as_slice().expect("contiguous")[start..end]
    }

    /// Materialize as a dense `(n_obs, latent_dim)` matrix view.
    /// Useful when handing `t` to a row-major basis evaluator
    /// (e.g. `build_duchon_basis`).
    pub fn as_matrix(&self) -> Array2<f64> {
        let mut out = Array2::<f64>::zeros((self.n_obs, self.latent_dim));
        for n in 0..self.n_obs {
            for k in 0..self.latent_dim {
                out[[n, k]] = self.values[n * self.latent_dim + k];
            }
        }
        out
    }

    /// Mutable write back of the flat value array, e.g. after a Newton step.
    pub fn set_flat(&mut self, flat: ArrayView1<'_, f64>) {
        debug_assert_eq!(flat.len(), self.values.len());
        self.values.assign(&flat);
        self.project_all_rows_to_manifold();
    }

    /// Apply a flat tangent update row-by-row through the manifold retraction.
    pub fn retract_flat_delta(&mut self, delta: ArrayView1<'_, f64>) {
        debug_assert_eq!(delta.len(), self.values.len());
        if self.manifold.is_euclidean() {
            for (t, dt) in self.values.iter_mut().zip(delta.iter()) {
                *t += *dt;
            }
            return;
        }
        for n in 0..self.n_obs {
            let start = n * self.latent_dim;
            let end = start + self.latent_dim;
            let current = self.values.slice(ndarray::s![start..end]);
            let xi = delta.slice(ndarray::s![start..end]);
            let next = self.manifold.retract(current, xi);
            for a in 0..self.latent_dim {
                self.values[start + a] = next[a];
            }
        }
    }

    fn project_all_rows_to_manifold(&mut self) {
        if self.manifold.is_euclidean() {
            return;
        }
        debug_assert_eq!(self.manifold.ambient_dim(self.latent_dim), self.latent_dim);
        for n in 0..self.n_obs {
            let start = n * self.latent_dim;
            let end = start + self.latent_dim;
            let projected = self.manifold.project_point(self.values.slice(ndarray::s![start..end]));
            for a in 0..self.latent_dim {
                self.values[start + a] = projected[a];
            }
        }
    }

    /// Apply this latent block back to a `TermCollectionSpec`-style covariate
    /// table: returns the `(N, d)` materialized matrix that downstream basis
    /// evaluators (Duchon, Matérn, ...) take as their feature input.
    ///
    /// This mirrors [`crate::terms::smooth::SpatialLogKappaCoords::apply_tospec`],
    /// but the carrier on the spec side is the data-row covariate block rather
    /// than the per-term `length_scale`. The spec-mutation is handled at the
    /// call site (the consuming term needs to know which columns of its
    /// feature view to overwrite).
    pub fn apply_tospec(&self) -> Array2<f64> {
        self.as_matrix()
    }

    /// Compute `∂Φ/∂t` for a radial-kernel design Φ — the original
    /// Duchon/Matérn path. See [`Self::design_gradient_wrt_t_dispatch`] for
    /// the basis-agnostic dispatch entry point.
    ///
    /// `centers` is `(n_centers, d)`.
    /// Returns a `(n_obs, n_centers, d)` jet whose `(n, k, a)` entry is
    /// `∂Φ_{n,k} / ∂t_{n,a} = q(r_{n,k}) · (t_{n,a} − c_{k,a})`.
    ///
    /// At `r = 0` the unit vector `(t − c)/r` is undefined; the radial scalar
    /// path therefore asks the kernel for the finite `q` limit and surfaces
    /// `BasisError::DegenerateAtCollision` when that limit does not exist.
    pub(crate) fn design_gradient_wrt_t(
        &self,
        centers: ArrayView2<'_, f64>,
        radial_kind: &RadialScalarKind,
    ) -> Result<Array3<f64>, BasisError> {
        let n_obs = self.n_obs;
        let d = self.latent_dim;
        let n_centers = centers.nrows();
        if centers.ncols() != d {
            return Err(BasisError::DimensionMismatch(format!(
                "LatentCoordValues::design_gradient_wrt_t center dimension mismatch: centers have {} cols but latent_dim is {}",
                centers.ncols(),
                d
            )));
        }
        let mut jet = Array3::<f64>::zeros((n_obs, n_centers, d));
        for n in 0..n_obs {
            let t_n = self.row(n);
            for k in 0..n_centers {
                let mut r2 = 0.0_f64;
                for a in 0..d {
                    let delta = t_n[a] - centers[[k, a]];
                    r2 += delta * delta;
                }
                let r = r2.sqrt();
                let (_, q, _) = radial_kind.eval_design_triplet(r)?;
                if q == 0.0 {
                    continue;
                }
                for a in 0..d {
                    jet[[n, k, a]] = q * (t_n[a] - centers[[k, a]]);
                }
            }
        }
        Ok(jet)
    }

    /// Compute `∂Φ/∂t` for an arbitrary supported basis kind, by dispatching
    /// to the right closed-form chain rule.
    ///
    /// All radial-kernel bases (Duchon, Matérn) reduce to the same
    /// `q(r) · (t − c)` chain that `design_gradient_wrt_t` already implements.
    /// Non-radial bases (sphere, periodic-cyclic B-spline, tensor
    /// B-spline) carry their own analytic `(N, K, d)` jet — the caller
    /// pre-builds that jet using the matching `*_first_derivative_nd` helper
    /// in [`crate::terms::basis`] and passes it in via
    /// [`InputLocationDerivative::Jet`].
    ///
    /// This is the single entry point the outer optimizer should call; it
    /// stays in lock-step with the kernel-parameter chain rule that
    /// `SpatialLogKappaCoords` uses (re-pointed at the first kernel argument
    /// rather than at kernel anisotropy).
    pub(crate) fn design_gradient_wrt_t_dispatch(
        &self,
        input: InputLocationDerivative<'_>,
    ) -> Result<Array3<f64>, BasisError> {
        match input {
            InputLocationDerivative::Radial {
                centers,
                radial_kind,
            } => self.design_gradient_wrt_t(centers, radial_kind),
            InputLocationDerivative::Jet(jet) => {
                if jet.shape() != &[self.n_obs, jet.shape()[1], self.latent_dim] {
                    return Err(BasisError::DimensionMismatch(format!(
                        "LatentCoordValues::design_gradient_wrt_t_dispatch jet shape {:?} does not match latent shape ({}, {}, {})",
                        jet.shape(),
                        self.n_obs,
                        jet.shape()[1],
                        self.latent_dim
                    )));
                }
                // The non-radial helpers already produce a (N, K, d) tensor
                // in the same layout `contract_gradient` consumes. Return a
                // copy so the caller owns the data and is decoupled from the
                // source array's lifetime.
                Ok(jet.to_owned())
            }
        }
    }

    /// Contract a downstream gradient `∂L/∂Φ ∈ ℝ^(n_obs × n_centers)` and a
    /// `design_gradient_wrt_t` jet into a flat `∂L/∂t ∈ ℝ^(n_obs * d)`.
    ///
    /// This is the N-D generalization of
    /// `gam_pyffi::contract_position_gradient` (1-D), used inside the
    /// `_backward` pyffi entry point.
    pub(crate) fn contract_gradient(
        grad_phi: ArrayView2<'_, f64>,
        jet: &Array3<f64>,
    ) -> Array1<f64> {
        let n_obs = jet.shape()[0];
        let n_centers = jet.shape()[1];
        let d = jet.shape()[2];
        assert_eq!(grad_phi.shape(), &[n_obs, n_centers]);
        let mut grad_t = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            for a in 0..d {
                let mut acc = 0.0_f64;
                for k in 0..n_centers {
                    acc += grad_phi[[n, k]] * jet[[n, k, a]];
                }
                grad_t[n * d + a] = acc;
            }
        }
        grad_t
    }

    /// Streaming contraction for radial-kernel input-location derivatives.
    ///
    /// This computes the same result as
    /// `contract_gradient(grad_phi, design_gradient_wrt_t(...))`, but never
    /// materializes the dense `(n_obs, n_centers, latent_dim)` jet. Peak
    /// storage is therefore `O(n_obs * latent_dim)`.
    pub(crate) fn contract_gradient_radial_streaming(
        &self,
        grad_phi: ArrayView2<'_, f64>,
        centers: ArrayView2<'_, f64>,
        radial_kind: &RadialScalarKind,
    ) -> Result<Array1<f64>, BasisError> {
        let n_obs = self.n_obs;
        let d = self.latent_dim;
        let n_centers = centers.nrows();
        assert_eq!(centers.ncols(), d);
        assert_eq!(grad_phi.shape(), &[n_obs, n_centers]);
        let mut grad_t = Array1::<f64>::zeros(n_obs * d);
        for n in 0..n_obs {
            self.contract_row_radial_gradient_into(
                n,
                grad_phi.row(n),
                centers,
                radial_kind,
                &mut grad_t,
                1.0,
            )?;
        }
        Ok(grad_t)
    }

    /// Streaming row contraction for one radial-kernel latent row.
    ///
    /// `grad_phi_row[k]` is the downstream adjoint for `Φ[n,k]`. The
    /// contribution is accumulated into `grad_t[n, :]` with an optional scalar
    /// multiplier.
    pub(crate) fn contract_row_radial_gradient_into(
        &self,
        n: usize,
        grad_phi_row: ArrayView1<'_, f64>,
        centers: ArrayView2<'_, f64>,
        radial_kind: &RadialScalarKind,
        grad_t: &mut Array1<f64>,
        scale: f64,
    ) -> Result<(), BasisError> {
        let d = self.latent_dim;
        let n_centers = centers.nrows();
        assert!(n < self.n_obs);
        assert_eq!(centers.ncols(), d);
        assert_eq!(grad_phi_row.len(), n_centers);
        assert_eq!(grad_t.len(), self.values.len());
        if scale == 0.0 {
            return Ok(());
        }
        let t_n = self.row(n);
        for k in 0..n_centers {
            let adjoint = grad_phi_row[k];
            if adjoint == 0.0 {
                continue;
            }
            let mut r2 = 0.0_f64;
            for a in 0..d {
                let delta = t_n[a] - centers[[k, a]];
                r2 += delta * delta;
            }
            let r = r2.sqrt();
            let (_, q, _) = radial_kind.eval_design_triplet(r)?;
            if q == 0.0 {
                continue;
            }
            let row_scale = scale * adjoint * q;
            for a in 0..d {
                grad_t[n * d + a] += row_scale * (t_n[a] - centers[[k, a]]);
            }
        }
        Ok(())
    }
}

fn wrap_angle(x: f64) -> f64 {
    let y = x.rem_euclid(TWO_PI);
    if y == TWO_PI { 0.0 } else { y }
}

fn normalize_or_axis(v: ArrayView1<'_, f64>, dim: usize) -> Array1<f64> {
    let mut norm_sq = 0.0_f64;
    for a in 0..dim {
        norm_sq += v[a] * v[a];
    }
    if norm_sq <= 0.0 || !norm_sq.is_finite() {
        // `LatentManifold::Sphere` requires unit-projectable ambient
        // vectors; the term builder validates this upstream.
        // SAFETY: a zero/non-finite norm means the upstream contract
        // was broken at the caller boundary.
        panic!(
            "LatentManifold::Sphere cannot normalize a zero or non-finite ambient vector"
        );
    }
    let inv = 1.0 / norm_sq.sqrt();
    let mut out = Array1::<f64>::zeros(dim);
    for a in 0..dim {
        out[a] = v[a] * inv;
    }
    out
}

fn dot_views(a: ArrayView1<'_, f64>, b: ArrayView1<'_, f64>) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = 0.0_f64;
    for i in 0..a.len() {
        acc += a[i] * b[i];
    }
    acc
}

fn matvec(a: ArrayView2<'_, f64>, x: ArrayView1<'_, f64>) -> Array1<f64> {
    debug_assert_eq!(a.ncols(), x.len());
    let mut out = Array1::<f64>::zeros(a.nrows());
    for i in 0..a.nrows() {
        let mut acc = 0.0_f64;
        for j in 0..a.ncols() {
            acc += a[[i, j]] * x[j];
        }
        out[i] = acc;
    }
    out
}

fn symmetrize(a: &mut Array2<f64>) {
    let n = a.nrows().min(a.ncols());
    for i in 0..n {
        for j in 0..i {
            let v = 0.5 * (a[[i, j]] + a[[j, i]]);
            a[[i, j]] = v;
            a[[j, i]] = v;
        }
    }
}

/// Auxiliary-prior penalty contribution: returns the per-row reference
/// coordinates `ĥ(u_n)` shape `(n_obs, d)` and the effective strength `μ`.
///
/// `t_target` is broadcast across the inner ridge of `½ μ · ‖t − t_target‖²`,
/// which the call site folds into the Y-stack via a virtual-row augmentation
/// (`y' = [y; √μ · t_target]`, `X' = [X; √μ · I_d ⊗ row-block]`). This
/// keeps the inner solver Gaussian-closed-form.
///
/// For `AuxPriorFamily::Ridge` the conditional mean is the closed-form ridge
/// regression `(UᵀU + ε I)⁻¹ UᵀT` evaluated at each row's `u_n`. For
/// `Linear` the ridge is zero (which raises if `UᵀU` is singular).
pub fn aux_prior_targets(
    t: ArrayView2<'_, f64>,
    u: ArrayView2<'_, f64>,
    family: AuxPriorFamily,
) -> Result<Array2<f64>, String> {
    let n_obs = t.nrows();
    let d = t.ncols();
    if u.nrows() != n_obs {
        return Err(format!(
            "aux_prior_targets: u has {} rows but t has {}",
            u.nrows(),
            n_obs
        ));
    }
    let p = u.ncols();
    if p == 0 {
        return Err("aux_prior_targets: auxiliary u must have at least one column".into());
    }
    // gram = UᵀU  (p × p)
    let mut gram = Array2::<f64>::zeros((p, p));
    for n in 0..n_obs {
        for i in 0..p {
            for j in 0..p {
                gram[[i, j]] += u[[n, i]] * u[[n, j]];
            }
        }
    }
    let ridge_eps = match family {
        AuxPriorFamily::Ridge => {
            let trace: f64 = (0..p).map(|i| gram[[i, i]]).sum();
            (1e-6 * trace / p as f64).max(1e-12)
        }
        AuxPriorFamily::Linear => 0.0,
    };
    for i in 0..p {
        gram[[i, i]] += ridge_eps;
    }
    // rhs = UᵀT  (p × d)
    let mut rhs = Array2::<f64>::zeros((p, d));
    for n in 0..n_obs {
        for i in 0..p {
            for k in 0..d {
                rhs[[i, k]] += u[[n, i]] * t[[n, k]];
            }
        }
    }
    let coeffs = solve_spd(gram.view(), rhs.view())?;
    // targets = U · coeffs  (n_obs × d)
    let mut targets = Array2::<f64>::zeros((n_obs, d));
    for n in 0..n_obs {
        for k in 0..d {
            let mut acc = 0.0_f64;
            for i in 0..p {
                acc += u[[n, i]] * coeffs[[i, k]];
            }
            targets[[n, k]] = acc;
        }
    }
    Ok(targets)
}

/// Lightweight Cholesky-based SPD solve. Keeps this module dependency-free
/// from the broader faer-wrapping surface; matrices here are tiny
/// (`p × p` with p = aux-feature count, typically O(10)).
fn solve_spd(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err("solve_spd: A must be square".into());
    }
    if b.nrows() != n {
        return Err("solve_spd: RHS row count mismatch".into());
    }
    // In-place Cholesky factorization. We pay the O(n³) copy + O(n³) factor
    // up front; n is tiny in the auxiliary-prior path.
    let mut l = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[[i, j]];
            for k in 0..j {
                sum -= l[[i, k]] * l[[j, k]];
            }
            if i == j {
                if sum <= 0.0 {
                    return Err(format!(
                        "solve_spd: non-positive pivot {sum} at index {i} \
                         (matrix is not positive definite)"
                    ));
                }
                l[[i, j]] = sum.sqrt();
            } else {
                l[[i, j]] = sum / l[[j, j]];
            }
        }
    }
    // Solve L y = b, then Lᵀ x = y, column by column.
    let d = b.ncols();
    let mut out = Array2::<f64>::zeros((n, d));
    for col in 0..d {
        let mut y = Array1::<f64>::zeros(n);
        for i in 0..n {
            let mut sum = b[[i, col]];
            for k in 0..i {
                sum -= l[[i, k]] * y[k];
            }
            y[i] = sum / l[[i, i]];
        }
        for i in (0..n).rev() {
            let mut sum = y[i];
            for k in (i + 1)..n {
                sum -= l[[k, i]] * out[[k, col]];
            }
            out[[i, col]] = sum / l[[i, i]];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn from_matrix_roundtrip() {
        let m = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let lc = LatentCoordValues::from_matrix(m.view(), LatentIdMode::None);
        assert_eq!(lc.n_obs(), 3);
        assert_eq!(lc.latent_dim(), 2);
        let back = lc.as_matrix();
        assert_eq!(back, m);
    }

    #[test]
    fn row_access() {
        let m = array![[1.0_f64, 2.0], [3.0, 4.0]];
        let lc = LatentCoordValues::from_matrix(m.view(), LatentIdMode::None);
        assert_eq!(lc.row(0), &[1.0, 2.0]);
        assert_eq!(lc.row(1), &[3.0, 4.0]);
    }
}
