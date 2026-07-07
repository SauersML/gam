use super::*;

/// Wrapper to send a raw pointer across thread boundaries for parallel buffer fills.
/// SAFETY: every `SendPtr` value must be built from live, properly aligned `f64`
/// storage whose mutable borrow is held until all worker threads finish; callers
/// may only dereference offsets that are in-bounds and disjoint across workers.
#[derive(Clone, Copy)]
pub(crate) struct SendPtr(pub(crate) *mut f64);

// SAFETY: SendPtr only grants raw-pointer transport. Actual dereferences occur
// at call sites after row-chunk partitioning proves each thread writes a
// distinct in-bounds element of the backing Array/Vec allocation.
unsafe impl Send for SendPtr {}

// SAFETY: shared references to SendPtr are sound because the pointee is never
// accessed through the wrapper without the call-site disjoint-offset proof.
unsafe impl Sync for SendPtr {}

impl SendPtr {
    #[inline(always)]
    pub(crate) fn add(self, offset: usize) -> *mut f64 {
        // SAFETY: callers pass offsets within the backing allocation and only
        // dereference the returned pointer after proving the target element is
        // uniquely owned by that worker's chunk for the whole parallel region.
        unsafe { self.0.add(offset) }
    }
}

/// Re-export of the neutral basis-error contract. #1521: `BasisError` lives
/// in `gam-problem` so `EstimationError` can wrap it (`#[from]`) without a
/// back-edge; gam-terms re-exports it to preserve `gam_terms::basis::BasisError`.
pub use gam_problem::BasisError;

// ============================================================================
// Unified Basis Generation API
// ============================================================================

/// Options for basis generation, controlling derivative order.
#[derive(Clone, Copy, Debug, Default)]
pub struct BasisOptions {
    /// Derivative order: 0 = value (default), 1 = first derivative, 2 = second derivative
    pub derivative_order: usize,
    /// Basis family to evaluate.
    pub basis_family: BasisFamily,
}

impl BasisOptions {
    /// Create options for evaluating basis functions (no derivative).
    pub const fn value() -> Self {
        Self {
            derivative_order: 0,
            basis_family: BasisFamily::BSpline,
        }
    }

    /// Create options for evaluating first derivatives of basis functions.
    pub const fn first_derivative() -> Self {
        Self {
            derivative_order: 1,
            basis_family: BasisFamily::BSpline,
        }
    }

    /// Create options for evaluating second derivatives of basis functions.
    pub const fn second_derivative() -> Self {
        Self {
            derivative_order: 2,
            basis_family: BasisFamily::BSpline,
        }
    }

    /// Create options for evaluating M-spline basis values.
    pub const fn m_spline() -> Self {
        Self {
            derivative_order: 0,
            basis_family: BasisFamily::MSpline,
        }
    }

    /// Create options for evaluating I-spline basis values.
    pub const fn i_spline() -> Self {
        Self {
            derivative_order: 0,
            basis_family: BasisFamily::ISpline,
        }
    }
}

/// Basis-family selector for 1D spline evaluation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum BasisFamily {
    /// Standard B-splines.
    #[default]
    BSpline,
    /// M-splines: normalized B-splines, M_i = ((k+1)/(t_{i+k+1}-t_i)) B_i.
    MSpline,
    /// I-splines: integrated M-splines, implemented by right-cumulative
    /// sums of B-splines at degree k+1.
    ISpline,
}

/// Specifies the source of knots for basis generation.
#[derive(Clone, Debug)]
pub enum KnotSource<'a> {
    /// Use a pre-computed knot vector.
    Provided(ArrayView1<'a, f64>),
    /// Generate uniformly spaced knots based on data range.
    Generate {
        /// Data range (min, max) for knot placement.
        data_range: (f64, f64),
        /// Number of internal knots to place between boundaries.
        num_internal_knots: usize,
    },
}
/// Thin-plate regression spline basis and penalty (order m=2).
///
/// The returned basis has columns `[K_c | P]` where:
/// - `K_c` is the constrained radial basis block (`K * Z`) with
///   `P(knots)^T * α = 0` enforced via nullspace projection
/// - `P` is the TPS polynomial null-space block containing all monomials of
///   total degree `< m`, where `m = thin_plate_penalty_order(d)` (so `P` is
///   just `[1, x_1, ..., x_d]` for `d <= 3`)
///
/// The returned penalty matrix is block-diagonal with:
/// - upper-left `Omega_c = Z^T Omega Z` for the constrained radial block
/// - zero lower-right block for unpenalized polynomial terms.
///
/// For double-penalty GAMs, a second ridge penalty `I` is also returned so the
/// caller can optimize `(lambda_bending, lambdaridge)` jointly.
#[derive(Debug, Clone)]
pub struct ThinPlateSplineBasis {
    pub basis: Array2<f64>,
    pub penalty_bending: Array2<f64>,
    pub penalty_ridge: Array2<f64>,
    pub num_kernel_basis: usize,
    pub num_polynomial_basis: usize,
    pub dimension: usize,
    /// Wood-TPRS radial reparameterization matrix `V`.
    ///
    /// Rows live in the side-constrained radial coefficient space. Columns are
    /// the retained positive bending eigendirections of `Z' Ω Z`; numerically
    /// near-null radial directions are dropped before the basis is exposed.
    /// Therefore `V` can be rectangular: design columns are `Φ Z V`, and the
    /// radial penalty is `diag(Λ_retained)`.
    pub radial_reparam: Array2<f64>,
}

/// Matérn smoothness parameter `nu` (half-integer variants with closed forms).
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MaternNu {
    Half,
    ThreeHalves,
    FiveHalves,
    SevenHalves,
    NineHalves,
}

impl MaternNu {
    /// The half-integer smoothness value ν as an `f64` (0.5, 1.5, …).
    pub const fn half_integer_value(self) -> f64 {
        match self {
            MaternNu::Half => 0.5,
            MaternNu::ThreeHalves => 1.5,
            MaternNu::FiveHalves => 2.5,
            MaternNu::SevenHalves => 3.5,
            MaternNu::NineHalves => 4.5,
        }
    }
}

/// Matérn radial basis and penalties.
#[derive(Debug, Clone)]
pub struct MaternSplineBasis {
    pub basis: Array2<f64>,
    pub penalty_kernel: Array2<f64>,
    pub penalty_ridge: Array2<f64>,
    pub num_kernel_basis: usize,
    pub num_polynomial_basis: usize,
    pub dimension: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct DuchonBasisDesign {
    pub(crate) basis: Array2<f64>,
}

/// Boundary-condition policy for one-dimensional smooth bases.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum OneDimensionalBoundary {
    /// Ordinary open interval basis with clamped endpoint behavior.
    #[default]
    Open,
    /// Periodic/cyclic basis over the half-open interval `[start, end)`.
    ///
    /// Values are evaluated modulo `period = end - start`; the basis and its
    /// first `degree - 1` derivatives agree at the two endpoints for B-splines.
    Cyclic { start: f64, end: f64 },
}

impl OneDimensionalBoundary {
    pub(crate) fn period(&self) -> Option<(f64, f64, f64)> {
        match *self {
            OneDimensionalBoundary::Open => None,
            OneDimensionalBoundary::Cyclic { start, end } if end > start => {
                Some((start, end, end - start))
            }
            OneDimensionalBoundary::Cyclic { .. } => None,
        }
    }
}

/// Which knot strategy to use for 1D B-spline bases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BSplineKnotSpec {
    Generate {
        data_range: (f64, f64),
        num_internal_knots: usize,
    },
    /// Uniform cyclic B-spline basis on `[data_range.0, data_range.1)`.
    ///
    /// The first and last endpoints are identified, so evaluating at `x` and
    /// `x + m * period` gives identical rows. `num_basis` is the number of
    /// periodic control sites around the loop and must be at least
    /// `degree + 1` for an unaliased local support stencil.
    PeriodicUniform {
        data_range: (f64, f64),
        num_basis: usize,
    },
    Automatic {
        num_internal_knots: Option<usize>,
        placement: BSplineKnotPlacement,
    },
    Provided(Array1<f64>),
    /// Natural cubic regression spline (`bs="cr"`/`"cs"`) knot set (#1074).
    ///
    /// Unlike the open-spline variants above, these `knots` are the `k`
    /// Lancaster–Salkauskas knots `x*_1 < … < x*_k` that *directly* index the
    /// basis values `β_i = f(x*_i)` — the basis dimension equals `knots.len()`
    /// (not `knots.len() - degree - 1`). The 1-D builder routes this variant to
    /// the cubic-regression builder; the cr identity therefore round-trips
    /// through freeze/reload by virtue of the variant itself (no separate
    /// metadata marker is required), and tensor margins inherit cr by carrying
    /// this knotspec into `build_bspline_basis_1d`.
    NaturalCubicRegression {
        knots: Array1<f64>,
    },
}

/// Internal-knot placement strategy when knots are automatically inferred.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BSplineKnotPlacement {
    Uniform,
    Quantile,
}

/// 1D B-spline basis configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BSplineBasisSpec {
    pub degree: usize,
    pub penalty_order: usize,
    pub knotspec: BSplineKnotSpec,
    pub double_penalty: bool,
    pub identifiability: BSplineIdentifiability,
    #[serde(default)]
    pub boundary: OneDimensionalBoundary,
    /// Optional endpoint boundary constraints (Hermite-style pin of value and/or
    /// derivative at the left/right knot extents). Default = `Free` on both
    /// sides which is a no-op.
    #[serde(default)]
    pub boundary_conditions: BSplineBoundaryConditions,
}

/// Per-endpoint boundary constraint policy for B-spline 1D bases.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub enum BSplineEndpointBoundaryCondition {
    /// No endpoint constraint.
    #[default]
    Free,
    /// Pin the first derivative to zero at this endpoint.
    Clamped,
    /// Pin the value at this endpoint to `value` (currently only `value == 0`
    /// is accepted in the builder; non-zero anchors require an affine offset).
    Anchored { value: f64 },
}

/// Left/right pair of B-spline endpoint constraints.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct BSplineBoundaryConditions {
    #[serde(default)]
    pub left: BSplineEndpointBoundaryCondition,
    #[serde(default)]
    pub right: BSplineEndpointBoundaryCondition,
}

impl BSplineBoundaryConditions {
    pub const fn is_free(&self) -> bool {
        matches!(self.left, BSplineEndpointBoundaryCondition::Free)
            && matches!(self.right, BSplineEndpointBoundaryCondition::Free)
    }
}

/// Per-smooth identifiability policy for 1D B-spline bases.
///
/// These constraints are applied directly in the builder via a reparameterization
/// `B_constrained = B * Z`, and every penalty matrix is projected as
/// `S_constrained = Z' S Z`, so solver geometry stays consistent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BSplineIdentifiability {
    /// Keep unconstrained basis columns.
    None,
    /// Enforce weighted sum-to-zero: `B' w = 0` (or unweighted when `weights=None`).
    // Smooth terms are centered by default to avoid intercept confounding.
    WeightedSumToZero { weights: Option<Array1<f64>> },
    /// Remove intercept + linear trend in coefficient space using Greville geometry.
    RemoveLinearTrend,
    /// Enforce orthogonality to supplied design columns `C` (n x q):
    /// `B_c' W C = 0` (or unweighted when `weights=None`).
    ///
    /// To enforce `[intercept, x, ...]`, provide `columns` with those columns.
    OrthogonalToDesignColumns {
        columns: Array2<f64>,
        weights: Option<Array1<f64>>,
    },
    /// Apply an explicit coefficient-space transform `Z` learned at fit time.
    ///
    /// This freezes identifiability behavior so prediction cannot drift based on
    /// new-data distribution. The constrained basis is `B * Z`.
    FrozenTransform { transform: Array2<f64> },
}

impl Default for BSplineIdentifiability {
    fn default() -> Self {
        BSplineIdentifiability::WeightedSumToZero { weights: None }
    }
}

/// Spatial center selection strategy.
///
/// `num_centers` is the exact number of knot/center rows selected by the
/// strategy. Polynomial nullspace columns are added separately by each basis
/// builder and must never be folded into this count.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CenterStrategy {
    Auto(Box<CenterStrategy>),
    UserProvided(Array2<f64>),
    /// Joint multidimensional equal-mass partitioning in the full smooth space.
    EqualMass {
        num_centers: usize,
    },
    /// Covariate-representative equal-mass partitioning along one selected axis.
    EqualMassCovarRepresentative {
        num_centers: usize,
    },
    FarthestPoint {
        num_centers: usize,
    },
    KMeans {
        num_centers: usize,
        max_iter: usize,
    },
    UniformGrid {
        points_per_dim: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum CenterStrategyKind {
    UserProvided,
    EqualMass,
    EqualMassCovarRepresentative,
    FarthestPoint,
    KMeans,
    UniformGrid,
}

/// Adaptive default center count for spatial smooths (TPS, Duchon, Matérn).
///
/// Use this when the user has not explicitly specified a knot/center count.
/// The basis size is the sub-linear `ceil(8 * d_factor * n^0.4)`, clamped above
/// at `K_MAX = 2000` and below at a *data-proportional* floor `min(200, n/8)` so
/// the floor only engages once there are enough observations to support a rich
/// basis. The result is additionally capped at `n/4` so the penalty matrices
/// stay well-conditioned relative to the data:
///
/// | n      | d=1  | d=2  | d=5  |
/// |--------|------|------|------|
/// | 800    | 116  | 134  | 186  |
/// | 1 000  | 127  | 146  | 200  |
/// | 2 000  | 200  | 200  | 268  |
/// | 10 000 | 319  | 367  | 510  |
/// | 100 000| 801  | 921  | 1281 |
/// | 400 000| 1393 | 1602 | 2000 |
/// | 1 000 000| 2000 | 2000 | 2000 |
///
/// The flat `200` floor used to inflate moderate-`n` spatial smooths (a few
/// hundred to ~2000 rows) up to a dense 200-column design even though the raw
/// sub-linear count — and the mesh/knot density that mgcv and R-INLA use on the
/// same data — is far smaller. On ~800 rows that turned a single 2-D thin-plate
/// REML fit into an `O(n·p² + p³)` grind at `p ≈ 200` (#718). Smoothness is
/// already controlled by REML's penalty weight λ, not by the center count, so a
/// data-proportional floor recovers the same surface at a fraction of the cost.
///
/// # Arguments
/// * `n` - sample size (number of observations)
/// * `d` - covariate dimensionality (number of input variables in the smooth)
pub fn default_num_centers(n: usize, d: usize) -> usize {
    const K_MIN: usize = 200;
    const K_MAX: usize = 2000;
    const ALPHA: f64 = 0.4;
    const C: f64 = 8.0;
    /// Per-extra-dimension growth in the center count: each covariate axis
    /// beyond the first widens the basis by 15% to keep the per-axis mesh
    /// density roughly constant as the smooth's domain dimensionality grows.
    const PER_DIM_GROWTH: f64 = 0.15;
    /// Divisor for the data-proportional floor: the `K_MIN` floor only engages
    /// once `n` exceeds `K_MIN * FLOOR_N_DIVISOR`, so small samples are not
    /// forced up to a dense `K_MIN`-column design.
    const FLOOR_N_DIVISOR: usize = 8;
    /// Divisor for the conditioning cap: the center count never exceeds `n /
    /// COND_N_DIVISOR`, keeping the penalty matrices well-conditioned relative
    /// to the data.
    const COND_N_DIVISOR: usize = 4;

    let d_factor = 1.0 + PER_DIM_GROWTH * (d.max(1) - 1) as f64;
    let raw = (C * d_factor * (n as f64).powf(ALPHA)).ceil() as usize;

    // Data-proportional floor: never inflate beyond n/FLOOR_N_DIVISOR, so the
    // K_MIN-center floor only takes effect once n is large enough (~1600) to
    // genuinely support that many basis columns.
    let floor = K_MIN.min(n / FLOOR_N_DIVISOR);
    let k = raw.clamp(floor, K_MAX);

    // Never exceed n itself; cap at n/COND_N_DIVISOR to keep the penalty
    // matrices well-conditioned relative to the data.
    k.min(n).min(n / COND_N_DIVISOR)
}

/// Conservative center count for a *secondary* (distributional) predictor's
/// spatial smooth — e.g. the log-σ scale model in a Gaussian location-scale
/// fit.
///
/// The mean is identified directly by the response, so it warrants the
/// generous [`default_num_centers`] basis. A scale/shape predictor is
/// identified only through (noisy) squared residuals: handing it a basis sized
/// for the mean lets REML/LAML smoothing selection over-fit it, because where
/// the fitted scale is driven small the *observed* information collapses and
/// the determinant penalty stops holding the wiggle down (#501). This mirrors
/// standard GAMLSS/mgcv practice of giving distribution parameters a modest
/// default (mgcv's modest default basis for a 1-D `s()`), grown gently with
/// dimensionality and never exceeding the generous primary-predictor default.
pub fn conservative_secondary_centers(n: usize, d: usize) -> usize {
    const BASE_1D_CENTERS: usize = 15;
    let modest = BASE_1D_CENTERS.saturating_mul(d.max(1));
    default_num_centers(n, d).min(modest).max(1)
}

/// Resource-aware plan for a spatial smooth (Duchon / Matérn / TPS).
///
/// Returned by [`plan_spatial_basis`]. Captures the resolved center count,
/// final basis dimension `p`, the dense byte cost for the value matrix and
/// each derivative tier, and a recommended storage mode that is consistent
/// with the supplied [`gam_runtime::resource::ResourcePolicy`].
#[derive(Clone, Debug)]
pub struct SpatialBasisPlan {
    pub n: usize,
    pub d: usize,
    pub centers: usize,
    pub p_final_estimate: usize,
    pub dense_design_bytes: usize,
    pub first_derivative_dense_bytes: usize,
    pub second_derivative_dense_bytes: usize,
    pub recommended_storage: SpatialStorageMode,
}

/// Storage mode recommended by [`plan_spatial_basis`].
///
/// * `DenseValueDenseDerivatives` — both the value design and its derivative
///   matrices fit under the policy's single-materialization budget.
/// * `LazyValueImplicitDerivatives` — the value design fits dense but the
///   derivative matrices do not; switch derivatives to the implicit operator.
/// * `OperatorOnly` — neither the design nor its derivatives fit; everything
///   must be operator-backed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpatialStorageMode {
    DenseValueDenseDerivatives,
    LazyValueImplicitDerivatives,
    OperatorOnly,
}

/// How [`plan_spatial_basis`] should pick the spatial center count.
#[derive(Clone, Copy, Debug)]
pub enum CenterCountRequest {
    /// Use the heuristic [`default_num_centers`].
    Default,
    /// Use the caller-supplied count exactly.
    Explicit(usize),
    /// Use [`default_num_centers`] but cap at `cap` to bound dense cost.
    HeuristicCapped { cap: usize },
}

/// Build a resource-aware plan for a spatial smooth basis.
///
/// Computes the resolved center count, final basis dimension, dense byte
/// estimates for the value design and first/second derivative tiers, and a
/// recommended [`SpatialStorageMode`] derived from `policy`. This is the
/// resource-aware replacement for ad-hoc calls to [`default_num_centers`] /
/// [`heuristic_centers`](crate::term_builder::heuristic_centers).
pub fn plan_spatial_basis(
    n: usize,
    d: usize,
    requested_centers: CenterCountRequest,
    nullspace_order: DuchonNullspaceOrder,
    scale_dims: bool,
    policy: &gam_runtime::resource::ResourcePolicy,
) -> Result<SpatialBasisPlan, BasisError> {
    if n == 0 {
        crate::bail_invalid_basis!("plan_spatial_basis: n must be >= 1");
    }
    if d == 0 {
        crate::bail_invalid_basis!("plan_spatial_basis: d must be >= 1");
    }

    // 1. Resolve center count.
    let centers = match requested_centers {
        CenterCountRequest::Default => default_num_centers(n, d),
        CenterCountRequest::Explicit(k) => k,
        CenterCountRequest::HeuristicCapped { cap } => default_num_centers(n, d).min(cap),
    };

    // 2. Nullspace dimension (Duchon polynomial null space of degree p-1).
    //    `duchon_p_from_nullspace_order` returns m such that the null space is
    //    polynomials of total degree < m, matching `duchon_nullspace_dimension`'s
    //    `max_total_degree = m - 1` argument.
    let m = duchon_p_from_nullspace_order(nullspace_order);
    let nullspace_dim = if m == 0 {
        0
    } else {
        duchon_nullspace_dimension(d, m - 1)
    };

    let p = centers.saturating_add(nullspace_dim);

    // 3. Dense byte estimates.
    let derivative_axes = if scale_dims { d } else { 0 };
    let bytes_per_f64 = std::mem::size_of::<f64>();
    let dense_design_bytes = bytes_per_f64.saturating_mul(n).saturating_mul(p);
    let first_derivative_dense_bytes = dense_design_bytes.saturating_mul(derivative_axes);
    // Diagonal second derivatives are also (D × n × p); off-diagonal cross terms
    // would scale as D^2 but the planner reports the diagonal tier here.
    let second_derivative_dense_bytes = first_derivative_dense_bytes;

    // 4. Pick storage mode based on policy.
    let recommended_storage = match policy.derivative_storage_mode {
        gam_runtime::resource::DerivativeStorageMode::AnalyticOperatorRequired => {
            SpatialStorageMode::OperatorOnly
        }
        gam_runtime::resource::DerivativeStorageMode::MaterializeIfSmall => {
            let budget = policy.max_single_materialization_bytes;
            if derivative_axes == 0 {
                if dense_design_bytes <= budget {
                    SpatialStorageMode::DenseValueDenseDerivatives
                } else {
                    SpatialStorageMode::LazyValueImplicitDerivatives
                }
            } else {
                let total = dense_design_bytes
                    .saturating_add(first_derivative_dense_bytes)
                    .saturating_add(second_derivative_dense_bytes);
                if total <= budget {
                    SpatialStorageMode::DenseValueDenseDerivatives
                } else if dense_design_bytes <= budget {
                    SpatialStorageMode::LazyValueImplicitDerivatives
                } else {
                    SpatialStorageMode::OperatorOnly
                }
            }
        }
        gam_runtime::resource::DerivativeStorageMode::DiagnosticsOnly => {
            // Diagnostic mode still prefers analytic storage for correctness.
            SpatialStorageMode::OperatorOnly
        }
    };

    Ok(SpatialBasisPlan {
        n,
        d,
        centers,
        p_final_estimate: p,
        dense_design_bytes,
        first_derivative_dense_bytes,
        second_derivative_dense_bytes,
        recommended_storage,
    })
}

pub const fn default_spatial_center_strategy(num_centers: usize, d: usize) -> CenterStrategy {
    if d <= 3 {
        CenterStrategy::FarthestPoint { num_centers }
    } else {
        CenterStrategy::EqualMassCovarRepresentative { num_centers }
    }
}

pub fn auto_spatial_center_strategy(num_centers: usize, d: usize) -> CenterStrategy {
    let strategy = if d == 1 {
        // In one dimension, farthest-point selection is the deterministic
        // maximin grid over the observed domain. Equal-mass midpoints leave the
        // low-frequency Duchon radial block slightly under-resolved at the
        // boundaries, and REML then compensates with an over-smooth λ on
        // low-noise signals (#504). The maximin grid matches the native
        // reproducing-kernel interpolation geometry. The default strategy below
        // extends the same space-filling contract to low-dimensional spatial
        // GP bases, where kriging accuracy is governed by fill distance rather
        // than marginal quantile balance.
        CenterStrategy::FarthestPoint { num_centers }
    } else {
        default_spatial_center_strategy(num_centers, d)
    };
    CenterStrategy::Auto(Box::new(strategy))
}

pub const fn center_strategy_is_auto(strategy: &CenterStrategy) -> bool {
    matches!(strategy, CenterStrategy::Auto(_))
}

pub(crate) fn realized_center_strategy(strategy: &CenterStrategy) -> &CenterStrategy {
    match strategy {
        CenterStrategy::Auto(inner) => inner.as_ref(),
        other => other,
    }
}

pub fn center_strategy_kind(strategy: &CenterStrategy) -> CenterStrategyKind {
    match strategy {
        CenterStrategy::Auto(inner) => center_strategy_kind(inner.as_ref()),
        CenterStrategy::UserProvided(_) => CenterStrategyKind::UserProvided,
        CenterStrategy::EqualMass { .. } => CenterStrategyKind::EqualMass,
        CenterStrategy::EqualMassCovarRepresentative { .. } => {
            CenterStrategyKind::EqualMassCovarRepresentative
        }
        CenterStrategy::FarthestPoint { .. } => CenterStrategyKind::FarthestPoint,
        CenterStrategy::KMeans { .. } => CenterStrategyKind::KMeans,
        CenterStrategy::UniformGrid { .. } => CenterStrategyKind::UniformGrid,
    }
}

pub fn center_strategy_num_centers(strategy: &CenterStrategy) -> Option<usize> {
    match strategy {
        CenterStrategy::Auto(inner) => center_strategy_num_centers(inner.as_ref()),
        CenterStrategy::UserProvided(centers) => Some(centers.nrows()),
        CenterStrategy::EqualMass { num_centers }
        | CenterStrategy::EqualMassCovarRepresentative { num_centers }
        | CenterStrategy::FarthestPoint { num_centers }
        | CenterStrategy::KMeans { num_centers, .. } => Some(*num_centers),
        CenterStrategy::UniformGrid { .. } => None,
    }
}

pub fn center_strategy_with_num_centers(
    strategy: &CenterStrategy,
    num_centers: usize,
) -> Result<CenterStrategy, BasisError> {
    validate_center_count(num_centers)?;
    fn rebuild_inner(
        strategy: &CenterStrategy,
        num_centers: usize,
    ) -> Result<CenterStrategy, BasisError> {
        match strategy {
            CenterStrategy::Auto(inner) => rebuild_inner(inner.as_ref(), num_centers),
            CenterStrategy::EqualMass { .. } => Ok(CenterStrategy::EqualMass { num_centers }),
            CenterStrategy::EqualMassCovarRepresentative { .. } => {
                Ok(CenterStrategy::EqualMassCovarRepresentative { num_centers })
            }
            CenterStrategy::FarthestPoint { .. } => {
                Ok(CenterStrategy::FarthestPoint { num_centers })
            }
            CenterStrategy::KMeans { max_iter, .. } => Ok(CenterStrategy::KMeans {
                num_centers,
                max_iter: *max_iter,
            }),
            CenterStrategy::UserProvided(_) | CenterStrategy::UniformGrid { .. } => {
                Err(BasisError::InvalidInput(format!(
                    "cannot replace center count for {:?} strategy",
                    center_strategy_kind(strategy)
                )))
            }
        }
    }
    let rebuilt = rebuild_inner(strategy, num_centers)?;
    Ok(match strategy {
        CenterStrategy::Auto(_) => CenterStrategy::Auto(Box::new(rebuilt)),
        _ => rebuilt,
    })
}

/// Thin-plate basis configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinPlateBasisSpec {
    pub center_strategy: CenterStrategy,
    #[serde(default)]
    pub periodic: Option<Vec<Option<f64>>>,
    pub length_scale: f64,
    pub double_penalty: bool,
    #[serde(default)]
    pub identifiability: SpatialIdentifiability,
    /// Frozen Wood-TPRS radial reparameterization. When `Some`, the builder
    /// reuses this `(raw_radial_cols) × (kept_radial_cols)` matrix instead of
    /// recomputing it from the constrained kernel penalty eigensystem. The
    /// rectangular case is the truncated regression-spline path; carrying it
    /// into prediction guarantees identical radial modes to fit-time.
    #[serde(default)]
    pub radial_reparam: Option<Array2<f64>>,
}

/// Per-smooth identifiability policy for spatial (TPS / Duchon) bases.
///
/// For a raw local basis `B` and parametric design block `C`, the orthogonalized
/// basis is `B_c = B Z` where columns of `Z` span `null((B^T C)^T)`. This enforces:
///   `B_c^T C = 0`
/// in the unweighted inner product, so spatial effects cannot absorb parametric
/// directions that actually exist in the model. The standalone basis builder has
/// only an implicit intercept available, so it centers smooths against that
/// intercept. The term-collection builder augments `C` with explicit linear
/// terms when those terms are present in the formula.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub enum SpatialIdentifiability {
    /// Keep unconstrained basis columns.
    None,
    /// Orthogonalize the smooth against model-owned parametric columns.
    // "Magic" default for modular GAMs with explicit parametric block:
    // keep spatial smooth orthogonal to intercept/linear terms.
    // ApproxKind: Exact (orthogonalization is an exact projection).
    #[default]
    OrthogonalToParametric,
    /// Freeze a fit-time transform `Z`; prediction uses `B_new * Z` unchanged.
    FrozenTransform { transform: Array2<f64> },
}

pub(crate) use sphere_kernels::{
    wahba_sphere_kernel_derivative_dcos_kind, wahba_sphere_kernel_from_cos_kind,
    wahba_sphere_kernel_from_cos_simd_kind, wahba_sphere_kernel_sobolev_derivative_dcos,
};

pub use sphere_spectral::{
    pseudo_s2_truncated_coefficients, sobolev_s2_truncated_coefficients,
    sphere_truncated_spectral_eval,
};

/// Matérn basis configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaternBasisSpec {
    pub center_strategy: CenterStrategy,
    #[serde(default)]
    pub periodic: Option<Vec<Option<f64>>>,
    pub length_scale: f64,
    pub nu: MaternNu,
    #[serde(default)]
    pub include_intercept: bool,
    pub double_penalty: bool,
    #[serde(default)]
    pub identifiability: MaternIdentifiability,
    /// Per-axis anisotropy log-scales η_a (contrasts with Ση_a = 0).
    ///
    /// This implements geometric anisotropy: Λ = κA where A = diag(exp(η_a)),
    /// det(A) = 1. The kernel is evaluated at r = κ|Ah| instead of r = κ|h|.
    /// The decomposition preserves the isotropic scaling law for global κ
    /// and adds d−1 shape parameters for directional relevance.
    ///
    /// Conditional positive definiteness is preserved under any invertible
    /// linear coordinate transform (Schoenberg), so the kernel remains valid.
    ///
    /// When Some, the distance is r = √(Σ_a exp(2η_a) · (x_a - c_a)²).
    /// When None, isotropic distance r = ‖x - c‖ is used.
    #[serde(default)]
    pub aniso_log_scales: Option<Vec<f64>>,
    /// Frozen double-penalty nullspace-shrinkage decision (gam#787/#860).
    ///
    /// `None` (the default, and the cold-build value) = decide whether to emit
    /// the `DoublePenaltyNullspace` candidate via the κ-dependent spectral test in
    /// `build_nullspace_shrinkage_penalty`. `Some(b)` = force the decision (set by
    /// the freeze step from the bootstrap-κ build, mirrored from
    /// `MaternIdentifiability::FrozenTransform`) so the learned-penalty count stays
    /// invariant as the κ-optimizer rebuilds the design at each trial length-scale.
    /// Only consulted when `double_penalty` is true.
    #[serde(default)]
    pub nullspace_shrinkage_survived: Option<bool>,
}

/// Per-smooth identifiability policy for Matérn kernel coefficients.
///
/// These constraints are geometric (center-based), so they are stable across
/// train/predict and do not depend on response weights.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub enum MaternIdentifiability {
    /// Keep the unconstrained kernel coefficient space.
    None,
    /// Enforce `1^T alpha = 0` at center locations (removes constant drift).
    // Safe default with model intercepts: prevent kernel block from absorbing
    // a global mean level.
    #[default]
    CenterSumToZero,
    /// Enforce orthogonality to `[1, c_1, ..., c_d]` at centers.
    /// Use this when explicit linear terms should own global trends.
    CenterLinearOrthogonal,
    /// Freeze a fit-time transform `Z` so prediction cannot drift.
    ///
    /// `nullspace_shrinkage_survived` freezes the double-penalty
    /// nullspace-shrinkage decision alongside the transform (gam#787/#860). The
    /// matern double-penalty path emits a `DoublePenaltyNullspace` candidate iff
    /// `build_nullspace_shrinkage_penalty(&projected_kernel)` finds a near-zero
    /// eigenvalue — but that spectral test is κ-DEPENDENT (its tolerance scales
    /// with `λ_max`), so a near-zero eigenvalue can cross the threshold as the
    /// κ-optimizer rebuilds the design at each trial length-scale. That flips the
    /// learned-penalty count 6↔7 across the rebuild and the rebuilt design's ρ
    /// dimension then disagrees with the frozen joint setup ("joint hyper rho
    /// dimension mismatch" → every κ seed fails startup validation). Freezing the
    /// bootstrap-κ decision here (`Some(true)` = always emit the shrinkage
    /// candidate, `Some(false)` = never) keeps the penalty count INVARIANT across
    /// the κ rebuild so κ actually optimizes. `None` = decide via the spectral
    /// test (the non-frozen / cold-build behavior; also the serde back-compat
    /// default for transforms frozen before this field existed).
    FrozenTransform {
        transform: Array2<f64>,
        #[serde(default)]
        nullspace_shrinkage_survived: Option<bool>,
    },
}

/// Duchon null-space polynomial degree.
///
/// Controls the polynomial null space of the Duchon / polyharmonic spline. The
/// Duchon seminorm `‖D^m f‖²` annihilates all polynomials of total degree
/// `< m`, so those polynomials must be handled as explicit unpenalized columns.
///
/// The user-facing `order` knob selects the polynomial degree cutoff `r`, and
/// the resulting polynomial null space has dimension `C(d + r, r)` where `d`
/// is the covariate dimension.  In the `duchon(...)` formula DSL:
///
/// | `order=` | Variant         | max total degree | null-space dim  |
/// |----------|-----------------|------------------|-----------------|
/// | `0`      | `Zero`          | 0                | `C(d+0,0) = 1`  |
/// | `1`      | `Linear`        | 1                | `C(d+1,1) = d+1`|
/// | `k≥2`    | `Degree(k)`     | k                | `C(d+k,k)`      |
///
/// **How the polynomial null space is consumed during basis construction:**
///
/// 1. `polynomial_block_from_order` materialises an `(n, C(d+r,r))` block `P`
///    of monomials up to total degree `r` at the selected `centers`.
/// 2. `kernel_constraint_nullspace` computes `Z = null(P_centers^T)`, a
///    `(k, k − C(d+r,r))` matrix. Reparameterising the radial kernel
///    coefficients as `α = Z γ` enforces the side condition `P_centers^T α = 0`
///    and yields `k − C(d+r,r)` free kernel parameters.
/// 3. The polynomial block `P_data` evaluated at the data rows is appended to
///    the kernel block `Φ Z`, giving a total of
///    `(k − C(d+r,r)) + C(d+r,r) = k` columns before the spatial
///    identifiability transform.  Crucially, the total width equals the
///    requested center count `k`, **not** `k + C(d+r,r)`.
///
/// **Example — `duchon(PC1, PC2, PC3, centers=10, order=1)` (d=3):**
///
/// - Polynomial null space: `C(3+1,1) = 4` monomials `{1, x₁, x₂, x₃}`.
/// - Kernel columns after constraint: `10 − 4 = 6`.
/// - Appended polynomial block: 4 columns.
/// - Pre-identifiability total: `6 + 4 = 10` columns, i.e. exactly `centers`.
///
/// The variant naming matches the Duchon `m` parameter:
/// `Zero` → `m=1`, `Linear` → `m=2`, `Degree(k)` → `m=k+1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DuchonNullspaceOrder {
    Zero,
    Linear,
    Degree(usize),
}

/// Duchon-like basis configuration with explicit low-frequency null-space
/// control and explicit spectral power.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DuchonBasisSpec {
    pub center_strategy: CenterStrategy,
    #[serde(default)]
    pub periodic: Option<Vec<Option<f64>>>,
    /// Optional hybrid Matérn width. `None` means pure scale-free Duchon with
    /// spectrum `||w||^(2p + 2s)`. `Some(length_scale)` enables the hybrid
    /// spectrum `||w||^(2p) * (kappa^2 + ||w||^2)^s`, `kappa = 1/length_scale`.
    pub length_scale: Option<f64>,
    /// Literal Duchon spectral power `s` (`f64`, fractional values fully
    /// threaded end-to-end). The pure-Duchon kernel exponent is `2(p + s) − d`,
    /// so this is the knob that sets `φ(r)`: `s = 0` is the integer-order Duchon
    /// kernel `r^{2p−d}` (its `r²·log r` log case in even `d`, ≡ the thin-plate
    /// kernel); `s = (d − 1)/2` gives the cubic `r³` in every dimension.
    ///
    /// This field is taken LITERALLY by the basis builder — `power = 0` means
    /// `s = 0`, NOT "use a default". The magic cubic default (applied when the
    /// user gives no explicit power) is a request-layer choice resolved by the
    /// formula / CLI / pyffi front-ends via [`duchon_cubic_default`]; by the time
    /// a spec reaches the builder this value is the final intended `s`. The
    /// hybrid Duchon–Matérn path (`length_scale = Some`) still requires an
    /// integer `s` (read via `spec.power_as_usize()`).
    pub power: f64,
    pub nullspace_order: DuchonNullspaceOrder,
    #[serde(default)]
    pub identifiability: SpatialIdentifiability,
    /// Per-axis anisotropy log-scales η_a.
    ///
    /// For hybrid Duchon (`length_scale=Some`), these are centered contrasts in
    /// the decomposition Λ = κA with det(A)=1. For pure Duchon
    /// (`length_scale=None`), they parameterize shape-only axis warping on the
    /// public path and are centered before basis evaluation/writeback so no
    /// global length scale is introduced.
    ///
    /// When Some, the distance is r = √(Σ_a exp(2η_a) · (x_a - c_a)²).
    /// When None, isotropic distance r = ‖x - c‖ is used.
    #[serde(default)]
    pub aniso_log_scales: Option<Vec<f64>>,
    #[serde(default)]
    pub operator_penalties: DuchonOperatorPenaltySpec,
    #[serde(default)]
    pub boundary: OneDimensionalBoundary,
    /// Data-metric radial reparameterization `V` (#1355), mirroring the
    /// thin-plate Wood-TPRS reparam. When `Some`, the constrained kernel
    /// transform is folded to `Z·V` so the realized design columns rotate into
    /// the `G_c`-orthonormal generalized eigenbasis of `Ω_c v = μ G_c v` and the
    /// native penalty becomes the diagonal curvature-per-unit-data-variance
    /// spectrum (mgcv's cliff), preventing the REML over-smoothing collapse to
    /// EDF = 1. Frozen at the cold dense build and replayed verbatim by the
    /// predict / κ-trial / ψ-derivative paths so they stay bit-consistent with
    /// the fit-time design. `None` on the lazy/streaming path (huge `n`), which
    /// retains the original constrained basis.
    #[serde(default)]
    pub radial_reparam: Option<Array2<f64>>,
}

impl DuchonBasisSpec {
    /// Integer view of `power` for the existing integer-only downstream chain.
    /// Non-finite or non-integer values fall back to `0` (the integer-only
    /// validators downstream already reject this case with a clear message).
    pub fn power_as_usize(&self) -> usize {
        duchon_power_to_usize(self.power)
    }
}

/// Convert a Duchon spectral-power `f64` into the integer view used by the
/// closed-form code paths. Non-finite, negative, or fractional values clamp to
/// `0` so the validator downstream emits the canonical error.
pub fn duchon_power_to_usize(power: f64) -> usize {
    if !power.is_finite() || power < 0.0 {
        return 0;
    }
    let rounded = power.round();
    if (rounded - power).abs() > 1e-9 {
        return 0;
    }
    rounded as usize
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DuchonOperatorPenaltySpec {
    pub mass: OperatorPenaltySpec,
    pub tension: OperatorPenaltySpec,
    pub stiffness: OperatorPenaltySpec,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum OperatorPenaltySpec {
    Active {
        initial_log_lambda: f64,
        prior: Option<RhoPrior>,
    },
    Disabled,
}

impl Default for DuchonOperatorPenaltySpec {
    fn default() -> Self {
        // ALL ON. The Duchon penalty is a Hilbert scale: curvature is the
        // always-on exact RKHS `Primary` Gram and the trend ridge is always on;
        // the lower orders — mass (amplitude `Σ(f−f̄)²`) and tension (first-order
        // roughness `Σ‖∇f‖²`) — are active here, collocated on a density-blind
        // data-support sample. REML deselects any the data don't support (SPEC:
        // recover the null by default, opt INTO overfitting). Stiffness (`D2`)
        // stays off — `Primary` is the exact, superior curvature. (The Matérn
        // collocation overlay builds its own `all_active()`; SAE atoms, which
        // ship only `Primary`, use `all_disabled()`.)
        Self {
            mass: OperatorPenaltySpec::Active {
                initial_log_lambda: 0.0,
                prior: None,
            },
            tension: OperatorPenaltySpec::Active {
                initial_log_lambda: 0.0,
                prior: None,
            },
            stiffness: OperatorPenaltySpec::Disabled,
        }
    }
}

impl DuchonOperatorPenaltySpec {
    pub fn has_active_operator_penalty(&self) -> bool {
        matches!(self.mass, OperatorPenaltySpec::Active { .. })
            || matches!(self.tension, OperatorPenaltySpec::Active { .. })
            || matches!(self.stiffness, OperatorPenaltySpec::Active { .. })
    }

    pub fn all_disabled() -> Self {
        Self {
            mass: OperatorPenaltySpec::Disabled,
            tension: OperatorPenaltySpec::Disabled,
            stiffness: OperatorPenaltySpec::Disabled,
        }
    }

    /// All three operator dials active — used by the Matérn collocation overlay.
    pub fn all_active() -> Self {
        let active = || OperatorPenaltySpec::Active {
            initial_log_lambda: 0.0,
            prior: None,
        };
        Self {
            mass: active(),
            tension: active(),
            stiffness: active(),
        }
    }

    /// Operator-penalty dials appropriate for a Matérn-ν kernel in dimension `d`.
    ///
    /// The Matérn-ν RKHS is the Sobolev space `H^m` with `m = ν + d/2`: its
    /// squared norm controls the order-`j` derivative in L2 exactly when
    /// `j ≤ m`. The collocation overlay penalizes the squared L2 norms of the
    /// value (mass, `D0`, j=0), gradient (tension, `D1`, j=1) and Hessian
    /// (stiffness, `D2`, j=2). Activating a penalty whose derivative order
    /// exceeds the RKHS smoothness (`j > m`) imposes a roughness constraint the
    /// true kernel does NOT — it over-smooths the reduced-rank fit relative to
    /// the exact GP (mgcv `bs="gp"`, GpGp).
    ///
    /// Concretely the roughest Matérn, ν=1/2 in d=1 (`m = 1`), is the
    /// Ornstein–Uhlenbeck/exponential kernel: an H¹ process whose sample paths
    /// are continuous but non-differentiable. Although `∫(f')²` is finite on
    /// its RKHS, the kernel itself already encodes the H¹ control; layering an
    /// extra tension dial on top biases the reduced-rank fit toward the smooth
    /// `C¹` functions the kernel does not favour (and stiffness `D2` toward
    /// `C²`), collapsing held-out oscillation (#707). We therefore gate each
    /// operator on `j < m` STRICTLY: mass (j=0) is always on, tension (j=1) is
    /// on for `m > 1`, stiffness (j=2) is on for `m > 2`. For ν ≥ 3/2 (or any
    /// d ≥ 2) every dial is active, recovering `all_active`; only the
    /// genuinely rough ν=1/2 (d=1) kernel — where the Sobolev order sits
    /// exactly on a derivative boundary — drops the higher operators.
    pub fn matern_for_smoothness(nu: MaternNu, d: usize) -> Self {
        let m = nu.half_integer_value() + 0.5 * d as f64;
        // Tolerance so an exact half-integer Sobolev order (e.g. m = 1.0 for
        // ν=1/2, d=1) reliably DISABLES the matching-order operator instead
        // of flipping on a float-equality knife-edge.
        const ORDER_EPS: f64 = 1e-9;
        let active = || OperatorPenaltySpec::Active {
            initial_log_lambda: 0.0,
            prior: None,
        };
        let gate = |order: f64| {
            if m > order + ORDER_EPS {
                active()
            } else {
                OperatorPenaltySpec::Disabled
            }
        };
        Self {
            mass: active(),
            tension: gate(1.0),
            stiffness: gate(2.0),
        }
    }
}

pub fn minimum_duchon_power_for_operator_penalties(
    dim: usize,
    nullspace_order: DuchonNullspaceOrder,
    max_operator_derivative_order: usize,
) -> usize {
    let p = duchon_p_from_nullspace_order(nullspace_order);
    let mut s = 0usize;
    while 2 * (p + s) <= dim + max_operator_derivative_order {
        s += 1;
    }
    s
}

/// Resolve a fully admissible Duchon `(nullspace_order, power)` pair.
///
/// Three constraints fold into one resolution:
///   (a) operator collocation up to `max_op`:        `2(p + s) > d + max_op`
///   (b) pure-mode CPD vs polynomial nullspace P_p:  `2s < d`
///       (Wendland Thm 8.17: pure polyharmonic kernel of order m = p+s in
///        R^d is CPD of order `m − ⌊d/2⌋ + 1[d even, log] / m − (d−1)/2
///        [d odd]`, and Duchon interpolation against P_p is well-posed iff
///        CPD order ≤ p, which collapses to `2s < d` since 2s, d are
///        integers and 2s is even.)
///   (a) implies the kernel-existence condition `2(p + s) > d`.
///   (b) is dropped when `length_scale` is `Some` (hybrid Matérn-blended
///       kernel is strictly PD, CPD order 0).
///
/// Strategy: at the requested `nullspace_order`, take the smallest `s`
/// satisfying (a). If that `s` violates (b) in pure mode, escalate the
/// nullspace order by one and retry. Termination: at `p ≥ ⌈(d+max_op)/2⌉ + 1`
/// the operator constraint (a) admits `s = 0`, and `0 < d` satisfies (b)
/// for any `d ≥ 1`, so escalation always converges.
///
/// The returned nullspace order is monotone in the request: it never
/// decreases the user's requested order — only strengthens it when pure-mode
/// CPD requires a richer polynomial absorption space.
pub fn resolve_duchon_orders(
    dim: usize,
    requested_nullspace_order: DuchonNullspaceOrder,
    max_operator_derivative_order: usize,
    length_scale: Option<f64>,
) -> (DuchonNullspaceOrder, usize) {
    assert!(dim >= 1, "Duchon basis requires dim >= 1");
    let pure = length_scale.is_none();
    let mut nullspace = requested_nullspace_order;
    // Bounded loop: escalation terminates by the argument above.
    for _ in 0..=(dim + max_operator_derivative_order + 1) {
        let p = duchon_p_from_nullspace_order(nullspace);
        // Smallest s with 2(p + s) > d + max_op:
        //   2p > d + max_op            ⇒ s = 0
        //   else s = ⌈(d + max_op + 1 − 2p) / 2⌉ = (d + max_op + 2 − 2p) / 2
        let s_op = if 2 * p > dim + max_operator_derivative_order {
            0
        } else {
            (dim + max_operator_derivative_order + 2 - 2 * p) / 2
        };
        if !pure || 2 * s_op < dim {
            return (nullspace, s_op);
        }
        nullspace = duchon_next_nullspace_order(nullspace);
    }
    // Bounded-loop fallback: by the analysis in the docstring, for
    // `p >= ceil((dim + max_op) / 2) + 1` the operator constraint admits
    // `s = 0` and (in pure mode) `0 < dim` satisfies the kernel-existence
    // condition. The loop above always reaches that regime within the bound,
    // so returning the last `nullspace` with `s = 0` is a valid answer.
    (nullspace, 0)
}

#[inline]
pub(crate) fn duchon_next_nullspace_order(order: DuchonNullspaceOrder) -> DuchonNullspaceOrder {
    match order {
        DuchonNullspaceOrder::Zero => DuchonNullspaceOrder::Linear,
        DuchonNullspaceOrder::Linear => DuchonNullspaceOrder::Degree(2),
        DuchonNullspaceOrder::Degree(k) => DuchonNullspaceOrder::Degree(k + 1),
    }
}

pub(crate) fn duchon_previous_nullspace_order(order: DuchonNullspaceOrder) -> DuchonNullspaceOrder {
    match order {
        DuchonNullspaceOrder::Zero => DuchonNullspaceOrder::Zero,
        DuchonNullspaceOrder::Linear => DuchonNullspaceOrder::Zero,
        DuchonNullspaceOrder::Degree(2) => DuchonNullspaceOrder::Linear,
        DuchonNullspaceOrder::Degree(k) => DuchonNullspaceOrder::Degree(k - 1),
    }
}

/// Returns the maximum derivative order required by the *active* operator
/// penalties: 2 if stiffness is Active, else 1 if tension is Active, else 0.
/// Mass-only (or no active operator) penalties only require kernel validity
/// (`2(p+s) > d`), tension requires D1 collocation (`2(p+s) > d+1`), and
/// stiffness requires D2 collocation (`2(p+s) > d+2`).
pub fn duchon_max_active_operator_derivative_order(
    operator_penalties: &DuchonOperatorPenaltySpec,
) -> usize {
    if matches!(
        operator_penalties.stiffness,
        OperatorPenaltySpec::Active { .. }
    ) {
        2
    } else if matches!(
        operator_penalties.tension,
        OperatorPenaltySpec::Active { .. }
    ) {
        1
    } else {
        0
    }
}

/// Metadata returned by generic basis builders.
#[derive(Debug, Clone)]
pub enum BasisMetadata {
    BSpline1D {
        knots: Array1<f64>,
        identifiability_transform: Option<Array2<f64>>,
        periodic: Option<(f64, f64, usize)>,
        /// Effective B-spline polynomial degree carried by `knots`.
        ///
        /// Persisted alongside `knots` so prediction can reconstruct an
        /// evaluator that matches fit-time geometry, even when the fit-time
        /// auto-shrink (issue #340) reduced the user's requested degree to
        /// fit the available data (`n` too small for cubic ⇒ quadratic ⇒
        /// linear). When `None` the consumer should fall back to the
        /// upstream `BSplineBasisSpec.degree` (legacy / non-shrunk path).
        degree: Option<usize>,
        /// Human-readable description of an automatic basis shrink (issue #340)
        /// when the user's requested `(degree, num_internal_knots)` exceeded the
        /// available evaluation count `n`. `Some(note)` records the before→after
        /// configuration; `None` means no auto-shrink occurred for this basis.
        auto_shrink_note: Option<String>,
    },
    /// Natural cubic regression spline (`bs="cr"`/`"cs"`) metadata (#1074).
    ///
    /// `knots` are the `k` Lancaster–Salkauskas knots that index the basis
    /// values directly (basis dim = `knots.len()`). Predict-time rebuilds
    /// reconstruct the cr geometry from `knots` and replay the captured
    /// `identifiability_transform` exactly, mirroring `BSpline1D`.
    CubicRegression1D {
        knots: Array1<f64>,
        identifiability_transform: Option<Array2<f64>>,
    },
    ThinPlate {
        centers: Array2<f64>,
        length_scale: f64,
        periodic: Option<Vec<Option<f64>>>,
        identifiability_transform: Option<Array2<f64>>,
        /// Per-column standard deviations used for input standardization (d > 1).
        input_scales: Option<Vec<f64>>,
        /// Wood-TPRS radial reparameterization carried into prediction so the
        /// rotated radial basis at predict-time matches fit-time exactly. `None`
        /// in the lazy/streaming path which retains the original basis.
        radial_reparam: Option<Array2<f64>>,
    },
    Sphere {
        centers: Array2<f64>,
        penalty_order: usize,
        method: SphereMethod,
        max_degree: Option<usize>,
        wahba_kernel: SphereWahbaKernel,
        constraint_transform: Option<Array2<f64>>,
    },
    /// Constant-curvature (`M_κ`) geodesic-kernel smooth (#944). `kappa` and
    /// the realized `length_scale` are persisted so predict-time (and the
    /// future ψ-channel per-trial) rebuilds replay the exact fit-time
    /// geometry; `constraint_transform` is the composed `z · z_parametric`
    /// frozen by the global identifiability pipeline (#532 pattern).
    ConstantCurvature {
        centers: Array2<f64>,
        kappa: f64,
        length_scale: f64,
        constraint_transform: Option<Array2<f64>>,
    },
    /// Measure-jet spline smooth: multiscale local-jet-residual energy of the
    /// empirical measure, quadratured on the center set. `centers` are the
    /// REALIZED barycenter nodes; `order_s` stores the spec's order sentinel
    /// verbatim as the mode marker (0.0 = per-level/spectral, > 0 = fused
    /// pin — persisting a realized default would flip the rebuilt mode). The
    /// penalty depends on the FIT data through `masses`, the realized
    /// `eps_band`, the support anchors, and the normalization scales, so all
    /// are persisted and replayed verbatim by
    /// predict-time (and per-ψ-trial) rebuilds — recomputing either from
    /// predict rows would change the penalty the coefficients were estimated
    /// under. `constraint_transform` is the composed `z · z_parametric`
    /// frozen by the global identifiability pipeline (#532 pattern).
    MeasureJet {
        centers: Array2<f64>,
        input_scales: Option<Vec<f64>>,
        length_scale: f64,
        eps_band: Vec<f64>,
        order_s: f64,
        alpha: f64,
        tau0: f64,
        masses: Array1<f64>,
        support_means: Vec<f64>,
        penalty_normalization_scales: Vec<f64>,
        raw_penalty_normalization_scales: Vec<f64>,
        fused_penalty_normalization_scale: Option<f64>,
        constraint_transform: Option<Array2<f64>>,
    },
    Matern {
        centers: Array2<f64>,
        length_scale: f64,
        periodic: Option<Vec<Option<f64>>>,
        nu: MaternNu,
        include_intercept: bool,
        identifiability_transform: Option<Array2<f64>>,
        /// Per-column standard deviations used for input standardization (d > 1).
        input_scales: Option<Vec<f64>>,
        /// Per-axis anisotropy log-scales η_a for geometric anisotropy.
        /// When Some, distance is r = √(Σ_a exp(2η_a) · (x_a - c_a)²).
        aniso_log_scales: Option<Vec<f64>>,
        /// Realized double-penalty nullspace-shrinkage decision at this build
        /// (gam#787/#860). The freeze step pins this into
        /// `MaternIdentifiability::FrozenTransform::nullspace_shrinkage_survived`
        /// so the κ-optimizer's per-trial rebuilds keep the learned-penalty count
        /// invariant (otherwise the κ-dependent spectral test flips it 6↔7 → "joint
        /// hyper rho dimension mismatch").
        nullspace_shrinkage_survived: bool,
    },
    Duchon {
        centers: Array2<f64>,
        length_scale: Option<f64>,
        periodic: Option<Vec<Option<f64>>>,
        power: f64,
        nullspace_order: DuchonNullspaceOrder,
        identifiability_transform: Option<Array2<f64>>,
        /// Per-column standard deviations used for input standardization (d > 1).
        input_scales: Option<Vec<f64>>,
        /// Per-axis anisotropy log-scales η_a, stored for prediction.
        aniso_log_scales: Option<Vec<f64>>,
        /// Support points used to build the active lower-order operator
        /// penalties (mass/tension/stiffness). Stored so runtime adaptive
        /// caches can rebuild the exact same operator rows instead of guessing
        /// from centers.
        operator_collocation_points: Option<Array2<f64>>,
        /// Data-metric radial reparameterization `V` (#1355). When `Some`, the
        /// constrained kernel transform is folded to `Z·V` so predict-time and
        /// κ-trial rebuilds replay the exact fit-time rotated radial basis.
        /// `None` on the lazy/streaming path (original constrained basis).
        radial_reparam: Option<Array2<f64>>,
    },
    Pca {
        feature_cols: Vec<usize>,
        basis_matrix: Array2<f64>,
        centered: bool,
        smooth_penalty: f64,
        center_mean: Option<Array1<f64>>,
        pca_basis_path: Option<std::path::PathBuf>,
        chunk_size: usize,
    },
    TensorBSpline {
        feature_cols: Vec<usize>,
        knots: Vec<Array1<f64>>,
        degrees: Vec<usize>,
        periods: Vec<Option<f64>>,
        /// Per-margin flag: `true` when that margin is a natural cubic
        /// regression spline (`NaturalCubicRegression` knotspec) rather than an
        /// open/periodic B-spline (#1074). Persisted so the tensor freeze
        /// rebuilds the cr marginal knotspec (value-at-knot) instead of an open
        /// `Provided(knots)` B-spline, keeping predict-time marginals identical
        /// to the fit-time cr margins. Defaults to all-`false` (legacy B-spline
        /// tensors) when deserialized from an older persisted model (the
        /// older-model default is applied on the persisted `SmoothBasisSpec`
        /// side; `BasisMetadata` itself is transient builder output and is not
        /// serde-serialized, so it carries no `#[serde]` attributes).
        is_cr: Vec<bool>,
        identifiability_transform: Option<Array2<f64>>,
    },
    SphereHarmonics {
        max_degree: usize,
        radians: bool,
    },
    /// Wrap an inner basis metadata to record a multiplicative `by` (continuous or
    /// factor) along a column of the dataset.
    BySmooth {
        inner: Box<BasisMetadata>,
        by_col: usize,
        levels: Option<Vec<u64>>,
        ordered: bool,
    },
    /// Factor-by-smooth (mgcv-style `s(x, by=g, bs="fs"|"sz"|"re")`).
    FactorSmooth {
        continuous_cols: Vec<usize>,
        group_col: usize,
        knots: Array1<f64>,
        degree: usize,
        periodic: Option<(f64, f64, usize)>,
        group_levels: Vec<u64>,
        flavour: String,
        /// `true` when the per-level marginal is a cubic regression spline
        /// (`NaturalCubicRegression` knotspec, mgcv's `bs="sz"` default marginal,
        /// #1074). Predict-time freeze must then restore a cr knotspec from the
        /// stored value-knots rather than treating them as a B-spline knot
        /// vector. Defaults to `false` (B-spline marginal) for backward compat.
        marginal_is_cr: bool,
    },
}

/// Standardized basis build result for engine-level composition.
#[derive(Clone)]
pub struct BasisBuildResult {
    pub design: DesignMatrix,
    pub penalties: Vec<Array2<f64>>,
    pub nullspace_dims: Vec<usize>,
    pub penaltyinfo: Vec<PenaltyInfo>,
    pub metadata: BasisMetadata,
    /// Optional factored rowwise-Kronecker representation for tensor-product
    /// bases. When present, downstream code can keep the design operator-backed
    /// instead of forcing a fully materialized `n x prod(q_j)` block.
    pub kronecker_factored: Option<KroneckerFactoredBasis>,
    /// Per-active-penalty operator handles (parallel to `penalties`). Each
    /// entry is `Some(op)` when the closed-form factory emitted an op-form
    /// penalty bit-equivalent to the dense matrix, `None` for ordinary dense
    /// penalties. Downstream consumers route through the `Some` entries to
    /// avoid materializing dense `p x p` Grams in exact operator algebra.
    pub ops: Vec<Option<std::sync::Arc<dyn crate::analytic_penalties::PenaltyOp>>>,
    /// Per-active-penalty null-space eigenvector matrices (parallel to
    /// `penalties`). Each entry is `Some(U_null)` with `U_null.ncols() ==
    /// nullspace_dims[k]` when the active block has a non-trivial null space
    /// (eigenvalues ≤ spectral tolerance), and `None` when the block is
    /// already full-rank. The columns of `U_null` are the eigenvectors of
    /// `sym_penalty` at the (near-)zero eigenvalues — i.e., an orthonormal
    /// basis of `null(S_block)` in the block's own coordinate system.
    ///
    /// This is the raw spectral data that the construction pipeline uses to
    /// absorb each smooth's penalty null space into the parametric block
    /// (reparameterize-and-split). Without absorption the inner Newton solve
    /// cannot converge on data whose unpenalized signal lies along a null
    /// direction of `S` (phantom-multiplier refusal at the KKT certificate).
    pub null_eigenvectors: Vec<Option<Array2<f64>>>,
    /// Joint-null absorption rotation for this basis, when the basis carries
    /// any penalties with a non-trivial joint null space.
    ///
    /// `Some(rotation)` records `Q = [U_range | U_null]` where `U_null` spans
    /// the joint null space `null(Σ_k S_k)` over this basis's active
    /// penalties (unscaled — the structural joint null is independent of
    /// `λ`). After the basis pipeline applies this rotation, the design
    /// becomes `X · Q` and each penalty becomes `Qᵀ S_k Q`, block-diagonal
    /// with a guaranteed-zero null tail. The same `Q` must be replayed at
    /// prediction time, so it is persisted in the fitted model. `None`
    /// indicates either no penalties on this basis, or a full-rank joint
    /// penalty (joint nullity = 0). A `Some` value is never recorded with
    /// `joint_nullity == 0` — the `None` discriminant is canonical for
    /// "nothing to absorb".
    ///
    /// Stage-2 commit A: this field is plumbed into the struct but neither
    /// computed nor applied yet. Stage-2 commit B populates it; Stage-2
    /// commit D applies the rotation to `design` and `penalties`.
    pub joint_null_rotation: Option<JointNullRotation>,
}

/// Joint-null absorption rotation, attached to a smooth's basis when the
/// basis's joint penalty `Σ_k S_k` has a non-trivial null space.
///
/// The `rotation` field stores the orthonormal eigenvector matrix
/// `Q = [U_range | U_null]` of the symmetric joint penalty: the first
/// `range_dim = rotation.ncols() - joint_nullity` columns span
/// `range(Σ_k S_k)`; the remaining `joint_nullity` columns span
/// `null(Σ_k S_k)`. After the pipeline applies the rotation, the smooth's
/// coefficient vector satisfies `β = Q · γ`, the design becomes `X · Q`,
/// and each per-block penalty `S_k` becomes `Qᵀ S_k Q`, which is guaranteed
/// block-diagonal with a zero `(joint_nullity × joint_nullity)` tail
/// (because the joint null annihilates every active `S_k`).
#[derive(Clone, Serialize, Deserialize)]
pub struct JointNullRotation {
    /// `(p_smooth × p_smooth)` orthonormal matrix; range columns first,
    /// joint-null columns last.
    pub rotation: Array2<f64>,
    /// Number of columns at the tail of `rotation` that span the joint
    /// null space. Always `> 0` when wrapped in `Some` — the value `0`
    /// is encoded as `None`.
    pub joint_nullity: usize,
}

impl std::fmt::Debug for JointNullRotation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JointNullRotation")
            .field(
                "rotation",
                &format_args!("{}×{}", self.rotation.nrows(), self.rotation.ncols()),
            )
            .field("joint_nullity", &self.joint_nullity)
            .finish()
    }
}

impl std::fmt::Debug for BasisBuildResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BasisBuildResult")
            .field("design", &self.design)
            .field("penalties", &self.penalties)
            .field("nullspace_dims", &self.nullspace_dims)
            .field("penaltyinfo", &self.penaltyinfo)
            .field("metadata", &self.metadata)
            .field("kronecker_factored", &self.kronecker_factored)
            .field(
                "ops",
                &format_args!(
                    "[{}]",
                    self.ops
                        .iter()
                        .map(|o| if o.is_some() { "Some" } else { "None" })
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )
            .field(
                "null_eigenvectors",
                &format_args!(
                    "[{}]",
                    self.null_eigenvectors
                        .iter()
                        .map(|u| match u {
                            Some(m) => format!("Some({}x{})", m.nrows(), m.ncols()),
                            None => "None".to_string(),
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )
            .field("joint_null_rotation", &self.joint_null_rotation)
            .finish()
    }
}

/// Factored tensor-product basis metadata for operator-backed downstream use.
#[derive(Debug)]
pub struct KroneckerFactoredBasis {
    /// Marginal design matrices: `marginal_designs[j]` is `(n, q_j)`.
    pub marginal_designs: Vec<Array2<f64>>,
    /// Marginal penalty matrices: `marginal_penalties[k]` is `(q_k, q_k)`.
    pub marginal_penalties: Vec<Array2<f64>>,
    /// Marginal basis dimensions: `[q_0, ..., q_{d-1}]`.
    pub marginal_dims: Vec<usize>,
    /// Whether the system includes a global ridge (double) penalty.
    pub has_double_penalty: bool,
    /// λ-invariant tensor structure (marginal eigensystems, reparameterized
    /// marginals, shrinkage scale), memoized once per fit. The marginal
    /// designs/penalties are fixed for the whole fit, so the expensive marginal
    /// `eigh()` and `B_k·U_k` GEMMs only need to run once — every outer REML
    /// iterate (50+ on the #1082 tensor cases) then reuses this. Filled lazily
    /// on first use via [`Self::invariant_structure`]. NOT serialized and reset
    /// to empty on `Clone` (it is purely a within-fit performance cache; a fresh
    /// owner recomputes on first demand, keeping every result bit-identical).
    invariant: std::sync::OnceLock<std::sync::Arc<crate::kronecker::KroneckerInvariantStructure>>,
}

impl Clone for KroneckerFactoredBasis {
    fn clone(&self) -> Self {
        Self {
            marginal_designs: self.marginal_designs.clone(),
            marginal_penalties: self.marginal_penalties.clone(),
            marginal_dims: self.marginal_dims.clone(),
            has_double_penalty: self.has_double_penalty,
            // Propagate the memoized structure when present so a clone made
            // mid-fit keeps the hoist; otherwise start empty (recomputed on
            // first demand, identical result).
            invariant: match self.invariant.get() {
                Some(s) => {
                    let cell = std::sync::OnceLock::new();
                    cell.get_or_init(|| std::sync::Arc::clone(s));
                    cell
                }
                None => std::sync::OnceLock::new(),
            },
        }
    }
}

impl KroneckerFactoredBasis {
    /// Construct from the fixed marginal data with an empty invariant cache.
    pub fn new(
        marginal_designs: Vec<Array2<f64>>,
        marginal_penalties: Vec<Array2<f64>>,
        marginal_dims: Vec<usize>,
        has_double_penalty: bool,
    ) -> Self {
        Self {
            marginal_designs,
            marginal_penalties,
            marginal_dims,
            has_double_penalty,
            invariant: std::sync::OnceLock::new(),
        }
    }

    /// Lazily compute (once) and return the λ-invariant tensor structure
    /// (marginal eigensystems, reparameterized marginals, shrinkage scale).
    ///
    /// Computed from the fixed marginal designs/penalties, so the result is the
    /// same on every call within a fit; the first call pays the `eigh()` cost
    /// and every later call is a pointer load. Because the cache is keyed on the
    /// fixed marginal data and `marginal_penalties`/`marginal_designs` are
    /// immutable for the fit's lifetime, no invalidation is needed.
    pub fn invariant_structure(
        &self,
    ) -> Result<std::sync::Arc<crate::kronecker::KroneckerInvariantStructure>, BasisError> {
        // Fast path: already memoized.
        if let Some(s) = self.invariant.get() {
            return Ok(std::sync::Arc::clone(s));
        }
        // Compute outside the cell (fallible) and install via `get_or_init`. If a
        // concurrent racer already won, `get_or_init` drops our `computed` and
        // returns the stored one; either way the value is the unique function of
        // the fixed marginal data, so the returned Arc is correct.
        let computed = std::sync::Arc::new(crate::kronecker::KroneckerInvariantStructure::compute(
            &self.marginal_designs,
            &self.marginal_penalties,
            &self.marginal_dims,
        )?);
        let installed = self.invariant.get_or_init(|| computed);
        Ok(std::sync::Arc::clone(installed))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PenaltySource {
    Primary,
    DoublePenaltyNullspace,
    OperatorMass,
    OperatorTension,
    OperatorStiffness,
    /// One per input axis `a` of a multivariate Duchon smooth: the gradient
    /// energy along axis `a`, `Σ(∂f/∂x_a)²`, each with its own REML λ_a. REML
    /// shrinks an axis's contribution toward flat only when it does not earn
    /// its keep — penalty-based ARD / variable relevance, the replacement for
    /// brittle kernel-η optimization. Emitted when `scale_dims` is on.
    OperatorRelevance {
        axis: usize,
    },
    TensorMarginal {
        dim: usize,
    },
    TensorSeparable {
        penalized_margins: Vec<usize>,
    },
    TensorGlobalRidge,
    Other(String),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PenaltyDropReason {
    ZeroMatrix,
    NumericalRankZero,
}

fn default_normalization_scale() -> f64 {
    1.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PenaltyInfo {
    pub source: PenaltySource,
    pub original_index: usize,
    pub active: bool,
    pub effective_rank: usize,
    pub dropped_reason: Option<PenaltyDropReason>,
    pub nullspace_dim_hint: usize,
    #[serde(default = "default_normalization_scale")]
    pub normalization_scale: f64,
    /// Kronecker factors preserved from tensor penalty construction.
    /// When present, spectral decomposition can use per-factor eigendecomposition.
    #[serde(skip)]
    pub kronecker_factors: Option<Vec<Array2<f64>>>,
}

#[derive(Clone)]
pub struct PenaltyCandidate {
    pub matrix: Array2<f64>,
    pub nullspace_dim_hint: usize,
    pub source: PenaltySource,
    pub normalization_scale: f64,
    /// Optional Kronecker factors whose product equals `matrix`.
    /// When present, spectral decomposition can be done per-factor
    /// (O(Σ q_j³) instead of O((Π q_j)³)).
    pub kronecker_factors: Option<Vec<Array2<f64>>>,
    /// Optional operator-form handle whose `as_dense()` matches `matrix`. When
    /// populated by the closed-form factories, this is propagated through to
    /// `CanonicalPenaltyBlock` so downstream consumers can use exact matvec
    /// algebra without rebuilding the dense Gram. When `None`, only the dense
    /// `matrix` path is available.
    pub op: Option<std::sync::Arc<dyn crate::analytic_penalties::PenaltyOp>>,
}

impl std::fmt::Debug for PenaltyCandidate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PenaltyCandidate")
            .field(
                "matrix",
                &format_args!("{}×{}", self.matrix.nrows(), self.matrix.ncols()),
            )
            .field("nullspace_dim_hint", &self.nullspace_dim_hint)
            .field("source", &self.source)
            .field("normalization_scale", &self.normalization_scale)
            .field(
                "kronecker_factors",
                &self.kronecker_factors.as_ref().map(|v| v.len()),
            )
            .field("op", &self.op.as_ref().map(|o| o.dim()))
            .finish()
    }
}

#[derive(Clone)]
pub struct CanonicalPenaltyBlock {
    pub sym_penalty: Array2<f64>,
    /// Eigenvalues from spectral decomposition (retained to avoid recomputation).
    pub eigenvalues: Array1<f64>,
    /// Eigenvectors from spectral decomposition (retained to avoid recomputation).
    pub eigenvectors: Array2<f64>,
    pub rank: usize,
    pub nullity: usize,
    /// Number of genuine negative-curvature eigendirections (`ev < -tol`).
    /// A non-PSD penalty has `negative_dim > 0`; these directions are
    /// neither range nor null and are never absorbed as unpenalized (#1425).
    pub negative_dim: usize,
    pub tol: f64,
    pub iszero: bool,
    /// Optional operator-form handle that is bit-equivalent to `sym_penalty`.
    /// Propagated from `PenaltyCandidate.op` when present so downstream
    /// consumers can use matvec without rebuilding the dense Gram.
    pub op: Option<std::sync::Arc<dyn crate::analytic_penalties::PenaltyOp>>,
}

impl std::fmt::Debug for CanonicalPenaltyBlock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CanonicalPenaltyBlock")
            .field(
                "sym_penalty",
                &format_args!("{}×{}", self.sym_penalty.nrows(), self.sym_penalty.ncols()),
            )
            .field("eigenvalues", &self.eigenvalues)
            .field(
                "eigenvectors",
                &format_args!(
                    "{}×{}",
                    self.eigenvectors.nrows(),
                    self.eigenvectors.ncols()
                ),
            )
            .field("rank", &self.rank)
            .field("nullity", &self.nullity)
            .field("negative_dim", &self.negative_dim)
            .field("tol", &self.tol)
            .field("iszero", &self.iszero)
            .field("op", &self.op.as_ref().map(|o| o.dim()))
            .finish()
    }
}

#[derive(Debug)]
pub struct BasisPsiDerivativeResult {
    pub design_derivative: Array2<f64>,
    pub penalties_derivative: Vec<Array2<f64>>,
    /// Operator-backed design derivative for standalone first-derivative
    /// callers. Bundled first+second callers receive the shared operator on
    /// `BasisPsiDerivativeBundle` instead.
    pub implicit_operator: Option<ImplicitDesignPsiDerivative>,
}

#[derive(Debug)]
pub struct BasisPsiSecondDerivativeResult {
    pub designsecond_derivative: Array2<f64>,
    pub penaltiessecond_derivative: Vec<Array2<f64>>,
    /// Operator-backed design derivative for standalone second-derivative
    /// callers. Bundled first+second callers receive the shared operator on
    /// `BasisPsiDerivativeBundle` instead.
    pub implicit_operator: Option<ImplicitDesignPsiDerivative>,
}

#[derive(Debug)]
pub struct BasisPsiDerivativeBundle {
    pub first: BasisPsiDerivativeResult,
    pub second: BasisPsiSecondDerivativeResult,
    /// Shared operator-backed design derivative for the first and second
    /// psi derivatives. Bundled callers consume this once instead of storing
    /// duplicate materialized/streaming operators in both derivative payloads.
    pub implicit_operator: Option<ImplicitDesignPsiDerivative>,
}

/// Per-axis psi_a derivative package for anisotropic spatial terms.
///
/// For a d-dimensional anisotropic term, the kernel phi(r) depends on
/// the anisotropic distance r = |Lambda h| where Lambda = diag(kappa_a). Each axis a
/// has its own log-scale psi_a = log(kappa_a), yielding d first derivatives,
/// d diagonal second derivatives, and d*(d-1)/2 cross second derivatives.
///
/// The cross second derivative d2 phi/(d psi_a d psi_b) = t * s_a * s_b (a != b)
/// is rank-1, so we store the t_values and s_components vectors rather
/// than materializing d^2 matrices.
#[derive(Clone)]
pub struct AnisoBasisPsiDerivatives {
    /// d matrices, each (n x p_smooth): dX/d psi_a.
    pub design_first: Vec<Array2<f64>>,
    /// d matrices, each (n x p_smooth): d2X/d psi_a^2 (diagonal second derivatives).
    pub design_second_diag: Vec<Array2<f64>>,
    /// Cross second derivatives d2X/(d psi_a d psi_b) for a < b.
    pub design_second_cross: Vec<Array2<f64>>,
    /// Axis-pair indices corresponding to `design_second_cross`.
    pub design_second_cross_pairs: Vec<(usize, usize)>,
    /// d x num_penalties: dS_m/d psi_a for each axis a and penalty m.
    pub penalties_first: Vec<Vec<Array2<f64>>>,
    /// d x num_penalties: d2S_m/d psi_a^2 for each axis a and penalty m.
    pub penalties_second_diag: Vec<Vec<Array2<f64>>>,
    /// The (a, b) axis pairs supported by the on-demand cross-penalty
    /// provider. Only the upper triangle (a < b) is stored.
    pub penalties_cross_pairs: Vec<(usize, usize)>,
    /// On-demand cross-penalty second-derivative provider. Exact anisotropic
    /// cross-axis penalty seconds are streamed one pair at a time rather than
    /// stored as a dense upper triangle of blocks.
    pub penalties_cross_provider: Option<AnisoPenaltyCrossProvider>,
    /// Shared operator-backed representation of the anisotropic kernel-side
    /// design derivatives. When `design_first` / `design_second_diag` are empty,
    /// callers must use this operator directly; when they are present, this
    /// operator still provides exact cross-axis second derivatives without
    /// duplicating separate `t` / `s_a` storage layouts.
    pub implicit_operator: Option<ImplicitDesignPsiDerivative>,
}

#[derive(Clone)]
pub struct AnisoPenaltyCrossProvider(
    std::sync::Arc<
        dyn Fn(usize, usize) -> Result<Vec<Array2<f64>>, BasisError> + Send + Sync + 'static,
    >,
);

impl AnisoPenaltyCrossProvider {
    pub(crate) fn new<F>(f: F) -> Self
    where
        F: Fn(usize, usize) -> Result<Vec<Array2<f64>>, BasisError> + Send + Sync + 'static,
    {
        Self(std::sync::Arc::new(f))
    }

    pub fn evaluate(&self, axis_a: usize, axis_b: usize) -> Result<Vec<Array2<f64>>, BasisError> {
        (self.0)(axis_a, axis_b)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Implicit derivative operator for scalable anisotropic REML gradients
// ═══════════════════════════════════════════════════════════════════════════

pub(crate) const SPATIAL_CENTER_CENTER_MAX_BYTES: usize = 512 * 1024 * 1024; // 512 MiB
pub(crate) const DESIGN_CROSS_CHUNK_SIZE: usize = 1024;

/// Determine whether implicit operators should be used based on problem size
/// and the supplied [`ResourcePolicy`].
///
/// Returns `true` when the dense materialization of D first-derivative
/// matrices would exceed `policy.max_single_materialization_bytes`.
///
/// For D axes with n data points and p_smooth basis columns, the dense path
/// allocates D * n * p_smooth * 8 bytes for first-derivative matrices alone
/// (plus a similar amount for second derivatives). The implicit path stores
/// only the compact (n * n_knots) radial jets plus (n * n_knots * D) axis
/// fractions, which is O(n * k * D) instead of O(n * p * D).
pub fn should_use_implicit_operators_with_policy(
    n: usize,
    p: usize,
    d: usize,
    policy: &gam_runtime::resource::ResourcePolicy,
) -> bool {
    // Each first-derivative matrix is (n x p) f64 → n*p*8 bytes.
    // We need D of them for first derivatives, D for second diag, plus
    // the cross-t matrix and s_components. Conservative estimate: 3*D matrices.
    let dense_bytes = 3usize
        .saturating_mul(n)
        .saturating_mul(p)
        .saturating_mul(d)
        .saturating_mul(8);
    dense_bytes > policy.max_single_materialization_bytes
}

pub(crate) fn implicit_radial_cache_bytes(n: usize, k: usize, n_axes: usize) -> usize {
    n.saturating_mul(k)
        .saturating_mul(n_axes.saturating_add(3))
        .saturating_mul(8)
}

pub(crate) fn should_cache_implicit_radial_components(
    n: usize,
    k: usize,
    n_axes: usize,
    policy: &gam_runtime::resource::ResourcePolicy,
) -> bool {
    implicit_radial_cache_bytes(n, k, n_axes) <= policy.max_operator_cache_bytes
}

pub fn assert_no_dense_derivative_materialization(n: usize, p: usize, d_pc: usize) {
    let first = dense_design_bytes(n, p).saturating_mul(d_pc);
    let second = dense_design_bytes(n, p).saturating_mul(d_pc.saturating_mul(d_pc));
    // Consult the library default ResourcePolicy. Production large-scale runs
    // configure `AnalyticOperatorRequired`, which still refuses every dense
    // materialization here. The default `MaterializeIfSmall` mode lets tiny
    // problems (and small-data/test usage) materialize as long as the combined
    // first- and second-order dense bytes fit under the single-materialization
    // byte budget. `DiagnosticsOnly` is treated like `MaterializeIfSmall` for
    // this guard: it permits dense materialization under the same byte cap.
    let policy = gam_runtime::resource::ResourcePolicy::default_library();
    let budget = policy.max_single_materialization_bytes;
    let needed = first.saturating_add(second);
    match policy.derivative_storage_mode {
        gam_runtime::resource::DerivativeStorageMode::AnalyticOperatorRequired => {
            // SAFETY: this assertion helper exists specifically to enforce
            // the large-scale invariant that spatial-PC Duchon derivative
            // designs never persist as dense `Array2<f64>` storage. When the
            // resource policy is `AnalyticOperatorRequired`, any caller that
            // reached this point has materialized something the strict
            // operator contract forbids.
            // SAFETY: AnalyticOperatorRequired forbids dense derivative materialization.
            panic!(
                "spatial PC Duchon derivative designs must remain operator-backed; refused persistent dense derivative materialization (n={n}, p={p}, d_pc={d_pc}, first_order={:.1} MiB, second_order={:.1} MiB)",
                first as f64 / (1024.0 * 1024.0),
                second as f64 / (1024.0 * 1024.0),
            );
        }
        gam_runtime::resource::DerivativeStorageMode::MaterializeIfSmall
        | gam_runtime::resource::DerivativeStorageMode::DiagnosticsOnly => {
            // SAFETY: exceeding the single-materialization budget here is a
            // contract violation by an upstream caller that must route through
            // the operator-backed path; failing loudly surfaces it rather than
            // silently materializing an oversized dense derivative design.
            assert!(
                needed <= budget,
                "spatial PC Duchon derivative designs would exceed the single-materialization budget; refused persistent dense derivative materialization (n={n}, p={p}, d_pc={d_pc}, first_order={:.1} MiB, second_order={:.1} MiB, budget={:.1} MiB)",
                first as f64 / (1024.0 * 1024.0),
                second as f64 / (1024.0 * 1024.0),
                budget as f64 / (1024.0 * 1024.0),
            );
        }
    }
}

pub fn assert_spatial_centers_below_large_scale_cap(
    d_pc: usize,
    centers: ArrayView2<'_, f64>,
) -> Result<(), BasisError> {
    if centers.ncols() != d_pc {
        crate::bail_dim_basis!(
            "spatial PC center dimension mismatch: centers have {} columns, expected {d_pc}",
            centers.ncols()
        );
    }
    let k = centers.nrows();
    let centers_bytes = dense_design_bytes(k, d_pc);
    let center_center_bytes = dense_design_bytes(k, k);
    if centers_bytes > SPATIAL_CENTER_CENTER_MAX_BYTES {
        crate::bail_invalid_basis!(
            "spatial PC centers exceed center storage cap: K={k}, d_pc={d_pc}, centers={:.1} MiB, cap={:.1} MiB",
            centers_bytes as f64 / (1024.0 * 1024.0),
            SPATIAL_CENTER_CENTER_MAX_BYTES as f64 / (1024.0 * 1024.0),
        );
    }
    if center_center_bytes > SPATIAL_CENTER_CENTER_MAX_BYTES {
        crate::bail_invalid_basis!(
            "spatial PC centers exceed center-center large-scale cap: K={k}, d_pc={d_pc}, KxK={:.1} MiB, cap={:.1} MiB",
            center_center_bytes as f64 / (1024.0 * 1024.0),
            SPATIAL_CENTER_CENTER_MAX_BYTES as f64 / (1024.0 * 1024.0),
        );
    }
    Ok(())
}

pub(crate) fn dense_design_bytes(n: usize, p: usize) -> usize {
    n.saturating_mul(p)
        .saturating_mul(std::mem::size_of::<f64>())
}

pub(crate) fn should_use_lazy_spatial_design(
    n: usize,
    p: usize,
    policy: &gam_runtime::resource::ResourcePolicy,
) -> bool {
    dense_design_bytes(n, p) > policy.max_single_materialization_bytes
}

pub(crate) fn wrap_dense_design_with_transform(
    design: DesignMatrix,
    transform: &Array2<f64>,
    label: &str,
) -> Result<DesignMatrix, BasisError> {
    match design {
        DesignMatrix::Dense(inner) => {
            let op = CoefficientTransformOperator::new(inner, transform.clone()).map_err(|e| {
                BasisError::InvalidInput(format!("{label} coefficient transform failed: {e}"))
            })?;
            Ok(DesignMatrix::Dense(
                gam_linalg::matrix::DenseDesignMatrix::from(Arc::new(op)),
            ))
        }
        DesignMatrix::Sparse(_) => Err(BasisError::InvalidInput(format!(
            "{label} coefficient transform requires a dense/operator-backed design"
        ))),
    }
}

/// Single-pass `(Bᵀ(W·C), BᵀB)` accumulation over the streamed design.
///
/// Materialises each row chunk of the design **once** and reuses it for both
/// the constraint cross `Bᵀ(W·C)` and the Gram `BᵀB`. On the lazy chunked
/// spatial path each `try_row_chunk` re-evaluates all kernel columns for the
/// chunk, so accumulating both products in a single sweep halves the per-build
/// kernel re-evaluation work (the dominant cost at large scale) versus two
/// independent streaming passes — without changing the result beyond
/// floating-point reassociation. The cross is masked off (`q == 0`) by the
/// caller, which never invokes this when there is no constraint block.
pub(crate) fn design_cross_and_gram(
    design: &DesignMatrix,
    constraint_matrix: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<(Array2<f64>, Array2<f64>), BasisError> {
    let n = design.nrows();
    let k = design.ncols();
    if constraint_matrix.nrows() != n {
        return Err(BasisError::ConstraintMatrixRowMismatch {
            basisrows: n,
            constraintrows: constraint_matrix.nrows(),
        });
    }
    if let Some(w) = weights
        && w.len() != n
    {
        return Err(BasisError::WeightsDimensionMismatch {
            expected: n,
            found: w.len(),
        });
    }
    let q = constraint_matrix.ncols();
    let mut cross = Array2::<f64>::zeros((k, q));
    let mut gram = Array2::<f64>::zeros((k, k));
    for start in (0..n).step_by(DESIGN_CROSS_CHUNK_SIZE) {
        let end = (start + DESIGN_CROSS_CHUNK_SIZE).min(n);
        let basis_chunk = design
            .try_row_chunk(start..end)
            .map_err(|e| BasisError::InvalidInput(e.to_string()))?;
        let mut constraint_chunk = constraint_matrix.slice(s![start..end, ..]).to_owned();
        if let Some(w) = weights {
            for (mut row, &weight) in constraint_chunk
                .axis_iter_mut(Axis(0))
                .zip(w.slice(s![start..end]).iter())
            {
                row *= weight;
            }
        }
        cross += &fast_atb(&basis_chunk, &constraint_chunk);
        gram += &fast_atb(&basis_chunk, &basis_chunk);
    }
    Ok((cross, gram))
}

pub(crate) fn positive_spectral_whitener_from_gram(
    gram: &Array2<f64>,
) -> Result<Array2<f64>, BasisError> {
    // Inverse-square-root for the positive part of `gram`. Eigenvalues at or
    // below the relative rank tolerance `α·ε·n·max_eval` are *dropped*: the
    // returned whitener has shape `(n × keep)` where `keep` counts strictly
    // positive eigendirections of `gram`.
    //
    // Dropping (rather than ridging) is what makes the result a true
    // square-root inverse on the column space of `gram`. This whitener is
    // used by `stabilized_orthogonality_transform_from_gram` to make a
    // pre-existing transform `K_raw` orthonormal under the W-inner product:
    // when some columns of `K_raw` map to zero (or near-zero) under `B`, the
    // constrained Gram `K_raw^T G K_raw` is rank-deficient. Ridging those
    // tail directions with `1/sqrt(ε)` produced spurious basis columns
    // whose coefficient norms blew up to `~1/sqrt(ε)` while their image in
    // `B` was floating-point zero, contaminating downstream linear algebra
    // (in particular it forced `smooth.rs` to widen the post-transform
    // orthogonality residual tolerance to absorb a `cond ≈ 1/sqrt(ε)`
    // rounding floor). Dropping these directions is the right behavior:
    // they contribute nothing to `B`'s column space, and removing them
    // tightens the orthogonality residual back down to the genuine
    // floating-point limit.
    let (eigenvalues, eigenvectors) = gram.eigh(Side::Lower).map_err(BasisError::LinalgError)?;
    let n = gram.nrows();
    let max_eval = eigenvalues.iter().copied().fold(0.0_f64, f64::max);
    // Scale-invariant rank tolerance: the cutoff must track the Gram's own
    // spectrum (`α·ε·n·max_eval`), not an absolute floor. An earlier `max_eval
    // .max(1.0)` clamped the reference scale to 1.0, which is only harmless when
    // `max_eval ≥ 1`; for a genuinely well-conditioned but small-magnitude Gram
    // (e.g. a Duchon hybrid whose evaluated kernel sits far below unit scale in
    // moderate-to-high d) it inflated the tolerance to an absolute `α·ε·n` floor
    // that swallows the entire — perfectly valid — spectrum, spuriously reporting
    // `keep == 0`. Using the true `max_eval` makes `keep` invariant to a uniform
    // rescaling of the Gram (which scales every eigenvalue and the cutoff
    // identically). The residual `.max(f64::EPSILON)` only guards the degenerate
    // all-zero Gram so that numerical-zero roundoff directions are still dropped.
    let tol =
        (default_rrqr_rank_alpha() * f64::EPSILON * (n.max(1) as f64) * max_eval).max(f64::EPSILON);
    let keep = eigenvalues.iter().filter(|&&ev| ev > tol).count();
    if keep == 0 {
        let min_ev = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
        return Err(BasisError::ConstraintNullspaceCollapsed {
            site: "positive_spectral_whitener_from_gram",
            cross_rank: 0,
            coeff_dim: gram.nrows(),
            cross_frobenius: gram.iter().map(|v| v * v).sum::<f64>().sqrt(),
            gram_spectrum: format!(
                "max eigenvalue {max_eval:.3e} (min {min_ev:.3e}, spectral tolerance {tol:.3e})"
            ),
        });
    }
    // `eigh` returns eigenvalues in ascending order, so the largest `keep`
    // eigenvalues live at the tail.
    let eig_start = eigenvalues.len() - keep;
    let kept_vectors = eigenvectors.slice(s![.., eig_start..]).to_owned();
    let mut inv_sqrt = Array2::<f64>::zeros((keep, keep));
    for (out_i, eig_i) in (eig_start..eigenvalues.len()).enumerate() {
        inv_sqrt[[out_i, out_i]] = 1.0 / eigenvalues[eig_i].sqrt();
    }
    Ok(fast_ab(&kept_vectors, &inv_sqrt))
}

pub(crate) fn stabilized_orthogonality_transform_from_gram(
    gram: &Array2<f64>,
    transform: &Array2<f64>,
) -> Result<Array2<f64>, BasisError> {
    let constrained_gram = {
        let gt = fast_ab(gram, transform);
        fast_atb(transform, &gt)
    };
    let whitening = positive_spectral_whitener_from_gram(&constrained_gram)?;
    Ok(fast_ab(transform, &whitening))
}

pub(crate) fn orthogonality_transform_from_cross_and_gram(
    constraint_cross: &Array2<f64>,
    gram: &Array2<f64>,
) -> Result<Array2<f64>, BasisError> {
    // Compute null(M^T) directly on M = B^T W C (k × q) via column-pivoted QR.
    // Working in the original k-dim coefficient space rather than first
    // whitening by B^T B avoids a fundamental failure mode: when B is heavily
    // collinear, `positive_spectral_whitener_from_gram` truncates the design
    // column-space to a `keep`-dim subspace, and if `keep <= q` the subsequent
    // nullspace search has no room — even though dim null(M^T) = k - rank(M)
    // ≥ k - q is always positive when k > q. The constraint nullspace is a
    // property of M alone; conditioning of the design only matters for the
    // downstream stabilization of B*K_raw.
    let k = constraint_cross.nrows();
    if k == 0 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: 0 });
    }
    let (transform_raw, rank) = rrqr_nullspace_basis(constraint_cross, default_rrqr_rank_alpha())
        .map_err(BasisError::LinalgError)?;
    if rank >= k || transform_raw.ncols() == 0 {
        return Err(BasisError::ConstraintNullspaceCollapsed {
            site: "orthogonality_transform_from_cross_and_gram",
            cross_rank: rank,
            coeff_dim: k,
            cross_frobenius: constraint_cross.iter().map(|v| v * v).sum::<f64>().sqrt(),
            gram_spectrum: "not computed (structural cross-rank collapse: null(Mᵀ) is empty, \
                            so no constrained design exists to eigendecompose)"
                .to_string(),
        });
    }

    // Make the constrained design B*K_raw orthonormal under the W-inner product.
    // If the constrained Gram K_raw^T G K_raw is rank-deficient (because some
    // directions in null(M^T) collapse under B), the spectral whitener drops
    // them — that is the right behavior: a degenerate column never contributes
    // to B's column space and shouldn't appear in the reparameterized basis.
    stabilized_orthogonality_transform_from_gram(gram, &transform_raw)
}

pub fn orthogonality_transform_for_design(
    design: &DesignMatrix,
    constraint_matrix: ArrayView2<'_, f64>,
    weights: Option<ArrayView1<'_, f64>>,
) -> Result<Array2<f64>, BasisError> {
    let k = design.ncols();
    if k == 0 {
        return Err(BasisError::InsufficientColumnsForConstraint { found: 0 });
    }
    let q = constraint_matrix.ncols();
    if q == 0 {
        return Ok(Array2::eye(k));
    }
    let (constraint_cross, gram) = design_cross_and_gram(design, constraint_matrix, weights)?;
    orthogonality_transform_from_cross_and_gram(&constraint_cross, &gram)
}
