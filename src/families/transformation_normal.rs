//! Conditional transformation model: estimate h(y|x) such that h(Y|x) ~ N(0,1).
//!
//! Given a response variable y and covariates x with a pre-built covariate design
//! operator, this family estimates a smooth monotone transformation h(y | x) mapping
//! the conditional distribution of Y|x onto a standard normal.
//!
//! The response-direction basis is `[1, I_1(y), ..., I_K(y)]`, tensored with an
//! arbitrary covariate design operator. Column 0 is an unconstrained location
//! component `b(x)`. The I-spline columns are shape components with squared
//! covariate-side coefficients, giving the SCOP representation
//! `h(y, x) = b(x) + ε·(y−median_y) + Σ_k I_k(y) γ_k(x)^2` and
//! `h'(y, x) = ε + Σ_k M_k(y) γ_k(x)^2`. Monotonicity is structural:
//! the fixed derivative floor `ε` keeps the change-of-variables log-density
//! away from the `log(0)` singularity, while the non-negative M-spline basis
//! and squared covariate-side coefficients supply the learned shape.
//!
//! The log-likelihood per observation is the finite-support normalized
//! change-of-variables density for a standard normal target:
//!
//!   ℓ_i = -½ h_i² + log(h'_i) - log(Φ(h_U(x_i)) - Φ(h_L(x_i)))
//!
//! where `h_i = b(x_i) + ε·(y_i−median_y) + Σ_k I_k(y_i) γ_k(x_i)^2`
//! and `h'_i = ε + Σ_k M_k(y_i) γ_k(x_i)^2`. The endpoint normalizer is
//! required because the I-spline response basis saturates at finite support
//! values rather than mapping onto the full real line.

use crate::basis::{
    BasisOptions, Dense, KnotSource, create_basis, create_difference_penalty_matrix,
    create_ispline_derivative_dense,
};
use crate::faer_ndarray::{fast_ab, fast_abt, fast_atb};
use crate::families::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative,
    CustomFamilyPsiDerivativeOperator, CustomFamilyWarmStart, ExactNewtonJointGradientEvaluation,
    ExactNewtonJointHessianWorkspace, ExactNewtonJointPsiSecondOrderTerms,
    ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace, FamilyEvaluation,
    MaterializablePsiDerivativeOperator, ParameterBlockSpec, ParameterBlockState, PenaltyMatrix,
    build_block_spatial_psi_derivatives, evaluate_custom_family_joint_hyper,
    evaluate_custom_family_joint_hyper_efs, fit_custom_family, fit_custom_family_fixed_log_lambdas,
};
use crate::families::gamlss::{
    initializewiggle_knots_from_seed, solve_penalizedweighted_projection,
};
use crate::inference::model::{TRANSFORMATION_SCORE_PIT_CLIP_EPS, TransformationScoreCalibration};
use crate::matrix::{
    DenseDesignMatrix, DenseDesignOperator, DesignMatrix, LinearOperator, SymmetricMatrix,
};
use crate::pirls::LinearInequalityConstraints;
use crate::probability::{log1mexp_positive, normal_logcdf, standard_normal_quantile};
use crate::resource::{MatrixMaterializationError, ResourcePolicy};
use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design, optimize_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices,
};
use crate::solver::estimate::UnifiedFitResult;
use crate::solver::estimate::reml::unified::{
    DriftDerivResult, HyperOperator, ProjectedFactorCache, ProjectedFactorKey,
};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, s};
use std::cell::RefCell;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::sync::{Arc, Mutex, OnceLock};

// ---------------------------------------------------------------------------
// Typed errors
// ---------------------------------------------------------------------------

/// Typed errors emitted by the transformation-normal family pipeline.
///
/// Each variant carries a pre-formatted `reason` so `Display` is
/// byte-equivalent to the original `format!(...)` strings the module used
/// before the typed-error migration. The category split lets callers
/// pattern-match on the failure kind (e.g. distinguish a degenerate
/// covariate design from a non-finite intermediate) without parsing text.
///
/// Public/trait boundaries (e.g. `CustomFamily::evaluate`) still return
/// `Result<_, String>`; the `From<TransformationNormalError> for String`
/// impl below provides the shim so every typed error flushes through `?`
/// or `.into()` at the boundary without per-callsite `.map_err`.
#[derive(Debug, Clone)]
pub enum TransformationNormalError {
    /// Shape/length/dimension/contract violations on inputs to a routine
    /// (e.g. response/covariate row mismatch, beta length mismatch,
    /// wrong number of blocks, malformed configuration parameters).
    InvalidInput { reason: String },
    /// A required covariate design or weight configuration cannot support
    /// the routine — empty design, zero total weight, residual variance
    /// not representable, warm-start coefficients all non-finite.
    DesignDegenerate { reason: String },
    /// A numeric intermediate (response transform, derivative,
    /// log-likelihood, weight, offset, gradient component, calibration
    /// quantity) came out non-finite or non-positive where positive
    /// finite is required.
    NonFinite { reason: String },
    /// The fitted monotone transform's derivative dropped to or below
    /// zero, or the response endpoint ordering required by the latent
    /// score (lower < h < upper) was not satisfied at evaluation time.
    MonotonicityViolated { reason: String },
    /// A numerical step that maps through the standard-normal CDF
    /// (endpoint mass, log-difference, PIT probability, derivative
    /// ratio) underflowed or became non-representable at the requested
    /// arguments.
    NumericalFailure { reason: String },
}

impl std::fmt::Display for TransformationNormalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TransformationNormalError::InvalidInput { reason }
            | TransformationNormalError::DesignDegenerate { reason }
            | TransformationNormalError::NonFinite { reason }
            | TransformationNormalError::MonotonicityViolated { reason }
            | TransformationNormalError::NumericalFailure { reason } => f.write_str(reason),
        }
    }
}

impl std::error::Error for TransformationNormalError {}

impl From<TransformationNormalError> for String {
    /// Shim for the many `Result<_, String>` signatures the module exposes
    /// (notably the `CustomFamily` and joint-Hessian / psi-workspace
    /// trait surfaces). Lets a typed `Err(TransformationNormalError::…)`
    /// flow through `?` or `.into()` without per-callsite stringification.
    fn from(err: TransformationNormalError) -> String {
        err.to_string()
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the response-direction basis in the transformation model.
#[derive(Clone, Debug)]
pub struct TransformationNormalConfig {
    /// B-spline degree for the response-direction deviation basis (default 3).
    pub response_degree: usize,
    /// Number of interior knots for the response-direction deviation basis (default 10).
    pub response_num_internal_knots: usize,
    /// Difference penalty order for the response-direction roughness penalty (default 2).
    pub response_penalty_order: usize,
    /// Additional penalty orders for the response-direction (default [1]).
    pub response_extra_penalty_orders: Vec<usize>,
    /// Whether to add a global identity (ridge) penalty (default true).
    pub double_penalty: bool,
}

impl Default for TransformationNormalConfig {
    fn default() -> Self {
        Self {
            response_degree: 3,
            response_num_internal_knots: 10,
            response_penalty_order: 2,
            response_extra_penalty_orders: vec![1],
            double_penalty: true,
        }
    }
}

/// Baseline cap for the tensor-product width used by the transformation-normal
/// response basis. Small datasets should stay compact because the fit
/// repeatedly factorizes dense penalized Hessians.
const BASE_TRANSFORMATION_TENSOR_WIDTH: usize = 160;
/// Large samples can support a richer response basis without the aggressive
/// underfitting forced by the small-sample cap above. This upper cap keeps the
/// tensor width bounded even when the covariate side is narrow.
const LARGE_SAMPLE_TRANSFORMATION_TENSOR_WIDTH: usize = 320;
/// E[log |Z|] for Z ~ N(0, 1), used to put local log-absolute residual
/// projections on the standard-normal scale.
const STANDARD_NORMAL_MEAN_LOG_ABS: f64 = -0.635_181_422_730_739_1;
/// Strict-feasibility margin for `h' > 0` on the monotonicity grid. Used
/// both by the fit-time fraction-to-boundary line search (so accepted β
/// keeps `h'(grid) ≥ EPS`) and by the predict-time monotonicity check
/// in `inference::predict_input` (which rejects predictions whose minimum
/// `h'` on the response grid drops below this threshold). Keeping these
/// in sync prevents the predict path from rejecting fits that the
/// optimizer accepted as feasible — and vice versa.
pub const TRANSFORMATION_MONOTONICITY_EPS: f64 = 1.0e-8;
/// Absolute bound for feasible transformation scores on the standard-normal
/// scale. The CTN likelihood targets `h(Y|x) ~ N(0,1)`; accepting exact-Newton
/// iterates with finite positive `h'` but astronomical `|h|` lets curvature
/// diagnostics overflow into meaningless values. This is a numerical runaway
/// guard, not a statistical plausibility filter: startup seeds can temporarily
/// land outside practically observable normal quantiles before the line search
/// moves them back into the likelihood's high-density region.
pub const TRANSFORMATION_NORMAL_H_ABS_MAX: f64 = 1.0e6;
/// Number of dense-spectral factor columns processed per exact ψψ HVP row pass.
/// At biobank CTN dimensions p≈800, this keeps the per-worker accumulator well
/// under 1 MiB while reducing repeated SCOP row-invariant work by 32× relative
/// to one-column HVP dispatch.
const SCOP_PSI_PSI_HVP_TILE_COLS: usize = 32;
/// Exact dense SCOP coefficient Hessian cache limit for the inner `H·v` path.
///
/// The biobank CTN calibration fit has many rows but a moderate coefficient
/// dimension (for example n=20k, p=264). In that regime repeated PCG products
/// against the same Hessian should pay the row-streaming chain rule once, then
/// serve subsequent products as dense BLAS matvecs. Keep the cache restricted to
/// genuinely moderate p so wide CTN fits remain row-streamed.
const SCOP_HESSIAN_HVP_DENSE_CACHE_MAX_DIM: usize = 384;
const SCOP_HESSIAN_HVP_DENSE_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;

fn beta_bits_match(cached: &Array1<f64>, candidate: &Array1<f64>) -> bool {
    cached.len() == candidate.len()
        && cached
            .iter()
            .zip(candidate.iter())
            .all(|(&left, &right)| left.to_bits() == right.to_bits())
}

/// Optional warm-start for the transformation model: per-observation location and
/// scale values from a prior mean/SD normalizer.
#[derive(Clone, Debug)]
pub struct TransformationWarmStart {
    /// μ(x_i): conditional mean of the response at each observation's covariates.
    pub location: Array1<f64>,
    /// τ(x_i): conditional standard deviation at each observation's covariates.
    pub scale: Array1<f64>,
}

// ---------------------------------------------------------------------------
// Persistent dense-Hessian cache (P2.1)
// ---------------------------------------------------------------------------

/// Cache key for the SCOP-CTN dense joint Hessian.
///
/// The Hessian depends on `(β, row_quantities(β))`; row_quantities is keyed
/// on β by exact f64 bits, so a `(beta_hash, row_quantities_version)` pair
/// uniquely identifies a build. We hash β bits (not values) so the key is
/// bit-exact and avoids f64-Hash issues.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CtnDenseHessianKey {
    pub beta_hash: u64,
    pub row_quantities_version: u64,
    /// Outer-score subsample identity tag. `None` for full-data builds;
    /// `Some(hash)` for subsampled fits. Including this in the key prevents
    /// a subsampled Hessian (which is the HT estimator at this β) from
    /// aliasing a later full-data probe at the same β.
    pub outer_subsample_hash: Option<u64>,
}

impl CtnDenseHessianKey {
    fn from(
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        outer_subsample_hash: Option<u64>,
    ) -> Self {
        let mut hasher = DefaultHasher::new();
        beta.len().hash(&mut hasher);
        for value in beta.iter() {
            value.to_bits().hash(&mut hasher);
        }
        let beta_hash = hasher.finish();
        CtnDenseHessianKey {
            beta_hash,
            row_quantities_version: row_quantities.version,
            outer_subsample_hash,
        }
    }
}

/// Process-local persistent dense-Hessian cache shared across workspace
/// re-creations.
///
/// **Single-slot, not LRU.** The access pattern inside the SCOP joint Newton
/// inner solve is "many HVP probes share the same β within one trust-region
/// step; β advances between steps." A 1-entry slot therefore hits exactly
/// when it should (consecutive probes at the same β) and misses on every β
/// advance — no eviction policy is needed. Without this persistent slot a
/// fresh `TransformationNormalJointHessianWorkspace` is built every time the
/// outer evaluator calls `exact_newton_joint_hessian_workspace`, and its
/// inner `OnceLock` only amortizes within a single workspace lifetime; that
/// is the source of the ~310× CTN dense Hessian rebuild storm at biobank
/// scale.
#[derive(Default)]
pub(crate) struct CtnPersistentDenseHessianCache {
    slot: Mutex<Option<(CtnDenseHessianKey, Arc<Array2<f64>>)>>,
}

impl CtnPersistentDenseHessianCache {
    fn get(&self, key: &CtnDenseHessianKey) -> Option<Arc<Array2<f64>>> {
        let slot = self
            .slot
            .lock()
            .expect("CTN persistent dense Hessian cache mutex poisoned");
        slot.as_ref().and_then(|(cached_key, cached)| {
            if cached_key == key {
                Some(Arc::clone(cached))
            } else {
                None
            }
        })
    }

    fn install(&self, key: CtnDenseHessianKey, hessian: Arc<Array2<f64>>) {
        let mut slot = self
            .slot
            .lock()
            .expect("CTN persistent dense Hessian cache mutex poisoned");
        *slot = Some((key, hessian));
    }
}

// ---------------------------------------------------------------------------
// The family
// ---------------------------------------------------------------------------

/// Conditional transformation model mapping Y|x to N(0,1).
///
/// Single-block `CustomFamily`. The block design is `x_val` (tensor product of
/// response value basis × covariate design). The family internally holds `x_deriv`
/// (tensor product of response derivative basis × covariate design) for the
/// Jacobian term in the likelihood.
#[derive(Clone)]
pub struct TransformationNormalFamily {
    // --- Tensor product design matrices ---
    /// Value design operator: keeps the tensor factors separate and materializes
    /// only row chunks or explicitly requested dense diagnostics.
    x_val_kron: KroneckerDesign,
    /// Derivative design operator: keeps the tensor factors separate.
    x_deriv_kron: KroneckerDesign,
    // --- Response-direction basis (fixed, does not depend on κ) ---
    /// Response value basis: n × p_resp. Columns: [1, I_1(y), ..., I_k(y)].
    response_val_basis: Array2<f64>,
    /// Response value basis at the finite lower support endpoint.
    response_lower_basis: Array1<f64>,
    /// Response value basis at the finite upper support endpoint.
    response_upper_basis: Array1<f64>,
    /// Response derivative basis: n × p_resp. Columns: [0, M_1(y), ..., M_k(y)].
    response_deriv_basis: Array2<f64>,

    // --- Covariate side (rebuilt on κ change) ---
    /// Original covariate design used on the right side of the tensor product.
    covariate_design: DesignMatrix,
    /// Dense covariate block shared by row-quantity and endpoint evaluations.
    ///
    /// CTN row quantities are rebuilt at every accepted/probed β, but the
    /// covariate design is fixed for the family. Caching this immutable
    /// `n × p_cov` block avoids repeated chunk materialization and keeps
    /// biobank-scale runs from churning large transient allocations.
    covariate_dense_cache: Arc<Mutex<Option<Arc<Array2<f64>>>>>,
    /// Optional non-negative row weights folded directly into the likelihood.
    weights: Arc<Array1<f64>>,
    /// Additive offset for the transformation linear predictor.
    offset: Arc<Array1<f64>>,
    // --- Tensor penalties ---
    tensor_penalties: Vec<PenaltyMatrix>,

    // --- Initial values ---
    initial_beta: Array1<f64>,
    initial_log_lambdas: Array1<f64>,

    // --- Config ---
    block_name: String,

    // --- Response basis metadata (for reconstruction at predict time) ---
    response_knots: Array1<f64>,
    response_transform: Array2<f64>,
    response_degree: usize,
    response_median: f64,
    response_floor_offset: Arc<Array1<f64>>,
    response_lower_floor_offset: f64,
    response_upper_floor_offset: f64,

    /// Last row-space transformation quantities for an exact beta vector.
    ///
    /// CTN line searches and exact-Newton workspace construction frequently ask
    /// for likelihood, gradient, and Hessian row factors at the same candidate
    /// coefficients. This cache keeps the expensive Khatri-Rao forward products
    /// and reciprocal powers behind a single exact-keyed entry instead of
    /// recomputing `h`, `h'`, `1/h'`, and derivative powers per call.
    row_quantity_cache: Arc<Mutex<Option<TransformationNormalRowQuantityCache>>>,
    /// Monotonic counter bumped each time a fresh row_quantity build is
    /// installed. The value tags every freshly built
    /// `TransformationNormalRowQuantityCache.version` so downstream caches
    /// (the persistent dense-Hessian slot) can key on
    /// `(beta_bits, row_quantities_version)` without re-hashing every row
    /// quantity. Atomic so the counter survives `Family::clone` (the family
    /// clones its `Arc<...>` interior state — version counts must be shared
    /// across the same logical family instance).
    row_quantity_version: Arc<AtomicU64>,
    /// Persistent dense Hessian cache (P2.1). Survives workspace
    /// re-creation; keyed on the exact `(β bits, row_quantities version)`
    /// pair. See [`CtnPersistentDenseHessianCache`] for the access-pattern
    /// rationale (single-slot, not LRU).
    persistent_dense_hessian: Arc<CtnPersistentDenseHessianCache>,
    /// Optional outer-score Horvitz-Thompson per-row weights.
    ///
    /// When present, this is an `n`-vector equal to the original `weights`
    /// pre-multiplied row-wise by the HT inverse-inclusion multiplier `m_i`
    /// (`m_i = 1/π_i` on sampled rows, `0.0` on unsampled rows). Assembly
    /// sites read row weights via [`Self::effective_weights`], which returns
    /// this array when present and `self.weights` otherwise. Because every
    /// per-row CTN contribution is linear in `w_i`, masking at this site
    /// gives `E[Σ_i (m_i · w_i) · f(row_i)] = Σ_i w_i · f(row_i) = full-sum`
    /// — i.e. an unbiased estimator across log-likelihood, gradient, joint
    /// Hessian (dense / matvec / diagonal), ψ, and ψ-ψ kernels.
    ///
    /// `None` preserves byte-identical legacy behavior (`effective_weights`
    /// returns the original `weights` array).
    outer_subsample_weights: Option<Arc<Array1<f64>>>,
    /// Subsample-identity tag used to key the row-quantity and persistent
    /// dense-Hessian caches when an outer-score subsample is active. `None`
    /// for the full-data family; `Some(hash)` for subsampled clones. The hash
    /// is over the HT mask + per-row weight bits so that two subsamples with
    /// the same β never alias each other's caches.
    outer_subsample_hash: Option<u64>,
}

#[derive(Clone)]
struct TransformationNormalRowQuantityCache {
    beta: Arc<Array1<f64>>,
    gamma: Arc<Array2<f64>>,
    h: Arc<Array1<f64>>,
    h_prime: Arc<Array1<f64>>,
    h_lower: Arc<Array1<f64>>,
    h_upper: Arc<Array1<f64>>,
    endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    log_likelihood: f64,
    /// Monotonic version tag set at construction time. Used by the
    /// persistent dense-Hessian cache key.
    version: u64,
}

#[derive(Debug)]
struct TransformationNormalRowDerived {
    log_likelihood: f64,
    endpoint_q: Vec<LogNormalCdfDiffDerivatives>,
}

impl TransformationNormalRowQuantityCache {
    fn matches_beta(&self, beta: &Array1<f64>) -> bool {
        beta_bits_match(&self.beta, beta)
    }
}

fn build_transformation_row_derived(
    h: &Array1<f64>,
    h_prime: &Array1<f64>,
    h_lower: &Array1<f64>,
    h_upper: &Array1<f64>,
    weights: &Array1<f64>,
) -> Result<TransformationNormalRowDerived, String> {
    let n = h_prime.len();
    debug_assert_eq!(h.len(), n);
    debug_assert_eq!(h_lower.len(), n);
    debug_assert_eq!(h_upper.len(), n);
    debug_assert_eq!(weights.len(), n);

    if let Some((i, value)) = h
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(TransformationNormalError::NonFinite {
            reason: format!(
                "TransformationNormalFamily row_quantities: h[{i}] = {value} is not finite"
            ),
        }
        .into());
    }
    if let Some((i, value)) = weights
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite())
    {
        return Err(TransformationNormalError::NonFinite {
            reason: format!(
                "TransformationNormalFamily row_quantities: weight[{i}] = {value} is not finite"
            ),
        }
        .into());
    }

    // Parallelize the per-row endpoint-normalizer build: each row runs
    // `log_normal_cdf_diff_derivatives` (two `normal_logcdf` calls, three
    // 5x5 truncated polynomial multiplies, 32 `signed_normal_pdf_ratio`
    // calls) which dominates this function's runtime at biobank scale.
    // Rows are fully independent — no shared state, no OnceLock guards —
    // and `LogNormalCdfDiffDerivatives` is a POD struct that's `Send`.
    // The fast finiteness check rolls all eight derived quantities into
    // a single short-circuit `||` chain so the named-field error format
    // only runs on the non-finite slow path.
    use rayon::iter::{IntoParallelIterator, ParallelIterator};
    let rows: Vec<(f64, LogNormalCdfDiffDerivatives)> = (0..n)
        .into_par_iter()
        .map(|i| -> Result<(f64, LogNormalCdfDiffDerivatives), String> {
            let hp = h_prime[i];
            let inv_h_prime = 1.0 / hp;
            let inv_h_prime_sq = inv_h_prime * inv_h_prime;
            let inv_h_prime_cu = inv_h_prime_sq * inv_h_prime;
            let inv_h_prime_qu = inv_h_prime_sq * inv_h_prime_sq;
            let w_i = weights[i];
            let h_i = h[i];
            let weighted_h = w_i * h_i;
            let weighted_inv_h_prime = w_i * inv_h_prime;
            let weighted_inv_h_prime_sq = w_i * inv_h_prime_sq;
            let q = log_normal_cdf_diff_derivatives(h_upper[i], h_lower[i]).map_err(|e| {
                format!("TransformationNormalFamily row_quantities: row {i} invalid endpoint normalizer: {e}")
            })?;
            let log_z = q.log_z;
            let row_ll = w_i * (-0.5 * h_i * h_i + hp.ln() - log_z);
            // Fast path: a single short-circuited finiteness check. Only
            // when something is non-finite do we walk the named-field
            // table to produce a precise diagnostic.
            if !(inv_h_prime.is_finite()
                && inv_h_prime_sq.is_finite()
                && inv_h_prime_cu.is_finite()
                && inv_h_prime_qu.is_finite()
                && weighted_h.is_finite()
                && weighted_inv_h_prime.is_finite()
                && weighted_inv_h_prime_sq.is_finite()
                && log_z.is_finite())
            {
                let derived_values = [
                    ("1/h'", inv_h_prime),
                    ("1/h'^2", inv_h_prime_sq),
                    ("1/h'^3", inv_h_prime_cu),
                    ("1/h'^4", inv_h_prime_qu),
                    ("w*h", weighted_h),
                    ("w/h'", weighted_inv_h_prime),
                    ("w/h'^2", weighted_inv_h_prime_sq),
                    ("log normalizer", log_z),
                ];
                for (name, value) in derived_values {
                    if !value.is_finite() {
                        return Err(TransformationNormalError::NonFinite { reason: format!(
                            "TransformationNormalFamily row_quantities: {name} at row {i} is not finite ({value}); h'={hp} is outside the finite exact-derivative range",
                        ) }.into());
                    }
                }
                return Err(TransformationNormalError::NonFinite { reason: format!(
                    "TransformationNormalFamily row_quantities: row {i} entered non-finite branch but no named field was non-finite; h'={hp}",
                ) }.into());
            }
            Ok((row_ll, q))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Sum row contributions in index order so the result is bit-identical
    // to the previous serial accumulation. The parallel section above only
    // parallelized the independent per-row computation; the final scalar
    // reduction stays serial to preserve numerical reproducibility against
    // existing tests.
    let mut log_likelihood = 0.0;
    let mut endpoint_q = Vec::with_capacity(n);
    for (row_ll, q) in rows {
        log_likelihood += row_ll;
        endpoint_q.push(q);
    }
    if !log_likelihood.is_finite() {
        return Err(TransformationNormalError::NonFinite { reason: format!(
            "TransformationNormalFamily row_quantities: log-likelihood is not finite ({log_likelihood})"
        ) }.into());
    }

    Ok(TransformationNormalRowDerived {
        log_likelihood,
        endpoint_q,
    })
}

fn log_normal_cdf_diff(upper: f64, lower: f64) -> Result<f64, String> {
    if !(upper.is_finite() && lower.is_finite()) {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!("finite support endpoints required, got lower={lower}, upper={upper}"),
        }
        .into());
    }
    if upper <= lower {
        return Err(TransformationNormalError::MonotonicityViolated { reason: format!(
            "upper endpoint score must exceed lower endpoint score, got lower={lower:.6e}, upper={upper:.6e}"
        ) }.into());
    }
    if lower > 0.0 {
        return log_normal_cdf_diff(-lower, -upper);
    }
    let log_upper = normal_logcdf(upper);
    let log_lower = normal_logcdf(lower);
    let gap = log_upper - log_lower;
    if !(gap.is_finite() && gap > 0.0) {
        return Err(TransformationNormalError::NumericalFailure { reason: format!(
            "normal CDF endpoint mass is not representable, lower={lower:.6e}, upper={upper:.6e}"
        ) }.into());
    }
    let log_z = log_upper + log1mexp_positive(gap);
    if !log_z.is_finite() {
        return Err(TransformationNormalError::NumericalFailure {
            reason: format!(
                "normal CDF endpoint mass underflowed, lower={lower:.6e}, upper={upper:.6e}"
            ),
        }
        .into());
    }
    Ok(log_z)
}

pub(crate) fn transformation_normal_pit_score(
    h: f64,
    lower: f64,
    upper: f64,
    clip_eps: f64,
) -> Result<f64, String> {
    if !(clip_eps.is_finite() && clip_eps > 0.0 && clip_eps < 0.5) {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "transformation-normal PIT requires clip_eps in (0, 0.5), got {clip_eps}"
            ),
        }
        .into());
    }
    if !(h.is_finite() && lower.is_finite() && upper.is_finite()) {
        return Err(TransformationNormalError::InvalidInput { reason: format!(
            "transformation-normal PIT requires finite h/lower/upper, got h={h}, lower={lower}, upper={upper}"
        ) }.into());
    }
    if upper <= lower {
        return Err(TransformationNormalError::MonotonicityViolated { reason: format!(
            "transformation-normal PIT endpoint order violated: lower={lower:.6e}, upper={upper:.6e}"
        ) }.into());
    }

    // Extrapolation outside `[lower, upper]` is *not* a malformed input —
    // a test sample whose response sits at-or-beyond the training response
    // support will produce a finite `h` slightly below `lower` (or slightly
    // above `upper`) by exactly the amount the kernel reconstructs the
    // boundary. The PIT mapping is still well-defined: `u → 0` when
    // `h ≤ lower`, `u → 1` when `h ≥ upper`, and the `clip_eps` clamp on
    // the standard-normal quantile call at the end of this function turns
    // both into the extreme-quantile finite values that downstream
    // calibration code expects. Refusing here was surfacing routine
    // boundary roundoff at biobank shape (`p_resp` coefficients × O(1)
    // basis evaluations introduce ~`p_resp·ε·scale` noise — 64·ε·scale
    // is below that floor) as a hard prediction failure.
    //
    // A debug-level log preserves visibility for genuinely far-out
    // inputs without aborting the prediction. Non-finite `h` is already
    // rejected above at the `is_finite()` guard.
    if h < lower || h > upper {
        log::debug!(
            "transformation-normal PIT extrapolation: h={h:.6e}, lower={lower:.6e}, upper={upper:.6e} — clamping to support and continuing"
        );
    }
    let h_inside = h.clamp(lower, upper);
    let u = if h_inside <= lower {
        0.0
    } else if h_inside >= upper {
        1.0
    } else {
        let log_num = log_normal_cdf_diff(h_inside, lower)?;
        let log_den = log_normal_cdf_diff(upper, lower)?;
        let ratio = (log_num - log_den).exp();
        if !(ratio.is_finite() && ratio >= -1.0e-12 && ratio <= 1.0 + 1.0e-12) {
            return Err(TransformationNormalError::NumericalFailure { reason: format!(
                "transformation-normal PIT probability is not representable: h={h:.6e}, lower={lower:.6e}, upper={upper:.6e}, ratio={ratio}"
            ) }.into());
        }
        ratio.clamp(0.0, 1.0)
    };
    standard_normal_quantile(u.clamp(clip_eps, 1.0 - clip_eps))
        .map_err(|err| format!("transformation-normal PIT quantile failed: {err}"))
}

fn signed_normal_pdf_ratio(
    x: f64,
    polynomial_factor: f64,
    log_z: f64,
    factorial_scale: f64,
) -> f64 {
    if polynomial_factor == 0.0 {
        return 0.0;
    }
    const LOG_SQRT_2PI: f64 = 0.918_938_533_204_672_7;
    let log_abs =
        polynomial_factor.abs().ln() - 0.5 * x * x - LOG_SQRT_2PI - factorial_scale.ln() - log_z;
    polynomial_factor.signum() * log_abs.exp()
}

#[derive(Clone, Copy, Debug)]
struct LogNormalCdfDiffDerivatives {
    log_z: f64,
    first: [f64; 2],
    second: [[f64; 2]; 2],
    third: [[[f64; 2]; 2]; 2],
    fourth: [[[[f64; 2]; 2]; 2]; 2],
}

fn endpoint_chain_first(q: &LogNormalCdfDiffDerivatives, a: [f64; 2]) -> f64 {
    q.first[0] * a[0] + q.first[1] * a[1]
}

fn endpoint_chain_second(
    q: &LogNormalCdfDiffDerivatives,
    a: [f64; 2],
    b: [f64; 2],
    ab: [f64; 2],
) -> f64 {
    let mut out = endpoint_chain_first(q, ab);
    for i in 0..2 {
        for j in 0..2 {
            out += q.second[i][j] * a[i] * b[j];
        }
    }
    out
}

fn endpoint_chain_third(
    q: &LogNormalCdfDiffDerivatives,
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    ab: [f64; 2],
    ac: [f64; 2],
    bc: [f64; 2],
    abc: [f64; 2],
) -> f64 {
    let mut out = endpoint_chain_first(q, abc);
    for i in 0..2 {
        for j in 0..2 {
            out += q.second[i][j] * (ab[i] * c[j] + ac[i] * b[j] + bc[i] * a[j]);
            for k in 0..2 {
                out += q.third[i][j][k] * a[i] * b[j] * c[k];
            }
        }
    }
    out
}

fn endpoint_chain_fourth(
    q: &LogNormalCdfDiffDerivatives,
    a: [f64; 2],
    b: [f64; 2],
    c: [f64; 2],
    d: [f64; 2],
    ab: [f64; 2],
    ac: [f64; 2],
    ad: [f64; 2],
    bc: [f64; 2],
    bd: [f64; 2],
    cd: [f64; 2],
    abc: [f64; 2],
    abd: [f64; 2],
    acd: [f64; 2],
    bcd: [f64; 2],
    abcd: [f64; 2],
) -> f64 {
    let mut out = endpoint_chain_first(q, abcd);
    for i in 0..2 {
        for j in 0..2 {
            out += q.second[i][j]
                * (abc[i] * d[j]
                    + abd[i] * c[j]
                    + acd[i] * b[j]
                    + bcd[i] * a[j]
                    + ab[i] * cd[j]
                    + ac[i] * bd[j]
                    + ad[i] * bc[j]);
            for k in 0..2 {
                out += q.third[i][j][k]
                    * (ab[i] * c[j] * d[k]
                        + ac[i] * b[j] * d[k]
                        + ad[i] * b[j] * c[k]
                        + bc[i] * a[j] * d[k]
                        + bd[i] * a[j] * c[k]
                        + cd[i] * a[j] * b[k]);
                for l in 0..2 {
                    out += q.fourth[i][j][k][l] * a[i] * b[j] * c[k] * d[l];
                }
            }
        }
    }
    out
}

fn factorial(n: usize) -> f64 {
    match n {
        0 | 1 => 1.0,
        2 => 2.0,
        3 => 6.0,
        4 => 24.0,
        // CTN normalizer derivatives only need order <= 4; compute generically
        // as a safe fallback for any unexpected higher orders.
        other => {
            let mut acc = 24.0_f64;
            let mut k = 5usize;
            while k <= other {
                acc *= k as f64;
                k += 1;
            }
            acc
        }
    }
}

fn poly_mul_truncated(a: &[[f64; 5]; 5], b: &[[f64; 5]; 5]) -> [[f64; 5]; 5] {
    let mut out = [[0.0; 5]; 5];
    for ia in 0..=4 {
        for ib in 0..=(4 - ia) {
            let av = a[ia][ib];
            if av == 0.0 {
                continue;
            }
            for ja in 0..=(4 - ia) {
                for jb in 0..=(4 - ia - ja).min(4 - ib) {
                    let bv = b[ja][jb];
                    if bv != 0.0 && ia + ib + ja + jb <= 4 {
                        out[ia + ja][ib + jb] += av * bv;
                    }
                }
            }
        }
    }
    out
}

fn log_normal_cdf_diff_derivatives(
    upper: f64,
    lower: f64,
) -> Result<LogNormalCdfDiffDerivatives, String> {
    let log_z = log_normal_cdf_diff(upper, lower)?;
    if !log_z.is_finite() {
        return Err(TransformationNormalError::NonFinite {
            reason: format!(
                "normal CDF endpoint log-mass is not finite, lower={lower:.6e}, upper={upper:.6e}"
            ),
        }
        .into());
    }

    let s_u = [
        0.0,
        1.0,
        -upper,
        upper * upper - 1.0,
        -(upper * upper * upper - 3.0 * upper),
    ];
    let s_l = [
        0.0,
        -1.0,
        lower,
        -(lower * lower - 1.0),
        lower * lower * lower - 3.0 * lower,
    ];

    let mut r = [[0.0; 5]; 5];
    for order in 1..=4 {
        let factor = factorial(order);
        r[order][0] = signed_normal_pdf_ratio(upper, s_u[order], log_z, factor);
        r[0][order] = signed_normal_pdf_ratio(lower, s_l[order], log_z, factor);
        if !(r[order][0].is_finite() && r[0][order].is_finite()) {
            return Err(TransformationNormalError::NumericalFailure {
                reason: format!(
                    "normal CDF endpoint derivative ratio is not representable at order {order}, \
                 lower={lower:.6e}, upper={upper:.6e}, log_z={log_z:.6e}"
                ),
            }
            .into());
        }
    }

    let r2 = poly_mul_truncated(&r, &r);
    let r3 = poly_mul_truncated(&r2, &r);
    let r4 = poly_mul_truncated(&r3, &r);
    let mut q = [[0.0; 5]; 5];
    for i in 0..=4 {
        for j in 0..=(4 - i) {
            q[i][j] = r[i][j] - 0.5 * r2[i][j] + r3[i][j] / 3.0 - 0.25 * r4[i][j];
        }
    }

    let mut first = [0.0; 2];
    first[0] = q[1][0];
    first[1] = q[0][1];

    let mut second = [[0.0; 2]; 2];
    let mut third = [[[0.0; 2]; 2]; 2];
    let mut fourth = [[[[0.0; 2]; 2]; 2]; 2];
    for a in 0..2 {
        for b in 0..2 {
            let nu = (a == 0) as usize + (b == 0) as usize;
            let nl = 2 - nu;
            second[a][b] = q[nu][nl] * factorial(nu) * factorial(nl);
            for c in 0..2 {
                let nu = (a == 0) as usize + (b == 0) as usize + (c == 0) as usize;
                let nl = 3 - nu;
                third[a][b][c] = q[nu][nl] * factorial(nu) * factorial(nl);
                for d in 0..2 {
                    let nu = (a == 0) as usize
                        + (b == 0) as usize
                        + (c == 0) as usize
                        + (d == 0) as usize;
                    let nl = 4 - nu;
                    fourth[a][b][c][d] = q[nu][nl] * factorial(nu) * factorial(nl);
                }
            }
        }
    }

    Ok(LogNormalCdfDiffDerivatives {
        log_z,
        first,
        second,
        third,
        fourth,
    })
}

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

impl TransformationNormalFamily {
    /// Build a transformation model from response values and a pre-built covariate
    /// design operator with associated penalties.
    ///
    /// # Arguments
    ///
    /// * `response` - The response variable y (n observations).
    /// * `covariate_design` - Pre-built covariate-side design operator (n × p_cov).
    /// * `covariate_penalties` - Penalty matrices for the covariate basis.
    /// * `config` - Response-direction basis configuration.
    /// * `warm_start` - Optional location/scale from a prior normalizer.
    pub fn new(
        response: &Array1<f64>,
        weights: &Array1<f64>,
        offset: &Array1<f64>,
        covariate_design: DesignMatrix,
        covariate_penalties: Vec<PenaltyMatrix>,
        config: &TransformationNormalConfig,
        warm_start: Option<&TransformationWarmStart>,
    ) -> Result<Self, String> {
        let n = response.len();
        if covariate_design.nrows() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "response length {} != covariate design rows {}",
                    n,
                    covariate_design.nrows()
                ),
            }
            .into());
        }
        let p_cov = covariate_design.ncols();
        if p_cov == 0 {
            return Err(TransformationNormalError::DesignDegenerate {
                reason: "covariate design has zero columns".to_string(),
            }
            .into());
        }
        if weights.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!("response length {} != weights length {}", n, weights.len()),
            }
            .into());
        }
        if offset.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!("response length {} != offset length {}", n, offset.len()),
            }
            .into());
        }
        for (i, &weight) in weights.iter().enumerate() {
            if !weight.is_finite() {
                return Err(TransformationNormalError::NonFinite {
                    reason: format!("weights[{i}] is not finite: {weight}"),
                }
                .into());
            }
            if weight < 0.0 {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!("weights[{i}] must be non-negative: {weight}"),
                }
                .into());
            }
        }
        for (i, &value) in offset.iter().enumerate() {
            if !value.is_finite() {
                return Err(TransformationNormalError::NonFinite {
                    reason: format!("offset[{i}] is not finite: {value}"),
                }
                .into());
            }
        }
        for (i, sp) in covariate_penalties.iter().enumerate() {
            let (r, c) = sp.shape();
            if r != p_cov || c != p_cov {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!(
                        "covariate penalty {} has shape ({r}, {c}), expected ({p_cov}, {p_cov})",
                        i,
                    ),
                }
                .into());
            }
        }

        // ----- 1. Build response-direction basis -----
        let (resp_val, resp_deriv, resp_penalties, resp_knots, resp_transform) =
            build_response_basis(response, config)?;
        let p_resp = resp_val.ncols();
        let (response_lower_basis, response_upper_basis) =
            response_endpoint_value_bases(&resp_transform);

        // ----- 2. Row-wise Kronecker product (operator form) -----
        let x_val_kron = KroneckerDesign::new_khatri_rao(&resp_val, covariate_design.clone())?;
        let x_deriv_kron = KroneckerDesign::new_khatri_rao(&resp_deriv, covariate_design.clone())?;
        let p_total = p_resp * p_cov;
        debug_assert_eq!(x_val_kron.ncols(), p_total);
        debug_assert_eq!(x_deriv_kron.ncols(), p_total);

        // ----- 3. Warm start -----
        let initial_beta = compute_warm_start(
            response,
            weights,
            offset,
            &x_val_kron,
            &x_deriv_kron,
            &covariate_design,
            &covariate_penalties,
            p_resp,
            p_cov,
            warm_start,
        )?;

        // ----- 4. Tensor penalties (Kronecker-separable) -----
        let tensor_penalties = build_tensor_penalties_kronecker(
            &resp_penalties,
            covariate_penalties,
            p_resp,
            p_cov,
            config,
        )?;
        let policy = ResourcePolicy::default_library();
        let x_val_weighted_gram = x_val_kron.weighted_gram(weights, &policy);

        // ----- 5. CTN-specific smoothing seed from likelihood/penalty scales -----
        let initial_log_lambdas =
            ctn_penalty_scale_log_lambdas(&tensor_penalties, &x_val_weighted_gram);

        // Compute response median for anchoring
        let mut sorted_resp = response.to_vec();
        sorted_resp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let resp_median = if sorted_resp.len() % 2 == 1 {
            sorted_resp[sorted_resp.len() / 2]
        } else {
            0.5 * (sorted_resp[sorted_resp.len() / 2 - 1] + sorted_resp[sorted_resp.len() / 2])
        };
        let (response_floor_offset, response_lower_floor_offset, response_upper_floor_offset) =
            response_floor_offsets(response, &resp_knots, resp_median);

        Ok(Self {
            x_val_kron,
            x_deriv_kron,
            response_val_basis: resp_val,
            response_lower_basis,
            response_upper_basis,
            response_deriv_basis: resp_deriv,
            covariate_design,
            weights: Arc::new(weights.clone()),
            offset: Arc::new(offset.clone()),
            tensor_penalties,
            initial_beta,
            initial_log_lambdas,
            block_name: "transformation".to_string(),
            response_knots: resp_knots,
            response_transform: resp_transform,
            response_degree: config.response_degree,
            response_median: resp_median,
            response_floor_offset: Arc::new(response_floor_offset),
            response_lower_floor_offset,
            response_upper_floor_offset,
            covariate_dense_cache: Arc::new(Mutex::new(None)),
            row_quantity_cache: Arc::new(Mutex::new(None)),
            row_quantity_version: Arc::new(AtomicU64::new(0)),
            persistent_dense_hessian: Arc::new(CtnPersistentDenseHessianCache::default()),
            outer_subsample_weights: None,
            outer_subsample_hash: None,
        })
    }

    /// Build from a prebuilt response basis, skipping response basis construction.
    ///
    /// For the outer loop where the response basis is precomputed once and reused
    /// across κ iterations.
    pub fn from_prebuilt_response_basis(
        response: &Array1<f64>,
        response_val_basis: Array2<f64>,
        response_deriv_basis: Array2<f64>,
        response_penalties: Vec<Array2<f64>>,
        response_knots: Array1<f64>,
        response_degree: usize,
        response_transform: Array2<f64>,
        weights: &Array1<f64>,
        offset: &Array1<f64>,
        covariate_design: DesignMatrix,
        covariate_penalties: Vec<PenaltyMatrix>,
        config: &TransformationNormalConfig,
        warm_start: Option<&TransformationWarmStart>,
    ) -> Result<Self, String> {
        let n = response_val_basis.nrows();
        if n == 0 {
            return Err(TransformationNormalError::InvalidInput {
                reason: "response basis has zero rows".to_string(),
            }
            .into());
        }
        if response.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "response length {} != response basis rows {}",
                    response.len(),
                    n
                ),
            }
            .into());
        }
        if covariate_design.nrows() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "response basis rows {} != covariate design rows {}",
                    n,
                    covariate_design.nrows()
                ),
            }
            .into());
        }
        let p_cov = covariate_design.ncols();
        if p_cov == 0 {
            return Err(TransformationNormalError::DesignDegenerate {
                reason: "covariate design has zero columns".to_string(),
            }
            .into());
        }
        if weights.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "response basis rows {} != weights length {}",
                    n,
                    weights.len()
                ),
            }
            .into());
        }
        if offset.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "response basis rows {} != offset length {}",
                    n,
                    offset.len()
                ),
            }
            .into());
        }
        for (i, &weight) in weights.iter().enumerate() {
            if !weight.is_finite() {
                return Err(TransformationNormalError::NonFinite {
                    reason: format!("weights[{i}] is not finite: {weight}"),
                }
                .into());
            }
            if weight < 0.0 {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!("weights[{i}] must be non-negative: {weight}"),
                }
                .into());
            }
        }
        for (i, &value) in offset.iter().enumerate() {
            if !value.is_finite() {
                return Err(TransformationNormalError::NonFinite {
                    reason: format!("offset[{i}] is not finite: {value}"),
                }
                .into());
            }
        }
        for (i, sp) in covariate_penalties.iter().enumerate() {
            let (r, c) = sp.shape();
            if r != p_cov || c != p_cov {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!(
                        "covariate penalty {} has shape ({r}, {c}), expected ({p_cov}, {p_cov})",
                        i,
                    ),
                }
                .into());
            }
        }

        let p_resp = response_val_basis.ncols();
        if response_transform.ncols() + 1 != p_resp {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "response transform columns {} imply p_resp {}, but response value basis has {} columns",
                response_transform.ncols(),
                response_transform.ncols() + 1,
                p_resp
            ) }.into());
        }
        let (response_lower_basis, response_upper_basis) =
            response_endpoint_value_bases(&response_transform);

        // Row-wise Kronecker product (operator form).
        let x_val_kron =
            KroneckerDesign::new_khatri_rao(&response_val_basis, covariate_design.clone())?;
        let x_deriv_kron =
            KroneckerDesign::new_khatri_rao(&response_deriv_basis, covariate_design.clone())?;
        let p_total = p_resp * p_cov;
        debug_assert_eq!(x_val_kron.ncols(), p_total);
        debug_assert_eq!(x_deriv_kron.ncols(), p_total);

        let initial_beta = compute_warm_start(
            response,
            weights,
            offset,
            &x_val_kron,
            &x_deriv_kron,
            &covariate_design,
            &covariate_penalties,
            p_resp,
            p_cov,
            warm_start,
        )?;

        // Tensor penalties (Kronecker-separable).
        let tensor_penalties = build_tensor_penalties_kronecker(
            &response_penalties,
            covariate_penalties,
            p_resp,
            p_cov,
            config,
        )?;
        let policy = ResourcePolicy::default_library();
        let x_val_weighted_gram = x_val_kron.weighted_gram(weights, &policy);

        let initial_log_lambdas =
            ctn_penalty_scale_log_lambdas(&tensor_penalties, &x_val_weighted_gram);

        // Compute response median.
        let mut sorted_resp = response.to_vec();
        sorted_resp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let resp_median = if sorted_resp.len() % 2 == 1 {
            sorted_resp[sorted_resp.len() / 2]
        } else {
            0.5 * (sorted_resp[sorted_resp.len() / 2 - 1] + sorted_resp[sorted_resp.len() / 2])
        };
        let (response_floor_offset, response_lower_floor_offset, response_upper_floor_offset) =
            response_floor_offsets(response, &response_knots, resp_median);

        Ok(Self {
            x_val_kron,
            x_deriv_kron,
            response_val_basis,
            response_lower_basis,
            response_upper_basis,
            response_deriv_basis,
            covariate_design,
            weights: Arc::new(weights.clone()),
            offset: Arc::new(offset.clone()),
            tensor_penalties,
            initial_beta,
            initial_log_lambdas,
            block_name: "transformation".to_string(),
            response_knots: response_knots.clone(),
            response_transform: response_transform.clone(),
            response_degree,
            response_median: resp_median,
            response_floor_offset: Arc::new(response_floor_offset),
            response_lower_floor_offset,
            response_upper_floor_offset,
            covariate_dense_cache: Arc::new(Mutex::new(None)),
            row_quantity_cache: Arc::new(Mutex::new(None)),
            row_quantity_version: Arc::new(AtomicU64::new(0)),
            persistent_dense_hessian: Arc::new(CtnPersistentDenseHessianCache::default()),
            outer_subsample_weights: None,
            outer_subsample_hash: None,
        })
    }

    /// Response basis metadata for serialization/prediction.
    pub fn response_knots(&self) -> &Array1<f64> {
        &self.response_knots
    }
    pub fn response_transform(&self) -> &Array2<f64> {
        &self.response_transform
    }
    pub fn response_degree(&self) -> usize {
        self.response_degree
    }
    pub fn response_median(&self) -> f64 {
        self.response_median
    }

    /// Return the `ParameterBlockSpec` for this family (single block).
    pub fn block_spec(&self) -> ParameterBlockSpec {
        let offset = self.offset.as_ref() + self.response_floor_offset.as_ref();
        ParameterBlockSpec {
            name: self.block_name.clone(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(self.x_val_kron.clone()))),
            offset,
            penalties: self.tensor_penalties.clone(),
            nullspace_dims: vec![],
            initial_log_lambdas: self.initial_log_lambdas.clone(),
            initial_beta: Some(self.initial_beta.clone()),
        }
    }

    /// Total number of coefficients.
    pub fn p_total(&self) -> usize {
        self.x_val_kron.ncols()
    }

    /// Number of observations.
    pub fn n_obs(&self) -> usize {
        self.x_val_kron.nrows()
    }

    /// Per-row weight array used by every row-streaming SCOP assembly site.
    ///
    /// Returns the masked HT weights when an outer-score subsample is active
    /// (`outer_subsample_weights = Some(_)`), else the original `weights`.
    ///
    /// Math invariant: every CTN per-row contribution to the gradient,
    /// negative-Hessian, ψ-term, ψ-ψ-term, and log-likelihood is **linear**
    /// in this scalar — i.e. each `for i in 0..n` step is of the form
    /// `wᵢ · g(row_quantities_i, β)` with `wᵢ` appearing to the first power
    /// only. Replacing `wᵢ` with `wᵢ · m_i` (where `m_i = 1/πᵢ` on sampled
    /// rows and `0` on unsampled) yields an unbiased Horvitz-Thompson
    /// estimator: `E[Σᵢ mᵢ wᵢ g(row_i)] = Σᵢ wᵢ g(row_i) = full sum`.
    #[inline]
    pub(crate) fn effective_weights(&self) -> &Array1<f64> {
        match self.outer_subsample_weights.as_ref() {
            Some(w) => w.as_ref(),
            None => self.weights.as_ref(),
        }
    }

    /// Subsample-identity tag (`None` for full-data) used to key the
    /// row-quantity cache and `CtnDenseHessianKey`.
    #[inline]
    pub(crate) fn outer_subsample_tag(&self) -> Option<u64> {
        self.outer_subsample_hash
    }

    /// Clone the family with an outer-score Horvitz-Thompson mask installed.
    ///
    /// The mask `m` (length `n`) is `1/πᵢ` for sampled rows and `0.0` for
    /// unsampled. The returned family carries `outer_subsample_weights =
    /// Some(weights ⊙ m)`. The row-quantity cache and persistent dense
    /// Hessian cache are reset (they were keyed on β alone; the masked
    /// family's `log_likelihood` and Hessian differ from the full-data
    /// build at the same β so they must not alias). The subsample hash is
    /// computed over `m` so that two distinct masks at the same β never
    /// share a cache entry.
    fn with_outer_subsample(&self, mask: &Array1<f64>) -> Result<Self, TransformationNormalError> {
        let n = self.weights.len();
        if mask.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "outer-score subsample mask length {} != n={}",
                    mask.len(),
                    n
                ),
            });
        }
        let mut effective = Array1::<f64>::zeros(n);
        for i in 0..n {
            let m = mask[i];
            if !m.is_finite() || m < 0.0 {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!(
                        "outer-score subsample mask[{i}] = {m} is invalid (must be finite and >= 0)"
                    ),
                });
            }
            effective[i] = self.weights[i] * m;
        }
        let mut hasher = DefaultHasher::new();
        n.hash(&mut hasher);
        for value in mask.iter() {
            value.to_bits().hash(&mut hasher);
        }
        let subsample_hash = hasher.finish();
        Ok(Self {
            // Inherit immutable design / response state cheaply via Arc / clone.
            x_val_kron: self.x_val_kron.clone(),
            x_deriv_kron: self.x_deriv_kron.clone(),
            response_val_basis: self.response_val_basis.clone(),
            response_lower_basis: self.response_lower_basis.clone(),
            response_upper_basis: self.response_upper_basis.clone(),
            response_deriv_basis: self.response_deriv_basis.clone(),
            covariate_design: self.covariate_design.clone(),
            covariate_dense_cache: Arc::clone(&self.covariate_dense_cache),
            weights: Arc::clone(&self.weights),
            offset: Arc::clone(&self.offset),
            tensor_penalties: self.tensor_penalties.clone(),
            initial_beta: self.initial_beta.clone(),
            initial_log_lambdas: self.initial_log_lambdas.clone(),
            block_name: self.block_name.clone(),
            response_knots: self.response_knots.clone(),
            response_transform: self.response_transform.clone(),
            response_degree: self.response_degree,
            response_median: self.response_median,
            response_floor_offset: Arc::clone(&self.response_floor_offset),
            response_lower_floor_offset: self.response_lower_floor_offset,
            response_upper_floor_offset: self.response_upper_floor_offset,
            // Caches must NOT be shared between full-data and subsampled
            // families: the row-quantity cache stores the LL (mask-dependent),
            // and the persistent dense Hessian is keyed on β alone.
            row_quantity_cache: Arc::new(Mutex::new(None)),
            row_quantity_version: Arc::new(AtomicU64::new(0)),
            persistent_dense_hessian: Arc::new(CtnPersistentDenseHessianCache::default()),
            outer_subsample_weights: Some(Arc::new(effective)),
            outer_subsample_hash: Some(subsample_hash),
        })
    }

    /// Build an outer-subsample clone from a `BlockwiseFitOptions` row mask,
    /// returning `None` when no subsample is requested.
    fn maybe_with_outer_subsample_from_options(
        &self,
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Self>, TransformationNormalError> {
        let Some(sub) = options.outer_score_subsample.as_ref() else {
            return Ok(None);
        };
        let n = self.weights.len();
        let mut mask = Array1::<f64>::zeros(n);
        for row in sub.rows.iter() {
            if row.index < n {
                mask[row.index] = row.weight;
            }
        }
        Ok(Some(self.with_outer_subsample(&mask)?))
    }

    // --- Internal helpers ---

    fn covariate_dense_arc(&self) -> Result<Arc<Array2<f64>>, String> {
        let mut cache = self
            .covariate_dense_cache
            .lock()
            .expect("CTN covariate dense cache mutex poisoned");
        if let Some(cached) = cache.as_ref() {
            return Ok(cached.clone());
        }
        let dense = Arc::new(
            self.covariate_design
                .try_row_chunk(0..self.response_val_basis.nrows())
                .map_err(|e| format!("SCOP covariate dense materialization failed: {e}"))?,
        );
        *cache = Some(dense.clone());
        Ok(dense)
    }

    #[cfg(test)]
    fn scop_endpoint_values(
        &self,
        beta: &Array1<f64>,
        beta_mat: ArrayView2<'_, f64>,
        cov: ArrayView2<'_, f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), String> {
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        if beta.len() != p_resp * self.covariate_design.ncols() {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP endpoint beta length {} != p_resp({p_resp}) * p_cov({})",
                    beta.len(),
                    self.covariate_design.ncols()
                ),
            }
            .into());
        }
        let mut lower = Array1::<f64>::zeros(n);
        let mut upper = Array1::<f64>::zeros(n);
        let mut gamma = vec![0.0; p_resp];
        for i in 0..n {
            let cov_row = cov.row(i);
            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
            }
            let mut h_l = self.response_lower_basis[0] * gamma[0]
                + self.offset[i]
                + self.response_lower_floor_offset;
            let mut h_u = self.response_upper_basis[0] * gamma[0]
                + self.offset[i]
                + self.response_upper_floor_offset;
            for k in 1..p_resp {
                h_l += self.response_lower_basis[k] * gamma[k] * gamma[k];
                h_u += self.response_upper_basis[k] * gamma[k] * gamma[k];
            }
            lower[i] = h_l;
            upper[i] = h_u;
        }
        Ok((lower, upper))
    }

    fn row_quantities(
        &self,
        beta: &Array1<f64>,
    ) -> Result<TransformationNormalRowQuantityCache, String> {
        {
            let cache = self
                .row_quantity_cache
                .lock()
                .expect("CTN row quantity cache mutex poisoned");
            if let Some(cached) = cache.as_ref().filter(|cached| cached.matches_beta(beta)) {
                return Ok(cached.clone());
            }
        }

        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP endpoint beta reshape failed: {e}"))?;
        let cov = self.covariate_dense_arc()?;

        // SCOP-CTN: h(y, x) = b(x) + Σ_k γ_k(x)² · I_k(y), with
        // γ_k(x) = ψ(x)ᵀ Γ_{k,:} and h'(y, x) = Σ_k γ_k(x)² · M_k(y).
        // Response column 0 is the unconstrained affine/location component
        // b(x); all remaining response columns are squared shape components.
        //
        // The observed value, derivative value, and finite-support endpoints
        // all depend on the same covariate-side γ_k(x_i).  Compute γ once and
        // fan it out exactly; the previous path projected β through the same
        // covariate design three times per row-quantity build.
        let gamma = fast_abt(cov.as_ref(), &beta_mat);
        let n = gamma.nrows();
        let mut h = Array1::<f64>::zeros(n);
        let mut h_prime = Array1::<f64>::zeros(n);
        let mut h_lower = Array1::<f64>::zeros(n);
        let mut h_upper = Array1::<f64>::zeros(n);
        // Write directly into the four preallocated arrays in parallel; the
        // previous path collected a `Vec<(f64,f64,f64,f64)>` then serially
        // scattered into these arrays, costing 32 bytes per row of transient
        // allocation and a single-threaded post-pass at biobank scale.
        ndarray::Zip::indexed(&mut h)
            .and(&mut h_prime)
            .and(&mut h_lower)
            .and(&mut h_upper)
            .par_for_each(|i, h_i, hp_i, lower_i, upper_i| {
                let gamma_row = gamma.row(i);
                let val_row = self.response_val_basis.row(i);
                let deriv_row = self.response_deriv_basis.row(i);
                let g0 = gamma_row[0];
                let offset_i = self.offset[i];
                let mut h_acc = val_row[0] * g0 + offset_i + self.response_floor_offset[i];
                let mut hp_acc = deriv_row[0] * g0 + TRANSFORMATION_MONOTONICITY_EPS;
                let mut lower_acc =
                    self.response_lower_basis[0] * g0 + offset_i + self.response_lower_floor_offset;
                let mut upper_acc =
                    self.response_upper_basis[0] * g0 + offset_i + self.response_upper_floor_offset;
                for k in 1..p_resp {
                    let g_sq = gamma_row[k] * gamma_row[k];
                    h_acc += val_row[k] * g_sq;
                    hp_acc += deriv_row[k] * g_sq;
                    lower_acc += self.response_lower_basis[k] * g_sq;
                    upper_acc += self.response_upper_basis[k] * g_sq;
                }
                *h_i = h_acc;
                *hp_i = hp_acc;
                *lower_i = lower_acc;
                *upper_i = upper_acc;
            });
        for (i, &value) in h.iter().enumerate() {
            if !value.is_finite() {
                return Err(TransformationNormalError::NonFinite {
                    reason: format!(
                        "TransformationNormalFamily row_quantities: h[{i}] = {value} is not finite"
                    ),
                }
                .into());
            }
            if value.abs() > TRANSFORMATION_NORMAL_H_ABS_MAX {
                return Err(TransformationNormalError::InvalidInput { reason: format!(
                    "TransformationNormalFamily row_quantities: h[{i}] = {value:.6e} exceeds the standard-normal domain bound ±{TRANSFORMATION_NORMAL_H_ABS_MAX}"
                ) }.into());
            }
        }
        // Hard monotonicity / finiteness gate: the reciprocal powers `1/h'^k`
        // for k ∈ {1,2,3,4} feed the gradient, Hessian, and psi-psi outer
        // Hessian formulas. A non-finite or non-positive h' produces +∞ /
        // signed-∞ reciprocals which then collide with zero-valued probe
        // vectors (`v_*_deriv * weights`) to yield NaN entries throughout the
        // dense psi-psi block (`hessian_psi_psi`). The likelihood gate in
        // `evaluate` already rejects such β; surface the same error here so
        // outer-Hessian probe callsites that call `row_quantities` directly
        // (psi/psi second-order terms, etc.) produce a clean Err for the
        // outer evaluator to retreat on, rather than a NaN dense block that
        // routes a flagrant non-finite Hessian back into the planner.
        let mut min_hp = f64::INFINITY;
        let mut nonfinite_idx: Option<usize> = None;
        for (i, &hp) in h_prime.iter().enumerate() {
            if !hp.is_finite() {
                nonfinite_idx = Some(i);
                break;
            }
            if hp < min_hp {
                min_hp = hp;
            }
        }
        if let Some(i) = nonfinite_idx {
            return Err(TransformationNormalError::NonFinite {
                reason: format!(
                    "TransformationNormalFamily row_quantities: h'[{i}] = {} is not finite",
                    h_prime[i]
                ),
            }
            .into());
        }
        if min_hp <= 0.0 {
            return Err(TransformationNormalError::MonotonicityViolated { reason: format!(
                "TransformationNormalFamily row_quantities: h' has non-positive values (min = {min_hp:.6e}). \
                 Monotonicity constraint may be violated."
            ) }.into());
        }
        // Compute exact f64 row derivatives. If any required reciprocal power
        // is outside the finite representable range, surface an evaluation
        // error so the outer solver can retreat; do not clamp or approximate
        // the analytic Hessian terms.
        let derived = build_transformation_row_derived(
            &h,
            &h_prime,
            &h_lower,
            &h_upper,
            self.effective_weights(),
        )?;
        // Stamp this build with a fresh monotonic version so the persistent
        // dense-Hessian cache key advances exactly once per row_quantity
        // rebuild. `fetch_add` returns the *previous* value; +1 keeps the
        // first installed cache at version 1 (0 is reserved for "never
        // built").
        let version = self
            .row_quantity_version
            .fetch_add(1, AtomicOrdering::Relaxed)
            .saturating_add(1);
        let row_quantities = TransformationNormalRowQuantityCache {
            beta: Arc::new(beta.clone()),
            gamma: Arc::new(gamma),
            h: Arc::new(h),
            h_prime: Arc::new(h_prime),
            h_lower: Arc::new(h_lower),
            h_upper: Arc::new(h_upper),
            endpoint_q: Arc::new(derived.endpoint_q),
            log_likelihood: derived.log_likelihood,
            version,
        };

        let mut cache = self
            .row_quantity_cache
            .lock()
            .expect("CTN row quantity cache mutex poisoned");
        *cache = Some(row_quantities.clone());
        Ok(row_quantities)
    }

    fn scop_gradient_and_negative_hessian(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<(Array1<f64>, Array2<f64>), String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP gradient beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                    beta.len()
                ),
            }
            .into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP gradient/Hessian received row quantities for a different beta".to_string(),
            );
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP gradient/Hessian received row quantities for a different beta".to_string(),
            );
        }
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP gradient requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let endpoint_q = row_quantities.endpoint_q.as_ref();
        let gamma_rows = row_quantities.gamma.as_ref();
        if gamma_rows.nrows() != n || gamma_rows.ncols() != p_resp {
            return Err(format!(
                "SCOP gradient/Hessian gamma cache shape mismatch: got {}x{}, expected {}x{}",
                gamma_rows.nrows(),
                gamma_rows.ncols(),
                n,
                p_resp
            ));
        }
        let response_val_basis = &self.response_val_basis;
        let response_deriv_basis = &self.response_deriv_basis;
        let response_lower_basis = &self.response_lower_basis;
        let response_upper_basis = &self.response_upper_basis;

        struct ScopAccum {
            gradient: Array1<f64>,
            hessian: Array2<f64>,
        }

        impl ScopAccum {
            fn new(p_total: usize) -> Self {
                Self {
                    gradient: Array1::<f64>::zeros(p_total),
                    hessian: Array2::<f64>::zeros((p_total, p_total)),
                }
            }
        }

        let policy = ResourcePolicy::default_library();
        let accum_bytes = p_total
            .saturating_mul(p_total.saturating_add(1))
            .saturating_mul(std::mem::size_of::<f64>())
            .max(1);
        let memory_bound_chunks = (policy.max_single_materialization_bytes / accum_bytes).max(1);
        let target_chunks = rayon::current_num_threads()
            .saturating_mul(4)
            .max(1)
            .min(memory_bound_chunks)
            .min(n.max(1));
        let chunk_rows = n.max(1).div_ceil(target_chunks);
        let row_chunks: Vec<(usize, usize)> = (0..n)
            .step_by(chunk_rows)
            .map(|start| (start, (start + chunk_rows).min(n)))
            .collect();

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        // Rayon collects this indexed iterator into `Vec` in row-chunk order;
        // the final serial fold below preserves that order so results do not
        // depend on worker scheduling.
        let partials: Vec<ScopAccum> = row_chunks
            .into_par_iter()
            .map(|(start, end)| {
                let mut acc = ScopAccum::new(p_total);
                let mut dh_factor = vec![0.0; p_resp];
                let mut dhp_factor = vec![0.0; p_resp];
                let mut second_diag = vec![0.0; p_resp];
                let mut lower_factor = vec![0.0; p_resp];
                let mut upper_factor = vec![0.0; p_resp];

                for i in start..end {
                    let cov_row = cov.row(i);
                    let rv = response_val_basis.row(i);
                    let rd = response_deriv_basis.row(i);
                    let gamma = gamma_rows.row(i);
                    let wi = weights[i];
                    let hi = h[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;

                    let q = endpoint_q[i];
                    lower_factor[0] = response_lower_basis[0];
                    upper_factor[0] = response_upper_basis[0];
                    for k in 1..p_resp {
                        lower_factor[k] = 2.0 * response_lower_basis[k] * gamma[k];
                        upper_factor[k] = 2.0 * response_upper_basis[k] * gamma[k];
                    }

                    second_diag.fill(0.0);
                    dh_factor[0] = rv[0];
                    dhp_factor[0] = rd[0];
                    for k in 1..p_resp {
                        dh_factor[k] = 2.0 * rv[k] * gamma[k];
                        dhp_factor[k] = 2.0 * rd[k] * gamma[k];
                        second_diag[k] = 2.0 * (hi * rv[k] - rd[k] * inv_hp);
                    }

                    for k in 0..p_resp {
                        let normalizer_score_factor =
                            q.first[0] * upper_factor[k] + q.first[1] * lower_factor[k];
                        let score_factor = wi
                            * (-hi * dh_factor[k] + dhp_factor[k] * inv_hp
                                - normalizer_score_factor);
                        for c in 0..p_cov {
                            acc.gradient[k * p_cov + c] += score_factor * cov_row[c];
                        }
                    }

                    for k in 0..p_resp {
                        for l in 0..p_resp {
                            let mut block_factor = dh_factor[k] * dh_factor[l]
                                + dhp_factor[k] * dhp_factor[l] * inv_hp_sq;
                            if k == l {
                                block_factor += second_diag[k];
                            }
                            let upper_ab = if k == l && k > 0 {
                                2.0 * response_upper_basis[k]
                            } else {
                                0.0
                            };
                            let lower_ab = if k == l && k > 0 {
                                2.0 * response_lower_basis[k]
                            } else {
                                0.0
                            };
                            block_factor += q.first[0] * upper_ab
                                + q.first[1] * lower_ab
                                + q.second[0][0] * upper_factor[k] * upper_factor[l]
                                + q.second[0][1] * upper_factor[k] * lower_factor[l]
                                + q.second[1][0] * lower_factor[k] * upper_factor[l]
                                + q.second[1][1] * lower_factor[k] * lower_factor[l];
                            block_factor *= wi;
                            if block_factor == 0.0 {
                                continue;
                            }
                            for c in 0..p_cov {
                                let row_idx = k * p_cov + c;
                                let left = block_factor * cov_row[c];
                                for d in 0..p_cov {
                                    acc.hessian[[row_idx, l * p_cov + d]] += left * cov_row[d];
                                }
                            }
                        }
                    }
                }

                acc
            })
            .collect();

        let mut gradient = Array1::<f64>::zeros(p_total);
        let mut hessian = Array2::<f64>::zeros((p_total, p_total));
        for partial in partials {
            gradient.scaled_add(1.0, &partial.gradient);
            hessian.scaled_add(1.0, &partial.hessian);
        }

        Ok((gradient, hessian))
    }

    fn scop_gradient(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<Array1<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP gradient beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                    beta.len()
                ),
            }
            .into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err(TransformationNormalError::InvalidInput {
                reason: "SCOP gradient received row quantities for a different beta".to_string(),
            }
            .into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err("SCOP gradient received row quantities for a different beta".to_string());
        }
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP gradient requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let gamma_rows = row_quantities.gamma.as_ref();
        if gamma_rows.nrows() != n || gamma_rows.ncols() != p_resp {
            return Err(format!(
                "SCOP gradient gamma cache shape mismatch: got {}x{}, expected {}x{}",
                gamma_rows.nrows(),
                gamma_rows.ncols(),
                n,
                p_resp
            ));
        }
        let mut gradient = Array1::<f64>::zeros(p_total);
        let mut lower_factor = vec![0.0; p_resp];
        let mut upper_factor = vec![0.0; p_resp];

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let gamma = gamma_rows.row(i);
            let wi = weights[i];
            let hi = h[i];
            let inv_hp = 1.0 / h_prime[i];

            let q = row_quantities.endpoint_q[i];
            lower_factor[0] = self.response_lower_basis[0];
            upper_factor[0] = self.response_upper_basis[0];
            for k in 1..p_resp {
                lower_factor[k] = 2.0 * self.response_lower_basis[k] * gamma[k];
                upper_factor[k] = 2.0 * self.response_upper_basis[k] * gamma[k];
            }

            let normalizer_score0 = q.first[0] * upper_factor[0] + q.first[1] * lower_factor[0];
            let score0 = wi * (-hi * rv[0] + rd[0] * inv_hp - normalizer_score0);
            for c in 0..p_cov {
                gradient[c] += score0 * cov_row[c];
            }
            for k in 1..p_resp {
                let normalizer_score = q.first[0] * upper_factor[k] + q.first[1] * lower_factor[k];
                let score_factor =
                    wi * (2.0 * gamma[k] * (-hi * rv[k] + rd[k] * inv_hp) - normalizer_score);
                let offset = k * p_cov;
                for c in 0..p_cov {
                    gradient[offset + c] += score_factor * cov_row[c];
                }
            }
        }

        Ok(gradient)
    }

    /// Directional derivative of the SCOP negative Hessian.
    ///
    /// For one observation with covariate row `c`, response value row `r`, and
    /// derivative row `m`, define `g_k = beta_k·c`, `u_k = direction_k·c`,
    ///
    /// `h  = r_0 g_0 + Σ_{k>0} r_k g_k^2`
    /// `hp = m_0 g_0 + Σ_{k>0} m_k g_k^2`
    /// `Hu  = r_0 u_0 + Σ_{k>0} 2 r_k g_k u_k`
    /// `HPu = m_0 u_0 + Σ_{k>0} 2 m_k g_k u_k`.
    ///
    /// For coefficient `a=(k,p)`:
    ///
    /// `h_a  = r_k c_p` for `k=0`, else `2 r_k g_k c_p`
    /// `hp_a = m_k c_p` for `k=0`, else `2 m_k g_k c_p`
    /// `D h_a[u]  = 0` for `k=0`, else `2 r_k u_k c_p`
    /// `D hp_a[u] = 0` for `k=0`, else `2 m_k u_k c_p`.
    ///
    /// The per-row negative Hessian block is
    ///
    /// `H_ab = w [ h_a h_b + h h_ab + hp_a hp_b / hp^2 - hp_ab / hp ]`,
    ///
    /// where `h_ab = 2 r_k c_p c_q` and `hp_ab = 2 m_k c_p c_q` only when
    /// `a` and `b` are in the same squared shape component `k>0`; otherwise
    /// both second derivatives are zero. Differentiating this expression in
    /// direction `u` gives the exact formula implemented below:
    ///
    /// `D H_ab[u] = w [ (D h_a)h_b + h_a(D h_b) + Hu h_ab
    ///              + ((D hp_a)hp_b + hp_a(D hp_b))/hp^2
    ///              - 2 hp_a hp_b HPu/hp^3 + hp_ab HPu/hp^2 ]`.
    fn scop_hessian_directional_derivative(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<Array2<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total || direction.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP Hessian directional derivative length mismatch: beta={}, direction={}, expected={p_total}",
                beta.len(),
                direction.len()
            ) }.into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP Hessian directional derivative received row quantities for a different beta"
                    .to_string(),
            );
        }
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP direction reshape failed: {e}"))?;
        let cov = self.covariate_dense_arc().map_err(|e| {
            format!("SCOP Hessian directional derivative requires cached covariate design: {e}")
        })?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let gamma_rows = row_quantities.gamma.as_ref();
        if gamma_rows.nrows() != n || gamma_rows.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP Hessian directional derivative gamma cache shape mismatch: got {}x{}, expected {}x{}",
                gamma_rows.nrows(),
                gamma_rows.ncols(),
                n,
                p_resp
            ) }.into());
        }
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        const TARGET_CHUNK_COUNT: usize = 32;
        let chunk_size = n.div_ceil(TARGET_CHUNK_COUNT).max(1);
        let n_chunks = n.div_ceil(chunk_size);

        // Fixed row chunks give each rayon worker a private matrix accumulator while
        // preserving chunk-index order for deterministic floating-point reduction.
        let chunk_outputs: Vec<Array2<f64>> = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(n);
                let mut chunk_out = Array2::<f64>::zeros((p_total, p_total));

                // Hoist per-row scratch buffers above the row loop; each iteration
                // overwrites them after a `.fill(0.0)` of the slots whose `k=0`
                // entry stays at zero (the k>=1 entries are fully overwritten).
                let mut gamma_dir = vec![0.0; p_resp];
                let mut h_factor = vec![0.0; p_resp];
                let mut hp_factor = vec![0.0; p_resp];
                let mut h_factor_dir = vec![0.0; p_resp];
                let mut hp_factor_dir = vec![0.0; p_resp];
                let mut endpoint_factor_0 = vec![0.0; p_resp];
                let mut endpoint_factor_1 = vec![0.0; p_resp];
                let mut endpoint_factor_dir_0 = vec![0.0; p_resp];
                let mut endpoint_factor_dir_1 = vec![0.0; p_resp];

                for i in start..end {
                    let cov_row = cov.row(i);
                    let rv = self.response_val_basis.row(i);
                    let rd = self.response_deriv_basis.row(i);
                    let wi = weights[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;

                    let gamma = gamma_rows.row(i);
                    for k in 0..p_resp {
                        gamma_dir[k] = dir_mat.row(k).dot(&cov_row);
                    }

                    let mut h_dir = rv[0] * gamma_dir[0];
                    let mut hp_dir = rd[0] * gamma_dir[0];
                    let mut endpoint_dir = [
                        self.response_upper_basis[0] * gamma_dir[0],
                        self.response_lower_basis[0] * gamma_dir[0],
                    ];
                    for k in 1..p_resp {
                        h_dir += 2.0 * rv[k] * gamma[k] * gamma_dir[k];
                        hp_dir += 2.0 * rd[k] * gamma[k] * gamma_dir[k];
                        endpoint_dir[0] +=
                            2.0 * self.response_upper_basis[k] * gamma[k] * gamma_dir[k];
                        endpoint_dir[1] +=
                            2.0 * self.response_lower_basis[k] * gamma[k] * gamma_dir[k];
                    }
                    let q = row_quantities.endpoint_q[i];

                    // Reset the per-row scratch arrays. k=0 is set explicitly below;
                    // the *_dir factors only ever store k>=1 values so their k=0 slot
                    // must stay zero.
                    h_factor_dir[0] = 0.0;
                    hp_factor_dir[0] = 0.0;
                    endpoint_factor_dir_0[0] = 0.0;
                    endpoint_factor_dir_1[0] = 0.0;
                    h_factor[0] = rv[0];
                    hp_factor[0] = rd[0];
                    endpoint_factor_0[0] = self.response_upper_basis[0];
                    endpoint_factor_1[0] = self.response_lower_basis[0];
                    for k in 1..p_resp {
                        h_factor[k] = 2.0 * rv[k] * gamma[k];
                        hp_factor[k] = 2.0 * rd[k] * gamma[k];
                        h_factor_dir[k] = 2.0 * rv[k] * gamma_dir[k];
                        hp_factor_dir[k] = 2.0 * rd[k] * gamma_dir[k];
                        endpoint_factor_0[k] = 2.0 * self.response_upper_basis[k] * gamma[k];
                        endpoint_factor_1[k] = 2.0 * self.response_lower_basis[k] * gamma[k];
                        endpoint_factor_dir_0[k] =
                            2.0 * self.response_upper_basis[k] * gamma_dir[k];
                        endpoint_factor_dir_1[k] =
                            2.0 * self.response_lower_basis[k] * gamma_dir[k];
                    }
                    let endpoint_factor = [&endpoint_factor_0[..], &endpoint_factor_1[..]];
                    let endpoint_factor_dir =
                        [&endpoint_factor_dir_0[..], &endpoint_factor_dir_1[..]];

                    for k in 0..p_resp {
                        for l in 0..p_resp {
                            let same_shape = k == l && k > 0;
                            let mut normalizer_block = 0.0;
                            for a in 0..2 {
                                let h_a_ab = if same_shape {
                                    2.0 * if a == 0 {
                                        self.response_upper_basis[k]
                                    } else {
                                        self.response_lower_basis[k]
                                    }
                                } else {
                                    0.0
                                };
                                for b in 0..2 {
                                    normalizer_block += q.second[a][b] * endpoint_dir[b] * h_a_ab;
                                    normalizer_block += q.second[a][b]
                                        * (endpoint_factor_dir[a][k] * endpoint_factor[b][l]
                                            + endpoint_factor[a][k] * endpoint_factor_dir[b][l]);
                                    for c_ep in 0..2 {
                                        normalizer_block += q.third[a][b][c_ep]
                                            * endpoint_dir[c_ep]
                                            * endpoint_factor[a][k]
                                            * endpoint_factor[b][l];
                                    }
                                }
                            }
                            for c in 0..p_cov {
                                let row_idx = k * p_cov + c;
                                let h_a = h_factor[k] * cov_row[c];
                                let hp_a = hp_factor[k] * cov_row[c];
                                let dh_a = h_factor_dir[k] * cov_row[c];
                                let dhp_a = hp_factor_dir[k] * cov_row[c];
                                for d in 0..p_cov {
                                    let col_idx = l * p_cov + d;
                                    let h_b = h_factor[l] * cov_row[d];
                                    let hp_b = hp_factor[l] * cov_row[d];
                                    let dh_b = h_factor_dir[l] * cov_row[d];
                                    let dhp_b = hp_factor_dir[l] * cov_row[d];
                                    let (h_ab, hp_ab) = if same_shape {
                                        (
                                            2.0 * rv[k] * cov_row[c] * cov_row[d],
                                            2.0 * rd[k] * cov_row[c] * cov_row[d],
                                        )
                                    } else {
                                        (0.0, 0.0)
                                    };
                                    let value = dh_a * h_b
                                        + h_a * dh_b
                                        + h_dir * h_ab
                                        + (dhp_a * hp_b + hp_a * dhp_b) * inv_hp_sq
                                        - 2.0 * hp_a * hp_b * hp_dir * inv_hp_cu
                                        + hp_ab * hp_dir * inv_hp_sq
                                        + normalizer_block * cov_row[c] * cov_row[d];
                                    chunk_out[[row_idx, col_idx]] += wi * value;
                                }
                            }
                        }
                    }
                }

                chunk_out
            })
            .collect();

        let mut out = Array2::<f64>::zeros((p_total, p_total));
        for chunk in chunk_outputs {
            out.scaled_add(1.0, &chunk);
        }

        Ok(0.5 * (&out + &out.t()))
    }

    /// Second directional derivative of the SCOP negative Hessian.
    ///
    /// Reuse the notation from `scop_hessian_directional_derivative`, with
    /// two beta-space directions `u` and `v`. Since each shape component is
    /// quadratic in `g_k`, all third beta derivatives of `h` and `hp` are zero,
    /// while the mixed second directional derivatives are
    ///
    /// `Huv  = Σ_{k>0} 2 r_k u_k v_k`
    /// `HPuv = Σ_{k>0} 2 m_k u_k v_k`.
    ///
    /// Differentiating
    /// `D H_ab[u] = w[(D h_a)h_b + h_a(D h_b) + Hu h_ab
    ///              + ((D hp_a)hp_b + hp_a(D hp_b))/hp^2
    ///              - 2 hp_a hp_b HPu/hp^3 + hp_ab HPu/hp^2]`
    /// once more in direction `v` gives the expression implemented below:
    ///
    /// `D²H_ab[u,v] = w[
    ///      h_{a,u} h_{b,v} + h_{a,v} h_{b,u} + Huv h_ab
    ///    + (hp_{a,u} hp_{b,v} + hp_{a,v} hp_{b,u})/hp²
    ///    - 2(hp_{a,u} hp_b + hp_a hp_{b,u}) HPv/hp³
    ///    - 2(hp_{a,v} hp_b + hp_a hp_{b,v}) HPu/hp³
    ///    - 2 hp_a hp_b HPuv/hp³
    ///    + 6 hp_a hp_b HPu HPv/hp⁴
    ///    + hp_ab HPuv/hp²
    ///    - 2 hp_ab HPu HPv/hp³ ]`.
    fn scop_hessian_second_directional_derivative(
        &self,
        beta: &Array1<f64>,
        direction_u: &Array1<f64>,
        direction_v: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<Array2<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total || direction_u.len() != p_total || direction_v.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP Hessian second directional derivative length mismatch: beta={}, u={}, v={}, expected={p_total}",
                beta.len(),
                direction_u.len(),
                direction_v.len()
            ) }.into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP Hessian second directional derivative received row quantities for a different beta"
                    .to_string(),
            );
        }
        let dir_u_mat = direction_u
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP u direction reshape failed: {e}"))?;
        let dir_v_mat = direction_v
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP v direction reshape failed: {e}"))?;
        let cov = self.covariate_dense_arc().map_err(|e| {
            format!(
                "SCOP Hessian second directional derivative requires cached covariate design: {e}"
            )
        })?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let gamma_rows = row_quantities.gamma.as_ref();
        if gamma_rows.nrows() != n || gamma_rows.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP Hessian second directional derivative gamma cache shape mismatch: got {}x{}, expected {}x{}",
                gamma_rows.nrows(),
                gamma_rows.ncols(),
                n,
                p_resp
            ) }.into());
        }
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        const TARGET_CHUNK_COUNT: usize = 32;
        let chunk_size = n.div_ceil(TARGET_CHUNK_COUNT).max(1);
        let n_chunks = n.div_ceil(chunk_size);

        // Fixed row chunks give each rayon worker a private matrix accumulator while
        // preserving chunk-index order for deterministic floating-point reduction.
        let chunk_outputs: Vec<Array2<f64>> = (0..n_chunks)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_size;
                let end = (start + chunk_size).min(n);
                let mut chunk_out = Array2::<f64>::zeros((p_total, p_total));

                // Hoist per-row scratch buffers above the row loop; each iteration
                // fully overwrites the entries it reads (k>=1 entries reset by
                // arithmetic, k=0 entries reassigned below).
                let mut gamma_u = vec![0.0; p_resp];
                let mut gamma_v = vec![0.0; p_resp];
                let mut h_factor = vec![0.0; p_resp];
                let mut hp_factor = vec![0.0; p_resp];
                let mut h_factor_u = vec![0.0; p_resp];
                let mut hp_factor_u = vec![0.0; p_resp];
                let mut h_factor_v = vec![0.0; p_resp];
                let mut hp_factor_v = vec![0.0; p_resp];
                let mut endpoint_factor_0 = vec![0.0; p_resp];
                let mut endpoint_factor_1 = vec![0.0; p_resp];
                let mut endpoint_factor_u_0 = vec![0.0; p_resp];
                let mut endpoint_factor_u_1 = vec![0.0; p_resp];
                let mut endpoint_factor_v_0 = vec![0.0; p_resp];
                let mut endpoint_factor_v_1 = vec![0.0; p_resp];

                for i in start..end {
                    let cov_row = cov.row(i);
                    let rv = self.response_val_basis.row(i);
                    let rd = self.response_deriv_basis.row(i);
                    let wi = weights[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;

                    let gamma = gamma_rows.row(i);
                    for k in 0..p_resp {
                        gamma_u[k] = dir_u_mat.row(k).dot(&cov_row);
                        gamma_v[k] = dir_v_mat.row(k).dot(&cov_row);
                    }

                    let mut hp_u = rd[0] * gamma_u[0];
                    let mut hp_v = rd[0] * gamma_v[0];
                    let mut h_uv = 0.0;
                    let mut hp_uv = 0.0;
                    let mut endpoint_u = [
                        self.response_upper_basis[0] * gamma_u[0],
                        self.response_lower_basis[0] * gamma_u[0],
                    ];
                    let mut endpoint_v = [
                        self.response_upper_basis[0] * gamma_v[0],
                        self.response_lower_basis[0] * gamma_v[0],
                    ];
                    let mut endpoint_uv = [0.0, 0.0];
                    for k in 1..p_resp {
                        hp_u += 2.0 * rd[k] * gamma[k] * gamma_u[k];
                        hp_v += 2.0 * rd[k] * gamma[k] * gamma_v[k];
                        h_uv += 2.0 * rv[k] * gamma_u[k] * gamma_v[k];
                        hp_uv += 2.0 * rd[k] * gamma_u[k] * gamma_v[k];
                        endpoint_u[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_u[k];
                        endpoint_u[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_u[k];
                        endpoint_v[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_v[k];
                        endpoint_v[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_v[k];
                        endpoint_uv[0] +=
                            2.0 * self.response_upper_basis[k] * gamma_u[k] * gamma_v[k];
                        endpoint_uv[1] +=
                            2.0 * self.response_lower_basis[k] * gamma_u[k] * gamma_v[k];
                    }
                    let q = row_quantities.endpoint_q[i];

                    // Reset the per-row scratch arrays. k=0 entries are reassigned
                    // below; the *_u / *_v factors only ever hold k>=1 contributions,
                    // so their k=0 slot must be zeroed.
                    h_factor_u[0] = 0.0;
                    hp_factor_u[0] = 0.0;
                    h_factor_v[0] = 0.0;
                    hp_factor_v[0] = 0.0;
                    endpoint_factor_u_0[0] = 0.0;
                    endpoint_factor_u_1[0] = 0.0;
                    endpoint_factor_v_0[0] = 0.0;
                    endpoint_factor_v_1[0] = 0.0;
                    h_factor[0] = rv[0];
                    hp_factor[0] = rd[0];
                    endpoint_factor_0[0] = self.response_upper_basis[0];
                    endpoint_factor_1[0] = self.response_lower_basis[0];
                    for k in 1..p_resp {
                        h_factor[k] = 2.0 * rv[k] * gamma[k];
                        hp_factor[k] = 2.0 * rd[k] * gamma[k];
                        h_factor_u[k] = 2.0 * rv[k] * gamma_u[k];
                        hp_factor_u[k] = 2.0 * rd[k] * gamma_u[k];
                        h_factor_v[k] = 2.0 * rv[k] * gamma_v[k];
                        hp_factor_v[k] = 2.0 * rd[k] * gamma_v[k];
                        endpoint_factor_0[k] = 2.0 * self.response_upper_basis[k] * gamma[k];
                        endpoint_factor_1[k] = 2.0 * self.response_lower_basis[k] * gamma[k];
                        endpoint_factor_u_0[k] = 2.0 * self.response_upper_basis[k] * gamma_u[k];
                        endpoint_factor_u_1[k] = 2.0 * self.response_lower_basis[k] * gamma_u[k];
                        endpoint_factor_v_0[k] = 2.0 * self.response_upper_basis[k] * gamma_v[k];
                        endpoint_factor_v_1[k] = 2.0 * self.response_lower_basis[k] * gamma_v[k];
                    }
                    let endpoint_factor = [&endpoint_factor_0[..], &endpoint_factor_1[..]];
                    let endpoint_factor_u = [&endpoint_factor_u_0[..], &endpoint_factor_u_1[..]];
                    let endpoint_factor_v = [&endpoint_factor_v_0[..], &endpoint_factor_v_1[..]];

                    for k in 0..p_resp {
                        for l in 0..p_resp {
                            let same_shape = k == l && k > 0;
                            let mut normalizer_block = 0.0;
                            for a in 0..2 {
                                let h_a_ab = if same_shape {
                                    2.0 * if a == 0 {
                                        self.response_upper_basis[k]
                                    } else {
                                        self.response_lower_basis[k]
                                    }
                                } else {
                                    0.0
                                };
                                for b in 0..2 {
                                    normalizer_block += q.second[a][b] * endpoint_uv[b] * h_a_ab;
                                    for c_ep in 0..2 {
                                        normalizer_block += q.third[a][b][c_ep]
                                            * endpoint_v[c_ep]
                                            * endpoint_u[b]
                                            * h_a_ab;
                                        normalizer_block += q.third[a][b][c_ep]
                                            * endpoint_uv[c_ep]
                                            * endpoint_factor[a][k]
                                            * endpoint_factor[b][l];
                                        normalizer_block += q.third[a][b][c_ep]
                                            * endpoint_u[c_ep]
                                            * (endpoint_factor_v[a][k] * endpoint_factor[b][l]
                                                + endpoint_factor[a][k] * endpoint_factor_v[b][l]);
                                        normalizer_block += q.third[a][b][c_ep]
                                            * endpoint_v[c_ep]
                                            * endpoint_factor_u[a][k]
                                            * endpoint_factor[b][l];
                                        normalizer_block += q.third[a][b][c_ep]
                                            * endpoint_v[c_ep]
                                            * endpoint_factor[a][k]
                                            * endpoint_factor_u[b][l];
                                        for d_ep in 0..2 {
                                            normalizer_block += q.fourth[a][b][c_ep][d_ep]
                                                * endpoint_v[d_ep]
                                                * endpoint_u[c_ep]
                                                * endpoint_factor[a][k]
                                                * endpoint_factor[b][l];
                                        }
                                    }
                                    normalizer_block += q.second[a][b]
                                        * (endpoint_factor_u[a][k] * endpoint_factor_v[b][l]
                                            + endpoint_factor_v[a][k] * endpoint_factor_u[b][l]);
                                }
                            }
                            for c in 0..p_cov {
                                let row_idx = k * p_cov + c;
                                let hp_a = hp_factor[k] * cov_row[c];
                                let dh_a_u = h_factor_u[k] * cov_row[c];
                                let dhp_a_u = hp_factor_u[k] * cov_row[c];
                                let dh_a_v = h_factor_v[k] * cov_row[c];
                                let dhp_a_v = hp_factor_v[k] * cov_row[c];
                                for d in 0..p_cov {
                                    let col_idx = l * p_cov + d;
                                    let hp_b = hp_factor[l] * cov_row[d];
                                    let dh_b_u = h_factor_u[l] * cov_row[d];
                                    let dhp_b_u = hp_factor_u[l] * cov_row[d];
                                    let dh_b_v = h_factor_v[l] * cov_row[d];
                                    let dhp_b_v = hp_factor_v[l] * cov_row[d];
                                    let (h_ab, hp_ab) = if same_shape {
                                        (
                                            2.0 * rv[k] * cov_row[c] * cov_row[d],
                                            2.0 * rd[k] * cov_row[c] * cov_row[d],
                                        )
                                    } else {
                                        (0.0, 0.0)
                                    };
                                    let value = dh_a_u * dh_b_v
                                        + dh_a_v * dh_b_u
                                        + h_uv * h_ab
                                        + (dhp_a_u * dhp_b_v + dhp_a_v * dhp_b_u) * inv_hp_sq
                                        - 2.0
                                            * (dhp_a_u * hp_b + hp_a * dhp_b_u)
                                            * hp_v
                                            * inv_hp_cu
                                        - 2.0
                                            * (dhp_a_v * hp_b + hp_a * dhp_b_v)
                                            * hp_u
                                            * inv_hp_cu
                                        - 2.0 * hp_a * hp_b * hp_uv * inv_hp_cu
                                        + 6.0 * hp_a * hp_b * hp_u * hp_v * inv_hp_qu
                                        + hp_ab * hp_uv * inv_hp_sq
                                        - 2.0 * hp_ab * hp_u * hp_v * inv_hp_cu
                                        + normalizer_block * cov_row[c] * cov_row[d];
                                    chunk_out[[row_idx, col_idx]] += wi * value;
                                }
                            }
                        }
                    }
                }

                chunk_out
            })
            .collect();

        let mut out = Array2::<f64>::zeros((p_total, p_total));
        for chunk in chunk_outputs {
            out.scaled_add(1.0, &chunk);
        }

        Ok(0.5 * (&out + &out.t()))
    }

    fn scop_hessian_matvec_into(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        probe: &Array1<f64>,
        out: &mut Array1<f64>,
    ) -> Result<(), String> {
        let stage_start = std::time::Instant::now();
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total || probe.len() != p_total || out.len() != p_total {
            return Err(format!(
                "SCOP Hessian matvec length mismatch: beta={}, probe={}, out={}, expected={p_total}",
                beta.len(),
                probe.len(),
                out.len()
            ));
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP Hessian matvec received row quantities for a different beta".to_string(),
            );
        }
        let probe_mat = probe
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP probe reshape failed: {e}"))?;
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP Hessian matvec requires cached covariate design: {e}"))?;
        let weights = self.weights.as_ref();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let gamma_rows = row_quantities.gamma.as_ref();
        if gamma_rows.nrows() != n || gamma_rows.ncols() != p_resp {
            return Err(format!(
                "SCOP Hessian matvec gamma cache shape mismatch: got {}x{}, expected {}x{}",
                gamma_rows.nrows(),
                gamma_rows.ncols(),
                n,
                p_resp
            ));
        }

        out.fill(0.0);
        let mut probe_gamma = vec![0.0; p_resp];

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let gamma = gamma_rows.row(i);
            let wi = weights[i];
            let hi = h[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;

            for k in 0..p_resp {
                probe_gamma[k] = probe_mat.row(k).dot(&cov_row);
            }

            let mut h_probe = rv[0] * probe_gamma[0];
            let mut hp_probe = rd[0] * probe_gamma[0];
            let mut lower_probe = self.response_lower_basis[0] * probe_gamma[0];
            let mut upper_probe = self.response_upper_basis[0] * probe_gamma[0];
            for k in 1..p_resp {
                let pg = probe_gamma[k];
                let gamma_k = gamma[k];
                h_probe += 2.0 * rv[k] * gamma_k * pg;
                hp_probe += 2.0 * rd[k] * gamma_k * pg;
                lower_probe += 2.0 * self.response_lower_basis[k] * gamma_k * pg;
                upper_probe += 2.0 * self.response_upper_basis[k] * gamma_k * pg;
            }
            let q = row_quantities.endpoint_q[i];

            for k in 0..p_resp {
                let h_factor = if k == 0 {
                    rv[0]
                } else {
                    2.0 * rv[k] * gamma[k]
                };
                let hp_factor = if k == 0 {
                    rd[0]
                } else {
                    2.0 * rd[k] * gamma[k]
                };
                let lower_factor = if k == 0 {
                    self.response_lower_basis[0]
                } else {
                    2.0 * self.response_lower_basis[k] * gamma[k]
                };
                let upper_factor = if k == 0 {
                    self.response_upper_basis[0]
                } else {
                    2.0 * self.response_upper_basis[k] * gamma[k]
                };
                let pg = probe_gamma[k];
                let second_probe = if k == 0 {
                    0.0
                } else {
                    2.0 * (hi * rv[k] - rd[k] * inv_hp) * pg
                };
                let lower_factor_probe = if k == 0 {
                    0.0
                } else {
                    2.0 * self.response_lower_basis[k] * pg
                };
                let upper_factor_probe = if k == 0 {
                    0.0
                } else {
                    2.0 * self.response_upper_basis[k] * pg
                };
                let normalizer_probe = q.first[0] * upper_factor_probe
                    + q.first[1] * lower_factor_probe
                    + (q.second[0][0] * upper_factor + q.second[1][0] * lower_factor) * upper_probe
                    + (q.second[0][1] * upper_factor + q.second[1][1] * lower_factor) * lower_probe;
                let scalar = wi
                    * (h_factor * h_probe
                        + hp_factor * hp_probe * inv_hp_sq
                        + second_probe
                        + normalizer_probe);
                let row_offset = k * p_cov;
                for c in 0..p_cov {
                    out[row_offset + c] += scalar * cov_row[c];
                }
            }
        }

        log::info!(
            "[STAGE] CTN scop_hessian_matvec n={} p={} elapsed={:.3}s",
            n,
            p_total,
            stage_start.elapsed().as_secs_f64(),
        );
        Ok(())
    }

    fn scop_hessian_directional_matvec(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        probe: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let mut probes = Array2::<f64>::zeros((probe.len(), 1));
        probes.column_mut(0).assign(probe);
        let out = self.scop_hessian_directional_matmat(beta, direction, row_quantities, &probes)?;
        Ok(out.column(0).to_owned())
    }

    fn scop_hessian_directional_matmat(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        probes: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let stage_start = std::time::Instant::now();
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let n_probe = probes.ncols();
        if beta.len() != p_total || direction.len() != p_total || probes.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP dH matmat length mismatch: beta={}, direction={}, probes rows={}, expected={p_total}",
                beta.len(),
                direction.len(),
                probes.nrows()
            ) }.into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err("SCOP dH matmat received row quantities for a different beta".to_string());
        }
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP direction reshape failed: {e}"))?;
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP dH matmat requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let gamma_rows = row_quantities.gamma.as_ref();
        if gamma_rows.nrows() != n || gamma_rows.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP dH matmat gamma cache shape mismatch: got {}x{}, expected {}x{}",
                    gamma_rows.nrows(),
                    gamma_rows.ncols(),
                    n,
                    p_resp
                ),
            }
            .into());
        }
        let mut out = Array2::<f64>::zeros((p_total, n_probe));
        let mut gamma_dir = vec![0.0; p_resp];
        let mut gamma_probe = vec![0.0; p_resp * n_probe];
        let mut h_probe = vec![0.0; n_probe];
        let mut hp_probe = vec![0.0; n_probe];
        let mut h_dir_probe = vec![0.0; n_probe];
        let mut hp_dir_probe = vec![0.0; n_probe];
        let mut endpoint_probe = [vec![0.0; n_probe], vec![0.0; n_probe]];
        let mut endpoint_dir_probe = [vec![0.0; n_probe], vec![0.0; n_probe]];

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;

            let gamma = gamma_rows.row(i);
            for k in 0..p_resp {
                gamma_dir[k] = dir_mat.row(k).dot(&cov_row);
                let row_offset = k * p_cov;
                let probe_offset = k * n_probe;
                for j in 0..n_probe {
                    let mut value = 0.0;
                    for c in 0..p_cov {
                        value += probes[[row_offset + c, j]] * cov_row[c];
                    }
                    gamma_probe[probe_offset + j] = value;
                }
            }

            let mut h_dir = rv[0] * gamma_dir[0];
            let mut hp_dir = rd[0] * gamma_dir[0];
            let mut endpoint_dir = [
                self.response_upper_basis[0] * gamma_dir[0],
                self.response_lower_basis[0] * gamma_dir[0],
            ];
            for j in 0..n_probe {
                h_probe[j] = rv[0] * gamma_probe[j];
                hp_probe[j] = rd[0] * gamma_probe[j];
                h_dir_probe[j] = 0.0;
                hp_dir_probe[j] = 0.0;
                endpoint_probe[0][j] = self.response_upper_basis[0] * gamma_probe[j];
                endpoint_probe[1][j] = self.response_lower_basis[0] * gamma_probe[j];
                endpoint_dir_probe[0][j] = 0.0;
                endpoint_dir_probe[1][j] = 0.0;
            }
            for k in 1..p_resp {
                let probe_offset = k * n_probe;
                let gamma_k = gamma[k];
                let gamma_dir_k = gamma_dir[k];
                h_dir += 2.0 * rv[k] * gamma[k] * gamma_dir[k];
                hp_dir += 2.0 * rd[k] * gamma[k] * gamma_dir[k];
                endpoint_dir[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_dir[k];
                endpoint_dir[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_dir[k];
                for j in 0..n_probe {
                    let pg = gamma_probe[probe_offset + j];
                    h_probe[j] += 2.0 * rv[k] * gamma_k * pg;
                    hp_probe[j] += 2.0 * rd[k] * gamma_k * pg;
                    h_dir_probe[j] += 2.0 * rv[k] * gamma_dir_k * pg;
                    hp_dir_probe[j] += 2.0 * rd[k] * gamma_dir_k * pg;
                    endpoint_probe[0][j] += 2.0 * self.response_upper_basis[k] * gamma_k * pg;
                    endpoint_probe[1][j] += 2.0 * self.response_lower_basis[k] * gamma_k * pg;
                    endpoint_dir_probe[0][j] +=
                        2.0 * self.response_upper_basis[k] * gamma_dir_k * pg;
                    endpoint_dir_probe[1][j] +=
                        2.0 * self.response_lower_basis[k] * gamma_dir_k * pg;
                }
            }
            let q = row_quantities.endpoint_q[i];

            for k in 0..p_resp {
                let probe_offset = k * n_probe;
                let h_factor = if k == 0 {
                    rv[0]
                } else {
                    2.0 * rv[k] * gamma[k]
                };
                let hp_factor = if k == 0 {
                    rd[0]
                } else {
                    2.0 * rd[k] * gamma[k]
                };
                let h_factor_dir = if k == 0 {
                    0.0
                } else {
                    2.0 * rv[k] * gamma_dir[k]
                };
                let hp_factor_dir = if k == 0 {
                    0.0
                } else {
                    2.0 * rd[k] * gamma_dir[k]
                };
                let endpoint_factor = [
                    if k == 0 {
                        self.response_upper_basis[0]
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma[k]
                    },
                    if k == 0 {
                        self.response_lower_basis[0]
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma[k]
                    },
                ];
                let endpoint_factor_dir = [
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma_dir[k]
                    },
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma_dir[k]
                    },
                ];
                for j in 0..n_probe {
                    let pg = gamma_probe[probe_offset + j];
                    let h_second_probe = if k == 0 { 0.0 } else { 2.0 * rv[k] * pg };
                    let hp_second_probe = if k == 0 { 0.0 } else { 2.0 * rd[k] * pg };
                    let endpoint_factor_probe = [
                        if k == 0 {
                            0.0
                        } else {
                            2.0 * self.response_upper_basis[k] * pg
                        },
                        if k == 0 {
                            0.0
                        } else {
                            2.0 * self.response_lower_basis[k] * pg
                        },
                    ];
                    let mut normalizer_scalar = 0.0;
                    for a in 0..2 {
                        for b in 0..2 {
                            normalizer_scalar +=
                                q.second[a][b] * endpoint_dir[b] * endpoint_factor_probe[a];
                            normalizer_scalar += q.second[a][b]
                                * (endpoint_factor_dir[a] * endpoint_probe[b][j]
                                    + endpoint_factor[a] * endpoint_dir_probe[b][j]);
                            for c_ep in 0..2 {
                                normalizer_scalar += q.third[a][b][c_ep]
                                    * endpoint_dir[c_ep]
                                    * endpoint_factor[a]
                                    * endpoint_probe[b][j];
                            }
                        }
                    }
                    let scalar = wi
                        * (h_factor_dir * h_probe[j]
                            + h_factor * h_dir_probe[j]
                            + h_dir * h_second_probe
                            + (hp_factor_dir * hp_probe[j] + hp_factor * hp_dir_probe[j])
                                * inv_hp_sq
                            - 2.0 * hp_factor * hp_probe[j] * hp_dir * inv_hp_cu
                            + hp_second_probe * hp_dir * inv_hp_sq
                            + normalizer_scalar);
                    for c in 0..p_cov {
                        out[[k * p_cov + c, j]] += scalar * cov_row[c];
                    }
                }
            }
        }

        log::info!(
            "[STAGE] CTN scop_hessian_directional_matmat n={} p={} k={} elapsed={:.3}s",
            n,
            p_total,
            n_probe,
            stage_start.elapsed().as_secs_f64(),
        );
        Ok(out)
    }

    fn scop_projected_response_gram_table(
        &self,
        factor: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let rank = factor.ncols();
        if factor.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP projected response Gram factor row mismatch: factor_rows={}, expected={p_total}",
                factor.nrows()
            ) }.into());
        }
        let cov = self.covariate_dense_arc().map_err(|e| {
            format!("SCOP projected response Gram requires cached covariate design: {e}")
        })?;
        let stride = p_resp * p_resp;
        let mut grams = vec![0.0_f64; n * stride];

        let fill_row = |i: usize, row_out: &mut [f64], projected: &mut [f64]| {
            let cov_row = cov.row(i);
            projected.fill(0.0);
            for k in 0..p_resp {
                let factor_row_base = k * p_cov;
                let projected_base = k * rank;
                for c in 0..p_cov {
                    let x_ic = cov_row[c];
                    if x_ic == 0.0 {
                        continue;
                    }
                    let factor_row = factor_row_base + c;
                    for col in 0..rank {
                        projected[projected_base + col] += x_ic * factor[[factor_row, col]];
                    }
                }
            }
            for k in 0..p_resp {
                let k_base = k * rank;
                for l in 0..p_resp {
                    let l_base = l * rank;
                    let mut value = 0.0;
                    for col in 0..rank {
                        value += projected[k_base + col] * projected[l_base + col];
                    }
                    row_out[k * p_resp + l] = value;
                }
            }
        };

        if rayon::current_thread_index().is_some() {
            let mut projected = vec![0.0_f64; p_resp * rank];
            for (i, row_out) in grams.chunks_mut(stride).enumerate() {
                fill_row(i, row_out, &mut projected);
            }
        } else {
            use rayon::iter::{IndexedParallelIterator, ParallelIterator};
            use rayon::slice::ParallelSliceMut;
            grams.par_chunks_mut(stride).enumerate().for_each_init(
                || vec![0.0_f64; p_resp * rank],
                |projected, (i, row_out)| fill_row(i, row_out, projected),
            );
        }

        Array2::from_shape_vec((n, stride), grams)
            .map_err(|e| format!("SCOP projected response Gram table shape failed: {e}"))
    }

    fn scop_hessian_directional_trace_from_response_grams(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        row_grams: ArrayView2<'_, f64>,
    ) -> Result<f64, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total || direction.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP dH projected trace length mismatch: beta={}, direction={}, expected={p_total}",
                beta.len(),
                direction.len()
            ) }.into());
        }
        if row_grams.nrows() != n || row_grams.ncols() != p_resp * p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP dH projected trace Gram shape {}x{} != expected {}x{}",
                    row_grams.nrows(),
                    row_grams.ncols(),
                    n,
                    p_resp * p_resp
                ),
            }
            .into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP dH projected trace received row quantities for a different beta".to_string(),
            );
        }
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP dH projected trace direction reshape failed: {e}"))?;
        let cov = self.covariate_dense_arc().map_err(|e| {
            format!("SCOP dH projected trace requires cached covariate design: {e}")
        })?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let row_gamma = row_quantities.gamma.as_ref();

        struct DhTraceScratch {
            gamma: Vec<f64>,
            gamma_dir: Vec<f64>,
            h_factor: Vec<f64>,
            hp_factor: Vec<f64>,
            h_factor_dir: Vec<f64>,
            hp_factor_dir: Vec<f64>,
            endpoint_factor: [Vec<f64>; 2],
            endpoint_factor_dir: [Vec<f64>; 2],
        }

        impl DhTraceScratch {
            fn new(p_resp: usize) -> Self {
                Self {
                    gamma: vec![0.0; p_resp],
                    gamma_dir: vec![0.0; p_resp],
                    h_factor: vec![0.0; p_resp],
                    hp_factor: vec![0.0; p_resp],
                    h_factor_dir: vec![0.0; p_resp],
                    hp_factor_dir: vec![0.0; p_resp],
                    endpoint_factor: [vec![0.0; p_resp], vec![0.0; p_resp]],
                    endpoint_factor_dir: [vec![0.0; p_resp], vec![0.0; p_resp]],
                }
            }
        }

        let row_trace = |i: usize, scratch: &mut DhTraceScratch| {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;
            let gamma_row = row_gamma.row(i);
            for k in 0..p_resp {
                scratch.gamma[k] = gamma_row[k];
                scratch.gamma_dir[k] = dir_mat.row(k).dot(&cov_row);
            }

            let mut h_dir = rv[0] * scratch.gamma_dir[0];
            let mut hp_dir = rd[0] * scratch.gamma_dir[0];
            let mut endpoint_dir = [
                self.response_upper_basis[0] * scratch.gamma_dir[0],
                self.response_lower_basis[0] * scratch.gamma_dir[0],
            ];
            for k in 1..p_resp {
                h_dir += 2.0 * rv[k] * scratch.gamma[k] * scratch.gamma_dir[k];
                hp_dir += 2.0 * rd[k] * scratch.gamma[k] * scratch.gamma_dir[k];
                endpoint_dir[0] +=
                    2.0 * self.response_upper_basis[k] * scratch.gamma[k] * scratch.gamma_dir[k];
                endpoint_dir[1] +=
                    2.0 * self.response_lower_basis[k] * scratch.gamma[k] * scratch.gamma_dir[k];
            }
            let q = row_quantities.endpoint_q[i];

            scratch.h_factor[0] = rv[0];
            scratch.hp_factor[0] = rd[0];
            scratch.h_factor_dir[0] = 0.0;
            scratch.hp_factor_dir[0] = 0.0;
            scratch.endpoint_factor[0][0] = self.response_upper_basis[0];
            scratch.endpoint_factor[1][0] = self.response_lower_basis[0];
            scratch.endpoint_factor_dir[0][0] = 0.0;
            scratch.endpoint_factor_dir[1][0] = 0.0;
            for k in 1..p_resp {
                scratch.h_factor[k] = 2.0 * rv[k] * scratch.gamma[k];
                scratch.hp_factor[k] = 2.0 * rd[k] * scratch.gamma[k];
                scratch.h_factor_dir[k] = 2.0 * rv[k] * scratch.gamma_dir[k];
                scratch.hp_factor_dir[k] = 2.0 * rd[k] * scratch.gamma_dir[k];
                scratch.endpoint_factor[0][k] =
                    2.0 * self.response_upper_basis[k] * scratch.gamma[k];
                scratch.endpoint_factor[1][k] =
                    2.0 * self.response_lower_basis[k] * scratch.gamma[k];
                scratch.endpoint_factor_dir[0][k] =
                    2.0 * self.response_upper_basis[k] * scratch.gamma_dir[k];
                scratch.endpoint_factor_dir[1][k] =
                    2.0 * self.response_lower_basis[k] * scratch.gamma_dir[k];
            }

            let gram_row = row_grams.row(i);
            let mut total = 0.0;
            for k in 0..p_resp {
                for l in 0..p_resp {
                    let same_shape = k == l && k > 0;
                    let mut normalizer_block = 0.0;
                    for a in 0..2 {
                        let endpoint_second = if same_shape {
                            2.0 * if a == 0 {
                                self.response_upper_basis[k]
                            } else {
                                self.response_lower_basis[k]
                            }
                        } else {
                            0.0
                        };
                        for b in 0..2 {
                            normalizer_block += q.second[a][b] * endpoint_dir[b] * endpoint_second;
                            normalizer_block += q.second[a][b]
                                * (scratch.endpoint_factor_dir[a][k]
                                    * scratch.endpoint_factor[b][l]
                                    + scratch.endpoint_factor[a][k]
                                        * scratch.endpoint_factor_dir[b][l]);
                            for c_ep in 0..2 {
                                normalizer_block += q.third[a][b][c_ep]
                                    * endpoint_dir[c_ep]
                                    * scratch.endpoint_factor[a][k]
                                    * scratch.endpoint_factor[b][l];
                            }
                        }
                    }
                    let second_h = if same_shape { 2.0 * rv[k] } else { 0.0 };
                    let second_hp = if same_shape { 2.0 * rd[k] } else { 0.0 };
                    let q_kl = scratch.h_factor_dir[k] * scratch.h_factor[l]
                        + scratch.h_factor[k] * scratch.h_factor_dir[l]
                        + h_dir * second_h
                        + (scratch.hp_factor_dir[k] * scratch.hp_factor[l]
                            + scratch.hp_factor[k] * scratch.hp_factor_dir[l])
                            * inv_hp_sq
                        - 2.0 * scratch.hp_factor[k] * scratch.hp_factor[l] * hp_dir * inv_hp_cu
                        + second_hp * hp_dir * inv_hp_sq
                        + normalizer_block;
                    total += q_kl * gram_row[k * p_resp + l];
                }
            }
            wi * total
        };

        if rayon::current_thread_index().is_some() {
            let mut scratch = DhTraceScratch::new(p_resp);
            Ok((0..n).map(|i| row_trace(i, &mut scratch)).sum())
        } else {
            use rayon::iter::{IntoParallelIterator, ParallelIterator};
            Ok((0..n)
                .into_par_iter()
                .fold(
                    || (DhTraceScratch::new(p_resp), 0.0),
                    |(mut scratch, mut sum), i| {
                        sum += row_trace(i, &mut scratch);
                        (scratch, sum)
                    },
                )
                .map(|(_, sum)| sum)
                .sum())
        }
    }

    fn scop_hessian_second_directional_matvec(
        &self,
        beta: &Array1<f64>,
        direction_u: &Array1<f64>,
        direction_v: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        probe: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let mut probes = Array2::<f64>::zeros((probe.len(), 1));
        probes.column_mut(0).assign(probe);
        let out = self.scop_hessian_second_directional_matmat(
            beta,
            direction_u,
            direction_v,
            row_quantities,
            &probes,
        )?;
        Ok(out.column(0).to_owned())
    }

    fn scop_hessian_second_directional_matmat(
        &self,
        beta: &Array1<f64>,
        direction_u: &Array1<f64>,
        direction_v: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        probes: &Array2<f64>,
    ) -> Result<Array2<f64>, String> {
        let stage_start = std::time::Instant::now();
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let n_probe = probes.ncols();
        if beta.len() != p_total
            || direction_u.len() != p_total
            || direction_v.len() != p_total
            || probes.nrows() != p_total
        {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP d2H matmat length mismatch: beta={}, u={}, v={}, probes rows={}, expected={p_total}",
                beta.len(),
                direction_u.len(),
                direction_v.len(),
                probes.nrows()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP beta reshape failed: {e}"))?;
        let dir_u_mat = direction_u
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP u direction reshape failed: {e}"))?;
        let dir_v_mat = direction_v
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP v direction reshape failed: {e}"))?;
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP d2H matmat requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let mut out = Array2::<f64>::zeros((p_total, n_probe));
        let mut gamma = vec![0.0; p_resp];
        let mut gamma_u = vec![0.0; p_resp];
        let mut gamma_v = vec![0.0; p_resp];
        let mut gamma_probe = vec![0.0; p_resp * n_probe];
        let mut hp_probe = vec![0.0; n_probe];
        let mut h_u_probe = vec![0.0; n_probe];
        let mut hp_u_probe = vec![0.0; n_probe];
        let mut h_v_probe = vec![0.0; n_probe];
        let mut hp_v_probe = vec![0.0; n_probe];
        let mut endpoint_probe = [vec![0.0; n_probe], vec![0.0; n_probe]];
        let mut endpoint_u_probe = [vec![0.0; n_probe], vec![0.0; n_probe]];
        let mut endpoint_v_probe = [vec![0.0; n_probe], vec![0.0; n_probe]];

        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;
            let inv_hp_qu = inv_hp_sq * inv_hp_sq;

            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
                gamma_u[k] = dir_u_mat.row(k).dot(&cov_row);
                gamma_v[k] = dir_v_mat.row(k).dot(&cov_row);
                let row_offset = k * p_cov;
                let probe_offset = k * n_probe;
                for j in 0..n_probe {
                    let mut value = 0.0;
                    for c in 0..p_cov {
                        value += probes[[row_offset + c, j]] * cov_row[c];
                    }
                    gamma_probe[probe_offset + j] = value;
                }
            }

            let mut hp_u = rd[0] * gamma_u[0];
            let mut hp_v = rd[0] * gamma_v[0];
            let mut h_uv = 0.0;
            let mut hp_uv = 0.0;
            let mut endpoint_u = [
                self.response_upper_basis[0] * gamma_u[0],
                self.response_lower_basis[0] * gamma_u[0],
            ];
            let mut endpoint_v = [
                self.response_upper_basis[0] * gamma_v[0],
                self.response_lower_basis[0] * gamma_v[0],
            ];
            let mut endpoint_uv = [0.0, 0.0];
            for j in 0..n_probe {
                hp_probe[j] = rd[0] * gamma_probe[j];
                h_u_probe[j] = 0.0;
                hp_u_probe[j] = 0.0;
                h_v_probe[j] = 0.0;
                hp_v_probe[j] = 0.0;
                endpoint_probe[0][j] = self.response_upper_basis[0] * gamma_probe[j];
                endpoint_probe[1][j] = self.response_lower_basis[0] * gamma_probe[j];
                endpoint_u_probe[0][j] = 0.0;
                endpoint_u_probe[1][j] = 0.0;
                endpoint_v_probe[0][j] = 0.0;
                endpoint_v_probe[1][j] = 0.0;
            }
            for k in 1..p_resp {
                let probe_offset = k * n_probe;
                let gamma_k = gamma[k];
                let gamma_u_k = gamma_u[k];
                let gamma_v_k = gamma_v[k];
                hp_u += 2.0 * rd[k] * gamma[k] * gamma_u[k];
                hp_v += 2.0 * rd[k] * gamma[k] * gamma_v[k];
                h_uv += 2.0 * rv[k] * gamma_u[k] * gamma_v[k];
                hp_uv += 2.0 * rd[k] * gamma_u[k] * gamma_v[k];
                endpoint_u[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_u[k];
                endpoint_u[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_u[k];
                endpoint_v[0] += 2.0 * self.response_upper_basis[k] * gamma[k] * gamma_v[k];
                endpoint_v[1] += 2.0 * self.response_lower_basis[k] * gamma[k] * gamma_v[k];
                endpoint_uv[0] += 2.0 * self.response_upper_basis[k] * gamma_u[k] * gamma_v[k];
                endpoint_uv[1] += 2.0 * self.response_lower_basis[k] * gamma_u[k] * gamma_v[k];
                for j in 0..n_probe {
                    let pg = gamma_probe[probe_offset + j];
                    hp_probe[j] += 2.0 * rd[k] * gamma_k * pg;
                    h_u_probe[j] += 2.0 * rv[k] * gamma_u_k * pg;
                    hp_u_probe[j] += 2.0 * rd[k] * gamma_u_k * pg;
                    h_v_probe[j] += 2.0 * rv[k] * gamma_v_k * pg;
                    hp_v_probe[j] += 2.0 * rd[k] * gamma_v_k * pg;
                    endpoint_probe[0][j] += 2.0 * self.response_upper_basis[k] * gamma_k * pg;
                    endpoint_probe[1][j] += 2.0 * self.response_lower_basis[k] * gamma_k * pg;
                    endpoint_u_probe[0][j] += 2.0 * self.response_upper_basis[k] * gamma_u_k * pg;
                    endpoint_u_probe[1][j] += 2.0 * self.response_lower_basis[k] * gamma_u_k * pg;
                    endpoint_v_probe[0][j] += 2.0 * self.response_upper_basis[k] * gamma_v_k * pg;
                    endpoint_v_probe[1][j] += 2.0 * self.response_lower_basis[k] * gamma_v_k * pg;
                }
            }
            let q = row_quantities.endpoint_q[i];

            for k in 0..p_resp {
                let probe_offset = k * n_probe;
                let hp_factor = if k == 0 {
                    rd[0]
                } else {
                    2.0 * rd[k] * gamma[k]
                };
                let h_factor_u = if k == 0 {
                    0.0
                } else {
                    2.0 * rv[k] * gamma_u[k]
                };
                let hp_factor_u = if k == 0 {
                    0.0
                } else {
                    2.0 * rd[k] * gamma_u[k]
                };
                let h_factor_v = if k == 0 {
                    0.0
                } else {
                    2.0 * rv[k] * gamma_v[k]
                };
                let hp_factor_v = if k == 0 {
                    0.0
                } else {
                    2.0 * rd[k] * gamma_v[k]
                };
                let endpoint_factor = [
                    if k == 0 {
                        self.response_upper_basis[0]
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma[k]
                    },
                    if k == 0 {
                        self.response_lower_basis[0]
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma[k]
                    },
                ];
                let endpoint_factor_u = [
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma_u[k]
                    },
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma_u[k]
                    },
                ];
                let endpoint_factor_v = [
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_upper_basis[k] * gamma_v[k]
                    },
                    if k == 0 {
                        0.0
                    } else {
                        2.0 * self.response_lower_basis[k] * gamma_v[k]
                    },
                ];
                for j in 0..n_probe {
                    let pg = gamma_probe[probe_offset + j];
                    let h_second_probe = if k == 0 { 0.0 } else { 2.0 * rv[k] * pg };
                    let hp_second_probe = if k == 0 { 0.0 } else { 2.0 * rd[k] * pg };
                    let endpoint_factor_probe = [
                        if k == 0 {
                            0.0
                        } else {
                            2.0 * self.response_upper_basis[k] * pg
                        },
                        if k == 0 {
                            0.0
                        } else {
                            2.0 * self.response_lower_basis[k] * pg
                        },
                    ];
                    let mut normalizer_scalar = 0.0;
                    for a in 0..2 {
                        for b in 0..2 {
                            normalizer_scalar +=
                                q.second[a][b] * endpoint_uv[b] * endpoint_factor_probe[a];
                            for c_ep in 0..2 {
                                normalizer_scalar += q.third[a][b][c_ep]
                                    * endpoint_v[c_ep]
                                    * endpoint_u[b]
                                    * endpoint_factor_probe[a];
                                normalizer_scalar += q.third[a][b][c_ep]
                                    * endpoint_uv[c_ep]
                                    * endpoint_factor[a]
                                    * endpoint_probe[b][j];
                                normalizer_scalar += q.third[a][b][c_ep]
                                    * endpoint_u[c_ep]
                                    * (endpoint_factor_v[a] * endpoint_probe[b][j]
                                        + endpoint_factor[a] * endpoint_v_probe[b][j]);
                                normalizer_scalar += q.third[a][b][c_ep]
                                    * endpoint_v[c_ep]
                                    * endpoint_factor_u[a]
                                    * endpoint_probe[b][j];
                                normalizer_scalar += q.third[a][b][c_ep]
                                    * endpoint_v[c_ep]
                                    * endpoint_factor[a]
                                    * endpoint_u_probe[b][j];
                                for d_ep in 0..2 {
                                    normalizer_scalar += q.fourth[a][b][c_ep][d_ep]
                                        * endpoint_v[d_ep]
                                        * endpoint_u[c_ep]
                                        * endpoint_factor[a]
                                        * endpoint_probe[b][j];
                                }
                            }
                            normalizer_scalar += q.second[a][b]
                                * (endpoint_factor_u[a] * endpoint_v_probe[b][j]
                                    + endpoint_factor_v[a] * endpoint_u_probe[b][j]);
                        }
                    }
                    let scalar = wi
                        * (h_factor_u * h_v_probe[j]
                            + h_factor_v * h_u_probe[j]
                            + h_uv * h_second_probe
                            + (hp_factor_u * hp_v_probe[j] + hp_factor_v * hp_u_probe[j])
                                * inv_hp_sq
                            - 2.0
                                * (hp_factor_u * hp_probe[j] + hp_factor * hp_u_probe[j])
                                * hp_v
                                * inv_hp_cu
                            - 2.0
                                * (hp_factor_v * hp_probe[j] + hp_factor * hp_v_probe[j])
                                * hp_u
                                * inv_hp_cu
                            - 2.0 * hp_factor * hp_probe[j] * hp_uv * inv_hp_cu
                            + 6.0 * hp_factor * hp_probe[j] * hp_u * hp_v * inv_hp_qu
                            + hp_second_probe * hp_uv * inv_hp_sq
                            - 2.0 * hp_second_probe * hp_u * hp_v * inv_hp_cu
                            + normalizer_scalar);
                    for c in 0..p_cov {
                        out[[k * p_cov + c, j]] += scalar * cov_row[c];
                    }
                }
            }
        }

        log::info!(
            "[STAGE] CTN scop_hessian_second_directional_matmat n={} p={} k={} elapsed={:.3}s",
            n,
            p_total,
            n_probe,
            stage_start.elapsed().as_secs_f64(),
        );
        Ok(out)
    }

    fn scop_hessian_diagonal(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
    ) -> Result<Array1<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP Hessian diagonal beta length {} != expected {p_total}",
                    beta.len()
                ),
            }
            .into());
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP Hessian diagonal received row quantities for a different beta".to_string(),
            );
        }
        if !row_quantities.matches_beta(beta) {
            return Err(
                "SCOP Hessian diagonal received row quantities for a different beta".to_string(),
            );
        }
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP Hessian diagonal requires cached covariate design: {e}"))?;
        let weights = self.effective_weights();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let gamma_rows = row_quantities.gamma.as_ref();
        if gamma_rows.nrows() != n || gamma_rows.ncols() != p_resp {
            return Err(format!(
                "SCOP Hessian diagonal gamma cache shape mismatch: got {}x{}, expected {}x{}",
                gamma_rows.nrows(),
                gamma_rows.ncols(),
                n,
                p_resp
            ));
        }
        let mut diag = Array1::<f64>::zeros(p_total);
        for i in 0..n {
            let cov_row = cov.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let gamma = gamma_rows.row(i);
            let wi = weights[i];
            let hi = h[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let q = row_quantities.endpoint_q[i];

            // k = 0 prologue: second / lower_second / upper_second vanish.
            {
                let h_factor = rv[0];
                let hp_factor = rd[0];
                let lower_factor = self.response_lower_basis[0];
                let upper_factor = self.response_upper_basis[0];
                let normalizer_second = q.second[0][0] * upper_factor * upper_factor
                    + (q.second[0][1] + q.second[1][0]) * upper_factor * lower_factor
                    + q.second[1][1] * lower_factor * lower_factor;
                let coeff = wi
                    * (h_factor * h_factor + hp_factor * hp_factor * inv_hp_sq + normalizer_second);
                for c in 0..p_cov {
                    let cc = cov_row[c] * cov_row[c];
                    diag[c] += coeff * cc;
                }
            }

            for k in 1..p_resp {
                let two_gamma_k = 2.0 * gamma[k];
                let h_factor = rv[k] * two_gamma_k;
                let hp_factor = rd[k] * two_gamma_k;
                let second = 2.0 * (hi * rv[k] - rd[k] * inv_hp);
                let lower_factor = self.response_lower_basis[k] * two_gamma_k;
                let upper_factor = self.response_upper_basis[k] * two_gamma_k;
                let lower_second = 2.0 * self.response_lower_basis[k];
                let upper_second = 2.0 * self.response_upper_basis[k];
                let normalizer_second = q.first[0] * upper_second
                    + q.first[1] * lower_second
                    + q.second[0][0] * upper_factor * upper_factor
                    + (q.second[0][1] + q.second[1][0]) * upper_factor * lower_factor
                    + q.second[1][1] * lower_factor * lower_factor;
                let coeff = wi
                    * (h_factor * h_factor
                        + hp_factor * hp_factor * inv_hp_sq
                        + second
                        + normalizer_second);
                let row_offset = k * p_cov;
                for c in 0..p_cov {
                    let cc = cov_row[c] * cov_row[c];
                    diag[row_offset + c] += coeff * cc;
                }
            }
        }
        Ok(diag)
    }

    /// First derivative of the profiled objective pieces with respect to one
    /// covariate-basis deformation axis `psi`.
    ///
    /// The SCOP map is the same as above, but `psi` differentiates the
    /// covariate row, not `beta`: `g_k,psi = beta_k·c_psi`. Thus
    /// `h_psi = r_0 g_0,psi + Σ_{k>0} 2 r_k g_k g_k,psi` and likewise for
    /// `hp_psi`. The objective contribution is
    ///
    /// `L_psi = w [ h h_psi - hp_psi / hp ]`
    ///
    /// because the minimized criterion is `0.5 h^2 - log(hp)`. Differentiating
    /// once with respect to `beta_a` gives `score_psi`; differentiating again
    /// gives `hessian_psi`. The code below expands those derivatives directly,
    /// including both paths through `c` and `c_psi` for same-component squared
    /// shape terms.
    fn scop_psi_terms(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        op: &TensorKroneckerPsiOperator,
        op_arc: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        axis: usize,
    ) -> Result<ExactNewtonJointPsiTerms, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi terms beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                    beta.len()
                ),
            }
            .into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi beta reshape failed: {e}"))?;
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP psi terms require cached covariate design: {e}"))?;
        let cov_psi = op
            .materialize_cov_first_axis(axis)
            .map_err(|e| format!("SCOP psi materialize_cov_first failed: {e}"))?;
        if cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi covariate derivative shape {}x{} != expected {}x{}",
                    cov_psi.nrows(),
                    cov_psi.ncols(),
                    n,
                    p_cov
                ),
            }
            .into());
        }

        let weights = self.effective_weights();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let mut objective_psi = 0.0;
        let mut score_psi = Array1::<f64>::zeros(p_total);
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];
        let mut gamma = vec![0.0; p_resp];
        let mut gamma_psi = vec![0.0; p_resp];
        let mut endpoint_factor = vec![[0.0; 2]; p_resp];
        let mut endpoint_psi_cov_factor = vec![[0.0; 2]; p_resp];
        let mut endpoint_psi_psi_factor = vec![[0.0; 2]; p_resp];
        let mut h_factor = vec![0.0; p_resp];
        let mut hp_factor = vec![0.0; p_resp];
        let mut hpsi_cov_factor = vec![0.0; p_resp];
        let mut hppsi_cov_factor = vec![0.0; p_resp];
        let mut hpsi_psi_factor = vec![0.0; p_resp];
        let mut hppsi_psi_factor = vec![0.0; p_resp];

        for i in 0..n {
            let cov_row = cov.row(i);
            let psi_row = cov_psi.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hi = h[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let q = row_quantities.endpoint_q[i];
            let gamma_row = row_quantities.gamma.row(i);

            gamma.fill(0.0);
            gamma_psi.fill(0.0);
            for k in 0..p_resp {
                gamma[k] = gamma_row[k];
                gamma_psi[k] = beta_mat.row(k).dot(&psi_row);
            }

            let mut h_psi = rv[0] * gamma_psi[0];
            let mut hp_psi = rd[0] * gamma_psi[0];
            for k in 1..p_resp {
                h_psi += 2.0 * rv[k] * gamma[k] * gamma_psi[k];
                hp_psi += 2.0 * rd[k] * gamma[k] * gamma_psi[k];
            }

            let mut endpoint_psi = [0.0; 2];
            endpoint_factor.fill([0.0; 2]);
            endpoint_psi_cov_factor.fill([0.0; 2]);
            endpoint_psi_psi_factor.fill([0.0; 2]);
            for e in 0..2 {
                let basis = endpoint_basis[e];
                endpoint_psi[e] = basis[0] * gamma_psi[0];
                endpoint_factor[0][e] = basis[0];
                endpoint_psi_psi_factor[0][e] = basis[0];
                for k in 1..p_resp {
                    endpoint_psi[e] += 2.0 * basis[k] * gamma[k] * gamma_psi[k];
                    endpoint_factor[k][e] = 2.0 * basis[k] * gamma[k];
                    endpoint_psi_cov_factor[k][e] = 2.0 * basis[k] * gamma_psi[k];
                    endpoint_psi_psi_factor[k][e] = 2.0 * basis[k] * gamma[k];
                }
            }

            objective_psi +=
                wi * (hi * h_psi - hp_psi * inv_hp + endpoint_chain_first(&q, endpoint_psi));

            h_factor.fill(0.0);
            hp_factor.fill(0.0);
            hpsi_cov_factor.fill(0.0);
            hppsi_cov_factor.fill(0.0);
            hpsi_psi_factor.fill(0.0);
            hppsi_psi_factor.fill(0.0);
            h_factor[0] = rv[0];
            hp_factor[0] = rd[0];
            hpsi_psi_factor[0] = rv[0];
            hppsi_psi_factor[0] = rd[0];
            for k in 1..p_resp {
                h_factor[k] = 2.0 * rv[k] * gamma[k];
                hp_factor[k] = 2.0 * rd[k] * gamma[k];
                hpsi_cov_factor[k] = 2.0 * rv[k] * gamma_psi[k];
                hppsi_cov_factor[k] = 2.0 * rd[k] * gamma_psi[k];
                hpsi_psi_factor[k] = 2.0 * rv[k] * gamma[k];
                hppsi_psi_factor[k] = 2.0 * rd[k] * gamma[k];
            }

            for k in 0..p_resp {
                for c in 0..p_cov {
                    let idx = k * p_cov + c;
                    let h_a = h_factor[k] * cov_row[c];
                    let hp_a = hp_factor[k] * cov_row[c];
                    let hpsi_a = hpsi_cov_factor[k] * cov_row[c] + hpsi_psi_factor[k] * psi_row[c];
                    let hppsi_a =
                        hppsi_cov_factor[k] * cov_row[c] + hppsi_psi_factor[k] * psi_row[c];
                    let endpoint_a = [
                        endpoint_factor[k][0] * cov_row[c],
                        endpoint_factor[k][1] * cov_row[c],
                    ];
                    let endpoint_psi_a = [
                        endpoint_psi_cov_factor[k][0] * cov_row[c]
                            + endpoint_psi_psi_factor[k][0] * psi_row[c],
                        endpoint_psi_cov_factor[k][1] * cov_row[c]
                            + endpoint_psi_psi_factor[k][1] * psi_row[c],
                    ];
                    score_psi[idx] += wi
                        * (h_a * h_psi + hi * hpsi_a - hppsi_a * inv_hp
                            + hp_psi * hp_a * inv_hp_sq
                            + endpoint_chain_second(&q, endpoint_psi, endpoint_a, endpoint_psi_a));
                }
            }
        }

        let hessian_psi_operator: Arc<dyn HyperOperator> =
            Arc::new(TransformationNormalPsiHessianOperator::new(
                Arc::new(self.clone()),
                beta.clone(),
                Arc::clone(&op_arc),
                axis,
                Arc::clone(&row_quantities.gamma),
                Arc::clone(&row_quantities.h),
                Arc::clone(&row_quantities.h_prime),
                Arc::clone(&row_quantities.endpoint_q),
            ));

        Ok(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi: Array2::zeros((0, 0)),
            hessian_psi_operator: Some(hessian_psi_operator),
        })
    }

    fn scop_psi_hessian_apply_from_operator(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        op: &TensorKroneckerPsiOperator,
        axis: usize,
        direction: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let cov = self
            .covariate_dense_arc()
            .map_err(|e| format!("SCOP psi Hessian apply requires cached covariate design: {e}"))?;
        let cov_psi = op
            .materialize_cov_first_axis(axis)
            .map_err(|e| format!("SCOP psi Hessian apply materialize_cov_first failed: {e}"))?;
        self.scop_psi_hessian_apply_from_operator_with_cov(
            beta,
            row_quantities,
            axis,
            &cov,
            &cov_psi,
            direction,
        )
    }

    fn scop_psi_hessian_apply_from_operator_with_cov(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        axis: usize,
        cov: &Array2<f64>,
        cov_psi: &Array2<f64>,
        direction: &Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if cov.nrows() != n || cov.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi Hessian apply covariate shape {}x{} != expected {}x{}",
                    cov.nrows(),
                    cov.ncols(),
                    n,
                    p_cov
                ),
            }
            .into());
        }
        if beta.len() != p_total || direction.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian apply length mismatch: beta={}, direction={}, expected={p_total}",
                beta.len(),
                direction.len()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi Hessian apply beta reshape failed: {e}"))?;
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi Hessian apply direction reshape failed: {e}"))?;
        if cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian apply covariate derivative shape {}x{} for axis {axis} != expected {}x{}",
                cov_psi.nrows(),
                cov_psi.ncols(),
                n,
                p_cov
            ) }.into());
        }

        let weights = self.effective_weights();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];
        let mut out = Array1::<f64>::zeros(p_total);
        let mut gamma = vec![0.0; p_resp];
        let mut gamma_dir = vec![0.0; p_resp];
        let mut gamma_psi = vec![0.0; p_resp];
        let mut gamma_psi_dir = vec![0.0; p_resp];
        let mut endpoint_factor = vec![[0.0; 2]; p_resp];
        let mut endpoint_factor_dir = vec![[0.0; 2]; p_resp];
        let mut endpoint_psi_cov_factor = vec![[0.0; 2]; p_resp];
        let mut endpoint_psi_psi_factor = vec![[0.0; 2]; p_resp];
        let mut endpoint_psi_cov_factor_dir = vec![[0.0; 2]; p_resp];
        let mut endpoint_psi_psi_factor_dir = vec![[0.0; 2]; p_resp];
        let mut h_factor = vec![0.0; p_resp];
        let mut hp_factor = vec![0.0; p_resp];
        let mut h_factor_dir = vec![0.0; p_resp];
        let mut hp_factor_dir = vec![0.0; p_resp];
        let mut hpsi_cov_factor = vec![0.0; p_resp];
        let mut hppsi_cov_factor = vec![0.0; p_resp];
        let mut hpsi_psi_factor = vec![0.0; p_resp];
        let mut hppsi_psi_factor = vec![0.0; p_resp];
        let mut hpsi_cov_factor_dir = vec![0.0; p_resp];
        let mut hppsi_cov_factor_dir = vec![0.0; p_resp];
        let mut hpsi_psi_factor_dir = vec![0.0; p_resp];
        let mut hppsi_psi_factor_dir = vec![0.0; p_resp];

        for i in 0..n {
            let cov_row = cov.row(i);
            let psi_row = cov_psi.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hi = h[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;
            let q = row_quantities.endpoint_q[i];
            let gamma_row = row_quantities.gamma.row(i);

            for k in 0..p_resp {
                gamma[k] = gamma_row[k];
                gamma_dir[k] = dir_mat.row(k).dot(&cov_row);
                gamma_psi[k] = beta_mat.row(k).dot(&psi_row);
                gamma_psi_dir[k] = dir_mat.row(k).dot(&psi_row);
            }

            let mut h_dir = rv[0] * gamma_dir[0];
            let mut hp_dir = rd[0] * gamma_dir[0];
            let mut h_psi = rv[0] * gamma_psi[0];
            let mut hp_psi = rd[0] * gamma_psi[0];
            let mut h_psi_dir = rv[0] * gamma_psi_dir[0];
            let mut hp_psi_dir = rd[0] * gamma_psi_dir[0];
            for k in 1..p_resp {
                h_dir += 2.0 * rv[k] * gamma[k] * gamma_dir[k];
                hp_dir += 2.0 * rd[k] * gamma[k] * gamma_dir[k];
                h_psi += 2.0 * rv[k] * gamma[k] * gamma_psi[k];
                hp_psi += 2.0 * rd[k] * gamma[k] * gamma_psi[k];
                h_psi_dir +=
                    2.0 * rv[k] * (gamma_dir[k] * gamma_psi[k] + gamma[k] * gamma_psi_dir[k]);
                hp_psi_dir +=
                    2.0 * rd[k] * (gamma_dir[k] * gamma_psi[k] + gamma[k] * gamma_psi_dir[k]);
            }
            let d_inv_hp = -hp_dir * inv_hp_sq;
            let d_inv_hp_sq = -2.0 * hp_dir * inv_hp_cu;

            let mut endpoint_psi = [0.0; 2];
            let mut endpoint_dir = [0.0; 2];
            let mut endpoint_psi_dir = [0.0; 2];
            endpoint_factor.fill([0.0; 2]);
            endpoint_factor_dir.fill([0.0; 2]);
            endpoint_psi_cov_factor.fill([0.0; 2]);
            endpoint_psi_psi_factor.fill([0.0; 2]);
            endpoint_psi_cov_factor_dir.fill([0.0; 2]);
            endpoint_psi_psi_factor_dir.fill([0.0; 2]);
            for e in 0..2 {
                let basis = endpoint_basis[e];
                endpoint_psi[e] = basis[0] * gamma_psi[0];
                endpoint_dir[e] = basis[0] * gamma_dir[0];
                endpoint_psi_dir[e] = basis[0] * gamma_psi_dir[0];
                endpoint_factor[0][e] = basis[0];
                endpoint_psi_psi_factor[0][e] = basis[0];
                for k in 1..p_resp {
                    endpoint_psi[e] += 2.0 * basis[k] * gamma[k] * gamma_psi[k];
                    endpoint_dir[e] += 2.0 * basis[k] * gamma[k] * gamma_dir[k];
                    endpoint_psi_dir[e] += 2.0
                        * basis[k]
                        * (gamma_dir[k] * gamma_psi[k] + gamma[k] * gamma_psi_dir[k]);
                    endpoint_factor[k][e] = 2.0 * basis[k] * gamma[k];
                    endpoint_factor_dir[k][e] = 2.0 * basis[k] * gamma_dir[k];
                    endpoint_psi_cov_factor[k][e] = 2.0 * basis[k] * gamma_psi[k];
                    endpoint_psi_psi_factor[k][e] = 2.0 * basis[k] * gamma[k];
                    endpoint_psi_cov_factor_dir[k][e] = 2.0 * basis[k] * gamma_psi_dir[k];
                    endpoint_psi_psi_factor_dir[k][e] = 2.0 * basis[k] * gamma_dir[k];
                }
            }

            h_factor.fill(0.0);
            hp_factor.fill(0.0);
            h_factor_dir.fill(0.0);
            hp_factor_dir.fill(0.0);
            hpsi_cov_factor.fill(0.0);
            hppsi_cov_factor.fill(0.0);
            hpsi_psi_factor.fill(0.0);
            hppsi_psi_factor.fill(0.0);
            hpsi_cov_factor_dir.fill(0.0);
            hppsi_cov_factor_dir.fill(0.0);
            hpsi_psi_factor_dir.fill(0.0);
            hppsi_psi_factor_dir.fill(0.0);
            h_factor[0] = rv[0];
            hp_factor[0] = rd[0];
            hpsi_psi_factor[0] = rv[0];
            hppsi_psi_factor[0] = rd[0];
            for k in 1..p_resp {
                h_factor[k] = 2.0 * rv[k] * gamma[k];
                hp_factor[k] = 2.0 * rd[k] * gamma[k];
                h_factor_dir[k] = 2.0 * rv[k] * gamma_dir[k];
                hp_factor_dir[k] = 2.0 * rd[k] * gamma_dir[k];
                hpsi_cov_factor[k] = 2.0 * rv[k] * gamma_psi[k];
                hppsi_cov_factor[k] = 2.0 * rd[k] * gamma_psi[k];
                hpsi_psi_factor[k] = 2.0 * rv[k] * gamma[k];
                hppsi_psi_factor[k] = 2.0 * rd[k] * gamma[k];
                hpsi_cov_factor_dir[k] = 2.0 * rv[k] * gamma_psi_dir[k];
                hppsi_cov_factor_dir[k] = 2.0 * rd[k] * gamma_psi_dir[k];
                hpsi_psi_factor_dir[k] = 2.0 * rv[k] * gamma_dir[k];
                hppsi_psi_factor_dir[k] = 2.0 * rd[k] * gamma_dir[k];
            }

            for k in 0..p_resp {
                for c in 0..p_cov {
                    let idx = k * p_cov + c;
                    let h_a = h_factor[k] * cov_row[c];
                    let hp_a = hp_factor[k] * cov_row[c];
                    let h_a_dir = h_factor_dir[k] * cov_row[c];
                    let hp_a_dir = hp_factor_dir[k] * cov_row[c];
                    let hpsi_a = hpsi_cov_factor[k] * cov_row[c] + hpsi_psi_factor[k] * psi_row[c];
                    let hppsi_a =
                        hppsi_cov_factor[k] * cov_row[c] + hppsi_psi_factor[k] * psi_row[c];
                    let hpsi_a_dir =
                        hpsi_cov_factor_dir[k] * cov_row[c] + hpsi_psi_factor_dir[k] * psi_row[c];
                    let hppsi_a_dir =
                        hppsi_cov_factor_dir[k] * cov_row[c] + hppsi_psi_factor_dir[k] * psi_row[c];
                    let endpoint_a = [
                        endpoint_factor[k][0] * cov_row[c],
                        endpoint_factor[k][1] * cov_row[c],
                    ];
                    let endpoint_a_dir = [
                        endpoint_factor_dir[k][0] * cov_row[c],
                        endpoint_factor_dir[k][1] * cov_row[c],
                    ];
                    let endpoint_psi_a = [
                        endpoint_psi_cov_factor[k][0] * cov_row[c]
                            + endpoint_psi_psi_factor[k][0] * psi_row[c],
                        endpoint_psi_cov_factor[k][1] * cov_row[c]
                            + endpoint_psi_psi_factor[k][1] * psi_row[c],
                    ];
                    let endpoint_psi_a_dir = [
                        endpoint_psi_cov_factor_dir[k][0] * cov_row[c]
                            + endpoint_psi_psi_factor_dir[k][0] * psi_row[c],
                        endpoint_psi_cov_factor_dir[k][1] * cov_row[c]
                            + endpoint_psi_psi_factor_dir[k][1] * psi_row[c],
                    ];
                    let value =
                        h_a_dir * h_psi + h_a * h_psi_dir + h_dir * hpsi_a + hi * hpsi_a_dir
                            - hppsi_a_dir * inv_hp
                            - hppsi_a * d_inv_hp
                            + hp_psi_dir * hp_a * inv_hp_sq
                            + hp_psi * hp_a_dir * inv_hp_sq
                            + hp_psi * hp_a * d_inv_hp_sq
                            + endpoint_chain_third(
                                &q,
                                endpoint_psi,
                                endpoint_a,
                                endpoint_dir,
                                endpoint_psi_a,
                                endpoint_psi_dir,
                                endpoint_a_dir,
                                endpoint_psi_a_dir,
                            );
                    out[idx] += wi * value;
                }
            }
        }

        Ok(out)
    }

    fn scop_psi_hessian_hvp_mat_from_cov(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        axis: usize,
        cov: &Array2<f64>,
        cov_psi: &Array2<f64>,
        factor: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let rank = factor.ncols();
        if cov.nrows() != n || cov.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi Hessian batched apply covariate shape {}x{} != expected {}x{}",
                    cov.nrows(),
                    cov.ncols(),
                    n,
                    p_cov
                ),
            }
            .into());
        }
        if cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian batched apply covariate derivative shape {}x{} for axis {axis} != expected {}x{}",
                cov_psi.nrows(),
                cov_psi.ncols(),
                n,
                p_cov
            ) }.into());
        }
        if beta.len() != p_total || factor.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian batched apply length mismatch: beta={}, factor_rows={}, expected={p_total}",
                beta.len(),
                factor.nrows()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi Hessian batched apply beta reshape failed: {e}"))?;
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        struct PsiBatchedAccum {
            hvp: Array2<f64>,
            gamma: Vec<f64>,
            gamma_psi: Vec<f64>,
            gamma_dir: Vec<f64>,
            gamma_psi_dir: Vec<f64>,
            h_dir: Vec<f64>,
            hp_dir: Vec<f64>,
            h_psi_dir: Vec<f64>,
            hp_psi_dir: Vec<f64>,
            endpoint_dir: Vec<[f64; 2]>,
            endpoint_psi_dir: Vec<[f64; 2]>,
        }

        impl PsiBatchedAccum {
            fn new(p_total: usize, p_resp: usize, rank: usize) -> Self {
                let projected_len = p_resp * rank;
                Self {
                    hvp: Array2::<f64>::zeros((p_total, rank)),
                    gamma: vec![0.0; p_resp],
                    gamma_psi: vec![0.0; p_resp],
                    gamma_dir: vec![0.0; projected_len],
                    gamma_psi_dir: vec![0.0; projected_len],
                    h_dir: vec![0.0; rank],
                    hp_dir: vec![0.0; rank],
                    h_psi_dir: vec![0.0; rank],
                    hp_psi_dir: vec![0.0; rank],
                    endpoint_dir: vec![[0.0; 2]; rank],
                    endpoint_psi_dir: vec![[0.0; 2]; rank],
                }
            }

            fn merge(mut self, rhs: Self) -> Self {
                self.hvp += &rhs.hvp;
                self
            }
        }

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let weights = self.effective_weights();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let accum = (0..n)
            .into_par_iter()
            .fold(
                || PsiBatchedAccum::new(p_total, p_resp, rank),
                |mut acc, i| {
                    let cov_row = cov.row(i);
                    let psi_row = cov_psi.row(i);
                    let rv = self.response_val_basis.row(i);
                    let rd = self.response_deriv_basis.row(i);
                    let wi = weights[i];
                    let hi = h[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let q = row_quantities.endpoint_q[i];
                    let gamma_row = row_quantities.gamma.row(i);

                    for k in 0..p_resp {
                        acc.gamma[k] = gamma_row[k];
                        acc.gamma_psi[k] = beta_mat.row(k).dot(&psi_row);
                    }

                    acc.gamma_dir.fill(0.0);
                    acc.gamma_psi_dir.fill(0.0);
                    for k in 0..p_resp {
                        let factor_row_base = k * p_cov;
                        let projected_base = k * rank;
                        for cidx in 0..p_cov {
                            let factor_row = factor_row_base + cidx;
                            let cov_v = cov_row[cidx];
                            let psi_v = psi_row[cidx];
                            for col in 0..rank {
                                let coeff = factor[[factor_row, col]];
                                let idx = projected_base + col;
                                acc.gamma_dir[idx] += coeff * cov_v;
                                acc.gamma_psi_dir[idx] += coeff * psi_v;
                            }
                        }
                    }

                    let mut h_psi = rv[0] * acc.gamma_psi[0];
                    let mut hp_psi = rd[0] * acc.gamma_psi[0];
                    for k in 1..p_resp {
                        h_psi += 2.0 * rv[k] * acc.gamma[k] * acc.gamma_psi[k];
                        hp_psi += 2.0 * rd[k] * acc.gamma[k] * acc.gamma_psi[k];
                    }

                    let mut endpoint_psi = [0.0; 2];
                    for e in 0..2 {
                        let basis = endpoint_basis[e];
                        endpoint_psi[e] = basis[0] * acc.gamma_psi[0];
                        for k in 1..p_resp {
                            endpoint_psi[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_psi[k];
                        }
                    }

                    for col in 0..rank {
                        acc.h_dir[col] = rv[0] * acc.gamma_dir[col];
                        acc.hp_dir[col] = rd[0] * acc.gamma_dir[col];
                        acc.h_psi_dir[col] = rv[0] * acc.gamma_psi_dir[col];
                        acc.hp_psi_dir[col] = rd[0] * acc.gamma_psi_dir[col];
                        acc.endpoint_dir[col] = [
                            endpoint_basis[0][0] * acc.gamma_dir[col],
                            endpoint_basis[1][0] * acc.gamma_dir[col],
                        ];
                        acc.endpoint_psi_dir[col] = [
                            endpoint_basis[0][0] * acc.gamma_psi_dir[col],
                            endpoint_basis[1][0] * acc.gamma_psi_dir[col],
                        ];
                    }
                    for k in 1..p_resp {
                        let g = acc.gamma[k];
                        let g_psi = acc.gamma_psi[k];
                        for col in 0..rank {
                            let idx = k * rank + col;
                            let g_dir = acc.gamma_dir[idx];
                            let g_psi_dir = acc.gamma_psi_dir[idx];
                            acc.h_dir[col] += 2.0 * rv[k] * g * g_dir;
                            acc.hp_dir[col] += 2.0 * rd[k] * g * g_dir;
                            acc.h_psi_dir[col] += 2.0 * rv[k] * (g_dir * g_psi + g * g_psi_dir);
                            acc.hp_psi_dir[col] += 2.0 * rd[k] * (g_dir * g_psi + g * g_psi_dir);
                            for e in 0..2 {
                                let basis = endpoint_basis[e];
                                acc.endpoint_dir[col][e] += 2.0 * basis[k] * g * g_dir;
                                acc.endpoint_psi_dir[col][e] +=
                                    2.0 * basis[k] * (g_dir * g_psi + g * g_psi_dir);
                            }
                        }
                    }

                    for k in 0..p_resp {
                        let offset = k * p_cov;
                        let rvk = rv[k];
                        let rdk = rd[k];
                        let g = acc.gamma[k];
                        let g_psi = acc.gamma_psi[k];
                        let h_factor = if k == 0 { rvk } else { 2.0 * rvk * g };
                        let hp_factor = if k == 0 { rdk } else { 2.0 * rdk * g };
                        let hpsi_cov_factor = if k == 0 { 0.0 } else { 2.0 * rvk * g_psi };
                        let hppsi_cov_factor = if k == 0 { 0.0 } else { 2.0 * rdk * g_psi };
                        let hpsi_psi_factor = if k == 0 { rvk } else { 2.0 * rvk * g };
                        let hppsi_psi_factor = if k == 0 { rdk } else { 2.0 * rdk * g };
                        let endpoint_factor = [
                            if k == 0 {
                                endpoint_basis[0][k]
                            } else {
                                2.0 * endpoint_basis[0][k] * g
                            },
                            if k == 0 {
                                endpoint_basis[1][k]
                            } else {
                                2.0 * endpoint_basis[1][k] * g
                            },
                        ];
                        let endpoint_psi_cov_factor = [
                            if k == 0 {
                                0.0
                            } else {
                                2.0 * endpoint_basis[0][k] * g_psi
                            },
                            if k == 0 {
                                0.0
                            } else {
                                2.0 * endpoint_basis[1][k] * g_psi
                            },
                        ];
                        let endpoint_psi_psi_factor = [
                            if k == 0 {
                                endpoint_basis[0][k]
                            } else {
                                2.0 * endpoint_basis[0][k] * g
                            },
                            if k == 0 {
                                endpoint_basis[1][k]
                            } else {
                                2.0 * endpoint_basis[1][k] * g
                            },
                        ];
                        for cidx in 0..p_cov {
                            let c = cov_row[cidx];
                            let psi = psi_row[cidx];
                            let h_a = h_factor * c;
                            let hp_a = hp_factor * c;
                            let hpsi_a = hpsi_cov_factor * c + hpsi_psi_factor * psi;
                            let hppsi_a = hppsi_cov_factor * c + hppsi_psi_factor * psi;
                            let endpoint_a = [endpoint_factor[0] * c, endpoint_factor[1] * c];
                            let endpoint_psi_a = [
                                endpoint_psi_cov_factor[0] * c + endpoint_psi_psi_factor[0] * psi,
                                endpoint_psi_cov_factor[1] * c + endpoint_psi_psi_factor[1] * psi,
                            ];
                            let out_idx = offset + cidx;
                            for col in 0..rank {
                                let projected_idx = k * rank + col;
                                let g_dir = acc.gamma_dir[projected_idx];
                                let g_psi_dir = acc.gamma_psi_dir[projected_idx];
                                let h_factor_dir = if k == 0 { 0.0 } else { 2.0 * rvk * g_dir };
                                let hp_factor_dir = if k == 0 { 0.0 } else { 2.0 * rdk * g_dir };
                                let hpsi_cov_factor_dir =
                                    if k == 0 { 0.0 } else { 2.0 * rvk * g_psi_dir };
                                let hppsi_cov_factor_dir =
                                    if k == 0 { 0.0 } else { 2.0 * rdk * g_psi_dir };
                                let hpsi_psi_factor_dir =
                                    if k == 0 { 0.0 } else { 2.0 * rvk * g_dir };
                                let hppsi_psi_factor_dir =
                                    if k == 0 { 0.0 } else { 2.0 * rdk * g_dir };
                                let h_a_dir = h_factor_dir * c;
                                let hp_a_dir = hp_factor_dir * c;
                                let hpsi_a_dir =
                                    hpsi_cov_factor_dir * c + hpsi_psi_factor_dir * psi;
                                let hppsi_a_dir =
                                    hppsi_cov_factor_dir * c + hppsi_psi_factor_dir * psi;
                                let endpoint_factor_dir = [
                                    if k == 0 {
                                        0.0
                                    } else {
                                        2.0 * endpoint_basis[0][k] * g_dir
                                    },
                                    if k == 0 {
                                        0.0
                                    } else {
                                        2.0 * endpoint_basis[1][k] * g_dir
                                    },
                                ];
                                let endpoint_psi_cov_factor_dir = [
                                    if k == 0 {
                                        0.0
                                    } else {
                                        2.0 * endpoint_basis[0][k] * g_psi_dir
                                    },
                                    if k == 0 {
                                        0.0
                                    } else {
                                        2.0 * endpoint_basis[1][k] * g_psi_dir
                                    },
                                ];
                                let endpoint_psi_psi_factor_dir = [
                                    if k == 0 {
                                        0.0
                                    } else {
                                        2.0 * endpoint_basis[0][k] * g_dir
                                    },
                                    if k == 0 {
                                        0.0
                                    } else {
                                        2.0 * endpoint_basis[1][k] * g_dir
                                    },
                                ];
                                let endpoint_a_dir =
                                    [endpoint_factor_dir[0] * c, endpoint_factor_dir[1] * c];
                                let endpoint_psi_a_dir = [
                                    endpoint_psi_cov_factor_dir[0] * c
                                        + endpoint_psi_psi_factor_dir[0] * psi,
                                    endpoint_psi_cov_factor_dir[1] * c
                                        + endpoint_psi_psi_factor_dir[1] * psi,
                                ];
                                let d_inv_hp = -acc.hp_dir[col] * inv_hp_sq;
                                let d_inv_hp_sq = -2.0 * acc.hp_dir[col] * inv_hp_cu;
                                let value = h_a_dir * h_psi
                                    + h_a * acc.h_psi_dir[col]
                                    + acc.h_dir[col] * hpsi_a
                                    + hi * hpsi_a_dir
                                    - hppsi_a_dir * inv_hp
                                    - hppsi_a * d_inv_hp
                                    + acc.hp_psi_dir[col] * hp_a * inv_hp_sq
                                    + hp_psi * hp_a_dir * inv_hp_sq
                                    + hp_psi * hp_a * d_inv_hp_sq
                                    + endpoint_chain_third(
                                        &q,
                                        endpoint_psi,
                                        endpoint_a,
                                        acc.endpoint_dir[col],
                                        endpoint_psi_a,
                                        acc.endpoint_psi_dir[col],
                                        endpoint_a_dir,
                                        endpoint_psi_a_dir,
                                    );
                                acc.hvp[[out_idx, col]] += wi * value;
                            }
                        }
                    }
                    acc
                },
            )
            .reduce(
                || PsiBatchedAccum::new(p_total, p_resp, rank),
                |left, right| left.merge(right),
            );

        Ok(accum.hvp)
    }

    fn scop_psi_hessian_trace_factor_from_cov(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        axis: usize,
        cov: &Array2<f64>,
        cov_psi: &Array2<f64>,
        factor: ArrayView2<'_, f64>,
    ) -> Result<f64, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let rank = factor.ncols();
        if cov.nrows() != n || cov.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi Hessian projected trace covariate shape {}x{} != expected {}x{}",
                    cov.nrows(),
                    cov.ncols(),
                    n,
                    p_cov
                ),
            }
            .into());
        }
        if cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian projected trace covariate derivative shape {}x{} for axis {axis} != expected {}x{}",
                cov_psi.nrows(),
                cov_psi.ncols(),
                n,
                p_cov
            ) }.into());
        }
        if beta.len() != p_total || factor.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian projected trace length mismatch: beta={}, factor_rows={}, expected={p_total}",
                beta.len(),
                factor.nrows()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi Hessian projected trace beta reshape failed: {e}"))?;
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        struct PsiTraceAccum {
            value: f64,
            gamma: Vec<f64>,
            gamma_psi: Vec<f64>,
            gamma_dir: Vec<f64>,
            gamma_psi_dir: Vec<f64>,
            h_dir: Vec<f64>,
            hp_dir: Vec<f64>,
            h_vv: Vec<f64>,
            hp_vv: Vec<f64>,
            h_psi_dir: Vec<f64>,
            hp_psi_dir: Vec<f64>,
            h_psi_vv: Vec<f64>,
            hp_psi_vv: Vec<f64>,
            endpoint_dir: Vec<[f64; 2]>,
            endpoint_psi_dir: Vec<[f64; 2]>,
            endpoint_vv: Vec<[f64; 2]>,
            endpoint_psi_vv: Vec<[f64; 2]>,
        }

        impl PsiTraceAccum {
            fn new(p_resp: usize, rank: usize) -> Self {
                let projected_len = p_resp * rank;
                Self {
                    value: 0.0,
                    gamma: vec![0.0; p_resp],
                    gamma_psi: vec![0.0; p_resp],
                    gamma_dir: vec![0.0; projected_len],
                    gamma_psi_dir: vec![0.0; projected_len],
                    h_dir: vec![0.0; rank],
                    hp_dir: vec![0.0; rank],
                    h_vv: vec![0.0; rank],
                    hp_vv: vec![0.0; rank],
                    h_psi_dir: vec![0.0; rank],
                    hp_psi_dir: vec![0.0; rank],
                    h_psi_vv: vec![0.0; rank],
                    hp_psi_vv: vec![0.0; rank],
                    endpoint_dir: vec![[0.0; 2]; rank],
                    endpoint_psi_dir: vec![[0.0; 2]; rank],
                    endpoint_vv: vec![[0.0; 2]; rank],
                    endpoint_psi_vv: vec![[0.0; 2]; rank],
                }
            }

            fn merge(mut self, rhs: Self) -> Self {
                self.value += rhs.value;
                self
            }
        }

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let weights = self.effective_weights();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let accum = (0..n)
            .into_par_iter()
            .fold(
                || PsiTraceAccum::new(p_resp, rank),
                |mut acc, i| {
                    let cov_row = cov.row(i);
                    let psi_row = cov_psi.row(i);
                    let rv = self.response_val_basis.row(i);
                    let rd = self.response_deriv_basis.row(i);
                    let wi = weights[i];
                    let hi = h[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let q = row_quantities.endpoint_q[i];
                    let gamma_row = row_quantities.gamma.row(i);

                    for k in 0..p_resp {
                        acc.gamma[k] = gamma_row[k];
                        acc.gamma_psi[k] = beta_mat.row(k).dot(&psi_row);
                    }

                    acc.gamma_dir.fill(0.0);
                    acc.gamma_psi_dir.fill(0.0);
                    for k in 0..p_resp {
                        let factor_row_base = k * p_cov;
                        let projected_base = k * rank;
                        for cidx in 0..p_cov {
                            let factor_row = factor_row_base + cidx;
                            let cov_v = cov_row[cidx];
                            let psi_v = psi_row[cidx];
                            for col in 0..rank {
                                let coeff = factor[[factor_row, col]];
                                let idx = projected_base + col;
                                acc.gamma_dir[idx] += coeff * cov_v;
                                acc.gamma_psi_dir[idx] += coeff * psi_v;
                            }
                        }
                    }

                    let mut h_psi = rv[0] * acc.gamma_psi[0];
                    let mut hp_psi = rd[0] * acc.gamma_psi[0];
                    for k in 1..p_resp {
                        h_psi += 2.0 * rv[k] * acc.gamma[k] * acc.gamma_psi[k];
                        hp_psi += 2.0 * rd[k] * acc.gamma[k] * acc.gamma_psi[k];
                    }

                    let mut endpoint_psi = [0.0; 2];
                    for e in 0..2 {
                        let basis = endpoint_basis[e];
                        endpoint_psi[e] = basis[0] * acc.gamma_psi[0];
                        for k in 1..p_resp {
                            endpoint_psi[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_psi[k];
                        }
                    }

                    for col in 0..rank {
                        acc.h_dir[col] = rv[0] * acc.gamma_dir[col];
                        acc.hp_dir[col] = rd[0] * acc.gamma_dir[col];
                        acc.h_vv[col] = 0.0;
                        acc.hp_vv[col] = 0.0;
                        acc.h_psi_dir[col] = rv[0] * acc.gamma_psi_dir[col];
                        acc.hp_psi_dir[col] = rd[0] * acc.gamma_psi_dir[col];
                        acc.h_psi_vv[col] = 0.0;
                        acc.hp_psi_vv[col] = 0.0;
                        acc.endpoint_dir[col] = [
                            endpoint_basis[0][0] * acc.gamma_dir[col],
                            endpoint_basis[1][0] * acc.gamma_dir[col],
                        ];
                        acc.endpoint_psi_dir[col] = [
                            endpoint_basis[0][0] * acc.gamma_psi_dir[col],
                            endpoint_basis[1][0] * acc.gamma_psi_dir[col],
                        ];
                        acc.endpoint_vv[col] = [0.0; 2];
                        acc.endpoint_psi_vv[col] = [0.0; 2];
                    }
                    for k in 1..p_resp {
                        let g = acc.gamma[k];
                        let g_psi = acc.gamma_psi[k];
                        for col in 0..rank {
                            let idx = k * rank + col;
                            let g_dir = acc.gamma_dir[idx];
                            let g_psi_dir = acc.gamma_psi_dir[idx];
                            acc.h_dir[col] += 2.0 * rv[k] * g * g_dir;
                            acc.hp_dir[col] += 2.0 * rd[k] * g * g_dir;
                            acc.h_vv[col] += 2.0 * rv[k] * g_dir * g_dir;
                            acc.hp_vv[col] += 2.0 * rd[k] * g_dir * g_dir;
                            acc.h_psi_dir[col] += 2.0 * rv[k] * (g_dir * g_psi + g * g_psi_dir);
                            acc.hp_psi_dir[col] += 2.0 * rd[k] * (g_dir * g_psi + g * g_psi_dir);
                            acc.h_psi_vv[col] += 4.0 * rv[k] * g_dir * g_psi_dir;
                            acc.hp_psi_vv[col] += 4.0 * rd[k] * g_dir * g_psi_dir;
                            for e in 0..2 {
                                let basis = endpoint_basis[e];
                                acc.endpoint_dir[col][e] += 2.0 * basis[k] * g * g_dir;
                                acc.endpoint_psi_dir[col][e] +=
                                    2.0 * basis[k] * (g_dir * g_psi + g * g_psi_dir);
                                acc.endpoint_vv[col][e] += 2.0 * basis[k] * g_dir * g_dir;
                                acc.endpoint_psi_vv[col][e] += 4.0 * basis[k] * g_dir * g_psi_dir;
                            }
                        }
                    }

                    for col in 0..rank {
                        let barrier = -acc.hp_psi_vv[col] * inv_hp
                            + 2.0 * acc.hp_psi_dir[col] * acc.hp_dir[col] * inv_hp_sq
                            + hp_psi * acc.hp_vv[col] * inv_hp_sq
                            - 2.0 * hp_psi * acc.hp_dir[col] * acc.hp_dir[col] * inv_hp_sq * inv_hp;
                        acc.value += wi
                            * (acc.h_vv[col] * h_psi
                                + 2.0 * acc.h_dir[col] * acc.h_psi_dir[col]
                                + hi * acc.h_psi_vv[col]
                                + barrier
                                + endpoint_chain_third(
                                    &q,
                                    endpoint_psi,
                                    acc.endpoint_dir[col],
                                    acc.endpoint_dir[col],
                                    acc.endpoint_psi_dir[col],
                                    acc.endpoint_psi_dir[col],
                                    acc.endpoint_vv[col],
                                    acc.endpoint_psi_vv[col],
                                ));
                    }
                    acc
                },
            )
            .reduce(
                || PsiTraceAccum::new(p_resp, rank),
                |left, right| left.merge(right),
            );

        Ok(accum.value)
    }

    /// Single-pass fused projected-trace evaluation across every ψ axis.
    ///
    /// Computes the projected trace `tr(factor^T B_e factor)` for every axis
    /// `e` in `0..cov_psi_per_axis.len()` in ONE row-streaming parallel pass.
    /// Per-row state that is independent of the ψ axis (γ, γ_dir, h_dir,
    /// hp_dir, h_vv, hp_vv, endpoint_dir, endpoint_vv) is computed exactly
    /// once per row and reused across every axis; only the ψ-row-driven
    /// axis-DEP state (γ_psi, γ_psi_dir, h_psi, hp_psi, endpoint_psi, the
    /// `*_psi_dir` / `*_psi_vv` buffers and the per-axis trace contribution)
    /// is recomputed inside the per-row axis loop.
    ///
    /// The arithmetic is reorganised but bit-identical to running
    /// [`scop_psi_hessian_trace_factor_from_cov`] once per axis: the same
    /// scalar accumulations, evaluated in the same order per row, with the
    /// rayon row reduction summing axis-INDEP contributions exactly once per
    /// row and axis-DEP contributions exactly once per `(row, axis)`.
    ///
    /// Returns a `Vec<f64>` of length `cov_psi_per_axis.len()` with the trace
    /// of each axis's projected ψ-Hessian.
    fn scop_psi_hessian_trace_factor_all_axes_chunk_from_cov(
        &self,
        beta: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        row_start: usize,
        cov: ArrayView2<'_, f64>,
        cov_psi_per_axis: &[ArrayView2<'_, f64>],
        factor: ArrayView2<'_, f64>,
    ) -> Result<Vec<f64>, String> {
        let total_n = self.response_val_basis.nrows();
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let rank = factor.ncols();
        let n_psi = cov_psi_per_axis.len();
        if n_psi == 0 {
            return Ok(Vec::new());
        }
        if row_start > total_n || row_start + n > total_n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian projected trace row window [{row_start}, {}) exceeds n={total_n}",
                row_start + n
            ) }.into());
        }
        if cov.nrows() != n || cov.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian projected trace covariate chunk shape {}x{} != expected {}x{}",
                cov.nrows(),
                cov.ncols(),
                n,
                p_cov
            ) }.into());
        }
        for (axis, cov_psi) in cov_psi_per_axis.iter().enumerate() {
            if cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
                return Err(TransformationNormalError::InvalidInput { reason: format!(
                    "SCOP psi Hessian projected trace covariate derivative chunk shape {}x{} for axis {axis} != expected {}x{}",
                    cov_psi.nrows(),
                    cov_psi.ncols(),
                    n,
                    p_cov
                ) }.into());
            }
        }
        if beta.len() != p_total || factor.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian projected trace length mismatch: beta={}, factor_rows={}, expected={p_total}",
                beta.len(),
                factor.nrows()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi Hessian projected trace beta reshape failed: {e}"))?;
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        struct PsiAllAxesTraceAccum {
            values: Vec<f64>,
            gamma: Vec<f64>,
            gamma_dir: Vec<f64>,
            h_dir: Vec<f64>,
            hp_dir: Vec<f64>,
            h_vv: Vec<f64>,
            hp_vv: Vec<f64>,
            endpoint_dir: Vec<[f64; 2]>,
            endpoint_vv: Vec<[f64; 2]>,
            gamma_psi: Vec<f64>,
            gamma_psi_dir: Vec<f64>,
            h_psi_dir: Vec<f64>,
            hp_psi_dir: Vec<f64>,
            h_psi_vv: Vec<f64>,
            hp_psi_vv: Vec<f64>,
            endpoint_psi_dir: Vec<[f64; 2]>,
            endpoint_psi_vv: Vec<[f64; 2]>,
        }

        impl PsiAllAxesTraceAccum {
            fn new(p_resp: usize, rank: usize, n_psi: usize) -> Self {
                let projected_len = p_resp * rank;
                Self {
                    values: vec![0.0; n_psi],
                    gamma: vec![0.0; p_resp],
                    gamma_dir: vec![0.0; projected_len],
                    h_dir: vec![0.0; rank],
                    hp_dir: vec![0.0; rank],
                    h_vv: vec![0.0; rank],
                    hp_vv: vec![0.0; rank],
                    endpoint_dir: vec![[0.0; 2]; rank],
                    endpoint_vv: vec![[0.0; 2]; rank],
                    gamma_psi: vec![0.0; p_resp],
                    gamma_psi_dir: vec![0.0; projected_len],
                    h_psi_dir: vec![0.0; rank],
                    hp_psi_dir: vec![0.0; rank],
                    h_psi_vv: vec![0.0; rank],
                    hp_psi_vv: vec![0.0; rank],
                    endpoint_psi_dir: vec![[0.0; 2]; rank],
                    endpoint_psi_vv: vec![[0.0; 2]; rank],
                }
            }

            fn merge(mut self, rhs: Self) -> Self {
                for (a, v) in rhs.values.into_iter().enumerate() {
                    self.values[a] += v;
                }
                self
            }
        }

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let weights = self.effective_weights();
        let h = row_quantities.h.as_ref();
        let h_prime = row_quantities.h_prime.as_ref();
        let accum = (0..n)
            .into_par_iter()
            .fold(
                || PsiAllAxesTraceAccum::new(p_resp, rank, n_psi),
                |mut acc, local_i| {
                    let i = row_start + local_i;
                    let cov_row = cov.row(local_i);
                    let rv = self.response_val_basis.row(i);
                    let rd = self.response_deriv_basis.row(i);
                    let wi = weights[i];
                    let hi = h[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let q = row_quantities.endpoint_q[i];
                    let gamma_row = row_quantities.gamma.row(i);

                    // ---- Axis-INDEP per-row state (computed exactly once) ----
                    for k in 0..p_resp {
                        acc.gamma[k] = gamma_row[k];
                    }

                    acc.gamma_dir.fill(0.0);
                    for k in 0..p_resp {
                        let factor_row_base = k * p_cov;
                        let projected_base = k * rank;
                        for cidx in 0..p_cov {
                            let factor_row = factor_row_base + cidx;
                            let cov_v = cov_row[cidx];
                            for col in 0..rank {
                                let coeff = factor[[factor_row, col]];
                                let idx = projected_base + col;
                                acc.gamma_dir[idx] += coeff * cov_v;
                            }
                        }
                    }

                    for col in 0..rank {
                        acc.h_dir[col] = rv[0] * acc.gamma_dir[col];
                        acc.hp_dir[col] = rd[0] * acc.gamma_dir[col];
                        acc.h_vv[col] = 0.0;
                        acc.hp_vv[col] = 0.0;
                        acc.endpoint_dir[col] = [
                            endpoint_basis[0][0] * acc.gamma_dir[col],
                            endpoint_basis[1][0] * acc.gamma_dir[col],
                        ];
                        acc.endpoint_vv[col] = [0.0; 2];
                    }
                    for k in 1..p_resp {
                        let g = acc.gamma[k];
                        for col in 0..rank {
                            let idx = k * rank + col;
                            let g_dir = acc.gamma_dir[idx];
                            acc.h_dir[col] += 2.0 * rv[k] * g * g_dir;
                            acc.hp_dir[col] += 2.0 * rd[k] * g * g_dir;
                            acc.h_vv[col] += 2.0 * rv[k] * g_dir * g_dir;
                            acc.hp_vv[col] += 2.0 * rd[k] * g_dir * g_dir;
                            for e in 0..2 {
                                let basis = endpoint_basis[e];
                                acc.endpoint_dir[col][e] += 2.0 * basis[k] * g * g_dir;
                                acc.endpoint_vv[col][e] += 2.0 * basis[k] * g_dir * g_dir;
                            }
                        }
                    }

                    // ---- Axis-DEP per-(row,axis) state ----
                    for axis_idx in 0..n_psi {
                        let psi_row = cov_psi_per_axis[axis_idx].row(local_i);

                        for k in 0..p_resp {
                            acc.gamma_psi[k] = beta_mat.row(k).dot(&psi_row);
                        }

                        acc.gamma_psi_dir.fill(0.0);
                        for k in 0..p_resp {
                            let factor_row_base = k * p_cov;
                            let projected_base = k * rank;
                            for cidx in 0..p_cov {
                                let factor_row = factor_row_base + cidx;
                                let psi_v = psi_row[cidx];
                                for col in 0..rank {
                                    let coeff = factor[[factor_row, col]];
                                    let idx = projected_base + col;
                                    acc.gamma_psi_dir[idx] += coeff * psi_v;
                                }
                            }
                        }

                        let mut h_psi = rv[0] * acc.gamma_psi[0];
                        let mut hp_psi = rd[0] * acc.gamma_psi[0];
                        for k in 1..p_resp {
                            h_psi += 2.0 * rv[k] * acc.gamma[k] * acc.gamma_psi[k];
                            hp_psi += 2.0 * rd[k] * acc.gamma[k] * acc.gamma_psi[k];
                        }

                        let mut endpoint_psi = [0.0; 2];
                        for e in 0..2 {
                            let basis = endpoint_basis[e];
                            endpoint_psi[e] = basis[0] * acc.gamma_psi[0];
                            for k in 1..p_resp {
                                endpoint_psi[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_psi[k];
                            }
                        }

                        for col in 0..rank {
                            acc.h_psi_dir[col] = rv[0] * acc.gamma_psi_dir[col];
                            acc.hp_psi_dir[col] = rd[0] * acc.gamma_psi_dir[col];
                            acc.h_psi_vv[col] = 0.0;
                            acc.hp_psi_vv[col] = 0.0;
                            acc.endpoint_psi_dir[col] = [
                                endpoint_basis[0][0] * acc.gamma_psi_dir[col],
                                endpoint_basis[1][0] * acc.gamma_psi_dir[col],
                            ];
                            acc.endpoint_psi_vv[col] = [0.0; 2];
                        }
                        for k in 1..p_resp {
                            let g = acc.gamma[k];
                            let g_psi = acc.gamma_psi[k];
                            for col in 0..rank {
                                let idx = k * rank + col;
                                let g_dir = acc.gamma_dir[idx];
                                let g_psi_dir = acc.gamma_psi_dir[idx];
                                acc.h_psi_dir[col] += 2.0 * rv[k] * (g_dir * g_psi + g * g_psi_dir);
                                acc.hp_psi_dir[col] +=
                                    2.0 * rd[k] * (g_dir * g_psi + g * g_psi_dir);
                                acc.h_psi_vv[col] += 4.0 * rv[k] * g_dir * g_psi_dir;
                                acc.hp_psi_vv[col] += 4.0 * rd[k] * g_dir * g_psi_dir;
                                for e in 0..2 {
                                    let basis = endpoint_basis[e];
                                    acc.endpoint_psi_dir[col][e] +=
                                        2.0 * basis[k] * (g_dir * g_psi + g * g_psi_dir);
                                    acc.endpoint_psi_vv[col][e] +=
                                        4.0 * basis[k] * g_dir * g_psi_dir;
                                }
                            }
                        }

                        let mut axis_value = 0.0;
                        for col in 0..rank {
                            let barrier = -acc.hp_psi_vv[col] * inv_hp
                                + 2.0 * acc.hp_psi_dir[col] * acc.hp_dir[col] * inv_hp_sq
                                + hp_psi * acc.hp_vv[col] * inv_hp_sq
                                - 2.0
                                    * hp_psi
                                    * acc.hp_dir[col]
                                    * acc.hp_dir[col]
                                    * inv_hp_sq
                                    * inv_hp;
                            axis_value += wi
                                * (acc.h_vv[col] * h_psi
                                    + 2.0 * acc.h_dir[col] * acc.h_psi_dir[col]
                                    + hi * acc.h_psi_vv[col]
                                    + barrier
                                    + endpoint_chain_third(
                                        &q,
                                        endpoint_psi,
                                        acc.endpoint_dir[col],
                                        acc.endpoint_dir[col],
                                        acc.endpoint_psi_dir[col],
                                        acc.endpoint_psi_dir[col],
                                        acc.endpoint_vv[col],
                                        acc.endpoint_psi_vv[col],
                                    ));
                        }
                        acc.values[axis_idx] += axis_value;
                    }
                    acc
                },
            )
            .reduce(
                || PsiAllAxesTraceAccum::new(p_resp, rank, n_psi),
                |left, right| left.merge(right),
            );

        Ok(accum.values)
    }

    fn scop_psi_psi_value_score_hvp_from_cov(
        &self,
        beta: &Array1<f64>,
        cached_gamma: ArrayView2<'_, f64>,
        cached_h: ArrayView1<'_, f64>,
        cached_h_prime: ArrayView1<'_, f64>,
        cov: ArrayView2<'_, f64>,
        cov_i: ArrayView2<'_, f64>,
        cov_j: ArrayView2<'_, f64>,
        cov_ij: ArrayView2<'_, f64>,
        row_start: usize,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        direction: Option<&Array1<f64>>,
    ) -> Result<(f64, Array1<f64>, Option<Array1<f64>>), String> {
        let total_n = self.response_val_basis.nrows();
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if row_start > total_n || row_start + n > total_n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi row window [{row_start}, {}) exceeds n={total_n}",
                    row_start + n
                ),
            }
            .into());
        }
        if beta.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                    beta.len()
                ),
            }
            .into());
        }
        if endpoint_q.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi endpoint normalizer cache length {} != n={n}",
                    endpoint_q.len()
                ),
            }
            .into());
        }
        if cached_h.len() != n || cached_h_prime.len() != n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi row-quantity cache length mismatch: h={}, h_prime={}, expected={n}",
                cached_h.len(),
                cached_h_prime.len()
            ) }.into());
        }
        if cached_gamma.nrows() != n || cached_gamma.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi gamma cache shape {}x{} != expected {}x{}",
                    cached_gamma.nrows(),
                    cached_gamma.ncols(),
                    n,
                    p_resp
                ),
            }
            .into());
        }
        for (name, mat) in [
            ("cov", cov),
            ("cov_i", cov_i),
            ("cov_j", cov_j),
            ("cov_ij", cov_ij),
        ] {
            if mat.nrows() != n || mat.ncols() != p_cov {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!(
                        "SCOP psi-psi {name} shape {}x{} != expected {}x{}",
                        mat.nrows(),
                        mat.ncols(),
                        n,
                        p_cov
                    ),
                }
                .into());
            }
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi beta reshape failed: {e}"))?;
        let direction_mat = match direction {
            Some(v) => {
                if v.len() != p_total {
                    return Err(TransformationNormalError::InvalidInput {
                        reason: format!(
                            "SCOP psi-psi HVP direction length {} != p_total {p_total}",
                            v.len()
                        ),
                    }
                    .into());
                }
                Some(
                    v.view()
                        .into_shape_with_order((p_resp, p_cov))
                        .map_err(|e| format!("SCOP psi-psi direction reshape failed: {e}"))?,
                )
            }
            None => None,
        };
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        if direction_mat.is_none() {
            let weights = self.effective_weights();

            struct PsiPairScoreAccum {
                objective: f64,
                score: Array1<f64>,
                gamma: Vec<f64>,
                gamma_i: Vec<f64>,
                gamma_j: Vec<f64>,
                gamma_ij: Vec<f64>,
            }

            impl PsiPairScoreAccum {
                fn new(p_total: usize, p_resp: usize) -> Self {
                    Self {
                        objective: 0.0,
                        score: Array1::<f64>::zeros(p_total),
                        gamma: vec![0.0; p_resp],
                        gamma_i: vec![0.0; p_resp],
                        gamma_j: vec![0.0; p_resp],
                        gamma_ij: vec![0.0; p_resp],
                    }
                }

                fn merge(mut self, rhs: Self) -> Self {
                    self.objective += rhs.objective;
                    self.score.scaled_add(1.0, &rhs.score);
                    self
                }
            }

            let accum = (0..n)
                .into_par_iter()
                .fold(
                    || PsiPairScoreAccum::new(p_total, p_resp),
                    |mut acc, row_idx| {
                        let cov_row = cov.row(row_idx);
                        let cov_i_row = cov_i.row(row_idx);
                        let cov_j_row = cov_j.row(row_idx);
                        let cov_ij_row = cov_ij.row(row_idx);
                        let global_row = row_start + row_idx;
                        let rv = self.response_val_basis.row(global_row);
                        let rd = self.response_deriv_basis.row(global_row);
                        let gamma_row = cached_gamma.row(row_idx);

                        for k in 0..p_resp {
                            let beta_k = beta_mat.row(k);
                            acc.gamma[k] = gamma_row[k];
                            acc.gamma_i[k] = beta_k.dot(&cov_i_row);
                            acc.gamma_j[k] = beta_k.dot(&cov_j_row);
                            acc.gamma_ij[k] = beta_k.dot(&cov_ij_row);
                        }

                        let h = cached_h[row_idx];
                        let hp = cached_h_prime[row_idx];
                        let mut h_i = rv[0] * acc.gamma_i[0];
                        let mut h_j = rv[0] * acc.gamma_j[0];
                        let mut h_ij = rv[0] * acc.gamma_ij[0];
                        let mut hp_i = rd[0] * acc.gamma_i[0];
                        let mut hp_j = rd[0] * acc.gamma_j[0];
                        let mut hp_ij = rd[0] * acc.gamma_ij[0];
                        for k in 1..p_resp {
                            let g = acc.gamma[k];
                            let gi = acc.gamma_i[k];
                            let gj = acc.gamma_j[k];
                            let gij = acc.gamma_ij[k];
                            h_i += 2.0 * rv[k] * g * gi;
                            h_j += 2.0 * rv[k] * g * gj;
                            h_ij += 2.0 * rv[k] * (gj * gi + g * gij);
                            hp_i += 2.0 * rd[k] * g * gi;
                            hp_j += 2.0 * rd[k] * g * gj;
                            hp_ij += 2.0 * rd[k] * (gj * gi + g * gij);
                        }

                        let inv_hp = 1.0 / hp;
                        let inv_hp_sq = inv_hp * inv_hp;
                        let inv_hp_cu = inv_hp_sq * inv_hp;
                        let q = endpoint_q[row_idx];
                        let mut endpoint_i = [0.0; 2];
                        let mut endpoint_j = [0.0; 2];
                        let mut endpoint_ij = [0.0; 2];
                        for e in 0..2 {
                            let basis = endpoint_basis[e];
                            endpoint_i[e] = basis[0] * acc.gamma_i[0];
                            endpoint_j[e] = basis[0] * acc.gamma_j[0];
                            endpoint_ij[e] = basis[0] * acc.gamma_ij[0];
                            for k in 1..p_resp {
                                endpoint_i[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_i[k];
                                endpoint_j[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_j[k];
                                endpoint_ij[e] += 2.0
                                    * basis[k]
                                    * (acc.gamma_j[k] * acc.gamma_i[k]
                                        + acc.gamma[k] * acc.gamma_ij[k]);
                            }
                        }
                        let value = h_i * h_j + h * h_ij - hp_ij * inv_hp
                            + hp_i * hp_j * inv_hp_sq
                            + endpoint_chain_second(&q, endpoint_i, endpoint_j, endpoint_ij);
                        let wi = weights[global_row];
                        acc.objective += wi * value;

                        for k in 0..p_resp {
                            let offset = k * p_cov;
                            let (rvk, rdk) = (rv[k], rd[k]);
                            let (g, gi, gj, gij) = (
                                acc.gamma[k],
                                acc.gamma_i[k],
                                acc.gamma_j[k],
                                acc.gamma_ij[k],
                            );
                            for cidx in 0..p_cov {
                                let c = cov_row[cidx];
                                let ci = cov_i_row[cidx];
                                let cj = cov_j_row[cidx];
                                let cij = cov_ij_row[cidx];
                                let (dh, dhp, dh_i, dh_j, dh_ij, dhp_i, dhp_j, dhp_ij) = if k == 0 {
                                    (
                                        rvk * c,
                                        rdk * c,
                                        rvk * ci,
                                        rvk * cj,
                                        rvk * cij,
                                        rdk * ci,
                                        rdk * cj,
                                        rdk * cij,
                                    )
                                } else {
                                    (
                                        2.0 * rvk * g * c,
                                        2.0 * rdk * g * c,
                                        2.0 * rvk * (gi * c + g * ci),
                                        2.0 * rvk * (gj * c + g * cj),
                                        2.0 * rvk * (gj * ci + gi * cj + gij * c + g * cij),
                                        2.0 * rdk * (gi * c + g * ci),
                                        2.0 * rdk * (gj * c + g * cj),
                                        2.0 * rdk * (gj * ci + gi * cj + gij * c + g * cij),
                                    )
                                };
                                let endpoint_a = if k == 0 {
                                    [endpoint_basis[0][k] * c, endpoint_basis[1][k] * c]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * g * c,
                                        2.0 * endpoint_basis[1][k] * g * c,
                                    ]
                                };
                                let endpoint_i_a = if k == 0 {
                                    [endpoint_basis[0][k] * ci, endpoint_basis[1][k] * ci]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * (gi * c + g * ci),
                                        2.0 * endpoint_basis[1][k] * (gi * c + g * ci),
                                    ]
                                };
                                let endpoint_j_a = if k == 0 {
                                    [endpoint_basis[0][k] * cj, endpoint_basis[1][k] * cj]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * (gj * c + g * cj),
                                        2.0 * endpoint_basis[1][k] * (gj * c + g * cj),
                                    ]
                                };
                                let endpoint_ij_a = if k == 0 {
                                    [endpoint_basis[0][k] * cij, endpoint_basis[1][k] * cij]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k]
                                            * (gj * ci + gi * cj + gij * c + g * cij),
                                        2.0 * endpoint_basis[1][k]
                                            * (gj * ci + gi * cj + gij * c + g * cij),
                                    ]
                                };
                                let grad = dh_i * h_j + h_i * dh_j + dh * h_ij + h * dh_ij
                                    - dhp_ij * inv_hp
                                    + hp_ij * dhp * inv_hp_sq
                                    + (dhp_i * hp_j + hp_i * dhp_j) * inv_hp_sq
                                    - 2.0 * hp_i * hp_j * dhp * inv_hp_cu
                                    + endpoint_chain_third(
                                        &q,
                                        endpoint_i,
                                        endpoint_j,
                                        endpoint_a,
                                        endpoint_ij,
                                        endpoint_i_a,
                                        endpoint_j_a,
                                        endpoint_ij_a,
                                    );
                                acc.score[offset + cidx] += wi * grad;
                            }
                        }
                        acc
                    },
                )
                .reduce(
                    || PsiPairScoreAccum::new(p_total, p_resp),
                    |left, right| left.merge(right),
                );

            return Ok((accum.objective, accum.score, None));
        }

        let weights = self.effective_weights();
        let direction_mat = direction_mat.expect("directional CTN psi-psi path requires direction");

        struct PsiPairDirectionalAccum {
            hvp: Array1<f64>,
            gamma: Vec<f64>,
            gamma_i: Vec<f64>,
            gamma_j: Vec<f64>,
            gamma_ij: Vec<f64>,
            gamma_dot: Vec<f64>,
            gamma_i_dot: Vec<f64>,
            gamma_j_dot: Vec<f64>,
            gamma_ij_dot: Vec<f64>,
        }

        impl PsiPairDirectionalAccum {
            fn new(p_total: usize, p_resp: usize) -> Self {
                Self {
                    hvp: Array1::<f64>::zeros(p_total),
                    gamma: vec![0.0; p_resp],
                    gamma_i: vec![0.0; p_resp],
                    gamma_j: vec![0.0; p_resp],
                    gamma_ij: vec![0.0; p_resp],
                    gamma_dot: vec![0.0; p_resp],
                    gamma_i_dot: vec![0.0; p_resp],
                    gamma_j_dot: vec![0.0; p_resp],
                    gamma_ij_dot: vec![0.0; p_resp],
                }
            }

            fn merge(mut self, rhs: Self) -> Self {
                self.hvp.scaled_add(1.0, &rhs.hvp);
                self
            }
        }

        let accum = (0..n)
            .into_par_iter()
            .fold(
                || PsiPairDirectionalAccum::new(p_total, p_resp),
                |mut acc, row_idx| {
                    let cov_row = cov.row(row_idx);
                    let cov_i_row = cov_i.row(row_idx);
                    let cov_j_row = cov_j.row(row_idx);
                    let cov_ij_row = cov_ij.row(row_idx);
                    let global_row = row_start + row_idx;
                    let rv = self.response_val_basis.row(global_row);
                    let rd = self.response_deriv_basis.row(global_row);
                    let gamma_row = cached_gamma.row(row_idx);

                    for k in 0..p_resp {
                        let beta_k = beta_mat.row(k);
                        let dir_k = direction_mat.row(k);
                        acc.gamma[k] = gamma_row[k];
                        acc.gamma_i[k] = beta_k.dot(&cov_i_row);
                        acc.gamma_j[k] = beta_k.dot(&cov_j_row);
                        acc.gamma_ij[k] = beta_k.dot(&cov_ij_row);
                        acc.gamma_dot[k] = dir_k.dot(&cov_row);
                        acc.gamma_i_dot[k] = dir_k.dot(&cov_i_row);
                        acc.gamma_j_dot[k] = dir_k.dot(&cov_j_row);
                        acc.gamma_ij_dot[k] = dir_k.dot(&cov_ij_row);
                    }

                    let h = cached_h[row_idx];
                    let hp = cached_h_prime[row_idx];
                    let mut h_i = rv[0] * acc.gamma_i[0];
                    let mut h_j = rv[0] * acc.gamma_j[0];
                    let mut h_ij = rv[0] * acc.gamma_ij[0];
                    let mut hp_i = rd[0] * acc.gamma_i[0];
                    let mut hp_j = rd[0] * acc.gamma_j[0];
                    let mut hp_ij = rd[0] * acc.gamma_ij[0];
                    let mut h_dot = rv[0] * acc.gamma_dot[0];
                    let mut hp_dot = rd[0] * acc.gamma_dot[0];
                    let mut h_i_dot = rv[0] * acc.gamma_i_dot[0];
                    let mut h_j_dot = rv[0] * acc.gamma_j_dot[0];
                    let mut h_ij_dot = rv[0] * acc.gamma_ij_dot[0];
                    let mut hp_i_dot = rd[0] * acc.gamma_i_dot[0];
                    let mut hp_j_dot = rd[0] * acc.gamma_j_dot[0];
                    let mut hp_ij_dot = rd[0] * acc.gamma_ij_dot[0];

                    for k in 1..p_resp {
                        let g = acc.gamma[k];
                        let gi = acc.gamma_i[k];
                        let gj = acc.gamma_j[k];
                        let gij = acc.gamma_ij[k];
                        let u = acc.gamma_dot[k];
                        let ui = acc.gamma_i_dot[k];
                        let uj = acc.gamma_j_dot[k];
                        let uij = acc.gamma_ij_dot[k];
                        h_i += 2.0 * rv[k] * g * gi;
                        h_j += 2.0 * rv[k] * g * gj;
                        h_ij += 2.0 * rv[k] * (gj * gi + g * gij);
                        hp_i += 2.0 * rd[k] * g * gi;
                        hp_j += 2.0 * rd[k] * g * gj;
                        hp_ij += 2.0 * rd[k] * (gj * gi + g * gij);
                        h_dot += 2.0 * rv[k] * g * u;
                        hp_dot += 2.0 * rd[k] * g * u;
                        h_i_dot += 2.0 * rv[k] * (u * gi + g * ui);
                        h_j_dot += 2.0 * rv[k] * (u * gj + g * uj);
                        h_ij_dot += 2.0 * rv[k] * (uj * gi + gj * ui + u * gij + g * uij);
                        hp_i_dot += 2.0 * rd[k] * (u * gi + g * ui);
                        hp_j_dot += 2.0 * rd[k] * (u * gj + g * uj);
                        hp_ij_dot += 2.0 * rd[k] * (uj * gi + gj * ui + u * gij + g * uij);
                    }

                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;
                    let wi = weights[global_row];
                    let q = endpoint_q[row_idx];
                    let mut endpoint_i = [0.0; 2];
                    let mut endpoint_j = [0.0; 2];
                    let mut endpoint_ij = [0.0; 2];
                    let mut endpoint_d = [0.0; 2];
                    let mut endpoint_i_d = [0.0; 2];
                    let mut endpoint_j_d = [0.0; 2];
                    let mut endpoint_ij_d = [0.0; 2];
                    for e in 0..2 {
                        let basis = endpoint_basis[e];
                        endpoint_i[e] = basis[0] * acc.gamma_i[0];
                        endpoint_j[e] = basis[0] * acc.gamma_j[0];
                        endpoint_ij[e] = basis[0] * acc.gamma_ij[0];
                        endpoint_d[e] = basis[0] * acc.gamma_dot[0];
                        endpoint_i_d[e] = basis[0] * acc.gamma_i_dot[0];
                        endpoint_j_d[e] = basis[0] * acc.gamma_j_dot[0];
                        endpoint_ij_d[e] = basis[0] * acc.gamma_ij_dot[0];
                        for k in 1..p_resp {
                            endpoint_i[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_i[k];
                            endpoint_j[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_j[k];
                            endpoint_ij[e] += 2.0
                                * basis[k]
                                * (acc.gamma_j[k] * acc.gamma_i[k]
                                    + acc.gamma[k] * acc.gamma_ij[k]);
                            endpoint_d[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_dot[k];
                            endpoint_i_d[e] += 2.0
                                * basis[k]
                                * (acc.gamma_dot[k] * acc.gamma_i[k]
                                    + acc.gamma[k] * acc.gamma_i_dot[k]);
                            endpoint_j_d[e] += 2.0
                                * basis[k]
                                * (acc.gamma_dot[k] * acc.gamma_j[k]
                                    + acc.gamma[k] * acc.gamma_j_dot[k]);
                            endpoint_ij_d[e] += 2.0
                                * basis[k]
                                * (acc.gamma_j_dot[k] * acc.gamma_i[k]
                                    + acc.gamma_j[k] * acc.gamma_i_dot[k]
                                    + acc.gamma_dot[k] * acc.gamma_ij[k]
                                    + acc.gamma[k] * acc.gamma_ij_dot[k]);
                        }
                    }

                    for k in 0..p_resp {
                        let offset = k * p_cov;
                        let (rvk, rdk) = (rv[k], rd[k]);
                        let (g, gi, gj, gij) = (
                            acc.gamma[k],
                            acc.gamma_i[k],
                            acc.gamma_j[k],
                            acc.gamma_ij[k],
                        );
                        let (u, ui, uj, uij) = (
                            acc.gamma_dot[k],
                            acc.gamma_i_dot[k],
                            acc.gamma_j_dot[k],
                            acc.gamma_ij_dot[k],
                        );
                        for cidx in 0..p_cov {
                            let c = cov_row[cidx];
                            let ci = cov_i_row[cidx];
                            let cj = cov_j_row[cidx];
                            let cij = cov_ij_row[cidx];
                            let (
                                dh,
                                dhp,
                                dh_i,
                                dh_j,
                                dh_ij,
                                dhp_i,
                                dhp_j,
                                dhp_ij,
                                ddh,
                                ddhp,
                                ddh_i,
                                ddh_j,
                                ddh_ij,
                                ddhp_i,
                                ddhp_j,
                                ddhp_ij,
                            ) = if k == 0 {
                                (
                                    rvk * c,
                                    rdk * c,
                                    rvk * ci,
                                    rvk * cj,
                                    rvk * cij,
                                    rdk * ci,
                                    rdk * cj,
                                    rdk * cij,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                    0.0,
                                )
                            } else {
                                (
                                    2.0 * rvk * g * c,
                                    2.0 * rdk * g * c,
                                    2.0 * rvk * (gi * c + g * ci),
                                    2.0 * rvk * (gj * c + g * cj),
                                    2.0 * rvk * (gj * ci + gi * cj + gij * c + g * cij),
                                    2.0 * rdk * (gi * c + g * ci),
                                    2.0 * rdk * (gj * c + g * cj),
                                    2.0 * rdk * (gj * ci + gi * cj + gij * c + g * cij),
                                    2.0 * rvk * u * c,
                                    2.0 * rdk * u * c,
                                    2.0 * rvk * (ui * c + u * ci),
                                    2.0 * rvk * (uj * c + u * cj),
                                    2.0 * rvk * (uj * ci + ui * cj + uij * c + u * cij),
                                    2.0 * rdk * (ui * c + u * ci),
                                    2.0 * rdk * (uj * c + u * cj),
                                    2.0 * rdk * (uj * ci + ui * cj + uij * c + u * cij),
                                )
                            };

                            let endpoint_a = if k == 0 {
                                [endpoint_basis[0][k] * c, endpoint_basis[1][k] * c]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k] * g * c,
                                    2.0 * endpoint_basis[1][k] * g * c,
                                ]
                            };
                            let endpoint_i_a = if k == 0 {
                                [endpoint_basis[0][k] * ci, endpoint_basis[1][k] * ci]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k] * (gi * c + g * ci),
                                    2.0 * endpoint_basis[1][k] * (gi * c + g * ci),
                                ]
                            };
                            let endpoint_j_a = if k == 0 {
                                [endpoint_basis[0][k] * cj, endpoint_basis[1][k] * cj]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k] * (gj * c + g * cj),
                                    2.0 * endpoint_basis[1][k] * (gj * c + g * cj),
                                ]
                            };
                            let endpoint_ij_a = if k == 0 {
                                [endpoint_basis[0][k] * cij, endpoint_basis[1][k] * cij]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k]
                                        * (gj * ci + gi * cj + gij * c + g * cij),
                                    2.0 * endpoint_basis[1][k]
                                        * (gj * ci + gi * cj + gij * c + g * cij),
                                ]
                            };
                            let endpoint_a_d = if k == 0 {
                                [0.0; 2]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k] * u * c,
                                    2.0 * endpoint_basis[1][k] * u * c,
                                ]
                            };
                            let endpoint_i_a_d = if k == 0 {
                                [0.0; 2]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k] * (ui * c + u * ci),
                                    2.0 * endpoint_basis[1][k] * (ui * c + u * ci),
                                ]
                            };
                            let endpoint_j_a_d = if k == 0 {
                                [0.0; 2]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k] * (uj * c + u * cj),
                                    2.0 * endpoint_basis[1][k] * (uj * c + u * cj),
                                ]
                            };
                            let endpoint_ij_a_d = if k == 0 {
                                [0.0; 2]
                            } else {
                                [
                                    2.0 * endpoint_basis[0][k]
                                        * (uj * ci + ui * cj + uij * c + u * cij),
                                    2.0 * endpoint_basis[1][k]
                                        * (uj * ci + ui * cj + uij * c + u * cij),
                                ]
                            };
                            let n1 = dhp_i * hp_j + hp_i * dhp_j;
                            let n1_dot =
                                ddhp_i * hp_j + dhp_i * hp_j_dot + hp_i_dot * dhp_j + hp_i * ddhp_j;
                            let n2_dot =
                                hp_i_dot * hp_j * dhp + hp_i * hp_j_dot * dhp + hp_i * hp_j * ddhp;
                            let hv = ddh_i * h_j
                                + dh_i * h_j_dot
                                + h_i_dot * dh_j
                                + h_i * ddh_j
                                + ddh * h_ij
                                + dh * h_ij_dot
                                + h_dot * dh_ij
                                + h * ddh_ij
                                - ddhp_ij * inv_hp
                                + dhp_ij * hp_dot * inv_hp_sq
                                + hp_ij_dot * dhp * inv_hp_sq
                                + hp_ij * ddhp * inv_hp_sq
                                - 2.0 * hp_ij * dhp * hp_dot * inv_hp_cu
                                + n1_dot * inv_hp_sq
                                - 2.0 * n1 * hp_dot * inv_hp_cu
                                - 2.0 * n2_dot * inv_hp_cu
                                + 6.0 * hp_i * hp_j * dhp * hp_dot * inv_hp_qu
                                + endpoint_chain_fourth(
                                    &q,
                                    endpoint_i,
                                    endpoint_j,
                                    endpoint_a,
                                    endpoint_d,
                                    endpoint_ij,
                                    endpoint_i_a,
                                    endpoint_i_d,
                                    endpoint_j_a,
                                    endpoint_j_d,
                                    endpoint_a_d,
                                    endpoint_ij_a,
                                    endpoint_ij_d,
                                    endpoint_i_a_d,
                                    endpoint_j_a_d,
                                    endpoint_ij_a_d,
                                );
                            acc.hvp[offset + cidx] += wi * hv;
                        }
                    }
                    acc
                },
            )
            .reduce(
                || PsiPairDirectionalAccum::new(p_total, p_resp),
                |left, right| left.merge(right),
            );

        Ok((0.0, Array1::<f64>::zeros(p_total), Some(accum.hvp)))
    }

    fn scop_psi_psi_hvp_mat_from_cov(
        &self,
        beta: &Array1<f64>,
        cached_gamma: ArrayView2<'_, f64>,
        cached_h: ArrayView1<'_, f64>,
        cached_h_prime: ArrayView1<'_, f64>,
        cov: ArrayView2<'_, f64>,
        cov_i: ArrayView2<'_, f64>,
        cov_j: ArrayView2<'_, f64>,
        cov_ij: ArrayView2<'_, f64>,
        row_start: usize,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        factor: ArrayView2<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let total_n = self.response_val_basis.nrows();
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let rank = factor.ncols();
        if row_start > total_n || row_start + n > total_n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi batched HVP row window [{row_start}, {}) exceeds n={total_n}",
                    row_start + n
                ),
            }
            .into());
        }
        if beta.len() != p_total || factor.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi batched HVP length mismatch: beta={}, factor_rows={}, expected={p_total}",
                beta.len(),
                factor.nrows()
            ) }.into());
        }
        if endpoint_q.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi batched HVP endpoint normalizer cache length {} != n={n}",
                    endpoint_q.len()
                ),
            }
            .into());
        }
        if cached_h.len() != n || cached_h_prime.len() != n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi batched HVP row-quantity cache length mismatch: h={}, h_prime={}, expected={n}",
                cached_h.len(),
                cached_h_prime.len()
            ) }.into());
        }
        if cached_gamma.nrows() != n || cached_gamma.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi batched HVP gamma cache shape {}x{} != expected {}x{}",
                    cached_gamma.nrows(),
                    cached_gamma.ncols(),
                    n,
                    p_resp
                ),
            }
            .into());
        }
        for (name, mat) in [
            ("cov", cov),
            ("cov_i", cov_i),
            ("cov_j", cov_j),
            ("cov_ij", cov_ij),
        ] {
            if mat.nrows() != n || mat.ncols() != p_cov {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!(
                        "SCOP psi-psi batched HVP {name} shape {}x{} != expected {}x{}",
                        mat.nrows(),
                        mat.ncols(),
                        n,
                        p_cov
                    ),
                }
                .into());
            }
        }

        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi batched HVP beta reshape failed: {e}"))?;
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        struct PsiPairBatchedAccum {
            hvp: Array2<f64>,
            gamma: Vec<f64>,
            gamma_i: Vec<f64>,
            gamma_j: Vec<f64>,
            gamma_ij: Vec<f64>,
            gamma_dot: Vec<f64>,
            gamma_i_dot: Vec<f64>,
            gamma_j_dot: Vec<f64>,
            gamma_ij_dot: Vec<f64>,
        }

        impl PsiPairBatchedAccum {
            fn new(p_total: usize, p_resp: usize, rank: usize) -> Self {
                let projected_len = p_resp * rank;
                Self {
                    hvp: Array2::<f64>::zeros((p_total, rank)),
                    gamma: vec![0.0; p_resp],
                    gamma_i: vec![0.0; p_resp],
                    gamma_j: vec![0.0; p_resp],
                    gamma_ij: vec![0.0; p_resp],
                    gamma_dot: vec![0.0; projected_len],
                    gamma_i_dot: vec![0.0; projected_len],
                    gamma_j_dot: vec![0.0; projected_len],
                    gamma_ij_dot: vec![0.0; projected_len],
                }
            }

            fn merge(mut self, rhs: Self) -> Self {
                self.hvp += &rhs.hvp;
                self
            }
        }

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let weights = self.effective_weights();
        let accum = (0..n)
            .into_par_iter()
            .fold(
                || PsiPairBatchedAccum::new(p_total, p_resp, rank),
                |mut acc, row_idx| {
                    let cov_row = cov.row(row_idx);
                    let cov_i_row = cov_i.row(row_idx);
                    let cov_j_row = cov_j.row(row_idx);
                    let cov_ij_row = cov_ij.row(row_idx);
                    let global_row = row_start + row_idx;
                    let rv = self.response_val_basis.row(global_row);
                    let rd = self.response_deriv_basis.row(global_row);
                    let gamma_row = cached_gamma.row(row_idx);

                    for k in 0..p_resp {
                        let beta_k = beta_mat.row(k);
                        acc.gamma[k] = gamma_row[k];
                        acc.gamma_i[k] = beta_k.dot(&cov_i_row);
                        acc.gamma_j[k] = beta_k.dot(&cov_j_row);
                        acc.gamma_ij[k] = beta_k.dot(&cov_ij_row);
                    }

                    let h = cached_h[row_idx];
                    let hp = cached_h_prime[row_idx];
                    let mut h_i = rv[0] * acc.gamma_i[0];
                    let mut h_j = rv[0] * acc.gamma_j[0];
                    let mut h_ij = rv[0] * acc.gamma_ij[0];
                    let mut hp_i = rd[0] * acc.gamma_i[0];
                    let mut hp_j = rd[0] * acc.gamma_j[0];
                    let mut hp_ij = rd[0] * acc.gamma_ij[0];

                    for k in 1..p_resp {
                        let g = acc.gamma[k];
                        let gi = acc.gamma_i[k];
                        let gj = acc.gamma_j[k];
                        let gij = acc.gamma_ij[k];
                        h_i += 2.0 * rv[k] * g * gi;
                        h_j += 2.0 * rv[k] * g * gj;
                        h_ij += 2.0 * rv[k] * (gj * gi + g * gij);
                        hp_i += 2.0 * rd[k] * g * gi;
                        hp_j += 2.0 * rd[k] * g * gj;
                        hp_ij += 2.0 * rd[k] * (gj * gi + g * gij);
                    }

                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;
                    let wi = weights[global_row];
                    let q = endpoint_q[row_idx];
                    let mut endpoint_i = [0.0; 2];
                    let mut endpoint_j = [0.0; 2];
                    let mut endpoint_ij = [0.0; 2];
                    for e in 0..2 {
                        let basis = endpoint_basis[e];
                        endpoint_i[e] = basis[0] * acc.gamma_i[0];
                        endpoint_j[e] = basis[0] * acc.gamma_j[0];
                        endpoint_ij[e] = basis[0] * acc.gamma_ij[0];
                        for k in 1..p_resp {
                            endpoint_i[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_i[k];
                            endpoint_j[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_j[k];
                            endpoint_ij[e] += 2.0
                                * basis[k]
                                * (acc.gamma_j[k] * acc.gamma_i[k]
                                    + acc.gamma[k] * acc.gamma_ij[k]);
                        }
                    }

                    acc.gamma_dot.fill(0.0);
                    acc.gamma_i_dot.fill(0.0);
                    acc.gamma_j_dot.fill(0.0);
                    acc.gamma_ij_dot.fill(0.0);
                    for k in 0..p_resp {
                        let factor_row_base = k * p_cov;
                        let projected_base = k * rank;
                        for cidx in 0..p_cov {
                            let factor_row = factor_row_base + cidx;
                            let cov_v = cov_row[cidx];
                            let cov_i_v = cov_i_row[cidx];
                            let cov_j_v = cov_j_row[cidx];
                            let cov_ij_v = cov_ij_row[cidx];
                            for col in 0..rank {
                                let coeff = factor[[factor_row, col]];
                                let idx = projected_base + col;
                                acc.gamma_dot[idx] += coeff * cov_v;
                                acc.gamma_i_dot[idx] += coeff * cov_i_v;
                                acc.gamma_j_dot[idx] += coeff * cov_j_v;
                                acc.gamma_ij_dot[idx] += coeff * cov_ij_v;
                            }
                        }
                    }

                    for col in 0..rank {
                        let mut h_dot = rv[0] * acc.gamma_dot[col];
                        let mut hp_dot = rd[0] * acc.gamma_dot[col];
                        let mut h_i_dot = rv[0] * acc.gamma_i_dot[col];
                        let mut h_j_dot = rv[0] * acc.gamma_j_dot[col];
                        let mut h_ij_dot = rv[0] * acc.gamma_ij_dot[col];
                        let mut hp_i_dot = rd[0] * acc.gamma_i_dot[col];
                        let mut hp_j_dot = rd[0] * acc.gamma_j_dot[col];
                        let mut hp_ij_dot = rd[0] * acc.gamma_ij_dot[col];
                        for k in 1..p_resp {
                            let idx = k * rank + col;
                            let g = acc.gamma[k];
                            let gi = acc.gamma_i[k];
                            let gj = acc.gamma_j[k];
                            let gij = acc.gamma_ij[k];
                            let u = acc.gamma_dot[idx];
                            let ui = acc.gamma_i_dot[idx];
                            let uj = acc.gamma_j_dot[idx];
                            let uij = acc.gamma_ij_dot[idx];
                            h_dot += 2.0 * rv[k] * g * u;
                            hp_dot += 2.0 * rd[k] * g * u;
                            h_i_dot += 2.0 * rv[k] * (u * gi + g * ui);
                            h_j_dot += 2.0 * rv[k] * (u * gj + g * uj);
                            h_ij_dot += 2.0 * rv[k] * (uj * gi + gj * ui + u * gij + g * uij);
                            hp_i_dot += 2.0 * rd[k] * (u * gi + g * ui);
                            hp_j_dot += 2.0 * rd[k] * (u * gj + g * uj);
                            hp_ij_dot += 2.0 * rd[k] * (uj * gi + gj * ui + u * gij + g * uij);
                        }

                        let mut endpoint_d = [0.0; 2];
                        let mut endpoint_i_d = [0.0; 2];
                        let mut endpoint_j_d = [0.0; 2];
                        let mut endpoint_ij_d = [0.0; 2];
                        for e in 0..2 {
                            let basis = endpoint_basis[e];
                            endpoint_d[e] = basis[0] * acc.gamma_dot[col];
                            endpoint_i_d[e] = basis[0] * acc.gamma_i_dot[col];
                            endpoint_j_d[e] = basis[0] * acc.gamma_j_dot[col];
                            endpoint_ij_d[e] = basis[0] * acc.gamma_ij_dot[col];
                            for k in 1..p_resp {
                                let idx = k * rank + col;
                                endpoint_d[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_dot[idx];
                                endpoint_i_d[e] += 2.0
                                    * basis[k]
                                    * (acc.gamma_dot[idx] * acc.gamma_i[k]
                                        + acc.gamma[k] * acc.gamma_i_dot[idx]);
                                endpoint_j_d[e] += 2.0
                                    * basis[k]
                                    * (acc.gamma_dot[idx] * acc.gamma_j[k]
                                        + acc.gamma[k] * acc.gamma_j_dot[idx]);
                                endpoint_ij_d[e] += 2.0
                                    * basis[k]
                                    * (acc.gamma_j_dot[idx] * acc.gamma_i[k]
                                        + acc.gamma_j[k] * acc.gamma_i_dot[idx]
                                        + acc.gamma_dot[idx] * acc.gamma_ij[k]
                                        + acc.gamma[k] * acc.gamma_ij_dot[idx]);
                            }
                        }

                        for k in 0..p_resp {
                            let offset = k * p_cov;
                            let (rvk, rdk) = (rv[k], rd[k]);
                            let (g, gi, gj, gij) = (
                                acc.gamma[k],
                                acc.gamma_i[k],
                                acc.gamma_j[k],
                                acc.gamma_ij[k],
                            );
                            let (u, ui, uj, uij) = (
                                acc.gamma_dot[k * rank + col],
                                acc.gamma_i_dot[k * rank + col],
                                acc.gamma_j_dot[k * rank + col],
                                acc.gamma_ij_dot[k * rank + col],
                            );
                            for cidx in 0..p_cov {
                                let c = cov_row[cidx];
                                let ci = cov_i_row[cidx];
                                let cj = cov_j_row[cidx];
                                let cij = cov_ij_row[cidx];
                                let (
                                    dh,
                                    dhp,
                                    dh_i,
                                    dh_j,
                                    dh_ij,
                                    dhp_i,
                                    dhp_j,
                                    dhp_ij,
                                    ddh,
                                    ddhp,
                                    ddh_i,
                                    ddh_j,
                                    ddh_ij,
                                    ddhp_i,
                                    ddhp_j,
                                    ddhp_ij,
                                ) = if k == 0 {
                                    (
                                        rvk * c,
                                        rdk * c,
                                        rvk * ci,
                                        rvk * cj,
                                        rvk * cij,
                                        rdk * ci,
                                        rdk * cj,
                                        rdk * cij,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                        0.0,
                                    )
                                } else {
                                    (
                                        2.0 * rvk * g * c,
                                        2.0 * rdk * g * c,
                                        2.0 * rvk * (gi * c + g * ci),
                                        2.0 * rvk * (gj * c + g * cj),
                                        2.0 * rvk * (gj * ci + gi * cj + gij * c + g * cij),
                                        2.0 * rdk * (gi * c + g * ci),
                                        2.0 * rdk * (gj * c + g * cj),
                                        2.0 * rdk * (gj * ci + gi * cj + gij * c + g * cij),
                                        2.0 * rvk * u * c,
                                        2.0 * rdk * u * c,
                                        2.0 * rvk * (ui * c + u * ci),
                                        2.0 * rvk * (uj * c + u * cj),
                                        2.0 * rvk * (uj * ci + ui * cj + uij * c + u * cij),
                                        2.0 * rdk * (ui * c + u * ci),
                                        2.0 * rdk * (uj * c + u * cj),
                                        2.0 * rdk * (uj * ci + ui * cj + uij * c + u * cij),
                                    )
                                };

                                let endpoint_a = if k == 0 {
                                    [endpoint_basis[0][k] * c, endpoint_basis[1][k] * c]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * g * c,
                                        2.0 * endpoint_basis[1][k] * g * c,
                                    ]
                                };
                                let endpoint_i_a = if k == 0 {
                                    [endpoint_basis[0][k] * ci, endpoint_basis[1][k] * ci]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * (gi * c + g * ci),
                                        2.0 * endpoint_basis[1][k] * (gi * c + g * ci),
                                    ]
                                };
                                let endpoint_j_a = if k == 0 {
                                    [endpoint_basis[0][k] * cj, endpoint_basis[1][k] * cj]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * (gj * c + g * cj),
                                        2.0 * endpoint_basis[1][k] * (gj * c + g * cj),
                                    ]
                                };
                                let endpoint_ij_a = if k == 0 {
                                    [endpoint_basis[0][k] * cij, endpoint_basis[1][k] * cij]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k]
                                            * (gj * ci + gi * cj + gij * c + g * cij),
                                        2.0 * endpoint_basis[1][k]
                                            * (gj * ci + gi * cj + gij * c + g * cij),
                                    ]
                                };
                                let endpoint_a_d = if k == 0 {
                                    [0.0; 2]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * u * c,
                                        2.0 * endpoint_basis[1][k] * u * c,
                                    ]
                                };
                                let endpoint_i_a_d = if k == 0 {
                                    [0.0; 2]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * (ui * c + u * ci),
                                        2.0 * endpoint_basis[1][k] * (ui * c + u * ci),
                                    ]
                                };
                                let endpoint_j_a_d = if k == 0 {
                                    [0.0; 2]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k] * (uj * c + u * cj),
                                        2.0 * endpoint_basis[1][k] * (uj * c + u * cj),
                                    ]
                                };
                                let endpoint_ij_a_d = if k == 0 {
                                    [0.0; 2]
                                } else {
                                    [
                                        2.0 * endpoint_basis[0][k]
                                            * (uj * ci + ui * cj + uij * c + u * cij),
                                        2.0 * endpoint_basis[1][k]
                                            * (uj * ci + ui * cj + uij * c + u * cij),
                                    ]
                                };
                                let n1 = dhp_i * hp_j + hp_i * dhp_j;
                                let n1_dot = ddhp_i * hp_j
                                    + dhp_i * hp_j_dot
                                    + hp_i_dot * dhp_j
                                    + hp_i * ddhp_j;
                                let n2_dot = hp_i_dot * hp_j * dhp
                                    + hp_i * hp_j_dot * dhp
                                    + hp_i * hp_j * ddhp;
                                let hv = ddh_i * h_j
                                    + dh_i * h_j_dot
                                    + h_i_dot * dh_j
                                    + h_i * ddh_j
                                    + ddh * h_ij
                                    + dh * h_ij_dot
                                    + h_dot * dh_ij
                                    + h * ddh_ij
                                    - ddhp_ij * inv_hp
                                    + dhp_ij * hp_dot * inv_hp_sq
                                    + hp_ij_dot * dhp * inv_hp_sq
                                    + hp_ij * ddhp * inv_hp_sq
                                    - 2.0 * hp_ij * dhp * hp_dot * inv_hp_cu
                                    + n1_dot * inv_hp_sq
                                    - 2.0 * n1 * hp_dot * inv_hp_cu
                                    - 2.0 * n2_dot * inv_hp_cu
                                    + 6.0 * hp_i * hp_j * dhp * hp_dot * inv_hp_qu
                                    + endpoint_chain_fourth(
                                        &q,
                                        endpoint_i,
                                        endpoint_j,
                                        endpoint_a,
                                        endpoint_d,
                                        endpoint_ij,
                                        endpoint_i_a,
                                        endpoint_i_d,
                                        endpoint_j_a,
                                        endpoint_j_d,
                                        endpoint_a_d,
                                        endpoint_ij_a,
                                        endpoint_ij_d,
                                        endpoint_i_a_d,
                                        endpoint_j_a_d,
                                        endpoint_ij_a_d,
                                    );
                                acc.hvp[[offset + cidx, col]] += wi * hv;
                            }
                        }
                    }
                    acc
                },
            )
            .reduce(
                || PsiPairBatchedAccum::new(p_total, p_resp, rank),
                |left, right| left.merge(right),
            );

        Ok(accum.hvp)
    }

    fn scop_psi_psi_bilinear_from_cov(
        &self,
        beta: &Array1<f64>,
        cached_gamma: ArrayView2<'_, f64>,
        cached_h: ArrayView1<'_, f64>,
        cached_h_prime: ArrayView1<'_, f64>,
        cov: ArrayView2<'_, f64>,
        cov_i: ArrayView2<'_, f64>,
        cov_j: ArrayView2<'_, f64>,
        cov_ij: ArrayView2<'_, f64>,
        row_start: usize,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        left: ArrayView1<'_, f64>,
        right: ArrayView1<'_, f64>,
    ) -> Result<f64, String> {
        let total_n = self.response_val_basis.nrows();
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if row_start > total_n || row_start + n > total_n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi bilinear row window [{row_start}, {}) exceeds n={total_n}",
                    row_start + n
                ),
            }
            .into());
        }
        if beta.len() != p_total || left.len() != p_total || right.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi bilinear length mismatch: beta={}, left={}, right={}, expected={p_total}",
                beta.len(),
                left.len(),
                right.len()
            ) }.into());
        }
        if endpoint_q.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi bilinear endpoint normalizer cache length {} != n={n}",
                    endpoint_q.len()
                ),
            }
            .into());
        }
        if cached_h.len() != n || cached_h_prime.len() != n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi bilinear row-quantity cache length mismatch: h={}, h_prime={}, expected={n}",
                cached_h.len(),
                cached_h_prime.len()
            ) }.into());
        }
        if cached_gamma.nrows() != n || cached_gamma.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi bilinear gamma cache shape {}x{} != expected {}x{}",
                    cached_gamma.nrows(),
                    cached_gamma.ncols(),
                    n,
                    p_resp
                ),
            }
            .into());
        }
        for (name, mat) in [
            ("cov", cov),
            ("cov_i", cov_i),
            ("cov_j", cov_j),
            ("cov_ij", cov_ij),
        ] {
            if mat.nrows() != n || mat.ncols() != p_cov {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!(
                        "SCOP psi-psi bilinear {name} shape {}x{} != expected {}x{}",
                        mat.nrows(),
                        mat.ncols(),
                        n,
                        p_cov
                    ),
                }
                .into());
            }
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi bilinear beta reshape failed: {e}"))?;
        let left_mat = left
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi bilinear left reshape failed: {e}"))?;
        let right_mat = right
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi bilinear right reshape failed: {e}"))?;
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        struct PsiPairBilinearAccum {
            value: f64,
            gamma: Vec<f64>,
            gamma_i: Vec<f64>,
            gamma_j: Vec<f64>,
            gamma_ij: Vec<f64>,
            left: Vec<f64>,
            left_i: Vec<f64>,
            left_j: Vec<f64>,
            left_ij: Vec<f64>,
            right: Vec<f64>,
            right_i: Vec<f64>,
            right_j: Vec<f64>,
            right_ij: Vec<f64>,
        }

        impl PsiPairBilinearAccum {
            fn new(p_resp: usize) -> Self {
                Self {
                    value: 0.0,
                    gamma: vec![0.0; p_resp],
                    gamma_i: vec![0.0; p_resp],
                    gamma_j: vec![0.0; p_resp],
                    gamma_ij: vec![0.0; p_resp],
                    left: vec![0.0; p_resp],
                    left_i: vec![0.0; p_resp],
                    left_j: vec![0.0; p_resp],
                    left_ij: vec![0.0; p_resp],
                    right: vec![0.0; p_resp],
                    right_i: vec![0.0; p_resp],
                    right_j: vec![0.0; p_resp],
                    right_ij: vec![0.0; p_resp],
                }
            }
        }

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let weights = self.effective_weights();
        let total = (0..n)
            .into_par_iter()
            .fold(
                || PsiPairBilinearAccum::new(p_resp),
                |mut acc, row_idx| {
                    let cov_row = cov.row(row_idx);
                    let cov_i_row = cov_i.row(row_idx);
                    let cov_j_row = cov_j.row(row_idx);
                    let cov_ij_row = cov_ij.row(row_idx);
                    let global_row = row_start + row_idx;
                    let rv = self.response_val_basis.row(global_row);
                    let rd = self.response_deriv_basis.row(global_row);
                    let gamma_row = cached_gamma.row(row_idx);

                    for k in 0..p_resp {
                        let beta_k = beta_mat.row(k);
                        let left_k = left_mat.row(k);
                        let right_k = right_mat.row(k);
                        acc.gamma[k] = gamma_row[k];
                        acc.gamma_i[k] = beta_k.dot(&cov_i_row);
                        acc.gamma_j[k] = beta_k.dot(&cov_j_row);
                        acc.gamma_ij[k] = beta_k.dot(&cov_ij_row);
                        acc.left[k] = left_k.dot(&cov_row);
                        acc.left_i[k] = left_k.dot(&cov_i_row);
                        acc.left_j[k] = left_k.dot(&cov_j_row);
                        acc.left_ij[k] = left_k.dot(&cov_ij_row);
                        acc.right[k] = right_k.dot(&cov_row);
                        acc.right_i[k] = right_k.dot(&cov_i_row);
                        acc.right_j[k] = right_k.dot(&cov_j_row);
                        acc.right_ij[k] = right_k.dot(&cov_ij_row);
                    }

                    let h = cached_h[row_idx];
                    let hp = cached_h_prime[row_idx];
                    let mut h_i = rv[0] * acc.gamma_i[0];
                    let mut h_j = rv[0] * acc.gamma_j[0];
                    let mut h_ij = rv[0] * acc.gamma_ij[0];
                    let mut hp_i = rd[0] * acc.gamma_i[0];
                    let mut hp_j = rd[0] * acc.gamma_j[0];
                    let mut hp_ij = rd[0] * acc.gamma_ij[0];

                    let mut h_l = rv[0] * acc.left[0];
                    let mut hp_l = rd[0] * acc.left[0];
                    let mut h_i_l = rv[0] * acc.left_i[0];
                    let mut h_j_l = rv[0] * acc.left_j[0];
                    let mut h_ij_l = rv[0] * acc.left_ij[0];
                    let mut hp_i_l = rd[0] * acc.left_i[0];
                    let mut hp_j_l = rd[0] * acc.left_j[0];
                    let mut hp_ij_l = rd[0] * acc.left_ij[0];

                    let mut h_r = rv[0] * acc.right[0];
                    let mut hp_r = rd[0] * acc.right[0];
                    let mut h_i_r = rv[0] * acc.right_i[0];
                    let mut h_j_r = rv[0] * acc.right_j[0];
                    let mut h_ij_r = rv[0] * acc.right_ij[0];
                    let mut hp_i_r = rd[0] * acc.right_i[0];
                    let mut hp_j_r = rd[0] * acc.right_j[0];
                    let mut hp_ij_r = rd[0] * acc.right_ij[0];

                    let mut h_lr = 0.0;
                    let mut hp_lr = 0.0;
                    let mut h_i_lr = 0.0;
                    let mut h_j_lr = 0.0;
                    let mut h_ij_lr = 0.0;
                    let mut hp_i_lr = 0.0;
                    let mut hp_j_lr = 0.0;
                    let mut hp_ij_lr = 0.0;

                    for k in 1..p_resp {
                        let g = acc.gamma[k];
                        let gi = acc.gamma_i[k];
                        let gj = acc.gamma_j[k];
                        let gij = acc.gamma_ij[k];
                        let l = acc.left[k];
                        let li = acc.left_i[k];
                        let lj = acc.left_j[k];
                        let lij = acc.left_ij[k];
                        let r = acc.right[k];
                        let ri = acc.right_i[k];
                        let rj = acc.right_j[k];
                        let rij = acc.right_ij[k];

                        h_i += 2.0 * rv[k] * g * gi;
                        h_j += 2.0 * rv[k] * g * gj;
                        h_ij += 2.0 * rv[k] * (gj * gi + g * gij);
                        hp_i += 2.0 * rd[k] * g * gi;
                        hp_j += 2.0 * rd[k] * g * gj;
                        hp_ij += 2.0 * rd[k] * (gj * gi + g * gij);

                        h_l += 2.0 * rv[k] * g * l;
                        hp_l += 2.0 * rd[k] * g * l;
                        h_i_l += 2.0 * rv[k] * (l * gi + g * li);
                        h_j_l += 2.0 * rv[k] * (l * gj + g * lj);
                        h_ij_l += 2.0 * rv[k] * (lj * gi + gj * li + l * gij + g * lij);
                        hp_i_l += 2.0 * rd[k] * (l * gi + g * li);
                        hp_j_l += 2.0 * rd[k] * (l * gj + g * lj);
                        hp_ij_l += 2.0 * rd[k] * (lj * gi + gj * li + l * gij + g * lij);

                        h_r += 2.0 * rv[k] * g * r;
                        hp_r += 2.0 * rd[k] * g * r;
                        h_i_r += 2.0 * rv[k] * (r * gi + g * ri);
                        h_j_r += 2.0 * rv[k] * (r * gj + g * rj);
                        h_ij_r += 2.0 * rv[k] * (rj * gi + gj * ri + r * gij + g * rij);
                        hp_i_r += 2.0 * rd[k] * (r * gi + g * ri);
                        hp_j_r += 2.0 * rd[k] * (r * gj + g * rj);
                        hp_ij_r += 2.0 * rd[k] * (rj * gi + gj * ri + r * gij + g * rij);

                        h_lr += 2.0 * rv[k] * l * r;
                        hp_lr += 2.0 * rd[k] * l * r;
                        h_i_lr += 2.0 * rv[k] * (l * ri + r * li);
                        h_j_lr += 2.0 * rv[k] * (l * rj + r * lj);
                        h_ij_lr += 2.0 * rv[k] * (lj * ri + rj * li + l * rij + r * lij);
                        hp_i_lr += 2.0 * rd[k] * (l * ri + r * li);
                        hp_j_lr += 2.0 * rd[k] * (l * rj + r * lj);
                        hp_ij_lr += 2.0 * rd[k] * (lj * ri + rj * li + l * rij + r * lij);
                    }

                    let q = endpoint_q[row_idx];
                    let mut endpoint_i = [0.0; 2];
                    let mut endpoint_j = [0.0; 2];
                    let mut endpoint_ij = [0.0; 2];
                    let mut endpoint_l = [0.0; 2];
                    let mut endpoint_r = [0.0; 2];
                    let mut endpoint_i_l = [0.0; 2];
                    let mut endpoint_j_l = [0.0; 2];
                    let mut endpoint_ij_l = [0.0; 2];
                    let mut endpoint_i_r = [0.0; 2];
                    let mut endpoint_j_r = [0.0; 2];
                    let mut endpoint_ij_r = [0.0; 2];
                    let mut endpoint_l_r = [0.0; 2];
                    let mut endpoint_i_l_r = [0.0; 2];
                    let mut endpoint_j_l_r = [0.0; 2];
                    let mut endpoint_ij_l_r = [0.0; 2];
                    for e in 0..2 {
                        let basis = endpoint_basis[e];
                        endpoint_i[e] = basis[0] * acc.gamma_i[0];
                        endpoint_j[e] = basis[0] * acc.gamma_j[0];
                        endpoint_ij[e] = basis[0] * acc.gamma_ij[0];
                        endpoint_l[e] = basis[0] * acc.left[0];
                        endpoint_r[e] = basis[0] * acc.right[0];
                        endpoint_i_l[e] = basis[0] * acc.left_i[0];
                        endpoint_j_l[e] = basis[0] * acc.left_j[0];
                        endpoint_ij_l[e] = basis[0] * acc.left_ij[0];
                        endpoint_i_r[e] = basis[0] * acc.right_i[0];
                        endpoint_j_r[e] = basis[0] * acc.right_j[0];
                        endpoint_ij_r[e] = basis[0] * acc.right_ij[0];
                        for k in 1..p_resp {
                            let basis_k = basis[k];
                            let g = acc.gamma[k];
                            let gi = acc.gamma_i[k];
                            let gj = acc.gamma_j[k];
                            let gij = acc.gamma_ij[k];
                            let l = acc.left[k];
                            let li = acc.left_i[k];
                            let lj = acc.left_j[k];
                            let lij = acc.left_ij[k];
                            let r = acc.right[k];
                            let ri = acc.right_i[k];
                            let rj = acc.right_j[k];
                            let rij = acc.right_ij[k];
                            endpoint_i[e] += 2.0 * basis_k * g * gi;
                            endpoint_j[e] += 2.0 * basis_k * g * gj;
                            endpoint_ij[e] += 2.0 * basis_k * (gj * gi + g * gij);
                            endpoint_l[e] += 2.0 * basis_k * g * l;
                            endpoint_r[e] += 2.0 * basis_k * g * r;
                            endpoint_i_l[e] += 2.0 * basis_k * (l * gi + g * li);
                            endpoint_j_l[e] += 2.0 * basis_k * (l * gj + g * lj);
                            endpoint_ij_l[e] +=
                                2.0 * basis_k * (lj * gi + gj * li + l * gij + g * lij);
                            endpoint_i_r[e] += 2.0 * basis_k * (r * gi + g * ri);
                            endpoint_j_r[e] += 2.0 * basis_k * (r * gj + g * rj);
                            endpoint_ij_r[e] +=
                                2.0 * basis_k * (rj * gi + gj * ri + r * gij + g * rij);
                            endpoint_l_r[e] += 2.0 * basis_k * l * r;
                            endpoint_i_l_r[e] += 2.0 * basis_k * (l * ri + r * li);
                            endpoint_j_l_r[e] += 2.0 * basis_k * (l * rj + r * lj);
                            endpoint_ij_l_r[e] +=
                                2.0 * basis_k * (lj * ri + rj * li + l * rij + r * lij);
                        }
                    }

                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;
                    let numerator_l = hp_i_l * hp_j + hp_i * hp_j_l;
                    let numerator_r = hp_i_r * hp_j + hp_i * hp_j_r;
                    let numerator_lr =
                        hp_i_lr * hp_j + hp_i_l * hp_j_r + hp_i_r * hp_j_l + hp_i * hp_j_lr;
                    let value_lr = h_i_lr * h_j
                        + h_i_l * h_j_r
                        + h_i_r * h_j_l
                        + h_i * h_j_lr
                        + h_lr * h_ij
                        + h_l * h_ij_r
                        + h_r * h_ij_l
                        + h * h_ij_lr
                        - hp_ij_lr * inv_hp
                        + hp_ij_l * hp_r * inv_hp_sq
                        + hp_ij_r * hp_l * inv_hp_sq
                        + hp_ij * hp_lr * inv_hp_sq
                        - 2.0 * hp_ij * hp_l * hp_r * inv_hp_cu
                        + numerator_lr * inv_hp_sq
                        - 2.0 * numerator_l * hp_r * inv_hp_cu
                        - 2.0 * numerator_r * hp_l * inv_hp_cu
                        - 2.0 * hp_i * hp_j * hp_lr * inv_hp_cu
                        + 6.0 * hp_i * hp_j * hp_l * hp_r * inv_hp_qu
                        + endpoint_chain_fourth(
                            &q,
                            endpoint_i,
                            endpoint_j,
                            endpoint_l,
                            endpoint_r,
                            endpoint_ij,
                            endpoint_i_l,
                            endpoint_i_r,
                            endpoint_j_l,
                            endpoint_j_r,
                            endpoint_l_r,
                            endpoint_ij_l,
                            endpoint_ij_r,
                            endpoint_i_l_r,
                            endpoint_j_l_r,
                            endpoint_ij_l_r,
                        );
                    acc.value += weights[global_row] * value_lr;
                    acc
                },
            )
            .reduce(
                || PsiPairBilinearAccum::new(p_resp),
                |mut left, right| {
                    left.value += right.value;
                    left
                },
            )
            .value;
        Ok(total)
    }

    fn scop_psi_psi_trace_factor_from_cov(
        &self,
        beta: &Array1<f64>,
        cached_gamma: ArrayView2<'_, f64>,
        cached_h: ArrayView1<'_, f64>,
        cached_h_prime: ArrayView1<'_, f64>,
        cov: ArrayView2<'_, f64>,
        cov_i: ArrayView2<'_, f64>,
        cov_j: ArrayView2<'_, f64>,
        cov_ij: ArrayView2<'_, f64>,
        row_start: usize,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        factor: ArrayView2<'_, f64>,
    ) -> Result<f64, String> {
        let total_n = self.response_val_basis.nrows();
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let rank = factor.ncols();
        if row_start > total_n || row_start + n > total_n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi projected trace row window [{row_start}, {}) exceeds n={total_n}",
                    row_start + n
                ),
            }
            .into());
        }
        if beta.len() != p_total || factor.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi projected trace length mismatch: beta={}, factor_rows={}, expected={p_total}",
                beta.len(),
                factor.nrows()
            ) }.into());
        }
        if cached_gamma.nrows() != n || cached_gamma.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi projected trace gamma cache shape {}x{} != expected {}x{}",
                    cached_gamma.nrows(),
                    cached_gamma.ncols(),
                    n,
                    p_resp
                ),
            }
            .into());
        }
        let factor_data = factor.as_slice().ok_or_else(|| {
            "SCOP psi-psi projected trace factor matrix must be standard contiguous".to_string()
        })?;
        if endpoint_q.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi projected trace endpoint normalizer cache length {} != n={n}",
                    endpoint_q.len()
                ),
            }
            .into());
        }
        if cached_h.len() != n || cached_h_prime.len() != n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi projected trace row-quantity cache length mismatch: h={}, h_prime={}, expected={n}",
                cached_h.len(),
                cached_h_prime.len()
            ) }.into());
        }
        for (name, mat) in [
            ("cov", cov),
            ("cov_i", cov_i),
            ("cov_j", cov_j),
            ("cov_ij", cov_ij),
        ] {
            if mat.nrows() != n || mat.ncols() != p_cov {
                return Err(TransformationNormalError::InvalidInput {
                    reason: format!(
                        "SCOP psi-psi projected trace {name} shape {}x{} != expected {}x{}",
                        mat.nrows(),
                        mat.ncols(),
                        n,
                        p_cov
                    ),
                }
                .into());
            }
        }

        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi-psi projected trace beta reshape failed: {e}"))?;
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];

        struct PsiPairTraceAccum {
            value: f64,
            gamma: Vec<f64>,
            gamma_i: Vec<f64>,
            gamma_j: Vec<f64>,
            gamma_ij: Vec<f64>,
            f: Vec<f64>,
            f_i: Vec<f64>,
            f_j: Vec<f64>,
            f_ij: Vec<f64>,
        }

        impl PsiPairTraceAccum {
            fn new(p_resp: usize, rank: usize) -> Self {
                let projected_len = p_resp * rank;
                Self {
                    value: 0.0,
                    gamma: vec![0.0; p_resp],
                    gamma_i: vec![0.0; p_resp],
                    gamma_j: vec![0.0; p_resp],
                    gamma_ij: vec![0.0; p_resp],
                    f: vec![0.0; projected_len],
                    f_i: vec![0.0; projected_len],
                    f_j: vec![0.0; projected_len],
                    f_ij: vec![0.0; projected_len],
                }
            }
        }

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let weights = self.effective_weights();
        let total = (0..n)
            .into_par_iter()
            .fold(
                || PsiPairTraceAccum::new(p_resp, rank),
                |mut acc, row_idx| {
                    let cov_row = cov.row(row_idx);
                    let cov_i_row = cov_i.row(row_idx);
                    let cov_j_row = cov_j.row(row_idx);
                    let cov_ij_row = cov_ij.row(row_idx);
                    let global_row = row_start + row_idx;
                    let rv = self.response_val_basis.row(global_row);
                    let rd = self.response_deriv_basis.row(global_row);
                    let gamma_row = cached_gamma.row(row_idx);

                    for k in 0..p_resp {
                        let beta_k = beta_mat.row(k);
                        acc.gamma[k] = gamma_row[k];
                        acc.gamma_i[k] = beta_k.dot(&cov_i_row);
                        acc.gamma_j[k] = beta_k.dot(&cov_j_row);
                        acc.gamma_ij[k] = beta_k.dot(&cov_ij_row);
                    }

                    let h = cached_h[row_idx];
                    let hp = cached_h_prime[row_idx];
                    let mut h_i = rv[0] * acc.gamma_i[0];
                    let mut h_j = rv[0] * acc.gamma_j[0];
                    let mut h_ij = rv[0] * acc.gamma_ij[0];
                    let mut hp_i = rd[0] * acc.gamma_i[0];
                    let mut hp_j = rd[0] * acc.gamma_j[0];
                    let mut hp_ij = rd[0] * acc.gamma_ij[0];
                    for k in 1..p_resp {
                        let g = acc.gamma[k];
                        let gi = acc.gamma_i[k];
                        let gj = acc.gamma_j[k];
                        let gij = acc.gamma_ij[k];
                        h_i += 2.0 * rv[k] * g * gi;
                        h_j += 2.0 * rv[k] * g * gj;
                        h_ij += 2.0 * rv[k] * (gj * gi + g * gij);
                        hp_i += 2.0 * rd[k] * g * gi;
                        hp_j += 2.0 * rd[k] * g * gj;
                        hp_ij += 2.0 * rd[k] * (gj * gi + g * gij);
                    }

                    let q = endpoint_q[row_idx];
                    let mut endpoint_i = [0.0; 2];
                    let mut endpoint_j = [0.0; 2];
                    let mut endpoint_ij = [0.0; 2];
                    for e in 0..2 {
                        let basis = endpoint_basis[e];
                        endpoint_i[e] = basis[0] * acc.gamma_i[0];
                        endpoint_j[e] = basis[0] * acc.gamma_j[0];
                        endpoint_ij[e] = basis[0] * acc.gamma_ij[0];
                        for k in 1..p_resp {
                            endpoint_i[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_i[k];
                            endpoint_j[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_j[k];
                            endpoint_ij[e] += 2.0
                                * basis[k]
                                * (acc.gamma_j[k] * acc.gamma_i[k]
                                    + acc.gamma[k] * acc.gamma_ij[k]);
                        }
                    }

                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;
                    let wi = weights[global_row];

                    acc.f.fill(0.0);
                    acc.f_i.fill(0.0);
                    acc.f_j.fill(0.0);
                    acc.f_ij.fill(0.0);
                    for k in 0..p_resp {
                        let factor_row_base = k * p_cov;
                        let projected_base = k * rank;
                        for cidx in 0..p_cov {
                            let coeff_base = (factor_row_base + cidx) * rank;
                            let cov_v = cov_row[cidx];
                            let cov_i_v = cov_i_row[cidx];
                            let cov_j_v = cov_j_row[cidx];
                            let cov_ij_v = cov_ij_row[cidx];
                            for col in 0..rank {
                                let coeff = factor_data[coeff_base + col];
                                let idx = projected_base + col;
                                acc.f[idx] += coeff * cov_v;
                                acc.f_i[idx] += coeff * cov_i_v;
                                acc.f_j[idx] += coeff * cov_j_v;
                                acc.f_ij[idx] += coeff * cov_ij_v;
                            }
                        }
                    }

                    for col in 0..rank {
                        let mut h_f = rv[0] * acc.f[col];
                        let mut hp_f = rd[0] * acc.f[col];
                        let mut h_i_f = rv[0] * acc.f_i[col];
                        let mut h_j_f = rv[0] * acc.f_j[col];
                        let mut h_ij_f = rv[0] * acc.f_ij[col];
                        let mut hp_i_f = rd[0] * acc.f_i[col];
                        let mut hp_j_f = rd[0] * acc.f_j[col];
                        let mut hp_ij_f = rd[0] * acc.f_ij[col];

                        let mut h_ff = 0.0;
                        let mut hp_ff = 0.0;
                        let mut h_i_ff = 0.0;
                        let mut h_j_ff = 0.0;
                        let mut h_ij_ff = 0.0;
                        let mut hp_i_ff = 0.0;
                        let mut hp_j_ff = 0.0;
                        let mut hp_ij_ff = 0.0;

                        for k in 1..p_resp {
                            let g = acc.gamma[k];
                            let gi = acc.gamma_i[k];
                            let gj = acc.gamma_j[k];
                            let gij = acc.gamma_ij[k];
                            let projected_idx = k * rank + col;
                            let f = acc.f[projected_idx];
                            let fi = acc.f_i[projected_idx];
                            let fj = acc.f_j[projected_idx];
                            let fij = acc.f_ij[projected_idx];

                            h_f += 2.0 * rv[k] * g * f;
                            hp_f += 2.0 * rd[k] * g * f;
                            h_i_f += 2.0 * rv[k] * (f * gi + g * fi);
                            h_j_f += 2.0 * rv[k] * (f * gj + g * fj);
                            h_ij_f += 2.0 * rv[k] * (fj * gi + gj * fi + f * gij + g * fij);
                            hp_i_f += 2.0 * rd[k] * (f * gi + g * fi);
                            hp_j_f += 2.0 * rd[k] * (f * gj + g * fj);
                            hp_ij_f += 2.0 * rd[k] * (fj * gi + gj * fi + f * gij + g * fij);

                            h_ff += 2.0 * rv[k] * f * f;
                            hp_ff += 2.0 * rd[k] * f * f;
                            h_i_ff += 4.0 * rv[k] * f * fi;
                            h_j_ff += 4.0 * rv[k] * f * fj;
                            h_ij_ff += 2.0 * rv[k] * (fj * fi + fj * fi + f * fij + f * fij);
                            hp_i_ff += 4.0 * rd[k] * f * fi;
                            hp_j_ff += 4.0 * rd[k] * f * fj;
                            hp_ij_ff += 2.0 * rd[k] * (fj * fi + fj * fi + f * fij + f * fij);
                        }

                        let mut endpoint_f = [0.0; 2];
                        let mut endpoint_i_f = [0.0; 2];
                        let mut endpoint_j_f = [0.0; 2];
                        let mut endpoint_ij_f = [0.0; 2];
                        let mut endpoint_ff = [0.0; 2];
                        let mut endpoint_i_ff = [0.0; 2];
                        let mut endpoint_j_ff = [0.0; 2];
                        let mut endpoint_ij_ff = [0.0; 2];
                        for e in 0..2 {
                            let basis = endpoint_basis[e];
                            endpoint_f[e] = basis[0] * acc.f[col];
                            endpoint_i_f[e] = basis[0] * acc.f_i[col];
                            endpoint_j_f[e] = basis[0] * acc.f_j[col];
                            endpoint_ij_f[e] = basis[0] * acc.f_ij[col];
                            for k in 1..p_resp {
                                let basis_k = basis[k];
                                let g = acc.gamma[k];
                                let gi = acc.gamma_i[k];
                                let gj = acc.gamma_j[k];
                                let gij = acc.gamma_ij[k];
                                let projected_idx = k * rank + col;
                                let f = acc.f[projected_idx];
                                let fi = acc.f_i[projected_idx];
                                let fj = acc.f_j[projected_idx];
                                let fij = acc.f_ij[projected_idx];
                                endpoint_f[e] += 2.0 * basis_k * g * f;
                                endpoint_i_f[e] += 2.0 * basis_k * (f * gi + g * fi);
                                endpoint_j_f[e] += 2.0 * basis_k * (f * gj + g * fj);
                                endpoint_ij_f[e] +=
                                    2.0 * basis_k * (fj * gi + gj * fi + f * gij + g * fij);
                                endpoint_ff[e] += 2.0 * basis_k * f * f;
                                endpoint_i_ff[e] += 4.0 * basis_k * f * fi;
                                endpoint_j_ff[e] += 4.0 * basis_k * f * fj;
                                endpoint_ij_ff[e] += 4.0 * basis_k * (fj * fi + f * fij);
                            }
                        }

                        let numerator_f = hp_i_f * hp_j + hp_i * hp_j_f;
                        let numerator_ff = hp_i_ff * hp_j + 2.0 * hp_i_f * hp_j_f + hp_i * hp_j_ff;
                        let value_ff = h_i_ff * h_j
                            + 2.0 * h_i_f * h_j_f
                            + h_i * h_j_ff
                            + h_ff * h_ij
                            + 2.0 * h_f * h_ij_f
                            + h * h_ij_ff
                            - hp_ij_ff * inv_hp
                            + 2.0 * hp_ij_f * hp_f * inv_hp_sq
                            + hp_ij * hp_ff * inv_hp_sq
                            - 2.0 * hp_ij * hp_f * hp_f * inv_hp_cu
                            + numerator_ff * inv_hp_sq
                            - 4.0 * numerator_f * hp_f * inv_hp_cu
                            - 2.0 * hp_i * hp_j * hp_ff * inv_hp_cu
                            + 6.0 * hp_i * hp_j * hp_f * hp_f * inv_hp_qu
                            + endpoint_chain_fourth(
                                &q,
                                endpoint_i,
                                endpoint_j,
                                endpoint_f,
                                endpoint_f,
                                endpoint_ij,
                                endpoint_i_f,
                                endpoint_i_f,
                                endpoint_j_f,
                                endpoint_j_f,
                                endpoint_ff,
                                endpoint_ij_f,
                                endpoint_ij_f,
                                endpoint_i_ff,
                                endpoint_j_ff,
                                endpoint_ij_ff,
                            );
                        acc.value += wi * value_ff;
                    }
                    acc
                },
            )
            .reduce(
                || PsiPairTraceAccum::new(p_resp, rank),
                |mut left, right| {
                    left.value += right.value;
                    left
                },
            )
            .value;
        Ok(total)
    }

    fn scop_psi_pair_rows_per_chunk(&self, p_cov: usize) -> usize {
        let policy = ResourcePolicy::default_library();
        crate::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, 4 * p_cov.max(1))
            .max(1)
    }

    fn scop_psi_pair_cov_chunks(
        &self,
        op: &TensorKroneckerPsiOperator,
        axis_i: usize,
        axis_j: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), String> {
        let cov = self
            .covariate_dense_arc()?
            .slice(s![rows.clone(), ..])
            .to_owned();
        let cov_i = op
            .cov_first_axis_row_chunk(axis_i, rows.clone())
            .map_err(|e| format!("SCOP psi-psi covariate first-axis row chunk(i) failed: {e}"))?;
        let cov_j = op
            .cov_first_axis_row_chunk(axis_j, rows.clone())
            .map_err(|e| format!("SCOP psi-psi covariate first-axis row chunk(j) failed: {e}"))?;
        let cov_ij = op
            .cov_second_axis_row_chunk(axis_i, axis_j, rows)
            .map_err(|e| format!("SCOP psi-psi covariate second-axis row chunk failed: {e}"))?;
        Ok((cov, cov_i, cov_j, cov_ij))
    }

    fn scop_psi_psi_value_score_hvp_from_operator(
        &self,
        beta: &Array1<f64>,
        op: &TensorKroneckerPsiOperator,
        axis_i: usize,
        axis_j: usize,
        cached_gamma: ArrayView2<'_, f64>,
        cached_h: ArrayView1<'_, f64>,
        cached_h_prime: ArrayView1<'_, f64>,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        direction: Option<&Array1<f64>>,
    ) -> Result<(f64, Array1<f64>, Option<Array1<f64>>), String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if endpoint_q.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi operator endpoint normalizer cache length {} != n={n}",
                    endpoint_q.len()
                ),
            }
            .into());
        }
        if cached_h.len() != n || cached_h_prime.len() != n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi operator row-quantity cache length mismatch: h={}, h_prime={}, expected={n}",
                cached_h.len(),
                cached_h_prime.len()
            ) }.into());
        }
        if cached_gamma.nrows() != n || cached_gamma.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi operator gamma cache shape {}x{} != expected {}x{}",
                    cached_gamma.nrows(),
                    cached_gamma.ncols(),
                    n,
                    p_resp
                ),
            }
            .into());
        }
        let rows_per_chunk = self.scop_psi_pair_rows_per_chunk(p_cov).min(n.max(1));
        let mut objective = 0.0;
        let mut score = Array1::<f64>::zeros(p_total);
        let mut hvp = direction.map(|_| Array1::<f64>::zeros(p_total));

        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let (cov, cov_i, cov_j, cov_ij) =
                self.scop_psi_pair_cov_chunks(op, axis_i, axis_j, rows.clone())?;
            let (obj_chunk, score_chunk, hvp_chunk) = self.scop_psi_psi_value_score_hvp_from_cov(
                beta,
                cached_gamma.slice(s![start..end, ..]),
                cached_h.slice(s![start..end]),
                cached_h_prime.slice(s![start..end]),
                cov.view(),
                cov_i.view(),
                cov_j.view(),
                cov_ij.view(),
                start,
                &endpoint_q[start..end],
                direction,
            )?;
            objective += obj_chunk;
            score.scaled_add(1.0, &score_chunk);
            if let (Some(total), Some(chunk)) = (hvp.as_mut(), hvp_chunk.as_ref()) {
                total.scaled_add(1.0, chunk);
            }
        }

        Ok((objective, score, hvp))
    }

    fn scop_psi_psi_bilinear_from_operator(
        &self,
        beta: &Array1<f64>,
        op: &TensorKroneckerPsiOperator,
        axis_i: usize,
        axis_j: usize,
        cached_gamma: ArrayView2<'_, f64>,
        cached_h: ArrayView1<'_, f64>,
        cached_h_prime: ArrayView1<'_, f64>,
        endpoint_q: &[LogNormalCdfDiffDerivatives],
        left: ArrayView1<'_, f64>,
        right: ArrayView1<'_, f64>,
    ) -> Result<f64, String> {
        let n = self.response_val_basis.nrows();
        let p_cov = self.covariate_design.ncols();
        if endpoint_q.len() != n {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi bilinear operator endpoint normalizer cache length {} != n={n}",
                    endpoint_q.len()
                ),
            }
            .into());
        }
        if cached_h.len() != n || cached_h_prime.len() != n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi-psi bilinear operator row-quantity cache length mismatch: h={}, h_prime={}, expected={n}",
                cached_h.len(),
                cached_h_prime.len()
            ) }.into());
        }
        let p_resp = self.response_val_basis.ncols();
        if cached_gamma.nrows() != n || cached_gamma.ncols() != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi bilinear operator gamma cache shape {}x{} != expected {}x{}",
                    cached_gamma.nrows(),
                    cached_gamma.ncols(),
                    n,
                    p_resp
                ),
            }
            .into());
        }
        let rows_per_chunk = self.scop_psi_pair_rows_per_chunk(p_cov).min(n.max(1));
        let mut total = 0.0;
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let (cov, cov_i, cov_j, cov_ij) =
                self.scop_psi_pair_cov_chunks(op, axis_i, axis_j, rows.clone())?;
            total += self.scop_psi_psi_bilinear_from_cov(
                beta,
                cached_gamma.slice(s![start..end, ..]),
                cached_h.slice(s![start..end]),
                cached_h_prime.slice(s![start..end]),
                cov.view(),
                cov_i.view(),
                cov_j.view(),
                cov_ij.view(),
                start,
                &endpoint_q[start..end],
                left,
                right,
            )?;
        }
        Ok(total)
    }

    fn scop_psi_hessian_directional_derivative(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        op: &TensorKroneckerPsiOperator,
        axis: usize,
    ) -> Result<Array2<f64>, String> {
        let n = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total || direction.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian directional derivative length mismatch: beta={}, direction={}, expected={p_total}",
                beta.len(),
                direction.len()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi hessian beta reshape failed: {e}"))?;
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi hessian direction reshape failed: {e}"))?;
        let cov = self.covariate_dense_arc().map_err(|e| {
            format!("SCOP psi hessian direction requires cached covariate design: {e}")
        })?;
        let cov_psi_arc = op
            .materialize_cov_first_axis_arc(axis)
            .map_err(|e| format!("SCOP psi hessian materialize_cov_first failed: {e}"))?;
        let cov_psi = cov_psi_arc.view();
        if cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi hessian covariate derivative shape {}x{} != expected {}x{}",
                    cov_psi.nrows(),
                    cov_psi.ncols(),
                    n,
                    p_cov
                ),
            }
            .into());
        }

        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let endpoint_basis = [
            self.response_upper_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint upper basis is not contiguous".to_string())?,
            self.response_lower_basis
                .as_slice()
                .ok_or_else(|| "SCOP endpoint lower basis is not contiguous".to_string())?,
        ];
        // Parallelise the outer `for i in 0..n` row accumulation across
        // Rayon threads. The inner k×l×c×d loop dominates wall-clock at
        // biobank scale; per-row contributions to `out` are independent
        // (each row only contributes additively), so a thread-local
        // accumulator + reduction is safe and gives ~Nthreads× wall-clock
        // win. Per-thread scratch buffers are created once via
        // `fold(|| init, …)` and reused across all rows assigned to that
        // thread.
        struct Scratch {
            out: Array2<f64>,
            gamma: Vec<f64>,
            gamma_dir: Vec<f64>,
            gamma_psi: Vec<f64>,
            gamma_psi_dir: Vec<f64>,
            endpoint_factor: Vec<[f64; 2]>,
            endpoint_factor_dir: Vec<[f64; 2]>,
            endpoint_psi_cov_factor: Vec<[f64; 2]>,
            endpoint_psi_psi_factor: Vec<[f64; 2]>,
            endpoint_psi_cov_factor_dir: Vec<[f64; 2]>,
            endpoint_psi_psi_factor_dir: Vec<[f64; 2]>,
            h_factor: Vec<f64>,
            hp_factor: Vec<f64>,
            h_factor_dir: Vec<f64>,
            hp_factor_dir: Vec<f64>,
            hpsi_cov_factor: Vec<f64>,
            hppsi_cov_factor: Vec<f64>,
            hpsi_psi_factor: Vec<f64>,
            hppsi_psi_factor: Vec<f64>,
            hpsi_cov_factor_dir: Vec<f64>,
            hppsi_cov_factor_dir: Vec<f64>,
            hpsi_psi_factor_dir: Vec<f64>,
            hppsi_psi_factor_dir: Vec<f64>,
        }
        let init_scratch = || Scratch {
            out: Array2::<f64>::zeros((p_total, p_total)),
            gamma: vec![0.0; p_resp],
            gamma_dir: vec![0.0; p_resp],
            gamma_psi: vec![0.0; p_resp],
            gamma_psi_dir: vec![0.0; p_resp],
            endpoint_factor: vec![[0.0_f64; 2]; p_resp],
            endpoint_factor_dir: vec![[0.0_f64; 2]; p_resp],
            endpoint_psi_cov_factor: vec![[0.0_f64; 2]; p_resp],
            endpoint_psi_psi_factor: vec![[0.0_f64; 2]; p_resp],
            endpoint_psi_cov_factor_dir: vec![[0.0_f64; 2]; p_resp],
            endpoint_psi_psi_factor_dir: vec![[0.0_f64; 2]; p_resp],
            h_factor: vec![0.0; p_resp],
            hp_factor: vec![0.0; p_resp],
            h_factor_dir: vec![0.0; p_resp],
            hp_factor_dir: vec![0.0; p_resp],
            hpsi_cov_factor: vec![0.0; p_resp],
            hppsi_cov_factor: vec![0.0; p_resp],
            hpsi_psi_factor: vec![0.0; p_resp],
            hppsi_psi_factor: vec![0.0; p_resp],
            hpsi_cov_factor_dir: vec![0.0; p_resp],
            hppsi_cov_factor_dir: vec![0.0; p_resp],
            hpsi_psi_factor_dir: vec![0.0; p_resp],
            hppsi_psi_factor_dir: vec![0.0; p_resp],
        };

        use rayon::prelude::*;
        let process_row = |scratch: &mut Scratch, i: usize| {
            let Scratch {
                out,
                gamma,
                gamma_dir,
                gamma_psi,
                gamma_psi_dir,
                endpoint_factor,
                endpoint_factor_dir,
                endpoint_psi_cov_factor,
                endpoint_psi_psi_factor,
                endpoint_psi_cov_factor_dir,
                endpoint_psi_psi_factor_dir,
                h_factor,
                hp_factor,
                h_factor_dir,
                hp_factor_dir,
                hpsi_cov_factor,
                hppsi_cov_factor,
                hpsi_psi_factor,
                hppsi_psi_factor,
                hpsi_cov_factor_dir,
                hppsi_cov_factor_dir,
                hpsi_psi_factor_dir,
                hppsi_psi_factor_dir,
            } = scratch;
            let cov_row = cov.row(i);
            let psi_row = cov_psi.row(i);
            let rv = self.response_val_basis.row(i);
            let rd = self.response_deriv_basis.row(i);
            let wi = weights[i];
            let hp = h_prime[i];
            let inv_hp = 1.0 / hp;
            let inv_hp_sq = inv_hp * inv_hp;
            let inv_hp_cu = inv_hp_sq * inv_hp;
            let inv_hp_qu = inv_hp_sq * inv_hp_sq;

            // Reset endpoint/factor buffers whose index 0 is never written
            // explicitly below — original code relied on `vec![0.0; …]`
            // zero-init for those slots. The four `gamma` Vecs are fully
            // overwritten in the loop just below, so they don't need filling.
            endpoint_factor.fill([0.0; 2]);
            endpoint_factor_dir.fill([0.0; 2]);
            endpoint_psi_cov_factor.fill([0.0; 2]);
            endpoint_psi_psi_factor.fill([0.0; 2]);
            endpoint_psi_cov_factor_dir.fill([0.0; 2]);
            endpoint_psi_psi_factor_dir.fill([0.0; 2]);
            h_factor.fill(0.0);
            hp_factor.fill(0.0);
            h_factor_dir.fill(0.0);
            hp_factor_dir.fill(0.0);
            hpsi_cov_factor.fill(0.0);
            hppsi_cov_factor.fill(0.0);
            hpsi_psi_factor.fill(0.0);
            hppsi_psi_factor.fill(0.0);
            hpsi_cov_factor_dir.fill(0.0);
            hppsi_cov_factor_dir.fill(0.0);
            hpsi_psi_factor_dir.fill(0.0);
            hppsi_psi_factor_dir.fill(0.0);
            for k in 0..p_resp {
                gamma[k] = beta_mat.row(k).dot(&cov_row);
                gamma_dir[k] = dir_mat.row(k).dot(&cov_row);
                gamma_psi[k] = beta_mat.row(k).dot(&psi_row);
                gamma_psi_dir[k] = dir_mat.row(k).dot(&psi_row);
            }

            let mut h_dir = rv[0] * gamma_dir[0];
            let mut hp_dir = rd[0] * gamma_dir[0];
            let mut hp_psi = rd[0] * gamma_psi[0];
            let mut h_psi_dir = rv[0] * gamma_psi_dir[0];
            let mut hp_psi_dir = rd[0] * gamma_psi_dir[0];
            for k in 1..p_resp {
                h_dir += 2.0 * rv[k] * gamma[k] * gamma_dir[k];
                hp_dir += 2.0 * rd[k] * gamma[k] * gamma_dir[k];
                hp_psi += 2.0 * rd[k] * gamma[k] * gamma_psi[k];
                h_psi_dir +=
                    2.0 * rv[k] * (gamma_dir[k] * gamma_psi[k] + gamma[k] * gamma_psi_dir[k]);
                hp_psi_dir +=
                    2.0 * rd[k] * (gamma_dir[k] * gamma_psi[k] + gamma[k] * gamma_psi_dir[k]);
            }
            let q = row_quantities.endpoint_q[i];
            let mut endpoint_psi = [0.0; 2];
            let mut endpoint_dir = [0.0; 2];
            let mut endpoint_psi_dir = [0.0; 2];
            for e in 0..2 {
                let basis = endpoint_basis[e];
                endpoint_psi[e] = basis[0] * gamma_psi[0];
                endpoint_dir[e] = basis[0] * gamma_dir[0];
                endpoint_psi_dir[e] = basis[0] * gamma_psi_dir[0];
                endpoint_factor[0][e] = basis[0];
                endpoint_psi_psi_factor[0][e] = basis[0];
                for k in 1..p_resp {
                    endpoint_psi[e] += 2.0 * basis[k] * gamma[k] * gamma_psi[k];
                    endpoint_dir[e] += 2.0 * basis[k] * gamma[k] * gamma_dir[k];
                    endpoint_psi_dir[e] += 2.0
                        * basis[k]
                        * (gamma_dir[k] * gamma_psi[k] + gamma[k] * gamma_psi_dir[k]);
                    endpoint_factor[k][e] = 2.0 * basis[k] * gamma[k];
                    endpoint_factor_dir[k][e] = 2.0 * basis[k] * gamma_dir[k];
                    endpoint_psi_cov_factor[k][e] = 2.0 * basis[k] * gamma_psi[k];
                    endpoint_psi_psi_factor[k][e] = 2.0 * basis[k] * gamma[k];
                    endpoint_psi_cov_factor_dir[k][e] = 2.0 * basis[k] * gamma_psi_dir[k];
                    endpoint_psi_psi_factor_dir[k][e] = 2.0 * basis[k] * gamma_dir[k];
                }
            }
            let d_inv_hp = -hp_dir * inv_hp_sq;
            let d_inv_hp_sq = -2.0 * hp_dir * inv_hp_cu;
            let d_inv_hp_cu = -3.0 * hp_dir * inv_hp_qu;

            h_factor[0] = rv[0];
            hp_factor[0] = rd[0];
            hpsi_psi_factor[0] = rv[0];
            hppsi_psi_factor[0] = rd[0];
            for k in 1..p_resp {
                h_factor[k] = 2.0 * rv[k] * gamma[k];
                hp_factor[k] = 2.0 * rd[k] * gamma[k];
                h_factor_dir[k] = 2.0 * rv[k] * gamma_dir[k];
                hp_factor_dir[k] = 2.0 * rd[k] * gamma_dir[k];
                hpsi_cov_factor[k] = 2.0 * rv[k] * gamma_psi[k];
                hppsi_cov_factor[k] = 2.0 * rd[k] * gamma_psi[k];
                hpsi_psi_factor[k] = 2.0 * rv[k] * gamma[k];
                hppsi_psi_factor[k] = 2.0 * rd[k] * gamma[k];
                hpsi_cov_factor_dir[k] = 2.0 * rv[k] * gamma_psi_dir[k];
                hppsi_cov_factor_dir[k] = 2.0 * rd[k] * gamma_psi_dir[k];
                hpsi_psi_factor_dir[k] = 2.0 * rv[k] * gamma_dir[k];
                hppsi_psi_factor_dir[k] = 2.0 * rd[k] * gamma_dir[k];
            }

            for k in 0..p_resp {
                for l in 0..p_resp {
                    let same_shape = k == l && k > 0;
                    for c in 0..p_cov {
                        let row_idx = k * p_cov + c;
                        let h_a = h_factor[k] * cov_row[c];
                        let hp_a = hp_factor[k] * cov_row[c];
                        let h_a_dir = h_factor_dir[k] * cov_row[c];
                        let hp_a_dir = hp_factor_dir[k] * cov_row[c];
                        let hpsi_a =
                            hpsi_cov_factor[k] * cov_row[c] + hpsi_psi_factor[k] * psi_row[c];
                        let hppsi_a =
                            hppsi_cov_factor[k] * cov_row[c] + hppsi_psi_factor[k] * psi_row[c];
                        let hpsi_a_dir = hpsi_cov_factor_dir[k] * cov_row[c]
                            + hpsi_psi_factor_dir[k] * psi_row[c];
                        let hppsi_a_dir = hppsi_cov_factor_dir[k] * cov_row[c]
                            + hppsi_psi_factor_dir[k] * psi_row[c];
                        for d in 0..p_cov {
                            let col_idx = l * p_cov + d;
                            let h_b = h_factor[l] * cov_row[d];
                            let hp_b = hp_factor[l] * cov_row[d];
                            let h_b_dir = h_factor_dir[l] * cov_row[d];
                            let hp_b_dir = hp_factor_dir[l] * cov_row[d];
                            let hpsi_b =
                                hpsi_cov_factor[l] * cov_row[d] + hpsi_psi_factor[l] * psi_row[d];
                            let hppsi_b =
                                hppsi_cov_factor[l] * cov_row[d] + hppsi_psi_factor[l] * psi_row[d];
                            let hpsi_b_dir = hpsi_cov_factor_dir[l] * cov_row[d]
                                + hpsi_psi_factor_dir[l] * psi_row[d];
                            let hppsi_b_dir = hppsi_cov_factor_dir[l] * cov_row[d]
                                + hppsi_psi_factor_dir[l] * psi_row[d];
                            let (h_ab, hp_ab, hpsi_ab, hppsi_ab) = if same_shape {
                                (
                                    2.0 * rv[k] * cov_row[c] * cov_row[d],
                                    2.0 * rd[k] * cov_row[c] * cov_row[d],
                                    2.0 * rv[k]
                                        * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                    2.0 * rd[k]
                                        * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                )
                            } else {
                                (0.0, 0.0, 0.0, 0.0)
                            };
                            let endpoint_a = [
                                endpoint_factor[k][0] * cov_row[c],
                                endpoint_factor[k][1] * cov_row[c],
                            ];
                            let endpoint_b = [
                                endpoint_factor[l][0] * cov_row[d],
                                endpoint_factor[l][1] * cov_row[d],
                            ];
                            let endpoint_psi_a = [
                                endpoint_psi_cov_factor[k][0] * cov_row[c]
                                    + endpoint_psi_psi_factor[k][0] * psi_row[c],
                                endpoint_psi_cov_factor[k][1] * cov_row[c]
                                    + endpoint_psi_psi_factor[k][1] * psi_row[c],
                            ];
                            let endpoint_psi_b = [
                                endpoint_psi_cov_factor[l][0] * cov_row[d]
                                    + endpoint_psi_psi_factor[l][0] * psi_row[d],
                                endpoint_psi_cov_factor[l][1] * cov_row[d]
                                    + endpoint_psi_psi_factor[l][1] * psi_row[d],
                            ];
                            let endpoint_a_dir = [
                                endpoint_factor_dir[k][0] * cov_row[c],
                                endpoint_factor_dir[k][1] * cov_row[c],
                            ];
                            let endpoint_b_dir = [
                                endpoint_factor_dir[l][0] * cov_row[d],
                                endpoint_factor_dir[l][1] * cov_row[d],
                            ];
                            let endpoint_psi_a_dir = [
                                endpoint_psi_cov_factor_dir[k][0] * cov_row[c]
                                    + endpoint_psi_psi_factor_dir[k][0] * psi_row[c],
                                endpoint_psi_cov_factor_dir[k][1] * cov_row[c]
                                    + endpoint_psi_psi_factor_dir[k][1] * psi_row[c],
                            ];
                            let endpoint_psi_b_dir = [
                                endpoint_psi_cov_factor_dir[l][0] * cov_row[d]
                                    + endpoint_psi_psi_factor_dir[l][0] * psi_row[d],
                                endpoint_psi_cov_factor_dir[l][1] * cov_row[d]
                                    + endpoint_psi_psi_factor_dir[l][1] * psi_row[d],
                            ];
                            let (endpoint_ab, endpoint_psi_ab) = if same_shape {
                                (
                                    [
                                        2.0 * endpoint_basis[0][k] * cov_row[c] * cov_row[d],
                                        2.0 * endpoint_basis[1][k] * cov_row[c] * cov_row[d],
                                    ],
                                    [
                                        2.0 * endpoint_basis[0][k]
                                            * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                        2.0 * endpoint_basis[1][k]
                                            * (psi_row[d] * cov_row[c] + psi_row[c] * cov_row[d]),
                                    ],
                                )
                            } else {
                                ([0.0; 2], [0.0; 2])
                            };
                            let numerator = hppsi_a * hp_b + hp_a * hppsi_b;
                            let numerator_dir = hppsi_a_dir * hp_b
                                + hppsi_a * hp_b_dir
                                + hp_a_dir * hppsi_b
                                + hp_a * hppsi_b_dir;
                            let barrier_product = hp_a * hp_b * hp_psi;
                            let barrier_product_dir = hp_a_dir * hp_b * hp_psi
                                + hp_a * hp_b_dir * hp_psi
                                + hp_a * hp_b * hp_psi_dir;
                            let value = hpsi_a_dir * h_b
                                + hpsi_a * h_b_dir
                                + h_a_dir * hpsi_b
                                + h_a * hpsi_b_dir
                                + h_psi_dir * h_ab
                                + h_dir * hpsi_ab
                                + numerator_dir * inv_hp_sq
                                + numerator * d_inv_hp_sq
                                - 2.0
                                    * (barrier_product_dir * inv_hp_cu
                                        + barrier_product * d_inv_hp_cu)
                                - hppsi_ab * d_inv_hp
                                + hp_ab * hp_psi_dir * inv_hp_sq
                                + hp_ab * hp_psi * d_inv_hp_sq
                                + endpoint_chain_fourth(
                                    &q,
                                    endpoint_psi,
                                    endpoint_a,
                                    endpoint_b,
                                    endpoint_dir,
                                    endpoint_psi_a,
                                    endpoint_psi_b,
                                    endpoint_psi_dir,
                                    endpoint_ab,
                                    endpoint_a_dir,
                                    endpoint_b_dir,
                                    endpoint_psi_ab,
                                    endpoint_psi_a_dir,
                                    endpoint_psi_b_dir,
                                    [0.0; 2],
                                    [0.0; 2],
                                );
                            out[[row_idx, col_idx]] += wi * value;
                        }
                    }
                }
            }
        };

        // Drive the per-row work in parallel. Each Rayon thread accumulates
        // into its own Scratch (out + scratch buffers); we then reduce the
        // per-thread `out` matrices by summation. Only `out` needs reducing —
        // the other scratch fields are temporaries.
        let mut out: Array2<f64> = (0..n)
            .into_par_iter()
            .fold(init_scratch, |mut scratch, i| {
                process_row(&mut scratch, i);
                scratch
            })
            .map(|s| s.out)
            .reduce(|| Array2::<f64>::zeros((p_total, p_total)), |a, b| a + b);

        // In-place symmetrization: avoid allocating an extra `p_total × p_total`
        // f64 matrix for `&out + &out.t()`. Mathematically identical to
        // `0.5 * (out + out.T)`.
        for i in 0..p_total {
            for j in (i + 1)..p_total {
                let s = 0.5 * (out[[i, j]] + out[[j, i]]);
                out[[i, j]] = s;
                out[[j, i]] = s;
            }
        }
        Ok(out)
    }

    fn scop_psi_hessian_directional_trace_factor_chunk_from_cov(
        &self,
        beta: &Array1<f64>,
        direction: &Array1<f64>,
        row_quantities: &TransformationNormalRowQuantityCache,
        row_start: usize,
        cov: ArrayView2<'_, f64>,
        cov_psi: ArrayView2<'_, f64>,
        factor: ArrayView2<'_, f64>,
        projected_cov_f: Option<ArrayView2<'_, f64>>,
        projected_psi_f: Option<ArrayView2<'_, f64>>,
    ) -> Result<f64, String> {
        let total_n = self.response_val_basis.nrows();
        let n = cov.nrows();
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        let rank = factor.ncols();
        if row_start > total_n || row_start + n > total_n {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian directional projected trace row window [{row_start}, {}) exceeds n={total_n}",
                row_start + n
            ) }.into());
        }
        if cov.ncols() != p_cov || cov_psi.nrows() != n || cov_psi.ncols() != p_cov {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian directional projected trace chunk shape mismatch: cov={}x{}, cov_psi={}x{}, expected n={} p_cov={}",
                cov.nrows(),
                cov.ncols(),
                cov_psi.nrows(),
                cov_psi.ncols(),
                n,
                p_cov
            ) }.into());
        }
        if beta.len() != p_total || direction.len() != p_total || factor.nrows() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "SCOP psi Hessian directional projected trace length mismatch: beta={}, direction={}, factor_rows={}, expected={p_total}",
                beta.len(),
                direction.len(),
                factor.nrows()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi directional trace beta reshape failed: {e}"))?;
        let dir_mat = direction
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("SCOP psi directional trace direction reshape failed: {e}"))?;
        let endpoint_basis = [
            self.response_upper_basis.as_slice().ok_or_else(|| {
                "SCOP psi directional trace endpoint upper basis is not contiguous".to_string()
            })?,
            self.response_lower_basis.as_slice().ok_or_else(|| {
                "SCOP psi directional trace endpoint lower basis is not contiguous".to_string()
            })?,
        ];

        struct PsiDhTraceAccum {
            value: f64,
            gamma: Vec<f64>,
            gamma_dir: Vec<f64>,
            gamma_psi: Vec<f64>,
            gamma_psi_dir: Vec<f64>,
            gamma_f: Vec<f64>,
            gamma_psi_f: Vec<f64>,
            h_f: Vec<f64>,
            hp_f: Vec<f64>,
            h_f_dir: Vec<f64>,
            hp_f_dir: Vec<f64>,
            h_ff: Vec<f64>,
            hp_ff: Vec<f64>,
            hpsi_f: Vec<f64>,
            hppsi_f: Vec<f64>,
            hpsi_f_dir: Vec<f64>,
            hppsi_f_dir: Vec<f64>,
            hpsi_ff: Vec<f64>,
            hppsi_ff: Vec<f64>,
            endpoint_f: Vec<[f64; 2]>,
            endpoint_f_dir: Vec<[f64; 2]>,
            endpoint_ff: Vec<[f64; 2]>,
            endpoint_psi_f: Vec<[f64; 2]>,
            endpoint_psi_f_dir: Vec<[f64; 2]>,
            endpoint_psi_ff: Vec<[f64; 2]>,
        }

        impl PsiDhTraceAccum {
            fn new(p_resp: usize, rank: usize) -> Self {
                let projected_len = p_resp * rank;
                Self {
                    value: 0.0,
                    gamma: vec![0.0; p_resp],
                    gamma_dir: vec![0.0; p_resp],
                    gamma_psi: vec![0.0; p_resp],
                    gamma_psi_dir: vec![0.0; p_resp],
                    gamma_f: vec![0.0; projected_len],
                    gamma_psi_f: vec![0.0; projected_len],
                    h_f: vec![0.0; rank],
                    hp_f: vec![0.0; rank],
                    h_f_dir: vec![0.0; rank],
                    hp_f_dir: vec![0.0; rank],
                    h_ff: vec![0.0; rank],
                    hp_ff: vec![0.0; rank],
                    hpsi_f: vec![0.0; rank],
                    hppsi_f: vec![0.0; rank],
                    hpsi_f_dir: vec![0.0; rank],
                    hppsi_f_dir: vec![0.0; rank],
                    hpsi_ff: vec![0.0; rank],
                    hppsi_ff: vec![0.0; rank],
                    endpoint_f: vec![[0.0; 2]; rank],
                    endpoint_f_dir: vec![[0.0; 2]; rank],
                    endpoint_ff: vec![[0.0; 2]; rank],
                    endpoint_psi_f: vec![[0.0; 2]; rank],
                    endpoint_psi_f_dir: vec![[0.0; 2]; rank],
                    endpoint_psi_ff: vec![[0.0; 2]; rank],
                }
            }

            fn merge(mut self, rhs: Self) -> Self {
                self.value += rhs.value;
                self
            }
        }

        // Project the low-rank REML factor through each response-slice of the
        // covariate design with BLAS instead of recomputing
        // `sum_c F[(k,c), r] * x_ic` inside every row. This is an exact
        // algebraic refactor of the old row loop: for each response basis `k`,
        // `projected_cov_f[:, k, :] = cov · factor_k` and
        // `projected_psi_f[:, k, :] = cov_psi · factor_k`. Callers that sweep
        // multiple ψ-Hessian directional traces may supply cached projections;
        // otherwise we build the chunk-local projections here.
        let projected_len = p_resp * rank;
        let mut projected_cov_storage;
        let mut projected_psi_storage;
        let projected_cov_f = match projected_cov_f {
            Some(view) => {
                if view.nrows() != n || view.ncols() != projected_len {
                    return Err(format!(
                        "SCOP psi Hessian directional projected cov-factor shape {}x{} != expected {}x{}",
                        view.nrows(),
                        view.ncols(),
                        n,
                        projected_len
                    ));
                }
                view
            }
            None => {
                projected_cov_storage = Array2::<f64>::zeros((n, projected_len));
                if rank > 0 && n > 0 {
                    for k in 0..p_resp {
                        let factor_block = factor.slice(s![k * p_cov..(k + 1) * p_cov, ..]);
                        let cov_projection = fast_ab(&cov, &factor_block);
                        projected_cov_storage
                            .slice_mut(s![.., k * rank..(k + 1) * rank])
                            .assign(&cov_projection);
                    }
                }
                projected_cov_storage.view()
            }
        };
        let projected_psi_f = match projected_psi_f {
            Some(view) => {
                if view.nrows() != n || view.ncols() != projected_len {
                    return Err(format!(
                        "SCOP psi Hessian directional projected psi-factor shape {}x{} != expected {}x{}",
                        view.nrows(),
                        view.ncols(),
                        n,
                        projected_len
                    ));
                }
                view
            }
            None => {
                projected_psi_storage = Array2::<f64>::zeros((n, projected_len));
                if rank > 0 && n > 0 {
                    for k in 0..p_resp {
                        let factor_block = factor.slice(s![k * p_cov..(k + 1) * p_cov, ..]);
                        let psi_projection = fast_ab(&cov_psi, &factor_block);
                        projected_psi_storage
                            .slice_mut(s![.., k * rank..(k + 1) * rank])
                            .assign(&psi_projection);
                    }
                }
                projected_psi_storage.view()
            }
        };

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let weights = self.effective_weights();
        let h_prime = row_quantities.h_prime.as_ref();
        let accum = (0..n)
            .into_par_iter()
            .fold(
                || PsiDhTraceAccum::new(p_resp, rank),
                |mut acc, local_i| {
                    let i = row_start + local_i;
                    let cov_row = cov.row(local_i);
                    let psi_row = cov_psi.row(local_i);
                    let rv = self.response_val_basis.row(i);
                    let rd = self.response_deriv_basis.row(i);
                    let wi = weights[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;
                    let q = row_quantities.endpoint_q[i];

                    for k in 0..p_resp {
                        acc.gamma[k] = beta_mat.row(k).dot(&cov_row);
                        acc.gamma_dir[k] = dir_mat.row(k).dot(&cov_row);
                        acc.gamma_psi[k] = beta_mat.row(k).dot(&psi_row);
                        acc.gamma_psi_dir[k] = dir_mat.row(k).dot(&psi_row);
                    }

                    let projected_cov_row = projected_cov_f.row(local_i);
                    let projected_psi_row = projected_psi_f.row(local_i);
                    acc.gamma_f.copy_from_slice(
                        projected_cov_row
                            .as_slice()
                            .expect("projected CTN covariate-factor row should be contiguous"),
                    );
                    acc.gamma_psi_f.copy_from_slice(
                        projected_psi_row
                            .as_slice()
                            .expect("projected CTN psi-factor row should be contiguous"),
                    );

                    let mut hp_psi = rd[0] * acc.gamma_psi[0];
                    let mut h_dir = rv[0] * acc.gamma_dir[0];
                    let mut hp_dir = rd[0] * acc.gamma_dir[0];
                    let mut h_psi_dir = rv[0] * acc.gamma_psi_dir[0];
                    let mut hp_psi_dir = rd[0] * acc.gamma_psi_dir[0];
                    for k in 1..p_resp {
                        hp_psi += 2.0 * rd[k] * acc.gamma[k] * acc.gamma_psi[k];
                        h_dir += 2.0 * rv[k] * acc.gamma[k] * acc.gamma_dir[k];
                        hp_dir += 2.0 * rd[k] * acc.gamma[k] * acc.gamma_dir[k];
                        h_psi_dir += 2.0
                            * rv[k]
                            * (acc.gamma_dir[k] * acc.gamma_psi[k]
                                + acc.gamma[k] * acc.gamma_psi_dir[k]);
                        hp_psi_dir += 2.0
                            * rd[k]
                            * (acc.gamma_dir[k] * acc.gamma_psi[k]
                                + acc.gamma[k] * acc.gamma_psi_dir[k]);
                    }
                    let d_inv_hp = -hp_dir * inv_hp_sq;
                    let d_inv_hp_sq = -2.0 * hp_dir * inv_hp_cu;
                    let d_inv_hp_cu = -3.0 * hp_dir * inv_hp_qu;

                    let mut endpoint_psi = [0.0_f64; 2];
                    let mut endpoint_dir = [0.0_f64; 2];
                    let mut endpoint_psi_dir = [0.0_f64; 2];
                    for e in 0..2 {
                        let basis = endpoint_basis[e];
                        endpoint_psi[e] = basis[0] * acc.gamma_psi[0];
                        endpoint_dir[e] = basis[0] * acc.gamma_dir[0];
                        endpoint_psi_dir[e] = basis[0] * acc.gamma_psi_dir[0];
                        for k in 1..p_resp {
                            endpoint_psi[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_psi[k];
                            endpoint_dir[e] += 2.0 * basis[k] * acc.gamma[k] * acc.gamma_dir[k];
                            endpoint_psi_dir[e] += 2.0
                                * basis[k]
                                * (acc.gamma_dir[k] * acc.gamma_psi[k]
                                    + acc.gamma[k] * acc.gamma_psi_dir[k]);
                        }
                    }

                    for col in 0..rank {
                        acc.h_f[col] = rv[0] * acc.gamma_f[col];
                        acc.hp_f[col] = rd[0] * acc.gamma_f[col];
                        acc.h_f_dir[col] = 0.0;
                        acc.hp_f_dir[col] = 0.0;
                        acc.h_ff[col] = 0.0;
                        acc.hp_ff[col] = 0.0;
                        acc.hpsi_f[col] = rv[0] * acc.gamma_psi_f[col];
                        acc.hppsi_f[col] = rd[0] * acc.gamma_psi_f[col];
                        acc.hpsi_f_dir[col] = 0.0;
                        acc.hppsi_f_dir[col] = 0.0;
                        acc.hpsi_ff[col] = 0.0;
                        acc.hppsi_ff[col] = 0.0;
                        acc.endpoint_f[col] = [
                            endpoint_basis[0][0] * acc.gamma_f[col],
                            endpoint_basis[1][0] * acc.gamma_f[col],
                        ];
                        acc.endpoint_f_dir[col] = [0.0; 2];
                        acc.endpoint_ff[col] = [0.0; 2];
                        acc.endpoint_psi_f[col] = [
                            endpoint_basis[0][0] * acc.gamma_psi_f[col],
                            endpoint_basis[1][0] * acc.gamma_psi_f[col],
                        ];
                        acc.endpoint_psi_f_dir[col] = [0.0; 2];
                        acc.endpoint_psi_ff[col] = [0.0; 2];
                    }
                    for k in 1..p_resp {
                        let g = acc.gamma[k];
                        let gd = acc.gamma_dir[k];
                        let gp = acc.gamma_psi[k];
                        let gpd = acc.gamma_psi_dir[k];
                        for col in 0..rank {
                            let idx = k * rank + col;
                            let gf = acc.gamma_f[idx];
                            let gpf = acc.gamma_psi_f[idx];
                            acc.h_f[col] += 2.0 * rv[k] * g * gf;
                            acc.hp_f[col] += 2.0 * rd[k] * g * gf;
                            acc.h_f_dir[col] += 2.0 * rv[k] * gd * gf;
                            acc.hp_f_dir[col] += 2.0 * rd[k] * gd * gf;
                            acc.h_ff[col] += 2.0 * rv[k] * gf * gf;
                            acc.hp_ff[col] += 2.0 * rd[k] * gf * gf;
                            acc.hpsi_f[col] += 2.0 * rv[k] * (gf * gp + g * gpf);
                            acc.hppsi_f[col] += 2.0 * rd[k] * (gf * gp + g * gpf);
                            acc.hpsi_f_dir[col] += 2.0 * rv[k] * (gf * gpd + gd * gpf);
                            acc.hppsi_f_dir[col] += 2.0 * rd[k] * (gf * gpd + gd * gpf);
                            acc.hpsi_ff[col] += 4.0 * rv[k] * gf * gpf;
                            acc.hppsi_ff[col] += 4.0 * rd[k] * gf * gpf;
                            for e in 0..2 {
                                let basis = endpoint_basis[e];
                                acc.endpoint_f[col][e] += 2.0 * basis[k] * g * gf;
                                acc.endpoint_f_dir[col][e] += 2.0 * basis[k] * gd * gf;
                                acc.endpoint_ff[col][e] += 2.0 * basis[k] * gf * gf;
                                acc.endpoint_psi_f[col][e] += 2.0 * basis[k] * (gf * gp + g * gpf);
                                acc.endpoint_psi_f_dir[col][e] +=
                                    2.0 * basis[k] * (gf * gpd + gd * gpf);
                                acc.endpoint_psi_ff[col][e] += 4.0 * basis[k] * gf * gpf;
                            }
                        }
                    }

                    for col in 0..rank {
                        let numerator = 2.0 * acc.hppsi_f[col] * acc.hp_f[col];
                        let numerator_dir = 2.0
                            * (acc.hppsi_f_dir[col] * acc.hp_f[col]
                                + acc.hppsi_f[col] * acc.hp_f_dir[col]);
                        let barrier_product = acc.hp_f[col] * acc.hp_f[col] * hp_psi;
                        let barrier_product_dir = 2.0 * acc.hp_f_dir[col] * acc.hp_f[col] * hp_psi
                            + acc.hp_f[col] * acc.hp_f[col] * hp_psi_dir;
                        let value = 2.0 * acc.hpsi_f_dir[col] * acc.h_f[col]
                            + 2.0 * acc.hpsi_f[col] * acc.h_f_dir[col]
                            + h_psi_dir * acc.h_ff[col]
                            + h_dir * acc.hpsi_ff[col]
                            + numerator_dir * inv_hp_sq
                            + numerator * d_inv_hp_sq
                            - 2.0
                                * (barrier_product_dir * inv_hp_cu + barrier_product * d_inv_hp_cu)
                            - acc.hppsi_ff[col] * d_inv_hp
                            + acc.hp_ff[col] * hp_psi_dir * inv_hp_sq
                            + acc.hp_ff[col] * hp_psi * d_inv_hp_sq
                            + endpoint_chain_fourth(
                                &q,
                                endpoint_psi,
                                acc.endpoint_f[col],
                                acc.endpoint_f[col],
                                endpoint_dir,
                                acc.endpoint_psi_f[col],
                                acc.endpoint_psi_f[col],
                                endpoint_psi_dir,
                                acc.endpoint_ff[col],
                                acc.endpoint_f_dir[col],
                                acc.endpoint_f_dir[col],
                                acc.endpoint_psi_ff[col],
                                acc.endpoint_psi_f_dir[col],
                                acc.endpoint_psi_f_dir[col],
                                [0.0; 2],
                                [0.0; 2],
                            );
                        acc.value += wi * value;
                    }
                    acc
                },
            )
            .reduce(
                || PsiDhTraceAccum::new(p_resp, rank),
                |left, right| left.merge(right),
            );
        Ok(accum.value)
    }
}

fn ctn_penalty_scale_log_lambdas(
    penalties: &[PenaltyMatrix],
    likelihood_gram: &Array2<f64>,
) -> Array1<f64> {
    if penalties.is_empty() {
        return Array1::zeros(0);
    }

    let likelihood_scale = matrix_diag_mean_abs(likelihood_gram).max(1.0e-8);
    Array1::from_iter(penalties.iter().map(|penalty| {
        let penalty_scale = penalty_diag_scale(penalty).max(1.0e-8);
        // Lower-bound the SEED log-lambda at 0 (i.e., λ ≥ 1) so we never
        // start the outer optimizer in the under-regularized regime where
        // the CTN inner is structurally rank-deficient (small-n / p > n).
        // The outer BFGS is free to step below 0 when there's enough data
        // to support it; only the cold-start is constrained. Without this
        // floor, ratios like n=64, p_resp×p_cov ≈ 200 produce a seed of
        // log_lambda ≈ -12 (λ ≈ 6e-6), which leaves the inner solve to
        // pick wild β coefficients and cascade into predict-time monotonicity
        // violations (h' < 0 on the response grid, observed as -1e15 spikes
        // in CI synthetic-biobank tests).
        (likelihood_scale / penalty_scale).ln().clamp(0.0, 12.0)
    }))
}

fn penalty_diag_scale(penalty: &PenaltyMatrix) -> f64 {
    match penalty {
        PenaltyMatrix::Dense(matrix) => {
            matrix_diag_mean_abs(matrix).max(matrix_frobenius_rms(matrix))
        }
        PenaltyMatrix::KroneckerFactored { left, right } => {
            let diag_scale = matrix_diag_mean_abs(left) * matrix_diag_mean_abs(right);
            let frob_scale = matrix_frobenius_rms(left) * matrix_frobenius_rms(right);
            diag_scale.max(frob_scale)
        }
        PenaltyMatrix::Blockwise { local, .. } => {
            matrix_diag_mean_abs(local).max(matrix_frobenius_rms(local))
        }
        PenaltyMatrix::Labeled { inner, .. } => penalty_diag_scale(inner),
    }
}

fn matrix_diag_mean_abs(matrix: &Array2<f64>) -> f64 {
    let d = matrix.nrows().min(matrix.ncols());
    if d == 0 {
        return 0.0;
    }
    matrix.diag().iter().map(|v| v.abs()).sum::<f64>() / d as f64
}

fn matrix_frobenius_rms(matrix: &Array2<f64>) -> f64 {
    let d = matrix.nrows().max(1).min(matrix.ncols().max(1));
    (matrix.iter().map(|v| v * v).sum::<f64>() / d as f64).sqrt()
}

/// Weighted cross-product of two rowwise-Kronecker designs, kept strictly
/// factored: output block (a, c) equals `B^T diag(w_i A_{ia} C_{ic}) D`.
fn factored_weighted_cross(
    a: &Array2<f64>,
    b: &Array2<f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    c: &Array2<f64>,
    d: &Array2<f64>,
    policy: &ResourcePolicy,
) -> Result<Array2<f64>, String> {
    let n = weights.len();
    if a.nrows() != n || b.nrows() != n || c.nrows() != n || d.nrows() != n {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "factored_weighted_cross row mismatch: weights={n}, a={}, b={}, c={}, d={}",
                a.nrows(),
                b.nrows(),
                c.nrows(),
                d.nrows()
            ),
        }
        .into());
    }
    let pa = a.ncols();
    let pc = c.ncols();
    let pb = b.ncols();
    let pd = d.ncols();

    let mut out = Array2::<f64>::zeros((pa * pb, pc * pd));
    let mut pair_weights = Array1::<f64>::zeros(n);

    for ia in 0..pa {
        let a_col = a.column(ia);
        for ic in 0..pc {
            let c_col = c.column(ic);
            for r in 0..n {
                pair_weights[r] = weights[r] * a_col[r] * c_col[r];
            }
            let block = chunked_weighted_bt_d(b, pair_weights.view(), d, policy);
            let mut slice = out.slice_mut(s![ia * pb..(ia + 1) * pb, ic * pd..(ic + 1) * pd]);
            slice.assign(&block);
        }
    }

    Ok(out)
}

/// Chunked weighted B^T diag(w) D product without materializing any
/// full rowwise-Kronecker intermediate.
///
/// Each chunk weights `D` rows in-place, then accumulates `B[chunk]^T · DW`
/// directly into `out` via a faer SIMD matmul with `Accum::Add`. This
/// eliminates the per-chunk `Array2` from the previous `out += &bl.t().dot(&dw)`
/// pattern (one allocation + one element-wise `+=` pass per chunk) and uses
/// the multi-threaded faer kernel instead of ndarray's serial dot.
fn chunked_weighted_bt_d(
    b: &Array2<f64>,
    weights: ndarray::ArrayView1<'_, f64>,
    d: &Array2<f64>,
    policy: &ResourcePolicy,
) -> Array2<f64> {
    use crate::faer_ndarray::{FaerArrayView, array2_to_matmut, matmul_parallelism};
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let n = weights.len();
    let pb = b.ncols();
    let pd = d.ncols();
    let rows_per_chunk =
        crate::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, pb + pd);
    let mut out = Array2::<f64>::zeros((pb, pd));
    if n == 0 || pb == 0 || pd == 0 {
        return out;
    }
    let mut out_view = array2_to_matmut(&mut out);
    let mut dw_buf = Array2::<f64>::zeros((rows_per_chunk.min(n), pd));
    for start in (0..n).step_by(rows_per_chunk) {
        let end = (start + rows_per_chunk).min(n);
        let rows = end - start;
        let bl = b.slice(s![start..end, ..]);
        let dl = d.slice(s![start..end, ..]);
        {
            let mut dw_slice = dw_buf.slice_mut(s![..rows, ..]);
            for local in 0..rows {
                let w = weights[start + local];
                let drow = dl.row(local);
                let mut wrow = dw_slice.row_mut(local);
                ndarray::Zip::from(&mut wrow)
                    .and(&drow)
                    .for_each(|dst, &src| *dst = w * src);
            }
        }
        let bl_view = FaerArrayView::new(&bl);
        let dw_slice = dw_buf.slice(s![..rows, ..]);
        let dw_view = FaerArrayView::new(&dw_slice);
        let par = matmul_parallelism(pb, pd, rows);
        matmul(
            out_view.as_mut(),
            Accum::Add,
            bl_view.as_ref().transpose(),
            dw_view.as_ref(),
            1.0,
            par,
        );
    }
    out
}

/// Chunked weighted `B^T diag(w) D` product where `B` and `D` are
/// operator-backed `DesignMatrix` instances. Materializes only one row chunk
/// at a time using the operator's `row_chunk` primitive, so neither factor's
/// full dense form ever lives in memory.
///
/// Each chunk's contribution accumulates into `out` via a faer SIMD matmul
/// with `Accum::Add` rather than `out += &bl.t().dot(&dw)`. This drops one
/// `Array2` allocation per chunk and routes the inner GEMM through faer's
/// multi-threaded kernel with a work-aware parallelism choice.
fn chunked_weighted_bt_d_designmatrix(
    b: &DesignMatrix,
    weights: ndarray::ArrayView1<'_, f64>,
    d: &DesignMatrix,
    policy: &ResourcePolicy,
) -> Result<Array2<f64>, String> {
    use crate::faer_ndarray::{FaerArrayView, array2_to_matmut, matmul_parallelism};
    use faer::Accum;
    use faer::linalg::matmul::matmul;

    let n = weights.len();
    let pb = b.ncols();
    let pd = d.ncols();
    let rows_per_chunk =
        crate::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, pb + pd);
    let mut out = Array2::<f64>::zeros((pb, pd));
    if n == 0 || pb == 0 || pd == 0 {
        return Ok(out);
    }
    let mut out_view = array2_to_matmut(&mut out);
    for start in (0..n).step_by(rows_per_chunk) {
        let end = (start + rows_per_chunk).min(n);
        let rows = end - start;
        let bl = b.try_row_chunk(start..end).map_err(|e| e.to_string())?;
        let mut dw = d.try_row_chunk(start..end).map_err(|e| e.to_string())?;
        for local in 0..rows {
            let w = weights[start + local];
            if w != 1.0 {
                let mut wrow = dw.row_mut(local);
                wrow.mapv_inplace(|v| w * v);
            }
        }
        let bl_view = FaerArrayView::new(&bl);
        let dw_view = FaerArrayView::new(&dw);
        let par = matmul_parallelism(pb, pd, rows);
        matmul(
            out_view.as_mut(),
            Accum::Add,
            bl_view.as_ref().transpose(),
            dw_view.as_ref(),
            1.0,
            par,
        );
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// CustomFamily implementation
// ---------------------------------------------------------------------------

impl CustomFamily for TransformationNormalFamily {
    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        if block_states.len() != 1 {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "TransformationNormalFamily expects 1 block, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let evaluate_start = std::time::Instant::now();
        let beta = &block_states[0].beta;
        let row_q_start = std::time::Instant::now();
        let row_quantities = self.row_quantities(beta)?;
        log::info!(
            "[STAGE] CTN row_quantities (h, h', 1/h', powers) n={} elapsed={:.3}s",
            row_quantities.h.len(),
            row_q_start.elapsed().as_secs_f64(),
        );
        let h = row_quantities.h.as_ref();
        let n = h.len();

        let log_likelihood = row_quantities.log_likelihood;
        // SCOP gradient and exact negative Hessian. Response column 0 is the
        // linear location component b(x); response columns >=1 are squared
        // γ_k(x)^2 shape components.
        let grad_start = std::time::Instant::now();
        let (grad, hessian) = self.scop_gradient_and_negative_hessian(beta, &row_quantities)?;
        log::info!(
            "[STAGE] CTN gradient terms n={} p={} elapsed={:.3}s",
            n,
            grad.len(),
            grad_start.elapsed().as_secs_f64(),
        );

        let hess_start = std::time::Instant::now();
        let p_dim = hessian.nrows() as u64;
        let n_u64 = n as u64;
        log::info!(
            "[STAGE] CTN hessian terms (SCOP exact dense) n={} p={} flops~{} elapsed={:.3}s",
            n,
            p_dim,
            n_u64.saturating_mul(p_dim).saturating_mul(p_dim),
            hess_start.elapsed().as_secs_f64(),
        );
        log::info!(
            "[STAGE] CTN evaluate end n={} p={} elapsed={:.3}s",
            n,
            p_dim,
            evaluate_start.elapsed().as_secs_f64(),
        );

        Ok(FamilyEvaluation {
            log_likelihood,
            blockworking_sets: vec![BlockWorkingSet::ExactNewton {
                gradient: grad,
                hessian: SymmetricMatrix::Dense(hessian),
            }],
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        if block_states.len() != 1 {
            return Err(TransformationNormalError::InvalidInput {
                reason: "expected 1 block".to_string(),
            }
            .into());
        }
        // The line search uses NEG_INFINITY as the barrier-violation signal,
        // so we can't propagate the row_quantities Err here. Translate any
        // h' validation failure back into the NEG_INFINITY rejection contract.
        let row_quantities = match self.row_quantities(&block_states[0].beta) {
            Ok(rq) => rq,
            Err(_) => return Ok(f64::NEG_INFINITY),
        };
        Ok(row_quantities.log_likelihood)
    }

    fn log_likelihood_only_with_options(
        &self,
        block_states: &[ParameterBlockState],
        options: &BlockwiseFitOptions,
    ) -> Result<f64, String> {
        // When an outer-score subsample is installed, route through a
        // mask-aware family clone whose `effective_weights()` returns the
        // HT-weighted per-row weights. Because every term inside
        // `build_transformation_row_derived` is linear in `wᵢ`, the row-LL
        // accumulator yields `Σᵢ (mᵢ · wᵢ) · row_ll_i` — the unbiased
        // Horvitz-Thompson estimator of the full-data LL.
        match self.maybe_with_outer_subsample_from_options(options) {
            Ok(Some(masked)) => masked.log_likelihood_only(block_states),
            Ok(None) => self.log_likelihood_only(block_states),
            Err(e) => Err(e.into()),
        }
    }

    /// Log-likelihood + flat joint gradient without building the dense Hessian.
    ///
    /// The default trait implementation returns `None`, so the joint-Newton
    /// inner solver falls back to `evaluate()` to obtain the gradient — and
    /// that side-effects a full `Θ(n p²)` `weighted_gram` Hessian build at
    /// every inner iteration. CTN's gradient is structurally
    ///
    ///   `∇ℓ = -X_val^T (w·h) + X_deriv^T (w/h')`,
    ///
    /// which is two `transpose_mul`s through the existing Khatri-Rao operators
    /// and one `Θ(n)` row reduction — `Θ(n p)` total. At biobank scale that is
    /// ~10⁷ FLOPs per call versus ~3·10¹⁰ for the full `evaluate`, so wiring
    /// this override is the gating condition for routing CTN's inner solve
    /// through the matrix-free joint-Newton path without paying the dense H
    /// tax on every gradient refresh.
    fn exact_newton_joint_gradient_evaluation(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
    ) -> Result<Option<ExactNewtonJointGradientEvaluation>, String> {
        if block_states.len() != 1 {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "TransformationNormalFamily expects 1 block, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let beta = &block_states[0].beta;
        let row_quantities = self.row_quantities(beta)?;
        let log_likelihood = row_quantities.log_likelihood;
        let gradient = self.scop_gradient(beta, &row_quantities)?;
        Ok(Some(ExactNewtonJointGradientEvaluation {
            log_likelihood,
            gradient,
        }))
    }

    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        // The Hessian depends on β through 1/h'² where h' = X_deriv · β.
        true
    }

    fn coefficient_hessian_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        drop(specs);
        // Khatri–Rao tensor design: the coefficient block is X = R ⊙ C with
        // rows length p_resp · p_cov. Two regimes:
        //
        // * **Dense regime** (small enough that the unified evaluator builds
        //   `weighted_gram` directly): per-evaluation cost is the dense
        //   `n · (p_resp · p_cov)²` Khatri–Rao gram build.
        //
        // * **Matrix-free regime** (large enough that
        //   `use_joint_matrix_free_path` returns true and the evaluator
        //   factors `H v` through `forward_mul` / `transpose_mul` on the
        //   Khatri–Rao operands): per-`Hv` matvec cost is just
        //   `n · (p_resp + p_cov)` flops — see `ctn_matrix_free_workspace`.
        //   This is only an inner coefficient-space cost estimate; outer
        //   θθ Hessian availability is declared separately.
        let n_usize = self.response_val_basis.nrows();
        let p_resp = self.response_val_basis.ncols() as u64;
        let p_cov = self.covariate_design.ncols() as u64;
        let p_total = p_resp.saturating_mul(p_cov);
        let n = n_usize as u64;
        if crate::custom_family::use_joint_matrix_free_path(p_total as usize, n_usize) {
            n.saturating_mul(p_resp.saturating_add(p_cov))
        } else {
            n.saturating_mul(p_total.saturating_mul(p_total))
        }
    }

    fn coefficient_gradient_cost(&self, specs: &[ParameterBlockSpec]) -> u64 {
        // One row-quantity pass plus two transpose products. The SCOP derivative
        // is structurally positive, so coefficient line searches no longer run a
        // full derivative-grid fraction-to-boundary scan on every attempt.
        self.coefficient_hessian_cost(specs) / 2
    }

    fn outer_derivative_policy(
        &self,
        specs: &[crate::families::custom_family::ParameterBlockSpec],
        psi_dim: usize,
        options: &crate::families::custom_family::BlockwiseFitOptions,
    ) -> crate::families::custom_family::OuterDerivativePolicy {
        // The generic default model in `CustomFamily::outer_derivative_policy`
        // uses `coefficient_hessian_cost × (rho_dim + psi_dim)`, which
        // overstates CTN's actual per-eval Hessian work because the SCOP
        // joint-Hessian path is row-streaming through the Khatri-Rao jet
        // (its `O(n · p)` matrix-free HVP, not `O(n · p²)` dense build).
        // Use a CTN-specific shape:
        //
        // * gradient ≈ `n · (rho_dim + psi_dim) · p_total`
        //   (one directional jet sweep per outer coordinate, row-streamed)
        // * Hessian  ≈ min(dense build, matrix-free HVP loop)
        //   * dense  ≈ `n · (rho_dim + psi_dim) · p_total^2`
        //   * mfree  ≈ `n · (rho_dim + psi_dim) · p_total · rho_dim`
        let capability = self.exact_outer_derivative_order(specs, options);
        let n = specs.first().map_or(0u128, |s| s.design.nrows() as u128);
        let p_total: u128 = specs
            .iter()
            .map(|s| s.design.ncols() as u128)
            .fold(0u128, |acc, x| acc.saturating_add(x));
        let rho_dim: u128 = specs
            .iter()
            .map(|s| s.penalties.len() as u128)
            .fold(0u128, |acc, x| acc.saturating_add(x));
        let k = rho_dim.saturating_add(psi_dim as u128).max(1);
        let p_eff = p_total.max(1);
        // Gradient work: one row sweep per outer coordinate.
        let work_grad = n.saturating_mul(k).saturating_mul(p_eff);
        // Hessian work: pick whichever access shape would dominate. The
        // amortization gate in `should_build_dense` (P2.2) picks the
        // cheaper path at execution time; the policy budget mirrors that
        // by taking the min so that genuinely Hessian-prohibitive
        // problems still downgrade through the budget ceiling.
        let dense_hess = work_grad.saturating_mul(p_eff);
        let mfree_hess = work_grad.saturating_mul(rho_dim.max(1));
        let work_hess = dense_hess.min(mfree_hess);
        crate::families::custom_family::OuterDerivativePolicy {
            capability,
            predicted_hessian_work: work_hess,
            predicted_gradient_work: work_grad,
            // CTN's outer-score reductions are mathematically per-row
            // sums whose contributions are linear in `wᵢ` at every assembly
            // site (gradient, joint Hessian dense / matvec / diagonal, ψ,
            // ψ-ψ, log-likelihood). The `_with_options` overrides install a
            // mask-aware family clone whose `effective_weights()` returns
            // `wᵢ · mᵢ` (HT-weighted), yielding an unbiased estimator
            // `E[score_subsample] = score_full`. The persistent
            // dense-Hessian cache is keyed on the mask hash so subsampled
            // and full-data builds at the same β do not alias.
            subsample_capable: true,
        }
    }

    fn outer_seed_config(&self, n_params: usize) -> crate::seeding::SeedConfig {
        crate::seeding::SeedConfig {
            bounds: (-12.0, 12.0),
            max_seeds: if n_params <= 8 { 1 } else { 2 },
            seed_budget: 1,
            screen_max_inner_iterations: 2,
            risk_profile: crate::seeding::SeedRiskProfile::Gaussian,
            num_auxiliary_trailing: 0,
        }
    }

    fn max_feasible_step_size(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        if block_index != 0 {
            return Ok(None);
        }
        if block_states.len() != 1 {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "TransformationNormalFamily expects 1 block, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        if delta.len() != block_states[0].beta.len() {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "CTN line-search step length {} != beta length {}",
                    delta.len(),
                    block_states[0].beta.len()
                ),
            }
            .into());
        }
        // SCOP encodes monotonicity as
        //   h'(y, x) = epsilon + sum_k M_k(y) * gamma_k(x)^2.
        // With nonnegative M-spline derivative basis rows, every finite beta is
        // interior-feasible. A derivative-grid fraction-to-boundary scan is pure
        // overhead and was the dominant CTN biobank-scale line-search cost.
        Ok(None)
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_index: usize,
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_index != 0 {
            return Ok(None);
        }
        // The CTN tensor design is intentionally factored. Strict monotonicity
        // is encoded structurally as `h' = ε + Σ M_r γ_r²`, so there are no
        // dense active-set constraints to expose here.
        Ok(None)
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_index: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if block_index != 0 {
            return Ok(None);
        }
        let beta = &block_states[0].beta;
        let row_quantities = self.row_quantities(beta)?;
        let dd = self.scop_hessian_directional_derivative(beta, d_beta, &row_quantities)?;
        Ok(Some(dd))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        // Single block: joint Hessian = block Hessian.
        let beta = &block_states[0].beta;
        let row_quantities = self.row_quantities(beta)?;
        let (_, hessian) = self.scop_gradient_and_negative_hessian(beta, &row_quantities)?;
        Ok(Some(hessian))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        self.exact_newton_hessian_directional_derivative(block_states, 0, d_beta_flat)
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let beta = &block_states[0].beta;
        let row_quantities = self.row_quantities(beta)?;
        let d2 = self.scop_hessian_second_directional_derivative(
            beta,
            d_beta_u_flat,
            d_beta_v_flat,
            &row_quantities,
        )?;
        Ok(Some(d2))
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        psi_derivs: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if psi_derivs.is_empty() || psi_index >= psi_derivs[0].len() {
            return Ok(None);
        }
        let psi_first_start = std::time::Instant::now();
        let deriv = &psi_derivs[0][psi_index];
        let beta = &block_states[0].beta;
        let row = self.row_quantities(beta)?;
        let op = deriv
            .implicit_operator
            .as_ref()
            .and_then(|op| op.as_any().downcast_ref::<TensorKroneckerPsiOperator>())
            .ok_or_else(|| {
                "TransformationNormalFamily requires tensor psi derivatives to remain operator-backed"
                    .to_string()
            })?;
        let axis = deriv.implicit_axis;
        let op_arc = Arc::clone(
            deriv
                .implicit_operator
                .as_ref()
                .expect("validated CTN psi derivative operator disappeared"),
        );
        let terms = self.scop_psi_terms(beta, &row, op, op_arc, axis)?;

        log::info!(
            "[STAGE] CTN psi first-order terms axis={} psi_index={} elapsed={:.3}s",
            deriv.implicit_axis,
            psi_index,
            psi_first_start.elapsed().as_secs_f64(),
        );

        Ok(Some(terms))
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        psi_derivs: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if psi_derivs.is_empty() || psi_i >= psi_derivs[0].len() || psi_j >= psi_derivs[0].len() {
            return Ok(None);
        }
        let psi_pair_start = std::time::Instant::now();
        let deriv_i = &psi_derivs[0][psi_i];
        let deriv_j = &psi_derivs[0][psi_j];
        let beta = &block_states[0].beta;
        let row = self.row_quantities(beta)?;
        let p_resp = self.response_val_basis.ncols();
        let p_cov = self.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "SCOP psi-psi terms beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                    beta.len()
                ),
            }
            .into());
        }

        let op = deriv_i
            .implicit_operator
            .as_ref()
            .and_then(|op| op.as_any().downcast_ref::<TensorKroneckerPsiOperator>())
            .ok_or_else(|| {
                "TransformationNormalFamily requires tensor psi derivatives to remain operator-backed"
                    .to_string()
            })?;
        let axis_i = deriv_i.implicit_axis;
        let axis_j = deriv_j.implicit_axis;

        let (objective_psi_psi, score_psi_psi, _) = self
            .scop_psi_psi_value_score_hvp_from_operator(
                beta,
                op,
                axis_i,
                axis_j,
                row.gamma.view(),
                row.h.view(),
                row.h_prime.view(),
                row.endpoint_q.as_slice(),
                None,
            )?;
        let hessian_psi_psi_operator: Box<dyn HyperOperator> =
            Box::new(TransformationNormalPsiPsiHessianOperator::new(
                Arc::new(self.clone()),
                beta.clone(),
                Arc::clone(
                    deriv_i
                        .implicit_operator
                        .as_ref()
                        .expect("validated CTN psi derivative has an implicit operator"),
                ),
                axis_i,
                axis_j,
                Arc::clone(&row.gamma),
                Arc::clone(&row.h),
                Arc::clone(&row.h_prime),
                Arc::clone(&row.endpoint_q),
            ));

        // Result-validation gate. A trial point can still make the SCOP row
        // terms non-finite through an invalid h' or an exploding ψ second
        // derivative in the covariate basis. Surface that as an infeasible
        // exact-Newton evaluation instead of passing NaNs into the unified
        // outer evaluator.
        if !objective_psi_psi.is_finite() || !score_psi_psi.iter().all(|v| v.is_finite()) {
            return Err(TransformationNormalError::NonFinite {
                reason: format!(
                    "TransformationNormalFamily exact ψ-ψ second-order terms produced \
                 non-finite values at psi_i={psi_i}, psi_j={psi_j}: \
                 obj_finite={}, score_all_finite={}. \
                 The outer evaluator should retreat from this trial point.",
                    objective_psi_psi.is_finite(),
                    score_psi_psi.iter().all(|v| v.is_finite()),
                ),
            }
            .into());
        }

        log::info!(
            "[STAGE] CTN psi-psi pair (psi_i={}, psi_j={}, axes={},{}) elapsed={:.3}s",
            psi_i,
            psi_j,
            deriv_i.implicit_axis,
            deriv_j.implicit_axis,
            psi_pair_start.elapsed().as_secs_f64(),
        );

        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(hessian_psi_psi_operator),
        }))
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        _: &[ParameterBlockSpec],
        psi_derivs: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if psi_derivs.is_empty() || psi_index >= psi_derivs[0].len() {
            return Ok(None);
        }
        let deriv = &psi_derivs[0][psi_index];
        let beta = &block_states[0].beta;
        let op = deriv
            .implicit_operator
            .as_ref()
            .and_then(|op| op.as_any().downcast_ref::<TensorKroneckerPsiOperator>())
            .ok_or_else(|| {
                "TransformationNormalFamily requires tensor psi derivatives to remain operator-backed"
                    .to_string()
            })?;
        let axis = deriv.implicit_axis;
        let row = self.row_quantities(beta)?;
        let hess =
            self.scop_psi_hessian_directional_derivative(beta, d_beta_flat, &row, op, axis)?;
        Ok(Some(hess))
    }

    fn exact_newton_joint_hessian_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        drop(specs);
        if block_states.len() != 1 {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "TransformationNormalFamily expects 1 block, got {}",
                    block_states.len()
                ),
            }
            .into());
        }
        let beta = &block_states[0].beta;
        let row_quantities = self.row_quantities(beta)?;
        // Expected HVP reuse this workspace will service before its
        // `(β, row_quantities)` key advances. The outer-eval trace path
        // performs ~`2·rho_dim` HVPs plus one diagonal call against the
        let workspace = TransformationNormalJointHessianWorkspace::new(
            Arc::new(self.clone()),
            beta.clone(),
            row_quantities.clone(),
        )?;
        Ok(Some(
            Arc::new(workspace) as Arc<dyn ExactNewtonJointHessianWorkspace>
        ))
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        drop(specs);
        Ok(Some(Arc::new(TransformationNormalPsiWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            derivative_blocks.to_vec(),
        ))))
    }

    fn exact_newton_joint_hessian_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointHessianWorkspace>>, String> {
        // Route through a mask-aware family clone when an outer-score
        // subsample is active. The cloned family's `effective_weights()`
        // returns `wᵢ · mᵢ` (`mᵢ = 1/πᵢ` on sampled rows, `0` elsewhere),
        // and every CTN assembly site reads weights through that accessor.
        // Each per-row contribution is linear in `wᵢ`, so the workspace's
        // gradient / dense Hessian / matrix-free HVP / diagonal are exact
        // Horvitz-Thompson estimators of the full-data quantities.
        match self.maybe_with_outer_subsample_from_options(options)? {
            Some(masked) => masked.exact_newton_joint_hessian_workspace(block_states, specs),
            None => self.exact_newton_joint_hessian_workspace(block_states, specs),
        }
    }

    fn exact_newton_joint_psi_workspace_with_options(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        options: &BlockwiseFitOptions,
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        drop(specs);
        // Route through a mask-aware family clone when an outer-score
        // subsample is active. Every CTN ψ assembly site — including the
        // workspace's `compute_all_axes` (per-row reduction near line ~13916)
        // and `compute_pair_cache` (per-row reduction near line ~14263) —
        // reads its row weight via `self.family.effective_weights()`, which
        // on the cloned family returns `wᵢ · mᵢ`. Because each per-row
        // contribution is linear in `wᵢ`, the workspace's per-axis ψ and
        // per-axis-pair ψ-ψ outputs are exact Horvitz-Thompson estimators
        // of the full-data quantities. The persistent dense-Hessian cache
        // and `row_quantity_cache` on the cloned family are fresh, so
        // subsampled builds cannot alias a later full-data probe at the
        // same β.
        let family = match self.maybe_with_outer_subsample_from_options(options)? {
            Some(masked) => masked,
            None => self.clone(),
        };
        Ok(Some(Arc::new(TransformationNormalPsiWorkspace::new(
            family,
            block_states.to_vec(),
            derivative_blocks.to_vec(),
        ))))
    }

    fn exact_newton_joint_psi_workspace_for_first_order_terms(&self) -> bool {
        // CTN's per-axis [`scop_psi_terms`] kernel walks all `n` rows serially
        // and is invoked once per ψ axis. Opting in here amortizes the per-row
        // state load across axes and parallelizes the row walk via the
        // workspace's [`compute_all_axes`] kernel — the dominant outer
        // gradient-evaluation cost at biobank scale.
        true
    }

    fn inner_coefficient_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        drop(specs);
        // CTN's SCOP coefficient-space joint Hessian is supplied as a
        // row-streaming matrix-free Hv operator.
        true
    }

    fn outer_hyper_hessian_hvp_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        drop(specs);
        true
    }

    fn outer_hyper_hessian_dense_available(&self, specs: &[ParameterBlockSpec]) -> bool {
        drop(specs);
        // Dense materialization remains mathematically available through the
        // outer-HVP operator, but SCOP's primary production path is the
        // matrix-free θθ operator above.
        true
    }
}

// ---------------------------------------------------------------------------
// Matrix-free joint Hessian workspace (Khatri-Rao operator-only)
// ---------------------------------------------------------------------------

/// Per-evaluation workspace for the SCOP-CTN joint Hessian.
///
/// The old linear-`h` CTN Hessian had the form
/// `X_val' W X_val + X_deriv' diag(w / h'^2) X_deriv`. SCOP is nonlinear in
/// the shape rows, so `H v` must be evaluated through the rowwise chain rule.
/// This workspace keeps the accepted `β` and row quantities and applies
/// `H`, `D H[u]`, and `D²H[u,v]` without materializing a dense p×p matrix.
struct TransformationNormalJointHessianWorkspace {
    /// Shared family handle. Cloning the workspace's family for each downstream
    /// matrix-free operator (dH, d²H per psi coord and per pair) would copy
    /// the full row-space Kronecker designs (~hundreds of MiB at biobank
    /// scale) per call. Arc-sharing makes operator construction O(1).
    family: Arc<TransformationNormalFamily>,
    beta: Array1<f64>,
    row_quantities: TransformationNormalRowQuantityCache,
    dense_hessian_cache: OnceLock<Array2<f64>>,
}

impl TransformationNormalJointHessianWorkspace {
    fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        row_quantities: TransformationNormalRowQuantityCache,
    ) -> Result<Self, String> {
        Ok(Self {
            family,
            beta,
            row_quantities,
            dense_hessian_cache: OnceLock::new(),
        })
    }

    fn p_total(&self) -> usize {
        self.family.x_val_kron.ncols()
    }

    fn dense_hessian_cache_enabled(&self) -> bool {
        let p_total = self.p_total();
        if p_total > SCOP_HESSIAN_HVP_DENSE_CACHE_MAX_DIM {
            return false;
        }
        p_total
            .checked_mul(p_total)
            .and_then(|entries| entries.checked_mul(std::mem::size_of::<f64>()))
            .is_some_and(|bytes| bytes <= SCOP_HESSIAN_HVP_DENSE_CACHE_MAX_BYTES)
    }

    fn dense_hessian(&self) -> Result<&Array2<f64>, String> {
        if let Some(hessian) = self.dense_hessian_cache.get() {
            return Ok(hessian);
        }
        let dense_start = std::time::Instant::now();
        let (_, hessian) = self
            .family
            .scop_gradient_and_negative_hessian(&self.beta, &self.row_quantities)?;
        if hessian.nrows() != self.p_total() || hessian.ncols() != self.p_total() {
            return Err(format!(
                "CTN dense Hessian cache shape mismatch: got {}x{}, expected {}x{}",
                hessian.nrows(),
                hessian.ncols(),
                self.p_total(),
                self.p_total()
            ));
        }
        if hessian.iter().any(|value| !value.is_finite()) {
            return Err("CTN dense Hessian cache produced non-finite values".to_string());
        }
        drop(self.dense_hessian_cache.set(hessian));
        log::info!(
            "[STAGE] CTN dense Hessian cache build p={} elapsed={:.3}s",
            self.p_total(),
            dense_start.elapsed().as_secs_f64(),
        );
        self.dense_hessian_cache
            .get()
            .ok_or_else(|| "CTN dense Hessian cache was not initialized".to_string())
    }

    fn apply_hessian(&self, v: &Array1<f64>) -> Result<Array1<f64>, String> {
        if v.len() != self.p_total() {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "CTN joint Hessian matvec: input length {} != p_total {}",
                    v.len(),
                    self.p_total()
                ),
            }
            .into());
        }
        let mut out = Array1::<f64>::zeros(self.p_total());
        self.apply_hessian_into(v, &mut out)?;
        Ok(out)
    }

    fn apply_hessian_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<(), String> {
        if v.len() != self.p_total() || out.len() != self.p_total() {
            return Err(format!(
                "CTN joint Hessian matvec_into dimension mismatch: v={} out={} p_total={}",
                v.len(),
                out.len(),
                self.p_total()
            ));
        }
        if self.dense_hessian_cache_enabled() {
            let hessian = self.dense_hessian()?;
            crate::faer_ndarray::fast_av_view_into(hessian, v, out.view_mut());
            return Ok(());
        }
        self.family
            .scop_hessian_matvec_into(&self.beta, &self.row_quantities, v, out)
    }

    /// Exact diagonal of the unpenalized joint Hessian.
    fn compute_diagonal(&self) -> Result<Array1<f64>, String> {
        if self.dense_hessian_cache_enabled() {
            return Ok(self.dense_hessian()?.diag().to_owned());
        }
        self.family
            .scop_hessian_diagonal(&self.beta, &self.row_quantities)
    }
}

impl ExactNewtonJointHessianWorkspace for TransformationNormalJointHessianWorkspace {
    fn hessian_dense(&self) -> Result<Option<Array2<f64>>, String> {
        Ok(Some(self.dense_hessian()?.clone()))
    }

    fn hessian_matvec(&self, v: &Array1<f64>) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.apply_hessian(v)?))
    }

    fn hessian_matvec_into(&self, v: &Array1<f64>, out: &mut Array1<f64>) -> Result<bool, String> {
        self.apply_hessian_into(v, out)?;
        Ok(true)
    }

    fn hessian_diagonal(&self) -> Result<Option<Array1<f64>>, String> {
        Ok(Some(self.compute_diagonal()?))
    }

    fn directional_derivative(&self, _: &Array1<f64>) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    fn directional_derivative_operator(
        &self,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let p_total = self.p_total();
        if d_beta_flat.len() != p_total {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "CTN directional_derivative_operator length mismatch: got {}, expected {}",
                    d_beta_flat.len(),
                    p_total
                ),
            }
            .into());
        }
        let op = TransformationNormalDhMatrixFreeOperator::new(
            Arc::clone(&self.family),
            self.beta.clone(),
            self.row_quantities.clone(),
            d_beta_flat.clone(),
        );
        Ok(Some(Arc::new(op) as Arc<dyn HyperOperator>))
    }

    fn second_directional_derivative(
        &self,
        _: &Array1<f64>,
        _: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        Ok(None)
    }

    fn second_directional_derivative_operator(
        &self,
        d_beta_u: &Array1<f64>,
        d_beta_v: &Array1<f64>,
    ) -> Result<Option<Arc<dyn HyperOperator>>, String> {
        let p_total = self.p_total();
        if d_beta_u.len() != p_total || d_beta_v.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "CTN second_directional_derivative_operator length mismatch: u={}, v={}, expected {}",
                d_beta_u.len(),
                d_beta_v.len(),
                p_total
            ) }.into());
        }
        let op = TransformationNormalD2hMatrixFreeOperator::new(
            Arc::clone(&self.family),
            self.beta.clone(),
            self.row_quantities.clone(),
            d_beta_u.clone(),
            d_beta_v.clone(),
        );
        Ok(Some(Arc::new(op) as Arc<dyn HyperOperator>))
    }
}

/// Matrix-free directional derivative of the CTN joint Hessian.
///
/// SCOP makes the derivative row-dependent through `γ_k(x)`, so this operator
/// evaluates `D H[direction] · v` by streaming rows through the exact chain
/// rule instead of using the old scalar-weighted `X_deriv' diag(.) X_deriv`
/// identity.
struct TransformationNormalDhMatrixFreeOperator {
    family: Arc<TransformationNormalFamily>,
    beta: Array1<f64>,
    row_quantities: TransformationNormalRowQuantityCache,
    direction: Array1<f64>,
}

impl TransformationNormalDhMatrixFreeOperator {
    fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        row_quantities: TransformationNormalRowQuantityCache,
        direction: Array1<f64>,
    ) -> Self {
        Self {
            family,
            beta,
            row_quantities,
            direction,
        }
    }

    fn p_total(&self) -> usize {
        self.family.x_deriv_kron.ncols()
    }

    fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        self.family
            .scop_hessian_directional_matvec(&self.beta, &self.direction, &self.row_quantities, v)
            .expect("validated CTN dH operator inputs should not fail")
    }

    fn projected_gram_cache_id(&self) -> usize {
        let family_ptr = Arc::as_ptr(&self.family) as usize;
        let design_dims = self.family.covariate_design.nrows()
            ^ self.family.covariate_design.ncols().rotate_left(11);
        family_ptr ^ design_dims.rotate_left(23)
    }
}

impl HyperOperator for TransformationNormalDhMatrixFreeOperator {
    fn dim(&self) -> usize {
        self.p_total()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(v.len(), self.p_total());
        self.apply(v)
    }

    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        debug_assert_eq!(factor.nrows(), self.p_total());
        self.family
            .scop_hessian_directional_matmat(
                &self.beta,
                &self.direction,
                &self.row_quantities,
                factor,
            )
            .expect("validated CTN dH batched operator inputs should not fail")
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        debug_assert_eq!(factor.nrows(), self.p_total());
        let row_grams = self
            .family
            .scop_projected_response_gram_table(factor.view())
            .expect("validated CTN dH projected Gram inputs should not fail");
        self.family
            .scop_hessian_directional_trace_from_response_grams(
                &self.beta,
                &self.direction,
                &self.row_quantities,
                row_grams.view(),
            )
            .expect("validated CTN dH projected trace inputs should not fail")
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        debug_assert_eq!(factor.nrows(), self.p_total());
        let key =
            ProjectedFactorKey::from_factor_view(self.projected_gram_cache_id(), factor.view());
        let row_grams = cache.get_or_insert_with(key, || {
            self.family
                .scop_projected_response_gram_table(factor.view())
                .expect("validated CTN dH cached projected Gram inputs should not fail")
        });
        self.family
            .scop_hessian_directional_trace_from_response_grams(
                &self.beta,
                &self.direction,
                &self.row_quantities,
                row_grams.view(),
            )
            .expect("validated CTN dH cached projected trace inputs should not fail")
    }

    fn to_dense(&self) -> Array2<f64> {
        self.family
            .scop_hessian_directional_derivative(&self.beta, &self.direction, &self.row_quantities)
            .expect("validated CTN dH operator inputs should not fail")
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

/// Matrix-free second directional derivative of the CTN joint Hessian.
///
/// This is the SCOP rowwise chain-rule operator for `D²H[u, v] · w`; it keeps
/// the memory profile of matrix-free REML while matching the dense exact
/// second derivative.
struct TransformationNormalD2hMatrixFreeOperator {
    family: Arc<TransformationNormalFamily>,
    beta: Array1<f64>,
    row_quantities: TransformationNormalRowQuantityCache,
    direction_u: Array1<f64>,
    direction_v: Array1<f64>,
}

impl TransformationNormalD2hMatrixFreeOperator {
    fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        row_quantities: TransformationNormalRowQuantityCache,
        direction_u: Array1<f64>,
        direction_v: Array1<f64>,
    ) -> Self {
        Self {
            family,
            beta,
            row_quantities,
            direction_u,
            direction_v,
        }
    }

    fn p_total(&self) -> usize {
        self.family.x_deriv_kron.ncols()
    }

    fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        self.family
            .scop_hessian_second_directional_matvec(
                &self.beta,
                &self.direction_u,
                &self.direction_v,
                &self.row_quantities,
                v,
            )
            .expect("validated CTN d2H operator inputs should not fail")
    }
}

impl HyperOperator for TransformationNormalD2hMatrixFreeOperator {
    fn dim(&self) -> usize {
        self.p_total()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(v.len(), self.p_total());
        self.apply(v)
    }

    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        debug_assert_eq!(factor.nrows(), self.p_total());
        self.family
            .scop_hessian_second_directional_matmat(
                &self.beta,
                &self.direction_u,
                &self.direction_v,
                &self.row_quantities,
                factor,
            )
            .expect("validated CTN d2H batched operator inputs should not fail")
    }

    fn to_dense(&self) -> Array2<f64> {
        self.family
            .scop_hessian_second_directional_derivative(
                &self.beta,
                &self.direction_u,
                &self.direction_v,
                &self.row_quantities,
            )
            .expect("validated CTN d2H operator inputs should not fail")
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

struct TransformationNormalPsiHessianOperator {
    family: Arc<TransformationNormalFamily>,
    beta: Array1<f64>,
    op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    axis: usize,
    trace_axes: Arc<Vec<usize>>,
    trace_axis_pos: usize,
    trace_cache_id: usize,
    row_quantities: TransformationNormalRowQuantityCache,
}

impl TransformationNormalPsiHessianOperator {
    fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        axis: usize,
        row_gamma: Arc<Array2<f64>>,
        row_h: Arc<Array1<f64>>,
        row_h_prime: Arc<Array1<f64>>,
        endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    ) -> Self {
        Self::new_with_trace_axes(
            family,
            beta,
            op,
            axis,
            Arc::new(vec![axis]),
            0,
            row_gamma,
            row_h,
            row_h_prime,
            endpoint_q,
        )
    }

    fn new_with_trace_axes(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        axis: usize,
        trace_axes: Arc<Vec<usize>>,
        trace_axis_pos: usize,
        row_gamma: Arc<Array2<f64>>,
        row_h: Arc<Array1<f64>>,
        row_h_prime: Arc<Array1<f64>>,
        endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    ) -> Self {
        let log_likelihood = 0.0;
        let op_ptr = Arc::as_ptr(&op) as *const () as usize;
        let row_ptr = Arc::as_ptr(&row_gamma) as usize;
        let axes_ptr = Arc::as_ptr(&trace_axes) as usize;
        let trace_cache_id = op_ptr ^ row_ptr.rotate_left(17) ^ axes_ptr.rotate_left(31);
        Self {
            family,
            beta: beta.clone(),
            op,
            axis,
            trace_axes,
            trace_axis_pos,
            trace_cache_id,
            row_quantities: TransformationNormalRowQuantityCache {
                beta: Arc::new(beta),
                gamma: row_gamma,
                h_lower: Arc::new(Array1::zeros(row_h.len())),
                h_upper: Arc::new(Array1::zeros(row_h.len())),
                h: row_h,
                h_prime: row_h_prime,
                endpoint_q,
                log_likelihood,
                // Synthetic row-quantity instance built from materialized
                // pieces (psi/psi second-order test path); not connected to
                // the family's version counter. Version 0 is reserved as
                // "never installed" — the persistent dense-Hessian slot
                // built from this entry will key on version=0 and is
                // therefore distinct from any production build.
                version: 0,
            },
        }
    }

    fn tensor_op(&self) -> &TensorKroneckerPsiOperator {
        self.op
            .as_any()
            .downcast_ref::<TensorKroneckerPsiOperator>()
            .expect("validated CTN psi operator must remain tensor-backed")
    }

    fn apply_columns_with_shared_cov(
        &self,
        factor: &Array2<f64>,
        cov: &Array2<f64>,
        cov_psi: &Array2<f64>,
    ) -> Array2<f64> {
        self.family
            .scop_psi_hessian_hvp_mat_from_cov(
                &self.beta,
                &self.row_quantities,
                self.axis,
                cov,
                cov_psi,
                factor.view(),
            )
            .expect("validated CTN psi Hessian operator batched input should not fail")
    }

    fn projected_trace_table(&self, factor: &Array2<f64>) -> Array2<f64> {
        debug_assert_eq!(factor.nrows(), self.dim());
        let axes = self.trace_axes.as_slice();
        if axes.is_empty() {
            return Array2::<f64>::zeros((0, 1));
        }
        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let policy = ResourcePolicy::default_library();
        let rows_per_chunk = crate::resource::rows_for_target_bytes(
            policy.row_chunk_target_bytes,
            p_cov.saturating_mul(axes.len() + 1).max(1),
        )
        .max(1)
        .min(n.max(1));

        let op = self.tensor_op();
        let mut traces = vec![0.0_f64; axes.len()];
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let cov = self
                .family
                .covariate_design
                .try_row_chunk(rows.clone())
                .expect(
                    "validated CTN psi Hessian projected trace covariate chunk should not fail",
                );
            let mut cov_psi_chunks: Vec<Array2<f64>> = Vec::with_capacity(axes.len());
            for &axis in axes {
                cov_psi_chunks.push(
                    op.cov_first_axis_row_chunk_streaming(axis, rows.clone())
                        .expect("validated CTN psi Hessian projected trace first-axis chunk should not fail"),
                );
            }
            let cov_psi_views: Vec<ArrayView2<'_, f64>> =
                cov_psi_chunks.iter().map(|chunk| chunk.view()).collect();
            let chunk_traces = self
                .family
                .scop_psi_hessian_trace_factor_all_axes_chunk_from_cov(
                    &self.beta,
                    &self.row_quantities,
                    start,
                    cov.view(),
                    &cov_psi_views,
                    factor.view(),
                )
                .expect(
                    "validated CTN psi Hessian all-axis projected trace inputs should not fail",
                );
            debug_assert_eq!(chunk_traces.len(), traces.len());
            for (total, value) in traces.iter_mut().zip(chunk_traces.into_iter()) {
                *total += value;
            }
        }
        Array2::from_shape_vec((traces.len(), 1), traces)
            .expect("validated CTN psi Hessian projected trace table shape")
    }
}

impl HyperOperator for TransformationNormalPsiHessianOperator {
    fn dim(&self) -> usize {
        self.beta.len()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        self.family
            .scop_psi_hessian_apply_from_operator(
                &self.beta,
                &self.row_quantities,
                self.tensor_op(),
                self.axis,
                v,
            )
            .expect("validated CTN psi Hessian operator inputs should not fail")
    }

    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        debug_assert_eq!(factor.nrows(), self.dim());
        let p = factor.nrows();
        let k = factor.ncols();
        let cov = self
            .family
            .covariate_dense_arc()
            .expect("validated CTN psi Hessian operator covariate cache should not fail");
        let cov_psi = self
            .tensor_op()
            .materialize_cov_first_axis(self.axis)
            .expect("validated CTN psi Hessian operator covariate derivative should not fail");
        let out = self.apply_columns_with_shared_cov(factor, cov.as_ref(), &cov_psi);
        debug_assert_eq!(out.nrows(), p);
        debug_assert_eq!(out.ncols(), k);
        out
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        debug_assert_eq!(factor.nrows(), self.dim());
        let cov = self
            .family
            .covariate_dense_arc()
            .expect("validated CTN psi Hessian projected trace covariate cache should not fail");
        let cov_psi = self
            .tensor_op()
            .materialize_cov_first_axis(self.axis)
            .expect(
                "validated CTN psi Hessian projected trace covariate derivative should not fail",
            );
        self.family
            .scop_psi_hessian_trace_factor_from_cov(
                &self.beta,
                &self.row_quantities,
                self.axis,
                cov.as_ref(),
                &cov_psi,
                factor.view(),
            )
            .expect("validated CTN psi Hessian projected trace inputs should not fail")
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        let key = ProjectedFactorKey::from_factor_view(self.trace_cache_id, factor.view());
        let table = cache.get_or_insert_with(key, || self.projected_trace_table(factor));
        table[[self.trace_axis_pos, 0]]
    }

    fn to_dense(&self) -> Array2<f64> {
        let p = self.dim();
        let dense = self.mul_mat(&Array2::<f64>::eye(p));
        0.5 * (&dense + &dense.t())
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

struct TransformationNormalPsiDhMatrixFreeOperator {
    family: Arc<TransformationNormalFamily>,
    beta: Array1<f64>,
    direction: Array1<f64>,
    op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    axis: usize,
    row_quantities: TransformationNormalRowQuantityCache,
    // `RayonSafeOnce` (not `OnceLock`): `dense_matrix()` materializes via
    // `scop_psi_hessian_directional_derivative`, which dispatches an
    // `into_par_iter` over rows. This operator is invoked from outer
    // par_iter contexts (HyperOperator HVP / dense build paths); a plain
    // `OnceLock::get_or_init` here would park sibling rayon workers on the
    // OS mutex while the leader tries to schedule child rayon tasks no
    // worker is free to pick up — classic OnceLock-under-rayon deadlock
    // (see `feedback_oncelock_rayon_deadlock`). `RayonSafeOnce` keeps the
    // init lock-free; racers redundantly compute but `set()` discards the
    // losers.
    dense_cache: crate::resource::RayonSafeOnce<Array2<f64>>,
}

impl TransformationNormalPsiDhMatrixFreeOperator {
    fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        direction: Array1<f64>,
        op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        axis: usize,
        row_quantities: TransformationNormalRowQuantityCache,
    ) -> Self {
        Self {
            family,
            beta,
            direction,
            op,
            axis,
            row_quantities,
            dense_cache: crate::resource::RayonSafeOnce::new(),
        }
    }

    fn p_total(&self) -> usize {
        self.beta.len()
    }

    fn tensor_op(&self) -> &TensorKroneckerPsiOperator {
        self.op
            .as_any()
            .downcast_ref::<TensorKroneckerPsiOperator>()
            .expect("validated CTN psi dH operator must remain tensor-backed")
    }

    fn dense_matrix(&self) -> &Array2<f64> {
        self.dense_cache.get_or_init(|| {
            self.family
                .scop_psi_hessian_directional_derivative(
                    &self.beta,
                    &self.direction,
                    &self.row_quantities,
                    self.tensor_op(),
                    self.axis,
                )
                .expect("validated CTN psi dH dense materialization inputs should not fail")
        })
    }

    fn trace_projected_factor_dense(&self, factor: &Array2<f64>) -> f64 {
        let dense_factor = crate::faer_ndarray::fast_ab(self.dense_matrix(), factor);
        factor
            .iter()
            .zip(dense_factor.iter())
            .map(|(&f, &bf)| f * bf)
            .sum()
    }

    fn projected_factor_cache_id(&self) -> usize {
        let family_ptr = Arc::as_ptr(&self.family) as usize;
        family_ptr
            ^ self.axis.wrapping_add(0x9e37_79b9).rotate_left(17)
            ^ self.family.response_val_basis.nrows().rotate_left(7)
            ^ self.family.covariate_design.ncols().rotate_left(29)
    }

    fn projected_factor_table_bytes(&self, factor: &Array2<f64>) -> usize {
        let n = self.family.response_val_basis.nrows();
        let p_resp = self.family.response_val_basis.ncols();
        let rank = factor.ncols();
        n.saturating_mul(p_resp)
            .saturating_mul(rank)
            .saturating_mul(2)
            .saturating_mul(std::mem::size_of::<f64>())
    }

    fn projected_factor_table_fits_budget(&self, factor: &Array2<f64>) -> bool {
        let bytes = self.projected_factor_table_bytes(factor);
        let policy = ResourcePolicy::default_library();
        bytes <= policy.max_single_materialization_bytes && bytes <= policy.max_operator_cache_bytes
    }

    fn projected_factor_table(&self, factor: &Array2<f64>) -> Array2<f64> {
        debug_assert_eq!(factor.nrows(), self.p_total());
        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let p_resp = self.family.response_val_basis.ncols();
        let rank = factor.ncols();
        let projected_len = p_resp * rank;
        let mut table = Array2::<f64>::zeros((n, 2 * projected_len));
        if n == 0 || rank == 0 {
            return table;
        }
        let op = self.tensor_op();
        let policy = ResourcePolicy::default_library();
        let live_cols = p_cov
            .saturating_mul(2)
            .saturating_add(p_resp.saturating_mul(rank).saturating_mul(2))
            .max(1);
        let rows_per_chunk =
            crate::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, live_cols)
                .max(1)
                .min(n.max(1));
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let cov = self
                .family
                .covariate_design
                .try_row_chunk(rows.clone())
                .expect("validated CTN psi dH projected-table covariate chunk should not fail");
            let cov_psi = op
                .cov_first_axis_row_chunk_streaming(self.axis, rows.clone())
                .expect("validated CTN psi dH projected-table covariate derivative chunk should not fail");
            for k in 0..p_resp {
                let factor_block = factor.slice(s![k * p_cov..(k + 1) * p_cov, ..]);
                let cov_projection = fast_ab(&cov, &factor_block);
                let psi_projection = fast_ab(&cov_psi, &factor_block);
                table
                    .slice_mut(s![start..end, k * rank..(k + 1) * rank])
                    .assign(&cov_projection);
                table
                    .slice_mut(s![
                        start..end,
                        projected_len + k * rank..projected_len + (k + 1) * rank
                    ])
                    .assign(&psi_projection);
            }
        }
        table
    }

    fn trace_projected_factor_with_projected_table(
        &self,
        factor: &Array2<f64>,
        table: ArrayView2<'_, f64>,
    ) -> f64 {
        debug_assert_eq!(factor.nrows(), self.p_total());
        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let p_resp = self.family.response_val_basis.ncols();
        let rank = factor.ncols();
        let projected_len = p_resp * rank;
        debug_assert_eq!(table.dim(), (n, 2 * projected_len));
        let op = self.tensor_op();
        let policy = ResourcePolicy::default_library();
        let live_cols = p_cov.saturating_mul(2).max(1);
        let rows_per_chunk =
            crate::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, live_cols)
                .max(1)
                .min(n.max(1));
        let mut total = 0.0;
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let cov = self
                .family
                .covariate_design
                .try_row_chunk(rows.clone())
                .expect(
                    "validated CTN psi dH cached projected trace covariate chunk should not fail",
                );
            let cov_psi = op
                .cov_first_axis_row_chunk_streaming(self.axis, rows.clone())
                .expect("validated CTN psi dH cached projected trace covariate derivative chunk should not fail");
            let projected_cov = table.slice(s![start..end, ..projected_len]);
            let projected_psi = table.slice(s![start..end, projected_len..]);
            total += self
                .family
                .scop_psi_hessian_directional_trace_factor_chunk_from_cov(
                    &self.beta,
                    &self.direction,
                    &self.row_quantities,
                    start,
                    cov.view(),
                    cov_psi.view(),
                    factor.view(),
                    Some(projected_cov),
                    Some(projected_psi),
                )
                .expect("validated CTN psi dH cached projected trace inputs should not fail");
        }
        total
    }

    fn trace_projected_factor_streaming(&self, factor: &Array2<f64>) -> f64 {
        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let rank = factor.ncols();
        let p_resp = self.family.response_val_basis.ncols();
        let policy = ResourcePolicy::default_library();
        let live_cols = p_cov
            .saturating_mul(2)
            .saturating_add(p_resp.saturating_mul(rank).saturating_mul(2))
            .max(1);
        let rows_per_chunk =
            crate::resource::rows_for_target_bytes(policy.row_chunk_target_bytes, live_cols)
                .max(1)
                .min(n.max(1));
        let op = self.tensor_op();
        let mut total = 0.0;
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let cov = self
                .family
                .covariate_design
                .try_row_chunk(rows.clone())
                .expect("validated CTN psi dH projected trace covariate chunk should not fail");
            let cov_psi = op
                .cov_first_axis_row_chunk_streaming(self.axis, rows.clone())
                .expect("validated CTN psi dH projected trace covariate derivative chunk should not fail");
            total += self
                .family
                .scop_psi_hessian_directional_trace_factor_chunk_from_cov(
                    &self.beta,
                    &self.direction,
                    &self.row_quantities,
                    start,
                    cov.view(),
                    cov_psi.view(),
                    factor.view(),
                    None,
                    None,
                )
                .expect("validated CTN psi dH projected trace inputs should not fail");
        }
        total
    }
}

impl HyperOperator for TransformationNormalPsiDhMatrixFreeOperator {
    fn dim(&self) -> usize {
        self.p_total()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(v.len(), self.p_total());
        self.dense_matrix().dot(v)
    }

    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        debug_assert_eq!(factor.nrows(), self.p_total());
        self.dense_matrix().dot(factor)
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        debug_assert_eq!(factor.nrows(), self.p_total());

        // At the CTN biobank benchmark shape (`p≈264`, `n≈20k`), the
        // coefficient-space directional Hessian is tiny (< 1 MiB) while the
        // streaming projected trace repeats a full row-kernel pass for every
        // outer-gradient coordinate and every BFGS line-search evaluation.
        // Materializing the exact p×p directional derivative once and doing a
        // dense BLAS3 projection is mathematically identical to the streaming
        // contraction and is several times faster at these moderate p values.
        // Keep the old streaming path for larger coefficient systems where a
        // dense p×p cache can dominate memory or construction cost.
        if self.p_total() <= 512 {
            return self.trace_projected_factor_dense(factor);
        }

        self.trace_projected_factor_streaming(factor)
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &crate::solver::estimate::reml::unified::ProjectedFactorCache,
    ) -> f64 {
        debug_assert_eq!(factor.nrows(), self.p_total());
        if self.p_total() <= 512 || !self.projected_factor_table_fits_budget(factor) {
            return self.trace_projected_factor(factor);
        }
        let key =
            ProjectedFactorKey::from_factor_view(self.projected_factor_cache_id(), factor.view());
        let table = cache.get_or_insert_with(key, || self.projected_factor_table(factor));
        self.trace_projected_factor_with_projected_table(factor, table.view())
    }

    fn to_dense(&self) -> Array2<f64> {
        self.dense_matrix().clone()
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

struct TransformationNormalPsiPsiHessianOperator {
    family: Arc<TransformationNormalFamily>,
    beta: Array1<f64>,
    op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    axis_i: usize,
    axis_j: usize,
    trace_axes: Arc<Vec<usize>>,
    trace_axis_i_pos: usize,
    trace_axis_j_pos: usize,
    trace_cache_id: usize,
    row_gamma: Arc<Array2<f64>>,
    row_h: Arc<Array1<f64>>,
    row_h_prime: Arc<Array1<f64>>,
    endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
}

impl TransformationNormalPsiPsiHessianOperator {
    fn new(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        axis_i: usize,
        axis_j: usize,
        row_gamma: Arc<Array2<f64>>,
        row_h: Arc<Array1<f64>>,
        row_h_prime: Arc<Array1<f64>>,
        endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    ) -> Self {
        let trace_axes = if axis_i == axis_j {
            Arc::new(vec![axis_i])
        } else {
            Arc::new(vec![axis_i, axis_j])
        };
        let trace_axis_i_pos = 0;
        let trace_axis_j_pos = if axis_i == axis_j { 0 } else { 1 };
        Self::new_with_trace_axes(
            family,
            beta,
            op,
            axis_i,
            axis_j,
            trace_axes,
            trace_axis_i_pos,
            trace_axis_j_pos,
            row_gamma,
            row_h,
            row_h_prime,
            endpoint_q,
        )
    }

    fn new_with_trace_axes(
        family: Arc<TransformationNormalFamily>,
        beta: Array1<f64>,
        op: Arc<dyn CustomFamilyPsiDerivativeOperator>,
        axis_i: usize,
        axis_j: usize,
        trace_axes: Arc<Vec<usize>>,
        trace_axis_i_pos: usize,
        trace_axis_j_pos: usize,
        row_gamma: Arc<Array2<f64>>,
        row_h: Arc<Array1<f64>>,
        row_h_prime: Arc<Array1<f64>>,
        endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    ) -> Self {
        let op_ptr = Arc::as_ptr(&op) as *const () as usize;
        let row_ptr = Arc::as_ptr(&row_gamma) as usize;
        let axes_ptr = Arc::as_ptr(&trace_axes) as usize;
        let trace_cache_id = op_ptr ^ row_ptr.rotate_left(17) ^ axes_ptr.rotate_left(31);
        Self {
            family,
            beta,
            op,
            axis_i,
            axis_j,
            trace_axes,
            trace_axis_i_pos,
            trace_axis_j_pos,
            trace_cache_id,
            row_gamma,
            row_h,
            row_h_prime,
            endpoint_q,
        }
    }

    fn p_total(&self) -> usize {
        self.beta.len()
    }

    fn tensor_op(&self) -> &TensorKroneckerPsiOperator {
        self.op
            .as_any()
            .downcast_ref::<TensorKroneckerPsiOperator>()
            .expect("validated CTN psi-psi operator must remain tensor-backed")
    }

    fn apply(&self, v: &Array1<f64>) -> Array1<f64> {
        self.family
            .scop_psi_psi_value_score_hvp_from_operator(
                &self.beta,
                self.tensor_op(),
                self.axis_i,
                self.axis_j,
                self.row_gamma.view(),
                self.row_h.view(),
                self.row_h_prime.view(),
                self.endpoint_q.as_slice(),
                Some(v),
            )
            .expect("validated CTN psi-psi operator inputs should not fail")
            .2
            .expect("CTN psi-psi operator called without HVP output")
    }

    fn apply_columns_with_shared_cov(
        &self,
        factor: &Array2<f64>,
        cov: &Array2<f64>,
        cov_i: &Array2<f64>,
        cov_j: &Array2<f64>,
        cov_ij: &Array2<f64>,
        row_start: usize,
        row_end: usize,
    ) -> Array2<f64> {
        debug_assert_eq!(factor.nrows(), self.p_total());
        let p = factor.nrows();
        let k = factor.ncols();
        let mut out = Array2::<f64>::zeros((p, k));
        let tile_cols = SCOP_PSI_PSI_HVP_TILE_COLS.min(k.max(1));
        for start_col in (0..k).step_by(tile_cols) {
            let end_col = (start_col + tile_cols).min(k);
            let tile = factor.slice(s![.., start_col..end_col]);
            let applied = self
                .family
                .scop_psi_psi_hvp_mat_from_cov(
                    &self.beta,
                    self.row_gamma.slice(s![row_start..row_end, ..]),
                    self.row_h.slice(s![row_start..row_end]),
                    self.row_h_prime.slice(s![row_start..row_end]),
                    cov.view(),
                    cov_i.view(),
                    cov_j.view(),
                    cov_ij.view(),
                    row_start,
                    &self.endpoint_q[row_start..row_end],
                    tile,
                )
                .expect("validated CTN psi-psi batched HVP inputs should not fail");
            out.slice_mut(s![.., start_col..end_col]).assign(&applied);
        }
        out
    }

    fn trace_columns_with_shared_cov(
        &self,
        factor: &Array2<f64>,
        cov: &Array2<f64>,
        cov_i: &Array2<f64>,
        cov_j: &Array2<f64>,
        cov_ij: &Array2<f64>,
        row_start: usize,
        row_end: usize,
    ) -> f64 {
        self.family
            .scop_psi_psi_trace_factor_from_cov(
                &self.beta,
                self.row_gamma.slice(s![row_start..row_end, ..]),
                self.row_h.slice(s![row_start..row_end]),
                self.row_h_prime.slice(s![row_start..row_end]),
                cov.view(),
                cov_i.view(),
                cov_j.view(),
                cov_ij.view(),
                row_start,
                &self.endpoint_q[row_start..row_end],
                factor.view(),
            )
            .expect("validated CTN psi-psi projected trace inputs should not fail")
    }

    fn projected_trace_table(&self, factor: &Array2<f64>) -> Array2<f64> {
        debug_assert_eq!(factor.nrows(), self.p_total());
        let n_axes = self.trace_axes.len();
        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let policy = ResourcePolicy::default_library();
        let rows_per_chunk = crate::resource::rows_for_target_bytes(
            policy.row_chunk_target_bytes,
            p_cov.saturating_mul(n_axes + 2).max(1),
        )
        .max(1)
        .min(n.max(1));

        let op = self.tensor_op();
        let mut out = Array2::<f64>::zeros((n_axes, n_axes));
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let cov = self
                .family
                .covariate_design
                .try_row_chunk(rows.clone())
                .expect("validated CTN psi-psi projected trace covariate chunk should not fail");
            let mut cov_psi_chunks: Vec<Array2<f64>> = Vec::with_capacity(n_axes);
            for &axis in self.trace_axes.iter() {
                cov_psi_chunks.push(
                    op.cov_first_axis_row_chunk_streaming(axis, rows.clone())
                        .expect("validated CTN psi-psi projected trace first-axis chunk should not fail"),
                );
            }

            for i in 0..n_axes {
                for j in i..n_axes {
                    let cov_ij = op
                        .cov_second_axis_row_chunk(self.trace_axes[i], self.trace_axes[j], rows.clone())
                        .expect("validated CTN psi-psi projected trace second-axis chunk should not fail");
                    let value = self.trace_columns_with_shared_cov(
                        factor,
                        &cov,
                        &cov_psi_chunks[i],
                        &cov_psi_chunks[j],
                        &cov_ij,
                        start,
                        end,
                    );
                    out[[i, j]] += value;
                    if i != j {
                        out[[j, i]] += value;
                    }
                }
            }
        }
        out
    }
}

impl HyperOperator for TransformationNormalPsiPsiHessianOperator {
    fn dim(&self) -> usize {
        self.p_total()
    }

    fn mul_vec(&self, v: &Array1<f64>) -> Array1<f64> {
        debug_assert_eq!(v.len(), self.p_total());
        self.apply(v)
    }

    fn bilinear_view(&self, v: ArrayView1<'_, f64>, u: ArrayView1<'_, f64>) -> f64 {
        debug_assert_eq!(v.len(), self.p_total());
        debug_assert_eq!(u.len(), self.p_total());
        self.family
            .scop_psi_psi_bilinear_from_operator(
                &self.beta,
                self.tensor_op(),
                self.axis_i,
                self.axis_j,
                self.row_gamma.view(),
                self.row_h.view(),
                self.row_h_prime.view(),
                self.endpoint_q.as_slice(),
                v,
                u,
            )
            .expect("validated CTN psi-psi bilinear inputs should not fail")
    }

    fn has_fast_bilinear_view(&self) -> bool {
        true
    }

    fn trace_projected_factor(&self, factor: &Array2<f64>) -> f64 {
        debug_assert_eq!(factor.nrows(), self.p_total());
        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let rows_per_chunk = self
            .family
            .scop_psi_pair_rows_per_chunk(p_cov)
            .min(n.max(1));

        let mut total = 0.0;
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let (cov, cov_i, cov_j, cov_ij) = self
                .family
                .scop_psi_pair_cov_chunks(self.tensor_op(), self.axis_i, self.axis_j, start..end)
                .expect("validated CTN psi-psi projected trace covariate chunks should not fail");
            total += self
                .trace_columns_with_shared_cov(factor, &cov, &cov_i, &cov_j, &cov_ij, start, end);
        }
        total
    }

    fn trace_projected_factor_cached(
        &self,
        factor: &Array2<f64>,
        cache: &ProjectedFactorCache,
    ) -> f64 {
        let key = ProjectedFactorKey::from_factor_view(self.trace_cache_id, factor.view());
        let table = cache.get_or_insert_with(key, || self.projected_trace_table(factor));
        table[[self.trace_axis_i_pos, self.trace_axis_j_pos]]
    }

    fn mul_mat(&self, factor: &Array2<f64>) -> Array2<f64> {
        debug_assert_eq!(factor.nrows(), self.p_total());
        let p = factor.nrows();
        let k = factor.ncols();
        let mut out = Array2::<f64>::zeros((p, k));
        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let rows_per_chunk = self
            .family
            .scop_psi_pair_rows_per_chunk(p_cov)
            .min(n.max(1));

        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let (cov, cov_i, cov_j, cov_ij) = self
                .family
                .scop_psi_pair_cov_chunks(self.tensor_op(), self.axis_i, self.axis_j, start..end)
                .expect("validated CTN psi-psi operator covariate chunks should not fail");
            let applied = self
                .apply_columns_with_shared_cov(factor, &cov, &cov_i, &cov_j, &cov_ij, start, end);
            out += &applied;
        }
        out
    }

    fn to_dense(&self) -> Array2<f64> {
        let p = self.p_total();
        let identity = Array2::<f64>::eye(p);
        self.mul_mat(&identity)
    }

    fn is_implicit(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// Response-direction basis construction
// ---------------------------------------------------------------------------

/// Build the response-direction basis: an unconstrained location column plus
/// I-spline values `I_k(y)` with derivatives `M_k(y) = I'_k(y)`.
///
/// Returns (value_basis = `[1, I_k]`, derivative_basis = `[0, M_k]`,
/// penalties embedded with an unpenalized location row/column, regenerated
/// I-spline knots, identity coef_transform for the I-spline shape block).
fn build_response_basis(
    response: &Array1<f64>,
    config: &TransformationNormalConfig,
) -> Result<
    (
        Array2<f64>,
        Array2<f64>,
        Vec<Array2<f64>>,
        Array1<f64>,
        Array2<f64>,
    ),
    String,
> {
    let n = response.len();
    if n < 4 {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!("need at least 4 observations, got {n}"),
        }
        .into());
    }
    for (i, &v) in response.iter().enumerate() {
        if !v.is_finite() {
            return Err(TransformationNormalError::NonFinite {
                reason: format!("response[{i}] is not finite: {v}"),
            }
            .into());
        }
    }

    let response_degree = config.response_degree;
    if response_degree < 1 {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "response_degree must be >= 1 for the I-spline basis, got {response_degree}"
            ),
        }
        .into());
    }
    let k_internal = config.response_num_internal_knots;
    let k_prime = k_internal.checked_sub(2).ok_or_else(|| {
        format!(
            "response_num_internal_knots = {k_internal}; I-spline contract \
             requires K' = K − 2 ≥ 0, so need K ≥ 2"
        )
    })?;

    // Regenerate clamped knots. The I-spline builder integrates a degree
    // `(response_degree + 1)` B-spline basis into a degree-`response_degree`
    // value basis, so the seed-time degree passed here is `response_degree + 1`.
    let mut knots =
        initializewiggle_knots_from_seed(response.view(), response_degree + 1, k_prime)?;
    let response_min = response.iter().copied().fold(f64::INFINITY, f64::min);
    let response_max = response.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let response_span = (response_max - response_min).abs().max(1.0);
    let support_guard = response_span * 1.0e-3;
    let boundary_repeats = response_degree + 2;
    if knots.len() >= 2 * boundary_repeats {
        for idx in 0..boundary_repeats {
            knots[idx] = response_min - support_guard;
            let right_idx = knots.len() - 1 - idx;
            knots[right_idx] = response_max + support_guard;
        }
    }

    // I-spline value basis I_k(y).
    let (i_val_basis, _) = create_basis::<Dense>(
        response.view(),
        KnotSource::Provided(knots.view()),
        response_degree,
        BasisOptions::i_spline(),
    )
    .map_err(|e| e.to_string())?;
    let shape_val = i_val_basis.as_ref().clone();
    let p_shape = shape_val.ncols();

    // M-spline derivative basis M_k(y) = I'_k(y).
    let shape_deriv = create_ispline_derivative_dense(response.view(), &knots, response_degree, 1)
        .map_err(|e| e.to_string())?;
    if shape_deriv.ncols() != p_shape {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "I-spline derivative column count {} does not match value basis {p_shape}",
                shape_deriv.ncols()
            ),
        }
        .into());
    }
    if shape_deriv.nrows() != n {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "I-spline derivative row count {} does not match n = {n}",
                shape_deriv.nrows()
            ),
        }
        .into());
    }

    let p_resp = p_shape + 1;
    let mut resp_val = Array2::<f64>::zeros((n, p_resp));
    resp_val.column_mut(0).fill(1.0);
    resp_val.slice_mut(s![.., 1..]).assign(&shape_val);
    let mut resp_deriv = Array2::<f64>::zeros((n, p_resp));
    resp_deriv.slice_mut(s![.., 1..]).assign(&shape_deriv);

    // SCOP-CTN coef-transform is identity: I-splines have no constant in
    // their span, and squaring γ removes the per-component sign null direction.
    let transform = Array2::<f64>::eye(p_shape);

    // SCOP keeps a Gaussian quadratic prior on the latent γ shape factors,
    // not on the final non-negative I-spline coefficient α = γ². That is a
    // deliberate latent-prior penalty: replacing it with a final-function
    // roughness penalty would make the penalty nonlinear in β and would need
    // a different outer objective normalizer than the quadratic REML term.
    // Embed the latent response penalty into the full response block with an
    // unpenalized location row/column.
    let mut resp_penalties = Vec::new();
    let add_penalty = |order: usize, penalties: &mut Vec<Array2<f64>>| -> Result<(), String> {
        if order == 0 || order >= p_shape {
            return Ok(());
        }
        let shape_pen =
            create_difference_penalty_matrix(p_shape, order, None).map_err(|e| e.to_string())?;
        let mut full_pen = Array2::<f64>::zeros((p_resp, p_resp));
        full_pen.slice_mut(s![1.., 1..]).assign(&shape_pen);
        penalties.push(full_pen);
        Ok(())
    };
    add_penalty(config.response_penalty_order, &mut resp_penalties)?;
    for &order in &config.response_extra_penalty_orders {
        if order == config.response_penalty_order {
            continue;
        }
        add_penalty(order, &mut resp_penalties)?;
    }

    Ok((resp_val, resp_deriv, resp_penalties, knots, transform))
}

fn response_endpoint_value_bases(transform: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let mut lower = Array1::<f64>::zeros(transform.ncols() + 1);
    let mut upper = Array1::<f64>::zeros(transform.ncols() + 1);
    lower[0] = 1.0;
    upper[0] = 1.0;
    for col in 0..transform.ncols() {
        upper[col + 1] = transform.column(col).sum();
    }
    (lower, upper)
}

fn response_floor_offsets(
    response: &Array1<f64>,
    knots: &Array1<f64>,
    response_median: f64,
) -> (Array1<f64>, f64, f64) {
    let row_offsets = response.mapv(|y| TRANSFORMATION_MONOTONICITY_EPS * (y - response_median));
    let lower_y = knots
        .first()
        .copied()
        .unwrap_or_else(|| response.iter().copied().fold(f64::INFINITY, f64::min));
    let upper_y = knots
        .last()
        .copied()
        .unwrap_or_else(|| response.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    (
        row_offsets,
        TRANSFORMATION_MONOTONICITY_EPS * (lower_y - response_median),
        TRANSFORMATION_MONOTONICITY_EPS * (upper_y - response_median),
    )
}

fn effective_response_num_internal_knots(
    config: &TransformationNormalConfig,
    n_obs: usize,
    p_cov: usize,
) -> usize {
    // I-spline contract requires K' = K − 2 ≥ 0, i.e. K ≥ 2 internal knots.
    let min_internal = 2usize;
    let sample_cap = (n_obs / 10).max(min_internal);
    let tensor_width_cap = (BASE_TRANSFORMATION_TENSOR_WIDTH + n_obs / 25)
        .min(LARGE_SAMPLE_TRANSFORMATION_TENSOR_WIDTH);
    let max_resp_cols_from_tensor =
        (tensor_width_cap / p_cov.max(1)).max(config.response_degree + 2);
    // One response column is the unconstrained location; the remaining columns
    // are the I-spline shape block controlled by response_num_internal_knots.
    let max_shape_cols_from_tensor = max_resp_cols_from_tensor.saturating_sub(1);
    let tensor_cap = max_shape_cols_from_tensor
        .saturating_sub(config.response_degree + 1)
        .max(min_internal);
    config
        .response_num_internal_knots
        .min(sample_cap)
        .min(tensor_cap)
        .max(min_internal)
}

// ---------------------------------------------------------------------------
// Tensor product construction
// ---------------------------------------------------------------------------

/// Row-wise Kronecker product of two matrices (same number of rows).
///
/// output\[i, j * p_b + k\] = a\[i, j\] * b\[i, k\]
fn rowwise_kronecker(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    rowwise_kronecker_views(a.view(), b.view())
}

fn rowwise_kronecker_views(a: ArrayView2<'_, f64>, b: ArrayView2<'_, f64>) -> Array2<f64> {
    assert_eq!(a.nrows(), b.nrows());
    let n = a.nrows();
    let pa = a.ncols();
    let pb = b.ncols();
    let mut out = Array2::<f64>::zeros((n, pa * pb));
    {
        use rayon::prelude::*;
        out.axis_chunks_iter_mut(ndarray::Axis(0), 1024)
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_idx, mut out_chunk)| {
                let start = chunk_idx * 1024;
                let rows = out_chunk.nrows();
                for local in 0..rows {
                    let i = start + local;
                    for j in 0..pa {
                        let a_ij = a[[i, j]];
                        if a_ij == 0.0 {
                            continue;
                        }
                        for k in 0..pb {
                            out_chunk[[local, j * pb + k]] = a_ij * b[[i, k]];
                        }
                    }
                }
            });
    }
    out
}

fn assert_rowwise_kronecker_dimensions(n: usize, p_resp: usize, p_cov: usize, context: &str) {
    assert!(
        p_resp > 0 && p_cov > 0,
        "{context} rowwise Kronecker dimensions must be non-empty: n={n}, p_resp={p_resp}, p_cov={p_cov}"
    );
}

fn assert_no_rowwise_kronecker_materialization(n: usize, p_resp: usize, p_cov: usize) -> ! {
    let bytes = n
        .saturating_mul(p_resp)
        .saturating_mul(p_cov)
        .saturating_mul(std::mem::size_of::<f64>());
    // This helper enforces the biobank-scale invariant that CTN
    // `KroneckerDesign` never persists as dense `n × p_resp × p_cov`.
    // SAFETY: return type `!` makes the panic the only valid behavior;
    // reaching here means a caller bypassed the factored-Kron dispatch
    // and would otherwise silently allocate a process-killing buffer.
    panic!(
        "CTN KroneckerDesign must remain factored; refused persistent n x p_response x p_covariate materialization (n={n}, p_response={p_resp}, p_covariate={p_cov}, dense={:.1} MiB)",
        bytes as f64 / (1024.0 * 1024.0),
    );
}

// ---------------------------------------------------------------------------
// Kronecker-aware operator for biobank-scale tensor products
// ---------------------------------------------------------------------------

/// Row-wise Kronecker (face-splitting / Khatri-Rao) design operator for
/// transformation-normal value and derivative rows. It computes `forward_mul`
/// and `transpose_mul` from the natural factor pair without ever materializing
/// the full matrix.
#[derive(Clone)]
enum KroneckerDesign {
    /// Row-wise Khatri–Rao product `A ⊙ B`.
    ///
    /// Element-wise definition (with `n` shared rows, `p_a` and `p_b` columns):
    /// ```text
    ///     (A ⊙ B)[i, a*p_b + b]  =  A[i, a] · B[i, b]
    /// ```
    /// Forward identity (used by `forward_mul`):
    /// ```text
    ///     ((A ⊙ B) β)[i] = Σ_{a,b} A[i,a] · B[i,b] · β_mat[a,b]
    ///                    = Σ_a A[i,a] · (B · β_mat[a, :])[i]
    /// ```
    /// where `β_mat[a, b] = β[a*p_b + b]` (row-major reshape into `p_a × p_b`).
    ///
    /// Storage: `O(n·p_a + storage(B))`. The dense `n × (p_a · p_b)`
    /// materialization is never built.
    KhatriRao {
        left: Array2<f64>,   // n × p_a
        right: DesignMatrix, // n × p_b
    },
}

impl KroneckerDesign {
    fn new_khatri_rao(left: &Array2<f64>, right: DesignMatrix) -> Result<Self, String> {
        if left.nrows() != right.nrows() {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "KroneckerDesign row mismatch: left={}, right={}",
                    left.nrows(),
                    right.nrows()
                ),
            }
            .into());
        }
        assert_rowwise_kronecker_dimensions(left.nrows(), left.ncols(), right.ncols(), "CTN");
        Ok(KroneckerDesign::KhatriRao {
            left: left.clone(),
            right,
        })
    }

    fn nrows(&self) -> usize {
        match self {
            KroneckerDesign::KhatriRao { left, .. } => left.nrows(),
        }
    }

    fn ncols(&self) -> usize {
        match self {
            KroneckerDesign::KhatriRao { left, right } => left.ncols() * right.ncols(),
        }
    }

    /// Compute `self · beta` where beta has length p_a * p_b.
    /// Returns an n-vector.
    fn forward_mul(&self, beta: &Array1<f64>) -> Array1<f64> {
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                let pa = left.ncols();
                let pb = right.ncols();
                let n = left.nrows();
                debug_assert_eq!(beta.len(), pa * pb);
                let beta_mat = beta.view().into_shape_with_order((pa, pb)).unwrap();
                let mut result = Array1::zeros(n);
                if let Some(right_dense) = right.as_dense_ref() {
                    let right_beta = fast_abt(right_dense, &beta_mat);
                    ndarray::Zip::from(&mut result)
                        .and(left.rows())
                        .and(right_beta.rows())
                        .par_for_each(|r, l_row, rb_row| {
                            let mut acc = 0.0;
                            for j in 0..pa {
                                acc += l_row[j] * rb_row[j];
                            }
                            *r = acc;
                        });
                    return result;
                }
                for j in 0..pa {
                    let cov_part = right.apply(&beta_mat.row(j).to_owned());
                    ndarray::Zip::from(&mut result)
                        .and(&cov_part)
                        .and(left.column(j))
                        .par_for_each(|r, &c, &l| *r += l * c);
                }
                result
            }
        }
    }

    /// SCOP-CTN forward: compute
    /// `left[i,0] · γ_0(x_i) + Σ_{k>=1} left[i,k] · γ_k(x_i)²`
    /// where `γ_k(x_i) = (right · β_mat[k, :])[i]` and
    /// `β_mat[k, j] = beta[k * p_cov + j]` (row-major reshape into
    /// `p_resp × p_cov`).
    ///
    /// Equivalent to forming `γ_mat = right · β_matᵀ` (shape `n × p_resp`),
    /// pointwise squaring columns `k>=1`, and contracting against the
    /// corresponding response basis row. The squaring is post-operator — the
    /// underlying `right` operator and the row-replicated Khatri-Rao image
    /// are never materialized. Storage cost matches `forward_mul`: an
    /// intermediate `n × p_resp` `right_beta` plus the `n` output.
    fn scop_affine_squared_forward(&self, beta: &Array1<f64>) -> Array1<f64> {
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                let pa = left.ncols();
                let pb = right.ncols();
                let n = left.nrows();
                debug_assert_eq!(beta.len(), pa * pb);
                let beta_mat = beta.view().into_shape_with_order((pa, pb)).unwrap();
                let mut result = Array1::zeros(n);
                if let Some(right_dense) = right.as_dense_ref() {
                    // right_beta[i, k] = γ_k(x_i)
                    let right_beta = fast_abt(right_dense, &beta_mat);
                    ndarray::Zip::from(&mut result)
                        .and(left.rows())
                        .and(right_beta.rows())
                        .par_for_each(|r, l_row, gamma_row| {
                            let mut acc = l_row[0] * gamma_row[0];
                            for k in 1..pa {
                                let g = gamma_row[k];
                                acc += l_row[k] * g * g;
                            }
                            *r = acc;
                        });
                    return result;
                }
                // Sparse-right fallback: materialize γ_k column-by-column.
                let mut gamma_cols = Array2::<f64>::zeros((n, pa));
                for k in 0..pa {
                    let cov_part = right.apply(&beta_mat.row(k).to_owned());
                    gamma_cols.column_mut(k).assign(&cov_part);
                }
                ndarray::Zip::from(&mut result)
                    .and(left.rows())
                    .and(gamma_cols.rows())
                    .par_for_each(|r, l_row, gamma_row| {
                        let mut acc = l_row[0] * gamma_row[0];
                        for k in 1..pa {
                            let g = gamma_row[k];
                            acc += l_row[k] * g * g;
                        }
                        *r = acc;
                    });
                result
            }
        }
    }

    /// Compute `self^T · v` where v is an n-vector.
    /// Returns a (p_a * p_b)-vector.
    fn transpose_mul(&self, v: &Array1<f64>) -> Array1<f64> {
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                let n = left.nrows();
                let pa = left.ncols();
                let pb = right.ncols();
                debug_assert_eq!(v.len(), n);
                if let Some(right_dense) = right.as_dense_ref() {
                    let weighted_left = weight_rows(left, v);
                    let blocks = fast_atb(right_dense, &weighted_left).reversed_axes();
                    let mut out = Array1::<f64>::zeros(pa * pb);
                    for j in 0..pa {
                        out.slice_mut(s![j * pb..(j + 1) * pb])
                            .assign(&blocks.row(j));
                    }
                    return out;
                }
                let mut out = Array1::<f64>::zeros(pa * pb);
                for j in 0..pa {
                    let mut weighted_v = Array1::<f64>::zeros(n);
                    ndarray::Zip::from(&mut weighted_v)
                        .and(v)
                        .and(left.column(j))
                        .par_for_each(|w, &vi, &li| *w = vi * li);
                    let cov_block = right.apply_transpose(&weighted_v);
                    out.slice_mut(s![j * pb..(j + 1) * pb]).assign(&cov_block);
                }
                out
            }
        }
    }

    /// Compute `self^T · diag(w) · self` (weighted Gram).
    ///
    /// Thin wrapper over `weighted_cross_with(self, self, ...)`. Callers thread
    /// a real `ResourcePolicy` so chunk sizing matches the surrounding workload.
    fn weighted_gram(&self, w: &Array1<f64>, policy: &ResourcePolicy) -> Array2<f64> {
        self.weighted_cross_with(w.view(), self, policy)
            .expect("validated KroneckerDesign weighted Gram dimensions")
    }

    /// Compute `self^T · diag(w) · other` while keeping rowwise-Kronecker
    /// designs in factored form. Returns a dense (pa*pb) x (pc*pd) block matrix.
    pub(crate) fn weighted_cross_with(
        &self,
        weights: ndarray::ArrayView1<'_, f64>,
        other: &KroneckerDesign,
        policy: &ResourcePolicy,
    ) -> Result<Array2<f64>, String> {
        match (self, other) {
            (
                KroneckerDesign::KhatriRao { left: a, right: b },
                KroneckerDesign::KhatriRao { left: c, right: d },
            ) => {
                // If both covariate sides are dense, stay fully factored.
                if let (Some(b_dense), Some(d_dense)) = (b.as_dense_ref(), d.as_dense_ref()) {
                    return factored_weighted_cross(a, b_dense, weights, c, d_dense, policy);
                }
                // Fallback: operator-backed covariate side — iterate (a, c)
                // pairs and let the operator handle the B^T diag(w) D block.
                let n = weights.len();
                let pa = a.ncols();
                let pc = c.ncols();
                let pb = b.ncols();
                let pd = d.ncols();
                if a.nrows() != n || b.nrows() != n || c.nrows() != n || d.nrows() != n {
                    return Err(TransformationNormalError::InvalidInput {
                        reason: format!(
                            "KroneckerDesign::weighted_cross_with row mismatch: weights={n}, \
                         a={}, b={}, c={}, d={}",
                            a.nrows(),
                            b.nrows(),
                            c.nrows(),
                            d.nrows()
                        ),
                    }
                    .into());
                }
                let mut out = Array2::<f64>::zeros((pa * pb, pc * pd));
                let mut pair_weights = Array1::<f64>::zeros(n);
                for ia in 0..pa {
                    let a_col = a.column(ia);
                    for ic in 0..pc {
                        let c_col = c.column(ic);
                        for r in 0..n {
                            pair_weights[r] = weights[r] * a_col[r] * c_col[r];
                        }
                        // Route through the chunked DesignMatrix helper so the
                        // operator-backed covariate factors stay row-chunkable
                        // and never materialize n × p_cov in one shot.
                        let block =
                            chunked_weighted_bt_d_designmatrix(b, pair_weights.view(), d, policy)?;
                        out.slice_mut(s![ia * pb..(ia + 1) * pb, ic * pd..(ic + 1) * pd])
                            .assign(&block);
                    }
                }
                Ok(out)
            }
        }
    }
}

impl LinearOperator for KroneckerDesign {
    fn nrows(&self) -> usize {
        KroneckerDesign::nrows(self)
    }

    fn ncols(&self) -> usize {
        KroneckerDesign::ncols(self)
    }

    fn apply(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.forward_mul(vector)
    }

    fn apply_transpose(&self, vector: &Array1<f64>) -> Array1<f64> {
        self.transpose_mul(vector)
    }

    fn diag_xtw_x(&self, weights: &Array1<f64>) -> Result<Array2<f64>, String> {
        if weights.len() != self.nrows() {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "KroneckerDesign::diag_xtw_x dimension mismatch: weights={}, nrows={}",
                    weights.len(),
                    self.nrows()
                ),
            }
            .into());
        }
        // The `LinearOperator` trait fixes the signature, so this entry point
        // defaults the resource policy. Internal callers in this file go
        // through `weighted_gram` directly with their own policy.
        let policy = ResourcePolicy::default_library();
        Ok(self.weighted_gram(weights, &policy))
    }
}

impl DenseDesignOperator for KroneckerDesign {
    fn row_chunk_into(
        &self,
        rows: std::ops::Range<usize>,
        mut out: ArrayViewMut2<'_, f64>,
    ) -> Result<(), MatrixMaterializationError> {
        if out.nrows() != rows.end - rows.start || out.ncols() != self.ncols() {
            return Err(MatrixMaterializationError::MissingRowChunk {
                context: "KroneckerDesign::row_chunk_into shape mismatch",
            });
        }
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                assert_rowwise_kronecker_dimensions(
                    rows.end.saturating_sub(rows.start),
                    left.ncols(),
                    right.ncols(),
                    "CTN row chunk",
                );
                let left_chunk = left.slice(s![rows.clone(), ..]).to_owned();
                let right_chunk = right.try_row_chunk(rows)?;
                out.assign(&rowwise_kronecker(&left_chunk, &right_chunk));
            }
        }
        Ok(())
    }

    fn to_dense(&self) -> Array2<f64> {
        match self {
            KroneckerDesign::KhatriRao { left, right } => {
                assert_no_rowwise_kronecker_materialization(
                    left.nrows(),
                    left.ncols(),
                    right.ncols(),
                );
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Kronecker-form penalties
// ---------------------------------------------------------------------------

/// A penalty matrix in separable Kronecker form: `S_left ⊗ S_right`.
///
/// Build tensor product penalties in Kronecker-separable form.
fn build_tensor_penalties_kronecker(
    response_penalties: &[Array2<f64>],
    covariate_penalties: Vec<PenaltyMatrix>,
    p_resp: usize,
    p_cov: usize,
    config: &TransformationNormalConfig,
) -> Result<Vec<PenaltyMatrix>, String> {
    let eye_cov = Array2::<f64>::eye(p_cov);
    let mut penalties = Vec::new();

    let mut shape_resp = Array2::<f64>::eye(p_resp);
    shape_resp[[0, 0]] = 0.0;

    // Covariate roughness is a latent γ prior on the squared monotone shape
    // rows. The derivative-free location row is the conditional centering
    // field itself; penalizing it by REML under-corrects broad population
    // shifts and leaves h(Y|x) calibrated only marginally instead of
    // conditionally.
    for s_cov in covariate_penalties {
        let right = match s_cov {
            PenaltyMatrix::Dense(right) => right,
            penalty @ PenaltyMatrix::Blockwise { .. } => penalty.to_dense(),
            PenaltyMatrix::Labeled { inner, .. } => inner.to_dense(),
            PenaltyMatrix::KroneckerFactored { .. } => {
                return Err(
                    "transformation covariate penalties must be single-block, not already Kronecker-factored"
                        .to_string(),
                )
            }
        };
        penalties.push(PenaltyMatrix::KroneckerFactored {
            left: shape_resp.clone(),
            right,
        });
    }

    // Response penalties: S_resp_m ⊗ I_cov
    for s_resp in response_penalties {
        penalties.push(PenaltyMatrix::KroneckerFactored {
            left: s_resp.clone(),
            right: eye_cov.clone(),
        });
    }

    // Double penalty: shape-row ridge only. The location row is identified by
    // the likelihood as the conditional centering field; keep it outside every
    // SCOP roughness/shrinkage penalty so population shifts can be calibrated
    // in the selected covariate span.
    if config.double_penalty {
        penalties.push(PenaltyMatrix::KroneckerFactored {
            left: shape_resp,
            right: eye_cov,
        });
    }

    Ok(penalties)
}

// ---------------------------------------------------------------------------
// Warm start
// ---------------------------------------------------------------------------

/// Compute initial β so that the SCOP-CTN model starts with a positive
/// derivative and approximately centered transformed response.
///
/// SCOP is nonlinear in the shape rows: `h'=Σ M_k(y)γ_k(x)^2`. A linear joint
/// least-squares solve fits the wrong parameterization, so the warm start
/// initializes shape rows to a positive constant derivative scale and then
/// solves only the location row `b(x)` against the remaining affine target.
fn compute_warm_start(
    response: &Array1<f64>,
    weights: &Array1<f64>,
    offset: &Array1<f64>,
    x_val_kron: &KroneckerDesign,
    x_deriv_kron: &KroneckerDesign,
    covariate_design: &DesignMatrix,
    covariate_penalties: &[PenaltyMatrix],
    p_resp: usize,
    p_cov: usize,
    warm_start: Option<&TransformationWarmStart>,
) -> Result<Array1<f64>, String> {
    let n = response.len();
    let p_total = p_resp * p_cov;
    if p_resp < 2 {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "transformation warm start requires at least 2 response basis columns, got {p_resp}"
            ),
        }
        .into());
    }

    let default_ws;
    let ws = match warm_start {
        Some(ws) => ws,
        None => {
            default_ws = estimate_default_warm_start(
                response,
                weights,
                covariate_design,
                covariate_penalties,
            )?;
            &default_ws
        }
    };
    if ws.location.len() != n || ws.scale.len() != n {
        return Err(TransformationNormalError::InvalidInput {
            reason: "warm start location/scale length mismatch".to_string(),
        }
        .into());
    }

    // Per-row affine targets for the transformation scale.
    let mut target_h = Array1::<f64>::zeros(n);
    let mut target_hp = Array1::<f64>::zeros(n);
    for i in 0..n {
        let tau = ws.scale[i].max(1e-12);
        let inv_tau = 1.0 / tau;
        target_h[i] = (response[i] - ws.location[i]) * inv_tau - offset[i];
        target_hp[i] = inv_tau;
    }

    // β-native SCOP seed. A tempting alternative is to solve a linear
    // least-squares problem for α_k(x)=γ_k(x)^2 and then project sqrt(α_k)
    // back into the covariate basis. That is not an invariant transformation:
    // squaring the projected sqrt field no longer equals the solved α field,
    // and small projection errors can explode into huge positive I-spline
    // shape terms. Instead seed the monotone shape rows directly with a
    // constant positive γ that matches the weighted average derivative target,
    // then solve only the unconstrained location row in β-space.
    let weight_sum = weights.iter().copied().sum::<f64>();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err(TransformationNormalError::DesignDegenerate {
            reason: "SCOP warm start requires positive finite total weight".to_string(),
        }
        .into());
    }
    let mean_target_hp = weights
        .iter()
        .zip(target_hp.iter())
        .map(|(&w, &hp)| w * hp)
        .sum::<f64>()
        / weight_sum;
    if !(mean_target_hp.is_finite() && mean_target_hp > 0.0) {
        return Err(TransformationNormalError::NonFinite {
            reason: format!(
                "SCOP warm start derivative target is not positive finite: {mean_target_hp}"
            ),
        }
        .into());
    }

    let mut beta = Array1::<f64>::zeros(p_total);
    for k in 1..p_resp {
        beta[k * p_cov] = 1.0;
    }
    let unit_shape_hp = x_deriv_kron.scop_affine_squared_forward(&beta);
    let mean_unit_shape_hp = weights
        .iter()
        .zip(unit_shape_hp.iter())
        .map(|(&w, &hp)| w * hp)
        .sum::<f64>()
        / weight_sum;
    if !(mean_unit_shape_hp.is_finite() && mean_unit_shape_hp > 0.0) {
        return Err(TransformationNormalError::NonFinite {
            reason: format!(
                "SCOP warm start unit shape derivative is not positive finite: {mean_unit_shape_hp}"
            ),
        }
        .into());
    }
    let gamma_const = (mean_target_hp / mean_unit_shape_hp).sqrt();
    if !(gamma_const.is_finite() && gamma_const > 0.0) {
        return Err(TransformationNormalError::NonFinite {
            reason: format!("SCOP warm start shape scale is not positive finite: {gamma_const}"),
        }
        .into());
    }
    beta.fill(0.0);
    for k in 1..p_resp {
        beta[k * p_cov] = gamma_const;
    }

    let shape_h = x_val_kron.scop_affine_squared_forward(&beta);
    let location_target = &target_h - &shape_h;
    let zero_offset = Array1::<f64>::zeros(n);
    let log_lambdas = Array1::<f64>::zeros(covariate_penalties.len());
    let location_beta = solve_penalizedweighted_projection(
        covariate_design,
        &zero_offset,
        &location_target,
        weights,
        covariate_penalties,
        &log_lambdas,
        1e-12,
    )?;
    for c in 0..p_cov {
        beta[c] = location_beta[c];
    }

    if beta.iter().any(|v| !v.is_finite()) {
        return Err(TransformationNormalError::DesignDegenerate {
            reason: "SCOP warm start produced non-finite coefficients".to_string(),
        }
        .into());
    }
    Ok(beta)
}

fn estimate_default_warm_start(
    response: &Array1<f64>,
    weights: &Array1<f64>,
    covariate_design: &DesignMatrix,
    covariate_penalties: &[PenaltyMatrix],
) -> Result<TransformationWarmStart, String> {
    let n = response.len();
    if weights.len() != n {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "transformation warm start weights length mismatch: response={}, weights={}",
                n,
                weights.len()
            ),
        }
        .into());
    }
    let zero_offset = Array1::zeros(n);
    let log_lambdas = Array1::zeros(covariate_penalties.len());
    let beta_location = solve_penalizedweighted_projection(
        covariate_design,
        &zero_offset,
        response,
        weights,
        covariate_penalties,
        &log_lambdas,
        1e-8,
    )?;
    let location = covariate_design.matrixvectormultiply(&beta_location);
    let weight_sum = weights.iter().copied().sum::<f64>();
    if !(weight_sum.is_finite() && weight_sum > 0.0) {
        return Err(TransformationNormalError::DesignDegenerate {
            reason: "transformation warm start requires positive finite total weight".to_string(),
        }
        .into());
    }
    let weighted_ss = response
        .iter()
        .zip(location.iter())
        .zip(weights.iter())
        .map(|((&y, &mu), &w)| {
            let resid = y - mu;
            w * resid * resid
        })
        .sum::<f64>();
    if !weighted_ss.is_finite() {
        return Err(TransformationNormalError::DesignDegenerate {
            reason: "transformation warm start residual variance is not finite".to_string(),
        }
        .into());
    }
    let global_scale = (weighted_ss / weight_sum).sqrt().max(1e-6);
    let residual_floor = global_scale * 1e-3 + 1e-12;
    let log_scale_target =
        Array1::from_iter(response.iter().zip(location.iter()).map(|(&y, &mu)| {
            (y - mu).abs().max(residual_floor).ln() - STANDARD_NORMAL_MEAN_LOG_ABS
        }));
    let beta_log_scale = solve_penalizedweighted_projection(
        covariate_design,
        &zero_offset,
        &log_scale_target,
        weights,
        covariate_penalties,
        &log_lambdas,
        1e-8,
    )?;
    let scale = covariate_design
        .matrixvectormultiply(&beta_log_scale)
        .mapv(|eta| eta.exp().max(residual_floor));

    Ok(TransformationWarmStart { location, scale })
}

fn calibrate_transformation_scores(
    family: &TransformationNormalFamily,
    mut fit: UnifiedFitResult,
) -> Result<(UnifiedFitResult, TransformationScoreCalibration), String> {
    let Some(block_state) = fit.block_states.first() else {
        return Err(TransformationNormalError::InvalidInput {
            reason: "transformation score calibration requires one fitted block".to_string(),
        }
        .into());
    };
    let p_resp = family.response_val_basis.ncols();
    let p_cov = family.covariate_design.ncols();
    let p_total = p_resp * p_cov;
    if block_state.beta.len() != p_total {
        return Err(TransformationNormalError::InvalidInput {
            reason: format!(
                "transformation calibration beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                block_state.beta.len()
            ),
        }
        .into());
    }

    let row_quantities = family.row_quantities(&block_state.beta)?;
    let mut pit_values = Vec::with_capacity(family.n_obs());
    for i in 0..family.n_obs() {
        pit_values.push(
            transformation_normal_pit_score(
                row_quantities.h[i],
                row_quantities.h_lower[i],
                row_quantities.h_upper[i],
                TRANSFORMATION_SCORE_PIT_CLIP_EPS,
            )
            .map_err(|err| {
                format!("transformation-normal fitted PIT score failed at row {i}: {err}")
            })?,
        );
    }
    let calibrated_h = Array1::from_vec(pit_values);
    if calibrated_h
        .iter()
        .any(|value| !value.is_finite() || value.abs() > TRANSFORMATION_NORMAL_H_ABS_MAX)
    {
        return Err(
            "transformation PIT calibration produced non-finite or out-of-range scores".to_string(),
        );
    }

    if let Some(state) = fit.block_states.first_mut() {
        state.eta = calibrated_h;
    }
    fit.log_likelihood = row_quantities.log_likelihood;
    fit.deviance = -2.0 * row_quantities.log_likelihood;
    Ok((fit, TransformationScoreCalibration::finite_support_pit()))
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

/// Multiply each row of a matrix by the corresponding weight.
fn weight_rows(x: &Array2<f64>, w: &Array1<f64>) -> Array2<f64> {
    let n = x.nrows();
    let p = x.ncols();
    debug_assert_eq!(n, w.len());
    let mut out = Array2::zeros((n, p));
    ndarray::Zip::from(out.rows_mut())
        .and(x.rows())
        .and(w)
        .par_for_each(|mut o_row, x_row, &wi| {
            for j in 0..p {
                o_row[j] = x_row[j] * wi;
            }
        });
    out
}

struct TensorKroneckerPsiOperator {
    response_val_basis: Arc<Array2<f64>>,
    covariate_design: DesignMatrix,
    covariate_derivs: Vec<CustomFamilyBlockPsiDerivative>,
    covariate_first_cache: Arc<Vec<Mutex<Option<Arc<Array2<f64>>>>>>,
}

impl TensorKroneckerPsiOperator {
    fn n_data(&self) -> usize {
        self.response_val_basis.nrows()
    }

    fn p_resp(&self) -> usize {
        self.response_val_basis.ncols()
    }

    fn p_cov(&self) -> usize {
        self.covariate_design.ncols()
    }

    fn p_out(&self) -> usize {
        self.p_resp() * self.p_cov()
    }

    fn cov_deriv(
        &self,
        axis: usize,
    ) -> Result<&CustomFamilyBlockPsiDerivative, crate::terms::basis::BasisError> {
        self.covariate_derivs.get(axis).ok_or_else(|| {
            crate::terms::basis::BasisError::InvalidInput(format!(
                "tensor Kronecker psi axis {axis} out of bounds for {} axes",
                self.covariate_derivs.len()
            ))
        })
    }

    fn cov_forward_first(
        &self,
        axis: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let deriv = self.cov_deriv(axis)?;
        if deriv.x_psi.nrows() == self.n_data() && deriv.x_psi.ncols() == self.p_cov() {
            return Ok(crate::faer_ndarray::fast_av(&deriv.x_psi, u));
        }
        let Some(op) = deriv.implicit_operator.as_ref() else {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "missing covariate psi operator for axis {axis}"
            )));
        };
        op.forward_mul(deriv.implicit_axis, u)
    }

    fn cov_transpose_first(
        &self,
        axis: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let deriv = self.cov_deriv(axis)?;
        if deriv.x_psi.nrows() == self.n_data() && deriv.x_psi.ncols() == self.p_cov() {
            return Ok(crate::faer_ndarray::fast_atv(&deriv.x_psi, v));
        }
        let Some(op) = deriv.implicit_operator.as_ref() else {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "missing covariate psi transpose operator for axis {axis}"
            )));
        };
        op.transpose_mul(deriv.implicit_axis, v)
    }

    fn cov_forward_second(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let deriv_d = self.cov_deriv(axis_d)?;
        if let Some(op) = deriv_d.implicit_operator.as_ref()
            && deriv_d.implicit_group_id.is_some()
            && deriv_d.implicit_group_id == self.cov_deriv(axis_e)?.implicit_group_id
        {
            if deriv_d.implicit_axis == self.cov_deriv(axis_e)?.implicit_axis {
                return op.forward_mul_second_diag(deriv_d.implicit_axis, u);
            }
            return op.forward_mul_second_cross(
                deriv_d.implicit_axis,
                self.cov_deriv(axis_e)?.implicit_axis,
                u,
            );
        }
        if let Some(rows) = deriv_d.x_psi_psi.as_ref()
            && let Some(mat) = rows.get(axis_e)
        {
            if mat.nrows() == self.n_data() && mat.ncols() == self.p_cov() {
                return Ok(crate::faer_ndarray::fast_av(mat, u));
            }
        }
        Ok(Array1::<f64>::zeros(self.n_data()))
    }

    fn cov_transpose_second(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let deriv_d = self.cov_deriv(axis_d)?;
        if let Some(op) = deriv_d.implicit_operator.as_ref()
            && deriv_d.implicit_group_id.is_some()
            && deriv_d.implicit_group_id == self.cov_deriv(axis_e)?.implicit_group_id
        {
            if deriv_d.implicit_axis == self.cov_deriv(axis_e)?.implicit_axis {
                return op.transpose_mul_second_diag(deriv_d.implicit_axis, v);
            }
            return op.transpose_mul_second_cross(
                deriv_d.implicit_axis,
                self.cov_deriv(axis_e)?.implicit_axis,
                v,
            );
        }
        if let Some(rows) = deriv_d.x_psi_psi.as_ref()
            && let Some(mat) = rows.get(axis_e)
        {
            if mat.nrows() == self.n_data() && mat.ncols() == self.p_cov() {
                return Ok(crate::faer_ndarray::fast_atv(mat, v));
            }
        }
        Ok(Array1::<f64>::zeros(self.p_cov()))
    }

    fn cov_first_axis_row_chunk(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let deriv = self.cov_deriv(axis)?;
        if deriv.x_psi.nrows() == self.n_data() && deriv.x_psi.ncols() == self.p_cov() {
            return Ok(deriv.x_psi.slice(s![rows, ..]).to_owned());
        }
        let Some(op) = deriv.implicit_operator.as_ref() else {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "missing covariate psi row chunk operator for axis {axis}"
            )));
        };
        if self.cov_first_axis_cache_fits_budget() && op.as_materializable().is_some() {
            let cached = self.materialize_cov_first_axis_arc(axis)?;
            return Ok(cached.slice(s![rows, ..]).to_owned());
        }
        op.row_chunk_first(deriv.implicit_axis, rows)
    }

    fn cov_first_axis_row_chunk_streaming(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let deriv = self.cov_deriv(axis)?;
        if deriv.x_psi.nrows() == self.n_data() && deriv.x_psi.ncols() == self.p_cov() {
            return Ok(deriv.x_psi.slice(s![rows, ..]).to_owned());
        }
        let Some(op) = deriv.implicit_operator.as_ref() else {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "missing covariate psi streaming row chunk operator for axis {axis}"
            )));
        };
        op.row_chunk_first(deriv.implicit_axis, rows)
    }

    fn cov_second_axis_row_chunk(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let deriv_d = self.cov_deriv(axis_d)?;
        let deriv_e = self.cov_deriv(axis_e)?;
        if let Some(op) = deriv_d.implicit_operator.as_ref()
            && deriv_d.implicit_group_id.is_some()
            && deriv_d.implicit_group_id == deriv_e.implicit_group_id
        {
            if deriv_d.implicit_axis == deriv_e.implicit_axis {
                return op.row_chunk_second_diag(deriv_d.implicit_axis, rows);
            }
            return op.row_chunk_second_cross(deriv_d.implicit_axis, deriv_e.implicit_axis, rows);
        }
        if let Some(second_rows) = deriv_d.x_psi_psi.as_ref()
            && let Some(mat) = second_rows.get(axis_e)
        {
            if mat.nrows() == self.n_data() && mat.ncols() == self.p_cov() {
                return Ok(mat.slice(s![rows, ..]).to_owned());
            }
        }
        Ok(Array2::<f64>::zeros((rows.end - rows.start, self.p_cov())))
    }

    fn lifted_row_chunk_from_cov(
        &self,
        rows: std::ops::Range<usize>,
        cov: &Array2<f64>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let n_rows = rows.end - rows.start;
        if cov.nrows() != n_rows || cov.ncols() != self.p_cov() {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "tensor Kronecker covariate row chunk shape {}x{} != expected {}x{}",
                cov.nrows(),
                cov.ncols(),
                n_rows,
                self.p_cov()
            )));
        }
        let resp = self.response_val_basis.slice(s![rows, ..]);
        Ok(rowwise_kronecker_views(resp, cov.view()))
    }

    fn materialize_cov_first_axis_uncached(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let deriv = self.cov_deriv(axis)?;
        if deriv.x_psi.nrows() == self.n_data() && deriv.x_psi.ncols() == self.p_cov() {
            return Ok(deriv.x_psi.clone());
        }
        let Some(op) = deriv.implicit_operator.as_ref() else {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "missing covariate psi materialization for axis {axis}"
            )));
        };
        let mat_op = op.as_materializable().ok_or_else(|| {
            crate::terms::basis::BasisError::InvalidInput(format!(
                "covariate psi operator for axis {axis} does not support dense materialization"
            ))
        })?;
        mat_op.materialize_first(deriv.implicit_axis)
    }

    fn cov_first_axis_cache_fits_budget(&self) -> bool {
        let policy = ResourcePolicy::default_library();
        let per_axis_bytes = self
            .n_data()
            .saturating_mul(self.p_cov())
            .saturating_mul(std::mem::size_of::<f64>());
        let all_axes_bytes = per_axis_bytes.saturating_mul(self.covariate_derivs.len());
        per_axis_bytes <= policy.max_single_materialization_bytes
            && all_axes_bytes <= policy.max_operator_cache_bytes
    }

    fn materialize_cov_first_axis_arc(
        &self,
        axis: usize,
    ) -> Result<Arc<Array2<f64>>, crate::terms::basis::BasisError> {
        if axis >= self.covariate_derivs.len() {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "tensor Kronecker psi axis {axis} out of bounds for {} axes",
                self.covariate_derivs.len()
            )));
        }
        let axis_cache = &self.covariate_first_cache[axis];
        let mut cache = axis_cache.lock().map_err(|_| {
            crate::terms::basis::BasisError::InvalidInput(format!(
                "tensor Kronecker covariate first-derivative cache mutex poisoned for axis {axis}"
            ))
        })?;
        if let Some(cached) = cache.as_ref() {
            return Ok(cached.clone());
        }

        let materialized = Arc::new(self.materialize_cov_first_axis_uncached(axis)?);
        *cache = Some(materialized.clone());
        Ok(materialized)
    }

    fn materialize_cov_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let mut unit = Array1::<f64>::zeros(self.covariate_derivs.len());
        unit[axis] = 1.0;
        self.materialize_cov_directional(&unit.view())
    }

    /// Per-axis covariate first-derivative materialization for axis `axis`,
    /// equivalent to the unit-vector dispatch through
    /// [`materialize_cov_directional`].
    fn materialize_cov_first_axis(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok((*self.materialize_cov_first_axis_arc(axis)?).clone())
    }

    /// Directional `Σ_j v_psi[j] · ∂C/∂ψ_j` returning an `n × p_cov` matrix.
    /// Calling with `v_psi = e_axis` matches [`materialize_cov_first_axis`] at axis.
    /// Used by the directional outer-HVP path to compute `cov_v` once per HVP
    /// instead of materializing each per-axis cov_j.
    fn materialize_cov_directional(
        &self,
        v_psi: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        if v_psi.len() != self.covariate_derivs.len() {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "directional ψ vector length {} does not match {} ψ axes",
                v_psi.len(),
                self.covariate_derivs.len()
            )));
        }
        let mut out = Array2::<f64>::zeros((self.n_data(), self.p_cov()));
        for (axis, &coef) in v_psi.iter().enumerate() {
            if coef == 0.0 {
                continue;
            }
            let contrib = self.materialize_cov_first_axis(axis)?;
            out.scaled_add(coef, &contrib);
        }
        Ok(out)
    }

    fn lifted_forward(
        &self,
        resp_basis: &Array2<f64>,
        axis: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let n = self.n_data();
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let beta = u
            .to_owned()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|_| {
                crate::terms::basis::BasisError::InvalidInput(
                    "tensor psi coefficient reshape failed".to_string(),
                )
            })?;
        let mut out = Array1::<f64>::zeros(n);
        for j in 0..p_resp {
            let cov_part = self.cov_forward_first(axis, &beta.row(j))?;
            ndarray::Zip::from(&mut out)
                .and(&cov_part)
                .and(resp_basis.column(j))
                .par_for_each(|o, &c, &r| *o += r * c);
        }
        Ok(out)
    }

    fn lifted_transpose(
        &self,
        resp_basis: &Array2<f64>,
        axis: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let n = self.n_data();
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let mut out = Array1::<f64>::zeros(p_resp * p_cov);
        for j in 0..p_resp {
            let mut weighted_v = Array1::<f64>::zeros(n);
            ndarray::Zip::from(&mut weighted_v)
                .and(resp_basis.column(j))
                .and(v)
                .par_for_each(|w, &r, &vi| *w = r * vi);
            let cov_block = self.cov_transpose_first(axis, &weighted_v.view())?;
            out.slice_mut(s![j * p_cov..(j + 1) * p_cov])
                .assign(&cov_block);
        }
        Ok(out)
    }

    fn lifted_forward_second(
        &self,
        resp_basis: &Array2<f64>,
        axis_d: usize,
        axis_e: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let n = self.n_data();
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let beta = u
            .to_owned()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|_| {
                crate::terms::basis::BasisError::InvalidInput(
                    "tensor psi second coefficient reshape failed".to_string(),
                )
            })?;
        let mut out = Array1::<f64>::zeros(n);
        for j in 0..p_resp {
            let cov_part = self.cov_forward_second(axis_d, axis_e, &beta.row(j))?;
            ndarray::Zip::from(&mut out)
                .and(&cov_part)
                .and(resp_basis.column(j))
                .par_for_each(|o, &c, &r| *o += r * c);
        }
        Ok(out)
    }

    fn lifted_transpose_second(
        &self,
        resp_basis: &Array2<f64>,
        axis_d: usize,
        axis_e: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let n = self.n_data();
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let mut out = Array1::<f64>::zeros(p_resp * p_cov);
        for j in 0..p_resp {
            let mut weighted_v = Array1::<f64>::zeros(n);
            ndarray::Zip::from(&mut weighted_v)
                .and(resp_basis.column(j))
                .and(v)
                .par_for_each(|w, &r, &vi| *w = r * vi);
            let cov_block = self.cov_transpose_second(axis_d, axis_e, &weighted_v.view())?;
            out.slice_mut(s![j * p_cov..(j + 1) * p_cov])
                .assign(&cov_block);
        }
        Ok(out)
    }

    fn materialize_lifted(&self, resp_basis: &Array2<f64>, cov: &Array2<f64>) -> Array2<f64> {
        rowwise_kronecker(resp_basis, cov)
    }

    /// Internal directional accumulator on a chosen response basis:
    /// returns `Σ_j v_psi[j] · lifted_forward(resp_basis, j, β)`.
    ///
    /// Skips axes with `v_psi[j] == 0`, so calls with `v_psi = e_k` are
    /// numerically equivalent to a single direct `lifted_forward(_, k, β)` call
    /// (no extra n-vector accumulation, no rounding).
    fn lifted_forward_directional(
        &self,
        resp_basis: &Array2<f64>,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        beta: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        if v_psi.len() != self.covariate_derivs.len() {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "directional ψ vector length {} does not match {} ψ axes",
                v_psi.len(),
                self.covariate_derivs.len()
            )));
        }
        let mut out = Array1::<f64>::zeros(self.n_data());
        for (axis, &coef) in v_psi.iter().enumerate() {
            if coef == 0.0 {
                continue;
            }
            let contrib = self.lifted_forward(resp_basis, axis, beta)?;
            out.scaled_add(coef, &contrib);
        }
        Ok(out)
    }

    /// Internal directional accumulator on a chosen response basis:
    /// returns `Σ_j v_psi[j] · lifted_transpose(resp_basis, j, residual)`.
    ///
    /// Returns a single `(p_resp · p_cov)`-vector, NOT a stack indexed by axis.
    fn lifted_transpose_directional(
        &self,
        resp_basis: &Array2<f64>,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        residual: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        if v_psi.len() != self.covariate_derivs.len() {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "directional ψ vector length {} does not match {} ψ axes",
                v_psi.len(),
                self.covariate_derivs.len()
            )));
        }
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let mut out = Array1::<f64>::zeros(p_resp * p_cov);
        for (axis, &coef) in v_psi.iter().enumerate() {
            if coef == 0.0 {
                continue;
            }
            let contrib = self.lifted_transpose(resp_basis, axis, residual)?;
            out.scaled_add(coef, &contrib);
        }
        Ok(out)
    }

    /// Internal bilinear directional accumulator on a chosen response basis:
    /// returns `Σ_{j,k} v_psi[j] · w_psi[k] · lifted_transpose_second(resp_basis, j, k, residual)`.
    /// Mirror of [`lifted_forward_second_directional`] for the transpose direction.
    fn lifted_transpose_second_directional(
        &self,
        resp_basis: &Array2<f64>,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        w_psi: &ndarray::ArrayView1<'_, f64>,
        residual: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let q = self.covariate_derivs.len();
        if v_psi.len() != q || w_psi.len() != q {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "directional ψ vectors length ({}, {}) do not match {} ψ axes",
                v_psi.len(),
                w_psi.len(),
                q
            )));
        }
        let p_resp = resp_basis.ncols();
        let p_cov = self.p_cov();
        let mut out = Array1::<f64>::zeros(p_resp * p_cov);
        for j in 0..q {
            for k in j..q {
                let coef = if j == k {
                    v_psi[j] * w_psi[j]
                } else {
                    v_psi[j] * w_psi[k] + v_psi[k] * w_psi[j]
                };
                if coef == 0.0 {
                    continue;
                }
                let contrib = self.lifted_transpose_second(resp_basis, j, k, residual)?;
                out.scaled_add(coef, &contrib);
            }
        }
        Ok(out)
    }

    /// Internal bilinear directional accumulator on a chosen response basis:
    /// returns `Σ_{j,k} v_psi[j] · w_psi[k] · lifted_forward_second(resp_basis, j, k, β)`.
    fn lifted_forward_second_directional(
        &self,
        resp_basis: &Array2<f64>,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        w_psi: &ndarray::ArrayView1<'_, f64>,
        beta: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let q = self.covariate_derivs.len();
        if v_psi.len() != q || w_psi.len() != q {
            return Err(crate::terms::basis::BasisError::InvalidInput(format!(
                "directional ψ vectors length ({}, {}) do not match {} ψ axes",
                v_psi.len(),
                w_psi.len(),
                q
            )));
        }
        let mut out = Array1::<f64>::zeros(self.n_data());
        for j in 0..q {
            for k in j..q {
                let coef = if j == k {
                    v_psi[j] * w_psi[j]
                } else {
                    v_psi[j] * w_psi[k] + v_psi[k] * w_psi[j]
                };
                if coef == 0.0 {
                    continue;
                }
                let contrib = self.lifted_forward_second(resp_basis, j, k, beta)?;
                out.scaled_add(coef, &contrib);
            }
        }
        Ok(out)
    }

    /// Directional `Σ_j v_psi[j] · ∂(X · β)/∂ψ_j` on the value response basis.
    /// Calling with `v_psi = e_k` returns the same n-vector as
    /// [`forward_mul`](Self::forward_mul) at axis `k` (zero entries skipped).
    fn forward_directional(
        &self,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        beta: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let resp_basis = self.response_val_basis.clone();
        self.lifted_forward_directional(&resp_basis, v_psi, beta)
    }

    /// Directional transpose against the value response basis.
    /// Calling with `v_psi = e_k` matches the per-axis `transpose_mul(k, residual)`
    /// surface on the trait.
    fn transpose_directional(
        &self,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        residual: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let resp_basis = self.response_val_basis.clone();
        self.lifted_transpose_directional(&resp_basis, v_psi, residual)
    }

    /// Bilinear directional `Σ_{j,k} v_psi[j] · w_psi[k] · ∂²(X · β)/∂ψ_j∂ψ_k`
    /// on the value response basis. With `v_psi = e_a, w_psi = e_b` matches
    /// `forward_mul_second_diag(a)` (when a==b) or `forward_mul_second_cross(a,b)`.
    fn forward_second_directional(
        &self,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        w_psi: &ndarray::ArrayView1<'_, f64>,
        beta: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let resp_basis = self.response_val_basis.clone();
        self.lifted_forward_second_directional(&resp_basis, v_psi, w_psi, beta)
    }

    /// Bilinear directional second-order transpose on the value response basis.
    /// With `v_psi = e_a, w_psi = e_b` matches the per-axis-pair
    /// `transpose_mul_second_diag(a)` (when a==b) or `transpose_mul_second_cross(a,b)`.
    fn transpose_second_directional(
        &self,
        v_psi: &ndarray::ArrayView1<'_, f64>,
        w_psi: &ndarray::ArrayView1<'_, f64>,
        residual: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let resp_basis = self.response_val_basis.clone();
        self.lifted_transpose_second_directional(&resp_basis, v_psi, w_psi, residual)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_family::custom_family_outer_derivatives;
    use crate::testing::assert_matrix_derivativefd;
    use ndarray::array;

    fn dense_first_order_psi_hessian(terms: &ExactNewtonJointPsiTerms) -> Array2<f64> {
        if terms.hessian_psi.nrows() > 0 {
            terms.hessian_psi.clone()
        } else {
            terms
                .hessian_psi_operator
                .as_ref()
                .expect("CTN psi first-order terms must expose either dense Hessian or operator")
                .to_dense()
        }
    }

    #[test]
    fn ctn_penalty_scale_seed_uses_likelihood_to_penalty_ratio() {
        let likelihood_gram = array![[8.0, 0.0], [0.0, 8.0]];
        let penalties = vec![
            PenaltyMatrix::Dense(array![[2.0, 0.0], [0.0, 2.0]]),
            PenaltyMatrix::Dense(array![[4.0, 0.0], [0.0, 4.0]]),
        ];
        let rho = ctn_penalty_scale_log_lambdas(&penalties, &likelihood_gram);
        assert!((rho[0] - 4.0_f64.ln()).abs() < 1.0e-12);
        assert!((rho[1] - 2.0_f64.ln()).abs() < 1.0e-12);
    }

    #[test]
    fn tensor_psi_penalty_derivatives_follow_shape_only_scop_layout() {
        let response = array![-1.0, -0.2, 0.6, 1.3];
        let (val_basis, deriv_basis, knots, transform, p_resp) = toy_response_basis(&response);
        let weights = Array1::from_elem(response.len(), 1.0);
        let offset = Array1::zeros(response.len());
        let cov_design = array![[1.0, 0.2], [1.0, -0.1], [1.0, 0.4], [1.0, -0.3]];
        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            &response,
            val_basis,
            deriv_basis,
            vec![],
            knots,
            toy_scop_ctn_config().response_degree,
            transform,
            &weights,
            &offset,
            DesignMatrix::Dense(DenseDesignMatrix::from(cov_design.clone())),
            vec![],
            &toy_scop_ctn_config(),
            None,
        )
        .expect("toy transformation family");

        let ds0 = array![[1.0, 0.25], [0.25, 2.0]];
        let ds1 = array![[3.0, -0.5], [-0.5, 4.0]];
        let ds1_second = array![[5.0, 0.75], [0.75, 6.0]];
        let mut cov_deriv = CustomFamilyBlockPsiDerivative::new(
            None,
            Array2::zeros((response.len(), cov_design.ncols())),
            Array2::zeros((0, 0)),
            Some(vec![(0, ds0.clone()), (1, ds1.clone())]),
            None,
            None,
            Some(vec![vec![(1, ds1_second.clone())]]),
        );
        cov_deriv.s_psi_penalty_components = Some(vec![
            (0, PenaltyMatrix::Dense(ds0.clone())),
            (1, PenaltyMatrix::Dense(ds1.clone())),
        ]);
        cov_deriv.s_psi_psi_penalty_components =
            Some(vec![vec![(1, PenaltyMatrix::Dense(ds1_second.clone()))]]);

        let tensor_derivs =
            build_tensor_psi_derivatives(&family, &[cov_deriv]).expect("tensor derivatives");
        let first = tensor_derivs[0]
            .s_psi_penalty_components
            .as_ref()
            .expect("first derivatives");
        let got_indices: Vec<usize> = first.iter().map(|(idx, _)| *idx).collect();
        assert_eq!(got_indices, vec![0, 1]);
        assert_shape_penalty_component(&first[0].1, p_resp, &ds0);
        assert_shape_penalty_component(&first[1].1, p_resp, &ds1);

        let second = tensor_derivs[0]
            .s_psi_psi_penalty_components
            .as_ref()
            .expect("second derivatives");
        assert_eq!(second.len(), 1);
        let got_second_indices: Vec<usize> = second[0].iter().map(|(idx, _)| *idx).collect();
        assert_eq!(got_second_indices, vec![1]);
        assert_shape_penalty_component(&second[0][0].1, p_resp, &ds1_second);
    }

    #[test]
    fn tensor_psi_row_chunks_are_window_consistent() {
        let response = array![-1.0, -0.2, 0.6, 1.3];
        let (val_basis, deriv_basis, knots, transform, _) = toy_response_basis(&response);
        let psi = array![0.15, -0.10];
        let (cov_design, cov_derivs) = toy_covariate_design_and_derivs(&psi);
        let weights = Array1::from_elem(response.len(), 1.0);
        let offset = Array1::zeros(response.len());
        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            &response,
            val_basis,
            deriv_basis,
            vec![],
            knots,
            toy_scop_ctn_config().response_degree,
            transform,
            &weights,
            &offset,
            DesignMatrix::Dense(DenseDesignMatrix::from(cov_design)),
            vec![],
            &toy_scop_ctn_config(),
            None,
        )
        .expect("toy transformation family");

        let tensor_derivs =
            build_tensor_psi_derivatives(&family, &cov_derivs).expect("tensor derivatives");
        let op = tensor_derivs[0]
            .implicit_operator
            .as_ref()
            .expect("tensor psi operator should be implicit");
        let mat_op = op
            .as_materializable()
            .expect("toy tensor psi operator should remain materializable for reference");
        let rows = 1..3;

        let first_dense = mat_op
            .materialize_first(0)
            .expect("dense first derivative reference");
        let first_chunk = op
            .row_chunk_first(0, rows.clone())
            .expect("chunked first derivative");
        assert_eq!(
            first_chunk,
            first_dense.slice(s![rows.clone(), ..]).to_owned()
        );

        let second_diag_full = op
            .row_chunk_second_diag(0, 0..op.n_data())
            .expect("full row-chunk second diagonal reference");
        let second_diag_chunk = op
            .row_chunk_second_diag(0, rows.clone())
            .expect("chunked second diagonal derivative");
        assert_eq!(
            second_diag_chunk,
            second_diag_full.slice(s![rows.clone(), ..]).to_owned()
        );

        let second_cross_full = op
            .row_chunk_second_cross(0, 1, 0..op.n_data())
            .expect("full row-chunk second cross reference");
        let second_cross_chunk = op
            .row_chunk_second_cross(0, 1, rows.clone())
            .expect("chunked second cross derivative");
        assert_eq!(
            second_cross_chunk,
            second_cross_full.slice(s![rows, ..]).to_owned()
        );
    }

    fn assert_shape_penalty_component(
        penalty: &PenaltyMatrix,
        p_resp: usize,
        expected_right: &Array2<f64>,
    ) {
        let PenaltyMatrix::KroneckerFactored { left, right } = penalty else {
            panic!("expected KroneckerFactored penalty component");
        };
        assert_eq!(right, expected_right);
        assert_eq!(left.nrows(), p_resp);
        assert_eq!(left.ncols(), p_resp);
        for r in 0..p_resp {
            for c in 0..p_resp {
                let expected = if r == c && r > 0 { 1.0 } else { 0.0 };
                assert_eq!(left[[r, c]], expected);
            }
        }
    }

    fn toy_covariate_design_and_derivs(
        psi: &Array1<f64>,
    ) -> (Array2<f64>, Vec<CustomFamilyBlockPsiDerivative>) {
        let x0 = array![[1.00, 0.40], [1.10, 0.35], [1.20, 0.45], [0.95, 0.50],];
        let x_a = array![[0.10, -0.02], [0.08, 0.01], [0.12, -0.01], [0.09, 0.03],];
        let x_b = array![[-0.04, 0.06], [-0.02, 0.05], [-0.03, 0.04], [-0.01, 0.07],];
        let x_aa = array![[0.02, 0.00], [0.01, 0.01], [0.02, -0.01], [0.01, 0.02],];
        let x_ab = array![[0.01, -0.01], [0.00, 0.02], [0.01, 0.01], [0.00, -0.01],];
        let x_bb = array![[-0.01, 0.02], [-0.02, 0.01], [-0.01, 0.00], [-0.02, 0.02],];
        let design = &x0
            + &(x_a.clone() * psi[0])
            + &(x_b.clone() * psi[1])
            + &(x_aa.clone() * (0.5 * psi[0] * psi[0]))
            + &(x_ab.clone() * (psi[0] * psi[1]))
            + &(x_bb.clone() * (0.5 * psi[1] * psi[1]));
        let d_a = &x_a + &(x_aa.clone() * psi[0]) + &(x_ab.clone() * psi[1]);
        let d_b = &x_b + &(x_ab.clone() * psi[0]) + &(x_bb.clone() * psi[1]);
        let deriv_a = CustomFamilyBlockPsiDerivative::new(
            None,
            d_a,
            Array2::zeros((0, 0)),
            None,
            Some(vec![x_aa.clone(), x_ab.clone()]),
            None,
            None,
        );
        let deriv_b = CustomFamilyBlockPsiDerivative::new(
            None,
            d_b,
            Array2::zeros((0, 0)),
            None,
            Some(vec![x_ab, x_bb]),
            None,
            None,
        );
        (design, vec![deriv_a, deriv_b])
    }

    /// Minimal SCOP-CTN config used by every toy fixture in this test module:
    /// degree-1 I-splines on 2 internal knots produce the smallest valid
    /// SCOP-CTN configuration (p_resp = 4 monotone basis columns).
    fn toy_scop_ctn_config() -> TransformationNormalConfig {
        TransformationNormalConfig {
            double_penalty: false,
            response_degree: 1,
            response_num_internal_knots: 2,
            ..TransformationNormalConfig::default()
        }
    }

    /// Build (val, deriv, knots, transform, p_resp) from a real
    /// `build_response_basis` call so test fixtures match the production
    /// I-spline contract exactly.
    fn toy_response_basis(
        response: &Array1<f64>,
    ) -> (Array2<f64>, Array2<f64>, Array1<f64>, Array2<f64>, usize) {
        let config = toy_scop_ctn_config();
        let (val, deriv, _penalties, knots, transform) =
            build_response_basis(response, &config).expect("toy response basis builds");
        let p_resp = val.ncols();
        (val, deriv, knots, transform, p_resp)
    }

    /// Deterministic probe vector of length `p_total` used by tests that
    /// previously hand-rolled p_total=4 arrays. Generated from a tiny PRNG so
    /// each call with a different seed yields linearly-independent probes.
    fn toy_probe_vector(p_total: usize, seed: u64) -> Array1<f64> {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        Array1::from_iter((0..p_total).map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let bits = (state >> 11) as f64 / (1u64 << 53) as f64;
            (bits - 0.5) * 0.8
        }))
    }

    fn toy_family_and_derivatives(
        psi: &Array1<f64>,
    ) -> (
        TransformationNormalFamily,
        Vec<Vec<CustomFamilyBlockPsiDerivative>>,
        ParameterBlockState,
        ParameterBlockSpec,
    ) {
        let response = array![-1.0, -0.2, 0.6, 1.3];
        let (val_basis, deriv_basis, knots, transform, p_resp) = toy_response_basis(&response);
        let weights = Array1::from_elem(response.len(), 1.0);
        let offset = Array1::zeros(response.len());
        let (cov_design, cov_derivs) = toy_covariate_design_and_derivs(psi);
        let p_cov = cov_design.ncols();
        let p_total = p_resp * p_cov;
        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            &response,
            val_basis,
            deriv_basis,
            vec![],
            knots,
            toy_scop_ctn_config().response_degree,
            transform,
            &weights,
            &offset,
            DesignMatrix::Dense(DenseDesignMatrix::from(cov_design)),
            vec![],
            &toy_scop_ctn_config(),
            None,
        )
        .expect("toy transformation family");
        let derivative_blocks =
            vec![build_tensor_psi_derivatives(&family, &cov_derivs).expect("tensor psi derivs")];
        // Positive γ across the response axis with mild covariate variation so
        // h' = (M ⊗_row B_cov)·β stays strictly positive on every row (M-splines
        // are non-negative; the toy covariate design is positive-valued).
        let mut beta_vec = Vec::with_capacity(p_total);
        for k in 0..p_resp {
            let base = 0.6 + 0.05 * k as f64;
            for j in 0..p_cov {
                if j == 0 {
                    beta_vec.push(base);
                } else {
                    beta_vec.push(0.05 + 0.02 * k as f64 * (j as f64));
                }
            }
        }
        let beta = Array1::from(beta_vec);
        assert_eq!(beta.len(), p_total);
        let h_prime = family.x_deriv_kron.forward_mul(&beta);
        assert!(
            h_prime.iter().all(|v| *v > 0.25),
            "toy beta must keep h' positive, got {h_prime:?}"
        );
        let state = ParameterBlockState {
            beta,
            eta: Array1::zeros(h_prime.len()),
        };
        let spec = family.block_spec();
        (family, derivative_blocks, state, spec)
    }

    #[test]
    fn ctn_row_quantity_cache_matches_direct_formulas() {
        let psi = array![0.15, -0.10];
        let (family, _, state, _) = toy_family_and_derivatives(&psi);
        let row = family
            .row_quantities(&state.beta)
            .expect("toy row quantities");
        // SCOP-CTN forward: h = X_val · γ²-affine + offset + ε(y−median),
        // h' = X_deriv · γ²-affine + ε.
        let direct_h = family.x_val_kron.scop_affine_squared_forward(&state.beta)
            + family.offset.as_ref()
            + family.response_floor_offset.as_ref();
        let direct_h_prime = family
            .x_deriv_kron
            .scop_affine_squared_forward(&state.beta)
            .mapv(|hp| hp + TRANSFORMATION_MONOTONICITY_EPS);
        let weights = family.weights.as_ref();

        for i in 0..direct_h.len() {
            assert!(
                (row.h[i] - direct_h[i]).abs() <= 1.0e-14,
                "h[{i}] mismatch: cached={} direct={}",
                row.h[i],
                direct_h[i]
            );
            assert!(
                (row.h_prime[i] - direct_h_prime[i]).abs() <= 1.0e-14,
                "h_prime[{i}] mismatch: cached={} direct={}",
                row.h_prime[i],
                direct_h_prime[i]
            );
        }

        let p_resp = family.response_val_basis.ncols();
        let p_cov = family.covariate_design.ncols();
        let beta_mat = state
            .beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .expect("toy beta reshape");
        let cov = family
            .covariate_design
            .try_row_chunk(0..family.n_obs())
            .expect("toy covariate rows");
        let (h_lower, h_upper) = family
            .scop_endpoint_values(&state.beta, beta_mat, cov.view())
            .expect("toy endpoint values");

        let mut expected_ll = 0.0;
        for i in 0..direct_h.len() {
            assert!(
                (row.h_lower[i] - h_lower[i]).abs() <= 1.0e-14,
                "h_lower[{i}] mismatch: cached={} direct={}",
                row.h_lower[i],
                h_lower[i]
            );
            assert!(
                (row.h_upper[i] - h_upper[i]).abs() <= 1.0e-14,
                "h_upper[{i}] mismatch: cached={} direct={}",
                row.h_upper[i],
                h_upper[i]
            );
            let hp = direct_h_prime[i];
            let log_z = log_normal_cdf_diff(h_upper[i], h_lower[i]).expect("endpoint mass");
            expected_ll += weights[i] * (-0.5 * direct_h[i] * direct_h[i] + hp.ln() - log_z);
        }

        assert!(
            (row.log_likelihood - expected_ll).abs() <= 1.0e-14,
            "cached log-likelihood={} expected={expected_ll}",
            row.log_likelihood
        );
    }

    #[test]
    fn ctn_endpoint_normalizer_derivatives_are_finite_in_positive_tail() {
        let q =
            log_normal_cdf_diff_derivatives(38.0, 37.0).expect("positive-tail endpoint normalizer");
        assert!(q.first[0].is_finite());
        assert!(q.first[1].is_finite());
        assert!(q.second[0][0].is_finite());
        assert!(q.third[0][0][0].is_finite());
        assert!(q.fourth[0][0][0][0].is_finite());
        assert!(q.first[0] > 0.0);
        assert!(q.first[1] < 0.0);
    }

    #[test]
    fn transformation_normal_pit_score_uses_finite_support_normalizer() {
        let center =
            transformation_normal_pit_score(0.0, -2.0, 2.0, 1.0e-12).expect("symmetric PIT score");
        assert!(center.abs() <= 1.0e-12);

        let positive_tail = transformation_normal_pit_score(37.5, 37.0, 38.0, 1.0e-12)
            .expect("positive-tail PIT score");
        assert!(positive_tail.is_finite());

        // Extrapolation past the upper endpoint is *not* an error: the PIT
        // mapping clamps `h` to `[lower, upper]` so `u → 1`, and the
        // `clip_eps` clamp on the standard-normal quantile call yields the
        // upper-tail extreme finite value (`≈ Φ⁻¹(1 - clip_eps)`). At
        // biobank shape, an honest test sample at-or-just-beyond the
        // training response support routinely lands here from boundary
        // roundoff alone, so failing closed would ship a hard prediction
        // error on every CTN bootstrap pass.
        let above_upper = transformation_normal_pit_score(2.1, -2.0, 2.0, 1.0e-12)
            .expect("extrapolation above upper endpoint should clamp, not error");
        assert!(above_upper.is_finite());
        assert!(above_upper > 0.0, "h>upper must produce upper-tail PIT");
        let below_lower = transformation_normal_pit_score(-2.1, -2.0, 2.0, 1.0e-12)
            .expect("extrapolation below lower endpoint should clamp, not error");
        assert!(below_lower.is_finite());
        assert!(below_lower < 0.0, "h<lower must produce lower-tail PIT");

        // Genuinely-malformed input (NaN h) must still be rejected by the
        // early `is_finite()` guard — the soft-clamp is for legitimate
        // numerical extrapolation, not for non-finite values.
        let nan_err = transformation_normal_pit_score(f64::NAN, -2.0, 2.0, 1.0e-12)
            .expect_err("NaN h must still be rejected");
        assert!(nan_err.contains("finite"));
    }

    #[test]
    fn ctn_row_quantity_cache_is_exact_beta_keyed() {
        let psi = array![0.15, -0.10];
        let (family, _, state, _) = toy_family_and_derivatives(&psi);
        let row_a = family
            .row_quantities(&state.beta)
            .expect("first row quantity build");
        let row_a_again = family
            .row_quantities(&state.beta)
            .expect("same beta row quantity lookup");
        assert!(Arc::ptr_eq(&row_a.h, &row_a_again.h));
        assert!(Arc::ptr_eq(&row_a.h_prime, &row_a_again.h_prime));

        let mut beta_b = state.beta.clone();
        beta_b[0] += 0.125;
        let row_b = family
            .row_quantities(&beta_b)
            .expect("updated beta row quantity build");
        assert!(!Arc::ptr_eq(&row_a.h, &row_b.h));
        assert!(row_b.matches_beta(&beta_b));
        assert!(!row_b.matches_beta(&state.beta));
        assert!(
            row_a
                .h
                .iter()
                .zip(row_b.h.iter())
                .any(|(&left, &right)| left.to_bits() != right.to_bits())
        );

        let row_b_again = family
            .row_quantities(&beta_b)
            .expect("updated beta row quantity lookup");
        assert!(Arc::ptr_eq(&row_b.h, &row_b_again.h));
    }

    #[test]
    fn ctn_row_quantities_reject_nonrepresentable_exact_derivatives() {
        let h = array![0.0];
        let h_prime = array![1.0e-100];
        let h_lower = array![-8.0];
        let h_upper = array![8.0];
        let weights = array![1.0];
        let err = build_transformation_row_derived(&h, &h_prime, &h_lower, &h_upper, &weights)
            .expect_err("1/h'^4 overflows f64 and must not be clamped");
        assert!(
            err.contains("1/h'^4") && err.contains("outside the finite exact-derivative range"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn transformation_normal_uses_compact_gaussian_outer_seeding() {
        let psi = array![0.15, -0.10];
        let (family, _, _, _) = toy_family_and_derivatives(&psi);
        let seed_config = family.outer_seed_config(6);
        assert_eq!(seed_config.bounds, (-12.0, 12.0));
        assert_eq!(seed_config.max_seeds, 1);
        assert_eq!(seed_config.seed_budget, 1);
        assert_eq!(seed_config.screen_max_inner_iterations, 2);
        assert_eq!(
            seed_config.risk_profile,
            crate::seeding::SeedRiskProfile::Gaussian
        );
        assert_eq!(seed_config.num_auxiliary_trailing, 0);
    }

    #[test]
    fn max_feasible_step_size_is_unconstrained_for_scop_derivative() {
        let psi = array![0.15, -0.10];
        let (family, _, state, _) = toy_family_and_derivatives(&psi);
        let p_total = state.beta.len();
        let mut delta = toy_probe_vector(p_total, 0xDE17A);
        delta[0] = -0.30;

        let block_states = vec![state.clone()];
        let alpha_prod = family
            .max_feasible_step_size(&block_states, 0, &delta)
            .expect("toy max_feasible_step_size returns Ok");
        assert_eq!(alpha_prod, None);

        let bad_delta = Array1::<f64>::zeros(p_total + 1);
        assert!(
            family
                .max_feasible_step_size(&block_states, 0, &bad_delta)
                .is_err(),
            "dimension mismatches should still be rejected before line search"
        );
    }

    #[test]
    fn warm_start_absorbs_offset_into_affine_seed() {
        // The SCOP squared-γ warm start is built directly in β-space: choose a
        // positive constant shape seed for h', subtract its induced value
        // contribution, then solve the unconstrained location row. The fixed
        // monotonicity floor is part of h, so the value target includes
        // ε(y-median) and the derivative target includes ε.
        let response = array![2.0, 3.0, 4.0, 5.0];
        let (val_basis, deriv_basis, knots, transform, _p_resp) = toy_response_basis(&response);
        let weights = Array1::from_elem(response.len(), 1.0);
        let offset = Array1::from_elem(response.len(), 0.7);
        let cov_rows = response.len();
        let covariate_design = DesignMatrix::Dense(DenseDesignMatrix::from(Array2::from_elem(
            (cov_rows, 1),
            1.0,
        )));
        let warm_start = TransformationWarmStart {
            location: Array1::from_elem(response.len(), 1.0),
            scale: Array1::from_elem(response.len(), 2.0),
        };
        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            &response,
            val_basis,
            deriv_basis,
            vec![],
            knots,
            toy_scop_ctn_config().response_degree,
            transform,
            &weights,
            &offset,
            covariate_design,
            vec![],
            &toy_scop_ctn_config(),
            Some(&warm_start),
        )
        .expect("transformation family");

        let row = family
            .row_quantities(&family.initial_beta)
            .expect("row quantities at initial beta");
        let h = row.h.as_ref();
        let h_prime = row.h_prime.as_ref();
        // expected_h[i] = (response[i] - location)/scale = (y - 1)/2.
        let expected_h: Array1<f64> = response.mapv(|y| {
            (y - 1.0) / 2.0 + TRANSFORMATION_MONOTONICITY_EPS * (y - family.response_median())
        });
        let expected_h_prime =
            Array1::from_elem(response.len(), 0.5 + TRANSFORMATION_MONOTONICITY_EPS);

        for i in 0..expected_h.len() {
            assert!(
                (h[i] - expected_h[i]).abs() < 1e-9,
                "h[{i}] mismatch: got {}, expected {}",
                h[i],
                expected_h[i]
            );
            assert!(
                (h_prime[i] - expected_h_prime[i]).abs() < 1e-9,
                "h_prime[{i}] mismatch: got {}, expected {}",
                h_prime[i],
                expected_h_prime[i]
            );
        }

        assert_eq!(response.len(), family.n_obs());
    }

    #[test]
    fn kronecker_dense_fast_paths_match_dense_materialization() {
        let left = array![[1.0, -0.4], [0.5, 0.3], [-0.2, 0.9], [1.1, -0.7],];
        let right = array![
            [0.2, 1.0, -0.3],
            [0.4, -0.5, 0.8],
            [0.7, 0.1, 0.6],
            [-0.2, 0.9, 0.5],
        ];
        let weights = array![0.7, 1.4, 0.9, 1.2];
        let v = array![0.6, -0.3, 0.5, 0.8];
        let kron = KroneckerDesign::new_khatri_rao(
            &left,
            DesignMatrix::Dense(DenseDesignMatrix::from(right.clone())),
        )
        .expect("kronecker design");

        let dense = rowwise_kronecker(&left, &right);
        let expected_transpose = dense.t().dot(&v);
        let expected_gram = fast_atb(&weight_rows(&dense, &weights), &dense);

        let got_transpose = kron.transpose_mul(&v);
        let got_gram = kron.weighted_gram(&weights, &ResourcePolicy::default_library());

        let transpose_err = (&got_transpose - &expected_transpose)
            .iter()
            .fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        let gram_err = (&got_gram - &expected_gram)
            .iter()
            .fold(0.0_f64, |acc, &x| acc.max(x.abs()));
        assert!(
            transpose_err < 1e-10,
            "Kronecker transpose fast path mismatch: max_abs={transpose_err}"
        );
        assert!(
            gram_err < 1e-10,
            "Kronecker weighted Gram fast path mismatch: max_abs={gram_err}"
        );
    }

    #[test]
    fn large_samples_allow_richer_response_basis_than_small_samples() {
        let config = TransformationNormalConfig::default();
        let small = effective_response_num_internal_knots(&config, 40, 20);
        let large = effective_response_num_internal_knots(&config, 4000, 20);
        assert!(large >= small);
        assert!(
            large > small,
            "large-sample tensor cap should relax the small-sample response bottleneck"
        );
    }

    #[test]
    fn transformation_normal_joint_psi_second_order_terms_match_fd() {
        let psi = array![0.15, -0.10];
        let h = 1e-6;
        let row_offset = Arc::new(array![0.70, -0.20, 0.40, -0.50]);
        let (mut family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
        family.offset = Arc::clone(&row_offset);
        let states = vec![state.clone()];
        let specs = vec![spec];

        let analytic = family
            .exact_newton_joint_psisecond_order_terms(&states, &specs, &derivative_blocks, 0, 1)
            .expect("analytic psi second-order terms")
            .expect("psi second-order terms should be present");

        let eval_first = |psi_eval: &Array1<f64>| {
            let (mut f_eval, deriv_eval, state_eval, spec_eval) =
                toy_family_and_derivatives(psi_eval);
            f_eval.offset = Arc::clone(&row_offset);
            let states_eval = vec![state_eval];
            let specs_eval = vec![spec_eval];
            f_eval
                .exact_newton_joint_psi_terms(&states_eval, &specs_eval, &deriv_eval, 0)
                .expect("first-order psi terms")
                .expect("first-order terms should be present")
        };

        let mut psi_plus = psi.clone();
        psi_plus[1] += h;
        let plus = eval_first(&psi_plus);
        let mut psi_minus = psi.clone();
        psi_minus[1] -= h;
        let minus = eval_first(&psi_minus);

        let objective_fd = (plus.objective_psi - minus.objective_psi) / (2.0 * h);
        assert!(
            (analytic.objective_psi_psi - objective_fd).abs() < 1e-5,
            "objective psi second-order mismatch: analytic={}, fd={objective_fd}",
            analytic.objective_psi_psi
        );

        let score_fd = (&plus.score_psi - &minus.score_psi) / (2.0 * h);
        for idx in 0..score_fd.len() {
            assert!(
                (analytic.score_psi_psi[idx] - score_fd[idx]).abs() < 1e-5,
                "score psi second-order mismatch at {idx}: analytic={}, fd={}",
                analytic.score_psi_psi[idx],
                score_fd[idx]
            );
        }

        let hess_fd = (dense_first_order_psi_hessian(&plus)
            - dense_first_order_psi_hessian(&minus))
            / (2.0 * h);
        // The CTN psi-psi second-order kernel exposes its dense p_total×p_total
        // block through `hessian_psi_psi` when the family materializes it
        // eagerly, or through an operator-backed `hessian_psi_psi_operator`
        // when the family stages the Hessian as HVPs. The FD comparison needs
        // the dense matrix either way, so materialize the operator on demand.
        let analytic_hessian = if analytic.hessian_psi_psi.nrows() > 0 {
            analytic.hessian_psi_psi.clone()
        } else {
            analytic
                .hessian_psi_psi_operator
                .as_ref()
                .expect("CTN psi-psi must expose either dense Hessian or operator")
                .to_dense()
        };
        assert_matrix_derivativefd(
            &hess_fd,
            &analytic_hessian,
            2e-4,
            "transformation normal psi second-order Hessian",
        );
    }

    #[test]
    fn transformation_normal_joint_psi_first_order_matches_normalized_loglik_fd() {
        let psi = array![0.15, -0.10];
        let h = 1e-6;
        let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
        let beta = state.beta.clone();
        let states = vec![state.clone()];
        let specs = vec![spec];

        let analytic = family
            .exact_newton_joint_psi_terms(&states, &specs, &derivative_blocks, 0)
            .expect("analytic psi first-order terms")
            .expect("first-order terms should be present");

        let eval_negative_loglik = |psi_eval: &Array1<f64>| {
            let (f_eval, _, mut state_eval, _) = toy_family_and_derivatives(psi_eval);
            state_eval.beta = beta.clone();
            -f_eval
                .log_likelihood_only(std::slice::from_ref(&state_eval))
                .expect("log-likelihood at perturbed psi")
        };

        let mut psi_plus = psi.clone();
        psi_plus[0] += h;
        let mut psi_minus = psi.clone();
        psi_minus[0] -= h;
        let fd = (eval_negative_loglik(&psi_plus) - eval_negative_loglik(&psi_minus)) / (2.0 * h);

        assert!(
            (analytic.objective_psi - fd).abs() < 1e-6,
            "normalized CTN psi objective mismatch: analytic={}, fd={fd}",
            analytic.objective_psi
        );

        assert_eq!(analytic.hessian_psi.nrows(), 0);
        assert_eq!(analytic.hessian_psi.ncols(), 0);
        let op = analytic
            .hessian_psi_operator
            .as_ref()
            .expect("CTN psi first-order Hessian must be operator-backed");
        assert_eq!(op.dim(), beta.len());

        let direction = toy_probe_vector(beta.len(), 407);
        let h_beta = 1e-6;
        let eval_score = |beta_eval: &Array1<f64>| {
            let mut state_eval = state.clone();
            state_eval.beta = beta_eval.clone();
            family
                .exact_newton_joint_psi_terms(
                    std::slice::from_ref(&state_eval),
                    &specs,
                    &derivative_blocks,
                    0,
                )
                .expect("first-order psi terms at shifted beta")
                .expect("shifted first-order terms should be present")
                .score_psi
        };
        let beta_plus = &beta + &(direction.clone() * h_beta);
        let beta_minus = &beta - &(direction.clone() * h_beta);
        let score_fd = (eval_score(&beta_plus) - eval_score(&beta_minus)) / (2.0 * h_beta);
        let hvp = op.mul_vec(&direction);
        for idx in 0..hvp.len() {
            let tol = 2e-5 * score_fd[idx].abs().max(1.0);
            assert!(
                (hvp[idx] - score_fd[idx]).abs() <= tol,
                "first-order psi Hessian operator mismatch at {idx}: analytic={:.6e}, fd={:.6e}",
                hvp[idx],
                score_fd[idx]
            );
        }

        let mut factor = Array2::<f64>::zeros((beta.len(), 4));
        for (col, seed) in [408_u64, 409, 410, 411].into_iter().enumerate() {
            factor
                .column_mut(col)
                .assign(&toy_probe_vector(beta.len(), seed));
        }
        let got_mat = op.mul_mat(&factor);
        for col in 0..factor.ncols() {
            let want_col = op.mul_vec(&factor.column(col).to_owned());
            for row in 0..beta.len() {
                let tol = 1.0e-11 * want_col[row].abs().max(1.0) + 1.0e-11;
                assert!(
                    (got_mat[[row, col]] - want_col[row]).abs() <= tol,
                    "first-order psi Hessian batched mul_mat mismatch at ({row}, {col}): got={:.6e}, want={:.6e}",
                    got_mat[[row, col]],
                    want_col[row],
                );
            }
        }
        let got_trace = op.trace_projected_factor(&factor);
        let want_trace = factor
            .iter()
            .zip(got_mat.iter())
            .map(|(&f, &bf)| f * bf)
            .sum::<f64>();
        let tol = 1.0e-11 * want_trace.abs().max(1.0) + 1.0e-11;
        assert!(
            (got_trace - want_trace).abs() <= tol,
            "first-order psi Hessian projected trace mismatch: got={:.6e}, want={:.6e}",
            got_trace,
            want_trace,
        );
    }

    #[test]
    fn ctn_psi_workspace_first_order_matches_per_axis_path_bit_equivalent() {
        // Bit-equivalence guard for `TransformationNormalPsiWorkspace`. The
        // workspace's single-pass kernel must produce the same per-axis
        // `objective_psi` and `score_psi` as the per-axis `scop_psi_terms`
        // path that the previous CTN code path used. We compare across every
        // ψ axis at once — there is no axis whose accumulated state can
        // mask a bug in another axis.
        let psi = array![0.15, -0.10];
        let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
        let states = vec![state.clone()];
        let specs = vec![spec];
        let n_psi = derivative_blocks[0].len();
        assert!(
            n_psi >= 2,
            "toy CTN fixture must expose at least 2 ψ axes for the workspace check, got {n_psi}"
        );

        // Per-axis ground truth via the existing direct hook.
        let mut per_axis: Vec<ExactNewtonJointPsiTerms> = Vec::with_capacity(n_psi);
        for psi_index in 0..n_psi {
            per_axis.push(
                family
                    .exact_newton_joint_psi_terms(&states, &specs, &derivative_blocks, psi_index)
                    .expect("per-axis ψ terms")
                    .expect("per-axis ψ terms must be present"),
            );
        }

        // All-axes pass via the workspace.
        let workspace = family
            .exact_newton_joint_psi_workspace(&states, &specs, &derivative_blocks)
            .expect("CTN ψ workspace constructor")
            .expect("CTN ψ workspace must be present");
        let mut shared_factor = Array2::<f64>::zeros((state.beta.len(), 3));
        for (col, seed) in [70_001_u64, 80_001_u64, 90_001_u64].into_iter().enumerate() {
            shared_factor
                .column_mut(col)
                .assign(&toy_probe_vector(state.beta.len(), seed));
        }
        let projected_cache = ProjectedFactorCache::default();

        for psi_index in 0..n_psi {
            let cached = workspace
                .first_order_terms(psi_index)
                .expect("workspace first-order terms")
                .expect("workspace first-order terms must be present");
            let expected = &per_axis[psi_index];

            // Objective: the workspace fold is order-permutation-equivalent
            // to the per-axis fold; allow a tiny floating-point slack on top
            // of bit equality so reductions over different chunk shapes
            // (rayon's deterministic-order fold groups rows differently than
            // the serial loop) do not flake the test.
            let obj_diff = (cached.objective_psi - expected.objective_psi).abs();
            let obj_scale = expected.objective_psi.abs().max(1.0);
            assert!(
                obj_diff <= 1.0e-12 * obj_scale,
                "ψ workspace objective_psi[axis={psi_index}] mismatch: cached={}, per-axis={}, |diff|={obj_diff}",
                cached.objective_psi,
                expected.objective_psi,
            );

            assert_eq!(
                cached.score_psi.len(),
                expected.score_psi.len(),
                "ψ workspace score_psi length mismatch at axis {psi_index}"
            );
            for idx in 0..expected.score_psi.len() {
                let diff = (cached.score_psi[idx] - expected.score_psi[idx]).abs();
                let scale = expected.score_psi[idx].abs().max(1.0);
                assert!(
                    diff <= 1.0e-12 * scale,
                    "ψ workspace score_psi[axis={psi_index}, idx={idx}] mismatch: cached={}, per-axis={}, |diff|={diff}",
                    cached.score_psi[idx],
                    expected.score_psi[idx],
                );
            }

            // The per-axis matrix-free Hessian operator must remain present
            // and dimension-matching; we do not compare its action here
            // because the operator is constructed directly from the same
            // `row_quantities` cache the per-axis path uses.
            let cached_op = cached
                .hessian_psi_operator
                .as_ref()
                .expect("workspace ψ Hessian operator must be present");
            assert_eq!(cached_op.dim(), state.beta.len());
            let cached_trace =
                cached_op.trace_projected_factor_cached(&shared_factor, &projected_cache);
            let direct_trace = cached_op.trace_projected_factor(&shared_factor);
            let trace_tol = 1.0e-10 * direct_trace.abs().max(1.0) + 1.0e-10;
            assert!(
                (cached_trace - direct_trace).abs() <= trace_tol,
                "workspace ψ cached projected trace mismatch at axis {psi_index}: cached={cached_trace:.6e}, direct={direct_trace:.6e}",
            );
        }
    }

    /// Direct kernel-level bit-equivalence guard for the fused all-axes
    /// projected-trace path (Fix #2). Compares
    /// [`scop_psi_hessian_trace_factor_all_axes_chunk_from_cov`] called once
    /// with every ψ axis's `cov_psi` against
    /// [`scop_psi_hessian_trace_factor_from_cov`] called once per axis. Both
    /// kernels accumulate over the same rows in the same parallel rayon
    /// reduction tree, so the fused output must equal the per-axis output to
    /// well within any reasonable floating-point reduction tolerance.
    #[test]
    fn ctn_psi_hessian_trace_all_axes_matches_per_axis_path_bit_equivalent() {
        let psi = array![0.15, -0.10];
        let (family, derivative_blocks, state, _spec) = toy_family_and_derivatives(&psi);
        let n_psi = derivative_blocks[0].len();
        assert!(
            n_psi >= 2,
            "toy CTN fixture must expose at least 2 ψ axes for the fused trace check, got {n_psi}"
        );

        let row_quantities = family
            .row_quantities(&state.beta)
            .expect("toy CTN row quantities");

        // Build a non-trivial dense factor so every block of the ψ-Hessian
        // contributes to the projected trace. Three columns exercise both the
        // diagonal and off-diagonal Kronecker structure.
        let p_total = state.beta.len();
        let rank = 3;
        let mut factor = Array2::<f64>::zeros((p_total, rank));
        for col in 0..rank {
            let seed = 17_001_u64.wrapping_add(col as u64 * 13_337);
            factor
                .column_mut(col)
                .assign(&toy_probe_vector(p_total, seed));
        }

        // Materialise covariate and per-axis cov_psi over the full row range.
        let cov_arc = family
            .covariate_dense_arc()
            .expect("toy CTN covariate dense");
        let cov: &Array2<f64> = cov_arc.as_ref();
        let block_derivs = &derivative_blocks[0];
        let op_arc = block_derivs[0]
            .implicit_operator
            .as_ref()
            .expect("toy CTN ψ operator")
            .clone();
        let op = op_arc
            .as_any()
            .downcast_ref::<TensorKroneckerPsiOperator>()
            .expect("toy CTN ψ operator must be tensor-backed");
        let mut cov_psi_arrays: Vec<Array2<f64>> = Vec::with_capacity(n_psi);
        for deriv in block_derivs.iter() {
            cov_psi_arrays.push(
                op.materialize_cov_first_axis(deriv.implicit_axis)
                    .expect("toy CTN ψ cov derivative materialise"),
            );
        }
        let cov_psi_views: Vec<ArrayView2<'_, f64>> =
            cov_psi_arrays.iter().map(|m| m.view()).collect();

        // Per-axis ground truth: invoke the legacy single-axis kernel n_psi
        // times.
        let per_axis_traces: Vec<f64> = (0..n_psi)
            .map(|axis_idx| {
                family
                    .scop_psi_hessian_trace_factor_from_cov(
                        &state.beta,
                        &row_quantities,
                        block_derivs[axis_idx].implicit_axis,
                        cov,
                        &cov_psi_arrays[axis_idx],
                        factor.view(),
                    )
                    .expect("per-axis ψ projected trace")
            })
            .collect();

        // Fused all-axes pass: a single row-streaming traversal across every
        // axis. Calling the chunked kernel with `row_start=0` and the full-n
        // covariate views is equivalent to streaming a single chunk that
        // covers the entire dataset.
        let fused_traces = family
            .scop_psi_hessian_trace_factor_all_axes_chunk_from_cov(
                &state.beta,
                &row_quantities,
                0,
                cov.view(),
                &cov_psi_views,
                factor.view(),
            )
            .expect("fused all-axes ψ projected trace");

        assert_eq!(
            per_axis_traces.len(),
            fused_traces.len(),
            "per-axis vs fused trace length mismatch"
        );
        for (axis_idx, (&per_axis, &fused)) in
            per_axis_traces.iter().zip(fused_traces.iter()).enumerate()
        {
            let scale = per_axis.abs().max(fused.abs()).max(1.0);
            let abs_diff = (per_axis - fused).abs();
            let rel_diff = abs_diff / scale;
            assert!(
                rel_diff < 1.0e-12,
                "axis {axis_idx}: per-axis kernel = {per_axis:.6e}, fused kernel = {fused:.6e}, |Δ| = {abs_diff:.3e}, rel = {rel_diff:.3e}"
            );
        }
    }

    #[test]
    fn ctn_psi_workspace_second_order_matches_per_pair_path() {
        let psi = array![0.15, -0.10];
        let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
        let states = vec![state.clone()];
        let specs = vec![spec];
        let n_psi = derivative_blocks[0].len();

        let workspace = family
            .exact_newton_joint_psi_workspace(&states, &specs, &derivative_blocks)
            .expect("CTN ψ workspace constructor")
            .expect("CTN ψ workspace must be present");
        let mut shared_factor = Array2::<f64>::zeros((state.beta.len(), 3));
        for (col, seed) in [10_001_u64, 20_001_u64, 30_001_u64].into_iter().enumerate() {
            shared_factor
                .column_mut(col)
                .assign(&toy_probe_vector(state.beta.len(), seed));
        }
        let projected_cache = ProjectedFactorCache::default();

        for psi_i in 0..n_psi {
            for psi_j in psi_i..n_psi {
                let direct = family
                    .exact_newton_joint_psisecond_order_terms(
                        &states,
                        &specs,
                        &derivative_blocks,
                        psi_i,
                        psi_j,
                    )
                    .expect("direct ψ-ψ terms")
                    .expect("direct ψ-ψ terms must be present");
                let cached = workspace
                    .second_order_terms(psi_i, psi_j)
                    .expect("workspace ψ-ψ terms")
                    .expect("workspace ψ-ψ terms must be present");

                let obj_diff = (cached.objective_psi_psi - direct.objective_psi_psi).abs();
                let obj_scale = direct.objective_psi_psi.abs().max(1.0);
                assert!(
                    obj_diff <= 1.0e-12 * obj_scale,
                    "ψ workspace objective_psi_psi[{psi_i},{psi_j}] mismatch: cached={}, direct={}, |diff|={obj_diff}",
                    cached.objective_psi_psi,
                    direct.objective_psi_psi,
                );

                assert_eq!(
                    cached.score_psi_psi.len(),
                    direct.score_psi_psi.len(),
                    "ψ workspace score_psi_psi length mismatch at pair ({psi_i},{psi_j})"
                );
                for idx in 0..direct.score_psi_psi.len() {
                    let diff = (cached.score_psi_psi[idx] - direct.score_psi_psi[idx]).abs();
                    let scale = direct.score_psi_psi[idx].abs().max(1.0);
                    assert!(
                        diff <= 1.0e-12 * scale,
                        "ψ workspace score_psi_psi[pair=({psi_i},{psi_j}), idx={idx}] mismatch: cached={}, direct={}, |diff|={diff}",
                        cached.score_psi_psi[idx],
                        direct.score_psi_psi[idx],
                    );
                }

                let cached_op = cached
                    .hessian_psi_psi_operator
                    .as_ref()
                    .expect("workspace ψ-ψ Hessian operator must be present");
                let direct_op = direct
                    .hessian_psi_psi_operator
                    .as_ref()
                    .expect("direct ψ-ψ Hessian operator must be present");
                assert_eq!(cached_op.dim(), direct_op.dim());
                assert_eq!(cached_op.dim(), state.beta.len());

                let cached_trace =
                    cached_op.trace_projected_factor_cached(&shared_factor, &projected_cache);
                let direct_trace = cached_op.trace_projected_factor(&shared_factor);
                let trace_tol = 1.0e-10 * direct_trace.abs().max(1.0) + 1.0e-10;
                assert!(
                    (cached_trace - direct_trace).abs() <= trace_tol,
                    "workspace ψ-ψ cached projected trace mismatch at pair ({psi_i},{psi_j}): cached={cached_trace:.6e}, direct={direct_trace:.6e}",
                );
            }
        }
    }

    #[test]
    fn transformation_normal_joint_psi_second_order_terms_are_operator_backed() {
        let psi = array![0.15, -0.10];
        let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
        let states = vec![state.clone()];
        let specs = vec![spec];

        let terms = family
            .exact_newton_joint_psisecond_order_terms(&states, &specs, &derivative_blocks, 0, 1)
            .expect("analytic psi second-order terms")
            .expect("psi second-order terms should be present");

        assert_eq!(terms.hessian_psi_psi.nrows(), 0);
        assert_eq!(terms.hessian_psi_psi.ncols(), 0);
        let op = terms
            .hessian_psi_psi_operator
            .as_ref()
            .expect("CTN psi-psi Hessian must be operator-backed");
        assert!(op.is_implicit());
        let p = state.beta.len();
        assert_eq!(op.dim(), p);
        assert!(op.has_fast_bilinear_view());

        let dense = op.to_dense();
        assert_eq!(dense.nrows(), p);
        assert_eq!(dense.ncols(), p);

        let v = toy_probe_vector(p, 901);
        let got_vec = op.mul_vec(&v);
        let want_vec = dense.dot(&v);
        for i in 0..p {
            let tol = 1e-10 * want_vec[i].abs().max(1.0) + 1e-10;
            assert!(
                (got_vec[i] - want_vec[i]).abs() <= tol,
                "psi-psi operator matvec mismatch at {i}: got={:.6e}, want={:.6e}",
                got_vec[i],
                want_vec[i]
            );
        }

        let mut factor = Array2::<f64>::zeros((p, 3));
        for (col, seed) in [902_u64, 903, 904].into_iter().enumerate() {
            factor.column_mut(col).assign(&toy_probe_vector(p, seed));
        }
        let got_mat = op.mul_mat(&factor);
        let want_mat = dense.dot(&factor);
        for row in 0..p {
            for col in 0..factor.ncols() {
                let tol = 1e-10 * want_mat[[row, col]].abs().max(1.0) + 1e-10;
                assert!(
                    (got_mat[[row, col]] - want_mat[[row, col]]).abs() <= tol,
                    "psi-psi operator mul_mat mismatch at ({row}, {col}): got={:.6e}, want={:.6e}",
                    got_mat[[row, col]],
                    want_mat[[row, col]]
                );
            }
        }

        let left = toy_probe_vector(p, 905);
        let right = toy_probe_vector(p, 906);
        let got_bilinear = op.bilinear_view(left.view(), right.view());
        let want_bilinear = right.dot(&dense.dot(&left));
        let tol = 1e-10 * want_bilinear.abs().max(1.0) + 1e-10;
        assert!(
            (got_bilinear - want_bilinear).abs() <= tol,
            "psi-psi operator bilinear mismatch: got={:.6e}, want={:.6e}",
            got_bilinear,
            want_bilinear
        );

        let got_trace = op.trace_projected_factor(&factor);
        let want_trace = factor
            .iter()
            .zip(want_mat.iter())
            .map(|(&f, &bf)| f * bf)
            .sum::<f64>();
        let tol = 1e-10 * want_trace.abs().max(1.0) + 1e-10;
        assert!(
            (got_trace - want_trace).abs() <= tol,
            "psi-psi operator projected trace mismatch: got={:.6e}, want={:.6e}",
            got_trace,
            want_trace
        );
    }

    #[test]
    fn transformation_normal_joint_psihessian_directional_derivative_matches_fd() {
        let psi = array![0.15, -0.10];
        let h = 1e-6;
        let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
        let direction = toy_probe_vector(spec.design.ncols(), 701);
        let specs = vec![spec];

        let analytic = family
            .exact_newton_joint_psihessian_directional_derivative(
                std::slice::from_ref(&state),
                &specs,
                &derivative_blocks,
                0,
                &direction,
            )
            .expect("analytic psi hessian directional derivative")
            .expect("psi hessian directional derivative should be present");

        let eval_hess = |beta: &Array1<f64>| {
            let mut shifted_state = state.clone();
            shifted_state.beta = beta.clone();
            let terms = family
                .exact_newton_joint_psi_terms(
                    std::slice::from_ref(&shifted_state),
                    &specs,
                    &derivative_blocks,
                    0,
                )
                .expect("first-order psi terms at shifted beta")
                .expect("shifted first-order terms should be present");
            dense_first_order_psi_hessian(&terms)
        };

        let beta_plus = &state.beta + &(direction.clone() * h);
        let beta_minus = &state.beta - &(direction.clone() * h);
        let fd = (eval_hess(&beta_plus) - eval_hess(&beta_minus)) / (2.0 * h);
        assert_matrix_derivativefd(
            &fd,
            &analytic,
            2e-4,
            "transformation normal psi hessian directional derivative",
        );

        let workspace = family
            .exact_newton_joint_psi_workspace(&[state.clone()], &specs, &derivative_blocks)
            .expect("CTN psi workspace constructor")
            .expect("CTN psi workspace must be present");
        let drift_op = workspace
            .hessian_directional_derivative(0, &direction)
            .expect("workspace psi dH operator")
            .expect("workspace psi dH operator must be present");
        let DriftDerivResult::Operator(drift_op) = drift_op else {
            panic!("CTN workspace psi dH must be operator-backed");
        };
        let probe = toy_probe_vector(state.beta.len(), 90_001_u64);
        let got_vec = drift_op.mul_vec(&probe);
        let want_vec = analytic.dot(&probe);
        for i in 0..state.beta.len() {
            let vec_tol = 1.0e-10 * want_vec[i].abs().max(1.0) + 1.0e-10;
            assert!(
                (got_vec[i] - want_vec[i]).abs() <= vec_tol,
                "workspace psi dH matvec mismatch at {i}: got={:.6e}, want={:.6e}",
                got_vec[i],
                want_vec[i],
            );
        }
        let mut factor = Array2::<f64>::zeros((state.beta.len(), 3));
        for (col, seed) in [91_001_u64, 92_001_u64, 93_001_u64].into_iter().enumerate() {
            factor
                .column_mut(col)
                .assign(&toy_probe_vector(state.beta.len(), seed));
        }
        let got_mat = drift_op.mul_mat(&factor);
        let want_mat = analytic.dot(&factor);
        for row in 0..state.beta.len() {
            for col in 0..factor.ncols() {
                let mat_tol = 1.0e-10 * want_mat[[row, col]].abs().max(1.0) + 1.0e-10;
                assert!(
                    (got_mat[[row, col]] - want_mat[[row, col]]).abs() <= mat_tol,
                    "workspace psi dH matmat mismatch at ({row}, {col}): got={:.6e}, want={:.6e}",
                    got_mat[[row, col]],
                    want_mat[[row, col]],
                );
            }
        }
        let got_trace = drift_op.trace_projected_factor(&factor);
        let want_trace = factor
            .iter()
            .zip(want_mat.iter())
            .map(|(&f, &bf)| f * bf)
            .sum::<f64>();
        let trace_tol = 1.0e-10 * want_trace.abs().max(1.0) + 1.0e-10;
        assert!(
            (got_trace - want_trace).abs() <= trace_tol,
            "workspace psi dH projected trace mismatch: got={got_trace:.6e}, want={want_trace:.6e}",
        );
    }

    #[test]
    fn transformation_normal_joint_hessian_second_directional_derivative_matches_fd() {
        assert!(file!().ends_with(".rs"));
        let psi = array![0.15, -0.10];
        let h = 1e-6;
        let (family, _, state, _) = toy_family_and_derivatives(&psi);
        let p = state.beta.len();
        let dir_u = toy_probe_vector(p, 801);
        let dir_v = toy_probe_vector(p, 802);

        let analytic = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                std::slice::from_ref(&state),
                &dir_u,
                &dir_v,
            )
            .expect("analytic second directional derivative")
            .expect("second directional derivative should be present");

        let eval_dh = |beta: &Array1<f64>| {
            let shifted_state = ParameterBlockState {
                beta: beta.clone(),
                eta: state.eta.clone(),
            };
            family
                .exact_newton_joint_hessian_directional_derivative(
                    std::slice::from_ref(&shifted_state),
                    &dir_u,
                )
                .expect("first directional derivative at shifted beta")
                .expect("shifted first directional derivative should be present")
        };

        let beta_plus = &state.beta + &(dir_v.clone() * h);
        let beta_minus = &state.beta - &(dir_v * h);
        let fd = (eval_dh(&beta_plus) - eval_dh(&beta_minus)) / (2.0 * h);
        assert_matrix_derivativefd(&fd, &analytic, 2e-4, "transformation normal joint d2H");
    }

    #[test]
    fn ctn_joint_hessian_workspace_matvec_matches_dense() {
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();

        let dense = family
            .exact_newton_joint_hessian(std::slice::from_ref(&state))
            .expect("dense joint Hessian build")
            .expect("dense joint Hessian present");
        assert_eq!(dense.nrows(), p);
        assert_eq!(dense.ncols(), p);

        let workspace = family
            .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
            .expect("workspace build")
            .expect("workspace present");

        // `hessian_dense` is amortization-gated; the toy spec carries no
        // penalties, so `expected_reuse=1` against `p/SAFETY≥2` correctly
        // routes through matrix-free. We're testing dense/HVP agreement,
        // not the gate, so force the dense build via `hessian_dense_forced`.
        // The amortization-gate behavior is exercised separately in
        // `ctn_dense_hessian_amortization_gate_picks_matrix_free_when_p_dominates_reuse`.
        let dense_from_workspace = workspace
            .hessian_dense_forced()
            .expect("workspace forced dense Hessian call")
            .expect("workspace forced dense Hessian present");
        assert_eq!(dense_from_workspace.nrows(), p);
        assert_eq!(dense_from_workspace.ncols(), p);
        for i in 0..p {
            for j in 0..p {
                let want = dense[[i, j]];
                let got = dense_from_workspace[[i, j]];
                assert!(
                    (want - got).abs() <= 1e-12 * want.abs().max(1.0) + 1e-12,
                    "workspace dense mismatch at ({i}, {j}): dense={want:.6e}, workspace={got:.6e}"
                );
            }
        }

        // Diagonal must agree element-wise (matrix-free pre-square path vs. dense gram).
        let diag_op = workspace
            .hessian_diagonal()
            .expect("diagonal call")
            .expect("diagonal present");
        assert_eq!(diag_op.len(), p);
        for i in 0..p {
            let want = dense[[i, i]];
            let got = diag_op[i];
            assert!(
                (want - got).abs() <= 1e-12 * want.abs().max(1.0) + 1e-12,
                "diagonal mismatch at {i}: dense={want:.6e}, workspace={got:.6e}"
            );
        }

        // Hessian-vector product must agree with dense H · v across a few
        // randomly chosen directions (deterministic seed for stability).
        let directions = vec![
            toy_probe_vector(p, 101),
            toy_probe_vector(p, 102),
            toy_probe_vector(p, 103),
        ];
        for (k, v) in directions.iter().enumerate() {
            assert_eq!(v.len(), p);
            let want = dense.dot(v);
            let got = workspace
                .hessian_matvec(v)
                .expect("matvec call")
                .expect("matvec present");
            assert_eq!(got.len(), p);
            for i in 0..p {
                let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
                assert!(
                    (want[i] - got[i]).abs() <= tol,
                    "matvec[{k}, {i}] mismatch: dense={:.6e}, workspace={:.6e}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn ctn_joint_hessian_workspace_matvec_into_primes_dense_cache() {
        let psi = array![0.15, -0.10];
        let (family, _, state, _) = toy_family_and_derivatives(&psi);
        let p = state.beta.len();
        let row_quantities = family.row_quantities(&state.beta).expect("row quantities");
        let workspace = TransformationNormalJointHessianWorkspace::new(
            Arc::new(family.clone()),
            state.beta.clone(),
            row_quantities,
        )
        .expect("workspace build");
        assert!(workspace.dense_hessian_cache_enabled());
        assert!(workspace.dense_hessian_cache.get().is_none());

        let dense = family
            .exact_newton_joint_hessian(std::slice::from_ref(&state))
            .expect("dense joint Hessian build")
            .expect("dense joint Hessian present");
        let v = toy_probe_vector(p, 12_345);
        let want = dense.dot(&v);
        let mut got = Array1::<f64>::zeros(p);
        workspace
            .apply_hessian_into(&v, &mut got)
            .expect("workspace matvec_into");
        assert!(workspace.dense_hessian_cache.get().is_some());
        for i in 0..p {
            let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
            assert!(
                (want[i] - got[i]).abs() <= tol,
                "cached matvec_into mismatch at {i}: dense={:.6e}, workspace={:.6e}",
                want[i],
                got[i]
            );
        }

        let v2 = toy_probe_vector(p, 12_346);
        let want2 = dense.dot(&v2);
        workspace
            .apply_hessian_into(&v2, &mut got)
            .expect("second workspace matvec_into");
        for i in 0..p {
            let tol = 1e-12 * want2[i].abs().max(1.0) + 1e-12;
            assert!(
                (want2[i] - got[i]).abs() <= tol,
                "second cached matvec_into mismatch at {i}: dense={:.6e}, workspace={:.6e}",
                want2[i],
                got[i]
            );
        }
    }

    #[test]
    fn ctn_coefficient_hessian_cost_uses_dense_for_small_problems() {
        // Toy family: n=4, p_resp=2, p_cov=2 → p_total=4. The matrix-free
        // gate `use_joint_matrix_free_path(4, 4)` returns false (well below
        // every threshold), so the override must report the dense Khatri–Rao
        // gram cost n·(p_resp·p_cov)² = 4·16 = 64.
        let psi = array![0.15, -0.10];
        let (family, _, _, _) = toy_family_and_derivatives(&psi);
        let n = family.response_val_basis.nrows() as u64;
        let p_resp = family.response_val_basis.ncols() as u64;
        let p_cov = family.covariate_design.ncols() as u64;
        assert!(!crate::custom_family::use_joint_matrix_free_path(
            (p_resp * p_cov) as usize,
            n as usize,
        ));
        let p_total = p_resp * p_cov;
        let expected_dense = n * p_total * p_total;
        assert_eq!(family.coefficient_hessian_cost(&[]), expected_dense);
    }

    #[test]
    fn ctn_coefficient_hessian_cost_switches_to_matvec_when_matrix_free_active() {
        // p_cov=256 keeps p_total = p_resp · p_cov ≥ JOINT_MATRIX_FREE_MIN_DIM
        // so matrix-free is ALWAYS active for any n. The override must report
        // the per-Hv matvec cost n·(p_resp + p_cov), not the dense p² gram.
        // n=8 keeps the test allocation small (~16 KB for covariate_design).
        let n = 8usize;
        let p_cov = 256usize;
        let response = Array1::from_iter((0..n).map(|i| (i as f64) / (n - 1) as f64));
        let (val_basis, deriv_basis, knots, transform, _p_resp) = toy_response_basis(&response);
        let weights = Array1::from_elem(n, 1.0);
        let offset = Array1::zeros(n);
        // Non-degenerate covariate design: small column-wise variation makes
        // the joint warm-start solve well-posed without changing the
        // matrix-free gating behavior tested below.
        let mut cov_design = Array2::<f64>::zeros((n, p_cov));
        for i in 0..n {
            for j in 0..p_cov {
                cov_design[[i, j]] = 0.1 + 0.01 * (i as f64) + 0.001 * (j as f64);
            }
        }
        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            &response,
            val_basis,
            deriv_basis,
            vec![],
            knots,
            toy_scop_ctn_config().response_degree,
            transform,
            &weights,
            &offset,
            DesignMatrix::Dense(DenseDesignMatrix::from(cov_design)),
            vec![],
            &toy_scop_ctn_config(),
            None,
        )
        .expect("matrix-free-eligible CTN family");
        let p_resp = family.response_val_basis.ncols() as u64;
        let actual_p_cov = family.covariate_design.ncols() as u64;
        let p_total = p_resp * actual_p_cov;
        assert!(crate::custom_family::use_joint_matrix_free_path(
            p_total as usize,
            n,
        ));
        let expected_matvec = (n as u64) * (p_resp + actual_p_cov);
        assert_eq!(family.coefficient_hessian_cost(&[]), expected_matvec);
        // Sanity: the matrix-free cost is dramatically smaller than the dense
        // would have been (the whole point of branching).
        let dense_cost = (n as u64) * p_total * p_total;
        assert!(expected_matvec < dense_cost / 100);
    }

    #[test]
    fn ctn_inner_and_outer_hvp_capabilities_are_advertised() {
        let psi = array![0.15, -0.10];
        let (family, derivative_blocks, _, spec) = toy_family_and_derivatives(&psi);
        let specs = std::slice::from_ref(&spec);

        assert!(family.inner_coefficient_hessian_hvp_available(specs));
        assert!(family.outer_hyper_hessian_hvp_available(specs));
        assert!(family.outer_hyper_hessian_dense_available(specs));
        assert_eq!(
            family.exact_outer_derivative_order(specs, &BlockwiseFitOptions::default()),
            crate::custom_family::ExactOuterDerivativeOrder::Second
        );

        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            use_outer_hessian: true,
            ..BlockwiseFitOptions::default()
        };
        let (gradient, hessian) = custom_family_outer_derivatives(&family, specs, &options);
        assert_eq!(
            gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
        assert_eq!(
            hessian,
            crate::solver::outer_strategy::DeclaredHessianForm::Either
        );

        let rho_dim = spec.initial_log_lambdas.len();
        let psi_dim = derivative_blocks[0].len();
        let outer_plan =
            crate::solver::outer_strategy::plan(&crate::solver::outer_strategy::OuterCapability {
                gradient,
                hessian,
                n_params: rho_dim + psi_dim,
                psi_dim,
                fixed_point_available: false,
                barrier_config: None,
                prefer_gradient_only: false,
                disable_fixed_point: true,
            });
        assert_eq!(
            outer_plan.solver,
            crate::solver::outer_strategy::Solver::Arc
        );
        assert_eq!(
            outer_plan.hessian_source,
            crate::solver::outer_strategy::HessianSource::Analytic
        );
    }

    #[test]
    fn ctn_large_n_outer_hvp_capability_selects_operator_path() {
        let psi = array![0.15, -0.10];
        let (family, derivative_blocks, _, spec) = toy_family_and_derivatives(&psi);
        let specs = std::slice::from_ref(&spec);
        assert!(family.outer_hyper_hessian_hvp_available(specs));

        let rho_dim = spec.initial_log_lambdas.len();
        let psi_dim = derivative_blocks[0].len();
        let k_outer = rho_dim + psi_dim;
        // `use_outer_hessian_operator_path` is purely a cost-based crossover
        // over `(n_obs, p_dim, k_outer)`; commit 7f7705c removed the
        // callback-kernel short-circuit that previously let CTN trip the
        // operator path on its analytic HVP alone.  Per the current
        // function docstring, family-supplied directional θθ operators
        // route via `HessianDerivativeProvider::family_outer_hessian_operator`
        // and short-circuit this predicate at the call site.  The
        // meaningful invariant for this test is therefore the dispatcher
        // verdict below — `custom_family_outer_derivatives` must still
        // return `Analytic / Analytic` for both gradient and Hessian.
        // We retain the threshold-tuple sanity check on the predicate so
        // a future regression that broke the cost crossover (e.g. flipped
        // a `>=` to `>`) would still be caught here.
        assert!(
            crate::solver::estimate::reml::unified::use_outer_hessian_operator_path(
                crate::solver::estimate::reml::unified::MATRIX_FREE_OUTER_HESSIAN_LARGE_N_THRESHOLD,
                crate::solver::estimate::reml::unified::MATRIX_FREE_OUTER_HESSIAN_DIM_AT_LARGE_N,
                k_outer,
                false,
            )
        );

        let options = BlockwiseFitOptions {
            use_remlobjective: true,
            use_outer_hessian: true,
            ..BlockwiseFitOptions::default()
        };
        let (gradient, hessian) = custom_family_outer_derivatives(&family, specs, &options);
        assert_eq!(
            gradient,
            crate::solver::outer_strategy::Derivative::Analytic
        );
        assert_eq!(
            hessian,
            crate::solver::outer_strategy::DeclaredHessianForm::Either
        );
    }

    #[test]
    fn ctn_joint_hessian_workspace_dh_operator_matches_dense() {
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();
        let d_beta = toy_probe_vector(p, 201);
        assert_eq!(d_beta.len(), p);

        let dense_dh = family
            .exact_newton_joint_hessian_directional_derivative(
                std::slice::from_ref(&state),
                &d_beta,
            )
            .expect("dense dH build")
            .expect("dense dH present");

        let workspace = family
            .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
            .expect("workspace build")
            .expect("workspace present");
        let dh_op = workspace
            .directional_derivative_operator(&d_beta)
            .expect("dH operator call")
            .expect("dH operator present");

        let probes = vec![
            toy_probe_vector(p, 202),
            toy_probe_vector(p, 203),
            toy_probe_vector(p, 204),
        ];
        let mut probe_mat = Array2::<f64>::zeros((p, probes.len()));
        for (j, w) in probes.iter().enumerate() {
            probe_mat.column_mut(j).assign(w);
        }
        let want_mat = dense_dh.dot(&probe_mat);
        let got_mat = dh_op.mul_mat(&probe_mat);
        for i in 0..p {
            for j in 0..probes.len() {
                let tol = 1e-12 * want_mat[[i, j]].abs().max(1.0) + 1e-12;
                assert!(
                    (want_mat[[i, j]] - got_mat[[i, j]]).abs() <= tol,
                    "dH op matmat[{}, {}] mismatch: dense={:.6e}, op={:.6e}",
                    i,
                    j,
                    want_mat[[i, j]],
                    got_mat[[i, j]]
                );
            }
        }
        let want_trace = probe_mat
            .iter()
            .zip(want_mat.iter())
            .map(|(&f, &bf)| f * bf)
            .sum::<f64>();
        let got_trace = dh_op.trace_projected_factor(&probe_mat);
        let trace_tol = 1e-12 * want_trace.abs().max(1.0) + 1e-12;
        assert!(
            (want_trace - got_trace).abs() <= trace_tol,
            "dH op projected trace mismatch: dense={want_trace:.6e}, op={got_trace:.6e}"
        );
        let cache = ProjectedFactorCache::default();
        let cached_trace = dh_op.trace_projected_factor_cached(&probe_mat, &cache);
        assert!(
            (want_trace - cached_trace).abs() <= trace_tol,
            "dH op cached projected trace mismatch: dense={want_trace:.6e}, op={cached_trace:.6e}"
        );
        let d_beta_2 = toy_probe_vector(p, 205);
        let dense_dh_2 = family
            .exact_newton_joint_hessian_directional_derivative(
                std::slice::from_ref(&state),
                &d_beta_2,
            )
            .expect("second dense dH build")
            .expect("second dense dH present");
        let dh_op_2 = workspace
            .directional_derivative_operator(&d_beta_2)
            .expect("second dH operator call")
            .expect("second dH operator present");
        let want_mat_2 = dense_dh_2.dot(&probe_mat);
        let want_trace_2 = probe_mat
            .iter()
            .zip(want_mat_2.iter())
            .map(|(&f, &bf)| f * bf)
            .sum::<f64>();
        let cached_trace_2 = dh_op_2.trace_projected_factor_cached(&probe_mat, &cache);
        let trace_tol_2 = 1e-12 * want_trace_2.abs().max(1.0) + 1e-12;
        assert!(
            (want_trace_2 - cached_trace_2).abs() <= trace_tol_2,
            "second dH op cached projected trace mismatch: dense={want_trace_2:.6e}, op={cached_trace_2:.6e}"
        );
        for (k, w) in probes.iter().enumerate() {
            assert_eq!(w.len(), p);
            let want = dense_dh.dot(w);
            let got = dh_op.mul_vec(w);
            assert_eq!(got.len(), p);
            for i in 0..p {
                let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
                assert!(
                    (want[i] - got[i]).abs() <= tol,
                    "dH op matvec[{k}, {i}] mismatch: dense={:.6e}, op={:.6e}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    #[test]
    fn ctn_joint_hessian_workspace_d2h_operator_matches_dense() {
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();
        let dir_u = toy_probe_vector(p, 301);
        let dir_v = toy_probe_vector(p, 302);

        let dense_d2h = family
            .exact_newton_joint_hessiansecond_directional_derivative(
                std::slice::from_ref(&state),
                &dir_u,
                &dir_v,
            )
            .expect("dense d2H build")
            .expect("dense d2H present");

        let workspace = family
            .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
            .expect("workspace build")
            .expect("workspace present");
        let d2h_op = workspace
            .second_directional_derivative_operator(&dir_u, &dir_v)
            .expect("d2H operator call")
            .expect("d2H operator present");

        let probes = vec![
            toy_probe_vector(p, 303),
            toy_probe_vector(p, 304),
            toy_probe_vector(p, 305),
        ];
        let mut probe_mat = Array2::<f64>::zeros((p, probes.len()));
        for (j, w) in probes.iter().enumerate() {
            probe_mat.column_mut(j).assign(w);
        }
        let want_mat = dense_d2h.dot(&probe_mat);
        let got_mat = d2h_op.mul_mat(&probe_mat);
        for i in 0..p {
            for j in 0..probes.len() {
                let tol = 1e-12 * want_mat[[i, j]].abs().max(1.0) + 1e-12;
                assert!(
                    (want_mat[[i, j]] - got_mat[[i, j]]).abs() <= tol,
                    "d2H op matmat[{}, {}] mismatch: dense={:.6e}, op={:.6e}",
                    i,
                    j,
                    want_mat[[i, j]],
                    got_mat[[i, j]]
                );
            }
        }
        for (k, w) in probes.iter().enumerate() {
            assert_eq!(w.len(), p);
            let want = dense_d2h.dot(w);
            let got = d2h_op.mul_vec(w);
            assert_eq!(got.len(), p);
            for i in 0..p {
                let tol = 1e-12 * want[i].abs().max(1.0) + 1e-12;
                assert!(
                    (want[i] - got[i]).abs() <= tol,
                    "d2H op matvec[{k}, {i}] mismatch: dense={:.6e}, op={:.6e}",
                    want[i],
                    got[i]
                );
            }
        }
    }

    /// Cached CTN barrier dH operator check (third-derivative formula
    /// `D(∇²B)[u]v = -2 μ Dᵀ((Du)(Dv)/c³)`).
    ///
    /// At fixed direction `d_beta`, builds `H(β ± ε d_beta) v` matrix-free via
    /// `apply_hessian` and checks that the centered perturbation quotient
    /// converges to the operator's `mul_vec(v)`. This locks in both the analytic formula and the
    /// `inv_hp_cu` cache (a stale cache would only show up under ε perturbation,
    /// not in the dense-equivalence test that probes a single iterate).
    #[test]
    fn ctn_dh_operator_matches_fd_under_beta_perturbation() {
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();
        let d_beta = toy_probe_vector(p, 401);
        let v = toy_probe_vector(p, 402);
        assert_eq!(d_beta.len(), p);
        assert_eq!(v.len(), p);

        let workspace = family
            .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
            .expect("workspace build")
            .expect("workspace present");
        let want = workspace
            .directional_derivative_operator(&d_beta)
            .expect("dH op call")
            .expect("dH op present")
            .mul_vec(&v);

        let eps = 1e-5;
        let make_state = |scale: f64| ParameterBlockState {
            beta: &state.beta + &(d_beta.mapv(|b| scale * b)),
            eta: state.eta.clone(),
        };
        let plus = family
            .exact_newton_joint_hessian_workspace(
                std::slice::from_ref(&make_state(eps)),
                &[spec.clone()],
            )
            .expect("plus workspace")
            .expect("plus workspace present");
        let minus = family
            .exact_newton_joint_hessian_workspace(
                std::slice::from_ref(&make_state(-eps)),
                &[spec.clone()],
            )
            .expect("minus workspace")
            .expect("minus workspace present");
        let hv_plus = plus
            .hessian_matvec(&v)
            .expect("plus matvec")
            .expect("plus matvec");
        let hv_minus = minus
            .hessian_matvec(&v)
            .expect("minus matvec")
            .expect("minus matvec");
        let fd: Array1<f64> = (&hv_plus - &hv_minus).mapv(|x| x / (2.0 * eps));

        for i in 0..p {
            let scale = want[i].abs().max(1.0);
            // O(ε²) centered FD on a smooth Hessian gives ~1e-7 relative error
            // at ε=1e-5; loose 5e-5 tolerance covers the dominant truncation
            // term plus the inflation by `||v||·||d_beta||`.
            let tol = 5e-5 * scale + 5e-7;
            assert!(
                (want[i] - fd[i]).abs() <= tol,
                "dH FD mismatch at {i}: op={:.6e}, fd={:.6e}, tol={:.6e}",
                want[i],
                fd[i],
                tol,
            );
        }
    }

    /// Cached CTN barrier d²H operator check (fourth-derivative
    /// formula `D²(∇²B)[u,w]v = 6 μ Dᵀ((Du)(Dw)(Dv)/c⁴)`).
    ///
    /// A centered perturbation of the dH operator along `dir_w` recovers d²H[u, w] · v;
    /// this exercises both the cached `inv_hp_qu` and the chained Khatri–Rao
    /// apply on the perturbed iterate.
    #[test]
    fn ctn_d2h_operator_matches_fd_under_beta_perturbation() {
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();
        let dir_u = toy_probe_vector(p, 501);
        let dir_w = toy_probe_vector(p, 502);
        let v = toy_probe_vector(p, 503);

        let workspace = family
            .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
            .expect("workspace build")
            .expect("workspace present");
        let want = workspace
            .second_directional_derivative_operator(&dir_u, &dir_w)
            .expect("d2H op call")
            .expect("d2H op present")
            .mul_vec(&v);

        let eps = 1e-5;
        let make_state = |scale: f64| ParameterBlockState {
            beta: &state.beta + &(dir_w.mapv(|b| scale * b)),
            eta: state.eta.clone(),
        };
        let plus_ws = family
            .exact_newton_joint_hessian_workspace(
                std::slice::from_ref(&make_state(eps)),
                &[spec.clone()],
            )
            .expect("plus ws")
            .expect("plus ws present");
        let minus_ws = family
            .exact_newton_joint_hessian_workspace(
                std::slice::from_ref(&make_state(-eps)),
                &[spec.clone()],
            )
            .expect("minus ws")
            .expect("minus ws present");
        let dh_plus = plus_ws
            .directional_derivative_operator(&dir_u)
            .expect("plus dH op call")
            .expect("plus dH op present")
            .mul_vec(&v);
        let dh_minus = minus_ws
            .directional_derivative_operator(&dir_u)
            .expect("minus dH op call")
            .expect("minus dH op present")
            .mul_vec(&v);
        let fd: Array1<f64> = (&dh_plus - &dh_minus).mapv(|x| x / (2.0 * eps));

        for i in 0..p {
            let scale = want[i].abs().max(1.0);
            let tol = 5e-5 * scale + 5e-7;
            assert!(
                (want[i] - fd[i]).abs() <= tol,
                "d2H FD mismatch at {i}: op={:.6e}, fd={:.6e}, tol={:.6e}",
                want[i],
                fd[i],
                tol,
            );
        }
    }

    /// FD check for the CTN barrier `∇²B v` operator itself: centered FD on the
    /// log-likelihood gradient w.r.t. β reproduces `H(β) v` (to within FD
    /// truncation). This is the `μ Dᵀ((Dv)/c²)` formula plus the
    /// β-independent `X_val^T W X_val` term.
    #[test]
    fn ctn_hessian_matvec_matches_grad_fd() {
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();
        let v = toy_probe_vector(p, 601);

        let workspace = family
            .exact_newton_joint_hessian_workspace(std::slice::from_ref(&state), &[spec.clone()])
            .expect("workspace build")
            .expect("workspace present");
        let hv = workspace
            .hessian_matvec(&v)
            .expect("matvec call")
            .expect("matvec present");

        let eps = 1e-6;
        // CTN's `evaluate()` returns the score (gradient of log-likelihood)
        // through the working-set; the joint Hessian is `-d²ℓ/dβ²`, so
        // `H · v ≈ -[grad(β + εv) - grad(β - εv)] / (2ε)`.
        let make_state = |scale: f64| ParameterBlockState {
            beta: &state.beta + &(v.mapv(|b| scale * b)),
            eta: state.eta.clone(),
        };
        let grad_at = |st: &ParameterBlockState| -> Array1<f64> {
            let eval = family
                .evaluate(std::slice::from_ref(st))
                .expect("evaluate must succeed");
            match &eval.blockworking_sets[0] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient.clone(),
                _ => panic!("CTN must report ExactNewton working set"),
            }
        };
        let grad_plus = grad_at(&make_state(eps));
        let grad_minus = grad_at(&make_state(-eps));
        // The score is +∂ℓ/∂β, and H = -∂²ℓ/∂β². Centered FD on the score gives
        // dscore/dβ · v = -H · v, so we negate to compare against `hv`.
        let fd: Array1<f64> = (&grad_plus - &grad_minus).mapv(|x| -x / (2.0 * eps));

        for i in 0..p {
            let scale = hv[i].abs().max(1.0);
            let tol = 1e-4 * scale + 1e-6;
            assert!(
                (hv[i] - fd[i]).abs() <= tol,
                "Hv FD mismatch at {i}: op={:.6e}, fd={:.6e}, tol={:.6e}",
                hv[i],
                fd[i],
                tol,
            );
        }
    }

    #[test]
    fn ctn_scop_gradient_matches_loglikelihood_fd() {
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();

        let analytic = family
            .exact_newton_joint_gradient_evaluation(std::slice::from_ref(&state), &[spec])
            .expect("SCOP analytic gradient evaluation")
            .expect("SCOP analytic gradient must be present");
        assert_eq!(analytic.gradient.len(), p);

        let eps = 1e-6;
        for coord in 0..p {
            let mut beta_plus = state.beta.clone();
            beta_plus[coord] += eps;
            let plus_state = ParameterBlockState {
                beta: beta_plus,
                eta: state.eta.clone(),
            };
            let ll_plus = family
                .log_likelihood_only(std::slice::from_ref(&plus_state))
                .expect("positive perturbation remains feasible");

            let mut beta_minus = state.beta.clone();
            beta_minus[coord] -= eps;
            let minus_state = ParameterBlockState {
                beta: beta_minus,
                eta: state.eta.clone(),
            };
            let ll_minus = family
                .log_likelihood_only(std::slice::from_ref(&minus_state))
                .expect("negative perturbation remains feasible");

            let fd = (ll_plus - ll_minus) / (2.0 * eps);
            let scale = fd.abs().max(analytic.gradient[coord].abs()).max(1.0);
            let tol = 5e-6 * scale + 5e-8;
            assert!(
                (analytic.gradient[coord] - fd).abs() <= tol,
                "SCOP gradient FD mismatch at {coord}: analytic={:.6e}, fd={:.6e}, tol={:.6e}",
                analytic.gradient[coord],
                fd,
                tol,
            );
        }
    }

    #[test]
    fn ctn_exact_newton_joint_gradient_evaluation_matches_evaluate() {
        // The joint-Newton inner solver prefers
        // `exact_newton_joint_gradient_evaluation` over `evaluate()` to refresh
        // the gradient between cycles. Lock in that the override returns
        // exactly the same log-likelihood and flat gradient that the dense
        // path produces (up to floating-point summation order).
        let psi = array![0.15, -0.10];
        let (family, _, state, spec) = toy_family_and_derivatives(&psi);
        let p = spec.design.ncols();

        let eval = family
            .evaluate(std::slice::from_ref(&state))
            .expect("evaluate must succeed on the toy fixture");
        let want_ll = eval.log_likelihood;
        let want_grad = match &eval.blockworking_sets[0] {
            BlockWorkingSet::ExactNewton { gradient, .. } => gradient.clone(),
            _ => panic!("CTN must report an ExactNewton block working set"),
        };
        assert_eq!(want_grad.len(), p);

        let gradient_eval = family
            .exact_newton_joint_gradient_evaluation(std::slice::from_ref(&state), &[spec.clone()])
            .expect("gradient-only call")
            .expect("gradient-only result must be present");
        assert!(
            (want_ll - gradient_eval.log_likelihood).abs()
                <= 1e-12 * want_ll.abs().max(1.0) + 1e-12,
            "log-likelihood mismatch: evaluate={:.6e}, gradient-only={:.6e}",
            want_ll,
            gradient_eval.log_likelihood,
        );
        assert_eq!(gradient_eval.gradient.len(), p);
        for i in 0..p {
            let tol = 1e-12 * want_grad[i].abs().max(1.0) + 1e-12;
            assert!(
                (want_grad[i] - gradient_eval.gradient[i]).abs() <= tol,
                "gradient mismatch at {i}: evaluate={:.6e}, gradient-only={:.6e}",
                want_grad[i],
                gradient_eval.gradient[i],
            );
        }
    }

    /// Pairwise oracle for Phase-2 outer-HVP cross-checking.
    ///
    /// Builds the toy CTN fixture (n=4, p_resp=2, p_cov=2, ψ_dim=2), calls the
    /// existing pairwise body `exact_newton_joint_psisecond_order_terms(i, j)`
    /// for every (i, j) pair, computes the directional contraction
    /// `Σ_j v_j · pair(i, j)` for a fixed direction `v`, and writes the full
    /// likelihood-only result to `/tmp/ctn_pairwise_oracle.json`.
    ///
    /// Independent verification path for the SCOP CTN HVP work. The old Python
    /// scripts used the pre-SCOP linear tensor likelihood and were removed so
    /// they cannot be mistaken for ground truth.
    ///
    /// Run via:
    ///     cargo test --release ctn_pairwise_oracle_dumps_json -- --nocapture
    #[test]
    fn ctn_pairwise_oracle_dumps_json() {
        let psi = array![0.15, -0.10];
        let v = array![0.4, -0.7];

        let (family, derivative_blocks, state, spec) = toy_family_and_derivatives(&psi);
        let block_states = vec![state.clone()];
        let specs = vec![spec.clone()];
        let beta = state.beta.clone();

        let psi_dim = psi.len();
        let p_total = beta.len();
        let n_obs = family.weights.as_ref().len();

        eprintln!(
            "[oracle] toy CTN: n={n_obs}, p_resp=2, p_cov=2, p_total={p_total}, ψ_dim={psi_dim}"
        );
        eprintln!("[oracle] β = {:?}", beta.as_slice().unwrap());
        eprintln!("[oracle] ψ = {:?}", psi.as_slice().unwrap());
        eprintln!("[oracle] v = {:?}", v.as_slice().unwrap());

        // The CTN psi-psi second-order kernel can return the dense block
        // either eagerly (`hessian_psi_psi`, p_total×p_total) or lazily
        // (`hessian_psi_psi_operator`, materialized via `to_dense`). The
        // oracle dump records the dense numbers, so materialize on demand.
        let dense_pair_hessian = |terms: &ExactNewtonJointPsiSecondOrderTerms| -> Array2<f64> {
            if terms.hessian_psi_psi.nrows() > 0 {
                terms.hessian_psi_psi.clone()
            } else {
                terms
                    .hessian_psi_psi_operator
                    .as_ref()
                    .expect("CTN psi-psi must expose either dense Hessian or operator")
                    .to_dense()
            }
        };

        // Per-pair pairwise body — likelihood pieces only (no penalty/logdet).
        let mut pair_records = Vec::new();
        for i in 0..psi_dim {
            for j in 0..psi_dim {
                let terms = family
                    .exact_newton_joint_psisecond_order_terms(
                        &block_states,
                        &specs,
                        &derivative_blocks,
                        i,
                        j,
                    )
                    .expect("pairwise call ok")
                    .expect("pairwise returns Some for valid i,j");
                let g_inf = terms
                    .score_psi_psi
                    .iter()
                    .fold(0.0f64, |m, x| m.max(x.abs()));
                let b_dense = dense_pair_hessian(&terms);
                let b_inf = b_dense.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                eprintln!(
                    "[oracle] pair (i={i}, j={j}): a={:.10}, ‖g‖∞={:.6e}, ‖b_mat‖∞={:.6e}",
                    terms.objective_psi_psi, g_inf, b_inf,
                );
                pair_records.push(serde_json::json!({
                    "i": i,
                    "j": j,
                    "a": terms.objective_psi_psi,
                    "g": terms.score_psi_psi.to_vec(),
                    "b_mat": b_dense.iter().copied().collect::<Vec<f64>>(),
                    "b_mat_shape": [b_dense.nrows(), b_dense.ncols()],
                }));
            }
        }

        // Directional contraction: Σ_j v_j · pair(i, j) for each free axis i.
        let mut a_dir = Array1::<f64>::zeros(psi_dim);
        let mut g_dir = Array2::<f64>::zeros((psi_dim, p_total));
        let mut b_dir = vec![Array2::<f64>::zeros((p_total, p_total)); psi_dim];
        for i in 0..psi_dim {
            for j in 0..psi_dim {
                let terms = family
                    .exact_newton_joint_psisecond_order_terms(
                        &block_states,
                        &specs,
                        &derivative_blocks,
                        i,
                        j,
                    )
                    .expect("pairwise call ok")
                    .expect("pairwise returns Some for valid i,j");
                a_dir[i] += v[j] * terms.objective_psi_psi;
                let mut g_row = g_dir.slice_mut(s![i, ..]);
                g_row.scaled_add(v[j], &terms.score_psi_psi);
                b_dir[i].scaled_add(v[j], &dense_pair_hessian(&terms));
            }
        }

        eprintln!("[oracle] directional contraction Σ_j v_j · pair(i, j):");
        for i in 0..psi_dim {
            eprintln!(
                "[oracle]   i={i}: a_dir={:.10}, ‖g_dir‖∞={:.6e}, ‖b_dir‖∞={:.6e}",
                a_dir[i],
                g_dir.row(i).iter().fold(0.0f64, |m, x| m.max(x.abs())),
                b_dir[i].iter().fold(0.0f64, |m, x| m.max(x.abs())),
            );
        }

        let directional_records: Vec<_> = (0..psi_dim)
            .map(|i| {
                serde_json::json!({
                    "i": i,
                    "a_dir": a_dir[i],
                    "g_dir": g_dir.row(i).to_vec(),
                    "b_dir": b_dir[i].iter().copied().collect::<Vec<f64>>(),
                    "b_dir_shape": [p_total, p_total],
                })
            })
            .collect();

        let blob = serde_json::json!({
            "config": {
                "n": n_obs,
                "p_resp": 2,
                "p_cov": 2,
                "p_total": p_total,
                "psi_dim": psi_dim,
                "beta": beta.to_vec(),
                "psi": psi.to_vec(),
                "v": v.to_vec(),
            },
            "pairwise": pair_records,
            "directional_contraction": directional_records,
            "note": "Likelihood-only pieces from exact_newton_joint_psisecond_order_terms. \
                     Penalty/logdet contributions are added by the unified evaluator's \
                     outer_hessian_entry. Cross-check this against sympy-shadow's symbolic \
                     derivation of the same likelihood quantities at the same toy config.",
        });

        let path = "/tmp/ctn_pairwise_oracle.json";
        std::fs::write(
            path,
            serde_json::to_string_pretty(&blob).expect("serialize ok"),
        )
        .expect("write ok");
        eprintln!("[oracle] wrote {path}");

        // Sanity assertions: nothing NaN, directional contraction is consistent
        // with element-wise summation.
        assert!(a_dir.iter().all(|x| x.is_finite()));
        assert!(g_dir.iter().all(|x| x.is_finite()));
        assert!(b_dir.iter().all(|m| m.iter().all(|x| x.is_finite())));
    }
}

impl CustomFamilyPsiDerivativeOperator for TensorKroneckerPsiOperator {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn n_data(&self) -> usize {
        TensorKroneckerPsiOperator::n_data(self)
    }

    fn p_out(&self) -> usize {
        TensorKroneckerPsiOperator::p_out(self)
    }

    fn transpose_mul(
        &self,
        axis: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        // Per-axis trait method routes through the directional accumulator with
        // a unit basis vector e_axis. The accumulator skips zero entries, so
        // this loops once over `axis` and yields the same p-vector that
        // `lifted_transpose(_, axis, v)` would, while keeping the directional
        // kernel as a live production caller (substrate for the future HVP).
        let mut unit = Array1::<f64>::zeros(self.covariate_derivs.len());
        unit[axis] = 1.0;
        self.transpose_directional(&unit.view(), v)
    }

    fn forward_mul(
        &self,
        axis: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let mut unit = Array1::<f64>::zeros(self.covariate_derivs.len());
        unit[axis] = 1.0;
        self.forward_directional(&unit.view(), u)
    }

    fn transpose_mul_second_diag(
        &self,
        axis: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let q = self.covariate_derivs.len();
        let mut unit = Array1::<f64>::zeros(q);
        unit[axis] = 1.0;
        self.transpose_second_directional(&unit.view(), &unit.view(), v)
    }

    fn transpose_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        v: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let q = self.covariate_derivs.len();
        let mut unit_d = Array1::<f64>::zeros(q);
        let mut unit_e = Array1::<f64>::zeros(q);
        unit_d[axis_d] = 1.0;
        unit_e[axis_e] = 1.0;
        self.transpose_second_directional(&unit_d.view(), &unit_e.view(), v)
    }

    fn forward_mul_second_diag(
        &self,
        axis: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        // Bilinear directional accumulator with `v_psi = w_psi = e_axis` skips
        // every (j,k) pair except (axis, axis), so this is numerically equivalent
        // to `lifted_forward_second(_, axis, axis, u)`.
        let q = self.covariate_derivs.len();
        let mut unit = Array1::<f64>::zeros(q);
        unit[axis] = 1.0;
        self.forward_second_directional(&unit.view(), &unit.view(), u)
    }

    fn forward_mul_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        u: &ndarray::ArrayView1<'_, f64>,
    ) -> Result<Array1<f64>, crate::terms::basis::BasisError> {
        let q = self.covariate_derivs.len();
        let mut unit_d = Array1::<f64>::zeros(q);
        let mut unit_e = Array1::<f64>::zeros(q);
        unit_d[axis_d] = 1.0;
        unit_e[axis_e] = 1.0;
        self.forward_second_directional(&unit_d.view(), &unit_e.view(), u)
    }

    fn row_chunk_first(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let cov = self.cov_first_axis_row_chunk(axis, rows.clone())?;
        self.lifted_row_chunk_from_cov(rows, &cov)
    }

    fn row_chunk_second_diag(
        &self,
        axis: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let cov = self.cov_second_axis_row_chunk(axis, axis, rows.clone())?;
        self.lifted_row_chunk_from_cov(rows, &cov)
    }

    fn row_chunk_second_cross(
        &self,
        axis_d: usize,
        axis_e: usize,
        rows: std::ops::Range<usize>,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        let cov = self.cov_second_axis_row_chunk(axis_d, axis_e, rows.clone())?;
        self.lifted_row_chunk_from_cov(rows, &cov)
    }

    fn as_materializable(&self) -> Option<&dyn MaterializablePsiDerivativeOperator> {
        Some(self)
    }
}

impl MaterializablePsiDerivativeOperator for TensorKroneckerPsiOperator {
    fn materialize_first(
        &self,
        axis: usize,
    ) -> Result<Array2<f64>, crate::terms::basis::BasisError> {
        Ok(self.materialize_lifted(&self.response_val_basis, &self.materialize_cov_first(axis)?))
    }
}

// ---------------------------------------------------------------------------
// Per-evaluation ψ workspace
// ---------------------------------------------------------------------------

/// Per-evaluation ψ workspace for `TransformationNormalFamily`.
///
/// The CTN row-streaming first-order ψ kernel ([`scop_psi_terms`]) walks all
/// `n` rows serially and is invoked once per ψ axis. At biobank scale that is
/// the dominant outer-evaluation cost. All `n_psi` axes share the same per-row
/// state — `γ`, `h`, `h'`, `endpoint_q`, the response basis rows `rv`/`rd`,
/// the covariate row, and the row weight. The only per-axis input is
/// `cov_psi[axis] = ∂C/∂ψ_axis`, which this workspace streams in bounded row
/// chunks so the exact all-axis gradient never has to cache all
/// `n * p_cov * n_psi` derivative entries at once.
///
/// This workspace
///
/// 1. precomputes all per-axis ψ first-order terms in a single rayon
///    fold/reduce over rows so the per-row state is loaded once and reused
///    across axes, and
/// 2. caches the resulting per-axis terms so subsequent
///    `first_order_terms(idx)` lookups are O(p_total) clones rather than full
///    row walks.
///
/// `second_order_terms` are cached as a full symmetric ψ-pair table after the
/// first request so full-Hessian outer evaluations do not repeatedly traverse
/// the same CTN rows for each `(ψ_i, ψ_j)` callback. The mixed
/// `hessian_directional_derivative` hook still delegates to the direct path.
struct TransformationNormalPsiWorkspaceCacheEntry {
    objective_psi: f64,
    score_psi: Array1<f64>,
    op_arc: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    axis: usize,
    trace_axes: Arc<Vec<usize>>,
    trace_axis_pos: usize,
    row_gamma: Arc<Array2<f64>>,
    row_h: Arc<Array1<f64>>,
    row_h_prime: Arc<Array1<f64>>,
    endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    beta: Arc<Array1<f64>>,
}

struct TransformationNormalPsiWorkspaceAxisSnapshot {
    op_arc: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    axis: usize,
    row_gamma: Arc<Array2<f64>>,
    row_h: Arc<Array1<f64>>,
    row_h_prime: Arc<Array1<f64>>,
    endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    beta: Arc<Array1<f64>>,
}

struct TransformationNormalPsiWorkspacePairCacheEntry {
    objective_psi_psi: f64,
    score_psi_psi: Array1<f64>,
    op_arc: Arc<dyn CustomFamilyPsiDerivativeOperator>,
    axis_i: usize,
    axis_j: usize,
    trace_axes: Arc<Vec<usize>>,
    trace_axis_i_pos: usize,
    trace_axis_j_pos: usize,
    row_gamma: Arc<Array2<f64>>,
    row_h: Arc<Array1<f64>>,
    row_h_prime: Arc<Array1<f64>>,
    endpoint_q: Arc<Vec<LogNormalCdfDiffDerivatives>>,
    beta: Arc<Array1<f64>>,
}

struct TransformationNormalPsiWorkspace {
    family: TransformationNormalFamily,
    block_states: Vec<ParameterBlockState>,
    derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    cache: Mutex<Option<Vec<TransformationNormalPsiWorkspaceCacheEntry>>>,
    pair_cache:
        Mutex<Option<Vec<Vec<Option<Arc<TransformationNormalPsiWorkspacePairCacheEntry>>>>>>,
}

impl TransformationNormalPsiWorkspace {
    fn new(
        family: TransformationNormalFamily,
        block_states: Vec<ParameterBlockState>,
        derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    ) -> Self {
        Self {
            family,
            block_states,
            derivative_blocks,
            cache: Mutex::new(None),
            pair_cache: Mutex::new(None),
        }
    }

    /// Compute all per-axis ψ first-order terms in a single parallel row pass.
    ///
    /// Each row's per-row state (γ, h, h', endpoint_q, rv, rd, cov_row, weight)
    /// is loaded once and reused across every ψ axis, in contrast to the
    /// per-axis [`scop_psi_terms`] path that reloads it once per axis. Op
    /// counts are identical to the per-axis path; only the loop nesting and
    /// reduction shape change.
    fn compute_all_axes(&self) -> Result<Vec<TransformationNormalPsiWorkspaceCacheEntry>, String> {
        if self.block_states.len() != 1 {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "TransformationNormalFamily expects 1 block, got {}",
                    self.block_states.len()
                ),
            }
            .into());
        }
        if self.derivative_blocks.is_empty() {
            return Ok(Vec::new());
        }
        let block_derivs = &self.derivative_blocks[0];
        let n_psi = block_derivs.len();
        if n_psi == 0 {
            return Ok(Vec::new());
        }

        let beta = &self.block_states[0].beta;
        let row = self.family.row_quantities(beta)?;
        let n = self.family.response_val_basis.nrows();
        let p_resp = self.family.response_val_basis.ncols();
        let p_cov = self.family.covariate_design.ncols();
        let p_total = p_resp * p_cov;
        if beta.len() != p_total {
            return Err(TransformationNormalError::InvalidInput { reason: format!(
                "TransformationNormalPsiWorkspace beta length {} != p_resp({p_resp}) * p_cov({p_cov})",
                beta.len()
            ) }.into());
        }
        let beta_mat = beta
            .view()
            .into_shape_with_order((p_resp, p_cov))
            .map_err(|e| format!("ψ workspace beta reshape failed: {e}"))?;

        // Resolve and validate the shared tensor-Kronecker ψ operator across
        // every axis. CTN is single-block so all entries point at the same
        // operator instance; we still loop to validate the contract.
        let mut op_arcs: Vec<Arc<dyn CustomFamilyPsiDerivativeOperator>> =
            Vec::with_capacity(n_psi);
        let mut axes: Vec<usize> = Vec::with_capacity(n_psi);
        for deriv in block_derivs.iter() {
            let op_arc = deriv
                .implicit_operator
                .as_ref()
                .ok_or_else(|| {
                    "TransformationNormalFamily ψ workspace requires implicit operator on each axis"
                        .to_string()
                })?
                .clone();
            // Validate tensor-Kronecker backing without holding a reference
            // across the move into `op_arcs`.
            if op_arc
                .as_any()
                .downcast_ref::<TensorKroneckerPsiOperator>()
                .is_none()
            {
                return Err(
                    "TransformationNormalFamily ψ workspace requires tensor-backed operator"
                        .to_string(),
                );
            }
            axes.push(deriv.implicit_axis);
            op_arcs.push(op_arc);
        }
        // The shared instance is whichever axis we resolve first; downcast it
        // again for row-chunk streaming. CTN's `build_tensor_psi_derivatives`
        // guarantees this is the same instance across every axis.
        let shared_op_arc = Arc::clone(&op_arcs[0]);
        let Some(op) = shared_op_arc
            .as_any()
            .downcast_ref::<TensorKroneckerPsiOperator>()
        else {
            return Err(
                "TransformationNormalFamily ψ workspace lost tensor-backed operator after validation"
                    .to_string(),
            );
        };

        let weights = self.family.effective_weights();
        let h = row.h.as_ref();
        let h_prime = row.h_prime.as_ref();
        let endpoint_q = row.endpoint_q.as_ref();
        let endpoint_basis =
            [
                self.family.response_upper_basis.as_slice().ok_or_else(|| {
                    "ψ workspace endpoint upper basis is not contiguous".to_string()
                })?,
                self.family.response_lower_basis.as_slice().ok_or_else(|| {
                    "ψ workspace endpoint lower basis is not contiguous".to_string()
                })?,
            ];

        // Single-pass row walk: for each row, load the per-row state once and
        // accumulate every axis's `objective_psi`/`score_psi` in lockstep.
        struct PsiAllAxesAccum {
            objective_psi: Vec<f64>,
            score_psi: Vec<Array1<f64>>,
        }

        impl PsiAllAxesAccum {
            fn new(n_psi: usize, p_total: usize) -> Self {
                Self {
                    objective_psi: vec![0.0; n_psi],
                    score_psi: (0..n_psi).map(|_| Array1::<f64>::zeros(p_total)).collect(),
                }
            }

            fn merge(mut self, rhs: Self) -> Self {
                for (a, v) in rhs.objective_psi.into_iter().enumerate() {
                    self.objective_psi[a] += v;
                }
                for (a, score) in rhs.score_psi.into_iter().enumerate() {
                    self.score_psi[a].scaled_add(1.0, &score);
                }
                self
            }
        }

        let policy = ResourcePolicy::default_library();
        let row_bytes = p_cov
            .saturating_mul(n_psi + 1)
            .saturating_mul(std::mem::size_of::<f64>())
            .max(1);
        let target_chunk_bytes =
            (16 * 1024 * 1024).min((policy.max_single_materialization_bytes / 8).max(row_bytes));
        let chunk_rows = (target_chunk_bytes / row_bytes).clamp(1, n.max(1));
        let row_chunks: Vec<(usize, usize)> = (0..n)
            .step_by(chunk_rows)
            .map(|start| (start, (start + chunk_rows).min(n)))
            .collect();

        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let partials: Vec<Result<PsiAllAxesAccum, String>> = row_chunks
            .into_par_iter()
            .map(|(start, end)| {
                let cov = self
                    .family
                    .covariate_design
                    .try_row_chunk(start..end)
                    .map_err(|e| format!("ψ workspace covariate row chunk {start}..{end}: {e}"))?;
                let mut cov_psi_chunks: Vec<Array2<f64>> = Vec::with_capacity(n_psi);
                for &axis in &axes {
                    let cov_psi = op
                        .cov_first_axis_row_chunk_streaming(axis, start..end)
                        .map_err(|e| {
                            format!("ψ workspace covariate ψ row chunk axis {axis} {start}..{end}: {e}")
                        })?;
                    if cov_psi.nrows() != end - start || cov_psi.ncols() != p_cov {
                        return Err(TransformationNormalError::InvalidInput { reason: format!(
                            "ψ workspace covariate derivative chunk shape {}x{} for axis {axis} rows {start}..{end} != expected {}x{}",
                            cov_psi.nrows(),
                            cov_psi.ncols(),
                            end - start,
                            p_cov
                        ) }.into());
                    }
                    cov_psi_chunks.push(cov_psi);
                }

                let mut acc = PsiAllAxesAccum::new(n_psi, p_total);
                let mut gamma = vec![0.0; p_resp];
                let mut h_factor = vec![0.0; p_resp];
                let mut hp_factor = vec![0.0; p_resp];
                let mut endpoint_factor = vec![[0.0_f64; 2]; p_resp];
                let mut gamma_psi = vec![0.0; p_resp];
                let mut hpsi_cov_factor = vec![0.0; p_resp];
                let mut hppsi_cov_factor = vec![0.0; p_resp];
                let mut hpsi_psi_factor = vec![0.0; p_resp];
                let mut hppsi_psi_factor = vec![0.0; p_resp];
                let mut endpoint_psi = [0.0_f64; 2];
                let mut endpoint_psi_cov_factor = vec![[0.0_f64; 2]; p_resp];
                let mut endpoint_psi_psi_factor = vec![[0.0_f64; 2]; p_resp];

                for local_i in 0..(end - start) {
                    let i = start + local_i;
                    let cov_row = cov.row(local_i);
                    let rv = self.family.response_val_basis.row(i);
                    let rd = self.family.response_deriv_basis.row(i);
                    let wi = weights[i];
                    let hi = h[i];
                    let hp = h_prime[i];
                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let q = &endpoint_q[i];
                    let gamma_row = row.gamma.row(i);

                    for k in 0..p_resp {
                        gamma[k] = gamma_row[k];
                    }

                    h_factor[0] = rv[0];
                    hp_factor[0] = rd[0];
                    for k in 1..p_resp {
                        h_factor[k] = 2.0 * rv[k] * gamma[k];
                        hp_factor[k] = 2.0 * rd[k] * gamma[k];
                    }
                    for e in 0..2 {
                        let basis = endpoint_basis[e];
                        endpoint_factor[0][e] = basis[0];
                        for k in 1..p_resp {
                            endpoint_factor[k][e] = 2.0 * basis[k] * gamma[k];
                        }
                    }

                    for axis_idx in 0..n_psi {
                        let psi_row = cov_psi_chunks[axis_idx].row(local_i);

                        for k in 0..p_resp {
                            gamma_psi[k] = beta_mat.row(k).dot(&psi_row);
                        }

                        let mut h_psi = rv[0] * gamma_psi[0];
                        let mut hp_psi = rd[0] * gamma_psi[0];
                        for k in 1..p_resp {
                            h_psi += 2.0 * rv[k] * gamma[k] * gamma_psi[k];
                            hp_psi += 2.0 * rd[k] * gamma[k] * gamma_psi[k];
                        }

                        for e in 0..2 {
                            let basis = endpoint_basis[e];
                            endpoint_psi[e] = basis[0] * gamma_psi[0];
                            endpoint_psi_psi_factor[0][e] = basis[0];
                            endpoint_psi_cov_factor[0][e] = 0.0;
                            for k in 1..p_resp {
                                endpoint_psi[e] += 2.0 * basis[k] * gamma[k] * gamma_psi[k];
                                endpoint_psi_cov_factor[k][e] = 2.0 * basis[k] * gamma_psi[k];
                                endpoint_psi_psi_factor[k][e] = 2.0 * basis[k] * gamma[k];
                            }
                        }

                        acc.objective_psi[axis_idx] += wi
                            * (hi * h_psi
                                - hp_psi * inv_hp
                                + endpoint_chain_first(q, endpoint_psi));

                        hpsi_psi_factor[0] = rv[0];
                        hppsi_psi_factor[0] = rd[0];
                        hpsi_cov_factor[0] = 0.0;
                        hppsi_cov_factor[0] = 0.0;
                        for k in 1..p_resp {
                            hpsi_cov_factor[k] = 2.0 * rv[k] * gamma_psi[k];
                            hppsi_cov_factor[k] = 2.0 * rd[k] * gamma_psi[k];
                            hpsi_psi_factor[k] = 2.0 * rv[k] * gamma[k];
                            hppsi_psi_factor[k] = 2.0 * rd[k] * gamma[k];
                        }

                        let score_axis = &mut acc.score_psi[axis_idx];
                        for k in 0..p_resp {
                            for c in 0..p_cov {
                                let idx = k * p_cov + c;
                                let h_a = h_factor[k] * cov_row[c];
                                let hp_a = hp_factor[k] * cov_row[c];
                                let hpsi_a = hpsi_cov_factor[k] * cov_row[c]
                                    + hpsi_psi_factor[k] * psi_row[c];
                                let hppsi_a = hppsi_cov_factor[k] * cov_row[c]
                                    + hppsi_psi_factor[k] * psi_row[c];
                                let endpoint_a = [
                                    endpoint_factor[k][0] * cov_row[c],
                                    endpoint_factor[k][1] * cov_row[c],
                                ];
                                let endpoint_psi_a = [
                                    endpoint_psi_cov_factor[k][0] * cov_row[c]
                                        + endpoint_psi_psi_factor[k][0] * psi_row[c],
                                    endpoint_psi_cov_factor[k][1] * cov_row[c]
                                        + endpoint_psi_psi_factor[k][1] * psi_row[c],
                                ];
                                score_axis[idx] += wi
                                    * (h_a * h_psi + hi * hpsi_a - hppsi_a * inv_hp
                                        + hp_psi * hp_a * inv_hp_sq
                                        + endpoint_chain_second(
                                            q,
                                            endpoint_psi,
                                            endpoint_a,
                                            endpoint_psi_a,
                                        ));
                            }
                        }
                    }
                }
                Ok(acc)
            })
            .collect();
        let mut accum = PsiAllAxesAccum::new(n_psi, p_total);
        for partial in partials {
            accum = accum.merge(partial?);
        }

        // Stash the cached numeric data plus per-axis operator handles. The
        // matrix-free `TransformationNormalPsiHessianOperator` is reconstructed
        // on each `first_order_terms()` call from the cached Arc-shared row
        // state — Arc clones are O(1) and the cached operator instance carries
        // no per-evaluation mutable state.
        let PsiAllAxesAccum {
            objective_psi,
            mut score_psi,
        } = accum;
        let beta_arc = Arc::new(beta.clone());
        let trace_axes = Arc::new(axes.clone());
        let mut out: Vec<TransformationNormalPsiWorkspaceCacheEntry> = Vec::with_capacity(n_psi);
        for (axis_idx, &axis) in axes.iter().enumerate() {
            // Take the per-axis score buffer out of the accumulator without
            // cloning. The order matches the construction order so each axis
            // is consumed exactly once.
            let score_axis = std::mem::replace(&mut score_psi[axis_idx], Array1::<f64>::zeros(0));
            out.push(TransformationNormalPsiWorkspaceCacheEntry {
                objective_psi: objective_psi[axis_idx],
                score_psi: score_axis,
                op_arc: Arc::clone(&op_arcs[axis_idx]),
                axis,
                trace_axes: Arc::clone(&trace_axes),
                trace_axis_pos: axis_idx,
                row_gamma: Arc::clone(&row.gamma),
                row_h: Arc::clone(&row.h),
                row_h_prime: Arc::clone(&row.h_prime),
                endpoint_q: Arc::clone(&row.endpoint_q),
                beta: Arc::clone(&beta_arc),
            });
        }
        Ok(out)
    }

    fn axis_snapshots(&self) -> Result<Vec<TransformationNormalPsiWorkspaceAxisSnapshot>, String> {
        let mut guard = self
            .cache
            .lock()
            .map_err(|_| "TransformationNormalPsiWorkspace cache poisoned".to_string())?;
        if guard.is_none() {
            let computed = self.compute_all_axes()?;
            *guard = Some(computed);
        }
        let cached = guard.as_ref().expect("populated above");
        Ok(cached
            .iter()
            .map(|entry| TransformationNormalPsiWorkspaceAxisSnapshot {
                op_arc: Arc::clone(&entry.op_arc),
                axis: entry.axis,
                row_gamma: Arc::clone(&entry.row_gamma),
                row_h: Arc::clone(&entry.row_h),
                row_h_prime: Arc::clone(&entry.row_h_prime),
                endpoint_q: Arc::clone(&entry.endpoint_q),
                beta: Arc::clone(&entry.beta),
            })
            .collect())
    }

    fn compute_pair_cache(
        &self,
    ) -> Result<Vec<Vec<Option<Arc<TransformationNormalPsiWorkspacePairCacheEntry>>>>, String> {
        let axes = self.axis_snapshots()?;
        let n_psi = axes.len();
        if n_psi == 0 {
            return Ok(Vec::new());
        }

        let pair_count = n_psi * (n_psi + 1) / 2;
        let pair_from_index = |pair_idx: usize| -> (usize, usize) {
            let span = 2 * n_psi + 1;
            let discriminant = span * span - 8 * pair_idx;
            let row = ((span as f64 - (discriminant as f64).sqrt()) * 0.5) as usize;
            let row_start = row * (2 * n_psi - row + 1) / 2;
            (row, row + pair_idx - row_start)
        };
        let trace_axes = Arc::new(axes.iter().map(|entry| entry.axis).collect::<Vec<_>>());

        let op = axes[0]
            .op_arc
            .as_any()
            .downcast_ref::<TensorKroneckerPsiOperator>()
            .ok_or_else(|| {
                "TransformationNormalPsiWorkspace psi-psi pair cache requires tensor-backed operator"
                    .to_string()
            })?;
        for (psi_index, entry) in axes.iter().enumerate() {
            if entry
                .op_arc
                .as_any()
                .downcast_ref::<TensorKroneckerPsiOperator>()
                .is_none()
            {
                return Err(TransformationNormalError::InvalidInput { reason: format!(
                    "TransformationNormalPsiWorkspace psi-psi pair cache requires tensor-backed operator at axis {psi_index}"
                ) }.into());
            }
        }

        let n = self.family.response_val_basis.nrows();
        let p_cov = self.family.covariate_design.ncols();
        let p_total = self.family.response_val_basis.ncols() * p_cov;
        let policy = ResourcePolicy::default_library();
        let rows_per_chunk = crate::resource::rows_for_target_bytes(
            policy.row_chunk_target_bytes,
            p_cov.saturating_mul(n_psi + 2).max(1),
        )
        .max(1)
        .min(n.max(1));

        struct PsiPairCacheAccum {
            objective: f64,
            score: Array1<f64>,
        }

        impl PsiPairCacheAccum {
            fn new(p_total: usize) -> Self {
                Self {
                    objective: 0.0,
                    score: Array1::<f64>::zeros(p_total),
                }
            }
        }

        let mut accum: Vec<PsiPairCacheAccum> = (0..pair_count)
            .map(|_| PsiPairCacheAccum::new(p_total))
            .collect();
        for start in (0..n).step_by(rows_per_chunk) {
            let end = (start + rows_per_chunk).min(n);
            let rows = start..end;
            let cov = self
                .family
                .covariate_design
                .try_row_chunk(rows.clone())
                .map_err(|e| {
                    format!(
                        "TransformationNormalPsiWorkspace psi-psi pair cache covariate chunk {start}..{end}: {e}"
                    )
                })?;
            let mut cov_psi_chunks: Vec<Array2<f64>> = Vec::with_capacity(n_psi);
            for (psi_index, entry) in axes.iter().enumerate() {
                let cov_psi = op
                    .cov_first_axis_row_chunk_streaming(entry.axis, rows.clone())
                    .map_err(|e| {
                        format!(
                            "TransformationNormalPsiWorkspace psi-psi pair cache first-axis chunk \
                             psi_index={psi_index}, axis={} rows {start}..{end}: {e}",
                            entry.axis
                        )
                    })?;
                if cov_psi.nrows() != end - start || cov_psi.ncols() != p_cov {
                    return Err(TransformationNormalError::InvalidInput { reason: format!(
                        "TransformationNormalPsiWorkspace psi-psi pair cache first-axis chunk shape {}x{} \
                         for psi_index={psi_index}, axis={} rows {start}..{end} != expected {}x{}",
                        cov_psi.nrows(),
                        cov_psi.ncols(),
                        entry.axis,
                        end - start,
                        p_cov
                    ) }.into());
                }
                cov_psi_chunks.push(cov_psi);
            }

            for pair_idx in 0..pair_count {
                let (psi_i, psi_j) = pair_from_index(pair_idx);
                let entry_i = &axes[psi_i];
                let entry_j = &axes[psi_j];
                let cov_ij = op
                    .cov_second_axis_row_chunk(entry_i.axis, entry_j.axis, rows.clone())
                    .map_err(|e| {
                        format!(
                            "TransformationNormalPsiWorkspace psi-psi pair cache second-axis chunk \
                             pair=({psi_i},{psi_j}), axes=({}, {}) rows {start}..{end}: {e}",
                            entry_i.axis, entry_j.axis
                        )
                    })?;
                if cov_ij.nrows() != end - start || cov_ij.ncols() != p_cov {
                    return Err(TransformationNormalError::InvalidInput { reason: format!(
                        "TransformationNormalPsiWorkspace psi-psi pair cache second-axis chunk shape {}x{} \
                         for pair=({psi_i},{psi_j}), axes=({}, {}) rows {start}..{end} != expected {}x{}",
                        cov_ij.nrows(),
                        cov_ij.ncols(),
                        entry_i.axis,
                        entry_j.axis,
                        end - start,
                        p_cov
                    ) }.into());
                }
                let (objective_chunk, score_chunk, _) =
                    self.family.scop_psi_psi_value_score_hvp_from_cov(
                        entry_i.beta.as_ref(),
                        entry_i.row_gamma.slice(s![start..end, ..]),
                        entry_i.row_h.slice(s![start..end]),
                        entry_i.row_h_prime.slice(s![start..end]),
                        cov.view(),
                        cov_psi_chunks[psi_i].view(),
                        cov_psi_chunks[psi_j].view(),
                        cov_ij.view(),
                        start,
                        &entry_i.endpoint_q[start..end],
                        None,
                    )?;
                accum[pair_idx].objective += objective_chunk;
                accum[pair_idx].score.scaled_add(1.0, &score_chunk);
            }
        }

        let mut table = vec![vec![None; n_psi]; n_psi];
        for (pair_idx, acc) in accum.into_iter().enumerate() {
            let (i, j) = pair_from_index(pair_idx);
            if !acc.objective.is_finite() || !acc.score.iter().all(|v: &f64| v.is_finite()) {
                return Err(TransformationNormalError::NonFinite { reason: format!(
                    "TransformationNormalPsiWorkspace psi-psi pair cache produced non-finite values at \
                     psi_i={i}, psi_j={j}: obj_finite={}, score_all_finite={}",
                    acc.objective.is_finite(),
                    acc.score.iter().all(|v: &f64| v.is_finite()),
                ) }.into());
            }
            let entry_i = &axes[i];
            let entry_j = &axes[j];
            let entry = Arc::new(TransformationNormalPsiWorkspacePairCacheEntry {
                objective_psi_psi: acc.objective,
                score_psi_psi: acc.score,
                op_arc: Arc::clone(&entry_i.op_arc),
                axis_i: entry_i.axis,
                axis_j: entry_j.axis,
                trace_axes: Arc::clone(&trace_axes),
                trace_axis_i_pos: i,
                trace_axis_j_pos: j,
                row_gamma: Arc::clone(&entry_i.row_gamma),
                row_h: Arc::clone(&entry_i.row_h),
                row_h_prime: Arc::clone(&entry_i.row_h_prime),
                endpoint_q: Arc::clone(&entry_i.endpoint_q),
                beta: Arc::clone(&entry_i.beta),
            });
            table[i][j] = Some(Arc::clone(&entry));
            table[j][i] = Some(entry);
        }
        Ok(table)
    }
}

impl ExactNewtonJointPsiWorkspace for TransformationNormalPsiWorkspace {
    fn first_order_terms(
        &self,
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        let mut guard = self
            .cache
            .lock()
            .map_err(|_| "TransformationNormalPsiWorkspace cache poisoned".to_string())?;
        if guard.is_none() {
            let computed = self.compute_all_axes()?;
            *guard = Some(computed);
        }
        let cached = guard.as_ref().expect("populated above");
        if psi_index >= cached.len() {
            return Ok(None);
        }
        let entry = &cached[psi_index];
        // Reconstruct the matrix-free first-order Hessian operator on each
        // call. Arc-cloning shared row state and `op_arc` is O(1); the
        // operator carries no mutable per-evaluation state. The cached
        // numeric `score_psi` buffer is cloned because `ExactNewtonJointPsiTerms`
        // is not `Clone`-derivable through the `dyn HyperOperator` field.
        let hessian_psi_operator: Arc<dyn HyperOperator> =
            Arc::new(TransformationNormalPsiHessianOperator::new_with_trace_axes(
                Arc::new(self.family.clone()),
                (*entry.beta).clone(),
                Arc::clone(&entry.op_arc),
                entry.axis,
                Arc::clone(&entry.trace_axes),
                entry.trace_axis_pos,
                Arc::clone(&entry.row_gamma),
                Arc::clone(&entry.row_h),
                Arc::clone(&entry.row_h_prime),
                Arc::clone(&entry.endpoint_q),
            ));
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi: entry.objective_psi,
            score_psi: entry.score_psi.clone(),
            hessian_psi: Array2::zeros((0, 0)),
            hessian_psi_operator: Some(hessian_psi_operator),
        }))
    }

    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let start = std::time::Instant::now();
        let entry = {
            let mut guard = self
                .pair_cache
                .lock()
                .map_err(|_| "TransformationNormalPsiWorkspace pair cache poisoned".to_string())?;
            if guard.is_none() {
                let computed = self.compute_pair_cache()?;
                *guard = Some(computed);
            }
            let cached = guard.as_ref().expect("populated above");
            if psi_i >= cached.len() || psi_j >= cached.len() {
                return Ok(None);
            }
            cached[psi_i][psi_j].as_ref().map(Arc::clone)
        };
        let Some(entry) = entry else {
            return Ok(None);
        };

        let hessian_psi_psi_operator: Box<dyn HyperOperator> = Box::new(
            TransformationNormalPsiPsiHessianOperator::new_with_trace_axes(
                Arc::new(self.family.clone()),
                entry.beta.as_ref().clone(),
                Arc::clone(&entry.op_arc),
                entry.axis_i,
                entry.axis_j,
                Arc::clone(&entry.trace_axes),
                entry.trace_axis_i_pos,
                entry.trace_axis_j_pos,
                Arc::clone(&entry.row_gamma),
                Arc::clone(&entry.row_h),
                Arc::clone(&entry.row_h_prime),
                Arc::clone(&entry.endpoint_q),
            ),
        );
        log::info!(
            "[STAGE] CTN psi-psi workspace pair (psi_i={}, psi_j={}, axes={},{}) elapsed={:.3}s",
            psi_i,
            psi_j,
            entry.axis_i,
            entry.axis_j,
            start.elapsed().as_secs_f64(),
        );
        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi: entry.objective_psi_psi,
            score_psi_psi: entry.score_psi_psi.clone(),
            hessian_psi_psi: Array2::zeros((0, 0)),
            hessian_psi_psi_operator: Some(hessian_psi_psi_operator),
        }))
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<DriftDerivResult>, String> {
        let mut guard = self
            .cache
            .lock()
            .map_err(|_| "TransformationNormalPsiWorkspace cache poisoned".to_string())?;
        if guard.is_none() {
            let computed = self.compute_all_axes()?;
            *guard = Some(computed);
        }
        let cached = guard.as_ref().expect("populated above");
        if psi_index >= cached.len() {
            return Ok(None);
        }
        let entry = &cached[psi_index];
        if d_beta_flat.len() != entry.beta.len() {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "TransformationNormalPsiWorkspace psi dH direction length {} != expected {}",
                    d_beta_flat.len(),
                    entry.beta.len()
                ),
            }
            .into());
        }
        let row_quantities = TransformationNormalRowQuantityCache {
            beta: Arc::clone(&entry.beta),
            gamma: Arc::clone(&entry.row_gamma),
            h: Arc::clone(&entry.row_h),
            h_prime: Arc::clone(&entry.row_h_prime),
            h_lower: Arc::new(Array1::zeros(entry.row_h.len())),
            h_upper: Arc::new(Array1::zeros(entry.row_h.len())),
            endpoint_q: Arc::clone(&entry.endpoint_q),
            log_likelihood: 0.0,
            // Reconstructed from psi-cached row jets; not derived from the
            // family's `row_quantities` builder. Version 0 (the
            // "never built" sentinel) keeps any persistent-dense-Hessian
            // entry built from this instance distinct from production builds.
            version: 0,
        };
        let op = TransformationNormalPsiDhMatrixFreeOperator::new(
            Arc::new(self.family.clone()),
            entry.beta.as_ref().clone(),
            d_beta_flat.clone(),
            Arc::clone(&entry.op_arc),
            entry.axis,
            row_quantities,
        );
        Ok(Some(DriftDerivResult::Operator(Arc::new(op))))
    }
}

fn extract_covariate_penalty_factor(penalty: &PenaltyMatrix) -> Result<Array2<f64>, String> {
    match penalty {
        PenaltyMatrix::Dense(matrix) => Ok(matrix.clone()),
        PenaltyMatrix::Blockwise { .. } => Ok(penalty.to_dense()),
        PenaltyMatrix::Labeled { inner, .. } => extract_covariate_penalty_factor(inner),
        PenaltyMatrix::KroneckerFactored { .. } => Err(
            "transformation covariate psi penalties must be single-block, not already Kronecker-factored"
                .to_string(),
        ),
    }
}

// ---------------------------------------------------------------------------
// Psi derivative builder (for κ optimization of the covariate basis)
// ---------------------------------------------------------------------------

/// Build `CustomFamilyBlockPsiDerivative` objects for the tensor product.
///
/// Given covariate-side psi derivatives (∂cov_design/∂ψ_a and ∂S_cov/∂ψ_a),
/// this constructs the corresponding tensor-product-space psi derivatives that
/// the REML evaluator needs.
///
/// Each output entry contains:
/// - implicit `x_psi` / `x_psi_psi` operators that preserve Kronecker structure
/// - factored tensor penalty derivatives matching the SCOP penalty layout:
///   `E_shape ⊗ ∂S_cov/∂ψ` at the same covariate penalty index `m`.
pub fn build_tensor_psi_derivatives(
    family: &TransformationNormalFamily,
    covariate_psi_derivs: &[CustomFamilyBlockPsiDerivative],
) -> Result<Vec<CustomFamilyBlockPsiDerivative>, String> {
    let p_resp = family.response_val_basis.ncols();
    let n_axes = covariate_psi_derivs.len();
    let mut shape_resp = Array2::<f64>::eye(p_resp);
    shape_resp[[0, 0]] = 0.0;
    let shared_operator: Arc<dyn CustomFamilyPsiDerivativeOperator> =
        Arc::new(TensorKroneckerPsiOperator {
            response_val_basis: Arc::new(family.response_val_basis.clone()),
            covariate_design: family.covariate_design.clone(),
            covariate_derivs: covariate_psi_derivs.to_vec(),
            covariate_first_cache: Arc::new(
                (0..n_axes).map(|_| Mutex::new(None)).collect::<Vec<_>>(),
            ),
        });

    let mut derivs = Vec::with_capacity(n_axes);
    for a in 0..n_axes {
        let cov_deriv = &covariate_psi_derivs[a];
        let s_psi_penalty_components = cov_deriv
            .s_psi_penalty_components
            .as_ref()
            .map(|components| lift_covariate_penalty_derivative_components(components, &shape_resp))
            .transpose()?
            .or_else(|| {
                cov_deriv.s_psi_components.as_ref().map(|components| {
                    lift_dense_covariate_penalty_derivative_components(components, &shape_resp)
                })
            });
        let s_psi_psi_penalty_components = cov_deriv
            .s_psi_psi_penalty_components
            .as_ref()
            .map(|rows| {
                rows.iter()
                    .map(|cov_pen_pairs| -> Result<_, String> {
                        lift_covariate_penalty_derivative_components(cov_pen_pairs, &shape_resp)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .or_else(|| {
                cov_deriv.s_psi_psi_components.as_ref().map(|rows| {
                    rows.iter()
                        .map(|cov_pen_pairs| {
                            lift_dense_covariate_penalty_derivative_components(
                                cov_pen_pairs,
                                &shape_resp,
                            )
                        })
                        .collect::<Vec<_>>()
                })
            });

        let mut deriv = CustomFamilyBlockPsiDerivative::new(
            None,
            Array2::<f64>::zeros((0, 0)),
            Array2::<f64>::zeros((0, 0)),
            None,
            None,
            None,
            None,
        );
        deriv.s_psi_penalty_components = s_psi_penalty_components;
        deriv.s_psi_psi_penalty_components = s_psi_psi_penalty_components;
        deriv.implicit_operator = Some(Arc::clone(&shared_operator));
        deriv.implicit_axis = a;
        deriv.implicit_group_id = Some(0);
        derivs.push(deriv);
    }

    Ok(derivs)
}

fn lift_dense_covariate_penalty_derivative_components(
    components: &[(usize, Array2<f64>)],
    shape_resp: &Array2<f64>,
) -> Vec<(usize, PenaltyMatrix)> {
    let mut out = Vec::with_capacity(components.len());
    for &(idx, ref ds_cov) in components {
        push_lifted_covariate_penalty_component(&mut out, idx, ds_cov.clone(), shape_resp);
    }
    out
}

fn lift_covariate_penalty_derivative_components(
    components: &[(usize, PenaltyMatrix)],
    shape_resp: &Array2<f64>,
) -> Result<Vec<(usize, PenaltyMatrix)>, String> {
    let mut out = Vec::with_capacity(components.len());
    for (idx, ds_cov) in components {
        push_lifted_covariate_penalty_component(
            &mut out,
            *idx,
            extract_covariate_penalty_factor(ds_cov)?,
            shape_resp,
        );
    }
    Ok(out)
}

fn push_lifted_covariate_penalty_component(
    out: &mut Vec<(usize, PenaltyMatrix)>,
    cov_penalty_idx: usize,
    ds_cov: Array2<f64>,
    shape_resp: &Array2<f64>,
) {
    out.push((
        cov_penalty_idx,
        PenaltyMatrix::KroneckerFactored {
            left: shape_resp.clone(),
            right: ds_cov,
        },
    ));
}

#[derive(Clone)]
struct TransformationExactGeometryCache {
    key: Vec<u64>,
    covariate_spec_resolved: TermCollectionSpec,
    covariate_design: TermCollectionDesign,
    family: TransformationNormalFamily,
    blocks: Vec<ParameterBlockSpec>,
    derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
}

#[derive(Clone)]
struct TransformationExactWarmStart {
    theta: Array1<f64>,
    warm_start: CustomFamilyWarmStart,
}

impl TransformationExactWarmStart {
    fn is_compatible_with(&self, theta: &Array1<f64>, rho: &Array1<f64>) -> bool {
        const MAX_THETA_DISTANCE: f64 = 1.5;

        self.theta.len() == theta.len()
            && self
                .theta
                .iter()
                .zip(theta.iter())
                .all(|(&a, &b)| (a - b).abs() <= MAX_THETA_DISTANCE)
            && self.warm_start.compatible_with_rho(rho)
    }
}

impl TransformationExactGeometryCache {
    fn update_initial_log_lambdas(&mut self, log_lambdas: &Array1<f64>) -> Result<(), String> {
        let spec = self
            .blocks
            .first_mut()
            .ok_or_else(|| "missing transformation block spec".to_string())?;
        if log_lambdas.len() != spec.initial_log_lambdas.len() {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "transformation final fit rho length mismatch: got {}, expected {}",
                    log_lambdas.len(),
                    spec.initial_log_lambdas.len()
                ),
            }
            .into());
        }
        spec.initial_log_lambdas = log_lambdas.clone();
        Ok(())
    }
}

fn transformation_spatial_geometry_key(
    spec: &TermCollectionSpec,
    spatial_terms: &[usize],
) -> Result<Vec<u64>, String> {
    let mut key = Vec::new();
    key.push(spatial_terms.len() as u64);
    for &term_idx in spatial_terms {
        let term = spec.smooth_terms.get(term_idx).ok_or_else(|| {
            format!(
                "transformation spatial geometry key term index {term_idx} out of range for {} smooth terms",
                spec.smooth_terms.len()
            )
        })?;
        key.push(term_idx as u64);

        // The CTN exact-family cache is valid only for an identical covariate
        // geometry. Length-scale and anisotropy scalars are not enough: the
        // family also embeds frozen centers, input standardization scales,
        // identifiability transforms, and active penalty topology. Serialize
        // the already-frozen term and store the exact bytes in the key so a
        // cache hit means the saved prediction design will replay the same
        // matrix used by the final inner fit.
        let payload = serde_json::to_vec(term).map_err(|err| {
            format!("failed to serialize transformation spatial geometry term {term_idx}: {err}")
        })?;
        key.push(payload.len() as u64);
        for chunk in payload.chunks(8) {
            let mut bytes = [0u8; 8];
            for (dst, src) in bytes.iter_mut().zip(chunk.iter().copied()) {
                *dst = src;
            }
            key.push(u64::from_le_bytes(bytes));
        }
    }
    Ok(key)
}

// ---------------------------------------------------------------------------
// Top-level fit function
// ---------------------------------------------------------------------------

/// Result of `fit_transformation_normal`.
#[derive(Clone)]
pub struct TransformationNormalFitResult {
    pub family: TransformationNormalFamily,
    pub fit: UnifiedFitResult,
    pub covariate_spec_resolved: TermCollectionSpec,
    pub covariate_design: TermCollectionDesign,
    pub score_calibration: TransformationScoreCalibration,
}

/// Fit a conditional transformation model with N-block spatial length-scale
/// optimization over the covariate side.
///
/// The response-direction basis is built once (it does not depend on κ).
/// If no spatial length-scale terms are present in the covariate spec, the
/// model is fit directly. Otherwise, the N-block joint hyper-parameter
/// optimizer is used with a single block (the covariate spec).
pub fn fit_transformation_normal(
    response: &Array1<f64>,
    weights: &Array1<f64>,
    offset: &Array1<f64>,
    covariate_data: ArrayView2<'_, f64>,
    covariate_spec: &TermCollectionSpec,
    config: &TransformationNormalConfig,
    options: &BlockwiseFitOptions,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
    warm_start: Option<&TransformationWarmStart>,
) -> Result<TransformationNormalFitResult, String> {
    let options = options.clone();
    // CTN advertises profiled outer-Hessian HVP support and supplies the
    // callback derivative kernel consumed by the unified REML/LAML evaluator.
    // Keep analytic curvature enabled here: the evaluator routes CTN Hessians
    // through the matrix-free operator path instead of dense pairwise assembly.
    let covariate_spec = covariate_spec.clone();

    // 1. Build a bootstrap covariate design first so the response basis can
    // adapt to the tensor width instead of always using the global default.
    let boot_design = build_term_collection_design(covariate_data, &covariate_spec)
        .map_err(|e| format!("failed to build bootstrap covariate design: {e}"))?;
    let boot_spec = freeze_term_collection_from_design(&covariate_spec, &boot_design)
        .map_err(|e| format!("failed to freeze bootstrap covariate spatial basis centers: {e}"))?;
    let mut effective_config = config.clone();
    effective_config.response_num_internal_knots =
        effective_response_num_internal_knots(config, response.len(), boot_design.design.ncols());

    // 2. Build response basis ONCE — it is independent of κ once the effective
    // response complexity has been chosen.
    let (resp_val, resp_deriv, resp_penalties, resp_knots, resp_transform) =
        build_response_basis(response, &effective_config)?;

    // 3. Check whether spatial κ optimization is needed.
    let spatial_terms = spatial_length_scale_term_indices(&covariate_spec);

    if spatial_terms.is_empty() || !kappa_options.enabled {
        // ------------------------------------------------------------------
        // NO κ: build family directly, fit, return.
        // ------------------------------------------------------------------
        let cov_design = boot_design;
        let cov_spec_resolved = boot_spec;

        let family = TransformationNormalFamily::from_prebuilt_response_basis(
            response,
            resp_val,
            resp_deriv,
            resp_penalties,
            resp_knots.clone(),
            effective_config.response_degree,
            resp_transform,
            weights,
            offset,
            cov_design.design.clone(),
            cov_design
                .penalties
                .iter()
                .map(|bp| PenaltyMatrix::from_blockwise(bp.clone(), cov_design.design.ncols()))
                .collect(),
            &effective_config,
            warm_start,
        )?;
        let blocks = vec![family.block_spec()];
        let fit = fit_custom_family(&family, &blocks, &options)
            .map_err(|e| format!("transformation fit failed: {e}"))?;
        let (fit, score_calibration) = calibrate_transformation_scores(&family, fit)?;

        return Ok(TransformationNormalFitResult {
            family,
            fit,
            covariate_spec_resolved: cov_spec_resolved,
            covariate_design: cov_design,
            score_calibration,
        });
    }

    // ------------------------------------------------------------------
    // YES κ: use the N-block spatial length-scale optimizer (1 block).
    // ------------------------------------------------------------------

    let kappa0 = SpatialLogKappaCoords::from_length_scales_aniso(
        &covariate_spec,
        &spatial_terms,
        kappa_options,
    )
    .reseed_from_data(
        covariate_data,
        &covariate_spec,
        &spatial_terms,
        kappa_options,
    );
    let kappa_dims = kappa0.dims_per_term().to_vec();
    let kappa_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        covariate_data,
        &covariate_spec,
        &spatial_terms,
        &kappa_dims,
        kappa_options,
    );
    let kappa_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        covariate_data,
        &covariate_spec,
        &spatial_terms,
        &kappa_dims,
        kappa_options,
    );
    // Project seed onto bounds; spec.length_scale is a hint, not a constraint.
    let kappa0 = kappa0.clamp_to_bounds(&kappa_lower, &kappa_upper);

    // Check analytic derivative capability.
    let analytic_psi_available =
        build_block_spatial_psi_derivatives(covariate_data, &boot_spec, &boot_design)?.is_some();

    // Build an initial family + blocks for capability probing.
    let probe_family = TransformationNormalFamily::from_prebuilt_response_basis(
        response,
        resp_val.clone(),
        resp_deriv.clone(),
        resp_penalties.clone(),
        resp_knots.clone(),
        effective_config.response_degree,
        resp_transform.clone(),
        weights,
        offset,
        boot_design.design.clone(),
        boot_design
            .penalties
            .iter()
            .map(|bp| PenaltyMatrix::from_blockwise(bp.clone(), boot_design.design.ncols()))
            .collect(),
        &effective_config,
        warm_start,
    )?;
    let probe_block = probe_family.block_spec();
    let n_penalties = probe_block.initial_log_lambdas.len();
    log::info!(
        "[transformation-normal] exact joint setup: rho_dim={} log_kappa_dim={} dims_per_term={:?}",
        n_penalties,
        kappa0.len(),
        kappa_dims,
    );
    let rho0 = probe_block.initial_log_lambdas.clone();
    let rho_floor = -12.0;
    let rho_lower = Array1::<f64>::from_elem(n_penalties, rho_floor);
    let rho_upper = Array1::<f64>::from_elem(n_penalties, 12.0);
    let probe_blocks = vec![probe_block.clone()];
    let (_, cap_hessian) = crate::families::custom_family::custom_family_outer_derivatives(
        &probe_family,
        &probe_blocks,
        &options,
    );
    let analytic_gradient = analytic_psi_available;
    let analytic_hessian_supported = analytic_gradient && cap_hessian.is_analytic();
    let analytic_hessian = false;
    if analytic_hessian_supported {
        log::info!(
            "[transformation-normal] CTN exact joint analytic outer Hessian is available but disabled for spatial kappa optimization; using analytic-gradient outer solves to avoid callback logdet trace work"
        );
    }

    let (rho0_min, rho0_max) = if rho0.is_empty() {
        (0.0, 0.0)
    } else {
        (
            rho0.iter().copied().fold(f64::INFINITY, f64::min),
            rho0.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    log::info!(
        "[transformation-normal] skipping baseline custom-family prefit before exact joint optimization \
         (rho_dim={}, log_kappa_dim={}, rho0_range=[{:.3}, {:.3}]); using CTN warm start and penalty-scale rho seed",
        n_penalties,
        kappa0.len(),
        rho0_min,
        rho0_max,
    );

    if !analytic_psi_available {
        return Err(
            "transformation-normal spatial length-scale optimization requires analytic spatial psi derivatives"
                .to_string(),
        );
    }

    // Shared mutable state for warm-starting across optimizer iterations.
    let exact_warm_start: RefCell<Option<TransformationExactWarmStart>> = RefCell::new(None);

    let joint_setup =
        ExactJointHyperSetup::new(rho0, rho_lower, rho_upper, kappa0, kappa_lower, kappa_upper);

    // Clone response basis parts for use inside closures.
    let rv = resp_val.clone();
    let rd = resp_deriv.clone();
    let rp = resp_penalties.clone();
    let rk = resp_knots.clone();
    let rt = resp_transform.clone();
    let rdeg = effective_config.response_degree;
    let cfg = effective_config.clone();
    let ws = warm_start.cloned();

    // Helper: build family from prebuilt response basis + covariate design.
    let make_family =
        |cov_design: &TermCollectionDesign| -> Result<TransformationNormalFamily, String> {
            TransformationNormalFamily::from_prebuilt_response_basis(
                response,
                rv.clone(),
                rd.clone(),
                rp.clone(),
                rk.clone(),
                rdeg,
                rt.clone(),
                weights,
                offset,
                cov_design.design.clone(),
                cov_design
                    .penalties
                    .iter()
                    .map(|bp| PenaltyMatrix::from_blockwise(bp.clone(), cov_design.design.ncols()))
                    .collect(),
                &cfg,
                ws.as_ref(),
            )
        };

    let block_specs_slice = [boot_spec.clone()];
    let block_term_indices_slice = [spatial_terms.clone()];
    let exact_geometry_cache: RefCell<Option<TransformationExactGeometryCache>> =
        RefCell::new(None);
    let spatial_terms_for_cache = spatial_terms.clone();

    let ensure_exact_geometry = |spec: &TermCollectionSpec,
                                 design: &TermCollectionDesign|
     -> Result<(), String> {
        let effective_spec = freeze_term_collection_from_design(spec, design)
            .map_err(|e| format!("failed to freeze transformation geometry key: {e}"))?;
        let key = transformation_spatial_geometry_key(&effective_spec, &spatial_terms_for_cache)?;
        let needs_rebuild = exact_geometry_cache
            .borrow()
            .as_ref()
            .map(|cached| cached.key != key)
            .unwrap_or(true);
        if !needs_rebuild {
            return Ok(());
        }

        let geom_start = std::time::Instant::now();
        let exact_design = build_term_collection_design(covariate_data, &effective_spec)
            .map_err(|e| format!("failed to rebuild frozen transformation geometry: {e}"))?;
        let family = make_family(&exact_design)?;
        let cov_psi_derivs =
            build_block_spatial_psi_derivatives(covariate_data, &effective_spec, &exact_design)?
                .ok_or_else(|| {
                    "missing covariate spatial psi derivatives for transformation model".to_string()
                })?;
        let tensor_derivs = build_tensor_psi_derivatives(&family, &cov_psi_derivs)?;

        log::debug!(
            "[transformation-normal] rebuilt exact geometry cache for {} spatial terms in {:.3}s",
            spatial_terms_for_cache.len(),
            geom_start.elapsed().as_secs_f64(),
        );

        exact_geometry_cache.replace(Some(TransformationExactGeometryCache {
            key,
            covariate_spec_resolved: effective_spec,
            covariate_design: exact_design,
            blocks: vec![family.block_spec()],
            family,
            derivative_blocks: vec![tensor_derivs],
        }));
        Ok(())
    };

    let compatible_warm_start =
        |theta: &Array1<f64>, rho: &Array1<f64>| -> Option<CustomFamilyWarmStart> {
            exact_warm_start
                .borrow()
                .as_ref()
                .filter(|warm| warm.is_compatible_with(theta, rho))
                .map(|warm| warm.warm_start.clone())
        };
    let store_warm_start = |theta: &Array1<f64>, warm_start: CustomFamilyWarmStart| {
        exact_warm_start
            .borrow_mut()
            .replace(TransformationExactWarmStart {
                theta: theta.clone(),
                warm_start,
            });
    };

    log::info!(
        "[transformation-normal] entering exact joint outer optimization \
         (analytic_gradient={}, analytic_hessian={})",
        analytic_gradient,
        analytic_hessian,
    );
    // Outer derivative policy (P2.3): consult the family's CTN-specific
    // override so the cost gate uses the Khatri–Rao row-streamed shape
    // (`O(n · (rho + psi) · p)` gradient; `min(dense, mfree)` Hessian)
    // rather than the generic `coefficient_*_cost × K` default.
    let outer_derivative_policy =
        probe_family.outer_derivative_policy(&probe_blocks, joint_setup.log_kappa_dim(), &options);

    let solved = optimize_spatial_length_scale_exact_joint(
        covariate_data,
        &block_specs_slice,
        &block_term_indices_slice,
        kappa_options,
        &joint_setup,
        crate::seeding::SeedRiskProfile::Gaussian,
        analytic_gradient,
        analytic_hessian,
        // Transformation-normal has β-dependent H (through 1/h'²), so the
        // EFS Wood-Fasiolo PSD invariant fails. Keep fixed-point disabled,
        // but do not expose CTN's analytic Hessian to ARC: its callback
        // trace route applies full-rank logdet operators at biobank shape.
        true,
        None,
        outer_derivative_policy,
        // fit_fn
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            ensure_exact_geometry(&specs[0], &designs[0])?;
            let mut cache_ref = exact_geometry_cache.borrow_mut();
            let geometry = cache_ref
                .as_mut()
                .ok_or_else(|| "missing transformation exact geometry cache".to_string())?;
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            geometry.update_initial_log_lambdas(&rho)?;
            let warm_start = compatible_warm_start(theta, &rho);
            let fit = fit_custom_family_fixed_log_lambdas(
                &geometry.family,
                &geometry.blocks,
                &options,
                warm_start.as_ref(),
                0,
                None,
                true,
            )
            .map_err(|e| format!("transformation fit_fn: {e}"))?;
            if let Some(block) = fit.block_states.first() {
                *geometry
                    .family
                    .row_quantity_cache
                    .lock()
                    .expect("CTN row quantity cache mutex poisoned") = None;
                let final_rows = geometry.family.row_quantities(&block.beta)?;
                let max_abs_h = final_rows
                    .h
                    .iter()
                    .copied()
                    .map(f64::abs)
                    .fold(0.0, f64::max);
                let cov_chunk = geometry
                    .family
                    .covariate_design
                    .try_row_chunk(0..response.len())
                    .map_err(|err| {
                        format!("final CTN covariate design validation failed: {err}")
                    })?;
                let max_abs_cov = cov_chunk.iter().copied().map(f64::abs).fold(0.0, f64::max);
                log::info!(
                    "[transformation-normal] final fixed-rho CTN validation: max_abs_h={:.6e} max_abs_covariate_basis={:.6e}",
                    max_abs_h,
                    max_abs_cov
                );
            }
            Ok(TransformationNormalFitResult {
                family: geometry.family.clone(),
                fit,
                covariate_spec_resolved: geometry.covariate_spec_resolved.clone(),
                covariate_design: geometry.covariate_design.clone(),
                score_calibration: TransformationScoreCalibration::finite_support_pit(),
            })
        },
        // exact_fn
        |theta,
         specs: &[TermCollectionSpec],
         designs: &[TermCollectionDesign],
         eval_mode,
         row_set| {
            drop(row_set);
            ensure_exact_geometry(&specs[0], &designs[0])?;
            let mut cache_ref = exact_geometry_cache.borrow_mut();
            let geometry = cache_ref
                .as_mut()
                .ok_or_else(|| "missing transformation exact geometry cache".to_string())?;
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let warm_start = compatible_warm_start(theta, &rho);

            let eval = evaluate_custom_family_joint_hyper(
                &geometry.family,
                &geometry.blocks,
                options,
                &rho,
                &geometry.derivative_blocks,
                warm_start.as_ref(),
                eval_mode,
            )
            .map_err(|e| format!("transformation exact_fn: {e}"))?;

            if !eval.objective.is_finite() {
                log::warn!(
                    "transformation exact joint returned non-finite objective: eval_mode={:?} rho={:?} gradient_len={}",
                    eval_mode,
                    rho,
                    eval.gradient.len(),
                );
            }

            if eval.objective.is_finite() && eval.gradient.iter().all(|value| value.is_finite()) {
                store_warm_start(theta, eval.warm_start.clone());
            }

            if !eval.inner_converged {
                return Err(format!(
                    "transformation exact joint inner solve did not converge for eval_mode={eval_mode:?}; cached warm start for retry"
                ));
            }

            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            ensure_exact_geometry(&specs[0], &designs[0])?;
            let mut cache_ref = exact_geometry_cache.borrow_mut();
            let geometry = cache_ref
                .as_mut()
                .ok_or_else(|| "missing transformation exact geometry cache".to_string())?;
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let warm_start = compatible_warm_start(theta, &rho);
            let eval = evaluate_custom_family_joint_hyper_efs(
                &geometry.family,
                &geometry.blocks,
                options,
                &rho,
                &geometry.derivative_blocks,
                warm_start.as_ref(),
            )
            .map_err(|e| format!("transformation exact_efs_fn: {e}"))?;
            store_warm_start(theta, eval.warm_start.clone());
            if !eval.inner_converged {
                return Err(
                    "transformation exact joint EFS inner solve did not converge; cached warm start for retry"
                        .to_string(),
                );
            }
            Ok(eval.efs_eval)
        },
    )?;

    let mut fit = solved.fit;
    let (calibrated_fit, score_calibration) =
        calibrate_transformation_scores(&fit.family, fit.fit.clone())?;
    fit.fit = calibrated_fit;
    fit.score_calibration = score_calibration;
    Ok(fit)
}
