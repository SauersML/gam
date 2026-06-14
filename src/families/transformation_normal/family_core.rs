use endpoint_normalizer::{
    LogNormalCdfDiffDerivatives, endpoint_chain_first, endpoint_chain_fourth,
    endpoint_chain_second, endpoint_chain_third, log_normal_cdf_diff,
    log_normal_cdf_diff_derivatives,
};


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


impl From<crate::util::block_count::BlockCountMismatch> for TransformationNormalError {
    fn from(err: crate::util::block_count::BlockCountMismatch) -> TransformationNormalError {
        TransformationNormalError::InvalidInput {
            reason: err.message(),
        }
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
    /// When true, `response_num_internal_knots` is treated as an already-resolved
    /// effective value: `fit_transformation_normal` uses it verbatim instead of
    /// re-running `effective_response_num_internal_knots`. This is required by the
    /// cross-fit Stage-1 calibration, which pins the knot count once at the
    /// smallest fold complement so `p_resp` (and hence `p₁ = p_resp · p_cov`)
    /// is fold-invariant; the data-driven complexity cap would otherwise round
    /// to different counts on each fold's response subsample (workflow.rs §3).
    pub response_num_internal_knots_pinned: bool,
}


impl Default for TransformationNormalConfig {
    fn default() -> Self {
        Self {
            response_degree: 3,
            response_num_internal_knots: 10,
            response_penalty_order: 2,
            response_extra_penalty_orders: vec![1],
            double_penalty: true,
            response_num_internal_knots_pinned: false,
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
/// At large-scale CTN dimensions p≈800, this keeps the per-worker accumulator well
/// under 1 MiB while reducing repeated SCOP row-invariant work by 32× relative
/// to one-column HVP dispatch.
const SCOP_PSI_PSI_HVP_TILE_COLS: usize = 32;

/// Exact dense SCOP coefficient Hessian cache limit for the inner `H·v` path.
///
/// The large-scale CTN calibration fit has many rows but a moderate coefficient
/// dimension (for example n=20k, p=264). In that regime repeated PCG products
/// against the same Hessian should pay the row-streaming chain rule once, then
/// serve subsequent products as dense BLAS matvecs. Keep the cache restricted to
/// genuinely moderate p so wide CTN fits remain row-streamed.
const SCOP_HESSIAN_HVP_DENSE_CACHE_MAX_DIM: usize = 384;

const SCOP_HESSIAN_HVP_DENSE_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;


/// CTN-scoped ceiling on the custom-family inner exact-Newton cycle budget.
///
/// The global `DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES = 1200` exists for the
/// large-scale survival marginal-slope path, whose inner mode has a long,
/// rank-deficient KKT tail that genuinely needs hundreds of cycles. CTN is a
/// different regime: its coefficient block is a *bounded-dimension* Khatri–Rao
/// tensor (capped by `BASE/LARGE_SAMPLE_TRANSFORMATION_TENSOR_WIDTH`), and the
/// objective is strictly convex by construction — the `double_penalty` ridge
/// plus the order-2/order-1 roughness penalties make the penalized Hessian
/// positive definite even where the likelihood is flat on weakly-identified
/// shape×covariate directions. An exact-Newton iteration on a strictly convex,
/// bounded-dimension block converges in a handful of cycles; the only way the
/// fit reaches 1200 inner cycles is by polishing weakly-identified directions
/// that contribute nothing to the likelihood (the #720 timeout). Scaling the
/// cap with the realized coefficient dimension keeps a generous margin for a
/// genuinely nonlinear, high-dimensional transformation while refusing to grind
/// the production large-scale cap on an easy near-Gaussian shift.
const CTN_INNER_MAX_CYCLES_BASE: usize = 64;

const CTN_INNER_MAX_CYCLES_PER_DIM: usize = 2;

const CTN_INNER_MAX_CYCLES_CEILING: usize = 400;


/// Numerical floor on a Gram/penalty diagonal scale before it enters the
/// `likelihood_scale / penalty_scale` ratio that seeds the outer log-λ search.
/// A genuinely zero diagonal (an all-zero penalty block, or a degenerate
/// likelihood Gram) would otherwise produce a `0/0` or `x/0` seed; flooring
/// both scales at a value far below any meaningful curvature keeps the ratio
/// finite without perturbing well-posed problems.
const CTN_SEED_SCALE_FLOOR: f64 = 1.0e-8;


/// Lower bound on the cold-start seed log-λ (i.e. λ ≥ 1). Keeps the outer
/// optimizer out of the under-regularized regime where the CTN inner solve is
/// structurally rank-deficient (small-n / p > n); the optimizer is free to step
/// below this once the data support it. See `ctn_penalty_scale_log_lambdas`.
const CTN_SEED_LOG_LAMBDA_MIN: f64 = 0.0;


/// Upper bound on the cold-start seed log-λ, matching the outer ρ-bound used
/// across the location-scale families: λ ≈ e¹² caps the seed in the strongly
/// over-smoothed regime so a tiny penalty scale cannot seed an absurd λ.
const CTN_SEED_LOG_LAMBDA_MAX: f64 = 12.0;


/// Floor on the warm-start global residual scale `sqrt(weighted_ss / Σw)`.
/// Guards the degenerate near-perfect-fit case (residuals collapse to numerical
/// zero) so the per-residual `residual_floor` below — and the subsequent
/// `ln(|y−μ|)` log-scale target — stay finite. Well below any real response
/// spread, so it never perturbs a genuine fit.
const WARMSTART_GLOBAL_SCALE_FLOOR: f64 = 1e-6;


/// Per-residual floor used to form the log-scale warm-start target
/// `ln(|y−μ|) − E[ln|N(0,1)|]`. Built as `global_scale · WARMSTART_RESIDUAL_REL_FLOOR
/// + WARMSTART_RESIDUAL_ABS_FLOOR`: the relative term keeps an exactly-fit point
/// (|y−μ| = 0) from sending `ln(0) → −∞` at 1/1000 of the data scale, and the
/// absolute term backstops the case where `global_scale` itself sits at its floor.
const WARMSTART_RESIDUAL_REL_FLOOR: f64 = 1e-3;

const WARMSTART_RESIDUAL_ABS_FLOOR: f64 = 1e-12;


/// Floor on a per-row warm-start scale τ before forming `1/τ` when building the
/// affine transformation seed targets. A degenerate τ = 0 (a collapsed warm-start
/// scale block) would otherwise produce a non-finite reciprocal; the floor sits
/// far below any meaningful scale so it only fires on the degenerate path.
const WARMSTART_INV_SCALE_FLOOR: f64 = 1e-12;


/// Ridge stabilization floor for the penalized least-squares projections that
/// produce the default warm-start location and log-scale coefficients. These
/// seeds only need to land in the right basin (the outer solver refines them),
/// so a mild ridge that keeps the projection well-posed under a near-rank-
/// deficient covariate design is preferable to the tighter floor used for the
/// production inner solve.
const WARMSTART_PROJECTION_RIDGE_FLOOR: f64 = 1e-8;


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
    /// large-scale runs from churning large transient allocations.
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
    assert_eq!(h.len(), n);
    assert_eq!(h_lower.len(), n);
    assert_eq!(h_upper.len(), n);
    assert_eq!(weights.len(), n);

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
    // calls) which dominates this function's runtime at large scale.
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
    // boundary roundoff at large-scale shape (`p_resp` coefficients × O(1)
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
        if !(ratio.is_finite() && (-1.0e-12..=1.0 + 1.0e-12).contains(&ratio)) {
            return Err(TransformationNormalError::NumericalFailure { reason: format!(
                "transformation-normal PIT probability is not representable: h={h:.6e}, lower={lower:.6e}, upper={upper:.6e}, ratio={ratio}"
            ) }.into());
        }
        ratio.clamp(0.0, 1.0)
    };
    standard_normal_quantile(u.clamp(clip_eps, 1.0 - clip_eps))
        .map_err(|err| format!("transformation-normal PIT quantile failed: {err}"))
}


/// Accumulates the second-order monotone-transform quantities
/// `(h_i, h_j, h_ij, hp_i, hp_j, hp_ij)` for one row from the response value /
/// derivative bases and the per-response-knot gamma directional derivatives.
/// Shared verbatim across the SCOP Hessian/HVP/bilinear row loops.
fn scop_second_order_h(
    rv: ArrayView1<'_, f64>,
    rd: ArrayView1<'_, f64>,
    p_resp: usize,
    gamma: &[f64],
    gamma_i: &[f64],
    gamma_j: &[f64],
    gamma_ij: &[f64],
) -> [f64; 6] {
    let mut h_i = rv[0] * gamma_i[0];
    let mut h_j = rv[0] * gamma_j[0];
    let mut h_ij = rv[0] * gamma_ij[0];
    let mut hp_i = rd[0] * gamma_i[0];
    let mut hp_j = rd[0] * gamma_j[0];
    let mut hp_ij = rd[0] * gamma_ij[0];
    for k in 1..p_resp {
        let g = gamma[k];
        let gi = gamma_i[k];
        let gj = gamma_j[k];
        let gij = gamma_ij[k];
        h_i += 2.0 * rv[k] * g * gi;
        h_j += 2.0 * rv[k] * g * gj;
        h_ij += 2.0 * rv[k] * (gj * gi + g * gij);
        hp_i += 2.0 * rd[k] * g * gi;
        hp_j += 2.0 * rd[k] * g * gj;
        hp_ij += 2.0 * rd[k] * (gj * gi + g * gij);
    }
    [h_i, h_j, h_ij, hp_i, hp_j, hp_ij]
}


/// Accumulates the second-order endpoint-normalizer chain inputs
/// `(endpoint_i, endpoint_j, endpoint_ij)` for one row. Shared verbatim across
/// the SCOP Hessian/HVP/bilinear row loops.
fn scop_second_order_endpoints(
    endpoint_basis: [&[f64]; 2],
    p_resp: usize,
    gamma: &[f64],
    gamma_i: &[f64],
    gamma_j: &[f64],
    gamma_ij: &[f64],
) -> ([f64; 2], [f64; 2], [f64; 2]) {
    let mut endpoint_i = [0.0; 2];
    let mut endpoint_j = [0.0; 2];
    let mut endpoint_ij = [0.0; 2];
    for e in 0..2 {
        let basis = endpoint_basis[e];
        endpoint_i[e] = basis[0] * gamma_i[0];
        endpoint_j[e] = basis[0] * gamma_j[0];
        endpoint_ij[e] = basis[0] * gamma_ij[0];
        for k in 1..p_resp {
            endpoint_i[e] += 2.0 * basis[k] * gamma[k] * gamma_i[k];
            endpoint_j[e] += 2.0 * basis[k] * gamma[k] * gamma_j[k];
            endpoint_ij[e] += 2.0 * basis[k] * (gamma_j[k] * gamma_i[k] + gamma[k] * gamma_ij[k]);
        }
    }
    (endpoint_i, endpoint_j, endpoint_ij)
}


/// Accumulates the psi-direction transform quantities `(h_psi, hp_psi,
/// endpoint_psi)` for one row from the response bases and the per-knot psi
/// directional derivatives. Shared verbatim across the SCOP psi setup loops.
fn scop_psi_marginal(
    rv: ArrayView1<'_, f64>,
    rd: ArrayView1<'_, f64>,
    p_resp: usize,
    endpoint_basis: [&[f64]; 2],
    gamma: &[f64],
    gamma_psi: &[f64],
) -> (f64, f64, [f64; 2]) {
    let mut h_psi = rv[0] * gamma_psi[0];
    let mut hp_psi = rd[0] * gamma_psi[0];
    for k in 1..p_resp {
        h_psi += 2.0 * rv[k] * gamma[k] * gamma_psi[k];
        hp_psi += 2.0 * rd[k] * gamma[k] * gamma_psi[k];
    }

    let mut endpoint_psi = [0.0; 2];
    for e in 0..2 {
        let basis = endpoint_basis[e];
        endpoint_psi[e] = basis[0] * gamma_psi[0];
        for k in 1..p_resp {
            endpoint_psi[e] += 2.0 * basis[k] * gamma[k] * gamma_psi[k];
        }
    }
    (h_psi, hp_psi, endpoint_psi)
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
        assert_eq!(x_val_kron.ncols(), p_total);
        assert_eq!(x_deriv_kron.ncols(), p_total);

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
            outer_subsample_weights: None,
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
        assert_eq!(x_val_kron.ncols(), p_total);
        assert_eq!(x_deriv_kron.ncols(), p_total);

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
            outer_subsample_weights: None,
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
            gauge_priority: 100,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
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

    /// Number of response-direction basis columns `p_resp` (`[1, I_1, …, I_K]`).
    pub(crate) fn p_resp(&self) -> usize {
        self.response_val_basis.ncols()
    }

    /// Number of covariate-side design columns `p_cov`.
    pub(crate) fn p_cov(&self) -> usize {
        self.covariate_design.ncols()
    }

    /// Response value basis evaluated at the finite lower support endpoint
    /// (row-independent; `[1, I_1(y_min), …, I_K(y_min)]`).
    pub(crate) fn response_lower_basis(&self) -> &Array1<f64> {
        &self.response_lower_basis
    }

    /// Response value basis evaluated at the finite upper support endpoint
    /// (row-independent; `[1, I_1(y_max), …, I_K(y_max)]`).
    pub(crate) fn response_upper_basis(&self) -> &Array1<f64> {
        &self.response_upper_basis
    }

    /// Monotonicity floor offset applied to the lower-endpoint score
    /// `ε·(y_min − median_y)`.
    pub(crate) fn response_lower_floor_offset(&self) -> f64 {
        self.response_lower_floor_offset
    }

    /// Monotonicity floor offset applied to the upper-endpoint score
    /// `ε·(y_max − median_y)`.
    pub(crate) fn response_upper_floor_offset(&self) -> f64 {
        self.response_upper_floor_offset
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

    /// Evaluate the response value basis `[1, I_1(y), …, I_K(y)]` (n × p_resp)
    /// at arbitrary response values using the *fitted* clamped knots and degree.
    ///
    /// This is the out-of-sample analogue of the in-sample `response_val_basis`:
    /// it reuses the exact I-spline kernel and stored knot vector so that the
    /// score-influence Jacobian (and any other predict-time geometry) evaluates
    /// `I_k(y)` consistently with how `h` was built during the fit. Knots are
    /// taken from the family (not re-derived from `response`), so the basis is
    /// identical to training whenever the response values coincide.
    pub(crate) fn evaluate_response_value_basis(
        &self,
        response: ArrayView1<'_, f64>,
    ) -> Result<Array2<f64>, String> {
        let n = response.len();
        for (i, &v) in response.iter().enumerate() {
            if !v.is_finite() {
                return Err(TransformationNormalError::NonFinite {
                    reason: format!(
                        "evaluate_response_value_basis: response[{i}] is not finite: {v}"
                    ),
                }
                .into());
            }
        }
        let (i_val_basis, _) = create_basis::<Dense>(
            response,
            KnotSource::Provided(self.response_knots.view()),
            self.response_degree,
            BasisOptions::i_spline(),
        )
        .map_err(|e| format!("evaluate_response_value_basis: I-spline build failed: {e}"))?;
        let shape_val = i_val_basis.as_ref();
        let p_shape = shape_val.ncols();
        let p_resp = self.response_val_basis.ncols();
        if p_shape + 1 != p_resp {
            return Err(TransformationNormalError::InvalidInput {
                reason: format!(
                    "evaluate_response_value_basis: rebuilt shape columns {p_shape} imply p_resp {}, \
                     but fitted basis has {p_resp} columns",
                    p_shape + 1
                ),
            }
            .into());
        }
        let mut resp_val = Array2::<f64>::zeros((n, p_resp));
        resp_val.column_mut(0).fill(1.0);
        resp_val.slice_mut(s![.., 1..]).assign(shape_val);
        Ok(resp_val)
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
            crate::bail_invalid_tnorm!(
                "outer-score subsample mask length {} != n={}",
                mask.len(),
                n
            );
        }
        let mut effective = Array1::<f64>::zeros(n);
        for i in 0..n {
            let m = mask[i];
            if !m.is_finite() || m < 0.0 {
                crate::bail_invalid_tnorm!(
                    "outer-score subsample mask[{i}] = {m} is invalid (must be finite and >= 0)"
                );
            }
            effective[i] = self.weights[i] * m;
        }
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
            outer_subsample_weights: Some(Arc::new(effective)),
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
        // allocation and a single-threaded post-pass at large scale.
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
        let row_quantities = TransformationNormalRowQuantityCache {
            beta: Arc::new(beta.clone()),
            gamma: Arc::new(gamma),
            h: Arc::new(h),
            h_prime: Arc::new(h_prime),
            h_lower: Arc::new(h_lower),
            h_upper: Arc::new(h_upper),
            endpoint_q: Arc::new(derived.endpoint_q),
            log_likelihood: derived.log_likelihood,
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

                    let (h_psi, hp_psi, endpoint_psi) = scop_psi_marginal(
                        rv,
                        rd,
                        p_resp,
                        endpoint_basis,
                        &acc.gamma,
                        &acc.gamma_psi,
                    );

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

                    let (h_psi, hp_psi, endpoint_psi) = scop_psi_marginal(
                        rv,
                        rd,
                        p_resp,
                        endpoint_basis,
                        &acc.gamma,
                        &acc.gamma_psi,
                    );

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

                        let (h_psi, hp_psi, endpoint_psi) = scop_psi_marginal(
                            rv,
                            rd,
                            p_resp,
                            endpoint_basis,
                            &acc.gamma,
                            &acc.gamma_psi,
                        );

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
                        let [h_i, h_j, h_ij, hp_i, hp_j, hp_ij] = scop_second_order_h(
                            rv,
                            rd,
                            p_resp,
                            &acc.gamma,
                            &acc.gamma_i,
                            &acc.gamma_j,
                            &acc.gamma_ij,
                        );

                        let inv_hp = 1.0 / hp;
                        let inv_hp_sq = inv_hp * inv_hp;
                        let inv_hp_cu = inv_hp_sq * inv_hp;
                        let q = endpoint_q[row_idx];
                        let (endpoint_i, endpoint_j, endpoint_ij) = scop_second_order_endpoints(
                            endpoint_basis,
                            p_resp,
                            &acc.gamma,
                            &acc.gamma_i,
                            &acc.gamma_j,
                            &acc.gamma_ij,
                        );
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
                    let [h_i, h_j, h_ij, hp_i, hp_j, hp_ij] = scop_second_order_h(
                        rv,
                        rd,
                        p_resp,
                        &acc.gamma,
                        &acc.gamma_i,
                        &acc.gamma_j,
                        &acc.gamma_ij,
                    );

                    let inv_hp = 1.0 / hp;
                    let inv_hp_sq = inv_hp * inv_hp;
                    let inv_hp_cu = inv_hp_sq * inv_hp;
                    let inv_hp_qu = inv_hp_sq * inv_hp_sq;
                    let wi = weights[global_row];
                    let q = endpoint_q[row_idx];
                    let (endpoint_i, endpoint_j, endpoint_ij) = scop_second_order_endpoints(
                        endpoint_basis,
                        p_resp,
                        &acc.gamma,
                        &acc.gamma_i,
                        &acc.gamma_j,
                        &acc.gamma_ij,
                    );

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
                    let [h_i, h_j, h_ij, hp_i, hp_j, hp_ij] = scop_second_order_h(
                        rv,
                        rd,
                        p_resp,
                        &acc.gamma,
                        &acc.gamma_i,
                        &acc.gamma_j,
                        &acc.gamma_ij,
                    );

                    let q = endpoint_q[row_idx];
                    let (endpoint_i, endpoint_j, endpoint_ij) = scop_second_order_endpoints(
                        endpoint_basis,
                        p_resp,
                        &acc.gamma,
                        &acc.gamma_i,
                        &acc.gamma_j,
                        &acc.gamma_ij,
                    );

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
        // large scale; per-row contributions to `out` are independent
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

    let likelihood_scale = matrix_diag_mean_abs(likelihood_gram).max(CTN_SEED_SCALE_FLOOR);
    Array1::from_iter(penalties.iter().map(|penalty| {
        let penalty_scale = penalty_diag_scale(penalty).max(CTN_SEED_SCALE_FLOOR);
        // Lower-bound the SEED log-lambda at 0 (i.e., λ ≥ 1) so we never
        // start the outer optimizer in the under-regularized regime where
        // the CTN inner is structurally rank-deficient (small-n / p > n).
        // The outer BFGS is free to step below 0 when there's enough data
        // to support it; only the cold-start is constrained. Without this
        // floor, ratios like n=64, p_resp×p_cov ≈ 200 produce a seed of
        // log_lambda ≈ -12 (λ ≈ 6e-6), which leaves the inner solve to
        // pick wild β coefficients and cascade into predict-time monotonicity
        // violations (h' < 0 on the response grid, observed as -1e15 spikes
        // in CI synthetic-large-scale tests).
        (likelihood_scale / penalty_scale)
            .ln()
            .clamp(CTN_SEED_LOG_LAMBDA_MIN, CTN_SEED_LOG_LAMBDA_MAX)
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
        PenaltyMatrix::Fixed { inner, .. } => penalty_diag_scale(inner),
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
