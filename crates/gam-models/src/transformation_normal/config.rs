use ndarray::Array1;

#[derive(Clone, Debug)]
pub struct TransformationNormalConfig {
    /// B-spline degree for the response-direction deviation basis (default 3).
    pub response_degree: usize,
    /// Number of interior knots for the response-direction deviation basis (default 10).
    pub response_num_internal_knots: usize,
    /// Difference penalty order for the response-direction roughness penalty (default 2).
    pub response_penalty_order: usize,
    /// Additional penalty orders for the response-direction (default \[1\]).
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
    /// A pre-resolved response knot vector, used verbatim by
    /// `build_response_basis` instead of regenerating one from this fit's own
    /// response. The knot vector fixes the response-basis chart and the
    /// boundary anchors of its affine tails, so it is a shared object whenever
    /// several CTN fits must produce comparable scores.
    ///
    /// The cross-fit Stage-1 calibration is that case and it is why this exists
    /// (gam#2680): it refits the CTN on each fold complement and scores the
    /// held-out rows. Re-resolving the response knots on each complement would
    /// express those `K` transforms in `K` different spline charts and anchor
    /// their affine tails at `K` different points. `None` means "resolve it from
    /// this fit's own response", which is right for every single-fit path.
    pub response_knots_pinned: Option<Array1<f64>>,
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
            response_knots_pinned: None,
        }
    }
}

/// Baseline cap for the tensor-product width used by the transformation-normal
/// response basis. Small datasets should stay compact because the fit
/// repeatedly factorizes dense penalized Hessians.
pub(crate) const BASE_TRANSFORMATION_TENSOR_WIDTH: usize = 160;

/// Large samples can support a richer response basis without the aggressive
/// underfitting forced by the small-sample cap above. This upper cap keeps the
/// tensor width bounded even when the covariate side is narrow.
pub(crate) const LARGE_SAMPLE_TRANSFORMATION_TENSOR_WIDTH: usize = 320;

/// E[log |Z|] for Z ~ N(0, 1), used to put local log-absolute residual
/// projections on the standard-normal scale.
pub(crate) const STANDARD_NORMAL_MEAN_LOG_ABS: f64 = -0.635_181_422_730_739_1;

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
pub(crate) const SCOP_PSI_PSI_HVP_TILE_COLS: usize = 32;

/// Exact dense SCOP coefficient Hessian cache limit for the inner `H·v` path.
///
/// The large-scale CTN calibration fit has many rows but a moderate coefficient
/// dimension (for example n=20k, p=264). In that regime repeated PCG products
/// against the same Hessian should pay the row-streaming chain rule once, then
/// serve subsequent products as dense BLAS matvecs. Keep the cache restricted to
/// genuinely moderate p so wide CTN fits remain row-streamed.
pub(crate) const SCOP_HESSIAN_HVP_DENSE_CACHE_MAX_DIM: usize = 384;

pub(crate) const SCOP_HESSIAN_HVP_DENSE_CACHE_MAX_BYTES: usize = 64 * 1024 * 1024;

/// CTN-scoped ceiling on the custom-family inner exact-Newton cycle budget.
///
/// The global `DEFAULT_CUSTOM_FAMILY_INNER_MAX_CYCLES = 1200` exists for the
/// large-scale survival marginal-slope path, whose inner mode has a long,
/// rank-deficient KKT tail that genuinely needs hundreds of cycles. CTN is a
/// different regime: its coefficient block is a *bounded-dimension* Khatri–Rao
/// tensor (capped by `BASE/LARGE_SAMPLE_TRANSFORMATION_TENSOR_WIDTH`), and the
/// objective is convex by construction. The convexity is the LIKELIHOOD's, not
/// the penalty's: the negative log-likelihood is `Σ w (½h² − log h')` with `h`
/// and `h'` both linear in β, so its exact Hessian is
/// `Σ w (∇h ∇hᵀ + ∇h' ∇h'ᵀ / h'²)` — two Gram matrices, positive definite
/// wherever no nonzero β annihilates `h` on every row. (Before gam#2600 this
/// paragraph credited the `double_penalty` ridge and the roughness penalties,
/// and it was false in both halves: the endpoint renormalizer made the
/// likelihood non-convex, and gam#2600's predecessor `faf3f3fde` had already
/// put the affine transformation in every penalty's null space.) An
/// exact-Newton iteration on a convex, bounded-dimension block converges in a
/// handful of cycles; the only way the fit reaches 1200 inner cycles is by
/// polishing weakly-identified directions that contribute nothing to the
/// likelihood (the #720 timeout). Scaling the cap with the realized coefficient
/// dimension keeps a generous margin for a genuinely nonlinear, high-dimensional
/// transformation while refusing to grind the production large-scale cap on an
/// easy near-Gaussian shift.
pub(crate) const CTN_INNER_MAX_CYCLES_BASE: usize = 64;

pub(crate) const CTN_INNER_MAX_CYCLES_PER_DIM: usize = 2;

pub(crate) const CTN_INNER_MAX_CYCLES_CEILING: usize = 400;

/// Floor on the warm-start global residual scale `sqrt(weighted_ss / Σw)`.
/// Guards the degenerate near-perfect-fit case (residuals collapse to numerical
/// zero) so the per-residual `residual_floor` below — and the subsequent
/// `ln(|y−μ|)` log-scale target — stay finite. Well below any real response
/// spread, so it never perturbs a genuine fit.
pub(crate) const WARMSTART_GLOBAL_SCALE_FLOOR: f64 = 1e-6;

/// Per-residual floor used to form the log-scale warm-start target
/// `ln(|y−μ|) − E[ln|N(0,1)|]`. Built as `global_scale · WARMSTART_RESIDUAL_REL_FLOOR
/// + WARMSTART_RESIDUAL_ABS_FLOOR`: the relative term keeps an exactly-fit point
/// (|y−μ| = 0) from sending `ln(0) → −∞` at 1/1000 of the data scale, and the
/// absolute term backstops the case where `global_scale` itself sits at its floor.
pub(crate) const WARMSTART_RESIDUAL_REL_FLOOR: f64 = 1e-3;

pub(crate) const WARMSTART_RESIDUAL_ABS_FLOOR: f64 = 1e-12;

/// Floor on a per-row warm-start scale τ before forming `1/τ` when building the
/// affine transformation seed targets. A degenerate τ = 0 (a collapsed warm-start
/// scale block) would otherwise produce a non-finite reciprocal; the floor sits
/// far below any meaningful scale so it only fires on the degenerate path.
pub(crate) const WARMSTART_INV_SCALE_FLOOR: f64 = 1e-12;

/// Ridge stabilization floor for the penalized least-squares projections that
/// produce the default warm-start location and log-scale coefficients. These
/// seeds only need to land in the right basin (the outer solver refines them),
/// so a mild ridge that keeps the projection well-posed under a near-rank-
/// deficient covariate design is preferable to the tighter floor used for the
/// production inner solve.
pub(crate) const WARMSTART_PROJECTION_RIDGE_FLOOR: f64 = 1e-8;
