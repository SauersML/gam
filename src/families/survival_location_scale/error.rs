/// Typed errors emitted by the survival location-scale family pipeline.
///
/// Each variant carries a pre-formatted `reason` string so `Display` is
/// byte-equivalent to the original `format!(...)` outputs the module used
/// before the typed-error migration. The category split lets callers
/// pattern-match on the failure kind without dragging the string apart.
#[derive(Debug, Clone)]
pub enum SurvivalLocationScaleError {
    /// Row/column/length disagreement between vectors, matrices, designs,
    /// penalty blocks, or coefficient/parameter dimensions.
    DimensionMismatch { reason: String },
    /// Spec-level validation: tolerances, iteration caps, knot-vector
    /// lengths, time intervals, weight values, or missing/contradictory
    /// configuration fields the user supplied.
    InvalidConfiguration { reason: String },
    /// Structural constraint violated at runtime: monotonicity guards,
    /// lower bounds on coefficients, nonnegativity, derivative-basis
    /// sign, or values outside an allowed semantic range.
    ConstraintViolation { reason: String },
    /// A numerical step produced a non-finite or out-of-domain value
    /// downstream code cannot consume (NaN products, invalid pdf,
    /// survival probability out of (0,1], etc.).
    NumericalFailure { reason: String },
    /// Internal invariant about pipeline state (empty block markers,
    /// unexpected ranks, schema/state inconsistencies surfaced from
    /// inner helpers).
    InternalInvariant { reason: String },
}

impl std::fmt::Display for SurvivalLocationScaleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SurvivalLocationScaleError::DimensionMismatch { reason }
            | SurvivalLocationScaleError::InvalidConfiguration { reason }
            | SurvivalLocationScaleError::ConstraintViolation { reason }
            | SurvivalLocationScaleError::NumericalFailure { reason }
            | SurvivalLocationScaleError::InternalInvariant { reason } => f.write_str(reason),
        }
    }
}

impl std::error::Error for SurvivalLocationScaleError {}

impl From<SurvivalLocationScaleError> for String {
    fn from(err: SurvivalLocationScaleError) -> String {
        err.to_string()
    }
}

impl From<String> for SurvivalLocationScaleError {
    /// Inbound conversion from the many `Result<_, String>` helpers this
    /// module still calls into. The text is preserved verbatim; we only
    /// pick a generic category so external messages flow through `?`
    /// without per-callsite `.map_err`.
    fn from(reason: String) -> SurvivalLocationScaleError {
        SurvivalLocationScaleError::InternalInvariant { reason }
    }
}

// ---------------------------------------------------------------------------
// Overflow-safe arithmetic for the survival exact-Newton chain
// ---------------------------------------------------------------------------
//
// The survival location-scale model computes inv_sigma = exp(-eta_ls) and
// multiplies it through many intermediate quantities (q0, qdot, g, ...).
// When eta_ls is very negative (sigma → 0, distribution very concentrated),
// exp(-eta_ls) can overflow to inf, poisoning downstream sums with NaN via
// inf * 0 or inf - inf patterns.
//
// The protection strategy is layered:
//
//   Layer 1 – `exp_neg_stable`: cap the exp argument at +500 (one-sided)
//     so inv_sigma ≤ exp(500) ≈ 1.4e217, preventing overflow at the
//     source.  Underflow (exp(-x) → 0 for large positive x) is allowed
//     because it is the mathematically correct limit.  Products like
//     inv_sigma * eta_t stay finite for any eta_t below ~1e91.
//
//   Layer 2 – `survival_q0_from_eta`: uses log-space arithmetic to detect
//     when |eta_t * inv_sigma| would exceed the clamp ceiling and saturates
//     to ±MAX instead of overflowing.
//
//   Layer 3 – factorized time-derivative algebra and compensated subtraction:
//     the base dq/dt chain is evaluated as exp(-eta_ls) * (eta_t*eta_ls' - eta_t')
//     so the shared exp(-eta_ls) factor is applied only once, and
//     d_eta/dt = d_raw + qdot is formed with a compensated sum that
//     carries an explicit roundoff bound into the monotonicity gate.
//
//   Layer 4 – `safe_product` / `safe_sum2` plus `exact_row_kernel`: the generic
//     arithmetic guards still clamp inf products to MAX/MIN and map
//     inf + (-inf) → 0 as defense in depth, and the row kernel splits the old
//     `!g.is_finite()` hard error
//     into NaN (hard error for genuinely bad data) and ±inf (clamped to MAX
//     so the monotonicity guard can apply).
//
// The invariant: no NaN ever reaches the solver; all overflow paths saturate
// to large finite values that the monotonicity floor and penalty then control.
// ---------------------------------------------------------------------------

// Layer 1 (one-sided overflow guard on the inverse-sigma link), its
// helper `exp_neg_stable`, and `exp_sigma_inverse_from_eta_scalar` now
// live in `crate::families::sigma_link` so every consumer — solver
// internals here, `main.rs` callers, and any Rust↔Python boundary
// code — picks up the same clamp. Keeping a local copy here previously
// allowed silent semantic divergence between the canonical sigma_link
// version (unclamped) and the survival-local clamped version.
