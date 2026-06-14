use numeric_guards::{
    compensated_difference, safe_hadamard_product, safe_linear_combo2_arrays, safe_product,
    safe_product3, safe_sum2, safe_sum3, sanitize_survival_weight_vector, softplus,
};


const SURVIVAL_ROW_PARALLEL_THRESHOLD: usize = 256;

const SURVIVAL_ROW_PARALLEL_CHUNK: usize = 64;


/// Relative slack tolerating round-off when checking a represented
/// nonnegativity / linear-inequality constraint (`value < -tol·max(1, |scale|)`
/// rejects). A coefficient or slack that is negative only by this much is
/// floating-point noise about an active constraint, not a real violation.
const CONSTRAINT_NONNEGATIVITY_REL_TOL: f64 = 1e-10;


/// Maximum number of Dykstra alternating-projection sweeps when projecting an
/// initial coefficient guess onto the represented linear inequality
/// constraints. The projection converges geometrically; this caps the rare
/// near-degenerate constraint set and keeps the warm-start best-effort.
const DYKSTRA_PROJECTION_MAX_SWEEPS: usize = 100;


/// Absolute feasibility tolerance at which the Dykstra projection sweep is
/// declared converged (max constraint violation below this stops the loop).
const DYKSTRA_PROJECTION_TOL: f64 = 1e-10;


/// Squared-row-norm floor below which a constraint row is treated as
/// structurally empty and skipped during Dykstra projection (avoids dividing
/// the projection step by a vanishing normal).
const DYKSTRA_ROW_DEGENERACY_FLOOR: f64 = 1e-18;


/// Relative tolerance (× the largest |eigenvalue|) for accepting a covariance
/// block as positive semidefinite, floored by an absolute value so an
/// all-tiny-eigenvalue block is not rejected on pure round-off. Eigenvalues
/// below `-tol` flag a genuine indefinite block.
const PSD_EIGENVALUE_REL_TOL: f64 = 1e-12;

const PSD_EIGENVALUE_ABS_FLOOR: f64 = 1e-14;


/// Levenberg damping schedule for the direct parametric-AFT Newton solve. When
/// the Hessian is not Cholesky-factorizable, damping starts at
/// `INITIAL × max(1, ‖diag H‖∞)`, grows by `GROWTH` per failed factorization,
/// and the solve aborts once it would exceed `MAX × max(1, ‖diag H‖∞)` (the
/// Hessian is then numerically unsalvageable). All three scale with the
/// Hessian's diagonal magnitude so the schedule is units-invariant.
const LEVENBERG_INITIAL_DAMPING_REL: f64 = 1e-8;

const LEVENBERG_DAMPING_GROWTH: f64 = 10.0;

const LEVENBERG_MAX_DAMPING_REL: f64 = 1e8;


/// Outer (smoothing-parameter) loop budget for the blockwise location-scale
/// fit: at most this many outer iterations, stopping once the outer relative
/// change falls below the tolerance. The dead-flat time-smoothing ridge of the
/// constant-scale case is what makes a finite cap necessary.
const BLOCKWISE_OUTER_MAX_ITER: usize = 60;

const BLOCKWISE_OUTER_TOL: f64 = 1e-5;


/// Lower bound on the gradient tolerance handed to the reduced parametric-AFT
/// direct MLE. The inner-solve tolerance can be configured arbitrarily small;
/// flooring it here keeps the Newton stopping test above the noise of the
/// log-likelihood gradient evaluation.
const REDUCED_AFT_GRAD_TOL_FLOOR: f64 = 1e-8;


/// Relative ridge added to the normal-equations diagonal of the structural
/// time-coefficient warm-start least squares (× the largest diagonal of XᵀX,
/// floored at 1). Stabilizes the best-effort guess against a rank-deficient
/// derivative design without materially biasing it.
const STRUCTURAL_GUESS_RIDGE_REL: f64 = 1e-6;


/// Floor on the exit age when forming the `1/age` structural-derivative target
/// for the time warm-start, guarding against a divide-by-zero at age 0.
const STRUCTURAL_GUESS_AGE_FLOOR: f64 = 1e-9;


/// Target byte budget for one row-chunk when streaming a design matrix's
/// trailing columns into a dense buffer. The per-chunk row count is derived as
/// `BUDGET / (p · sizeof(f64))`, so wide designs use proportionally fewer rows
/// per chunk and the working set stays near this size regardless of `p`.
const ROW_CHUNK_BYTE_BUDGET: usize = 8 * 1024 * 1024;


/// Relative floor on the monotonicity-guard round-off slack: the compensated
/// subtraction's low-part residual is the primary slack estimate, but this
/// `1e-12 × (1 + ‖state‖∞)` term remains as a floor for moderate-magnitude
/// inputs where the residual underestimates accumulated error.
const MONOTONICITY_GUARD_SLACK_REL: f64 = 1e-12;


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

fn safe_fast_xt_diag_x(x: &Array2<f64>, weights: &Array1<f64>) -> Array2<f64> {
    let sanitized = sanitize_survival_weight_vector(weights);
    fast_xt_diag_x(x, &sanitized)
}


fn safe_fast_xt_diag_x_with_parallelism(
    x: &Array2<f64>,
    weights: &Array1<f64>,
    par: faer::Par,
) -> Array2<f64> {
    let sanitized = sanitize_survival_weight_vector(weights);
    fast_xt_diag_x_with_parallelism(x, &sanitized, par)
}


/// Horvitz-Thompson outer-subsample row mask. When `mask` is `None` this
/// returns `weighted_crossprod_dense(left, weights, right)` verbatim, which is
/// the byte-identical pre-refactor expression. When `mask` is `Some(m)`, the
/// per-row weight `weights[i]` is replaced by `weights[i] * m[i]` before the
/// cross product. The math invariant is that each survival-LS assembly site
/// is row-additive — `Σ_i x_i y_iᵀ · w_i` — so per-row HT-masking is unbiased.
#[inline]
fn mxtwx(
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
    mask: Option<&Array1<f64>>,
) -> Result<Array2<f64>, String> {
    match mask {
        Some(m) => weighted_crossprod_dense(left, &(weights * m), right),
        None => weighted_crossprod_dense(left, weights, right),
    }
}


#[inline]
fn mxtwxd(x: &Array2<f64>, weights: &Array1<f64>, mask: Option<&Array1<f64>>) -> Array2<f64> {
    match mask {
        Some(m) => safe_fast_xt_diag_x(x, &(weights * m)),
        None => safe_fast_xt_diag_x(x, weights),
    }
}


/// Multiply a per-row weight by the HT mask. The `None` branch returns the
/// caller's array unmodified (zero-copy borrow), so any downstream
/// `X.t().dot(&out)` / `out.sum()` / `out.dot(&other)` aggregate is
/// byte-identical to the pre-refactor path. The `Some` branch produces an
/// owned masked copy.
#[inline]
fn mask_row_vec<'a>(
    weights: &'a Array1<f64>,
    mask: Option<&Array1<f64>>,
) -> std::borrow::Cow<'a, Array1<f64>> {
    match mask {
        Some(m) => std::borrow::Cow::Owned(weights * m),
        None => std::borrow::Cow::Borrowed(weights),
    }
}


/// HT-mask-aware variant of [`weighted_crossprod_psi_maps`]. `None` is
/// byte-identical to the pre-refactor call. `Some(m)` multiplies the
/// per-row weight view by `m` before the cross product.
#[inline]
fn mxtwx_psi(
    left: crate::families::custom_family::CustomFamilyPsiLinearMapRef<'_>,
    weights: ArrayView1<'_, f64>,
    right: crate::families::custom_family::CustomFamilyPsiLinearMapRef<'_>,
    mask: Option<&Array1<f64>>,
) -> Result<Array2<f64>, String> {
    match mask {
        Some(m) => {
            let masked = &weights * m;
            weighted_crossprod_psi_maps(left, masked.view(), right)
        }
        None => weighted_crossprod_psi_maps(left, weights, right),
    }
}


/// Layer 2 defense: compute q0 = -eta_t * exp(-eta_ls) with log-space
/// overflow detection.  When log|q0| = ln|eta_t| + (-eta_ls) exceeds the
/// clamp ceiling, the product would overflow; we saturate to ±MAX instead.
#[inline]
fn survival_q0_from_eta(eta_t: f64, eta_ls: f64) -> f64 {
    if eta_t == 0.0 {
        return 0.0;
    }
    let log_abs = eta_t.abs().ln() + (-eta_ls).min(EXP_NEG_STABLE_MAX_ARG);
    if log_abs > EXP_NEG_STABLE_MAX_ARG {
        if eta_t > 0.0 { -f64::MAX } else { f64::MAX }
    } else {
        -eta_t * exp_sigma_inverse_from_eta_scalar(eta_ls)
    }
}


#[inline]
fn probit_survival_value(eta: f64) -> f64 {
    if eta.is_nan() {
        f64::NAN
    } else if eta == f64::INFINITY {
        0.0
    } else if eta == f64::NEG_INFINITY {
        1.0
    } else {
        0.5 * erfc(eta / std::f64::consts::SQRT_2)
    }
}


#[inline]
fn probit_log_survival_and_ratio_derivatives(eta: f64) -> (f64, f64, f64, f64, f64) {
    if eta.is_nan() {
        return (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN);
    }
    if eta == f64::NEG_INFINITY {
        return (0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let x = eta / std::f64::consts::SQRT_2;
    let (log_survival, ratio) = if eta >= 0.0 {
        // erfcx(x) = exp(x²)·erfc(x); compute once and reuse for both
        // log-survival and the hazard ratio.
        let erfcx_val = erfcx_nonnegative(x);
        let log_surv = -0.5 * eta * eta + (0.5 * erfcx_val).ln();
        let r = std::f64::consts::FRAC_2_SQRT_PI / (std::f64::consts::SQRT_2 * erfcx_val);
        (log_surv, r)
    } else {
        let survival = probit_survival_value(eta);
        (survival.ln(), normal_pdf(eta) / survival)
    };
    let dr = ratio * (ratio - eta);
    let ddr = 2.0 * ratio.powi(3) - 3.0 * eta * ratio.powi(2) + (eta * eta - 1.0) * ratio;
    let dddr = 6.0 * ratio.powi(4) - 12.0 * eta * ratio.powi(3)
        + (7.0 * eta * eta - 4.0) * ratio.powi(2)
        + (-eta * eta * eta + 3.0 * eta) * ratio;
    (log_survival, ratio, dr, ddr, dddr)
}


#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ResidualDistribution {
    Gaussian,
    Gumbel,
    Logistic,
}


pub trait ResidualDistributionOps {
    fn cdf(&self, z: f64) -> f64;
    fn pdf(&self, z: f64) -> f64;
    fn pdf_derivative(&self, z: f64) -> f64;
    fn pdfsecond_derivative(&self, z: f64) -> f64;
    fn pdfthird_derivative(&self, z: f64) -> f64;

    /// Fourth derivative of the residual-distribution PDF, f''''(z).
    ///
    /// This is the m4 ingredient for the outer REML Hessian's Q[v_k, v_l] term.
    /// The second directional derivative of the inner Hessian (used by the outer
    /// Hessian drift) requires the 4th derivative of the composed likelihood
    /// F_αβγδ via the Arbogast chain rule. That chain rule's leading term
    /// m4·u_α·u_β·u_γ·u_δ needs this quantity.
    ///
    /// See response.md Section 6 for the mathematical derivation.
    fn pdffourth_derivative(&self, z: f64) -> f64;
}


impl ResidualDistributionOps for ResidualDistribution {
    fn cdf(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => normal_cdf(z),
            ResidualDistribution::Gumbel => {
                component_inverse_link_jet(crate::types::LinkComponent::CLogLog, z).mu
            }
            ResidualDistribution::Logistic => {
                component_inverse_link_jet(crate::types::LinkComponent::Logit, z).mu
            }
        }
    }

    fn pdf(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => normal_pdf(z),
            ResidualDistribution::Gumbel => {
                component_inverse_link_jet(crate::types::LinkComponent::CLogLog, z).d1
            }
            ResidualDistribution::Logistic => {
                component_inverse_link_jet(crate::types::LinkComponent::Logit, z).d1
            }
        }
    }

    fn pdf_derivative(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => -z * normal_pdf(z),
            ResidualDistribution::Gumbel => {
                component_inverse_link_jet(crate::types::LinkComponent::CLogLog, z).d2
            }
            ResidualDistribution::Logistic => {
                component_inverse_link_jet(crate::types::LinkComponent::Logit, z).d2
            }
        }
    }

    fn pdfsecond_derivative(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => {
                let f = normal_pdf(z);
                (z * z - 1.0) * f
            }
            ResidualDistribution::Gumbel => {
                component_inverse_link_jet(crate::types::LinkComponent::CLogLog, z).d3
            }
            ResidualDistribution::Logistic => {
                component_inverse_link_jet(crate::types::LinkComponent::Logit, z).d3
            }
        }
    }

    fn pdfthird_derivative(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => {
                let f = normal_pdf(z);
                -(z * z * z - 3.0 * z) * f
            }
            ResidualDistribution::Gumbel => inverse_link_pdfthird_derivative_for_inverse_link(
                &InverseLink::Standard(StandardLink::CLogLog),
                z,
            )
            .expect("standard cloglog inverse-link third derivative should evaluate"),
            ResidualDistribution::Logistic => inverse_link_pdfthird_derivative_for_inverse_link(
                &InverseLink::Standard(StandardLink::Logit),
                z,
            )
            .expect("standard logit inverse-link third derivative should evaluate"),
        }
    }

    /// Fourth derivative of the residual-distribution PDF.
    ///
    /// # Derivations
    ///
    /// **Gaussian**: f(z) = φ(z). The n-th derivative of the Gaussian PDF is
    /// (-1)^n He_n(z) φ(z) where He_n is the probabilist's Hermite polynomial.
    /// He_4(z) = z⁴ - 6z² + 3, so f''''(z) = (z⁴ - 6z² + 3) φ(z).
    ///
    /// **Logistic**: f(z) = s(1-s) with s = σ(z). The k-th derivative of f is
    /// f · P_k(s) where P_k satisfies the Euler-polynomial recurrence
    /// P_{k+1}(s) = (1-2s) P_k(s) + s(1-s) P_k'(s).
    /// P_4(s) = 1 - 30s + 150s² - 240s³ + 120s⁴.
    ///
    /// **Gumbel**: f(z) = exp(z - e^z). Let e = e^z. The k-th derivative of f
    /// is f · Q_k(e) where Q_k satisfies Q_{k+1}(e) = (1-e) Q_k(e) + e Q_k'(e).
    /// Q_4(e) = 1 - 15e + 25e² - 10e³ + e⁴.
    fn pdffourth_derivative(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => {
                let f = normal_pdf(z);
                let z2 = z * z;
                // He_4(z) = z^4 - 6z^2 + 3
                (z2 * z2 - 6.0 * z2 + 3.0) * f
            }
            ResidualDistribution::Gumbel => inverse_link_pdffourth_derivative_for_inverse_link(
                &InverseLink::Standard(StandardLink::CLogLog),
                z,
            )
            .expect("standard cloglog inverse-link fourth derivative should evaluate"),
            ResidualDistribution::Logistic => inverse_link_pdffourth_derivative_for_inverse_link(
                &InverseLink::Standard(StandardLink::Logit),
                z,
            )
            .expect("standard logit inverse-link fourth derivative should evaluate"),
        }
    }
}


#[inline]
fn residual_distribution_link(distribution: ResidualDistribution) -> StandardLink {
    match distribution {
        ResidualDistribution::Gaussian => StandardLink::Probit,
        ResidualDistribution::Gumbel => StandardLink::CLogLog,
        ResidualDistribution::Logistic => StandardLink::Logit,
    }
}


#[inline]
pub fn residual_distribution_inverse_link(distribution: ResidualDistribution) -> InverseLink {
    InverseLink::Standard(residual_distribution_link(distribution))
}


/// Maps an `InverseLink` to its `ResidualDistribution` counterpart when the
/// link is one of the three standard survival residual-distribution links
/// (Probit/Logit/CLogLog). Returns `None` for stateful / mixture links (Sas,
/// BetaLogistic, Mixture, LatentCLogLog) and for non-residual-distribution
/// standard links — those carry their full state via `payload.link` and have
/// no `ResidualDistribution` representation.
#[inline]
pub fn residual_distribution_from_inverse_link(link: &InverseLink) -> Option<ResidualDistribution> {
    match link {
        InverseLink::Standard(StandardLink::Probit) => Some(ResidualDistribution::Gaussian),
        InverseLink::Standard(StandardLink::CLogLog) => Some(ResidualDistribution::Gumbel),
        InverseLink::Standard(StandardLink::Logit) => Some(ResidualDistribution::Logistic),
        _ => None,
    }
}


/// Fourth derivative of the inverse-link PDF (= 5th derivative of the CDF).
///
/// This is the f'''' quantity used in the 4th derivative of log f(u), which
/// in turn enters the m4 ingredient of the Arbogast chain rule for
/// the outer REML Hessian Q[v_k, v_l] term.
///
/// For the three standard survival residual distributions (Probit, Logit,
/// CLogLog), uses the closed-form ResidualDistribution implementations.
/// For all other inverse links (SAS, BetaLogistic, Mixture), delegates
/// to the generic `inverse_link_pdffourth_derivative_for_inverse_link`
/// dispatcher in mixture_link.rs.
fn inverse_link_pdffourth_derivative(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<f64, SurvivalLocationScaleError> {
    match inverse_link {
        InverseLink::Standard(StandardLink::Probit) => {
            Ok(ResidualDistribution::Gaussian.pdffourth_derivative(eta))
        }
        InverseLink::Standard(StandardLink::Logit) => {
            Ok(ResidualDistribution::Logistic.pdffourth_derivative(eta))
        }
        InverseLink::Standard(StandardLink::CLogLog) => {
            Ok(ResidualDistribution::Gumbel.pdffourth_derivative(eta))
        }
        _ => crate::solver::mixture_link::inverse_link_pdffourth_derivative_for_inverse_link(
            inverse_link,
            eta,
        )
        .map_err(|e| SurvivalLocationScaleError::NumericalFailure {
            reason: format!("inverse link fourth-derivative evaluation failed at eta={eta}: {e}"),
        }),
    }
}


/// How a time block's parameterization enforces the derivative-guard
/// monotonicity `q'(t) ≥ guard`.
///
/// The constraint set fed to the inner active-set / KKT machinery depends on
/// the variant; consuming families dispatch on this to choose the right
/// constraint shape and to refuse a mismatched parameterization (e.g.
/// `survival_marginal_slope` cannot ride a coordinate-cone-only basis
/// without re-introducing the phantom-multiplier bug it solved with the
/// row-wise representation; `survival_location_scale` cannot ride a
/// row-wise representation without making its reduced KKT system
/// rank-deficient on the cone basis).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TimeBlockMonotonicity {
    /// The time block's coefficients are constrained by a per-coordinate
    /// cone `β_j ≥ 0` (with appropriate offsets handled by the family).
    /// Used by location-scale / latent paths whose bases produce a
    /// non-negative derivative whenever the cone holds.
    EnforcedByCoordinateCone,
    /// The time block's coefficients are constrained by row-wise
    /// `D β + o ≥ guard` over every observation row; needed when the
    /// basis admits negative-derivative directions that no coordinate
    /// cone can encode without leaving phantom KKT multipliers when a
    /// row binds. Used by `survival_marginal_slope` under the additive
    /// base.
    EnforcedByRowConstraint,
    /// The base is a structurally-monotone parameterization (e.g.
    /// `q'(t) = guard + I(t)·γ` with `γ ≥ 0`). Monotonicity holds
    /// pointwise from the cone; the family treats this exactly as a
    /// coordinate cone for constraint generation but the geometric
    /// claim is stronger and is recorded here for diagnostics and for
    /// future fast paths (e.g. skipping per-row validation).
    StructuralISpline,
}


impl TimeBlockMonotonicity {
    /// True when the variant can be enforced by a coordinate cone alone
    /// (no row-wise constraints required). Both `EnforcedByCoordinateCone`
    /// and `StructuralISpline` satisfy this; only `EnforcedByRowConstraint`
    /// requires the row-wise `D β ≥ b` constraint matrix.
    #[inline]
    pub fn is_coordinate_cone(self) -> bool {
        matches!(
            self,
            Self::EnforcedByCoordinateCone | Self::StructuralISpline
        )
    }

    /// True when row-wise `D β + o ≥ guard` constraints must be emitted
    /// for the inner active-set/KKT machinery to capture binding
    /// multipliers correctly.
    #[inline]
    pub fn requires_row_constraints(self) -> bool {
        matches!(self, Self::EnforcedByRowConstraint)
    }
}


#[derive(Clone)]
pub struct TimeBlockInput {
    pub design_entry: DesignMatrix,
    pub design_exit: DesignMatrix,
    pub design_derivative_exit: DesignMatrix,
    pub offset_entry: Array1<f64>,
    pub offset_exit: Array1<f64>,
    pub derivative_offset_exit: Array1<f64>,
    /// How the time block enforces `q'(t) ≥ guard`. The consuming family
    /// dispatches the constraint shape on this and refuses a mismatch
    /// rather than silently producing a degenerate KKT system.
    pub time_monotonicity: TimeBlockMonotonicity,
    pub penalties: Vec<Array2<f64>>,
    /// Structural nullspace dimension of each penalty matrix.
    pub nullspace_dims: Vec<usize>,
    pub initial_log_lambdas: Option<Array1<f64>>,
    pub initial_beta: Option<Array1<f64>>,
}


pub(crate) fn structural_time_coefficient_constraints(
    design_derivative_exit: &DesignMatrix,
    derivative_offset_exit: &Array1<f64>,
    derivative_guard: f64,
) -> Result<Option<LinearInequalityConstraints>, String> {
    time_derivative_guard_constraints(
        design_derivative_exit,
        derivative_offset_exit,
        derivative_guard,
    )
}


/// Location-scale guard policy: a degenerate `guard == 0` (a bare
/// non-negativity request on `q'(t)`) is admissible here, and feasibility of
/// coefficient-free rows uses the family's historical absolute slack.
const LOCATION_SCALE_GUARD_POLICY: GuardConstraintPolicy = GuardConstraintPolicy {
    guard_policy: GuardPolicy::NonNegative,
    feasibility: FeasibilityTolerance::LegacyAbsolute,
};


fn time_derivative_guard_constraints(
    design_derivative_exit: &DesignMatrix,
    derivative_offset_exit: &Array1<f64>,
    derivative_guard: f64,
) -> Result<Option<LinearInequalityConstraints>, String> {
    build_time_derivative_guard_constraints(
        design_derivative_exit,
        derivative_offset_exit,
        derivative_guard,
        LOCATION_SCALE_GUARD_POLICY,
    )
    .map_err(map_guard_constraint_failure)
}


/// Render a shared guard-constraint failure into the location-scale error
/// vocabulary, preserving the family's historical wording.
fn map_guard_constraint_failure(failure: GuardConstraintFailure) -> String {
    match failure {
        GuardConstraintFailure::RowOffsetMismatch { rows, offsets } => {
            SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "time derivative guard constraints require matching rows/offsets: rows={rows}, offsets={offsets}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::GuardOutOfRange { guard, range } => {
            SurvivalLocationScaleError::ConstraintViolation {
                reason: format!(
                    "time derivative guard must be finite and {range}, got {guard}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::NonFiniteOffset { row, offset } => {
            SurvivalLocationScaleError::ConstraintViolation {
                reason: format!(
                    "time derivative guard constraints require finite derivative offsets; found offset[{row}]={offset}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::NonFiniteDesign { row, col } => {
            SurvivalLocationScaleError::ConstraintViolation {
                reason: format!(
                    "time derivative guard constraints require finite derivative design entries; found row {row}, column {col}"
                ),
            }
            .into()
        }
        GuardConstraintFailure::InfeasibleRow {
            row,
            offset,
            guard,
            no_time_coefficients,
        } => {
            let detail = if no_time_coefficients {
                "with no time coefficients"
            } else {
                "zero derivative design row"
            };
            let reason = if no_time_coefficients {
                format!(
                    "time derivative guard is infeasible at row {row}: offset={offset:.3e} < guard={guard:.3e} {detail}"
                )
            } else {
                format!(
                    "time derivative guard is infeasible at row {row}: {detail} with offset={offset:.3e} < guard={guard:.3e}"
                )
            };
            SurvivalLocationScaleError::ConstraintViolation { reason }.into()
        }
    }
}


/// A covariate block whose linear predictor depends on the survival time axis
/// via a tensor product: covariate design (n x p_cov) ⊗ B-spline on log(time).
///
/// At row i the linear predictor evaluated at time t is
///
///   eta(t) = [ x_cov(i,:) ⊗ B_time(t) ] @ beta
///
/// where B_time(t) is a B-spline basis row evaluated at log(t).
/// The entry and exit tensor designs are precomputed:
///   X_entry[i,:] = x_cov(i,:) ⊗ B_time(t_entry_i)
///   X_exit[i,:]  = x_cov(i,:) ⊗ B_time(t_exit_i)
#[derive(Clone)]
pub struct TimeDependentCovariateBlockInput {
    /// Covariate design matrix (n x p_cov), same for all time points.
    pub design_covariates: DesignMatrix,
    /// B-spline time basis at entry times (n x p_time).
    pub time_basis_entry: Array2<f64>,
    /// B-spline time basis at exit times (n x p_time).
    pub time_basis_exit: Array2<f64>,
    /// Derivative of the time basis with respect to clock time at exit.
    pub time_basis_derivative_exit: Array2<f64>,
    /// Combined Kronecker penalties for the tensor product.
    pub penalties: Vec<PenaltyMatrix>,
    pub initial_log_lambdas: Option<Array1<f64>>,
    pub initial_beta: Option<Array1<f64>>,
    pub offset: Array1<f64>,
}


/// Whether a covariate block (threshold or log-sigma) is time-invariant or
/// depends on the survival time axis via a tensor product.
#[derive(Clone)]
pub enum CovariateBlockKind {
    Static(ParameterBlockInput),
    TimeVarying(TimeDependentCovariateBlockInput),
}


#[derive(Clone)]
pub struct LinkWiggleBlockInput {
    pub design: DesignMatrix,
    pub knots: Array1<f64>,
    pub degree: usize,
    pub penalties: Vec<crate::solver::estimate::PenaltySpec>,
    /// Structural nullspace dimension of each penalty matrix.
    pub nullspace_dims: Vec<usize>,
    pub initial_log_lambdas: Option<Array1<f64>>,
    pub initial_beta: Option<Array1<f64>>,
}


#[derive(Clone)]
pub struct TimeWiggleBlockInput {
    pub knots: Array1<f64>,
    pub degree: usize,
    pub ncols: usize,
}


#[derive(Clone)]
struct SurvivalLocationScaleSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<f64>,
    pub weights: Array1<f64>,
    pub inverse_link: InverseLink,
    pub derivative_guard: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub time_block: TimeBlockInput,
    pub threshold_block: CovariateBlockKind,
    pub log_sigma_block: CovariateBlockKind,
    pub timewiggle_block: Option<TimeWiggleBlockInput>,
    pub linkwiggle_block: Option<LinkWiggleBlockInput>,
    /// Persistent warm-start cache session threaded from the workflow
    /// dispatcher. See [`BlockwiseFitOptions::cache_session`].
    pub cache_session: Option<std::sync::Arc<crate::cache::Session>>,
    /// Persistent warm-start mirror sessions; see
    /// [`BlockwiseFitOptions::cache_mirror_sessions`].
    pub cache_mirror_sessions: Vec<std::sync::Arc<crate::cache::Session>>,
}


#[derive(Clone)]
pub enum SurvivalCovariateTermBlockTemplate {
    Static,
    TimeVarying {
        time_basis_entry: Array2<f64>,
        time_basis_exit: Array2<f64>,
        time_basis_derivative_exit: Array2<f64>,
        time_penalties: Vec<Array2<f64>>,
    },
}


#[derive(Clone)]
pub struct SurvivalLocationScaleTermSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<f64>,
    pub weights: Array1<f64>,
    pub inverse_link: InverseLink,
    /// Strict lower bound on d_eta/dt used by both the event Jacobian term
    /// and the time monotonicity constraints.
    pub derivative_guard: f64,
    pub max_iter: usize,
    pub tol: f64,
    pub time_block: TimeBlockInput,
    pub thresholdspec: TermCollectionSpec,
    pub log_sigmaspec: TermCollectionSpec,
    pub threshold_offset: Array1<f64>,
    pub log_sigma_offset: Array1<f64>,
    pub threshold_template: SurvivalCovariateTermBlockTemplate,
    pub log_sigma_template: SurvivalCovariateTermBlockTemplate,
    pub timewiggle_block: Option<TimeWiggleBlockInput>,
    pub linkwiggle_block: Option<LinkWiggleBlockInput>,
    /// Optional warm-start seed for the threshold-block log-smoothing parameters (ρ).
    /// When `Some`, its length must equal the number of threshold penalties; values are
    /// clamped to the outer-loop ρ bounds before being injected into `rho0`.
    /// Used by the outer baseline-config optimizer to thread converged smoothing
    /// from one probe into the next.
    pub initial_threshold_log_lambdas: Option<Array1<f64>>,
    /// Optional warm-start seed for the log-sigma-block log-smoothing parameters (ρ).
    /// Same semantics as `initial_threshold_log_lambdas`.
    pub initial_log_sigma_log_lambdas: Option<Array1<f64>>,
    /// Persistent warm-start cache session, threaded from the workflow
    /// dispatcher. See
    /// [`crate::families::custom_family::BlockwiseFitOptions::cache_session`].
    pub cache_session: Option<std::sync::Arc<crate::cache::Session>>,
    /// Persistent warm-start mirror sessions, threaded from the workflow
    /// dispatcher. See
    /// [`crate::families::custom_family::BlockwiseFitOptions::cache_mirror_sessions`].
    pub cache_mirror_sessions: Vec<std::sync::Arc<crate::cache::Session>>,
}


pub const DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD: f64 = 1e-6;


pub struct SurvivalLocationScaleTermFitResult {
    pub fit: UnifiedFitResult,
    pub resolved_thresholdspec: TermCollectionSpec,
    pub resolved_log_sigmaspec: TermCollectionSpec,
    pub threshold_design: TermCollectionDesign,
    pub log_sigma_design: TermCollectionDesign,
    /// Per-row gradient of unpenalized NLL w.r.t. the three additive time-block
    /// offset channels (entry / exit / derivative-at-exit) at the converged β.
    /// Contracted with `∂o/∂θ_baseline` this yields the analytic θ-gradient
    /// used by the with-gradient baseline optimizer.
    pub baseline_offset_residuals: OffsetChannelResiduals,
    /// 3×3 NLL Hessian per row on the offset channels, in
    /// `(entry, exit, derivative)` order. Diagonal under location-scale —
    /// the row likelihood is separable in `(u0, u1, g)`. Used by the analytic
    /// θ-Hessian builder (chain rule second derivative).
    pub baseline_offset_curvatures: OffsetChannelCurvatures,
    /// Exact data-fit gradient `∂(−ℓ)/∂θ_link` of the unpenalized
    /// log-likelihood w.r.t. the inverse-link parameters at the converged β̂
    /// (`None` when the inverse link carries no free parameters). Equals the
    /// envelope-theorem θ_link-gradient of the profile penalized NLL, consumed
    /// by the inverse-link BFGS optimizer.
    pub link_param_data_fit_gradient: Option<Array1<f64>>,
}


/// Helper struct so callers can build a `UnifiedFitResult` from
/// survival-specific fields without knowing about the unified layout.
pub struct SurvivalLocationScaleFitResultParts {
    pub beta_time: Array1<f64>,
    pub beta_threshold: Array1<f64>,
    pub beta_log_sigma: Array1<f64>,
    pub beta_link_wiggle: Option<Array1<f64>>,
    pub link_wiggle_knots: Option<Array1<f64>>,
    pub link_wiggle_degree: Option<usize>,
    pub lambdas_time: Array1<f64>,
    pub lambdas_threshold: Array1<f64>,
    pub lambdas_log_sigma: Array1<f64>,
    pub lambdas_linkwiggle: Option<Array1<f64>>,
    pub log_likelihood: f64,
    pub reml_score: f64,
    pub stable_penalty_term: f64,
    pub penalized_objective: f64,
    pub outer_iterations: usize,
    /// `None` = no gradient measured at termination; `Some(g)` = measured.
    /// `outer_converged` is the authoritative convergence signal.
    pub outer_gradient_norm: Option<f64>,
    pub outer_converged: bool,
    pub covariance_conditional: Option<Array2<f64>>,
    pub geometry: Option<FitGeometry>,
}


#[derive(Clone)]
struct PreparedSurvivalLocationScaleModel {
    family: SurvivalLocationScaleFamily,
    blockspecs: Vec<ParameterBlockSpec>,
    time_transform: TimeIdentifiabilityTransform,
    threshold_fixed_cols: usize,
    threshold_full_ncols: usize,
    log_sigma_fixed_cols: usize,
    log_sigma_full_ncols: usize,
    k_time: usize,
    k_threshold: usize,
    k_log_sigma: usize,
    k_wiggle: usize,
}


impl PreparedSurvivalLocationScaleModel {
    /// Whether this prepared model is the fully reduced, unpenalized
    /// constant-scale PARAMETRIC AFT regime (issue #736/#735/#721).
    ///
    /// In this regime the time block has collapsed to its identifiable affine
    /// null space (`reduce_time_to_parametric` fired, so `k_time == 0`), the
    /// scale is a single constant log-σ (`k_log_sigma == 0`), the mean is rigid
    /// or a plain parametric covariate effect whose default shrinkage ridge has
    /// been dropped by `survival_reduced_parametric_aft_regime` (`k_threshold ==
    /// 0`), and there is no link-wiggle or monotone time-wiggle (`k_wiggle ==
    /// 0`, `x_link_wiggle == None`, `time_wiggle_ncols == 0`). Every block is
    /// therefore parametric and UNPENALIZED — zero smoothing parameters — so the
    /// model is a plain few-parameter AFT MLE (loglogistic / lognormal, exactly
    /// what `survreg`/`lifelines` fit, including a parametric `~ age` effect) and
    /// the REML/LAML outer search is vacuous. Such fits are routed to a direct,
    /// robust parametric MLE (`fit_parametric_aft_direct_mle`) instead of the
    /// coupled exact-joint REML optimizer, which does not converge on this tiny
    /// unpenalized likelihood.
    ///
    /// Any genuinely flexible/penalized survival LS fit — smooth scale
    /// (`noise_formula = s(...)`, log_sigma smoothing penalties), smooth mean
    /// (`threshold ~ s(z)` wiggliness penalties), a link-wiggle, or an active
    /// monotone time-wiggle — keeps at least one nonzero `k_*` (the ridge-drop
    /// predicate excludes any block carrying a `nullspace_dim > 0` smoothing
    /// penalty) and so does NOT match here, keeping the full coupled exact-joint
    /// path unchanged.
    fn is_reduced_parametric_aft(&self) -> bool {
        self.k_time == 0
            && self.k_threshold == 0
            && self.k_log_sigma == 0
            && self.k_wiggle == 0
            && self.family.x_link_wiggle.is_none()
            && self.family.time_wiggle_ncols == 0
    }
}


#[derive(Clone, Copy)]
struct SurvivalLambdaLayout {
    k_time: usize,
    k_threshold: usize,
    k_log_sigma: usize,
    k_wiggle: usize,
}


impl SurvivalLambdaLayout {
    fn new(k_time: usize, k_threshold: usize, k_log_sigma: usize, k_wiggle: usize) -> Self {
        Self {
            k_time,
            k_threshold,
            k_log_sigma,
            k_wiggle,
        }
    }

    fn total(&self) -> usize {
        self.k_time + self.k_threshold + self.k_log_sigma + self.k_wiggle
    }

    fn time_range(&self) -> std::ops::Range<usize> {
        0..self.k_time
    }

    fn threshold_range(&self) -> std::ops::Range<usize> {
        self.k_time..self.k_time + self.k_threshold
    }

    fn log_sigma_range(&self) -> std::ops::Range<usize> {
        self.k_time + self.k_threshold..self.k_time + self.k_threshold + self.k_log_sigma
    }

    fn wiggle_range(&self) -> std::ops::Range<usize> {
        self.k_time + self.k_threshold + self.k_log_sigma..self.total()
    }

    fn validate_rho(&self, rho: &Array1<f64>, label: &str) -> Result<(), String> {
        if rho.len() != self.total() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "{label} rho length mismatch: got {}, expected {}",
                    rho.len(),
                    self.total()
                ),
            }
            .into());
        }
        Ok::<(), _>(())
    }

    fn time_from(&self, rho: &Array1<f64>) -> Array1<f64> {
        let range = self.time_range();
        rho.slice(s![range.start..range.end]).to_owned()
    }

    fn threshold_from(&self, rho: &Array1<f64>) -> Array1<f64> {
        let range = self.threshold_range();
        rho.slice(s![range.start..range.end]).to_owned()
    }

    fn log_sigma_from(&self, rho: &Array1<f64>) -> Array1<f64> {
        let range = self.log_sigma_range();
        rho.slice(s![range.start..range.end]).to_owned()
    }

    fn wiggle_from(&self, rho: &Array1<f64>) -> Option<Array1<f64>> {
        if self.k_wiggle == 0 {
            None
        } else {
            let range = self.wiggle_range();
            Some(rho.slice(s![range.start..range.end]).to_owned())
        }
    }
}


/// Build a `UnifiedFitResult` from survival-specific fields.
pub fn survival_fit_from_parts(
    parts: SurvivalLocationScaleFitResultParts,
) -> Result<UnifiedFitResult, String> {
    let SurvivalLocationScaleFitResultParts {
        beta_time,
        beta_threshold,
        beta_log_sigma,
        beta_link_wiggle,
        link_wiggle_knots,
        link_wiggle_degree,
        lambdas_time,
        lambdas_threshold,
        lambdas_log_sigma,
        lambdas_linkwiggle,
        log_likelihood,
        reml_score,
        stable_penalty_term,
        penalized_objective,
        outer_iterations,
        outer_gradient_norm,
        outer_converged,
        covariance_conditional,
        geometry,
    } = parts;

    // Validation (preserved from the old impl).
    validate_all_finite_estimation("survival_fit.beta_time", beta_time.iter().copied())
        .map_err(|e| e.to_string())?;
    validate_all_finite_estimation(
        "survival_fit.beta_threshold",
        beta_threshold.iter().copied(),
    )
    .map_err(|e| e.to_string())?;
    validate_all_finite_estimation(
        "survival_fit.beta_log_sigma",
        beta_log_sigma.iter().copied(),
    )
    .map_err(|e| e.to_string())?;
    if let Some(beta_wiggle) = beta_link_wiggle.as_ref() {
        validate_all_finite_estimation(
            "survival_fit.beta_link_wiggle",
            beta_wiggle.iter().copied(),
        )
        .map_err(|e| e.to_string())?;
        let knots = link_wiggle_knots.as_ref().ok_or_else(|| {
            "survival_fit.beta_link_wiggle requires link_wiggle_knots".to_string()
        })?;
        validate_all_finite_estimation("survival_fit.link_wiggle_knots", knots.iter().copied())
            .map_err(|e| e.to_string())?;
        if link_wiggle_degree.is_none() {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "survival_fit.beta_link_wiggle requires link_wiggle_degree".to_string(),
            }
            .into());
        }
    } else if link_wiggle_knots.is_some() || link_wiggle_degree.is_some() {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: "survival_fit link-wiggle metadata requires beta_link_wiggle coefficients"
                .to_string(),
        }
        .into());
    }
    validate_all_finite_estimation("survival_fit.lambdas_time", lambdas_time.iter().copied())
        .map_err(|e| e.to_string())?;
    validate_all_finite_estimation(
        "survival_fit.lambdas_threshold",
        lambdas_threshold.iter().copied(),
    )
    .map_err(|e| e.to_string())?;
    validate_all_finite_estimation(
        "survival_fit.lambdas_log_sigma",
        lambdas_log_sigma.iter().copied(),
    )
    .map_err(|e| e.to_string())?;
    // Each block's smoothing-parameter count counts the number of distinct
    // penalty terms acting on that block's coefficients. A penalty term cannot
    // outnumber the coefficients it penalizes, so reject `lambdas_<block>`
    // vectors longer than the corresponding `beta_<block>`. This catches stale
    // / misaligned lambda slices that would otherwise propagate silently into
    // downstream inference where the per-block penalty bookkeeping is
    // unrecoverable.
    if lambdas_time.len() > beta_time.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "survival_fit.lambdas_time has {} entries but beta_time has only {} \
                 coefficients; each lambda corresponds to a penalty term on this block",
                lambdas_time.len(),
                beta_time.len()
            ),
        }
        .into());
    }
    if lambdas_threshold.len() > beta_threshold.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "survival_fit.lambdas_threshold has {} entries but beta_threshold has only {} \
                 coefficients; each lambda corresponds to a penalty term on this block",
                lambdas_threshold.len(),
                beta_threshold.len()
            ),
        }
        .into());
    }
    if lambdas_log_sigma.len() > beta_log_sigma.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "survival_fit.lambdas_log_sigma has {} entries but beta_log_sigma has only {} \
                 coefficients; each lambda corresponds to a penalty term on this block",
                lambdas_log_sigma.len(),
                beta_log_sigma.len()
            ),
        }
        .into());
    }
    if let Some(lambdas_wiggle) = lambdas_linkwiggle.as_ref() {
        if beta_link_wiggle.is_none() {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "survival_fit.lambdas_linkwiggle requires beta_link_wiggle".to_string(),
            }
            .into());
        }
        validate_all_finite_estimation(
            "survival_fit.lambdas_linkwiggle",
            lambdas_wiggle.iter().copied(),
        )
        .map_err(|e| e.to_string())?;
        let wiggle_len = beta_link_wiggle.as_ref().map_or(0, |beta| beta.len());
        if lambdas_wiggle.len() > wiggle_len {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival_fit.lambdas_linkwiggle has {} entries but beta_link_wiggle has \
                     only {} coefficients; each lambda corresponds to a penalty term on this block",
                    lambdas_wiggle.len(),
                    wiggle_len
                ),
            }
            .into());
        }
    }
    ensure_finite_scalar_estimation("survival_fit.log_likelihood", log_likelihood)
        .map_err(|e| e.to_string())?;
    ensure_finite_scalar_estimation("survival_fit.reml_score", reml_score)
        .map_err(|e| e.to_string())?;
    ensure_finite_scalar_estimation("survival_fit.stable_penalty_term", stable_penalty_term)
        .map_err(|e| e.to_string())?;
    ensure_finite_scalar_estimation("survival_fit.penalized_objective", penalized_objective)
        .map_err(|e| e.to_string())?;
    if let Some(g) = outer_gradient_norm {
        ensure_finite_scalar_estimation("survival_fit.outer_gradient_norm", g)
            .map_err(|e| e.to_string())?;
    }

    let total_p = beta_time.len()
        + beta_threshold.len()
        + beta_log_sigma.len()
        + beta_link_wiggle.as_ref().map_or(0, |beta| beta.len());
    if let Some(cov) = covariance_conditional.as_ref() {
        validate_all_finite_estimation("survival_fit.covariance_conditional", cov.iter().copied())
            .map_err(|e| e.to_string())?;
        let (rows, cols) = cov.dim();
        if rows != total_p || cols != total_p {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "survival_fit.covariance_conditional must be {}x{}, got {}x{}",
                    total_p, total_p, rows, cols
                ),
            }
            .into());
        }
    }
    if let Some(geom) = geometry.as_ref() {
        geom.validate_numeric_finiteness()
            .map_err(|e| e.to_string())?;
        let (rows, cols) = geom.penalized_hessian.dim();
        if rows != total_p || cols != total_p {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "survival_fit.geometry.penalized_hessian must be {}x{}, got {}x{}",
                    total_p, total_p, rows, cols
                ),
            }
            .into());
        }
        if geom.working_weights.len() != geom.working_response.len() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival_fit.geometry working length mismatch: weights={}, response={}",
                    geom.working_weights.len(),
                    geom.working_response.len()
                ),
            }
            .into());
        }
    }

    // Build blocks for the unified representation.
    use crate::solver::estimate::{BlockRole, FittedBlock, FittedLinkState, UnifiedFitResultParts};
    let mut blocks = vec![
        FittedBlock {
            beta: beta_time.clone(),
            role: BlockRole::Time,
            edf: 0.0,
            lambdas: lambdas_time.clone(),
        },
        FittedBlock {
            beta: beta_threshold.clone(),
            role: BlockRole::Threshold,
            edf: 0.0,
            lambdas: lambdas_threshold.clone(),
        },
        FittedBlock {
            beta: beta_log_sigma.clone(),
            role: BlockRole::Scale,
            edf: 0.0,
            lambdas: lambdas_log_sigma.clone(),
        },
    ];
    if let Some(ref bw) = beta_link_wiggle {
        blocks.push(FittedBlock {
            beta: bw.clone(),
            role: BlockRole::LinkWiggle,
            edf: 0.0,
            lambdas: lambdas_linkwiggle
                .clone()
                .unwrap_or_else(|| Array1::zeros(0)),
        });
    }
    let all_lambdas: Vec<f64> = blocks
        .iter()
        .flat_map(|b| b.lambdas.iter().copied())
        .collect();
    let log_lambdas = Array1::from_vec(
        all_lambdas
            .iter()
            .map(|&v| if v > 0.0 { v.ln() } else { f64::NEG_INFINITY })
            .collect(),
    );
    let deviance = -2.0 * log_likelihood;
    crate::solver::estimate::UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks,
        log_lambdas,
        lambdas: Array1::from_vec(all_lambdas),
        likelihood_family: None,
        likelihood_scale: crate::types::LikelihoodScaleMetadata::Unspecified,
        log_likelihood_normalization: crate::types::LogLikelihoodNormalization::UserProvided,
        log_likelihood,
        deviance,
        reml_score,
        stable_penalty_term,
        penalized_objective,
        outer_iterations,
        outer_converged,
        outer_gradient_norm,
        standard_deviation: 1.0,
        covariance_conditional,
        covariance_corrected: None,
        inference: None,
        fitted_link: FittedLinkState::Standard(None),
        geometry,
        block_states: Vec::new(),
        pirls_status: crate::pirls::PirlsStatus::Converged,
        max_abs_eta: 0.0,
        constraint_kkt: None,
        artifacts: crate::solver::estimate::FitArtifacts {
            pirls: None,
            null_space_logdet: None,
            null_space_dim: None,
            survival_link_wiggle_knots: link_wiggle_knots,
            survival_link_wiggle_degree: link_wiggle_degree,
            criterion_certificate: None,
            rho_posterior_certificate: None,
            rho_posterior_escalation: None,
        },
        inner_cycles: 0,
    })
    .map_err(|e| e.to_string())
}


#[derive(Clone)]
pub struct SurvivalLocationScalePredictInput {
    pub x_time_exit: Array2<f64>,
    pub eta_time_offset_exit: Array1<f64>,
    pub time_wiggle_knots: Option<Array1<f64>>,
    pub time_wiggle_degree: Option<usize>,
    pub time_wiggle_ncols: usize,
    pub x_threshold: DesignMatrix,
    pub eta_threshold_offset: Array1<f64>,
    pub x_log_sigma: DesignMatrix,
    pub eta_log_sigma_offset: Array1<f64>,
    pub x_link_wiggle: Option<DesignMatrix>,
    pub link_wiggle_knots: Option<Array1<f64>>,
    pub link_wiggle_degree: Option<usize>,
    pub inverse_link: InverseLink,
}


#[derive(Clone, Debug)]
pub struct SurvivalLocationScalePredictResult {
    pub eta: Array1<f64>,
    pub survival_prob: Array1<f64>,
}


#[derive(Clone)]
pub struct SurvivalLocationScalePredictUncertaintyResult {
    pub eta: Array1<f64>,
    pub survival_prob: Array1<f64>,
    pub eta_standard_error: Array1<f64>,
    pub response_standard_error: Option<Array1<f64>>,
}


#[derive(Clone)]
struct SurvivalLocationScaleFamily {
    n: usize,
    y: Array1<f64>,
    w: Array1<f64>,
    inverse_link: InverseLink,
    derivative_guard: f64,
    x_time_entry: Arc<Array2<f64>>,
    x_time_exit: Arc<Array2<f64>>,
    x_time_deriv: Arc<Array2<f64>>,
    time_wiggle_knots: Option<Array1<f64>>,
    time_wiggle_degree: Option<usize>,
    time_wiggle_ncols: usize,
    time_linear_constraints: Option<LinearInequalityConstraints>,
    /// Exit design for threshold block (always present; used as main design).
    x_threshold: DesignMatrix,
    /// Entry design for threshold block when time-varying.
    /// When `None`, the block is time-invariant: q0 = q1 (current behavior).
    x_threshold_entry: Option<DesignMatrix>,
    /// Exit-time derivative design for threshold when time-varying.
    x_threshold_deriv: Option<DesignMatrix>,
    /// Exit design for log-sigma block (always present; used as main design).
    x_log_sigma: DesignMatrix,
    /// Entry design for log-sigma block when time-varying.
    x_log_sigma_entry: Option<DesignMatrix>,
    /// Exit-time derivative design for log-sigma when time-varying.
    x_log_sigma_deriv: Option<DesignMatrix>,
    x_link_wiggle: Option<DesignMatrix>,
    wiggle_knots: Option<Array1<f64>>,
    wiggle_degree: Option<usize>,
    /// σ-scaled log-t AFT location baseline (issue #892). `Some` only in the
    /// rank-1 reduced parametric-AFT regime, where the time warp is removed
    /// (`h ≡ 0`) and the `log t` baseline rides the σ-scaled `q` (location)
    /// channel instead: the effective location is shifted to `η_t − log t` with a
    /// time-derivative `−1/t`, so `u = inv_sigma·(log t − η_t) = (log t − μ)/σ`
    /// and the event Jacobian gains `−log σ − log t`. `None` everywhere else.
    location_log_time: Option<LocationLogTimeOffset>,
    policy: crate::resource::ResourcePolicy,
}


/// The σ-scaled log-t AFT location baseline (issue #892), applied to the `q`
/// channel in the rank-1 reduced parametric-AFT regime. Each field is a per-row
/// shift of the effective location predictor (`η_t → η_t + value`, derivative
/// `+ deriv`), so the standardized residual becomes `inv_sigma·(log t − η_t)`.
#[derive(Clone, Debug)]
struct LocationLogTimeOffset {
    /// `−log t_exit`: shifts the exit-time effective location by `−log t`.
    value_exit: Array1<f64>,
    /// `−log t_entry`: shifts the entry-time effective location by `−log t`.
    value_entry: Array1<f64>,
    /// `−1/t_exit`: the exit-time derivative of the `−log t` location shift,
    /// feeding the `q`-channel `qdot` so `g` carries `inv_sigma/t`.
    deriv_exit: Array1<f64>,
}


#[derive(Clone, Copy)]
struct SurvivalPredictorState {
    h0: f64,
    h1: f64,
    g: f64,
    /// q evaluated at entry time. When the threshold/sigma blocks are
    /// time-invariant, q0 == q1.
    q0: f64,
    /// q evaluated at exit time.
    q1: f64,
    /// Explicit roundoff envelope from the compensated `d_raw + qdot`
    /// subtraction used to form `g`.
    g_roundoff_slack: f64,
    /// max(|d_raw|, |qdot|): kept only for diagnostics so monotonicity errors
    /// can report the scale of the operands that produced `g`.
    g_operand_scale: f64,
}


#[derive(Clone, Copy)]
struct SurvivalRowDerivatives {
    ll: f64,
    /// d ell / dq summed over entry+exit (= d1_q0 + d1_q1).
    d1_q: f64,
    /// d² ell / dq² summed (= d2_q0 + d2_q1 when q0=q1; used for time-invariant blocks).
    d2_q: f64,
    /// d³ ell / dq³ summed.
    d3_q: f64,
    /// Entry-only derivative: d ell / dq0 = w * r(u0).
    d1_q0: f64,
    /// Entry-only second derivative: d² ell / dq0² = w * r'(u0).
    d2_q0: f64,
    /// Entry-only third derivative: d³ ell / dq0³ = w * r''(u0).
    d3_q0: f64,
    /// Exit-only derivative: d ell / dq1.
    d1_q1: f64,
    /// Exit-only second derivative: d² ell / dq1².
    d2_q1: f64,
    /// Exit-only third derivative: d³ ell / dq1³.
    d3_q1: f64,
    /// Exit-only derivatives with respect to qdot1 = dq/dt at the event time.
    d1_qdot1: f64,
    d2_qdot1: f64,
    grad_time_eta_h0: f64,
    grad_time_eta_h1: f64,
    grad_time_eta_d: f64,
    h_time_h0: f64,
    h_time_h1: f64,
    h_time_d: f64,
    d_h_h0: f64,
    d_h_h1: f64,
    d_h_d: f64,
}


#[derive(Clone, Copy)]
struct SurvivalExactRowKernel {
    w: f64,
    d: f64,
    log_s0: f64,
    r0: f64,
    dr0: f64,
    ddr0: f64,
    dddr0: f64,
    log_s1: f64,
    r1: f64,
    dr1: f64,
    ddr1: f64,
    dddr1: f64,
    logphi1: f64,
    dlogphi1: f64,
    d2logphi1: f64,
    d3logphi1: f64,
    d4logphi1: f64,
    log_g: f64,
    d_log_g: f64,
    d2_log_g: f64,
    d3_log_g: f64,
    d4_log_g: f64,
}


/// Mix event and censored contributions, avoiding `0 * Inf = NaN` when
/// `d ∈ {0, 1}` and one branch is non-finite.
#[inline]
fn event_mix(d: f64, event_val: f64, censored_val: f64) -> f64 {
    if d == 1.0 {
        event_val
    } else if d == 0.0 {
        censored_val
    } else {
        d * event_val + (1.0 - d) * censored_val
    }
}


impl SurvivalExactRowKernel {
    #[inline]
    fn log_likelihood(self) -> f64 {
        self.w * (event_mix(self.d, self.logphi1 + self.log_g, self.log_s1) - self.log_s0)
    }

    #[inline]
    fn nll_index_tower(self) -> crate::families::jet_tower::Tower4<3> {
        use crate::families::jet_tower::Tower4;

        let u0 = Tower4::<3>::variable(0.0, 0);
        let u1 = Tower4::<3>::variable(0.0, 1);
        let g = Tower4::<3>::variable(0.0, 2);
        let mut nll = u0
            .compose_unary([self.log_s0, -self.r0, -self.dr0, -self.ddr0, -self.dddr0])
            .scale(self.w);

        let censored_weight = self.w * (1.0 - self.d);
        if censored_weight != 0.0 {
            nll = nll
                + u1.compose_unary([self.log_s1, -self.r1, -self.dr1, -self.ddr1, -self.dddr1])
                    .scale(-censored_weight);
        }

        let event_weight = self.w * self.d;
        if event_weight != 0.0 {
            nll = nll
                + u1.compose_unary([
                    self.logphi1,
                    self.dlogphi1,
                    self.d2logphi1,
                    self.d3logphi1,
                    self.d4logphi1,
                ])
                .scale(-event_weight)
                + g.compose_unary([
                    self.log_g,
                    self.d_log_g,
                    self.d2_log_g,
                    self.d3_log_g,
                    self.d4_log_g,
                ])
                .scale(-event_weight);
        }

        nll
    }
}


struct SurvivalJointQuantities {
    /// Per-row log-likelihood `ell_i` (NOT negated). Rows excluded by the
    /// degeneracy guard (`row_derivatives_rescaled` returns `None`) keep `0.0`,
    /// matching their zero derivative slots. The RowKernel adapter uses this
    /// to expose `nll_i = -ell_i` without recomputing row survival values.
    ll: Array1<f64>,
    d1_q: Array1<f64>,
    d2_q: Array1<f64>,
    d3_q: Array1<f64>,
    /// Entry-only derivatives of ell w.r.t. q0.
    d1_q0: Array1<f64>,
    d2_q0: Array1<f64>,
    d3_q0: Array1<f64>,
    /// Exit-only derivatives of ell w.r.t. q1.
    d1_q1: Array1<f64>,
    d2_q1: Array1<f64>,
    d3_q1: Array1<f64>,
    /// Exit-only derivatives of ell w.r.t. qdot1 = dq/dt.
    d1_qdot1: Array1<f64>,
    d2_qdot1: Array1<f64>,
    h_time_h0: Array1<f64>,
    h_time_h1: Array1<f64>,
    h_time_d: Array1<f64>,
    d_h_h0: Array1<f64>,
    d_h_h1: Array1<f64>,
    d_h_d: Array1<f64>,
    /// Exit-side dq/d(eta_t) = -exp(-eta_ls_exit).
    dq_t: Array1<f64>,
    /// Exit-side dq/d(eta_ls).
    dq_ls: Array1<f64>,
    d2q_tls: Array1<f64>,
    d2q_ls: Array1<f64>,
    d3q_tls_ls: Array1<f64>,
    d3q_ls: Array1<f64>,
    /// Entry-side dq0/d(eta_t_entry) = -exp(-eta_ls_entry) (only for time-varying).
    dq_t_entry: Option<Array1<f64>>,
    /// Entry-side q-chain derivatives at entry (only for time-varying sigma).
    dq_ls_entry: Option<Array1<f64>>,
    d2q_tls_entry: Option<Array1<f64>>,
    d2q_ls_entry: Option<Array1<f64>>,
    d3q_tls_ls_entry: Option<Array1<f64>>,
    d3q_ls_entry: Option<Array1<f64>>,
    dqdot_t: Array1<f64>,
    dqdot_ls: Array1<f64>,
    dqdot_td: Array1<f64>,
    dqdot_lsd: Array1<f64>,
    d2qdot_tt: Array1<f64>,
    d2qdot_tls: Array1<f64>,
    d2qdot_ttd: Array1<f64>,
    d2qdot_tlsd: Array1<f64>,
    d2qdot_ls: Array1<f64>,
    d2qdot_lstd: Array1<f64>,
    d2qdot_lslsd: Array1<f64>,
    d3qdot_tls_ls: Array1<f64>,
    d3qdot_tls_lsd: Array1<f64>,
    d3qdot_td_ls_ls: Array1<f64>,
    d3qdot_ls_ls_ls: Array1<f64>,
    d3qdot_ls_ls_lsd: Array1<f64>,
}


struct SurvivalJointPsiDirection {
    x_t_exit_psi: Option<Array2<f64>>,
    x_t_entry_psi: Option<Array2<f64>>,
    x_ls_exit_psi: Option<Array2<f64>>,
    x_ls_entry_psi: Option<Array2<f64>>,
    z_t_exit_psi: Array1<f64>,
    z_t_entry_psi: Array1<f64>,
    z_ls_exit_psi: Array1<f64>,
    z_ls_entry_psi: Array1<f64>,
    x_t_exit_action: Option<CustomFamilyPsiDesignAction>,
    x_t_entry_action: Option<CustomFamilyPsiDesignAction>,
    x_ls_exit_action: Option<CustomFamilyPsiDesignAction>,
    x_ls_entry_action: Option<CustomFamilyPsiDesignAction>,
}


fn split_survival_psi_design(
    x_psi: &Array2<f64>,
    n: usize,
    time_varying: bool,
    label: &str,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    if time_varying {
        if x_psi.nrows() != 2 * n && x_psi.nrows() != 3 * n {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "{label} stacked psi design row mismatch: got {}, expected {} or {}",
                    x_psi.nrows(),
                    2 * n,
                    3 * n,
                ),
            }
            .into());
        }
        Ok((
            x_psi.slice(s![0..n, ..]).to_owned(),
            x_psi.slice(s![n..2 * n, ..]).to_owned(),
        ))
    } else {
        if x_psi.nrows() != n {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "{label} psi design row mismatch: got {}, expected {}",
                    x_psi.nrows(),
                    n
                ),
            }
            .into());
        }
        Ok((x_psi.clone(), x_psi.clone()))
    }
}


/// Number of linear-predictor primary channels for the survival
/// location-scale row kernel (non-wiggle configurations).
///
/// The row likelihood `ell = w[d(log f(u1)+log g) + (1-d)log S(u1) - log S(u0)]`
/// depends on three indices `(u0, u1, g)`, each an **affine** function of the
/// model's linear predictors. We make those linear predictors the primary
/// space so the row Jacobian is fixed (the `RowKernel` framework requires
/// this), and fold the nonlinear scale map `q = -eta_t·exp(-eta_ls)` into the
/// per-row kernel. The nine channels are:
///
/// | idx | predictor       | design                              | feeds |
/// |-----|-----------------|-------------------------------------|-------|
/// | 0   | h0  (time entry)| `time_jac_entry`                    | u0    |
/// | 1   | h1  (time exit) | `time_jac_exit`                     | u1    |
/// | 2   | d_raw (time dot)| `time_jac_deriv`                    | g     |
/// | 3   | eta_t_exit      | `x_threshold`                       | u1, g |
/// | 4   | eta_t_entry     | `x_threshold_entry` (or threshold)  | u0    |
/// | 5   | eta_t_deriv     | `x_threshold_deriv` (or none)       | g     |
/// | 6   | eta_ls_exit     | `x_log_sigma`                       | u1, g |
/// | 7   | eta_ls_entry    | `x_log_sigma_entry` (or log_sigma)  | u0    |
/// | 8   | eta_ls_deriv    | `x_log_sigma_deriv` (or none)       | g     |
///
/// `H[a][b] = -Σ_i (ell_ii·D_i[a]·D_i[b] + ell_i·D2_i[a][b])` reproduces
/// `assemble_joint_hessian_from_quantities` term-for-term (verified by the
/// equivalence test). Indices `i ∈ {u0,u1,g}` are functionally independent so
/// the index-space derivative tensors are diagonal in `i`.
const SLS_ROW_K: usize = 9;


/// `RowKernel<9>` adapter for the survival location-scale joint likelihood
/// (non-wiggle path). Holds the per-β quantities already computed by
/// [`SurvivalLocationScaleFamily::collect_joint_quantities_rescaled`] and
/// [`SurvivalLocationScaleFamily::build_dynamic_geometry`]; every trait method
/// is a pure repackaging of those scalars into linear-predictor primary space,
/// so the math is identical to the bespoke assembly by construction.
struct SurvivalLsRowKernel<'a> {
    family: &'a SurvivalLocationScaleFamily,
    q: &'a SurvivalJointQuantities,
    dynamic: &'a SurvivalDynamicGeometry,
    /// Joint block offsets `[0, p_time, p_time+p_thr, p_total]` (3 blocks).
    offsets: Vec<usize>,
}


/// Per-index `(D, D2, D3)` map-derivative tensors for one row, plus the
/// index-space log-likelihood derivatives. `D[i][a] = ∂(index i)/∂(channel a)`,
/// `D2[i][a][b] = ∂²(index i)/∂a∂b`, `D3[i][a][b][c] = ∂³(index i)/∂a∂b∂c`.
struct SlsRowMaps {
    /// ell_i  = (ell_u0, ell_u1, ell_g)
    l1: [f64; 3],
    /// ell_ii = (ell_u0u0, ell_u1u1, ell_gg)
    l2: [f64; 3],
    /// ell_iii = (ell_u0u0u0, ell_u1u1u1, ell_ggg)
    l3: [f64; 3],
    d: [[f64; SLS_ROW_K]; 3],
    d2: [[[f64; SLS_ROW_K]; SLS_ROW_K]; 3],
    d3: [[[[f64; SLS_ROW_K]; SLS_ROW_K]; SLS_ROW_K]; 3],
}


impl SurvivalLsRowKernel<'_> {
    /// Resolve the design for a threshold/log-sigma channel, falling back to the
    /// exit design when the entry/derivative variant is absent (time-invariant).
    #[inline]
    fn entry_design<'b>(
        opt: &'b Option<DesignMatrix>,
        fallback: &'b DesignMatrix,
    ) -> &'b DesignMatrix {
        opt.as_ref().unwrap_or(fallback)
    }

    /// Build the per-row index/map derivative tensors from the cached scalars.
    /// Symmetric `D2`/`D3` entries are written in every permuted slot so the
    /// uniform accumulation loops never have to special-case ordering.
    fn row_maps(&self, row: usize) -> SlsRowMaps {
        let q = self.q;
        let mut m = SlsRowMaps {
            l1: [q.d1_q0[row], q.d1_q1[row], q.d1_qdot1[row]],
            l2: [q.d2_q0[row], q.d2_q1[row], q.d2_qdot1[row]],
            // ell_ggg = w·d·d3_log_g = -d_h_d (d_h_d stores the NLL-sign value).
            l3: [q.d3_q0[row], q.d3_q1[row], -q.d_h_d[row]],
            d: [[0.0; SLS_ROW_K]; 3],
            d2: [[[0.0; SLS_ROW_K]; SLS_ROW_K]; 3],
            d3: [[[[0.0; SLS_ROW_K]; SLS_ROW_K]; SLS_ROW_K]; 3],
        };
        // helper closures to set symmetric entries
        let set2 = |t: &mut [[f64; SLS_ROW_K]; SLS_ROW_K], a: usize, b: usize, v: f64| {
            t[a][b] = v;
            t[b][a] = v;
        };
        let set3 = |t: &mut [[[f64; SLS_ROW_K]; SLS_ROW_K]; SLS_ROW_K],
                    a: usize,
                    b: usize,
                    c: usize,
                    v: f64| {
            for &(i, j, k) in &[
                (a, b, c),
                (a, c, b),
                (b, a, c),
                (b, c, a),
                (c, a, b),
                (c, b, a),
            ] {
                t[i][j][k] = v;
            }
        };

        // Entry-side q-chain derivatives are always populated (equal to the
        // exit values in the time-invariant case).
        let dq_t_en = self.q.dq_t_entry.as_ref().map_or(q.dq_t[row], |a| a[row]);
        let dq_ls_en = self.q.dq_ls_entry.as_ref().map_or(q.dq_ls[row], |a| a[row]);
        let d2q_tls_en = self
            .q
            .d2q_tls_entry
            .as_ref()
            .map_or(q.d2q_tls[row], |a| a[row]);
        let d2q_ls_en = self
            .q
            .d2q_ls_entry
            .as_ref()
            .map_or(q.d2q_ls[row], |a| a[row]);
        let d3q_tls_ls_en = self
            .q
            .d3q_tls_ls_entry
            .as_ref()
            .map_or(q.d3q_tls_ls[row], |a| a[row]);
        let d3q_ls_en = self
            .q
            .d3q_ls_entry
            .as_ref()
            .map_or(q.d3q_ls[row], |a| a[row]);

        // Index 0: u0 = h0 + q0(eta_t_entry=ch4, eta_ls_entry=ch7).
        m.d[0][0] = 1.0;
        m.d[0][4] = dq_t_en;
        m.d[0][7] = dq_ls_en;
        set2(&mut m.d2[0], 4, 7, d2q_tls_en);
        m.d2[0][7][7] = d2q_ls_en;
        set3(&mut m.d3[0], 4, 7, 7, d3q_tls_ls_en);
        m.d3[0][7][7][7] = d3q_ls_en;

        // Index 1: u1 = h1 + q1(eta_t_exit=ch3, eta_ls_exit=ch6).
        m.d[1][1] = 1.0;
        m.d[1][3] = q.dq_t[row];
        m.d[1][6] = q.dq_ls[row];
        set2(&mut m.d2[1], 3, 6, q.d2q_tls[row]);
        m.d2[1][6][6] = q.d2q_ls[row];
        set3(&mut m.d3[1], 3, 6, 6, q.d3q_tls_ls[row]);
        m.d3[1][6][6][6] = q.d3q_ls[row];

        // Index 2: g = d_raw + qdot1(eta_t_exit=ch3, eta_t_deriv=ch5,
        // eta_ls_exit=ch6, eta_ls_deriv=ch8).
        m.d[2][2] = 1.0;
        m.d[2][3] = q.dqdot_t[row];
        m.d[2][5] = q.dqdot_td[row];
        m.d[2][6] = q.dqdot_ls[row];
        m.d[2][8] = q.dqdot_lsd[row];
        m.d2[2][3][3] = q.d2qdot_tt[row];
        set2(&mut m.d2[2], 3, 6, q.d2qdot_tls[row]);
        set2(&mut m.d2[2], 3, 5, q.d2qdot_ttd[row]);
        set2(&mut m.d2[2], 3, 8, q.d2qdot_tlsd[row]);
        m.d2[2][6][6] = q.d2qdot_ls[row];
        set2(&mut m.d2[2], 6, 5, q.d2qdot_lstd[row]);
        set2(&mut m.d2[2], 6, 8, q.d2qdot_lslsd[row]);
        set3(&mut m.d3[2], 3, 6, 6, q.d3qdot_tls_ls[row]);
        set3(&mut m.d3[2], 3, 6, 8, q.d3qdot_tls_lsd[row]);
        set3(&mut m.d3[2], 5, 6, 6, q.d3qdot_td_ls_ls[row]);
        m.d3[2][6][6][6] = q.d3qdot_ls_ls_ls[row];
        set3(&mut m.d3[2], 6, 6, 8, q.d3qdot_ls_ls_lsd[row]);

        m
    }

    /// Per-row dense design row for each channel within its coefficient block:
    /// returns `(block_index, row_vector)` for channels `0..9`. Used by the
    /// pullback / diagonal assembly. Channels with an absent derivative design
    /// (time-invariant derivative channels) return `None` and contribute
    /// nothing.
    fn channel_block(&self, ch: usize) -> Option<usize> {
        match ch {
            0 | 1 | 2 => Some(Self::THRESHOLD_BLOCK_TIME),
            3 | 4 | 5 => Some(Self::THRESHOLD_BLOCK_THR),
            6 | 7 | 8 => Some(Self::THRESHOLD_BLOCK_LS),
            _ => None,
        }
    }
    const THRESHOLD_BLOCK_TIME: usize = 0;
    const THRESHOLD_BLOCK_THR: usize = 1;
    const THRESHOLD_BLOCK_LS: usize = 2;

    /// Dense per-row design vector for `channel` (length = its block width), or
    /// `None` when the channel's design is absent (time-invariant deriv channel,
    /// which carries no coefficients of its own).
    fn channel_row(&self, ch: usize, row: usize) -> Option<Array1<f64>> {
        let fam = self.family;
        match ch {
            0 => Some(self.dynamic.time_jac_entry.row(row).to_owned()),
            1 => Some(self.dynamic.time_jac_exit.row(row).to_owned()),
            2 => Some(self.dynamic.time_jac_deriv.row(row).to_owned()),
            3 => Some(design_dense_row(&fam.x_threshold, row)),
            4 => Some(design_dense_row(
                Self::entry_design(&fam.x_threshold_entry, &fam.x_threshold),
                row,
            )),
            5 => fam
                .x_threshold_deriv
                .as_ref()
                .map(|d| design_dense_row(d, row)),
            6 => Some(design_dense_row(&fam.x_log_sigma, row)),
            7 => Some(design_dense_row(
                Self::entry_design(&fam.x_log_sigma_entry, &fam.x_log_sigma),
                row,
            )),
            8 => fam
                .x_log_sigma_deriv
                .as_ref()
                .map(|d| design_dense_row(d, row)),
            _ => None,
        }
    }
}


/// Materialize `X[row, :]` as a dense length-`ncols` vector (no sparse-aware
/// fast path — used only by the dense-Hessian / diagonal assembly, never the
/// hot matvec inner loop).
fn design_dense_row(d: &DesignMatrix, row: usize) -> Array1<f64> {
    let mut out = Array1::<f64>::zeros(d.ncols());
    d.axpy_row_into(row, 1.0, &mut out.view_mut())
        .expect("design_dense_row: ncols-sized buffer matches design width");
    out
}


/// Accumulate `alpha * jac[row, :]` into the coefficient slice `out` for a dense
/// time Jacobian (the survival time block is materialized densely as
/// `time_jac_*`, so it has no sparse axpy primitive).
#[inline]
fn axpy_dense_row_into(jac: &Array2<f64>, row: usize, alpha: f64, out: &mut [f64]) {
    if alpha == 0.0 {
        return;
    }
    let jr = jac.row(row);
    for (o, &j) in out.iter_mut().zip(jr.iter()) {
        *o += alpha * j;
    }
}


fn row_set_from_survival_mask(
    row_mask: Option<&Array1<f64>>,
    n: usize,
) -> crate::families::row_kernel::RowSet {
    let Some(mask) = row_mask else {
        return crate::families::row_kernel::RowSet::All;
    };
    let rows = mask
        .iter()
        .enumerate()
        .filter_map(|(index, &weight)| {
            (weight != 0.0).then_some(crate::families::marginal_slope_shared::WeightedOuterRow {
                index,
                weight,
                stratum: 0,
            })
        })
        .collect::<Vec<_>>();
    crate::families::row_kernel::RowSet::Subsample {
        rows: Arc::new(rows),
        n_full: n,
    }
}


impl crate::families::row_kernel::RowKernel<SLS_ROW_K> for SurvivalLsRowKernel<'_> {
    fn n_rows(&self) -> usize {
        self.family.n
    }

    fn n_coefficients(&self) -> usize {
        *self.offsets.last().expect("offsets has block bounds")
    }

    fn row_kernel(
        &self,
        row: usize,
    ) -> Result<(f64, [f64; SLS_ROW_K], [[f64; SLS_ROW_K]; SLS_ROW_K]), String> {
        let m = self.row_maps(row);
        // NLL = -ell. Gradient and Hessian carry the overall minus sign.
        let mut grad = [0.0_f64; SLS_ROW_K];
        let mut hess = [[0.0_f64; SLS_ROW_K]; SLS_ROW_K];
        for i in 0..3 {
            let l1 = m.l1[i];
            let l2 = m.l2[i];
            let di = &m.d[i];
            for a in 0..SLS_ROW_K {
                grad[a] -= l1 * di[a];
                if di[a] != 0.0 {
                    for b in 0..SLS_ROW_K {
                        hess[a][b] -= l2 * di[a] * di[b];
                    }
                }
            }
            let d2i = &m.d2[i];
            for a in 0..SLS_ROW_K {
                for b in 0..SLS_ROW_K {
                    if d2i[a][b] != 0.0 {
                        hess[a][b] -= l1 * d2i[a][b];
                    }
                }
            }
        }
        Ok((-self.q.ll[row], grad, hess))
    }

    fn jacobian_action(&self, row: usize, d_beta: &[f64]) -> [f64; SLS_ROW_K] {
        let d_beta = ndarray::ArrayView1::from(d_beta);
        let d_time = d_beta.slice(s![self.offsets[0]..self.offsets[1]]);
        let d_thr = d_beta.slice(s![self.offsets[1]..self.offsets[2]]);
        let d_ls = d_beta.slice(s![self.offsets[2]..self.offsets[3]]);
        let fam = self.family;
        let t_entry = Self::entry_design(&fam.x_threshold_entry, &fam.x_threshold);
        let ls_entry = Self::entry_design(&fam.x_log_sigma_entry, &fam.x_log_sigma);
        let ch5 = fam
            .x_threshold_deriv
            .as_ref()
            .map_or(0.0, |d| d.dot_row_view(row, d_thr));
        let ch8 = fam
            .x_log_sigma_deriv
            .as_ref()
            .map_or(0.0, |d| d.dot_row_view(row, d_ls));
        [
            self.dynamic.time_jac_entry.row(row).dot(&d_time),
            self.dynamic.time_jac_exit.row(row).dot(&d_time),
            self.dynamic.time_jac_deriv.row(row).dot(&d_time),
            fam.x_threshold.dot_row_view(row, d_thr),
            t_entry.dot_row_view(row, d_thr),
            ch5,
            fam.x_log_sigma.dot_row_view(row, d_ls),
            ls_entry.dot_row_view(row, d_ls),
            ch8,
        ]
    }

    fn jacobian_transpose_action(&self, row: usize, v: &[f64; SLS_ROW_K], out: &mut [f64]) {
        let fam = self.family;
        // Time block: channels 0,1,2 via the dense time Jacobians.
        {
            let time = &mut out[self.offsets[0]..self.offsets[1]];
            axpy_dense_row_into(&self.dynamic.time_jac_entry, row, v[0], time);
            axpy_dense_row_into(&self.dynamic.time_jac_exit, row, v[1], time);
            axpy_dense_row_into(&self.dynamic.time_jac_deriv, row, v[2], time);
        }
        // Threshold block: channels 3 (exit), 4 (entry), 5 (deriv).
        {
            let mut thr = ndarray::ArrayViewMut1::from(&mut out[self.offsets[1]..self.offsets[2]]);
            fam.x_threshold
                .axpy_row_into(row, v[3], &mut thr)
                .expect("threshold exit axpy");
            Self::entry_design(&fam.x_threshold_entry, &fam.x_threshold)
                .axpy_row_into(row, v[4], &mut thr)
                .expect("threshold entry axpy");
            if let Some(d) = fam.x_threshold_deriv.as_ref() {
                d.axpy_row_into(row, v[5], &mut thr)
                    .expect("threshold deriv axpy");
            }
        }
        // Log-sigma block: channels 6 (exit), 7 (entry), 8 (deriv).
        {
            let mut ls = ndarray::ArrayViewMut1::from(&mut out[self.offsets[2]..self.offsets[3]]);
            fam.x_log_sigma
                .axpy_row_into(row, v[6], &mut ls)
                .expect("log_sigma exit axpy");
            Self::entry_design(&fam.x_log_sigma_entry, &fam.x_log_sigma)
                .axpy_row_into(row, v[7], &mut ls)
                .expect("log_sigma entry axpy");
            if let Some(d) = fam.x_log_sigma_deriv.as_ref() {
                d.axpy_row_into(row, v[8], &mut ls)
                    .expect("log_sigma deriv axpy");
            }
        }
    }

    fn add_pullback_hessian(
        &self,
        row: usize,
        h: &[[f64; SLS_ROW_K]; SLS_ROW_K],
        target: &mut Array2<f64>,
    ) {
        // Materialize each channel's dense block row once, then accumulate
        // h[a][b]·(row_a ⊗ row_b) into the (block_a, block_b) sub-block.
        let rows: Vec<Option<(usize, Array1<f64>)>> = (0..SLS_ROW_K)
            .map(|ch| self.channel_block(ch).zip(self.channel_row(ch, row)))
            .collect();
        for a in 0..SLS_ROW_K {
            let Some((ba, ra)) = rows[a].as_ref() else {
                continue;
            };
            let off_a = self.offsets[*ba];
            for b in 0..SLS_ROW_K {
                let hab = h[a][b];
                if hab == 0.0 {
                    continue;
                }
                let Some((bb, rb)) = rows[b].as_ref() else {
                    continue;
                };
                let off_b = self.offsets[*bb];
                for (ia, &va) in ra.iter().enumerate() {
                    if va == 0.0 {
                        continue;
                    }
                    let w = hab * va;
                    let mut trow = target.row_mut(off_a + ia);
                    for (ib, &vb) in rb.iter().enumerate() {
                        trow[off_b + ib] += w * vb;
                    }
                }
            }
        }
    }

    fn add_diagonal_quadratic(
        &self,
        row: usize,
        h: &[[f64; SLS_ROW_K]; SLS_ROW_K],
        diag: &mut [f64],
    ) {
        // diag[c] += Σ_{a,b ∈ block(c)} h[a][b]·row_a[c]·row_b[c]. Only
        // same-block channel pairs touch a given coefficient's diagonal slot.
        let rows: Vec<Option<(usize, Array1<f64>)>> = (0..SLS_ROW_K)
            .map(|ch| self.channel_block(ch).zip(self.channel_row(ch, row)))
            .collect();
        for a in 0..SLS_ROW_K {
            let Some((ba, ra)) = rows[a].as_ref() else {
                continue;
            };
            for b in 0..SLS_ROW_K {
                let hab = h[a][b];
                if hab == 0.0 {
                    continue;
                }
                let Some((bb, rb)) = rows[b].as_ref() else {
                    continue;
                };
                if ba != bb {
                    continue;
                }
                let off = self.offsets[*ba];
                for (k, (&va, &vb)) in ra.iter().zip(rb.iter()).enumerate() {
                    diag[off + k] += hab * va * vb;
                }
            }
        }
    }

    fn row_third_contracted(
        &self,
        row: usize,
        dir: &[f64; SLS_ROW_K],
    ) -> Result<[[f64; SLS_ROW_K]; SLS_ROW_K], String> {
        let m = self.row_maps(row);
        // Δ_i = Σ_c D_i[c]·dir[c]  (rate of change of index i along dir).
        // dD_i[a] = Σ_c D2_i[a][c]·dir[c]; dD2_i[a][b] = Σ_c D3_i[a][b][c]·dir[c].
        // d(ell_ii)/dt = ell_iii·Δ_i; d(ell_i)/dt = ell_ii·Δ_i.
        // dH[a][b] = -Σ_i [ ell_iii·Δ_i·D_i[a]·D_i[b]
        //                 + ell_ii·(dD_i[a]·D_i[b] + D_i[a]·dD_i[b])
        //                 + ell_ii·Δ_i·D2_i[a][b]
        //                 + ell_i·dD2_i[a][b] ].
        let mut out = [[0.0_f64; SLS_ROW_K]; SLS_ROW_K];
        for i in 0..3 {
            let di = &m.d[i];
            let d2i = &m.d2[i];
            let d3i = &m.d3[i];
            let mut delta = 0.0;
            let mut dd = [0.0_f64; SLS_ROW_K];
            for c in 0..SLS_ROW_K {
                let s = dir[c];
                if s == 0.0 {
                    continue;
                }
                delta += di[c] * s;
                for a in 0..SLS_ROW_K {
                    dd[a] += d2i[a][c] * s;
                }
            }
            let l2 = m.l2[i];
            let l3 = m.l3[i];
            let l1 = m.l1[i];
            for a in 0..SLS_ROW_K {
                for b in 0..SLS_ROW_K {
                    let mut t = l3 * delta * di[a] * di[b]
                        + l2 * (dd[a] * di[b] + di[a] * dd[b])
                        + l2 * delta * d2i[a][b];
                    if l1 != 0.0 {
                        let mut dd2 = 0.0;
                        for c in 0..SLS_ROW_K {
                            let s = dir[c];
                            if s != 0.0 {
                                dd2 += d3i[a][b][c] * s;
                            }
                        }
                        t += l1 * dd2;
                    }
                    out[a][b] -= t;
                }
            }
        }
        Ok(out)
    }

    fn row_fourth_contracted(
        &self,
        row: usize,
        dir_u: &[f64; SLS_ROW_K],
        dir_v: &[f64; SLS_ROW_K],
    ) -> Result<[[f64; SLS_ROW_K]; SLS_ROW_K], String> {
        // The survival location-scale family carries derivative quantities only
        // up to third order (`d_h_*` are third index derivatives; the fourth
        // index derivatives `dddr0` / `d4logphi1` are computed in
        // `exact_*_derivatives_fourth_rescaled` but deliberately not stored).
        // Its REML outer Hessian is assembled from the **third-order**
        // directional-derivative operator, never an explicit fourth-order
        // tensor, so this entry point is not on the location-scale path. Routing
        // through the generic `row_kernel_second_directional_derivative` would
        // require persisting the fourth index derivatives first.
        let u_norm = dir_u.iter().map(|value| value * value).sum::<f64>().sqrt();
        let v_norm = dir_v.iter().map(|value| value * value).sum::<f64>().sqrt();
        Err(format!(
            "survival location-scale RowKernel does not provide a fourth-order \
             contracted derivative at row {row} (u_norm={u_norm:.6e}, \
             v_norm={v_norm:.6e}): the family's REML uses the third-order \
             directional operator (no fourth-order tensor is computed)"
        ))
    }
}


impl SurvivalLocationScaleFamily {
    const BLOCK_TIME: usize = 0;
    const BLOCK_THRESHOLD: usize = 1;
    const BLOCK_LOG_SIGMA: usize = 2;
    const BLOCK_LINK_WIGGLE: usize = 3;
    const EVALUATE_PARALLEL_ROW_THRESHOLD: usize = 1024;

    /// The `RowKernel<K>` engine assumes a fixed linear coefficient-to-primary
    /// Jacobian for the row. Survival LS satisfies that after choosing the nine
    /// linear predictors as primary channels, but link-wiggle does not: its
    /// basis rows are evaluated at q(eta_threshold, eta_log_sigma), so the row
    /// design itself changes with beta and contributes dJ/dβ terms outside the
    /// current trait contract.
    #[inline]
    fn row_kernel_joint_hessian_supported(&self) -> bool {
        self.x_link_wiggle.is_none()
    }

    /// First directional derivatives require third qdot map derivatives when
    /// threshold/log-sigma derivative designs are present; those live in
    /// `SurvivalJointQuantities`, so every non-wiggle shape can use the
    /// `RowKernel<9>` path.
    #[inline]
    fn row_kernel_directional_supported(&self) -> bool {
        self.x_link_wiggle.is_none()
    }

    fn survival_ls_row_kernel<'a>(
        &'a self,
        q: &'a SurvivalJointQuantities,
        dynamic: &'a SurvivalDynamicGeometry,
    ) -> SurvivalLsRowKernel<'a> {
        SurvivalLsRowKernel {
            family: self,
            q,
            dynamic,
            offsets: self.joint_block_offsets(),
        }
    }

    #[inline]
    fn time_wiggle_range(&self) -> std::ops::Range<usize> {
        let p_total = self.x_time_exit.ncols();
        let p_w = self.time_wiggle_ncols.min(p_total);
        p_total - p_w..p_total
    }

    #[inline]
    fn time_derivative_lower_bound(&self) -> f64 {
        assert!(
            self.derivative_guard.is_finite() && self.derivative_guard > 0.0,
            "survival location-scale derivative guard must be finite and positive: derivative_guard={}",
            self.derivative_guard
        );
        self.derivative_guard
    }

    fn max_feasible_time_step(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        let Some(constraints) = self.time_linear_constraints.as_ref() else {
            // No time constraints. With the rank-1 unit-log-t warp pin (#892) the
            // time block has ZERO free coefficients and its monotone warp is a
            // fixed positive offset (X' z_norm = 1/t > 0), so there is no
            // derivative-guard half-space to cap against — the step is uncapped.
            // (Every constrained time block, reduced or flexible, carries ≥1
            // column and a guard, so this `None` arises only for the pinned
            // empty block.)
            return Ok(None);
        };
        crate::families::marginal_slope_shared::feasible_step_fraction(
            constraints,
            beta,
            delta,
            |beta_len, delta_len, expected| {
                SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                    "survival location-scale time-step constraint dimension mismatch: beta={beta_len}, delta={delta_len}, constraints={expected}"
                ) }.into()
            },
            |row, slack| {
                SurvivalLocationScaleError::ConstraintViolation { reason: format!(
                    "survival location-scale current time block violates linear constraint at row {row}: slack={slack:.3e}"
                ) }.into()
            },
        )
        .map(Some)
    }

    fn max_feasible_link_wiggle_step(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        if beta.len() != delta.len() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival location-scale linkwiggle-step dimension mismatch: beta={}, delta={}",
                    beta.len(),
                    delta.len()
                ),
            }
            .into());
        }
        let mut alpha = 1.0f64;
        for j in 0..beta.len() {
            let slack = beta[j];
            if slack < -CONSTRAINT_NONNEGATIVITY_REL_TOL {
                return Err(SurvivalLocationScaleError::ConstraintViolation { reason: format!(
                    "survival location-scale current linkwiggle block violates nonnegativity at coefficient {j}: beta={slack:.3e}"
                ) }.into());
            }
            let drift = delta[j];
            if drift < 0.0 {
                alpha = alpha.min((slack / -drift).clamp(0.0, 1.0));
            }
        }
        if alpha >= 1.0 {
            Ok(Some(1.0))
        } else {
            Ok(Some((0.995 * alpha).clamp(0.0, 1.0)))
        }
    }

    #[inline]
    fn expected_blocks(&self) -> usize {
        if self.x_link_wiggle.is_some() { 4 } else { 3 }
    }

    #[inline]
    fn joint_block_dims(&self) -> Vec<usize> {
        let mut dims = vec![
            self.x_time_entry.ncols(),
            self.x_threshold.ncols(),
            self.x_log_sigma.ncols(),
        ];
        if let Some(xw) = self.x_link_wiggle.as_ref() {
            dims.push(xw.ncols());
        }
        dims
    }

    fn validate_joint_specs(
        &self,
        specs: &[ParameterBlockSpec],
        context: &str,
    ) -> Result<(), String> {
        let dims = self.joint_block_dims();
        if specs.len() != dims.len() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "{context} expects {} specs, got {}",
                    dims.len(),
                    specs.len()
                ),
            }
            .into());
        }
        for (block_idx, (spec, expected_width)) in specs.iter().zip(dims.iter()).enumerate() {
            let width = spec.design.ncols();
            if width != *expected_width {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "{context} spec {block_idx} width mismatch: got {width}, expected {expected_width}"
                    ),
                }
                .into());
            }
        }
        Ok(())
    }

    #[inline]
    fn joint_block_offsets(&self) -> Vec<usize> {
        let dims = self.joint_block_dims();
        let mut offsets = Vec::with_capacity(dims.len() + 1);
        offsets.push(0);
        let mut acc = 0usize;
        for dim in dims {
            acc += dim;
            offsets.push(acc);
        }
        offsets
    }

    fn wiggle_geometry(
        &self,
        q0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<SurvivalWiggleGeometry>, String> {
        let (Some(knots), Some(degree)) = (self.wiggle_knots.as_ref(), self.wiggle_degree) else {
            return Ok(None);
        };
        let basis = survival_wiggle_basis_with_options(q0, knots, degree, BasisOptions::value())?;
        let basis_d1 = survival_wiggle_basis_with_options(
            q0,
            knots,
            degree,
            BasisOptions::first_derivative(),
        )?;
        let basis_d2 = survival_wiggle_basis_with_options(
            q0,
            knots,
            degree,
            BasisOptions::second_derivative(),
        )?;
        let basis_d3 = survival_wiggle_third_basis(q0, knots, degree)?;
        if basis.ncols() != beta_w.len()
            || basis_d1.ncols() != beta_w.len()
            || basis_d2.ncols() != beta_w.len()
            || basis_d3.ncols() != beta_w.len()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival linkwiggle basis/beta mismatch: B={} B'={} B''={} B'''={} betaw={}",
                    basis.ncols(),
                    basis_d1.ncols(),
                    basis_d2.ncols(),
                    basis_d3.ncols(),
                    beta_w.len()
                ),
            }
            .into());
        }
        let dq_dq0 = fast_av(&basis_d1, &beta_w) + 1.0;
        let d2q_dq02 = fast_av(&basis_d2, &beta_w);
        let d3q_dq03 = fast_av(&basis_d3, &beta_w);
        Ok(Some(SurvivalWiggleGeometry {
            basis,
            basis_d1,
            basis_d2,
            dq_dq0,
            d2q_dq02,
            d3q_dq03,
        }))
    }

    fn time_wiggle_geometry(
        &self,
        h0: ndarray::ArrayView1<'_, f64>,
        beta_w: ndarray::ArrayView1<'_, f64>,
    ) -> Result<Option<SurvivalWiggleGeometry>, String> {
        let (Some(knots), Some(degree)) =
            (self.time_wiggle_knots.as_ref(), self.time_wiggle_degree)
        else {
            return Ok(None);
        };
        let basis = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 0)?;
        let basis_d1 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 1)?;
        let basis_d2 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 2)?;
        let basis_d3 = monotone_wiggle_basis_with_derivative_order(h0, knots, degree, 3)?;
        if basis.ncols() != beta_w.len()
            || basis_d1.ncols() != beta_w.len()
            || basis_d2.ncols() != beta_w.len()
            || basis_d3.ncols() != beta_w.len()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival timewiggle basis/beta mismatch: B={} B'={} B''={} B'''={} betaw={}",
                    basis.ncols(),
                    basis_d1.ncols(),
                    basis_d2.ncols(),
                    basis_d3.ncols(),
                    beta_w.len()
                ),
            }
            .into());
        }
        let dq = fast_av(&basis_d1, &beta_w) + 1.0;
        let d2 = fast_av(&basis_d2, &beta_w);
        let d3 = fast_av(&basis_d3, &beta_w);
        Ok(Some(SurvivalWiggleGeometry {
            basis,
            basis_d1,
            basis_d2,
            dq_dq0: dq,
            d2q_dq02: d2,
            d3q_dq03: d3,
        }))
    }

    /// Returns
    /// `(h0, h1, d_raw, eta_t_exit, eta_ls_exit, eta_t_entry, eta_ls_entry,
    ///   eta_t_deriv_exit, eta_ls_deriv_exit, etaw)`.
    ///
    /// The time block eta is stored as `[exit; entry; derivative_exit]` to
    /// match the stacked design, but callers consume `(entry, exit, deriv)`.
    /// For time-invariant blocks, `eta_t_entry == eta_t_exit` and likewise for ls.
    /// For time-varying threshold/log-sigma blocks, the block eta is 3n long:
    /// `[exit; entry; derivative_exit]`.
    /// The solver's ParameterBlockSpec uses the EXIT value design first.
    fn validate_joint_states<'a>(
        &self,
        block_states: &'a [ParameterBlockState],
    ) -> Result<
        (
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            ndarray::ArrayView1<'a, f64>,
            Option<ndarray::ArrayView1<'a, f64>>,
            Option<ndarray::ArrayView1<'a, f64>>,
            Option<&'a Array1<f64>>,
        ),
        String,
    > {
        if block_states.len() != self.expected_blocks() {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "SurvivalLocationScaleFamily expects {} blocks, got {}",
                    self.expected_blocks(),
                    block_states.len()
                ),
            }
            .into());
        }
        let n = self.n;
        let eta_time = &block_states[Self::BLOCK_TIME].eta;
        let eta_t_raw = &block_states[Self::BLOCK_THRESHOLD].eta;
        let eta_ls_raw = &block_states[Self::BLOCK_LOG_SIGMA].eta;
        let etaw = self
            .x_link_wiggle
            .as_ref()
            .map(|_| &block_states[Self::BLOCK_LINK_WIGGLE].eta);
        if eta_time.len() != 3 * n {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival location-scale time eta length mismatch: got {}, expected {}",
                    eta_time.len(),
                    3 * n
                ),
            }
            .into());
        }
        // For time-varying blocks the stacked design is
        // [exit_design; entry_design; derivative_exit_design], giving eta of
        // length 3n. For time-invariant blocks eta is length n.
        let (eta_t_exit, eta_t_entry, eta_t_deriv_exit) = if self.x_threshold_entry.is_some() {
            if eta_t_raw.len() != 3 * n {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "time-varying threshold eta length mismatch: got {}, expected {}",
                        eta_t_raw.len(),
                        3 * n
                    ),
                }
                .into());
            }
            (
                eta_t_raw.slice(s![0..n]),
                eta_t_raw.slice(s![n..2 * n]),
                Some(eta_t_raw.slice(s![2 * n..3 * n])),
            )
        } else {
            if eta_t_raw.len() != n {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "threshold eta length mismatch: got {}, expected {n}",
                        eta_t_raw.len()
                    ),
                }
                .into());
            }
            (eta_t_raw.slice(s![0..n]), eta_t_raw.slice(s![0..n]), None)
        };
        let (eta_ls_exit, eta_ls_entry, eta_ls_deriv_exit) = if self.x_log_sigma_entry.is_some() {
            if eta_ls_raw.len() != 3 * n {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "time-varying log-sigma eta length mismatch: got {}, expected {}",
                        eta_ls_raw.len(),
                        3 * n
                    ),
                }
                .into());
            }
            (
                eta_ls_raw.slice(s![0..n]),
                eta_ls_raw.slice(s![n..2 * n]),
                Some(eta_ls_raw.slice(s![2 * n..3 * n])),
            )
        } else {
            if eta_ls_raw.len() != n {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "log-sigma eta length mismatch: got {}, expected {n}",
                        eta_ls_raw.len()
                    ),
                }
                .into());
            }
            (eta_ls_raw.slice(s![0..n]), eta_ls_raw.slice(s![0..n]), None)
        };
        if let Some(w) = etaw
            && w.len() != n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival location-scale wiggle eta length mismatch: got {}, expected {n}",
                    w.len()
                ),
            }
            .into());
        }
        Ok((
            eta_time.slice(s![n..2 * n]),
            eta_time.slice(s![0..n]),
            eta_time.slice(s![2 * n..3 * n]),
            eta_t_exit,
            eta_ls_exit,
            eta_t_entry,
            eta_ls_entry,
            eta_t_deriv_exit,
            eta_ls_deriv_exit,
            etaw,
        ))
    }

    fn collect_joint_quantities(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalJointQuantities, String> {
        self.collect_joint_quantities_rescaled(block_states, 0.0)
    }

    /// Collect per-row derivative quantities with a log-scale shift applied
    /// to the derivative magnitudes.  When `deriv_log_scale > 0`, all
    /// derivative arrays are uniformly scaled by `exp(-deriv_log_scale)`.
    /// The caller must account for this in the logdet:
    ///   `logdet(H) = logdet(H_scaled) + p * deriv_log_scale`.
    fn collect_joint_quantities_rescaled(
        &self,
        block_states: &[ParameterBlockState],
        deriv_log_scale: f64,
    ) -> Result<SurvivalJointQuantities, String> {
        let n = self.n;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let mut ll = Array1::<f64>::zeros(n);
        let mut d1_q = Array1::<f64>::zeros(n);
        let mut d2_q = Array1::<f64>::zeros(n);
        let mut d3_q = Array1::<f64>::zeros(n);
        let mut d1_q0 = Array1::<f64>::zeros(n);
        let mut d2_q0 = Array1::<f64>::zeros(n);
        let mut d3_q0 = Array1::<f64>::zeros(n);
        let mut d1_q1 = Array1::<f64>::zeros(n);
        let mut d2_q1 = Array1::<f64>::zeros(n);
        let mut d3_q1 = Array1::<f64>::zeros(n);
        let mut d1_qdot1 = Array1::<f64>::zeros(n);
        let mut d2_qdot1 = Array1::<f64>::zeros(n);
        let mut h_time_h0 = Array1::<f64>::zeros(n);
        let mut h_time_h1 = Array1::<f64>::zeros(n);
        let mut h_time_d = Array1::<f64>::zeros(n);
        let mut d_h_h0 = Array1::<f64>::zeros(n);
        let mut d_h_h1 = Array1::<f64>::zeros(n);
        let mut d_h_d = Array1::<f64>::zeros(n);

        // Write each row's 21 derivative scalars directly into the
        // preallocated output arrays in parallel. The previous path collected
        // a `Vec<Option<SurvivalRowDerivatives>>` (21 fields per row) and then
        // serially scattered into 21 `Array1`s — at large scale that is the
        // worst-case transient allocation among the family row builders.
        // Rows where `row_derivatives_rescaled` returns `Ok(None)` keep their
        // zero-initialized slots (matching the previous `continue` branch).
        /// Wrapper to send raw pointers across threads for disjoint per-row
        /// writes.  SAFETY: each parallel iteration writes a unique index `i`
        /// into a buffer of length `n`, and the pointers do not outlive the
        /// surrounding scope.
        #[derive(Clone, Copy)]
        struct SendPtr(*mut f64);
        // SAFETY: SendPtr is constructed from Array1::as_mut_ptr() on
        // length-n buffers; the rayon (0..n).into_par_iter() driver gives
        // each thread a unique i, so writes via SendPtr never alias.
        unsafe impl Send for SendPtr {}
        // SAFETY: same disjoint-index invariant as the Send impl above.
        unsafe impl Sync for SendPtr {}
        impl SendPtr {
            #[inline(always)]
            // SAFETY: caller passes `i < n` (the buffer length used to take
            // `self.0`); rayon's `(0..n).into_par_iter()` driver guarantees
            // exclusive ownership of `i` per thread, so the write is unaliased.
            unsafe fn write(self, i: usize, v: f64) {
                // SAFETY: `i < n` from the function contract; `self.0.add(i)`
                // is in-bounds and the disjoint-index invariant means no other
                // thread accesses this slot.
                unsafe { *self.0.add(i) = v };
            }
        }

        let p_ll = SendPtr(ll.as_mut_ptr());
        let p_d1_q = SendPtr(d1_q.as_mut_ptr());
        let p_d2_q = SendPtr(d2_q.as_mut_ptr());
        let p_d3_q = SendPtr(d3_q.as_mut_ptr());
        let p_d1_q0 = SendPtr(d1_q0.as_mut_ptr());
        let p_d2_q0 = SendPtr(d2_q0.as_mut_ptr());
        let p_d3_q0 = SendPtr(d3_q0.as_mut_ptr());
        let p_d1_q1 = SendPtr(d1_q1.as_mut_ptr());
        let p_d2_q1 = SendPtr(d2_q1.as_mut_ptr());
        let p_d3_q1 = SendPtr(d3_q1.as_mut_ptr());
        let p_d1_qdot1 = SendPtr(d1_qdot1.as_mut_ptr());
        let p_d2_qdot1 = SendPtr(d2_qdot1.as_mut_ptr());
        let p_h_time_h0 = SendPtr(h_time_h0.as_mut_ptr());
        let p_h_time_h1 = SendPtr(h_time_h1.as_mut_ptr());
        let p_h_time_d = SendPtr(h_time_d.as_mut_ptr());
        let p_d_h_h0 = SendPtr(d_h_h0.as_mut_ptr());
        let p_d_h_h1 = SendPtr(d_h_h1.as_mut_ptr());
        let p_d_h_d = SendPtr(d_h_d.as_mut_ptr());

        let dyn_ref = &dynamic;
        (0..n)
            .into_par_iter()
            .try_for_each(move |i| -> Result<(), String> {
                let state = self.row_predictor_state(
                    dyn_ref.h_entry[i],
                    dyn_ref.h_exit[i],
                    dyn_ref.hdot_exit[i],
                    dyn_ref.q_entry[i],
                    dyn_ref.q_exit[i],
                    dyn_ref.qdot_exit[i],
                );
                let Some(row) = self.row_derivatives_rescaled(i, state, deriv_log_scale)? else {
                    return Ok(());
                };
                // SAFETY: rayon `(0..n).into_par_iter()` yields each `i < n`
                // exactly once; pointers target distinct length-`n` `Array1`
                // buffers not read until the parallel loop completes.
                unsafe {
                    p_ll.write(i, row.ll);
                    p_d1_q.write(i, row.d1_q);
                    p_d2_q.write(i, row.d2_q);
                    p_d3_q.write(i, row.d3_q);
                    p_d1_q0.write(i, row.d1_q0);
                    p_d2_q0.write(i, row.d2_q0);
                    p_d3_q0.write(i, row.d3_q0);
                    p_d1_q1.write(i, row.d1_q1);
                    p_d2_q1.write(i, row.d2_q1);
                    p_d3_q1.write(i, row.d3_q1);
                    p_d1_qdot1.write(i, row.d1_qdot1);
                    p_d2_qdot1.write(i, row.d2_qdot1);
                    p_h_time_h0.write(i, row.h_time_h0);
                    p_h_time_h1.write(i, row.h_time_h1);
                    p_h_time_d.write(i, row.h_time_d);
                    p_d_h_h0.write(i, row.d_h_h0);
                    p_d_h_h1.write(i, row.d_h_h1);
                    p_d_h_d.write(i, row.d_h_d);
                }
                Ok(())
            })?;

        Ok(SurvivalJointQuantities {
            ll,
            d1_q,
            d2_q,
            d3_q,
            d1_q0,
            d2_q0,
            d3_q0,
            d1_q1,
            d2_q1,
            d3_q1,
            d1_qdot1,
            d2_qdot1,
            h_time_h0,
            h_time_h1,
            h_time_d,
            d_h_h0,
            d_h_h1,
            d_h_d,
            dq_t: dynamic.dq_t_exit,
            dq_ls: dynamic.dq_ls_exit,
            d2q_tls: dynamic.d2q_tls_exit,
            d2q_ls: dynamic.d2q_ls_exit,
            d3q_tls_ls: dynamic.d3q_tls_ls_exit,
            d3q_ls: dynamic.d3q_ls_exit,
            dq_t_entry: Some(dynamic.dq_t_entry),
            dq_ls_entry: Some(dynamic.dq_ls_entry),
            d2q_tls_entry: Some(dynamic.d2q_tls_entry),
            d2q_ls_entry: Some(dynamic.d2q_ls_entry),
            d3q_tls_ls_entry: Some(dynamic.d3q_tls_ls_entry),
            d3q_ls_entry: Some(dynamic.d3q_ls_entry),
            dqdot_t: dynamic.dqdot_t,
            dqdot_ls: dynamic.dqdot_ls,
            dqdot_td: dynamic.dqdot_td,
            dqdot_lsd: dynamic.dqdot_lsd,
            d2qdot_tt: dynamic.d2qdot_tt,
            d2qdot_tls: dynamic.d2qdot_tls,
            d2qdot_ttd: dynamic.d2qdot_ttd,
            d2qdot_tlsd: dynamic.d2qdot_tlsd,
            d2qdot_ls: dynamic.d2qdot_ls,
            d2qdot_lstd: dynamic.d2qdot_lstd,
            d2qdot_lslsd: dynamic.d2qdot_lslsd,
            d3qdot_tls_ls: dynamic.d3qdot_tls_ls,
            d3qdot_tls_lsd: dynamic.d3qdot_tls_lsd,
            d3qdot_td_ls_ls: dynamic.d3qdot_td_ls_ls,
            d3qdot_ls_ls_ls: dynamic.d3qdot_ls_ls_ls,
            d3qdot_ls_ls_lsd: dynamic.d3qdot_ls_ls_lsd,
        })
    }

    /// Per-row NLL gradient and curvature with respect to the three additive
    /// time-block offset channels `(o_E, o_X, o_D)` (entry / exit / derivative-
    /// at-exit). The baseline configuration enters the location-scale fit
    /// **only** through these three offsets, so contracting these residuals
    /// against `∂o/∂θ_baseline` gives the analytic θ-gradient of the
    /// unpenalized NLL at converged β (envelope theorem on the penalized
    /// objective; the penalty has no θ dependence).
    ///
    /// Algebra. With `ell_i = w_i[d(log f(u1) + log g) + (1-d) log S(u1) − log S(u0)]`
    /// and `u0 = h0 + q0`, `u1 = h1 + q1`, `g = d_raw + qdot1`:
    ///
    ///   ∂(−ell_i)/∂h0   = − w_i r(u0)
    ///   ∂(−ell_i)/∂h1   = − w_i [d ψ(u1) − (1−d) r(u1)]
    ///   ∂(−ell_i)/∂dRaw = − w_i d / g                                (event-row only)
    ///
    /// and the row Hessian is diagonal in (h0, h1, dRaw) because `u0`, `u1`,
    /// `g` are functionally independent (h0→u0, h1→u1, dRaw→g):
    ///
    ///   ∂²(−ell_i)/∂h0²   = − w_i r'(u0)
    ///   ∂²(−ell_i)/∂h1²   = − w_i [d ψ'(u1) − (1−d) r'(u1)]
    ///   ∂²(−ell_i)/∂dRaw² =   w_i d / g²
    ///
    /// The fields `grad_time_eta_*` / `h_time_*` produced by
    /// [`Self::row_derivatives`] are the corresponding log-likelihood (not
    /// NLL) partials; we negate `grad_time_eta_*` and the entry/exit second
    /// derivatives (`h_time_h0`, `h_time_h1`) to recover the NLL convention.
    /// The derivative-channel Hessian field `h_time_d` is already stored in
    /// NLL sign (the joint Hessian builder uses `+h_time_d` whereas it uses
    /// `−h_time_h0` / `−h_time_h1` for entry/exit; see the exact joint
    /// `safe_fast_xt_diag_x` assembly).
    pub(crate) fn offset_channel_geometry(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<(OffsetChannelResiduals, OffsetChannelCurvatures), String> {
        let n = self.n;
        // Defensive degraded-fit path: a custom-family `fit_custom_family` whose
        // outer ARC stalled into the "deterministic-replay" branch can land in
        // `blockwise_fit_from_parts` with the final inner refit's `block_states`
        // cleared (the path at `custom_family.rs` post-degraded-plan rebuilds
        // `BlockwiseFitResultParts` without re-populating per-block state).
        // Surfaces here as `block_states.len() == 0` when the value-closure
        // cache (`last_geometry` in `fit_survival_location_scale_terms`) is
        // unset and the fallback refit runs through this method.
        // `build_dynamic_geometry` would then fail with the cryptic
        // `SurvivalLocationScaleFamily expects 3 blocks, got 0` — which
        // propagates all the way to the Python wrapper.
        //
        // Returning zero residuals + zero curvatures here lets the outer
        // baseline BFGS treat this candidate as a stationary point (no
        // gradient contribution from these rows) instead of crashing the
        // whole fit. The next BFGS step gets `‖g‖ = 0`, so the optimizer
        // terminates at the current θ rather than wandering into NaN
        // territory. Production loss is at most a slightly suboptimal
        // baseline θ at this BMA parent set — far preferable to a hard
        // exception from PyPI.
        if block_states.is_empty() {
            log::warn!(
                "SurvivalLocationScaleFamily::offset_channel_geometry: \
                 block_states is empty (degraded fit, likely ARC \
                 deterministic-replay stall); returning zero residuals + \
                 curvatures (n={n})"
            );
            return Ok((
                OffsetChannelResiduals {
                    exit: Array1::<f64>::zeros(n),
                    entry: Array1::<f64>::zeros(n),
                    derivative: Array1::<f64>::zeros(n),
                    // Location-scale has no interval upper-bound channel.
                    right: Array1::<f64>::zeros(n),
                },
                OffsetChannelCurvatures {
                    rows: vec![[[0.0_f64; 3]; 3]; n],
                },
            ));
        }
        let dynamic = self.build_dynamic_geometry(block_states)?;

        let mut entry = Array1::<f64>::zeros(n);
        let mut exit = Array1::<f64>::zeros(n);
        let mut derivative = Array1::<f64>::zeros(n);
        let mut curvatures = vec![[[0.0_f64; 3]; 3]; n];

        let rows = (0..n)
            .into_par_iter()
            .map(
                |i| -> Result<(usize, f64, f64, f64, [[f64; 3]; 3]), String> {
                    let state = self.row_predictor_state(
                        dynamic.h_entry[i],
                        dynamic.h_exit[i],
                        dynamic.hdot_exit[i],
                        dynamic.q_entry[i],
                        dynamic.q_exit[i],
                        dynamic.qdot_exit[i],
                    );
                    let Some(row) = self.row_derivatives(i, state)? else {
                        return Ok((i, 0.0, 0.0, 0.0, [[0.0; 3]; 3]));
                    };
                    // Convert ℓ-partials (h_time_*, grad_time_eta_*) to NLL partials.
                    // grad_time_eta_* hold ∂ℓ/∂{h0,h1,d_raw}; ∂NLL/∂o = −∂ℓ/∂h.
                    let r_entry = -row.grad_time_eta_h0;
                    let r_exit = -row.grad_time_eta_h1;
                    let r_deriv = -row.grad_time_eta_d;
                    // NLL Hessian on (h0,h1,d_raw): diagonal because the row likelihood
                    // factors through (u0, u1, g) which are functionally independent
                    // in (h0, h1, d_raw). Signs follow the exact-joint Hessian assembly
                    // which uses (−h_time_h0, −h_time_h1, +h_time_d) for the NLL block.
                    let mut curv = [[0.0_f64; 3]; 3];
                    curv[0][0] = -row.h_time_h0;
                    curv[1][1] = -row.h_time_h1;
                    curv[2][2] = row.h_time_d;
                    Ok((i, r_entry, r_exit, r_deriv, curv))
                },
            )
            .collect::<Result<Vec<_>, String>>()?;

        for (i, r_entry, r_exit, r_deriv, curv) in rows {
            entry[i] = r_entry;
            exit[i] = r_exit;
            derivative[i] = r_deriv;
            curvatures[i] = curv;
        }

        Ok((
            OffsetChannelResiduals {
                exit,
                entry,
                derivative,
                // Location-scale has no interval upper-bound channel.
                right: Array1::<f64>::zeros(n),
            },
            OffsetChannelCurvatures { rows: curvatures },
        ))
    }

    /// Exact data-fit gradient `Σ_i ∂ℓ_i/∂θ_link` of the unpenalized
    /// log-likelihood with respect to the inverse-link parameters θ_link
    /// (SAS `(ε, log δ)`, BetaLogistic `(ε, log δ)`, or Mixture `ρ`), holding
    /// the fitted β and λ fixed.
    ///
    /// The per-row log-likelihood is
    ///   ℓ_i = w_i·( event_mix(d_i, logφ(u1_i) + log g_i, log S(u1_i)) − log S(u0_i) ),
    /// where `u0 = h0 + q0` and `u1 = h1 + q1` are the standardized residuals
    /// the inverse link evaluates (entry/exit), `log g` is the time-derivative
    /// Jacobian (link-independent), and the link enters ONLY through the scalar
    /// `log S(u) = log(1 − μ(u;θ))` and `log φ(u) = log d1(u;θ)` terms. Hence
    ///   ∂(log S)/∂θ = −(∂μ/∂θ)/S,   ∂(log φ)/∂θ = (∂d1/∂θ)/d1,
    /// with `S = 1 − μ`, `μ = jet.mu`, `d1 = jet.d1`, and the parameter partials
    /// `∂μ/∂θ`, `∂d1/∂θ` supplied analytically by
    /// [`InverseLinkKernel::param_partials`]. The higher-order ratio/pdf
    /// derivatives (r, dr, …, fppp) carry the inner-Newton curvature only and do
    /// NOT appear in the scalar ℓ, so the data-fit θ-gradient needs only the
    /// `(μ, d1)` jet components and their param partials — all exact.
    ///
    /// At the converged β̂ the envelope theorem makes this the exact θ-gradient
    /// of the profile penalized NLL `−ℓ + ½βᵀSβ` (β profiled out; the penalty
    /// has no θ_link dependence). Returns a length-`n_link_params` vector
    /// (`∂(−ℓ)/∂θ` so it matches the profile-cost sign), or `None` when the
    /// inverse link carries no free parameters.
    pub(crate) fn link_param_data_fit_gradient(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array1<f64>>, String> {
        use crate::solver::mixture_link::{InverseLinkKernel, LinkParamPartials};
        let n = self.n;
        if block_states.is_empty() {
            return Ok(None);
        }
        // ∂(log S)/∂θ and ∂(log φ)/∂θ contributions per row are accumulated
        // into a θ-length vector. Probe the parameter count from the link's
        // partials at a finite argument; `None` ⇒ no free link parameters.
        let probe = self
            .inverse_link
            .param_partials(0.0)
            .map_err(|e| format!("inverse-link param partials probe failed: {e}"))?;
        let n_theta = match &probe {
            None => return Ok(None),
            Some(LinkParamPartials::Sas(_)) => 2,
            Some(LinkParamPartials::Mixture(m)) => m.djet_drho.len(),
        };
        if n_theta == 0 {
            return Ok(None);
        }
        let dynamic = self.build_dynamic_geometry(block_states)?;
        // ∂(log S)/∂θ = −(∂μ/∂θ)/S at argument u (S = 1 − μ);
        // ∂(log φ)/∂θ = (∂d1/∂θ)/d1 at argument u.
        let dlog_survival_dtheta = |u: f64| -> Result<Vec<f64>, String> {
            let partials = self
                .inverse_link
                .param_partials(u)
                .map_err(|e| format!("inverse-link survival param partials failed: {e}"))?
                .ok_or_else(|| "inverse-link reported no param partials".to_string())?;
            let jet = self
                .inverse_link
                .jet(u)
                .map_err(|e| format!("inverse-link jet failed at u={u}: {e}"))?;
            let s = (1.0 - jet.mu).clamp(f64::MIN_POSITIVE, 1.0);
            let map = |dmu: f64| -dmu / s;
            Ok(match partials {
                LinkParamPartials::Sas(p) => {
                    vec![map(p.djet_depsilon.mu), map(p.djet_dlog_delta.mu)]
                }
                LinkParamPartials::Mixture(p) => {
                    p.djet_drho.iter().map(|j| map(j.mu)).collect()
                }
            })
        };
        let dlog_pdf_dtheta = |u: f64| -> Result<Vec<f64>, String> {
            let partials = self
                .inverse_link
                .param_partials(u)
                .map_err(|e| format!("inverse-link pdf param partials failed: {e}"))?
                .ok_or_else(|| "inverse-link reported no param partials".to_string())?;
            let jet = self
                .inverse_link
                .jet(u)
                .map_err(|e| format!("inverse-link jet failed at u={u}: {e}"))?;
            let f = jet.d1;
            if !(f.is_finite() && f > 0.0) {
                return Err(format!(
                    "inverse-link pdf (d1) must be finite positive for θ-gradient, got {f} at u={u}"
                ));
            }
            let map = |dd1: f64| dd1 / f;
            Ok(match partials {
                LinkParamPartials::Sas(p) => {
                    vec![map(p.djet_depsilon.d1), map(p.djet_dlog_delta.d1)]
                }
                LinkParamPartials::Mixture(p) => {
                    p.djet_drho.iter().map(|j| map(j.d1)).collect()
                }
            })
        };
        // Accumulate ∂(−ℓ)/∂θ = −Σ_i w_i·( event_mix(d, ∂logφ(u1), ∂logS(u1))
        //                                    − ∂logS(u0) ).
        let mut grad = Array1::<f64>::zeros(n_theta);
        for i in 0..n {
            let w = self.w[i];
            if w <= 0.0 {
                continue;
            }
            let d = self.validated_event_target(i)?;
            let u0 = dynamic.h_entry[i] + dynamic.q_entry[i];
            let u1 = dynamic.h_exit[i] + dynamic.q_exit[i];
            let dls_u0 = dlog_survival_dtheta(u0)?;
            // Entry channel always contributes (left-truncation term −log S(u0)).
            for k in 0..n_theta {
                grad[k] += w * dls_u0[k];
            }
            if d <= 0.0 {
                // Censored: +log S(u1).
                let dls_u1 = dlog_survival_dtheta(u1)?;
                for k in 0..n_theta {
                    grad[k] -= w * dls_u1[k];
                }
            } else if d >= 1.0 {
                // Event: +log φ(u1) (log g is link-independent).
                let dlp_u1 = dlog_pdf_dtheta(u1)?;
                for k in 0..n_theta {
                    grad[k] -= w * dlp_u1[k];
                }
            } else {
                // Fractional event weight: mix both branches.
                let dls_u1 = dlog_survival_dtheta(u1)?;
                let dlp_u1 = dlog_pdf_dtheta(u1)?;
                for k in 0..n_theta {
                    grad[k] -= w * (d * dlp_u1[k] + (1.0 - d) * dls_u1[k]);
                }
            }
        }
        Ok(Some(grad))
    }

    fn exact_newton_joint_psi_direction(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<SurvivalJointPsiDirection>, String> {
        if block_states.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "SurvivalLocationScaleFamily joint psi direction expects {} blocks and derivative lists, got {} and {}",
                self.expected_blocks(),
                block_states.len(),
                derivative_blocks.len()
            ) }.into());
        }

        let n = self.n;
        let pt = self.x_threshold.ncols();
        let pls = self.x_log_sigma.ncols();
        let beta_t = &block_states[Self::BLOCK_THRESHOLD].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;
        let t_time_varying = self.x_threshold_entry.is_some();
        let ls_time_varying = self.x_log_sigma_entry.is_some();

        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for deriv in block_derivs {
                if global == psi_index {
                    let mut x_t_exit_psi = None;
                    let mut x_t_entry_psi = None;
                    let mut x_ls_exit_psi = None;
                    let mut x_ls_entry_psi = None;
                    let mut x_t_exit_action = None;
                    let mut x_t_entry_action = None;
                    let mut x_ls_exit_action = None;
                    let mut x_ls_entry_action = None;
                    let mut z_t_exit_psi = Array1::<f64>::zeros(n);
                    let mut z_t_entry_psi = Array1::<f64>::zeros(n);
                    let mut z_ls_exit_psi = Array1::<f64>::zeros(n);
                    let mut z_ls_entry_psi = Array1::<f64>::zeros(n);
                    match block_idx {
                        Self::BLOCK_THRESHOLD => {
                            let total_rows = if t_time_varying { 3 * n } else { n };
                            match resolve_custom_family_x_psi_map(
                                deriv,
                                total_rows,
                                pt,
                                0..total_rows,
                                "SurvivalLocationScaleFamily threshold",
                                &self.policy,
                            )? {
                                PsiDesignMap::First { action } => {
                                    if t_time_varying {
                                        let exit_action = action.slice_rows(0..n)?;
                                        let entry_action = action.slice_rows(n..2 * n)?;
                                        z_t_exit_psi = exit_action.forward_mul(beta_t.view());
                                        z_t_entry_psi = entry_action.forward_mul(beta_t.view());
                                        x_t_exit_action = Some(exit_action);
                                        x_t_entry_action = Some(entry_action);
                                    } else {
                                        z_t_exit_psi = action.forward_mul(beta_t.view());
                                        z_t_entry_psi = z_t_exit_psi.clone();
                                        x_t_exit_action = Some(action.clone());
                                        x_t_entry_action = Some(action);
                                    }
                                }
                                PsiDesignMap::Dense { matrix } => {
                                    let (exit, entry) = split_survival_psi_design(
                                        &matrix,
                                        n,
                                        t_time_varying,
                                        "SurvivalLocationScaleFamily threshold",
                                    )?;
                                    z_t_exit_psi = fast_av(&exit, beta_t);
                                    z_t_entry_psi = fast_av(&entry, beta_t);
                                    x_t_exit_psi = Some(exit);
                                    x_t_entry_psi = Some(entry);
                                }
                                PsiDesignMap::Zero { .. } => {}
                                PsiDesignMap::Second { .. } => {
                                    return Err(SurvivalLocationScaleError::DimensionMismatch { reason: "SurvivalLocationScaleFamily threshold: unexpected Second variant from _psi_map"
                                            .to_string(), }.into());
                                }
                            }
                        }
                        Self::BLOCK_LOG_SIGMA => {
                            let total_rows = if ls_time_varying { 3 * n } else { n };
                            match resolve_custom_family_x_psi_map(
                                deriv,
                                total_rows,
                                pls,
                                0..total_rows,
                                "SurvivalLocationScaleFamily log-sigma",
                                &self.policy,
                            )? {
                                PsiDesignMap::First { action } => {
                                    if ls_time_varying {
                                        let exit_action = action.slice_rows(0..n)?;
                                        let entry_action = action.slice_rows(n..2 * n)?;
                                        z_ls_exit_psi = exit_action.forward_mul(beta_ls.view());
                                        z_ls_entry_psi = entry_action.forward_mul(beta_ls.view());
                                        x_ls_exit_action = Some(exit_action);
                                        x_ls_entry_action = Some(entry_action);
                                    } else {
                                        z_ls_exit_psi = action.forward_mul(beta_ls.view());
                                        z_ls_entry_psi = z_ls_exit_psi.clone();
                                        x_ls_exit_action = Some(action.clone());
                                        x_ls_entry_action = Some(action);
                                    }
                                }
                                PsiDesignMap::Dense { matrix } => {
                                    let (exit, entry) = split_survival_psi_design(
                                        &matrix,
                                        n,
                                        ls_time_varying,
                                        "SurvivalLocationScaleFamily log-sigma",
                                    )?;
                                    z_ls_exit_psi = fast_av(&exit, beta_ls);
                                    z_ls_entry_psi = fast_av(&entry, beta_ls);
                                    x_ls_exit_psi = Some(exit);
                                    x_ls_entry_psi = Some(entry);
                                }
                                PsiDesignMap::Zero { .. } => {}
                                PsiDesignMap::Second { .. } => {
                                    return Err(SurvivalLocationScaleError::DimensionMismatch { reason: "SurvivalLocationScaleFamily log-sigma: unexpected Second variant from _psi_map"
                                            .to_string(), }.into());
                                }
                            }
                        }
                        _ => return Ok(None),
                    }
                    return Ok(Some(SurvivalJointPsiDirection {
                        x_t_exit_psi,
                        x_t_entry_psi,
                        x_ls_exit_psi,
                        x_ls_entry_psi,
                        z_t_exit_psi,
                        z_t_entry_psi,
                        z_ls_exit_psi,
                        z_ls_entry_psi,
                        x_t_exit_action,
                        x_t_entry_action,
                        x_ls_exit_action,
                        x_ls_entry_action,
                    }));
                }
                global += 1;
            }
        }
        Ok(None)
    }

    /// Hazard-like survival ratio and its first derivative.
    ///
    /// Let `F` be the CDF, `f = F'` the PDF, and `S = 1 - F` the survival
    /// function so `S' = -f`.
    ///
    /// Define `r = f / S`. By quotient rule:
    /// `r' = (f' S - f S') / S^2`.
    /// Since `S' = -f`, this becomes:
    /// `r' = f'/S + f^2/S^2 = f'/S + r^2`.
    ///
    /// Sign note: the `f'/S` term is strictly additive. A minus here is wrong.
    fn survival_ratio_first_derivative(f: f64, fp: f64, s: f64) -> (f64, f64) {
        let r = f / s;
        let dr = (r * r) + fp / s;
        (r, dr)
    }

    /// Second derivative of the survival ratio `r = f/S`.
    ///
    /// Starting from `r' = f'/S + r^2`:
    /// `r'' = d/du[f'/S] + 2 r r'`.
    /// With `S' = -f`, we get:
    /// `d/du[f'/S] = f''/S + f' f / S^2`.
    /// Therefore:
    /// `r'' = 2 r r' + f''/S + f' f / S^2`.
    ///
    /// Equivalent expanded form:
    /// `r'' = f''/S + 3 f f' / S^2 + 2 f^3 / S^3`.
    fn survival_ratiosecond_derivative(r: f64, dr: f64, f: f64, fp: f64, fpp: f64, s: f64) -> f64 {
        (2.0 * r * dr) + (fpp / s + fp * f / (s * s))
    }

    /// Third derivative of the survival ratio `r = f/S`.
    ///
    /// Starting from `r'' = 2 r r' + f''/S + f' f / S²`:
    ///
    /// ```text
    /// r''' = d/du[2 r r'] + d/du[f''/S + f'f/S²]
    ///      = 2(r')² + 2 r r'' + f'''/S + f''f/S² + f'²/S² + 2f'f²/S³ + f''f/S²
    ///      = 2(r')² + 2 r r'' + f'''/S + 2f''f/S² + (f')²/S² + 2f(f')²/S³ ... wait
    /// ```
    ///
    /// More carefully: let A = f''/S, B = f'f/S². Then r'' = 2rr' + A + B.
    ///
    /// ```text
    /// d/du[A] = f'''/S + f''f/S²   (using S' = -f)
    /// d/du[B] = (f''f + f'²)/S² + 2f'f²/S³
    /// ```
    ///
    /// So:
    /// ```text
    /// r''' = 2(r')² + 2rr'' + f'''/S + 2f''f/S² + (f')²/S² + 2f'f²/S³
    /// ```
    ///
    /// This is needed for d⁴ℓ/dq0⁴ (the entry-side 4th likelihood derivative)
    /// and d⁴ℓ/dq1⁴ (the exit-side 4th likelihood derivative), which enter the
    /// outer REML Hessian's Q[v_k, v_l] term via the Arbogast formula.
    fn survival_ratio_third_derivative(
        r: f64,
        dr: f64,
        ddr: f64,
        f: f64,
        fp: f64,
        fpp: f64,
        fppp: f64,
        s: f64,
    ) -> f64 {
        let s2 = s * s;
        let s3 = s2 * s;
        2.0 * dr * dr
            + 2.0 * r * ddr
            + fppp / s
            + 2.0 * fpp * f / s2
            + fp * fp / s2
            + 2.0 * fp * f * f / s3
    }

    /// Like [`Self::exact_log_pdf_derivatives_rescaled`] but with a log-scale shift
    /// on the derivative magnitudes.  For CLogLog the `exp(eta)` terms in
    /// the derivatives become `exp(eta - deriv_log_scale)`, and the constant
    /// term in `d/deta log f = 1 - exp(eta)` is scaled by the same factor.
    /// The function value is returned unshifted.
    fn exact_log_pdf_derivatives_rescaled(
        inverse_link: &InverseLink,
        eta: f64,
        deriv_log_scale: f64,
    ) -> Result<(f64, f64, f64, f64, f64), String> {
        match inverse_link {
            InverseLink::Standard(StandardLink::Probit) => Ok((
                -0.5 * eta * eta - 0.5 * (2.0 * std::f64::consts::PI).ln(),
                -eta,
                -1.0,
                0.0,
                0.0,
            )),
            InverseLink::Standard(StandardLink::Logit) => {
                let mu = crate::solver::mixture_link::component_inverse_link_jet(
                    crate::types::LinkComponent::Logit,
                    eta,
                )
                .mu;
                let w = mu * (1.0 - mu);
                Ok((
                    -softplus(eta) - softplus(-eta),
                    1.0 - 2.0 * mu,
                    -2.0 * w,
                    -2.0 * w * (1.0 - 2.0 * mu),
                    -2.0 * w * (1.0 - 6.0 * w),
                ))
            }
            InverseLink::Standard(StandardLink::CLogLog) => {
                let t_val = eta.exp(); // for function value (may be Inf)
                let t_deriv = (eta - deriv_log_scale).exp(); // for derivatives
                let deriv_scale = (-deriv_log_scale).exp();
                Ok((
                    eta - t_val,
                    deriv_scale - t_deriv,
                    -t_deriv,
                    -t_deriv,
                    -t_deriv,
                ))
            }
            InverseLink::Standard(StandardLink::Identity) => Ok((0.0, 0.0, 0.0, 0.0, 0.0)),
            _ => {
                let jet = inverse_link_jet_for_inverse_link(inverse_link, eta)
                    .map_err(|e| format!("inverse link evaluation failed at eta={eta}: {e}"))?;
                let f = jet.d1;
                if !(f.is_finite() && f > 0.0) {
                    return Err(SurvivalLocationScaleError::NumericalFailure {
                        reason: format!(
                            "inverse-link pdf must be finite and positive, got {f} at eta={eta}"
                        ),
                    }
                    .into());
                }
                let fp = jet.d2;
                let fpp = jet.d3;
                let fppp = inverse_link_pdfthird_derivative_for_inverse_link(inverse_link, eta)
                    .map_err(|e| {
                        format!("inverse link third-derivative evaluation failed at eta={eta}: {e}")
                    })?;
                let fpppp = inverse_link_pdffourth_derivative(inverse_link, eta)?;
                let d1 = fp / f;
                let d2 = fpp / f - d1 * d1;
                let d3 = fppp / f - 3.0 * fp * fpp / (f * f) + 2.0 * fp.powi(3) / f.powi(3);
                let d4 = fpppp / f - 4.0 * fp * fppp / f.powi(2) - 3.0 * fpp * fpp / f.powi(2)
                    + 12.0 * fp.powi(2) * fpp / f.powi(3)
                    - 6.0 * fp.powi(4) / f.powi(4);
                Ok((f.ln(), d1, d2, d3, d4))
            }
        }
    }

    /// Like [`Self::exact_survival_neglog_derivatives_fourth_rescaled`] but with a
    /// log-scale shift applied to the **derivative** magnitudes (not the
    /// function value).  For CLogLog the derivatives are `exp(eta)`, so
    /// shifting gives `exp(eta - deriv_log_scale)` — always finite when
    /// the shift equals the maximum `eta` across rows.  The function
    /// value (`-exp(eta)` = `log S`) is returned unshifted.
    fn exact_survival_neglog_derivatives_fourth_rescaled(
        inverse_link: &InverseLink,
        eta: f64,
        deriv_log_scale: f64,
    ) -> Result<(f64, f64, f64, f64, f64), String> {
        match inverse_link {
            InverseLink::Standard(StandardLink::Probit) => {
                let (log_s, r, dr, ddr, dddr) = probit_log_survival_and_ratio_derivatives(eta);
                Ok((log_s, r, dr, ddr, dddr))
            }
            InverseLink::Standard(StandardLink::Logit) => {
                let mu = crate::solver::mixture_link::component_inverse_link_jet(
                    crate::types::LinkComponent::Logit,
                    eta,
                )
                .mu;
                let w = mu * (1.0 - mu);
                Ok((
                    -softplus(eta),
                    mu,
                    w,
                    w * (1.0 - 2.0 * mu),
                    w * (1.0 - 6.0 * w),
                ))
            }
            InverseLink::Standard(StandardLink::CLogLog) => {
                let t_val = eta.exp(); // for the function value (may be Inf)
                let t_deriv = (eta - deriv_log_scale).exp(); // for derivatives (finite when shifted)
                Ok((-t_val, t_deriv, t_deriv, t_deriv, t_deriv))
            }
            InverseLink::Standard(StandardLink::Identity) => {
                let s = 1.0 - eta;
                if !(s.is_finite() && s > 0.0) {
                    return Err(SurvivalLocationScaleError::NumericalFailure {
                        reason: format!("identity-link survival invalid at eta={eta}: S={s}"),
                    }
                    .into());
                }
                let inv = s.recip();
                Ok((s.ln(), inv, inv * inv, 2.0 * inv.powi(3), 6.0 * inv.powi(4)))
            }
            _ => {
                let jet = inverse_link_jet_for_inverse_link(inverse_link, eta)
                    .map_err(|e| format!("inverse link evaluation failed at eta={eta}: {e}"))?;
                let s = inverse_link_survival_probvalue(inverse_link, eta);
                if !(s.is_finite() && s > 0.0 && s <= 1.0) {
                    return Err(SurvivalLocationScaleError::NumericalFailure { reason: format!(
                        "inverse-link survival probability must lie in (0,1], got {s} at eta={eta}"
                    ) }.into());
                }
                let fppp = inverse_link_pdfthird_derivative_for_inverse_link(inverse_link, eta)
                    .map_err(|e| {
                        format!("inverse link third-derivative evaluation failed at eta={eta}: {e}")
                    })?;
                let (r, dr) = Self::survival_ratio_first_derivative(jet.d1, jet.d2, s);
                let ddr = Self::survival_ratiosecond_derivative(r, dr, jet.d1, jet.d2, jet.d3, s);
                let dddr = Self::survival_ratio_third_derivative(
                    r, dr, ddr, jet.d1, jet.d2, jet.d3, fppp, s,
                );
                Ok((s.ln(), r, dr, ddr, dddr))
            }
        }
    }

    /// Fused CLogLog evaluator for the exit-row pair: returns the
    /// `(log_s, r, dr, ddr, dddr)` survival tuple and the
    /// `(logphi, d1, d2, d3, d4)` log-pdf tuple while computing the two
    /// expensive `exp` calls once.  This duplicates the CLogLog branches of
    /// `exact_survival_neglog_derivatives_fourth_rescaled` and
    /// `exact_log_pdf_derivatives_rescaled` to share their work.
    #[inline]
    fn clglog_exit_pair(
        u1: f64,
        deriv_log_scale: f64,
    ) -> ((f64, f64, f64, f64, f64), (f64, f64, f64, f64, f64)) {
        let t_val = u1.exp();
        let t_deriv = (u1 - deriv_log_scale).exp();
        let deriv_scale = (-deriv_log_scale).exp();
        let surv = (-t_val, t_deriv, t_deriv, t_deriv, t_deriv);
        let logpdf = (
            u1 - t_val,
            deriv_scale - t_deriv,
            -t_deriv,
            -t_deriv,
            -t_deriv,
        );
        (surv, logpdf)
    }

    /// Exact `log(x)` value and first four derivatives on the positive domain.
    fn logwith_derivatives_positive(x: f64) -> (f64, f64, f64, f64, f64) {
        assert!(
            x.is_finite() && x > 0.0,
            "log derivative kernel requires finite positive x: x={x}"
        );
        let inv = 1.0 / x;
        (
            x.ln(),
            inv,
            -inv * inv,
            2.0 * inv * inv * inv,
            -6.0 * inv * inv * inv * inv,
        )
    }

    /// Build the row predictor state with possibly distinct entry/exit
    /// evaluations of threshold and sigma.
    ///
    /// For time-invariant blocks, the caller passes the same value for both
    /// entry and exit.
    fn row_predictor_state(
        &self,
        h0: f64,
        h1: f64,
        d_raw: f64,
        q0: f64,
        q1: f64,
        qdot1: f64,
    ) -> SurvivalPredictorState {
        let g_diff = compensated_difference(d_raw, -qdot1);
        SurvivalPredictorState {
            h0,
            h1,
            g: g_diff.value,
            q0,
            q1,
            g_roundoff_slack: g_diff.roundoff_slack,
            g_operand_scale: g_diff.operand_scale,
        }
    }

    #[inline]
    fn validated_event_target(&self, row: usize) -> Result<f64, String> {
        let d = self.y[row];
        if !(d.is_finite() && (0.0..=1.0).contains(&d)) {
            return Err(SurvivalLocationScaleError::ConstraintViolation {
                reason: format!(
                    "survival location-scale event target must lie in [0,1] at row {row}, got {d}"
                ),
            }
            .into());
        }
        Ok(d)
    }

    fn exact_row_kernel(
        &self,
        row: usize,
        state: SurvivalPredictorState,
    ) -> Result<Option<SurvivalExactRowKernel>, String> {
        self.exact_row_kernel_rescaled(row, state, 0.0)
    }

    /// Like [`Self::exact_row_kernel`] but with a log-scale shift on the
    /// derivative magnitudes, propagated to the survival/pdf derivative
    /// functions.  Used by the logdet Hessian path to avoid overflow.
    fn exact_row_kernel_rescaled(
        &self,
        row: usize,
        state: SurvivalPredictorState,
        deriv_log_scale: f64,
    ) -> Result<Option<SurvivalExactRowKernel>, String> {
        let w = self.w[row];
        if w <= 0.0 {
            return Ok(None);
        }
        let d = self.validated_event_target(row)?;
        let u0 = state.h0 + state.q0;
        let u1 = state.h1 + state.q1;

        let (log_s0, r0, dr0, ddr0, dddr0) =
            Self::exact_survival_neglog_derivatives_fourth_rescaled(
                &self.inverse_link,
                u0,
                deriv_log_scale,
            )
            .map_err(|e| {
                format!("inverse-link survival evaluation failed at row {row} entry: {e}")
            })?;

        // Fast path: for CLogLog the survival and log-pdf evaluators each
        // compute `exp(u1)` and `exp(u1 - deriv_log_scale)`.  Share that work
        // when both are called back-to-back on the exit row.
        let ((log_s1, r1, dr1, ddr1, dddr1), (logphi1, dlogphi1, d2logphi1, d3logphi1, d4logphi1)) =
            if matches!(
                &self.inverse_link,
                InverseLink::Standard(StandardLink::CLogLog)
            ) {
                Self::clglog_exit_pair(u1, deriv_log_scale)
            } else {
                let surv = Self::exact_survival_neglog_derivatives_fourth_rescaled(
                    &self.inverse_link,
                    u1,
                    deriv_log_scale,
                )
                .map_err(|e| {
                    format!("inverse-link survival evaluation failed at row {row} exit: {e}")
                })?;

                let pdf = Self::exact_log_pdf_derivatives_rescaled(
                    &self.inverse_link,
                    u1,
                    deriv_log_scale,
                )
                .map_err(|e| {
                    format!("inverse-link log-pdf evaluation failed at row {row} exit: {e}")
                })?;
                (surv, pdf)
            };

        // Row degeneracy guard: when any hazard/pdf derivative is non-finite
        // (e.g. CLogLog with u > ~709 where exp(u) overflows), the row's
        // survival probability has underflowed to 0 and the derivatives
        // cannot be represented in f64.  Exclude the row — same principle
        // as the w <= 0 early-return above.
        if !(r0.is_finite()
            && dr0.is_finite()
            && ddr0.is_finite()
            && dddr0.is_finite()
            && r1.is_finite()
            && dr1.is_finite()
            && ddr1.is_finite()
            && dddr1.is_finite()
            && dlogphi1.is_finite()
            && d2logphi1.is_finite()
            && d3logphi1.is_finite()
            && d4logphi1.is_finite())
        {
            log::debug!(
                "skipping row {row}: survival derivatives non-finite \
                 (u0={u0:.2e}, u1={u1:.2e})"
            );
            return Ok(None);
        }

        let guard = self.time_derivative_lower_bound();
        let mut g = state.g;
        // Layer 4: NaN is a hard error (genuinely bad data or upstream logic
        // bug).  ±inf is clamped to finite extremes so downstream log(g) is
        // well-defined; the monotonicity guard will then floor g if needed.
        if g.is_nan() {
            return Err(SurvivalLocationScaleError::NumericalFailure { reason: format!(
                "survival location-scale time derivative is non-finite at row {row}: d_eta/dt={g}"
            ) }.into());
        }
        if g == f64::INFINITY {
            g = f64::MAX;
        } else if g == f64::NEG_INFINITY {
            g = f64::MIN;
        }
        // Adaptive roundoff slack for the monotonicity guard.
        //
        // `g` is now formed with a compensated subtraction, so the low-part
        // residual from that subtraction is the primary estimate of how much
        // rounding error the d_eta/dt reconstruction may have accumulated.
        // The older state-scale heuristic remains as a floor for moderate
        // inputs.
        let legacy_slack = MONOTONICITY_GUARD_SLACK_REL
            * (1.0
                + state
                    .h0
                    .abs()
                    .max(state.h1.abs())
                    .max(state.q0.abs())
                    .max(state.q1.abs()));
        let roundoff_slack = state.g_roundoff_slack.max(legacy_slack);
        if g < guard && g >= guard - roundoff_slack {
            g = guard;
        }
        // `d_raw` is structurally constrained, but the full event Jacobian is
        // `g = d_raw + qdot`. The threshold/log-sigma contribution can nudge an
        // otherwise valid monotone state below the numeric guard while still
        // remaining strictly positive. The row kernel only needs `log(g)` on the
        // positive domain, so clamp positive near-boundary values to the guard
        // and reserve hard failure for true non-monotone states.
        if g > 0.0 && g < guard {
            g = guard;
        }
        if g <= 0.0 {
            return Err(SurvivalLocationScaleError::ConstraintViolation {
                reason: format!(
                    "survival location-scale monotonicity violated at row {row}: \
                 d_eta/dt={g:.3e} <= 0 (lower_bound={guard:.3e}) \
                 (operand_scale={:.3e}, roundoff_slack={roundoff_slack:.3e})",
                    state.g_operand_scale
                ),
            }
            .into());
        }
        let (log_g, d_log_g, d2_log_g, d3_log_g, d4_log_g) = Self::logwith_derivatives_positive(g);

        Ok(Some(SurvivalExactRowKernel {
            w,
            d,
            log_s0,
            r0,
            dr0,
            ddr0,
            dddr0,
            log_s1,
            r1,
            dr1,
            ddr1,
            dddr1,
            logphi1,
            dlogphi1,
            d2logphi1,
            d3logphi1,
            d4logphi1,
            log_g,
            d_log_g,
            d2_log_g,
            d3_log_g,
            d4_log_g,
        }))
    }

    fn row_derivatives(
        &self,
        row: usize,
        state: SurvivalPredictorState,
    ) -> Result<Option<SurvivalRowDerivatives>, String> {
        self.row_derivatives_rescaled(row, state, 0.0)
    }

    fn row_derivatives_rescaled(
        &self,
        row: usize,
        state: SurvivalPredictorState,
        deriv_log_scale: f64,
    ) -> Result<Option<SurvivalRowDerivatives>, String> {
        let Some(kernel) = self.exact_row_kernel_rescaled(row, state, deriv_log_scale)? else {
            return Ok(None);
        };
        let tower = kernel.nll_index_tower();
        let d1_q0 = -tower.g[0];
        let d2_q0 = -tower.h[0][0];
        let d3_q0 = -tower.t3[0][0][0];
        let d1_q1 = -tower.g[1];
        let d2_q1 = -tower.h[1][1];
        let d3_q1 = -tower.t3[1][1][1];
        let d1_qdot1 = -tower.g[2];
        let d2_qdot1 = -tower.h[2][2];
        let d1_q = d1_q0 + d1_q1;
        let d2_q = d2_q0 + d2_q1;
        let d3_q = d3_q0 + d3_q1;
        Ok(Some(SurvivalRowDerivatives {
            ll: kernel.log_likelihood(),
            d1_q,
            d2_q,
            d3_q,
            d1_q0,
            d2_q0,
            d3_q0,
            d1_q1,
            d2_q1,
            d3_q1,
            d1_qdot1,
            d2_qdot1,
            grad_time_eta_h0: d1_q0,
            grad_time_eta_h1: d1_q1,
            grad_time_eta_d: d1_qdot1,
            h_time_h0: d2_q0,
            h_time_h1: d2_q1,
            h_time_d: tower.h[2][2],
            d_h_h0: d3_q0,
            d_h_h1: d3_q1,
            d_h_d: tower.t3[2][2][2],
        }))
    }
}


/// Scalar chain-rule derivatives of
/// q(eta_t, eta_ls) = -eta_t * exp(-eta_ls).
///
/// Returns (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) — the full set of
/// partials up to third order needed by both the survival and GAMLSS engines.
#[inline]
pub(crate) fn q_chain_derivs_scalar(eta_t: f64, eta_ls: f64) -> (f64, f64, f64, f64, f64, f64) {
    let inv_sigma = exp_sigma_inverse_from_eta_scalar(eta_ls);
    let q = -safe_product(eta_t, inv_sigma);
    (-inv_sigma, -q, inv_sigma, q, -inv_sigma, -q)
}


fn validate_cov_block(
    name: &str,
    n: usize,
    b: &ParameterBlockInput,
) -> Result<(), SurvivalLocationScaleError> {
    if b.design.nrows() != n {
        crate::bail_dim_sls!(
            "{name} design row mismatch: got {}, expected {n}",
            b.design.nrows()
        );
    }
    if b.offset.len() != n {
        crate::bail_dim_sls!(
            "{name} offset length mismatch: got {}, expected {n}",
            b.offset.len()
        );
    }
    let p = b.design.ncols();
    if let Some(beta0) = &b.initial_beta
        && beta0.len() != p
    {
        crate::bail_dim_sls!(
            "{name} initial_beta length mismatch: got {}, expected {p}",
            beta0.len()
        );
    }
    let k = b.penalties.len();
    if let Some(rho0) = &b.initial_log_lambdas
        && rho0.len() != k
    {
        crate::bail_dim_sls!(
            "{name} initial_log_lambdas length mismatch: got {}, expected {k}",
            rho0.len()
        );
    }
    for (idx, s) in b.penalties.iter().enumerate() {
        match s {
            crate::solver::estimate::PenaltySpec::Block {
                local, col_range, ..
            } => {
                if col_range.end > p
                    || local.nrows() != col_range.len()
                    || local.ncols() != col_range.len()
                {
                    crate::bail_dim_sls!(
                        "{name} penalty {idx} block shape mismatch: col_range={}..{}, local={}x{}, total_dim={p}",
                        col_range.start,
                        col_range.end,
                        local.nrows(),
                        local.ncols()
                    );
                }
            }
            crate::solver::estimate::PenaltySpec::Dense(m)
            | crate::solver::estimate::PenaltySpec::DenseWithMean { matrix: m, .. } => {
                let (r, c) = m.dim();
                if r != p || c != p {
                    crate::bail_dim_sls!("{name} penalty {idx} must be {p}x{p}, got {r}x{c}");
                }
            }
        }
    }
    Ok(())
}


fn validate_cov_block_kind(
    name: &str,
    n: usize,
    bk: &CovariateBlockKind,
) -> Result<(), SurvivalLocationScaleError> {
    match bk {
        CovariateBlockKind::Static(b) => validate_cov_block(name, n, b),
        CovariateBlockKind::TimeVarying(tv) => {
            if tv.design_covariates.nrows() != n {
                crate::bail_dim_sls!(
                    "{name} time-varying covariate design row mismatch: got {}, expected {n}",
                    tv.design_covariates.nrows()
                );
            }
            if tv.time_basis_entry.nrows() != n || tv.time_basis_exit.nrows() != n {
                crate::bail_dim_sls!(
                    "{name} time-varying time basis row mismatch: entry={}, exit={}, expected {n}",
                    tv.time_basis_entry.nrows(),
                    tv.time_basis_exit.nrows()
                );
            }
            if tv.time_basis_derivative_exit.nrows() != n {
                crate::bail_dim_sls!(
                    "{name} time-varying derivative basis row mismatch: got {}, expected {n}",
                    tv.time_basis_derivative_exit.nrows()
                );
            }
            if tv.offset.len() != n {
                crate::bail_dim_sls!(
                    "{name} time-varying offset length mismatch: got {}, expected {n}",
                    tv.offset.len()
                );
            }
            let p_cov = tv.design_covariates.ncols();
            let p_time = tv.time_basis_exit.ncols();
            if tv.time_basis_entry.ncols() != p_time {
                crate::bail_dim_sls!(
                    "{name} time-varying time basis column mismatch: entry={}, exit={}",
                    tv.time_basis_entry.ncols(),
                    p_time
                );
            }
            if tv.time_basis_derivative_exit.ncols() != p_time {
                crate::bail_dim_sls!(
                    "{name} time-varying derivative basis column mismatch: derivative={}, exit={}",
                    tv.time_basis_derivative_exit.ncols(),
                    p_time
                );
            }
            let p_tensor = p_cov * p_time;
            let k = tv.penalties.len();
            if let Some(beta0) = &tv.initial_beta
                && beta0.len() != p_tensor
            {
                crate::bail_dim_sls!(
                    "{name} time-varying initial_beta length mismatch: got {}, expected {p_tensor}",
                    beta0.len()
                );
            }
            if let Some(rho0) = &tv.initial_log_lambdas
                && rho0.len() != k
            {
                crate::bail_dim_sls!(
                    "{name} time-varying initial_log_lambdas length mismatch: got {}, expected {k}",
                    rho0.len()
                );
            }
            for (idx, s) in tv.penalties.iter().enumerate() {
                let (r, c) = s.shape();
                if r != p_tensor || c != p_tensor {
                    crate::bail_dim_sls!(
                        "{name} time-varying penalty {idx} must be {p_tensor}x{p_tensor}, got {r}x{c}"
                    );
                }
            }
            Ok(())
        }
    }
}


/// Build row-wise Kronecker product: each row of the result is
/// kron(cov_row[i,:], time_row[i,:]).
fn assert_rowwise_kronecker_dimensions(n: usize, p_resp: usize, p_cov: usize, context: &str) {
    assert!(
        p_resp > 0 && p_cov > 0,
        "{context} rowwise Kronecker dimensions must be non-empty: n={n}, p_resp={p_resp}, p_cov={p_cov}"
    );
}


fn rowwise_kronecker(cov_design: &DesignMatrix, time_basis: &Array2<f64>) -> DesignMatrix {
    let n = cov_design.nrows();
    let p_cov = cov_design.ncols();
    let p_time = time_basis.ncols();
    assert_rowwise_kronecker_dimensions(n, p_time, p_cov, "survival");
    let op = RowwiseKroneckerOperator::new(cov_design.clone(), shared_dense_arc(time_basis))
        .expect("rowwise kronecker design should have matched row counts");
    DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(op)))
}


fn design_block_from_matrix(design: DesignMatrix) -> DesignBlock {
    match design {
        DesignMatrix::Dense(matrix) => DesignBlock::Dense(matrix),
        other => DesignBlock::Dense(DenseDesignMatrix::from(Arc::new(other))),
    }
}


fn design_column_tail(
    design: &DesignMatrix,
    first_col: usize,
    label: &str,
) -> Result<DesignMatrix, String> {
    let p = design.ncols();
    if first_col > p {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!("{label}: first retained column {first_col} exceeds design width {p}"),
        }
        .into());
    }
    if first_col == 0 {
        return Ok(design.clone());
    }
    let n = design.nrows();
    let active_p = p - first_col;
    if active_p == 0 {
        return Ok(DesignMatrix::from(Array2::<f64>::zeros((n, 0))));
    }

    let chunk_rows = (ROW_CHUNK_BYTE_BUDGET / (p.max(1) * std::mem::size_of::<f64>()))
        .max(1)
        .min(n.max(1));
    let mut out = Array2::<f64>::zeros((n, active_p));
    for start in (0..n).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(n);
        let chunk = design
            .try_row_chunk(start..end)
            .map_err(|e| format!("{label}: failed to materialize design rows: {e}"))?;
        out.slice_mut(s![start..end, ..])
            .assign(&chunk.slice(s![.., first_col..]));
    }
    Ok(DesignMatrix::from(out))
}


fn drop_leading_initial_beta(
    beta: Option<Array1<f64>>,
    fixed_cols: usize,
    full_dim: usize,
    label: &str,
) -> Result<Option<Array1<f64>>, String> {
    let Some(beta) = beta else {
        return Ok(None);
    };
    if beta.len() != full_dim {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "{label}: initial_beta length mismatch before identifiability reduction: got {}, expected {full_dim}",
            beta.len()
        ) }.into());
    }
    Ok(Some(beta.slice(s![fixed_cols..]).to_owned()))
}


fn expand_leading_fixed_beta(
    beta_active: &Array1<f64>,
    fixed_cols: usize,
    full_dim: usize,
    label: &str,
) -> Result<Array1<f64>, String> {
    if fixed_cols > full_dim {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "{label}: fixed column count {fixed_cols} exceeds full width {full_dim}"
            ),
        }
        .into());
    }
    let active_dim = full_dim - fixed_cols;
    if beta_active.len() != active_dim {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "{label}: active beta length mismatch: got {}, expected {active_dim}",
                beta_active.len()
            ),
        }
        .into());
    }
    if fixed_cols == 0 {
        return Ok(beta_active.clone());
    }
    let mut beta_full = Array1::<f64>::zeros(full_dim);
    beta_full.slice_mut(s![fixed_cols..]).assign(beta_active);
    Ok(beta_full)
}


fn drop_leading_penalty_columns(
    penalties: &[PenaltyMatrix],
    nullspace_dims: &[usize],
    initial_log_lambdas: Array1<f64>,
    fixed_cols: usize,
    full_dim: usize,
    label: &str,
) -> Result<(Vec<PenaltyMatrix>, Vec<usize>, Array1<f64>), String> {
    if fixed_cols > full_dim {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "{label}: fixed column count {fixed_cols} exceeds full penalty width {full_dim}"
            ),
        }
        .into());
    }
    if initial_log_lambdas.len() != penalties.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "{label}: initial log-lambda length {} does not match {} penalties",
                initial_log_lambdas.len(),
                penalties.len()
            ),
        }
        .into());
    }
    if fixed_cols == 0 {
        return Ok((
            penalties.to_vec(),
            nullspace_dims.to_vec(),
            initial_log_lambdas,
        ));
    }

    let active_dim = full_dim - fixed_cols;
    if active_dim == 0 {
        return Ok((Vec::new(), Vec::new(), Array1::zeros(0)));
    }

    let structural_nullspace_available = nullspace_dims.len() == penalties.len();
    let mut structural_nullspace_exact = structural_nullspace_available;
    let mut retained_penalties = Vec::new();
    let mut retained_nullspace_dims = Vec::new();
    let mut retained_log_lambdas = Vec::new();

    for (idx, penalty) in penalties.iter().enumerate() {
        if penalty.dim() != full_dim {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "{label}: penalty {idx} has dimension {}, expected {full_dim}",
                    penalty.dim()
                ),
            }
            .into());
        }

        let reduced = match penalty {
            PenaltyMatrix::Blockwise {
                local,
                col_range,
                total_dim,
            } => {
                if *total_dim != full_dim {
                    return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                        "{label}: blockwise penalty {idx} total_dim {total_dim} does not match {full_dim}"
                    ) }.into());
                }
                if col_range.end <= fixed_cols {
                    None
                } else {
                    let active_start = col_range.start.max(fixed_cols);
                    let active_end = col_range.end;
                    let local_start = active_start - col_range.start;
                    let local_end = active_end - col_range.start;
                    if local_start != 0 || local_end != local.nrows() {
                        structural_nullspace_exact = false;
                    }
                    Some(PenaltyMatrix::Blockwise {
                        local: local
                            .slice(s![local_start..local_end, local_start..local_end])
                            .to_owned(),
                        col_range: (active_start - fixed_cols)..(active_end - fixed_cols),
                        total_dim: active_dim,
                    })
                }
            }
            PenaltyMatrix::Dense(matrix) => {
                structural_nullspace_exact = false;
                Some(PenaltyMatrix::Dense(
                    matrix
                        .slice(s![fixed_cols..full_dim, fixed_cols..full_dim])
                        .to_owned(),
                ))
            }
            PenaltyMatrix::KroneckerFactored { .. } => {
                structural_nullspace_exact = false;
                let dense = penalty.to_dense();
                Some(PenaltyMatrix::Dense(
                    dense
                        .slice(s![fixed_cols..full_dim, fixed_cols..full_dim])
                        .to_owned(),
                ))
            }
            PenaltyMatrix::Labeled { label, inner } => {
                structural_nullspace_exact = false;
                let dense = inner.to_dense();
                Some(
                    PenaltyMatrix::Dense(
                        dense
                            .slice(s![fixed_cols..full_dim, fixed_cols..full_dim])
                            .to_owned(),
                    )
                    .with_precision_label(label.clone()),
                )
            }
            PenaltyMatrix::Fixed { log_lambda, inner } => {
                structural_nullspace_exact = false;
                let dense = inner.to_dense();
                Some(
                    PenaltyMatrix::Dense(
                        dense
                            .slice(s![fixed_cols..full_dim, fixed_cols..full_dim])
                            .to_owned(),
                    )
                    .with_fixed_log_lambda(*log_lambda),
                )
            }
        };

        if let Some(reduced) = reduced {
            retained_penalties.push(reduced);
            retained_log_lambdas.push(initial_log_lambdas[idx]);
            if structural_nullspace_available {
                retained_nullspace_dims.push(nullspace_dims[idx]);
            }
        }
    }

    if !structural_nullspace_exact {
        retained_nullspace_dims.clear();
    }

    Ok((
        retained_penalties,
        retained_nullspace_dims,
        Array1::from_vec(retained_log_lambdas),
    ))
}


/// Prepared covariate block data for the family struct.
struct PreparedCovBlock {
    /// Exit design (used as the solver's primary).
    design_exit: DesignMatrix,
    /// Entry design, only for time-varying blocks.
    design_entry: Option<DesignMatrix>,
    /// Exit-time derivative design, only for time-varying blocks.
    design_derivative_exit: Option<DesignMatrix>,
    /// Offset (same for both entry/exit since it comes from other terms).
    offset: Array1<f64>,
    penalties: Vec<PenaltyMatrix>,
    nullspace_dims: Vec<usize>,
    initial_log_lambdas: Option<Array1<f64>>,
    initial_beta: Option<Array1<f64>>,
}


fn prepare_cov_block_kind(
    bk: &CovariateBlockKind,
) -> Result<PreparedCovBlock, SurvivalLocationScaleError> {
    match bk {
        CovariateBlockKind::Static(b) => Ok(PreparedCovBlock {
            design_exit: b.design.clone(),
            design_entry: None,
            design_derivative_exit: None,
            offset: b.offset.clone(),
            penalties: {
                let p = b.design.ncols();
                b.penalties
                    .iter()
                    .map(|spec| match spec {
                        crate::solver::estimate::PenaltySpec::Block {
                            local, col_range, ..
                        } => PenaltyMatrix::Blockwise {
                            local: local.clone(),
                            col_range: col_range.clone(),
                            total_dim: p,
                        },
                        crate::solver::estimate::PenaltySpec::Dense(m)
                        | crate::solver::estimate::PenaltySpec::DenseWithMean {
                            matrix: m, ..
                        } => PenaltyMatrix::Dense(m.clone()),
                    })
                    .collect()
            },
            nullspace_dims: b.nullspace_dims.clone(),
            initial_log_lambdas: b.initial_log_lambdas.clone(),
            initial_beta: b.initial_beta.clone(),
        }),
        CovariateBlockKind::TimeVarying(tv) => {
            let design_exit = rowwise_kronecker(&tv.design_covariates, &tv.time_basis_exit);
            let design_entry = rowwise_kronecker(&tv.design_covariates, &tv.time_basis_entry);
            let design_derivative_exit =
                rowwise_kronecker(&tv.design_covariates, &tv.time_basis_derivative_exit);
            Ok(PreparedCovBlock {
                design_exit,
                design_entry: Some(design_entry),
                design_derivative_exit: Some(design_derivative_exit),
                offset: tv.offset.clone(),
                penalties: tv.penalties.clone(),
                nullspace_dims: vec![],
                initial_log_lambdas: tv.initial_log_lambdas.clone(),
                initial_beta: tv.initial_beta.clone(),
            })
        }
    }
}


fn build_survival_covariate_block_from_design(
    cov_design: &TermCollectionDesign,
    template: &SurvivalCovariateTermBlockTemplate,
    offset: &Array1<f64>,
    initial_log_lambdas: Option<Array1<f64>>,
    initial_beta: Option<Array1<f64>>,
) -> Result<CovariateBlockKind, String> {
    match template {
        SurvivalCovariateTermBlockTemplate::Static => {
            Ok(CovariateBlockKind::Static(ParameterBlockInput {
                design: cov_design.design.clone(),
                offset: offset.clone(),
                penalties: cov_design
                    .penalties
                    .iter()
                    .map(crate::solver::estimate::PenaltySpec::from_blockwise_ref)
                    .collect(),
                nullspace_dims: cov_design.nullspace_dims.clone(),
                initial_log_lambdas,
                initial_beta,
            }))
        }
        SurvivalCovariateTermBlockTemplate::TimeVarying {
            time_basis_entry,
            time_basis_exit,
            time_basis_derivative_exit,
            time_penalties,
        } => {
            let p_cov = cov_design.design.ncols();
            let p_time = time_basis_exit.ncols();
            let design_covariates = cov_design.design.clone();
            let i_cov = Array2::<f64>::eye(p_cov);
            let i_time = Array2::<f64>::eye(p_time);
            let cov_dense_for_kronecker: Vec<Array2<f64>> = cov_design
                .penalties
                .iter()
                .map(|bp| bp.to_global(p_cov))
                .collect();
            let mut penalties =
                Vec::with_capacity(cov_dense_for_kronecker.len() + time_penalties.len());
            for s_cov in &cov_dense_for_kronecker {
                penalties.push(PenaltyMatrix::KroneckerFactored {
                    left: s_cov.clone(),
                    right: i_time.clone(),
                });
            }
            for s_time in time_penalties {
                penalties.push(PenaltyMatrix::KroneckerFactored {
                    left: i_cov.clone(),
                    right: s_time.clone(),
                });
            }
            Ok(CovariateBlockKind::TimeVarying(
                TimeDependentCovariateBlockInput {
                    design_covariates,
                    time_basis_entry: time_basis_entry.clone(),
                    time_basis_exit: time_basis_exit.clone(),
                    time_basis_derivative_exit: time_basis_derivative_exit.clone(),
                    penalties,
                    initial_log_lambdas,
                    initial_beta,
                    offset: offset.clone(),
                },
            ))
        }
    }
}


/// Survival time-varying tensorization adapter for the shared spatial-ψ engine.
///
/// A time-dependent survival covariate represents each spatial design row as the
/// rowwise-Kronecker of the (spatial) base row against three time bases — exit,
/// entry, and the exit-time derivative — stacked vertically, while each spatial
/// penalty is Kronecker-multiplied against the time identity. This is a *uniform*
/// coordinate change applied to every block the shared engine assembles, so we
/// invert the dependency: the engine owns the spatial-ψ block construction and
/// this adapter only supplies the tensorization via [`SpatialPsiBlockTransform`].
struct SurvivalTimeVaryingPsiTransform {
    time_basis_entry: Array2<f64>,
    time_basis_exit: Array2<f64>,
    time_basis_derivative_exit: Array2<f64>,
}


impl crate::families::spatial_psi_bridge::SpatialPsiBlockTransform
    for SurvivalTimeVaryingPsiTransform
{
    fn transform_operator(
        &self,
        op: Arc<dyn crate::custom_family::CustomFamilyPsiDerivativeOperator>,
    ) -> Result<Arc<dyn crate::custom_family::CustomFamilyPsiDerivativeOperator>, String> {
        build_rowwise_kronecker_psi_operator(
            op,
            vec![
                shared_dense_arc(&self.time_basis_exit),
                shared_dense_arc(&self.time_basis_entry),
                shared_dense_arc(&self.time_basis_derivative_exit),
            ],
        )
    }

    fn transform_design(&self, base: Array2<f64>) -> Array2<f64> {
        let base_dm = DesignMatrix::Dense(DenseDesignMatrix::from(base));
        let exit_design = rowwise_kronecker(&base_dm, &self.time_basis_exit);
        let entry_design = rowwise_kronecker(&base_dm, &self.time_basis_entry);
        let deriv_design = rowwise_kronecker(&base_dm, &self.time_basis_derivative_exit);
        let exit_cow = exit_design.to_dense_cow();
        let entry_cow = entry_design.to_dense_cow();
        let deriv_cow = deriv_design.to_dense_cow();
        let n = exit_cow.nrows();
        let p = exit_cow.ncols();
        let mut stacked = Array2::<f64>::zeros((3 * n, p));
        stacked.slice_mut(s![0..n, ..]).assign(&*exit_cow);
        stacked.slice_mut(s![n..2 * n, ..]).assign(&*entry_cow);
        stacked.slice_mut(s![2 * n..3 * n, ..]).assign(&*deriv_cow);
        stacked
    }

    fn transform_penalty(&self, base: Array2<f64>) -> Array2<f64> {
        let i_time = Array2::<f64>::eye(self.time_basis_exit.ncols());
        kronecker_product(&base, &i_time)
    }
}


/// Survival covariate spatial-ψ derivatives: a thin adapter over the shared
/// exact-derivative engine [`build_block_spatial_psi_derivatives_with_transform`].
/// The `Static` template emits blocks unchanged; the `TimeVarying` template
/// supplies a [`SurvivalTimeVaryingPsiTransform`] so the same engine produces the
/// time-tensorized blocks without re-implementing block assembly.
fn build_survival_covariate_block_psi_derivatives(
    data: ndarray::ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    template: &SurvivalCovariateTermBlockTemplate,
) -> Result<Option<Vec<CustomFamilyBlockPsiDerivative>>, String> {
    match template {
        SurvivalCovariateTermBlockTemplate::Static => {
            crate::families::spatial_psi_bridge::build_block_spatial_psi_derivatives(
                data,
                resolvedspec,
                design,
            )
        }
        SurvivalCovariateTermBlockTemplate::TimeVarying {
            time_basis_entry,
            time_basis_exit,
            time_basis_derivative_exit,
            ..
        } => {
            let transform = SurvivalTimeVaryingPsiTransform {
                time_basis_entry: time_basis_entry.clone(),
                time_basis_exit: time_basis_exit.clone(),
                time_basis_derivative_exit: time_basis_derivative_exit.clone(),
            };
            crate::families::spatial_psi_bridge::build_block_spatial_psi_derivatives_with_transform(
                data,
                resolvedspec,
                design,
                &transform,
            )
        }
    }
}


fn survival_psi_derivatives_support_exact_joint_hessian(
    derivs: &[CustomFamilyBlockPsiDerivative],
) -> bool {
    let psi_dim = derivs.len();
    derivs.iter().all(|deriv| {
        let design_ok = deriv.implicit_operator.is_some()
            || deriv
                .x_psi_psi
                .as_ref()
                .is_some_and(|rows| rows.len() == psi_dim);
        let penalty_ok = deriv
            .s_psi_psi_components
            .as_ref()
            .is_some_and(|rows| rows.len() == psi_dim)
            || deriv
                .s_psi_psi
                .as_ref()
                .is_some_and(|rows| rows.len() == psi_dim);
        design_ok && penalty_ok
    })
}


fn build_survival_two_block_exact_joint_setup(
    data: ndarray::ArrayView2<'_, f64>,
    thresholdspec: &TermCollectionSpec,
    log_sigmaspec: &TermCollectionSpec,
    rho0: Array1<f64>,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> ExactJointHyperSetup {
    // Survival location-scale uses the shared engine directly: the rho seed is
    // already assembled by the caller (penalty + link-wiggle layout), and the
    // two linear predictors (threshold, log sigma) supply the per-block
    // log(kappa) geometry in theta order.
    build_location_scale_exact_joint_setup(
        data,
        &[thresholdspec, log_sigmaspec],
        rho0,
        kappa_options,
    )
}


fn filtered_initial_beta(hint: Option<&Array1<f64>>, expected: usize) -> Option<Array1<f64>> {
    hint.filter(|beta| beta.len() == expected).cloned()
}


fn structural_time_initial_beta_guess(
    design_derivative_exit: &Array2<f64>,
    derivative_offset_exit: &Array1<f64>,
    age_exit: &Array1<f64>,
    derivative_guard: f64,
    coefficient_lower_bounds: Option<&Array1<f64>>,
) -> Option<Array1<f64>> {
    let n = design_derivative_exit.nrows();
    let p = design_derivative_exit.ncols();
    if p == 0 || n == 0 || derivative_offset_exit.len() != n || age_exit.len() != n {
        return None;
    }

    let mut target = Array1::<f64>::zeros(n);
    for i in 0..n {
        let desired = 1.0 / age_exit[i].max(STRUCTURAL_GUESS_AGE_FLOOR);
        target[i] = (desired - derivative_offset_exit[i]).max(0.0);
    }

    let xtx = crate::faer_ndarray::fast_ata(design_derivative_exit);
    let xty = fast_atv(design_derivative_exit, &target);
    let eps =
        STRUCTURAL_GUESS_RIDGE_REL * (0..p).map(|i| xtx[[i, i]]).fold(0.0_f64, f64::max).max(1.0);
    let mut lhs = xtx;
    for i in 0..p {
        lhs[[i, i]] += eps;
    }

    use crate::faer_ndarray::FaerCholesky;
    let chol = lhs.cholesky(faer::Side::Lower).ok()?;
    let mut beta_init = chol.solvevec(&xty);
    if let Some(lower_bounds) = coefficient_lower_bounds
        && let Some(constraints) = lower_bound_constraints(lower_bounds)
    {
        // `beta_init` is the length-`p` ridge solution and `constraints` is
        // derived from the same `p`-column derivative design, so the projection
        // is dimensionally consistent by construction. If a future refactor
        // breaks that invariant, abandon the structural guess rather than
        // propagate a hard error out of this best-effort warm start.
        beta_init = project_onto_linear_constraints(p, &constraints, Some(&beta_init)).ok()?;
    }

    let d_raw_init = fast_av(design_derivative_exit, &beta_init) + derivative_offset_exit;
    if d_raw_init
        .iter()
        .all(|v| v.is_finite() && *v >= derivative_guard)
    {
        Some(beta_init)
    } else {
        None
    }
}


/// Whether the scale block carries no penalties — a single constant `σ`
/// (the parametric-AFT regime). This is exactly the condition under which
/// `prepare_survival_location_scale_model` pins the time-warp ρ seed AT the
/// inner ρ box bound (the affine-baseline limit). On that dead-flat,
/// statistically-unidentified time ridge the seed-screening cascade has no
/// useful signal to rank — every capped proxy fit collapses to non-finite
/// cost and the cascade escalates to its uncapped final stage, paying a full
/// inner solve per seed on the near-singular Hessian (the multi-minute
/// no-iteration-log stall, #736/#735/#721). The pinned seed is already the
/// correct optimum, so screening is pure cost here.
///
/// A genuinely flexible scale (`noise_formula = s(...)`) carries log-sigma
/// penalties, never reaches the seed-pinning branch, and keeps full
/// screening.
fn survival_constant_scale(spec: &SurvivalLocationScaleSpec) -> bool {
    match &spec.log_sigma_block {
        CovariateBlockKind::Static(block) => block.penalties.is_empty(),
        CovariateBlockKind::TimeVarying(block) => block.penalties.is_empty(),
    }
}


fn survival_blockwise_fit_options(spec: &SurvivalLocationScaleSpec) -> BlockwiseFitOptions {
    BlockwiseFitOptions {
        inner_max_cycles: spec.max_iter,
        inner_tol: spec.tol,
        outer_max_iter: BLOCKWISE_OUTER_MAX_ITER,
        outer_tol: BLOCKWISE_OUTER_TOL,
        compute_covariance: true,
        cache_session: spec.cache_session.clone(),
        cache_mirror_sessions: spec.cache_mirror_sessions.clone(),
        // Constant-scale (parametric-AFT) fits pin the time-warp ρ seed at the
        // identified affine-baseline limit; re-screening that already-correct
        // seed across the flat unidentified time ridge only stalls. Genuinely
        // flexible scale/spatial fits keep the default `true` and full screening.
        screen_initial_rho: !survival_constant_scale(spec),
        ..BlockwiseFitOptions::default()
    }
}


fn validate_survival_location_scale_spec(
    spec: &SurvivalLocationScaleSpec,
) -> Result<(), SurvivalLocationScaleError> {
    let n = spec.event_target.len();
    let monotone_time_wiggle_ncols = spec.timewiggle_block.as_ref().map_or(0, |w| w.ncols);
    match &spec.inverse_link {
        InverseLink::Standard(StandardLink::Log) => {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "fit_survival_location_scale does not support Standard(Log)".to_string(),
            });
        }
        InverseLink::Standard(StandardLink::Logit)
        | InverseLink::Standard(StandardLink::Probit)
        | InverseLink::Standard(StandardLink::CLogLog)
        | InverseLink::Standard(StandardLink::Identity)
        | InverseLink::LatentCLogLog(_)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => {}
    }
    if n == 0 {
        return Err(SurvivalLocationScaleError::InternalInvariant {
            reason: "fit_survival_location_scale: empty dataset".to_string(),
        });
    }
    if spec.age_entry.len() != n || spec.age_exit.len() != n || spec.weights.len() != n {
        crate::bail_dim_sls!("fit_survival_location_scale: top-level input size mismatch");
    }
    if !(spec.tol.is_finite() && spec.tol > 0.0) {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!("fit_survival_location_scale: invalid tol {}", spec.tol),
        });
    }
    if spec.max_iter == 0 {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: "fit_survival_location_scale: max_iter must be > 0".to_string(),
        });
    }
    if !spec.derivative_guard.is_finite() || spec.derivative_guard <= 0.0 {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!(
                "fit_survival_location_scale: derivative_guard must be > 0, got {}",
                spec.derivative_guard
            ),
        });
    }
    validate_time_block(
        n,
        &spec.time_block,
        spec.derivative_guard,
        monotone_time_wiggle_ncols,
    )?;
    validate_cov_block_kind("threshold_block", n, &spec.threshold_block)?;
    validate_cov_block_kind("log_sigma_block", n, &spec.log_sigma_block)?;
    if let Some(w) = spec.timewiggle_block.as_ref() {
        if w.ncols == 0 {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "timewiggle_block must have at least one coefficient".to_string(),
            });
        }
        if w.ncols >= spec.time_block.design_exit.ncols() {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "timewiggle_block.ncols must be smaller than time_block columns: wiggle={}, total={}",
                    w.ncols,
                    spec.time_block.design_exit.ncols()
                ),
            });
        }
        if w.knots.len() < 2 * (w.degree + 1) {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "timewiggle_block knot vector is too short for degree {}: got {} knots",
                    w.degree,
                    w.knots.len()
                ),
            });
        }
    }
    if let Some(w) = spec.linkwiggle_block.as_ref() {
        validatewiggle_block(n, w)?;
    }
    for i in 0..n {
        if !spec.age_entry[i].is_finite()
            || !spec.age_exit[i].is_finite()
            || spec.age_exit[i] < spec.age_entry[i]
        {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "fit_survival_location_scale: invalid interval at row {} (entry={}, exit={})",
                    i + 1,
                    spec.age_entry[i],
                    spec.age_exit[i]
                ),
            });
        }
        if !spec.weights[i].is_finite() || spec.weights[i] < 0.0 {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: format!(
                    "fit_survival_location_scale: invalid weight at row {} ({})",
                    i + 1,
                    spec.weights[i]
                ),
            });
        }
        if !spec.event_target[i].is_finite() || !(0.0..=1.0).contains(&spec.event_target[i]) {
            return Err(SurvivalLocationScaleError::ConstraintViolation {
                reason: format!(
                    "fit_survival_location_scale: event_target must be in [0,1], found {} at row {}",
                    spec.event_target[i],
                    i + 1
                ),
            });
        }
    }
    Ok(())
}


fn prepare_survival_location_scale_model(
    spec: &SurvivalLocationScaleSpec,
) -> Result<PreparedSurvivalLocationScaleModel, String> {
    validate_survival_location_scale_spec(spec)?;
    let n = spec.event_target.len();
    let protected_timewiggle_cols = spec.timewiggle_block.as_ref().map_or(0, |w| w.ncols);
    // Constant-scale AFT regime: a single global σ identifies the time baseline
    // only through its affine `1 + log t` transform (the parametric AFT), so the
    // flexible I-spline time-warp's non-affine deviation is statistically
    // unidentified (issue #736/#735/#721). When there is also no monotone
    // timewiggle reintroducing flexibility, reduce the time block to its
    // identifiable parametric (affine null-space) rank so the inner coupled
    // exact-joint solve has no unconstrained free direction to choke on. A
    // genuinely flexible scale (`noise_formula = s(...)`, log_sigma penalties
    // present) or an active timewiggle keeps the full monotone I-spline because
    // the varying σ / wiggle DOES identify the non-affine baseline shape.
    let reduce_time_to_parametric = survival_constant_scale(spec) && protected_timewiggle_cols == 0;
    // Log entry/exit times for the canonical unit-log-t warp pin (issue #892).
    // The reduced AFT warp is folded into the geometry offsets as the EXACT
    // `log t` transform built straight from the event times — `log t` value at
    // entry/exit and `1/t` derivative at exit — bypassing the I-spline's curved
    // image of log t (the residual curvature was what kept σ miscalibrated). The
    // floor matches `checked_log_survival_times` (survival_construction.rs), the
    // same map under which the I-spline time basis is built over `log t`.
    let log_time_entry = spec.age_entry.mapv(|t| {
        t.max(crate::families::survival_construction::SURVIVAL_TIME_FLOOR)
            .ln()
    });
    let log_time_exit = spec.age_exit.mapv(|t| {
        t.max(crate::families::survival_construction::SURVIVAL_TIME_FLOOR)
            .ln()
    });
    let mut time_prepared = prepare_identified_time_block(
        &spec.time_block,
        spec.derivative_guard,
        protected_timewiggle_cols,
        reduce_time_to_parametric,
        log_time_entry.view(),
        log_time_exit.view(),
    )?;

    if time_prepared.initial_beta.is_none() {
        // Use the AUGMENTED derivative offset (issue #892): on the pinned-warp
        // path the guard `(X' z_c) β_c + offset' ≥ guard` is built against the
        // folded offset, so the seed must satisfy the same offset to land
        // feasible.
        time_prepared.initial_beta = structural_time_initial_beta_guess(
            &time_prepared.design_derivative_exit,
            &time_prepared.derivative_offset_exit,
            &spec.age_exit,
            spec.derivative_guard,
            time_prepared.coefficient_lower_bounds.as_ref(),
        );
    }

    let time_solver_design = DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(
        MultiChannelOperator::new(vec![
            DesignMatrix::Dense(DenseDesignMatrix::from(shared_dense_arc(
                &time_prepared.design_entry,
            ))),
            DesignMatrix::Dense(DenseDesignMatrix::from(shared_dense_arc(
                &time_prepared.design_exit,
            ))),
            DesignMatrix::Dense(DenseDesignMatrix::from(shared_dense_arc(
                &time_prepared.design_derivative_exit,
            ))),
        ])?,
    )));
    // Augmented offsets (issue #892): on the pinned-warp reduce path these carry
    // the folded unit-log-t value/derivative contributions; on every other path
    // they equal `spec.time_block.offset_*` verbatim.
    let time_stacked_offset = crate::linalg::utils::stack_offsets(&[
        &time_prepared.offset_entry,
        &time_prepared.offset_exit,
        &time_prepared.derivative_offset_exit,
    ]);
    // Canonical n-row view of the time block: `spec.design` is the n-row
    // exit design (one row per observation, len(eta_canonical) = n).
    // The solver's stacked `[entry; exit; derivative_exit]` operator and
    // its matching `3*n`-row offset live in `spec.stacked_design` /
    // `spec.stacked_offset`; the solver consumes those via
    // `solver_design()` / `solver_offset()`.  The audit and shape policy
    // only read `spec.design`, so every block's audit-visible row count
    // is `n`.
    let time_canonical_design: DesignMatrix =
        DesignMatrix::Dense(DenseDesignMatrix::from(time_prepared.design_exit.clone()));
    let timespec = ParameterBlockSpec {
        name: "time_transform".to_string(),
        design: time_canonical_design,
        offset: time_prepared.offset_exit.clone(),
        penalties: time_prepared
            .penalties
            .iter()
            .cloned()
            .map(PenaltyMatrix::Dense)
            .collect(),
        nullspace_dims: time_prepared.nullspace_dims.clone(),
        // A caller-supplied per-penalty time seed is indexed by the ORIGINAL
        // (un-reduced) penalty set. When the constant-scale-AFT reduction
        // dropped those penalties (`time_prepared.penalties` empty / shorter
        // than the original), that seed no longer matches and is irrelevant —
        // the reduced affine block is unpenalized — so fall back to the empty
        // seed for the (zero) retained penalties.
        initial_log_lambdas: initial_log_lambdas(
            &time_prepared.penalties,
            if time_prepared.penalties.len() == spec.time_block.penalties.len() {
                spec.time_block.initial_log_lambdas.clone()
            } else {
                None
            },
        )?,
        initial_beta: time_prepared.initial_beta.clone(),
        // Canonical-gauge ownership for the location-scale joint design.
        //
        // The three coupled blocks (`time_transform`, `threshold`,
        // `log_sigma`) each contribute a constant / intercept-like direction
        // into the flat n-row joint design the pre-fit identifiability audit
        // RRQRs (`solver::identifiability_audit::audit_identifiability`).
        // Those constant directions are mutually aliased (e.g. for a single
        // linear covariate the `time_transform[0] ~ threshold[0]` overlap is
        // ≈ 0.98), so the joint design is genuinely rank-deficient by exactly
        // the number of surplus constants. The audit can only *attribute and
        // drop* a redundant joint column to a strictly lower-priority block;
        // with the previous uniform `gauge_priority: 100` the surplus
        // direction was un-attributable and the audit escalated to
        // `fatal = true`, refusing every well-posed fit (issue #366).
        //
        // Assigning strictly descending priorities makes the surplus
        // constant deterministically attributable: `time_transform` owns the
        // shared constant (it carries the structural monotone baseline that
        // anchors the whole location-scale parameterisation), and any aliased
        // column is dropped from the lower-priority `threshold` / `log_sigma`
        // / `linkwiggle` blocks. This is the exact gauge-ownership contract
        // documented by `identifiability_canonical::
        // canonical_five_block_gauge_ownership_succeeds_with_attribution` and
        // already used by survival marginal-slope (time=200 highest).
        gauge_priority: 200,
        jacobian_callback: None,
        stacked_design: Some(time_solver_design),
        stacked_offset: Some(time_stacked_offset),
    };

    let threshold_prep = prepare_cov_block_kind(&spec.threshold_block)?;
    let threshold_full_ncols = threshold_prep.design_exit.ncols();
    let time_reduced_to_parametric = time_block_reduces_to_parametric(
        &spec.time_block.penalties,
        spec.time_block.design_exit.ncols(),
        survival_constant_scale(spec),
        protected_timewiggle_cols,
    );
    let threshold_fixed_cols = if time_reduced_to_parametric {
        if time_prepared.pinned_free_row_constant {
            // Reduced + unit-log-t warp PINNED (issue #892): the single surviving
            // free time column `z_c` is ROW-CONSTANT — it now carries the
            // location level. Keeping the threshold intercept too would put TWO
            // constant columns into the direct-MLE joint design, making the
            // Hessian PSD and rank-deficient by 1; the damped Newton then stalls
            // along the alias and leaves the lowest-leverage coefficient (e.g. an
            // x0:x1 interaction) stuck at its cold-start 0. Drop the LEADING
            // threshold intercept(s) so the location level is owned solely by the
            // pinned time constant `z_c`. The threshold then contributes only its
            // genuine covariate slopes; finalize pads the dropped intercept slot
            // with 0 (intercept-invariant for the g-contrast and the surface
            // anchor). Use the same leading-intercept inference the flexible path
            // uses (returns 0 for an intercept-free threshold design), so this is
            // robust to designs that carry no constant column to alias.
            infer_non_intercept_start_design(&threshold_prep.design_exit, &spec.weights)?
                .min(threshold_full_ncols)
        } else {
            // Reduced but pin did NOT fire (both-columns-free fallback): the
            // reduced I-spline columns are strictly monotone in t and span no
            // constant-in-t direction, so the time block carries no location
            // intercept the gauge contract would attribute to it. Keep the
            // threshold (location) intercept here — it is the free location level
            // b0, NOT aliased with the multiplicative scale constant nor with any
            // time-warp constant (there is none) — mirroring why the constant
            // log_sigma block keeps its intercept (`log_sigma_fixed_cols = 0`).
            // Dropping it (#736) left the raw covariate column to double as both
            // level and slope, pinning the covariate to a wrong-signed value.
            0
        }
    } else {
        infer_non_intercept_start_design(&threshold_prep.design_exit, &spec.weights)?
            .min(threshold_full_ncols)
    };
    let threshold_design = design_column_tail(
        &threshold_prep.design_exit,
        threshold_fixed_cols,
        "survival location-scale threshold design",
    )?;
    let threshold_entry_design = if let Some(x_entry) = threshold_prep.design_entry.as_ref() {
        Some(design_column_tail(
            x_entry,
            threshold_fixed_cols,
            "survival location-scale threshold entry design",
        )?)
    } else {
        None
    };
    let threshold_deriv_design =
        if let Some(x_deriv) = threshold_prep.design_derivative_exit.as_ref() {
            Some(design_column_tail(
                x_deriv,
                threshold_fixed_cols,
                "survival location-scale threshold derivative design",
            )?)
        } else {
            None
        };
    let threshold_initial_log_lambdas = initial_log_lambdas(
        &threshold_prep.penalties,
        threshold_prep.initial_log_lambdas.clone(),
    )?;
    let (threshold_penalties, threshold_nullspace_dims, threshold_initial_log_lambdas) =
        drop_leading_penalty_columns(
            &threshold_prep.penalties,
            &threshold_prep.nullspace_dims,
            threshold_initial_log_lambdas,
            threshold_fixed_cols,
            threshold_full_ncols,
            "survival location-scale threshold penalties",
        )?;
    let threshold_initial_beta = drop_leading_initial_beta(
        threshold_prep.initial_beta.clone(),
        threshold_fixed_cols,
        threshold_full_ncols,
        "survival location-scale threshold",
    )?;
    // For time-varying threshold blocks, the solver consumes a stacked
    // `[exit; entry; deriv]` operator (3*n rows) via `solver_design()`;
    // the canonical `spec.design` is the n-row exit channel only — the
    // single field both audit and shape policy read.  Non-time-varying
    // threshold blocks have no stacking: `stacked_design`/`stacked_offset`
    // stay `None` and the solver reads `design` directly.
    let (threshold_stacked_design, threshold_stacked_offset) =
        if let Some(x_entry) = threshold_entry_design.as_ref() {
            let x_deriv = threshold_deriv_design.as_ref().ok_or_else(|| {
                "time-varying threshold block is missing its exit derivative design".to_string()
            })?;
            (
                Some(DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(
                    MultiChannelOperator::new(vec![
                        threshold_design.clone(),
                        x_entry.clone(),
                        x_deriv.clone(),
                    ])?,
                )))),
                Some(crate::linalg::utils::stack_offsets(&[
                    &threshold_prep.offset,
                    &threshold_prep.offset,
                    &Array1::zeros(n),
                ])),
            )
        } else {
            (None, None)
        };
    let survival_primary_design = DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(
        BlockDesignOperator::new(vec![
            DesignBlock::Dense(DenseDesignMatrix::from(shared_dense_arc(
                &time_prepared.design_exit,
            ))),
            design_block_from_matrix(threshold_design.clone()),
        ])?,
    )));

    let log_sigma_prep = prepare_cov_block_kind(&spec.log_sigma_block)?;
    let non_intercept_start =
        infer_non_intercept_start_design(&log_sigma_prep.design_exit, &spec.weights)?;
    let log_sigma_full_ncols = log_sigma_prep.design_exit.ncols();
    // The scale channel enters the survival location-scale likelihood as
    // `z = (h(t) - eta_t(x)) / exp(eta_sigma)`: `eta_sigma` is MULTIPLICATIVE,
    // not an additive predictor. The constant (intercept) direction of the
    // scale block is therefore the free overall sigma parameter — it is NOT
    // aliased with the additive location/time constant that `time_transform`
    // and `threshold` share, so no higher-priority block owns the scale-block
    // gauge direction. The only genuine alias the scale block can carry is
    // between its NON-intercept covariate columns and the location predictor,
    // and that aliasing is already removed by the scale-deviation
    // reparameterisation below (which residualises columns from
    // `non_intercept_start` onward against the primary location design).
    // Hence the scale block drops NO leading columns: dropping its lone
    // intercept (the constant-sigma case, `non_intercept_start == full_ncols`)
    // would canonicalise a genuinely identifiable free parameter to width 0
    // and refuse the coupled three-block startup certification (#736), and
    // over-reducing it desynchronises the raw/active block widths at the
    // covariance-lift boundary (#735).
    let log_sigma_fixed_cols = 0usize;
    let scale_transform = build_scale_deviation_transform_design(
        &survival_primary_design,
        &log_sigma_prep.design_exit,
        &spec.weights,
        non_intercept_start,
    )?;
    let log_sigma_full_design = build_scale_deviation_operator(
        survival_primary_design.clone(),
        log_sigma_prep.design_exit.clone(),
        &scale_transform,
    )?;
    let log_sigma_design = design_column_tail(
        &log_sigma_full_design,
        log_sigma_fixed_cols,
        "survival location-scale log-sigma design",
    )?;
    let log_sigma_entry_design = if let Some(x_ls_entry) = log_sigma_prep.design_entry.as_ref() {
        let full_entry = build_scale_deviation_operator(
            survival_primary_design.clone(),
            x_ls_entry.clone(),
            &scale_transform,
        )?;
        Some(design_column_tail(
            &full_entry,
            log_sigma_fixed_cols,
            "survival location-scale log-sigma entry design",
        )?)
    } else {
        None
    };
    let log_sigma_deriv_design =
        if let Some(ls_deriv) = log_sigma_prep.design_derivative_exit.as_ref() {
            Some(design_column_tail(
                ls_deriv,
                log_sigma_fixed_cols,
                "survival location-scale log-sigma derivative design",
            )?)
        } else {
            None
        };
    let log_sigma_initial_log_lambdas = initial_log_lambdas(
        &log_sigma_prep.penalties,
        log_sigma_prep.initial_log_lambdas.clone(),
    )?;
    let (log_sigma_penalties, log_sigma_nullspace_dims, log_sigma_initial_log_lambdas) =
        drop_leading_penalty_columns(
            &log_sigma_prep.penalties,
            &log_sigma_prep.nullspace_dims,
            log_sigma_initial_log_lambdas,
            log_sigma_fixed_cols,
            log_sigma_full_ncols,
            "survival location-scale log-sigma penalties",
        )?;
    let log_sigma_initial_beta = drop_leading_initial_beta(
        log_sigma_prep.initial_beta.clone(),
        log_sigma_fixed_cols,
        log_sigma_full_ncols,
        "survival location-scale log-sigma",
    )?;

    // Reduced parametric-AFT regime (issue #736/#735/#721): when the time-warp
    // has collapsed to its affine null space, there is no wiggle, and every
    // surviving location/scale penalty is a full-rank parametric ridge
    // (`nullspace_dim == 0` — e.g. the linear-term `LinearTermRidge` on `age`),
    // drop those ridges. They are NOT wiggliness penalties: a single linear
    // coefficient has nothing to smooth, so the ridge carries no smoothing
    // parameter worth a vacuous outer ρ coordinate, and its default λ would only
    // bias the parametric coefficient away from the unpenalized
    // `survreg`/`lifelines` MLE this regime must reproduce. Dropping them
    // (exactly as the reduced time block drops its projected-to-zero penalties)
    // takes `k_threshold`/`k_log_sigma` to 0, so the dispatch
    // (`is_reduced_parametric_aft`) routes the fit to the direct unpenalized
    // parametric-AFT Newton MLE with zero outer coordinates. The OUTER ρ layout
    // (`fit_survival_location_scale_terms`) applies the SAME predicate to the
    // same boot-design penalties, so the inner and outer counts stay identical.
    // Evaluate the regime predicate on the PRE-drop block penalties
    // (`threshold_prep`/`log_sigma_prep`), which are an exact copy of the boot
    // designs the OUTER layout (`fit_survival_location_scale_terms`) inspects —
    // `prepare_cov_block_kind` clones `b.penalties`/`b.nullspace_dims` straight
    // from the block built off that boot design. Reading the same source on both
    // sides guarantees the inner and outer ρ counts are computed from identical
    // penalty/null-space metadata, so they can never diverge (a divergence would
    // desynchronise `k_threshold` between the layout and the prepared model).
    let drop_parametric_ridges = survival_reduced_parametric_aft_regime(
        &spec.time_block.penalties,
        spec.time_block.design_exit.ncols(),
        survival_constant_scale(spec),
        protected_timewiggle_cols,
        &threshold_prep.nullspace_dims,
        threshold_prep.penalties.len(),
        &log_sigma_prep.nullspace_dims,
        log_sigma_prep.penalties.len(),
        spec.linkwiggle_block.is_some(),
    );
    let (threshold_penalties, threshold_nullspace_dims, threshold_initial_log_lambdas) =
        if drop_parametric_ridges {
            (Vec::new(), Vec::new(), Array1::<f64>::zeros(0))
        } else {
            (
                threshold_penalties,
                threshold_nullspace_dims,
                threshold_initial_log_lambdas,
            )
        };
    let (log_sigma_penalties, log_sigma_nullspace_dims, log_sigma_initial_log_lambdas) =
        if drop_parametric_ridges {
            (Vec::new(), Vec::new(), Array1::<f64>::zeros(0))
        } else {
            (
                log_sigma_penalties,
                log_sigma_nullspace_dims,
                log_sigma_initial_log_lambdas,
            )
        };

    let thresholdspec = ParameterBlockSpec {
        name: "threshold".to_string(),
        design: threshold_design.clone(),
        offset: threshold_prep.offset.clone(),
        penalties: threshold_penalties.clone(),
        nullspace_dims: threshold_nullspace_dims.clone(),
        initial_log_lambdas: threshold_initial_log_lambdas,
        initial_beta: threshold_initial_beta,
        // Lower than `time_transform` (200): the location-channel covariate
        // block yields the shared constant direction to the time baseline.
        // See the canonical-gauge ownership note on the `time_transform`
        // spec above (issue #366).
        gauge_priority: 150,
        jacobian_callback: None,
        stacked_design: threshold_stacked_design,
        stacked_offset: threshold_stacked_offset,
    };

    // Same canonical-vs-stacked split as the threshold block: time-varying
    // log_sigma stacks `[exit; entry; deriv]` (3*n rows) into
    // `stacked_design`; the canonical `spec.design` is the n-row exit
    // channel only.
    let (log_sigma_stacked_design, log_sigma_stacked_offset) =
        if let Some(ref ls_entry) = log_sigma_entry_design {
            let ls_deriv = log_sigma_deriv_design.as_ref().ok_or_else(|| {
                "time-varying log-sigma block is missing its exit derivative design".to_string()
            })?;
            (
                Some(DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(
                    MultiChannelOperator::new(vec![
                        log_sigma_design.clone(),
                        ls_entry.clone(),
                        ls_deriv.clone(),
                    ])?,
                )))),
                Some(crate::linalg::utils::stack_offsets(&[
                    &log_sigma_prep.offset,
                    &log_sigma_prep.offset,
                    &Array1::zeros(n),
                ])),
            )
        } else {
            (None, None)
        };
    let log_sigmaspec = ParameterBlockSpec {
        name: "log_sigma".to_string(),
        design: log_sigma_design.clone(),
        offset: log_sigma_prep.offset.clone(),
        penalties: log_sigma_penalties.clone(),
        nullspace_dims: log_sigma_nullspace_dims.clone(),
        initial_log_lambdas: log_sigma_initial_log_lambdas,
        initial_beta: log_sigma_initial_beta,
        // Below `time_transform` (200) and `threshold` (150): the scale
        // channel yields the shared constant direction to the location
        // blocks. See the canonical-gauge ownership note on the
        // `time_transform` spec above (issue #366).
        gauge_priority: 120,
        jacobian_callback: None,
        stacked_design: log_sigma_stacked_design,
        stacked_offset: log_sigma_stacked_offset,
    };
    let wigglespec = if let Some(w) = spec.linkwiggle_block.as_ref() {
        Some(ParameterBlockSpec {
            name: "linkwiggle".to_string(),
            design: w.design.clone(),
            offset: Array1::zeros(n),
            penalties: {
                let p_wiggle = w.design.ncols();
                w.penalties
                    .iter()
                    .map(|spec| match spec {
                        crate::solver::estimate::PenaltySpec::Block {
                            local, col_range, ..
                        } => PenaltyMatrix::Blockwise {
                            local: local.clone(),
                            col_range: col_range.clone(),
                            total_dim: p_wiggle,
                        },
                        crate::solver::estimate::PenaltySpec::Dense(m)
                        | crate::solver::estimate::PenaltySpec::DenseWithMean {
                            matrix: m, ..
                        } => PenaltyMatrix::Dense(m.clone()),
                    })
                    .collect()
            },
            nullspace_dims: w.nullspace_dims.clone(),
            initial_log_lambdas: initial_log_lambdas(&w.penalties, w.initial_log_lambdas.clone())?,
            initial_beta: w.initial_beta.clone(),
            // Lowest of the four location-scale blocks: the optional
            // link-wiggle correction yields the shared constant direction to
            // every structural block above it. See the canonical-gauge
            // ownership note on the `time_transform` spec above (issue #366).
            gauge_priority: 80,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        })
    } else {
        None
    };

    // σ-scaled log-t AFT location baseline (issue #892). When the rank-1 reduced
    // parametric-AFT regime fired, the time warp was removed (`h ≡ 0`); the `log t`
    // baseline is instead applied here as a per-row shift of the effective
    // location predictor on the σ-scaled `q` channel: value `−log t` at entry/exit
    // and derivative `−1/t` at exit. Then `u = inv_sigma·(log t − η_t)` and the
    // event Jacobian gains `log_g = −η_ls − log t`, the `−log σ` term that
    // identifies σ. `log_time_*` already carry the `SURVIVAL_TIME_FLOOR` floor, so
    // `1/t_exit = exp(−log t_exit)` is finite and positive.
    let location_log_time = if time_prepared.location_log_time_offset {
        Some(LocationLogTimeOffset {
            value_exit: log_time_exit.mapv(|lt| -lt),
            value_entry: log_time_entry.mapv(|lt| -lt),
            deriv_exit: log_time_exit.mapv(|lt| -((-lt).exp())),
        })
    } else {
        None
    };

    let family = SurvivalLocationScaleFamily {
        n,
        y: spec.event_target.clone(),
        w: spec.weights.clone(),
        inverse_link: spec.inverse_link.clone(),
        derivative_guard: spec.derivative_guard,
        location_log_time,
        x_time_entry: Arc::new(time_prepared.design_entry.clone()),
        x_time_exit: Arc::new(time_prepared.design_exit.clone()),
        x_time_deriv: Arc::new(time_prepared.design_derivative_exit.clone()),
        time_wiggle_knots: spec.timewiggle_block.as_ref().map(|w| w.knots.clone()),
        time_wiggle_degree: spec.timewiggle_block.as_ref().map(|w| w.degree),
        time_wiggle_ncols: protected_timewiggle_cols,
        time_linear_constraints: time_prepared.linear_constraints.clone(),
        x_threshold: threshold_design,
        x_threshold_entry: threshold_entry_design,
        x_threshold_deriv: threshold_deriv_design,
        x_log_sigma: log_sigma_design,
        x_log_sigma_entry: log_sigma_entry_design,
        x_log_sigma_deriv: log_sigma_deriv_design,
        x_link_wiggle: wigglespec.as_ref().map(|s| s.design.clone()),
        wiggle_knots: spec.linkwiggle_block.as_ref().map(|w| w.knots.clone()),
        wiggle_degree: spec.linkwiggle_block.as_ref().map(|w| w.degree),
        policy: crate::resource::ResourcePolicy::default_library(),
    };

    let mut blockspecs = vec![timespec, thresholdspec, log_sigmaspec];
    if let Some(w) = wigglespec {
        blockspecs.push(w);
    }

    Ok(PreparedSurvivalLocationScaleModel {
        family,
        blockspecs,
        time_transform: time_prepared.transform,
        threshold_fixed_cols,
        threshold_full_ncols,
        log_sigma_fixed_cols,
        log_sigma_full_ncols,
        // Time-warp smoothing-parameter count. The reduced constant-scale-AFT
        // time block is unpenalized (`k_time == 0`); the flexible regime keeps
        // one ρ per time penalty. This MUST be the same value the OUTER ρ layout
        // (`fit_survival_location_scale_terms`) computes, otherwise the inner
        // blockwise λ slicing, the outer REML search, and the reduced-parametric
        // dispatch (`is_reduced_parametric_aft`) disagree on whether the time
        // block carries a ρ.
        //
        // Source it from `survival_time_rho_count` — the single source of truth
        // for that decision — evaluated on the SAME un-reduced inputs the outer
        // layout uses (`spec.time_block.penalties`, the original time width, the
        // constant-scale/timewiggle regime). Deriving it here from
        // `time_prepared.penalties.len()` instead made `k_time` depend on whether
        // the inner reduction branch inside `prepare_identified_time_block`
        // happened to fire and clear the projected-to-zero penalties; when that
        // inner collapse did not align with the regime predicate the dispatch saw
        // a stray `k_time == 1` and routed a genuinely unpenalized parametric AFT
        // (#736: constant scale, linear mean, loglogistic) down the coupled
        // exact-joint REML path it cannot certify, instead of the direct MLE
        // bypass. Tying `k_time` to `survival_time_rho_count` makes the inner and
        // outer counts provably identical (same function, same arguments) and the
        // bypass fire exactly when the regime is fully reduced (#736 #735 #721).
        k_time: survival_time_rho_count(
            &spec.time_block.penalties,
            spec.time_block.design_exit.ncols(),
            survival_constant_scale(spec),
            protected_timewiggle_cols,
        ),
        k_threshold: threshold_penalties.len(),
        k_log_sigma: log_sigma_penalties.len(),
        k_wiggle: spec
            .linkwiggle_block
            .as_ref()
            .map_or(0, |w| w.penalties.len()),
    })
}


fn finalize_survival_location_scale_fit(
    prepared: &PreparedSurvivalLocationScaleModel,
    fit: &UnifiedFitResult,
) -> Result<UnifiedFitResult, String> {
    let beta_time_reduced = fit.block_states[SurvivalLocationScaleFamily::BLOCK_TIME]
        .beta
        .clone();
    // Affine lift (issue #892): `β_time_raw = z · β_reduced + affine_shift`. The
    // `affine_shift` is the pinned unit-log-t warp coefficient on the canonical
    // gauge (zero on the non-pin/identity paths, so the lift stays plain linear).
    let beta_time =
        prepared.time_transform.z.dot(&beta_time_reduced) + &prepared.time_transform.affine_shift;
    let beta_threshold_active = fit.block_states[SurvivalLocationScaleFamily::BLOCK_THRESHOLD]
        .beta
        .clone();
    let beta_threshold = expand_leading_fixed_beta(
        &beta_threshold_active,
        prepared.threshold_fixed_cols,
        prepared.threshold_full_ncols,
        "survival location-scale threshold final beta",
    )?;
    let beta_log_sigma_active = fit.block_states[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA]
        .beta
        .clone();
    let beta_log_sigma = expand_leading_fixed_beta(
        &beta_log_sigma_active,
        prepared.log_sigma_fixed_cols,
        prepared.log_sigma_full_ncols,
        "survival location-scale log-sigma final beta",
    )?;
    let beta_link_wiggle = if prepared.family.x_link_wiggle.is_some() {
        Some(
            fit.block_states[SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE]
                .beta
                .clone(),
        )
    } else {
        None
    };
    let lambdas = fit.log_lambdas.mapv(f64::exp);
    let lambdas_time = lambdas.slice(s![0..prepared.k_time]).to_owned();
    let lambdas_threshold = lambdas
        .slice(s![prepared.k_time..prepared.k_time + prepared.k_threshold])
        .to_owned();
    let lambdas_log_sigma = lambdas
        .slice(s![prepared.k_time + prepared.k_threshold
            ..prepared.k_time
                + prepared.k_threshold
                + prepared.k_log_sigma])
        .to_owned();
    let lambdas_linkwiggle = if prepared.k_wiggle > 0 {
        Some(
            lambdas
                .slice(s![
                    prepared.k_time + prepared.k_threshold + prepared.k_log_sigma
                        ..prepared.k_time
                            + prepared.k_threshold
                            + prepared.k_log_sigma
                            + prepared.k_wiggle
                ])
                .to_owned(),
        )
    } else {
        None
    };
    let covariance_conditional = fit
        .covariance_conditional
        .as_ref()
        .map(|cov_reduced| {
            lift_conditional_covariance(
                cov_reduced,
                &prepared.time_transform.z,
                beta_threshold_active.len(),
                beta_threshold.len(),
                prepared.threshold_fixed_cols,
                beta_log_sigma_active.len(),
                beta_log_sigma.len(),
                prepared.log_sigma_fixed_cols,
                beta_link_wiggle.as_ref().map_or(0, |b| b.len()),
            )
        })
        .transpose()?;
    let geometry = fit
        .geometry
        .as_ref()
        .and_then(|geom| {
            if prepared.threshold_fixed_cols > 0 || prepared.log_sigma_fixed_cols > 0 {
                None
            } else {
                Some(
                    lift_conditional_covariance(
                        &geom.penalized_hessian,
                        &prepared.time_transform.z,
                        beta_threshold_active.len(),
                        beta_threshold.len(),
                        prepared.threshold_fixed_cols,
                        beta_log_sigma_active.len(),
                        beta_log_sigma.len(),
                        prepared.log_sigma_fixed_cols,
                        beta_link_wiggle.as_ref().map_or(0, |b| b.len()),
                    )
                    .map(|penalized_hessian| FitGeometry {
                        // Boundary adapter: wrap the lifted raw Hessian as
                        // `UnscaledPrecision` for the newtype storage.
                        penalized_hessian: penalized_hessian.into(),
                        working_weights: geom.working_weights.clone(),
                        working_response: geom.working_response.clone(),
                    }),
                )
            }
        })
        .transpose()?;
    survival_fit_from_parts(SurvivalLocationScaleFitResultParts {
        beta_time,
        beta_threshold,
        beta_log_sigma,
        beta_link_wiggle,
        link_wiggle_knots: prepared.family.wiggle_knots.clone(),
        link_wiggle_degree: prepared.family.wiggle_degree,
        lambdas_time,
        lambdas_threshold,
        lambdas_log_sigma,
        lambdas_linkwiggle,
        log_likelihood: fit.log_likelihood,
        reml_score: fit.reml_score,
        stable_penalty_term: fit.stable_penalty_term,
        penalized_objective: fit.penalized_objective,
        outer_iterations: fit.outer_iterations,
        outer_gradient_norm: fit.outer_gradient_norm,
        outer_converged: fit.outer_converged,
        covariance_conditional,
        geometry,
    })
}


fn validatewiggle_block(
    n: usize,
    b: &LinkWiggleBlockInput,
) -> Result<(), SurvivalLocationScaleError> {
    if b.design.nrows() != n {
        crate::bail_dim_sls!(
            "linkwiggle_block design row mismatch: got {}, expected {n}",
            b.design.nrows()
        );
    }
    let p = b.design.ncols();
    if b.knots.len() < b.degree + 2 {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!(
                "linkwiggle_block knot vector is too short for degree {}: got {} knots",
                b.degree,
                b.knots.len()
            ),
        });
    }
    if let Some(beta0) = &b.initial_beta
        && beta0.len() != p
    {
        crate::bail_dim_sls!(
            "linkwiggle_block initial_beta length mismatch: got {}, expected {p}",
            beta0.len()
        );
    }
    if let Some(beta0) = &b.initial_beta {
        if let Some(beta0_slice) = beta0.as_slice() {
            validate_monotone_wiggle_beta_nonnegative(
                beta0_slice,
                "linkwiggle_block initial_beta",
            )?;
        } else {
            let beta0_values = beta0.iter().copied().collect::<Vec<_>>();
            validate_monotone_wiggle_beta_nonnegative(
                &beta0_values,
                "linkwiggle_block initial_beta",
            )?;
        }
    }
    let k = b.penalties.len();
    if let Some(rho0) = &b.initial_log_lambdas
        && rho0.len() != k
    {
        crate::bail_dim_sls!(
            "linkwiggle_block initial_log_lambdas length mismatch: got {}, expected {k}",
            rho0.len()
        );
    }
    for (idx, s) in b.penalties.iter().enumerate() {
        match s {
            crate::solver::estimate::PenaltySpec::Block {
                local, col_range, ..
            } => {
                if col_range.end > p
                    || local.nrows() != col_range.len()
                    || local.ncols() != col_range.len()
                {
                    crate::bail_dim_sls!(
                        "linkwiggle_block penalty {idx} block shape mismatch: col_range={}..{}, local={}x{}, total_dim={p}",
                        col_range.start,
                        col_range.end,
                        local.nrows(),
                        local.ncols()
                    );
                }
            }
            crate::solver::estimate::PenaltySpec::Dense(m)
            | crate::solver::estimate::PenaltySpec::DenseWithMean { matrix: m, .. } => {
                let (r, c) = m.dim();
                if r != p || c != p {
                    crate::bail_dim_sls!(
                        "linkwiggle_block penalty {idx} must be {p}x{p}, got {r}x{c}"
                    );
                }
            }
        }
    }
    Ok(())
}


fn validate_time_block(
    n: usize,
    b: &TimeBlockInput,
    derivative_guard: f64,
    monotone_time_wiggle_ncols: usize,
) -> Result<(), SurvivalLocationScaleError> {
    if b.design_entry.nrows() != n
        || b.design_exit.nrows() != n
        || b.design_derivative_exit.nrows() != n
        || b.offset_entry.len() != n
        || b.offset_exit.len() != n
        || b.derivative_offset_exit.len() != n
    {
        crate::bail_dim_sls!("time_block input size mismatch");
    }
    let p = b.design_exit.ncols();
    if b.design_entry.ncols() != p || b.design_derivative_exit.ncols() != p {
        crate::bail_dim_sls!("time_block design column mismatch across entry/exit/derivative");
    }
    if !b.time_monotonicity.is_coordinate_cone() {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: format!(
                "time_block requires a coordinate-cone monotonicity strategy by construction; got {:?}",
                b.time_monotonicity
            ),
        });
    }
    structural_time_coefficient_lower_bounds_with_monotone_time_wiggle(
        &b.design_derivative_exit,
        &b.derivative_offset_exit,
        derivative_guard,
        monotone_time_wiggle_ncols,
    )?;
    if let Some(beta0) = &b.initial_beta
        && beta0.len() != p
    {
        crate::bail_dim_sls!(
            "time_block initial_beta length mismatch: got {}, expected {p}",
            beta0.len()
        );
    }
    let k = b.penalties.len();
    if let Some(rho0) = &b.initial_log_lambdas
        && rho0.len() != k
    {
        crate::bail_dim_sls!(
            "time_block initial_log_lambdas length mismatch: got {}, expected {k}",
            rho0.len()
        );
    }
    for (idx, s) in b.penalties.iter().enumerate() {
        let (r, c) = s.dim();
        if r != p || c != p {
            crate::bail_dim_sls!("time_block penalty {idx} must be {p}x{p}, got {r}x{c}");
        }
    }
    Ok(())
}


#[derive(Clone, Debug)]
struct TimeIdentifiabilityTransform {
    /// Maps the inner solver's reduced (active) time coefficients back to the
    /// raw I-spline layout: `β_time_raw = z · β_time_reduced + affine_shift`.
    z: Array2<f64>,
    /// Fixed raw-coefficient contribution folded out of the free design when the
    /// reduced parametric-AFT warp slope is pinned to the canonical unit-log-t
    /// gauge (issue #892). For the non-pin/identity paths this is the zero
    /// vector (length `z.nrows()`), so the lift is the plain linear `z · β`.
    affine_shift: Array1<f64>,
}


#[derive(Clone, Debug)]
struct TimeBlockPrepared {
    design_entry: Array2<f64>,
    design_exit: Array2<f64>,
    design_derivative_exit: Array2<f64>,
    coefficient_lower_bounds: Option<Array1<f64>>,
    linear_constraints: Option<LinearInequalityConstraints>,
    penalties: Vec<Array2<f64>>,
    /// Structural null-space dimension of each (possibly reduced) penalty,
    /// aligned with `penalties`. Carries the reduced count when the block has
    /// been collapsed to its identifiable parametric form so the REML log-det
    /// accounting matches the actual `zᵀ S z` rank rather than the raw basis.
    nullspace_dims: Vec<usize>,
    initial_beta: Option<Array1<f64>>,
    transform: TimeIdentifiabilityTransform,
    /// Augmented geometry offsets the caller must use in place of
    /// `spec.time_block.offset_*`. They equal the input offsets passed through
    /// unchanged on the non-reduce / identity / non-clean-split paths, and the
    /// input offsets plus the folded unit-log-t value/derivative contributions
    /// when the warp slope is pinned (issue #892).
    offset_entry: Array1<f64>,
    offset_exit: Array1<f64>,
    derivative_offset_exit: Array1<f64>,
    /// True iff the unit-log-t warp-slope pin fired (issue #892), so the single
    /// surviving free time column `z_c` is ROW-CONSTANT (it carries the location
    /// level). The threshold block must then DROP its own intercept to avoid a
    /// two-constant alias in the joint Hessian. False on every other path,
    /// including the both-columns-free fallback reduce — there the reduced
    /// I-spline columns are strictly monotone (not row-constant), so the
    /// threshold keeps its intercept.
    pinned_free_row_constant: bool,
    /// True iff the rank-1 reduced parametric-AFT regime fired (issue #892), in
    /// which the time warp is removed entirely (zero free columns, `h ≡ 0`) and
    /// the `log t` baseline is instead applied as a per-row LOCATION offset that
    /// rides the existing σ-scaled `q` channel: `u = inv_sigma·(log t − η_t)`.
    /// The caller threads the exact `−log t` (value at entry/exit) and `−1/t`
    /// (time-derivative at exit) into the family so the standardized residual
    /// carries the canonical survreg/lifelines AFT gauge — the event Jacobian
    /// then contributes `log_g = −η_ls − log t = −log σ − log t`, the `−log σ`
    /// term that identifies σ. On every other path this is `false`.
    location_log_time_offset: bool,
}


fn lower_bound_constraints(lower_bounds: &Array1<f64>) -> Option<LinearInequalityConstraints> {
    LinearInequalityConstraints::from_per_coordinate_lower_bounds(lower_bounds)
}


fn append_linear_constraints(
    first: Option<LinearInequalityConstraints>,
    second: Option<LinearInequalityConstraints>,
) -> Result<Option<LinearInequalityConstraints>, String> {
    match (first, second) {
        (None, None) => Ok(None),
        (Some(constraints), None) | (None, Some(constraints)) => Ok(Some(constraints)),
        (Some(lhs), Some(rhs)) => {
            if lhs.a.ncols() != rhs.a.ncols() {
                return Err(SurvivalLocationScaleError::DimensionMismatch {
                    reason: format!(
                        "time linear constraint width mismatch: left={}, right={}",
                        lhs.a.ncols(),
                        rhs.a.ncols()
                    ),
                }
                .into());
            }
            let rows = lhs.a.nrows() + rhs.a.nrows();
            let cols = lhs.a.ncols();
            let mut a = Array2::<f64>::zeros((rows, cols));
            let mut b = Array1::<f64>::zeros(rows);
            a.slice_mut(s![..lhs.a.nrows(), ..]).assign(&lhs.a);
            a.slice_mut(s![lhs.a.nrows().., ..]).assign(&rhs.a);
            b.slice_mut(s![..lhs.b.len()]).assign(&lhs.b);
            b.slice_mut(s![lhs.b.len()..]).assign(&rhs.b);
            Ok(Some(LinearInequalityConstraints::from_paired(a, b)))
        }
    }
}


fn structural_time_coefficient_lower_bounds(
    design_derivative_exit: &DesignMatrix,
    derivative_offset_exit: &Array1<f64>,
    lower_bound: f64,
) -> Result<Option<Array1<f64>>, String> {
    if design_derivative_exit.nrows() != derivative_offset_exit.len() {
        return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: format!(
            "structural time coefficient bounds require matching rows/offsets: rows={}, offsets={}",
            design_derivative_exit.nrows(),
            derivative_offset_exit.len()
        ) }.into());
    }
    if design_derivative_exit.ncols() == 0 {
        return Ok(None);
    }
    if !lower_bound.is_finite() || lower_bound <= 0.0 {
        return Err(SurvivalLocationScaleError::ConstraintViolation {
            reason: format!(
                "structural time coefficient lower bound must be finite and > 0, got {lower_bound}"
            ),
        }
        .into());
    }

    const DERIVATIVE_TOL: f64 = 1e-12;
    const FEASIBILITY_TOL: f64 = 1e-12;
    // Diagnostics only: entries with magnitude in this open band are reported as
    // "sub-tolerance nonzeros" to explain a missing structural lower bound. The
    // lower edge separates genuine round-off from a hard zero; the upper edge is
    // the derivative-activity tolerance above.
    const SUBTOL_NONZERO_FLOOR: f64 = 1e-30;
    // How many leading columns' max(|·|) to surface in the diagnostic message
    // when no derivative-active column is found.
    const DIAGNOSTIC_COLUMN_PREVIEW: usize = 8;

    let p = design_derivative_exit.ncols();
    let nrows = design_derivative_exit.nrows();
    let mut lower_bounds = Array1::from_elem(p, f64::NEG_INFINITY);
    let mut has_structural_support = false;
    for (row, &offset) in derivative_offset_exit.iter().enumerate() {
        if !offset.is_finite() {
            return Err(SurvivalLocationScaleError::ConstraintViolation { reason: format!(
                "structural time coefficient bounds require finite derivative offsets; found offset[{row}]={offset}"
            ) }.into());
        }
        if lower_bound - offset > FEASIBILITY_TOL {
            return Err(SurvivalLocationScaleError::ConstraintViolation { reason: format!(
                "structural time coefficient bounds require derivative offsets to encode the derivative guard at row {row}: offset={offset:.3e} < guard={lower_bound:.3e}"
            ) }.into());
        }
    }
    // Stream column-by-column so operator-backed (Lazy) designs never have to
    // materialize as a single nrows×ncols dense buffer. `extract_column` is
    // O(n) for dense, O(nnz_j) for sparse, and O(matvec_n) for lazy operators
    // — the operator-form path the strict policy demands.
    let mut col_maxes: Vec<(usize, f64)> = Vec::with_capacity(p.min(DIAGNOSTIC_COLUMN_PREVIEW));
    let mut total_subtol_nonzeros = 0_usize;
    for col in 0..p {
        let column = design_derivative_exit.extract_column(col);
        if column.len() != nrows {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "structural time coefficient bounds: extract_column returned {} entries for column {col}, expected {nrows}",
                column.len()
            ) }.into());
        }
        let mut has_positive_support = false;
        let mut col_max = 0.0_f64;
        for (row, &value) in column.iter().enumerate() {
            if !value.is_finite() {
                return Err(SurvivalLocationScaleError::ConstraintViolation { reason: format!(
                    "structural time coefficient bounds require finite derivative design entries; found row {row}, column {col}"
                ) }.into());
            }
            if value < -DERIVATIVE_TOL {
                return Err(SurvivalLocationScaleError::ConstraintViolation { reason: format!(
                    "structural time coefficient bounds require a non-negative derivative basis at row {row}, column {col}; found {value:.3e}"
                ) }.into());
            }
            if value > DERIVATIVE_TOL {
                has_positive_support = true;
            }
            let abs_value = value.abs();
            if abs_value > col_max {
                col_max = abs_value;
            }
            if abs_value > SUBTOL_NONZERO_FLOOR && abs_value <= DERIVATIVE_TOL {
                total_subtol_nonzeros += 1;
            }
        }
        if has_positive_support {
            lower_bounds[col] = 0.0;
            has_structural_support = true;
        }
        if col < DIAGNOSTIC_COLUMN_PREVIEW {
            col_maxes.push((col, col_max));
        }
    }

    if !has_structural_support {
        // No derivative-active column on this candidate's exit-time design.
        //
        // Two distinct regimes reach this branch and only one of them is
        // surprising:
        //
        // 1. `learn_timewiggle = true` (the large-scale survival
        //    marginal-slope path). `main.rs:3846` deliberately routes to
        //    `SurvivalTimeBasisConfig::None`, which produces an `(n, 0)`
        //    empty time-basis: the parametric baseline plus the timewiggle
        //    block carry the entire time structure, and the exit-time
        //    derivative information lives in `derivative_offset_exit`,
        //    not in any basis column. `prepare_survival_time_stack` then
        //    appends `tw.ncols` **exactly zero** tail columns to keep
        //    shapes aligned with the timewiggle-extended coefficient
        //    vector. Those tail zeros correctly carry no exit-derivative
        //    signal, there is nothing to constrain with a structural
        //    lower-bound ridge, and the right answer is `Ok(None)`.
        //
        // 2. `--time-basis ispline` (or bspline) without timewiggle. Here
        //    the basis was *intended* to span exit-time variation; a
        //    degenerate build that produces only sub-tolerance entries
        //    points at a real numerical bug upstream (knot inference,
        //    cell-moment construction, derivative formula, etc.).
        //
        // The two regimes differentiate by whether the design has any
        // entry whose magnitude exceeds 1e-30 but stays at or below
        // `DERIVATIVE_TOL`. Regime 1 leaves the tail columns at exact
        // zero (no entry passes 1e-30); regime 2 leaves residual
        // float-scale entries from the upstream basis builder. We log
        // warn-level only in the surprising regime.
        if total_subtol_nonzeros > 0 {
            log::warn!(
                "structural time coefficient bounds: no derivative-active column on this candidate's exit-time design ({} rows × {} cols, sub-tolerance nonzero entries ({:.0e} < |v| ≤ {:.0e}): {}, first-{} col max(|.|): {:?}); skipping the structural lower-bound ridge — fit may converge to a non-monotone-in-time hazard",
                nrows,
                p,
                SUBTOL_NONZERO_FLOOR,
                DERIVATIVE_TOL,
                total_subtol_nonzeros,
                DIAGNOSTIC_COLUMN_PREVIEW,
                col_maxes,
            );
        }
        return Ok(None);
    }
    Ok(Some(lower_bounds))
}


fn structural_time_coefficient_lower_bounds_with_monotone_time_wiggle(
    design_derivative_exit: &DesignMatrix,
    derivative_offset_exit: &Array1<f64>,
    lower_bound: f64,
    monotone_time_wiggle_ncols: usize,
) -> Result<Option<Array1<f64>>, String> {
    let mut lower_bounds = structural_time_coefficient_lower_bounds(
        design_derivative_exit,
        derivative_offset_exit,
        lower_bound,
    )?;
    let Some(bounds) = lower_bounds.as_mut() else {
        return Ok(None);
    };
    if monotone_time_wiggle_ncols == 0 {
        return Ok(lower_bounds);
    }
    if monotone_time_wiggle_ncols > bounds.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "structural time coefficient bounds cannot reserve {monotone_time_wiggle_ncols} monotone wiggle columns from {} coefficients",
            bounds.len()
        ) }.into());
    }

    // Time wiggle columns are appended as zero-derivative tail columns in the
    // linear time block, but they re-enter through a monotone I-spline
    // composition h = h_base + I(h_base) @ beta_w. For that composition,
    // beta_w >= 0 implies dq/dh_base = 1 + I'(h_base) @ beta_w >= 1 because
    // I' is an M-spline and therefore non-negative. The sign of the baseline
    // hazard trend does not change this requirement: negative beta_w is the
    // wrong monotonicity direction for the time wiggle in every case.
    let tail_start = bounds.len() - monotone_time_wiggle_ncols;
    for col in tail_start..bounds.len() {
        if !bounds[col].is_finite() || bounds[col] < 0.0 {
            bounds[col] = 0.0;
        }
    }
    Ok(lower_bounds)
}


/// Project `beta0` (or the origin when `beta0` is `None`) onto the feasible
/// polytope `{x : A x >= b}` via cyclic Dykstra projections.
///
/// The geometry only makes sense when every operand lives in the same
/// `dim`-dimensional space, so the function validates its three independent
/// dimensions up front rather than letting an ndarray broadcast mismatch
/// `unwrap()` into a process-wide panic (see issue #374: a stale, lower-
/// dimensional warm-start hint reached this projection with `beta0.len() !=
/// dim` and the `&beta + &corrections.row(i)` add panicked with
/// `IncompatibleShape`). A length mismatch is a caller contract violation,
/// so it is surfaced as a structured `Result::Err` that the marginal-slope /
/// location-scale pipelines turn into a clean `GamError` instead of a panic
/// crossing the Rust/Python boundary.
pub fn project_onto_linear_constraints(
    dim: usize,
    constraints: &LinearInequalityConstraints,
    beta0: Option<&Array1<f64>>,
) -> Result<Array1<f64>, String> {
    if let Some(b0) = beta0
        && b0.len() != dim
    {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "project_onto_linear_constraints: beta0 length {} does not match dim {dim}",
                b0.len()
            ),
        }
        .into());
    }
    if constraints.a.nrows() != constraints.b.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "project_onto_linear_constraints: constraint A has {} rows but b has length {}",
                constraints.a.nrows(),
                constraints.b.len()
            ),
        }
        .into());
    }
    let mut beta = beta0.cloned().unwrap_or_else(|| Array1::zeros(dim));
    if constraints.a.ncols() != dim || constraints.a.nrows() == 0 {
        return Ok(beta);
    }
    let mut corrections = Array2::<f64>::zeros((constraints.a.nrows(), dim));
    for _ in 0..DYKSTRA_PROJECTION_MAX_SWEEPS {
        let mut max_violation = 0.0_f64;
        for i in 0..constraints.a.nrows() {
            let row = constraints.a.row(i);
            let row_norm_sq = row.dot(&row);
            if row_norm_sq <= DYKSTRA_ROW_DEGENERACY_FLOOR {
                continue;
            }
            let y = &beta + &corrections.row(i);
            let slack = row.dot(&y) - constraints.b[i];
            max_violation = max_violation.max((-slack).max(0.0));
            if slack >= 0.0 {
                corrections.row_mut(i).assign(&(&y - &beta));
                continue;
            }
            let step = (constraints.b[i] - row.dot(&y)) / row_norm_sq;
            let projected = &y + &(row.to_owned() * step);
            corrections.row_mut(i).assign(&(&y - &projected));
            beta.assign(&projected);
        }
        if max_violation <= DYKSTRA_PROJECTION_TOL {
            break;
        }
    }
    Ok(beta)
}


fn validate_linear_constraints(
    label: &str,
    beta: &Array1<f64>,
    constraints: &LinearInequalityConstraints,
) -> Result<(), String> {
    if beta.len() != constraints.a.ncols() {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "survival location-scale {label} constraint dimension mismatch: beta={}, constraints={}",
            beta.len(),
            constraints.a.ncols()
        ) }.into());
    }
    if constraints.a.nrows() != constraints.b.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "survival location-scale {label} constraint row mismatch: A rows={}, b len={}",
                constraints.a.nrows(),
                constraints.b.len()
            ),
        }
        .into());
    }

    let mut worst_row = None;
    let mut worst_slack = 0.0_f64;
    let mut worst_tol = 0.0_f64;
    for row in 0..constraints.a.nrows() {
        let a_row = constraints.a.row(row);
        let slack = a_row.dot(beta) - constraints.b[row];
        let scale = a_row
            .iter()
            .zip(beta.iter())
            .map(|(a, b)| (a * b).abs())
            .sum::<f64>()
            .max(constraints.b[row].abs())
            .max(1.0);
        let tol = CONSTRAINT_NONNEGATIVITY_REL_TOL * scale;
        if slack < -tol && (worst_row.is_none() || slack < worst_slack) {
            worst_row = Some(row);
            worst_slack = slack;
            worst_tol = tol;
        }
    }
    if let Some(row) = worst_row {
        return Err(SurvivalLocationScaleError::ConstraintViolation { reason: format!(
            "survival location-scale {label} violates represented linear constraint at row {row}: slack={worst_slack:.3e}, tol={worst_tol:.3e}"
        ) }.into());
    }
    Ok(())
}


/// Orthonormal basis `z` (raw `p` × reduced `r`) of the penalty null space —
/// the affine `{1, log t}` AFT baseline an I-spline 2nd-order difference penalty
/// leaves unpenalized. The penalized (curvature) directions are exactly the
/// non-affine deviation the constant-scale data cannot identify, so the
/// null-space columns are precisely the identifiable parametric subspace.
///
/// The basis is the eigenvectors of the summed time penalty whose eigenvalues
/// sit at the bottom of the spectrum (geometric kernel). `r` is read off the
/// eigenvalue gap with the same relative threshold the I-spline builder uses to
/// report `nullspace_dims`, so this is the structural null-space dimension
/// rather than a hard-coded `2`.
fn time_parametric_null_space_basis(penalties: &[Array2<f64>], p: usize) -> Option<Array2<f64>> {
    if p == 0 || penalties.is_empty() {
        return None;
    }
    let mut total = Array2::<f64>::zeros((p, p));
    for s_mat in penalties {
        if s_mat.nrows() != p || s_mat.ncols() != p {
            return None;
        }
        total += s_mat;
    }
    let (evals, evecs) = total.eigh(faer::Side::Lower).ok()?;
    let max_ev = evals
        .iter()
        .copied()
        .fold(0.0_f64, |a, b| a.max(b.abs()))
        .max(1.0);
    // Mirror the I-spline builder's null-space threshold (survival_construction).
    let threshold = 100.0 * (p as f64) * f64::EPSILON * max_ev;
    // `eigh` returns ascending eigenvalues with eigenvectors as the matching
    // columns of `evecs`; the kernel is the leading low-eigenvalue block.
    let null_cols: Vec<usize> = evals
        .iter()
        .enumerate()
        .filter(|&(_, &e)| e <= threshold)
        .map(|(idx, _)| idx)
        .collect();
    if null_cols.is_empty() || null_cols.len() >= p {
        // No surplus flexibility to remove (or a degenerate all-null penalty):
        // there is nothing to reduce, keep the full basis.
        return None;
    }
    Some(evecs.select(ndarray::Axis(1), &null_cols))
}


/// Does the constant-scale-AFT regime actually reduce the time block to its
/// unpenalized affine parametric null space?
///
/// This is the single predicate that both the inner block preparation
/// (`prepare_identified_time_block`) and the OUTER ρ layout
/// (`SurvivalLambdaLayout`) consult so they agree on the reduced time block's
/// smoothing-parameter count. The reduction fires only when the regime is
/// constant-scale with no monotone timewiggle reintroducing flexibility AND the
/// time penalty actually has an affine null space to collapse onto. When it
/// fires the reduced block is genuinely unpenalized (`zᵀ S z ≈ 0` on the
/// null space), so it carries ZERO smoothing parameters and must contribute no
/// ρ coordinate to the outer REML search — exactly like the constant `log_sigma`
/// and rigid `threshold` blocks (issue #736/#735/#721).
fn time_block_reduces_to_parametric(
    time_penalties: &[Array2<f64>],
    time_ncols: usize,
    constant_scale: bool,
    protected_timewiggle_cols: usize,
) -> bool {
    constant_scale
        && protected_timewiggle_cols == 0
        && time_parametric_null_space_basis(time_penalties, time_ncols).is_some()
}


/// Number of time-warp smoothing parameters (outer ρ coordinates) the survival
/// location-scale model exposes. The flexible regime keeps one ρ per time
/// penalty; the reduced constant-scale-AFT regime drops them all because the
/// affine parametric block it collapses to is unpenalized.
fn survival_time_rho_count(
    time_penalties: &[Array2<f64>],
    time_ncols: usize,
    constant_scale: bool,
    protected_timewiggle_cols: usize,
) -> usize {
    if time_block_reduces_to_parametric(
        time_penalties,
        time_ncols,
        constant_scale,
        protected_timewiggle_cols,
    ) {
        0
    } else {
        time_penalties.len()
    }
}


/// Whether this fit is the reduced, fully PARAMETRIC constant-scale AFT regime,
/// in which the location and scale carry no genuine smoothing — only (at most)
/// full-rank parametric shrinkage ridges (`nullspace_dim == 0`, e.g. the
/// linear-term `LinearTermRidge` gam places on a non-intercept covariate such as
/// `age` in `Surv(time,event) ~ age`) — and so must be fit as a plain
/// few-parameter AFT MLE (loglogistic / lognormal), exactly like
/// `survreg`/`lifelines` (issue #736/#735/#721).
///
/// The conditions are:
///   * the time-warp has collapsed to its identifiable affine null space
///     (`survival_time_rho_count == 0`), i.e. constant scale with no protected
///     timewiggle;
///   * there is no link-wiggle and no monotone time-wiggle;
///   * every threshold and every log-σ penalty is a full-rank parametric ridge
///     (`nullspace_dim == 0`) — NEVER a wiggliness/smoothing penalty, whose
///     structural null space (the unpenalized polynomial/affine subspace) is
///     always nonzero.
///
/// In this regime those parametric ridges are dropped from BOTH the inner
/// prepared model and the outer ρ layout (mirroring how the reduced time block
/// drops its projected-to-zero penalties): the affine time-warp plus the
/// location intercept already identify the location/scale, the ridge has no
/// smoothing parameter worth a vacuous outer ρ coordinate (its default λ would
/// merely bias the parametric coefficient away from the `survreg`/`lifelines`
/// MLE), so the fit becomes an unpenalized direct parametric-AFT Newton MLE
/// (`fit_parametric_aft_direct_mle`) with zero outer coordinates — converging in
/// milliseconds instead of stalling the coupled exact-joint REML optimizer on a
/// flat, vacuous ρ surface.
///
/// Certification is conservative: if either block's `nullspace_dims` metadata is
/// absent or length-mismatched (so a null space cannot be certified zero) the
/// regime is NOT recognized and the fit stays on the full coupled path. A
/// genuinely flexible fit (smooth mean `~ s(z)`, smooth scale
/// `noise_formula = s(...)`, a link-wiggle, an active timewiggle, or a varying
/// scale) carries a wiggliness penalty with `nullspace_dim > 0` or a surviving
/// time ρ, so it never matches here.
fn survival_reduced_parametric_aft_regime(
    time_penalties: &[Array2<f64>],
    time_ncols: usize,
    constant_scale: bool,
    protected_timewiggle_cols: usize,
    threshold_nullspace_dims: &[usize],
    threshold_npenalties: usize,
    log_sigma_nullspace_dims: &[usize],
    log_sigma_npenalties: usize,
    has_linkwiggle: bool,
) -> bool {
    if has_linkwiggle || protected_timewiggle_cols > 0 {
        return false;
    }
    if survival_time_rho_count(
        time_penalties,
        time_ncols,
        constant_scale,
        protected_timewiggle_cols,
    ) != 0
    {
        return false;
    }
    block_penalties_all_parametric_ridges(threshold_nullspace_dims, threshold_npenalties)
        && block_penalties_all_parametric_ridges(log_sigma_nullspace_dims, log_sigma_npenalties)
}


/// True iff a block's `npenalties` penalties are all full-rank parametric ridges
/// — every certified structural null-space dimension is zero. A block with no
/// penalties trivially passes. When the `nullspace_dims` metadata is absent or
/// length-mismatched the null space cannot be certified, so this conservatively
/// returns `false` (treat as potentially smoothing).
fn block_penalties_all_parametric_ridges(nullspace_dims: &[usize], npenalties: usize) -> bool {
    if npenalties == 0 {
        return true;
    }
    nullspace_dims.len() == npenalties && nullspace_dims.iter().all(|&d| d == 0)
}


/// Data-scale slope of `design_exit · direction` regressed on `log_time_exit`
/// (centered). `Some(s)` where `s = Σ(logt_c · y) / Σ(logt_c²)`; `None` when
/// `log t` has no spread (all exit times equal) or the time direction is flat
/// (zero slope), so the unit-log-t normalization would be undefined. Dividing the
/// raw direction by `s` yields a design image with unit slope vs log t — the
/// canonical survreg/lifelines AFT gauge.
fn unit_log_time_slope(
    design_exit: &Array2<f64>,
    direction: &Array1<f64>,
    log_time_exit: ndarray::ArrayView1<f64>,
) -> Option<f64> {
    let n = design_exit.nrows();
    if n == 0 || log_time_exit.len() != n {
        return None;
    }
    let y = design_exit.dot(direction);
    let log_mean = log_time_exit.sum() / n as f64;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    for i in 0..n {
        let xc = log_time_exit[i] - log_mean;
        sxx += xc * xc;
        sxy += xc * y[i];
    }
    let y_scale = y.iter().fold(0.0_f64, |acc, &v| acc.max(v.abs())).max(1.0);
    if !sxx.is_finite() || sxx <= f64::EPSILON {
        return None;
    }
    let slope = sxy / sxx;
    if !slope.is_finite() || slope.abs() <= f64::EPSILON * y_scale {
        return None;
    }
    Some(slope)
}


/// Does the rank-1 reduced parametric-AFT regime apply (issue #892)?
///
/// The real survival time penalty is a 1st-difference penalty, so its null space
/// is DIMENSION 1: a single monotone log-t trend column `z` (p×1). When it does,
/// the time warp is REMOVED entirely (`h ≡ 0`) and the `log t` baseline is
/// carried as a per-row σ-scaled LOCATION offset instead — `u = inv_sigma·(log t
/// − η_t) = (log t − μ)/σ` — so the event Jacobian gains the `−log σ` term that
/// identifies σ (the survreg / lifelines / flexsurv AFT gauge).
///
/// This predicate only certifies the regime is genuinely log-t parametric: the
/// null space is rank-1, sized to the basis, and the single null-space direction
/// has a usable data-scale slope versus log t (so `log t` actually varies and the
/// floored times are finite). It does NOT build any design — the warp is gone and
/// the log-t baseline rides the location channel, threaded from the caller. When
/// it returns `false` the caller falls through to the prior both-columns-free
/// reduce. `log t` values come from the same floor (`SURVIVAL_TIME_FLOOR`) as
/// `checked_log_survival_times`.
fn rank1_reduced_time_warp_applies(
    z: &Array2<f64>,
    design_exit: &Array2<f64>,
    log_time_entry: ndarray::ArrayView1<f64>,
    log_time_exit: ndarray::ArrayView1<f64>,
) -> bool {
    if z.ncols() != 1 {
        return false;
    }
    let n = design_exit.nrows();
    if log_time_entry.len() != n || log_time_exit.len() != n {
        return false;
    }
    if log_time_entry.iter().any(|v| !v.is_finite()) || log_time_exit.iter().any(|v| !v.is_finite())
    {
        return false;
    }
    // The null-space direction must have a usable data-scale slope versus log t —
    // i.e. `log t` genuinely varies across the sample — else the σ-scaled log-t
    // gauge is degenerate and the prior reduce should handle it instead.
    let z_dir = z.column(0).to_owned();
    unit_log_time_slope(design_exit, &z_dir, log_time_exit).is_some()
}


/// Result of pinning the reduced parametric-AFT time-warp slope to the canonical
/// unit-log-t gauge (issue #892). `z_c` (p×1) is the kept-free row-constant
/// direction; `z_t` (p-vector) is the pinned unit-log-t direction folded into
/// the geometry offsets; the three `reduced_*` matrices (n×1) are the free design
/// `X · z_c` for entry / exit / derivative-exit.
struct PinnedTimeWarp {
    z_c: Array2<f64>,
    z_t: Array1<f64>,
    reduced_entry: Array2<f64>,
    reduced_exit: Array2<f64>,
    reduced_derivative_exit: Array2<f64>,
}


/// Split the 2-D affine null-space basis `z` (p×2, orthonormal columns) into the
/// row-constant location direction `z_c` (kept free) and the time-varying warp
/// direction `z_t`, normalized so `design_exit · z_t` has unit data-scale slope
/// versus `log t_exit` (the canonical survreg/lifelines AFT gauge). Returns
/// `None` when the split is not clean — no usable row-constant direction
/// (`‖z_c_raw‖` tiny) or a degenerate log-t slope (`|s|` tiny) — so the caller
/// can fall back to the prior both-columns-free reduce behavior without
/// regressing the non-pin case.
fn pin_reduced_time_warp_slope(
    z: &Array2<f64>,
    design_entry: &Array2<f64>,
    design_exit: &Array2<f64>,
    design_derivative_exit: &Array2<f64>,
    log_time_exit: ndarray::ArrayView1<f64>,
) -> Option<PinnedTimeWarp> {
    let p = z.nrows();
    let n = design_exit.nrows();
    if z.ncols() != 2 || n == 0 || log_time_exit.len() != n {
        return None;
    }
    // `G = design_exit · z` (n×2). The row-constant direction `a` minimizes
    // ‖G a − 1_n‖² (least-squares constant fit): a = (GᵀG)⁻¹ Gᵀ 1_n. Solving the
    // 2×2 normal equations in closed form keeps the helper self-contained.
    let g = design_exit.dot(z);
    let m00 = g.column(0).dot(&g.column(0));
    let m01 = g.column(0).dot(&g.column(1));
    let m11 = g.column(1).dot(&g.column(1));
    let ones = Array1::<f64>::ones(n);
    let r0 = g.column(0).dot(&ones);
    let r1 = g.column(1).dot(&ones);
    let det = m00 * m11 - m01 * m01;
    // Scale-relative singularity guard for the 2×2 GramGram: a degenerate G has
    // no distinct constant/time split to exploit.
    let gram_scale = m00.max(m11).max(1.0);
    if !det.is_finite() || det.abs() <= f64::EPSILON * gram_scale * gram_scale {
        return None;
    }
    let a0 = (m11 * r0 - m01 * r1) / det;
    let a1 = (m00 * r1 - m01 * r0) / det;
    // `z_c_raw = z · a` (p-vector): the raw-coefficient direction whose design
    // image `G a` is the best row-constant column. Normalize to a unit-norm
    // basis vector so the reduced free coefficient is well scaled.
    let z_c_raw = z.dot(&Array1::from(vec![a0, a1]));
    let z_c_norm = z_c_raw.dot(&z_c_raw).sqrt();
    if !z_c_norm.is_finite() || z_c_norm <= f64::EPSILON * (p as f64).sqrt() {
        return None;
    }
    let z_c_vec = &z_c_raw / z_c_norm;
    // Time-varying direction: the in-span(z) complement of `a` in the 2-D
    // coefficient plane. With `a = [a0, a1]`, `a_perp = [-a1, a0]` is orthogonal
    // to `a`, so `z_t_raw = z · a_perp` is the part of span(z) carrying the
    // non-constant (log-t trend) warp.
    let a_perp = Array1::from(vec![-a1, a0]);
    let z_t_raw = z.dot(&a_perp);
    // Normalize `z_t` to unit data-scale slope vs log t (the canonical
    // unit-log-t AFT gauge): `design_exit · z_t` rises by exactly 1 per unit of
    // log t, so the derivative design reproduces u'(t) = 1/t.
    let slope = unit_log_time_slope(design_exit, &z_t_raw, log_time_exit)?;
    let z_t = &z_t_raw / slope;
    // p×1 free design columns and the kept-free basis matrix.
    let z_c = z_c_vec.insert_axis(ndarray::Axis(1));
    let reduced_entry = design_entry.dot(&z_c);
    let reduced_exit = design_exit.dot(&z_c);
    let reduced_derivative_exit = design_derivative_exit.dot(&z_c);
    Some(PinnedTimeWarp {
        z_c,
        z_t,
        reduced_entry,
        reduced_exit,
        reduced_derivative_exit,
    })
}


/// Build the reduced time block for the canonical σ-scaled log-t AFT gauge
/// (issue #892): the time warp is REMOVED entirely (`h ≡ 0`, zero free columns,
/// no penalties, no monotonicity constraint) and the `log t` baseline is carried
/// as a per-row LOCATION offset on the σ-scaled `q` channel
/// (`location_log_time_offset = true`). This is the clean parametric-AFT
/// representation `u = (log t − μ)/σ` that `survreg`/`lifelines` fit: it has no
/// free time coefficient (so no `1e7` cold-start gradient, no ill-conditioned
/// time curvature), no derivative-guard inequality (so the joint Newton step is
/// never frozen by a binding time row), and no time-warp constant aliased with
/// the threshold intercept.
fn location_logt_offset_time_block(
    design_entry: &Array2<f64>,
    design_exit: &Array2<f64>,
    design_derivative_exit: &Array2<f64>,
    p: usize,
) -> TimeBlockPrepared {
    TimeBlockPrepared {
        design_entry: Array2::<f64>::zeros((design_entry.nrows(), 0)),
        design_exit: Array2::<f64>::zeros((design_exit.nrows(), 0)),
        design_derivative_exit: Array2::<f64>::zeros((design_derivative_exit.nrows(), 0)),
        coefficient_lower_bounds: None,
        // No free time coefficients → no derivative-guard constraint. The warp
        // derivative is `h′ ≡ 0`; monotonicity holds because the q channel
        // contributes `qdot = inv_sigma/t > 0` to `g`.
        linear_constraints: None,
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_beta: Some(Array1::<f64>::zeros(0)),
        transform: TimeIdentifiabilityTransform {
            z: Array2::<f64>::zeros((p, 0)),
            affine_shift: Array1::zeros(p),
        },
        offset_entry: Array1::zeros(design_entry.nrows()),
        offset_exit: Array1::zeros(design_exit.nrows()),
        derivative_offset_exit: Array1::zeros(design_derivative_exit.nrows()),
        pinned_free_row_constant: false,
        location_log_time_offset: true,
    }
}


/// Does the time-penalty null space `z` (raw `p` × reduced `r`) genuinely carry
/// the `log t` AFT baseline, so collapsing the whole warp to the σ-scaled log-t
/// LOCATION offset (`location_logt_offset_time_block`) is an EXACT
/// reparameterisation of a parametric constant-scale AFT?
///
/// The rank-1 gauge (`rank1_reduced_time_warp_applies`) answers this for the
/// single-column null space by checking that direction's image has a usable
/// data-scale log-t slope. This generalises it to any rank: least-squares
/// project `log t` onto the null-space warp image `G = design_exit · z` and ask
/// whether the best-fitting null direction `z · c` has a usable log-t slope. When
/// it does, `{log t}` lies in the span the warp would otherwise fit freely, so
/// the higher null directions are spurious parametric-AFT flexibility and the
/// log-t offset captures the baseline exactly. A degenerate or non-log-t basis
/// fails the check and the caller keeps the (constrained) free-warp fallback.
fn reduced_warp_logt_baseline_usable(
    z: &Array2<f64>,
    design_exit: &Array2<f64>,
    log_time_exit: ndarray::ArrayView1<f64>,
) -> bool {
    use crate::faer_ndarray::FaerCholesky;
    let n = design_exit.nrows();
    let r = z.ncols();
    if n == 0 || r == 0 || log_time_exit.len() != n {
        return false;
    }
    let g = design_exit.dot(z); // (n, r) warp images of the null directions
    let gtg = g.t().dot(&g); // (r, r)
    let gtl = g.t().dot(&log_time_exit); // (r,)
    // Tiny Levenberg ridge so a rank-deficient G (some null directions producing
    // a (near-)zero warp image) still yields a well-posed projection rather than
    // a Cholesky failure that would spuriously reject the collapse.
    let scale = gtg
        .diag()
        .iter()
        .fold(0.0_f64, |acc, &v| acc.max(v.abs()))
        .max(1.0);
    let mut ridged = gtg;
    for i in 0..r {
        ridged[[i, i]] += 1e-10 * scale;
    }
    let Ok(chol) = ridged.cholesky(faer::Side::Lower) else {
        return false;
    };
    let c = chol.solvevec(&gtl);
    if c.iter().any(|v| !v.is_finite()) {
        return false;
    }
    let direction = z.dot(&c);
    unit_log_time_slope(design_exit, &direction, log_time_exit).is_some()
}


fn prepare_identified_time_block(
    input: &TimeBlockInput,
    derivative_guard: f64,
    monotone_time_wiggle_ncols: usize,
    reduce_to_parametric: bool,
    log_time_entry: ndarray::ArrayView1<f64>,
    log_time_exit: ndarray::ArrayView1<f64>,
) -> Result<TimeBlockPrepared, String> {
    let p = input.design_exit.ncols();
    if !input.time_monotonicity.is_coordinate_cone() {
        return Err(SurvivalLocationScaleError::InvalidConfiguration { reason: format!(
            "time_block requires a coordinate-cone monotonicity strategy by construction; got {:?}",
            input.time_monotonicity
        ) }.into());
    }
    // Materialize to dense at the location-scale boundary — the hot path
    // uses dense matrix operations (scale_dense_rows, weighted_crossprod_dense).
    let design_entry = input.design_entry.to_dense();
    let design_exit = input.design_exit.to_dense();
    let design_derivative_exit = input.design_derivative_exit.to_dense();

    // Constant-scale AFT regime: reduce the unidentified flexible I-spline
    // time-warp to its identifiable affine parametric form before any
    // constraint is generated (issue #736/#735/#721).
    //
    // `z` (raw `p` × reduced `r`) is an orthonormal basis of the time penalty's
    // null space — the affine `1 + log t` AFT baseline. Projecting the three
    // exit/entry/derivative designs onto `X z` leaves a clean `r`-column
    // parametric block whose every direction is data-identified, so the coupled
    // exact-joint inner solve has no free penalized-stationarity null space to
    // choke on. The penalty `zᵀ S z` collapses to (numerically) zero on the
    // null space — the correct, unpenalized parametric block, exactly like the
    // constant-scale `log_sigma` block — and the regime's pinned `ρ = 10` time
    // seed (`build_survival_two_block_exact_joint_setup`) certifies its flat
    // box-bound KKT immediately rather than crawling the old 12-column ridge.
    //
    // Monotonicity is preserved structurally: the per-row derivative guard
    // `(X' z) β_r + offset ≥ guard` is built from the *reduced* derivative
    // design, so `h(t)` stays non-decreasing pointwise at every observed time.
    // (The coordinate-cone per-column `γ ≥ 0` lower bounds are dropped — they
    // are a property of individual I-spline columns, not of the affine
    // generators that span their null space — and the row-wise guard takes over
    // that role exactly.)
    if reduce_to_parametric && let Some(z) = time_parametric_null_space_basis(&input.penalties, p) {
        let r = z.ncols();
        // Canonical log-t gauge (issue #892). In the reduced constant-scale
        // parametric-AFT regime the I-spline time-warp collapses onto its log-t
        // affine null space (the basis is over `log t`, survival_construction.rs).
        // The full multi-column I-spline warp carries a numerically unidentified
        // ridge, so the unconstrained direct-MLE Newton picks an arbitrary scale
        // and miscalibrates the absolute survival curve.
        //
        // RANK-1 case (the one that actually fires for real fits): the survival
        // time penalty is a 1st-difference penalty, so its null space is
        // DIMENSION 1 — a single monotone log-t trend column. Pin the warp SHAPE
        // to exactly `log t` (built straight from the event times, NOT the
        // I-spline's curved image of it) but keep its SCALE `θ` a single FREE
        // coefficient: `h(t) = θ · log t`. The standardized residual is
        // `u = h − η_loc/σ` with the warp UN-scaled by σ, so a lognormal/loglogistic
        // AFT `(log t − μ)/σ` needs the warp to carry slope `1/σ` versus log t;
        // the MLE drives `θ → 1/σ` and σ recovers to truth (folding `θ ≡ 1` instead
        // would lock the residual log-t slope at 1 and over-determine σ). `θ` is
        // identified — no flat ridge — by the event Jacobian's `log|h′| = log θ −
        // log t` term, so the collapse to one log-t column is well posed. The
        // single free column is the (non-constant) log-t warp, so the threshold
        // keeps its intercept and `pinned_free_row_constant` stays false.
        if r == 1
            && z.nrows() == p
            && rank1_reduced_time_warp_applies(&z, &design_exit, log_time_entry, log_time_exit)
        {
            // σ-scaled log-t AFT gauge (issue #892). REMOVE the time warp entirely
            // (`h ≡ 0`, zero free time columns) and instead carry the `log t`
            // baseline as a per-row LOCATION offset on the existing σ-scaled `q`
            // channel: `u = inv_sigma·(log t − η_t) = (log t − μ)/σ`. The caller
            // threads the exact `−log t` value (entry/exit) and `−1/t` derivative
            // into the family (`location_log_time_offset = true` below), so the
            // standardized residual matches survreg/lifelines and the event
            // Jacobian gains `log_g = −η_ls − log t = −log σ − log t` — the `−log σ`
            // term that IDENTIFIES σ.
            //
            // Why not a free warp scale θ (the prior attempt): with the warp
            // un-scaled by σ the pair `(η_t, σ)` co-scaled freely (only `η_t/σ`
            // identified, no `−log σ` term), so every parameter shrank by a common
            // factor. Routing `log t` through the σ-scaled `q` channel supplies the
            // missing `−log σ` Jacobian and pins σ. The warp is gone, so the time
            // block is empty (no free columns, no penalties, no constraints); all
            // the σ-coupling rides the existing `q`-derivative/Hessian stack, with
            // no new time×log_sigma cross-terms.
            return Ok(location_logt_offset_time_block(
                &design_entry,
                &design_exit,
                &design_derivative_exit,
                p,
            ));
        }
        // RANK-2 case (2nd-difference penalty `{1, log t}`): kept for correctness
        // where it occurs (golden unit test), though real fits use rank-1 above.
        if r == 2
            && z.nrows() == p
            && let Some(pinned) = pin_reduced_time_warp_slope(
                &z,
                &design_entry,
                &design_exit,
                &design_derivative_exit,
                log_time_exit,
            )
        {
            let PinnedTimeWarp {
                z_c,
                z_t,
                reduced_entry,
                reduced_exit,
                reduced_derivative_exit,
            } = pinned;
            // Augmented offsets carry the pinned unit-log-t warp out of the free
            // design. `design_* · z_t` is the fixed value/derivative
            // contribution of the unit-slope `log t` direction.
            let offset_entry = &input.offset_entry + &design_entry.dot(&z_t);
            let offset_exit = &input.offset_exit + &design_exit.dot(&z_t);
            let derivative_offset_exit =
                &input.derivative_offset_exit + &design_derivative_exit.dot(&z_t);
            let reduced_derivative_design =
                DesignMatrix::Dense(DenseDesignMatrix::from(reduced_derivative_exit.clone()));
            // Pointwise monotonicity uses the AUGMENTED derivative offset so the
            // guard `(X' z_c) β_c + offset' ≥ guard` accounts for the pinned
            // warp's own (positive, unit-log-t) derivative.
            let linear_constraints = time_derivative_guard_constraints(
                &reduced_derivative_design,
                &derivative_offset_exit,
                derivative_guard,
            )?;
            // Project the caller seed onto the single free constant direction.
            // The pinned warp lives entirely in the offset, so the reduced seed
            // only needs the `z_c` component.
            let initial_beta = match (linear_constraints.as_ref(), input.initial_beta.as_ref()) {
                (Some(constraints), Some(beta0)) => Some(project_onto_linear_constraints(
                    1,
                    constraints,
                    Some(&z_c.t().dot(beta0)),
                )?),
                (_, Some(beta0)) => Some(z_c.t().dot(beta0)),
                _ => None,
            };
            return Ok(TimeBlockPrepared {
                design_entry: reduced_entry,
                design_exit: reduced_exit,
                design_derivative_exit: reduced_derivative_exit,
                coefficient_lower_bounds: None,
                linear_constraints,
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_beta,
                transform: TimeIdentifiabilityTransform {
                    z: z_c,
                    affine_shift: z_t,
                },
                offset_entry,
                offset_exit,
                derivative_offset_exit,
                // Pin fired: the free `z_c` column is row-constant, so the
                // threshold intercept must be dropped to break the alias (#892).
                pinned_free_row_constant: true,
                // Rank-2 path keeps a free warp; no location-channel log-t offset.
                location_log_time_offset: false,
            });
        }
        // GENERAL case (r ≥ 3, or r ∈ {1,2} where the clean rank-1 / rank-2 pin
        // gauges did not match): for a fully PARAMETRIC constant-scale AFT the
        // baseline transform is exactly `log t`, so the I-spline null space's
        // higher affine directions are spurious flexibility. When the null space
        // genuinely carries the `log t` baseline, collapse the ENTIRE warp to the
        // canonical σ-scaled log-t LOCATION offset — the same clean representation
        // the rank-1 gauge uses — instead of keeping a free monotone warp with a
        // derivative-guard inequality.
        //
        // The prior free-warp fallback (kept below as a last resort) made the
        // reduced time block carry `r` free columns AND a per-row monotonicity
        // constraint. For this regime that cold-started (β = 0) at a degenerate
        // warp with a ~1e7 gradient and a derivative-guard row active at the
        // boundary; the direct-MLE joint Newton's single global step length was
        // then capped to 0 by that one binding time row, FREEZING every block —
        // including the fully unconstrained location covariate — at its cold-start
        // 0 (gam#1110: log-logistic AFT `age` pinned to exactly 0, RMSE 0.267 vs
        // lifelines 0.024). Collapsing to the log-t offset removes the free warp,
        // the constraint, and the ill-conditioning together, recovering the clean
        // survreg/lifelines-style parametric AFT MLE.
        if reduced_warp_logt_baseline_usable(&z, &design_exit, log_time_exit) {
            return Ok(location_logt_offset_time_block(
                &design_entry,
                &design_exit,
                &design_derivative_exit,
                p,
            ));
        }
        let reduced_entry = design_entry.dot(&z);
        let reduced_exit = design_exit.dot(&z);
        let reduced_derivative_exit = design_derivative_exit.dot(&z);
        // `z` spans the penalty null space, so every `zᵀ S z` is (numerically)
        // the zero `r×r` matrix: the reduced affine block has NO curvature left
        // to penalize. An unpenalized parametric block has no smoothing
        // parameter to select, so we drop the projected-to-zero penalties (and
        // their null-space-dimension bookkeeping) entirely rather than carry a
        // list of zero matrices. This is what makes the reduced time block
        // contribute ZERO ρ coordinates to the outer REML search — identical to
        // the constant `log_sigma` and rigid `threshold` blocks — so the outer
        // optimizer no longer crawls a flat, irrelevant time-smoothing ridge
        // (issue #736/#735/#721). The parametric design and the row-wise
        // monotonicity guard below carry all the time-warp structure; the
        // dropped penalties were exactly zero and contributed nothing.
        let reduced_penalties: Vec<Array2<f64>> = Vec::new();
        let reduced_nullspace_dims: Vec<usize> = Vec::new();
        let reduced_derivative_design =
            DesignMatrix::Dense(DenseDesignMatrix::from(reduced_derivative_exit.clone()));
        // Pointwise monotonicity in the reduced affine space: enforce the
        // derivative guard directly via row constraints on `X' z`. There is no
        // coordinate-cone lower-bound ridge here because the affine generators
        // are not individually sign-definite I-spline columns.
        let linear_constraints = time_derivative_guard_constraints(
            &reduced_derivative_design,
            &input.derivative_offset_exit,
            derivative_guard,
        )?;
        let initial_beta = match (linear_constraints.as_ref(), input.initial_beta.as_ref()) {
            (Some(constraints), Some(beta0)) => Some(project_onto_linear_constraints(
                r,
                constraints,
                Some(&z.t().dot(beta0)),
            )?),
            (_, Some(beta0)) => Some(z.t().dot(beta0)),
            _ => None,
        };
        return Ok(TimeBlockPrepared {
            design_entry: reduced_entry,
            design_exit: reduced_exit,
            design_derivative_exit: reduced_derivative_exit,
            coefficient_lower_bounds: None,
            linear_constraints,
            penalties: reduced_penalties,
            nullspace_dims: reduced_nullspace_dims,
            initial_beta,
            // Non-clean split (r != 2 or degenerate constant/time split): keep
            // both affine columns free with no pinned warp, offsets passthrough.
            transform: TimeIdentifiabilityTransform {
                z,
                affine_shift: Array1::zeros(p),
            },
            offset_entry: input.offset_entry.clone(),
            offset_exit: input.offset_exit.clone(),
            derivative_offset_exit: input.derivative_offset_exit.clone(),
            // Fallback reduce: the reduced I-spline columns are strictly
            // monotone (not row-constant), so the threshold keeps its intercept.
            pinned_free_row_constant: false,
            // Fallback reduce keeps a free warp; no location-channel log-t offset.
            location_log_time_offset: false,
        });
    }

    let penalties = input.penalties.clone();
    let coefficient_lower_bounds = structural_time_coefficient_lower_bounds_with_monotone_time_wiggle(
        &input.design_derivative_exit,
        &input.derivative_offset_exit,
        derivative_guard,
        monotone_time_wiggle_ncols,
    )?
    .ok_or_else(|| {
        "structural time block requires derivative offsets to encode the derivative guard and a non-negative derivative basis"
            .to_string()
    })?;
    let coefficient_constraints = lower_bound_constraints(&coefficient_lower_bounds);
    let derivative_constraints = time_derivative_guard_constraints(
        &input.design_derivative_exit,
        &input.derivative_offset_exit,
        derivative_guard,
    )?;
    let linear_constraints =
        append_linear_constraints(coefficient_constraints.clone(), derivative_constraints)?;
    let initial_beta = match (linear_constraints.as_ref(), input.initial_beta.as_ref()) {
        (Some(constraints), Some(beta0)) => Some(project_onto_linear_constraints(
            p,
            constraints,
            Some(beta0),
        )?),
        (_, Some(beta0)) => Some(beta0.clone()),
        _ => None,
    };

    Ok(TimeBlockPrepared {
        design_entry,
        design_exit,
        design_derivative_exit,
        coefficient_lower_bounds: Some(coefficient_lower_bounds),
        linear_constraints,
        penalties,
        nullspace_dims: input.nullspace_dims.clone(),
        initial_beta,
        // Identity (non-reduce) path: the raw time block passes through
        // unchanged, so the lift is `z = I`, no pinned warp, offsets verbatim.
        transform: TimeIdentifiabilityTransform {
            z: Array2::eye(p),
            affine_shift: Array1::zeros(p),
        },
        offset_entry: input.offset_entry.clone(),
        offset_exit: input.offset_exit.clone(),
        derivative_offset_exit: input.derivative_offset_exit.clone(),
        // Full flexible I-spline: no pin, threshold keeps its intercept.
        pinned_free_row_constant: false,
        // Flexible regime keeps the full warp; no location-channel log-t offset.
        location_log_time_offset: false,
    })
}


fn initial_log_lambdas<T>(
    penalties: &[T],
    rho0: Option<Array1<f64>>,
) -> Result<Array1<f64>, String> {
    let k = penalties.len();
    let rho = rho0.unwrap_or_else(|| Array1::zeros(k));
    if rho.len() != k {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "initial_log_lambdas mismatch: got {}, expected {k}",
                rho.len()
            ),
        }
        .into());
    }
    Ok(rho)
}


const DENSE_WEIGHTED_CROSSPROD_PARALLEL_FLOP_THRESHOLD: u64 = 200_000;

const DENSE_ROW_SCALE_PARALLEL_ELEM_THRESHOLD: usize = 100_000;

const DENSE_ROW_CHUNKS_PER_THREAD: usize = 4;


#[inline]
fn should_use_survival_rayon(work_items: u64) -> bool {
    rayon::current_num_threads() > 1
        && rayon::current_thread_index().is_none()
        && work_items >= DENSE_WEIGHTED_CROSSPROD_PARALLEL_FLOP_THRESHOLD
}


#[inline]
fn dense_row_chunk_count(nrows: usize) -> usize {
    let max_chunks = rayon::current_num_threads()
        .saturating_mul(DENSE_ROW_CHUNKS_PER_THREAD)
        .max(1);
    nrows.min(max_chunks).max(1)
}


fn accumulate_weighted_crossprod_dense_stable_rows(
    out: &mut Array2<f64>,
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
    rows: std::ops::Range<usize>,
) {
    for i in rows {
        let wi = weights[i];
        if wi == 0.0 {
            continue;
        }
        for j in 0..left.ncols() {
            let lij = left[[i, j]];
            if lij == 0.0 {
                continue;
            }
            for k in 0..right.ncols() {
                let rijk = right[[i, k]];
                if rijk == 0.0 {
                    continue;
                }
                let contrib = safe_product3(wi, lij, rijk);
                out[[j, k]] = safe_sum2(out[[j, k]], contrib);
            }
        }
    }
}


fn accumulate_weighted_crossprod_dense_rows(
    out: &mut Array2<f64>,
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
    rows: std::ops::Range<usize>,
) -> bool {
    for i in rows {
        let wi = weights[i];
        if wi == 0.0 {
            continue;
        }
        for j in 0..left.ncols() {
            let lij = left[[i, j]];
            if lij == 0.0 {
                continue;
            }
            let weighted_lij = wi * lij;
            if !weighted_lij.is_finite() {
                return false;
            }
            for k in 0..right.ncols() {
                let rijk = right[[i, k]];
                if rijk == 0.0 {
                    continue;
                }
                let contrib = weighted_lij * rijk;
                let updated = out[[j, k]] + contrib;
                if !contrib.is_finite() || !updated.is_finite() {
                    return false;
                }
                out[[j, k]] = updated;
            }
        }
    }
    true
}


fn weighted_crossprod_dense_stable(
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    if left.nrows() != weights.len() || right.nrows() != weights.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "weighted_crossprod_dense stable row mismatch: left is {}x{}, weights has {}, right is {}x{}",
            left.nrows(),
            left.ncols(),
            weights.len(),
            right.nrows(),
            right.ncols()
        ) }.into());
    }

    let nrows = weights.len();
    let out_dim = (left.ncols(), right.ncols());
    let work = (nrows as u64)
        .saturating_mul(left.ncols() as u64)
        .saturating_mul(right.ncols() as u64);

    let out = if nrows > 1 && should_use_survival_rayon(work) {
        use rayon::prelude::*;

        let chunk_count = dense_row_chunk_count(nrows);
        let chunk_rows = nrows.div_ceil(chunk_count);
        let partials: Vec<Array2<f64>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_rows;
                let end = (start + chunk_rows).min(nrows);
                let mut local = Array2::<f64>::zeros(out_dim);
                if start < end {
                    accumulate_weighted_crossprod_dense_stable_rows(
                        &mut local,
                        left,
                        weights,
                        right,
                        start..end,
                    );
                }
                local
            })
            .collect();

        let mut reduced = Array2::<f64>::zeros(out_dim);
        for local in partials {
            for (dst, src) in reduced.iter_mut().zip(local.iter()) {
                *dst = safe_sum2(*dst, *src);
            }
        }
        reduced
    } else {
        let mut serial = Array2::<f64>::zeros(out_dim);
        accumulate_weighted_crossprod_dense_stable_rows(
            &mut serial,
            left,
            weights,
            right,
            0..nrows,
        );
        serial
    };

    if out.iter().any(|value| !value.is_finite()) {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: "weighted_crossprod_dense stable accumulation produced non-finite values"
                .to_string(),
        }
        .into());
    }
    Ok(out)
}


fn weighted_crossprod_dense(
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    weighted_crossprod_dense_with_parallelism(left, weights, right, faer::get_global_parallelism())
}


fn weighted_crossprod_dense_with_parallelism(
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
    par: faer::Par,
) -> Result<Array2<f64>, String> {
    if left.nrows() != weights.len() || right.nrows() != weights.len() {
        return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
            "weighted_crossprod_dense row mismatch: left is {}x{}, weights has {}, right is {}x{}",
            left.nrows(),
            left.ncols(),
            weights.len(),
            right.nrows(),
            right.ncols()
        ) }.into());
    }
    if left.iter().any(|value| !value.is_finite()) || right.iter().any(|value| !value.is_finite()) {
        return Err(SurvivalLocationScaleError::InvalidConfiguration {
            reason: "weighted_crossprod_dense inputs contain non-finite design values".to_string(),
        }
        .into());
    }

    let nrows = weights.len();
    let sanitized_weights = sanitize_survival_weight_vector(weights);
    let work = (nrows as u64)
        .saturating_mul(left.ncols() as u64)
        .saturating_mul(right.ncols() as u64);

    if nrows > 1 && should_use_survival_rayon(work) {
        use rayon::prelude::*;

        let out_dim = (left.ncols(), right.ncols());
        let chunk_count = dense_row_chunk_count(nrows);
        let chunk_rows = nrows.div_ceil(chunk_count);
        let partials: Vec<Option<Array2<f64>>> = (0..chunk_count)
            .into_par_iter()
            .map(|chunk_idx| {
                let start = chunk_idx * chunk_rows;
                let end = (start + chunk_rows).min(nrows);
                let mut local = Array2::<f64>::zeros(out_dim);
                if start < end
                    && !accumulate_weighted_crossprod_dense_rows(
                        &mut local,
                        left,
                        &sanitized_weights,
                        right,
                        start..end,
                    )
                {
                    return None;
                }
                Some(local)
            })
            .collect();

        if partials.iter().all(Option::is_some) {
            let mut out = Array2::<f64>::zeros(out_dim);
            let mut fast_path_ok = true;
            'reduce: for local in partials.into_iter().flatten() {
                for (dst, src) in out.iter_mut().zip(local.iter()) {
                    let updated = *dst + *src;
                    if !updated.is_finite() {
                        fast_path_ok = false;
                        break 'reduce;
                    }
                    *dst = updated;
                }
            }
            if fast_path_ok {
                return Ok(out);
            }
        }
    } else {
        let mut weighted_right = right.clone();
        let mut fast_path_ok = true;
        'outer: for i in 0..weighted_right.nrows() {
            let wi = sanitized_weights[i];
            if wi == 0.0 {
                weighted_right.row_mut(i).fill(0.0);
                continue;
            }
            if wi == 1.0 {
                continue;
            }
            for j in 0..weighted_right.ncols() {
                let scaled = wi * weighted_right[[i, j]];
                if !scaled.is_finite() {
                    fast_path_ok = false;
                    break 'outer;
                }
                weighted_right[[i, j]] = scaled;
            }
        }
        if fast_path_ok {
            let out = fast_atb_with_parallelism(left, &weighted_right, par);
            if out.iter().all(|value| value.is_finite()) {
                return Ok(out);
            }
        }
    }

    weighted_crossprod_dense_stable(left, &sanitized_weights, right)
}


pub(crate) fn scale_dense_rows(
    mat: &Array2<f64>,
    coeffs: &Array1<f64>,
) -> Result<Array2<f64>, SurvivalLocationScaleError> {
    if mat.nrows() != coeffs.len() {
        crate::bail_dim_sls!(
            "row scaling dimension mismatch: matrix has {} rows but coeffs have {} entries",
            mat.nrows(),
            coeffs.len()
        );
    }
    let sanitized_coeffs = sanitize_survival_weight_vector(coeffs);
    let work = mat.nrows().saturating_mul(mat.ncols());
    let mut out = mat.clone();

    if mat.nrows() > 1
        && rayon::current_num_threads() > 1
        && rayon::current_thread_index().is_none()
        && work >= DENSE_ROW_SCALE_PARALLEL_ELEM_THRESHOLD
    {
        use rayon::prelude::*;

        let chunk_count = dense_row_chunk_count(mat.nrows());
        let chunk_rows = mat.nrows().div_ceil(chunk_count);
        out.axis_chunks_iter_mut(Axis(0), chunk_rows)
            .into_par_iter()
            .enumerate()
            .for_each(|(chunk_idx, mut rows)| {
                let start = chunk_idx * chunk_rows;
                for (local_i, mut row) in rows.rows_mut().into_iter().enumerate() {
                    let coeff = sanitized_coeffs[start + local_i];
                    row.mapv_inplace(|value| safe_product(value, coeff));
                }
            });
    } else {
        for i in 0..out.nrows() {
            let coeff = sanitized_coeffs[i];
            out.row_mut(i)
                .mapv_inplace(|value| safe_product(value, coeff));
        }
    }

    if out.iter().any(|value| value.is_nan()) {
        return Err(SurvivalLocationScaleError::NumericalFailure {
            reason: "row scaling produced NaN values".to_string(),
        });
    }
    Ok(out)
}


fn embed_tail_columns(
    local: &Array2<f64>,
    total_cols: usize,
    tail_range: std::ops::Range<usize>,
) -> Result<Array2<f64>, String> {
    if tail_range.end > total_cols || tail_range.len() != local.ncols() {
        return Err(SurvivalLocationScaleError::DimensionMismatch {
            reason: format!(
                "tail embedding mismatch: local_cols={}, total_cols={}, tail={:?}",
                local.ncols(),
                total_cols,
                tail_range
            ),
        }
        .into());
    }
    let mut out = Array2::<f64>::zeros((local.nrows(), total_cols));
    out.slice_mut(s![.., tail_range]).assign(local);
    Ok(out)
}


fn assign_block(target: &mut Array2<f64>, row_start: usize, col_start: usize, block: &Array2<f64>) {
    let row_end = row_start + block.nrows();
    let col_end = col_start + block.ncols();
    target
        .slice_mut(s![row_start..row_end, col_start..col_end])
        .assign(block);
}


fn assign_symmetric_block(
    target: &mut Array2<f64>,
    row_start: usize,
    col_start: usize,
    block: &Array2<f64>,
) {
    assign_block(target, row_start, col_start, block);
    if row_start != col_start || block.nrows() != block.ncols() {
        assign_block(target, col_start, row_start, &block.t().to_owned());
    }
}


fn validate_predict_inverse_link(
    inverse_link: &InverseLink,
) -> Result<(), SurvivalLocationScaleError> {
    match inverse_link {
        InverseLink::Standard(StandardLink::Log) => {
            Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "prediction does not support Standard(Log) for survival models".to_string(),
            })
        }
        InverseLink::Standard(StandardLink::Logit)
        | InverseLink::Standard(StandardLink::Probit)
        | InverseLink::Standard(StandardLink::CLogLog)
        | InverseLink::Standard(StandardLink::Identity)
        | InverseLink::LatentCLogLog(_)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => Ok(()),
    }
}


fn inverse_link_failure_prob_checked(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<f64, SurvivalLocationScaleError> {
    inverse_link_jet_for_inverse_link(inverse_link, eta)
        .map(|j| j.mu.clamp(0.0, 1.0))
        .map_err(|e| SurvivalLocationScaleError::NumericalFailure {
            reason: format!("inverse link prediction failed at eta={eta}: {e}"),
        })
}


fn inverse_link_survival_prob_checked(
    inverse_link: &InverseLink,
    eta: f64,
) -> Result<f64, SurvivalLocationScaleError> {
    inverse_link_failure_prob_checked(inverse_link, eta).map(|f| (1.0 - f).clamp(0.0, 1.0))
}


fn inverse_link_survival_probvalue(inverse_link: &InverseLink, eta: f64) -> f64 {
    match inverse_link {
        InverseLink::Standard(StandardLink::Probit) => probit_survival_value(eta),
        InverseLink::Standard(StandardLink::Logit) => 1.0 / (1.0 + eta.exp()),
        InverseLink::Standard(StandardLink::CLogLog) => (-(eta.exp())).exp(),
        InverseLink::Standard(StandardLink::Identity) => 1.0 - eta,
        InverseLink::Standard(StandardLink::Log) => {
            // SAFETY: survival families register only Probit/Logit/CLogLog/
            // Identity/LatentCLogLog/Sas/BetaLogistic/Mixture inverse links;
            // `validate_predict_inverse_link` rejects `Standard(Log)` upstream
            // so this arm is unreachable on a validated survival model.
            panic!("state-less log inverse link is invalid for survival prediction")
        }
        InverseLink::LatentCLogLog(_)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => inverse_link_survival_prob_checked(inverse_link, eta)
            .expect("validated inverse link should evaluate during prediction"),
    }
}


fn linear_predictor_se(x: ndarray::ArrayView2<'_, f64>, cov: &Array2<f64>) -> Array1<f64> {
    let xc = crate::faer_ndarray::fast_ab(&x, cov);
    Array1::from_iter((0..x.nrows()).map(|i| x.row(i).dot(&xc.row(i)).max(0.0).sqrt()))
}


#[derive(Clone)]
struct SurvivalWiggleGeometry {
    basis: Array2<f64>,
    basis_d1: Array2<f64>,
    basis_d2: Array2<f64>,
    dq_dq0: Array1<f64>,
    d2q_dq02: Array1<f64>,
    d3q_dq03: Array1<f64>,
}


#[derive(Clone, Copy)]
struct SurvivalBaseQScalars {
    eta_t: f64,
    inv_sigma: f64,
    q: f64,
    q_t: f64,
    q_ls: f64,
    q_tl: f64,
    q_ll: f64,
    q_tl_ls: f64,
    q_ll_ls: f64,
}


struct SurvivalDynamicGeometryRowsMut<'a> {
    q_exit: &'a mut [f64],
    q_entry: &'a mut [f64],
    qdot_exit: &'a mut [f64],
    dq_t_exit: &'a mut [f64],
    dq_t_entry: &'a mut [f64],
    dq_ls_exit: &'a mut [f64],
    dq_ls_entry: &'a mut [f64],
    d2q_tls_exit: &'a mut [f64],
    d2q_tls_entry: &'a mut [f64],
    d2q_ls_exit: &'a mut [f64],
    d2q_ls_entry: &'a mut [f64],
    d3q_tls_ls_exit: &'a mut [f64],
    d3q_tls_ls_entry: &'a mut [f64],
    d3q_ls_exit: &'a mut [f64],
    d3q_ls_entry: &'a mut [f64],
    dqdot_t: &'a mut [f64],
    dqdot_ls: &'a mut [f64],
    dqdot_td: &'a mut [f64],
    dqdot_lsd: &'a mut [f64],
    d2qdot_tt: &'a mut [f64],
    d2qdot_tls: &'a mut [f64],
    d2qdot_ttd: &'a mut [f64],
    d2qdot_tlsd: &'a mut [f64],
    d2qdot_ls: &'a mut [f64],
    d2qdot_lstd: &'a mut [f64],
    d2qdot_lslsd: &'a mut [f64],
    d3qdot_tls_ls: &'a mut [f64],
    d3qdot_tls_lsd: &'a mut [f64],
    d3qdot_td_ls_ls: &'a mut [f64],
    d3qdot_ls_ls_ls: &'a mut [f64],
    d3qdot_ls_ls_lsd: &'a mut [f64],
}


impl<'a> SurvivalDynamicGeometryRowsMut<'a> {
    #[inline]
    fn len(&self) -> usize {
        self.q_exit.len()
    }

    fn split_at_mut(self, mid: usize) -> (Self, Self) {
        let (q_exit_l, q_exit_r) = self.q_exit.split_at_mut(mid);
        let (q_entry_l, q_entry_r) = self.q_entry.split_at_mut(mid);
        let (qdot_exit_l, qdot_exit_r) = self.qdot_exit.split_at_mut(mid);
        let (dq_t_exit_l, dq_t_exit_r) = self.dq_t_exit.split_at_mut(mid);
        let (dq_t_entry_l, dq_t_entry_r) = self.dq_t_entry.split_at_mut(mid);
        let (dq_ls_exit_l, dq_ls_exit_r) = self.dq_ls_exit.split_at_mut(mid);
        let (dq_ls_entry_l, dq_ls_entry_r) = self.dq_ls_entry.split_at_mut(mid);
        let (d2q_tls_exit_l, d2q_tls_exit_r) = self.d2q_tls_exit.split_at_mut(mid);
        let (d2q_tls_entry_l, d2q_tls_entry_r) = self.d2q_tls_entry.split_at_mut(mid);
        let (d2q_ls_exit_l, d2q_ls_exit_r) = self.d2q_ls_exit.split_at_mut(mid);
        let (d2q_ls_entry_l, d2q_ls_entry_r) = self.d2q_ls_entry.split_at_mut(mid);
        let (d3q_tls_ls_exit_l, d3q_tls_ls_exit_r) = self.d3q_tls_ls_exit.split_at_mut(mid);
        let (d3q_tls_ls_entry_l, d3q_tls_ls_entry_r) = self.d3q_tls_ls_entry.split_at_mut(mid);
        let (d3q_ls_exit_l, d3q_ls_exit_r) = self.d3q_ls_exit.split_at_mut(mid);
        let (d3q_ls_entry_l, d3q_ls_entry_r) = self.d3q_ls_entry.split_at_mut(mid);
        let (dqdot_t_l, dqdot_t_r) = self.dqdot_t.split_at_mut(mid);
        let (dqdot_ls_l, dqdot_ls_r) = self.dqdot_ls.split_at_mut(mid);
        let (dqdot_td_l, dqdot_td_r) = self.dqdot_td.split_at_mut(mid);
        let (dqdot_lsd_l, dqdot_lsd_r) = self.dqdot_lsd.split_at_mut(mid);
        let (d2qdot_tt_l, d2qdot_tt_r) = self.d2qdot_tt.split_at_mut(mid);
        let (d2qdot_tls_l, d2qdot_tls_r) = self.d2qdot_tls.split_at_mut(mid);
        let (d2qdot_ttd_l, d2qdot_ttd_r) = self.d2qdot_ttd.split_at_mut(mid);
        let (d2qdot_tlsd_l, d2qdot_tlsd_r) = self.d2qdot_tlsd.split_at_mut(mid);
        let (d2qdot_ls_l, d2qdot_ls_r) = self.d2qdot_ls.split_at_mut(mid);
        let (d2qdot_lstd_l, d2qdot_lstd_r) = self.d2qdot_lstd.split_at_mut(mid);
        let (d2qdot_lslsd_l, d2qdot_lslsd_r) = self.d2qdot_lslsd.split_at_mut(mid);
        let (d3qdot_tls_ls_l, d3qdot_tls_ls_r) = self.d3qdot_tls_ls.split_at_mut(mid);
        let (d3qdot_tls_lsd_l, d3qdot_tls_lsd_r) = self.d3qdot_tls_lsd.split_at_mut(mid);
        let (d3qdot_td_ls_ls_l, d3qdot_td_ls_ls_r) = self.d3qdot_td_ls_ls.split_at_mut(mid);
        let (d3qdot_ls_ls_ls_l, d3qdot_ls_ls_ls_r) = self.d3qdot_ls_ls_ls.split_at_mut(mid);
        let (d3qdot_ls_ls_lsd_l, d3qdot_ls_ls_lsd_r) = self.d3qdot_ls_ls_lsd.split_at_mut(mid);

        (
            Self {
                q_exit: q_exit_l,
                q_entry: q_entry_l,
                qdot_exit: qdot_exit_l,
                dq_t_exit: dq_t_exit_l,
                dq_t_entry: dq_t_entry_l,
                dq_ls_exit: dq_ls_exit_l,
                dq_ls_entry: dq_ls_entry_l,
                d2q_tls_exit: d2q_tls_exit_l,
                d2q_tls_entry: d2q_tls_entry_l,
                d2q_ls_exit: d2q_ls_exit_l,
                d2q_ls_entry: d2q_ls_entry_l,
                d3q_tls_ls_exit: d3q_tls_ls_exit_l,
                d3q_tls_ls_entry: d3q_tls_ls_entry_l,
                d3q_ls_exit: d3q_ls_exit_l,
                d3q_ls_entry: d3q_ls_entry_l,
                dqdot_t: dqdot_t_l,
                dqdot_ls: dqdot_ls_l,
                dqdot_td: dqdot_td_l,
                dqdot_lsd: dqdot_lsd_l,
                d2qdot_tt: d2qdot_tt_l,
                d2qdot_tls: d2qdot_tls_l,
                d2qdot_ttd: d2qdot_ttd_l,
                d2qdot_tlsd: d2qdot_tlsd_l,
                d2qdot_ls: d2qdot_ls_l,
                d2qdot_lstd: d2qdot_lstd_l,
                d2qdot_lslsd: d2qdot_lslsd_l,
                d3qdot_tls_ls: d3qdot_tls_ls_l,
                d3qdot_tls_lsd: d3qdot_tls_lsd_l,
                d3qdot_td_ls_ls: d3qdot_td_ls_ls_l,
                d3qdot_ls_ls_ls: d3qdot_ls_ls_ls_l,
                d3qdot_ls_ls_lsd: d3qdot_ls_ls_lsd_l,
            },
            Self {
                q_exit: q_exit_r,
                q_entry: q_entry_r,
                qdot_exit: qdot_exit_r,
                dq_t_exit: dq_t_exit_r,
                dq_t_entry: dq_t_entry_r,
                dq_ls_exit: dq_ls_exit_r,
                dq_ls_entry: dq_ls_entry_r,
                d2q_tls_exit: d2q_tls_exit_r,
                d2q_tls_entry: d2q_tls_entry_r,
                d2q_ls_exit: d2q_ls_exit_r,
                d2q_ls_entry: d2q_ls_entry_r,
                d3q_tls_ls_exit: d3q_tls_ls_exit_r,
                d3q_tls_ls_entry: d3q_tls_ls_entry_r,
                d3q_ls_exit: d3q_ls_exit_r,
                d3q_ls_entry: d3q_ls_entry_r,
                dqdot_t: dqdot_t_r,
                dqdot_ls: dqdot_ls_r,
                dqdot_td: dqdot_td_r,
                dqdot_lsd: dqdot_lsd_r,
                d2qdot_tt: d2qdot_tt_r,
                d2qdot_tls: d2qdot_tls_r,
                d2qdot_ttd: d2qdot_ttd_r,
                d2qdot_tlsd: d2qdot_tlsd_r,
                d2qdot_ls: d2qdot_ls_r,
                d2qdot_lstd: d2qdot_lstd_r,
                d2qdot_lslsd: d2qdot_lslsd_r,
                d3qdot_tls_ls: d3qdot_tls_ls_r,
                d3qdot_tls_lsd: d3qdot_tls_lsd_r,
                d3qdot_td_ls_ls: d3qdot_td_ls_ls_r,
                d3qdot_ls_ls_ls: d3qdot_ls_ls_ls_r,
                d3qdot_ls_ls_lsd: d3qdot_ls_ls_lsd_r,
            },
        )
    }
}


struct SurvivalDynamicGeometryRowInputs<'a> {
    eta_t_exit: ndarray::ArrayView1<'a, f64>,
    eta_ls_exit: ndarray::ArrayView1<'a, f64>,
    eta_t_entry: ndarray::ArrayView1<'a, f64>,
    eta_ls_entry: ndarray::ArrayView1<'a, f64>,
    eta_t_deriv_exit: &'a Array1<f64>,
    eta_ls_deriv_exit: &'a Array1<f64>,
    wiggle_exit: Option<&'a SurvivalWiggleGeometry>,
    wiggle_entry: Option<&'a SurvivalWiggleGeometry>,
    link_beta: Option<ndarray::ArrayView1<'a, f64>>,
}


const SURVIVAL_DYNAMIC_GEOMETRY_PAR_CHUNK: usize = 1024;


fn fill_survival_dynamic_geometry_rows(
    rows: SurvivalDynamicGeometryRowsMut<'_>,
    row_start: usize,
    inputs: &SurvivalDynamicGeometryRowInputs<'_>,
) {
    let len = rows.len();
    if len <= SURVIVAL_DYNAMIC_GEOMETRY_PAR_CHUNK {
        fill_survival_dynamic_geometry_rows_serial(rows, row_start, inputs);
    } else {
        let mid = len / 2;
        let (left, right) = rows.split_at_mut(mid);
        rayon::join(
            || fill_survival_dynamic_geometry_rows(left, row_start, inputs),
            || fill_survival_dynamic_geometry_rows(right, row_start + mid, inputs),
        );
    }
}


fn fill_survival_dynamic_geometry_rows_serial(
    rows: SurvivalDynamicGeometryRowsMut<'_>,
    row_start: usize,
    inputs: &SurvivalDynamicGeometryRowInputs<'_>,
) {
    for offset in 0..rows.len() {
        let i = row_start + offset;
        let base_exit = survival_base_q_scalars(inputs.eta_t_exit[i], inputs.eta_ls_exit[i]);
        let base_entry = survival_base_q_scalars(inputs.eta_t_entry[i], inputs.eta_ls_entry[i]);
        let exit_dyn = if let (Some(wig), Some(beta_w)) = (inputs.wiggle_exit, inputs.link_beta) {
            compose_survival_dynamic_q(
                base_exit,
                inputs.eta_t_deriv_exit[i],
                inputs.eta_ls_deriv_exit[i],
                wig.basis.row(i).dot(&beta_w),
                wig.dq_dq0[i],
                wig.d2q_dq02[i],
                wig.d3q_dq03[i],
            )
        } else {
            compose_survival_dynamic_q(
                base_exit,
                inputs.eta_t_deriv_exit[i],
                inputs.eta_ls_deriv_exit[i],
                0.0,
                1.0,
                0.0,
                0.0,
            )
        };
        let entry_dyn = if let (Some(wig), Some(beta_w)) = (inputs.wiggle_entry, inputs.link_beta) {
            compose_survival_dynamic_q(
                base_entry,
                0.0,
                0.0,
                wig.basis.row(i).dot(&beta_w),
                wig.dq_dq0[i],
                wig.d2q_dq02[i],
                wig.d3q_dq03[i],
            )
        } else {
            compose_survival_dynamic_q(base_entry, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        };
        rows.q_exit[offset] = exit_dyn.q;
        rows.q_entry[offset] = entry_dyn.q;
        rows.qdot_exit[offset] = exit_dyn.qdot;
        rows.dq_t_exit[offset] = exit_dyn.q_t;
        rows.dq_t_entry[offset] = entry_dyn.q_t;
        rows.dq_ls_exit[offset] = exit_dyn.q_ls;
        rows.dq_ls_entry[offset] = entry_dyn.q_ls;
        rows.d2q_tls_exit[offset] = exit_dyn.q_tl;
        rows.d2q_tls_entry[offset] = entry_dyn.q_tl;
        rows.d2q_ls_exit[offset] = exit_dyn.q_ll;
        rows.d2q_ls_entry[offset] = entry_dyn.q_ll;
        rows.d3q_tls_ls_exit[offset] = exit_dyn.q_tl_ls;
        rows.d3q_tls_ls_entry[offset] = entry_dyn.q_tl_ls;
        rows.d3q_ls_exit[offset] = exit_dyn.q_ll_ls;
        rows.d3q_ls_entry[offset] = entry_dyn.q_ll_ls;
        rows.dqdot_t[offset] = exit_dyn.qdot_t;
        rows.dqdot_ls[offset] = exit_dyn.qdot_ls;
        rows.dqdot_td[offset] = exit_dyn.qdot_td;
        rows.dqdot_lsd[offset] = exit_dyn.qdot_lsd;
        rows.d2qdot_tt[offset] = exit_dyn.qdot_tt;
        rows.d2qdot_tls[offset] = exit_dyn.qdot_tls;
        rows.d2qdot_ttd[offset] = exit_dyn.qdot_ttd;
        rows.d2qdot_tlsd[offset] = exit_dyn.qdot_tlsd;
        rows.d2qdot_ls[offset] = exit_dyn.qdot_ll;
        rows.d2qdot_lstd[offset] = exit_dyn.qdot_lstd;
        rows.d2qdot_lslsd[offset] = exit_dyn.qdot_llsd;
        rows.d3qdot_tls_ls[offset] = exit_dyn.qdot_tll;
        rows.d3qdot_tls_lsd[offset] = exit_dyn.qdot_tlsd_ls;
        rows.d3qdot_td_ls_ls[offset] = exit_dyn.qdot_tdll;
        rows.d3qdot_ls_ls_ls[offset] = exit_dyn.qdot_lll;
        rows.d3qdot_ls_ls_lsd[offset] = exit_dyn.qdot_llsd_ls;
    }
}


#[derive(Clone, Copy)]
struct SurvivalDynamicQScalars {
    q: f64,
    q_t: f64,
    q_ls: f64,
    q_tl: f64,
    q_ll: f64,
    q_tl_ls: f64,
    q_ll_ls: f64,
    qdot: f64,
    qdot_t: f64,
    qdot_ls: f64,
    qdot_td: f64,
    qdot_lsd: f64,
    qdot_tt: f64,
    qdot_tls: f64,
    qdot_ttd: f64,
    qdot_tlsd: f64,
    qdot_ll: f64,
    qdot_lstd: f64,
    qdot_llsd: f64,
    qdot_tll: f64,
    qdot_tlsd_ls: f64,
    qdot_tdll: f64,
    qdot_lll: f64,
    qdot_llsd_ls: f64,
}


#[derive(Clone)]
struct SurvivalDynamicGeometry {
    h_exit: Array1<f64>,
    h_entry: Array1<f64>,
    hdot_exit: Array1<f64>,
    time_base_derivative_exit: Array1<f64>,
    time_jac_entry: Array2<f64>,
    time_jac_exit: Array2<f64>,
    time_jac_deriv: Array2<f64>,
    time_wiggle_basis_d1_entry: Option<Array2<f64>>,
    time_wiggle_basis_d1_exit: Option<Array2<f64>>,
    time_wiggle_basis_d2_exit: Option<Array2<f64>>,
    time_wiggle_d2_entry: Option<Array1<f64>>,
    time_wiggle_d2_exit: Option<Array1<f64>>,
    time_wiggle_d3_exit: Option<Array1<f64>>,
    eta_ls_exit: Array1<f64>,
    eta_ls_entry: Array1<f64>,
    q_exit: Array1<f64>,
    q_entry: Array1<f64>,
    qdot_exit: Array1<f64>,
    inv_sigma_exit: Array1<f64>,
    inv_sigma_entry: Array1<f64>,
    dq_t_exit: Array1<f64>,
    dq_t_entry: Array1<f64>,
    dq_ls_exit: Array1<f64>,
    dq_ls_entry: Array1<f64>,
    d2q_tls_exit: Array1<f64>,
    d2q_tls_entry: Array1<f64>,
    d2q_ls_exit: Array1<f64>,
    d2q_ls_entry: Array1<f64>,
    d3q_tls_ls_exit: Array1<f64>,
    d3q_tls_ls_entry: Array1<f64>,
    d3q_ls_exit: Array1<f64>,
    d3q_ls_entry: Array1<f64>,
    dqdot_t: Array1<f64>,
    dqdot_ls: Array1<f64>,
    dqdot_td: Array1<f64>,
    dqdot_lsd: Array1<f64>,
    d2qdot_tt: Array1<f64>,
    d2qdot_tls: Array1<f64>,
    d2qdot_ttd: Array1<f64>,
    d2qdot_tlsd: Array1<f64>,
    d2qdot_ls: Array1<f64>,
    d2qdot_lstd: Array1<f64>,
    d2qdot_lslsd: Array1<f64>,
    d3qdot_tls_ls: Array1<f64>,
    d3qdot_tls_lsd: Array1<f64>,
    d3qdot_td_ls_ls: Array1<f64>,
    d3qdot_ls_ls_ls: Array1<f64>,
    d3qdot_ls_ls_lsd: Array1<f64>,
    wiggle_basis_exit: Option<Array2<f64>>,
    wiggle_basis_entry: Option<Array2<f64>>,
    wiggle_basis_d1_exit: Option<Array2<f64>>,
    wiggle_basis_d1_entry: Option<Array2<f64>>,
    wiggle_basis_d2_exit: Option<Array2<f64>>,
    wiggle_qdot_basis_exit: Option<Array2<f64>>,
}


impl SurvivalDynamicGeometry {
    fn validate_precomputed_channels(&self) -> Result<(), String> {
        let n = self.h_exit.len();
        if self.time_base_derivative_exit.len() != n {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                "survival dynamic geometry derivative length mismatch: base_derivative={}, rows={n}",
                self.time_base_derivative_exit.len()
            ) }.into());
        }
        if let Some(basis) = self.time_wiggle_basis_d1_entry.as_ref()
            && basis.nrows() != n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival dynamic geometry wiggle d1 entry row mismatch: rows={}, expected {n}",
                    basis.nrows()
                ),
            }
            .into());
        }
        if let Some(basis) = self.time_wiggle_basis_d1_exit.as_ref()
            && basis.nrows() != n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival dynamic geometry wiggle d1 exit row mismatch: rows={}, expected {n}",
                    basis.nrows()
                ),
            }
            .into());
        }
        if let Some(basis) = self.time_wiggle_basis_d2_exit.as_ref()
            && basis.nrows() != n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch {
                reason: format!(
                    "survival dynamic geometry wiggle d2 exit row mismatch: rows={}, expected {n}",
                    basis.nrows()
                ),
            }
            .into());
        }
        if let Some(values) = self.time_wiggle_d2_entry.as_ref()
            && values.len() != n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                    "survival dynamic geometry wiggle d2 entry length mismatch: len={}, expected {n}",
                    values.len()
                ) }.into());
        }
        if let Some(values) = self.time_wiggle_d2_exit.as_ref()
            && values.len() != n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                    "survival dynamic geometry wiggle d2 exit length mismatch: len={}, expected {n}",
                    values.len()
                ) }.into());
        }
        if let Some(values) = self.time_wiggle_d3_exit.as_ref()
            && values.len() != n
        {
            return Err(SurvivalLocationScaleError::DimensionMismatch { reason: format!(
                    "survival dynamic geometry wiggle d3 exit length mismatch: len={}, expected {n}",
                    values.len()
                ) }.into());
        }
        Ok(())
    }
}


fn survival_wiggle_basis_with_options(
    q0: ndarray::ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    options: BasisOptions,
) -> Result<Array2<f64>, String> {
    monotone_wiggle_basis_with_derivative_order(q0, knots, degree, options.derivative_order)
}


fn survival_wiggle_third_basis(
    q0: ndarray::ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
) -> Result<Array2<f64>, String> {
    monotone_wiggle_basis_with_derivative_order(q0, knots, degree, 3)
}


fn survival_base_q_scalars(eta_t: f64, eta_ls: f64) -> SurvivalBaseQScalars {
    let (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) = q_chain_derivs_scalar(eta_t, eta_ls);
    let inv_sigma = exp_sigma_inverse_from_eta_scalar(eta_ls);
    SurvivalBaseQScalars {
        eta_t,
        inv_sigma,
        q: survival_q0_from_eta(eta_t, eta_ls),
        q_t,
        q_ls,
        q_tl,
        q_ll,
        q_tl_ls,
        q_ll_ls,
    }
}


#[inline]
fn survival_q0dot_from_base(
    base: SurvivalBaseQScalars,
    eta_t_deriv: f64,
    eta_ls_deriv: f64,
) -> f64 {
    let local_derivative = base.eta_t.mul_add(eta_ls_deriv, -eta_t_deriv);
    safe_product(base.inv_sigma, local_derivative)
}


fn compose_survival_dynamic_q(
    base: SurvivalBaseQScalars,
    eta_t_deriv: f64,
    eta_ls_deriv: f64,
    wiggle_value: f64,
    dq_dq0: f64,
    d2q_dq02: f64,
    d3q_dq03: f64,
) -> SurvivalDynamicQScalars {
    let a = base.q_t;
    let b = base.q_ls;
    let c = base.q_tl;
    let d = base.q_ll;
    let e = base.q_tl_ls;
    let f = base.q_ll_ls;
    let m1 = dq_dq0;
    let m2 = d2q_dq02;
    let m3 = d3q_dq03;
    let r = survival_q0dot_from_base(base, eta_t_deriv, eta_ls_deriv);
    let r_t = safe_product(c, eta_ls_deriv);
    let r_ls = safe_sum2(safe_product(c, eta_t_deriv), safe_product(d, eta_ls_deriv));
    let r_ll = safe_sum2(safe_product(e, eta_t_deriv), safe_product(f, eta_ls_deriv));
    let q_t = safe_product(m1, a);
    let q_ls = safe_product(m1, b);
    let q_tl = safe_sum2(safe_product(m2, safe_product(a, b)), safe_product(m1, c));
    let q_ll = safe_sum2(safe_product(m2, safe_product(b, b)), safe_product(m1, d));
    let q_tl_ls = safe_sum3(
        safe_product(m3, safe_product(a, safe_product(b, b))),
        safe_product(m2, safe_sum2(safe_product(a, d), 2.0 * safe_product(b, c))),
        safe_product(m1, e),
    );
    let q_ll_ls = safe_sum3(
        safe_product(m3, safe_product(b, safe_product(b, b))),
        safe_product(m2, 3.0 * safe_product(b, d)),
        safe_product(m1, f),
    );

    SurvivalDynamicQScalars {
        q: base.q + wiggle_value,
        q_t,
        q_ls,
        q_tl,
        q_ll,
        q_tl_ls,
        q_ll_ls,
        qdot: safe_product(m1, r),
        qdot_t: safe_sum2(safe_product(m2, safe_product(a, r)), safe_product(m1, r_t)),
        qdot_ls: safe_sum2(safe_product(m2, safe_product(b, r)), safe_product(m1, r_ls)),
        qdot_td: q_t,
        qdot_lsd: q_ls,
        qdot_tt: safe_sum2(
            safe_product(m3, safe_product(safe_product(a, a), r)),
            2.0 * safe_product(m2, safe_product(a, r_t)),
        ),
        qdot_tls: safe_sum3(
            safe_product(m3, safe_product(safe_product(a, b), r)),
            safe_product(
                m2,
                safe_sum3(
                    safe_product(c, r),
                    safe_product(a, r_ls),
                    safe_product(b, r_t),
                ),
            ),
            safe_product(m1, safe_product(e, eta_ls_deriv)),
        ),
        qdot_ttd: safe_product(m2, safe_product(a, a)),
        qdot_tlsd: safe_sum2(safe_product(m2, safe_product(a, b)), safe_product(m1, c)),
        qdot_ll: safe_sum3(
            safe_product(m3, safe_product(safe_product(b, b), r)),
            safe_product(
                m2,
                safe_sum2(safe_product(d, r), 2.0 * safe_product(b, r_ls)),
            ),
            safe_product(m1, r_ll),
        ),
        qdot_lstd: safe_sum2(safe_product(m2, safe_product(a, b)), safe_product(m1, c)),
        qdot_llsd: safe_sum2(safe_product(m2, safe_product(b, b)), safe_product(m1, d)),
        qdot_tll: safe_sum3(
            safe_product(m3, safe_product(a, safe_product(b, r))),
            safe_product(
                m2,
                safe_sum3(
                    safe_product(e, r),
                    safe_product(c, r_ls),
                    safe_product(d, r_t),
                ),
            ),
            safe_product(m1, safe_product(-e, eta_ls_deriv)),
        ),
        qdot_tlsd_ls: safe_sum3(
            safe_product(m3, safe_product(a, safe_product(b, b))),
            safe_product(m2, safe_sum2(safe_product(a, d), 2.0 * safe_product(b, c))),
            safe_product(m1, e),
        ),
        qdot_tdll: safe_sum3(
            safe_product(m3, safe_product(a, safe_product(b, b))),
            safe_product(m2, safe_sum2(safe_product(a, d), 2.0 * safe_product(b, c))),
            safe_product(m1, e),
        ),
        qdot_lll: safe_sum3(
            safe_product(m3, safe_product(safe_product(b, b), safe_product(b, r))),
            safe_product(
                m2,
                safe_sum3(
                    safe_product(f, r),
                    safe_product(3.0 * d, r_ls),
                    safe_product(3.0 * b, r_ll),
                ),
            ),
            safe_product(m1, -r_ll),
        ),
        qdot_llsd_ls: safe_sum3(
            safe_product(m3, safe_product(b, safe_product(b, b))),
            safe_product(m2, 3.0 * safe_product(b, d)),
            safe_product(m1, f),
        ),
    }
}


impl SurvivalLocationScaleFamily {
    fn build_dynamic_geometry(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalDynamicGeometry, String> {
        let n = self.n;
        let joint_states = self.validate_joint_states(block_states)?;
        let h_entry_base = joint_states.0.to_owned();
        let h_exit_base = joint_states.1.to_owned();
        let d_base = joint_states.2.to_owned();
        let eta_t_exit_view = joint_states.3;
        let eta_ls_exit = joint_states.4;
        let eta_t_entry_view = joint_states.5;
        let eta_ls_entry = joint_states.6;
        let eta_t_deriv_exit = joint_states.7;
        let eta_ls_deriv_exit = joint_states.8;
        let mut eta_t_deriv_exit = eta_t_deriv_exit
            .map(|v| v.to_owned())
            .unwrap_or_else(|| Array1::zeros(n));
        let eta_ls_deriv_exit = eta_ls_deriv_exit
            .map(|v| v.to_owned())
            .unwrap_or_else(|| Array1::zeros(n));
        // σ-scaled log-t AFT location baseline (issue #892). In the rank-1 reduced
        // parametric-AFT regime the time warp is removed (`h ≡ 0`); the `log t`
        // baseline instead shifts the effective location predictor on the σ-scaled
        // `q` channel — `η_t → η_t − log t` (value) with derivative `−1/t` — so the
        // standardized residual is `u = inv_sigma·(log t − η_t) = (log t − μ)/σ`
        // and the event Jacobian gains `qdot = inv_sigma/t → log_g = −η_ls − log t`,
        // the `−log σ` term that identifies σ. Shifting the effective location here
        // (before q0 / the q-row kernel) routes the whole σ coupling through the
        // existing `q`-derivative/Hessian stack — no new time×log_sigma cross-terms.
        let (eta_t_exit, eta_t_entry) = if let Some(loc) = self.location_log_time.as_ref() {
            eta_t_deriv_exit += &loc.deriv_exit;
            (
                &eta_t_exit_view + &loc.value_exit,
                &eta_t_entry_view + &loc.value_entry,
            )
        } else {
            (eta_t_exit_view.to_owned(), eta_t_entry_view.to_owned())
        };
        let inv_sigma_exit = eta_ls_exit.mapv(exp_sigma_inverse_from_eta_scalar);
        let inv_sigma_entry = eta_ls_entry.mapv(exp_sigma_inverse_from_eta_scalar);
        let q0_exit = Array1::from_iter(
            eta_t_exit
                .iter()
                .zip(eta_ls_exit.iter())
                .map(|(&t, &ls)| survival_q0_from_eta(t, ls)),
        );
        let q0_entry = Array1::from_iter(
            eta_t_entry
                .iter()
                .zip(eta_ls_entry.iter())
                .map(|(&t, &ls)| survival_q0_from_eta(t, ls)),
        );
        let time_wiggle_range = self.time_wiggle_range();
        let beta_time_w = if self.time_wiggle_ncols > 0 {
            Some(
                block_states[Self::BLOCK_TIME]
                    .beta
                    .slice(s![time_wiggle_range.start..time_wiggle_range.end]),
            )
        } else {
            None
        };
        let time_wiggle_entry = if let Some(beta_w) = beta_time_w {
            self.time_wiggle_geometry(h_entry_base.view(), beta_w)?
        } else {
            None
        };
        let time_wiggle_exit = if let Some(beta_w) = beta_time_w {
            self.time_wiggle_geometry(h_exit_base.view(), beta_w)?
        } else {
            None
        };
        let beta_w = if self.x_link_wiggle.is_some() {
            Some(block_states[Self::BLOCK_LINK_WIGGLE].beta.view())
        } else {
            None
        };
        let wiggle_exit = if let Some(beta_w) = beta_w {
            self.wiggle_geometry(q0_exit.view(), beta_w)?
        } else {
            None
        };
        let wiggle_entry = if let Some(beta_w) = beta_w {
            self.wiggle_geometry(q0_entry.view(), beta_w)?
        } else {
            None
        };
        if self.x_link_wiggle.is_some() && (wiggle_exit.is_none() || wiggle_entry.is_none()) {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "survival location-scale linkwiggle requires dynamic knot/degree metadata"
                    .to_string(),
            }
            .into());
        }
        if self.time_wiggle_ncols > 0 && (time_wiggle_exit.is_none() || time_wiggle_entry.is_none())
        {
            return Err(SurvivalLocationScaleError::InvalidConfiguration {
                reason: "survival location-scale timewiggle requires dynamic knot/degree metadata"
                    .to_string(),
            }
            .into());
        }

        let mut h_entry = h_entry_base.clone();
        let mut h_exit = h_exit_base.clone();
        let mut hdot_exit = d_base.clone();
        let mut time_jac_entry = self.x_time_entry.as_ref().clone();
        let mut time_jac_exit = self.x_time_exit.as_ref().clone();
        let mut time_jac_deriv = self.x_time_deriv.as_ref().clone();
        let mut time_wiggle_basis_d1_entry = None;
        let mut time_wiggle_basis_d1_exit = None;
        let mut time_wiggle_basis_d2_exit = None;
        let mut time_wiggle_d2_entry = None;
        let mut time_wiggle_d2_exit = None;
        let mut time_wiggle_d3_exit = None;

        if let (Some(wig_entry), Some(wig_exit), Some(beta_w)) = (
            time_wiggle_entry.as_ref(),
            time_wiggle_exit.as_ref(),
            beta_time_w,
        ) {
            h_entry = &h_entry_base + &fast_av(&wig_entry.basis, &beta_w);
            h_exit = &h_exit_base + &fast_av(&wig_exit.basis, &beta_w);
            hdot_exit = &wig_exit.dq_dq0 * &d_base;
            time_jac_entry = scale_dense_rows(self.x_time_entry.as_ref(), &wig_entry.dq_dq0)?;
            time_jac_exit = scale_dense_rows(self.x_time_exit.as_ref(), &wig_exit.dq_dq0)?;
            time_jac_deriv = scale_dense_rows(
                self.x_time_exit.as_ref(),
                &safe_hadamard_product(&wig_exit.d2q_dq02, &d_base)?,
            )? + &scale_dense_rows(self.x_time_deriv.as_ref(), &wig_exit.dq_dq0)?;
            let wiggle_entry_full = embed_tail_columns(
                &wig_entry.basis,
                time_jac_entry.ncols(),
                time_wiggle_range.clone(),
            )?;
            let wiggle_exit_full = embed_tail_columns(
                &wig_exit.basis,
                time_jac_exit.ncols(),
                time_wiggle_range.clone(),
            )?;
            time_jac_entry
                .slice_mut(s![.., time_wiggle_range.start..time_wiggle_range.end])
                .assign(
                    &wiggle_entry_full
                        .slice(s![.., time_wiggle_range.start..time_wiggle_range.end]),
                );
            time_jac_exit
                .slice_mut(s![.., time_wiggle_range.start..time_wiggle_range.end])
                .assign(
                    &wiggle_exit_full.slice(s![.., time_wiggle_range.start..time_wiggle_range.end]),
                );
            let wiggle_qdot = scale_dense_rows(&wig_exit.basis_d1, &d_base)?;
            let wiggle_qdot_full = embed_tail_columns(
                &wiggle_qdot,
                time_jac_deriv.ncols(),
                time_wiggle_range.clone(),
            )?;
            time_jac_deriv
                .slice_mut(s![.., time_wiggle_range.start..time_wiggle_range.end])
                .assign(
                    &wiggle_qdot_full.slice(s![.., time_wiggle_range.start..time_wiggle_range.end]),
                );
            time_wiggle_basis_d1_entry = Some(wig_entry.basis_d1.clone());
            time_wiggle_basis_d1_exit = Some(wig_exit.basis_d1.clone());
            time_wiggle_basis_d2_exit = Some(wig_exit.basis_d2.clone());
            time_wiggle_d2_entry = Some(wig_entry.d2q_dq02.clone());
            time_wiggle_d2_exit = Some(wig_exit.d2q_dq02.clone());
            time_wiggle_d3_exit = Some(wig_exit.d3q_dq03.clone());
        }

        let mut q_exit = Array1::<f64>::zeros(n);
        let mut q_entry = Array1::<f64>::zeros(n);
        let mut qdot_exit = Array1::<f64>::zeros(n);
        let mut dq_t_exit = Array1::<f64>::zeros(n);
        let mut dq_t_entry = Array1::<f64>::zeros(n);
        let mut dq_ls_exit = Array1::<f64>::zeros(n);
        let mut dq_ls_entry = Array1::<f64>::zeros(n);
        let mut d2q_tls_exit = Array1::<f64>::zeros(n);
        let mut d2q_tls_entry = Array1::<f64>::zeros(n);
        let mut d2q_ls_exit = Array1::<f64>::zeros(n);
        let mut d2q_ls_entry = Array1::<f64>::zeros(n);
        let mut d3q_tls_ls_exit = Array1::<f64>::zeros(n);
        let mut d3q_tls_ls_entry = Array1::<f64>::zeros(n);
        let mut d3q_ls_exit = Array1::<f64>::zeros(n);
        let mut d3q_ls_entry = Array1::<f64>::zeros(n);
        let mut dqdot_t = Array1::<f64>::zeros(n);
        let mut dqdot_ls = Array1::<f64>::zeros(n);
        let mut dqdot_td = Array1::<f64>::zeros(n);
        let mut dqdot_lsd = Array1::<f64>::zeros(n);
        let mut d2qdot_tt = Array1::<f64>::zeros(n);
        let mut d2qdot_tls = Array1::<f64>::zeros(n);
        let mut d2qdot_ttd = Array1::<f64>::zeros(n);
        let mut d2qdot_tlsd = Array1::<f64>::zeros(n);
        let mut d2qdot_ls = Array1::<f64>::zeros(n);
        let mut d2qdot_lstd = Array1::<f64>::zeros(n);
        let mut d2qdot_lslsd = Array1::<f64>::zeros(n);
        let mut d3qdot_tls_ls = Array1::<f64>::zeros(n);
        let mut d3qdot_tls_lsd = Array1::<f64>::zeros(n);
        let mut d3qdot_td_ls_ls = Array1::<f64>::zeros(n);
        let mut d3qdot_ls_ls_ls = Array1::<f64>::zeros(n);
        let mut d3qdot_ls_ls_lsd = Array1::<f64>::zeros(n);

        let dynamic_row_inputs = SurvivalDynamicGeometryRowInputs {
            eta_t_exit: eta_t_exit.view(),
            eta_ls_exit,
            eta_t_entry: eta_t_entry.view(),
            eta_ls_entry,
            eta_t_deriv_exit: &eta_t_deriv_exit,
            eta_ls_deriv_exit: &eta_ls_deriv_exit,
            wiggle_exit: wiggle_exit.as_ref(),
            wiggle_entry: wiggle_entry.as_ref(),
            link_beta: beta_w,
        };
        let dynamic_rows = SurvivalDynamicGeometryRowsMut {
            q_exit: q_exit.as_slice_mut().expect("q_exit must be contiguous"),
            q_entry: q_entry.as_slice_mut().expect("q_entry must be contiguous"),
            qdot_exit: qdot_exit
                .as_slice_mut()
                .expect("qdot_exit must be contiguous"),
            dq_t_exit: dq_t_exit
                .as_slice_mut()
                .expect("dq_t_exit must be contiguous"),
            dq_t_entry: dq_t_entry
                .as_slice_mut()
                .expect("dq_t_entry must be contiguous"),
            dq_ls_exit: dq_ls_exit
                .as_slice_mut()
                .expect("dq_ls_exit must be contiguous"),
            dq_ls_entry: dq_ls_entry
                .as_slice_mut()
                .expect("dq_ls_entry must be contiguous"),
            d2q_tls_exit: d2q_tls_exit
                .as_slice_mut()
                .expect("d2q_tls_exit must be contiguous"),
            d2q_tls_entry: d2q_tls_entry
                .as_slice_mut()
                .expect("d2q_tls_entry must be contiguous"),
            d2q_ls_exit: d2q_ls_exit
                .as_slice_mut()
                .expect("d2q_ls_exit must be contiguous"),
            d2q_ls_entry: d2q_ls_entry
                .as_slice_mut()
                .expect("d2q_ls_entry must be contiguous"),
            d3q_tls_ls_exit: d3q_tls_ls_exit
                .as_slice_mut()
                .expect("d3q_tls_ls_exit must be contiguous"),
            d3q_tls_ls_entry: d3q_tls_ls_entry
                .as_slice_mut()
                .expect("d3q_tls_ls_entry must be contiguous"),
            d3q_ls_exit: d3q_ls_exit
                .as_slice_mut()
                .expect("d3q_ls_exit must be contiguous"),
            d3q_ls_entry: d3q_ls_entry
                .as_slice_mut()
                .expect("d3q_ls_entry must be contiguous"),
            dqdot_t: dqdot_t.as_slice_mut().expect("dqdot_t must be contiguous"),
            dqdot_ls: dqdot_ls
                .as_slice_mut()
                .expect("dqdot_ls must be contiguous"),
            dqdot_td: dqdot_td
                .as_slice_mut()
                .expect("dqdot_td must be contiguous"),
            dqdot_lsd: dqdot_lsd
                .as_slice_mut()
                .expect("dqdot_lsd must be contiguous"),
            d2qdot_tt: d2qdot_tt
                .as_slice_mut()
                .expect("d2qdot_tt must be contiguous"),
            d2qdot_tls: d2qdot_tls
                .as_slice_mut()
                .expect("d2qdot_tls must be contiguous"),
            d2qdot_ttd: d2qdot_ttd
                .as_slice_mut()
                .expect("d2qdot_ttd must be contiguous"),
            d2qdot_tlsd: d2qdot_tlsd
                .as_slice_mut()
                .expect("d2qdot_tlsd must be contiguous"),
            d2qdot_ls: d2qdot_ls
                .as_slice_mut()
                .expect("d2qdot_ls must be contiguous"),
            d2qdot_lstd: d2qdot_lstd
                .as_slice_mut()
                .expect("d2qdot_lstd must be contiguous"),
            d2qdot_lslsd: d2qdot_lslsd
                .as_slice_mut()
                .expect("d2qdot_lslsd must be contiguous"),
            d3qdot_tls_ls: d3qdot_tls_ls
                .as_slice_mut()
                .expect("d3qdot_tls_ls must be contiguous"),
            d3qdot_tls_lsd: d3qdot_tls_lsd
                .as_slice_mut()
                .expect("d3qdot_tls_lsd must be contiguous"),
            d3qdot_td_ls_ls: d3qdot_td_ls_ls
                .as_slice_mut()
                .expect("d3qdot_td_ls_ls must be contiguous"),
            d3qdot_ls_ls_ls: d3qdot_ls_ls_ls
                .as_slice_mut()
                .expect("d3qdot_ls_ls_ls must be contiguous"),
            d3qdot_ls_ls_lsd: d3qdot_ls_ls_lsd
                .as_slice_mut()
                .expect("d3qdot_ls_ls_lsd must be contiguous"),
        };
        fill_survival_dynamic_geometry_rows(dynamic_rows, 0, &dynamic_row_inputs);

        let wiggle_qdot_basis_exit = wiggle_exit.as_ref().map(|wig| {
            use rayon::prelude::*;

            let mut out = wig.basis_d1.clone();
            let r = Array1::from_vec(
                (0..n)
                    .into_par_iter()
                    .map(|i| {
                        let base_exit = survival_base_q_scalars(eta_t_exit[i], eta_ls_exit[i]);
                        survival_q0dot_from_base(
                            base_exit,
                            eta_t_deriv_exit[i],
                            eta_ls_deriv_exit[i],
                        )
                    })
                    .collect(),
            );
            let ncols = out.ncols();
            out.as_slice_mut()
                .expect("wiggle qdot basis must be contiguous")
                .par_chunks_mut(ncols)
                .enumerate()
                .for_each(|(i, row)| {
                    for value in row {
                        *value *= r[i];
                    }
                });
            out
        });

        let dynamic = SurvivalDynamicGeometry {
            h_exit,
            h_entry,
            hdot_exit,
            time_base_derivative_exit: d_base,
            time_jac_entry,
            time_jac_exit,
            time_jac_deriv,
            time_wiggle_basis_d1_entry,
            time_wiggle_basis_d1_exit,
            time_wiggle_basis_d2_exit,
            time_wiggle_d2_entry,
            time_wiggle_d2_exit,
            time_wiggle_d3_exit,
            eta_ls_exit: eta_ls_exit.to_owned(),
            eta_ls_entry: eta_ls_entry.to_owned(),
            q_exit,
            q_entry,
            qdot_exit,
            inv_sigma_exit,
            inv_sigma_entry,
            dq_t_exit,
            dq_t_entry,
            dq_ls_exit,
            dq_ls_entry,
            d2q_tls_exit,
            d2q_tls_entry,
            d2q_ls_exit,
            d2q_ls_entry,
            d3q_tls_ls_exit,
            d3q_tls_ls_entry,
            d3q_ls_exit,
            d3q_ls_entry,
            dqdot_t,
            dqdot_ls,
            dqdot_td,
            dqdot_lsd,
            d2qdot_tt,
            d2qdot_tls,
            d2qdot_ttd,
            d2qdot_tlsd,
            d2qdot_ls,
            d2qdot_lstd,
            d2qdot_lslsd,
            d3qdot_tls_ls,
            d3qdot_tls_lsd,
            d3qdot_td_ls_ls,
            d3qdot_ls_ls_ls,
            d3qdot_ls_ls_lsd,
            wiggle_basis_exit: wiggle_exit.as_ref().map(|w| w.basis.clone()),
            wiggle_basis_entry: wiggle_entry.as_ref().map(|w| w.basis.clone()),
            wiggle_basis_d1_exit: wiggle_exit.as_ref().map(|w| w.basis_d1.clone()),
            wiggle_basis_d1_entry: wiggle_entry.as_ref().map(|w| w.basis_d1.clone()),
            wiggle_basis_d2_exit: wiggle_exit.as_ref().map(|w| w.basis_d2.clone()),
            wiggle_qdot_basis_exit,
        };
        dynamic.validate_precomputed_channels()?;
        Ok(dynamic)
    }
}


struct PredictionLinearPredictors {
    h: Array1<f64>,
    time_jac: Array2<f64>,
    eta_t: Array1<f64>,
    inv_sigma: Array1<f64>,
    etaw: Option<Array1<f64>>,
    wiggle_design: Option<Array2<f64>>,
    dq_dq0: Option<Array1<f64>>,
}
