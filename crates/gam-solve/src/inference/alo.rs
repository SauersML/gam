use crate::estimate::{EstimationError, FitGeometry, UnifiedFitResult, dispersion_from_likelihood};
use crate::pirls;
use faer::Mat as FaerMat;
use faer::linalg::matmul::matmul;
use faer::prelude::ReborrowMut;
use faer::{Accum, Par};
use gam_linalg::faer_ndarray::{FaerArrayView, FaerCholesky};
use gam_linalg::matrix::{DesignMatrix, PsdWeightsView, SignedWeightsView};
use gam_linalg::utils::{
    CertifiedSpdFactor, certified_spd_factorize, symmetric_extremes,
    validate_finite_symmetric_matrix,
};
use gam_math::probability::signed_log_sum_exp;
use gam_problem::{Dispersion, LikelihoodScaleMetadata, LinkFunction, ResponseFamily};
use ndarray::{Array1, Array2, ArrayView1, ShapeBuilder, s};
use opt::{BacktrackConfig, backtracking_line_search};
use std::convert::Infallible;
use std::fmt;
use std::ops::Range;

/// Typed error variants for the ALO (approximate leave-one-out) diagnostics
/// module.
///
/// Public entry points continue to return `Result<_, EstimationError>`; this
/// enum is materialized at leaf sites and converted at the boundary via
/// `From<AloError> for EstimationError` so error text remains byte-identical
/// to the previous `EstimationError::InvalidInput(format!(...))` /
/// `ModelIsIllConditioned { ... }` output.
#[derive(Debug, Clone)]
pub enum AloError {
    /// Caller-supplied configuration is structurally invalid: dimension
    /// mismatch, non-finite inputs that are not weights/response, missing
    /// PIRLS / geometry artifacts, or out-of-range scalar parameters.
    InvalidInput { reason: String },
    /// IRLS weights or working response contain a non-finite entry, or the
    /// working response itself is invalid.
    WeightInvalid { reason: String },
    /// The dense design matrix required for ALO could not be materialized
    /// from the underlying PIRLS artifact (e.g. sparse-only export).
    DesignDegenerate { reason: String },
    /// Per-observation ALO computation produced a non-finite value (variance,
    /// denominator, or corrected η̃) at convergence.
    LooComputationFailed { reason: String },
}

impl fmt::Display for AloError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AloError::InvalidInput { reason }
            | AloError::WeightInvalid { reason }
            | AloError::DesignDegenerate { reason }
            | AloError::LooComputationFailed { reason } => f.write_str(reason),
        }
    }
}

impl std::error::Error for AloError {}

impl From<AloError> for EstimationError {
    fn from(err: AloError) -> EstimationError {
        match err {
            AloError::InvalidInput { reason }
            | AloError::WeightInvalid { reason }
            | AloError::DesignDegenerate { reason }
            | AloError::LooComputationFailed { reason } => EstimationError::InvalidInput(reason),
        }
    }
}

impl From<AloError> for String {
    fn from(err: AloError) -> String {
        err.to_string()
    }
}

/// Approximate leave-one-out diagnostics derived from a fitted model.
#[derive(Debug, Clone)]
pub struct AloDiagnostics {
    pub eta_tilde: Array1<f64>,
    /// Bayesian/conditional standard error on eta:
    /// sqrt(phi * x_i^T H^{-1} x_i).
    pub se_bayes: Array1<f64>,
    /// Frequentist sandwich-style standard error on eta:
    /// sqrt(phi * x_i^T H^{-1} X^T W X H^{-1} x_i).
    pub se_sandwich: Array1<f64>,
    /// Observed-curvature row leverage `W_H,i x_iᵀH⁻¹x_i`. This is signed for
    /// non-canonical links; negative values are valid and are never projected.
    pub leverage: Array1<f64>,
}

#[inline]
fn alo_eta_updatewith_offset(
    eta_hat: f64,
    z: f64,
    offset: f64,
    x_hinv_x: f64,
    score_weight: f64,
    denom: f64,
) -> f64 {
    // PIRLS working-response algebra is centered on offset, so the scalar
    // score uses (eta - offset) - (z - offset).
    let eta_centered = eta_hat - offset;
    let z_centered = z - offset;
    let score = score_weight * (eta_centered - z_centered);
    offset + eta_centered + x_hinv_x * score / denom
}

/// Per-row score and curvature of the penalized NLL contribution as functions
/// of the row's linear predictor `eta`.
///
/// Returns `(ℓ_i'(eta), ℓ_i''(eta))` where `ℓ_i` is the (dispersion-scaled)
/// negative log-likelihood of observation `i` viewed as a univariate function
/// of `eta_i = x_i^T β`. This is the local family geometry that the ALO
/// frozen-curvature fixed point [`alo_eta_exact_frozen_curvature`] iterates to
/// convergence; supplying it upgrades the single-Newton-step ALO correction to
/// the exact leave-`i`-out predictor under a frozen penalized Hessian.
pub type AloScalarScoreCurvature<'a> =
    dyn Fn(usize, f64) -> Result<(f64, f64), AloError> + Sync + 'a;

/// Maximum scalar Newton iterations for the exact frozen-curvature ALO fixed
/// point. The map `r(η) = η − η̂ − a_ii ℓ_i'(η)` is one-dimensional and
/// strongly contractive for the well-leveraged majority of points, so this
/// caps the rare high-leverage / near-separation rows where convergence is
/// slow without ever exceeding O(1) work per observation.
const ALO_EXACT_SCALAR_MAX_ITERS: usize = 64;

/// Backward-error allowance for the three-term scalar residual
/// `η - η̂ - a·ℓ'(η)`. It scales with the largest term and shrinks all the way
/// to exact zero, so small predictors do not inherit an absolute error floor.
#[inline]
fn alo_scalar_residual_allowance(eta: f64, eta_hat: f64, score_step: f64) -> f64 {
    32.0 * f64::EPSILON * eta.abs().max(eta_hat.abs()).max(score_step.abs())
}

/// Solve the frozen-curvature ALO leave-`i`-out fixed point exactly.
///
/// The leave-`i`-out optimum differs from the full fit only through the removed
/// observation, whose gradient/Hessian depend on `β` solely via the scalar
/// `η_i = x_i^T β`. Freezing the penalized Hessian `H` at its converged value
/// reduces the exact leave-`i`-out condition to the scalar equation
///
///   η = η̂_i + a_ii · ℓ_i'(η),     a_ii = x_i^T H^{-1} x_i,
///
/// where `ℓ_i'(η)` is the row's NLL score (so that `∇F = ℓ_i'(η_i) x_i` at the
/// leave-`i`-out point). The single-Newton-step ALO is exactly the first
/// iterate of Newton's method on `r(η) = η − η̂_i − a_ii ℓ_i'(η)` started at
/// `η̂_i`; iterating to convergence captures the change in the held-out point's
/// likelihood curvature (the dominant first-order error on small-`n`, curved
/// likelihoods such as binomial logistic regression near separation).
///
/// `score_curvature(eta)` returns `(ℓ_i'(eta), ℓ_i''(eta))`. The returned value
/// is the corrected linear predictor `η̃_i`. Failure to reach the residual
/// tolerance is reported to the caller; no one-step approximation is substituted
/// for a failed exact solve.
#[derive(Debug, Clone, PartialEq)]
enum AloExactScalarError {
    EvaluationFailed {
        eta: f64,
        reason: String,
    },
    NonFiniteScoreCurvature {
        eta: f64,
        ell_prime: f64,
        ell_double: f64,
    },
    DegenerateJacobian {
        eta: f64,
        jacobian: f64,
    },
    NonFiniteStep {
        eta: f64,
        residual: f64,
        jacobian: f64,
        next: f64,
    },
    MaxIterations {
        iterations: usize,
        residual: f64,
        tolerance: f64,
        eta: f64,
    },
}

impl fmt::Display for AloExactScalarError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            AloExactScalarError::EvaluationFailed { eta, ref reason } => {
                write!(
                    f,
                    "score/curvature evaluation failed at eta={eta:.6e}: {reason}"
                )
            }
            AloExactScalarError::NonFiniteScoreCurvature {
                eta,
                ell_prime,
                ell_double,
            } => write!(
                f,
                "non-finite score/curvature at eta={eta:.6e}: ell_prime={ell_prime:.6e}, ell_double={ell_double:.6e}"
            ),
            AloExactScalarError::DegenerateJacobian { eta, jacobian } => write!(
                f,
                "degenerate Newton Jacobian at eta={eta:.6e}: jacobian={jacobian:.6e}"
            ),
            AloExactScalarError::NonFiniteStep {
                eta,
                residual,
                jacobian,
                next,
            } => write!(
                f,
                "non-finite Newton step from eta={eta:.6e}: residual={residual:.6e}, jacobian={jacobian:.6e}, next={next:.6e}"
            ),
            AloExactScalarError::MaxIterations {
                iterations,
                residual,
                tolerance,
                eta,
            } => write!(
                f,
                "did not converge within {iterations} iterations: residual={residual:.6e}, eta={eta:.6e}, backward-error allowance={tolerance:.6e}"
            ),
        }
    }
}

/// Maximum number of step halvings in the backtracking line search that
/// globalizes the scalar Newton iteration. `2^{-40}` shrinks a unit step below
/// one ulp relative to an ordinary finite η, so a row that cannot make progress
/// within this budget is genuinely stalled rather than merely under-damped.
const ALO_EXACT_SCALAR_BACKTRACKS: usize = 40;

#[inline]
fn alo_eta_exact_frozen_curvature(
    eta_hat: f64,
    a_ii: f64,
    score_curvature: &dyn Fn(f64) -> Result<(f64, f64), AloError>,
) -> Result<f64, AloExactScalarError> {
    // Residual of the leave-i-out fixed point η = η̂ + a_ii ℓ'(η):
    //   r(η) = η − η̂ − a_ii ℓ'(η),     r'(η) = 1 − a_ii ℓ''(η) = jac.
    // For an exponential-family NLL score ℓ'(η) = c_i(μ(η) − y) on a non-linear
    // (e.g. log) link the curvature ℓ''(η) = c_i μ'(η) grows without bound, so
    // r(η) is concave with an interior maximum where the weighted leverage
    // a_ii ℓ'' passes 1 (jac = 0): the leave-i-out root that limits to η̂ as
    // a_ii → 0 sits on the jac > 0 branch anchored at η̂, while beyond the
    // maximum r turns over and diverges as μ(η) explodes.
    //
    // Two safeguards make the scalar solve globally convergent to that root:
    //
    //   1. Anchor the iteration at η̂ itself, not at the classical one-step ALO
    //      predictor. At η̂ the weighted leverage a_ii ℓ''(η̂) < 1, so jac ≈ 1
    //      and we start strictly inside the correct basin; the brute-force
    //      n-fold reference solves the identical fixed point anchored at η̂.
    //      Seeding at the one-step predictor instead can land a high-leverage
    //      row *past* the interior maximum on the runaway branch, from which no
    //      Newton iteration returns (Poisson/log row 198: η ≈ 6.3, r ≈ −577).
    //
    //   2. Backtrack on the merit ½r(η)². The Newton direction d = −r/jac
    //      satisfies (½r²)'·d = r·jac·(−r/jac) = −r² < 0 for any finite nonzero
    //      jac, so halving the step until |r| strictly decreases never leaves
    //      the basin even if a full step would overshoot the maximum.
    let residual_and_jac = |eta: f64| -> Result<(f64, f64, f64), AloExactScalarError> {
        let (ell_prime, ell_double) =
            score_curvature(eta).map_err(|error| AloExactScalarError::EvaluationFailed {
                eta,
                reason: error.to_string(),
            })?;
        if !ell_prime.is_finite() || !ell_double.is_finite() {
            return Err(AloExactScalarError::NonFiniteScoreCurvature {
                eta,
                ell_prime,
                ell_double,
            });
        }
        let score_step = a_ii * ell_prime;
        let residual = eta - eta_hat - score_step;
        let jacobian = 1.0 - a_ii * ell_double;
        let tolerance = alo_scalar_residual_allowance(eta, eta_hat, score_step);
        if !score_step.is_finite()
            || !residual.is_finite()
            || !jacobian.is_finite()
            || !tolerance.is_finite()
        {
            return Err(AloExactScalarError::NonFiniteStep {
                eta,
                residual,
                jacobian,
                next: f64::NAN,
            });
        }
        Ok((residual, jacobian, tolerance))
    };

    let mut eta = eta_hat;
    let (mut residual, mut jac, mut tolerance) = residual_and_jac(eta)?;
    for _ in 0..ALO_EXACT_SCALAR_MAX_ITERS {
        if residual.abs() <= tolerance {
            return Ok(eta);
        }
        if jac == 0.0 || !jac.is_finite() {
            return Err(AloExactScalarError::DegenerateJacobian { eta, jacobian: jac });
        }
        let step = residual / jac;
        if !step.is_finite() {
            return Err(AloExactScalarError::NonFiniteStep {
                eta,
                residual,
                jacobian: jac,
                next: eta - step,
            });
        }
        // Backtracking line search: take the longest damped Newton step
        // 2^{-k} that strictly reduces the merit |r|. A trial whose
        // score/curvature evaluation errors (the runaway branch) is INVALID
        // (`Ok(None)`), so the search retreats toward η̂ without consulting
        // the merit test.
        let accepted = match backtracking_line_search::<_, Infallible>(
            BacktrackConfig {
                max_steps: ALO_EXACT_SCALAR_BACKTRACKS,
                ..BacktrackConfig::default()
            },
            |t| {
                let trial = eta - t * step;
                Ok(residual_and_jac(trial)
                    .ok()
                    .map(|(r_trial, j_trial, tol_trial)| {
                        (r_trial.abs(), (trial, r_trial, j_trial, tol_trial))
                    }))
            },
            |_t, merit| merit < residual.abs(),
        ) {
            Ok(result) => result,
            Err(never) => match never {},
        };
        let Some(step) = accepted else {
            break;
        };
        (eta, residual, jac, tolerance) = step.payload;
    }
    Err(AloExactScalarError::MaxIterations {
        iterations: ALO_EXACT_SCALAR_MAX_ITERS,
        residual,
        tolerance,
        eta,
    })
}

/// Evaluate `rhs' solution` after a residual-certified SPD solve. Neumaier
/// compensation is the allocation-free hot path. Only an overflowed,
/// underflowed, or non-positive cancellation result pays for the signed-log
/// reconstruction, which can distinguish an unrepresentable result from a
/// silently wrong sign without multiplying two huge coordinates first.
fn spd_quadratic_after_certified_solve(
    row: usize,
    rhs: ArrayView1<'_, f64>,
    solution: ArrayView1<'_, f64>,
) -> Result<f64, AloError> {
    if rhs.len() != solution.len() {
        return Err(AloError::LooComputationFailed {
            reason: format!(
                "ALO certified quadratic dimension mismatch at row {row}: rhs={}, solution={}",
                rhs.len(),
                solution.len()
            ),
        });
    }
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    let mut rhs_nonzero = false;
    let mut fast_path_finite = true;
    for (&left, &right) in rhs.iter().zip(solution.iter()) {
        if !left.is_finite() || !right.is_finite() {
            return Err(AloError::LooComputationFailed {
                reason: format!(
                    "ALO certified solve produced a non-finite quadratic coordinate at row {row}: rhs={left}, solution={right}"
                ),
            });
        }
        rhs_nonzero |= left != 0.0;
        let term = left * right;
        if !term.is_finite() {
            fast_path_finite = false;
            continue;
        }
        let next = sum + term;
        if !next.is_finite() {
            fast_path_finite = false;
            continue;
        }
        compensation += if sum.abs() >= term.abs() {
            (sum - next) + term
        } else {
            (term - next) + sum
        };
        sum = next;
    }
    let fast = sum + compensation;
    if !rhs_nonzero {
        return Ok(0.0);
    }
    if fast_path_finite && fast.is_finite() && fast > 0.0 {
        return Ok(fast);
    }

    let mut log_magnitudes = Vec::with_capacity(rhs.len());
    let mut signs = Vec::with_capacity(rhs.len());
    for (&left, &right) in rhs.iter().zip(solution.iter()) {
        if left == 0.0 || right == 0.0 {
            log_magnitudes.push(f64::NEG_INFINITY);
            signs.push(0.0);
        } else {
            log_magnitudes.push(left.abs().ln() + right.abs().ln());
            signs.push(left.signum() * right.signum());
        }
    }
    let (log_magnitude, sign) = signed_log_sum_exp(&log_magnitudes, &signs);
    if sign <= 0.0 || !log_magnitude.is_finite() {
        return Err(AloError::LooComputationFailed {
            reason: format!(
                "ALO SPD quadratic could not be represented as strictly positive at row {row}: sign={sign}, log_magnitude={log_magnitude}, fast_value={fast}"
            ),
        });
    }
    let value = log_magnitude.exp();
    if !value.is_finite() || value == 0.0 {
        return Err(AloError::LooComputationFailed {
            reason: format!(
                "ALO SPD quadratic lies outside the nonzero finite f64 range at row {row}: log_magnitude={log_magnitude}"
            ),
        });
    }
    Ok(value)
}

/// Sum `weights[i] * values[i]^2` without forming `values[i]^2` first.
/// Every term is non-negative by contract. A logarithmic reconstruction is
/// needed only if direct products or the positive accumulation leave f64.
fn finite_weighted_square_sum(
    observation: usize,
    weights: ArrayView1<'_, f64>,
    values: &[f64],
) -> Result<f64, AloError> {
    if weights.len() != values.len() {
        return Err(AloError::LooComputationFailed {
            reason: format!(
                "ALO sandwich quadratic dimension mismatch for observation {observation}: weights={}, values={}",
                weights.len(),
                values.len()
            ),
        });
    }
    let mut sum = 0.0_f64;
    let mut compensation = 0.0_f64;
    let mut has_mathematically_positive_term = false;
    let mut fast_path_finite = true;
    for (&weight, &value) in weights.iter().zip(values.iter()) {
        if !weight.is_finite() || weight < 0.0 || !value.is_finite() {
            return Err(AloError::LooComputationFailed {
                reason: format!(
                    "ALO sandwich quadratic has an invalid coordinate for observation {observation}: weight={weight}, value={value}"
                ),
            });
        }
        if weight == 0.0 || value == 0.0 {
            continue;
        }
        has_mathematically_positive_term = true;
        let term = (weight * value) * value;
        if !term.is_finite() || term == 0.0 {
            fast_path_finite = false;
            continue;
        }
        let next = sum + term;
        if !next.is_finite() {
            fast_path_finite = false;
            continue;
        }
        compensation += if sum.abs() >= term {
            (sum - next) + term
        } else {
            (term - next) + sum
        };
        sum = next;
    }
    let fast = sum + compensation;
    if !has_mathematically_positive_term {
        return Ok(0.0);
    }
    if fast_path_finite && fast.is_finite() && fast > 0.0 {
        return Ok(fast);
    }

    let mut log_magnitudes = Vec::with_capacity(values.len());
    let mut signs = Vec::with_capacity(values.len());
    for (&weight, &value) in weights.iter().zip(values.iter()) {
        if weight == 0.0 || value == 0.0 {
            log_magnitudes.push(f64::NEG_INFINITY);
            signs.push(0.0);
        } else {
            log_magnitudes.push(weight.ln() + 2.0 * value.abs().ln());
            signs.push(1.0);
        }
    }
    let (log_magnitude, sign) = signed_log_sum_exp(&log_magnitudes, &signs);
    let value = log_magnitude.exp();
    if sign != 1.0 || !value.is_finite() || value == 0.0 {
        return Err(AloError::LooComputationFailed {
            reason: format!(
                "ALO sandwich quadratic lies outside the positive finite f64 range for observation {observation}: sign={sign}, log_magnitude={log_magnitude}"
            ),
        });
    }
    Ok(value)
}

fn finite_nonnegative_product(
    row: usize,
    quantity: &'static str,
    left: f64,
    right: f64,
) -> Result<f64, AloError> {
    if !(left.is_finite() && left >= 0.0 && right.is_finite() && right >= 0.0) {
        return Err(AloError::LooComputationFailed {
            reason: format!(
                "ALO {quantity} requires finite non-negative factors at row {row}: left={left}, right={right}"
            ),
        });
    }
    if left == 0.0 || right == 0.0 {
        return Ok(0.0);
    }
    let direct = left * right;
    if direct.is_finite() && direct > 0.0 {
        return Ok(direct);
    }
    let log_magnitude = left.ln() + right.ln();
    let value = log_magnitude.exp();
    if !value.is_finite() || value == 0.0 {
        return Err(AloError::LooComputationFailed {
            reason: format!(
                "ALO {quantity} lies outside the positive finite f64 range at row {row}: log_magnitude={log_magnitude}"
            ),
        });
    }
    Ok(value)
}

fn finite_signed_product(
    row: usize,
    quantity: &'static str,
    left: f64,
    right: f64,
) -> Result<f64, AloError> {
    if !left.is_finite() || !right.is_finite() {
        return Err(AloError::LooComputationFailed {
            reason: format!(
                "ALO {quantity} requires finite factors at row {row}: left={left}, right={right}"
            ),
        });
    }
    if left == 0.0 || right == 0.0 {
        return Ok(0.0);
    }
    let direct = left * right;
    if direct.is_finite() && direct != 0.0 {
        return Ok(direct);
    }
    let log_magnitude = left.abs().ln() + right.abs().ln();
    let value = left.signum() * right.signum() * log_magnitude.exp();
    if !value.is_finite() || value == 0.0 {
        return Err(AloError::LooComputationFailed {
            reason: format!(
                "ALO {quantity} lies outside the nonzero finite f64 range at row {row}: sign={}, log_magnitude={log_magnitude}",
                left.signum() * right.signum()
            ),
        });
    }
    Ok(value)
}

const LEVERAGE_HIGH_THRESHOLD: f64 = 0.99;
const LEVERAGE_VERY_HIGH_THRESHOLD: f64 = 0.999;
const LEVERAGE_RATE_THRESHOLDS: [f64; 3] = [0.90, 0.95, 0.99];
const LEVERAGE_PERCENTILES: [f64; 3] = [0.50, 0.95, 0.99];
const MULTIBLOCK_ALO_MEMORY_BUDGET_BYTES: usize = 256 * 1024 * 1024;

/// Number of observation columns solved per blocked right-hand-side batch in the
/// scalar-leverage path. Sizes the reusable `(p, .)` and `(e_rank, .)` scratch
/// buffers so the dense multi-RHS solve stays BLAS-3 (good cache reuse) without
/// materializing all `n` columns at once. The final batch is the remainder.
const ALO_MAX_RHS_BLOCK_COLS: usize = 8192;

/// Choose the scalar ALO solve width from the actual live scratch footprint.
///
/// One column retains `X'H^-1` input (`p`), the certified solution and its
/// product/residual workspaces (conservatively `4p`), and `X H^-1 x_i` (`n`).
/// The old fixed width of 8192 made the supposedly blocked `n x width` scratch
/// consume multiple GiB on ordinary large fits. Saturating dimension arithmetic
/// makes even an impossible allocation request resolve to a one-column attempt
/// instead of wrapping the budget calculation.
#[inline]
fn alo_rhs_block_cols(n: usize, p: usize) -> usize {
    let scalars_per_col = n.saturating_add(p.saturating_mul(5)).max(1);
    let bytes_per_col = std::mem::size_of::<f64>().saturating_mul(scalars_per_col);
    (MULTIBLOCK_ALO_MEMORY_BUDGET_BYTES / bytes_per_col.max(1))
        .max(1)
        .min(ALO_MAX_RHS_BLOCK_COLS)
}

/// Roundoff multiplier for sign checks on local PSD quadratics. The deletion
/// systems themselves use an operation-count-derived formation and solve bound;
/// see [`identity_minus_product_lu_tolerance`].
const LOCAL_DELETE_SOLVE_ROUNDOFF_FACTOR: f64 = 8.0;

#[inline]
fn percentile_index(sample_size: usize, quantile: f64) -> usize {
    if sample_size <= 1 {
        return 0;
    }
    let max_index = sample_size - 1;
    ((quantile * max_index as f64).round() as usize).min(max_index)
}

#[inline]
fn percentile_from_sorted(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        0.0
    } else {
        sorted[percentile_index(sorted.len(), quantile)]
    }
}

#[inline]
fn compute_alo_diagnostics_from_pirls_impl(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_diagnostics_from_pirls_inner(base, y).map_err(EstimationError::from)
}

/// Resolve the multiplier on `H^-1` from the likelihood metadata and the
/// converged profiled-Gaussian residual geometry. This is deliberately not a
/// link switch: Gamma, Tweedie, Beta, NB, Poisson, Binomial, and fixed-scale
/// Gaussian already carry their full scale in the working Hessian and therefore
/// all have coefficient-covariance multiplier one.
fn alo_covariance_scale(base: &pirls::PirlsResult) -> Result<f64, AloError> {
    let dispersion = match (&base.likelihood.spec.response, base.likelihood.scale) {
        (ResponseFamily::Gaussian, LikelihoodScaleMetadata::ProfiledGaussian) => {
            let rss = base.deviance;
            if !(rss.is_finite() && rss >= 0.0) {
                return Err(AloError::InvalidInput {
                    reason: format!(
                        "ALO requires a finite non-negative profiled-Gaussian residual sum of squares; got {rss}"
                    ),
                });
            }
            let mut positive_rows = 0usize;
            for (row, &weight) in base.finalweights.iter().enumerate() {
                if !weight.is_finite() || weight < 0.0 {
                    return Err(AloError::WeightInvalid {
                        reason: format!(
                            "profiled-Gaussian ALO requires finite non-negative converged weights; row {row} has {weight}"
                        ),
                    });
                }
                positive_rows += usize::from(weight > 0.0);
            }
            let residual_dof = positive_rows as f64 - base.edf;
            if !(residual_dof.is_finite() && residual_dof > 0.0) {
                return Err(AloError::InvalidInput {
                    reason: format!(
                        "profiled-Gaussian ALO requires positive residual degrees of freedom; positive_rows={positive_rows}, edf={}, residual_dof={residual_dof}",
                        base.edf
                    ),
                });
            }
            let phi = rss / residual_dof;
            if !phi.is_finite() || (rss > 0.0 && phi == 0.0) {
                return Err(AloError::InvalidInput {
                    reason: format!(
                        "profiled-Gaussian ALO residual variance is not representable: rss={rss}, residual_dof={residual_dof}, phi={phi}"
                    ),
                });
            }
            Dispersion::estimated(phi).map_err(|error| AloError::InvalidInput {
                reason: format!("invalid profiled-Gaussian ALO dispersion: {error}"),
            })?
        }
        _ => dispersion_from_likelihood(&base.likelihood, None).map_err(|error| {
            AloError::InvalidInput {
                reason: format!("ALO could not resolve likelihood scale metadata: {error}"),
            }
        })?,
    };
    let scale = base
        .likelihood
        .coefficient_covariance_scale(dispersion.phi())
        .map_err(|error| AloError::InvalidInput {
            reason: format!("ALO could not resolve coefficient-covariance scale: {error}"),
        })?;
    if !(scale.is_finite() && scale > 0.0) {
        return Err(AloError::InvalidInput {
            reason: format!(
                "ALO coefficient covariance is unavailable at non-positive or non-finite scale {scale}"
            ),
        });
    }
    Ok(scale)
}

/// True when the fitted GLM uses a *curved* canonical link, so that the row NLL
/// score and curvature satisfy `ℓ_i'(η) = c_i(μ(η)−y_i)` and `ℓ_i''(η) = c_i μ'(η)`
/// with a single per-row scale `c_i = (prior weight)/φ`. This is the exact
/// condition under which the frozen-curvature ALO scalar fixed point matches
/// the leave-`i`-out refit; only these families enable the exact refinement.
///
/// Gaussian identity is canonical too, but its per-row curvature is *constant*
/// (`μ'(η) ≡ 1`), so the classical Sherman–Morrison one-step ALO is already the
/// exact frozen-Hessian leave-`i`-out solution. Routing it through the scalar
/// Newton closure would only add an O(n) nonlinear solve to diagnostics and
/// quality sweeps without changing the answer, so it is excluded here and falls
/// back to the (exact, for this family) one-step formula.
fn alo_link_needs_exact_curvature_refinement(likelihood: &gam_problem::GlmLikelihoodSpec) -> bool {
    use gam_problem::ResponseFamily;
    matches!(
        (&likelihood.spec.response, likelihood.link_function()),
        (ResponseFamily::Binomial, LinkFunction::Logit)
            | (ResponseFamily::Poisson, LinkFunction::Log)
    )
}

fn compute_alo_diagnostics_from_pirls_inner(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
) -> Result<AloDiagnostics, AloError> {
    let x_dense_arc = base
        .x_transformed
        .try_to_dense_arc("ALO diagnostics require dense transformed design")
        .map_err(|reason| AloError::DesignDegenerate { reason })?;
    let x_dense = x_dense_arc.as_ref();
    let n = x_dense.nrows();
    if y.len() != n {
        return Err(AloError::InvalidInput {
            reason: format!(
                "ALO response length must match the design row count; got {} responses for {n} rows",
                y.len()
            ),
        });
    }
    if alo_link_needs_exact_curvature_refinement(&base.likelihood) {
        for (row, &response) in y.iter().enumerate() {
            let valid = response.is_finite()
                && match &base.likelihood.spec.response {
                    ResponseFamily::Binomial => (0.0..=1.0).contains(&response),
                    ResponseFamily::Poisson => response >= 0.0,
                    _ => true,
                };
            if !valid {
                return Err(AloError::InvalidInput {
                    reason: format!(
                        "ALO canonical refinement received an invalid response at row {row}: {response}"
                    ),
                });
            }
        }
    }

    let phi = alo_covariance_scale(base)?;

    // ALO needs the exact penalized Hessian materialized densely for chunked,
    // residual-certified SPD solves. The PIRLS export path validates the matrix
    // instead of falling back to a numerical Hessian approximation.
    let h_dense_for_alo = base
        .dense_stabilizedhessian_transformed(
            "ALO diagnostics require exact dense stabilized penalized Hessian",
        )
        .map_err(|e| match e {
            EstimationError::InvalidInput(reason) => AloError::InvalidInput { reason },
            other => AloError::InvalidInput {
                reason: format!("{other:?}"),
            },
        })?;

    // Exact frozen-curvature ALO refinement for canonical-link GLMs.
    //
    // For a canonical link the row NLL score and curvature are
    //   ℓ_i'(η)  = c_i · (μ(η) − y_i),     ℓ_i''(η) = c_i · μ'(η),
    // with c_i = (prior weight)/φ recovered from the converged geometry as
    // c_i = W_H[i] / μ'(η̂_i) (since W_H[i] = c_i μ'(η̂_i) at convergence).
    // Supplying this evaluator lets `compute_alo_from_input_inner` solve the
    // leave-i-out scalar fixed point η = η̂_i + a_ii ℓ_i'(η) exactly instead of
    // taking a single Newton step, removing the first-order linearization error
    // that dominates on small-n, strongly curved likelihoods (binomial logit).
    //
    // Restricted to canonical links because only there does the observed
    // curvature carried by the frozen Hessian (W_H) coincide with c_i μ'(η) for
    // every trial η; non-canonical links retain the classical one-step ALO.
    // Per-row scale c_i = W_H[i]/μ'(η̂_i). This is an exact ratio, not a
    // thresholded one: tiny positive curvature remains informative. If both
    // numerator and derivative are exactly zero, the row has no representable
    // local influence and c_i is exactly zero; every other nonrepresentable
    // ratio is an explicit error.
    let canonical_scale: Option<Array1<f64>> = if alo_link_needs_exact_curvature_refinement(
        &base.likelihood,
    ) {
        let mut c = Array1::<f64>::zeros(n);
        for i in 0..n {
            let dmu = base.solve_dmu_deta[i];
            let w_h = base.finalweights[i];
            if !dmu.is_finite() || !w_h.is_finite() || dmu < 0.0 || w_h < 0.0 {
                return Err(AloError::WeightInvalid {
                    reason: format!(
                        "canonical ALO requires finite non-negative local derivative and curvature; row {i} has dmu_deta={dmu}, weight={w_h}"
                    ),
                });
            }
            let scale = if dmu == 0.0 {
                if w_h == 0.0 {
                    0.0
                } else {
                    return Err(AloError::LooComputationFailed {
                        reason: format!(
                            "canonical ALO scale is undefined at row {i}: nonzero curvature {w_h} divided by zero inverse-link derivative"
                        ),
                    });
                }
            } else {
                w_h / dmu
            };
            if !scale.is_finite() || scale < 0.0 || (w_h > 0.0 && scale == 0.0) {
                return Err(AloError::LooComputationFailed {
                    reason: format!(
                        "canonical ALO scale is not representable at row {i}: weight={w_h}, dmu_deta={dmu}, scale={scale}"
                    ),
                });
            }
            c[i] = scale;
        }
        Some(c)
    } else {
        None
    };

    let inv_link_for_closure = base.likelihood.spec.link.clone();
    let score_curvature_closure = canonical_scale.as_ref().map(|scale| {
        move |i: usize, eta: f64| -> Result<(f64, f64), AloError> {
            let (mu, dmu) = crate::mixture_link::inverse_link_mu_d1_for_inverse_link(
                &inv_link_for_closure,
                eta,
            )
            .map_err(|error| AloError::LooComputationFailed {
                reason: format!(
                    "ALO inverse-link evaluation failed at row {i}, eta={eta}: {error}"
                ),
            })?;
            let c_i = scale[i];
            let score = c_i * (mu - y[i]);
            let curvature = c_i * dmu;
            if !score.is_finite() || !curvature.is_finite() {
                return Err(AloError::LooComputationFailed {
                    reason: format!(
                        "ALO canonical row geometry is not representable at row {i}, eta={eta}: score={score}, curvature={curvature}"
                    ),
                });
            }
            Ok((score, curvature))
        }
    });
    let score_curvature_ref: Option<&AloScalarScoreCurvature> = score_curvature_closure
        .as_ref()
        .map(|f| f as &AloScalarScoreCurvature);

    // Build model-agnostic AloInput from PIRLS geometry, then delegate.
    // #1868: the PIRLS row fields are now shared `ArcArray1`; `AloInput` borrows
    // `&Array1`, so materialise owned copies for this cold post-fit inference
    // path (ALO runs once after the fit, not per κ trial).
    let alo_working_response = base.solveworking_response.to_owned();
    let alo_final_eta = base.final_eta.to_owned();
    let alo_final_offset = base.final_offset.to_owned();
    let input = AloInput {
        design: x_dense,
        penalized_hessian: &h_dense_for_alo,
        hessian_weights: base.final_weights_signed(),
        score_weights: base.solve_weights_psd(),
        working_response: &alo_working_response,
        eta: &alo_final_eta,
        offset: &alo_final_offset,
        phi,
        score_curvature: score_curvature_ref,
    };

    let result = compute_alo_from_input_inner(&input)?;

    // PIRLS-specific post-hoc leverage diagnostics logging.
    log_leverage_diagnostics(&result.leverage, phi);

    Ok(result)
}

/// Log detailed leverage percentile diagnostics for a completed ALO computation.
fn log_leverage_diagnostics(leverage: &Array1<f64>, phi: f64) {
    let n = leverage.len();
    if n == 0 {
        return;
    }

    let mut invalid_count = 0usize;
    let mut high_leverage_count = 0usize;
    let mut threshold_counts = [0usize; LEVERAGE_RATE_THRESHOLDS.len()];
    let mut finite_leverage = Vec::with_capacity(n);

    for (obs, &ai) in leverage.iter().enumerate() {
        if ai.is_finite() {
            finite_leverage.push(ai);
        }

        // Signed observed curvature permits exact negative row leverages for
        // non-canonical links. They are not invalid: the corresponding ALO
        // denominator `1-a_ii` is more strongly positive. Non-finite values
        // remain invalid, while positive near-one leverage is the instability
        // diagnostic of interest.
        if !ai.is_finite() {
            invalid_count += 1;
            log::warn!("[GAM ALO] invalid leverage at i={}, a_ii={:.6e}", obs, ai);
        } else if ai > LEVERAGE_HIGH_THRESHOLD {
            high_leverage_count += 1;
            if ai > LEVERAGE_VERY_HIGH_THRESHOLD {
                log::warn!("[GAM ALO] very high leverage at i={}, a_ii={:.6e}", obs, ai);
            }
        }

        for (idx, threshold) in LEVERAGE_RATE_THRESHOLDS.iter().enumerate() {
            if ai > *threshold {
                threshold_counts[idx] += 1;
            }
        }
    }

    if invalid_count > 0 || high_leverage_count > 0 {
        log::warn!(
            "[GAM ALO] leverage diagnostics: {} invalid values, {} high values (>0.99)",
            invalid_count,
            high_leverage_count
        );
    }

    finite_leverage.sort_by(f64::total_cmp);

    let finite_n = finite_leverage.len();
    let a_mean = if finite_n > 0 {
        finite_leverage.iter().copied().sum::<f64>() / finite_n as f64
    } else {
        0.0
    };
    let a_median = percentile_from_sorted(&finite_leverage, LEVERAGE_PERCENTILES[0]);
    let a_p95 = percentile_from_sorted(&finite_leverage, LEVERAGE_PERCENTILES[1]);
    let a_p99 = percentile_from_sorted(&finite_leverage, LEVERAGE_PERCENTILES[2]);
    let a_max = finite_leverage.last().copied().unwrap_or(0.0);

    // Routine per-ALO leverage summary: a diagnostic snapshot, not an
    // anomaly. Emitted at `info!` so it is visible when the host raises
    // verbosity (CLI `-v`; `gamfit.set_log_level("info")`) but silent at the
    // default `Warn` level (genuine anomalies — invalid / very
    // high leverage — are logged at `warn!` above and stay visible). This
    // line fires once per ALO computation, which recurs across the outer
    // smoothing loop, so at `warn!` it was a dominant source of stderr noise
    // on perfectly healthy fits (#1689).
    log::info!(
        "[GAM ALO] leverage: n={}, mean={:.3e}, median={:.3e}, p95={:.3e}, p99={:.3e}, max={:.3e}",
        n,
        a_mean,
        a_median,
        a_p95,
        a_p99,
        a_max
    );
    log::info!(
        "[GAM ALO] high-leverage: a>0.90: {:.2}%, a>0.95: {:.2}%, a>0.99: {:.2}%, dispersion phi={:.3e}",
        100.0 * (threshold_counts[0] as f64) / n as f64,
        100.0 * (threshold_counts[1] as f64) / n as f64,
        100.0 * (threshold_counts[2] as f64) / n as f64,
        phi
    );
}

/// Model-agnostic input for ALO diagnostics.
///
/// Any model with a design matrix, penalized Hessian, and IRLS geometry can
/// compute ALO leverages and leave-one-out predictions. This decouples ALO
/// from the single-block PIRLS solver and enables diagnostics for GAMLSS,
/// survival, and joint models.
pub struct AloInput<'a> {
    /// Dense design matrix X (n × p).
    pub design: &'a Array2<f64>,
    /// Penalized Hessian H = X'WX + S(λ) at convergence (p × p).
    pub penalized_hessian: &'a Array2<f64>,
    /// Hessian-side IRLS weights W_H at convergence (n). Sign-honest: for
    /// non-canonical links the observed-information diagonal can have negative
    /// entries, so the typed [`SignedWeightsView`] is the contract here. PSD
    /// callers needing to promote (e.g. the canonical-link case where the
    /// caller has discharged W_H ≥ 0 algebraically) can route through
    /// `SignedWeightsView::as_psd()` at the consumer.
    pub hessian_weights: SignedWeightsView<'a>,
    /// Score-side IRLS weights W_S paired with `working_response` (n).
    /// PSD-by-construction: the score-side Fisher weights `h'²/(φ V(μ)) ≥ 0`.
    pub score_weights: PsdWeightsView<'a>,
    /// IRLS working response at convergence (n).
    pub working_response: &'a Array1<f64>,
    /// Fitted linear predictor η̂ (n).
    pub eta: &'a Array1<f64>,
    /// Offset vector (n). Pass zeros if no offset.
    pub offset: &'a Array1<f64>,
    /// Dispersion parameter φ. For non-Gaussian families this is 1.0.
    pub phi: f64,
    /// Optional per-row score/curvature evaluator `(i, η) → (ℓ_i'(η), ℓ_i''(η))`.
    ///
    /// When supplied, the leave-`i`-out predictor is obtained by solving the
    /// frozen-curvature scalar fixed point `η = η̂_i + a_ii ℓ_i'(η)` to
    /// convergence (see [`alo_eta_exact_frozen_curvature`]) instead of taking a
    /// single Newton step. This eliminates the first-order linearization error
    /// that the one-step ALO incurs on small-`n`, strongly curved likelihoods
    /// (e.g. binomial logistic regression). Non-convergence or invalid scalar
    /// Newton geometry is returned as an ALO error. When `None`, the classical
    /// single-Newton-step ALO formula is used. The evaluator must be consistent
    /// with `hessian_weights` at convergence: `ℓ_i''(η̂_i) = W_H[i]` and
    /// `ℓ_i'(η̂_i) = W_S[i]·((η̂_i−o_i) − (z_i−o_i))`.
    pub score_curvature: Option<&'a AloScalarScoreCurvature<'a>>,
}

impl<'a> AloInput<'a> {
    /// Build an `AloInput` from `FitGeometry` and associated vectors.
    pub fn from_geometry(
        geom: &'a FitGeometry,
        design: &'a Array2<f64>,
        eta: &'a Array1<f64>,
        offset: &'a Array1<f64>,
        phi: f64,
    ) -> Self {
        // FitGeometry stores one working-weight vector, so this constructor is
        // exact only when the score- and Hessian-side IRLS weights coincide
        // (canonical-link case where Fisher == Observed). In that path the
        // diagonal is the Fisher weight `h'²/(φ V(μ)) ≥ 0`, so the PSD
        // obligation is discharged algebraically without a runtime scan;
        // `as_signed()` re-views the same buffer for the Hessian-side slot.
        let psd_w = PsdWeightsView::from_view_unchecked(geom.working_weights.view());
        Self {
            design,
            penalized_hessian: &geom.penalized_hessian,
            hessian_weights: psd_w.as_signed(),
            score_weights: psd_w,
            working_response: &geom.working_response,
            eta,
            offset,
            phi,
            score_curvature: None,
        }
    }

    /// Build an `AloInput` from an exact saved penalized Hessian plus externally
    /// supplied working weights / working response.
    ///
    /// The row-sized IRLS working vectors are *derived* quantities: at
    /// convergence they are deterministic functions of the linear predictor
    /// `η̂ = Xβ̂`, the response `y`, and the family (`w_i = h'(η̂_i)²/(φ V(μ̂_i))·
    /// prior_i`, `z_i = η̂_i + (y_i−μ̂_i)/h'(η̂_i)`). A saved-model consumer
    /// reconstructs them from the saved `β` by replaying the same PIRLS
    /// working-state update the fit used, then feeds them here. The precision
    /// comes from the canonical fit's exact unscaled Hessian accessor; callers
    /// do not need a second `FitGeometry` wrapper or a covariance inversion.
    ///
    /// Same canonical (Fisher == Observed) contract as [`from_geometry`]: the
    /// supplied `working_weights` are the score-side Fisher weights and are
    /// re-viewed for the Hessian-side slot via `as_signed()`.
    ///
    /// [`from_geometry`]: AloInput::from_geometry
    pub fn from_penalized_hessian_with_working_state(
        penalized_hessian: &'a Array2<f64>,
        design: &'a Array2<f64>,
        eta: &'a Array1<f64>,
        offset: &'a Array1<f64>,
        phi: f64,
        working_weights: &'a Array1<f64>,
        working_response: &'a Array1<f64>,
    ) -> Self {
        let psd_w = PsdWeightsView::from_view_unchecked(working_weights.view());
        Self {
            design,
            penalized_hessian,
            hessian_weights: psd_w.as_signed(),
            score_weights: psd_w,
            working_response,
            eta,
            offset,
            phi,
            score_curvature: None,
        }
    }
}

/// Compute ALO diagnostics from model-agnostic inputs.
///
/// This is the generalized entry point that works for any model type.
/// For standard single-block GAMs, prefer `compute_alo_diagnostics_from_fit`
/// which automatically extracts the PIRLS geometry (including sandwich SE).
pub fn compute_alo_from_input(input: &AloInput) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_from_input_inner(input).map_err(EstimationError::from)
}

fn compute_alo_from_input_inner(input: &AloInput) -> Result<AloDiagnostics, AloError> {
    let x_dense = input.design;
    let n = x_dense.nrows();
    let p = x_dense.ncols();
    // Bind the underlying ArrayView1 once so the loop body can index and
    // borrow as before; the sign-character contract lives in the
    // `AloInput` field types, not in this local binding.
    let w_h = input.hessian_weights.view();
    let w_s = input.score_weights.view();

    validate_alo_solve_setup(input, n, p)?;

    let factor = certified_spd_factorize(input.penalized_hessian, "ALO penalized Hessian")
        .map_err(|error| AloError::InvalidInput {
            reason: format!(
                "ALO requires an unperturbed positive-definite penalized Hessian with a certified solve: {error}"
            ),
        })?;

    let xt = x_dense.t();
    let phi = input.phi;

    let mut aii = Array1::<f64>::zeros(n);
    let mut x_hinv_x_diag = Array1::<f64>::zeros(n);
    let mut se_bayes = Array1::<f64>::zeros(n);
    let mut se_sandwich = Array1::<f64>::zeros(n);

    let block_cols = alo_rhs_block_cols(n, p);
    // Allocate the RHS scratch in column-major (Fortran) order so its column
    // slices are contiguous and align with faer's column-major solve output.
    // This removes redundant `xrow = x_dense.row(obs)` indirection inside the
    // per-observation loop: rhs_chunk_buf already holds X^T at the right cols.
    let mut rhs_chunk_buf = Array2::<f64>::zeros((p, block_cols).f());
    // Reusable faer column-major buffer for X*S, where S = H^{-1}X_i for the
    // current RHS chunk. The sandwich SE uses the same frozen-curvature meat
    // as the exact LOO reference, `X' W_S X`, directly; no redundant penalty
    // root or ridge surrogate is carried through this API.
    let mut xs_chunk_storage = FaerMat::<f64>::zeros(n, block_cols);
    let x_dense_view = FaerArrayView::new(x_dense);

    for chunk_start in (0..n).step_by(block_cols) {
        let chunk_end = (chunk_start + block_cols).min(n);
        let width = chunk_end - chunk_start;

        rhs_chunk_buf
            .slice_mut(s![.., ..width])
            .assign(&xt.slice(s![.., chunk_start..chunk_end]));

        let rhs_chunkview = rhs_chunk_buf.slice(s![.., ..width]);
        let rhs_chunk = rhs_chunkview.to_owned();
        let (s_chunk, _solve_certificate) = factor.solve_matrix(&rhs_chunk).map_err(|error| {
            AloError::LooComputationFailed {
                reason: format!(
                    "ALO penalized-Hessian solve could not be certified for rows {chunk_start}..{chunk_end}: {error}"
                ),
            }
        })?;
        let s_chunk_view = FaerArrayView::new(&s_chunk);

        let mut xs_target = xs_chunk_storage.as_mut().subcols_mut(0, width);
        matmul(
            xs_target.rb_mut(),
            Accum::Replace,
            x_dense_view.as_ref(),
            s_chunk_view.as_ref(),
            1.0,
            Par::Seq,
        );

        let rhs_view = rhs_chunk_buf.slice(s![.., ..width]);

        for local_col in 0..width {
            let obs = chunk_start + local_col;
            // The RHS stays column-major; the certified solution is indexed
            // with its native ndarray strides so this path does not assume a
            // storage order chosen inside the factor API.
            let rhs_col = rhs_view.column(local_col);
            let solution_col = s_chunk.column(local_col);
            let x_hinv_x = spd_quadratic_after_certified_solve(obs, rhs_col, solution_col)?;
            // The bread uses the observed Hessian surface. For a non-canonical
            // link W_H is signed, so the exact row leverage W_H,i x_i'H^-1x_i
            // can be negative even though the assembled penalized H is SPD.
            // Projecting that row to zero changes both the ALO denominator and
            // corrected predictor; only the separate score-covariance meat is
            // PSD (and uses W_S below).
            let ai = finite_signed_product(obs, "leverage", w_h[obs], x_hinv_x)?;
            aii[obs] = ai;
            x_hinv_x_diag[obs] = x_hinv_x;

            let var_bayes = finite_nonnegative_product(obs, "Bayesian variance", phi, x_hinv_x)?;
            let xs_slice = xs_chunk_storage.col_as_slice(local_col);
            // Sandwich meat is the SCORE covariance Xᵀ diag(W_S) X (Fisher,
            // PSD by construction), not the observed-information Hessian
            // weight W_H. The scale-safe sum preserves that non-negative
            // contract without projecting a signed result after the fact.
            let meat_quad = finite_weighted_square_sum(obs, w_s, xs_slice)?;
            let var_sandwich =
                finite_nonnegative_product(obs, "sandwich variance", phi, meat_quad)?;

            se_bayes[obs] = var_bayes.sqrt();
            se_sandwich[obs] = var_sandwich.sqrt();
        }
    }

    let eta_hat = input.eta;
    let z = input.working_response;
    let offset = input.offset;

    use rayon::prelude::*;
    let eta_tilde_vec: Vec<f64> = (0..n)
        .into_par_iter()
        .map(|i| {
            let denom_raw = 1.0 - aii[i];
            if denom_raw == 0.0 || !denom_raw.is_finite() {
                return Err(AloError::LooComputationFailed {
                    reason: format!(
                        "ALO deletion denominator is not invertible at row {i}: a_ii={:.6e}, 1-a_ii={:.6e}",
                        aii[i], denom_raw
                    ),
                });
            }
            let one_step = alo_eta_updatewith_offset(
                eta_hat[i],
                z[i],
                offset[i],
                x_hinv_x_diag[i],
                w_s[i],
                denom_raw,
            );
            // When the family score/curvature evaluator is supplied, solve the
            // exact frozen-curvature leave-i-out fixed point (anchored at η̂_i,
            // the basin that limits to the in-sample fit) instead of taking the
            // single Newton step. a_ii here is the unweighted influence
            // x_i^T H^{-1} x_i (= x_hinv_x_diag[i]); the per-row curvature
            // W_H[i] = ℓ_i''(η̂_i) is folded into the scalar fixed point via
            // score_curvature. Non-canonical links fall back to `one_step`.
            let v = if let Some(score_curvature) = input.score_curvature {
                alo_eta_exact_frozen_curvature(
                    eta_hat[i],
                    x_hinv_x_diag[i],
                    &|eta| score_curvature(i, eta),
                )
                .map_err(|err| AloError::LooComputationFailed {
                    reason: format!(
                        "ALO exact frozen-curvature solve failed at row {i}: {err}"
                    ),
                })?
            } else {
                one_step
            };
            if !v.is_finite() {
                return Err(AloError::LooComputationFailed {
                    reason: format!("ALO eta_tilde is not finite at row {i}: eta_tilde={v}"),
                });
            }
            Ok(v)
        })
        .collect::<Result<_, _>>()?;
    let eta_tilde = Array1::from(eta_tilde_vec);

    Ok(AloDiagnostics {
        eta_tilde,
        se_bayes,
        se_sandwich,
        leverage: aii,
    })
}

fn validate_alo_solve_setup(input: &AloInput, n: usize, p: usize) -> Result<(), AloError> {
    let h = input.penalized_hessian;
    if h.nrows() != p || h.ncols() != p {
        return Err(AloError::InvalidInput {
            reason: format!(
                "ALO diagnostics require a dense exact penalized Hessian with shape {p}x{p}; got {}x{}",
                h.nrows(),
                h.ncols()
            ),
        });
    }
    let vector_lengths = [
        ("hessian_weights", input.hessian_weights.len()),
        ("score_weights", input.score_weights.len()),
        ("working_response", input.working_response.len()),
        ("eta", input.eta.len()),
        ("offset", input.offset.len()),
    ];
    for (name, len) in vector_lengths {
        if len != n {
            return Err(AloError::InvalidInput {
                reason: format!("ALO diagnostics require {name} length {n}; got {len}"),
            });
        }
    }
    if input.hessian_weights.view().iter().any(|v| !v.is_finite()) {
        return Err(AloError::WeightInvalid {
            reason: "ALO diagnostics require finite Hessian-side weights".to_string(),
        });
    }
    if let Some((row, value)) = input
        .score_weights
        .view()
        .iter()
        .copied()
        .enumerate()
        .find(|(_, value)| !value.is_finite() || *value < 0.0)
    {
        return Err(AloError::WeightInvalid {
            reason: format!(
                "ALO diagnostics require finite non-negative score-side weights; row {row} has {value:?}"
            ),
        });
    }
    if input.working_response.iter().any(|v| !v.is_finite()) {
        return Err(AloError::WeightInvalid {
            reason: "ALO diagnostics require finite working responses".to_string(),
        });
    }
    if input.eta.iter().any(|v| !v.is_finite()) || input.offset.iter().any(|v| !v.is_finite()) {
        return Err(AloError::InvalidInput {
            reason: "ALO diagnostics require finite linear predictors and offsets".to_string(),
        });
    }
    if !input.phi.is_finite() || input.phi <= 0.0 {
        return Err(AloError::InvalidInput {
            reason: format!(
                "ALO diagnostics require positive finite dispersion phi; got {}",
                input.phi
            ),
        });
    }
    Ok(())
}

/// Compute ALO diagnostics (eta_tilde, SE, leverage) from a fitted GAM result.
pub fn compute_alo_diagnostics_from_fit(
    fit: &UnifiedFitResult,
    y: ArrayView1<f64>,
) -> Result<AloDiagnostics, EstimationError> {
    let pirls = fit
        .artifacts
        .pirls
        .as_ref()
        .ok_or_else(|| AloError::InvalidInput {
            reason:
                "ALO diagnostics require a PIRLS-backed fit; this fit does not expose PIRLS geometry"
                    .to_string(),
        })
        .map_err(EstimationError::from)?;
    compute_alo_diagnostics_from_pirls_impl(pirls, y)
}

/// Compute ALO diagnostics from a `UnifiedFitResult`.
///
/// Extracts `FitGeometry` from `unified.geometry`, builds an `AloInput`
/// via `from_geometry`, and delegates to `compute_alo_from_input`.
/// This avoids requiring a full `UnifiedFitResult` with PIRLS artifacts.
pub fn compute_alo_diagnostics_from_unified(
    unified: &UnifiedFitResult,
    design: &Array2<f64>,
    eta: &Array1<f64>,
    offset: &Array1<f64>,
    phi: f64,
) -> Result<AloDiagnostics, EstimationError> {
    let geom = unified
        .geometry
        .as_ref()
        .ok_or_else(|| AloError::InvalidInput {
            reason: "UnifiedFitResult does not contain working-set geometry; \
             ALO diagnostics require geometry at convergence"
                .to_string(),
        })
        .map_err(EstimationError::from)?;
    let input = AloInput::from_geometry(geom, design, eta, offset, phi);
    compute_alo_from_input(&input)
}

/// Compute ALO diagnostics from a PIRLS result for lower-level callers.
pub fn compute_alo_diagnostics_from_pirls(
    base: &pirls::PirlsResult,
    y: ArrayView1<f64>,
) -> Result<AloDiagnostics, EstimationError> {
    compute_alo_diagnostics_from_pirls_impl(base, y)
}

/// Exact (one-step) case-deletion influence from a converged PIRLS fit, via
/// the one `FitSensitivity` operator (#935).
///
/// This is the diagnostic the sensitivity operator's `case_deletion` channel
/// was built to expose but had no production entry point for: per-observation
/// dfbetas `β̂ − β̂₍ᵢ₎`, hat-value leverage `h_ii = w_i x_iᵀ H⁻¹ x_i`, and
/// Cook's distance. It is the same factored inverse the REML gradient (IFT),
/// ALO, and the Riesz debias already contract — built once at the optimum,
/// asked in the leave-one-out direction — so no call site can disagree about
/// which `H⁻¹` is meant (the bug class #935 dismantles).
///
/// The penalized Hessian, design, working weights `w_i = W_H[i]` and working
/// residual `z_i − η̂_i` are read straight from the converged geometry — the
/// same PIRLS state [`compute_alo_diagnostics_from_pirls`] consumes — so the
/// IRLS reduction `scale = w_i r_i / (1 − h_ii)` is exact for the Gaussian
/// identity link and the one-step Newton deletion for canonical-link GLMs.
/// Returns `None` (rather than emitting `∞`) for any observation whose
/// leverage is one, or if the dense Hessian / design is unavailable.
pub fn compute_case_deletion_from_pirls(
    base: &pirls::PirlsResult,
) -> Result<Option<crate::sensitivity::CaseDeletionInfluence>, EstimationError> {
    let x_dense_arc = base
        .x_transformed
        .try_to_dense_arc("case-deletion diagnostics require dense transformed design")
        .map_err(|reason| EstimationError::InvalidInput(reason))?;
    let x_dense = x_dense_arc.as_ref();
    let n = x_dense.nrows();
    let p = x_dense.ncols();
    if n == 0 || p == 0 {
        return Ok(None);
    }

    let phi = alo_covariance_scale(base).map_err(EstimationError::from)?;

    // The same dense stabilized penalized Hessian ALO materializes; the one
    // factored inverse every sensitivity channel shares.
    let h_dense = base
        .dense_stabilizedhessian_transformed(
            "case-deletion diagnostics require exact dense stabilized penalized Hessian",
        )
        .map_err(|e| match e {
            EstimationError::InvalidInput(reason) => EstimationError::InvalidInput(reason),
            other => EstimationError::InvalidInput(format!("{other:?}")),
        })?;

    let factor = match h_dense.cholesky(faer::Side::Lower) {
        Ok(f) => f,
        // A non-SPD stabilized Hessian means the optimum is rank-deficient in a
        // way the dense Cholesky case-deletion path cannot invert; decline
        // rather than fabricate an influence diagnostic.
        Err(_) => return Ok(None),
    };

    // Working weights and working residual straight from the IRLS reduction:
    // w_i = W_H[i] and r_i = z_i − η̂_i, so w_i r_i is the working score the
    // closed-form deletion `scale = w_i r_i / (1 − h_ii)` consumes.
    let working_weights = base.finalweights.clone();
    let working_residual = &base.solveworking_response - &base.final_eta;

    let sensitivity = crate::sensitivity::FitSensitivity::from_faer_cholesky(&factor, p);
    Ok(sensitivity.case_deletion(
        x_dense,
        working_weights.view(),
        working_residual.view(),
        phi,
    ))
}

// Multi-block ALO for multi-predictor models (GAMLSS, survival, joint)

/// Diagnostics returned by multi-block ALO.
#[derive(Debug, Clone)]
pub struct MultiBlockAloDiagnostics {
    /// Corrected linear predictors η̃^{(-i)} for each observation.
    /// Outer length = n_obs, inner length = n_coordinates (B).
    pub eta_tilde: Vec<Array1<f64>>,
    /// Per-observation leverage tr(H_ii) where H_ii is the B×B hat-matrix block.
    pub leverage: Array1<f64>,
    /// Per-observation ALO variance diagonals: for each observation i,
    /// Var(Δη_i) ≈ A_i (I - W_i A_i)⁻¹ C_i (I - A_i W_i)⁻¹ A_iᵀ,
    /// where C_i is the score covariance (not assumed equal to W_i).
    /// Outer length = n_obs, inner length = n_coordinates (B) containing the
    /// diagonal entries of the variance matrix.
    pub alo_variance: Vec<Array1<f64>>,
    /// Cook-type ALO influence: D_i = Δη_iᵀ C_i Δη_i.
    /// Length = n_obs.
    pub cook_distance: Array1<f64>,
}

/// Model-agnostic input for multi-predictor ALO diagnostics.
///
/// Generalises [`AloInput`] to models with B > 1 linear predictors per
/// observation (e.g. location-scale GAMLSS with B=2, or survival models
/// with time-dependent predictors).
///
/// # Mathematical setup
///
/// For observation i the per-observation Jacobian is a B × p_tot matrix X_i.
/// Row b embeds row i of `coordinate_designs[b]` at
/// `coordinate_coefficient_ranges[b]`. Ranges may overlap: this is required
/// for risk-set and latent-variable coordinates that share coefficients. The
/// joint hat-matrix block is
///
///   H_ii = X_i H⁻¹ X_iᵀ W_i     (B × B)
///
/// where H = Σ_i X_iᵀ W_i X_i + S is the total penalized Hessian and W_i
/// is the B × B per-observation weight matrix (negative Hessian of the
/// log-likelihood w.r.t. the B predictors at observation i).
///
/// The ALO leave-one-out correction is
///
///   Δη_i^ALO = A_i (I_B − W_i A_i)⁻¹ s_i
///
/// where A_i = X_i H⁻¹ X_iᵀ (the B×B per-observation influence matrix),
/// W_i is the B×B per-observation NLL Hessian, and
/// s_i = ∇_{η_i} NLL_i(η̂_i) is the B-dimensional score vector.
/// This is algebraically equivalent to (I_B − H_ii)⁻¹ H_ii W_i⁻¹ s_i
/// but does NOT require W_i⁻¹, which is critical when W_i is singular
/// (e.g. at boundary observations in survival models).
/// For B = 1 this reduces to the classical scalar ALO formula.
pub struct MultiBlockAloInput<'a> {
    /// Number of observations.
    pub n_obs: usize,
    /// Number of local likelihood coordinates per observation (B).
    pub n_coordinates: usize,
    /// B possibly operator-backed local design matrices, each n_obs × p_b.
    /// ALO materializes only bounded row chunks.
    pub coordinate_designs: &'a [DesignMatrix],
    /// Parameter alignment for each local design. Range b has length p_b and
    /// identifies the columns of the saved p_tot-dimensional Hessian touched
    /// by coordinate b. Ranges may overlap and need not cover every parameter.
    pub coordinate_coefficient_ranges: &'a [Range<usize>],
    /// Exact unscaled penalized Hessian H (p_tot × p_tot). ALO factors this
    /// matrix once and certifies blocked solves; it never materializes H⁻¹.
    pub penalized_hessian: &'a Array2<f64>,
    /// Per-observation observed NLL Hessians W_i (B × B). These drive the
    /// deletion denominator and may be indefinite even when H is SPD.
    pub observed_hessians: &'a [Array2<f64>],
    /// Per-observation score covariance matrices C_i (B × B). These drive
    /// variance and Cook influence, and must be positive semidefinite.
    pub score_covariances: &'a [Array2<f64>],
    /// Per-observation score vectors s_i = ∇_{η_i} NLL_i.  Length = n_obs,
    /// each entry is B-dimensional.
    pub scores: &'a [Array1<f64>],
    /// Fitted local-coordinate vectors η̂_i. Length = n_obs, each entry is
    /// B-dimensional. These are the exact row arguments paired with the
    /// coordinate designs; they need not be response means.
    pub coordinate_values: &'a [Array1<f64>],
}

/// Compute multi-block ALO diagnostics: corrected η̃ and leverages.
///
/// # Optimisation note
///
/// The dominant cost is forming X_i H⁻¹ X_iᵀ for every observation.
/// Rather than forming the B × p_tot row-block X_i and multiplying naïvely,
/// we solve for each coordinate b and bounded row chunk the matrix
///
///   H Q_b = X_bᵀ      (p_tot × chunk)
///
/// Then the (a, b) entry of the B × B matrix X_i H⁻¹ X_iᵀ is simply
///
///   (X_i H⁻¹ X_iᵀ)_{a,b} = x_{a,i}ᵀ Q_b[:,i]
///                           = Σ_k  X_a[i,k] · Q_b[k,i]
///
/// where x_{a,i} is the i-th row of coordinate-design a. This turns the per-
/// observation work from O(B · p_tot²) into O(B² · p_tot), and the
/// solve stays bounded without forming a dense inverse or an n × p_tot panel.
pub fn compute_multiblock_alo(
    input: &MultiBlockAloInput,
) -> Result<MultiBlockAloDiagnostics, EstimationError> {
    compute_multiblock_alo_inner(input).map_err(EstimationError::from)
}

fn validate_multiblock_alo_input(input: &MultiBlockAloInput<'_>) -> Result<(), AloError> {
    let n = input.n_obs;
    let b = input.n_coordinates;
    if n == 0 || b == 0 {
        return Err(AloError::InvalidInput {
            reason: format!(
                "multi-block ALO requires positive observation and coordinate counts; got n={n}, B={b}"
            ),
        });
    }
    if input.coordinate_designs.len() != b {
        return Err(AloError::InvalidInput {
            reason: format!(
                "multi-block ALO expected {b} coordinate designs, got {}",
                input.coordinate_designs.len()
            ),
        });
    }
    let p_tot = input.penalized_hessian.nrows();
    if input.penalized_hessian.ncols() != p_tot || p_tot == 0 {
        return Err(AloError::InvalidInput {
            reason: format!(
                "multi-block ALO penalized Hessian must be non-empty and square; got {}x{}",
                input.penalized_hessian.nrows(),
                input.penalized_hessian.ncols()
            ),
        });
    }
    if input.coordinate_coefficient_ranges.len() != b {
        return Err(AloError::InvalidInput {
            reason: format!(
                "multi-block ALO expected {b} coordinate coefficient ranges, got {}",
                input.coordinate_coefficient_ranges.len()
            ),
        });
    }
    for (coordinate, (design, coefficient_range)) in input
        .coordinate_designs
        .iter()
        .zip(input.coordinate_coefficient_ranges)
        .enumerate()
    {
        if design.nrows() != n {
            return Err(AloError::InvalidInput {
                reason: format!(
                    "multi-block ALO coordinate design {coordinate} has {} rows; expected {n}",
                    design.nrows()
                ),
            });
        }
        if design.ncols() == 0 || coefficient_range.is_empty() {
            return Err(AloError::InvalidInput {
                reason: format!(
                    "multi-block ALO coordinate {coordinate} has an empty local design or coefficient range"
                ),
            });
        }
        if coefficient_range.len() != design.ncols() || coefficient_range.end > p_tot {
            return Err(AloError::InvalidInput {
                reason: format!(
                    "multi-block ALO coordinate {coordinate} design has {} columns but parameter range {}..{} has length {} in a {p_tot}-dimensional saved Hessian",
                    design.ncols(),
                    coefficient_range.start,
                    coefficient_range.end,
                    coefficient_range.len()
                ),
            });
        }
    }
    for (label, length) in [
        ("observed_hessians", input.observed_hessians.len()),
        ("score_covariances", input.score_covariances.len()),
        ("scores", input.scores.len()),
        ("coordinate_values", input.coordinate_values.len()),
    ] {
        if length != n {
            return Err(AloError::InvalidInput {
                reason: format!("multi-block ALO requires {label} length {n}; got {length}"),
            });
        }
    }
    for row in 0..n {
        let observed = &input.observed_hessians[row];
        let score_covariance = &input.score_covariances[row];
        for (label, matrix) in [
            ("observed Hessian", observed),
            ("score covariance", score_covariance),
        ] {
            if matrix.dim() != (b, b) {
                return Err(AloError::InvalidInput {
                    reason: format!(
                        "multi-block ALO row {row} {label} has shape {}x{}; expected {b}x{b}",
                        matrix.nrows(),
                        matrix.ncols()
                    ),
                });
            }
            validate_finite_symmetric_matrix(matrix, &format!("multi-block ALO row {row} {label}"))
                .map_err(|error| AloError::InvalidInput {
                    reason: error.to_string(),
                })?;
        }
        let covariance_scale = score_covariance
            .iter()
            .fold(0.0_f64, |scale, value| scale.max(value.abs()));
        let (minimum, maximum) =
            symmetric_extremes(score_covariance).ok_or_else(|| AloError::InvalidInput {
                reason: format!(
                    "multi-block ALO row {row} score-covariance eigendecomposition failed"
                ),
            })?;
        let psd_tolerance = LOCAL_DELETE_SOLVE_ROUNDOFF_FACTOR
            * b as f64
            * f64::EPSILON
            * covariance_scale.max(maximum.abs());
        if minimum < -psd_tolerance {
            return Err(AloError::InvalidInput {
                reason: format!(
                    "multi-block ALO row {row} score covariance is not positive semidefinite: minimum eigenvalue {minimum:.6e}, roundoff allowance {psd_tolerance:.6e}"
                ),
            });
        }
        for (label, vector) in [
            ("score", &input.scores[row]),
            ("coordinate value", &input.coordinate_values[row]),
        ] {
            if vector.len() != b {
                return Err(AloError::InvalidInput {
                    reason: format!(
                        "multi-block ALO row {row} {label} has length {}; expected {b}",
                        vector.len()
                    ),
                });
            }
            if let Some((coordinate, value)) = vector
                .iter()
                .copied()
                .enumerate()
                .find(|(_, value)| !value.is_finite())
            {
                return Err(AloError::InvalidInput {
                    reason: format!(
                        "multi-block ALO row {row} {label} coordinate {coordinate} is non-finite: {value}"
                    ),
                });
            }
        }
    }
    Ok(())
}

fn compute_multiblock_alo_inner(
    input: &MultiBlockAloInput,
) -> Result<MultiBlockAloDiagnostics, AloError> {
    use rayon::prelude::*;

    let n = input.n_obs;
    let b = input.n_coordinates;
    let p_tot = input.penalized_hessian.nrows();
    validate_multiblock_alo_input(input)?;
    let factor = certified_spd_factorize(input.penalized_hessian, "multi-block ALO penalized Hessian")
        .map_err(|error| AloError::InvalidInput {
            reason: format!(
                "multi-block ALO requires an unperturbed positive-definite saved penalized Hessian: {error}"
            ),
        })?;

    let (chunk_size, max_concurrent_chunks) = multiblock_alo_parallel_plan(p_tot, b, n);
    let chunk_starts: Vec<usize> = (0..n).step_by(chunk_size).collect();

    // Each Rayon worker owns its small B×B/B-vector scratch buffers via
    // `map_init`, avoiding cross-thread mutation and avoiding per-observation
    // allocations.  The much larger Q panels are bounded by the parallel chunk
    // size and by wave-level concurrency, so at most roughly one global memory
    // budget worth of p_total × chunk_len panels can be live across workers.
    let mut chunk_results: Vec<Result<MultiBlockAloChunkDiagnostics, AloError>> =
        Vec::with_capacity(chunk_starts.len());
    for chunk_wave in chunk_starts.chunks(max_concurrent_chunks) {
        let mut wave_results: Vec<Result<MultiBlockAloChunkDiagnostics, AloError>> = chunk_wave
            .par_iter()
            .map_init(
                || MultiBlockAloScratch::new(b),
                |scratch, &chunk_start| {
                    let chunk_end = (chunk_start + chunk_size).min(n);
                    compute_multiblock_alo_chunk(input, &factor, chunk_start, chunk_end, scratch)
                },
            )
            .collect();
        chunk_results.append(&mut wave_results);
    }

    let mut eta_tilde = Vec::with_capacity(n);
    let mut leverage = Array1::<f64>::zeros(n);
    let mut alo_variance = Vec::with_capacity(n);
    let mut cook_distance = Array1::<f64>::zeros(n);

    let mut chunks = Vec::with_capacity(chunk_results.len());
    for result in chunk_results {
        chunks.push(result?);
    }
    chunks.sort_unstable_by_key(|chunk| chunk.chunk_start);

    for chunk in chunks {
        let chunk_start = chunk.chunk_start;
        eta_tilde.extend(chunk.eta_tilde);
        alo_variance.extend(chunk.alo_variance);
        for (local_i, lev) in chunk.leverage.into_iter().enumerate() {
            leverage[chunk_start + local_i] = lev;
        }
        for (local_i, cook) in chunk.cook_distance.into_iter().enumerate() {
            cook_distance[chunk_start + local_i] = cook;
        }
    }

    Ok(MultiBlockAloDiagnostics {
        eta_tilde,
        leverage,
        alo_variance,
        cook_distance,
    })
}

#[inline]
fn multiblock_alo_parallel_plan(
    p_tot: usize,
    n_coordinates: usize,
    n_obs: usize,
) -> (usize, usize) {
    if p_tot == 0 || n_coordinates == 0 || n_obs == 0 {
        return (1, 1);
    }
    // Each live row keeps one p-vector in the materialized Jacobian chunk and
    // one in its certified solution, for every local coordinate.
    let bytes_per_obs = p_tot
        .saturating_mul(n_coordinates)
        .saturating_mul(2)
        .saturating_mul(std::mem::size_of::<f64>())
        .max(1);
    let workers = rayon::current_num_threads().max(1);
    let max_concurrent_chunks = (MULTIBLOCK_ALO_MEMORY_BUDGET_BYTES / bytes_per_obs)
        .max(1)
        .min(workers);
    let per_worker_budget =
        (MULTIBLOCK_ALO_MEMORY_BUDGET_BYTES / max_concurrent_chunks).max(bytes_per_obs);
    let budget_obs = (per_worker_budget / bytes_per_obs).max(1);
    (budget_obs.min(n_obs), max_concurrent_chunks)
}

struct MultiBlockAloScratch {
    a_i: Vec<f64>,
    wa: Vec<f64>,
    aw: Vec<f64>,
    imwa: Vec<f64>,
    imaw: Vec<f64>,
    perm_imwa: Vec<usize>,
    perm_imaw: Vec<usize>,
    delta_eta: Vec<f64>,
    rhs_buf: Vec<f64>,
    covariance_u: Vec<f64>,
    var_diag_buf: Vec<f64>,
    w_flat: Vec<f64>,
    covariance_flat: Vec<f64>,
    lu_scratch: Vec<f64>,
    original_rhs: Vec<f64>,
}

impl MultiBlockAloScratch {
    fn new(b: usize) -> Self {
        let bb_sz = b * b;
        Self {
            a_i: vec![0.0f64; bb_sz],
            wa: vec![0.0f64; bb_sz],
            aw: vec![0.0f64; bb_sz],
            imwa: vec![0.0f64; bb_sz],
            imaw: vec![0.0f64; bb_sz],
            perm_imwa: vec![0usize; b],
            perm_imaw: vec![0usize; b],
            delta_eta: vec![0.0f64; b],
            rhs_buf: vec![0.0f64; b],
            covariance_u: vec![0.0f64; b],
            var_diag_buf: vec![0.0f64; b],
            w_flat: vec![0.0f64; bb_sz],
            covariance_flat: vec![0.0f64; bb_sz],
            lu_scratch: vec![0.0f64; b],
            original_rhs: vec![0.0f64; b],
        }
    }
}

struct MultiBlockAloChunkDiagnostics {
    chunk_start: usize,
    eta_tilde: Vec<Array1<f64>>,
    leverage: Vec<f64>,
    alo_variance: Vec<Array1<f64>>,
    cook_distance: Vec<f64>,
}

fn compute_multiblock_alo_chunk(
    input: &MultiBlockAloInput,
    factor: &CertifiedSpdFactor<'_>,
    chunk_start: usize,
    chunk_end: usize,
    scratch: &mut MultiBlockAloScratch,
) -> Result<MultiBlockAloChunkDiagnostics, AloError> {
    let b = input.n_coordinates;
    let p_tot = input.penalized_hessian.nrows();
    let chunk_len = chunk_end - chunk_start;

    let mut design_chunks = Vec::with_capacity(b);
    let mut q_blocks = Vec::with_capacity(b);
    for coordinate in 0..b {
        let design_chunk = input.coordinate_designs[coordinate]
            .try_row_chunk(chunk_start..chunk_end)
            .map_err(|reason| AloError::DesignDegenerate {
                reason: format!(
                    "multi-block ALO could not materialize coordinate {coordinate} rows {chunk_start}..{chunk_end}: {reason}"
                ),
            })?;
        if let Some(((row, column), value)) = design_chunk
            .indexed_iter()
            .map(|(index, &value)| (index, value))
            .find(|(_, value)| !value.is_finite())
        {
            return Err(AloError::DesignDegenerate {
                reason: format!(
                    "multi-block ALO coordinate {coordinate} design is non-finite at source row {}, column {column}: {value}",
                    chunk_start + row
                ),
            });
        }
        let coefficient_range = input.coordinate_coefficient_ranges[coordinate].clone();
        let mut rhs = Array2::<f64>::zeros((p_tot, chunk_len));
        rhs.slice_mut(s![coefficient_range, ..])
            .assign(&design_chunk.t());
        let (solution, _) = factor.solve_matrix(&rhs).map_err(|error| {
            AloError::LooComputationFailed {
                reason: format!(
                    "multi-block ALO saved-Hessian solve failed for coordinate {coordinate}, rows {chunk_start}..{chunk_end}: {error}"
                ),
            }
        })?;
        design_chunks.push(design_chunk);
        q_blocks.push(solution);
    }

    let mut eta_tilde = Vec::with_capacity(chunk_len);
    let mut leverage = vec![0.0f64; chunk_len];
    let mut alo_variance = Vec::with_capacity(chunk_len);
    let mut cook_distance = vec![0.0f64; chunk_len];

    for local_i in 0..chunk_len {
        let i = chunk_start + local_i;
        let w_i = &input.observed_hessians[i];
        let covariance_i = &input.score_covariances[i];

        // Flatten the distinct observed-Hessian and score-covariance surfaces
        // once per observation (row-major).
        for r in 0..b {
            for c in 0..b {
                scratch.w_flat[r * b + c] = w_i[(r, c)];
                scratch.covariance_flat[r * b + c] = covariance_i[(r, c)];
            }
        }

        // --- Assemble A_i = X_i H⁻¹ X_iᵀ  (B × B), row-major flat. ---
        for a in 0..b {
            let x_a = &design_chunks[a];
            let p_a = x_a.ncols();
            let off_a = input.coordinate_coefficient_ranges[a].start;
            let xa_row = x_a.row(local_i);
            for bb in 0..b {
                let q_bb = &q_blocks[bb];
                let mut dot = 0.0f64;
                for k in 0..p_a {
                    dot += xa_row[k] * q_bb[(off_a + k, local_i)];
                }
                scratch.a_i[a * b + bb] = dot;
            }
        }

        // WA = W_i · A_i (row-major).
        mat_mul_flat(&scratch.w_flat, &scratch.a_i, &mut scratch.wa, b);
        // AW = A_i · W_i (row-major).
        mat_mul_flat(&scratch.a_i, &scratch.w_flat, &mut scratch.aw, b);

        // Trace of H_ii = A_i W_i (= AW): leverage[i].
        // (Original code wrote H_ii = A · W — the same operator we already have in `aw`.)
        let mut tr = 0.0f64;
        for d in 0..b {
            tr += scratch.aw[d * b + d];
        }
        leverage[local_i] = tr;

        // Build (I - W A) and (I - A W) into imwa/imaw.
        for r in 0..b {
            for c in 0..b {
                let idx = r * b + c;
                let id = if r == c { 1.0 } else { 0.0 };
                scratch.imwa[idx] = id - scratch.wa[idx];
                scratch.imaw[idx] = id - scratch.aw[idx];
            }
        }

        // A singular frozen-H deletion system is diagnostic information, not a
        // request to alter the estimand with a local ridge. The uncertainty in
        // `I - product` is governed by the magnitudes of the two multiplicands,
        // even when their product cancels the identity almost completely. A
        // tolerance scaled only by the already-cancelled matrix would erase
        // exactly the information needed to recognize unit deletion leverage.
        let imwa_tolerance =
            identity_minus_product_lu_tolerance(&scratch.w_flat, &scratch.a_i, &scratch.wa, b)?;
        if !lu_factor_in_place(&mut scratch.imwa, &mut scratch.perm_imwa, b, imwa_tolerance) {
            return Err(AloError::LooComputationFailed {
                reason: format!(
                    "multi-block ALO deletion system I-WA is singular at row {i}; local pivot allowance {imwa_tolerance:.6e}, leverage trace {:.6e}",
                    leverage[local_i]
                ),
            });
        }
        let imaw_tolerance =
            identity_minus_product_lu_tolerance(&scratch.a_i, &scratch.w_flat, &scratch.aw, b)?;
        if !lu_factor_in_place(&mut scratch.imaw, &mut scratch.perm_imaw, b, imaw_tolerance) {
            return Err(AloError::LooComputationFailed {
                reason: format!(
                    "multi-block ALO transpose deletion system I-AW is singular at row {i}; local pivot allowance {imaw_tolerance:.6e}, leverage trace {:.6e}",
                    leverage[local_i]
                ),
            });
        }

        // v_i = (I - W A)⁻¹ s_i  -- solve into rhs_buf.
        let s_i = &input.scores[i];
        for k in 0..b {
            scratch.rhs_buf[k] = s_i[k];
        }
        if let Err(failure) = solve_identity_minus_product_in_place(
            &scratch.imwa,
            &scratch.perm_imwa,
            &scratch.wa,
            &mut scratch.rhs_buf,
            &mut scratch.lu_scratch,
            &mut scratch.original_rhs,
            imwa_tolerance,
            b,
        ) {
            return Err(AloError::LooComputationFailed {
                reason: format!(
                    "multi-block ALO deletion solve I-WA failed backward-error certification at row {i}: residual {:.6e}, allowance {:.6e}",
                    failure.residual_norm, failure.allowance
                ),
            });
        }
        // delta_eta = A_i · v_i
        for r in 0..b {
            let mut acc = 0.0f64;
            let row_off = r * b;
            for k in 0..b {
                acc += scratch.a_i[row_off + k] * scratch.rhs_buf[k];
            }
            scratch.delta_eta[r] = acc;
        }

        let eta_i = &input.coordinate_values[i];
        let mut corrected = Array1::<f64>::zeros(b);
        for d in 0..b {
            corrected[d] = eta_i[d] + scratch.delta_eta[d];
            if !scratch.delta_eta[d].is_finite() || !corrected[d].is_finite() {
                return Err(AloError::LooComputationFailed {
                    reason: format!(
                        "multi-block ALO correction is non-finite at row {i}, coordinate {d}: delta={}, corrected={}",
                        scratch.delta_eta[d], corrected[d]
                    ),
                });
            }
        }
        eta_tilde.push(corrected);

        // Cook's distance uses score covariance, not observed curvature.
        let mut cook = 0.0f64;
        let mut cook_scale = 0.0f64;
        for r in 0..b {
            let mut covariance_delta_r = 0.0f64;
            let row_off = r * b;
            for k in 0..b {
                covariance_delta_r += scratch.covariance_flat[row_off + k] * scratch.delta_eta[k];
            }
            let term = scratch.delta_eta[r] * covariance_delta_r;
            cook += term;
            cook_scale += term.abs();
        }
        let cook_tolerance =
            LOCAL_DELETE_SOLVE_ROUNDOFF_FACTOR * b as f64 * f64::EPSILON * cook_scale;
        if !cook.is_finite() || cook < -cook_tolerance {
            return Err(AloError::LooComputationFailed {
                reason: format!(
                    "multi-block ALO Cook influence is invalid at row {i}: value {cook:.6e}, roundoff allowance {cook_tolerance:.6e}"
                ),
            });
        }
        cook_distance[local_i] = cook.max(0.0);

        // var_diag[d] = a_d^T (I-WA)⁻¹ C (I-AW)⁻¹ a_d
        // where a_d is the d-th row of A_i.
        // Reuses already-factored imwa and imaw (one LU factorization each, reused
        // across all B right-hand sides — major saving over the original which redid
        // both LU decompositions B times per observation).
        for d in 0..b {
            let row_off = d * b;
            // u_d = (I - A W)⁻¹ a_d
            for k in 0..b {
                scratch.rhs_buf[k] = scratch.a_i[row_off + k];
            }
            if let Err(failure) = solve_identity_minus_product_in_place(
                &scratch.imaw,
                &scratch.perm_imaw,
                &scratch.aw,
                &mut scratch.rhs_buf,
                &mut scratch.lu_scratch,
                &mut scratch.original_rhs,
                imaw_tolerance,
                b,
            ) {
                return Err(AloError::LooComputationFailed {
                    reason: format!(
                        "multi-block ALO transpose variance solve I-AW failed backward-error certification at row {i}, coordinate {d}: residual {:.6e}, allowance {:.6e}",
                        failure.residual_norm, failure.allowance
                    ),
                });
            }
            // covariance_u = C u_d
            for r in 0..b {
                let mut acc = 0.0f64;
                let wr = r * b;
                for k in 0..b {
                    acc += scratch.covariance_flat[wr + k] * scratch.rhs_buf[k];
                }
                scratch.covariance_u[r] = acc;
            }
            // t_d = (I - W A)⁻¹ C u_d.
            if let Err(failure) = solve_identity_minus_product_in_place(
                &scratch.imwa,
                &scratch.perm_imwa,
                &scratch.wa,
                &mut scratch.covariance_u,
                &mut scratch.lu_scratch,
                &mut scratch.original_rhs,
                imwa_tolerance,
                b,
            ) {
                return Err(AloError::LooComputationFailed {
                    reason: format!(
                        "multi-block ALO variance solve I-WA failed backward-error certification at row {i}, coordinate {d}: residual {:.6e}, allowance {:.6e}",
                        failure.residual_norm, failure.allowance
                    ),
                });
            }
            // v_dd = a_d^T t_d
            let mut v_dd = 0.0f64;
            for k in 0..b {
                v_dd += scratch.a_i[row_off + k] * scratch.covariance_u[k];
            }
            let variance_scale = scratch.a_i[row_off..row_off + b]
                .iter()
                .zip(scratch.covariance_u.iter())
                .map(|(left, right)| (left * right).abs())
                .sum::<f64>();
            let variance_tolerance =
                LOCAL_DELETE_SOLVE_ROUNDOFF_FACTOR * b as f64 * f64::EPSILON * variance_scale;
            if !v_dd.is_finite() || v_dd < -variance_tolerance {
                return Err(AloError::LooComputationFailed {
                    reason: format!(
                        "multi-block ALO variance is invalid at row {i}, coordinate {d}: value {v_dd:.6e}, roundoff allowance {variance_tolerance:.6e}"
                    ),
                });
            }
            scratch.var_diag_buf[d] = v_dd.max(0.0);
        }
        let mut var_diag = Array1::<f64>::zeros(b);
        for d in 0..b {
            var_diag[d] = scratch.var_diag_buf[d];
        }
        alo_variance.push(var_diag);
    }

    Ok(MultiBlockAloChunkDiagnostics {
        chunk_start,
        eta_tilde,
        leverage,
        alo_variance,
        cook_distance,
    })
}

/// B × B row-major matmul: out = a · b.
#[inline]
fn mat_mul_flat(a: &[f64], b_mat: &[f64], out: &mut [f64], b: usize) {
    for r in 0..b {
        let ar = r * b;
        let or = r * b;
        for c in 0..b {
            let mut acc = 0.0f64;
            for k in 0..b {
                acc += a[ar + k] * b_mat[k * b + c];
            }
            out[or + c] = acc;
        }
    }
}

/// Standard `gamma_n = n*u/(1-n*u)` bound for `n` rounded operations, where
/// binary64 unit roundoff under round-to-nearest is `u = eps/2`.
#[inline]
fn floating_point_gamma(operation_count: usize) -> f64 {
    let accumulated = operation_count as f64 * (0.5 * f64::EPSILON);
    if accumulated < 1.0 {
        accumulated / (1.0 - accumulated)
    } else {
        f64::INFINITY
    }
}

/// Pivot allowance for a row-major `I - left * right` local system.
///
/// The scale is `max(||I-left*right||_inf,
/// 1 + || |left||right| ||_inf)`: the second operand-derived term is the
/// magnitude envelope before cancellation, while the first retains the actual
/// operator scale when it is larger. Forming every product entry takes at most
/// `2B` rounded multiply/add operations and subtracting it from the identity
/// takes one more. The LU term accounts for the division/multiply/subtract chain
/// along at most `B` partial-pivoting elimination stages. The resulting bound
/// cannot collapse merely because `left * right` rounded close to the identity.
fn identity_minus_product_lu_tolerance(
    left: &[f64],
    right: &[f64],
    product: &[f64],
    b: usize,
) -> Result<f64, AloError> {
    let expected_len = b.checked_mul(b).ok_or_else(|| AloError::InvalidInput {
        reason: format!(
            "multi-block ALO local deletion dimension B={b} overflows the square matrix size"
        ),
    })?;
    for (name, actual_len) in [
        ("left operand", left.len()),
        ("right operand", right.len()),
        ("precomputed product", product.len()),
    ] {
        if actual_len != expected_len {
            return Err(AloError::InvalidInput {
                reason: format!(
                    "multi-block ALO local deletion {name} has length {actual_len}, expected B*B={expected_len} for B={b}"
                ),
            });
        }
    }

    let mut operand_envelope_inf = 0.0_f64;
    let mut system_norm_inf = 0.0_f64;
    for row in 0..b {
        let mut operand_row_envelope = 1.0_f64;
        let mut system_row_norm = 0.0_f64;
        for column in 0..b {
            let mut product_entry_envelope = 0.0_f64;
            for inner in 0..b {
                product_entry_envelope +=
                    left[row * b + inner].abs() * right[inner * b + column].abs();
            }
            operand_row_envelope += product_entry_envelope;
            let identity = if row == column { 1.0 } else { 0.0 };
            system_row_norm += (identity - product[row * b + column]).abs();
        }
        operand_envelope_inf = operand_envelope_inf.max(operand_row_envelope);
        system_norm_inf = system_norm_inf.max(system_row_norm);
    }

    let formation_operations = b.saturating_mul(2).saturating_add(1);
    let elimination_operations = b.saturating_mul(3);
    let backward_error_scale = operand_envelope_inf.max(system_norm_inf);
    Ok(
        (floating_point_gamma(formation_operations) + floating_point_gamma(elimination_operations))
            * backward_error_scale,
    )
}

/// LU-decompose a B × B row-major matrix in place with partial pivoting and
/// physical row swaps. Returns false if any pivot is within the caller's
/// scale-aware backward-error allowance.
/// On success, `m` holds L (strict lower, unit diag implicit) and U (upper, diag
/// included); `perm[k]` records the original-row index that ended up in physical
/// row k after pivoting.
fn lu_factor_in_place(m: &mut [f64], perm: &mut [usize], b: usize, pivot_tolerance: f64) -> bool {
    for i in 0..b {
        perm[i] = i;
    }
    for col in 0..b {
        // Partial pivot on column `col` over physical rows `[col..b]`.
        let mut max_val = m[col * b + col].abs();
        let mut max_idx = col;
        for row in (col + 1)..b {
            let v = m[row * b + col].abs();
            if v > max_val {
                max_val = v;
                max_idx = row;
            }
        }
        if !max_val.is_finite() || max_val <= pivot_tolerance {
            return false;
        }
        if max_idx != col {
            // Physically swap rows `col` and `max_idx` (full row, all columns).
            for k in 0..b {
                m.swap(col * b + k, max_idx * b + k);
            }
            perm.swap(col, max_idx);
        }
        let pivot = m[col * b + col];
        for row in (col + 1)..b {
            let factor = m[row * b + col] / pivot;
            m[row * b + col] = factor; // store L below diag
            for k in (col + 1)..b {
                let upd = factor * m[col * b + k];
                m[row * b + k] -= upd;
            }
        }
    }
    true
}

/// Solve L U x = P rhs using a previously factored matrix (LU in `m`, perm).
/// Writes the solution back into `rhs`. `scratch` must have length ≥ b.
fn lu_solve_in_place(m: &[f64], perm: &[usize], rhs: &mut [f64], scratch: &mut [f64], b: usize) {
    // Forward substitution Ly = P rhs (L is unit-diag, strict lower of m).
    let y = &mut scratch[..b];
    for row in 0..b {
        let mut s = rhs[perm[row]];
        for k in 0..row {
            s -= m[row * b + k] * y[k];
        }
        y[row] = s;
    }
    // Back substitution U x = y.  Write into rhs[].
    for row in (0..b).rev() {
        let mut s = y[row];
        for k in (row + 1)..b {
            s -= m[row * b + k] * rhs[k];
        }
        rhs[row] = s / m[row * b + row];
    }
}

#[derive(Clone, Copy, Debug)]
struct LocalSolveResidualFailure {
    residual_norm: f64,
    allowance: f64,
}

/// Solve a factored `I - product` system and certify the result against the
/// unfactored operator. The residual allowance contains both the uncertainty in
/// forming the operator and the operation-count-derived error from LU,
/// triangular substitution, and residual evaluation. This is a backward-error
/// certificate, so a well-resolved but ill-conditioned system remains valid;
/// only a solve unsupported by its own arithmetic is refused.
fn solve_identity_minus_product_in_place(
    lu: &[f64],
    permutation: &[usize],
    product: &[f64],
    rhs: &mut [f64],
    lu_scratch: &mut [f64],
    original_rhs: &mut [f64],
    operator_error_bound: f64,
    b: usize,
) -> Result<(), LocalSolveResidualFailure> {
    original_rhs[..b].copy_from_slice(&rhs[..b]);
    lu_solve_in_place(lu, permutation, rhs, lu_scratch, b);

    let rhs_norm = original_rhs[..b]
        .iter()
        .fold(0.0_f64, |norm, value| norm.max(value.abs()));
    let solution_norm = rhs[..b]
        .iter()
        .fold(0.0_f64, |norm, value| norm.max(value.abs()));
    let mut system_norm = 0.0_f64;
    let mut residual_norm = 0.0_f64;
    for row in 0..b {
        let mut row_norm = 0.0_f64;
        let mut residual = original_rhs[row];
        for column in 0..b {
            let identity = if row == column { 1.0 } else { 0.0 };
            let matrix_entry = identity - product[row * b + column];
            row_norm += matrix_entry.abs();
            residual -= matrix_entry * rhs[column];
        }
        system_norm = system_norm.max(row_norm);
        residual_norm = residual_norm.max(residual.abs());
    }

    // Per dimension: at most 3B factorization operations on a surviving entry,
    // 2B in each triangular substitution, and 3B to reconstruct/evaluate the
    // residual from `I - product`.
    let certification_operations = b.saturating_mul(10);
    let arithmetic_scale = system_norm * solution_norm + rhs_norm;
    let allowance = floating_point_gamma(certification_operations) * arithmetic_scale
        + operator_error_bound * solution_norm;
    if rhs[..b].iter().any(|value| !value.is_finite())
        || !residual_norm.is_finite()
        || !allowance.is_finite()
        || residual_norm > allowance
    {
        Err(LocalSolveResidualFailure {
            residual_norm,
            allowance,
        })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ALO_EXACT_SCALAR_MAX_ITERS, AloExactScalarError, AloInput, alo_eta_exact_frozen_curvature,
        alo_eta_updatewith_offset, compute_alo_from_input_inner, finite_weighted_square_sum,
        percentile_from_sorted, percentile_index, spd_quadratic_after_certified_solve,
    };
    use gam_linalg::matrix::{PsdWeightsView, SignedWeightsView};

    #[test]
    fn alo_offset_update_matches_centered_algebra() {
        let eta_hat = 11.0;
        let z = 13.0;
        let offset = 10.0;
        let x_hinv_x = 0.2;
        let hessian_weight = 1.0;
        let score_weight = 1.0;
        // centered: eta~=off + ((eta-off)-a(z-off))/(1-a) when W_S = W_H.
        let leverage = hessian_weight * x_hinv_x;
        let expected = offset + ((eta_hat - offset) - leverage * (z - offset)) / (1.0 - leverage);
        let got =
            alo_eta_updatewith_offset(eta_hat, z, offset, x_hinv_x, score_weight, 1.0 - leverage);
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn alo_offset_update_reduces_to_classicwhen_offsetzero() {
        let eta_hat = 1.25;
        let z = -0.5;
        let x_hinv_x = 0.35;
        let hessian_weight = 1.0;
        let score_weight = 1.0;
        let leverage = hessian_weight * x_hinv_x;
        let expected = (eta_hat - leverage * z) / (1.0 - leverage);
        let got =
            alo_eta_updatewith_offset(eta_hat, z, 0.0, x_hinv_x, score_weight, 1.0 - leverage);
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn alo_offset_update_uses_distinct_score_and_hessian_weights() {
        let eta_hat = 1.7;
        let z = 0.4;
        let offset = -0.2;
        let x_hinv_x = 0.15;
        let hessian_weight = 3.0;
        let score_weight = 5.0;
        let expected = offset
            + (eta_hat - offset)
            + x_hinv_x * score_weight * ((eta_hat - offset) - (z - offset))
                / (1.0 - hessian_weight * x_hinv_x);
        let got = alo_eta_updatewith_offset(
            eta_hat,
            z,
            offset,
            x_hinv_x,
            score_weight,
            1.0 - hessian_weight * x_hinv_x,
        );
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn alo_offset_update_handles_zero_hessian_weight() {
        let eta_hat = 0.8;
        let z = -0.3;
        let offset = 0.1;
        let x_hinv_x = 0.4;
        let hessian_weight = 0.0;
        let score_weight = 2.5;
        let expected = offset
            + (eta_hat - offset)
            + x_hinv_x * score_weight * ((eta_hat - offset) - (z - offset));
        let got = alo_eta_updatewith_offset(
            eta_hat,
            z,
            offset,
            x_hinv_x,
            score_weight,
            1.0 - hessian_weight * x_hinv_x,
        );
        assert!((got - expected).abs() < 1e-12);
    }

    #[test]
    fn alo_exact_frozen_curvature_converges_to_fixed_point() {
        let eta_hat = 1.0;
        let a_ii = 0.4;
        let got =
            alo_eta_exact_frozen_curvature(eta_hat, a_ii, &|eta| Ok((0.5 * (eta - 2.0), 0.5)))
                .expect("linear scalar fixed point should converge in one Newton step");
        assert!((got - 0.75).abs() < 1e-12);
    }

    #[test]
    fn alo_exact_frozen_curvature_reports_nonconvergence() {
        let err = alo_eta_exact_frozen_curvature(0.0, 1.0, &|eta| Ok((eta + 1.0, 0.0)))
            .expect_err("constant residual should exhaust the scalar iteration budget");
        let AloExactScalarError::MaxIterations { iterations, .. } = err else {
            panic!("constant residual must report MaxIterations, got {err:?}");
        };
        assert_eq!(
            iterations, ALO_EXACT_SCALAR_MAX_ITERS,
            "non-convergence must report the full scalar iteration budget"
        );
    }

    #[test]
    fn alo_input_reports_exact_scalar_nonconvergence_with_row_context() {
        let design = Array2::from_elem((1, 1), 1.0);
        let penalized_hessian = Array2::from_elem((1, 1), 1.0);
        let hessian_weights = Array1::from_vec(vec![0.0]);
        let score_weights = Array1::from_vec(vec![0.0]);
        let working_response = Array1::from_vec(vec![0.0]);
        let eta = Array1::from_vec(vec![0.0]);
        let offset = Array1::from_vec(vec![0.0]);
        let score_curvature = |_: usize, eta: f64| Ok((eta + 1.0, 0.0));
        let input = AloInput {
            design: &design,
            penalized_hessian: &penalized_hessian,
            hessian_weights: SignedWeightsView::from_array(&hessian_weights),
            score_weights: PsdWeightsView::try_from_array(&score_weights).expect("psd weights"),
            working_response: &working_response,
            eta: &eta,
            offset: &offset,
            phi: 1.0,
            score_curvature: Some(&score_curvature),
        };

        let err =
            compute_alo_from_input_inner(&input).expect_err("non-converged exact ALO must error");
        let msg = err.to_string();
        assert!(
            msg.contains("ALO exact frozen-curvature solve failed at row 0"),
            "missing row context in exact ALO error: {msg}"
        );
        assert!(
            msg.contains("did not converge within"),
            "missing non-convergence cause in exact ALO error: {msg}"
        );
    }

    #[test]
    fn alo_scale_safe_quadratics_preserve_tiny_weights_without_false_overflow() {
        let weights = Array1::from_vec(vec![1e-300, 2.0]);
        let values = [1e200, 3.0];
        let meat = finite_weighted_square_sum(0, weights.view(), &values)
            .expect("weighted square sum is representable");
        assert!(meat.is_finite());
        assert!((meat - 1e100).abs() <= 8.0 * f64::EPSILON * 1e100);

        let rhs = Array1::from_vec(vec![2.0, -1.0]);
        let solution = Array1::from_vec(vec![1.5, 0.5]);
        let quadratic =
            spd_quadratic_after_certified_solve(0, rhs.view(), solution.view()).unwrap();
        assert_eq!(quadratic, 2.5);
    }

    #[test]
    fn sandwich_meat_uses_score_weights_not_hessian_weights_noncanonical() {
        // Regression for the sandwich-SE "meat" weight bug: the meat must be the
        // SCORE covariance Xᵀ diag(W_S) X (Fisher, PSD), NOT the observed-info
        // Hessian weight W_H (signed). This fixture mimics a non-canonical link
        // (W_H ≠ W_S) with mixed-sign observed curvature.
        //
        // Single column (p = 1) makes H a scalar, so the sandwich variance is
        // closed form: with H = Σ W_H·x² + s0 (> 0 after the penalty), the meat
        // for obs is x_obs²·H⁻²·Σ_row W_S·x_row², and se = sqrt(φ·meat).
        let x = Array2::from_shape_vec((5, 1), vec![1.0, 2.0, 1.0, 2.0, 1.0]).unwrap();
        // Mixed-sign observed-information weights; the negative rows carry the
        // larger design values so Σ W_H·x² is NEGATIVE (see assert below).
        let w_h_vec = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0, 0.5]);
        // Score/Fisher weights are strictly positive (PSD by construction).
        let w_s_vec = Array1::from_vec(vec![1.0, 0.8, 1.2, 0.6, 0.9]);
        let phi = 1.3;

        let n = x.nrows();
        let sum_wh_x2: f64 = (0..n).map(|i| w_h_vec[i] * x[[i, 0]] * x[[i, 0]]).sum();
        let sum_ws_x2: f64 = (0..n).map(|i| w_s_vec[i] * x[[i, 0]] * x[[i, 0]]).sum();
        // The whole point: Σ W_H·x² < 0 < Σ W_S·x². With W_H the meat is negative
        // and the "materially negative sandwich variance" guard would trip
        // (spurious LooComputationFailed); with W_S it is a valid PSD meat.
        assert!(sum_wh_x2 < 0.0, "fixture must exercise a negative W_H meat");
        assert!(sum_ws_x2 > 0.0);

        // Penalize enough that the penalized Hessian is PD despite Σ W_H·x² < 0.
        let s0 = 8.0_f64;
        let h = s0 + sum_wh_x2; // = 2.5
        assert!(h > 0.0, "penalized Hessian must stay PD");
        let penalized_hessian = Array2::from_elem((1, 1), h);

        // Pre-fix arithmetic check: the OLD W_H meat would be materially negative
        // for the larger-x rows, so the old code returned LooComputationFailed.
        let old_meat_obs1 = x[[1, 0]] * x[[1, 0]] / (h * h) * sum_wh_x2;
        assert!(phi * old_meat_obs1 < 0.0, "the pre-fix W_H meat is signed");

        let working_response = Array1::from_vec(vec![0.3, -0.2, 0.5, 0.1, -0.4]);
        let eta = Array1::from_vec(vec![0.2, 0.1, 0.4, -0.1, 0.05]);
        let offset = Array1::zeros(n);
        let input = AloInput {
            design: &x,
            penalized_hessian: &penalized_hessian,
            hessian_weights: SignedWeightsView::from_array(&w_h_vec),
            score_weights: PsdWeightsView::try_from_array(&w_s_vec).expect("psd weights"),
            working_response: &working_response,
            eta: &eta,
            offset: &offset,
            phi,
            score_curvature: None,
        };

        // The fix must let this succeed (no spurious negative-meat failure)...
        let diag = compute_alo_from_input_inner(&input)
            .expect("fixed sandwich meat (W_S) must not trip the negative-variance guard");

        // ...and match the closed-form W_S reference for every row.
        for obs in 0..n {
            let expected = (phi * x[[obs, 0]] * x[[obs, 0]] / (h * h) * sum_ws_x2).sqrt();
            assert!(
                (diag.se_sandwich[obs] - expected).abs() <= 1e-10 * expected.max(1.0),
                "row {obs}: se_sandwich={} expected={expected}",
                diag.se_sandwich[obs]
            );
            let expected_leverage = w_h_vec[obs] * x[[obs, 0]] * x[[obs, 0]] / h;
            assert!(
                (diag.leverage[obs] - expected_leverage).abs()
                    <= 1e-12 * expected_leverage.abs().max(1.0),
                "row {obs}: signed leverage={} expected={expected_leverage}",
                diag.leverage[obs]
            );
        }
        assert!(
            diag.leverage[1] < 0.0,
            "negative observed curvature must remain signed"
        );
    }

    #[test]
    fn percentile_index_matches_expected_rounding() {
        assert_eq!(percentile_index(0, 0.95), 0);
        assert_eq!(percentile_index(1, 0.95), 0);
        assert_eq!(percentile_index(10, 0.50), 5);
        assert_eq!(percentile_index(10, 0.95), 9);
    }

    #[test]
    fn percentile_from_sorted_returns_order_statistic() {
        let values = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(percentile_from_sorted(&values, 0.50), 3.0);
        assert_eq!(percentile_from_sorted(&values, 0.95), 5.0);
        assert_eq!(percentile_from_sorted(&[], 0.95), 0.0);
    }

    // --- Multi-block ALO tests ---

    use super::{
        MultiBlockAloInput, compute_multiblock_alo, floating_point_gamma,
        identity_minus_product_lu_tolerance, lu_factor_in_place, mat_mul_flat,
    };
    use gam_linalg::matrix::DesignMatrix;
    use ndarray::{Array1, Array2};

    fn local_identity_minus_product_is_factorable(left: &[f64], right: &[f64], b: usize) -> bool {
        let mut product = vec![0.0; b * b];
        mat_mul_flat(left, right, &mut product, b);
        let mut system = vec![0.0; b * b];
        for row in 0..b {
            for column in 0..b {
                let identity = if row == column { 1.0 } else { 0.0 };
                system[row * b + column] = identity - product[row * b + column];
            }
        }
        let tolerance = identity_minus_product_lu_tolerance(left, right, &product, b)
            .expect("test matrices satisfy the B-by-B local deletion contract");
        let mut permutation = vec![0; b];
        lu_factor_in_place(&mut system, &mut permutation, b, tolerance)
    }

    #[test]
    fn multiblock_b1_matches_scalar_leverage() {
        // With B=1 the multi-block formula should reduce to the scalar case.
        // H_ii = x_i^T H^{-1} x_i * w_i  (scalar).
        let n = 3;
        let p = 2;
        let x = Array2::from_shape_vec((n, p), vec![1.0, 0.5, 0.8, -0.3, 0.2, 1.1]).unwrap();
        // H = X'WX + I (simple regularisation).
        let w = [1.0, 2.0, 0.5];
        let mut h = Array2::<f64>::eye(p);
        for i in 0..n {
            for r in 0..p {
                for c in 0..p {
                    h[(r, c)] += w[i] * x[(i, r)] * x[(i, c)];
                }
            }
        }
        // Invert H (2x2).
        let det = h[(0, 0)] * h[(1, 1)] - h[(0, 1)] * h[(1, 0)];
        let mut h_inv = Array2::<f64>::zeros((p, p));
        h_inv[(0, 0)] = h[(1, 1)] / det;
        h_inv[(1, 1)] = h[(0, 0)] / det;
        h_inv[(0, 1)] = -h[(0, 1)] / det;
        h_inv[(1, 0)] = -h[(1, 0)] / det;

        // Scalar leverages: a_ii = w_i * x_i^T H^{-1} x_i
        let mut scalar_lev = vec![0.0f64; n];
        for i in 0..n {
            let mut xhx = 0.0;
            for r in 0..p {
                for c in 0..p {
                    xhx += x[(i, r)] * h_inv[(r, c)] * x[(i, c)];
                }
            }
            scalar_lev[i] = w[i] * xhx;
        }

        // Multi-block with B=1. The score covariance is deliberately supplied
        // separately even though this well-specified fixture sets C_i = W_i.
        let coordinate_designs = vec![DesignMatrix::from(x.clone())];
        let coordinate_coefficient_ranges = vec![0..p];
        let observed_hessians: Vec<Array2<f64>> =
            w.iter().map(|&wi| Array2::from_elem((1, 1), wi)).collect();
        let score_covariances = observed_hessians.clone();
        let scores: Vec<Array1<f64>> = (0..n).map(|_| Array1::from_vec(vec![0.1])).collect();
        let coordinate_values: Vec<Array1<f64>> =
            (0..n).map(|i| Array1::from_vec(vec![i as f64])).collect();

        let input = MultiBlockAloInput {
            n_obs: n,
            n_coordinates: 1,
            coordinate_designs: &coordinate_designs,
            coordinate_coefficient_ranges: &coordinate_coefficient_ranges,
            penalized_hessian: &h,
            observed_hessians: &observed_hessians,
            score_covariances: &score_covariances,
            scores: &scores,
            coordinate_values: &coordinate_values,
        };

        let result = compute_multiblock_alo(&input).unwrap();
        for i in 0..n {
            assert!(
                (result.leverage[i] - scalar_lev[i]).abs() < 1e-10,
                "leverage mismatch at i={}: got {}, expected {}",
                i,
                result.leverage[i],
                scalar_lev[i]
            );
        }
    }

    #[test]
    fn multiblock_b2_matches_closed_form_with_cross_geometry() {
        // A one-row B=2 identity-Jacobian fixture pins every matrix ordering:
        // A=H^-1, M=I-WA, delta=A M^-1 s, leverage=tr(AW), and the distinct
        // score covariance C drives Cook/variance. Diagonal-only or C=W code
        // cannot pass this fixture. Both coordinates address the same full
        // parameter range, exercising the shared-coefficient contract used by
        // survival and latent rows.
        let coordinate_designs = vec![
            DesignMatrix::from(Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap()),
            DesignMatrix::from(Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap()),
        ];
        let coordinate_coefficient_ranges = vec![0..2, 0..2];
        let h = Array2::from_shape_vec((2, 2), vec![2.0, 0.25, 0.25, 3.0]).unwrap();
        let w = Array2::from_shape_vec((2, 2), vec![0.2, 0.05, 0.05, 0.3]).unwrap();
        let c = Array2::from_shape_vec((2, 2), vec![0.5, 0.1, 0.1, 0.4]).unwrap();
        let observed_hessians = vec![w.clone()];
        let score_covariances = vec![c.clone()];
        let scores = vec![Array1::from_vec(vec![0.4, -0.2])];
        let coordinate_values = vec![Array1::from_vec(vec![1.0, -0.5])];
        let input = MultiBlockAloInput {
            n_obs: 1,
            n_coordinates: 2,
            coordinate_designs: &coordinate_designs,
            coordinate_coefficient_ranges: &coordinate_coefficient_ranges,
            penalized_hessian: &h,
            observed_hessians: &observed_hessians,
            score_covariances: &score_covariances,
            scores: &scores,
            coordinate_values: &coordinate_values,
        };

        let det_h = h[[0, 0]] * h[[1, 1]] - h[[0, 1]] * h[[1, 0]];
        let a = Array2::from_shape_vec(
            (2, 2),
            vec![
                h[[1, 1]] / det_h,
                -h[[0, 1]] / det_h,
                -h[[1, 0]] / det_h,
                h[[0, 0]] / det_h,
            ],
        )
        .unwrap();
        let m = Array2::<f64>::eye(2) - w.dot(&a);
        let det_m = m[[0, 0]] * m[[1, 1]] - m[[0, 1]] * m[[1, 0]];
        let m_inv = Array2::from_shape_vec(
            (2, 2),
            vec![
                m[[1, 1]] / det_m,
                -m[[0, 1]] / det_m,
                -m[[1, 0]] / det_m,
                m[[0, 0]] / det_m,
            ],
        )
        .unwrap();
        let delta = a.dot(&m_inv.dot(&scores[0]));
        let expected_eta = &coordinate_values[0] + &delta;
        let expected_leverage = (a.dot(&w)).diag().sum();
        let expected_cook = delta.dot(&c.dot(&delta));
        let variance = a.dot(&m_inv).dot(&c).dot(&m_inv.t()).dot(&a.t());

        let result = compute_multiblock_alo(&input).expect("B=2 closed-form ALO");
        for coordinate in 0..2 {
            assert!((result.eta_tilde[0][coordinate] - expected_eta[coordinate]).abs() < 2e-12);
            assert!(
                (result.alo_variance[0][coordinate] - variance[[coordinate, coordinate]]).abs()
                    < 2e-12
            );
        }
        assert!((result.leverage[0] - expected_leverage).abs() < 2e-12);
        assert!((result.cook_distance[0] - expected_cook).abs() < 2e-12);
    }

    #[test]
    fn multiblock_singular_weight_still_corrects() {
        // When W_i = 0 (singular), the W_i⁻¹-free formula still works:
        // (I - W_i A_i)⁻¹ = I, so Δη = A_i s_i.
        // A_i = x H⁻¹ xᵀ = 1.0² + 0.5² = 1.25 (scalar, B=1).
        let n = 1;
        let p = 2;
        let x = Array2::from_shape_vec((1, p), vec![1.0, 0.5]).unwrap();
        let h = Array2::eye(p);
        let coordinate_designs = vec![DesignMatrix::from(x.clone())];
        let coordinate_coefficient_ranges = vec![0..p];
        let observed_hessians = vec![Array2::from_elem((1, 1), 0.0)];
        let score_covariances = observed_hessians.clone();
        let scores = vec![Array1::from_vec(vec![1.0])];
        let coordinate_values = vec![Array1::from_vec(vec![std::f64::consts::PI])];

        let input = MultiBlockAloInput {
            n_obs: n,
            n_coordinates: 1,
            coordinate_designs: &coordinate_designs,
            coordinate_coefficient_ranges: &coordinate_coefficient_ranges,
            penalized_hessian: &h,
            observed_hessians: &observed_hessians,
            score_covariances: &score_covariances,
            scores: &scores,
            coordinate_values: &coordinate_values,
        };
        let result = compute_multiblock_alo(&input).unwrap();
        // Δη = A_i * s_i = 1.25 * 1.0 = 1.25
        let expected = std::f64::consts::PI + 1.25;
        assert!(
            (result.eta_tilde[0][0] - expected).abs() < 1e-12,
            "expected {}, got {}",
            expected,
            result.eta_tilde[0][0]
        );
        // Cook's distance should be 0 since C_i = 0.
        assert!(result.cook_distance[0].abs() < 1e-14);
        // ALO variance should be 0 since C_i = 0.
        assert!(result.alo_variance[0][0].abs() < 1e-14);
    }

    #[test]
    fn multiblock_unit_leverage_refuses_instead_of_changing_estimand() {
        let coordinate_designs = vec![DesignMatrix::from(Array2::from_elem((1, 1), 1.0))];
        let coordinate_coefficient_ranges = vec![0..1];
        let h = Array2::from_elem((1, 1), 2.0);
        let observed_hessians = vec![Array2::from_elem((1, 1), 2.0)];
        let score_covariances = vec![Array2::from_elem((1, 1), 1.0)];
        let scores = vec![Array1::from_vec(vec![0.4])];
        let coordinate_values = vec![Array1::from_vec(vec![1.0])];
        let input = MultiBlockAloInput {
            n_obs: 1,
            n_coordinates: 1,
            coordinate_designs: &coordinate_designs,
            coordinate_coefficient_ranges: &coordinate_coefficient_ranges,
            penalized_hessian: &h,
            observed_hessians: &observed_hessians,
            score_covariances: &score_covariances,
            scores: &scores,
            coordinate_values: &coordinate_values,
        };
        let error = compute_multiblock_alo(&input)
            .expect_err("unit deletion leverage must be reported as singular");
        assert!(
            error
                .to_string()
                .contains("deletion system I-WA is singular")
        );
    }

    #[test]
    fn multiblock_b2_identity_cancellation_is_numerically_singular() {
        // The stored operands differ from an exact inverse pair by one ulp, so
        // the formed diagonal of I-WA is nonzero but smaller than the error in
        // forming the product. Scaling by ||I-WA|| alone would accept it.
        let above_two = f64::from_bits(2.0_f64.to_bits() + 1);
        let w = [above_two, 0.0, 0.0, above_two];
        let a = [0.5, 0.0, 0.0, 0.5];
        assert!(!local_identity_minus_product_is_factorable(&w, &a, 2));
        assert!(!local_identity_minus_product_is_factorable(&a, &w, 2));
    }

    #[test]
    fn multiblock_b2_safely_near_singular_deletion_is_accepted() {
        // This system is ill-conditioned, but its smallest pivot is sqrt(eps),
        // well outside the O(eps) formation uncertainty of its operands.
        let gap = f64::EPSILON.sqrt();
        let identity = [1.0, 0.0, 0.0, 1.0];
        let product_operand = [1.0 - gap, 0.0, 0.0, 0.5];
        assert!(local_identity_minus_product_is_factorable(
            &identity,
            &product_operand,
            2
        ));
        assert!(local_identity_minus_product_is_factorable(
            &product_operand,
            &identity,
            2
        ));
    }

    #[test]
    fn multiblock_trace_one_but_invertible_deletion_is_not_refused() {
        // tr(AW)=1 is only a scalar summary. Here I-AW has eigenvalues 3/4 and
        // 1/4, so a leverage gate would reject a perfectly regular exact solve.
        let coordinate_designs = vec![
            DesignMatrix::from(Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap()),
            DesignMatrix::from(Array2::from_shape_vec((1, 2), vec![0.0, 1.0]).unwrap()),
        ];
        let coordinate_coefficient_ranges = vec![0..2, 0..2];
        let penalized_hessian = Array2::<f64>::eye(2);
        let observed_hessians =
            vec![Array2::from_shape_vec((2, 2), vec![0.25, 0.0, 0.0, 0.75]).unwrap()];
        let score_covariances = vec![Array2::<f64>::zeros((2, 2))];
        let scores = vec![Array1::from_vec(vec![0.75, -0.25])];
        let coordinate_values = vec![Array1::<f64>::zeros(2)];
        let input = MultiBlockAloInput {
            n_obs: 1,
            n_coordinates: 2,
            coordinate_designs: &coordinate_designs,
            coordinate_coefficient_ranges: &coordinate_coefficient_ranges,
            penalized_hessian: &penalized_hessian,
            observed_hessians: &observed_hessians,
            score_covariances: &score_covariances,
            scores: &scores,
            coordinate_values: &coordinate_values,
        };

        let result = compute_multiblock_alo(&input)
            .expect("trace-one but invertible deletion system must be solved exactly");
        let roundoff = floating_point_gamma(16);
        assert!((result.leverage[0] - 1.0).abs() <= roundoff);
        assert!((result.eta_tilde[0][0] - 1.0).abs() <= roundoff);
        assert!((result.eta_tilde[0][1] + 1.0).abs() <= roundoff);
    }
}
