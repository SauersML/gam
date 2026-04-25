use crate::basis::BasisOptions;
use crate::custom_family::{
    BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative,
    CustomFamilyJointDesignChannel, CustomFamilyJointDesignPairContribution,
    CustomFamilyJointPsiOperator, CustomFamilyPsiDesignAction, CustomFamilyPsiLinearMapRef,
    CustomFamilyPsiSecondDesignAction, CustomFamilyWarmStart, ExactNewtonJointPsiDirectCache,
    ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms, ExactNewtonJointPsiWorkspace,
    ExactNewtonOuterCurvature, FamilyEvaluation, ParameterBlockSpec, ParameterBlockState,
    PenaltyMatrix, PsiDesignMap, build_embedded_dense_psi_operator,
    build_rowwise_kronecker_psi_operator, evaluate_custom_family_joint_hyper,
    evaluate_custom_family_joint_hyper_efs, first_psi_linear_map, fit_custom_family,
    resolve_custom_family_x_psi_map, resolve_custom_family_x_psi_psi_map, second_psi_linear_map,
    shared_dense_arc, slice_joint_into_block_working_sets, weighted_crossprod_psi_maps,
    wrap_spatial_implicit_psi_operator,
};
use crate::faer_ndarray::{FaerEigh, fast_xt_diag_x};
use crate::families::bernoulli_marginal_slope::erfcx_nonnegative;
use crate::families::gamlss::{
    SelectedWiggleBasis, WiggleBlockConfig, monotone_wiggle_basis_with_derivative_order,
    monotone_wiggle_nonnegative_constraints, select_wiggle_basis_from_seed,
    validate_monotone_wiggle_beta_nonnegative,
};
use crate::families::scale_design::{
    build_scale_deviation_operator, build_scale_deviation_transform_design,
    infer_non_intercept_start_design,
};
use crate::matrix::{
    BlockDesignOperator, DenseDesignMatrix, DesignBlock, DesignMatrix, EmbeddedColumnBlock,
    EmbeddedSquareBlock, MultiChannelOperator, RowwiseKroneckerOperator,
};
use crate::mixture_link::{
    component_inverse_link_jet, inverse_link_jet_for_inverse_link,
    inverse_link_pdffourth_derivative_for_inverse_link,
    inverse_link_pdfthird_derivative_for_inverse_link,
};
use crate::pirls::LinearInequalityConstraints;
use crate::probability::{normal_cdf, normal_pdf};
use crate::smooth::{
    ExactJointHyperSetup, SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords,
    TermCollectionDesign, TermCollectionSpec, build_term_collection_design,
    freeze_term_collection_from_design, optimize_spatial_length_scale_exact_joint,
    spatial_length_scale_term_indices, try_build_spatial_log_kappa_derivativeinfo_list,
};
use crate::solver::estimate::UnifiedFitResult;
use crate::solver::estimate::{
    FitGeometry, ensure_finite_scalar_estimation, validate_all_finite_estimation,
};
use crate::terms::construction::kronecker_product;
use crate::types::{InverseLink, LinkFunction};
use ndarray::{Array1, Array2, s};
use statrs::function::erf::erfc;
use std::collections::HashMap;
use std::sync::Arc;

#[inline]
fn softplus(x: f64) -> f64 {
    if x.is_nan() {
        f64::NAN
    } else if x == f64::INFINITY {
        f64::INFINITY
    } else if x == f64::NEG_INFINITY {
        0.0
    } else if x >= 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
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

/// Maximum exponent argument for overflow-safe exp in the survival chain.
/// exp(500) ≈ 1.4e217 leaves ~91 orders of magnitude of headroom before
/// reaching MAX ≈ 1.8e308, sufficient for any reasonable multiplicative chain.
const EXP_NEG_STABLE_MAX_ARG: f64 = 500.0;

/// Overflow-safe exp(-x): guards against overflow when x is very negative
/// (i.e. exp(-x) very large) by capping the exponent at +500, but allows
/// natural IEEE 754 underflow to 0.0 when x is very positive.
///
/// The one-sided guard is critical: for x = 701 the correct value is
/// exp(-701) ≈ 5e-305 (essentially zero), NOT exp(-500) ≈ 7e-218.
/// A two-sided clamp would return the latter, which is ~1e87× too large
/// and destroys far-tail exact derivatives.
#[inline]
fn exp_neg_stable(x: f64) -> f64 {
    (-x).min(EXP_NEG_STABLE_MAX_ARG).exp()
}

#[inline]
fn exp_sigma_inverse_from_eta_scalar(eta: f64) -> f64 {
    exp_neg_stable(eta)
}

/// Layer 3 defense: clamp products that overflow to ±inf back to ±MAX.
/// With layer 1 (exp_neg_stable) active this should not trigger in normal
/// operation; it guards against edge cases where two independently large
/// (but sub-overflow) factors multiply to exceed MAX.
#[inline]
fn safe_product(lhs: f64, rhs: f64) -> f64 {
    if lhs == 0.0 || rhs == 0.0 {
        0.0
    } else {
        let v = lhs * rhs;
        if v == f64::INFINITY {
            f64::MAX
        } else if v == f64::NEG_INFINITY {
            f64::MIN
        } else {
            v
        }
    }
}

/// Layer 3 defense: when a + b produces NaN from inf + (-inf), return 0.
///
/// In the survival chain, g = d_raw + qdot1 where both terms scale as
/// inv_sigma × (something).  When inv_sigma is very large, both terms
/// overflow independently even though their sum is finite.  Mapping
/// the cancellation to 0 is conservative: it says "the correction is
/// negligible", and the monotonicity guard in exact_row_kernel will floor
/// g upward if needed.
#[inline]
fn safe_sum2(a: f64, b: f64) -> f64 {
    let sum = a + b;
    if sum.is_nan() {
        if a == 0.0 {
            return b;
        } else if b == 0.0 {
            return a;
        }
        if (a == f64::INFINITY && b == f64::NEG_INFINITY)
            || (a == f64::NEG_INFINITY && b == f64::INFINITY)
        {
            return 0.0;
        }
        sum
    } else {
        sum
    }
}

#[inline]
fn safe_sum3(a: f64, b: f64, c: f64) -> f64 {
    safe_sum2(safe_sum2(a, b), c)
}

#[inline]
fn safe_product3(a: f64, b: f64, c: f64) -> f64 {
    let mut factors = [a, b, c];
    factors.sort_by(|lhs, rhs| lhs.abs().total_cmp(&rhs.abs()));
    safe_product(safe_product(factors[0], factors[1]), factors[2])
}

fn safe_hadamard_product(lhs: &Array1<f64>, rhs: &Array1<f64>) -> Result<Array1<f64>, String> {
    if lhs.len() != rhs.len() {
        return Err(format!(
            "safe_hadamard_product length mismatch: lhs has {}, rhs has {}",
            lhs.len(),
            rhs.len()
        ));
    }
    let out = Array1::from_shape_fn(lhs.len(), |i| safe_product(lhs[i], rhs[i]));
    if out.iter().any(|value| value.is_nan()) {
        return Err("safe_hadamard_product produced NaN values".to_string());
    }
    Ok(out)
}

fn safe_linear_combo2_arrays(
    a: &Array1<f64>,
    b: &Array1<f64>,
    c: &Array1<f64>,
    d: &Array1<f64>,
) -> Result<Array1<f64>, String> {
    if a.len() != b.len() || a.len() != c.len() || a.len() != d.len() {
        return Err(format!(
            "safe_linear_combo2_arrays length mismatch: a={}, b={}, c={}, d={}",
            a.len(),
            b.len(),
            c.len(),
            d.len()
        ));
    }
    let out = Array1::from_shape_fn(a.len(), |i| {
        safe_sum2(safe_product(a[i], b[i]), safe_product(c[i], d[i]))
    });
    if out.iter().any(|value| value.is_nan()) {
        return Err("safe_linear_combo2_arrays produced NaN values".to_string());
    }
    Ok(out)
}

fn sanitize_survival_weight_vector(weights: &Array1<f64>) -> Array1<f64> {
    Array1::from_shape_fn(weights.len(), |i| {
        let value = weights[i];
        if value.is_finite() {
            value
        } else if value == f64::INFINITY {
            f64::MAX
        } else if value == f64::NEG_INFINITY {
            f64::MIN
        } else {
            0.0
        }
    })
}

fn safe_fast_xt_diag_x(x: &Array2<f64>, weights: &Array1<f64>) -> Array2<f64> {
    let sanitized = sanitize_survival_weight_vector(weights);
    fast_xt_diag_x(x, &sanitized)
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

#[derive(Clone, Copy)]
struct StableDifference {
    value: f64,
    roundoff_slack: f64,
    operand_scale: f64,
}

#[inline]
fn two_diff(lhs: f64, rhs: f64) -> (f64, f64) {
    let high = lhs - rhs;
    let z = high - lhs;
    let low = (lhs - (high - z)) - (rhs + z);
    (high, low)
}

#[inline]
fn compensated_difference(lhs: f64, rhs: f64) -> StableDifference {
    let operand_scale = lhs.abs().max(rhs.abs());
    if lhs.is_nan() || rhs.is_nan() {
        return StableDifference {
            value: f64::NAN,
            roundoff_slack: 0.0,
            operand_scale,
        };
    }
    if !lhs.is_finite() || !rhs.is_finite() {
        // Compensated subtraction is undefined for infinite operands.
        // Use a conservative slack: if the difference rounded to 0 (from
        // inf − inf via safe_sum2), the true value could be anywhere, so
        // make the slack large enough that the monotonicity guard will
        // clamp rather than hard-error.
        let diff = safe_sum2(lhs, -rhs);
        let slack = if diff == 0.0 && operand_scale > 0.0 {
            // inf − inf ≈ 0: the true difference is unknown; use a large
            // slack so the guard floor can absorb it.
            operand_scale
        } else {
            // One finite, one infinite, or both same-sign infinite:
            // the result is ±inf or a well-defined finite value.
            0.0
        };
        return StableDifference {
            value: diff,
            roundoff_slack: slack,
            operand_scale,
        };
    }
    let (high, low) = two_diff(lhs, rhs);
    if !high.is_finite() {
        return StableDifference {
            value: high,
            roundoff_slack: 0.0,
            operand_scale,
        };
    }
    let value = high + low;
    // |low| is the exact rounding error of the final lhs − rhs subtraction.
    // The 128ε term bounds accumulated upstream error: d_raw and qdot each
    // pass through ~45 chained safe_product / safe_sum operations, giving
    // ≤90ε × operand_scale total propagated error.  128 rounds up to the
    // next power of two for a conservative margin.
    let roundoff_slack = low.abs() + 128.0 * f64::EPSILON * operand_scale.max(value.abs());
    StableDifference {
        value,
        roundoff_slack,
        operand_scale,
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
    let survival = probit_survival_value(eta);
    let log_survival = if eta >= 0.0 {
        -0.5 * eta * eta + (0.5 * erfcx_nonnegative(x)).ln()
    } else {
        survival.ln()
    };
    let ratio = if eta >= 0.0 {
        std::f64::consts::FRAC_2_SQRT_PI / (std::f64::consts::SQRT_2 * erfcx_nonnegative(x))
    } else {
        normal_pdf(eta) / survival
    };
    let dr = ratio * (ratio - eta);
    let ddr = 2.0 * ratio.powi(3) - 3.0 * eta * ratio.powi(2) + (eta * eta - 1.0) * ratio;
    let dddr = 6.0 * ratio.powi(4) - 12.0 * eta * ratio.powi(3)
        + (7.0 * eta * eta - 4.0) * ratio.powi(2)
        + (-eta * eta * eta + 3.0 * eta) * ratio;
    (log_survival, ratio, dr, ddr, dddr)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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
                &InverseLink::Standard(LinkFunction::CLogLog),
                z,
            )
            .expect("standard cloglog inverse-link third derivative should evaluate"),
            ResidualDistribution::Logistic => inverse_link_pdfthird_derivative_for_inverse_link(
                &InverseLink::Standard(LinkFunction::Logit),
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
                &InverseLink::Standard(LinkFunction::CLogLog),
                z,
            )
            .expect("standard cloglog inverse-link fourth derivative should evaluate"),
            ResidualDistribution::Logistic => inverse_link_pdffourth_derivative_for_inverse_link(
                &InverseLink::Standard(LinkFunction::Logit),
                z,
            )
            .expect("standard logit inverse-link fourth derivative should evaluate"),
        }
    }
}

#[inline]
fn residual_distribution_link(distribution: ResidualDistribution) -> LinkFunction {
    match distribution {
        ResidualDistribution::Gaussian => LinkFunction::Probit,
        ResidualDistribution::Gumbel => LinkFunction::CLogLog,
        ResidualDistribution::Logistic => LinkFunction::Logit,
    }
}

#[inline]
pub fn residual_distribution_inverse_link(distribution: ResidualDistribution) -> InverseLink {
    InverseLink::Standard(residual_distribution_link(distribution))
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
fn inverse_link_pdffourth_derivative(inverse_link: &InverseLink, eta: f64) -> Result<f64, String> {
    match inverse_link {
        InverseLink::Standard(LinkFunction::Probit) => {
            Ok(ResidualDistribution::Gaussian.pdffourth_derivative(eta))
        }
        InverseLink::Standard(LinkFunction::Logit) => {
            Ok(ResidualDistribution::Logistic.pdffourth_derivative(eta))
        }
        InverseLink::Standard(LinkFunction::CLogLog) => {
            Ok(ResidualDistribution::Gumbel.pdffourth_derivative(eta))
        }
        _ => crate::solver::mixture_link::inverse_link_pdffourth_derivative_for_inverse_link(
            inverse_link,
            eta,
        )
        .map_err(|e| format!("inverse link fourth-derivative evaluation failed at eta={eta}: {e}")),
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
    pub structural_monotonicity: bool,
    pub penalties: Vec<Array2<f64>>,
    /// Structural nullspace dimension of each penalty matrix.
    pub nullspace_dims: Vec<usize>,
    pub initial_log_lambdas: Option<Array1<f64>>,
    pub initial_beta: Option<Array1<f64>>,
}

pub(crate) fn structural_time_coefficient_constraints(
    design_derivative_exit: &Array2<f64>,
    derivative_offset_exit: &Array1<f64>,
    derivative_guard: f64,
) -> Result<Option<LinearInequalityConstraints>, String> {
    if design_derivative_exit.ncols() == 0 {
        return Ok(None);
    }
    let lower_bounds = structural_time_coefficient_lower_bounds(
        design_derivative_exit,
        derivative_offset_exit,
        derivative_guard,
    )?
    .ok_or_else(|| {
        "structural time coefficient constraints require derivative offsets to encode the derivative guard and a non-negative derivative basis".to_string()
    })?;
    Ok(lower_bound_constraints(&lower_bounds))
}

#[derive(Clone)]
pub struct CovariateBlockInput {
    pub design: DesignMatrix,
    pub offset: Array1<f64>,
    pub penalties: Vec<crate::solver::estimate::PenaltySpec>,
    /// Structural nullspace dimension of each penalty matrix.
    pub nullspace_dims: Vec<usize>,
    pub initial_log_lambdas: Option<Array1<f64>>,
    pub initial_beta: Option<Array1<f64>>,
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
    Static(CovariateBlockInput),
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
}

pub const DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD: f64 = 1e-6;

pub struct SurvivalLocationScaleTermFitResult {
    pub fit: UnifiedFitResult,
    pub resolved_thresholdspec: TermCollectionSpec,
    pub resolved_log_sigmaspec: TermCollectionSpec,
    pub threshold_design: TermCollectionDesign,
    pub log_sigma_design: TermCollectionDesign,
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
    pub outer_gradient_norm: f64,
    pub outer_converged: bool,
    pub covariance_conditional: Option<Array2<f64>>,
    pub geometry: Option<FitGeometry>,
}

#[derive(Clone)]
struct PreparedSurvivalLocationScaleModel {
    family: SurvivalLocationScaleFamily,
    blockspecs: Vec<ParameterBlockSpec>,
    time_transform: TimeIdentifiabilityTransform,
    k_time: usize,
    k_threshold: usize,
    k_log_sigma: usize,
    k_wiggle: usize,
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
            return Err(format!(
                "{label} rho length mismatch: got {}, expected {}",
                rho.len(),
                self.total()
            ));
        }
        Ok(())
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
            return Err("survival_fit.beta_link_wiggle requires link_wiggle_degree".to_string());
        }
    } else if link_wiggle_knots.is_some() || link_wiggle_degree.is_some() {
        return Err(
            "survival_fit link-wiggle metadata requires beta_link_wiggle coefficients".to_string(),
        );
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
    if let Some(lambdas_wiggle) = lambdas_linkwiggle.as_ref() {
        if beta_link_wiggle.is_none() {
            return Err("survival_fit.lambdas_linkwiggle requires beta_link_wiggle".to_string());
        }
        validate_all_finite_estimation(
            "survival_fit.lambdas_linkwiggle",
            lambdas_wiggle.iter().copied(),
        )
        .map_err(|e| e.to_string())?;
    }
    ensure_finite_scalar_estimation("survival_fit.log_likelihood", log_likelihood)
        .map_err(|e| e.to_string())?;
    ensure_finite_scalar_estimation("survival_fit.reml_score", reml_score)
        .map_err(|e| e.to_string())?;
    ensure_finite_scalar_estimation("survival_fit.stable_penalty_term", stable_penalty_term)
        .map_err(|e| e.to_string())?;
    ensure_finite_scalar_estimation("survival_fit.penalized_objective", penalized_objective)
        .map_err(|e| e.to_string())?;
    ensure_finite_scalar_estimation("survival_fit.outer_gradient_norm", outer_gradient_norm)
        .map_err(|e| e.to_string())?;

    let total_p = beta_time.len()
        + beta_threshold.len()
        + beta_log_sigma.len()
        + beta_link_wiggle.as_ref().map_or(0, |beta| beta.len());
    if let Some(cov) = covariance_conditional.as_ref() {
        validate_all_finite_estimation("survival_fit.covariance_conditional", cov.iter().copied())
            .map_err(|e| e.to_string())?;
        let (rows, cols) = cov.dim();
        if rows != total_p || cols != total_p {
            return Err(format!(
                "survival_fit.covariance_conditional must be {}x{}, got {}x{}",
                total_p, total_p, rows, cols
            ));
        }
    }
    if let Some(geom) = geometry.as_ref() {
        geom.validate_numeric_finiteness()
            .map_err(|e| e.to_string())?;
        let (rows, cols) = geom.penalized_hessian.dim();
        if rows != total_p || cols != total_p {
            return Err(format!(
                "survival_fit.geometry.penalized_hessian must be {}x{}, got {}x{}",
                total_p, total_p, rows, cols
            ));
        }
        if geom.working_weights.len() != geom.working_response.len() {
            return Err(format!(
                "survival_fit.geometry working length mismatch: weights={}, response={}",
                geom.working_weights.len(),
                geom.working_response.len()
            ));
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
            survival_link_wiggle_knots: link_wiggle_knots,
            survival_link_wiggle_degree: link_wiggle_degree,
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
    time_derivative_offset_exit: Arc<Array1<f64>>,
    time_wiggle_knots: Option<Array1<f64>>,
    time_wiggle_degree: Option<usize>,
    time_wiggle_ncols: usize,
    time_coefficient_lower_bounds: Option<Array1<f64>>,
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
    policy: crate::resource::ResourcePolicy,
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
    /// Entry-only fourth derivative: d⁴ ell / dq0⁴ = w * r'''(u0).
    d4_q0: f64,
    /// Exit-only derivative: d ell / dq1.
    d1_q1: f64,
    /// Exit-only second derivative: d² ell / dq1².
    d2_q1: f64,
    /// Exit-only third derivative: d³ ell / dq1³.
    d3_q1: f64,
    /// Exit-only fourth derivative: d⁴ ell / dq1⁴.
    d4_q1: f64,
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
    /// d⁴ ell / d(h0)⁴ — the 4th derivative of ℓ w.r.t. the entry time
    /// predictor h0. This is the bilinear coefficient for D²H[u,v] in the
    /// time-time block of the outer Hessian. Previously approximated via
    /// 3rd-derivative products; now computed exactly.
    d2_h_h0: f64,
    /// d⁴ ell / d(h1)⁴ — analogous to d2_h_h0 for the exit side.
    d2_h_h1: f64,
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
}

struct SurvivalJointQuantities {
    d1_q: Array1<f64>,
    d2_q: Array1<f64>,
    d3_q: Array1<f64>,
    /// Entry-only derivatives of ell w.r.t. q0.
    d1_q0: Array1<f64>,
    d2_q0: Array1<f64>,
    d3_q0: Array1<f64>,
    d4_q0: Array1<f64>,
    /// Exit-only derivatives of ell w.r.t. q1.
    d1_q1: Array1<f64>,
    d2_q1: Array1<f64>,
    d3_q1: Array1<f64>,
    d4_q1: Array1<f64>,
    /// Exit-only derivatives of ell w.r.t. qdot1 = dq/dt.
    d1_qdot1: Array1<f64>,
    d2_qdot1: Array1<f64>,
    h_time_h0: Array1<f64>,
    h_time_h1: Array1<f64>,
    h_time_d: Array1<f64>,
    d_h_h0: Array1<f64>,
    d_h_h1: Array1<f64>,
    d_h_d: Array1<f64>,
    /// d⁴ℓ/d(h0)⁴ for the exact bilinear D²H[u,v] time-time coefficient.
    d2_h_h0: Array1<f64>,
    /// d⁴ℓ/d(h1)⁴ for the exact bilinear D²H[u,v] time-time coefficient.
    d2_h_h1: Array1<f64>,
    /// Exit-side dq/d(eta_t) = -exp(-eta_ls_exit).
    dq_t: Array1<f64>,
    /// Exit-side dq/d(eta_ls).
    dq_ls: Array1<f64>,
    d2q_tls: Array1<f64>,
    d2q_ls: Array1<f64>,
    d3q_tls_ls: Array1<f64>,
    d3q_ls: Array1<f64>,
    d4q_tls_ls_ls: Array1<f64>,
    d4q_ls: Array1<f64>,
    /// Entry-side dq0/d(eta_t_entry) = -exp(-eta_ls_entry) (only for time-varying).
    dq_t_entry: Option<Array1<f64>>,
    /// Entry-side q-chain derivatives at entry (only for time-varying sigma).
    dq_ls_entry: Option<Array1<f64>>,
    d2q_tls_entry: Option<Array1<f64>>,
    d2q_ls_entry: Option<Array1<f64>>,
    d3q_tls_ls_entry: Option<Array1<f64>>,
    d3q_ls_entry: Option<Array1<f64>>,
    d4q_tls_ls_ls_entry: Option<Array1<f64>>,
    d4q_ls_entry: Option<Array1<f64>>,
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
}

struct SurvivalJointPsiDirection {
    block_idx: usize,
    local_idx: usize,
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

struct SurvivalJointPsiSecondDrifts {
    x_t_exit_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    x_t_entry_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    x_ls_exit_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    x_ls_entry_ab_action: Option<CustomFamilyPsiSecondDesignAction>,
    x_t_exit_ab: Option<Array2<f64>>,
    x_t_entry_ab: Option<Array2<f64>>,
    x_ls_exit_ab: Option<Array2<f64>>,
    x_ls_entry_ab: Option<Array2<f64>>,
    z_t_exit_ab: Array1<f64>,
    z_t_entry_ab: Array1<f64>,
    z_ls_exit_ab: Array1<f64>,
    z_ls_entry_ab: Array1<f64>,
}

struct SurvivalExactNewtonJointPsiWorkspace {
    family: SurvivalLocationScaleFamily,
    block_states: Vec<ParameterBlockState>,
    derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    joint_quantities: SurvivalJointQuantities,
    psi_directions: ExactNewtonJointPsiDirectCache<SurvivalJointPsiDirection>,
}

fn split_survival_psi_design(
    x_psi: &Array2<f64>,
    n: usize,
    time_varying: bool,
    label: &str,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    if time_varying {
        if x_psi.nrows() != 2 * n && x_psi.nrows() != 3 * n {
            return Err(format!(
                "{label} stacked psi design row mismatch: got {}, expected {} or {}",
                x_psi.nrows(),
                2 * n,
                3 * n,
            ));
        }
        Ok((
            x_psi.slice(s![0..n, ..]).to_owned(),
            x_psi.slice(s![n..2 * n, ..]).to_owned(),
        ))
    } else {
        if x_psi.nrows() != n {
            return Err(format!(
                "{label} psi design row mismatch: got {}, expected {}",
                x_psi.nrows(),
                n
            ));
        }
        Ok((x_psi.clone(), x_psi.clone()))
    }
}

impl SurvivalLocationScaleFamily {
    const BLOCK_TIME: usize = 0;
    const BLOCK_THRESHOLD: usize = 1;
    const BLOCK_LOG_SIGMA: usize = 2;
    const BLOCK_LINK_WIGGLE: usize = 3;

    #[inline]
    fn time_wiggle_range(&self) -> std::ops::Range<usize> {
        let p_total = self.x_time_exit.ncols();
        let p_w = self.time_wiggle_ncols.min(p_total);
        p_total - p_w..p_total
    }

    #[inline]
    fn time_derivative_lower_bound(&self) -> f64 {
        assert!(self.derivative_guard.is_finite() && self.derivative_guard > 0.0);
        self.derivative_guard
    }

    fn max_feasible_time_step(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        let Some(lower_bounds) = self.time_coefficient_lower_bounds.as_ref() else {
            return Err(
                "survival location-scale time block missing structural coefficient lower bounds"
                    .to_string(),
            );
        };
        if beta.len() != lower_bounds.len() || delta.len() != lower_bounds.len() {
            return Err(format!(
                "survival location-scale time-step lower-bound dimension mismatch: beta={}, delta={}, bounds={}",
                beta.len(),
                delta.len(),
                lower_bounds.len()
            ));
        }
        let mut alpha = 1.0f64;
        for j in 0..lower_bounds.len() {
            let lower_bound = lower_bounds[j];
            if !lower_bound.is_finite() {
                continue;
            }
            let slack = beta[j] - lower_bound;
            if slack < -1e-10 {
                return Err(format!(
                    "survival location-scale current time coefficient violates structural lower bound at coefficient {j}: slack={slack:.3e}"
                ));
            }
            let drift = delta[j];
            if drift < 0.0 {
                alpha = alpha.min((slack / -drift).clamp(0.0, 1.0));
            }
        }
        // Apply the 0.995 buffer to the structural coefficient lower-bound
        // throttle before considering the derivative-guard gate.  Lower-bound
        // throttling is a smooth policy: approach the bound, re-solve on the
        // active face.  The derivative-guard policy below is different.
        let coefficient_alpha = if alpha >= 1.0 {
            1.0
        } else {
            (0.995 * alpha).clamp(0.0, 1.0)
        };
        // Per-row derivative-guard gate: d_raw = offset + x_time_deriv·beta
        // must stay at or above derivative_guard along the path. Derivation of
        // the binary gate policy versus a throttle:
        //
        //   Let g_r(α) = offset_r + x_time_deriv[r,:]·(beta + α·delta) for row r.
        //   Feasibility requires g_r(α) ≥ derivative_guard for every row.
        //   With `drift_r = x_time_deriv[r,:]·delta < 0`, the maximal α that
        //   keeps g_r at exactly the guard is
        //       α*_r = (current_r − guard) / (−drift_r)          [slack / |drift|]
        //   The smallest α*_r across rows is the per-row throttle used for
        //   structural *coefficient* lower bounds above.
        //
        //   For the derivative guard the throttle policy is WRONG.  The
        //   guard is a structural positivity requirement on the hazard
        //   derivative, not a soft lower bound on a coefficient.  A throttled
        //   step ending at g_r = 0.995·(guard+drift·α*) leaves the solver on
        //   the guard boundary where any rounding error pushes it below, and
        //   numerical differentiation / Newton steps taken from that state
        //   become unstable.  The correct semantic is:
        //       if the FULL step (α = 1) would violate the guard anywhere,
        //       reject the step (α = 0) and let the outer solver pick a
        //       non-violating direction.  Otherwise the guard is inactive.
        //
        //   This is exactly the "binary gate" interpretation the
        //   time_block_feasible_step_stays_inside_derivative_guard test
        //   encodes: beta = [0.1], delta = [-2.0], guard = 1e-8, x_deriv rows
        //   all = 1.  current = 0.1, drift = -2.0, so α*_r = 0.05 and the
        //   throttle would return 0.995·0.05 = 0.04975.  The test asserts
        //   α = 0.0 exactly — gate semantic.
        //
        //   Implementation: compute the guard's predicate at α = 1.
        //   If any row has g_r(1) = current_r + drift_r < guard, return 0.
        let deriv = self.x_time_deriv.as_ref();
        if deriv.ncols() == beta.len() {
            let guard = self.derivative_guard;
            for row in 0..deriv.nrows() {
                let row_view = deriv.row(row);
                let current = self.time_derivative_offset_exit[row] + row_view.dot(beta);
                let drift = row_view.dot(delta);
                let slack = current - guard;
                if slack < -1e-10 {
                    return Err(format!(
                        "survival location-scale current time derivative violates guard at row {row}: slack={slack:.3e}"
                    ));
                }
                // Full-step value of g_r at α = 1; if below the guard, gate closes.
                if current + drift < guard {
                    return Ok(Some(0.0));
                }
            }
        }
        Ok(Some(coefficient_alpha))
    }

    fn max_feasible_link_wiggle_step(
        &self,
        beta: &Array1<f64>,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        if beta.len() != delta.len() {
            return Err(format!(
                "survival location-scale linkwiggle-step dimension mismatch: beta={}, delta={}",
                beta.len(),
                delta.len()
            ));
        }
        let mut alpha = 1.0f64;
        for j in 0..beta.len() {
            let slack = beta[j];
            if slack < -1e-10 {
                return Err(format!(
                    "survival location-scale current linkwiggle block violates nonnegativity at coefficient {j}: beta={slack:.3e}"
                ));
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
            return Err(format!(
                "survival linkwiggle basis/beta mismatch: B={} B'={} B''={} B'''={} betaw={}",
                basis.ncols(),
                basis_d1.ncols(),
                basis_d2.ncols(),
                basis_d3.ncols(),
                beta_w.len()
            ));
        }
        let dq_dq0 = basis_d1.dot(&beta_w) + 1.0;
        let d2q_dq02 = basis_d2.dot(&beta_w);
        let d3q_dq03 = basis_d3.dot(&beta_w);
        let d4q_dq04 = survival_wiggle_fourth_q(q0, knots, degree, beta_w)?;
        Ok(Some(SurvivalWiggleGeometry {
            basis,
            basis_d1,
            basis_d2,
            dq_dq0,
            d2q_dq02,
            d3q_dq03,
            d4q_dq04,
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
            return Err(format!(
                "survival timewiggle basis/beta mismatch: B={} B'={} B''={} B'''={} betaw={}",
                basis.ncols(),
                basis_d1.ncols(),
                basis_d2.ncols(),
                basis_d3.ncols(),
                beta_w.len()
            ));
        }
        let dq = basis_d1.dot(&beta_w) + 1.0;
        let d2 = basis_d2.dot(&beta_w);
        let d3 = basis_d3.dot(&beta_w);
        Ok(Some(SurvivalWiggleGeometry {
            basis,
            basis_d1,
            basis_d2,
            dq_dq0: dq,
            d2q_dq02: d2,
            d3q_dq03: d3,
            d4q_dq04: Array1::zeros(h0.len()),
        }))
    }

    /// Returns
    /// `(h0, h1, d_raw, eta_t_exit, eta_ls_exit, eta_t_entry, eta_ls_entry,
    ///   eta_t_deriv_exit, eta_ls_deriv_exit, etaw)`.
    ///
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
            return Err(format!(
                "SurvivalLocationScaleFamily expects {} blocks, got {}",
                self.expected_blocks(),
                block_states.len()
            ));
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
            return Err("survival location-scale time eta dimension mismatch".to_string());
        }
        // For time-varying blocks the stacked design is
        // [exit_design; entry_design; derivative_exit_design], giving eta of
        // length 3n. For time-invariant blocks eta is length n.
        let (eta_t_exit, eta_t_entry, eta_t_deriv_exit) = if self.x_threshold_entry.is_some() {
            if eta_t_raw.len() != 3 * n {
                return Err(format!(
                    "time-varying threshold eta length mismatch: got {}, expected {}",
                    eta_t_raw.len(),
                    3 * n
                ));
            }
            (
                eta_t_raw.slice(s![0..n]),
                eta_t_raw.slice(s![n..2 * n]),
                Some(eta_t_raw.slice(s![2 * n..3 * n])),
            )
        } else {
            if eta_t_raw.len() != n {
                return Err(format!(
                    "threshold eta length mismatch: got {}, expected {n}",
                    eta_t_raw.len()
                ));
            }
            (eta_t_raw.slice(s![0..n]), eta_t_raw.slice(s![0..n]), None)
        };
        let (eta_ls_exit, eta_ls_entry, eta_ls_deriv_exit) = if self.x_log_sigma_entry.is_some() {
            if eta_ls_raw.len() != 3 * n {
                return Err(format!(
                    "time-varying log-sigma eta length mismatch: got {}, expected {}",
                    eta_ls_raw.len(),
                    3 * n
                ));
            }
            (
                eta_ls_raw.slice(s![0..n]),
                eta_ls_raw.slice(s![n..2 * n]),
                Some(eta_ls_raw.slice(s![2 * n..3 * n])),
            )
        } else {
            if eta_ls_raw.len() != n {
                return Err(format!(
                    "log-sigma eta length mismatch: got {}, expected {n}",
                    eta_ls_raw.len()
                ));
            }
            (eta_ls_raw.slice(s![0..n]), eta_ls_raw.slice(s![0..n]), None)
        };
        if let Some(w) = etaw
            && w.len() != n
        {
            return Err("survival location-scale wiggle eta dimension mismatch".to_string());
        }
        Ok((
            eta_time.slice(s![0..n]),
            eta_time.slice(s![n..2 * n]),
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
        let mut d1_q = Array1::<f64>::zeros(n);
        let mut d2_q = Array1::<f64>::zeros(n);
        let mut d3_q = Array1::<f64>::zeros(n);
        let mut d1_q0 = Array1::<f64>::zeros(n);
        let mut d2_q0 = Array1::<f64>::zeros(n);
        let mut d3_q0 = Array1::<f64>::zeros(n);
        let mut d4_q0 = Array1::<f64>::zeros(n);
        let mut d1_q1 = Array1::<f64>::zeros(n);
        let mut d2_q1 = Array1::<f64>::zeros(n);
        let mut d3_q1 = Array1::<f64>::zeros(n);
        let mut d4_q1 = Array1::<f64>::zeros(n);
        let mut d1_qdot1 = Array1::<f64>::zeros(n);
        let mut d2_qdot1 = Array1::<f64>::zeros(n);
        let mut h_time_h0 = Array1::<f64>::zeros(n);
        let mut h_time_h1 = Array1::<f64>::zeros(n);
        let mut h_time_d = Array1::<f64>::zeros(n);
        let mut d_h_h0 = Array1::<f64>::zeros(n);
        let mut d_h_h1 = Array1::<f64>::zeros(n);
        let mut d_h_d = Array1::<f64>::zeros(n);
        let mut d2_h_h0 = Array1::<f64>::zeros(n);
        let mut d2_h_h1 = Array1::<f64>::zeros(n);

        for i in 0..n {
            let state = self.row_predictor_state(
                dynamic.h_entry[i],
                dynamic.h_exit[i],
                dynamic.hdot_exit[i],
                dynamic.q_entry[i],
                dynamic.q_exit[i],
                dynamic.qdot_exit[i],
            );
            let Some(row) = self.row_derivatives_rescaled(i, state, deriv_log_scale)? else {
                continue;
            };
            d1_q[i] = row.d1_q;
            d2_q[i] = row.d2_q;
            d3_q[i] = row.d3_q;
            d1_q0[i] = row.d1_q0;
            d2_q0[i] = row.d2_q0;
            d3_q0[i] = row.d3_q0;
            d4_q0[i] = row.d4_q0;
            d1_q1[i] = row.d1_q1;
            d2_q1[i] = row.d2_q1;
            d3_q1[i] = row.d3_q1;
            d4_q1[i] = row.d4_q1;
            d1_qdot1[i] = row.d1_qdot1;
            d2_qdot1[i] = row.d2_qdot1;
            h_time_h0[i] = row.h_time_h0;
            h_time_h1[i] = row.h_time_h1;
            h_time_d[i] = row.h_time_d;
            d_h_h0[i] = row.d_h_h0;
            d_h_h1[i] = row.d_h_h1;
            d_h_d[i] = row.d_h_d;
            d2_h_h0[i] = row.d2_h_h0;
            d2_h_h1[i] = row.d2_h_h1;
        }

        Ok(SurvivalJointQuantities {
            d1_q,
            d2_q,
            d3_q,
            d1_q0,
            d2_q0,
            d3_q0,
            d4_q0,
            d1_q1,
            d2_q1,
            d3_q1,
            d4_q1,
            d1_qdot1,
            d2_qdot1,
            h_time_h0,
            h_time_h1,
            h_time_d,
            d_h_h0,
            d_h_h1,
            d_h_d,
            d2_h_h0,
            d2_h_h1,
            dq_t: dynamic.dq_t_exit.clone(),
            dq_ls: dynamic.dq_ls_exit.clone(),
            d2q_tls: dynamic.d2q_tls_exit.clone(),
            d2q_ls: dynamic.d2q_ls_exit.clone(),
            d3q_tls_ls: dynamic.d3q_tls_ls_exit.clone(),
            d3q_ls: dynamic.d3q_ls_exit.clone(),
            d4q_tls_ls_ls: dynamic.d4q_tls_ls_ls_exit.clone(),
            d4q_ls: dynamic.d4q_ls_exit.clone(),
            dq_t_entry: Some(dynamic.dq_t_entry.clone()),
            dq_ls_entry: Some(dynamic.dq_ls_entry.clone()),
            d2q_tls_entry: Some(dynamic.d2q_tls_entry.clone()),
            d2q_ls_entry: Some(dynamic.d2q_ls_entry.clone()),
            d3q_tls_ls_entry: Some(dynamic.d3q_tls_ls_entry.clone()),
            d3q_ls_entry: Some(dynamic.d3q_ls_entry.clone()),
            d4q_tls_ls_ls_entry: Some(dynamic.d4q_tls_ls_ls_entry.clone()),
            d4q_ls_entry: Some(dynamic.d4q_ls_entry.clone()),
            dqdot_t: dynamic.dqdot_t.clone(),
            dqdot_ls: dynamic.dqdot_ls.clone(),
            dqdot_td: dynamic.dqdot_td.clone(),
            dqdot_lsd: dynamic.dqdot_lsd.clone(),
            d2qdot_tt: dynamic.d2qdot_tt.clone(),
            d2qdot_tls: dynamic.d2qdot_tls.clone(),
            d2qdot_ttd: dynamic.d2qdot_ttd.clone(),
            d2qdot_tlsd: dynamic.d2qdot_tlsd.clone(),
            d2qdot_ls: dynamic.d2qdot_ls.clone(),
            d2qdot_lstd: dynamic.d2qdot_lstd.clone(),
            d2qdot_lslsd: dynamic.d2qdot_lslsd.clone(),
        })
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
            return Err(format!(
                "SurvivalLocationScaleFamily joint psi direction expects {} blocks and derivative lists, got {} and {}",
                self.expected_blocks(),
                block_states.len(),
                derivative_blocks.len()
            ));
        }

        let n = self.n;
        let pt = self.x_threshold.ncols();
        let pls = self.x_log_sigma.ncols();
        let beta_t = &block_states[Self::BLOCK_THRESHOLD].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;
        let _row_chunk_target_bytes = self.policy.row_chunk_target_bytes;
        let t_time_varying = self.x_threshold_entry.is_some();
        let ls_time_varying = self.x_log_sigma_entry.is_some();

        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for (local_idx, deriv) in block_derivs.iter().enumerate() {
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
                                    z_t_exit_psi = exit.dot(beta_t);
                                    z_t_entry_psi = entry.dot(beta_t);
                                    x_t_exit_psi = Some(exit);
                                    x_t_entry_psi = Some(entry);
                                }
                                PsiDesignMap::Zero { .. } => {}
                                PsiDesignMap::Second { .. } => {
                                    return Err(
                                        "SurvivalLocationScaleFamily threshold: unexpected Second variant from _psi_map"
                                            .to_string(),
                                    );
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
                                    z_ls_exit_psi = exit.dot(beta_ls);
                                    z_ls_entry_psi = entry.dot(beta_ls);
                                    x_ls_exit_psi = Some(exit);
                                    x_ls_entry_psi = Some(entry);
                                }
                                PsiDesignMap::Zero { .. } => {}
                                PsiDesignMap::Second { .. } => {
                                    return Err(
                                        "SurvivalLocationScaleFamily log-sigma: unexpected Second variant from _psi_map"
                                            .to_string(),
                                    );
                                }
                            }
                        }
                        _ => return Ok(None),
                    }
                    return Ok(Some(SurvivalJointPsiDirection {
                        block_idx,
                        local_idx,
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

    fn exact_newton_joint_psisecond_design_drifts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_a: &SurvivalJointPsiDirection,
        psi_b: &SurvivalJointPsiDirection,
    ) -> Result<SurvivalJointPsiSecondDrifts, String> {
        let n = self.n;
        let pt = self.x_threshold.ncols();
        let pls = self.x_log_sigma.ncols();
        let beta_t = &block_states[Self::BLOCK_THRESHOLD].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;
        let t_time_varying = self.x_threshold_entry.is_some();
        let ls_time_varying = self.x_log_sigma_entry.is_some();

        let mut x_t_exit_ab_action = None;
        let mut x_t_entry_ab_action = None;
        let mut x_ls_exit_ab_action = None;
        let mut x_ls_entry_ab_action = None;
        let mut x_t_exit_ab = None;
        let mut x_t_entry_ab = None;
        let mut x_ls_exit_ab = None;
        let mut x_ls_entry_ab = None;

        if psi_a.block_idx == psi_b.block_idx {
            let deriv = &derivative_blocks[psi_a.block_idx][psi_a.local_idx];
            let deriv_b = &derivative_blocks[psi_b.block_idx][psi_b.local_idx];
            match psi_a.block_idx {
                Self::BLOCK_THRESHOLD => {
                    let total_rows = if t_time_varying { 3 * n } else { n };
                    match resolve_custom_family_x_psi_psi_map(
                        deriv,
                        deriv_b,
                        psi_b.local_idx,
                        total_rows,
                        pt,
                        0..total_rows,
                        "SurvivalLocationScaleFamily threshold",
                        &self.policy,
                    )? {
                        PsiDesignMap::Second { action } => {
                            if t_time_varying {
                                x_t_exit_ab_action = Some(action.slice_rows(0..n)?);
                                x_t_entry_ab_action = Some(action.slice_rows(n..2 * n)?);
                            } else {
                                x_t_exit_ab_action = Some(action.clone());
                                x_t_entry_ab_action = Some(action);
                            }
                        }
                        PsiDesignMap::Dense { matrix } => {
                            let (exit, entry) = split_survival_psi_design(
                                matrix.as_ref(),
                                n,
                                t_time_varying,
                                "SurvivalLocationScaleFamily threshold",
                            )?;
                            x_t_exit_ab = Some(exit);
                            x_t_entry_ab = Some(entry);
                        }
                        PsiDesignMap::Zero { .. } => {}
                        PsiDesignMap::First { .. } => {
                            return Err(
                                "SurvivalLocationScaleFamily threshold: unexpected First variant from _psi_psi_map"
                                    .to_string(),
                            );
                        }
                    }
                }
                Self::BLOCK_LOG_SIGMA => {
                    let total_rows = if ls_time_varying { 3 * n } else { n };
                    match resolve_custom_family_x_psi_psi_map(
                        deriv,
                        deriv_b,
                        psi_b.local_idx,
                        total_rows,
                        pls,
                        0..total_rows,
                        "SurvivalLocationScaleFamily log-sigma",
                        &self.policy,
                    )? {
                        PsiDesignMap::Second { action } => {
                            if ls_time_varying {
                                x_ls_exit_ab_action = Some(action.slice_rows(0..n)?);
                                x_ls_entry_ab_action = Some(action.slice_rows(n..2 * n)?);
                            } else {
                                x_ls_exit_ab_action = Some(action.clone());
                                x_ls_entry_ab_action = Some(action);
                            }
                        }
                        PsiDesignMap::Dense { matrix } => {
                            let (exit, entry) = split_survival_psi_design(
                                matrix.as_ref(),
                                n,
                                ls_time_varying,
                                "SurvivalLocationScaleFamily log-sigma",
                            )?;
                            x_ls_exit_ab = Some(exit);
                            x_ls_entry_ab = Some(entry);
                        }
                        PsiDesignMap::Zero { .. } => {}
                        PsiDesignMap::First { .. } => {
                            return Err(
                                "SurvivalLocationScaleFamily log-sigma: unexpected First variant from _psi_psi_map"
                                    .to_string(),
                            );
                        }
                    }
                }
                _ => {}
            }
        }

        let z_t_exit_ab =
            second_psi_linear_map(x_t_exit_ab_action.as_ref(), x_t_exit_ab.as_ref(), n, pt)
                .forward_mul(beta_t.view());
        let z_t_entry_ab =
            second_psi_linear_map(x_t_entry_ab_action.as_ref(), x_t_entry_ab.as_ref(), n, pt)
                .forward_mul(beta_t.view());
        let z_ls_exit_ab =
            second_psi_linear_map(x_ls_exit_ab_action.as_ref(), x_ls_exit_ab.as_ref(), n, pls)
                .forward_mul(beta_ls.view());
        let z_ls_entry_ab = second_psi_linear_map(
            x_ls_entry_ab_action.as_ref(),
            x_ls_entry_ab.as_ref(),
            n,
            pls,
        )
        .forward_mul(beta_ls.view());
        Ok(SurvivalJointPsiSecondDrifts {
            x_t_exit_ab_action,
            x_t_entry_ab_action,
            x_ls_exit_ab_action,
            x_ls_entry_ab_action,
            x_t_exit_ab,
            x_t_entry_ab,
            x_ls_exit_ab,
            x_ls_entry_ab,
            z_t_exit_ab,
            z_t_entry_ab,
            z_ls_exit_ab,
            z_ls_entry_ab,
        })
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
            InverseLink::Standard(LinkFunction::Probit) => Ok((
                -0.5 * eta * eta - 0.5 * (2.0 * std::f64::consts::PI).ln(),
                -eta,
                -1.0,
                0.0,
                0.0,
            )),
            InverseLink::Standard(LinkFunction::Logit) => {
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
            InverseLink::Standard(LinkFunction::CLogLog) => {
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
            InverseLink::Standard(LinkFunction::Identity) => Ok((0.0, 0.0, 0.0, 0.0, 0.0)),
            _ => {
                let jet = inverse_link_jet_for_inverse_link(inverse_link, eta)
                    .map_err(|e| format!("inverse link evaluation failed at eta={eta}: {e}"))?;
                let f = jet.d1;
                if !(f.is_finite() && f > 0.0) {
                    return Err(format!(
                        "inverse-link pdf must be finite and positive, got {f} at eta={eta}"
                    ));
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
            InverseLink::Standard(LinkFunction::Probit) => {
                let (log_s, r, dr, ddr, dddr) = probit_log_survival_and_ratio_derivatives(eta);
                Ok((log_s, r, dr, ddr, dddr))
            }
            InverseLink::Standard(LinkFunction::Logit) => {
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
            InverseLink::Standard(LinkFunction::CLogLog) => {
                let t_val = eta.exp(); // for the function value (may be Inf)
                let t_deriv = (eta - deriv_log_scale).exp(); // for derivatives (finite when shifted)
                Ok((-t_val, t_deriv, t_deriv, t_deriv, t_deriv))
            }
            InverseLink::Standard(LinkFunction::Identity) => {
                let s = 1.0 - eta;
                if !(s.is_finite() && s > 0.0) {
                    return Err(format!(
                        "identity-link survival invalid at eta={eta}: S={s}"
                    ));
                }
                let inv = s.recip();
                Ok((s.ln(), inv, inv * inv, 2.0 * inv.powi(3), 6.0 * inv.powi(4)))
            }
            _ => {
                let jet = inverse_link_jet_for_inverse_link(inverse_link, eta)
                    .map_err(|e| format!("inverse link evaluation failed at eta={eta}: {e}"))?;
                let s = inverse_link_survival_probvalue(inverse_link, eta);
                if !(s.is_finite() && s > 0.0 && s <= 1.0) {
                    return Err(format!(
                        "inverse-link survival probability must lie in (0,1], got {s} at eta={eta}"
                    ));
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

    /// Exact `log(x)` value and first four derivatives on the positive domain.
    fn logwith_derivatives_positive(x: f64) -> (f64, f64, f64, f64, f64) {
        assert!(x.is_finite() && x > 0.0);
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
            return Err(format!(
                "survival location-scale event target must lie in [0,1] at row {row}, got {d}"
            ));
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

        let (log_s1, r1, dr1, ddr1, dddr1) =
            Self::exact_survival_neglog_derivatives_fourth_rescaled(
                &self.inverse_link,
                u1,
                deriv_log_scale,
            )
            .map_err(|e| {
                format!("inverse-link survival evaluation failed at row {row} exit: {e}")
            })?;

        let (logphi1, dlogphi1, d2logphi1, d3logphi1, d4logphi1) =
            Self::exact_log_pdf_derivatives_rescaled(&self.inverse_link, u1, deriv_log_scale)
                .map_err(|e| {
                    format!("inverse-link log-pdf evaluation failed at row {row} exit: {e}")
                })?;

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
            return Err(format!(
                "survival location-scale time derivative is non-finite at row {row}: d_eta/dt={g}"
            ));
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
        let legacy_slack = 1e-12
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
            return Err(format!(
                "survival location-scale monotonicity violated at row {row}: \
                 d_eta/dt={g:.3e} <= 0 (lower_bound={guard:.3e}) \
                 (operand_scale={:.3e}, roundoff_slack={roundoff_slack:.3e})",
                state.g_operand_scale
            ));
        }
        let (log_g, d_log_g, d2_log_g, d3_log_g, ..) = Self::logwith_derivatives_positive(g);

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
        // The row likelihood is written in terms of the survival values
        //
        //   S(u0),  S(u1),
        //
        // not in terms of the failure probability `mu = F(u)`.
        //
        // Numerically, reconstructing `S` as `1 - mu` is unsafe in the upper
        // tail. For cloglog/Gumbel in particular, fitted rows can legitimately
        // land near `S(u) ~ 1e-12`, where `mu` is already within a few ulps of 1.
        // Then:
        //
        //   S_direct  = exp(-exp(u))
        //   S_naive   = 1 - (1 - S_direct)
        //
        // and the latter loses the very quantity the objective differentiates.
        //
        // The exact score / Hessian algebra from the derivation assumes the row
        // objective and its derivatives are taken with respect to the *same*
        // scalar function
        //
        //   ell = w [ d(log f(u1) + log g) + (1-d) log S(u1) - log S(u0) ].
        //
        // So we evaluate both log-density and log-survival through the same
        // inverse-link-specific exact formulas used by the derivative algebra.

        // With
        //
        //   ell = w [ d(log f(u1) + log g) + (1-d) log S(u1) - log S(u0) ],
        //   u0 = q0 + h0,
        //   u1 = q1 + h1,
        //
        // the entry-only derivatives (w.r.t. q0):
        //
        //   ell_q0   = w r(u0)
        //   ell_q0q0 = w r'(u0)
        //   ell_q0q0q0 = w r''(u0)
        //   ell_q0q0q0q0 = w r'''(u0)        ← 4th-order entry derivative
        //
        // and exit-only derivatives (w.r.t. q1):
        //
        //   ell_q1   = w [ d d/du log f(u1) + (1-d) (-r(u1)) ]
        //   ell_q1q1 = w [ d d²/du² log f(u1) + (1-d) (-r'(u1)) ]
        //   ell_q1q1q1 = w [ d d³/du³ log f(u1) + (1-d) (-r''(u1)) ]
        //   ell_q1q1q1q1 = w [ d d⁴/du⁴ log f(u1) + (1-d) (-r'''(u1)) ]  ← 4th-order exit derivative
        //
        // When q0 = q1 = q (time-invariant blocks), ell_q = ell_q0 + ell_q1.
        //
        // Cross-Hessian d²ell/(dq0 dq1) = 0 because u0 depends only on q0
        // and u1 depends only on q1.
        //
        // The time-side partials follow from u0 = q0 + h0 and u1 = q1 + h1:
        //
        //   ell_h0   = ell_q0 = w r(u0)
        //   ell_h1   = ell_q1
        //   ell_h0q0 = w r'(u0)
        //   ell_h1q1 = w [ d d²/du² log f(u1) - (1-d) r'(u1) ]
        //
        // The 4th-order derivatives d4_q0 and d4_q1 are the m4 quantities
        // needed by the Arbogast chain rule for the outer REML Hessian.
        // They enter F_αβγδ = m4·u_α·u_β·u_γ·u_δ + ... in the (s,s,s,s)
        // and (ϑ,s,s,s) blocks. See response.md Section 6.
        // Use `event_mix` for d * (event term) + (1-d) * (censored term) to
        // avoid 0 * Inf = NaN when d ∈ {0, 1} and one branch is non-finite.
        let d1_q0 = kernel.w * kernel.r0;
        let d2_q0 = kernel.w * kernel.dr0;
        let d3_q0 = kernel.w * kernel.ddr0;
        let d4_q0 = kernel.w * kernel.dddr0;
        let d1_q1 = kernel.w * event_mix(kernel.d, kernel.dlogphi1, -kernel.r1);
        let d2_q1 = kernel.w * event_mix(kernel.d, kernel.d2logphi1, -kernel.dr1);
        let d3_q1 = kernel.w * event_mix(kernel.d, kernel.d3logphi1, -kernel.ddr1);
        let d4_q1 = kernel.w * event_mix(kernel.d, kernel.d4logphi1, -kernel.dddr1);
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
            d4_q0,
            d1_q1,
            d2_q1,
            d3_q1,
            d4_q1,
            d1_qdot1: kernel.w * kernel.d * kernel.d_log_g,
            d2_qdot1: kernel.w * kernel.d * kernel.d2_log_g,
            grad_time_eta_h0: kernel.w * kernel.r0,
            grad_time_eta_h1: kernel.w * event_mix(kernel.d, kernel.dlogphi1, -kernel.r1),
            grad_time_eta_d: kernel.w * kernel.d * kernel.d_log_g,
            h_time_h0: kernel.w * kernel.dr0,
            h_time_h1: kernel.w * event_mix(kernel.d, kernel.d2logphi1, -kernel.dr1),
            h_time_d: -kernel.w * kernel.d * kernel.d2_log_g,
            d_h_h0: kernel.w * kernel.ddr0,
            d_h_h1: kernel.w * event_mix(kernel.d, kernel.d3logphi1, -kernel.ddr1),
            d_h_d: -kernel.w * kernel.d * kernel.d3_log_g,
            // 4th derivatives of ℓ w.r.t. the time predictors h0, h1.
            // These are the exact bilinear coefficients for D²H[u,v] in the
            // time-time block. Since u = q + h, d⁴ℓ/dh⁴ = d⁴ℓ/du⁴
            // (same sign because (+1)⁴ = 1), we have:
            d2_h_h0: kernel.w * kernel.dddr0,
            d2_h_h1: kernel.w * event_mix(kernel.d, kernel.d4logphi1, -kernel.dddr1),
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

/// Extended scalar chain-rule derivatives of
/// q(eta_t, eta_ls) = -eta_t * exp(-eta_ls)
/// through **4th order**.
///
/// Returns the same 6 values as `q_chain_derivs_scalar` plus two 4th-order terms:
///
///   u_ϑsss = ∂⁴q / ∂η_ϑ ∂η_s³
///   u_ssss = ∂⁴q / ∂η_s⁴
///
/// # Alternating-sign pattern for exp-link chain derivatives
///
/// With σ = exp(η_s), all derivatives of σ w.r.t. η_s equal σ itself:
/// σ' = σ'' = σ''' = σ'''' = σ. The chain-rule derivatives of
/// q = -η_ϑ/σ then exhibit a clean alternating-sign pattern:
///
/// ```text
///   u_ϑ    = -σ⁻¹       u_s    =  ϑ/σ
///   u_ϑs   =  σ⁻¹       u_ss   = -ϑ/σ
///   u_ϑss  = -σ⁻¹       u_sss  =  ϑ/σ
///   u_ϑsss =  σ⁻¹       u_ssss = -ϑ/σ
/// ```
///
/// Each additional η_s derivative multiplies by d/d(η_s)[σ⁻¹] = -σ⁻¹,
/// producing the sign flip.
///
/// # Why 4th order is needed (see response.md Section 6)
///
/// The outer REML Hessian's Q[v_k, v_l] term requires the 4th derivative
/// of the composed likelihood via the Arbogast formula:
///
/// ```text
///   F_αβγδ = m4·u_α·u_β·u_γ·u_δ
///          + m3·Σ(6 perms) u_αβ·u_γ·u_δ
///          + m2·Σ(3 perms) u_αβ·u_γδ
///          + m2·Σ(4 perms) u_αβγ·u_δ
///          + m1·u_αβγδ          ← requires u_ϑsss and u_ssss
/// ```
///
/// The last term m1·u_αβγδ is nonzero only for (ϑ,s,s,s) and (s,s,s,s).
/// Without these terms the outer Hessian drift is incomplete.
#[inline]
pub(crate) fn q_chain_derivs_fourth_scalar(
    eta_t: f64,
    eta_ls: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let inv_sigma = exp_sigma_inverse_from_eta_scalar(eta_ls);
    let q = -safe_product(eta_t, inv_sigma);
    (-inv_sigma, -q, inv_sigma, q, -inv_sigma, -q, inv_sigma, q)
}

fn validate_cov_block(name: &str, n: usize, b: &CovariateBlockInput) -> Result<(), String> {
    if b.design.nrows() != n {
        return Err(format!(
            "{name} design row mismatch: got {}, expected {n}",
            b.design.nrows()
        ));
    }
    if b.offset.len() != n {
        return Err(format!(
            "{name} offset length mismatch: got {}, expected {n}",
            b.offset.len()
        ));
    }
    let p = b.design.ncols();
    if let Some(beta0) = &b.initial_beta
        && beta0.len() != p
    {
        return Err(format!(
            "{name} initial_beta length mismatch: got {}, expected {p}",
            beta0.len()
        ));
    }
    let k = b.penalties.len();
    if let Some(rho0) = &b.initial_log_lambdas
        && rho0.len() != k
    {
        return Err(format!(
            "{name} initial_log_lambdas length mismatch: got {}, expected {k}",
            rho0.len()
        ));
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
                    return Err(format!(
                        "{name} penalty {idx} block shape mismatch: col_range={}..{}, local={}x{}, total_dim={p}",
                        col_range.start,
                        col_range.end,
                        local.nrows(),
                        local.ncols()
                    ));
                }
            }
            crate::solver::estimate::PenaltySpec::Dense(m) => {
                let (r, c) = m.dim();
                if r != p || c != p {
                    return Err(format!("{name} penalty {idx} must be {p}x{p}, got {r}x{c}"));
                }
            }
        }
    }
    Ok(())
}

fn validate_cov_block_kind(name: &str, n: usize, bk: &CovariateBlockKind) -> Result<(), String> {
    match bk {
        CovariateBlockKind::Static(b) => validate_cov_block(name, n, b),
        CovariateBlockKind::TimeVarying(tv) => {
            if tv.design_covariates.nrows() != n {
                return Err(format!(
                    "{name} time-varying covariate design row mismatch: got {}, expected {n}",
                    tv.design_covariates.nrows()
                ));
            }
            if tv.time_basis_entry.nrows() != n || tv.time_basis_exit.nrows() != n {
                return Err(format!(
                    "{name} time-varying time basis row mismatch: entry={}, exit={}, expected {n}",
                    tv.time_basis_entry.nrows(),
                    tv.time_basis_exit.nrows()
                ));
            }
            if tv.time_basis_derivative_exit.nrows() != n {
                return Err(format!(
                    "{name} time-varying derivative basis row mismatch: got {}, expected {n}",
                    tv.time_basis_derivative_exit.nrows()
                ));
            }
            if tv.offset.len() != n {
                return Err(format!(
                    "{name} time-varying offset length mismatch: got {}, expected {n}",
                    tv.offset.len()
                ));
            }
            let p_cov = tv.design_covariates.ncols();
            let p_time = tv.time_basis_exit.ncols();
            if tv.time_basis_entry.ncols() != p_time {
                return Err(format!(
                    "{name} time-varying time basis column mismatch: entry={}, exit={}",
                    tv.time_basis_entry.ncols(),
                    p_time
                ));
            }
            if tv.time_basis_derivative_exit.ncols() != p_time {
                return Err(format!(
                    "{name} time-varying derivative basis column mismatch: derivative={}, exit={}",
                    tv.time_basis_derivative_exit.ncols(),
                    p_time
                ));
            }
            let p_tensor = p_cov * p_time;
            let k = tv.penalties.len();
            if let Some(beta0) = &tv.initial_beta
                && beta0.len() != p_tensor
            {
                return Err(format!(
                    "{name} time-varying initial_beta length mismatch: got {}, expected {p_tensor}",
                    beta0.len()
                ));
            }
            if let Some(rho0) = &tv.initial_log_lambdas
                && rho0.len() != k
            {
                return Err(format!(
                    "{name} time-varying initial_log_lambdas length mismatch: got {}, expected {k}",
                    rho0.len()
                ));
            }
            for (idx, s) in tv.penalties.iter().enumerate() {
                let (r, c) = s.shape();
                if r != p_tensor || c != p_tensor {
                    return Err(format!(
                        "{name} time-varying penalty {idx} must be {p_tensor}x{p_tensor}, got {r}x{c}"
                    ));
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

fn prepare_cov_block_kind(bk: &CovariateBlockKind) -> Result<PreparedCovBlock, String> {
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
                        crate::solver::estimate::PenaltySpec::Dense(m) => {
                            PenaltyMatrix::Dense(m.clone())
                        }
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
            Ok(CovariateBlockKind::Static(CovariateBlockInput {
                design: cov_design.design.clone(),
                offset: offset.clone(),
                penalties: cov_design
                    .penalties
                    .iter()
                    .map(|bp| crate::solver::estimate::PenaltySpec::from_blockwise_ref(bp))
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

fn build_survival_covariate_block_psi_derivatives(
    data: ndarray::ArrayView2<'_, f64>,
    resolvedspec: &TermCollectionSpec,
    design: &TermCollectionDesign,
    template: &SurvivalCovariateTermBlockTemplate,
) -> Result<Option<Vec<CustomFamilyBlockPsiDerivative>>, String> {
    let spatial_terms = spatial_length_scale_term_indices(resolvedspec);
    let Some(info_list) =
        try_build_spatial_log_kappa_derivativeinfo_list(data, resolvedspec, design, &spatial_terms)
            .map_err(|e| e.to_string())?
    else {
        return Ok(None);
    };
    let psi_dim = info_list.len();
    let axis_lookup: HashMap<(usize, usize), usize> = info_list
        .iter()
        .enumerate()
        .filter_map(|(idx, info)| {
            info.aniso_group_id
                .map(|gid| ((gid, info.implicit_axis), idx))
        })
        .collect();
    Ok(Some(
        info_list
            .into_iter()
            .enumerate()
            .map(
                |(psi_idx, info)| -> Result<CustomFamilyBlockPsiDerivative, String> {
                    let penalty_indices = info.penalty_indices.clone();
                    let embed_design = |local: &Array2<f64>| {
                        EmbeddedColumnBlock::new(local, info.global_range.clone(), info.total_p)
                            .materialize()
                    };
                    let embed_penalty = |local: &Array2<f64>| {
                        EmbeddedSquareBlock::new(local, info.global_range.clone(), info.total_p)
                            .materialize()
                    };
                    match template {
                        SurvivalCovariateTermBlockTemplate::Static => {
                            let implicit_operator = info.implicit_operator.as_ref().map(|op| {
                                wrap_spatial_implicit_psi_operator(
                                    Arc::clone(op),
                                    info.global_range.clone(),
                                    info.total_p,
                                )
                            });
                            let dense_operator =
                                if implicit_operator.is_none() && !info.x_psi_local.is_empty() {
                                    Some(build_embedded_dense_psi_operator(
                                        &info.x_psi_local,
                                        &info.x_psi_psi_local,
                                        info.aniso_cross_designs.as_ref(),
                                        info.global_range.clone(),
                                        info.total_p,
                                        info.implicit_axis,
                                    )?)
                                } else {
                                    None
                                };
                            let design_operator = implicit_operator.or(dense_operator);
                            let materialize_dense_design =
                                !info.x_psi_local.is_empty() && design_operator.is_none();
                            let x_full = if !materialize_dense_design {
                                Array2::<f64>::zeros((0, 0))
                            } else {
                                embed_design(&info.x_psi_local)
                            };
                            let s_components: Vec<(usize, Array2<f64>)> = info
                                .penalty_indices
                                .iter()
                                .copied()
                                .zip(info.s_psi_components_local.iter().map(embed_penalty))
                                .collect();
                            let x_psi_psi = if !materialize_dense_design {
                                None
                            } else {
                                let mut rows =
                                    vec![
                                        Array2::<f64>::zeros((x_full.nrows(), x_full.ncols()));
                                        psi_dim
                                    ];
                                rows[psi_idx] = embed_design(&info.x_psi_psi_local);
                                if let (Some(gid), Some(cross_designs)) =
                                    (info.aniso_group_id, info.aniso_cross_designs.as_ref())
                                {
                                    for (axis_j, local) in cross_designs {
                                        if let Some(&global_j) = axis_lookup.get(&(gid, *axis_j)) {
                                            rows[global_j] = embed_design(local);
                                        }
                                    }
                                }
                                Some(rows)
                            };
                            let mut s_psi_psi_components =
                                vec![Vec::<(usize, Array2<f64>)>::new(); psi_dim];
                            s_psi_psi_components[psi_idx] = penalty_indices
                                .iter()
                                .copied()
                                .zip(info.s_psi_psi_components_local.iter().map(embed_penalty))
                                .collect();
                            if let (Some(gid), Some(cross_penalty_provider)) = (
                                info.aniso_group_id,
                                info.aniso_cross_penalty_provider.as_ref(),
                            ) {
                                for ((group_id, axis_j), global_j) in &axis_lookup {
                                    if *group_id != gid || *axis_j == info.implicit_axis {
                                        continue;
                                    }
                                    let local_components = cross_penalty_provider(*axis_j)
                                        .map_err(|err| err.to_string())?;
                                    if local_components.is_empty() {
                                        continue;
                                    }
                                    s_psi_psi_components[*global_j] = penalty_indices
                                        .iter()
                                        .copied()
                                        .zip(local_components.iter().map(embed_penalty))
                                        .collect();
                                }
                            }
                            Ok(CustomFamilyBlockPsiDerivative {
                                penalty_index: Some(info.penalty_index),
                                x_psi: x_full,
                                s_psi: Array2::<f64>::zeros((0, 0)),
                                s_psi_components: Some(s_components),
                                s_psi_penalty_components: None,
                                x_psi_psi,
                                s_psi_psi: None,
                                s_psi_psi_components: Some(s_psi_psi_components),
                                s_psi_psi_penalty_components: None,
                                implicit_operator: design_operator,
                                implicit_axis: info.implicit_axis,
                                implicit_group_id: info.aniso_group_id,
                            })
                        }
                        SurvivalCovariateTermBlockTemplate::TimeVarying {
                            time_basis_entry,
                            time_basis_exit,
                            time_basis_derivative_exit,
                            ..
                        } => {
                            let tensorize_design = |base: &Array2<f64>| {
                                let base_dm = DesignMatrix::Dense(
                                    crate::matrix::DenseDesignMatrix::from(base.clone()),
                                );
                                let exit_dm = rowwise_kronecker(&base_dm, time_basis_exit);
                                let exit_cow = exit_dm.as_dense_cow();
                                let entry_dm = rowwise_kronecker(&base_dm, time_basis_entry);
                                let entry_cow = entry_dm.as_dense_cow();
                                let deriv_dm =
                                    rowwise_kronecker(&base_dm, time_basis_derivative_exit);
                                let deriv_cow = deriv_dm.as_dense_cow();
                                let n = exit_cow.nrows();
                                let p = exit_cow.ncols();
                                let mut stacked = Array2::<f64>::zeros((3 * n, p));
                                stacked.slice_mut(s![0..n, ..]).assign(&*exit_cow);
                                stacked.slice_mut(s![n..2 * n, ..]).assign(&*entry_cow);
                                stacked.slice_mut(s![2 * n..3 * n, ..]).assign(&*deriv_cow);
                                stacked
                            };
                            let i_time = Array2::<f64>::eye(time_basis_exit.ncols());
                            let tensorize_penalty =
                                |base: &Array2<f64>| kronecker_product(base, &i_time);
                            let base_operator = if let Some(op) = info.implicit_operator.as_ref() {
                                Some(wrap_spatial_implicit_psi_operator(
                                    Arc::clone(op),
                                    info.global_range.clone(),
                                    info.total_p,
                                ))
                            } else if !info.x_psi_local.is_empty() {
                                Some(build_embedded_dense_psi_operator(
                                    &info.x_psi_local,
                                    &info.x_psi_psi_local,
                                    info.aniso_cross_designs.as_ref(),
                                    info.global_range.clone(),
                                    info.total_p,
                                    info.implicit_axis,
                                )?)
                            } else {
                                None
                            };
                            let implicit_operator = base_operator
                                .as_ref()
                                .map(|op| {
                                    build_rowwise_kronecker_psi_operator(
                                        Arc::clone(op),
                                        vec![
                                            shared_dense_arc(time_basis_exit),
                                            shared_dense_arc(time_basis_entry),
                                            shared_dense_arc(time_basis_derivative_exit),
                                        ],
                                    )
                                })
                                .transpose()?;
                            let materialize_dense_design =
                                !info.x_psi_local.is_empty() && implicit_operator.is_none();
                            let x_psi = if !materialize_dense_design {
                                Array2::<f64>::zeros((0, 0))
                            } else {
                                tensorize_design(&embed_design(&info.x_psi_local))
                            };
                            let s_components: Vec<(usize, Array2<f64>)> = info
                                .penalty_indices
                                .iter()
                                .copied()
                                .zip(
                                    info.s_psi_components_local
                                        .iter()
                                        .map(embed_penalty)
                                        .map(|full| tensorize_penalty(&full)),
                                )
                                .collect();
                            let x_psi_psi = if !materialize_dense_design {
                                None
                            } else {
                                let mut rows =
                                    vec![
                                        Array2::<f64>::zeros((x_psi.nrows(), x_psi.ncols()));
                                        psi_dim
                                    ];
                                rows[psi_idx] =
                                    tensorize_design(&embed_design(&info.x_psi_psi_local));
                                if let (Some(gid), Some(cross_designs)) =
                                    (info.aniso_group_id, info.aniso_cross_designs.as_ref())
                                {
                                    for (axis_j, local) in cross_designs {
                                        if let Some(&global_j) = axis_lookup.get(&(gid, *axis_j)) {
                                            rows[global_j] = tensorize_design(&embed_design(local));
                                        }
                                    }
                                }
                                Some(rows)
                            };
                            let mut s_psi_psi_components =
                                vec![Vec::<(usize, Array2<f64>)>::new(); psi_dim];
                            s_psi_psi_components[psi_idx] = penalty_indices
                                .iter()
                                .copied()
                                .zip(
                                    info.s_psi_psi_components_local
                                        .iter()
                                        .map(embed_penalty)
                                        .map(|full| tensorize_penalty(&full)),
                                )
                                .collect();
                            if let (Some(gid), Some(cross_penalty_provider)) = (
                                info.aniso_group_id,
                                info.aniso_cross_penalty_provider.as_ref(),
                            ) {
                                for ((group_id, axis_j), global_j) in &axis_lookup {
                                    if *group_id != gid || *axis_j == info.implicit_axis {
                                        continue;
                                    }
                                    let local_components = cross_penalty_provider(*axis_j)
                                        .map_err(|err| err.to_string())?;
                                    if local_components.is_empty() {
                                        continue;
                                    }
                                    s_psi_psi_components[*global_j] = penalty_indices
                                        .iter()
                                        .copied()
                                        .zip(
                                            local_components
                                                .iter()
                                                .map(embed_penalty)
                                                .map(|full| tensorize_penalty(&full)),
                                        )
                                        .collect();
                                }
                            }
                            Ok(CustomFamilyBlockPsiDerivative {
                                penalty_index: Some(info.penalty_index),
                                x_psi,
                                s_psi: Array2::<f64>::zeros((0, 0)),
                                s_psi_components: Some(s_components),
                                s_psi_penalty_components: None,
                                x_psi_psi,
                                s_psi_psi: None,
                                s_psi_psi_components: Some(s_psi_psi_components),
                                s_psi_psi_penalty_components: None,
                                implicit_operator,
                                implicit_axis: info.implicit_axis,
                                implicit_group_id: info.aniso_group_id,
                            })
                        }
                    }
                },
            )
            .collect::<Result<Vec<_>, _>>()?,
    ))
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
    let threshold_terms = spatial_length_scale_term_indices(thresholdspec);
    let log_sigma_terms = spatial_length_scale_term_indices(log_sigmaspec);
    let rho_lower = Array1::<f64>::from_elem(rho0.len(), -12.0);
    let rho_upper = Array1::<f64>::from_elem(rho0.len(), 12.0);

    let threshold_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        thresholdspec,
        &threshold_terms,
        kappa_options,
    )
    .reseed_from_data(data, thresholdspec, &threshold_terms, kappa_options);
    let log_sigma_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        log_sigmaspec,
        &log_sigma_terms,
        kappa_options,
    )
    .reseed_from_data(data, log_sigmaspec, &log_sigma_terms, kappa_options);
    let mut all_values = threshold_kappa.as_array().to_vec();
    all_values.extend(log_sigma_kappa.as_array().iter());
    let threshold_dims = threshold_kappa.dims_per_term().to_vec();
    let log_sigma_dims = log_sigma_kappa.dims_per_term().to_vec();
    let mut all_dims = threshold_dims.clone();
    all_dims.extend(log_sigma_dims.iter().copied());
    let log_kappa0 =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(all_values), all_dims.clone());
    let threshold_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        thresholdspec,
        &threshold_terms,
        &threshold_dims,
        kappa_options,
    );
    let log_sigma_lower = SpatialLogKappaCoords::lower_bounds_aniso_from_data(
        data,
        log_sigmaspec,
        &log_sigma_terms,
        &log_sigma_dims,
        kappa_options,
    );
    let mut lower_vals = threshold_lower.as_array().to_vec();
    lower_vals.extend(log_sigma_lower.as_array().iter());
    let log_kappa_lower =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(lower_vals), all_dims.clone());
    let threshold_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        thresholdspec,
        &threshold_terms,
        &threshold_dims,
        kappa_options,
    );
    let log_sigma_upper = SpatialLogKappaCoords::upper_bounds_aniso_from_data(
        data,
        log_sigmaspec,
        &log_sigma_terms,
        &log_sigma_dims,
        kappa_options,
    );
    let mut upper_vals = threshold_upper.as_array().to_vec();
    upper_vals.extend(log_sigma_upper.as_array().iter());
    let log_kappa_upper =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(upper_vals), all_dims);
    // Project seed onto bounds; spec.length_scale is a hint, not a constraint.
    let log_kappa0 = log_kappa0.clamp_to_bounds(&log_kappa_lower, &log_kappa_upper);

    ExactJointHyperSetup::new(
        rho0,
        rho_lower,
        rho_upper,
        log_kappa0,
        log_kappa_lower,
        log_kappa_upper,
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
        let desired = 1.0 / age_exit[i].max(1e-9);
        target[i] = (desired - derivative_offset_exit[i]).max(0.0);
    }

    let xtx = design_derivative_exit.t().dot(design_derivative_exit);
    let xty = design_derivative_exit.t().dot(&target);
    let eps = 1e-6 * (0..p).map(|i| xtx[[i, i]]).fold(0.0_f64, f64::max).max(1.0);
    let mut lhs = xtx;
    for i in 0..p {
        lhs[[i, i]] += eps;
    }

    use crate::faer_ndarray::FaerCholesky;
    let chol = lhs.cholesky(faer::Side::Lower).ok()?;
    let mut beta_init = chol.solvevec(&xty);
    if let Some(lower_bounds) = coefficient_lower_bounds {
        if let Some(constraints) = lower_bound_constraints(lower_bounds) {
            beta_init = project_onto_linear_constraints(p, &constraints, Some(&beta_init));
        }
    }

    let d_raw_init = design_derivative_exit.dot(&beta_init) + derivative_offset_exit;
    if d_raw_init
        .iter()
        .all(|v| v.is_finite() && *v >= derivative_guard)
    {
        Some(beta_init)
    } else {
        None
    }
}

fn survival_blockwise_fit_options(spec: &SurvivalLocationScaleSpec) -> BlockwiseFitOptions {
    BlockwiseFitOptions {
        inner_max_cycles: spec.max_iter,
        inner_tol: spec.tol,
        outer_max_iter: 60,
        outer_tol: 1e-5,
        compute_covariance: true,
        ..BlockwiseFitOptions::default()
    }
}

fn validate_survival_location_scale_spec(spec: &SurvivalLocationScaleSpec) -> Result<(), String> {
    let n = spec.event_target.len();
    let monotone_time_wiggle_ncols = spec.timewiggle_block.as_ref().map_or(0, |w| w.ncols);
    if n == 0 {
        return Err("fit_survival_location_scale: empty dataset".to_string());
    }
    if spec.age_entry.len() != n || spec.age_exit.len() != n || spec.weights.len() != n {
        return Err("fit_survival_location_scale: top-level input size mismatch".to_string());
    }
    if !(spec.tol.is_finite() && spec.tol > 0.0) {
        return Err(format!(
            "fit_survival_location_scale: invalid tol {}",
            spec.tol
        ));
    }
    if spec.max_iter == 0 {
        return Err("fit_survival_location_scale: max_iter must be > 0".to_string());
    }
    if !spec.derivative_guard.is_finite() || spec.derivative_guard <= 0.0 {
        return Err(format!(
            "fit_survival_location_scale: derivative_guard must be > 0, got {}",
            spec.derivative_guard
        ));
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
            return Err("timewiggle_block must have at least one coefficient".to_string());
        }
        if w.ncols >= spec.time_block.design_exit.ncols() {
            return Err(format!(
                "timewiggle_block.ncols must be smaller than time_block columns: wiggle={}, total={}",
                w.ncols,
                spec.time_block.design_exit.ncols()
            ));
        }
        if w.knots.len() < 2 * (w.degree + 1) {
            return Err(format!(
                "timewiggle_block knot vector is too short for degree {}: got {} knots",
                w.degree,
                w.knots.len()
            ));
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
            return Err(format!(
                "fit_survival_location_scale: invalid interval at row {} (entry={}, exit={})",
                i + 1,
                spec.age_entry[i],
                spec.age_exit[i]
            ));
        }
        if !spec.weights[i].is_finite() || spec.weights[i] < 0.0 {
            return Err(format!(
                "fit_survival_location_scale: invalid weight at row {} ({})",
                i + 1,
                spec.weights[i]
            ));
        }
        if !spec.event_target[i].is_finite() || !(0.0..=1.0).contains(&spec.event_target[i]) {
            return Err(format!(
                "fit_survival_location_scale: event_target must be in [0,1], found {} at row {}",
                spec.event_target[i],
                i + 1
            ));
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
    let mut time_prepared = prepare_identified_time_block(
        &spec.time_block,
        spec.derivative_guard,
        protected_timewiggle_cols,
    )?;

    if time_prepared.initial_beta.is_none() {
        time_prepared.initial_beta = structural_time_initial_beta_guess(
            &time_prepared.design_derivative_exit,
            &spec.time_block.derivative_offset_exit,
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
    let time_stacked_offset = stack_offsets(&[
        &spec.time_block.offset_entry,
        &spec.time_block.offset_exit,
        &spec.time_block.derivative_offset_exit,
    ]);
    let timespec = ParameterBlockSpec {
        name: "time_transform".to_string(),
        design: time_solver_design,
        offset: time_stacked_offset,
        penalties: time_prepared
            .penalties
            .iter()
            .cloned()
            .map(PenaltyMatrix::Dense)
            .collect(),
        nullspace_dims: spec.time_block.nullspace_dims.clone(),
        initial_log_lambdas: initial_log_lambdas(
            &time_prepared.penalties,
            spec.time_block.initial_log_lambdas.clone(),
        )?,
        initial_beta: time_prepared.initial_beta.clone(),
    };

    let threshold_prep = prepare_cov_block_kind(&spec.threshold_block)?;
    let (threshold_solver_design, threshold_solver_offset) =
        if let Some(x_entry) = threshold_prep.design_entry.as_ref() {
            let x_deriv = threshold_prep
                .design_derivative_exit
                .as_ref()
                .ok_or_else(|| {
                    "time-varying threshold block is missing its exit derivative design".to_string()
                })?;
            (
                DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(
                    MultiChannelOperator::new(vec![
                        threshold_prep.design_exit.clone(),
                        x_entry.clone(),
                        x_deriv.clone(),
                    ])?,
                ))),
                stack_offsets(&[
                    &threshold_prep.offset,
                    &threshold_prep.offset,
                    &Array1::zeros(n),
                ]),
            )
        } else {
            (
                threshold_prep.design_exit.clone(),
                threshold_prep.offset.clone(),
            )
        };
    let thresholdspec = ParameterBlockSpec {
        name: "threshold".to_string(),
        design: threshold_solver_design,
        offset: threshold_solver_offset,
        penalties: threshold_prep.penalties.clone(),
        nullspace_dims: threshold_prep.nullspace_dims.clone(),
        initial_log_lambdas: initial_log_lambdas(
            &threshold_prep.penalties,
            threshold_prep.initial_log_lambdas.clone(),
        )?,
        initial_beta: threshold_prep.initial_beta.clone(),
    };

    let survival_primary_design = DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(
        BlockDesignOperator::new(vec![
            DesignBlock::Dense(DenseDesignMatrix::from(shared_dense_arc(
                &time_prepared.design_exit,
            ))),
            design_block_from_matrix(threshold_prep.design_exit.clone()),
        ])?,
    )));

    let log_sigma_prep = prepare_cov_block_kind(&spec.log_sigma_block)?;
    let non_intercept_start =
        infer_non_intercept_start_design(&log_sigma_prep.design_exit, &spec.weights)?;
    let scale_transform = build_scale_deviation_transform_design(
        &survival_primary_design,
        &log_sigma_prep.design_exit,
        &spec.weights,
        non_intercept_start,
    )?;
    let log_sigma_design = build_scale_deviation_operator(
        survival_primary_design.clone(),
        log_sigma_prep.design_exit.clone(),
        &scale_transform,
    )?;
    let log_sigma_entry_design = if let Some(x_ls_entry) = log_sigma_prep.design_entry.as_ref() {
        Some(build_scale_deviation_operator(
            survival_primary_design.clone(),
            x_ls_entry.clone(),
            &scale_transform,
        )?)
    } else {
        None
    };
    let (log_sigma_solver_design, log_sigma_solver_offset) =
        if let Some(ref ls_entry) = log_sigma_entry_design {
            let ls_deriv = log_sigma_prep
                .design_derivative_exit
                .as_ref()
                .ok_or_else(|| {
                    "time-varying log-sigma block is missing its exit derivative design".to_string()
                })?;
            (
                DesignMatrix::Dense(DenseDesignMatrix::from(Arc::new(
                    MultiChannelOperator::new(vec![
                        log_sigma_design.clone(),
                        ls_entry.clone(),
                        ls_deriv.clone(),
                    ])?,
                ))),
                stack_offsets(&[
                    &log_sigma_prep.offset,
                    &log_sigma_prep.offset,
                    &Array1::zeros(n),
                ]),
            )
        } else {
            (log_sigma_design.clone(), log_sigma_prep.offset.clone())
        };
    let log_sigmaspec = ParameterBlockSpec {
        name: "log_sigma".to_string(),
        design: log_sigma_solver_design,
        offset: log_sigma_solver_offset,
        penalties: log_sigma_prep.penalties.clone(),
        nullspace_dims: log_sigma_prep.nullspace_dims.clone(),
        initial_log_lambdas: initial_log_lambdas(
            &log_sigma_prep.penalties,
            log_sigma_prep.initial_log_lambdas.clone(),
        )?,
        initial_beta: log_sigma_prep.initial_beta.clone(),
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
                        crate::solver::estimate::PenaltySpec::Dense(m) => {
                            PenaltyMatrix::Dense(m.clone())
                        }
                    })
                    .collect()
            },
            nullspace_dims: w.nullspace_dims.clone(),
            initial_log_lambdas: initial_log_lambdas(&w.penalties, w.initial_log_lambdas.clone())?,
            initial_beta: w.initial_beta.clone(),
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
        x_time_entry: Arc::new(time_prepared.design_entry.clone()),
        x_time_exit: Arc::new(time_prepared.design_exit.clone()),
        x_time_deriv: Arc::new(time_prepared.design_derivative_exit.clone()),
        time_derivative_offset_exit: Arc::new(spec.time_block.derivative_offset_exit.clone()),
        time_wiggle_knots: spec.timewiggle_block.as_ref().map(|w| w.knots.clone()),
        time_wiggle_degree: spec.timewiggle_block.as_ref().map(|w| w.degree),
        time_wiggle_ncols: protected_timewiggle_cols,
        time_coefficient_lower_bounds: time_prepared.coefficient_lower_bounds.clone(),
        x_threshold: threshold_prep.design_exit.clone(),
        x_threshold_entry: threshold_prep.design_entry.clone(),
        x_threshold_deriv: threshold_prep.design_derivative_exit.clone(),
        x_log_sigma: log_sigma_design,
        x_log_sigma_entry: log_sigma_entry_design,
        x_log_sigma_deriv: log_sigma_prep.design_derivative_exit.clone(),
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
        k_time: spec.time_block.penalties.len(),
        k_threshold: threshold_prep.penalties.len(),
        k_log_sigma: log_sigma_prep.penalties.len(),
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
    let beta_time = prepared.time_transform.z.dot(&beta_time_reduced);
    let beta_threshold = fit.block_states[SurvivalLocationScaleFamily::BLOCK_THRESHOLD]
        .beta
        .clone();
    let beta_log_sigma = fit.block_states[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA]
        .beta
        .clone();
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
    let covariance_conditional = fit.covariance_conditional.as_ref().map(|cov_reduced| {
        lift_conditional_covariance(
            cov_reduced,
            &prepared.time_transform.z,
            beta_threshold.len(),
            beta_log_sigma.len(),
            beta_link_wiggle.as_ref().map_or(0, |b| b.len()),
        )
    });
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
        outer_iterations: fit.inner_cycles,
        outer_gradient_norm: fit.outer_gradient_norm,
        outer_converged: fit.outer_converged,
        covariance_conditional,
        geometry: fit.geometry.clone(),
    })
}

fn validatewiggle_block(n: usize, b: &LinkWiggleBlockInput) -> Result<(), String> {
    if b.design.nrows() != n {
        return Err(format!(
            "linkwiggle_block design row mismatch: got {}, expected {n}",
            b.design.nrows()
        ));
    }
    let p = b.design.ncols();
    if b.knots.len() < b.degree + 2 {
        return Err(format!(
            "linkwiggle_block knot vector is too short for degree {}: got {} knots",
            b.degree,
            b.knots.len()
        ));
    }
    if let Some(beta0) = &b.initial_beta
        && beta0.len() != p
    {
        return Err(format!(
            "linkwiggle_block initial_beta length mismatch: got {}, expected {p}",
            beta0.len()
        ));
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
        return Err(format!(
            "linkwiggle_block initial_log_lambdas length mismatch: got {}, expected {k}",
            rho0.len()
        ));
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
                    return Err(format!(
                        "linkwiggle_block penalty {idx} block shape mismatch: col_range={}..{}, local={}x{}, total_dim={p}",
                        col_range.start,
                        col_range.end,
                        local.nrows(),
                        local.ncols()
                    ));
                }
            }
            crate::solver::estimate::PenaltySpec::Dense(m) => {
                let (r, c) = m.dim();
                if r != p || c != p {
                    return Err(format!(
                        "linkwiggle_block penalty {idx} must be {p}x{p}, got {r}x{c}"
                    ));
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
) -> Result<(), String> {
    if b.design_entry.nrows() != n
        || b.design_exit.nrows() != n
        || b.design_derivative_exit.nrows() != n
        || b.offset_entry.len() != n
        || b.offset_exit.len() != n
        || b.derivative_offset_exit.len() != n
    {
        return Err("time_block input size mismatch".to_string());
    }
    let p = b.design_exit.ncols();
    if b.design_entry.ncols() != p || b.design_derivative_exit.ncols() != p {
        return Err("time_block design column mismatch across entry/exit/derivative".to_string());
    }
    if !b.structural_monotonicity {
        return Err(
            "time_block requires structural monotonicity by construction; non-structural time transforms are no longer supported"
                .to_string(),
        );
    }
    structural_time_coefficient_lower_bounds_with_monotone_time_wiggle(
        &b.design_derivative_exit.to_dense(),
        &b.derivative_offset_exit,
        derivative_guard,
        monotone_time_wiggle_ncols,
    )?;
    if let Some(beta0) = &b.initial_beta
        && beta0.len() != p
    {
        return Err(format!(
            "time_block initial_beta length mismatch: got {}, expected {p}",
            beta0.len()
        ));
    }
    let k = b.penalties.len();
    if let Some(rho0) = &b.initial_log_lambdas
        && rho0.len() != k
    {
        return Err(format!(
            "time_block initial_log_lambdas length mismatch: got {}, expected {k}",
            rho0.len()
        ));
    }
    for (idx, s) in b.penalties.iter().enumerate() {
        let (r, c) = s.dim();
        if r != p || c != p {
            return Err(format!(
                "time_block penalty {idx} must be {p}x{p}, got {r}x{c}"
            ));
        }
    }
    Ok(())
}

fn stack_offsets(parts: &[&Array1<f64>]) -> Array1<f64> {
    let total: usize = parts.iter().map(|part| part.len()).sum();
    let mut out = Array1::<f64>::zeros(total);
    let mut offset = 0usize;
    for part in parts {
        let next = offset + part.len();
        out.slice_mut(s![offset..next]).assign(part);
        offset = next;
    }
    out
}

#[derive(Clone, Debug)]
struct TimeIdentifiabilityTransform {
    z: Array2<f64>,
}

#[derive(Clone, Debug)]
struct TimeBlockPrepared {
    design_entry: Array2<f64>,
    design_exit: Array2<f64>,
    design_derivative_exit: Array2<f64>,
    coefficient_lower_bounds: Option<Array1<f64>>,
    penalties: Vec<Array2<f64>>,
    initial_beta: Option<Array1<f64>>,
    transform: TimeIdentifiabilityTransform,
}

fn lower_bound_constraints(lower_bounds: &Array1<f64>) -> Option<LinearInequalityConstraints> {
    let active_rows: Vec<usize> = (0..lower_bounds.len())
        .filter(|&i| lower_bounds[i].is_finite())
        .collect();
    if active_rows.is_empty() {
        return None;
    }
    let p = lower_bounds.len();
    let mut a = Array2::<f64>::zeros((active_rows.len(), p));
    let mut b = Array1::<f64>::zeros(active_rows.len());
    for (row, &idx) in active_rows.iter().enumerate() {
        a[[row, idx]] = 1.0;
        b[row] = lower_bounds[idx];
    }
    Some(LinearInequalityConstraints { a, b })
}

fn structural_time_coefficient_lower_bounds(
    design_derivative_exit: &Array2<f64>,
    derivative_offset_exit: &Array1<f64>,
    lower_bound: f64,
) -> Result<Option<Array1<f64>>, String> {
    if design_derivative_exit.nrows() != derivative_offset_exit.len() {
        return Err(format!(
            "structural time coefficient bounds require matching rows/offsets: rows={}, offsets={}",
            design_derivative_exit.nrows(),
            derivative_offset_exit.len()
        ));
    }
    if design_derivative_exit.ncols() == 0 {
        return Ok(None);
    }
    if !lower_bound.is_finite() || lower_bound <= 0.0 {
        return Err(format!(
            "structural time coefficient lower bound must be finite and > 0, got {lower_bound}"
        ));
    }

    const DERIVATIVE_TOL: f64 = 1e-12;
    const FEASIBILITY_TOL: f64 = 1e-12;

    let p = design_derivative_exit.ncols();
    let mut lower_bounds = Array1::from_elem(p, f64::NEG_INFINITY);
    let mut has_structural_support = false;
    for (row, &offset) in derivative_offset_exit.iter().enumerate() {
        if !offset.is_finite() {
            return Err(format!(
                "structural time coefficient bounds require finite derivative offsets; found offset[{row}]={offset}"
            ));
        }
        if lower_bound - offset > FEASIBILITY_TOL {
            return Err(format!(
                "structural time coefficient bounds require derivative offsets to encode the derivative guard at row {row}: offset={offset:.3e} < guard={lower_bound:.3e}"
            ));
        }
    }
    for col in 0..p {
        let mut has_positive_support = false;
        for row in 0..design_derivative_exit.nrows() {
            let value = design_derivative_exit[[row, col]];
            if !value.is_finite() {
                return Err(format!(
                    "structural time coefficient bounds require finite derivative design entries; found row {row}, column {col}"
                ));
            }
            if value < -DERIVATIVE_TOL {
                return Err(format!(
                    "structural time coefficient bounds require a non-negative derivative basis at row {row}, column {col}; found {value:.3e}"
                ));
            }
            if value > DERIVATIVE_TOL {
                has_positive_support = true;
            }
        }
        if has_positive_support {
            lower_bounds[col] = 0.0;
            has_structural_support = true;
        }
    }

    if !has_structural_support {
        return Err(
            "structural time coefficient bounds require at least one derivative-active column"
                .to_string(),
        );
    }
    Ok(Some(lower_bounds))
}

fn structural_time_coefficient_lower_bounds_with_monotone_time_wiggle(
    design_derivative_exit: &Array2<f64>,
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
        return Err(format!(
            "structural time coefficient bounds cannot reserve {monotone_time_wiggle_ncols} monotone wiggle columns from {} coefficients",
            bounds.len()
        ));
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

pub(crate) fn project_onto_linear_constraints(
    dim: usize,
    constraints: &LinearInequalityConstraints,
    beta0: Option<&Array1<f64>>,
) -> Array1<f64> {
    let mut beta = beta0.cloned().unwrap_or_else(|| Array1::zeros(dim));
    if constraints.a.ncols() != dim || constraints.a.nrows() == 0 {
        return beta;
    }
    let mut corrections = Array2::<f64>::zeros((constraints.a.nrows(), dim));
    for _ in 0..100 {
        let mut max_violation = 0.0_f64;
        for i in 0..constraints.a.nrows() {
            let row = constraints.a.row(i);
            let row_norm_sq = row.dot(&row);
            if row_norm_sq <= 1e-18 {
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
        if max_violation <= 1e-10 {
            break;
        }
    }
    beta
}

fn prepare_identified_time_block(
    input: &TimeBlockInput,
    derivative_guard: f64,
    monotone_time_wiggle_ncols: usize,
) -> Result<TimeBlockPrepared, String> {
    let p = input.design_exit.ncols();
    if !input.structural_monotonicity {
        return Err(
            "time_block requires structural monotonicity by construction; non-structural time transforms are no longer supported"
                .to_string(),
        );
    }
    // Materialize to dense at the location-scale boundary — the hot path
    // uses dense matrix operations (scale_dense_rows, weighted_crossprod_dense).
    let design_entry = input.design_entry.to_dense();
    let design_exit = input.design_exit.to_dense();
    let design_derivative_exit = input.design_derivative_exit.to_dense();
    let penalties = input.penalties.clone();
    let coefficient_lower_bounds = structural_time_coefficient_lower_bounds_with_monotone_time_wiggle(
        &design_derivative_exit,
        &input.derivative_offset_exit,
        derivative_guard,
        monotone_time_wiggle_ncols,
    )?
    .ok_or_else(|| {
        "structural time block requires derivative offsets to encode the derivative guard and a non-negative derivative basis"
            .to_string()
    })?;
    let linear_constraints = lower_bound_constraints(&coefficient_lower_bounds);
    let initial_beta = match (linear_constraints.as_ref(), input.initial_beta.as_ref()) {
        (Some(constraints), Some(beta0)) => {
            Some(project_onto_linear_constraints(p, constraints, Some(beta0)))
        }
        (_, Some(beta0)) => Some(beta0.clone()),
        _ => None,
    };

    Ok(TimeBlockPrepared {
        design_entry,
        design_exit,
        design_derivative_exit,
        coefficient_lower_bounds: Some(coefficient_lower_bounds),
        penalties,
        initial_beta,
        transform: TimeIdentifiabilityTransform { z: Array2::eye(p) },
    })
}

fn initial_log_lambdas<T>(
    penalties: &[T],
    rho0: Option<Array1<f64>>,
) -> Result<Array1<f64>, String> {
    let k = penalties.len();
    let rho = rho0.unwrap_or_else(|| Array1::zeros(k));
    if rho.len() != k {
        return Err(format!(
            "initial_log_lambdas mismatch: got {}, expected {k}",
            rho.len()
        ));
    }
    Ok(rho)
}

fn weighted_crossprod_dense_stable(
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    let mut out = Array2::<f64>::zeros((left.ncols(), right.ncols()));
    for i in 0..weights.len() {
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
    if out.iter().any(|value| !value.is_finite()) {
        return Err(
            "weighted_crossprod_dense stable accumulation produced non-finite values".to_string(),
        );
    }
    Ok(out)
}

fn weighted_crossprod_dense(
    left: &Array2<f64>,
    weights: &Array1<f64>,
    right: &Array2<f64>,
) -> Result<Array2<f64>, String> {
    if left.nrows() != weights.len() || right.nrows() != weights.len() {
        return Err(format!(
            "weighted_crossprod_dense row mismatch: left is {}x{}, weights has {}, right is {}x{}",
            left.nrows(),
            left.ncols(),
            weights.len(),
            right.nrows(),
            right.ncols()
        ));
    }
    if left.iter().any(|value| !value.is_finite()) || right.iter().any(|value| !value.is_finite()) {
        return Err("weighted_crossprod_dense inputs contain non-finite design values".to_string());
    }

    let sanitized_weights = sanitize_survival_weight_vector(weights);
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
        let out = left.t().dot(&weighted_right);
        if out.iter().all(|value| value.is_finite()) {
            return Ok(out);
        }
    }

    weighted_crossprod_dense_stable(left, &sanitized_weights, right)
}

fn scale_dense_rows(mat: &Array2<f64>, coeffs: &Array1<f64>) -> Result<Array2<f64>, String> {
    if mat.nrows() != coeffs.len() {
        return Err(format!(
            "row scaling dimension mismatch: matrix has {} rows but coeffs have {} entries",
            mat.nrows(),
            coeffs.len()
        ));
    }
    let sanitized_coeffs = sanitize_survival_weight_vector(coeffs);
    let out = Array2::from_shape_fn(mat.dim(), |(i, j)| {
        safe_product(mat[[i, j]], sanitized_coeffs[i])
    });
    if out.iter().any(|value| value.is_nan()) {
        return Err("row scaling produced NaN values".to_string());
    }
    Ok(out)
}

fn embed_tail_columns(
    local: &Array2<f64>,
    total_cols: usize,
    tail_range: std::ops::Range<usize>,
) -> Result<Array2<f64>, String> {
    if tail_range.end > total_cols || tail_range.len() != local.ncols() {
        return Err(format!(
            "tail embedding mismatch: local_cols={}, total_cols={}, tail={:?}",
            local.ncols(),
            total_cols,
            tail_range
        ));
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

fn validate_predict_inverse_link(inverse_link: &InverseLink) -> Result<(), String> {
    match inverse_link {
        InverseLink::Standard(LinkFunction::Log) => Err(
            "prediction does not support Standard(Log) for survival models".to_string(),
        ),
        InverseLink::Standard(LinkFunction::Sas) => Err(
            "prediction requires explicit SasLinkState; state-less Standard(Sas) is unsupported"
                .to_string(),
        ),
        InverseLink::Standard(LinkFunction::BetaLogistic) => Err(
            "prediction requires explicit Beta-Logistic link state; state-less Standard(BetaLogistic) is unsupported"
                .to_string(),
        ),
        _ => Ok(()),
    }
}

fn inverse_link_failure_prob_checked(inverse_link: &InverseLink, eta: f64) -> Result<f64, String> {
    inverse_link_jet_for_inverse_link(inverse_link, eta)
        .map(|j| j.mu.clamp(0.0, 1.0))
        .map_err(|e| format!("inverse link prediction failed at eta={eta}: {e}"))
}

fn inverse_link_survival_prob_checked(inverse_link: &InverseLink, eta: f64) -> Result<f64, String> {
    inverse_link_failure_prob_checked(inverse_link, eta).map(|f| (1.0 - f).clamp(0.0, 1.0))
}

fn inverse_link_survival_probvalue(inverse_link: &InverseLink, eta: f64) -> f64 {
    match inverse_link {
        InverseLink::Standard(LinkFunction::Probit) => probit_survival_value(eta),
        InverseLink::Standard(LinkFunction::Logit) => 1.0 / (1.0 + eta.exp()),
        InverseLink::Standard(LinkFunction::CLogLog) => (-(eta.exp())).exp(),
        InverseLink::Standard(LinkFunction::Identity) => 1.0 - eta,
        InverseLink::Standard(LinkFunction::Log) => {
            panic!("state-less log inverse link is invalid for survival prediction")
        }
        InverseLink::LatentCLogLog(_)
        | InverseLink::Sas(_)
        | InverseLink::BetaLogistic(_)
        | InverseLink::Mixture(_) => inverse_link_survival_prob_checked(inverse_link, eta)
            .expect("validated inverse link should evaluate during prediction"),
        InverseLink::Standard(LinkFunction::Sas)
        | InverseLink::Standard(LinkFunction::BetaLogistic) => {
            panic!("state-less SAS/Beta-Logistic inverse link is invalid for prediction")
        }
    }
}

fn linear_predictor_se(x: ndarray::ArrayView2<'_, f64>, cov: &Array2<f64>) -> Array1<f64> {
    let xc = x.dot(cov);
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
    d4q_dq04: Array1<f64>,
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
    q_tl_ls_ls: f64,
    q_llll: f64,
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
    q_tl_ls_ls: f64,
    q_llll: f64,
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
    d4q_tls_ls_ls_exit: Array1<f64>,
    d4q_tls_ls_ls_entry: Array1<f64>,
    d4q_ls_exit: Array1<f64>,
    d4q_ls_entry: Array1<f64>,
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
            return Err(format!(
                "survival dynamic geometry derivative length mismatch: base_derivative={}, rows={n}",
                self.time_base_derivative_exit.len()
            ));
        }
        if let Some(basis) = self.time_wiggle_basis_d1_entry.as_ref() {
            if basis.nrows() != n {
                return Err(format!(
                    "survival dynamic geometry wiggle d1 entry row mismatch: rows={}, expected {n}",
                    basis.nrows()
                ));
            }
        }
        if let Some(basis) = self.time_wiggle_basis_d1_exit.as_ref() {
            if basis.nrows() != n {
                return Err(format!(
                    "survival dynamic geometry wiggle d1 exit row mismatch: rows={}, expected {n}",
                    basis.nrows()
                ));
            }
        }
        if let Some(basis) = self.time_wiggle_basis_d2_exit.as_ref() {
            if basis.nrows() != n {
                return Err(format!(
                    "survival dynamic geometry wiggle d2 exit row mismatch: rows={}, expected {n}",
                    basis.nrows()
                ));
            }
        }
        if let Some(values) = self.time_wiggle_d2_entry.as_ref() {
            if values.len() != n {
                return Err(format!(
                    "survival dynamic geometry wiggle d2 entry length mismatch: len={}, expected {n}",
                    values.len()
                ));
            }
        }
        if let Some(values) = self.time_wiggle_d2_exit.as_ref() {
            if values.len() != n {
                return Err(format!(
                    "survival dynamic geometry wiggle d2 exit length mismatch: len={}, expected {n}",
                    values.len()
                ));
            }
        }
        if let Some(values) = self.time_wiggle_d3_exit.as_ref() {
            if values.len() != n {
                return Err(format!(
                    "survival dynamic geometry wiggle d3 exit length mismatch: len={}, expected {n}",
                    values.len()
                ));
            }
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

fn survival_wiggle_fourth_q(
    q0: ndarray::ArrayView1<'_, f64>,
    knots: &Array1<f64>,
    degree: usize,
    beta_w: ndarray::ArrayView1<'_, f64>,
) -> Result<Array1<f64>, String> {
    let basis_d4 = monotone_wiggle_basis_with_derivative_order(q0, knots, degree, 4)?;
    if basis_d4.ncols() != beta_w.len() {
        return Err(format!(
            "survival linkwiggle fourth-derivative dimension mismatch: basis has {} columns but beta has {} entries",
            basis_d4.ncols(),
            beta_w.len()
        ));
    }
    Ok(basis_d4.dot(&beta_w))
}

fn survival_base_q_scalars(eta_t: f64, eta_ls: f64) -> SurvivalBaseQScalars {
    let (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls, q_tl_ls_ls, q_llll) =
        q_chain_derivs_fourth_scalar(eta_t, eta_ls);
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
        q_tl_ls_ls,
        q_llll,
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
    d4q_dq04: f64,
) -> SurvivalDynamicQScalars {
    let a = base.q_t;
    let b = base.q_ls;
    let c = base.q_tl;
    let d = base.q_ll;
    let e = base.q_tl_ls;
    let f = base.q_ll_ls;
    let g = base.q_tl_ls_ls;
    let h = base.q_llll;
    let m1 = dq_dq0;
    let m2 = d2q_dq02;
    let m3 = d3q_dq03;
    let m4 = d4q_dq04;
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
    let q_tl_ls_ls = safe_sum3(
        safe_product(m4, safe_product(a, safe_product(b, safe_product(b, b)))),
        safe_product(
            m3,
            safe_sum2(
                3.0 * safe_product(safe_product(b, b), c),
                3.0 * safe_product(safe_product(a, b), d),
            ),
        ),
        safe_sum2(
            safe_product(
                m2,
                safe_sum3(
                    safe_product(a, f),
                    3.0 * safe_product(c, d),
                    3.0 * safe_product(b, e),
                ),
            ),
            safe_product(m1, g),
        ),
    );
    let q_llll = safe_sum3(
        safe_product(m4, safe_product(safe_product(b, b), safe_product(b, b))),
        safe_product(m3, 6.0 * safe_product(safe_product(b, b), d)),
        safe_sum2(
            safe_product(
                m2,
                safe_sum2(3.0 * safe_product(d, d), 4.0 * safe_product(b, f)),
            ),
            safe_product(m1, h),
        ),
    );

    SurvivalDynamicQScalars {
        q: base.q + wiggle_value,
        q_t,
        q_ls,
        q_tl,
        q_ll,
        q_tl_ls,
        q_ll_ls,
        q_tl_ls_ls,
        q_llll,
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
        let eta_t_exit = joint_states.3;
        let eta_ls_exit = joint_states.4;
        let eta_t_entry = joint_states.5;
        let eta_ls_entry = joint_states.6;
        let eta_t_deriv_exit = joint_states.7;
        let eta_ls_deriv_exit = joint_states.8;
        let eta_t_deriv_exit = eta_t_deriv_exit
            .map(|v| v.to_owned())
            .unwrap_or_else(|| Array1::zeros(n));
        let eta_ls_deriv_exit = eta_ls_deriv_exit
            .map(|v| v.to_owned())
            .unwrap_or_else(|| Array1::zeros(n));
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
            return Err(
                "survival location-scale linkwiggle requires dynamic knot/degree metadata"
                    .to_string(),
            );
        }
        if self.time_wiggle_ncols > 0 && (time_wiggle_exit.is_none() || time_wiggle_entry.is_none())
        {
            return Err(
                "survival location-scale timewiggle requires dynamic knot/degree metadata"
                    .to_string(),
            );
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
            h_entry = &h_entry_base + &wig_entry.basis.dot(&beta_w);
            h_exit = &h_exit_base + &wig_exit.basis.dot(&beta_w);
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
        let mut d4q_tls_ls_ls_exit = Array1::<f64>::zeros(n);
        let mut d4q_tls_ls_ls_entry = Array1::<f64>::zeros(n);
        let mut d4q_ls_exit = Array1::<f64>::zeros(n);
        let mut d4q_ls_entry = Array1::<f64>::zeros(n);
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

        for i in 0..n {
            let base_exit = survival_base_q_scalars(eta_t_exit[i], eta_ls_exit[i]);
            let base_entry = survival_base_q_scalars(eta_t_entry[i], eta_ls_entry[i]);
            let exit_dyn = if let Some(wig) = wiggle_exit.as_ref() {
                compose_survival_dynamic_q(
                    base_exit,
                    eta_t_deriv_exit[i],
                    eta_ls_deriv_exit[i],
                    wig.basis
                        .row(i)
                        .dot(&block_states[Self::BLOCK_LINK_WIGGLE].beta),
                    wig.dq_dq0[i],
                    wig.d2q_dq02[i],
                    wig.d3q_dq03[i],
                    wig.d4q_dq04[i],
                )
            } else {
                compose_survival_dynamic_q(
                    base_exit,
                    eta_t_deriv_exit[i],
                    eta_ls_deriv_exit[i],
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                )
            };
            let entry_dyn = if let Some(wig) = wiggle_entry.as_ref() {
                compose_survival_dynamic_q(
                    base_entry,
                    0.0,
                    0.0,
                    wig.basis
                        .row(i)
                        .dot(&block_states[Self::BLOCK_LINK_WIGGLE].beta),
                    wig.dq_dq0[i],
                    wig.d2q_dq02[i],
                    wig.d3q_dq03[i],
                    wig.d4q_dq04[i],
                )
            } else {
                compose_survival_dynamic_q(base_entry, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
            };
            q_exit[i] = exit_dyn.q;
            q_entry[i] = entry_dyn.q;
            qdot_exit[i] = exit_dyn.qdot;
            dq_t_exit[i] = exit_dyn.q_t;
            dq_t_entry[i] = entry_dyn.q_t;
            dq_ls_exit[i] = exit_dyn.q_ls;
            dq_ls_entry[i] = entry_dyn.q_ls;
            d2q_tls_exit[i] = exit_dyn.q_tl;
            d2q_tls_entry[i] = entry_dyn.q_tl;
            d2q_ls_exit[i] = exit_dyn.q_ll;
            d2q_ls_entry[i] = entry_dyn.q_ll;
            d3q_tls_ls_exit[i] = exit_dyn.q_tl_ls;
            d3q_tls_ls_entry[i] = entry_dyn.q_tl_ls;
            d3q_ls_exit[i] = exit_dyn.q_ll_ls;
            d3q_ls_entry[i] = entry_dyn.q_ll_ls;
            d4q_tls_ls_ls_exit[i] = exit_dyn.q_tl_ls_ls;
            d4q_tls_ls_ls_entry[i] = entry_dyn.q_tl_ls_ls;
            d4q_ls_exit[i] = exit_dyn.q_llll;
            d4q_ls_entry[i] = entry_dyn.q_llll;
            dqdot_t[i] = exit_dyn.qdot_t;
            dqdot_ls[i] = exit_dyn.qdot_ls;
            dqdot_td[i] = exit_dyn.qdot_td;
            dqdot_lsd[i] = exit_dyn.qdot_lsd;
            d2qdot_tt[i] = exit_dyn.qdot_tt;
            d2qdot_tls[i] = exit_dyn.qdot_tls;
            d2qdot_ttd[i] = exit_dyn.qdot_ttd;
            d2qdot_tlsd[i] = exit_dyn.qdot_tlsd;
            d2qdot_ls[i] = exit_dyn.qdot_ll;
            d2qdot_lstd[i] = exit_dyn.qdot_lstd;
            d2qdot_lslsd[i] = exit_dyn.qdot_llsd;
        }

        let wiggle_qdot_basis_exit = wiggle_exit.as_ref().map(|wig| {
            let mut out = wig.basis_d1.clone();
            let r = Array1::from_iter((0..n).map(|i| {
                let base_exit = survival_base_q_scalars(eta_t_exit[i], eta_ls_exit[i]);
                survival_q0dot_from_base(base_exit, eta_t_deriv_exit[i], eta_ls_deriv_exit[i])
            }));
            for i in 0..n {
                out.row_mut(i).mapv_inplace(|v| v * r[i]);
            }
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
            d4q_tls_ls_ls_exit,
            d4q_tls_ls_ls_entry,
            d4q_ls_exit,
            d4q_ls_entry,
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
    eta_ls: Array1<f64>,
    etaw: Option<Array1<f64>>,
    wiggle_design: Option<Array2<f64>>,
    dq_dq0: Option<Array1<f64>>,
}

fn prediction_linear_predictors(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
) -> Result<PredictionLinearPredictors, String> {
    validate_predict_inverse_link(&input.inverse_link)?;
    let n = input.x_time_exit.nrows();
    let beta_time = fit.beta_time();
    let beta_threshold = fit.beta_threshold();
    let beta_log_sigma = fit.beta_log_sigma();
    let beta_link_wiggle = fit.beta_link_wiggle();
    if input.x_time_exit.ncols() != beta_time.len() {
        return Err(format!(
            "predict_survival_location_scale: time design/beta mismatch: {} vs {}",
            input.x_time_exit.ncols(),
            beta_time.len()
        ));
    }
    if input.eta_time_offset_exit.len() != n
        || input.x_threshold.nrows() != n
        || input.eta_threshold_offset.len() != n
        || input.x_log_sigma.nrows() != n
        || input.eta_log_sigma_offset.len() != n
    {
        return Err("predict_survival_location_scale: row mismatch across inputs".to_string());
    }
    let h_base = input.x_time_exit.dot(&beta_time) + &input.eta_time_offset_exit;
    let mut h = h_base.clone();
    let mut time_jac = input.x_time_exit.clone();
    if input.time_wiggle_ncols > 0 {
        let p_time = beta_time.len();
        let p_w = input.time_wiggle_ncols.min(p_time);
        let time_tail = p_time - p_w..p_time;
        let knots = input.time_wiggle_knots.as_ref().ok_or_else(|| {
            "predict_survival_location_scale: timewiggle coefficients are missing knot metadata"
                .to_string()
        })?;
        let degree = input.time_wiggle_degree.ok_or_else(|| {
            "predict_survival_location_scale: timewiggle coefficients are missing degree metadata"
                .to_string()
        })?;
        let beta_time_w = beta_time
            .slice(s![time_tail.start..time_tail.end])
            .to_owned();
        let time_basis =
            monotone_wiggle_basis_with_derivative_order(h_base.view(), knots, degree, 0)?;
        let time_basis_d1 =
            monotone_wiggle_basis_with_derivative_order(h_base.view(), knots, degree, 1)?;
        if time_basis.ncols() != beta_time_w.len() || time_basis_d1.ncols() != beta_time_w.len() {
            return Err(format!(
                "predict_survival_location_scale: timewiggle design/beta mismatch: value={} deriv={} beta={}",
                time_basis.ncols(),
                time_basis_d1.ncols(),
                beta_time_w.len()
            ));
        }
        let dq = time_basis_d1.dot(&beta_time_w) + 1.0;
        h = &h_base + &time_basis.dot(&beta_time_w);
        time_jac = scale_dense_rows(&input.x_time_exit, &dq)?;
        time_jac
            .slice_mut(s![.., time_tail.start..time_tail.end])
            .assign(&time_basis);
    }
    let eta_t =
        input.x_threshold.matrixvectormultiply(&beta_threshold) + &input.eta_threshold_offset;
    let eta_ls =
        input.x_log_sigma.matrixvectormultiply(&beta_log_sigma) + &input.eta_log_sigma_offset;
    let resolved_wiggle_knots = input
        .link_wiggle_knots
        .as_ref()
        .or(fit.artifacts.survival_link_wiggle_knots.as_ref());
    let resolved_wiggle_degree = input
        .link_wiggle_degree
        .or(fit.artifacts.survival_link_wiggle_degree);
    let q0 = Array1::from_iter(
        eta_t
            .iter()
            .zip(eta_ls.iter())
            .map(|(&t, &ls)| survival_q0_from_eta(t, ls)),
    );
    let (wiggle_design, dq_dq0, etaw) = if let Some(betaw) = beta_link_wiggle.as_ref() {
        let knots = resolved_wiggle_knots.ok_or_else(|| {
            "predict_survival_location_scale: link-wiggle coefficients are missing knot metadata"
                .to_string()
        })?;
        let degree = resolved_wiggle_degree.ok_or_else(|| {
            "predict_survival_location_scale: link-wiggle coefficients are missing degree metadata"
                .to_string()
        })?;
        let design =
            survival_wiggle_basis_with_options(q0.view(), knots, degree, BasisOptions::value())?;
        if design.ncols() != betaw.len() {
            return Err(format!(
                "predict_survival_location_scale: link-wiggle design/beta mismatch: {} vs {}",
                design.ncols(),
                betaw.len()
            ));
        }
        let basis_d1 = survival_wiggle_basis_with_options(
            q0.view(),
            knots,
            degree,
            BasisOptions::first_derivative(),
        )?;
        let dq = Some(basis_d1.dot(betaw) + 1.0);
        let etaw = design.dot(betaw);
        (Some(design), dq, Some(etaw))
    } else {
        (None, None, None)
    };
    Ok(PredictionLinearPredictors {
        h,
        time_jac,
        eta_t,
        eta_ls,
        etaw,
        wiggle_design,
        dq_dq0,
    })
}

fn survival_response_moment_block_ranges(
    p_time: usize,
    p_t: usize,
    p_ls: usize,
    pw: usize,
) -> (
    std::ops::Range<usize>,
    std::ops::Range<usize>,
    std::ops::Range<usize>,
    Option<std::ops::Range<usize>>,
) {
    let time = 0..p_time;
    let threshold = time.end..time.end + p_t;
    let log_sigma = threshold.end..threshold.end + p_ls;
    let wiggle = (pw > 0).then_some(log_sigma.end..log_sigma.end + pw);
    (time, threshold, log_sigma, wiggle)
}

fn projected_survival_response_moment_covariance(
    covariance: &Array2<f64>,
    a_h: &Array1<f64>,
    a_t: &Array1<f64>,
    a_ls: &Array1<f64>,
    p_time: usize,
    p_t: usize,
    p_ls: usize,
) -> [[f64; 3]; 3] {
    let (time, threshold, log_sigma, _) =
        survival_response_moment_block_ranges(p_time, p_t, p_ls, 0);
    let cov_hh = covariance.slice(s![time.start..time.end, time.start..time.end]);
    let cov_tt = covariance.slice(s![
        threshold.start..threshold.end,
        threshold.start..threshold.end
    ]);
    let cov_ll = covariance.slice(s![
        log_sigma.start..log_sigma.end,
        log_sigma.start..log_sigma.end
    ]);
    let cov_ht = covariance.slice(s![time.start..time.end, threshold.start..threshold.end]);
    let cov_hl = covariance.slice(s![time.start..time.end, log_sigma.start..log_sigma.end]);
    let cov_tl = covariance.slice(s![
        threshold.start..threshold.end,
        log_sigma.start..log_sigma.end
    ]);
    let var_h = a_h.dot(&cov_hh.dot(a_h));
    let var_t = a_t.dot(&cov_tt.dot(a_t));
    let var_ls = a_ls.dot(&cov_ll.dot(a_ls));
    let cov_ht_i = a_h.dot(&cov_ht.dot(a_t));
    let cov_hl_i = a_h.dot(&cov_hl.dot(a_ls));
    let cov_tl_i = a_t.dot(&cov_tl.dot(a_ls));
    [
        [var_h, cov_ht_i, cov_hl_i],
        [cov_ht_i, var_t, cov_tl_i],
        [cov_hl_i, cov_tl_i, var_ls],
    ]
}

fn covariance3_to_array2(cov: [[f64; 3]; 3]) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((3, 3));
    for i in 0..3 {
        for j in 0..3 {
            out[[i, j]] = cov[i][j];
        }
    }
    out
}

fn symmetrize_and_clip_covariance(cov: &Array2<f64>) -> Array2<f64> {
    let mut out = cov.clone();
    for i in 0..out.nrows() {
        out[[i, i]] = out[[i, i]].max(0.0);
        for j in (i + 1)..out.ncols() {
            let avg = 0.5 * (out[[i, j]] + out[[j, i]]);
            out[[i, j]] = avg;
            out[[j, i]] = avg;
        }
    }
    out
}

struct LowRankGaussianFactor {
    factor: Array2<f64>,
    eigenvectors: Array2<f64>,
    inv_sqrt_eigenvalues: Array1<f64>,
}

// Exact projected-Gaussian handling for possibly singular covariance blocks.
// We integrate over the active standard-normal coordinates rather than adding
// jitter or inverting the covariance directly.
fn factorize_psd_covariance(
    covariance: &Array2<f64>,
    label: &str,
) -> Result<LowRankGaussianFactor, String> {
    let covariance = symmetrize_and_clip_covariance(covariance);
    let (eigenvalues, eigenvectors_full) = covariance
        .eigh(faer::Side::Lower)
        .map_err(|e| format!("{label} eigendecomposition failed: {e}"))?;
    let max_abs_eigenvalue = eigenvalues
        .iter()
        .fold(0.0_f64, |acc, &ev| acc.max(ev.abs()));
    let tol = (max_abs_eigenvalue * 1e-12).max(1e-14);
    if eigenvalues.iter().any(|&ev| ev < -tol) {
        return Err(format!(
            "{label} is not positive semidefinite: minimum eigenvalue {:.3e}",
            eigenvalues
                .iter()
                .fold(f64::INFINITY, |acc, &ev| acc.min(ev))
        ));
    }

    let active = eigenvalues
        .iter()
        .enumerate()
        .filter_map(|(idx, &ev)| (ev > tol).then_some((idx, ev.sqrt())))
        .collect::<Vec<_>>();
    let mut factor = Array2::<f64>::zeros((covariance.nrows(), active.len()));
    let mut eigenvectors = Array2::<f64>::zeros((covariance.nrows(), active.len()));
    let mut inv_sqrt_eigenvalues = Array1::<f64>::zeros(active.len());
    for (col, (idx, sqrt_ev)) in active.into_iter().enumerate() {
        eigenvectors
            .column_mut(col)
            .assign(&eigenvectors_full.column(idx));
        factor
            .column_mut(col)
            .assign(&(&eigenvectors_full.column(idx) * sqrt_ev));
        inv_sqrt_eigenvalues[col] = 1.0 / sqrt_ev;
    }

    Ok(LowRankGaussianFactor {
        factor,
        eigenvectors,
        inv_sqrt_eigenvalues,
    })
}

fn apply_low_rank_gaussian_factor3(mu: [f64; 3], factor: &Array2<f64>, z: &[f64]) -> [f64; 3] {
    let mut x = mu;
    for row in 0..3 {
        for (col, &latent) in z.iter().enumerate() {
            x[row] += factor[[row, col]] * latent;
        }
    }
    x
}

fn low_rank_normal_expectation_pair_3d_result<F>(
    quadctx: &crate::quadrature::QuadratureContext,
    mu: [f64; 3],
    covariance: [[f64; 3]; 3],
    max_n: usize,
    label: &str,
    integrand: F,
) -> Result<(f64, f64), String>
where
    F: Fn([f64; 3], &[f64]) -> Result<(f64, f64), String>,
{
    let factorization = factorize_psd_covariance(&covariance3_to_array2(covariance), label)?;
    match factorization.factor.ncols() {
        0 => integrand(mu, &[]),
        1 => crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, String>(
            quadctx,
            [0.0],
            [[1.0]],
            max_n,
            |z| {
                let latent = [z[0]];
                integrand(
                    apply_low_rank_gaussian_factor3(mu, &factorization.factor, &latent),
                    &latent,
                )
            },
        ),
        2 => crate::quadrature::normal_expectation_nd_adaptive_result::<2, _, _, String>(
            quadctx,
            [0.0, 0.0],
            [[1.0, 0.0], [0.0, 1.0]],
            max_n,
            |z| {
                let latent = [z[0], z[1]];
                integrand(
                    apply_low_rank_gaussian_factor3(mu, &factorization.factor, &latent),
                    &latent,
                )
            },
        ),
        3 => crate::quadrature::normal_expectation_nd_adaptive_result::<3, _, _, String>(
            quadctx,
            [0.0, 0.0, 0.0],
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            max_n,
            |z| {
                let latent = [z[0], z[1], z[2]];
                integrand(
                    apply_low_rank_gaussian_factor3(mu, &factorization.factor, &latent),
                    &latent,
                )
            },
        ),
        rank => Err(format!("{label} unexpectedly has rank {rank} > 3")),
    }
}

// Exact response moments must stay in the original Gaussian coordinates:
// [h, threshold, log_sigma] for non-wiggle predictions, with a nested
// conditional Gaussian over the scalar link-wiggle contribution when present.
fn exact_survival_response_moments_row(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
    x_threshold_dense: &Array2<f64>,
    x_log_sigma_dense: &Array2<f64>,
    row: usize,
    quadctx: &crate::quadrature::QuadratureContext,
) -> Result<(f64, f64), String> {
    if input.time_wiggle_ncols > 0 {
        return Err(
            "predict_survival_location_scale: exact response moments are not implemented for time-wiggle models"
                .to_string(),
        );
    }

    let beta_time = fit.beta_time();
    let beta_threshold = fit.beta_threshold();
    let beta_log_sigma = fit.beta_log_sigma();
    let beta_link_wiggle = fit.beta_link_wiggle();
    let p_time = beta_time.len();
    let p_t = beta_threshold.len();
    let p_ls = beta_log_sigma.len();
    let pw = beta_link_wiggle.as_ref().map_or(0, |beta| beta.len());
    let (time, threshold, log_sigma, wiggle) =
        survival_response_moment_block_ranges(p_time, p_t, p_ls, pw);

    let a_h = input.x_time_exit.row(row).to_owned();
    let a_t = x_threshold_dense.row(row).to_owned();
    let a_ls = x_log_sigma_dense.row(row).to_owned();

    let mu_h = a_h.dot(&beta_time) + input.eta_time_offset_exit[row];
    let mu_t = a_t.dot(&beta_threshold) + input.eta_threshold_offset[row];
    let mu_ls = a_ls.dot(&beta_log_sigma) + input.eta_log_sigma_offset[row];
    let mu = [mu_h, mu_t, mu_ls];
    let cov_htl = projected_survival_response_moment_covariance(
        covariance, &a_h, &a_t, &a_ls, p_time, p_t, p_ls,
    );

    if let (Some(beta_w), Some(wiggle_range)) = (beta_link_wiggle.as_ref(), wiggle) {
        let knots = input
            .link_wiggle_knots
            .as_ref()
            .or(fit.artifacts.survival_link_wiggle_knots.as_ref())
            .ok_or_else(|| {
                "predict_survival_location_scale: link-wiggle coefficients are missing knot metadata"
                    .to_string()
            })?;
        let degree = input
            .link_wiggle_degree
            .or(fit.artifacts.survival_link_wiggle_degree)
            .ok_or_else(|| {
                "predict_survival_location_scale: link-wiggle coefficients are missing degree metadata"
                    .to_string()
            })?;

        let htl_factor = factorize_psd_covariance(
            &covariance3_to_array2(cov_htl),
            "survival response-moment projected covariance",
        )?;

        let cov_wy = {
            let mut out = Array2::<f64>::zeros((pw, 3));
            let cov_wh = covariance
                .slice(s![
                    wiggle_range.start..wiggle_range.end,
                    time.start..time.end
                ])
                .to_owned();
            let cov_wt = covariance
                .slice(s![
                    wiggle_range.start..wiggle_range.end,
                    threshold.start..threshold.end
                ])
                .to_owned();
            let cov_wl = covariance
                .slice(s![
                    wiggle_range.start..wiggle_range.end,
                    log_sigma.start..log_sigma.end
                ])
                .to_owned();
            out.column_mut(0).assign(&cov_wh.dot(&a_h));
            out.column_mut(1).assign(&cov_wt.dot(&a_t));
            out.column_mut(2).assign(&cov_wl.dot(&a_ls));
            out
        };
        let cov_ww = covariance
            .slice(s![
                wiggle_range.start..wiggle_range.end,
                wiggle_range.start..wiggle_range.end
            ])
            .to_owned();
        let mut regression = cov_wy.dot(&htl_factor.eigenvectors);
        for col in 0..regression.ncols() {
            let scale = htl_factor.inv_sqrt_eigenvalues[col];
            regression
                .column_mut(col)
                .mapv_inplace(|value| value * scale);
        }
        let cov_cond =
            symmetrize_and_clip_covariance(&(cov_ww - regression.dot(&regression.t().to_owned())));

        return low_rank_normal_expectation_pair_3d_result(
            quadctx,
            mu,
            cov_htl,
            15,
            "survival response-moment projected covariance",
            |x, z| {
                let mut cond_mean = beta_w.to_owned();
                for j in 0..pw {
                    for (col, &latent) in z.iter().enumerate() {
                        cond_mean[j] += regression[[j, col]] * latent;
                    }
                }
                let q0 = survival_q0_from_eta(x[1], x[2]);
                let q0_arr = Array1::from_vec(vec![q0]);
                let basis = survival_wiggle_basis_with_options(
                    q0_arr.view(),
                    knots,
                    degree,
                    BasisOptions::value(),
                )?;
                if basis.ncols() != cond_mean.len() {
                    return Err(format!(
                        "predict_survival_location_scale: link-wiggle basis/beta mismatch: {} vs {}",
                        basis.ncols(),
                        cond_mean.len()
                    ));
                }
                let b = basis.row(0).to_owned();
                let w_mean = b.dot(&cond_mean);
                let w_var = b.dot(&cov_cond.dot(&b)).max(0.0);
                crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, String>(
                    quadctx,
                    [x[0] + q0 + w_mean],
                    [[w_var]],
                    21,
                    |eta| {
                        let p = inverse_link_survival_prob_checked(&input.inverse_link, eta[0])?;
                        Ok((p, p * p))
                    },
                )
            },
        )
        .map(|(first, second)| (first.clamp(0.0, 1.0), second.clamp(0.0, 1.0)));
    }

    low_rank_normal_expectation_pair_3d_result(
        quadctx,
        mu,
        cov_htl,
        15,
        "survival response-moment projected covariance",
        |x, _| {
            let p = inverse_link_survival_prob_checked(
                &input.inverse_link,
                x[0] + survival_q0_from_eta(x[1], x[2]),
            )?;
            Ok((p, p * p))
        },
    )
    .map(|(first, second)| (first.clamp(0.0, 1.0), second.clamp(0.0, 1.0)))
}

fn exact_survival_response_moments(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
) -> Result<(Array1<f64>, Array1<f64>), String> {
    validate_predict_inverse_link(&input.inverse_link)?;

    let n = input.x_time_exit.nrows();
    let p_time = fit.beta_time().len();
    let p_t = fit.beta_threshold().len();
    let p_ls = fit.beta_log_sigma().len();
    let pw = fit.beta_link_wiggle().map_or(0, |beta| beta.len());
    let p_total = p_time + p_t + p_ls + pw;
    if covariance.nrows() != p_total || covariance.ncols() != p_total {
        return Err(format!(
            "predict_survival_location_scale: covariance shape mismatch: got {}x{}, expected {}x{}",
            covariance.nrows(),
            covariance.ncols(),
            p_total,
            p_total
        ));
    }
    if input.x_time_exit.ncols() != p_time {
        return Err(format!(
            "predict_survival_location_scale: time design/beta mismatch: {} vs {}",
            input.x_time_exit.ncols(),
            p_time
        ));
    }
    if input.eta_time_offset_exit.len() != n
        || input.x_threshold.nrows() != n
        || input.eta_threshold_offset.len() != n
        || input.x_log_sigma.nrows() != n
        || input.eta_log_sigma_offset.len() != n
    {
        return Err("predict_survival_location_scale: row mismatch across inputs".to_string());
    }

    let quadctx = crate::quadrature::QuadratureContext::new();
    let x_threshold_dense = input.x_threshold.to_dense_arc();
    let x_log_sigma_dense = input.x_log_sigma.to_dense_arc();
    let mut first = Array1::<f64>::zeros(n);
    let mut second = Array1::<f64>::zeros(n);
    for row in 0..n {
        let (m1, m2) = exact_survival_response_moments_row(
            input,
            fit,
            covariance,
            &x_threshold_dense,
            &x_log_sigma_dense,
            row,
            &quadctx,
        )?;
        first[row] = m1;
        second[row] = m2;
    }
    Ok((first, second))
}

fn lift_conditional_covariance(
    cov_reduced: &Array2<f64>,
    z: &Array2<f64>,
    p_threshold: usize,
    p_log_sigma: usize,
    p_linkwiggle: usize,
) -> Array2<f64> {
    let p_time_reduced = z.ncols();
    let p_time_full = z.nrows();
    let p_reduced = p_time_reduced + p_threshold + p_log_sigma + p_linkwiggle;
    let p_full = p_time_full + p_threshold + p_log_sigma + p_linkwiggle;
    if cov_reduced.nrows() != p_reduced || cov_reduced.ncols() != p_reduced {
        return cov_reduced.clone();
    }

    let mut t_map = Array2::<f64>::zeros((p_full, p_reduced));
    t_map
        .slice_mut(s![0..p_time_full, 0..p_time_reduced])
        .assign(z);
    for j in 0..p_threshold {
        t_map[[p_time_full + j, p_time_reduced + j]] = 1.0;
    }
    for j in 0..p_log_sigma {
        t_map[[
            p_time_full + p_threshold + j,
            p_time_reduced + p_threshold + j,
        ]] = 1.0;
    }
    for j in 0..p_linkwiggle {
        t_map[[
            p_time_full + p_threshold + p_log_sigma + j,
            p_time_reduced + p_threshold + p_log_sigma + j,
        ]] = 1.0;
    }
    t_map.dot(cov_reduced).dot(&t_map.t())
}

impl SurvivalLocationScaleFamily {
    fn assemble_joint_hessian_from_quantities(
        &self,
        q: &SurvivalJointQuantities,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let joint_states = self.validate_joint_states(block_states)?;
        let eta_t_exit = joint_states.3;
        let eta_t_entry = joint_states.5;
        let eta_t_deriv_exit = joint_states.7;
        let eta_ls_deriv_exit = joint_states.8;
        let eta_t_deriv_exit = eta_t_deriv_exit
            .map(|v| v.to_owned())
            .unwrap_or_else(|| Array1::zeros(self.n));
        let eta_ls_deriv_exit = eta_ls_deriv_exit
            .map(|v| v.to_owned())
            .unwrap_or_else(|| Array1::zeros(self.n));
        let offsets = self.joint_block_offsets();
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        let x_threshold_exit_cow = self.x_threshold.as_dense_cow();
        let x_threshold_exit = &*x_threshold_exit_cow;
        let x_threshold_entry_cow = self
            .x_threshold_entry
            .as_ref()
            .map(DesignMatrix::as_dense_cow);
        let x_threshold_entry = x_threshold_entry_cow
            .as_ref()
            .map_or(x_threshold_exit, |c| &**c);
        let x_threshold_deriv_cow = self
            .x_threshold_deriv
            .as_ref()
            .map(DesignMatrix::as_dense_cow);
        let x_threshold_deriv = x_threshold_deriv_cow.as_ref().map(|c| &**c);
        let x_log_sigma_exit_cow = self.x_log_sigma.as_dense_cow();
        let x_log_sigma_exit = &*x_log_sigma_exit_cow;
        let x_log_sigma_entry_cow = self
            .x_log_sigma_entry
            .as_ref()
            .map(DesignMatrix::as_dense_cow);
        let x_log_sigma_entry = x_log_sigma_entry_cow
            .as_ref()
            .map_or(x_log_sigma_exit, |c| &**c);
        let x_log_sigma_deriv_cow = self
            .x_log_sigma_deriv
            .as_ref()
            .map(DesignMatrix::as_dense_cow);
        let x_log_sigma_deriv = x_log_sigma_deriv_cow.as_ref().map(|c| &**c);
        let mut joint = Array2::<f64>::zeros((p_total, p_total));
        let add_cross = |acc: &mut Array2<f64>,
                         left: &Array2<f64>,
                         weights: &Array1<f64>,
                         right: &Array2<f64>|
         -> Result<(), String> {
            *acc += &weighted_crossprod_dense(left, weights, right)?;
            Ok(())
        };

        let h_time = safe_fast_xt_diag_x(&dynamic.time_jac_entry, &(-&q.h_time_h0))
            + safe_fast_xt_diag_x(&dynamic.time_jac_exit, &(-&q.h_time_h1))
            + safe_fast_xt_diag_x(&dynamic.time_jac_deriv, &q.h_time_d);
        assign_symmetric_block(&mut joint, offsets[0], offsets[0], &h_time);

        if let Some(x_t_deriv) = x_threshold_deriv {
            let h_exit = -(&q.d2_q1 * &q.dq_t.mapv(|v| safe_product(v, v))
                + &q.d2_qdot1 * &q.dqdot_t.mapv(|v| safe_product(v, v))
                + &q.d1_qdot1 * &q.d2qdot_tt);
            let h_entry =
                -(&q.d2_q0 * &q.dq_t_entry.as_ref().unwrap().mapv(|v| safe_product(v, v)));
            let h_deriv = -(&q.d2_qdot1 * &q.dqdot_td.mapv(|v| safe_product(v, v)));
            let h_exit_deriv =
                -(&q.d2_qdot1 * &(&q.dqdot_t * &q.dqdot_td) + &q.d1_qdot1 * &q.d2qdot_ttd);
            let mut h_tt = weighted_crossprod_dense(x_threshold_exit, &h_exit, x_threshold_exit)?
                + weighted_crossprod_dense(x_threshold_entry, &h_entry, x_threshold_entry)?
                + weighted_crossprod_dense(x_t_deriv, &h_deriv, x_t_deriv)?;
            let cross = weighted_crossprod_dense(x_threshold_exit, &h_exit_deriv, x_t_deriv)?;
            h_tt += &cross;
            h_tt += &cross.t().to_owned();
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &h_tt);
        } else {
            let h_t = -(&q.d2_q1 * &q.dq_t.mapv(|v| safe_product(v, v))
                + &q.d2_q0 * &q.dq_t_entry.as_ref().unwrap().mapv(|v| safe_product(v, v))
                + &q.d2_qdot1 * &q.dqdot_t.mapv(|v| safe_product(v, v))
                + &q.d1_qdot1 * &q.d2qdot_tt);
            let h_tt = weighted_crossprod_dense(&x_threshold_exit, &h_t, &x_threshold_exit)?;
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &h_tt);
        }

        if let Some(x_ls_deriv) = x_log_sigma_deriv {
            let dq_ls_entry = q.dq_ls_entry.as_ref().unwrap();
            let d2q_ls_entry = q.d2q_ls_entry.as_ref().unwrap();
            let h_exit = -(&q.d2_q1 * &q.dq_ls.mapv(|v| safe_product(v, v))
                + &(&q.d1_q1 * &q.d2q_ls)
                + &q.d2_qdot1 * &q.dqdot_ls.mapv(|v| safe_product(v, v))
                + &(&q.d1_qdot1 * &q.d2qdot_ls));
            let h_entry = -(&q.d2_q0 * &dq_ls_entry.mapv(|v| safe_product(v, v))
                + &(&q.d1_q0 * d2q_ls_entry));
            let h_deriv = -(&q.d2_qdot1 * &q.dqdot_lsd.mapv(|v| safe_product(v, v)));
            let h_exit_deriv =
                -(&q.d2_qdot1 * &(&q.dqdot_ls * &q.dqdot_lsd) + &q.d1_qdot1 * &q.d2qdot_lslsd);
            let mut h_ll = weighted_crossprod_dense(x_log_sigma_exit, &h_exit, x_log_sigma_exit)?
                + weighted_crossprod_dense(x_log_sigma_entry, &h_entry, x_log_sigma_entry)?
                + weighted_crossprod_dense(x_ls_deriv, &h_deriv, x_ls_deriv)?;
            let cross = weighted_crossprod_dense(x_log_sigma_exit, &h_exit_deriv, x_ls_deriv)?;
            h_ll += &cross;
            h_ll += &cross.t().to_owned();
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &h_ll);
        } else {
            let h_ls = -(&q.d2_q1 * &q.dq_ls.mapv(|v| safe_product(v, v))
                + &(&q.d1_q1 * &q.d2q_ls)
                + &q.d2_q0 * &q.dq_ls_entry.as_ref().unwrap().mapv(|v| safe_product(v, v))
                + &(&q.d1_q0 * q.d2q_ls_entry.as_ref().unwrap())
                + &q.d2_qdot1 * &q.dqdot_ls.mapv(|v| safe_product(v, v))
                + &(&q.d1_qdot1 * &q.d2qdot_ls));
            let h_ll = weighted_crossprod_dense(&x_log_sigma_exit, &h_ls, &x_log_sigma_exit)?;
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &h_ll);
        }

        {
            let mut h_tl = Array2::<f64>::zeros((offsets[2] - offsets[1], offsets[3] - offsets[2]));
            let w_exit = -(&q.d2_q1 * &(&q.dq_t * &q.dq_ls) + &(&q.d1_q1 * &q.d2q_tls));
            let w_entry = -(&q.d2_q0
                * &(q.dq_t_entry.as_ref().unwrap() * q.dq_ls_entry.as_ref().unwrap())
                + &(&q.d1_q0 * q.d2q_tls_entry.as_ref().unwrap()));
            add_cross(&mut h_tl, x_threshold_exit, &w_exit, x_log_sigma_exit)?;
            add_cross(&mut h_tl, x_threshold_entry, &w_entry, x_log_sigma_entry)?;
            let w_qdot_exit =
                -(&q.d2_qdot1 * &(&q.dqdot_t * &q.dqdot_ls) + &(&q.d1_qdot1 * &q.d2qdot_tls));
            add_cross(&mut h_tl, x_threshold_exit, &w_qdot_exit, x_log_sigma_exit)?;
            if let Some(x_ls_deriv) = x_log_sigma_deriv {
                let w =
                    -(&q.d2_qdot1 * &(&q.dqdot_t * &q.dqdot_lsd) + &(&q.d1_qdot1 * &q.d2qdot_tlsd));
                add_cross(&mut h_tl, x_threshold_exit, &w, x_ls_deriv)?;
            }
            if let Some(x_t_deriv) = x_threshold_deriv {
                let w =
                    -(&q.d2_qdot1 * &(&q.dqdot_td * &q.dqdot_ls) + &(&q.d1_qdot1 * &q.d2qdot_lstd));
                add_cross(&mut h_tl, x_t_deriv, &w, x_log_sigma_exit)?;
                if let Some(x_ls_deriv) = x_log_sigma_deriv {
                    let wdd = -(&q.d2_qdot1 * &(&q.dqdot_td * &q.dqdot_lsd));
                    add_cross(&mut h_tl, x_t_deriv, &wdd, x_ls_deriv)?;
                }
            }
            assign_symmetric_block(&mut joint, offsets[1], offsets[2], &h_tl);
        }

        let mut h_ht = weighted_crossprod_dense(
            &self.x_time_entry,
            &(-&q.h_time_h0 * q.dq_t_entry.as_ref().unwrap()),
            x_threshold_entry,
        )? + weighted_crossprod_dense(
            &self.x_time_exit,
            &(-&q.h_time_h1 * &q.dq_t),
            x_threshold_exit,
        )? + weighted_crossprod_dense(
            &self.x_time_deriv,
            &(-&q.h_time_d * &q.dqdot_t),
            x_threshold_exit,
        )?;
        if let Some(x_t_deriv) = x_threshold_deriv {
            h_ht += &weighted_crossprod_dense(
                &self.x_time_deriv,
                &(-&q.h_time_d * &q.dqdot_td),
                x_t_deriv,
            )?;
        }
        assign_symmetric_block(&mut joint, offsets[0], offsets[1], &h_ht);

        let mut h_hl = weighted_crossprod_dense(
            &self.x_time_entry,
            &(-&q.h_time_h0 * q.dq_ls_entry.as_ref().unwrap()),
            x_log_sigma_entry,
        )? + weighted_crossprod_dense(
            &self.x_time_exit,
            &(-&q.h_time_h1 * &q.dq_ls),
            x_log_sigma_exit,
        )? + weighted_crossprod_dense(
            &self.x_time_deriv,
            &(-&q.h_time_d * &q.dqdot_ls),
            x_log_sigma_exit,
        )?;
        if let Some(x_ls_deriv) = x_log_sigma_deriv {
            h_hl += &weighted_crossprod_dense(
                &self.x_time_deriv,
                &(-&q.h_time_d * &q.dqdot_lsd),
                x_ls_deriv,
            )?;
        }
        assign_symmetric_block(&mut joint, offsets[0], offsets[2], &h_hl);

        if let (
            Some(xw_exit),
            Some(xw_entry),
            Some(xw_qdot),
            Some(xw_d1_exit),
            Some(xw_d1_entry),
            Some(xw_d2_exit),
            Some(w_offset),
        ) = (
            dynamic.wiggle_basis_exit.as_ref(),
            dynamic.wiggle_basis_entry.as_ref(),
            dynamic.wiggle_qdot_basis_exit.as_ref(),
            dynamic.wiggle_basis_d1_exit.as_ref(),
            dynamic.wiggle_basis_d1_entry.as_ref(),
            dynamic.wiggle_basis_d2_exit.as_ref(),
            offsets.get(3).copied(),
        ) {
            let hww = weighted_crossprod_dense(xw_exit, &(-&q.d2_q1), xw_exit)?
                + weighted_crossprod_dense(xw_entry, &(-&q.d2_q0), xw_entry)?
                + weighted_crossprod_dense(xw_qdot, &(-&q.d2_qdot1), xw_qdot)?;
            assign_symmetric_block(&mut joint, w_offset, w_offset, &hww);
            let q0_t_entry = Array1::from_iter(dynamic.inv_sigma_entry.iter().map(|&r| -r));
            let q0_t_exit = Array1::from_iter(dynamic.inv_sigma_exit.iter().map(|&r| -r));
            let q0_ls_entry = Array1::from_iter(
                (0..self.n)
                    .map(|i| q_chain_derivs_scalar(eta_t_entry[i], dynamic.eta_ls_entry[i]).1),
            );
            let q0_ls_exit = Array1::from_iter(
                (0..self.n).map(|i| q_chain_derivs_scalar(eta_t_exit[i], dynamic.eta_ls_exit[i]).1),
            );
            let r_base_exit = safe_linear_combo2_arrays(
                &q0_t_exit,
                &eta_t_deriv_exit,
                &q0_ls_exit,
                &eta_ls_deriv_exit,
            )?;
            let r_t_base_exit = Array1::from_iter((0..self.n).map(|i| {
                safe_product(
                    q_chain_derivs_scalar(eta_t_exit[i], dynamic.eta_ls_exit[i]).2,
                    eta_ls_deriv_exit[i],
                )
            }));
            let r_ls_base_exit = Array1::from_iter((0..self.n).map(|i| {
                let (_, _, q_tl, q_ll, _, _) =
                    q_chain_derivs_scalar(eta_t_exit[i], dynamic.eta_ls_exit[i]);
                safe_sum2(
                    safe_product(q_tl, eta_t_deriv_exit[i]),
                    safe_product(q_ll, eta_ls_deriv_exit[i]),
                )
            }));
            let tw_entry_d2 = scale_dense_rows(xw_d1_entry, &q0_t_entry)?;
            let tw_exit_d2 = scale_dense_rows(xw_d1_exit, &q0_t_exit)?;
            let lw_entry_d2 = scale_dense_rows(xw_d1_entry, &q0_ls_entry)?;
            let lw_exit_d2 = scale_dense_rows(xw_d1_exit, &q0_ls_exit)?;
            let qdot_t_w = scale_dense_rows(
                xw_d2_exit,
                &safe_hadamard_product(&q0_t_exit, &r_base_exit)?,
            )? + scale_dense_rows(xw_d1_exit, &r_t_base_exit)?;
            let qdot_ls_w = scale_dense_rows(
                xw_d2_exit,
                &safe_hadamard_product(&q0_ls_exit, &r_base_exit)?,
            )? + scale_dense_rows(xw_d1_exit, &r_ls_base_exit)?;
            let qdot_td_w = scale_dense_rows(xw_d1_exit, &q0_t_exit)?;
            let qdot_lsd_w = scale_dense_rows(xw_d1_exit, &q0_ls_exit)?;

            let mut h_tw = Array2::<f64>::zeros((offsets[2] - offsets[1], offsets[4] - offsets[3]));
            h_tw += &weighted_crossprod_dense(x_threshold_exit, &(-&q.d2_q1 * &q.dq_t), xw_exit)?;
            h_tw += &weighted_crossprod_dense(
                x_threshold_exit,
                &(-&q.d1_q1 * &q0_t_exit),
                &tw_exit_d2,
            )?;
            h_tw += &weighted_crossprod_dense(
                x_threshold_entry,
                &(-&q.d2_q0 * q.dq_t_entry.as_ref().unwrap()),
                xw_entry,
            )?;
            h_tw += &weighted_crossprod_dense(
                x_threshold_entry,
                &(-&q.d1_q0 * &q0_t_entry),
                &tw_entry_d2,
            )?;
            h_tw +=
                &weighted_crossprod_dense(x_threshold_exit, &(-&q.d2_qdot1 * &q.dqdot_t), xw_qdot)?;
            h_tw += &weighted_crossprod_dense(x_threshold_exit, &(-&q.d1_qdot1), &qdot_t_w)?;
            if let Some(x_t_deriv) = x_threshold_deriv {
                h_tw +=
                    &weighted_crossprod_dense(x_t_deriv, &(-&q.d2_qdot1 * &q.dqdot_td), xw_qdot)?;
                h_tw += &weighted_crossprod_dense(x_t_deriv, &(-&q.d1_qdot1), &qdot_td_w)?;
            }
            assign_symmetric_block(&mut joint, offsets[1], w_offset, &h_tw);

            let mut h_lw = Array2::<f64>::zeros((offsets[3] - offsets[2], offsets[4] - offsets[3]));
            h_lw += &weighted_crossprod_dense(x_log_sigma_exit, &(-&q.d2_q1 * &q.dq_ls), xw_exit)?;
            h_lw += &weighted_crossprod_dense(
                x_log_sigma_exit,
                &(-(&q.d1_q1 * &q0_ls_exit)),
                &lw_exit_d2,
            )?;
            h_lw += &weighted_crossprod_dense(
                x_log_sigma_entry,
                &(-&q.d2_q0 * q.dq_ls_entry.as_ref().unwrap()),
                xw_entry,
            )?;
            h_lw += &weighted_crossprod_dense(
                x_log_sigma_entry,
                &(-(&q.d1_q0 * &q0_ls_entry)),
                &lw_entry_d2,
            )?;
            h_lw += &weighted_crossprod_dense(
                x_log_sigma_exit,
                &(-&q.d2_qdot1 * &q.dqdot_ls),
                xw_qdot,
            )?;
            h_lw += &weighted_crossprod_dense(x_log_sigma_exit, &(-&q.d1_qdot1), &qdot_ls_w)?;
            if let Some(x_ls_deriv) = x_log_sigma_deriv {
                h_lw +=
                    &weighted_crossprod_dense(x_ls_deriv, &(-&q.d2_qdot1 * &q.dqdot_lsd), xw_qdot)?;
                h_lw += &weighted_crossprod_dense(x_ls_deriv, &(-&q.d1_qdot1), &qdot_lsd_w)?;
            }
            assign_symmetric_block(&mut joint, offsets[2], w_offset, &h_lw);

            let h_hw = weighted_crossprod_dense(&self.x_time_entry, &(-&q.h_time_h0), xw_entry)?
                + weighted_crossprod_dense(&self.x_time_exit, &(-&q.h_time_h1), xw_exit)?
                + weighted_crossprod_dense(&self.x_time_deriv, &(-&q.h_time_d), xw_qdot)?;
            assign_symmetric_block(&mut joint, offsets[0], w_offset, &h_hw);
        }

        Ok(Some(joint))
    }

    /// Compute the log-scale shift needed to keep CLogLog survival
    /// derivatives finite.  Returns `L >= 0` such that `exp(u - L) <= exp(500)`
    /// for all row linear predictors `u`.  For non-CLogLog links, returns 0.
    fn hessian_deriv_log_rescale(&self, block_states: &[ParameterBlockState]) -> f64 {
        if !matches!(
            self.inverse_link,
            InverseLink::Standard(LinkFunction::CLogLog)
        ) {
            return 0.0;
        }
        let dynamic = match self.build_dynamic_geometry(block_states) {
            Ok(d) => d,
            Err(_) => return 0.0,
        };
        let mut max_u = f64::NEG_INFINITY;
        for i in 0..self.n {
            if self.w[i] <= 0.0 {
                continue;
            }
            let u0 = dynamic.h_entry[i] + dynamic.q_entry[i];
            let u1 = dynamic.h_exit[i] + dynamic.q_exit[i];
            max_u = max_u.max(u0).max(u1);
        }
        // Shift so the largest exp(u - L) ~ exp(500), well within f64 range.
        (max_u - 500.0).max(0.0)
    }

    /// Rescaled joint Hessian for logdet computation.  Returns
    /// `(H_scaled, L)` where `H_scaled = exp(-L) * H_exact` and
    /// `logdet(H_exact) = logdet(H_scaled) + p * L`.
    fn exact_newton_joint_hessian_rescaled(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<(Array2<f64>, f64)>, String> {
        let log_scale = self.hessian_deriv_log_rescale(block_states);
        if log_scale == 0.0 {
            return Ok(self
                .exact_newton_joint_hessian(block_states)?
                .map(|h| (h, 0.0)));
        }
        let q = self.collect_joint_quantities_rescaled(block_states, log_scale)?;
        Ok(self
            .assemble_joint_hessian_from_quantities(&q, block_states)?
            .map(|h| (h, log_scale)))
    }

    fn exact_newton_joint_hessian_directional_derivative_rescaled(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
        log_rescale: f64,
    ) -> Result<Option<Array2<f64>>, String> {
        let q = self.collect_joint_quantities_rescaled(block_states, log_rescale)?;
        let offsets = self.joint_block_offsets();
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_flat.len() != p_total {
            return Err(format!(
                "joint d_beta length mismatch: got {}, expected {p_total}",
                d_beta_flat.len()
            ));
        }

        let dynamic = self.build_dynamic_geometry(block_states)?;

        let time_dir = d_beta_flat.slice(s![offsets[0]..offsets[1]]).to_owned();
        let threshold_dir = d_beta_flat.slice(s![offsets[1]..offsets[2]]).to_owned();
        let log_sigma_dir = d_beta_flat.slice(s![offsets[2]..offsets[3]]).to_owned();
        let wiggle_dir = if self.x_link_wiggle.is_some() {
            Some(d_beta_flat.slice(s![offsets[3]..offsets[4]]).to_owned())
        } else {
            None
        };

        let delta_h0 = dynamic.time_jac_entry.dot(&time_dir);
        let delta_h1 = dynamic.time_jac_exit.dot(&time_dir);
        let delta_d = dynamic.time_jac_deriv.dot(&time_dir);
        let delta_t_exit = self.x_threshold.matrixvectormultiply(&threshold_dir);
        let delta_ls_exit = self.x_log_sigma.matrixvectormultiply(&log_sigma_dir);
        let deltaw = match (self.x_link_wiggle.as_ref(), wiggle_dir.as_ref()) {
            (Some(xw), Some(dir)) => xw.matrixvectormultiply(dir),
            _ => Array1::zeros(self.n),
        };

        let delta_q_exit = &q.dq_t * &delta_t_exit + &q.dq_ls * &delta_ls_exit + &deltaw;
        let delta_q_t_exit = &q.d2q_tls * &delta_ls_exit;
        let delta_q_ls_exit = &q.d2q_tls * &delta_t_exit + &q.d2q_ls * &delta_ls_exit;
        let delta_q_tls_exit = &q.d3q_tls_ls * &delta_ls_exit;
        let delta_q_ls_ls_exit = &q.d3q_tls_ls * &delta_t_exit + &q.d3q_ls * &delta_ls_exit;

        let d_d1_q_exit = &q.d2_q1 * &delta_q_exit + &q.h_time_h1 * &delta_h1;
        let d_d2_q_exit = &q.d3_q1 * &delta_q_exit + &q.d_h_h1 * &delta_h1;

        let x_threshold_exit_cow = self.x_threshold.as_dense_cow();
        let x_threshold_exit = &*x_threshold_exit_cow;
        let x_threshold_entry_cow = self
            .x_threshold_entry
            .as_ref()
            .map(DesignMatrix::as_dense_cow);
        let x_threshold_entry = x_threshold_entry_cow.as_ref().map(|c| &**c);
        let x_log_sigma_exit_cow = self.x_log_sigma.as_dense_cow();
        let x_log_sigma_exit = &*x_log_sigma_exit_cow;
        let x_log_sigma_entry_cow = self
            .x_log_sigma_entry
            .as_ref()
            .map(DesignMatrix::as_dense_cow);
        let x_log_sigma_entry = x_log_sigma_entry_cow.as_ref().map(|c| &**c);
        let xw_cow = self.x_link_wiggle.as_ref().map(DesignMatrix::as_dense_cow);
        let xw = xw_cow.as_ref().map(|c| &**c);
        let mut joint = Array2::<f64>::zeros((p_total, p_total));

        struct EntryDeltas {
            delta_q: Array1<f64>,
            delta_q_t: Array1<f64>,
            delta_q_ls: Array1<f64>,
            delta_q_tls: Array1<f64>,
            delta_q_ls_ls: Array1<f64>,
            d_d1_q: Array1<f64>,
            d_d2_q: Array1<f64>,
        }
        let entry_deltas = if x_threshold_entry.is_some() || x_log_sigma_entry.is_some() {
            let dt_en = self
                .x_threshold_entry
                .as_ref()
                .map(|x| x.matrixvectormultiply(&threshold_dir))
                .unwrap_or_else(|| delta_t_exit.clone());
            let dls_en = self
                .x_log_sigma_entry
                .as_ref()
                .map(|x| x.matrixvectormultiply(&log_sigma_dir))
                .unwrap_or_else(|| delta_ls_exit.clone());
            let dq_t_en = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
            let dq_ls_en = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
            let d2q_tls_en = q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls);
            let d3q_tls_ls_en = q.d3q_tls_ls_entry.as_ref().unwrap_or(&q.d3q_tls_ls);
            let d3q_ls_en = q.d3q_ls_entry.as_ref().unwrap_or(&q.d3q_ls);
            let d2q_ls_en = q.d2q_ls_entry.as_ref().unwrap_or(&q.d2q_ls);
            let dq_en = dq_t_en * &dt_en + dq_ls_en * &dls_en + &deltaw;
            EntryDeltas {
                delta_q_t: d2q_tls_en * &dls_en,
                delta_q_ls: d2q_tls_en * &dt_en + d2q_ls_en * &dls_en,
                delta_q_tls: d3q_tls_ls_en * &dls_en,
                delta_q_ls_ls: d3q_tls_ls_en * &dt_en + d3q_ls_en * &dls_en,
                d_d1_q: &q.d2_q0 * &dq_en + &q.h_time_h0 * &delta_h0,
                d_d2_q: &q.d3_q0 * &dq_en + &q.d_h_h0 * &delta_h0,
                delta_q: dq_en,
            }
        } else {
            EntryDeltas {
                delta_q: delta_q_exit.clone(),
                delta_q_t: delta_q_t_exit.clone(),
                delta_q_ls: delta_q_ls_exit.clone(),
                delta_q_tls: delta_q_tls_exit.clone(),
                delta_q_ls_ls: delta_q_ls_ls_exit.clone(),
                d_d1_q: &q.d2_q0 * &delta_q_exit + &q.h_time_h0 * &delta_h0,
                d_d2_q: &q.d3_q0 * &delta_q_exit + &q.d_h_h0 * &delta_h0,
            }
        };

        // Time-time directional derivative of the joint Hessian block.
        //
        // The stored joint "Hessian" H equals -∂²ℓ/∂β². The time-time
        // diagonal base assembly (assemble_joint_hessian_from_quantities)
        // is
        //
        //   H_tt = X_entry'·diag(-h_time_h0)·X_entry
        //        + X_exit' ·diag(-h_time_h1)·X_exit
        //        + X_deriv'·diag(+h_time_d)·X_deriv
        //
        // with h_time_h0 = w·r'(u0)     (= ∂²ℓ/∂h0²),
        //      h_time_h1 = w·event_mix  (= ∂²ℓ/∂h1²),
        //      h_time_d  = -w·d·(d2logg) (= -∂²ℓ/∂d_raw²).
        //
        // Differentiating H_tt along Δβ_t (with Δh0 = X_entry·Δβ_t,
        // Δh1 = X_exit·Δβ_t, Δd = X_deriv·Δβ_t, and q0/q1 invariant in
        // a pure time direction) gives
        //
        //   dH_tt = X_entry'·diag(-w·r''(u0)·Δh0)·X_entry
        //         + X_exit' ·diag(-w·r'''-mixed·Δh1)·X_exit
        //         + X_deriv'·diag(+d_h_d·Δd)·X_deriv
        //         = X_entry'·diag(-d_h_h0·Δh0)·X_entry
        //         + X_exit' ·diag(-d_h_h1·Δh1)·X_exit
        //         + X_deriv'·diag(+d_h_d ·Δd )·X_deriv
        //
        // For non-time directions (Δβ_t = 0) the inner variables Δh0/Δh1/Δd
        // vanish but u0/u1 still shift through q0/q1. Chain rule:
        //   Δu0 = Δh0 + Δq_entry   (for pure time direction: Δq_entry = 0)
        //   Δu1 = Δh1 + Δq_exit
        // so the general form tracks Δu0 = Δh0 + entry_deltas.delta_q and
        // Δu1 = Δh1 + delta_q_exit. (No q-dependence on d_raw ⇒ Δd alone
        // drives the deriv-side weight derivative.)
        let du0 = &delta_h0 + &entry_deltas.delta_q;
        let du1 = &delta_h1 + &delta_q_exit;
        let dh_h0 = &q.d_h_h0 * &du0;
        let dh_h1 = &q.d_h_h1 * &du1;
        let dh_d = &q.d_h_d * &delta_d;
        let d_h_time = safe_fast_xt_diag_x(&dynamic.time_jac_entry, &(-&dh_h0))
            + safe_fast_xt_diag_x(&dynamic.time_jac_exit, &(-&dh_h1))
            + safe_fast_xt_diag_x(&dynamic.time_jac_deriv, &dh_d);
        assign_symmetric_block(&mut joint, offsets[0], offsets[0], &d_h_time);

        if let Some(x_t_en) = x_threshold_entry.as_ref() {
            let dq_t_en = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
            let d_h_exit = -(&d_d2_q_exit * &q.dq_t.mapv(|v| safe_product(v, v))
                + &(&q.d2_q1 * &(2.0 * &delta_q_t_exit * &q.dq_t)));
            let d_h_entry = -(&entry_deltas.d_d2_q * &dq_t_en.mapv(|v| safe_product(v, v))
                + &(&q.d2_q0 * &(2.0 * &entry_deltas.delta_q_t * dq_t_en)));
            let d_h_tt = weighted_crossprod_dense(&x_threshold_exit, &d_h_exit, &x_threshold_exit)?
                + weighted_crossprod_dense(x_t_en, &d_h_entry, x_t_en)?;
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &d_h_tt);
        } else {
            let d_d2_q_ti = &q.d3_q * &delta_q_exit + &q.d_h_h0 * &delta_h0 + &q.d_h_h1 * &delta_h1;
            let d_h_t = -(&d_d2_q_ti * &q.dq_t.mapv(|v| safe_product(v, v))
                + &(&q.d2_q * &(2.0 * &delta_q_t_exit * &q.dq_t)));
            let d_h_tt = weighted_crossprod_dense(&x_threshold_exit, &d_h_t, &x_threshold_exit)?;
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &d_h_tt);
        }

        {
            let has_t_entry = x_threshold_entry.is_some();
            let has_ls_entry = x_log_sigma_entry.is_some();
            if has_t_entry || has_ls_entry {
                let x_t_en = x_threshold_entry.unwrap_or(x_threshold_exit);
                let x_ls_en = x_log_sigma_entry.unwrap_or(x_log_sigma_exit);
                let dq_t_en = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
                let dq_ls_en = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
                let d2q_tls_en = q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls);
                let w_exit = -(&d_d2_q_exit * &(&q.dq_t * &q.dq_ls)
                    + &(&q.d2_q1 * &(&delta_q_t_exit * &q.dq_ls + &q.dq_t * &delta_q_ls_exit))
                    + &(&d_d1_q_exit * &q.d2q_tls)
                    + &(&q.d1_q1 * &delta_q_tls_exit));
                let w_entry = -(&entry_deltas.d_d2_q * &(dq_t_en * dq_ls_en)
                    + &(&q.d2_q0
                        * &(&entry_deltas.delta_q_t * dq_ls_en
                            + dq_t_en * &entry_deltas.delta_q_ls))
                    + &(&entry_deltas.d_d1_q * d2q_tls_en)
                    + &(&q.d1_q0 * &entry_deltas.delta_q_tls));
                let d_h_tl =
                    weighted_crossprod_dense(&x_threshold_exit, &w_exit, &x_log_sigma_exit)?
                        + weighted_crossprod_dense(x_t_en, &w_entry, x_ls_en)?;
                assign_symmetric_block(&mut joint, offsets[1], offsets[2], &d_h_tl);
            } else {
                let d_d1_q =
                    &q.d2_q * &delta_q_exit + &q.h_time_h0 * &delta_h0 + &q.h_time_h1 * &delta_h1;
                let d_d2_q =
                    &q.d3_q * &delta_q_exit + &q.d_h_h0 * &delta_h0 + &q.d_h_h1 * &delta_h1;
                let d_h_tlweights = -(&d_d2_q * &(&q.dq_t * &q.dq_ls)
                    + &(&q.d2_q * &(&delta_q_t_exit * &q.dq_ls + &q.dq_t * &delta_q_ls_exit))
                    + &(&d_d1_q * &q.d2q_tls)
                    + &(&q.d1_q * &delta_q_tls_exit));
                let d_h_tl =
                    weighted_crossprod_dense(&x_threshold_exit, &d_h_tlweights, &x_log_sigma_exit)?;
                assign_symmetric_block(&mut joint, offsets[1], offsets[2], &d_h_tl);
            }
        }

        if let Some(x_ls_en) = x_log_sigma_entry.as_ref() {
            let dq_ls_en = q.dq_ls_entry.as_ref().unwrap();
            let d2q_ls_en = q.d2q_ls_entry.as_ref().unwrap();
            let d_h_exit = -(&d_d2_q_exit * &q.dq_ls.mapv(|v| safe_product(v, v))
                + &(&q.d2_q1 * &(2.0 * &delta_q_ls_exit * &q.dq_ls))
                + &(&d_d1_q_exit * &q.d2q_ls)
                + &(&q.d1_q1 * &delta_q_ls_ls_exit));
            let d_h_entry = -(&entry_deltas.d_d2_q * &dq_ls_en.mapv(|v| safe_product(v, v))
                + &(&q.d2_q0 * &(2.0 * &entry_deltas.delta_q_ls * dq_ls_en))
                + &(&entry_deltas.d_d1_q * d2q_ls_en)
                + &(&q.d1_q0 * &entry_deltas.delta_q_ls_ls));
            let d_h_ll = weighted_crossprod_dense(&x_log_sigma_exit, &d_h_exit, &x_log_sigma_exit)?
                + weighted_crossprod_dense(x_ls_en, &d_h_entry, x_ls_en)?;
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &d_h_ll);
        } else {
            let d_d1_q =
                &q.d2_q * &delta_q_exit + &q.h_time_h0 * &delta_h0 + &q.h_time_h1 * &delta_h1;
            let d_d2_q = &q.d3_q * &delta_q_exit + &q.d_h_h0 * &delta_h0 + &q.d_h_h1 * &delta_h1;
            let d_h_l = -(&d_d2_q * &q.dq_ls.mapv(|v| safe_product(v, v))
                + &(&q.d2_q * &(2.0 * &delta_q_ls_exit * &q.dq_ls))
                + &(&d_d1_q * &q.d2q_ls)
                + &(&q.d1_q * &delta_q_ls_ls_exit));
            let d_h_ll = weighted_crossprod_dense(&x_log_sigma_exit, &d_h_l, &x_log_sigma_exit)?;
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &d_h_ll);
        }

        if let (Some(x_t_en), Some(dq_t_en)) = (x_threshold_entry.as_ref(), q.dq_t_entry.as_ref()) {
            let d_h_h0_t = weighted_crossprod_dense(
                &self.x_time_entry,
                &(-(&dh_h0 * dq_t_en + &q.h_time_h0 * &entry_deltas.delta_q_t)),
                x_t_en,
            )?;
            let d_h_h1_t = weighted_crossprod_dense(
                &self.x_time_exit,
                &(-(&dh_h1 * &q.dq_t + &q.h_time_h1 * &delta_q_t_exit)),
                &x_threshold_exit,
            )?;
            assign_symmetric_block(&mut joint, offsets[0], offsets[1], &(d_h_h0_t + d_h_h1_t));
        } else {
            let delta_q_t = &delta_q_t_exit;
            let d_h_h0_t = weighted_crossprod_dense(
                &self.x_time_entry,
                &(-(&dh_h0 * &q.dq_t + &q.h_time_h0 * delta_q_t)),
                &x_threshold_exit,
            )?;
            let d_h_h1_t = weighted_crossprod_dense(
                &self.x_time_exit,
                &(-(&dh_h1 * &q.dq_t + &q.h_time_h1 * delta_q_t)),
                &x_threshold_exit,
            )?;
            assign_symmetric_block(&mut joint, offsets[0], offsets[1], &(d_h_h0_t + d_h_h1_t));
        }

        if let (Some(x_ls_en), Some(dq_ls_en)) =
            (x_log_sigma_entry.as_ref(), q.dq_ls_entry.as_ref())
        {
            let d_h_h0_l = weighted_crossprod_dense(
                &self.x_time_entry,
                &(-(&dh_h0 * dq_ls_en + &q.h_time_h0 * &entry_deltas.delta_q_ls)),
                x_ls_en,
            )?;
            let d_h_h1_l = weighted_crossprod_dense(
                &self.x_time_exit,
                &(-(&dh_h1 * &q.dq_ls + &q.h_time_h1 * &delta_q_ls_exit)),
                &x_log_sigma_exit,
            )?;
            assign_symmetric_block(&mut joint, offsets[0], offsets[2], &(d_h_h0_l + d_h_h1_l));
        } else {
            let delta_q_ls = &delta_q_ls_exit;
            let d_h_h0_l = weighted_crossprod_dense(
                &self.x_time_entry,
                &(-(&dh_h0 * &q.dq_ls + &q.h_time_h0 * delta_q_ls)),
                &x_log_sigma_exit,
            )?;
            let d_h_h1_l = weighted_crossprod_dense(
                &self.x_time_exit,
                &(-(&dh_h1 * &q.dq_ls + &q.h_time_h1 * delta_q_ls)),
                &x_log_sigma_exit,
            )?;
            assign_symmetric_block(&mut joint, offsets[0], offsets[2], &(d_h_h0_l + d_h_h1_l));
        }

        if let (Some(xw_dense), Some(w_offset)) = (xw, offsets.get(3).copied()) {
            let d_d2_q_combined = if x_threshold_entry.is_some() || x_log_sigma_entry.is_some() {
                &d_d2_q_exit + &entry_deltas.d_d2_q
            } else {
                &q.d3_q * &delta_q_exit + &q.d_h_h0 * &delta_h0 + &q.d_h_h1 * &delta_h1
            };
            if let (Some(x_t_en), Some(dq_t_en)) =
                (x_threshold_entry.as_ref(), q.dq_t_entry.as_ref())
            {
                let d_h_tw_exit = weighted_crossprod_dense(
                    &x_threshold_exit,
                    &(-(&d_d2_q_exit * &q.dq_t + &q.d2_q1 * &delta_q_t_exit)),
                    xw_dense,
                )?;
                let d_h_tw_entry = weighted_crossprod_dense(
                    x_t_en,
                    &(-(&entry_deltas.d_d2_q * dq_t_en + &q.d2_q0 * &entry_deltas.delta_q_t)),
                    xw_dense,
                )?;
                assign_symmetric_block(
                    &mut joint,
                    offsets[1],
                    w_offset,
                    &(d_h_tw_exit + d_h_tw_entry),
                );
            } else {
                let d_h_tw = weighted_crossprod_dense(
                    &x_threshold_exit,
                    &(-(&d_d2_q_combined * &q.dq_t + &q.d2_q * &delta_q_t_exit)),
                    xw_dense,
                )?;
                assign_symmetric_block(&mut joint, offsets[1], w_offset, &d_h_tw);
            }

            if let (Some(x_ls_en), Some(dq_ls_en)) =
                (x_log_sigma_entry.as_ref(), q.dq_ls_entry.as_ref())
            {
                let d_h_lw_exit = weighted_crossprod_dense(
                    &x_log_sigma_exit,
                    &(-(&d_d2_q_exit * &q.dq_ls + &q.d2_q1 * &delta_q_ls_exit)),
                    xw_dense,
                )?;
                let d_h_lw_entry = weighted_crossprod_dense(
                    x_ls_en,
                    &(-(&entry_deltas.d_d2_q * dq_ls_en + &q.d2_q0 * &entry_deltas.delta_q_ls)),
                    xw_dense,
                )?;
                assign_symmetric_block(
                    &mut joint,
                    offsets[2],
                    w_offset,
                    &(d_h_lw_exit + d_h_lw_entry),
                );
            } else {
                let d_h_lw = weighted_crossprod_dense(
                    &x_log_sigma_exit,
                    &(-(&d_d2_q_combined * &q.dq_ls + &q.d2_q * &delta_q_ls_exit)),
                    xw_dense,
                )?;
                assign_symmetric_block(&mut joint, offsets[2], w_offset, &d_h_lw);
            }

            let d_hww = weighted_crossprod_dense(xw_dense, &(-&d_d2_q_combined), xw_dense)?;
            assign_symmetric_block(&mut joint, w_offset, w_offset, &d_hww);

            let d_h_h0w = weighted_crossprod_dense(&self.x_time_entry, &(-&dh_h0), xw_dense)?;
            let d_h_h1w = weighted_crossprod_dense(&self.x_time_exit, &(-&dh_h1), xw_dense)?;
            assign_symmetric_block(&mut joint, offsets[0], w_offset, &(d_h_h0w + d_h_h1w));
        }

        Ok(Some(joint))
    }

    fn exact_newton_joint_psisecond_order_terms_from_parts(
        &self,
        block_states: &[ParameterBlockState],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        q: &SurvivalJointQuantities,
        dir_i: &SurvivalJointPsiDirection,
        dir_j: &SurvivalJointPsiDirection,
    ) -> Result<ExactNewtonJointPsiSecondOrderTerms, String> {
        let second_drifts = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            dir_i,
            dir_j,
        )?;

        let offsets = self.joint_block_offsets();
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;

        let x_threshold_exit_cow = self.x_threshold.as_dense_cow();
        let x_threshold_exit = &*x_threshold_exit_cow;
        let x_threshold_entry_cow = self
            .x_threshold_entry
            .as_ref()
            .map(DesignMatrix::as_dense_cow);
        let x_threshold_entry = x_threshold_entry_cow
            .as_ref()
            .map_or(x_threshold_exit, |c| &**c);
        let x_log_sigma_exit_cow = self.x_log_sigma.as_dense_cow();
        let x_log_sigma_exit = &*x_log_sigma_exit_cow;
        let x_log_sigma_entry_cow = self
            .x_log_sigma_entry
            .as_ref()
            .map(DesignMatrix::as_dense_cow);
        let x_log_sigma_entry = x_log_sigma_entry_cow
            .as_ref()
            .map_or(x_log_sigma_exit, |c| &**c);
        let xw_cow = self.x_link_wiggle.as_ref().map(DesignMatrix::as_dense_cow);
        let xw = xw_cow.as_ref().map(|c| &**c);
        let x_t_exit_i_map = first_psi_linear_map(
            dir_i.x_t_exit_action.as_ref(),
            dir_i.x_t_exit_psi.as_ref(),
            self.n,
            x_threshold_exit.ncols(),
        );
        let x_t_entry_i_map = first_psi_linear_map(
            dir_i.x_t_entry_action.as_ref(),
            dir_i.x_t_entry_psi.as_ref(),
            self.n,
            x_threshold_entry.ncols(),
        );
        let x_ls_exit_i_map = first_psi_linear_map(
            dir_i.x_ls_exit_action.as_ref(),
            dir_i.x_ls_exit_psi.as_ref(),
            self.n,
            x_log_sigma_exit.ncols(),
        );
        let x_ls_entry_i_map = first_psi_linear_map(
            dir_i.x_ls_entry_action.as_ref(),
            dir_i.x_ls_entry_psi.as_ref(),
            self.n,
            x_log_sigma_entry.ncols(),
        );
        let x_t_exit_j_map = first_psi_linear_map(
            dir_j.x_t_exit_action.as_ref(),
            dir_j.x_t_exit_psi.as_ref(),
            self.n,
            x_threshold_exit.ncols(),
        );
        let x_t_entry_j_map = first_psi_linear_map(
            dir_j.x_t_entry_action.as_ref(),
            dir_j.x_t_entry_psi.as_ref(),
            self.n,
            x_threshold_entry.ncols(),
        );
        let x_ls_exit_j_map = first_psi_linear_map(
            dir_j.x_ls_exit_action.as_ref(),
            dir_j.x_ls_exit_psi.as_ref(),
            self.n,
            x_log_sigma_exit.ncols(),
        );
        let x_ls_entry_j_map = first_psi_linear_map(
            dir_j.x_ls_entry_action.as_ref(),
            dir_j.x_ls_entry_psi.as_ref(),
            self.n,
            x_log_sigma_entry.ncols(),
        );
        let x_t_exit_ab_map = second_psi_linear_map(
            second_drifts.x_t_exit_ab_action.as_ref(),
            second_drifts.x_t_exit_ab.as_ref(),
            self.n,
            x_threshold_exit.ncols(),
        );
        let x_t_entry_ab_map = second_psi_linear_map(
            second_drifts.x_t_entry_ab_action.as_ref(),
            second_drifts.x_t_entry_ab.as_ref(),
            self.n,
            x_threshold_entry.ncols(),
        );
        let x_ls_exit_ab_map = second_psi_linear_map(
            second_drifts.x_ls_exit_ab_action.as_ref(),
            second_drifts.x_ls_exit_ab.as_ref(),
            self.n,
            x_log_sigma_exit.ncols(),
        );
        let x_ls_entry_ab_map = second_psi_linear_map(
            second_drifts.x_ls_entry_ab_action.as_ref(),
            second_drifts.x_ls_entry_ab.as_ref(),
            self.n,
            x_log_sigma_entry.ncols(),
        );

        let dq_t_entry = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
        let dq_ls_entry = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
        let d2q_tls_entry = q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls);
        let d2q_ls_entry = q.d2q_ls_entry.as_ref().unwrap_or(&q.d2q_ls);
        let d3q_tls_ls_entry = q.d3q_tls_ls_entry.as_ref().unwrap_or(&q.d3q_tls_ls);
        let d3q_ls_entry = q.d3q_ls_entry.as_ref().unwrap_or(&q.d3q_ls);

        let entry_cross = &(&dir_i.z_t_entry_psi * &dir_j.z_ls_entry_psi)
            + &(&dir_j.z_t_entry_psi * &dir_i.z_ls_entry_psi);
        let exit_cross = &(&dir_i.z_t_exit_psi * &dir_j.z_ls_exit_psi)
            + &(&dir_j.z_t_exit_psi * &dir_i.z_ls_exit_psi);

        let q0_i = &(&dir_i.z_t_entry_psi * dq_t_entry) + &(&dir_i.z_ls_entry_psi * dq_ls_entry);
        let q1_i = &(&dir_i.z_t_exit_psi * &q.dq_t) + &(&dir_i.z_ls_exit_psi * &q.dq_ls);
        let q0_j = &(&dir_j.z_t_entry_psi * dq_t_entry) + &(&dir_j.z_ls_entry_psi * dq_ls_entry);
        let q1_j = &(&dir_j.z_t_exit_psi * &q.dq_t) + &(&dir_j.z_ls_exit_psi * &q.dq_ls);

        let dq_t_entry_i = d2q_tls_entry * &dir_i.z_ls_entry_psi;
        let dq_t_exit_i = &q.d2q_tls * &dir_i.z_ls_exit_psi;
        let dq_t_entry_j = d2q_tls_entry * &dir_j.z_ls_entry_psi;
        let dq_t_exit_j = &q.d2q_tls * &dir_j.z_ls_exit_psi;

        let dq_ls_entry_i =
            d2q_tls_entry * &dir_i.z_t_entry_psi + d2q_ls_entry * &dir_i.z_ls_entry_psi;
        let dq_ls_exit_i = &q.d2q_tls * &dir_i.z_t_exit_psi + &q.d2q_ls * &dir_i.z_ls_exit_psi;
        let dq_ls_entry_j =
            d2q_tls_entry * &dir_j.z_t_entry_psi + d2q_ls_entry * &dir_j.z_ls_entry_psi;
        let dq_ls_exit_j = &q.d2q_tls * &dir_j.z_t_exit_psi + &q.d2q_ls * &dir_j.z_ls_exit_psi;

        let d2q_tls_entry_i = d3q_tls_ls_entry * &dir_i.z_ls_entry_psi;
        let d2q_tls_exit_i = &q.d3q_tls_ls * &dir_i.z_ls_exit_psi;
        let d2q_tls_entry_j = d3q_tls_ls_entry * &dir_j.z_ls_entry_psi;
        let d2q_tls_exit_j = &q.d3q_tls_ls * &dir_j.z_ls_exit_psi;

        let d2q_ls_entry_i =
            d3q_tls_ls_entry * &dir_i.z_t_entry_psi + d3q_ls_entry * &dir_i.z_ls_entry_psi;
        let d2q_ls_exit_i = &q.d3q_tls_ls * &dir_i.z_t_exit_psi + &q.d3q_ls * &dir_i.z_ls_exit_psi;
        let d2q_ls_entry_j =
            d3q_tls_ls_entry * &dir_j.z_t_entry_psi + d3q_ls_entry * &dir_j.z_ls_entry_psi;
        let d2q_ls_exit_j = &q.d3q_tls_ls * &dir_j.z_t_exit_psi + &q.d3q_ls * &dir_j.z_ls_exit_psi;

        let q0_ab = &(dq_t_entry * &second_drifts.z_t_entry_ab)
            + &(dq_ls_entry * &second_drifts.z_ls_entry_ab)
            + &(d2q_tls_entry * &entry_cross)
            + &(d2q_ls_entry * &(&dir_i.z_ls_entry_psi * &dir_j.z_ls_entry_psi));
        let q1_ab = &(&q.dq_t * &second_drifts.z_t_exit_ab)
            + &(&q.dq_ls * &second_drifts.z_ls_exit_ab)
            + &(&q.d2q_tls * &exit_cross)
            + &(&q.d2q_ls * &(&dir_i.z_ls_exit_psi * &dir_j.z_ls_exit_psi));

        let dq_t_entry_ab = &(d2q_tls_entry * &second_drifts.z_ls_entry_ab)
            + &(d3q_tls_ls_entry * &(&dir_i.z_ls_entry_psi * &dir_j.z_ls_entry_psi));
        let dq_t_exit_ab = &(&q.d2q_tls * &second_drifts.z_ls_exit_ab)
            + &(&q.d3q_tls_ls * &(&dir_i.z_ls_exit_psi * &dir_j.z_ls_exit_psi));

        let dq_ls_entry_ab = &(d2q_tls_entry * &second_drifts.z_t_entry_ab)
            + &(d2q_ls_entry * &second_drifts.z_ls_entry_ab)
            + &(d3q_tls_ls_entry * &entry_cross)
            + &(d3q_ls_entry * &(&dir_i.z_ls_entry_psi * &dir_j.z_ls_entry_psi));
        let dq_ls_exit_ab = &(&q.d2q_tls * &second_drifts.z_t_exit_ab)
            + &(&q.d2q_ls * &second_drifts.z_ls_exit_ab)
            + &(&q.d3q_tls_ls * &exit_cross)
            + &(&q.d3q_ls * &(&dir_i.z_ls_exit_psi * &dir_j.z_ls_exit_psi));

        let objective_psi_psi = (&q.d2_q0 * &(&q0_i * &q0_j)).sum()
            + q.d1_q0.dot(&q0_ab)
            + (&q.d2_q1 * &(&q1_i * &q1_j)).sum()
            + q.d1_q1.dot(&q1_ab);

        let mut score_psi_psi = Array1::<f64>::zeros(p_total);
        let time_score = self
            .x_time_entry
            .t()
            .dot(&(-(&q.d3_q0 * &(&q0_i * &q0_j) + &q.d2_q0 * &q0_ab)))
            + self
                .x_time_exit
                .t()
                .dot(&(-(&q.d3_q1 * &(&q1_i * &q1_j) + &q.d2_q1 * &q1_ab)));
        score_psi_psi
            .slice_mut(s![offsets[0]..offsets[1]])
            .assign(&time_score);

        let threshold_score_row_exit = &q.d1_q1 * &q.dq_t;
        let threshold_score_row_entry = &q.d1_q0 * dq_t_entry;
        let d_threshold_score_row_exit_i = &q.d2_q1 * &q1_i * &q.dq_t + &q.d1_q1 * &dq_t_exit_i;
        let d_threshold_score_row_entry_i =
            &q.d2_q0 * &q0_i * dq_t_entry + &q.d1_q0 * &dq_t_entry_i;
        let d_threshold_score_row_exit_j = &q.d2_q1 * &q1_j * &q.dq_t + &q.d1_q1 * &dq_t_exit_j;
        let d_threshold_score_row_entry_j =
            &q.d2_q0 * &q0_j * dq_t_entry + &q.d1_q0 * &dq_t_entry_j;
        let d2_threshold_score_row_exit = &(&q.d3_q1 * &(&q1_i * &q1_j) * &q.dq_t)
            + &(&q.d2_q1 * &q1_ab * &q.dq_t)
            + &(&q.d2_q1 * &(&q1_i * &dq_t_exit_j + &q1_j * &dq_t_exit_i))
            + &(&q.d1_q1 * dq_t_exit_ab);
        let d2_threshold_score_row_entry = &(&q.d3_q0 * &(&q0_i * &q0_j) * dq_t_entry)
            + &(&q.d2_q0 * &q0_ab * dq_t_entry)
            + &(&q.d2_q0 * &(&q0_i * &dq_t_entry_j + &q0_j * &dq_t_entry_i))
            + &(&q.d1_q0 * dq_t_entry_ab);
        let threshold_score = x_t_exit_ab_map.transpose_mul(threshold_score_row_exit.view())
            + x_t_exit_i_map.transpose_mul(d_threshold_score_row_exit_j.view())
            + x_t_exit_j_map.transpose_mul(d_threshold_score_row_exit_i.view())
            + x_threshold_exit.t().dot(&d2_threshold_score_row_exit)
            + x_t_entry_ab_map.transpose_mul(threshold_score_row_entry.view())
            + x_t_entry_i_map.transpose_mul(d_threshold_score_row_entry_j.view())
            + x_t_entry_j_map.transpose_mul(d_threshold_score_row_entry_i.view())
            + x_threshold_entry.t().dot(&d2_threshold_score_row_entry);
        score_psi_psi
            .slice_mut(s![offsets[1]..offsets[2]])
            .assign(&threshold_score);

        let log_sigma_score_row_exit = &q.d1_q1 * &q.dq_ls;
        let log_sigma_score_row_entry = &q.d1_q0 * dq_ls_entry;
        let d_log_sigma_score_row_exit_i = &q.d2_q1 * &q1_i * &q.dq_ls + &q.d1_q1 * &dq_ls_exit_i;
        let d_log_sigma_score_row_entry_i =
            &q.d2_q0 * &q0_i * dq_ls_entry + &q.d1_q0 * &dq_ls_entry_i;
        let d_log_sigma_score_row_exit_j = &q.d2_q1 * &q1_j * &q.dq_ls + &q.d1_q1 * &dq_ls_exit_j;
        let d_log_sigma_score_row_entry_j =
            &q.d2_q0 * &q0_j * dq_ls_entry + &q.d1_q0 * &dq_ls_entry_j;
        let d2_log_sigma_score_row_exit = &(&q.d3_q1 * &(&q1_i * &q1_j) * &q.dq_ls)
            + &(&q.d2_q1 * &q1_ab * &q.dq_ls)
            + &(&q.d2_q1 * &(&q1_i * &dq_ls_exit_j + &q1_j * &dq_ls_exit_i))
            + &(&q.d1_q1 * dq_ls_exit_ab);
        let d2_log_sigma_score_row_entry = &(&q.d3_q0 * &(&q0_i * &q0_j) * dq_ls_entry)
            + &(&q.d2_q0 * &q0_ab * dq_ls_entry)
            + &(&q.d2_q0 * &(&q0_i * &dq_ls_entry_j + &q0_j * &dq_ls_entry_i))
            + &(&q.d1_q0 * dq_ls_entry_ab);
        let log_sigma_score = x_ls_exit_ab_map.transpose_mul(log_sigma_score_row_exit.view())
            + x_ls_exit_i_map.transpose_mul(d_log_sigma_score_row_exit_j.view())
            + x_ls_exit_j_map.transpose_mul(d_log_sigma_score_row_exit_i.view())
            + x_log_sigma_exit.t().dot(&d2_log_sigma_score_row_exit)
            + x_ls_entry_ab_map.transpose_mul(log_sigma_score_row_entry.view())
            + x_ls_entry_i_map.transpose_mul(d_log_sigma_score_row_entry_j.view())
            + x_ls_entry_j_map.transpose_mul(d_log_sigma_score_row_entry_i.view())
            + x_log_sigma_entry.t().dot(&d2_log_sigma_score_row_entry);
        score_psi_psi
            .slice_mut(s![offsets[2]..offsets[3]])
            .assign(&log_sigma_score);

        if let (Some(xw_dense), Some(w_offset)) = (xw, offsets.get(3).copied()) {
            let wiggle_score = xw_dense.t().dot(
                &(&q.d3_q0 * &(&q0_i * &q0_j)
                    + &q.d2_q0 * &q0_ab
                    + &q.d3_q1 * &(&q1_i * &q1_j)
                    + &q.d2_q1 * &q1_ab),
            );
            score_psi_psi
                .slice_mut(s![w_offset..offsets[4]])
                .assign(&wiggle_score);
        }

        let mut hessian_psi_psi = Array2::<f64>::zeros((p_total, p_total));
        let h_time_time = safe_fast_xt_diag_x(
            &self.x_time_entry,
            &(-(&q.d3_q0 * &q0_ab) - &(&q.d2_h_h0 * &(&q0_i * &q0_j))),
        ) + safe_fast_xt_diag_x(
            &self.x_time_exit,
            &(-(&q.d3_q1 * &q1_ab) - &(&q.d2_h_h1 * &(&q1_i * &q1_j))),
        );
        assign_symmetric_block(&mut hessian_psi_psi, offsets[0], offsets[0], &h_time_time);

        let h_tt_entry = -(&q.d2_q0 * &dq_t_entry.mapv(|v| safe_product(v, v)));
        let h_tt_exit = -(&q.d2_q1 * &q.dq_t.mapv(|v| safe_product(v, v)));
        let dh_tt_entry_i = -(&q.d3_q0 * &q0_i * &dq_t_entry.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q0 * dq_t_entry * &dq_t_entry_i));
        let dh_tt_exit_i = -(&q.d3_q1 * &q1_i * &q.dq_t.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q1 * &q.dq_t * &dq_t_exit_i));
        let dh_tt_entry_j = -(&q.d3_q0 * &q0_j * &dq_t_entry.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q0 * dq_t_entry * &dq_t_entry_j));
        let dh_tt_exit_j = -(&q.d3_q1 * &q1_j * &q.dq_t.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q1 * &q.dq_t * &dq_t_exit_j));
        let h_threshold_threshold = weighted_crossprod_psi_maps(
            x_t_exit_ab_map,
            h_tt_exit.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
            h_tt_exit.view(),
            x_t_exit_ab_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_exit_i_map,
            h_tt_exit.view(),
            x_t_exit_j_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_exit_j_map,
            h_tt_exit.view(),
            x_t_exit_i_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_exit_i_map,
            dh_tt_exit_j.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
            dh_tt_exit_j.view(),
            x_t_exit_i_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_exit_j_map,
            dh_tt_exit_i.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
            dh_tt_exit_i.view(),
            x_t_exit_j_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_entry_ab_map,
            h_tt_entry.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
            h_tt_entry.view(),
            x_t_entry_ab_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_entry_i_map,
            h_tt_entry.view(),
            x_t_entry_j_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_entry_j_map,
            h_tt_entry.view(),
            x_t_entry_i_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_entry_i_map,
            dh_tt_entry_j.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
            dh_tt_entry_j.view(),
            x_t_entry_i_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_entry_j_map,
            dh_tt_entry_i.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
            dh_tt_entry_i.view(),
            x_t_entry_j_map,
        )?;
        assign_symmetric_block(
            &mut hessian_psi_psi,
            offsets[1],
            offsets[1],
            &h_threshold_threshold,
        );

        let h_ll_entry =
            -(&q.d2_q0 * &dq_ls_entry.mapv(|v| safe_product(v, v)) + &(&q.d1_q0 * d2q_ls_entry));
        let h_ll_exit =
            -(&q.d2_q1 * &q.dq_ls.mapv(|v| safe_product(v, v)) + &(&q.d1_q1 * &q.d2q_ls));
        let dh_ll_entry_i = -(&q.d3_q0 * &q0_i * &dq_ls_entry.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q0 * dq_ls_entry * &dq_ls_entry_i)
            + &(&q.d2_q0 * &q0_i * d2q_ls_entry)
            + &(&q.d1_q0 * &d2q_ls_entry_i));
        let dh_ll_exit_i = -(&q.d3_q1 * &q1_i * &q.dq_ls.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q1 * &q.dq_ls * &dq_ls_exit_i)
            + &(&q.d2_q1 * &q1_i * &q.d2q_ls)
            + &(&q.d1_q1 * &d2q_ls_exit_i));
        let dh_ll_entry_j = -(&q.d3_q0 * &q0_j * &dq_ls_entry.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q0 * dq_ls_entry * &dq_ls_entry_j)
            + &(&q.d2_q0 * &q0_j * d2q_ls_entry)
            + &(&q.d1_q0 * &d2q_ls_entry_j));
        let dh_ll_exit_j = -(&q.d3_q1 * &q1_j * &q.dq_ls.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q1 * &q.dq_ls * &dq_ls_exit_j)
            + &(&q.d2_q1 * &q1_j * &q.d2q_ls)
            + &(&q.d1_q1 * &d2q_ls_exit_j));
        let h_log_sigma_log_sigma = weighted_crossprod_psi_maps(
            x_ls_exit_ab_map,
            h_ll_exit.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
            h_ll_exit.view(),
            x_ls_exit_ab_map,
        )? + &weighted_crossprod_psi_maps(
            x_ls_exit_i_map,
            h_ll_exit.view(),
            x_ls_exit_j_map,
        )? + &weighted_crossprod_psi_maps(
            x_ls_exit_j_map,
            h_ll_exit.view(),
            x_ls_exit_i_map,
        )? + &weighted_crossprod_psi_maps(
            x_ls_exit_i_map,
            dh_ll_exit_j.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
            dh_ll_exit_j.view(),
            x_ls_exit_i_map,
        )? + &weighted_crossprod_psi_maps(
            x_ls_exit_j_map,
            dh_ll_exit_i.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
            dh_ll_exit_i.view(),
            x_ls_exit_j_map,
        )? + &weighted_crossprod_psi_maps(
            x_ls_entry_ab_map,
            h_ll_entry.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
            h_ll_entry.view(),
            x_ls_entry_ab_map,
        )? + &weighted_crossprod_psi_maps(
            x_ls_entry_i_map,
            h_ll_entry.view(),
            x_ls_entry_j_map,
        )? + &weighted_crossprod_psi_maps(
            x_ls_entry_j_map,
            h_ll_entry.view(),
            x_ls_entry_i_map,
        )? + &weighted_crossprod_psi_maps(
            x_ls_entry_i_map,
            dh_ll_entry_j.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
            dh_ll_entry_j.view(),
            x_ls_entry_i_map,
        )? + &weighted_crossprod_psi_maps(
            x_ls_entry_j_map,
            dh_ll_entry_i.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
            dh_ll_entry_i.view(),
            x_ls_entry_j_map,
        )?;
        assign_symmetric_block(
            &mut hessian_psi_psi,
            offsets[2],
            offsets[2],
            &h_log_sigma_log_sigma,
        );

        let h_tl_entry = -(&q.d2_q0 * &(dq_t_entry * dq_ls_entry) + &(&q.d1_q0 * d2q_tls_entry));
        let h_tl_exit = -(&q.d2_q1 * &(&q.dq_t * &q.dq_ls) + &(&q.d1_q1 * &q.d2q_tls));
        let dh_tl_entry_i = -(&q.d3_q0 * &q0_i * &(dq_t_entry * dq_ls_entry)
            + &(&q.d2_q0 * &(&dq_t_entry_i * dq_ls_entry + dq_t_entry * &dq_ls_entry_i))
            + &(&q.d2_q0 * &q0_i * d2q_tls_entry)
            + &(&q.d1_q0 * &d2q_tls_entry_i));
        let dh_tl_exit_i = -(&q.d3_q1 * &q1_i * &(&q.dq_t * &q.dq_ls)
            + &(&q.d2_q1 * &(&dq_t_exit_i * &q.dq_ls + &q.dq_t * &dq_ls_exit_i))
            + &(&q.d2_q1 * &q1_i * &q.d2q_tls)
            + &(&q.d1_q1 * &d2q_tls_exit_i));
        let dh_tl_entry_j = -(&q.d3_q0 * &q0_j * &(dq_t_entry * dq_ls_entry)
            + &(&q.d2_q0 * &(&dq_t_entry_j * dq_ls_entry + dq_t_entry * &dq_ls_entry_j))
            + &(&q.d2_q0 * &q0_j * d2q_tls_entry)
            + &(&q.d1_q0 * &d2q_tls_entry_j));
        let dh_tl_exit_j = -(&q.d3_q1 * &q1_j * &(&q.dq_t * &q.dq_ls)
            + &(&q.d2_q1 * &(&dq_t_exit_j * &q.dq_ls + &q.dq_t * &dq_ls_exit_j))
            + &(&q.d2_q1 * &q1_j * &q.d2q_tls)
            + &(&q.d1_q1 * &d2q_tls_exit_j));
        let h_threshold_log_sigma = weighted_crossprod_psi_maps(
            x_t_exit_ab_map,
            h_tl_exit.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
            h_tl_exit.view(),
            x_ls_exit_ab_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_exit_i_map,
            h_tl_exit.view(),
            x_ls_exit_j_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_exit_j_map,
            h_tl_exit.view(),
            x_ls_exit_i_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_exit_i_map,
            dh_tl_exit_j.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
            dh_tl_exit_j.view(),
            x_ls_exit_i_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_exit_j_map,
            dh_tl_exit_i.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
            dh_tl_exit_i.view(),
            x_ls_exit_j_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_entry_ab_map,
            h_tl_entry.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
            h_tl_entry.view(),
            x_ls_entry_ab_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_entry_i_map,
            h_tl_entry.view(),
            x_ls_entry_j_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_entry_j_map,
            h_tl_entry.view(),
            x_ls_entry_i_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_entry_i_map,
            dh_tl_entry_j.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
            dh_tl_entry_j.view(),
            x_ls_entry_i_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_entry_j_map,
            dh_tl_entry_i.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
            dh_tl_entry_i.view(),
            x_ls_entry_j_map,
        )?;
        assign_symmetric_block(
            &mut hessian_psi_psi,
            offsets[1],
            offsets[2],
            &h_threshold_log_sigma,
        );

        let h_h0_t = &q.d2_q0 * dq_t_entry;
        let h_h1_t = &q.d2_q1 * &q.dq_t;
        let dh_h0_t_i = &q.d3_q0 * &q0_i * dq_t_entry + &q.d2_q0 * &dq_t_entry_i;
        let dh_h1_t_i = &q.d3_q1 * &q1_i * &q.dq_t + &q.d2_q1 * &dq_t_exit_i;
        let dh_h0_t_j = &q.d3_q0 * &q0_j * dq_t_entry + &q.d2_q0 * &dq_t_entry_j;
        let dh_h1_t_j = &q.d3_q1 * &q1_j * &q.dq_t + &q.d2_q1 * &dq_t_exit_j;
        let h_time_threshold = weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
            dh_h0_t_j.view(),
            x_t_entry_i_map,
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
            dh_h0_t_i.view(),
            x_t_entry_j_map,
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
            h_h0_t.view(),
            x_t_entry_ab_map,
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
            dh_h1_t_j.view(),
            x_t_exit_i_map,
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
            dh_h1_t_i.view(),
            x_t_exit_j_map,
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
            h_h1_t.view(),
            x_t_exit_ab_map,
        )?;
        assign_symmetric_block(
            &mut hessian_psi_psi,
            offsets[0],
            offsets[1],
            &h_time_threshold,
        );

        let h_h0_ls = &q.d2_q0 * dq_ls_entry;
        let h_h1_ls = &q.d2_q1 * &q.dq_ls;
        let dh_h0_ls_i = &q.d3_q0 * &q0_i * dq_ls_entry + &q.d2_q0 * &dq_ls_entry_i;
        let dh_h1_ls_i = &q.d3_q1 * &q1_i * &q.dq_ls + &q.d2_q1 * &dq_ls_exit_i;
        let dh_h0_ls_j = &q.d3_q0 * &q0_j * dq_ls_entry + &q.d2_q0 * &dq_ls_entry_j;
        let dh_h1_ls_j = &q.d3_q1 * &q1_j * &q.dq_ls + &q.d2_q1 * &dq_ls_exit_j;
        let h_time_log_sigma = weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
            dh_h0_ls_j.view(),
            x_ls_entry_i_map,
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
            dh_h0_ls_i.view(),
            x_ls_entry_j_map,
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
            h_h0_ls.view(),
            x_ls_entry_ab_map,
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
            dh_h1_ls_j.view(),
            x_ls_exit_i_map,
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
            dh_h1_ls_i.view(),
            x_ls_exit_j_map,
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
            h_h1_ls.view(),
            x_ls_exit_ab_map,
        )?;
        assign_symmetric_block(
            &mut hessian_psi_psi,
            offsets[0],
            offsets[2],
            &h_time_log_sigma,
        );

        if let (Some(xw_dense), Some(w_offset)) = (xw, offsets.get(3).copied()) {
            let h_ww = -(&q.d3_q0 * &q0_ab + &q.d3_q1 * &q1_ab);
            let h_wiggle_wiggle = weighted_crossprod_dense(xw_dense, &h_ww, xw_dense)?;
            assign_symmetric_block(&mut hessian_psi_psi, w_offset, w_offset, &h_wiggle_wiggle);

            let h_tw_entry = -(&q.d2_q0 * dq_t_entry);
            let h_tw_exit = -(&q.d2_q1 * &q.dq_t);
            let dh_tw_entry_i = -(&q.d3_q0 * &q0_i * dq_t_entry + &q.d2_q0 * &dq_t_entry_i);
            let dh_tw_exit_i = -(&q.d3_q1 * &q1_i * &q.dq_t + &q.d2_q1 * &dq_t_exit_i);
            let dh_tw_entry_j = -(&q.d3_q0 * &q0_j * dq_t_entry + &q.d2_q0 * &dq_t_entry_j);
            let dh_tw_exit_j = -(&q.d3_q1 * &q1_j * &q.dq_t + &q.d2_q1 * &dq_t_exit_j);
            let h_threshold_wiggle = weighted_crossprod_psi_maps(
                x_t_exit_i_map,
                dh_tw_exit_j.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )? + &weighted_crossprod_psi_maps(
                x_t_exit_j_map,
                dh_tw_exit_i.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )? + &weighted_crossprod_psi_maps(
                x_t_exit_ab_map,
                h_tw_exit.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )? + &weighted_crossprod_psi_maps(
                x_t_entry_i_map,
                dh_tw_entry_j.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )? + &weighted_crossprod_psi_maps(
                x_t_entry_j_map,
                dh_tw_entry_i.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )? + &weighted_crossprod_psi_maps(
                x_t_entry_ab_map,
                h_tw_entry.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )?;
            assign_symmetric_block(
                &mut hessian_psi_psi,
                offsets[1],
                w_offset,
                &h_threshold_wiggle,
            );

            let h_lw_entry = -(&q.d2_q0 * dq_ls_entry);
            let h_lw_exit = -(&q.d2_q1 * &q.dq_ls);
            let dh_lw_entry_i = -(&q.d3_q0 * &q0_i * dq_ls_entry + &q.d2_q0 * &dq_ls_entry_i);
            let dh_lw_exit_i = -(&q.d3_q1 * &q1_i * &q.dq_ls + &q.d2_q1 * &dq_ls_exit_i);
            let dh_lw_entry_j = -(&q.d3_q0 * &q0_j * dq_ls_entry + &q.d2_q0 * &dq_ls_entry_j);
            let dh_lw_exit_j = -(&q.d3_q1 * &q1_j * &q.dq_ls + &q.d2_q1 * &dq_ls_exit_j);
            let h_log_sigma_wiggle = weighted_crossprod_psi_maps(
                x_ls_exit_i_map,
                dh_lw_exit_j.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )? + &weighted_crossprod_psi_maps(
                x_ls_exit_j_map,
                dh_lw_exit_i.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )? + &weighted_crossprod_psi_maps(
                x_ls_exit_ab_map,
                h_lw_exit.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )? + &weighted_crossprod_psi_maps(
                x_ls_entry_i_map,
                dh_lw_entry_j.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )? + &weighted_crossprod_psi_maps(
                x_ls_entry_j_map,
                dh_lw_entry_i.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )? + &weighted_crossprod_psi_maps(
                x_ls_entry_ab_map,
                h_lw_entry.view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )?;
            assign_symmetric_block(
                &mut hessian_psi_psi,
                offsets[2],
                w_offset,
                &h_log_sigma_wiggle,
            );

            let h_time_wiggle = weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
                (&q.d3_q0 * &q0_ab).view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )? + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
                (&q.d3_q1 * &q1_ab).view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )?;
            assign_symmetric_block(&mut hessian_psi_psi, offsets[0], w_offset, &h_time_wiggle);
        }

        Ok(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
            hessian_psi_psi_operator: None,
        })
    }

    fn exact_newton_joint_psihessian_directional_derivative_from_parts(
        &self,
        q: &SurvivalJointQuantities,
        dir: &SurvivalJointPsiDirection,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Array2<f64>, String> {
        let offsets = self.joint_block_offsets();
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_flat.len() != p_total {
            return Err(format!(
                "joint psi hessian directional derivative length mismatch: got {}, expected {p_total}",
                d_beta_flat.len()
            ));
        }

        let time_dir = d_beta_flat.slice(s![offsets[0]..offsets[1]]).to_owned();
        let threshold_dir = d_beta_flat.slice(s![offsets[1]..offsets[2]]).to_owned();
        let log_sigma_dir = d_beta_flat.slice(s![offsets[2]..offsets[3]]).to_owned();
        let wiggle_dir = if self.x_link_wiggle.is_some() {
            Some(d_beta_flat.slice(s![offsets[3]..offsets[4]]).to_owned())
        } else {
            None
        };

        let delta_h0 = self.x_time_entry.dot(&time_dir);
        let delta_h1 = self.x_time_exit.dot(&time_dir);
        let delta_t_exit = self.x_threshold.matrixvectormultiply(&threshold_dir);
        let delta_ls_exit = self.x_log_sigma.matrixvectormultiply(&log_sigma_dir);
        let deltaw = match (self.x_link_wiggle.as_ref(), wiggle_dir.as_ref()) {
            (Some(xw), Some(dir_w)) => xw.matrixvectormultiply(dir_w),
            _ => Array1::zeros(self.n),
        };

        let delta_q_exit = &q.dq_t * &delta_t_exit + &q.dq_ls * &delta_ls_exit + &deltaw;
        let delta_q_t_exit = &q.d2q_tls * &delta_ls_exit;
        let delta_q_ls_exit = &q.d2q_tls * &delta_t_exit + &q.d2q_ls * &delta_ls_exit;
        let delta_q_tls_exit = &q.d3q_tls_ls * &delta_ls_exit;
        let delta_q_ls_ls_exit = &q.d3q_tls_ls * &delta_t_exit + &q.d3q_ls * &delta_ls_exit;

        struct EntryDeltas {
            delta_q: Array1<f64>,
            delta_q_t: Array1<f64>,
            delta_q_ls: Array1<f64>,
            delta_q_tls: Array1<f64>,
            delta_q_ls_ls: Array1<f64>,
            d_d1_q: Array1<f64>,
            d_d2_q: Array1<f64>,
        }
        let entry_deltas = if self.x_threshold_entry.is_some() || self.x_log_sigma_entry.is_some() {
            let dt_en = self
                .x_threshold_entry
                .as_ref()
                .map(|x| x.matrixvectormultiply(&threshold_dir))
                .unwrap_or_else(|| delta_t_exit.clone());
            let dls_en = self
                .x_log_sigma_entry
                .as_ref()
                .map(|x| x.matrixvectormultiply(&log_sigma_dir))
                .unwrap_or_else(|| delta_ls_exit.clone());
            let dq_t_en = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
            let dq_ls_en = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
            let d2q_tls_en = q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls);
            let d3q_tls_ls_en = q.d3q_tls_ls_entry.as_ref().unwrap_or(&q.d3q_tls_ls);
            let d3q_ls_en = q.d3q_ls_entry.as_ref().unwrap_or(&q.d3q_ls);
            let d2q_ls_en = q.d2q_ls_entry.as_ref().unwrap_or(&q.d2q_ls);
            let dq_en = dq_t_en * &dt_en + dq_ls_en * &dls_en + &deltaw;
            EntryDeltas {
                delta_q_t: d2q_tls_en * &dls_en,
                delta_q_ls: d2q_tls_en * &dt_en + d2q_ls_en * &dls_en,
                delta_q_tls: d3q_tls_ls_en * &dls_en,
                delta_q_ls_ls: d3q_tls_ls_en * &dt_en + d3q_ls_en * &dls_en,
                d_d1_q: &q.d2_q0 * &dq_en + &q.h_time_h0 * &delta_h0,
                d_d2_q: &q.d3_q0 * &dq_en + &q.d_h_h0 * &delta_h0,
                delta_q: dq_en,
            }
        } else {
            EntryDeltas {
                delta_q: delta_q_exit.clone(),
                delta_q_t: delta_q_t_exit.clone(),
                delta_q_ls: delta_q_ls_exit.clone(),
                delta_q_tls: delta_q_tls_exit.clone(),
                delta_q_ls_ls: delta_q_ls_ls_exit.clone(),
                d_d1_q: &q.d2_q0 * &delta_q_exit + &q.h_time_h0 * &delta_h0,
                d_d2_q: &q.d3_q0 * &delta_q_exit + &q.d_h_h0 * &delta_h0,
            }
        };

        let x_threshold_exit_cow = self.x_threshold.as_dense_cow();
        let x_threshold_exit = &*x_threshold_exit_cow;
        let x_threshold_entry_cow = self
            .x_threshold_entry
            .as_ref()
            .map(DesignMatrix::as_dense_cow);
        let x_threshold_entry = x_threshold_entry_cow
            .as_ref()
            .map_or(x_threshold_exit, |c| &**c);
        let x_log_sigma_exit_cow = self.x_log_sigma.as_dense_cow();
        let x_log_sigma_exit = &*x_log_sigma_exit_cow;
        let x_log_sigma_entry_cow = self
            .x_log_sigma_entry
            .as_ref()
            .map(DesignMatrix::as_dense_cow);
        let x_log_sigma_entry = x_log_sigma_entry_cow
            .as_ref()
            .map_or(x_log_sigma_exit, |c| &**c);
        let xw_cow = self.x_link_wiggle.as_ref().map(DesignMatrix::as_dense_cow);
        let xw = xw_cow.as_ref().map(|c| &**c);
        let x_t_exit_map = first_psi_linear_map(
            dir.x_t_exit_action.as_ref(),
            dir.x_t_exit_psi.as_ref(),
            self.n,
            x_threshold_exit.ncols(),
        );
        let x_t_entry_map = first_psi_linear_map(
            dir.x_t_entry_action.as_ref(),
            dir.x_t_entry_psi.as_ref(),
            self.n,
            x_threshold_entry.ncols(),
        );
        let x_ls_exit_map = first_psi_linear_map(
            dir.x_ls_exit_action.as_ref(),
            dir.x_ls_exit_psi.as_ref(),
            self.n,
            x_log_sigma_exit.ncols(),
        );
        let x_ls_entry_map = first_psi_linear_map(
            dir.x_ls_entry_action.as_ref(),
            dir.x_ls_entry_psi.as_ref(),
            self.n,
            x_log_sigma_entry.ncols(),
        );

        let dq_t_entry = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
        let dq_ls_entry = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
        let d2q_ls_entry = q.d2q_ls_entry.as_ref().unwrap_or(&q.d2q_ls);

        let q0_psi = &(dq_t_entry * &dir.z_t_entry_psi) + &(dq_ls_entry * &dir.z_ls_entry_psi);
        let q1_psi = &(&q.dq_t * &dir.z_t_exit_psi) + &(&q.dq_ls * &dir.z_ls_exit_psi);
        let z_t_entry_psi_u = x_t_entry_map.forward_mul(threshold_dir.view());
        let z_t_exit_psi_u = x_t_exit_map.forward_mul(threshold_dir.view());
        let z_ls_entry_psi_u = x_ls_entry_map.forward_mul(log_sigma_dir.view());
        let z_ls_exit_psi_u = x_ls_exit_map.forward_mul(log_sigma_dir.view());
        let q0_psi_u = &(&entry_deltas.delta_q_t * &dir.z_t_entry_psi)
            + &(dq_t_entry * &z_t_entry_psi_u)
            + &(&entry_deltas.delta_q_ls * &dir.z_ls_entry_psi)
            + &(dq_ls_entry * &z_ls_entry_psi_u);
        let q1_psi_u = &(&delta_q_t_exit * &dir.z_t_exit_psi)
            + &(&q.dq_t * &z_t_exit_psi_u)
            + &(&delta_q_ls_exit * &dir.z_ls_exit_psi)
            + &(&q.dq_ls * &z_ls_exit_psi_u);
        let mut out = Array2::<f64>::zeros((p_total, p_total));

        let time_time = safe_fast_xt_diag_x(
            &self.x_time_entry,
            &(-(&q.d2_h_h0 * &entry_deltas.delta_q * q0_psi) - &(&q.d3_q0 * &q0_psi_u)),
        ) + safe_fast_xt_diag_x(
            &self.x_time_exit,
            &(-(&q.d2_h_h1 * &delta_q_exit * q1_psi) - &(&q.d3_q1 * &q1_psi_u)),
        );
        assign_symmetric_block(&mut out, offsets[0], offsets[0], &time_time);

        let h_tt_entry_u = -(&entry_deltas.d_d2_q * &dq_t_entry.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q0 * dq_t_entry * &entry_deltas.delta_q_t));
        let h_tt_exit_u = -(&(&q.d3_q1 * &delta_q_exit + &q.d_h_h1 * &delta_h1)
            * &q.dq_t.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q1 * &q.dq_t * &delta_q_t_exit));
        let threshold_threshold = weighted_crossprod_psi_maps(
            x_t_exit_map,
            h_tt_exit_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
            h_tt_exit_u.view(),
            x_t_exit_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_entry_map,
            h_tt_entry_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
            h_tt_entry_u.view(),
            x_t_entry_map,
        )?;
        assign_symmetric_block(&mut out, offsets[1], offsets[1], &threshold_threshold);

        let h_ll_entry_u = -(&entry_deltas.d_d2_q * &dq_ls_entry.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q0 * dq_ls_entry * &entry_deltas.delta_q_ls)
            + &(&entry_deltas.d_d1_q * d2q_ls_entry)
            + &(&q.d1_q0 * &entry_deltas.delta_q_ls_ls));
        let h_ll_exit_u = -(&(&q.d3_q1 * &delta_q_exit + &q.d_h_h1 * &delta_h1)
            * &q.dq_ls.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q1 * &q.dq_ls * &delta_q_ls_exit)
            + &((&q.d2_q1 * &delta_q_exit + &q.h_time_h1 * &delta_h1) * &q.d2q_ls)
            + &(&q.d1_q1 * &delta_q_ls_ls_exit));
        let log_sigma_log_sigma = weighted_crossprod_psi_maps(
            x_ls_exit_map,
            h_ll_exit_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
            h_ll_exit_u.view(),
            x_ls_exit_map,
        )? + &weighted_crossprod_psi_maps(
            x_ls_entry_map,
            h_ll_entry_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
            h_ll_entry_u.view(),
            x_ls_entry_map,
        )?;
        assign_symmetric_block(&mut out, offsets[2], offsets[2], &log_sigma_log_sigma);

        let h_tl_entry_u = -(&entry_deltas.d_d2_q * &(dq_t_entry * dq_ls_entry)
            + &(&q.d2_q0
                * &(&entry_deltas.delta_q_t * dq_ls_entry
                    + dq_t_entry * &entry_deltas.delta_q_ls))
            + &(&entry_deltas.d_d1_q * q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls))
            + &(&q.d1_q0 * &entry_deltas.delta_q_tls));
        let h_tl_exit_u = -(&(&q.d3_q1 * &delta_q_exit + &q.d_h_h1 * &delta_h1)
            * &(&q.dq_t * &q.dq_ls)
            + &(&q.d2_q1 * &(&delta_q_t_exit * &q.dq_ls + &q.dq_t * &delta_q_ls_exit))
            + &((&q.d2_q1 * &delta_q_exit + &q.h_time_h1 * &delta_h1) * &q.d2q_tls)
            + &(&q.d1_q1 * &delta_q_tls_exit));
        let threshold_log_sigma = weighted_crossprod_psi_maps(
            x_t_exit_map,
            h_tl_exit_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
            h_tl_exit_u.view(),
            x_ls_exit_map,
        )? + &weighted_crossprod_psi_maps(
            x_t_entry_map,
            h_tl_entry_u.view(),
            CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
            h_tl_entry_u.view(),
            x_ls_entry_map,
        )?;
        assign_symmetric_block(&mut out, offsets[1], offsets[2], &threshold_log_sigma);

        let h_h0_t_u = &entry_deltas.d_d1_q * dq_t_entry + &q.d2_q0 * &entry_deltas.delta_q_t;
        let h_h1_t_u = &(&q.d2_q1 * &delta_q_exit + &q.h_time_h1 * &delta_h1) * &q.dq_t
            + &q.d2_q1 * &delta_q_t_exit;
        let h_h0_ls_u = &entry_deltas.d_d1_q * dq_ls_entry + &q.d2_q0 * &entry_deltas.delta_q_ls;
        let h_h1_ls_u = &(&q.d2_q1 * &delta_q_exit + &q.h_time_h1 * &delta_h1) * &q.dq_ls
            + &q.d2_q1 * &delta_q_ls_exit;
        let time_threshold = weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
            h_h0_t_u.view(),
            x_t_entry_map,
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
            h_h1_t_u.view(),
            x_t_exit_map,
        )?;
        let time_log_sigma = weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
            h_h0_ls_u.view(),
            x_ls_entry_map,
        )? + &weighted_crossprod_psi_maps(
            CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
            h_h1_ls_u.view(),
            x_ls_exit_map,
        )?;
        assign_symmetric_block(&mut out, offsets[0], offsets[1], &time_threshold);
        assign_symmetric_block(&mut out, offsets[0], offsets[2], &time_log_sigma);

        if let (Some(xw_dense), Some(w_offset)) = (xw, offsets.get(3).copied()) {
            let d_d2_q_combined =
                if self.x_threshold_entry.is_some() || self.x_log_sigma_entry.is_some() {
                    &(&q.d3_q1 * &delta_q_exit + &q.d_h_h1 * &delta_h1) + &entry_deltas.d_d2_q
                } else {
                    &q.d3_q * &delta_q_exit + &q.d_h_h0 * &delta_h0 + &q.d_h_h1 * &delta_h1
                };
            let threshold_wiggle = weighted_crossprod_psi_maps(
                x_t_exit_map,
                (-(&d_d2_q_combined * &q.dq_t)).view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )? + &weighted_crossprod_psi_maps(
                x_t_entry_map,
                (-(&d_d2_q_combined * dq_t_entry)).view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )?;
            let log_sigma_wiggle = weighted_crossprod_psi_maps(
                x_ls_exit_map,
                (-(&d_d2_q_combined * &q.dq_ls)).view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )? + &weighted_crossprod_psi_maps(
                x_ls_entry_map,
                (-(&d_d2_q_combined * dq_ls_entry)).view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )?;
            let time_wiggle = weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
                (&q.d3_q0 * &q0_psi_u).view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )? + &weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
                (&q.d3_q1 * &q1_psi_u).view(),
                CustomFamilyPsiLinearMapRef::Dense(xw_dense),
            )?;
            assign_symmetric_block(&mut out, offsets[1], w_offset, &threshold_wiggle);
            assign_symmetric_block(&mut out, offsets[2], w_offset, &log_sigma_wiggle);
            assign_symmetric_block(&mut out, offsets[0], w_offset, &time_wiggle);
        }

        Ok(out)
    }
}

impl SurvivalExactNewtonJointPsiWorkspace {
    fn new(
        family: SurvivalLocationScaleFamily,
        block_states: Vec<ParameterBlockState>,
        derivative_blocks: Vec<Vec<CustomFamilyBlockPsiDerivative>>,
    ) -> Result<Self, String> {
        let joint_quantities = family.collect_joint_quantities(&block_states)?;
        let psi_dim = derivative_blocks.iter().map(Vec::len).sum();
        Ok(Self {
            family,
            block_states,
            derivative_blocks,
            joint_quantities,
            psi_directions: ExactNewtonJointPsiDirectCache::new(psi_dim),
        })
    }

    fn psi_direction(
        &self,
        psi_index: usize,
    ) -> Result<Option<Arc<SurvivalJointPsiDirection>>, String> {
        self.psi_directions.get_or_try_init(psi_index, || {
            self.family.exact_newton_joint_psi_direction(
                &self.block_states,
                &self.derivative_blocks,
                psi_index,
            )
        })
    }
}

impl ExactNewtonJointPsiWorkspace for SurvivalExactNewtonJointPsiWorkspace {
    fn second_order_terms(
        &self,
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        let Some(dir_i) = self.psi_direction(psi_i)? else {
            return Ok(None);
        };
        let Some(dir_j) = self.psi_direction(psi_j)? else {
            return Ok(None);
        };
        Ok(Some(
            self.family
                .exact_newton_joint_psisecond_order_terms_from_parts(
                    &self.block_states,
                    &self.derivative_blocks,
                    &self.joint_quantities,
                    dir_i.as_ref(),
                    dir_j.as_ref(),
                )?,
        ))
    }

    fn hessian_directional_derivative(
        &self,
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<crate::solver::estimate::reml::unified::DriftDerivResult>, String> {
        let Some(dir) = self.psi_direction(psi_index)? else {
            return Ok(None);
        };
        Ok(Some(
            crate::solver::estimate::reml::unified::DriftDerivResult::Dense(
                self.family
                    .exact_newton_joint_psihessian_directional_derivative_from_parts(
                        &self.joint_quantities,
                        dir.as_ref(),
                        d_beta_flat,
                    )?,
            ),
        ))
    }
}

/// Observed vs expected information: The survival location-scale family uses
/// `BlockWorkingSet::ExactNewton` which provides the actual gradient and Hessian
/// (-nabla^2 log L) from the survival likelihood. This is the **observed** Hessian
/// by construction, which is the correct quantity for the outer REML Laplace
/// approximation (see response.md Section 3). No Fisher surrogate is used here.
impl CustomFamily for SurvivalLocationScaleFamily {
    fn exact_newton_joint_hessian_beta_dependent(&self) -> bool {
        true
    }

    fn evaluate(&self, block_states: &[ParameterBlockState]) -> Result<FamilyEvaluation, String> {
        let n = self.n;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let mut ll = 0.0;

        // Per-row derivative collection — used for gradient only.
        let mut grad_time_eta_h0 = Array1::<f64>::zeros(n);
        let mut grad_time_eta_h1 = Array1::<f64>::zeros(n);
        let mut grad_time_eta_d = Array1::<f64>::zeros(n);
        let mut d1_q0 = Array1::<f64>::zeros(n);
        let mut d1_q1 = Array1::<f64>::zeros(n);
        let mut d1_qdot = Array1::<f64>::zeros(n);

        for i in 0..n {
            let state = self.row_predictor_state(
                dynamic.h_entry[i],
                dynamic.h_exit[i],
                dynamic.hdot_exit[i],
                dynamic.q_entry[i],
                dynamic.q_exit[i],
                dynamic.qdot_exit[i],
            );
            let Some(row) = self.row_derivatives(i, state)? else {
                continue;
            };
            ll += row.ll;
            d1_q0[i] = row.d1_q0;
            d1_q1[i] = row.d1_q1;
            d1_qdot[i] = row.d1_qdot1;
            grad_time_eta_h0[i] = row.grad_time_eta_h0;
            grad_time_eta_h1[i] = row.grad_time_eta_h1;
            grad_time_eta_d[i] = row.grad_time_eta_d;
        }

        // Per-block gradients (O(n·p_k) each, no Hessian algebra).
        let grad_time = dynamic.time_jac_entry.t().dot(&grad_time_eta_h0)
            + dynamic.time_jac_exit.t().dot(&grad_time_eta_h1)
            + dynamic.time_jac_deriv.t().dot(&grad_time_eta_d);

        let grad_t = if let (Some(x_t_entry), Some(x_t_deriv)) = (
            self.x_threshold_entry.as_ref(),
            self.x_threshold_deriv.as_ref(),
        ) {
            let grad_exit = &d1_q1 * &dynamic.dq_t_exit + &d1_qdot * &dynamic.dqdot_t;
            let grad_entry = &d1_q0 * &dynamic.dq_t_entry;
            let grad_deriv = &d1_qdot * &dynamic.dqdot_td;
            self.x_threshold.transpose_vector_multiply(&grad_exit)
                + x_t_entry.transpose_vector_multiply(&grad_entry)
                + x_t_deriv.transpose_vector_multiply(&grad_deriv)
        } else {
            self.x_threshold.transpose_vector_multiply(
                &(&d1_q1 * &dynamic.dq_t_exit
                    + &d1_q0 * &dynamic.dq_t_entry
                    + &d1_qdot * &dynamic.dqdot_t),
            )
        };

        let grad_ls = if let (Some(x_ls_entry), Some(x_ls_deriv)) = (
            self.x_log_sigma_entry.as_ref(),
            self.x_log_sigma_deriv.as_ref(),
        ) {
            let grad_exit = &d1_q1 * &dynamic.dq_ls_exit + &d1_qdot * &dynamic.dqdot_ls;
            let grad_entry = &d1_q0 * &dynamic.dq_ls_entry;
            let grad_deriv = &d1_qdot * &dynamic.dqdot_lsd;
            self.x_log_sigma.transpose_vector_multiply(&grad_exit)
                + x_ls_entry.transpose_vector_multiply(&grad_entry)
                + x_ls_deriv.transpose_vector_multiply(&grad_deriv)
        } else {
            self.x_log_sigma.transpose_vector_multiply(
                &(&d1_q1 * &dynamic.dq_ls_exit
                    + &d1_q0 * &dynamic.dq_ls_entry
                    + &d1_qdot * &dynamic.dqdot_ls),
            )
        };

        let mut block_gradients = vec![grad_time, grad_t, grad_ls];
        if let (Some(xw_exit), Some(xw_entry), Some(xw_qdot)) = (
            dynamic.wiggle_basis_exit.as_ref(),
            dynamic.wiggle_basis_entry.as_ref(),
            dynamic.wiggle_qdot_basis_exit.as_ref(),
        ) {
            let gradw =
                xw_exit.t().dot(&d1_q1) + xw_entry.t().dot(&d1_q0) + xw_qdot.t().dot(&d1_qdot);
            block_gradients.push(gradw);
        }

        // Block Hessians are principal diagonal blocks of the joint Hessian
        // (single source of truth — no separate blockwise Hessian derivation).
        let joint = self
            .exact_newton_joint_hessian(block_states)?
            .ok_or("SurvivalLocationScaleFamily: joint hessian unavailable")?;
        let offsets = self.joint_block_offsets();
        let block_ranges: Vec<std::ops::Range<usize>> = (0..block_gradients.len())
            .map(|k| offsets[k]..offsets[k + 1])
            .collect();
        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets: slice_joint_into_block_working_sets(
                block_gradients,
                &joint,
                &block_ranges,
            ),
        })
    }

    fn log_likelihood_only(&self, block_states: &[ParameterBlockState]) -> Result<f64, String> {
        // Fast path for backtracking line search: compute only the scalar
        // log-likelihood, skipping all gradient/Hessian/derivative assembly.
        let n = self.n;
        let dynamic = self.build_dynamic_geometry(block_states)?;

        let mut ll = 0.0;

        for i in 0..n {
            let state = self.row_predictor_state(
                dynamic.h_entry[i],
                dynamic.h_exit[i],
                dynamic.hdot_exit[i],
                dynamic.q_entry[i],
                dynamic.q_exit[i],
                dynamic.qdot_exit[i],
            );
            let Some(kernel) = self.exact_row_kernel(i, state)? else {
                continue;
            };
            ll += kernel.log_likelihood();
        }

        Ok(ll)
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let dims = self.joint_block_dims();
        if block_idx >= dims.len() {
            return Ok(None);
        }
        if d_beta.len() != dims[block_idx] {
            return Err(format!(
                "block {block_idx} d_beta length mismatch: got {}, expected {}",
                d_beta.len(),
                dims[block_idx]
            ));
        }
        let offsets = self.joint_block_offsets();
        let mut d_beta_flat = Array1::<f64>::zeros(*offsets.last().unwrap());
        d_beta_flat
            .slice_mut(s![offsets[block_idx]..offsets[block_idx + 1]])
            .assign(d_beta);
        // The block-level directional derivative must differentiate the
        // UNRESCALED Hessian (from exact_newton_joint_hessian / evaluate()),
        // not the rescaled one used in the outer curvature path.  Pass
        // log_rescale = 0 so quantities match what evaluate() returns.
        let d_joint = self
            .exact_newton_joint_hessian_directional_derivative_rescaled(
                block_states,
                &d_beta_flat,
                0.0,
            )?
            .ok_or_else(|| {
                "missing survival location-scale exact joint directional Hessian".to_string()
            })?;
        Ok(Some(
            d_joint
                .slice(s![
                    offsets[block_idx]..offsets[block_idx + 1],
                    offsets[block_idx]..offsets[block_idx + 1]
                ])
                .to_owned(),
        ))
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let q = self.collect_joint_quantities(block_states)?;
        self.assemble_joint_hessian_from_quantities(&q, block_states)
    }

    fn exact_newton_outer_curvature(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<ExactNewtonOuterCurvature>, String> {
        Ok(self
            .exact_newton_joint_hessian_rescaled(block_states)?
            .map(|(hessian, log_scale)| {
                let p = hessian.nrows();
                ExactNewtonOuterCurvature {
                    hessian,
                    rho_curvature_scale: (-log_scale).exp(),
                    hessian_logdet_correction: p as f64 * log_scale,
                }
            }))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        // The trait method uses the full rescale for the outer curvature path.
        self.exact_newton_joint_hessian_directional_derivative_rescaled(
            block_states,
            d_beta_flat,
            self.hessian_deriv_log_rescale(block_states),
        )
    }

    fn exact_newton_joint_psi_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
    ) -> Result<Option<ExactNewtonJointPsiTerms>, String> {
        if specs.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(format!(
                "SurvivalLocationScaleFamily joint psi terms expect {} specs and derivative blocks, got {} and {}",
                self.expected_blocks(),
                specs.len(),
                derivative_blocks.len()
            ));
        }
        let Some(dir) =
            self.exact_newton_joint_psi_direction(block_states, derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let z_t_exit_psi = &dir.z_t_exit_psi;
        let z_t_entry_psi = &dir.z_t_entry_psi;
        let z_ls_exit_psi = &dir.z_ls_exit_psi;
        let z_ls_entry_psi = &dir.z_ls_entry_psi;
        let q = self.collect_joint_quantities(block_states)?;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let offsets = self.joint_block_offsets();
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;

        let x_threshold_exit_cow = self.x_threshold.as_dense_cow();
        let x_threshold_exit = &*x_threshold_exit_cow;
        let x_threshold_entry_cow = self
            .x_threshold_entry
            .as_ref()
            .map(DesignMatrix::as_dense_cow);
        let x_threshold_entry = x_threshold_entry_cow
            .as_ref()
            .map_or(x_threshold_exit, |c| &**c);
        let x_log_sigma_exit_cow = self.x_log_sigma.as_dense_cow();
        let x_log_sigma_exit = &*x_log_sigma_exit_cow;
        let x_log_sigma_entry_cow = self
            .x_log_sigma_entry
            .as_ref()
            .map(DesignMatrix::as_dense_cow);
        let x_log_sigma_entry = x_log_sigma_entry_cow
            .as_ref()
            .map_or(x_log_sigma_exit, |c| &**c);
        let xw_cow = self.x_link_wiggle.as_ref().map(DesignMatrix::as_dense_cow);
        let xw = xw_cow.as_ref().map(|c| &**c);
        let x_t_exit_map = first_psi_linear_map(
            dir.x_t_exit_action.as_ref(),
            dir.x_t_exit_psi.as_ref(),
            self.n,
            x_threshold_exit.ncols(),
        );
        let x_t_entry_map = first_psi_linear_map(
            dir.x_t_entry_action.as_ref(),
            dir.x_t_entry_psi.as_ref(),
            self.n,
            x_threshold_entry.ncols(),
        );
        let x_ls_exit_map = first_psi_linear_map(
            dir.x_ls_exit_action.as_ref(),
            dir.x_ls_exit_psi.as_ref(),
            self.n,
            x_log_sigma_exit.ncols(),
        );
        let x_ls_entry_map = first_psi_linear_map(
            dir.x_ls_entry_action.as_ref(),
            dir.x_ls_entry_psi.as_ref(),
            self.n,
            x_log_sigma_entry.ncols(),
        );

        let dq_t_entry = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
        let dq_ls_entry = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
        let d2q_tls_entry = q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls);
        let d2q_ls_entry = q.d2q_ls_entry.as_ref().unwrap_or(&q.d2q_ls);
        let d3q_tls_ls_entry = q.d3q_tls_ls_entry.as_ref().unwrap_or(&q.d3q_tls_ls);
        let d3q_ls_entry = q.d3q_ls_entry.as_ref().unwrap_or(&q.d3q_ls);

        let q0_psi = &(dq_t_entry * z_t_entry_psi) + &(dq_ls_entry * z_ls_entry_psi);
        let q1_psi = &(&q.dq_t * z_t_exit_psi) + &(&q.dq_ls * z_ls_exit_psi);
        let dq_t_entry_psi = d2q_tls_entry * z_ls_entry_psi;
        let dq_t_exit_psi = &q.d2q_tls * z_ls_exit_psi;
        let dq_ls_entry_psi = d2q_tls_entry * z_t_entry_psi + d2q_ls_entry * z_ls_entry_psi;
        let dq_ls_exit_psi = &q.d2q_tls * z_t_exit_psi + &q.d2q_ls * z_ls_exit_psi;
        let d2q_tls_entry_psi = d3q_tls_ls_entry * z_ls_entry_psi;
        let d2q_tls_exit_psi = &q.d3q_tls_ls * z_ls_exit_psi;
        let d2q_ls_entry_psi = d3q_tls_ls_entry * z_t_entry_psi + d3q_ls_entry * z_ls_entry_psi;
        let d2q_ls_exit_psi = &q.d3q_tls_ls * z_t_exit_psi + &q.d3q_ls * z_ls_exit_psi;

        let objective_psi = q.d1_q0.dot(&q0_psi) + q.d1_q1.dot(&q1_psi);

        let mut score_psi = Array1::<f64>::zeros(p_total);
        let time_score = dynamic.time_jac_entry.t().dot(&(-&q.d2_q0 * &q0_psi))
            + dynamic.time_jac_exit.t().dot(&(-&q.d2_q1 * &q1_psi));
        score_psi
            .slice_mut(s![offsets[0]..offsets[1]])
            .assign(&time_score);

        let threshold_score_row_exit = &q.d1_q1 * &q.dq_t;
        let threshold_score_row_entry = &q.d1_q0 * dq_t_entry;
        let d_threshold_score_row_exit = &q.d2_q1 * &q1_psi * &q.dq_t + &q.d1_q1 * &dq_t_exit_psi;
        let d_threshold_score_row_entry =
            &q.d2_q0 * &q0_psi * dq_t_entry + &q.d1_q0 * &dq_t_entry_psi;
        let threshold_score = x_t_exit_map.transpose_mul(threshold_score_row_exit.view())
            + x_threshold_exit.t().dot(&d_threshold_score_row_exit)
            + x_t_entry_map.transpose_mul(threshold_score_row_entry.view())
            + x_threshold_entry.t().dot(&d_threshold_score_row_entry);
        score_psi
            .slice_mut(s![offsets[1]..offsets[2]])
            .assign(&threshold_score);

        let log_sigma_score_row_exit = &q.d1_q1 * &q.dq_ls;
        let log_sigma_score_row_entry = &q.d1_q0 * dq_ls_entry;
        let d_log_sigma_score_row_exit = &q.d2_q1 * &q1_psi * &q.dq_ls + &q.d1_q1 * &dq_ls_exit_psi;
        let d_log_sigma_score_row_entry =
            &q.d2_q0 * &q0_psi * dq_ls_entry + &q.d1_q0 * &dq_ls_entry_psi;
        let log_sigma_score = x_ls_exit_map.transpose_mul(log_sigma_score_row_exit.view())
            + x_log_sigma_exit.t().dot(&d_log_sigma_score_row_exit)
            + x_ls_entry_map.transpose_mul(log_sigma_score_row_entry.view())
            + x_log_sigma_entry.t().dot(&d_log_sigma_score_row_entry);
        score_psi
            .slice_mut(s![offsets[2]..offsets[3]])
            .assign(&log_sigma_score);

        if let (Some(xw_dense), Some(w_offset)) = (xw, offsets.get(3).copied()) {
            let wiggle_score = xw_dense.t().dot(&(&q.d2_q0 * &q0_psi + &q.d2_q1 * &q1_psi));
            score_psi
                .slice_mut(s![w_offset..offsets[4]])
                .assign(&wiggle_score);
        }

        let h_time_time = safe_fast_xt_diag_x(&dynamic.time_jac_entry, &(-&q.d3_q0 * &q0_psi))
            + safe_fast_xt_diag_x(&dynamic.time_jac_exit, &(-&q.d3_q1 * &q1_psi));

        let h_tt_entry = -(&q.d2_q0 * &dq_t_entry.mapv(|v| safe_product(v, v)));
        let h_tt_exit = -(&q.d2_q1 * &q.dq_t.mapv(|v| safe_product(v, v)));
        let dh_tt_entry = -(&q.d3_q0 * &q0_psi * &dq_t_entry.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q0 * dq_t_entry * &dq_t_entry_psi));
        let dh_tt_exit = -(&q.d3_q1 * &q1_psi * &q.dq_t.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q1 * &q.dq_t * &dq_t_exit_psi));

        let h_ll_entry =
            -(&q.d2_q0 * &dq_ls_entry.mapv(|v| safe_product(v, v)) + &(&q.d1_q0 * d2q_ls_entry));
        let h_ll_exit =
            -(&q.d2_q1 * &q.dq_ls.mapv(|v| safe_product(v, v)) + &(&q.d1_q1 * &q.d2q_ls));
        let dh_ll_entry = -(&q.d3_q0 * &q0_psi * &dq_ls_entry.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q0 * dq_ls_entry * &dq_ls_entry_psi)
            + &(&q.d2_q0 * &q0_psi * d2q_ls_entry)
            + &(&q.d1_q0 * &d2q_ls_entry_psi));
        let dh_ll_exit = -(&q.d3_q1 * &q1_psi * &q.dq_ls.mapv(|v| safe_product(v, v))
            + &(2.0 * &q.d2_q1 * &q.dq_ls * &dq_ls_exit_psi)
            + &(&q.d2_q1 * &q1_psi * &q.d2q_ls)
            + &(&q.d1_q1 * &d2q_ls_exit_psi));

        let h_tl_entry = -(&q.d2_q0 * &(dq_t_entry * dq_ls_entry) + &(&q.d1_q0 * d2q_tls_entry));
        let h_tl_exit = -(&q.d2_q1 * &(&q.dq_t * &q.dq_ls) + &(&q.d1_q1 * &q.d2q_tls));
        let dh_tl_entry = -(&q.d3_q0 * &q0_psi * &(dq_t_entry * dq_ls_entry)
            + &(&q.d2_q0 * &(&dq_t_entry_psi * dq_ls_entry + dq_t_entry * &dq_ls_entry_psi))
            + &(&q.d2_q0 * &q0_psi * d2q_tls_entry)
            + &(&q.d1_q0 * &d2q_tls_entry_psi));
        let dh_tl_exit = -(&q.d3_q1 * &q1_psi * &(&q.dq_t * &q.dq_ls)
            + &(&q.d2_q1 * &(&dq_t_exit_psi * &q.dq_ls + &q.dq_t * &dq_ls_exit_psi))
            + &(&q.d2_q1 * &q1_psi * &q.d2q_tls)
            + &(&q.d1_q1 * &d2q_tls_exit_psi));

        let h_h0_t = &q.d2_q0 * dq_t_entry;
        let h_h1_t = &q.d2_q1 * &q.dq_t;
        let dh_h0_t = &q.d3_q0 * &q0_psi * dq_t_entry + &q.d2_q0 * &dq_t_entry_psi;
        let dh_h1_t = &q.d3_q1 * &q1_psi * &q.dq_t + &q.d2_q1 * &dq_t_exit_psi;

        let h_h0_ls = &q.d2_q0 * dq_ls_entry;
        let h_h1_ls = &q.d2_q1 * &q.dq_ls;
        let dh_h0_ls = &q.d3_q0 * &q0_psi * dq_ls_entry + &q.d2_q0 * &dq_ls_entry_psi;
        let dh_h1_ls = &q.d3_q1 * &q1_psi * &q.dq_ls + &q.d2_q1 * &dq_ls_exit_psi;
        let h_tw_entry = -(&q.d2_q0 * dq_t_entry);
        let h_tw_exit = -(&q.d2_q1 * &q.dq_t);
        let dh_tw_entry = -(&q.d3_q0 * &q0_psi * dq_t_entry + &q.d2_q0 * &dq_t_entry_psi);
        let dh_tw_exit = -(&q.d3_q1 * &q1_psi * &q.dq_t + &q.d2_q1 * &dq_t_exit_psi);
        let h_lw_entry = -(&q.d2_q0 * dq_ls_entry);
        let h_lw_exit = -(&q.d2_q1 * &q.dq_ls);
        let dh_lw_entry = -(&q.d3_q0 * &q0_psi * dq_ls_entry + &q.d2_q0 * &dq_ls_entry_psi);
        let dh_lw_exit = -(&q.d3_q1 * &q1_psi * &q.dq_ls + &q.d2_q1 * &dq_ls_exit_psi);

        if dir.x_t_exit_action.is_some()
            || dir.x_t_entry_action.is_some()
            || dir.x_ls_exit_action.is_some()
            || dir.x_ls_entry_action.is_some()
        {
            let mut channels = vec![
                CustomFamilyJointDesignChannel::new(
                    offsets[0]..offsets[1],
                    shared_dense_arc(&self.x_time_entry),
                    None,
                ),
                CustomFamilyJointDesignChannel::new(
                    offsets[0]..offsets[1],
                    shared_dense_arc(&self.x_time_exit),
                    None,
                ),
                CustomFamilyJointDesignChannel::new(
                    offsets[1]..offsets[2],
                    shared_dense_arc(x_threshold_exit),
                    dir.x_t_exit_action.clone(),
                ),
                CustomFamilyJointDesignChannel::new(
                    offsets[1]..offsets[2],
                    shared_dense_arc(x_threshold_entry),
                    dir.x_t_entry_action.clone(),
                ),
                CustomFamilyJointDesignChannel::new(
                    offsets[2]..offsets[3],
                    shared_dense_arc(x_log_sigma_exit),
                    dir.x_ls_exit_action.clone(),
                ),
                CustomFamilyJointDesignChannel::new(
                    offsets[2]..offsets[3],
                    shared_dense_arc(x_log_sigma_entry),
                    dir.x_ls_entry_action.clone(),
                ),
            ];
            let mut pairs = vec![
                CustomFamilyJointDesignPairContribution::new(
                    0,
                    0,
                    Array1::zeros(self.x_time_entry.nrows()),
                    -&q.d3_q0 * &q0_psi,
                ),
                CustomFamilyJointDesignPairContribution::new(
                    1,
                    1,
                    Array1::zeros(self.x_time_exit.nrows()),
                    -&q.d3_q1 * &q1_psi,
                ),
                CustomFamilyJointDesignPairContribution::new(
                    2,
                    2,
                    h_tt_exit.clone(),
                    dh_tt_exit.clone(),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    3,
                    3,
                    h_tt_entry.clone(),
                    dh_tt_entry.clone(),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    4,
                    4,
                    h_ll_exit.clone(),
                    dh_ll_exit.clone(),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    5,
                    5,
                    h_ll_entry.clone(),
                    dh_ll_entry.clone(),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    2,
                    4,
                    h_tl_exit.clone(),
                    dh_tl_exit.clone(),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    4,
                    2,
                    h_tl_exit.clone(),
                    dh_tl_exit.clone(),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    3,
                    5,
                    h_tl_entry.clone(),
                    dh_tl_entry.clone(),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    5,
                    3,
                    h_tl_entry.clone(),
                    dh_tl_entry.clone(),
                ),
                CustomFamilyJointDesignPairContribution::new(0, 3, h_h0_t.clone(), dh_h0_t.clone()),
                CustomFamilyJointDesignPairContribution::new(3, 0, h_h0_t.clone(), dh_h0_t.clone()),
                CustomFamilyJointDesignPairContribution::new(1, 2, h_h1_t.clone(), dh_h1_t.clone()),
                CustomFamilyJointDesignPairContribution::new(2, 1, h_h1_t.clone(), dh_h1_t.clone()),
                CustomFamilyJointDesignPairContribution::new(
                    0,
                    5,
                    h_h0_ls.clone(),
                    dh_h0_ls.clone(),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    5,
                    0,
                    h_h0_ls.clone(),
                    dh_h0_ls.clone(),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    1,
                    4,
                    h_h1_ls.clone(),
                    dh_h1_ls.clone(),
                ),
                CustomFamilyJointDesignPairContribution::new(
                    4,
                    1,
                    h_h1_ls.clone(),
                    dh_h1_ls.clone(),
                ),
            ];
            if let (Some(xw_dense), Some(w_offset)) = (xw, offsets.get(3).copied()) {
                channels.push(CustomFamilyJointDesignChannel::new(
                    w_offset..offsets[4],
                    shared_dense_arc(xw_dense),
                    None,
                ));
                let w_idx = channels.len() - 1;
                let zero_w = Array1::zeros(xw_dense.nrows());
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    w_idx,
                    zero_w.clone(),
                    -&q.d3_q0 * &q0_psi - &q.d3_q1 * &q1_psi,
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    2,
                    w_idx,
                    h_tw_exit.clone(),
                    dh_tw_exit.clone(),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    2,
                    h_tw_exit.clone(),
                    dh_tw_exit.clone(),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    3,
                    w_idx,
                    h_tw_entry.clone(),
                    dh_tw_entry.clone(),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    3,
                    h_tw_entry.clone(),
                    dh_tw_entry.clone(),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    4,
                    w_idx,
                    h_lw_exit.clone(),
                    dh_lw_exit.clone(),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    4,
                    h_lw_exit.clone(),
                    dh_lw_exit.clone(),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    5,
                    w_idx,
                    h_lw_entry.clone(),
                    dh_lw_entry.clone(),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    5,
                    h_lw_entry.clone(),
                    dh_lw_entry.clone(),
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    0,
                    w_idx,
                    zero_w.clone(),
                    &q.d3_q0 * &q0_psi,
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    0,
                    zero_w.clone(),
                    &q.d3_q0 * &q0_psi,
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    1,
                    w_idx,
                    zero_w.clone(),
                    &q.d3_q1 * &q1_psi,
                ));
                pairs.push(CustomFamilyJointDesignPairContribution::new(
                    w_idx,
                    1,
                    zero_w,
                    &q.d3_q1 * &q1_psi,
                ));
            }
            return Ok(Some(ExactNewtonJointPsiTerms {
                objective_psi,
                score_psi,
                hessian_psi: Array2::zeros((0, 0)),
                hessian_psi_operator: Some(std::sync::Arc::new(CustomFamilyJointPsiOperator::new(
                    p_total, channels, pairs,
                ))),
            }));
        }
        let mut hessian_psi = Array2::<f64>::zeros((p_total, p_total));
        assign_symmetric_block(&mut hessian_psi, offsets[0], offsets[0], &h_time_time);
        let h_threshold_threshold =
            weighted_crossprod_psi_maps(
                x_t_exit_map,
                h_tt_exit.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
            )? + weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
                h_tt_exit.view(),
                x_t_exit_map,
            )? + weighted_crossprod_dense(&x_threshold_exit, &dh_tt_exit, &x_threshold_exit)?
                + weighted_crossprod_psi_maps(
                    x_t_entry_map,
                    h_tt_entry.view(),
                    CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
                )?
                + weighted_crossprod_psi_maps(
                    CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
                    h_tt_entry.view(),
                    x_t_entry_map,
                )?
                + weighted_crossprod_dense(x_threshold_entry, &dh_tt_entry, x_threshold_entry)?;
        assign_symmetric_block(
            &mut hessian_psi,
            offsets[1],
            offsets[1],
            &h_threshold_threshold,
        );
        let h_log_sigma_log_sigma =
            weighted_crossprod_psi_maps(
                x_ls_exit_map,
                h_ll_exit.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
            )? + weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
                h_ll_exit.view(),
                x_ls_exit_map,
            )? + weighted_crossprod_dense(&x_log_sigma_exit, &dh_ll_exit, &x_log_sigma_exit)?
                + weighted_crossprod_psi_maps(
                    x_ls_entry_map,
                    h_ll_entry.view(),
                    CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
                )?
                + weighted_crossprod_psi_maps(
                    CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
                    h_ll_entry.view(),
                    x_ls_entry_map,
                )?
                + weighted_crossprod_dense(x_log_sigma_entry, &dh_ll_entry, x_log_sigma_entry)?;
        assign_symmetric_block(
            &mut hessian_psi,
            offsets[2],
            offsets[2],
            &h_log_sigma_log_sigma,
        );
        let h_threshold_log_sigma =
            weighted_crossprod_psi_maps(
                x_t_exit_map,
                h_tl_exit.view(),
                CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_exit),
            )? + weighted_crossprod_psi_maps(
                CustomFamilyPsiLinearMapRef::Dense(x_threshold_exit),
                h_tl_exit.view(),
                x_ls_exit_map,
            )? + weighted_crossprod_dense(&x_threshold_exit, &dh_tl_exit, &x_log_sigma_exit)?
                + weighted_crossprod_psi_maps(
                    x_t_entry_map,
                    h_tl_entry.view(),
                    CustomFamilyPsiLinearMapRef::Dense(x_log_sigma_entry),
                )?
                + weighted_crossprod_psi_maps(
                    CustomFamilyPsiLinearMapRef::Dense(x_threshold_entry),
                    h_tl_entry.view(),
                    x_ls_entry_map,
                )?
                + weighted_crossprod_dense(x_threshold_entry, &dh_tl_entry, x_log_sigma_entry)?;
        assign_symmetric_block(
            &mut hessian_psi,
            offsets[1],
            offsets[2],
            &h_threshold_log_sigma,
        );
        let h_time_threshold =
            weighted_crossprod_dense(&self.x_time_entry, &dh_h0_t, x_threshold_entry)?
                + weighted_crossprod_psi_maps(
                    CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
                    h_h0_t.view(),
                    x_t_entry_map,
                )?
                + weighted_crossprod_dense(&self.x_time_exit, &dh_h1_t, &x_threshold_exit)?
                + weighted_crossprod_psi_maps(
                    CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
                    h_h1_t.view(),
                    x_t_exit_map,
                )?;
        assign_symmetric_block(&mut hessian_psi, offsets[0], offsets[1], &h_time_threshold);
        let h_time_log_sigma =
            weighted_crossprod_dense(&self.x_time_entry, &dh_h0_ls, x_log_sigma_entry)?
                + weighted_crossprod_psi_maps(
                    CustomFamilyPsiLinearMapRef::Dense(&self.x_time_entry),
                    h_h0_ls.view(),
                    x_ls_entry_map,
                )?
                + weighted_crossprod_dense(&self.x_time_exit, &dh_h1_ls, &x_log_sigma_exit)?
                + weighted_crossprod_psi_maps(
                    CustomFamilyPsiLinearMapRef::Dense(&self.x_time_exit),
                    h_h1_ls.view(),
                    x_ls_exit_map,
                )?;
        assign_symmetric_block(&mut hessian_psi, offsets[0], offsets[2], &h_time_log_sigma);

        if let (Some(xw_dense), Some(w_offset)) = (xw, offsets.get(3).copied()) {
            let h_ww = -(&q.d3_q0 * &q0_psi + &q.d3_q1 * &q1_psi);
            let h_wiggle_wiggle = weighted_crossprod_dense(xw_dense, &h_ww, xw_dense)?;
            assign_symmetric_block(&mut hessian_psi, w_offset, w_offset, &h_wiggle_wiggle);
            let h_threshold_wiggle =
                weighted_crossprod_psi_maps(
                    x_t_exit_map,
                    h_tw_exit.view(),
                    CustomFamilyPsiLinearMapRef::Dense(xw_dense),
                )? + weighted_crossprod_dense(&x_threshold_exit, &dh_tw_exit, xw_dense)?
                    + weighted_crossprod_psi_maps(
                        x_t_entry_map,
                        h_tw_entry.view(),
                        CustomFamilyPsiLinearMapRef::Dense(xw_dense),
                    )?
                    + weighted_crossprod_dense(x_threshold_entry, &dh_tw_entry, xw_dense)?;
            assign_symmetric_block(&mut hessian_psi, offsets[1], w_offset, &h_threshold_wiggle);
            let h_log_sigma_wiggle =
                weighted_crossprod_psi_maps(
                    x_ls_exit_map,
                    h_lw_exit.view(),
                    CustomFamilyPsiLinearMapRef::Dense(xw_dense),
                )? + weighted_crossprod_dense(&x_log_sigma_exit, &dh_lw_exit, xw_dense)?
                    + weighted_crossprod_psi_maps(
                        x_ls_entry_map,
                        h_lw_entry.view(),
                        CustomFamilyPsiLinearMapRef::Dense(xw_dense),
                    )?
                    + weighted_crossprod_dense(x_log_sigma_entry, &dh_lw_entry, xw_dense)?;
            assign_symmetric_block(&mut hessian_psi, offsets[2], w_offset, &h_log_sigma_wiggle);
            let h_time_wiggle =
                weighted_crossprod_dense(&self.x_time_entry, &(&q.d3_q0 * &q0_psi), xw_dense)?
                    + weighted_crossprod_dense(&self.x_time_exit, &(&q.d3_q1 * &q1_psi), xw_dense)?;
            assign_symmetric_block(&mut hessian_psi, offsets[0], w_offset, &h_time_wiggle);
        }

        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
            hessian_psi_operator: None,
        }))
    }

    fn exact_newton_joint_psisecond_order_terms(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_i: usize,
        psi_j: usize,
    ) -> Result<Option<ExactNewtonJointPsiSecondOrderTerms>, String> {
        if specs.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(format!(
                "SurvivalLocationScaleFamily joint psi second-order terms expect {} specs and derivative blocks, got {} and {}",
                self.expected_blocks(),
                specs.len(),
                derivative_blocks.len()
            ));
        }
        let Some(dir_i) =
            self.exact_newton_joint_psi_direction(block_states, derivative_blocks, psi_i)?
        else {
            return Ok(None);
        };
        let Some(dir_j) =
            self.exact_newton_joint_psi_direction(block_states, derivative_blocks, psi_j)?
        else {
            return Ok(None);
        };
        let q = self.collect_joint_quantities(block_states)?;
        Ok(Some(
            self.exact_newton_joint_psisecond_order_terms_from_parts(
                block_states,
                derivative_blocks,
                &q,
                &dir_i,
                &dir_j,
            )?,
        ))
    }

    fn exact_newton_joint_psi_workspace(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
    ) -> Result<Option<Arc<dyn ExactNewtonJointPsiWorkspace>>, String> {
        if block_states.len() != self.expected_blocks()
            || specs.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(format!(
                "SurvivalLocationScaleFamily joint psi workspace expects {} states, specs, and derivative blocks, got {} / {} / {}",
                self.expected_blocks(),
                block_states.len(),
                specs.len(),
                derivative_blocks.len()
            ));
        }
        Ok(Some(Arc::new(SurvivalExactNewtonJointPsiWorkspace::new(
            self.clone(),
            block_states.to_vec(),
            derivative_blocks.to_vec(),
        )?)))
    }

    fn exact_newton_joint_psihessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        specs: &[ParameterBlockSpec],
        derivative_blocks: &[Vec<CustomFamilyBlockPsiDerivative>],
        psi_index: usize,
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        if specs.len() != self.expected_blocks()
            || derivative_blocks.len() != self.expected_blocks()
        {
            return Err(format!(
                "SurvivalLocationScaleFamily joint psi hessian directional derivative expects {} specs and derivative blocks, got {} and {}",
                self.expected_blocks(),
                specs.len(),
                derivative_blocks.len()
            ));
        }
        let Some(dir) =
            self.exact_newton_joint_psi_direction(block_states, derivative_blocks, psi_index)?
        else {
            return Ok(None);
        };
        let q = self.collect_joint_quantities(block_states)?;
        Ok(Some(
            self.exact_newton_joint_psihessian_directional_derivative_from_parts(
                &q,
                &dir,
                d_beta_flat,
            )?,
        ))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let q = self.collect_joint_quantities_rescaled(
            block_states,
            self.hessian_deriv_log_rescale(block_states),
        )?;
        let dynamic = self.build_dynamic_geometry(block_states)?;
        let offsets = self.joint_block_offsets();
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        if d_beta_u_flat.len() != p_total || d_beta_v_flat.len() != p_total {
            return Err(format!(
                "joint d_beta length mismatch: got ({}, {}), expected {p_total}",
                d_beta_u_flat.len(),
                d_beta_v_flat.len()
            ));
        }

        // Split both directions into per-block slices.
        let time_dir_u = d_beta_u_flat.slice(s![offsets[0]..offsets[1]]).to_owned();
        let threshold_dir_u = d_beta_u_flat.slice(s![offsets[1]..offsets[2]]).to_owned();
        let log_sigma_dir_u = d_beta_u_flat.slice(s![offsets[2]..offsets[3]]).to_owned();
        let wiggle_dir_u = if self.x_link_wiggle.is_some() {
            Some(d_beta_u_flat.slice(s![offsets[3]..offsets[4]]).to_owned())
        } else {
            None
        };

        let time_dir_v = d_beta_v_flat.slice(s![offsets[0]..offsets[1]]).to_owned();
        let threshold_dir_v = d_beta_v_flat.slice(s![offsets[1]..offsets[2]]).to_owned();
        let log_sigma_dir_v = d_beta_v_flat.slice(s![offsets[2]..offsets[3]]).to_owned();
        let wiggle_dir_v = if self.x_link_wiggle.is_some() {
            Some(d_beta_v_flat.slice(s![offsets[3]..offsets[4]]).to_owned())
        } else {
            None
        };

        // -- Predictor-space deltas for direction u --
        let delta_h0_u = dynamic.time_jac_entry.dot(&time_dir_u);
        let delta_h1_u = dynamic.time_jac_exit.dot(&time_dir_u);
        let delta_d_u = dynamic.time_jac_deriv.dot(&time_dir_u);
        let delta_t_exit_u = self.x_threshold.matrixvectormultiply(&threshold_dir_u);
        let delta_ls_exit_u = self.x_log_sigma.matrixvectormultiply(&log_sigma_dir_u);
        let deltaw_u = match (self.x_link_wiggle.as_ref(), wiggle_dir_u.as_ref()) {
            (Some(xw), Some(dir)) => xw.matrixvectormultiply(dir),
            _ => Array1::zeros(self.n),
        };

        // -- Predictor-space deltas for direction v --
        let delta_h0_v = dynamic.time_jac_entry.dot(&time_dir_v);
        let delta_h1_v = dynamic.time_jac_exit.dot(&time_dir_v);
        let delta_d_v = dynamic.time_jac_deriv.dot(&time_dir_v);
        let delta_t_exit_v = self.x_threshold.matrixvectormultiply(&threshold_dir_v);
        let delta_ls_exit_v = self.x_log_sigma.matrixvectormultiply(&log_sigma_dir_v);
        let deltaw_v = match (self.x_link_wiggle.as_ref(), wiggle_dir_v.as_ref()) {
            (Some(xw), Some(dir)) => xw.matrixvectormultiply(dir),
            _ => Array1::zeros(self.n),
        };

        // Exit-side chain-rule deltas for u and v.
        let delta_q_exit_u = &q.dq_t * &delta_t_exit_u + &q.dq_ls * &delta_ls_exit_u + &deltaw_u;
        let delta_q_t_exit_u = &q.d2q_tls * &delta_ls_exit_u;
        let delta_q_ls_exit_u = &q.d2q_tls * &delta_t_exit_u + &q.d2q_ls * &delta_ls_exit_u;
        let delta_q_tls_exit_u = &q.d3q_tls_ls * &delta_ls_exit_u;
        let delta_q_ls_ls_exit_u = &q.d3q_tls_ls * &delta_t_exit_u + &q.d3q_ls * &delta_ls_exit_u;

        let delta_q_exit_v = &q.dq_t * &delta_t_exit_v + &q.dq_ls * &delta_ls_exit_v + &deltaw_v;
        let delta_q_t_exit_v = &q.d2q_tls * &delta_ls_exit_v;
        let delta_q_ls_exit_v = &q.d2q_tls * &delta_t_exit_v + &q.d2q_ls * &delta_ls_exit_v;
        let delta_q_tls_exit_v = &q.d3q_tls_ls * &delta_ls_exit_v;
        let delta_q_ls_ls_exit_v = &q.d3q_tls_ls * &delta_t_exit_v + &q.d3q_ls * &delta_ls_exit_v;

        // Perturbed curvature quantities for directions u and v on the exit side.
        let d_d1_q_exit_u = &q.d2_q1 * &delta_q_exit_u + &q.h_time_h1 * &delta_h1_u;
        let d_d2_q_exit_u = &q.d3_q1 * &delta_q_exit_u + &q.d_h_h1 * &delta_h1_u;
        let d_d1_q_exit_v = &q.d2_q1 * &delta_q_exit_v + &q.h_time_h1 * &delta_h1_v;
        let d_d2_q_exit_v = &q.d3_q1 * &delta_q_exit_v + &q.d_h_h1 * &delta_h1_v;

        let x_threshold_exit_cow = self.x_threshold.as_dense_cow();
        let x_threshold_exit = &*x_threshold_exit_cow;
        let x_threshold_entry_cow = self
            .x_threshold_entry
            .as_ref()
            .map(DesignMatrix::as_dense_cow);
        let x_threshold_entry = x_threshold_entry_cow.as_ref().map(|c| &**c);
        let x_log_sigma_exit_cow = self.x_log_sigma.as_dense_cow();
        let x_log_sigma_exit = &*x_log_sigma_exit_cow;
        let x_log_sigma_entry_cow = self
            .x_log_sigma_entry
            .as_ref()
            .map(DesignMatrix::as_dense_cow);
        let x_log_sigma_entry = x_log_sigma_entry_cow.as_ref().map(|c| &**c);
        let xw_cow = self.x_link_wiggle.as_ref().map(DesignMatrix::as_dense_cow);
        let xw = xw_cow.as_ref().map(|c| &**c);
        let mut joint = Array2::<f64>::zeros((p_total, p_total));

        // --- Entry-side deltas (analogous to first derivative) ---
        struct EntryDeltas2 {
            delta_t_u: Array1<f64>,
            delta_ls_u: Array1<f64>,
            delta_q_u: Array1<f64>,
            delta_q_t_u: Array1<f64>,
            delta_q_ls_u: Array1<f64>,
            delta_q_tls_u: Array1<f64>,
            delta_q_ls_ls_u: Array1<f64>,
            d_d1_q_u: Array1<f64>,
            d_d2_q_u: Array1<f64>,
            delta_t_v: Array1<f64>,
            delta_ls_v: Array1<f64>,
            delta_q_v: Array1<f64>,
            delta_q_t_v: Array1<f64>,
            delta_q_ls_v: Array1<f64>,
            delta_q_tls_v: Array1<f64>,
            delta_q_ls_ls_v: Array1<f64>,
            d_d1_q_v: Array1<f64>,
            d_d2_q_v: Array1<f64>,
        }

        let entry_deltas = if x_threshold_entry.is_some() || x_log_sigma_entry.is_some() {
            // Compute entry-side deltas for both u and v directions.
            let compute_entry = |threshold_dir: &Array1<f64>,
                                 log_sigma_dir: &Array1<f64>,
                                 deltaw: &Array1<f64>,
                                 delta_h0: &Array1<f64>|
             -> (
                Array1<f64>,
                Array1<f64>,
                Array1<f64>,
                Array1<f64>,
                Array1<f64>,
                Array1<f64>,
                Array1<f64>,
                Array1<f64>,
                Array1<f64>,
            ) {
                let dt_en = self
                    .x_threshold_entry
                    .as_ref()
                    .map(|x| x.matrixvectormultiply(threshold_dir))
                    .unwrap_or_else(|| self.x_threshold.matrixvectormultiply(threshold_dir));
                let dls_en = self
                    .x_log_sigma_entry
                    .as_ref()
                    .map(|x| x.matrixvectormultiply(log_sigma_dir))
                    .unwrap_or_else(|| self.x_log_sigma.matrixvectormultiply(log_sigma_dir));
                let dq_t_en = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
                let dq_ls_en = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
                let d2q_tls_en = q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls);
                let d3q_tls_ls_en = q.d3q_tls_ls_entry.as_ref().unwrap_or(&q.d3q_tls_ls);
                let d3q_ls_en = q.d3q_ls_entry.as_ref().unwrap_or(&q.d3q_ls);
                let d2q_ls_en = q.d2q_ls_entry.as_ref().unwrap_or(&q.d2q_ls);
                let dq_en = dq_t_en * &dt_en + dq_ls_en * &dls_en + deltaw;
                let dq_t = d2q_tls_en * &dls_en;
                let dq_ls = d2q_tls_en * &dt_en + d2q_ls_en * &dls_en;
                let dq_tls = d3q_tls_ls_en * &dls_en;
                let dq_ls_ls = d3q_tls_ls_en * &dt_en + d3q_ls_en * &dls_en;
                let d_d1_q = &q.d2_q0 * &dq_en + &q.h_time_h0 * delta_h0;
                let d_d2_q = &q.d3_q0 * &dq_en + &q.d_h_h0 * delta_h0;
                (
                    dt_en, dls_en, dq_en, dq_t, dq_ls, dq_tls, dq_ls_ls, d_d1_q, d_d2_q,
                )
            };
            let (dt_u, dls_u, dq_u, dqt_u, dqls_u, dqtls_u, dqlsls_u, dd1_u, dd2_u) =
                compute_entry(&threshold_dir_u, &log_sigma_dir_u, &deltaw_u, &delta_h0_u);
            let (dt_v, dls_v, dq_v, dqt_v, dqls_v, dqtls_v, dqlsls_v, dd1_v, dd2_v) =
                compute_entry(&threshold_dir_v, &log_sigma_dir_v, &deltaw_v, &delta_h0_v);
            EntryDeltas2 {
                delta_t_u: dt_u,
                delta_ls_u: dls_u,
                delta_q_u: dq_u,
                delta_q_t_u: dqt_u,
                delta_q_ls_u: dqls_u,
                delta_q_tls_u: dqtls_u,
                delta_q_ls_ls_u: dqlsls_u,
                d_d1_q_u: dd1_u,
                d_d2_q_u: dd2_u,
                delta_t_v: dt_v,
                delta_ls_v: dls_v,
                delta_q_v: dq_v,
                delta_q_t_v: dqt_v,
                delta_q_ls_v: dqls_v,
                delta_q_tls_v: dqtls_v,
                delta_q_ls_ls_v: dqlsls_v,
                d_d1_q_v: dd1_v,
                d_d2_q_v: dd2_v,
            }
        } else {
            // Time-invariant: entry deltas = exit deltas.
            EntryDeltas2 {
                delta_t_u: delta_t_exit_u.clone(),
                delta_ls_u: delta_ls_exit_u.clone(),
                delta_q_u: delta_q_exit_u.clone(),
                delta_q_t_u: delta_q_t_exit_u.clone(),
                delta_q_ls_u: delta_q_ls_exit_u.clone(),
                delta_q_tls_u: delta_q_tls_exit_u.clone(),
                delta_q_ls_ls_u: delta_q_ls_ls_exit_u.clone(),
                d_d1_q_u: &q.d2_q0 * &delta_q_exit_u + &q.h_time_h0 * &delta_h0_u,
                d_d2_q_u: &q.d3_q0 * &delta_q_exit_u + &q.d_h_h0 * &delta_h0_u,
                delta_t_v: delta_t_exit_v.clone(),
                delta_ls_v: delta_ls_exit_v.clone(),
                delta_q_v: delta_q_exit_v.clone(),
                delta_q_t_v: delta_q_t_exit_v.clone(),
                delta_q_ls_v: delta_q_ls_exit_v.clone(),
                delta_q_tls_v: delta_q_tls_exit_v.clone(),
                delta_q_ls_ls_v: delta_q_ls_ls_exit_v.clone(),
                d_d1_q_v: &q.d2_q0 * &delta_q_exit_v + &q.h_time_h0 * &delta_h0_v,
                d_d2_q_v: &q.d3_q0 * &delta_q_exit_v + &q.d_h_h0 * &delta_h0_v,
            }
        };

        // === Second-order perturbation weights ===
        //
        // For D²H[u,v], we differentiate D_u H w.r.t. v. The key second-order
        // weights for each observation are products of the two perturbation
        // directions multiplied by the appropriate curvature derivative.
        //
        // The pattern: for each Hessian weight w(β), the first derivative is
        //   D_u w = w' · δ_u,
        // and the second derivative is
        //   D²_{u,v} w = w'' · δ_u · δ_v + w' · δ²_{u,v}
        // where δ²_{u,v} captures cross-terms from the chain rule on δ_u itself.
        //
        // For the time block, the Hessian has three contributions:
        //   h_time_h0[i], h_time_h1[i], h_time_d[i]
        // Their first derivatives w.r.t. β are d_h_h0, d_h_h1, d_h_d.
        // The second derivatives use the 4th-order row quantities d2_h_h0
        // and d2_h_h1 (d⁴ℓ/dh⁴), which are now stored in
        // SurvivalJointQuantities.
        //
        // The 4th derivatives of ℓ w.r.t. q (d4_q0, d4_q1, d4_q) are also
        // now available. They enter the bilinear D²(d2_q) computation as the
        // leading coefficient: D²_{u,v}(d2_q) = d4_q · δq_u · δq_v + ...
        //
        // The 4th-order chain-rule derivatives of q (d4q_tls_ls_ls = u_ϑsss,
        // d4q_ls = u_ssss) enter the bilinear derivatives of the 2nd-order
        // chain quantities: D²_{ψ}(d2q_tls) and D²_{ψ}(d2q_ls).
        //
        // Together these provide the complete 4th-order Arbogast formula
        // for the (ϑ,s,s,s) and (s,s,s,s) blocks of the outer Hessian drift
        // Q[v_k, v_l]. See response.md Section 6.
        //
        // --- Time block D²H[u,v] ---
        let xi_h0_u = &delta_h0_u + &entry_deltas.delta_q_u;
        let xi_h1_u = &delta_h1_u + &delta_q_exit_u;
        let xi_h0_v = &delta_h0_v + &entry_deltas.delta_q_v;
        let xi_h1_v = &delta_h1_v + &delta_q_exit_v;
        let d2q_tls_entry = q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls);
        let d2q_ls_entry = q.d2q_ls_entry.as_ref().unwrap_or(&q.d2q_ls);
        let d3q_tls_ls_entry = q.d3q_tls_ls_entry.as_ref().unwrap_or(&q.d3q_tls_ls);
        let d3q_ls_entry = q.d3q_ls_entry.as_ref().unwrap_or(&q.d3q_ls);
        let d4q_tls_ls_ls_entry = q.d4q_tls_ls_ls_entry.as_ref().unwrap_or(&q.d4q_tls_ls_ls);
        let d4q_ls_entry = q.d4q_ls_entry.as_ref().unwrap_or(&q.d4q_ls);

        let delta_q_uv_exit = &(&q.d2q_tls
            * &(&delta_t_exit_u * &delta_ls_exit_v + &delta_t_exit_v * &delta_ls_exit_u))
            + &(&q.d2q_ls * &(&delta_ls_exit_u * &delta_ls_exit_v));
        let delta_q_t_uv_exit = &q.d3q_tls_ls * &(&delta_ls_exit_u * &delta_ls_exit_v);
        let delta_q_ls_uv_exit = &(&q.d3q_tls_ls
            * &(&delta_ls_exit_u * &delta_t_exit_v + &delta_ls_exit_v * &delta_t_exit_u))
            + &(&q.d3q_ls * &(&delta_ls_exit_u * &delta_ls_exit_v));
        let delta_q_tls_uv_exit = &q.d4q_tls_ls_ls * &(&delta_ls_exit_u * &delta_ls_exit_v);
        let delta_q_ls_ls_uv_exit = &(&q.d4q_tls_ls_ls
            * &(&delta_ls_exit_u * &delta_t_exit_v + &delta_ls_exit_v * &delta_t_exit_u))
            + &(&q.d4q_ls * &(&delta_ls_exit_u * &delta_ls_exit_v));

        let delta_q_uv_entry = &(d2q_tls_entry
            * &(&entry_deltas.delta_t_u * &entry_deltas.delta_ls_v
                + &entry_deltas.delta_t_v * &entry_deltas.delta_ls_u))
            + &(d2q_ls_entry * &(&entry_deltas.delta_ls_u * &entry_deltas.delta_ls_v));
        let delta_q_t_uv_entry =
            d3q_tls_ls_entry * &(&entry_deltas.delta_ls_u * &entry_deltas.delta_ls_v);
        let delta_q_ls_uv_entry = &(d3q_tls_ls_entry
            * &(&entry_deltas.delta_ls_u * &entry_deltas.delta_t_v
                + &entry_deltas.delta_ls_v * &entry_deltas.delta_t_u))
            + &(d3q_ls_entry * &(&entry_deltas.delta_ls_u * &entry_deltas.delta_ls_v));
        let delta_q_tls_uv_entry =
            d4q_tls_ls_ls_entry * &(&entry_deltas.delta_ls_u * &entry_deltas.delta_ls_v);
        let delta_q_ls_ls_uv_entry = &(d4q_tls_ls_ls_entry
            * &(&entry_deltas.delta_ls_u * &entry_deltas.delta_t_v
                + &entry_deltas.delta_ls_v * &entry_deltas.delta_t_u))
            + &(d4q_ls_entry * &(&entry_deltas.delta_ls_u * &entry_deltas.delta_ls_v));

        let d_d1_q_combined_u =
            &q.d2_q * &delta_q_exit_u + &q.h_time_h0 * &delta_h0_u + &q.h_time_h1 * &delta_h1_u;
        let d_d1_q_combined_v =
            &q.d2_q * &delta_q_exit_v + &q.h_time_h0 * &delta_h0_v + &q.h_time_h1 * &delta_h1_v;
        let d_d2_q_combined_u =
            &q.d3_q * &delta_q_exit_u + &q.d_h_h0 * &delta_h0_u + &q.d_h_h1 * &delta_h1_u;
        let d_d2_q_combined_v =
            &q.d3_q * &delta_q_exit_v + &q.d_h_h0 * &delta_h0_v + &q.d_h_h1 * &delta_h1_v;

        let d2_d1_q_entry_exact = &q.d3_q0 * &(&xi_h0_u * &xi_h0_v) + &q.d2_q0 * &delta_q_uv_entry;
        let d2_d1_q_exit_exact = &q.d3_q1 * &(&xi_h1_u * &xi_h1_v) + &q.d2_q1 * &delta_q_uv_exit;
        let d2_d1_q_combined_exact = &d2_d1_q_entry_exact + &d2_d1_q_exit_exact;
        let d2_d2_q_entry_exact = &q.d4_q0 * &(&xi_h0_u * &xi_h0_v) + &q.d3_q0 * &delta_q_uv_entry;
        let d2_d2_q_exit_exact = &q.d4_q1 * &(&xi_h1_u * &xi_h1_v) + &q.d3_q1 * &delta_q_uv_exit;
        let d2_d2_q_combined_exact = &d2_d2_q_entry_exact + &d2_d2_q_exit_exact;

        // Second-order time-time weight: bilinear in perturbation directions.
        //
        // The exact D²H[u,v] for the time block uses the 4th derivative of ℓ
        // w.r.t. the time predictors (d2_h_h0, d2_h_h1). Previously this used
        // d_h_h0 (= d³ℓ/dh0³) which is the coefficient for D¹H, not D²H.
        // The correct bilinear coefficient is d⁴ℓ/dh0⁴ for the leading term.
        // See response.md Section 6 for why 4th-order derivatives are needed.
        let d2h_h0 = &q.d2_h_h0 * &(&xi_h0_u * &xi_h0_v);
        let d2h_h1 = &q.d2_h_h1 * &(&xi_h1_u * &xi_h1_v);
        let d2h_d = &q.d_h_d * &(&delta_d_u * &delta_d_v);
        let d2_h_time = safe_fast_xt_diag_x(&dynamic.time_jac_entry, &d2h_h0)
            + safe_fast_xt_diag_x(&dynamic.time_jac_exit, &d2h_h1)
            + safe_fast_xt_diag_x(&dynamic.time_jac_deriv, &d2h_d);
        assign_symmetric_block(&mut joint, offsets[0], offsets[0], &d2_h_time);

        // --- Threshold-threshold D²H[u,v] ---
        if let Some(x_t_en) = x_threshold_entry.as_ref() {
            let dq_t_en = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
            let d2_w_exit = &d2_d2_q_exit_exact * &q.dq_t.mapv(|v| safe_product(v, v))
                + &(2.0 * &d_d2_q_exit_u * &q.dq_t * &delta_q_t_exit_v)
                + &(2.0 * &d_d2_q_exit_v * &q.dq_t * &delta_q_t_exit_u)
                + &(2.0 * &q.d2_q1 * &delta_q_t_exit_u * &delta_q_t_exit_v)
                + &(2.0 * &q.d2_q1 * &q.dq_t * &delta_q_t_uv_exit);
            let d2_w_entry = &d2_d2_q_entry_exact * &dq_t_en.mapv(|v| safe_product(v, v))
                + &(2.0 * &entry_deltas.d_d2_q_u * dq_t_en * &entry_deltas.delta_q_t_v)
                + &(2.0 * &entry_deltas.d_d2_q_v * dq_t_en * &entry_deltas.delta_q_t_u)
                + &(2.0 * &q.d2_q0 * &entry_deltas.delta_q_t_u * &entry_deltas.delta_q_t_v)
                + &(2.0 * &q.d2_q0 * dq_t_en * &delta_q_t_uv_entry);
            let d2_h_tt =
                weighted_crossprod_dense(&x_threshold_exit, &(-&d2_w_exit), &x_threshold_exit)?
                    + weighted_crossprod_dense(x_t_en, &(-&d2_w_entry), x_t_en)?;
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &d2_h_tt);
        } else {
            let d2_w = &d2_d2_q_combined_exact * &q.dq_t.mapv(|v| safe_product(v, v))
                + &(2.0 * &d_d2_q_combined_u * &q.dq_t * &delta_q_t_exit_v)
                + &(2.0 * &d_d2_q_combined_v * &q.dq_t * &delta_q_t_exit_u)
                + &(2.0 * &q.d2_q * &delta_q_t_exit_u * &delta_q_t_exit_v)
                + &(2.0 * &q.d2_q * &q.dq_t * &delta_q_t_uv_exit);
            let d2_h_tt =
                weighted_crossprod_dense(&x_threshold_exit, &(-&d2_w), &x_threshold_exit)?;
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &d2_h_tt);
        }

        // --- Log-sigma-log-sigma D²H[u,v] ---
        if let Some(x_ls_en) = x_log_sigma_entry.as_ref() {
            let dq_ls_en = q.dq_ls_entry.as_ref().unwrap();
            let d2_w_exit = &d2_d2_q_exit_exact * &q.dq_ls.mapv(|v| safe_product(v, v))
                + &(2.0 * &d_d2_q_exit_u * &q.dq_ls * &delta_q_ls_exit_v)
                + &(2.0 * &d_d2_q_exit_v * &q.dq_ls * &delta_q_ls_exit_u)
                + &(2.0 * &q.d2_q1 * &delta_q_ls_exit_u * &delta_q_ls_exit_v)
                + &(2.0 * &q.d2_q1 * &q.dq_ls * &delta_q_ls_uv_exit)
                + &d2_d1_q_exit_exact * &q.d2q_ls
                + &d_d1_q_exit_u * &delta_q_ls_ls_exit_v
                + &d_d1_q_exit_v * &delta_q_ls_ls_exit_u
                + &(&q.d1_q1 * &delta_q_ls_ls_uv_exit);
            let d2_w_entry = &d2_d2_q_entry_exact * &dq_ls_en.mapv(|v| safe_product(v, v))
                + &(2.0 * &entry_deltas.d_d2_q_u * dq_ls_en * &entry_deltas.delta_q_ls_v)
                + &(2.0 * &entry_deltas.d_d2_q_v * dq_ls_en * &entry_deltas.delta_q_ls_u)
                + &(2.0 * &q.d2_q0 * &entry_deltas.delta_q_ls_u * &entry_deltas.delta_q_ls_v)
                + &(2.0 * &q.d2_q0 * dq_ls_en * &delta_q_ls_uv_entry)
                + &d2_d1_q_entry_exact * d2q_ls_entry
                + &entry_deltas.d_d1_q_u * &entry_deltas.delta_q_ls_ls_v
                + &entry_deltas.d_d1_q_v * &entry_deltas.delta_q_ls_ls_u
                + &(&q.d1_q0 * &delta_q_ls_ls_uv_entry);
            let d2_h_ll =
                weighted_crossprod_dense(&x_log_sigma_exit, &(-&d2_w_exit), &x_log_sigma_exit)?
                    + weighted_crossprod_dense(x_ls_en, &(-&d2_w_entry), x_ls_en)?;
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &d2_h_ll);
        } else {
            let d2_w = &d2_d2_q_combined_exact * &q.dq_ls.mapv(|v| safe_product(v, v))
                + &(2.0 * &d_d2_q_combined_u * &q.dq_ls * &delta_q_ls_exit_v)
                + &(2.0 * &d_d2_q_combined_v * &q.dq_ls * &delta_q_ls_exit_u)
                + &(2.0 * &q.d2_q * &delta_q_ls_exit_u * &delta_q_ls_exit_v)
                + &(2.0 * &q.d2_q * &q.dq_ls * &delta_q_ls_uv_exit)
                + &d2_d1_q_combined_exact * &q.d2q_ls
                + &d_d1_q_combined_u * &delta_q_ls_ls_exit_v
                + &d_d1_q_combined_v * &delta_q_ls_ls_exit_u
                + &(&q.d1_q * &delta_q_ls_ls_uv_exit);
            let d2_h_ll =
                weighted_crossprod_dense(&x_log_sigma_exit, &(-&d2_w), &x_log_sigma_exit)?;
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &d2_h_ll);
        }

        // --- Threshold-log-sigma cross D²H[u,v] ---
        {
            let has_t_entry = x_threshold_entry.is_some();
            let has_ls_entry = x_log_sigma_entry.is_some();
            if has_t_entry || has_ls_entry {
                let x_t_en = x_threshold_entry.unwrap_or(x_threshold_exit);
                let x_ls_en = x_log_sigma_entry.unwrap_or(x_log_sigma_exit);
                let dq_t_en = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
                let dq_ls_en = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
                let d2_w_exit = &d2_d2_q_exit_exact * &(&q.dq_t * &q.dq_ls)
                    + &d_d2_q_exit_u
                        * &(&delta_q_t_exit_v * &q.dq_ls + &q.dq_t * &delta_q_ls_exit_v)
                    + &d_d2_q_exit_v
                        * &(&delta_q_t_exit_u * &q.dq_ls + &q.dq_t * &delta_q_ls_exit_u)
                    + &q.d2_q1
                        * &(&delta_q_t_uv_exit * &q.dq_ls
                            + &delta_q_t_exit_u * &delta_q_ls_exit_v
                            + &delta_q_t_exit_v * &delta_q_ls_exit_u
                            + &q.dq_t * &delta_q_ls_uv_exit)
                    + &d2_d1_q_exit_exact * &q.d2q_tls
                    + &d_d1_q_exit_u * &delta_q_tls_exit_v
                    + &d_d1_q_exit_v * &delta_q_tls_exit_u
                    + &(&q.d1_q1 * &delta_q_tls_uv_exit);
                let d2_w_entry = &d2_d2_q_entry_exact * &(dq_t_en * dq_ls_en)
                    + &entry_deltas.d_d2_q_u
                        * &(&entry_deltas.delta_q_t_v * dq_ls_en
                            + dq_t_en * &entry_deltas.delta_q_ls_v)
                    + &entry_deltas.d_d2_q_v
                        * &(&entry_deltas.delta_q_t_u * dq_ls_en
                            + dq_t_en * &entry_deltas.delta_q_ls_u)
                    + &q.d2_q0
                        * &(&delta_q_t_uv_entry * dq_ls_en
                            + &entry_deltas.delta_q_t_u * &entry_deltas.delta_q_ls_v
                            + &entry_deltas.delta_q_t_v * &entry_deltas.delta_q_ls_u
                            + dq_t_en * &delta_q_ls_uv_entry)
                    + &d2_d1_q_entry_exact * d2q_tls_entry
                    + &entry_deltas.d_d1_q_u * &entry_deltas.delta_q_tls_v
                    + &entry_deltas.d_d1_q_v * &entry_deltas.delta_q_tls_u
                    + &(&q.d1_q0 * &delta_q_tls_uv_entry);
                let d2_h_tl =
                    weighted_crossprod_dense(&x_threshold_exit, &(-&d2_w_exit), &x_log_sigma_exit)?
                        + weighted_crossprod_dense(x_t_en, &(-&d2_w_entry), x_ls_en)?;
                assign_symmetric_block(&mut joint, offsets[1], offsets[2], &d2_h_tl);
            } else {
                let d2_w = &d2_d2_q_combined_exact * &(&q.dq_t * &q.dq_ls)
                    + &d_d2_q_combined_u
                        * &(&delta_q_t_exit_v * &q.dq_ls + &q.dq_t * &delta_q_ls_exit_v)
                    + &d_d2_q_combined_v
                        * &(&delta_q_t_exit_u * &q.dq_ls + &q.dq_t * &delta_q_ls_exit_u)
                    + &q.d2_q
                        * &(&delta_q_t_uv_exit * &q.dq_ls
                            + &delta_q_t_exit_u * &delta_q_ls_exit_v
                            + &delta_q_t_exit_v * &delta_q_ls_exit_u
                            + &q.dq_t * &delta_q_ls_uv_exit)
                    + &d2_d1_q_combined_exact * &q.d2q_tls
                    + &d_d1_q_combined_u * &delta_q_tls_exit_v
                    + &d_d1_q_combined_v * &delta_q_tls_exit_u
                    + &(&q.d1_q * &delta_q_tls_uv_exit);
                let d2_h_tl =
                    weighted_crossprod_dense(&x_threshold_exit, &(-&d2_w), &x_log_sigma_exit)?;
                assign_symmetric_block(&mut joint, offsets[1], offsets[2], &d2_h_tl);
            }
        }

        // --- Time-threshold cross D²H[u,v] ---
        {
            let dh_h0_u = &q.d_h_h0 * &(&delta_h0_u + &entry_deltas.delta_q_u);
            let dh_h1_u = &q.d_h_h1 * &(&delta_h1_u + &delta_q_exit_u);
            let dh_h0_v = &q.d_h_h0 * &(&delta_h0_v + &entry_deltas.delta_q_v);
            let dh_h1_v = &q.d_h_h1 * &(&delta_h1_v + &delta_q_exit_v);
            if let (Some(x_t_en), Some(_)) = (x_threshold_entry.as_ref(), q.dq_t_entry.as_ref()) {
                let d2_w_exit = &dh_h1_u * &delta_q_t_exit_v
                    + &dh_h1_v * &delta_q_t_exit_u
                    + &q.h_time_h1 * &(&delta_q_t_exit_u * &xi_h1_v + &delta_q_t_exit_v * &xi_h1_u);
                let d2_w_entry = &dh_h0_u * &entry_deltas.delta_q_t_v
                    + &dh_h0_v * &entry_deltas.delta_q_t_u
                    + &q.h_time_h0
                        * &(&entry_deltas.delta_q_t_u * &xi_h0_v
                            + &entry_deltas.delta_q_t_v * &xi_h0_u);
                let d2_h_ht_exit =
                    weighted_crossprod_dense(&self.x_time_exit, &(-&d2_w_exit), &x_threshold_exit)?;
                let d2_h_ht_entry =
                    weighted_crossprod_dense(&self.x_time_entry, &(-&d2_w_entry), x_t_en)?;
                assign_symmetric_block(
                    &mut joint,
                    offsets[0],
                    offsets[1],
                    &(d2_h_ht_exit + d2_h_ht_entry),
                );
            } else {
                // The combined weight d2_w = h0 + h1 contributions is split across
                // the two design matrices (x_time_entry, x_time_exit) below.
                let d2_h_ht_0 = weighted_crossprod_dense(
                    &self.x_time_entry,
                    &(-&(&dh_h0_u * &delta_q_t_exit_v
                        + &dh_h0_v * &delta_q_t_exit_u
                        + &q.h_time_h0
                            * &(&delta_q_t_exit_u * &xi_h0_v + &delta_q_t_exit_v * &xi_h0_u))),
                    &x_threshold_exit,
                )?;
                let d2_h_ht_1 = weighted_crossprod_dense(
                    &self.x_time_exit,
                    &(-&(&dh_h1_u * &delta_q_t_exit_v
                        + &dh_h1_v * &delta_q_t_exit_u
                        + &q.h_time_h1
                            * &(&delta_q_t_exit_u * &xi_h1_v + &delta_q_t_exit_v * &xi_h1_u))),
                    &x_threshold_exit,
                )?;
                assign_symmetric_block(
                    &mut joint,
                    offsets[0],
                    offsets[1],
                    &(d2_h_ht_0 + d2_h_ht_1),
                );
            }
        }

        // --- Time-log-sigma cross D²H[u,v] ---
        {
            let dh_h0_u = &q.d_h_h0 * &(&delta_h0_u + &entry_deltas.delta_q_u);
            let dh_h1_u = &q.d_h_h1 * &(&delta_h1_u + &delta_q_exit_u);
            let dh_h0_v = &q.d_h_h0 * &(&delta_h0_v + &entry_deltas.delta_q_v);
            let dh_h1_v = &q.d_h_h1 * &(&delta_h1_v + &delta_q_exit_v);
            if let (Some(x_ls_en), Some(_)) = (x_log_sigma_entry.as_ref(), q.dq_ls_entry.as_ref()) {
                let d2_w_exit = &dh_h1_u * &delta_q_ls_exit_v
                    + &dh_h1_v * &delta_q_ls_exit_u
                    + &q.h_time_h1
                        * &(&delta_q_ls_exit_u * &xi_h1_v + &delta_q_ls_exit_v * &xi_h1_u);
                let d2_w_entry = &dh_h0_u * &entry_deltas.delta_q_ls_v
                    + &dh_h0_v * &entry_deltas.delta_q_ls_u
                    + &q.h_time_h0
                        * &(&entry_deltas.delta_q_ls_u * &xi_h0_v
                            + &entry_deltas.delta_q_ls_v * &xi_h0_u);
                let d2_h_hl_exit =
                    weighted_crossprod_dense(&self.x_time_exit, &(-&d2_w_exit), &x_log_sigma_exit)?;
                let d2_h_hl_entry =
                    weighted_crossprod_dense(&self.x_time_entry, &(-&d2_w_entry), x_ls_en)?;
                assign_symmetric_block(
                    &mut joint,
                    offsets[0],
                    offsets[2],
                    &(d2_h_hl_exit + d2_h_hl_entry),
                );
            } else {
                let d2_h_hl_0 = weighted_crossprod_dense(
                    &self.x_time_entry,
                    &(-&(&dh_h0_u * &delta_q_ls_exit_v
                        + &dh_h0_v * &delta_q_ls_exit_u
                        + &q.h_time_h0
                            * &(&delta_q_ls_exit_u * &xi_h0_v + &delta_q_ls_exit_v * &xi_h0_u))),
                    &x_log_sigma_exit,
                )?;
                let d2_h_hl_1 = weighted_crossprod_dense(
                    &self.x_time_exit,
                    &(-&(&dh_h1_u * &delta_q_ls_exit_v
                        + &dh_h1_v * &delta_q_ls_exit_u
                        + &q.h_time_h1
                            * &(&delta_q_ls_exit_u * &xi_h1_v + &delta_q_ls_exit_v * &xi_h1_u))),
                    &x_log_sigma_exit,
                )?;
                assign_symmetric_block(
                    &mut joint,
                    offsets[0],
                    offsets[2],
                    &(d2_h_hl_0 + d2_h_hl_1),
                );
            }
        }

        // --- Wiggle cross-blocks D²H[u,v] ---
        if let (Some(xw_dense), Some(w_offset)) = (xw, offsets.get(3).copied()) {
            let d2_d2_q_combined = d2_d2_q_combined_exact.clone();

            // Threshold-wiggle D²H[u,v].
            if let (Some(x_t_en), Some(dq_t_en)) =
                (x_threshold_entry.as_ref(), q.dq_t_entry.as_ref())
            {
                let d2_tw_exit = &d2_d2_q_exit_exact * &q.dq_t
                    + &q.d2_q1 * &(&delta_q_t_exit_u * &deltaw_v + &delta_q_t_exit_v * &deltaw_u);
                let d2_tw_entry = &d2_d2_q_entry_exact * dq_t_en
                    + &q.d2_q0
                        * &(&entry_deltas.delta_q_t_u * &deltaw_v
                            + &entry_deltas.delta_q_t_v * &deltaw_u);
                let d2_h_tw =
                    weighted_crossprod_dense(&x_threshold_exit, &(-&d2_tw_exit), xw_dense)?
                        + weighted_crossprod_dense(x_t_en, &(-&d2_tw_entry), xw_dense)?;
                assign_symmetric_block(&mut joint, offsets[1], w_offset, &d2_h_tw);
            } else {
                let d2_tw = &d2_d2_q_combined * &q.dq_t
                    + &q.d2_q * &(&delta_q_t_exit_u * &deltaw_v + &delta_q_t_exit_v * &deltaw_u);
                let d2_h_tw = weighted_crossprod_dense(&x_threshold_exit, &(-&d2_tw), xw_dense)?;
                assign_symmetric_block(&mut joint, offsets[1], w_offset, &d2_h_tw);
            }

            // Log-sigma-wiggle D²H[u,v].
            if let (Some(x_ls_en), Some(dq_ls_en)) =
                (x_log_sigma_entry.as_ref(), q.dq_ls_entry.as_ref())
            {
                let d2_lw_exit = &d2_d2_q_exit_exact * &q.dq_ls
                    + &q.d2_q1 * &(&delta_q_ls_exit_u * &deltaw_v + &delta_q_ls_exit_v * &deltaw_u);
                let d2_lw_entry = &d2_d2_q_entry_exact * dq_ls_en
                    + &q.d2_q0
                        * &(&entry_deltas.delta_q_ls_u * &deltaw_v
                            + &entry_deltas.delta_q_ls_v * &deltaw_u);
                let d2_h_lw =
                    weighted_crossprod_dense(&x_log_sigma_exit, &(-&d2_lw_exit), xw_dense)?
                        + weighted_crossprod_dense(x_ls_en, &(-&d2_lw_entry), xw_dense)?;
                assign_symmetric_block(&mut joint, offsets[2], w_offset, &d2_h_lw);
            } else {
                let d2_lw = &d2_d2_q_combined * &q.dq_ls
                    + &q.d2_q * &(&delta_q_ls_exit_u * &deltaw_v + &delta_q_ls_exit_v * &deltaw_u);
                let d2_h_lw = weighted_crossprod_dense(&x_log_sigma_exit, &(-&d2_lw), xw_dense)?;
                assign_symmetric_block(&mut joint, offsets[2], w_offset, &d2_h_lw);
            }

            // Wiggle-wiggle D²H[u,v].
            let d2_hww = weighted_crossprod_dense(xw_dense, &(-&d2_d2_q_combined), xw_dense)?;
            assign_symmetric_block(&mut joint, w_offset, w_offset, &d2_hww);

            // Time-wiggle D²H[u,v]: bilinear in (u,v) perturbation directions.
            let d2_tw_h0 = &q.d_h_h0 * &(&xi_h0_u * &xi_h0_v);
            let d2_tw_h1 = &q.d_h_h1 * &(&xi_h1_u * &xi_h1_v);
            let d2_h0w = weighted_crossprod_dense(&self.x_time_entry, &(-&d2_tw_h0), xw_dense)?;
            let d2_h1w = weighted_crossprod_dense(&self.x_time_exit, &(-&d2_tw_h1), xw_dense)?;
            assign_symmetric_block(&mut joint, offsets[0], w_offset, &(d2_h0w + d2_h1w));
        }

        Ok(Some(joint))
    }

    fn block_linear_constraints(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        spec: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx == Self::BLOCK_LINK_WIGGLE {
            return Ok(monotone_wiggle_nonnegative_constraints(spec.design.ncols()));
        }
        if block_idx != Self::BLOCK_TIME {
            return Ok(None);
        }
        Ok(self
            .time_coefficient_lower_bounds
            .as_ref()
            .and_then(lower_bound_constraints))
    }

    fn max_feasible_step_size(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        delta: &Array1<f64>,
    ) -> Result<Option<f64>, String> {
        if block_idx == Self::BLOCK_TIME {
            return self.max_feasible_time_step(&block_states[Self::BLOCK_TIME].beta, delta);
        }
        if block_idx == Self::BLOCK_LINK_WIGGLE {
            return self
                .max_feasible_link_wiggle_step(&block_states[Self::BLOCK_LINK_WIGGLE].beta, delta);
        }
        Ok(None)
    }

    fn post_update_block_beta(
        &self,
        _: &[ParameterBlockState],
        block_idx: usize,
        _: &ParameterBlockSpec,
        mut beta: Array1<f64>,
    ) -> Result<Array1<f64>, String> {
        // Accepted line-search trials are not guaranteed to satisfy the
        // structural lower bounds at zero tolerance: the QP solver's internal
        // `project_to_lower_bounds` runs only inside the constrained solve,
        // but the joint-Newton line search computes
        // `trial = old + alpha * delta` and assigns it directly. Over
        // iterations, floating-point drift pushes the binding coefficient a
        // few ulp below its bound (the active-set classifier tolerates
        // 1e-6, so intermediate iterates can land well into the negative),
        // which the next iteration's `max_feasible_time_step` guard rejects
        // at -1e-10. Project back to the feasible set here so every accepted
        // beta is feasible at zero tolerance — a single source of truth.
        if block_idx == Self::BLOCK_TIME
            && let Some(lower_bounds) = self.time_coefficient_lower_bounds.as_ref()
        {
            // Dim mismatch here means the caller and the structural bound
            // vector disagree about the block's coefficient count — the
            // projection has no consistent interpretation in that state.
            // Silently skipping would turn the structural lower-bound
            // guarantee into a no-op and let β drift below the bound on
            // the very next line-search step, which downstream
            // `max_feasible_time_step` then rejects with an opaque
            // "current time coefficient violates structural lower bound"
            // error.  Fail fast with a precise, actionable message.
            if beta.len() != lower_bounds.len() {
                return Err(format!(
                    "survival location-scale time post-update dimension mismatch: beta={}, bounds={}",
                    beta.len(),
                    lower_bounds.len()
                ));
            }
            for j in 0..beta.len() {
                let lb = lower_bounds[j];
                if lb.is_finite() && beta[j] < lb {
                    beta[j] = lb;
                }
            }
        } else if block_idx == Self::BLOCK_LINK_WIGGLE && self.x_link_wiggle.is_some() {
            // Link-wiggle coefficients are structurally non-negative (see
            // max_feasible_link_wiggle_step, which checks beta[j] >= 0).
            for j in 0..beta.len() {
                if beta[j] < 0.0 {
                    beta[j] = 0.0;
                }
            }
        }
        Ok(beta)
    }
}

fn fit_survival_location_scale(
    spec: SurvivalLocationScaleSpec,
) -> Result<UnifiedFitResult, String> {
    let prepared = prepare_survival_location_scale_model(&spec)?;
    let fit = fit_custom_family(
        &prepared.family,
        &prepared.blockspecs,
        &survival_blockwise_fit_options(&spec),
    )?;
    finalize_survival_location_scale_fit(&prepared, &fit)
}

pub(crate) fn select_survival_link_wiggle_basis_from_pilot(
    pilot: &SurvivalLocationScaleTermFitResult,
    wiggle_cfg: &WiggleBlockConfig,
    wiggle_penalty_orders: &[usize],
) -> Result<SelectedWiggleBasis, String> {
    let eta_threshold = pilot
        .threshold_design
        .design
        .dot(&pilot.fit.beta_threshold());
    let eta_log_sigma = pilot
        .log_sigma_design
        .design
        .dot(&pilot.fit.beta_log_sigma());
    let q_seed = Array1::from_iter(
        eta_threshold
            .iter()
            .zip(eta_log_sigma.iter())
            .map(|(&threshold, &ls)| survival_q0_from_eta(threshold, ls)),
    );
    select_wiggle_basis_from_seed(q_seed.view(), wiggle_cfg, wiggle_penalty_orders)
}

fn linkwiggle_block_input_from_selected_basis(
    selected_wiggle_basis: SelectedWiggleBasis,
) -> LinkWiggleBlockInput {
    let crate::families::gamlss::SelectedWiggleBasis {
        block,
        knots,
        degree,
        ..
    } = selected_wiggle_basis;
    let crate::families::gamlss::ParameterBlockInput {
        design,
        penalties,
        nullspace_dims,
        initial_log_lambdas,
        initial_beta,
        ..
    } = block;
    LinkWiggleBlockInput {
        design,
        knots,
        degree,
        penalties,
        nullspace_dims,
        initial_log_lambdas,
        initial_beta,
    }
}

pub(crate) fn fit_survival_location_scale_terms_with_selected_wiggle(
    data: ndarray::ArrayView2<'_, f64>,
    mut spec: SurvivalLocationScaleTermSpec,
    selected_wiggle_basis: SelectedWiggleBasis,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<SurvivalLocationScaleTermFitResult, String> {
    spec.linkwiggle_block = Some(linkwiggle_block_input_from_selected_basis(
        selected_wiggle_basis,
    ));
    fit_survival_location_scale_terms(data, spec, kappa_options)
}

pub(crate) fn fit_survival_location_scale_terms(
    data: ndarray::ArrayView2<'_, f64>,
    spec: SurvivalLocationScaleTermSpec,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> Result<SurvivalLocationScaleTermFitResult, String> {
    let threshold_boot_design =
        build_term_collection_design(data, &spec.thresholdspec).map_err(|e| e.to_string())?;
    let log_sigma_boot_design =
        build_term_collection_design(data, &spec.log_sigmaspec).map_err(|e| e.to_string())?;
    let threshold_bootspec =
        freeze_term_collection_from_design(&spec.thresholdspec, &threshold_boot_design)
            .map_err(|e| e.to_string())?;
    let log_sigma_bootspec =
        freeze_term_collection_from_design(&spec.log_sigmaspec, &log_sigma_boot_design)
            .map_err(|e| e.to_string())?;

    let threshold_boot_derivs = build_survival_covariate_block_psi_derivatives(
        data,
        &threshold_bootspec,
        &threshold_boot_design,
        &spec.threshold_template,
    )?;
    let log_sigma_boot_derivs = build_survival_covariate_block_psi_derivatives(
        data,
        &log_sigma_bootspec,
        &log_sigma_boot_design,
        &spec.log_sigma_template,
    )?;
    let analytic_joint_gradient_available =
        threshold_boot_derivs.is_some() && log_sigma_boot_derivs.is_some();
    let analytic_joint_hessian_available = threshold_boot_derivs
        .as_ref()
        .is_some_and(|derivs| survival_psi_derivatives_support_exact_joint_hessian(derivs))
        && log_sigma_boot_derivs
            .as_ref()
            .is_some_and(|derivs| survival_psi_derivatives_support_exact_joint_hessian(derivs));

    let wiggle_rho0 = spec
        .linkwiggle_block
        .as_ref()
        .and_then(|w| w.initial_log_lambdas.clone())
        .unwrap_or_else(|| Array1::zeros(0));
    let time_rho0 = spec
        .time_block
        .initial_log_lambdas
        .clone()
        .unwrap_or_else(|| Array1::zeros(spec.time_block.penalties.len()));
    let layout = SurvivalLambdaLayout::new(
        spec.time_block.penalties.len(),
        threshold_boot_design.penalties.len(),
        log_sigma_boot_design.penalties.len(),
        wiggle_rho0.len(),
    );
    let mut rho0 = Array1::<f64>::zeros(layout.total());
    if layout.k_time > 0 {
        if time_rho0.len() != layout.k_time {
            return Err(format!(
                "survival time initial_log_lambdas length mismatch: got {}, expected {}",
                time_rho0.len(),
                layout.k_time
            ));
        }
        let range = layout.time_range();
        rho0.slice_mut(s![range.start..range.end])
            .assign(&time_rho0);
    }
    if layout.k_wiggle > 0 {
        let range = layout.wiggle_range();
        rho0.slice_mut(s![range.start..range.end])
            .assign(&wiggle_rho0);
    }
    let joint_setup = build_survival_two_block_exact_joint_setup(
        data.view(),
        &spec.thresholdspec,
        &spec.log_sigmaspec,
        rho0,
        kappa_options,
    );

    let time_beta_hint = std::cell::RefCell::new(spec.time_block.initial_beta.clone());
    let threshold_beta_hint = std::cell::RefCell::new(None::<Array1<f64>>);
    let log_sigma_beta_hint = std::cell::RefCell::new(None::<Array1<f64>>);
    let wiggle_beta_hint = std::cell::RefCell::new(
        spec.linkwiggle_block
            .as_ref()
            .and_then(|w| w.initial_beta.clone()),
    );
    let exact_warm_start = std::cell::RefCell::new(None::<CustomFamilyWarmStart>);

    let build_spec = |rho: &Array1<f64>,
                      _: &TermCollectionSpec,
                      _: &TermCollectionSpec,
                      threshold_design: &TermCollectionDesign,
                      log_sigma_design: &TermCollectionDesign|
     -> Result<SurvivalLocationScaleSpec, String> {
        layout.validate_rho(rho, "survival term fit")?;
        let time_beta = filtered_initial_beta(
            time_beta_hint.borrow().as_ref(),
            spec.time_block.design_exit.ncols(),
        );
        let threshold_block = build_survival_covariate_block_from_design(
            threshold_design,
            &spec.threshold_template,
            &spec.threshold_offset,
            Some(layout.threshold_from(rho)),
            filtered_initial_beta(
                threshold_beta_hint.borrow().as_ref(),
                match &spec.threshold_template {
                    SurvivalCovariateTermBlockTemplate::Static => threshold_design.design.ncols(),
                    SurvivalCovariateTermBlockTemplate::TimeVarying {
                        time_basis_exit, ..
                    } => threshold_design.design.ncols() * time_basis_exit.ncols(),
                },
            ),
        )?;
        let log_sigma_block = build_survival_covariate_block_from_design(
            log_sigma_design,
            &spec.log_sigma_template,
            &spec.log_sigma_offset,
            Some(layout.log_sigma_from(rho)),
            filtered_initial_beta(
                log_sigma_beta_hint.borrow().as_ref(),
                match &spec.log_sigma_template {
                    SurvivalCovariateTermBlockTemplate::Static => log_sigma_design.design.ncols(),
                    SurvivalCovariateTermBlockTemplate::TimeVarying {
                        time_basis_exit, ..
                    } => log_sigma_design.design.ncols() * time_basis_exit.ncols(),
                },
            ),
        )?;
        let linkwiggle_block = spec
            .linkwiggle_block
            .as_ref()
            .map(|wiggle| LinkWiggleBlockInput {
                design: wiggle.design.clone(),
                knots: wiggle.knots.clone(),
                degree: wiggle.degree,
                penalties: wiggle.penalties.clone(),
                nullspace_dims: wiggle.nullspace_dims.clone(),
                initial_log_lambdas: layout.wiggle_from(rho),
                initial_beta: filtered_initial_beta(
                    wiggle_beta_hint.borrow().as_ref(),
                    wiggle.design.ncols(),
                ),
            });
        Ok(SurvivalLocationScaleSpec {
            age_entry: spec.age_entry.clone(),
            age_exit: spec.age_exit.clone(),
            event_target: spec.event_target.clone(),
            weights: spec.weights.clone(),
            inverse_link: spec.inverse_link.clone(),
            derivative_guard: spec.derivative_guard,
            max_iter: spec.max_iter,
            tol: spec.tol,
            time_block: TimeBlockInput {
                design_entry: spec.time_block.design_entry.clone(),
                design_exit: spec.time_block.design_exit.clone(),
                design_derivative_exit: spec.time_block.design_derivative_exit.clone(),
                offset_entry: spec.time_block.offset_entry.clone(),
                offset_exit: spec.time_block.offset_exit.clone(),
                derivative_offset_exit: spec.time_block.derivative_offset_exit.clone(),
                structural_monotonicity: spec.time_block.structural_monotonicity,
                penalties: spec.time_block.penalties.clone(),
                nullspace_dims: spec.time_block.nullspace_dims.clone(),
                initial_log_lambdas: Some(layout.time_from(rho)),
                initial_beta: time_beta,
            },
            threshold_block,
            log_sigma_block,
            timewiggle_block: spec.timewiggle_block.clone(),
            linkwiggle_block,
        })
    };

    let threshold_terms = spatial_length_scale_term_indices(&spec.thresholdspec);
    let log_sigma_terms = spatial_length_scale_term_indices(&spec.log_sigmaspec);
    // Survival location-scale is a multi-block family with β-dependent
    // joint Hessian: disable EFS/HybridEFS at plan time so the outer never
    // pays for a stalled fixed-point attempt before landing on BFGS.
    let solved = optimize_spatial_length_scale_exact_joint(
        data,
        &[spec.thresholdspec.clone(), spec.log_sigmaspec.clone()],
        &[threshold_terms, log_sigma_terms],
        kappa_options,
        &joint_setup,
        crate::seeding::SeedRiskProfile::Survival,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        true,
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let fit = fit_survival_location_scale(build_spec(
                &rho,
                &specs[0],
                &specs[1],
                &designs[0],
                &designs[1],
            )?)?;
            time_beta_hint.replace(Some(fit.beta_time()));
            threshold_beta_hint.replace(Some(fit.beta_threshold()));
            log_sigma_beta_hint.replace(Some(fit.beta_log_sigma()));
            wiggle_beta_hint.replace(fit.beta_link_wiggle());
            Ok(fit)
        },
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign], need_hessian| {
            if !analytic_joint_gradient_available {
                return Err(
                    "analytic spatial psi derivatives are unavailable for survival exact two-block path"
                        .to_string(),
                );
            }
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let assembled = build_spec(&rho, &specs[0], &specs[1], &designs[0], &designs[1])?;
            let prepared = prepare_survival_location_scale_model(&assembled)?;
            let threshold_derivs = build_survival_covariate_block_psi_derivatives(
                data,
                &specs[0],
                &designs[0],
                &spec.threshold_template,
            )?
            .ok_or_else(|| "missing survival threshold spatial psi derivatives".to_string())?;
            let log_sigma_derivs = build_survival_covariate_block_psi_derivatives(
                data,
                &specs[1],
                &designs[1],
                &spec.log_sigma_template,
            )?
            .ok_or_else(|| "missing survival log-sigma spatial psi derivatives".to_string())?;
            let mut derivative_blocks = vec![Vec::new(), threshold_derivs, log_sigma_derivs];
            if prepared.family.x_link_wiggle.is_some() {
                derivative_blocks.push(Vec::new());
            }
            let eval = evaluate_custom_family_joint_hyper(
                &prepared.family,
                &prepared.blockspecs,
                &survival_blockwise_fit_options(&assembled),
                &rho,
                &derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                if need_hessian && analytic_joint_hessian_available {
                    crate::solver::estimate::reml::unified::EvalMode::ValueGradientHessian
                } else {
                    crate::solver::estimate::reml::unified::EvalMode::ValueAndGradient
                },
            )
            .map_err(|e| e.to_string())?;
            exact_warm_start.replace(Some(eval.warm_start));
            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
        |theta, specs: &[TermCollectionSpec], designs: &[TermCollectionDesign]| {
            if !analytic_joint_gradient_available {
                return Err(
                    "analytic spatial psi derivatives are unavailable for survival exact two-block path"
                        .to_string(),
                );
            }
            let rho = theta.slice(s![..joint_setup.rho_dim()]).to_owned();
            let assembled = build_spec(&rho, &specs[0], &specs[1], &designs[0], &designs[1])?;
            let prepared = prepare_survival_location_scale_model(&assembled)?;
            let threshold_derivs = build_survival_covariate_block_psi_derivatives(
                data,
                &specs[0],
                &designs[0],
                &spec.threshold_template,
            )?
            .ok_or_else(|| "missing survival threshold spatial psi derivatives".to_string())?;
            let log_sigma_derivs = build_survival_covariate_block_psi_derivatives(
                data,
                &specs[1],
                &designs[1],
                &spec.log_sigma_template,
            )?
            .ok_or_else(|| "missing survival log-sigma spatial psi derivatives".to_string())?;
            let mut derivative_blocks = vec![Vec::new(), threshold_derivs, log_sigma_derivs];
            if prepared.family.x_link_wiggle.is_some() {
                derivative_blocks.push(Vec::new());
            }
            let eval = evaluate_custom_family_joint_hyper_efs(
                &prepared.family,
                &prepared.blockspecs,
                &survival_blockwise_fit_options(&assembled),
                &rho,
                &derivative_blocks,
                exact_warm_start.borrow().as_ref(),
            )
            .map_err(|e| e.to_string())?;
            exact_warm_start.replace(Some(eval.warm_start));
            Ok(eval.efs_eval)
        },
    )?;

    let mut resolved_specs = solved.resolved_specs;
    let mut designs = solved.designs;
    Ok(SurvivalLocationScaleTermFitResult {
        fit: solved.fit,
        resolved_thresholdspec: resolved_specs.remove(0),
        resolved_log_sigmaspec: resolved_specs.remove(0),
        threshold_design: designs.remove(0),
        log_sigma_design: designs.remove(0),
    })
}

pub fn predict_survival_location_scale(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
) -> Result<SurvivalLocationScalePredictResult, String> {
    let predictors = prediction_linear_predictors(input, fit)?;
    let n = input.x_time_exit.nrows();
    let inv_sigma = predictors.eta_ls.mapv(exp_sigma_inverse_from_eta_scalar);
    let eta = Array1::from_iter(
        predictors
            .h
            .iter()
            .zip(predictors.eta_t.iter())
            .zip(inv_sigma.iter())
            .enumerate()
            .map(|(i, ((&hh, &tt), &r))| {
                let mut q = hh - tt * r;
                if let Some(w) = predictors.etaw.as_ref() {
                    q += w[i];
                }
                q
            }),
    );
    let mut survival_prob = Array1::<f64>::zeros(n);
    for (i, &v) in eta.iter().enumerate() {
        survival_prob[i] = inverse_link_survival_prob_checked(&input.inverse_link, v)?;
    }
    Ok(SurvivalLocationScalePredictResult { eta, survival_prob })
}

pub fn predict_survival_location_scale_posterior_mean(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
) -> Result<SurvivalLocationScalePredictResult, String> {
    let pred = predict_survival_location_scale(input, fit)?;
    let (survival_prob, _) = exact_survival_response_moments(input, fit, covariance)?;

    Ok(SurvivalLocationScalePredictResult {
        eta: pred.eta,
        survival_prob,
    })
}

pub fn predict_survival_location_scalewith_uncertainty(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
    covariance: &Array2<f64>,
    posterior_mean: bool,
    include_response_sd: bool,
) -> Result<SurvivalLocationScalePredictUncertaintyResult, String> {
    let base = predict_survival_location_scale(input, fit)?;
    let n = input.x_time_exit.nrows();
    let p_time = fit.beta_time().len();
    let p_t = fit.beta_threshold().len();
    let p_ls = fit.beta_log_sigma().len();
    let beta_link_wiggle = fit.beta_link_wiggle();
    let pw = beta_link_wiggle.as_ref().map_or(0, |b| b.len());
    let resolved_wiggle_knots = input
        .link_wiggle_knots
        .as_ref()
        .or(fit.artifacts.survival_link_wiggle_knots.as_ref());
    let resolved_wiggle_degree = input
        .link_wiggle_degree
        .or(fit.artifacts.survival_link_wiggle_degree);
    let p_total = p_time + p_t + p_ls + pw;
    if covariance.nrows() != p_total || covariance.ncols() != p_total {
        return Err(format!(
            "predict_survival_location_scalewith_uncertainty: covariance shape mismatch: got {}x{}, expected {}x{}",
            covariance.nrows(),
            covariance.ncols(),
            p_total,
            p_total
        ));
    }
    if pw > 0
        && (beta_link_wiggle.is_none()
            || resolved_wiggle_knots.is_none()
            || resolved_wiggle_degree.is_none())
    {
        return Err(
            "predict_survival_location_scalewith_uncertainty: dynamic link-wiggle metadata is incomplete"
                .to_string(),
        );
    }

    let predictors = prediction_linear_predictors(input, fit)?;
    if input.x_threshold.nrows() != n || input.x_log_sigma.nrows() != n {
        return Err(
            "predict_survival_location_scalewith_uncertainty: row mismatch across design views"
                .to_string(),
        );
    }
    let inv_sigma = predictors.eta_ls.mapv(exp_sigma_inverse_from_eta_scalar);
    let wiggle_design = predictors.wiggle_design.as_ref();
    let dq_dq0 = predictors.dq_dq0.as_ref();
    let x_t_dense = input.x_threshold.to_dense();
    let x_ls_dense = input.x_log_sigma.to_dense();
    let mut grad = Array2::<f64>::zeros((n, p_total));
    for i in 0..n {
        for j in 0..p_time {
            grad[[i, j]] = predictors.time_jac[[i, j]];
        }
        let scale = dq_dq0.as_ref().map_or(1.0, |v| v[i]);
        for j in 0..p_t {
            grad[[i, p_time + j]] = -scale * inv_sigma[i] * x_t_dense[[i, j]];
        }
        let coeff_ls = scale * predictors.eta_t[i] * inv_sigma[i];
        for j in 0..p_ls {
            grad[[i, p_time + p_t + j]] = coeff_ls * x_ls_dense[[i, j]];
        }
        if let Some(xw) = wiggle_design {
            for j in 0..pw {
                grad[[i, p_time + p_t + p_ls + j]] = xw[[i, j]];
            }
        }
    }
    let eta_se = linear_predictor_se(grad.view(), covariance);

    let exact_response_moments = if posterior_mean || include_response_sd {
        Some(exact_survival_response_moments(input, fit, covariance)?)
    } else {
        None
    };
    let posterior_mean_response = exact_response_moments
        .as_ref()
        .map(|(mean, _)| mean.clone());
    let posterior_second_moment = exact_response_moments
        .as_ref()
        .map(|(_, second)| second.clone());

    let survival_prob = if posterior_mean {
        posterior_mean_response
            .as_ref()
            .expect("posterior-mean path computes exact response moments")
            .clone()
    } else {
        base.survival_prob.clone()
    };

    let response_standard_error = if include_response_sd {
        let mean = posterior_mean_response
            .as_ref()
            .expect("response-sd path computes exact response moments");
        let second = posterior_second_moment
            .as_ref()
            .expect("response-sd path computes exact response moments");
        Some(Array1::from_iter(
            (0..n).map(|i| (second[i] - mean[i] * mean[i]).max(0.0).sqrt()),
        ))
    } else {
        None
    };

    Ok(SurvivalLocationScalePredictUncertaintyResult {
        eta: base.eta,
        survival_prob,
        eta_standard_error: eta_se,
        response_standard_error,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::custom_family::BlockWorkingSet;
    use crate::mixture_link::{state_from_beta_logisticspec, state_from_sasspec, state_fromspec};
    use crate::types::{LinkComponent, MixtureLinkSpec, SasLinkSpec};
    use faer::sparse::{SparseColMat, Triplet};
    use ndarray::{Array1, array};

    fn sparse_design_from_dense(dense: &Array2<f64>) -> DesignMatrix {
        let mut triplets = Vec::new();
        for i in 0..dense.nrows() {
            for j in 0..dense.ncols() {
                let value = dense[[i, j]];
                if value != 0.0 {
                    triplets.push(Triplet::new(i, j, value));
                }
            }
        }
        DesignMatrix::from(
            SparseColMat::try_new_from_triplets(dense.nrows(), dense.ncols(), &triplets)
                .expect("build sparse design"),
        )
    }

    fn test_link_wiggle_metadata(beta_link_wiggle: &Array1<f64>) -> (Array1<f64>, usize) {
        let seed = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        for degree in [2usize, 3, 1] {
            for num_internal_knots in 0..=8 {
                let cfg = WiggleBlockConfig {
                    degree,
                    num_internal_knots,
                    penalty_order: 2,
                    double_penalty: false,
                };
                if let Ok((block, knots)) =
                    crate::families::gamlss::buildwiggle_block_input_from_seed(seed.view(), &cfg)
                    && block.design.ncols() == beta_link_wiggle.len()
                {
                    return (knots, degree);
                }
            }
        }
        panic!(
            "could not synthesize valid link wiggle metadata for {} coefficients",
            beta_link_wiggle.len()
        );
    }

    fn test_survival_fit(
        beta_time: Array1<f64>,
        beta_threshold: Array1<f64>,
        beta_log_sigma: Array1<f64>,
        beta_link_wiggle: Option<Array1<f64>>,
    ) -> UnifiedFitResult {
        let lambdas_linkwiggle = beta_link_wiggle.as_ref().map(|_| Array1::zeros(0));
        let (link_wiggle_knots, link_wiggle_degree) = beta_link_wiggle
            .as_ref()
            .map(|beta| {
                let (knots, degree) = test_link_wiggle_metadata(beta);
                (Some(knots), Some(degree))
            })
            .unwrap_or((None, None));
        survival_fit_from_parts(SurvivalLocationScaleFitResultParts {
            beta_time,
            beta_threshold,
            beta_log_sigma,
            beta_link_wiggle,
            link_wiggle_knots,
            link_wiggle_degree,
            lambdas_time: Array1::zeros(0),
            lambdas_threshold: Array1::zeros(0),
            lambdas_log_sigma: Array1::zeros(0),
            lambdas_linkwiggle,
            log_likelihood: 0.0,
            reml_score: 0.0,
            stable_penalty_term: 0.0,
            penalized_objective: 0.0,
            outer_iterations: 0,
            outer_gradient_norm: 0.0,
            outer_converged: true,
            covariance_conditional: None,
            geometry: None,
        })
        .expect("valid survival test fit")
    }

    fn survival_exact_newton_test_family() -> SurvivalLocationScaleFamily {
        SurvivalLocationScaleFamily {
            n: 3,
            y: array![1.0, 0.0, 1.0],
            w: array![1.0, 0.8, 1.2],
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
            derivative_guard: 1e-8,
            x_time_entry: Arc::new(array![[1.0], [1.0], [1.0]]),
            x_time_exit: Arc::new(array![[1.2], [0.9], [1.4]]),
            x_time_deriv: Arc::new(array![[1.0], [1.0], [1.0]]),
            time_derivative_offset_exit: Arc::new(Array1::from_elem(3, 1e-8)),
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            time_coefficient_lower_bounds: Some(array![0.0]),
            x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                [1.0],
                [0.4],
                [-0.6]
            ])),
            x_threshold_entry: None,
            x_threshold_deriv: None,
            x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                [1.0],
                [-0.3],
                [0.5]
            ])),
            x_log_sigma_entry: None,
            x_log_sigma_deriv: None,
            x_link_wiggle: None,
            wiggle_knots: None,
            wiggle_degree: None,
            policy: crate::resource::ResourcePolicy::default_library(),
        }
    }

    #[test]
    fn time_block_post_update_leaves_beta_unchanged() {
        // The time block carries structural lower bounds (see
        // `structural_time_coefficient_lower_bounds`).  `post_update_block_beta`
        // projects the accepted line-search β onto the feasible box so the
        // *next* `max_feasible_time_step` call never sees an ulp-level
        // infeasibility.  The projection must satisfy two invariants:
        //   (i)  feasible β is returned unchanged (idempotence on the feasible
        //        set),
        //   (ii) β projected from outside the feasible set lies on the feasible
        //        set (every coefficient is ≥ its lower bound).
        // The test family has `time_coefficient_lower_bounds = [0.0]`, matching
        // the single structural time column in `x_time_exit`.
        let family = survival_exact_newton_test_family();
        let spec = ParameterBlockSpec {
            name: "time_transform".to_string(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::zeros((1, 1)))),
            offset: Array1::zeros(1),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
        };

        // Feasible β: already ≥ the 0.0 bound, so the projection is a no-op.
        let feasible = family
            .post_update_block_beta(
                &[ParameterBlockState {
                    beta: array![0.0],
                    eta: array![0.0, 0.0, 0.0],
                }],
                SurvivalLocationScaleFamily::BLOCK_TIME,
                &spec,
                array![0.5],
            )
            .expect("return time beta");
        assert_eq!(feasible, array![0.5]);

        // Infeasible β: -2.0 sits below the 0.0 lower bound, so the projection
        // clamps it back onto the feasible set.  The invariant the test asserts
        // is "every coefficient is ≥ its structural lower bound" — the exact
        // returned value is the projection-to-box fixed point.
        let projected = family
            .post_update_block_beta(
                &[ParameterBlockState {
                    beta: array![0.0],
                    eta: array![0.0, 0.0, 0.0],
                }],
                SurvivalLocationScaleFamily::BLOCK_TIME,
                &spec,
                array![-2.0],
            )
            .expect("return time beta");
        assert!(projected[0] >= 0.0);
    }

    #[test]
    fn time_block_feasible_step_stays_inside_derivative_guard() {
        let family = survival_exact_newton_test_family();
        let states = vec![
            ParameterBlockState {
                beta: array![0.1],
                eta: array![0.0, 0.0, 0.0],
            },
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0, 0.0, 0.0],
            },
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0, 0.0, 0.0],
            },
        ];
        let alpha = family
            .max_feasible_step_size(
                &states,
                SurvivalLocationScaleFamily::BLOCK_TIME,
                &array![-2.0],
            )
            .expect("time step ceiling")
            .expect("time step should be bounded");
        assert_eq!(alpha, 0.0);
        let feasible = states[0].beta[0] + alpha * -2.0;
        assert!(feasible >= 0.0);
    }

    #[test]
    fn time_block_feasible_step_accepts_zero_beta_when_offset_encodes_guard() {
        let family = survival_exact_newton_test_family();
        let states = vec![
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0, 0.0, 1e-8],
            },
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0, 0.0, 0.0],
            },
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0, 0.0, 0.0],
            },
        ];
        let alpha = family
            .max_feasible_step_size(
                &states,
                SurvivalLocationScaleFamily::BLOCK_TIME,
                &array![0.0],
            )
            .expect("zero-step structural state should be valid")
            .expect("time step should be bounded");
        assert_eq!(alpha, 1.0);
    }

    #[test]
    fn linkwiggle_block_post_update_leaves_beta_unchanged() {
        let mut family = survival_exact_newton_test_family();
        family.x_link_wiggle = Some(DesignMatrix::Dense(DenseDesignMatrix::from(array![
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])));
        family.wiggle_knots = Some(array![-2.0, -2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0]);
        family.wiggle_degree = Some(3);
        let spec = ParameterBlockSpec {
            name: "linkwiggle".to_string(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(Array2::<f64>::zeros((1, 2)))),
            offset: Array1::zeros(1),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::zeros(0),
            initial_beta: None,
        };
        let returned = family
            .post_update_block_beta(
                &[
                    ParameterBlockState {
                        beta: array![0.0],
                        eta: array![0.0, 0.0, 0.0],
                    },
                    ParameterBlockState {
                        beta: array![0.0],
                        eta: array![0.0, 0.0, 0.0],
                    },
                    ParameterBlockState {
                        beta: array![0.0],
                        eta: array![0.0, 0.0, 0.0],
                    },
                    ParameterBlockState {
                        beta: array![0.1, 0.2],
                        eta: array![0.0, 0.0, 0.0],
                    },
                ],
                SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE,
                &spec,
                array![0.3, 0.0],
            )
            .expect("return linkwiggle beta");
        assert_eq!(returned, array![0.3, 0.0]);
    }

    #[test]
    fn linkwiggle_block_feasible_step_stays_nonnegative() {
        let mut family = survival_exact_newton_test_family();
        family.x_link_wiggle = Some(DesignMatrix::Dense(DenseDesignMatrix::from(array![
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ])));
        family.wiggle_knots = Some(array![-2.0, -2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 2.0]);
        family.wiggle_degree = Some(3);
        let states = vec![
            ParameterBlockState {
                beta: array![0.1],
                eta: array![0.0, 0.0, 0.0],
            },
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0, 0.0, 0.0],
            },
            ParameterBlockState {
                beta: array![0.0],
                eta: array![0.0, 0.0, 0.0],
            },
            ParameterBlockState {
                beta: array![0.2, 0.4],
                eta: array![0.0, 0.0, 0.0],
            },
        ];
        let alpha = family
            .max_feasible_step_size(
                &states,
                SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE,
                &array![-1.0, -0.1],
            )
            .expect("linkwiggle step ceiling")
            .expect("linkwiggle step should be bounded");
        assert!(alpha > 0.0 && alpha < 1.0);
        let feasible = &states[SurvivalLocationScaleFamily::BLOCK_LINK_WIGGLE].beta
            + &(array![-1.0, -0.1] * alpha);
        assert!(feasible.iter().all(|&value| value >= 0.0));
    }

    fn survival_exact_newton_test_familywith_inverse_link(
        inverse_link: InverseLink,
    ) -> SurvivalLocationScaleFamily {
        SurvivalLocationScaleFamily {
            inverse_link,
            ..survival_exact_newton_test_family()
        }
    }

    fn sparse_survival_exact_newton_test_family() -> SurvivalLocationScaleFamily {
        let mut family = survival_exact_newton_test_family();
        family.x_threshold = sparse_design_from_dense(&array![[1.0], [0.4], [-0.6]]);
        family.x_log_sigma = sparse_design_from_dense(&array![[1.0], [-0.3], [0.5]]);
        family
    }

    #[test]
    fn compose_survival_dynamic_q_uses_correct_qdot_ll_coefficient() {
        let base = survival_base_q_scalars(0.8, -0.35);
        let eta_t_deriv = 1.4;
        let eta_ls_deriv = -0.6;
        let wiggle_value = 0.2;
        let dq_dq0 = 1.1;
        let d2q_dq02 = -0.7;
        let d3q_dq03 = 0.45;
        let d4q_dq04 = -0.15;

        let dyn_q = compose_survival_dynamic_q(
            base,
            eta_t_deriv,
            eta_ls_deriv,
            wiggle_value,
            dq_dq0,
            d2q_dq02,
            d3q_dq03,
            d4q_dq04,
        );

        let a = base.q_t;
        let b = base.q_ls;
        let d = base.q_ll;
        let e = base.q_tl_ls;
        let f = base.q_ll_ls;
        let r = safe_sum2(safe_product(a, eta_t_deriv), safe_product(b, eta_ls_deriv));
        let r_ls = safe_sum2(
            safe_product(base.q_tl, eta_t_deriv),
            safe_product(d, eta_ls_deriv),
        );
        let r_ll = safe_sum2(safe_product(e, eta_t_deriv), safe_product(f, eta_ls_deriv));
        let expected = safe_sum3(
            safe_product(d3q_dq03, safe_product(safe_product(b, b), r)),
            safe_product(
                d2q_dq02,
                safe_sum2(safe_product(d, r), 2.0 * safe_product(b, r_ls)),
            ),
            safe_product(dq_dq0, r_ll),
        );

        assert!(
            (dyn_q.qdot_ll - expected).abs() <= 1e-12,
            "qdot_ll mismatch: got {}, expected {}",
            dyn_q.qdot_ll,
            expected
        );
    }

    fn survival_exact_newton_test_states(beta_t: f64) -> Vec<ParameterBlockState> {
        vec![
            ParameterBlockState {
                beta: array![0.2],
                eta: array![0.1, 0.35, -0.2, 0.25, 0.6, 0.15, 0.5, 0.7, 0.6],
            },
            ParameterBlockState {
                beta: array![beta_t],
                eta: array![beta_t, 0.4 * beta_t, -0.6 * beta_t],
            },
            ParameterBlockState {
                beta: array![-0.15],
                eta: array![-0.15, 0.045, -0.075],
            },
        ]
    }

    fn survival_exact_newton_rebuild_states(
        beta_time: &Array1<f64>,
        beta_threshold: &Array1<f64>,
        beta_log_sigma: &Array1<f64>,
    ) -> Vec<ParameterBlockState> {
        vec![
            ParameterBlockState {
                beta: beta_time.clone(),
                eta: array![
                    beta_time[0],
                    beta_time[0],
                    beta_time[0],
                    1.2 * beta_time[0],
                    0.9 * beta_time[0],
                    1.4 * beta_time[0],
                    beta_time[0] + 0.5,
                    beta_time[0] + 0.7,
                    beta_time[0] + 0.6
                ],
            },
            ParameterBlockState {
                beta: beta_threshold.clone(),
                eta: array![
                    beta_threshold[0],
                    0.4 * beta_threshold[0],
                    -0.6 * beta_threshold[0]
                ],
            },
            ParameterBlockState {
                beta: beta_log_sigma.clone(),
                eta: array![
                    beta_log_sigma[0],
                    -0.3 * beta_log_sigma[0],
                    0.5 * beta_log_sigma[0]
                ],
            },
        ]
    }

    fn survival_outergradient_testspecs() -> Vec<ParameterBlockSpec> {
        vec![
            ParameterBlockSpec {
                name: "time_transform".to_string(),
                design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.2],
                    [0.9],
                    [1.4],
                    [1.0],
                    [1.0],
                    [1.0]
                ])),
                offset: Array1::zeros(9),
                penalties: vec![PenaltyMatrix::Dense(Array2::eye(1))],
                nullspace_dims: vec![],
                initial_log_lambdas: array![0.0],
                initial_beta: Some(array![0.2]),
            },
            ParameterBlockSpec {
                name: "threshold".to_string(),
                design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                    [1.0],
                    [0.4],
                    [-0.6]
                ])),
                offset: Array1::zeros(3),
                penalties: vec![],
                nullspace_dims: vec![],
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: Some(array![0.35]),
            },
            ParameterBlockSpec {
                name: "log_sigma".to_string(),
                design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![
                    [1.0],
                    [-0.3],
                    [0.5]
                ])),
                offset: Array1::zeros(3),
                penalties: vec![],
                nullspace_dims: vec![],
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: Some(array![-0.15]),
            },
        ]
    }

    fn survival_non_probit_test_links() -> Vec<(&'static str, InverseLink)> {
        vec![
            (
                "logistic",
                residual_distribution_inverse_link(ResidualDistribution::Logistic),
            ),
            (
                "cloglog",
                residual_distribution_inverse_link(ResidualDistribution::Gumbel),
            ),
            (
                "sas",
                InverseLink::Sas(
                    state_from_sasspec(SasLinkSpec {
                        initial_epsilon: 0.1,
                        initial_log_delta: -0.2,
                    })
                    .expect("sas state"),
                ),
            ),
            (
                "beta-logistic",
                InverseLink::BetaLogistic(
                    state_from_beta_logisticspec(SasLinkSpec {
                        initial_epsilon: 0.05,
                        initial_log_delta: 0.1,
                    })
                    .expect("beta-logistic state"),
                ),
            ),
        ]
    }

    #[test]
    fn wip_outergradient_testspecs_shape() {
        let specs = survival_outergradient_testspecs();
        assert_eq!(specs.len(), 3);
        assert_eq!(specs[0].name, "time_transform");
        assert_eq!(specs[1].name, "threshold");
        assert_eq!(specs[2].name, "log_sigma");
    }

    #[test]
    fn identified_time_block_preserves_input_designs() {
        let design_entry = array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]];
        let design_exit = array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]];
        let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
        let time_block = TimeBlockInput {
            design_entry: DesignMatrix::from(design_entry.clone()),
            design_exit: DesignMatrix::from(design_exit.clone()),
            design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
            offset_entry: Array1::zeros(3),
            offset_exit: Array1::zeros(3),
            derivative_offset_exit: Array1::from_elem(3, 1e-6),
            structural_monotonicity: true,
            penalties: vec![Array2::eye(3)],
            nullspace_dims: vec![],
            initial_log_lambdas: None,
            initial_beta: None,
        };
        let prepared =
            prepare_identified_time_block(&time_block, 1e-6, 0).expect("prepare time block");
        assert_eq!(prepared.design_entry, design_entry);
        assert_eq!(prepared.design_exit, design_exit);
        assert_eq!(prepared.design_derivative_exit, design_derivative_exit);
    }

    #[test]
    fn identified_time_block_preserves_expected_nullspace_dimension() {
        let design_entry = array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]];
        let design_exit = array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]];
        let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
        let time_block = TimeBlockInput {
            design_entry: DesignMatrix::from(design_entry),
            design_exit: DesignMatrix::from(design_exit),
            design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
            offset_entry: Array1::zeros(3),
            offset_exit: Array1::zeros(3),
            derivative_offset_exit: Array1::from_elem(3, 1e-6),
            structural_monotonicity: true,
            penalties: vec![Array2::eye(3)],
            nullspace_dims: vec![],
            initial_log_lambdas: None,
            initial_beta: None,
        };

        let prepared =
            prepare_identified_time_block(&time_block, 1e-6, 0).expect("prepare time block");
        let p = time_block.design_entry.ncols();

        assert_eq!(
            prepared.transform.z.nrows(),
            p,
            "identifiability transform must stay in the original coefficient space"
        );
        assert_eq!(
            prepared.transform.z.ncols(),
            p,
            "anchored time basis should keep the full coefficient dimension"
        );
        assert_eq!(
            prepared.design_entry.ncols(),
            p,
            "prepared entry design should keep the full anchored basis width"
        );
        assert_eq!(
            prepared.design_exit.ncols(),
            p,
            "prepared exit design should keep the full anchored basis width"
        );
        assert_eq!(prepared.transform.z, Array2::<f64>::eye(p));
    }

    #[test]
    fn identified_time_block_uses_structural_coefficient_constraints() {
        let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
        let time_block = TimeBlockInput {
            design_entry: DesignMatrix::from(array![
                [1.0, 0.0, 0.2],
                [1.0, 1.0, 0.5],
                [1.0, 2.0, 1.0]
            ]),
            design_exit: DesignMatrix::from(array![
                [1.0, 0.5, 0.3],
                [1.0, 1.5, 0.8],
                [1.0, 2.5, 1.4]
            ]),
            design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
            offset_entry: Array1::zeros(3),
            offset_exit: Array1::zeros(3),
            derivative_offset_exit: Array1::from_elem(3, 1e-6),
            structural_monotonicity: true,
            penalties: vec![Array2::eye(3)],
            nullspace_dims: vec![],
            initial_log_lambdas: None,
            initial_beta: Some(array![-0.5, 0.2, -1.5]),
        };
        let prepared =
            prepare_identified_time_block(&time_block, 1e-6, 0).expect("prepare time block");
        assert_eq!(
            prepared.coefficient_lower_bounds,
            Some(array![f64::NEG_INFINITY, 0.0, 0.0])
        );
        let constraints = lower_bound_constraints(
            prepared
                .coefficient_lower_bounds
                .as_ref()
                .expect("time coefficient lower bounds"),
        )
        .expect("time coefficient constraints");
        assert_eq!(constraints.a, array![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        assert_eq!(constraints.b, Array1::<f64>::zeros(2));
        assert_eq!(prepared.initial_beta, Some(array![-0.5, 0.2, 0.0]));
    }

    #[test]
    fn identified_time_block_constrains_monotone_timewiggle_tail_coefficients() {
        let design_derivative_exit = array![
            [0.0, 1.0, 0.2, 0.0],
            [0.0, 1.0, 0.3, 0.0],
            [0.0, 1.0, 0.4, 0.0]
        ];
        let time_block = TimeBlockInput {
            design_entry: DesignMatrix::from(array![
                [1.0, 0.0, 0.2, 0.0],
                [1.0, 1.0, 0.5, 0.0],
                [1.0, 2.0, 1.0, 0.0]
            ]),
            design_exit: DesignMatrix::from(array![
                [1.0, 0.5, 0.3, 0.0],
                [1.0, 1.5, 0.8, 0.0],
                [1.0, 2.5, 1.4, 0.0]
            ]),
            design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
            offset_entry: Array1::zeros(3),
            offset_exit: Array1::zeros(3),
            derivative_offset_exit: Array1::from_elem(3, 1e-6),
            structural_monotonicity: true,
            penalties: vec![Array2::eye(4)],
            nullspace_dims: vec![],
            initial_log_lambdas: None,
            initial_beta: Some(array![-0.5, 0.2, -1.5, -2.0]),
        };
        let prepared =
            prepare_identified_time_block(&time_block, 1e-6, 1).expect("prepare time block");
        assert_eq!(
            prepared.coefficient_lower_bounds,
            Some(array![f64::NEG_INFINITY, 0.0, 0.0, 0.0])
        );
        assert_eq!(prepared.initial_beta, Some(array![-0.5, 0.2, 0.0, 0.0]));
    }

    #[test]
    fn identified_time_block_rejects_offsets_below_derivative_guard() {
        let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
        let time_block = TimeBlockInput {
            design_entry: DesignMatrix::from(array![
                [1.0, 0.0, 0.2],
                [1.0, 1.0, 0.5],
                [1.0, 2.0, 1.0]
            ]),
            design_exit: DesignMatrix::from(array![
                [1.0, 0.5, 0.3],
                [1.0, 1.5, 0.8],
                [1.0, 2.5, 1.4]
            ]),
            design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
            offset_entry: Array1::zeros(3),
            offset_exit: Array1::zeros(3),
            derivative_offset_exit: Array1::zeros(3),
            structural_monotonicity: true,
            penalties: vec![Array2::eye(3)],
            nullspace_dims: vec![],
            initial_log_lambdas: None,
            initial_beta: None,
        };
        let err = match prepare_identified_time_block(&time_block, 1e-6, 0) {
            Ok(_) => panic!("offsets below the guard must be rejected"),
            Err(err) => err,
        };
        assert!(
            err.contains("require derivative offsets to encode the derivative guard"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn prepare_model_accepts_time_initializer_when_offset_completes_guard() {
        let n = 3usize;
        let derivative_guard = 5e-10;
        let derivative_offset_exit = Array1::from_elem(n, 6e-10);
        let spec = SurvivalLocationScaleSpec {
            age_entry: Array1::from_elem(n, 1.0),
            age_exit: Array1::from_elem(n, 5e9),
            event_target: array![1.0, 0.0, 1.0],
            weights: Array1::ones(n),
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
            derivative_guard,
            max_iter: 4,
            tol: 1e-8,
            time_block: TimeBlockInput {
                design_entry: DesignMatrix::from(Array2::zeros((n, 1))),
                design_exit: DesignMatrix::from(Array2::zeros((n, 1))),
                design_derivative_exit: DesignMatrix::from(Array2::ones((n, 1))),
                offset_entry: Array1::zeros(n),
                offset_exit: Array1::zeros(n),
                derivative_offset_exit: derivative_offset_exit.clone(),
                structural_monotonicity: true,
                penalties: vec![Array2::zeros((1, 1))],
                nullspace_dims: vec![1],
                initial_log_lambdas: None,
                initial_beta: None,
            },
            threshold_block: CovariateBlockKind::Static(CovariateBlockInput {
                design: DesignMatrix::from(Array2::ones((n, 1))),
                offset: Array1::zeros(n),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_log_lambdas: None,
                initial_beta: None,
            }),
            log_sigma_block: CovariateBlockKind::Static(CovariateBlockInput {
                design: DesignMatrix::from(Array2::ones((n, 1))),
                offset: Array1::zeros(n),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_log_lambdas: None,
                initial_beta: None,
            }),
            timewiggle_block: None,
            linkwiggle_block: None,
        };

        let prepared = prepare_survival_location_scale_model(&spec)
            .expect("offset-supported time initializer should be accepted");
        let beta_init = prepared.blockspecs[0]
            .initial_beta
            .as_ref()
            .expect("time initializer should be present");
        let d_raw_init = Array2::ones((n, 1)).dot(beta_init) + &derivative_offset_exit;
        assert!(
            d_raw_init.iter().all(|v| *v >= derivative_guard),
            "initializer must satisfy derivative guard once offsets are included: {d_raw_init:?}"
        );
    }

    #[test]
    fn prepare_model_seeds_structural_time_initializer_when_offset_equals_guard() {
        let n = 20usize;
        let p_time = 8usize;
        let derivative_guard = DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD;
        let derivative_offset_exit = Array1::from_elem(n, derivative_guard);
        let age_exit = Array1::from_iter((0..n).map(|i| 4.0 + (i as f64) * 14.0));
        let mut design_derivative_exit = Array2::<f64>::zeros((n, p_time));
        for i in 0..n {
            let t = (i as f64) / ((n - 1) as f64);
            for j in 0..p_time {
                let center = (j as f64 + 0.5) / (p_time as f64);
                let x = 8.0 * (t - center);
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                design_derivative_exit[[i, j]] = 8.0 * sigmoid * (1.0 - sigmoid) / age_exit[i];
            }
        }

        let spec = SurvivalLocationScaleSpec {
            age_entry: Array1::from_elem(n, 1e-9),
            age_exit: age_exit.clone(),
            event_target: Array1::zeros(n),
            weights: Array1::ones(n),
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
            derivative_guard,
            max_iter: 4,
            tol: 1e-8,
            time_block: TimeBlockInput {
                design_entry: DesignMatrix::from(Array2::zeros((n, p_time))),
                design_exit: DesignMatrix::from(Array2::zeros((n, p_time))),
                design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
                offset_entry: Array1::zeros(n),
                offset_exit: Array1::zeros(n),
                derivative_offset_exit: derivative_offset_exit.clone(),
                structural_monotonicity: true,
                penalties: vec![Array2::eye(p_time)],
                nullspace_dims: vec![],
                initial_log_lambdas: None,
                initial_beta: None,
            },
            threshold_block: CovariateBlockKind::Static(CovariateBlockInput {
                design: DesignMatrix::from(Array2::ones((n, 1))),
                offset: Array1::zeros(n),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_log_lambdas: None,
                initial_beta: None,
            }),
            log_sigma_block: CovariateBlockKind::Static(CovariateBlockInput {
                design: DesignMatrix::from(Array2::ones((n, 1))),
                offset: Array1::zeros(n),
                penalties: Vec::new(),
                nullspace_dims: Vec::new(),
                initial_log_lambdas: None,
                initial_beta: None,
            }),
            timewiggle_block: None,
            linkwiggle_block: None,
        };

        let prepared = prepare_survival_location_scale_model(&spec)
            .expect("guard-sized derivative offset should still seed time initializer");
        let beta_init = prepared.blockspecs[0]
            .initial_beta
            .as_ref()
            .expect("time initializer should be present");
        let d_raw_init = design_derivative_exit.dot(beta_init) + &derivative_offset_exit;

        assert!(beta_init.iter().all(|v| v.is_finite() && *v >= 0.0));
        assert!(beta_init.iter().any(|v| *v > 0.0));
        assert!(
            d_raw_init
                .iter()
                .all(|v| v.is_finite() && *v >= derivative_guard),
            "initializer must satisfy derivative guard once offsets are included: {d_raw_init:?}"
        );
    }

    #[test]
    fn identified_time_block_degenerate_entry_preserves_full_dimension() {
        let design_entry = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let design_exit = array![[0.1, 0.5, 0.9], [0.2, 0.6, 1.0], [0.3, 0.7, 1.0]];
        let design_derivative_exit = array![[0.1, 0.1, 0.0], [0.1, 0.1, 0.0], [0.1, 0.1, 0.0]];
        let time_block = TimeBlockInput {
            design_entry: DesignMatrix::from(design_entry.clone()),
            design_exit: DesignMatrix::from(design_exit.clone()),
            design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
            offset_entry: Array1::zeros(3),
            offset_exit: Array1::zeros(3),
            derivative_offset_exit: Array1::from_elem(3, 1e-6),
            structural_monotonicity: true,
            penalties: vec![Array2::eye(3)],
            nullspace_dims: vec![],
            initial_log_lambdas: None,
            initial_beta: None,
        };
        let prepared =
            prepare_identified_time_block(&time_block, 1e-6, 0).expect("prepare time block");
        assert_eq!(prepared.design_entry, design_entry);
        assert_eq!(prepared.design_exit, design_exit);
        assert_eq!(prepared.design_derivative_exit, design_derivative_exit);
    }

    #[test]
    fn resolve_survival_time_anchor_defaults_to_earliest_entry() {
        let age_entry = array![5.0, 1.0, 3.0];
        let anchor = crate::families::survival_construction::resolve_survival_time_anchor_value(
            &age_entry, None,
        )
        .expect("resolve default anchor");
        assert!((anchor - 1.0).abs() <= 1e-12);
    }

    #[test]
    fn survival_ratio_derivatives_prefer_correct_signs() {
        let dists = [
            ResidualDistribution::Gaussian,
            ResidualDistribution::Gumbel,
            ResidualDistribution::Logistic,
        ];
        let zs = [-1.2, -0.5, 0.4, 0.6, 1.1];
        let h = 1e-6_f64;
        let tie_tol = 1e-12_f64;
        let nondeg_tol = 1e-12_f64;
        let mut saw_strict_dr = false;
        let mut saw_strict_ddr = false;

        for &dist in &dists {
            for &z in &zs {
                let r = |u: f64| {
                    let f = dist.pdf(u);
                    let s = 1.0 - dist.cdf(u);
                    f / s
                };
                let dr_plus = |u: f64| {
                    let f = dist.pdf(u);
                    let s = 1.0 - dist.cdf(u);
                    let fp = dist.pdf_derivative(u);
                    let ratio = f / s;
                    (ratio * ratio) + fp / s
                };
                let dr_minus = |u: f64| {
                    let f = dist.pdf(u);
                    let s = 1.0 - dist.cdf(u);
                    let fp = dist.pdf_derivative(u);
                    let ratio = f / s;
                    (ratio * ratio) - fp / s
                };
                let ddr_plus = |u: f64| {
                    let f = dist.pdf(u);
                    let s = 1.0 - dist.cdf(u);
                    let fp = dist.pdf_derivative(u);
                    let fpp = dist.pdfsecond_derivative(u);
                    let ratio = f / s;
                    let dr = (ratio * ratio) + fp / s;
                    (2.0 * ratio * dr) + (fpp / s + fp * f / (s * s))
                };
                let ddr_minus = |u: f64| {
                    let f = dist.pdf(u);
                    let s = 1.0 - dist.cdf(u);
                    let fp = dist.pdf_derivative(u);
                    let fpp = dist.pdfsecond_derivative(u);
                    let ratio = f / s;
                    let dr = (ratio * ratio) - fp / s;
                    (2.0 * ratio * dr) - (fpp / s + fp * f / (s * s))
                };

                let drfd = (r(z + h) - r(z - h)) / (2.0 * h);
                let ddrfd = (dr_plus(z + h) - dr_plus(z - h)) / (2.0 * h);
                let dr_plus_err = (dr_plus(z) - drfd).abs();
                let dr_minus_err = (dr_minus(z) - drfd).abs();
                let ddr_plus_err = (ddr_plus(z) - ddrfd).abs();
                let ddr_minus_err = (ddr_minus(z) - ddrfd).abs();
                let f = dist.pdf(z);
                let s = 1.0 - dist.cdf(z);
                let fp = dist.pdf_derivative(z);
                let fpp = dist.pdfsecond_derivative(z);
                let dr_signal = (fp / s).abs();
                let ddr_signal = (fpp / s + fp * f / (s * s)).abs();

                if dr_signal > nondeg_tol {
                    saw_strict_dr = true;
                    assert!(
                        dr_plus_err + tie_tol < dr_minus_err,
                        "dr sign check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                        dist,
                        z,
                        dr_plus_err,
                        dr_minus_err,
                        dr_signal
                    );
                } else {
                    // At stationary points (fp≈0), plus/minus formulas coincide to first order.
                    assert!(
                        (dr_plus_err - dr_minus_err).abs() <= tie_tol,
                        "dr tie check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                        dist,
                        z,
                        dr_plus_err,
                        dr_minus_err,
                        dr_signal
                    );
                }

                if ddr_signal > nondeg_tol {
                    saw_strict_ddr = true;
                    assert!(
                        ddr_plus_err + tie_tol < ddr_minus_err,
                        "ddr sign check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                        dist,
                        z,
                        ddr_plus_err,
                        ddr_minus_err,
                        ddr_signal
                    );
                } else {
                    assert!(
                        (ddr_plus_err - ddr_minus_err).abs() <= tie_tol,
                        "ddr tie check failed for {:?} at z={}: plus_err={}, minus_err={}, signal={}",
                        dist,
                        z,
                        ddr_plus_err,
                        ddr_minus_err,
                        ddr_signal
                    );
                }
            }
        }

        assert!(
            saw_strict_dr,
            "expected at least one non-degenerate dr check"
        );
        assert!(
            saw_strict_ddr,
            "expected at least one non-degenerate ddr check"
        );
    }

    #[test]
    fn survival_ratio_helper_matches_closed_form_identities() {
        let dists = [
            ResidualDistribution::Gaussian,
            ResidualDistribution::Gumbel,
            ResidualDistribution::Logistic,
        ];
        let zs = [-1.4, -0.7, -0.1, 0.3, 0.9, 1.4];

        for &dist in &dists {
            for &z in &zs {
                let f = dist.pdf(z);
                let s = 1.0 - dist.cdf(z);
                let fp = dist.pdf_derivative(z);
                let fpp = dist.pdfsecond_derivative(z);

                let (r, dr) =
                    SurvivalLocationScaleFamily::survival_ratio_first_derivative(f, fp, s);
                let ddr = SurvivalLocationScaleFamily::survival_ratiosecond_derivative(
                    r, dr, f, fp, fpp, s,
                );

                let r_expected = f / s;
                let dr_expected = (r_expected * r_expected) + fp / s;
                let ddr_expected = (2.0 * r_expected * dr_expected) + (fpp / s + fp * f / (s * s));

                assert!(
                    (r - r_expected).abs() <= 1e-14,
                    "r mismatch for {:?} at z={}: got {}, expected {}",
                    dist,
                    z,
                    r,
                    r_expected
                );
                assert!(
                    (dr - dr_expected).abs() <= 1e-12,
                    "dr mismatch for {:?} at z={}: got {}, expected {}",
                    dist,
                    z,
                    dr,
                    dr_expected
                );
                assert!(
                    (ddr - ddr_expected).abs() <= 1e-10,
                    "ddr mismatch for {:?} at z={}: got {}, expected {}",
                    dist,
                    z,
                    ddr,
                    ddr_expected
                );
            }
        }
    }

    #[test]
    fn residual_pdfthird_derivative_matchessecond_derivativefd() {
        let dists = [
            ResidualDistribution::Gaussian,
            ResidualDistribution::Gumbel,
            ResidualDistribution::Logistic,
        ];
        let zs = [-1.1, -0.4, 0.2, 0.9];
        let h = 1e-6_f64;

        for &dist in &dists {
            for &z in &zs {
                let fd = (dist.pdfsecond_derivative(z + h) - dist.pdfsecond_derivative(z - h))
                    / (2.0 * h);
                let analytic = dist.pdfthird_derivative(z);
                assert_eq!(
                    analytic.signum(),
                    fd.signum(),
                    "pdf''' sign mismatch for {:?} at z={}: analytic={} fd={}",
                    dist,
                    z,
                    analytic,
                    fd
                );
                assert!(
                    (analytic - fd).abs() < 5e-5,
                    "pdf''' mismatch for {:?} at z={}: analytic={} fd={}",
                    dist,
                    z,
                    analytic,
                    fd
                );
            }
        }
    }

    #[test]
    fn exact_log_pdf_derivatives_match_probit_closed_form() {
        let eta = 3.25;
        let (logf, d1, d2, d3, d4) =
            SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(
                &InverseLink::Standard(LinkFunction::Probit),
                eta,
                0.0,
            )
            .expect("exact probit log-pdf derivatives");
        let expected_logf = -0.5 * eta * eta - 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((logf - expected_logf).abs() <= 1e-15);
        assert!((d1 + eta).abs() <= 1e-15);
        assert!((d2 + 1.0).abs() <= 1e-15);
        assert_eq!(d3, 0.0);
        assert_eq!(d4, 0.0);
    }

    #[test]
    fn exact_log_pdf_derivatives_rescaled_scale_cloglog_uniformly() {
        let eta = 501.0;
        let log_scale = 1.0;
        let (logf, d1, d2, d3, d4) =
            SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(
                &InverseLink::Standard(LinkFunction::CLogLog),
                eta,
                log_scale,
            )
            .expect("rescaled cloglog log-pdf derivatives");
        let (unscaled_logf, u1, u2, u3, u4) =
            SurvivalLocationScaleFamily::exact_log_pdf_derivatives_rescaled(
                &InverseLink::Standard(LinkFunction::CLogLog),
                eta,
                0.0,
            )
            .expect("unscaled cloglog log-pdf derivatives");
        let scale = (-log_scale).exp();
        let expected_d1 = scale * u1;
        let expected_d2 = scale * u2;
        let expected_d3 = scale * u3;
        let expected_d4 = scale * u4;

        assert_eq!(logf, unscaled_logf);
        assert!((d1 - expected_d1).abs() <= 1e-12 * expected_d1.abs());
        assert!((d2 - expected_d2).abs() <= 1e-12 * expected_d2.abs());
        assert!((d3 - expected_d3).abs() <= 1e-12 * expected_d3.abs());
        assert!((d4 - expected_d4).abs() <= 1e-12 * expected_d4.abs());
    }

    #[test]
    fn exact_survival_neglog_derivatives_match_identity_closed_form() {
        let eta = 0.25;
        let s = 1.0 - eta;
        let inv = 1.0 / s;
        let (log_s, r, dr, ddr, dddr) =
            SurvivalLocationScaleFamily::exact_survival_neglog_derivatives_fourth_rescaled(
                &InverseLink::Standard(LinkFunction::Identity),
                eta,
                0.0,
            )
            .expect("exact identity survival derivatives");
        assert!((log_s - s.ln()).abs() <= 1e-15);
        assert!((r - inv).abs() <= 1e-15);
        assert!((dr - inv * inv).abs() <= 1e-15);
        assert!((ddr - 2.0 * inv.powi(3)).abs() <= 1e-15);
        assert!((dddr - 6.0 * inv.powi(4)).abs() <= 1e-12);
    }

    #[test]
    fn survival_log_likelihood_only_matches_sum_of_exact_row_kernels() {
        let family = survival_exact_newton_test_family();
        let states =
            survival_exact_newton_rebuild_states(&array![0.1], &array![0.2], &array![-0.15]);
        let (h0, h1, d_raw, ..) = family.validate_joint_states(&states).expect("joint states");
        let dynamic = family
            .build_dynamic_geometry(&states)
            .expect("dynamic geometry");

        let mut row_sum = 0.0;
        for i in 0..family.n {
            let state = family.row_predictor_state(
                h0[i],
                h1[i],
                d_raw[i],
                dynamic.q_entry[i],
                dynamic.q_exit[i],
                dynamic.qdot_exit[i],
            );
            if let Some(kernel) = family.exact_row_kernel(i, state).expect("exact row kernel") {
                row_sum += kernel.log_likelihood();
            }
        }

        let scalar = family
            .log_likelihood_only(&states)
            .expect("scalar log-likelihood");
        assert!(
            (scalar - row_sum).abs() < 1e-12,
            "scalar survival log-likelihood should equal the sum of exact row kernels; scalar={} row_sum={}",
            scalar,
            row_sum
        );
    }

    #[test]
    fn survival_exact_row_kernel_rejects_invalid_event_target_instead_of_clamping() {
        let mut family = survival_exact_newton_test_family();
        family.y[0] = 1.5;
        let states =
            survival_exact_newton_rebuild_states(&array![0.1], &array![0.2], &array![-0.15]);
        let err = match family.log_likelihood_only(&states) {
            Ok(_) => panic!("invalid event target should error"),
            Err(err) => err,
        };
        assert!(
            err.contains("event target must lie in [0,1]"),
            "expected explicit event-target validation error, got: {err}"
        );
    }

    #[test]
    fn logwith_derivatives_positive_matches_exact_log() {
        let x = 0.25;
        let (log_x, d1, d2, d3, d4) = SurvivalLocationScaleFamily::logwith_derivatives_positive(x);
        assert!((log_x - x.ln()).abs() <= 1e-15);
        assert!((d1 - 1.0 / x).abs() <= 1e-15);
        assert!((d2 + 1.0 / (x * x)).abs() <= 1e-15);
        assert!((d3 - 2.0 / (x * x * x)).abs() <= 1e-15);
        assert!((d4 + 6.0 / (x * x * x * x)).abs() <= 1e-12);
    }

    #[test]
    fn inverse_link_survival_prob_complements_failure_prob() {
        let eta = 0.37;
        let failure = inverse_link_failure_prob_checked(
            &residual_distribution_inverse_link(ResidualDistribution::Gaussian),
            eta,
        )
        .expect("failure probability");
        let survival = inverse_link_survival_prob_checked(
            &residual_distribution_inverse_link(ResidualDistribution::Gaussian),
            eta,
        )
        .expect("survival probability");
        assert!((survival - (1.0 - failure)).abs() <= 1e-14);
    }

    #[test]
    fn lift_conditional_covariance_preserveswiggle_block() {
        let z = array![[1.0, 0.0], [0.5, 1.0], [0.0, 1.0]];
        let cov_reduced = array![
            [2.0, 0.1, 0.2, 0.3, 0.4],
            [0.1, 3.0, 0.5, 0.6, 0.7],
            [0.2, 0.5, 4.0, 0.8, 0.9],
            [0.3, 0.6, 0.8, 5.0, 1.1],
            [0.4, 0.7, 0.9, 1.1, 6.0],
        ];
        let lifted = lift_conditional_covariance(&cov_reduced, &z, 1, 1, 1);
        assert_eq!(lifted.dim(), (6, 6));
        assert!((lifted[[5, 5]] - 6.0).abs() <= 1e-12);
        assert!((lifted[[0, 5]] - 0.4).abs() <= 1e-12);
        assert!((lifted[[3, 5]] - 0.9).abs() <= 1e-12);
        assert!((lifted[[4, 5]] - 1.1).abs() <= 1e-12);
    }

    #[test]
    fn weighted_crossprod_dense_falls_back_when_row_scaling_would_overflow() {
        let left = array![[1.0e-200]];
        let right = array![[1.0e200]];
        let weights = array![1.0e200];

        let cross = weighted_crossprod_dense(&left, &weights, &right)
            .expect("stable weighted cross-product should avoid overflow");
        let expected = 1.0e200;
        let rel_err = ((cross[[0, 0]] - expected) / expected).abs();

        assert!(cross[[0, 0]].is_finite());
        assert!(
            rel_err <= 1e-12,
            "unexpected weighted cross-product: {}",
            cross[[0, 0]]
        );
    }

    #[test]
    fn scale_dense_rows_saturates_without_nan_when_coefficients_are_huge() {
        let mat = array![[1.0e200], [2.0e-200]];
        let coeffs = array![1.0e200, 1.0e200];

        let scaled = scale_dense_rows(&mat, &coeffs)
            .expect("row scaling should saturate overflow instead of producing NaN");

        assert!(scaled.iter().all(|value| value.is_finite()));
        assert!(scaled[[0, 0]] > 1.0e300);
        assert!((scaled[[1, 0]] - 2.0).abs() <= 1e-12);
    }

    #[test]
    fn threshold_exact_newton_hessian_matches_negative_gradient_jacobian() {
        let family = survival_exact_newton_test_family();
        let beta_t = 0.35;
        let states = survival_exact_newton_test_states(beta_t);
        let eval = family.evaluate(&states).expect("evaluate at center");
        let BlockWorkingSet::ExactNewton { gradient, hessian } =
            &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD]
        else {
            panic!("threshold block should use exact newton");
        };
        let hessian = hessian.to_dense();

        let eps = 1e-6;
        let eval_plus = family
            .evaluate(&survival_exact_newton_test_states(beta_t + eps))
            .expect("evaluate at beta + eps");
        let eval_minus = family
            .evaluate(&survival_exact_newton_test_states(beta_t - eps))
            .expect("evaluate at beta - eps");
        let grad_plus =
            match &eval_plus.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("threshold block should use exact newton"),
            };
        let grad_minus =
            match &eval_minus.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("threshold block should use exact newton"),
            };
        let fd_neggrad_jac = -(grad_plus - grad_minus) / (2.0 * eps);

        assert!(
            (gradient[0]).is_finite() && hessian[[0, 0]].is_finite(),
            "non-finite threshold exact-newton quantities: grad={} hess={}",
            gradient[0],
            hessian[[0, 0]]
        );
        assert_eq!(
            hessian[[0, 0]].signum(),
            fd_neggrad_jac.signum(),
            "threshold Hessian sign mismatch: analytic={} fd={}",
            hessian[[0, 0]],
            fd_neggrad_jac
        );
        assert!(
            (hessian[[0, 0]] - fd_neggrad_jac).abs() <= 1e-5,
            "threshold Hessian mismatch: analytic={} fd={}",
            hessian[[0, 0]],
            fd_neggrad_jac
        );
    }

    #[test]
    fn log_sigma_exact_newton_hessian_matches_negative_gradient_jacobian() {
        let family = survival_exact_newton_test_familywith_inverse_link(
            residual_distribution_inverse_link(ResidualDistribution::Logistic),
        );
        let beta_time = array![0.2];
        let beta_threshold = array![0.35];
        let beta_log_sigma = array![-0.15];
        let states =
            survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
        let eval = family.evaluate(&states).expect("evaluate at center");
        let BlockWorkingSet::ExactNewton { hessian, .. } =
            &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA]
        else {
            panic!("log-sigma block should use exact newton");
        };
        let hessian = hessian.to_dense();

        let eps = 1e-6;
        let grad_at = |beta_ls: f64| -> f64 {
            let eval = family
                .evaluate(&survival_exact_newton_rebuild_states(
                    &beta_time,
                    &beta_threshold,
                    &array![beta_ls],
                ))
                .expect("evaluate shifted log-sigma");
            match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA] {
                BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                _ => panic!("log-sigma block should use exact newton"),
            }
        };
        let fd_neggrad_jac =
            -(grad_at(beta_log_sigma[0] + eps) - grad_at(beta_log_sigma[0] - eps)) / (2.0 * eps);

        assert_eq!(
            hessian[[0, 0]].signum(),
            fd_neggrad_jac.signum(),
            "log-sigma Hessian sign mismatch: analytic={} fd={}",
            hessian[[0, 0]],
            fd_neggrad_jac
        );
        assert!(
            (hessian[[0, 0]] - fd_neggrad_jac).abs() <= 1e-5,
            "log-sigma Hessian mismatch: analytic={} fd={}",
            hessian[[0, 0]],
            fd_neggrad_jac
        );
    }

    #[test]
    fn exact_newton_block_directional_derivatives_matchfd_for_non_probit_links() {
        let extracthessian = |eval: FamilyEvaluation, block_idx: usize| -> Array2<f64> {
            match &eval.blockworking_sets[block_idx] {
                BlockWorkingSet::ExactNewton { hessian, .. } => hessian.to_dense(),
                BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton block"),
            }
        };

        let beta_time = array![0.2];
        let beta_threshold = array![0.35];
        let beta_log_sigma = array![-0.15];
        let eps = 1e-6;

        for (label, inverse_link) in survival_non_probit_test_links() {
            let family = survival_exact_newton_test_familywith_inverse_link(inverse_link);
            let states =
                survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
            let base_eval = family.evaluate(&states).expect("base eval");

            for (block_idx, direction) in [
                (SurvivalLocationScaleFamily::BLOCK_TIME, array![1.0]),
                (SurvivalLocationScaleFamily::BLOCK_THRESHOLD, array![1.0]),
                (SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA, array![1.0]),
            ] {
                let analytic = family
                    .exact_newton_hessian_directional_derivative(&states, block_idx, &direction)
                    .expect("analytic dH")
                    .expect("expected exact dH");

                let mut beta_time_plus = beta_time.clone();
                let mut beta_threshold_plus = beta_threshold.clone();
                let mut beta_log_sigma_plus = beta_log_sigma.clone();
                match block_idx {
                    SurvivalLocationScaleFamily::BLOCK_TIME => {
                        beta_time_plus += &(eps * &direction);
                    }
                    SurvivalLocationScaleFamily::BLOCK_THRESHOLD => {
                        beta_threshold_plus += &(eps * &direction);
                    }
                    SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA => {
                        beta_log_sigma_plus += &(eps * &direction);
                    }
                    _ => panic!("unexpected block"),
                }

                let plus_states = survival_exact_newton_rebuild_states(
                    &beta_time_plus,
                    &beta_threshold_plus,
                    &beta_log_sigma_plus,
                );
                let h_plus =
                    extracthessian(family.evaluate(&plus_states).expect("plus eval"), block_idx);
                let h_base = extracthessian(base_eval.clone(), block_idx);
                let fd = (h_plus - h_base) / eps;
                crate::testing::assert_matrix_derivativefd(
                    &fd,
                    &analytic,
                    5e-4,
                    &format!("survival {label} block {} dH", block_idx),
                );
            }
        }
    }

    #[test]
    fn joint_exact_newton_hessian_matches_negative_gradient_jacobian_for_non_probit_links() {
        let beta_time = array![0.2];
        let beta_threshold = array![0.35];
        let beta_log_sigma = array![-0.15];
        let eps = 1e-6;

        for (label, inverse_link) in survival_non_probit_test_links() {
            let family = survival_exact_newton_test_familywith_inverse_link(inverse_link);
            let states =
                survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
            let analytic = family
                .exact_newton_joint_hessian(&states)
                .expect("joint exact hessian")
                .expect("expected exact joint hessian");

            let flattengrad = |eval: FamilyEvaluation| -> Array1<f64> {
                let mut out = Array1::<f64>::zeros(3);
                for (block_idx, slot) in out.iter_mut().enumerate() {
                    *slot = match &eval.blockworking_sets[block_idx] {
                        BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                        BlockWorkingSet::Diagonal { .. } => panic!("expected exact newton block"),
                    };
                }
                out
            };

            let mut fd = Array2::<f64>::zeros((3, 3));
            for j in 0..3 {
                let mut beta_time_plus = beta_time.clone();
                let mut beta_threshold_plus = beta_threshold.clone();
                let mut beta_log_sigma_plus = beta_log_sigma.clone();
                let mut beta_time_minus = beta_time.clone();
                let mut beta_threshold_minus = beta_threshold.clone();
                let mut beta_log_sigma_minus = beta_log_sigma.clone();
                match j {
                    0 => {
                        beta_time_plus[0] += eps;
                        beta_time_minus[0] -= eps;
                    }
                    1 => {
                        beta_threshold_plus[0] += eps;
                        beta_threshold_minus[0] -= eps;
                    }
                    2 => {
                        beta_log_sigma_plus[0] += eps;
                        beta_log_sigma_minus[0] -= eps;
                    }
                    _ => unreachable!(),
                }
                let grad_plus = flattengrad(
                    family
                        .evaluate(&survival_exact_newton_rebuild_states(
                            &beta_time_plus,
                            &beta_threshold_plus,
                            &beta_log_sigma_plus,
                        ))
                        .expect("eval plus"),
                );
                let grad_minus = flattengrad(
                    family
                        .evaluate(&survival_exact_newton_rebuild_states(
                            &beta_time_minus,
                            &beta_threshold_minus,
                            &beta_log_sigma_minus,
                        ))
                        .expect("eval minus"),
                );
                let col = -(grad_plus - grad_minus) / (2.0 * eps);
                fd.column_mut(j).assign(&col);
            }

            crate::testing::assert_matrix_derivativefd(
                &fd,
                &analytic,
                2e-4,
                &format!("survival {label} joint H"),
            );
        }
    }

    #[test]
    fn joint_exact_newton_score_matches_loglikelihoodfd_for_non_probit_links() {
        let beta_time = array![0.2];
        let beta_threshold = array![0.35];
        let beta_log_sigma = array![-0.15];
        let eps = 1e-6;

        for (label, inverse_link) in survival_non_probit_test_links() {
            let family = survival_exact_newton_test_familywith_inverse_link(inverse_link);
            let states =
                survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
            let eval = family.evaluate(&states).expect("evaluate");
            let analytic = Array1::from_vec(vec![
                match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_TIME] {
                    BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                    _ => panic!("expected exact newton block"),
                },
                match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD] {
                    BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                    _ => panic!("expected exact newton block"),
                },
                match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA] {
                    BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                    _ => panic!("expected exact newton block"),
                },
            ]);

            let objective = |bt: &Array1<f64>, bth: &Array1<f64>, bls: &Array1<f64>| -> f64 {
                family
                    .evaluate(&survival_exact_newton_rebuild_states(bt, bth, bls))
                    .expect("eval objective")
                    .log_likelihood
            };

            let mut fd = Array1::<f64>::zeros(3);
            fd[0] = (objective(
                &array![beta_time[0] + eps],
                &beta_threshold,
                &beta_log_sigma,
            ) - objective(
                &array![beta_time[0] - eps],
                &beta_threshold,
                &beta_log_sigma,
            )) / (2.0 * eps);
            fd[1] = (objective(
                &beta_time,
                &array![beta_threshold[0] + eps],
                &beta_log_sigma,
            ) - objective(
                &beta_time,
                &array![beta_threshold[0] - eps],
                &beta_log_sigma,
            )) / (2.0 * eps);
            fd[2] = (objective(
                &beta_time,
                &beta_threshold,
                &array![beta_log_sigma[0] + eps],
            ) - objective(
                &beta_time,
                &beta_threshold,
                &array![beta_log_sigma[0] - eps],
            )) / (2.0 * eps);

            for j in 0..3 {
                let abs = (analytic[j] - fd[j]).abs();
                if analytic[j].abs().max(fd[j].abs()) >= 1e-8 {
                    assert_eq!(
                        analytic[j].signum(),
                        fd[j].signum(),
                        "survival {label} joint score sign mismatch at {j}: analytic={} fd={}",
                        analytic[j],
                        fd[j]
                    );
                }
                assert!(
                    abs <= 1e-5,
                    "survival {label} joint score mismatch at {j}: analytic={} fd={} abs={}",
                    analytic[j],
                    fd[j],
                    abs
                );
            }
        }
    }

    #[test]
    fn joint_exact_newton_log_sigma_block_matches_fd_in_far_exp_tail() {
        let family = survival_exact_newton_test_family();
        let beta_time = array![0.2];
        let beta_threshold = array![0.1 * crate::families::sigma_link::safe_exp(700.0)];
        let beta_log_sigma0 = 701.0_f64;
        let beta_log_sigma = array![beta_log_sigma0];

        let states =
            survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
        let eval = family.evaluate(&states).expect("evaluate");
        let (analytic_score, analytic_info) =
            match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA] {
                BlockWorkingSet::ExactNewton { gradient, hessian } => {
                    (gradient[0], hessian.to_dense()[[0, 0]])
                }
                _ => panic!("expected exact newton log-sigma block"),
            };

        let objective = |beta_ls: &Array1<f64>| -> f64 {
            family
                .evaluate(&survival_exact_newton_rebuild_states(
                    &beta_time,
                    &beta_threshold,
                    beta_ls,
                ))
                .expect("eval objective")
                .log_likelihood
        };
        let h = 1e-4;
        let ll_plus = objective(&array![beta_log_sigma0 + h]);
        let ll0 = objective(&array![beta_log_sigma0]);
        let ll_minus = objective(&array![beta_log_sigma0 - h]);
        let score_fd = (ll_plus - ll_minus) / (2.0 * h);
        let info_fd = -(ll_plus - 2.0 * ll0 + ll_minus) / (h * h);
        assert!(
            (analytic_score - score_fd).abs() < 1e-8,
            "the exact-newton survival log-sigma score should match the far-tail finite difference at beta_log_sigma={beta_log_sigma0}; got {} vs {}",
            analytic_score,
            score_fd
        );
        assert!(
            (analytic_info - info_fd).abs() < 1e-5,
            "the exact-newton survival log-sigma information should match the far-tail finite difference at beta_log_sigma={beta_log_sigma0}; got {} vs {}",
            analytic_info,
            info_fd
        );
    }

    #[test]
    fn survival_q_chain_derivatives_match_exact_exp_link_in_far_tails() {
        let eta_t = 2.0;
        for &eta_ls in &[701.0_f64, -30.0_f64] {
            let inv_sigma = exp_sigma_inverse_from_eta_scalar(eta_ls);
            let (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) = q_chain_derivs_scalar(eta_t, eta_ls);
            assert!((q_t + inv_sigma).abs() <= 1e-15);
            assert!((q_ls - eta_t * inv_sigma).abs() <= 1e-15);
            assert!((q_tl - inv_sigma).abs() <= 1e-15);
            assert!((q_ll + eta_t * inv_sigma).abs() <= 1e-15);
            assert!((q_tl_ls + inv_sigma).abs() <= 1e-15);
            assert!((q_ll_ls - eta_t * inv_sigma).abs() <= 1e-15);
            let h = 1e-6;
            let q = |ls: f64| -eta_t * exp_sigma_inverse_from_eta_scalar(ls);
            let q_fd = (q(eta_ls + h) - q(eta_ls - h)) / (2.0 * h);
            assert!(
                (q_ls - q_fd).abs() <= (1e-8 * q_fd.abs()).max(1e-8),
                "q_s finite difference mismatch at eta_ls={eta_ls}: analytic={q_ls} fd={q_fd}"
            );
        }
    }

    #[test]
    fn survival_exact_log_sigma_dh_matches_far_tail_third_derivative() {
        let family = survival_exact_newton_test_family();
        let beta_time = array![0.2];
        let beta_threshold = array![0.1 * crate::families::sigma_link::safe_exp(700.0)];
        let beta_log_sigma0 = 701.0_f64;
        let beta_log_sigma = array![beta_log_sigma0];
        let states =
            survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);

        let analytic = family
            .exact_newton_hessian_directional_derivative(
                &states,
                SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA,
                &array![1.0],
            )
            .expect("analytic dH")
            .expect("expected exact dH");

        let objective = |beta_ls: f64| -> f64 {
            family
                .evaluate(&survival_exact_newton_rebuild_states(
                    &beta_time,
                    &beta_threshold,
                    &array![beta_ls],
                ))
                .expect("eval objective")
                .log_likelihood
        };
        let h = 1e-4_f64;
        let fd3 = (objective(beta_log_sigma0 + 2.0 * h) - 2.0 * objective(beta_log_sigma0 + h)
            + 2.0 * objective(beta_log_sigma0 - h)
            - objective(beta_log_sigma0 - 2.0 * h))
            / (2.0 * h.powi(3));
        assert!(
            (analytic[[0, 0]] + fd3).abs() < 1e-3,
            "the exact-newton survival log-sigma dH entry should equal the negative third derivative in the far tail at beta_log_sigma={beta_log_sigma0}; got analytic {} vs expected {}",
            analytic[[0, 0]],
            -fd3
        );
    }

    #[test]
    fn survival_joint_exact_log_sigma_dh_matches_far_tail_third_derivative() {
        let family = survival_exact_newton_test_family();
        let beta_time = array![0.2];
        let beta_threshold = array![0.1 * crate::families::sigma_link::safe_exp(700.0)];
        let beta_log_sigma0 = 701.0_f64;
        let beta_log_sigma = array![beta_log_sigma0];
        let states =
            survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);

        let analytic = family
            .exact_newton_joint_hessian_directional_derivative(&states, &array![0.0, 0.0, 1.0])
            .expect("analytic joint dH")
            .expect("expected exact joint dH");

        let objective = |beta_ls: f64| -> f64 {
            family
                .evaluate(&survival_exact_newton_rebuild_states(
                    &beta_time,
                    &beta_threshold,
                    &array![beta_ls],
                ))
                .expect("eval objective")
                .log_likelihood
        };
        let h = 1e-4_f64;
        let fd3 = (objective(beta_log_sigma0 + 2.0 * h) - 2.0 * objective(beta_log_sigma0 + h)
            + 2.0 * objective(beta_log_sigma0 - h)
            - objective(beta_log_sigma0 - 2.0 * h))
            / (2.0 * h.powi(3));
        assert!(
            (analytic[[2, 2]] + fd3).abs() < 1e-3,
            "the exact joint survival dH log-sigma/log-sigma entry should equal the negative third derivative in the far tail at beta_log_sigma={beta_log_sigma0}; got analytic {} vs expected {}",
            analytic[[2, 2]],
            -fd3
        );
    }

    #[test]
    fn joint_exact_newton_score_matches_loglikelihoodfd_near_fitted_non_probit_points() {
        let eps = 1e-6;
        let cases = vec![
            (
                "logistic-near-fit",
                residual_distribution_inverse_link(ResidualDistribution::Logistic),
                array![0.7746886451475979],
                array![-0.6407086184606554],
                array![-0.15],
            ),
            (
                "cloglog-near-fit",
                residual_distribution_inverse_link(ResidualDistribution::Gumbel),
                array![0.8153913537182474],
                array![14.123707996892579],
                array![1.4355329717917449],
            ),
        ];

        for (label, inverse_link, beta_time, beta_threshold, beta_log_sigma) in cases {
            let family = survival_exact_newton_test_familywith_inverse_link(inverse_link);
            let states =
                survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
            let eval = family.evaluate(&states).expect("evaluate");
            let analytic = Array1::from_vec(vec![
                match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_TIME] {
                    BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                    _ => panic!("expected exact newton block"),
                },
                match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_THRESHOLD] {
                    BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                    _ => panic!("expected exact newton block"),
                },
                match &eval.blockworking_sets[SurvivalLocationScaleFamily::BLOCK_LOG_SIGMA] {
                    BlockWorkingSet::ExactNewton { gradient, .. } => gradient[0],
                    _ => panic!("expected exact newton block"),
                },
            ]);

            let objective = |bt: &Array1<f64>, bth: &Array1<f64>, bls: &Array1<f64>| -> f64 {
                family
                    .evaluate(&survival_exact_newton_rebuild_states(bt, bth, bls))
                    .expect("eval objective")
                    .log_likelihood
            };

            let mut fd = Array1::<f64>::zeros(3);
            fd[0] = (objective(
                &array![beta_time[0] + eps],
                &beta_threshold,
                &beta_log_sigma,
            ) - objective(
                &array![beta_time[0] - eps],
                &beta_threshold,
                &beta_log_sigma,
            )) / (2.0 * eps);
            fd[1] = (objective(
                &beta_time,
                &array![beta_threshold[0] + eps],
                &beta_log_sigma,
            ) - objective(
                &beta_time,
                &array![beta_threshold[0] - eps],
                &beta_log_sigma,
            )) / (2.0 * eps);
            fd[2] = (objective(
                &beta_time,
                &beta_threshold,
                &array![beta_log_sigma[0] + eps],
            ) - objective(
                &beta_time,
                &beta_threshold,
                &array![beta_log_sigma[0] - eps],
            )) / (2.0 * eps);

            for j in 0..3 {
                let abs = (analytic[j] - fd[j]).abs();
                if analytic[j].abs().max(fd[j].abs()) >= 1e-8 {
                    assert_eq!(
                        analytic[j].signum(),
                        fd[j].signum(),
                        "survival {label} joint score sign mismatch at {j}: analytic={} fd={}",
                        analytic[j],
                        fd[j]
                    );
                }
                assert!(
                    abs <= 5e-4,
                    "survival {label} joint score mismatch at {j}: analytic={} fd={} abs={}",
                    analytic[j],
                    fd[j],
                    abs
                );
            }
        }
    }

    #[test]
    fn row_derivative_identities_hold_for_non_probit_links() {
        let beta_time = array![0.8153913537182474];
        let beta_threshold = array![0.35];
        let beta_log_sigma = array![0.4];

        for (label, inverse_link) in survival_non_probit_test_links() {
            let family = survival_exact_newton_test_familywith_inverse_link(inverse_link);
            let states =
                survival_exact_newton_rebuild_states(&beta_time, &beta_threshold, &beta_log_sigma);
            let (h0, h1, d_raw, eta_t_exit, eta_ls_exit, eta_t_entry, eta_ls_entry, .., etaw) =
                family.validate_joint_states(&states).expect("joint states");
            // For time-invariant blocks, eta_ls_entry == eta_ls_exit.
            let inv_sigma = eta_ls_exit.mapv(exp_sigma_inverse_from_eta_scalar);
            let inv_sigma_entry = eta_ls_entry.mapv(exp_sigma_inverse_from_eta_scalar);

            for i in 0..family.n {
                let state = family.row_predictor_state(
                    h0[i],
                    h1[i],
                    d_raw[i],
                    -eta_t_entry[i] * inv_sigma_entry[i] + etaw.map_or(0.0, |w| w[i]),
                    -eta_t_exit[i] * inv_sigma[i] + etaw.map_or(0.0, |w| w[i]),
                    0.0,
                );
                let row = family
                    .row_derivatives(i, state)
                    .expect("row derivatives")
                    .expect("active row");

                let ell_h0 = row.grad_time_eta_h0;
                let ell_h1 = row.grad_time_eta_h1;
                let ell_q = row.d1_q;
                let ell_h0q = row.h_time_h0;
                let ell_h1q = row.h_time_h1;
                let ell_qq = row.d2_q;
                assert!(
                    (ell_q - ell_h0 - ell_h1).abs() <= 1e-10,
                    "survival {label} row {i} violated ell_q = ell_h0 + ell_h1: q={} h0={} h1={}",
                    ell_q,
                    ell_h0,
                    ell_h1
                );
                assert!(
                    (ell_qq - ell_h0q - ell_h1q).abs() <= 1e-10,
                    "survival {label} row {i} violated ell_qq = ell_h0q + ell_h1q: qq={} h0q={} h1q={}",
                    ell_qq,
                    ell_h0q,
                    ell_h1q
                );
            }
        }
    }

    #[test]
    fn posterior_mean_prediction_matches_deterministicwhen_covariance_iszero() {
        let input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.2],
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0, -0.2
            ]])),
            eta_threshold_offset: array![0.0],
            x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0, 0.3
            ]])),
            eta_log_sigma_offset: array![0.0],
            x_link_wiggle: None,
            link_wiggle_knots: None,
            link_wiggle_degree: None,
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        };
        let fit = test_survival_fit(array![0.4, -0.1], array![0.2, 0.3], array![-0.5, 0.1], None);
        let deterministic = predict_survival_location_scale(&input, &fit).expect("predict");
        let expected =
            inverse_link_survival_prob_checked(&input.inverse_link, deterministic.eta[0])
                .expect("expected survival");
        assert!((deterministic.survival_prob[0] - expected).abs() <= 1e-12);
        let posterior =
            predict_survival_location_scale_posterior_mean(&input, &fit, &Array2::zeros((6, 6)))
                .expect("posterior mean");
        assert!((deterministic.survival_prob[0] - posterior.survival_prob[0]).abs() <= 1e-10);
        let uncertainty = predict_survival_location_scalewith_uncertainty(
            &input,
            &fit,
            &Array2::zeros((6, 6)),
            false,
            true,
        )
        .expect("uncertainty");
        assert!(
            uncertainty
                .response_standard_error
                .as_ref()
                .expect("response sd")[0]
                <= 1e-12
        );
    }

    #[test]
    fn sparse_exact_newton_matches_denseworking_sets() {
        let dense_family = survival_exact_newton_test_family();
        let sparse_family = sparse_survival_exact_newton_test_family();
        let states = survival_exact_newton_test_states(0.35);

        let dense_eval = dense_family.evaluate(&states).expect("dense evaluate");
        let sparse_eval = sparse_family.evaluate(&states).expect("sparse evaluate");
        assert!((dense_eval.log_likelihood - sparse_eval.log_likelihood).abs() <= 1e-12);
        assert_eq!(
            dense_eval.blockworking_sets.len(),
            sparse_eval.blockworking_sets.len()
        );
        for (dense_block, sparse_block) in dense_eval
            .blockworking_sets
            .iter()
            .zip(sparse_eval.blockworking_sets.iter())
        {
            match (dense_block, sparse_block) {
                (
                    BlockWorkingSet::ExactNewton {
                        gradient: dense_g,
                        hessian: dense_h,
                    },
                    BlockWorkingSet::ExactNewton {
                        gradient: sparse_g,
                        hessian: sparse_h,
                    },
                ) => {
                    let dense_h = dense_h.to_dense();
                    let sparse_h = sparse_h.to_dense();
                    assert_eq!(dense_g.len(), sparse_g.len());
                    assert_eq!(dense_h.dim(), sparse_h.dim());
                    for i in 0..dense_g.len() {
                        assert!((dense_g[i] - sparse_g[i]).abs() <= 1e-12);
                    }
                    for i in 0..dense_h.nrows() {
                        for j in 0..dense_h.ncols() {
                            assert!((dense_h[[i, j]] - sparse_h[[i, j]]).abs() <= 1e-12);
                        }
                    }
                }
                _ => panic!("expected exact-newton blocks"),
            }
        }

        let direction = array![0.2];
        let dense_dh = dense_family
            .exact_newton_hessian_directional_derivative(&states, 1, &direction)
            .expect("dense directional derivative")
            .expect("dense threshold directional derivative");
        let sparse_dh = sparse_family
            .exact_newton_hessian_directional_derivative(&states, 1, &direction)
            .expect("sparse directional derivative")
            .expect("sparse threshold directional derivative");
        assert_eq!(dense_dh.dim(), sparse_dh.dim());
        for i in 0..dense_dh.nrows() {
            for j in 0..dense_dh.ncols() {
                assert!((dense_dh[[i, j]] - sparse_dh[[i, j]]).abs() <= 1e-12);
            }
        }
    }

    #[test]
    fn prediction_applies_threshold_and_log_sigma_offsets() {
        let fit = test_survival_fit(array![0.4, -0.1], array![0.2, 0.3], array![-0.5, 0.1], None);
        let input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.2],
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0, -0.2
            ]])),
            eta_threshold_offset: array![0.7],
            x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0, 0.3
            ]])),
            eta_log_sigma_offset: array![0.4],
            x_link_wiggle: None,
            link_wiggle_knots: None,
            link_wiggle_degree: None,
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        };
        let pred = predict_survival_location_scale(&input, &fit).expect("predict");

        let eta_t = array![1.0, -0.2].dot(&fit.beta_threshold()) + input.eta_threshold_offset[0];
        let eta_ls = array![1.0, 0.3].dot(&fit.beta_log_sigma()) + input.eta_log_sigma_offset[0];
        let inv_sigma = exp_sigma_inverse_from_eta_scalar(eta_ls);
        let h = array![1.0, 0.5].dot(&fit.beta_time()) + input.eta_time_offset_exit[0];
        let expected_eta = h - eta_t * inv_sigma;
        let expected_survival =
            inverse_link_survival_prob_checked(&input.inverse_link, expected_eta)
                .expect("expected survival");

        assert!((pred.eta[0] - expected_eta).abs() <= 1e-12);
        assert!((pred.survival_prob[0] - expected_survival).abs() <= 1e-12);
    }

    #[test]
    fn sparse_prediction_and_uncertainty_match_dense() {
        let fit = test_survival_fit(
            array![0.4, -0.1],
            array![0.2, 0.3],
            array![-0.5, 0.1],
            Some(array![0.05, -0.02]),
        );
        let x_threshold_dense = array![[1.0, -0.2], [0.0, 0.6]];
        let x_log_sigma_dense = array![[1.0, 0.3], [0.0, -0.4]];
        let eta_t =
            x_threshold_dense.dot(&fit.beta_threshold()) + Array1::from_vec(vec![0.7, -0.2]);
        let eta_ls =
            x_log_sigma_dense.dot(&fit.beta_log_sigma()) + Array1::from_vec(vec![0.4, 0.1]);
        let q0 = Array1::from_iter(
            eta_t
                .iter()
                .zip(eta_ls.iter())
                .map(|(&t, &ls)| -t * exp_sigma_inverse_from_eta_scalar(ls)),
        );
        let link_wiggle_degree = fit
            .artifacts
            .survival_link_wiggle_degree
            .expect("fit wiggle degree");
        let link_wiggle_knots = fit
            .artifacts
            .survival_link_wiggle_knots
            .clone()
            .expect("fit wiggle knots");
        let xwiggle_dense = survival_wiggle_basis_with_options(
            q0.view(),
            &link_wiggle_knots,
            link_wiggle_degree,
            BasisOptions::value(),
        )
        .expect("link wiggle design");
        let dense_input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5], [1.0, -0.3]],
            eta_time_offset_exit: array![0.2, -0.1],
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                x_threshold_dense.clone(),
            )),
            eta_threshold_offset: array![0.7, -0.2],
            x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                x_log_sigma_dense.clone(),
            )),
            eta_log_sigma_offset: array![0.4, 0.1],
            x_link_wiggle: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                xwiggle_dense.clone(),
            ))),
            link_wiggle_knots: Some(link_wiggle_knots.clone()),
            link_wiggle_degree: Some(link_wiggle_degree),
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        };
        let sparse_input = SurvivalLocationScalePredictInput {
            x_threshold: sparse_design_from_dense(&x_threshold_dense),
            x_log_sigma: sparse_design_from_dense(&x_log_sigma_dense),
            x_link_wiggle: Some(sparse_design_from_dense(&xwiggle_dense)),
            ..dense_input.clone()
        };
        let covariance = array![
            [0.03, 0.01, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
            [0.01, 0.02, 0.0, 0.0, 0.0, 0.0, -0.005, 0.0],
            [0.0, 0.0, 0.04, 0.01, 0.0, 0.0, 0.006, 0.001],
            [0.0, 0.0, 0.01, 0.03, 0.0, 0.0, -0.004, 0.002],
            [0.0, 0.0, 0.0, 0.0, 0.02, 0.005, 0.003, 0.001],
            [0.0, 0.0, 0.0, 0.0, 0.005, 0.02, -0.002, 0.004],
            [0.01, -0.005, 0.006, -0.004, 0.003, -0.002, 0.025, 0.006],
            [0.0, 0.0, 0.001, 0.002, 0.001, 0.004, 0.006, 0.018],
        ];

        let dense_pred =
            predict_survival_location_scale(&dense_input, &fit).expect("dense predict");
        let sparse_pred =
            predict_survival_location_scale(&sparse_input, &fit).expect("sparse predict");
        assert_eq!(dense_pred.eta.len(), sparse_pred.eta.len());
        for i in 0..dense_pred.eta.len() {
            assert!((dense_pred.eta[i] - sparse_pred.eta[i]).abs() <= 1e-12);
            assert!((dense_pred.survival_prob[i] - sparse_pred.survival_prob[i]).abs() <= 1e-12);
        }

        let dense_unc = predict_survival_location_scalewith_uncertainty(
            &dense_input,
            &fit,
            &covariance,
            false,
            true,
        )
        .expect("dense uncertainty");
        let sparse_unc = predict_survival_location_scalewith_uncertainty(
            &sparse_input,
            &fit,
            &covariance,
            false,
            true,
        )
        .expect("sparse uncertainty");
        for i in 0..dense_unc.eta.len() {
            assert!((dense_unc.eta[i] - sparse_unc.eta[i]).abs() <= 1e-12);
            assert!((dense_unc.survival_prob[i] - sparse_unc.survival_prob[i]).abs() <= 1e-12);
            assert!(
                (dense_unc.eta_standard_error[i] - sparse_unc.eta_standard_error[i]).abs() <= 1e-12
            );
            let dense_sd = dense_unc
                .response_standard_error
                .as_ref()
                .expect("dense response sd")[i];
            let sparse_sd = sparse_unc
                .response_standard_error
                .as_ref()
                .expect("sparse response sd")[i];
            assert!((dense_sd - sparse_sd).abs() <= 1e-12);
        }

        let dense_pm =
            predict_survival_location_scale_posterior_mean(&dense_input, &fit, &covariance)
                .expect("dense wiggle posterior mean");
        let sparse_pm =
            predict_survival_location_scale_posterior_mean(&sparse_input, &fit, &covariance)
                .expect("sparse wiggle posterior mean");
        for i in 0..dense_pm.eta.len() {
            assert!((dense_pm.eta[i] - sparse_pm.eta[i]).abs() <= 1e-12);
            assert!((dense_pm.survival_prob[i] - sparse_pm.survival_prob[i]).abs() <= 1e-10);
        }
    }

    #[test]
    fn gaussian_posterior_mean_matches_3d_ghq_small_case() {
        let input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.1],
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0, 0.25
            ]])),
            eta_threshold_offset: array![0.0],
            x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0, -0.15
            ]])),
            eta_log_sigma_offset: array![0.0],
            x_link_wiggle: None,
            link_wiggle_knots: None,
            link_wiggle_degree: None,
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        };
        let fit = test_survival_fit(
            array![0.3, -0.2],
            array![0.1, 0.2],
            array![-0.4, 0.15],
            None,
        );
        let covariance = array![
            [0.03, 0.01, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.02, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.04, 0.01, 0.0, 0.0],
            [0.0, 0.0, 0.01, 0.03, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.02, 0.005],
            [0.0, 0.0, 0.0, 0.0, 0.005, 0.02],
        ];
        let predicted = predict_survival_location_scale_posterior_mean(&input, &fit, &covariance)
            .expect("posterior mean");

        let mu_h = input.x_time_exit.row(0).dot(&fit.beta_time()) + input.eta_time_offset_exit[0];
        let x_t = input.x_threshold.to_dense_arc();
        let x_ls = input.x_log_sigma.to_dense_arc();
        let mu_t = x_t.row(0).dot(&fit.beta_threshold());
        let mu_ls = x_ls.row(0).dot(&fit.beta_log_sigma());
        let cov_hh = covariance.slice(s![0..2, 0..2]).to_owned();
        let cov_tt = covariance.slice(s![2..4, 2..4]).to_owned();
        let cov_ll = covariance.slice(s![4..6, 4..6]).to_owned();
        let cov_ht = covariance.slice(s![0..2, 2..4]).to_owned();
        let cov_hl = covariance.slice(s![0..2, 4..6]).to_owned();
        let cov_tl = covariance.slice(s![2..4, 4..6]).to_owned();
        let var_h = input
            .x_time_exit
            .row(0)
            .dot(&cov_hh.dot(&input.x_time_exit.row(0).to_owned()));
        let var_t = x_t.row(0).dot(&cov_tt.dot(&x_t.row(0).to_owned()));
        let var_ls = x_ls.row(0).dot(&cov_ll.dot(&x_ls.row(0).to_owned()));
        let cov_ht_i = input
            .x_time_exit
            .row(0)
            .dot(&cov_ht.dot(&x_t.row(0).to_owned()));
        let cov_hl_i = input
            .x_time_exit
            .row(0)
            .dot(&cov_hl.dot(&x_ls.row(0).to_owned()));
        let cov_tl_i = x_t.row(0).dot(&cov_tl.dot(&x_ls.row(0).to_owned()));
        let quadctx = crate::quadrature::QuadratureContext::new();
        let ghq = crate::quadrature::normal_expectation_3d_adaptive(
            &quadctx,
            [mu_h, mu_t, mu_ls],
            [
                [var_h, cov_ht_i, cov_hl_i],
                [cov_ht_i, var_t, cov_tl_i],
                [cov_hl_i, cov_tl_i, var_ls],
            ],
            |h, t, ls| {
                inverse_link_survival_probvalue(
                    &input.inverse_link,
                    h - t * exp_sigma_inverse_from_eta_scalar(ls),
                )
            },
        );
        assert!((predicted.survival_prob[0] - ghq).abs() <= 2e-4);
    }

    #[test]
    fn sparse_posterior_mean_matches_dense() {
        let x_threshold_dense = array![[1.0, 0.25], [0.0, -0.1]];
        let x_log_sigma_dense = array![[1.0, -0.15], [0.0, 0.2]];
        let dense_input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5], [1.0, -0.4]],
            eta_time_offset_exit: array![0.1, -0.2],
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                x_threshold_dense.clone(),
            )),
            eta_threshold_offset: array![0.0, 0.05],
            x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                x_log_sigma_dense.clone(),
            )),
            eta_log_sigma_offset: array![0.0, -0.03],
            x_link_wiggle: None,
            link_wiggle_knots: None,
            link_wiggle_degree: None,
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        };
        let sparse_input = SurvivalLocationScalePredictInput {
            x_threshold: sparse_design_from_dense(&x_threshold_dense),
            x_log_sigma: sparse_design_from_dense(&x_log_sigma_dense),
            ..dense_input.clone()
        };
        let fit = test_survival_fit(
            array![0.3, -0.2],
            array![0.1, 0.2],
            array![-0.4, 0.15],
            None,
        );
        let covariance = array![
            [0.03, 0.01, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.02, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.04, 0.01, 0.0, 0.0],
            [0.0, 0.0, 0.01, 0.03, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.02, 0.005],
            [0.0, 0.0, 0.0, 0.0, 0.005, 0.02],
        ];

        let dense_pm =
            predict_survival_location_scale_posterior_mean(&dense_input, &fit, &covariance)
                .expect("dense posterior mean");
        let sparse_pm =
            predict_survival_location_scale_posterior_mean(&sparse_input, &fit, &covariance)
                .expect("sparse posterior mean");
        for i in 0..dense_pm.eta.len() {
            assert!((dense_pm.eta[i] - sparse_pm.eta[i]).abs() <= 1e-12);
            assert!((dense_pm.survival_prob[i] - sparse_pm.survival_prob[i]).abs() <= 1e-10);
        }
    }

    #[test]
    fn wiggle_posterior_mean_matches_exact_nested_4d_quadrature_small_case() {
        let fit = test_survival_fit(
            array![0.4, -0.1],
            array![0.2, 0.3],
            array![-0.5, 0.1],
            Some(array![0.05, -0.02]),
        );
        let x_threshold_dense = array![[1.0, -0.2]];
        let x_log_sigma_dense = array![[1.0, 0.3]];
        let eta_t = x_threshold_dense.dot(&fit.beta_threshold());
        let eta_ls = x_log_sigma_dense.dot(&fit.beta_log_sigma());
        let q0 = Array1::from_iter(
            eta_t
                .iter()
                .zip(eta_ls.iter())
                .map(|(&t, &ls)| -t * exp_sigma_inverse_from_eta_scalar(ls)),
        );
        let link_wiggle_degree = fit
            .artifacts
            .survival_link_wiggle_degree
            .expect("fit wiggle degree");
        let link_wiggle_knots = fit
            .artifacts
            .survival_link_wiggle_knots
            .clone()
            .expect("fit wiggle knots");
        let x_link_wiggle = survival_wiggle_basis_with_options(
            q0.view(),
            &link_wiggle_knots,
            link_wiggle_degree,
            BasisOptions::value(),
        )
        .expect("link wiggle design");
        let input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.2],
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                x_threshold_dense,
            )),
            eta_threshold_offset: array![0.0],
            x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                x_log_sigma_dense,
            )),
            eta_log_sigma_offset: array![0.0],
            x_link_wiggle: Some(DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(
                x_link_wiggle,
            ))),
            link_wiggle_knots: Some(link_wiggle_knots),
            link_wiggle_degree: Some(link_wiggle_degree),
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        };
        let covariance = array![
            [0.03, 0.01, 0.0, 0.0, 0.0, 0.0, 0.01, 0.0],
            [0.01, 0.02, 0.0, 0.0, 0.0, 0.0, -0.005, 0.0],
            [0.0, 0.0, 0.04, 0.01, 0.0, 0.0, 0.006, 0.001],
            [0.0, 0.0, 0.01, 0.03, 0.0, 0.0, -0.004, 0.002],
            [0.0, 0.0, 0.0, 0.0, 0.02, 0.005, 0.003, 0.001],
            [0.0, 0.0, 0.0, 0.0, 0.005, 0.02, -0.002, 0.004],
            [0.01, -0.005, 0.006, -0.004, 0.003, -0.002, 0.025, 0.006],
            [0.0, 0.0, 0.001, 0.002, 0.001, 0.004, 0.006, 0.018],
        ];
        let predicted = predict_survival_location_scale_posterior_mean(&input, &fit, &covariance)
            .expect("wiggle posterior mean");

        let x_t = input.x_threshold.to_dense_arc();
        let x_ls = input.x_log_sigma.to_dense_arc();
        let mu_h = input.x_time_exit.row(0).dot(&fit.beta_time()) + input.eta_time_offset_exit[0];
        let mu_t = x_t.row(0).dot(&fit.beta_threshold()) + input.eta_threshold_offset[0];
        let mu_ls = x_ls.row(0).dot(&fit.beta_log_sigma()) + input.eta_log_sigma_offset[0];
        let cov_hh = covariance.slice(s![0..2, 0..2]).to_owned();
        let cov_tt = covariance.slice(s![2..4, 2..4]).to_owned();
        let cov_ll = covariance.slice(s![4..6, 4..6]).to_owned();
        let cov_ht = covariance.slice(s![0..2, 2..4]).to_owned();
        let cov_hl = covariance.slice(s![0..2, 4..6]).to_owned();
        let cov_hw = covariance.slice(s![0..2, 6..8]).to_owned();
        let cov_tl = covariance.slice(s![2..4, 4..6]).to_owned();
        let cov_tw = covariance.slice(s![2..4, 6..8]).to_owned();
        let cov_lw = covariance.slice(s![4..6, 6..8]).to_owned();
        let var_h = input
            .x_time_exit
            .row(0)
            .dot(&cov_hh.dot(&input.x_time_exit.row(0).to_owned()));
        let var_t = x_t.row(0).dot(&cov_tt.dot(&x_t.row(0).to_owned()));
        let var_ls = x_ls.row(0).dot(&cov_ll.dot(&x_ls.row(0).to_owned()));
        let cov_ht_i = input
            .x_time_exit
            .row(0)
            .dot(&cov_ht.dot(&x_t.row(0).to_owned()));
        let cov_hl_i = input
            .x_time_exit
            .row(0)
            .dot(&cov_hl.dot(&x_ls.row(0).to_owned()));
        let cov_tl_i = x_t.row(0).dot(&cov_tl.dot(&x_ls.row(0).to_owned()));
        let quadctx = crate::quadrature::QuadratureContext::new();
        let cov_htl = [
            [var_h, cov_ht_i, cov_hl_i],
            [cov_ht_i, var_t, cov_tl_i],
            [cov_hl_i, cov_tl_i, var_ls],
        ];
        let htl_factor = factorize_psd_covariance(
            &covariance3_to_array2(cov_htl),
            "wiggle posterior mean test projected covariance",
        )
        .expect("factor projected covariance");
        let cov_wy = {
            let mut out = Array2::<f64>::zeros((2, 3));
            out.column_mut(0)
                .assign(&cov_hw.t().dot(&input.x_time_exit.row(0).to_owned()));
            out.column_mut(1)
                .assign(&cov_tw.t().dot(&x_t.row(0).to_owned()));
            out.column_mut(2)
                .assign(&cov_lw.t().dot(&x_ls.row(0).to_owned()));
            out
        };
        let cov_ww = covariance.slice(s![6..8, 6..8]).to_owned();
        let mut regression = cov_wy.dot(&htl_factor.eigenvectors);
        for col in 0..regression.ncols() {
            let scale = htl_factor.inv_sqrt_eigenvalues[col];
            regression
                .column_mut(col)
                .mapv_inplace(|value| value * scale);
        }
        let cov_cond =
            symmetrize_and_clip_covariance(&(cov_ww - regression.dot(&regression.t().to_owned())));
        let ghq = low_rank_normal_expectation_pair_3d_result(
            &quadctx,
            [mu_h, mu_t, mu_ls],
            cov_htl,
            15,
            "wiggle posterior mean test projected covariance",
            |x, z| {
                let mut cond_beta_w = fit.beta_link_wiggle().expect("wiggle beta");
                for j in 0..cond_beta_w.len() {
                    for (col, &latent) in z.iter().enumerate() {
                        cond_beta_w[j] += regression[[j, col]] * latent;
                    }
                }
                let q0 = survival_q0_from_eta(x[1], x[2]);
                let q0_arr = Array1::from_vec(vec![q0]);
                let basis = survival_wiggle_basis_with_options(
                    q0_arr.view(),
                    input.link_wiggle_knots.as_ref().expect("wiggle knots"),
                    input.link_wiggle_degree.expect("wiggle degree"),
                    BasisOptions::value(),
                )?;
                let b = basis.row(0).to_owned();
                let w_mean = b.dot(&cond_beta_w);
                let w_var = b.dot(&cov_cond.dot(&b)).max(0.0);
                crate::quadrature::normal_expectation_nd_adaptive_result::<1, _, _, String>(
                    &quadctx,
                    [x[0] + q0 + w_mean],
                    [[w_var]],
                    21,
                    |eta| {
                        let p = inverse_link_survival_prob_checked(&input.inverse_link, eta[0])?;
                        Ok((p, p * p))
                    },
                )
            },
        )
        .expect("exact conditional wiggle ghq");
        assert!((predicted.survival_prob[0] - ghq.0).abs() <= 2e-4);
    }

    #[test]
    fn predict_rejects_stateless_beta_logistic_inverse_link() {
        let fit = test_survival_fit(array![0.4, -0.1], array![0.2, 0.3], array![-0.5, 0.1], None);
        let input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.2],
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0, -0.2
            ]])),
            eta_threshold_offset: array![0.0],
            x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0, 0.3
            ]])),
            eta_log_sigma_offset: array![0.0],
            x_link_wiggle: None,
            link_wiggle_knots: None,
            link_wiggle_degree: None,
            inverse_link: InverseLink::Standard(LinkFunction::BetaLogistic),
        };

        let err = predict_survival_location_scale(&input, &fit)
            .err()
            .expect("should reject");
        assert!(err.contains("state-less Standard(BetaLogistic)"));
    }

    #[test]
    fn predict_supports_sas_beta_logistic_and_mixture_links() {
        let fit = test_survival_fit(array![0.4, -0.1], array![0.2, 0.3], array![-0.5, 0.1], None);
        let base = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.2],
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0, -0.2
            ]])),
            eta_threshold_offset: array![0.0],
            x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(array![[
                1.0, 0.3
            ]])),
            eta_log_sigma_offset: array![0.0],
            x_link_wiggle: None,
            link_wiggle_knots: None,
            link_wiggle_degree: None,
            inverse_link: InverseLink::Standard(LinkFunction::Probit),
        };

        let sas = InverseLink::Sas(
            state_from_sasspec(SasLinkSpec {
                initial_epsilon: 0.1,
                initial_log_delta: -0.2,
            })
            .expect("sas state"),
        );
        let beta_logistic = InverseLink::BetaLogistic(
            state_from_beta_logisticspec(SasLinkSpec {
                initial_epsilon: 0.05,
                initial_log_delta: 0.1,
            })
            .expect("beta-logistic state"),
        );
        let mixture = InverseLink::Mixture(
            state_fromspec(&MixtureLinkSpec {
                components: vec![LinkComponent::Probit, LinkComponent::Logit],
                initial_rho: array![0.2],
            })
            .expect("mixture state"),
        );

        for link in [sas, beta_logistic, mixture] {
            let mut input = base.clone();
            input.inverse_link = link;
            let pred = predict_survival_location_scale(&input, &fit).expect("predict");
            assert!(pred.survival_prob[0].is_finite());
            assert!(pred.survival_prob[0] > 0.0 && pred.survival_prob[0] < 1.0);
            let cov = Array2::eye(6) * 1e-3;
            let pm = predict_survival_location_scale_posterior_mean(&input, &fit, &cov)
                .expect("posterior mean");
            assert!(pm.survival_prob[0].is_finite());
            assert!(pm.survival_prob[0] > 0.0 && pm.survival_prob[0] < 1.0);
        }
    }

    /// Full-path structural monotonicity regression for the
    /// heart_failure_survival workflow setup.
    #[test]
    fn heart_failure_full_fit_structural_time_coefficients() {
        // 20 rows with realistic-ish I-spline-like structure.
        let n = 20;
        let p_time = 8; // 8 time basis columns

        // Entry times all near zero (left-truncation at 0) — like __entry=0.
        let age_entry = Array1::from_elem(n, 1e-9_f64);
        // Exit times spread out like real survival data.
        let mut age_exit = Array1::<f64>::zeros(n);
        for i in 0..n {
            age_exit[i] = 4.0 + (i as f64) * 14.0; // 4 to 270
        }

        // Events: ~1/3 event rate.
        let mut event_target = Array1::<f64>::zeros(n);
        for i in [0, 3, 5, 8, 12, 17] {
            event_target[i] = 1.0;
        }
        let weights = Array1::ones(n);

        // Build I-spline-like time designs.
        // Entry design is all zeros (I-spline = 0 below knot range).
        let design_entry = Array2::<f64>::zeros((n, p_time));

        // Exit design: monotonically increasing I-spline-like columns.
        let mut design_exit = Array2::<f64>::zeros((n, p_time));
        for i in 0..n {
            let t = (i as f64) / ((n - 1) as f64); // 0 to 1
            for j in 0..p_time {
                let center = (j as f64 + 0.5) / (p_time as f64);
                // Smooth sigmoid-like I-spline approximation.
                let x = 8.0 * (t - center);
                design_exit[[i, j]] = 1.0 / (1.0 + (-x).exp());
            }
        }

        // Derivative design: derivative of I-spline columns.
        let mut design_derivative_exit = Array2::<f64>::zeros((n, p_time));
        for i in 0..n {
            let t = (i as f64) / ((n - 1) as f64);
            for j in 0..p_time {
                let center = (j as f64 + 0.5) / (p_time as f64);
                let x = 8.0 * (t - center);
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                // Derivative of sigmoid * chain_rule (1/t).
                let deriv = 8.0 * sigmoid * (1.0 - sigmoid);
                let chain = 1.0 / age_exit[i];
                design_derivative_exit[[i, j]] = deriv * chain;
            }
        }

        // The workflow carries the derivative floor in the offsets, so the
        // structural time coefficients only need to stay non-negative.
        let derivative_offset_exit =
            Array1::from_elem(n, DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD);
        let offset_entry = Array1::<f64>::zeros(n);
        let offset_exit = Array1::<f64>::zeros(n);

        // Simple difference penalty.
        let mut penalty = Array2::<f64>::zeros((p_time, p_time));
        for i in 0..(p_time - 1) {
            penalty[[i, i]] += 1.0;
            penalty[[i, i + 1]] -= 1.0;
            penalty[[i + 1, i]] -= 1.0;
            penalty[[i + 1, i + 1]] += 1.0;
        }

        let spec = SurvivalLocationScaleSpec {
            age_entry,
            age_exit,
            event_target,
            weights,
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
            derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
            max_iter: 400,
            tol: 1e-6,
            time_block: TimeBlockInput {
                design_entry: DesignMatrix::from(design_entry),
                design_exit: DesignMatrix::from(design_exit),
                design_derivative_exit: DesignMatrix::from(design_derivative_exit.clone()),
                offset_entry,
                offset_exit,
                derivative_offset_exit: derivative_offset_exit.clone(),
                structural_monotonicity: true,
                penalties: vec![penalty.clone()],
                nullspace_dims: vec![],
                initial_log_lambdas: Some(array![0.0]),
                initial_beta: None,
            },
            threshold_block: CovariateBlockKind::Static(CovariateBlockInput {
                design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::ones(
                    (n, 1),
                ))),
                offset: Array1::zeros(n),
                penalties: Vec::new(),
                nullspace_dims: vec![],
                initial_log_lambdas: None,
                initial_beta: None,
            }),
            log_sigma_block: CovariateBlockKind::Static(CovariateBlockInput {
                design: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::ones(
                    (n, 1),
                ))),
                offset: Array1::zeros(n),
                penalties: Vec::new(),
                nullspace_dims: vec![],
                initial_log_lambdas: None,
                initial_beta: None,
            }),
            timewiggle_block: None,
            linkwiggle_block: None,
        };

        match fit_survival_location_scale(spec) {
            Ok(result) => {
                eprintln!(
                    "fit succeeded: log_likelihood={:.6e}",
                    result.log_likelihood
                );
                eprintln!("beta_time: {:?}", result.beta_time());
                eprintln!("beta_threshold: {:?}", result.beta_threshold());
                eprintln!("beta_log_sigma: {:?}", result.beta_log_sigma());
                // Structural-monotonicity invariant implied by the test's
                // name: the I-spline-like time block carries structural
                // lower bounds of zero (see
                // `structural_time_coefficient_lower_bounds`), and
                // `post_update_block_beta` projects the iterate onto that
                // box after every accepted line-search step.  Every
                // accepted coefficient must therefore satisfy β ≥ 0 — the
                // precondition for the monotone I-spline reconstruction
                // the workflow consumes downstream.
                assert!(
                    result.beta_time().iter().all(|&b| b.is_finite()),
                    "structural time coefficients must be finite: {:?}",
                    result.beta_time(),
                );
                assert!(
                    result.beta_time().iter().all(|&b| b >= 0.0),
                    "structural time coefficients must be non-negative after projection: {:?}",
                    result.beta_time(),
                );
                // Parallel invariant for BLOCK_LINK_WIGGLE: monotone-link
                // wiggle coefficients are structurally non-negative and
                // `post_update_block_beta` clamps them at zero
                // (survival_location_scale.rs:8972-8979).  This test
                // configures `linkwiggle_block: None`, so the block is
                // absent — but if it is ever enabled here the same
                // non-negativity invariant must hold.
                if let Some(beta_link_wiggle) = result.beta_link_wiggle() {
                    assert!(
                        beta_link_wiggle.iter().all(|&b| b.is_finite()),
                        "link-wiggle coefficients must be finite: {beta_link_wiggle:?}",
                    );
                    assert!(
                        beta_link_wiggle.iter().all(|&b| b >= 0.0),
                        "link-wiggle coefficients must be non-negative after projection: {beta_link_wiggle:?}",
                    );
                }
            }
            Err(e) => {
                panic!("fit_survival_location_scale failed: {e}");
            }
        }
    }

    /// Small structural-monotonicity regression for the
    /// heart_failure_survival workflow setup.
    #[test]
    fn heart_failure_structural_time_small() {
        // 6 rows: 3 events, 3 non-events.  Single time column for simplicity.
        let n = 6;
        // I-spline-like designs: entry is all zero (left truncation at t=0),
        // exit has non-trivial values, derivative is the B-spline derivative.
        let x_entry = Array2::<f64>::zeros((n, 2));
        let x_exit = array![
            [0.1, 0.05],
            [0.3, 0.15],
            [0.5, 0.35],
            [0.7, 0.55],
            [0.9, 0.80],
            [1.0, 0.95],
        ];
        let x_deriv = array![
            [0.2, 0.1],
            [0.3, 0.2],
            [0.3, 0.3],
            [0.3, 0.3],
            [0.2, 0.3],
            [0.1, 0.2],
        ];
        let offset_deriv = Array1::from_elem(n, DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD);

        let family = SurvivalLocationScaleFamily {
            n,
            y: array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            w: Array1::ones(n),
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
            derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
            x_time_entry: Arc::new(x_entry),
            x_time_exit: Arc::new(x_exit.clone()),
            x_time_deriv: Arc::new(x_deriv.clone()),
            time_derivative_offset_exit: Arc::new(offset_deriv.clone()),
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            time_coefficient_lower_bounds: Some(array![0.0, 0.0]),
            x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::ones(
                (n, 1),
            ))),
            x_threshold_entry: None,
            x_threshold_deriv: None,
            x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::ones(
                (n, 1),
            ))),
            x_log_sigma_entry: None,
            x_log_sigma_deriv: None,
            x_link_wiggle: None,
            wiggle_knots: None,
            wiggle_degree: None,
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        // Build initial states with beta=0 and a feasible positive derivative offset.
        let states = vec![
            ParameterBlockState {
                beta: Array1::zeros(2),
                eta: {
                    let mut eta = Array1::<f64>::zeros(3 * n);
                    eta.slice_mut(ndarray::s![2 * n..3 * n])
                        .fill(DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD);
                    eta
                },
            },
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: Array1::zeros(n),
            },
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: Array1::zeros(n),
            },
        ];

        // Step 1: Verify initial evaluate succeeds on the feasible domain.
        let eval = family
            .evaluate(&states)
            .expect("initial evaluate with positive d_eta/dt should succeed");
        eprintln!("initial log-likelihood: {:.6e}", eval.log_likelihood);

        // Step 2: Extract time block gradient and Hessian.
        let (grad, hess) = match &eval.blockworking_sets[0] {
            BlockWorkingSet::ExactNewton { gradient, hessian } => {
                (gradient.clone(), hessian.to_dense())
            }
            _ => panic!("expected exact-newton for time block"),
        };
        eprintln!("time block gradient: {:?}", grad);
        eprintln!(
            "time block Hessian diagonal: {:?}",
            (0..hess.nrows()).map(|i| hess[[i, i]]).collect::<Vec<_>>()
        );
        eprintln!("time block Hessian:\n{:.6e}", hess);

        // Step 3: Simulate Newton step (H + ridge*I) * delta = grad - S*beta.
        // With beta=0 and no penalty: (H + ridge*I) * delta = grad.
        let ridge = 1e-6_f64;
        let p = 2;
        let mut lhs = hess.clone();
        for i in 0..p {
            lhs[[i, i]] += ridge;
        }
        // Solve via direct inversion (2x2).
        let det = lhs[[0, 0]] * lhs[[1, 1]] - lhs[[0, 1]] * lhs[[1, 0]];
        eprintln!("LHS determinant: {:.6e}", det);
        let delta = if det.abs() > 1e-30 {
            let inv00 = lhs[[1, 1]] / det;
            let inv01 = -lhs[[0, 1]] / det;
            let inv10 = -lhs[[1, 0]] / det;
            let inv11 = lhs[[0, 0]] / det;
            array![
                inv00 * grad[0] + inv01 * grad[1],
                inv10 * grad[0] + inv11 * grad[1]
            ]
        } else {
            eprintln!("SINGULAR: det={:.6e}", det);
            Array1::zeros(p)
        };
        eprintln!("Newton delta: {:?}", delta);
        assert!(
            delta.iter().all(|v| v.is_finite()),
            "Newton delta has non-finite entries: {:?}",
            delta
        );

        // Step 4: Compute new d_raw after the step.
        let new_d_raw = x_deriv.dot(&delta) + &offset_deriv;
        eprintln!("new d_raw after Newton step: {:?}", new_d_raw);
        for (i, &v) in new_d_raw.iter().enumerate() {
            assert!(
                v.is_finite(),
                "d_raw[{i}] is non-finite ({v}) after Newton step with delta={:?}",
                delta
            );
        }

        // Step 5: Verify evaluate succeeds with the new state.
        let new_eta_time = {
            let mut eta = Array1::<f64>::zeros(3 * n);
            // h0 = x_entry * delta (all zero since x_entry is zero)
            // h1 = x_exit * delta
            let h1 = x_exit.dot(&delta);
            eta.slice_mut(ndarray::s![n..2 * n]).assign(&h1);
            // d_raw = x_deriv * delta + offset_deriv
            eta.slice_mut(ndarray::s![2 * n..3 * n]).assign(&new_d_raw);
            eta
        };
        let new_states = vec![
            ParameterBlockState {
                beta: delta.clone(),
                eta: new_eta_time,
            },
            states[1].clone(),
            states[2].clone(),
        ];
        match family.evaluate(&new_states) {
            Ok(eval2) => eprintln!("post-step log-likelihood: {:.6e}", eval2.log_likelihood),
            Err(e) => {
                eprintln!("post-step evaluate FAILED: {e}");
                eprintln!("delta was: {:?}", delta);
                eprintln!("new d_raw was: {:?}", new_d_raw);
                panic!("evaluate failed after Newton step: {e}");
            }
        }
    }

    #[test]
    fn evaluate_survival_location_scale_rejects_non_finite_d_eta_dt() {
        let n = 2;
        let family = SurvivalLocationScaleFamily {
            n,
            y: array![1.0, 0.0],
            w: Array1::ones(n),
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
            derivative_guard: DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
            x_time_entry: Arc::new(Array2::zeros((n, 1))),
            x_time_exit: Arc::new(Array2::ones((n, 1))),
            x_time_deriv: Arc::new(Array2::ones((n, 1))),
            time_derivative_offset_exit: Arc::new(Array1::from_elem(
                n,
                DEFAULT_SURVIVAL_LOCATION_SCALE_DERIVATIVE_GUARD,
            )),
            time_wiggle_knots: None,
            time_wiggle_degree: None,
            time_wiggle_ncols: 0,
            time_coefficient_lower_bounds: Some(array![0.0]),
            x_threshold: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::ones(
                (n, 1),
            ))),
            x_threshold_entry: None,
            x_threshold_deriv: None,
            x_log_sigma: DesignMatrix::Dense(crate::matrix::DenseDesignMatrix::from(Array2::ones(
                (n, 1),
            ))),
            x_log_sigma_entry: None,
            x_log_sigma_deriv: None,
            x_link_wiggle: None,
            wiggle_knots: None,
            wiggle_degree: None,
            policy: crate::resource::ResourcePolicy::default_library(),
        };

        let mut eta_time = Array1::<f64>::zeros(3 * n);
        eta_time[2 * n] = f64::NAN;
        eta_time[2 * n + 1] = 0.25;
        let states = vec![
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: eta_time,
            },
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: Array1::zeros(n),
            },
            ParameterBlockState {
                beta: Array1::zeros(1),
                eta: Array1::zeros(n),
            },
        ];

        let eval = match family.evaluate(&states) {
            Ok(_) => panic!("non-finite d_eta/dt must be rejected"),
            Err(err) => err,
        };
        assert!(eval.contains("non-finite"));
    }

    #[test]
    fn q_chain_derivatives_match_exact_exp_link_in_lower_tail() {
        let eta_t = 2.0;
        let eta_ls = -30.0;
        let q = |ls: f64| -eta_t * exp_sigma_inverse_from_eta_scalar(ls);
        let h = 1e-6;
        let q_left = q(eta_ls - h);
        let q_mid = q(eta_ls);
        let q_right = q(eta_ls + h);
        assert!(
            q_left != q_mid && q_right != q_mid,
            "exact exp-link q should remain eta_ls-sensitive in the lower tail"
        );

        let (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) = q_chain_derivs_scalar(eta_t, eta_ls);
        let inv_sigma = exp_sigma_inverse_from_eta_scalar(eta_ls);
        assert!((q_t + inv_sigma).abs() <= 1e-15);
        assert!((q_ls - eta_t * inv_sigma).abs() <= 1e-15);
        assert!((q_tl - inv_sigma).abs() <= 1e-15);
        assert!((q_ll + eta_t * inv_sigma).abs() <= 1e-15);
        assert!((q_tl_ls + inv_sigma).abs() <= 1e-15);
        assert!((q_ll_ls - eta_t * inv_sigma).abs() <= 1e-15);
    }

    #[test]
    fn survival_q0dot_from_base_preserves_far_tail_cancellation() {
        let eta_t = 1e-10;
        let eta_ls = -700.0;
        let eta_t_deriv = 1.0 - 1e-12;
        let eta_ls_deriv = 1e10;
        let base = survival_base_q_scalars(eta_t, eta_ls);

        let factorized = survival_q0dot_from_base(base, eta_t_deriv, eta_ls_deriv);
        let expected = safe_product(
            exp_sigma_inverse_from_eta_scalar(eta_ls),
            eta_t.mul_add(eta_ls_deriv, -eta_t_deriv),
        );
        let expanded = safe_sum2(
            safe_product(base.q_t, eta_t_deriv),
            safe_product(base.q_ls, eta_ls_deriv),
        );

        assert!(factorized.is_finite());
        assert!(expected.is_finite());
        assert!(
            (factorized - expected).abs() <= 1e-12 * expected.abs().max(1.0),
            "factorized qdot mismatch: got {factorized}, expected {expected}"
        );
        assert!(expanded.abs() >= 1e200);
        assert!(factorized.abs() <= 1e206);
    }

    #[test]
    fn compensated_difference_carries_explicit_roundoff_bound() {
        let lhs = 1.0e217 + 1.0e201;
        let rhs = 1.0e217;
        let diff = compensated_difference(lhs, rhs);

        assert!(diff.value.is_finite());
        assert!(diff.roundoff_slack.is_finite());
        assert!(diff.roundoff_slack >= 0.0);
        assert!(diff.operand_scale >= rhs.abs());
    }

    #[test]
    fn logistic_residual_tail_derivatives_should_match_stable_closed_forms() {
        let z = 50.0_f64;
        let e = (-z).exp();
        let denom = 1.0_f64 + e;
        let stable_pdf = e / denom.powi(2);
        let stable_d1 = e * (e - 1.0) / denom.powi(3);
        let stable_d2 = e * (e * e - 4.0 * e + 1.0) / denom.powi(4);
        let stable_d3 = e * (e * e * e - 11.0 * e * e + 11.0 * e - 1.0) / denom.powi(5);

        let dist = ResidualDistribution::Logistic;
        assert!(
            (dist.pdf(z) - stable_pdf).abs() < 1e-30,
            "logistic residual pdf should equal the stable tail formula at z={z}; got {} vs {}",
            dist.pdf(z),
            stable_pdf
        );
        assert!(
            (dist.pdf_derivative(z) - stable_d1).abs() < 1e-30,
            "logistic residual pdf' should equal the stable tail formula at z={z}; got {} vs {}",
            dist.pdf_derivative(z),
            stable_d1
        );
        assert!(
            (dist.pdfsecond_derivative(z) - stable_d2).abs() < 1e-30,
            "logistic residual pdf'' should equal the stable tail formula at z={z}; got {} vs {}",
            dist.pdfsecond_derivative(z),
            stable_d2
        );
        assert!(
            (dist.pdfthird_derivative(z) - stable_d3).abs() < 1e-30,
            "logistic residual pdf''' should equal the stable tail formula at z={z}; got {} vs {}",
            dist.pdfthird_derivative(z),
            stable_d3
        );
    }

    #[test]
    fn gumbel_cdf_negative_tail_should_match_expm1_form() {
        let z = -50.0_f64;
        let ez = z.exp();
        let stable_cdf = -(-ez).exp_m1();
        let dist = ResidualDistribution::Gumbel;
        assert!(stable_cdf > 0.0);
        assert!(
            (dist.cdf(z) - stable_cdf).abs() < 1e-30,
            "gumbel cdf should equal -expm1(-exp(z)) in the negative tail at z={z}; got {} vs {}",
            dist.cdf(z),
            stable_cdf
        );
    }

    #[test]
    fn probit_survival_helper_matches_upper_tail_probability() {
        let eta = 10.0_f64;
        let stable_survival = 0.5 * statrs::function::erf::erfc(eta / std::f64::consts::SQRT_2);
        assert!(stable_survival > 0.0);
        let helper =
            inverse_link_survival_probvalue(&InverseLink::Standard(LinkFunction::Probit), eta);
        assert!(
            (helper - stable_survival).abs() < 1e-30,
            "probit survival helper should use the upper-tail erfc form at eta={eta}; got {} vs {}",
            helper,
            stable_survival
        );
    }

    #[test]
    fn cloglog_survival_helper_matches_negative_tail_function() {
        let eta = -100.0_f64;
        let stable_survival = (-(eta.exp())).exp();
        let helper =
            inverse_link_survival_probvalue(&InverseLink::Standard(LinkFunction::CLogLog), eta);
        assert_eq!(stable_survival, 1.0);
        assert!(
            (helper - stable_survival).abs() < 1e-30,
            "cloglog survival helper should evaluate exp(-exp(eta)) itself, not a clamped surrogate, at eta={eta}; got {} vs {}",
            helper,
            stable_survival
        );
    }

    #[test]
    fn positive_log_cumulative_hazard_maps_to_baseline_cloglog_survival() {
        let cumulative_hazard = 4.0_f64;
        let eta = cumulative_hazard.ln();
        let survival =
            inverse_link_survival_probvalue(&InverseLink::Standard(LinkFunction::CLogLog), eta);
        let expected = (-cumulative_hazard).exp();
        assert!(
            (survival - expected).abs() < 1e-15,
            "baseline cloglog survival should be exp(-H0) when eta = log(H0); got {} vs {}",
            survival,
            expected
        );
    }
}
