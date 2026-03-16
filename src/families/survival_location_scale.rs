use crate::custom_family::{
    BlockWorkingSet, BlockwiseFitOptions, CustomFamily, CustomFamilyBlockPsiDerivative,
    CustomFamilyWarmStart, ExactNewtonJointPsiSecondOrderTerms, ExactNewtonJointPsiTerms,
    FamilyEvaluation, ParameterBlockSpec, ParameterBlockState, evaluate_custom_family_joint_hyper,
    fit_custom_family, resolve_custom_family_x_psi, resolve_custom_family_x_psi_psi,
};
use crate::faer_ndarray::{default_rrqr_rank_alpha, fast_xt_diag_x, rrqr_nullspace_basis};
use crate::families::gamlss::{
    SelectedWiggleBasis, WiggleBlockConfig, select_wiggle_basis_from_seed,
};
use crate::families::scale_design::{
    apply_scale_deviation_transform, build_scale_deviation_transform, infer_non_intercept_start,
};
use crate::families::sigma_link::{
    exp_sigma_derivs_up_to_fourth, exp_sigma_derivs_up_to_third,
    exp_sigma_derivs_up_to_third_scalar,
};
use crate::matrix::{
    DesignMatrix, EmbeddedColumnBlock, EmbeddedSquareBlock, SymmetricMatrix, xt_diag_x_symmetric,
};
use crate::mixture_link::{
    inverse_link_jet_for_inverse_link, inverse_link_pdfthird_derivative_for_inverse_link,
};
use crate::pirls::LinearInequalityConstraints;
use crate::probability::{normal_cdf, normal_pdf};
use crate::smooth::{
    SpatialLengthScaleOptimizationOptions, SpatialLogKappaCoords, TermCollectionDesign,
    TermCollectionSpec, TwoBlockExactJointHyperSetup, build_term_collection_design,
    freeze_spatial_length_scale_terms_from_design,
    optimize_two_block_spatial_length_scale_exact_joint, spatial_length_scale_term_indices,
    try_build_spatial_log_kappa_derivativeinfo_list,
};
use crate::solver::estimate::UnifiedFitResult;
use crate::solver::estimate::{
    FitGeometry, ensure_finite_scalar_estimation, validate_all_finite_estimation,
};
use crate::terms::construction::kronecker_product;
use crate::types::{InverseLink, LinkFunction};
use ndarray::{Array1, Array2, Axis, s};

const MIN_PROB: f64 = 1e-12;

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
    /// F_αβγδ via the Faà di Bruno chain rule. That chain rule's leading term
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
                // F(z)=1-exp(-exp(z))
                if z == f64::INFINITY {
                    return 1.0;
                }
                if z == f64::NEG_INFINITY {
                    return 0.0;
                }
                if z > 700.0 {
                    return 1.0;
                }
                let ez = z.exp();
                1.0 - (-ez).exp()
            }
            ResidualDistribution::Logistic => {
                if z == f64::INFINITY {
                    1.0
                } else if z == f64::NEG_INFINITY {
                    0.0
                } else if z >= 0.0 {
                    let e = (-z).exp();
                    1.0 / (1.0 + e)
                } else {
                    let e = z.exp();
                    e / (1.0 + e)
                }
            }
        }
    }

    fn pdf(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => normal_pdf(z),
            ResidualDistribution::Gumbel => {
                if z.is_infinite() {
                    return 0.0;
                }
                let log_f = z - z.exp();
                if log_f < -745.0 { 0.0 } else { log_f.exp() }
            }
            ResidualDistribution::Logistic => {
                let s = self.cdf(z);
                s * (1.0 - s)
            }
        }
    }

    fn pdf_derivative(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => -z * normal_pdf(z),
            ResidualDistribution::Gumbel => {
                if z.is_infinite() {
                    return 0.0;
                }
                let log_f = z - z.exp();
                if log_f < -745.0 {
                    return 0.0;
                }
                let f = log_f.exp();
                let ez = z.clamp(-700.0, 700.0).exp();
                f * (1.0 - ez)
            }
            ResidualDistribution::Logistic => {
                let s = self.cdf(z);
                let f = s * (1.0 - s);
                f * (1.0 - 2.0 * s)
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
                if z.is_infinite() {
                    return 0.0;
                }
                let log_f = z - z.exp();
                if log_f < -745.0 {
                    return 0.0;
                }
                let f = log_f.exp();
                let ez = z.clamp(-700.0, 700.0).exp();
                f * (1.0 - 3.0 * ez + ez * ez)
            }
            ResidualDistribution::Logistic => {
                let s = self.cdf(z);
                let f = s * (1.0 - s);
                f * (1.0 - 6.0 * s + 6.0 * s * s)
            }
        }
    }

    fn pdfthird_derivative(&self, z: f64) -> f64 {
        match self {
            ResidualDistribution::Gaussian => {
                let f = normal_pdf(z);
                -(z * z * z - 3.0 * z) * f
            }
            ResidualDistribution::Gumbel => {
                if z.is_infinite() {
                    return 0.0;
                }
                let log_f = z - z.exp();
                if log_f < -745.0 {
                    return 0.0;
                }
                let f = log_f.exp();
                let ez = z.clamp(-700.0, 700.0).exp();
                f * (1.0 - 7.0 * ez + 6.0 * ez * ez - ez * ez * ez)
            }
            ResidualDistribution::Logistic => {
                let s = self.cdf(z);
                let f = s * (1.0 - s);
                f * (1.0 - 14.0 * s + 36.0 * s * s - 24.0 * s * s * s)
            }
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
            ResidualDistribution::Gumbel => {
                if z.is_infinite() {
                    return 0.0;
                }
                let log_f = z - z.exp();
                if log_f < -745.0 {
                    return 0.0;
                }
                let f = log_f.exp();
                let ez = z.clamp(-700.0, 700.0).exp();
                let ez2 = ez * ez;
                // Q_4(e) = 1 - 15e + 25e² - 10e³ + e⁴
                f * (1.0 - 15.0 * ez + 25.0 * ez2 - 10.0 * ez2 * ez + ez2 * ez2)
            }
            ResidualDistribution::Logistic => {
                let s = self.cdf(z);
                let f = s * (1.0 - s);
                let s2 = s * s;
                // P_4(s) = 1 - 30s + 150s² - 240s³ + 120s⁴
                f * (1.0 - 30.0 * s + 150.0 * s2 - 240.0 * s2 * s + 120.0 * s2 * s2)
            }
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
/// in turn enters the m4 ingredient of the Faà di Bruno chain rule for
/// the outer REML Hessian Q[v_k, v_l] term.
///
/// Currently supports the three standard survival residual distributions:
/// - Probit (Gaussian): f''''(u) = (u⁴ - 6u² + 3) φ(u)
/// - Logit (Logistic):  f''''(u) = s(1-s)(1 - 30s + 150s² - 240s³ + 120s⁴)
/// - CLogLog (Gumbel):  f''''(u) = f · (1 - 15e^u + 25e^{2u} - 10e^{3u} + e^{4u})
///
/// For mixture or non-standard links, this returns 0.0 as a conservative
/// fallback (the 4th-order correction is then incomplete but the lower-order
/// terms remain correct).
fn inverse_link_pdffourth_derivative(inverse_link: &InverseLink, eta: f64) -> f64 {
    match inverse_link {
        InverseLink::Standard(LinkFunction::Probit) => {
            ResidualDistribution::Gaussian.pdffourth_derivative(eta)
        }
        InverseLink::Standard(LinkFunction::Logit) => {
            ResidualDistribution::Logistic.pdffourth_derivative(eta)
        }
        InverseLink::Standard(LinkFunction::CLogLog) => {
            ResidualDistribution::Gumbel.pdffourth_derivative(eta)
        }
        // Conservative fallback: for non-standard links the 4th-order
        // correction term m1·u_αβγδ is dropped. The remaining terms
        // (m4·u products, m3·u products, m2·u products) are still present
        // because they only need m1..m3 and u through 3rd order.
        _ => 0.0,
    }
}

#[derive(Clone)]
pub struct TimeBlockInput {
    pub design_entry: Array2<f64>,
    pub design_exit: Array2<f64>,
    pub design_derivative_exit: Array2<f64>,
    pub offset_entry: Array1<f64>,
    pub offset_exit: Array1<f64>,
    pub derivative_offset_exit: Array1<f64>,
    pub penalties: Vec<Array2<f64>>,
    pub initial_log_lambdas: Option<Array1<f64>>,
    pub initial_beta: Option<Array1<f64>>,
}

#[derive(Clone)]
pub struct CovariateBlockInput {
    pub design: DesignMatrix,
    pub offset: Array1<f64>,
    pub penalties: Vec<Array2<f64>>,
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
    /// Combined Kronecker penalties for the tensor product.
    pub penalties: Vec<Array2<f64>>,
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
    pub penalties: Vec<Array2<f64>>,
    pub initial_log_lambdas: Option<Array1<f64>>,
    pub initial_beta: Option<Array1<f64>>,
}

#[derive(Clone)]
struct SurvivalLocationScaleSpec {
    pub age_entry: Array1<f64>,
    pub age_exit: Array1<f64>,
    pub event_target: Array1<f64>,
    pub weights: Array1<f64>,
    pub inverse_link: InverseLink,
    pub derivative_guard: f64,
    pub derivative_softness: f64,
    /// Optional anchor time for identifiability of h(t).
    ///
    /// If `None`, the model anchors at the earliest observed entry time.
    /// If `Some(t_anchor)`, the nearest observed entry-time row is used.
    pub time_anchor: Option<f64>,
    pub max_iter: usize,
    pub tol: f64,
    pub time_block: TimeBlockInput,
    pub threshold_block: CovariateBlockKind,
    pub log_sigma_block: CovariateBlockKind,
    pub linkwiggle_block: Option<LinkWiggleBlockInput>,
}

#[derive(Clone)]
pub enum SurvivalCovariateTermBlockTemplate {
    Static,
    TimeVarying {
        time_basis_entry: Array2<f64>,
        time_basis_exit: Array2<f64>,
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
    pub derivative_guard: f64,
    pub derivative_softness: f64,
    pub time_anchor: Option<f64>,
    pub max_iter: usize,
    pub tol: f64,
    pub time_block: TimeBlockInput,
    pub thresholdspec: TermCollectionSpec,
    pub log_sigmaspec: TermCollectionSpec,
    pub threshold_template: SurvivalCovariateTermBlockTemplate,
    pub log_sigma_template: SurvivalCovariateTermBlockTemplate,
    pub linkwiggle_block: Option<LinkWiggleBlockInput>,
}

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
    crate::solver::estimate::UnifiedFitResult::try_from_parts(UnifiedFitResultParts {
        blocks,
        log_lambdas,
        lambdas: Array1::from_vec(all_lambdas),
        log_likelihood,
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
        artifacts: crate::solver::estimate::FitArtifacts { pirls: None },
        inner_cycles: 0,
    })
    .map_err(|e| e.to_string())
}

#[derive(Clone)]
pub struct SurvivalLocationScalePredictInput {
    pub x_time_exit: Array2<f64>,
    pub eta_time_offset_exit: Array1<f64>,
    pub x_threshold: DesignMatrix,
    pub eta_threshold_offset: Array1<f64>,
    pub x_log_sigma: DesignMatrix,
    pub eta_log_sigma_offset: Array1<f64>,
    pub x_link_wiggle: Option<DesignMatrix>,
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
    derivative_softness: f64,
    x_time_entry: Array2<f64>,
    x_time_exit: Array2<f64>,
    x_time_deriv: Array2<f64>,
    offset_time_deriv: Array1<f64>,
    /// Exit design for threshold block (always present; used as main design).
    x_threshold: DesignMatrix,
    /// Entry design for threshold block when time-varying.
    /// When `None`, the block is time-invariant: q0 = q1 (current behavior).
    x_threshold_entry: Option<DesignMatrix>,
    /// Exit design for log-sigma block (always present; used as main design).
    x_log_sigma: DesignMatrix,
    /// Entry design for log-sigma block when time-varying.
    x_log_sigma_entry: Option<DesignMatrix>,
    x_link_wiggle: Option<DesignMatrix>,
}

#[derive(Clone, Copy)]
struct SurvivalPredictorState {
    h0: f64,
    h1: f64,
    d_raw: f64,
    /// q evaluated at entry time. When the threshold/sigma blocks are
    /// time-invariant, q0 == q1.
    q0: f64,
    /// q evaluated at exit time.
    q1: f64,
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
    /// d⁴ ell / dq⁴ summed.
    ///
    /// This is the m4 quantity from the Faà di Bruno chain rule. It is needed
    /// by the outer REML Hessian's Q[v_k, v_l] term: the 4th-order composed
    /// derivative F_αβγδ = m4·u_α·u_β·u_γ·u_δ + ... uses this as the leading
    /// coefficient. See response.md Section 6 for the derivation confirming
    /// that 3rd-order chain terms in q are insufficient for the outer Hessian.
    d4_q: f64,
    /// Entry-only derivative: d ell / dq0 = w * r(u0).
    d1_q0: f64,
    /// Entry-only second derivative: d² ell / dq0² = w * r'(u0).
    d2_q0: f64,
    /// Entry-only third derivative: d³ ell / dq0³ = w * r''(u0).
    d3_q0: f64,
    /// Entry-only fourth derivative: d⁴ ell / dq0⁴ = w * r'''(u0).
    ///
    /// The 3rd derivative of the survival ratio r = f/S enters the (s,s,s,s)
    /// and (ϑ,s,s,s) blocks of the outer Hessian drift via the 4th-order
    /// Faà di Bruno formula.
    d4_q0: f64,
    /// Exit-only derivative: d ell / dq1.
    d1_q1: f64,
    /// Exit-only second derivative: d² ell / dq1².
    d2_q1: f64,
    /// Exit-only third derivative: d³ ell / dq1³.
    d3_q1: f64,
    /// Exit-only fourth derivative: d⁴ ell / dq1⁴.
    ///
    /// Analogous to d4_q0 but for the exit side.
    d4_q1: f64,
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

struct SurvivalJointQuantities {
    d1_q: Array1<f64>,
    d2_q: Array1<f64>,
    d3_q: Array1<f64>,
    /// d⁴ℓ/dq⁴ summed over entry+exit.
    ///
    /// The m4 quantity needed by the 4th-order Faà di Bruno chain rule for the
    /// outer REML Hessian's Q[v_k, v_l] term. See response.md Section 6.
    d4_q: Array1<f64>,
    /// Entry-only derivatives of ell w.r.t. q0.
    d1_q0: Array1<f64>,
    d2_q0: Array1<f64>,
    d3_q0: Array1<f64>,
    /// Entry-only 4th derivative: d⁴ℓ/dq0⁴.
    d4_q0: Array1<f64>,
    /// Exit-only derivatives of ell w.r.t. q1.
    d1_q1: Array1<f64>,
    d2_q1: Array1<f64>,
    d3_q1: Array1<f64>,
    /// Exit-only 4th derivative: d⁴ℓ/dq1⁴.
    d4_q1: Array1<f64>,
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
    /// Exit-side dq/d(eta_t) = -1/sigma_exit.
    dq_t: Array1<f64>,
    /// Exit-side dq/d(eta_ls).
    dq_ls: Array1<f64>,
    d2q_tls: Array1<f64>,
    d2q_ls: Array1<f64>,
    d3q_tls_ls: Array1<f64>,
    d3q_ls: Array1<f64>,
    /// Entry-side dq0/d(eta_t_entry) = -1/sigma_entry (only for time-varying).
    dq_t_entry: Option<Array1<f64>>,
    /// Entry-side chain-rule derivatives for sigma at entry (only for time-varying sigma).
    dq_ls_entry: Option<Array1<f64>>,
    d2q_tls_entry: Option<Array1<f64>>,
    d2q_ls_entry: Option<Array1<f64>>,
    d3q_tls_ls_entry: Option<Array1<f64>>,
    d3q_ls_entry: Option<Array1<f64>>,
}

struct SurvivalJointPsiDirection {
    block_idx: usize,
    local_idx: usize,
    x_t_exit_psi: Array2<f64>,
    x_t_entry_psi: Array2<f64>,
    x_ls_exit_psi: Array2<f64>,
    x_ls_entry_psi: Array2<f64>,
    z_t_exit_psi: Array1<f64>,
    z_t_entry_psi: Array1<f64>,
    z_ls_exit_psi: Array1<f64>,
    z_ls_entry_psi: Array1<f64>,
}

fn split_survival_psi_design(
    x_psi: &Array2<f64>,
    n: usize,
    time_varying: bool,
    label: &str,
) -> Result<(Array2<f64>, Array2<f64>), String> {
    if time_varying {
        if x_psi.nrows() != 2 * n {
            return Err(format!(
                "{label} stacked psi design row mismatch: got {}, expected {}",
                x_psi.nrows(),
                2 * n
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

    /// Returns `(h0, h1, d_raw, eta_t_exit, eta_ls_exit, eta_t_entry, eta_ls_entry, etaw)`.
    ///
    /// For time-invariant blocks, `eta_t_entry == eta_t_exit` and likewise for ls.
    /// For time-varying blocks, the block eta is 2n long: `[exit; entry]`.
    /// The solver's ParameterBlockSpec uses the EXIT design, so exit comes first
    /// in the stacked eta.
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
        // For time-varying blocks the stacked design is [exit_design; entry_design],
        // giving eta of length 2n. For time-invariant blocks eta is length n.
        let (eta_t_exit, eta_t_entry) = if self.x_threshold_entry.is_some() {
            if eta_t_raw.len() != 2 * n {
                return Err(format!(
                    "time-varying threshold eta length mismatch: got {}, expected {}",
                    eta_t_raw.len(),
                    2 * n
                ));
            }
            (eta_t_raw.slice(s![0..n]), eta_t_raw.slice(s![n..2 * n]))
        } else {
            if eta_t_raw.len() != n {
                return Err(format!(
                    "threshold eta length mismatch: got {}, expected {n}",
                    eta_t_raw.len()
                ));
            }
            (eta_t_raw.slice(s![0..n]), eta_t_raw.slice(s![0..n]))
        };
        let (eta_ls_exit, eta_ls_entry) = if self.x_log_sigma_entry.is_some() {
            if eta_ls_raw.len() != 2 * n {
                return Err(format!(
                    "time-varying log-sigma eta length mismatch: got {}, expected {}",
                    eta_ls_raw.len(),
                    2 * n
                ));
            }
            (eta_ls_raw.slice(s![0..n]), eta_ls_raw.slice(s![n..2 * n]))
        } else {
            if eta_ls_raw.len() != n {
                return Err(format!(
                    "log-sigma eta length mismatch: got {}, expected {n}",
                    eta_ls_raw.len()
                ));
            }
            (eta_ls_raw.slice(s![0..n]), eta_ls_raw.slice(s![0..n]))
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
            etaw,
        ))
    }

    fn collect_joint_quantities(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<SurvivalJointQuantities, String> {
        let n = self.n;
        let (h0, h1, d_raw, eta_t_exit, eta_ls_exit, eta_t_entry, eta_ls_entry, etaw) =
            self.validate_joint_states(block_states)?;
        // For joint Hessian / D_u H we use the EXIT evaluations of sigma and its
        // derivatives (the exit design is the solver's primary design).
        //
        // We now compute sigma derivatives through 4th order because the outer
        // REML Hessian's Q[v_k, v_l] term requires 4th-order chain-rule
        // derivatives of q w.r.t. the block predictors (u_ϑsss and u_ssss).
        // See response.md Section 6.
        let (sigma, ds, d2s, d3s, d4s) = exp_sigma_derivs_up_to_fourth(eta_ls_exit.view());
        let entry_sigma_derivs = if self.x_log_sigma_entry.is_some() {
            Some(exp_sigma_derivs_up_to_fourth(eta_ls_entry.view()))
        } else {
            None
        };
        let mut d1_q = Array1::<f64>::zeros(n);
        let mut d2_q = Array1::<f64>::zeros(n);
        let mut d3_q = Array1::<f64>::zeros(n);
        let mut d4_q = Array1::<f64>::zeros(n);
        let mut d1_q0 = Array1::<f64>::zeros(n);
        let mut d2_q0 = Array1::<f64>::zeros(n);
        let mut d3_q0 = Array1::<f64>::zeros(n);
        let mut d4_q0 = Array1::<f64>::zeros(n);
        let mut d1_q1 = Array1::<f64>::zeros(n);
        let mut d2_q1 = Array1::<f64>::zeros(n);
        let mut d3_q1 = Array1::<f64>::zeros(n);
        let mut d4_q1 = Array1::<f64>::zeros(n);
        let mut h_time_h0 = Array1::<f64>::zeros(n);
        let mut h_time_h1 = Array1::<f64>::zeros(n);
        let mut h_time_d = Array1::<f64>::zeros(n);
        let mut d_h_h0 = Array1::<f64>::zeros(n);
        let mut d_h_h1 = Array1::<f64>::zeros(n);
        let mut d_h_d = Array1::<f64>::zeros(n);
        let mut d2_h_h0 = Array1::<f64>::zeros(n);
        let mut d2_h_h1 = Array1::<f64>::zeros(n);

        for i in 0..n {
            let sigma_entry_i = entry_sigma_derivs
                .as_ref()
                .map_or(sigma[i], |entry| entry.0[i]);
            let state = self.row_predictor_state(
                h0[i],
                h1[i],
                d_raw[i],
                eta_t_entry[i],
                eta_t_exit[i],
                sigma_entry_i,
                sigma[i],
                etaw.map(|w| w[i]),
            );
            let Some(row) = self.row_derivatives(i, state)? else {
                continue;
            };
            d1_q[i] = row.d1_q;
            d2_q[i] = row.d2_q;
            d3_q[i] = row.d3_q;
            d4_q[i] = row.d4_q;
            d1_q0[i] = row.d1_q0;
            d2_q0[i] = row.d2_q0;
            d3_q0[i] = row.d3_q0;
            d4_q0[i] = row.d4_q0;
            d1_q1[i] = row.d1_q1;
            d2_q1[i] = row.d2_q1;
            d3_q1[i] = row.d3_q1;
            d4_q1[i] = row.d4_q1;
            h_time_h0[i] = row.h_time_h0;
            h_time_h1[i] = row.h_time_h1;
            h_time_d[i] = row.h_time_d;
            d_h_h0[i] = row.d_h_h0;
            d_h_h1[i] = row.d_h_h1;
            d_h_d[i] = row.d_h_d;
            d2_h_h0[i] = row.d2_h_h0;
            d2_h_h1[i] = row.d2_h_h1;
        }

        // q(eta_t, eta_ls, etaw) = -eta_t / max(sigma(eta_ls), 1e-12) + etaw.
        //
        // For the joint Hessian and D_u H, the partial derivatives of q
        // w.r.t. the block parameters are evaluated at the EXIT linear predictor
        // (the solver's primary design). The entry contribution is handled
        // separately by the h0 branch of the time-side derivatives.
        //
        // The exact derivatives used in the joint Hessian and D_u H are:
        //
        //   q_t      = -1 / max(sigma, 1e-12)
        //   q_ls     = eta_t * sigma' / max(sigma, 1e-12)^2
        //   q_t,ls   = sigma' / max(sigma, 1e-12)^2
        //   q_ls,ls  = eta_t * (sigma''/s^2 - 2 sigma'^2/s^3)
        //   q_t,ls,ls= sigma''/s^2 - 2 sigma'^2/s^3
        //   q_ls,ls,ls
        //            = eta_t * (sigma'''/s^2 - 6 sigma' sigma''/s^3
        //                        + 6 sigma'^3/s^4),
        //
        // where s = max(sigma, 1e-12). On the active floor branch the coded q
        // is locally constant in eta_ls, so all eta_ls-derivatives vanish.
        //
        // These are the scalar q_{i,b}, q_{i,bc}, q_{i,bcd}, q_{i,bcde} objects
        // from the derivation, specialized to the threshold / scale / wiggle blocks.
        // The 4th-order terms (q_{i,bcde}) are new — needed for the outer REML
        // Hessian drift. See response.md Section 6.
        let exit_qd = compute_q_chain_derivs(&eta_t_exit, &sigma, &ds, &d2s, &d3s, &d4s);

        // Entry-side chain rule derivatives for time-varying blocks.
        let dq_t_entry = if self.x_threshold_entry.is_some() {
            let se = entry_sigma_derivs
                .as_ref()
                .map(|(s, _, _, _, _)| s)
                .unwrap();
            Some(se.mapv(|s| -1.0 / s.max(1e-12)))
        } else {
            None
        };
        let entry_qd = if let Some((sigma_entry, ds_entry, d2s_entry, d3s_entry, d4s_entry)) =
            entry_sigma_derivs.as_ref()
        {
            Some(compute_q_chain_derivs(
                &eta_t_entry,
                sigma_entry,
                ds_entry,
                d2s_entry,
                d3s_entry,
                d4s_entry,
            ))
        } else {
            None
        };

        Ok(SurvivalJointQuantities {
            d1_q,
            d2_q,
            d3_q,
            d4_q,
            d1_q0,
            d2_q0,
            d3_q0,
            d4_q0,
            d1_q1,
            d2_q1,
            d3_q1,
            d4_q1,
            h_time_h0,
            h_time_h1,
            h_time_d,
            d_h_h0,
            d_h_h1,
            d_h_d,
            d2_h_h0,
            d2_h_h1,
            dq_t: exit_qd.dq_t,
            dq_ls: exit_qd.dq_ls,
            d2q_tls: exit_qd.d2q_tls,
            d2q_ls: exit_qd.d2q_ls,
            d3q_tls_ls: exit_qd.d3q_tls_ls,
            d3q_ls: exit_qd.d3q_ls,
            dq_t_entry,
            dq_ls_entry: entry_qd.as_ref().map(|q| q.dq_ls.clone()),
            d2q_tls_entry: entry_qd.as_ref().map(|q| q.d2q_tls.clone()),
            d2q_ls_entry: entry_qd.as_ref().map(|q| q.d2q_ls.clone()),
            d3q_tls_ls_entry: entry_qd.as_ref().map(|q| q.d3q_tls_ls.clone()),
            d3q_ls_entry: entry_qd.as_ref().map(|q| q.d3q_ls.clone()),
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
        let t_time_varying = self.x_threshold_entry.is_some();
        let ls_time_varying = self.x_log_sigma_entry.is_some();

        let mut global = 0usize;
        for (block_idx, block_derivs) in derivative_blocks.iter().enumerate() {
            for (local_idx, deriv) in block_derivs.iter().enumerate() {
                if global == psi_index {
                    let mut x_t_exit_psi = Array2::<f64>::zeros((n, pt));
                    let mut x_t_entry_psi = Array2::<f64>::zeros((n, pt));
                    let mut x_ls_exit_psi = Array2::<f64>::zeros((n, pls));
                    let mut x_ls_entry_psi = Array2::<f64>::zeros((n, pls));
                    match block_idx {
                        Self::BLOCK_THRESHOLD => {
                            let x_psi = resolve_custom_family_x_psi(
                                deriv,
                                if t_time_varying { 2 * n } else { n },
                                pt,
                                "SurvivalLocationScaleFamily threshold",
                            )?;
                            let (exit, entry) = split_survival_psi_design(
                                &x_psi,
                                n,
                                t_time_varying,
                                "SurvivalLocationScaleFamily threshold",
                            )?;
                            x_t_exit_psi.assign(&exit);
                            x_t_entry_psi.assign(&entry);
                        }
                        Self::BLOCK_LOG_SIGMA => {
                            let x_psi = resolve_custom_family_x_psi(
                                deriv,
                                if ls_time_varying { 2 * n } else { n },
                                pls,
                                "SurvivalLocationScaleFamily log-sigma",
                            )?;
                            let (exit, entry) = split_survival_psi_design(
                                &x_psi,
                                n,
                                ls_time_varying,
                                "SurvivalLocationScaleFamily log-sigma",
                            )?;
                            x_ls_exit_psi.assign(&exit);
                            x_ls_entry_psi.assign(&entry);
                        }
                        _ => return Ok(None),
                    }
                    let z_t_exit_psi = x_t_exit_psi.dot(beta_t);
                    let z_t_entry_psi = x_t_entry_psi.dot(beta_t);
                    let z_ls_exit_psi = x_ls_exit_psi.dot(beta_ls);
                    let z_ls_entry_psi = x_ls_entry_psi.dot(beta_ls);
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
    ) -> Result<
        (
            Array2<f64>,
            Array2<f64>,
            Array2<f64>,
            Array2<f64>,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
            Array1<f64>,
        ),
        String,
    > {
        let n = self.n;
        let pt = self.x_threshold.ncols();
        let pls = self.x_log_sigma.ncols();
        let beta_t = &block_states[Self::BLOCK_THRESHOLD].beta;
        let beta_ls = &block_states[Self::BLOCK_LOG_SIGMA].beta;
        let t_time_varying = self.x_threshold_entry.is_some();
        let ls_time_varying = self.x_log_sigma_entry.is_some();

        let mut x_t_exit_ab = Array2::<f64>::zeros((n, pt));
        let mut x_t_entry_ab = Array2::<f64>::zeros((n, pt));
        let mut x_ls_exit_ab = Array2::<f64>::zeros((n, pls));
        let mut x_ls_entry_ab = Array2::<f64>::zeros((n, pls));

        if psi_a.block_idx == psi_b.block_idx {
            let deriv = &derivative_blocks[psi_a.block_idx][psi_a.local_idx];
            let deriv_b = &derivative_blocks[psi_b.block_idx][psi_b.local_idx];
            match psi_a.block_idx {
                Self::BLOCK_THRESHOLD => {
                    let x_ab = resolve_custom_family_x_psi_psi(
                        deriv,
                        deriv_b,
                        psi_b.local_idx,
                        if t_time_varying { 2 * n } else { n },
                        pt,
                        "SurvivalLocationScaleFamily threshold",
                    )?;
                    let (exit, entry) = split_survival_psi_design(
                        &x_ab,
                        n,
                        t_time_varying,
                        "SurvivalLocationScaleFamily threshold",
                    )?;
                    x_t_exit_ab.assign(&exit);
                    x_t_entry_ab.assign(&entry);
                }
                Self::BLOCK_LOG_SIGMA => {
                    let x_ab = resolve_custom_family_x_psi_psi(
                        deriv,
                        deriv_b,
                        psi_b.local_idx,
                        if ls_time_varying { 2 * n } else { n },
                        pls,
                        "SurvivalLocationScaleFamily log-sigma",
                    )?;
                    let (exit, entry) = split_survival_psi_design(
                        &x_ab,
                        n,
                        ls_time_varying,
                        "SurvivalLocationScaleFamily log-sigma",
                    )?;
                    x_ls_exit_ab.assign(&exit);
                    x_ls_entry_ab.assign(&entry);
                }
                _ => {}
            }
        }

        let z_t_exit_ab = x_t_exit_ab.dot(beta_t);
        let z_t_entry_ab = x_t_entry_ab.dot(beta_t);
        let z_ls_exit_ab = x_ls_exit_ab.dot(beta_ls);
        let z_ls_entry_ab = x_ls_entry_ab.dot(beta_ls);
        Ok((
            x_t_exit_ab,
            x_t_entry_ab,
            x_ls_exit_ab,
            x_ls_entry_ab,
            z_t_exit_ab,
            z_t_entry_ab,
            z_ls_exit_ab,
            z_ls_entry_ab,
        ))
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
    /// outer REML Hessian's Q[v_k, v_l] term via the Faà di Bruno formula.
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

    /// Clamp-aware log-pdf and its first/second/third derivatives.
    ///
    /// Let `L(u) = log f(u)` on the unclamped branch. The exact derivatives are:
    ///
    /// `L'   = f'/f`
    ///
    /// `L''  = d/du(f'/f)
    ///       = (f'' f - (f')²) / f²
    ///       = f''/f - (f'/f)²`
    ///
    /// For the third derivative, differentiate `L'' = f''/f - (f'/f)^2`:
    ///
    /// `d/du[f''/f]   = f'''/f - f'f''/f²`
    ///
    /// `d/du[(f'/f)²] = 2(f'/f)(f''/f - (f'/f)²)
    ///                = 2f'f''/f² - 2(f')³/f³`
    ///
    /// so
    ///
    /// `L''' = f'''/f - 3 f'f''/f² + 2(f')³/f³`.
    ///
    /// This is the exact `d³(log f)/du³` term used in the survival exact-Newton
    /// Hessian directional derivative. If it is dropped, the event contribution
    /// to `d³ℓ/dq³` is wrong, which then corrupts the block Hessian drift
    /// `D H[u]`.
    ///
    /// The objective actually uses `log(max(f, MIN_PROB))`, not `log f`, so once
    /// `f <= MIN_PROB` the active branch is constant and all derivatives must be
    /// zero. That is why the clamped branch returns `(log MIN_PROB, 0, 0, 0)`.
    fn clamped_log_pdfwith_derivatives(
        f: f64,
        fp: f64,
        fpp: f64,
        fppp: f64,
    ) -> (f64, f64, f64, f64) {
        if f <= MIN_PROB {
            (MIN_PROB.ln(), 0.0, 0.0, 0.0)
        } else {
            let d1 = fp / f;
            let d2 = fpp / f - d1 * d1;
            let d3 = fppp / f - 3.0 * fp * fpp / (f * f) + 2.0 * fp * fp * fp / (f * f * f);
            (f.ln(), d1, d2, d3)
        }
    }

    /// Fourth derivative of log f(u), given f through f''''.
    ///
    /// ```text
    /// L'''' = f''''/f - 4f'f'''/f² - 3(f'')²/f² + 12(f')²f''/f³ - 6(f')⁴/f⁴
    /// ```
    ///
    /// Derivation: differentiate L''' = f'''/f - 3f'f''/f² + 2(f')³/f³.
    ///
    /// This is needed for d⁴ℓ/dq1⁴ (the exit-side 4th derivative of the
    /// event contribution), which enters the outer REML Hessian Q[v_k, v_l].
    fn log_pdf_fourth_derivative(f: f64, fp: f64, fpp: f64, fppp: f64, fpppp: f64) -> f64 {
        if f <= MIN_PROB {
            return 0.0;
        }
        let f2 = f * f;
        let f3 = f2 * f;
        let f4 = f3 * f;
        let fp2 = fp * fp;
        fpppp / f - 4.0 * fp * fppp / f2 - 3.0 * fpp * fpp / f2 + 12.0 * fp2 * fpp / f3
            - 6.0 * fp2 * fp2 / f4
    }

    /// Clamp-aware survival value and derivatives of `-log(clamp(S, MIN_PROB, 1))`.
    ///
    /// Returns `(S_clamped, r, dr, ddr)` where:
    /// - `r   = d/du[-log S_clamped]`
    /// - `dr  = d²/du²[-log S_clamped]`
    /// - `ddr = d³/du³[-log S_clamped]`
    ///
    /// If `S` is clamped at either bound (`S <= MIN_PROB` or `S >= 1`), these
    /// derivatives are all zero because the clamped log term is locally constant.
    fn clamped_survival_neglog_derivatives(
        raw_s: f64,
        f: f64,
        fp: f64,
        fpp: f64,
    ) -> (f64, f64, f64, f64) {
        let s = raw_s.clamp(MIN_PROB, 1.0);
        if raw_s <= MIN_PROB || raw_s >= 1.0 {
            (s, 0.0, 0.0, 0.0)
        } else {
            let (r, dr) = Self::survival_ratio_first_derivative(f, fp, s);
            let ddr = Self::survival_ratiosecond_derivative(r, dr, f, fp, fpp, s);
            (s, r, dr, ddr)
        }
    }

    /// Clamp-aware survival value and derivatives of `-log(clamp(S, MIN_PROB, 1))`
    /// through **4th order**.
    ///
    /// Returns `(S_clamped, r, dr, ddr, dddr)` where dddr = d⁴/du⁴[-log S].
    ///
    /// The 4th derivative of -log S is r''' (the 3rd derivative of the
    /// survival ratio r = f/S). This is needed by the outer REML Hessian's
    /// Q[v_k, v_l] term for the entry-side contribution.
    fn clamped_survival_neglog_derivatives_fourth(
        raw_s: f64,
        f: f64,
        fp: f64,
        fpp: f64,
        fppp: f64,
    ) -> (f64, f64, f64, f64, f64) {
        let s = raw_s.clamp(MIN_PROB, 1.0);
        if raw_s <= MIN_PROB || raw_s >= 1.0 {
            (s, 0.0, 0.0, 0.0, 0.0)
        } else {
            let (r, dr) = Self::survival_ratio_first_derivative(f, fp, s);
            let ddr = Self::survival_ratiosecond_derivative(r, dr, f, fp, fpp, s);
            let dddr = Self::survival_ratio_third_derivative(r, dr, ddr, f, fp, fpp, fppp, s);
            (s, r, dr, ddr, dddr)
        }
    }

    /// Clamp-aware `log(max(x, floor))` value and first three derivatives.
    ///
    /// Once `x <= floor`, the active branch is constant so every derivative is
    /// zero. This keeps the exact-Newton derivatives aligned with the objective.
    fn clamped_logwith_derivatives(raw_x: f64, floor: f64) -> (f64, f64, f64, f64) {
        let x = raw_x.max(floor);
        if raw_x <= floor {
            (x.ln(), 0.0, 0.0, 0.0)
        } else {
            let inv = 1.0 / x;
            (x.ln(), inv, -inv * inv, 2.0 * inv * inv * inv)
        }
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
        eta_t_entry: f64,
        eta_t_exit: f64,
        sigma_entry: f64,
        sigma_exit: f64,
        etaw: Option<f64>,
    ) -> SurvivalPredictorState {
        let w = etaw.unwrap_or(0.0);
        SurvivalPredictorState {
            h0,
            h1,
            d_raw,
            q0: -eta_t_entry / sigma_entry.max(1e-12) + w,
            q1: -eta_t_exit / sigma_exit.max(1e-12) + w,
        }
    }

    fn row_derivatives(
        &self,
        row: usize,
        state: SurvivalPredictorState,
    ) -> Result<Option<SurvivalRowDerivatives>, String> {
        let w = self.w[row];
        if w <= 0.0 {
            return Ok(None);
        }
        let d = self.y[row].clamp(0.0, 1.0);
        let u0 = -state.h0 + state.q0;
        let u1 = -state.h1 + state.q1;
        let j0 = inverse_link_jet_for_inverse_link(&self.inverse_link, u0)
            .map_err(|e| format!("inverse link evaluation failed at row {row} entry: {e}"))?;
        let j1 = inverse_link_jet_for_inverse_link(&self.inverse_link, u1)
            .map_err(|e| format!("inverse link evaluation failed at row {row} exit: {e}"))?;
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
        // So we evaluate `S` directly through the inverse-link-specific stable
        // survival helper and only use the inverse-link jet for the density-side
        // derivatives `(f, f', f'')`.
        let raw_s0 = inverse_link_survival_probvalue(&self.inverse_link, u0);
        let raw_s1 = inverse_link_survival_probvalue(&self.inverse_link, u1);

        // For the 4th-order derivatives (needed by the outer REML Hessian),
        // we need the survival ratio through r''' and log-pdf through L''''.
        // The entry side needs f'''(u0) for r'''(u0); the exit side needs
        // f''''(u1) for L''''(u1).
        //
        // f''' comes from inverse_link_pdfthird_derivative; f'''' from our
        // local inverse_link_pdffourth_derivative helper.
        let fppp0 = inverse_link_pdfthird_derivative_for_inverse_link(&self.inverse_link, u0)
            .map_err(|e| {
                format!("inverse link third-derivative evaluation failed at row {row} entry: {e}")
            })?;
        let (s0, r0, dr0, ddr0, dddr0) =
            Self::clamped_survival_neglog_derivatives_fourth(raw_s0, j0.d1, j0.d2, j0.d3, fppp0);

        let fppp1 = inverse_link_pdfthird_derivative_for_inverse_link(&self.inverse_link, u1)
            .map_err(|e| {
                format!("inverse link third-derivative evaluation failed at row {row} exit: {e}")
            })?;
        let (s1, r1, dr1, ddr1, dddr1) =
            Self::clamped_survival_neglog_derivatives_fourth(raw_s1, j1.d1, j1.d2, j1.d3, fppp1);

        let (logphi1, dlogphi1, d2logphi1, d3logphi1) =
            Self::clamped_log_pdfwith_derivatives(j1.d1, j1.d2, j1.d3, fppp1);

        // 4th derivative of log f(u1) for the exit-side event contribution.
        // f'''' is the 4th PDF derivative; L'''' uses f through f''''.
        let fpppp1 = inverse_link_pdffourth_derivative(&self.inverse_link, u1);
        let d4logphi1 = Self::log_pdf_fourth_derivative(j1.d1, j1.d2, j1.d3, fppp1, fpppp1);

        let guard = self.derivative_guard;
        let soft = self.derivative_softness.max(0.0);
        let (g, log_g_safe, d_log_g, d2_log_g, d3_log_g) = if state.d_raw.is_finite() {
            let g_val = state.d_raw;
            let (log_g, d1, d2, d3) = Self::clamped_logwith_derivatives(g_val + soft, 1e-12);
            (g_val, log_g, d1, d2, d3)
        } else {
            // Keep the likelihood/derivative surface finite when line-search
            // probes an invalid time-derivative state. Treat this as an active
            // floor branch: finite objective contribution with zero local
            // derivative curvature in d_eta/dt.
            let g_floor = 0.0;
            let log_g = soft.max(1e-12).ln();
            (g_floor, log_g, 0.0, 0.0, 0.0)
        };
        if guard > 0.0 && g <= guard {
            return Err(format!(
                "survival location-scale monotonicity violated at row {row}: d_eta/dt={g:.3e} <= guard={:.3e}",
                guard
            ));
        }

        // With
        //
        //   ell = w [ d(log f(u1) + log g) + (1-d) log S(u1) - log S(u0) ],
        //   u0 = q0 - h0,
        //   u1 = q1 - h1,
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
        // The time-side partials follow from u0 = q0 - h0 and u1 = q1 - h1:
        //
        //   ell_h0   = -ell_q0 = -w r(u0)
        //   ell_h1   = -ell_q1
        //   ell_h0q0 = -w r'(u0)
        //   ell_h1q1 = -w [ d d²/du² log f(u1) - (1-d) r'(u1) ]
        //
        // The 4th-order derivatives d4_q0 and d4_q1 are the m4 quantities
        // needed by the Faà di Bruno chain rule for the outer REML Hessian.
        // They enter F_αβγδ = m4·u_α·u_β·u_γ·u_δ + ... in the (s,s,s,s)
        // and (ϑ,s,s,s) blocks. See response.md Section 6.
        let d1_q0 = w * r0;
        let d2_q0 = w * dr0;
        let d3_q0 = w * ddr0;
        let d4_q0 = w * dddr0;
        let d1_q1 = w * (d * dlogphi1 + (1.0 - d) * (-r1));
        let d2_q1 = w * (d * d2logphi1 + (1.0 - d) * (-dr1));
        let d3_q1 = w * (d * d3logphi1 + (1.0 - d) * (-ddr1));
        let d4_q1 = w * (d * d4logphi1 + (1.0 - d) * (-dddr1));
        let d1_q = d1_q0 + d1_q1;
        let d2_q = d2_q0 + d2_q1;
        let d3_q = d3_q0 + d3_q1;
        let d4_q = d4_q0 + d4_q1;

        Ok(Some(SurvivalRowDerivatives {
            ll: w * (d * (logphi1 + log_g_safe) + (1.0 - d) * s1.ln() - s0.ln()),
            d1_q,
            d2_q,
            d3_q,
            d4_q,
            d1_q0,
            d2_q0,
            d3_q0,
            d4_q0,
            d1_q1,
            d2_q1,
            d3_q1,
            d4_q1,
            grad_time_eta_h0: -w * r0,
            grad_time_eta_h1: -w * (d * dlogphi1 + (1.0 - d) * (-r1)),
            grad_time_eta_d: w * d * d_log_g,
            h_time_h0: -w * dr0,
            h_time_h1: -w * (d * d2logphi1 + (1.0 - d) * (-dr1)),
            h_time_d: -w * d * d2_log_g,
            d_h_h0: w * ddr0,
            d_h_h1: w * d * d3logphi1 - w * (1.0 - d) * ddr1,
            d_h_d: -w * d * d3_log_g,
            // 4th derivatives of ℓ w.r.t. the time predictors h0, h1.
            // These are the exact bilinear coefficients for D²H[u,v] in the
            // time-time block. Since u = q - h and d⁴ℓ/dh⁴ = d⁴ℓ/du⁴
            // (same sign because (-1)⁴ = 1), we have:
            d2_h_h0: w * dddr0,
            d2_h_h1: w * (d * d4logphi1 + (1.0 - d) * (-dddr1)),
        }))
    }
}

/// Scalar chain-rule derivatives of
/// q(eta_t, eta_ls) = -eta_t / max(sigma(eta_ls), 1e-12).
///
/// Returns (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) — the full set of
/// partials up to third order needed by both the survival and GAMLSS engines.
#[inline]
pub(crate) fn q_chain_derivs_scalar(
    eta_t: f64,
    sigma: f64,
    dsigma: f64,
    d2sigma: f64,
    d3sigma: f64,
) -> (f64, f64, f64, f64, f64, f64) {
    let s = sigma.max(1e-12);
    let q_t = -1.0 / s;
    if sigma <= 1e-12 {
        return (q_t, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let s2 = s * s;
    let s3 = s2 * s;
    let s4 = s3 * s;
    let q_tl = dsigma / s2;
    let q_ls = eta_t * q_tl;
    let q_tl_ls = d2sigma / s2 - 2.0 * dsigma * dsigma / s3;
    let q_ll = eta_t * q_tl_ls;
    let q_ll_ls = eta_t * (d3sigma / s2 - 6.0 * dsigma * d2sigma / s3 + 6.0 * dsigma.powi(3) / s4);
    (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls)
}

/// Extended scalar chain-rule derivatives of
/// q(eta_t, eta_ls) = -eta_t / max(sigma(eta_ls), 1e-12)
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
/// of the composed likelihood via the Faà di Bruno formula:
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
    sigma: f64,
    dsigma: f64,
    d2sigma: f64,
    d3sigma: f64,
    d4sigma: f64,
) -> (f64, f64, f64, f64, f64, f64, f64, f64) {
    let s = sigma.max(1e-12);
    let q_t = -1.0 / s;
    if sigma <= 1e-12 {
        return (q_t, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    let s2 = s * s;
    let s3 = s2 * s;
    let s4 = s3 * s;
    let s5 = s4 * s;

    // 1st order
    let q_tl = dsigma / s2;
    let q_ls = eta_t * q_tl;

    // 2nd order
    let q_tl_ls = d2sigma / s2 - 2.0 * dsigma * dsigma / s3;
    let q_ll = eta_t * q_tl_ls;

    // 3rd order
    let ds2 = dsigma * dsigma;
    let q_ll_ls = eta_t * (d3sigma / s2 - 6.0 * dsigma * d2sigma / s3 + 6.0 * ds2 * dsigma / s4);

    // 4th order: u_ϑssss and u_ssss
    //
    // u_ϑsss (3rd order in s, mixed with ϑ) was:
    //   σ'''/σ² - 6σ'σ''/σ³ + 6(σ')³/σ⁴
    //
    // u_ϑssss = d/d(η_s)[u_ϑsss]:
    //   d/d(η_s)[σ'''/σ²]        = σ''''/σ² - 2σ'σ'''/σ³
    //   d/d(η_s)[-6σ'σ''/σ³]     = -6(σ''² + σ'σ''')/σ³ + 18(σ')²σ''/σ⁴
    //   d/d(η_s)[6(σ')³/σ⁴]      = 18(σ')²σ''/σ⁴ - 24(σ')⁴/σ⁵
    //
    // Collecting:
    //   u_ϑssss = σ''''/σ² - 8σ'σ'''/σ³ - 6(σ'')²/σ³ + 36(σ')²σ''/σ⁴ - 24(σ')⁴/σ⁵
    //
    // For exp link (σ=σ'=σ''=σ'''=σ''''):
    //   = σ⁻¹(1 - 8 - 6 + 36 - 24) = -σ⁻¹
    //
    // Wait — that gives -σ⁻¹, but the alternating pattern predicts +σ⁻¹ for
    // u_ϑsss (4th partial, 3 s-slots) and the *next* one should be -σ⁻¹.
    //
    // Clarification: the naming is careful here.
    //   u_ϑs    = 1st s-deriv of u_ϑ   = +σ⁻¹
    //   u_ϑss   = 2nd s-deriv of u_ϑ   = -σ⁻¹
    //   u_ϑsss  = 3rd s-deriv of u_ϑ   = +σ⁻¹
    //   u_ϑssss = 4th s-deriv of u_ϑ   = -σ⁻¹  ← this is what we compute here
    //
    // The formula above gives -σ⁻¹ for exp link, which matches the 4th
    // s-derivative (even count → negative). The Faà di Bruno entry for the
    // (ϑ,s,s,s) block uses u_ϑsss (3rd s-deriv = +σ⁻¹), NOT u_ϑssss.
    // But response.md Section 6 labels the needed quantity as u_ϑsss = σ⁻¹,
    // meaning the partial with indices (ϑ,s,s,s) = 3 s-slots = +σ⁻¹.
    //
    // So the variable naming convention here is:
    //   q_tl_ls_ls = u_ϑssss = 4th s-deriv of u_ϑ (what we compute)
    //
    // BUT for the QChainDerivs struct, d4q_tls_ls_ls stores u_ϑsss
    // (the partial ∂⁴q/∂η_ϑ∂η_s³ with exactly 3 s-indices), which is
    // the 3rd s-derivative of u_ϑ = the quantity labeled u_ϑsss in
    // response.md Section 6.
    //
    // The scalar function here computes the actual 4th-order derivatives
    // of q w.r.t. η_s, using σ'''' = d4sigma.
    //
    // For clarity: q_tl_ls_ls = ∂⁴q/(∂η_ϑ ∂η_s³), which is u_ϑsss.
    // This is the 3rd partial in η_s of q_t = -1/σ, i.e., the derivative
    // chain applied 3 times through σ(η_s).
    //
    // Recurrence: q_tl_ls_ls = d/d(η_s)[q_tl_ls] where q_tl_ls = u_ϑss.
    // But q_tl_ls is the 2nd s-partial, and differentiating once more gives
    // the 3rd s-partial = u_ϑsss. So:
    //   q_tl_ls_ls = u_ϑsss = σ'''/σ² - 6σ'σ''/σ³ + 6(σ')³/σ⁴
    //
    // That's the SAME as q_ll_ls / eta_t. So for the true 4th-order
    // (4 s-slots), we need one more derivative:
    //
    // d4q_tls_ls_ls in the struct = ∂⁴q/(∂η_ϑ ∂η_s³) = u_ϑsss
    //   = σ'''/σ² - 6σ'σ''/σ³ + 6(σ')³/σ⁴   (this is 3rd-order, already have it)
    //
    // d4q_ls in the struct = ∂⁴q/∂η_s⁴ = u_ssss
    //   = η_ϑ · [σ''''/σ² - 8σ'σ'''/σ³ - 6(σ'')²/σ³ + 36(σ')²σ''/σ⁴ - 24(σ')⁴/σ⁵]
    //
    // So the 4th-order q chain derivatives that are introduced here are:
    //   (a) u_ϑsss = 3rd s-partial of u_ϑ — already computed as q_ll_ls/eta_t
    //       but stored explicitly for the Faà di Bruno formula
    //   (b) u_ssss = 4th s-partial of q — needs σ''''
    // Recompute cleanly to avoid division issues when eta_t ≈ 0:
    let q_tl_ls_ls_clean = d3sigma / s2 - 6.0 * dsigma * d2sigma / s3 + 6.0 * ds2 * dsigma / s4;

    // u_ssss = ∂⁴q/∂η_s⁴ needs σ'''' and the full 4th-order chain rule:
    let q_llll = eta_t
        * (d4sigma / s2 - 8.0 * dsigma * d3sigma / s3 - 6.0 * d2sigma * d2sigma / s3
            + 36.0 * ds2 * d2sigma / s4
            - 24.0 * ds2 * ds2 / s5);

    (
        q_t,
        q_ls,
        q_tl,
        q_ll,
        q_tl_ls,
        q_ll_ls,
        q_tl_ls_ls_clean,
        q_llll,
    )
}

/// All chain-rule partial derivatives of
/// q(eta_t, eta_ls) = -eta_t / max(sigma(eta_ls), 1e-12)
/// with respect to the block linear predictors eta_t and eta_ls.
///
/// Includes 4th-order terms needed by the outer REML Hessian drift.
struct QChainDerivs {
    dq_t: Array1<f64>,       // ∂q/∂eta_t = -1/s with s = max(σ, 1e-12)
    dq_ls: Array1<f64>,      // ∂q/∂eta_ls = eta_t · σ'/s²
    d2q_tls: Array1<f64>,    // ∂²q/∂eta_t∂eta_ls = σ'/s²
    d2q_ls: Array1<f64>,     // ∂²q/∂eta_ls² = eta_t · (σ''/s² - 2σ'²/s³)
    d3q_tls_ls: Array1<f64>, // ∂³q/∂eta_t∂eta_ls² = σ''/s² - 2σ'²/s³
    d3q_ls: Array1<f64>,     // ∂³q/∂eta_ls³ = eta_t · (σ'''/s² - 6σ'σ''/s³ + 6σ'³/s⁴)

    /// ∂⁴q/∂η_ϑ∂η_s³ = u_ϑsss.
    ///
    /// For exp link: +σ⁻¹ (alternating-sign pattern, 3rd s-derivative of u_ϑ).
    /// General formula: σ'''/σ² - 6σ'σ''/σ³ + 6(σ')³/σ⁴.
    ///
    /// This enters the (ϑ,s,s,s) block of the 4th-order Faà di Bruno formula
    /// via the m1·u_αβγδ term. See response.md Section 6.
    d4q_tls_ls_ls: Array1<f64>,

    /// ∂⁴q/∂η_s⁴ = u_ssss.
    ///
    /// For exp link: -ϑ/σ (alternating-sign pattern, 4th s-derivative of q).
    /// General formula: η_ϑ · (σ''''/σ² - 8σ'σ'''/σ³ - 6(σ'')²/σ³
    ///                         + 36(σ')²σ''/σ⁴ - 24(σ')⁴/σ⁵).
    ///
    /// This enters the (s,s,s,s) block of the 4th-order Faà di Bruno formula
    /// via the m1·u_αβγδ term. See response.md Section 6.
    d4q_ls: Array1<f64>,
}

/// Compute all chain-rule derivatives of
/// q = -eta_t / max(sigma(eta_ls), 1e-12) as length-n arrays,
/// including 4th-order terms needed by the outer REML Hessian.
fn compute_q_chain_derivs(
    eta_t: &ndarray::ArrayBase<impl ndarray::Data<Elem = f64>, ndarray::Ix1>,
    sigma: &Array1<f64>,
    ds: &Array1<f64>,
    d2s: &Array1<f64>,
    d3s: &Array1<f64>,
    d4s: &Array1<f64>,
) -> QChainDerivs {
    let n = eta_t.len();
    let mut r = QChainDerivs {
        dq_t: Array1::zeros(n),
        dq_ls: Array1::zeros(n),
        d2q_tls: Array1::zeros(n),
        d2q_ls: Array1::zeros(n),
        d3q_tls_ls: Array1::zeros(n),
        d3q_ls: Array1::zeros(n),
        d4q_tls_ls_ls: Array1::zeros(n),
        d4q_ls: Array1::zeros(n),
    };
    for i in 0..n {
        let (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls, q_tl_ls_ls, q_llll) =
            q_chain_derivs_fourth_scalar(eta_t[i], sigma[i], ds[i], d2s[i], d3s[i], d4s[i]);
        r.dq_t[i] = q_t;
        r.dq_ls[i] = q_ls;
        r.d2q_tls[i] = q_tl;
        r.d2q_ls[i] = q_ll;
        r.d3q_tls_ls[i] = q_tl_ls;
        r.d3q_ls[i] = q_ll_ls;
        r.d4q_tls_ls_ls[i] = q_tl_ls_ls;
        r.d4q_ls[i] = q_llll;
    }
    r
}

/// Compute chain-rule derivatives of q through 3rd order only (no 4th-order terms).
///
/// This is the inner-loop version used by the P-IRLS evaluate path, which
/// only needs gradient and Hessian weights (not the outer REML Hessian drift).
/// The 4th-order fields `d4q_tls_ls_ls` and `d4q_ls` are set to zero.
fn compute_q_chain_derivs_third(
    eta_t: &ndarray::ArrayBase<impl ndarray::Data<Elem = f64>, ndarray::Ix1>,
    sigma: &Array1<f64>,
    ds: &Array1<f64>,
    d2s: &Array1<f64>,
    d3s: &Array1<f64>,
) -> QChainDerivs {
    let n = eta_t.len();
    let mut r = QChainDerivs {
        dq_t: Array1::zeros(n),
        dq_ls: Array1::zeros(n),
        d2q_tls: Array1::zeros(n),
        d2q_ls: Array1::zeros(n),
        d3q_tls_ls: Array1::zeros(n),
        d3q_ls: Array1::zeros(n),
        d4q_tls_ls_ls: Array1::zeros(n),
        d4q_ls: Array1::zeros(n),
    };
    for i in 0..n {
        let (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) =
            q_chain_derivs_scalar(eta_t[i], sigma[i], ds[i], d2s[i], d3s[i]);
        r.dq_t[i] = q_t;
        r.dq_ls[i] = q_ls;
        r.d2q_tls[i] = q_tl;
        r.d2q_ls[i] = q_ll;
        r.d3q_tls_ls[i] = q_tl_ls;
        r.d3q_ls[i] = q_ll_ls;
        // d4q_tls_ls_ls and d4q_ls remain zero — not needed for P-IRLS.
    }
    r
}

/// Chain-rule gradient and negative Hessian diagonal weights for a single side
/// (exit or entry) of a parameter block.
///
///   grad_eta[i] = d1_q[i] · dq[i]
///   h_eta[i]    = -(d2_q[i] · dq[i]² + d1_q[i] · d2q[i])
///
/// When `d2q` is None the second term is omitted (e.g. threshold block where
/// d²q_t/deta_t² = 0).
fn chain_rule_weights(
    d1_q: &Array1<f64>,
    d2_q: &Array1<f64>,
    dq: &Array1<f64>,
    d2q: Option<&Array1<f64>>,
) -> (Array1<f64>, Array1<f64>) {
    let grad = d1_q * dq;
    let mut hess = -(d2_q * &dq.mapv(|v| v * v));
    if let Some(d2q) = d2q {
        hess = &hess - &(d1_q * d2q);
    }
    (grad, hess)
}

/// Directional Hessian weights D_u[diag(h)] for a single side.
///
///   dh[i] = -[d3_q · dq³ + 3 · d2_q · dq · d2q + d1_q · d3q] · d_eta[i]
///
/// When `d2q`/`d3q` are None, the second and third terms are omitted
/// (threshold block where all higher dq derivatives vanish).
fn directional_hessian_weights(
    d1_q: &Array1<f64>,
    d2_q: &Array1<f64>,
    d3_q: &Array1<f64>,
    dq: &Array1<f64>,
    d2q: Option<&Array1<f64>>,
    d3q: Option<&Array1<f64>>,
    d_eta: &Array1<f64>,
) -> Array1<f64> {
    let mut w = -(&(d3_q * &dq.mapv(|v| v * v * v)));
    if let Some(d2q) = d2q {
        w = &w - &(d2_q * &(3.0 * dq * d2q));
    }
    if let Some(d3q) = d3q {
        w = &w - &(d1_q * d3q);
    }
    &w * d_eta
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
        let (r, c) = s.dim();
        if r != p || c != p {
            return Err(format!("{name} penalty {idx} must be {p}x{p}, got {r}x{c}"));
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
                let (r, c) = s.dim();
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
fn rowwise_kronecker(cov_design: &DesignMatrix, time_basis: &Array2<f64>) -> DesignMatrix {
    let n = cov_design.nrows();
    let p_cov = cov_design.ncols();
    let p_time = time_basis.ncols();
    let p_tensor = p_cov * p_time;

    // For B-spline time basis with degree d, each row has d+1 nonzeros.
    // If covariate design is dense: density = (d+1)/p_time ≈ 4/12 ≈ 33% → sparse.
    // If covariate design is also sparse: even sparser.
    // Use sparse when estimated density < 50%.
    let cov_is_sparse = matches!(cov_design, DesignMatrix::Sparse(_));
    let max_time_nnz_per_row = p_time.min(8); // B-spline degree 3 → 4 nonzero
    let estimated_nnz_per_row = if cov_is_sparse {
        // Conservative estimate
        p_cov.min(p_cov) * max_time_nnz_per_row
    } else {
        p_cov * max_time_nnz_per_row
    };
    let use_sparse = estimated_nnz_per_row * 2 < p_tensor;

    if use_sparse {
        let mut triplets = Vec::with_capacity(n * estimated_nnz_per_row);
        match cov_design {
            DesignMatrix::Dense(x_cov) => {
                for i in 0..n {
                    for jc in 0..p_cov {
                        let cv = x_cov[[i, jc]];
                        if cv == 0.0 {
                            continue;
                        }
                        for jt in 0..p_time {
                            let tv = time_basis[[i, jt]];
                            if tv == 0.0 {
                                continue;
                            }
                            triplets.push(faer::sparse::Triplet::new(i, jc * p_time + jt, cv * tv));
                        }
                    }
                }
            }
            DesignMatrix::Sparse(xs) => {
                let csr = xs.to_csr_arc().expect("CSR view for rowwise kronecker");
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                for i in 0..n {
                    for ptr in row_ptr[i]..row_ptr[i + 1] {
                        let jc = col_idx[ptr];
                        let cv = vals[ptr];
                        if cv == 0.0 {
                            continue;
                        }
                        for jt in 0..p_time {
                            let tv = time_basis[[i, jt]];
                            if tv == 0.0 {
                                continue;
                            }
                            triplets.push(faer::sparse::Triplet::new(i, jc * p_time + jt, cv * tv));
                        }
                    }
                }
            }
        }
        DesignMatrix::from(
            faer::sparse::SparseColMat::try_new_from_triplets(n, p_tensor, &triplets)
                .expect("build sparse tensor product design"),
        )
    } else {
        let mut out = Array2::<f64>::zeros((n, p_tensor));
        match cov_design {
            DesignMatrix::Dense(x_cov) => {
                for i in 0..n {
                    for jc in 0..p_cov {
                        let cv = x_cov[[i, jc]];
                        if cv == 0.0 {
                            continue;
                        }
                        for jt in 0..p_time {
                            out[[i, jc * p_time + jt]] = cv * time_basis[[i, jt]];
                        }
                    }
                }
            }
            DesignMatrix::Sparse(xs) => {
                let csr = xs.to_csr_arc().expect("CSR view for rowwise kronecker");
                let sym = csr.symbolic();
                let row_ptr = sym.row_ptr();
                let col_idx = sym.col_idx();
                let vals = csr.val();
                for i in 0..n {
                    for ptr in row_ptr[i]..row_ptr[i + 1] {
                        let jc = col_idx[ptr];
                        let cv = vals[ptr];
                        for jt in 0..p_time {
                            out[[i, jc * p_time + jt]] = cv * time_basis[[i, jt]];
                        }
                    }
                }
            }
        }
        DesignMatrix::Dense(out)
    }
}

/// Prepared covariate block data for the family struct.
struct PreparedCovBlock {
    /// Exit design (used as the solver's primary).
    design_exit: DesignMatrix,
    /// Entry design, only for time-varying blocks.
    design_entry: Option<DesignMatrix>,
    /// Offset (same for both entry/exit since it comes from other terms).
    offset: Array1<f64>,
    penalties: Vec<Array2<f64>>,
    initial_log_lambdas: Option<Array1<f64>>,
    initial_beta: Option<Array1<f64>>,
}

fn prepare_cov_block_kind(bk: &CovariateBlockKind) -> Result<PreparedCovBlock, String> {
    match bk {
        CovariateBlockKind::Static(b) => Ok(PreparedCovBlock {
            design_exit: b.design.clone(),
            design_entry: None,
            offset: b.offset.clone(),
            penalties: b.penalties.clone(),
            initial_log_lambdas: b.initial_log_lambdas.clone(),
            initial_beta: b.initial_beta.clone(),
        }),
        CovariateBlockKind::TimeVarying(tv) => {
            let design_exit = rowwise_kronecker(&tv.design_covariates, &tv.time_basis_exit);
            let design_entry = rowwise_kronecker(&tv.design_covariates, &tv.time_basis_entry);
            Ok(PreparedCovBlock {
                design_exit,
                design_entry: Some(design_entry),
                offset: tv.offset.clone(),
                penalties: tv.penalties.clone(),
                initial_log_lambdas: tv.initial_log_lambdas.clone(),
                initial_beta: tv.initial_beta.clone(),
            })
        }
    }
}

fn build_survival_covariate_block_from_design(
    cov_design: &TermCollectionDesign,
    template: &SurvivalCovariateTermBlockTemplate,
    initial_log_lambdas: Option<Array1<f64>>,
    initial_beta: Option<Array1<f64>>,
) -> Result<CovariateBlockKind, String> {
    match template {
        SurvivalCovariateTermBlockTemplate::Static => {
            Ok(CovariateBlockKind::Static(CovariateBlockInput {
                design: DesignMatrix::Dense(cov_design.design.clone()),
                offset: Array1::zeros(cov_design.design.nrows()),
                penalties: cov_design.penalties.clone(),
                initial_log_lambdas,
                initial_beta,
            }))
        }
        SurvivalCovariateTermBlockTemplate::TimeVarying {
            time_basis_entry,
            time_basis_exit,
            time_penalties,
        } => {
            let p_cov = cov_design.design.ncols();
            let p_time = time_basis_exit.ncols();
            let design_covariates = DesignMatrix::Dense(cov_design.design.clone());
            let i_cov = Array2::<f64>::eye(p_cov);
            let i_time = Array2::<f64>::eye(p_time);
            let mut penalties =
                Vec::with_capacity(cov_design.penalties.len() + time_penalties.len());
            for s_cov in &cov_design.penalties {
                penalties.push(kronecker_product(s_cov, &i_time));
            }
            for s_time in time_penalties {
                penalties.push(kronecker_product(&i_cov, s_time));
            }
            Ok(CovariateBlockKind::TimeVarying(
                TimeDependentCovariateBlockInput {
                    design_covariates,
                    time_basis_entry: time_basis_entry.clone(),
                    time_basis_exit: time_basis_exit.clone(),
                    penalties,
                    initial_log_lambdas,
                    initial_beta,
                    offset: Array1::zeros(cov_design.design.nrows()),
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
    Ok(Some(
        info_list
            .into_iter()
            .enumerate()
            .map(|(psi_idx, info)| {
                let penalty_indices = info.penalty_indices.clone();
                match template {
                    SurvivalCovariateTermBlockTemplate::Static => {
                        let x_full = EmbeddedColumnBlock::new(
                            &info.x_psi_local,
                            info.global_range.clone(),
                            info.total_p,
                        )
                        .materialize();
                        let s_full = EmbeddedSquareBlock::new(
                            &info.s_psi_local,
                            info.global_range.clone(),
                            info.total_p,
                        )
                        .materialize();
                        CustomFamilyBlockPsiDerivative {
                            penalty_index: Some(info.penalty_index),
                            x_psi: x_full.clone(),
                            s_psi: s_full,
                            s_psi_components: Some(
                                info.penalty_indices
                                    .into_iter()
                                    .zip(info.s_psi_components_local.into_iter().map(|local| {
                                        EmbeddedSquareBlock::new(
                                            &local,
                                            info.global_range.clone(),
                                            info.total_p,
                                        )
                                        .materialize()
                                    }))
                                    .collect(),
                            ),
                            x_psi_psi: Some({
                                let mut rows =
                                    vec![
                                        Array2::<f64>::zeros((x_full.nrows(), x_full.ncols()));
                                        psi_dim
                                    ];
                                rows[psi_idx] = EmbeddedColumnBlock::new(
                                    &info.x_psi_psi_local,
                                    info.global_range.clone(),
                                    info.total_p,
                                )
                                .materialize();
                                rows
                            }),
                            s_psi_psi: Some({
                                let mut rows =
                                    vec![
                                        Array2::<f64>::zeros((info.total_p, info.total_p));
                                        psi_dim
                                    ];
                                rows[psi_idx] = EmbeddedSquareBlock::new(
                                    &info.s_psi_psi_local,
                                    info.global_range.clone(),
                                    info.total_p,
                                )
                                .materialize();
                                rows
                            }),
                            s_psi_psi_components: Some({
                                let mut rows = vec![Vec::<(usize, Array2<f64>)>::new(); psi_dim];
                                rows[psi_idx] = penalty_indices
                                    .into_iter()
                                    .zip(info.s_psi_psi_components_local.into_iter().map(|local| {
                                        EmbeddedSquareBlock::new(
                                            &local,
                                            info.global_range.clone(),
                                            info.total_p,
                                        )
                                        .materialize()
                                    }))
                                    .collect();
                                rows
                            }),
                            implicit_operator: info.implicit_operator.clone(),
                            implicit_axis: info.implicit_axis,
                            implicit_group_id: info.aniso_group_id,
                        }
                    }
                    SurvivalCovariateTermBlockTemplate::TimeVarying {
                        time_basis_entry,
                        time_basis_exit,
                        ..
                    } => {
                        let base_x_psi = if info.x_psi_local.nrows() == design.design.nrows()
                            && info.x_psi_local.ncols() == info.global_range.len()
                        {
                            EmbeddedColumnBlock::new(
                                &info.x_psi_local,
                                info.global_range.clone(),
                                info.total_p,
                            )
                            .materialize()
                        } else if let Some(op) = info.implicit_operator.as_ref() {
                            op.materialize_first(info.implicit_axis)
                        } else {
                            EmbeddedColumnBlock::new(
                                &info.x_psi_local,
                                info.global_range.clone(),
                                info.total_p,
                            )
                            .materialize()
                        };
                        let exit = rowwise_kronecker(
                            &DesignMatrix::Dense(base_x_psi.clone()),
                            time_basis_exit,
                        )
                        .to_dense();
                        let entry =
                            rowwise_kronecker(&DesignMatrix::Dense(base_x_psi), time_basis_entry)
                                .to_dense();
                        let n = exit.nrows();
                        let p = exit.ncols();
                        let mut stacked = Array2::<f64>::zeros((2 * n, p));
                        stacked.slice_mut(s![0..n, ..]).assign(&exit);
                        stacked.slice_mut(s![n..2 * n, ..]).assign(&entry);

                        let i_time = Array2::<f64>::eye(time_basis_exit.ncols());
                        let mut s_components = Vec::new();
                        for (penalty_idx, local) in
                            info.s_psi_components_local.into_iter().enumerate()
                        {
                            s_components.push((
                                penalty_indices[penalty_idx],
                                kronecker_product(&local, &i_time),
                            ));
                        }
                        let p_total = p;
                        CustomFamilyBlockPsiDerivative {
                            penalty_index: Some(info.penalty_index),
                            x_psi: stacked,
                            s_psi: s_components
                                .iter()
                                .fold(Array2::<f64>::zeros((p_total, p_total)), |acc, (_, m)| {
                                    acc + m
                                }),
                            s_psi_components: Some(s_components),
                            x_psi_psi: Some(vec![Array2::<f64>::zeros((2 * n, p_total)); psi_dim]),
                            s_psi_psi: Some(vec![
                                Array2::<f64>::zeros((p_total, p_total));
                                psi_dim
                            ]),
                            s_psi_psi_components: Some(vec![Vec::new(); psi_dim]),
                            implicit_operator: None,
                            implicit_axis: info.implicit_axis,
                            implicit_group_id: info.aniso_group_id,
                        }
                    }
                }
            })
            .collect(),
    ))
}

fn build_survival_two_block_exact_joint_setup(
    thresholdspec: &TermCollectionSpec,
    log_sigmaspec: &TermCollectionSpec,
    rho0: Array1<f64>,
    kappa_options: &SpatialLengthScaleOptimizationOptions,
) -> TwoBlockExactJointHyperSetup {
    let threshold_terms = spatial_length_scale_term_indices(thresholdspec);
    let log_sigma_terms = spatial_length_scale_term_indices(log_sigmaspec);
    let rho_lower = Array1::<f64>::from_elem(rho0.len(), -12.0);
    let rho_upper = Array1::<f64>::from_elem(rho0.len(), 12.0);

    let threshold_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        thresholdspec,
        &threshold_terms,
        kappa_options,
    );
    let log_sigma_kappa = SpatialLogKappaCoords::from_length_scales_aniso(
        log_sigmaspec,
        &log_sigma_terms,
        kappa_options,
    );
    let mut all_values = threshold_kappa.as_array().to_vec();
    all_values.extend(log_sigma_kappa.as_array().iter());
    let mut all_dims = threshold_kappa.dims_per_term().to_vec();
    all_dims.extend(log_sigma_kappa.dims_per_term());
    let log_kappa0 =
        SpatialLogKappaCoords::new_with_dims(Array1::from_vec(all_values), all_dims.clone());
    let log_kappa_lower = SpatialLogKappaCoords::lower_bounds_aniso(&all_dims, kappa_options);
    let log_kappa_upper = SpatialLogKappaCoords::upper_bounds_aniso(&all_dims, kappa_options);

    TwoBlockExactJointHyperSetup::new(
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

fn survival_blockwise_fit_options(spec: &SurvivalLocationScaleSpec) -> BlockwiseFitOptions {
    BlockwiseFitOptions {
        inner_max_cycles: spec.max_iter,
        inner_tol: spec.tol,
        outer_max_iter: 60,
        outer_tol: 1e-5,
        ..BlockwiseFitOptions::default()
    }
}

fn validate_survival_location_scale_spec(spec: &SurvivalLocationScaleSpec) -> Result<(), String> {
    let n = spec.event_target.len();
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
    validate_time_block(n, &spec.time_block)?;
    validate_cov_block_kind("threshold_block", n, &spec.threshold_block)?;
    validate_cov_block_kind("log_sigma_block", n, &spec.log_sigma_block)?;
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
    let anchorrow = select_anchorrow(&spec.age_entry, spec.time_anchor)?;
    let mut time_prepared = prepare_identified_time_block(&spec.time_block, anchorrow)?;

    if time_prepared.initial_beta.is_none() {
        let deriv_offset_max = spec
            .time_block
            .derivative_offset_exit
            .iter()
            .fold(0.0_f64, |a, &v| a.max(v.abs()));
        if deriv_offset_max < 1e-8 {
            let x = &time_prepared.design_derivative_exit;
            let p = x.ncols();
            if p > 0 && n > 0 {
                let mut target = Array1::<f64>::zeros(n);
                for i in 0..n {
                    target[i] = 1.0 / spec.age_exit[i].max(1e-9);
                }
                let xtx = x.t().dot(x);
                let xty = x.t().dot(&target);
                let eps = 1e-6 * (0..p).map(|i| xtx[[i, i]]).fold(0.0_f64, f64::max).max(1.0);
                let mut lhs = xtx;
                for i in 0..p {
                    lhs[[i, i]] += eps;
                }
                use crate::faer_ndarray::FaerCholesky;
                if let Ok(chol) = lhs.cholesky(faer::Side::Lower) {
                    let beta_init = chol.solvevec(&xty);
                    let d_raw_init = x.dot(&beta_init);
                    if d_raw_init.iter().all(|v| v.is_finite() && *v > 0.0) {
                        time_prepared.initial_beta = Some(beta_init);
                    }
                }
            }
        }
    }

    let time_stacked_design = stack_time_design(&TimeBlockInput {
        design_entry: time_prepared.design_entry.clone(),
        design_exit: time_prepared.design_exit.clone(),
        design_derivative_exit: time_prepared.design_derivative_exit.clone(),
        offset_entry: spec.time_block.offset_entry.clone(),
        offset_exit: spec.time_block.offset_exit.clone(),
        derivative_offset_exit: spec.time_block.derivative_offset_exit.clone(),
        penalties: time_prepared.penalties.clone(),
        initial_log_lambdas: spec.time_block.initial_log_lambdas.clone(),
        initial_beta: time_prepared.initial_beta.clone(),
    });
    let time_stacked_offset = stack_time_offset(&spec.time_block);
    let timespec = ParameterBlockSpec {
        name: "time_transform".to_string(),
        design: DesignMatrix::Dense(time_stacked_design),
        offset: time_stacked_offset,
        penalties: time_prepared.penalties.clone(),
        initial_log_lambdas: initial_log_lambdas(
            &time_prepared.penalties,
            spec.time_block.initial_log_lambdas.clone(),
        )?,
        initial_beta: time_prepared.initial_beta.clone(),
    };

    let threshold_prep = prepare_cov_block_kind(&spec.threshold_block)?;
    let (threshold_solver_design, threshold_solver_offset) =
        if let Some(x_entry) = threshold_prep.design_entry.as_ref() {
            let exit_dense = threshold_prep.design_exit.to_dense();
            let entry_dense = x_entry.to_dense();
            let p = exit_dense.ncols();
            let mut stacked = Array2::<f64>::zeros((2 * n, p));
            stacked.slice_mut(s![0..n, ..]).assign(&exit_dense);
            stacked.slice_mut(s![n..2 * n, ..]).assign(&entry_dense);
            let mut offset_stacked = Array1::<f64>::zeros(2 * n);
            offset_stacked
                .slice_mut(s![0..n])
                .assign(&threshold_prep.offset);
            offset_stacked
                .slice_mut(s![n..2 * n])
                .assign(&threshold_prep.offset);
            (DesignMatrix::Dense(stacked), offset_stacked)
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
        initial_log_lambdas: initial_log_lambdas(
            &threshold_prep.penalties,
            threshold_prep.initial_log_lambdas.clone(),
        )?,
        initial_beta: threshold_prep.initial_beta.clone(),
    };

    let time_primary_design = time_prepared.design_exit.clone();
    let threshold_primary_design = threshold_prep.design_exit.to_dense();
    let mut survival_primary_design = Array2::<f64>::zeros((
        n,
        time_primary_design.ncols() + threshold_primary_design.ncols(),
    ));
    survival_primary_design
        .slice_mut(s![.., 0..time_primary_design.ncols()])
        .assign(&time_primary_design);
    survival_primary_design
        .slice_mut(s![.., time_primary_design.ncols()..])
        .assign(&threshold_primary_design);

    let log_sigma_prep = prepare_cov_block_kind(&spec.log_sigma_block)?;
    let raw_log_sigma_design = log_sigma_prep.design_exit.to_dense();
    let non_intercept_start = infer_non_intercept_start(&raw_log_sigma_design, &spec.weights);
    let scale_transform = build_scale_deviation_transform(
        &survival_primary_design,
        &raw_log_sigma_design,
        &spec.weights,
        non_intercept_start,
    )?;
    let log_sigma_design = apply_scale_deviation_transform(
        &survival_primary_design,
        &raw_log_sigma_design,
        &scale_transform,
    )?;
    let log_sigma_entry_design = if let Some(x_ls_entry) = log_sigma_prep.design_entry.as_ref() {
        Some(apply_scale_deviation_transform(
            &survival_primary_design,
            &x_ls_entry.to_dense(),
            &scale_transform,
        )?)
    } else {
        None
    };
    let (log_sigma_solver_design, log_sigma_solver_offset) =
        if let Some(ref ls_entry) = log_sigma_entry_design {
            let p = log_sigma_design.ncols();
            let mut stacked = Array2::<f64>::zeros((2 * n, p));
            stacked.slice_mut(s![0..n, ..]).assign(&log_sigma_design);
            stacked.slice_mut(s![n..2 * n, ..]).assign(ls_entry);
            let mut offset_stacked = Array1::<f64>::zeros(2 * n);
            offset_stacked
                .slice_mut(s![0..n])
                .assign(&log_sigma_prep.offset);
            offset_stacked
                .slice_mut(s![n..2 * n])
                .assign(&log_sigma_prep.offset);
            (DesignMatrix::Dense(stacked), offset_stacked)
        } else {
            (
                DesignMatrix::Dense(log_sigma_design.clone()),
                log_sigma_prep.offset.clone(),
            )
        };
    let log_sigmaspec = ParameterBlockSpec {
        name: "log_sigma".to_string(),
        design: log_sigma_solver_design,
        offset: log_sigma_solver_offset,
        penalties: log_sigma_prep.penalties.clone(),
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
            penalties: w.penalties.clone(),
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
        derivative_softness: spec.derivative_softness,
        x_time_entry: time_prepared.design_entry.clone(),
        x_time_exit: time_prepared.design_exit.clone(),
        x_time_deriv: time_prepared.design_derivative_exit.clone(),
        offset_time_deriv: spec.time_block.derivative_offset_exit.clone(),
        x_threshold: threshold_prep.design_exit.clone(),
        x_threshold_entry: threshold_prep.design_entry.clone(),
        x_log_sigma: DesignMatrix::Dense(log_sigma_design),
        x_log_sigma_entry: log_sigma_entry_design.map(DesignMatrix::Dense),
        x_link_wiggle: wigglespec.as_ref().map(|s| s.design.clone()),
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
    if let Some(beta0) = &b.initial_beta
        && beta0.len() != p
    {
        return Err(format!(
            "linkwiggle_block initial_beta length mismatch: got {}, expected {p}",
            beta0.len()
        ));
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
        let (r, c) = s.dim();
        if r != p || c != p {
            return Err(format!(
                "linkwiggle_block penalty {idx} must be {p}x{p}, got {r}x{c}"
            ));
        }
    }
    Ok(())
}

fn inverse_link_supports_joint_wiggle(inverse_link: &InverseLink) -> bool {
    matches!(
        inverse_link,
        InverseLink::Standard(LinkFunction::Identity)
            | InverseLink::Standard(LinkFunction::Logit)
            | InverseLink::Standard(LinkFunction::Probit)
            | InverseLink::Standard(LinkFunction::CLogLog)
    )
}

fn validate_time_block(n: usize, b: &TimeBlockInput) -> Result<(), String> {
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

fn stack_time_design(input: &TimeBlockInput) -> Array2<f64> {
    let n = input.design_exit.nrows();
    let p = input.design_exit.ncols();
    let mut out = Array2::<f64>::zeros((3 * n, p));
    out.slice_mut(s![0..n, ..]).assign(&input.design_entry);
    out.slice_mut(s![n..2 * n, ..]).assign(&input.design_exit);
    out.slice_mut(s![2 * n..3 * n, ..])
        .assign(&input.design_derivative_exit);
    out
}

fn stack_time_offset(input: &TimeBlockInput) -> Array1<f64> {
    let n = input.offset_exit.len();
    let mut out = Array1::<f64>::zeros(3 * n);
    out.slice_mut(s![0..n]).assign(&input.offset_entry);
    out.slice_mut(s![n..2 * n]).assign(&input.offset_exit);
    out.slice_mut(s![2 * n..3 * n])
        .assign(&input.derivative_offset_exit);
    out
}

#[derive(Clone)]
struct TimeIdentifiabilityTransform {
    z: Array2<f64>,
}

#[derive(Clone)]
struct TimeBlockPrepared {
    design_entry: Array2<f64>,
    design_exit: Array2<f64>,
    design_derivative_exit: Array2<f64>,
    penalties: Vec<Array2<f64>>,
    initial_beta: Option<Array1<f64>>,
    transform: TimeIdentifiabilityTransform,
}

fn prepare_identified_time_block(
    input: &TimeBlockInput,
    anchorrow: usize,
) -> Result<TimeBlockPrepared, String> {
    let p = input.design_exit.ncols();
    if p < 2 {
        return Err(format!(
            "time_block needs at least 2 columns for identifiability, got {p}"
        ));
    }
    if anchorrow >= input.design_exit.nrows() {
        return Err(format!(
            "time_block anchor row out of bounds: got {anchorrow}, nrows={}",
            input.design_exit.nrows()
        ));
    }

    // Identifiability: enforce h(t_anchor)=0 by constraining c^T beta = 0,
    // where c is the time basis row at anchor time. Reparameterize beta = Z theta
    // with columns of Z spanning null(c^T), so the constraint is exact for all theta.
    //
    // When entry times are degenerate (no left truncation, all entry = 0),
    // the entry design row is all zeros (I-splines = 0 at the left boundary),
    // making the constraint trivial. Fall back to the exit design row,
    // which always has non-trivial basis values.
    let c_entry = input.design_entry.row(anchorrow);
    let entry_norm = c_entry.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let c = if entry_norm > 1e-12 {
        c_entry.to_owned()
    } else {
        input.design_exit.row(anchorrow).to_owned()
    };
    let c_col = c.view().insert_axis(Axis(1)).to_owned();
    let (z, rank) = rrqr_nullspace_basis(&c_col, default_rrqr_rank_alpha())
        .map_err(|e| format!("time_block identifiability RRQR failed: {e}"))?;
    if rank >= p || z.ncols() == 0 {
        return Err(
            "time_block identifiability constraint removed all columns; add richer time basis"
                .to_string(),
        );
    }
    // For a single anchor constraint c^T beta = 0, the admissible coefficient
    // space is the nullspace of the p x 1 column matrix c.
    let design_entry = input.design_entry.dot(&z);
    let design_exit = input.design_exit.dot(&z);
    let design_derivative_exit = input.design_derivative_exit.dot(&z);
    let penalties = input
        .penalties
        .iter()
        .map(|s| z.t().dot(s).dot(&z))
        .collect::<Vec<_>>();
    let initial_beta = input.initial_beta.as_ref().map(|b| z.t().dot(b));

    Ok(TimeBlockPrepared {
        design_entry,
        design_exit,
        design_derivative_exit,
        penalties,
        initial_beta,
        transform: TimeIdentifiabilityTransform { z },
    })
}

fn initial_log_lambdas(
    penalties: &[Array2<f64>],
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

fn select_anchorrow(age_entry: &Array1<f64>, time_anchor: Option<f64>) -> Result<usize, String> {
    if age_entry.is_empty() {
        return Err("select_anchorrow: empty age_entry".to_string());
    }
    match time_anchor {
        None => age_entry
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .ok_or_else(|| "select_anchorrow: failed to select earliest entry".to_string()),
        Some(t_anchor) => {
            if !t_anchor.is_finite() {
                return Err(format!(
                    "fit_survival_location_scale: non-finite time_anchor {t_anchor}"
                ));
            }
            age_entry
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    let da = (a.1 - t_anchor).abs();
                    let db = (b.1 - t_anchor).abs();
                    da.total_cmp(&db)
                })
                .map(|(i, _)| i)
                .ok_or_else(|| "select_anchorrow: failed to select nearest entry".to_string())
        }
    }
}

fn design_times_dense(design: &DesignMatrix, rhs: &Array2<f64>) -> Result<Array2<f64>, String> {
    if design.ncols() != rhs.nrows() {
        return Err(format!(
            "design_times_dense shape mismatch: design is {}x{}, rhs is {}x{}",
            design.nrows(),
            design.ncols(),
            rhs.nrows(),
            rhs.ncols()
        ));
    }
    match design {
        DesignMatrix::Dense(x) => Ok(x.dot(rhs)),
        DesignMatrix::Sparse(_) => {
            let n = design.nrows();
            let q = rhs.ncols();
            let mut out = Array2::<f64>::zeros((n, q));
            for j in 0..q {
                let col = rhs.column(j).to_owned();
                out.column_mut(j).assign(&design.matrixvectormultiply(&col));
            }
            Ok(out)
        }
    }
}

fn rowwise_dot_designwith_dense(
    design: &DesignMatrix,
    rowvalues: &Array2<f64>,
) -> Result<Array1<f64>, String> {
    if design.nrows() != rowvalues.nrows() || design.ncols() != rowvalues.ncols() {
        return Err(format!(
            "rowwise_dot_designwith_dense shape mismatch: design is {}x{}, rowvalues is {}x{}",
            design.nrows(),
            design.ncols(),
            rowvalues.nrows(),
            rowvalues.ncols()
        ));
    }
    match design {
        DesignMatrix::Dense(x) => Ok(Array1::from_iter(
            (0..x.nrows()).map(|i| x.row(i).dot(&rowvalues.row(i))),
        )),
        DesignMatrix::Sparse(xs) => {
            let csr = xs.to_csr_arc().ok_or_else(|| {
                "rowwise_dot_designwith_dense: failed to obtain CSR view".to_string()
            })?;
            let sym = csr.symbolic();
            let row_ptr = sym.row_ptr();
            let col_idx = sym.col_idx();
            let vals = csr.val();
            let mut out = Array1::<f64>::zeros(xs.nrows());
            for i in 0..xs.nrows() {
                let start = row_ptr[i];
                let end = row_ptr[i + 1];
                let mut acc = 0.0_f64;
                for ptr in start..end {
                    let j = col_idx[ptr];
                    acc += vals[ptr] * rowvalues[[i, j]];
                }
                out[i] = acc;
            }
            Ok(out)
        }
    }
}

fn rowwise_cross_quadratic_dense_design(
    left_dense: &Array2<f64>,
    middle: &Array2<f64>,
    right: &DesignMatrix,
) -> Result<Array1<f64>, String> {
    if left_dense.ncols() != middle.nrows() || middle.ncols() != right.ncols() {
        return Err(format!(
            "rowwise_cross_quadratic_dense_design shape mismatch: left is {}x{}, middle is {}x{}, right is {}x{}",
            left_dense.nrows(),
            left_dense.ncols(),
            middle.nrows(),
            middle.ncols(),
            right.nrows(),
            right.ncols()
        ));
    }
    if left_dense.nrows() != right.nrows() {
        return Err(format!(
            "rowwise_cross_quadratic_dense_design row mismatch: left has {} rows, right has {} rows",
            left_dense.nrows(),
            right.nrows()
        ));
    }

    let mut out = Array1::<f64>::zeros(left_dense.nrows());
    let mut row_scratch = vec![0.0_f64; middle.ncols()];
    match right {
        DesignMatrix::Dense(x_right) => {
            for i in 0..left_dense.nrows() {
                row_scratch.fill(0.0);
                for j in 0..left_dense.ncols() {
                    let left_ij = left_dense[[i, j]];
                    if left_ij == 0.0 {
                        continue;
                    }
                    for k in 0..middle.ncols() {
                        row_scratch[k] += left_ij * middle[[j, k]];
                    }
                }
                let mut acc = 0.0_f64;
                for k in 0..x_right.ncols() {
                    acc += row_scratch[k] * x_right[[i, k]];
                }
                out[i] = acc;
            }
        }
        DesignMatrix::Sparse(xs) => {
            let csr = xs.to_csr_arc().ok_or_else(|| {
                "rowwise_cross_quadratic_dense_design: failed to obtain CSR view".to_string()
            })?;
            let sym = csr.symbolic();
            let row_ptr = sym.row_ptr();
            let col_idx = sym.col_idx();
            let vals = csr.val();
            for i in 0..left_dense.nrows() {
                row_scratch.fill(0.0);
                for j in 0..left_dense.ncols() {
                    let left_ij = left_dense[[i, j]];
                    if left_ij == 0.0 {
                        continue;
                    }
                    for k in 0..middle.ncols() {
                        row_scratch[k] += left_ij * middle[[j, k]];
                    }
                }
                let mut acc = 0.0_f64;
                for ptr in row_ptr[i]..row_ptr[i + 1] {
                    let k = col_idx[ptr];
                    acc += vals[ptr] * row_scratch[k];
                }
                out[i] = acc;
            }
        }
    }
    Ok(out)
}

fn rowwise_cross_quadratic_design(
    left: &DesignMatrix,
    middle: &Array2<f64>,
    right: &DesignMatrix,
) -> Result<Array1<f64>, String> {
    let left_middle = design_times_dense(left, middle)?;
    rowwise_dot_designwith_dense(right, &left_middle)
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
    let mut weighted_right = right.clone();
    for i in 0..weighted_right.nrows() {
        let wi = weights[i];
        if wi == 1.0 {
            continue;
        }
        weighted_right.row_mut(i).mapv_inplace(|v| wi * v);
    }
    Ok(left.t().dot(&weighted_right))
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
        InverseLink::Standard(LinkFunction::Probit) => (1.0 - normal_cdf(eta)).clamp(0.0, 1.0),
        InverseLink::Standard(LinkFunction::Logit) => {
            let e = eta.clamp(-700.0, 700.0);
            (1.0 / (1.0 + e.exp())).clamp(0.0, 1.0)
        }
        InverseLink::Standard(LinkFunction::CLogLog) => {
            let e = eta.clamp(-30.0, 30.0);
            (-e.exp()).exp().clamp(0.0, 1.0)
        }
        InverseLink::Standard(LinkFunction::Identity) => (1.0 - eta).clamp(0.0, 1.0),
        InverseLink::Standard(LinkFunction::Log) => {
            panic!("state-less log inverse link is invalid for survival prediction")
        }
        InverseLink::Sas(_) | InverseLink::BetaLogistic(_) | InverseLink::Mixture(_) => {
            inverse_link_survival_prob_checked(inverse_link, eta)
                .expect("validated inverse link should evaluate during prediction")
        }
        InverseLink::Standard(LinkFunction::Sas)
        | InverseLink::Standard(LinkFunction::BetaLogistic) => {
            panic!("state-less SAS/Beta-Logistic inverse link is invalid for prediction")
        }
    }
}

struct PredictionLinearPredictors {
    h: Array1<f64>,
    eta_t: Array1<f64>,
    eta_ls: Array1<f64>,
    etaw: Option<Array1<f64>>,
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
    if let (Some(xw), Some(betaw)) = (&input.x_link_wiggle, &beta_link_wiggle) {
        if xw.nrows() != n {
            return Err(format!(
                "predict_survival_location_scale: link-wiggle row mismatch: got {}, expected {n}",
                xw.nrows()
            ));
        }
        if xw.ncols() != betaw.len() {
            return Err(format!(
                "predict_survival_location_scale: link-wiggle design/beta mismatch: {} vs {}",
                xw.ncols(),
                betaw.len()
            ));
        }
    } else if input.x_link_wiggle.is_some() || beta_link_wiggle.is_some() {
        return Err(
            "predict_survival_location_scale: link-wiggle metadata is partial; both design and beta must be provided"
                .to_string(),
        );
    }
    Ok(PredictionLinearPredictors {
        h: input.x_time_exit.dot(&beta_time) + &input.eta_time_offset_exit,
        eta_t: input.x_threshold.matrixvectormultiply(&beta_threshold)
            + &input.eta_threshold_offset,
        eta_ls: input.x_log_sigma.matrixvectormultiply(&beta_log_sigma)
            + &input.eta_log_sigma_offset,
        etaw: if let (Some(xw), Some(betaw)) = (&input.x_link_wiggle, &beta_link_wiggle) {
            Some(xw.matrixvectormultiply(betaw))
        } else {
            None
        },
    })
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
        let (h0, h1, d_raw, eta_t_exit, eta_ls_exit, eta_t_entry, eta_ls_entry, etaw) =
            self.validate_joint_states(block_states)?;
        let (sigma_exit, ds_exit, d2s_exit, _) = exp_sigma_derivs_up_to_third(eta_ls_exit.view());
        let sigma_entry = if self.x_log_sigma_entry.is_some() {
            Some(eta_ls_entry.mapv(crate::families::sigma_link::safe_exp))
        } else {
            None
        };
        let mut ll = 0.0;

        let mut grad_time_eta_h0 = Array1::<f64>::zeros(n);
        let mut grad_time_eta_h1 = Array1::<f64>::zeros(n);
        let mut grad_time_eta_d = Array1::<f64>::zeros(n);
        let mut h_time_h0 = Array1::<f64>::zeros(n);
        let mut h_time_h1 = Array1::<f64>::zeros(n);
        let mut h_time_d = Array1::<f64>::zeros(n);

        let mut d1_q = Array1::<f64>::zeros(n);
        let mut d2_q = Array1::<f64>::zeros(n);
        let mut d1_q0 = Array1::<f64>::zeros(n);
        let mut d2_q0 = Array1::<f64>::zeros(n);
        let mut d1_q1 = Array1::<f64>::zeros(n);
        let mut d2_q1 = Array1::<f64>::zeros(n);

        for i in 0..n {
            let sigma_entry_i = sigma_entry.as_ref().map_or(sigma_exit[i], |se| se[i]);
            let state = self.row_predictor_state(
                h0[i],
                h1[i],
                d_raw[i],
                eta_t_entry[i],
                eta_t_exit[i],
                sigma_entry_i,
                sigma_exit[i],
                etaw.map(|w| w[i]),
            );
            let Some(row) = self.row_derivatives(i, state)? else {
                continue;
            };
            ll += row.ll;
            d1_q[i] = row.d1_q;
            d2_q[i] = row.d2_q;
            d1_q0[i] = row.d1_q0;
            d2_q0[i] = row.d2_q0;
            d1_q1[i] = row.d1_q1;
            d2_q1[i] = row.d2_q1;
            grad_time_eta_h0[i] = row.grad_time_eta_h0;
            grad_time_eta_h1[i] = row.grad_time_eta_h1;
            grad_time_eta_d[i] = row.grad_time_eta_d;
            h_time_h0[i] = row.h_time_h0;
            h_time_h1[i] = row.h_time_h1;
            h_time_d[i] = row.h_time_d;
        }

        // Block 0: exact beta-space gradient/Hessian
        let grad_time = self.x_time_entry.t().dot(&grad_time_eta_h0)
            + self.x_time_exit.t().dot(&grad_time_eta_h1)
            + self.x_time_deriv.t().dot(&grad_time_eta_d);
        let hess_time = fast_xt_diag_x(&self.x_time_entry, &h_time_h0)
            + fast_xt_diag_x(&self.x_time_exit, &h_time_h1)
            + fast_xt_diag_x(&self.x_time_deriv, &h_time_d);

        // Block 1: threshold (dq_t = -1/σ, d²q_t = 0).
        let (grad_t, hess_t) = if let Some(x_t_entry) = self.x_threshold_entry.as_ref() {
            let dq_t_exit = sigma_exit.mapv(|s| -1.0 / s.max(1e-12));
            let dq_t_entry = sigma_entry
                .as_ref()
                .map(|se| se.mapv(|s| -1.0 / s.max(1e-12)))
                .unwrap_or_else(|| dq_t_exit.clone());
            let (ge, he) = chain_rule_weights(&d1_q1, &d2_q1, &dq_t_exit, None);
            let (gn, hn) = chain_rule_weights(&d1_q0, &d2_q0, &dq_t_entry, None);
            let grad = self.x_threshold.transpose_vector_multiply(&ge)
                + x_t_entry.transpose_vector_multiply(&gn);
            let hess = xt_diag_x_symmetric(&self.x_threshold, &he)?
                .add(&xt_diag_x_symmetric(x_t_entry, &hn)?)?;
            (grad, hess)
        } else {
            let dq_t = sigma_exit.mapv(|s| -1.0 / s.max(1e-12));
            let (g, h) = chain_rule_weights(&d1_q, &d2_q, &dq_t, None);
            let grad = self.x_threshold.transpose_vector_multiply(&g);
            let hess = xt_diag_x_symmetric(&self.x_threshold, &h)?;
            (grad, hess)
        };

        // Block 2: log-sigma (needs dq_ls and d²q_ls via chain rule).
        let (grad_ls, hess_ls) = if let Some(x_ls_entry) = self.x_log_sigma_entry.as_ref() {
            let sigma_e = sigma_entry.as_ref().unwrap();
            // For exp link: σ' = σ = exp(η_ls)
            let exit_qd = compute_q_chain_derivs_third(
                &eta_t_exit,
                &sigma_exit,
                &ds_exit,
                &d2s_exit,
                &d2s_exit,
            );
            let ds_entry = eta_ls_entry.mapv(crate::families::sigma_link::safe_exp);
            let entry_qd = compute_q_chain_derivs_third(
                &eta_t_entry,
                sigma_e,
                &ds_entry,
                &ds_entry,
                &ds_entry,
            );
            let (ge, he) =
                chain_rule_weights(&d1_q1, &d2_q1, &exit_qd.dq_ls, Some(&exit_qd.d2q_ls));
            let (gn, hn) =
                chain_rule_weights(&d1_q0, &d2_q0, &entry_qd.dq_ls, Some(&entry_qd.d2q_ls));
            let grad = self.x_log_sigma.transpose_vector_multiply(&ge)
                + x_ls_entry.transpose_vector_multiply(&gn);
            let hess = xt_diag_x_symmetric(&self.x_log_sigma, &he)?
                .add(&xt_diag_x_symmetric(x_ls_entry, &hn)?)?;
            (grad, hess)
        } else {
            let exit_qd = compute_q_chain_derivs_third(
                &eta_t_exit,
                &sigma_exit,
                &ds_exit,
                &d2s_exit,
                &d2s_exit,
            );
            let (g, h) = chain_rule_weights(&d1_q, &d2_q, &exit_qd.dq_ls, Some(&exit_qd.d2q_ls));
            let grad = self.x_log_sigma.transpose_vector_multiply(&g);
            let hess = xt_diag_x_symmetric(&self.x_log_sigma, &h)?;
            (grad, hess)
        };

        let mut blockworking_sets = vec![
            BlockWorkingSet::ExactNewton {
                gradient: grad_time,
                hessian: SymmetricMatrix::Dense(hess_time),
            },
            BlockWorkingSet::ExactNewton {
                gradient: grad_t,
                hessian: hess_t,
            },
            BlockWorkingSet::ExactNewton {
                gradient: grad_ls,
                hessian: hess_ls,
            },
        ];
        if let Some(xw) = self.x_link_wiggle.as_ref() {
            let gradw = xw.transpose_vector_multiply(&d1_q);
            let hessw = xt_diag_x_symmetric(xw, &(-&d2_q))?;
            blockworking_sets.push(BlockWorkingSet::ExactNewton {
                gradient: gradw,
                hessian: hessw,
            });
        }

        Ok(FamilyEvaluation {
            log_likelihood: ll,
            blockworking_sets,
        })
    }

    fn exact_newton_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        block_idx: usize,
        d_beta: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let n = self.n;
        let (h0, h1, d_raw, eta_t_exit, eta_ls_exit, eta_t_entry, eta_ls_entry, etaw) =
            self.validate_joint_states(block_states)?;
        let (sigma, ds, d2s, d3s) = exp_sigma_derivs_up_to_third(eta_ls_exit.view());
        let sigma_entry_vec = if self.x_log_sigma_entry.is_some() {
            Some(eta_ls_entry.mapv(crate::families::sigma_link::safe_exp))
        } else {
            None
        };
        let mut d1_q = Array1::<f64>::zeros(n);
        let mut d2_q = Array1::<f64>::zeros(n);
        let mut d3_q = Array1::<f64>::zeros(n);
        let mut d1_q0 = Array1::<f64>::zeros(n);
        let mut d2_q0 = Array1::<f64>::zeros(n);
        let mut d3_q0 = Array1::<f64>::zeros(n);
        let mut d1_q1 = Array1::<f64>::zeros(n);
        let mut d2_q1 = Array1::<f64>::zeros(n);
        let mut d3_q1 = Array1::<f64>::zeros(n);
        let mut d_h_h0 = Array1::<f64>::zeros(n);
        let mut d_h_h1 = Array1::<f64>::zeros(n);
        let mut d_h_d = Array1::<f64>::zeros(n);

        for i in 0..n {
            let sigma_entry_i = sigma_entry_vec.as_ref().map_or(sigma[i], |se| se[i]);
            let state = self.row_predictor_state(
                h0[i],
                h1[i],
                d_raw[i],
                eta_t_entry[i],
                eta_t_exit[i],
                sigma_entry_i,
                sigma[i],
                etaw.map(|w| w[i]),
            );
            let Some(row) = self.row_derivatives(i, state)? else {
                continue;
            };
            d1_q[i] = row.d1_q;
            d2_q[i] = row.d2_q;
            d3_q[i] = row.d3_q;
            d1_q0[i] = row.d1_q0;
            d2_q0[i] = row.d2_q0;
            d3_q0[i] = row.d3_q0;
            d1_q1[i] = row.d1_q1;
            d2_q1[i] = row.d2_q1;
            d3_q1[i] = row.d3_q1;
            d_h_h0[i] = row.d_h_h0;
            d_h_h1[i] = row.d_h_h1;
            d_h_d[i] = row.d_h_d;
        }

        match block_idx {
            Self::BLOCK_TIME => {
                if d_beta.len() != self.x_time_entry.ncols() {
                    return Err(format!(
                        "time block d_beta length mismatch: got {}, expected {}",
                        d_beta.len(),
                        self.x_time_entry.ncols()
                    ));
                }
                let d_eta_h0 = self.x_time_entry.dot(d_beta);
                let d_eta_h1 = self.x_time_exit.dot(d_beta);
                let d_eta_d = self.x_time_deriv.dot(d_beta);
                let w0 = &d_h_h0 * &d_eta_h0;
                let w1 = &d_h_h1 * &d_eta_h1;
                let wd = &d_h_d * &d_eta_d;
                let d_h = fast_xt_diag_x(&self.x_time_entry, &w0)
                    + fast_xt_diag_x(&self.x_time_exit, &w1)
                    + fast_xt_diag_x(&self.x_time_deriv, &wd);
                Ok(Some(d_h))
            }
            Self::BLOCK_THRESHOLD => {
                if d_beta.len() != self.x_threshold.ncols() {
                    return Err(format!(
                        "threshold block d_beta length mismatch: got {}, expected {}",
                        d_beta.len(),
                        self.x_threshold.ncols()
                    ));
                }
                if let Some(x_t_entry) = self.x_threshold_entry.as_ref() {
                    let sigma_e = sigma_entry_vec.as_ref().unwrap();
                    let dq_t_exit = sigma.mapv(|s| -1.0 / s.max(1e-12));
                    let dq_t_entry = sigma_e.mapv(|s| -1.0 / s.max(1e-12));
                    let d_eta_exit = self.x_threshold.matrixvectormultiply(d_beta);
                    let d_eta_entry = x_t_entry.matrixvectormultiply(d_beta);
                    let dh_exit = directional_hessian_weights(
                        &d1_q1,
                        &d2_q1,
                        &d3_q1,
                        &dq_t_exit,
                        None,
                        None,
                        &d_eta_exit,
                    );
                    let dh_entry = directional_hessian_weights(
                        &d1_q0,
                        &d2_q0,
                        &d3_q0,
                        &dq_t_entry,
                        None,
                        None,
                        &d_eta_entry,
                    );
                    let d_h = xt_diag_x_symmetric(&self.x_threshold, &dh_exit)?
                        .add(&xt_diag_x_symmetric(x_t_entry, &dh_entry)?)?;
                    Ok(Some(d_h.to_dense()))
                } else {
                    let dq_t = sigma.mapv(|s| -1.0 / s.max(1e-12));
                    let d_eta_t = self.x_threshold.matrixvectormultiply(d_beta);
                    let dh = directional_hessian_weights(
                        &d1_q, &d2_q, &d3_q, &dq_t, None, None, &d_eta_t,
                    );
                    Ok(Some(
                        xt_diag_x_symmetric(&self.x_threshold, &dh)?.to_dense(),
                    ))
                }
            }
            Self::BLOCK_LOG_SIGMA => {
                if d_beta.len() != self.x_log_sigma.ncols() {
                    return Err(format!(
                        "log-sigma block d_beta length mismatch: got {}, expected {}",
                        d_beta.len(),
                        self.x_log_sigma.ncols()
                    ));
                }
                if let Some(x_ls_entry) = self.x_log_sigma_entry.as_ref() {
                    let sigma_e = sigma_entry_vec.as_ref().unwrap();
                    // For exp link: σ' = σ'' = σ''' = σ
                    let exit_qd =
                        compute_q_chain_derivs_third(&eta_t_exit, &sigma, &ds, &d2s, &d3s);
                    let entry_qd = compute_q_chain_derivs_third(
                        &eta_t_entry.to_owned(),
                        sigma_e,
                        sigma_e,
                        sigma_e,
                        sigma_e,
                    );
                    let d_eta_exit = self.x_log_sigma.matrixvectormultiply(d_beta);
                    let d_eta_entry = x_ls_entry.matrixvectormultiply(d_beta);
                    let dh_exit = directional_hessian_weights(
                        &d1_q1,
                        &d2_q1,
                        &d3_q1,
                        &exit_qd.dq_ls,
                        Some(&exit_qd.d2q_ls),
                        Some(&exit_qd.d3q_ls),
                        &d_eta_exit,
                    );
                    let dh_entry = directional_hessian_weights(
                        &d1_q0,
                        &d2_q0,
                        &d3_q0,
                        &entry_qd.dq_ls,
                        Some(&entry_qd.d2q_ls),
                        Some(&entry_qd.d3q_ls),
                        &d_eta_entry,
                    );
                    let d_h = xt_diag_x_symmetric(&self.x_log_sigma, &dh_exit)?
                        .add(&xt_diag_x_symmetric(x_ls_entry, &dh_entry)?)?;
                    Ok(Some(d_h.to_dense()))
                } else {
                    let exit_qd =
                        compute_q_chain_derivs_third(&eta_t_exit, &sigma, &ds, &d2s, &d3s);
                    let d_eta_ls = self.x_log_sigma.matrixvectormultiply(d_beta);
                    let dh = directional_hessian_weights(
                        &d1_q,
                        &d2_q,
                        &d3_q,
                        &exit_qd.dq_ls,
                        Some(&exit_qd.d2q_ls),
                        Some(&exit_qd.d3q_ls),
                        &d_eta_ls,
                    );
                    Ok(Some(
                        xt_diag_x_symmetric(&self.x_log_sigma, &dh)?.to_dense(),
                    ))
                }
            }
            Self::BLOCK_LINK_WIGGLE => {
                let xw = self.x_link_wiggle.as_ref().ok_or_else(|| {
                    "link wiggle directional derivative requested but wiggle block is disabled"
                        .to_string()
                })?;
                if d_beta.len() != xw.ncols() {
                    return Err(format!(
                        "link-wiggle block d_beta length mismatch: got {}, expected {}",
                        d_beta.len(),
                        xw.ncols()
                    ));
                }
                let d_etaw = xw.matrixvectormultiply(d_beta);
                let d_h_eta = -(&d3_q * &d_etaw);
                let d_h = xt_diag_x_symmetric(xw, &d_h_eta)?;
                Ok(Some(d_h.to_dense()))
            }
            _ => Ok(None),
        }
    }

    fn exact_newton_joint_hessian(
        &self,
        block_states: &[ParameterBlockState],
    ) -> Result<Option<Array2<f64>>, String> {
        let q = self.collect_joint_quantities(block_states)?;
        let offsets = self.joint_block_offsets();
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;
        let x_threshold_exit = self.x_threshold.to_dense();
        let x_threshold_entry = self.x_threshold_entry.as_ref().map(DesignMatrix::to_dense);
        let x_log_sigma_exit = self.x_log_sigma.to_dense();
        let x_log_sigma_entry = self.x_log_sigma_entry.as_ref().map(DesignMatrix::to_dense);
        let xw = self.x_link_wiggle.as_ref().map(DesignMatrix::to_dense);
        let mut joint = Array2::<f64>::zeros((p_total, p_total));

        // Full predictor-space negative Hessian:
        //
        //   H = -d²ℓ/dβ²
        //
        // The time block contributes through three distinct linear predictors
        // (entry h0, exit h1, derivative d), while the remaining blocks enter
        // through the scalar coupled predictor
        //
        //   q(eta_t, eta_ls, etaw) = -eta_t / sigma(eta_ls) + etaw.
        //
        // For time-varying blocks, q0 (entry) and q1 (exit) evaluate at
        // different designs but share the same beta. The Hessian for a
        // coefficient-block pair (b, c) splits:
        //
        //   H_bc = H_bc^{exit} + H_bc^{entry}
        //
        // where H_bc^{exit}  = X_b_exit'  diag(w_exit)  X_c_exit
        //       H_bc^{entry} = X_b_entry' diag(w_entry) X_c_entry
        //
        // with w_exit  = -[ d²ℓ/dq1² * q1_b * q1_c + dℓ/dq1 * q1_bc ]
        //      w_entry = -[ d²ℓ/dq0² * q0_b * q0_c + dℓ/dq0 * q0_bc ].
        //
        // For time-invariant blocks, q0=q1=q, X_entry=X_exit, and entry is
        // omitted (the combined d1_q etc. are used).
        let h_time = fast_xt_diag_x(&self.x_time_entry, &q.h_time_h0)
            + fast_xt_diag_x(&self.x_time_exit, &q.h_time_h1)
            + fast_xt_diag_x(&self.x_time_deriv, &q.h_time_d);
        assign_symmetric_block(&mut joint, offsets[0], offsets[0], &h_time);

        // Threshold-threshold block.
        if let (Some(x_t_en), Some(dq_t_en)) = (x_threshold_entry.as_ref(), q.dq_t_entry.as_ref()) {
            let h_exit = -(&q.d2_q1 * &q.dq_t.mapv(|v| v * v));
            let h_entry = -(&q.d2_q0 * &dq_t_en.mapv(|v| v * v));
            let h_tt = weighted_crossprod_dense(&x_threshold_exit, &h_exit, &x_threshold_exit)?
                + weighted_crossprod_dense(x_t_en, &h_entry, x_t_en)?;
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &h_tt);
        } else {
            let h_t = -(&q.d2_q * &q.dq_t.mapv(|v| v * v));
            let h_tt = weighted_crossprod_dense(&x_threshold_exit, &h_t, &x_threshold_exit)?;
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &h_tt);
        }

        // Log-sigma-log-sigma block.
        if let (Some(x_ls_en), Some(dq_ls_en), Some(d2q_ls_en)) = (
            x_log_sigma_entry.as_ref(),
            q.dq_ls_entry.as_ref(),
            q.d2q_ls_entry.as_ref(),
        ) {
            let h_exit = -(&q.d2_q1 * &q.dq_ls.mapv(|v| v * v) + &(&q.d1_q1 * &q.d2q_ls));
            let h_entry = -(&q.d2_q0 * &dq_ls_en.mapv(|v| v * v) + &(&q.d1_q0 * d2q_ls_en));
            let h_ll = weighted_crossprod_dense(&x_log_sigma_exit, &h_exit, &x_log_sigma_exit)?
                + weighted_crossprod_dense(x_ls_en, &h_entry, x_ls_en)?;
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &h_ll);
        } else {
            let h_ls = -(&q.d2_q * &q.dq_ls.mapv(|v| v * v) + &(&q.d1_q * &q.d2q_ls));
            let h_ll = weighted_crossprod_dense(&x_log_sigma_exit, &h_ls, &x_log_sigma_exit)?;
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &h_ll);
        }

        // Threshold-log-sigma cross block.
        {
            let has_t_entry = x_threshold_entry.is_some();
            let has_ls_entry = x_log_sigma_entry.is_some();
            if has_t_entry || has_ls_entry {
                // At least one block is time-varying. We need to sum the
                // separate exit and entry contributions.
                let dq_t_en = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
                let dq_ls_en = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
                let d2q_tls_en = q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls);
                let x_t_en = x_threshold_entry.as_ref().unwrap_or(&x_threshold_exit);
                let x_ls_en = x_log_sigma_entry.as_ref().unwrap_or(&x_log_sigma_exit);
                let w_exit = -(&q.d2_q1 * &(&q.dq_t * &q.dq_ls) + &(&q.d1_q1 * &q.d2q_tls));
                let w_entry = -(&q.d2_q0 * &(dq_t_en * dq_ls_en) + &(&q.d1_q0 * d2q_tls_en));
                let h_tl = weighted_crossprod_dense(&x_threshold_exit, &w_exit, &x_log_sigma_exit)?
                    + weighted_crossprod_dense(x_t_en, &w_entry, x_ls_en)?;
                assign_symmetric_block(&mut joint, offsets[1], offsets[2], &h_tl);
            } else {
                let h_tlweights = -(&q.d2_q * &(&q.dq_t * &q.dq_ls) + &(&q.d1_q * &q.d2q_tls));
                let h_tl =
                    weighted_crossprod_dense(&x_threshold_exit, &h_tlweights, &x_log_sigma_exit)?;
                assign_symmetric_block(&mut joint, offsets[1], offsets[2], &h_tl);
            }
        }

        // Time-threshold cross block.
        // H_h0,t = X_time_entry' diag(-h_time_h0 * dq_t) X_threshold
        // For time-varying threshold: entry u0 depends on h0 AND q0, and the
        // cross is d²ℓ/(dh0 dq0) * dq0/d(eta_t). But h_time_h0 already encodes
        // d²ℓ/(dh0 du0) = -d²ℓ/(dh0 dq0). So the cross with threshold is
        // h_time_h0 * dq_t evaluated at the appropriate (exit or entry) sigma.
        //
        // h0 only couples with q0 (entry), and h1 only couples with q1 (exit).
        // For time-varying threshold, h0's cross uses dq_t_entry and X_threshold_entry,
        // while h1's cross uses dq_t_exit and X_threshold_exit.
        if let (Some(x_t_en), Some(dq_t_en)) = (x_threshold_entry.as_ref(), q.dq_t_entry.as_ref()) {
            // h0 couples with entry q0: uses entry designs/sigma
            let h_h0_t =
                weighted_crossprod_dense(&self.x_time_entry, &(-&q.h_time_h0 * dq_t_en), x_t_en)?;
            // h1 couples with exit q1: uses exit designs/sigma
            let h_h1_t = weighted_crossprod_dense(
                &self.x_time_exit,
                &(-&q.h_time_h1 * &q.dq_t),
                &x_threshold_exit,
            )?;
            assign_symmetric_block(&mut joint, offsets[0], offsets[1], &(h_h0_t + h_h1_t));
        } else {
            let h_h0_t = weighted_crossprod_dense(
                &self.x_time_entry,
                &(-&q.h_time_h0 * &q.dq_t),
                &x_threshold_exit,
            )?;
            let h_h1_t = weighted_crossprod_dense(
                &self.x_time_exit,
                &(-&q.h_time_h1 * &q.dq_t),
                &x_threshold_exit,
            )?;
            assign_symmetric_block(&mut joint, offsets[0], offsets[1], &(h_h0_t + h_h1_t));
        }

        // Time-log-sigma cross block.
        if let (Some(x_ls_en), Some(dq_ls_en)) =
            (x_log_sigma_entry.as_ref(), q.dq_ls_entry.as_ref())
        {
            let h_h0_ls =
                weighted_crossprod_dense(&self.x_time_entry, &(-&q.h_time_h0 * dq_ls_en), x_ls_en)?;
            let h_h1_ls = weighted_crossprod_dense(
                &self.x_time_exit,
                &(-&q.h_time_h1 * &q.dq_ls),
                &x_log_sigma_exit,
            )?;
            assign_symmetric_block(&mut joint, offsets[0], offsets[2], &(h_h0_ls + h_h1_ls));
        } else {
            let h_h0_ls = weighted_crossprod_dense(
                &self.x_time_entry,
                &(-&q.h_time_h0 * &q.dq_ls),
                &x_log_sigma_exit,
            )?;
            let h_h1_ls = weighted_crossprod_dense(
                &self.x_time_exit,
                &(-&q.h_time_h1 * &q.dq_ls),
                &x_log_sigma_exit,
            )?;
            assign_symmetric_block(&mut joint, offsets[0], offsets[2], &(h_h0_ls + h_h1_ls));
        }

        if let (Some(xw_dense), Some(w_offset)) = (xw.as_ref(), offsets.get(3).copied()) {
            let hww = weighted_crossprod_dense(xw_dense, &(-&q.d2_q), xw_dense)?;
            assign_symmetric_block(&mut joint, w_offset, w_offset, &hww);

            // Threshold-wiggle cross block.
            // d²ℓ/(d(eta_t) dw) = d²ℓ/dq² * dq/d(eta_t) * dq/dw
            // where dq/dw = 1. For time-varying threshold, split entry/exit.
            if let (Some(x_t_en), Some(dq_t_en)) =
                (x_threshold_entry.as_ref(), q.dq_t_entry.as_ref())
            {
                let h_tw_exit =
                    weighted_crossprod_dense(&x_threshold_exit, &(-&q.d2_q1 * &q.dq_t), xw_dense)?;
                let h_tw_entry =
                    weighted_crossprod_dense(x_t_en, &(-&q.d2_q0 * dq_t_en), xw_dense)?;
                assign_symmetric_block(&mut joint, offsets[1], w_offset, &(h_tw_exit + h_tw_entry));
            } else {
                let h_tw =
                    weighted_crossprod_dense(&x_threshold_exit, &(-&q.d2_q * &q.dq_t), xw_dense)?;
                assign_symmetric_block(&mut joint, offsets[1], w_offset, &h_tw);
            }

            // Log-sigma-wiggle cross block.
            // d²ℓ/(d(eta_ls) dw) = d²ℓ/dq² * dq/d(eta_ls) * dq/dw + dℓ/dq * d²q/(d(eta_ls) dw)
            // The second term is zero (d²q/(d(eta_ls)dw) = 0 since dq/dw=1 is
            // independent of eta_ls). So only the first term remains.
            if let (Some(x_ls_en), Some(dq_ls_en)) =
                (x_log_sigma_entry.as_ref(), q.dq_ls_entry.as_ref())
            {
                let h_lw_exit =
                    weighted_crossprod_dense(&x_log_sigma_exit, &(-&q.d2_q1 * &q.dq_ls), xw_dense)?;
                let h_lw_entry =
                    weighted_crossprod_dense(x_ls_en, &(-&q.d2_q0 * dq_ls_en), xw_dense)?;
                assign_symmetric_block(&mut joint, offsets[2], w_offset, &(h_lw_exit + h_lw_entry));
            } else {
                let h_lw =
                    weighted_crossprod_dense(&x_log_sigma_exit, &(-&q.d2_q * &q.dq_ls), xw_dense)?;
                assign_symmetric_block(&mut joint, offsets[2], w_offset, &h_lw);
            }

            let h_h0w = weighted_crossprod_dense(&self.x_time_entry, &(-&q.h_time_h0), xw_dense)?;
            let h_h1w = weighted_crossprod_dense(&self.x_time_exit, &(-&q.h_time_h1), xw_dense)?;
            assign_symmetric_block(&mut joint, offsets[0], w_offset, &(h_h0w + h_h1w));
        }

        Ok(Some(joint))
    }

    fn exact_newton_joint_hessian_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let q = self.collect_joint_quantities(block_states)?;
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
        let delta_d = self.x_time_deriv.dot(&time_dir);
        // Exit predictor-space deltas (always present).
        let delta_t_exit = self.x_threshold.matrixvectormultiply(&threshold_dir);
        let delta_ls_exit = self.x_log_sigma.matrixvectormultiply(&log_sigma_dir);
        let deltaw = match (self.x_link_wiggle.as_ref(), wiggle_dir.as_ref()) {
            (Some(xw), Some(dir)) => xw.matrixvectormultiply(dir),
            _ => Array1::zeros(self.n),
        };

        // Exit-side chain-rule deltas.
        let delta_q_exit = &q.dq_t * &delta_t_exit + &q.dq_ls * &delta_ls_exit + &deltaw;
        let delta_q_t_exit = &q.d2q_tls * &delta_ls_exit;
        let delta_q_ls_exit = &q.d2q_tls * &delta_t_exit + &q.d2q_ls * &delta_ls_exit;
        let delta_q_tls_exit = &q.d3q_tls_ls * &delta_ls_exit;
        let delta_q_ls_ls_exit = &q.d3q_tls_ls * &delta_t_exit + &q.d3q_ls * &delta_ls_exit;

        // For the time block's D_u H we need the full combined delta_q that
        // includes both exit and entry contributions through h0 and h1.
        // For time-invariant blocks delta_q = delta_q_exit already.
        // For time-varying blocks, h0 couples through entry q0 and h1 through
        // exit q1, so the time block sees combined delta_q through both paths.
        let d_d1_q_exit = &q.d2_q1 * &delta_q_exit + &q.h_time_h1 * &delta_h1;
        let d_d2_q_exit = &q.d3_q1 * &delta_q_exit - &q.d_h_h1 * &delta_h1;

        let x_threshold_exit = self.x_threshold.to_dense();
        let x_threshold_entry = self.x_threshold_entry.as_ref().map(DesignMatrix::to_dense);
        let x_log_sigma_exit = self.x_log_sigma.to_dense();
        let x_log_sigma_entry = self.x_log_sigma_entry.as_ref().map(DesignMatrix::to_dense);
        let xw = self.x_link_wiggle.as_ref().map(DesignMatrix::to_dense);
        let mut joint = Array2::<f64>::zeros((p_total, p_total));

        // Entry-side predictor-space deltas and chain-rule deltas.
        // When time-varying blocks are present, we compute entry-side counterparts
        // of all exit-side delta quantities. When no block is time-varying, these
        // are zero placeholders that are never consumed.
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
                d_d2_q: &q.d3_q0 * &dq_en - &q.d_h_h0 * &delta_h0,
                delta_q: dq_en,
            }
        } else {
            // Time-invariant: entry and exit designs are identical, so
            // delta_q_entry = delta_q_exit.  The h0 branch of the time block
            // needs the entry-side delta_q to correctly compute D_d(u0).
            EntryDeltas {
                delta_q: delta_q_exit.clone(),
                delta_q_t: delta_q_t_exit.clone(),
                delta_q_ls: delta_q_ls_exit.clone(),
                delta_q_tls: delta_q_tls_exit.clone(),
                delta_q_ls_ls: delta_q_ls_ls_exit.clone(),
                d_d1_q: &q.d2_q0 * &delta_q_exit + &q.h_time_h0 * &delta_h0,
                d_d2_q: &q.d3_q0 * &delta_q_exit - &q.d_h_h0 * &delta_h0,
            }
        };

        // Time block D_u H.
        // The combined delta_q for the time block's cross term:
        // d_h_h0 couples through delta_q via entry path, d_h_h1 through exit.
        let dh_h0 = &q.d_h_h0 * &(&delta_h0 - &entry_deltas.delta_q);
        let dh_h1 = &q.d_h_h1 * &(&delta_h1 - &delta_q_exit);
        let dh_d = &q.d_h_d * &delta_d;
        let d_h_time = fast_xt_diag_x(&self.x_time_entry, &dh_h0)
            + fast_xt_diag_x(&self.x_time_exit, &dh_h1)
            + fast_xt_diag_x(&self.x_time_deriv, &dh_d);
        assign_symmetric_block(&mut joint, offsets[0], offsets[0], &d_h_time);

        // Threshold-threshold D_u H.
        if let Some(x_t_en) = x_threshold_entry.as_ref() {
            let dq_t_en = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
            let d_h_exit = -(&d_d2_q_exit * &q.dq_t.mapv(|v| v * v)
                + &(&q.d2_q1 * &(2.0 * &delta_q_t_exit * &q.dq_t)));
            let d_h_entry = -(&entry_deltas.d_d2_q * &dq_t_en.mapv(|v| v * v)
                + &(&q.d2_q0 * &(2.0 * &entry_deltas.delta_q_t * dq_t_en)));
            let d_h_tt = weighted_crossprod_dense(&x_threshold_exit, &d_h_exit, &x_threshold_exit)?
                + weighted_crossprod_dense(x_t_en, &d_h_entry, x_t_en)?;
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &d_h_tt);
        } else {
            let d_d2_q_ti = &q.d3_q * &delta_q_exit - &q.d_h_h0 * &delta_h0 - &q.d_h_h1 * &delta_h1;
            let d_h_t = -(&d_d2_q_ti * &q.dq_t.mapv(|v| v * v)
                + &(&q.d2_q * &(2.0 * &delta_q_t_exit * &q.dq_t)));
            let d_h_tt = weighted_crossprod_dense(&x_threshold_exit, &d_h_t, &x_threshold_exit)?;
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &d_h_tt);
        }

        // Threshold-log-sigma cross D_u H.
        {
            let has_t_entry = x_threshold_entry.is_some();
            let has_ls_entry = x_log_sigma_entry.is_some();
            if has_t_entry || has_ls_entry {
                let x_t_en = x_threshold_entry.as_ref().unwrap_or(&x_threshold_exit);
                let x_ls_en = x_log_sigma_entry.as_ref().unwrap_or(&x_log_sigma_exit);
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
                    &q.d3_q * &delta_q_exit - &q.d_h_h0 * &delta_h0 - &q.d_h_h1 * &delta_h1;
                let d_h_tlweights = -(&d_d2_q * &(&q.dq_t * &q.dq_ls)
                    + &(&q.d2_q * &(&delta_q_t_exit * &q.dq_ls + &q.dq_t * &delta_q_ls_exit))
                    + &(&d_d1_q * &q.d2q_tls)
                    + &(&q.d1_q * &delta_q_tls_exit));
                let d_h_tl =
                    weighted_crossprod_dense(&x_threshold_exit, &d_h_tlweights, &x_log_sigma_exit)?;
                assign_symmetric_block(&mut joint, offsets[1], offsets[2], &d_h_tl);
            }
        }

        // Log-sigma-log-sigma D_u H.
        if let Some(x_ls_en) = x_log_sigma_entry.as_ref() {
            let dq_ls_en = q.dq_ls_entry.as_ref().unwrap();
            let d2q_ls_en = q.d2q_ls_entry.as_ref().unwrap();
            let d_h_exit = -(&d_d2_q_exit * &q.dq_ls.mapv(|v| v * v)
                + &(&q.d2_q1 * &(2.0 * &delta_q_ls_exit * &q.dq_ls))
                + &(&d_d1_q_exit * &q.d2q_ls)
                + &(&q.d1_q1 * &delta_q_ls_ls_exit));
            let d_h_entry = -(&entry_deltas.d_d2_q * &dq_ls_en.mapv(|v| v * v)
                + &(&q.d2_q0 * &(2.0 * &entry_deltas.delta_q_ls * dq_ls_en))
                + &(&entry_deltas.d_d1_q * d2q_ls_en)
                + &(&q.d1_q0 * &entry_deltas.delta_q_ls_ls));
            let d_h_ll = weighted_crossprod_dense(&x_log_sigma_exit, &d_h_exit, &x_log_sigma_exit)?
                + weighted_crossprod_dense(x_ls_en, &d_h_entry, x_ls_en)?;
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &d_h_ll);
        } else {
            let d_d1_q =
                &q.d2_q * &delta_q_exit + &q.h_time_h0 * &delta_h0 + &q.h_time_h1 * &delta_h1;
            let d_d2_q = &q.d3_q * &delta_q_exit - &q.d_h_h0 * &delta_h0 - &q.d_h_h1 * &delta_h1;
            let d_h_l = -(&d_d2_q * &q.dq_ls.mapv(|v| v * v)
                + &(&q.d2_q * &(2.0 * &delta_q_ls_exit * &q.dq_ls))
                + &(&d_d1_q * &q.d2q_ls)
                + &(&q.d1_q * &delta_q_ls_ls_exit));
            let d_h_ll = weighted_crossprod_dense(&x_log_sigma_exit, &d_h_l, &x_log_sigma_exit)?;
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &d_h_ll);
        }

        // Time-threshold cross D_u H.
        if let (Some(x_t_en), Some(dq_t_en)) = (x_threshold_entry.as_ref(), q.dq_t_entry.as_ref()) {
            // h0 couples with entry: delta_q_t pertains to entry sigma derivatives.
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

        // Time-log-sigma cross D_u H.
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

        if let (Some(xw_dense), Some(w_offset)) = (xw.as_ref(), offsets.get(3).copied()) {
            // For wiggle cross-terms, use combined d_d2_q and d2_q.
            let d_d2_q_combined = if x_threshold_entry.is_some() || x_log_sigma_entry.is_some() {
                &d_d2_q_exit + &entry_deltas.d_d2_q
            } else {
                &q.d3_q * &delta_q_exit - &q.d_h_h0 * &delta_h0 - &q.d_h_h1 * &delta_h1
            };
            // Threshold-wiggle D_u H cross block.
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

            // Log-sigma-wiggle D_u H cross block.
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
        let q = self.collect_joint_quantities(block_states)?;
        let offsets = self.joint_block_offsets();
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;

        let x_threshold_exit = self.x_threshold.to_dense();
        let x_threshold_entry_owned = self.x_threshold_entry.as_ref().map(DesignMatrix::to_dense);
        let x_threshold_entry = x_threshold_entry_owned
            .as_ref()
            .unwrap_or(&x_threshold_exit);
        let x_log_sigma_exit = self.x_log_sigma.to_dense();
        let x_log_sigma_entry_owned = self.x_log_sigma_entry.as_ref().map(DesignMatrix::to_dense);
        let x_log_sigma_entry = x_log_sigma_entry_owned
            .as_ref()
            .unwrap_or(&x_log_sigma_exit);
        let xw = self.x_link_wiggle.as_ref().map(DesignMatrix::to_dense);

        let dq_t_entry = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
        let dq_ls_entry = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
        let d2q_tls_entry = q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls);
        let d2q_ls_entry = q.d2q_ls_entry.as_ref().unwrap_or(&q.d2q_ls);
        let d3q_tls_ls_entry = q.d3q_tls_ls_entry.as_ref().unwrap_or(&q.d3q_tls_ls);
        let d3q_ls_entry = q.d3q_ls_entry.as_ref().unwrap_or(&q.d3q_ls);

        let q0_psi = &(dq_t_entry * &dir.z_t_entry_psi) + &(dq_ls_entry * &dir.z_ls_entry_psi);
        let q1_psi = &(&q.dq_t * &dir.z_t_exit_psi) + &(&q.dq_ls * &dir.z_ls_exit_psi);
        let dq_t_entry_psi = d2q_tls_entry * &dir.z_ls_entry_psi;
        let dq_t_exit_psi = &q.d2q_tls * &dir.z_ls_exit_psi;
        let dq_ls_entry_psi =
            d2q_tls_entry * &dir.z_t_entry_psi + d2q_ls_entry * &dir.z_ls_entry_psi;
        let dq_ls_exit_psi = &q.d2q_tls * &dir.z_t_exit_psi + &q.d2q_ls * &dir.z_ls_exit_psi;
        let d2q_tls_entry_psi = d3q_tls_ls_entry * &dir.z_ls_entry_psi;
        let d2q_tls_exit_psi = &q.d3q_tls_ls * &dir.z_ls_exit_psi;
        let d2q_ls_entry_psi =
            d3q_tls_ls_entry * &dir.z_t_entry_psi + d3q_ls_entry * &dir.z_ls_entry_psi;
        let d2q_ls_exit_psi = &q.d3q_tls_ls * &dir.z_t_exit_psi + &q.d3q_ls * &dir.z_ls_exit_psi;

        let objective_psi = q.d1_q0.dot(&q0_psi) + q.d1_q1.dot(&q1_psi);

        let mut score_psi = Array1::<f64>::zeros(p_total);
        let time_score = self.x_time_entry.t().dot(&(-&q.d2_q0 * &q0_psi))
            + self.x_time_exit.t().dot(&(-&q.d2_q1 * &q1_psi));
        score_psi
            .slice_mut(s![offsets[0]..offsets[1]])
            .assign(&time_score);

        let threshold_score_row_exit = &q.d1_q1 * &q.dq_t;
        let threshold_score_row_entry = &q.d1_q0 * dq_t_entry;
        let d_threshold_score_row_exit = &q.d2_q1 * &q1_psi * &q.dq_t + &q.d1_q1 * &dq_t_exit_psi;
        let d_threshold_score_row_entry =
            &q.d2_q0 * &q0_psi * dq_t_entry + &q.d1_q0 * &dq_t_entry_psi;
        let threshold_score = dir.x_t_exit_psi.t().dot(&threshold_score_row_exit)
            + x_threshold_exit.t().dot(&d_threshold_score_row_exit)
            + dir.x_t_entry_psi.t().dot(&threshold_score_row_entry)
            + x_threshold_entry.t().dot(&d_threshold_score_row_entry);
        score_psi
            .slice_mut(s![offsets[1]..offsets[2]])
            .assign(&threshold_score);

        let log_sigma_score_row_exit = &q.d1_q1 * &q.dq_ls;
        let log_sigma_score_row_entry = &q.d1_q0 * dq_ls_entry;
        let d_log_sigma_score_row_exit = &q.d2_q1 * &q1_psi * &q.dq_ls + &q.d1_q1 * &dq_ls_exit_psi;
        let d_log_sigma_score_row_entry =
            &q.d2_q0 * &q0_psi * dq_ls_entry + &q.d1_q0 * &dq_ls_entry_psi;
        let log_sigma_score = dir.x_ls_exit_psi.t().dot(&log_sigma_score_row_exit)
            + x_log_sigma_exit.t().dot(&d_log_sigma_score_row_exit)
            + dir.x_ls_entry_psi.t().dot(&log_sigma_score_row_entry)
            + x_log_sigma_entry.t().dot(&d_log_sigma_score_row_entry);
        score_psi
            .slice_mut(s![offsets[2]..offsets[3]])
            .assign(&log_sigma_score);

        if let (Some(xw_dense), Some(w_offset)) = (xw.as_ref(), offsets.get(3).copied()) {
            let wiggle_score = xw_dense.t().dot(&(&q.d2_q0 * &q0_psi + &q.d2_q1 * &q1_psi));
            score_psi
                .slice_mut(s![w_offset..offsets[4]])
                .assign(&wiggle_score);
        }

        let mut hessian_psi = Array2::<f64>::zeros((p_total, p_total));
        let h_time_time = fast_xt_diag_x(&self.x_time_entry, &(-&q.d3_q0 * &q0_psi))
            + fast_xt_diag_x(&self.x_time_exit, &(-&q.d3_q1 * &q1_psi));
        assign_symmetric_block(&mut hessian_psi, offsets[0], offsets[0], &h_time_time);

        let h_tt_entry = -(&q.d2_q0 * &dq_t_entry.mapv(|v| v * v));
        let h_tt_exit = -(&q.d2_q1 * &q.dq_t.mapv(|v| v * v));
        let dh_tt_entry = -(&q.d3_q0 * &q0_psi * &dq_t_entry.mapv(|v| v * v)
            + &(2.0 * &q.d2_q0 * dq_t_entry * &dq_t_entry_psi));
        let dh_tt_exit = -(&q.d3_q1 * &q1_psi * &q.dq_t.mapv(|v| v * v)
            + &(2.0 * &q.d2_q1 * &q.dq_t * &dq_t_exit_psi));
        let h_threshold_threshold =
            weighted_crossprod_dense(&dir.x_t_exit_psi, &h_tt_exit, &x_threshold_exit)?
                + weighted_crossprod_dense(&x_threshold_exit, &h_tt_exit, &dir.x_t_exit_psi)?
                + weighted_crossprod_dense(&x_threshold_exit, &dh_tt_exit, &x_threshold_exit)?
                + weighted_crossprod_dense(&dir.x_t_entry_psi, &h_tt_entry, x_threshold_entry)?
                + weighted_crossprod_dense(x_threshold_entry, &h_tt_entry, &dir.x_t_entry_psi)?
                + weighted_crossprod_dense(x_threshold_entry, &dh_tt_entry, x_threshold_entry)?;
        assign_symmetric_block(
            &mut hessian_psi,
            offsets[1],
            offsets[1],
            &h_threshold_threshold,
        );

        let h_ll_entry = -(&q.d2_q0 * &dq_ls_entry.mapv(|v| v * v) + &(&q.d1_q0 * d2q_ls_entry));
        let h_ll_exit = -(&q.d2_q1 * &q.dq_ls.mapv(|v| v * v) + &(&q.d1_q1 * &q.d2q_ls));
        let dh_ll_entry = -(&q.d3_q0 * &q0_psi * &dq_ls_entry.mapv(|v| v * v)
            + &(2.0 * &q.d2_q0 * dq_ls_entry * &dq_ls_entry_psi)
            + &(&q.d2_q0 * &q0_psi * d2q_ls_entry)
            + &(&q.d1_q0 * &d2q_ls_entry_psi));
        let dh_ll_exit = -(&q.d3_q1 * &q1_psi * &q.dq_ls.mapv(|v| v * v)
            + &(2.0 * &q.d2_q1 * &q.dq_ls * &dq_ls_exit_psi)
            + &(&q.d2_q1 * &q1_psi * &q.d2q_ls)
            + &(&q.d1_q1 * &d2q_ls_exit_psi));
        let h_log_sigma_log_sigma =
            weighted_crossprod_dense(&dir.x_ls_exit_psi, &h_ll_exit, &x_log_sigma_exit)?
                + weighted_crossprod_dense(&x_log_sigma_exit, &h_ll_exit, &dir.x_ls_exit_psi)?
                + weighted_crossprod_dense(&x_log_sigma_exit, &dh_ll_exit, &x_log_sigma_exit)?
                + weighted_crossprod_dense(&dir.x_ls_entry_psi, &h_ll_entry, x_log_sigma_entry)?
                + weighted_crossprod_dense(x_log_sigma_entry, &h_ll_entry, &dir.x_ls_entry_psi)?
                + weighted_crossprod_dense(x_log_sigma_entry, &dh_ll_entry, x_log_sigma_entry)?;
        assign_symmetric_block(
            &mut hessian_psi,
            offsets[2],
            offsets[2],
            &h_log_sigma_log_sigma,
        );

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
        let h_threshold_log_sigma =
            weighted_crossprod_dense(&dir.x_t_exit_psi, &h_tl_exit, &x_log_sigma_exit)?
                + weighted_crossprod_dense(&x_threshold_exit, &h_tl_exit, &dir.x_ls_exit_psi)?
                + weighted_crossprod_dense(&x_threshold_exit, &dh_tl_exit, &x_log_sigma_exit)?
                + weighted_crossprod_dense(&dir.x_t_entry_psi, &h_tl_entry, x_log_sigma_entry)?
                + weighted_crossprod_dense(x_threshold_entry, &h_tl_entry, &dir.x_ls_entry_psi)?
                + weighted_crossprod_dense(x_threshold_entry, &dh_tl_entry, x_log_sigma_entry)?;
        assign_symmetric_block(
            &mut hessian_psi,
            offsets[1],
            offsets[2],
            &h_threshold_log_sigma,
        );

        let h_h0_t = &q.d2_q0 * dq_t_entry;
        let h_h1_t = &q.d2_q1 * &q.dq_t;
        let dh_h0_t = &q.d3_q0 * &q0_psi * dq_t_entry + &q.d2_q0 * &dq_t_entry_psi;
        let dh_h1_t = &q.d3_q1 * &q1_psi * &q.dq_t + &q.d2_q1 * &dq_t_exit_psi;
        let h_time_threshold =
            weighted_crossprod_dense(&self.x_time_entry, &dh_h0_t, x_threshold_entry)?
                + weighted_crossprod_dense(&self.x_time_entry, &h_h0_t, &dir.x_t_entry_psi)?
                + weighted_crossprod_dense(&self.x_time_exit, &dh_h1_t, &x_threshold_exit)?
                + weighted_crossprod_dense(&self.x_time_exit, &h_h1_t, &dir.x_t_exit_psi)?;
        assign_symmetric_block(&mut hessian_psi, offsets[0], offsets[1], &h_time_threshold);

        let h_h0_ls = &q.d2_q0 * dq_ls_entry;
        let h_h1_ls = &q.d2_q1 * &q.dq_ls;
        let dh_h0_ls = &q.d3_q0 * &q0_psi * dq_ls_entry + &q.d2_q0 * &dq_ls_entry_psi;
        let dh_h1_ls = &q.d3_q1 * &q1_psi * &q.dq_ls + &q.d2_q1 * &dq_ls_exit_psi;
        let h_time_log_sigma =
            weighted_crossprod_dense(&self.x_time_entry, &dh_h0_ls, x_log_sigma_entry)?
                + weighted_crossprod_dense(&self.x_time_entry, &h_h0_ls, &dir.x_ls_entry_psi)?
                + weighted_crossprod_dense(&self.x_time_exit, &dh_h1_ls, &x_log_sigma_exit)?
                + weighted_crossprod_dense(&self.x_time_exit, &h_h1_ls, &dir.x_ls_exit_psi)?;
        assign_symmetric_block(&mut hessian_psi, offsets[0], offsets[2], &h_time_log_sigma);

        if let (Some(xw_dense), Some(w_offset)) = (xw.as_ref(), offsets.get(3).copied()) {
            let h_ww = -(&q.d3_q0 * &q0_psi + &q.d3_q1 * &q1_psi);
            let h_wiggle_wiggle = weighted_crossprod_dense(xw_dense, &h_ww, xw_dense)?;
            assign_symmetric_block(&mut hessian_psi, w_offset, w_offset, &h_wiggle_wiggle);

            let h_tw_entry = -(&q.d2_q0 * dq_t_entry);
            let h_tw_exit = -(&q.d2_q1 * &q.dq_t);
            let dh_tw_entry = -(&q.d3_q0 * &q0_psi * dq_t_entry + &q.d2_q0 * &dq_t_entry_psi);
            let dh_tw_exit = -(&q.d3_q1 * &q1_psi * &q.dq_t + &q.d2_q1 * &dq_t_exit_psi);
            let h_threshold_wiggle =
                weighted_crossprod_dense(&dir.x_t_exit_psi, &h_tw_exit, xw_dense)?
                    + weighted_crossprod_dense(&x_threshold_exit, &dh_tw_exit, xw_dense)?
                    + weighted_crossprod_dense(&dir.x_t_entry_psi, &h_tw_entry, xw_dense)?
                    + weighted_crossprod_dense(x_threshold_entry, &dh_tw_entry, xw_dense)?;
            assign_symmetric_block(&mut hessian_psi, offsets[1], w_offset, &h_threshold_wiggle);

            let h_lw_entry = -(&q.d2_q0 * dq_ls_entry);
            let h_lw_exit = -(&q.d2_q1 * &q.dq_ls);
            let dh_lw_entry = -(&q.d3_q0 * &q0_psi * dq_ls_entry + &q.d2_q0 * &dq_ls_entry_psi);
            let dh_lw_exit = -(&q.d3_q1 * &q1_psi * &q.dq_ls + &q.d2_q1 * &dq_ls_exit_psi);
            let h_log_sigma_wiggle =
                weighted_crossprod_dense(&dir.x_ls_exit_psi, &h_lw_exit, xw_dense)?
                    + weighted_crossprod_dense(&x_log_sigma_exit, &dh_lw_exit, xw_dense)?
                    + weighted_crossprod_dense(&dir.x_ls_entry_psi, &h_lw_entry, xw_dense)?
                    + weighted_crossprod_dense(x_log_sigma_entry, &dh_lw_entry, xw_dense)?;
            assign_symmetric_block(&mut hessian_psi, offsets[2], w_offset, &h_log_sigma_wiggle);

            let h_time_wiggle =
                weighted_crossprod_dense(&self.x_time_entry, &(&q.d3_q0 * &q0_psi), xw_dense)?
                    + weighted_crossprod_dense(&self.x_time_exit, &(&q.d3_q1 * &q1_psi), xw_dense)?;
            assign_symmetric_block(&mut hessian_psi, offsets[0], w_offset, &h_time_wiggle);
        }

        let _ = dir.block_idx;
        Ok(Some(ExactNewtonJointPsiTerms {
            objective_psi,
            score_psi,
            hessian_psi,
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
        let (
            x_t_exit_ab,
            x_t_entry_ab,
            x_ls_exit_ab,
            x_ls_entry_ab,
            z_t_exit_ab,
            z_t_entry_ab,
            z_ls_exit_ab,
            z_ls_entry_ab,
        ) = self.exact_newton_joint_psisecond_design_drifts(
            block_states,
            derivative_blocks,
            &dir_i,
            &dir_j,
        )?;

        let q = self.collect_joint_quantities(block_states)?;
        let offsets = self.joint_block_offsets();
        let p_total = *offsets
            .last()
            .ok_or_else(|| "missing joint block offsets".to_string())?;

        let x_threshold_exit = self.x_threshold.to_dense();
        let x_threshold_entry_owned = self.x_threshold_entry.as_ref().map(DesignMatrix::to_dense);
        let x_threshold_entry = x_threshold_entry_owned
            .as_ref()
            .unwrap_or(&x_threshold_exit);
        let x_log_sigma_exit = self.x_log_sigma.to_dense();
        let x_log_sigma_entry_owned = self.x_log_sigma_entry.as_ref().map(DesignMatrix::to_dense);
        let x_log_sigma_entry = x_log_sigma_entry_owned
            .as_ref()
            .unwrap_or(&x_log_sigma_exit);
        let xw = self.x_link_wiggle.as_ref().map(DesignMatrix::to_dense);

        let dq_t_entry = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
        let dq_ls_entry = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
        let d2q_tls_entry = q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls);
        let d2q_ls_entry = q.d2q_ls_entry.as_ref().unwrap_or(&q.d2q_ls);
        let d3q_tls_ls_entry = q.d3q_tls_ls_entry.as_ref().unwrap_or(&q.d3q_tls_ls);
        let d3q_ls_entry = q.d3q_ls_entry.as_ref().unwrap_or(&q.d3q_ls);
        // 4th-order chain-rule derivatives for the second-order Hessian drift.
        // These enter the (ϑ,s,s,s) and (s,s,s,s) blocks of the bilinear
        // d²q_chain/dψ² computation via the Faà di Bruno formula.
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

        let q0_ab = &(dq_t_entry * &z_t_entry_ab)
            + &(dq_ls_entry * &z_ls_entry_ab)
            + &(d2q_tls_entry * &entry_cross)
            + &(d2q_ls_entry * &(&dir_i.z_ls_entry_psi * &dir_j.z_ls_entry_psi));
        let q1_ab = &(&q.dq_t * &z_t_exit_ab)
            + &(&q.dq_ls * &z_ls_exit_ab)
            + &(&q.d2q_tls * &exit_cross)
            + &(&q.d2q_ls * &(&dir_i.z_ls_exit_psi * &dir_j.z_ls_exit_psi));

        let dq_t_entry_ab = &(d2q_tls_entry * &z_ls_entry_ab)
            + &(d3q_tls_ls_entry * &(&dir_i.z_ls_entry_psi * &dir_j.z_ls_entry_psi));
        let dq_t_exit_ab = &(&q.d2q_tls * &z_ls_exit_ab)
            + &(&q.d3q_tls_ls * &(&dir_i.z_ls_exit_psi * &dir_j.z_ls_exit_psi));

        let dq_ls_entry_ab = &(d2q_tls_entry * &z_t_entry_ab)
            + &(d2q_ls_entry * &z_ls_entry_ab)
            + &(d3q_tls_ls_entry * &entry_cross)
            + &(d3q_ls_entry * &(&dir_i.z_ls_entry_psi * &dir_j.z_ls_entry_psi));
        let dq_ls_exit_ab = &(&q.d2q_tls * &z_t_exit_ab)
            + &(&q.d2q_ls * &z_ls_exit_ab)
            + &(&q.d3q_tls_ls * &exit_cross)
            + &(&q.d3q_ls * &(&dir_i.z_ls_exit_psi * &dir_j.z_ls_exit_psi));

        // Bilinear derivatives of 2nd-order chain-rule quantities w.r.t. ψ.
        //
        // These now include the 4th-order chain-rule terms from response.md
        // Section 6. The general formula for d²(∂²q/∂α∂β)/dψ_i dψ_j is:
        //
        //   Σ_k (∂³q/∂α∂β∂η_k) · z_k_ab
        //     + Σ_{k,l} (∂⁴q/∂α∂β∂η_k∂η_l) · z_k_i · z_l_j
        //
        // The second sum is the additional 4th-order contribution. It is nonzero
        // only when at least one of k,l is an η_s index, because:
        //   u_ϑsss = ∂⁴q/∂η_ϑ∂η_s³ ≠ 0
        //   u_ssss = ∂⁴q/∂η_s⁴ ≠ 0
        //
        // For the (ϑ,s) cross block: ∂²(d2q_tls)/dψ² needs d4q_tls_ls_ls.
        // For the (s,s) block: ∂²(d2q_ls)/dψ² needs d4q_tls_ls_ls and d4q_ls.
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
        let threshold_score = x_t_exit_ab.t().dot(&threshold_score_row_exit)
            + dir_i.x_t_exit_psi.t().dot(&d_threshold_score_row_exit_j)
            + dir_j.x_t_exit_psi.t().dot(&d_threshold_score_row_exit_i)
            + x_threshold_exit.t().dot(&d2_threshold_score_row_exit)
            + x_t_entry_ab.t().dot(&threshold_score_row_entry)
            + dir_i.x_t_entry_psi.t().dot(&d_threshold_score_row_entry_j)
            + dir_j.x_t_entry_psi.t().dot(&d_threshold_score_row_entry_i)
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
        let log_sigma_score = x_ls_exit_ab.t().dot(&log_sigma_score_row_exit)
            + dir_i.x_ls_exit_psi.t().dot(&d_log_sigma_score_row_exit_j)
            + dir_j.x_ls_exit_psi.t().dot(&d_log_sigma_score_row_exit_i)
            + x_log_sigma_exit.t().dot(&d2_log_sigma_score_row_exit)
            + x_ls_entry_ab.t().dot(&log_sigma_score_row_entry)
            + dir_i.x_ls_entry_psi.t().dot(&d_log_sigma_score_row_entry_j)
            + dir_j.x_ls_entry_psi.t().dot(&d_log_sigma_score_row_entry_i)
            + x_log_sigma_entry.t().dot(&d2_log_sigma_score_row_entry);
        score_psi_psi
            .slice_mut(s![offsets[2]..offsets[3]])
            .assign(&log_sigma_score);

        if let (Some(xw_dense), Some(w_offset)) = (xw.as_ref(), offsets.get(3).copied()) {
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

        // Second-order family Hessian contribution.
        //
        // SurvivalJointQuantities now carries d4_q0/d4_q1 (4th derivatives of ℓ
        // w.r.t. q), d4q_tls_ls_ls and d4q_ls (4th-order chain-rule derivatives
        // of q w.r.t. block predictors), and d2_h_h0/d2_h_h1 (4th derivatives
        // of ℓ w.r.t. time predictors). These are used in the second-order
        // Hessian drift (D²H[u,v]) computation. See response.md Section 6.
        //
        // The bilinear psi-psi Hessian below uses 3rd-order row derivatives
        // (d3_q, d_h_h) for the leading product-rule terms. The 4th-order
        // corrections (d4_q, d2_h_h) enter the D²H bilinear computation in
        // exact_newton_joint_psi_second_order_terms, not here.
        let mut hessian_psi_psi = Array2::<f64>::zeros((p_total, p_total));
        let h_time_time = fast_xt_diag_x(
            &self.x_time_entry,
            &(-(&q.d3_q0 * &q0_ab) - &(&q.d_h_h0 * &(&q0_i * &q0_j))),
        ) + fast_xt_diag_x(
            &self.x_time_exit,
            &(-(&q.d3_q1 * &q1_ab) - &(&q.d_h_h1 * &(&q1_i * &q1_j))),
        );
        assign_symmetric_block(&mut hessian_psi_psi, offsets[0], offsets[0], &h_time_time);

        let h_tt_entry = -(&q.d2_q0 * &dq_t_entry.mapv(|v| v * v));
        let h_tt_exit = -(&q.d2_q1 * &q.dq_t.mapv(|v| v * v));
        let dh_tt_entry_i = -(&q.d3_q0 * &q0_i * &dq_t_entry.mapv(|v| v * v)
            + &(2.0 * &q.d2_q0 * dq_t_entry * &dq_t_entry_i));
        let dh_tt_exit_i = -(&q.d3_q1 * &q1_i * &q.dq_t.mapv(|v| v * v)
            + &(2.0 * &q.d2_q1 * &q.dq_t * &dq_t_exit_i));
        let dh_tt_entry_j = -(&q.d3_q0 * &q0_j * &dq_t_entry.mapv(|v| v * v)
            + &(2.0 * &q.d2_q0 * dq_t_entry * &dq_t_entry_j));
        let dh_tt_exit_j = -(&q.d3_q1 * &q1_j * &q.dq_t.mapv(|v| v * v)
            + &(2.0 * &q.d2_q1 * &q.dq_t * &dq_t_exit_j));
        let h_threshold_threshold =
            weighted_crossprod_dense(&x_t_exit_ab, &h_tt_exit, &x_threshold_exit)?
                + weighted_crossprod_dense(&x_threshold_exit, &h_tt_exit, &x_t_exit_ab)?
                + weighted_crossprod_dense(&dir_i.x_t_exit_psi, &h_tt_exit, &dir_j.x_t_exit_psi)?
                + weighted_crossprod_dense(&dir_j.x_t_exit_psi, &h_tt_exit, &dir_i.x_t_exit_psi)?
                + weighted_crossprod_dense(&dir_i.x_t_exit_psi, &dh_tt_exit_j, &x_threshold_exit)?
                + weighted_crossprod_dense(&x_threshold_exit, &dh_tt_exit_j, &dir_i.x_t_exit_psi)?
                + weighted_crossprod_dense(&dir_j.x_t_exit_psi, &dh_tt_exit_i, &x_threshold_exit)?
                + weighted_crossprod_dense(&x_threshold_exit, &dh_tt_exit_i, &dir_j.x_t_exit_psi)?
                + weighted_crossprod_dense(&x_t_entry_ab, &h_tt_entry, x_threshold_entry)?
                + weighted_crossprod_dense(x_threshold_entry, &h_tt_entry, &x_t_entry_ab)?
                + weighted_crossprod_dense(
                    &dir_i.x_t_entry_psi,
                    &h_tt_entry,
                    &dir_j.x_t_entry_psi,
                )?
                + weighted_crossprod_dense(
                    &dir_j.x_t_entry_psi,
                    &h_tt_entry,
                    &dir_i.x_t_entry_psi,
                )?
                + weighted_crossprod_dense(
                    &dir_i.x_t_entry_psi,
                    &dh_tt_entry_j,
                    x_threshold_entry,
                )?
                + weighted_crossprod_dense(
                    x_threshold_entry,
                    &dh_tt_entry_j,
                    &dir_i.x_t_entry_psi,
                )?
                + weighted_crossprod_dense(
                    &dir_j.x_t_entry_psi,
                    &dh_tt_entry_i,
                    x_threshold_entry,
                )?
                + weighted_crossprod_dense(
                    x_threshold_entry,
                    &dh_tt_entry_i,
                    &dir_j.x_t_entry_psi,
                )?;
        assign_symmetric_block(
            &mut hessian_psi_psi,
            offsets[1],
            offsets[1],
            &h_threshold_threshold,
        );

        let h_ll_entry = -(&q.d2_q0 * &dq_ls_entry.mapv(|v| v * v) + &(&q.d1_q0 * d2q_ls_entry));
        let h_ll_exit = -(&q.d2_q1 * &q.dq_ls.mapv(|v| v * v) + &(&q.d1_q1 * &q.d2q_ls));
        let dh_ll_entry_i = -(&q.d3_q0 * &q0_i * &dq_ls_entry.mapv(|v| v * v)
            + &(2.0 * &q.d2_q0 * dq_ls_entry * &dq_ls_entry_i)
            + &(&q.d2_q0 * &q0_i * d2q_ls_entry)
            + &(&q.d1_q0 * &d2q_ls_entry_i));
        let dh_ll_exit_i = -(&q.d3_q1 * &q1_i * &q.dq_ls.mapv(|v| v * v)
            + &(2.0 * &q.d2_q1 * &q.dq_ls * &dq_ls_exit_i)
            + &(&q.d2_q1 * &q1_i * &q.d2q_ls)
            + &(&q.d1_q1 * &d2q_ls_exit_i));
        let dh_ll_entry_j = -(&q.d3_q0 * &q0_j * &dq_ls_entry.mapv(|v| v * v)
            + &(2.0 * &q.d2_q0 * dq_ls_entry * &dq_ls_entry_j)
            + &(&q.d2_q0 * &q0_j * d2q_ls_entry)
            + &(&q.d1_q0 * &d2q_ls_entry_j));
        let dh_ll_exit_j = -(&q.d3_q1 * &q1_j * &q.dq_ls.mapv(|v| v * v)
            + &(2.0 * &q.d2_q1 * &q.dq_ls * &dq_ls_exit_j)
            + &(&q.d2_q1 * &q1_j * &q.d2q_ls)
            + &(&q.d1_q1 * &d2q_ls_exit_j));
        let h_log_sigma_log_sigma =
            weighted_crossprod_dense(&x_ls_exit_ab, &h_ll_exit, &x_log_sigma_exit)?
                + weighted_crossprod_dense(&x_log_sigma_exit, &h_ll_exit, &x_ls_exit_ab)?
                + weighted_crossprod_dense(&dir_i.x_ls_exit_psi, &h_ll_exit, &dir_j.x_ls_exit_psi)?
                + weighted_crossprod_dense(&dir_j.x_ls_exit_psi, &h_ll_exit, &dir_i.x_ls_exit_psi)?
                + weighted_crossprod_dense(&dir_i.x_ls_exit_psi, &dh_ll_exit_j, &x_log_sigma_exit)?
                + weighted_crossprod_dense(&x_log_sigma_exit, &dh_ll_exit_j, &dir_i.x_ls_exit_psi)?
                + weighted_crossprod_dense(&dir_j.x_ls_exit_psi, &dh_ll_exit_i, &x_log_sigma_exit)?
                + weighted_crossprod_dense(&x_log_sigma_exit, &dh_ll_exit_i, &dir_j.x_ls_exit_psi)?
                + weighted_crossprod_dense(&x_ls_entry_ab, &h_ll_entry, x_log_sigma_entry)?
                + weighted_crossprod_dense(x_log_sigma_entry, &h_ll_entry, &x_ls_entry_ab)?
                + weighted_crossprod_dense(
                    &dir_i.x_ls_entry_psi,
                    &h_ll_entry,
                    &dir_j.x_ls_entry_psi,
                )?
                + weighted_crossprod_dense(
                    &dir_j.x_ls_entry_psi,
                    &h_ll_entry,
                    &dir_i.x_ls_entry_psi,
                )?
                + weighted_crossprod_dense(
                    &dir_i.x_ls_entry_psi,
                    &dh_ll_entry_j,
                    x_log_sigma_entry,
                )?
                + weighted_crossprod_dense(
                    x_log_sigma_entry,
                    &dh_ll_entry_j,
                    &dir_i.x_ls_entry_psi,
                )?
                + weighted_crossprod_dense(
                    &dir_j.x_ls_entry_psi,
                    &dh_ll_entry_i,
                    x_log_sigma_entry,
                )?
                + weighted_crossprod_dense(
                    x_log_sigma_entry,
                    &dh_ll_entry_i,
                    &dir_j.x_ls_entry_psi,
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
        let h_threshold_log_sigma =
            weighted_crossprod_dense(&x_t_exit_ab, &h_tl_exit, &x_log_sigma_exit)?
                + weighted_crossprod_dense(&x_threshold_exit, &h_tl_exit, &x_ls_exit_ab)?
                + weighted_crossprod_dense(&dir_i.x_t_exit_psi, &h_tl_exit, &dir_j.x_ls_exit_psi)?
                + weighted_crossprod_dense(&dir_j.x_t_exit_psi, &h_tl_exit, &dir_i.x_ls_exit_psi)?
                + weighted_crossprod_dense(&dir_i.x_t_exit_psi, &dh_tl_exit_j, &x_log_sigma_exit)?
                + weighted_crossprod_dense(&x_threshold_exit, &dh_tl_exit_j, &dir_i.x_ls_exit_psi)?
                + weighted_crossprod_dense(&dir_j.x_t_exit_psi, &dh_tl_exit_i, &x_log_sigma_exit)?
                + weighted_crossprod_dense(&x_threshold_exit, &dh_tl_exit_i, &dir_j.x_ls_exit_psi)?
                + weighted_crossprod_dense(&x_t_entry_ab, &h_tl_entry, x_log_sigma_entry)?
                + weighted_crossprod_dense(x_threshold_entry, &h_tl_entry, &x_ls_entry_ab)?
                + weighted_crossprod_dense(
                    &dir_i.x_t_entry_psi,
                    &h_tl_entry,
                    &dir_j.x_ls_entry_psi,
                )?
                + weighted_crossprod_dense(
                    &dir_j.x_t_entry_psi,
                    &h_tl_entry,
                    &dir_i.x_ls_entry_psi,
                )?
                + weighted_crossprod_dense(
                    &dir_i.x_t_entry_psi,
                    &dh_tl_entry_j,
                    x_log_sigma_entry,
                )?
                + weighted_crossprod_dense(
                    x_threshold_entry,
                    &dh_tl_entry_j,
                    &dir_i.x_ls_entry_psi,
                )?
                + weighted_crossprod_dense(
                    &dir_j.x_t_entry_psi,
                    &dh_tl_entry_i,
                    x_log_sigma_entry,
                )?
                + weighted_crossprod_dense(
                    x_threshold_entry,
                    &dh_tl_entry_i,
                    &dir_j.x_ls_entry_psi,
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
        let h_time_threshold =
            weighted_crossprod_dense(&self.x_time_entry, &dh_h0_t_j, &dir_i.x_t_entry_psi)?
                + weighted_crossprod_dense(&self.x_time_entry, &dh_h0_t_i, &dir_j.x_t_entry_psi)?
                + weighted_crossprod_dense(&self.x_time_entry, &h_h0_t, &x_t_entry_ab)?
                + weighted_crossprod_dense(&self.x_time_exit, &dh_h1_t_j, &dir_i.x_t_exit_psi)?
                + weighted_crossprod_dense(&self.x_time_exit, &dh_h1_t_i, &dir_j.x_t_exit_psi)?
                + weighted_crossprod_dense(&self.x_time_exit, &h_h1_t, &x_t_exit_ab)?;
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
        let h_time_log_sigma =
            weighted_crossprod_dense(&self.x_time_entry, &dh_h0_ls_j, &dir_i.x_ls_entry_psi)?
                + weighted_crossprod_dense(&self.x_time_entry, &dh_h0_ls_i, &dir_j.x_ls_entry_psi)?
                + weighted_crossprod_dense(&self.x_time_entry, &h_h0_ls, &x_ls_entry_ab)?
                + weighted_crossprod_dense(&self.x_time_exit, &dh_h1_ls_j, &dir_i.x_ls_exit_psi)?
                + weighted_crossprod_dense(&self.x_time_exit, &dh_h1_ls_i, &dir_j.x_ls_exit_psi)?
                + weighted_crossprod_dense(&self.x_time_exit, &h_h1_ls, &x_ls_exit_ab)?;
        assign_symmetric_block(
            &mut hessian_psi_psi,
            offsets[0],
            offsets[2],
            &h_time_log_sigma,
        );

        if let (Some(xw_dense), Some(w_offset)) = (xw.as_ref(), offsets.get(3).copied()) {
            let h_ww = -(&q.d3_q0 * &q0_ab + &q.d3_q1 * &q1_ab);
            let h_wiggle_wiggle = weighted_crossprod_dense(xw_dense, &h_ww, xw_dense)?;
            assign_symmetric_block(&mut hessian_psi_psi, w_offset, w_offset, &h_wiggle_wiggle);

            let h_tw_entry = -(&q.d2_q0 * dq_t_entry);
            let h_tw_exit = -(&q.d2_q1 * &q.dq_t);
            let dh_tw_entry_i = -(&q.d3_q0 * &q0_i * dq_t_entry + &q.d2_q0 * &dq_t_entry_i);
            let dh_tw_exit_i = -(&q.d3_q1 * &q1_i * &q.dq_t + &q.d2_q1 * &dq_t_exit_i);
            let dh_tw_entry_j = -(&q.d3_q0 * &q0_j * dq_t_entry + &q.d2_q0 * &dq_t_entry_j);
            let dh_tw_exit_j = -(&q.d3_q1 * &q1_j * &q.dq_t + &q.d2_q1 * &dq_t_exit_j);
            let h_threshold_wiggle =
                weighted_crossprod_dense(&dir_i.x_t_exit_psi, &dh_tw_exit_j, xw_dense)?
                    + weighted_crossprod_dense(&dir_j.x_t_exit_psi, &dh_tw_exit_i, xw_dense)?
                    + weighted_crossprod_dense(&x_t_exit_ab, &h_tw_exit, xw_dense)?
                    + weighted_crossprod_dense(&dir_i.x_t_entry_psi, &dh_tw_entry_j, xw_dense)?
                    + weighted_crossprod_dense(&dir_j.x_t_entry_psi, &dh_tw_entry_i, xw_dense)?
                    + weighted_crossprod_dense(&x_t_entry_ab, &h_tw_entry, xw_dense)?;
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
            let h_log_sigma_wiggle =
                weighted_crossprod_dense(&dir_i.x_ls_exit_psi, &dh_lw_exit_j, xw_dense)?
                    + weighted_crossprod_dense(&dir_j.x_ls_exit_psi, &dh_lw_exit_i, xw_dense)?
                    + weighted_crossprod_dense(&x_ls_exit_ab, &h_lw_exit, xw_dense)?
                    + weighted_crossprod_dense(&dir_i.x_ls_entry_psi, &dh_lw_entry_j, xw_dense)?
                    + weighted_crossprod_dense(&dir_j.x_ls_entry_psi, &dh_lw_entry_i, xw_dense)?
                    + weighted_crossprod_dense(&x_ls_entry_ab, &h_lw_entry, xw_dense)?;
            assign_symmetric_block(
                &mut hessian_psi_psi,
                offsets[2],
                w_offset,
                &h_log_sigma_wiggle,
            );

            let h_time_wiggle =
                weighted_crossprod_dense(&self.x_time_entry, &(&q.d3_q0 * &q0_ab), xw_dense)?
                    + weighted_crossprod_dense(&self.x_time_exit, &(&q.d3_q1 * &q1_ab), xw_dense)?;
            assign_symmetric_block(&mut hessian_psi_psi, offsets[0], w_offset, &h_time_wiggle);
        }

        Ok(Some(ExactNewtonJointPsiSecondOrderTerms {
            objective_psi_psi,
            score_psi_psi,
            hessian_psi_psi,
        }))
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
                d_d2_q: &q.d3_q0 * &dq_en - &q.d_h_h0 * &delta_h0,
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
                d_d2_q: &q.d3_q0 * &delta_q_exit - &q.d_h_h0 * &delta_h0,
            }
        };

        let x_threshold_exit = self.x_threshold.to_dense();
        let x_threshold_entry_owned = self.x_threshold_entry.as_ref().map(DesignMatrix::to_dense);
        let x_threshold_entry = x_threshold_entry_owned
            .as_ref()
            .unwrap_or(&x_threshold_exit);
        let x_log_sigma_exit = self.x_log_sigma.to_dense();
        let x_log_sigma_entry_owned = self.x_log_sigma_entry.as_ref().map(DesignMatrix::to_dense);
        let x_log_sigma_entry = x_log_sigma_entry_owned
            .as_ref()
            .unwrap_or(&x_log_sigma_exit);
        let xw = self.x_link_wiggle.as_ref().map(DesignMatrix::to_dense);

        let dq_t_entry = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
        let dq_ls_entry = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
        let d2q_ls_entry = q.d2q_ls_entry.as_ref().unwrap_or(&q.d2q_ls);

        let q0_psi = &(dq_t_entry * &dir.z_t_entry_psi) + &(dq_ls_entry * &dir.z_ls_entry_psi);
        let q1_psi = &(&q.dq_t * &dir.z_t_exit_psi) + &(&q.dq_ls * &dir.z_ls_exit_psi);
        let z_t_entry_psi_u = dir.x_t_entry_psi.dot(&threshold_dir);
        let z_t_exit_psi_u = dir.x_t_exit_psi.dot(&threshold_dir);
        let z_ls_entry_psi_u = dir.x_ls_entry_psi.dot(&log_sigma_dir);
        let z_ls_exit_psi_u = dir.x_ls_exit_psi.dot(&log_sigma_dir);
        let q0_psi_u = &(&entry_deltas.delta_q_t * &dir.z_t_entry_psi)
            + &(dq_t_entry * &z_t_entry_psi_u)
            + &(&entry_deltas.delta_q_ls * &dir.z_ls_entry_psi)
            + &(dq_ls_entry * &z_ls_entry_psi_u);
        let q1_psi_u = &(&delta_q_t_exit * &dir.z_t_exit_psi)
            + &(&q.dq_t * &z_t_exit_psi_u)
            + &(&delta_q_ls_exit * &dir.z_ls_exit_psi)
            + &(&q.dq_ls * &z_ls_exit_psi_u);
        let mut out = Array2::<f64>::zeros((p_total, p_total));

        let time_time = fast_xt_diag_x(
            &self.x_time_entry,
            &(-(&q.d_h_h0 * &entry_deltas.delta_q * q0_psi) - &(&q.d3_q0 * &q0_psi_u)),
        ) + fast_xt_diag_x(
            &self.x_time_exit,
            &(-(&q.d_h_h1 * &delta_q_exit * q1_psi) - &(&q.d3_q1 * &q1_psi_u)),
        );
        assign_symmetric_block(&mut out, offsets[0], offsets[0], &time_time);

        let h_tt_entry = -(&q.d2_q0 * &dq_t_entry.mapv(|v| v * v));
        let h_tt_exit = -(&q.d2_q1 * &q.dq_t.mapv(|v| v * v));
        let h_tt_entry_u = -(&entry_deltas.d_d2_q * &dq_t_entry.mapv(|v| v * v)
            + &(2.0 * &q.d2_q0 * dq_t_entry * &entry_deltas.delta_q_t));
        let h_tt_exit_u = -(&(&q.d3_q1 * &delta_q_exit - &q.d_h_h1 * &delta_h1)
            * &q.dq_t.mapv(|v| v * v)
            + &(2.0 * &q.d2_q1 * &q.dq_t * &delta_q_t_exit));
        let threshold_threshold =
            weighted_crossprod_dense(&dir.x_t_exit_psi, &h_tt_exit_u, &x_threshold_exit)?
                + weighted_crossprod_dense(&x_threshold_exit, &h_tt_exit_u, &dir.x_t_exit_psi)?
                + weighted_crossprod_dense(&dir.x_t_entry_psi, &h_tt_entry_u, x_threshold_entry)?
                + weighted_crossprod_dense(x_threshold_entry, &h_tt_entry_u, &dir.x_t_entry_psi)?;
        let _ = h_tt_entry;
        let _ = h_tt_exit;
        assign_symmetric_block(&mut out, offsets[1], offsets[1], &threshold_threshold);

        let h_ll_entry = -(&q.d2_q0 * &dq_ls_entry.mapv(|v| v * v) + &(&q.d1_q0 * d2q_ls_entry));
        let h_ll_exit = -(&q.d2_q1 * &q.dq_ls.mapv(|v| v * v) + &(&q.d1_q1 * &q.d2q_ls));
        let h_ll_entry_u = -(&entry_deltas.d_d2_q * &dq_ls_entry.mapv(|v| v * v)
            + &(2.0 * &q.d2_q0 * dq_ls_entry * &entry_deltas.delta_q_ls)
            + &(&entry_deltas.d_d1_q * d2q_ls_entry)
            + &(&q.d1_q0 * &entry_deltas.delta_q_ls_ls));
        let h_ll_exit_u = -(&(&q.d3_q1 * &delta_q_exit - &q.d_h_h1 * &delta_h1)
            * &q.dq_ls.mapv(|v| v * v)
            + &(2.0 * &q.d2_q1 * &q.dq_ls * &delta_q_ls_exit)
            + &((&q.d2_q1 * &delta_q_exit + &q.h_time_h1 * &delta_h1) * &q.d2q_ls)
            + &(&q.d1_q1 * &delta_q_ls_ls_exit));
        let log_sigma_log_sigma =
            weighted_crossprod_dense(&dir.x_ls_exit_psi, &h_ll_exit_u, &x_log_sigma_exit)?
                + weighted_crossprod_dense(&x_log_sigma_exit, &h_ll_exit_u, &dir.x_ls_exit_psi)?
                + weighted_crossprod_dense(&dir.x_ls_entry_psi, &h_ll_entry_u, x_log_sigma_entry)?
                + weighted_crossprod_dense(x_log_sigma_entry, &h_ll_entry_u, &dir.x_ls_entry_psi)?;
        let _ = h_ll_entry;
        let _ = h_ll_exit;
        assign_symmetric_block(&mut out, offsets[2], offsets[2], &log_sigma_log_sigma);

        let h_tl_entry = -(&q.d2_q0 * &(dq_t_entry * dq_ls_entry)
            + &(&q.d1_q0 * q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls)));
        let h_tl_exit = -(&q.d2_q1 * &(&q.dq_t * &q.dq_ls) + &(&q.d1_q1 * &q.d2q_tls));
        let h_tl_entry_u = -(&entry_deltas.d_d2_q * &(dq_t_entry * dq_ls_entry)
            + &(&q.d2_q0
                * &(&entry_deltas.delta_q_t * dq_ls_entry
                    + dq_t_entry * &entry_deltas.delta_q_ls))
            + &(&entry_deltas.d_d1_q * q.d2q_tls_entry.as_ref().unwrap_or(&q.d2q_tls))
            + &(&q.d1_q0 * &entry_deltas.delta_q_tls));
        let h_tl_exit_u = -(&(&q.d3_q1 * &delta_q_exit - &q.d_h_h1 * &delta_h1)
            * &(&q.dq_t * &q.dq_ls)
            + &(&q.d2_q1 * &(&delta_q_t_exit * &q.dq_ls + &q.dq_t * &delta_q_ls_exit))
            + &((&q.d2_q1 * &delta_q_exit + &q.h_time_h1 * &delta_h1) * &q.d2q_tls)
            + &(&q.d1_q1 * &delta_q_tls_exit));
        let threshold_log_sigma =
            weighted_crossprod_dense(&dir.x_t_exit_psi, &h_tl_exit_u, &x_log_sigma_exit)?
                + weighted_crossprod_dense(&x_threshold_exit, &h_tl_exit_u, &dir.x_ls_exit_psi)?
                + weighted_crossprod_dense(&dir.x_t_entry_psi, &h_tl_entry_u, x_log_sigma_entry)?
                + weighted_crossprod_dense(x_threshold_entry, &h_tl_entry_u, &dir.x_ls_entry_psi)?;
        let _ = h_tl_entry;
        let _ = h_tl_exit;
        assign_symmetric_block(&mut out, offsets[1], offsets[2], &threshold_log_sigma);

        let h_h0_t_u = &entry_deltas.d_d1_q * dq_t_entry + &q.d2_q0 * &entry_deltas.delta_q_t;
        let h_h1_t_u = &(&q.d2_q1 * &delta_q_exit + &q.h_time_h1 * &delta_h1) * &q.dq_t
            + &q.d2_q1 * &delta_q_t_exit;
        let h_h0_ls_u = &entry_deltas.d_d1_q * dq_ls_entry + &q.d2_q0 * &entry_deltas.delta_q_ls;
        let h_h1_ls_u = &(&q.d2_q1 * &delta_q_exit + &q.h_time_h1 * &delta_h1) * &q.dq_ls
            + &q.d2_q1 * &delta_q_ls_exit;
        let time_threshold =
            weighted_crossprod_dense(&self.x_time_entry, &h_h0_t_u, &dir.x_t_entry_psi)?
                + weighted_crossprod_dense(&self.x_time_exit, &h_h1_t_u, &dir.x_t_exit_psi)?;
        let time_log_sigma =
            weighted_crossprod_dense(&self.x_time_entry, &h_h0_ls_u, &dir.x_ls_entry_psi)?
                + weighted_crossprod_dense(&self.x_time_exit, &h_h1_ls_u, &dir.x_ls_exit_psi)?;
        assign_symmetric_block(&mut out, offsets[0], offsets[1], &time_threshold);
        assign_symmetric_block(&mut out, offsets[0], offsets[2], &time_log_sigma);

        if let (Some(xw_dense), Some(w_offset)) = (xw.as_ref(), offsets.get(3).copied()) {
            let d_d2_q_combined =
                if self.x_threshold_entry.is_some() || self.x_log_sigma_entry.is_some() {
                    &(&q.d3_q1 * &delta_q_exit - &q.d_h_h1 * &delta_h1) + &entry_deltas.d_d2_q
                } else {
                    &q.d3_q * &delta_q_exit - &q.d_h_h0 * &delta_h0 - &q.d_h_h1 * &delta_h1
                };
            let threshold_wiggle = weighted_crossprod_dense(
                &dir.x_t_exit_psi,
                &(-(&d_d2_q_combined * &q.dq_t)),
                xw_dense,
            )? + weighted_crossprod_dense(
                &dir.x_t_entry_psi,
                &(-(&d_d2_q_combined * dq_t_entry)),
                xw_dense,
            )?;
            let log_sigma_wiggle = weighted_crossprod_dense(
                &dir.x_ls_exit_psi,
                &(-(&d_d2_q_combined * &q.dq_ls)),
                xw_dense,
            )? + weighted_crossprod_dense(
                &dir.x_ls_entry_psi,
                &(-(&d_d2_q_combined * dq_ls_entry)),
                xw_dense,
            )?;
            let time_wiggle =
                weighted_crossprod_dense(&self.x_time_entry, &(&q.d3_q0 * &q0_psi_u), xw_dense)?
                    + weighted_crossprod_dense(
                        &self.x_time_exit,
                        &(&q.d3_q1 * &q1_psi_u),
                        xw_dense,
                    )?;
            assign_symmetric_block(&mut out, offsets[1], w_offset, &threshold_wiggle);
            assign_symmetric_block(&mut out, offsets[2], w_offset, &log_sigma_wiggle);
            assign_symmetric_block(&mut out, offsets[0], w_offset, &time_wiggle);
        }

        Ok(Some(out))
    }

    fn exact_newton_joint_hessiansecond_directional_derivative(
        &self,
        block_states: &[ParameterBlockState],
        d_beta_u_flat: &Array1<f64>,
        d_beta_v_flat: &Array1<f64>,
    ) -> Result<Option<Array2<f64>>, String> {
        let q = self.collect_joint_quantities(block_states)?;
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
        let delta_h0_u = self.x_time_entry.dot(&time_dir_u);
        let delta_h1_u = self.x_time_exit.dot(&time_dir_u);
        let delta_d_u = self.x_time_deriv.dot(&time_dir_u);
        let delta_t_exit_u = self.x_threshold.matrixvectormultiply(&threshold_dir_u);
        let delta_ls_exit_u = self.x_log_sigma.matrixvectormultiply(&log_sigma_dir_u);
        let deltaw_u = match (self.x_link_wiggle.as_ref(), wiggle_dir_u.as_ref()) {
            (Some(xw), Some(dir)) => xw.matrixvectormultiply(dir),
            _ => Array1::zeros(self.n),
        };

        // -- Predictor-space deltas for direction v --
        let delta_h0_v = self.x_time_entry.dot(&time_dir_v);
        let delta_h1_v = self.x_time_exit.dot(&time_dir_v);
        let delta_d_v = self.x_time_deriv.dot(&time_dir_v);
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

        // Perturbed curvature quantities for direction u.
        let d_d1_q_exit_u = &q.d2_q1 * &delta_q_exit_u + &q.h_time_h1 * &delta_h1_u;
        let d_d2_q_exit_u = &q.d3_q1 * &delta_q_exit_u - &q.d_h_h1 * &delta_h1_u;

        // Perturbed curvature quantities for direction v.
        let d_d1_q_exit_v = &q.d2_q1 * &delta_q_exit_v + &q.h_time_h1 * &delta_h1_v;
        let d_d2_q_exit_v = &q.d3_q1 * &delta_q_exit_v - &q.d_h_h1 * &delta_h1_v;

        let x_threshold_exit = self.x_threshold.to_dense();
        let x_threshold_entry = self.x_threshold_entry.as_ref().map(DesignMatrix::to_dense);
        let x_log_sigma_exit = self.x_log_sigma.to_dense();
        let x_log_sigma_entry = self.x_log_sigma_entry.as_ref().map(DesignMatrix::to_dense);
        let xw = self.x_link_wiggle.as_ref().map(DesignMatrix::to_dense);
        let mut joint = Array2::<f64>::zeros((p_total, p_total));

        // --- Entry-side deltas (analogous to first derivative) ---
        struct EntryDeltas2 {
            delta_q_u: Array1<f64>,
            delta_q_t_u: Array1<f64>,
            delta_q_ls_u: Array1<f64>,
            delta_q_tls_u: Array1<f64>,
            delta_q_ls_ls_u: Array1<f64>,
            d_d1_q_u: Array1<f64>,
            d_d2_q_u: Array1<f64>,
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
                let d_d2_q = &q.d3_q0 * &dq_en - &q.d_h_h0 * delta_h0;
                (dq_en, dq_t, dq_ls, dq_tls, dq_ls_ls, d_d1_q, d_d2_q)
            };
            let (dq_u, dqt_u, dqls_u, dqtls_u, dqlsls_u, dd1_u, dd2_u) =
                compute_entry(&threshold_dir_u, &log_sigma_dir_u, &deltaw_u, &delta_h0_u);
            let (dq_v, dqt_v, dqls_v, dqtls_v, dqlsls_v, dd1_v, dd2_v) =
                compute_entry(&threshold_dir_v, &log_sigma_dir_v, &deltaw_v, &delta_h0_v);
            EntryDeltas2 {
                delta_q_u: dq_u,
                delta_q_t_u: dqt_u,
                delta_q_ls_u: dqls_u,
                delta_q_tls_u: dqtls_u,
                delta_q_ls_ls_u: dqlsls_u,
                d_d1_q_u: dd1_u,
                d_d2_q_u: dd2_u,
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
                delta_q_u: delta_q_exit_u.clone(),
                delta_q_t_u: delta_q_t_exit_u.clone(),
                delta_q_ls_u: delta_q_ls_exit_u.clone(),
                delta_q_tls_u: delta_q_tls_exit_u.clone(),
                delta_q_ls_ls_u: delta_q_ls_ls_exit_u.clone(),
                d_d1_q_u: &q.d2_q0 * &delta_q_exit_u + &q.h_time_h0 * &delta_h0_u,
                d_d2_q_u: &q.d3_q0 * &delta_q_exit_u - &q.d_h_h0 * &delta_h0_u,
                delta_q_v: delta_q_exit_v.clone(),
                delta_q_t_v: delta_q_t_exit_v.clone(),
                delta_q_ls_v: delta_q_ls_exit_v.clone(),
                delta_q_tls_v: delta_q_tls_exit_v.clone(),
                delta_q_ls_ls_v: delta_q_ls_ls_exit_v.clone(),
                d_d1_q_v: &q.d2_q0 * &delta_q_exit_v + &q.h_time_h0 * &delta_h0_v,
                d_d2_q_v: &q.d3_q0 * &delta_q_exit_v - &q.d_h_h0 * &delta_h0_v,
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
        // Together these provide the complete 4th-order Faà di Bruno formula
        // for the (ϑ,s,s,s) and (s,s,s,s) blocks of the outer Hessian drift
        // Q[v_k, v_l]. See response.md Section 6.
        //
        // For the exit contribution:
        //   d²(d2_q1)/dβ² [u,v] uses d³ℓ/dq1³ products and would need d⁴ℓ/dq1⁴.
        //   We use the available d3_q1 as the coefficient.
        //
        // The cross-term structure for the Hessian weight d2_q1 under β-perturbation:
        //   D_u(d2_q1) = d3_q1 · δq_u + correction_from_h
        //   D²_{u,v}(d2_q1) = d3_q1 · δq_u · δq_v (leading term, ignoring 4th deriv)
        //     + d2_q1 contributions from cross-perturbations of δq
        //
        // For robustness, we use:
        //   D²_{u,v}(d2_q) ≈ d3_q · δq_u · δq_v (fourth-deriv terms dropped)

        // --- Time block D²H[u,v] ---
        // The time Hessian weight for h0 is h_time_h0[i].
        // D_u(h_time_h0) = d_h_h0 · (δh0_u - δq_u)  (from row_derivatives: sign)
        // D²_{u,v}(h_time_h0) involves the fourth derivative of ℓ w.r.t. q0.
        // We approximate by products of the directional perturbation weights.
        //
        // The key products:
        //   dh_h0_u = d_h_h0 * (delta_h0_u - delta_q_entry_u)
        //   dh_h1_u = d_h_h1 * (delta_h1_u - delta_q_exit_u)
        //   dh_d_u  = d_h_d * delta_d_u
        //
        // For the second derivative, the bilinear contribution is:
        //   d²h/d(h0)² * (δh0_u - δq0_u) * (δh0_v - δq0_v)
        // where d²h/d(h0)² is the fourth derivative of ℓ w.r.t. h0.
        //
        // Since d_h_h0 = d³ℓ/dh0³ and d²h/dh0² = d⁴ℓ/dh0⁴, and we lack
        // the fourth derivative, we use the third-derivative product structure
        // which captures the leading contribution.

        let xi_h0_u = &delta_h0_u - &entry_deltas.delta_q_u;
        let xi_h1_u = &delta_h1_u - &delta_q_exit_u;
        let xi_h0_v = &delta_h0_v - &entry_deltas.delta_q_v;
        let xi_h1_v = &delta_h1_v - &delta_q_exit_v;

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
        let d2_h_time = fast_xt_diag_x(&self.x_time_entry, &d2h_h0)
            + fast_xt_diag_x(&self.x_time_exit, &d2h_h1)
            + fast_xt_diag_x(&self.x_time_deriv, &d2h_d);
        assign_symmetric_block(&mut joint, offsets[0], offsets[0], &d2_h_time);

        // --- Threshold-threshold D²H[u,v] ---
        if let Some(x_t_en) = x_threshold_entry.as_ref() {
            let dq_t_en = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
            // Exit contribution: bilinear product of the two perturbation directions.
            //
            // NOTE: This split-case formula uses D_u(d2_q1)·δq_v (a product of
            // first directional derivatives) rather than the exact bilinear
            // D²_{u,v}(d2_q1) = d4_q1·δu1_u·δu1_v. The 4th-order quantities
            // d4_q0, d4_q1, d2_h_h0, d2_h_h1 are available in q for a future
            // upgrade to the exact split-case bilinear. See response.md Section 6.
            let d2_w_exit = &d_d2_q_exit_u * &delta_q_exit_v * &q.dq_t.mapv(|v| v * v)
                + &d_d2_q_exit_v * &delta_q_exit_u * &q.dq_t.mapv(|v| v * v)
                + &q.d2_q1 * &(2.0 * &delta_q_t_exit_u * &delta_q_t_exit_v);
            // Entry contribution.
            let d2_w_entry =
                &entry_deltas.d_d2_q_u * &entry_deltas.delta_q_v * &dq_t_en.mapv(|v| v * v)
                    + &entry_deltas.d_d2_q_v * &entry_deltas.delta_q_u * &dq_t_en.mapv(|v| v * v)
                    + &q.d2_q0 * &(2.0 * &entry_deltas.delta_q_t_u * &entry_deltas.delta_q_t_v);
            let d2_h_tt =
                weighted_crossprod_dense(&x_threshold_exit, &(-&d2_w_exit), &x_threshold_exit)?
                    + weighted_crossprod_dense(x_t_en, &(-&d2_w_entry), x_t_en)?;
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &d2_h_tt);
        } else {
            // Exact 4th-order bilinear D²(d2_q) — same correction as the
            // log-sigma block. See comment there for derivation.
            let d2_d2_q = &q.d4_q * &(&delta_q_exit_u * &delta_q_exit_v)
                - &q.d2_h_h0 * &(&delta_h0_u * &delta_h0_v)
                - &q.d2_h_h1 * &(&delta_h1_u * &delta_h1_v);
            let d2_w = &d2_d2_q * &q.dq_t.mapv(|v| v * v)
                + &q.d2_q * &(2.0 * &delta_q_t_exit_u * &delta_q_t_exit_v);
            let d2_h_tt =
                weighted_crossprod_dense(&x_threshold_exit, &(-&d2_w), &x_threshold_exit)?;
            assign_symmetric_block(&mut joint, offsets[1], offsets[1], &d2_h_tt);
        }

        // --- Log-sigma-log-sigma D²H[u,v] ---
        if let Some(x_ls_en) = x_log_sigma_entry.as_ref() {
            let dq_ls_en = q.dq_ls_entry.as_ref().unwrap();
            // NOTE: Same first-order product approximation as TT block above.
            // Uses D_u(d2_q)·δq_v instead of exact D²(d2_q) = d4_q·δu_u·δu_v.
            // The 4th-order quantities are available for future upgrade.
            let d2_w_exit = &d_d2_q_exit_u * &delta_q_exit_v * &q.dq_ls.mapv(|v| v * v)
                + &d_d2_q_exit_v * &delta_q_exit_u * &q.dq_ls.mapv(|v| v * v)
                + &q.d2_q1 * &(2.0 * &delta_q_ls_exit_u * &delta_q_ls_exit_v)
                + &d_d1_q_exit_u * &delta_q_ls_ls_exit_v
                + &d_d1_q_exit_v * &delta_q_ls_ls_exit_u;
            let d2_w_entry =
                &entry_deltas.d_d2_q_u * &entry_deltas.delta_q_v * &dq_ls_en.mapv(|v| v * v)
                    + &entry_deltas.d_d2_q_v * &entry_deltas.delta_q_u * &dq_ls_en.mapv(|v| v * v)
                    + &q.d2_q0 * &(2.0 * &entry_deltas.delta_q_ls_u * &entry_deltas.delta_q_ls_v)
                    + &entry_deltas.d_d1_q_u * &entry_deltas.delta_q_ls_ls_v
                    + &entry_deltas.d_d1_q_v * &entry_deltas.delta_q_ls_ls_u;
            let d2_h_ll =
                weighted_crossprod_dense(&x_log_sigma_exit, &(-&d2_w_exit), &x_log_sigma_exit)?
                    + weighted_crossprod_dense(x_ls_en, &(-&d2_w_entry), x_ls_en)?;
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &d2_h_ll);
        } else {
            let d_d1_q_u =
                &q.d2_q * &delta_q_exit_u + &q.h_time_h0 * &delta_h0_u + &q.h_time_h1 * &delta_h1_u;
            let d_d1_q_v =
                &q.d2_q * &delta_q_exit_v + &q.h_time_h0 * &delta_h0_v + &q.h_time_h1 * &delta_h1_v;
            // Bilinear D²(d2_q) using the exact 4th-order ℓ derivatives.
            //
            // d2_q = d2_q0 + d2_q1 where d2_q0 is a function of u0 = q - h0
            // and d2_q1 is a function of u1 = q - h1.
            //
            // D²_{u,v}(d2_q0) = d4_q0 · δu0_u · δu0_v = d4_q0 · (δq-δh0)_u · (δq-δh0)_v
            // D²_{u,v}(d2_q1) = d4_q1 · δu1_u · δu1_v = d4_q1 · (δq-δh1)_u · (δq-δh1)_v
            //
            // For time-invariant blocks, δq0 = δq1 = δq_exit.
            // In the combined form:
            //   D²(d2_q) ≈ d4_q · (δq_exit_u · δq_exit_v)
            //             - d2_h_h0 · (δh0_u · δh0_v)
            //             - d2_h_h1 · (δh1_u · δh1_v)
            //
            // Previously this used d3_q and d_h_h0/d_h_h1 (3rd-order approx).
            // Now uses the exact 4th derivatives. See response.md Section 6.
            let d2_d2_q = &q.d4_q * &(&delta_q_exit_u * &delta_q_exit_v)
                - &q.d2_h_h0 * &(&delta_h0_u * &delta_h0_v)
                - &q.d2_h_h1 * &(&delta_h1_u * &delta_h1_v);
            let d2_w = &d2_d2_q * &q.dq_ls.mapv(|v| v * v)
                + &q.d2_q * &(2.0 * &delta_q_ls_exit_u * &delta_q_ls_exit_v)
                + &d_d1_q_u * &delta_q_ls_ls_exit_v
                + &d_d1_q_v * &delta_q_ls_ls_exit_u;
            let d2_h_ll =
                weighted_crossprod_dense(&x_log_sigma_exit, &(-&d2_w), &x_log_sigma_exit)?;
            assign_symmetric_block(&mut joint, offsets[2], offsets[2], &d2_h_ll);
        }

        // --- Threshold-log-sigma cross D²H[u,v] ---
        {
            let has_t_entry = x_threshold_entry.is_some();
            let has_ls_entry = x_log_sigma_entry.is_some();
            if has_t_entry || has_ls_entry {
                let x_t_en = x_threshold_entry.as_ref().unwrap_or(&x_threshold_exit);
                let x_ls_en = x_log_sigma_entry.as_ref().unwrap_or(&x_log_sigma_exit);
                let dq_t_en = q.dq_t_entry.as_ref().unwrap_or(&q.dq_t);
                let dq_ls_en = q.dq_ls_entry.as_ref().unwrap_or(&q.dq_ls);
                // d2q_tls_entry is resolved but not needed in the bilinear cross term;
                // the entry bilinear uses per-axis deltas directly.
                //
                // NOTE: Same first-order product approximation as TT/LL split blocks.
                // Uses D_u(d2_q)·δq_v instead of exact D²(d2_q) = d4_q·δu_u·δu_v.
                // Exit bilinear.
                let d2_w_exit = &d_d2_q_exit_u * &delta_q_exit_v * &(&q.dq_t * &q.dq_ls)
                    + &d_d2_q_exit_v * &delta_q_exit_u * &(&q.dq_t * &q.dq_ls)
                    + &q.d2_q1
                        * &(&delta_q_t_exit_u * &delta_q_ls_exit_v
                            + &delta_q_t_exit_v * &delta_q_ls_exit_u)
                    + &d_d1_q_exit_u * &delta_q_tls_exit_v
                    + &d_d1_q_exit_v * &delta_q_tls_exit_u;
                // Entry bilinear.
                let d2_w_entry =
                    &entry_deltas.d_d2_q_u * &entry_deltas.delta_q_v * &(dq_t_en * dq_ls_en)
                        + &entry_deltas.d_d2_q_v * &entry_deltas.delta_q_u * &(dq_t_en * dq_ls_en)
                        + &q.d2_q0
                            * &(&entry_deltas.delta_q_t_u * &entry_deltas.delta_q_ls_v
                                + &entry_deltas.delta_q_t_v * &entry_deltas.delta_q_ls_u)
                        + &entry_deltas.d_d1_q_u * &entry_deltas.delta_q_tls_v
                        + &entry_deltas.d_d1_q_v * &entry_deltas.delta_q_tls_u;
                let d2_h_tl =
                    weighted_crossprod_dense(&x_threshold_exit, &(-&d2_w_exit), &x_log_sigma_exit)?
                        + weighted_crossprod_dense(x_t_en, &(-&d2_w_entry), x_ls_en)?;
                assign_symmetric_block(&mut joint, offsets[1], offsets[2], &d2_h_tl);
            } else {
                let d_d1_q_u = &q.d2_q * &delta_q_exit_u
                    + &q.h_time_h0 * &delta_h0_u
                    + &q.h_time_h1 * &delta_h1_u;
                let d_d1_q_v = &q.d2_q * &delta_q_exit_v
                    + &q.h_time_h0 * &delta_h0_v
                    + &q.h_time_h1 * &delta_h1_v;
                // Exact 4th-order bilinear D²(d2_q) for the cross block.
                let d2_d2_q = &q.d4_q * &(&delta_q_exit_u * &delta_q_exit_v)
                    - &q.d2_h_h0 * &(&delta_h0_u * &delta_h0_v)
                    - &q.d2_h_h1 * &(&delta_h1_u * &delta_h1_v);
                let d2_w = &d2_d2_q * &(&q.dq_t * &q.dq_ls)
                    + &q.d2_q
                        * &(&delta_q_t_exit_u * &delta_q_ls_exit_v
                            + &delta_q_t_exit_v * &delta_q_ls_exit_u)
                    + &d_d1_q_u * &delta_q_tls_exit_v
                    + &d_d1_q_v * &delta_q_tls_exit_u;
                let d2_h_tl =
                    weighted_crossprod_dense(&x_threshold_exit, &(-&d2_w), &x_log_sigma_exit)?;
                assign_symmetric_block(&mut joint, offsets[1], offsets[2], &d2_h_tl);
            }
        }

        // --- Time-threshold cross D²H[u,v] ---
        {
            let dh_h0_u = &q.d_h_h0 * &(&delta_h0_u - &entry_deltas.delta_q_u);
            let dh_h1_u = &q.d_h_h1 * &(&delta_h1_u - &delta_q_exit_u);
            let dh_h0_v = &q.d_h_h0 * &(&delta_h0_v - &entry_deltas.delta_q_v);
            let dh_h1_v = &q.d_h_h1 * &(&delta_h1_v - &delta_q_exit_v);
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
            let dh_h0_u = &q.d_h_h0 * &(&delta_h0_u - &entry_deltas.delta_q_u);
            let dh_h1_u = &q.d_h_h1 * &(&delta_h1_u - &delta_q_exit_u);
            let dh_h0_v = &q.d_h_h0 * &(&delta_h0_v - &entry_deltas.delta_q_v);
            let dh_h1_v = &q.d_h_h1 * &(&delta_h1_v - &delta_q_exit_v);
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
        if let (Some(xw_dense), Some(w_offset)) = (xw.as_ref(), offsets.get(3).copied()) {
            let d2_d2_q_combined = if x_threshold_entry.is_some() || x_log_sigma_entry.is_some() {
                &d_d2_q_exit_u * &delta_q_exit_v
                    + &d_d2_q_exit_v * &delta_q_exit_u
                    + &entry_deltas.d_d2_q_u * &entry_deltas.delta_q_v
                    + &entry_deltas.d_d2_q_v * &entry_deltas.delta_q_u
            } else {
                let prod = &delta_q_exit_u * &delta_q_exit_v;
                &q.d3_q * &prod
                    - &q.d_h_h0 * &(&delta_h0_u * &delta_h0_v)
                    - &q.d_h_h1 * &(&delta_h1_u * &delta_h1_v)
            };

            // Threshold-wiggle D²H[u,v].
            if let (Some(x_t_en), Some(dq_t_en)) =
                (x_threshold_entry.as_ref(), q.dq_t_entry.as_ref())
            {
                let d2_tw_exit = &d_d2_q_exit_u * &delta_q_exit_v * &q.dq_t
                    + &d_d2_q_exit_v * &delta_q_exit_u * &q.dq_t
                    + &q.d2_q1 * &(&delta_q_t_exit_u * &deltaw_v + &delta_q_t_exit_v * &deltaw_u);
                let d2_tw_entry = &entry_deltas.d_d2_q_u * &entry_deltas.delta_q_v * dq_t_en
                    + &entry_deltas.d_d2_q_v * &entry_deltas.delta_q_u * dq_t_en
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
                let d2_lw_exit = &d_d2_q_exit_u * &delta_q_exit_v * &q.dq_ls
                    + &d_d2_q_exit_v * &delta_q_exit_u * &q.dq_ls
                    + &q.d2_q1 * &(&delta_q_ls_exit_u * &deltaw_v + &delta_q_ls_exit_v * &deltaw_u);
                let d2_lw_entry = &entry_deltas.d_d2_q_u * &entry_deltas.delta_q_v * dq_ls_en
                    + &entry_deltas.d_d2_q_v * &entry_deltas.delta_q_u * dq_ls_en
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
        _: &ParameterBlockSpec,
    ) -> Result<Option<LinearInequalityConstraints>, String> {
        if block_idx != Self::BLOCK_TIME {
            return Ok(None);
        }
        let n = self.x_time_deriv.nrows();
        let p = self.x_time_deriv.ncols();
        if self.offset_time_deriv.len() != n {
            return Err(format!(
                "time derivative offset length mismatch: got {}, expected {n}",
                self.offset_time_deriv.len()
            ));
        }
        if n == 0 || p == 0 || self.derivative_guard <= 0.0 {
            return Ok(None);
        }
        let mut a = Array2::<f64>::zeros((n, p));
        a.assign(&self.x_time_deriv);
        let mut b = Array1::<f64>::zeros(n);
        let guard = self.derivative_guard;
        for i in 0..n {
            b[i] = guard - self.offset_time_deriv[i];
        }
        Ok(Some(LinearInequalityConstraints { a, b }))
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
    let sigma = eta_log_sigma.mapv(f64::exp);
    let q_seed = Array1::from_iter(
        eta_threshold
            .iter()
            .zip(sigma.iter())
            .map(|(&threshold, &scale)| -threshold / scale.max(1e-12)),
    );
    select_wiggle_basis_from_seed(q_seed.view(), wiggle_cfg, wiggle_penalty_orders)
}

fn linkwiggle_block_input_from_selected_basis(
    selected_wiggle_basis: SelectedWiggleBasis,
) -> LinkWiggleBlockInput {
    let crate::families::gamlss::SelectedWiggleBasis { block, .. } = selected_wiggle_basis;
    let crate::families::gamlss::ParameterBlockInput {
        design,
        penalties,
        initial_log_lambdas,
        initial_beta,
        ..
    } = block;
    LinkWiggleBlockInput {
        design,
        penalties,
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
    if spec.linkwiggle_block.is_some() && !inverse_link_supports_joint_wiggle(&spec.inverse_link) {
        return Err(
            "fit_survival_location_scale_terms: link wiggle does not support SAS/BetaLogistic/Mixture links; wiggle is only available for jointly fitted standard links"
                .to_string(),
        );
    }
    let threshold_boot_design =
        build_term_collection_design(data, &spec.thresholdspec).map_err(|e| e.to_string())?;
    let log_sigma_boot_design =
        build_term_collection_design(data, &spec.log_sigmaspec).map_err(|e| e.to_string())?;
    let threshold_bootspec =
        freeze_spatial_length_scale_terms_from_design(&spec.thresholdspec, &threshold_boot_design)
            .map_err(|e| e.to_string())?;
    let log_sigma_bootspec =
        freeze_spatial_length_scale_terms_from_design(&spec.log_sigmaspec, &log_sigma_boot_design)
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
    let analytic_joint_hessian_available = analytic_joint_gradient_available;

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
                penalties: wiggle.penalties.clone(),
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
            derivative_softness: spec.derivative_softness,
            time_anchor: spec.time_anchor,
            max_iter: spec.max_iter,
            tol: spec.tol,
            time_block: TimeBlockInput {
                design_entry: spec.time_block.design_entry.clone(),
                design_exit: spec.time_block.design_exit.clone(),
                design_derivative_exit: spec.time_block.design_derivative_exit.clone(),
                offset_entry: spec.time_block.offset_entry.clone(),
                offset_exit: spec.time_block.offset_exit.clone(),
                derivative_offset_exit: spec.time_block.derivative_offset_exit.clone(),
                penalties: spec.time_block.penalties.clone(),
                initial_log_lambdas: Some(layout.time_from(rho)),
                initial_beta: time_beta,
            },
            threshold_block,
            log_sigma_block,
            linkwiggle_block,
        })
    };

    let solved = optimize_two_block_spatial_length_scale_exact_joint(
        data,
        &spec.thresholdspec,
        &spec.log_sigmaspec,
        kappa_options,
        &joint_setup,
        analytic_joint_gradient_available,
        analytic_joint_hessian_available,
        |rho,
         thresholdspec_resolved,
         log_sigmaspec_resolved,
         threshold_design,
         log_sigma_design| {
            let fit = fit_survival_location_scale(build_spec(
                rho,
                thresholdspec_resolved,
                log_sigmaspec_resolved,
                threshold_design,
                log_sigma_design,
            )?)?;
            Ok(if fit.reml_score.is_finite() {
                fit.reml_score
            } else {
                fit.penalized_objective
            })
        },
        |rho,
         thresholdspec_resolved,
         log_sigmaspec_resolved,
         threshold_design,
         log_sigma_design| {
            let fit = fit_survival_location_scale(build_spec(
                rho,
                thresholdspec_resolved,
                log_sigmaspec_resolved,
                threshold_design,
                log_sigma_design,
            )?)?;
            time_beta_hint.replace(Some(fit.beta_time()));
            threshold_beta_hint.replace(Some(fit.beta_threshold()));
            log_sigma_beta_hint.replace(Some(fit.beta_log_sigma()));
            wiggle_beta_hint.replace(fit.beta_link_wiggle());
            Ok(fit)
        },
        |rho,
         thresholdspec_resolved,
         log_sigmaspec_resolved,
         threshold_design,
         log_sigma_design,
         need_hessian| {
            if !analytic_joint_gradient_available {
                return Err(
                    "analytic spatial psi derivatives are unavailable for survival exact two-block path"
                        .to_string(),
                );
            }
            let assembled = build_spec(
                rho,
                thresholdspec_resolved,
                log_sigmaspec_resolved,
                threshold_design,
                log_sigma_design,
            )?;
            let prepared = prepare_survival_location_scale_model(&assembled)?;
            let threshold_derivs = build_survival_covariate_block_psi_derivatives(
                data,
                thresholdspec_resolved,
                threshold_design,
                &spec.threshold_template,
            )?
            .ok_or_else(|| "missing survival threshold spatial psi derivatives".to_string())?;
            let log_sigma_derivs = build_survival_covariate_block_psi_derivatives(
                data,
                log_sigmaspec_resolved,
                log_sigma_design,
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
                rho,
                &derivative_blocks,
                exact_warm_start.borrow().as_ref(),
                need_hessian && analytic_joint_hessian_available,
            )
            .map_err(|e| e.to_string())?;
            exact_warm_start.replace(Some(eval.warm_start));
            Ok((eval.objective, eval.gradient, eval.outer_hessian))
        },
    )?;

    Ok(SurvivalLocationScaleTermFitResult {
        fit: solved.fit,
        resolved_thresholdspec: solved.resolved_meanspec,
        resolved_log_sigmaspec: solved.resolved_noisespec,
        threshold_design: solved.mean_design,
        log_sigma_design: solved.noise_design,
    })
}

pub fn predict_survival_location_scale(
    input: &SurvivalLocationScalePredictInput,
    fit: &UnifiedFitResult,
) -> Result<SurvivalLocationScalePredictResult, String> {
    let predictors = prediction_linear_predictors(input, fit)?;
    let n = input.x_time_exit.nrows();
    let (sigma, _, _, _) = exp_sigma_derivs_up_to_third(predictors.eta_ls.view());
    let eta = Array1::from_iter(
        predictors
            .h
            .iter()
            .zip(predictors.eta_t.iter())
            .zip(sigma.iter())
            .enumerate()
            .map(|(i, ((&hh, &tt), &ss))| {
                let mut q = -hh - tt / ss.max(1e-12);
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
    // Uncertainty-aware survival posterior mean with conditional Gaussian
    // reduction.
    //
    // The deterministic survival predictor already computes the latent pieces
    //
    //   h  = time block linear predictor
    //   t  = threshold block linear predictor
    //   ls = log-sigma block linear predictor
    //   w  = optional link-wiggle predictor.
    //
    // Under the Gaussian coefficient approximation, the rowwise latent vector
    //
    //   (h, t, ls, w)
    //
    // is jointly Gaussian, with w identically zero when no wiggle block is
    // present. The posterior-mean target is
    //
    //   E[g(eta)],
    //   eta = -h - t / sigma(ls) + w.
    //
    // A direct evaluation is a 3D expectation without wiggle and a 4D
    // expectation with wiggle. We do not need to integrate over all latent
    // dimensions directly. Conditioning on ls is enough:
    //
    //   (h, t, w) | ls  is Gaussian,
    //   eta | ls        is affine in (h, t, w),
    //
    // so eta | ls is itself Gaussian with exact conditional mean and variance.
    // That reduces the full posterior mean to
    //
    //   E[g(eta)] = E_ls[ E[g(eta) | ls] ],
    //
    // i.e. a 1D outer Gaussian expectation over ls, where the inner object is
    // the existing Gaussian-uncertain scalar inverse-link expectation.
    //
    // If the conditioning algebra becomes numerically unsafe, this routine
    // falls back to direct adaptive Gaussian expectation rather than forcing
    // the reduction.
    let pred = predict_survival_location_scale(input, fit)?;
    let n = input.x_time_exit.nrows();
    let predictors = prediction_linear_predictors(input, fit)?;
    let p_time = fit.beta_time().len();
    let p_t = fit.beta_threshold().len();
    let p_ls = fit.beta_log_sigma().len();
    let pw = fit.beta_link_wiggle().map_or(0, |b| b.len());
    let p_total = p_time + p_t + p_ls + pw;
    if covariance.nrows() != p_total || covariance.ncols() != p_total {
        return Err(format!(
            "predict_survival_location_scale_posterior_mean: covariance shape mismatch: got {}x{}, expected {}x{}",
            covariance.nrows(),
            covariance.ncols(),
            p_total,
            p_total
        ));
    }

    if input.x_threshold.nrows() != n || input.x_log_sigma.nrows() != n {
        return Err(
            "predict_survival_location_scale_posterior_mean: row mismatch across design views"
                .to_string(),
        );
    }

    let cov_hh = covariance.slice(s![0..p_time, 0..p_time]).to_owned();
    let cov_tt = covariance
        .slice(s![p_time..p_time + p_t, p_time..p_time + p_t])
        .to_owned();
    let cov_ll = covariance
        .slice(s![
            p_time + p_t..p_time + p_t + p_ls,
            p_time + p_t..p_time + p_t + p_ls
        ])
        .to_owned();
    let cov_ht = covariance
        .slice(s![0..p_time, p_time..p_time + p_t])
        .to_owned();
    let cov_hl = covariance
        .slice(s![0..p_time, p_time + p_t..p_time + p_t + p_ls])
        .to_owned();
    let cov_tl = covariance
        .slice(s![p_time..p_time + p_t, p_time + p_t..p_time + p_t + p_ls])
        .to_owned();
    let (varwrows, cov_hwrows, cov_twrows, cov_lwrows) =
        if let Some(xw) = input.x_link_wiggle.as_ref() {
            let covww = covariance
                .slice(s![
                    p_time + p_t + p_ls..p_total,
                    p_time + p_t + p_ls..p_total
                ])
                .to_owned();
            let cov_hw = covariance
                .slice(s![0..p_time, p_time + p_t + p_ls..p_total])
                .to_owned();
            let cov_tw = covariance
                .slice(s![p_time..p_time + p_t, p_time + p_t + p_ls..p_total])
                .to_owned();
            let cov_lw = covariance
                .slice(s![
                    p_time + p_t..p_time + p_t + p_ls,
                    p_time + p_t + p_ls..p_total
                ])
                .to_owned();
            (
                Some(xw.quadratic_form_diag(&covww)?),
                Some(rowwise_cross_quadratic_dense_design(
                    &input.x_time_exit,
                    &cov_hw,
                    xw,
                )?),
                Some(rowwise_cross_quadratic_design(
                    &input.x_threshold,
                    &cov_tw,
                    xw,
                )?),
                Some(rowwise_cross_quadratic_design(
                    &input.x_log_sigma,
                    &cov_lw,
                    xw,
                )?),
            )
        } else {
            (None, None, None, None)
        };

    let xh_hh = input.x_time_exit.dot(&cov_hh);
    let var_trows = input.x_threshold.quadratic_form_diag(&cov_tt)?;
    let var_lsrows = input.x_log_sigma.quadratic_form_diag(&cov_ll)?;
    let cov_htrows =
        rowwise_cross_quadratic_dense_design(&input.x_time_exit, &cov_ht, &input.x_threshold)?;
    let cov_hlrows =
        rowwise_cross_quadratic_dense_design(&input.x_time_exit, &cov_hl, &input.x_log_sigma)?;
    let cov_tlrows =
        rowwise_cross_quadratic_design(&input.x_threshold, &cov_tl, &input.x_log_sigma)?;

    let mu_h = predictors.h;
    let mu_t = predictors.eta_t;
    let mu_ls = predictors.eta_ls;
    let muw = predictors.etaw;
    let link = input.inverse_link.link_function();
    let mixture_state = input.inverse_link.mixture_state();
    let sas_state = input.inverse_link.sas_state();
    let quadctx = crate::quadrature::QuadratureContext::new();
    const VAR_L_DEGENERATE_TOL: f64 = 1e-12;
    const CROSS_L_DEGENERATE_TOL: f64 = 1e-10;
    let gaussian_survival_mean = |mu_loc: f64, var_loc: f64| {
        let var_loc = var_loc.max(0.0);
        if matches!(input.inverse_link, InverseLink::Mixture(_)) {
            return crate::quadrature::normal_expectation_1d_adaptive(
                &quadctx,
                mu_loc,
                var_loc.sqrt(),
                |z| inverse_link_survival_probvalue(&input.inverse_link, z),
            )
            .clamp(0.0, 1.0);
        }
        crate::quadrature::integrated_inverse_link_jetwith_state(
            &quadctx,
            link,
            mu_loc,
            var_loc.sqrt(),
            mixture_state,
            sas_state,
        )
        .map(|jet| (1.0 - jet.mean).clamp(0.0, 1.0))
        .unwrap_or_else(|_| {
            if link == LinkFunction::Probit {
                let denom = (1.0 + var_loc).sqrt().max(1e-12);
                (1.0 - normal_cdf(mu_loc / denom)).clamp(0.0, 1.0)
            } else {
                crate::quadrature::normal_expectation_1d_adaptive(
                    &quadctx,
                    mu_loc,
                    var_loc.sqrt(),
                    |z| inverse_link_survival_probvalue(&input.inverse_link, z),
                )
                .clamp(0.0, 1.0)
            }
        })
    };

    let fallbackrow = |i: usize| {
        if let Some(muw) = muw.as_ref() {
            crate::quadrature::normal_expectation_nd_adaptive::<4, _>(
                &quadctx,
                [mu_h[i], mu_t[i], mu_ls[i], muw[i]],
                [
                    [
                        var_hrow(i, input, &xh_hh),
                        cov_htrows[i],
                        cov_hlrows[i],
                        cov_hwrows.as_ref().expect("wiggle cov_hw rows")[i],
                    ],
                    [
                        cov_htrows[i],
                        var_trows[i],
                        cov_tlrows[i],
                        cov_twrows.as_ref().expect("wiggle cov_tw rows")[i],
                    ],
                    [
                        cov_hlrows[i],
                        cov_tlrows[i],
                        var_lsrows[i],
                        cov_lwrows.as_ref().expect("wiggle cov_lw rows")[i],
                    ],
                    [
                        cov_hwrows.as_ref().expect("wiggle cov_hw rows")[i],
                        cov_twrows.as_ref().expect("wiggle cov_tw rows")[i],
                        cov_lwrows.as_ref().expect("wiggle cov_lw rows")[i],
                        varwrows.as_ref().expect("wiggle var rows")[i],
                    ],
                ],
                11,
                |x| {
                    let sigma = exp_sigma_derivs_up_to_third_scalar(x[2]).0.max(1e-12);
                    inverse_link_survival_probvalue(
                        &input.inverse_link,
                        -x[0] - x[1] / sigma + x[3],
                    )
                },
            )
            .clamp(0.0, 1.0)
        } else {
            crate::quadrature::normal_expectation_3d_adaptive(
                &quadctx,
                [mu_h[i], mu_t[i], mu_ls[i]],
                [
                    [var_hrow(i, input, &xh_hh), cov_htrows[i], cov_hlrows[i]],
                    [cov_htrows[i], var_trows[i], cov_tlrows[i]],
                    [cov_hlrows[i], cov_tlrows[i], var_lsrows[i]],
                ],
                |h, t, ls| {
                    let sigma = exp_sigma_derivs_up_to_third_scalar(ls).0.max(1e-12);
                    inverse_link_survival_probvalue(&input.inverse_link, -h - t / sigma)
                },
            )
            .clamp(0.0, 1.0)
        }
    };

    let survival_prob = Array1::from_iter((0..n).map(|i| {
        let var_h = var_hrow(i, input, &xh_hh);
        let var_t = var_trows[i];
        let var_ls = var_lsrows[i];
        let cov_ht_i = cov_htrows[i];
        let cov_hl_i = cov_hlrows[i];
        let cov_tl_i = cov_tlrows[i];
        let muw_i = muw.as_ref().map_or(0.0, |vals| vals[i]);
        let varw_i = varwrows.as_ref().map_or(0.0, |vals| vals[i]);
        let cov_hw_i = cov_hwrows.as_ref().map_or(0.0, |vals| vals[i]);
        let cov_tw_i = cov_twrows.as_ref().map_or(0.0, |vals| vals[i]);
        let cov_lw_i = cov_lwrows.as_ref().map_or(0.0, |vals| vals[i]);
        let mu_l_i = mu_ls[i];

        if !(var_h.is_finite()
            && var_t.is_finite()
            && var_ls.is_finite()
            && varw_i.is_finite()
            && cov_ht_i.is_finite()
            && cov_hl_i.is_finite()
            && cov_tl_i.is_finite()
            && cov_hw_i.is_finite()
            && cov_tw_i.is_finite()
            && cov_lw_i.is_finite())
        {
            return fallbackrow(i);
        }

        // Exact degenerate limit: if Var(L)=0, PSD implies the cross-covariances
        // with L vanish, so the outer expectation collapses to the point mass
        // L = mu_l.
        if var_ls <= VAR_L_DEGENERATE_TOL {
            if cov_hl_i.abs() > CROSS_L_DEGENERATE_TOL
                || cov_tl_i.abs() > CROSS_L_DEGENERATE_TOL
                || cov_lw_i.abs() > CROSS_L_DEGENERATE_TOL
            {
                return fallbackrow(i);
            }
            let sigma = exp_sigma_derivs_up_to_third_scalar(mu_l_i).0.max(1e-12);
            let q_l = 1.0 / sigma;
            let mu_base = -mu_h[i] - q_l * mu_t[i] + muw_i;
            let var_base = var_h + q_l * q_l * var_t + varw_i + 2.0 * q_l * cov_ht_i
                - 2.0 * cov_hw_i
                - 2.0 * q_l * cov_tw_i;
            let mu_loc = mu_base;
            let var_loc = var_base.max(0.0);
            return gaussian_survival_mean(mu_loc, var_loc);
        }

        crate::quadrature::normal_expectation_1d_adaptive(&quadctx, mu_l_i, var_ls.sqrt(), |ls| {
            let sigma = exp_sigma_derivs_up_to_third_scalar(ls).0.max(1e-12);
            let q_l = 1.0 / sigma;
            let delta_l = ls - mu_l_i;
            let mu_base = -mu_h[i] - q_l * mu_t[i] + muw_i;
            let cov_eta_l = cov_hl_i + q_l * cov_tl_i - cov_lw_i;
            let mu_loc = mu_base - (cov_eta_l / var_ls) * delta_l;
            let var_base = var_h + q_l * q_l * var_t + varw_i + 2.0 * q_l * cov_ht_i
                - 2.0 * cov_hw_i
                - 2.0 * q_l * cov_tw_i;
            let var_loc = (var_base - cov_eta_l * cov_eta_l / var_ls).max(0.0);
            if !mu_loc.is_finite() || !var_loc.is_finite() {
                return fallbackrow(i);
            }
            // This is the payoff of the conditional Gaussian reduction above:
            // for fixed ls, eta = -h - t / sigma(ls) + w is Gaussian, so we
            // hand its conditional mean and standard deviation straight to
            // the shared integrated-expectation dispatcher instead of
            // integrating over h, t, and w explicitly.
            gaussian_survival_mean(mu_loc, var_loc)
        })
        .clamp(0.0, 1.0)
    }));

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
    if pw > 0 && (beta_link_wiggle.is_none() || input.x_link_wiggle.is_none()) {
        return Err(
            "predict_survival_location_scalewith_uncertainty: wiggle covariance provided but wiggle design/beta is partial"
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
    if input
        .x_link_wiggle
        .as_ref()
        .is_some_and(|xw| xw.nrows() != n)
    {
        return Err(
            "predict_survival_location_scalewith_uncertainty: x_link_wiggle row mismatch"
                .to_string(),
        );
    }

    let (sigma, ds, _, _) = exp_sigma_derivs_up_to_third(predictors.eta_ls.view());

    let cov_hh = covariance.slice(s![0..p_time, 0..p_time]).to_owned();
    let cov_tt = covariance
        .slice(s![p_time..p_time + p_t, p_time..p_time + p_t])
        .to_owned();
    let cov_ll = covariance
        .slice(s![
            p_time + p_t..p_time + p_t + p_ls,
            p_time + p_t..p_time + p_t + p_ls
        ])
        .to_owned();
    let cov_ht = covariance
        .slice(s![0..p_time, p_time..p_time + p_t])
        .to_owned();
    let cov_hl = covariance
        .slice(s![0..p_time, p_time + p_t..p_time + p_t + p_ls])
        .to_owned();
    let cov_tl = covariance
        .slice(s![p_time..p_time + p_t, p_time + p_t..p_time + p_t + p_ls])
        .to_owned();
    let xh_hh = input.x_time_exit.dot(&cov_hh);
    let var_hrows =
        Array1::from_iter((0..n).map(|i| input.x_time_exit.row(i).dot(&xh_hh.row(i)).max(0.0)));
    let var_trows = input.x_threshold.quadratic_form_diag(&cov_tt)?;
    let var_lsrows = input.x_log_sigma.quadratic_form_diag(&cov_ll)?;
    let cov_htrows =
        rowwise_cross_quadratic_dense_design(&input.x_time_exit, &cov_ht, &input.x_threshold)?;
    let cov_hlrows =
        rowwise_cross_quadratic_dense_design(&input.x_time_exit, &cov_hl, &input.x_log_sigma)?;
    let cov_tlrows =
        rowwise_cross_quadratic_design(&input.x_threshold, &cov_tl, &input.x_log_sigma)?;
    let (varwrows, cov_hwrows, cov_twrows, cov_lwrows) =
        if let Some(xw) = input.x_link_wiggle.as_ref() {
            let covww = covariance
                .slice(s![
                    p_time + p_t + p_ls..p_total,
                    p_time + p_t + p_ls..p_total
                ])
                .to_owned();
            let cov_hw = covariance
                .slice(s![0..p_time, p_time + p_t + p_ls..p_total])
                .to_owned();
            let cov_tw = covariance
                .slice(s![p_time..p_time + p_t, p_time + p_t + p_ls..p_total])
                .to_owned();
            let cov_lw = covariance
                .slice(s![
                    p_time + p_t..p_time + p_t + p_ls,
                    p_time + p_t + p_ls..p_total
                ])
                .to_owned();
            (
                Some(xw.quadratic_form_diag(&covww)?),
                Some(rowwise_cross_quadratic_dense_design(
                    &input.x_time_exit,
                    &cov_hw,
                    xw,
                )?),
                Some(rowwise_cross_quadratic_design(
                    &input.x_threshold,
                    &cov_tw,
                    xw,
                )?),
                Some(rowwise_cross_quadratic_design(
                    &input.x_log_sigma,
                    &cov_lw,
                    xw,
                )?),
            )
        } else {
            (None, None, None, None)
        };

    let mut etavar = Array1::<f64>::zeros(n);
    for i in 0..n {
        let inv_sigma = 1.0 / sigma[i].max(1e-12);
        let coeff_ls = predictors.eta_t[i] * ds[i] / sigma[i].powi(2).max(1e-12);
        let mut acc = var_hrows[i]
            + inv_sigma * inv_sigma * var_trows[i]
            + coeff_ls * coeff_ls * var_lsrows[i]
            + 2.0 * inv_sigma * cov_htrows[i]
            - 2.0 * coeff_ls * cov_hlrows[i]
            - 2.0 * inv_sigma * coeff_ls * cov_tlrows[i];
        if let Some(varw) = varwrows.as_ref() {
            acc += varw[i];
        }
        if let Some(cov_hw) = cov_hwrows.as_ref() {
            acc -= 2.0 * cov_hw[i];
        }
        if let Some(cov_tw) = cov_twrows.as_ref() {
            acc -= 2.0 * inv_sigma * cov_tw[i];
        }
        if let Some(cov_lw) = cov_lwrows.as_ref() {
            acc += 2.0 * coeff_ls * cov_lw[i];
        }
        etavar[i] = acc.max(0.0);
    }
    let eta_se = etavar.mapv(f64::sqrt);

    let survival_prob = if posterior_mean {
        predict_survival_location_scale_posterior_mean(input, fit, covariance)?.survival_prob
    } else {
        base.survival_prob.clone()
    };

    let response_standard_error = if include_response_sd {
        let quadctx = crate::quadrature::QuadratureContext::new();
        Some(Array1::from_iter((0..n).map(|i| {
            let m2 = crate::quadrature::normal_expectation_1d_adaptive(
                &quadctx,
                base.eta[i],
                eta_se[i],
                |x| {
                    let p = inverse_link_survival_probvalue(&input.inverse_link, x);
                    p * p
                },
            );
            (m2 - survival_prob[i] * survival_prob[i]).max(0.0).sqrt()
        })))
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

#[inline]
fn var_hrow(i: usize, input: &SurvivalLocationScalePredictInput, xh_hh: &Array2<f64>) -> f64 {
    input.x_time_exit.row(i).dot(&xh_hh.row(i)).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn test_survival_fit(
        beta_time: Array1<f64>,
        beta_threshold: Array1<f64>,
        beta_log_sigma: Array1<f64>,
        beta_link_wiggle: Option<Array1<f64>>,
    ) -> UnifiedFitResult {
        let lambdas_linkwiggle = beta_link_wiggle.as_ref().map(|_| Array1::zeros(0));
        survival_fit_from_parts(SurvivalLocationScaleFitResultParts {
            beta_time,
            beta_threshold,
            beta_log_sigma,
            beta_link_wiggle,
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
            derivative_softness: 1e-6,
            x_time_entry: array![[1.0], [1.0], [1.0]],
            x_time_exit: array![[1.2], [0.9], [1.4]],
            x_time_deriv: array![[1.0], [1.0], [1.0]],
            offset_time_deriv: array![0.5, 0.7, 0.6],
            x_threshold: DesignMatrix::Dense(array![[1.0], [0.4], [-0.6]]),
            x_threshold_entry: None,
            x_log_sigma: DesignMatrix::Dense(array![[1.0], [-0.3], [0.5]]),
            x_log_sigma_entry: None,
            x_link_wiggle: None,
        }
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
                design: DesignMatrix::Dense(array![
                    [1.0],
                    [1.0],
                    [1.0],
                    [1.2],
                    [0.9],
                    [1.4],
                    [1.0],
                    [1.0],
                    [1.0]
                ]),
                offset: Array1::zeros(9),
                penalties: vec![Array2::eye(1)],
                initial_log_lambdas: array![0.0],
                initial_beta: Some(array![0.2]),
            },
            ParameterBlockSpec {
                name: "threshold".to_string(),
                design: DesignMatrix::Dense(array![[1.0], [0.4], [-0.6]]),
                offset: Array1::zeros(3),
                penalties: Vec::new(),
                initial_log_lambdas: Array1::zeros(0),
                initial_beta: Some(array![0.35]),
            },
            ParameterBlockSpec {
                name: "log_sigma".to_string(),
                design: DesignMatrix::Dense(array![[1.0], [-0.3], [0.5]]),
                offset: Array1::zeros(3),
                penalties: Vec::new(),
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
    fn identified_time_blockzeroes_anchorrow() {
        let design_entry = array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]];
        let design_exit = array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]];
        let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
        let time_block = TimeBlockInput {
            design_entry,
            design_exit,
            design_derivative_exit,
            offset_entry: Array1::zeros(3),
            offset_exit: Array1::zeros(3),
            derivative_offset_exit: Array1::zeros(3),
            penalties: vec![Array2::eye(3)],
            initial_log_lambdas: None,
            initial_beta: None,
        };
        let prepared = prepare_identified_time_block(&time_block, 0).expect("prepare time block");
        let entry_anchorrow = prepared.design_entry.row(0);
        let entry_max_abs = entry_anchorrow
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max);
        assert!(
            entry_max_abs <= 1e-10,
            "entry anchor row not zero after identifiability transform: max_abs={entry_max_abs}"
        );

        let exit_anchorrow = prepared.design_exit.row(0);
        let exit_max_abs = exit_anchorrow
            .iter()
            .copied()
            .map(f64::abs)
            .fold(0.0, f64::max);
        assert!(
            exit_max_abs > 1e-6,
            "test setup should keep exit anchor row distinct from zero to detect anchor mixups"
        );
    }

    #[test]
    fn identified_time_block_preserves_expected_nullspace_dimension() {
        let design_entry = array![[1.0, 0.0, 0.2], [1.0, 1.0, 0.5], [1.0, 2.0, 1.0]];
        let design_exit = array![[1.0, 0.5, 0.3], [1.0, 1.5, 0.8], [1.0, 2.5, 1.4]];
        let design_derivative_exit = array![[0.0, 1.0, 0.2], [0.0, 1.0, 0.3], [0.0, 1.0, 0.4]];
        let time_block = TimeBlockInput {
            design_entry,
            design_exit,
            design_derivative_exit,
            offset_entry: Array1::zeros(3),
            offset_exit: Array1::zeros(3),
            derivative_offset_exit: Array1::zeros(3),
            penalties: vec![Array2::eye(3)],
            initial_log_lambdas: None,
            initial_beta: None,
        };

        let prepared = prepare_identified_time_block(&time_block, 0).expect("prepare time block");
        let p = time_block.design_entry.ncols();

        assert_eq!(
            prepared.transform.z.nrows(),
            p,
            "identifiability transform must stay in the original coefficient space"
        );
        assert_eq!(
            prepared.transform.z.ncols(),
            p - 1,
            "a single nonzero anchor row should remove exactly one time coefficient dimension"
        );
        assert_eq!(
            prepared.design_entry.ncols(),
            p - 1,
            "prepared entry design should keep the p-1 nullspace basis columns"
        );
        assert_eq!(
            prepared.design_exit.ncols(),
            p - 1,
            "prepared exit design should keep the p-1 nullspace basis columns"
        );
    }

    #[test]
    fn identified_time_block_degenerate_entry_uses_exit_design() {
        // When entry design row is all zeros (degenerate entry times),
        // the identifiability constraint should fall back to the exit design.
        let design_entry = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        let design_exit = array![[0.1, 0.5, 0.9], [0.2, 0.6, 1.0], [0.3, 0.7, 1.0]];
        let design_derivative_exit = array![[0.1, 0.1, 0.0], [0.1, 0.1, 0.0], [0.1, 0.1, 0.0]];
        let time_block = TimeBlockInput {
            design_entry,
            design_exit,
            design_derivative_exit,
            offset_entry: Array1::zeros(3),
            offset_exit: Array1::zeros(3),
            derivative_offset_exit: Array1::zeros(3),
            penalties: vec![Array2::eye(3)],
            initial_log_lambdas: None,
            initial_beta: None,
        };
        let prepared = prepare_identified_time_block(&time_block, 0).expect("prepare time block");
        // With a non-trivial exit design row, one dimension should be removed.
        assert_eq!(
            prepared.design_exit.ncols(),
            2,
            "degenerate entry should still remove one dimension using exit design"
        );
        // The exit anchor row (after transform) should be zeroed by the constraint.
        let exit_anchor = prepared.design_exit.row(0);
        let exit_anchor_max = exit_anchor.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
        assert!(
            exit_anchor_max < 1e-10,
            "exit anchor row should be near zero after constraint: max_abs={exit_anchor_max}"
        );
    }

    #[test]
    fn select_anchorrow_defaults_to_earliest_entry() {
        let age_entry = array![5.0, 1.0, 3.0];
        let idx = select_anchorrow(&age_entry, None).expect("select default anchor");
        assert_eq!(idx, 1);
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
                let s = (1.0 - dist.cdf(z)).clamp(MIN_PROB, 1.0);
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
    fn clamped_log_pdf_derivatives_arezero_in_saturated_region() {
        let f = MIN_PROB * 0.1;
        let fp = 3.0;
        let fpp = -7.0;
        let fppp = 11.0;
        let (logf, d1, d2, d3) =
            SurvivalLocationScaleFamily::clamped_log_pdfwith_derivatives(f, fp, fpp, fppp);
        assert!((logf - MIN_PROB.ln()).abs() <= 1e-15);
        assert_eq!(d1, 0.0);
        assert_eq!(d2, 0.0);
        assert_eq!(d3, 0.0);
    }

    #[test]
    fn clamped_survival_neglog_derivatives_arezero_on_clamp_bounds() {
        // Lower clamp active.
        let (s_low, r_low, dr_low, ddr_low) =
            SurvivalLocationScaleFamily::clamped_survival_neglog_derivatives(
                MIN_PROB * 0.1,
                0.2,
                -0.3,
                0.4,
            );
        assert_eq!(s_low, MIN_PROB);
        assert_eq!(r_low, 0.0);
        assert_eq!(dr_low, 0.0);
        assert_eq!(ddr_low, 0.0);

        // Upper clamp active.
        let (s_high, r_high, dr_high, ddr_high) =
            SurvivalLocationScaleFamily::clamped_survival_neglog_derivatives(1.1, 0.2, -0.3, 0.4);
        assert_eq!(s_high, 1.0);
        assert_eq!(r_high, 0.0);
        assert_eq!(dr_high, 0.0);
        assert_eq!(ddr_high, 0.0);
    }

    #[test]
    fn clamped_logwith_derivatives_is_flat_below_floor() {
        let (log_x, d1, d2, d3) =
            SurvivalLocationScaleFamily::clamped_logwith_derivatives(-0.25, 1e-12);
        assert!((log_x - 1e-12_f64.ln()).abs() <= 1e-15);
        assert_eq!(d1, 0.0);
        assert_eq!(d2, 0.0);
        assert_eq!(d3, 0.0);
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
    fn joint_exact_newton_log_sigma_block_should_be_flat_on_safe_exp_plateau() {
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
        assert_eq!(
            score_fd, 0.0,
            "safe_exp is constant for eta_ls > 700, so the coded survival log-likelihood is locally flat in the log-sigma coefficient on that plateau"
        );
        assert_eq!(
            info_fd, 0.0,
            "safe_exp is constant for eta_ls > 700, so the coded survival log-likelihood has zero second derivative in the log-sigma coefficient on that plateau"
        );
        assert!(
            (analytic_score - score_fd).abs() < 1e-30,
            "the exact-newton survival log-sigma score should be the derivative of the coded plateau log-likelihood at beta_log_sigma={beta_log_sigma0}; got {} vs {}",
            analytic_score,
            score_fd
        );
        assert!(
            (analytic_info - info_fd).abs() < 1e-20,
            "the exact-newton survival log-sigma information should be the negative second derivative of the coded plateau log-likelihood at beta_log_sigma={beta_log_sigma0}; got {} vs {}",
            analytic_info,
            info_fd
        );
    }

    #[test]
    fn survival_q_chain_derivatives_vanish_on_safe_exp_upper_plateau() {
        let eta_t = 2.0;
        let eta_ls = 701.0_f64;
        let (sigma, ds, d2s, d3s) = exp_sigma_derivs_up_to_third_scalar(eta_ls);
        let q = |ls: f64| -eta_t / exp_sigma_derivs_up_to_third_scalar(ls).0.max(1e-12);
        let h = 1e-6;
        let q_left = q(eta_ls - h);
        let q_mid = q(eta_ls);
        let q_right = q(eta_ls + h);
        assert_eq!(
            q_left, q_mid,
            "safe_exp is constant beyond the upper clamp, so q should be locally constant in eta_ls on that plateau"
        );
        assert_eq!(
            q_right, q_mid,
            "safe_exp is constant beyond the upper clamp, so q should be locally constant in eta_ls on that plateau"
        );

        let (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) =
            q_chain_derivs_scalar(eta_t, sigma, ds, d2s, d3s);
        assert_eq!(q_t, -1.0 / sigma.max(1e-12));
        assert!(
            q_ls == 0.0,
            "q = -eta_t / max(safe_exp(eta_ls), 1e-12) is constant in eta_ls on the upper safe_exp plateau, so dq/deta_ls must be 0; got {q_ls}"
        );
        assert!(
            q_tl == 0.0,
            "q_t is constant in eta_ls on the upper safe_exp plateau, so d2q/(deta_t deta_ls) must be 0; got {q_tl}"
        );
        assert!(
            q_ll == 0.0,
            "q is constant in eta_ls on the upper safe_exp plateau, so d2q/deta_ls2 must be 0; got {q_ll}"
        );
        assert!(
            q_tl_ls == 0.0,
            "q_t is constant in eta_ls on the upper safe_exp plateau, so d3q/(deta_t deta_ls2) must be 0; got {q_tl_ls}"
        );
        assert!(
            q_ll_ls == 0.0,
            "q is constant in eta_ls on the upper safe_exp plateau, so d3q/deta_ls3 must be 0; got {q_ll_ls}"
        );
    }

    #[test]
    fn survival_exact_log_sigma_dh_should_match_zero_third_derivative_on_plateau() {
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
        assert_eq!(
            fd3, 0.0,
            "safe_exp is constant for eta_ls > 700, so the coded survival log-likelihood has zero third derivative in beta_log_sigma on that plateau"
        );
        assert!(
            (analytic[[0, 0]] + fd3).abs() < 1e-20,
            "the exact-newton survival log-sigma dH entry should equal the negative third derivative of the coded plateau log-likelihood at beta_log_sigma={beta_log_sigma0}; got analytic {} vs expected {}",
            analytic[[0, 0]],
            -fd3
        );
    }

    #[test]
    fn survival_joint_exact_log_sigma_dh_should_match_zero_third_derivative_on_plateau() {
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
        assert_eq!(
            fd3, 0.0,
            "safe_exp is constant for eta_ls > 700, so the coded survival log-likelihood has zero third derivative in beta_log_sigma on that plateau"
        );
        assert!(
            (analytic[[2, 2]] + fd3).abs() < 1e-20,
            "the exact joint survival dH log-sigma/log-sigma entry should equal the negative third derivative of the coded plateau log-likelihood at beta_log_sigma={beta_log_sigma0}; got analytic {} vs expected {}",
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
            let (h0, h1, d_raw, eta_t_exit, eta_ls_exit, eta_t_entry, eta_ls_entry, etaw) =
                family.validate_joint_states(&states).expect("joint states");
            // For time-invariant blocks, eta_ls_entry == eta_ls_exit.
            let (sigma, _, _, _) = exp_sigma_derivs_up_to_third(eta_ls_exit.view());
            let (sigma_entry, _, _, _) = exp_sigma_derivs_up_to_third(eta_ls_entry.view());

            for i in 0..family.n {
                let state = family.row_predictor_state(
                    h0[i],
                    h1[i],
                    d_raw[i],
                    eta_t_entry[i],
                    eta_t_exit[i],
                    sigma_entry[i],
                    sigma[i],
                    etaw.map(|w| w[i]),
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
                    (ell_q + ell_h0 + ell_h1).abs() <= 1e-10,
                    "survival {label} row {i} violated ell_q = -ell_h0 - ell_h1: q={} h0={} h1={}",
                    ell_q,
                    ell_h0,
                    ell_h1
                );
                assert!(
                    (ell_qq + ell_h0q + ell_h1q).abs() <= 1e-10,
                    "survival {label} row {i} violated ell_qq = -ell_h0q - ell_h1q: qq={} h0q={} h1q={}",
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
            x_threshold: DesignMatrix::Dense(array![[1.0, -0.2]]),
            eta_threshold_offset: array![0.0],
            x_log_sigma: DesignMatrix::Dense(array![[1.0, 0.3]]),
            eta_log_sigma_offset: array![0.0],
            x_link_wiggle: None,
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
            x_threshold: DesignMatrix::Dense(array![[1.0, -0.2]]),
            eta_threshold_offset: array![0.7],
            x_log_sigma: DesignMatrix::Dense(array![[1.0, 0.3]]),
            eta_log_sigma_offset: array![0.4],
            x_link_wiggle: None,
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        };
        let pred = predict_survival_location_scale(&input, &fit).expect("predict");

        let eta_t = array![1.0, -0.2].dot(&fit.beta_threshold()) + input.eta_threshold_offset[0];
        let eta_ls = array![1.0, 0.3].dot(&fit.beta_log_sigma()) + input.eta_log_sigma_offset[0];
        let sigma = exp_sigma_derivs_up_to_third_scalar(eta_ls).0;
        let h = array![1.0, 0.5].dot(&fit.beta_time()) + input.eta_time_offset_exit[0];
        let expected_eta = -h - eta_t / sigma.max(1e-12);
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
        let xwiggle_dense = array![[1.0, 0.1], [0.0, -0.2]];
        let dense_input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5], [1.0, -0.3]],
            eta_time_offset_exit: array![0.2, -0.1],
            x_threshold: DesignMatrix::Dense(x_threshold_dense.clone()),
            eta_threshold_offset: array![0.7, -0.2],
            x_log_sigma: DesignMatrix::Dense(x_log_sigma_dense.clone()),
            eta_log_sigma_offset: array![0.4, 0.1],
            x_link_wiggle: Some(DesignMatrix::Dense(xwiggle_dense.clone())),
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
    fn gaussian_posterior_mean_reduction_matches_3d_ghq_small_case() {
        let input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.1],
            x_threshold: DesignMatrix::Dense(array![[1.0, 0.25]]),
            eta_threshold_offset: array![0.0],
            x_log_sigma: DesignMatrix::Dense(array![[1.0, -0.15]]),
            eta_log_sigma_offset: array![0.0],
            x_link_wiggle: None,
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
        let reduced = predict_survival_location_scale_posterior_mean(&input, &fit, &covariance)
            .expect("reduced posterior mean");

        let mu_h = input.x_time_exit.row(0).dot(&fit.beta_time) + input.eta_time_offset_exit[0];
        let x_t = input.x_threshold.to_dense_arc();
        let x_ls = input.x_log_sigma.to_dense_arc();
        let mu_t = x_t.row(0).dot(&fit.beta_threshold);
        let mu_ls = x_ls.row(0).dot(&fit.beta_log_sigma);
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
                let sigma = exp_sigma_derivs_up_to_third_scalar(ls).0;
                (1.0 - normal_cdf(-h - t / sigma.max(1e-12))).clamp(0.0, 1.0)
            },
        );
        assert!((reduced.survival_prob[0] - ghq).abs() <= 2e-4);
    }

    #[test]
    fn sparse_posterior_mean_matches_dense() {
        let x_threshold_dense = array![[1.0, 0.25], [0.0, -0.1]];
        let x_log_sigma_dense = array![[1.0, -0.15], [0.0, 0.2]];
        let dense_input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5], [1.0, -0.4]],
            eta_time_offset_exit: array![0.1, -0.2],
            x_threshold: DesignMatrix::Dense(x_threshold_dense.clone()),
            eta_threshold_offset: array![0.0, 0.05],
            x_log_sigma: DesignMatrix::Dense(x_log_sigma_dense.clone()),
            eta_log_sigma_offset: array![0.0, -0.03],
            x_link_wiggle: None,
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
    fn wiggle_posterior_mean_reduction_matches_4d_ghq_small_case() {
        let input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.2],
            x_threshold: DesignMatrix::Dense(array![[1.0, -0.2]]),
            eta_threshold_offset: array![0.0],
            x_log_sigma: DesignMatrix::Dense(array![[1.0, 0.3]]),
            eta_log_sigma_offset: array![0.0],
            x_link_wiggle: Some(DesignMatrix::Dense(array![[1.0, 0.1]])),
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
        };
        let fit = test_survival_fit(
            array![0.4, -0.1],
            array![0.2, 0.3],
            array![-0.5, 0.1],
            Some(array![0.05, -0.02]),
        );
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
        let reduced = predict_survival_location_scale_posterior_mean(&input, &fit, &covariance)
            .expect("wiggle posterior mean");

        let x_t = input.x_threshold.to_dense_arc();
        let x_ls = input.x_log_sigma.to_dense_arc();
        let xw = input
            .x_link_wiggle
            .as_ref()
            .expect("wiggle design")
            .to_dense_arc();
        let mu_h = input.x_time_exit.row(0).dot(&fit.beta_time) + input.eta_time_offset_exit[0];
        let mu_t = x_t.row(0).dot(&fit.beta_threshold);
        let mu_ls = x_ls.row(0).dot(&fit.beta_log_sigma);
        let muw = xw
            .row(0)
            .dot(fit.beta_link_wiggle.as_ref().expect("wiggle beta"));
        let cov_hh = covariance.slice(s![0..2, 0..2]).to_owned();
        let cov_tt = covariance.slice(s![2..4, 2..4]).to_owned();
        let cov_ll = covariance.slice(s![4..6, 4..6]).to_owned();
        let covww = covariance.slice(s![6..8, 6..8]).to_owned();
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
        let varw = xw.row(0).dot(&covww.dot(&xw.row(0).to_owned()));
        let cov_ht_i = input
            .x_time_exit
            .row(0)
            .dot(&cov_ht.dot(&x_t.row(0).to_owned()));
        let cov_hl_i = input
            .x_time_exit
            .row(0)
            .dot(&cov_hl.dot(&x_ls.row(0).to_owned()));
        let cov_hw_i = input
            .x_time_exit
            .row(0)
            .dot(&cov_hw.dot(&xw.row(0).to_owned()));
        let cov_tl_i = x_t.row(0).dot(&cov_tl.dot(&x_ls.row(0).to_owned()));
        let cov_tw_i = x_t.row(0).dot(&cov_tw.dot(&xw.row(0).to_owned()));
        let cov_lw_i = x_ls.row(0).dot(&cov_lw.dot(&xw.row(0).to_owned()));
        let quadctx = crate::quadrature::QuadratureContext::new();
        let ghq = crate::quadrature::normal_expectation_nd_adaptive::<4, _>(
            &quadctx,
            [mu_h, mu_t, mu_ls, muw],
            [
                [var_h, cov_ht_i, cov_hl_i, cov_hw_i],
                [cov_ht_i, var_t, cov_tl_i, cov_tw_i],
                [cov_hl_i, cov_tl_i, var_ls, cov_lw_i],
                [cov_hw_i, cov_tw_i, cov_lw_i, varw],
            ],
            11,
            |x| {
                let sigma = exp_sigma_derivs_up_to_third_scalar(x[2]).0.max(1e-12);
                (1.0 - normal_cdf(-x[0] - x[1] / sigma + x[3])).clamp(0.0, 1.0)
            },
        );
        assert!((reduced.survival_prob[0] - ghq).abs() <= 2e-4);
    }

    #[test]
    fn predict_rejects_stateless_beta_logistic_inverse_link() {
        let fit = test_survival_fit(array![0.4, -0.1], array![0.2, 0.3], array![-0.5, 0.1], None);
        let input = SurvivalLocationScalePredictInput {
            x_time_exit: array![[1.0, 0.5]],
            eta_time_offset_exit: array![0.2],
            x_threshold: DesignMatrix::Dense(array![[1.0, -0.2]]),
            eta_threshold_offset: array![0.0],
            x_log_sigma: DesignMatrix::Dense(array![[1.0, 0.3]]),
            eta_log_sigma_offset: array![0.0],
            x_link_wiggle: None,
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
            x_threshold: DesignMatrix::Dense(array![[1.0, -0.2]]),
            eta_threshold_offset: array![0.0],
            x_log_sigma: DesignMatrix::Dense(array![[1.0, 0.3]]),
            eta_log_sigma_offset: array![0.0],
            x_link_wiggle: None,
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

    /// Full-path reproducer: runs fit_survival_location_scale with zero
    /// derivative offsets, mimicking the heart_failure_survival scenario
    /// where the linear baseline produces d_raw=0 at initialization.
    #[test]
    fn heart_failure_full_fit_zero_deriv_offset() {
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

        // Zero derivative offsets (linear baseline → (0,0)).
        let derivative_offset_exit = Array1::<f64>::zeros(n);
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
            derivative_guard: 0.0,
            derivative_softness: 1e-6,
            time_anchor: None,
            max_iter: 400,
            tol: 1e-6,
            time_block: TimeBlockInput {
                design_entry,
                design_exit,
                design_derivative_exit,
                offset_entry,
                offset_exit,
                derivative_offset_exit,
                penalties: vec![penalty.clone()],
                initial_log_lambdas: Some(array![0.0]),
                initial_beta: None,
            },
            threshold_block: CovariateBlockKind::Static(CovariateBlockInput {
                design: DesignMatrix::Dense(Array2::ones((n, 1))),
                offset: Array1::zeros(n),
                penalties: Vec::new(),
                initial_log_lambdas: None,
                initial_beta: None,
            }),
            log_sigma_block: CovariateBlockKind::Static(CovariateBlockInput {
                design: DesignMatrix::Dense(Array2::ones((n, 1))),
                offset: Array1::zeros(n),
                penalties: Vec::new(),
                initial_log_lambdas: None,
                initial_beta: None,
            }),
            linkwiggle_block: None,
        };

        match fit_survival_location_scale(spec) {
            Ok(result) => {
                eprintln!(
                    "fit succeeded: log_likelihood={:.6e}",
                    result.log_likelihood
                );
                eprintln!("beta_time: {:?}", result.beta_time);
                eprintln!("beta_threshold: {:?}", result.beta_threshold);
                eprintln!("beta_log_sigma: {:?}", result.beta_log_sigma);
            }
            Err(e) => {
                panic!("fit_survival_location_scale failed: {e}");
            }
        }
    }

    /// Reproducer for heart_failure_survival NaN crash.
    ///
    /// Mimics the real scenario: zero derivative offsets (linear baseline),
    /// zero initial beta, probit link (Gaussian residual distribution),
    /// multiple event/non-event rows.
    #[test]
    fn heart_failure_zero_offset_nan_small() {
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
        // Zero derivative offsets (linear baseline)
        let offset_deriv = Array1::<f64>::zeros(n);

        let family = SurvivalLocationScaleFamily {
            n,
            y: array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            w: Array1::ones(n),
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
            derivative_guard: 0.0,
            derivative_softness: 1e-6,
            x_time_entry: x_entry,
            x_time_exit: x_exit.clone(),
            x_time_deriv: x_deriv.clone(),
            offset_time_deriv: offset_deriv.clone(),
            x_threshold: DesignMatrix::Dense(Array2::ones((n, 1))),
            x_threshold_entry: None,
            x_log_sigma: DesignMatrix::Dense(Array2::ones((n, 1))),
            x_log_sigma_entry: None,
            x_link_wiggle: None,
        };

        // Build initial states with beta=0 → d_raw = 0 for all rows.
        let states = vec![
            ParameterBlockState {
                beta: Array1::zeros(2),
                eta: Array1::zeros(3 * n), // [h0; h1; d_raw] all zero
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

        // Step 1: Verify initial evaluate succeeds with d_raw=0.
        let eval = family
            .evaluate(&states)
            .expect("initial evaluate with d_raw=0 should succeed");
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
    fn evaluate_survival_location_scale_soft_clamps_non_finite_d_eta_dt() {
        let n = 2;
        let family = SurvivalLocationScaleFamily {
            n,
            y: array![1.0, 0.0],
            w: Array1::ones(n),
            inverse_link: residual_distribution_inverse_link(ResidualDistribution::Gaussian),
            derivative_guard: 0.0,
            derivative_softness: 1e-6,
            x_time_entry: Array2::zeros((n, 1)),
            x_time_exit: Array2::ones((n, 1)),
            x_time_deriv: Array2::ones((n, 1)),
            offset_time_deriv: Array1::zeros(n),
            x_threshold: DesignMatrix::Dense(Array2::ones((n, 1))),
            x_threshold_entry: None,
            x_log_sigma: DesignMatrix::Dense(Array2::ones((n, 1))),
            x_log_sigma_entry: None,
            x_link_wiggle: None,
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

        let eval = family
            .evaluate(&states)
            .expect("non-finite d_eta/dt should be soft-clamped, not fatal");
        assert!(eval.log_likelihood.is_finite());
    }

    #[test]
    fn q_chain_derivatives_vanish_when_sigma_floor_is_active() {
        let eta_t = 2.0;
        let eta_ls = -30.0;
        let (sigma, ds, d2s, d3s) = exp_sigma_derivs_up_to_third_scalar(eta_ls);
        assert!(sigma < 1e-12, "test requires the sigma floor branch");

        let q = |ls: f64| -eta_t / exp_sigma_derivs_up_to_third_scalar(ls).0.max(1e-12);
        let h = 1e-6;
        let q_left = q(eta_ls - h);
        let q_mid = q(eta_ls);
        let q_right = q(eta_ls + h);
        assert_eq!(
            q_left, q_mid,
            "the floor branch should make q locally constant in eta_ls"
        );
        assert_eq!(
            q_right, q_mid,
            "the floor branch should make q locally constant in eta_ls"
        );

        let (q_t, q_ls, q_tl, q_ll, q_tl_ls, q_ll_ls) =
            q_chain_derivs_scalar(eta_t, sigma, ds, d2s, d3s);

        assert_eq!(q_t, -1.0 / 1e-12);
        assert!(
            q_ls == 0.0,
            "q = -eta_t / max(sigma, 1e-12) is constant in eta_ls on the active floor branch, so dq/deta_ls must be 0; got {q_ls}"
        );
        assert!(
            q_tl == 0.0,
            "q_t = -1 / max(sigma, 1e-12) is constant in eta_ls on the active floor branch, so d2q/(deta_t deta_ls) must be 0; got {q_tl}"
        );
        assert!(
            q_ll == 0.0,
            "q is locally constant in eta_ls on the active floor branch, so d2q/deta_ls2 must be 0; got {q_ll}"
        );
        assert!(
            q_tl_ls == 0.0,
            "q_t is locally constant in eta_ls on the active floor branch, so d3q/(deta_t deta_ls2) must be 0; got {q_tl_ls}"
        );
        assert!(
            q_ll_ls == 0.0,
            "q is locally constant in eta_ls on the active floor branch, so d3q/deta_ls3 must be 0; got {q_ll_ls}"
        );
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
    fn probit_survival_helper_loses_upper_tail_probability() {
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
    fn cloglog_survival_helper_changes_the_negative_tail_function() {
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
}
