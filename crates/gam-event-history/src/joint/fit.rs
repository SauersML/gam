//! The K ≥ 1 joint latent-signature fit (#2961): function strengths by the
//! joint Laplace evidence of `strength_fit.rs`, over the normalized final
//! function priors, and a fit object only from a converged strength optimum.
//!
//! The coefficient law served with the fit is the Gaussian N(θ̂, S⁻¹) of the
//! joint Laplace step. Forecasts average final probabilities over it; no
//! coefficient point is inserted into a forecast.
use super::function_prior::JointFunctionPriors;
use super::law::invalid;
use super::law::numerical::Running;
use super::strength_fit::{
    FaceFit, LaplaceSystem, PriorCurvature, PriorRounding, StrengthPrior, StrengthSearch, ZeroEffectLimit,
    optimize_strengths,
};
use crate::EventHistoryError;
use gam_math::nested_dual::JetField;
use ndarray::Array2;
use serde::{Deserialize, Serialize};

/// A converged K ≥ 1 fit: the coefficient posterior at the empirical-Bayes
/// strengths, and the evidence and certificate those strengths came from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SignatureFit {
    /// θ̂, the mean of the joint Laplace coefficient law, zero on pinned blocks.
    pub coefficients: Vec<f64>,
    /// S⁻¹, its covariance, zero on pinned blocks' rows and columns.
    pub coefficient_covariance: Array2<f64>,
    /// Strengths at their zero-effect limit, each with the status its boundary
    /// score decided at this fit. Empty for an interior optimum of the full model.
    pub limits: Vec<ZeroEffectLimit>,
    /// Indices of the strengths this fit optimizes, and their log strengths.
    pub strengths: Vec<usize>,
    pub log_strengths: Vec<f64>,
    /// The joint Laplace log evidence at the returned strengths.
    pub log_evidence: f64,
    /// Its rounding floor, the error structure comparisons carry.
    pub log_evidence_band: f64,
    /// Recomputed strength score at the returned strengths.
    pub strength_score: Vec<f64>,
    /// Remaining Newton step of θ̂, in posterior standard deviations.
    pub newton_decrement: f64,
    pub iterations: usize,
}

/// The normalized function priors in the dense form the evidence needs. The f64
/// instance serves the search; `checked` holds the same measures built once over
/// Running, read only at convergence and classification checks (brief 19:05).
/// Every block is assembled from the priors' own analytic products, one
/// coefficient direction at a time.
struct FunctionPriorStrengths<'p, 'm> {
    priors: &'p JointFunctionPriors<'m>,
    checked: &'p JointFunctionPriors<'m, Running>,
}

fn unit<S: JetField>(width: usize, index: usize, zero: &S) -> Vec<S> {
    (0..width)
        .map(|k| zero.with_value(f64::from(u8::from(k == index))))
        .collect()
}

fn columns<S: JetField>(
    width: usize,
    zero: &S,
    mut product: impl FnMut(&[S]) -> Result<Vec<S>, EventHistoryError>,
) -> Result<Array2<S>, EventHistoryError> {
    let mut matrix = Array2::from_elem((width, width), zero.with_value(0.0));
    for k in 0..width {
        let column = product(&unit(width, k, zero))?;
        if column.len() != width {
            return Err(invalid(
                "function prior product has a width inconsistent with the coefficients",
            ));
        }
        for (i, value) in column.into_iter().enumerate() {
            matrix[[i, k]] = value;
        }
    }
    Ok(matrix)
}

/// Rows ∂θ∂ρ_j log π, one coefficient-strength product per strength.
fn mixed_rows<S: JetField>(
    coefficients: usize,
    strengths: usize,
    zero: &S,
    mut product: impl FnMut(&[S]) -> Result<Vec<S>, EventHistoryError>,
) -> Result<Array2<S>, EventHistoryError> {
    let mut matrix = Array2::from_elem((strengths, coefficients), zero.with_value(0.0));
    for j in 0..strengths {
        let row = product(&unit(strengths, j, zero))?;
        if row.len() != coefficients {
            return Err(invalid(
                "function prior product has a width inconsistent with the coefficients",
            ));
        }
        for (q, value) in row.into_iter().enumerate() {
            matrix[[j, q]] = value;
        }
    }
    Ok(matrix)
}

/// Inputs taken as exact operands of the checked route.
fn exact(values: &[f64]) -> Vec<Running> {
    values.iter().map(|&v| Running::exact(v)).collect()
}

fn errors(values: &[Running]) -> Vec<f64> {
    values.iter().map(Running::rounding).collect()
}

impl StrengthPrior for FunctionPriorStrengths<'_, '_> {
    fn strengths(&self) -> usize {
        self.priors.penalties().len()
    }

    fn evaluate(&self, theta: &[f64], rho: &[f64]) -> Result<PriorCurvature, EventHistoryError> {
        let evaluation = self.priors.evaluate(theta, rho)?;
        let p = theta.len();
        let h = self.strengths();
        let strength_curvature = (0..h)
            .map(|j| columns(p, &0.0, |v| evaluation.strength_hessian_product(j, v)))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PriorCurvature {
            log_density: *evaluation.log_density(),
            gradient: evaluation.gradient().to_vec(),
            strength_gradient: evaluation.log_strength_gradient().to_vec(),
            strength_second: evaluation.log_strength_second_derivative().to_vec(),
            negative_hessian: columns(p, &0.0, |v| evaluation.negative_hessian_product(v))?,
            strength_curvature,
            mixed: mixed_rows(p, h, &0.0, |u| evaluation.coefficient_strength_product(u))?,
        })
    }

    fn rounding(&self, theta: &[f64], rho: &[f64]) -> Result<PriorRounding, EventHistoryError> {
        let evaluation = self.checked.evaluate(&exact(theta), &exact(rho))?;
        let zero = Running::exact(0.0);
        let p = theta.len();
        let h = self.strengths();
        let strength_curvature = (0..h)
            .map(|j| {
                columns(p, &zero, |v| evaluation.strength_hessian_product(j, v))
                    .map(|m| m.map(Running::rounding))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PriorRounding {
            log_density: evaluation.log_density().rounding(),
            gradient: errors(evaluation.gradient()),
            strength_gradient: errors(evaluation.log_strength_gradient()),
            negative_hessian: columns(p, &zero, |v| evaluation.negative_hessian_product(v))?
                .map(Running::rounding),
            strength_curvature,
            mixed: mixed_rows(p, h, &zero, |u| evaluation.coefficient_strength_product(u))?
                .map(Running::rounding),
        })
    }

    fn curvature_derivative(
        &self,
        theta: &[f64],
        rho: &[f64],
        direction: &[f64],
        direction_error: &[f64],
    ) -> Result<(Array2<f64>, Array2<f64>), EventHistoryError> {
        // The direction carries its errors as running bounds, so |D H_π| acts on
        // them through Running's own rules and no cancellation hides them.
        let evaluation = self.checked.evaluate(&exact(theta), &exact(rho))?;
        let carried: Vec<Running> = direction
            .iter()
            .zip(direction_error)
            .map(|(&value, &error)| Running {
                value,
                mu: error / f64::EPSILON,
            })
            .collect();
        let derivative = columns(theta.len(), &Running::exact(0.0), |v| {
            evaluation.third_derivative_product(&carried, v)
        })?;
        Ok((derivative.map(|v| v.value), derivative.map(Running::rounding)))
    }
}

/// Fit the strengths of `priors` for the cohort's joint Laplace system.
///
/// Strengths start at zero log strength: the priors measure every function on
/// its normalized cohort function measure, so unit strength is a function of
/// unit size under that measure. No tolerance is supplied: the coefficient mode
/// and the strength score are certified at their own rounding resolution
/// (`strength_fit.rs`).
///
/// S and the prior curvature are dense p×p here. The principal route keeps S
/// dense, because its assembly dominates at O(n·T·K·p²). The prior curvature
/// becomes block products over each penalty's disjoint support. A structured
/// log det S is taken only if the A6 receipts measure dense S failing.
///
/// `checked` is the same priors built once over Running
/// (`function_priors_over::<Running>`), read only at checks.
pub(super) fn fit_signatures<'m, Sys: LaplaceSystem>(
    system: &mut Sys,
    priors: &JointFunctionPriors<'m>,
    checked: &JointFunctionPriors<'m, Running>,
    initial_coefficients: &[f64],
) -> Result<SignatureSearch, EventHistoryError> {
    let strengths = FunctionPriorStrengths { priors, checked };
    let initial = vec![0.0; strengths.strengths()];
    let width = system.coefficients();
    let search = optimize_strengths(system, &strengths, initial_coefficients, &initial)?;
    Ok(signature_search(search, width))
}

/// A lattice fit in the parent layout: zeros on pinned coordinates and on their
/// covariance rows and columns.
fn signature_fit(fit: FaceFit, width: usize) -> SignatureFit {
    let covariance = fit.mode.covariance();
    let mut coefficients = vec![0.0; width];
    let mut coefficient_covariance = Array2::zeros((width, width));
    for (a, &i) in fit.kept.iter().enumerate() {
        coefficients[i] = fit.mode.coefficients[a];
        for (b, &k) in fit.kept.iter().enumerate() {
            coefficient_covariance[[i, k]] = covariance[[a, b]];
        }
    }
    SignatureFit {
        coefficients,
        coefficient_covariance,
        limits: fit.limits,
        strengths: fit.strengths,
        log_strengths: fit.log_strengths,
        log_evidence: fit.evidence.log_evidence,
        log_evidence_band: fit.evidence.evidence_band,
        strength_score: fit.evidence.gradient,
        newton_decrement: fit.mode.certificate.lambda_sq.sqrt(),
        iterations: fit.iterations,
    }
}

fn signature_search(search: StrengthSearch, width: usize) -> SignatureSearch {
    match search {
        StrengthSearch::Fitted(fit) => SignatureSearch::Fitted(signature_fit(fit, width)),
        StrengthSearch::EvidenceTie { interior, limit } => SignatureSearch::EvidenceTie {
            interior: signature_fit(interior, width),
            limit: Box::new(signature_search(*limit, width)),
        },
    }
}

/// What fitting one structure's strengths returns: the certified fit of the model
/// with the larger evidence, where strengths at their zero-effect limit carry the
/// status their boundary score decided at that fit (a first-order tie is reported,
/// never hidden), or both fits when their evidence values tie (brief 20:50). A
/// strength limit is decided inside the fit, never by the structure comparison.
pub(super) enum SignatureSearch {
    Fitted(SignatureFit),
    EvidenceTie {
        interior: SignatureFit,
        limit: Box<SignatureSearch>,
    },
}
