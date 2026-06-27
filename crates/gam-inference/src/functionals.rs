use gam_linalg::faer_ndarray::FaerCholesky;
use gam_solve::model_types::EstimationError;
use gam_solve::sensitivity::FitSensitivity;
use faer::Side;
use ndarray::{Array1, ArrayView1, ArrayView2};

#[derive(Clone, Debug)]
pub struct FunctionalEstimate {
    pub theta_plugin: f64,
    pub theta_onestep: f64,
    pub se: f64,
    pub penalty_bias: f64,
    pub n_effective: usize,
}

pub struct GaussianIdentityAverageDerivativeInput<'a> {
    pub design: ArrayView2<'a, f64>,
    pub derivative_design: ArrayView2<'a, f64>,
    pub y: ArrayView1<'a, f64>,
    pub mu: ArrayView1<'a, f64>,
    pub beta: ArrayView1<'a, f64>,
    /// Scaled penalty matrix `λS` actually applied to this fit. The one-step
    /// correction is built against the penalized Hessian `XᵀX + λS` — the
    /// information of the estimator that produced `beta` — so this matrix
    /// must accompany the `penalty_beta = λSβ̂` gradient.
    pub penalty: ArrayView2<'a, f64>,
    pub penalty_beta: ArrayView1<'a, f64>,
}

pub fn average_derivative_gaussian_identity(
    input: &GaussianIdentityAverageDerivativeInput<'_>,
) -> Result<FunctionalEstimate, EstimationError> {
    validate_average_derivative_input(input)?;

    // Penalized Hessian H = XᵀX + λS — the information of the *penalized*
    // estimator that produced `beta`. The one-step correction is the efficient
    // influence function of the average-derivative functional evaluated at this
    // estimator, so the Riesz representer must solve against H, not the raw XᵀX
    // information (which would unwind the penalty entirely and reproduce the
    // high-variance OLS plug-in instead of debiasing it).
    let mut information = input.design.t().dot(&input.design);
    information += &input.penalty;
    let h_factor = information.cholesky(Side::Lower).map_err(|err| {
        EstimationError::InvalidInput(format!(
            "average-derivative functional requires SPD penalized Hessian: {err}"
        ))
    })?;
    let sensitivity = FitSensitivity::from_faer_cholesky(&h_factor, input.beta.len());
    average_derivative_gaussian_identity_with_sensitivity(input, &sensitivity)
}

pub fn average_derivative_gaussian_identity_with_sensitivity(
    input: &GaussianIdentityAverageDerivativeInput<'_>,
    sensitivity: &FitSensitivity<'_>,
) -> Result<FunctionalEstimate, EstimationError> {
    validate_average_derivative_input(input)?;
    let p = input.beta.len();
    if sensitivity.dim() != p {
        gam_problem::bail_invalid_estim!(
            "average-derivative functional sensitivity dimension {} must equal beta length {p}",
            sensitivity.dim()
        );
    }

    let n = input.design.nrows();
    let mut a_theta = Array1::<f64>::zeros(p);
    for row in input.derivative_design.rows() {
        for j in 0..p {
            a_theta[j] += row[j] / n as f64;
        }
    }

    let theta_plugin = a_theta.dot(&input.beta);
    let riesz = sensitivity.apply(&a_theta);
    if riesz.iter().any(|value| !value.is_finite()) {
        gam_problem::bail_invalid_estim!(
            "average-derivative functional H^-1 gradient solve produced non-finite values"
        );
    }

    let penalty_bias = riesz.dot(&input.penalty_beta);
    let mut influence_sq_sum = 0.0_f64;
    for i in 0..n {
        let residual = input.y[i] - input.mu[i];
        let row_score_projection = input.design.row(i).dot(&riesz) * residual;
        // One-step (von Mises) debiasing of the oversmoothed plugin theta=a'beta.
        // The penalized score residual is X'(y - mu) = λS β̂, and the Riesz solve
        // above is a'·H⁻¹ against the penalized Hessian H = X'X + λS. The
        // resulting correction a'·H⁻¹·(λS β̂) removes the leading smoothing bias
        // of the plug-in without unwinding the penalty back to the high-variance
        // OLS estimate, so the per-observation influence below shares this H⁻¹a.
        let phi_i = (n as f64) * row_score_projection;
        influence_sq_sum += phi_i * phi_i;
    }

    let theta_onestep = theta_plugin + penalty_bias;
    let se = influence_sq_sum.sqrt() / n as f64;
    if !theta_plugin.is_finite()
        || !theta_onestep.is_finite()
        || !se.is_finite()
        || !penalty_bias.is_finite()
    {
        gam_problem::bail_invalid_estim!("average-derivative functional produced non-finite estimate");
    }

    Ok(FunctionalEstimate {
        theta_plugin,
        theta_onestep,
        se,
        penalty_bias,
        n_effective: n,
    })
}

pub fn penalty_times_beta(penalty: ArrayView2<'_, f64>, beta: ArrayView1<'_, f64>) -> Array1<f64> {
    penalty.dot(&beta)
}

fn validate_average_derivative_input(
    input: &GaussianIdentityAverageDerivativeInput<'_>,
) -> Result<(), EstimationError> {
    let n = input.design.nrows();
    let p = input.design.ncols();
    if n == 0 || p == 0 {
        gam_problem::bail_invalid_estim!(
            "average-derivative functional requires non-empty design, got {n}x{p}"
        );
    }
    if input.derivative_design.nrows() != n || input.derivative_design.ncols() != p {
        gam_problem::bail_invalid_estim!(
            "average-derivative derivative design shape {}x{} must match design {n}x{p}",
            input.derivative_design.nrows(),
            input.derivative_design.ncols()
        );
    }
    if input.y.len() != n || input.mu.len() != n {
        gam_problem::bail_invalid_estim!(
            "average-derivative y/mu lengths must equal design rows {n}, got y={} mu={}",
            input.y.len(),
            input.mu.len()
        );
    }
    if input.beta.len() != p || input.penalty_beta.len() != p {
        gam_problem::bail_invalid_estim!(
            "average-derivative beta/penalty_beta lengths must equal design columns {p}, got beta={} penalty_beta={}",
            input.beta.len(),
            input.penalty_beta.len()
        );
    }
    if input.penalty.nrows() != p || input.penalty.ncols() != p {
        gam_problem::bail_invalid_estim!(
            "average-derivative penalty matrix shape {}x{} must be square in design columns {p}",
            input.penalty.nrows(),
            input.penalty.ncols()
        );
    }
    if input.design.iter().any(|value| !value.is_finite())
        || input
            .derivative_design
            .iter()
            .any(|value| !value.is_finite())
        || input.y.iter().any(|value| !value.is_finite())
        || input.mu.iter().any(|value| !value.is_finite())
        || input.beta.iter().any(|value| !value.is_finite())
        || input.penalty.iter().any(|value| !value.is_finite())
        || input.penalty_beta.iter().any(|value| !value.is_finite())
    {
        gam_problem::bail_invalid_estim!(
            "average-derivative functional requires finite design, derivative design, response, fit, and penalty-gradient inputs"
        );
    }
    Ok(())
}
