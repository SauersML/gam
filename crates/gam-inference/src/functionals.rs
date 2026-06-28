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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    /// Intercept-only input: X = I_n (identity), derivative_design = I_n,
    /// penalty = 0, penalty_beta = 0.  For y == mu (perfect fit) the SE must
    /// be zero and the one-step estimate must equal the plug-in.
    #[test]
    fn zero_penalty_perfect_fit_se_is_zero_and_onestep_equals_plugin() {
        let n = 4_usize;
        let x = Array2::<f64>::eye(n);
        let beta = array![1.0_f64, 2.0, 3.0, 4.0];
        let mu = x.dot(&beta); // mu = X @ beta (perfect plug-in fit)
        let y = mu.clone(); // y == mu → zero residuals
        let penalty = Array2::<f64>::zeros((n, n));
        let penalty_beta = Array1::<f64>::zeros(n);
        let input = GaussianIdentityAverageDerivativeInput {
            design: x.view(),
            derivative_design: x.view(), // derivative_design = X
            y: y.view(),
            mu: mu.view(),
            beta: beta.view(),
            penalty: penalty.view(),
            penalty_beta: penalty_beta.view(),
        };
        let est = average_derivative_gaussian_identity(&input).expect("functional estimate");
        // theta_plugin = a'@beta, a = (1/n, …, 1/n)^T (row mean of identity = e_i/n summed)
        let expected_plugin = beta.mean().unwrap();
        assert!(
            (est.theta_plugin - expected_plugin).abs() < 1e-12,
            "theta_plugin: got {:.6e}, expected {:.6e}",
            est.theta_plugin,
            expected_plugin
        );
        // No penalty → penalty_bias = 0 → one-step == plugin
        assert!(
            est.penalty_bias.abs() < 1e-12,
            "penalty_bias must be zero: got {:.6e}",
            est.penalty_bias
        );
        assert!(
            (est.theta_onestep - est.theta_plugin).abs() < 1e-12,
            "theta_onestep must equal theta_plugin when penalty=0"
        );
        // Perfect fit → zero residuals → se = 0
        assert!(
            est.se.abs() < 1e-12,
            "se must be zero for perfect fit, got {:.6e}",
            est.se
        );
        assert_eq!(est.n_effective, n);
    }

    /// Non-zero penalty introduces a penalty_bias and makes the one-step
    /// estimate differ from the plug-in.
    #[test]
    fn nonzero_penalty_shifts_onestep() {
        let n = 3_usize;
        let x = Array2::<f64>::eye(n);
        let beta = array![2.0_f64, 2.0, 2.0]; // all-twos
        let mu = x.dot(&beta);
        let y = mu.clone();
        // penalty = I, penalty_beta = I @ beta = beta
        let penalty = Array2::<f64>::eye(n);
        let penalty_beta = beta.clone();
        let input = GaussianIdentityAverageDerivativeInput {
            design: x.view(),
            derivative_design: x.view(),
            y: y.view(),
            mu: mu.view(),
            beta: beta.view(),
            penalty: penalty.view(),
            penalty_beta: penalty_beta.view(),
        };
        let est = average_derivative_gaussian_identity(&input).expect("functional estimate");
        // H = XᵀX + λS = I + I = 2I, H⁻¹ = 0.5I
        // a = (1/3, 1/3, 1/3)
        // riesz = H⁻¹ a = 0.5 * (1/3, …) = (1/6, …)
        // penalty_bias = riesz @ penalty_beta = riesz @ beta = 3 * (1/6) * 2 = 1
        // theta_plugin = a @ beta = 2.0
        // theta_onestep = 2.0 + 1.0 = 3.0
        assert!((est.theta_plugin - 2.0).abs() < 1e-10, "plugin={}", est.theta_plugin);
        assert!((est.penalty_bias - 1.0).abs() < 1e-10, "bias={}", est.penalty_bias);
        assert!((est.theta_onestep - 3.0).abs() < 1e-10, "onestep={}", est.theta_onestep);
    }

    /// Empty design returns an error.
    #[test]
    fn empty_design_returns_error() {
        let x = Array2::<f64>::zeros((0, 0));
        let empty1d = Array1::<f64>::zeros(0);
        let penalty = Array2::<f64>::zeros((0, 0));
        let input = GaussianIdentityAverageDerivativeInput {
            design: x.view(),
            derivative_design: x.view(),
            y: empty1d.view(),
            mu: empty1d.view(),
            beta: empty1d.view(),
            penalty: penalty.view(),
            penalty_beta: empty1d.view(),
        };
        assert!(
            average_derivative_gaussian_identity(&input).is_err(),
            "empty design must return an error"
        );
    }
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
