use ndarray::{ArrayView1, ArrayView2};

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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

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
        assert!(
            (est.theta_plugin - 2.0).abs() < 1e-10,
            "plugin={}",
            est.theta_plugin
        );
        assert!(
            (est.penalty_bias - 1.0).abs() < 1e-10,
            "bias={}",
            est.penalty_bias
        );
        assert!(
            (est.theta_onestep - 3.0).abs() < 1e-10,
            "onestep={}",
            est.theta_onestep
        );
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

