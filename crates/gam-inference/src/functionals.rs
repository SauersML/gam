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

