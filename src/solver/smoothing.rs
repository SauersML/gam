use crate::seeding::{SeedConfig, SeedRiskProfile};
use ndarray::Array1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SmoothingOptimizerKind {
    Bfgs,
    Arc,
}

#[derive(Clone, Debug)]
pub struct SmoothingBfgsOptions {
    pub max_iter: usize,
    pub tol: f64,
    pub finite_diff_step: f64,
    /// Retained for API compatibility.
    /// This setting is only used by finite-difference fallback paths.
    pub fdhessian_max_dim: usize,
    pub optimizer_kind: SmoothingOptimizerKind,
    pub seed_config: SeedConfig,
}

impl Default for SmoothingBfgsOptions {
    fn default() -> Self {
        Self {
            max_iter: 200,
            tol: 1e-5,
            finite_diff_step: 1e-3,
            fdhessian_max_dim: usize::MAX,
            optimizer_kind: SmoothingOptimizerKind::Bfgs,
            seed_config: SeedConfig {
                risk_profile: SeedRiskProfile::GeneralizedLinear,
                ..SeedConfig::default()
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct SmoothingBfgsResult {
    pub rho: Array1<f64>,
    pub final_value: f64,
    pub iterations: usize,
    pub finalgrad_norm: f64,
    pub final_stationarity_residual: f64,
    pub final_boundviolation: f64,
    pub stationary: bool,
}
