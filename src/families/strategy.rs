use crate::types::{LikelihoodFamily, LinkFunction};
use crate::inference::generative::NoiseModel;
use ndarray::Array1;
use crate::quadrature::{QuadratureContext, IntegratedMomentsJet};
use crate::estimate::EstimationError;

/// The strategy interface for different likelihood/link families.
pub trait FamilyStrategy: std::fmt::Debug + Send + Sync {
    /// Name of the family.
    fn name(&self) -> &'static str;
    
    /// The base likelihood enum.
    fn family(&self) -> LikelihoodFamily;
    
    /// The specific link function.
    fn link_function(&self) -> LinkFunction;

    /// Generates the noise model for generative sampling.
    fn simulate_noise(
        &self, 
        mean: Array1<f64>,
        gaussian_scale: Option<f64>
    ) -> Result<NoiseModel, EstimationError>;

    /// Computes integrated family moments for PIRLS.
    fn integrated_moments(
        &self,
        quad_ctx: &QuadratureContext,
        eta: f64,
        se_eta: f64,
    ) -> Result<IntegratedMomentsJet, EstimationError>;
}
