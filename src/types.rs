use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use std::ops::{Deref, DerefMut};

pub use crate::hull::PeeledHull;
use crate::survival::SurvivalSpec;

/// Shared engine-level link selector for generalized models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LinkFunction {
    Logit,
    Identity,
}

/// Engine-level likelihood selector used by generic solver entrypoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LikelihoodFamily {
    GaussianIdentity,
    BinomialLogit,
    RoystonParmar,
}

/// Engine-level model family selector used by internal solver pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFamily {
    Gam(LinkFunction),
    Survival(SurvivalSpec),
}

pub fn default_reml_parallel_threshold() -> usize {
    4
}

pub fn default_mcmc_enabled() -> bool {
    true
}

/// Engine-only optimizer configuration.
/// This intentionally excludes any domain feature configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model_family: ModelFamily,
    pub convergence_tolerance: f64,
    pub max_iterations: usize,
    pub reml_convergence_tolerance: f64,
    pub reml_max_iterations: u64,
    #[serde(default)]
    pub firth_bias_reduction: bool,
    #[serde(default = "default_reml_parallel_threshold")]
    pub reml_parallel_threshold: usize,
    #[serde(default = "default_mcmc_enabled")]
    pub mcmc_enabled: bool,
}

impl ModelConfig {
    pub fn external(
        link: LinkFunction,
        reml_tol: f64,
        reml_max_iter: usize,
        firth_bias_reduction: bool,
    ) -> Self {
        Self {
            model_family: ModelFamily::Gam(link),
            convergence_tolerance: reml_tol,
            max_iterations: 500,
            reml_convergence_tolerance: reml_tol,
            reml_max_iterations: reml_max_iter as u64,
            firth_bias_reduction,
            reml_parallel_threshold: default_reml_parallel_threshold(),
            mcmc_enabled: true,
        }
    }

    pub fn link_function(&self) -> Result<LinkFunction, &'static str> {
        match self.model_family {
            ModelFamily::Gam(link) => Ok(link),
            ModelFamily::Survival(_) => {
                Err("link_function is not applicable for survival model family")
            }
        }
    }

    pub fn survival_spec(&self) -> Option<SurvivalSpec> {
        match self.model_family {
            ModelFamily::Gam(_) => None,
            ModelFamily::Survival(spec) => Some(spec),
        }
    }
}

/// Optional joint single-index link data for calibrated predictions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointLinkModel {
    pub knot_range: (f64, f64),
    pub knot_vector: Array1<f64>,
    pub link_transform: ndarray::Array2<f64>,
    pub beta_link: Array1<f64>,
    pub degree: usize,
}

#[repr(transparent)]
#[derive(Clone, Debug, PartialEq)]
pub struct Coefficients(pub Array1<f64>);

impl Coefficients {
    pub fn new(values: Array1<f64>) -> Self {
        Self(values)
    }

    pub fn zeros(len: usize) -> Self {
        Self(Array1::zeros(len))
    }
}

impl Deref for Coefficients {
    type Target = Array1<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for Coefficients {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AsRef<Array1<f64>> for Coefficients {
    fn as_ref(&self) -> &Array1<f64> {
        &self.0
    }
}

impl From<Array1<f64>> for Coefficients {
    fn from(values: Array1<f64>) -> Self {
        Self(values)
    }
}

impl From<Coefficients> for Array1<f64> {
    fn from(values: Coefficients) -> Self {
        values.0
    }
}

#[repr(transparent)]
#[derive(Clone, Debug, PartialEq)]
pub struct LinearPredictor(pub Array1<f64>);

impl LinearPredictor {
    pub fn new(values: Array1<f64>) -> Self {
        Self(values)
    }

    pub fn zeros(len: usize) -> Self {
        Self(Array1::zeros(len))
    }
}

impl Deref for LinearPredictor {
    type Target = Array1<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LinearPredictor {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AsRef<Array1<f64>> for LinearPredictor {
    fn as_ref(&self) -> &Array1<f64> {
        &self.0
    }
}

impl From<Array1<f64>> for LinearPredictor {
    fn from(values: Array1<f64>) -> Self {
        Self(values)
    }
}

impl From<LinearPredictor> for Array1<f64> {
    fn from(values: LinearPredictor) -> Self {
        values.0
    }
}

#[repr(transparent)]
#[derive(Clone, Debug, PartialEq)]
pub struct LogSmoothingParams(pub Array1<f64>);

impl LogSmoothingParams {
    pub fn new(values: Array1<f64>) -> Self {
        Self(values)
    }
}

impl Deref for LogSmoothingParams {
    type Target = Array1<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for LogSmoothingParams {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl From<Array1<f64>> for LogSmoothingParams {
    fn from(values: Array1<f64>) -> Self {
        Self(values)
    }
}

impl From<LogSmoothingParams> for Array1<f64> {
    fn from(values: LogSmoothingParams) -> Self {
        values.0
    }
}

#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct LogSmoothingParamsView<'a>(pub ArrayView1<'a, f64>);

impl<'a> LogSmoothingParamsView<'a> {
    pub fn new(values: ArrayView1<'a, f64>) -> Self {
        Self(values)
    }

    pub fn exp(&self) -> Array1<f64> {
        self.0.mapv(f64::exp)
    }
}

impl<'a> Deref for LogSmoothingParamsView<'a> {
    type Target = ArrayView1<'a, f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
