#![deny(dead_code)]
#![deny(unused_variables)]
#![deny(unused_imports)]

pub mod families;
pub mod inference;
pub mod linalg;
pub mod solver;
pub mod terms;
#[cfg(test)]
pub mod testing;
pub mod types;

pub use inference::{alo, data, diagnostics, generative, hmc, predict, probability, quadrature};
pub use linalg::{faer_ndarray, matrix, utils};
pub use solver::{estimate, mixture_link, pirls, seeding, smoothing, visualizer};
pub use terms::{basis, construction, hull, layout, smooth};

pub use families::bernoulli_marginal_slope;
pub use families::custom_family;
pub use families::gamlss;
pub use families::survival;
pub use families::survival_location_scale;
pub use families::transformation_normal;
pub use solver::workflow::{
    BernoulliMarginalSlopeFitRequest, BinomialLocationScaleFitRequest, FitRequest, FitResult,
    GaussianLocationScaleFitRequest, LinkWiggleConfig, StandardBinomialWiggleConfig,
    StandardFitRequest, StandardFitResult, SurvivalLocationScaleFitRequest,
    SurvivalLocationScaleFitResult, fit_model,
};
