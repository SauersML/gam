#![deny(dead_code)]
#![deny(unused_imports)]
#![allow(non_snake_case)]

pub mod alo;
pub mod basis;
pub mod construction;
pub mod diagnostics;
pub mod estimate;
pub mod faer_ndarray;
pub mod families;
pub mod hmc;
pub mod hull;
pub mod joint;
pub mod matrix;
pub mod pirls;
pub mod quadrature;
pub mod seeding;
pub mod survival;
pub mod types;
pub mod visualizer;

pub use estimate::{FitOptions, FitResult, PredictResult, fit_gam, optimize_external_design, predict_gam};
pub use types::LikelihoodFamily;
