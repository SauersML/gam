use crate::estimate::EstimationError;
use crate::families::strategy::{FamilyStrategy, strategy_for_family};
use crate::types::{InverseLink, LikelihoodFamily};
use ndarray::{Array1, ArrayView1};
use statrs::function::erf::erfc;

/// Standard normal PDF.
#[inline]
pub fn normal_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}
