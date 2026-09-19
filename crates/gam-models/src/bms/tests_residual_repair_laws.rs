//! Covariances, score laws and the residual sub-block adapter the residual
//! repair tests share: the row-program gates in `residual_repair` and the
//! five-primary equivalence and finite-difference gates in
//! `residual_repair_kernel`.

use super::*;
use gam_math::jet_scalar::SymmetricQuadraticCoefficients;

/// A random SPD `(K+1) × (K+1)` matrix with unit `(0, 0)` entry, the
/// score's variance.
pub(crate) fn covariance(k: usize, seed: u64) -> MarginalSlopeCovariance {
    let mut state = seed;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state % 10_000) as f64 / 10_000.0 - 0.5
    };
    let dim = k + 1;
    let mut l = Array2::<f64>::zeros((dim, dim));
    for i in 0..dim {
        for j in 0..=i {
            l[[i, j]] = if i == j { 0.6 + 0.4 * next().abs() } else { 0.5 * next() };
        }
    }
    let mut sigma = l.dot(&l.t());
    let scale = sigma[[0, 0]].sqrt();
    sigma.mapv_inplace(|v| v / (scale * scale));
    MarginalSlopeCovariance::full(sigma).expect("SPD covariance")
}

/// A skewed finite law of the score: a discretised `exp(0.7·u)`,
/// standardised to mean 0 and variance 1, on 41 ascending nodes.
pub(crate) fn skewed_grid() -> EmpiricalZGrid {
    let u: Vec<f64> = (0..41).map(|i| -3.0 + 0.15 * i as f64).collect();
    let raw_weights: Vec<f64> = u.iter().map(|&v| (-0.5 * v * v).exp()).collect();
    let total: f64 = raw_weights.iter().sum();
    let weights: Vec<f64> = raw_weights.iter().map(|w| w / total).collect();
    let raw_nodes: Vec<f64> = u.iter().map(|&v| (0.7 * v).exp()).collect();
    let mean: f64 = raw_nodes.iter().zip(&weights).map(|(x, w)| x * w).sum();
    let variance: f64 = raw_nodes
        .iter()
        .zip(&weights)
        .map(|(x, w)| w * (x - mean) * (x - mean))
        .sum();
    let nodes = raw_nodes
        .iter()
        .map(|x| (x - mean) / variance.sqrt())
        .collect();
    EmpiricalZGrid::new(nodes, weights, "residual repair test law")
        .expect("standardised ascending nodes with normalised weights form a finite law")
}

/// The probabilists' Gauss–Hermite law: the finite law on which the mixture
/// anchor is the Gaussian closed form to quadrature tolerance.
pub(crate) fn hermite_grid(m: usize) -> EmpiricalZGrid {
    let rule = gam_math::quadrature::gauss_hermite_rule(m)
        .expect("the test orders are positive Gauss-Hermite orders");
    let mut pairs: Vec<(f64, f64)> = rule
        .nodes
        .iter()
        .zip(&rule.weights)
        .map(|(&x, &w)| (std::f64::consts::SQRT_2 * x, w / std::f64::consts::PI.sqrt()))
        .collect();
    pairs.sort_by(|a, b| a.0.total_cmp(&b.0));
    let total: f64 = pairs.iter().map(|p| p.1).sum();
    EmpiricalZGrid::new(
        pairs.iter().map(|p| p.0).collect(),
        pairs.iter().map(|p| p.1 / total).collect(),
        "hermite test law",
    )
    .expect("sorted Hermite nodes with normalised weights form a finite law")
}

/// The residual sub-block `Σ_rr` of a joint covariance, as quadratic-form
/// coefficients.
pub(crate) struct ResidualBlock<'a>(pub(crate) &'a MarginalSlopeCovariance);

impl SymmetricQuadraticCoefficients for ResidualBlock<'_> {
    fn dimension(&self) -> usize {
        self.0.dim() - 1
    }
    fn multiply(&self, input: &[f64], output: &mut [f64]) {
        let mut lifted = vec![0.0_f64; input.len() + 1];
        lifted[1..].copy_from_slice(input);
        let mut image = vec![0.0_f64; input.len() + 1];
        self.0.multiply(&lifted, &mut image);
        output.copy_from_slice(&image[1..]);
    }
    fn coefficient(&self, row: usize, column: usize) -> f64 {
        self.0.coefficient(row + 1, column + 1)
    }
}
