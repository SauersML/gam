//! Pure dense projection kernels shared by the projected-trace assembly in
//! `reml_outer_engine.rs`: the projected-factor trace `tr(Uᵀ A U)` evaluated without
//! materializing the projection, and the dense projected matrix `Uᵀ A U`. Both
//! are dependency-free `Array2`-only kernels relocated verbatim from the parent,
//! which re-imports them so every call site is unchanged.

use ndarray::Array2;

pub(crate) fn dense_trace_projected_factor(matrix: &Array2<f64>, factor: &Array2<f64>) -> f64 {
    let matrix_factor = matrix.dot(factor);
    factor
        .iter()
        .zip(matrix_factor.iter())
        .map(|(&f, &mf)| f * mf)
        .sum()
}

pub(crate) fn dense_projected_matrix(matrix: &Array2<f64>, factor: &Array2<f64>) -> Array2<f64> {
    factor.t().dot(&matrix.dot(factor))
}
