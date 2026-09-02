use gam::{EuclideanManifold, RiemannianLBFGS, RiemannianObjective};
use ndarray::{Array1, Array2, ArrayView1, arr1, arr2};

/// `f(x) = ½ xᵀAx` for a fixed SPD `A`, so `∇f(x) = Ax`.
///
/// `A` is deliberately NOT the identity. Under `A = I` the gradient IS the
/// point, so the secant pair collapses to `y = x⋆ − x₀ = s` and the curvature
/// `sᵀy = ‖s‖²` is positive for any optimizer that moves at all — the check
/// becomes an algebraic tautology that no transport or sign defect could fail.
/// With `A = diag(3, ½)` the pair is `y = A s ≠ s`, so `sᵀy = sᵀAs` genuinely
/// probes the metric curvature condition the L-BFGS update depends on.
struct Quadratic {
    a: Array2<f64>,
}

impl RiemannianObjective for Quadratic {
    fn value_gradient(
        &mut self,
        point: ArrayView1<'_, f64>,
    ) -> gam::GeometryResult<(f64, Array1<f64>)> {
        let gradient = self.a.dot(&point);
        let value = 0.5 * point.dot(&gradient);
        Ok((value, gradient))
    }
}

