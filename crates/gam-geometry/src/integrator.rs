use ndarray::{Array1, Array2, ArrayView1};

use crate::manifold::{GeometryResult, RiemannianManifold, check_len};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeodesicIntegrator {
    pub steps: usize,
    pub step_size: f64,
}

impl Default for GeodesicIntegrator {
    fn default() -> Self {
        Self {
            steps: 32,
            step_size: 1.0 / 32.0,
        }
    }
}

impl GeodesicIntegrator {
    /// Integrate the geodesic ODE on the manifold by composing the closed-form
    /// geodesic flow `exp_p(v · h)` with parallel transport of the velocity
    /// along that segment. This is the mathematically correct flow on every
    /// Riemannian manifold whose `RiemannianManifold` impl supplies an
    /// `exp_map` and `parallel_transport` consistent with its metric — it
    /// reduces to a Christoffel-driven leap-frog integrator on coordinates
    /// where Γ ≡ 0 (e.g. Euclidean / flat tori) but stays on the manifold
    /// and conserves the Riemannian kinetic energy on positively curved
    /// spaces such as the sphere, which previously drifted off because the
    /// extrinsic Christoffel symbols are not zero.
    pub fn integrate(
        &self,
        manifold: &dyn RiemannianManifold,
        point: ArrayView1<'_, f64>,
        tangent: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let d = manifold.ambient_dim();
        check_len("geodesic integrator point", point.len(), d)?;
        check_len("geodesic integrator tangent", tangent.len(), d)?;
        // Integrate the geodesic from parameter t=0 to t=1 so the result is
        // the canonical unit-time endpoint exp_p(v). The IVP energy ½|v(t)|²
        // is invariant along any geodesic; the BVP quantity ½|log_p(γ(1))|²
        // matches ½|v|² only when total integration time equals 1. We honour
        // both `steps` (substep count) and `step_size` (substep duration) by
        // taking the finer of the two — `n = max(steps, ⌈1/step_size⌉)` —
        // each of duration `h = 1/n`. This gives strictly more accuracy when
        // either bound is tightened and never overshoots the unit-time
        // endpoint, so closed-form exp_map/parallel_transport composition is
        // numerically exact for any caller-provided refinement.
        let mut x = point.to_owned();
        let mut v = manifold.project_tangent(point, tangent)?;
        let steps_from_size = if self.step_size.is_finite() && self.step_size > 0.0 {
            (1.0 / self.step_size).ceil() as usize
        } else {
            1
        };
        let n_substeps = self.steps.max(1).max(steps_from_size);
        let h = 1.0 / n_substeps as f64;
        for _ in 0..n_substeps {
            let step_tangent = &v * h;
            let x_next = manifold.exp_map(x.view(), step_tangent.view())?;
            let mut path = Array2::<f64>::zeros((2, d));
            path.row_mut(0).assign(&x);
            path.row_mut(1).assign(&x_next);
            let v_next = manifold.parallel_transport(path.view(), v.view())?;
            x = x_next;
            v = manifold.project_tangent(x.view(), v_next.view())?;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifolds::euclidean::EuclideanManifold;
    use ndarray::array;

    fn integrator_default() -> GeodesicIntegrator {
        GeodesicIntegrator::default()
    }

    #[test]
    fn default_has_expected_steps_and_step_size() {
        let g = integrator_default();
        assert_eq!(g.steps, 32);
        assert!((g.step_size - 1.0 / 32.0).abs() < 1e-15);
    }

    #[test]
    fn euclidean_geodesic_is_straight_line() {
        // On R^n the geodesic is a straight line: exp_p(v) = p + v.
        // The integrator should recover p + v exactly regardless of step count.
        let m = EuclideanManifold::new(2);
        let p = array![1.0_f64, 2.0];
        let v = array![3.0_f64, 4.0];
        let result = integrator_default()
            .integrate(&m, p.view(), v.view())
            .unwrap();
        assert!((result[0] - 4.0).abs() < 1e-12, "x: {}", result[0]);
        assert!((result[1] - 6.0).abs() < 1e-12, "y: {}", result[1]);
    }

    #[test]
    fn euclidean_zero_tangent_returns_same_point() {
        let m = EuclideanManifold::new(3);
        let p = array![5.0_f64, -1.0, 2.0];
        let v = array![0.0_f64, 0.0, 0.0];
        let result = integrator_default()
            .integrate(&m, p.view(), v.view())
            .unwrap();
        for i in 0..3 {
            assert!((result[i] - p[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn dimension_mismatch_point_returns_error() {
        let m = EuclideanManifold::new(3);
        let p = array![1.0_f64, 2.0]; // wrong length
        let v = array![0.0_f64, 0.0, 0.0];
        assert!(integrator_default().integrate(&m, p.view(), v.view()).is_err());
    }

    #[test]
    fn single_step_integrator_still_recovers_euclidean_geodesic() {
        let m = EuclideanManifold::new(2);
        let g = GeodesicIntegrator { steps: 1, step_size: 1.0 };
        let p = array![0.0_f64, 0.0];
        let v = array![2.0_f64, -3.0];
        let result = g.integrate(&m, p.view(), v.view()).unwrap();
        assert!((result[0] - 2.0).abs() < 1e-12);
        assert!((result[1] - (-3.0)).abs() < 1e-12);
    }
}
