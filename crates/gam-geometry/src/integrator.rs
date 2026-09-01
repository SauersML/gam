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
        assert!(
            integrator_default()
                .integrate(&m, p.view(), v.view())
                .is_err()
        );
    }

    #[test]
    fn single_step_integrator_still_recovers_euclidean_geodesic() {
        let m = EuclideanManifold::new(2);
        let g = GeodesicIntegrator {
            steps: 1,
            step_size: 1.0,
        };
        let p = array![0.0_f64, 0.0];
        let v = array![2.0_f64, -3.0];
        let result = g.integrate(&m, p.view(), v.view()).unwrap();
        assert!((result[0] - 2.0).abs() < 1e-12);
        assert!((result[1] - (-3.0)).abs() < 1e-12);
    }
}
