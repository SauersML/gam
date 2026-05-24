use ndarray::{Array1, ArrayView1};

use crate::geometry::manifold::{GeometryResult, RiemannianManifold, check_len};

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
    pub fn integrate(
        &self,
        manifold: &dyn RiemannianManifold,
        point: ArrayView1<'_, f64>,
        tangent: ArrayView1<'_, f64>,
    ) -> GeometryResult<Array1<f64>> {
        let d = manifold.ambient_dim();
        check_len("geodesic integrator point", point.len(), d)?;
        check_len("geodesic integrator tangent", tangent.len(), d)?;
        let mut x = point.to_owned();
        let mut v = manifold.project_tangent(point, tangent)?;
        let steps = self.steps.max(1);
        let h = self.step_size;
        for _ in 0..steps {
            let a = acceleration(manifold, x.view(), v.view())?;
            x = &x + &(v.clone() * h) + &(a.clone() * (0.5 * h * h));
            v = manifold.project_tangent(x.view(), (&v + &(a * h)).view())?;
        }
        manifold.retract(point, manifold.log_map(point, x.view())?.view())
    }
}

fn acceleration(
    manifold: &dyn RiemannianManifold,
    point: ArrayView1<'_, f64>,
    velocity: ArrayView1<'_, f64>,
) -> GeometryResult<Array1<f64>> {
    let gamma = manifold.christoffel_symbols(point)?;
    let d = velocity.len();
    let mut out = Array1::<f64>::zeros(d);
    for k in 0..d {
        let mut acc = 0.0;
        for i in 0..d {
            for j in 0..d {
                acc -= gamma[k][[i, j]] * velocity[i] * velocity[j];
            }
        }
        out[k] = acc;
    }
    Ok(out)
}
