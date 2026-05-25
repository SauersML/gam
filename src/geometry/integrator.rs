use ndarray::{Array1, Array2, ArrayView1};

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
        let mut x = point.to_owned();
        let mut v = manifold.project_tangent(point, tangent)?;
        let steps = self.steps.max(1);
        let h = self.step_size;
        for _ in 0..steps {
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
