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
