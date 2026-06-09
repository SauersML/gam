use gam::{EuclideanManifold, RiemannianObjective, RiemannianTrustRegion};
use ndarray::{Array1, ArrayView1, arr1};

/// Trivial linear objective `f(x) = gᵀx` on Euclidean space. The gradient is the
/// constant `g`, so with no Hessian-vector product the trust-region subproblem
/// returns the exact Cauchy point: the steepest-descent direction stretched to
/// the trust-region boundary, i.e. a step of norm exactly `Δ`. This makes the
/// first-step length an exact function of the initial radius, so the cap can be
/// checked with no numerical slack.
struct Linear {
    grad: Array1<f64>,
}

impl RiemannianObjective for Linear {
    fn value_gradient(
        &mut self,
        point: ArrayView1<'_, f64>,
    ) -> gam::GeometryResult<(f64, Array1<f64>)> {
        Ok((point.dot(&self.grad), self.grad.clone()))
    }
}

/// Regression for #957: when the configured initial `radius` exceeds the
/// `max_radius` hard cap, the *first* trust-region step must still honor
/// `max_radius`. Previously `delta` was initialized to `self.radius` and only
/// clamped on later expansions, so the very first Cauchy step could overshoot
/// the advertised maximum (e.g. `Δ₀ = 10`, `Δmax = 1` produced a step of norm
/// `10`). The initial radius is now clamped into `(0, max_radius]`.
#[test]
fn trust_region_first_step_respects_max_radius() {
    let manifold = EuclideanManifold::new(2);
    // Unit-norm gradient ⇒ the Cauchy step length equals `Δ` exactly.
    let mut objective = Linear {
        grad: arr1(&[3.0 / 5.0, 4.0 / 5.0]),
    };
    let max_radius = 1.0;
    let solver = RiemannianTrustRegion {
        radius: 10.0, // deliberately larger than the cap
        max_radius,
        max_iter: 1, // a single step, so we measure the very first move
        grad_tol: 0.0,
    };
    let x0 = arr1(&[0.0, 0.0]);
    let x1 = solver
        .minimize(&manifold, &mut objective, x0.view())
        .expect("trust-region minimize should succeed");

    let step_norm = (&x1 - &x0).mapv(|v| v * v).sum().sqrt();
    assert!(
        step_norm <= max_radius + 1.0e-12,
        "first trust-region step norm {step_norm} exceeded the max_radius cap {max_radius}; \
         radius > max_radius must be clamped before the first step (#957)"
    );
    // The step should actually reach the (capped) boundary, confirming we
    // clamped to `max_radius` rather than collapsing the step to zero.
    assert!(
        step_norm >= max_radius - 1.0e-9,
        "first step norm {step_norm} fell short of the capped radius {max_radius}; \
         the radius should be clamped to max_radius, not shrunk away"
    );
}
