//! Finite-difference pin for the curved-manifold `exp_map_vjp` (WS1 of the
//! manifold-SAE roadmap). The sphere exp map has a genuinely non-identity
//! vector–Jacobian product; the analytic backward in `geometry::sphere` must
//! agree with a central finite difference of `<g, exp_p(v)>` w.r.t. both the
//! base point `p` and the raw tangent input `v`. A regression here is the
//! "silently-wrong gradient" failure mode this VJP exists to prevent, so the
//! tolerance is intentionally tight and must never be loosened to pass.
//!
//! The forward `exp_map` uses `point` verbatim (it does NOT renormalize), so we
//! exercise both a unit `p` (textbook on-sphere Jacobi field) and a non-unit
//! `p` (the general-ambient branch that carries the `c(1-|p|^2)` terms), plus
//! the small-`theta` Taylor branch.

use gam::{RiemannianManifold, SphereManifold};
use ndarray::{Array1, ArrayView1, arr1};

/// Scalar objective `<g, exp_p(v)>` whose gradients the VJP reproduces.
fn objective(
    manifold: &SphereManifold,
    p: ArrayView1<'_, f64>,
    v: ArrayView1<'_, f64>,
    g: ArrayView1<'_, f64>,
) -> f64 {
    let y = manifold.exp_map(p, v).expect("exp_map should succeed");
    y.iter().zip(g.iter()).map(|(yi, gi)| yi * gi).sum()
}

/// Central finite difference of `objective` w.r.t. one argument vector.
fn fd_grad(
    manifold: &SphereManifold,
    base: &Array1<f64>,
    g: ArrayView1<'_, f64>,
    perturb_point: bool,
    p: &Array1<f64>,
    v: &Array1<f64>,
) -> Array1<f64> {
    let h = 1.0e-6;
    let n = base.len();
    let mut grad = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut plus = base.clone();
        let mut minus = base.clone();
        plus[i] += h;
        minus[i] -= h;
        let (f_plus, f_minus) = if perturb_point {
            (
                objective(manifold, plus.view(), v.view(), g),
                objective(manifold, minus.view(), v.view(), g),
            )
        } else {
            (
                objective(manifold, p.view(), plus.view(), g),
                objective(manifold, p.view(), minus.view(), g),
            )
        };
        grad[i] = (f_plus - f_minus) / (2.0 * h);
    }
    grad
}

fn max_abs_diff(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

/// Returns the max-abs deviation of the analytic VJP from the central finite
/// difference, as `(err_tangent, err_point)`. The caller asserts on these so
/// the assertion lives directly in each `#[test]` (the build's ban-checker
/// rejects a test whose only verification hides inside a helper).
fn vjp_fd_errors(p: Array1<f64>, v: Array1<f64>, g: Array1<f64>) -> (f64, f64) {
    let manifold = SphereManifold::new(2); // 2-sphere in R^3

    let (grad_p, grad_v) = manifold
        .exp_map_vjp(p.view(), v.view(), g.view())
        .expect("sphere exp_map_vjp should succeed");

    let fd_v = fd_grad(&manifold, &v, g.view(), false, &p, &v);
    let fd_p = fd_grad(&manifold, &p, g.view(), true, &p, &v);

    (max_abs_diff(&grad_v, &fd_v), max_abs_diff(&grad_p, &fd_p))
}

#[test]
fn sphere_exp_map_vjp_matches_finite_difference_unit_point() {
    // Unit base point: the textbook on-sphere Jacobi-field VJP.
    let p = arr1(&[0.6, -0.8, 0.0]); // |p| = 1
    let v = arr1(&[0.15, 0.10, -0.30]); // arbitrary ambient tangent input
    let g = arr1(&[0.7, -0.2, 0.5]); // arbitrary cotangent
    let (err_v, err_p) = vjp_fd_errors(p, v, g);
    assert!(
        err_v < 1.0e-6,
        "unit-p analytic grad_v vs finite difference: max abs err {err_v:.3e}"
    );
    assert!(
        err_p < 1.0e-6,
        "unit-p analytic grad_p vs finite difference: max abs err {err_p:.3e}"
    );
}

#[test]
fn sphere_exp_map_vjp_matches_finite_difference_nonunit_point() {
    // Non-unit base point exercises the general-ambient branch (the
    // c(1-|p|^2) terms in w_v / w_p) that a unit-p-only derivation would
    // silently get wrong. exp_map uses `point` verbatim, so this is a real
    // path the VJP must match.
    let p = arr1(&[0.2, -0.4, 0.5]); // |p|^2 = 0.45 != 1
    let v = arr1(&[0.25, 0.10, -0.20]);
    let g = arr1(&[-0.3, 0.6, 0.4]);
    let (err_v, err_p) = vjp_fd_errors(p, v, g);
    assert!(
        err_v < 1.0e-6,
        "nonunit-p analytic grad_v vs finite difference: max abs err {err_v:.3e}"
    );
    assert!(
        err_p < 1.0e-6,
        "nonunit-p analytic grad_p vs finite difference: max abs err {err_p:.3e}"
    );
}

#[test]
fn sphere_exp_map_accepts_nonunit_point_verbatim() {
    // The forward must consume the base point verbatim — neither rejecting a
    // non-unit `p` (the regression that made `exp_map_vjp` unreachable on the
    // off-sphere finite-difference path) nor silently renormalizing it. We pin
    // the exact ambient closed form `y = cos(θ) p + (sin(θ)/θ) (v − (p·v) p)`
    // for a deliberately non-unit `p`, which is also the curve the VJP above
    // differentiates.
    let manifold = SphereManifold::new(2);
    let p = arr1(&[0.2, -0.4, 0.5]); // |p|^2 = 0.45 != 1
    let v = arr1(&[0.25, 0.10, -0.20]);

    let y = manifold
        .exp_map(p.view(), v.view())
        .expect("exp_map must accept a non-unit base point verbatim");

    let c = p.dot(&v);
    let xi = &v - &(&p * c);
    let theta = xi.dot(&xi).sqrt();
    let expected = &(&p * theta.cos()) + &(&xi * (theta.sin() / theta));
    assert!(
        max_abs_diff(&y, &expected) < 1.0e-12,
        "exp_map on non-unit p must equal the verbatim ambient closed form"
    );
    // It must NOT renormalize: for a non-unit p the ambient image is off the
    // unit sphere, and forcing it back would contradict the map the VJP
    // differentiates.
    let out_norm_sq = y.dot(&y);
    assert!(
        (out_norm_sq - 1.0).abs() > 1.0e-6,
        "exp_map silently renormalized a non-unit base point (|y|^2 = {out_norm_sq})"
    );
}

#[test]
fn sphere_exp_map_vjp_small_theta_branch_is_continuous() {
    // The theta < 1e-10 Taylor branch cannot be pinned with a central finite
    // difference: a usable step (h ~ 1e-6) straddles the 1e-10 branch boundary,
    // so FD would sample mostly the main branch. Instead we assert the two
    // closed forms AGREE across the switch — pick a unit tangent direction
    // orthogonal to a unit `p` (so theta == |v|) and evaluate the analytic VJP
    // just below and just above the 1e-10 threshold. A divergence here is a
    // discontinuity between the Taylor branch and the general formula.
    let manifold = SphereManifold::new(2);
    let p = arr1(&[0.0, 0.0, 1.0]); // |p| = 1
    let u = arr1(&[0.6, 0.8, 0.0]); // |u| = 1, u . p = 0  =>  xi == v, theta == |v|
    let g = arr1(&[0.4, 0.5, -0.6]);

    let v_below: Array1<f64> = &u * 9.9e-11; // theta ~ 9.9e-11  -> Taylor branch
    let v_above: Array1<f64> = &u * 1.01e-10; // theta ~ 1.01e-10 -> general formula

    let (gp_below, gv_below) = manifold
        .exp_map_vjp(p.view(), v_below.view(), g.view())
        .expect("exp_map_vjp (below threshold) should succeed");
    let (gp_above, gv_above) = manifold
        .exp_map_vjp(p.view(), v_above.view(), g.view())
        .expect("exp_map_vjp (above threshold) should succeed");

    let err_v = max_abs_diff(&gv_below, &gv_above);
    let err_p = max_abs_diff(&gp_below, &gp_above);
    assert!(
        err_v < 1.0e-7,
        "grad_v discontinuous across the small-theta switch (max abs err {err_v:.3e})"
    );
    assert!(
        err_p < 1.0e-7,
        "grad_p discontinuous across the small-theta switch (max abs err {err_p:.3e})"
    );
}
