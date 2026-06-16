//! Bug hunt: `ShapeMonotonicityPenalty::hvp` returns a Hessian-vector product that is
//! too large by a factor of `1 / smoothing_eps`.
//!
//! The penalty contribution for an adjacent pair `(a, b=a+1)` and column `j` is
//! (src/terms/analytic_penalties.rs:3729 `edge_value`)
//!
//!     P = weight * softplus(z) * eps,    z = -direction * (t_b - t_a) / eps.
//!
//! Note the **outer `* eps`**. Differentiating once:
//!
//!     dP/dt_b = weight * eps * softplus'(z) * (dz/dt_b)
//!             = weight * eps * sigma(z) * (-direction / eps)
//!             = weight * (-direction) * sigma(z),
//!
//! where the `eps` cancels exactly **one** of the `1/eps` from `dz/dt`. The
//! gradient code (`edge_grad`, :3747) gets this right.
//!
//! Differentiating a second time (z is linear in t, so d²z/dt² = 0):
//!
//!     d²P/dt_b² = weight * eps * softplus''(z) * (dz/dt_b)²
//!               = weight * eps * sigma(z)(1 - sigma(z)) * (1/eps²)
//!               = weight * sigma(z)(1 - sigma(z)) / eps.
//!
//! The outer `eps` cancels **one** of the two `1/eps²` factors, leaving `1/eps`.
//! But `hvp` (src/terms/analytic_penalties.rs:3835) computes
//!
//!     let h = weight * sigma * (1.0 - sigma) / (eps * eps);
//!
//! i.e. it keeps the full `1/eps²` and drops the outer `eps`. The accompanying
//! comment ("the (dz/dtarget)² factor is 1/eps²") accounts for the chain-rule
//! square but forgets the `* eps` that multiplies softplus in the value. So the
//! returned curvature is `1/eps` times too large (4× for `eps = 0.25`).
//!
//! `value` and `grad_target` are internally consistent (a finite-difference of
//! the value matches the gradient to ~1e-10), which is what isolates the bug to
//! the second-derivative path. Any Newton / PIRLS / REML step that consumes this
//! penalty's `hvp` as curvature is fed a Hessian inflated by `1/smoothing_eps`.
//!
//! The struct doc claims "The Hessian is positive semidefinite (softplus is
//! convex) so the penalty composes cleanly with PIRLS/REML"
//! (src/terms/analytic_penalties.rs:3654) — PSD is preserved by a positive
//! scalar, but the *magnitude* is wrong, which is what curvature consumers rely
//! on.
//!
//! This test pins the exact closed-form Hessian against `hvp` and, independently,
//! against a central finite-difference of the (correct) gradient. Both currently
//! fail; both pass once the `(eps * eps)` is corrected to `eps`.

use gam::terms::analytic_penalties::{AnalyticPenalty, ShapeMonotonicityPenalty};
use ndarray::Array1;

fn sigmoid(z: f64) -> f64 {
    if z > 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let ez = z.exp();
        ez / (1.0 + ez)
    }
}

#[test]
fn monotonicity_hvp_matches_exact_hessian() {
    // Single column (d = 1), one adjacent pair (n_eff = 2): the Hessian is a
    // 2x2 block. Fixed (non-learnable) weight so rho is empty.
    let weight = 1.3_f64;
    let eps = 0.5_f64;
    let direction = 1.0_f64;
    let t0 = 0.0_f64;
    let t1 = 0.3_f64;

    let p = ShapeMonotonicityPenalty::new(weight, 2, direction, eps, false)
        .expect("construct monotonicity penalty");
    let target = Array1::from(vec![t0, t1]);
    let rho: Array1<f64> = Array1::from(vec![]);

    // Exact second derivative of P w.r.t. t_b (= t1), derived above.
    let slope = t1 - t0;
    let z = -direction * slope / eps;
    let sigma = sigmoid(z);
    let kappa = weight * sigma * (1.0 - sigma) / eps; // correct curvature magnitude

    // H = kappa * [[1, -1], [-1, 1]]. Probe column 0 via v = e_0.
    let v = Array1::from(vec![1.0, 0.0]);
    let hv = p.hvp(target.view(), rho.view(), v.view());

    // Expected H * e_0 = [kappa, -kappa].
    let expected = [kappa, -kappa];
    for i in 0..2 {
        assert!(
            (hv[i] - expected[i]).abs() < 1e-9,
            "hvp(e0)[{i}] = {} but exact Hessian column is {} \
             (off by a factor of ~1/eps = {:.3}); kappa = {kappa:.6}",
            hv[i],
            expected[i],
            1.0 / eps
        );
    }
}

#[test]
fn monotonicity_hvp_matches_finite_difference_of_gradient() {
    // The gradient is correct (it matches a finite-difference of the value);
    // the hvp must therefore match a finite-difference of the gradient along v.
    let weight = 0.9_f64;
    let eps = 0.25_f64;
    let p = ShapeMonotonicityPenalty::new(weight, 3, 1.0, eps, true)
        .expect("construct monotonicity penalty");

    // (n_eff = 3, d = 2) row-major latent block, generic smooth point.
    let target = Array1::from(vec![0.37, -1.2, 0.8, -0.4, 1.7, -0.9]);
    let rho = Array1::from(vec![0.15]); // learnable weight axis

    // Sanity: gradient agrees with a finite difference of the value.
    let g = p.grad_target(target.view(), rho.view());
    let h = 1e-6;
    for i in 0..target.len() {
        let mut xp = target.clone();
        let mut xm = target.clone();
        xp[i] += h;
        xm[i] -= h;
        let fd = (p.value(xp.view(), rho.view()) - p.value(xm.view(), rho.view())) / (2.0 * h);
        assert!(
            (fd - g[i]).abs() < 1e-6,
            "gradient[{i}] inconsistent with value FD: {} vs {}",
            g[i],
            fd
        );
    }

    // Direction with a nontrivial component on every coordinate.
    let v = Array1::from(vec![0.6, -1.1, 0.4, 0.9, -0.7, 1.3]);
    let hv = p.hvp(target.view(), rho.view(), v.view());

    let mut xp = target.clone();
    let mut xm = target.clone();
    for i in 0..target.len() {
        xp[i] += h * v[i];
        xm[i] -= h * v[i];
    }
    let gp = p.grad_target(xp.view(), rho.view());
    let gm = p.grad_target(xm.view(), rho.view());
    for i in 0..target.len() {
        let fd = (gp[i] - gm[i]) / (2.0 * h);
        assert!(
            (fd - hv[i]).abs() < 1e-5,
            "hvp[{i}] = {} disagrees with central-difference of grad_target = {} \
             (hvp carries an extra 1/eps = {:.1} factor)",
            hv[i],
            fd,
            1.0 / eps
        );
    }
}
