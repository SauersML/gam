//! Pins the Poincaré tangent-space decoder backward
//! (`geometry::poincare::tangent_decode_backward`) against the forward map.
//!
//! Two complementary checks, because the map is smooth inside the ball but
//! singular at the boundary:
//!
//! * INSIDE the ball the decode `(atoms, gates) -> x_hat` is smooth, so a
//!   central finite difference of `<G, x_hat>` w.r.t. both atoms and gates
//!   must match the analytic VJP to tight tolerance. This pins Steps 1–3
//!   (the `phi`/`psi` Jacobians and the gate↔tangent contraction).
//!
//! * OUTSIDE the ball an atom is radially clamped onto the boundary, where
//!   `atanh(t)` (t → 1) makes the analytic gradient ~1e11 and wildly
//!   nonlinear — finite differences are meaningless there. Instead we assert
//!   the exact property Step 4's radial-projection chain rule guarantees:
//!   moving a clamped atom radially does not change its projection, so the
//!   radial component of its gradient is annihilated (`grad_atom · â == 0`).
//!   That is precisely the factor an earlier pass-through implementation
//!   dropped, so this is the regression guard for it. Do not relax it.

use gam::geometry::poincare::{tangent_decode_backward, tangent_decode_forward};
use ndarray::{Array2, ArrayView2, arr2};

const CURVATURE: f64 = -1.0; // unit-curvature ball: radius ~1.

/// `<G, x_hat(atoms, gates)>` — the scalar whose gradients the VJP reproduces.
fn objective(atoms: &Array2<f64>, gates: &Array2<f64>, g: ArrayView2<'_, f64>) -> f64 {
    let (x_hat, _cache) = tangent_decode_forward(atoms.view(), gates.view(), CURVATURE)
        .expect("forward decode should succeed");
    x_hat.iter().zip(g.iter()).map(|(xi, gi)| xi * gi).sum()
}

/// Central finite difference of `objective` w.r.t. one entry of `target`,
/// where `target` is either `atoms` or `gates` (selected by `perturb_atoms`).
fn fd_grad(
    atoms: &Array2<f64>,
    gates: &Array2<f64>,
    g: ArrayView2<'_, f64>,
    perturb_atoms: bool,
) -> Array2<f64> {
    let h = 1.0e-6;
    let base = if perturb_atoms { atoms } else { gates };
    let (rows, cols) = base.dim();
    let mut grad = Array2::<f64>::zeros((rows, cols));
    for r in 0..rows {
        for c in 0..cols {
            let mut plus = base.clone();
            let mut minus = base.clone();
            plus[[r, c]] += h;
            minus[[r, c]] -= h;
            let (f_plus, f_minus) = if perturb_atoms {
                (objective(&plus, gates, g), objective(&minus, gates, g))
            } else {
                (objective(atoms, &plus, g), objective(atoms, &minus, g))
            };
            grad[[r, c]] = (f_plus - f_minus) / (2.0 * h);
        }
    }
    grad
}

fn max_abs_diff(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

#[test]
fn poincare_decode_backward_matches_finite_difference_inside_ball() {
    // All atoms well inside the unit ball so the map is smooth (no clamp, no
    // boundary atanh blow-up). F = 2 atoms, d = 2, batch = 2.
    let atoms = arr2(&[[0.30, -0.20], [0.10, 0.25]]);
    let gates = arr2(&[[0.60, 0.40], [0.20, 0.50]]);
    let g = arr2(&[[0.70, -0.30], [0.40, 0.50]]); // cotangent on x_hat (batch, d)

    let (x_hat, cache) = tangent_decode_forward(atoms.view(), gates.view(), CURVATURE)
        .expect("forward decode should succeed");
    assert_eq!(x_hat.dim(), (2, 2));

    let (grad_gates, grad_atoms) =
        tangent_decode_backward(&cache, g.view()).expect("backward decode should succeed");

    let fd_atoms = fd_grad(&atoms, &gates, g.view(), true);
    let fd_gates = fd_grad(&atoms, &gates, g.view(), false);

    let err_atoms = max_abs_diff(&grad_atoms, &fd_atoms);
    let err_gates = max_abs_diff(&grad_gates, &fd_gates);
    assert!(
        err_atoms < 1.0e-6,
        "analytic grad_atoms disagrees with finite difference (max abs err {err_atoms:.3e})"
    );
    assert!(
        err_gates < 1.0e-6,
        "analytic grad_gates disagrees with finite difference (max abs err {err_gates:.3e})"
    );
}

#[test]
fn poincare_decode_backward_annihilates_radial_gradient_for_clamped_atom() {
    // A single atom seeded well OUTSIDE the unit ball: the forward radially
    // projects it onto the boundary (s_f < 1). Step 4's chain rule must then
    // remove the radial component of its gradient — moving the raw atom
    // outward along its own direction cannot change the clamped projection.
    // We seed it along a non-axis direction so the annihilation is a genuine
    // vector condition, not a coordinate coincidence.
    let dir = [
        3.0_f64.recip().sqrt() * 2.0,
        3.0_f64.recip().sqrt() * 2.0_f64.sqrt(),
    ];
    // |dir| chosen > 1 so the atom is outside the radius-~1 ball.
    let atoms = arr2(&[[dir[0] * 2.0, dir[1] * 2.0]]); // clearly outside
    let gates = arr2(&[[1.0]]); // batch = 1, F = 1
    let g = arr2(&[[0.7, -0.4]]);

    let (_x_hat, cache) = tangent_decode_forward(atoms.view(), gates.view(), CURVATURE)
        .expect("forward decode should succeed");
    let (_grad_gates, grad_atoms) =
        tangent_decode_backward(&cache, g.view()).expect("backward decode should succeed");

    // Unit radial direction of the raw atom (projection is radial, so the
    // projected atom shares this direction).
    let a0 = atoms[[0, 0]];
    let a1 = atoms[[0, 1]];
    let norm = (a0 * a0 + a1 * a1).sqrt();
    let (ahat0, ahat1) = (a0 / norm, a1 / norm);

    // The gradient must not be trivially zero (the tangential part survives).
    let mag = (grad_atoms[[0, 0]].powi(2) + grad_atoms[[0, 1]].powi(2)).sqrt();
    assert!(
        mag > 1.0e-9,
        "clamped-atom gradient collapsed to zero; tangential component should survive"
    );
    // Radial component must be annihilated. The check is RELATIVE, not a tiny
    // absolute bound: at the boundary psi'(t) ~ 1e11, so Step 4 forms the
    // radial cancellation as a difference of ~1e11 terms and f64 roundoff
    // leaves ~1e-6 relative residual — that is the conditioning limit, not a
    // defect. A *missing* Step 4 (the regression this guards) would leave the
    // full radial term, i.e. |radial|/|grad| ~ O(1), so 1e-4 separates the two
    // unambiguously. Do not turn this into a looser O(1) bound.
    let radial = grad_atoms[[0, 0]] * ahat0 + grad_atoms[[0, 1]] * ahat1;
    assert!(
        radial.abs() < 1.0e-4 * mag,
        "Step 4 must annihilate the radial gradient component of a clamped atom; \
         |radial|/|grad| = {:.3e} (grad = [{:.6e}, {:.6e}])",
        radial.abs() / mag,
        grad_atoms[[0, 0]],
        grad_atoms[[0, 1]]
    );
}
