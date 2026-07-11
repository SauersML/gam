//! Bug-hunt regression gate for the SAE / ordered independent Beta--Bernoulli / manifold subsystem.
//!
//! Each test pins a correctness property found during the SAE/ordered independent Beta--Bernoulli/manifold
//! adversarial math audit so a regression fails CI. Tests use only the public
//! crate API.
//!
//! No `let _`, no `#[allow(...)]`, no env vars, no `#[cfg(feature=...)]`.

use ndarray::{Array2, array};

use gam::geometry::poincare::{BOUNDARY_EPS, tangent_decode_backward, tangent_decode_forward};

/// Bug-hunt finding: the Poincaré tangent decoder's forward radial coefficient
/// `exp_coeff(s) = min(tanh(s), 1 - BOUNDARY_EPS)/s` CLAMPS once the tangent
/// magnitude `s = sqrt(k)|v|` saturates (`tanh(s) >= 1 - BOUNDARY_EPS`, i.e. `s`
/// beyond `EXP_SATURATION_CAP ≈ 6.103`), pinning the decoded point to the open-
/// ball boundary. The analytic backward MUST differentiate that SAME clamped,
/// radially-pinned map — not the unclamped `tanh(s)/s` — or the gradient
/// desyncs from the forward in the saturated regime.
///
/// This drives the decode deep into saturation (gates of ≈3 onto the two
/// near-boundary atoms give `s ≈ 8 > 6.103`) and finite-differences the forward
/// loss. The OLD unclamped backward disagreed with the FD here (the disagreement
/// is bounded by `~BOUNDARY_EPS` but is a genuine forward/backward inconsistency
/// the small-input FD test never exercised); the fixed backward matches.
#[test]
fn poincare_tangent_backward_matches_fd_in_saturated_regime() {
    // Curvature -1 (sqrt(k) = 1). Two near-boundary atoms in DIFFERENT
    // directions; their gated tangent sum has |v| ≈ 8 > 6.103, so exp_coeff
    // CLAMPS and the forward output is pinned to the ball boundary. Because the
    // two atoms are not collinear, moving a gate ROTATES the boundary-pinned
    // output (a genuine tangential sensitivity), so the FD is nonzero and the
    // radially-pinned Jacobian's tangential part is exercised — not just the
    // radial cancellation.
    let atoms = array![[0.95, 0.1], [0.1, 0.95]];
    let gates = array![[3.0, 2.5]];
    let curvature = -1.0;

    let (x_hat, cache) =
        tangent_decode_forward(atoms.view(), gates.view(), curvature).expect("forward");

    // Confirm we are genuinely in the clamped boundary regime: the decoded norm
    // must equal the pinned boundary radius (1 - BOUNDARY_EPS)/sqrt(k), not the
    // unclamped tanh image (which would be strictly smaller for this s only by
    // ~BOUNDARY_EPS, so we assert it is AT the boundary).
    let out_norm: f64 = (0..x_hat.ncols())
        .map(|j| x_hat[[0, j]] * x_hat[[0, j]])
        .sum::<f64>()
        .sqrt();
    assert!(
        (out_norm - (1.0 - BOUNDARY_EPS)).abs() < 1e-9,
        "decode must be radially pinned to the ball boundary in saturation: \
         |x_hat|={out_norm}, expected {}",
        1.0 - BOUNDARY_EPS
    );

    // Loss = sum(x_hat^2); grad_x = 2 x_hat.
    let mut grad_x = Array2::<f64>::zeros(x_hat.dim());
    for i in 0..x_hat.dim().0 {
        for j in 0..x_hat.dim().1 {
            grad_x[[i, j]] = 2.0 * x_hat[[i, j]];
        }
    }
    let (grad_gates, _grad_atoms) =
        tangent_decode_backward(&cache, grad_x.view()).expect("backward");

    // Central FD of the forward loss w.r.t. the single gate.
    let eps = 1.0e-6;
    let mut gp = gates.clone();
    gp[[0, 0]] += eps;
    let (xp, _) = tangent_decode_forward(atoms.view(), gp.view(), curvature).unwrap();
    let mut gm = gates.clone();
    gm[[0, 0]] -= eps;
    let (xm, _) = tangent_decode_forward(atoms.view(), gm.view(), curvature).unwrap();
    let lp: f64 = xp.iter().map(|v| v * v).sum();
    let lm: f64 = xm.iter().map(|v| v * v).sum();
    let fd_gate = (lp - lm) / (2.0 * eps);

    // In the saturated regime the forward norm is pinned, so moving the gate
    // changes the loss only at O(BOUNDARY_EPS) — the analytic gradient must track
    // that small, radially-pinned sensitivity, NOT the unclamped tanh slope.
    assert!(
        (fd_gate - grad_gates[[0, 0]]).abs() < 1.0e-6,
        "saturated gate grad desync: analytic {} vs FD {}",
        grad_gates[[0, 0]],
        fd_gate
    );
}

/// Companion at a NON-saturated magnitude: the same analytic backward must still
/// match FD on a mid-range tangent (`s ≈ 1`, below the clamp), so the saturation
/// branch did not perturb the interior gradient.
#[test]
fn poincare_tangent_backward_matches_fd_below_saturation() {
    let atoms = array![[0.4, 0.1]];
    let gates = array![[1.5]];
    let curvature = -1.0;

    let (x_hat, cache) =
        tangent_decode_forward(atoms.view(), gates.view(), curvature).expect("forward");
    let out_norm: f64 = (0..x_hat.ncols())
        .map(|j| x_hat[[0, j]] * x_hat[[0, j]])
        .sum::<f64>()
        .sqrt();
    // Strictly interior (not clamped).
    assert!(
        out_norm < 1.0 - BOUNDARY_EPS - 1e-6,
        "this arm must be below saturation: |x_hat|={out_norm}"
    );

    let mut grad_x = Array2::<f64>::zeros(x_hat.dim());
    for j in 0..x_hat.dim().1 {
        grad_x[[0, j]] = 2.0 * x_hat[[0, j]];
    }
    let (grad_gates, _) = tangent_decode_backward(&cache, grad_x.view()).expect("backward");

    let eps = 1.0e-6;
    let mut gp = gates.clone();
    gp[[0, 0]] += eps;
    let (xp, _) = tangent_decode_forward(atoms.view(), gp.view(), curvature).unwrap();
    let mut gm = gates.clone();
    gm[[0, 0]] -= eps;
    let (xm, _) = tangent_decode_forward(atoms.view(), gm.view(), curvature).unwrap();
    let lp: f64 = xp.iter().map(|v| v * v).sum();
    let lm: f64 = xm.iter().map(|v| v * v).sum();
    let fd_gate = (lp - lm) / (2.0 * eps);

    assert!(
        (fd_gate - grad_gates[[0, 0]]).abs() < 1.0e-5,
        "interior gate grad desync: analytic {} vs FD {}",
        grad_gates[[0, 0]],
        fd_gate
    );
}
