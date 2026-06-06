//! Bug hunt: `SoftmaxAssignmentSparsityPenalty` exposes a non-PSD "majorizer".
//!
//! The `AnalyticPenalty` trait documents `psd_majorizer_diag` as the diagonal
//! of a PSD majorizer `B` with `B ⪰ ∂²P` everywhere AND `B ⪰ 0`, and states
//! that nonconvex penalties must override it (the default just returns
//! `hessian_diag`).
//!
//! `SoftmaxAssignmentSparsityPenalty` is the SAE soft-assignment entropy prior
//! `λ·Σ_i H(softmax(logits_i))`. Its own doc says the exact Hessian "is dense in
//! each row and can be indefinite because entropy is concave in assignment
//! space, so callers must use the HVP rather than a diagonal Hessian shortcut."
//! Yet:
//!   * it returns `Some(diagonal)` from `hessian_diag`, and
//!   * it does NOT override `psd_majorizer_diag` / `psd_majorizer_hvp`.
//!
//! So the default `psd_majorizer_diag` returns that exact, *indefinite*
//! diagonal, and the default `psd_majorizer_hvp` (which short-circuits on
//! `psd_majorizer_diag = Some`) applies it as `diag ⊙ v` — i.e. PSD consumers
//! receive a curvature block that is both non-PSD and structurally wrong (a
//! diagonal where the true operator is dense).
//!
//! For a near-uniform row (all-equal logits with softmax weight `a = 1/K`), the
//! exact diagonal is `(λ/τ²)·a·((1−2a)(meanL−L_k) + a − 1) = (λ/τ²)·(1/K)·((1−K)/K)`
//! which is strictly negative for `K > 1`. So `psd_majorizer_diag` returns a
//! negative entry, violating `B ⪰ 0`.
//!
//! This must FAIL now (majorizer == exact negative diagonal) and PASS once the
//! penalty supplies a genuine PSD majorizer. Related: the sibling ScadMcp
//! majorizer bug filed in this run, and the (closed) JumpReLU majorizer #796.

use gam::terms::analytic_penalties::{AnalyticPenalty, SoftmaxAssignmentSparsityPenalty};
use ndarray::array;

#[test]
fn softmax_entropy_psd_majorizer_is_actually_psd() {
    let k_atoms = 4usize;
    let temperature = 1.0;
    let pen = SoftmaxAssignmentSparsityPenalty::new(k_atoms, temperature);

    // Two rows, both near-uniform (all-equal logits) so each softmax weight is
    // 1/K and the exact entropy-Hessian diagonal is strictly negative.
    let target = array![0.0_f64, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1];
    let rho = array![0.0_f64]; // weight * exp(0) = 1

    let maj = pen
        .psd_majorizer_diag(target.view(), rho.view())
        .expect("entropy penalty is coordinate-indexed: a diagonal is returned");
    let hess = pen
        .hessian_diag(target.view(), rho.view())
        .expect("entropy penalty exposes an analytic diagonal");

    for i in 0..target.len() {
        // Contract: B ⪰ 0.
        assert!(
            maj[i] >= -1e-12,
            "psd_majorizer_diag[{i}] = {} is negative (B ⪰ 0 violated)",
            maj[i]
        );
        // Contract: B ⪰ ∂²P.
        assert!(
            maj[i] >= hess[i] - 1e-12,
            "psd_majorizer_diag[{i}] = {} is below the exact Hessian {} (B ⪰ ∂²P violated)",
            maj[i],
            hess[i]
        );
    }
}
