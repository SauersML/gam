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
//! This had to FAIL when it was filed (majorizer == exact negative diagonal) and to PASS once the
//! penalty supplied a genuine PSD majorizer. It does supply one now: `psd_majorizer_diag` is
//! overridden with the Gershgorin / diagonal-dominance majorizer of the dense per-row block
//! (`psd_majorizer_abs_row_sums`), a diagonal with `D ⪰ H` and `D ⪰ 0`. So this file is the pin on
//! that repair, not an open bug. Related: the sibling ScadMcp majorizer bug filed in the same run,
//! and the (closed) smooth-threshold majorizer #796.

use gam::terms::analytic_penalties::{AnalyticPenalty, SoftmaxAssignmentSparsityPenalty};
use ndarray::{Array1, array};

#[test]
fn softmax_entropy_psd_majorizer_is_actually_psd() {
    let k_atoms = 4usize;
    let temperature = 1.0;
    let pen = SoftmaxAssignmentSparsityPenalty::new(k_atoms, temperature);

    // Two rows, both near-uniform (all-equal logits) so each softmax weight is
    // 1/K and the exact entropy-Hessian diagonal is strictly negative.
    let target = array![0.0_f64, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1];
    // The entropy strength is the penalty's own `weight`, never an outer coordinate (#4291): on
    // this chart — the full `(N, K)` logit matrix — the energy is bounded and shift-invariant, so
    // `∫exp(−λ·ΣH) dℓ` diverges for every `λ` and there is no prior mass to price. `new` leaves
    // `weight = 1`, which is the strength the retired `rho = [0]` used to name.
    let rho = Array1::<f64>::zeros(0);

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
