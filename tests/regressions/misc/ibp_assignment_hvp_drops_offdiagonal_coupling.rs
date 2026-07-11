//! Bug hunt: `OrderedBetaBernoulliPenalty::hvp` returns only `diag(H) ⊙ v`, dropping
//! the entire off-diagonal block of its Hessian — so the value it returns is
//! **not** the Hessian-vector product `H v` the trait contract promises.
//!
//! The `AnalyticPenalty::hvp` doc (src/terms/analytic_penalties/mod.rs:370-380)
//! is explicit:
//!
//!     "Hessian-vector product `H v = (∂²P/∂target²) v`, in closed form.
//!      The default covers every penalty whose Hessian is diagonal: it reads
//!      the analytic `hessian_diag` and forms `diag ⊙ v`. Penalties with a
//!      dense (non-diagonal) Hessian … return `None` from `hessian_diag` and
//!      supply their own analytic `hvp` override."
//!
//! `OrderedBetaBernoulliPenalty` violates that contract. Its penalty value couples
//! **every row within a column** through the per-column aggregate
//! `active_mass[k] = Σ_row z[row, k]` that drives `pi_map`
//! (src/terms/analytic_penalties/mod.rs:2177-2192, used in `value` at 2200-2219).
//! Because `pi[k]` is a function of all rows in column `k`, the cross-row second
//! derivatives `∂²P / ∂target[i,k] ∂target[j,k]` (i ≠ j) are nonzero — the
//! Hessian is block-diagonal *per column* but **dense within each column**.
//!
//! Yet `OrderedBetaBernoulliPenalty` implements `hessian_diag` (returns `Some`,
//! src/terms/analytic_penalties/mod.rs:2265-2321) and supplies **no** `hvp`
//! override. So `hvp` falls through to the trait default
//! (src/terms/analytic_penalties/mod.rs:381-407), which forms `diag(H) ⊙ v` and
//! silently discards the within-column coupling. Empirically that coupling is
//! the *dominant* part of the operator (≈85% of the Hessian's Frobenius norm),
//! so the returned vector is far from `H v`.
//!
//! Any consumer that treats `hvp` as a genuine Hessian-vector product — the
//! inner Newton / preconditioner / curvature path of the SAE assignment
//! optimization — gets curvature that ignores exactly the shared-feature
//! coupling the finite-ordered independent Beta--Bernoulli prior exists to model.
//!
//! The reference below is a central finite difference of `grad_target` (whose
//! correctness is independently checkable against `value`): by definition
//! `H v = d/dε grad(target + ε v)|₀`. The diagonal of the analytic `hvp`
//! agrees with the FD diagonal, but the full vectors differ by ~0.12 here —
//! orders of magnitude beyond finite-difference error. The assertion encodes
//! the trait contract `hvp(t, v) == H v`; it fails today and will pass once
//! `OrderedBetaBernoulliPenalty` supplies a closed-form dense (within-column) `hvp`.
//!
//! Related: #804, #805, #796, #794 (sibling SAE-penalty curvature defects).

use gam::terms::analytic_penalties::{AnalyticPenalty, OrderedBetaBernoulliPenalty};
use ndarray::Array1;

#[test]
fn ordered_beta_bernoulli_assignment_hvp_equals_true_hessian_vector_product() {
    let k_max = 3usize;
    // n_eff = 4 rows are implied by the 12-element target below (len / k_max).

    // alpha = 1, tau = 1, non-learnable so the ρ vector is empty.
    let penalty = OrderedBetaBernoulliPenalty::new(k_max, 1.0, 1.0, false);
    let rho = Array1::<f64>::zeros(0);

    // Target laid out row-major with `k_max` interleaved columns:
    // index = row * k_max + k. Moderate magnitudes keep z = σ(target/τ) and
    // the per-column π away from their clamp guards, so the Hessian is the
    // genuine smooth second derivative (no clamp-induced kinks).
    let target = Array1::from(vec![
        0.3, -0.4, 0.7, // row 0
        -0.2, 0.5, -0.6, // row 1
        0.1, 0.8, -0.3, // row 2
        -0.5, 0.2, 0.4, // row 3
    ]);
    let v = Array1::from(vec![
        0.5, -0.3, 0.2, //
        0.4, 0.1, -0.6, //
        -0.2, 0.7, 0.3, //
        0.6, -0.4, 0.1, //
    ]);

    let analytic_hv = penalty.hvp(target.view(), rho.view(), v.view());

    // True Hessian-vector product via a central difference of the (correct)
    // analytic gradient: H v = d/dε grad(target + ε v) |_{ε=0}.
    let h = 1e-5_f64;
    let target_plus = &target + &(&v * h);
    let target_minus = &target - &(&v * h);
    let grad_plus = penalty.grad_target(target_plus.view(), rho.view());
    let grad_minus = penalty.grad_target(target_minus.view(), rho.view());
    let fd_hv = (&grad_plus - &grad_minus) / (2.0 * h);

    let max_abs_diff = analytic_hv
        .iter()
        .zip(fd_hv.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    // Central-difference truncation error on a smooth penalty at h = 1e-5 is
    // ~1e-9; a tolerance of 1e-5 is generous and still far below the true
    // off-diagonal coupling (~0.1) the diagonal-only hvp drops.
    assert!(
        max_abs_diff < 1e-5,
        "OrderedBetaBernoulliPenalty::hvp does not equal the true Hessian-vector \
         product: max|hvp - H·v| = {max_abs_diff:.6e}. The default diagonal-only \
         hvp drops the within-column (row-coupling) off-diagonal Hessian block.\n\
         analytic hvp = {analytic_hv:?}\n\
         true   H·v   = {fd_hv:?}"
    );
}
