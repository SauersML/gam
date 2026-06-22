//! Cross-row analytic-penalty arrow-Schur solve.
//!
//! Regression + correctness test for the cross-row Psi-penalty path in
//! [`gam::solver::arrow_schur`]. Historically the solver REJECTED any analytic
//! Psi-tier penalty whose Hessian couples distinct latent rows
//! (non-row-block-diagonal) with the error "couples latent rows; cross-row
//! Hessian contributions are not yet supported on any production solver path."
//! The arrow elimination folds each per-row `d × d` Hessian into
//! `rows[i].htt` and eliminates the latent block with `N` independent `d × d`
//! solves — an algebra that cannot represent off-row blocks `∂²P/∂t_i∂t_j`
//! (`i ≠ j`).
//!
//! The production path now SUPPORTS them: the penalty gradient is still folded
//! into `g_t`, but its full curvature is applied as a matrix-free
//! Hessian-vector product `P_cross · Δt` over the flat latent vector, and the
//! whole bordered `(t, β)` Newton system is solved by preconditioned CG with
//! the exact arrow block-diagonal inverse as the preconditioner. The route is
//! auto-selected from the presence of a cross-row penalty — no flag.
//!
//! This test drives a small system with a [`TotalVariationPenalty`]
//! (`ForwardDiff1D` over the rows) registered as a Psi-tier analytic penalty
//! and asserts:
//!   1. the solve no longer returns the "couples latent rows" error;
//!   2. the produced Newton step `(Δt, Δβ)` satisfies the FULL Newton
//!      equations `K · [Δt; Δβ] + [g_t; g_β] = 0`, where `K` is built densely
//!      and independently in the test — including the TV cross-row Hessian
//!      block — to a tight relative tolerance.

use gam::solver::arrow_schur::{ArrowSchurError, ArrowSchurSystem};
use gam::terms::analytic_penalties::{
    AnalyticPenalty, AnalyticPenaltyKind, AnalyticPenaltyRegistry, DifferenceOpKind,
    TotalVariationPenalty,
};
use ndarray::{Array1, Array2, ArrayView1};
use std::sync::Arc;

/// Build a small, well-conditioned arrow system: `N` latent rows of dimension
/// `d`, sharing a `k`-dimensional β block. The per-row Gauss–Newton blocks,
/// cross-blocks, and β block are arbitrary but symmetric-positive-definite so
/// the bordered Newton operator is PD before the (PSD) TV curvature is added.
fn build_system(n: usize, d: usize, k: usize) -> ArrowSchurSystem {
    let mut sys = ArrowSchurSystem::new(n, d, k);

    // Shared β block: SPD (diagonally dominant) plus a small dense coupling.
    for a in 0..k {
        for b in 0..k {
            let v = if a == b {
                3.0 + a as f64 * 0.5
            } else {
                0.2 / (1.0 + (a as f64 - b as f64).abs())
            };
            sys.hbb[[a, b]] = v;
        }
    }
    for a in 0..k {
        sys.gb[a] = 0.3 * (a as f64 + 1.0) - 0.7;
    }

    // Per-row blocks.
    for i in 0..n {
        let row = &mut sys.rows[i];
        // H_tt^(i): SPD, row-dependent so the elimination is non-trivial.
        for a in 0..d {
            for b in 0..d {
                let v = if a == b {
                    2.5 + 0.3 * (i as f64) + 0.1 * (a as f64)
                } else {
                    0.15 / (1.0 + (a as f64 - b as f64).abs())
                };
                row.htt[[a, b]] = v;
            }
        }
        // H_tβ^(i): dense cross-block.
        for a in 0..d {
            for c in 0..k {
                row.htbeta[[a, c]] = 0.1 * ((i + 1) as f64) * ((a + 1) as f64) / (1.0 + c as f64)
                    - 0.05 * (a as f64 - c as f64);
            }
        }
        // g_t^(i): the model/likelihood latent gradient (the TV gradient is
        // added on top by `add_analytic_penalty_contributions`).
        for a in 0..d {
            row.gt[a] = 0.4 * ((i + 1) as f64) - 0.2 * (a as f64) + 0.1;
        }
    }

    sys
}

/// Assemble the dense block-diagonal-plus-cross-block arrow operator `K0`
/// (WITHOUT any cross-row penalty) over the stacked `[t; β]` vector of length
/// `N·d + k`. Layout: `t` rows first (row-major `i·d + a`), then β.
fn dense_arrow_operator(sys: &ArrowSchurSystem) -> Array2<f64> {
    let n = sys.rows.len();
    let d = sys.d;
    let k = sys.k;
    let nt = n * d;
    let dim = nt + k;
    let mut a = Array2::<f64>::zeros((dim, dim));

    for i in 0..n {
        let row = &sys.rows[i];
        let base = i * d;
        // H_tt^(i) on the diagonal latent block.
        for p in 0..d {
            for q in 0..d {
                a[[base + p, base + q]] = row.htt[[p, q]];
            }
        }
        // H_tβ^(i) and its transpose H_βt^(i).
        for p in 0..d {
            for c in 0..k {
                let v = row.htbeta[[p, c]];
                a[[base + p, nt + c]] = v;
                a[[nt + c, base + p]] = v;
            }
        }
    }
    // H_ββ block.
    for p in 0..k {
        for q in 0..k {
            a[[nt + p, nt + q]] = sys.hbb[[p, q]];
        }
    }
    a
}

/// Materialize the cross-row penalty Hessian on the latent block by probing the
/// penalty's exact Hessian-vector product against each of the `N·d` standard
/// basis vectors. This builds `P_cross` INDEPENDENTLY of the solver internals,
/// using only the penalty's public `hvp`, so the residual check is a genuine
/// cross-validation of the solver's matrix-free application.
fn dense_cross_row_hessian(
    penalty: &TotalVariationPenalty,
    target_t: ArrayView1<'_, f64>,
    rho_local: ArrayView1<'_, f64>,
    nt: usize,
) -> Array2<f64> {
    let mut p_cross = Array2::<f64>::zeros((nt, nt));
    let mut e = Array1::<f64>::zeros(nt);
    for j in 0..nt {
        e.fill(0.0);
        e[j] = 1.0;
        let col = penalty.hvp(target_t, rho_local, e.view());
        assert_eq!(col.len(), nt, "TV hvp must return a length-N·d vector");
        for i in 0..nt {
            p_cross[[i, j]] = col[i];
        }
    }
    p_cross
}

#[test]
fn cross_row_total_variation_solve_satisfies_full_newton_equations() {
    let n = 5usize;
    let d = 2usize;
    let k = 3usize;
    let nt = n * d;

    let mut sys = build_system(n, d, k);

    // Current latent iterate (flat row-major, N·d) the penalty curvature is
    // linearized at. Non-trivial so the smoothed-TV Hessian weights vary per
    // edge and are genuinely cross-row.
    let mut target_t = Array1::<f64>::zeros(nt);
    for i in 0..n {
        for a in 0..d {
            target_t[i * d + a] =
                0.5 * (i as f64) - 0.3 * (a as f64) + 0.2 * ((i * d + a) as f64).sin();
        }
    }
    let target_beta = Array1::<f64>::from(vec![0.1, -0.2, 0.05]);

    // A smoothed-L¹ total-variation penalty over the rows (forward 1-D
    // differences) — its Hessian Dᵀ diag(φ''(D t)) D couples adjacent rows, so
    // it is NOT row-block-diagonal and exercises the cross-row solver path.
    let tv = TotalVariationPenalty::new(
        0.8, // weight
        n,   // n_eff = number of rows
        DifferenceOpKind::ForwardDiff1D,
        1e-2,  // smoothing_eps
        false, // fixed weight (no learnable rho axis)
    )
    .expect("TV penalty constructs");
    assert!(
        !<TotalVariationPenalty as gam::terms::analytic_penalties::PenaltyManifest>::ROW_BLOCK_DIAGONAL,
        "TV must be classified cross-row for this test to exercise the new path"
    );

    let tv_arc = Arc::new(tv);
    let penalty_kind = AnalyticPenaltyKind::TotalVariation(Arc::clone(&tv_arc));
    assert!(
        !penalty_kind.is_row_block_diagonal(),
        "the registered penalty kind must report cross-row coupling"
    );

    let mut registry = AnalyticPenaltyRegistry::new();
    registry.push(penalty_kind);

    // The penalty owns no learnable ρ-axis, so the global ρ vector is empty.
    let rho_global = Array1::<f64>::zeros(registry.total_rho_count());

    // Fold the penalty: gradient into g_t, curvature captured as a cross-row
    // operator. This must NOT return the legacy "couples latent rows" error.
    let result = sys.add_analytic_penalty_contributions(
        &registry,
        target_t.view(),
        target_beta.view(),
        rho_global.view(),
    );
    assert!(
        result.is_ok(),
        "add_analytic_penalty_contributions must succeed for a cross-row TV penalty, got {result:?}"
    );
    assert_eq!(
        sys.cross_row_penalties.len(),
        1,
        "the TV penalty must be captured as a cross-row penalty"
    );

    // Capture the FULL latent gradient g_t AFTER folding (model gt + TV grad),
    // and the β gradient g_β. These define the Newton RHS.
    let mut g_full = Array1::<f64>::zeros(nt + k);
    for i in 0..n {
        for a in 0..d {
            g_full[i * d + a] = sys.rows[i].gt[a];
        }
    }
    for c in 0..k {
        g_full[nt + c] = sys.gb[c];
    }

    // Independent cross-check that the captured g_t equals the model gradient
    // plus the penalty's own grad_target (no curvature leaked into g_t).
    {
        let tv_grad = tv_arc.grad_target(target_t.view(), rho_global.view());
        // Rebuild a fresh system to read the pristine model gradient.
        let baseline = build_system(n, d, k);
        for i in 0..n {
            for a in 0..d {
                let expected = baseline.rows[i].gt[a] + tv_grad[i * d + a];
                let got = sys.rows[i].gt[a];
                assert!(
                    (expected - got).abs() <= 1e-12 + 1e-9 * expected.abs(),
                    "g_t[{i},{a}] should be model grad + TV grad: expected {expected}, got {got}"
                );
            }
        }
    }

    // Solve the Newton step. With zero ridge this is the exact Newton system.
    let (delta_t, delta_beta, diag) = sys
        .solve(0.0, 0.0)
        .expect("cross-row arrow-Schur solve must succeed");
    assert_eq!(delta_t.len(), nt);
    assert_eq!(delta_beta.len(), k);
    assert!(
        diag.iterations >= 1,
        "the cross-row full-system CG must take at least one iteration"
    );

    // Build the dense full Newton operator K = K0 + P_cross independently.
    let mut k_dense = dense_arrow_operator(&sys);
    let p_cross = dense_cross_row_hessian(&tv_arc, target_t.view(), rho_global.view(), nt);
    for i in 0..nt {
        for j in 0..nt {
            k_dense[[i, j]] += p_cross[[i, j]];
        }
    }

    // Stack the solved step [Δt; Δβ].
    let dim = nt + k;
    let mut step = Array1::<f64>::zeros(dim);
    for i in 0..nt {
        step[i] = delta_t[i];
    }
    for c in 0..k {
        step[nt + c] = delta_beta[c];
    }

    // Residual of the full Newton equations: r = K · step + g.
    // The solver returns the increments satisfying K · step = −g, so the
    // residual must be ~0.
    let mut residual = Array1::<f64>::zeros(dim);
    for i in 0..dim {
        let mut acc = g_full[i];
        for j in 0..dim {
            acc += k_dense[[i, j]] * step[j];
        }
        residual[i] = acc;
    }

    let res_norm = residual.iter().map(|v| v * v).sum::<f64>().sqrt();
    let g_norm = g_full.iter().map(|v| v * v).sum::<f64>().sqrt();
    let rel = res_norm / g_norm.max(1e-300);
    assert!(
        rel <= 1e-6,
        "full Newton residual ‖K·[Δt;Δβ] + g‖ / ‖g‖ = {rel:e} exceeds 1e-6 \
         (‖r‖={res_norm:e}, ‖g‖={g_norm:e}); the cross-row solve does not \
         satisfy the true Hessian system"
    );

    // P_cross must genuinely couple distinct rows (otherwise the test would be
    // vacuous — a diagonal penalty would pass through the legacy path). Assert
    // at least one off-row block entry is non-negligible.
    let mut has_off_row = false;
    for i in 0..n {
        for jrow in 0..n {
            if i == jrow {
                continue;
            }
            for a in 0..d {
                for b in 0..d {
                    if p_cross[[i * d + a, jrow * d + b]].abs() > 1e-8 {
                        has_off_row = true;
                    }
                }
            }
        }
    }
    assert!(
        has_off_row,
        "the TV cross-row Hessian must have non-zero off-row blocks for this test to be meaningful"
    );
}

/// Guard the magic-by-default auto-selection: a system with NO cross-row
/// penalty must NOT capture anything and must keep the exact one-shot Schur
/// path (no cross-row penalties registered).
#[test]
fn row_block_diagonal_only_system_keeps_exact_schur_path() {
    let n = 4usize;
    let d = 2usize;
    let k = 2usize;
    let mut sys = build_system(n, d, k);

    let registry = AnalyticPenaltyRegistry::new();
    let target_t = Array1::<f64>::zeros(n * d);
    let target_beta = Array1::<f64>::zeros(k);
    let rho_global = Array1::<f64>::zeros(0);

    sys.add_analytic_penalty_contributions(
        &registry,
        target_t.view(),
        target_beta.view(),
        rho_global.view(),
    )
    .expect("empty registry folds trivially");

    assert!(
        sys.cross_row_penalties.is_empty(),
        "no cross-row penalty must be captured when none is registered"
    );

    let solved = sys.solve(0.0, 0.0);
    assert!(
        solved.is_ok(),
        "the row-block-diagonal exact Schur path must still solve: {:?}",
        solved.err()
    );
}

/// Sanity: the legacy rejection error must no longer be constructible from this
/// path — a cross-row penalty solve returns `Ok`, never an `ArrowSchurError`
/// describing "couples latent rows". This pins the limitation removal.
#[test]
fn cross_row_penalty_does_not_return_couples_latent_rows_error() {
    let n = 3usize;
    let d = 2usize;
    let k = 2usize;
    let mut sys = build_system(n, d, k);

    let tv = TotalVariationPenalty::new(0.5, n, DifferenceOpKind::ForwardDiff1D, 1e-2, false)
        .expect("TV constructs");
    let mut registry = AnalyticPenaltyRegistry::new();
    registry.push(AnalyticPenaltyKind::TotalVariation(Arc::new(tv)));

    let target_t = Array1::<f64>::from(vec![0.1, -0.2, 0.3, 0.0, -0.1, 0.4]);
    let target_beta = Array1::<f64>::zeros(k);
    let rho_global = Array1::<f64>::zeros(registry.total_rho_count());

    sys.add_analytic_penalty_contributions(
        &registry,
        target_t.view(),
        target_beta.view(),
        rho_global.view(),
    )
    .expect("cross-row fold must succeed");

    match sys.solve(0.0, 0.0) {
        Ok(_) => {}
        Err(ArrowSchurError::PcgFailed { reason }) => {
            panic!("cross-row solve unexpectedly failed: {reason}");
        }
        Err(other) => panic!("cross-row solve returned an unexpected error: {other:?}"),
    }
}
