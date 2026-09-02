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

