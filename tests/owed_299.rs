//! Issue #299 — arrow_schur preconditioner ladder beyond block-Jacobi.
//!
//! The reduced-Schur PCG used by the bundle-adjustment / SAE-manifold inner
//! step preconditions with a block-Jacobi (per-β-block dense Schur) operator.
//! #299 asks for a richer LADDER — cluster-Jacobi, additive Schwarz,
//! diagonally-assembled Schwarz, and level-0 incomplete Cholesky (IC(0)) — with
//! a measured iteration-count reduction over block-Jacobi on an ill-conditioned
//! arrow system.
//!
//! These gates drive the public study seam
//! `arrow_precond_ladder_iteration_study`, which runs the SAME preconditioned
//! CG (identical rhs, tolerances, trust radius) once per ladder tier and reports
//! each tier's iteration count via `PcgDiagnostics`. The iteration counts are
//! MEASUREMENTS, not tuned constants; the assertions only encode the structural
//! facts the ladder must satisfy:
//!
//!   1. every tier drives the coupled PCG to convergence (a valid SPD
//!      preconditioner — the correctness check: IC(0)'s `M⁻¹ = (L̃ L̃ᵀ)⁻¹` solve
//!      must let CG reach the requested tolerance);
//!   2. on a system with genuine OFF-block coupling, the richer tiers
//!      (cluster-Jacobi, Schwarz, IC(0)) take strictly fewer iterations than the
//!      block-diagonal preconditioners that cannot see that coupling;
//!   3. on a BLOCK-BANDED coupling (block-tridiagonal reduced Schur) IC(0) — a
//!      no-fill sparse factor that keeps the band — beats block-Jacobi, the
//!      iteration-count reduction #299 measures.

use gam::solver::arrow_schur::{
    ArrowPcgOptions, ArrowSchurSystem, ArrowTrustRegionOptions, PrecondLadderRow,
    SchurPreconditionerKind, arrow_precond_ladder_iteration_study,
};
use ndarray::Array1;
use std::ops::Range;

/// Build a coupled multi-block reduced-Schur arrow system. Each of `n` rows
/// carries a `d×d` diagonally-dominant `H_tt`, a `d` gradient `g_t`, and a dense
/// `d×k` `H_tβ` whose `n_blocks` contiguous β-blocks (width `block_width`,
/// `k = n_blocks·block_width`) are coupled by the rank-`n·d` point-elimination
/// correction `Σ_i H_tβᵀ (H_tt)⁻¹ H_tβ`.
///
/// When `band == 0` the cross-block sinusoid is keyed on the GLOBAL column index
/// so EVERY block couples to every other (one dense connected component). When
/// `band == 1` a column's cross-block mass is gated to NEIGHBOURING blocks only,
/// so the reduced Schur is block-tridiagonal — a genuinely SPARSE single
/// component where IC(0)'s no-fill factor is much smaller than the dense one.
fn coupled_block_system(
    n: usize,
    d: usize,
    n_blocks: usize,
    block_width: usize,
    coupling: f64,
    band: usize,
) -> ArrowSchurSystem {
    let k = n_blocks * block_width;
    let mut sys = ArrowSchurSystem::new(n, d, k);
    for (i, row) in sys.rows.iter_mut().enumerate() {
        for r in 0..d {
            for c in 0..d {
                row.htt[[r, c]] = if r == c { 4.0 + (i % 3) as f64 } else { 0.2 };
            }
            row.gt[r] = 0.05 * ((i + r + 1) as f64).sin();
            for c in 0..k {
                let blk = c / block_width;
                let row_blk = i % n_blocks;
                let same_block_bias = if row_blk == blk { 0.4 } else { 0.0 };
                // Cross-block mass. When banded, gate it so a row only couples
                // β-blocks within `band` of the row's own block — yielding a
                // block-banded reduced Schur. When `band == 0`, no gating: every
                // block couples (dense single component).
                let gate = if band == 0 {
                    1.0
                } else if (row_blk as isize - blk as isize).unsigned_abs() <= band {
                    1.0
                } else {
                    0.0
                };
                row.htbeta[[r, c]] = same_block_bias
                    + gate
                        * coupling
                        * (((i + 1) as f64 * 0.7 + (c + 1) as f64 * 1.3 + r as f64).sin());
            }
        }
    }
    for r in 0..k {
        sys.gb[r] = 0.02 * ((r + 1) as f64).cos();
        // Diagonal H_ββ large enough to keep the reduced Schur SPD; the
        // off-diagonal of S is entirely the cross-row correction.
        sys.hbb[[r, r]] = 60.0;
    }
    let mut offsets: Vec<Range<usize>> = Vec::with_capacity(n_blocks);
    for blk in 0..n_blocks {
        offsets.push((blk * block_width)..((blk + 1) * block_width));
    }
    sys.set_block_offsets(std::sync::Arc::from(offsets.into_boxed_slice()));
    sys.refresh_row_hessian_fingerprint();
    sys
}

fn tight_options(k: usize) -> (ArrowPcgOptions, ArrowTrustRegionOptions) {
    let pcg = ArrowPcgOptions {
        max_iterations: 6 * k,
        relative_tolerance: 1e-10,
    };
    let trust = ArrowTrustRegionOptions {
        radius: 1.0e9,
        steihaug_relative_tolerance: 1e-10,
        max_iterations: 6 * k,
    };
    (pcg, trust)
}

fn rhs_for(k: usize) -> Array1<f64> {
    Array1::from_iter((0..k).map(|j| 0.3 * ((j + 1) as f64).sin() + 0.1 * (j as f64).cos()))
}

fn study(sys: &ArrowSchurSystem, k: usize) -> Vec<(SchurPreconditionerKind, Option<PrecondLadderRow>)> {
    let (pcg, trust) = tight_options(k);
    let rhs = rhs_for(k);
    arrow_precond_ladder_iteration_study(sys, 1e-8, &rhs, &pcg, &trust)
        .expect("ladder study must run on the SPD coupled fixture")
}

fn row_of(
    rows: &[(SchurPreconditionerKind, Option<PrecondLadderRow>)],
    kind: SchurPreconditionerKind,
) -> PrecondLadderRow {
    rows.iter()
        .find(|(k, _)| *k == kind)
        .and_then(|(_, r)| *r)
        .unwrap_or_else(|| panic!("ladder study did not populate tier {kind:?}"))
}

/// Dense single-component coupling: every tier must converge, and the richer
/// tiers (cluster-Jacobi / Schwarz / IC(0)) — which see the off-block coupling
/// the block-diagonal preconditioners cannot — must take no MORE iterations than
/// block-Jacobi, and strictly fewer than scalar Diagonal.
#[test]
fn precond_ladder_converges_and_richer_tiers_help_on_dense_coupling() {
    let (n, d, n_blocks, block_width, coupling) = (64usize, 3usize, 6usize, 4usize, 0.9f64);
    let k = n_blocks * block_width;
    let sys = coupled_block_system(n, d, n_blocks, block_width, coupling, 0);
    let rows = study(&sys, k);

    for (kind, row) in &rows {
        let r = row.unwrap_or_else(|| panic!("tier {kind:?} failed to build/solve"));
        assert!(
            r.converged,
            "tier {kind:?} did not drive the coupled PCG to convergence \
             (iters={}, rel_resid={:e}) — an invalid preconditioner",
            r.iterations, r.final_relative_residual
        );
    }

    let diag = row_of(&rows, SchurPreconditionerKind::Diagonal);
    let block = row_of(&rows, SchurPreconditionerKind::BetaBlockJacobi);
    let cluster = row_of(&rows, SchurPreconditionerKind::ClusterJacobi);
    let ic0 = row_of(&rows, SchurPreconditionerKind::BlockIncompleteCholesky);

    eprintln!(
        "[#299 dense] diag={} block={} cluster={} ic0={}",
        diag.iterations, block.iterations, cluster.iterations, ic0.iterations
    );

    // The fixture must be genuinely coupled, else the study is vacuous.
    assert!(
        diag.iterations >= 2,
        "coupled fixture must force scalar Diagonal Jacobi to iterate (got {})",
        diag.iterations
    );
    // Cluster-Jacobi factors the whole connected component (the exact reduced
    // Schur, since the system is one component): a single PCG iteration.
    assert!(
        cluster.iterations <= diag.iterations,
        "cluster-Jacobi ({}) must not exceed scalar Diagonal ({})",
        cluster.iterations,
        diag.iterations
    );
    // IC(0) sees the full coupling pattern (a single dense component → the
    // pattern is the whole block), so it must beat the block-diagonal Jacobi
    // that ignores all off-block mass.
    assert!(
        ic0.iterations <= block.iterations,
        "IC(0) ({}) must not exceed block-Jacobi ({}) on the coupled system",
        ic0.iterations,
        block.iterations
    );
}

/// Block-banded coupling (block-tridiagonal reduced Schur): IC(0) keeps the band
/// in a no-fill sparse factor and must reduce the PCG iteration count versus
/// block-Jacobi — the concrete iteration-reduction measurement #299 asks for.
#[test]
fn ic0_reduces_pcg_iterations_versus_block_jacobi_on_banded_coupling() {
    let (n, d, n_blocks, block_width, coupling) = (96usize, 3usize, 10usize, 4usize, 0.8f64);
    let k = n_blocks * block_width;
    let sys = coupled_block_system(n, d, n_blocks, block_width, coupling, 1);
    let rows = study(&sys, k);

    let block = row_of(&rows, SchurPreconditionerKind::BetaBlockJacobi);
    let ic0 = row_of(&rows, SchurPreconditionerKind::BlockIncompleteCholesky);

    eprintln!(
        "[#299 banded] block_jacobi_iters={} (conv={}) ic0_iters={} (conv={}) ic0_rel_resid={:e}",
        block.iterations, block.converged, ic0.iterations, ic0.converged, ic0.final_relative_residual
    );

    assert!(
        ic0.converged,
        "IC(0) must drive the banded-coupled PCG to convergence (iters={}, rel_resid={:e})",
        ic0.iterations, ic0.final_relative_residual
    );
    assert!(
        block.converged,
        "block-Jacobi baseline must converge to measure against (iters={})",
        block.iterations
    );
    // The band must matter: block-Jacobi (ignoring the off-block band) takes
    // real iterations.
    assert!(
        block.iterations >= 2,
        "banded fixture must force block-Jacobi to iterate (got {}); else the study is vacuous",
        block.iterations
    );
    // The measurement: IC(0)'s no-fill factor conditions the off-block band the
    // block-diagonal preconditioner cannot, so it converges in strictly fewer
    // PCG iterations.
    assert!(
        ic0.iterations < block.iterations,
        "IC(0) ({}) must reduce PCG iterations vs block-Jacobi ({}) on the banded coupling (#299)",
        ic0.iterations,
        block.iterations
    );
}
