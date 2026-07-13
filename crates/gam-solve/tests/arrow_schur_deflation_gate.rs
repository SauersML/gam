//! Gate for the deflation-aware β-Schur selected inverse
//! (`ArrowFactorCache::schur_inverse_apply_deflated` /
//! `schur_inverse_block_deflated`) — the shared primitive the λ→0 REML EDF fix
//! and the Path-C log-det HVP both consume.
//!
//! Two arms:
//!   (a) INTERIOR — a well-conditioned S_β: no eigen-direction sits below the
//!       canonical rank floor, so zero deflation fires and the deflated selected
//!       inverse reproduces the plain one to round-off (no silent bias).
//!   (b) BOUNDARY — an S_β with one doubly-null (data-null AND penalty-null)
//!       curvature direction: the PLAIN selected inverse divides by the ~zero
//!       pivot and blows up (the exact λ→0 EDF divergence); the DEFLATED one
//!       drops that direction and stays finite and bounded, equal to the
//!       pseudo-inverse over the kept subspace.
//!
//! Lives as a standalone integration test (only the gam-solve public API +
//! this one binary) so it is immune to unrelated tears in the crate's `#[cfg(test)]`
//! unit-test modules.

use gam_solve::arrow_schur::{
    ArrowSchurSystem, ArrowSolveOptions, solve_arrow_newton_step_with_options,
};
use ndarray::{Array1, array};

/// `|a - b| <= tol` with a helpful panic message.
fn close(a: f64, b: f64, tol: f64, what: &str) {
    assert!(
        (a - b).abs() <= tol,
        "{what}: {a} vs {b} (|Δ|={})",
        (a - b).abs()
    );
}

#[test]
fn deflated_selected_inverse_finite_at_boundary_matches_plain_interior() {
    // --- (a) INTERIOR: deflated == plain on a well-conditioned system. ---
    {
        let n = 3usize;
        let d = 2usize;
        let k = 2usize;
        let mut sys = ArrowSchurSystem::new(n, d, k);
        sys.rows[0].htt = array![[4.0_f64, 0.5], [0.5, 3.0]];
        sys.rows[0].htbeta = array![[1.0_f64, 0.2], [-0.3, 0.7]];
        sys.rows[1].htt = array![[5.0_f64, -0.4], [-0.4, 2.5]];
        sys.rows[1].htbeta = array![[0.6_f64, -0.1], [0.4, 0.9]];
        sys.rows[2].htt = array![[3.5_f64, 0.2], [0.2, 4.5]];
        sys.rows[2].htbeta = array![[-0.2_f64, 0.5], [0.8, -0.6]];
        for row in sys.rows.iter_mut() {
            row.gt = array![0.0_f64, 0.0];
        }
        sys.hbb = array![[12.0_f64, 0.7], [0.7, 10.0]];
        sys.gb = array![0.0_f64, 0.0];

        let options = ArrowSolveOptions::direct();
        let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
            .expect("direct arrow solve should factor this SPD system");

        for col in 0..k {
            let mut e = Array1::<f64>::zeros(k);
            e[col] = 1.0;
            let plain = cache.schur_inverse_apply(e.view()).expect("plain apply");
            let deflated = cache
                .schur_inverse_apply_deflated(e.view())
                .expect("deflated apply");
            for r in 0..k {
                close(
                    deflated[r],
                    plain[r],
                    1e-9,
                    "interior deflated vs plain apply",
                );
            }
        }
        let plain_block = cache.schur_inverse_block(0..k).expect("plain block");
        let deflated_block = cache
            .schur_inverse_block_deflated(0..k)
            .expect("deflated block");
        for r in 0..k {
            for c in 0..k {
                close(
                    deflated_block[[r, c]],
                    plain_block[[r, c]],
                    1e-9,
                    "interior deflated vs plain block",
                );
            }
        }
    }

    // --- (b) BOUNDARY: S_β has one strong (λ≈10) and one doubly-null (λ≈1e-10)
    // eigen-direction. With htbeta = 0 the reduced Schur is exactly hbb, whose
    // eigenpairs are v = [1,1]/√2 (λ=10) and u = [1,-1]/√2 (λ=1e-10). ---
    {
        let n = 1usize;
        let d = 1usize;
        let k = 2usize;
        let mut sys = ArrowSchurSystem::new(n, d, k);
        sys.rows[0].htt = array![[1.0_f64]];
        sys.rows[0].htbeta = array![[0.0_f64, 0.0]];
        sys.rows[0].gt = array![0.0_f64];
        // hbb = 10·v vᵀ + 1e-10·u uᵀ, with v=[1,1]/√2, u=[1,-1]/√2.
        let big = 10.0_f64;
        let tiny = 1.0e-10_f64;
        let diag = 0.5 * (big + tiny);
        let offd = 0.5 * (big - tiny);
        sys.hbb = array![[diag, offd], [offd, diag]];
        sys.gb = array![0.0_f64, 0.0];

        let options = ArrowSolveOptions::direct();
        let (_dt, _db, cache) = solve_arrow_newton_step_with_options(&sys, 0.0, 0.0, &options)
            .expect("near-singular but PD hbb still factors");

        let rhs = array![1.0_f64, 0.0];

        // PLAIN selected inverse divides by the ~1e-10 pivot: the u-component of
        // the solution is ~1e10 — the divergence the EDF trace hits at λ→0.
        let plain = cache.schur_inverse_apply(rhs.view()).expect("plain apply");
        let plain_norm = plain.dot(&plain).sqrt();
        assert!(
            plain_norm > 1.0e8,
            "plain selected inverse must blow up on the doubly-null direction, got norm {plain_norm}"
        );

        // DEFLATED selected inverse drops the null direction: M⁺ rhs =
        // (1/10)·(vᵀrhs)·v, finite. vᵀrhs = 1/√2 ⇒ result = [0.05, 0.05].
        let deflated = cache
            .schur_inverse_apply_deflated(rhs.view())
            .expect("deflated apply");
        assert!(
            deflated.iter().all(|x| x.is_finite()),
            "deflated selected inverse must be finite, got {deflated:?}"
        );
        close(deflated[0], 0.05, 1e-9, "boundary deflated apply[0]");
        close(deflated[1], 0.05, 1e-9, "boundary deflated apply[1]");

        // Block form: M⁺ = (1/10)·v vᵀ = 0.1·[[.5,.5],[.5,.5]] — every entry 0.05.
        let block = cache
            .schur_inverse_block_deflated(0..k)
            .expect("deflated block");
        for r in 0..k {
            for c in 0..k {
                close(block[[r, c]], 0.05, 1e-9, "boundary deflated block");
            }
        }
    }
}
