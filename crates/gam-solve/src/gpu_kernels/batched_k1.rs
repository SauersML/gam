//! Batched independent K=1 border solves — the post-SAC GPU arrow-Schur target.
//!
//! SAC's backfitting phase (SAC_PLAN Phase 2) refits each atom against its
//! leave-one-out residual. Atoms whose supports do not overlap are mutually
//! independent, so a *color class* of support-disjoint atoms is a batch of `B`
//! small, independent K=1 arrow/border systems — embarrassingly parallel and a
//! far better match for the B200s than the retired giant joint `K × K` system
//! (see `BATCHED_K1_DESIGN.md` in this directory, especially §7 for why the
//! monolith is no longer a GPU target).
//!
//! This module is the seam A2's `stagewise.rs` backfitting sweep calls. It ships
//! today with:
//!   * the dispatch entry [`solve_batched_k1_border`],
//!   * the CPU reference path, which is ALSO the bit-parity oracle the device
//!     kernel will be validated against, and
//!   * the per-atom decline contract (a capability mismatch on one atom is served
//!     by the CPU reference for that atom; a genuine numerical PD failure is
//!     returned per-atom so the caller can bump that atom's ridge).
//!
//! The device batched kernel (one thread-block per atom over the CSR pack of
//! `BATCHED_K1_DESIGN.md` §3) is the follow-up: [`try_device_batched_k1`] is where
//! `cuda::solve_batched_k1` will attach. Until it lands, every class declines to
//! the CPU reference, so the numbers are correct on every host and the device path
//! is purely an acceleration of an already-validated result.

use crate::arrow_schur::ArrowSchurSystem;
use crate::gpu_kernels::arrow_schur::{
    ArrowSchurGpuFailure, ArrowSchurGpuSolution, solve_arrow_newton_step_dense_reference,
};

/// Refit a color class of mutually support-disjoint K=1 atoms concurrently.
///
/// `systems[a]` is atom `a`'s leave-one-out arrow/border system (assembled by the
/// caller, who owns the residual/Σ state). Returns one result per atom,
/// positionally aligned with `systems`: coloring guarantees independence, so the
/// results compose without interaction and the caller writes each accepted
/// `(δt, δβ)` straight back into its atom's chart/gate.
///
/// A per-atom capability decline (a matrix-free atom, or a device transient) is
/// served transparently by the CPU reference — that element is still `Ok`. Only a
/// genuine numerical PD failure yields a per-atom `Err`, which the caller escalates
/// by re-calling with that atom's bumped ridge. This mirrors the recoverable-vs-
/// fatal split the single-system device seams use, so a non-dense atom can never
/// surface a fatal `RemlConvergenceError`.
#[must_use]
pub fn solve_batched_k1_border(
    systems: &[ArrowSchurSystem],
    ridge_t: f64,
    ridge_beta: f64,
) -> Vec<Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure>> {
    if let Some(batched) = try_device_batched_k1(systems) {
        return batched;
    }
    systems
        .iter()
        .map(|sys| cpu_reference_k1(sys, ridge_t, ridge_beta))
        .collect()
}

/// One atom's K=1 border solve on the CPU. This is the canonical bit-parity oracle
/// the device batched kernel is validated against, and the fallback whenever the
/// device declines the class or a single atom. A non-PD/rank-deficient system
/// surfaces as [`ArrowSchurGpuFailure::SchurFactorFailed`], the same numerical
/// variant the caller's LM escalation responds to with a ridge bump.
fn cpu_reference_k1(
    sys: &ArrowSchurSystem,
    ridge_t: f64,
    ridge_beta: f64,
) -> Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure> {
    solve_arrow_newton_step_dense_reference(sys, ridge_t, ridge_beta)
        .map_err(|reason| ArrowSchurGpuFailure::SchurFactorFailed { reason })
}

/// Device admission for the batched color class. Applies the same work-based
/// offload floor the single-system reduced-Schur paths use, keyed on the class's
/// AGGREGATE active-row mass and mean border width (CG budget 1: a K=1 Direct
/// solve is a single factor, not a CG loop — see `BATCHED_K1_DESIGN.md` §5).
/// Returns `None` to decline the whole class to the CPU reference. Off Linux
/// there is no CUDA path: `GpuRuntime::global()` is `None` and the class
/// declines through the same admission flow.
///
/// The batched per-atom device kernel is not yet attached, so an admitted class
/// still declines to the CPU reference rather than fabricate a step: this function
/// is the seam where `cuda::solve_batched_k1` will produce the per-atom results
/// (and where the caller's ridge pair re-enters the signature once consumed).
fn try_device_batched_k1(
    systems: &[ArrowSchurSystem],
) -> Option<Vec<Result<ArrowSchurGpuSolution, ArrowSchurGpuFailure>>> {
    if systems.is_empty() {
        return None;
    }
    let runtime = gam_gpu::device_runtime::GpuRuntime::global()?;
    let total_rows: usize = systems.iter().map(|s| s.rows.len()).sum();
    let mean_k = systems.iter().map(|s| s.k).sum::<usize>() / systems.len();
    let max_d = systems.iter().map(|s| s.d).max().unwrap_or(0);
    if !runtime
        .policy()
        .reduced_schur_matvec_should_offload(total_rows, mean_k, max_d, 1)
    {
        return None;
    }
    // Admitted by the work floor, but the batched kernel is the follow-up; decline
    // to the CPU reference (bit-parity oracle) rather than emit an unvalidated step.
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A well-posed K=1 arrow/border atom: PD per-row blocks, a PD border, and a
    /// small deterministic cross-block, so the dense reference solves cleanly.
    fn pd_k1_system(n: usize, d: usize, k: usize, seed: f64) -> ArrowSchurSystem {
        let mut sys = ArrowSchurSystem::new(n, d, k);
        for (i, row) in sys.rows.iter_mut().enumerate() {
            for r in 0..d {
                row.htt[[r, r]] = 2.0 + seed;
                row.gt[r] = 0.1 * (i as f64 + 1.0) + seed;
                for c in 0..k {
                    row.htbeta[[r, c]] = 0.05 * ((r + c + i) as f64 + 1.0);
                }
            }
        }
        for r in 0..k {
            sys.hbb[[r, r]] = 2.0 + seed;
            sys.gb[r] = 0.2 * (r as f64 + 1.0) + seed;
        }
        sys
    }

    /// A K=1 atom with a negative-definite per-row block: the guarded Cholesky in
    /// the dense reference hits a negative pivot and returns an error, standing in
    /// for a per-atom numerical decline.
    fn indefinite_k1_system(n: usize, d: usize, k: usize) -> ArrowSchurSystem {
        let mut sys = pd_k1_system(n, d, k, 0.0);
        for row in sys.rows.iter_mut() {
            for r in 0..d {
                row.htt[[r, r]] = -2.0;
            }
        }
        sys
    }

    fn assert_solution_eq(a: &ArrowSchurGpuSolution, b: &ArrowSchurGpuSolution) {
        assert_eq!(a.delta_t.len(), b.delta_t.len());
        assert_eq!(a.delta_beta.len(), b.delta_beta.len());
        for (x, y) in a.delta_t.iter().zip(b.delta_t.iter()) {
            assert!((x - y).abs() < 1e-12, "delta_t mismatch: {x} vs {y}");
        }
        for (x, y) in a.delta_beta.iter().zip(b.delta_beta.iter()) {
            assert!((x - y).abs() < 1e-12, "delta_beta mismatch: {x} vs {y}");
        }
    }

    #[test]
    fn empty_class_returns_empty() {
        let out = solve_batched_k1_border(&[], 1e-6, 1e-6);
        assert!(out.is_empty(), "an empty color class must return no results");
    }

    #[test]
    fn single_atom_matches_dense_reference() {
        let sys = pd_k1_system(5, 2, 3, 0.0);
        let batched = solve_batched_k1_border(std::slice::from_ref(&sys), 1e-6, 1e-6);
        assert_eq!(batched.len(), 1);
        let reference = solve_arrow_newton_step_dense_reference(&sys, 1e-6, 1e-6)
            .expect("PD reference atom must solve");
        let got = batched[0].as_ref().expect("batched single atom must solve");
        assert_solution_eq(got, &reference);
    }

    #[test]
    fn class_results_are_positional_and_independent() {
        // Three atoms with distinct data: the batched result for each must equal
        // the atom solved on its own (support-disjoint atoms do not interact).
        let systems = [
            pd_k1_system(4, 2, 2, 0.0),
            pd_k1_system(6, 2, 3, 0.5),
            pd_k1_system(3, 1, 2, 1.0),
        ];
        let batched = solve_batched_k1_border(&systems, 1e-6, 1e-6);
        assert_eq!(batched.len(), systems.len());
        for (idx, sys) in systems.iter().enumerate() {
            let alone = solve_arrow_newton_step_dense_reference(sys, 1e-6, 1e-6)
                .expect("each PD atom must solve on its own");
            let in_class = batched[idx].as_ref().expect("each atom must solve in-class");
            assert_solution_eq(in_class, &alone);
        }
    }

    #[test]
    fn per_atom_decline_is_isolated_never_fatal() {
        // A class mixing a PD atom with a numerically-declining atom: the decline
        // must be confined to its own element (a returned `Err`), never fail the
        // class or panic. This is the contract that keeps a non-dense/decline atom
        // from ever escalating into a fatal RemlConvergenceError.
        let systems = [pd_k1_system(4, 2, 2, 0.0), indefinite_k1_system(4, 2, 2)];
        let batched = solve_batched_k1_border(&systems, 0.0, 0.0);
        assert_eq!(batched.len(), 2);
        assert!(batched[0].is_ok(), "the PD atom must still solve in a mixed class");
        // Only the `Err` variants are `Debug` (the `Ok` solution type is not), so
        // match rather than format the whole `Result`.
        match &batched[1] {
            Err(ArrowSchurGpuFailure::SchurFactorFailed { .. }) => {}
            Err(other) => panic!("the indefinite atom must decline as SchurFactorFailed; got {other:?}"),
            Ok(_) => panic!("the indefinite atom must decline per-atom, but it solved"),
        }
    }
}
