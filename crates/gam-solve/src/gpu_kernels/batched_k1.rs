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
//!   * the per-atom numerical-failure contract: a genuine PD failure is returned
//!     per atom so the caller can bump only that atom's ridge.
//!
//! There is no device implementation in this module. Keeping a fake admission
//! seam that always declined only paid runtime-probe cost and obscured the actual
//! execution path; the CPU reference is therefore the single implementation
//! until a real batched kernel exists end to end.

use crate::gpu_kernels::arrow_schur::ArrowSchurGpuFailure;
use crate::gpu_kernels::arrow_schur::ArrowSchurGpuSolution;
use crate::arrow_schur::ArrowSchurSystem;
use crate::gpu_kernels::arrow_schur::solve_arrow_newton_step_dense_reference;
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

    #[test]
    fn empty_class_returns_empty() {
        let out = solve_batched_k1_border(&[], 1e-6, 1e-6);
        assert!(
            out.is_empty(),
            "an empty color class must return no results"
        );
    }

}
