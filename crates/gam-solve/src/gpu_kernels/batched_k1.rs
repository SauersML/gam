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

