//! #2023 Increment 5a (Stage 1, dense) — route the co-fit's linear tier through the
//! unified arrow-Schur inner solver instead of the hand-rolled block coordinate
//! descent.
//!
//! The thesis (#2232): there is ONE joint solver. The block-sparse linear tier a
//! `BlockSparseFit` produces is a set of `d = b` Euclidean (linear) atoms of that
//! solver — a linear atom is the degree-1 / `b₂ = 0` special case of the curved
//! atom. This module builds those linear atoms + a frozen-support assignment from a
//! block routing `(decoder, blocks, codes, γ)` and runs
//! [`SaeManifoldTerm::run_joint_fit_arrow_schur_for_quasi_laplace`] on them, then
//! reads the composed reconstruction + the fixed-point certificate back out.
//!
//! **Stage 1 scope (this file).** DENSE assignment, moderate `K`: the parity
//! evidence the fold needs does not require the massive-`K` support-sparse state
//! (risk #1 of the design applies to PRODUCTION routing, not the parity fixtures).
//! The behaviour-parity claim is that the arrow-routed linear fit **matches or
//! beats** the direct block-sparse linear reconstruction in explained variance —
//! the joint solve descends the same linear model, warm-started from the same
//! routing, so it cannot do worse. The support-sparse in-core seam (an engine
//! entry consuming `SaeAssignmentState::from_topk_support`) is Stage 2, specced to
//! the engine lane on #2023 — this module never touches the driver internals, it
//! only CALLS `SaeManifoldTerm`.
//!
//! **Why this module lives under [`crate::manifold`] and NOT under
//! [`crate::sparse_dict`] (#2693 / #985 E1).** It is a caller of the DENSE
//! manifold engine: the engine's only constructor is
//! [`SaeManifoldTerm::new(atoms, assignment)`](SaeManifoldTerm::new), whose
//! `assignment` is a [`SaeAssignment`] — the dense `N x K` routing state, stored
//! as an `Array2<f64>` of logits. Entering the dense engine therefore REQUIRES
//! materializing that state; there is no sparse in-core entry (that is the Stage 2
//! `SaeAssignmentState::from_topk_support` seam, still specced to the engine lane
//! on #2023). `sparse_dict` is defined by "no dense `N x K` object anywhere", and
//! `sparse_lane_constructs_no_dense_assignment` locks that. So this bridge from a
//! block routing into the dense certification engine belongs on the dense side of
//! the boundary, and it consumes the sparse lane's public surface
//! ([`crate::sparse_dict`]) rather than living inside it.
//!
//! **#2275 certificate reconciliation.** The joint solver's
//! `EvidenceJointFitOutcome.fixed_point` is a construction gate, not report
//! telemetry: `false` returns a non-convergence error, so every
//! [`ArrowCofitReport`] necessarily came from an idempotent re-entry.

use ndarray::Array2;

use crate::sparse_dict::BlockChartComposeConfig;

/// Result of the arrow-Schur-routed co-fit linear tier (Stage 1).
#[derive(Clone, Debug)]
pub struct ArrowCofitReport {
    /// Composed reconstruction `N×P` read back from the fitted term.
    pub reconstructed: Array2<f32>,
    /// Explained variance of `reconstructed` against the target (mean-baseline via
    /// the shared `explained_variance_from_reconstruction` helper cofit uses).
    pub explained_variance: f64,
    /// Number of curved (periodic) atoms folded into the joint solve — the count
    /// of blocks whose BIC-gated chart discovery ([`compose_block_coordinate_charts`])
    /// promoted them from a flat linear atom to a curved chart. `0` for the
    /// linear-only path ([`cofit_linear_via_arrow`]). This is the curved-birth
    /// count the migration ledger banks.
    pub n_curved_atoms: usize,
    /// Total BIC complexity charge (`Σ ½·d_eff·ln n_eff`, nats) of the curved
    /// charts folded in — the description-length currency the ledger records as
    /// `dl_bits`. `0.0` for the linear-only path.
    pub curved_charge: f64,
}

/// Tuning for the arrow-routed linear fit. `max_iter` must be generous enough for
/// the evidence policy to settle: a too-small iteration budget is a
/// non-convergence error and can never produce an [`ArrowCofitReport`].
#[derive(Clone, Debug)]
pub struct ArrowCofitConfig {
    pub log_lambda_sparse: f64,
    pub log_lambda_smooth: f64,
    pub max_iter: usize,
    pub step_size: f64,
    pub ridge_ext_coord: f64,
    pub ridge_beta: f64,
    /// Number of periodic-harmonic basis columns `M = 2·h + 1` for a folded
    /// curved atom (must be odd, `>= 3`). `3` is one harmonic — an exact circle,
    /// the ring the chart-discovery lane certifies. Used only by
    /// [`cofit_composed_via_arrow`].
    pub curved_num_basis: usize,
    /// Chart-discovery configuration passed to [`compose_block_coordinate_charts`]
    /// to decide WHICH blocks fold in as curved atoms. Its `block_size` /
    /// `block_topk` / `gamma` are overwritten from the passed routing so the tiers
    /// always agree on geometry. Used only by [`cofit_composed_via_arrow`].
    pub chart: BlockChartComposeConfig,
}

impl Default for ArrowCofitConfig {
    fn default() -> Self {
        Self {
            log_lambda_sparse: (1.0e-4f64).ln(),
            log_lambda_smooth: (1.0e-4f64).ln(),
            max_iter: 128,
            step_size: 1.0,
            ridge_ext_coord: 1.0e-6,
            ridge_beta: 1.0e-6,
            curved_num_basis: 3,
            chart: BlockChartComposeConfig::default(),
        }
    }
}

