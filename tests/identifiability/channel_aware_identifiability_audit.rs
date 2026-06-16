//! Channel-aware identifiability audit regression test.
//!
//! Demonstrates that
//! `gam::identifiability::audit::audit_identifiability_channel_aware`
//! correctly handles the multi-channel survival scenario where two
//! blocks share IDENTICAL raw-X column vectors at the training rows
//! but contribute to ORTHOGONAL `K`-channels of the row Jacobian. The
//! flat `audit_identifiability` reports those columns as fatal hard-
//! aliases; the channel-aware path correctly classifies them as
//! structurally identifiable.
//!
//! Layout: two single-column blocks. Block A's row Jacobian writes to
//! channel 0; block B's row Jacobian writes to channel 1. The raw
//! design column for both blocks is the same length-`n` vector
//! `[1, 2, ..., n]`.

use gam::families::custom_family::ParameterBlockSpec;
use gam::identifiability::audit::{audit_identifiability, audit_identifiability_channel_aware};
use gam::identifiability::families::compiler::{IdentityRowHessian, RowJacobianOperator};
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::{Array1, Array2, Array3};
use std::sync::Arc;

const N: usize = 64;

/// A K=2 row Jacobian operator carrying a single coefficient that
/// contributes to exactly one of the two channels.
struct SingleChannelOperator {
    design: Array2<f64>,
    channel: usize,
}

impl SingleChannelOperator {
    fn new(design: Array2<f64>, channel: usize) -> Self {
        assert!(channel < 2);
        Self { design, channel }
    }
}

impl RowJacobianOperator for SingleChannelOperator {
    fn k(&self) -> usize {
        2
    }
    fn ncols(&self) -> usize {
        self.design.ncols()
    }
    fn nrows(&self) -> usize {
        self.design.nrows()
    }
    fn apply_row(&self, row: usize, delta_beta: &[f64], out: &mut [f64]) {
        assert_eq!(out.len(), 2);
        let mut acc = 0.0_f64;
        for (j, &b) in delta_beta.iter().enumerate() {
            acc += self.design[[row, j]] * b;
        }
        out[0] = 0.0;
        out[1] = 0.0;
        out[self.channel] = acc;
    }
    fn evaluate_full(&self) -> Array3<f64> {
        let n = self.design.nrows();
        let p = self.design.ncols();
        let k = self.k();
        let mut out = Array3::<f64>::zeros((n, p, k));
        for i in 0..n {
            for j in 0..p {
                out[[i, j, self.channel]] = self.design[[i, j]];
            }
        }
        out
    }
}

fn spec_from_dense(name: &str, design: Array2<f64>) -> ParameterBlockSpec {
    let n = design.nrows();
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::<f64>::zeros(n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

fn shared_column_block(name: &str) -> ParameterBlockSpec {
    let mut design = Array2::<f64>::zeros((N, 1));
    for i in 0..N {
        design[[i, 0]] = (i as f64) + 1.0;
    }
    spec_from_dense(name, design)
}

#[test]
fn flat_audit_flags_identical_raw_columns_as_fatal() {
    let specs = [
        shared_column_block("block_a"),
        shared_column_block("block_b"),
    ];
    let audit = audit_identifiability(&specs).expect("flat audit must run");
    assert!(
        audit.fatal,
        "flat audit must flag identical raw-X columns as fatal hard-alias; \
         summary: {}",
        audit.summary,
    );
}

#[test]
fn channel_aware_audit_passes_when_blocks_live_in_orthogonal_channels() {
    let specs = [
        shared_column_block("block_a"),
        shared_column_block("block_b"),
    ];

    let design_a = match &specs[0].design {
        DesignMatrix::Dense(d) => d.to_dense(),
        _ => panic!("expected dense"),
    };
    let design_b = match &specs[1].design {
        DesignMatrix::Dense(d) => d.to_dense(),
        _ => panic!("expected dense"),
    };

    // Block A writes to channel 0; block B writes to channel 1.
    // Their raw-X columns are IDENTICAL but the row Jacobian
    // contributions are orthogonal.
    let operators: Vec<Arc<dyn RowJacobianOperator>> = vec![
        Arc::new(SingleChannelOperator::new(design_a, 0)),
        Arc::new(SingleChannelOperator::new(design_b, 1)),
    ];
    let row_hess = IdentityRowHessian::new(N, 2);

    let audit = audit_identifiability_channel_aware(&specs, &operators, &row_hess)
        .expect("channel-aware audit must run");

    assert!(
        !audit.fatal,
        "channel-aware audit must NOT flag orthogonal-channel blocks as fatal; \
         summary: {}",
        audit.summary,
    );
    assert!(
        audit.dropped_columns.is_empty(),
        "no columns should be dropped when blocks live in orthogonal channels; \
         got: {:?}",
        audit.dropped_columns,
    );
    assert!(
        audit.aliased_pairs.is_empty(),
        "no alias pairs should be reported when blocks live in orthogonal channels; \
         got: {:?}",
        audit.aliased_pairs,
    );
}

#[test]
fn channel_aware_audit_still_flags_real_aliases_within_one_channel() {
    // Two blocks BOTH in channel 0 with identical raw-X — this IS a
    // true alias in the channel-aware view and must be reported as
    // fatal.
    let specs = [shared_column_block("a"), shared_column_block("b")];
    let design_a = match &specs[0].design {
        DesignMatrix::Dense(d) => d.to_dense(),
        _ => panic!("expected dense"),
    };
    let design_b = match &specs[1].design {
        DesignMatrix::Dense(d) => d.to_dense(),
        _ => panic!("expected dense"),
    };
    let operators: Vec<Arc<dyn RowJacobianOperator>> = vec![
        Arc::new(SingleChannelOperator::new(design_a, 0)),
        Arc::new(SingleChannelOperator::new(design_b, 0)),
    ];
    let row_hess = IdentityRowHessian::new(N, 2);
    let audit = audit_identifiability_channel_aware(&specs, &operators, &row_hess)
        .expect("channel-aware audit must run");
    assert!(
        audit.fatal,
        "real same-channel alias must be fatal in the channel-aware view; \
         summary: {}",
        audit.summary,
    );
}

#[test]
fn channel_aware_audit_accepts_intra_block_rank_deficiency() {
    // Block A (channel 0) is full-rank.  Block B (channel 1) carries a
    // genuinely rank-deficient design: column 0 is independent, columns
    // 1..p are exact copies of column 0 — INTRA-block aliased, NOT
    // cross-block aliased (channels are disjoint, so no pairwise
    // cross-block alias can be reported by the channel-aware scan).
    //
    // The pre-fix audit treated this as fatal because joint_rank <
    // joint_cols AND the gauge_priority resolution path only fires when
    // dropped columns are attributed to a LOWER-priority block of a
    // detected cross-block alias pair — a condition that pure intra-block
    // deficiency by construction cannot satisfy.  Canonicalisation
    // already drops the redundant columns and the fit proceeds on the
    // reduced basis; the audit must surface this as NON-fatal.
    //
    // This mirrors the large-scale GAMLSS jointpc failure mode: shared
    // covariates between the threshold and log-sigma designs leave the
    // log-sigma block with most columns linearly dependent (after the
    // scale-deviation reparameterisation residualises against the
    // threshold span), and the channel-aware view correctly sees that
    // the deficiency is intra-block — there are no above-threshold
    // cross-block alias pairs because the two channels are structurally
    // disjoint by construction.
    let p_b: usize = 4;
    let mut design_a = Array2::<f64>::zeros((N, 1));
    let mut design_b = Array2::<f64>::zeros((N, p_b));
    for i in 0..N {
        design_a[[i, 0]] = (i as f64).sin();
        let v = (i as f64).cos();
        for j in 0..p_b {
            design_b[[i, j]] = v;
        }
    }
    let specs = [
        spec_from_dense("a", design_a.clone()),
        spec_from_dense("b", design_b.clone()),
    ];
    let operators: Vec<Arc<dyn RowJacobianOperator>> = vec![
        Arc::new(SingleChannelOperator::new(design_a, 0)),
        Arc::new(SingleChannelOperator::new(design_b, 1)),
    ];
    let row_hess = IdentityRowHessian::new(N, 2);
    let audit = audit_identifiability_channel_aware(&specs, &operators, &row_hess)
        .expect("channel-aware audit must run");
    assert!(
        audit.aliased_pairs.is_empty(),
        "blocks live in orthogonal channels — no cross-block alias pair \
         should be reported; got {:?}",
        audit.aliased_pairs,
    );
    // Per-block intra-block redundancy is absorbed by the compiler's
    // `t_lw` reduction (eigenspace-keep), not surfaced via
    // `dropped_columns` — the latter only carries trailing-pivot RRQR
    // drops that fire when the JOINT design (after per-block reduction)
    // is still rank-deficient. Channel-disjoint blocks have no cross-
    // block residual mass, so the trailing-pivot RRQR is clean and
    // `dropped_columns` is empty; the intra-block deficiency lives in
    // the difference between `b.original_dim` and `b.effective_dim`.
    let intra_block_reduction: usize = audit
        .blocks
        .iter()
        .map(|b| b.original_dim.saturating_sub(b.effective_dim))
        .sum();
    assert!(
        intra_block_reduction > 0,
        "block B has 3 redundant copies of column 0; the channel-aware \
         compiler must reduce its effective_dim below original_dim. \
         blocks: {:?}",
        audit.blocks,
    );
    assert!(
        !audit.fatal,
        "pure intra-block rank deficiency (no cross-block alias above \
         leverage-based report threshold) must NOT be fatal; the inner \
         penalised solve uses the reduced basis surfaced via t_lw and \
         the penalty pull-back. summary: {}",
        audit.summary,
    );
}

#[test]
fn channel_aware_audit_clean_when_blocks_are_distinct() {
    // Different raw-X, different channels: no aliasing, no drops.
    let mut design_a = Array2::<f64>::zeros((N, 1));
    let mut design_b = Array2::<f64>::zeros((N, 1));
    for i in 0..N {
        design_a[[i, 0]] = (i as f64).sin();
        design_b[[i, 0]] = (i as f64).cos();
    }
    let specs = [
        spec_from_dense("a", design_a.clone()),
        spec_from_dense("b", design_b.clone()),
    ];
    let operators: Vec<Arc<dyn RowJacobianOperator>> = vec![
        Arc::new(SingleChannelOperator::new(design_a, 0)),
        Arc::new(SingleChannelOperator::new(design_b, 1)),
    ];
    let row_hess = IdentityRowHessian::new(N, 2);
    let audit = audit_identifiability_channel_aware(&specs, &operators, &row_hess)
        .expect("channel-aware audit must run");
    assert!(
        !audit.fatal,
        "clean distinct blocks must not be fatal; summary: {}",
        audit.summary,
    );
}
