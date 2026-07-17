//! Tests for identifiability-audit drift.
//!
//! Verifies that:
//!
//! 1. The drift detection log fires when the rank verdict changes between
//!    the pilot β and a non-trivial β
//!    (audit_drift_detection_logs_when_rank_changes).
//! 2. The drift detection is silent when the rank verdict is stable
//!    (audit_drift_silent_when_rank_stable).
//!
//! Architecture: all tests are self-contained — no survival family construction,
//! no cargo --release, no parallel cargo.

use gam::families::custom_family::ParameterBlockSpec;
use gam::identifiability::audit::{IdentifiabilityAudit, maybe_log_audit_drift};
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::{Array1, Array2};

// ── helpers ────────────────────────────────────────────────────────────────

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

// ── Audit drift when rank changes ─────────────────────────────────────────

/// Build a minimal audit result with a given set of dropped columns.
fn fake_audit(effective_dims: &[usize], original_dims: &[usize]) -> IdentifiabilityAudit {
    use gam::identifiability::audit::{BlockIdentity, DroppedColumn};
    let blocks: Vec<BlockIdentity> = original_dims
        .iter()
        .zip(effective_dims.iter())
        .enumerate()
        .map(|(i, (&orig, &eff))| BlockIdentity {
            block_name: format!("block_{i}"),
            original_dim: orig,
            effective_dim: eff,
            design_range_rank: eff,
            // Populated only for within-block rank deficiency; these fake
            // audit blocks are full-rank by construction.
            singular_spectrum: String::new(),
        })
        .collect();
    // Pilot drops: any column where effective < original.
    let dropped_columns: Vec<DroppedColumn> = original_dims
        .iter()
        .zip(effective_dims.iter())
        .enumerate()
        .flat_map(|(bi, (&orig, &eff))| {
            (eff..orig).map(move |col| DroppedColumn {
                block: format!("block_{bi}"),
                column: col,
                reason: "test".to_string(),
            })
        })
        .collect();
    IdentifiabilityAudit {
        blocks,
        aliased_pairs: Vec::new(),
        dropped_columns,
        fatal: false,
        summary: "test audit".to_string(),
    }
}

/// The drift detection should fire (return Some(summary)) when the rank
/// verdict changes.  We create a simple synthetic setup where two single-
/// column blocks share the same raw design column (so the flat audit would
/// detect aliasing at the current β), but the pilot audit says rank 2/2.
/// Because `maybe_log_audit_drift` re-runs the flat audit at beta_current,
/// it will detect the drop.
///
/// Note: we verify via the returned `AuditDriftSummary`, not by inspecting
/// log output (test harness doesn't capture log).
#[test]
fn audit_drift_detection_fires_when_rank_changes() {
    let n: usize = 32;

    // Two identical single-column blocks → the flat audit at current β
    // drops one of them (joint rank = 1, not 2).
    let col: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
    let design = Array2::from_shape_vec((n, 1), col).unwrap();

    let specs = vec![
        spec_from_dense("block_a", design.clone()),
        spec_from_dense("block_b", design.clone()),
    ];

    // Pilot audit says rank 2 (no drops — this is the "before" state).
    let pilot_audit = fake_audit(&[1, 1], &[1, 1]);
    let beta_pilot = vec![0.0_f64; 2];

    // Current β differs significantly: relative change = ‖[5,5]‖/ε >> 0.5.
    let beta_current = vec![5.0_f64, 5.0_f64];

    // The drift check threshold: relative change 5/ε >> 0.5, so the
    // large-beta-movement branch fires immediately.
    let summary = maybe_log_audit_drift(
        &specs,
        &pilot_audit,
        &beta_pilot,
        &beta_current,
        None,
        0, // outer_iter
        10,
        1.0,
    )
    .expect("current-state audit must succeed");

    // Drift re-audit finds rank 1 (one drop) vs pilot rank 2 → summary present.
    let summary = summary.expect("audit_drift_detection_fires_when_rank_changes: expected Some");
    assert_eq!(summary.pilot_rank, 2, "pilot_rank should be 2");
    assert!(
        summary.current_rank < 2,
        "current_rank should be < 2 (got {}); the re-audit at current β should find the alias",
        summary.current_rank,
    );
}

// ── Stable audit rank ─────────────────────────────────────────────────────

/// When the rank verdict is stable (no change between pilot and current β),
/// `maybe_log_audit_drift` should still return `Some` (the check ran), but
/// the summary shows no newly_dropped or recovered, and pilot_rank ==
/// current_rank.
///
/// We use two linearly independent columns so the flat audit always says
/// rank 2/2 regardless of β.
#[test]
fn audit_drift_silent_when_rank_stable() {
    let n: usize = 16;

    // Two linearly independent single-column blocks.
    let col_a: Vec<f64> = (0..n).map(|i| (i as f64) + 1.0).collect();
    let col_b: Vec<f64> = (0..n).map(|i| ((i + 1) as f64).sin()).collect();
    let design_a = Array2::from_shape_vec((n, 1), col_a).unwrap();
    let design_b = Array2::from_shape_vec((n, 1), col_b).unwrap();

    let specs = vec![
        spec_from_dense("block_a", design_a),
        spec_from_dense("block_b", design_b),
    ];

    // Pilot audit: rank 2/2, no drops.
    let pilot_audit = fake_audit(&[1, 1], &[1, 1]);
    let beta_pilot = vec![0.0_f64; 2];
    // Large relative β change (> 0.5 threshold) to force the re-audit.
    let beta_current = vec![10.0_f64, 10.0_f64];

    let summary = maybe_log_audit_drift(
        &specs,
        &pilot_audit,
        &beta_pilot,
        &beta_current,
        None,
        0,
        10,
        1.0,
    )
    .expect("current-state audit must succeed");

    let summary = summary.expect("audit_drift_silent_when_rank_stable: expected Some");
    assert_eq!(
        summary.pilot_rank, summary.current_rank,
        "pilot_rank and current_rank should be equal when rank is stable \
         (pilot={} current={})",
        summary.pilot_rank, summary.current_rank,
    );
    assert!(
        summary.newly_dropped.is_empty(),
        "no newly dropped columns expected when rank is stable"
    );
    assert!(
        summary.recovered.is_empty(),
        "no recovered columns expected when rank is stable"
    );
}
