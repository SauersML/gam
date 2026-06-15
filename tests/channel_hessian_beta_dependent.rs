//! Tests for the β-dependent channel Hessian refresh (T34).
//!
//! Verifies that:
//!
//! 1. `SurvivalRowHessian::channel_hessian_at` returns a different W when
//!    the primary state changes (survival_marginal_slope_w_changes_with_beta).
//! 2. The drift detection log fires when the rank verdict changes between
//!    the pilot β and a non-trivial β
//!    (audit_drift_detection_logs_when_rank_changes).
//! 3. The drift detection is silent when the rank verdict is stable
//!    (audit_drift_silent_when_rank_stable).
//! 4. The TensorChannelHessian (β-independent default path) returns the
//!    same W regardless of β (gaussian_identity_w_is_beta_independent).
//!
//! Architecture: all tests are self-contained — no survival family construction,
//! no cargo --release, no parallel cargo.

use gam::families::custom_family::{FamilyChannelHessian, ParameterBlockSpec};
use gam::families::survival_marginal_slope_identifiability::SurvivalRowHessian;
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam::solver::identifiability_audit::{IdentifiabilityAudit, maybe_log_audit_drift};
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

/// Build a minimal `SurvivalRowHessian` from explicit primary-state arrays.
/// Uses n=4 rows so the test stays tiny.
fn make_survival_row_hessian(
    q0: &[f64],
    q1: &[f64],
    qd1: &[f64],
    g: &[f64],
    z: &[f64],
) -> SurvivalRowHessian {
    let n = q0.len();
    let q0_arr = Array1::from_vec(q0.to_vec());
    let q1_arr = Array1::from_vec(q1.to_vec());
    let qd1_arr = Array1::from_vec(qd1.to_vec());
    let g_arr = Array1::from_vec(g.to_vec());
    let z_arr = Array1::from_vec(z.to_vec());
    let weights = Array1::from_elem(n, 1.0_f64);
    let event = Array1::from_elem(n, 1.0_f64);
    SurvivalRowHessian::from_pilot_primary_state(
        &q0_arr, &q1_arr, &qd1_arr, &g_arr, &z_arr, &weights, &event,
        1e-6, // derivative_guard
        1.0,  // probit_scale
    )
    .expect("make_survival_row_hessian: from_pilot_primary_state failed")
}

// ── T34 test 1: W changes with β ──────────────────────────────────────────

/// `channel_hessian_at(beta_b, scalars_b)` must differ from
/// `channel_hessian_at(beta_a, scalars_a)` when the primary state changes.
///
/// We build two primary-state configurations that differ substantially in
/// `g_i`: state A has g=0 (β=0 pilot), state B has g=2.0 (non-trivial β).
/// At g=0 the cross-channel off-diagonal terms W[0,3] = W[3,0] etc. are zero
/// (q·c1 + s·z = 0 when g=0 → c1=0). At g=2 those cross-terms are non-zero.
/// So W_A ≠ W_B in the off-diagonal entries, and the test asserts that at
/// least one entry differs by more than 1e-6.
#[test]
fn survival_marginal_slope_w_changes_with_beta() {
    use gam::families::survival_marginal_slope::SurvivalMarginalSlopeFamilyScalars;
    use std::sync::Arc;

    let n = 4;

    // State A: pilot β=0 → g=0, c=1, q0=q1=qd1=0.5.
    let hess_a = make_survival_row_hessian(
        &[0.5, 0.5, 0.5, 0.5],   // q0
        &[0.5, 0.5, 0.5, 0.5],   // q1
        &[1.0, 1.0, 1.0, 1.0],   // qd1 (must be > derivative_guard)
        &[0.0, 0.0, 0.0, 0.0],   // g = 0
        &[1.0, -1.0, 0.5, -0.5], // z
    );

    // W at pilot β: call channel_hessian_at with beta=all zeros, scalars=None.
    let w_a = hess_a
        .channel_hessian_at(&[0.0_f64; 0], None)
        .expect("channel_hessian_at beta_a failed");
    let tensor_a = w_a.evaluate_full();

    // State B: non-trivial β → g=2.0. Build scalars.
    let g_b = vec![2.0_f64; n];
    let q0_b = vec![0.5_f64; n];
    let q1_b = vec![0.5_f64; n];
    let qd1_b = vec![1.0_f64; n];
    let z_b = vec![1.0, -1.0, 0.5, -0.5];
    let s_b = 1.0_f64;
    let scalars_b =
        SurvivalMarginalSlopeFamilyScalars::new(q0_b, q1_b, qd1_b, g_b.clone(), s_b, z_b.clone());
    let scalars_arc: Arc<dyn std::any::Any + Send + Sync> = Arc::new(scalars_b);

    // Build a hessian from the pilot primary state (same as hess_a but we
    // want to call channel_hessian_at on a fresh instance with the B scalars).
    let hess_for_b = make_survival_row_hessian(
        &[0.5, 0.5, 0.5, 0.5],
        &[0.5, 0.5, 0.5, 0.5],
        &[1.0, 1.0, 1.0, 1.0],
        &[0.0, 0.0, 0.0, 0.0],
        &[1.0, -1.0, 0.5, -0.5],
    );

    let beta_b = vec![1.0_f64; 1]; // non-trivial beta
    let w_b = hess_for_b
        .channel_hessian_at(&beta_b, Some(&scalars_arc))
        .expect("channel_hessian_at beta_b failed");
    let tensor_b = w_b.evaluate_full();

    // W_a and W_b must differ: at g=0 the off-diagonal W[0,3]=W[3,0] etc.
    // are zero; at g=2 they are non-zero. Assert at least one entry differs
    // by more than 1e-6.
    let max_diff = tensor_a
        .iter()
        .zip(tensor_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    assert!(
        max_diff > 1e-6,
        "W(beta_a) and W(beta_b) should differ for distinct primary states: max_diff={max_diff}"
    );
}

// ── T34 test 2: drift detected when rank changes ───────────────────────────

/// Build a minimal audit result with a given set of dropped columns.
fn fake_audit(effective_dims: &[usize], original_dims: &[usize]) -> IdentifiabilityAudit {
    use gam::solver::identifiability_audit::{BlockIdentity, DroppedColumn};
    let blocks: Vec<BlockIdentity> = original_dims
        .iter()
        .zip(effective_dims.iter())
        .enumerate()
        .map(|(i, (&orig, &eff))| BlockIdentity {
            block_name: format!("block_{i}"),
            original_dim: orig,
            effective_dim: eff,
            design_range_rank: eff,
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
    );

    // Drift re-audit finds rank 1 (one drop) vs pilot rank 2 → summary present.
    let summary = summary.expect("audit_drift_detection_fires_when_rank_changes: expected Some");
    assert_eq!(summary.pilot_rank, 2, "pilot_rank should be 2");
    assert!(
        summary.current_rank < 2,
        "current_rank should be < 2 (got {}); the re-audit at current β should find the alias",
        summary.current_rank,
    );
}

// ── T34 test 3: drift silent when rank stable ──────────────────────────────

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
    );

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

// ── T34 test 4: Gaussian-identity W is β-independent ─────────────────────

/// The default `channel_hessian_at` implementation (used by families whose
/// W is β-independent) must return the same tensor regardless of β.
///
/// We construct a `TensorChannelHessian` (the default-impl wrapper) with a
/// known 2×2 per-subject tensor and verify that two calls with different β
/// produce identical outputs.
#[test]
fn gaussian_identity_w_is_beta_independent() {
    use gam::families::custom_family::TensorChannelHessian;

    let n = 4;
    let k = 2;

    // Build a (4, 2, 2) tensor with non-trivial values.
    let mut h = ndarray::Array3::<f64>::zeros((n, k, k));
    for i in 0..n {
        h[[i, 0, 0]] = (i as f64) + 1.0;
        h[[i, 1, 1]] = (i as f64) + 2.0;
        h[[i, 0, 1]] = 0.1;
        h[[i, 1, 0]] = 0.1;
    }

    let w = TensorChannelHessian { h: h.clone() };

    // Call 1: beta = [0, 0], no scalars.
    let w_at_0 = w
        .channel_hessian_at(&[0.0, 0.0], None)
        .expect("channel_hessian_at beta=0 failed");
    let tensor_0 = w_at_0.evaluate_full();

    // Call 2: beta = [5.0, -3.0], no scalars (still β-independent).
    let w_at_b = w
        .channel_hessian_at(&[5.0, -3.0], None)
        .expect("channel_hessian_at beta=[5,-3] failed");
    let tensor_b = w_at_b.evaluate_full();

    // Both must equal the original tensor exactly.
    let max_diff_0 = h
        .iter()
        .zip(tensor_0.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff_0 < 1e-14,
        "TensorChannelHessian at beta=0 differs from original: max_diff={max_diff_0}"
    );

    let max_diff_b = h
        .iter()
        .zip(tensor_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff_b < 1e-14,
        "TensorChannelHessian at beta=[5,-3] differs from original: max_diff={max_diff_b}"
    );

    // Also verify the two are identical to each other (β-independence).
    let max_diff_ab = tensor_0
        .iter()
        .zip(tensor_b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff_ab < 1e-14,
        "TensorChannelHessian should be identical at different β values; max_diff={max_diff_ab}"
    );
}
