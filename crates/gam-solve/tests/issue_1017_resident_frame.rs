//! Public-path regression for issue #1017's LM-ladder SAE residency lifetime.

use std::sync::{Arc, Mutex};

use gam_solve::arrow_schur::{
    ArrowPcgDiagnostics, ArrowSchurSystem, ArrowSolveOptions, DEFAULT_PROXIMAL_INITIAL_RIDGE,
    DeviceSaePcgData, solve_with_lm_escalation_inner,
};
use gam_solve::gpu_kernels::arrow_schur::{ArrowSchurGpuFailure, SaeResidentFrame};
use ndarray::{Array1, Array2};

struct RecordingFrame {
    calls: Arc<Mutex<Vec<(f64, f64)>>>,
}

impl SaeResidentFrame for RecordingFrame {
    fn resolve(
        &self,
        sys: &ArrowSchurSystem,
        ridge_t: f64,
        ridge_beta: f64,
        rhs_beta: &Array1<f64>,
        max_iterations: usize,
        relative_tolerance: f64,
    ) -> Result<(Array1<f64>, ArrowPcgDiagnostics), ArrowSchurGpuFailure> {
        assert_eq!(
            rhs_beta.len(),
            sys.k,
            "resident resolve must receive the beta-block RHS"
        );
        assert!(
            max_iterations > 0 && relative_tolerance > 0.0,
            "resident resolve must receive a live PCG budget"
        );
        let mut calls = self.calls.lock().expect("record resident resolves");
        calls.push((ridge_t, ridge_beta));
        if calls.len() == 1 {
            return Err(ArrowSchurGpuFailure::RidgeBumpRequired {
                row: 0,
                bump: DEFAULT_PROXIMAL_INITIAL_RIDGE,
            });
        }
        Ok((
            Array1::<f64>::zeros(sys.k),
            ArrowPcgDiagnostics {
                matvec_calls: 73,
                ..ArrowPcgDiagnostics::default()
            },
        ))
    }
}

fn device_sae_fixture() -> ArrowSchurSystem {
    let mut sys = ArrowSchurSystem::new(2, 1, 2);
    for row in &mut sys.rows {
        row.htt[[0, 0]] = 2.0;
    }
    sys.hbb = Array2::<f64>::eye(sys.k) * 3.0;
    sys.set_device_sae_pcg_data(DeviceSaePcgData {
        p: 1,
        beta_dim: sys.k,
        a_phi: Arc::from(vec![Vec::new(), Vec::new()].into_boxed_slice()),
        local_jac: Arc::from(vec![vec![0.0], vec![0.0]].into_boxed_slice()),
        smooth_blocks: Vec::new(),
        sparse_g_blocks: Vec::new(),
        frame: None,
    });
    sys.refresh_row_hessian_fingerprint();
    sys
}

/// The production large-border mode (`InexactPCG`) must consume one
/// ladder-scoped SAE resident frame across every recoverable LM ridge trial.
///
/// A recording frame makes the lifetime contract CPU-observable. Its first
/// resolve reports a non-PD row block, forcing the public escalation entry to
/// retry at the next ridge; the same frame then returns a marker step. If the
/// InexactPCG branch bypasses the resident frame, or escalation drops it between
/// trials, the marker diagnostics and call ledger cannot be produced.
#[test]
fn inexact_pcg_reuses_one_sae_resident_frame_across_lm_ridge_trials_1017() {
    let sys = device_sae_fixture();
    let calls = Arc::new(Mutex::new(Vec::new()));
    let mut options = ArrowSolveOptions::inexact_pcg();
    options.sae_resident_frame = Some(Arc::new(RecordingFrame {
        calls: Arc::clone(&calls),
    }));

    let (_delta_t, _delta_beta, diagnostics) =
        solve_with_lm_escalation_inner(&sys, 0.0, 0.0, &options)
            .expect("second resolve on the same resident frame succeeds");

    assert_eq!(diagnostics.ridge_escalations, 1);
    assert_eq!(
        diagnostics.matvec_calls, 73,
        "the successful step must come from the resident InexactPCG resolve"
    );
    assert_eq!(
        calls
            .lock()
            .expect("inspect resident resolve ledger")
            .as_slice(),
        &[
            (0.0, 0.0),
            (
                DEFAULT_PROXIMAL_INITIAL_RIDGE,
                DEFAULT_PROXIMAL_INITIAL_RIDGE,
            ),
        ],
        "one resident frame must receive both ridge trials in ladder order"
    );
}
