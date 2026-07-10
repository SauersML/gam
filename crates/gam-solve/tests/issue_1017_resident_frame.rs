//! Public-path regression for issue #1017's LM-ladder SAE residency lifetime.

use std::sync::{Arc, Mutex};

use gam_solve::arrow_schur::{
    ArrowPcgDiagnostics, ArrowSchurSystem, ArrowSolveOptions, DEFAULT_PROXIMAL_INITIAL_RIDGE,
    DeviceSaeFrameData, DeviceSaePcgData, prepare_sae_resident_frame,
    solve_with_lm_escalation_inner,
};
use gam_solve::gpu_kernels::arrow_schur::{ArrowSchurGpuFailure, SaeResidentFrame};
use ndarray::{Array1, Array2};

struct RecordingFrame {
    calls: Arc<Mutex<Vec<(f64, f64)>>>,
    refreshes: Arc<Mutex<Vec<f64>>>,
}

impl SaeResidentFrame for RecordingFrame {
    fn refresh(&self, sys: &ArrowSchurSystem) -> Result<(), ArrowSchurGpuFailure> {
        self.refreshes
            .lock()
            .expect("record resident refreshes")
            .push(sys.rows[0].htt[[0, 0]]);
        Ok(())
    }

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

fn framed_device_data(row_value: f64) -> DeviceSaePcgData {
    DeviceSaePcgData {
        p: 1,
        beta_dim: 2,
        a_phi: Arc::from(Vec::new().into_boxed_slice()),
        local_jac: Arc::from(Vec::new().into_boxed_slice()),
        smooth_blocks: Vec::new(),
        sparse_g_blocks: Vec::new(),
        frame: Some(DeviceSaeFrameData {
            ranks: vec![1],
            basis_sizes: vec![2],
            border_offsets: vec![0],
            frame_blocks: Vec::new(),
            smooth_ranks: Vec::new(),
            row_htbeta: vec![vec![row_value; 2]; 2],
        }),
    }
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
        refreshes: Arc::new(Mutex::new(Vec::new())),
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

/// A caller retaining the frame across nonlinear assemblies must keep the same
/// frame allocation while refreshing it from each new numerical system.
#[test]
fn nonlinear_prepare_keeps_frame_identity_and_refreshes_current_content_1017() {
    let mut sys = device_sae_fixture();
    Arc::get_mut(
        sys.device_sae_pcg
            .as_mut()
            .expect("fixture device descriptor uniquely owned"),
    )
    .expect("fixture device descriptor uniquely owned")
    .frame = Some(DeviceSaeFrameData {
        ranks: Vec::new(),
        basis_sizes: Vec::new(),
        border_offsets: Vec::new(),
        frame_blocks: Vec::new(),
        smooth_ranks: Vec::new(),
        row_htbeta: vec![Vec::new(); sys.rows.len()],
    });
    let refreshes = Arc::new(Mutex::new(Vec::new()));
    let frame: Arc<dyn SaeResidentFrame + Send + Sync> = Arc::new(RecordingFrame {
        calls: Arc::new(Mutex::new(Vec::new())),
        refreshes: Arc::clone(&refreshes),
    });
    let options = ArrowSolveOptions::direct();

    let prepared = prepare_sae_resident_frame(&sys, &options, Some(Arc::clone(&frame)))
        .expect("compatible frame refreshes in place");
    assert!(Arc::ptr_eq(&prepared, &frame));
    sys.rows[0].htt[[0, 0]] = 7.0;
    let prepared_again = prepare_sae_resident_frame(&sys, &options, Some(prepared))
        .expect("same allocation refreshes for the next iterate");
    assert!(Arc::ptr_eq(&prepared_again, &frame));
    assert_eq!(
        refreshes
            .lock()
            .expect("inspect nonlinear refresh ledger")
            .as_slice(),
        &[2.0, 7.0],
        "refresh must observe the newly assembled Hessian, not retain old content"
    );
    eprintln!(
        "#1017 nonlinear frame telemetry: stable_frame_identity=true refresh_htt={:?}",
        refreshes
            .lock()
            .expect("report nonlinear refresh ledger")
            .as_slice(),
    );
}

/// The framed descriptor's dominant nested row-cross slabs are allocation
/// workspace too: refill them in place, but replace every numerical value.
#[test]
fn framed_device_descriptor_reuses_nested_row_cross_allocations_1017() {
    let mut sys = ArrowSchurSystem::new(2, 1, 2);
    sys.set_device_sae_pcg_data(framed_device_data(1.0));
    let recycled = sys.device_sae_pcg.take().expect("first framed descriptor");
    let descriptor_ptr = Arc::as_ptr(&recycled) as usize;
    let row_ptr = recycled
        .frame
        .as_ref()
        .and_then(|frame| frame.row_htbeta.first())
        .map_or(0, |row| row.as_ptr() as usize);

    sys.set_device_sae_pcg_data_reusing(framed_device_data(7.0), Some(recycled));
    let refreshed = sys
        .device_sae_pcg
        .as_ref()
        .expect("refreshed framed descriptor");
    let refreshed_row = refreshed
        .frame
        .as_ref()
        .and_then(|frame| frame.row_htbeta.first())
        .expect("refreshed row-cross slab");
    assert_eq!(descriptor_ptr, Arc::as_ptr(refreshed) as usize);
    assert_ne!(row_ptr, 0);
    assert_eq!(row_ptr, refreshed_row.as_ptr() as usize);
    assert!(refreshed_row.iter().all(|&value| value == 7.0));
}
