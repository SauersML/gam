//! Issue #370 regression: BMS Jacobian callbacks used by the pre-fit
//! identifiability audit must self-compute their row scalars at `beta = []`.
//!
//! The original crash happened before fitting: the audit called
//! `effective_jacobian_at(beta=[], family_scalars=None)`, while the BMS
//! log-slope offset/baseline made `g_i != 0`. The old callback treated that as
//! an impossible state and demanded caller-supplied `BmsFamilyScalars`.
//!
//! This test pins the contract at the exact failing boundary without running a
//! full Bernoulli marginal-slope solve: both BMS callbacks own enough data to
//! compute `q_i`, `g_i`, `c_i`, and `z_i` themselves.

use gam::families::bms::{BmsLogslopeJacobian, BmsMarginalJacobian};
use gam::custom_family::{BlockEffectiveJacobian, FamilyLinearizationState};
use ndarray::{Array1, Array2};
use std::sync::Arc;

fn assert_close(label: &str, got: f64, expected: f64) {
    let scale = expected.abs().max(1.0);
    let rel = (got - expected).abs() / scale;
    assert!(
        rel < 1e-12,
        "{label}: got {got:.17e}, expected {expected:.17e}, rel={rel:.3e}"
    );
}

#[test]
fn bms_callbacks_self_compute_nonzero_logslope_baseline_at_beta_zero_370() {
    let marginal = Arc::new(
        Array2::from_shape_vec(
            (3, 2),
            vec![
                1.0, -0.4, //
                0.5, 0.8, //
                -0.2, 1.3,
            ],
        )
        .unwrap(),
    );
    let logslope = Arc::new(
        Array2::from_shape_vec(
            (3, 2),
            vec![
                0.7, -0.1, //
                -0.3, 0.9, //
                0.4, 0.6,
            ],
        )
        .unwrap(),
    );
    let offset_m = Array1::from_vec(vec![0.2, -0.5, 0.9]);
    let offset_s = Array1::from_vec(vec![1.7, -1.3, 0.8]);
    let z = Arc::new(Array1::from_vec(vec![-0.6, 0.4, 1.1]));
    let probit_scale = 0.75_f64;
    let state = FamilyLinearizationState {
        beta: &[],
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: probit_scale,
    };

    let marginal_cb = BmsMarginalJacobian::new(
        Arc::clone(&marginal),
        Arc::clone(&logslope),
        offset_m.clone(),
        offset_s.clone(),
        marginal.ncols(),
    );
    let logslope_cb = BmsLogslopeJacobian::new(
        Arc::clone(&marginal),
        Arc::clone(&logslope),
        offset_m.clone(),
        offset_s.clone(),
        Arc::clone(&z),
        marginal.ncols(),
    );

    let marginal_j = marginal_cb
        .effective_jacobian_at(&state)
        .expect("marginal callback must not demand family scalars at beta=0");
    let logslope_j = logslope_cb
        .effective_jacobian_at(&state)
        .expect("logslope callback must not demand family scalars at beta=0");

    assert_eq!(marginal_j.dim(), (marginal.nrows(), marginal.ncols()));
    assert_eq!(logslope_j.dim(), (logslope.nrows(), logslope.ncols()));

    for i in 0..marginal.nrows() {
        let q_i = offset_m[i];
        let g_i = offset_s[i];
        assert!(
            g_i.abs() > 0.0,
            "fixture must keep the #370 precondition g_i != 0 at beta=0"
        );
        let c_i = (1.0 + (probit_scale * g_i).powi(2)).sqrt();
        let logslope_factor = q_i * probit_scale * probit_scale * g_i / c_i + probit_scale * z[i];

        for j in 0..marginal.ncols() {
            assert_close(
                &format!("marginal row {i} col {j}"),
                marginal_j[[i, j]],
                c_i * marginal[[i, j]],
            );
            assert_close(
                &format!("logslope row {i} col {j}"),
                logslope_j[[i, j]],
                logslope_factor * logslope[[i, j]],
            );
        }
    }
}
