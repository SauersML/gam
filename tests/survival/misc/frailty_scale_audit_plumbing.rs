//! Plumbing test: frailty scale s_f = 1/√(1+σ²) flows correctly through the
//! effective Jacobian for the survival marginal-slope logslope block when σ is
//! non-trivial (σ > 0, so s_f < 1).
//!
//! # What this verifies
//!
//! 1. `FamilyLinearizationState::probit_frailty_scale` is read by
//!    `LogslopeBlockJacobian::effective_jacobian_at` and changes the Jacobian
//!    output proportionally: the β=0 logslope Jacobian row for output η0 is
//!    `s_f · z_i · G[i,:]`, so doubling s_f doubles the Jacobian.
//!
//! 2. The flat identifiability audit (`audit_identifiability`) sees the logslope
//!    spec's `RowScaledJacobian` with s_f · z scaling and correctly identifies
//!    the logslope block as non-aliased with the marginal block (which has no
//!    scaling callback and contributes via a different column design).
//!
//! 3. A state with `probit_frailty_scale = 1.0` (no frailty) yields a
//!    Jacobian that differs from a state with `probit_frailty_scale = s_f < 1`
//!    by a factor of exactly `s_f / 1.0 = s_f` on each non-zero element at
//!    β = 0.
//!
//! # σ-as-parameter verdict
//!
//! σ is always **fixed** in the survival marginal-slope family:
//! `validate_spec` rejects `FrailtySpec::GaussianShift { sigma_fixed: None }`
//! with an explicit error. There is no code path where σ is part of β.
//! Therefore `∂s_f/∂σ` never appears in the β-Jacobian and no σ-column is
//! needed. The `probit_frailty_scale` field on `FamilyLinearizationState`
//! carries the construction-time value through to the Jacobian callback at
//! evaluation time, ensuring outer-loop σ updates (which rebuild the family
//! with a new fixed σ) are reflected without requiring spec reconstruction.

use gam::custom_family::{
    BlockEffectiveJacobian, FamilyLinearizationState, ParameterBlockSpec, RowScaledJacobian,
};
use gam::families::survival::marginal_slope::LogslopeBlockJacobian;
use gam::identifiability::audit::audit_identifiability;
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::{Array1, Array2};
use std::sync::Arc;

const N: usize = 40;
const P: usize = 5;

/// Build a random-ish design matrix deterministically.
fn make_design(seed: u64) -> Array2<f64> {
    let mut out = Array2::<f64>::zeros((N, P));
    let mut state = seed;
    for i in 0..N {
        for j in 0..P {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            out[[i, j]] = ((state >> 33) as f64) / (u32::MAX as f64) - 0.5;
        }
    }
    out
}

/// Build z scores deterministically (positive, mean ~1).
fn make_z(seed: u64) -> Vec<f64> {
    let mut state = seed ^ 0xdeadbeef;
    (0..N)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            0.3 + ((state >> 33) as f64) / (u32::MAX as f64) * 1.4
        })
        .collect()
}

/// Build a logslope spec with a `RowScaledJacobian` callback using s_f·z scaling.
///
/// This expresses the β=0 flat single-output effective design `diag(s_f·z)·design`
/// through the unified `jacobian_callback` path.  For tests that need the full
/// β-dependent multi-output `LogslopeBlockJacobian`, construct it directly.
fn make_logslope_spec(design: &Array2<f64>, z: &[f64], s_f: f64) -> ParameterBlockSpec {
    let sf_z: Arc<[f64]> = z.iter().map(|&zi| s_f * zi).collect::<Vec<f64>>().into();
    let jac_cb: Arc<dyn BlockEffectiveJacobian> = Arc::new(RowScaledJacobian {
        design: Arc::new(design.clone()),
        eta_scaling: sf_z,
    });
    ParameterBlockSpec {
        name: "logslope_surface".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design.clone())),
        offset: Array1::<f64>::zeros(N),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: 120,
        jacobian_callback: Some(jac_cb),
        stacked_design: None,
        stacked_offset: None,
    }
}

/// Build a marginal block spec with a different design (no frailty involvement).
fn make_marginal_spec(design: &Array2<f64>) -> ParameterBlockSpec {
    ParameterBlockSpec {
        name: "marginal_surface".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design.clone())),
        offset: Array1::<f64>::zeros(N),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: 150,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

/// At β=0 the logslope Jacobian's η0 channel should equal s_f · z_i · G[i,j].
/// Verify that the Jacobian from `effective_jacobian_at` with
/// `probit_frailty_scale = s_f` has the right s_f scaling.
#[test]
fn logslope_jacobian_incorporates_sf_from_state_at_beta_zero() {
    let design = make_design(42);
    let z = make_z(42);
    let s_f = 0.75_f64; // σ = √(1/s_f² − 1) ≈ 0.882

    let jac_cb = LogslopeBlockJacobian::new(design.clone(), z.clone(), s_f);

    let beta_zero = vec![0.0f64; P];

    // State with the correct s_f.
    let state_sf = FamilyLinearizationState {
        beta: &beta_zero,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: s_f,
    };
    let jac_sf = jac_cb
        .effective_jacobian_at(&state_sf)
        .expect("effective_jacobian_at with s_f must succeed");

    // State with s_f = 1.0 (no frailty default).
    let state_1 = FamilyLinearizationState {
        beta: &beta_zero,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: 1.0,
    };
    let jac_1 = jac_cb
        .effective_jacobian_at(&state_1)
        .expect("effective_jacobian_at with s_f=1 must succeed");

    assert_eq!(jac_sf.nrows(), 3 * N, "Jacobian must have 3*N rows");
    assert_eq!(jac_sf.ncols(), P, "Jacobian must have P cols");

    // At β=0 and no family_scalars, the η0 and η1 rows are s_f·z_i·G[i,j].
    // The ad1 rows are zero.
    // So jac_sf[i,j] / jac_1[i,j] == s_f for all i,j where z_i != 0.
    let mut max_ratio_err = 0.0_f64;
    for i in 0..N {
        for j in 0..P {
            let got_sf = jac_sf[[i, j]];
            let got_1 = jac_1[[i, j]];
            let expected = s_f * z[i] * design[[i, j]];
            let err_sf = (got_sf - expected).abs();
            let denom = expected.abs().max(1e-14);
            assert!(
                err_sf / denom < 1e-10 || err_sf < 1e-12,
                "η0[{i},{j}]: got {got_sf:.6e} expected {expected:.6e}",
            );
            // jac_1 should be 1/s_f times jac_sf (since s_f→1.0 means scale by 1/s_f).
            if got_sf.abs() > 1e-14 {
                let ratio = got_1 / got_sf;
                max_ratio_err = max_ratio_err.max((ratio - 1.0 / s_f).abs());
            }
        }
    }
    assert!(
        max_ratio_err < 1e-10,
        "Ratio jac_1/jac_sf must equal 1/s_f = {:.6} everywhere; max err = {max_ratio_err:.3e}",
        1.0 / s_f,
    );
}

/// The logslope spec's `RowScaledJacobian` uses s_f·z, not bare z.
/// Verify that the flat single-output effective Jacobian (via `RowScaledJacobian`)
/// has the correct s_f factor.
#[test]
fn logslope_row_scaled_jacobian_includes_sf() {
    let design = make_design(7);
    let z = make_z(7);
    let s_f = 0.82_f64;

    let spec = make_logslope_spec(&design, &z, s_f);

    // The `RowScaledJacobian` callback returns the single-output (N×P) scaled design.
    // Verify jac[i, j] == s_f * z[i] * design[i, j].
    let beta_zero = vec![0.0f64; P];
    let state = FamilyLinearizationState {
        beta: &beta_zero,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: s_f,
    };
    let jac = spec
        .effective_jacobian_at("test", &state)
        .expect("effective_jacobian_at must succeed");

    assert_eq!(jac.nrows(), N, "RowScaledJacobian must have N rows");
    assert_eq!(jac.ncols(), P, "RowScaledJacobian must have P cols");

    for i in 0..N {
        for j in 0..P {
            let got = jac[[i, j]];
            let expected = s_f * z[i] * design[[i, j]];
            let err = (got - expected).abs();
            let denom = expected.abs().max(1e-14);
            assert!(
                err / denom < 1e-10 || err < 1e-12,
                "row {i} col {j}: got {got:.6e} expected s_f*z*G = {expected:.6e}",
            );
        }
    }
}

/// The identifiability audit on a marginal + logslope pair with non-trivial
/// s_f must pass (no fatal alias) because the logslope effective design is
/// s_f·z·G (different row-weights than the marginal's unscaled design M).
#[test]
fn audit_marginal_logslope_not_aliased_under_nontrivial_sf() {
    // Use different designs for marginal and logslope so they are structurally distinct.
    let design_log = make_design(100);
    let design_marg = make_design(200); // deliberately different
    let z = make_z(100);
    let s_f = 0.6_f64;

    let logslope_spec = make_logslope_spec(&design_log, &z, s_f);
    let marginal_spec = make_marginal_spec(&design_marg);

    let specs = [marginal_spec, logslope_spec];
    let audit = audit_identifiability(&specs).expect("audit must succeed");

    assert!(
        !audit.fatal,
        "marginal + logslope with s_f={s_f} must not be fatal; summary: {}",
        audit.summary,
    );
    assert!(
        audit.dropped_columns.is_empty(),
        "no columns should be dropped for structurally distinct blocks; got: {:?}",
        audit.dropped_columns,
    );
}

/// Changing s_f changes the effective Jacobian: two specs with the same design
/// and z but different s_f values must produce different effective Jacobians
/// at β=0 — confirming s_f is consumed at eval time, not baked in at construction.
#[test]
fn different_sf_values_produce_different_jacobians() {
    let design = make_design(999);
    let z = make_z(999);

    let s_f_a = 1.0_f64;
    let s_f_b = 0.5_f64;

    // Both specs use the same design and z, but different s_f captured at construction.
    let jac_a = LogslopeBlockJacobian::new(design.clone(), z.clone(), s_f_a);
    let jac_b = LogslopeBlockJacobian::new(design.clone(), z.clone(), s_f_b);

    let beta_zero = vec![0.0f64; P];

    // Test with state.probit_frailty_scale = s_f_b for both callbacks.
    // jac_b should return s_f_b·z·G; jac_a should return s_f_b·z·G too (state wins).
    let state_b = FamilyLinearizationState {
        beta: &beta_zero,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: s_f_b,
    };
    let out_a_with_b_state = jac_a
        .effective_jacobian_at(&state_b)
        .expect("effective_jacobian_at a with s_f_b state");
    let out_b_with_b_state = jac_b
        .effective_jacobian_at(&state_b)
        .expect("effective_jacobian_at b with s_f_b state");

    // Both should agree: state value takes precedence over captured self.s.
    let max_diff: f64 = out_a_with_b_state
        .iter()
        .zip(out_b_with_b_state.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_diff < 1e-12,
        "When state.probit_frailty_scale = s_f_b, both callbacks (constructed with s_f_a \
         and s_f_b) must produce the same Jacobian (state wins); max diff = {max_diff:.3e}",
    );

    // With s_f_a state vs s_f_b state, jac_a should differ by ratio s_f_a/s_f_b on each row.
    let state_a = FamilyLinearizationState {
        beta: &beta_zero,
        family_scalars: None,
        channel_hessian: None,
        probit_frailty_scale: s_f_a,
    };
    let out_a_with_a_state = jac_a
        .effective_jacobian_at(&state_a)
        .expect("effective_jacobian_at a with s_f_a state");

    // At β=0: row i should scale by s_f; ratio = s_f_a / s_f_b = 2.0.
    let expected_ratio = s_f_a / s_f_b;
    let mut max_ratio_err = 0.0_f64;
    for i in 0..N {
        for j in 0..P {
            let with_a = out_a_with_a_state[[i, j]];
            let with_b = out_a_with_b_state[[i, j]];
            if with_b.abs() > 1e-14 {
                let ratio = with_a / with_b;
                max_ratio_err = max_ratio_err.max((ratio - expected_ratio).abs());
            }
        }
    }
    assert!(
        max_ratio_err < 1e-10,
        "η0 rows at β=0: ratio with_s_f_a/with_s_f_b must equal {expected_ratio:.2}; \
         max err = {max_ratio_err:.3e}",
    );
}
