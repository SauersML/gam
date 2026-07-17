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

// The direct `LogslopeBlockJacobian` construction tests moved in-crate
// (crates/gam-models/src/survival/marginal_slope/tests.rs::
// logslope_jacobian_reads_probit_scale_from_state_at_beta_zero) when the
// constructor went crate-internal (#2352). The public-surface guards below
// (RowScaledJacobian scaling and the identifiability audit) remain here.

use gam::custom_family::{
    BlockEffectiveJacobian, FamilyLinearizationState, ParameterBlockSpec, RowScaledJacobian,
};
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

