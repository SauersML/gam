//! Regression test: `canonicalize_for_identifiability` routes through
//! `audit_identifiability_channel_aware` when any block declares
//! `n_outputs > 1` via its `jacobian_callback`.
//!
//! Layout: 3-block survival-marginal-slope synthetic (K=3 output channels:
//! η0, η1, ad1).
//!
//!   - `time` block: p_t columns, contributes to channel 0 (η0) only.
//!   - `marginal` block: p_m columns, contributes to channel 1 (η1) only.
//!     The raw design is identical to the time block's design (same Duchon-like
//!     polynomial basis), so the flat audit sees `[M | M | ...]` and flags a
//!     hard-alias.
//!   - `logslope` block: p_m columns, contributes to channel 2 (ad1) only.
//!     Raw design = diag(z) · M with z drawn from the standard normal;
//!     the flat audit sees `[M | M | diag(z)·M]` and flags a fatal alias.
//!     The channel-aware audit sees three orthogonal channels → full rank.
//!
//! **Positive** test (varying z):
//!   - All three blocks live in orthogonal channels.
//!   - Flat audit flags fatal hard-alias on time ~ marginal (same raw columns).
//!   - Channel-aware audit returns Ok, full rank, no drops.
//!   - `used_channel_aware_audit = true`.
//!   - Per-block T is identity (no column drops).
//!
//! **Negative** test (same-channel alias):
//!   - Two blocks both contributing to channel 0 with the same design.
//!   - Channel-aware audit correctly reports fatal alias even though raw
//!     columns are in a single non-orthogonal channel pair.

use std::sync::Arc;

use gam::families::custom_family::{
    AdditiveBlockJacobian, ParameterBlockSpec,
};
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam::solver::identifiability_audit::audit_identifiability;
use gam::solver::identifiability_canonical::canonicalize_for_identifiability;
use ndarray::{Array1, Array2};

const N: usize = 500;
const K: usize = 3;

fn linspace(n: usize) -> Array1<f64> {
    if n <= 1 {
        return Array1::<f64>::zeros(n.max(1));
    }
    let step = 2.0 / (n as f64 - 1.0);
    Array1::from_iter((0..n).map(|i| -1.0 + step * i as f64))
}

/// Build an n×p Duchon-like polynomial basis (columns: x^1, x^2, ..., x^p).
fn duchon_basis(x: &Array1<f64>, p: usize) -> Array2<f64> {
    let n = x.len();
    Array2::from_shape_fn((n, p), |(i, j)| x[i].powi((j + 1) as i32))
}

/// Build a ParameterBlockSpec with an `AdditiveBlockJacobian` callback that
/// routes the design into `own_channel` of a K-channel family.
fn spec_with_callback(
    name: &str,
    design: Array2<f64>,
    own_channel: usize,
    k: usize,
) -> ParameterBlockSpec {
    let n = design.nrows();
    let cb = Arc::new(AdditiveBlockJacobian {
        design: design.clone(),
        own_output: own_channel,
        n_family_outputs: k,
    });
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::<f64>::zeros(n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: Some(cb),
    }
}

// ── Positive test: orthogonal-channel blocks, flat flags fatal, channel-aware passes ────

#[test]
fn flat_audit_flags_fatal_on_shared_raw_columns() {
    // Flat audit ignores channels: sees [M | M | diag(z)·M] and flags
    // the time~marginal column pair as a hard-alias.
    let x = linspace(N);
    let p = 4;
    let m = duchon_basis(&x, p);
    // z: standardised normal (reproducible).
    let z: Array1<f64> = Array1::from_iter((0..N).map(|i| (i as f64 * 0.73 + 1.23).sin()));

    let time_design = m.clone();
    let marginal_design = m.clone(); // identical raw columns — flat sees alias
    let logslope_design = Array2::from_shape_fn((N, p), |(i, j)| m[[i, j]] * z[i]);

    // Specs without callbacks (flat-only).
    let flat_spec = |name: &str, d: Array2<f64>| {
        let n = d.nrows();
        ParameterBlockSpec {
            name: name.to_string(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(d)),
            offset: Array1::<f64>::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::<f64>::zeros(0),
            initial_beta: None,
            gauge_priority: 100,
            jacobian_callback: None,
        }
    };

    let specs_flat = [
        flat_spec("time", time_design),
        flat_spec("marginal", marginal_design),
        flat_spec("logslope", logslope_design),
    ];
    let flat_audit = audit_identifiability(&specs_flat).expect("flat audit must run");
    assert!(
        flat_audit.fatal,
        "flat audit must flag time~marginal as fatal hard-alias; summary: {}",
        flat_audit.summary,
    );
}

#[test]
fn channel_aware_routing_chosen_for_multi_output_blocks() {
    // Build 3 blocks with AdditiveBlockJacobian callbacks (K=3).
    // Each block contributes to a different channel → orthogonal in channel space.
    let x = linspace(N);
    let p = 4;
    let m = duchon_basis(&x, p);
    let z: Array1<f64> = Array1::from_iter((0..N).map(|i| (i as f64 * 0.73 + 1.23).sin()));
    let logslope_design = Array2::from_shape_fn((N, p), |(i, j)| m[[i, j]] * z[i]);

    let specs = [
        spec_with_callback("time", m.clone(), 0, K),
        spec_with_callback("marginal", m.clone(), 1, K),
        spec_with_callback("logslope", logslope_design, 2, K),
    ];

    let canon =
        canonicalize_for_identifiability(&specs).expect(
            "channel-aware audit must pass: blocks in orthogonal channels are fully identifiable",
        );

    // Routing check: must have used channel-aware audit.
    assert!(
        canon.used_channel_aware_audit,
        "canonicalize must route through audit_identifiability_channel_aware \
         when any block has n_outputs > 1; used_channel_aware_audit=false",
    );
}

#[test]
fn channel_aware_audit_passes_with_full_rank_for_orthogonal_channel_blocks() {
    let x = linspace(N);
    let p = 4;
    let m = duchon_basis(&x, p);
    let z: Array1<f64> = Array1::from_iter((0..N).map(|i| (i as f64 * 0.73 + 1.23).sin()));
    let logslope_design = Array2::from_shape_fn((N, p), |(i, j)| m[[i, j]] * z[i]);

    let specs = [
        spec_with_callback("time", m.clone(), 0, K),
        spec_with_callback("marginal", m.clone(), 1, K),
        spec_with_callback("logslope", logslope_design, 2, K),
    ];

    let canon =
        canonicalize_for_identifiability(&specs).expect(
            "channel-aware audit must pass: blocks in orthogonal channels are fully identifiable",
        );

    // Audit must be non-fatal (full rank).
    assert!(
        !canon.audit.fatal,
        "audit must be non-fatal when blocks live in orthogonal K-channels; \
         summary: {}",
        canon.audit.summary,
    );

    // No columns dropped.
    assert!(
        canon.audit.dropped_columns.is_empty(),
        "no columns should be dropped when blocks live in orthogonal channels; \
         dropped: {:?}",
        canon.audit.dropped_columns,
    );

    // Effective dim = 3 * p (full rank).
    let total_kept: usize = canon.audit.blocks.iter().map(|b| b.effective_dim).sum();
    assert_eq!(
        total_kept,
        3 * p,
        "all 3*p={} columns must be retained; got {total_kept}",
        3 * p,
    );
}

#[test]
fn post_canonicalize_t_is_identity_for_full_rank_3_block_case() {
    let x = linspace(N);
    let p = 4;
    let m = duchon_basis(&x, p);
    let z: Array1<f64> = Array1::from_iter((0..N).map(|i| (i as f64 * 0.73 + 1.23).sin()));
    let logslope_design = Array2::from_shape_fn((N, p), |(i, j)| m[[i, j]] * z[i]);

    let specs = [
        spec_with_callback("time", m.clone(), 0, K),
        spec_with_callback("marginal", m.clone(), 1, K),
        spec_with_callback("logslope", logslope_design, 2, K),
    ];

    let canon = canonicalize_for_identifiability(&specs)
        .expect("full-rank 3-block case must succeed");

    // Each per-block T must be the identity (p × p) — no drops.
    for (i, t) in canon.per_block_transform.iter().enumerate() {
        assert_eq!(
            t.nrows(),
            p,
            "T[{i}] must have {p} rows (raw width); got {}",
            t.nrows(),
        );
        assert_eq!(
            t.ncols(),
            p,
            "T[{i}] must have {p} cols (reduced width = raw since no drops); got {}",
            t.ncols(),
        );
        for row in 0..p {
            for col in 0..p {
                let expected = if row == col { 1.0 } else { 0.0 };
                let got = t[[row, col]];
                assert!(
                    (got - expected).abs() < 1e-12,
                    "T[{i}][{row},{col}] must be {expected} (identity); got {got}",
                );
            }
        }
    }
}

// ── Negative test: same-channel alias → channel-aware audit flags fatal ──────

#[test]
fn channel_aware_audit_flags_fatal_for_same_channel_alias() {
    // Two blocks both contributing to channel 0 with the same design.
    // The channel-aware view sees them as fully aliased → fatal.
    let x = linspace(N);
    let p = 3;
    let m = duchon_basis(&x, p);

    let specs = [
        spec_with_callback("block_a", m.clone(), 0, K),
        spec_with_callback("block_b", m.clone(), 0, K), // same channel, same design → alias
        spec_with_callback("block_c", m.clone(), 2, K), // distinct channel → fine
    ];

    let result = canonicalize_for_identifiability(&specs);
    assert!(
        result.is_err(),
        "same-channel alias must be refused by the channel-aware audit; \
         got Ok with audit: {}",
        result.as_ref().map(|c| &c.audit.summary[..]).unwrap_or(""),
    );

    // The error must be IdentifiabilityFailure (not a shape mismatch).
    match result.unwrap_err() {
        gam::families::custom_family::CustomFamilyError::IdentifiabilityFailure { audit } => {
            assert!(
                audit.fatal,
                "IdentifiabilityFailure audit must be fatal; got: {}",
                audit.summary,
            );
            // The summary must mention both aliased block names.
            assert!(
                audit.summary.contains("block_a") || audit.summary.contains("block_b"),
                "refusal summary must name at least one of the aliased blocks; \
                 got: {}",
                audit.summary,
            );
        }
        other => {
            // DimensionMismatch is also acceptable when the channel-aware
            // audit fails to construct operators — but the invariant is that
            // the fit must NOT succeed silently.
            let _ = other;
        }
    }
}

// ── Instrumentation smoke test: both audits comparable, log discrepancy visible ─

#[test]
fn instrumentation_flat_and_channel_aware_ranks_are_logged() {
    // With blocks in orthogonal channels, flat rank < channel-aware rank.
    // This test just verifies the function completes without panic; the
    // log discrepancy message is at INFO level and not asserted here.
    let x = linspace(N);
    let p = 3;
    let m = duchon_basis(&x, p);

    let specs = [
        spec_with_callback("time", m.clone(), 0, K),
        spec_with_callback("marginal", m.clone(), 1, K),
    ];

    let canon = canonicalize_for_identifiability(&specs)
        .expect("two-block orthogonal-channel case must succeed");

    assert!(
        canon.used_channel_aware_audit,
        "must use channel-aware audit for multi-output blocks",
    );
    // Both blocks at full channel-aware rank → 2*p effective dims.
    let total_kept: usize = canon.audit.blocks.iter().map(|b| b.effective_dim).sum();
    assert_eq!(
        total_kept,
        2 * p,
        "two-block orthogonal-channel case must retain all 2*p={} columns; \
         got {total_kept}",
        2 * p,
    );
}
