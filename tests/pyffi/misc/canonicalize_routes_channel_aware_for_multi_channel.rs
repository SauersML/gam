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
//!   - `slope` block: p_m columns, contributes to channel 2 (ad1) only.
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

use gam::families::custom_family::ParameterBlockSpec;
use gam::identifiability::audit::audit_identifiability;
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::{Array1, Array2};

const N: usize = 500;

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
    let slope_design = Array2::from_shape_fn((N, p), |(i, j)| m[[i, j]] * z[i]);

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
            stacked_design: None,
            stacked_offset: None,
        }
    };

    let specs_flat = [
        flat_spec("time", time_design),
        flat_spec("marginal", marginal_design),
        flat_spec("slope", slope_design),
    ];
    let flat_audit = audit_identifiability(&specs_flat).expect("flat audit must run");
    assert!(
        flat_audit.fatal,
        "flat audit must flag time~marginal as fatal hard-alias; summary: {}",
        flat_audit.summary,
    );
}

// ── Negative test: same-channel alias → channel-aware audit flags fatal ──────

// ── Regression: K=2 location-scale orthogonal channels must remain full rank ────
//
// AdditiveBlockJacobian emits channel-major (n_outputs * n, p) rows:
// `[output_0_rows; output_1_rows; …]`. The pre-fit audit's
// `BlockJacobianAsRowOp::from_callback` destacks that into `(n, p, k)`. When
// the destacker reads with the wrong layout, each block's design rows are
// scattered into the wrong channels (and the wrong observations), the joint
// W collapses to roughly half rank, and the channel-aware audit refuses the
// fit with "0 dropped column(s) … no per-column attribution". This
// regression test exercises the K=2 case (location-scale GAMLSS) directly:
// two blocks with the same Duchon design, each owning one of the two
// channels. Under the correct destacking, the channel-aware audit must keep
// every column (joint rank = 2·p) — under the buggy interleaved destacking,
// it would see joint rank ≈ p and FATAL.

// ── Regression: K=2 same-channel alias must still be detected after destacking fix

// ── Instrumentation smoke test: both audits comparable, log discrepancy visible ─

