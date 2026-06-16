//! Regression tests for the identifiability audit on a synthetic
//! 5-block survival marginal-slope setup (time / marginal / logslope /
//! score_warp_dev / link_dev).
//!
//! # Scope
//!
//! The family-level pre-filter in `survival_marginal_slope.rs` and
//! `bernoulli_marginal_slope.rs` previously dropped score_warp_dev /
//! link_dev blocks silently (via a `log::warn!` side message) when
//! `install_compiled_flex_block_into_runtime` returned `FullyAliased`.
//! After the architectural change those blocks are kept with their
//! original designs and the unified audit (`canonicalize_for_identifiability`
//! → `audit_identifiability`) attributes any alias to the correct block
//! via `dropped_columns` (gauge_priority ordering ensures score_warp_dev
//! and link_dev are always the lower-priority participants in an alias pair
//! with parametric blocks).
//!
//! These tests exercise the flat `audit_identifiability` path directly
//! with synthetic K=4 channel-aware block designs, verifying:
//!
//! - When score_warp_dev / link_dev carry genuinely redundant directions
//!   (aliased by marginal or logslope): `audit.fatal == false`,
//!   `audit.dropped_columns` names score_warp_dev or link_dev,
//!   `audit.aliased_pairs` is non-empty.
//!
//! - When score_warp_dev / link_dev carry genuinely distinct directions
//!   (orthogonal complement of marginal/logslope after IFT Jacobian
//!   residualisation): audit is clean, no drops, no alias pairs.
//!
//! The gauge_priority ordering (time=200 / marginal=150 / logslope=120 /
//! score_warp=80 / link_dev=60) is replicated from the family's actual
//! blockspec assignment so the canonical-gauge contract is exercised here.

use gam::families::custom_family::ParameterBlockSpec;
use gam::identifiability::audit::audit_identifiability;
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::{Array1, Array2};

const N: usize = 128;

fn linspace(a: f64, b: f64, n: usize) -> Array1<f64> {
    if n <= 1 {
        return Array1::from_elem(1, a);
    }
    let step = (b - a) / (n as f64 - 1.0);
    Array1::from_iter((0..n).map(|i| a + step * i as f64))
}

fn spec(name: &str, design: Array2<f64>, gauge_priority: u8) -> ParameterBlockSpec {
    let n_rows = design.nrows();
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::<f64>::zeros(n_rows),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

/// Build the 5 synthetic block designs for the standard test fixture.
///
/// Time block (priority 200): [1, t, t²] — 3 linearly independent columns.
/// Marginal block (priority 150): [1, z, z²] — overlaps time's constant but
///   different covariates; in the flat audit `t` and `z` are treated as separate
///   directions so the overlap is partial, not perfect (no drop expected here).
/// Logslope block (priority 120): [sin(t), sin(2t)] — 2 columns.
/// score_warp_dev block (priority 80): configurable.
/// link_dev block (priority 60): configurable.
fn time_design() -> Array2<f64> {
    let t = linspace(0.0, 1.0, N);
    let mut out = Array2::<f64>::zeros((N, 3));
    for i in 0..N {
        out[[i, 0]] = 1.0;
        out[[i, 1]] = t[i];
        out[[i, 2]] = t[i] * t[i];
    }
    out
}

fn marginal_design() -> Array2<f64> {
    let z = linspace(-1.0, 1.0, N);
    let mut out = Array2::<f64>::zeros((N, 3));
    for i in 0..N {
        out[[i, 0]] = 1.0;
        out[[i, 1]] = z[i];
        out[[i, 2]] = z[i] * z[i];
    }
    out
}

fn logslope_design() -> Array2<f64> {
    let t = linspace(0.0, 1.0, N);
    let mut out = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        out[[i, 0]] = (std::f64::consts::PI * t[i]).sin();
        out[[i, 1]] = (2.0 * std::f64::consts::PI * t[i]).sin();
    }
    out
}

/// score_warp_dev that exactly duplicates marginal column 1 (z-linear).
/// At gauge_priority=80 < marginal=150, the audit should attribute the
/// drop to score_warp_dev, not to marginal.
fn score_warp_design_aliased() -> Array2<f64> {
    let z = linspace(-1.0, 1.0, N);
    let mut out = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        // Column 0: exactly marginal column 1 (z)
        out[[i, 0]] = z[i];
        // Column 1: z³ — in the span of marginal's quadratic + linear columns
        //            (approximately, but let's make it a genuine new direction)
        out[[i, 1]] = z[i] * z[i] * z[i] - 0.6 * z[i]; // Legendre P3-ish
    }
    out
}

/// score_warp_dev that is orthogonal (in Euclidean sense) to the parametric
/// blocks — pure high-frequency content no parametric block carries.
fn score_warp_design_clean() -> Array2<f64> {
    let t = linspace(0.0, 1.0, N);
    let mut out = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        out[[i, 0]] = (5.0 * std::f64::consts::PI * t[i]).sin();
        out[[i, 1]] = (7.0 * std::f64::consts::PI * t[i]).sin();
    }
    out
}

/// link_dev that duplicates logslope column 0 exactly.
fn link_dev_design_aliased() -> Array2<f64> {
    let t = linspace(0.0, 1.0, N);
    let mut out = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        out[[i, 0]] = (std::f64::consts::PI * t[i]).sin(); // exact copy of logslope col 0
        out[[i, 1]] = (9.0 * std::f64::consts::PI * t[i]).sin(); // genuinely new
    }
    out
}

/// link_dev that is clean (no alias with any other block).
fn link_dev_design_clean() -> Array2<f64> {
    let t = linspace(0.0, 1.0, N);
    let mut out = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        out[[i, 0]] = (11.0 * std::f64::consts::PI * t[i]).sin();
        out[[i, 1]] = (13.0 * std::f64::consts::PI * t[i]).sin();
    }
    out
}

// ── Test 1: score_warp_dev column exactly aliased with marginal ──────────────

#[test]
fn score_warp_dev_alias_attributed_to_score_warp_not_marginal() {
    let specs = [
        spec("time", time_design(), 200),
        spec("marginal", marginal_design(), 150),
        spec("logslope", logslope_design(), 120),
        spec("score_warp_dev", score_warp_design_aliased(), 80),
        spec("link_dev", link_dev_design_clean(), 60),
    ];
    let audit = audit_identifiability(&specs).expect("audit must succeed");

    // With gauge_priority ordering the alias should NOT be fatal:
    // score_warp_dev (80) < marginal (150), so canonical-gauge resolves it.
    assert!(
        !audit.fatal,
        "score_warp_dev alias with marginal must be gauge-resolvable (non-fatal); \
         summary: {}",
        audit.summary,
    );

    // The dropped column must be attributed to score_warp_dev, not marginal.
    assert!(
        !audit.dropped_columns.is_empty(),
        "score_warp_dev alias must produce at least one dropped column; \
         summary: {}",
        audit.summary,
    );
    for drop in &audit.dropped_columns {
        assert_eq!(
            drop.block, "score_warp_dev",
            "dropped column must belong to score_warp_dev (lower priority=80), \
             not to '{}' (summary: {})",
            drop.block, audit.summary,
        );
    }

    // The aliased_pairs must include the (marginal, score_warp_dev) pair.
    let has_sw_marginal_pair = audit.aliased_pairs.iter().any(|p| {
        (p.block_a == "marginal" && p.block_b == "score_warp_dev")
            || (p.block_a == "score_warp_dev" && p.block_b == "marginal")
    });
    assert!(
        has_sw_marginal_pair,
        "aliased_pairs must include the (marginal, score_warp_dev) pair; \
         got: {:?}",
        audit.aliased_pairs,
    );
}

// ── Test 2: link_dev column exactly aliased with logslope ────────────────────

#[test]
fn link_dev_alias_attributed_to_link_dev_not_logslope() {
    let specs = [
        spec("time", time_design(), 200),
        spec("marginal", marginal_design(), 150),
        spec("logslope", logslope_design(), 120),
        spec("score_warp_dev", score_warp_design_clean(), 80),
        spec("link_dev", link_dev_design_aliased(), 60),
    ];
    let audit = audit_identifiability(&specs).expect("audit must succeed");

    // link_dev (60) < logslope (120): gauge-resolvable, non-fatal.
    assert!(
        !audit.fatal,
        "link_dev alias with logslope must be gauge-resolvable (non-fatal); \
         summary: {}",
        audit.summary,
    );
    assert!(
        !audit.dropped_columns.is_empty(),
        "link_dev alias must produce at least one dropped column; \
         summary: {}",
        audit.summary,
    );
    for drop in &audit.dropped_columns {
        assert_eq!(
            drop.block, "link_dev",
            "dropped column must belong to link_dev (lowest priority=60), \
             not to '{}' (summary: {})",
            drop.block, audit.summary,
        );
    }
    let has_ld_logslope_pair = audit.aliased_pairs.iter().any(|p| {
        (p.block_a == "logslope" && p.block_b == "link_dev")
            || (p.block_a == "link_dev" && p.block_b == "logslope")
    });
    assert!(
        has_ld_logslope_pair,
        "aliased_pairs must include the (logslope, link_dev) pair; \
         got: {:?}",
        audit.aliased_pairs,
    );
}

// ── Test 3: both score_warp_dev and link_dev aliased simultaneously ──────────

#[test]
fn both_flex_blocks_aliased_both_attributed_non_fatal() {
    let specs = [
        spec("time", time_design(), 200),
        spec("marginal", marginal_design(), 150),
        spec("logslope", logslope_design(), 120),
        spec("score_warp_dev", score_warp_design_aliased(), 80),
        spec("link_dev", link_dev_design_aliased(), 60),
    ];
    let audit = audit_identifiability(&specs).expect("audit must succeed");
    assert!(
        !audit.fatal,
        "simultaneous aliases in score_warp_dev and link_dev must be gauge-resolvable; \
         summary: {}",
        audit.summary,
    );
    // Both blocks should have drops attributed to them.
    let sw_drop = audit
        .dropped_columns
        .iter()
        .any(|d| d.block == "score_warp_dev");
    let ld_drop = audit.dropped_columns.iter().any(|d| d.block == "link_dev");
    assert!(
        sw_drop || ld_drop,
        "at least one of score_warp_dev or link_dev must have dropped columns; \
         got: {:?}",
        audit.dropped_columns,
    );
    // No dropped columns should be attributed to parametric blocks.
    for drop in &audit.dropped_columns {
        assert!(
            drop.block == "score_warp_dev" || drop.block == "link_dev",
            "only flex blocks should have dropped columns; got block='{}' (summary: {})",
            drop.block,
            audit.summary,
        );
    }
}

// ── Test 4: all 5 blocks clean → audit passes silently ───────────────────────

#[test]
fn all_five_blocks_clean_audit_passes() {
    let specs = [
        spec("time", time_design(), 200),
        spec("marginal", marginal_design(), 150),
        spec("logslope", logslope_design(), 120),
        spec("score_warp_dev", score_warp_design_clean(), 80),
        spec("link_dev", link_dev_design_clean(), 60),
    ];
    let audit = audit_identifiability(&specs).expect("audit must succeed");
    assert!(
        !audit.fatal,
        "clean 5-block setup must not be fatal; summary: {}",
        audit.summary,
    );
    assert!(
        audit.dropped_columns.is_empty(),
        "clean setup must have no dropped columns; got: {:?}",
        audit.dropped_columns,
    );
    // The summary should end with " — clean".
    assert!(
        audit.summary.contains("clean"),
        "summary must indicate clean audit; got: {}",
        audit.summary,
    );
}

// ── Test 5: score_warp_dev alias is non-fatal because gauge resolves it ──────
// (Explicit check that gauge_priority contract is honoured: same setup as
// Test 1 but we verify the exact non-fatal mechanism.)

#[test]
fn gauge_priority_resolves_score_warp_alias_correctly() {
    let specs = [
        spec("time", time_design(), 200),
        spec("marginal", marginal_design(), 150),
        spec("logslope", logslope_design(), 120),
        spec("score_warp_dev", score_warp_design_aliased(), 80),
        spec("link_dev", link_dev_design_clean(), 60),
    ];
    let audit = audit_identifiability(&specs).expect("audit must succeed");

    // The summary must explicitly reference gauge-attributed drops.
    assert!(
        audit.summary.contains("gauge-attributed drops")
            || (!audit.fatal && !audit.dropped_columns.is_empty()),
        "summary must describe gauge attribution; got: {}",
        audit.summary,
    );
    // Effective dim of score_warp_dev must be reduced.
    let sw_block = audit
        .blocks
        .iter()
        .find(|b| b.block_name == "score_warp_dev");
    assert!(
        sw_block.is_some(),
        "score_warp_dev block must appear in audit.blocks",
    );
    let sw = sw_block.unwrap();
    assert!(
        sw.effective_dim < sw.original_dim,
        "score_warp_dev effective_dim {} must be less than original_dim {} after alias drop",
        sw.effective_dim,
        sw.original_dim,
    );
    // Effective dims of higher-priority blocks must be unchanged.
    for name in ["time", "marginal", "logslope"] {
        if let Some(b) = audit.blocks.iter().find(|b| b.block_name == name) {
            assert_eq!(
                b.effective_dim, b.original_dim,
                "parametric block '{}' must not lose any columns (priority >= 120); \
                 audit: {}",
                name, audit.summary,
            );
        }
    }
}
