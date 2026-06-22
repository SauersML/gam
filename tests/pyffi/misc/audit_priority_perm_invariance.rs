//! Verifies that `audit_identifiability` produces the same rank verdict
//! regardless of whether the spec list is already sorted by descending
//! `gauge_priority` or is scrambled.
//!
//! The audit internally reorders columns by priority before feeding RRQR,
//! so the rank decision must be permutation-invariant. Additionally, each
//! attributed drop must land in the correct lower-priority block regardless
//! of the input order.
//!
//! Synthetic setup (3 blocks, 2-D shared constant direction):
//!
//!   - `high`   (gauge_priority=200): columns [ones, sin(x), cos(x)]
//!   - `mid`    (gauge_priority=150): columns [ones, sin(2x), cos(2x)]
//!   - `low`    (gauge_priority= 80): columns [ones, sin(3x), cos(3x)]
//!
//! Each block carries the constant column `ones`, so the joint design has
//! a 2-D null space among the three constant copies. With distinct priorities
//! the audit must attribute both drops to the two lower-priority blocks
//! (`mid` and `low`), never to `high`.

use gam::families::custom_family::ParameterBlockSpec;
use gam::identifiability::audit::audit_identifiability;
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::{Array1, Array2};

const N: usize = 80;

fn linspace(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![0.0; n.max(1)];
    }
    let step = 2.0 * std::f64::consts::PI / (n as f64);
    (0..n).map(|i| step * i as f64).collect()
}

fn spec_with_priority(name: &str, design: Array2<f64>, priority: u8) -> ParameterBlockSpec {
    let n = design.nrows();
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::<f64>::zeros(n),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: priority,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    }
}

/// Build the three-block spec list. The blocks always have the content
/// described in the module doc; `scramble` controls the order in which
/// they are presented to the audit.
///
/// Priority-ordered (natural):  [high(200), mid(150), low(80)]
/// Scrambled:                   [low(80), high(200), mid(150)]
fn build_specs(scramble: bool) -> Vec<ParameterBlockSpec> {
    let x = linspace(N);

    let mut high_design = Array2::<f64>::zeros((N, 3));
    let mut mid_design = Array2::<f64>::zeros((N, 3));
    let mut low_design = Array2::<f64>::zeros((N, 3));
    for i in 0..N {
        high_design[[i, 0]] = 1.0;
        high_design[[i, 1]] = x[i].sin();
        high_design[[i, 2]] = x[i].cos();

        mid_design[[i, 0]] = 1.0;
        mid_design[[i, 1]] = (2.0 * x[i]).sin();
        mid_design[[i, 2]] = (2.0 * x[i]).cos();

        low_design[[i, 0]] = 1.0;
        low_design[[i, 1]] = (3.0 * x[i]).sin();
        low_design[[i, 2]] = (3.0 * x[i]).cos();
    }

    let high = spec_with_priority("high", high_design, 200);
    let mid = spec_with_priority("mid", mid_design, 150);
    let low = spec_with_priority("low", low_design, 80);

    if scramble {
        // Present in ascending-priority order (worst case for the identity path).
        vec![low, high, mid]
    } else {
        vec![high, mid, low]
    }
}

/// The rank verdict (joint_rank = total_cols − drops) must be the same
/// whether specs arrive in descending or scrambled priority order.
#[test]
fn rank_verdict_invariant_under_priority_scramble() {
    let sorted_audit =
        audit_identifiability(&build_specs(false)).expect("sorted-order audit must succeed");
    let scrambled_audit =
        audit_identifiability(&build_specs(true)).expect("scrambled-order audit must succeed");

    // Total columns = 3 + 3 + 3 = 9. Two shared constant columns are
    // redundant → joint rank = 7. Both audits must agree.
    let sorted_rank: usize = sorted_audit.blocks.iter().map(|b| b.effective_dim).sum();
    let scrambled_rank: usize = scrambled_audit.blocks.iter().map(|b| b.effective_dim).sum();
    assert_eq!(
        sorted_rank, scrambled_rank,
        "rank verdict must be identical regardless of spec ordering; \
         sorted_rank={sorted_rank} scrambled_rank={scrambled_rank}\n\
         sorted summary:   {}\n\
         scrambled summary: {}",
        sorted_audit.summary, scrambled_audit.summary,
    );

    // Neither audit should be fatal (distinct priorities resolve the alias).
    assert!(
        !sorted_audit.fatal,
        "sorted audit must be non-fatal with distinct gauge_priority; \
         got: {}",
        sorted_audit.summary,
    );
    assert!(
        !scrambled_audit.fatal,
        "scrambled audit must be non-fatal with distinct gauge_priority; \
         got: {}",
        scrambled_audit.summary,
    );

    // Exact rank check: 9 − 2 = 7 kept columns.
    assert_eq!(
        sorted_rank, 7,
        "expected joint rank 7 (9 total − 2 shared constants); got {sorted_rank}"
    );
}

/// Drops must always be attributed to the two lower-priority blocks, never
/// to the highest-priority block, regardless of spec order.
#[test]
fn drops_attributed_to_lower_priority_blocks_in_both_orderings() {
    for scramble in [false, true] {
        let audit = audit_identifiability(&build_specs(scramble)).expect("audit must succeed");

        // Exactly two drops expected (two redundant constant columns).
        assert_eq!(
            audit.dropped_columns.len(),
            2,
            "expected exactly 2 dropped columns (scramble={scramble}); \
             got {} — summary: {}",
            audit.dropped_columns.len(),
            audit.summary,
        );

        for drop in &audit.dropped_columns {
            assert_ne!(
                drop.block, "high",
                "highest-priority block 'high' must never receive a drop \
                 (scramble={scramble}); got drop on '{}' — summary: {}",
                drop.block, audit.summary,
            );
        }

        // Each of `mid` and `low` must receive exactly one drop.
        let mid_drops = audit
            .dropped_columns
            .iter()
            .filter(|d| d.block == "mid")
            .count();
        let low_drops = audit
            .dropped_columns
            .iter()
            .filter(|d| d.block == "low")
            .count();
        assert_eq!(
            mid_drops, 1,
            "'mid' must receive exactly 1 drop (scramble={scramble}); \
             got {mid_drops} — summary: {}",
            audit.summary,
        );
        assert_eq!(
            low_drops, 1,
            "'low' must receive exactly 1 drop (scramble={scramble}); \
             got {low_drops} — summary: {}",
            audit.summary,
        );
    }
}

/// When all three blocks have equal priority the audit must be fatal
/// (no ordering exists to resolve the alias). This is the negative control.
#[test]
fn equal_priority_blocks_are_fatal_on_shared_constant() {
    let x = linspace(N);
    let mut high_design = Array2::<f64>::zeros((N, 3));
    let mut mid_design = Array2::<f64>::zeros((N, 3));
    let mut low_design = Array2::<f64>::zeros((N, 3));
    for i in 0..N {
        high_design[[i, 0]] = 1.0;
        high_design[[i, 1]] = x[i].sin();
        high_design[[i, 2]] = x[i].cos();
        mid_design[[i, 0]] = 1.0;
        mid_design[[i, 1]] = (2.0 * x[i]).sin();
        mid_design[[i, 2]] = (2.0 * x[i]).cos();
        low_design[[i, 0]] = 1.0;
        low_design[[i, 1]] = (3.0 * x[i]).sin();
        low_design[[i, 2]] = (3.0 * x[i]).cos();
    }
    // All priorities equal → gauge cannot resolve.
    let specs = vec![
        spec_with_priority("block_a", high_design, 100),
        spec_with_priority("block_b", mid_design, 100),
        spec_with_priority("block_c", low_design, 100),
    ];
    let audit = audit_identifiability(&specs).expect("audit must run");
    assert!(
        audit.fatal,
        "equal-priority blocks with shared constant must be fatal; \
         got: {}",
        audit.summary,
    );
}
