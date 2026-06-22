//! RED tests for task #5: identifiability audit halt-or-repair gate.
//!
//! Contract pinned here:
//!
//! 1. Any post-rotation alias pair with overlap `>= 0.99` is a hard refusal.
//!    Even when RRQR "attributes" the deficiency by demoting a column,
//!    two distinct blocks contributing the same direction (to within 1%)
//!    is a structural identifiability failure: the inner KKT system has
//!    no unique minimiser in that direction, and the outer optimiser
//!    will spin (the large-scale failure mode).
//!
//! 2. `joint_rank < joint_cols` AFTER attribution is a hard refusal.
//!    The existing `fatal` flag only fires when the deficiency cannot be
//!    attributed AT ALL (`aliased_pairs.is_empty() && dropped_columns.is_empty()`).
//!    That gate misses the large-scale case (179 pairs >=0.95, 13 cols dropped,
//!    rank 38/51) entirely. The new gate must halt whenever the residual
//!    rank after attribution is still below the column count.
//!
//! 3. The fatal `summary` string names the offending block(s), the offending
//!    columns, and a concrete reparameterisation suggestion — so the
//!    failure surfaces actionable information rather than a generic
//!    "audit refused the fit" terminal print.
//!
//! These tests will turn GREEN after the gate edit in
//! `src/identifiability/audit.rs`.

use gam::identifiability::audit::audit_identifiability;
use ndarray::Array2;

mod common {
    use gam::families::custom_family::ParameterBlockSpec;
    use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    use ndarray::{Array1, Array2};

    pub fn spec_from_dense(name: &str, design: Array2<f64>) -> ParameterBlockSpec {
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

    pub fn linspace_minus_one_to_one(n: usize) -> Array1<f64> {
        if n <= 1 {
            return Array1::<f64>::zeros(n.max(1));
        }
        let step = 2.0 / (n as f64 - 1.0);
        Array1::from_iter((0..n).map(|i| -1.0 + step * i as f64))
    }
}

/// Hard-halt case (a): a pair with overlap == 1.0 (exact alias) between
/// two named blocks must be fatal. The large-scale failing run reported 179
/// pairs in this regime; today's gate lets them through.
#[test]
fn audit_exact_alias_pair_must_halt() {
    let n = 64;
    let x = common::linspace_minus_one_to_one(n);
    // Two blocks, each carrying `x` as their first column. Overlap = 1.0.
    let mut block_a = Array2::<f64>::zeros((n, 2));
    let mut block_b = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        block_a[[i, 0]] = x[i];
        block_a[[i, 1]] = x[i] * x[i];
        block_b[[i, 0]] = x[i];
        block_b[[i, 1]] = x[i].sin();
    }
    let specs = [
        common::spec_from_dense("time_surface", block_a),
        common::spec_from_dense("marginal_surface", block_b),
    ];
    let audit = audit_identifiability(&specs).expect("audit must succeed mechanically");
    assert!(
        audit.fatal,
        "exact x~x alias between two named blocks must be fatal under the new gate; \
         summary was: {}",
        audit.summary,
    );
    // Actionable message: must name BOTH offending blocks AND the column index.
    assert!(
        audit.summary.contains("time_surface"),
        "fatal summary must name the time_surface block; got {:?}",
        audit.summary,
    );
    assert!(
        audit.summary.contains("marginal_surface"),
        "fatal summary must name the marginal_surface block; got {:?}",
        audit.summary,
    );
    assert!(
        audit.summary.contains("reparam")
            || audit.summary.contains("sum-to-zero")
            || audit.summary.contains("orthogonal")
            || audit.summary.contains("absorb"),
        "fatal summary must include a reparameterisation suggestion; got {:?}",
        audit.summary,
    );
}

/// Hard-halt case (b): joint_rank < joint_cols after attribution is fatal
/// even when RRQR populated `dropped_columns`. Today's predicate
/// `fatal = joint_rank_deficient && aliased_pairs.is_empty() && dropped_columns.is_empty()`
/// declares this case non-fatal (alias-attributed) — the large-scale hang
/// pattern.
#[test]
fn audit_rank_deficient_with_attribution_must_halt() {
    let n = 96;
    let x = common::linspace_minus_one_to_one(n);
    // Three blocks where block C's only column lies in the span of A and B.
    // RRQR will attribute and demote one column; today that returns
    // fatal=false. New gate: still fatal because rank < cols.
    let block_a = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| 1.0 + 0.4 * x[i]);
    let block_b = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| 1.0 - 0.4 * x[i]);
    let block_c = Array2::<f64>::from_shape_fn((n, 1), |(i, _)| 1.0 + 0.0 * x[i]); // = 1
    let specs = [
        common::spec_from_dense("anchor", block_a),
        common::spec_from_dense("mirror", block_b),
        common::spec_from_dense("constant_alias", block_c),
    ];
    let audit = audit_identifiability(&specs).expect("audit must succeed mechanically");
    // Sanity: RRQR attributed at least one drop (this is the "warn-and-proceed"
    // shape we are explicitly refusing now).
    assert!(
        !audit.dropped_columns.is_empty(),
        "RRQR is expected to attribute the deficiency; got {:?}",
        audit.dropped_columns,
    );
    let total_kept: usize = audit.blocks.iter().map(|b| b.effective_dim).sum();
    let joint_cols: usize = audit.blocks.iter().map(|b| b.original_dim).sum();
    assert!(
        total_kept < joint_cols,
        "test precondition: joint rank must be deficient; kept={total_kept} cols={joint_cols}",
    );
    assert!(
        audit.fatal,
        "joint_rank < joint_cols (kept={total_kept}/{joint_cols}) with attribution must \
         be fatal under the new gate; summary was: {}",
        audit.summary,
    );
}

/// Clean specs (no alias above 0.99 and full rank) must remain non-fatal.
/// Pins that the new gate does not regress the happy path.
#[test]
fn audit_clean_specs_remain_non_fatal() {
    let n = 64;
    let x = common::linspace_minus_one_to_one(n);
    let mut a = Array2::<f64>::zeros((n, 2));
    let mut b = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        a[[i, 0]] = 1.0;
        a[[i, 1]] = x[i];
        b[[i, 0]] = x[i].powi(2);
        b[[i, 1]] = x[i].powi(3);
    }
    let specs = [
        common::spec_from_dense("parametric", a),
        common::spec_from_dense("smooth", b),
    ];
    let audit = audit_identifiability(&specs).expect("audit must succeed");
    assert!(
        !audit.fatal,
        "clean full-rank specs must not be fatal: {}",
        audit.summary,
    );
}

/// Below-threshold alias (overlap < 0.99) must NOT halt by itself: the
/// gate is for *structurally* unfittable designs, not partial collinearity
/// that the penalty + line search can still resolve.
#[test]
fn audit_partial_overlap_below_threshold_does_not_halt() {
    let n = 64;
    let x = common::linspace_minus_one_to_one(n);
    // Two columns with overlap ~0.986 (high but below the 0.99 ceiling this
    // test pins). The joint RRQR keeps both because they span 2 dimensions.
    // The earlier 0.25·x³ perturbation only reached overlap 0.9983, ABOVE the
    // 0.99 ceiling the test's own guard enforces; a unit cubic term drops the
    // overlap to ~0.986, the intended "high but partial" regime.
    let mut block_a = Array2::<f64>::zeros((n, 1));
    let mut block_b = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        block_a[[i, 0]] = x[i];
        // x + cubic perturbation: overlap stays ~0.986.
        block_b[[i, 0]] = x[i] + 1.0 * x[i].powi(3);
    }
    let specs = [
        common::spec_from_dense("a", block_a),
        common::spec_from_dense("b", block_b),
    ];
    let audit = audit_identifiability(&specs).expect("audit must succeed");
    // The high-overlap pair may or may not appear depending on the
    // exact overlap; what matters is the gate must NOT fire here.
    if audit.aliased_pairs.iter().any(|p| p.overlap >= 0.99) {
        panic!(
            "test fixture must keep overlap below 0.99 to pin the partial-overlap \
             tolerance; got pairs {:?}",
            audit.aliased_pairs,
        );
    }
    let kept: usize = audit.blocks.iter().map(|b| b.effective_dim).sum();
    let cols: usize = audit.blocks.iter().map(|b| b.original_dim).sum();
    assert_eq!(kept, cols, "test precondition: joint rank must be full");
    assert!(
        !audit.fatal,
        "partial overlap below 0.99 must not halt; summary: {}",
        audit.summary,
    );
}

/// Regression test for issue #292: survival marginal-slope cross-block alias.
///
/// Reproduces the 33-col / rank-23 scenario where `marginal_surface` and
/// `logslope_surface` share the same raw basis (overlap=1.0) but the
/// canonical-gauge pipeline is configured to handle it via gauge_priority
/// (time=200, marginal=150, logslope=120).
///
/// The pre-fix audit FATALed on this configuration.  Post-fix it must:
///   - return `fatal=false` (gauge resolves the alias)
///   - populate `dropped_columns` attributing drops to `logslope_surface`
///   - NOT halt on the `marginal_surface ~ logslope_surface` alias pair
///   - survive both the pilot path AND the outer-inner-fit path.
#[test]
fn cross_block_alias_with_distinct_priorities_is_not_fatal() {
    use gam::families::custom_family::{ParameterBlockSpec, RowScaledJacobian};
    use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
    use ndarray::Array1;

    // n = 200 (representative of large scale, small enough for fast test)
    let n = 200usize;
    let x: ndarray::Array1<f64> =
        ndarray::Array1::from_iter((0..n).map(|i| -1.0 + 2.0 * i as f64 / (n as f64 - 1.0)));

    // time_surface: 12 columns — Legendre-like polynomial basis up to degree 11.
    let mut time = ndarray::Array2::<f64>::zeros((n, 12));
    for i in 0..n {
        time[[i, 0]] = 1.0;
        for deg in 1usize..12 {
            time[[i, deg]] = x[i].powi(deg as i32);
        }
    }

    // marginal_surface: 11 columns — Duchon-PC-like smooth basis.
    let mut marginal = ndarray::Array2::<f64>::zeros((n, 11));
    for i in 0..n {
        for k in 0..11 {
            marginal[[i, k]] = ((k as f64 + 1.0) * x[i]).sin();
        }
    }

    // logslope_surface: 10 columns — SAME basis as marginal (the large-scale case:
    // marginal and logslope share the Duchon-PC basis evaluated at training
    // rows). The raw audit sees overlap=1.0 between marginal[:,k] and
    // logslope[:,k] for every k.
    let mut logslope = ndarray::Array2::<f64>::zeros((n, 10));
    for i in 0..n {
        for k in 0..10 {
            logslope[[i, k]] = ((k as f64 + 1.0) * x[i]).sin();
        }
    }

    // z_primary: per-row z values (PRS or score covariate, varies row-by-row).
    // The logslope block's effective design for identifiability is diag(z) ·
    // logslope: scaling by z[i] makes it linearly independent of marginal.
    let z_primary: ndarray::Array1<f64> =
        ndarray::Array1::from_iter((0..n).map(|i| 0.2 + 0.8 * i as f64 / (n as f64 - 1.0)));

    let make_spec = |name: &str, design: ndarray::Array2<f64>, priority: u8| {
        let n_rows = design.nrows();
        ParameterBlockSpec {
            name: name.to_string(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
            offset: Array1::<f64>::zeros(n_rows),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::<f64>::zeros(0),
            initial_beta: None,
            gauge_priority: priority,
            jacobian_callback: None,
            stacked_design: None,
            stacked_offset: None,
        }
    };

    // Flat audit on raw designs (no z-scaling): the marginal~logslope overlap
    // is 1.0.  With equal priorities this would be fatal.
    {
        let specs_equal_priority = [
            make_spec("time_surface", time.clone(), 100),
            make_spec("marginal_surface", marginal.clone(), 100),
            make_spec("logslope_surface", logslope.clone(), 100),
        ];
        let audit = audit_identifiability(&specs_equal_priority)
            .expect("flat audit on equal-priority must run");
        assert!(
            audit.fatal,
            "equal-priority alias (overlap=1.0) must still be fatal; summary: {}",
            audit.summary,
        );
    }

    // Flat audit with DISTINCT priorities but WITHOUT z-scaling: the raw
    // marginal~logslope overlap is still 1.0, but now gauge can resolve it.
    // Post-fix: fatal=false, drops attributed to logslope_surface.
    {
        let specs_distinct_priority = [
            make_spec("time_surface", time.clone(), 200),
            make_spec("marginal_surface", marginal.clone(), 150),
            make_spec("logslope_surface", logslope.clone(), 120),
        ];
        let audit = audit_identifiability(&specs_distinct_priority)
            .expect("flat audit with distinct priorities must run");
        assert!(
            !audit.fatal,
            "cross-block alias with distinct priorities must NOT be fatal (gauge resolves); \
             summary: {}",
            audit.summary,
        );
        assert!(
            !audit.dropped_columns.is_empty(),
            "gauge-resolved drops must be populated; summary: {}",
            audit.summary,
        );
        // Every drop must be from logslope_surface (lowest priority = 120),
        // never from time_surface (priority = 200) or marginal_surface (150).
        for drop in &audit.dropped_columns {
            assert_eq!(
                drop.block, "logslope_surface",
                "gauge must attribute drops to the lowest-priority block (logslope_surface); \
                 got drop on '{}' (summary: {})",
                drop.block, audit.summary,
            );
        }
        // Total kept = 33 − 10 = 23 (the 10 logslope columns are aliases
        // of the first 10 marginal columns and get dropped).
        let total_kept: usize = audit.blocks.iter().map(|b| b.effective_dim).sum();
        assert_eq!(
            total_kept, 23,
            "expected 33 - 10 = 23 kept columns; got {total_kept} (summary: {})",
            audit.summary,
        );
    }

    // Flat audit WITH z-scaling on logslope: the scaled rows are linearly
    // independent of the unscaled marginal rows (z varies row-by-row), so
    // the joint rank is full and no drops are needed.
    {
        let z_scaling: std::sync::Arc<[f64]> = std::sync::Arc::from(
            z_primary
                .as_slice()
                .expect("z_primary must be C-contiguous"),
        );
        let logslope_scaled_spec = ParameterBlockSpec {
            name: "logslope_surface".to_string(),
            design: DesignMatrix::Dense(DenseDesignMatrix::from(logslope.clone())),
            offset: Array1::<f64>::zeros(n),
            penalties: Vec::new(),
            nullspace_dims: Vec::new(),
            initial_log_lambdas: Array1::<f64>::zeros(0),
            initial_beta: None,
            gauge_priority: 120,
            jacobian_callback: Some(std::sync::Arc::new(RowScaledJacobian {
                design: std::sync::Arc::new(logslope.clone()),
                eta_scaling: z_scaling,
            })),
            stacked_design: None,
            stacked_offset: None,
        };
        let specs_with_z_scaling = [
            make_spec("time_surface", time, 200),
            make_spec("marginal_surface", marginal, 150),
            logslope_scaled_spec,
        ];
        let audit = audit_identifiability(&specs_with_z_scaling).expect("z-scaled audit must run");
        assert!(
            !audit.fatal,
            "z-scaled logslope audit must be non-fatal (diag(z)·basis is independent \
             of unscaled marginal basis when z varies); summary: {}",
            audit.summary,
        );
        // With z-scaling the joint design is full rank: all 33 columns kept.
        let total_kept: usize = audit.blocks.iter().map(|b| b.effective_dim).sum();
        assert_eq!(
            total_kept, 33,
            "z-scaled logslope: all 33 columns must be kept (full rank); \
             got {total_kept} (summary: {})",
            audit.summary,
        );
    }
}
