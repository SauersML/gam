//! Tests for the MAP estimate uniqueness condition:
//!   ker(J^T W J) ∩ ker(S) = {0}
//!
//! # Direct `check_map_uniqueness` tests
//!
//! Tests call `check_map_uniqueness` directly with a controlled (n, p) design
//! matrix J that is rank-deficient by 1. The two blocks' column offsets are
//! wired by hand.
//!
//! ## 2-block construction
//!
//! n = 32 observations, p_total = 3 (block A: 2 cols, block B: 1 col).
//!
//! J columns:
//!   col 0 (block A, col 0): v_shared = [1, 1, ..., 1]
//!   col 1 (block A, col 1): v_a      = [1, 2, ..., 32]
//!   col 2 (block B, col 0): v_b      = [2, 4, ..., 64]  (= 2 * v_a)
//!
//! J has rank 2 (v_b = 2 * v_a). Null direction: n = (0, -2, 1)/sqrt(5).
//!   n_A = (0, -2/sqrt(5))  in block A's column space
//!   n_B = (1/sqrt(5))      in block B's column space
//!
//! PASS: S_A = [[0,0],[0,4]], S_B = [[1]].
//!   n_A^T S_A n_A = 4 * 4/5 = 16/5.
//!   n_B^T S_B n_B = 1 * 1/5 = 1/5.
//!   n^T S n = 17/5 > 0.  CHECK PASSES.
//!
//! FAIL: S_A = [[1,0],[0,0]] (penalises v_shared only), S_B = [[0]].
//!   n_A = (0, -2/sqrt(5))  →  n_A^T [[1,0],[0,0]] n_A = 0  (n_A[0] = 0).
//!   n_B^T [[0]] n_B = 0.
//!   n^T S n = 0.  CHECK FAILS.
//!
//! # `canonicalize_for_identifiability` end-to-end test
//!
//! A clean 2-block model (no aliasing) with full-rank penalties on both
//! blocks passes canonicalize without triggering any failure.

use gam::families::custom_family::{ParameterBlockSpec, PenaltyMatrix};
use gam::identifiability::audit::{MapUniquenessError, check_map_uniqueness};
use gam::identifiability::canonical::canonicalize_for_identifiability;
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use ndarray::{Array1, Array2};

const N: usize = 32;

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Build J = [v_shared | v_a | v_b] where v_b = 2 * v_a.
/// Shape (N, 3), rank 2.  Null direction: n = (0, -2, 1)/sqrt(5).
fn build_rank_deficient_j() -> Array2<f64> {
    let mut j = Array2::<f64>::zeros((N, 3));
    for i in 0..N {
        j[[i, 0]] = 1.0; // v_shared
        j[[i, 1]] = (i + 1) as f64; // v_a = linramp
        j[[i, 2]] = 2.0 * (i + 1) as f64; // v_b = 2 * v_a
    }
    j
}

/// Build dummy specs and col_offsets for the 3-column design:
///   block_a: cols [0, 1] (2 cols), block_b: col [2] (1 col).
fn build_two_block_specs() -> (Vec<ParameterBlockSpec>, Vec<usize>) {
    let mut design_a = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        design_a[[i, 0]] = 1.0;
        design_a[[i, 1]] = (i + 1) as f64;
    }
    let mut design_b = Array2::<f64>::zeros((N, 1));
    for i in 0..N {
        design_b[[i, 0]] = 2.0 * (i + 1) as f64;
    }
    let spec_a = ParameterBlockSpec {
        name: "block_a".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design_a)),
        offset: Array1::<f64>::zeros(N),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let spec_b = ParameterBlockSpec {
        name: "block_b".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design_b)),
        offset: Array1::<f64>::zeros(N),
        penalties: Vec::new(),
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::<f64>::zeros(0),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let col_offsets = vec![0usize, 2, 3];
    (vec![spec_a, spec_b], col_offsets)
}

// ── Direct check_map_uniqueness tests ────────────────────────────────────────

/// PASS: 2-block rank-deficient J AND non-trivial S covering the null direction.
///
/// n = (0, -2, 1)/sqrt(5).
/// S = blockdiag([[0,0],[0,4]], [[1]]).
/// n^T S n = 16/5 + 1/5 = 17/5 > 0 → check passes.
#[test]
fn map_uniqueness_check_passes_with_covering_penalty() {
    let j = build_rank_deficient_j();
    // S = blockdiag([[0,0],[0,4]], [[1]]).
    let mut s = Array2::<f64>::zeros((3, 3));
    s[[1, 1]] = 4.0; // S_A[1,1]
    s[[2, 2]] = 1.0; // S_B[0,0]
    let (specs, col_offsets) = build_two_block_specs();

    let result: Result<(), MapUniquenessError> =
        check_map_uniqueness(&j, &[], &s, &specs, &col_offsets);

    assert!(
        result.is_ok(),
        "MAP uniqueness check must pass when both block penalties cover the null \
         direction n = (0,-2,1)/sqrt(5); n^T S n = 17/5 > 0; got: {:?}",
        result.err(),
    );
}

/// FAIL: 2-block rank-deficient J AND S = 0 on the null direction.
///
/// S_A = [[1,0],[0,0]] penalises v_shared only (n_A[0] = 0 → no contribution).
/// S_B = [[0]] zero penalty.
/// n^T S n = 0 → check fails, naming the dominant block.
#[test]
fn map_uniqueness_check_fails_when_s_zero_on_null_direction() {
    let j = build_rank_deficient_j();
    // S = blockdiag([[1,0],[0,0]], [[0]]) — zero on the null direction.
    let mut s = Array2::<f64>::zeros((3, 3));
    s[[0, 0]] = 1.0; // S_A[0,0] — penalises v_shared; null direction has n[0]=0
    let (specs, col_offsets) = build_two_block_specs();

    let result: Result<(), MapUniquenessError> =
        check_map_uniqueness(&j, &[], &s, &specs, &col_offsets);

    match result {
        Err(err) => {
            assert!(
                err.dominant_block == "block_a" || err.dominant_block == "block_b",
                "dominant block must be block_a or block_b; got '{}'",
                err.dominant_block,
            );
            assert!(
                err.penalty_quadratic_form < 1e-8,
                "penalty quadratic form must be near zero for the failing null direction; \
                 got {}",
                err.penalty_quadratic_form,
            );
            assert!(
                err.message.contains(&err.dominant_block),
                "error message must contain the dominant block name; message: '{}'",
                err.message,
            );
        }
        Ok(()) => {
            panic!(
                "MAP uniqueness check must fail when S = 0 on the null direction; \
                 null direction n = (0,-2,1)/sqrt(5) has n^T S n = 0; check passed incorrectly"
            );
        }
    }
}

// ── canonicalize_for_identifiability end-to-end tests ────────────────────────

/// A clean 2-block model (no aliasing, full-rank J, non-trivial penalties)
/// passes `canonicalize_for_identifiability` without any error.
#[test]
fn canonicalize_clean_model_with_penalties_passes_map_check() {
    // Block A: [v0 = linramp, v1 = sin(x)] — 2 independent columns.
    // Block B: [v2 = cos(x), v3 = exp(-x)] — 2 independent columns.
    // No cross-block aliasing. J is full column rank.
    let mut design_a = Array2::<f64>::zeros((N, 2));
    let mut design_b = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        let x = (i as f64 + 1.0) * std::f64::consts::PI / (N as f64);
        design_a[[i, 0]] = (i + 1) as f64;
        design_a[[i, 1]] = x.sin();
        design_b[[i, 0]] = x.cos();
        design_b[[i, 1]] = (-0.1 * (i as f64)).exp();
    }
    // Diagonal penalties: cover all directions.
    let mut pen_a = Array2::<f64>::zeros((2, 2));
    pen_a[[0, 0]] = 1.0;
    pen_a[[1, 1]] = 1.0;
    let mut pen_b = Array2::<f64>::zeros((2, 2));
    pen_b[[0, 0]] = 1.0;
    pen_b[[1, 1]] = 1.0;

    let spec_a = ParameterBlockSpec {
        name: "block_a".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design_a)),
        offset: Array1::<f64>::zeros(N),
        penalties: vec![PenaltyMatrix::Dense(pen_a)],
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::from(vec![0.0]),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };
    let spec_b = ParameterBlockSpec {
        name: "block_b".to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design_b)),
        offset: Array1::<f64>::zeros(N),
        penalties: vec![PenaltyMatrix::Dense(pen_b)],
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::from(vec![0.0]),
        initial_beta: None,
        gauge_priority: 100,
        jacobian_callback: None,
        stacked_design: None,
        stacked_offset: None,
    };

    let result = canonicalize_for_identifiability(&[spec_a, spec_b]);
    assert!(
        result.is_ok(),
        "clean 2-block model with full-rank J and non-trivial penalties must pass \
         all checks including MAP uniqueness; got: {:?}",
        result.err(),
    );
}
