//! Tests for the MAP estimate uniqueness condition:
//!   ker(J^T W J) ∩ ker(S) = {0}
//!
//! Construction:
//!
//! n = 32 observations.
//! v_shared = constant vector (all ones).
//! v_a = linramp = [1, 2, ..., n].
//! v_b = 2 * linramp (proportional to v_a).
//!
//! Block A: [v_shared, v_a]   (2 raw columns).
//! Block B: [v_shared, v_b]   (2 raw columns; v_shared is aliased with A col 0).
//!
//! A at gauge_priority = 200, B at gauge_priority = 100.
//!
//! After `canonicalize_for_identifiability`:
//!   - B col 0 (v_shared) is aliased with A col 0 (v_shared) → B col 0 dropped.
//!   - After T: A has [v_shared, v_a] (2 cols), B has [v_b = 2*v_a] (1 col).
//!   - J_can columns: [v_shared, v_a, 2*v_a] → rank 2.
//!
//! Null direction of J_can^T J_can (in reduced [v_shared, v_a, v_b] space):
//!   v_a * (-2) + v_b * 1 = 0  →  n = (0, -2, 1) / sqrt(5).
//!   n_A = (0, -2/sqrt(5))  (A's reduced [v_shared, v_a] components)
//!   n_B = (1/sqrt(5))       (B's reduced [v_b] component)
//!
//! PASS case:
//!   A's raw penalty = [[0,0],[0,4]]:  penalises v_a (col 1 of A).
//!   B's raw penalty = [[0,0],[0,1]]:  penalises v_b = col 1 of raw B
//!                     (after T, kept col = raw B col 1, reduced penalty = [[1]]).
//!   n^T S n = n_A^T [[0,0],[0,4]] n_A  +  n_B^T [[1]] n_B
//!           = 4*(4/5) + 1*(1/5) = 16/5 + 1/5 = 17/5 > 0.  PASS.
//!
//! FAIL case:
//!   A's raw penalty = [[1,0],[0,0]]:  penalises v_shared (col 0 of A), NOT v_a.
//!   B's raw penalty = [[0,0],[0,0]]:  zero.
//!   After T, B's reduced penalty = [[0]].
//!   n_A^T [[1,0],[0,0]] n_A = 1 * 0² = 0  (n_A[0] = 0).
//!   n_B^T [[0]] n_B = 0.
//!   n^T S n = 0.  MAP non-unique → FAIL.

use gam::families::custom_family::{CustomFamilyError, ParameterBlockSpec, PenaltyMatrix};
use gam::linalg::matrix::{DenseDesignMatrix, DesignMatrix};
use gam::solver::identifiability_canonical::canonicalize_for_identifiability;
use ndarray::{Array1, Array2};

const N: usize = 32;

fn build_spec(
    name: &str,
    design: Array2<f64>,
    penalty: Array2<f64>,
    priority: u8,
) -> ParameterBlockSpec {
    let n = design.nrows();
    ParameterBlockSpec {
        name: name.to_string(),
        design: DesignMatrix::Dense(DenseDesignMatrix::from(design)),
        offset: Array1::<f64>::zeros(n),
        penalties: vec![PenaltyMatrix::Dense(penalty)],
        nullspace_dims: Vec::new(),
        initial_log_lambdas: Array1::from(vec![0.0]),
        initial_beta: None,
        gauge_priority: priority,
        row_scaling: None,
        jacobian_callback: None,
    }
}

/// Block A design: [v_shared = ones, v_a = linramp].
fn design_a() -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        m[[i, 0]] = 1.0;           // v_shared
        m[[i, 1]] = (i + 1) as f64; // v_a = linramp
    }
    m
}

/// Block B design: [v_shared = ones, v_b = 2 * linramp].
fn design_b() -> Array2<f64> {
    let mut m = Array2::<f64>::zeros((N, 2));
    for i in 0..N {
        m[[i, 0]] = 1.0;                  // v_shared (aliased with A col 0)
        m[[i, 1]] = 2.0 * (i + 1) as f64; // v_b = 2 * linramp
    }
    m
}

/// PASS: both penalties cover the post-canonicalize null direction.
///
/// After canonicalize:
///   A reduced = [[v_shared, v_a]], penalty = [[0,0],[0,4]].
///   B reduced = [[v_b]], penalty = [[1]] (from raw [[0,0],[0,1]] pulled back to col 1).
///
/// Null direction n = (0, -2, 1)/sqrt(5).
///   n^T S n = 4*(4/5) + 1*(1/5) = 17/5 > 0. PASS.
#[test]
fn penalty_joint_nullspace_check_passes_when_both_penalties_cover_null() {
    // A's raw penalty = [[0, 0], [0, 4]] — penalises v_a (col 1).
    let mut pen_a = Array2::<f64>::zeros((2, 2));
    pen_a[[1, 1]] = 4.0;

    // B's raw penalty = [[0, 0], [0, 1]] — col 1 is v_b.
    // After T drops B col 0, kept=[1], so reduced B penalty = [[raw[1,1]]] = [[1]].
    let mut pen_b = Array2::<f64>::zeros((2, 2));
    pen_b[[1, 1]] = 1.0;

    let spec_a = build_spec("block_a", design_a(), pen_a, 200);
    let spec_b = build_spec("block_b", design_b(), pen_b, 100);

    let result = canonicalize_for_identifiability(&[spec_a, spec_b]);
    assert!(
        result.is_ok(),
        "MAP uniqueness check must pass when both penalties cover the null direction; \
         got: {:?}",
        result.err(),
    );
}

/// FAIL: S = 0 on the post-canonicalize null direction.
///
/// After canonicalize:
///   A reduced = [[v_shared, v_a]], penalty = [[1,0],[0,0]] (only v_shared penalised).
///   B reduced = [[v_b]], penalty = [[0]] (from raw [[0,0],[0,0]] pulled back to col 1).
///
/// Null direction n = (0, -2, 1)/sqrt(5).
///   n_A = (0, -2/sqrt(5)) → n_A^T [[1,0],[0,0]] n_A = 0.
///   n_B = (1/sqrt(5))    → n_B^T [[0]] n_B = 0.
///   n^T S n = 0. MAP non-unique.
///
/// Expected: `MapUniquenessFailure` (or `IdentifiabilityFailure` if the audit
/// independently refuses the model for rank-deficiency reasons).
#[test]
fn penalty_joint_nullspace_check_fails_when_s_zero_on_null_direction() {
    // A's raw penalty = [[1,0],[0,0]] — only v_shared penalised, NOT v_a.
    let mut pen_a = Array2::<f64>::zeros((2, 2));
    pen_a[[0, 0]] = 1.0;

    // B's raw penalty = [[0,0],[0,0]] — zero everywhere.
    let pen_b = Array2::<f64>::zeros((2, 2));

    let spec_a = build_spec("block_a", design_a(), pen_a, 200);
    let spec_b = build_spec("block_b", design_b(), pen_b, 100);

    let result = canonicalize_for_identifiability(&[spec_a, spec_b]);

    match result {
        Err(CustomFamilyError::MapUniquenessFailure { ref error }) => {
            assert!(
                error.dominant_block == "block_a" || error.dominant_block == "block_b",
                "dominant block must be block_a or block_b; got '{}'",
                error.dominant_block,
            );
            assert!(
                error.penalty_quadratic_form < 1e-8,
                "penalty quadratic form must be near zero on the null direction; got {}",
                error.penalty_quadratic_form,
            );
        }
        Err(CustomFamilyError::IdentifiabilityFailure { ref audit }) => {
            // The cross-block audit also catches the rank deficiency with gauge
            // resolution in some code paths.  Acceptable — the model IS non-unique.
            assert!(
                audit.fatal,
                "IdentifiabilityFailure audit must be fatal; got {}",
                audit.summary,
            );
        }
        Ok(_) => {
            panic!(
                "MAP uniqueness check must not pass when S = 0 on the null direction; \
                 got Ok (check silently accepted a non-unique model)"
            );
        }
        Err(other) => {
            panic!(
                "expected MapUniquenessFailure or IdentifiabilityFailure; got {:?}",
                other,
            );
        }
    }
}
