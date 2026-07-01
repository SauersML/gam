use gam::families::survival::location_scale::project_onto_linear_constraints;
use gam::solver::pirls::LinearInequalityConstraints;
use ndarray::{Array1, array};

#[test]
fn project_onto_linear_constraints_keeps_already_feasible_one_sided_seed() {
    let constraints = LinearInequalityConstraints {
        a: array![[1.0, 0.0], [0.0, 1.0]],
        b: array![0.0, 0.0],
    };
    let v = Array1::from_vec(vec![2.5, 3.5]);

    let projected: Array1<f64> = project_onto_linear_constraints(2, &constraints, Some(&v))
        .expect("dimensionally consistent projection must succeed");

    for (i, &value) in projected.iter().enumerate() {
        assert!(
            (value - v[i]).abs() <= 1e-12,
            "projection onto the one-sided cone β >= 0 should leave an already-feasible seed \
             unchanged; coordinate {i} moved from {} to {value}",
            v[i],
        );
    }
}

#[test]
fn project_onto_linear_constraints_enforces_equalities_encoded_as_opposing_inequalities() {
    let constraints = LinearInequalityConstraints {
        a: array![[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        b: array![0.0, 0.0, 0.0, 0.0],
    };
    let v = Array1::from_vec(vec![2.5, 3.5]);

    let projected: Array1<f64> = project_onto_linear_constraints(2, &constraints, Some(&v))
        .expect("dimensionally consistent equality-pair projection must succeed");

    for row in 0..constraints.a.nrows() {
        let residual = projected.dot(&constraints.a.row(row)) - constraints.b[row];
        assert!(
            residual.abs() <= 1e-10,
            "opposing inequality pair should encode an equality; row {row} residual was {residual} \
             for projected={projected:?}"
        );
    }
}

/// Regression for issue #374: a survival marginal-slope fit with
/// `logslope_formula="1"` fired the rigid pilot, which seeded an
/// identifiability-reduced `time_beta` hint whose length no longer matched the
/// raw `design_exit.ncols()` used to build the time-block constraints. That
/// hint was fed straight into `project_onto_linear_constraints`, where
/// `&beta + &corrections.row(i)` broadcast a length-`beta0.len()` vector
/// against a length-`dim` row and ndarray's `unwrap()` turned the
/// `IncompatibleShape` error into a Rust panic that crossed the Python
/// boundary as `GamError("... panicked inside Rust boundary ...")`.
///
/// The projection must now validate its operands up front and return a
/// structured `Err` on a `beta0.len() != dim` mismatch instead of panicking.
#[test]
fn project_onto_linear_constraints_rejects_beta0_dim_mismatch_without_panicking() {
    // Constraints live in R^3 (dim = 3), but the warm-start hint is the
    // shorter, reduced-dimension β a pilot would have produced.
    let constraints = LinearInequalityConstraints {
        a: array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        b: array![0.0, 0.0, 0.0],
    };
    let stale_reduced_hint = Array1::from_vec(vec![0.3, -0.1]); // length 2, not 3

    let result = project_onto_linear_constraints(3, &constraints, Some(&stale_reduced_hint));

    let err = result.expect_err(
        "a length-2 hint projected into a 3-dimensional constraint space must be a structured \
         error, never a panic or a silently truncated/padded vector",
    );
    assert!(
        err.contains("beta0 length") && err.contains("2") && err.contains("dim 3"),
        "expected a dimension-mismatch message naming the offending lengths, got: {err}"
    );

    // And the happy path with a correctly-sized hint still projects into the
    // feasible cone {x : x >= 0}.
    let good_hint = Array1::from_vec(vec![0.3, -0.1, 0.5]);
    let projected = project_onto_linear_constraints(3, &constraints, Some(&good_hint))
        .expect("a correctly-sized hint must project successfully");
    for (i, &v) in projected.iter().enumerate() {
        assert!(
            v.is_finite() && v >= -1e-9,
            "projected coordinate {i} should be feasible (>= 0), got {v}"
        );
    }
}
