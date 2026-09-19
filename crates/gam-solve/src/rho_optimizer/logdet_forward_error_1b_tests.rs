//! #2954 stage 1b: a certificate charges no log-determinant forward error the
//! object that priced the criterion's `log|H|` did not derive.

use crate::estimate::reml::reml_outer_engine::PenaltySubspaceTrace;
use ndarray::Array2;

fn kernel(logdet_correction: f64) -> PenaltySubspaceTrace {
    PenaltySubspaceTrace {
        u_s: Array2::eye(2),
        h_proj_inverse: Array2::eye(2),
        logdet_correction,
    }
}

#[test]
fn a_kernel_that_replaces_the_determinant_charges_no_operator_bound_2954() {
    let operator = Some(3.0e-15);
    // The criterion's `log|H|` is the kernel's pseudo-determinant, which carries
    // no derived bound: none is charged, so no verdict is taken, rather than the
    // operator's bound for a determinant the criterion does not read.
    assert_eq!(kernel(-0.25).determinant_forward_error(operator), None);
    assert_eq!(kernel(-0.25).determinant_forward_error(None), None);
    // A kernel that adds nothing to the operator's determinant leaves the
    // operator's bound in force.
    assert_eq!(kernel(0.0).determinant_forward_error(operator), operator);
}
