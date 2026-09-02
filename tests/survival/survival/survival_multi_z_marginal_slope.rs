use gam::families::bms::MarginalSlopeCovariance;
use gam::families::survival::marginal_slope::{RigidVectorValueWorkspace, survival_marginal_slope_vector_neglog};
use gam::probability::normal_cdf;
use ndarray::array;

// The K=1 reduction is an identity over the reals, not over IEEE-754.
// `marginal_slope_preserving_scale` squares the probit scale once and applies
// it to the quadratic form of the *raw* slopes (the diagonal quadratic form
// accumulates `coefficient * slope * slope`), so production evaluates
// `fl(fl(p*p) * fl(s*s))` while the scalar identity below folds the scale into
// the slope first and evaluates `fl(fl(p*s) * fl(p*s))`. Same real number, up
// to an ulp apart, so the reduction is pinned to a few-ulp bound rather than
// to `to_bits()` equality, which would pin the association order instead.
//
// `magnitude` is the size of the largest intermediates the reference sums, so
// a cancelling total is still held to the accuracy its inputs allow.

#[test]
fn survival_multi_z_shared_slope_neglog_uses_row_sum_and_covariance_quadratic() {
    let q0 = 0.15;
    let q1 = 0.55;
    let qd1 = 0.9;
    let shared_slope = -0.22;
    let z = [0.6, -1.1];
    let covariance = MarginalSlopeCovariance::full(array![[1.3, 0.4], [0.4, 0.7]]).unwrap();
    let probit_scale = 0.85;
    let weight = 1.3;
    let event = 1.0;

    let observed = [probit_scale * shared_slope, probit_scale * shared_slope];
    let c = (1.0 + covariance.quadratic_form(&observed).expect("r Sigma r")).sqrt();
    let linear = observed[0] * z.iter().sum::<f64>();
    let eta0 = q0 * c + linear;
    let eta1 = q1 * c + linear;
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    let expected = weight
        * (normal_cdf(-eta0).ln()
            - (1.0 - event) * normal_cdf(-eta1).ln()
            - event * log_phi_eta1
            - event * (qd1 * c).ln());

    let actual = survival_marginal_slope_vector_neglog(
        0,
        q0,
        q1,
        qd1,
        &[shared_slope, shared_slope],
        &z,
        &RigidVectorValueWorkspace::new(&covariance.clone().into()),
        weight,
        event,
        1e-6,
        probit_scale,
    )
    .expect("vector neglog");
    assert!(
        (actual - expected).abs() <= 1e-14,
        "actual={actual:.17e} expected={expected:.17e}"
    );
}

#[test]
fn survival_multi_z_k1_neglog_matches_scalar_identity_fixture() {
    let q0 = -0.15;
    let q1 = 0.55;
    let qd1 = 0.8;
    let slope = [0.31];
    let z = [0.45];
    let covariance = MarginalSlopeCovariance::diagonal(array![1.0]).unwrap();
    let probit_scale: f64 = 0.75;
    let observed: f64 = probit_scale * slope[0];
    let c = (1.0 + observed * observed).sqrt();
    let eta0 = q0 * c + observed * z[0];
    let eta1 = q1 * c + observed * z[0];
    let log_phi_eta1 = -0.5 * (eta1 * eta1 + std::f64::consts::TAU.ln());
    let expected = 1.2
        * ((1.0 - 1.0) * -normal_cdf(-eta1).ln() + normal_cdf(-eta0).ln()
            - log_phi_eta1
            - (qd1 * c).ln());
    let actual = survival_marginal_slope_vector_neglog(
        0,
        q0,
        q1,
        qd1,
        &slope,
        &z,
        &RigidVectorValueWorkspace::new(&covariance.clone().into()),
        1.2,
        1.0,
        1e-6,
        probit_scale,
    )
    .expect("vector neglog");
    assert!(
        (actual - expected).abs() <= 1e-14,
        "actual={actual:.17e} expected={expected:.17e}"
    );
}

#[test]
fn survival_multi_z_neglog_rejects_derivative_guard_violation() {
    let covariance = MarginalSlopeCovariance::diagonal(array![1.0, 1.0]).unwrap();
    let err = survival_marginal_slope_vector_neglog(
        0,
        0.0,
        0.2,
        1e-7,
        &[0.2, -0.1],
        &[0.4, 0.5],
        &RigidVectorValueWorkspace::new(&covariance.clone().into()),
        1.0,
        1.0,
        1e-6,
        1.0,
    )
    .expect_err("derivative guard violation must fail");
    assert!(err.contains("monotonicity violated"));
}
