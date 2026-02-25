use gam::survival::{CrudeRiskResult, calculate_crude_risk_quadrature};
use ndarray::array;

fn run_constant_hazard_crude(
    t0: f64,
    t1: f64,
    log_lambda_d: f64,
    log_lambda_m: f64,
) -> CrudeRiskResult {
    let lambda_d = log_lambda_d.exp();
    let lambda_m = log_lambda_m.exp();
    let h_dis_t0 = lambda_d * (t0 + 1.0);
    let h_mor_t0 = lambda_m * (t0 + 1.0);
    let design_d_t0 = array![h_dis_t0.ln()];
    let design_m_t0 = array![h_mor_t0.ln()];

    calculate_crude_risk_quadrature(
        t0,
        t1,
        &[t0, t1],
        h_dis_t0,
        h_mor_t0,
        design_d_t0.view(),
        design_m_t0.view(),
        |u, design_d, deriv_d, design_m| {
            let h_d = lambda_d * (u + 1.0);
            let h_m = lambda_m * (u + 1.0);
            design_d[0] = h_d.ln();
            deriv_d[0] = 1.0 / (u + 1.0);
            design_m[0] = h_m.ln();
            Ok((lambda_d, h_d, h_m))
        },
    )
    .expect("crude-risk quadrature should succeed")
}

#[test]
fn crude_risk_matches_constant_hazard_analytic_solution() {
    let lambda_d = 0.1_f64;
    let lambda_m = 0.1_f64;
    let t0 = 0.0_f64;
    let t1 = 10.0_f64;
    let result = run_constant_hazard_crude(t0, t1, lambda_d.ln(), lambda_m.ln());
    let expected =
        (lambda_d / (lambda_d + lambda_m)) * (1.0 - (-(lambda_d + lambda_m) * (t1 - t0)).exp());
    assert!((result.risk - expected).abs() < 2e-3);
}

#[test]
fn crude_risk_zero_mortality_and_high_mortality_limits() {
    let t0 = 0.0_f64;
    let t1 = 10.0_f64;

    let zero_mort = run_constant_hazard_crude(t0, t1, 0.1_f64.ln(), (1e-12_f64).ln());
    let expected_zero = 1.0 - (-(0.1_f64) * (t1 - t0)).exp();
    assert!((zero_mort.risk - expected_zero).abs() < 2e-3);

    let high_mort = run_constant_hazard_crude(t0, t1, 0.1_f64.ln(), 100.0_f64.ln());
    assert!(high_mort.risk < 2e-3);
}
