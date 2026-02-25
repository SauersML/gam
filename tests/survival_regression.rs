use gam::survival::calculate_crude_risk_quadrature;
use ndarray::array;

fn crude_risk_constant_hazard(lambda_d: f64, lambda_m: f64, t0: f64, t1: f64) -> f64 {
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
    .risk
}

#[test]
fn cumulative_incidence_constant_hazard_grid_matches_closed_form() {
    let lambda_d = 0.12_f64;
    let lambda_m = 0.05_f64;
    let t0 = 50.0_f64;
    let horizons = [58.0_f64, 61.0_f64, 64.0_f64, 67.0_f64];

    for &t1 in &horizons {
        let computed = crude_risk_constant_hazard(lambda_d, lambda_m, t0, t1);
        let expected =
            (lambda_d / (lambda_d + lambda_m)) * (1.0 - (-(lambda_d + lambda_m) * (t1 - t0)).exp());
        assert!(
            (computed - expected).abs() < 2e-3,
            "risk mismatch at t1={t1}: computed={computed} expected={expected}"
        );
    }
}

#[test]
fn crude_risk_is_monotone_in_horizon() {
    let lambda_d = 0.08_f64;
    let lambda_m = 0.04_f64;
    let t0 = 45.0_f64;
    let horizons = [48.0_f64, 52.0_f64, 56.0_f64, 60.0_f64];
    let mut risks = Vec::with_capacity(horizons.len());
    for &t1 in &horizons {
        risks.push(crude_risk_constant_hazard(lambda_d, lambda_m, t0, t1));
    }
    assert!(risks.windows(2).all(|w| w[1] + 1e-12 >= w[0]));
}
