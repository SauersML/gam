use gam::survival::calculate_crude_risk_quadrature;
use ndarray::array;

fn crude_risk_constant_hazard(lambda_d: f64, lambda_m: f64, t0: f64, t1: f64) -> f64 {
    let h_dis_t0 = lambda_d * (t0 + 1.0);
    let h_mor_t0 = lambda_m * (t0 + 1.0);
    let design_d_t0 = array![1.0];
    let design_m_t0 = array![1.0];

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
            design_d[0] = 1.0;
            deriv_d[0] = 0.0;
            design_m[0] = 1.0;
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
    // --- Existing monotonicity check, with diagnostic message ---
    for (i, w) in risks.windows(2).enumerate() {
        assert!(
            w[1] + 1e-12 >= w[0],
            "crude risk must be non-decreasing in horizon: risk(t={})={} > risk(t={})={}",
            horizons[i],
            w[0],
            horizons[i + 1],
            w[1],
        );
    }

    // --- New: probability-bound contract ---
    // Crude risk is a probability of the disease event in [t0, t1], so it
    // must lie in [0, 1] for every horizon.
    for (h, r) in horizons.iter().zip(risks.iter()) {
        assert!(
            (0.0..=1.0).contains(r),
            "crude risk at t1={h} fell outside [0,1]: {r}"
        );
    }

    // --- New: short-horizon limit ---
    // For a horizon barely above t0, the crude risk integrates a tiny
    // interval, so it must be near zero (and bounded above by lambda_d *
    // (t1-t0) for any constant-hazard cause).
    let short_t1 = t0 + 1e-3;
    let short_risk = crude_risk_constant_hazard(lambda_d, lambda_m, t0, short_t1);
    assert!(
        short_risk < 1e-3,
        "crude risk over a tiny horizon must vanish: t0={t0}, t1={short_t1}, risk={short_risk}"
    );

    // --- New: long-horizon limit ---
    // As t1 - t0 -> infty, crude risk for the disease cause approaches the
    // competing-risks share lambda_d / (lambda_d + lambda_m).
    let long_t1 = t0 + 500.0;
    let long_risk = crude_risk_constant_hazard(lambda_d, lambda_m, t0, long_t1);
    let asymptote = lambda_d / (lambda_d + lambda_m);
    assert!(
        (long_risk - asymptote).abs() < 1e-3,
        "long-horizon crude risk must approach lambda_d/(lambda_d + lambda_m) = {asymptote}; \
         got {long_risk} at t1={long_t1}"
    );

    // --- New: comparative statics in lambda_d / lambda_m ---
    // Increasing the disease hazard at fixed competing-mortality hazard must
    // strictly increase the disease-specific crude risk over a finite
    // horizon.
    let baseline = crude_risk_constant_hazard(lambda_d, lambda_m, t0, 60.0);
    let higher_disease = crude_risk_constant_hazard(2.0 * lambda_d, lambda_m, t0, 60.0);
    assert!(
        higher_disease > baseline,
        "increasing disease hazard must increase disease-specific crude risk: \
         baseline={baseline}, doubled-disease={higher_disease}"
    );

    // Conversely, raising the competing-mortality hazard at fixed disease
    // hazard must reduce the disease-specific crude risk (more people die
    // of competing causes before the disease event).
    let higher_mortality = crude_risk_constant_hazard(lambda_d, 4.0 * lambda_m, t0, 60.0);
    assert!(
        higher_mortality < baseline,
        "increasing competing mortality must reduce disease-specific crude risk: \
         baseline={baseline}, raised-mortality={higher_mortality}"
    );
}
