//! TEMPORARY #1561 diagnostic (revert after reading CI log). Fits the exact
//! driver fixture (no gamlss/R) and PANICS with the scale-block smoothing
//! parameters, EDF, and inner/outer convergence — the suspect-#2 data needed to
//! see WHY gam over-selects λ_σ. CI surfaces panic output via
//! `failure-output = "immediate-final"`; grep the log for `DIAG_1561`.
use gam::estimate::BlockRole;
use gam::gamlss::GaussianLocationScaleFitResult;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};

#[test]
fn zz_diag_1561_lambda_dump() {
    init_parallelism();
    let n = 200usize;
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut state: u64 = 42;
    let mut next_unit = || -> f64 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x: Vec<f64> = (0..n).map(|_| next_unit()).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut z: Vec<f64> = Vec::with_capacity(n);
    while z.len() < n {
        let u1 = next_unit().max(1e-300);
        let u2 = next_unit();
        let r = (-2.0 * u1.ln()).sqrt();
        z.push(r * (two_pi * u2).cos());
        if z.len() < n {
            z.push(r * (two_pi * u2).sin());
        }
    }
    let mu_true = |t: f64| (two_pi * t).sin();
    let sigma_true = |t: f64| 0.1 + 0.2 * (two_pi * t).sin();
    let y: Vec<f64> = (0..n).map(|i| mu_true(x[i]) + sigma_true(x[i]) * z[i]).collect();

    let headers: Vec<String> = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = (0..n)
        .map(|i| csv::StringRecord::from(vec![format!("{:.17e}", x[i]), format!("{:.17e}", y[i])]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some("1 + s(x, bs='tp')".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, bs='tp')", &ds, &cfg).expect("gam fit");
    let FitResult::GaussianLocationScale(GaussianLocationScaleFitResult { fit, .. }) = result else {
        panic!("expected Gaussian location-scale fit");
    };
    let u = &fit.fit;
    let scale = u.block_by_role(BlockRole::Scale).expect("scale block");
    let mean = u.block_by_role(BlockRole::Location).expect("mean block");

    panic!(
        "DIAG_1561: scale_lambdas={:?} scale_edf={:.5} scale_p={} | mean_lambdas={:?} mean_edf={:.5} mean_p={} | all_lambdas={:?} | outer_converged={} outer_iters={} inner_cycles={} pirls_status={:?} reml_score={:.6}",
        scale.lambdas.to_vec(),
        scale.edf,
        scale.beta.len(),
        mean.lambdas.to_vec(),
        mean.edf,
        mean.beta.len(),
        u.lambdas.to_vec(),
        u.outer_converged,
        u.outer_iterations,
        u.inner_cycles,
        u.pirls_status,
        u.reml_score,
    );
}
