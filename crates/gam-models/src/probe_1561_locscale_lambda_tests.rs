//! Fast gam-models-level probe for #1561 (Gaussian location-scale over-smooths
//! the log-σ surface). Calls `fit_from_formula` directly so it compiles ONLY
//! gam-models (immune to the gam-umbrella recompile churn). Measures
//! pearson(fitted logσ, true logσ), rmse, the selected lambdas, the REML score,
//! and the per-block EDF under three genuinely distinct penalty configs:
//!   (a) shipped default (mean double penalty on, scale double penalty off),
//!   (b) explicitly add the scale-block null-space double penalty,
//!   (c) disable the double penalty on both blocks.
//! It is a measurement probe (not a permanent quality gate): each config must
//! still produce a finite, positively-correlated log-σ surface.
#![cfg(test)]

use crate::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_data::encode_recordswith_inferred_schema;
use gam_linalg::matrix::LinearOperator;
use gam_solve::estimate::BlockRole;

const LOGB_SIGMA_FLOOR: f64 = 0.01;

fn next_unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

fn mean_truth(x: f64) -> f64 {
    (2.0 * std::f64::consts::PI * x).sin()
}
fn sigma_truth(x: f64) -> f64 {
    0.1 + 0.2 * (2.0 * std::f64::consts::PI * x).sin()
}

fn locscale_data(n: usize) -> (Vec<f64>, Vec<f64>) {
    let two_pi = 2.0 * std::f64::consts::PI;
    let mut state = 42_u64;
    let mut x: Vec<f64> = (0..n).map(|_| next_unit(&mut state)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).expect("finite x"));
    let mut z = Vec::with_capacity(n);
    while z.len() < n {
        let u1 = next_unit(&mut state).max(1e-300);
        let u2 = next_unit(&mut state);
        let r = (-2.0 * u1.ln()).sqrt();
        z.push(r * (two_pi * u2).cos());
        if z.len() < n {
            z.push(r * (two_pi * u2).sin());
        }
    }
    let y = (0..n)
        .map(|i| mean_truth(x[i]) + sigma_truth(x[i]) * z[i])
        .collect();
    (x, y)
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut sab = 0.0;
    let mut saa = 0.0;
    let mut sbb = 0.0;
    for (&x, &y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    sab / (saa.sqrt() * sbb.sqrt())
}

fn rmse(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    (a.iter()
        .zip(b)
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f64>()
        / n)
        .sqrt()
}

fn fmt_vec(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x:.4}")).collect();
    format!("[{}]", parts.join(", "))
}

fn run_case(label: &str, mean_formula: &str, noise_formula: &str, n: usize) -> f64 {
    let (x, y) = locscale_data(n);
    let headers = vec!["x".to_string(), "y".to_string()];
    let rows: Vec<csv::StringRecord> = x
        .iter()
        .zip(&y)
        .map(|(&x, &y)| csv::StringRecord::from(vec![format!("{x:.17e}"), format!("{y:.17e}")]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        noise_formula: Some(noise_formula.to_string()),
        ..FitConfig::default()
    };
    let result = match fit_from_formula(mean_formula, &ds, &cfg) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[{label}] FIT ERROR: {e}");
            return f64::NAN;
        }
    };
    let FitResult::GaussianLocationScale(res) = result else {
        eprintln!("[{label}] not a location-scale fit");
        return f64::NAN;
    };
    let response_scale = res.response_scale;
    let fit = res.fit;

    let beta_loc = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("loc")
        .beta
        .clone();
    let beta_scale = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale")
        .beta
        .clone();

    // Use the FROZEN training-row designs (no rebuild needed).
    let gam_mu = fit.mean_design.design.apply(&beta_loc).to_vec();
    let gam_eta_sigma = fit.noise_design.design.apply(&beta_scale).to_vec();
    let gam_log_sigma: Vec<f64> = gam_eta_sigma
        .iter()
        .map(|&e| (response_scale * LOGB_SIGMA_FLOOR + e.exp()).ln())
        .collect();

    let true_mu: Vec<f64> = x.iter().map(|&t| mean_truth(t)).collect();
    let true_log_sigma: Vec<f64> = x.iter().map(|&t| sigma_truth(t).abs().ln()).collect();

    let rmse_mu = rmse(&gam_mu, &true_mu);
    let corr = pearson(&gam_log_sigma, &true_log_sigma);
    let rmse_ls = rmse(&gam_log_sigma, &true_log_sigma);

    let lambdas = fit.fit.lambdas.to_vec();
    let log_lambdas = fit.fit.log_lambdas.to_vec();
    let reml = fit.fit.reml_score;
    let (edf_by_block, edf_total) = if let Some(inf) = fit.fit.inference.as_ref() {
        (inf.edf_by_block.clone(), inf.edf_total)
    } else {
        (vec![], f64::NAN)
    };

    eprintln!(
        "[{label}] pearson={corr:.5} rmse_ls={rmse_ls:.5} rmse_mu={rmse_mu:.5} reml={reml:.4} \
         | lambdas={} log_lambdas={} | edf_by_block={} edf_total={edf_total:.3} \
         | response_scale={response_scale:.4} outer_conv=certified iters={}",
        fmt_vec(&lambdas),
        fmt_vec(&log_lambdas),
        fmt_vec(&edf_by_block),
        fit.fit.outer_iterations
    );
    corr
}

#[test]
fn probe_1561_locscale_penalty_configs() {
    let n = 200;
    eprintln!("=== #1561 gam-models probe: Gaussian location-scale log-σ recovery (n={n}) ===");
    let default_corr = run_case("default", "y ~ s(x, bs='tp')", "1 + s(x, bs='tp')", n);
    let scale_with_double_corr = run_case(
        "scale_with_double",
        "y ~ s(x, bs='tp')",
        "1 + s(x, bs='tp', double_penalty=true)",
        n,
    );
    let both_nodbl_corr = run_case(
        "both_nodbl",
        "y ~ s(x, bs='tp', double_penalty=false)",
        "1 + s(x, bs='tp', double_penalty=false)",
        n,
    );
    for (label, c) in [
        ("default", default_corr),
        ("scale_with_double", scale_with_double_corr),
        ("both_nodbl", both_nodbl_corr),
    ] {
        assert!(
            c.is_finite(),
            "#1561 probe: {label} produced a non-finite log-σ pearson ({c}) — fit broke"
        );
        assert!(
            c > 0.0,
            "#1561 probe: {label} log-σ pearson {c:.4} <= 0 — fitted scale surface degenerate"
        );
    }
}
