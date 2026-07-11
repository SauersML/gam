//! Fast gam-models-level probe for #1561 (Gaussian location-scale over-smooths
//! the log-σ surface). Calls `fit_from_formula` directly so it compiles ONLY
//! gam-models (immune to the gam-umbrella recompile churn). Measures
//! pearson(fitted logσ, true logσ), rmse, the selected lambdas, the REML score,
//! and the per-block EDF under three genuinely distinct penalty configs:
//!   (a) shipped null-recovery default (both double penalties on),
//!   (b) explicitly disable the scale-block null-space double penalty,
//!   (c) disable the double penalty on both blocks.
//! It is a measurement probe (not a permanent quality gate): each config must
//! still produce a finite, positively-correlated log-σ surface.
#![cfg(test)]

use crate::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use faer::Side;
use gam_data::{encode_recordswith_inferred_schema, load_csvwith_inferred_schema};
use gam_linalg::faer_ndarray::FaerEigh;
use gam_linalg::matrix::LinearOperator;
use gam_solve::estimate::BlockRole;
use gam_terms::smooth::build_term_collection_design;
use ndarray::Array2;
use std::path::Path;

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

fn penalized_hessian_spectrum(
    fit: &gam_solve::model_types::UnifiedFitResult,
) -> Option<(f64, f64, usize)> {
    let geometry = fit.geometry.as_ref()?;
    let (eigenvalues, _) = geometry
        .penalized_hessian
        .as_array()
        .eigh(Side::Lower)
        .ok()?;
    let min = eigenvalues.iter().copied().fold(f64::INFINITY, f64::min);
    let max = eigenvalues
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let negative = eigenvalues.iter().filter(|&&value| value < 0.0).count();
    Some((min, max, negative))
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
    let spectrum = penalized_hessian_spectrum(&fit.fit);

    eprintln!(
        "[{label}] pearson={corr:.5} rmse_ls={rmse_ls:.5} rmse_mu={rmse_mu:.5} reml={reml:.4} \
         | lambdas={} log_lambdas={} | edf_by_block={} edf_total={edf_total:.3} \
         | response_scale={response_scale:.4} outer_conv=certified iters={} spectrum={spectrum:?}",
        fmt_vec(&lambdas),
        fmt_vec(&log_lambdas),
        fmt_vec(&edf_by_block),
        fit.fit.outer_iterations
    );
    corr
}

fn gaussian_nll(y: &[f64], mu: &[f64], sigma: &[f64]) -> f64 {
    let half_log_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    y.iter()
        .zip(mu)
        .zip(sigma)
        .map(|((&observed, &mean), &sd)| {
            let z = (observed - mean) / sd;
            half_log_2pi + sd.ln() + 0.5 * z * z
        })
        .sum::<f64>()
        / y.len() as f64
}

#[test]
fn probe_1561_locscale_penalty_configs() {
    let n = 200;
    eprintln!("=== #1561 gam-models probe: Gaussian location-scale log-σ recovery (n={n}) ===");
    let default_corr = run_case("default", "y ~ s(x, bs='tp')", "1 + s(x, bs='tp')", n);
    let scale_nodbl_corr = run_case(
        "scale_nodbl",
        "y ~ s(x, bs='tp')",
        "1 + s(x, bs='tp', double_penalty=false)",
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
        ("scale_nodbl", scale_nodbl_corr),
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

#[test]
fn probe_1561_gagurine_scale_geometry() {
    let dataset_path = Path::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../bench/datasets/gagurine.csv"
    ));
    let dataset = load_csvwith_inferred_schema(dataset_path).expect("load gagurine");
    let columns = dataset.column_map();
    let age_col = columns["Age"];
    let gag_col = columns["GAG"];
    let age = dataset.values.column(age_col).to_vec();
    let gag = dataset.values.column(gag_col).to_vec();
    let train_rows: Vec<usize> = (0..age.len()).filter(|index| index % 4 != 0).collect();
    let test_rows: Vec<usize> = (0..age.len()).filter(|index| index % 4 == 0).collect();
    let mut train = dataset.clone();
    train.values = Array2::from_shape_fn((train_rows.len(), dataset.values.ncols()), |(i, j)| {
        dataset.values[[train_rows[i], j]]
    });
    let train_gag: Vec<f64> = train_rows.iter().map(|&index| gag[index]).collect();
    let test_age: Vec<f64> = test_rows.iter().map(|&index| age[index]).collect();
    let test_gag: Vec<f64> = test_rows.iter().map(|&index| gag[index]).collect();

    let result = fit_from_formula(
        "GAG ~ s(Age, bs='tp')",
        &train,
        &FitConfig {
            family: Some("gaussian".to_string()),
            noise_formula: Some("1 + s(Age, bs='tp')".to_string()),
            ..FitConfig::default()
        },
    )
    .expect("gagurine location-scale probe fit");
    let FitResult::GaussianLocationScale(result) = result else {
        panic!("expected Gaussian location-scale fit");
    };
    let response_scale = result.response_scale;
    let fit = result.fit;
    let location = fit
        .fit
        .block_by_role(BlockRole::Location)
        .expect("location block");
    let scale = fit
        .fit
        .block_by_role(BlockRole::Scale)
        .expect("scale block");
    let train_mu = fit.mean_design.design.apply(&location.beta).to_vec();
    let train_eta_scale = fit.noise_design.design.apply(&scale.beta).to_vec();

    let mut test_grid = Array2::<f64>::zeros((test_age.len(), dataset.values.ncols()));
    for (row, &value) in test_age.iter().enumerate() {
        test_grid[[row, age_col]] = value;
    }
    let test_mean_design = build_term_collection_design(test_grid.view(), &fit.meanspec_resolved)
        .expect("rebuild gagurine mean design");
    let test_scale_design = build_term_collection_design(test_grid.view(), &fit.noisespec_resolved)
        .expect("rebuild gagurine scale design");
    let test_mu = test_mean_design.design.apply(&location.beta).to_vec();
    let test_eta_scale = test_scale_design.design.apply(&scale.beta).to_vec();
    let raw_floor = response_scale * LOGB_SIGMA_FLOOR;
    let train_sigma: Vec<f64> = train_eta_scale
        .iter()
        .map(|&eta| raw_floor + eta.exp())
        .collect();
    let test_sigma: Vec<f64> = test_eta_scale
        .iter()
        .map(|&eta| raw_floor + eta.exp())
        .collect();
    let constant_sigma = (train_gag
        .iter()
        .zip(&train_mu)
        .map(|(&observed, &mean)| (observed - mean).powi(2))
        .sum::<f64>()
        / train_gag.len() as f64)
        .sqrt();
    let nll = gaussian_nll(&test_gag, &test_mu, &test_sigma);
    let constant_nll = gaussian_nll(&test_gag, &test_mu, &vec![constant_sigma; test_gag.len()]);
    let standardized_residual_energy = |observed: &[f64], mean: &[f64], sigma: &[f64]| {
        observed
            .iter()
            .zip(mean)
            .zip(sigma)
            .map(|((&value, &location), &scale)| ((value - location) / scale).powi(2))
            .sum::<f64>()
            / observed.len() as f64
    };
    let train_calibration = standardized_residual_energy(&train_gag, &train_mu, &train_sigma);
    let test_calibration = standardized_residual_energy(&test_gag, &test_mu, &test_sigma);
    let mut heldout_nll_deltas: Vec<(f64, usize)> = test_gag
        .iter()
        .enumerate()
        .map(|(index, &observed)| {
            let residual = observed - test_mu[index];
            let varying = test_sigma[index].ln() + 0.5 * (residual / test_sigma[index]).powi(2);
            let constant = constant_sigma.ln() + 0.5 * (residual / constant_sigma).powi(2);
            (varying - constant, index)
        })
        .collect();
    heldout_nll_deltas.sort_by(|left, right| {
        right
            .0
            .partial_cmp(&left.0)
            .expect("finite held-out NLL deltas")
    });
    let range = |values: &[f64]| {
        (
            values.iter().copied().fold(f64::INFINITY, f64::min),
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        )
    };
    let edf = fit
        .fit
        .inference
        .as_ref()
        .map(|inference| (inference.edf_by_block.clone(), inference.edf_total));
    let spectrum = penalized_hessian_spectrum(&fit.fit);
    eprintln!(
        "[#1561 gagurine] response_scale={response_scale:.6} raw_floor={raw_floor:.6} nll={nll:.6} constant_nll={constant_nll:.6} constant_sigma={constant_sigma:.6} train_calibration={train_calibration:.6} test_calibration={test_calibration:.6} train_sigma_range={:?} test_sigma_range={:?} location_edf={:.6} location_lambdas={} scale_edf={:.6} scale_lambdas={} log_lambdas={} edf={edf:?} spectrum={spectrum:?}",
        range(&train_sigma),
        range(&test_sigma),
        location.edf,
        fmt_vec(
            location
                .lambdas
                .as_slice()
                .expect("contiguous location lambdas")
        ),
        scale.edf,
        fmt_vec(scale.lambdas.as_slice().expect("contiguous scale lambdas")),
        fmt_vec(fit.fit.log_lambdas.as_slice().expect("contiguous lambdas")),
    );
    for (delta, index) in heldout_nll_deltas {
        eprintln!(
            "[#1561 gagurine row] delta_nll={delta:.6} age={:.6} observed={:.6} mean={:.6} sigma={:.6} residual={:.6}",
            test_age[index],
            test_gag[index],
            test_mu[index],
            test_sigma[index],
            test_gag[index] - test_mu[index],
        );
    }

    assert!(nll.is_finite());
    assert!(
        test_sigma
            .iter()
            .all(|sigma| sigma.is_finite() && *sigma > 0.0)
    );
    assert!(train_calibration.is_finite() && test_calibration.is_finite());

    // Gaussian location-scale LAML must have exactly the formula-native penalty
    // coordinates. #1561 exposed an extra implicit scale-level projector here;
    // when the native smooth already has its selection penalty, that extra rho
    // shrinks the global log-sigma intercept and changes the statistical model.
    assert_eq!(
        location.lambdas.len(),
        fit.mean_design.penalties.len(),
        "#1561 location block has a non-formula penalty coordinate"
    );
    assert_eq!(
        scale.lambdas.len(),
        fit.noise_design.penalties.len(),
        "#1561 scale block has a non-formula penalty coordinate"
    );
    assert_eq!(
        fit.fit.lambdas.len(),
        fit.mean_design.penalties.len() + fit.noise_design.penalties.len(),
        "#1561 joint rho vector is not the concatenation of formula penalties"
    );

    // The observed joint curvature used by LAML must remain a genuine SPD
    // geometry at the certified solution; this catches a dense/operator or
    // observed-derivative desynchronization without any fitted-score threshold.
    let (minimum_eigenvalue, maximum_eigenvalue, negative_eigenvalues) =
        spectrum.expect("#1561 fitted geometry must retain its penalized Hessian");
    assert!(
        minimum_eigenvalue > 0.0 && maximum_eigenvalue.is_finite(),
        "#1561 penalized Hessian is not SPD: min={minimum_eigenvalue}, max={maximum_eigenvalue}"
    );
    assert_eq!(
        negative_eigenvalues, 0,
        "#1561 penalized Hessian has negative eigenvalues"
    );
}
