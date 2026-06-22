//! Predict-at-training reproduces the in-fit fitted values across the full
//! basis menu and across families.
//!
//! Contract: calling the predict-time design rebuild
//! (`build_term_collection_design`) at the EXACT training input rows must
//! reproduce the in-sample linear predictor `X β` (and hence the fitted mean
//! after the inverse link) to machine precision, for every basis. Any
//! discrepancy is a design-replay bug — the predict-time basis differs from
//! the fit-time basis (different knots, centering, reparam, or column order).
//!
//! This complements the single-`s(x)` regression
//! (`predict_at_training_points_matches_fitted_values`) by sweeping
//! ps / cr / tp / cc / ds / te / ti / fs / re and a Poisson (log-link) arm,
//! which is where replay drift historically hides.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal};

/// One numeric covariate `x` plus a Gaussian response on a smooth truth.
fn numeric_data(n: usize, seed: u64) -> (Vec<f64>, gam::data::EncodedDataset) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let x: Vec<f64> = (0..n)
        .map(|i| 0.01 + 0.98 * i as f64 / (n as f64 - 1.0))
        .collect();
    let y: Vec<f64> = x
        .iter()
        .map(|&t| 0.6 * (2.0 * std::f64::consts::PI * t).sin() + noise.sample(&mut rng))
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    (x, data)
}

/// Two numeric covariates for tensor (te/ti) bases.
fn numeric_data_2d(n: usize, seed: u64) -> (Vec<[f64; 2]>, gam::data::EncodedDataset) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let mut pts: Vec<[f64; 2]> = Vec::with_capacity(n);
    for i in 0..n {
        let a = 0.01 + 0.98 * (i as f64) / (n as f64 - 1.0);
        let b = 0.01 + 0.98 * (((i * 7) % n) as f64) / (n as f64 - 1.0);
        pts.push([a, b]);
    }
    let headers = ["x", "z", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = pts
        .iter()
        .map(|p| {
            let y = (2.0 * std::f64::consts::PI * p[0]).sin() * (p[1] - 0.5)
                + noise.sample(&mut rng);
            StringRecord::from(vec![p[0].to_string(), p[1].to_string(), y.to_string()])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    (pts, data)
}

/// Numeric covariate + a balanced grouping factor for factor-smooth (fs) and
/// random-effect (re) terms.
fn grouped_data(n: usize, levels: usize, seed: u64) -> (Vec<f64>, Vec<String>, gam::data::EncodedDataset) {
    let mut rng = StdRng::seed_from_u64(seed);
    let noise = Normal::new(0.0, 0.05).expect("normal");
    let x: Vec<f64> = (0..n)
        .map(|i| 0.01 + 0.98 * i as f64 / (n as f64 - 1.0))
        .collect();
    let g: Vec<String> = (0..n).map(|i| format!("g{}", i % levels)).collect();
    let y: Vec<f64> = x
        .iter()
        .zip(g.iter())
        .map(|(&t, gi)| {
            let bump = (gi.trim_start_matches('g').parse::<f64>().unwrap_or(0.0)) * 0.3;
            0.6 * (2.0 * std::f64::consts::PI * t).sin() + bump + noise.sample(&mut rng)
        })
        .collect();
    let headers = ["x", "g", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![x[i].to_string(), g[i].clone(), y[i].to_string()])
        })
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    (x, g, data)
}

/// Fit `formula`, rebuild the predict-time design at the SAME numeric rows,
/// and return `(max |fit - pred|, n)` of the linear predictor `X β`.
fn replay_max_diff(
    formula: &str,
    cfg: &FitConfig,
    data: &gam::data::EncodedDataset,
    new_data: Array2<f64>,
) -> (f64, usize, usize) {
    let result = fit_from_formula(formula, data, cfg).unwrap_or_else(|e| {
        panic!("fit failed for `{formula}`: {e:?}");
    });
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit for `{formula}`");
    };
    let fitted = fit.design.design.apply(&fit.fit.beta).to_vec();
    let pred_design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
        .unwrap_or_else(|e| panic!("predict rebuild failed for `{formula}`: {e:?}"));
    let pred = pred_design.design.apply(&fit.fit.beta).to_vec();
    assert_eq!(
        fitted.len(),
        pred.len(),
        "fit/pred length mismatch for `{formula}`"
    );
    let n = fitted.len();
    let mut max_diff = 0.0_f64;
    let mut bad = 0usize;
    for i in 0..n {
        let d = (fitted[i] - pred[i]).abs();
        if d > max_diff {
            max_diff = d;
        }
        if d > 1e-9 {
            bad += 1;
        }
    }
    (max_diff, bad, n)
}

const TOL: f64 = 1e-9;

#[test]
fn predict_at_training_matches_fitted_across_1d_bases() {
    init_parallelism();
    let n = 200usize;
    let (x, data) = numeric_data(n, 229);
    let gaussian = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let mut new_data = Array2::<f64>::zeros((n, 2));
    for (i, &t) in x.iter().enumerate() {
        new_data[[i, 0]] = t;
    }

    let bases: &[(&str, &str)] = &[
        ("tp", "y ~ s(x, bs=\"tp\", k=10)"),
        ("ps", "y ~ s(x, bs=\"ps\", k=10)"),
        ("cr", "y ~ s(x, bs=\"cr\", k=10)"),
        ("cc", "y ~ s(x, bs=\"cc\", k=10)"),
        ("ds", "y ~ s(x, bs=\"ds\", k=10)"),
    ];

    let mut failures = Vec::<String>::new();
    for (label, formula) in bases {
        let (max_diff, bad, _n) = replay_max_diff(formula, &gaussian, &data, new_data.clone());
        eprintln!("[replay-1d] {label:4} max_diff={max_diff:.3e} bad={bad}");
        if max_diff > TOL {
            failures.push(format!(
                "{label}: max |fit-pred| = {max_diff:.3e} ({bad} pts > {TOL:.0e})"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "predict-at-training design replay drifted for 1-D bases:\n  - {}",
        failures.join("\n  - ")
    );
}

#[test]
fn predict_at_training_matches_fitted_poisson_log_link() {
    init_parallelism();
    let n = 200usize;
    // Poisson counts on a log-mean smooth.
    let mut rng = StdRng::seed_from_u64(733);
    let x: Vec<f64> = (0..n)
        .map(|i| 0.01 + 0.98 * i as f64 / (n as f64 - 1.0))
        .collect();
    use rand_distr::Poisson;
    let y: Vec<f64> = x
        .iter()
        .map(|&t| {
            let lambda = (1.5 + (2.0 * std::f64::consts::PI * t).sin()).exp();
            let p = Poisson::new(lambda).expect("poisson");
            p.sample(&mut rng)
        })
        .collect();
    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    let data = encode_recordswith_inferred_schema(headers, rows).expect("encode");
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let mut new_data = Array2::<f64>::zeros((n, 2));
    for (i, &t) in x.iter().enumerate() {
        new_data[[i, 0]] = t;
    }
    let (max_diff, bad, _n) =
        replay_max_diff("y ~ s(x, k=10)", &cfg, &data, new_data);
    eprintln!("[replay-poisson] max_diff={max_diff:.3e} bad={bad}");
    assert!(
        max_diff <= TOL,
        "Poisson log-link predict-at-training drifted: max |fit-pred| = {max_diff:.3e} ({bad} pts)"
    );
}

#[test]
fn predict_at_training_matches_fitted_tensor_bases() {
    init_parallelism();
    let n = 256usize;
    let (pts, data) = numeric_data_2d(n, 451);
    let gaussian = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let mut new_data = Array2::<f64>::zeros((n, 3));
    for (i, p) in pts.iter().enumerate() {
        new_data[[i, 0]] = p[0];
        new_data[[i, 1]] = p[1];
    }
    let bases: &[(&str, &str)] = &[
        ("te", "y ~ te(x, z, k=5)"),
        ("ti", "y ~ ti(x, z, k=5)"),
    ];
    let mut failures = Vec::<String>::new();
    for (label, formula) in bases {
        let (max_diff, bad, _n) = replay_max_diff(formula, &gaussian, &data, new_data.clone());
        eprintln!("[replay-tensor] {label:4} max_diff={max_diff:.3e} bad={bad}");
        if max_diff > TOL {
            failures.push(format!(
                "{label}: max |fit-pred| = {max_diff:.3e} ({bad} pts > {TOL:.0e})"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "predict-at-training design replay drifted for tensor bases:\n  - {}",
        failures.join("\n  - ")
    );
}

#[test]
fn predict_at_training_matches_fitted_factor_and_random_effect() {
    init_parallelism();
    let n = 240usize;
    let levels = 4usize;
    let (x, g, data) = grouped_data(n, levels, 919);
    let gaussian = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    // new_data column order follows the encoded schema [x, g, y]; the factor
    // column carries the integer code, but the predict rebuild resolves it
    // from the resolved spec, so we replay the same numeric design rows by
    // re-encoding the identical training records.
    let bases: &[(&str, &str)] = &[
        ("fs", "y ~ s(x, g, bs=\"fs\", k=6)"),
        ("re", "y ~ s(g, bs=\"re\")"),
    ];
    let mut failures = Vec::<String>::new();
    for (label, formula) in bases {
        // Rebuild predict design from the same encoded dataset's design rows.
        let result = fit_from_formula(formula, &data, &gaussian)
            .unwrap_or_else(|e| panic!("fit failed for `{formula}`: {e:?}"));
        let FitResult::Standard(fit) = result else {
            panic!("expected standard fit for `{formula}`");
        };
        let fitted = fit.design.design.apply(&fit.fit.beta).to_vec();
        // Re-encode identical rows to obtain a fresh numeric design matrix that
        // the predict rebuild can consume with the SAME factor coding.
        let headers = ["x", "g", "y"].into_iter().map(String::from).collect();
        let rows: Vec<StringRecord> = (0..n)
            .map(|i| StringRecord::from(vec![x[i].to_string(), g[i].clone(), "0".to_string()]))
            .collect();
        let replay = encode_recordswith_inferred_schema(headers, rows).expect("re-encode");
        let new_data = replay.values.clone();
        let pred_design = build_term_collection_design(new_data.view(), &fit.resolvedspec)
            .unwrap_or_else(|e| panic!("predict rebuild failed for `{formula}`: {e:?}"));
        let pred = pred_design.design.apply(&fit.fit.beta).to_vec();
        let mut max_diff = 0.0_f64;
        let mut bad = 0usize;
        for i in 0..fitted.len() {
            let d = (fitted[i] - pred[i]).abs();
            if d > max_diff {
                max_diff = d;
            }
            if d > 1e-9 {
                bad += 1;
            }
        }
        eprintln!("[replay-grouped] {label:4} max_diff={max_diff:.3e} bad={bad}");
        if max_diff > TOL {
            failures.push(format!(
                "{label}: max |fit-pred| = {max_diff:.3e} ({bad} pts > {TOL:.0e})"
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "predict-at-training design replay drifted for factor/random-effect terms:\n  - {}",
        failures.join("\n  - ")
    );
}
