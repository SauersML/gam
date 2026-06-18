//! Duchon sin8 quality regression. Default `duchon(x)` and an explicitly
//! well-resolved `centers=50` basis must recover sin(2π·8·x) at σ=0.10 within
//! the same absolute max-error budget used by the original failing ticket.
//!
//! `centers=20` is different: 20 knots over eight cycles gives only about 2.5
//! knots per period, a near-Nyquist design. In that regime an arbitrary absolute
//! max-error cutoff confounds smoother quality with resolution. The objective
//! assertion for this deliberately under-resolved variant is therefore
//! match-or-beat-mgcv on truth recovery at the same `k` (`bs="ds"`, REML), with
//! a small slack factor for robustness near Nyquist. That keeps the quality bar
//! tied to a mature Duchon implementation without pretending 20 centers can
//! always resolve eight noisy cycles.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, QualityDiagnostics, rmse, run_r};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn make_sin_dataset(freq: f64, sigma: f64, n: usize, seed: u64) -> gam::data::EncodedDataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let ux = Uniform::new(0.0, 1.0).expect("uniform");
    let noise = Normal::new(0.0, sigma).expect("normal");
    let mut x: Vec<f64> = (0..n).map(|_| ux.sample(&mut rng)).collect();
    x.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let two_pi_f = 2.0 * std::f64::consts::PI * freq;
    let y_noisy: Vec<f64> = x
        .iter()
        .map(|&t| (two_pi_f * t).sin() + noise.sample(&mut rng))
        .collect();

    let headers = ["x", "y"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = x
        .iter()
        .zip(y_noisy.iter())
        .map(|(a, b)| StringRecord::from(vec![a.to_string(), b.to_string()]))
        .collect();
    encode_recordswith_inferred_schema(headers, rows).expect("encode sin dataset")
}

fn fit_and_predict_diagnostics(
    label: &str,
    formula: &str,
    data: &gam::data::EncodedDataset,
    x_test: &[f64],
    truth: &[f64],
) -> Vec<f64> {
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(formula, data, &cfg).expect("duchon fit succeeded");
    let FitResult::Standard(fit) = result else {
        panic!("expected standard fit")
    };
    let n = x_test.len();
    let mut m = Array2::<f64>::zeros((n, data.headers.len()));
    let x_col = data.column_map()["x"];
    for (i, &t) in x_test.iter().enumerate() {
        m[[i, x_col]] = t;
    }
    let test_design = build_term_collection_design(m.view(), &fit.resolvedspec)
        .expect("rebuild design from frozen spec");
    let pred = test_design.design.apply(&fit.fit.beta).to_vec();
    QualityDiagnostics::from_standard_fit(label, &fit)
        .with_truth_rmse(&pred, truth)
        .emit();
    pred
}

fn mgcv_duchon_predict(data: &gam::data::EncodedDataset, x_test: &[f64], k: usize) -> Vec<f64> {
    let x_col = data.column_map()["x"];
    let y_col = data.column_map()["y"];
    let mut x_all = data.values.column(x_col).to_vec();
    x_all.extend_from_slice(x_test);
    let mut y_all = data.values.column(y_col).to_vec();
    y_all.extend(std::iter::repeat_n(0.0, x_test.len()));
    let mut is_train = vec![1.0; data.values.nrows()];
    is_train.extend(std::iter::repeat_n(0.0, x_test.len()));

    let r = run_r(
        &[
            Column::new("x", &x_all),
            Column::new("y", &y_all),
            Column::new("is_train", &is_train),
        ],
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            train <- df[df$is_train > 0.5, ]
            grid  <- df[df$is_train < 0.5, ]
            m <- gam(y ~ s(x, bs = "ds", k = {k}, m = c(2, 0)),
                     data = train, method = "REML")
            emit("fitted", as.numeric(predict(m, newdata = grid)))
            "#
        ),
    );
    r.vector("fitted").to_vec()
}

fn max_abs_err(yhat: &[f64], y: &[f64]) -> f64 {
    yhat.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

#[test]
fn duchon_sin8_max_error_within_budget() {
    init_parallelism();
    let data = make_sin_dataset(8.0, 0.10, 240, 11);
    let x_test: Vec<f64> = (0..400).map(|i| 0.001 + 0.998 * i as f64 / 399.0).collect();
    let y_truth_test: Vec<f64> = x_test
        .iter()
        .map(|t| (2.0 * std::f64::consts::PI * 8.0 * t).sin())
        .collect();

    // Truth peak-to-peak is 2.0; 30% of that is 0.60. These two cases have
    // enough basis resolution to make the absolute recovery budget meaningful.
    let budget = 0.60_f64;
    let absolute_cases: &[(&str, &str)] = &[
        ("duchon-default", "duchon(x)"),
        ("duchon-centers50", "duchon(x, centers=50)"),
    ];
    let mut violations = Vec::<String>::new();
    for (label, body) in absolute_cases {
        let yhat = fit_and_predict_diagnostics(
            label,
            &format!("y ~ {body}"),
            &data,
            &x_test,
            &y_truth_test,
        );
        let m = max_abs_err(&yhat, &y_truth_test);
        eprintln!("[duchon-sin8] {label:18} max_err={m:.4}");
        if m > budget {
            violations.push(format!(
                "{label}: max_err {m:.4} > {budget:.2} (truth peak=2.0, 30% budget)"
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "resolved Duchon variants oversmooth sin8 at σ=0.10:\n  - {}",
        violations.join("\n  - "),
    );

    let gam_centers20 = fit_and_predict_diagnostics(
        "duchon-centers20",
        "y ~ duchon(x, centers=20)",
        &data,
        &x_test,
        &y_truth_test,
    );
    let mgcv_centers20 = mgcv_duchon_predict(&data, &x_test, 20);
    let gam_max = max_abs_err(&gam_centers20, &y_truth_test);
    let mgcv_max = max_abs_err(&mgcv_centers20, &y_truth_test);
    let gam_truth_rmse = rmse(&gam_centers20, &y_truth_test);
    let mgcv_truth_rmse = rmse(&mgcv_centers20, &y_truth_test);
    eprintln!(
        "[duchon-sin8] duchon-centers20 max_err={gam_max:.4} \
         truth_rmse={gam_truth_rmse:.4}; mgcv-ds-k20 max_err={mgcv_max:.4} \
         truth_rmse={mgcv_truth_rmse:.4}"
    );
    // Match-or-beat with a small slack factor: 20 knots over eight cycles is a
    // near-Nyquist design where an exact RMSE tie is fragile, so allow gam to be
    // within 10% of the mature mgcv `bs="ds"` k=20 truth-recovery RMSE.
    let slack = 1.10_f64;
    assert!(
        gam_truth_rmse <= slack * mgcv_truth_rmse,
        "near-Nyquist centers=20 Duchon must match-or-beat (within {slack:.2}×) mgcv \
         bs=\"ds\" k=20 on truth RMSE: gam={gam_truth_rmse:.4}, mgcv={mgcv_truth_rmse:.4} \
         (max_err gam={gam_max:.4}, mgcv={mgcv_max:.4})"
    );
}
