//! Duchon sin8 quality regression, framed by BASIS RESOLUTION against the mature
//! Duchon comparator (mgcv `bs="ds"`, `m=c(2,0)`, REML) on byte-identical data.
//!
//! sin(2π·8·x) at σ=0.10 is eight cycles of signal. Whether a Duchon smoother can
//! recover it is set by how many knots span those cycles, and the measured mgcv
//! reference (`experiments/duchon_mgcv_reference/mgcv_reference_msi_run.json`)
//! makes the three regimes explicit:
//!
//! * `centers=50` (~6 knots/period): the mature smoother ESCAPES the shelf —
//!   sum_edf≈47, truth RMSE≈0.049, max error≈0.19. This is a genuine-recovery
//!   regime, so we hold gam to an absolute max-error budget (a real statement
//!   about recovering a peak-2.0 sine) AND to match-or-beat mgcv on truth RMSE.
//! * DEFAULT rank: the mature smoother SHELVES to a near-flat fit — sum_edf≈2,
//!   truth RMSE≈0.70 (the trivial-predictor level). An absolute recovery budget
//!   is not a meaningful quality bar there because mgcv itself cannot meet one;
//!   the objective assertion is that gam shelves NO WORSE than mgcv on truth
//!   recovery. Escape is the job of an explicitly-resolved basis, not the default.
//! * `centers=20` (~2.5 knots/period, near-Nyquist): also resolution-limited, so
//!   again match-or-beat-mgcv on truth recovery at the same `k`, with a small
//!   slack factor for robustness near Nyquist.
//!
//! Every arm is therefore tied to a mature Duchon implementation at its own
//! resolution, and the absolute budget is asserted only where the basis can
//! actually support recovery.

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
    let diagnostics =
        QualityDiagnostics::from_standard_fit(label, &fit).with_truth_rmse(&pred, truth);
    eprintln!("{}", diagnostics.report());
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

fn mgcv_duchon_diag(data: &gam::data::EncodedDataset, k: usize) -> (f64, f64) {
    // Returns (sum_edf, sp) for mgcv bs="ds", m=c(2,0), REML on the training data.
    let x_col = data.column_map()["x"];
    let y_col = data.column_map()["y"];
    let x_all = data.values.column(x_col).to_vec();
    let y_all = data.values.column(y_col).to_vec();
    let r = run_r(
        &[Column::new("x", &x_all), Column::new("y", &y_all)],
        &format!(
            r#"
            suppressPackageStartupMessages(library(mgcv))
            m <- gam(y ~ s(x, bs = "ds", k = {k}, m = c(2, 0)), data = df, method = "REML")
            emit("edf", as.numeric(sum(m$edf)))
            emit("sp",  as.numeric(m$sp))
            "#
        ),
    );
    (r.vector("edf")[0], r.vector("sp")[0])
}

#[test]
fn zz_measure_lambda_gap() {
    use gam::smooth::build_term_collection_design;
    use gam::solver::gaussian_reml::gaussian_reml_closed_form;
    use ndarray::Array1;

    init_parallelism();
    let data = make_sin_dataset(8.0, 0.10, 240, 11);
    let x_test: Vec<f64> = (0..400).map(|i| 0.001 + 0.998 * i as f64 / 399.0).collect();
    let y_truth: Vec<f64> = x_test
        .iter()
        .map(|t| (2.0 * std::f64::consts::PI * 8.0 * t).sin())
        .collect();

    for (label, body, k) in [
        ("duchon-default", "duchon(x)", 40usize),
        ("duchon-centers50", "duchon(x, centers=50)", 50),
    ] {
        // ---- production gam fit ----
        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let result = fit_from_formula(&format!("y ~ {body}"), &data, &cfg).expect("fit");
        let FitResult::Standard(fit) = result else {
            panic!("standard")
        };
        let prod_lambdas = fit.fit.lambdas.to_vec();
        let prod_loglam = fit.fit.log_lambdas.to_vec();
        let prod_edf = fit.fit.inference.as_ref().map(|i| i.edf_total);

        // predictions on truth grid
        let n_t = x_test.len();
        let mut mt = Array2::<f64>::zeros((n_t, data.headers.len()));
        let x_col = data.column_map()["x"];
        for (i, &t) in x_test.iter().enumerate() {
            mt[[i, x_col]] = t;
        }
        let test_design = build_term_collection_design(mt.view(), &fit.resolvedspec)
            .expect("test design");
        let prod_pred = test_design.design.apply(&fit.fit.beta).to_vec();
        let prod_max = max_abs_err(&prod_pred, &y_truth);
        let prod_amp = prod_pred.iter().cloned().fold(f64::MIN, f64::max)
            - prod_pred.iter().cloned().fold(f64::MAX, f64::min);

        // ---- reconstruct train design X + penalty S from frozen spec ----
        let n_tr = data.values.nrows();
        let mut mtr = Array2::<f64>::zeros((n_tr, data.headers.len()));
        for i in 0..n_tr {
            mtr[[i, x_col]] = data.values[[i, x_col]];
        }
        let train_design = build_term_collection_design(mtr.view(), &fit.resolvedspec)
            .expect("train design");
        let x_dense = train_design.design.to_dense();
        let p = x_dense.ncols();
        let mut s = Array2::<f64>::zeros((p, p));
        for bp in &train_design.penalties {
            let r = bp.col_range.clone();
            for (li, gi) in r.clone().enumerate() {
                for (lj, gj) in r.clone().enumerate() {
                    s[[gi, gj]] += bp.local[[li, lj]];
                }
            }
        }
        let nulldim: usize = train_design.nullspace_dims.iter().sum();
        let y_tr = data.values.column(data.column_map()["y"]).to_owned();

        // gam's OWN global-grid REML optimum on this basis (single summed penalty)
        let cf = gaussian_reml_closed_form(
            x_dense.view(),
            y_tr.view(),
            s.view(),
            None,
            None,
        )
        .expect("closed form");
        // predict closed-form coefficients on truth grid
        let xt = test_design.design.to_dense();
        let cf_beta = Array1::from(cf.coefficients.to_vec());
        let cf_pred = xt.dot(&cf_beta).to_vec();
        let cf_max = max_abs_err(&cf_pred, &y_truth);
        let cf_amp = cf_pred.iter().cloned().fold(f64::MIN, f64::max)
            - cf_pred.iter().cloned().fold(f64::MAX, f64::min);

        // ---- mgcv ----
        let (mgcv_edf, mgcv_sp) = mgcv_duchon_diag(&data, k);
        let mgcv_pred = mgcv_duchon_predict(&data, &x_test, k);
        let mgcv_max = max_abs_err(&mgcv_pred, &y_truth);
        let mgcv_amp = mgcv_pred.iter().cloned().fold(f64::MIN, f64::max)
            - mgcv_pred.iter().cloned().fold(f64::MAX, f64::min);

        eprintln!("\n===== [{label}] p={p} nulldim={nulldim} n_penalties={} =====",
            train_design.penalties.len());
        eprintln!("  PROD  gam: log_lambda={prod_loglam:?} lambda={prod_lambdas:?} edf={prod_edf:?}");
        eprintln!("        max_err={prod_max:.4} amp(pp)={prod_amp:.4}");
        eprintln!("  CF    gam closed-form global-REML: rho={:.4} lambda={:.4e} edf={:.3} score={:.4}",
            cf.rho, cf.lambda, cf.edf, cf.reml_score);
        eprintln!("        max_err={cf_max:.4} amp(pp)={cf_amp:.4}");
        eprintln!("  MGCV  ds k={k}: sp={mgcv_sp:.4e} edf={mgcv_edf:.3}");
        eprintln!("        max_err={mgcv_max:.4} amp(pp)={mgcv_amp:.4}");
    }
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

    // ---- centers=50: genuine-recovery regime (mgcv escapes here) ----
    // A well-resolved basis must actually reconstruct the sine, so it carries BOTH
    // an absolute max-error budget and a match-or-beat-mgcv truth-RMSE check.
    // Truth peak-to-peak is 2.0; the measured MSI run reaches max_err≈0.11 (gam)
    // and ≈0.19 (mgcv), so 0.25 is a real recovery budget an order below the
    // ≈1.10 max error of the near-flat shelf — not the old 0.60 rubber-stamp.
    let c50 = fit_and_predict_diagnostics(
        "duchon-centers50",
        "y ~ duchon(x, centers=50)",
        &data,
        &x_test,
        &y_truth_test,
    );
    let c50_max = max_abs_err(&c50, &y_truth_test);
    let c50_rmse = rmse(&c50, &y_truth_test);
    let mgcv_c50 = mgcv_duchon_predict(&data, &x_test, 50);
    let mgcv_c50_rmse = rmse(&mgcv_c50, &y_truth_test);
    let c50_budget = 0.25_f64;
    eprintln!(
        "[duchon-sin8] duchon-centers50 max_err={c50_max:.4} truth_rmse={c50_rmse:.4}; \
         mgcv-ds-k50 truth_rmse={mgcv_c50_rmse:.4}"
    );
    assert!(
        c50_max <= c50_budget,
        "well-resolved centers=50 Duchon oversmooths sin8 at σ=0.10: \
         max_err {c50_max:.4} > {c50_budget:.2} (truth peak=2.0)"
    );
    assert!(
        c50_rmse <= 1.10 * mgcv_c50_rmse,
        "centers=50 Duchon recovers sin8 worse than mgcv bs=\"ds\" k=50: \
         gam truth_rmse={c50_rmse:.4} > 1.10 * mgcv={mgcv_c50_rmse:.4}"
    );

    // ---- default rank: resolution-limited regime (mgcv shelves here too) ----
    // At the default center count eight noisy cycles are under-resolved and the
    // mature Duchon collapses to a near-flat fit (truth RMSE≈0.70). An absolute
    // recovery budget is not meaningful (mgcv cannot meet one); the objective bar
    // is that gam shelves no worse than mgcv on truth recovery.
    let dflt = fit_and_predict_diagnostics(
        "duchon-default",
        "y ~ duchon(x)",
        &data,
        &x_test,
        &y_truth_test,
    );
    let dflt_rmse = rmse(&dflt, &y_truth_test);
    let mgcv_dflt = mgcv_duchon_predict(&data, &x_test, 40);
    let mgcv_dflt_rmse = rmse(&mgcv_dflt, &y_truth_test);
    eprintln!(
        "[duchon-sin8] duchon-default truth_rmse={dflt_rmse:.4}; \
         mgcv-ds-k40 truth_rmse={mgcv_dflt_rmse:.4}"
    );
    assert!(
        dflt_rmse <= 1.10 * mgcv_dflt_rmse,
        "default-rank Duchon recovers sin8 worse than the equally resolution-limited \
         mgcv bs=\"ds\" k=40: gam truth_rmse={dflt_rmse:.4} > 1.10 * mgcv={mgcv_dflt_rmse:.4}"
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
