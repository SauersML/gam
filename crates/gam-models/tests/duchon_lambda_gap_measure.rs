//! MEASUREMENT (not a pass/fail gate): quantify the Duchon sin8 over-smoothing
//! λ-gap against mgcv. Fits the exact `duchon_sin8_max_error_within_budget`
//! fixture (n=240, σ=0.10, sin(2π·8x)), reports:
//!   - production gam: fitted log λ, edf_total, truth max-error + amplitude
//!   - gam's OWN global-grid closed-form REML optimum on the same basis
//!     (independent optimizer) — its rho/lambda/edf/score + amplitude
//!   - mgcv bs="ds" m=c(2,0) REML: sp, sum(edf) + amplitude
//! The three-way comparison splits the mechanism: production-edf ≈ closed-form-edf
//! ⇒ gam's REML score itself prefers the over-smoothed λ (score-assembly, a);
//! closed-form-edf ≫ production-edf at a better score ⇒ production outer optimizer
//! stuck in the wrong basin (optimizer, b).

use csv::StringRecord;
use gam_data::{EncodedDataset, encode_recordswith_inferred_schema};
use gam_models::fit_orchestration::{FitConfig, FitResult, fit_from_formula};
use gam_solve::gaussian_reml::gaussian_reml_closed_form;
use gam_terms::smooth::build_term_collection_design;
use gam_test_support::reference::{Column, run_r};
use ndarray::{Array1, Array2};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

fn make_sin_dataset(freq: f64, sigma: f64, n: usize, seed: u64) -> EncodedDataset {
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

fn max_abs_err(yhat: &[f64], y: &[f64]) -> f64 {
    yhat.iter()
        .zip(y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max)
}

fn amp_pp(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::MIN, f64::max) - v.iter().cloned().fold(f64::MAX, f64::min)
}

fn mgcv_diag(data: &EncodedDataset, x_test: &[f64], k: usize) -> (f64, f64, Vec<f64>) {
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
            m <- gam(y ~ s(x, bs = "ds", k = {k}, m = c(2, 0)), data = train, method = "REML")
            emit("edf", as.numeric(sum(m$edf)))
            emit("sp",  as.numeric(m$sp))
            emit("fitted", as.numeric(predict(m, newdata = grid)))
            "#
        ),
    );
    (
        r.vector("edf")[0],
        r.vector("sp")[0],
        r.vector("fitted").to_vec(),
    )
}

#[test]
fn zz_measure_duchon_sin8_lambda_gap() {
    let data = make_sin_dataset(8.0, 0.10, 240, 11);
    let x_test: Vec<f64> = (0..400).map(|i| 0.001 + 0.998 * i as f64 / 399.0).collect();
    let y_truth: Vec<f64> = x_test
        .iter()
        .map(|t| (2.0 * std::f64::consts::PI * 8.0 * t).sin())
        .collect();
    let x_col = data.column_map()["x"];
    let y_col = data.column_map()["y"];
    let ncols = data.headers.len();

    for (label, body, k) in [
        ("duchon-default", "duchon(x)", 40usize),
        ("duchon-centers50", "duchon(x, centers=50)", 50),
    ] {
        // ---- production gam fit ----
        let cfg = FitConfig {
            family: Some("gaussian".to_string()),
            ..FitConfig::default()
        };
        let FitResult::Standard(fit) =
            fit_from_formula(&format!("y ~ {body}"), &data, &cfg).expect("fit")
        else {
            panic!("standard")
        };
        let prod_loglam = fit.fit.log_lambdas.to_vec();
        let prod_lam = fit.fit.lambdas.to_vec();
        let prod_edf = fit.fit.edf_total();
        let prod_score = fit.fit.reml_score;

        // test-grid design (frozen spec) for predictions
        let n_t = x_test.len();
        let mut mt = Array2::<f64>::zeros((n_t, ncols));
        for (i, &t) in x_test.iter().enumerate() {
            mt[[i, x_col]] = t;
        }
        let test_design =
            build_term_collection_design(mt.view(), &fit.resolvedspec).expect("test design");
        let prod_pred = test_design.design.apply(&fit.fit.beta).to_vec();
        let prod_max = max_abs_err(&prod_pred, &y_truth);

        // ---- train design X + summed penalty S from the frozen spec ----
        let n_tr = data.values.nrows();
        let mut mtr = Array2::<f64>::zeros((n_tr, ncols));
        for i in 0..n_tr {
            mtr[[i, x_col]] = data.values[[i, x_col]];
        }
        let train_design =
            build_term_collection_design(mtr.view(), &fit.resolvedspec).expect("train design");
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
        let y_tr = data.values.column(y_col).to_owned();

        // gam's OWN global-grid REML optimum on this basis (single summed penalty)
        let cf = gaussian_reml_closed_form(x_dense.view(), y_tr.view(), s.view(), None, None)
            .expect("closed form");
        let xt = test_design.design.to_dense();
        let cf_beta = Array1::from(cf.coefficients.to_vec());
        let cf_pred = xt.dot(&cf_beta).to_vec();
        let cf_max = max_abs_err(&cf_pred, &y_truth);

        // ---- mgcv reference ----
        let (mgcv_edf, mgcv_sp, mgcv_pred) = mgcv_diag(&data, &x_test, k);
        let mgcv_max = max_abs_err(&mgcv_pred, &y_truth);

        eprintln!(
            "\n===== [{label}] p={p} nulldim={nulldim} n_penalties={} truth_amp_pp=2.0 =====",
            train_design.penalties.len()
        );
        eprintln!(
            "  PROD gam  : log_lambda={prod_loglam:?} lambda={prod_lam:?}\n              edf={prod_edf:?} reml_score={prod_score:.4} max_err={prod_max:.4} amp_pp={:.4}",
            amp_pp(&prod_pred)
        );
        eprintln!(
            "  CF   gam  : rho={:.4} lambda={:.4e} edf={:.3} reml_score={:.4} max_err={cf_max:.4} amp_pp={:.4}",
            cf.rho,
            cf.lambda,
            cf.edf,
            cf.reml_score,
            amp_pp(&cf_pred)
        );
        eprintln!(
            "  MGCV ds k={k}: sp={mgcv_sp:.4e} edf={mgcv_edf:.3} max_err={mgcv_max:.4} amp_pp={:.4}",
            amp_pp(&mgcv_pred)
        );
    }
}
