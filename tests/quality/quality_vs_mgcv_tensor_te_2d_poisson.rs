//! End-to-end quality: gam's tensor-product 2-D smooth `te(x, z)` under a
//! **non-Gaussian** family (Poisson, log link) must RECOVER THE TRUE log-mean
//! surface from which the counts were drawn.
//!
//! This is the essential *combination* test: tensor products under a
//! non-Gaussian family. A Gaussian tensor smooth can fit perfectly while a
//! Poisson one diverges if the tensor-product penalty is recomputed incorrectly
//! across PIRLS iterations or the log-link gradient/weights are mishandled. The
//! data are generated from a KNOWN function, `eta_true(x, z) = sin(pi*x) *
//! cos(pi*z)`, with `count ~ Poisson(exp(eta_true))`, so there is an objective
//! ground truth to recover.
//!
//! OBJECTIVE METRIC (primary): the test asserts gam's fitted linear predictor
//! (the log-mean surface, the scale on which the tensor penalty acts and the
//! natural place a PIRLS / link-inversion bug would show) recovers the true
//! `eta_true` at the training points with `RMSE(gam_eta, eta_true)` below a
//! principled bar tied to the irreducible noise of the design. `mgcv` is fit on
//! the IDENTICAL data and demoted to a **match-or-beat accuracy baseline**: gam's
//! truth-recovery error must be no worse than `mgcv`'s by more than 10%. We do
//! NOT assert gam reproduces mgcv's (itself noisy) fitted surface — only that gam
//! recovers the truth at least as well as the mature reference does.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};
use std::f64::consts::PI;
use std::path::Path;

use gam::test_support::reference::{Column, pad_to, pearson, relative_l2, rmse, run_r};

// Source: `badhealth` from the R package `COUNT` (Hilbe, "Negative Binomial
// Regression", 2nd ed.); German health-survey counts of doctor visits.
// Columns: numvisit (count of doctor visits), badh (1 = self-rated bad health),
// age (years). Vendored at bench/datasets/badhealth.csv.
const BADHEALTH_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/badhealth.csv");

/// Poisson deviance of predictions `mu` (the fitted mean, > 0) against observed
/// counts `y`: `2 * sum( y*log(y/mu) - (y - mu) )`, with the `y == 0` term taken
/// as its limit `2*mu`. Lower is better; this is the natural held-out goodness
/// metric for count data (the Poisson analogue of RMSE).
fn poisson_deviance(mu: &[f64], y: &[f64]) -> f64 {
    assert_eq!(mu.len(), y.len(), "poisson_deviance length mismatch");
    let mut dev = 0.0;
    for (&m, &obs) in mu.iter().zip(y) {
        let m = m.max(1e-12);
        let term = if obs > 0.0 {
            obs * (obs / m).ln() - (obs - m)
        } else {
            m
        };
        dev += 2.0 * term;
    }
    dev
}

#[test]
fn gam_tensor_te_2d_poisson_matches_mgcv() {
    init_parallelism();

    // ---- synthetic Poisson-count truth on the unit square ------------------
    // count_expected = exp(sin(pi*x) * cos(pi*z)); count ~ Poisson(count_expected).
    // Fixed seed => the SAME draws feed gam and mgcv, so any disagreement is in
    // the fitting, not the data.
    let n = 300usize;
    let mut rng = StdRng::seed_from_u64(20260530);
    let u = Uniform::new(0.0_f64, 1.0).expect("uniform [0,1]");

    let mut x = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    let mut count = Vec::with_capacity(n);
    let mut eta_true = Vec::with_capacity(n);
    for _ in 0..n {
        let xi = u.sample(&mut rng);
        let zi = u.sample(&mut rng);
        let eta = (PI * xi).sin() * (PI * zi).cos(); // true log-mean
        let lambda = eta.exp().max(1e-12);
        let draw: f64 = Poisson::new(lambda)
            .expect("valid Poisson rate")
            .sample(&mut rng);
        x.push(xi);
        z.push(zi);
        count.push(draw);
        eta_true.push(eta);
    }

    // ---- fit with gam: count ~ te(x, z, k=7), Poisson / log link, REML ------
    let headers = ["x", "z", "count"].into_iter().map(String::from).collect();
    let rows: Vec<StringRecord> = (0..n)
        .map(|i| {
            StringRecord::from(vec![
                x[i].to_string(),
                z[i].to_string(),
                count[i].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode poisson dataset");
    let col = ds.column_map();
    let x_idx = col["x"];
    let z_idx = col["z"];

    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("count ~ te(x, z, k=7)", &ds, &cfg).expect("gam poisson te fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Poisson te(x, z)");
    };

    // gam linear predictor (log scale) at the training points: rebuilding the
    // frozen design and applying beta yields eta = design*beta directly, BEFORE
    // the log-link inverse — exactly the scale on which the tensor penalty acts.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, x_idx]] = x[i];
        grid[[i, z_idx]] = z[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild te design at training points");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with mgcv (the mature reference) ---------------
    // family = poisson(link = "log"), method = "REML"; emit the linear predictor.
    let r = run_r(
        &[
            Column::new("x", &x),
            Column::new("z", &z),
            Column::new("count", &count),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(count ~ te(x, z, k = 7), data = df,
                 family = poisson(link = "log"), method = "REML")
        emit("eta", as.numeric(predict(m, type = "link")))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_eta = r.vector("eta");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_eta.len(), n, "mgcv linear-predictor length mismatch");

    // ---- OBJECTIVE METRIC: truth recovery on the log (linear-predictor) scale
    // The data were drawn from a KNOWN log-mean surface eta_true; the quality
    // claim is that gam recovers it. Both gam and mgcv are scored by RMSE of
    // their fitted linear predictor against eta_true at the training points.
    let gam_err = rmse(&gam_eta, &eta_true);
    let mgcv_err = rmse(mgcv_eta, &eta_true);

    // Context only (NOT a pass criterion): how close gam's surface is to mgcv's
    // own (itself noisy) fit. Matching mgcv proves nothing; we print it so a
    // reviewer can see the two surfaces are in the same family.
    let rel_to_mgcv = relative_l2(&gam_eta, mgcv_eta);
    let corr_to_mgcv = pearson(&gam_eta, mgcv_eta);

    eprintln!(
        "te(x,z) Poisson/log: n={n} mgcv_edf={mgcv_edf:.3} \
         rmse_to_truth(gam)={gam_err:.4} rmse_to_truth(mgcv)={mgcv_err:.4} \
         [context] rel_l2(gam,mgcv)={rel_to_mgcv:.4} pearson(gam,mgcv)={corr_to_mgcv:.5}"
    );

    // PRIMARY: gam must recover the true log-mean surface. The true eta_true =
    // sin(pi*x)*cos(pi*z) ranges over [-1, 1] (signal range 2). With n=200 and
    // counts as small as exp(-1) ~ 0.37, the Poisson information is genuinely
    // sparse, so a smoother cannot reproduce eta_true exactly; but a correct
    // tensor-product / log-link fit must stay well inside a fraction of the
    // signal range. We require RMSE(gam_eta, eta_true) < 0.30 — under 15% of the
    // 2.0 signal span. A per-iteration penalty mis-application or a botched
    // link-gradient distorts the surface far beyond this and is caught here.
    assert!(
        gam_err < 0.30,
        "gam should recover the true log-mean surface: rmse_to_truth={gam_err:.4} (bar 0.30)"
    );

    // MATCH-OR-BEAT: gam's truth-recovery error must be no worse than the mature
    // reference's by more than 10%. This holds mgcv as an accuracy BASELINE on
    // the objective metric rather than as a fit to be reproduced.
    assert!(
        gam_err <= mgcv_err * 1.10,
        "gam's truth-recovery error must match-or-beat mgcv: \
         rmse_to_truth(gam)={gam_err:.4} vs mgcv*1.10={:.4}",
        mgcv_err * 1.10
    );
}

/// REAL-DATA arm: the SAME capability (a 2-D tensor-product `te(age, badh)`
/// under Poisson/log) on the `badhealth` survey counts, where the truth is
/// unknown. Quality is judged out-of-sample: a deterministic split (every 4th
/// row held out), fit on train, predict the held-out doctor-visit counts, and
/// assert OBJECTIVE held-out Poisson deviance — both an absolute bar and a
/// match-or-beat against mgcv on that SAME metric. mgcv is a baseline to beat,
/// never a fit to reproduce.
#[test]
fn gam_tensor_te_2d_poisson_matches_mgcv_on_real_data() {
    init_parallelism();

    // ---- load the badhealth dataset (age, badh -> numvisit count) ----------
    let ds = load_csvwith_inferred_schema(Path::new(BADHEALTH_CSV)).expect("load badhealth.csv");
    let col = ds.column_map();
    let age_idx = col["age"];
    let badh_idx = col["badh"];
    let numvisit_idx = col["numvisit"];
    let age: Vec<f64> = ds.values.column(age_idx).to_vec();
    let badh: Vec<f64> = ds.values.column(badh_idx).to_vec();
    let numvisit: Vec<f64> = ds.values.column(numvisit_idx).to_vec();
    let n = age.len();
    assert!(n > 1000, "badhealth should have ~1127 rows, got {n}");

    // ---- deterministic train/test split: every 4th row held out -----------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 700 && test_rows.len() > 200,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_age: Vec<f64> = train_rows.iter().map(|&i| age[i]).collect();
    let train_badh: Vec<f64> = train_rows.iter().map(|&i| badh[i]).collect();
    let train_numvisit: Vec<f64> = train_rows.iter().map(|&i| numvisit[i]).collect();
    let test_age: Vec<f64> = test_rows.iter().map(|&i| age[i]).collect();
    let test_badh: Vec<f64> = test_rows.iter().map(|&i| badh[i]).collect();
    let test_numvisit: Vec<f64> = test_rows.iter().map(|&i| numvisit[i]).collect();

    // Build a training-only dataset by sub-setting the encoded rows; headers,
    // schema and column kinds are unchanged, so the formula resolves identically.
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: numvisit ~ te(age, badh, k=[5, 2]), Poisson/log ---
    // `badh` is BINARY {0,1} (exactly 2 unique values), so its tensor margin can
    // hold at most a 2-function (linear-across-the-two-levels) basis. We give it
    // k=2 explicitly and let the continuous `age` margin carry the smooth shape
    // at k=5. This mirrors the per-margin k=c(5,2) the mgcv reference uses below,
    // keeping both fits on the SAME tensor basis dimensions (apples-to-apples).
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("numvisit ~ te(age, badh, k=[5, 2])", &train_ds, &cfg)
        .expect("gam poisson te fit on badhealth");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Poisson te(age, badh)");
    };

    // gam predictions at the held-out points: rebuild the design from the frozen
    // spec; design*beta is the log-mean, so exp() gives the predicted count mean.
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for i in 0..test_rows.len() {
        test_grid[[i, age_idx]] = test_age[i];
        test_grid[[i, badh_idx]] = test_badh[i];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild te design at held-out points");
    let gam_test_eta: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
    let gam_test_mu: Vec<f64> = gam_test_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME model on TRAIN with mgcv, predict the SAME TEST -------
    // mgcv is the mature baseline. The harness exposes one data.frame per call,
    // so we ride the held-out age/badh along as parallel columns (padded to the
    // train length) plus a count of how many test rows to read back, and predict
    // on the response scale.
    let r = run_r(
        &[
            Column::new("age", &train_age),
            Column::new("badh", &train_badh),
            Column::new("numvisit", &train_numvisit),
            Column::new("test_age", &pad_to(&test_age, train_age.len())),
            Column::new("test_badh", &pad_to(&test_badh, train_age.len())),
            Column::new("test_n", &vec![test_age.len() as f64; train_age.len()]),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        # badh is BINARY {0,1} (exactly 2 unique values), so its tensor margin can
        # hold at most a 2-function basis. mgcv's default `cr` (cubic-regression)
        # margin has a hard minimum of k=3 and would REJECT k=2 ("k too small -
        # reset to default") then fail constructing 5 knots on a 2-value covariate.
        # A thin-plate (`tp`) margin DOES support k=2, giving exactly the linear-
        # across-the-two-levels effect that fully represents a binary covariate.
        # age is continuous and keeps the smooth `cr` margin at k=5. The resulting
        # tensor basis dimensions c(5, 2) match the gam side (apples-to-apples).
        m <- gam(numvisit ~ te(age, badh, bs = c("cr", "tp"), k = c(5, 2)), data = df,
                 family = poisson(link = "log"), method = "REML")
        k <- df$test_n[1]
        newd <- data.frame(age = df$test_age[1:k], badh = df$test_badh[1:k])
        emit("test_mu", as.numeric(predict(m, newdata = newd, type = "response")))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_test_mu = r.vector("test_mu");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_test_mu.len(),
        test_rows.len(),
        "mgcv held-out prediction length mismatch"
    );

    // ---- OBJECTIVE METRIC: held-out Poisson deviance (per held-out row) -----
    let gam_dev = poisson_deviance(&gam_test_mu, &test_numvisit) / test_rows.len() as f64;
    let mgcv_dev = poisson_deviance(mgcv_test_mu, &test_numvisit) / test_rows.len() as f64;

    // Baseline reference for the absolute bar: the deviance of the constant-mean
    // (intercept-only) predictor, i.e. predict the TRAIN mean count everywhere.
    let train_mean = train_numvisit.iter().sum::<f64>() / train_numvisit.len() as f64;
    let null_mu = vec![train_mean; test_rows.len()];
    let null_dev = poisson_deviance(&null_mu, &test_numvisit) / test_rows.len() as f64;

    // Context only (NOT a pass criterion): closeness of the two fitted surfaces.
    let rel_to_mgcv = relative_l2(&gam_test_mu, mgcv_test_mu);
    let corr_to_mgcv = pearson(&gam_test_mu, mgcv_test_mu);

    eprintln!(
        "badhealth te(age,badh) Poisson held-out: n_train={} n_test={} mgcv_edf={mgcv_edf:.3} \
         dev_gam={gam_dev:.4} dev_mgcv={mgcv_dev:.4} dev_null={null_dev:.4} \
         [context] rel_l2(gam,mgcv)={rel_to_mgcv:.4} pearson(gam,mgcv)={corr_to_mgcv:.5}",
        train_rows.len(),
        test_rows.len(),
    );

    // ---- PRIMARY objective assertion: gam beats the constant-mean predictor --
    // A competent count smoother of doctor visits on age + bad-health must
    // explain held-out structure the null model cannot. We require gam's mean
    // held-out deviance to be at least 5% below the intercept-only baseline.
    assert!(
        gam_dev < null_dev * 0.95,
        "gam's held-out Poisson deviance {gam_dev:.4} must beat the constant-mean \
         baseline {null_dev:.4} (bar: < 95% of null)"
    );

    // ---- BASELINE (match-or-beat): no worse than mgcv on held-out deviance --
    assert!(
        gam_dev <= mgcv_dev * 1.10,
        "gam held-out Poisson deviance {gam_dev:.4} exceeds mgcv {mgcv_dev:.4} * 1.10"
    );
}
