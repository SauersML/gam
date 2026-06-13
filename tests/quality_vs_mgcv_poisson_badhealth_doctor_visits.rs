//! End-to-end OBJECTIVE quality: gam's Poisson(log) GAM with a smooth age term
//! must PREDICT held-out doctor-visit counts well — not merely reproduce mgcv's
//! in-sample fit.
//!
//! Real data: the `badhealth` dataset from Hilbe's COUNT package (German
//! Socioeconomic Panel), 1127 patients. Columns:
//!   * `numvisit` — number of doctor visits (count response, 0..40, nonneg int),
//!   * `age`      — patient age in years (20..60, continuous smooth covariate),
//!   * `badh`     — self-reported bad-health indicator (0/1, linear term).
//! Source (direct, no auth):
//!   https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/COUNT/badhealth.csv
//!
//! Use case: count regression with a smooth covariate and a binary covariate,
//! `numvisit ~ s(age) + linear(badh)`, family = Poisson, log link — the canonical
//! health-utilization count model. Visit propensity is genuinely nonlinear in
//! age, so a penalized smooth on age should out-predict a constant baseline. We
//! make a deterministic train/test split (every 4th row held out), fit on the
//! training rows only, predict the held-out rows, and score gam's OWN predictions
//! on count-appropriate, tool-free metrics:
//!
//!   PRIMARY (objective, tool-free): held-out mean Poisson deviance per obs must
//!     beat the intercept-only (train-mean) null model. The null predicts the
//!     training mean count for every held-out row; a model that has learned a real
//!     age/health signal achieves strictly lower held-out deviance. We require
//!     gam's held-out deviance <= 0.97 * null deviance (a clear, not marginal,
//!     improvement) AND a positive held-out predicted-vs-actual correlation.
//!
//!   BASELINE (match-or-beat): mgcv — the mature, standard GAM implementation —
//!     fits the SAME training rows with `gam(numvisit ~ s(age) + badh,
//!     family = poisson, method = "REML")` and predicts the SAME held-out rows on
//!     the response (count) scale. gam's held-out Poisson deviance must be no
//!     worse than `mgcv_test_deviance * 1.05`. mgcv is a baseline to match-or-beat
//!     on held-out accuracy, NOT a fitted target to reproduce.
//!
//! The in-sample rel_l2 between gam's and mgcv's fitted counts is printed for
//! context only and is deliberately NOT a pass criterion: matching another tool's
//! fitted output proves nothing about correctness.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pad_to, pearson, relative_l2, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const BADHEALTH_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/badhealth.csv");

/// Mean Poisson deviance per observation between observed counts `y` and fitted
/// means `mu`: `(1/n) * sum 2*( y*log(y/mu) - (y - mu) )`, with the convention
/// `y*log(y/mu) = 0` when `y == 0`. This is the natural, scale-aware loss for
/// count predictions — it penalizes a fitted mean that is too small or too large
/// asymmetrically, exactly as the Poisson likelihood does, and equals zero only
/// when `mu == y` everywhere.
fn mean_poisson_deviance(y: &[f64], mu: &[f64]) -> f64 {
    assert_eq!(y.len(), mu.len(), "poisson deviance length mismatch");
    let n = y.len() as f64;
    let mut acc = 0.0;
    for (&yi, &mui) in y.iter().zip(mu) {
        let m = mui.max(1e-12);
        let log_term = if yi > 0.0 { yi * (yi / m).ln() } else { 0.0 };
        acc += 2.0 * (log_term - (yi - m));
    }
    acc / n.max(1.0)
}

#[test]
fn gam_poisson_predicts_badhealth_visits_better_than_baseline() {
    init_parallelism();

    // ---- load the real badhealth dataset (age, badh -> numvisit) ----------
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
    // Sanity: the response really is nonnegative integer counts.
    assert!(
        numvisit
            .iter()
            .all(|&v| v >= 0.0 && (v - v.round()).abs() < 1e-9),
        "numvisit must be nonnegative integer counts"
    );

    // ---- deterministic train/test split: every 4th row is held out -------
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

    // ---- fit gam on TRAIN: numvisit ~ s(age) + linear(badh), Poisson, REML --
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result =
        fit_from_formula("numvisit ~ s(age) + linear(badh)", &train_ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the Poisson(log) family");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at the held-out rows: rebuild the design from the frozen
    // spec, apply beta to get the log-link predictor eta, then exp() to the mean
    // count (the response scale).
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &row) in test_rows.iter().enumerate() {
        test_grid[[i, age_idx]] = age[row];
        test_grid[[i, badh_idx]] = badh[row];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out points");
    let gam_test_eta = test_design.design.apply(&fit.fit.beta);
    let gam_test_mu: Vec<f64> = gam_test_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME model on TRAIN with mgcv, predict the SAME TEST -----
    // mgcv is the mature baseline; we read back its in-sample fitted counts (context),
    // edf, and held-out response-scale predictions in a SINGLE R subprocess to avoid
    // paying two mgcv fit costs (one per run_r call).  Test covariates ride along as
    // parallel columns padded to the training length; only the first `test_n` entries
    // are read back for prediction inside R.
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
        m <- gam(numvisit ~ s(age) + badh, data = df, family = poisson, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        k <- df$test_n[1]
        newd <- data.frame(age = df$test_age[1:k], badh = df$test_badh[1:k])
        emit("test_pred", as.numeric(predict(m, newdata = newd, type = "response")))
        "#,
    );
    let mgcv_train_fitted = r.vector("fitted").to_vec();
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_train_fitted.len(),
        train_rows.len(),
        "mgcv in-sample fitted length mismatch"
    );
    let mgcv_test_mu = r.vector("test_pred");
    assert_eq!(
        mgcv_test_mu.len(),
        test_rows.len(),
        "mgcv held-out prediction length mismatch"
    );

    // ---- objective metrics on gam's OWN held-out predictions --------------
    let gam_test_dev = mean_poisson_deviance(&test_numvisit, &gam_test_mu);
    let mgcv_test_dev = mean_poisson_deviance(&test_numvisit, mgcv_test_mu);

    // Null (intercept-only) baseline: predict the TRAINING mean count for every
    // held-out row. Its held-out deviance is the bar a real signal must beat.
    let train_mean = train_numvisit.iter().sum::<f64>() / train_numvisit.len() as f64;
    let null_test_mu = vec![train_mean; test_rows.len()];
    let null_test_dev = mean_poisson_deviance(&test_numvisit, &null_test_mu);

    // Positive predicted-vs-actual correlation confirms the ranking of held-out
    // counts is recovered (a degenerate constant fit would give ~0).
    let gam_test_corr = pearson(&gam_test_mu, &test_numvisit);

    // Context-only diagnostic: closeness of gam's in-sample fitted counts vs
    // mgcv's. NOT a pass criterion.
    let mut gam_train_grid = Array2::<f64>::zeros((train_rows.len(), p));
    for (i, &row) in train_rows.iter().enumerate() {
        gam_train_grid[[i, age_idx]] = age[row];
        gam_train_grid[[i, badh_idx]] = badh[row];
    }
    let gam_train_design = build_term_collection_design(gam_train_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_train_fitted: Vec<f64> = gam_train_design
        .design
        .apply(&fit.fit.beta)
        .iter()
        .map(|e| e.exp())
        .collect();
    let insample_rel = relative_l2(&gam_train_fitted, &mgcv_train_fitted);

    eprintln!(
        "badhealth s(age)+badh Poisson held-out: n_train={} n_test={} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         null_dev={null_test_dev:.4} gam_dev={gam_test_dev:.4} mgcv_dev={mgcv_test_dev:.4} \
         gam_test_corr={gam_test_corr:.4} (context: in-sample rel_l2 vs mgcv={insample_rel:.4})",
        train_rows.len(),
        test_rows.len(),
    );

    // ---- PRIMARY objective assertion: gam beats the intercept-only null -----
    // A model that has learned a genuine age/health signal predicts held-out
    // counts with strictly lower Poisson deviance than predicting the constant
    // training mean. We demand a clear (>=3%) improvement, not a marginal one.
    assert!(
        gam_test_dev <= 0.97 * null_test_dev,
        "gam held-out Poisson deviance {gam_test_dev:.4} does not clearly beat the \
         intercept-only null {null_test_dev:.4} (need <= 0.97 * null)"
    );

    // The held-out predicted-vs-actual correlation must be positive — gam ranks
    // high-visit patients above low-visit ones out of sample.
    assert!(
        gam_test_corr > 0.10,
        "gam held-out predicted-vs-actual correlation too low: {gam_test_corr:.4}"
    );

    // ---- BASELINE (match-or-beat): no worse than mgcv on held-out deviance --
    assert!(
        gam_test_dev <= mgcv_test_dev * 1.05,
        "gam held-out Poisson deviance {gam_test_dev:.4} exceeds mgcv {mgcv_test_dev:.4} * 1.05"
    );

    // ---- complexity sanity: edf in a signal-appropriate range (not matched) -
    // A nonlinear age smooth plus an intercept and the badh coefficient should
    // use a handful of effective parameters, well below an overfit blow-up.
    assert!(
        gam_edf > 2.0 && gam_edf < 15.0,
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}

/// Real-data 2-D arm: the SAME Poisson(log) count capability, but now gam must
/// recover a full **2-D log-mean surface** `log E[numvisit] = f(age, badh)` via a
/// tensor-product smooth `te(age, badh)` instead of the additive
/// `s(age) + linear(badh)` of the first arm. This lets the age profile of visit
/// propensity differ between healthy and self-reported-bad-health patients (an
/// age x health interaction on the log-mean), which the additive model cannot
/// represent. Truth is unknown on real data, so quality is again the held-out
/// Poisson deviance scored on gam's OWN response-scale predictions:
///
///   PRIMARY (objective, tool-free): the 2-D surface beats the intercept-only
///     (train-mean) null on held-out mean Poisson deviance by a clear margin
///     (`gam_dev <= 0.97 * null_dev`) with a positive predicted-vs-actual
///     correlation out of sample.
///
///   BASELINE (match-or-beat): mgcv fits the SAME training rows with
///     `gam(numvisit ~ s(age, by = factor(badh)) + factor(badh),
///     family = poisson, method = "REML")` and predicts the SAME held-out rows
///     on the response scale; gam's held-out deviance must be no worse than
///     `mgcv_dev * 1.05`. mgcv is a baseline to match-or-beat, NOT a fitted
///     target to reproduce.
///
/// badh is binary {0,1}, so a tensor `te(age, badh)` margin on badh is NOT
/// constructible in mgcv: it resets the binary margin's `k=2` back up to a
/// default that exceeds the two distinct values and then errors / fails the
/// inner loop. The mgcv-idiomatic encoding of the SAME effect that gam's
/// `te(age, badh)` represents over a binary margin — an age profile that differs
/// by health status — is the smooth-by-factor `s(age, by = factor(badh))` plus
/// the `factor(badh)` main effect (a separate age curve per badh level).
#[test]
fn gam_poisson_predicts_badhealth_visits_better_than_baseline_on_real_data() {
    init_parallelism();

    // ---- load the real badhealth dataset (age, badh -> numvisit) ----------
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
    assert!(
        numvisit
            .iter()
            .all(|&v| v >= 0.0 && (v - v.round()).abs() < 1e-9),
        "numvisit must be nonnegative integer counts"
    );

    // ---- deterministic train/test split: every 4th row is held out -------
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
    let test_numvisit: Vec<f64> = test_rows.iter().map(|&i| numvisit[i]).collect();

    // Training-only dataset (subset encoded rows; schema/headers unchanged so the
    // formula resolves identically). IDENTICAL rows, IDENTICAL order are handed to
    // both gam and mgcv.
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: numvisit ~ te(age, badh), Poisson, REML ---------
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("numvisit ~ te(age, badh)", &train_ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the Poisson(log) family");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at the held-out rows: rebuild the design from the frozen
    // spec, apply beta to get the log-link predictor eta, then exp() to the mean.
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &row) in test_rows.iter().enumerate() {
        test_grid[[i, age_idx]] = age[row];
        test_grid[[i, badh_idx]] = badh[row];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out points");
    let gam_test_eta = test_design.design.apply(&fit.fit.beta);
    let gam_test_mu: Vec<f64> = gam_test_eta.iter().map(|e| e.exp()).collect();

    // ---- fit the SAME 2-D model on TRAIN with mgcv, predict the SAME TEST ---
    // Held-out test covariates ride along as parallel columns padded to the
    // training length; only the first `test_n` entries are read back inside R.
    // A SINGLE R subprocess emits fitted values, edf, and test predictions so we
    // pay only one mgcv fit cost (vs two sequential run_r calls).
    let test_age: Vec<f64> = test_rows.iter().map(|&i| age[i]).collect();
    let test_badh: Vec<f64> = test_rows.iter().map(|&i| badh[i]).collect();
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
        # badh is binary {0,1}, so a tensor te(age, badh) margin on badh cannot
        # support any usable basis: mgcv resets the binary margin's k=2 back up to
        # its default (> 2 unique values) and then errors / fails the inner loop.
        # The mgcv-idiomatic way to model the SAME age x badh interaction (gam's
        # te(age, badh) over a binary margin = an age profile that differs by
        # health status) is a smooth-by-factor: a separate s(age) curve per badh
        # level plus the badh main effect.
        df$badhf <- factor(df$badh)
        m <- gam(numvisit ~ s(age, by = badhf) + badhf, data = df,
                 family = poisson, method = "REML")
        emit("fitted", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        k <- df$test_n[1]
        newd <- data.frame(age = df$test_age[1:k],
                           badhf = factor(df$test_badh[1:k], levels = levels(df$badhf)))
        emit("test_pred", as.numeric(predict(m, newdata = newd, type = "response")))
        "#,
    );
    let mgcv_train_fitted = r.vector("fitted").to_vec();
    let mgcv_edf = r.scalar("edf");
    assert_eq!(
        mgcv_train_fitted.len(),
        train_rows.len(),
        "mgcv in-sample fitted length mismatch"
    );
    let mgcv_test_mu = r.vector("test_pred");
    assert_eq!(
        mgcv_test_mu.len(),
        test_rows.len(),
        "mgcv held-out prediction length mismatch"
    );

    // ---- objective metrics on gam's OWN held-out predictions --------------
    let gam_test_dev = mean_poisson_deviance(&test_numvisit, &gam_test_mu);
    let mgcv_test_dev = mean_poisson_deviance(&test_numvisit, mgcv_test_mu);

    let train_mean = train_numvisit.iter().sum::<f64>() / train_numvisit.len() as f64;
    let null_test_mu = vec![train_mean; test_rows.len()];
    let null_test_dev = mean_poisson_deviance(&test_numvisit, &null_test_mu);

    let gam_test_corr = pearson(&gam_test_mu, &test_numvisit);

    // Context-only diagnostic: closeness of gam's vs mgcv's in-sample surface.
    let mut gam_train_grid = Array2::<f64>::zeros((train_rows.len(), p));
    for (i, &row) in train_rows.iter().enumerate() {
        gam_train_grid[[i, age_idx]] = age[row];
        gam_train_grid[[i, badh_idx]] = badh[row];
    }
    let gam_train_design = build_term_collection_design(gam_train_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_train_fitted: Vec<f64> = gam_train_design
        .design
        .apply(&fit.fit.beta)
        .iter()
        .map(|e| e.exp())
        .collect();
    let insample_rel = relative_l2(&gam_train_fitted, &mgcv_train_fitted);

    eprintln!(
        "badhealth te(age,badh) Poisson held-out: n_train={} n_test={} \
         gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         null_dev={null_test_dev:.4} gam_dev={gam_test_dev:.4} mgcv_dev={mgcv_test_dev:.4} \
         gam_test_corr={gam_test_corr:.4} (context: in-sample rel_l2 vs mgcv={insample_rel:.4})",
        train_rows.len(),
        test_rows.len(),
    );

    // ---- PRIMARY objective assertion: 2-D surface beats the null ----------
    assert!(
        gam_test_dev <= 0.97 * null_test_dev,
        "gam 2-D held-out Poisson deviance {gam_test_dev:.4} does not clearly beat the \
         intercept-only null {null_test_dev:.4} (need <= 0.97 * null)"
    );
    assert!(
        gam_test_corr > 0.10,
        "gam 2-D held-out predicted-vs-actual correlation too low: {gam_test_corr:.4}"
    );

    // ---- BASELINE (match-or-beat): no worse than mgcv on held-out deviance --
    assert!(
        gam_test_dev <= mgcv_test_dev * 1.05,
        "gam 2-D held-out Poisson deviance {gam_test_dev:.4} exceeds mgcv {mgcv_test_dev:.4} * 1.05"
    );

    // ---- complexity sanity: a 2-D tensor over a binary margin uses a modest
    // number of effective parameters, well below an overfit blow-up.
    assert!(
        gam_edf > 2.0 && gam_edf < 25.0,
        "gam 2-D effective dof out of sane range: {gam_edf:.3}"
    );
}
