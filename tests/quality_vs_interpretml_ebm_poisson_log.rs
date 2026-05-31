//! End-to-end OBJECTIVE quality: gam's penalized PoissonGAM must RECOVER THE
//! KNOWN MEAN SURFACE `mu_true(x) = exp(true_eta(x))` on held-out data, and do so
//! at least as accurately as **InterpretML's `ExplainableBoostingRegressor`**
//! (the ML-world glassbox GAM) fit to the identical count data.
//!
//! The data are generated from a fully known additive log-mean
//!   true_eta = 0.5 + 0.2*sin(pi*x1/5) + 0.15*cos(pi*x2/5),  count ~ Poisson(exp(true_eta)),
//! so there is a GROUND TRUTH mean `mu_true = exp(true_eta)` at every covariate.
//! The objective quality of any Poisson learner is how close its fitted mean comes
//! to that true mean on data it did not train on — NOT how close it comes to a peer
//! tool's (equally noisy) fit. So the pass/fail criterion is truth recovery, and
//! the EBM is demoted to a baseline-to-match-or-beat on that same truth metric.
//!
//! OBJECTIVE METRIC (the primary assertion): on a held-out 30% test fold, the
//! Poisson deviance of gam's fitted mean against the TRUE mean,
//!   D_true(mu_hat) = 2*sum( mu_true*log(mu_true/mu_hat) - (mu_true - mu_hat) ),
//! must be small in absolute terms — at most a small fraction of the irreducible
//! deviance the truth itself carries against the realized counts. Equivalently the
//! held-out RMSE of mu_hat against mu_true must be a small fraction of the signal
//! amplitude. This is a real recovery claim: a broken PIRLS loop or a mis-inverted
//! log link cannot recover exp(true_eta) and fails it.
//!
//! Setup (identical bytes fed to both engines):
//!   * synthetic n=250, x1,x2 ~ U[0,10], seed 20260530, counts as above.
//!   * a fixed 70/30 train/test split (indices computed identically Rust-side and
//!     handed to EBM as a `fold` column: 1 = test, 0 = train).
//!   * gam fits `count ~ s(x1, k=6) + s(x2, k=6)` (Poisson, log link, REML) on the
//!     TRAIN rows, then predicts mu on the TEST rows.
//!   * EBM fits on the TRAIN rows, predicts mu on the TEST rows.
//!
//! Assertions:
//!   1. TRUTH RECOVERY (primary): gam's held-out RMSE against the TRUE mean
//!      `mu_true` is a small fraction of the true-mean range — gam has recovered
//!      the smooth count surface, not merely tracked the noisy realization.
//!   2. MATCH-OR-BEAT (baseline): gam's held-out truth-deviance is no worse than
//!      the EBM's truth-deviance times 1.10. The mature ML additive learner sets
//!      the accuracy bar; gam must meet or beat it on recovering the SAME truth.
//! A genuine recovery shortfall failing is a real bug in gam's Poisson/PIRLS path,
//! never a reason to weaken these bounds.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Poisson, Uniform};
use std::path::Path;

/// German health-care utilization data ("badhealth"), source: `gamlss.data`
/// (Rigby & Stasinopoulos), bundled at bench/datasets/badhealth.csv. Columns:
/// `numvisit` (count of doctor visits, the Poisson response), `badh` (1 if the
/// subject self-reports bad health, else 0), `age` (years).
const BADHEALTH_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/badhealth.csv");

const N: usize = 250;
const SEED: u64 = 20260530;

fn truth_eta(x1: f64, x2: f64) -> f64 {
    let pi = std::f64::consts::PI;
    0.5 + 0.2 * (pi * x1 / 5.0).sin() + 0.15 * (pi * x2 / 5.0).cos()
}

/// Per-row Poisson deviance contribution: 2*(y*log(y/mu) - (y - mu)), with the
/// y=0 limit y*log(y/mu) -> 0. This is the exact quantity both engines minimize.
fn poisson_dev_unit(y: f64, mu: f64) -> f64 {
    let term = if y > 0.0 { y * (y / mu).ln() } else { 0.0 };
    2.0 * (term - (y - mu))
}

#[test]
fn gam_poisson_log_matches_interpretml_ebm() {
    init_parallelism();

    // ---- synthetic count data + fixed train/test fold (identical to both) ----
    let mut rng = StdRng::seed_from_u64(SEED);
    let ux = Uniform::new(0.0, 10.0).expect("uniform 0..10");
    let mut x1 = Vec::with_capacity(N);
    let mut x2 = Vec::with_capacity(N);
    let mut y = Vec::with_capacity(N);
    for _ in 0..N {
        let a = ux.sample(&mut rng);
        let b = ux.sample(&mut rng);
        let pois = Poisson::new(truth_eta(a, b).exp()).expect("poisson mean > 0");
        let count: f64 = pois.sample(&mut rng);
        x1.push(a);
        x2.push(b);
        y.push(count);
    }

    // Deterministic 70/30 split by row index: every 10th-and-beyond-7 row is
    // test. fold[i] == 1.0 means TEST, 0.0 means TRAIN — handed verbatim to EBM.
    let fold: Vec<f64> = (0..N)
        .map(|i| if i % 10 >= 7 { 1.0 } else { 0.0 })
        .collect();
    let train_idx: Vec<usize> = (0..N).filter(|&i| fold[i] == 0.0).collect();
    let test_idx: Vec<usize> = (0..N).filter(|&i| fold[i] == 1.0).collect();
    assert!(
        train_idx.len() > 100 && test_idx.len() > 50,
        "split sanity: {} train / {} test",
        train_idx.len(),
        test_idx.len()
    );

    // ---- fit with gam on TRAIN rows: count ~ s(x1,k=6)+s(x2,k=6), Poisson -----
    let headers = ["x1", "x2", "count"]
        .into_iter()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = train_idx
        .iter()
        .map(|&i| StringRecord::from(vec![x1[i].to_string(), x2[i].to_string(), y[i].to_string()]))
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode train dataset");
    let col = ds.column_map();
    let x1_idx = col["x1"];
    let x2_idx = col["x2"];

    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("count ~ s(x1, k=6) + s(x2, k=6)", &ds, &cfg)
        .expect("gam poisson fit on train fold");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for the Poisson(log) family");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Rebuild the frozen design at the TEST covariates; for a log link
    // design*beta is eta_hat and exp(eta_hat) is the fitted mean mu_hat.
    let n_test = test_idx.len();
    let mut grid = Array2::<f64>::zeros((n_test, ds.headers.len()));
    for (row, &i) in test_idx.iter().enumerate() {
        grid[[row, x1_idx]] = x1[i];
        grid[[row, x2_idx]] = x2[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at test points");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let gam_mu: Vec<f64> = gam_eta.iter().map(|e| e.exp()).collect();
    assert_eq!(gam_mu.len(), n_test, "gam test-mu length mismatch");

    // ---- fit the SAME data with InterpretML EBM (Poisson deviance) -----------
    // ExplainableBoostingRegressor with the Poisson-deviance objective is the ML
    // analog of a PoissonGAM: additive shape functions of x1 and x2, log link.
    // We pass the fold column so the Python side splits on the EXACT same rows.
    let r = run_python(
        &[
            Column::new("x1", &x1),
            Column::new("x2", &x2),
            Column::new("count", &y),
            Column::new("fold", &fold),
        ],
        r#"
import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor

x1   = np.asarray(df["x1"],    dtype=float)
x2   = np.asarray(df["x2"],    dtype=float)
y    = np.asarray(df["count"], dtype=float)
fold = np.asarray(df["fold"],  dtype=float)

train = fold == 0.0
test  = fold == 1.0
X = np.column_stack([x1, x2])

# Poisson-deviance objective => log link, additive shape functions only
# (interactions=0 keeps it a pure additive GAM, matching s(x1)+s(x2)).
ebm = ExplainableBoostingRegressor(
    objective="poisson_deviance",
    interactions=0,
    random_state=20260530,
)
ebm.fit(X[train], y[train])
mu = np.asarray(ebm.predict(X[test]), dtype=float)
# EBM's regression predictions are on the response (mean) scale already; guard
# against any non-positive mean before it reaches a Poisson deviance.
mu = np.clip(mu, 1e-8, None)
emit("mu", mu)
"#,
    );
    let ebm_mu = r.vector("mu");
    assert_eq!(ebm_mu.len(), n_test, "EBM test-mu length mismatch");

    // ---- OBJECTIVE evaluation on the TEST fold against the KNOWN truth --------
    // The data were generated from a known log-mean, so the true mean at each test
    // covariate is mu_true = exp(true_eta). Objective quality is recovery of THAT
    // surface on held-out points, not agreement with a peer tool's noisy fit.
    let mu_true: Vec<f64> = test_idx
        .iter()
        .map(|&i| truth_eta(x1[i], x2[i]).exp())
        .collect();
    let y_test: Vec<f64> = test_idx.iter().map(|&i| y[i]).collect();

    // Truth-recovery deviance: each learner's fitted mean against the TRUE mean.
    // (D_true uses mu_true in the y-slot of the Poisson deviance; the y=0 limit is
    //  never hit because exp(true_eta) > 0 everywhere.)
    let gam_truth_dev: f64 = mu_true
        .iter()
        .zip(&gam_mu)
        .map(|(&mt, &mh)| poisson_dev_unit(mt, mh))
        .sum();
    let ebm_truth_dev: f64 = mu_true
        .iter()
        .zip(ebm_mu)
        .map(|(&mt, &mh)| poisson_dev_unit(mt, mh))
        .sum();

    // (1) Truth-recovery RMSE on the held-out mean surface, reported relative to
    //     the amplitude (range) of the true mean over the test fold. This is the
    //     primary objective bar: a small fraction of the signal range.
    let n_test_f = n_test as f64;
    let gam_truth_rmse = (gam_mu
        .iter()
        .zip(&mu_true)
        .map(|(&mh, &mt)| (mh - mt).powi(2))
        .sum::<f64>()
        / n_test_f)
        .sqrt();
    let mu_true_min = mu_true.iter().cloned().fold(f64::INFINITY, f64::min);
    let mu_true_max = mu_true.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mu_true_range = (mu_true_max - mu_true_min).max(1e-12);
    let rel_truth_rmse = gam_truth_rmse / mu_true_range;

    // Irreducible deviance the truth itself carries against the realized counts —
    // a no-model-can-beat reference scale for the held-out deviance (context only).
    let truth_vs_counts_dev: f64 = y_test
        .iter()
        .zip(&mu_true)
        .map(|(&yi, &mt)| poisson_dev_unit(yi, mt))
        .sum();

    // Match-or-beat ratio on the SAME truth metric (gam vs the mature ML learner).
    let truth_dev_ratio = gam_truth_dev / ebm_truth_dev.max(1e-12);

    eprintln!(
        "poisson truth-recovery: n_train={} n_test={} gam_edf={gam_edf:.3} \
         gam_truth_dev={gam_truth_dev:.4} ebm_truth_dev={ebm_truth_dev:.4} (ratio={truth_dev_ratio:.4}) \
         gam_truth_rmse={gam_truth_rmse:.4} mu_true_range={mu_true_range:.4} (rel={rel_truth_rmse:.4}) \
         irreducible_dev(truth_vs_counts)={truth_vs_counts_dev:.4}",
        train_idx.len(),
        n_test
    );

    // (1) PRIMARY — truth recovery: gam's held-out fitted mean tracks the TRUE
    //     mean to well within a fifth of the true-mean amplitude. exp(true_eta)
    //     ranges only mildly (the signal is gentle), so a learner that recovered
    //     the smooth surface lands far inside this; a broken PIRLS loop or a
    //     mis-inverted log link cannot.
    assert!(
        rel_truth_rmse < 0.20,
        "gam did not recover the true held-out mean surface: \
         RMSE(mu_hat, mu_true)={gam_truth_rmse:.4} is {rel_truth_rmse:.4} of the \
         true-mean range {mu_true_range:.4} (bar 0.20)"
    );

    // (2) MATCH-OR-BEAT — gam's truth-deviance must be no worse than the mature
    //     EBM's by more than 10%. Both fit the identical training counts and are
    //     scored against the identical known truth; gam must be at least as
    //     accurate as the best-in-class ML additive learner at recovering it.
    assert!(
        truth_dev_ratio <= 1.10,
        "gam is materially less accurate than InterpretML EBM at recovering the \
         true mean: gam_truth_dev={gam_truth_dev:.4} ebm_truth_dev={ebm_truth_dev:.4} \
         (ratio={truth_dev_ratio:.4}, bar 1.10)"
    );
}

/// REAL-DATA arm of the SAME capability (penalized Poisson/log GAM): on the
/// `badhealth` doctor-visit counts there is NO known ground-truth mean, so the
/// objective quality of a count learner is its HELD-OUT predictive accuracy on
/// the realized counts. We make a deterministic 70/30 train/test split (fixed by
/// row index), fit gam on the train rows, predict μ on the test rows, and assert:
///
///   PRIMARY (objective, tool-free): mean held-out Poisson unit deviance of gam's
///     fitted mean against the ACTUAL test counts is below an absolute bar — gam
///     genuinely predicts the count surface out of sample, far better than the
///     intercept-only (constant-mean) Poisson predictor on the same test rows.
///   BASELINE (match-or-beat): gam's held-out deviance is no worse than
///     InterpretML's `ExplainableBoostingRegressor` (poisson_deviance objective)
///     fit to the IDENTICAL train rows and scored on the IDENTICAL test rows,
///     within a 10% margin. The mature ML additive learner is a baseline to meet
///     or beat on held-out accuracy, never an output to reproduce.
#[test]
fn gam_poisson_log_matches_interpretml_ebm_on_real_data() {
    init_parallelism();

    // ---- load the real badhealth dataset (age, badh -> numvisit count) -----
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

    // Deterministic 70/30 split by row index (identical to the synthetic arm):
    // fold[i] == 1.0 means TEST, 0.0 means TRAIN — handed verbatim to EBM.
    let fold: Vec<f64> = (0..n)
        .map(|i| if i % 10 >= 7 { 1.0 } else { 0.0 })
        .collect();
    let train_rows: Vec<usize> = (0..n).filter(|&i| fold[i] == 0.0).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| fold[i] == 1.0).collect();
    assert!(
        train_rows.len() > 700 && test_rows.len() > 250,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let test_numvisit: Vec<f64> = test_rows.iter().map(|&i| numvisit[i]).collect();

    // Build a training-only dataset by sub-setting the encoded rows; headers,
    // schema and column kinds are unchanged, so the formula resolves identically.
    let p_cols = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p_cols));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p_cols {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: numvisit ~ s(age, k=6) + badh, Poisson/log ------
    let cfg = FitConfig {
        family: Some("poisson".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("numvisit ~ s(age, k=6) + badh", &train_ds, &cfg)
        .expect("gam poisson fit on real train fold");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for Poisson numvisit ~ s(age, k=6) + badh");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at the held-out rows: rebuild the frozen design at the
    // test covariates; the log link => fitted mean μ = exp(design*beta).
    let n_test = test_rows.len();
    let mut test_grid = Array2::<f64>::zeros((n_test, p_cols));
    for (i, &row) in test_rows.iter().enumerate() {
        test_grid[[i, age_idx]] = age[row];
        test_grid[[i, badh_idx]] = badh[row];
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out points");
    let gam_test_eta: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();
    let gam_test_mu: Vec<f64> = gam_test_eta.iter().map(|e| e.exp()).collect();
    assert_eq!(gam_test_mu.len(), n_test, "gam test-mu length mismatch");
    assert!(
        gam_test_mu.iter().all(|v| v.is_finite() && *v > 0.0),
        "gam held-out mean must be finite and positive"
    );

    // ---- fit the SAME model on TRAIN with InterpretML EBM, predict TEST -----
    // ExplainableBoostingRegressor with the Poisson-deviance objective is the ML
    // analog of a PoissonGAM (additive shape functions, log link). We pass the
    // fold column so the Python side splits on the EXACT same rows; every Column
    // in this single call is full length (n), so lengths agree.
    let r = run_python(
        &[
            Column::new("age", &age),
            Column::new("badh", &badh),
            Column::new("numvisit", &numvisit),
            Column::new("fold", &fold),
        ],
        r#"
import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor

age   = np.asarray(df["age"],      dtype=float)
badh  = np.asarray(df["badh"],     dtype=float)
y     = np.asarray(df["numvisit"], dtype=float)
fold  = np.asarray(df["fold"],     dtype=float)

train = fold == 0.0
test  = fold == 1.0
X = np.column_stack([age, badh])

ebm = ExplainableBoostingRegressor(
    objective="poisson_deviance",
    interactions=0,
    random_state=20260530,
)
ebm.fit(X[train], y[train])
mu = np.asarray(ebm.predict(X[test]), dtype=float)
# Regression predictions are on the response (mean) scale; guard non-positive
# means before they reach a Poisson deviance.
mu = np.clip(mu, 1e-8, None)
emit("mu", mu)
"#,
    );
    let ebm_test_mu = r.vector("mu");
    assert_eq!(ebm_test_mu.len(), n_test, "EBM test-mu length mismatch");

    // ---- OBJECTIVE held-out count-deviance metric (computed in plain Rust) --
    // Mean Poisson unit deviance of each learner's fitted mean against the ACTUAL
    // held-out counts; lower is better. No ground truth is invoked.
    let n_test_f = n_test as f64;
    let gam_test_dev: f64 = test_numvisit
        .iter()
        .zip(&gam_test_mu)
        .map(|(&yi, &mu)| poisson_dev_unit(yi, mu))
        .sum::<f64>()
        / n_test_f;
    let ebm_test_dev: f64 = test_numvisit
        .iter()
        .zip(ebm_test_mu)
        .map(|(&yi, &mu)| poisson_dev_unit(yi, mu))
        .sum::<f64>()
        / n_test_f;

    // Intercept-only (constant-mean) Poisson predictor on the SAME test rows: the
    // train-mean count. Its held-out deviance is the no-covariate baseline that a
    // genuine learner must beat — a broken fit would not.
    let train_mean: f64 =
        train_rows.iter().map(|&i| numvisit[i]).sum::<f64>() / train_rows.len() as f64;
    let null_test_dev: f64 = test_numvisit
        .iter()
        .map(|&yi| poisson_dev_unit(yi, train_mean))
        .sum::<f64>()
        / n_test_f;

    let dev_ratio = gam_test_dev / ebm_test_dev.max(1e-12);

    eprintln!(
        "badhealth Poisson held-out: n_train={} n_test={n_test} gam_edf={gam_edf:.3} \
         gam_test_dev={gam_test_dev:.4} ebm_test_dev={ebm_test_dev:.4} (ratio={dev_ratio:.4}) \
         null_test_dev={null_test_dev:.4} train_mean={train_mean:.4}",
        train_rows.len()
    );

    // ---- PRIMARY objective assertion: gam predicts the held-out counts ------
    // gam's held-out mean Poisson deviance must beat the constant-mean baseline by
    // a clear margin — predicting age/badh structure genuinely reduces held-out
    // count deviance. A broken PIRLS loop or mis-inverted log link cannot.
    assert!(
        gam_test_dev < null_test_dev * 0.97,
        "gam did not beat the constant-mean Poisson baseline out of sample: \
         gam_test_dev={gam_test_dev:.4} vs null_test_dev={null_test_dev:.4} (bar 0.97x)"
    );

    // ---- BASELINE (match-or-beat): no worse than InterpretML EBM ------------
    // Both fit the identical train counts and are scored on the identical test
    // counts; gam must be at least as accurate as the mature ML additive learner
    // on held-out Poisson deviance, within a 10% margin.
    assert!(
        dev_ratio <= 1.10,
        "gam is materially less accurate than InterpretML EBM out of sample: \
         gam_test_dev={gam_test_dev:.4} ebm_test_dev={ebm_test_dev:.4} \
         (ratio={dev_ratio:.4}, bar 1.10)"
    );
}
