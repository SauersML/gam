//! End-to-end OBJECTIVE quality: gam's Binomial(logit) GAM must *generalize* —
//! it has to rank held-out cases well, not merely reproduce another tool's fit.
//!
//! OBJECTIVE METRIC ASSERTED (case 2, predictive accuracy on real data with no
//! known truth): a deterministic train/test split of the real `prostate.csv`
//! binary outcome. We fit `y ~ s(pc1,k=5)+s(pc2,k=5)` on the TRAIN rows only,
//! predict the held-out TEST rows, invert the logit link ourselves, and assert
//! gam's held-out **AUC** clears an absolute bar (`>= 0.70`) AND matches-or-beats
//! the best mature baseline on the SAME split (gam_test_auc >= best_ref - 0.02).
//! AUC on truly held-out cases is an honest generalization claim: a model that
//! merely memorized the training fit (or that another tool happened to also fit)
//! cannot game it. mgcv (penalized binomial GAM) and scikit-learn's
//! `LogisticRegression` are demoted to baselines-to-match-or-beat, NOT targets to
//! reproduce; their fitted values are printed for context only.
//!
//! Plus one GROUND-TRUTH correctness check that is NOT "same as a peer tool":
//! gam's 1-D *unpenalized* (penalized-zero) logistic regression on a single
//! linear term reduces *mathematically* to ordinary MLE logistic regression, the
//! exact convex objective scikit-learn (`penalty=None`) solves. Agreement there
//! is correctness vs the analytic MLE, so we keep it as a tight assertion.
//!
//! The link inversion (eta -> probability) is the single most common GLM bug, so
//! we apply `1/(1+exp(-eta))` ourselves on gam's own linear predictor. A genuine
//! shortfall here is a real bug, never a reason to loosen a bound.

use gam::data::EncodedDataset;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pad_to, relative_l2, run_python, run_r};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const PROSTATE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/prostate.csv");

/// Logistic (inverse-logit) link: eta -> probability.
fn inv_logit(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

/// Area under the ROC curve via the Mann-Whitney U statistic — the rank
/// agreement of a probability score against the binary truth.
fn auc(scores: &[f64], labels: &[f64]) -> f64 {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    // Average ranks (1-based) to handle ties correctly.
    let mut ranks = vec![0.0f64; scores.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let avg = ((i + j + 2) as f64) / 2.0; // mean of ranks (i+1..=j+1)
        for &k in &idx[i..=j] {
            ranks[k] = avg;
        }
        i = j + 1;
    }
    let n_pos: f64 = labels.iter().filter(|&&y| y > 0.5).count() as f64;
    let n_neg = labels.len() as f64 - n_pos;
    assert!(n_pos > 0.0 && n_neg > 0.0, "AUC needs both classes present");
    let sum_ranks_pos: f64 = ranks
        .iter()
        .zip(labels)
        .filter(|(_, y)| **y > 0.5)
        .map(|(r, _)| *r)
        .sum();
    (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

/// Build a row-subset of an `EncodedDataset` keeping the schema/encoding intact,
/// so the smaller dataset can be fed to `fit_from_formula` unchanged.
fn subset_rows(ds: &EncodedDataset, rows: &[usize]) -> EncodedDataset {
    let mut sub = ds.clone();
    let ncol = ds.values.ncols();
    let mut values = Array2::<f64>::zeros((rows.len(), ncol));
    for (new_r, &old_r) in rows.iter().enumerate() {
        for c in 0..ncol {
            values[[new_r, c]] = ds.values[[old_r, c]];
        }
    }
    sub.values = values;
    sub
}

#[test]
fn gam_binomial_logit_generalizes_on_heldout_prostate() {
    init_parallelism();

    // ---- load the real prostate binary outcome (y ~ pc1, pc2) -------------
    let ds = load_csvwith_inferred_schema(Path::new(PROSTATE_CSV)).expect("load prostate.csv");
    let col = ds.column_map();
    let pc1_idx = col["pc1"];
    let pc2_idx = col["pc2"];
    let y_idx = col["y"];
    let pc1: Vec<f64> = ds.values.column(pc1_idx).to_vec();
    let pc2: Vec<f64> = ds.values.column(pc2_idx).to_vec();
    let y: Vec<f64> = ds.values.column(y_idx).to_vec();
    let n = pc1.len();
    assert!(n > 600, "prostate should have ~654 rows, got {n}");
    for &yi in &y {
        assert!(yi == 0.0 || yi == 1.0, "y must be binary 0/1, saw {yi}");
    }

    // ---- deterministic train/test split (every 4th row -> test) -----------
    // Fixed, reproducible, no RNG: rows with index % 4 == 0 are held out for
    // evaluation; the rest train. ~25% held out is enough positives/negatives
    // for a stable held-out AUC on this dataset.
    let test_rows: Vec<usize> = (0..n).filter(|i| i % 4 == 0).collect();
    let train_rows: Vec<usize> = (0..n).filter(|i| i % 4 != 0).collect();
    let y_test: Vec<f64> = test_rows.iter().map(|&i| y[i]).collect();
    let pc1_train: Vec<f64> = train_rows.iter().map(|&i| pc1[i]).collect();
    let pc2_train: Vec<f64> = train_rows.iter().map(|&i| pc2[i]).collect();
    let y_train: Vec<f64> = train_rows.iter().map(|&i| y[i]).collect();
    let pc1_test: Vec<f64> = test_rows.iter().map(|&i| pc1[i]).collect();
    let pc2_test: Vec<f64> = test_rows.iter().map(|&i| pc2[i]).collect();
    {
        let n_pos = y_test.iter().filter(|&&v| v > 0.5).count();
        assert!(
            n_pos > 0 && n_pos < y_test.len(),
            "held-out test set must contain both classes (pos={n_pos}, n={})",
            y_test.len()
        );
    }

    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };

    // ---- (A) gam smooth binomial-logit fit on TRAIN, predict TEST ---------
    let ds_train = subset_rows(&ds, &train_rows);
    let result =
        fit_from_formula("y ~ s(pc1, k=5) + s(pc2, k=5)", &ds_train, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("binomial(logit) GLM with smooths should be a Standard fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Rebuild the design at the HELD-OUT test rows and apply beta -> eta, then
    // invert the logit link ourselves to get held-out fitted probabilities.
    let ntest = test_rows.len();
    let mut grid = Array2::<f64>::zeros((ntest, ds.headers.len()));
    for (r, &i) in test_rows.iter().enumerate() {
        grid[[r, pc1_idx]] = pc1[i];
        grid[[r, pc2_idx]] = pc2[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out test points");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let gam_prob: Vec<f64> = gam_eta.iter().map(|&e| inv_logit(e)).collect();
    assert_eq!(gam_prob.len(), ntest, "gam held-out probability length");

    // Every reference column is shipped at the common wire width `n` (the
    // longest natural length in this test — the full 654-row dataset used for
    // the 1-D MLE arm below). The ragged-column harness NaN-pads any shorter
    // column up to this width, so each reference body slices every column by its
    // own semantic length (`ntrain` train rows, `ntest` held-out rows, `nfull`
    // for the full-data arm) before use and never touches the NaN tail. Padding
    // the train/test columns to a fixed `n` (rather than to `train_rows.len()`)
    // is what keeps the 654-row full-data columns from being padded *down* to
    // the 490-row train width — the length mismatch that previously panicked.
    let ntrain = train_rows.len();

    // ---- (B) mgcv smooth baseline: fit TRAIN, predict TEST ----------------
    // Same penalized binomial GAM, trained on the identical TRAIN rows and
    // scored on the identical TEST rows. A baseline to match-or-beat on
    // held-out AUC, not a target to reproduce pointwise.
    let r = run_r(
        &[
            Column::new("pc1", &pad_to(&pc1_train, n)),
            Column::new("pc2", &pad_to(&pc2_train, n)),
            Column::new("y", &pad_to(&y_train, n)),
            Column::new("pc1_te", &pad_to(&pc1_test, n)),
            Column::new("pc2_te", &pad_to(&pc2_test, n)),
            Column::new("ntrain", &vec![ntrain as f64; n]),
            Column::new("ntest", &vec![ntest as f64; n]),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        ntrain <- as.integer(df$ntrain[1])
        ntest <- as.integer(df$ntest[1])
        tr <- data.frame(pc1 = df$pc1[1:ntrain], pc2 = df$pc2[1:ntrain],
                         y = df$y[1:ntrain])
        m <- gam(y ~ s(pc1, k=5) + s(pc2, k=5), data = tr,
                 family = binomial(link="logit"), method = "REML")
        newd <- data.frame(pc1 = df$pc1_te[1:ntest], pc2 = df$pc2_te[1:ntest])
        emit("prob", as.numeric(predict(m, newdata = newd, type = "response")))
        "#,
    );
    let mgcv_prob = r.vector("prob");
    assert_eq!(mgcv_prob.len(), ntest, "mgcv held-out probability length");

    // ---- (C) scikit-learn baseline: fit TRAIN, predict TEST ---------------
    // Unpenalized LogisticRegression on the SAME two TRAIN features, scored on
    // the SAME TEST rows. Linear-logit baseline to match-or-beat on AUC.
    // Also: a 1-D unpenalized logistic on pc1 over the FULL data, whose
    // coefficient is the analytic MLE ground truth for gam's linear fit below.
    let sk = run_python(
        &[
            Column::new("pc1", &pad_to(&pc1_train, n)),
            Column::new("pc2", &pad_to(&pc2_train, n)),
            Column::new("y", &pad_to(&y_train, n)),
            Column::new("pc1_te", &pad_to(&pc1_test, n)),
            Column::new("pc2_te", &pad_to(&pc2_test, n)),
            Column::new("ntrain", &vec![ntrain as f64; n]),
            Column::new("ntest", &vec![ntest as f64; n]),
            Column::new("pc1_full", &pc1),
            Column::new("y_full", &y),
            Column::new("nfull", &vec![n as f64; n]),
        ],
        r#"
from sklearn.linear_model import LogisticRegression
ntrain = int(np.asarray(df["ntrain"])[0])
ntest = int(np.asarray(df["ntest"])[0])
nfull = int(np.asarray(df["nfull"])[0])
Xtr = np.column_stack([np.asarray(df["pc1"], float)[:ntrain],
                       np.asarray(df["pc2"], float)[:ntrain]])
ytr = np.asarray(df["y"], float)[:ntrain]
Xte = np.column_stack([np.asarray(df["pc1_te"], float)[:ntest],
                       np.asarray(df["pc2_te"], float)[:ntest]])
clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
clf.fit(Xtr, ytr)
emit("prob", clf.predict_proba(Xte)[:, 1])

# 1-D unpenalized logistic on pc1 alone over the FULL data: MLE ground truth.
xf = np.asarray(df["pc1_full"], float)[:nfull].reshape(-1, 1)
yf = np.asarray(df["y_full"], float)[:nfull]
clf1 = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
clf1.fit(xf, yf)
emit("coef1", [float(clf1.coef_[0, 0]), float(clf1.intercept_[0])])
"#,
    );
    let sk_prob = sk.vector("prob");
    let sk_coef1 = sk.vector("coef1"); // [slope, intercept]
    assert_eq!(sk_prob.len(), ntest, "sklearn held-out probability length");

    // ---- (D) gam 1-D *linear* binomial-logit fit on FULL data: y ~ pc1 ----
    // No smooth: ordinary penalized-zero logistic regression on pc1, fit on the
    // SAME full data sklearn used. Its coefficient must match the analytic MLE.
    let lin = fit_from_formula("y ~ pc1", &ds, &cfg).expect("gam linear logit fit");
    let FitResult::Standard(linfit) = lin else {
        panic!("linear binomial(logit) should be a Standard fit");
    };
    // Locate the pc1 coefficient: rebuild the linear design and find the column
    // that responds to pc1 (unit change in pc1 -> change in eta == slope).
    let mut g0 = Array2::<f64>::zeros((2, ds.headers.len()));
    g0[[1, pc1_idx]] = 1.0; // row1 has pc1=1, row0 has pc1=0
    let lin_design = build_term_collection_design(g0.view(), &linfit.resolvedspec)
        .expect("rebuild linear design");
    let lin_eta: Vec<f64> = lin_design.design.apply(&linfit.fit.beta).to_vec();
    let gam_intercept = lin_eta[0];
    let gam_slope = lin_eta[1] - lin_eta[0];
    let sk_slope = sk_coef1[0];
    let sk_intercept = sk_coef1[1];

    // ---- objective metric: held-out AUC ----------------------------------
    let gam_auc = auc(&gam_prob, &y_test);
    let mgcv_auc = auc(mgcv_prob, &y_test);
    let sk_auc = auc(sk_prob, &y_test);
    let best_ref = mgcv_auc.max(sk_auc);
    // Printed for context only — NOT a pass criterion.
    let rel_mgcv = relative_l2(&gam_prob, mgcv_prob);
    let slope_rel = (gam_slope - sk_slope).abs() / sk_slope.abs().max(1e-6);
    let intercept_abs = (gam_intercept - sk_intercept).abs();

    eprintln!(
        "prostate binomial(logit) HELD-OUT (n_train={} n_test={ntest}): \
         gam_edf={gam_edf:.3} | held-out AUC gam={gam_auc:.4} mgcv={mgcv_auc:.4} \
         sklearn={sk_auc:.4} (best_ref={best_ref:.4}) | context: \
         prob_rel_l2(gam,mgcv)={rel_mgcv:.4} | 1D-MLE slope gam={gam_slope:.4} \
         sklearn={sk_slope:.4} (rel={slope_rel:.3}) intercept gam={gam_intercept:.4} \
         sklearn={sk_intercept:.4}",
        train_rows.len()
    );

    // (1) PRIMARY objective claim — gam generalizes: held-out AUC clears an
    // absolute bar. pc1/pc2 carry real signal for this outcome; a correct
    // penalized binomial GAM ranks held-out cases well above chance. 0.70 is a
    // principled floor (well above the 0.5 chance line) that a genuinely broken
    // link inversion or under/over-smoothed fit could not reach.
    assert!(
        gam_auc >= 0.70,
        "gam held-out AUC below objective bar: {gam_auc:.4} < 0.70"
    );

    // (2) Match-or-beat the best mature baseline on the SAME split. gam's
    // generalization must not trail mgcv or sklearn by more than 0.02 AUC.
    assert!(
        gam_auc >= best_ref - 0.02,
        "gam held-out AUC trails best baseline: gam={gam_auc:.4} \
         best_ref={best_ref:.4} (mgcv={mgcv_auc:.4} sklearn={sk_auc:.4})"
    );

    // (3) GROUND-TRUTH correctness (not peer-matching): gam's penalized-zero
    // logistic regression on a single linear term IS ordinary MLE logistic
    // regression — the exact convex objective sklearn(penalty=None) solves.
    // Slope within 2% relative, intercept within 0.02 absolute.
    assert!(
        slope_rel < 0.02,
        "1-D logit slope disagrees with the analytic MLE: gam={gam_slope:.5} \
         mle={sk_slope:.5} (rel={slope_rel:.4})"
    );
    assert!(
        intercept_abs < 0.02,
        "1-D logit intercept disagrees with the analytic MLE: gam={gam_intercept:.5} \
         mle={sk_intercept:.5}"
    );
}

/// Binary cross-entropy (log-loss) of probability scores against 0/1 labels,
/// averaged over the held-out cases. Probabilities are clamped away from {0,1}
/// so a single confident miss cannot send the metric to +inf — the standard
/// numerically-stable log-loss. Lower is better; the constant base rate gives
/// the entropy of the label distribution as a reference ceiling.
fn log_loss(probs: &[f64], labels: &[f64]) -> f64 {
    assert_eq!(probs.len(), labels.len(), "log_loss length mismatch");
    let eps = 1e-15;
    let s: f64 = probs
        .iter()
        .zip(labels)
        .map(|(&p, &y)| {
            let pc = p.clamp(eps, 1.0 - eps);
            -(y * pc.ln() + (1.0 - y) * (1.0 - pc).ln())
        })
        .sum();
    s / probs.len() as f64
}

/// SECOND real-data arm exercising the SAME gam capability — penalized
/// Binomial(logit) GAM smooths — on the real `prostate.csv` binary outcome,
/// but pinned to a DIFFERENT objective metric than the AUC arm above:
/// held-out **log-loss** (mean binary cross-entropy). Log-loss is a strictly
/// proper scoring rule, so it grades probability *calibration*, not just rank
/// order: a model that ranks held-out cases well (good AUC) but emits
/// over/under-confident probabilities (e.g. a broken link inversion or a
/// mis-scaled penalty) is penalized here even when its AUC looks fine. The
/// link inversion `1/(1+exp(-eta))` is applied by us on gam's own linear
/// predictor at the held-out rows.
///
/// Dataset SOURCE: `bench/datasets/prostate.csv` (pc1, pc2 -> binary y), the
/// same real prostate principal-component features used by the AUC arm.
///
/// Objective metric asserted (real data => truth unknown):
///   (1) absolute bar — gam held-out log-loss <= 0.62 (below the ~0.69 nats
///       entropy of a near-balanced coin, i.e. gam beats the no-skill base
///       rate by a real margin), and
///   (2) match-or-beat — gam log-loss <= best_ref + 0.02 against the best of
///       mgcv (penalized binomial GAM) and scikit-learn LogisticRegression on
///       the IDENTICAL train/test split. Lower is better, so the slack is on
///       the high side. The reference tools are baselines, never targets to
///       reproduce.
#[test]
fn gam_binomial_logit_generalizes_on_heldout_prostate_on_real_data() {
    init_parallelism();

    // ---- load the real prostate binary outcome (y ~ pc1, pc2) -------------
    let ds = load_csvwith_inferred_schema(Path::new(PROSTATE_CSV)).expect("load prostate.csv");
    let col = ds.column_map();
    let pc1_idx = col["pc1"];
    let pc2_idx = col["pc2"];
    let y_idx = col["y"];
    let pc1: Vec<f64> = ds.values.column(pc1_idx).to_vec();
    let pc2: Vec<f64> = ds.values.column(pc2_idx).to_vec();
    let y: Vec<f64> = ds.values.column(y_idx).to_vec();
    let n = pc1.len();
    assert!(n > 600, "prostate should have ~654 rows, got {n}");
    for &yi in &y {
        assert!(yi == 0.0 || yi == 1.0, "y must be binary 0/1, saw {yi}");
    }

    // ---- deterministic train/test split (every 4th row -> test) -----------
    // Identical fixed, RNG-free split as the AUC arm: rows with index % 4 == 0
    // are held out for evaluation, the rest train. The SAME train/test rows in
    // the SAME order are handed to gam and to both reference tools.
    let test_rows: Vec<usize> = (0..n).filter(|i| i % 4 == 0).collect();
    let train_rows: Vec<usize> = (0..n).filter(|i| i % 4 != 0).collect();
    let y_test: Vec<f64> = test_rows.iter().map(|&i| y[i]).collect();
    let pc1_train: Vec<f64> = train_rows.iter().map(|&i| pc1[i]).collect();
    let pc2_train: Vec<f64> = train_rows.iter().map(|&i| pc2[i]).collect();
    let y_train: Vec<f64> = train_rows.iter().map(|&i| y[i]).collect();
    let pc1_test: Vec<f64> = test_rows.iter().map(|&i| pc1[i]).collect();
    let pc2_test: Vec<f64> = test_rows.iter().map(|&i| pc2[i]).collect();
    let ntest = test_rows.len();
    {
        let n_pos = y_test.iter().filter(|&&v| v > 0.5).count();
        assert!(
            n_pos > 0 && n_pos < y_test.len(),
            "held-out test set must contain both classes (pos={n_pos}, n={ntest})"
        );
    }

    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };

    // ---- (A) gam smooth binomial-logit fit on TRAIN, predict TEST ---------
    let ds_train = subset_rows(&ds, &train_rows);
    let result =
        fit_from_formula("y ~ s(pc1, k=5) + s(pc2, k=5)", &ds_train, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("binomial(logit) GLM with smooths should be a Standard fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Rebuild the design at the HELD-OUT test rows, apply beta -> eta, then
    // invert the logit link ourselves to get held-out fitted probabilities.
    let mut grid = Array2::<f64>::zeros((ntest, ds.headers.len()));
    for (r, &i) in test_rows.iter().enumerate() {
        grid[[r, pc1_idx]] = pc1[i];
        grid[[r, pc2_idx]] = pc2[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at held-out test points");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let gam_prob: Vec<f64> = gam_eta.iter().map(|&e| inv_logit(e)).collect();
    assert_eq!(gam_prob.len(), ntest, "gam held-out probability length");

    // ---- (B) mgcv smooth baseline: fit TRAIN, predict TEST ----------------
    let r = run_r(
        &[
            Column::new("pc1", &pc1_train),
            Column::new("pc2", &pc2_train),
            Column::new("y", &y_train),
            Column::new("pc1_te", &pad_to(&pc1_test, train_rows.len())),
            Column::new("pc2_te", &pad_to(&pc2_test, train_rows.len())),
            Column::new("ntest", &vec![ntest as f64; train_rows.len()]),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(pc1, k=5) + s(pc2, k=5), data = df,
                 family = binomial(link="logit"), method = "REML")
        ntest <- as.integer(df$ntest[1])
        newd <- data.frame(pc1 = df$pc1_te[1:ntest], pc2 = df$pc2_te[1:ntest])
        emit("prob", as.numeric(predict(m, newdata = newd, type = "response")))
        "#,
    );
    let mgcv_prob = r.vector("prob");
    assert_eq!(mgcv_prob.len(), ntest, "mgcv held-out probability length");

    // ---- (C) scikit-learn baseline: fit TRAIN, predict TEST ---------------
    let sk = run_python(
        &[
            Column::new("pc1", &pc1_train),
            Column::new("pc2", &pc2_train),
            Column::new("y", &y_train),
            Column::new("pc1_te", &pad_to(&pc1_test, train_rows.len())),
            Column::new("pc2_te", &pad_to(&pc2_test, train_rows.len())),
            Column::new("ntest", &vec![ntest as f64; train_rows.len()]),
        ],
        r#"
from sklearn.linear_model import LogisticRegression
ntest = int(np.asarray(df["ntest"])[0])
Xtr = np.column_stack([np.asarray(df["pc1"], float), np.asarray(df["pc2"], float)])
ytr = np.asarray(df["y"], float)
Xte = np.column_stack([np.asarray(df["pc1_te"], float)[:ntest],
                       np.asarray(df["pc2_te"], float)[:ntest]])
clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
clf.fit(Xtr, ytr)
emit("prob", clf.predict_proba(Xte)[:, 1])
"#,
    );
    let sk_prob = sk.vector("prob");
    assert_eq!(sk_prob.len(), ntest, "sklearn held-out probability length");

    // ---- objective metric: held-out log-loss (mean binary cross-entropy) --
    let gam_ll = log_loss(&gam_prob, &y_test);
    let mgcv_ll = log_loss(mgcv_prob, &y_test);
    let sk_ll = log_loss(sk_prob, &y_test);
    let best_ref = mgcv_ll.min(sk_ll); // lower log-loss is better
    // Context only — NOT a pass criterion.
    let rel_mgcv = relative_l2(&gam_prob, mgcv_prob);
    let gam_auc_ctx = auc(&gam_prob, &y_test);

    eprintln!(
        "prostate binomial(logit) HELD-OUT log-loss (n_train={} n_test={ntest}): \
         gam_edf={gam_edf:.3} | held-out log-loss gam={gam_ll:.4} mgcv={mgcv_ll:.4} \
         sklearn={sk_ll:.4} (best_ref={best_ref:.4}) | context: AUC(gam)={gam_auc_ctx:.4} \
         prob_rel_l2(gam,mgcv)={rel_mgcv:.4}",
        train_rows.len()
    );

    // (1) PRIMARY objective claim — gam's held-out probabilities are well
    // calibrated: mean cross-entropy clears an absolute ceiling well below the
    // ~0.69 nats no-skill base-rate entropy of this near-balanced outcome.
    assert!(
        gam_ll <= 0.62,
        "gam held-out log-loss above objective bar: {gam_ll:.4} > 0.62"
    );

    // (2) Match-or-beat the best mature baseline on the SAME split. Lower is
    // better, so gam's log-loss may exceed the best reference by at most 0.02.
    assert!(
        gam_ll <= best_ref + 0.02,
        "gam held-out log-loss trails best baseline: gam={gam_ll:.4} \
         best_ref={best_ref:.4} (mgcv={mgcv_ll:.4} sklearn={sk_ll:.4})"
    );
}
