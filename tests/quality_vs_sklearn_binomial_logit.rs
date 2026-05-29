//! End-to-end quality: gam's Binomial(logit) GLM with smooth terms must agree
//! with scikit-learn's `LogisticRegression` — the de-facto standard binary
//! classifier — on the *probability scale*, and with `mgcv::gam(binomial)` on
//! the smooth fitted probabilities.
//!
//! Why this benchmark matters: the logit link is the most widely used binomial
//! link, and the single most common source of bugs in a GLM implementation is
//! the link inversion (eta -> probability). A correct penalized binomial fit
//! must (a) recover the same fitted probabilities as a mature smoother (mgcv),
//! (b) rank cases the same way a trusted classifier does (AUC agreement), and
//! (c) on a 1-D *linear* model reproduce the coefficient sklearn's unpenalized
//! logistic regression finds. We test all three on the real `prostate.csv`
//! binary outcome (y ~ pc1, pc2), feeding *identical* data to every engine.
//!
//! gam's design*beta yields the linear predictor eta; we apply the logistic
//! inverse `1/(1+exp(-eta))` ourselves, which is exactly the link inversion we
//! want to stress. A genuine divergence here is a real bug, not a reason to
//! loosen the bounds.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, run_python, run_r};
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
        .filter(|(_, &y)| y > 0.5)
        .map(|(r, _)| *r)
        .sum();
    (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

#[test]
fn gam_binomial_logit_matches_sklearn_and_mgcv() {
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

    // ---- (A) gam smooth binomial-logit fit: y ~ s(pc1,k=5)+s(pc2,k=5) -----
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(pc1, k=5) + s(pc2, k=5)", &ds, &cfg).expect("gam fit");
    let FitResult::Standard(fit) = result else {
        panic!("binomial(logit) GLM with smooths should be a Standard fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // Rebuild the design at the training rows and apply beta -> eta, then
    // invert the logit link ourselves to get fitted probabilities.
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for i in 0..n {
        grid[[i, pc1_idx]] = pc1[i];
        grid[[i, pc2_idx]] = pc2[i];
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild design at training points");
    let gam_eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let gam_prob: Vec<f64> = gam_eta.iter().map(|&e| inv_logit(e)).collect();
    assert_eq!(gam_prob.len(), n, "gam fitted-probability length mismatch");

    // ---- (B) mgcv smooth reference: SAME smooth binomial model ------------
    // mgcv::gam(y ~ s(pc1,k=5)+s(pc2,k=5), binomial) is the mature penalized
    // GAM; fitted(m) is on the probability scale (type="response" default).
    let r = run_r(
        &[
            Column::new("pc1", &pc1),
            Column::new("pc2", &pc2),
            Column::new("y", &y),
        ],
        r#"
        suppressPackageStartupMessages(library(mgcv))
        m <- gam(y ~ s(pc1, k=5) + s(pc2, k=5), data = df,
                 family = binomial(link="logit"), method = "REML")
        emit("prob", as.numeric(fitted(m)))
        emit("edf", sum(m$edf))
        "#,
    );
    let mgcv_prob = r.vector("prob");
    let mgcv_edf = r.scalar("edf");
    assert_eq!(mgcv_prob.len(), n, "mgcv fitted-probability length mismatch");

    // ---- (C) scikit-learn classifier reference ----------------------------
    // Unpenalized LogisticRegression (penalty=None, lbfgs) on the SAME two
    // features gives a trusted linear-logit probability score; we use it only
    // for AUC (rank) agreement against the smooth gam fit, since the linear
    // sklearn model and the penalized smooth need not coincide pointwise but
    // must rank the same binary outcome essentially identically here.
    let sk = run_python(
        &[
            Column::new("pc1", &pc1),
            Column::new("pc2", &pc2),
            Column::new("y", &y),
        ],
        r#"
from sklearn.linear_model import LogisticRegression
X = np.column_stack([np.asarray(df["pc1"], float), np.asarray(df["pc2"], float)])
yv = np.asarray(df["y"], float)
clf = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
clf.fit(X, yv)
emit("prob", clf.predict_proba(X)[:, 1])

# 1-D unpenalized logistic on pc1 alone: coefficient ground truth.
clf1 = LogisticRegression(penalty=None, solver="lbfgs", max_iter=10000)
clf1.fit(X[:, [0]], yv)
emit("coef1", [float(clf1.coef_[0, 0]), float(clf1.intercept_[0])])
"#,
    );
    let sk_prob = sk.vector("prob");
    let sk_coef1 = sk.vector("coef1"); // [slope, intercept]
    assert_eq!(sk_prob.len(), n, "sklearn probability length mismatch");

    // ---- (D) gam 1-D *linear* binomial-logit fit: y ~ pc1 -----------------
    // No smooth: an ordinary penalized-zero logistic regression on pc1. Its
    // coefficient must match sklearn's unpenalized logistic-regression slope.
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

    // ---- compare ----------------------------------------------------------
    let corr_mgcv = pearson(&gam_prob, mgcv_prob);
    let gam_auc = auc(&gam_prob, &y);
    let mgcv_auc = auc(mgcv_prob, &y);
    let sk_auc = auc(sk_prob, &y);
    let edf_rel = (gam_edf - mgcv_edf).abs() / mgcv_edf.abs().max(1.0);
    let slope_rel = (gam_slope - sk_slope).abs() / sk_slope.abs().max(1e-6);
    let intercept_abs = (gam_intercept - sk_intercept).abs();

    eprintln!(
        "prostate binomial(logit): n={n} gam_edf={gam_edf:.3} mgcv_edf={mgcv_edf:.3} \
         (edf_rel={edf_rel:.3}) prob_pearson(gam,mgcv)={corr_mgcv:.5} \
         AUC gam={gam_auc:.4} mgcv={mgcv_auc:.4} sklearn={sk_auc:.4} | \
         1D slope gam={gam_slope:.4} sklearn={sk_slope:.4} (rel={slope_rel:.3}) \
         intercept gam={gam_intercept:.4} sklearn={sk_intercept:.4}"
    );

    // (1) Probability-scale agreement with mgcv: both REML-fit the identical
    // penalized binomial smooth, so post-logit fitted probabilities must be
    // nearly collinear. gam vs mgcv on this data tracks at pearson ~0.999;
    // 0.99 is a tight bound that still tolerates basis/null-space convention
    // differences while catching any link-inversion or smoother bug.
    assert!(
        corr_mgcv > 0.99,
        "gam vs mgcv fitted probabilities diverge: pearson={corr_mgcv:.5}"
    );

    // (2) AUC (rank) agreement: a correct classifier must order cases like the
    // mature references. The smooth gam fit and the trusted classifiers should
    // differ by <0.01 in AUC (the spec bound).
    assert!(
        (gam_auc - mgcv_auc).abs() < 0.01,
        "gam vs mgcv AUC disagree: gam={gam_auc:.4} mgcv={mgcv_auc:.4}"
    );
    assert!(
        (gam_auc - sk_auc).abs() < 0.01,
        "gam vs sklearn AUC disagree: gam={gam_auc:.4} sklearn={sk_auc:.4}"
    );

    // (3) EDF same-ballpark complexity (basis-convention sensitive): within 30%.
    assert!(
        edf_rel < 0.30,
        "effective degrees of freedom disagree: gam={gam_edf:.3} mgcv={mgcv_edf:.3} (rel={edf_rel:.3})"
    );

    // (4) 1-D linear coefficient: gam's penalized-zero logistic regression on
    // a single linear term reduces to ordinary MLE logistic regression, which
    // is exactly what sklearn (penalty=None) computes. Slope must match within
    // 2% relative and intercept within 0.02 absolute — tight, since both solve
    // the same convex log-likelihood.
    assert!(
        slope_rel < 0.02,
        "1-D logit slope disagrees with sklearn: gam={gam_slope:.5} sklearn={sk_slope:.5} (rel={slope_rel:.4})"
    );
    assert!(
        intercept_abs < 0.02,
        "1-D logit intercept disagrees with sklearn: gam={gam_intercept:.5} sklearn={sk_intercept:.5}"
    );
}
