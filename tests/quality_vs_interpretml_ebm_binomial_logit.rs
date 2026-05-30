//! End-to-end quality: gam's penalized Binomial(logit) GAM must reach the same
//! predictive operating point as InterpretML's `ExplainableBoostingClassifier`
//! (EBM) — the de-facto ML-world interpretable alternative to a statistical GAM.
//!
//! Why benchmark against InterpretML EBM specifically? EBM is the most widely
//! deployed "glass-box" additive model outside the statistics community. Both
//! gam and EBM are *additive* models that *penalize complexity*, but they reach
//! their fits by completely different machinery: EBM uses cyclic gradient
//! boosting of shallow per-feature trees with interaction detection and
//! bagging/rebundling, whereas gam selects smoothing parameters by REML on a
//! penalized thin-plate basis. Agreement in predictive *rank* (AUC) and in
//! *aggregate loss* (binomial deviance, Brier) — without ever aligning the
//! per-feature shape functions — validates gam's binomial PIRLS loop and logit
//! link inversion against a totally independent additive learner. If two such
//! different additive machines land at the same operating point on the same
//! data, the operating point is real, not an artifact of either method.
//!
//! Deviance here is `2 * sum( y*log(y/p) + (1-y)*log((1-y)/p) )` evaluated at
//! the fitted probabilities `p`; with binary `y in {0,1}` the saturated
//! log-likelihood is 0, so deviance reduces to `-2 * sum( y*log(p) +
//! (1-y)*log(1-p) )` — twice the negative log-likelihood on the exact same
//! objective both engines implicitly optimize. We compare the *per-row deviance
//! contributions* (relative L2) so a divergence anywhere along the curve, not
//! just in the total, is caught.
//!
//! Identical data (real `prostate.csv`, y ~ pc1 + pc2) feeds both engines; gam
//! fits `y ~ s(pc1,k=5) + s(pc2,k=5)` with family=binomial, link=logit, REML.
//! gam's design*beta gives eta; we invert the logit link ourselves. A genuine
//! divergence is a real bug — bounds are NOT to be loosened.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, relative_l2, run_python};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const PROSTATE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/prostate.csv");

/// Logistic (inverse-logit) link: eta -> probability.
fn inv_logit(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

/// Clamp a probability away from {0,1} so log-loss / deviance stays finite.
/// 1e-12 is far below any meaningful predictive resolution on n~654 and is the
/// standard guard used by sklearn's log_loss; it cannot rescue a genuinely
/// mis-calibrated fit.
fn clamp_prob(p: f64) -> f64 {
    p.clamp(1e-12, 1.0 - 1e-12)
}

/// Per-row binomial deviance contribution for binary y: `-2*(y*log(p) +
/// (1-y)*log(1-p))`. (Saturated log-lik is 0 for y in {0,1}, so this is the
/// full deviance residual-squared term.)
fn deviance_terms(prob: &[f64], y: &[f64]) -> Vec<f64> {
    prob.iter()
        .zip(y)
        .map(|(&p, &yi)| {
            let p = clamp_prob(p);
            -2.0 * (yi * p.ln() + (1.0 - yi) * (1.0 - p).ln())
        })
        .collect()
}

/// Brier score: mean squared error on the probability scale.
fn brier(prob: &[f64], y: &[f64]) -> f64 {
    let s: f64 = prob.iter().zip(y).map(|(&p, &yi)| (p - yi) * (p - yi)).sum();
    s / prob.len().max(1) as f64
}

/// Area under the ROC curve via the Mann-Whitney U statistic — tie-aware via
/// average ranks. Pure rank agreement of a probability score against truth.
fn auc(scores: &[f64], labels: &[f64]) -> f64 {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[a].partial_cmp(&scores[b]).unwrap());
    let mut ranks = vec![0.0f64; scores.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let avg = ((i + j + 2) as f64) / 2.0; // mean of 1-based ranks i+1..=j+1
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

#[test]
fn gam_binomial_logit_matches_interpretml_ebm() {
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

    // ---- gam smooth binomial-logit fit: y ~ s(pc1,k=5)+s(pc2,k=5) ---------
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

    // Rebuild the design at the training rows, apply beta -> eta, invert logit.
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

    // ---- InterpretML EBM reference: SAME data, additive classifier --------
    // ExplainableBoostingClassifier is the ML-world interpretable additive
    // model. We fit it on the identical (pc1, pc2) features and read its
    // class-1 probabilities at the training points. interactions=0 keeps it a
    // purely additive model (the closest analog to a 2-term GAM); a fixed
    // random_state makes the bagged boosting deterministic.
    let py = run_python(
        &[
            Column::new("pc1", &pc1),
            Column::new("pc2", &pc2),
            Column::new("y", &y),
        ],
        r#"
from interpret.glassbox import ExplainableBoostingClassifier
X = np.column_stack([np.asarray(df["pc1"], float), np.asarray(df["pc2"], float)])
yv = np.asarray(df["y"], float).astype(int)
ebm = ExplainableBoostingClassifier(interactions=0, random_state=0)
ebm.fit(X, yv)
emit("prob", ebm.predict_proba(X)[:, 1])
"#,
    );
    let ebm_prob = py.vector("prob");
    assert_eq!(ebm_prob.len(), n, "EBM probability length mismatch");

    // ---- compare ----------------------------------------------------------
    let gam_dev = deviance_terms(&gam_prob, &y);
    let ebm_dev = deviance_terms(ebm_prob, &y);
    let dev_rel = relative_l2(&gam_dev, &ebm_dev);

    let gam_auc = auc(&gam_prob, &y);
    let ebm_auc = auc(ebm_prob, &y);
    let auc_diff = (gam_auc - ebm_auc).abs();

    let gam_brier = brier(&gam_prob, &y);
    let ebm_brier = brier(ebm_prob, &y);
    let brier_diff = (gam_brier - ebm_brier).abs();

    let gam_dev_total: f64 = gam_dev.iter().sum();
    let ebm_dev_total: f64 = ebm_dev.iter().sum();

    eprintln!(
        "prostate EBM vs gam binomial(logit): n={n} gam_edf={gam_edf:.3} | \
         AUC gam={gam_auc:.4} ebm={ebm_auc:.4} (Δ={auc_diff:.4}) | \
         deviance total gam={gam_dev_total:.2} ebm={ebm_dev_total:.2} (rel_l2={dev_rel:.4}) | \
         Brier gam={gam_brier:.4} ebm={ebm_brier:.4} (Δ={brier_diff:.4})"
    );

    // (1) AUC (rank) agreement: two independent additive learners on the same
    // additive structure must order the binary outcome essentially identically.
    // Δ < 0.015 is tight enough to catch a ranking regression while tolerating
    // EBM's boosting-vs-REML stylistic differences in the fitted shape.
    assert!(
        auc_diff < 0.015,
        "gam vs InterpretML EBM AUC disagree: gam={gam_auc:.4} ebm={ebm_auc:.4} (Δ={auc_diff:.4})"
    );

    // (2) Per-row binomial deviance (logit-scale loss) agreement: relative L2
    // of the per-observation deviance contributions. Both engines penalize
    // complexity on the same -2*loglik objective, so the loss profile must
    // track closely. rel_l2 < 0.08 is a principled bound — loose enough for the
    // genuinely different penalization criteria (REML vs boosting early-stop),
    // tight enough that a mis-calibrated link or a broken PIRLS loop fails it.
    assert!(
        dev_rel < 0.08,
        "gam vs EBM per-row deviance diverges: rel_l2={dev_rel:.4} \
         (gam_total={gam_dev_total:.2} ebm_total={ebm_dev_total:.2})"
    );

    // (3) Brier score (probability-scale calibration) agreement: |Δ| < 0.01.
    // Brier directly measures squared probability error; a 0.01 gap is roughly
    // a 0.1 systematic probability shift on a fraction of rows — clearly
    // detectable mis-calibration — yet tolerant of the two methods' distinct
    // smoothing. A correct link inversion lands well inside this.
    assert!(
        brier_diff < 0.01,
        "gam vs EBM Brier score disagree: gam={gam_brier:.4} ebm={ebm_brier:.4} (Δ={brier_diff:.4})"
    );
}
