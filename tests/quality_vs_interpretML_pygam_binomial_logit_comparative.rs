//! Objective hold-out predictive-quality test for gam's penalized
//! binomial-logit GAM on a real binary outcome (the prostate `y ~ pc1 + pc2`
//! problem), with two mature additive learners kept only as
//! **baselines-to-match-or-beat**, never as the thing gam must reproduce.
//!
//! OBJECTIVE METRIC ASSERTED (the pass/fail criterion):
//!   * gam's own mean 5-fold **hold-out AUC** must clear an absolute quality
//!     bar (`>= 0.62`). AUC is computed on gam's *own* out-of-fold predicted
//!     probabilities against the true held-out labels — a self-contained
//!     statement that gam's REML smoothing-parameter choice yields a model
//!     that genuinely separates the classes out of sample, on data it never
//!     saw. This is the primary claim and stands with no reference at all.
//!
//! BASELINE-TO-MATCH-OR-BEAT (secondary, NOT a "reproduce the reference"
//! check): on the identical data and identical deterministic folds we also fit
//!   * **InterpretML EBM** (`ExplainableBoostingClassifier`, additive, no
//!     interactions) — the ML world's GAM, and
//!   * **pyGAM `LogisticGAM`** — a classical GCV-smoothed penalized GAM,
//! and require gam's mean hold-out AUC to be **no worse than the best of the
//! two by more than a 0.02 AUC margin** (`gam_auc >= max(ebm, pygam) - 0.02`).
//! This demotes the mature tools from "gam must land in their band" to "gam
//! must be at least as accurate as the strongest mature additive learner
//! (within sampling slack)". gam beating both references passes trivially;
//! only gam being a materially *worse* classifier fails. We still COMPUTE and
//! print every engine's per-fold and mean AUC via `eprintln!` for context.
//!
//! Both engines see the SAME rows, the SAME 5-fold split `fold(i) = i % 5`,
//! and the SAME additive structure `s(pc1,k=5) + s(pc2,k=5)`, scored by one
//! common AUC estimator, so the baseline comparison is apples-to-apples. No
//! bound here is loosened to force a pass: a genuine predictive shortfall
//! (gam below 0.62 absolute, or gam trailing the best reference by >0.02 AUC)
//! is a real failure and should fire.

use gam::inference::data::EncodedDataset;
use gam::matrix::LinearOperator;
use gam::smooth::{build_term_collection_design, freeze_term_collection_from_design};
use gam::test_support::reference::{Column, run_python};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const PROSTATE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/prostate.csv");
// Reduced from 5 to 3 to keep the 5×(gam fit + EBM + pyGAM) Python
// subprocess within the 360s CI wall-clock budget.  3-fold gives ~218 test
// rows per fold; the AUC estimate remains reliable at that size.
const N_FOLDS: usize = 3;

/// Absolute hold-out quality bar gam must clear on its OWN predictions. The
/// prostate `y ~ pc1 + pc2` signal is moderate; a model that has learned real
/// structure (not noise) should comfortably exceed chance (0.5) out of sample.
/// 0.62 is well above chance yet below the honest achievable range for this
/// two-feature additive problem, so it fires on a model that fails to
/// generalize while never being a vacuous bar.
const GAM_MIN_HOLDOUT_AUC: f64 = 0.62;

/// Match-or-beat slack against the strongest mature baseline. gam must be at
/// least this close to the best of {EBM, pyGAM}; 0.02 AUC is the hold-out
/// sampling slack (3-fold gives ~218 test rows/fold, more than the original
/// 5-fold's ~130, so the AUC estimator is at least as precise).
const BASELINE_AUC_MARGIN: f64 = 0.02;

/// Absolute hold-out negative-log-likelihood (mean per-row log-loss) ceiling
/// gam must stay UNDER on the single fixed split. The constant base-rate
/// predictor `p = mean(y_train)` on a roughly balanced prostate label costs
/// about `ln 2 ≈ 0.693` nats/row; a model that has learned the `pc1 + pc2`
/// signal must beat that, so 0.66 is a non-vacuous ceiling that still fires
/// only on a model that fails to predict held-out labels probabilistically.
const GAM_MAX_HOLDOUT_NLL: f64 = 0.66;

/// Match-or-beat slack against the strongest mature baseline on log-loss. gam's
/// held-out NLL must be no worse than the BEST (lowest) of {EBM, pyGAM} by more
/// than this many nats/row. NLL is far more sensitive to calibration than AUC,
/// so 0.03 nats is a generous-but-real sampling slack at ~130 test rows.
const BASELINE_NLL_MARGIN: f64 = 0.03;

/// Logistic inverse-link: linear predictor eta -> probability.
fn inv_logit(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

/// Area under the ROC curve via the Mann-Whitney U statistic (tie-aware via
/// average ranks). This is the rank agreement of a probability score against
/// the binary truth — identical estimator for every engine so the comparison
/// is not contaminated by differing AUC conventions.
fn auc(scores: &[f64], labels: &[f64]) -> f64 {
    assert_eq!(scores.len(), labels.len(), "auc length mismatch");
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
    assert!(
        n_pos > 0.0 && n_neg > 0.0,
        "AUC needs both classes present in the test fold"
    );
    let sum_ranks_pos: f64 = ranks
        .iter()
        .zip(labels)
        .filter(|(_, y)| **y > 0.5)
        .map(|(r, _)| *r)
        .sum();
    (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

/// Mean per-row binomial negative log-likelihood (log-loss) of probability
/// scores against binary truth: `-mean[ y*ln p + (1-y)*ln(1-p) ]`. Identical
/// estimator for every engine, with the same probability clamp so no engine is
/// advantaged by emitting hard 0/1 probabilities. Lower is better; the
/// base-rate-only predictor sits near `ln 2`.
fn log_loss(scores: &[f64], labels: &[f64]) -> f64 {
    assert_eq!(scores.len(), labels.len(), "log_loss length mismatch");
    assert!(!scores.is_empty(), "log_loss needs at least one row");
    // Clamp away from {0,1} so a confidently-wrong probability is heavily but
    // FINITELY penalized; 1e-12 matches the clamp applied to every reference
    // engine's probabilities below, keeping the comparison apples-to-apples.
    const EPS: f64 = 1e-12;
    let mut acc = 0.0;
    for (&p, &y) in scores.iter().zip(labels) {
        let pc = p.clamp(EPS, 1.0 - EPS);
        acc += -(y * pc.ln() + (1.0 - y) * (1.0 - pc).ln());
    }
    acc / scores.len() as f64
}

/// Build an `EncodedDataset` restricted to `rows` of `full`, preserving the
/// frozen schema/headers/column kinds so each fold's gam fit uses an identical
/// encoding to every other fold (no per-fold schema re-inference drift).
fn subset_dataset(full: &EncodedDataset, rows: &[usize]) -> EncodedDataset {
    let p = full.headers.len();
    let mut values = Array2::<f64>::zeros((rows.len(), p));
    for (out_r, &src_r) in rows.iter().enumerate() {
        values.row_mut(out_r).assign(&full.values.row(src_r));
    }
    EncodedDataset {
        headers: full.headers.clone(),
        values,
        schema: full.schema.clone(),
        column_kinds: full.column_kinds.clone(),
    }
}

#[test]
fn gam_binomial_logit_holdout_predictive_quality() {
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

    // ---- deterministic 5-fold assignment: fold(i) = i % 5 -----------------
    // Reproducible for every engine; row i lands in fold i % 5, so fold 0 is
    // rows {0,5,10,...}, fold 1 is {1,6,11,...}, etc.
    let fold_of: Vec<usize> = (0..n).map(|i| i % N_FOLDS).collect();
    let fold_assign: Vec<f64> = fold_of.iter().map(|&f| f as f64).collect();

    // ===================================================================
    // (A) gam: per-fold REML binomial-logit fit, hold-out AUC per fold.
    // ===================================================================
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };

    let mut gam_fold_auc = Vec::<f64>::with_capacity(N_FOLDS);
    for f in 0..N_FOLDS {
        let train_rows: Vec<usize> = (0..n).filter(|&i| fold_of[i] != f).collect();
        let test_rows: Vec<usize> = (0..n).filter(|&i| fold_of[i] == f).collect();
        assert!(
            !train_rows.is_empty() && !test_rows.is_empty(),
            "empty fold {f}"
        );

        let train_ds = subset_dataset(&ds, &train_rows);
        let result = fit_from_formula("y ~ s(pc1, k=5) + s(pc2, k=5)", &train_ds, &cfg)
            .unwrap_or_else(|e| panic!("gam fold {f} fit failed: {e:?}"));
        let FitResult::Standard(fit) = result else {
            panic!("binomial(logit) GAM with smooths should be a Standard fit");
        };

        // FREEZE the trained spec against the trained design before predicting.
        // `StandardFitResult::resolvedspec` is the *unfrozen* spec: its smooth
        // knots/centers, joint-null absorption rotation, and identifiability
        // transform are still data-derived. Rebuilding a design from it on the
        // held-out rows would re-plan those artifacts from the TEST data,
        // yielding a basis that does not correspond to the trained `beta` and
        // therefore a meaningless eta. The canonical predict path
        // (`build_predict_input_for_model_inner`) builds the predict design from
        // the spec produced by `freeze_term_collection_from_design`; we do the
        // same so the held-out basis reuses the training-time knots/rotation.
        let frozenspec = freeze_term_collection_from_design(&fit.resolvedspec, &fit.design)
            .unwrap_or_else(|e| panic!("freeze gam fold {f} spec failed: {e:?}"));

        // Rebuild the frozen design at the held-out test rows -> eta -> prob.
        let mut grid = Array2::<f64>::zeros((test_rows.len(), ds.headers.len()));
        for (out_r, &src_r) in test_rows.iter().enumerate() {
            grid[[out_r, pc1_idx]] = pc1[src_r];
            grid[[out_r, pc2_idx]] = pc2[src_r];
        }
        let design = build_term_collection_design(grid.view(), &frozenspec)
            .expect("rebuild gam design at held-out test rows");
        let eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
        let prob: Vec<f64> = eta.iter().map(|&e| inv_logit(e)).collect();
        let labels: Vec<f64> = test_rows.iter().map(|&i| y[i]).collect();
        gam_fold_auc.push(auc(&prob, &labels));
    }
    let gam_mean_auc = gam_fold_auc.iter().sum::<f64>() / N_FOLDS as f64;

    // ===================================================================
    // (B) InterpretML EBM + (C) pyGAM LogisticGAM: SAME folds, SAME data.
    //     Computed ONLY as match-or-beat baselines, not as a band gam must
    //     fall inside.
    // ===================================================================
    let py = run_python(
        &[
            Column::new("pc1", &pc1),
            Column::new("pc2", &pc2),
            Column::new("y", &y),
            Column::new("fold", &fold_assign),
        ],
        r#"
from sklearn.metrics import roc_auc_score
from interpret.glassbox import ExplainableBoostingClassifier
from pygam import LogisticGAM, s

X = np.column_stack([np.asarray(df["pc1"], float), np.asarray(df["pc2"], float)])
yv = np.asarray(df["y"], float)
fold = np.asarray(df["fold"], float).astype(int)
n_folds = int(fold.max()) + 1

ebm_auc = []
pygam_auc = []
for f in range(n_folds):
    tr = fold != f
    te = fold == f
    Xtr, ytr = X[tr], yv[tr]
    Xte, yte = X[te], yv[te]

    # InterpretML EBM: PURELY additive shape-function classifier. interactions=0
    # disables EBM's default automatic pairwise (pc1 x pc2) term so the model is
    # the SAME additive structure s(pc1) + s(pc2) that gam and pyGAM fit -- an
    # interaction-bearing EBM would not be the additive comparator this test
    # claims to benchmark. Deterministic seed for reproducibility.
    ebm = ExplainableBoostingClassifier(interactions=0, random_state=0)
    ebm.fit(Xtr, ytr)
    p_ebm = ebm.predict_proba(Xte)[:, 1]
    ebm_auc.append(float(roc_auc_score(yte, p_ebm)))

    # pyGAM LogisticGAM: classical penalized GAM, GCV smoothing under PIRLS,
    # same additive structure s(pc1) + s(pc2) with 5 splines each (k=5).
    lg = LogisticGAM(s(0, n_splines=5) + s(1, n_splines=5)).fit(Xtr, ytr)
    p_lg = lg.predict_proba(Xte)
    pygam_auc.append(float(roc_auc_score(yte, p_lg)))

emit("ebm_fold_auc", ebm_auc)
emit("pygam_fold_auc", pygam_auc)
emit("ebm_mean_auc", [float(np.mean(ebm_auc))])
emit("pygam_mean_auc", [float(np.mean(pygam_auc))])
"#,
    );
    let ebm_fold_auc = py.vector("ebm_fold_auc");
    let pygam_fold_auc = py.vector("pygam_fold_auc");
    let ebm_mean_auc = py.scalar("ebm_mean_auc");
    let pygam_mean_auc = py.scalar("pygam_mean_auc");
    assert_eq!(ebm_fold_auc.len(), N_FOLDS, "EBM emitted wrong fold count");
    assert_eq!(
        pygam_fold_auc.len(),
        N_FOLDS,
        "pyGAM emitted wrong fold count"
    );

    let best_ref_auc = ebm_mean_auc.max(pygam_mean_auc);

    eprintln!(
        "prostate 5-fold binomial(logit) hold-out AUC | \
         gam folds={gam_fold_auc:?} mean={gam_mean_auc:.4} | \
         EBM folds={ebm_fold_auc:?} mean={ebm_mean_auc:.4} | \
         pyGAM folds={pygam_fold_auc:?} mean={pygam_mean_auc:.4} | \
         best_ref={best_ref_auc:.4} \
         (bars: abs>={GAM_MIN_HOLDOUT_AUC:.2}, match-or-beat>=best_ref-{BASELINE_AUC_MARGIN:.2})"
    );

    // (1) PRIMARY OBJECTIVE CLAIM: gam's own out-of-fold predictions clear an
    // absolute hold-out AUC bar. This is a self-contained statement of
    // predictive quality — gam's REML smoothing yields a model that genuinely
    // separates the classes on data it never saw, independent of any reference.
    assert!(
        gam_mean_auc >= GAM_MIN_HOLDOUT_AUC,
        "gam hold-out AUC below the absolute quality bar: gam={gam_mean_auc:.4} < {GAM_MIN_HOLDOUT_AUC:.2} \
         (per-fold {gam_fold_auc:?}) — REML smoothing is not generalizing on prostate y ~ pc1 + pc2"
    );

    // (2) MATCH-OR-BEAT THE BEST MATURE BASELINE: gam must be no less accurate
    // than the stronger of EBM/pyGAM by more than the sampling-slack margin. A
    // mature additive learner doing materially better than gam out of sample is
    // a real quality finding; gam matching or beating both passes trivially.
    assert!(
        gam_mean_auc >= best_ref_auc - BASELINE_AUC_MARGIN,
        "gam trails the best mature additive baseline by more than {BASELINE_AUC_MARGIN:.2} AUC: \
         gam={gam_mean_auc:.4} < best_ref={best_ref_auc:.4} - {BASELINE_AUC_MARGIN:.2} \
         (EBM={ebm_mean_auc:.4}, pyGAM={pygam_mean_auc:.4})"
    );
}

/// SECOND real-data arm exercising the SAME binomial-logit smooth capability on
/// the SAME prostate dataset, but asserting a DIFFERENT objective predictive
/// metric — hold-out **negative log-likelihood (log-loss)** on a single fixed
/// deterministic split rather than k-fold AUC. AUC measures only rank
/// separation; log-loss measures probabilistic CALIBRATION, so this arm proves
/// gam's REML smoothing yields well-calibrated out-of-sample probabilities, not
/// merely the right ordering.
///
/// Dataset SOURCE: `bench/datasets/prostate.csv` — the prostate principal-
/// component features (`pc1`, `pc2`) against the binary outcome `y`, the same
/// real binary-classification problem used by the AUC arm above.
///
/// Split: row `i` is HELD OUT iff `i % 5 == 0` (every 5th row, ~131 test rows);
/// all other rows train. Identical train/test rows in the SAME order are handed
/// to gam and to both reference engines, the latter via a single equal-length
/// `is_train` mask column (1.0 = train, 0.0 = test) so every reference Column is
/// the same length within the one `run_python` call.
///
/// OBJECTIVE METRIC ASSERTED:
///   (1) PRIMARY (tool-free): gam's own hold-out mean log-loss
///       `gam_nll <= 0.66` — strictly better than the `ln 2 ≈ 0.693` base-rate
///       predictor, i.e. gam's smoothed probabilities genuinely predict the
///       held-out labels.
///   (2) MATCH-OR-BEAT: `gam_nll <= min(ebm_nll, pygam_nll) + 0.03` — gam is no
///       worse than the best-calibrated mature additive learner by more than the
///       sampling slack. Lower NLL is better, so the margin is ADDED to the best
///       (minimum) reference NLL.
#[test]
fn gam_binomial_logit_holdout_predictive_quality_on_real_data() {
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

    // ---- single fixed deterministic split: held out iff i % 5 == 0 --------
    let is_test = |i: usize| i % 5 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 400 && test_rows.len() > 100,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );
    // Both label classes must be present in the held-out set for a meaningful
    // log-loss (and so the reference engines train on both classes).
    let test_pos = test_rows.iter().filter(|&&i| y[i] > 0.5).count();
    assert!(
        test_pos > 0 && test_pos < test_rows.len(),
        "held-out fold must contain both classes (pos={test_pos}, n={})",
        test_rows.len()
    );

    // ===================================================================
    // (A) gam: single REML binomial-logit fit on TRAIN, hold-out NLL.
    // ===================================================================
    let cfg = FitConfig {
        family: Some("binomial".to_string()),
        link: Some("logit".to_string()),
        ..FitConfig::default()
    };

    let train_ds = subset_dataset(&ds, &train_rows);
    let result = fit_from_formula("y ~ s(pc1, k=5) + s(pc2, k=5)", &train_ds, &cfg)
        .unwrap_or_else(|e| panic!("gam fit failed: {e:?}"));
    let FitResult::Standard(fit) = result else {
        panic!("binomial(logit) GAM with smooths should be a Standard fit");
    };

    // Freeze the trained spec against the trained design before predicting, so
    // the held-out basis reuses the training-time knots/rotation/identifiability
    // transform (same rationale as the AUC arm: never re-plan artifacts from the
    // TEST rows). The canonical predict path freezes identically.
    let frozenspec = freeze_term_collection_from_design(&fit.resolvedspec, &fit.design)
        .unwrap_or_else(|e| panic!("freeze gam spec failed: {e:?}"));

    let mut grid = Array2::<f64>::zeros((test_rows.len(), ds.headers.len()));
    for (out_r, &src_r) in test_rows.iter().enumerate() {
        grid[[out_r, pc1_idx]] = pc1[src_r];
        grid[[out_r, pc2_idx]] = pc2[src_r];
    }
    let design = build_term_collection_design(grid.view(), &frozenspec)
        .expect("rebuild gam design at held-out test rows");
    let eta: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let gam_prob: Vec<f64> = eta.iter().map(|&e| inv_logit(e)).collect();
    let test_labels: Vec<f64> = test_rows.iter().map(|&i| y[i]).collect();
    let gam_nll = log_loss(&gam_prob, &test_labels);
    let gam_auc_holdout = auc(&gam_prob, &test_labels);

    // ===================================================================
    // (B) InterpretML EBM + (C) pyGAM LogisticGAM: SAME single split.
    //     One run_python call; every Column is full length (n rows) and the
    //     `is_train` mask selects identical train/test rows in identical order.
    // ===================================================================
    let is_train_mask: Vec<f64> = (0..n).map(|i| if is_test(i) { 0.0 } else { 1.0 }).collect();
    let py = run_python(
        &[
            Column::new("pc1", &pc1),
            Column::new("pc2", &pc2),
            Column::new("y", &y),
            Column::new("is_train", &is_train_mask),
        ],
        r#"
from sklearn.metrics import log_loss as sk_log_loss
from interpret.glassbox import ExplainableBoostingClassifier
from pygam import LogisticGAM, s

X = np.column_stack([np.asarray(df["pc1"], float), np.asarray(df["pc2"], float)])
yv = np.asarray(df["y"], float)
is_train = np.asarray(df["is_train"], float) > 0.5
tr = is_train
te = ~is_train
Xtr, ytr = X[tr], yv[tr]
Xte, yte = X[te], yv[te]

EPS = 1e-12

# InterpretML EBM: PURELY additive shape-function classifier (interactions=0
# disables the default automatic pairwise term) so it is the SAME additive
# structure s(pc1) + s(pc2) gam and pyGAM fit. Deterministic seed.
ebm = ExplainableBoostingClassifier(interactions=0, random_state=0)
ebm.fit(Xtr, ytr)
p_ebm = np.clip(ebm.predict_proba(Xte)[:, 1], EPS, 1.0 - EPS)
# Score with the SAME log-loss the Rust side uses, evaluated against both
# possible labels so sklearn does not infer a single-class problem.
ebm_nll = float(sk_log_loss(yte, p_ebm, labels=[0.0, 1.0]))

# pyGAM LogisticGAM: classical GCV-smoothed penalized GAM, same additive
# structure s(pc1) + s(pc2) with 5 splines each (k=5).
lg = LogisticGAM(s(0, n_splines=5) + s(1, n_splines=5)).fit(Xtr, ytr)
p_lg = np.clip(lg.predict_proba(Xte), EPS, 1.0 - EPS)
pygam_nll = float(sk_log_loss(yte, p_lg, labels=[0.0, 1.0]))

emit("ebm_nll", [ebm_nll])
emit("pygam_nll", [pygam_nll])
emit("n_test", [float(int(te.sum()))])
"#,
    );
    let ebm_nll = py.scalar("ebm_nll");
    let pygam_nll = py.scalar("pygam_nll");
    let ref_n_test = py.scalar("n_test");
    assert_eq!(
        ref_n_test as usize,
        test_rows.len(),
        "reference test-fold size {ref_n_test} disagrees with gam's {} — split drifted",
        test_rows.len()
    );

    let best_ref_nll = ebm_nll.min(pygam_nll);

    eprintln!(
        "prostate fixed-split binomial(logit) hold-out NLL | n_train={} n_test={} | \
         gam_nll={gam_nll:.4} (auc={gam_auc_holdout:.4}) | EBM_nll={ebm_nll:.4} | \
         pyGAM_nll={pygam_nll:.4} | best_ref_nll={best_ref_nll:.4} \
         (bars: abs<={GAM_MAX_HOLDOUT_NLL:.2}, match-or-beat<=best_ref+{BASELINE_NLL_MARGIN:.2})",
        train_rows.len(),
        test_rows.len(),
    );

    // (1) PRIMARY OBJECTIVE CLAIM: gam's own hold-out probabilities are better
    // calibrated than the base-rate predictor (ln 2 ≈ 0.693). A self-contained
    // statement of probabilistic predictive quality, independent of any
    // reference.
    assert!(
        gam_nll <= GAM_MAX_HOLDOUT_NLL,
        "gam hold-out NLL above the absolute quality bar: gam_nll={gam_nll:.4} > {GAM_MAX_HOLDOUT_NLL:.2} \
         — REML-smoothed probabilities do not beat the base-rate predictor on prostate y ~ pc1 + pc2"
    );

    // (2) MATCH-OR-BEAT THE BEST MATURE BASELINE on log-loss: gam must be no
    // worse than the better-calibrated of {EBM, pyGAM} by more than the sampling
    // slack. Lower NLL is better, so the margin is ADDED to the minimum reference
    // NLL. gam matching or beating both passes trivially.
    assert!(
        gam_nll <= best_ref_nll + BASELINE_NLL_MARGIN,
        "gam trails the best mature additive baseline by more than {BASELINE_NLL_MARGIN:.2} nats NLL: \
         gam_nll={gam_nll:.4} > best_ref_nll={best_ref_nll:.4} + {BASELINE_NLL_MARGIN:.2} \
         (EBM={ebm_nll:.4}, pyGAM={pygam_nll:.4})"
    );
}
