//! Three-way HONEST comparative benchmark of gam's penalized binomial-logit GAM
//! against two mature, independent GAM/ML standards on *hold-out* predictive
//! performance:
//!   * **InterpretML EBM** (`interpret.glassbox.ExplainableBoostingClassifier`) —
//!     the machine-learning world's GAM: a cyclic gradient-boosted additive model
//!     of shape functions, with its own (greedy boosting + bagging) complexity
//!     control.
//!   * **pyGAM `LogisticGAM`** — a second, classical penalized GAM reference that
//!     selects its smoothing by generalized cross-validation under PIRLS.
//!
//! Unlike the pointwise mgcv/sklearn check (`quality_vs_sklearn_binomial_logit`),
//! this is a *meta-test*: it does not assert that gam reproduces a particular
//! engine's fitted curve. Instead it asks the only question that matters for
//! generalization — **does gam's REML lambda-selection produce systematically
//! worse (or better) hold-out classification than EBM's boosting or pyGAM's
//! GCV?** Three reasonable additive classifiers on the same binary problem
//! should land within a hair of each other on cross-validated AUC; gam should
//! sit *in the middle*, not be an outlier.
//!
//! Design that makes the comparison fair and reproducible:
//!   * Identical data and an identical deterministic 5-fold split is handed to
//!     every engine: fold f = { rows i : i % 5 == f }. Each engine trains on the
//!     4 complementary folds and predicts the held-out fold, so all three see
//!     the SAME train/test partition on the SAME `y ~ s(pc1,k=5)+s(pc2,k=5)`
//!     additive structure.
//!   * We report each engine's mean hold-out AUC over the 5 folds and assert
//!     (1) gam lies inside the band defined by the TWO MATURE REFERENCES only
//!     (EBM/pyGAM midpoint ± [half their gap + 0.015 AUC]) — a band gam can
//!     genuinely fall outside, unlike a symmetric ±kσ band over a triplet which
//!     is vacuous (any of three points lies within √2σ of their own mean) — AND
//!     (2) all three means agree to within 0.02 AUC.
//!
//! A failure here is a real finding: it means gam's smoothing-parameter choice
//! generalizes materially differently from two mature additive learners. The
//! bound is NOT loosened to hide that — 0.02 AUC is the spec tolerance and the
//! reference band is anchored on the references (never on gam itself).

use gam::smooth::{build_term_collection_design, freeze_term_collection_from_design};
use gam::matrix::LinearOperator;
use gam::test_support::reference::{Column, run_python};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use gam::inference::data::EncodedDataset;
use ndarray::Array2;
use std::path::Path;

const PROSTATE_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/prostate.csv");
const N_FOLDS: usize = 5;

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
    assert!(n_pos > 0.0 && n_neg > 0.0, "AUC needs both classes present in the test fold");
    let sum_ranks_pos: f64 = ranks
        .iter()
        .zip(labels)
        .filter(|(_, y)| **y > 0.5)
        .map(|(r, _)| *r)
        .sum();
    (sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)
}

/// Build an `EncodedDataset` restricted to `rows` of `full`, preserving the
/// frozen schema/headers/column kinds so each fold's gam fit uses an identical
/// encoding to every other fold (no per-fold schema re-inference drift).
fn subset_dataset(full: &EncodedDataset, rows: &[usize]) -> EncodedDataset {
    let p = full.headers.len();
    let mut values = Array2::<f64>::zeros((rows.len(), p));
    for (out_r, &src_r) in rows.iter().enumerate() {
        values
            .row_mut(out_r)
            .assign(&full.values.row(src_r));
    }
    EncodedDataset {
        headers: full.headers.clone(),
        values,
        schema: full.schema.clone(),
        column_kinds: full.column_kinds.clone(),
    }
}

/// Standard-deviation (population, divide by k) of a small slice.
fn pop_std(x: &[f64]) -> f64 {
    let n = x.len() as f64;
    let m = x.iter().sum::<f64>() / n;
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / n).sqrt()
}

#[test]
fn gam_ebm_pygam_binomial_logit_holdout_agreement() {
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
    // rows {0,5,10,...}, fold 1 is {1,6,11,...}, etc. (matches the spec).
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
        assert!(!train_rows.is_empty() && !test_rows.is_empty(), "empty fold {f}");

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
    // ===================================================================
    // We hand the identical fold assignment vector so every engine trains on
    // the same 4 folds and scores the same held-out fold. Each engine emits a
    // 5-vector of per-fold hold-out AUCs computed by its OWN predictions but
    // evaluated by sklearn's roc_auc_score (one common AUC estimator) so the
    // metric itself is identical across engines.
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
    assert_eq!(pygam_fold_auc.len(), N_FOLDS, "pyGAM emitted wrong fold count");

    // ===================================================================
    // Three-way comparison.
    // ===================================================================
    let means = [gam_mean_auc, ebm_mean_auc, pygam_mean_auc];
    let grand_mean = means.iter().sum::<f64>() / 3.0;
    let sigma_across = pop_std(&means);
    // gam is the engine under test, so its "not an outlier" band must be built
    // from the TWO MATURE REFERENCES ONLY (EBM, pyGAM) -- never from a set that
    // includes gam's own mean. A symmetric ±kσ band around the mean of all
    // three is mathematically vacuous for a triplet: any of three points lies
    // within √2·σ ≈ 1.414σ of their own mean by construction, so a ±1.5σ test
    // can NEVER fire. Instead we anchor on the references' midpoint and widen by
    // (a) the references' own disagreement and (b) a fixed 0.015 AUC margin (the
    // hold-out sampling slack at ~130 test rows/fold). gam CAN fall outside this
    // band, so the assertion is real.
    let ref_mid = 0.5 * (ebm_mean_auc + pygam_mean_auc);
    let ref_gap = (ebm_mean_auc - pygam_mean_auc).abs();
    let ref_halfwidth = 0.5 * ref_gap + 0.015;
    let lo = ref_mid - ref_halfwidth;
    let hi = ref_mid + ref_halfwidth;
    let max_pairwise = {
        let mut m = 0.0f64;
        for i in 0..3 {
            for j in (i + 1)..3 {
                m = m.max((means[i] - means[j]).abs());
            }
        }
        m
    };

    eprintln!(
        "prostate 5-fold binomial(logit) hold-out AUC | \
         gam folds={gam_fold_auc:?} mean={gam_mean_auc:.4} | \
         EBM folds={ebm_fold_auc:?} mean={ebm_mean_auc:.4} | \
         pyGAM folds={pygam_fold_auc:?} mean={pygam_mean_auc:.4} | \
         grand_mean={grand_mean:.4} sigma_across={sigma_across:.4} \
         ref_band=[{lo:.4},{hi:.4}] max_pairwise={max_pairwise:.4}"
    );

    // (1) gam must not be an outlier RELATIVE TO THE TWO MATURE REFERENCES: its
    // mean hold-out AUC lies inside the EBM/pyGAM reference band. With two
    // independent additive learners (EBM boosting, pyGAM GCV) bracketing the
    // honest range on the same split, this is the real "gam generalizes like the
    // field" test; falling outside means REML lambda-selection systematically
    // over- or under-smooths relative to both references.
    assert!(
        gam_mean_auc >= lo && gam_mean_auc <= hi,
        "gam is an outlier in hold-out AUC: gam={gam_mean_auc:.4} not in EBM/pyGAM reference band \
         [{lo:.4},{hi:.4}] (EBM={ebm_mean_auc:.4}, pyGAM={pygam_mean_auc:.4})"
    );

    // (2) All three engines must agree to within 0.02 AUC (the spec tolerance):
    // a 2-point AUC gap on the same additive problem is the threshold at which
    // the engines are no longer "all reasonable for binary classification".
    assert!(
        max_pairwise < 0.02,
        "three-engine hold-out AUC disagree by more than 0.02: \
         gam={gam_mean_auc:.4} EBM={ebm_mean_auc:.4} pyGAM={pygam_mean_auc:.4} (max_pairwise={max_pairwise:.4})"
    );
}
