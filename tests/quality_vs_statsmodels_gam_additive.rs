//! End-to-end quality: gam's *multi-smooth additive* model (p-spline + cyclic
//! cubic + Matérn) must RECOVER A KNOWN ADDITIVE TRUTH on noisy Gaussian data.
//!
//! OBJECTIVE METRIC (the pass criterion): the data are generated from a known
//! noise-free additive function `truth(x1,x2,x3) = sin(4π·x1) + cos(4π·x2) +
//! 0.5·x3` plus i.i.d. N(0,σ²) noise (σ = 0.10). The primary assertion is that
//! gam's fitted mean recovers that truth in root-mean-square:
//!     RMSE(gam_fitted, y_truth) <= σ.
//! A correct smoother averages out independent noise, so the fitted mean is
//! strictly closer to the truth than the noisy observations are; the per-point
//! noise scale σ is therefore a principled, non-trivial ceiling on the recovery
//! error (the raw observations sit at ≈σ from truth by construction). We also
//! assert the recovered surface is highly correlated with the truth in SHAPE
//! (Pearson) and, term by term, that each smooth recovers the shape of its own
//! true component — the direct test that the p-spline, cyclic, and Matérn terms
//! are each constructed, identified, and stacked correctly.
//!
//! Why statsmodels GAM is still here, as a BASELINE TO MATCH-OR-BEAT (not as the
//! pass criterion): `statsmodels.gam` is a mature, independently-implemented
//! additive B-spline reference. We fit the SAME data with it and require that
//! gam's recovery error is no worse than statsmodels' by more than 10%
//! (RMSE_gam <= 1.10 · RMSE_sm). This demotes the reference to an accuracy
//! benchmark: matching its noisy fitted *output* proves nothing, but recovering
//! the truth at least as accurately as a trusted tool is a real quality claim.
//! The cross-engine `rel_l2`/Pearson agreement is computed and printed for
//! context via `eprintln!`, but is NOT a pass/fail gate.
//!
//! Combinations are where bugs hide: overlapping bases, mixed penalty
//! structures, and identifiability constraints across three simultaneously-fit
//! smooths of different families. A genuine divergence failing any assertion
//! below is a real, useful signal — the bounds are principled and must NOT be
//! loosened to force a pass.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pad_to, pearson, r2, relative_l2, rmse, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
    load_csvwith_inferred_schema,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};
use std::path::Path;

/// Real lidar benchmark (range -> logratio). Source: Sigrist (1994) light
/// detection and ranging experiment, the canonical 1-D smoothing benchmark
/// distributed with R's `SemiPar`/`gamair` packages; vendored at
/// `bench/datasets/lidar.csv`.
const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

const N: usize = 250;
const SEED: u64 = 789;
const SIGMA: f64 = 0.10;

/// Noise-free additive truth used by BOTH engines and the recovery check.
/// f(x1,x2,x3) = sin(4π·x1) + cos(4π·x2) + 0.5·x3. The cosine term completes
/// exactly two periods on [0,1], so it is genuinely periodic on the unit
/// interval — the right structure to exercise the cyclic-cubic smooth.
fn truth(x1: f64, x2: f64, x3: f64) -> f64 {
    true_x1(x1) + true_x2(x2) + true_x3(x3)
}

/// True smooth component for x1 (the p-spline term's target shape).
fn true_x1(x1: f64) -> f64 {
    (4.0 * std::f64::consts::PI * x1).sin()
}

/// True smooth component for x2 (the cyclic-cubic term's target shape).
fn true_x2(x2: f64) -> f64 {
    (4.0 * std::f64::consts::PI * x2).cos()
}

/// True smooth component for x3 (the Matérn term's target shape).
fn true_x3(x3: f64) -> f64 {
    0.5 * x3
}

#[test]
fn gam_additive_matches_statsmodels_gam() {
    init_parallelism();

    // ---- synthesize identical data fed to gam AND statsmodels --------------
    let mut rng = StdRng::seed_from_u64(SEED);
    let unif = Uniform::new(0.0, 1.0).expect("uniform [0,1]");
    let noise = Normal::new(0.0, SIGMA).expect("normal noise");

    let x1: Vec<f64> = (0..N).map(|_| unif.sample(&mut rng)).collect();
    let x2: Vec<f64> = (0..N).map(|_| unif.sample(&mut rng)).collect();
    let x3: Vec<f64> = (0..N).map(|_| unif.sample(&mut rng)).collect();
    let y_truth: Vec<f64> = (0..N).map(|i| truth(x1[i], x2[i], x3[i])).collect();
    let y: Vec<f64> = (0..N)
        .map(|i| y_truth[i] + noise.sample(&mut rng))
        .collect();

    // gam dataset (header order: x1, x2, x3, y).
    let headers: Vec<String> = ["x1", "x2", "x3", "y"]
        .into_iter()
        .map(String::from)
        .collect();
    let rows: Vec<StringRecord> = (0..N)
        .map(|i| {
            StringRecord::from(vec![
                x1[i].to_string(),
                x2[i].to_string(),
                x3[i].to_string(),
                y[i].to_string(),
            ])
        })
        .collect();
    let ds = encode_recordswith_inferred_schema(headers, rows).expect("encode additive dataset");
    let col = ds.column_map();
    let (i1, i2, i3) = (col["x1"], col["x2"], col["x3"]);
    let ncols = ds.headers.len();

    // ---- fit gam: y ~ s(x1,ps) + cc(x2) + matern(x3), Gaussian/REML --------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula(
        "y ~ s(x1, bs='ps', k=10) + cc(x2, k=8, period_start=0, period_end=1) + matern(x3, nu=1.5, k=12)",
        &ds,
        &cfg,
    )
    .expect("gam additive fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a Standard Gaussian GAM fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");
    let beta = &fit.fit.beta;

    // Helper: gam fitted values at an (N × ncols) grid of covariate columns.
    let gam_predict = |grid: &Array2<f64>| -> Vec<f64> {
        let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
            .expect("rebuild gam design at grid");
        design.design.apply(beta).to_vec()
    };

    // Joint fitted values at the training points (identity link => Xβ = mean).
    let mut grid_full = Array2::<f64>::zeros((N, ncols));
    for i in 0..N {
        grid_full[[i, i1]] = x1[i];
        grid_full[[i, i2]] = x2[i];
        grid_full[[i, i3]] = x3[i];
    }
    let gam_fitted = gam_predict(&grid_full);

    // Per-term partial fits: vary one covariate, hold the others at their mean.
    // The intercept and held-constant terms add a constant offset, which is
    // irrelevant to the SHAPE comparison (Pearson is offset-invariant). This
    // isolates each smooth's recovered function — the direct test that the
    // p-spline, cyclic, and Matérn terms are each constructed correctly.
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    let (m1, m2, m3) = (mean(&x1), mean(&x2), mean(&x3));
    let partial = |idx: usize, vals: &[f64]| -> Vec<f64> {
        let mut g = Array2::<f64>::zeros((N, ncols));
        for i in 0..N {
            g[[i, i1]] = m1;
            g[[i, i2]] = m2;
            g[[i, i3]] = m3;
        }
        for i in 0..N {
            g[[i, idx]] = vals[i];
        }
        gam_predict(&g)
    };
    let gam_part_x1 = partial(i1, &x1);
    let gam_part_x2 = partial(i2, &x2);
    let gam_part_x3 = partial(i3, &x3);

    // ---- fit the SAME data with statsmodels GAM (the mature reference) -----
    // BSplines for x1 (df=10, penalized p-spline), CyclicCubicSplines for x2
    // (df=8, periodic), BSplines for x3 (df=12) as the penalized-smooth
    // analogue of gam's Matérn term. select_penweight() picks each smoother's
    // penalty by GCV — comparable to gam's REML smoothing-parameter selection.
    let r = run_python(
        &[
            Column::new("x1", &x1),
            Column::new("x2", &x2),
            Column::new("x3", &x3),
            Column::new("y", &y),
        ],
        r#"
import numpy as np
from statsmodels.gam.api import GLMGam, BSplines, CyclicCubicSplines
from statsmodels.gam.smooth_basis import GenericSmoothers
import statsmodels.api as sm

x1 = np.asarray(df["x1"], dtype=float)
x2 = np.asarray(df["x2"], dtype=float)
x3 = np.asarray(df["x3"], dtype=float)
y  = np.asarray(df["y"],  dtype=float)

# Individual smoothers, each on its own covariate.
bs1 = BSplines(x1.reshape(-1, 1), df=[10], degree=[3])
cc2 = CyclicCubicSplines(x2.reshape(-1, 1), df=[8])
bs3 = BSplines(x3.reshape(-1, 1), df=[12], degree=[3])
smoothers = GenericSmoothers(np.column_stack([x1, x2, x3]), [bs1, cc2, bs3])

alpha0 = [1.0, 1.0, 1.0]
gam = GLMGam(y, smoother=smoothers, alpha=alpha0,
             family=sm.families.Gaussian(sm.families.links.Identity()))
# GCV search over the per-smoother penalty weights, then refit at the optimum.
# select_penweight()'s return shape has varied across statsmodels versions
# (bare alpha array vs (alpha, ...) tuple); normalize to the alpha vector so a
# version change cannot silently mis-bind the optimum.
sel = gam.select_penweight()
alpha_opt = sel[0] if isinstance(sel, tuple) else sel
alpha_opt = np.asarray(alpha_opt, dtype=float).reshape(-1)
assert alpha_opt.shape[0] == 3, f"expected 3 penalty weights, got {alpha_opt.shape}"
gam = GLMGam(y, smoother=smoothers, alpha=list(alpha_opt),
             family=sm.families.Gaussian(sm.families.links.Identity()))
res = gam.fit()

emit("fitted", np.asarray(res.fittedvalues, dtype=float))
# Penalized effective degrees of freedom = trace of the penalized hat
# (smoother) matrix, res.hat_matrix_trace. Emitted for CONTEXT ONLY (printed
# alongside gam's edf_total); it is NOT a pass/fail gate — matching another
# tool's edf is not a quality claim.
emit("edf_total", [float(res.hat_matrix_trace)])
"#,
    );
    let sm_fitted = r.vector("fitted");
    let sm_edf = r.scalar("edf_total");

    assert_eq!(sm_fitted.len(), N, "statsmodels fitted length mismatch");

    // ---- true per-term components (ground-truth shapes) --------------------
    // Each gam smooth must recover the SHAPE of its own true additive component,
    // not the reference's (noisy) fitted partial. We compare against the truth.
    let truth_part_x1: Vec<f64> = x1.iter().map(|&v| true_x1(v)).collect();
    let truth_part_x2: Vec<f64> = x2.iter().map(|&v| true_x2(v)).collect();
    let truth_part_x3: Vec<f64> = x3.iter().map(|&v| true_x3(v)).collect();

    // ---- OBJECTIVE metric: recovery of the noise-free truth ----------------
    // RMSE of each engine's fitted mean against the noise-free truth. The raw
    // observations sit at ≈σ from truth by construction; a correct smoother
    // averages noise out and lands well below σ.
    let rmse_gam = rmse(&gam_fitted, &y_truth);
    let rmse_sm = rmse(sm_fitted, &y_truth);
    let corr_truth_gam = pearson(&gam_fitted, &y_truth);
    // Term-wise SHAPE recovery against the true components (Pearson, offset-free).
    let p1 = pearson(&gam_part_x1, &truth_part_x1);
    let p2 = pearson(&gam_part_x2, &truth_part_x2);
    let p3 = pearson(&gam_part_x3, &truth_part_x3);

    // ---- cross-engine agreement: CONTEXT ONLY, not a pass/fail gate --------
    let rel = relative_l2(&gam_fitted, sm_fitted);
    let corr_joint = pearson(&gam_fitted, sm_fitted);
    eprintln!(
        "additive ps+cc+matern: n={N} sigma={SIGMA:.3} \
         rmse_truth(gam={rmse_gam:.4}, sm={rmse_sm:.4}) ratio={:.3} \
         pearson_truth_gam={corr_truth_gam:.4} \
         term_pearson_truth(x1={p1:.4}, x2={p2:.4}, x3={p3:.4}) \
         [context only] gam_edf={gam_edf:.3} sm_edf={sm_edf:.3} \
         rel_l2_vs_sm={rel:.4} pearson_joint_vs_sm={corr_joint:.5}",
        rmse_gam / rmse_sm.max(1e-12)
    );

    // (1) PRIMARY CLAIM — gam recovers the truth. With σ=0.10 and n=250 a
    // correct additive smoother averages independent noise out, so its fitted
    // mean is strictly closer to truth than the σ-scale observations. Requiring
    // RMSE(gam, truth) <= σ is therefore a principled, non-trivial ceiling that
    // a mis-built term (wrong sign, dropped null space, mis-centred basis,
    // cross-term contamination) would blow past.
    assert!(
        rmse_gam <= SIGMA,
        "gam additive fit fails to recover truth: rmse={rmse_gam:.4} > sigma={SIGMA:.3}"
    );

    // (2) MATCH-OR-BEAT the mature reference on ACCURACY (not on output). gam's
    // recovery error must be no worse than statsmodels' by more than 10%.
    assert!(
        rmse_gam <= rmse_sm * 1.10,
        "gam recovery worse than statsmodels baseline: rmse_gam={rmse_gam:.4} > 1.10*rmse_sm={:.4}",
        rmse_sm * 1.10
    );

    // (3) Joint SHAPE recovery: the fitted surface must track the true mean.
    // >0.98 Pearson is the sanity floor a correct multi-smooth fit clears.
    assert!(
        corr_truth_gam > 0.98,
        "gam additive fit shape diverges from truth: pearson={corr_truth_gam:.4}"
    );

    // (4) Term-wise SHAPE recovery against the TRUE components. Each smooth must
    // recover the shape of its own true additive function. >0.95 Pearson per
    // term: a term that is over/under-smoothed, mis-parameterised, or
    // contaminated by a neighbouring term (the classic multi-smooth
    // identifiability bug) drops well below this. The cyclic term is the most
    // basis-sensitive.
    assert!(
        p1 > 0.95,
        "p-spline term s(x1) fails to recover sin component: pearson={p1:.4}"
    );
    assert!(
        p2 > 0.95,
        "cyclic term cc(x2) fails to recover cos component: pearson={p2:.4}"
    );
    assert!(
        p3 > 0.95,
        "matern term matern(x3) fails to recover linear component: pearson={p3:.4}"
    );
}

/// REAL-DATA ARM (companion to the synthetic known-truth recovery test above).
///
/// On real data the truth is unknown, so the objective quality of a 1-D smooth
/// is its OUT-OF-SAMPLE predictive accuracy. We use the canonical lidar
/// benchmark (`logratio ~ s(range)`), make a deterministic train/test split
/// (every 4th row held out), fit `s(range)` on the training rows only, predict
/// the held-out rows, and assert:
///
///   PRIMARY (objective, tool-free): held-out `R2 >= 0.55`. The lidar signal is
///     strongly nonlinear; a competent smoother explains well over half the
///     held-out variance, far above the constant-mean predictor (R2 = 0). This
///     would catch a smooth that under-/over-fits the held-out range.
///
///   BASELINE (match-or-beat): statsmodels' GAM (mature, independently
///     implemented additive B-spline reference) fits the SAME training rows and
///     predicts the SAME held-out rows; gam's held-out RMSE must be no worse
///     than `sm_test_rmse * 1.10`. statsmodels is a baseline to match-or-beat on
///     accuracy, NEVER a fitted target to replicate.
#[test]
fn gam_additive_matches_statsmodels_gam_on_real_data() {
    init_parallelism();

    // ---- load the real lidar dataset (range -> logratio) ------------------
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range.len();
    assert!(n > 100, "lidar should have ~221 rows, got {n}");

    // ---- deterministic train/test split: every 4th row is held out --------
    let is_test = |i: usize| i % 4 == 0;
    let train_rows: Vec<usize> = (0..n).filter(|&i| !is_test(i)).collect();
    let test_rows: Vec<usize> = (0..n).filter(|&i| is_test(i)).collect();
    assert!(
        train_rows.len() > 100 && test_rows.len() > 30,
        "split sizes: train={} test={}",
        train_rows.len(),
        test_rows.len()
    );

    let train_range: Vec<f64> = train_rows.iter().map(|&i| range[i]).collect();
    let train_logratio: Vec<f64> = train_rows.iter().map(|&i| logratio[i]).collect();
    let test_range: Vec<f64> = test_rows.iter().map(|&i| range[i]).collect();
    let test_logratio: Vec<f64> = test_rows.iter().map(|&i| logratio[i]).collect();

    // Training-only dataset by sub-setting the encoded rows; headers, schema and
    // column kinds are unchanged, so the formula resolves identically.
    let p = ds.headers.len();
    let mut train_values = Array2::<f64>::zeros((train_rows.len(), p));
    for (out_row, &src_row) in train_rows.iter().enumerate() {
        for c in 0..p {
            train_values[[out_row, c]] = ds.values[[src_row, c]];
        }
    }
    let mut train_ds = ds.clone();
    train_ds.values = train_values;

    // ---- fit gam on TRAIN: logratio ~ s(range), Gaussian/REML --------------
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ s(range)", &train_ds, &cfg).expect("gam real fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a Standard Gaussian GAM fit");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam predictions at the held-out range points: rebuild the design from the
    // frozen spec (identity link => design*beta = predicted mean).
    let mut test_grid = Array2::<f64>::zeros((test_rows.len(), p));
    for (i, &r) in test_range.iter().enumerate() {
        test_grid[[i, range_idx]] = r;
    }
    let test_design = build_term_collection_design(test_grid.view(), &fit.resolvedspec)
        .expect("rebuild gam design at held-out points");
    let gam_test_pred: Vec<f64> = test_design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model on TRAIN with statsmodels GAM, predict the SAME
    // held-out rows. One reference call => every Column must be equal length:
    // train range/logratio are train-length; the test range rides along padded
    // to train length with a test_n marker, and only the first test_n entries
    // are read back inside the body.
    let r = run_python(
        &[
            Column::new("range", &train_range),
            Column::new("logratio", &train_logratio),
            Column::new("test_range", &pad_to(&test_range, train_range.len())),
            Column::new("test_n", &vec![test_range.len() as f64; train_range.len()]),
        ],
        r#"
import numpy as np
from statsmodels.gam.api import GLMGam, BSplines
import statsmodels.api as sm

xtr = np.asarray(df["range"],    dtype=float)
ytr = np.asarray(df["logratio"], dtype=float)
k   = int(np.asarray(df["test_n"], dtype=float)[0])
xte = np.asarray(df["test_range"], dtype=float)[:k]

# Penalized cubic B-spline smoother of range, fit on the TRAINING rows only.
bs = BSplines(xtr.reshape(-1, 1), df=[20], degree=[3])
gam = GLMGam(ytr, smoother=bs, alpha=[1.0],
             family=sm.families.Gaussian(sm.families.links.Identity()))
# GCV search over the penalty weight, then refit at the optimum (statsmodels'
# analogue of gam's REML smoothing-parameter selection).
sel = gam.select_penweight()
alpha_opt = sel[0] if isinstance(sel, tuple) else sel
alpha_opt = np.asarray(alpha_opt, dtype=float).reshape(-1)
gam = GLMGam(ytr, smoother=bs, alpha=list(alpha_opt),
             family=sm.families.Gaussian(sm.families.links.Identity()))
res = gam.fit()

# Predict the held-out range by mapping the new covariate through the SAME
# spline basis (bs.transform) and evaluating the fitted GAM there.
exog_smooth = bs.transform(xte.reshape(-1, 1))
test_pred = np.asarray(res.predict(exog_smooth=exog_smooth), dtype=float).reshape(-1)
assert test_pred.shape[0] == k, f"expected {k} held-out predictions, got {test_pred.shape}"
emit("test_pred", test_pred)
"#,
    );
    let sm_test_pred = r.vector("test_pred");
    assert_eq!(
        sm_test_pred.len(),
        test_rows.len(),
        "statsmodels held-out prediction length mismatch"
    );

    // ---- objective metrics on gam's OWN held-out predictions ---------------
    let gam_test_r2 = r2(&gam_test_pred, &test_logratio);
    let gam_test_rmse = rmse(&gam_test_pred, &test_logratio);
    let sm_test_rmse = rmse(sm_test_pred, &test_logratio);

    eprintln!(
        "lidar s(range) held-out: n_train={} n_test={} gam_edf={gam_edf:.3} \
         gam_test_R2={gam_test_r2:.4} gam_test_rmse={gam_test_rmse:.4} \
         sm_test_rmse={sm_test_rmse:.4}",
        train_rows.len(),
        test_rows.len(),
    );

    // ---- PRIMARY objective assertion: gam predicts the held-out signal -----
    assert!(
        gam_test_r2 >= 0.55,
        "gam's held-out predictive R2 too low: {gam_test_r2:.4} (< 0.55)"
    );

    // ---- BASELINE (match-or-beat): no worse than statsmodels on held-out RMSE
    assert!(
        gam_test_rmse <= sm_test_rmse * 1.10,
        "gam held-out RMSE {gam_test_rmse:.4} exceeds statsmodels {sm_test_rmse:.4} * 1.10"
    );

    // ---- complexity sanity: edf in a signal-appropriate range (not matched) -
    assert!(
        gam_edf > 1.0 && gam_edf < 30.0,
        "gam effective dof out of sane range: {gam_edf:.3}"
    );
}
