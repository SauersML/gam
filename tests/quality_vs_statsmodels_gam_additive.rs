//! End-to-end quality: gam's *multi-smooth additive* model (p-spline + cyclic
//! cubic + Matérn) must agree with `statsmodels`' Generalized Additive Model —
//! a third mature, independently-implemented additive-spline reference (after
//! mgcv and pyGAM) — on the same Gaussian data.
//!
//! Why statsmodels GAM here: `statsmodels.gam` is the canonical Python additive
//! B-spline implementation. It supports exactly the basis mixture this test
//! exercises — `BSplines` (penalized p-spline, the analogue of gam's
//! `s(..., bs='ps')`), `CyclicCubicSplines` (periodic, the analogue of gam's
//! `cc(...)`), and a further penalized `BSplines` term standing in for gam's
//! `matern(...)` smooth of `x3` (statsmodels has no native Matérn kernel, so
//! the mature cross-engine analogue for "a penalized smooth of x3" is another
//! B-spline smoother; both engines must recover the same underlying function).
//! statsmodels selects its per-smoother penalty weights by GCV via
//! `select_penweight()`, so it targets a comparable smoothing objective to
//! gam's REML — the fitted additive surfaces should essentially coincide.
//!
//! Combinations are where bugs hide: overlapping bases, mixed penalty
//! structures, and identifiability constraints across three simultaneously-fit
//! smooths of different families. This test asserts that gam's multi-term
//! formula parsing, design stacking, identifiability handling, and REML smooth-
//! parameter selection produce an additive fit that matches statsmodels jointly
//! and term-by-term, and recovers the noise-free truth.
//!
//! A genuine divergence failing any assertion below is a real, useful signal —
//! the bounds are principled and must NOT be loosened to force a pass.

use csv::StringRecord;
use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_python};
use gam::{
    FitConfig, FitResult, encode_recordswith_inferred_schema, fit_from_formula, init_parallelism,
};
use ndarray::Array2;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Distribution, Normal, Uniform};

const N: usize = 250;
const SEED: u64 = 789;
const SIGMA: f64 = 0.10;

/// Noise-free additive truth used by BOTH engines and the recovery check.
/// f(x1,x2,x3) = sin(4π·x1) + cos(4π·x2) + 0.5·x3. The cosine term completes
/// exactly two periods on [0,1], so it is genuinely periodic on the unit
/// interval — the right structure to exercise the cyclic-cubic smooth.
fn truth(x1: f64, x2: f64, x3: f64) -> f64 {
    let tau = 2.0 * std::f64::consts::PI;
    (2.0 * tau * x1).sin() + (2.0 * tau * x2).cos() + 0.5 * x3
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
    let headers: Vec<String> = ["x1", "x2", "x3", "y"].into_iter().map(String::from).collect();
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
# (smoother) matrix, res.hat_matrix_trace (== sum(res.edf), the per-column edf).
# THIS is the quantity comparable to gam's penalized edf_total. (res.df_model is
# the unpenalized parametric rank = total basis-column count, which would
# spuriously disagree with a penalized fit and must NOT be used.) The trace
# already counts the intercept column.
emit("edf_total", [float(res.hat_matrix_trace)])

# Per-term partial predictions: evaluate each smoother's basis at the data
# with that smoother's own coefficients (offset-free term contribution).
params = np.asarray(res.params, dtype=float)
# Recover each smoother's own basis at the data; its column count IS the per-
# term coefficient width. Slice res.params by those widths instead of hard-
# coding df, and assert the layout [intercept | x1 | x2 | x3] adds up, so a
# basis-dimension mismatch fails loudly rather than producing garbage partials.
Bx1 = np.asarray(bs1.transform(x1.reshape(-1, 1)), dtype=float)
Bx2 = np.asarray(cc2.transform(x2.reshape(-1, 1)), dtype=float)
Bx3 = np.asarray(bs3.transform(x3.reshape(-1, 1)), dtype=float)
k1, k2, k3 = Bx1.shape[1], Bx2.shape[1], Bx3.shape[1]
assert params.shape[0] == 1 + k1 + k2 + k3, \
    f"unexpected GLMGam param layout: {params.shape[0]} != 1+{k1}+{k2}+{k3}"
o = 1
b1 = params[o:o+k1]; o += k1
b2 = params[o:o+k2]; o += k2
b3 = params[o:o+k3]; o += k3
emit("part_x1", Bx1 @ b1)
emit("part_x2", Bx2 @ b2)
emit("part_x3", Bx3 @ b3)
"#,
    );
    let sm_fitted = r.vector("fitted");
    let sm_edf = r.scalar("edf_total");
    let sm_part_x1 = r.vector("part_x1");
    let sm_part_x2 = r.vector("part_x2");
    let sm_part_x3 = r.vector("part_x3");

    assert_eq!(sm_fitted.len(), N, "statsmodels fitted length mismatch");

    // ---- compare -----------------------------------------------------------
    let rel = relative_l2(&gam_fitted, sm_fitted);
    let corr_joint = pearson(&gam_fitted, sm_fitted);
    let corr_truth_gam = pearson(&gam_fitted, &y_truth);
    let corr_truth_sm = pearson(sm_fitted, &y_truth);
    let edf_rel = (gam_edf - sm_edf).abs() / sm_edf.abs().max(1.0);
    let p1 = pearson(&gam_part_x1, sm_part_x1);
    let p2 = pearson(&gam_part_x2, sm_part_x2);
    let p3 = pearson(&gam_part_x3, sm_part_x3);

    eprintln!(
        "additive ps+cc+matern: n={N} gam_edf={gam_edf:.3} sm_edf={sm_edf:.3} \
         rel_l2={rel:.4} pearson_joint={corr_joint:.5} \
         pearson_truth(gam={corr_truth_gam:.4}, sm={corr_truth_sm:.4}) \
         term_pearson(x1={p1:.4}, x2={p2:.4}, x3={p3:.4}) edf_rel={edf_rel:.3}"
    );

    // (1) Joint fitted surfaces. gam (REML) and statsmodels (GCV) use different
    // penalty/PIRLS machinery on overlapping bases, so we allow a looser band
    // than the single-smooth-vs-mgcv case (which hits ~0.005). rel_l2 < 0.08 is
    // still tight: it asserts the two additive surfaces agree to within ~8% of
    // the fitted-vector norm, which a genuine stacking/identifiability bug
    // (wrong term sign, dropped null space, mis-centred basis) would blow past.
    assert!(
        rel < 0.08,
        "joint additive fit diverges from statsmodels: rel_l2={rel:.4}"
    );
    assert!(
        corr_joint > 0.98,
        "joint additive fit shape disagrees with statsmodels: pearson={corr_joint:.5}"
    );

    // (2) Recovery of the noise-free truth. With σ=0.10 and n=250 a correct
    // additive smoother recovers the true mean closely; >0.98 Pearson is the
    // sanity floor both engines must clear (a mis-built term destroys it).
    assert!(
        corr_truth_gam > 0.98,
        "gam additive fit fails to recover truth: pearson={corr_truth_gam:.4}"
    );
    assert!(
        corr_truth_sm > 0.98,
        "reference statsmodels fit fails to recover truth (data/ref sanity): \
         pearson={corr_truth_sm:.4}"
    );

    // (3) Term-wise agreement. Each smooth's recovered shape must track the
    // corresponding statsmodels smoother. >0.95 Pearson per term is the per-
    // smooth analogue of the "EDF within 25%" complexity-matching intent:
    // a term that is over/under-smoothed, mis-parameterised, or contaminated by
    // a neighbouring term (the classic multi-smooth identifiability bug) would
    // drop well below this. The cyclic term is the most basis-sensitive.
    assert!(p1 > 0.95, "p-spline term s(x1) disagrees with statsmodels: pearson={p1:.4}");
    assert!(p2 > 0.95, "cyclic term cc(x2) disagrees with statsmodels: pearson={p2:.4}");
    assert!(p3 > 0.95, "matern term matern(x3) disagrees with statsmodels: pearson={p3:.4}");

    // (4) Total PENALIZED effective degrees of freedom: gam's edf_total vs the
    // trace of statsmodels' penalized smoother hat matrix (res.edf). Both are
    // the true effective complexity of the *fitted* (penalized) model, so they
    // are the right cross-engine quantity to compare. Basis/null-space and
    // penalty-selection conventions (gam REML vs statsmodels GCV) still differ,
    // so we assert same-ballpark complexity, within 25% — a broken term that
    // inflates or collapses the total blows past this.
    assert!(
        edf_rel < 0.25,
        "total effective degrees of freedom disagree: gam={gam_edf:.3} sm={sm_edf:.3} (rel={edf_rel:.3})"
    );
}
