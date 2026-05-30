//! End-to-end quality: gam's 1-D **p-spline** smooth (`bs="ps"`) cross-checked
//! against **pyGAM** — an independent, mature GAM library built on scipy/
//! scikit-learn bases with its own PIRLS fit and lambda selection.
//!
//! The companion test `quality_vs_mgcv_gaussian_smooth.rs` already pins gam's
//! thin-plate smooth to mgcv (REML). This test adds a *second, independent*
//! GAM reference for the **penalized B-spline** family specifically:
//! pyGAM's `LinearGAM(s(0, n_splines=15))` is, by default, a cubic (degree-3)
//! B-spline with a 2nd-order difference penalty — exactly the basis gam builds
//! for `s(range, bs="ps", k=15)` (degree 3, penalty_order 2, and
//! `k = internal_knots + degree + 1 = 15` basis functions). Fitting the
//! canonical `lidar` benchmark with both engines and comparing the *fitted
//! function* validates that gam's p-spline tracks a third-party GAM standard,
//! not just mgcv.
//!
//! Why looser bounds than the mgcv test: we let pyGAM select its single
//! smoothing parameter from the data with `.gridsearch()` (its default search
//! minimizes generalized cross-validation under a PIRLS fit), whereas gam
//! selects lambda by REML. Plain `LinearGAM(...).fit()` would instead leave the
//! per-term penalty at pyGAM's fixed default `lam=0.6` — that is NOT a fitted
//! smoothing parameter, so we must call `.gridsearch()` for an apples-to-apples
//! "both engines auto-select smoothing" comparison. GCV and REML pick slightly
//! different amounts of smoothing on the same data, so the fitted curves agree
//! in shape (correlation) very tightly but can differ by a few percent in L2 —
//! hence `relative_l2 < 0.06` and `pearson > 0.995`, with EDF agreement only to
//! within 30% (different penalty conventions and lambda-selection criteria).
//! These bounds still falsify any real divergence in the smoother while
//! honestly accounting for the REML-vs-GCV slack.

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::test_support::reference::{Column, pearson, relative_l2, run_python};
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::Path;

const LIDAR_CSV: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/bench/datasets/lidar.csv");

#[test]
fn gam_pspline_matches_pygam_on_lidar() {
    init_parallelism();

    // ---- load the canonical lidar dataset (range -> logratio) -------------
    let ds = load_csvwith_inferred_schema(Path::new(LIDAR_CSV)).expect("load lidar.csv");
    let col = ds.column_map();
    let range_idx = col["range"];
    let logratio_idx = col["logratio"];
    let range: Vec<f64> = ds.values.column(range_idx).to_vec();
    let logratio: Vec<f64> = ds.values.column(logratio_idx).to_vec();
    let n = range.len();
    assert!(n > 100, "lidar should have ~221 rows, got {n}");

    // ---- fit with gam: logratio ~ s(range, bs="ps", k=15), Gaussian/REML --
    // `bs="ps"` => degree-3 B-spline with 2nd-order difference penalty;
    // `k=15` => 15 basis functions (internal_knots = 15 - degree - 1 = 11),
    // matching pyGAM's default `s(0, n_splines=15)`.
    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("logratio ~ s(range, bs=\"ps\", k=15)", &ds, &cfg)
        .expect("gam p-spline fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a Gaussian p-spline smooth");
    };
    let gam_edf = fit.fit.edf_total().expect("gam reports total edf");

    // gam fitted values at the training points: rebuild the frozen design at
    // the observed `range` (identity link => design*beta = mean response).
    let mut grid = Array2::<f64>::zeros((n, ds.headers.len()));
    for (i, &r) in range.iter().enumerate() {
        grid[[i, range_idx]] = r;
    }
    let design = build_term_collection_design(grid.view(), &fit.resolvedspec)
        .expect("rebuild p-spline design at training points");
    let gam_fitted: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();

    // ---- fit the SAME model with pyGAM (the independent GAM reference) -----
    // LinearGAM(s(0, n_splines=15)) is a default cubic (spline_order=3) B-spline
    // with a 2nd-order difference penalty (penalties='auto'), the same basis gam
    // builds for bs="ps", k=15. `.gridsearch()` selects the single penalty by
    // minimizing GCV (pyGAM's default search objective) so that, like gam's REML,
    // the smoothing parameter is fit from the data rather than left at pyGAM's
    // fixed default lam=0.6. `.predict` on the training inputs returns the fitted
    // function; statistics_['edof'] the effective degrees of freedom.
    let py = run_python(
        &[
            Column::new("range", &range),
            Column::new("logratio", &logratio),
        ],
        r#"
from pygam import LinearGAM, s
X = np.asarray(df["range"], dtype=float).reshape(-1, 1)
y = np.asarray(df["logratio"], dtype=float)
gam = LinearGAM(s(0, n_splines=15)).gridsearch(X, y, progress=False)
emit("fitted", gam.predict(X))
emit("edf", [float(gam.statistics_["edof"])])
"#,
    );
    let pygam_fitted = py.vector("fitted");
    let pygam_edf = py.scalar("edf");

    assert_eq!(pygam_fitted.len(), n, "pyGAM fitted length mismatch");

    // ---- compare ----------------------------------------------------------
    let rel = relative_l2(&gam_fitted, pygam_fitted);
    let corr = pearson(&gam_fitted, pygam_fitted);
    let edf_rel = (gam_edf - pygam_edf).abs() / pygam_edf.abs().max(1.0);

    eprintln!(
        "lidar s(range,bs=ps,k=15): n={n} gam_edf={gam_edf:.3} pygam_edf={pygam_edf:.3} \
         rel_l2={rel:.4} pearson={corr:.5} edf_rel={edf_rel:.3}"
    );

    // Shape agreement is the strongest claim: two independent GAM engines on the
    // same p-spline basis must trace essentially the same curve.
    assert!(
        corr > 0.995,
        "gam vs pyGAM p-spline shapes diverge: pearson={corr:.5}"
    );
    // L2 budget absorbs the REML-vs-GCV lambda-selection difference on the
    // heteroscedastic lidar data (the two criteria settle on modestly different
    // amounts of smoothing); 0.06 is loose enough to not flag that slack yet
    // tight enough to catch a real basis or penalty bug (a wrong difference-
    // penalty order or degree changes the fitted curve far more than 6%).
    assert!(
        rel < 0.06,
        "gam p-spline fit diverges from pyGAM: rel_l2={rel:.4}"
    );
    // EDF conventions differ (penalty normalization + lambda criterion), so we
    // assert same-ballpark model complexity within 30% relative.
    assert!(
        edf_rel < 0.30,
        "effective degrees of freedom disagree: gam={gam_edf:.3} pygam={pygam_edf:.3} (rel={edf_rel:.3})"
    );
}
