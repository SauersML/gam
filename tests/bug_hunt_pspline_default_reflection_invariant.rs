//! Regression test for issue #1548.
//!
//! The DEFAULT univariate P-spline smooth `s(x, bs="ps")` must be invariant to
//! reflecting the covariate `x -> -x`. Reflection is an EXACT symmetry of the
//! uniform B-spline basis plus discrete difference penalty: mirroring `x` simply
//! reverses the order of the basis columns and the penalty rows, so the fitted
//! function `f` must satisfy `f1(t) == f2(-t)` for the two fits to machine-ish
//! precision (only column-reorder / floating-point summation-order noise).
//!
//! The bug (per #1548): the Marra & Wood double-penalty null-space smoothing
//! parameter `lambda_null` lands on opposite shoulders of a weakly-identified,
//! near-FLAT REML ridge depending on covariate orientation, because reflection
//! reverses the basis column order and flips the floating-point landing shoulder
//! of the outer rho-optimizer. The reported defect is on the numpy legacy
//! `RandomState(2)` frame: a mirror fit drifting ~0.076 (~3.4% of the signal
//! range), with `lambda_null` jumping 0.69 -> 1.1e11 across orientations.
//!
//! This test loads the EXACT datasets from the issue's reproduction
//! (`np.random.RandomState(seed)` for seed in {0,1,2,3,4}; committed as CSV
//! fixtures so the failing seed-2 geometry is reproduced byte-for-byte without a
//! numpy dependency), fits `y ~ s(x, bs="ps")` on `(x, y)` and on `(-x, y)`,
//! predicts both at the issue's grid, and asserts the mirror curves agree to
//! `< 1e-3` (well below the ~0.076 observed defect, well above reorder noise).

use gam::matrix::LinearOperator;
use gam::smooth::build_term_collection_design;
use gam::{FitConfig, FitResult, fit_from_formula, init_parallelism, load_csvwith_inferred_schema};
use ndarray::Array2;
use std::path::PathBuf;

/// Load one of the committed numpy `RandomState(seed)` fixtures as (x, y).
fn load_seed(seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push(format!("bench/datasets/refl1548/seed{seed}.csv"));
    let ds = load_csvwith_inferred_schema(&path).expect("load seed fixture csv");
    let col = ds.column_map();
    let xi = col["x"];
    let yi = col["y"];
    let x = ds.values.column(xi).to_vec();
    let y = ds.values.column(yi).to_vec();
    (x, y)
}

/// Build an EncodedDataset (x, y) from raw columns via a scratch CSV so the
/// same inference path used in production loads it.
fn dataset_from(x: &[f64], y: &[f64], tag: &str) -> gam::data::EncodedDataset {
    use std::io::Write;
    let mut path = PathBuf::from(std::env::temp_dir());
    path.push(format!("gam_refl_1548_{tag}.csv"));
    {
        let mut f = std::fs::File::create(&path).expect("create scratch csv");
        writeln!(f, "x,y").unwrap();
        for (xi, yi) in x.iter().zip(y) {
            writeln!(f, "{xi:?},{yi:?}").unwrap();
        }
    }
    load_csvwith_inferred_schema(&path).expect("load scratch csv")
}

/// Fit `y ~ s(x, bs="ps")` (gaussian), then predict on `grid` (x values).
/// Returns (predictions, lambdas).
fn fit_and_predict(x: &[f64], y: &[f64], grid: &[f64], tag: &str) -> (Vec<f64>, Vec<f64>) {
    let ds = dataset_from(x, y, tag);
    let col = ds.column_map();
    let x_idx = col["x"];
    let n_cols = ds.headers.len();

    let cfg = FitConfig {
        family: Some("gaussian".to_string()),
        ..FitConfig::default()
    };
    let result = fit_from_formula("y ~ s(x, bs='ps')", &ds, &cfg).expect("gam p-spline fit");
    let FitResult::Standard(fit) = result else {
        panic!("expected a standard GAM fit for a gaussian p-spline smooth");
    };

    let mut grid_design = Array2::<f64>::zeros((grid.len(), n_cols));
    for (i, &g) in grid.iter().enumerate() {
        grid_design[[i, x_idx]] = g;
    }
    let design = build_term_collection_design(grid_design.view(), &fit.resolvedspec)
        .expect("rebuild design on grid");
    let pred: Vec<f64> = design.design.apply(&fit.fit.beta).to_vec();
    let lambdas: Vec<f64> = fit.fit.lambdas.to_vec();
    (pred, lambdas)
}

#[test]
fn pspline_default_is_reflection_invariant() {
    init_parallelism();

    // The issue's grid: np.linspace(-1.8, 1.8, 60).
    let n_grid = 60usize;
    let grid: Vec<f64> = (0..n_grid)
        .map(|i| -1.8 + 3.6 * (i as f64) / ((n_grid - 1) as f64))
        .collect();
    // Mirror grid is the elementwise negation (matches the issue's `-grid`):
    // p1[i] = f1(grid[i]); p2[i] = f2(-grid[i]); reflection => f2(-t) == f1(t).
    let grid_mirror: Vec<f64> = grid.iter().map(|g| -g).collect();

    let mut worst_drift = 0.0_f64;
    let mut worst_seed = 0u64;
    let mut failures = 0usize;

    for seed in 0u64..=4 {
        let (x, y) = load_seed(seed);
        let neg_x: Vec<f64> = x.iter().map(|v| -v).collect();

        let (pred_fwd, lam_fwd) = fit_and_predict(&x, &y, &grid, &format!("fwd{seed}"));
        let (pred_rev, lam_rev) = fit_and_predict(&neg_x, &y, &grid_mirror, &format!("rev{seed}"));

        let drift = pred_fwd
            .iter()
            .zip(&pred_rev)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);

        eprintln!(
            "seed {seed}: reflection drift={drift:.6}  lam_fwd={lam_fwd:?}  lam_rev={lam_rev:?}"
        );

        if drift > worst_drift {
            worst_drift = drift;
            worst_seed = seed;
        }
        if drift >= 1e-3 {
            failures += 1;
        }
    }

    eprintln!("worst drift={worst_drift:.6} at seed {worst_seed}; {failures}/5 seeds >= 1e-3");

    assert!(
        worst_drift < 1e-3,
        "p-spline default fit is NOT reflection invariant: worst max|f1(t)-f2(-t)|={worst_drift:.6} \
         at seed {worst_seed} (>= 1e-3 threshold from #1548); {failures}/5 seeds failed"
    );
}
